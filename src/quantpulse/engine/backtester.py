"""Event-driven backtesting engine orchestrator for QuantPulse."""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from quantpulse.core.models import Bar, Order
from quantpulse.core.types import OrderSide, OrderType
from quantpulse.engine.event_bus import EventQueue
from quantpulse.engine.events import MarketEvent
from quantpulse.engine.results import BacktestResult
from quantpulse.execution.broker import SimulatedBroker
from quantpulse.execution.fees import BaseFeeModel, MakerTakerFee
from quantpulse.execution.slippage import BaseSlippageModel, FixedBasisPointSlippage
from quantpulse.portfolio.portfolio import Portfolio
from quantpulse.risk.metrics import RiskMetrics
from quantpulse.risk.var import ValueAtRisk
from quantpulse.strategies.base import BaseStrategy


class _DefaultBuyAndHoldStrategy(BaseStrategy):
    """Fallback strategy that buys on first available bar and holds."""

    def __init__(self, quantity: float = 100.0) -> None:
        super().__init__(name="DefaultBuyAndHold")
        self.quantity = quantity
        self.has_bought: dict[str, bool] = {}

    def on_bar(self, bar: Bar) -> None:
        if not self.has_bought.get(bar.symbol, False):
            self.buy(symbol=bar.symbol, quantity=self.quantity, order_type=OrderType.MARKET)
            self.has_bought[bar.symbol] = True


class BacktestEngine:
    """Institutional-grade event-driven backtesting orchestrator."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        initial_cash: float | None = None,
        fee_model: BaseFeeModel | None = None,
        slippage_model: BaseSlippageModel | None = None,
        risk_free_rate: float = 0.0,
        benchmark_returns: pd.Series | None = None,
        strategy: BaseStrategy | None = None,
    ) -> None:
        cap = initial_cash if initial_cash is not None else initial_capital
        self.initial_capital = cap
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns

        self.fee_model: BaseFeeModel = fee_model or MakerTakerFee()
        self.slippage_model: BaseSlippageModel = slippage_model or FixedBasisPointSlippage()

        self.broker = SimulatedBroker(
            fee_model=self.fee_model,
            slippage_model=self.slippage_model,
        )
        self.portfolio = Portfolio(initial_capital=self.initial_capital)
        self.strategy: BaseStrategy | None = strategy
        self.event_queue = EventQueue()
        self.symbol_data: dict[str, list[Bar]] = {}

    def set_strategy(self, strategy: BaseStrategy) -> None:
        """Attach a trading strategy to the engine."""
        self.strategy = strategy
        self.strategy.set_portfolio(self.portfolio)

    def add_data(self, symbol: str, data: pd.DataFrame | list[Bar]) -> None:
        """Register historical OHLCV data for an asset."""
        if isinstance(data, pd.DataFrame):
            bars = self._dataframe_to_bars(symbol, data)
        else:
            bars = sorted(data, key=lambda b: b.timestamp)
        self.symbol_data[symbol] = bars

    def run(
        self,
        data: pd.DataFrame | None = None,
        price_data: pd.DataFrame | None = None,
        strategy: BaseStrategy | None = None,
    ) -> BacktestResult:
        """Execute the chronological event-driven backtest simulation.

        Args:
            data: Optional primary price dataframe.
            price_data: Alias for primary price dataframe.
            strategy: Optional strategy to execute (overrides set_strategy).

        Returns:
            BacktestResult: Complete tear-sheet metrics and equity curve.
        """
        # Support dataframe passed directly into run()
        input_df = data if data is not None else price_data
        if input_df is not None:
            self.add_data("ASSET", input_df)

        if strategy is not None:
            self.set_strategy(strategy)
        elif self.strategy is None:
            self.set_strategy(_DefaultBuyAndHoldStrategy())
        else:
            self.strategy.set_portfolio(self.portfolio)

        if not self.symbol_data:
            raise ValueError("No price data has been added to the BacktestEngine.")

        # Aggregate and sort all bars chronologically to guarantee no lookahead bias
        all_bars: list[Bar] = []
        for bars in self.symbol_data.values():
            all_bars.extend(bars)
        all_bars.sort(key=lambda b: (b.timestamp, b.symbol))

        # Lifecycle start
        self.strategy.on_start()

        # Group bars by timestamp to update cross-asset state simultaneously
        timestamp_groups: dict[datetime, list[Bar]] = {}
        for bar in all_bars:
            if bar.timestamp not in timestamp_groups:
                timestamp_groups[bar.timestamp] = []
            timestamp_groups[bar.timestamp].append(bar)

        sorted_timestamps = sorted(timestamp_groups.keys())

        # Main Event Loop
        for ts in sorted_timestamps:
            current_bars = timestamp_groups[ts]

            # 1. Market Price Ingestion & Matching Pending Resting Orders
            for bar in current_bars:
                market_event = MarketEvent(symbol=bar.symbol, timestamp=ts, bar=bar)
                self.event_queue.put(market_event)
                # Broker checks resting limit/stop orders against OHLC bar
                pending_reports = self.broker.process_bar(bar)
                for report in pending_reports:
                    self.portfolio.process_fill(report)
                    self.strategy.on_order_status(report)

            # 2. Strategy Event Handling
            for bar in current_bars:
                self.strategy.on_bar(bar)

            # 3. Process Newly Emitted Orders from Strategy
            new_orders: list[Order] = self.strategy.pop_orders()
            for order in new_orders:
                order.created_at = ts
                # Check purchasing power / margin
                current_price = self.broker.current_prices.get(order.symbol, order.price or 100.0)
                if self.portfolio.can_afford_order(order, current_price):
                    report = self.broker.submit_order(order)
                    if report is not None:
                        self.portfolio.process_fill(report)
                        self.strategy.on_order_status(report)

            # 4. End-of-Bar Mark-to-Market Valuation
            current_prices = {bar.symbol: bar.close for bar in current_bars}
            self.portfolio.mark_to_market(current_prices, timestamp=ts)

        # Lifecycle end
        self.strategy.on_end()

        return self._synthesize_results(sorted_timestamps)

    def _dataframe_to_bars(self, symbol: str, df: pd.DataFrame) -> list[Bar]:
        """Convert a pandas DataFrame into validated Bar domain models."""
        df_clean = df.copy()
        # Normalize column names to lowercase
        df_clean.columns = [str(c).lower().strip() for c in df_clean.columns]

        # Ensure required OHLC columns exist
        for col in ["open", "high", "low", "close"]:
            if col not in df_clean.columns:
                raise ValueError(f"DataFrame missing required column: '{col}'")

        if "volume" not in df_clean.columns:
            df_clean["volume"] = 0.0

        bars: list[Bar] = []
        is_datetime_index = isinstance(df_clean.index, pd.DatetimeIndex)

        start_time = datetime(2026, 1, 1, 9, 30)
        for idx, row in df_clean.iterrows():
            if is_datetime_index:
                ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            elif "date" in df_clean.columns:
                ts = pd.to_datetime(row["date"]).to_pydatetime()
            elif "timestamp" in df_clean.columns:
                ts = pd.to_datetime(row["timestamp"]).to_pydatetime()
            else:
                ts = start_time + timedelta(days=len(bars))

            bar = Bar(
                symbol=symbol,
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                vwap=float(row["vwap"]) if "vwap" in row and not pd.isna(row["vwap"]) else None,
            )
            bars.append(bar)

        bars.sort(key=lambda b: b.timestamp)
        return bars

    def _synthesize_results(self, timestamps: list[datetime]) -> BacktestResult:
        """Compute performance statistics, risk ratios, and tear-sheet data."""
        history = self.portfolio.history
        if not history:
            return BacktestResult(
                initial_capital=self.initial_capital,
                final_equity=self.initial_capital,
                total_return=0.0,
                cagr=0.0,
                annualized_volatility=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_duration=0,
                total_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                var_95=0.0,
                cvar_95=0.0,
            )

        equity_list = [state.equity for state in history]
        ts_list = [state.timestamp for state in history]
        equity_series = pd.Series(equity_list, index=ts_list)

        returns_series = equity_series.pct_change().fillna(0.0)
        returns_arr = returns_series.values

        final_equity = equity_list[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        cagr_val = RiskMetrics.cagr(equity_series)
        vol_ann = RiskMetrics.annualized_volatility(returns_arr)
        sharpe = RiskMetrics.sharpe_ratio(returns_arr, risk_free_rate=self.risk_free_rate)
        sortino = RiskMetrics.sortino_ratio(returns_arr, risk_free_rate=self.risk_free_rate)
        mdd = RiskMetrics.max_drawdown(equity_series)
        mdd_duration = RiskMetrics.max_drawdown_duration(equity_series)
        calmar = RiskMetrics.calmar_ratio(equity_series)

        _, dd_arr = RiskMetrics.drawdown_series(equity_series)
        drawdown_series = pd.Series(dd_arr, index=ts_list)

        # VaR and CVaR calculations
        var_95 = ValueAtRisk.parametric_var(returns_arr, confidence=0.95)
        cvar_95 = ValueAtRisk.parametric_cvar(returns_arr, confidence=0.95)
        var_99 = ValueAtRisk.parametric_var(returns_arr, confidence=0.99)
        cvar_99 = ValueAtRisk.parametric_cvar(returns_arr, confidence=0.99)

        # Trade analytics
        closed_pnls: list[float] = []
        trades_records: list[dict[str, Any]] = []
        for report in self.broker.execution_history:
            trades_records.append(
                {
                    "order_id": report.order_id,
                    "symbol": report.symbol,
                    "side": report.side.value,
                    "quantity": report.filled_quantity,
                    "price": report.fill_price,
                    "fee": report.fee,
                    "slippage": report.slippage,
                    "timestamp": report.timestamp,
                }
            )
            if report.side == OrderSide.SELL:
                # Approximate trade PnL from position manager realized gains
                closed_pnls.append(report.fill_price * report.filled_quantity - report.fee)

        win_rate_val = RiskMetrics.win_rate(closed_pnls)
        profit_factor_val = RiskMetrics.profit_factor(closed_pnls)

        # Benchmark analytics if provided
        alpha_val: float | None = None
        beta_val: float | None = None
        ir_val: float | None = None

        if self.benchmark_returns is not None and len(self.benchmark_returns) == len(returns_arr):
            b_arr = self.benchmark_returns.values
            beta_val = RiskMetrics.beta(returns_arr, b_arr)
            alpha_val = RiskMetrics.alpha(returns_arr, b_arr, risk_free_rate=self.risk_free_rate)
            ir_val = RiskMetrics.information_ratio(returns_arr, b_arr)

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            cagr=cagr_val,
            annualized_volatility=vol_ann,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=mdd,
            max_drawdown_duration=mdd_duration,
            total_trades=len(self.broker.execution_history),
            win_rate=win_rate_val,
            profit_factor=profit_factor_val,
            var_95=var_95,
            cvar_95=cvar_95,
            var_99=var_99,
            cvar_99=cvar_99,
            alpha=alpha_val,
            beta=beta_val,
            information_ratio=ir_val,
            equity_curve=equity_series,
            drawdown_curve=drawdown_series,
            returns=returns_series,
            trades=trades_records,
        )
