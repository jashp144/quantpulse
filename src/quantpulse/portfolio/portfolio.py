"""Multi-asset portfolio accounting and margin management for QuantPulse."""

from datetime import datetime

from quantpulse.core.models import ExecutionReport, Order, PortfolioState, Position
from quantpulse.core.types import OrderSide
from quantpulse.portfolio.position import PositionManager


class Portfolio:
    """Institutional-grade multi-asset portfolio accounting engine."""

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        initial_capital: float | None = None,
        initial_margin_pct: float = 0.50,  # Reg-T 50% initial margin
        maintenance_margin_pct: float = 0.25,  # 25% maintenance margin
    ) -> None:
        self.initial_cash: float = initial_capital if initial_capital is not None else initial_cash
        self.cash: float = self.initial_cash
        self.initial_margin_pct: float = max(0.01, min(1.0, initial_margin_pct))
        self.maintenance_margin_pct: float = max(0.01, min(1.0, maintenance_margin_pct))
        self.position_manager = PositionManager()
        self.history: list[PortfolioState] = []
        self.current_timestamp: datetime = datetime.utcnow()

    @property
    def positions(self) -> dict[str, Position]:
        """Access all tracked positions."""
        return self.position_manager.positions

    def can_afford_order(self, order: Order, current_price: float) -> bool:
        """Check if portfolio has sufficient purchasing power / margin."""
        notional = order.quantity * current_price
        required_margin = notional * self.initial_margin_pct

        # For buy orders: check if cash exceeds required initial margin
        if order.side == OrderSide.BUY:
            pos = self.position_manager.get_position(order.symbol)
            if pos.is_short:
                # Covering a short reduces margin requirement
                return True
            return self.cash >= required_margin
        else:
            # For sell orders: selling existing long reduces risk; shorting requires margin
            pos = self.position_manager.get_position(order.symbol)
            if pos.is_long:
                return True
            return self.cash >= required_margin

    def process_fill(self, report: ExecutionReport) -> None:
        """Update portfolio state upon trade fill execution report."""
        trade_notional = report.fill_price * report.filled_quantity

        if report.side == OrderSide.BUY:
            self.cash -= trade_notional + report.fee
        else:
            self.cash += trade_notional - report.fee

        self.position_manager.add_fill(
            symbol=report.symbol,
            side=report.side,
            quantity=report.filled_quantity,
            fill_price=report.fill_price,
            timestamp=report.timestamp,
            fee=report.fee,
        )
        self.current_timestamp = report.timestamp

    def mark_to_market(
        self,
        prices: dict[str, float],
        timestamp: datetime | None = None,
    ) -> PortfolioState:
        """Mark portfolio positions to market prices and record historical snapshot."""
        if timestamp is not None:
            self.current_timestamp = timestamp

        for symbol, price in prices.items():
            self.position_manager.mark_to_market(symbol, price)

        state = self.get_state(self.current_timestamp)
        self.history.append(state)
        return state

    def get_state(self, timestamp: datetime | None = None) -> PortfolioState:
        """Calculate real-time portfolio balance, margin, and equity snapshot."""
        ts = timestamp or self.current_timestamp

        gross_market_value = 0.0
        total_unrealized_pnl = 0.0
        open_positions = 0
        net_stock_value = 0.0

        for pos in self.positions.values():
            if not pos.is_flat:
                open_positions += 1
                pos_val = pos.quantity * pos.current_price
                net_stock_value += pos_val
                gross_market_value += abs(pos_val)
                total_unrealized_pnl += pos.unrealized_pnl

        # Total Equity = Cash + Net Market Value of all positions
        equity = self.cash + net_stock_value
        initial_margin = gross_market_value * self.initial_margin_pct
        maintenance_margin = gross_market_value * self.maintenance_margin_pct
        free_cash = self.cash - initial_margin
        leverage = (gross_market_value / equity) if equity > 1e-9 else 0.0

        return PortfolioState(
            timestamp=ts,
            cash=self.cash,
            equity=equity,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            gross_market_value=gross_market_value,
            leverage=leverage,
            free_cash=free_cash,
            open_positions_count=open_positions,
            total_realized_pnl=self.position_manager.total_realized_pnl,
            total_unrealized_pnl=total_unrealized_pnl,
        )

    @property
    def equity(self) -> float:
        """Current total liquidation value of the portfolio."""
        return self.get_state().equity

    def is_margin_call(self) -> bool:
        """Check if equity has breached maintenance margin requirements."""
        state = self.get_state()
        return state.equity < state.maintenance_margin
