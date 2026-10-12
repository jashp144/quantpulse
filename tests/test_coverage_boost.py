"""Supplementary unit tests targeting 100% coverage across event queue, tear sheets, and edge cases."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from quantpulse.core.models import Bar, ExecutionReport, Order, Tick
from quantpulse.core.types import OrderSide, OrderType
from quantpulse.engine.event_bus import EventQueue
from quantpulse.engine.events import MarketEvent
from quantpulse.engine.tear_sheet import TearSheetGenerator
from quantpulse.execution.broker import SimulatedBroker
from quantpulse.execution.slippage import AlmgrenChrissSlippage
from quantpulse.portfolio.portfolio import Portfolio
from quantpulse.portfolio.position import PositionManager
from quantpulse.risk.metrics import RiskMetrics
from quantpulse.risk.var import ValueAtRisk
from quantpulse.strategies.base import BaseStrategy
from quantpulse.strategies.ma_crossover import MovingAverageCrossoverStrategy


@pytest.mark.unit
def test_event_queue_operations():
    eq = EventQueue()
    assert eq.empty() is True
    assert eq.qsize() == 0
    assert eq.get() is None

    now = datetime(2026, 1, 1)
    ev = MarketEvent(symbol="AAPL", timestamp=now)
    eq.put(ev)

    assert eq.empty() is False
    assert eq.qsize() == 1
    popped = eq.get()
    assert popped == ev
    assert eq.empty() is True

    eq.put(ev)
    eq.clear()
    assert eq.empty() is True


@pytest.mark.unit
def test_tear_sheet_generator_direct():
    metrics = {
        "initial_capital": 100_000.0,
        "final_equity": 125_000.0,
        "total_return": 0.25,
        "cagr": 0.25,
        "annualized_volatility": 0.15,
        "sharpe_ratio": 1.6667,
        "sortino_ratio": 2.15,
        "calmar_ratio": 3.0,
        "max_drawdown": -0.08,
        "max_drawdown_duration": 12,
        "total_trades": 45,
        "win_rate": 0.62,
        "profit_factor": 2.10,
        "var_95": 0.015,
        "cvar_95": 0.022,
        "var_99": 0.028,
        "cvar_99": 0.035,
        "beta": 1.05,
        "alpha": 0.045,
        "information_ratio": 0.85,
    }

    md = TearSheetGenerator.generate_markdown(metrics)
    assert "# 📈 QuantPulse Performance Tear-Sheet" in md
    assert "Benchmark Comparison" in md
    assert "Information Ratio" in md

    ascii_rep = TearSheetGenerator.generate_ascii(metrics)
    assert "QUANTPULSE PERFORMANCE TEAR-SHEET" in ascii_rep
    assert "Sharpe Ratio" in ascii_rep


@pytest.mark.unit
def test_broker_tick_matching_for_limit_and_stop():
    broker = SimulatedBroker()
    now = datetime(2026, 1, 1, 10, 0)

    # 1. Limit Buy Order @ 100.0
    limit_buy = Order(
        symbol="IBM",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=50.0,
        price=100.0,
    )
    broker.submit_order(limit_buy)

    # Tick ask price 102.0 -> no fill
    tick1 = Tick(symbol="IBM", timestamp=now, bid_price=101.0, ask_price=102.0)
    reports1 = broker.process_tick(tick1)
    assert len(reports1) == 0

    # Tick ask price 99.50 -> fills limit buy
    tick2 = Tick(symbol="IBM", timestamp=now, bid_price=99.0, ask_price=99.50)
    reports2 = broker.process_tick(tick2)
    assert len(reports2) == 1
    assert reports2[0].order_id == limit_buy.id
    assert limit_buy.id not in broker.active_orders

    # 2. Limit Sell Order @ 110.0
    limit_sell = Order(
        symbol="IBM",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=50.0,
        price=110.0,
    )
    broker.submit_order(limit_sell)

    # Tick bid price 111.0 -> fills limit sell
    tick3 = Tick(symbol="IBM", timestamp=now, bid_price=111.0, ask_price=112.0)
    reports3 = broker.process_tick(tick3)
    assert len(reports3) == 1
    assert reports3[0].order_id == limit_sell.id

    # 3. Stop Buy Order @ 120.0
    stop_buy = Order(
        symbol="IBM",
        side=OrderSide.BUY,
        order_type=OrderType.STOP_LOSS,
        quantity=25.0,
        stop_price=120.0,
    )
    broker.submit_order(stop_buy)

    tick4 = Tick(symbol="IBM", timestamp=now, bid_price=121.0, ask_price=122.0)
    reports4 = broker.process_tick(tick4)
    assert len(reports4) == 1
    assert reports4[0].order_id == stop_buy.id

    # 4. Stop Sell Order @ 90.0
    stop_sell = Order(
        symbol="IBM",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_LOSS,
        quantity=25.0,
        stop_price=90.0,
    )
    broker.submit_order(stop_sell)

    tick5 = Tick(symbol="IBM", timestamp=now, bid_price=89.0, ask_price=90.0)
    reports5 = broker.process_tick(tick5)
    assert len(reports5) == 1
    assert reports5[0].order_id == stop_sell.id


@pytest.mark.unit
def test_position_manager_multiple_short_lots_and_flip():
    pm = PositionManager()
    t1 = datetime(2026, 1, 1, 10, 0)
    t2 = datetime(2026, 1, 1, 11, 0)
    t3 = datetime(2026, 1, 1, 12, 0)

    # Short Lot 1: 50 @ 100
    pm.add_fill("SHORT_STOCK", OrderSide.SELL, 50.0, 100.0, t1)
    # Short Lot 2: 50 @ 90
    pm.add_fill("SHORT_STOCK", OrderSide.SELL, 50.0, 90.0, t2)

    pos = pm.get_position("SHORT_STOCK")
    assert pos.quantity == -100.0
    assert pos.avg_price == 95.0

    # Cover 120 @ 80: Covers all 100 short lots (profit) and flips to Long 20 @ 80
    # Lot 1: (100 - 80) * 50 = 1000
    # Lot 2: (90 - 80) * 50 = 500
    # Total PnL = 1500
    pnl, pos = pm.add_fill("SHORT_STOCK", OrderSide.BUY, 120.0, 80.0, t3)
    assert pytest.approx(pnl, rel=1e-5) == 1500.0
    assert pos.is_long is True
    assert pos.quantity == 20.0
    assert pos.avg_price == 80.0


@pytest.mark.unit
def test_base_strategy_close_short_position():
    strat = DummyShortStrategy()
    port = Portfolio(initial_capital=50_000.0)
    strat.set_portfolio(port)

    # Create short position in portfolio
    report = ExecutionReport(
        order_id="ord-short",
        symbol="XYZ",
        side=OrderSide.SELL,
        filled_quantity=100.0,
        fill_price=150.0,
        timestamp=datetime(2026, 1, 1),
    )
    port.process_fill(report)
    assert port.positions["XYZ"].is_short is True

    # Call close_position: should emit BUY order
    order = strat.close_position("XYZ")
    assert order is not None
    assert order.side == OrderSide.BUY
    assert order.quantity == 100.0


class DummyShortStrategy(BaseStrategy):
    def on_bar(self, bar: Bar) -> None:
        pass


@pytest.mark.unit
def test_moving_average_crossover_sma_mode():
    strat = MovingAverageCrossoverStrategy(fast_period=3, slow_period=5, use_ema=False)
    port = Portfolio(initial_capital=100_000.0)
    strat.set_portfolio(port)

    start = datetime(2026, 1, 1)
    for i in range(10):
        bar = Bar(
            symbol="SMA_TEST",
            timestamp=start + timedelta(days=i),
            open=100.0 + i,
            high=102.0 + i,
            low=99.0 + i,
            close=100.0 + i,
        )
        strat.on_bar(bar)


@pytest.mark.unit
def test_almgren_chriss_zero_volume_fallback():
    slip_model = AlmgrenChrissSlippage(eta=0.1, alpha=0.5)
    fill, slip = slip_model.calculate_fill_price(
        side=OrderSide.BUY,
        current_price=100.0,
        quantity=50.0,
        bar_volume=0.0,
    )
    assert fill > 100.0
    assert slip > 0.0


@pytest.mark.unit
def test_risk_metrics_edge_cases():
    # Negative/declining returns for Sortino
    neg_returns = np.array([-0.01, -0.02, -0.015, -0.005])
    sortino = RiskMetrics.sortino_ratio(neg_returns)
    assert sortino < 0.0

    # Drawdown series for empty
    hwm, dd = RiskMetrics.drawdown_series([])
    assert len(hwm) == 0

    # Information ratio tracking error zero
    same_returns = np.array([0.01, 0.02, 0.03])
    ir = RiskMetrics.information_ratio(same_returns, same_returns)
    assert ir == 0.0

    # VaR with empty/short array
    assert ValueAtRisk.parametric_var([0.01]) == 0.0
    assert ValueAtRisk.historical_var([0.01]) == 0.0
