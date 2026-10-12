"""Unit tests for portfolio accounting, FIFO lot matching, margin, and PnL."""

from datetime import datetime

import pytest

from quantpulse.core.models import ExecutionReport, Order
from quantpulse.core.types import OrderSide, OrderType
from quantpulse.portfolio.portfolio import Portfolio
from quantpulse.portfolio.position import PositionManager


@pytest.mark.unit
def test_fifo_long_lot_matching():
    pm = PositionManager()
    t1 = datetime(2026, 1, 1, 10, 0)
    t2 = datetime(2026, 1, 1, 11, 0)
    t3 = datetime(2026, 1, 1, 12, 0)

    # Buy Lot 1: 100 @ $100
    pnl1, pos = pm.add_fill("AAPL", OrderSide.BUY, 100.0, 100.0, t1, fee=0.0)
    assert pnl1 == 0.0
    assert pos.quantity == 100.0
    assert pos.avg_price == 100.0
    assert len(pos.lots) == 1

    # Buy Lot 2: 100 @ $120
    pnl2, pos = pm.add_fill("AAPL", OrderSide.BUY, 100.0, 120.0, t2, fee=0.0)
    assert pnl2 == 0.0
    assert pos.quantity == 200.0
    assert pos.avg_price == 110.0  # (100*100 + 100*120)/200
    assert len(pos.lots) == 2

    # Sell 150 @ $130:
    # First 100 matched against Lot 1: (130 - 100) * 100 = $3,000
    # Next 50 matched against Lot 2:   (130 - 120) * 50  = $500
    # Total Realized PnL = $3,500
    pnl3, pos = pm.add_fill("AAPL", OrderSide.SELL, 150.0, 130.0, t3, fee=10.0)
    assert pytest.approx(pnl3, rel=1e-5) == 3_490.0  # 3500 - 10 fee
    assert pos.quantity == 50.0
    assert pos.avg_price == 120.0  # Remaining 50 shares belong to Lot 2 ($120)
    assert len(pos.lots) == 1
    assert pos.lots[0].quantity == 50.0
    assert pos.lots[0].entry_price == 120.0


@pytest.mark.unit
def test_fifo_short_lot_matching():
    pm = PositionManager()
    t1 = datetime(2026, 1, 1, 10, 0)
    t2 = datetime(2026, 1, 1, 11, 0)

    # Short 100 @ $150
    pnl1, pos = pm.add_fill("TSLA", OrderSide.SELL, 100.0, 150.0, t1, fee=5.0)
    assert pnl1 == -5.0
    assert pos.quantity == -100.0
    assert pos.is_short is True

    # Buy to Cover 100 @ $130 (profitable short): (150 - 130) * 100 = $2,000
    pnl2, pos = pm.add_fill("TSLA", OrderSide.BUY, 100.0, 130.0, t2, fee=5.0)
    assert pytest.approx(pnl2, rel=1e-5) == 1_995.0  # 2000 - 5 fee
    assert pos.is_flat is True
    assert pos.quantity == 0.0
    assert len(pos.lots) == 0


@pytest.mark.unit
def test_position_flip_long_to_short():
    pm = PositionManager()
    t1 = datetime(2026, 1, 1, 10, 0)
    t2 = datetime(2026, 1, 1, 11, 0)

    # Long 50 @ $100
    pm.add_fill("NVDA", OrderSide.BUY, 50.0, 100.0, t1)

    # Sell 80 @ $110: Closes 50 long (PnL = +$500), leaves 30 short @ $110
    pnl, pos = pm.add_fill("NVDA", OrderSide.SELL, 80.0, 110.0, t2)
    assert pytest.approx(pnl, rel=1e-5) == 500.0
    assert pos.is_short is True
    assert pos.quantity == -30.0
    assert pos.avg_price == 110.0
    assert len(pos.lots) == 1
    assert pos.lots[0].side == OrderSide.SELL


@pytest.mark.unit
def test_deterministic_trade_accounting():
    """Verify Survey 2 deterministic accounting specification.

    100 shares bought at $100, sold at $110 with 0.1% fees;
    Verify final equity is exactly $100,000 + $1,000 - $21 = $100,979.
    """
    portfolio = Portfolio(initial_capital=100_000.0)
    t1 = datetime(2026, 1, 1, 10, 0)
    t2 = datetime(2026, 1, 2, 10, 0)

    # 1. Buy 100 shares @ $100.0 with 0.1% fee ($10.0)
    buy_report = ExecutionReport(
        order_id="b1",
        symbol="ABC",
        side=OrderSide.BUY,
        filled_quantity=100.0,
        fill_price=100.0,
        fee=10.0,
        timestamp=t1,
    )
    portfolio.process_fill(buy_report)

    # Cash should be $100,000 - (100 * 100) - $10 = $89,990
    assert portfolio.cash == 89_990.0

    # Mark to market at entry price
    state1 = portfolio.mark_to_market({"ABC": 100.0}, timestamp=t1)
    # Equity = Cash ($89,990) + Stock ($10,000) = $99,990 ($10 fee accounted for)
    assert state1.equity == 99_990.0

    # 2. Sell 100 shares @ $110.0 with 0.1% fee ($11.0)
    sell_report = ExecutionReport(
        order_id="s1",
        symbol="ABC",
        side=OrderSide.SELL,
        filled_quantity=100.0,
        fill_price=110.0,
        fee=11.0,
        timestamp=t2,
    )
    portfolio.process_fill(sell_report)

    state2 = portfolio.mark_to_market({"ABC": 110.0}, timestamp=t2)
    # Cash = $89,990 + (100 * 110) - $11 = $100,979.0
    assert portfolio.cash == 100_979.0
    # Equity = Cash = $100,979.0
    assert state2.equity == 100_979.0
    assert state2.total_realized_pnl == 979.0  # 1,000 gain - 21 fees


@pytest.mark.unit
def test_margin_and_purchasing_power():
    portfolio = Portfolio(
        initial_capital=10_000.0,
        initial_margin_pct=0.50,
        maintenance_margin_pct=0.25,
    )
    # Order for 100 shares @ $100 = $10,000 notional. Required initial margin = $5,000.
    order1 = Order(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0)
    assert portfolio.can_afford_order(order1, current_price=100.0) is True

    # Order for 300 shares @ $100 = $30,000 notional. Required initial margin = $15,000 > $10,000.
    order2 = Order(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=300.0)
    assert portfolio.can_afford_order(order2, current_price=100.0) is False

    # Execute order1 with 100% margin utilization
    fill = ExecutionReport(
        order_id=order1.id,
        symbol="SPY",
        side=OrderSide.BUY,
        filled_quantity=100.0,
        fill_price=100.0,
        fee=0.0,
        timestamp=datetime(2026, 1, 1),
    )
    portfolio.process_fill(fill)
    assert portfolio.cash == 0.0

    # If SPY drops from $100 to $20:
    # Stock value = 100 * $20 = $2,000.
    # Equity = Cash ($0) + Stock ($2,000) = $2,000.
    # Maintenance margin = 25% of $2,000 = $500. Not a margin call yet.
    portfolio.mark_to_market({"SPY": 20.0})
    assert portfolio.is_margin_call() is False

    # What if price drops to $0.10?
    # Equity = $10. Maintenance margin = 25% of $10 = $2.50.
    portfolio.mark_to_market({"SPY": 0.10})
    assert portfolio.equity > 0
