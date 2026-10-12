"""Unit tests for QuantPulse core models, enums, and state transitions."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from quantpulse.core.models import (
    Bar,
    ExecutionReport,
    Order,
    PortfolioState,
    Position,
    PositionLot,
    Tick,
)
from quantpulse.core.types import (
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


@pytest.mark.unit
def test_tick_properties():
    tick = Tick(
        symbol="AAPL",
        timestamp=datetime(2026, 1, 1, 10, 0),
        bid_price=150.10,
        ask_price=150.30,
        bid_size=100.0,
        ask_size=200.0,
        last_price=150.20,
    )
    assert tick.symbol == "AAPL"
    assert pytest.approx(tick.mid_price, rel=1e-5) == 150.20
    assert pytest.approx(tick.spread, rel=1e-5) == 0.20

    # Negative price validation failure
    with pytest.raises(ValidationError):
        Tick(
            symbol="AAPL",
            timestamp=datetime(2026, 1, 1),
            bid_price=-10.0,
            ask_price=150.0,
        )


@pytest.mark.unit
def test_bar_properties():
    bar = Bar(
        symbol="SPY",
        timestamp=datetime(2026, 1, 1),
        open=450.0,
        high=455.0,
        low=448.0,
        close=453.0,
        volume=1_000_000.0,
    )
    assert pytest.approx(bar.typical_price, rel=1e-5) == (455.0 + 448.0 + 453.0) / 3.0
    assert pytest.approx(bar.range, rel=1e-5) == 7.0

    # Non-positive price validation
    with pytest.raises(ValidationError):
        Bar(
            symbol="SPY",
            timestamp=datetime(2026, 1, 1),
            open=0.0,
            high=455.0,
            low=448.0,
            close=453.0,
        )


@pytest.mark.unit
def test_order_lifecycle_and_state_transitions():
    order = Order(
        symbol="NVDA",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=50.0,
        price=120.0,
        time_in_force=TimeInForce.GTC,
    )
    assert order.status == OrderStatus.PENDING
    assert order.is_active is True
    assert order.remaining_quantity == 50.0

    # Partial fill update
    order.status = OrderStatus.PARTIALLY_FILLED
    order.filled_quantity = 20.0
    order.avg_fill_price = 120.0
    assert order.is_active is True
    assert order.remaining_quantity == 30.0

    # Complete fill
    order.filled_quantity = 50.0
    order.status = OrderStatus.FILLED
    assert order.is_active is False
    assert order.remaining_quantity == 0.0


@pytest.mark.unit
def test_position_attributes():
    pos = Position(symbol="TSLA", quantity=0.0, avg_price=0.0, current_price=200.0)
    assert pos.is_flat is True
    assert pos.is_long is False
    assert pos.is_short is False
    assert pos.market_value == 0.0
    assert pos.unrealized_pnl_pct == 0.0

    # Long position
    pos.quantity = 100.0
    pos.avg_price = 180.0
    pos.current_price = 200.0
    assert pos.is_long is True
    assert pos.is_flat is False
    assert pytest.approx(pos.market_value) == 20_000.0
    assert pytest.approx(pos.unrealized_pnl_pct, rel=1e-4) == (200.0 - 180.0) / 180.0

    # Short position
    pos.quantity = -50.0
    pos.avg_price = 220.0
    pos.current_price = 200.0
    assert pos.is_short is True
    assert pytest.approx(pos.market_value) == 10_000.0
    assert pytest.approx(pos.unrealized_pnl_pct, rel=1e-4) == (220.0 - 200.0) / 220.0


@pytest.mark.unit
def test_execution_report_and_position_lot():
    now = datetime(2026, 1, 1, 12, 0)
    report = ExecutionReport(
        order_id="ord-1",
        symbol="MSFT",
        side=OrderSide.BUY,
        filled_quantity=100.0,
        fill_price=400.0,
        fee=2.0,
        slippage=0.05,
        timestamp=now,
    )
    assert report.order_id == "ord-1"
    assert report.status == OrderStatus.FILLED
    assert report.fee == 2.0

    lot = PositionLot(
        symbol="MSFT",
        side=OrderSide.BUY,
        quantity=100.0,
        entry_price=400.0,
        entry_time=now,
    )
    assert lot.quantity == 100.0


@pytest.mark.unit
def test_portfolio_state_model():
    now = datetime(2026, 1, 1, 16, 0)
    state = PortfolioState(
        timestamp=now,
        cash=80_000.0,
        equity=105_000.0,
        initial_margin=12_500.0,
        maintenance_margin=6_250.0,
        gross_market_value=25_000.0,
        leverage=0.238,
        free_cash=67_500.0,
        open_positions_count=2,
        total_realized_pnl=1_000.0,
        total_unrealized_pnl=4_000.0,
    )
    assert state.cash == 80_000.0
    assert state.equity == 105_000.0
    assert state.open_positions_count == 2
