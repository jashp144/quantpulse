"""Unit tests for execution models, slippage, commissions, and simulated broker."""

from datetime import datetime

import pytest

from quantpulse.core.models import Bar, Order, Tick
from quantpulse.core.types import OrderSide, OrderStatus, OrderType
from quantpulse.execution.broker import SimulatedBroker
from quantpulse.execution.fees import FlatFeeModel, MakerTakerFee, TieredFeeModel
from quantpulse.execution.slippage import AlmgrenChrissSlippage, FixedBasisPointSlippage


@pytest.mark.unit
def test_fixed_basis_point_slippage():
    # 10 bps = 0.0010 = 0.1%
    model = FixedBasisPointSlippage(basis_points=10.0)

    # Buy at 100: fill = 100 + 0.1 = 100.1
    buy_fill, buy_slip = model.calculate_fill_price(
        side=OrderSide.BUY,
        current_price=100.0,
        quantity=100.0,
    )
    assert pytest.approx(buy_fill, rel=1e-5) == 100.10
    assert pytest.approx(buy_slip, rel=1e-5) == 0.10

    # Sell at 100: fill = 100 - 0.1 = 99.9
    sell_fill, sell_slip = model.calculate_fill_price(
        side=OrderSide.SELL,
        current_price=100.0,
        quantity=100.0,
    )
    assert pytest.approx(sell_fill, rel=1e-5) == 99.90
    assert pytest.approx(sell_slip, rel=1e-5) == 0.10


@pytest.mark.unit
def test_almgren_chriss_slippage():
    model = AlmgrenChrissSlippage(
        eta=0.1,
        alpha=0.5,
        half_spread_bps=2.0,  # 0.02%
        max_impact_pct=0.05,
    )

    # Trade 10,000 shares in 100,000 volume bar: participation = 0.10
    # sqrt(0.10) ~= 0.316227
    # Volume impact = 0.1 * 0.316227 = 0.0316227 (3.16%)
    # Half spread = 0.0002
    # Total impact fraction = 0.0318227
    fill_price, slip = model.calculate_fill_price(
        side=OrderSide.BUY,
        current_price=100.0,
        quantity=10_000.0,
        bar_volume=100_000.0,
    )
    assert fill_price > 100.0
    expected_slip = 100.0 * (0.1 * (0.1**0.5) + 0.0002)
    assert pytest.approx(slip, rel=1e-4) == expected_slip

    # Impact cap enforcement
    fill_capped, slip_capped = model.calculate_fill_price(
        side=OrderSide.BUY,
        current_price=100.0,
        quantity=500_000.0,
        bar_volume=10_000.0,
    )
    # Capped at 5% = $5.00 slippage
    assert pytest.approx(slip_capped, rel=1e-4) == 5.0
    assert pytest.approx(fill_capped, rel=1e-4) == 105.0


@pytest.mark.unit
def test_maker_taker_fee_model():
    model = MakerTakerFee(maker_fee_pct=0.0002, taker_fee_pct=0.0005, min_fee=1.0)

    # Taker fee on $10,000 trade: 10,000 * 0.0005 = $5.00
    fee_taker = model.calculate_fee(
        symbol="AAPL",
        side=OrderSide.BUY,
        fill_price=100.0,
        fill_quantity=100.0,
        is_maker=False,
    )
    assert pytest.approx(fee_taker, rel=1e-5) == 5.00

    # Maker fee on $10,000 trade: 10,000 * 0.0002 = $2.00
    fee_maker = model.calculate_fee(
        symbol="AAPL",
        side=OrderSide.BUY,
        fill_price=100.0,
        fill_quantity=100.0,
        is_maker=True,
    )
    assert pytest.approx(fee_maker, rel=1e-5) == 2.00

    # Min fee check on tiny $100 trade: fee would be $0.05, enforced at $1.00 min
    fee_min = model.calculate_fee(
        symbol="AAPL",
        side=OrderSide.BUY,
        fill_price=100.0,
        fill_quantity=1.0,
        is_maker=False,
    )
    assert pytest.approx(fee_min, rel=1e-5) == 1.00


@pytest.mark.unit
def test_flat_fee_model():
    model = FlatFeeModel(per_share=0.005, min_fee=1.0, max_pct=0.01)

    # 1,000 shares * $0.005 = $5.00
    fee_large = model.calculate_fee("SPY", OrderSide.BUY, 400.0, 1_000.0)
    assert pytest.approx(fee_large, rel=1e-5) == 5.00

    # 10 shares: 10 * 0.005 = $0.05 -> min fee $1.00
    fee_small = model.calculate_fee("SPY", OrderSide.BUY, 400.0, 10.0)
    assert pytest.approx(fee_small, rel=1e-5) == 1.00

    # Penny stock cap: 10,000 shares @ $0.01 = $100 notional.
    # Per-share would be $50, capped at 1% of $100 = $1.00
    fee_penny = model.calculate_fee("PENNY", OrderSide.BUY, 0.01, 10_000.0)
    assert pytest.approx(fee_penny, rel=1e-5) == 1.00


@pytest.mark.unit
def test_tiered_fee_model():
    tiers = [
        (50_000.0, 0.0010),  # 10 bps up to 50k
        (200_000.0, 0.0005),  # 5 bps up to 200k
        (float("inf"), 0.0002),  # 2 bps beyond
    ]
    model = TieredFeeModel(tiers=tiers)

    # First trade $30k: tier 1 (0.10%) = $30.00
    fee1 = model.calculate_fee("XYZ", OrderSide.BUY, 100.0, 300.0)
    assert pytest.approx(fee1, rel=1e-5) == 30.0
    assert model.cumulative_volume == 30_000.0

    # Second trade $40k: cross threshold into tier 2 (cumulative was 30k < 50k, so still tier 1)
    fee2 = model.calculate_fee("XYZ", OrderSide.BUY, 100.0, 400.0)
    assert pytest.approx(fee2, rel=1e-5) == 40.0
    assert model.cumulative_volume == 70_000.0

    # Third trade $50k: cumulative is 70k (between 50k and 200k) -> tier 2 (0.05%) = $25.00
    fee3 = model.calculate_fee("XYZ", OrderSide.BUY, 100.0, 500.0)
    assert pytest.approx(fee3, rel=1e-5) == 25.0

    model.reset_volume()
    assert model.cumulative_volume == 0.0


@pytest.mark.unit
def test_simulated_broker_market_order_fill():
    broker = SimulatedBroker(
        fee_model=MakerTakerFee(taker_fee_pct=0.001),
        slippage_model=FixedBasisPointSlippage(basis_points=10.0),  # 0.1%
    )
    # Seed price
    tick = Tick(
        symbol="AAPL",
        timestamp=datetime(2026, 1, 1, 10, 0),
        bid_price=100.0,
        ask_price=100.0,
        last_price=100.0,
    )
    broker.process_tick(tick)

    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100.0,
    )
    report = broker.submit_order(order)
    assert report is not None
    assert report.status == OrderStatus.FILLED
    # Fill price: 100.0 * 1.001 = 100.1
    assert pytest.approx(report.fill_price, rel=1e-4) == 100.10
    # Fee: 100.1 * 100 * 0.001 = 10.01
    assert pytest.approx(report.fee, rel=1e-4) == 10.01


@pytest.mark.unit
def test_simulated_broker_limit_and_stop_orders():
    broker = SimulatedBroker()
    ts = datetime(2026, 1, 1, 10, 0)

    # 1. Submit Limit Buy at 95.0 when price is 100.0
    limit_buy = Order(
        symbol="SPY",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=50.0,
        price=95.0,
    )
    assert broker.submit_order(limit_buy) is None
    assert limit_buy.id in broker.active_orders

    # Bar does not reach 95 (Low = 97.0) -> No fill
    bar1 = Bar(symbol="SPY", timestamp=ts, open=100.0, high=102.0, low=97.0, close=98.0)
    reports1 = broker.process_bar(bar1)
    assert len(reports1) == 0
    assert limit_buy.id in broker.active_orders

    # Bar penetrates 95 (Low = 94.0) -> Fills
    bar2 = Bar(symbol="SPY", timestamp=ts, open=96.0, high=97.0, low=94.0, close=95.5)
    reports2 = broker.process_bar(bar2)
    assert len(reports2) == 1
    assert reports2[0].order_id == limit_buy.id
    assert reports2[0].fill_price == 95.0
    assert limit_buy.id not in broker.active_orders

    # 2. Stop Loss Order
    stop_sell = Order(
        symbol="SPY",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_LOSS,
        quantity=50.0,
        stop_price=90.0,
    )
    broker.submit_order(stop_sell)

    # Bar touches 89.0 -> triggers stop
    bar3 = Bar(symbol="SPY", timestamp=ts, open=92.0, high=93.0, low=89.0, close=90.0)
    reports3 = broker.process_bar(bar3)
    assert len(reports3) == 1
    assert reports3[0].order_id == stop_sell.id
    assert stop_sell.status == OrderStatus.FILLED

    # 3. Order cancellation
    limit_sell = Order(
        symbol="SPY",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=10.0,
        price=120.0,
    )
    broker.submit_order(limit_sell)
    assert broker.cancel_order(limit_sell.id) is True
    assert broker.cancel_order("non-existent-id") is False
