"""Unit tests for trading strategies in QuantPulse."""

from datetime import datetime, timedelta

import pytest

from quantpulse.core.models import Bar
from quantpulse.core.types import OrderSide, OrderType
from quantpulse.portfolio.portfolio import Portfolio
from quantpulse.strategies.base import BaseStrategy
from quantpulse.strategies.ma_crossover import MovingAverageCrossoverStrategy
from quantpulse.strategies.mean_reversion import MeanReversionStrategy
from quantpulse.strategies.volatility_breakout import VolatilityBreakoutStrategy


class DummyTestStrategy(BaseStrategy):
    """Custom concrete test strategy to verify base lifecycle and order emissions."""

    def __init__(self) -> None:
        super().__init__(name="DummyTestStrategy")
        self.started = False
        self.ended = False
        self.bar_count = 0

    def on_start(self) -> None:
        self.started = True

    def on_bar(self, bar: Bar) -> None:
        self.bar_count += 1
        if self.bar_count == 1:
            self.buy(bar.symbol, 50.0, OrderType.LIMIT, price=100.0)

    def on_end(self) -> None:
        self.ended = True


@pytest.mark.unit
def test_base_strategy_lifecycle_and_orders():
    strat = DummyTestStrategy()
    port = Portfolio(initial_capital=100_000.0)
    strat.set_portfolio(port)

    strat.on_start()
    assert strat.started is True

    bar = Bar(
        symbol="TEST",
        timestamp=datetime(2026, 1, 1),
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
    )
    strat.on_bar(bar)
    assert strat.bar_count == 1

    orders = strat.pop_orders()
    assert len(orders) == 1
    assert orders[0].symbol == "TEST"
    assert orders[0].quantity == 50.0
    assert orders[0].order_type == OrderType.LIMIT
    assert orders[0].price == 100.0

    # Pop again should be empty
    assert len(strat.pop_orders()) == 0

    strat.on_end()
    assert strat.ended is True


@pytest.mark.unit
def test_moving_average_crossover_strategy():
    strat = MovingAverageCrossoverStrategy(fast_period=3, slow_period=6, trade_size=100.0)
    port = Portfolio(initial_capital=100_000.0)
    strat.set_portfolio(port)

    start_date = datetime(2026, 1, 1)

    # 1. Warm-up bars with flat prices
    for i in range(7):
        bar = Bar(
            symbol="TREND",
            timestamp=start_date + timedelta(days=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
        )
        strat.on_bar(bar)

    # 2. Strong upward move to trigger Golden Cross
    for i in range(7, 12):
        price = 100.0 + (i - 6) * 5.0  # 105, 110, 115, 120, 125
        bar = Bar(
            symbol="TREND",
            timestamp=start_date + timedelta(days=i),
            open=price - 1.0,
            high=price + 2.0,
            low=price - 2.0,
            close=price,
        )
        strat.on_bar(bar)

    orders = strat.pop_orders()
    assert len(orders) >= 1
    assert orders[0].side == OrderSide.BUY
    assert orders[0].quantity == 100.0


@pytest.mark.unit
def test_mean_reversion_strategy():
    strat = MeanReversionStrategy(
        window=10,
        z_threshold=1.5,
        rsi_period=6,
        oversold_rsi=40.0,
        trade_size=50.0,
        allow_short=True,
    )
    port = Portfolio(initial_capital=100_000.0)
    strat.set_portfolio(port)

    start_date = datetime(2026, 1, 1)

    # Establish baseline mean around 100.0
    for i in range(15):
        bar = Bar(
            symbol="MR",
            timestamp=start_date + timedelta(days=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0 + (i % 2) * 0.5,
        )
        strat.on_bar(bar)

    # Sharp sudden plunge to trigger oversold buy signal
    plunge_bar = Bar(
        symbol="MR",
        timestamp=start_date + timedelta(days=15),
        open=98.0,
        high=98.0,
        low=85.0,
        close=86.0,
    )
    strat.on_bar(plunge_bar)

    orders = strat.pop_orders()
    assert len(orders) == 1
    assert orders[0].side == OrderSide.BUY
    assert orders[0].quantity == 50.0


@pytest.mark.unit
def test_volatility_breakout_strategy():
    strat = VolatilityBreakoutStrategy(
        channel_period=10,
        atr_period=5,
        stop_atr_mult=2.0,
        trade_size=100.0,
    )
    port = Portfolio(initial_capital=100_000.0)
    strat.set_portfolio(port)

    start_date = datetime(2026, 1, 1)

    # Establish channel between 95 and 105
    for i in range(12):
        bar = Bar(
            symbol="VIX",
            timestamp=start_date + timedelta(days=i),
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
        )
        strat.on_bar(bar)

    # Breakout bar closing above 105 (e.g. 110)
    breakout_bar = Bar(
        symbol="VIX",
        timestamp=start_date + timedelta(days=13),
        open=106.0,
        high=112.0,
        low=106.0,
        close=110.0,
    )
    strat.on_bar(breakout_bar)

    orders = strat.pop_orders()
    assert len(orders) == 1
    assert orders[0].side == OrderSide.BUY
    assert orders[0].quantity == 100.0
    assert "VIX" in strat.trailing_stops
