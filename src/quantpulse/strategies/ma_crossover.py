"""Dual Moving Average (EMA/SMA) Crossover Strategy for QuantPulse."""

from collections import deque

from quantpulse.core.models import Bar
from quantpulse.strategies.base import BaseStrategy


class MovingAverageCrossoverStrategy(BaseStrategy):
    """Trend-following strategy utilizing fast and slow exponential moving average crossovers."""

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        trade_size: float = 100.0,
        use_ema: bool = True,
    ) -> None:
        super().__init__(name="MovingAverageCrossoverStrategy")
        self.fast_period = max(2, fast_period)
        self.slow_period = max(self.fast_period + 1, slow_period)
        self.trade_size = trade_size
        self.use_ema = use_ema

        self.fast_alpha = 2.0 / (self.fast_period + 1)
        self.slow_alpha = 2.0 / (self.slow_period + 1)

        self.fast_ma: dict[str, float] = {}
        self.slow_ma: dict[str, float] = {}
        self.prev_diff: dict[str, float] = {}
        self.price_history: dict[str, deque[float]] = {}

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol not in self.price_history:
            self.price_history[bar.symbol] = deque(maxlen=self.slow_period)

        history = self.price_history[bar.symbol]
        history.append(bar.close)

        if self.use_ema:
            if bar.symbol not in self.fast_ma:
                if len(history) < self.slow_period:
                    return
                # Initialize EMAs with SMA of initial window
                prices = list(history)
                self.fast_ma[bar.symbol] = sum(prices[-self.fast_period :]) / self.fast_period
                self.slow_ma[bar.symbol] = sum(prices) / self.slow_period
                self.prev_diff[bar.symbol] = self.fast_ma[bar.symbol] - self.slow_ma[bar.symbol]
                return

            # Update EMAs
            curr_fast = (bar.close * self.fast_alpha) + (
                self.fast_ma[bar.symbol] * (1.0 - self.fast_alpha)
            )
            curr_slow = (bar.close * self.slow_alpha) + (
                self.slow_ma[bar.symbol] * (1.0 - self.slow_alpha)
            )
            self.fast_ma[bar.symbol] = curr_fast
            self.slow_ma[bar.symbol] = curr_slow
        else:
            if len(history) < self.slow_period:
                return
            prices = list(history)
            curr_fast = sum(prices[-self.fast_period :]) / self.fast_period
            curr_slow = sum(prices) / self.slow_period

        curr_diff = curr_fast - curr_slow
        prev_diff = self.prev_diff.get(bar.symbol, curr_diff)
        self.prev_diff[bar.symbol] = curr_diff

        pos = None
        if self.portfolio_ref is not None:
            pos = self.portfolio_ref.positions.get(bar.symbol)

        is_long = pos is not None and pos.is_long
        is_flat = pos is None or pos.is_flat

        # Golden Cross: Fast EMA crosses above Slow EMA
        if prev_diff <= 0.0 and curr_diff > 0.0 and is_flat:
            self.buy(symbol=bar.symbol, quantity=self.trade_size)

        # Death Cross: Fast EMA crosses below Slow EMA
        elif prev_diff >= 0.0 and curr_diff < 0.0 and is_long:
            self.close_position(bar.symbol)
