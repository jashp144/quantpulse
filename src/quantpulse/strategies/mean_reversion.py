"""Mean Reversion Strategy based on Bollinger Bands, Z-score, and RSI."""

from collections import deque

import numpy as np

from quantpulse.core.models import Bar
from quantpulse.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion trading strategy using Bollinger Bands, Z-Score, and RSI."""

    def __init__(
        self,
        window: int = 20,
        z_threshold: float = 2.0,
        rsi_period: int = 14,
        oversold_rsi: float = 35.0,
        overbought_rsi: float = 65.0,
        trade_size: float = 100.0,
        allow_short: bool = False,
    ) -> None:
        super().__init__(name="MeanReversionStrategy")
        self.window = window
        self.z_threshold = z_threshold
        self.rsi_period = rsi_period
        self.oversold_rsi = oversold_rsi
        self.overbought_rsi = overbought_rsi
        self.trade_size = trade_size
        self.allow_short = allow_short

        self.price_history: dict[str, deque[float]] = {}

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol not in self.price_history:
            self.price_history[bar.symbol] = deque(maxlen=max(self.window, self.rsi_period + 5))

        history = self.price_history[bar.symbol]
        history.append(bar.close)

        if len(history) < max(self.window, self.rsi_period + 1):
            return

        prices = np.asarray(history, dtype=np.float64)
        window_prices = prices[-self.window :]
        mean = np.mean(window_prices)
        std = np.std(window_prices, ddof=1)

        if std < 1e-8:
            return

        z_score = (bar.close - mean) / std
        rsi = self._calculate_rsi(prices[-self.rsi_period - 1 :])

        pos = None
        if self.portfolio_ref is not None:
            pos = self.portfolio_ref.positions.get(bar.symbol)

        is_long = pos is not None and pos.is_long
        is_short = pos is not None and pos.is_short
        is_flat = pos is None or pos.is_flat

        if is_flat:
            # Enter Long on oversold extreme
            if z_score <= -self.z_threshold and rsi <= self.oversold_rsi:
                self.buy(symbol=bar.symbol, quantity=self.trade_size)
            # Enter Short on overbought extreme (if permitted)
            elif self.allow_short and z_score >= self.z_threshold and rsi >= self.overbought_rsi:
                self.sell(symbol=bar.symbol, quantity=self.trade_size)

        elif is_long and (z_score >= 0.0 or rsi >= 50.0):
            # Exit long position when mean reversion target reached
            self.close_position(bar.symbol)

        elif is_short and (z_score <= 0.0 or rsi <= 50.0):
            # Exit short position when mean reversion target reached
            self.close_position(bar.symbol)

    def _calculate_rsi(self, price_slice: np.ndarray) -> float:
        """Calculate Relative Strength Index (RSI)."""
        if len(price_slice) < 2:
            return 50.0

        deltas = np.diff(price_slice)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss < 1e-12:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))
