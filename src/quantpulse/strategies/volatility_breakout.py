"""Volatility Breakout Strategy (Donchian / ATR Channel) for QuantPulse."""

from collections import deque

import numpy as np

from quantpulse.core.models import Bar
from quantpulse.strategies.base import BaseStrategy


class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility Breakout Strategy combining Donchian Channel and Average True Range (ATR)."""

    def __init__(
        self,
        channel_period: int = 20,
        atr_period: int = 14,
        stop_atr_mult: float = 2.0,
        trade_size: float = 100.0,
    ) -> None:
        super().__init__(name="VolatilityBreakoutStrategy")
        self.channel_period = channel_period
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult
        self.trade_size = trade_size

        self.highs: dict[str, deque[float]] = {}
        self.lows: dict[str, deque[float]] = {}
        self.closes: dict[str, deque[float]] = {}
        self.tr_history: dict[str, deque[float]] = {}
        self.trailing_stops: dict[str, float] = {}

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol not in self.highs:
            self.highs[bar.symbol] = deque(maxlen=self.channel_period + 1)
            self.lows[bar.symbol] = deque(maxlen=self.channel_period + 1)
            self.closes[bar.symbol] = deque(maxlen=self.channel_period + 1)
            self.tr_history[bar.symbol] = deque(maxlen=self.atr_period)

        prev_close = self.closes[bar.symbol][-1] if self.closes[bar.symbol] else bar.open

        # True Range calculation
        tr = max(
            bar.high - bar.low,
            abs(bar.high - prev_close),
            abs(bar.low - prev_close),
        )
        self.tr_history[bar.symbol].append(tr)
        self.highs[bar.symbol].append(bar.high)
        self.lows[bar.symbol].append(bar.low)
        self.closes[bar.symbol].append(bar.close)

        min_required = max(self.channel_period, self.atr_period) + 1
        if len(self.closes[bar.symbol]) < min_required:
            return

        atr = float(np.mean(self.tr_history[bar.symbol]))
        # Donchian upper channel of preceding bars (excluding current bar)
        recent_highs = list(self.highs[bar.symbol])[:-1]
        upper_channel = max(recent_highs[-self.channel_period :])

        pos = None
        if self.portfolio_ref is not None:
            pos = self.portfolio_ref.positions.get(bar.symbol)

        is_long = pos is not None and pos.is_long
        is_flat = pos is None or pos.is_flat

        if is_flat:
            # Upside Breakout confirmation
            if bar.close > upper_channel:
                initial_stop = bar.close - (self.stop_atr_mult * atr)
                self.trailing_stops[bar.symbol] = initial_stop
                self.buy(symbol=bar.symbol, quantity=self.trade_size)

        elif is_long:
            current_stop = self.trailing_stops.get(bar.symbol, bar.close - 2.0 * atr)

            # Check if trailing stop triggered
            if bar.low <= current_stop:
                self.close_position(bar.symbol)
                self.trailing_stops.pop(bar.symbol, None)
            else:
                # Ratchet trailing stop higher
                candidate_stop = bar.close - (self.stop_atr_mult * atr)
                if candidate_stop > current_stop:
                    self.trailing_stops[bar.symbol] = candidate_stop
