"""Quantitative trading strategies for QuantPulse."""

from quantpulse.strategies.base import BaseStrategy
from quantpulse.strategies.ma_crossover import MovingAverageCrossoverStrategy
from quantpulse.strategies.mean_reversion import MeanReversionStrategy
from quantpulse.strategies.volatility_breakout import VolatilityBreakoutStrategy

__all__ = [
    "BaseStrategy",
    "MeanReversionStrategy",
    "MovingAverageCrossoverStrategy",
    "VolatilityBreakoutStrategy",
]
