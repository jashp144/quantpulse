"""Shared pytest fixtures, synthetic market data generators, and deterministic seeds."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def set_deterministic_seed():
    """Ensure all tests execute with deterministic random seeds."""
    np.random.seed(42)


@pytest.fixture
def gbm_price_series():
    """Generate Geometric Brownian Motion (GBM) synthetic daily price series."""
    n_days = 252
    dt = 1.0 / 252.0
    mu = 0.10  # 10% annual drift
    sigma = 0.20  # 20% annual volatility
    s0 = 100.0

    np.random.seed(42)
    shocks = np.random.normal(0, 1, n_days)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    prices = s0 * np.exp(np.cumsum(log_returns))
    prices = np.insert(prices, 0, s0)

    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(len(prices))]

    # Generate plausible OHLCV around close prices
    df = pd.DataFrame(
        {
            "open": prices * (1.0 + np.random.uniform(-0.005, 0.005, len(prices))),
            "high": prices * (1.0 + np.random.uniform(0.005, 0.02, len(prices))),
            "low": prices * (1.0 - np.random.uniform(0.005, 0.02, len(prices))),
            "close": prices,
            "volume": np.random.uniform(50_000, 200_000, len(prices)),
        },
        index=pd.DatetimeIndex(dates),
    )
    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
    df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))
    return df


@pytest.fixture
def trending_price_series():
    """Generate price series with initial dip followed by a strong upward trend."""
    n_days = 120
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(n_days)]
    part1 = np.linspace(100.0, 95.0, 25)
    part2 = np.linspace(95.0, 220.0, n_days - 25)
    base = np.concatenate([part1, part2])
    noise = np.random.normal(0, 0.5, n_days)
    closes = base + noise

    df = pd.DataFrame(
        {
            "open": closes - 0.5,
            "high": closes + 2.0,
            "low": closes - 2.0,
            "close": closes,
            "volume": 100_000.0,
        },
        index=pd.DatetimeIndex(dates),
    )
    return df


@pytest.fixture
def mean_reverting_price_series():
    """Generate Ornstein-Uhlenbeck mean-reverting price series."""
    n_days = 120
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(n_days)]

    theta = 0.3  # Mean-reversion speed
    mu = 100.0  # Long-term mean price
    sigma = 3.0  # Volatility
    dt = 1.0

    prices = [mu]
    np.random.seed(42)
    for _ in range(n_days - 1):
        prev = prices[-1]
        dp = theta * (mu - prev) * dt + sigma * np.random.normal(0, np.sqrt(dt))
        prices.append(prev + dp)

    closes = np.array(prices)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1.5,
            "low": closes - 1.5,
            "close": closes,
            "volume": 75_000.0,
        },
        index=pd.DatetimeIndex(dates),
    )
    return df
