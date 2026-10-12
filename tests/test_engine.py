"""Integration and end-to-end tests for QuantPulse BacktestEngine."""

import pandas as pd
import pytest

from quantpulse import (
    BacktestEngine,
    MovingAverageCrossoverStrategy,
)
from quantpulse.execution.fees import FlatFeeModel
from quantpulse.execution.slippage import AlmgrenChrissSlippage


@pytest.mark.integration
def test_project_interface_contract(gbm_price_series):
    """Assert PROJECT.md public interface contract runs and returns valid metrics."""
    engine = BacktestEngine(initial_cash=100_000.0)
    results = engine.run(gbm_price_series)

    assert results.sharpe_ratio is not None
    assert results.total_return is not None
    assert results.final_equity > 0.0
    assert len(results.equity_curve) > 0


@pytest.mark.integration
def test_end_to_end_backtest_with_strategy(trending_price_series):
    """End-to-end test on upward trending price series demonstrating deterministic trading."""
    engine = BacktestEngine(
        initial_capital=100_000.0,
        fee_model=FlatFeeModel(per_share=0.005, min_fee=1.0),
        slippage_model=AlmgrenChrissSlippage(eta=0.05, alpha=0.5),
    )
    strategy = MovingAverageCrossoverStrategy(fast_period=5, slow_period=15, trade_size=200.0)
    engine.set_strategy(strategy)
    engine.add_data("TREND_STOCK", trending_price_series)

    result = engine.run()

    assert result.initial_capital == 100_000.0
    assert result.final_equity > result.initial_capital  # Strong upward trend yields profit
    assert result.total_return > 0.0
    assert result.total_trades > 0
    assert result.max_drawdown <= 0.0

    # Tear sheet validation
    md_sheet = result.tear_sheet("markdown")
    assert "QuantPulse Performance Tear-Sheet" in md_sheet
    assert "Initial Capital" in md_sheet

    ascii_sheet = result.tear_sheet("ascii")
    assert "QUANTPULSE PERFORMANCE TEAR-SHEET" in ascii_sheet

    # DataFrame export validation
    df = result.to_dataframe()
    assert "equity" in df.columns
    assert "drawdown" in df.columns
    assert "returns" in df.columns


@pytest.mark.integration
def test_deterministic_reproducibility(gbm_price_series):
    """Ensure two backtests on identical input data produce float-identical outcomes."""
    engine1 = BacktestEngine(initial_capital=100_000.0)
    res1 = engine1.run(gbm_price_series)

    engine2 = BacktestEngine(initial_capital=100_000.0)
    res2 = engine2.run(gbm_price_series)

    assert res1.final_equity == res2.final_equity
    assert res1.sharpe_ratio == res2.sharpe_ratio
    assert res1.max_drawdown == res2.max_drawdown


@pytest.mark.integration
def test_multi_asset_backtest(trending_price_series, mean_reverting_price_series):
    """Verify simultaneous multi-asset processing and portfolio state tracking."""
    engine = BacktestEngine(initial_capital=200_000.0)
    engine.add_data("ASSET_A", trending_price_series)
    engine.add_data("ASSET_B", mean_reverting_price_series)

    result = engine.run()
    assert result.final_equity > 0.0
    assert len(result.equity_curve) > 0


@pytest.mark.integration
def test_missing_data_raises_error():
    """Verify clean error raised when running engine without data."""
    engine = BacktestEngine()
    with pytest.raises(ValueError, match="No price data has been added"):
        engine.run()


@pytest.mark.integration
def test_benchmark_metrics_integration(trending_price_series):
    """Verify benchmark comparative metrics (Alpha, Beta, Information Ratio)."""
    n_bars = len(trending_price_series)
    # Synthetic benchmark returns matching length
    benchmark_returns = pd.Series([0.0005] * n_bars)

    engine = BacktestEngine(
        initial_capital=100_000.0,
        benchmark_returns=benchmark_returns,
    )
    result = engine.run(trending_price_series)

    assert result.beta is not None
    assert result.alpha is not None
    assert result.information_ratio is not None
