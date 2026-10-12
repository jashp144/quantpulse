"""Unit tests for quantitative risk metrics, Sharpe, Sortino, MDD, VaR, CVaR, Alpha, Beta."""

import numpy as np
import pytest
from scipy import stats

from quantpulse.risk.metrics import RiskMetrics
from quantpulse.risk.var import ValueAtRisk


@pytest.mark.risk
def test_cagr_calculation():
    # An asset doubling over exactly 252 trading days has CAGR = 100%
    equity_curve = np.linspace(100.0, 200.0, 253)
    cagr = RiskMetrics.cagr(equity_curve, periods_per_year=252)
    assert pytest.approx(cagr, rel=1e-4) == 1.0

    # Constant equity curve has CAGR = 0%
    flat_curve = np.full(253, 100.0)
    assert RiskMetrics.cagr(flat_curve) == 0.0

    # Single point curve returns 0
    assert RiskMetrics.cagr([100.0]) == 0.0


@pytest.mark.risk
def test_annualized_volatility():
    # Fixed known returns
    returns = np.array([0.01, -0.01, 0.02, -0.02, 0.01, -0.01])
    sample_std = np.std(returns, ddof=1)
    expected_vol = sample_std * np.sqrt(252)

    vol = RiskMetrics.annualized_volatility(returns, periods_per_year=252)
    assert pytest.approx(vol, rel=1e-5) == expected_vol

    # Empty and single value return 0.0
    assert RiskMetrics.annualized_volatility([]) == 0.0
    assert RiskMetrics.annualized_volatility([0.05]) == 0.0


@pytest.mark.risk
def test_sharpe_ratio_formula():
    # Array with daily mean return 0.001 (0.1%), daily std 0.01 (1.0%)
    np.random.seed(42)
    daily_returns = np.random.normal(loc=0.001, scale=0.01, size=500)

    sample_mean = np.mean(daily_returns)
    sample_std = np.std(daily_returns, ddof=1)
    expected_sharpe = (sample_mean / sample_std) * np.sqrt(252)

    sharpe = RiskMetrics.sharpe_ratio(daily_returns, risk_free_rate=0.0, periods_per_year=252)
    assert pytest.approx(sharpe, rel=1e-5) == expected_sharpe

    # With risk-free rate
    rf = 0.04  # 4% annual
    rf_daily = 0.04 / 252
    expected_sharpe_rf = ((sample_mean - rf_daily) / sample_std) * np.sqrt(252)
    sharpe_rf = RiskMetrics.sharpe_ratio(daily_returns, risk_free_rate=rf, periods_per_year=252)
    assert pytest.approx(sharpe_rf, rel=1e-5) == expected_sharpe_rf

    # Zero volatility edge case
    assert RiskMetrics.sharpe_ratio(np.full(100, 0.01)) == 0.0


@pytest.mark.risk
def test_sortino_ratio_downside_deviation():
    # Only positive days: downside deviation is 0 -> Sortino is infinite or large
    all_positive = np.array([0.01, 0.02, 0.015, 0.03, 0.005])
    sortino_pos = RiskMetrics.sortino_ratio(all_positive)
    assert sortino_pos == float("inf")

    # Mixed returns
    mixed_returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.005])
    downside = np.minimum(0.0, mixed_returns)
    downside_std = np.sqrt(np.mean(downside**2)) * np.sqrt(252)
    expected_sortino = (np.mean(mixed_returns) * 252) / downside_std

    sortino = RiskMetrics.sortino_ratio(mixed_returns, periods_per_year=252)
    assert pytest.approx(sortino, rel=1e-5) == expected_sortino


@pytest.mark.risk
def test_max_drawdown_and_duration():
    # Step equity series: 100 -> 120 -> 60 -> 90 -> 150 -> 140
    # High water marks: 100, 120, 120, 120, 150, 150
    # Drawdowns: 0, 0, (60-120)/120 = -0.50, (90-120)/120 = -0.25, 0, (140-150)/150 = -0.0667
    # Max Drawdown = -50% (-0.50)
    equity = [100.0, 120.0, 60.0, 90.0, 150.0, 140.0]

    mdd = RiskMetrics.max_drawdown(equity)
    assert pytest.approx(mdd, rel=1e-5) == -0.50

    # Drawdown duration: peak at idx 1 (120), below peak at idx 2 (60), idx 3 (90), then new peak at idx 4 (150).
    # Duration was 2 periods.
    duration = RiskMetrics.max_drawdown_duration(equity)
    assert duration == 2

    # Calmar ratio
    calmar = RiskMetrics.calmar_ratio(equity, periods_per_year=252)
    assert calmar > 0.0


@pytest.mark.risk
def test_win_rate_and_profit_factor():
    pnls = [100.0, -50.0, 200.0, -50.0, 150.0]  # 3 wins, 2 losses -> 60% win rate
    assert pytest.approx(RiskMetrics.win_rate(pnls), rel=1e-5) == 0.60

    # Profit factor: (100 + 200 + 150) / (50 + 50) = 450 / 100 = 4.5
    assert pytest.approx(RiskMetrics.profit_factor(pnls), rel=1e-5) == 4.5

    # Zero losses gives inf profit factor
    assert RiskMetrics.profit_factor([100.0, 200.0]) == float("inf")
    # Empty trades gives 0.0
    assert RiskMetrics.win_rate([]) == 0.0


@pytest.mark.risk
def test_benchmark_beta_alpha_and_information_ratio():
    np.random.seed(42)
    bench_returns = np.random.normal(0.0005, 0.01, 200)

    # Perfect correlation with 1.5x leverage
    port_returns = 1.5 * bench_returns + 0.0002

    beta = RiskMetrics.beta(port_returns, bench_returns)
    assert pytest.approx(beta, rel=1e-4) == 1.5

    alpha = RiskMetrics.alpha(port_returns, bench_returns, risk_free_rate=0.0)
    # Excess return of portfolio over 1.5 * benchmark is 0.0002 daily * 252 = 0.0504
    assert pytest.approx(alpha, rel=1e-3) == 0.0002 * 252

    ir = RiskMetrics.information_ratio(port_returns, bench_returns)
    assert ir != 0.0


@pytest.mark.risk
def test_parametric_and_historical_var_cvar():
    np.random.seed(42)
    # 10,000 normally distributed samples: mu = 0, sigma = 0.02
    returns = np.random.normal(loc=0.0, scale=0.02, size=10_000)

    # Parametric VaR at 95%
    z_95 = stats.norm.ppf(0.95)  # ~1.64485
    expected_param_var = z_95 * 0.02
    param_var = ValueAtRisk.parametric_var(returns, confidence=0.95)
    assert pytest.approx(param_var, rel=0.05) == expected_param_var

    # Parametric CVaR at 95%
    pdf_z = stats.norm.pdf(z_95)
    expected_param_cvar = 0.02 * (pdf_z / 0.05)
    param_cvar = ValueAtRisk.parametric_cvar(returns, confidence=0.95)
    assert pytest.approx(param_cvar, rel=0.05) == expected_param_cvar
    assert param_cvar > param_var  # CVaR is always greater than or equal to VaR

    # Historical VaR and CVaR
    hist_var = ValueAtRisk.historical_var(returns, confidence=0.95)
    hist_cvar = ValueAtRisk.historical_cvar(returns, confidence=0.95)
    assert pytest.approx(hist_var, rel=0.05) == expected_param_var
    assert pytest.approx(hist_cvar, rel=0.05) == expected_param_cvar
    assert hist_cvar >= hist_var

    # 99% VaR and CVaR
    var_99 = ValueAtRisk.parametric_var(returns, confidence=0.99)
    assert var_99 > param_var
