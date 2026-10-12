"""Quantitative risk, return, and portfolio performance metrics for QuantPulse."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


class RiskMetrics:
    """Calculates institutional risk and performance metrics."""

    @staticmethod
    def cagr(
        equity: pd.Series | np.ndarray | Sequence[float],
        periods_per_year: int = 252,
    ) -> float:
        """Calculate Compound Annual Growth Rate (CAGR)."""
        arr = np.asarray(equity, dtype=np.float64)
        if len(arr) < 2 or arr[0] <= 0.0 or arr[-1] <= 0.0:
            return 0.0

        n_periods = len(arr) - 1
        total_return = arr[-1] / arr[0]
        years = n_periods / periods_per_year
        if years <= 0.0:
            return 0.0

        return float((total_return ** (1.0 / years)) - 1.0)

    @staticmethod
    def annualized_volatility(
        returns: pd.Series | np.ndarray | Sequence[float],
        periods_per_year: int = 252,
    ) -> float:
        """Calculate annualized volatility from returns."""
        arr = np.asarray(returns, dtype=np.float64)
        if len(arr) < 2:
            return 0.0
        std = np.std(arr, ddof=1)
        return float(std * np.sqrt(periods_per_year))

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series | np.ndarray | Sequence[float],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate annualized Sharpe Ratio.

        Formula:
            Sharpe = (mean(R) - r_f / periods) / std(R) * sqrt(periods)
        """
        arr = np.asarray(returns, dtype=np.float64)
        if len(arr) < 2:
            return 0.0

        std = np.std(arr, ddof=1)
        if std < 1e-12:
            return 0.0

        rf_period = risk_free_rate / periods_per_year
        excess_mean = np.mean(arr) - rf_period
        return float((excess_mean / std) * np.sqrt(periods_per_year))

    @staticmethod
    def sortino_ratio(
        returns: pd.Series | np.ndarray | Sequence[float],
        target_return: float = 0.0,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate annualized Sortino Ratio using downside semideviation.

        Formula:
            Sortino = (mean(R) - r_f / periods) / downside_std * sqrt(periods)
        """
        arr = np.asarray(returns, dtype=np.float64)
        if len(arr) < 2:
            return 0.0

        target_period = target_return / periods_per_year
        downside_diff = np.minimum(0.0, arr - target_period)
        downside_variance = np.mean(downside_diff**2)
        downside_std = np.sqrt(downside_variance) * np.sqrt(periods_per_year)

        rf_period = risk_free_rate / periods_per_year
        excess_mean = np.mean(arr) - rf_period
        annualized_excess = excess_mean * periods_per_year

        if downside_std < 1e-12:
            return float("inf") if annualized_excess > 0 else 0.0

        return float(annualized_excess / downside_std)

    @staticmethod
    def drawdown_series(
        equity: pd.Series | np.ndarray | Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute running high-water mark and drawdown percentage series.

        Returns:
            tuple[np.ndarray, np.ndarray]: (high_water_marks, drawdown_percentages)
        """
        arr = np.asarray(equity, dtype=np.float64)
        if len(arr) == 0:
            return np.array([]), np.array([])

        hwm = np.maximum.accumulate(arr)
        drawdowns = np.where(hwm > 1e-12, (arr - hwm) / hwm, 0.0)
        return hwm, drawdowns

    @staticmethod
    def max_drawdown(
        equity: pd.Series | np.ndarray | Sequence[float],
    ) -> float:
        """Calculate Maximum Drawdown (MDD) as a negative float (e.g. -0.20 for -20%)."""
        _, drawdowns = RiskMetrics.drawdown_series(equity)
        if len(drawdowns) == 0:
            return 0.0
        return float(np.min(drawdowns))

    @staticmethod
    def max_drawdown_duration(
        equity: pd.Series | np.ndarray | Sequence[float],
    ) -> int:
        """Calculate the longest drawdown period in bars/days."""
        arr = np.asarray(equity, dtype=np.float64)
        if len(arr) < 2:
            return 0

        hwm = np.maximum.accumulate(arr)
        max_duration = 0
        current_duration = 0

        for i in range(len(arr)):
            if arr[i] < hwm[i]:
                current_duration += 1
                if current_duration > max_duration:
                    max_duration = current_duration
            else:
                current_duration = 0

        return max_duration

    @staticmethod
    def calmar_ratio(
        equity: pd.Series | np.ndarray | Sequence[float],
        periods_per_year: int = 252,
    ) -> float:
        """Calculate Calmar Ratio: CAGR / abs(MDD)."""
        cagr_val = RiskMetrics.cagr(equity, periods_per_year)
        mdd = abs(RiskMetrics.max_drawdown(equity))
        if mdd < 1e-12:
            return float("inf") if cagr_val > 0 else 0.0
        return float(cagr_val / mdd)

    @staticmethod
    def win_rate(trade_pnls: Sequence[float]) -> float:
        """Calculate trade win rate proportion (0.0 to 1.0)."""
        pnls = [p for p in trade_pnls if abs(p) > 1e-12]
        if not pnls:
            return 0.0
        wins = sum(1 for p in pnls if p > 0.0)
        return float(wins / len(pnls))

    @staticmethod
    def profit_factor(trade_pnls: Sequence[float]) -> float:
        """Calculate Profit Factor (Gross Profits / Gross Losses)."""
        gross_profit = sum(p for p in trade_pnls if p > 0.0)
        gross_loss = abs(sum(p for p in trade_pnls if p < 0.0))

        if gross_loss < 1e-12:
            return float("inf") if gross_profit > 0.0 else 0.0
        return float(gross_profit / gross_loss)

    @staticmethod
    def beta(
        portfolio_returns: pd.Series | np.ndarray | Sequence[float],
        benchmark_returns: pd.Series | np.ndarray | Sequence[float],
    ) -> float:
        """Calculate Beta sensitivity relative to benchmark."""
        p_arr = np.asarray(portfolio_returns, dtype=np.float64)
        b_arr = np.asarray(benchmark_returns, dtype=np.float64)
        if len(p_arr) < 2 or len(b_arr) < 2 or len(p_arr) != len(b_arr):
            return 1.0

        var_b = np.var(b_arr, ddof=1)
        if var_b < 1e-12:
            return 1.0

        cov_pb = np.cov(p_arr, b_arr, ddof=1)[0, 1]
        return float(cov_pb / var_b)

    @staticmethod
    def alpha(
        portfolio_returns: pd.Series | np.ndarray | Sequence[float],
        benchmark_returns: pd.Series | np.ndarray | Sequence[float],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate annualized Jensen's Alpha."""
        p_arr = np.asarray(portfolio_returns, dtype=np.float64)
        b_arr = np.asarray(benchmark_returns, dtype=np.float64)
        if len(p_arr) < 2 or len(b_arr) < 2 or len(p_arr) != len(b_arr):
            return 0.0

        beta_val = RiskMetrics.beta(p_arr, b_arr)
        mean_p_ann = np.mean(p_arr) * periods_per_year
        mean_b_ann = np.mean(b_arr) * periods_per_year

        # Jensen's Alpha = (R_p - R_f) - beta * (R_m - R_f)
        return float((mean_p_ann - risk_free_rate) - beta_val * (mean_b_ann - risk_free_rate))

    @staticmethod
    def information_ratio(
        portfolio_returns: pd.Series | np.ndarray | Sequence[float],
        benchmark_returns: pd.Series | np.ndarray | Sequence[float],
        periods_per_year: int = 252,
    ) -> float:
        """Calculate annualized Information Ratio against benchmark."""
        p_arr = np.asarray(portfolio_returns, dtype=np.float64)
        b_arr = np.asarray(benchmark_returns, dtype=np.float64)
        if len(p_arr) < 2 or len(b_arr) < 2 or len(p_arr) != len(b_arr):
            return 0.0

        diff = p_arr - b_arr
        tracking_error = np.std(diff, ddof=1) * np.sqrt(periods_per_year)
        if tracking_error < 1e-12:
            return 0.0

        excess_mean_ann = np.mean(diff) * periods_per_year
        return float(excess_mean_ann / tracking_error)
