"""Value-at-Risk (VaR) and Conditional VaR (Expected Shortfall) models."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats


class ValueAtRisk:
    """Calculates Parametric and Historical Value at Risk (VaR) and Conditional VaR (CVaR)."""

    @staticmethod
    def parametric_var(
        returns: pd.Series | np.ndarray | Sequence[float],
        confidence: float = 0.95,
    ) -> float:
        """Calculate Parametric Value at Risk (Gaussian assumption).

        Returns VaR as a positive loss number (e.g. 0.025 indicates 2.5% loss).
        Formula:
            VaR_alpha = - (mu - z_alpha * sigma)
        """
        arr = np.asarray(returns, dtype=np.float64)
        if len(arr) < 2:
            return 0.0

        mu = np.mean(arr)
        sigma = np.std(arr, ddof=1)
        z = stats.norm.ppf(confidence)
        # VaR represents positive potential loss
        var_value = -(mu - z * sigma)
        return float(max(0.0, var_value))

    @staticmethod
    def parametric_cvar(
        returns: pd.Series | np.ndarray | Sequence[float],
        confidence: float = 0.95,
    ) -> float:
        """Calculate Parametric Conditional Value at Risk (Expected Shortfall).

        Formula:
            CVaR_alpha = - (mu - sigma * (phi(z_alpha) / (1 - alpha)))
        """
        arr = np.asarray(returns, dtype=np.float64)
        if len(arr) < 2:
            return 0.0

        mu = np.mean(arr)
        sigma = np.std(arr, ddof=1)
        z = stats.norm.ppf(confidence)
        pdf_z = stats.norm.pdf(z)

        # Expected shortfall for standard normal
        es_multiplier = pdf_z / (1.0 - confidence)
        cvar_value = -(mu - sigma * es_multiplier)
        return float(max(0.0, cvar_value))

    @staticmethod
    def historical_var(
        returns: pd.Series | np.ndarray | Sequence[float],
        confidence: float = 0.95,
    ) -> float:
        """Calculate Historical Simulation Value at Risk.

        Computes empirical percentile corresponding to (1 - confidence).
        """
        arr = np.asarray(returns, dtype=np.float64)
        if len(arr) < 2:
            return 0.0

        percentile_pct = (1.0 - confidence) * 100.0
        q = np.percentile(arr, percentile_pct)
        # VaR as positive loss
        return float(max(0.0, -q))

    @staticmethod
    def historical_cvar(
        returns: pd.Series | np.ndarray | Sequence[float],
        confidence: float = 0.95,
    ) -> float:
        """Calculate Historical Conditional Value at Risk (Expected Shortfall).

        Average of historical losses strictly beyond empirical VaR threshold.
        """
        arr = np.asarray(returns, dtype=np.float64)
        if len(arr) < 2:
            return 0.0

        percentile_pct = (1.0 - confidence) * 100.0
        q = np.percentile(arr, percentile_pct)
        tail_losses = arr[arr <= q]

        if len(tail_losses) == 0:
            return float(max(0.0, -q))

        mean_tail_loss = np.mean(tail_losses)
        return float(max(0.0, -mean_tail_loss))
