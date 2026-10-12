"""Container for backtest execution results and tear-sheet presentation."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantpulse.engine.tear_sheet import TearSheetGenerator


@dataclass
class BacktestResult:
    """Comprehensive backtest results container with performance attributes."""

    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    total_trades: int
    win_rate: float
    profit_factor: float
    var_95: float
    cvar_95: float
    var_99: float = 0.0
    cvar_99: float = 0.0
    alpha: float | None = None
    beta: float | None = None
    information_ratio: float | None = None
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    drawdown_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return a flat dictionary of key performance metrics."""
        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "var_99": self.var_99,
            "cvar_99": self.cvar_99,
            "alpha": self.alpha,
            "beta": self.beta,
            "information_ratio": self.information_ratio,
        }

    def tear_sheet(self, format_type: str = "markdown") -> str:
        """Generate tear-sheet in markdown or ascii format."""
        metrics = self.summary()
        if format_type.lower() == "ascii":
            return TearSheetGenerator.generate_ascii(metrics)
        return TearSheetGenerator.generate_markdown(metrics)

    def to_dataframe(self) -> pd.DataFrame:
        """Export portfolio state timeline to a pandas DataFrame."""
        df = pd.DataFrame(
            {
                "equity": self.equity_curve,
                "drawdown": self.drawdown_curve,
                "returns": self.returns,
            }
        )
        return df
