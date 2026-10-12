"""Performance tear-sheet generator for QuantPulse backtest results."""

from typing import Any


class TearSheetGenerator:
    """Renders formatted ASCII tables and Markdown tear-sheets."""

    @staticmethod
    def generate_markdown(metrics: dict[str, Any]) -> str:
        """Format metrics dictionary into a rich Markdown tear-sheet."""
        lines = [
            "# 📈 QuantPulse Performance Tear-Sheet",
            "",
            "## 1. Executive Performance Summary",
            "",
            "| Metric | Value |",
            "|:---|:---|",
            f"| **Initial Capital** | ${metrics.get('initial_capital', 0.0):,.2f} |",
            f"| **Final Equity** | ${metrics.get('final_equity', 0.0):,.2f} |",
            f"| **Total Return** | {metrics.get('total_return', 0.0) * 100:.2f}% |",
            f"| **CAGR (Annualized Return)** | {metrics.get('cagr', 0.0) * 100:.2f}% |",
            f"| **Annualized Volatility** | {metrics.get('annualized_volatility', 0.0) * 100:.2f}% |",
            "",
            "## 2. Risk-Adjusted Ratios",
            "",
            "| Metric | Value |",
            "|:---|:---|",
            f"| **Sharpe Ratio** | {metrics.get('sharpe_ratio', 0.0):.4f} |",
            f"| **Sortino Ratio** | {metrics.get('sortino_ratio', 0.0):.4f} |",
            f"| **Calmar Ratio** | {metrics.get('calmar_ratio', 0.0):.4f} |",
            f"| **Maximum Drawdown (MDD)** | {metrics.get('max_drawdown', 0.0) * 100:.2f}% |",
            f"| **Max Drawdown Duration** | {metrics.get('max_drawdown_duration', 0)} periods |",
            "",
            "## 3. Value-at-Risk (VaR) & Downside Tail Risk",
            "",
            "| Metric | Value |",
            "|:---|:---|",
            f"| **Daily Parametric VaR (95%)** | {metrics.get('var_95', 0.0) * 100:.2f}% |",
            f"| **Daily Conditional VaR / CVaR (95%)** | {metrics.get('cvar_95', 0.0) * 100:.2f}% |",
            f"| **Daily Parametric VaR (99%)** | {metrics.get('var_99', 0.0) * 100:.2f}% |",
            f"| **Daily Conditional VaR / CVaR (99%)** | {metrics.get('cvar_99', 0.0) * 100:.2f}% |",
            "",
            "## 4. Trade Execution Statistics",
            "",
            "| Metric | Value |",
            "|:---|:---|",
            f"| **Total Closed Trades** | {metrics.get('total_trades', 0)} |",
            f"| **Win Rate** | {metrics.get('win_rate', 0.0) * 100:.2f}% |",
            f"| **Profit Factor** | {metrics.get('profit_factor', 0.0):.4f} |",
        ]

        if metrics.get("beta") is not None:
            lines.extend(
                [
                    "",
                    "## 5. Benchmark Comparison",
                    "",
                    "| Metric | Value |",
                    "|:---|:---|",
                    f"| **Beta** | {metrics.get('beta', 1.0):.4f} |",
                    f"| **Alpha (Annualized)** | {metrics.get('alpha', 0.0) * 100:.2f}% |",
                    f"| **Information Ratio** | {metrics.get('information_ratio', 0.0):.4f} |",
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def generate_ascii(metrics: dict[str, Any]) -> str:
        """Format metrics dictionary into a clean ASCII table."""
        banner = "=" * 65
        sub_banner = "-" * 65
        lines = [
            banner,
            "              QUANTPULSE PERFORMANCE TEAR-SHEET",
            banner,
            f" Initial Capital:       ${metrics.get('initial_capital', 0.0):>14,.2f}",
            f" Final Equity:          ${metrics.get('final_equity', 0.0):>14,.2f}",
            f" Total Return:           {metrics.get('total_return', 0.0) * 100:>14.2f}%",
            f" CAGR:                   {metrics.get('cagr', 0.0) * 100:>14.2f}%",
            f" Annualized Volatility:  {metrics.get('annualized_volatility', 0.0) * 100:>14.2f}%",
            sub_banner,
            f" Sharpe Ratio:           {metrics.get('sharpe_ratio', 0.0):>15.4f}",
            f" Sortino Ratio:          {metrics.get('sortino_ratio', 0.0):>15.4f}",
            f" Calmar Ratio:           {metrics.get('calmar_ratio', 0.0):>15.4f}",
            f" Maximum Drawdown:       {metrics.get('max_drawdown', 0.0) * 100:>14.2f}%",
            f" Max Drawdown Duration:  {metrics.get('max_drawdown_duration', 0):>15} bars",
            sub_banner,
            f" Parametric VaR (95%):   {metrics.get('var_95', 0.0) * 100:>14.2f}%",
            f" Conditional VaR (95%):  {metrics.get('cvar_95', 0.0) * 100:>14.2f}%",
            f" Total Trades:           {metrics.get('total_trades', 0):>15}",
            f" Win Rate:               {metrics.get('win_rate', 0.0) * 100:>14.2f}%",
            f" Profit Factor:          {metrics.get('profit_factor', 0.0):>15.4f}",
            banner,
        ]
        return "\n".join(lines)
