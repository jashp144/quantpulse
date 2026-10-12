"""Slippage and market impact models for QuantPulse."""

from abc import ABC, abstractmethod

from quantpulse.core.types import OrderSide


class BaseSlippageModel(ABC):
    """Abstract base class for order execution slippage models."""

    @abstractmethod
    def calculate_fill_price(
        self,
        side: OrderSide,
        current_price: float,
        quantity: float,
        bar_volume: float = 0.0,
    ) -> tuple[float, float]:
        """Calculate effective execution fill price and total price slippage.

        Returns:
            tuple[float, float]: (effective_fill_price, slippage_amount_per_unit)
        """


class FixedBasisPointSlippage(BaseSlippageModel):
    """Fixed basis point slippage model."""

    def __init__(self, basis_points: float = 5.0) -> None:
        """Initialize fixed basis point slippage.

        Args:
            basis_points: Slippage penalty in basis points (1 bp = 0.0001 = 0.01%).
        """
        self.basis_points = max(0.0, basis_points)

    def calculate_fill_price(
        self,
        side: OrderSide,
        current_price: float,
        quantity: float,
        bar_volume: float = 0.0,
    ) -> tuple[float, float]:
        fraction = self.basis_points / 10000.0
        slippage_per_unit = current_price * fraction

        if side == OrderSide.BUY:
            fill_price = current_price + slippage_per_unit
        else:
            fill_price = max(0.01, current_price - slippage_per_unit)

        return fill_price, slippage_per_unit


class AlmgrenChrissSlippage(BaseSlippageModel):
    """Square-root volume participation and market impact model based on Almgren-Chriss."""

    def __init__(
        self,
        eta: float = 0.1,  # Volatility/liquidity impact parameter
        alpha: float = 0.5,  # Square-root law exponent
        half_spread_bps: float = 2.0,  # Bid-ask half-spread in basis points
        max_impact_pct: float = 0.05,  # Maximum cap on price impact (5%)
    ) -> None:
        self.eta = max(0.0, eta)
        self.alpha = max(0.0, alpha)
        self.half_spread_bps = max(0.0, half_spread_bps)
        self.max_impact_pct = max(0.0, max_impact_pct)

    def calculate_fill_price(
        self,
        side: OrderSide,
        current_price: float,
        quantity: float,
        bar_volume: float = 0.0,
    ) -> tuple[float, float]:
        abs_qty = abs(quantity)
        half_spread_frac = self.half_spread_bps / 10000.0

        if bar_volume > 0.0:
            participation_ratio = min(1.0, abs_qty / bar_volume)
            volume_impact = self.eta * (participation_ratio**self.alpha)
        else:
            # When volume is unrecorded, rely on base liquidity assumptions
            volume_impact = self.eta * 0.05

        total_impact_fraction = min(self.max_impact_pct, volume_impact + half_spread_frac)
        slippage_per_unit = current_price * total_impact_fraction

        if side == OrderSide.BUY:
            fill_price = current_price + slippage_per_unit
        else:
            fill_price = max(0.01, current_price - slippage_per_unit)

        return fill_price, slippage_per_unit
