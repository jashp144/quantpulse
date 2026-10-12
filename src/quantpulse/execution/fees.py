"""Transaction fee and commission models for QuantPulse."""

from abc import ABC, abstractmethod

from quantpulse.core.types import OrderSide


class BaseFeeModel(ABC):
    """Abstract base class for brokerage transaction cost models."""

    @abstractmethod
    def calculate_fee(
        self,
        symbol: str,
        side: OrderSide,
        fill_price: float,
        fill_quantity: float,
        is_maker: bool = False,
    ) -> float:
        """Calculate the total fee for an executed transaction."""


class MakerTakerFee(BaseFeeModel):
    """Maker-Taker percentage fee model commonly used on electronic exchanges."""

    def __init__(
        self,
        maker_fee_pct: float = 0.0002,  # 2 bps maker
        taker_fee_pct: float = 0.0005,  # 5 bps taker
        min_fee: float = 0.0,
    ) -> None:
        self.maker_fee_pct = max(0.0, maker_fee_pct)
        self.taker_fee_pct = max(0.0, taker_fee_pct)
        self.min_fee = max(0.0, min_fee)

    def calculate_fee(
        self,
        symbol: str,
        side: OrderSide,
        fill_price: float,
        fill_quantity: float,
        is_maker: bool = False,
    ) -> float:
        notional = fill_price * abs(fill_quantity)
        rate = self.maker_fee_pct if is_maker else self.taker_fee_pct
        calculated = notional * rate
        return max(self.min_fee, calculated)


class FlatFeeModel(BaseFeeModel):
    """Flat per-share commission model with minimum per-order fee."""

    def __init__(
        self,
        per_share: float = 0.005,  # $0.005 per share
        min_fee: float = 1.0,  # $1.00 min per trade
        max_pct: float = 0.01,  # Max 1.0% of trade notional
    ) -> None:
        self.per_share = max(0.0, per_share)
        self.min_fee = max(0.0, min_fee)
        self.max_pct = max(0.0, max_pct)

    def calculate_fee(
        self,
        symbol: str,
        side: OrderSide,
        fill_price: float,
        fill_quantity: float,
        is_maker: bool = False,
    ) -> float:
        notional = fill_price * abs(fill_quantity)
        raw_fee = max(self.min_fee, abs(fill_quantity) * self.per_share)
        if self.max_pct > 0.0 and notional > 0.0:
            raw_fee = min(raw_fee, notional * self.max_pct)
        return raw_fee


class TieredFeeModel(BaseFeeModel):
    """Tiered fee model based on rolling cumulative trading volume."""

    def __init__(
        self,
        tiers: list[tuple[float, float]] | None = None,
        min_fee: float = 0.0,
    ) -> None:
        # Default tiers: (volume_threshold, fee_rate)
        # up to 100k -> 0.10% (10 bps)
        # up to 1M   -> 0.05% (5 bps)
        # > 1M       -> 0.02% (2 bps)
        self.tiers = sorted(
            tiers or [(100_000.0, 0.0010), (1_000_000.0, 0.0005), (float("inf"), 0.0002)],
            key=lambda t: t[0],
        )
        self.min_fee = max(0.0, min_fee)
        self.cumulative_volume: float = 0.0

    def calculate_fee(
        self,
        symbol: str,
        side: OrderSide,
        fill_price: float,
        fill_quantity: float,
        is_maker: bool = False,
    ) -> float:
        notional = fill_price * abs(fill_quantity)
        current_tier_rate = self.tiers[-1][1]
        for threshold, rate in self.tiers:
            if self.cumulative_volume < threshold:
                current_tier_rate = rate
                break

        self.cumulative_volume += notional
        calculated = notional * current_tier_rate
        return max(self.min_fee, calculated)

    def reset_volume(self) -> None:
        """Reset cumulative volume counter."""
        self.cumulative_volume = 0.0
