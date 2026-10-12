"""Position management and FIFO lot tracking for QuantPulse."""

from collections import deque
from datetime import datetime

from quantpulse.core.models import Position, PositionLot
from quantpulse.core.types import OrderSide


class PositionManager:
    """Manages multi-asset positions with rigorous FIFO lot accounting."""

    def __init__(self) -> None:
        self.positions: dict[str, Position] = {}
        self.lot_queues: dict[str, deque[PositionLot]] = {}
        self.total_realized_pnl: float = 0.0

    def get_position(self, symbol: str) -> Position:
        """Get or create position object for a symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
            self.lot_queues[symbol] = deque()
        return self.positions[symbol]

    def add_fill(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        fill_price: float,
        timestamp: datetime,
        fee: float = 0.0,
    ) -> tuple[float, Position]:
        """Apply a trade fill using FIFO lot matching.

        Returns:
            tuple[float, Position]: (realized_pnl_from_this_fill, updated_position)
        """
        position = self.get_position(symbol)
        queue = self.lot_queues[symbol]
        fill_qty = abs(quantity)
        realized_pnl = 0.0

        if position.is_flat:
            # New position opened
            new_qty = fill_qty if side == OrderSide.BUY else -fill_qty
            lot = PositionLot(
                symbol=symbol,
                side=side,
                quantity=fill_qty,
                entry_price=fill_price,
                entry_time=timestamp,
            )
            queue.append(lot)
            position.quantity = new_qty
            position.avg_price = fill_price
            position.current_price = fill_price
            position.lots = list(queue)

        elif position.is_long:
            if side == OrderSide.BUY:
                # Adding to existing long position
                lot = PositionLot(
                    symbol=symbol,
                    side=side,
                    quantity=fill_qty,
                    entry_price=fill_price,
                    entry_time=timestamp,
                )
                queue.append(lot)
                total_qty = position.quantity + fill_qty
                # Weighted average entry price
                position.avg_price = (
                    (position.quantity * position.avg_price) + (fill_qty * fill_price)
                ) / total_qty
                position.quantity = total_qty
                position.lots = list(queue)
            else:
                # Selling: closing part, all, or flipping short
                remaining_sell = fill_qty
                while remaining_sell > 1e-9 and queue:
                    head_lot = queue[0]
                    match_qty = min(head_lot.quantity, remaining_sell)
                    lot_pnl = (fill_price - head_lot.entry_price) * match_qty
                    realized_pnl += lot_pnl
                    remaining_sell -= match_qty

                    if abs(head_lot.quantity - match_qty) <= 1e-9:
                        queue.popleft()
                    else:
                        updated_lot = PositionLot(
                            lot_id=head_lot.lot_id,
                            symbol=head_lot.symbol,
                            side=head_lot.side,
                            quantity=head_lot.quantity - match_qty,
                            entry_price=head_lot.entry_price,
                            entry_time=head_lot.entry_time,
                        )
                        queue[0] = updated_lot

                if remaining_sell > 1e-9:
                    # Flipped from Long to Short
                    short_lot = PositionLot(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=remaining_sell,
                        entry_price=fill_price,
                        entry_time=timestamp,
                    )
                    queue.append(short_lot)
                    position.quantity = -remaining_sell
                    position.avg_price = fill_price
                else:
                    position.quantity = sum(lot.quantity for lot in queue)
                    if queue:
                        total_cost = sum(lot.quantity * lot.entry_price for lot in queue)
                        position.avg_price = total_cost / position.quantity
                    else:
                        position.quantity = 0.0
                        position.avg_price = 0.0

                position.lots = list(queue)

        elif position.is_short:
            if side == OrderSide.SELL:
                # Adding to existing short position
                lot = PositionLot(
                    symbol=symbol,
                    side=side,
                    quantity=fill_qty,
                    entry_price=fill_price,
                    entry_time=timestamp,
                )
                queue.append(lot)
                current_abs_qty = abs(position.quantity)
                total_abs_qty = current_abs_qty + fill_qty
                position.avg_price = (
                    (current_abs_qty * position.avg_price) + (fill_qty * fill_price)
                ) / total_abs_qty
                position.quantity = -total_abs_qty
                position.lots = list(queue)
            else:
                # Buying to cover: closing part, all, or flipping long
                remaining_buy = fill_qty
                while remaining_buy > 1e-9 and queue:
                    head_lot = queue[0]
                    match_qty = min(head_lot.quantity, remaining_buy)
                    lot_pnl = (head_lot.entry_price - fill_price) * match_qty
                    realized_pnl += lot_pnl
                    remaining_buy -= match_qty

                    if abs(head_lot.quantity - match_qty) <= 1e-9:
                        queue.popleft()
                    else:
                        updated_lot = PositionLot(
                            lot_id=head_lot.lot_id,
                            symbol=head_lot.symbol,
                            side=head_lot.side,
                            quantity=head_lot.quantity - match_qty,
                            entry_price=head_lot.entry_price,
                            entry_time=head_lot.entry_time,
                        )
                        queue[0] = updated_lot

                if remaining_buy > 1e-9:
                    # Flipped from Short to Long
                    long_lot = PositionLot(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=remaining_buy,
                        entry_price=fill_price,
                        entry_time=timestamp,
                    )
                    queue.append(long_lot)
                    position.quantity = remaining_buy
                    position.avg_price = fill_price
                else:
                    remaining_short_qty = sum(lot.quantity for lot in queue)
                    position.quantity = -remaining_short_qty
                    if queue:
                        total_cost = sum(lot.quantity * lot.entry_price for lot in queue)
                        position.avg_price = total_cost / remaining_short_qty
                    else:
                        position.quantity = 0.0
                        position.avg_price = 0.0

                position.lots = list(queue)

        # Net fee deduction from trade realized PnL
        net_realized_pnl = realized_pnl - fee
        position.realized_pnl += net_realized_pnl
        self.total_realized_pnl += net_realized_pnl

        # Update unrealized PnL at fill price
        position.current_price = fill_price
        self._update_unrealized_pnl(position)

        return net_realized_pnl, position

    def mark_to_market(self, symbol: str, current_price: float) -> float:
        """Mark open position to market with latest price."""
        position = self.get_position(symbol)
        position.current_price = current_price
        return self._update_unrealized_pnl(position)

    def _update_unrealized_pnl(self, position: Position) -> float:
        """Calculate and update unrealized PnL based on current market price."""
        if position.is_flat:
            position.unrealized_pnl = 0.0
            return 0.0

        if position.is_long:
            position.unrealized_pnl = (
                position.current_price - position.avg_price
            ) * position.quantity
        else:
            position.unrealized_pnl = (position.avg_price - position.current_price) * abs(
                position.quantity
            )

        return position.unrealized_pnl
