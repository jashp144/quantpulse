"""Base strategy definition for QuantPulse."""

from abc import ABC, abstractmethod
from collections import deque
from typing import Any

from quantpulse.core.models import Bar, ExecutionReport, Order, Tick
from quantpulse.core.types import OrderSide, OrderType


class BaseStrategy(ABC):
    """Abstract base strategy with event handling and order generation facilities."""

    def __init__(self, name: str = "BaseStrategy") -> None:
        self.name = name
        self.order_queue: deque[Order] = deque()
        self.portfolio_ref: Any = None  # Injected by backtesting engine

    def set_portfolio(self, portfolio: Any) -> None:
        """Attach active portfolio instance."""
        self.portfolio_ref = portfolio

    def on_start(self) -> None:
        """Lifecycle hook invoked before backtest execution starts."""

    @abstractmethod
    def on_bar(self, bar: Bar) -> None:
        """Invoked on each incoming historical or real-time bar."""

    def on_tick(self, tick: Tick) -> None:
        """Invoked on each incoming market tick."""

    def on_order_status(self, report: ExecutionReport) -> None:
        """Invoked when an order execution or fill occurs."""

    def on_end(self) -> None:
        """Lifecycle hook invoked after all bars have been processed."""

    def buy(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """Emit a BUY order."""
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=order_type,
            quantity=abs(quantity),
            price=price,
            stop_price=stop_price,
        )
        self.order_queue.append(order)
        return order

    def sell(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """Emit a SELL order."""
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=order_type,
            quantity=abs(quantity),
            price=price,
            stop_price=stop_price,
        )
        self.order_queue.append(order)
        return order

    def close_position(self, symbol: str) -> Order | None:
        """Close current open position for symbol."""
        if self.portfolio_ref is None:
            return None

        pos = self.portfolio_ref.positions.get(symbol)
        if pos is None or pos.is_flat:
            return None

        if pos.is_long:
            return self.sell(symbol=symbol, quantity=pos.quantity, order_type=OrderType.MARKET)
        else:
            return self.buy(symbol=symbol, quantity=abs(pos.quantity), order_type=OrderType.MARKET)

    def pop_orders(self) -> list[Order]:
        """Extract and clear all generated orders."""
        orders = list(self.order_queue)
        self.order_queue.clear()
        return orders
