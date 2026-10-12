"""Simulated brokerage and exchange matching engine for QuantPulse."""

from datetime import datetime

from quantpulse.core.models import Bar, ExecutionReport, Order, Tick
from quantpulse.core.types import OrderSide, OrderStatus, OrderType
from quantpulse.execution.fees import BaseFeeModel, MakerTakerFee
from quantpulse.execution.slippage import BaseSlippageModel, FixedBasisPointSlippage


class SimulatedBroker:
    """Simulated exchange matching engine supporting Market, Limit, and Stop orders."""

    def __init__(
        self,
        fee_model: BaseFeeModel | None = None,
        slippage_model: BaseSlippageModel | None = None,
    ) -> None:
        self.fee_model: BaseFeeModel = fee_model or MakerTakerFee()
        self.slippage_model: BaseSlippageModel = slippage_model or FixedBasisPointSlippage()
        self.active_orders: dict[str, Order] = {}
        self.execution_history: list[ExecutionReport] = []
        self.current_prices: dict[str, float] = {}

    def submit_order(self, order: Order) -> ExecutionReport | None:
        """Submit an order to the matching engine.

        Market orders execute immediately against known prices if available.
        Limit and stop orders are queued into the active order book.
        """
        order.status = OrderStatus.SUBMITTED
        order.updated_at = datetime.utcnow()

        if order.order_type == OrderType.MARKET:
            current_price = self.current_prices.get(order.symbol)
            if current_price is not None:
                return self._fill_market_order(
                    order=order,
                    price=current_price,
                    volume=0.0,
                    timestamp=order.created_at,
                )

        self.active_orders[order.id] = order
        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending active order."""
        if order_id in self.active_orders:
            order = self.active_orders.pop(order_id)
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            return True
        return False

    def process_tick(self, tick: Tick) -> list[ExecutionReport]:
        """Update market price with tick and process matching against active orders."""
        self.current_prices[tick.symbol] = tick.last_price or tick.mid_price
        reports: list[ExecutionReport] = []

        active_order_ids = list(self.active_orders.keys())
        for order_id in active_order_ids:
            order = self.active_orders.get(order_id)
            if order is None or order.symbol != tick.symbol:
                continue

            report = self._evaluate_tick_match(order, tick)
            if report is not None:
                reports.append(report)
                if order.status == OrderStatus.FILLED:
                    self.active_orders.pop(order_id, None)

        return reports

    def process_bar(self, bar: Bar) -> list[ExecutionReport]:
        """Update market price with bar and process matching across active orders."""
        self.current_prices[bar.symbol] = bar.close
        reports: list[ExecutionReport] = []

        active_order_ids = list(self.active_orders.keys())
        for order_id in active_order_ids:
            order = self.active_orders.get(order_id)
            if order is None or order.symbol != bar.symbol:
                continue

            report = self._evaluate_bar_match(order, bar)
            if report is not None:
                reports.append(report)
                if order.status == OrderStatus.FILLED:
                    self.active_orders.pop(order_id, None)

        return reports

    def _evaluate_tick_match(self, order: Order, tick: Tick) -> ExecutionReport | None:
        """Evaluate order fill conditions against tick event."""
        if order.order_type == OrderType.MARKET:
            exec_price = tick.ask_price if order.side == OrderSide.BUY else tick.bid_price
            return self._fill_market_order(order, exec_price, tick.volume, tick.timestamp)

        if (
            order.order_type == OrderType.LIMIT
            and order.price is not None
            and (
                (order.side == OrderSide.BUY and tick.ask_price <= order.price)
                or (order.side == OrderSide.SELL and tick.bid_price >= order.price)
            )
        ):
            return self._fill_limit_order(order, order.price, tick.timestamp)

        if order.order_type == OrderType.STOP_LOSS and order.stop_price is not None:
            if order.side == OrderSide.BUY and tick.ask_price >= order.stop_price:
                return self._fill_market_order(order, tick.ask_price, tick.volume, tick.timestamp)
            elif order.side == OrderSide.SELL and tick.bid_price <= order.stop_price:
                return self._fill_market_order(order, tick.bid_price, tick.volume, tick.timestamp)

        return None

    def _evaluate_bar_match(self, order: Order, bar: Bar) -> ExecutionReport | None:
        """Evaluate order fill conditions across OHLC bar range."""
        if order.order_type == OrderType.MARKET:
            # Fills at bar open
            return self._fill_market_order(order, bar.open, bar.volume, bar.timestamp)

        if order.order_type == OrderType.LIMIT and order.price is not None:
            if order.side == OrderSide.BUY and bar.low <= order.price:
                # Fills at the better of bar open or limit price
                fill_price = min(bar.open, order.price) if bar.open <= order.price else order.price
                return self._fill_limit_order(order, fill_price, bar.timestamp)
            elif order.side == OrderSide.SELL and bar.high >= order.price:
                fill_price = max(bar.open, order.price) if bar.open >= order.price else order.price
                return self._fill_limit_order(order, fill_price, bar.timestamp)

        if order.order_type == OrderType.STOP_LOSS and order.stop_price is not None:
            if order.side == OrderSide.BUY and bar.high >= order.stop_price:
                trigger_price = max(bar.open, order.stop_price)
                return self._fill_market_order(order, trigger_price, bar.volume, bar.timestamp)
            elif order.side == OrderSide.SELL and bar.low <= order.stop_price:
                trigger_price = min(bar.open, order.stop_price)
                return self._fill_market_order(order, trigger_price, bar.volume, bar.timestamp)

        return None

    def _fill_market_order(
        self,
        order: Order,
        price: float,
        volume: float,
        timestamp: datetime,
    ) -> ExecutionReport:
        qty_to_fill = order.remaining_quantity
        fill_price, slippage_per_unit = self.slippage_model.calculate_fill_price(
            side=order.side,
            current_price=price,
            quantity=qty_to_fill,
            bar_volume=volume,
        )

        fee = self.fee_model.calculate_fee(
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            fill_quantity=qty_to_fill,
            is_maker=False,
        )

        order.filled_quantity += qty_to_fill
        order.avg_fill_price = fill_price
        order.status = OrderStatus.FILLED
        order.updated_at = timestamp

        report = ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            filled_quantity=qty_to_fill,
            fill_price=fill_price,
            fee=fee,
            slippage=slippage_per_unit * qty_to_fill,
            timestamp=timestamp,
            status=OrderStatus.FILLED,
            is_maker=False,
        )
        self.execution_history.append(report)
        return report

    def _fill_limit_order(
        self,
        order: Order,
        fill_price: float,
        timestamp: datetime,
    ) -> ExecutionReport:
        qty_to_fill = order.remaining_quantity
        fee = self.fee_model.calculate_fee(
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            fill_quantity=qty_to_fill,
            is_maker=True,
        )

        order.filled_quantity += qty_to_fill
        order.avg_fill_price = fill_price
        order.status = OrderStatus.FILLED
        order.updated_at = timestamp

        report = ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            filled_quantity=qty_to_fill,
            fill_price=fill_price,
            fee=fee,
            slippage=0.0,  # Resting limit orders achieve exact price or price improvement
            timestamp=timestamp,
            status=OrderStatus.FILLED,
            is_maker=True,
        )
        self.execution_history.append(report)
        return report
