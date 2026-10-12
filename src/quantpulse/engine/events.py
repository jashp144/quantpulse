"""Event-driven messaging definitions for QuantPulse backtesting engine."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from quantpulse.core.models import Bar, ExecutionReport, Order, Tick
from quantpulse.core.types import OrderSide


class EventType(str, Enum):
    """Core event types flowing through the backtest event queue."""

    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class Event(BaseModel):
    """Base event payload."""

    model_config = ConfigDict(frozen=True)

    event_type: EventType
    timestamp: datetime


class MarketEvent(Event):
    """Market price quote or bar arrival event."""

    event_type: EventType = Field(default=EventType.MARKET)
    symbol: str
    bar: Bar | None = None
    tick: Tick | None = None


class SignalEvent(Event):
    """Strategy trade signal indicating intended direction and sizing."""

    event_type: EventType = Field(default=EventType.SIGNAL)
    symbol: str
    side: OrderSide
    strength: float = Field(default=1.0, ge=0.0)
    suggested_price: float | None = None
    suggested_stop: float | None = None


class OrderEvent(Event):
    """Order placement event dispatched to broker."""

    event_type: EventType = Field(default=EventType.ORDER)
    order: Order


class FillEvent(Event):
    """Trade execution notification emitted by broker."""

    event_type: EventType = Field(default=EventType.FILL)
    report: ExecutionReport
