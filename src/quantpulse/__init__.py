"""QuantPulse: Institutional-Grade Event-Driven Backtesting & Risk Engine."""

__version__ = "1.0.0"
__author__ = "Jash Patel"

from quantpulse.core.models import (
    Bar,
    ExecutionReport,
    Order,
    PortfolioState,
    Position,
    PositionLot,
    Tick,
)
from quantpulse.core.types import (
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from quantpulse.engine.backtester import BacktestEngine
from quantpulse.engine.event_bus import EventQueue
from quantpulse.engine.events import (
    Event,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from quantpulse.engine.results import BacktestResult
from quantpulse.engine.tear_sheet import TearSheetGenerator
from quantpulse.execution.broker import SimulatedBroker
from quantpulse.execution.fees import (
    BaseFeeModel,
    FlatFeeModel,
    MakerTakerFee,
    TieredFeeModel,
)
from quantpulse.execution.slippage import (
    AlmgrenChrissSlippage,
    BaseSlippageModel,
    FixedBasisPointSlippage,
)
from quantpulse.portfolio.portfolio import Portfolio
from quantpulse.portfolio.position import PositionManager
from quantpulse.risk.metrics import RiskMetrics
from quantpulse.risk.var import ValueAtRisk
from quantpulse.strategies.base import BaseStrategy
from quantpulse.strategies.ma_crossover import MovingAverageCrossoverStrategy
from quantpulse.strategies.mean_reversion import MeanReversionStrategy
from quantpulse.strategies.volatility_breakout import VolatilityBreakoutStrategy

__all__ = [
    "AlmgrenChrissSlippage",
    "BacktestEngine",
    "BacktestResult",
    "Bar",
    "BaseFeeModel",
    "BaseSlippageModel",
    "BaseStrategy",
    "Event",
    "EventQueue",
    "EventType",
    "ExecutionReport",
    "FillEvent",
    "FixedBasisPointSlippage",
    "FlatFeeModel",
    "MakerTakerFee",
    "MarketEvent",
    "MeanReversionStrategy",
    "MovingAverageCrossoverStrategy",
    "Order",
    "OrderEvent",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Portfolio",
    "PortfolioState",
    "Position",
    "PositionLot",
    "PositionManager",
    "RiskMetrics",
    "SignalEvent",
    "SimulatedBroker",
    "TearSheetGenerator",
    "Tick",
    "TieredFeeModel",
    "TimeInForce",
    "ValueAtRisk",
    "VolatilityBreakoutStrategy",
]
