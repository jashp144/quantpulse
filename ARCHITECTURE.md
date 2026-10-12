# Architecture Specification: QuantPulse

**Document Version**: 1.0.0  
**Author**: Jash Patel ([@jashp144](https://github.com/jashp144))  
**Target Repository**: `quantpulse`  
**Status**: APPROVED ARCHITECTURE  

---

## 1. System Topology & Design Principles

`quantpulse` is an institutional-grade, event-driven quantitative trading framework designed around five foundational architectural principles:

1. **Zero Lookahead Bias**: Information at bar $t$ cannot observe, extrapolate, or execute on future timestamps $t+k$. Market data, order placement, and execution are strictly segregated in a chronological event queue.
2. **Realistic Market Microstructure**: Unlike naive vectorized frameworks that assume infinite liquidity and zero execution drag, simulated trades experience bid-ask spread crossing, order queue resting, volume participation limits, and temporary market impact.
3. **Rigorous FIFO Lot Accounting**: Multi-lot position tracking strictly pairs exits with the earliest open tax lots, calculating realized PnL and updating average entry cost with exact cash ledger reconciliation.
4. **Deterministic Reproducibility**: Given an identical initial seed and chronological data feed, backtest execution produces float-identical equity curves, drawdowns, and performance tear-sheets.
5. **Decoupled Strategy Abstraction**: Trading strategies operate solely via standard event hooks (`on_bar`, `on_tick`, `on_order_status`) and emit abstract orders without direct manipulation of portfolio balances or broker state.

---

## 2. Core Subsystems & Components

```
+--------------------------------------------------------------------------------+
|                              BacktestEngine Core                               |
+--------------------------------------------------------------------------------+
          |                                                       |
          v                                                       v
+-----------------------+                               +------------------------+
|  Ingestion & Storage  |                               |    EventQueue (FIFO)   |
| (Pandas / Bar Models) |                               | (Market/Signal/Orders) |
+-----------------------+                               +------------------------+
          |                                                       |
          +-----------------------+-------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------------------+
|                               Event Loop Cycle                                 |
|                                                                                |
|  1. Market Data Tick/Bar Arrival ---> MarketEvent                              |
|  2. Broker Process Bar (Limit/Stop Matching) ---> ExecutionReport              |
|  3. Portfolio Process Fills (Cash & FIFO Lots)                                 |
|  4. Strategy Event Handling (on_bar)                                           |
|  5. Strategy Order Generation (buy/sell) ---> OrderEvent                       |
|  6. Broker Submit Orders (Market Orders fill with Slippage & Fees)             |
|  7. Portfolio Process New Fills                                                |
|  8. Mark-to-Market Valuation & Historical Snapshot                             |
+--------------------------------------------------------------------------------+
          |                                                       |
          v                                                       v
+-----------------------+                               +------------------------+
|  Position Management  |                               |     Risk & Metrics     |
|   (FIFO Cost Basis)   |                               |  (Sharpe/Sortino/VaR)  |
+-----------------------+                               +------------------------+
```

### 2.1 Core Subsystem (`quantpulse.core`)
- Encapsulates domain value objects: `Tick`, `Bar`, `Order`, `ExecutionReport`, `Position`, `PositionLot`, and `PortfolioState`.
- Immutability enforced via frozen Pydantic schemas for ticks, bars, and execution reports to prevent state mutation across modules.

### 2.2 Execution Subsystem (`quantpulse.execution`)
- `SimulatedBroker`: Coordinates resting limit orders, trailing stops, and market order matching.
- `BaseSlippageModel`: Computes execution price deviations:
  - `FixedBasisPointSlippage`: Fixed basis point penalty.
  - `AlmgrenChrissSlippage`: Square-root law volume participation and temporary price impact.
- `BaseFeeModel`: Computes exchange transaction costs:
  - `MakerTakerFee`: Maker (passive) vs taker (aggressive) differential pricing.
  - `FlatFeeModel`: Per-share commissions with minimum and maximum notional caps.
  - `TieredFeeModel`: 30-day rolling volume tiers.

### 2.3 Portfolio Accounting Subsystem (`quantpulse.portfolio`)
- `PositionManager`: Tracks asset tax lots using double-ended queues (`deque[PositionLot]`). Supports long entries, long additions, partial FIFO liquidations, short sales, covers, and long-to-short flips.
- `Portfolio`: Tracks cash reserves, Reg-T initial margin (50%), maintenance margin (25%), gross market value, leverage, and marks positions to market at each timestamp.

### 2.4 Risk & Analytics Subsystem (`quantpulse.risk`)
- `RiskMetrics`: Annualized Sharpe, Sortino (downside semideviation), Calmar, CAGR, Annualized Volatility, High-Water Mark, Maximum Drawdown, Drawdown Duration, Win Rate, Profit Factor, Alpha, Beta, and Information Ratio.
- `ValueAtRisk`: Parametric Gaussian VaR/CVaR and Empirical Historical Simulation VaR/CVaR at 95% and 99% confidence levels.

### 2.5 Strategy Subsystem (`quantpulse.strategies`)
- `BaseStrategy`: Abstract base class providing lifecycle hooks and order helpers (`buy`, `sell`, `close_position`).
- `MovingAverageCrossoverStrategy`: Dual EMA/SMA crossover.
- `MeanReversionStrategy`: Bollinger Bands, Z-score, and RSI oversold/overbought mean reversion.
- `VolatilityBreakoutStrategy`: Donchian channel breakout with ATR trailing stop-loss.

---

## 3. Mathematical Equations & Formulations

### 3.1 Market Impact (Almgren-Chriss Framework)
$$\Delta P = \operatorname{sign}(Q) \cdot P_{\text{mid}} \cdot \left( \eta \cdot \left(\frac{|Q|}{\max(V_{\text{bar}}, |Q|)}\right)^\alpha + \frac{\text{half\_spread}}{P_{\text{mid}}} \right)$$
- $\eta$: Liquidity coefficient (default: 0.10)
- $\alpha$: Impact exponent (default: 0.50, square-root law)
- $V_{\text{bar}}$: Historical bar volume

### 3.2 Sharpe & Sortino Ratios
- **Sharpe Ratio**:
  $$\text{Sharpe} = \frac{\operatorname{mean}(R_t - r_f)}{\operatorname{std}(R_t)} \times \sqrt{252}$$
- **Sortino Ratio**:
  $$\text{Sortino} = \frac{\operatorname{mean}(R_t - r_f)}{\sqrt{\frac{1}{N} \sum_{t=1}^N \min(0, R_t - T)^2}} \times \sqrt{252}$$

### 3.3 Maximum Drawdown (MDD)
$$HWM_t = \max_{s \le t} Equity_s, \quad DD_t = \frac{Equity_t - HWM_t}{HWM_t}, \quad MDD = \min_{t} DD_t$$

### 3.4 Value at Risk (VaR) & Conditional VaR (Expected Shortfall)
- **Parametric VaR (95%)**:
  $$\text{VaR}_{0.95} = - (\mu - z_{0.95} \cdot \sigma)$$
- **Parametric CVaR (Expected Shortfall)**:
  $$\text{CVaR}_{0.95} = - \left(\mu - \sigma \cdot \frac{\phi(z_{0.95})}{1 - 0.95}\right)$$
- **Historical Simulation VaR**:
  $$\text{VaR}_{0.95}^{\text{hist}} = - \text{Percentile}(R, 5.0)$$
- **Historical Simulation CVaR**:
  $$\text{CVaR}_{0.95}^{\text{hist}} = - \frac{1}{|R_{\text{tail}}|} \sum_{r \in R_{\text{tail}}} r, \quad \text{where } R_{\text{tail}} = \{r \in R \mid r \le -\text{VaR}_{0.95}^{\text{hist}}\}$$

---

## 4. Architectural Decision Records (ADR)

### ADR-001: Event-Driven Chronological Loop vs. Vectorized Operations
- **Context**: Vectorized backtesting (`df['signal'].shift(1) * df['returns']`) is fast but structurally vulnerable to lookahead bias, cannot model resting limit orders, and fails on dynamic portfolio rebalancing under cash constraints.
- **Decision**: Implement a discrete, chronological FIFO event loop where time flows strictly forward.
- **Consequences**: Realistic execution timing, proper order rejection on margin exhaustion, and complete transparency into intermediate states.

### ADR-002: FIFO Tax Lot Queue for Cost Basis & Realized PnL
- **Context**: Moving-average cost basis fails to accurately track holding period returns and tax-lot allocations on partial position liquidations.
- **Decision**: Maintain double-ended queues (`collections.deque[PositionLot]`) per asset. Exits dequeue the oldest open lots first.
- **Consequences**: Exact trade accounting adhering to US IRS FIFO standards; deterministic realized PnL matching analytical trade ledgers.

### ADR-003: Almgren-Chriss Square-Root Market Impact
- **Context**: Fixed slippage models distort performance by under-penalizing large institutional block orders and over-penalizing small retail trades.
- **Decision**: Adopt the Almgren-Chriss square-root law where market impact scales with $\sqrt{Q / V_{\text{bar}}}$.
- **Consequences**: Institutional realism preventing strategies from simulating unachievable fills in illiquid assets.

### ADR-004: Unified Pydantic v2 Domain Models
- **Context**: Untyped dictionaries or loose tuples cause subtle type coercion bugs and silent field omission.
- **Decision**: All domain entities inherit from Pydantic `BaseModel` with strict type annotations and boundary validations (`Field(gt=0.0)`).
- **Consequences**: Runtime validation catching corrupted price feeds or negative order quantities before execution.

---

## 5. Security & Isolation Boundary
- Pure computational Python codebase with zero external network socket listeners or dynamic code evaluation (`eval`/`exec`).
- Memory safe: Pure standard library and NumPy/SciPy/Pandas in-memory matrix calculations.
