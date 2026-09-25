# Talent-Flow Alpha — Design

## Thesis
Talent migration is a *leading* indicator of enterprise value. When the best people are
net-migrating toward a company faster than the market has priced in, that company is
undervalued — both as an investment and as a place to work. We build the signal, backtest
it against public history ("old big companies"), and — if it holds — apply it to
private/pre-IPO names.

## Two-layer architecture

```
  ┌─────────────────────────── Python (research / ETL) ───────────────────────────┐
  │  ingest adapters      →   normalize   →   aggregate   →   PANEL (the contract)  │
  │  H1B/LCA, WARN,           TalentEvent      company×month     data/panel/*.csv   │
  │  SEC (existing), levels                     features                            │
  └────────────────────────────────────────────────────────────────────────────────┘
                                        │  panel.csv + prices.csv
                                        ▼
  ┌─────────────────────────── C++ / CUDA (backtest engine) ──────────────────────┐
  │  load panel  →  compute momentum score  →  cross-sectional rank  →  long/short  │
  │                 (kernel)                    (kernel)                portfolio    │
  │  →  forward returns  →  metrics: Sharpe, IC, drawdown, decile spread            │
  │  Parameter sweeps (lookback, decay, top-k, horizon) run in parallel on GPU.     │
  └────────────────────────────────────────────────────────────────────────────────┘
```

**Why the split.** Ingest is messy, I/O-bound, iterated often → Python. The backtest is a
dense numerical kernel over a `companies × months × params` cube → C++/CUDA. They meet at a
flat, columnar **panel file** so either side can be rewritten independently.

## The panel contract  (`data/panel/talent_panel.csv`)
One row per (company, month). Point-in-time: a row dated `YYYY-MM` may only use data whose
`first_seen_at` ≤ end of that month (no look-ahead).

| column          | type    | meaning                                                        |
|-----------------|---------|----------------------------------------------------------------|
| `company_id`    | string  | stable id                                                      |
| `ticker`        | string  | public ticker (empty for private names)                        |
| `month`         | YYYY-MM | observation month (month-end as-of)                            |
| `hires`         | float   | total inflow events this month                                 |
| `senior_hires`  | float   | seniority-weighted inflow (title→seniority in [0,1])           |
| `mean_wage`     | float   | mean offered wage of inflow (comp proxy, from LCA)             |
| `departures`    | float   | outflow events (WARN layoffs + SEC departures)                 |
| `source_mask`   | int     | bitmask of contributing sources (audit)                        |

Prices live separately in `data/panel/prices.csv`: `date,ticker,close[,volume]` — reused
from the existing Python backtester's format so both engines share one price file.

## Signal (v0)
`momentum_t = EWMA_λ(net_flow) ,  net_flow = senior_hires − departures`
`score_t    = z_cross_section( momentum_t )  +  β · z( Δ momentum_t )   (acceleration)`
Cross-sectional z-score each month so the strategy is dollar-neutral-ish and comparable
across regimes. Long top decile, short bottom decile, rebalanced monthly, held `horizon`.

## Metrics
Sharpe (annualized), Information Coefficient (rank corr of score vs forward return),
max drawdown, decile spread (top-minus-bottom), turnover.

## CUDA / MLIR roadmap
1. **CPU reference kernel** (`kernels/cpu_reference.cpp`) — correctness oracle, runs on macOS.
2. **CUDA kernel** (`kernels/cuda_backtest.cu`) — same interface, built/run on a Brev GPU.
   Parallelism: one thread-block per (parameter-combo × month), warp-level reductions for
   the cross-sectional mean/std and rank.
3. **MLIR/LLVM-IR seam** — the score expression (`net_flow`, EWMA, z-score) is represented
   as a tiny expression IR (`engine/include/expr.hpp`). Today it is interpreted; the seam
   lets us later lower it to LLVM IR / PTX (or an MLIR `linalg`→`gpu` pipeline) so new
   strategy expressions compile to kernels instead of being hand-written.

## Data sources (all free)
- **H1B/LCA** (DOL disclosure) — inflow, title→seniority, wage. Historical to ~2008. Backbone.
- **WARN** — state layoff notices → departures.
- **SEC Item 5.02** (existing adapter) — exec arrivals/departures, high-signal.
- **levels.fyi** — comp momentum, best-effort later.
- LinkedIn/Revelio — ToS/legal risk; opportunistic enrichment only, never the backbone.

## Non-goals / honesty
Not investment advice; no live trading. H1B sees only visa-sponsored hires (biased toward
larger tech employers) — a real but partial view. Private-market application requires a
separate valuation source (secondary marks) and is phase 3.
```
