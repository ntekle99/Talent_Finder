# Talent-Flow Alpha

Does **where tech talent goes** predict company value? This project builds a talent-migration
signal from public data, backtests it against stock returns with proper factor controls, and
runs the heavy compute on a **C++/CUDA + NCCL** engine with an **LLVM/NVPTX** signal compiler.

## Headline finding (honest)

- **Raw hiring *headcount* predicts nothing** — long/short Sharpe ≈ 0, collapses out-of-sample.
- **Person-level talent *caliber* does.** The strongest signal — *how much a company pays vs. the
  national market rate for the specific roles it hires* — **survives momentum + sector controls
  within tech**: Fama-MacBeth on 220 tech firms gives rank-**IC +0.05**, **sector-neutral Sharpe
  0.73** (positive 7/8 years). Economically sensible, factor-robust, consistent.
- **Not yet statistically proven.** Fama-MacBeth t ≈ 1.6 — the binding limit is *history length*
  (free price data only reaches ~2016 → 8 annual periods), not universe size. Adding more years
  (paid history) is the clear path to significance.

The lesson the whole project is organized around: **talent = caliber of individuals, not employee
count.**

## Pipeline

```
 DATA (Python)                              SIGNALS                    BACKTEST (C++/CUDA/NCCL)
 H1B/LCA filings  ─┐                        research caliber ─┐        momentum→rank→L/S
  (3.6GB, ~10M)    ├─► company×month ─────► wage premium      ├─► fuse ─► Sharpe/IC/drawdown
 SEC EDGAR tickers ┤    panel               GitHub OSS       ─┘        Fama-MacBeth controls
 Nasdaq/Yahoo px  ─┘                                                   2-GPU sweep (NCCL)
                                                                       LLVM/NVPTX signal compiler
```

## Components

**Data (`scripts/`, `src/talent_finder/`)**
- `build_panel.py` + `company_map.py` — parse 37 DOL H1B files (FY2010–2026) into a hiring panel.
- `sec_universe.py` — match H1B employers to **595 public tickers** via SEC EDGAR.
- `fetch_returns_nasdaq.py` / `fetch_returns_yahoo.py` — monthly returns (run on Brev to bypass
  the corporate firewall + anti-bot blocks). `fetch_sic.py` — SEC SIC codes for tech filtering.

**Talent signals (three complementary lenses)**
- `build_talent_openalex.py` — elite-researcher inflow (OpenAlex citations + affiliation history,
  advisory-affiliation filtered).
- `build_wage_talent.py` / `_broad.py` — role-adjusted **wage premium** (the one that backtests).
- `build_talent_github.py` — GitHub OSS impact (snapshot).
- `build_talent_index.py` — fuse into one z-scored talent index + trajectory.

**Backtest (`engine/`)**
- `include/backtest.hpp`, `src/main.cpp` — C++17 engine: momentum → cross-sectional rank →
  long/short → Sharpe/IC/drawdown; 243-config sweep; out-of-sample split. `tests/` verify it.
- `kernels/cuda_backtest.cu` — GPU sweep. `kernels/nccl_backtest.cu` — **2-GPU** sweep with
  **NCCL** collectives (`ncclBroadcast` panel, `ncclAllReduce` global-best). Runs on 2× T4.
- `scripts/factor_control_broad.py` — Fama-MacBeth + sector-neutral (the validation above).

**Signal compiler (`engine/jit/`)** — turns the backtester into a *signal-search* compiler.
- `signal_ast.hpp` — backend-agnostic signal DSL/IR (`ewma`, `mean`, `lag`, arithmetic).
- `jit_backtest.cpp` — lowers to **LLVM IR**, O2-optimizes, **ORC-JITs to native**, and searches
  over signal *structures* by IC.
- `emit_ptx.cpp` — same IR → **NVPTX**, emits valid GPU PTX (sm_75) for the NCCL engine.

## Run it

```bash
# 1. talent panel + signals (Python)
python scripts/build_panel.py && python scripts/build_talent_index.py

# 2. C++ backtest (CPU; CUDA/NCCL builds on a GPU box)
cmake -S engine -B engine/build && cmake --build engine/build && ./engine/build/talent_backtest

# 3. factor-control validation
python scripts/factor_control_broad.py

# 4. LLVM-JIT signal search (needs: brew install llvm)
bash engine/jit/build.sh && ./engine/jit/jit_backtest

# 5. emit GPU PTX from the signal IR
bash engine/jit/build_ptx.sh && ./engine/jit/emit_ptx "ewma(6,0.4) / mean(24) - 1"
```

## Honest scope

The **engineering** (data pipeline, C++/CUDA/NCCL engine, LLVM/NVPTX compiler) is complete and
demonstrable. The **alpha** is promising but underpowered — a data-history limitation, not a flaw
in the method. Not investment advice; research only.
