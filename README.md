# Talent-Flow Alpha

Does where tech talent moves predict company value? This project builds a talent-migration signal from public data and backtests it against stock returns with factor controls. The heavy compute runs on a C++/CUDA engine with NCCL across GPUs, and signals are compiled through LLVM/NVPTX.

## What we found

- Raw hiring headcount predicts nothing. The long/short Sharpe is about 0 and collapses out of sample.
- Person-level talent caliber does. The strongest signal is how much a company pays relative to the national market rate for the specific roles it hires, and it holds up after momentum and sector controls within tech. Fama-MacBeth on 220 tech firms gives a rank IC of +0.05 and a sector-neutral Sharpe of 0.73, positive in 7 of 8 years.
- It is not statistically proven yet. The Fama-MacBeth t is about 1.6. The limit is history length, not universe size: free price data only reaches back to about 2016, which leaves 8 annual periods. Paid history going further back is what would push it to significance.

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

### Data (`scripts/`, `src/talent_finder/`)
- `build_panel.py` + `company_map.py`: parse 37 DOL H1B files (FY2010-2026) into a hiring panel.
- `sec_universe.py`: match H1B employers to 595 public tickers via SEC EDGAR.
- `fetch_returns_nasdaq.py` / `fetch_returns_yahoo.py`: monthly returns (run on Brev to get past the corporate firewall and anti-bot blocks). `fetch_sic.py`: SEC SIC codes for tech filtering.

### Talent signals (three of them)
- `build_talent_openalex.py`: elite-researcher inflow (OpenAlex citations and affiliation history, filtered for advisory affiliations).
- `build_wage_talent.py` / `_broad.py`: role-adjusted wage premium, the signal that backtests.
- `build_talent_github.py`: GitHub OSS impact (snapshot).
- `build_talent_index.py`: fuse these into one z-scored talent index plus trajectory.

### Backtest (`engine/`)
- `include/backtest.hpp`, `src/main.cpp`: a C++17 engine that does momentum, cross-sectional rank, long/short, and Sharpe/IC/drawdown over a 243-config sweep with an out-of-sample split. `tests/` verify it.
- `kernels/cuda_backtest.cu`: GPU sweep. `kernels/nccl_backtest.cu`: 2-GPU sweep using NCCL collectives (`ncclBroadcast` for the panel, `ncclAllReduce` for the global best). Runs on 2× T4.
- `scripts/factor_control_broad.py`: Fama-MacBeth and sector-neutral checks (the validation above).

### Signal compiler (`engine/jit/`)
Turns the backtester into a signal search.
- `signal_ast.hpp`: backend-agnostic signal DSL/IR (`ewma`, `mean`, `lag`, arithmetic).
- `jit_backtest.cpp`: lowers to LLVM IR, optimizes at O2, ORC-JITs to native, and searches over signal structures by IC.
- `emit_ptx.cpp`: takes the same IR to NVPTX and emits GPU PTX (sm_75) for the NCCL engine.

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

## Scope

The engineering is finished and runs end to end: the data pipeline, the C++/CUDA/NCCL engine, and the LLVM/NVPTX compiler. The alpha is promising but underpowered, and that comes from how little history the free data covers, not from the method. Research only, not investment advice.
