# Talent Signal Trading MVP

This project researches whether observable talent movement contains incremental information about liquid U.S. equities over 1–6 month horizons. It produces explainable company rankings and paper positions; it does not place trades or provide investment advice.

## What counts as talent?

The MVP treats talent as role-relevant human capital. Evidence is strongest when a person has measurable prior impact, scarce skills, meaningful seniority or decision authority, and a credible fit with the destination company. Signals include executive appointments/departures, curated CSV events, research affiliations, open-source activity, and patents. Public sources do not provide complete employee mobility, so broad workforce coverage is intentionally a future paid-data adapter.

Every event has a source URL, first-seen timestamp, event timestamp, confidence, and extraction version. Backtests only expose events after `first_seen_at`, preventing look-ahead leakage.

The current scoring layer blends supplied event fields with explainable heuristics:

- role importance, such as CEO/CFO/CTO, president, VP, head-of, and director titles
- scarce skills, such as AI, machine learning, GPUs, semiconductors, clinical work, cybersecurity, and infrastructure
- company/role fit based on sector and role language
- source reliability, with SEC filings weighted above user CSVs and weaker public adapters
- surprise, such as external arrivals or departures

These are still research weights, not a trained alpha model. Before live trading, calibrate them on point-in-time historical data and compare performance against benchmark, sector, size, momentum, and liquidity controls.

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest
python -m talent_finder.cli demo
```

Ingest recent SEC Item 5.02 executive movement events:

```bash
talent-signal sec-ingest \
  --cik 0000320193 \
  --company-name "Apple Inc." \
  --ticker AAPL \
  --sector Technology \
  --limit 25 \
  --output data/events/apple-sec.jsonl
```

The command prints extracted events as JSON and optionally writes them to JSONL for downstream scoring/backtests.

For a multi-company universe, create a CSV with required columns `cik,name` and optional columns `ticker,company_id,sector,market_cap`:

```csv
cik,name,ticker,sector,market_cap
0000320193,Apple Inc.,AAPL,Technology,3000000000000
0000789019,Microsoft Corporation,MSFT,Technology,3500000000000
```

Then batch-ingest SEC events:

```bash
talent-signal sec-ingest-universe \
  --universe data/universe.csv \
  --limit-per-company 25 \
  --output data/events/sec-universe.jsonl \
  --errors-output data/events/sec-universe-errors.csv
```

By default, the batch command keeps going when one company fails and records errors separately. Use `--fail-fast` when debugging a specific ingestion issue.

Run a walk-forward backtest from saved events and a price CSV:

```bash
talent-signal backtest \
  --events data/events/apple-sec.jsonl \
  --prices data/prices.csv \
  --universe data/universe.csv \
  --benchmark-ticker SPY \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --frequency MS \
  --horizon-days 63 \
  --top-n 5 \
  --min-market-cap 1000000000 \
  --min-avg-dollar-volume 10000000 \
  --max-per-sector 2 \
  --returns-output data/backtests/apple-sec-returns.csv \
  --portfolio-output data/backtests/apple-sec-portfolio.csv \
  --buckets-output data/backtests/apple-sec-score-buckets.csv
```

Price CSVs require `date,ticker,close`; include `volume` to enable `--min-avg-dollar-volume`. The command prints summary metrics as JSON and can write per-position forward returns, per-rebalance portfolio returns, and score-bucket diagnostics to CSV.

Optional dashboard:

```bash
python -m streamlit run src/talent_finder/app.py
```

## Backtesting

`walk_forward` ranks companies at each rebalance date, buys the top positive scores, uses the next available price on or after the rebalance/horizon dates, and can report benchmark-relative excess returns with `benchmark_ticker`.

Backtests can now enforce basic tradability constraints:

- `--universe` restricts selection to a defined ticker universe and adds sector/market-cap metadata
- `--min-market-cap` removes small/untradable companies when universe market caps are provided
- `--min-avg-dollar-volume` removes illiquid names when price CSVs include `volume`
- `--max-per-sector` limits concentration in one sector

Metrics include portfolio return, excess return when a benchmark is provided, average positions per rebalance, eligible ticker count, event ticker coverage, unique ticker count, top ticker weight share, and sector concentration when universe sectors are available.

Score-bucket diagnostics test whether the score rank itself has information. For each rebalance date, the scored universe is split into quantile buckets from low to high score. The backtest reports top-bucket return, bottom-bucket return, top-minus-bottom spread, and a simple monotonicity statistic. A promising signal should usually have a positive top-minus-bottom spread and avoid depending on one lucky rebalance.

The backtester is still intentionally conservative and incomplete. A trading-grade version needs survivorship-safe prices, delisting returns, liquidity filters, sector-neutral portfolio construction, train/test splits, and paper-trading monitoring.

## Data policy

Use legally accessible public sources and retain citations. SEC submissions are the primary live adapter; OpenAlex, GitHub, USPTO, and user CSV inputs are enrichment adapters. Do not scrape access-controlled services. Configure a descriptive SEC `User-Agent` with `TALENT_SEC_USER_AGENT`.

The SEC adapter can now list 8-K Item 5.02 filings and conservatively extract obvious appointment/resignation language from filing text into normalized `TalentEvent` objects. Ambiguous filings should go to manual review instead of becoming strong signals automatically.

## CSV event format

Required columns: `person_id, person_name, company_id, company_name, event_type, event_at, first_seen_at, source_url, confidence`. Optional columns include `role, source_company_id, source_company_name, impact, scarcity, fit, seniority, sector`.
