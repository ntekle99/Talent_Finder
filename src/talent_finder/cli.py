import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from .backtest import walk_forward
from .ingest.sec import company_item_502_events
from .models import Company, EventType, Evidence, Person, TalentEvent
from .prices import load_prices_csv, rebalance_dates_from_prices
from .scoring import rank_companies
from .store import read_jsonl, write_jsonl
from .universe import ingest_sec_universe, load_universe_csv


def demo() -> None:
    now = datetime.now(UTC)
    company = Company(id="demo", name="Demo Systems", ticker="DEMO")
    event = TalentEvent(
        id="demo-event",
        person=Person(id="p1", name="Example Researcher"),
        destination=company,
        event_type=EventType.ARRIVAL,
        event_at=now,
        first_seen_at=now,
        evidence=[Evidence(source_url="https://example.com", source_name="demo", first_seen_at=now)],
        impact=0.9,
        scarcity=0.8,
        fit=0.9,
        seniority=0.7,
    )
    for score in rank_companies([event], now.date()):
        print(score.model_dump_json(indent=2))


def sec_ingest(args: argparse.Namespace) -> None:
    company = Company(
        id=args.company_id or args.ticker or args.cik,
        name=args.company_name,
        ticker=args.ticker,
        cik=args.cik,
        sector=args.sector,
    )
    events = company_item_502_events(company, limit=args.limit)
    if args.output:
        write_jsonl(events, args.output)
    for event in events:
        print(event.model_dump_json())


def _write_errors_csv(errors: list[dict[str, str]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["company_id", "ticker", "cik", "error"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)


def sec_ingest_universe(args: argparse.Namespace) -> None:
    companies = load_universe_csv(args.universe)
    result = ingest_sec_universe(
        companies,
        limit_per_company=args.limit_per_company,
        continue_on_error=not args.fail_fast,
    )
    if args.output:
        write_jsonl(result.events, args.output)
    if args.errors_output:
        _write_errors_csv(result.errors, args.errors_output)
    summary = {
        "companies": len(companies),
        "events": len(result.events),
        "errors": len(result.errors),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_backtest(args: argparse.Namespace) -> None:
    events = read_jsonl(args.events)
    prices = load_prices_csv(args.prices)
    universe = load_universe_csv(args.universe) if args.universe else None
    start = datetime.fromisoformat(args.start).date() if args.start else None
    end = datetime.fromisoformat(args.end).date() if args.end else None
    rebalance_dates = rebalance_dates_from_prices(prices, start=start, end=end, frequency=args.frequency)
    result = walk_forward(
        events=events,
        prices=prices,
        rebalance_dates=rebalance_dates,
        horizon_days=args.horizon_days,
        top_n=args.top_n,
        transaction_cost_bps=args.transaction_cost_bps,
        benchmark_ticker=args.benchmark_ticker,
        min_score=args.min_score,
        universe=universe,
        min_market_cap=args.min_market_cap,
        min_avg_dollar_volume=args.min_avg_dollar_volume,
        max_per_sector=args.max_per_sector,
        score_bucket_count=args.score_buckets,
    )
    if args.returns_output:
        result.returns.to_csv(args.returns_output, index=False)
    if args.portfolio_output:
        result.portfolio_returns.to_csv(args.portfolio_output, index=False)
    if args.buckets_output:
        result.score_buckets.to_csv(args.buckets_output, index=False)
    print(json.dumps(result.metrics, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo")

    sec_parser = subparsers.add_parser("sec-ingest")
    sec_parser.add_argument("--cik", required=True)
    sec_parser.add_argument("--company-name", required=True)
    sec_parser.add_argument("--ticker")
    sec_parser.add_argument("--company-id")
    sec_parser.add_argument("--sector")
    sec_parser.add_argument("--limit", type=int, default=25)
    sec_parser.add_argument("--output")

    sec_universe_parser = subparsers.add_parser("sec-ingest-universe")
    sec_universe_parser.add_argument("--universe", required=True)
    sec_universe_parser.add_argument("--limit-per-company", type=int, default=25)
    sec_universe_parser.add_argument("--output", required=True)
    sec_universe_parser.add_argument("--errors-output")
    sec_universe_parser.add_argument("--fail-fast", action="store_true")

    backtest_parser = subparsers.add_parser("backtest")
    backtest_parser.add_argument("--events", required=True)
    backtest_parser.add_argument("--prices", required=True)
    backtest_parser.add_argument("--benchmark-ticker")
    backtest_parser.add_argument("--universe")
    backtest_parser.add_argument("--returns-output")
    backtest_parser.add_argument("--portfolio-output")
    backtest_parser.add_argument("--buckets-output")
    backtest_parser.add_argument("--start")
    backtest_parser.add_argument("--end")
    backtest_parser.add_argument("--frequency", default="MS")
    backtest_parser.add_argument("--horizon-days", type=int, default=63)
    backtest_parser.add_argument("--top-n", type=int, default=5)
    backtest_parser.add_argument("--transaction-cost-bps", type=float, default=10)
    backtest_parser.add_argument("--min-score", type=float, default=0.0)
    backtest_parser.add_argument("--min-market-cap", type=float)
    backtest_parser.add_argument("--min-avg-dollar-volume", type=float)
    backtest_parser.add_argument("--max-per-sector", type=int)
    backtest_parser.add_argument("--score-buckets", type=int, default=5)

    args = parser.parse_args()
    if args.command == "demo":
        demo()
    if args.command == "sec-ingest":
        sec_ingest(args)
    if args.command == "sec-ingest-universe":
        sec_ingest_universe(args)
    if args.command == "backtest":
        run_backtest(args)


if __name__ == "__main__":
    main()
