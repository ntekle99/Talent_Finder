from datetime import UTC, date, datetime

import pandas as pd

from talent_finder.backtest import walk_forward
from talent_finder.cli import run_backtest, sec_ingest_universe
from talent_finder.features import inferred_quality
from talent_finder.ingest import sec
from talent_finder.ingest.sec import (
    company_item_502_events,
    events_from_item_502_text,
    filing_document_url,
)
from talent_finder.models import Company, EventType, Evidence, Person, TalentEvent
from talent_finder.prices import load_prices_csv, rebalance_dates_from_prices
from talent_finder.scoring import score_company, score_event
from talent_finder.store import read_jsonl, write_jsonl
from talent_finder.universe import ingest_sec_universe, load_universe_csv


def event(
    kind=EventType.ARRIVAL,
    first_seen="2024-01-01T00:00:00+00:00",
    ticker="ACME",
    company_id="c1",
    sector="Technology",
    role="Chief Technology Officer",
    impact=1,
    scarcity=1,
    fit=1,
    seniority=1,
    confidence=1,
):
    ts = datetime.fromisoformat(first_seen)
    return TalentEvent(
        id="e1",
        person=Person(id="p1", name="Ada", skills=["machine learning"]),
        destination=Company(id=company_id, name=f"{ticker} Co", ticker=ticker, sector=sector),
        event_type=kind,
        event_at=ts,
        first_seen_at=ts,
        evidence=[Evidence(source_url="https://example.com", source_name="sec", first_seen_at=ts, excerpt="x")],
        role=role,
        impact=impact,
        scarcity=scarcity,
        fit=fit,
        seniority=seniority,
        confidence=confidence,
    )


def test_arrival_positive_and_departure_negative():
    arrival = score_event(event(), date(2024, 1, 2))
    departure = score_event(event(EventType.DEPARTURE), date(2024, 1, 2))
    assert arrival > 0
    assert departure < 0


def test_future_event_is_not_visible():
    future = event(first_seen="2024-02-01T00:00:00+00:00")
    assert score_event(future, date(2024, 1, 31)) == 0


def test_score_exposes_factors():
    result = score_company([event()], date(2024, 1, 2))
    assert result.score > 0
    assert set(result.factors) == {
        "impact",
        "scarcity",
        "fit",
        "seniority",
        "confidence",
        "source_reliability",
        "surprise",
    }


def test_quality_can_be_inferred_from_role_and_evidence():
    quality = inferred_quality(
        TalentEvent(
            id="e2",
            person=Person(id="p2", name="Grace", skills=["AI infrastructure"]),
            destination=Company(id="c2", name="ComputeCo", ticker="CPU", sector="Technology"),
            event_type=EventType.ARRIVAL,
            event_at=datetime(2024, 1, 1, tzinfo=UTC),
            first_seen_at=datetime(2024, 1, 1, tzinfo=UTC),
            evidence=[
                Evidence(
                    source_url="https://example.com",
                    source_name="sec",
                    first_seen_at=datetime(2024, 1, 1, tzinfo=UTC),
                    excerpt="appointed as CTO",
                )
            ],
            role="Chief Technology Officer, AI Infrastructure",
        )
    )
    assert quality["seniority"] >= 0.9
    assert quality["scarcity"] >= 0.9
    assert quality["source_reliability"] == 1.0


def test_backtest_is_deterministic():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "ticker": ["ACME", "ACME"],
            "close": [100, 110],
        }
    )
    result = walk_forward([event()], prices, [date(2024, 1, 1)], horizon_days=91)
    assert result.metrics["observations"] == 1
    assert result.metrics["mean_forward_return"] < 0.1


def test_backtest_reports_benchmark_excess_return():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-04-01", "2024-01-01", "2024-04-01"]),
            "ticker": ["ACME", "ACME", "SPY", "SPY"],
            "close": [100, 110, 100, 105],
        }
    )
    result = walk_forward([event()], prices, [date(2024, 1, 1)], horizon_days=91, benchmark_ticker="SPY")
    assert result.metrics["mean_excess_return"] > 0
    assert "portfolio_excess_return" in result.portfolio_returns


def test_sec_item_502_text_extracts_executive_events():
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    company = Company(id="c1", name="Acme", ticker="ACME", cik="1", sector="Technology")
    events = events_from_item_502_text(
        "On January 1, 2024, Ada Lovelace was appointed as Chief Technology Officer.",
        company,
        ts,
        ts,
        "https://www.sec.gov/ixviewer/doc/action",
    )
    assert len(events) == 1
    assert events[0].event_type == EventType.ARRIVAL
    assert events[0].person.name == "Ada Lovelace"


def test_sec_filing_document_url_uses_archive_shape():
    assert (
        filing_document_url("0000320193", "0000320193-24-000001", "aapl-20240101.htm")
        == "https://www.sec.gov/Archives/edgar/data/320193/000032019324000001/aapl-20240101.htm"
    )


def test_company_item_502_events_fetches_and_extracts(monkeypatch):
    class FakeResponse:
        def __init__(self, payload=None, text=""):
            self.payload = payload
            self.text = text

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, headers, timeout):
        assert headers["User-Agent"]
        assert timeout == 30
        if "submissions" in url:
            return FakeResponse(
                {
                    "filings": {
                        "recent": {
                            "form": ["8-K"],
                            "items": ["5.02"],
                            "accessionNumber": ["0000000001-24-000001"],
                            "filingDate": ["2024-01-01"],
                            "primaryDocument": ["form8k.htm"],
                        }
                    }
                }
            )
        return FakeResponse(text="Ada Lovelace was appointed as Chief Technology Officer.")

    monkeypatch.setattr(sec.requests, "get", fake_get)
    events = company_item_502_events(
        Company(id="c1", name="Acme", ticker="ACME", cik="1", sector="Technology"),
        limit=1,
        first_seen_at=datetime(2024, 1, 2, tzinfo=UTC),
    )
    assert len(events) == 1
    assert events[0].evidence[0].source_name == "sec"
    assert events[0].event_at.date() == date(2024, 1, 1)


def test_write_jsonl_persists_events(tmp_path):
    output = tmp_path / "events.jsonl"
    write_jsonl([event()], output)
    text = output.read_text(encoding="utf-8")
    assert '"person"' in text
    assert text.endswith("\n")


def test_jsonl_round_trip(tmp_path):
    output = tmp_path / "events.jsonl"
    write_jsonl([event()], output)
    loaded = read_jsonl(output)
    assert len(loaded) == 1
    assert loaded[0].person.name == "Ada"


def test_load_prices_csv_validates_required_columns(tmp_path):
    prices_path = tmp_path / "prices.csv"
    prices_path.write_text("date,ticker,close\n2024-01-01,ACME,100\n", encoding="utf-8")
    prices = load_prices_csv(prices_path)
    assert list(prices.columns) == ["date", "ticker", "close"]
    assert prices.iloc[0].close == 100.0


def test_rebalance_dates_from_prices_uses_available_trade_dates():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-02"]),
            "ticker": ["ACME", "ACME", "ACME"],
            "close": [100, 101, 102],
        }
    )
    assert rebalance_dates_from_prices(prices, frequency="MS") == [date(2024, 2, 2)]


def test_cli_backtest_runs_from_files(tmp_path, capsys):
    events_path = tmp_path / "events.jsonl"
    prices_path = tmp_path / "prices.csv"
    returns_path = tmp_path / "returns.csv"
    portfolio_path = tmp_path / "portfolio.csv"
    buckets_path = tmp_path / "buckets.csv"
    write_jsonl([event()], events_path)
    prices_path.write_text(
        "date,ticker,close\n"
        "2024-01-01,ACME,100\n"
        "2024-04-01,ACME,110\n"
        "2024-01-01,SPY,100\n"
        "2024-04-01,SPY,105\n",
        encoding="utf-8",
    )

    class Args:
        events = str(events_path)
        prices = str(prices_path)
        benchmark_ticker = "SPY"
        returns_output = str(returns_path)
        portfolio_output = str(portfolio_path)
        buckets_output = str(buckets_path)
        start = "2024-01-01"
        end = "2024-01-01"
        frequency = "MS"
        horizon_days = 91
        top_n = 5
        transaction_cost_bps = 10
        min_score = 0.0
        universe = None
        min_market_cap = None
        min_avg_dollar_volume = None
        max_per_sector = None
        score_buckets = 5

    run_backtest(Args())
    captured = capsys.readouterr()
    assert "mean_excess_return" in captured.out
    assert returns_path.exists()
    assert portfolio_path.exists()
    assert buckets_path.exists()


def test_score_bucket_diagnostics_show_top_bottom_spread():
    events = [
        event(ticker="LOW", company_id="low", role="Associate", impact=0.1, scarcity=0.1, fit=0.1, seniority=0.1),
        event(ticker="MID", company_id="mid", role="Director", impact=0.4, scarcity=0.4, fit=0.4, seniority=0.4),
        event(ticker="HIGH", company_id="high", role="Chief Technology Officer", impact=1, scarcity=1, fit=1, seniority=1),
    ]
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-04-01", "2024-01-01", "2024-04-01", "2024-01-01", "2024-04-01"]
            ),
            "ticker": ["LOW", "LOW", "MID", "MID", "HIGH", "HIGH"],
            "close": [100, 95, 100, 105, 100, 120],
        }
    )
    result = walk_forward(events, prices, [date(2024, 1, 1)], horizon_days=91, top_n=3, score_bucket_count=3)
    assert len(result.score_buckets) == 3
    assert result.metrics["score_bucket_top_bottom_spread"] > 0
    assert result.metrics["score_bucket_return_monotonicity"] > 0


def test_backtest_filters_by_universe_market_cap_and_volume():
    rich = event(ticker="RICH", company_id="rich", sector="Technology")
    tiny = event(ticker="TINY", company_id="tiny", sector="Technology")
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-04-01", "2024-01-01", "2024-04-01"]
            ),
            "ticker": ["RICH", "RICH", "TINY", "TINY"],
            "close": [100, 110, 10, 20],
            "volume": [1_000_000, 1_000_000, 1_000, 1_000],
        }
    )
    universe = [
        Company(id="rich", name="Rich Co", ticker="RICH", sector="Technology", market_cap=10_000_000_000),
        Company(id="tiny", name="Tiny Co", ticker="TINY", sector="Technology", market_cap=100_000_000),
    ]
    result = walk_forward(
        [rich, tiny],
        prices,
        [date(2024, 1, 1)],
        horizon_days=91,
        top_n=5,
        universe=universe,
        min_market_cap=1_000_000_000,
        min_avg_dollar_volume=10_000_000,
    )
    assert set(result.returns["ticker"]) == {"RICH"}
    assert result.metrics["eligible_tickers"] == 1


def test_backtest_caps_positions_per_sector():
    tech_one = event(ticker="ONE", company_id="one", sector="Technology")
    tech_two = event(ticker="TWO", company_id="two", sector="Technology")
    health = event(ticker="HLTH", company_id="hlth", sector="Healthcare")
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-04-01", "2024-01-01", "2024-04-01", "2024-01-01", "2024-04-01"]
            ),
            "ticker": ["ONE", "ONE", "TWO", "TWO", "HLTH", "HLTH"],
            "close": [100, 110, 100, 111, 100, 108],
        }
    )
    universe = [
        Company(id="one", name="One", ticker="ONE", sector="Technology", market_cap=2_000_000_000),
        Company(id="two", name="Two", ticker="TWO", sector="Technology", market_cap=2_000_000_000),
        Company(id="hlth", name="Health", ticker="HLTH", sector="Healthcare", market_cap=2_000_000_000),
    ]
    result = walk_forward(
        [tech_one, tech_two, health],
        prices,
        [date(2024, 1, 1)],
        horizon_days=91,
        top_n=3,
        universe=universe,
        max_per_sector=1,
    )
    assert len(result.returns) == 2
    assert result.metrics["unique_sectors"] == 2
    assert result.metrics["top_sector_weight_share"] == 0.5


def test_load_universe_csv(tmp_path):
    universe_path = tmp_path / "universe.csv"
    universe_path.write_text(
        "cik,name,ticker,sector,market_cap\n"
        "0000000001,Acme,ACME,Technology,1000000000\n"
        "0000000002,Bravo,BRVO,Healthcare,\n",
        encoding="utf-8",
    )
    companies = load_universe_csv(universe_path)
    assert len(companies) == 2
    assert companies[0].id == "ACME"
    assert companies[0].market_cap == 1000000000
    assert companies[1].market_cap is None


def test_ingest_sec_universe_keeps_partial_failures():
    companies = [
        Company(id="ok", name="Okay", ticker="OK", cik="1"),
        Company(id="bad", name="Bad", ticker="BAD", cik="2"),
    ]

    def fetcher(company, limit):
        assert limit == 3
        if company.id == "bad":
            raise RuntimeError("boom")
        return [event(ticker=company.ticker)]

    result = ingest_sec_universe(companies, limit_per_company=3, fetcher=fetcher)
    assert len(result.events) == 1
    assert result.errors == [{"company_id": "bad", "ticker": "BAD", "cik": "2", "error": "boom"}]


def test_cli_sec_ingest_universe_writes_events_and_errors(tmp_path, monkeypatch, capsys):
    universe_path = tmp_path / "universe.csv"
    events_path = tmp_path / "events.jsonl"
    errors_path = tmp_path / "errors.csv"
    universe_path.write_text(
        "cik,name,ticker,sector\n"
        "0000000001,Acme,ACME,Technology\n"
        "0000000002,BadCo,BAD,Technology\n",
        encoding="utf-8",
    )

    def fake_ingest(companies, limit_per_company, continue_on_error):
        assert len(companies) == 2
        assert limit_per_company == 4
        assert continue_on_error
        return type(
            "Result",
            (),
            {
                "events": [event(ticker="ACME")],
                "errors": [{"company_id": "BAD", "ticker": "BAD", "cik": "0000000002", "error": "boom"}],
            },
        )()

    monkeypatch.setattr("talent_finder.cli.ingest_sec_universe", fake_ingest)

    class Args:
        universe = str(universe_path)
        limit_per_company = 4
        output = str(events_path)
        errors_output = str(errors_path)
        fail_fast = False

    sec_ingest_universe(Args())
    captured = capsys.readouterr()
    assert '"companies": 2' in captured.out
    assert '"events": 1' in captured.out
    assert read_jsonl(events_path)[0].destination.ticker == "ACME"
    assert "boom" in errors_path.read_text(encoding="utf-8")
