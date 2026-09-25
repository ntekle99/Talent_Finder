from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from .models import Company, PaperPosition, TalentEvent
from .scoring import rank_companies


@dataclass
class BacktestResult:
    returns: pd.DataFrame
    portfolio_returns: pd.DataFrame
    score_buckets: pd.DataFrame
    metrics: dict[str, float]


def positions(
    events: list[TalentEvent],
    as_of: date,
    top_n: int = 5,
    min_score: float = 0.0,
    universe: list[Company] | None = None,
    min_market_cap: float | None = None,
    max_per_sector: int | None = None,
) -> list[PaperPosition]:
    ranked = rank_companies(events, as_of)
    universe_by_ticker = {c.ticker: c for c in universe or [] if c.ticker}
    sector_counts: dict[str, int] = {}
    selected = []
    for item in ranked:
        ticker = item.company.ticker or item.company.id
        metadata = universe_by_ticker.get(ticker)
        market_cap = metadata.market_cap if metadata else item.company.market_cap
        sector = (metadata.sector if metadata else item.company.sector) or "Unknown"
        if item.score <= min_score:
            continue
        if min_market_cap is not None and (market_cap is None or market_cap < min_market_cap):
            continue
        if max_per_sector is not None and sector_counts.get(sector, 0) >= max_per_sector:
            continue
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        selected.append(item)
        if len(selected) >= top_n:
            break
    weight = 1 / len(selected) if selected else 0
    return [
        PaperPosition(
            as_of=as_of,
            ticker=item.company.ticker or item.company.id,
            direction=1,
            weight=weight,
            score=item.score,
            rationale="; ".join(item.explanations),
        )
        for item in selected
    ]


def _price_on_or_after(prices: pd.DataFrame, ticker: str, target: date) -> pd.Series | None:
    ticker_prices = prices[(prices.ticker == ticker) & (prices.date >= pd.Timestamp(target))].sort_values("date")
    if ticker_prices.empty:
        return None
    return ticker_prices.iloc[0]


def _forward_return(prices: pd.DataFrame, ticker: str, as_of: date, horizon_days: int) -> float | None:
    start = _price_on_or_after(prices, ticker, as_of)
    end = _price_on_or_after(prices, ticker, as_of + timedelta(days=horizon_days))
    if start is None or end is None:
        return None
    return float(end.close / start.close - 1)


def walk_forward(
    events: list[TalentEvent],
    prices: pd.DataFrame,
    rebalance_dates: list[date],
    horizon_days: int = 63,
    top_n: int = 5,
    transaction_cost_bps: float = 10,
    benchmark_ticker: str | None = None,
    min_score: float = 0.0,
    universe: list[Company] | None = None,
    min_market_cap: float | None = None,
    min_avg_dollar_volume: float | None = None,
    max_per_sector: int | None = None,
    score_bucket_count: int = 5,
) -> BacktestResult:
    """Prices require columns: date, ticker, close. Inputs must be point-in-time."""
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    tradable_tickers = _tradable_tickers(
        prices=prices,
        universe=universe,
        min_market_cap=min_market_cap,
        min_avg_dollar_volume=min_avg_dollar_volume,
    )
    rows = []
    for as_of in rebalance_dates:
        eligible_events = [e for e in events if (e.destination.ticker or e.destination.id) in tradable_tickers]
        held = positions(
            eligible_events,
            as_of,
            top_n,
            min_score=min_score,
            universe=universe,
            min_market_cap=min_market_cap,
            max_per_sector=max_per_sector,
        )
        benchmark_return = (
            _forward_return(prices, benchmark_ticker, as_of, horizon_days) if benchmark_ticker else None
        )
        for position in held:
            gross = _forward_return(prices, position.ticker, as_of, horizon_days)
            if gross is None:
                continue
            net = gross - transaction_cost_bps / 10000
            row = {
                "date": as_of,
                "ticker": position.ticker,
                "weight": position.weight,
                "score": position.score,
                "return": net,
            }
            sector = _sector_for_ticker(position.ticker, universe)
            if sector:
                row["sector"] = sector
            if benchmark_return is not None:
                row["benchmark_return"] = benchmark_return
                row["excess_return"] = net - benchmark_return
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return BacktestResult(frame, pd.DataFrame(), pd.DataFrame(), {"observations": 0.0})
    portfolio = frame.groupby("date").apply(lambda x: (x["return"] * x["weight"]).sum(), include_groups=False)
    portfolio_frame = portfolio.reset_index(name="portfolio_return")
    score_buckets = score_bucket_diagnostics(
        events=events,
        prices=prices,
        rebalance_dates=rebalance_dates,
        horizon_days=horizon_days,
        transaction_cost_bps=transaction_cost_bps,
        benchmark_ticker=benchmark_ticker,
        universe=universe,
        tradable_tickers=tradable_tickers,
        bucket_count=score_bucket_count,
    )
    metrics = {
        "observations": float(len(frame)),
        "rebalance_count": float(frame["date"].nunique()),
        "mean_forward_return": float(frame["return"].mean()),
        "hit_rate": float((frame["return"] > 0).mean()),
        "mean_portfolio_return": float(portfolio.mean()),
        "portfolio_hit_rate": float((portfolio > 0).mean()),
        "max_drawdown": float((portfolio.cumsum() - portfolio.cumsum().cummax()).min()),
        "avg_positions_per_rebalance": float(frame.groupby("date")["ticker"].nunique().mean()),
        "unique_tickers": float(frame["ticker"].nunique()),
        "eligible_tickers": float(len(tradable_tickers)),
        "event_ticker_coverage": float(
            len({e.destination.ticker or e.destination.id for e in events} & tradable_tickers)
            / max(1, len({e.destination.ticker or e.destination.id for e in events}))
        ),
    }
    concentration = frame.groupby("ticker")["weight"].sum()
    metrics["top_ticker_weight_share"] = float(concentration.max() / concentration.sum()) if concentration.sum() else 0.0
    if "sector" in frame:
        sector_exposure = frame.groupby("sector")["weight"].sum()
        metrics["unique_sectors"] = float(frame["sector"].nunique())
        metrics["top_sector_weight_share"] = (
            float(sector_exposure.max() / sector_exposure.sum()) if sector_exposure.sum() else 0.0
        )
    if "excess_return" in frame:
        excess = frame.groupby("date").apply(lambda x: (x["excess_return"] * x["weight"]).sum(), include_groups=False)
        portfolio_frame = portfolio_frame.merge(
            excess.reset_index(name="portfolio_excess_return"),
            on="date",
            how="left",
        )
        metrics["mean_excess_return"] = float(frame["excess_return"].mean())
        metrics["mean_portfolio_excess_return"] = float(excess.mean())
        metrics["excess_hit_rate"] = float((frame["excess_return"] > 0).mean())
    metrics.update(_score_bucket_metrics(score_buckets))
    return BacktestResult(frame, portfolio_frame, score_buckets, metrics)


def score_bucket_diagnostics(
    events: list[TalentEvent],
    prices: pd.DataFrame,
    rebalance_dates: list[date],
    horizon_days: int = 63,
    transaction_cost_bps: float = 10,
    benchmark_ticker: str | None = None,
    universe: list[Company] | None = None,
    tradable_tickers: set[str] | None = None,
    bucket_count: int = 5,
) -> pd.DataFrame:
    tradable = tradable_tickers or _tradable_tickers(prices, universe=universe)
    rows = []
    for as_of in rebalance_dates:
        eligible_events = [e for e in events if (e.destination.ticker or e.destination.id) in tradable]
        ranked = [
            score
            for score in rank_companies(eligible_events, as_of)
            if (score.company.ticker or score.company.id) in tradable and score.event_count > 0
        ]
        scored_rows = []
        benchmark_return = (
            _forward_return(prices, benchmark_ticker, as_of, horizon_days) if benchmark_ticker else None
        )
        for item in ranked:
            ticker = item.company.ticker or item.company.id
            forward_return = _forward_return(prices, ticker, as_of, horizon_days)
            if forward_return is None:
                continue
            net = forward_return - transaction_cost_bps / 10000
            row = {
                "date": as_of,
                "ticker": ticker,
                "score": item.score,
                "return": net,
            }
            if benchmark_return is not None:
                row["excess_return"] = net - benchmark_return
            scored_rows.append(row)
        if len(scored_rows) < 2:
            continue
        scored = pd.DataFrame(scored_rows).sort_values("score")
        buckets = min(bucket_count, len(scored))
        scored["bucket"] = pd.qcut(
            scored["score"].rank(method="first"),
            q=buckets,
            labels=range(1, buckets + 1),
        ).astype(int)
        rows.extend(scored.to_dict("records"))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    aggregations = {
        "ticker": "nunique",
        "score": "mean",
        "return": "mean",
    }
    if "excess_return" in frame:
        aggregations["excess_return"] = "mean"
    buckets = frame.groupby(["date", "bucket"], as_index=False).agg(aggregations)
    buckets = buckets.rename(
        columns={
            "ticker": "companies",
            "score": "mean_score",
            "return": "mean_return",
            "excess_return": "mean_excess_return",
        }
    )
    return buckets


def _score_bucket_metrics(score_buckets: pd.DataFrame) -> dict[str, float]:
    if score_buckets.empty:
        return {}
    by_bucket = score_buckets.groupby("bucket")["mean_return"].mean().sort_index()
    if len(by_bucket) < 2:
        return {}
    bottom = float(by_bucket.iloc[0])
    top = float(by_bucket.iloc[-1])
    monotonicity = by_bucket.reset_index()["bucket"].corr(by_bucket.reset_index()["mean_return"])
    metrics = {
        "score_bucket_bottom_mean_return": bottom,
        "score_bucket_top_mean_return": top,
        "score_bucket_top_bottom_spread": top - bottom,
        "score_bucket_return_monotonicity": float(monotonicity) if pd.notna(monotonicity) else 0.0,
    }
    if "mean_excess_return" in score_buckets:
        by_bucket_excess = score_buckets.groupby("bucket")["mean_excess_return"].mean().sort_index()
        metrics["score_bucket_top_bottom_excess_spread"] = float(by_bucket_excess.iloc[-1] - by_bucket_excess.iloc[0])
    return metrics


def _tradable_tickers(
    prices: pd.DataFrame,
    universe: list[Company] | None = None,
    min_market_cap: float | None = None,
    min_avg_dollar_volume: float | None = None,
) -> set[str]:
    tickers = set(prices["ticker"].astype(str).unique())
    if universe is not None:
        universe_tickers = {company.ticker for company in universe if company.ticker}
        tickers &= universe_tickers
        if min_market_cap is not None:
            tickers &= {
                company.ticker
                for company in universe
                if company.ticker and company.market_cap is not None and company.market_cap >= min_market_cap
            }
    if min_avg_dollar_volume is not None:
        if "volume" not in prices:
            raise ValueError("prices require a volume column when min_avg_dollar_volume is set")
        avg_dollar_volume = (prices["close"] * prices["volume"]).groupby(prices["ticker"]).mean()
        tickers &= set(avg_dollar_volume[avg_dollar_volume >= min_avg_dollar_volume].index.astype(str))
    return tickers


def _sector_for_ticker(ticker: str, universe: list[Company] | None) -> str | None:
    if not universe:
        return None
    for company in universe:
        if company.ticker == ticker:
            return company.sector
    return None
