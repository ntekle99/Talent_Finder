from datetime import date
from pathlib import Path

import pandas as pd

REQUIRED_PRICE_COLUMNS = {"date", "ticker", "close"}


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    prices = pd.read_csv(path)
    missing = REQUIRED_PRICE_COLUMNS - set(prices.columns)
    if missing:
        raise ValueError(f"prices CSV missing required column(s): {', '.join(sorted(missing))}")
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str)
    prices["close"] = prices["close"].astype(float)
    if "volume" in prices:
        prices["volume"] = prices["volume"].astype(float)
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def rebalance_dates_from_prices(
    prices: pd.DataFrame,
    start: date | None = None,
    end: date | None = None,
    frequency: str = "MS",
) -> list[date]:
    price_dates = pd.Series(pd.to_datetime(prices["date"]).dt.normalize().unique()).sort_values()
    if start:
        price_dates = price_dates[price_dates >= pd.Timestamp(start)]
    if end:
        price_dates = price_dates[price_dates <= pd.Timestamp(end)]
    if price_dates.empty:
        return []

    targets = pd.date_range(price_dates.iloc[0], price_dates.iloc[-1], freq=frequency)
    dates: list[date] = []
    for target in targets:
        available = price_dates[price_dates >= target]
        if not available.empty:
            dates.append(available.iloc[0].date())
    return sorted(set(dates))
