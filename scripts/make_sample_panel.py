#!/usr/bin/env python3
"""Generate a SYNTHETIC but plausible talent panel so the C++ engine runs end-to-end.

Each company has a hidden "trajectory" that drifts over time (rising stars vs. decliners).
Seniority-weighted hiring tracks the trajectory (with noise); departures move inversely;
monthly stock return responds to *changes* in trajectory plus market + idiosyncratic noise.
That makes talent momentum genuinely predictive of forward returns, but noisily -- the
realistic case the backtester must handle. Replace this with real H1B/WARN output later.

Outputs:
  data/panel/talent_panel.csv   company_id,ticker,month,hires,senior_hires,mean_wage,departures,source_mask
  data/panel/returns.csv        company_id,month,ret
"""
from __future__ import annotations
import csv, math, random
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "panel"

# (company_id, ticker) -- real large-cap tech names for a recognizable demo universe.
COMPANIES = [
    ("nvidia","NVDA"),("apple","AAPL"),("microsoft","MSFT"),("google","GOOGL"),
    ("meta","META"),("amazon","AMZN"),("tesla","TSLA"),("netflix","NFLX"),
    ("amd","AMD"),("intel","INTC"),("oracle","ORCL"),("salesforce","CRM"),
    ("adobe","ADBE"),("cisco","CSCO"),("ibm","IBM"),("qualcomm","QCOM"),
    ("broadcom","AVGO"),("paypal","PYPL"),("uber","UBER"),("snowflake","SNOW"),
    ("palantir","PLTR"),("servicenow","NOW"),("shopify","SHOP"),("datadog","DDOG"),
    ("crowdstrike","CRWD"),("zoom","ZM"),("twilio","TWLO"),("docusign","DOCU"),
    ("roku","ROKU"),("pinterest","PINS"),
]

N_MONTHS = 48
SEED = 7


def months(n: int) -> list[str]:
    out, y, m = [], 2021, 1
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1; y += 1
    return out


def main() -> None:
    random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    ms = months(N_MONTHS)

    # Hidden trajectory per company: a slow sinusoid + drift, distinct phase/slope each.
    traj = {}
    for cid, _ in COMPANIES:
        phase = random.uniform(0, 2 * math.pi)
        slope = random.uniform(-0.02, 0.03)          # secular rise/decline
        amp = random.uniform(0.3, 1.0)
        base = random.uniform(-0.5, 0.5)
        traj[cid] = [base + slope * t + amp * math.sin(0.25 * t + phase) for t in range(N_MONTHS)]

    panel_rows, ret_rows = [], []
    for cid, ticker in COMPANIES:
        tr = traj[cid]
        for t, mon in enumerate(ms):
            level = tr[t]
            # seniority-weighted hiring rises with trajectory level; departures fall with it
            senior = max(0.0, 20 + 25 * level + random.gauss(0, 4))
            hires = senior * random.uniform(1.8, 2.6)
            departures = max(0.0, 12 - 10 * level + random.gauss(0, 3))
            wage = 150_000 + 40_000 * level + random.gauss(0, 8_000)
            panel_rows.append((cid, ticker, mon, round(hires, 1), round(senior, 1),
                               int(wage), round(departures, 1), 1))
            # return responds to trajectory CHANGE (momentum) + market + noise
            dtraj = tr[t] - tr[t - 1] if t > 0 else 0.0
            market = random.gauss(0.008, 0.03)        # common factor
            ret = 0.06 * dtraj + market + random.gauss(0, 0.04)
            ret_rows.append((cid, mon, round(ret, 5)))

    with (OUT / "talent_panel.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["company_id","ticker","month","hires","senior_hires","mean_wage","departures","source_mask"])
        w.writerows(panel_rows)
    with (OUT / "returns.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["company_id","month","ret"])
        w.writerows(ret_rows)

    print(f"wrote {len(panel_rows)} panel rows, {len(ret_rows)} return rows "
          f"for {len(COMPANIES)} companies x {N_MONTHS} months -> {OUT}")


if __name__ == "__main__":
    main()
