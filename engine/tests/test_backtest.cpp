// Minimal self-checking test: build a panel where net_flow predicts next-month return,
// and assert the backtester recovers positive Sharpe and IC. No framework dependency.
#include "backtest.hpp"
#include <cstdio>
#include <random>

using namespace tf;

static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){ printf("FAIL: %s\n", msg); ++failures; } \
                              else printf("ok: %s\n", msg); } while(0)

int main() {
    // 40 companies, 60 months. net_flow ~ a persistent latent per company; next-month
    // return = 0.03 * latent + noise. Momentum should rank companies by latent.
    Panel p;
    p.n_companies = 40; p.n_months = 60;
    for (int c = 0; c < p.n_companies; ++c) {
        p.company_ids.push_back("C" + std::to_string(c));
        p.tickers.push_back("T" + std::to_string(c));
    }
    for (int t = 0; t < p.n_months; ++t) p.months.push_back("M" + std::to_string(t));
    p.net_flow.assign((size_t)p.n_companies*p.n_months, 0.f);
    p.ret.assign((size_t)p.n_companies*p.n_months, 0.f);

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.f, 0.02f);
    // Each company has a hiring TREND (slope). The scale-free signal detects acceleration
    // relative to a company's own baseline, so returns follow the slope, not the level.
    std::vector<float> slope(p.n_companies);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (int c = 0; c < p.n_companies; ++c) slope[c] = u(rng);

    for (int c = 0; c < p.n_companies; ++c)
        for (int t = 0; t < p.n_months; ++t) {
            p.net_flow[p.idx(c,t)] = 60.f + slope[c] * t + noise(rng) * 20.f; // trending flow
            p.ret[p.idx(c,t)]      = 0.04f * slope[c] + noise(rng);           // return follows slope
        }

    Params pr; pr.lookback = 12; pr.decay = 0.4f; pr.accel_beta = 0.f; pr.decile = 0.25f; pr.horizon = 1;
    Result r = backtest_one(p, pr);
    printf("n_periods=%d sharpe=%.3f ic=%.3f mean=%.4f\n", r.n_periods, r.sharpe, r.ic, r.mean_return);

    CHECK(r.n_periods > 30, "produced enough rebalance periods");
    CHECK(r.ic > 0.2, "information coefficient is clearly positive");
    CHECK(r.mean_return > 0.0, "long-short mean return is positive");
    CHECK(r.sharpe > 0.5, "annualized Sharpe is positive and material");

    // Sweep sanity: results returned for every combo.
    std::vector<Params> grid = { pr, Params{6,0.6f,0.5f,0.1f,3} };
    auto rs = backtest_sweep(p, grid);
    CHECK(rs.size() == grid.size(), "sweep returns one result per combo");

    printf(failures ? "\n%d FAILURES\n" : "\nALL PASSED\n", failures);
    return failures ? 1 : 0;
}
