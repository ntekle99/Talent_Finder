// Backtest core (CPU reference). The CUDA kernel mirrors this exact math.
#pragma once
#include "panel.hpp"
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace tf {

struct Params {
    int   lookback   = 12;    // months of history for EWMA momentum
    float decay      = 0.5f;  // EWMA weight on the most recent month (0..1]
    float accel_beta = 0.5f;  // weight on acceleration (delta-momentum) z-score
    float decile     = 0.2f;  // long top `decile`, short bottom `decile`
    int   horizon    = 1;     // forward-return holding period in months
};

struct Result {
    double sharpe = 0, ic = 0, max_drawdown = 0, decile_spread = 0,
           turnover = 0, mean_return = 0;
    int    n_periods = 0;
    Params params;
};

namespace detail {

// EWMA of net_flow up to and including month t, most-recent weight = decay.
inline float ewma_momentum(const Panel& p, int c, int t, const Params& pr) {
    float num = 0.f, den = 0.f, w = 1.f;
    int start = std::max(0, t - pr.lookback + 1);
    for (int s = t; s >= start; --s) {
        float v = p.net_flow[p.idx(c, s)];
        if (!is_nan(v)) { num += w * v; den += w; }
        w *= (1.f - pr.decay);
    }
    return den > 0 ? num / den : kNaN;
}

// Trailing mean of net_flow over [t-win, t-1] (the company's own recent baseline).
inline float trailing_mean(const Panel& p, int c, int t, int win) {
    double s = 0; int n = 0;
    for (int k = std::max(0, t - win); k < t; ++k) {
        float v = p.net_flow[p.idx(c, k)];
        if (!is_nan(v)) { s += v; ++n; }
    }
    return n > 0 ? (float)(s / n) : kNaN;
}

// SCALE-FREE, DE-SEASONALIZED company signal: this year's hiring pace vs. the SAME window
// one year earlier (YoY). Two fixes in one: (a) scale-free -> a big firm at its normal pace
// scores ~0 (kills size-dominance); (b) year-over-year -> the Q1 visa-cap spike that hits
// every firm cancels out, so we measure real acceleration, not the calendar.
constexpr int kSeason = 12;  // months in the seasonal cycle
inline float company_signal(const Panel& p, int c, int t, const Params& pr) {
    if (t < kSeason) return kNaN;
    float recent = ewma_momentum(p, c, t, pr);             // recent hiring window
    float prior  = ewma_momentum(p, c, t - kSeason, pr);   // same window a year ago
    if (is_nan(recent) || is_nan(prior)) return kNaN;
    float denom = std::max(std::fabs(prior), 5.0f);        // floor: tiny counts are noisy
    return (recent - prior) / denom;                       // YoY growth vs. self
}

// Cross-sectional z-score in place over valid entries.
inline void zscore(std::vector<float>& x) {
    double m = 0; int n = 0;
    for (float v : x) if (!is_nan(v)) { m += v; ++n; }
    if (n < 2) return;
    m /= n;
    double var = 0;
    for (float v : x) if (!is_nan(v)) var += (v - m) * (v - m);
    double sd = std::sqrt(var / n);
    if (sd < 1e-9) { for (float& v : x) if (!is_nan(v)) v = 0.f; return; }
    for (float& v : x) if (!is_nan(v)) v = (float)((v - m) / sd);
}

// Forward compounded return over [t+1, t+horizon].
inline float forward_return(const Panel& p, int c, int t, int horizon) {
    if (t + horizon >= p.n_months) return kNaN;
    float g = 1.f;
    for (int s = t + 1; s <= t + horizon; ++s) {
        float r = p.ret[p.idx(c, s)];
        if (is_nan(r)) return kNaN;
        g *= (1.f + r);
    }
    return g - 1.f;
}

// Spearman-style rank correlation over paired valid entries.
inline double rank_corr(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<int> idx;
    for (int i = 0; i < (int)a.size(); ++i)
        if (!is_nan(a[i]) && !is_nan(b[i])) idx.push_back(i);
    int n = (int)idx.size();
    if (n < 3) return 0.0;
    auto ranks = [&](auto get) {
        std::vector<int> order(idx.begin(), idx.end());
        std::sort(order.begin(), order.end(),
                  [&](int p, int q){ return get(p) < get(q); });
        std::vector<double> rk(a.size(), 0.0);
        for (int r = 0; r < n; ++r) rk[order[r]] = r;
        return rk;
    };
    auto ra = ranks([&](int i){ return a[i]; });
    auto rb = ranks([&](int i){ return b[i]; });
    double ma = (n - 1) / 2.0, cov = 0, va = 0, vb = 0;
    for (int i : idx) {
        cov += (ra[i]-ma)*(rb[i]-ma); va += (ra[i]-ma)*(ra[i]-ma); vb += (rb[i]-ma)*(rb[i]-ma);
    }
    return (va > 0 && vb > 0) ? cov / std::sqrt(va*vb) : 0.0;
}

} // namespace detail

// One parameter combo end-to-end. This is the unit the GPU parallelizes across.
// Rebalances only in [t_lo, t_hi) (t_hi<0 means "to the end"); the signal still reads
// history before t_lo, which is past data, not leakage. Used for in-sample/OOS splits.
inline Result backtest_one(const Panel& p, const Params& pr, int t_lo = 0, int t_hi = -1) {
    using namespace detail;
    Result R; R.params = pr;
    std::vector<double> period_ret;
    std::vector<double> ics;
    std::vector<int> prev_long;   // for turnover

    int hi = (t_hi < 0) ? p.n_months : std::min(t_hi, p.n_months);
    for (int t = std::max(0, t_lo); t < hi; ++t) {
        std::vector<float> mom(p.n_companies, kNaN), accel(p.n_companies, kNaN);
        for (int c = 0; c < p.n_companies; ++c) {
            mom[c] = company_signal(p, c, t, pr);
            float mprev = (t > 0) ? company_signal(p, c, t - 1, pr) : kNaN;
            if (!is_nan(mom[c]) && !is_nan(mprev)) accel[c] = mom[c] - mprev;
        }
        std::vector<float> zmom = mom, zacc = accel;
        zscore(zmom); zscore(zacc);
        std::vector<float> score(p.n_companies, kNaN);
        for (int c = 0; c < p.n_companies; ++c) {
            if (is_nan(zmom[c])) continue;
            score[c] = zmom[c] + pr.accel_beta * (is_nan(zacc[c]) ? 0.f : zacc[c]);
        }
        // forward returns for this rebalance
        std::vector<float> fwd(p.n_companies, kNaN);
        for (int c = 0; c < p.n_companies; ++c) fwd[c] = forward_return(p, c, t, pr.horizon);

        // eligible = valid score AND valid forward return
        std::vector<int> elig;
        for (int c = 0; c < p.n_companies; ++c)
            if (!is_nan(score[c]) && !is_nan(fwd[c])) elig.push_back(c);
        if ((int)elig.size() < 5) { prev_long.clear(); continue; }

        std::sort(elig.begin(), elig.end(),
                  [&](int a, int b){ return score[a] > score[b]; });
        int k = std::max(1, (int)std::floor(pr.decile * elig.size()));
        double long_ret = 0, short_ret = 0;
        std::vector<int> cur_long;
        for (int i = 0; i < k; ++i)          { long_ret  += fwd[elig[i]]; cur_long.push_back(elig[i]); }
        for (int i = 0; i < k; ++i)            short_ret += fwd[elig[elig.size()-1-i]];
        long_ret /= k; short_ret /= k;
        period_ret.push_back(long_ret - short_ret);
        ics.push_back(rank_corr(score, fwd));

        // turnover = fraction of long book that changed
        if (!prev_long.empty()) {
            int overlap = 0;
            for (int c : cur_long)
                if (std::find(prev_long.begin(), prev_long.end(), c) != prev_long.end()) ++overlap;
            R.turnover += 1.0 - (double)overlap / k;
        }
        prev_long = cur_long;
    }

    R.n_periods = (int)period_ret.size();
    if (R.n_periods == 0) return R;
    double mean = std::accumulate(period_ret.begin(), period_ret.end(), 0.0) / R.n_periods;
    double var = 0; for (double r : period_ret) var += (r-mean)*(r-mean);
    double sd = std::sqrt(var / std::max(1, R.n_periods - 1));
    R.mean_return   = mean;
    R.sharpe        = sd > 1e-12 ? mean / sd * std::sqrt(12.0 / pr.horizon) : 0.0; // annualized
    R.ic            = std::accumulate(ics.begin(), ics.end(), 0.0) / ics.size();
    R.decile_spread = mean; // long-short spread == mean period return by construction
    R.turnover      = R.n_periods > 1 ? R.turnover / (R.n_periods - 1) : 0.0;

    double cum = 0, peak = 0, mdd = 0;
    for (double r : period_ret) { cum += r; peak = std::max(peak, cum); mdd = std::min(mdd, cum - peak); }
    R.max_drawdown = mdd;
    return R;
}

// Parameter sweep. CPU loops; the CUDA build launches one thread-block per combo.
std::vector<Result> backtest_sweep(const Panel& p, const std::vector<Params>& grid);

} // namespace tf
