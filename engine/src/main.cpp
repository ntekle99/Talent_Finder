// talent-backtest: load panel, sweep strategy params, report best config + live ranking.
#include "backtest.hpp"
#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace tf;

static std::vector<Params> build_grid() {
    std::vector<Params> g;
    for (int lb : {6, 12, 24})
        for (float dc : {0.2f, 0.4f, 0.6f})
            for (float b : {0.0f, 0.5f, 1.0f})
                for (float dec : {0.1f, 0.2f, 0.3f})
                    for (int h : {1, 3, 6})
                        g.push_back(Params{lb, dc, b, dec, h});
    return g;
}

// Rank companies as-of the last month using a chosen config: the "where is talent going" list.
static void print_live_ranking(const Panel& p, const Params& pr, int top) {
    using namespace detail;
    int t = p.n_months - 1;
    std::vector<float> mom(p.n_companies, kNaN), acc(p.n_companies, kNaN);
    for (int c = 0; c < p.n_companies; ++c) {
        mom[c] = company_signal(p, c, t, pr);
        float mp = t > 0 ? company_signal(p, c, t-1, pr) : kNaN;
        if (!is_nan(mom[c]) && !is_nan(mp)) acc[c] = mom[c] - mp;
    }
    auto zm = mom, za = acc; zscore(zm); zscore(za);
    std::vector<std::pair<float,int>> ranked;
    for (int c = 0; c < p.n_companies; ++c)
        if (!is_nan(zm[c]))
            ranked.push_back({zm[c] + pr.accel_beta*(is_nan(za[c])?0.f:za[c]), c});
    std::sort(ranked.begin(), ranked.end(), std::greater<>());
    printf("\nTalent momentum ranking as of %s (top %d):\n", p.months[t].c_str(), top);
    printf("  %-4s %-14s %-10s %s\n", "rank", "company", "ticker", "score");
    for (int i = 0; i < std::min((int)ranked.size(), top); ++i) {
        int c = ranked[i].second;
        printf("  %-4d %-14s %-10s %+.3f\n", i+1,
               p.company_ids[c].c_str(),
               p.tickers[c].empty()?"(private)":p.tickers[c].c_str(),
               ranked[i].first);
    }
}

int main(int argc, char** argv) {
    std::string panel = "data/panel/talent_panel.csv";
    std::string returns = "data/panel/returns.csv";
    std::string split;   // e.g. "2022-01": fit on months before, test on/after
    int top = 15;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--panel") && i+1 < argc) panel = argv[++i];
        else if (!strcmp(argv[i], "--returns") && i+1 < argc) returns = argv[++i];
        else if (!strcmp(argv[i], "--top") && i+1 < argc) top = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--split") && i+1 < argc) split = argv[++i];
    }

    Panel p;
    try { p = load_panel(panel, returns); }
    catch (const std::exception& e) { fprintf(stderr, "load error: %s\n", e.what()); return 1; }
    printf("panel: %d companies x %d months (%s .. %s)\n",
           p.n_companies, p.n_months,
           p.months.empty()?"?":p.months.front().c_str(),
           p.months.empty()?"?":p.months.back().c_str());

    auto grid = build_grid();
    printf("sweeping %zu parameter combos...\n", grid.size());
    auto results = backtest_sweep(p, grid);

    std::sort(results.begin(), results.end(),
              [](const Result& a, const Result& b){ return a.sharpe > b.sharpe; });

    printf("\ntop configs by annualized Sharpe:\n");
    printf("  %-8s %-6s %-6s %-5s %-4s | %-8s %-7s %-8s %-6s %s\n",
           "lookback","decay","beta","dec","hor","Sharpe","IC","meanRet","maxDD","turnover");
    for (int i = 0; i < std::min((int)results.size(), 8); ++i) {
        const auto& r = results[i]; const auto& q = r.params;
        printf("  %-8d %-6.2f %-6.2f %-5.2f %-4d | %+8.3f %+7.3f %+8.4f %7.4f %.2f\n",
               q.lookback, q.decay, q.accel_beta, q.decile, q.horizon,
               r.sharpe, r.ic, r.mean_return, r.max_drawdown, r.turnover);
    }
    // Out-of-sample honesty check: pick best params on data BEFORE --split, test AFTER.
    if (!split.empty()) {
        int si = p.n_months;
        for (int i = 0; i < p.n_months; ++i) if (p.months[i] >= split) { si = i; break; }
        printf("\n=== out-of-sample validation (split at %s, index %d/%d) ===\n",
               split.c_str(), si, p.n_months);
        Result bestIS; bestIS.sharpe = -1e18;
        for (const auto& pr : grid) {
            Result r = backtest_one(p, pr, 0, si);
            if (r.n_periods >= 12 && r.sharpe > bestIS.sharpe) bestIS = r;
        }
        Result oos  = backtest_one(p, bestIS.params, si, p.n_months);
        const auto& q = bestIS.params;
        printf("  best in-sample config: lookback=%d decay=%.2f beta=%.2f dec=%.2f hor=%d\n",
               q.lookback, q.decay, q.accel_beta, q.decile, q.horizon);
        printf("  %-12s %-8s %-8s %-9s %-6s %s\n", "period", "Sharpe", "IC", "meanRet", "maxDD", "n");
        printf("  %-12s %+8.3f %+8.3f %+9.4f %6.3f %d\n", "in-sample",
               bestIS.sharpe, bestIS.ic, bestIS.mean_return, bestIS.max_drawdown, bestIS.n_periods);
        printf("  %-12s %+8.3f %+8.3f %+9.4f %6.3f %d\n", "OUT-OF-SAMP",
               oos.sharpe, oos.ic, oos.mean_return, oos.max_drawdown, oos.n_periods);
        double retention = bestIS.sharpe != 0 ? oos.sharpe / bestIS.sharpe : 0;
        printf("  verdict: OOS Sharpe %.3f is %.0f%% of in-sample %s\n",
               oos.sharpe, 100.0 * retention,
               oos.sharpe > 0.15 ? "-> signal SURVIVES out of sample"
                                 : "-> signal COLLAPSES (was overfit)");
    }

    if (!results.empty()) print_live_ranking(p, results.front().params, top);
    return 0;
}
