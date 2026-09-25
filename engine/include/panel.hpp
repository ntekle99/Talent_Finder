// Panel: the Python<->C++ contract, loaded into dense company x month grids.
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace tf {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
inline bool is_nan(float x) { return std::isnan(x); }

// Dense grids are row-major [company * n_months + month].
struct Panel {
    std::vector<std::string> company_ids;     // size = n_companies
    std::vector<std::string> tickers;         // size = n_companies
    std::vector<std::string> months;          // size = n_months, sorted "YYYY-MM"
    int n_companies = 0;
    int n_months = 0;

    std::vector<float> net_flow;   // senior_hires - departures   [C*M]
    std::vector<float> ret;        // monthly total return         [C*M], NaN if unknown

    int idx(int c, int t) const { return c * n_months + t; }
};

namespace detail {
inline std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    std::stringstream ss(line);
    while (std::getline(ss, cur, ',')) out.push_back(cur);
    if (!line.empty() && line.back() == ',') out.push_back("");
    return out;
}
inline float to_f(const std::string& s) {
    if (s.empty()) return kNaN;
    try { return std::stof(s); } catch (...) { return kNaN; }
}
// Header-name -> column index (case-sensitive, trimmed of trailing \r).
inline std::unordered_map<std::string,int> header_index(const std::string& header) {
    std::unordered_map<std::string,int> h;
    auto cols = split_csv(header);
    for (int i = 0; i < (int)cols.size(); ++i) {
        std::string c = cols[i];
        if (!c.empty() && c.back() == '\r') c.pop_back();
        h[c] = i;
    }
    return h;
}
} // namespace detail

// Load talent_panel.csv (company_id,ticker,month,...,senior_hires,departures)
// and returns.csv (company_id,month,ret), aligning both onto one grid.
inline Panel load_panel(const std::string& panel_csv, const std::string& returns_csv) {
    using namespace detail;
    std::ifstream pf(panel_csv);
    if (!pf) throw std::runtime_error("cannot open panel: " + panel_csv);

    // First pass: discover the sorted month axis and company set.
    std::string line;
    std::getline(pf, line);
    auto ph = header_index(line);
    for (const char* req : {"company_id","month","senior_hires","departures"})
        if (!ph.count(req)) throw std::runtime_error(std::string("panel missing column: ") + req);

    struct Row { std::string cid, ticker, month; float senior, dep; };
    std::vector<Row> rows;
    std::unordered_map<std::string,std::string> ticker_of;
    std::unordered_map<std::string,int> month_ids, comp_ids;
    while (std::getline(pf, line)) {
        if (line.empty()) continue;
        auto f = split_csv(line);
        Row r;
        r.cid = f[ph["company_id"]];
        r.ticker = ph.count("ticker") && ph["ticker"] < (int)f.size() ? f[ph["ticker"]] : "";
        r.month = f[ph["month"]];
        r.senior = to_f(f[ph["senior_hires"]]);
        r.dep = to_f(f[ph["departures"]]);
        if (!r.month.empty() && r.month.back() == '\r') r.month.pop_back();
        ticker_of[r.cid] = r.ticker;
        month_ids.emplace(r.month, 0);
        comp_ids.emplace(r.cid, 0);
        rows.push_back(std::move(r));
    }

    Panel p;
    for (auto& kv : month_ids) p.months.push_back(kv.first);
    std::sort(p.months.begin(), p.months.end());
    for (int i = 0; i < (int)p.months.size(); ++i) month_ids[p.months[i]] = i;
    for (auto& kv : comp_ids) p.company_ids.push_back(kv.first);
    std::sort(p.company_ids.begin(), p.company_ids.end());
    for (int i = 0; i < (int)p.company_ids.size(); ++i) comp_ids[p.company_ids[i]] = i;

    p.n_companies = (int)p.company_ids.size();
    p.n_months = (int)p.months.size();
    p.tickers.resize(p.n_companies);
    for (int c = 0; c < p.n_companies; ++c) p.tickers[c] = ticker_of[p.company_ids[c]];

    p.net_flow.assign((size_t)p.n_companies * p.n_months, kNaN);
    p.ret.assign((size_t)p.n_companies * p.n_months, kNaN);
    for (auto& r : rows) {
        int c = comp_ids[r.cid], t = month_ids[r.month];
        float sr = is_nan(r.senior) ? 0.f : r.senior;
        float dp = is_nan(r.dep) ? 0.f : r.dep;
        p.net_flow[p.idx(c, t)] = sr - dp;
    }

    // Second file: monthly returns.
    std::ifstream rf(returns_csv);
    if (!rf) throw std::runtime_error("cannot open returns: " + returns_csv);
    std::getline(rf, line);
    auto rh = header_index(line);
    for (const char* req : {"company_id","month","ret"})
        if (!rh.count(req)) throw std::runtime_error(std::string("returns missing column: ") + req);
    while (std::getline(rf, line)) {
        if (line.empty()) continue;
        auto f = split_csv(line);
        std::string cid = f[rh["company_id"]];
        std::string m = f[rh["month"]];
        if (!m.empty() && m.back() == '\r') m.pop_back();
        auto ci = comp_ids.find(cid); auto mi = month_ids.find(m);
        if (ci == comp_ids.end() || mi == month_ids.end()) continue;
        p.ret[p.idx(ci->second, mi->second)] = to_f(f[rh["ret"]]);
    }
    return p;
}

} // namespace tf
