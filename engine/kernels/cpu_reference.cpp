// CPU reference sweep — the correctness oracle. Optionally threaded with OpenMP.
#include "backtest.hpp"

namespace tf {

std::vector<Result> backtest_sweep(const Panel& p, const std::vector<Params>& grid) {
    std::vector<Result> out(grid.size());
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < (int)grid.size(); ++i)
        out[i] = backtest_one(p, grid[i]);
    return out;
}

} // namespace tf
