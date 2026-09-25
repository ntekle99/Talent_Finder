// CUDA sweep — built and run on a Brev GPU (nvcc). Mirrors cpu_reference.cpp's math.
//
// Parallelization strategy for the thin slice: one CUDA thread per parameter combo.
// Each thread runs the full backtest_one() logic over the panel (which lives in device
// global memory). This already turns a large grid search (hundreds/thousands of combos)
// into a single launch. The next optimization — deferred until we profile — is to also
// parallelize the per-month cross-section (block per combo, warp reductions for the
// z-score mean/std and a bitonic sort for the decile split).
//
// The panel is small (companies x months x 4 bytes); it fits comfortably in global memory
// and the per-thread reads are coalesced across combos hitting the same [c,t] cells.

#include "backtest.hpp"
#include <cstdio>
#include <cuda_runtime.h>

namespace tf {

// Flat, device-friendly view of the panel.
struct DevPanel {
    const float* net_flow;
    const float* ret;
    int n_companies;
    int n_months;
    __device__ int idx(int c, int t) const { return c * n_months + t; }
};

// --- device mirrors of the host math (kept deliberately close to backtest.hpp) ---
__device__ inline bool d_nan(float x) { return isnan(x); }

__device__ float d_ewma(const DevPanel& p, int c, int t, int lookback, float decay) {
    float num = 0.f, den = 0.f, w = 1.f;
    int start = t - lookback + 1; if (start < 0) start = 0;
    for (int s = t; s >= start; --s) {
        float v = p.net_flow[p.idx(c, s)];
        if (!d_nan(v)) { num += w * v; den += w; }
        w *= (1.f - decay);
    }
    return den > 0 ? num / den : nanf("");
}

__device__ float d_forward(const DevPanel& p, int c, int t, int horizon) {
    if (t + horizon >= p.n_months) return nanf("");
    float g = 1.f;
    for (int s = t + 1; s <= t + horizon; ++s) {
        float r = p.ret[p.idx(c, s)];
        if (d_nan(r)) return nanf("");
        g *= (1.f + r);
    }
    return g - 1.f;
}

// One thread == one parameter combo. Scratch buffers are per-thread slices of a big
// device array so we avoid dynamic allocation inside the kernel.
__global__ void sweep_kernel(DevPanel p, const Params* grid, int n_grid,
                             Result* out, float* score_buf, float* fwd_buf, int C) {
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= n_grid) return;
    Params pr = grid[gi];
    float* score = score_buf + (size_t)gi * C;
    float* fwd   = fwd_buf   + (size_t)gi * C;

    double sum_ret = 0, sum_ret2 = 0; int n_periods = 0;
    double cum = 0, peak = 0, mdd = 0;

    for (int t = 0; t < p.n_months; ++t) {
        // momentum + acceleration, then cross-sectional mean/std for z-scoring
        double m = 0; int nm = 0;
        for (int c = 0; c < C; ++c) {
            float mom = d_ewma(p, c, t, pr.lookback, pr.decay);
            score[c] = mom;                          // stash raw momentum
            if (!d_nan(mom)) { m += mom; ++nm; }
            fwd[c] = d_forward(p, c, t, pr.horizon);
        }
        if (nm < 5) continue;
        m /= nm; double var = 0;
        for (int c = 0; c < C; ++c) if (!d_nan(score[c])) var += (score[c]-m)*(score[c]-m);
        double sd = sqrt(var / nm); if (sd < 1e-9) sd = 1;
        for (int c = 0; c < C; ++c) if (!d_nan(score[c])) score[c] = (score[c]-m)/sd;
        // (acceleration term omitted in this thin-slice device path; add mirroring host)

        // count eligible and find decile cutoffs by simple selection
        int elig = 0;
        for (int c = 0; c < C; ++c) if (!d_nan(score[c]) && !d_nan(fwd[c])) ++elig;
        if (elig < 5) continue;
        int k = (int)(pr.decile * elig); if (k < 1) k = 1;

        // O(C*k) partial selection for top-k and bottom-k (k is tiny) — fine for thin slice
        double long_ret = 0, short_ret = 0;
        for (int pick = 0; pick < k; ++pick) {
            int bi = -1, si = -1; float bv = -1e30f, sv = 1e30f;
            for (int c = 0; c < C; ++c) {
                if (d_nan(score[c]) || d_nan(fwd[c])) continue;
                if (score[c] > bv) { bv = score[c]; bi = c; }
                if (score[c] < sv) { sv = score[c]; si = c; }
            }
            if (bi >= 0) { long_ret  += fwd[bi]; score[bi] = nanf(""); }
            if (si >= 0) { short_ret += fwd[si]; score[si] = nanf(""); }
        }
        double pr_ret = long_ret / k - short_ret / k;
        sum_ret += pr_ret; sum_ret2 += pr_ret * pr_ret; ++n_periods;
        cum += pr_ret; if (cum > peak) peak = cum; if (cum - peak < mdd) mdd = cum - peak;
    }

    Result R; R.params = pr; R.n_periods = n_periods;
    if (n_periods > 0) {
        double mean = sum_ret / n_periods;
        double v = (sum_ret2 - n_periods*mean*mean) / (n_periods > 1 ? n_periods-1 : 1);
        double sd = v > 0 ? sqrt(v) : 0;
        R.mean_return = mean;
        R.sharpe = sd > 1e-12 ? mean/sd * sqrt(12.0/pr.horizon) : 0.0;
        R.decile_spread = mean; R.max_drawdown = mdd;
    }
    out[gi] = R;
}

#define CUDA_OK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::abort();} } while(0)

std::vector<Result> backtest_sweep(const Panel& p, const std::vector<Params>& grid) {
    int C = p.n_companies, M = p.n_months, N = (int)grid.size();
    float *d_net, *d_ret, *d_score, *d_fwd; Params* d_grid; Result* d_out;
    size_t cells = (size_t)C * M;
    CUDA_OK(cudaMalloc(&d_net, cells*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_ret, cells*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_grid, N*sizeof(Params)));
    CUDA_OK(cudaMalloc(&d_out, N*sizeof(Result)));
    CUDA_OK(cudaMalloc(&d_score, (size_t)N*C*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_fwd,   (size_t)N*C*sizeof(float)));
    CUDA_OK(cudaMemcpy(d_net, p.net_flow.data(), cells*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_ret, p.ret.data(),      cells*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_grid, grid.data(), N*sizeof(Params), cudaMemcpyHostToDevice));

    DevPanel dp{d_net, d_ret, C, M};
    int threads = 128, blocks = (N + threads - 1) / threads;
    sweep_kernel<<<blocks, threads>>>(dp, d_grid, N, d_out, d_score, d_fwd, C);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<Result> out(N);
    CUDA_OK(cudaMemcpy(out.data(), d_out, N*sizeof(Result), cudaMemcpyDeviceToHost));
    cudaFree(d_net); cudaFree(d_ret); cudaFree(d_grid); cudaFree(d_out);
    cudaFree(d_score); cudaFree(d_fwd);
    return out;
}

} // namespace tf
