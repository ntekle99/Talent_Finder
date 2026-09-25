// Multi-GPU backtest sweep with NCCL inter-GPU communication.
//
// Demonstrates real GPU-to-GPU collectives on a single node with 2+ GPUs:
//   1. ncclBroadcast  -- panel data is loaded onto GPU 0, then broadcast to all GPUs.
//   2. per-GPU compute -- the parameter grid is split across GPUs; each runs its slice.
//   3. ncclAllReduce  -- each GPU's local best Sharpe is all-reduced (MAX) so every GPU
//                        learns the global best config across the whole cluster.
//
// Single-process / multi-device model via ncclCommInitAll (no MPI needed for one node).
// Build:  nvcc -O3 -I../include nccl_backtest.cu -lnccl -o talent_nccl
#include "backtest.hpp"
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>
#include <vector>
#include <numeric>

using namespace tf;

#define CUDACHECK(x) do { cudaError_t e=(x); if(e){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::abort(); } } while(0)
#define NCCLCHECK(x) do { ncclResult_t r=(x); if(r!=ncclSuccess){ \
  fprintf(stderr,"NCCL %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(r)); std::abort(); } } while(0)

struct DevPanel { const float* net_flow; const float* ret; int C; int M;
    __device__ int idx(int c,int t) const { return c*M + t; } };

__device__ inline bool d_nan(float x){ return isnan(x); }

__device__ float d_ewma(const DevPanel& p,int c,int t,int lookback,float decay){
    float num=0,den=0,w=1; int s0=t-lookback+1; if(s0<0)s0=0;
    for(int s=t;s>=s0;--s){ float v=p.net_flow[p.idx(c,s)]; if(!d_nan(v)){num+=w*v;den+=w;} w*=(1.f-decay); }
    return den>0?num/den:nanf(""); }

__device__ float d_forward(const DevPanel& p,int c,int t,int h){
    if(t+h>=p.M) return nanf(""); float g=1;
    for(int s=t+1;s<=t+h;++s){ float r=p.ret[p.idx(c,s)]; if(d_nan(r))return nanf(""); g*=(1.f+r);} return g-1; }

// One thread == one parameter combo (same math as cuda_backtest.cu).
__global__ void sweep_kernel(DevPanel p,const Params* grid,int n,Result* out,
                             float* score_buf,float* fwd_buf,int C){
    int gi=blockIdx.x*blockDim.x+threadIdx.x; if(gi>=n) return;
    Params pr=grid[gi];
    float* score=score_buf+(size_t)gi*C; float* fwd=fwd_buf+(size_t)gi*C;
    double sum=0,sum2=0; int np=0; double cum=0,peak=0,mdd=0;
    for(int t=0;t<p.M;++t){
        double m=0; int nm=0;
        for(int c=0;c<C;++c){ float mo=d_ewma(p,c,t,pr.lookback,pr.decay); score[c]=mo;
            if(!d_nan(mo)){m+=mo;++nm;} fwd[c]=d_forward(p,c,t,pr.horizon); }
        if(nm<5) continue; m/=nm; double var=0;
        for(int c=0;c<C;++c) if(!d_nan(score[c])) var+=(score[c]-m)*(score[c]-m);
        double sd=sqrt(var/nm); if(sd<1e-9)sd=1;
        for(int c=0;c<C;++c) if(!d_nan(score[c])) score[c]=(score[c]-m)/sd;
        int elig=0; for(int c=0;c<C;++c) if(!d_nan(score[c])&&!d_nan(fwd[c])) ++elig;
        if(elig<5) continue; int k=(int)(pr.decile*elig); if(k<1)k=1;
        double lo=0,sh=0;
        for(int pick=0;pick<k;++pick){ int bi=-1,si=-1; float bv=-1e30f,sv=1e30f;
            for(int c=0;c<C;++c){ if(d_nan(score[c])||d_nan(fwd[c]))continue;
                if(score[c]>bv){bv=score[c];bi=c;} if(score[c]<sv){sv=score[c];si=c;} }
            if(bi>=0){lo+=fwd[bi];score[bi]=nanf("");} if(si>=0){sh+=fwd[si];score[si]=nanf("");} }
        double pr_ret=lo/k-sh/k; sum+=pr_ret; sum2+=pr_ret*pr_ret; ++np;
        cum+=pr_ret; if(cum>peak)peak=cum; if(cum-peak<mdd)mdd=cum-peak;
    }
    Result R; R.params=pr; R.n_periods=np;
    if(np>0){ double mean=sum/np; double v=(sum2-np*mean*mean)/(np>1?np-1:1);
        double sd=v>0?sqrt(v):0; R.mean_return=mean; R.sharpe=sd>1e-12?mean/sd*sqrt(12.0/pr.horizon):0;
        R.max_drawdown=mdd; }
    out[gi]=R;
}

// Reduce a Result chunk to its max Sharpe (small n -> single thread is fine for the demo).
__global__ void reduce_best(const Result* out,int n,float* best){
    float m=-1e30f; for(int i=0;i<n;++i) if(out[i].sharpe>m) m=out[i].sharpe; *best=m;
}

static std::vector<Params> build_grid(){
    std::vector<Params> g;
    for(int lb:{6,12,24}) for(float dc:{0.2f,0.4f,0.6f}) for(float b:{0.f,0.5f,1.f})
        for(float dec:{0.1f,0.2f,0.3f}) for(int h:{1,3,6}) g.push_back(Params{lb,dc,b,dec,h});
    return g;
}

int main(int argc,char** argv){
    std::string panel="data/panel/talent_panel.csv", returns="data/panel/returns.csv";
    for(int i=1;i<argc;++i){ std::string a=argv[i];
        if(a=="--panel"&&i+1<argc)panel=argv[++i]; else if(a=="--returns"&&i+1<argc)returns=argv[++i]; }

    Panel p; try{ p=load_panel(panel,returns);}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}
    auto grid=build_grid(); int N=(int)grid.size(); int C=p.n_companies,M=p.n_months;
    size_t cells=(size_t)C*M;

    int nDev=0; CUDACHECK(cudaGetDeviceCount(&nDev));
    if(nDev<1){ fprintf(stderr,"no CUDA devices\n"); return 1; }
    printf("panel %dx%d | %d combos | %d GPU(s)\n",C,M,N,nDev);
    for(int d=0;d<nDev;++d){ cudaDeviceProp pr; cudaGetDeviceProperties(&pr,d);
        printf("  GPU %d: %s\n",d,pr.name); }

    std::vector<int> devs(nDev); std::iota(devs.begin(),devs.end(),0);
    std::vector<ncclComm_t> comms(nDev);
    NCCLCHECK(ncclCommInitAll(comms.data(),nDev,devs.data()));

    // per-GPU buffers
    std::vector<float*> d_net(nDev),d_ret(nDev),d_score(nDev),d_fwd(nDev),d_lbest(nDev),d_gbest(nDev);
    std::vector<Params*> d_grid(nDev); std::vector<Result*> d_out(nDev);
    std::vector<cudaStream_t> st(nDev);
    // grid split across GPUs
    int chunk=(N+nDev-1)/nDev;
    std::vector<int> off(nDev),cnt(nDev);
    for(int d=0;d<nDev;++d){ off[d]=std::min(d*chunk,N); cnt[d]=std::min(chunk,N-off[d]); }

    for(int d=0;d<nDev;++d){ CUDACHECK(cudaSetDevice(d)); CUDACHECK(cudaStreamCreate(&st[d]));
        CUDACHECK(cudaMalloc(&d_net[d],cells*sizeof(float)));
        CUDACHECK(cudaMalloc(&d_ret[d],cells*sizeof(float)));
        CUDACHECK(cudaMalloc(&d_grid[d],std::max(1,cnt[d])*sizeof(Params)));
        CUDACHECK(cudaMalloc(&d_out[d],std::max(1,cnt[d])*sizeof(Result)));
        CUDACHECK(cudaMalloc(&d_score[d],(size_t)std::max(1,cnt[d])*C*sizeof(float)));
        CUDACHECK(cudaMalloc(&d_fwd[d],(size_t)std::max(1,cnt[d])*C*sizeof(float)));
        CUDACHECK(cudaMalloc(&d_lbest[d],sizeof(float)));
        CUDACHECK(cudaMalloc(&d_gbest[d],sizeof(float)));
    }
    // load panel onto GPU 0 only, then BROADCAST to all GPUs via NCCL
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMemcpy(d_net[0],p.net_flow.data(),cells*sizeof(float),cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_ret[0],p.ret.data(),cells*sizeof(float),cudaMemcpyHostToDevice));
    printf("\n[NCCL] broadcasting panel from GPU 0 -> all GPUs (%.1f MB)\n", 2.0*cells*sizeof(float)/1e6);
    NCCLCHECK(ncclGroupStart());
    for(int d=0;d<nDev;++d){ NCCLCHECK(ncclBroadcast(d_net[d],d_net[d],cells,ncclFloat,0,comms[d],st[d])); }
    NCCLCHECK(ncclGroupEnd());
    NCCLCHECK(ncclGroupStart());
    for(int d=0;d<nDev;++d){ NCCLCHECK(ncclBroadcast(d_ret[d],d_ret[d],cells,ncclFloat,0,comms[d],st[d])); }
    NCCLCHECK(ncclGroupEnd());
    for(int d=0;d<nDev;++d){ CUDACHECK(cudaSetDevice(d)); CUDACHECK(cudaStreamSynchronize(st[d])); }

    // each GPU computes its slice of the sweep
    printf("[compute] splitting %d combos: ",N);
    for(int d=0;d<nDev;++d) printf("GPU%d=%d ",d,cnt[d]); printf("\n");
    for(int d=0;d<nDev;++d){ if(cnt[d]<=0) continue; CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaMemcpyAsync(d_grid[d],grid.data()+off[d],cnt[d]*sizeof(Params),cudaMemcpyHostToDevice,st[d]));
        DevPanel dp{d_net[d],d_ret[d],C,M};
        int th=64, bl=(cnt[d]+th-1)/th;
        sweep_kernel<<<bl,th,0,st[d]>>>(dp,d_grid[d],cnt[d],d_out[d],d_score[d],d_fwd[d],C);
        reduce_best<<<1,1,0,st[d]>>>(d_out[d],cnt[d],d_lbest[d]);
    }
    for(int d=0;d<nDev;++d){ CUDACHECK(cudaSetDevice(d)); CUDACHECK(cudaStreamSynchronize(st[d])); }

    // report each GPU's local best, then ALL-REDUCE (MAX) so all GPUs agree on the global best
    std::vector<float> lbest(nDev);
    for(int d=0;d<nDev;++d){ CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaMemcpy(&lbest[d],d_lbest[d],sizeof(float),cudaMemcpyDeviceToHost)); }
    printf("\n[local] per-GPU best Sharpe:");
    for(int d=0;d<nDev;++d) printf(" GPU%d=%.3f",d,lbest[d]); printf("\n");

    printf("[NCCL] all-reduce(MAX) local bests -> global best on every GPU\n");
    NCCLCHECK(ncclGroupStart());
    for(int d=0;d<nDev;++d) NCCLCHECK(ncclAllReduce(d_lbest[d],d_gbest[d],1,ncclFloat,ncclMax,comms[d],st[d]));
    NCCLCHECK(ncclGroupEnd());
    for(int d=0;d<nDev;++d){ CUDACHECK(cudaSetDevice(d)); CUDACHECK(cudaStreamSynchronize(st[d])); }

    std::vector<float> gbest(nDev);
    for(int d=0;d<nDev;++d){ CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaMemcpy(&gbest[d],d_gbest[d],sizeof(float),cudaMemcpyDeviceToHost)); }
    bool agree=true; for(int d=1;d<nDev;++d) if(gbest[d]!=gbest[0]) agree=false;
    printf("[global] every GPU now agrees best Sharpe = %.3f  (consistent across GPUs: %s)\n",
           gbest[0], agree?"yes":"NO");

    // find the winning config (gather results host-side for reporting)
    Result best; best.sharpe=-1e30;
    for(int d=0;d<nDev;++d){ if(cnt[d]<=0)continue; CUDACHECK(cudaSetDevice(d));
        std::vector<Result> h(cnt[d]);
        CUDACHECK(cudaMemcpy(h.data(),d_out[d],cnt[d]*sizeof(Result),cudaMemcpyDeviceToHost));
        for(auto&r:h) if(r.sharpe>best.sharpe) best=r; }
    printf("\nbest config: lookback=%d decay=%.2f decile=%.2f horizon=%d -> Sharpe %.3f\n",
           best.params.lookback,best.params.decay,best.params.decile,best.params.horizon,best.sharpe);

    for(int d=0;d<nDev;++d){ CUDACHECK(cudaSetDevice(d));
        cudaFree(d_net[d]);cudaFree(d_ret[d]);cudaFree(d_grid[d]);cudaFree(d_out[d]);
        cudaFree(d_score[d]);cudaFree(d_fwd[d]);cudaFree(d_lbest[d]);cudaFree(d_gbest[d]);
        cudaStreamDestroy(st[d]); }
    for(int d=0;d<nDev;++d) ncclCommDestroy(comms[d]);
    printf("\n[done] NCCL broadcast + all-reduce across %d GPUs\n",nDev);
    return 0;
}
