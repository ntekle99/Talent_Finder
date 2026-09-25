// LLVM-JIT signal compiler + backtest search.
//
// Lowers a signal AST (signal_ast.hpp) to LLVM IR, JIT-compiles it with ORC into a native
// function  float sig(const float* net_flow_row, int t, int M),  then backtests each compiled
// signal cross-sectionally. This turns the engine into a signal-SEARCH compiler: we sweep over
// signal *expressions* (not just numeric params), JIT each, and rank by information coefficient.
//
// Windows (ewma/mean/lag) are unrolled at compile time (k is a literal), with branchless NaN
// masking via select -- straight-line IR an optimizer vectorizes well.
//
// Build (see engine/jit/build.sh):
//   clang++ -std=c++17 jit_backtest.cpp -I../include $(llvm-config --cxxflags --ldflags --libs \
//           core orcjit native passes) -o jit_backtest
#include "signal_ast.hpp"
#include "panel.hpp"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/OptimizationLevel.h"
#include <cmath>
#include <functional>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdio>

using namespace llvm;
using namespace tf;
using SigFn = float(*)(const float*, int, int);

// ---------------- LLVM lowering ----------------
struct Codegen {
    LLVMContext& ctx; IRBuilder<>& b; Value* row; Value* t; Type* f32;
    Value* nan;

    Value* fconst(double v){ return ConstantFP::get(f32, v); }
    // Load net_flow[t - off] with a bounds+NaN guard; returns {value, validFlag(i1)}.
    std::pair<Value*,Value*> elem(int off){
        Value* s = b.CreateSub(t, b.getInt32(off));
        Value* sge = b.CreateICmpSGE(s, b.getInt32(0));
        Value* sidx = b.CreateSelect(sge, s, b.getInt32(0));
        Value* ptr = b.CreateGEP(f32, row, ArrayRef<Value*>{sidx});
        Value* v = b.CreateLoad(f32, ptr);
        Value* notnan = b.CreateNot(b.CreateFCmpUNO(v, v));   // !isnan
        return { v, b.CreateAnd(sge, notnan) };
    }
    // weighted window reduce: returns num/den (or NaN if den==0); weight_i supplied by cb.
    Value* window(int k, const std::function<double(int)>& weight){
        Value* num = fconst(0), *den = fconst(0);
        for(int i=0;i<k;++i){
            auto [v, ok] = elem(i);
            Value* w = fconst(weight(i));
            Value* contrib = b.CreateSelect(ok, b.CreateFMul(w, v), fconst(0));
            Value* dcontrib= b.CreateSelect(ok, w, fconst(0));
            num = b.CreateFAdd(num, contrib);
            den = b.CreateFAdd(den, dcontrib);
        }
        Value* has = b.CreateFCmpOGT(den, fconst(0));
        return b.CreateSelect(has, b.CreateFDiv(num, den), nan);
    }
    Value* gen(const sig::Node* n){
        using sig::Op;
        switch(n->op){
            case Op::Const: return fconst(n->c);
            case Op::Nf:    return elem(0).first;                       // net_flow[t]
            case Op::Lag:  { auto [v,ok]=elem(n->k); return b.CreateSelect(ok, v, nan); }
            case Op::Ewma: { double d=n->c; return window(n->k, [d](int i){ return std::pow(1.0-d, i); }); }
            case Op::Mean:  return window(n->k, [](int){ return 1.0; });
            case Op::Add:   return b.CreateFAdd(gen(n->a.get()), gen(n->b.get()));
            case Op::Sub:   return b.CreateFSub(gen(n->a.get()), gen(n->b.get()));
            case Op::Mul:   return b.CreateFMul(gen(n->a.get()), gen(n->b.get()));
            case Op::Div:   return b.CreateFDiv(gen(n->a.get()), gen(n->b.get()));
        }
        return fconst(0);
    }
};

static std::unique_ptr<Module> build_module(LLVMContext& ctx, const sig::Node* ast, bool dump){
    auto M = std::make_unique<Module>("sig", ctx);
    Type* f32 = Type::getFloatTy(ctx); Type* i32 = Type::getInt32Ty(ctx);
    Type* pf32 = PointerType::getUnqual(ctx);
    FunctionType* ft = FunctionType::get(f32, {pf32, i32, i32}, false);
    Function* fn = Function::Create(ft, Function::ExternalLinkage, "sig", M.get());
    auto args = fn->arg_begin();
    Value* row = &*args++; row->setName("row");
    Value* t   = &*args++; t->setName("t"); (&*args)->setName("M");
    IRBuilder<> b(BasicBlock::Create(ctx, "entry", fn));
    Codegen cg{ctx, b, row, t, f32, ConstantFP::getNaN(f32)};
    b.CreateRet(cg.gen(ast));
    if (verifyFunction(*fn, &errs())) throw std::runtime_error("bad IR");
    // O2 optimize (constant-folds weights, vectorizes the unrolled window)
    PassBuilder pb; LoopAnalysisManager lam; FunctionAnalysisManager fam;
    CGSCCAnalysisManager cam; ModuleAnalysisManager mam;
    pb.registerModuleAnalyses(mam); pb.registerCGSCCAnalyses(cam);
    pb.registerFunctionAnalyses(fam); pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cam, mam);
    pb.buildPerModuleDefaultPipeline(OptimizationLevel::O2).run(*M, mam);
    if (dump){ printf("\n----- generated LLVM IR (optimized) -----\n"); M->print(outs(), nullptr); printf("-----------------------------------------\n"); }
    return M;
}

// ---------------- backtest a compiled signal ----------------
static void backtest(const Panel& p, SigFn sig, double& sharpe, double& ic){
    std::vector<double> rets, ics;
    for(int t=13;t+1<p.n_months;++t){
        std::vector<float> s(p.n_companies), fwd(p.n_companies);
        std::vector<int> elig;
        for(int c=0;c<p.n_companies;++c){
            float v = sig(&p.net_flow[(size_t)c*p.n_months], t, p.n_months);
            float f = p.ret[(size_t)c*p.n_months + t+1];
            s[c]=v; fwd[c]=f;
            if(std::isfinite(v) && !is_nan(f)) elig.push_back(c);
        }
        if((int)elig.size()<10) continue;
        // z-score not needed for rank; sort by signal
        std::sort(elig.begin(),elig.end(),[&](int a,int b){return s[a]>s[b];});
        int k=std::max(1,(int)(0.2*elig.size()));
        double lo=0,sh=0; for(int i=0;i<k;++i){lo+=fwd[elig[i]];sh+=fwd[elig[elig.size()-1-i]];}
        rets.push_back(lo/k - sh/k);
        // rank IC
        int n=elig.size(); std::vector<int> bysig(elig), byret(elig);
        std::sort(byret.begin(),byret.end(),[&](int a,int b){return fwd[a]<fwd[b];});
        std::vector<double> rs(p.n_companies), rr(p.n_companies);
        for(int i=0;i<n;++i){ rs[bysig[i]]=i; rr[byret[i]]=i; }
        double m=(n-1)/2.0,cov=0,va=0,vb=0;
        for(int c:elig){cov+=(rs[c]-m)*(rr[c]-m);va+=(rs[c]-m)*(rs[c]-m);vb+=(rr[c]-m)*(rr[c]-m);}
        ics.push_back((va>0&&vb>0)?cov/std::sqrt(va*vb):0);
    }
    if(rets.empty()){sharpe=ic=0;return;}
    double mean=std::accumulate(rets.begin(),rets.end(),0.0)/rets.size(),var=0;
    for(double r:rets)var+=(r-mean)*(r-mean);
    double sd=std::sqrt(var/rets.size());
    sharpe = sd>1e-12? mean/sd*std::sqrt(12.0):0;
    ic = std::accumulate(ics.begin(),ics.end(),0.0)/ics.size();
}

int main(int argc,char**argv){
    std::string panel="data/panel/talent_panel.csv", returns="data/panel/returns.csv";
    for(int i=1;i<argc;++i){ std::string a=argv[i];
        if(a=="--panel"&&i+1<argc)panel=argv[++i]; else if(a=="--returns"&&i+1<argc)returns=argv[++i]; }
    Panel p; try{ p=load_panel(panel,returns);}catch(std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}
    printf("panel %dx%d\n\n", p.n_companies, p.n_months);

    InitializeNativeTarget(); InitializeNativeTargetAsmPrinter();

    // the SIGNAL SEARCH SPACE -- different signal *structures*, each JIT-compiled
    std::vector<std::string> dsl = {
        "nf",
        "ewma(6,0.4)",
        "mean(6) - mean(12)",
        "ewma(3,0.5) - ewma(12,0.3)",
        "ewma(6,0.4) / mean(24) - 1",
        "ewma(6,0.4) - lag(12)",
        "(ewma(3,0.5) - ewma(12,0.3)) / mean(24)",
    };
    using clk = std::chrono::high_resolution_clock;
    auto ms = [](clk::duration d){ return std::chrono::duration<double,std::milli>(d).count(); };
    struct R{ std::string s; double sharpe, ic, compile_ms, exec_ms; };
    std::vector<R> results;
    for(size_t i=0;i<dsl.size();++i){
        try{
            auto t0=clk::now();
            auto ast = sig::parse(dsl[i]);
            auto ctx = std::make_unique<LLVMContext>();
            auto M = build_module(*ctx, ast.get(), /*dump=*/i==3);  // dump IR for one example
            auto J = cantFail(orc::LLJITBuilder().create());
            cantFail(J->addIRModule(orc::ThreadSafeModule(std::move(M), std::move(ctx))));
            auto sym = cantFail(J->lookup("sig"));
            SigFn fn = sym.toPtr<float(*)(const float*,int,int)>();
            auto t1=clk::now();
            double sh,ic; backtest(p, fn, sh, ic);
            auto t2=clk::now();
            results.push_back({dsl[i], sh, ic, ms(t1-t0), ms(t2-t1)});
        }catch(std::exception&e){ printf("  [%s] ERROR: %s\n", dsl[i].c_str(), e.what()); }
    }
    std::sort(results.begin(),results.end(),[](const R&a,const R&b){return a.ic>b.ic;});
    printf("\n=== signal search (JIT-compiled expressions, ranked by IC) ===\n");
    printf("  %-42s %7s %7s %10s %9s\n","signal expression (DSL)","IC","Sharpe","compile","backtest");
    double tc=0,te=0;
    for(auto&r:results){ printf("  %-42s %+7.3f %+7.3f %8.2fms %7.2fms\n",
        r.s.c_str(), r.ic, r.sharpe, r.compile_ms, r.exec_ms); tc+=r.compile_ms; te+=r.exec_ms; }
    printf("\nJIT-compiled & backtested %zu signal structures: avg compile %.2f ms, avg backtest %.2f ms.\n",
           results.size(), tc/results.size(), te/results.size());
    return 0;
}
