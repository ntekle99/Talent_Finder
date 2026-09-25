// NVPTX backend for the signal IR: lowers a signal AST to a CUDA kernel and emits PTX.
//
// Same signal_ast.hpp the LLVM-JIT (jit_backtest.cpp) uses -- here we target the
// nvptx64-nvidia-cuda triple instead of the host, wrapping the per-company signal expression
// in a __global__ kernel (one thread per company). This is the GPU codegen path (#2): the
// emitted PTX is exactly what would load via cuModuleLoadData and run on the 2-GPU NCCL engine.
//
// No GPU required to GENERATE PTX (it's just codegen). Build: see engine/jit/build_ptx.sh
//   ./emit_ptx "ewma(6,0.4) / mean(24) - 1"
#include "signal_ast.hpp"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/TargetParser/Triple.h"
#include <cmath>
#include <functional>
#include <cstdio>

using namespace llvm;

struct Codegen {
    LLVMContext& ctx; IRBuilder<>& b; Value* row; Value* t; Type* f32; Value* nan;
    Value* fc(double v){ return ConstantFP::get(f32, v); }
    std::pair<Value*,Value*> elem(int off){
        Value* s = b.CreateSub(t, b.getInt32(off));
        Value* ge = b.CreateICmpSGE(s, b.getInt32(0));
        Value* si = b.CreateSelect(ge, s, b.getInt32(0));
        Value* p  = b.CreateGEP(f32, row, ArrayRef<Value*>{si});
        Value* v  = b.CreateLoad(f32, p);
        Value* nn = b.CreateNot(b.CreateFCmpUNO(v, v));
        return { v, b.CreateAnd(ge, nn) };
    }
    Value* window(int k, const std::function<double(int)>& w){
        Value* num=fc(0),*den=fc(0);
        for(int i=0;i<k;++i){ auto [v,ok]=elem(i); Value* wi=fc(w(i));
            num=b.CreateFAdd(num,b.CreateSelect(ok,b.CreateFMul(wi,v),fc(0)));
            den=b.CreateFAdd(den,b.CreateSelect(ok,wi,fc(0))); }
        return b.CreateSelect(b.CreateFCmpOGT(den,fc(0)), b.CreateFDiv(num,den), nan);
    }
    Value* gen(const tf::sig::Node* n){ using tf::sig::Op;
        switch(n->op){
            case Op::Const: return fc(n->c);
            case Op::Nf:    return elem(0).first;
            case Op::Lag:  { auto [v,ok]=elem(n->k); return b.CreateSelect(ok,v,nan); }
            case Op::Ewma: { double d=n->c; return window(n->k,[d](int i){return std::pow(1.0-d,i);}); }
            case Op::Mean:  return window(n->k,[](int){return 1.0;});
            case Op::Add: return b.CreateFAdd(gen(n->a.get()),gen(n->b.get()));
            case Op::Sub: return b.CreateFSub(gen(n->a.get()),gen(n->b.get()));
            case Op::Mul: return b.CreateFMul(gen(n->a.get()),gen(n->b.get()));
            case Op::Div: return b.CreateFDiv(gen(n->a.get()),gen(n->b.get()));
        } return fc(0);
    }
};

int main(int argc,char**argv){
    std::string expr = argc>1 ? argv[1] : "ewma(6,0.4) / mean(24) - 1";
    auto ast = tf::sig::parse(expr);

    InitializeAllTargetInfos(); InitializeAllTargets();
    InitializeAllTargetMCs(); InitializeAllAsmPrinters();

    LLVMContext ctx; auto M = std::make_unique<Module>("sig_gpu", ctx);
    Triple triple("nvptx64-nvidia-cuda");
    M->setTargetTriple(triple);
    Type* f32=Type::getFloatTy(ctx); Type* i32=Type::getInt32Ty(ctx);
    Type* pf=PointerType::getUnqual(ctx);
    // __global__ void sigk(float* net_flow, float* out, int t, int C, int M)
    FunctionType* ft=FunctionType::get(Type::getVoidTy(ctx), {pf,pf,i32,i32,i32}, false);
    Function* fn=Function::Create(ft, Function::ExternalLinkage, "sigk", M.get());
    fn->setCallingConv(CallingConv::PTX_Kernel);   // mark as CUDA kernel
    auto a=fn->arg_begin(); Value* nf=&*a++; Value* out=&*a++; Value* t=&*a++; Value* C=&*a++; Value* Mn=&*a++;
    nf->setName("net_flow"); out->setName("out"); t->setName("t"); C->setName("C"); Mn->setName("M");

    IRBuilder<> b(BasicBlock::Create(ctx,"entry",fn));
    // c = tid.x + ctaid.x * ntid.x
    auto tid  = Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::nvvm_read_ptx_sreg_tid_x);
    auto ctaid= Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
    auto ntid = Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::nvvm_read_ptx_sreg_ntid_x);
    Value* c = b.CreateAdd(b.CreateCall(tid), b.CreateMul(b.CreateCall(ctaid), b.CreateCall(ntid)));
    BasicBlock* body=BasicBlock::Create(ctx,"body",fn); BasicBlock* ret=BasicBlock::Create(ctx,"ret",fn);
    b.CreateCondBr(b.CreateICmpSGE(c,C), ret, body);              // if (c>=C) return
    b.SetInsertPoint(body);
    Value* idx = b.CreateSExt(b.CreateMul(c,Mn), Type::getInt64Ty(ctx));   // c*M
    Value* rowp= b.CreateGEP(f32, nf, ArrayRef<Value*>{idx});             // &net_flow[c*M]
    Codegen cg{ctx,b,rowp,t,f32,ConstantFP::getNaN(f32)};
    Value* val = cg.gen(ast.get());
    b.CreateStore(val, b.CreateGEP(f32, out, ArrayRef<Value*>{c}));       // out[c] = signal
    b.CreateBr(ret);
    b.SetInsertPoint(ret); b.CreateRetVoid();
    if(verifyFunction(*fn,&errs())){ fprintf(stderr,"bad IR\n"); return 1; }

    // emit PTX for sm_75 (Turing / T4)
    std::string err; auto target=TargetRegistry::lookupTarget(triple,err);
    if(!target){ fprintf(stderr,"no NVPTX target: %s\n",err.c_str()); return 1; }
    TargetOptions opt;
    auto tm = target->createTargetMachine(triple, "sm_75", "+ptx70",
                                          opt, std::optional<Reloc::Model>());
    M->setDataLayout(tm->createDataLayout());
    SmallString<4096> ptx; raw_svector_ostream os(ptx);
    legacy::PassManager pm;
    if(tm->addPassesToEmitFile(pm, os, nullptr, CodeGenFileType::AssemblyFile)){
        fprintf(stderr,"cannot emit PTX\n"); return 1; }
    pm.run(*M);
    fprintf(stderr,"; signal expr: %s   (target sm_75)\n", expr.c_str());
    printf("%s", std::string(ptx.c_str()).c_str());
    return 0;
}
