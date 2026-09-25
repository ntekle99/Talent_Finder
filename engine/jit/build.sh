#!/bin/bash
# Build the LLVM-JIT signal compiler. Requires: brew install llvm
set -e
LC="$(brew --prefix llvm)/bin/llvm-config"
cd "$(dirname "$0")"
# Use LLVM's include dir but keep our own exceptions/RTTI (llvm-config --cxxflags forces
# -fno-exceptions/-fno-rtti which breaks our try/catch).
clang++ -std=c++17 -O2 -fexceptions \
  -I"$($LC --includedir)" -I../include \
  jit_backtest.cpp \
  $($LC --ldflags --system-libs --libs core orcjit native passes) \
  -Wl,-rpath,"$($LC --libdir)" \
  -o jit_backtest
echo "built engine/jit/jit_backtest"
