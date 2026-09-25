#!/bin/bash
# Build the NVPTX signal->PTX emitter. Requires: brew install llvm
set -e
LC="$(brew --prefix llvm)/bin/llvm-config"
cd "$(dirname "$0")"
clang++ -std=c++17 -O2 -fexceptions \
  -I"$($LC --includedir)" -I../include \
  emit_ptx.cpp \
  $($LC --ldflags --system-libs --libs core native nvptxcodegen nvptxinfo mc target) \
  -Wl,-rpath,"$($LC --libdir)" \
  -o emit_ptx
echo "built engine/jit/emit_ptx"
