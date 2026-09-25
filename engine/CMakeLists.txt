cmake_minimum_required(VERSION 3.18)
project(talent_backtest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Optional OpenMP for the CPU sweep.
find_package(OpenMP QUIET)

# Detect CUDA. On a Brev GPU box this enables the .cu kernel; on macOS it falls back to CPU.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  message(STATUS "CUDA found -> GPU sweep kernel")
  add_executable(talent_backtest src/main.cpp kernels/cuda_backtest.cu)
  set_target_properties(talent_backtest PROPERTIES CUDA_ARCHITECTURES "70;80;90")
  target_compile_definitions(talent_backtest PRIVATE TF_CUDA=1)
else()
  message(STATUS "no CUDA -> CPU reference sweep")
  add_executable(talent_backtest src/main.cpp kernels/cpu_reference.cpp)
endif()

target_include_directories(talent_backtest PRIVATE include)
if(OpenMP_CXX_FOUND)
  target_link_libraries(talent_backtest PRIVATE OpenMP::OpenMP_CXX)
endif()

enable_testing()
add_executable(tf_tests tests/test_backtest.cpp kernels/cpu_reference.cpp)
target_include_directories(tf_tests PRIVATE include)
add_test(NAME tf_tests COMMAND tf_tests)
