cmake_minimum_required(VERSION 3.18)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_STANDARD 17)

option(FLIX_USE_KEY_CACHE "Reuse generated benchmark key caches" OFF)
set(FLIX_BENCHMARK_KEY_BITS 31 CACHE STRING "Common FliX benchmark key domain width")
option(FLIX_ENABLE_PROFILING "Enable host profiling instrumentation" OFF)
option(FLIX_BUILD_COMMON_PAPER_SWEEP "Build the selected backend's common paper sweep" OFF)
option(FLIX_BUILD_GPULSMOPT_SUCCESSOR_TEST
       "Build the GPULSMOpt successor correctness test" OFF)
option(FLIX_BUILD_LSMU_SUCCESSOR_TEST
       "Build the LSMu successor correctness and focused timing test" OFF)
option(FLIX_BUILD_LSMU_CLEANUP_RANGE_EXPERIMENT
       "Build the focused LSMu cleanup and range-sum experiment" OFF)

set(FLIX_NEEDS_OPTIX ON)
if(FLIX_BUILD_GPULSMOPT_SUCCESSOR_TEST OR FLIX_BUILD_LSMU_SUCCESSOR_TEST OR
   FLIX_BUILD_LSMU_CLEANUP_RANGE_EXPERIMENT)
  set(FLIX_NEEDS_OPTIX OFF)
elseif(DEFINED IFDEFS AND IFDEFS MATCHES "BASELINES" AND
   IFDEFS MATCHES "(GPULSMOPT|LSM_TREE|GPU_BTREE|HASHTABLE_WARPCORE|HASHTABLE_SLAB|SORTED_ARRAY)")
  set(FLIX_NEEDS_OPTIX OFF)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;120")
endif()

project(index_prototype C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

mark_as_advanced(CUDA_SDK_ROOT_DIR)

set(FLIX_EMBEDDED_PTX)
set(FLIX_OPTIONAL_SOURCES)
set(FLIX_USE_EXTERNAL_NVTX OFF)
set(FLIX_NVTX_FALLBACK_INCLUDE_DIR)

if(FLIX_NEEDS_OPTIX)
  include(cmake/configure_optix.cmake)
  include_directories(${OptiX_INCLUDE})
  add_definitions(-DFLIX_NEEDS_OPTIX)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ext/nvtx/nvtxImportedTargets.cmake")
  include(${CMAKE_CURRENT_SOURCE_DIR}/../ext/nvtx/nvtxImportedTargets.cmake)
  set(FLIX_USE_EXTERNAL_NVTX ON)
else()
  set(FLIX_NVTX_FALLBACK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nvtx_stub")
endif()

if(DEFINED IFDEFS AND (IFDEFS MATCHES "GPU_BTREE" OR FLIX_NEEDS_OPTIX))
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/gpu_btree)
endif()

#SET(IFDEFS "" CACHE STRING "ifdefs")
#add_compile_definitions(foo)
#add_definitions("-DCOMBINE_INSERT_DELETE")
add_definitions(${IFDEFS})  

if(FLIX_NEEDS_OPTIX)
  cuda_compile_and_embed(fine_granular_embedded_ptx_code device_code_fine_granular.cu)
  cuda_compile_and_embed(coarse_granular_embedded_ptx_code device_code_coarse_granular.cu)
  cuda_compile_and_embed(coarse_granular_embedded_updates_ptx_code device_code_coarse_granular_updates.cu)
  cuda_compile_and_embed(rtscan_embedded_ptx_code device_code_rtscan.cu)
  set(FLIX_EMBEDDED_PTX
          ${fine_granular_embedded_ptx_code}
          ${coarse_granular_embedded_ptx_code}
          ${coarse_granular_embedded_updates_ptx_code}
          ${rtscan_embedded_ptx_code})
  set(FLIX_OPTIONAL_SOURCES
          optix_pipeline.cu
          optix_pipeline.cuh
          optix_wrapper.cu
          optix_wrapper.cuh)
endif()

add_executable(index_prototype
        ${FLIX_EMBEDDED_PTX}
        benchmarks.cuh
        benchmarks_updates.cuh
        benchmarks_lookups.cuh
        build_coarse_granular.cuh
        cuda_buffer.cuh
        debug_definitions_updates.cuh
        definitions.cuh
        definitions_coarse_granular.cuh
        definitions_fine_granular.cuh
        definitions_updates.cuh
        impl_binsearch.cuh
        impl_cg_rtx_index.cuh
        impl_cg_rtx_index_updates.cuh
        impl_hashtable_warpcore.cuh
        impl_hashtable_slab.cuh
        impl_rtx_index.cuh
        impl_tree_awad.cuh
        impl_opt_sorted_array.cuh
        #impl_lsm_tree.cuh
        impl_gpulsmopt.cuh
        impl_scan.cuh
        input_generation.h
        launch_parameters.cuh
        main.cu
        ${FLIX_OPTIONAL_SOURCES}
        result_collector.h
        utilities.cuh
        impl_opt_binsearch.cuh
)

function(configure_paper_sweep_target target_name)
  set_target_properties(${target_name} PROPERTIES
          CUDA_SEPARABLE_COMPILATION ON
          CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_options(${target_name} PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:
          --generate-line-info --extended-lambda
          --use_fast_math --expt-relaxed-constexpr -O3
          -Xcompiler=-fopenmp>)
  target_link_libraries(${target_name} PRIVATE
          CUDA::cudart OpenMP::OpenMP_CXX)
  if(FLIX_NVTX_FALLBACK_INCLUDE_DIR)
    target_include_directories(${target_name} PRIVATE
            "${FLIX_NVTX_FALLBACK_INCLUDE_DIR}")
  endif()
endfunction()

if(FLIX_BUILD_COMMON_PAPER_SWEEP)
  add_executable(paper_sweep paper_lsmu_sweep.cu
                 ${FLIX_EMBEDDED_PTX} ${FLIX_OPTIONAL_SOURCES})
  target_compile_definitions(paper_sweep PRIVATE PAPER_SWEEP_COMMON
          UNSORTED_PROBES_CHECKS FLIX_COMPLETE_UNSORTED_LOOKUP)
  configure_paper_sweep_target(paper_sweep)
  if(FLIX_USE_EXTERNAL_NVTX)
    target_link_libraries(paper_sweep PRIVATE nvtx3-cpp)
  endif()
  if(FLIX_NEEDS_OPTIX)
    target_link_libraries(paper_sweep PRIVATE CUDA::cuda_driver)
  endif()
endif()

if(FLIX_BUILD_GPULSMOPT_SUCCESSOR_TEST)
  enable_testing()
  add_executable(gpulsmopt_successor_test
          ../tests/gpulsmopt_successor_test.cu)
  target_include_directories(gpulsmopt_successor_test PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}/..")
  configure_paper_sweep_target(gpulsmopt_successor_test)
  add_test(NAME gpulsmopt_successor_correctness
           COMMAND gpulsmopt_successor_test)
endif()

if(FLIX_BUILD_LSMU_SUCCESSOR_TEST)
  enable_testing()
  add_executable(lsmu_successor_test
          ../tests/lsmu_successor_test.cu)
  target_include_directories(lsmu_successor_test PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}")
  configure_paper_sweep_target(lsmu_successor_test)
  add_test(NAME lsmu_successor_correctness
           COMMAND lsmu_successor_test)
endif()

if(FLIX_BUILD_LSMU_CLEANUP_RANGE_EXPERIMENT)
  add_executable(lsmu_cleanup_range_experiment
          ../tests/lsmu_cleanup_range_experiment.cu)
  target_include_directories(lsmu_cleanup_range_experiment PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}")
  configure_paper_sweep_target(lsmu_cleanup_range_experiment)
endif()

set_target_properties(index_prototype PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

if(NOT DEFINED IFDEFS OR NOT IFDEFS MATCHES "FLIX_USE_KEY_CACHE")
  if(FLIX_USE_KEY_CACHE)
    target_compile_definitions(index_prototype PRIVATE FLIX_USE_KEY_CACHE=1)
  else()
    target_compile_definitions(index_prototype PRIVATE FLIX_USE_KEY_CACHE=0)
  endif()
endif()

# 75 TURING
# 80 AMPERE
# 86 AMPERE
# 89 ADA
# 90 HOPPER
# 120 BLACKWELL

target_compile_options(index_prototype PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra -Wpedantic -Werror
        ${OpenMP_CXX_FLAGS}
        >)

target_compile_options(index_prototype PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        #-funwind-tables -fno-omit-frame-pointer
        --extended-lambda --use_fast_math --expt-relaxed-constexpr
        -O3
        -Xcompiler=-fopenmp
#        -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wpedantic
        >)

target_compile_definitions(index_prototype PRIVATE
        UNSORTED_PROBES_CHECKS FLIX_COMPLETE_UNSORTED_LOOKUP
        FLIX_BENCHMARK_KEY_BITS=${FLIX_BENCHMARK_KEY_BITS})
if(FLIX_ENABLE_PROFILING)
  target_compile_options(index_prototype PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-g;-shared-libgcc;-fno-omit-frame-pointer>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--debug;--profile>")
endif()


target_link_libraries(index_prototype PRIVATE
        CUDA::cudart
        OpenMP::OpenMP_CXX
        )

if(FLIX_NVTX_FALLBACK_INCLUDE_DIR)
  target_include_directories(index_prototype PRIVATE "${FLIX_NVTX_FALLBACK_INCLUDE_DIR}")
endif()

if(FLIX_NEEDS_OPTIX)
  target_link_libraries(index_prototype PRIVATE
          CUDA::cuda_driver
          )
endif()

if(FLIX_USE_EXTERNAL_NVTX)
  target_link_libraries(index_prototype PRIVATE nvtx3-cpp)
endif()
