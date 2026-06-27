cmake_minimum_required(VERSION 3.18)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_STANDARD 17)

set(FLIX_NEEDS_OPTIX ON)
if(DEFINED IFDEFS AND IFDEFS MATCHES "BASELINES" AND
   IFDEFS MATCHES "(GPULSMOPT|LSM_TREE|GPU_BTREE|HASHTABLE_WARPCORE|HASHTABLE_SLAB)")
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
  include_directories(../btree)
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

set_target_properties(index_prototype PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
# 75 TURING
# 80 AMPERE
# 86 AMPERE
# 89 ADA
# 90 HOPPER
# 120 BLACKWELL

target_compile_options(index_prototype PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
        # profiling
        -g -shared-libgcc -fno-omit-frame-pointer
        -Wall -Wextra -Wpedantic -Werror
        ${OpenMP_CXX_FLAGS}
        >)

target_compile_options(index_prototype PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        # profiling
        --debug --profile --generate-line-info
        #-funwind-tables -fno-omit-frame-pointer
        --extended-lambda --use_fast_math --expt-relaxed-constexpr
        -O3
        -Xcompiler=-fopenmp
#        -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wpedantic
        >)



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
