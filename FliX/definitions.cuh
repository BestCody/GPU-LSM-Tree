// =============================================================================
// File: definitions.cuh
// Author: Justus Henneberg
// Description: Implements definitions     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef DEFINITIONS_CUH
#define DEFINITIONS_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>

#if !defined(GPULSMOPT)
#include <optix.h>
#include <optix_stubs.h>
#include "../ext/cudahelpers/cuda_helpers.cuh"
#else
#ifndef HOSTDEVICEQUALIFIER
#define HOSTDEVICEQUALIFIER __host__ __device__
#endif
#ifndef HOSTQUALIFIER
#define HOSTQUALIFIER __host__
#endif
#ifndef DEVICEQUALIFIER
#define DEVICEQUALIFIER __device__
#endif
#ifndef GLOBALQUALIFIER
#define GLOBALQUALIFIER __global__
#endif
#ifndef INLINEQUALIFIER
#define INLINEQUALIFIER __forceinline__
#endif
#ifndef SDIV
#define SDIV(x, y) (((x) + (y) - 1) / (y))
#endif
#ifndef MAXBLOCKSIZE
#define MAXBLOCKSIZE 256
#endif
DEVICEQUALIFIER INLINEQUALIFIER size_t global_thread_id() {
    return blockIdx.x * size_t(blockDim.x) + threadIdx.x;
}
#endif


// rethrow cuda error code as c++ exception
#define C2EX {                                                             \
    cudaError_t err;                                                       \
    if ((err = cudaGetLastError()) != cudaSuccess) {                       \
        std::stringstream error;                                           \
        error << cudaGetErrorString(err) << " (" << __FILE__ << ", " << __LINE__ << ")"; \
        throw std::runtime_error(error.str());                                   \
    }                                                                      \
}

#ifndef CUERR
#define CUERR C2EX
#endif


enum class operation_support {
    none = 0, async = 1, sync = 2
};


constexpr size_t seed = 42;

using key8 = uint8_t;
using key16 = uint16_t;
using key32 = uint32_t;
using key64 = uint64_t;
using smallsize = uint32_t;
using bigsize = uint64_t;

constexpr smallsize not_found = std::numeric_limits<smallsize>::max();

// never generate negative keys (conflict with b+ tree)
// never generate key 0 (conflict with b+ tree)
// never generate MAX_KEY (conflict with b+ tree and hash table)
// never generate MAX_KEY - 1 (conflict with hash table)
// also reserve 1 and MAX_KEY - 2 to test out-of-range lookups
// if the bit range is restricted, reserve (1 << bits) - 1 for out-of-range lookups instead
template <typename key_type>
constexpr key_type min_usable_key(uint8_t bit_restriction) {
    return 2;
}
template <typename key_type>
constexpr key_type min_usable_key() {
    return min_usable_key<key_type>(sizeof(key_type) * 8);
}

template <typename key_type>
constexpr key_type max_usable_key(uint8_t bit_restriction) {
    return bit_restriction < sizeof(key_type) * 8
        ? (key_type(1) << bit_restriction) - 2
        : std::numeric_limits<key_type>::max() - 3;
}
template <typename key_type>
constexpr key_type max_usable_key() {
    return max_usable_key<key_type>(sizeof(key_type) * 8);
}

#if !defined(GPULSMOPT)
#define OPTIX_CHECK(call)                                               \
  {                                                                     \
    OptixResult res = call;                                             \
    if (res != OPTIX_SUCCESS)                                            \
      {                                                                 \
        fprintf(stderr, "OptiX (%s) failed with code %d (line %d)\n", #call, res, __LINE__); \
        exit(2);                                                        \
      }                                                                 \
  }
#endif

#endif
