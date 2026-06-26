// =============================================================================
// File: total_keys_optimized.cuh
// Author: Rosina Kharal
// Description: Implements total_keys_optimized to sum the total keys in the DS as a correctness check after updates.    
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

// File: count_total_keys.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include "cuda_buffer.h"
#include <cooperative_groups.h>

typedef uint16_t smallsize;

#define GLOBALQUALIFIER __global__
#define DEVICEQUALIFIER __device__
#define CUERR { cudaError_t err; if ((err = cudaGetLastError()) != cudaSuccess) { \
    printf("CUDA Error: %s\n", cudaGetErrorString(err)); return; }}

////constexpr int MAXBLOCKSIZE = 1024;
//constexpr int TILE_SIZE = 32;
//constexpr int WARP_SIZE = 32;


namespace cg {
    template<typename T>
    __device__ __forceinline__ T extract(const void* ptr, size_t offset) {
        return *(reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(ptr) + offset));
    }
}

namespace coop_g = cooperative_groups;

// Optimized reduction using warp shuffle
__device__ __forceinline__ size_t warp_reduce_sum(size_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel to reduce key counts into one total
GLOBALQUALIFIER void reduce_total_keys_kernel(
    const smallsize* keys_per_bucket,
    size_t partition_count_with_overflow,
    size_t* total_keys)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    size_t my_sum = 0;
    if (i < partition_count_with_overflow)
        my_sum = keys_per_bucket[i];

    my_sum = warp_reduce_sum(my_sum);

    if ((tid & 31) == 0) {
        atomicAdd(total_keys, my_sum);
    }
}

// Host function to compute total keys
template <typename key_type>
void compute_total_keys_in_structure(
    void *node_buffer,
    void *allocation_buffer,
    smallsize node_size,
    smallsize node_stride,
    smallsize allocation_buffer_count,
    smallsize partition_count_with_overflow,
    cudaStream_t stream,
    cuda_buffer<size_t>& d_total_keys)
{
    cuda_buffer<smallsize> d_keys_per_bucket;
    d_keys_per_bucket.alloc(partition_count_with_overflow);
    d_keys_per_bucket.zero();

    int threadsPerBlock = MAXBLOCKSIZE;
    int numBlocks = (partition_count_with_overflow + threadsPerBlock - 1) / threadsPerBlock;

    count_keys_per_bucket_kernel<key_type><<<numBlocks, threadsPerBlock, 0, stream>>>(
        node_buffer, allocation_buffer,
        node_size, node_stride,
        allocation_buffer_count, partition_count_with_overflow,
        d_keys_per_bucket.ptr());

    d_total_keys.zero(stream);

    reduce_total_keys_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_keys_per_bucket.ptr(),
        partition_count_with_overflow,
        d_total_keys.ptr());

    cudaStreamSynchronize(stream);
    CUERR;
}
