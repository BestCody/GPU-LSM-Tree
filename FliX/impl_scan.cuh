// =============================================================================
// File: impl_scan.cuh
// Author: Justus Henneberg
// Description: Implements impl_scan     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef SCAN_BASELINE_H
#define SCAN_BASELINE_H

#include <cooperative_groups.h>

#include "definitions.cuh"
#include "cuda_buffer.cuh"
#include "utilities.cuh"


template <typename key_type>
GLOBALQUALIFIER
void naive_point_lookup_scan(const key_type* stored_keys, smallsize stored_size, const key_type* keys, smallsize* result, smallsize size) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    key_type key = keys[tid];
    for (smallsize i = 0; i < stored_size; ++i) {
        if (key == stored_keys[i]) {
            result[tid] = i;
            return;
        }
    }
    result[tid] = not_found;
}


template <typename key_type>
GLOBALQUALIFIER
void naive_range_lookup_scan(const key_type* stored_keys, smallsize stored_size, const key_type* lower, const key_type* upper, smallsize* result, smallsize size) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    key_type l = lower[tid];
    key_type u = upper[tid];

    smallsize agg = 0;
    for (smallsize i = 0; i < stored_size; ++i) {
        if (l <= stored_keys[i] && stored_keys[i] <= u) {
            agg += i;
        }
    }
    result[tid] = agg;
}


template <typename key_type_, uint8_t tile_size_log = 0>
class scan {
public:
    using key_type = key_type_;

private:
    const key_type* stored_keys = nullptr;
    size_t stored_size = 0;

    constexpr static size_t cg_size_log = 4;

public:
    static constexpr bool can_lookup = true;
    static constexpr bool can_multi_lookup = true;
    static constexpr bool can_range_lookup = true;
    static constexpr bool can_update = false;

    static std::string short_description() {
        return std::string("scan_") + std::to_string(tile_size_log);
    }

    static size_t estimate_build_bytes(size_t size) {
        return 0;
    }

    size_t gpu_resident_bytes() {
        return 0;
    }

    void build(const key_type* keys, size_t size, double* build_time_ms, size_t* build_bytes) {
        stored_keys = keys;
        stored_size = size;
    }

    void lookup(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        if constexpr (tile_size_log == 0) {
            naive_point_lookup_scan<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(stored_keys, stored_size, keys, result, size);
        } else {
            uint8_t cg_size = 1u << tile_size_log;
            //coop_point_lookup_scan<key_type, cg_size><<<SDIV(size << cg_size_log, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(stored_keys, stored_size, keys, result, size);
        }
    }

    void multi_lookup_sum(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        if constexpr (tile_size_log == 0) {
            naive_range_lookup_scan<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(stored_keys, stored_size, keys, keys, result, size);
        } else {
            //coop_range_lookup_scan<key_type, cg_size><<<SDIV(size << cg_size_log, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(stored_keys, stored_size, keys, keys, result, size);
        }
    }

    void range_lookup_sum(const key_type* lower, const key_type* upper, smallsize* result, size_t size, cudaStream_t stream) {
        if constexpr (tile_size_log == 0) {
            naive_range_lookup_scan<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(stored_keys, stored_size, lower, upper, result, size);
        } else {
            //coop_range_lookup_scan<key_type, cg_size><<<SDIV(size << cg_size_log, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(stored_keys, stored_size, lower, upper, result, size);
        }
    }

    void destroy() {
        stored_keys = nullptr;
        stored_size = 0;
    }

    void insert(const key_type* update_list, const smallsize* offsets, size_t size, cudaStream_t stream) {}
    void remove(const key_type* update_list, size_t size, cudaStream_t stream) {}
};

#endif
