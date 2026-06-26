// =============================================================================
// File: impl_binsearch.cuh
// Author: Justus Henneberg
// Description: Implements impl_binsearch     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef BINSEARCH_INDEX_H
#define BINSEARCH_INDEX_H

#include "definitions.cuh"
#include "cuda_buffer.cuh"
#include "utilities.cuh"


// for nvtx
struct nvtx_sorted_array_domain{ static constexpr char const* name{"sorted_array"}; };


template <typename element_type>
DEVICEQUALIFIER INLINEQUALIFIER
smallsize impl_device_binary_search(element_type key, const element_type* buf, smallsize size) {
    smallsize match_index = 0;
    for (smallsize skip = smallsize(1u) << 30u; skip != 0; skip >>= 1u) {
        if (match_index + skip >= size)
            continue;

        if (buf[match_index + skip] <= key)
            match_index += skip;
    }
    return match_index;
}


template <typename element_type>
DEVICEQUALIFIER INLINEQUALIFIER
smallsize impl_reverse_device_binary_search(element_type key, const element_type* buf, smallsize size) {
    smallsize match_index = size - 1;
    for (smallsize skip = smallsize(1u) << 30u; skip != 0; skip >>= 1u) {
        if (match_index < skip)
            continue;

        if (buf[match_index - skip] >= key)
            match_index -= skip;
    }
    return match_index;
}


template <typename key_type>
GLOBALQUALIFIER
void binsearch_lookup_kernel(const key_type* sorted_keys, const smallsize* sorted_offsets, smallsize stored_size, const key_type* keys, smallsize* result, smallsize size) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    key_type key = keys[tid];

    auto match_index = reverse_device_binary_search(key, sorted_keys, stored_size);
    if (sorted_keys[match_index] == key) {
        result[tid] = sorted_offsets[match_index];
    } else {
        result[tid] = not_found;
    }
}


template <typename key_type>
GLOBALQUALIFIER
void binsearch_range_lookup_kernel(const key_type* sorted_keys, const smallsize* sorted_offsets, smallsize stored_size, const key_type* lower, const key_type* upper, smallsize* result, smallsize size) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    key_type lower_bound = lower[tid];
    key_type upper_bound = upper[tid];
    auto lower_index = reverse_device_binary_search(lower_bound, sorted_keys, stored_size);

    smallsize agg = 0;
    for (size_t it = lower_index; it < stored_size; ++it) {
        if (sorted_keys[it] < lower_bound || sorted_keys[it] > upper_bound)
            break;
        agg += sorted_offsets[it];
    }
    result[tid] = agg;
}


template <typename key_type_>
class sorted_array {
public:
    using key_type = key_type_;

private:
    cuda_buffer<key_type> sorted_keys_buffer;
    cuda_buffer<smallsize> sorted_offsets_buffer;
    size_t stored_size = 0;

public:
    static constexpr bool can_lookup = true;
    static constexpr bool can_multi_lookup = true;
    static constexpr bool can_range_lookup = true;
    static constexpr bool can_update = false;

    static std::string short_description() {
        return std::string("sorted_array");
    }

    static size_t estimate_build_bytes(size_t size) {
        size_t sort_bytes = (sizeof(smallsize) + sizeof(key_type)) * size;
        size_t sort_aux_bytes = sizeof(smallsize) * size + find_pair_sort_buffer_size<key_type, smallsize>(size);
        return sort_bytes + sort_aux_bytes;
    }

    size_t gpu_resident_bytes() {
        return sorted_keys_buffer.size_in_bytes() + sorted_offsets_buffer.size_in_bytes();
    }

    void build(const key_type* keys, size_t size, double* build_time_ms, size_t* build_bytes) {
        cuda_buffer<uint8_t> temp_buffer;
        cuda_buffer<smallsize> offsets_buffer;

        sorted_keys_buffer.alloc(size);
        offsets_buffer.alloc(size);
        sorted_offsets_buffer.alloc(size);
        init_offsets(offsets_buffer, size, build_time_ms);

        cudaDeviceSynchronize(); CUERR

        size_t temp_storage_bytes = find_pair_sort_buffer_size<key_type, smallsize>(size);
        temp_buffer.alloc(temp_storage_bytes);
        timed_pair_sort(
            temp_buffer.raw_ptr, temp_storage_bytes,
            keys, sorted_keys_buffer.ptr(), offsets_buffer.ptr(), sorted_offsets_buffer.ptr(), size, build_time_ms);

        if (build_bytes) *build_bytes += sorted_keys_buffer.size_in_bytes() + sorted_offsets_buffer.size_in_bytes() + temp_buffer.size_in_bytes() + offsets_buffer.size_in_bytes();

        stored_size = size;

        cudaDeviceSynchronize(); CUERR
    }

    void lookup(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {

        nvtx3::scoped_range_in<nvtx_sorted_array_domain> launch{"launch"};
        binsearch_lookup_kernel<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
                sorted_keys_buffer.ptr(),
                sorted_offsets_buffer.ptr(),
                stored_size,
                keys,
                result,
                size
        );
    }

    void range_lookup_sum(const key_type* lower, const key_type* upper, smallsize* result, size_t size, cudaStream_t stream) {

        nvtx3::scoped_range_in<nvtx_sorted_array_domain> launch{"launch"};
        binsearch_range_lookup_kernel<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
                sorted_keys_buffer.ptr(),
                sorted_offsets_buffer.ptr(),
                stored_size,
                lower,
                upper,
                result,
                size
        );
    }

    void multi_lookup_sum(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        range_lookup_sum(keys, keys, result, size, stream);
    }

    void destroy() {
        sorted_keys_buffer.free();
        sorted_offsets_buffer.free();
        stored_size = 0;
    }

    void insert(const key_type* update_list, const smallsize* offsets, size_t size, cudaStream_t stream) {}
    void remove(const key_type* update_list, size_t size, cudaStream_t stream) {}
};

#endif
