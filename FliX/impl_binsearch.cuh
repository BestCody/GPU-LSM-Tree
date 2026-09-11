// =============================================================================
// File: impl_binsearch.cuh
// Author: Justus Henneberg
// Description: Implements impl_binsearch     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef BINSEARCH_INDEX_H
#define BINSEARCH_INDEX_H

#include <iostream>
#include "definitions.cuh"
#include "cuda_buffer.cuh"
#include "utilities.cuh"
#include <nvtx3/nvtx3.hpp>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <cuda/std/functional>
#include <limits>
#include <stdexcept>
#include <string>


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

    if (stored_size == 0) {
        result[tid] = not_found;
        return;
    }

    key_type key = keys[tid];

    auto match_index = impl_reverse_device_binary_search(key, sorted_keys, stored_size);
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

    if (stored_size == 0 || lower[tid] > upper[tid]) {
        result[tid] = 0;
        return;
    }

    key_type lower_bound = lower[tid];
    key_type upper_bound = upper[tid];
    auto lower_index = impl_reverse_device_binary_search(lower_bound, sorted_keys, stored_size);

    smallsize agg = 0;
    for (size_t it = lower_index; it < stored_size; ++it) {
        if (sorted_keys[it] < lower_bound || sorted_keys[it] > upper_bound)
            break;
        agg += sorted_offsets[it];
    }
    result[tid] = agg;
}


template <typename key_type>
struct binsearch_keep_record {
    const key_type* deleted;
    smallsize count;

    template <typename Pair>
    DEVICEQUALIFIER bool operator()(const Pair& record) const {
        const key_type key = thrust::get<0>(record);
        smallsize lo = 0, hi = count;
        while (lo < hi) {
            const smallsize mid = lo + (hi - lo) / 2;
            if (deleted[mid] < key) lo = mid + 1;
            else hi = mid;
        }
        return lo == count || deleted[lo] != key;
    }
};

GLOBALQUALIFIER void binsearch_init_offsets(smallsize* offsets, smallsize count) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) offsets[i] = i;
}

template <typename key_type_>
class sorted_array {
public:
    using key_type = key_type_;

private:
    cuda_buffer<key_type> sorted_keys_buffer;
    cuda_buffer<smallsize> sorted_offsets_buffer;
    cuda_buffer<key_type> output_keys_buffer, batch_keys_buffer;
    cuda_buffer<smallsize> output_offsets_buffer, batch_offsets_buffer;
    cuda_buffer<uint8_t> update_temp_buffer;
    cuda_buffer<smallsize> selected_count_buffer;
    size_t stored_size = 0;

    static void check(cudaError_t status) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("Sorted array: ") + cudaGetErrorString(status));
    }

    static int checked_count(size_t size) {
        if (size > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("Sorted array exceeds CUB's signed 32-bit item limit");
        return static_cast<int>(size);
    }

    // Only scratch/output buffers grow; their old contents are disposable.
    template <typename T>
    static void reserve(cuda_buffer<T>& buffer, size_t required) {
        if (required <= buffer.num_elements) return;
        const size_t limit = std::numeric_limits<size_t>::max() / sizeof(T);
        if (required > limit) throw std::overflow_error("Sorted-array allocation overflow");
        const size_t doubled = buffer.num_elements > limit / 2 ? limit : 2 * buffer.num_elements;
        cuda_buffer<T> replacement;
        replacement.num_elements = std::max(required, doubled);
        check(cudaMalloc(&replacement.raw_ptr, replacement.size_in_bytes()));
        buffer.swap(replacement);
    }

public:
    static constexpr operation_support can_lookup = operation_support::async;
    static constexpr operation_support can_multi_lookup = operation_support::async;
    static constexpr operation_support can_range_lookup = operation_support::async;
    static constexpr bool range_enumerates_records = true;
    static constexpr operation_support can_update = operation_support::async;
    static constexpr operation_support can_successor = operation_support::none;

    static std::string short_description() {
        return std::string("sorted_array");
    }

    static parameters_type parameters() {
        return {{"insert", "radix_sort_merge"},
                {"delete", "sorted_membership_select"},
                {"duplicate_keys", "retain"}};
    }

    static size_t estimate_build_bytes(size_t size) {
        checked_count(size);
        if (size == 0) return 0;
        size_t sort_bytes = (sizeof(smallsize) + sizeof(key_type)) * size;
        size_t sort_aux_bytes = sizeof(smallsize) * size + find_pair_sort_buffer_size<key_type, smallsize>(size);
        return sort_bytes + sort_aux_bytes;
    }

    size_t gpu_resident_bytes() {
        return sorted_keys_buffer.size_in_bytes() + sorted_offsets_buffer.size_in_bytes()
            + output_keys_buffer.size_in_bytes() + output_offsets_buffer.size_in_bytes()
            + batch_keys_buffer.size_in_bytes() + batch_offsets_buffer.size_in_bytes()
            + update_temp_buffer.size_in_bytes() + selected_count_buffer.size_in_bytes();
    }

    void build(const key_type* keys, size_t size, double* build_time_ms, size_t* build_bytes) {
        const int count = checked_count(size);
        if (size && !keys) throw std::invalid_argument("Sorted-array build has no keys");
        scoped_cuda_timer timer(0, build_time_ms);
        destroy();
        if (size == 0) return;
        cuda_buffer<uint8_t> temp_buffer;
        cuda_buffer<smallsize> offsets_buffer;
        reserve(sorted_keys_buffer, size);
        reserve(sorted_offsets_buffer, size);
        reserve(offsets_buffer, size);
        binsearch_init_offsets<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(offsets_buffer.ptr(), count);
        check(cudaGetLastError());
        size_t bytes = 0;
        auto sort = [&](void* temp, size_t& capacity) {
            return cub::DeviceRadixSort::SortPairs(temp, capacity, keys, sorted_keys_buffer.ptr(),
                offsets_buffer.ptr(), sorted_offsets_buffer.ptr(), count);
        };
        check(sort(nullptr, bytes));
        reserve(temp_buffer, std::max<size_t>(bytes, 1));
        check(sort(temp_buffer.ptr(), bytes));
        stored_size = size;
        if (build_bytes) *build_bytes += gpu_resident_bytes() + temp_buffer.size_in_bytes() + offsets_buffer.size_in_bytes();
        check(cudaStreamSynchronize(0));
    }

    void build(const key_type* keys, size_t size, size_t max_size, size_t available_memory_bytes,
               double* build_time_ms, size_t* build_bytes) {
        (void)max_size;
        (void)available_memory_bytes;
        build(keys, size, build_time_ms, build_bytes);
    }

    void lookup(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        checked_count(size);
        if (size == 0) return;
        nvtx3::scoped_range_in<nvtx_sorted_array_domain> launch{"launch"};
        binsearch_lookup_kernel<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
                sorted_keys_buffer.ptr(),
                sorted_offsets_buffer.ptr(),
                stored_size,
                keys,
                result,
                size
        );
        check(cudaGetLastError());
    }

    void range_lookup_sum(const key_type* lower, const key_type* upper, smallsize* result, size_t size, cudaStream_t stream) {
        checked_count(size);
        if (size == 0) return;
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
        check(cudaGetLastError());
    }

    void multi_lookup_sum(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        range_lookup_sum(keys, keys, result, size, stream);
    }

    void destroy() {
        sorted_keys_buffer.free();
        sorted_offsets_buffer.free();
        output_keys_buffer.free();
        output_offsets_buffer.free();
        batch_keys_buffer.free();
        batch_offsets_buffer.free();
        update_temp_buffer.free();
        selected_count_buffer.free();
        stored_size = 0;
    }

    void insert(const key_type* update_list, const smallsize* offsets, size_t size, cudaStream_t stream) {
        const int count = checked_count(size);
        checked_count(stored_size + size);
        if (size == 0) return;
        if (!update_list) throw std::invalid_argument("Sorted-array insert has no keys");
        reserve(batch_keys_buffer, size);
        reserve(batch_offsets_buffer, size);
        reserve(output_keys_buffer, stored_size + size);
        reserve(output_offsets_buffer, stored_size + size);
        if (!offsets) {
            binsearch_init_offsets<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
                output_offsets_buffer.ptr(), count);
            check(cudaGetLastError());
            offsets = output_offsets_buffer.ptr();
        }
        auto sort = [&](void* temp, size_t& bytes) {
            return cub::DeviceRadixSort::SortPairs(temp, bytes,
                update_list, batch_keys_buffer.ptr(), offsets, batch_offsets_buffer.ptr(),
                count, 0, sizeof(key_type) * 8, stream);
        };
        auto merge = [&](void* temp, size_t& bytes) {
            return cub::DeviceMerge::MergePairs(temp, bytes,
                sorted_keys_buffer.ptr(), sorted_offsets_buffer.ptr(), static_cast<int>(stored_size),
                batch_keys_buffer.ptr(), batch_offsets_buffer.ptr(), count,
                output_keys_buffer.ptr(), output_offsets_buffer.ptr(), cuda::std::less<key_type>{}, stream);
        };
        size_t sort_bytes = 0, merge_bytes = 0;
        check(sort(nullptr, sort_bytes));
        check(merge(nullptr, merge_bytes));
        reserve(update_temp_buffer, std::max<size_t>({sort_bytes, merge_bytes, 1}));
        check(sort(update_temp_buffer.ptr(), sort_bytes));
        check(merge(update_temp_buffer.ptr(), merge_bytes));
        sorted_keys_buffer.swap(output_keys_buffer);
        sorted_offsets_buffer.swap(output_offsets_buffer);
        stored_size += size;
    }

    void remove(const key_type* update_list, size_t size, cudaStream_t stream) {
        const int count = checked_count(size);
        if (size == 0 || stored_size == 0) return;
        if (!update_list) throw std::invalid_argument("Sorted-array removal has no keys");
        reserve(batch_keys_buffer, size);
        reserve(output_keys_buffer, stored_size);
        reserve(output_offsets_buffer, stored_size);
        reserve(selected_count_buffer, 1);
        auto input = thrust::make_zip_iterator(thrust::make_tuple(sorted_keys_buffer.ptr(), sorted_offsets_buffer.ptr()));
        auto output = thrust::make_zip_iterator(thrust::make_tuple(output_keys_buffer.ptr(), output_offsets_buffer.ptr()));
        const binsearch_keep_record<key_type> keep{batch_keys_buffer.ptr(), static_cast<smallsize>(size)};
        auto sort = [&](void* temp, size_t& bytes) {
            return cub::DeviceRadixSort::SortKeys(temp, bytes, update_list, batch_keys_buffer.ptr(),
                count, 0, sizeof(key_type) * 8, stream);
        };
        auto select = [&](void* temp, size_t& bytes) {
            return cub::DeviceSelect::If(temp, bytes, input, output, selected_count_buffer.ptr(),
                static_cast<int>(stored_size), keep, stream);
        };
        size_t sort_bytes = 0, select_bytes = 0;
        check(sort(nullptr, sort_bytes));
        check(select(nullptr, select_bytes));
        reserve(update_temp_buffer, std::max<size_t>({sort_bytes, select_bytes, 1}));
        check(sort(update_temp_buffer.ptr(), sort_bytes));
        check(select(update_temp_buffer.ptr(), select_bytes));
        smallsize survivors = 0;
        check(cudaMemcpyAsync(&survivors, selected_count_buffer.ptr(), sizeof(survivors), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        sorted_keys_buffer.swap(output_keys_buffer);
        sorted_offsets_buffer.swap(output_offsets_buffer);
        stored_size = survivors;
    }
};

#endif
