// =============================================================================
// File: impl_lsm_tree.cuh
// Author: Justus Henneberg
// Description: Implements impl_lsm_tree     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef IMPL_LSM_TREE_CUH
#define IMPL_LSM_TREE_CUH

#include "definitions_coarse_granular.cuh"
#include "definitions_opt.cuh"
#include "memory_layout.cuh"

#include <cmath>

#include <cub/cub.cuh>

// namespace cub {
#include <cub/device/device_merge.cuh>
#include "device_binary_search.cuh"

// file: lsm_debug_print.cuh
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include "lsm_debug_tree.cuh"

template <typename key_type, smallsize chunk_size, smallsize max_cached_levels>
__global__ void lsm_next_larger_kernel(
    const key_type *level_keys,
    const smallsize *level_values,
    const key_type *staged_keys,
    const smallsize *staged_values,
    const key_type *query_keys,
    key_type *results,
    smallsize size,
    smallsize level_count,
    smallsize inserted_chunk_counter,
    smallsize staged_size)
{
    constexpr smallsize max_cached_entries = smallsize(1) << max_cached_levels;
    __shared__ key_type top_level_keys[max_cached_entries];

    const smallsize tid = threadIdx.x + blockIdx.x * blockDim.x;
    const smallsize local_tid = threadIdx.x;

    key_type lookup;
    bool to_find = false;
    if (tid < size)
    {
        lookup = query_keys[tid];
        to_find = true;
    }
    key_type result = static_cast<key_type>(not_found);

    if (to_find && staged_size > 0)
    { // doesnt use caching for staged area
        smallsize staged_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
            (const uint8_t *)staged_keys, staged_size, sizeof(key_type), lookup);

        for (; staged_offset < staged_size; ++staged_offset)
        {
            const auto key_at_offset = staged_keys[staged_offset];
            if (key_at_offset < lookup)
                break;

            const auto value_at_offset = staged_values[staged_offset];
            // ignore tombstones
            if (value_at_offset == not_found)
                continue;

            result = staged_keys[staged_offset];
            to_find = result != lookup;
            break;
        }
    }

    for (int k = 0; k < level_count; ++k)
    {
        if ((inserted_chunk_counter & (1u << k)) == 0)
            continue;

        smallsize num_element_level_k = chunk_size << k;
        smallsize offset = num_element_level_k - chunk_size;

        const auto current_level_keys = level_keys + offset;

        smallsize search_levels = ilog2_gpu(num_element_level_k);
        smallsize cached_levels = min(max_cached_levels, search_levels);
        smallsize cached_stride = smallsize(1) << (search_levels - cached_levels);
        smallsize cached_entries_count = SDIV(num_element_level_k, cached_stride);
        smallsize cached_offset = num_element_level_k - 1 - cached_stride * (cached_entries_count - 1);

        __syncthreads();

        // load top keys
        // filling shared memory with top keys -- cached some levels of the LSM tree
        // ? check later (spawn an inefficient number of threads)
        if (local_tid < cached_entries_count)
        {
            top_level_keys[local_tid] = current_level_keys[cached_offset + local_tid * cached_stride];
        }

        __syncthreads();

        if (!to_find)
            continue;

        smallsize coarse_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
            (const uint8_t *)top_level_keys, cached_entries_count, sizeof(key_type), lookup);

        smallsize continue_offset = num_element_level_k - 1 - (cached_stride * (cached_entries_count - 1 - coarse_offset));
        smallsize continue_stride = cached_stride >> 1u;

        smallsize final_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
            (const uint8_t *)current_level_keys, sizeof(key_type), lookup, continue_offset, continue_stride);

        // code for uncached search
        // smallsize final_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
        //    (const uint8_t*) current_level_keys, num_element_level_k, sizeof(key_type), lookup);

        for (; final_offset < num_element_level_k; ++final_offset)
        {
            const auto key_at_offset = current_level_keys[final_offset];
            if (key_at_offset < lookup)
                break;

            const auto value_at_offset = level_values[offset + final_offset];
            // ignore tombstones
            if (value_at_offset == not_found)
                continue;

            const auto new_result = current_level_keys[final_offset];
            result = new_result < result ? new_result : result;
            to_find = result != lookup;
            break;
        }
    }
    __syncthreads();

    if (tid < size)
    {
        results[tid] = result;
    }

#ifdef LSM_DEBUG_SUCCESSOR_KERNEL_PRINT_TREE
    lsm_debug_print_tree_and_staging_flat_all<key_type, smallsize, static_cast<uint32_t>(chunk_size)>(
        level_keys,
        level_values,
        static_cast<uint32_t>(level_count),
        static_cast<uint32_t>(inserted_chunk_counter),
        staged_keys,
        staged_values,
        static_cast<uint32_t>(staged_size),
        static_cast<smallsize>(not_found),
        "after_successor");
#endif
}

template <typename key_type, smallsize chunk_size>
GLOBALQUALIFIER void lsm_delete_kernel(
    const key_type *level_keys,
    smallsize *level_values,
    const key_type *staged_keys,
    smallsize *staged_values,
    const key_type *delete_keys,
    smallsize size,
    smallsize level_count,
    smallsize inserted_chunk_counter,
    smallsize staged_size)
{
    const smallsize tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= size)
        return;

    key_type delete_target = delete_keys[tid];
    smallsize result = 0;

    if (staged_size > 0)
    {
        smallsize staged_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
            (const uint8_t *)staged_keys, staged_size, sizeof(key_type), delete_target);

        for (; staged_offset < staged_size; ++staged_offset)
        {
            if (staged_keys[staged_offset] != delete_target)
                break;
            staged_values[staged_offset] = not_found; //<smallsize>;
        }
    }

    for (int k = 0; k < level_count; ++k)
    {
        if ((inserted_chunk_counter & (1u << k)) == 0)
            continue;

        __syncthreads();

        smallsize num_element_level_k = chunk_size << k;
        smallsize offset = num_element_level_k - chunk_size;

        const auto current_level_keys = level_keys + offset;

        smallsize final_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
            (const uint8_t *)current_level_keys, num_element_level_k, sizeof(key_type), delete_target);

        for (; final_offset < num_element_level_k; ++final_offset)
        {
            if (current_level_keys[final_offset] != delete_target)
                break;
            level_values[offset + final_offset] = not_found; //<smallsize>;
        }
    }
}

template <typename key_type, smallsize chunk_size, smallsize max_cached_levels>
GLOBALQUALIFIER void lsm_lookup_kernel(
    const key_type *level_keys,
    const smallsize *level_values,
    const key_type *staged_keys,
    const smallsize *staged_values,
    const key_type *query_keys,
    smallsize *results,
    smallsize size,
    smallsize level_count,
    smallsize inserted_chunk_counter,
    smallsize staged_size)
{
    constexpr smallsize max_cached_entries = smallsize(1) << max_cached_levels;
    __shared__ key_type top_level_keys[max_cached_entries];

    const smallsize tid = threadIdx.x + blockIdx.x * blockDim.x;
    const smallsize local_tid = threadIdx.x;

    key_type lookup;
    bool to_find = false;
    if (tid < size)
    {
        lookup = query_keys[tid];
        to_find = true;
    }
    smallsize result = not_found; //<smallsize>;

    if (to_find && staged_size > 0)
    {
        smallsize staged_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
            (const uint8_t *)staged_keys, staged_size, sizeof(key_type), lookup);

        for (; staged_offset < staged_size; ++staged_offset)
        {
            const auto key_at_offset = staged_keys[staged_offset];
            if (key_at_offset != lookup)
                break;

            const auto value_at_offset = staged_values[staged_offset];
            // ignore tombstones
            // if (value_at_offset == not_found<smallsize>) continue;
            if (value_at_offset == not_found)
                continue;

            result = value_at_offset;
            to_find = false;
            break;
        }
    }

    for (int k = 0; k < level_count; ++k)
    {
        if ((inserted_chunk_counter & (1u << k)) == 0)
            continue;

        smallsize num_element_level_k = chunk_size << k;
        smallsize offset = num_element_level_k - chunk_size;

        const auto current_level_keys = level_keys + offset;

        smallsize search_levels = ilog2_gpu(num_element_level_k);
        smallsize cached_levels = min(max_cached_levels, search_levels);
        smallsize cached_stride = smallsize(1) << (search_levels - cached_levels);
        smallsize cached_entries_count = SDIV(num_element_level_k, cached_stride);
        smallsize cached_offset = num_element_level_k - 1 - cached_stride * (cached_entries_count - 1);

        __syncthreads();

        // load top keys
        if (local_tid < cached_entries_count)
        {
            top_level_keys[local_tid] = current_level_keys[cached_offset + local_tid * cached_stride];
        }

        __syncthreads();

        if (!to_find)
            continue;

        smallsize coarse_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
            (const uint8_t *)top_level_keys, cached_entries_count, sizeof(key_type), lookup);

        smallsize continue_offset = num_element_level_k - 1 - (cached_stride * (cached_entries_count - 1 - coarse_offset));
        smallsize continue_stride = cached_stride >> 1u;

        smallsize final_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
            (const uint8_t *)current_level_keys, sizeof(key_type), lookup, continue_offset, continue_stride);

        // code for uncached search
        // smallsize final_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
        //    (const uint8_t*) current_level_keys, num_element_level_k, sizeof(key_type), lookup);

        for (; final_offset < num_element_level_k; ++final_offset)
        {
            const auto key_at_offset = current_level_keys[final_offset];
            if (key_at_offset != lookup)
                break;

            const auto value_at_offset = level_values[offset + final_offset];
            // ignore tombstones
            // if (value_at_offset == not_found<smallsize>) continue;
            if (value_at_offset == not_found)
                continue;

            result = value_at_offset;
            to_find = false;
            break;
        }
    }
    __syncthreads();

    if (tid < size)
    {
        results[tid] = result;
    }
}

// lsm_delete_kernel.cuh (or wherever this kernel lives)

template <typename key_type, smallsize chunk_size>
GLOBALQUALIFIER void lsm_delete_kernel_debug(
    const key_type *level_keys,
    smallsize *level_values,
    const key_type *staged_keys,
    smallsize *staged_values,
    const key_type *delete_keys,
    smallsize size,
    smallsize level_count,
    smallsize inserted_chunk_counter,
    smallsize staged_size)
{
    const smallsize tid = threadIdx.x + blockIdx.x * blockDim.x;

    // IMPORTANT: no early return if you ever use __syncthreads() in the kernel.
    // This kernel doesn't need __syncthreads(), so we just avoid it completely.

    if (tid < size)
    {
        const key_type delete_target = delete_keys[tid];

        // Delete in staged area
        if (staged_size > 0)
        {
            smallsize staged_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
                (const uint8_t *)staged_keys, staged_size, sizeof(key_type), delete_target);

            for (; staged_offset < staged_size; ++staged_offset)
            {
                if (staged_keys[staged_offset] != delete_target)
                    break;
                // ----purpose bug ---staged_values[staged_offset] = not_found;
                staged_values[staged_offset] = not_found;

            }
        }

        // Delete in levels
        for (int k = 0; k < level_count; ++k)
        {
            if ((inserted_chunk_counter & (1u << k)) == 0)
                continue;

            const smallsize num_element_level_k = chunk_size << k;
            const smallsize offset = num_element_level_k - chunk_size;
            const key_type *current_level_keys = level_keys + offset;

            smallsize final_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
                (const uint8_t *)current_level_keys, num_element_level_k, sizeof(key_type), delete_target);

            for (; final_offset < num_element_level_k; ++final_offset)
            {
                if (current_level_keys[final_offset] != delete_target)
                    break;
                level_values[offset + final_offset] = not_found;
            }
        }
    }

#ifdef LSM_DEBUG_DELETE_KERNEL_PRINT_TREE
    // Print once per kernel launch (otherwise you'll spam output and tank performance).
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        lsm_debug_print_tree_and_staging_flat_all<key_type, smallsize, static_cast<uint32_t>(chunk_size)>(
            level_keys,
            level_values,
            static_cast<uint32_t>(level_count),
            static_cast<uint32_t>(inserted_chunk_counter),
            staged_keys,
            staged_values,
            static_cast<uint32_t>(staged_size),
            static_cast<smallsize>(not_found),
            "after_delete");
    }
#endif
}


template <typename key_type, smallsize chunk_size, smallsize max_cached_levels>
GLOBALQUALIFIER void lsm_lookup_kernel_old(
    const key_type *level_keys,
    const smallsize *level_values,
    const key_type *staged_keys,
    const smallsize *staged_values,
    const key_type *query_keys,
    smallsize *results,
    smallsize size,
    smallsize level_count,
    smallsize inserted_chunk_counter,
    smallsize staged_size)
{
    constexpr smallsize max_cached_entries = smallsize(1) << max_cached_levels;
    __shared__ key_type top_level_keys[max_cached_entries];

    const smallsize tid = threadIdx.x + blockIdx.x * blockDim.x;
    const smallsize local_tid = threadIdx.x;

    key_type lookup;
    bool to_find = false;
    if (tid < size)
    {
        lookup = query_keys[tid];
        to_find = true;
    }
    smallsize result = not_found; //<smallsize>;

    if (to_find && staged_size > 0)
    {
        smallsize staged_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
            (const uint8_t *)staged_keys, staged_size, sizeof(key_type), lookup);
        if (staged_keys[staged_offset] == lookup)
        {
            result = staged_values[staged_offset];
            to_find = false;
        }
    }

    for (int k = 0; k < level_count; ++k)
    {
        if ((inserted_chunk_counter & (1u << k)) == 0)
            continue;

        smallsize num_element_level_k = chunk_size << k;
        smallsize offset = num_element_level_k - chunk_size;

        const auto current_level_keys = level_keys + offset;

        smallsize search_levels = ilog2_gpu(num_element_level_k);
        smallsize cached_levels = min(max_cached_levels, search_levels);
        smallsize cached_stride = smallsize(1) << (search_levels - cached_levels);
        smallsize cached_entries_count = SDIV(num_element_level_k, cached_stride);
        smallsize cached_offset = num_element_level_k - 1 - cached_stride * (cached_entries_count - 1);

        __syncthreads();

        // load top keys
        if (local_tid < cached_entries_count)
        {
            top_level_keys[local_tid] = current_level_keys[cached_offset + local_tid * cached_stride];
        }

        __syncthreads();

        if (!to_find)
            continue;

        smallsize coarse_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
            (const uint8_t *)top_level_keys, cached_entries_count, sizeof(key_type), lookup);

        smallsize continue_offset = num_element_level_k - 1 - (cached_stride * (cached_entries_count - 1 - coarse_offset));
        smallsize continue_stride = cached_stride >> 1u;

        smallsize final_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
            (const uint8_t *)current_level_keys, sizeof(key_type), lookup, continue_offset, continue_stride);

        // code for full search
        // smallsize final_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
        //    (const uint8_t*) current_level_keys, num_element_level_k, sizeof(key_type), lookup);

        if (current_level_keys[final_offset] == lookup)
        {
            result = level_values[offset + final_offset];
            to_find = false;
        }
    }
    __syncthreads();

    if (tid < size)
    {
        results[tid] = result;
    }
}

template <typename key_type, smallsize chunk_size>
GLOBALQUALIFIER void lsm_naive_range_lookup_kernel(
    const key_type *level_keys,
    const smallsize *level_values,
    const key_type *staged_keys,
    const smallsize *staged_values,
    const key_type *lower_bounds,
    const key_type *upper_bounds,
    smallsize *results,
    smallsize size,
    smallsize level_count,
    smallsize inserted_chunk_counter,
    smallsize staged_size)
{
    const smallsize tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= size)
        return;

    key_type lower_bound = lower_bounds[tid];
    key_type upper_bound = upper_bounds[tid];
    smallsize result = 0;

    if (staged_size > 0)
    {
        smallsize staged_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
            (const uint8_t *)staged_keys, staged_size, sizeof(key_type), lower_bound);
        for (; staged_offset < staged_size; ++staged_offset)
        {
            if (staged_keys[staged_offset] > upper_bound)
                break;
            if (staged_keys[staged_offset] >= lower_bound)
                result += staged_values[staged_offset];
        }
    }

    for (int k = 0; k < level_count; ++k)
    {
        if ((inserted_chunk_counter & (1u << k)) == 0)
            continue;

        smallsize num_element_level_k = chunk_size << k;
        smallsize offset = num_element_level_k - chunk_size;

        const auto current_level_keys = level_keys + offset;

        smallsize final_offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
            (const uint8_t *)current_level_keys, num_element_level_k, sizeof(key_type), lower_bound);

        for (; final_offset < num_element_level_k; ++final_offset)
        {
            if (current_level_keys[final_offset] > upper_bound)
                break;
            if (current_level_keys[final_offset] >= lower_bound)
                result += level_values[offset + final_offset];
        }
    }

    results[tid] = result;
}

GLOBALQUALIFIER
void lsm_prepare_output_kernel(
    smallsize *results,
    smallsize size)
{
    const smallsize tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= size)
        return;

    results[tid] = 0;
}

template <typename key_type, smallsize chunk_size, smallsize cg_size_log>
GLOBALQUALIFIER void lsm_collaborative_range_lookup_kernel(
    const key_type *sorted_keys,
    const smallsize *sorted_values,
    smallsize sorted_size,
    const key_type *lower_bounds,
    const key_type *upper_bounds,
    smallsize *results,
    smallsize size)
{
    constexpr smallsize cg_size = 1u << cg_size_log;

    smallsize tid = blockIdx.x * blockDim.x + threadIdx.x;

    key_type lower_limit;
    key_type upper_limit;
    smallsize offset;
    smallsize result;
    bool to_find = false;

    if (tid < size)
    {
        to_find = true;
        lower_limit = lower_bounds[tid];
        upper_limit = upper_bounds[tid];

        offset = opt_reverse_device_binary_search<key_type, smallsize, true>(
            sorted_keys, sorted_size, lower_limit);
    }

    auto tile = cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());
    auto work_queue = tile.ballot(to_find);
    while (work_queue)
    {
        const auto local_id = __ffs(work_queue) - 1;
        const auto local_lower = tile.shfl(lower_limit, local_id);
        const auto local_upper = tile.shfl(upper_limit, local_id);
        const auto local_offset = tile.shfl(offset, local_id);
        const auto local_result = mem::rq(mem::col_store<key_type, smallsize, smallsize>(sorted_keys, sorted_values, sorted_size)).cooperative_range_query(tile, local_lower, local_upper, local_offset);

        if (local_id == tile.thread_rank())
        {
            result = local_result;
            to_find = false;
        }
        work_queue = tile.ballot(to_find);
    }

    if (tid < size)
    {
        results[tid] = result;
    }
}

template <typename key_type, typename value_type>
void untimed_pair_merge(
    void *temp, size_t temp_bytes,
    const key_type *k1, const key_type *k2, key_type *ko,
    const value_type *v1, const value_type *v2, value_type *vo,
    size_t input_size1, size_t input_size2, cudaStream_t stream)
{
    cub::DeviceMerge::MergePairs(
        temp, temp_bytes,
        k1, v1, input_size1,
        k2, v2, input_size2,
        ko, vo,
        {},
        stream);
}

template <typename key_type, typename value_type>
size_t find_pair_merge_buffer_size(size_t input_size1, size_t input_size2)
{
    size_t temp_bytes_required = 0;
    cub::DeviceMerge::MergePairs(
        nullptr, temp_bytes_required,
        (key_type *)nullptr, (value_type *)nullptr, input_size1,
        (key_type *)nullptr, (value_type *)nullptr, input_size2,
        (key_type *)nullptr, (value_type *)nullptr);
    return temp_bytes_required;
}

template <typename key_type_, smallsize chunk_size_log = 16, smallsize cg_size_log = 5>
class lsm_tree_ashkiani final
{
public:
    using key_type = key_type_;
    using value_type = smallsize;

private:
    // taken from original code
    static constexpr size_t threads_per_block = 512; // MAXBLOCKSIZE;  // 256;
    static constexpr size_t max_cached_levels = 8;
    static constexpr size_t chunk_size = size_t(1) << chunk_size_log;
    static constexpr size_t cg_size = size_t(1) << cg_size_log;

    cuda_buffer<key_type> level_keys_buffer;
    cuda_buffer<smallsize> level_values_buffer;
    cuda_buffer<key_type> staging_keys_buffer;
    cuda_buffer<smallsize> staging_values_buffer;

    cuda_buffer<uint8_t> temp_sort_buffer;
    cuda_buffer<key_type> temp_keys_buffer_a;
    cuda_buffer<smallsize> temp_values_buffer_a;
    cuda_buffer<key_type> temp_keys_buffer_b;
    cuda_buffer<smallsize> temp_values_buffer_b;

    size_t total_available_slots = 0;
    size_t level_count = 0;

    size_t staged_insert_size = 0;
    size_t inserted_chunk_counter = 0;

    void merge_at_level(size_t initial_level, const key_type *insert_keys, const smallsize *insert_values, cudaStream_t stream)
    {
        size_t new_chunk_count = 1u << initial_level;
        size_t level_size = chunk_size << initial_level;
        size_t level_offset = (level_size - chunk_size);

        auto source_keys = temp_keys_buffer_a.ptr();
        auto source_values = temp_values_buffer_a.ptr();
        auto dest_keys = temp_keys_buffer_b.ptr();
        auto dest_values = temp_values_buffer_b.ptr();

        size_t target_level = initial_level;
        while ((inserted_chunk_counter & (1u << target_level)) != 0)
            ++target_level;

        for (size_t level = initial_level; level < target_level; ++level)
        {
            // std::cerr << "merging with level " << level << " of size " << level_size << "\n";
            auto local_source_keys = level == initial_level ? insert_keys : source_keys;
            auto local_source_values = level == initial_level ? insert_values : source_values;
            untimed_pair_merge<key_type, smallsize>(
                temp_sort_buffer.raw_ptr, temp_sort_buffer.size_in_bytes(),
                local_source_keys, level_keys_buffer.ptr() + level_offset, dest_keys,
                local_source_values, level_values_buffer.ptr() + level_offset, dest_values,
                level_size, level_size, stream);
            level_offset += level_size;
            level_size *= 2;

            // swap temp buffers
            std::swap(source_keys, dest_keys);
            std::swap(source_values, dest_values);
        }

        // std::cerr << "writing result to level " << target_level << "\n";
        cudaMemcpyAsync(
            level_keys_buffer.ptr() + level_offset,
            initial_level == target_level ? insert_keys : source_keys,
            level_size * sizeof(key_type),
            cudaMemcpyDefault, stream);
        cudaMemcpyAsync(
            level_values_buffer.ptr() + level_offset,
            initial_level == target_level ? insert_values : source_values,
            level_size * sizeof(smallsize),
            cudaMemcpyDefault, stream);

        inserted_chunk_counter += new_chunk_count;
    }

public:
    static constexpr const char *name = "lsm_tree";

    static constexpr operation_support can_lookup = operation_support::async;
    static constexpr operation_support can_lower_bound_rank = operation_support::none;
    static constexpr operation_support can_multi_lookup = operation_support::none;
    static constexpr operation_support can_range_lookup = operation_support::async;
    static constexpr operation_support can_insert = operation_support::async;
    static constexpr operation_support can_delete = operation_support::async;
    static constexpr operation_support can_update = operation_support::async;

    static std::string short_description()
    {
        return std::string("lsm_tree_ashkiani");
    }

    static parameters_type parameters()
    {
        return {
            {"chunk_size_log", std::to_string(chunk_size_log)},
            {"threads_per_block", std::to_string(threads_per_block)},
            {"max_cached_levels", std::to_string(max_cached_levels)},
            {"cg_size", std::to_string(cg_size)},
        };
    }

    size_t gpu_resident_bytes()
{
    return level_keys_buffer.size_in_bytes() +
           level_values_buffer.size_in_bytes() +
           staging_keys_buffer.size_in_bytes() +
           staging_values_buffer.size_in_bytes() +
           temp_sort_buffer.size_in_bytes() +
           temp_keys_buffer_a.size_in_bytes() +
           temp_values_buffer_a.size_in_bytes() +
           temp_keys_buffer_b.size_in_bytes() +
           temp_values_buffer_b.size_in_bytes();
}


    size_t gpu_resident_bytes_previous()
    {
        return level_keys_buffer.size_in_bytes() + level_values_buffer.size_in_bytes() +
               staging_keys_buffer.size_in_bytes() + staging_values_buffer.size_in_bytes();
    }

    void build(const key_type *keys, size_t size, size_t max_size, size_t available_memory_bytes, double *build_time_ms, size_t *build_bytes)
    {

        level_count = 1; // level 0 has one chucnk size
        total_available_slots = chunk_size;
        //start at 1 because we have staging buffer
        while (true)
        {
            if (total_available_slots + (chunk_size - 1) >= max_size)
                break;
            level_count++;
            total_available_slots = (chunk_size << level_count) - chunk_size;
        }

        level_keys_buffer.alloc(total_available_slots);
        C2EX
            level_values_buffer.alloc(total_available_slots);
        C2EX
            staging_keys_buffer.alloc(chunk_size);
        C2EX
            staging_values_buffer.alloc(chunk_size);
        C2EX

        {
            size_t max_level_size = (chunk_size << (level_count - 1));
            size_t max_temp_sort_bytes = std::max(
                find_pair_sort_buffer_size<key_type, smallsize>(max_level_size),
                find_pair_merge_buffer_size<key_type, smallsize>(max_level_size, max_level_size));
            temp_sort_buffer.alloc(max_temp_sort_bytes);
            C2EX

                temp_keys_buffer_a.alloc(max_level_size);
            C2EX
                temp_values_buffer_a.alloc(max_level_size);
            C2EX
                temp_keys_buffer_b.alloc(max_level_size);
            C2EX
                temp_values_buffer_b.alloc(max_level_size);
            C2EX
        }

        // we want to report only the space we actually need during build
        size_t temp_storage_bytes = 0;
        {
            init_offsets(temp_values_buffer_a.ptr(), size, build_time_ms);

            temp_storage_bytes = find_pair_sort_buffer_size<key_type, smallsize>(size);
            timed_pair_sort(
                temp_sort_buffer.raw_ptr, temp_storage_bytes,
                keys, level_keys_buffer.ptr(), temp_values_buffer_a.ptr(), level_values_buffer.ptr(), size, build_time_ms);

            cudaDeviceSynchronize();
            C2EX
        };

        inserted_chunk_counter = size / chunk_size;
        staged_insert_size = size % chunk_size;

        size_t end_of_data_block = size;

        // copy last (incomplete) chunk to staging buffer
        if (staged_insert_size > 0)
        {
            // std::cerr << "copying last " << staged_insert_size << " elements to temp buffer\n";
            cudaMemcpyAsync(
                staging_keys_buffer.ptr(),
                level_keys_buffer.ptr() + end_of_data_block - staged_insert_size,
                staged_insert_size * sizeof(key_type),
                cudaMemcpyDefault);
            cudaMemcpyAsync(
                staging_values_buffer.ptr(),
                level_values_buffer.ptr() + end_of_data_block - staged_insert_size,
                staged_insert_size * sizeof(smallsize),
                cudaMemcpyDefault);
            end_of_data_block -= staged_insert_size;
        }

        // fill the main levels correctly
        for (size_t rev_shift = 0; rev_shift < 32; ++rev_shift)
        {
            auto shift = 31 - rev_shift;
            if ((inserted_chunk_counter & (1u << shift)) == 0)
                continue;
            size_t level_size = chunk_size << shift;
            size_t level_offset = (level_size - chunk_size);
            size_t source_offset = end_of_data_block - level_size;
            end_of_data_block -= level_size;
            // std::cerr << "copying level " << shift << " of size " << level_size << " from " << source_offset << " to " << level_offset << "\n";
            if (level_offset == source_offset)
                continue;
            cudaMemcpyAsync(
                temp_keys_buffer_b.ptr(),
                level_keys_buffer.ptr() + source_offset,
                level_size * sizeof(key_type),
                cudaMemcpyDefault);
            cudaMemcpyAsync(
                level_keys_buffer.ptr() + level_offset,
                temp_keys_buffer_b.ptr(),
                level_size * sizeof(key_type),
                cudaMemcpyDefault);
            cudaMemcpyAsync(
                temp_values_buffer_b.ptr(),
                level_values_buffer.ptr() + source_offset,
                level_size * sizeof(smallsize),
                cudaMemcpyDefault);
            cudaMemcpyAsync(
                level_values_buffer.ptr() + level_offset,
                temp_values_buffer_b.ptr(),
                level_size * sizeof(smallsize),
                cudaMemcpyDefault);
        }

        if (build_bytes)
            *build_bytes += gpu_resident_bytes() + sizeof(smallsize) * size + temp_storage_bytes;
    }

    void dump_tree()
    {
        size_t n = chunk_size;
        size_t o = 0;
        size_t k = 0;
        for (; k < level_count; ++k)
        {
            if ((inserted_chunk_counter & (1u << k)) == 0)
                continue;
            n = chunk_size << k;
            o = n - chunk_size;
            std::cerr << "LEVEL " << k << ": ";
            level_keys_buffer.debug_dump(n, o);
        }
        std::cerr << "STAGING : ";
        staging_keys_buffer.debug_dump(staged_insert_size);
    }

    void destroy()
    {
        level_keys_buffer.free();
        level_values_buffer.free();
        temp_sort_buffer.free();
        temp_keys_buffer_a.free();
        temp_values_buffer_a.free();
        temp_keys_buffer_b.free();
        temp_values_buffer_b.free();
        total_available_slots = 0;
        staged_insert_size = 0;
        inserted_chunk_counter = 0;
    }

    void lookup(const key_type *keys, value_type *result, size_t size, cudaStream_t stream)
    {
        lsm_lookup_kernel<key_type, chunk_size, max_cached_levels>
            <<<SDIV(size, threads_per_block), threads_per_block, 0, stream>>>(
                level_keys_buffer.ptr(),
                level_values_buffer.ptr(),
                staging_keys_buffer.ptr(),
                staging_values_buffer.ptr(),
                keys,
                result,
                size,
                level_count,
                inserted_chunk_counter,
                staged_insert_size);
    }

    // void next_larger(const key_type* keys, key_type* result, size_t size, cudaStream_t stream) {

    void lookups_successor(const key_type *keys, key_type *result, size_t size, cudaStream_t stream)
    {

        lsm_next_larger_kernel<key_type, chunk_size, max_cached_levels>
            <<<SDIV(size, threads_per_block), threads_per_block, 0, stream>>>(
                level_keys_buffer.ptr(),
                level_values_buffer.ptr(),
                staging_keys_buffer.ptr(),
                staging_values_buffer.ptr(),
                keys,
                result,
                size,
                level_count,
                inserted_chunk_counter,
                staged_insert_size);
    }
    void multi_lookup_sum(const key_type *keys, value_type *result, size_t size, cudaStream_t stream)
    {
        range_lookup_sum(keys, keys, result, size, stream);
    }

    void range_lookup_sum(const key_type *lower, const key_type *upper, value_type *result, size_t size, cudaStream_t stream)
    {
        if constexpr (cg_size_log == 0)
        {
            lsm_naive_range_lookup_kernel<key_type, chunk_size>
                <<<SDIV(size, threads_per_block), threads_per_block, 0, stream>>>(
                    level_keys_buffer.ptr(),
                    level_values_buffer.ptr(),
                    staging_keys_buffer.ptr(),
                    staging_values_buffer.ptr(),
                    lower,
                    upper,
                    result,
                    size,
                    level_count,
                    inserted_chunk_counter,
                    staged_insert_size);
        }
        else
        {
            lsm_prepare_output_kernel<<<SDIV(size, threads_per_block), threads_per_block, 0, stream>>>(
                result,
                size);

            if (staged_insert_size > 0)
            {
                lsm_collaborative_range_lookup_kernel<key_type, chunk_size, cg_size_log>
                    <<<SDIV(size, threads_per_block), threads_per_block, 0, stream>>>(
                        staging_keys_buffer.ptr(),
                        staging_values_buffer.ptr(),
                        staged_insert_size,
                        lower,
                        upper,
                        result,
                        size);
            }

            for (size_t k = 0; k < level_count; ++k)
            {
                if ((inserted_chunk_counter & (1u << k)) == 0)
                    continue;

                size_t num_element_level_k = chunk_size << k;
                size_t offset = num_element_level_k - chunk_size;

                lsm_collaborative_range_lookup_kernel<key_type, chunk_size, cg_size_log>
                    <<<SDIV(size, threads_per_block), threads_per_block, 0, stream>>>(
                        level_keys_buffer.ptr() + offset,
                        level_values_buffer.ptr() + offset,
                        num_element_level_k,
                        lower,
                        upper,
                        result,
                        size);
            }
        }
    }

    void insert(const key_type *insert_list, const smallsize *positions, size_t size, cudaStream_t stream)
    {

        size_t insert_list_offset = 0;

        size_t newly_inserted_chunk_counter = size / chunk_size;
        size_t new_staged_insert_size = size % chunk_size;

        if (new_staged_insert_size > 0)
        {
            // move overflowing elements to staging buffer
            size_t size_left_in_staging_buffer = chunk_size - staged_insert_size;
            size_t moved_size = std::min(size_left_in_staging_buffer, new_staged_insert_size);
            // copy to staging buffer next to the already existing data
            cudaMemcpyAsync(
                staging_keys_buffer.ptr() + staged_insert_size,
                insert_list,
                moved_size * sizeof(key_type),
                cudaMemcpyDefault, stream);
            cudaMemcpyAsync(
                staging_values_buffer.ptr() + staged_insert_size,
                positions,
                moved_size * sizeof(smallsize),
                cudaMemcpyDefault, stream);
            // sort to temp buffer a
            // might as well use a merge here
            untimed_pair_sort(
                temp_sort_buffer.raw_ptr, temp_sort_buffer.size_in_bytes(),
                staging_keys_buffer.ptr(), temp_keys_buffer_a.ptr(),
                staging_values_buffer.ptr(), temp_values_buffer_a.ptr(),
                staged_insert_size + moved_size, stream);

            if (new_staged_insert_size + staged_insert_size >= chunk_size)
            {
                // merge with existing levels
                merge_at_level(0, temp_keys_buffer_a.ptr(), temp_values_buffer_a.ptr(), stream);
                insert_list_offset += size_left_in_staging_buffer;

                size_t remaining_size = new_staged_insert_size + staged_insert_size - chunk_size;
                if (remaining_size > 0)
                {
                    // copy remaining elements to staging buffer
                    cudaMemcpyAsync(
                        staging_keys_buffer.ptr(),
                        insert_list + insert_list_offset,
                        remaining_size * sizeof(key_type),
                        cudaMemcpyDefault, stream);
                    cudaMemcpyAsync(
                        staging_values_buffer.ptr(),
                        positions + insert_list_offset,
                        remaining_size * sizeof(smallsize),
                        cudaMemcpyDefault, stream);
                    insert_list_offset += remaining_size;
                }
                staged_insert_size = remaining_size;
            }
            else
            {
                // copy back to staging buffer
                cudaMemcpyAsync(
                    staging_keys_buffer.ptr(),
                    temp_keys_buffer_a.ptr(),
                    (new_staged_insert_size + staged_insert_size) * sizeof(key_type),
                    cudaMemcpyDefault, stream);
                cudaMemcpyAsync(
                    staging_values_buffer.ptr(),
                    temp_values_buffer_a.ptr(),
                    (new_staged_insert_size + staged_insert_size) * sizeof(smallsize),
                    cudaMemcpyDefault, stream);
                staged_insert_size += new_staged_insert_size;
                insert_list_offset += new_staged_insert_size;
            }
        }

        // find the level into which to insert the new chunks
        for (size_t new_level = 0; new_level < level_count; ++new_level)
        {
            if ((newly_inserted_chunk_counter & (1u << new_level)) == 0)
                continue;
            merge_at_level(new_level, insert_list + insert_list_offset, positions + insert_list_offset, stream);
            insert_list_offset += chunk_size << new_level;
        }
    }

    void remove(const key_type *delete_list, size_t size, cudaStream_t stream)
    {
        // authors did not provide an implementation
        // the paper mentions using tombstones for deletions, so we do exactly that

#ifdef PRINT_REMOVE_VALUES
        printf("\n ");
        printf("****************************************************************\n");
        printf("REMOVE, Size of Delete List:\n", size);
        printf("****************************************************************\n");
        printf("\n ");

        assert(ordered_node_pairs_buffer.size_in_bytes() % node_stride == 0);
        std::vector<key_type> keys_host(size);

        cudaMemcpy(keys_host.data(), delete_list, size * sizeof(key_type), cudaMemcpyDeviceToHost);
        CUERR
        for (int i = 0; i < size; ++i)
        {
            printf("Key %d: %llu \n", i, static_cast<unsigned long long>(keys_host[i]));
        }
        // printf("Remove: partition_count %d, partition_count_with_overflow %d\n", partition_count, partition_count_with_overflow);
#endif

        printf("lsm_tree_ashkiani: delete operation called, performing tombstone Deletions...\n");
        // lsm_delete_kernel_debug<key_type, chunk_size>

        lsm_delete_kernel<key_type, chunk_size>
            <<<SDIV(size, threads_per_block), threads_per_block, 0, stream>>>(
                level_keys_buffer.ptr(),
                level_values_buffer.ptr(),
                staging_keys_buffer.ptr(),
                staging_values_buffer.ptr(),
                delete_list,
                size,
                level_count,
                inserted_chunk_counter,
                staged_insert_size);
    }
};

#endif
