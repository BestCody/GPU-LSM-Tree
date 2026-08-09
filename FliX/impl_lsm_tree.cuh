// =============================================================================
// File: impl_lsm_tree.cuh
// Description: GPU LSM (LSMu) following Ashkiani et al., arXiv:1707.05354v2
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef IMPL_LSM_TREE_CUH
#define IMPL_LSM_TREE_CUH

#include "definitions_coarse_granular.cuh"
#include "utilities.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cub/cub.cuh>
#include <cub/device/device_merge.cuh>

// The paper stores a 31-bit original key in bits [31:1] and its status in bit
// zero. A zero status is a tombstone and a one status is a regular element.
// CLEANUP uses a maximum-original-key tombstone as its trailing placebo.
template <typename Key>
struct lsm_paper_key
{
    static_assert(std::is_same<Key, std::uint32_t>::value,
                  "GPU LSM in 1707.05354v2 uses 32-bit encoded keys");

    static constexpr Key status_mask = Key{1};
    static constexpr Key placebo_original = std::numeric_limits<Key>::max() >> 1;
    static constexpr Key max_user_key = placebo_original;
    static constexpr Key placebo = placebo_original << 1;

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Key original(Key encoded)
    {
        return encoded >> 1;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_regular(Key encoded)
    {
        return (encoded & status_mask) != 0;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Key regular(Key key)
    {
        return (key << 1) | status_mask;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Key tombstone(Key key)
    {
        return key << 1;
    }
};

// Section IV-A: merge on the original key only. CUB's stable merge leaves the
// first input (the newer batch/level) before the second input on ties.
template <typename Key>
struct lsm_original_key_less
{
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    bool operator()(Key lhs, Key rhs) const
    {
        return lsm_paper_key<Key>::original(lhs) <
               lsm_paper_key<Key>::original(rhs);
    }
};

template <typename Key>
DEVICEQUALIFIER INLINEQUALIFIER
smallsize lsm_lower_bound_original(const Key *keys, smallsize size, Key query)
{
    smallsize first = 0;
    smallsize count = size;
    while (count != 0)
    {
        const smallsize step = count >> 1;
        const smallsize middle = first + step;
        if (lsm_paper_key<Key>::original(keys[middle]) < query)
        {
            first = middle + 1;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    return first;
}

template <typename Key>
DEVICEQUALIFIER INLINEQUALIFIER
smallsize lsm_upper_bound_original(const Key *keys, smallsize size, Key query)
{
    smallsize first = 0;
    smallsize count = size;
    while (count != 0)
    {
        const smallsize step = count >> 1;
        const smallsize middle = first + step;
        if (lsm_paper_key<Key>::original(keys[middle]) <= query)
        {
            first = middle + 1;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    return first;
}

template <typename Key>
GLOBALQUALIFIER void lsm_encode_update_batch_kernel(
    const Key *insert_keys,
    const smallsize *insert_values,
    smallsize insert_size,
    const Key *delete_keys,
    smallsize delete_size,
    smallsize real_batch_size,
    Key *encoded_keys,
    smallsize *encoded_values,
    smallsize batch_size)
{
    const smallsize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size)
        return;

    // Section IV-A permits completing a partial batch by duplicating an
    // arbitrary member. We duplicate its final member.
    const smallsize local = tid < real_batch_size ? tid : real_batch_size - 1;
    const smallsize source = local;
    const bool deletion = source >= insert_size;

    if (deletion)
    {
        const Key key = delete_keys[source - insert_size];
        encoded_keys[tid] = lsm_paper_key<Key>::tombstone(key);
        encoded_values[tid] = not_found;
    }
    else
    {
        const Key key = insert_keys[source];
        encoded_keys[tid] = lsm_paper_key<Key>::regular(key);
        encoded_values[tid] = insert_values[source];
    }
}

template <typename Key>
GLOBALQUALIFIER void lsm_encode_bulk_build_kernel(
    const Key *keys,
    smallsize real_size,
    Key *encoded_keys,
    smallsize *encoded_values,
    smallsize padded_size)
{
    const smallsize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= padded_size)
        return;

    // Section IV-A: a partial final batch is padded by duplicating an
    // arbitrary member. The bulk build then sorts all resident elements once.
    const smallsize source = tid < real_size ? tid : real_size - 1;
    encoded_keys[tid] = lsm_paper_key<Key>::regular(keys[source]);
    encoded_values[tid] = source;
}

template <typename Key>
GLOBALQUALIFIER void lsm_lookup_kernel_paper(
    const Key *level_keys,
    const smallsize *level_values,
    const Key *queries,
    smallsize *results,
    smallsize query_count,
    smallsize level_count,
    std::uint64_t num_batches,
    smallsize batch_size)
{
    const smallsize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= query_count)
        return;

    const Key query = queries[tid];
    smallsize result = not_found;

    if (query <= lsm_paper_key<Key>::max_user_key)
    {
        // Section III-D: lower levels are newer, so stop at the first segment.
        for (smallsize level = 0; level < level_count; ++level)
        {
            if ((num_batches & (std::uint64_t{1} << level)) == 0)
                continue;

            const smallsize level_size = batch_size << level;
            const smallsize level_offset = level_size - batch_size;
            const Key *keys = level_keys + level_offset;
            const smallsize index = lsm_lower_bound_original(keys, level_size, query);

            if (index == level_size ||
                lsm_paper_key<Key>::original(keys[index]) != query)
                continue;

            if (lsm_paper_key<Key>::is_regular(keys[index]))
                result = level_values[level_offset + index];
            break;
        }
    }

    results[tid] = result;
}

// Section IV-C/D, stage 1: compute the lower/upper position and preliminary
// candidate count for every (query, level) pair. Slots are query-major so a
// device-wide scan makes all candidates for a query one contiguous segment.
template <typename Key>
GLOBALQUALIFIER void lsm_query_bounds_kernel_paper(
    const Key *level_keys,
    const Key *lower_bounds,
    const Key *upper_bounds,
    smallsize *lower_indices,
    smallsize *upper_indices,
    smallsize *candidate_counts,
    smallsize query_count,
    smallsize level_count,
    std::uint64_t num_batches,
    smallsize batch_size)
{
    const smallsize tid = blockIdx.x * blockDim.x + threadIdx.x;
    const smallsize slot_count = query_count * level_count;
    if (tid >= slot_count)
        return;

    const smallsize query = tid / level_count;
    const smallsize level = tid - query * level_count;
    smallsize lower_index = 0;
    smallsize upper_index = 0;

    if (lower_bounds[query] <= upper_bounds[query] &&
        (num_batches & (std::uint64_t{1} << level)) != 0)
    {
        const smallsize level_size = batch_size << level;
        const smallsize level_offset = level_size - batch_size;
        const Key *keys = level_keys + level_offset;
        lower_index = lsm_lower_bound_original(
            keys, level_size, lower_bounds[query]);
        upper_index = lsm_upper_bound_original(
            keys, level_size, upper_bounds[query]);
    }

    lower_indices[tid] = lower_index;
    upper_indices[tid] = upper_index;
    candidate_counts[tid] = upper_index - lower_index;
}

// Section IV-C/D, stage 3: materialize all preliminary candidates. Occupied
// levels are visited from smallest/newest to largest/oldest, preserving the
// temporal order needed by the stable segmented radix sort in stage 4.
template <typename Key, bool GatherValues>
GLOBALQUALIFIER void lsm_gather_query_candidates_kernel_paper(
    const Key *level_keys,
    const smallsize *level_values,
    const smallsize *lower_indices,
    const smallsize *candidate_offsets,
    Key *candidate_keys,
    smallsize *candidate_values,
    smallsize query_count,
    smallsize level_count,
    smallsize batch_size)
{
    const smallsize global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const smallsize warp = global_thread >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const smallsize first_query = warp << 5;
    if (first_query >= query_count)
        return;

    // All lanes collaborate on the candidates of 32 consecutive queries.
    // Thus the warp writes consecutive result positions coalesced, as in
    // the paper's stage-3 implementation.
    const smallsize final_query = min(first_query + 32u, query_count);
    const smallsize first_slot = first_query * level_count;
    const smallsize final_slot = final_query * level_count;
    const smallsize first_output = candidate_offsets[first_slot];
    const smallsize final_output = candidate_offsets[final_slot];
    const smallsize rounds = (final_output - first_output + 31u) >> 5;

    for (smallsize round = 0; round < rounds; ++round)
    {
        const smallsize output = first_output + (round << 5) + lane;
        if (output >= final_output)
            continue;

        // Locate the per-level range that owns this output position. Using
        // upper-bound handles empty ranges with repeated scan offsets.
        smallsize left = first_slot + 1;
        smallsize right = final_slot + 1;
        while (left < right)
        {
            const smallsize middle = left + ((right - left) >> 1);
            if (candidate_offsets[middle] <= output)
                left = middle + 1;
            else
                right = middle;
        }
        const smallsize slot = left - 1;
        const smallsize level = slot % level_count;
        const smallsize level_size = batch_size << level;
        const smallsize level_offset = level_size - batch_size;
        const smallsize source = level_offset + lower_indices[slot] +
                                 output - candidate_offsets[slot];
        candidate_keys[output] = level_keys[source];
        if constexpr (GatherValues)
            candidate_values[output] = level_values[source];
    }
}

GLOBALQUALIFIER void lsm_make_query_segment_offsets_kernel(
    const smallsize *candidate_offsets,
    smallsize *query_offsets,
    smallsize query_count,
    smallsize level_count)
{
    const smallsize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid <= query_count)
        query_offsets[tid] = candidate_offsets[tid * level_count];
}

// Section IV-C, stage 5. Each lane owns one query and 32 neighboring queries
// collaborate through warp ballots, matching the paper's validation scheme.
template <typename Key>
GLOBALQUALIFIER void lsm_count_sorted_candidates_kernel_paper(
    const Key *keys,
    const smallsize *query_offsets,
    smallsize *results,
    smallsize query_count)
{
    const smallsize global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const smallsize query = global_thread;
    const bool active_query = query < query_count;
    const smallsize begin = active_query ? query_offsets[query] : 0;
    const smallsize end = active_query ? query_offsets[query + 1] : 0;
    smallsize rounds = end - begin;
    for (unsigned delta = 16; delta != 0; delta >>= 1)
        rounds = max(rounds,
                     __shfl_down_sync(0xffffffffu, rounds, delta));
    rounds = __shfl_sync(0xffffffffu, rounds, 0);
    smallsize count = 0;

    for (smallsize round = 0; round < rounds; ++round)
    {
        const smallsize index = begin + round;
        bool valid = false;
        if (active_query && index < end)
        {
            const Key encoded = keys[index];
            valid = lsm_paper_key<Key>::is_regular(encoded) &&
                    (index == begin ||
                     lsm_paper_key<Key>::original(keys[index - 1]) !=
                         lsm_paper_key<Key>::original(encoded));
        }
        const unsigned valid_mask = __ballot_sync(0xffffffffu, valid);
        count += (valid_mask >> lane) & 1u;
    }

    if (active_query)
        results[query] = count;
}

// RANGE follows the same candidate gathering and stable segmented sort as the
// paper. Its repository-specific final output is the sum of valid values.
template <typename Key>
GLOBALQUALIFIER void lsm_sum_sorted_candidates_kernel_paper(
    const Key *keys,
    const smallsize *values,
    const smallsize *query_offsets,
    smallsize *results,
    smallsize query_count)
{
    const smallsize global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const smallsize query = global_thread;
    const bool active_query = query < query_count;
    const smallsize begin = active_query ? query_offsets[query] : 0;
    const smallsize end = active_query ? query_offsets[query + 1] : 0;
    smallsize rounds = end - begin;
    for (unsigned delta = 16; delta != 0; delta >>= 1)
        rounds = max(rounds,
                     __shfl_down_sync(0xffffffffu, rounds, delta));
    rounds = __shfl_sync(0xffffffffu, rounds, 0);
    smallsize sum = 0;

    for (smallsize round = 0; round < rounds; ++round)
    {
        const smallsize index = begin + round;
        bool valid = false;
        if (active_query && index < end)
        {
            const Key encoded = keys[index];
            valid = lsm_paper_key<Key>::is_regular(encoded) &&
                    (index == begin ||
                     lsm_paper_key<Key>::original(keys[index - 1]) !=
                         lsm_paper_key<Key>::original(encoded));
        }
        const unsigned valid_mask = __ballot_sync(0xffffffffu, valid);
        if (((valid_mask >> lane) & 1u) != 0)
            sum += values[index];
    }

    if (active_query)
        results[query] = sum;
}

// Successor is an adapter extension noted as straightforward by the paper.
// It validates a candidate against all more-recent occupied levels.
template <typename Key>
DEVICEQUALIFIER INLINEQUALIFIER
bool lsm_key_occurs_in_newer_level(
    const Key *level_keys,
    smallsize candidate_level,
    std::uint64_t num_batches,
    smallsize batch_size,
    Key original_key)
{
    for (smallsize level = 0; level < candidate_level; ++level)
    {
        if ((num_batches & (std::uint64_t{1} << level)) == 0)
            continue;
        const smallsize level_size = batch_size << level;
        const smallsize level_offset = level_size - batch_size;
        const Key *keys = level_keys + level_offset;
        const smallsize index =
            lsm_lower_bound_original(keys, level_size, original_key);
        if (index != level_size &&
            lsm_paper_key<Key>::original(keys[index]) == original_key)
            return true;
    }
    return false;
}

template <typename Key>
GLOBALQUALIFIER void lsm_successor_kernel_paper(
    const Key *level_keys,
    const Key *queries,
    Key *results,
    smallsize query_count,
    smallsize level_count,
    std::uint64_t num_batches,
    smallsize batch_size)
{
    const smallsize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= query_count)
        return;

    const Key query = queries[tid];
    Key result = static_cast<Key>(not_found);

    for (smallsize level = 0; level < level_count; ++level)
    {
        if ((num_batches & (std::uint64_t{1} << level)) == 0)
            continue;
        const smallsize level_size = batch_size << level;
        const smallsize level_offset = level_size - batch_size;
        const Key *keys = level_keys + level_offset;
        smallsize index = lsm_lower_bound_original(keys, level_size, query);

        for (; index < level_size; ++index)
        {
            const Key key = lsm_paper_key<Key>::original(keys[index]);
            if (key >= result)
                break;
            if (!lsm_paper_key<Key>::is_regular(keys[index]))
                continue;
            if (index != 0 &&
                lsm_paper_key<Key>::original(keys[index - 1]) == key)
                continue;
            if (lsm_key_occurs_in_newer_level(
                    level_keys, level, num_batches, batch_size, key))
                continue;
            result = key;
            break;
        }
    }

    results[tid] = result;
}

template <typename Key>
GLOBALQUALIFIER void lsm_mark_cleanup_survivors_kernel(
    const Key *keys,
    std::uint8_t *flags,
    smallsize size)
{
    const smallsize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;

    const Key key = lsm_paper_key<Key>::original(keys[tid]);
    const bool first_in_segment =
        tid == 0 || lsm_paper_key<Key>::original(keys[tid - 1]) != key;
    flags[tid] = static_cast<std::uint8_t>(
        first_in_segment && lsm_paper_key<Key>::is_regular(keys[tid]));
}

template <typename Key>
GLOBALQUALIFIER void lsm_fill_placebos_kernel(
    Key *keys,
    smallsize *values,
    smallsize begin,
    smallsize end)
{
    const smallsize tid = blockIdx.x * blockDim.x + threadIdx.x + begin;
    if (tid >= end)
        return;
    keys[tid] = lsm_paper_key<Key>::placebo;
    values[tid] = not_found;
}

template <typename Key, typename Value>
void lsm_stable_merge_pairs(
    void *temp,
    size_t temp_bytes,
    const Key *newer_keys,
    const Value *newer_values,
    size_t newer_size,
    const Key *older_keys,
    const Value *older_values,
    size_t older_size,
    Key *output_keys,
    Value *output_values,
    cudaStream_t stream)
{
    cub::DeviceMerge::MergePairs(
        temp, temp_bytes,
        newer_keys, newer_values, newer_size,
        older_keys, older_values, older_size,
        output_keys, output_values,
        lsm_original_key_less<Key>{}, stream);
}

template <typename Key, typename Value>
size_t lsm_merge_temp_bytes(size_t first_size, size_t second_size)
{
    size_t bytes = 0;
    cub::DeviceMerge::MergePairs(
        nullptr, bytes,
        static_cast<const Key *>(nullptr),
        static_cast<const Value *>(nullptr), first_size,
        static_cast<const Key *>(nullptr),
        static_cast<const Value *>(nullptr), second_size,
        static_cast<Key *>(nullptr), static_cast<Value *>(nullptr),
        lsm_original_key_less<Key>{});
    return bytes;
}

template <typename T>
size_t lsm_select_temp_bytes(size_t size)
{
    size_t bytes = 0;
    cub::DeviceSelect::Flagged(
        nullptr, bytes,
        static_cast<const T *>(nullptr),
        static_cast<const std::uint8_t *>(nullptr),
        static_cast<T *>(nullptr),
        static_cast<smallsize *>(nullptr), size);
    return bytes;
}

template <typename key_type_, smallsize batch_size_log = 16,
          smallsize cg_size_log = 5>
class lsm_tree_ashkiani final
{
public:
    using key_type = key_type_;
    using value_type = smallsize;

    static_assert(std::is_same<key_type, std::uint32_t>::value,
                  "The paper's GPU LSM is restricted to 32-bit keys");
    static_assert(batch_size_log < 31, "LSMu batch size is too large");

    static constexpr key_type max_supported_key =
        lsm_paper_key<key_type>::max_user_key;
    static constexpr bool stores_tombstones = true;

private:
    static constexpr size_t threads_per_block = 256;
    static constexpr size_t batch_size = size_t{1} << batch_size_log;

    cuda_buffer<key_type> level_keys_buffer;
    cuda_buffer<smallsize> level_values_buffer;

    // One fixed-size input buffer and two ping-pong buffers implement the
    // paper's sort-and-stable-merge insertion path.
    cuda_buffer<key_type> batch_keys_buffer;
    cuda_buffer<smallsize> batch_values_buffer;
    cuda_buffer<key_type> temp_keys_buffer_a;
    cuda_buffer<smallsize> temp_values_buffer_a;
    cuda_buffer<key_type> temp_keys_buffer_b;
    cuda_buffer<smallsize> temp_values_buffer_b;
    cuda_buffer<std::uint8_t> primitive_temp_buffer;

    // Persistent workspace for the five-stage COUNT/RANGE pipeline in
    // Sections IV-C and IV-D. It grows to the largest query batch observed.
    cuda_buffer<smallsize> query_lower_indices_buffer;
    cuda_buffer<smallsize> query_upper_indices_buffer;
    cuda_buffer<smallsize> query_candidate_counts_buffer;
    cuda_buffer<smallsize> query_candidate_offsets_buffer;
    cuda_buffer<smallsize> query_segment_offsets_buffer;
    cuda_buffer<key_type> query_candidate_keys_buffer;
    cuda_buffer<smallsize> query_candidate_values_buffer;
    cuda_buffer<key_type> query_sorted_keys_buffer;
    cuda_buffer<smallsize> query_sorted_values_buffer;
    cuda_buffer<std::uint8_t> query_temp_buffer;

    size_t total_available_slots = 0;
    size_t max_level_size = 0;
    smallsize level_count = 0;
    std::uint64_t num_batches = 0;

    static size_t batches_for_elements(size_t size)
    {
        return size == 0 ? 0 : 1 + (size - 1) / batch_size;
    }

    template <typename T>
    static void ensure_capacity(cuda_buffer<T> &buffer, size_t count)
    {
        count = std::max<size_t>(count, 1);
        if (buffer.num_elements < count)
            buffer.resize(count);
    }

    static smallsize levels_for_batches(size_t required_batches)
    {
        smallsize levels = 1;
        std::uint64_t capacity = 1;
        while (capacity < std::max<size_t>(required_batches, 1))
        {
            if (levels == 31)
                throw std::overflow_error("LSMu exceeds its 32-bit level layout");
            ++levels;
            capacity = (std::uint64_t{1} << levels) - 1;
        }
        return levels;
    }

    size_t capacity_batches() const
    {
        return total_available_slots / batch_size;
    }

    void require_batch_capacity() const
    {
        if (num_batches >= capacity_batches())
            throw std::overflow_error(
                "LSMu capacity exhausted: max_size must include tombstones");
    }

    void sort_and_merge_current_batch(cudaStream_t stream)
    {
        require_batch_capacity();

        untimed_pair_sort(
            primitive_temp_buffer.raw_ptr,
            primitive_temp_buffer.size_in_bytes(),
            batch_keys_buffer.ptr(), temp_keys_buffer_a.ptr(),
            batch_values_buffer.ptr(), temp_values_buffer_a.ptr(),
            batch_size, stream);

        size_t target_level = 0;
        while ((num_batches & (std::uint64_t{1} << target_level)) != 0)
            ++target_level;
        if (target_level >= level_count)
            throw std::overflow_error("LSMu insertion carry exceeds allocated levels");

        const key_type *source_keys = temp_keys_buffer_a.ptr();
        const smallsize *source_values = temp_values_buffer_a.ptr();
        key_type *destination_keys = temp_keys_buffer_b.ptr();
        smallsize *destination_values = temp_values_buffer_b.ptr();
        size_t source_size = batch_size;

        for (size_t level = 0; level < target_level; ++level)
        {
            const size_t current_level_size = batch_size << level;
            const size_t level_offset = current_level_size - batch_size;
            lsm_stable_merge_pairs(
                primitive_temp_buffer.raw_ptr,
                primitive_temp_buffer.size_in_bytes(),
                source_keys, source_values, source_size,
                level_keys_buffer.ptr() + level_offset,
                level_values_buffer.ptr() + level_offset,
                current_level_size,
                destination_keys, destination_values, stream);

            // Figure 3, line 15: the carried level becomes empty.
            cudaMemsetAsync(level_keys_buffer.ptr() + level_offset, 0,
                            current_level_size * sizeof(key_type), stream);
            cudaMemsetAsync(level_values_buffer.ptr() + level_offset, 0,
                            current_level_size * sizeof(smallsize), stream);

            source_size += current_level_size;
            source_keys = destination_keys;
            source_values = destination_values;
            if (destination_keys == temp_keys_buffer_a.ptr())
            {
                destination_keys = temp_keys_buffer_b.ptr();
                destination_values = temp_values_buffer_b.ptr();
            }
            else
            {
                destination_keys = temp_keys_buffer_a.ptr();
                destination_values = temp_values_buffer_a.ptr();
            }
        }

        const size_t target_size = batch_size << target_level;
        const size_t target_offset = target_size - batch_size;
        cudaMemcpyAsync(level_keys_buffer.ptr() + target_offset, source_keys,
                        target_size * sizeof(key_type),
                        cudaMemcpyDefault, stream);
        cudaMemcpyAsync(level_values_buffer.ptr() + target_offset, source_values,
                        target_size * sizeof(smallsize),
                        cudaMemcpyDefault, stream);
        ++num_batches;
    }

    // Section V-B bulk build: radix-sort the complete initial set once, then
    // slice the sorted array into the occupied levels selected by num_batches.
    void bulk_build_data(
        const key_type *keys,
        size_t size,
        cudaStream_t stream)
    {
        if (size == 0)
            return;

        const size_t build_batches = batches_for_elements(size);
        const size_t resident_size = build_batches * batch_size;
        cuda_buffer<key_type> sorted_keys;
        cuda_buffer<smallsize> sorted_values;
        cuda_buffer<std::uint8_t> sort_temp;
        sorted_keys.alloc(resident_size);
        sorted_values.alloc(resident_size);
        const size_t sort_bytes =
            find_pair_sort_buffer_size<key_type, smallsize>(resident_size);
        sort_temp.alloc(sort_bytes);

        lsm_encode_bulk_build_kernel<<<
            SDIV(resident_size, threads_per_block),
            threads_per_block, 0, stream>>>(
                keys, static_cast<smallsize>(size),
                level_keys_buffer.ptr(), level_values_buffer.ptr(),
                static_cast<smallsize>(resident_size));
        untimed_pair_sort(
            sort_temp.raw_ptr, sort_temp.size_in_bytes(),
            level_keys_buffer.ptr(), sorted_keys.ptr(),
            level_values_buffer.ptr(), sorted_values.ptr(),
            resident_size, stream);

        cudaMemsetAsync(level_keys_buffer.ptr(), 0,
                        level_keys_buffer.size_in_bytes(), stream);
        cudaMemsetAsync(level_values_buffer.ptr(), 0,
                        level_values_buffer.size_in_bytes(), stream);

        size_t source_offset = 0;
        for (smallsize level = 0; level < level_count; ++level)
        {
            if ((build_batches & (size_t{1} << level)) == 0)
                continue;
            const size_t current_size = batch_size << level;
            const size_t current_offset = current_size - batch_size;
            cudaMemcpyAsync(level_keys_buffer.ptr() + current_offset,
                            sorted_keys.ptr() + source_offset,
                            current_size * sizeof(key_type),
                            cudaMemcpyDefault, stream);
            cudaMemcpyAsync(level_values_buffer.ptr() + current_offset,
                            sorted_values.ptr() + source_offset,
                            current_size * sizeof(smallsize),
                            cudaMemcpyDefault, stream);
            source_offset += current_size;
        }
        num_batches = build_batches;

        // The bulk-sort buffers are local and must outlive redistribution.
        cudaStreamSynchronize(stream);
        C2EX
    }

    template <bool CountOnly>
    void run_paper_range_pipeline(
        const key_type *lower,
        const key_type *upper,
        value_type *results,
        size_t size,
        cudaStream_t stream)
    {
        if (size == 0)
            return;
        if (size > std::numeric_limits<smallsize>::max())
            throw std::overflow_error("LSMu query count exceeds 32-bit layout");
        if (level_count != 0 &&
            size > (static_cast<size_t>(std::numeric_limits<int>::max()) - 1) /
                       level_count)
            throw std::overflow_error("LSMu query-level table exceeds CUB limits");

        const size_t slot_count = size * level_count;
        ensure_capacity(query_lower_indices_buffer, slot_count);
        ensure_capacity(query_upper_indices_buffer, slot_count);
        ensure_capacity(query_candidate_counts_buffer, slot_count + 1);
        ensure_capacity(query_candidate_offsets_buffer, slot_count + 1);
        ensure_capacity(query_segment_offsets_buffer, size + 1);

        lsm_query_bounds_kernel_paper<<<
            SDIV(slot_count, threads_per_block),
            threads_per_block, 0, stream>>>(
                level_keys_buffer.ptr(), lower, upper,
                query_lower_indices_buffer.ptr(),
                query_upper_indices_buffer.ptr(),
                query_candidate_counts_buffer.ptr(),
                static_cast<smallsize>(size), level_count, num_batches,
                static_cast<smallsize>(batch_size));
        cudaMemsetAsync(query_candidate_counts_buffer.ptr() + slot_count, 0,
                        sizeof(smallsize), stream);

        // Section IV-C/D, stage 2: one device-wide exclusive scan assigns
        // global positions to all per-level candidate ranges.
        size_t scan_bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            nullptr, scan_bytes,
            query_candidate_counts_buffer.ptr(),
            query_candidate_offsets_buffer.ptr(),
            static_cast<int>(slot_count + 1), stream);
        ensure_capacity(query_temp_buffer, scan_bytes);
        cub::DeviceScan::ExclusiveSum(
            query_temp_buffer.raw_ptr, scan_bytes,
            query_candidate_counts_buffer.ptr(),
            query_candidate_offsets_buffer.ptr(),
            static_cast<int>(slot_count + 1), stream);

        lsm_make_query_segment_offsets_kernel<<<
            SDIV(size + 1, threads_per_block),
            threads_per_block, 0, stream>>>(
                query_candidate_offsets_buffer.ptr(),
                query_segment_offsets_buffer.ptr(),
                static_cast<smallsize>(size), level_count);

        smallsize candidate_count = 0;
        cudaMemcpyAsync(&candidate_count,
                        query_candidate_offsets_buffer.ptr() + slot_count,
                        sizeof(candidate_count), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        C2EX

        if (candidate_count == 0)
        {
            cudaMemsetAsync(results, 0, size * sizeof(value_type), stream);
            return;
        }
        if (candidate_count > static_cast<smallsize>(
                                  std::numeric_limits<int>::max()))
            throw std::overflow_error("LSMu candidate array exceeds CUB limits");

        ensure_capacity(query_candidate_keys_buffer, candidate_count);
        ensure_capacity(query_sorted_keys_buffer, candidate_count);
        if constexpr (!CountOnly)
        {
            ensure_capacity(query_candidate_values_buffer, candidate_count);
            ensure_capacity(query_sorted_values_buffer, candidate_count);
        }

        // Stage 3: gather keys (COUNT) or key-value pairs (RANGE).
        constexpr size_t warps_per_block = threads_per_block / 32;
        lsm_gather_query_candidates_kernel_paper<key_type, !CountOnly><<<
            SDIV(size, 32 * warps_per_block),
            threads_per_block, 0, stream>>>(
                level_keys_buffer.ptr(), level_values_buffer.ptr(),
                query_lower_indices_buffer.ptr(),
                query_candidate_offsets_buffer.ptr(),
                query_candidate_keys_buffer.ptr(),
                CountOnly ? nullptr : query_candidate_values_buffer.ptr(),
                static_cast<smallsize>(size), level_count,
                static_cast<smallsize>(batch_size));

        // Stage 4: stable segmented radix sort by bits [31:1], deliberately
        // excluding the status bit so newest-first order is retained on ties.
        size_t segmented_sort_bytes = 0;
        if constexpr (CountOnly)
        {
            cub::DeviceSegmentedRadixSort::SortKeys(
                nullptr, segmented_sort_bytes,
                query_candidate_keys_buffer.ptr(),
                query_sorted_keys_buffer.ptr(),
                static_cast<int>(candidate_count), static_cast<int>(size),
                query_segment_offsets_buffer.ptr(),
                query_segment_offsets_buffer.ptr() + 1,
                1, sizeof(key_type) * 8, stream);
        }
        else
        {
            cub::DeviceSegmentedRadixSort::SortPairs(
                nullptr, segmented_sort_bytes,
                query_candidate_keys_buffer.ptr(),
                query_sorted_keys_buffer.ptr(),
                query_candidate_values_buffer.ptr(),
                query_sorted_values_buffer.ptr(),
                static_cast<int>(candidate_count), static_cast<int>(size),
                query_segment_offsets_buffer.ptr(),
                query_segment_offsets_buffer.ptr() + 1,
                1, sizeof(key_type) * 8, stream);
        }
        ensure_capacity(query_temp_buffer,
                        std::max(scan_bytes, segmented_sort_bytes));
        if constexpr (CountOnly)
        {
            cub::DeviceSegmentedRadixSort::SortKeys(
                query_temp_buffer.raw_ptr, segmented_sort_bytes,
                query_candidate_keys_buffer.ptr(),
                query_sorted_keys_buffer.ptr(),
                static_cast<int>(candidate_count), static_cast<int>(size),
                query_segment_offsets_buffer.ptr(),
                query_segment_offsets_buffer.ptr() + 1,
                1, sizeof(key_type) * 8, stream);
        }
        else
        {
            cub::DeviceSegmentedRadixSort::SortPairs(
                query_temp_buffer.raw_ptr, segmented_sort_bytes,
                query_candidate_keys_buffer.ptr(),
                query_sorted_keys_buffer.ptr(),
                query_candidate_values_buffer.ptr(),
                query_sorted_values_buffer.ptr(),
                static_cast<int>(candidate_count), static_cast<int>(size),
                query_segment_offsets_buffer.ptr(),
                query_segment_offsets_buffer.ptr() + 1,
                1, sizeof(key_type) * 8, stream);
        }

        // Stage 5: validate newest entries with warp-wide cooperation. RANGE
        // performs the requested aggregate instead of materializing pairs.
        if constexpr (CountOnly)
        {
            lsm_count_sorted_candidates_kernel_paper<<<
                SDIV(size, threads_per_block),
                threads_per_block, 0, stream>>>(
                    query_sorted_keys_buffer.ptr(),
                    query_segment_offsets_buffer.ptr(), results,
                    static_cast<smallsize>(size));
        }
        else
        {
            lsm_sum_sorted_candidates_kernel_paper<<<
                SDIV(size, threads_per_block),
                threads_per_block, 0, stream>>>(
                    query_sorted_keys_buffer.ptr(),
                    query_sorted_values_buffer.ptr(),
                    query_segment_offsets_buffer.ptr(), results,
                    static_cast<smallsize>(size));
        }
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
    static constexpr operation_support can_successor = operation_support::async;

    static std::string short_description()
    {
        return "lsm_tree_ashkiani";
    }

    static parameters_type parameters()
    {
        return {
            {"batch_size_log", std::to_string(batch_size_log)},
            {"threads_per_block", std::to_string(threads_per_block)},
            {"status_bit", "key_lsb"},
            {"stable_original_key_merge", "1"},
            {"cg_size", std::to_string(size_t{1} << cg_size_log)},
        };
    }

    static size_t estimate_build_bytes(size_t size)
    {
        const size_t batches = std::max<size_t>(batches_for_elements(size), 1);
        const smallsize levels = levels_for_batches(batches);
        const size_t capacity = batch_size * ((size_t{1} << levels) - 1);
        const size_t largest = batch_size << (levels - 1);
        const size_t records = capacity + batch_size + 2 * largest;
        const size_t resident = batches_for_elements(size) * batch_size;
        const size_t bulk_build_bytes =
            resident == 0
                ? 0
                : resident * (sizeof(key_type) + sizeof(value_type)) +
                      find_pair_sort_buffer_size<key_type, value_type>(resident);
        return records * (sizeof(key_type) + sizeof(value_type)) +
               std::max(find_pair_sort_buffer_size<key_type, value_type>(batch_size),
                        lsm_merge_temp_bytes<key_type, value_type>(
                            largest, largest)) +
               bulk_build_bytes;
    }

    size_t gpu_resident_bytes()
    {
        return level_keys_buffer.size_in_bytes() +
               level_values_buffer.size_in_bytes() +
               batch_keys_buffer.size_in_bytes() +
               batch_values_buffer.size_in_bytes() +
               temp_keys_buffer_a.size_in_bytes() +
               temp_values_buffer_a.size_in_bytes() +
               temp_keys_buffer_b.size_in_bytes() +
               temp_values_buffer_b.size_in_bytes() +
               primitive_temp_buffer.size_in_bytes() +
               query_lower_indices_buffer.size_in_bytes() +
               query_upper_indices_buffer.size_in_bytes() +
               query_candidate_counts_buffer.size_in_bytes() +
               query_candidate_offsets_buffer.size_in_bytes() +
               query_segment_offsets_buffer.size_in_bytes() +
               query_candidate_keys_buffer.size_in_bytes() +
               query_candidate_values_buffer.size_in_bytes() +
               query_sorted_keys_buffer.size_in_bytes() +
               query_sorted_values_buffer.size_in_bytes() +
               query_temp_buffer.size_in_bytes();
    }

    size_t gpu_resident_bytes_previous()
    {
        return level_keys_buffer.size_in_bytes() +
               level_values_buffer.size_in_bytes();
    }

    void build(
        const key_type *keys,
        size_t size,
        double *build_time_ms,
        size_t *build_bytes)
    {
        build(keys, size, size, std::numeric_limits<size_t>::max(),
              build_time_ms, build_bytes);
    }

    void build(
        const key_type *keys,
        size_t size,
        size_t max_size,
        size_t available_memory_bytes,
        double *build_time_ms,
        size_t *build_bytes)
    {
        if (level_keys_buffer.raw_ptr != nullptr)
            throw std::logic_error("LSMu::build called on an initialized index");
        if (size > std::numeric_limits<smallsize>::max())
            throw std::overflow_error("LSMu values are 32-bit offsets");

        const size_t required_batches = std::max<size_t>(
            batches_for_elements(std::max(size, max_size)), 1);
        level_count = levels_for_batches(required_batches);
        total_available_slots =
            batch_size * ((size_t{1} << level_count) - 1);
        max_level_size = batch_size << (level_count - 1);
        if (total_available_slots > std::numeric_limits<smallsize>::max())
            throw std::overflow_error(
                "LSMu paper layout uses 32-bit element indices");

        const size_t sort_bytes =
            find_pair_sort_buffer_size<key_type, smallsize>(batch_size);
        const size_t merge_bytes =
            lsm_merge_temp_bytes<key_type, smallsize>(
                max_level_size, max_level_size);
        const size_t primitive_bytes = std::max(sort_bytes, merge_bytes);
        const size_t initial_resident_size = batches_for_elements(size) * batch_size;
        const size_t bulk_build_bytes =
            initial_resident_size == 0
                ? 0
                : initial_resident_size *
                          (sizeof(key_type) + sizeof(smallsize)) +
                      find_pair_sort_buffer_size<key_type, smallsize>(
                          initial_resident_size);
        const size_t required_bytes =
            (total_available_slots + batch_size + 2 * max_level_size) *
                (sizeof(key_type) + sizeof(smallsize)) +
            primitive_bytes + bulk_build_bytes;
        if (required_bytes > available_memory_bytes)
            throw std::runtime_error(
                "not enough GPU memory for paper-faithful LSMu capacity");

        level_keys_buffer.alloc(total_available_slots);
        level_values_buffer.alloc(total_available_slots);
        batch_keys_buffer.alloc(batch_size);
        batch_values_buffer.alloc(batch_size);
        temp_keys_buffer_a.alloc(max_level_size);
        temp_values_buffer_a.alloc(max_level_size);
        temp_keys_buffer_b.alloc(max_level_size);
        temp_values_buffer_b.alloc(max_level_size);
        primitive_temp_buffer.alloc(primitive_bytes);
        C2EX

        level_keys_buffer.zero();
        level_values_buffer.zero();
        num_batches = 0;

        {
            scoped_cuda_timer timer(0, build_time_ms);
            if (size != 0)
                bulk_build_data(keys, size, 0);
        }
        cudaDeviceSynchronize();
        C2EX

        if (build_bytes)
            *build_bytes += gpu_resident_bytes();
    }

    void destroy()
    {
        if (level_keys_buffer.raw_ptr)
            level_keys_buffer.free();
        if (level_values_buffer.raw_ptr)
            level_values_buffer.free();
        if (batch_keys_buffer.raw_ptr)
            batch_keys_buffer.free();
        if (batch_values_buffer.raw_ptr)
            batch_values_buffer.free();
        if (temp_keys_buffer_a.raw_ptr)
            temp_keys_buffer_a.free();
        if (temp_values_buffer_a.raw_ptr)
            temp_values_buffer_a.free();
        if (temp_keys_buffer_b.raw_ptr)
            temp_keys_buffer_b.free();
        if (temp_values_buffer_b.raw_ptr)
            temp_values_buffer_b.free();
        if (primitive_temp_buffer.raw_ptr)
            primitive_temp_buffer.free();
        if (query_lower_indices_buffer.raw_ptr)
            query_lower_indices_buffer.free();
        if (query_upper_indices_buffer.raw_ptr)
            query_upper_indices_buffer.free();
        if (query_candidate_counts_buffer.raw_ptr)
            query_candidate_counts_buffer.free();
        if (query_candidate_offsets_buffer.raw_ptr)
            query_candidate_offsets_buffer.free();
        if (query_segment_offsets_buffer.raw_ptr)
            query_segment_offsets_buffer.free();
        if (query_candidate_keys_buffer.raw_ptr)
            query_candidate_keys_buffer.free();
        if (query_candidate_values_buffer.raw_ptr)
            query_candidate_values_buffer.free();
        if (query_sorted_keys_buffer.raw_ptr)
            query_sorted_keys_buffer.free();
        if (query_sorted_values_buffer.raw_ptr)
            query_sorted_values_buffer.free();
        if (query_temp_buffer.raw_ptr)
            query_temp_buffer.free();
        total_available_slots = 0;
        max_level_size = 0;
        level_count = 0;
        num_batches = 0;
    }

    void update(
        const key_type *insert_keys,
        const smallsize *insert_values,
        size_t insert_size,
        const key_type *delete_keys,
        size_t delete_size,
        cudaStream_t stream)
    {
        if (delete_size > std::numeric_limits<size_t>::max() - insert_size)
            throw std::overflow_error("LSMu update size overflow");
        const size_t total = insert_size + delete_size;
        if (total == 0)
            return;
        if (total > batch_size)
            throw std::invalid_argument(
                "GPU LSM updates must contain at most the fixed batch size b");

        // Figure 3 accepts one mixed batch. A short batch is completed by
        // duplicating its last member exactly as described in Section IV-A.
        lsm_encode_update_batch_kernel<<<
            SDIV(batch_size, threads_per_block),
            threads_per_block, 0, stream>>>(
                insert_keys, insert_values,
                static_cast<smallsize>(insert_size),
                delete_keys, static_cast<smallsize>(delete_size),
                static_cast<smallsize>(total),
                batch_keys_buffer.ptr(), batch_values_buffer.ptr(),
                static_cast<smallsize>(batch_size));
        sort_and_merge_current_batch(stream);
    }

    void insert(
        const key_type *insert_list,
        const smallsize *positions,
        size_t size,
        cudaStream_t stream)
    {
        update(insert_list, positions, size, nullptr, 0, stream);
    }

    void remove(
        const key_type *delete_list,
        size_t size,
        cudaStream_t stream)
    {
        update(nullptr, nullptr, 0, delete_list, size, stream);
    }

    // Compatibility with the benchmark's equal-sized combined-update API.
    void insert_and_remove(
        const key_type *insert_list,
        const smallsize *positions,
        size_t size,
        const key_type *delete_list,
        cudaStream_t stream)
    {
        update(insert_list, positions, size, delete_list, size, stream);
    }

    void lookup(
        const key_type *keys,
        value_type *results,
        size_t size,
        cudaStream_t stream)
    {
        if (size == 0)
            return;
        lsm_lookup_kernel_paper<<<
            SDIV(size, threads_per_block),
            threads_per_block, 0, stream>>>(
                level_keys_buffer.ptr(), level_values_buffer.ptr(),
                keys, results, static_cast<smallsize>(size),
                level_count, num_batches,
                static_cast<smallsize>(batch_size));
    }

    void count(
        const key_type *lower,
        const key_type *upper,
        value_type *results,
        size_t size,
        cudaStream_t stream)
    {
        run_paper_range_pipeline<true>(
            lower, upper, results, size, stream);
    }

    void range_lookup_sum(
        const key_type *lower,
        const key_type *upper,
        value_type *results,
        size_t size,
        cudaStream_t stream)
    {
        run_paper_range_pipeline<false>(
            lower, upper, results, size, stream);
    }

    void multi_lookup_sum(
        const key_type *keys,
        value_type *results,
        size_t size,
        cudaStream_t stream)
    {
        range_lookup_sum(keys, keys, results, size, stream);
    }

    void lookups_successor(
        const key_type *keys,
        key_type *results,
        size_t size,
        cudaStream_t stream)
    {
        if (size == 0)
            return;
        lsm_successor_kernel_paper<<<
            SDIV(size, threads_per_block),
            threads_per_block, 0, stream>>>(
                level_keys_buffer.ptr(), keys, results,
                static_cast<smallsize>(size), level_count, num_batches,
                static_cast<smallsize>(batch_size));
    }

    // Section IV-E CLEANUP. It is intentionally explicit, as in the paper:
    // users choose when the cost is worthwhile.
    size_t cleanup(cudaStream_t stream = 0)
    {
        const size_t resident_size = num_batches * batch_size;
        if (resident_size == 0)
            return 0;

        cuda_buffer<key_type> cleanup_keys_a;
        cuda_buffer<smallsize> cleanup_values_a;
        cuda_buffer<key_type> cleanup_keys_b;
        cuda_buffer<smallsize> cleanup_values_b;
        cuda_buffer<std::uint8_t> cleanup_flags;
        cuda_buffer<smallsize> selected_count;

        cleanup_keys_a.alloc(total_available_slots);
        cleanup_values_a.alloc(total_available_slots);
        cleanup_keys_b.alloc(total_available_slots);
        cleanup_values_b.alloc(total_available_slots);
        cleanup_flags.alloc(total_available_slots);
        selected_count.alloc(1);
        C2EX

        const size_t cleanup_merge_bytes =
            lsm_merge_temp_bytes<key_type, smallsize>(
                total_available_slots, max_level_size);
        const size_t cleanup_select_bytes = std::max(
            lsm_select_temp_bytes<key_type>(resident_size),
            lsm_select_temp_bytes<smallsize>(resident_size));
        cuda_buffer<std::uint8_t> cleanup_temp;
        cleanup_temp.alloc(std::max(cleanup_merge_bytes, cleanup_select_bytes));
        C2EX

        key_type *merged_keys = cleanup_keys_a.ptr();
        smallsize *merged_values = cleanup_values_a.ptr();
        size_t merged_size = 0;

        // Stable, newest-to-oldest iterative merge of all occupied levels.
        for (smallsize level = 0; level < level_count; ++level)
        {
            if ((num_batches & (std::uint64_t{1} << level)) == 0)
                continue;
            const size_t current_size = batch_size << level;
            const size_t current_offset = current_size - batch_size;

            if (merged_size == 0)
            {
                cudaMemcpyAsync(merged_keys,
                                level_keys_buffer.ptr() + current_offset,
                                current_size * sizeof(key_type),
                                cudaMemcpyDefault, stream);
                cudaMemcpyAsync(merged_values,
                                level_values_buffer.ptr() + current_offset,
                                current_size * sizeof(smallsize),
                                cudaMemcpyDefault, stream);
                merged_size = current_size;
                continue;
            }

            key_type *output_keys =
                merged_keys == cleanup_keys_a.ptr()
                    ? cleanup_keys_b.ptr()
                    : cleanup_keys_a.ptr();
            smallsize *output_values =
                merged_values == cleanup_values_a.ptr()
                    ? cleanup_values_b.ptr()
                    : cleanup_values_a.ptr();
            lsm_stable_merge_pairs(
                cleanup_temp.raw_ptr, cleanup_temp.size_in_bytes(),
                merged_keys, merged_values, merged_size,
                level_keys_buffer.ptr() + current_offset,
                level_values_buffer.ptr() + current_offset,
                current_size, output_keys, output_values, stream);
            merged_keys = output_keys;
            merged_values = output_values;
            merged_size += current_size;
        }

        lsm_mark_cleanup_survivors_kernel<<<
            SDIV(merged_size, threads_per_block),
            threads_per_block, 0, stream>>>(
                merged_keys, cleanup_flags.ptr(),
                static_cast<smallsize>(merged_size));

        key_type *compact_keys =
            merged_keys == cleanup_keys_a.ptr()
                ? cleanup_keys_b.ptr()
                : cleanup_keys_a.ptr();
        smallsize *compact_values =
            merged_values == cleanup_values_a.ptr()
                ? cleanup_values_b.ptr()
                : cleanup_values_a.ptr();

        size_t select_temp_bytes = cleanup_temp.size_in_bytes();
        cub::DeviceSelect::Flagged(
            cleanup_temp.raw_ptr, select_temp_bytes,
            merged_keys, cleanup_flags.ptr(), compact_keys,
            selected_count.ptr(), merged_size, stream);
        select_temp_bytes = cleanup_temp.size_in_bytes();
        cub::DeviceSelect::Flagged(
            cleanup_temp.raw_ptr, select_temp_bytes,
            merged_values, cleanup_flags.ptr(), compact_values,
            selected_count.ptr(), merged_size, stream);

        smallsize valid_size = 0;
        cudaMemcpyAsync(&valid_size, selected_count.ptr(), sizeof(valid_size),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        C2EX

        const size_t rebuilt_batches = batches_for_elements(valid_size);
        const size_t rebuilt_size = rebuilt_batches * batch_size;
        if (rebuilt_batches > capacity_batches())
            throw std::overflow_error("LSMu cleanup output exceeds capacity");

        if (rebuilt_size > valid_size)
        {
            lsm_fill_placebos_kernel<<<
                SDIV(rebuilt_size - valid_size, threads_per_block),
                threads_per_block, 0, stream>>>(
                    compact_keys, compact_values, valid_size,
                    static_cast<smallsize>(rebuilt_size));
        }

        cudaMemsetAsync(level_keys_buffer.ptr(), 0,
                        level_keys_buffer.size_in_bytes(), stream);
        cudaMemsetAsync(level_values_buffer.ptr(), 0,
                        level_values_buffer.size_in_bytes(), stream);

        size_t source_offset = 0;
        for (smallsize level = 0; level < level_count; ++level)
        {
            if ((rebuilt_batches & (size_t{1} << level)) == 0)
                continue;
            const size_t current_size = batch_size << level;
            const size_t current_offset = current_size - batch_size;
            cudaMemcpyAsync(level_keys_buffer.ptr() + current_offset,
                            compact_keys + source_offset,
                            current_size * sizeof(key_type),
                            cudaMemcpyDefault, stream);
            cudaMemcpyAsync(level_values_buffer.ptr() + current_offset,
                            compact_values + source_offset,
                            current_size * sizeof(smallsize),
                            cudaMemcpyDefault, stream);
            source_offset += current_size;
        }
        num_batches = rebuilt_batches;

        // Local cleanup buffers must remain alive through redistribution.
        cudaStreamSynchronize(stream);
        C2EX
        return valid_size;
    }

    void dump_tree()
    {
        for (smallsize level = 0; level < level_count; ++level)
        {
            if ((num_batches & (std::uint64_t{1} << level)) == 0)
                continue;
            const size_t current_size = batch_size << level;
            const size_t current_offset = current_size - batch_size;
            std::cerr << "LEVEL " << level << " (encoded keys): ";
            std::vector<key_type> values(current_size);
            cudaMemcpy(values.data(),
                       level_keys_buffer.ptr() + current_offset,
                       current_size * sizeof(key_type),
                       cudaMemcpyDeviceToHost);
            for (const key_type key : values)
                std::cerr << lsm_paper_key<key_type>::original(key)
                          << (lsm_paper_key<key_type>::is_regular(key)
                                  ? "+ "
                                  : "- ");
            std::cerr << '\n';
        }
    }
};

#endif
