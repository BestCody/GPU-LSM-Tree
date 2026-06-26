// =============================================================================
// File: device_code_coarse_granular.cu
// Author: Justus Henneberg
// Description: Launch experiments     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#include "launch_parameters.cuh"

#include "definitions_coarse_granular.cuh"


extern "C" __constant__ cg_params params;


template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER
key_type extract_key(const void* buf, smallsize i) {
    return cg::extract<key_type>(buf, i * (sizeof(key_type) + sizeof(smallsize)));
}


template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER
smallsize extract_offset(const void* buf, smallsize i) {
    return cg::extract<smallsize>(buf, i * (sizeof(key_type) + sizeof(smallsize)) + sizeof(key_type));
}


template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER
smallsize find_range_start_in_partition(const cg_params& params, smallsize partition_offset, key_type lower_bound) {
    const auto buf = params.ordered_key_offset_pairs;

    const smallsize first_offset = params.partition_size * partition_offset;
    const smallsize last_offset = min(first_offset + params.partition_size, params.stored_size);

    smallsize search_offset = last_offset - 1;
    smallsize initial_skip = 31u - __clz(last_offset - first_offset);
    for (smallsize skip = uint32_t(1) << initial_skip; skip > 0; skip >>= 1u) {
        if (search_offset < first_offset + skip) continue;
        if (extract_key<key_type>(buf, search_offset - skip) >= lower_bound)
            search_offset -= skip;
    }
    return search_offset;
}


template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER
void perform_lookup() {
    constexpr smallsize rays_per_thread = 1;

    const smallsize num_threads = optixGetLaunchDimensions().x;
    const smallsize thread_offset = optixGetLaunchIndex().x;
    const smallsize last_thread_offset = rays_per_thread * num_threads;
    const smallsize thread_increment = num_threads;

    const auto buf = params.ordered_key_offset_pairs;

    const key_type min_representative = extract_key<key_type>(buf, min(params.stored_size, params.partition_size) - 1);
    const key_type max_inserted_key = extract_key<key_type>(buf, params.stored_size - 1);

    for (smallsize ix = thread_offset; ix < last_thread_offset; ix += thread_increment) {

        key_type lower_bound = reinterpret_cast<const key_type*>(params.query_lower)[ix];
        key_type upper_bound = reinterpret_cast<const key_type*>(params.query_upper)[ix];

        smallsize partition_offset = not_found;
        if (lower_bound > max_inserted_key) {
            // key out of range, skip search entirely
        } else if (lower_bound <= min_representative) {
            // range starts before the smallest key, start searching in partition 0
            partition_offset = 0;
        } else {
            // perform tracing to find the partition
            partition_offset = cg::find_partition_offset_with_rays(params.traversable, params.partition_count, lower_bound);
        }

        // use with external partition search
        if (params.find_partition_only) {
            params.result[ix] = partition_offset;
            continue;
        }

        // search within partition
        smallsize result_offset = not_found;
        if (partition_offset != not_found) {
            result_offset = find_range_start_in_partition(params, partition_offset, lower_bound);
        }

        // compute result
        smallsize result;
        if (params.aggregate_results) {
            result = 0;
            // yes, this will work even when result_offset == not_found because the loop will never execute
            for (smallsize i = result_offset; i < params.stored_size; ++i) {
                key_type current_key = extract_key<key_type>(buf, i);
                // this case occurs when a range query starts between the initial position and the final position of a moved representative
                // as a result of moving the representative, the range query starts the scan one partition too early,
                // so we need to skip the last element of that partition to arrive at the correct partition
                if (current_key < lower_bound) continue;
                if (current_key > upper_bound) break;

                result += extract_offset<key_type>(buf, i);
            }
        } else {
            result = not_found;
            if (result_offset != not_found && extract_key<key_type>(buf, result_offset) == lower_bound) {
                result = extract_offset<key_type>(buf, result_offset);
            }
        }
        params.result[ix] = result;
    }
}


extern "C" GLOBALQUALIFIER void __raygen__lookup() {
    if (params.long_keys) {
        perform_lookup<key64>();
    } else {
        perform_lookup<key32>();
    }
}
