// =============================================================================
// File: impl_opt_binsearch.cuh
// Author: Justus Henneberg
// Description: Implements impl_opt_binsearch     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef IMPL_OPT_BINSEARCH_CUH
#define IMPL_OPT_BINSEARCH_CUH


// experimental: test various upgrades for binary search
// todo: efficient updatability with buckets, efficient range queries

#include <cmath>

#include <cub/cub.cuh>

/*
namespace sa {
    struct opt {
        static constexpr uint8_t none = 0;
        static constexpr uint8_t cul = 1;
        static constexpr uint8_t culp = cul | 2;
        static constexpr uint8_t slb = 4;
        static constexpr uint8_t sulb = slb | 8;

        static constexpr bool cache_upper_levels(uint8_t flags) {
            return flags & cul;
        }

        static constexpr bool cache_upper_levels_partial(uint8_t flags) {
            return flags & 2;
        }

        static constexpr bool sort_lookups_block(uint8_t flags) {
            return flags & slb;
        }

        static constexpr bool sort_unsort_lookups_block(uint8_t flags) {
            return flags & 8;
        }
    };

    template <typename key_type, uint16_t threads_per_block, uint16_t registers_per_thread>
    struct helper_algorithms {
        using sort = cub::BlockRadixSort<key_type, threads_per_block, registers_per_thread, uint16_t>;
        using sort_temp = typename sort::TempStorage;
        using shuffle = cub::BlockExchange<smallsize, threads_per_block, registers_per_thread>;
        using shuffle_temp = typename shuffle::TempStorage;
    };
}

// todo cooperative range query


DEVICEQUALIFIER INLINEQUALIFIER
smallsize ilog2_gpu(smallsize value) {
    return value == 0 ? 0 : 31u - __clz(value);
}


inline
smallsize ilog2_cpu(smallsize value) {
    return value == 0 ? 0 : 31u - __builtin_clz(value);
}


template <typename key_type>
GLOBALQUALIFIER
void extract_strided_kernel(
        const key_type* keys,
        key_type* extracted_keys,
        smallsize extract_count,
        smallsize stride_log
) {
    const smallsize tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= extract_count) return;

    extracted_keys[tid] = keys[tid << stride_log];
}


template <typename key_type>
void extract_strided(
        const key_type* keys,
        key_type* extracted_keys,
        size_t extract_count,
        size_t stride_log,
        cudaStream_t stream
) {
    extract_strided_kernel<<<SDIV(extract_count, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
            keys, extracted_keys, extract_count, stride_log);
}


template <typename key_type>
void extract_strided_timed(
        const key_type* keys,
        key_type* extracted_keys,
        size_t extract_count,
        size_t stride_log,
        cudaStream_t stream,
        double* time_ms
) {
    scoped_cuda_timer timer(0, time_ms);
    extract_strided(keys, extracted_keys, extract_count, stride_log, stream);
}


template <typename element_type>
DEVICEQUALIFIER INLINEQUALIFIER
smallsize reverse_device_binary_search(element_type key, const void* buf, smallsize initial_offset, smallsize initial_skip) {
    smallsize match_offset = initial_offset;
    for (smallsize skip = initial_skip; skip > 0; skip >>= 1u) {
        if (match_offset < skip)
            continue;

        auto current = cg::extract<element_type>(buf, (match_offset - skip) * sizeof(element_type));
        if (current >= key)
            match_offset -= skip;
    }
    return match_offset;
}


template <typename element_type>
DEVICEQUALIFIER INLINEQUALIFIER
smallsize reverse_device_binary_search(element_type key, const void* buf, smallsize entries_count) {
    return reverse_device_binary_search(key, buf, entries_count - 1, smallsize(1) << ilog2_gpu(entries_count));
}


template <typename element_type>
DEVICEQUALIFIER
smallsize mixed_granularity_reverse_device_binary_search(
        element_type lookup,
        const element_type* base_entries,
        smallsize base_entries_count,
        const element_type* strided_entries,
        const element_type* partial_level_entries,
        smallsize cached_entries_count, // must not be zero!
        smallsize partial_level_cached_entries_count,
        smallsize stride_log
) {
    smallsize initial_stride = smallsize(1) << ilog2_gpu(cached_entries_count);
    smallsize coarse_offset = reverse_device_binary_search(lookup, strided_entries, cached_entries_count - 1, initial_stride);

    smallsize reverse_offset = cached_entries_count - 1 - coarse_offset;
    smallsize continue_offset = base_entries_count - 1 - (reverse_offset << stride_log);
    smallsize continue_stride = smallsize(1) << (stride_log - 1);

    if (reverse_offset < partial_level_cached_entries_count) {
        if (partial_level_entries[partial_level_cached_entries_count - 1 - reverse_offset] >= lookup) {
            continue_offset -= continue_stride;
        }
        continue_stride >>= 1u;
    }

    smallsize final_offset = reverse_device_binary_search(lookup, base_entries, continue_offset, continue_stride);
    return final_offset;
}


// todo reverse order of temp mem and cache
template <typename key_type, uint8_t opt, uint16_t threads_per_block, uint16_t registers_per_thread, bool range_query>
GLOBALQUALIFIER
void optimized_binsearch_lookup_kernel(
        const key_type* sorted_keys,
        const smallsize* sorted_offsets,
        smallsize base_entries_count,
        const uint8_t* shared_memory_template,
        smallsize shared_bytes,
        smallsize stride_log,
        smallsize cached_entries_count,
        smallsize partial_level_cached_entries_count,
        const key_type* lookups,
        const key_type* upper_limits,
        smallsize* result,
        smallsize size
) {
    extern __shared__ uint8_t shared_memory[];

    using algs = sa::helper_algorithms<key_type, threads_per_block, registers_per_thread>;

    const auto strided_entries = reinterpret_cast<key_type*>(shared_memory);
    const auto partial_level_entries = reinterpret_cast<key_type*>(shared_memory + sizeof(key_type) * cached_entries_count);
    const auto block_sort_temp = reinterpret_cast<typename algs::sort_temp*>(shared_memory + shared_bytes);
    const auto block_shuffle_temp = reinterpret_cast<typename algs::shuffle_temp*>(shared_memory + shared_bytes);

    const smallsize bid = blockIdx.x;
    const smallsize local_tid = threadIdx.x;
    const smallsize block_size = threads_per_block;
    const smallsize grid_size = threads_per_block * gridDim.x;

    if constexpr (sa::opt::cache_upper_levels(opt)) {
        for (smallsize i = local_tid; i < shared_bytes; i += block_size) {
            shared_memory[i] = shared_memory_template[i];
        }
        __syncthreads();
    }

    key_type local_lookups[registers_per_thread];
    key_type local_upper_limits[registers_per_thread];
    uint16_t local_ranks[registers_per_thread];
    smallsize local_results[registers_per_thread];

    const smallsize elements_per_block = block_size * registers_per_thread;
    const smallsize grid_stride = grid_size * registers_per_thread;

    for (smallsize lookup_offset = bid * elements_per_block; lookup_offset < size; lookup_offset += grid_stride) {
        const auto block_count = std::min(elements_per_block, size - lookup_offset);

        #pragma unroll
        for (smallsize i = 0; i < registers_per_thread; ++i) {
            const auto block_rank = i * block_size + local_tid;
            local_lookups[i] = block_rank < block_count ? lookups[lookup_offset + block_rank] : std::numeric_limits<key_type>::max();
            if constexpr (range_query) {
                local_upper_limits[i] = block_rank < block_count ? upper_limits[lookup_offset + block_rank] : std::numeric_limits<key_type>::max();
            }
            local_ranks[i] = block_rank;
        }

        if constexpr (sa::opt::sort_lookups_block(opt)) {
            algs::sort(*block_sort_temp).SortBlockedToStriped(local_lookups, local_ranks);
            __syncthreads();
        }

        #pragma unroll
        for (smallsize i = 0; i < registers_per_thread; ++i) {
            if (local_ranks[i] >= block_count) continue;

            smallsize pos;
            if (cached_entries_count > 0) {
                pos = mixed_granularity_reverse_device_binary_search(
                        local_lookups[i],
                        sorted_keys,
                        base_entries_count,
                        strided_entries,
                        partial_level_entries,
                        cached_entries_count,
                        partial_level_cached_entries_count,
                        stride_log);
            } else {
                pos = reverse_device_binary_search(local_lookups[i], sorted_keys, base_entries_count);
            }

            smallsize local_result;
            if constexpr (range_query) {
                local_result = 0;
                for (smallsize it = pos; it < base_entries_count; ++it) {
                    if (sorted_keys[it] > local_lookups[i])
                        break;
                    local_result += sorted_offsets[it];

                    // todo start dynpar kernel here if query size exceeds XXXXX elements, and pass current result
                    // todo do not dynpar if gpu is sufficiently busy (set max thread allowance core_count/lookup_count)
                    // todo maybe keep track of current gpu load
                }
            } else {
                local_result = sorted_keys[pos] == local_lookups[i] ? sorted_offsets[pos] : not_found;
            }

            if constexpr (sa::opt::sort_unsort_lookups_block(opt)) {
                // delayed write
                local_results[i] = local_result;
            } else {
                // immediate write
                result[lookup_offset + local_ranks[i]] = local_result;
            }
        }

        if constexpr (sa::opt::sort_unsort_lookups_block(opt)) {
            algs::shuffle(*block_shuffle_temp).ScatterToStriped(local_results, local_results, local_ranks);
            __syncthreads();

            #pragma unroll
            for (smallsize i = 0; i < registers_per_thread; ++i) {
                const auto block_rank = i * block_size + local_tid;
                if (block_rank >= block_count) continue;
                result[lookup_offset + block_rank] = local_results[i];
            }
        }
    }
}


uint32_t get_max_shared_memory_per_block_optin_bytes() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.sharedMemPerBlockOptin;
}


uint32_t get_number_of_sms() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.multiProcessorCount;
}


template <typename key_type_, uint8_t opt, uint16_t threads_per_block = 128, uint16_t registers_per_thread = 8>
class opt_sorted_array {
public:
    using key_type = key_type_;

private:
    cuda_buffer_async<key_type> sorted_keys_buffer;
    cuda_buffer_async<smallsize> sorted_offsets_buffer;
    size_t base_entries_count = 0;

    cuda_buffer<uint8_t> shared_buffer;
    size_t stride_log = 0;
    size_t cached_entries_count = 0;
    size_t partial_level_cached_entries_count = 0;

    size_t shared_bytes_to_load = 0;
    size_t shared_bytes_for_shuffle = 0;

    uint32_t number_of_sms = 0;
    size_t max_shared_memory_bytes = 0;

    size_t shmem() {
        return shared_bytes_to_load + shared_bytes_for_shuffle;
    }

    void rebuild_shared_buffer(cudaStream_t stream) {
        if constexpr (!sa::opt::cache_upper_levels(opt)) return;

        constexpr bool partial = sa::opt::cache_upper_levels_partial(opt);
        constexpr size_t key_size = sizeof(key_type);

        size_t remaining_bytes = max_shared_memory_bytes - shared_bytes_for_shuffle;

        // smallest possible power-of-2 stride so that strided entries fit into shared memory
        stride_log = ilog2_cpu(key_size * base_entries_count / (remaining_bytes - key_size)) + 1;
        size_t stride = size_t(1) << stride_log;

        cached_entries_count = SDIV(size, stride);
        size_t strided_entries_bytes = footprint<key_type>(cached_entries_count);

        partial_level_cached_entries_count = !partial ? 0 : (remaining_bytes - strided_entries_bytes) / key_size;
        size_t partial_level_entries_bytes = footprint<key_type>(partial_level_cached_entries_count);

        shared_bytes_to_load = strided_entries_bytes + partial_level_entries_bytes;

        // __X___X___X___X
        // stride=4 len=15 -> ofst = (len-1) % stride
        extract_strided(
            sorted_keys_buffer.ptr() + (base_entries_count - 1) % stride,
            reinterpret_cast<key_type*>(shared_buffer.ptr() + shared_bytes_for_shuffle),
            cached_entries_count,
            stride_log;

        // __X___X_Y_X_Y_X
        // stride=4 len=15 partial=2 -> ofst = (len-1)+stride/2-stride*partial
        if (partial && stride > 1 && partial_level_cached_entries_count > 0) {
            extract_strided(
                sorted_keys_buffer.ptr() + (base_entries_count - 1) + (stride >> 1u) - partial_level_cached_entries_count * stride,
                reinterpret_cast<key_type*>(shared_buffer.ptr() + shared_bytes_for_shuffle + footprint<key_type>(cached_entries_count)),
                partial_level_cached_entries_count,
                stride_log);
        }
    }

public:
    static constexpr bool can_lookup = true;
    static constexpr bool can_multi_lookup = true;
    static constexpr bool can_range_lookup = true;
    static constexpr bool can_update = true;

    static std::string short_description() {
        std::string desc = "opt_sorted_array";
        if (sa::opt::cache_upper_levels_partial(opt))
            desc += "_culp";
        else if (sa::opt::cache_upper_levels(opt))
            desc += "_ulp";
        if (sa::opt::sort_unsort_lookups_block(opt))
            desc += "_sulb";
        else if (sa::opt::sort_lookups_block(opt))
            desc += "_slb";
        desc += "_" + std::to_string(threads_per_block);
        desc += "_" + std::to_string(registers_per_thread);
        return desc;
    }

    static size_t estimate_build_bytes(size_t size) {
        size_t sort_bytes = (sizeof(smallsize) + sizeof(key_type)) * size;
        size_t sort_aux_bytes = sizeof(smallsize) * size + find_pair_sort_buffer_size<key_type, smallsize>(size);
        size_t shared_bytes = sizeof(key_type) * size;
        return sort_bytes + sort_aux_bytes + shared_bytes;
    }

    size_t gpu_resident_bytes() {
        return sorted_keys_buffer.size_in_bytes() +
            sorted_offsets_buffer.size_in_bytes() +
            shared_buffer.size_in_bytes();
    }

    void build(const key_type* keys, size_t size, double* build_time_ms, size_t* build_bytes) {

        using algs = sa::helper_algorithms<key_type, threads_per_block, registers_per_thread>;

        number_of_sms = get_number_of_sms();
        max_shared_memory_bytes = get_max_shared_memory_per_block_optin_bytes();
        base_entries_count = size;

        size_t temp_storage_bytes = 0;
        {
            cuda_buffer<uint8_t> temp_buffer;
            cuda_buffer<smallsize> offsets_buffer;

            offsets_buffer.alloc(size); CUERR
            init_offsets(offsets_buffer.ptr(), size, build_time_ms);

            sorted_keys_buffer.alloc(size, 0); CUERR
            sorted_offsets_buffer.alloc(size, 0); CUERR

            cudaDeviceSynchronize(); CUERR

            temp_storage_bytes = find_pair_sort_buffer_size<key_type, smallsize>(size);
            temp_buffer.alloc(temp_storage_bytes); CUERR
            timed_pair_sort(
                temp_buffer.raw_ptr, temp_storage_bytes,
                keys, sorted_keys_buffer.ptr(), offsets_buffer.ptr(), sorted_offsets_buffer.ptr(), size, build_time_ms);

            cudaDeviceSynchronize(); CUERR
        }

        if constexpr (sa::opt::sort_lookups_block(opt)) {
            shared_bytes_to_load = std::max(sizeof(typename algs::sort_temp), sizeof(typename algs::shuffle_temp));
            // add padding so that shared_bytes_to_load is a multiple of 8 bytes
            shared_bytes_to_load = (shared_bytes_to_load + 7) & ~size_t(7);
        }

        // only allocate/extend shared memory if cache_upper_levels is enabled
        if constexpr (sa::opt::cache_upper_levels(opt)) {
            shared_buffer.alloc(max_shared_memory_bytes); CUERR

            cudaFuncSetAttribute(
                optimized_binsearch_lookup_kernel<key_type, opt, threads_per_block, registers_per_thread>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                max_shared_memory_bytes); CUERR
        }

        {
            scoped_cuda_timer timer(0, build_time_ms);
            rebuild_shared_buffer(0);
        }

        if (build_bytes) *build_bytes += sorted_keys_buffer.size_in_bytes() + sorted_offsets_buffer.size_in_bytes() +
                std::max<size_t>(temp_storage_bytes + sizeof(smallsize) * size, shared_buffer.size_in_bytes());

        cudaDeviceSynchronize(); CUERR
    }

    void lookup(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        optimized_binsearch_lookup_kernel
                <key_type, opt, threads_per_block, registers_per_thread, false>
                <<<number_of_sms, threads_per_block, shmem(), stream>>>(
            sorted_keys_buffer.ptr(), sorted_offsets_buffer.ptr(), base_entries_count,
            shared_buffer.ptr(), shared_bytes_to_load, stride_log, cached_entries_count, partial_level_cached_entries_count,
            keys, result, nullptr, size);
    }

    void range_lookup_sum(const key_type* lower, const key_type* upper, smallsize* result, size_t size, cudaStream_t stream) {
        optimized_binsearch_lookup_kernel
                <key_type, opt, threads_per_block, registers_per_thread, true>
                <<<number_of_sms, threads_per_block, shmem(), stream>>>(
            sorted_keys_buffer.ptr(), sorted_offsets_buffer.ptr(), base_entries_count,
            shared_buffer.ptr(), shared_bytes_to_load, stride_log, cached_entries_count, partial_level_cached_entries_count,
            keys, lower, upper, size);
    }

    void multi_lookup_sum(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        range_lookup_sum(keys, keys, result, size, stream);
    }

    void destroy() {
        sorted_keys_buffer.free();
        sorted_offsets_buffer.free();
        shared_buffer.free();
    }

    void insert(const key_type* update_list, const smallsize* offsets, size_t size, cudaStream_t stream) {

        cuda_buffer_async<uint8_t> temp_buffer;
        cuda_buffer_async<key_type> new_sorted_keys_buffer;
        cuda_buffer_async<smallsize> new_sorted_offsets_buffer;
        cuda_buffer_async<key_type> sorted_inserts_keys_buffer;
        cuda_buffer_async<smallsize> sorted_inserts_offsets_buffer;

        size_t sort_temp_storage_bytes;
        size_t merge_temp_storage_bytes;
        // todo find sizes


        temp_buffer.alloc(sort_temp_storage_bytes + merge_temp_storage_bytes, stream);

        sorted_inserts_keys_buffer.alloc(size, stream);
        sorted_inserts_offsets_buffer.alloc(size, stream);

        new_sorted_keys_buffer.alloc(base_entries_count + size, stream);
        new_sorted_offsets_buffer.alloc(base_entries_count + size, stream);

        // todo fill holes
        cub::DeviceRadixSort::SortPairs(
                temp_buffer.raw_ptr, sort_temp_storage_bytes,
                KEYS_IN, KEYS_OUT, VALS_IN, VALS_OUT, COUNT,
                0, sizeof(key_type) * 8, stream);

        cub::DeviceMerge::MergePairs(
                temp_buffer.raw_ptr, merge_temp_storage_bytes,
                STORED_KEYS, STORED_VALUES, STORED_COUNT,
                UPDATE_KEYS, UPDATE_VALUES, UPDATE_COUNT,
                KEYS_OUT, VALUES_OUT,
                {}, stream);

        std::swap(new_sorted_keys_buffer, sorted_keys_buffer);
        stad::swap(new_sorted_offsets_buffer, sorted_offsets_buffer);

        temp_buffer.free(stream);
        sorted_inserts_keys_buffer.free(stream);
        sorted_inserts_offsets_buffer.free(stream);
        new_sorted_keys_buffer.free(stream);
        new_sorted_offsets_buffer.free(stream);

        rebuild_shared_buffer(stream);
    }

    void remove(const key_type* update_list, size_t size, cudaStream_t stream) {
        cuda_buffer_async<uint8_t> temp_buffer;
        cuda_buffer_async<key_type> new_sorted_keys_buffer;
        cuda_buffer_async<smallsize> new_sorted_offsets_buffer;
        cuda_buffer_async<key_type> sorted_deletes_keys_buffer;
        cuda_buffer_async<smallsize> sorted_deletes_offsets_buffer;

        // todo alloc temp for sort and filter

        cub::DeviceRadixSort::SortKeys(
                temp_buffer.raw_ptr, sort_temp_storage_bytes,
                KEYS_IN, KEYS_OUT, COUNT,
                0, sizeof(key_type) * 8, stream);

        cub::DeviceSelect::If(
                temp_buffer.ptr(), filter_temp_storage_bytes,
                d_in, d_out, nullptr, num_items,
                [SORTED_DELETES, size] DEVICEQUALIFIER (const key_type &k) -> bool {
                    return XXXXXX;
                });

        std::swap(new_sorted_keys_buffer, sorted_keys_buffer);
        std::swap(new_sorted_offsets_buffer, sorted_offsets_buffer);

        temp_buffer.free(stream);
        sorted_deletes_keys_buffer.free(stream);
        sorted_deletes_offsets_buffer.free(stream);
        new_sorted_keys_buffer.free(stream);
        new_sorted_offsets_buffer.free(stream);

        rebuild_shared_buffer(stream);
    }
};
*/

#endif
