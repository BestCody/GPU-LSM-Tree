// =============================================================================
// File: impl_opt_sorted_array.cuh
// Author: Justus Henneberg
// Description: Implements impl_opt_sorted_array     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef IMPL_OPT_SORTED_ARRAY_CUH
#define IMPL_OPT_SORTED_ARRAY_CUH

#include "device_binary_search.cuh"
#include "definitions_coarse_granular.cuh"
#include "definitions_opt.cuh"
#include "memory_layout.cuh"

#include <cmath>

#include <cub/cub.cuh>


namespace mem = memory_layout;

// for nvtx
struct nvtx_opt_sorted_array_domain{ static constexpr char const* name{"opt_sorted_array"}; };


template <typename key_type, typename value_type, typename size_type, bool use_row_layout>
GLOBALQUALIFIER
void extract_strided_kernel(
        mem::store_type<key_type, value_type, size_type, use_row_layout> sorted_entries,
        smallsize offset,
        smallsize stride_log,
        key_type* extracted_keys,
        size_t extract_count
) {
    const smallsize tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= extract_count) return;

    extracted_keys[tid] = sorted_entries.extract_key(offset + (tid << stride_log));
}


template <typename key_type, typename value_type, typename size_type, bool use_row_layout>
void extract_strided(
        mem::store_type<key_type, value_type, size_type, use_row_layout> sorted_entries,
        size_t offset,
        size_t stride_log,
        key_type* extracted_keys,
        size_t extract_count,
        cudaStream_t stream
) {
    extract_strided_kernel<key_type, value_type, size_type, use_row_layout><<<SDIV(extract_count, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
            sorted_entries, offset, stride_log, extracted_keys, extract_count);
}


template <typename key_type, typename size_type, bool branchless>
DEVICEQUALIFIER
size_type mixed_granularity_reverse_device_binary_search(
        const uint8_t* base_entries,
        size_type base_entries_count,
        smallsize base_entries_stride,
        key_type lookup,
        const key_type* strided_entries,
        const key_type* partial_level_entries,
        smallsize cached_entries_count, // must not be zero!
        smallsize partial_level_cached_entries_count,
        smallsize stride_log
) {
    size_type coarse_offset = opt_reverse_device_binary_search<key_type, size_type, branchless>(
            (const uint8_t*) strided_entries, cached_entries_count, sizeof(key_type), lookup);

    size_type reverse_offset = cached_entries_count - 1 - coarse_offset;
    size_type continue_offset = base_entries_count - 1 - (reverse_offset << stride_log);
    size_type continue_skip = size_type(1) << (stride_log - 1);

    if (reverse_offset < partial_level_cached_entries_count) {
        if (partial_level_entries[partial_level_cached_entries_count - 1 - reverse_offset] >= lookup) {
            continue_offset -= continue_skip;
        }
        continue_skip >>= 1u;
    }

    size_type final_offset = opt_reverse_device_binary_search<key_type, size_type, branchless>(
            base_entries, base_entries_stride, lookup, continue_offset, continue_skip);

    return final_offset;
}


template <
        typename key_type,
        typename value_type,
        typename size_type,
        uint8_t opts,
        smallsize cg_size_log,
        smallsize cg_activation_factor,
        smallsize threads_per_block,
        smallsize registers_per_thread,
        bool use_row_layout,
        bool branchless,
        bool range_query>
GLOBALQUALIFIER
void optimized_binsearch_lookup_kernel(
        mem::store_type<key_type, value_type, size_type, use_row_layout> sorted_entries,
        smallsize bytes_reserved_for_shuffle,
        const uint8_t* shared_memory_template,
        smallsize shared_bytes,
        smallsize stride_log,
        smallsize cached_entries_count,
        smallsize partial_level_cached_entries_count,
        const key_type* lookups,
        const key_type* upper_limits,
        value_type* result,
        smallsize size
) {
    extern __shared__ uint8_t shared_memory[];

    using algs = opt::algorithms<key_type, threads_per_block, registers_per_thread, range_query>;
    using upper_type = typename algs::upper;

    constexpr smallsize cg_size = smallsize(1) << cg_size_log;
    constexpr smallsize single_thread_max_range_size = cg_size * cg_activation_factor;

    const auto strided_entries = reinterpret_cast<key_type*>(shared_memory + bytes_reserved_for_shuffle);
    const auto partial_level_entries = reinterpret_cast<key_type*>(shared_memory + bytes_reserved_for_shuffle + sizeof(key_type) * cached_entries_count);

    const smallsize bid = blockIdx.x;
    const smallsize local_tid = threadIdx.x;
    const smallsize block_size = threads_per_block;
    const smallsize grid_size = (smallsize) threads_per_block * gridDim.x;

    if constexpr (opt::f::cache_upper_levels(opts)) {
        for (smallsize i = local_tid; i < shared_bytes; i += block_size) {
            shared_memory[i + bytes_reserved_for_shuffle] = shared_memory_template[i];
        }
        __syncthreads();
    }

    const smallsize elements_per_block = block_size * registers_per_thread;
    const smallsize grid_stride = grid_size * registers_per_thread;

    key_type local_lookups[registers_per_thread];
    uint32_t local_ranks[registers_per_thread];
    upper_type local_upper_limits[registers_per_thread];
    value_type local_results[registers_per_thread];

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

        if constexpr (opt::f::sort_lookups_block(opts)) {
            algs::sort(shared_memory).to_striped(local_lookups, local_ranks, local_upper_limits);
            __syncthreads();
        }

        #pragma unroll
        for (smallsize i = 0; i < registers_per_thread; ++i) {
            const auto block_rank = i * block_size + local_tid;
            if (block_rank >= block_count) continue;

            size_type pos;
            if (cached_entries_count > 0) {
                pos = mixed_granularity_reverse_device_binary_search<key_type, size_type, branchless>(
                        sorted_entries.key_pointer(),
                        sorted_entries.size(),
                        sorted_entries.key_stride(),
                        local_lookups[i],
                        strided_entries,
                        partial_level_entries,
                        cached_entries_count,
                        partial_level_cached_entries_count,
                        stride_log);
            } else {
                pos = opt_reverse_device_binary_search<key_type, size_type, branchless>(
                        sorted_entries.key_pointer(),
                        sorted_entries.size(),
                        sorted_entries.key_stride(),
                        local_lookups[i]);
            }

            value_type local_result;
            if constexpr (range_query) {
                local_result = 0;
                size_type it = pos;
                size_type search_limit = cg_size_log == 0 ? std::numeric_limits<size_type>::max() : pos + single_thread_max_range_size;
                size_type range_limit = min(search_limit, sorted_entries.size());
                // short range queries can be processed without cooperative groups
                for (; it < range_limit; ++it) {
                    if (sorted_entries.extract_key(it) > local_upper_limits[i])
                        break;
                    local_result += sorted_entries.extract_value(it);
                }
                // for long ranges, switch to cooperative groups
                if constexpr (cg_size_log != 0) {
                    auto tile = cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());

                    bool to_find = block_rank < block_count && it == search_limit;
                    auto work_queue = tile.ballot(to_find);
                    while (work_queue) {
                        auto cur_rank = __ffs(work_queue) - 1;
                        auto cur_pos = tile.shfl(search_limit, cur_rank);
                        auto cur_lower = tile.shfl(local_lookups[i], cur_rank);
                        auto cur_upper = tile.shfl(local_upper_limits[i], cur_rank);

                        auto cur_result = mem::rq(sorted_entries).cooperative_range_query(tile, cur_lower, cur_upper, cur_pos);

                        if (cur_rank == tile.thread_rank()) {
                            local_result += cur_result;
                            to_find = false;
                        }
                        work_queue = tile.ballot(to_find);
                    }
                }
            } else {
                local_result = sorted_entries.extract_key(pos) == local_lookups[i] ? sorted_entries.extract_value(pos) : not_found; //<value_type>;
            }

            if constexpr (opt::f::sort_unsort_lookups_block(opts)) {
                // delayed write
                local_results[i] = local_result;
            } else {
                // immediate write
                result[lookup_offset + local_ranks[i]] = local_result;
            }
        }

        if constexpr (opt::f::sort_unsort_lookups_block(opts)) {
            algs::result_shuffle(shared_memory).to_striped(local_results, local_ranks);
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


template <
        typename key_type_,
        typename value_type_,
        typename size_type_,
        uint8_t opts,
        bool static_scheduling,
        bool use_row_layout,
        bool branchless,
        uint16_t shared_memory_occupancy,
        uint16_t cg_size_log,
        uint16_t cg_activation_factor,
        uint16_t threads_per_block,
        uint16_t registers_per_thread>
class opt_sorted_array {
    static_assert(cg_size_log <= 5, "cg_size_log must be between 0 and 5");
    static_assert(shared_memory_occupancy <= 200, "shared_memory_occupancy must be between 0 and 200");

public:
    using key_type = key_type_;
    using value_type = value_type_;
    using size_type = size_type_;

    // shortcut to try different activation factors, used for benchmarking
    template <smallsize new_cg_activation_factor>
    using with_activation_factor = opt_sorted_array<key_type, value_type, size_type, opts, static_scheduling, use_row_layout, branchless, shared_memory_occupancy, cg_size_log, new_cg_activation_factor, threads_per_block, registers_per_thread>;

private:
    static constexpr smallsize cg_size = 1u << cg_size_log;

    // override kernel setup for range queries
    // these parameters might not be ideal, this is just an educated guess
    static constexpr smallsize rq_threads_per_block = 512;
    static constexpr smallsize rq_registers_per_thread = 1;

    cuda_buffer<key_type> sorted_keys_buffer;
    cuda_buffer<value_type> sorted_offsets_buffer;
    cuda_buffer<uint8_t> sorted_key_offset_pairs_buffer;
    size_t base_entries_count = 0;

    cuda_buffer<uint8_t> shared_buffer;
    size_t stride_log = 0;
    size_t cached_entries_count = 0;
    size_t partial_level_cached_entries_count = 0;

    size_t shared_bytes_to_load = 0;
    size_t shared_bytes_for_shuffle = 0;

    size_t number_of_sms = 0;
    size_t max_shared_memory_bytes = 0;

    size_t shmem() {
        return shared_bytes_to_load + shared_bytes_for_shuffle;
    }

    mem::store_type<key_type, value_type, size_type, use_row_layout> make_store() {
        if constexpr (use_row_layout) {
            return mem::row_store<key_type, value_type, size_type>(sorted_key_offset_pairs_buffer.ptr(), base_entries_count);
        } else {
            return mem::col_store<key_type, value_type, size_type>(sorted_keys_buffer.ptr(), sorted_offsets_buffer.ptr(), base_entries_count);
        }
    }

    void extract_keys_for_shared_buffer() {
        if constexpr (!opt::f::cache_upper_levels(opts)) return;

        constexpr bool partial = opt::f::cache_upper_levels_partial(opts);
        constexpr size_t key_size = sizeof(key_type);

        size_t remaining_bytes = max_shared_memory_bytes - shared_bytes_for_shuffle;

        // smallest possible power-of-2 stride so that strided entries fit into shared memory
        // let base_entries_count be N, key_size be T, available mem be M, then
        // for stride s, we need [ceil(N/(2^s))] entries which is [T * ceil(N/(2^s))] bytes or [T(N/(2^s)+1)] bytes for simplicity
        // therefore, [M >= TN/(2^s) + T], or, equivalently [s >= log2(TN/(M-T))], we choose floor(log) plus one
        if (key_size * base_entries_count > remaining_bytes) {
            stride_log = ilog2_cpu(key_size * base_entries_count / (remaining_bytes - key_size)) + 1;
        } else {
            // we have enough space for all entries
            stride_log = 0;
        }
        size_t stride = size_t(1) << stride_log;

        cached_entries_count = SDIV(base_entries_count, stride);
        size_t strided_entries_bytes = footprint<key_type>(cached_entries_count);

        partial_level_cached_entries_count = !partial ? 0 : (remaining_bytes - strided_entries_bytes) / key_size;
        size_t partial_level_entries_bytes = footprint<key_type>(partial_level_cached_entries_count);

        shared_bytes_to_load = strided_entries_bytes + partial_level_entries_bytes;

        // __X___X___X___X___X
        // stride=4 len=19 -> ofst = (len-1) % stride
        extract_strided<key_type, value_type, size_type, use_row_layout>(
            make_store(),
            (base_entries_count - 1) % stride,
            stride_log,
            reinterpret_cast<key_type*>(shared_buffer.ptr()),
            cached_entries_count,
            0);

        // __X___X___X_Y_X_Y_X
        // stride=4 len=19 partial=2 -> ofst = (len-1)+stride/2-stride*partial
        if (partial && stride > 1 && partial_level_cached_entries_count > 0) {
            extract_strided<key_type, value_type, size_type, use_row_layout>(
                make_store(),
                (base_entries_count - 1) + (stride >> 1u) - partial_level_cached_entries_count * stride,
                stride_log,
                reinterpret_cast<key_type*>(shared_buffer.ptr() + footprint<key_type>(cached_entries_count)),
                partial_level_cached_entries_count,
                0);
        }
    }

public:
    static constexpr operation_support can_lookup = operation_support::async;
    static constexpr operation_support can_multi_lookup = operation_support::async;
    static constexpr operation_support can_range_lookup = operation_support::async;
    static constexpr operation_support can_insert = operation_support::none;
    static constexpr operation_support can_delete = operation_support::none;

    static std::string short_description() {
        std::string desc = "opt_sorted_array";
        return desc;
    }

    static parameters_type parameters() {
        std::string opts_summary;
        opts_summary += opt::f::cache_upper_levels_partial(opts) ? "cp" : opt::f::cache_upper_levels(opts) ? "c" : "";
        opts_summary += opt::f::sort_unsort_lookups_block(opts) ? "su" : opt::f::sort_lookups_block(opts) ? "s" : "";
        opts_summary += opts_summary.empty() ? "none" : "";
        return {
                {"opts", opts_summary},
                {"schedule", static_scheduling ? "static" : "dynamic"},
                {"layout", use_row_layout ? "row" : "column"},
                {"branch_style", branchless ? "add" : "if"},
                {"shared_memory_occupancy", std::to_string(shared_memory_occupancy)},
                {"threads_per_block", std::to_string(threads_per_block)},
                {"registers_per_thread", std::to_string(registers_per_thread)},
                {"rq_threads_per_block", std::to_string(rq_threads_per_block)},
                {"rq_registers_per_thread", std::to_string(rq_registers_per_thread)},
                {"cg_size", std::to_string(cg_size)},
                {"cg_af", std::to_string(cg_activation_factor)},
                {"index_bits", std::to_string(sizeof(size_type) * 8)}
        };
    }

    static size_t estimate_build_bytes(size_t size) {
        size_t base_bytes = (sizeof(value_type) + sizeof(key_type)) * size;
        size_t sort_aux_bytes = sizeof(value_type) * size + find_pair_sort_buffer_size<key_type, value_type>(size);
        size_t shared_bytes = std::min<size_t>(base_bytes, get_max_optin_shmem_per_block_bytes());
        size_t layout_transform_bytes = use_row_layout ? base_bytes : 0;
        return base_bytes + std::max<size_t>(sort_aux_bytes, layout_transform_bytes) + shared_bytes;
    }

    size_t gpu_resident_bytes() {
        // note that some of these buffers are never allocated
        return sorted_keys_buffer.size_in_bytes() +
            sorted_offsets_buffer.size_in_bytes() +
            sorted_key_offset_pairs_buffer.size_in_bytes() +
            shared_buffer.size_in_bytes();
    }

    void build(const key_type* keys, size_t size, size_t max_size, size_t available_memory_bytes, double* build_time_ms, size_t* build_bytes) {
        if (estimate_build_bytes(size) > available_memory_bytes) throw std::runtime_error("insufficient memory");

        constexpr size_t row_stride = sizeof(key_type) + sizeof(value_type);

        number_of_sms = get_number_of_sms();
        max_shared_memory_bytes = interpolate_shmem_bytes(shared_memory_occupancy);
        base_entries_count = size;

        size_t temp_storage_bytes = 0;
        {
            cuda_buffer<uint8_t> temp_buffer;
            cuda_buffer<value_type> offsets_buffer;

            offsets_buffer.alloc(size); C2EX
            init_offsets(offsets_buffer.ptr(), size, build_time_ms);

            sorted_keys_buffer.alloc(size); C2EX
            sorted_offsets_buffer.alloc(size); C2EX

            cudaDeviceSynchronize(); C2EX

            temp_storage_bytes = find_pair_sort_buffer_size<key_type, value_type>(size);
            temp_buffer.alloc(temp_storage_bytes); C2EX
            timed_pair_sort(
                temp_buffer.raw_ptr, temp_storage_bytes,
                keys, sorted_keys_buffer.ptr(), offsets_buffer.ptr(), sorted_offsets_buffer.ptr(), size, build_time_ms);

            cudaDeviceSynchronize(); C2EX
        }

        if constexpr (opt::f::sort_lookups_block(opts)) {
            using algs = opt::algorithms<key_type, threads_per_block, registers_per_thread, false>;
            using rq_algs = opt::algorithms<key_type, rq_threads_per_block, rq_registers_per_thread, true>;

            shared_bytes_for_shuffle = std::max(algs::temp_storage_bytes, rq_algs::temp_storage_bytes);
            // add padding so that shared_bytes_for_shuffle is a multiple of 8 bytes
            shared_bytes_for_shuffle = (shared_bytes_for_shuffle + 7) & ~size_t(7);
        }

        if (shared_bytes_for_shuffle > max_shared_memory_bytes) {
            throw std::runtime_error("shared memory for block-local reordering exceeds maximum shared memory size");
        }

        // transform to row layout if necessary
        if constexpr (use_row_layout) {
            sorted_key_offset_pairs_buffer.alloc(row_stride * size); C2EX
            transform_into_row_layout(
                sorted_keys_buffer.ptr(),
                sorted_offsets_buffer.ptr(),
                sorted_key_offset_pairs_buffer.ptr(),
                size,
                build_time_ms);
            sorted_keys_buffer.free();
            sorted_offsets_buffer.free();
        }

        // only allocate shared memory if cache_upper_levels is enabled
        if constexpr (opt::f::cache_upper_levels(opts)) {
            shared_buffer.alloc(max_shared_memory_bytes); C2EX
        }

        // build shared buffer
        {
            scoped_cuda_timer timer(0, build_time_ms);
            extract_keys_for_shared_buffer();
        }

        // point query
        cudaFuncSetAttribute(
            optimized_binsearch_lookup_kernel<key_type, value_type, size_type, opts, cg_size_log, cg_activation_factor, threads_per_block, registers_per_thread, use_row_layout, branchless, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem()); C2EX
        // range query
        cudaFuncSetAttribute(
            optimized_binsearch_lookup_kernel<key_type, value_type, size_type, opts, cg_size_log, cg_activation_factor, rq_threads_per_block, rq_registers_per_thread, use_row_layout, branchless, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem()); C2EX

        size_t layout_transform_bytes = use_row_layout ? row_stride * size : 0;
        if (build_bytes) *build_bytes += row_stride * size + std::max<size_t>({
            temp_storage_bytes + sizeof(value_type) * size,
            layout_transform_bytes,
            shared_buffer.size_in_bytes()});

        cudaDeviceSynchronize(); C2EX
    }

    void lookup(const key_type* keys, value_type* result, size_t size, cudaStream_t stream) {

        nvtx3::scoped_range_in<nvtx_opt_sorted_array_domain> launch{"launch"};

        size_t number_of_blocks = static_scheduling ? number_of_sms : SDIV(size, threads_per_block * registers_per_thread);
        optimized_binsearch_lookup_kernel
                <key_type, value_type, size_type, opts, cg_size_log, cg_activation_factor, threads_per_block, registers_per_thread, use_row_layout, branchless, false>
                <<<number_of_blocks, threads_per_block, shmem(), stream>>>(
            make_store(),
            shared_bytes_for_shuffle, shared_buffer.ptr(), shared_bytes_to_load, stride_log, cached_entries_count, partial_level_cached_entries_count,
            keys, nullptr, result, size);
    }

    void range_lookup_sum(const key_type* lower, const key_type* upper, value_type* result, size_t size, cudaStream_t stream) {

        nvtx3::scoped_range_in<nvtx_opt_sorted_array_domain> launch{"launch"};

        size_t number_of_blocks = static_scheduling ? number_of_sms : SDIV(size, rq_threads_per_block * rq_registers_per_thread);
        optimized_binsearch_lookup_kernel
                <key_type, value_type, size_type, opts, cg_size_log, cg_activation_factor, rq_threads_per_block, rq_registers_per_thread, use_row_layout, branchless, true>
                <<<number_of_blocks, rq_threads_per_block, shmem(), stream>>>(
            make_store(),
            shared_bytes_for_shuffle, shared_buffer.ptr(), shared_bytes_to_load, stride_log, cached_entries_count, partial_level_cached_entries_count,
            lower, upper, result, size);
    }

    void multi_lookup_sum(const key_type* keys, value_type* result, size_t size, cudaStream_t stream) {
        range_lookup_sum(keys, keys, result, size, stream);
    }

    void destroy() {
        sorted_keys_buffer.free();
        sorted_offsets_buffer.free();
        sorted_key_offset_pairs_buffer.free();
        shared_buffer.free();
    }
};


template <typename key_type, typename value_type, typename size_type, bool use_row_layout = false, uint16_t cg_size_log = 5, uint16_t shared_memory_occupancy = 200>
using opt_sorted_array_4090 = opt_sorted_array<key_type, value_type, size_type, opt::f::sulb, false, use_row_layout, true, shared_memory_occupancy, cg_size_log, 1, 128, 16>;

template <typename key_type, typename value_type, typename size_type, bool use_row_layout = false, uint16_t cg_size_log = 5, uint16_t shared_memory_occupancy = 200>
using opt_sorted_array_5090 = opt_sorted_array<key_type, value_type, size_type, opt::f::sulb, false, use_row_layout, true, shared_memory_occupancy, cg_size_log, 1, 128, 16>;

#endif
