// =============================================================================
// File: impl_opt_static_binary_tree.cuh
// Author: Justus Henneberg
// Description: Implements impl_opt_static_binary_tree     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef IMPL_OPT_STATIC_BINARY_TREE_CUH
#define IMPL_OPT_STATIC_BINARY_TREE_CUH

#include "definitions_opt.cuh"
#include "memory_layout.cuh"
#include "utilities.cuh"

#include <cmath>

#include <cub/cub.cuh>


namespace mem = memory_layout;

// for nvtx
struct nvtx_opt_static_binary_tree_domain{ static constexpr char const* name{"opt_static_binary_tree"}; };


template <uint16_t max_level_count>
struct binary_tree_metadata {
    smallsize level_count;
    smallsize total_level_entries;
    smallsize level_zero_stride_log;
    smallsize partition_size_log;
    // level 0 is the topmost level
    std::array<smallsize, max_level_count> entries_on_level;
};


template <typename key_type, bool use_row_layout, uint16_t max_level_count>
GLOBALQUALIFIER
void build_static_binary_tree_kernel(
        mem::store_type<key_type, use_row_layout> sorted_entries,
        binary_tree_metadata<max_level_count> metadata,
        key_type* tree
) {
    const smallsize tid = blockDim.x * blockIdx.x + threadIdx.x;
    smallsize local_index = tid;

    if (tid >= metadata.total_level_entries) return;

    // determine which level we are on
    smallsize level = 0;
    smallsize level_stride = smallsize(1) << metadata.level_zero_stride_log;
    for (; level < metadata.level_count; ++level) {
        smallsize entries_on_level = metadata.entries_on_level[level];
        if (local_index < entries_on_level) break;
        local_index -= entries_on_level;
        level_stride >>= 1;
    }

    // each thread loads and writes one item
    const auto first_item_offset = (level_stride >> 1u) - 1;
    const auto load_offset = first_item_offset + local_index * level_stride;
    // usually, we would write load_offset < size to check for valid bounds
    // but using the very last entry for an inner node would imply there is a valid child node to the right,
    // which can never be the case. so we explicitly exclude the last entry
    const auto exists = load_offset < sorted_entries.size() - 1;
    const auto write = exists ? sorted_entries.extract_key(load_offset) : std::numeric_limits<key_type>::max();
    tree[tid] = write;
}


template <typename key_type, bool caching_enabled, bool branchless, uint16_t max_level_count>
DEVICEQUALIFIER INLINEQUALIFIER
smallsize static_binary_tree_search(
        key_type lookup,
        const key_type* tree,
        binary_tree_metadata<max_level_count> metadata,
        const key_type* shmem_tree,
        smallsize cached_entries_count
) {
    smallsize global_entries_offset = 0;
    smallsize local_entries_offset = 0;
    for (smallsize level = 0; level < metadata.level_count; ++level) {
        smallsize offset = global_entries_offset + local_entries_offset;
        key_type entry;
        if (caching_enabled && offset < cached_entries_count) {
            entry = shmem_tree[offset];
        } else {
            entry = tree[offset];
        }
        const bool valid = entry != std::numeric_limits<key_type>::max();
        const bool gt = valid && lookup > entry;
        if constexpr (branchless) {
            local_entries_offset = (local_entries_offset << 1u) + gt;
        } else {
            if (gt) {
                local_entries_offset = (local_entries_offset << 1u) + 1;
            } else {
                local_entries_offset = (local_entries_offset << 1u);
            }
        }
        global_entries_offset += metadata.entries_on_level[level];
    }
    return local_entries_offset;
}


template <
        typename key_type,
        uint8_t opts,
        bool use_row_layout,
        bool branchless,
        uint16_t max_level_count,
        uint16_t cg_size_log,
        uint16_t cg_activation_factor,
        uint16_t threads_per_block,
        uint16_t registers_per_thread,
        bool range_query>
GLOBALQUALIFIER
void static_binary_tree_lookup_kernel(
        mem::store_type<key_type, use_row_layout> sorted_entries,
        smallsize cached_entries_count,
        smallsize bytes_reserved_for_shuffle,
        const key_type* tree,
        binary_tree_metadata<max_level_count> metadata,
        const key_type* lookups,
        const key_type* upper_limits,
        smallsize* results,
        smallsize size
) {
    static_assert(threads_per_block <= 1024, "cannot have more than 1024 threads per block");

    extern __shared__ uint8_t shared_memory[];

    using algs = opt::algorithms<key_type, threads_per_block, registers_per_thread, range_query>;
    using upper_type = typename algs::upper;

    constexpr smallsize cg_size = smallsize(1) << cg_size_log;
    constexpr smallsize single_thread_max_range_size = (smallsize)cg_size * cg_activation_factor;

    const auto shmem_tree = reinterpret_cast<key_type*>(shared_memory + bytes_reserved_for_shuffle);

    const smallsize bid = blockIdx.x;
    const smallsize local_tid = threadIdx.x;
    const smallsize block_size = threads_per_block;
    const smallsize grid_size = threads_per_block * gridDim.x;

    if (opt::f::cache_upper_levels(opts)) {
        for (smallsize i = local_tid; i < cached_entries_count; i += block_size) {
            shmem_tree[i] = tree[i];
        }
        __syncthreads();
    }

    key_type local_lookups[registers_per_thread];
    upper_type local_upper_limits[registers_per_thread];
    uint32_t local_ranks[registers_per_thread];
    smallsize local_results[registers_per_thread];

    const smallsize elements_per_block = block_size * registers_per_thread;
    const smallsize grid_stride = grid_size * registers_per_thread;

    for (smallsize lookup_offset = bid * elements_per_block; lookup_offset < size; lookup_offset += grid_stride) {
        const auto this_block_count = std::min(elements_per_block, size - lookup_offset);

        #pragma unroll
        for (smallsize reg = 0; reg < registers_per_thread; ++reg) {
            const auto block_rank = reg * block_size + local_tid;
            local_lookups[reg] = block_rank < this_block_count ? lookups[lookup_offset + block_rank] : std::numeric_limits<key_type>::max();
            if constexpr (range_query) {
                local_upper_limits[reg] = block_rank < this_block_count ? upper_limits[lookup_offset + block_rank] : std::numeric_limits<key_type>::max();
            }
            local_ranks[reg] = block_rank;
        }

        if constexpr (opt::f::sort_lookups_block(opts)) {
            algs::sort(shared_memory).to_striped(local_lookups, local_ranks, local_upper_limits);
            __syncthreads();
        }

        #pragma unroll
        for (smallsize reg = 0; reg < registers_per_thread; ++reg) {
            const auto block_rank = reg * block_size + local_tid;

            // todo check block_rank >= this_block_count

            smallsize local_result;

            auto partition = static_binary_tree_search<key_type, opt::f::cache_upper_levels(opts), branchless, max_level_count>(
                    local_lookups[reg], tree, metadata, shmem_tree, cached_entries_count);
            auto end_of_partition = (partition + 1) << metadata.partition_size_log;
            auto pos = min(end_of_partition, sorted_entries.size()) - 1;
            auto stride = smallsize(1) << (metadata.partition_size_log - 1);

            while (stride > 0) {
                if (pos < stride) continue;
                if constexpr (branchless) {
                    pos -= stride * bool(sorted_entries.extract_key(pos - stride) >= local_lookups[reg]);
                } else {
                    if (sorted_entries.extract_key(pos - stride) >= local_lookups[reg]) {
                        pos -= stride;
                    }
                }
                stride >>= 1u;
            }

            if constexpr (range_query) {
                local_result = 0;
                smallsize it = pos;
                smallsize search_limit = cg_size_log == 0 ? std::numeric_limits<smallsize>::max() : pos + single_thread_max_range_size;
                smallsize range_limit = min(search_limit, sorted_entries.size());
                // short range queries can be processed without cooperative groups
                for (; it < range_limit; ++it) {
                    if (sorted_entries.extract_key(it) > local_upper_limits[reg])
                        break;
                    local_result += sorted_entries.extract_offset(it);
                }
                // for long ranges, switch to cooperative groups
                if constexpr (cg_size_log != 0) {
                    auto tile = cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());

                    bool to_find = block_rank < this_block_count && it == search_limit;
                    auto work_queue = tile.ballot(to_find);
                    while (work_queue) {
                        auto cur_rank = __ffs(work_queue) - 1;
                        auto cur_pos = tile.shfl(search_limit, cur_rank);
                        auto cur_lower = tile.shfl(local_lookups[reg], cur_rank);
                        auto cur_upper = tile.shfl(local_upper_limits[reg], cur_rank);

                        auto cur_result = mem::cooperative_range_query(tile, cur_lower, cur_upper, cur_pos, sorted_entries);

                        if (cur_rank == tile.thread_rank()) {
                            local_result += cur_result;
                            to_find = false;
                        }
                        work_queue = tile.ballot(to_find);
                    }
                }
            } else {
                local_result = sorted_entries.extract_key(pos) == local_lookups[reg] ? sorted_entries.extract_offset(pos) : not_found;
            }

            if constexpr (opt::f::sort_unsort_lookups_block(opts)) {
                // delayed write
                local_results[reg] = local_result;
            } else {
                // immediate write
                results[lookup_offset + local_ranks[reg]] = local_result;
            }
        }

        if constexpr (opt::f::sort_unsort_lookups_block(opts)) {
            algs::result_shuffle(shared_memory).to_striped(local_results, local_ranks);
            __syncthreads();

            #pragma unroll
            for (smallsize reg = 0; reg < registers_per_thread; ++reg) {
                const auto block_rank = reg * block_size + local_tid;
                if (block_rank >= this_block_count) continue;
                results[lookup_offset + block_rank] = local_results[reg];
            }
        }
    }
}


template <
        typename key_type_,
        uint8_t opts,
        bool static_scheduling,
        bool use_row_layout,
        bool branchless,
        uint16_t max_level_count,
        uint16_t min_partition_size_log,
        uint16_t cg_size_log,
        uint16_t cg_activation_factor,
        uint16_t threads_per_block,
        uint16_t registers_per_thread>
class opt_static_binary_tree {
    static_assert(min_partition_size_log > 0, "min_partition_size_log must be at least 1");

public:
    using key_type = key_type_;

    // shortcut to try different activation factors, used for benchmarking
    template <uint16_t new_cg_activation_factor>
    using with_activation_factor = opt_static_binary_tree<key_type, opts, static_scheduling, use_row_layout, branchless, max_level_count, min_partition_size_log, cg_size_log, new_cg_activation_factor, threads_per_block, registers_per_thread>;

private:
    static constexpr size_t cg_size = 1u << cg_size_log;

    // override kernel setup for range queries
    // these parameters might not be ideal, this is just an educated guess
    static constexpr uint16_t rq_threads_per_block = 512;
    static constexpr uint16_t rq_registers_per_thread = 1;

    cuda_buffer<key_type> sorted_keys_buffer;
    cuda_buffer<smallsize> sorted_offsets_buffer;
    cuda_buffer<uint8_t> sorted_key_offset_pairs_buffer;
    size_t base_entries_count = 0;

    // buffer stores metadata and all tree levels in sequence
    cuda_buffer<key_type> tree_buffer;
    size_t shared_entries_count = 0;

    size_t shared_bytes_to_load = 0;
    size_t shared_bytes_for_shuffle = 0;

    uint32_t number_of_sms = 0;
    size_t max_shared_memory_bytes = 0;

    binary_tree_metadata<max_level_count> metadata;

    size_t shmem() {
        return shared_bytes_to_load + shared_bytes_for_shuffle;
    }

    mem::store_type<key_type, use_row_layout> make_store() {
        if constexpr (use_row_layout) {
            return mem::row_store<key_type>(sorted_key_offset_pairs_buffer.ptr(), base_entries_count);
        } else {
            return mem::col_store<key_type>(sorted_keys_buffer.ptr(), sorted_offsets_buffer.ptr(), base_entries_count);
        }
    }

public:
    static constexpr operation_support can_lookup = operation_support::async;
    static constexpr operation_support can_multi_lookup = operation_support::async;
    static constexpr operation_support can_range_lookup = operation_support::async;
    static constexpr operation_support can_update = operation_support::none;

    static std::string short_description() {
        std::string desc = "opt_static_binary_tree";
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
                {"threads_per_block", std::to_string(threads_per_block)},
                {"registers_per_thread", std::to_string(registers_per_thread)},
                {"rq_threads_per_block", std::to_string(rq_threads_per_block)},
                {"rq_registers_per_thread", std::to_string(rq_registers_per_thread)},
                {"cg_size", std::to_string(cg_size)},
                {"cg_af", std::to_string(cg_activation_factor)},
                {"max_level_count", std::to_string(max_level_count)},
                {"min_partition_size_log", std::to_string(min_partition_size_log)}
        };
    }

    static size_t estimate_build_bytes(size_t size) {
        size_t base_bytes = (sizeof(smallsize) + sizeof(key_type)) * size;
        size_t sort_aux_bytes = sizeof(smallsize) * size + find_pair_sort_buffer_size<key_type, smallsize>(size);
        size_t tree_bytes = base_bytes >> 1u;
        size_t layout_transform_bytes = use_row_layout ? base_bytes : 0;
        return std::max(base_bytes + sort_aux_bytes, base_bytes + tree_bytes + layout_transform_bytes);
    }

    size_t gpu_resident_bytes() {
        // note that some of these buffers are never allocated
        return sorted_keys_buffer.size_in_bytes() +
            sorted_offsets_buffer.size_in_bytes() +
            sorted_key_offset_pairs_buffer.size_in_bytes() +
            tree_buffer.size_in_bytes();
    }

    void build(const key_type* keys, size_t size, size_t max_size, size_t available_memory_bytes, double* build_time_ms, size_t* build_bytes) {
        if (estimate_build_bytes(size) > available_memory_bytes) throw std::runtime_error("insufficient memory");

        constexpr size_t key_size = sizeof(key_type);
        constexpr size_t row_stride = sizeof(key_type) + sizeof(smallsize);

        number_of_sms = get_number_of_sms();
        max_shared_memory_bytes = get_max_optin_shmem_per_block_bytes();
        base_entries_count = size;

        // sort keys
        size_t temp_storage_bytes = 0;
        {
            cuda_buffer<uint8_t> temp_buffer;
            cuda_buffer<smallsize> offsets_buffer;

            offsets_buffer.alloc(size); C2EX
            init_offsets(offsets_buffer.ptr(), size, build_time_ms);

            sorted_keys_buffer.alloc(size); C2EX
            sorted_offsets_buffer.alloc(size); C2EX

            cudaDeviceSynchronize(); C2EX

            temp_storage_bytes = find_pair_sort_buffer_size<key_type, smallsize>(size);
            temp_buffer.alloc(temp_storage_bytes); C2EX
            timed_pair_sort(
                temp_buffer.raw_ptr, temp_storage_bytes,
                keys, sorted_keys_buffer.ptr(), offsets_buffer.ptr(), sorted_offsets_buffer.ptr(), size, build_time_ms);

            cudaDeviceSynchronize(); C2EX
        }

        // figure out tree structure
        {
            size_t total_level_count = 0;
            size_t entries_on_current_level = SDIV(size, size_t(1) << min_partition_size_log);
            while (entries_on_current_level > 1) {
                // one inner node covers 2 nodes on lower level
                entries_on_current_level = SDIV(entries_on_current_level, 2);
                total_level_count++;
            }
            size_t level_zero_stride_log = min_partition_size_log + total_level_count;
            // clip level count
            size_t clipped_level_count = std::min(total_level_count, (size_t) max_level_count);
            size_t partition_size_log = level_zero_stride_log - clipped_level_count;
            // we now have clipped_level_count <= max_level_count and partition_size_log >= min_partition_size_log
            // we could end up with zero levels if the input size is too small

            metadata.level_count = clipped_level_count;
            metadata.level_zero_stride_log = level_zero_stride_log;
            metadata.partition_size_log = partition_size_log;
            metadata.total_level_entries = 0;

            entries_on_current_level = SDIV(size, size_t(1) << min_partition_size_log);
            for (size_t i = 0; i < total_level_count; i++) {
                entries_on_current_level = SDIV(entries_on_current_level, 2);
                size_t level = total_level_count - 1 - i;
                // we have reached the upper levels
                if (level < clipped_level_count) {
                    metadata.entries_on_level[level] = entries_on_current_level;
                    metadata.total_level_entries += entries_on_current_level;
                }
            }
        }

        // reserve shmem for sort/shuffle
        if constexpr (opt::f::sort_lookups_block(opts)) {
            using algs = opt::algorithms<key_type, threads_per_block, registers_per_thread, false>;
            using rq_algs = opt::algorithms<key_type, rq_threads_per_block, rq_registers_per_thread, true>;

            shared_bytes_for_shuffle = std::max(algs::temp_storage_bytes, rq_algs::temp_storage_bytes);
            // add padding so that shared_bytes_for_shuffle is a multiple of 8 bytes
            shared_bytes_for_shuffle = (shared_bytes_for_shuffle + 7) & ~size_t(7);
        }

        tree_buffer.alloc(metadata.total_level_entries); C2EX

        // only extend shared memory if cache_upper_levels is enabled
        if constexpr (opt::f::cache_upper_levels(opts)) {
            size_t remaining_bytes = max_shared_memory_bytes - shared_bytes_for_shuffle;
            size_t max_entry_count = remaining_bytes / key_size;

            shared_entries_count = 0;
            if (metadata.total_level_entries <= max_entry_count) {
                // if we can, cache everything
                shared_entries_count = metadata.total_level_entries;
            } else if (opt::f::cache_upper_levels_partial(opts)) {
                // otherwise, cache as much as we can fit
                shared_entries_count = max_entry_count;
            } else {
                // or only add full levels until we run out of space
                for (size_t i = 0; i < metadata.level_count; ++i) {
                    if (metadata.entries_on_level[i] > max_entry_count) break;
                    shared_entries_count += metadata.entries_on_level[i];
                    max_entry_count -= metadata.entries_on_level[i];
                }
            }
            shared_bytes_to_load = shared_entries_count * key_size;

            // point query
            cudaFuncSetAttribute(
                    static_binary_tree_lookup_kernel<key_type, opts, use_row_layout, branchless, max_level_count, cg_size_log, cg_activation_factor, threads_per_block, registers_per_thread, false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    shmem()); C2EX
            // range query
            cudaFuncSetAttribute(
                    static_binary_tree_lookup_kernel<key_type, opts, use_row_layout, branchless, max_level_count, cg_size_log, cg_activation_factor, threads_per_block, registers_per_thread, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    shmem()); C2EX
        }

        // build tree
        {
            scoped_cuda_timer timer(0, build_time_ms);
            auto nb = SDIV(metadata.total_level_entries, MAXBLOCKSIZE);
            build_static_binary_tree_kernel<key_type, false, max_level_count><<<nb, MAXBLOCKSIZE>>>
                (mem::col_store(sorted_keys_buffer.ptr(), nullptr, size), metadata, tree_buffer.ptr());
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

        size_t layout_transform_bytes = use_row_layout ? row_stride * size : 0;
        if (build_bytes) *build_bytes += row_stride * size + std::max<size_t>(
                temp_storage_bytes + sizeof(smallsize) * size,
                tree_buffer.size_in_bytes() + layout_transform_bytes);

        cudaDeviceSynchronize(); C2EX
    }

    void lookup(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {

        nvtx3::scoped_range_in<nvtx_opt_static_binary_tree_domain> launch{"launch"};

        size_t number_of_blocks = static_scheduling ? number_of_sms : SDIV(size, threads_per_block * registers_per_thread);
        static_binary_tree_lookup_kernel
                <key_type, opts, use_row_layout, branchless, max_level_count, cg_size_log, cg_activation_factor, threads_per_block, registers_per_thread, false>
                <<<number_of_blocks, threads_per_block, shmem(), stream>>>(
            make_store(),
            shared_entries_count, shared_bytes_for_shuffle,
            tree_buffer.ptr(), metadata,
            keys, nullptr, result, size);
    }

    void range_lookup_sum(const key_type* lower, const key_type* upper, smallsize* result, size_t size, cudaStream_t stream) {

        nvtx3::scoped_range_in<nvtx_opt_static_binary_tree_domain> launch{"launch"};

        size_t number_of_blocks = static_scheduling ? number_of_sms : SDIV(size, rq_threads_per_block * rq_registers_per_thread);
        static_binary_tree_lookup_kernel
                <key_type, opts, use_row_layout, branchless, max_level_count, cg_size_log, cg_activation_factor, rq_threads_per_block, rq_registers_per_thread, true>
                <<<number_of_blocks, rq_threads_per_block, shmem(), stream>>>(
            make_store(),
            shared_entries_count, shared_bytes_for_shuffle,
            tree_buffer.ptr(), metadata,
            lower, upper, result, size);
    }

    void multi_lookup_sum(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        range_lookup_sum(keys, keys, result, size, stream);
    }

    void destroy() {
        sorted_keys_buffer.free();
        sorted_offsets_buffer.free();
        sorted_key_offset_pairs_buffer.free();
        tree_buffer.free();
    }
};


// todo change
//template <typename key_type, bool use_row_layout = false, uint16_t cg_size_log = 5>
//using opt_static_binary_tree_4090 = opt_static_binary_tree<key_type, XXXXXX, false, use_row_layout, true, XXXXXX, XXXXXX, cg_size_log, 1, XXX, XXX>;

#endif
