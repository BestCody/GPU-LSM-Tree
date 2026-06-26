// =============================================================================
// File: benchmarks_lookups.cuh
// Author: Justus Henneberg
// Description: Implements benchmarks_lookups     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef BENCHMARKS_LOOKUPS_CUH
#define BENCHMARKS_LOOKUPS_CUH

#include "impl_hashtable_warpcore.cuh"
#include "impl_hashtable_slab.cuh"
#include "impl_tree_awad.cuh"
#include "utilities.cuh"


// experimental: test effects of warp scheduling and warp-internal reordering
// preliminary results: owens-style warp scheduling is a good idea; no significant benefit from warp-internal reordering
// todo test runtime cost difference: blockradixsort vs blockrank+shuffle


template<class wc_table>
GLOBALQUALIFIER
void warpcore_lookup_kernel_naive(wc_table hash_table, const typename wc_table::key_type* keys, smallsize* result, const smallsize num_in) {
    const size_t tid = global_thread_id();
    const smallsize gid = tid / wc_table::cg_size();
    const auto group = cooperative_groups::tiled_partition<wc_table::cg_size()>(cooperative_groups::this_thread_block());
    if (gid >= num_in) return;

    typename wc_table::key_type key = keys[gid];

    smallsize retrieved_index;
    const auto status = hash_table.retrieve(key, retrieved_index, group);

    if (group.thread_rank() == 0) {
        if (status.has_key_not_found() || status.has_probing_length_exceeded()) {
            result[gid] = not_found;
        } else {
            result[gid] = retrieved_index;
        }
    }
}


template<class wc_table>
GLOBALQUALIFIER
void warpcore_lookup_kernel_multi(wc_table hash_table, const typename wc_table::key_type* keys, smallsize* result, const smallsize num_in) {
    const size_t tid = global_thread_id();
    const smallsize gid = tid / wc_table::cg_size();
    const auto group = cooperative_groups::tiled_partition<wc_table::cg_size()>(cooperative_groups::this_thread_block());

    if ((tid - group.thread_rank()) >= num_in) return;

    typename wc_table::key_type my_lookup = 0;
    bool to_find = false;
    if (tid < num_in) {
          my_lookup = keys[tid];
          to_find = true;
          //if (tid < 32) printf("thread %u loaded key %u\n", (uint32_t) tid, my_lookup);
    }

    smallsize my_result = 0;
    auto work_queue = group.ballot(to_find);
    while (work_queue) {
        auto cur_rank = __ffs(work_queue) - 1;
        auto cur_lookup = group.shfl(my_lookup, cur_rank);

        smallsize cur_result;
        const auto status = hash_table.retrieve(cur_lookup, cur_result, group);
        //if (tid < 32) printf("thread %u looking up key %u\n", (uint32_t) tid, cur_lookup);
        if (status.has_key_not_found() || status.has_probing_length_exceeded()) {
            cur_result = not_found;
        }

        if (cur_rank == group.thread_rank()) {
            my_result = cur_result;
            to_find = false;
        }
        work_queue = group.ballot(to_find);
    }

    if (tid < num_in) {
        result[tid] = my_result;
    }
}


template <typename key_type, typename size_type, typename btree>
GLOBALQUALIFIER
void modified_find_kernel_naive(
    const key_type* keys,
    smallsize* results,
    const size_type keys_count,
    btree tree,
    bool concurrent = false
) {
    auto thread_id  = threadIdx.x + blockIdx.x * blockDim.x;
    const smallsize gid = thread_id / 16;

    auto block = cooperative_groups::this_thread_block();
    auto tile  = cooperative_groups::tiled_partition<btree::branching_factor>(block);

    if (gid >= keys_count) return;

    using allocator_type = device_allocator_context<typename btree::allocator_type>;
    allocator_type allocator{tree.allocator_, tile};

    auto cur_key = keys[gid];
    smallsize cur_result;
    cur_result = tree.cooperative_find(cur_key, tile, allocator, concurrent);
    results[gid] = cur_result != std::numeric_limits<uint32_t>::max() ? cur_result : not_found;
}


template <typename key_type, typename size_type, typename btree>
GLOBALQUALIFIER
void modified_find_kernel_multi(
    const key_type* keys,
    smallsize* results,
    const size_type keys_count,
    btree tree,
    bool concurrent = false
) {
    auto thread_id  = threadIdx.x + blockIdx.x * blockDim.x;

    auto block = cooperative_groups::this_thread_block();
    auto tile  = cooperative_groups::tiled_partition<btree::branching_factor>(block);

    if ((thread_id - tile.thread_rank()) >= keys_count) return;

    auto key     = btree::invalid_key;
    auto value   = btree::invalid_value;
    bool to_find = false;
    if (thread_id < keys_count) {
        key     = keys[thread_id];
        to_find = true;
    }

    using allocator_type = device_allocator_context<typename btree::allocator_type>;
    allocator_type allocator{tree.allocator_, tile};

    auto work_queue = tile.ballot(to_find);
    while (work_queue) {
        auto cur_rank = __ffs(work_queue) - 1;
        auto cur_key  = tile.shfl(key, cur_rank);
        smallsize cur_result;
        cur_result = tree.cooperative_find(cur_key, tile, allocator, concurrent);
        if (cur_rank == tile.thread_rank()) {
            value   = cur_result;
            to_find = false;
        }
        work_queue = tile.ballot(to_find);
    }

    if (thread_id < keys_count) {
        results[thread_id] = value != std::numeric_limits<uint32_t>::max() ? value : not_found;
    }
}


template <typename key_type, typename size_type, typename btree>
GLOBALQUALIFIER
void modified_find_kernel_multi_reg(
    const key_type* keys,
    smallsize* results,
    const size_type keys_count,
    btree tree,
    bool concurrent = false
) {
    constexpr smallsize REGS = 16;

    auto global_offset = blockIdx.x * blockDim.x * REGS + threadIdx.x;
    auto entry_stride = blockDim.x;

    auto block = cooperative_groups::this_thread_block();
    auto tile  = cooperative_groups::tiled_partition<btree::branching_factor>(block);

    if ((global_offset - tile.thread_rank()) >= keys_count) return;

    key_type local_keys[REGS];
    smallsize local_results[REGS];

    for (size_t i = 0; i < REGS; ++i) {
        local_keys[i] = btree::invalid_key;
        local_results[i] = btree::invalid_value;

        smallsize target_index = global_offset + entry_stride * i;

        if (target_index < keys_count) {
            local_keys[i] = keys[target_index];
        }
    }

    using allocator_type = device_allocator_context<typename btree::allocator_type>;
    allocator_type allocator{tree.allocator_, tile};

    for (size_t i = 0; i < REGS; ++i) {
        smallsize target_index = global_offset + entry_stride * i;

        bool to_find = target_index < keys_count;

        auto work_queue = tile.ballot(to_find);
        while (work_queue) {
            auto cur_rank = __ffs(work_queue) - 1;
            auto cur_key = tile.shfl(local_keys[i], cur_rank);
            smallsize cur_result;
            cur_result = tree.cooperative_find(cur_key, tile, allocator, concurrent);
            if (cur_rank == tile.thread_rank()) {
                local_results[i] = cur_result;
                to_find = false;
            }
            work_queue = tile.ballot(to_find);
        }
    }

    for (size_t i = 0; i < REGS; ++i) {
        smallsize target_index = global_offset + entry_stride * i;

        if (target_index < keys_count) {
            results[target_index] = local_results[i] != btree::invalid_value ? local_results[i] : not_found;
        }
    }
}


template <typename key_type>
std::vector<size_t> simulate_gpu_reorder(
        const std::vector<key_type>& entries,
        size_t warps_per_block,
        size_t groups_per_warp,
        size_t threads_per_group,
        size_t registers_per_thread,
        bool cg_aware
) {
    size_t threads_per_block = warps_per_block * 32;
    size_t entries_per_block = threads_per_block * registers_per_thread;
    size_t total_blocks_required = entries.size() / entries_per_block;

    auto p = identity_permutation(entries.size());
    for (size_t block_id = 0; block_id < total_blocks_required; ++block_id) {
        size_t block_offset = block_id * entries_per_block;

        thrust::sort(thrust::host, p.begin() + block_offset, p.begin() + block_offset + entries_per_block, [&entries](size_t a, size_t b) {
            return entries[a] < entries[b];
        });
    }

    if (!cg_aware) return p;

    auto p2 = identity_permutation(entries.size());
    for (size_t block_id = 0; block_id < total_blocks_required; ++block_id) {
        size_t block_offset = block_id * entries_per_block;

        for (size_t register_id = 0; register_id < registers_per_thread; ++register_id) {
        for (size_t thread_id = 0; thread_id < threads_per_group; ++thread_id) {
        for (size_t group_id = 0; group_id < groups_per_warp; ++group_id) {
        for (size_t warp_id = 0; warp_id < warps_per_block; ++warp_id) {
            size_t from = block_offset + register_id * (32 * warps_per_block) + thread_id * (groups_per_warp * warps_per_block) + warp_id * (groups_per_warp) + group_id;
            size_t to = block_offset + register_id * (32 * warps_per_block) + warp_id * 32 + group_id * threads_per_group + thread_id;

            p2[to] = p[from];
        }}}}
    }
    return p2;
}


#define B_UNSORTED 0
#define B_UNSORTED_MULTI 1
#define B_SORTED_MULTI 2
#define B_BLOCK_SORTED_MULTI 4
#define B_BLOCK_SORTED_CG_AWARE_MULTI 5
#define B_BLOCK_SORTED_MULTI_REG 6
#define B_BLOCK_SORTED_CG_AWARE_MULTI_REG 7

#define H_UNSORTED 10
#define H_UNSORTED_MULTI 11
#define H_SORTED_MULTI 12
#define H_BLOCK_SORTED_MULTI 14
#define H_BLOCK_SORTED_CG_AWARE_MULTI 15


template <typename key_type>
void benchmark_lookups() {
    size_t log_build_size = 27;
    size_t log_probe_size = 27;
    size_t build_size = size_t{1} << log_build_size;
    size_t probe_size = size_t{1} << log_probe_size;

    {
        //std::cout << std::string(160, '=') << std::endl;
        auto warps_per_block = 2;
        auto groups_per_warp = 4;
        auto threads_per_group = 8;
        auto registers_per_thread = 4;
        size_t test_size = warps_per_block * 32 * registers_per_thread;
        auto k = identity_permutation(test_size);
        auto p = simulate_gpu_reorder(k, warps_per_block, groups_per_warp, threads_per_group, registers_per_thread, true);
        apply_permutation(k, p);
        for (size_t i = 0; i < test_size; ++i) {
            if (i % (32 * warps_per_block) == 0) std::cout << "== reg " << std::setfill(' ') << std::setw(2) << (i / (32 * warps_per_block)) << " " << std::string(160 - 10, '=') << std::endl;
            if (i % 32 == 0) std::cout << " w" << ((i % (32 * warps_per_block)) / 32) << ":   ";
            std::cout << std::setfill(' ') << std::setw(3) << k[i] << " ";
            if ((i + 1) % 32 == 0) std::cout << std::endl;
            else if ((i + 1) % threads_per_group == 0) std::cout << "| ";
        }
        std::cout << std::string(160, '=') << std::endl;
    }

    for (auto experiment : {0, 1, 2, 4, 5, 6, 7, 10, 11, 12, 14, 15}) {
        // config for multi-register case
        size_t warps_per_block = 4;
        size_t registers_per_thread = 16;

        size_t threads_per_group = experiment < 10 ? 16 : 8;
        size_t groups_per_warp = 32 / threads_per_group;

        point_query_dataset<key_type, smallsize> dataset(
                seed,
                cache_directory,
                build_size,
                probe_size,
                1,
                0,
                true, false,
                0, 0.0, 0, 0,
                1,
                false,
                min_usable_key<key_type>(sizeof(key_type) * 8),
                max_usable_key<key_type>(sizeof(key_type) * 8));

        cuda_buffer<key_type> build_keys_buffer, probe_keys_buffer;
        cuda_buffer<smallsize> result_buffer;
        build_keys_buffer.alloc_and_upload(dataset.build_keys);
        result_buffer.alloc(probe_size);

        double time_ms = 0;
        std::string experiment_name;
        if (experiment < 10) {
            cuda_buffer<smallsize> sorted_offsets_buffer;
            sorted_offsets_buffer.alloc(build_size);
            init_offsets(sorted_offsets_buffer, build_size, nullptr);

            using tree_type = gpu_blink_tree<key_type, smallsize>;
            tree_type tree(build_size, 0);

            tree.bulk_load(build_keys_buffer, sorted_offsets_buffer, build_size, true, 0);

            result_buffer.zero();
            cudaDeviceSynchronize(); CUERR

            switch (experiment) {
                case B_UNSORTED:
                {
                    experiment_name = "B+, unsorted, threads pick same entry";
                    probe_keys_buffer.alloc_and_upload(dataset.probe_keys);
                    {
                        scoped_cuda_timer t(0, &time_ms);
                        modified_find_kernel_naive<<<SDIV(probe_size * threads_per_group, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
                        (probe_keys_buffer.ptr(), result_buffer.ptr(), probe_size, tree);
                    }
                    break;
                }
                case B_UNSORTED_MULTI:
                {
                    experiment_name = "B+, unsorted, threads pick distinct entries";
                    probe_keys_buffer.alloc_and_upload(dataset.probe_keys);
                    {
                        scoped_cuda_timer t(0, &time_ms);
                        modified_find_kernel_multi<<<SDIV(probe_size, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
                        (probe_keys_buffer.ptr(), result_buffer.ptr(), probe_size, tree);
                    }
                    break;
                }
                case B_SORTED_MULTI:
                {
                    experiment_name = "B+, sorted, threads pick distinct entries";
                    auto p = sort_permutation(dataset.probe_keys);
                    apply_permutation(dataset.probe_keys, p);
                    apply_permutation(dataset.expected_result, p);
                    probe_keys_buffer.alloc_and_upload(dataset.probe_keys);
                    {
                        scoped_cuda_timer t(0, &time_ms);
                        modified_find_kernel_multi<<<SDIV(probe_size, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
                        (probe_keys_buffer.ptr(), result_buffer.ptr(), probe_size, tree);
                    }
                    break;
                }
                case B_BLOCK_SORTED_MULTI:
                {
                    experiment_name = "B+, block-sorted, threads pick distinct entries";
                    auto p = simulate_gpu_reorder(dataset.probe_keys, warps_per_block, groups_per_warp, threads_per_group, 1, false);
                    apply_permutation(dataset.probe_keys, p);
                    apply_permutation(dataset.expected_result, p);
                    probe_keys_buffer.alloc_and_upload(dataset.probe_keys);
                    {
                        scoped_cuda_timer t(0, &time_ms);
                        modified_find_kernel_multi<<<SDIV(probe_size, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
                        (probe_keys_buffer.ptr(), result_buffer.ptr(), probe_size, tree);
                    }
                    break;
                }
                case B_BLOCK_SORTED_CG_AWARE_MULTI:
                {
                    experiment_name = "B+, block-sorted (cg aware), threads pick distinct entries";
                    auto p = simulate_gpu_reorder(dataset.probe_keys, warps_per_block, groups_per_warp, threads_per_group, 1, true);
                    apply_permutation(dataset.probe_keys, p);
                    apply_permutation(dataset.expected_result, p);
                    probe_keys_buffer.alloc_and_upload(dataset.probe_keys);
                    {
                        scoped_cuda_timer t(0, &time_ms);
                        modified_find_kernel_multi<<<SDIV(probe_size, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
                        (probe_keys_buffer.ptr(), result_buffer.ptr(), probe_size, tree);
                    }
                    break;
                }
                case B_BLOCK_SORTED_MULTI_REG:
                {
                    experiment_name = "B+, block-sorted, threads pick distinct entries, multiple registers";
                    auto p = simulate_gpu_reorder(dataset.probe_keys, warps_per_block, groups_per_warp, threads_per_group, registers_per_thread, false);
                    apply_permutation(dataset.probe_keys, p);
                    apply_permutation(dataset.expected_result, p);

                    probe_keys_buffer.alloc_and_upload(dataset.probe_keys);
                    size_t threads_per_block = warps_per_block * 32;
                    {
                        scoped_cuda_timer t(0, &time_ms);
                        modified_find_kernel_multi_reg<<<SDIV(probe_size / registers_per_thread, threads_per_block), threads_per_block>>>
                        (probe_keys_buffer.ptr(), result_buffer.ptr(), probe_size, tree);
                    }
                    break;
                }
                case B_BLOCK_SORTED_CG_AWARE_MULTI_REG:
                {
                    experiment_name = "B+, block-sorted (cg aware), threads pick distinct entries, multiple registers";
                    auto p = simulate_gpu_reorder(dataset.probe_keys, warps_per_block, groups_per_warp, threads_per_group, registers_per_thread, true);
                    apply_permutation(dataset.probe_keys, p);
                    apply_permutation(dataset.expected_result, p);

                    probe_keys_buffer.alloc_and_upload(dataset.probe_keys);
                    size_t threads_per_block = warps_per_block * 32;
                    {
                        scoped_cuda_timer t(0, &time_ms);
                        modified_find_kernel_multi_reg<<<SDIV(probe_size / registers_per_thread, threads_per_block), threads_per_block>>>
                        (probe_keys_buffer.ptr(), result_buffer.ptr(), probe_size, tree);
                    }
                    break;
                }
            }
        } else {
            using wc_table = warpcore::SingleValueHashTable<key_type, smallsize, key_type(-2), key_type(-1)>;
            wc_table hashtable(static_cast<size_t>(build_size * 100.0 / 80));

            warpcore_build_kernel<<<SDIV(build_size * wc_table::cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE>>>(hashtable, build_keys_buffer, build_size);

            switch (experiment) {
                case H_UNSORTED:
                {
                    experiment_name = "warpcore, unsorted, threads pick same entry";
                    probe_keys_buffer.alloc_and_upload(dataset.probe_keys);
                    {
                        scoped_cuda_timer t(0, &time_ms);
                        warpcore_lookup_kernel_naive
                        <<<SDIV(probe_size * 8, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
                        (hashtable, probe_keys_buffer, result_buffer, probe_size);
                    }
                    break;
                }
                case H_UNSORTED_MULTI:
                {
                    experiment_name = "warpcore, unsorted, threads pick distinct entries";
                    probe_keys_buffer.alloc_and_upload(dataset.probe_keys);
                    {
                        scoped_cuda_timer t(0, &time_ms);
                        warpcore_lookup_kernel_multi
                        <<<SDIV(probe_size, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
                        (hashtable, probe_keys_buffer, result_buffer, probe_size);
                    }
                    break;
                }
                case H_SORTED_MULTI:
                {
                    experiment_name = "warpcore, sorted, threads pick distinct entries";
                    auto p = sort_permutation(dataset.probe_keys);
                    apply_permutation(dataset.probe_keys, p);
                    apply_permutation(dataset.expected_result, p);
                    probe_keys_buffer.alloc_and_upload(dataset.probe_keys);
                    {
                        scoped_cuda_timer t(0, &time_ms);
                        warpcore_lookup_kernel_multi
                        <<<SDIV(probe_size, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
                        (hashtable, probe_keys_buffer, result_buffer, probe_size);
                    }
                    break;
                }
                case H_BLOCK_SORTED_MULTI:
                {
                    experiment_name = "warpcore, block-sorted, threads pick distinct entries";
                    auto p = simulate_gpu_reorder(dataset.probe_keys, warps_per_block, groups_per_warp, threads_per_group, 1, false);
                    apply_permutation(dataset.probe_keys, p);
                    apply_permutation(dataset.expected_result, p);
                    probe_keys_buffer.alloc_and_upload(dataset.probe_keys);
                    {
                        scoped_cuda_timer t(0, &time_ms);
                        warpcore_lookup_kernel_multi
                        <<<SDIV(probe_size, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
                        (hashtable, probe_keys_buffer, result_buffer, probe_size);
                    }
                    break;
                }
                case H_BLOCK_SORTED_CG_AWARE_MULTI:
                {
                    experiment_name = "warpcore, block-sorted (cg aware), threads pick distinct entries";
                    auto p = simulate_gpu_reorder(dataset.probe_keys, warps_per_block, groups_per_warp, threads_per_group, 1, true);
                    apply_permutation(dataset.probe_keys, p);
                    apply_permutation(dataset.expected_result, p);
                    probe_keys_buffer.alloc_and_upload(dataset.probe_keys);
                    {
                        scoped_cuda_timer t(0, &time_ms);
                        warpcore_lookup_kernel_multi
                        <<<SDIV(probe_size, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
                        (hashtable, probe_keys_buffer, result_buffer, probe_size);
                    }
                    break;
                }
            }
        }
        cudaDeviceSynchronize(); CUERR

        check_result(dataset.probe_keys, dataset.probe_keys, dataset.expected_result, result_buffer);

        std::cout << "EXPERIMENT: " << experiment_name << std::string(80 - experiment_name.size(), ' ') << " | TIME: " << time_ms << " ms" << std::endl;
    }
}

#endif
