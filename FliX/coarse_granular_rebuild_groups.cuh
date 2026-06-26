// =============================================================================
// File: coarse_granular_rebuild_groups.cuh
// Author: Rosina Kharal
// Description: Implements coarse_granular_rebuild_groups
//              Rebuild functions using groups aka tiles
//              Prior functions kept in for backward compatibility. 
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_REBUILD_GROUPS_CUH
#define INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_REBUILD_GROUPS_CUH

#include "coarse_granular_inserts.cuh"
#include "coarse_granular_rebuild.cuh"

#include <cuda_runtime.h>
#include <iostream>
// #include "cuda_buffer.h"
#include <cooperative_groups.h>
namespace coop_g = cooperative_groups;


//rebuilding and merging using one tile.
// re verticalize the data structure
template <typename key_type, int TILE_SIZE>
__global__ void rebuild_kernel_compact_one_tile(
    void *node_buffer,
    void *representative_temp_buffer,
    void *allocation_buffer,
    smallsize* d_nodes_per_bucket,
    smallsize node_size,
    smallsize node_stride,
    smallsize partition_count_with_overflow,
    smallsize total_nodes_used_from_AR,   // kept for optional debug functions for printing 
    smallsize *out_written_nodes)
{
    (void)total_nodes_used_from_AR;

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    // Exactly one tile does all the work.
    if (blockIdx.x != 0 || tile.meta_group_rank() != 0) return;

    if (node_size > TILE_SIZE) {
        if (tile.thread_rank() == 0) {
            printf("ERROR: node_size (%u) > TILE_SIZE (%d)\n",
                   static_cast<unsigned>(node_size), TILE_SIZE);
        }
        return;
    }
    const unsigned tid = tile.thread_rank();
    const smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    smallsize written = 0;
    smallsize local_merge_count = 0; // For optional debug: count how many merges we performed.
    smallsize global_merge_count = 0; // For optional debug: global count of merges (if we had multiple tiles, would need atomic add).

    for (smallsize bucket_id = 0; bucket_id < partition_count_with_overflow; ++bucket_id)
    {
        // Head node for this bucket is in the fixed bucket region of node_buffer.
        uint8_t *node_ptr = reinterpret_cast<uint8_t *>(node_buffer) + bucket_id * node_stride;
        local_merge_count = 0; // Reset local merge count for this bucket.

        while (node_ptr != nullptr)
        {
            //const unsigned tid = tile.thread_rank();

            // Read current node metadata.
            key_type curr_max{};
            smallsize curr_size = 0;
            smallsize curr_link = 0;

            if (tid == 0) {
                curr_max  = cg::extract<key_type>(node_ptr, 0);
                curr_size = cg::extract<smallsize>(node_ptr, sizeof(key_type));
                curr_link = cg::extract<smallsize>(node_ptr, lastposition_bytes);
            }

            // Broadcast metadata within the tile.
            curr_size = static_cast<smallsize>(tile.shfl(static_cast<unsigned>(curr_size), 0));
            // NOTE: for curr_max (arbitrary key_type), we only need it in tid==0 for writing header.
            // We'll re-extract it in tid==0 at write time to avoid shfl for non-integral types.

            // Determine next pointer (may be null).
            uintptr_t next_ptr_u = 0;
            if (tid == 0) {
                if (curr_link != 0) {
                    next_ptr_u = reinterpret_cast<uintptr_t>(
                        reinterpret_cast<uint8_t *>(allocation_buffer) + (curr_link - 1) * node_stride
                    );
                } else {
                    next_ptr_u = 0;
                }
            }
            next_ptr_u = tile.shfl(next_ptr_u, 0);
            uint8_t *next_ptr = reinterpret_cast<uint8_t *>(next_ptr_u);

            // Drop zero-size nodes 
            if (curr_size == 0 && (bucket_id != partition_count_with_overflow-1)) {
                node_ptr = next_ptr;
                tile.sync();
                continue;
            }

            // Lookahead for merge possibility
            bool do_merge = false;
            smallsize next_size = 0;
            key_type next_max{};
            smallsize next_link = 0;
            uintptr_t next_next_ptr_u = 0;

            const smallsize half = static_cast<smallsize>(node_size / 2);
            const smallsize fill_threshold = (node_size / 2);

            if (next_ptr != nullptr) {
                if (tid == 0) {
                    next_size = cg::extract<smallsize>(next_ptr, sizeof(key_type));
                    next_max  = cg::extract<key_type>(next_ptr, 0);
                    next_link = cg::extract<smallsize>(next_ptr, lastposition_bytes);

                    if (next_link != 0) {
                        next_next_ptr_u = reinterpret_cast<uintptr_t>(
                            reinterpret_cast<uint8_t *>(allocation_buffer) + (next_link - 1) * node_stride
                        );
                    } else {
                        next_next_ptr_u = 0;
                    }

                    const bool next_nonzero = (next_size != 0);
                    const bool curr_under   = (curr_size < half);
                    const bool next_under   = (next_size < half);
                    const bool fits         = (static_cast<smallsize>(curr_size + next_size) <= fill_threshold);

                    do_merge = (next_nonzero && curr_under && next_under && fits);
                }

                do_merge = static_cast<bool>(tile.shfl(static_cast<unsigned>(do_merge), 0));
                next_size = static_cast<smallsize>(tile.shfl(static_cast<unsigned>(next_size), 0));
            }

            const smallsize out_index = written;
            uint8_t *out_node =
                reinterpret_cast<uint8_t *>(representative_temp_buffer) + out_index * node_stride;

            // Write header/meta.
            if (tid == 0)
            {
                key_type out_max = cg::extract<key_type>(node_ptr, 0);
                smallsize out_size = curr_size;

                if (do_merge) {
                    out_max  = cg::extract<key_type>(next_ptr, 0); // max becomes next_node_max (per spec)
                    out_size = static_cast<smallsize>(curr_size + next_size);
                }

                cg::set<key_type>(out_node, 0, out_max);
                cg::set<smallsize>(out_node, sizeof(key_type), out_size);

                // Dense output: no link chain.
                cg::set<smallsize>(out_node, lastposition_bytes, static_cast<smallsize>(0));
            }

            // Cooperative copy of slots.
            //
            // Existing convention from your code:
            // tid in [0..node_size-1] 
            if (tid <= static_cast<unsigned>(node_size - 1))
            {
                const smallsize slot = static_cast<smallsize>(tid + 1);

                if (!do_merge)
                {
                    if (slot <= curr_size) {
                        const key_type k = extract_key_node<key_type>(node_ptr, slot);
                        const smallsize off = extract_offset_node<key_type>(node_ptr, slot);
                        set_key_node<key_type>(out_node, slot, k);
                        set_offset_node<key_type>(out_node, slot, off);
                    }
                }
                else
                {
                    local_merge_count++; 
                    const smallsize merged_size = static_cast<smallsize>(curr_size + next_size);
                    if (slot <= merged_size)
                    {
                        if (slot <= curr_size) {
                            const key_type k = extract_key_node<key_type>(node_ptr, slot);
                            const smallsize off = extract_offset_node<key_type>(node_ptr, slot);
                            set_key_node<key_type>(out_node, slot, k);
                            set_offset_node<key_type>(out_node, slot, off);
                        } else {
                            const smallsize src_slot = static_cast<smallsize>(slot - curr_size);
                            const key_type k = extract_key_node<key_type>(next_ptr, src_slot);
                            const smallsize off = extract_offset_node<key_type>(next_ptr, src_slot);
                            set_key_node<key_type>(out_node, slot, k);
                            set_offset_node<key_type>(out_node, slot, off);
                        }
                    }
                }
            }

            tile.sync();

            // Advance traversal pointer.
            if (do_merge)
            {
                // Skip over next node (merged into current output).
                if (tid == 0) {
                    // next_next_ptr_u was computed only in tid==0
                }
                next_next_ptr_u = tile.shfl(next_next_ptr_u, 0);
                node_ptr = reinterpret_cast<uint8_t *>(next_next_ptr_u);
            }
            else
            {
                node_ptr = next_ptr;
            }

            // One output node written (either single node or merged pair).
            ++written;
            tile.sync();
        }

        global_merge_count += local_merge_count;
    }

    if (tid == 0) {
        *out_written_nodes = written;  //send back
    }

#ifdef PRINT_REBUILD_MERGE_DATA
        if (tid == 0) printf("Total merges in the Data structure: %u\n", static_cast<unsigned>(global_merge_count));
#endif
    #ifdef PRINT_REBUILD_DATA_END
    if (tid == 0 )
    {
        printf("After TILED Compact REBUILD Total # Nodes: %u\n", written);
        print_set_nodes_and_links<key_type>(
            representative_temp_buffer,
            allocation_buffer,
            node_size,
            node_stride,
            written,
            total_nodes_used_from_AR);
    }
    #endif
}

template <typename key_type>
smallsize rebuild_gpu_structures_compact_one_tile(
    void *node_buffer,
    void *representative_temp_buffer,
    void *allocation_buffer,
    key_type *maxbuf, 
    smallsize node_size,
    smallsize node_stride,
    double *time_ms,
    smallsize total_nodes_used_from_AR,
    smallsize partition_count_with_overflow)
{
    (void)maxbuf;

   
    if (time_ms) *time_ms = 0.0;


    smallsize node_count = total_nodes_used_from_AR + partition_count_with_overflow;
    smallsize *nodes_per_bucket = new smallsize[partition_count_with_overflow];

    launch_count_nodes_per_bucket<key_type>(
        node_buffer, allocation_buffer, node_size, node_stride,
        total_nodes_used_from_AR, partition_count_with_overflow, nodes_per_bucket);

    cuda_buffer<smallsize> d_nodes_per_bucket;
    d_nodes_per_bucket.alloc_and_upload(
        std::vector<smallsize>(nodes_per_bucket, nodes_per_bucket + partition_count_with_overflow));
    // Device-side counter for how many nodes we actually wrote.
    smallsize *d_out_count = nullptr;
    cudaMalloc(&d_out_count, sizeof(smallsize));
    cudaMemset(d_out_count, 0, sizeof(smallsize));

    
    dim3 blocks(1);
    dim3 threads(TILE_SIZE);

    rebuild_kernel_compact_one_tile<key_type, TILE_SIZE><<<blocks, threads>>>(
        node_buffer,
        representative_temp_buffer,
        allocation_buffer,
        d_nodes_per_bucket,
        node_size,
        node_stride,
        partition_count_with_overflow,
        total_nodes_used_from_AR,
        d_out_count
    );

    cudaStreamSynchronize(0);
    CUERR;

    smallsize h_out_count = 0;
    cudaMemcpy(&h_out_count, d_out_count, sizeof(smallsize), cudaMemcpyDeviceToHost);
    cudaFree(d_out_count);

    return h_out_count;
}



template <typename key_type, int TILE_SIZE>
__global__ void rebuild_kernel_tile_per_node(
    void *node_buffer,
    void *representative_temp_buffer,
    void *allocation_buffer,
    smallsize node_size,
    smallsize node_stride,
    smallsize partition_count_with_overflow,
    smallsize total_nodes_used_from_AR,
    smallsize total_nodes,
    smallsize *nodes_per_bucket,
    smallsize *prefix_sum_array)
{
    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    const smallsize tiles_per_block = static_cast<smallsize>(blockDim.x / TILE_SIZE);
    const smallsize tile_id = static_cast<smallsize>(blockIdx.x) * tiles_per_block + tile.meta_group_rank();

    if (tile_id >= partition_count_with_overflow)
        return;

    if (node_size > TILE_SIZE) {
        if (tile.thread_rank() == 0) {
            printf("ERROR: node_size (%u) > TILE_SIZE (%d)\n",
                   static_cast<unsigned>(node_size), TILE_SIZE);
        }
        return;
    }

    const smallsize node_count_for_this_partition = nodes_per_bucket[tile_id];
    if (node_count_for_this_partition == 0)
        return;

    const smallsize prefix_sum_value = (tile_id == 0) ? 0 : prefix_sum_array[tile_id - 1];
    const smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    // Head node for this bucket.
    uint8_t *node_ptr = reinterpret_cast<uint8_t *>(node_buffer) + tile_id * node_stride;
    const unsigned tid = tile.thread_rank();
    for (smallsize node_index = 0; node_index < node_count_for_this_partition; ++node_index)
    {
        uint8_t *rep_node =
            reinterpret_cast<uint8_t *>(representative_temp_buffer)
            + (prefix_sum_value + node_index) * node_stride;

        

        // Thread 0 copies the node header/meta (max_key + size) and clears rep link.
        if (tid == 0)
        {
            key_type curr_max = cg::extract<key_type>(node_ptr, 0);
            smallsize curr_size = cg::extract<smallsize>(node_ptr, sizeof(key_type));

            cg::set<key_type>(rep_node, 0, curr_max);
            cg::set<smallsize>(rep_node, sizeof(key_type), curr_size);

            // Representative nodes are stored densely; no linked chain in the output.
            cg::set<smallsize>(rep_node, lastposition_bytes, static_cast<smallsize>(0));
        }

        // Threads cooperatively copy
        // Slot 0 is reserved for metadata
        if (tid <= static_cast<unsigned>(node_size - 1))
        {
            const smallsize slot = static_cast<smallsize>(tid + 1);

            const key_type k = extract_key_node<key_type>(node_ptr, slot);
            const smallsize off = extract_offset_node<key_type>(node_ptr, slot);

            set_key_node<key_type>(rep_node, slot, k);
            set_offset_node<key_type>(rep_node, slot, off);
        }

        tile.sync();

        // Advance to next node in the bucket chain (thread 0 reads link and broadcasts next pointer).
        if (node_index + 1 < node_count_for_this_partition)
        {
            smallsize link = 0;
            uintptr_t next_ptr_u = 0;

            if (tid == 0)
            {
                link = cg::extract<smallsize>(node_ptr, lastposition_bytes);
                if (link == 0) {
                    printf("ERROR: link == 0 | tile_id %d | node_index %u\n",
                           static_cast<int>(tile_id), static_cast<unsigned>(node_index));
                    next_ptr_u = 0;
                } else {
                    next_ptr_u = reinterpret_cast<uintptr_t>(
                        reinterpret_cast<uint8_t *>(allocation_buffer) + (link - 1) * node_stride
                    );
                }
            }

            next_ptr_u = tile.shfl(next_ptr_u, 0);
            if (next_ptr_u == 0)
                return;

            node_ptr = reinterpret_cast<uint8_t *>(next_ptr_u);
            tile.sync();
        }
    }

#ifdef PRINT_REBUILD_DATA_END
    if (tid == 0 && blockIdx.x == 0)
    {
        printf("After TILED REBUILD Total # Nodes: %u\n", static_cast<unsigned>(total_nodes));
        print_set_nodes_and_links<key_type>(
            representative_temp_buffer,
            allocation_buffer,
            node_size,
            node_stride,
            partition_count_with_overflow,
            total_nodes_used_from_AR);
    }
#endif
}

template <typename key_type, int TILE_SIZE>
__global__ void rebuild_kernel_tile_per_node_DEBUG(
    void *node_buffer,
    void *representative_temp_buffer,
    void *allocation_buffer,
    smallsize node_size,
    smallsize node_stride,
    smallsize partition_count_with_overflow,
    smallsize total_nodes_used_from_AR,
    smallsize total_nodes,
    smallsize *nodes_per_bucket,
    smallsize *prefix_sum_array)
{
    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    const smallsize tiles_per_block = static_cast<smallsize>(blockDim.x / TILE_SIZE);
    const smallsize tile_id = static_cast<smallsize>(blockIdx.x) * tiles_per_block + tile.meta_group_rank();

    if (tile_id >= partition_count_with_overflow)
        return;

    if (node_size > TILE_SIZE) {
        if (tile.thread_rank() == 0) {
            printf("ERROR: node_size (%u) > TILE_SIZE (%d)\n",
                   static_cast<unsigned>(node_size), TILE_SIZE);
            DEBUG_REBUILD_DEV("ERROR node_size>TILE", (unsigned)node_size, (unsigned)TILE_SIZE);
        }
        return;
    }

    const smallsize node_count_for_this_partition = nodes_per_bucket[tile_id];
    if (node_count_for_this_partition == 0) {
        if (tile.thread_rank() == 0) {
            DEBUG_REBUILD_DEV("ZeroNodes bucket", (unsigned)tile_id, (unsigned)partition_count_with_overflow);
        }
        return;
    }

    const smallsize prefix_sum_value = (tile_id == 0) ? 0 : prefix_sum_array[tile_id - 1];
    const smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    if (tile.thread_rank() == 0) {
        DEBUG_REBUILD_DEV("Bucket begin", (unsigned)tile_id, (unsigned)node_count_for_this_partition, (unsigned)prefix_sum_value);
    }

    // Head node for this bucket.
    uint8_t *node_ptr = reinterpret_cast<uint8_t *>(node_buffer) + tile_id * node_stride;

    for (smallsize node_index = 0; node_index < node_count_for_this_partition; ++node_index)
    {
        uint8_t *rep_node =
            reinterpret_cast<uint8_t *>(representative_temp_buffer)
            + (prefix_sum_value + node_index) * node_stride;

        const unsigned tid = tile.thread_rank();

        // Thread 0 copies the node header/meta (max_key + size) and clears rep link.
        if (tid == 0)
        {
            key_type curr_max = cg::extract<key_type>(node_ptr, 0);
            smallsize curr_size = cg::extract<smallsize>(node_ptr, sizeof(key_type));

            DEBUG_REBUILD_DEV("Node Meta data hdr",tile_id, curr_size, curr_max);
            //DEBUG_REBUILD_DEV("Node max", curr_max, tile_id, node_index);

            // Print keys/offsets in source node slot-by-slot (debug: "keys one by one")
            for (smallsize s = 1; s <= node_size; ++s) {
                const key_type k_dbg = extract_key_node<key_type>(node_ptr, s);
                const smallsize off_dbg = extract_offset_node<key_type>(node_ptr, s);

                DEBUG_REBUILD_DEV("Key(slot)", (unsigned long long)k_dbg, (unsigned)s, (unsigned)node_index);
                DEBUG_REBUILD_DEV("Off(slot)", (unsigned)off_dbg, (unsigned)s, (unsigned)node_index);
            }

            cg::set<key_type>(rep_node, 0, curr_max);
            cg::set<smallsize>(rep_node, sizeof(key_type), curr_size);

            // Representative nodes are stored densely; no linked chain in the output.
            cg::set<smallsize>(rep_node, lastposition_bytes, static_cast<smallsize>(0));
        }

        // Threads cooperatively copy key/offset slots [1 .. node_size-1].
        // Slot 0 is reserved for metadata/header.
        if (tid <= static_cast<unsigned>(node_size - 1))
        {
            const smallsize slot = static_cast<smallsize>(tid + 1);

            const key_type k = extract_key_node<key_type>(node_ptr, slot);
            const smallsize off = extract_offset_node<key_type>(node_ptr, slot);

            set_key_node<key_type>(rep_node, slot, k);
            set_offset_node<key_type>(rep_node, slot, off);

             if (tile_id == 2) {
                DEBUG_REBUILD_DEV("In Loop Copied node", tid, k, node_size);
             }
        }

        tile.sync();

        if (tile_id == 2) {
            DEBUG_REBUILD_DEV("Copied node", (unsigned)tid, (unsigned)node_index, (unsigned)node_size);
        }

        // Advance to next node in the bucket chain (thread 0 reads link and broadcasts next pointer).
        if (node_index + 1 < node_count_for_this_partition)
        {
            smallsize link = 0;
            uintptr_t next_ptr_u = 0;

            if (tid == 0)
            {
                link = cg::extract<smallsize>(node_ptr, lastposition_bytes);
                DEBUG_REBUILD_DEV("Link read", (unsigned)link, (unsigned)tile_id, (unsigned)node_index);

                if (link == 0) {
                    printf("ERROR: link == 0 | tile_id %d | node_index %u\n",
                           static_cast<int>(tile_id), static_cast<unsigned>(node_index));
                    DEBUG_REBUILD_DEV("ERROR link==0", (unsigned)tile_id, (unsigned)node_index);
                    next_ptr_u = 0;
                } else {
                    next_ptr_u = reinterpret_cast<uintptr_t>(
                        reinterpret_cast<uint8_t *>(allocation_buffer) + (link - 1) * node_stride
                    );
                }
            }

            next_ptr_u = tile.shfl(next_ptr_u, 0);
            if (next_ptr_u == 0)
                return;

            node_ptr = reinterpret_cast<uint8_t *>(next_ptr_u);
            tile.sync();

            if (tid == 0) {
                DEBUG_REBUILD_DEV("Advance was ok", (unsigned)tile_id, (unsigned)(node_index + 1), (unsigned)link);
            }
        }
        else
        {
            if (tid == 0) {
                DEBUG_REBUILD_DEV("Bucket done", (unsigned)tile_id, (unsigned)node_index, (unsigned)node_count_for_this_partition);
            }
        }
    }

#ifdef PRINT_REBUILD_DATA_END
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("After TILED REBUILD Total # Nodes: %u\n", static_cast<unsigned>(total_nodes));
        DEBUG_REBUILD_DEV("After rebuild nodes", (unsigned)total_nodes, (unsigned)total_nodes_used_from_AR, (unsigned)partition_count_with_overflow);
        print_set_nodes_and_links<key_type>(
            representative_temp_buffer,
            allocation_buffer,
            node_size,
            node_stride,
            partition_count_with_overflow,
            total_nodes_used_from_AR);
    }
#endif
}



template <typename key_type>
__global__ void rebuild_kernel_group(
    void *node_buffer,
    void *representative_temp_buffer,
    void *allocation_buffer,
    smallsize node_size,
    smallsize node_stride,
    smallsize partition_count_with_overflow,
    smallsize total_nodes_used_from_AR,
    smallsize total_nodes,
    smallsize *nodes_per_bucket,
    smallsize *prefix_sum_array,
    smallsize tile_size)
{
    // Cooperative group: get block and tile of size 8
    coop_g::thread_block block = coop_g::this_thread_block();
    // coop_g::thread_block_tile<tile_size> tile = coop_g::tiled_partition<tile_size>(block);
    coop_g::thread_block_tile<8> tile = coop_g::tiled_partition<8>(block);

    // Use 1 tile per bucket: so tile ID (one per tile) = partition index
    const smallsize tile_id = blockIdx.x * (blockDim.x / tile_size) + tile.meta_group_rank();

    if (tile_id >= partition_count_with_overflow)
        return;

    smallsize node_count_for_this_partition = nodes_per_bucket[tile_id];
    smallsize prefix_sum_value = (tile_id == 0) ? 0 : prefix_sum_array[tile_id - 1];
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    // Each tile handles one bucket's node chain
    if (tile.thread_rank() >= node_count_for_this_partition)
        return;

    smallsize total_nodes_this_bucket = node_count_for_this_partition;
    smallsize base = total_nodes_this_bucket / tile.size();
    smallsize extra = (tile.thread_rank() < (total_nodes_this_bucket % tile.size())) ? 1 : 0;
    smallsize nodes_to_copy = base + extra;

    uint8_t *curr_node = reinterpret_cast<uint8_t *>(node_buffer) + tile_id * node_stride;

    for (int k = 0; k < nodes_to_copy; k++)
    {
        // Traverse to node: thread_rank + k * tile.size()
        int index_in_chain = tile.thread_rank() + k * tile.size();

        uint8_t *node_ptr = curr_node;
        for (int step = 0; step < index_in_chain; step++)
        {
            smallsize link = cg::extract<smallsize>(node_ptr, lastposition_bytes);
            if (link == 0) {
                printf("ERROR: link == 0 | tile_id %d | thread %d | index_in_chain %d\n",
                       tile_id, tile.thread_rank(), index_in_chain);
                return; // Should not happen
            }
            node_ptr = reinterpret_cast<uint8_t *>(allocation_buffer) + (link - 1) * node_stride;
        }

        // Compute output position
        auto rep_node = reinterpret_cast<uint8_t *>(representative_temp_buffer) + (prefix_sum_value + index_in_chain) * node_stride;

        // Copy keys and offsets
        smallsize r = 0;
        for (smallsize m = 0; m <= node_size; m++)
        {
            key_type key = extract_key_node<key_type>(node_ptr, m);
            smallsize offset = extract_offset_node<key_type>(node_ptr, m);

            if (key == tombstone && offset == 0)
                continue;

            set_key_node<key_type>(rep_node, r, key);
            set_offset_node<key_type>(rep_node, r, offset);
            r++;
        }

        // Sanity check
        smallsize rep_link = cg::extract<smallsize>(rep_node, lastposition_bytes);
        if (rep_link != 0)
        {
            printf("ERROR: rep_node linked_ptr != 0 | tile_id %d | thread %d | node_index %d\n",
                   tile_id, tile.thread_rank(), index_in_chain);
            print_node<key_type>(rep_node, node_size);
        }
    }
#ifdef PRINT_REBUILD_DATA_END
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("After REBUILD Total # Nodes: %u\n", total_nodes);
        print_set_nodes_and_links<key_type>(representative_temp_buffer, allocation_buffer, node_size, node_stride, partition_count_with_overflow, total_nodes_used_from_AR);
    }
#endif
}

template <typename key_type>
__global__ void rebuild_kernel_group_01(
    void *node_buffer,
    void *representative_temp_buffer,
    void *allocation_buffer,
    smallsize node_size,
    smallsize node_stride,
    smallsize partition_count_with_overflow,
    smallsize total_nodes_used_from_AR,
    smallsize total_nodes,
    smallsize *nodes_per_bucket,
    smallsize *prefix_sum_array,
    smallsize tile_size)
{
    // Cooperative group: get block and tile of size 8
    coop_g::thread_block block = coop_g::this_thread_block();
    // coop_g::thread_block_tile<tile_size> tile = coop_g::tiled_partition<tile_size>(block);
    coop_g::thread_block_tile<8> tile = coop_g::tiled_partition<8>(block);

    // Use 1 tile per bucket: so tile ID (one per tile) = partition index
    const smallsize tile_id = blockIdx.x * (blockDim.x / tile_size) + tile.meta_group_rank();

    if (tile_id >= partition_count_with_overflow)
        return;

    smallsize node_count_for_this_partition = nodes_per_bucket[tile_id];
    smallsize prefix_sum_value = (tile_id == 0) ? 0 : prefix_sum_array[tile_id - 1];
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    // Each tile handles one bucket's node chain
    if (tile.thread_rank() >= node_count_for_this_partition)
        return;

    // Start from the node assigned to this bucket
    auto curr_node = reinterpret_cast<uint8_t *>(node_buffer) + tile_id * node_stride;

    // Traverse to the i-th node in the chain (based on thread's rank in the tile)
    for (int i = 0; i < tile.thread_rank(); i++)
    {
        smallsize link = cg::extract<smallsize>(curr_node, lastposition_bytes);
        if (link == 0)
            return;
        curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + (link - 1) * node_stride;
    }

    // Compute output location
    auto rep_node = reinterpret_cast<uint8_t *>(representative_temp_buffer) + (prefix_sum_value + tile.thread_rank()) * node_stride;

    // Copy keys and offsets
    smallsize r = 0;
    for (smallsize m = 0; m <= node_size; m++)
    {
        key_type key = extract_key_node<key_type>(curr_node, m);
        smallsize offset = extract_offset_node<key_type>(curr_node, m);

        if (key == tombstone && offset == 0)
            continue;

        set_key_node<key_type>(rep_node, r, key);
        set_offset_node<key_type>(rep_node, r, offset);
        r++;
    }

    // Error check: rep_node's link should be 0
    smallsize rep_link = cg::extract<smallsize>(rep_node, lastposition_bytes);
    if (rep_link != 0)
    {
        printf("ERROR: rep_node linked_ptr != 0 | tile_id %d\n", tile_id);
        print_node<key_type>(rep_node, node_size);
    }


#ifdef PRINT_REBUILD_DATA_END
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("After REBUILD Total # Nodes: %u\n", total_nodes);
        print_set_nodes_and_links<key_type>(representative_temp_buffer, allocation_buffer, node_size, node_stride, partition_count_with_overflow, total_nodes_used_from_AR);
    }
#endif
}

GLOBALQUALIFIER void compute_prefix_sum_array_group(smallsize *d_nodes_per_bucket, smallsize *d_prefix_sum, smallsize size)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_prefix_sum[0] = d_nodes_per_bucket[0];
        // DEBUG_REBUILD("Prefix Sum Array First Element", d_prefix_sum[0], d_nodes_per_bucket[0], size);
        for (smallsize i = 1; i < size; i++)
        {
            d_prefix_sum[i] = d_prefix_sum[i - 1] + d_nodes_per_bucket[i];
        }
    }
}

void compute_prefix_sum_on_gpu_group(cuda_buffer<smallsize> &nodes_per_bucket, cuda_buffer<smallsize> &prefix_sum_array, smallsize partition_count_with_overflow)
{
    prefix_sum_array.resize(partition_count_with_overflow);
    compute_prefix_sum_array<<<1, 1>>>(nodes_per_bucket.ptr(), prefix_sum_array.ptr(), partition_count_with_overflow);
    cudaStreamSynchronize(0);
}

// Function to count the number of zero nodes in the nodes_per_bucket array
smallsize count_zero_nodes_group(const smallsize *nodes_per_bucket, size_t partition_count_with_overflow)
{
    smallsize zero_count = 0;
    for (size_t i = 0; i < partition_count_with_overflow; ++i)
    {
        if (nodes_per_bucket[i] == 0)
        {
            zero_count++;
        }
    }
    return zero_count;
}

template <typename key_type>
smallsize rebuild_gpu_structures_tile(
    void *node_buffer,
    void *representative_temp_buffer,
    void *allocation_buffer,
    key_type *maxbuf,
    smallsize node_size,
    smallsize node_stride,
    double *time_ms,
    smallsize total_nodes_used_from_AR,
    smallsize partition_count_with_overflow)
{
    smallsize node_count = total_nodes_used_from_AR + partition_count_with_overflow;
    smallsize *nodes_per_bucket = new smallsize[partition_count_with_overflow];

    launch_count_nodes_per_bucket<key_type>(
        node_buffer, allocation_buffer, node_size, node_stride,
        total_nodes_used_from_AR, partition_count_with_overflow, nodes_per_bucket);

    cuda_buffer<smallsize> d_nodes_per_bucket;
    d_nodes_per_bucket.alloc_and_upload(
        std::vector<smallsize>(nodes_per_bucket, nodes_per_bucket + partition_count_with_overflow));

    // Compute PREFIX SUM ARRAY
    cuda_buffer<smallsize> d_prefix_sum_array;
    compute_prefix_sum_on_gpu(d_nodes_per_bucket, d_prefix_sum_array, partition_count_with_overflow);

    // Count zero nodes
    smallsize total_zero_nodes = count_zero_nodes(nodes_per_bucket, partition_count_with_overflow);

    // dfine tile params
    const int tile_size = TILE_SIZE; // Must match global definition
    const int total_tiles = partition_count_with_overflow;
    const int tiles_per_block = 16; // Adjust if needed
    const int threads_per_block = tile_size * tiles_per_block;
    const int blocks = (total_tiles + tiles_per_block - 1) / tiles_per_block;

    /* rebuild_kernel_group<key_type><<<blocks, threads_per_block>>>(
            node_buffer,
        representative_temp_buffer,
        allocation_buffer,
        node_size,
        node_stride,
        partition_count_with_overflow,
        total_nodes_used_from_AR,
        node_count,
        d_nodes_per_bucket.ptr(),
        d_prefix_sum_array.ptr(),
        tile_size
    ); */ 

   //rebuild_kernel_tile_per_node_DEBUG<key_type, TILE_SIZE><<<blocks, threads_per_block>>>(
      rebuild_kernel_tile_per_node<key_type, TILE_SIZE><<<blocks, threads_per_block>>>(
 
        node_buffer,
        representative_temp_buffer,
        allocation_buffer,
        node_size,
        node_stride,
        partition_count_with_overflow,
        total_nodes_used_from_AR,
        node_count,
        d_nodes_per_bucket.ptr(),
        d_prefix_sum_array.ptr()
    );

    cudaStreamSynchronize(0);
    CUERR;

    DEBUG_REBUILD_DEV("Total Zero Nodes", total_zero_nodes, partition_count_with_overflow, node_count);
    delete[] nodes_per_bucket;

    // Update the partition count with overflow to be total Node_Count - total_zero_nodes
    partition_count_with_overflow = node_count - total_zero_nodes;

    DEBUG_REBUILD_DEV("NEW Partition Count with Overflow", partition_count_with_overflow);
    // Free CUDA buffers
    d_nodes_per_bucket.free();
    d_prefix_sum_array.free();

    // Optional: inspect maxbuf
    /*
    const key_type *maxbuff = static_cast<const key_type *>(maxbuf);
    printf("PRINTING MAX VALUES at the end of rebuild_gpu_structures\n");
    for (int i = 0; i < partition_count_with_overflow; i++) {
        printf(" maxbuff[%d] %llu \n ", i, static_cast<unsigned long long>(maxbuff[i]));
    }
    */

    return (node_count - total_zero_nodes);
}

//--------------------------------- OLD KERNELS ---------------------------------

template <typename key_type>
GLOBALQUALIFIER void rebuild_kernel_old_group(void *node_buffer, void *representative_temp_buffer, void *allocation_buffer, smallsize node_size, smallsize node_stride, smallsize partition_count_with_overflow, smallsize total_nodes)
{

    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= total_nodes)
        return;

    /*
        smallsize total_nodes = partition_count_with_overflow - 1; // total nodes not including the overflow node
        smallsize key_offset_bytes = compute_key_offset_bytes<key_type>();
        smallsize my_node = tid / initial_num_keys_each_node; // which node this tid belongs to
        smallsize my_place_in_node = (tid + 1) % initial_num_keys_each_node;

        smallsize byte_offset = (node_stride * my_node) + ((tid % initial_num_keys_each_node) + 1) * key_offset_bytes;

        cg::set<key_type>(node_buffer, byte_offset, keys[tid]);
        cg::set<smallsize>(node_buffer, byte_offset + sizeof(key_type), offsets[tid]);

        // extra work if the thread is the last member in the group
        // insert maxkey and size at the front of the node
        if (my_place_in_node == 0)
        {
            // insert maxkey
            cg::set<key_type>(node_buffer, my_node * node_stride, keys[tid]);
            // insert size
            cg::set<smallsize>(node_buffer, my_node * node_stride + sizeof(key_type), initial_num_keys_each_node);
            // clear the lastptr to 0
            cg::set<smallsize>(node_buffer, my_node * node_stride + (node_size + 1) * key_offset_bytes, 0);
        }

        if (my_node == total_nodes - 1)
        { // last node not including overflow node

            smallsize remainder = key_count % initial_num_keys_each_node;
            bool has_remainder = (key_count % initial_num_keys_each_node != 0);
            if (has_remainder)
                my_place_in_node = ((tid + 1) % remainder);

            if (my_place_in_node == 0)
            {

                // this code block repeated in case there is a remainder. last node max needs to be set
                if (has_remainder)
                { // update size correctly in last actual node (not overflow node)

                    DEBUG_BUILD("Node Layout Kernel: Has Remainder true", 5, tid, remainder, my_node, my_place_in_node, keys[key_count - 1]);

                    // set the size to be remainder and the max_key to be the last key in the dataset
                    cg::set<smallsize>(node_buffer, (total_nodes - 1) * node_stride + sizeof(key_type), remainder);
                    cg::set<key_type>(node_buffer, my_node * node_stride, keys[key_count - 1]);
                    // clear the lastptr to 0
                    cg::set<smallsize>(node_buffer, my_node * node_stride + (node_size + 1) * key_offset_bytes, 0);
                }

                key_type datamax; // datamax is the max value for the overflow node
                if (sizeof(key_type) == 8)
                {
                    datamax = static_cast<key_type>(std::numeric_limits<key64>::max());
                }
                else if (sizeof(key_type) == 4)
                {
                    datamax = static_cast<key_type>(std::numeric_limits<key32>::max());
                }

                // set max for OVERFLOW NODE ONLY (overflow node size is 0 at initial build time)
                cg::set<key_type>(node_buffer, (total_nodes)*node_stride, datamax);
            }
        }



    template <typename key_type>
    void rebuild_gpu_structures_A(
        void *node_buffer,
        void *representative_temp_buffer,
        void *allocation_buffer,
        size_t node_size,
        size_t node_stride,
        double *time_ms,
        size_t total_nodes_used_from_AR,
        size_t partition_count_with_overflow)
    {
       // scoped_cuda_timer timer(0, time_ms);
       size_t node_count = total_nodes_used_from_AR +partition_count_with_overflow;
       size_t nodes_per_bucket[partition_count_with_overflow];

    ///--> HERE   kernel to compute the nodes per bucket



       rebuild_kernel<key_type><<<SDIV(node_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(node_buffer, representative_temp_buffer, allocation_buffer, node_size, node_stride, partition_count_with_overflow, node_count);
       cudaStreamSynchronize(0);
       CUERR
       // cudaStreamSynchronize(0);
        //CUERR
    }



    template <typename key_type>
    void rebuild_gpu_structures(
        void *node_buffer,
        void *representative_temp_buffer,
        void *allocation_buffer,
        smallsize node_size,
        smallsize node_stride,
        double *time_ms,
        smallsize total_nodes_used_from_AR,
        smallsize partition_count_with_overflow)
    {
        // Start GPU timer
        // scoped_cuda_timer timer(0, time_ms);

        smallsize node_count = total_nodes_used_from_AR + partition_count_with_overflow;

        // Allocate array for nodes per bucket on host
        smallsize *nodes_per_bucket = new smallsize[partition_count_with_overflow];

        /// ---> CALL THE KERNEL TO COMPUTE THE NODES PER BUCKET
        launch_count_nodes_per_bucket<key_type>(
            node_buffer, allocation_buffer, node_size, node_stride,
            total_nodes_used_from_AR, partition_count_with_overflow, nodes_per_bucket);

        // Allocate nodes_per_bucket on GPU using cuda_buffer
        cuda_buffer<smallsize> d_nodes_per_bucket;
        d_nodes_per_bucket.alloc_and_upload(std::vector<smallsize>(nodes_per_bucket, nodes_per_bucket + partition_count_with_overflow));


       // ----->> HERE  to do PREFIX SUM ARRAY compUTATION

        // Launch the rebuild kernel, passing nodes_per_bucket
       //---> for now Just One Thread per bucket
          // rebuild_kernel<key_type><<<SDIV(node_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(node_buffer, representative_temp_buffer, allocation_buffer, node_size, node_stride, partition_count_with_overflow, node_count, d_nodes_per_bucket.ptr());

            rebuild_kernel<key_type><<<SDIV(partition_count_with_overflow, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(
            node_buffer, representative_temp_buffer, allocation_buffer, node_size, node_stride,
            partition_count_with_overflow, node_count, d_nodes_per_bucket.ptr());

        // Synchronize kernel execution
        cudaStreamSynchronize(0);
        CUERR;

        // Cleanup host memory
        delete[] nodes_per_bucket;

        // cudaStreamSynchronize(0);
        // CUERR;
    }


    for (smallsize m = 0; m <= node_size; ) {  // No ++m in loop header
        key_type key = extract_key_node<key_type>(curr_node, m);
        smallsize offset = extract_offset_node<key_type>(curr_node, m);

        if (key == 1 && offset == 0) {
            // Skip incrementing m, effectively "retrying" same index
            continue;
        }

        set_key_node<key_type>(rep_node, m, key);
        set_offset_node<key_type>(rep_node, m, offset);

        m++;  // Increment manually
    }

        */
}

#endif

/*

 // Copy the node
        //we copy the full node_size bc we should have tombstone present
        smallsize c ,r = 0;
        while( c <= node_size)
        {
            key_type key = extract_key_node<key_type>(curr_node, c);
            smallsize offset = extract_offset_node<key_type>(curr_node, c);
            // check for tombstones and ignore. size of node should still be accurate
            // we are removing all unnecessary tombstones
            if (key == 1 && offset == 0){
                c++;
                continue;
            }

            set_key_node<key_type>(rep_node, r, key);
            set_offset_node<key_type>(rep_node, r, offset);
            c++; r++;
        }

        */