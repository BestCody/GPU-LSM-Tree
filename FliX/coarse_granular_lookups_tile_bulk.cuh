// =============================================================================
// File: coarse_granular_lookups_tile_bulk.cuh
// Author: Rosina Kharal
// Description: Implements coarse_granular_lookups_tile_bulk
//              Bulk Queries- Lookups in FliX
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INDEX_PROTOTYPE_COARSE_GRANULAR_LOOKUPS_TILE_BULK_CUH
#define INDEX_PROTOTYPE_COARSE_GRANULAR_LOOKUPS_TILE_BULK_CUH

// #include <cooperative_groups.h>
#include <cub/cub.cuh>
#include "coarse_granular_inserts_tiles.cuh"
#include "coarse_granular_inserts.cuh"
#include "definitions_updates.cuh"
#include "tile_utils.cuh"

// file: src/process_lookup_tile_bulk_ordered.cuh

#pragma once
#include <cooperative_groups.h>
// namespace cg = cooperative_groups;

// Assumes available: not_found, tombstone, TILE_SIZE,
// cg::extract/set, extract_key_node, extract_offset_node, get_lastposition_bytes.
// testing each thread also picks a key from the query list
template <typename key_type>
DEVICEQUALIFIER void process_lookup_tile_bulk_ordered1_querylist_opt(
    key_type bucket_max,
    smallsize minindex,
    smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const key_type *__restrict__ query_list,
    smallsize query_size,
    smallsize *__restrict__ results)
{
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize allocation_count = launch_params->allocation_buffer_count;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    const smallsize lastpos_off = get_lastposition_bytes<key_type>(node_size);
    const smallsize my_tid = tile.thread_rank();
    const smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();
    smallsize this_tid = my_tid;
    void *curr_node = starting_node;
    key_type curr_max = 0;

    smallsize i = minindex;
    while (i <= maxindex)
    {
        const key_type key = query_list[i]; // reuses name/flow

        // Tile leader walks the node chain until node's max >= key
        if (tile.thread_rank() == 0)
        {
            curr_max = cg::extract<key_type>(curr_node, 0);
            while (curr_max < key)
            {
                const smallsize next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_count)
                {
                    // no more nodes in this chain; nothing left to answer in this bucket
                    break;
                }
                curr_node = static_cast<uint8_t *>(allocation_buffer) + (next_ptr - 1) * node_stride;
                key_type next_max = cg::extract<key_type>(curr_node, 0);
                // safeguard against malformed/non-increasing chains
                if (next_max <= curr_max)
                {
                    curr_max = next_max;
                    break;
                }
                curr_max = next_max;
            }
        }

        // Broadcast current node pointer + max to the tile
        uintptr_t p = reinterpret_cast<uintptr_t>(curr_node);
        p = tile.shfl(p, 0);
        curr_node = reinterpret_cast<void *>(p);
        curr_max = tile.shfl(curr_max, 0);

        // If we can't cover current query or we exceeded bucket boundary, we are done
        // if (curr_max < key || key > bucket_max) break;

        const smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // Snapshot per-lane node entry (why: avoid reloading inside batches)
        key_type my_key = key_type(0);
        smallsize my_offset = 0;
        bool lane_live = false;
        if (my_tid < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, my_tid + 1);
            my_offset = extract_offset_node<smallsize>(curr_node, my_tid + 1);
            // if (my_key != key_type(0) && my_key != static_cast<key_type>(tombstone))
            lane_live = true;
        }

        tile.sync();

        // EVERY THREAD ALSO PICK UP A Key From the Query list beginning at index i, to index i + currsize

        smallsize my_query_index = not_found; // if matched, index in query_list
        unsigned match_mask = 0;              // accumulated across delete keys
        smallsize checks = 0;

        // if (lane_live && (i + my_tid < maxindex)) my_query_key = query_list[i + my_tid];

        // Determine the contiguous subrange [i, j_end) of queries that belong to this node (<= curr_max)
        smallsize scan_i = i;

        while (scan_i <= maxindex)
        {

            key_type lookup_key_loop = query_list[scan_i];

            if (lookup_key_loop > curr_max)
                break;

            const bool eq = lane_live && (my_key == lookup_key_loop);
            if (eq)
                results[scan_i] = my_offset; // to support duplicates
            // my_query_index = scan_i;  // remember match index for result buffer
            else
                results[scan_i] = not_found;

            // if (tile_id ==100 && my_tid == 0)
            //     DEBUG_PI_TILE_DELS("del key is:", del_key, eq, my_tid);

            const smallsize b = tile.ballot(eq); // same on all lanes
            match_mask |= b;                     // using OR accumulate across delete keys

            ++scan_i;
            ++checks;

            /*
            me++;
            if (me >3) {
                me=0;
                in_range = (my_tid <4 &&  (my_tid + scan_i <= maxindex)) ? true: false;
                if (in_range) del_key = update_list[scan_i + my_tid];
            }
            */
        }

        tile.sync();

        // Advance to first query not handled by this node
        i = scan_i;

        // If next query exceeds this bucket's max, nothing more for this tile
        if (query_list[i] > bucket_max)
            break;
        // Otherwise the outer loop continues; tile leader will hop to next node as needed
    }

#ifdef PRINT_PROCESS_LOOKUPS_END
    __syncthreads(); // only valid if all threads in block reach here
    if (tile_id == 0 && this_tid == 0)
    {
        printf("END TILE LOOKUPS: PRINT ALL NODES (debug)\n");
        print_set_nodes_and_links<key_type>(launch_params, this_tid);
    }
#endif
}

// Assumes: not_found, TILE_SIZE, cg::extract/set, extract_key_node, extract_offset_node, get_lastposition_bytes
template <typename key_type>
DEVICEQUALIFIER void process_lookup_tile_bulk_ordered(
    key_type bucket_max,
    smallsize minindex,
    smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const key_type *__restrict__ query_list,
    smallsize query_size,
    smallsize *__restrict__ results)
{
    void *allocation_buffer = launch_params->allocation_buffer;
    const smallsize allocation_count = launch_params->allocation_buffer_count;
    const smallsize node_stride = launch_params->node_stride;
    const smallsize node_size = launch_params->node_size;

    const smallsize lastpos_off = get_lastposition_bytes<key_type>(node_size);
    const smallsize lane = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();
    void *curr_node = starting_node;
    key_type curr_max = key_type(0);

    smallsize i = minindex;
    key_type next_search_key = query_list[i];

    // if (lane ==0) printf(" PROCESS LOOKUP TILE BULK ORDERED: tile_id=%d, bucket_max=%llu, minindex=%d, maxindex=%d\n",
    ///   tile_id, static_cast<unsigned long long>(bucket_max), static_cast<int>(minindex), static_cast<int>(maxindex));

    while (i < query_size && next_search_key <= bucket_max)
    {
        const key_type key = next_search_key;
        // Leader walks chain until node's max >= key
        if (lane == 0)
        {
            curr_max = cg::extract<key_type>(curr_node, 0);
            while (curr_max < key)
            {
                const smallsize next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_count)
                {
                    ERROR_INSERTS("ERROR: invalid link in node chain during tile bulk lookup", key, lane, tile_id);
                }

                curr_node = static_cast<uint8_t *>(allocation_buffer) + (static_cast<size_t>(next_ptr) - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
                // if (next_max <= curr_max) { curr_max = next_max; break; } // guard
                // curr_max = next_max;
            }
        }

        // Broadcast node pointer + max
        uintptr_t p = reinterpret_cast<uintptr_t>(curr_node);
        p = tile.shfl(p, 0);
        curr_node = reinterpret_cast<void *>(p);
        curr_max = tile.shfl(curr_max, 0);

        // Hard stop if node can’t cover current key or bucket bound exceeded
        // if (curr_max < key || key > bucket_max) break;

        const smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // Snapshot per-lane node entry once
        key_type my_key = key_type(0);
        smallsize my_offset = 0;
        bool lane_live = false;
        if (lane < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, lane + 1);
            my_offset = extract_offset_node<smallsize>(curr_node, lane + 1);
            lane_live = true; // keep simple; add tombstone filter later if needed
        }
        tile.sync();

        // Scan query_list while within this node’s max
        smallsize scan_i = i;
        // ----key_type prev_search_key = next_search_key;
        smallsize out = not_found;
        //  if (lane ==0) printf(" Before While loop: tile_id=%d, node_max=%llu, curr_size=%d, processing queries from index %d onwards\n",
        //      tile_id, static_cast<unsigned long long>(curr_max), static_cast<int>(curr_size), static_cast<int>(i));

        while (scan_i < query_size)
        {

            next_search_key = query_list[scan_i];
            if (next_search_key > curr_max)
                break;

            //-----if (prev_search_key != next_search_key) {
            // Compare this query against all per-lane node entries
            const bool eq = lane_live && (my_key == next_search_key);
            const unsigned mask = tile.ballot(eq);

            out = not_found;
            if (mask)
            {
                const smallsize winner = __ffs(mask) - 1;                     // pick one matching lane
                const smallsize result_offset = tile.shfl(my_offset, winner); // fetch its offset
                out = result_offset;
            }
            //----}
            // Single writer per query index to avoid races
            if (lane == 0)
            {

                results[scan_i] = out; // one deterministic write
                                       // printf(" TILE BULK ORDERED: tile_id=%d, query_index=%d, query_key=%llu, result_offset=%d\n",
                // tile_id, static_cast<int>(scan_i), static_cast<unsigned long long>(next_search_key), static_cast<int>(out));
            }
            ++scan_i;

            // tile.sync();
            //---prev_search_key = next_search_key;
        }

        // Advance to the first query outside this node’s range
        i = scan_i;

        // if (lane ==0) printf(" ADVANCE: tile_id=%d, next query index=%d\n", tile_id, static_cast<int>(i));
        tile.sync();
        // Stop if we stepped beyond this bucket
        if (i >= query_size)
            break;
        // --------next_search_key = query_list[i];
    }

#ifdef PRINT_LOOKUPS_END
    tile.sync(); // only valid if all threads in block reach here
    if (tile_id == 0 && lane == 0)
    {
        printf("END TILE LOOKUPS: PRINT ALL NODES (debug)\n");
        print_set_nodes_and_links<key_type>(launch_params, tile_id);
    }
#endif
}

// Assumes: not_found, TILE_SIZE, cg::extract/set, extract_key_node, extract_offset_node, get_lastposition_bytes
template <typename key_type>
DEVICEQUALIFIER void process_lookup_tile_bulk_ordered_dup(
    key_type bucket_max,
    smallsize minindex,
    smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const key_type *__restrict__ query_list,
    smallsize query_size,
    smallsize *__restrict__ results)
{
    void *allocation_buffer = launch_params->allocation_buffer;
    const smallsize allocation_count = launch_params->allocation_buffer_count;
    const smallsize node_stride = launch_params->node_stride;
    const smallsize node_size = launch_params->node_size;

    const smallsize lastpos_off = get_lastposition_bytes<key_type>(node_size);
    const smallsize lane = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();
    void *curr_node = starting_node;
    key_type curr_max = key_type(0);

    smallsize i = minindex;
    key_type next_search_key = query_list[i];

    /*
    if (lane ==0) {
            //printf(" PROCESS LOOKUP TILE BULK ORDERED: tile_id=%d, bucket_max=%llu, minindex=%d, maxindex=%d\n",
            //    tile_id, static_cast<unsigned long long>(bucket_max), static_cast<int>(minindex), static_cast<int>(maxindex));

            // print all query keys in query_list from minidex to the point where next_key is greater than bucket_max
            for (smallsize idx = minindex; idx < query_size; ++idx)
            {
                key_type qkey = query_list[idx];
                if (qkey > bucket_max)
                    break;
                printf(" QUERY LIST KEY: tile_id=%d, query_index=%d, query_key=%llu\n",
                    tile_id, static_cast<int>(idx), static_cast<unsigned long long>(qkey));
            }
        }

        */

    while (i < query_size && next_search_key <= bucket_max)
    {

        const key_type key = next_search_key;

       

        // Leader walks chain until node's max >= key
        if (lane == 0)
        {
            curr_max = cg::extract<key_type>(curr_node, 0);
            while (curr_max < key)
            {
                const smallsize next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_count)
                {
                    ERROR_INSERTS("ERROR: invalid link in node chain during tile bulk lookup", key, lane, tile_id);
                }

                curr_node = static_cast<uint8_t *>(allocation_buffer) + (static_cast<size_t>(next_ptr) - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
                // if (next_max <= curr_max) { curr_max = next_max; break; } // guard
                // curr_max = next_max;
            }
        }

        // if (next_search_key == 1754439130)
        //    printf("Search Key BULK ORDERED: tile_id=%d, query_key=%llu \n",
        //         tile_id, static_cast<unsigned long long>(next_search_key));

        // Broadcast node pointer + max
        uintptr_t p = reinterpret_cast<uintptr_t>(curr_node);
        p = tile.shfl(p, 0);
        curr_node = reinterpret_cast<void *>(p);
        curr_max = tile.shfl(curr_max, 0);

        // Hard stop if node can’t cover current key or bucket bound exceeded
        // if (curr_max < key || key > bucket_max) break;

        const smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // Snapshot per-lane node entry once
        key_type my_key = key_type(0);
        smallsize my_offset = 0;
        bool lane_live = false;
        if (lane < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, lane + 1);
            my_offset = extract_offset_node<smallsize>(curr_node, lane + 1);
            lane_live = true; // keep simple; add tombstone filter later if needed
        }
        tile.sync();

        // Scan query_list while within this node’s max
        smallsize scan_i = i;
        key_type prev_search_key = 0; // next_search_key;
        smallsize out = not_found;
        //  if (lane ==0) printf(" Before While loop: tile_id=%d, node_max=%llu, curr_size=%d, processing queries from index %d onwards\n",
        //      tile_id, static_cast<unsigned long long>(curr_max), static_cast<int>(curr_size), static_cast<int>(i));

        // if (next_search_key == 1754439130)
        //    printf("Search Key 1754439130 BULK ORDERED: tile_id=%d, query_index=%d, query_key=%llu, result_offset=%d\n",
        //           tile_id, static_cast<int>(scan_i), static_cast<unsigned long long>(next_search_key), static_cast<int>(out));

        while (scan_i < query_size)
        {

            next_search_key = query_list[scan_i];

            /* if (lane == 0)
                printf(" In While Loops TILE BULK ORDERED DUP: tile_id=%d, scan_i=%d, next_search_key=%llu, curr_max=%llu. out=%d\ notfound %d\n",
                       tile_id, static_cast<int>(scan_i), static_cast<unsigned long long>(next_search_key),
                       static_cast<unsigned long long>(curr_max), static_cast<int>(out), not_found);
            */

            if (next_search_key > curr_max)
                break;

            if (prev_search_key != next_search_key)
            {
                // Compare this query against all per-lane node entries
                const bool eq = lane_live && (my_key == next_search_key);
                const unsigned mask = tile.ballot(eq);

                out = not_found;
                if (mask)
                {
                    const smallsize winner = __ffs(mask) - 1;                     // pick one matching lane
                    const smallsize result_offset = tile.shfl(my_offset, winner); // fetch its offset
                    out = result_offset;
                }
            }
            // Single writer per query index to avoid races
            if (lane == 0)
            {

                results[scan_i] = out; // one deterministic write
                // if (next_search_key == 1754439130)
                // printf("RESULT BULK ORDERED: tile_id=%d, query_index=%d, query_key=%llu, result_offset=%d\n",
                //     tile_id, static_cast<int>(scan_i), static_cast<unsigned long long>(next_search_key), static_cast<int>(out));
            }
            ++scan_i;

            // tile.sync();
            prev_search_key = next_search_key;
        }

        // Advance to the first query outside this node’s range
        i = scan_i;

        // if (lane ==0) printf(" ADVANCE: tile_id=%d, next query index=%d\n", tile_id, static_cast<int>(i));
        tile.sync();
        // Stop if we stepped beyond this bucket
        if (i >= query_size)
            break;
        // --------next_search_key = query_list[i];
    }

#ifdef PRINT_LOOKUPS_END
    tile.sync(); // only valid if all threads in block reach here
    if (tile_id == 0 && lane == 0)
    {
        printf("END TILE LOOKUPS: PRINT ALL NODES (debug)\n");
        print_set_nodes_and_links<key_type>(launch_params, tile_id);
    }
#endif
}

// template <typename key_type, bool process_single_thread_per_bucket>
template <typename key_type>
GLOBALQUALIFIER void lookup_kernel_tile_ordered_min_maxindex(
    updatable_cg_params *__restrict__ launch_params,
    const key_type *__restrict__ query_list, // sorted queries
    smallsize *__restrict__ results,         // output
    smallsize query_size)
{

    const key_type *__restrict__ maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    const smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    const int tiles_per_block = blockDim.x / tile.size();
    const int tile_id = blockIdx.x * tiles_per_block + tile.meta_group_rank();
    const smallsize lane = tile.thread_rank();

    // if (lane == 0) { printf("LOOKUP TILE BULK ORDERED: tile_id=%d launched\n", tile_id); }

    if (tile_id >= partition_count_with_overflow)
        return;

    const key_type maxkey = maxbuf[tile_id];
    const key_type minkey = (tile_id > 0) ? (maxbuf[tile_id - 1] + static_cast<key_type>(1)) : static_cast<key_type>(1);

    // Find query subrange for this bucket
    int minindex = -1;
    int maxindex = -1;

    if (lane == 0)
    {
        // minindex: first position >= minkey
        minindex = binarySearchIndex_leftmost_ge<key_type>(query_list, minkey, 0, query_size, /*upper_bound?*/ false);
        // maxindex: last position <= maxkey
        // maxindex = binarySearchIndex<key_type>(query_list, maxkey, 0, query_size, /*upper_bound?*/ true);
    }
    minindex = tile.shfl(minindex, 0);
    // maxindex = tile.shfl(maxindex, 0);

    const bool empty = (minindex > maxindex) | (minindex < 0) | (maxindex < 0);
    if (tile.any(empty))
        return;
    if (empty)
        return;

    // Node for this bucket
    uint8_t *__restrict__ curr_node =
        static_cast<uint8_t *>(launch_params->ordered_node_pairs) +
        static_cast<size_t>(launch_params->node_stride) * static_cast<size_t>(tile_id);

   

    process_lookup_tile_bulk_ordered<key_type>(
        /*bucket_max*/ maxkey,
        /*minindex*/ static_cast<smallsize>(minindex),
        /*maxindex*/ static_cast<smallsize>(maxindex),
        launch_params,
        /*curr_node*/ curr_node,
        /*tile*/ tile,
        /*sorted queries*/ query_list,
        /*query_size*/ query_size,
        /*writeback*/ results);

    // if (lane ==0) printf("END LOOKUP TILE BULK ORDERED: tile_id=%d DONE\n", tile_id);
}

// template <typename key_type, bool process_single_thread_per_bucket>
template <typename key_type>
GLOBALQUALIFIER void lookup_kernel_tile_ordered(
    updatable_cg_params *__restrict__ launch_params,
    const key_type *__restrict__ query_list, // sorted queries
    smallsize *__restrict__ results,         // output
    smallsize query_size)
{

    const key_type *__restrict__ maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    const smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    const int tiles_per_block = blockDim.x / tile.size();
    const int tile_id = blockIdx.x * tiles_per_block + tile.meta_group_rank();
    const smallsize lane = tile.thread_rank();

    // if (lane == 0) printf("TOP LOOKUP TILE BULK ORDERED: tile_id=%d partition_count_with_overflow=%d query_size=%d \n", tile_id, partition_count_with_overflow, query_size);

    if (tile_id >= partition_count_with_overflow)
        return;

    const key_type maxkey = maxbuf[tile_id];
    const key_type minkey = (tile_id > 0) ? (maxbuf[tile_id - 1] + static_cast<key_type>(1)) : static_cast<key_type>(1);

    // Find query subrange for this bucket
    int minindex = -1;
    int maxindex = -1;

    if (lane == 0)
    {

        minindex = binarySearchIndex_leftmost_ge<key_type>(query_list, minkey, 0, query_size);
        // maxindex: last position <= maxkey
        // maxindex = binarySearchIndex_first_gt_leftmost_dup<key_type>(query_list, maxkey, 0, query_size);
    }
    minindex = tile.shfl(minindex, 0);
    // maxindex = tile.shfl(maxindex, 0);
    // if (lane ==0) printf("After Searches LOOKUP TILE BULK ORDERED: tile_id=%d, bucket=[%llu..%llu], query_range=[%d..%d]\n",
    // minindex, maxindex);
    // const bool empty = (minindex > maxindex) | (minindex < 0) | (maxindex < 0);
    const bool empty = (minindex < 0);
    if (tile.any(empty))
        // if (lane ==0) printf("EXITING LOOKUP TILE BULK ORDERED: tile_id=%d, empty=%d minkey=%llu maxkey=%llu , minindex=%d \n", tile_id, empty, static_cast<unsigned long long>(minkey), static_cast<unsigned long long>(maxkey), minindex);
        return;

    // if (empty) return;

    // Node for this bucket
    uint8_t *__restrict__ curr_node =
        static_cast<uint8_t *>(launch_params->ordered_node_pairs) +
        static_cast<size_t>(launch_params->node_stride) * static_cast<size_t>(tile_id);

  

    process_lookup_tile_bulk_ordered_dup<key_type>(
        // process_lookup_tile_bulk_ordered<key_type>(

        /*bucket_max*/ maxkey,
        /*minindex*/ static_cast<smallsize>(minindex),
        /*maxindex*/ static_cast<smallsize>(maxindex),
        launch_params,
        /*curr_node*/ curr_node,
        /*tile*/ tile,
        /*sorted queries*/ query_list,
        /*query_size*/ query_size,
        /*writeback*/ results);

    // if (lane ==0) printf("END LOOKUP TILE BULK ORDERED: tile_id=%d DONE\n", tile_id);
}





// Assumes: not_found, TILE_SIZE, cg::extract/set, extract_key_node, extract_offset_node, get_lastposition_bytes
template <typename key_type>
DEVICEQUALIFIER void process_lookup_successor_tile_bulk_ordered_dup(
    key_type bucket_max,
    smallsize minindex,
    smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    key_type next_bucket_first_key,
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const key_type *__restrict__ query_list,
    smallsize query_size,
    key_type *__restrict__ results)
{
    void *allocation_buffer = launch_params->allocation_buffer;
    const smallsize allocation_count = launch_params->allocation_buffer_count;
    const smallsize node_stride = launch_params->node_stride;
    const smallsize node_size = launch_params->node_size;

    const smallsize lastpos_off = get_lastposition_bytes<key_type>(node_size);
    const smallsize lane = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();
    void *curr_node = starting_node;
    key_type curr_max = key_type(0);

    smallsize i = minindex;
    key_type next_search_key = query_list[i];

    // check for last bucket in the DS, may have empty node with next_bucket_first_key = 0

    // NEW ADDED CHECK FOR SUBSEEQUENT NODE in NEXT BUCKET CHAIN

    if (next_bucket_first_key == 0)
    {
        /* next_bucket_node = static_cast<uint8_t *>(allocation_buffer) + (tile_id + 1) * node_stride;
        const smallsize next_bucket_next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
        if (next_bucket_next_ptr != 0 && (next_bucket_next_ptr - 1) < allocation_count)
        {
            // valid next node
            void *next_bucket_node = static_cast<uint8_t *>(allocation_buffer) + (static_cast<size_t>(next_bucket_next_ptr) - 1) * node_stride;
            next_bucket_first_key = extract_key_node<key_type>(next_bucket_node, 1);
        }
        else */

        next_bucket_first_key = static_cast<key_type>(not_found);
    }

    while (i < query_size && next_search_key <= bucket_max)
    {

        const key_type key = next_search_key;

        // if (next_search_key == 1754439130)
        // if (lane ==0) printf("TOP Search Key BULK ORDERED: tile_id=%d, key=%llu \n",
        //       tile_id, static_cast<unsigned long long>(key));

        // Leader walks chain until node's max >= key
        if (lane == 0)
        {
            curr_max = cg::extract<key_type>(curr_node, 0);
            while (curr_max < key)
            {
                const smallsize next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
                //if (next_ptr == 0 || (next_ptr - 1) >= allocation_count)
                if (next_ptr == 0 ) // || (next_ptr - 1) >= allocation_count)

                {
                    ERROR_INSERTS("ERROR: invalid link in node chain during tile bulk lookup", key, lane, tile_id);
                }

                curr_node = static_cast<uint8_t *>(allocation_buffer) + (static_cast<size_t>(next_ptr) - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
                // if (next_max <= curr_max) { curr_max = next_max; break; } // guard
                // curr_max = next_max;
            }
        }

        // Broadcast node pointer + max
        uintptr_t p = reinterpret_cast<uintptr_t>(curr_node);
        p = tile.shfl(p, 0);
        curr_node = reinterpret_cast<void *>(p);
        curr_max = tile.shfl(curr_max, 0);
        const smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // Must find next node if non empty
        const smallsize next_sister_node_ptr = cg::extract<smallsize>(curr_node, lastpos_off);

        key_type next_sister_node_first_key = next_bucket_first_key; // default if no next node in this chain of nodes
        if (next_sister_node_ptr != 0 && (next_sister_node_ptr - 1) < allocation_count)
        {
            // valid next node
            void *next_sister_node = static_cast<uint8_t *>(allocation_buffer) + (static_cast<size_t>(next_sister_node_ptr) - 1) * node_stride;
            next_sister_node_first_key = extract_key_node<key_type>(next_sister_node, 1);
        }

        // Snapshot per-lane node entry once
        key_type my_key = key_type(0);
        smallsize my_offset = 0;
        bool lane_live = false;
        if (lane < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, lane + 1);
            // my_offset = extract_offset_node<smallsize>(curr_node, lane + 1);
            lane_live = true; // keep simple; add tombstone filter later if needed
        }
        tile.sync();

        // Scan query_list while within this node’s max
        smallsize scan_i = i;
        key_type prev_search_key = 0; // next_search_key;
        key_type out = static_cast<key_type>(not_found);
        //  if (lane ==0) printf(" Before While loop: tile_id=%d, node_max=%llu, curr_size=%d, processing queries from index %d onwards\n",
        //      tile_id, static_cast<unsigned long long>(curr_max), static_cast<int>(curr_size), static_cast<int>(i));

        // if (next_search_key == 1754439130)
        //    printf("Search Key 1754439130 BULK ORDERED: tile_id=%d, query_index=%d, query_key=%llu, result_offset=%d\n",
        //           tile_id, static_cast<int>(scan_i), static_cast<unsigned long long>(next_search_key), static_cast<int>(out));

        while (scan_i < query_size)
        {

            next_search_key = query_list[scan_i];

            if (next_search_key > curr_max)
                break;

            if (prev_search_key != next_search_key)
            {
                /* if (lane == 0 && (next_search_key == 3244786635))
                {
                    printf(" In While Loops TILE BULK ORDERED DUP: tile_id=%d, scan_i=%d, next_search_key=%llu, curr_max=%llu. out=%d\ notfound %d\n",
                           tile_id, static_cast<int>(scan_i), static_cast<unsigned long long>(next_search_key),
                           static_cast<unsigned long long>(curr_max), static_cast<int>(out), not_found);

                    //print_set_nodes_and_links<key_type>(launch_params, tile_id);
                } 
                */
                // Compare this query against all per-lane node entries
                //----const bool eq = lane_live && (my_key == next_search_key);
                const bool eq = lane_live && (my_key == next_search_key);
                bool skip_greater_than_check = false;

                const unsigned mask = tile.ballot(eq);
                if (mask)
                {
                    const smallsize winner = __ffs(mask) - 1; // pick one matching lane
                    // const smallsize result_offset = tile.shfl(my_offset, winner); // fetch its offset
                    // out = result_offset;
                    const key_type result_key = tile.shfl(my_key, winner); // fetch its key
                    out = result_key;
                    skip_greater_than_check = true;
                }

                // const smallsize winner = __ffs(mask) - 1;
                /* if (lane == 0 && tile_id == 0)
                 {
                     printf("Checking successor for key=%llu, next_sister_node_first_key=%llu, tile_id=%d\n",
                            static_cast<unsigned long long>(next_search_key),
                            static_cast<unsigned long long>(next_sister_node_first_key),
                            tile_id);
                             //print_node(<key_type>(curr_node, node_size));
                 } */

                if (!skip_greater_than_check)
                {
                    out = next_sister_node_first_key; // default if not found in current node
                    const bool next_larger = lane_live && (my_key > next_search_key);
                    //----const unsigned mask = tile.ballot(eq);
                    const unsigned mask_next_larger = tile.ballot(next_larger);

                    // if (mask)
                    if (mask_next_larger)
                    {
                        const smallsize winner = __ffs(mask_next_larger) - 1; // pick one matching lane
                        // const smallsize result_offset = tile.shfl(my_offset, winner); // fetch its offset
                        // out = result_offset;
                        const key_type result_key = tile.shfl(my_key, winner); // fetch its key
                        out = result_key;
                        // exception if the search key == bucket_max, we need to check the next bucket first key.
                        // or search key is < bucket_max but it is the last key in the entire bucket... and its successor is in the next bucket
                    }
                }
                // else  // if not mask_next_larger
                //{
                //   out = not_found<key_type>;
                //  }
            }
            // Single writer per query index to avoid races
            if (lane == 0)
            {

                results[scan_i] = out; // one deterministic write
                // if (next_search_key == 1754439130)
                // printf("RESULT BULK ORDERED: tile_id=%d, query_index=%d, query_key=%llu, result_offset=%d\n",
                //     tile_id, static_cast<int>(scan_i), static_cast<unsigned long long>(next_search_key), static_cast<int>(out));
            }
            ++scan_i;

            // tile.sync();
            prev_search_key = next_search_key;
        }

        // Advance to the first query outside this node’s range
        i = scan_i;

        // if (lane ==0) printf(" ADVANCE: tile_id=%d, next query index=%d\n", tile_id, static_cast<int>(i));
        tile.sync();
        // Stop if we stepped beyond this bucket
        if (i >= query_size)
            break;
        // --------next_search_key = query_list[i];
    }

#ifdef PRINT_LOOKUPS_END
    tile.sync(); // only valid if all threads in block reach here
    if (tile_id == 0 && lane == 0)
    {
        printf("END TILE LOOKUPS: PRINT ALL NODES (debug)\n");
        print_set_nodes_and_links<key_type>(launch_params, tile_id);
    }
#endif
}

// template <typename key_type, bool process_single_thread_per_bucket>
template <typename key_type>
GLOBALQUALIFIER void lookup_kernel_successor_tile_ordered(
    updatable_cg_params *__restrict__ launch_params,
    const key_type *__restrict__ query_list, // sorted queries
    key_type *__restrict__ results,          // output
    smallsize query_size)
{
    const key_type *__restrict__ maxbuf =
        static_cast<const key_type *>(launch_params->maxvalues);

    const smallsize partition_count_with_overflow =
        launch_params->partition_count_with_overflow;

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    const int tiles_per_block = blockDim.x / tile.size();
    const int tile_id = blockIdx.x * tiles_per_block + tile.meta_group_rank();
    const smallsize lane = tile.thread_rank();

    const smallsize lastpos_off =
        get_lastposition_bytes<key_type>(launch_params->node_size);

    const smallsize node_stride = launch_params->node_stride;
    const smallsize allocation_count = launch_params->allocation_buffer_count;

    auto *__restrict__ ordered_nodes =
        static_cast<uint8_t *>(launch_params->ordered_node_pairs);

    auto *__restrict__ allocation_buffer =
        static_cast<uint8_t *>(launch_params->allocation_buffer);

    if (tile_id >= partition_count_with_overflow)
        return;

    const key_type maxkey = maxbuf[tile_id];
    const key_type minkey =
        (tile_id > 0)
            ? (maxbuf[tile_id - 1] + static_cast<key_type>(1))
            : static_cast<key_type>(1);

    int minindex = -1;
    int maxindex = -1;

    if (lane == 0)
    {
        minindex = binarySearchIndex_leftmost_ge<key_type>(
            query_list, minkey, 0, query_size);
        // maxindex: last position <= maxkey
        // maxindex = binarySearchIndex_first_gt_leftmost_dup<key_type>(query_list, maxkey, 0, query_size);
    }

    minindex = tile.shfl(minindex, 0);
    const bool empty = (minindex < 0);
    if (tile.any(empty))
        return;

    uint8_t *__restrict__ curr_node =
        ordered_nodes + static_cast<size_t>(node_stride) * static_cast<size_t>(tile_id);

    smallsize next_bucket_id = static_cast<smallsize>(tile_id + 1);
    key_type next_bucket_first_key;

    if (next_bucket_id >= partition_count_with_overflow)
    {
        next_bucket_first_key = static_cast<key_type>(not_found);
    }
    else
    {
        uint8_t *__restrict__ next_bucket_node =
            ordered_nodes + static_cast<size_t>(node_stride) * static_cast<size_t>(next_bucket_id);

        next_bucket_first_key = extract_key_node<key_type>(next_bucket_node, 1);

        while (next_bucket_first_key == 0)
        {
            const smallsize next_bucket_next_ptr =
                cg::extract<smallsize>(next_bucket_node, lastpos_off);

            //if (next_bucket_next_ptr != 0 &&  (next_bucket_next_ptr - 1) < allocation_count)
            if (next_bucket_next_ptr != 0 ) // &&  (next_bucket_next_ptr - 1) < allocation_count)
               
            {
                next_bucket_node =
                    allocation_buffer +
                    (static_cast<size_t>(next_bucket_next_ptr) - 1) * static_cast<size_t>(node_stride);

                next_bucket_first_key = extract_key_node<key_type>(next_bucket_node, 1);
            }
            else
            {
                ++next_bucket_id;
                if (next_bucket_id >= partition_count_with_overflow)
                {
                    next_bucket_first_key = static_cast<key_type>(not_found);
                    break;
                }

                next_bucket_node =
                    ordered_nodes + static_cast<size_t>(node_stride) * static_cast<size_t>(next_bucket_id);

                next_bucket_first_key = extract_key_node<key_type>(next_bucket_node, 1);
            }
        }
    }

    process_lookup_successor_tile_bulk_ordered_dup<key_type>(
        /*bucket_max*/ maxkey,
        /*minindex*/ static_cast<smallsize>(minindex),
        /*maxindex*/ static_cast<smallsize>(maxindex),
        launch_params,
        /*curr_node*/ curr_node,
        /*next_bucket_first_key*/ next_bucket_first_key,
        /*tile*/ tile,
        /*sorted queries*/ query_list,
        /*query_size*/ query_size,
        /*writeback*/ results);
}

// template <typename key_type, bool process_single_thread_per_bucket>
template <typename key_type>
GLOBALQUALIFIER void lookup_kernel_successor_tile_ordered_jan17(
    updatable_cg_params *__restrict__ launch_params,
    const key_type *__restrict__ query_list, // sorted queries
    key_type *__restrict__ results,          // output
    smallsize query_size)
{

    const key_type *__restrict__ maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    const smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    const int tiles_per_block = blockDim.x / tile.size();
    const int tile_id = blockIdx.x * tiles_per_block + tile.meta_group_rank();
    const smallsize lane = tile.thread_rank();
    const smallsize lastpos_off = get_lastposition_bytes<key_type>(launch_params->node_size);

    const smallsize node_stride = launch_params->node_stride;
    smallsize allocation_count = launch_params->allocation_buffer_count;

    auto *__restrict__ ordered_nodes =
        static_cast<uint8_t *>(launch_params->ordered_node_pairs);

    auto *__restrict__ allocation_buffer =
        static_cast<uint8_t *>(launch_params->allocation_buffer); // if exists in params

    // if (lane == 0) printf("TOP LOOKUP TILE BULK ORDERED: tile_id=%d partition_count_with_overflow=%d query_size=%d \n", tile_id, partition_count_with_overflow, query_size);

    if (tile_id >= partition_count_with_overflow)
        return;

    const key_type maxkey = maxbuf[tile_id];
    const key_type minkey = (tile_id > 0) ? (maxbuf[tile_id - 1] + static_cast<key_type>(1)) : static_cast<key_type>(1);

    // Find query subrange for this bucket
    int minindex = -1;
    int maxindex = -1;

    if (lane == 0)
    {

        minindex = binarySearchIndex_leftmost_ge<key_type>(query_list, minkey, 0, query_size);
        // maxindex: last position <= maxkey
        // maxindex = binarySearchIndex_first_gt_leftmost_dup<key_type>(query_list, maxkey, 0, query_size);
    }
    minindex = tile.shfl(minindex, 0);
    const bool empty = (minindex < 0);
    if (tile.any(empty))
        // if (lane ==0) printf("EXITING LOOKUP TILE BULK ORDERED: tile_id=%d, empty=%d minkey=%llu maxkey=%llu , minindex=%d \n", tile_id, empty, static_cast<unsigned long long>(minkey), static_cast<unsigned long long>(maxkey), minindex);
        return;

    // if (empty) return;

    // Node for this bucket
    uint8_t *__restrict__ curr_node =
        static_cast<uint8_t *>(launch_params->ordered_node_pairs) +
        static_cast<size_t>(launch_params->node_stride) * static_cast<size_t>(tile_id);

    smallsize next_bucket_id = tile_id + 1;
    key_type next_bucket_first_key;

    if (next_bucket_id >= partition_count_with_overflow)
    {
        // No next bucket; point to maxkey of self-- because this must be the last bucket
        next_bucket_first_key = static_cast<key_type>(not_found); // currnode max key
    }
    else
    {
        uint8_t *__restrict__ next_bucket_node =
            static_cast<uint8_t *>(launch_params->ordered_node_pairs) +
            static_cast<size_t>(launch_params->node_stride) * static_cast<size_t>(next_bucket_id);

        // next_bucket_first_key = cg::extract<key_type>(next_bucket_first_node, 1 * sizeof(key_type));
        next_bucket_first_key = extract_key_node<key_type>(next_bucket_node, 1);

        while (next_bucket_first_key == 0)
        {
            // NEW ADDED CHECK FOR SUBSEEQUENT NODE in NEXT BUCKET CHAIN
            // next_bucket_node = static_cast<uint8_t *>(allocation_buffer) + (tile_id + 1) * node_stride;
            const smallsize next_bucket_next_ptr = cg::extract<smallsize>(next_bucket_node, lastpos_off);
            if (next_bucket_next_ptr != 0 && (next_bucket_next_ptr - 1) < allocation_count)
            {
                // valid next node
                next_bucket_node = static_cast<uint8_t *>(allocation_buffer) + (static_cast<size_t>(next_bucket_next_ptr) - 1) * node_stride;
                next_bucket_first_key = extract_key_node<key_type>(next_bucket_node, 1);
            }
            else
            {
                // next bucket.
                next_bucket_id++;
                if (next_bucket_id >= partition_count_with_overflow)
                {
                    // No next bucket; point to maxkey of self-- because this must be the last bucket
                    next_bucket_first_key = static_cast<key_type>(not_found);
                    break;
                }
                uint8_t *__restrict__ next_bucket_node = static_cast<uint8_t *>(launch_params->ordered_node_pairs) + static_cast<size_t>(launch_params->node_stride) * static_cast<size_t>(next_bucket_id);
                next_bucket_first_key = extract_key_node<key_type>(next_bucket_node, 1);
            }
        }
    }

    // Perform ordered lookups over [minindex, maxindex] for this bucket
    // Must match your expected signature; additional params appended: query_list, query_size, results

    // if (lane ==0) printf("LOOKUP TILE BULK ORDERED: tile_id=%d, bucket=[%llu..%llu], query_range=[%d..%d]\n",
    //   tile_id, static_cast<unsigned long long>(minkey), static_cast<unsigned long long>(maxkey),
    // minindex, maxindex);

    // if (lane ==0) printf("GOING TO TILE BULK ORDERED: tile_id=%d, bucket=[%llu..%llu], query_range=[%d..%d]\n",
    //  tile_id, static_cast<unsigned long long>(minkey), static_cast<unsigned long long>(maxkey),
    //  minindex, maxindex);

    // if (lane == 0) printf("Going to builk_ordered_dup  tile_id=%d LAUNCHED minindex=%d \n", tile_id, minindex);

    process_lookup_successor_tile_bulk_ordered_dup<key_type>(
        //  process_lookup_successor_tile_bulk_ordered_dup<key_type>(
        // process_lookup_tile_bulk_ordered<key_type>(

        /*bucket_max*/ maxkey,
        /*minindex*/ static_cast<smallsize>(minindex),
        /*maxindex*/ static_cast<smallsize>(maxindex),
        launch_params,
        /*curr_node*/ curr_node, next_bucket_first_key,
        /*tile*/ tile,
        /*sorted queries*/ query_list,
        /*query_size*/ query_size,
        /*writeback*/ results);

    // if (lane ==0) printf("END LOOKUP TILE BULK ORDERED: tile_id=%d DONE\n", tile_id);
}

#endif

/*
****************

USING greater than or equal to

// Assumes: not_found, TILE_SIZE, cg::extract/set, extract_key_node, extract_offset_node, get_lastposition_bytes
template <typename key_type>
DEVICEQUALIFIER void process_lookup_successor_tile_bulk_ordered_dup(
    key_type bucket_max,
    smallsize minindex,
    smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    void *next_bucket_first_node,
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const key_type *__restrict__ query_list,
    smallsize query_size,
    key_type *__restrict__ results)
{
    void *allocation_buffer = launch_params->allocation_buffer;
    const smallsize allocation_count = launch_params->allocation_buffer_count;
    const smallsize node_stride = launch_params->node_stride;
    const smallsize node_size = launch_params->node_size;

    const smallsize lastpos_off = get_lastposition_bytes<key_type>(node_size);
    const smallsize lane = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();
    void *curr_node = starting_node;
    key_type curr_max = key_type(0);

    smallsize i = minindex;
    key_type next_search_key = query_list[i];


    while (i < query_size && next_search_key <= bucket_max)
    {


        const key_type key = next_search_key;

        // if (next_search_key == 1754439130)
           // if (lane ==0) printf("TOP Search Key BULK ORDERED: tile_id=%d, key=%llu \n",
            //       tile_id, static_cast<unsigned long long>(key));

        // Leader walks chain until node's max >= key
        if (lane == 0)
        {
            curr_max = cg::extract<key_type>(curr_node, 0);
            while (curr_max < key)
            {
                const smallsize next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_count)
                {
                    ERROR_INSERTS("ERROR: invalid link in node chain during tile bulk lookup", key, lane, tile_id);
                }

                curr_node = static_cast<uint8_t *>(allocation_buffer) + (static_cast<size_t>(next_ptr) - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
                // if (next_max <= curr_max) { curr_max = next_max; break; } // guard
                // curr_max = next_max;
            }
        }

       // if (next_search_key == 1754439130)
        //    printf("Search Key BULK ORDERED: tile_id=%d, query_key=%llu \n",
          //         tile_id, static_cast<unsigned long long>(next_search_key));

        // Broadcast node pointer + max
        uintptr_t p = reinterpret_cast<uintptr_t>(curr_node);
        p = tile.shfl(p, 0);
        curr_node = reinterpret_cast<void *>(p);
        curr_max = tile.shfl(curr_max, 0);

        // Hard stop if node can’t cover current key or bucket bound exceeded
        // if (curr_max < key || key > bucket_max) break;

        const smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // Snapshot per-lane node entry once
        key_type my_key = key_type(0);
        smallsize my_offset = 0;
        bool lane_live = false;
        if (lane < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, lane + 1);
            my_offset = extract_offset_node<smallsize>(curr_node, lane + 1);
            lane_live = true; // keep simple; add tombstone filter later if needed
        }
        tile.sync();

        // Scan query_list while within this node’s max
        smallsize scan_i = i;
        key_type prev_search_key = 0; // next_search_key;
        key_type out = not_found;
        //  if (lane ==0) printf(" Before While loop: tile_id=%d, node_max=%llu, curr_size=%d, processing queries from index %d onwards\n",
        //      tile_id, static_cast<unsigned long long>(curr_max), static_cast<int>(curr_size), static_cast<int>(i));

        //if (next_search_key == 1754439130)
         //   printf("Search Key 1754439130 BULK ORDERED: tile_id=%d, query_index=%d, query_key=%llu, result_offset=%d\n",
         //          tile_id, static_cast<int>(scan_i), static_cast<unsigned long long>(next_search_key), static_cast<int>(out));

        while (scan_i < query_size)
        {

            next_search_key = query_list[scan_i];


            if (next_search_key > curr_max)
                break;

            if (prev_search_key != next_search_key)
            {
                // Compare this query against all per-lane node entries
                const bool eq = lane_live && (my_key == next_search_key);
                const bool next_larger = lane_live && (my_key > next_search_key);
                const unsigned mask = tile.ballot(eq);
                const unsigned mask_next_larger = tile.ballot(next_larger);

                out = not_found<key_type>();
                if (mask)
                if (mask_next_larger == 0)
                {
                    const smallsize winner = __ffs(mask) - 1;                     // pick one matching lane
                    const smallsize result_offset = tile.shfl(my_offset, winner); // fetch its offset
                    out = result_offset;
                    const result_key = tile.shfl(my_key, winner +1); // fetch its key
                    // exception if the search key == bucket_max, we need to check the next bucket first key.
                    // or search key is < bucket_max but it is the last key in the entire bucket... and its successor is in the next bucket
                    if (next_search_key == bucket_max || (next_search_key < bucket_max && scan_i == maxindex)) {
                        // check next bucket first node
                        key_type next_bucket_first_key = cg::extract<key_type>(next_bucket_first_node, 1 * sizeof(key_type));
                        if (next_search_key == next_bucket_first_key) {

                            // i want the key value as the successor, not the offset, so we have to change results to key_type
                            const key_type next_bucket_first_key = cg::extract<key_type>(next_bucket_first_node, 1 * (sizeof(key_type) + sizeof(smallsize)));
                           // -----smallsize next_bucket_first_offset = cg::extract<smallsize>(next_bucket_first_node, 1 * sizeof(key_type) + sizeof(smallsize));
                           // ---- out = next_bucket_first_offset;
                        }
                }
            }
            // Single writer per query index to avoid races
            if (lane == 0)
            {

                results[scan_i] = out; // one deterministic write
                //if (next_search_key == 1754439130)
                    //printf("RESULT BULK ORDERED: tile_id=%d, query_index=%d, query_key=%llu, result_offset=%d\n",
                      //    tile_id, static_cast<int>(scan_i), static_cast<unsigned long long>(next_search_key), static_cast<int>(out));
            }
            ++scan_i;

            // tile.sync();
            prev_search_key = next_search_key;
        }

        // Advance to the first query outside this node’s range
        i = scan_i;

        // if (lane ==0) printf(" ADVANCE: tile_id=%d, next query index=%d\n", tile_id, static_cast<int>(i));
        tile.sync();
        // Stop if we stepped beyond this bucket
        if (i >= query_size)
            break;
        // --------next_search_key = query_list[i];
    }

#ifdef PRINT_LOOKUPS_END
    tile.sync(); // only valid if all threads in block reach here
    if (tile_id == 0 && lane == 0)
    {
        printf("END TILE LOOKUPS: PRINT ALL NODES (debug)\n");
        print_set_nodes_and_links<key_type>(launch_params, tile_id);
    }
#endif
}
*/

/*  COMPLETE VERSION

template <typename key_type>
DEVICEQUALIFIER void process_lookup_tile_bulk_ordered(
    key_type bucket_max,
    smallsize minindex,
    smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const key_type* __restrict__ query_list,
    smallsize query_size,
    smallsize* __restrict__ results)
{
    void*     allocation_buffer  = launch_params->allocation_buffer;
    smallsize allocation_count   = launch_params->allocation_buffer_count;
    smallsize node_stride        = launch_params->node_stride;
    smallsize node_size          = launch_params->node_size;

    const smallsize lastpos_off  = get_lastposition_bytes<key_type>(node_size);
    const smallsize my_tid       = tile.thread_rank();
    const smallsize tile_id      = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();
    (void)tile_id; // kept for optional debug prints

    void*    curr_node = starting_node;
    key_type curr_max  = key_type{0};

    smallsize i = minindex;
    while (i <= maxindex)
    {
        const key_type key = query_list[i];

        // Leader walks to a node that can cover 'key'
        if (tile.thread_rank() == 0)
        {
            curr_max = cg::extract<key_type>(curr_node, 0);
            while (curr_max < key)
            {
                const smallsize next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_count) {
                    break;
                }
                curr_node = static_cast<uint8_t*>(allocation_buffer) + (static_cast<size_t>(next_ptr) - 1) * node_stride;
                key_type next_max = cg::extract<key_type>(curr_node, 0);
                if (next_max <= curr_max) { curr_max = next_max; break; } // guard
                curr_max = next_max;
            }
        }

        // Broadcast node pointer and max
        uintptr_t p = reinterpret_cast<uintptr_t>(curr_node);
        p        = tile.shfl(p, 0);
        curr_node = reinterpret_cast<void*>(p);
        curr_max  = tile.shfl(curr_max, 0);

        // Stop if we can’t cover the current key or bucket bound exceeded
        if (curr_max < key || key > bucket_max) break;

        const smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // Snapshot per-lane node entry
        key_type  my_key    = key_type(0);
        smallsize my_offset = 0;
        bool      lane_live = false;
        if (my_tid < curr_size) {
            my_key    = extract_key_node<key_type>(curr_node, my_tid + 1);
            my_offset = extract_offset_node<smallsize>(curr_node, my_tid + 1);
            lane_live = true; // keep simple; can re-add zero/tombstone filter later
        }
        tile.sync();

        // Compute subrange [i, j_end) of queries covered by this node
        smallsize scan_i = i;
        while (scan_i <= maxindex && query_list[scan_i] <= curr_max) ++scan_i;
        const smallsize j_end = scan_i; // first index outside this node

        // Prefill results[i..j_end) = not_found (coalesced)
        for (smallsize base = i; base < j_end; base += tile.size()) {
            const smallsize qidx = base + my_tid;
            if (qidx < j_end) results[qidx] = not_found;
        }
        tile.sync();

        // Match-only writes: each lane compares against every query in [i, j_end)
        for (smallsize q = i; q < j_end; ++q) {
            const key_type qkey = query_list[q];
            const bool eq = lane_live && (my_key == qkey);
            if (eq) {
                results[q] = my_offset; // overwrite prefill
            }
        }

        // Advance past this node’s query range
        i = j_end;

        // Stop if next query outside the bucket
        if (i <= maxindex && query_list[i] > bucket_max) break;
    }
}




*/