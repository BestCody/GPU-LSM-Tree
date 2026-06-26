// =============================================================================
// File: coarse_granular_lookups_tile.cuh
// Author: Rosina Kharal
// Description: Implements coarse_granular_lookups_tile
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INDEX_PROTOTYPE_COARSE_GRANULAR_LOOKUPS_TILE_CUH
#define INDEX_PROTOTYPE_COARSE_GRANULAR_LOOKUPS_TILE_CUH

//#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include "coarse_granular_inserts_tiles.cuh"
#include "coarse_granular_inserts.cuh"
#include "definitions_updates.cuh"
#include "tile_utils.cuh"
// namespace coop_g = cooperative_groups;

//namespace coop_g = cooperative_groups;

// New Successor and Range Query operations
template <typename key_type>
DEVICEQUALIFIER void process_lookups_tile_successor(
    key_type /*curr_node_max*/,
    smallsize bucket_index,
    key_type probe_key,
    updatable_cg_params* params,
    void* initial_node,
    key_type /*curr_node_size_unused*/,
    smallsize query_id,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    const auto allocation_buffer = params->allocation_buffer;
    const auto ordered_nodes_base = params->ordered_node_pairs;
    const smallsize node_stride = __ldg(&params->node_stride);
    const smallsize node_size   = __ldg(&params->node_size);
    const smallsize part_cnt_w_overflow = __ldg(&params->partition_count_with_overflow);
    const smallsize alloc_count = __ldg(&params->allocation_buffer_count);
    const smallsize lastpos_bytes = get_lastposition_bytes<key_type>(node_size);
    const smallsize tid = tile.thread_rank();

    if (bucket_index >= part_cnt_w_overflow) { params->result[query_id] = not_found; return; }

    // Start at provided node
    void* curr_node = initial_node;

    while (true) {
        // --- Node-local successor: first key > probe_key ---
        const smallsize curr_sz = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // Each lane i checks its slot
        key_type my_key = 0;
        bool in_range = (tid < curr_sz);
        if (in_range) {
            my_key = extract_key_node<key_type>(curr_node, tid);
        }
        const unsigned mask = tile.ballot(in_range && (my_key > probe_key));

        if (mask) {
            const smallsize hit = __ffs(mask) - 1; // first strictly greater
            smallsize off = extract_offset_node<key_type>(curr_node, hit);
            if (tid == 0) params->result[query_id] = off;
            return;
        }

        // --- No successor in this node: try next node in same bucket ---
        bool has_next = false;
        void* next_node = curr_node;

        if (tid == 0) {
            smallsize last_pos = cg::extract<smallsize>(curr_node, lastpos_bytes);
            // Convention in your code: stored as (next_index + 1). 0 => no next.
            if (last_pos != 0) {
                const smallsize next_index = static_cast<smallsize>(last_pos - 1);
                if (next_index < alloc_count) {
                    next_node = reinterpret_cast<uint8_t*>(allocation_buffer) + (size_t)next_index * node_stride;
                    has_next = true;
                }
            }
        }
        has_next = tile.shfl(has_next, 0);
        uintptr_t nn_raw = tile.shfl(reinterpret_cast<uintptr_t>(next_node), 0);

        if (has_next) {
            curr_node = reinterpret_cast<void*>(nn_raw);
            continue; // keep searching within the same bucket
        }

        // --- No more nodes in this bucket: advance bucket ---
        if (tid == 0) ++bucket_index;
        bucket_index = tile.shfl(bucket_index, 0);

        if (bucket_index >= part_cnt_w_overflow) {
            if (tid == 0) params->result[query_id] = not_found;
            return;
        }

        // Jump to first node of the next bucket
        if (tid == 0) {
            curr_node = reinterpret_cast<uint8_t*>(ordered_nodes_base) + (size_t)bucket_index * node_stride;
        }
        uintptr_t cn_raw = tile.shfl(reinterpret_cast<uintptr_t>(curr_node), 0);
        curr_node = reinterpret_cast<void*>(cn_raw);
        // Loop continues on next bucket
    }
}


template <typename key_type>
DEVICEQUALIFIER void process_lookups_tile_ldg_rq(
    key_type curr_node_max,
    smallsize bucket_index,
    key_type probe_key,
    updatable_cg_params* params,
    void* initial_node,
    key_type /*curr_node_size_unused*/,
    smallsize query_id,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    // --- Read scalar params via __ldg (kept only for POD scalars) ---
    const auto  allocation_buffer        = params->allocation_buffer;
    const auto  ordered_nodes_base       = params->ordered_node_pairs;
    const key_type* __restrict__ maxbuf  = static_cast<const key_type*>(params->maxvalues);
    const key_type* __restrict__ query_upper =
        static_cast<const key_type*>(params->query_upper);

    const smallsize node_stride                 = __ldg(&params->node_stride);
    const smallsize node_size                   = __ldg(&params->node_size);
    const smallsize partition_count             = __ldg(&params->partition_count);
    const smallsize partition_count_with_overflow = __ldg(&params->partition_count_with_overflow);
    const smallsize allocation_buffer_count     = __ldg(&params->allocation_buffer_count);

    // Guard: bucket bounds
    if (bucket_index >= partition_count_with_overflow) {
        params->result[query_id] = not_found;
        return;
    }

    // Range [probe_key, top_range_key]
    const key_type top_range_key = query_upper[query_id];
    if (probe_key > top_range_key) {
        // Empty range, quick return
        params->result[query_id] = 0;
        return;
    }

    // Compute link field byte offset once
    const smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    // Position to the correct starting node (same as point version)
    void* curr_node = initial_node;
    key_type curr_max = cg::extract<key_type>(curr_node, 0);
    smallsize tid = tile.thread_rank();

    if (tid == 0) {
        while (curr_max < probe_key) {
            smallsize last_pos = cg::extract<smallsize>(curr_node, lastposition_bytes) - 1;
            if (last_pos >= allocation_buffer_count) {
                printf("RQ ERROR: next-node out of bounds, bucket=%u last_pos=%u probe=%llu\n",
                       (unsigned)bucket_index, (unsigned)last_pos, (unsigned long long)probe_key);
                params->result[query_id] = not_found;
                return;
            }
            curr_node = reinterpret_cast<uint8_t*>(allocation_buffer) + (size_t)last_pos * node_stride;
            curr_max  = cg::extract<key_type>(curr_node, 0);
        }
    }

    // Broadcast node pointer to the tile
    uintptr_t curr_node_raw = reinterpret_cast<uintptr_t>(curr_node);
    curr_node_raw = tile.shfl(curr_node_raw, 0);
    curr_node = reinterpret_cast<void*>(curr_node_raw);
    curr_max = tile.shfl(curr_max, 0); // optional

    // --- Accumulate over nodes, possibly across buckets ---
    unsigned long long total_sum = 0ULL;

    // Helper lambda: sum [lb, ub) within the current node in parallel
    auto sum_range_in_node = [&](void* node, smallsize lb, smallsize ub_excl) {
        unsigned long long partial = 0ULL;
        for (smallsize i = lb + tid; i < ub_excl; i += TILE_SIZE) {
            // Offset sum as requested (NOT summing keys)
            smallsize off = extract_offset_node<key_type>(node, i);
            partial += static_cast<unsigned long long>(off);
        }
        // Tile reduction
        for (int delta = TILE_SIZE / 2; delta > 0; delta >>= 1) {
            partial += tile.shfl_down(partial, delta);
        }
        if (tid == 0) total_sum += partial;
    };

    // Iterate nodes of the current and subsequent buckets until upper bound is passed
    while (true) {
        // [1] Determine current node size
        smallsize curr_sz = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // [2] Compute lb and ub_exclusive within this node (serial on lane0, node_size ≤ 32)
        smallsize lb = 0, ub_excl = 0;
        if (tid == 0) {
            // Find first idx with key >= probe_key
            smallsize i = 0;
            for (; i < curr_sz; ++i) {
                key_type k = extract_key_node<key_type>(curr_node, i);
                if (k >= probe_key) break;
            }
            lb = i;

            // Find first idx with key > top_range_key
            for (; i < curr_sz; ++i) {
                key_type k = extract_key_node<key_type>(curr_node, i);
                if (k > top_range_key) break;
            }
            ub_excl = i;
        }
        lb      = tile.shfl(lb, 0);
        ub_excl = tile.shfl(ub_excl, 0);

        // [3] Sum offsets in [lb, ub_excl)
        if (lb < ub_excl) {
            sum_range_in_node(curr_node, lb, ub_excl);
        }

        // [4] If this node's max already exceeds the upper bound, we’re done in this bucket
        bool stop_bucket = false;
        if (tid == 0) {
            stop_bucket = (curr_max > top_range_key);
        }
        stop_bucket = tile.shfl(stop_bucket, 0);

        if (!stop_bucket) {
            // Try next node in the same bucket via link
            bool has_next_node = false;
            void* next_node = curr_node;
            key_type next_max = curr_max;

            if (tid == 0) {
                smallsize last_pos = cg::extract<smallsize>(curr_node, lastposition_bytes) - 1;
                if (last_pos < allocation_buffer_count) {
                    next_node = reinterpret_cast<uint8_t*>(allocation_buffer) + (size_t)last_pos * node_stride;
                    next_max  = cg::extract<key_type>(next_node, 0);
                    has_next_node = true;
                }
            }
            has_next_node = tile.shfl(has_next_node, 0);
            uintptr_t next_raw = tile.shfl(reinterpret_cast<uintptr_t>(next_node), 0);
            key_type  next_mx  = tile.shfl(next_max, 0);

            if (has_next_node) {
                curr_node = reinterpret_cast<void*>(next_raw);
                curr_max  = next_mx;
                // continue loop within same bucket
                continue;
            }
        }

        // [5] Need to advance to the next bucket?
        // If upper bound still beyond this bucket’s max, move to bucket_index+1
        bool need_next_bucket = false;
        if (tid == 0) {
            // Guard: if top_range_key > max of this bucket, there might be more to include
            const key_type bucket_max = maxbuf[bucket_index];
            need_next_bucket = (top_range_key > bucket_max) &&
                               (bucket_index + 1 < partition_count_with_overflow);
        }
        need_next_bucket = tile.shfl(need_next_bucket, 0);

        if (!need_next_bucket) {
            break; // Done
        }

        // Advance to next bucket’s first node
        if (tid == 0) {
            ++bucket_index;
        }
        bucket_index = tile.shfl(bucket_index, 0);

        if (bucket_index >= partition_count_with_overflow) break;

        if (tid == 0) {
            curr_node = reinterpret_cast<uint8_t*>(ordered_nodes_base) + (size_t)node_stride * bucket_index;
            curr_max  = cg::extract<key_type>(curr_node, 0);
            // For the next bucket, the lower bound is implicitly 0 in its first node
            // but we still compute lb/ub on that node in the next loop iteration.
        }
        curr_node_raw = tile.shfl(reinterpret_cast<uintptr_t>(curr_node), 0);
        curr_node     = reinterpret_cast<void*>(curr_node_raw);
        curr_max      = tile.shfl(curr_max, 0);
        // loop continues on next bucket
    }

    // Cast/clip to smallsize as required by your result type
    if (tid == 0) {
        // NOTE: if you expect sums to exceed smallsize, change result type accordingly.
        params->result[query_id] = static_cast<smallsize>(total_sum);
    }
}

template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER bool search_sorted_array_with_tile(const key_type *maxvalues, smallsize num_elements, key_type key, smallsize &found_index, coop_g::thread_block_tile<TILE_SIZE> tile)
{
    int thread_id = tile.thread_rank();
    found_index = num_elements; // default to not found

    for (smallsize base = 0; base < num_elements; base += TILE_SIZE)
    {
        smallsize i = base + thread_id;
        bool found = false;

        if (i < num_elements && maxvalues[i] >= key)
        {
            found = true;
        }

        unsigned mask = tile.ballot(found);

        if (mask != 0)
        {
            // Find first thread with match
            int first_active = __ffs(mask) - 1;
            found_index = base + first_active;
            return true;
        }
    }

    return false; // key > all values
}


template <typename key_type>
DEVICEQUALIFIER void process_lookups_tile(
    key_type curr_node_max,
    smallsize bucket_index,
    key_type probe_key,
    updatable_cg_params *params,
    void *initial_node,
    key_type curr_node_size,
    smallsize query_id,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    const auto allocation_buffer = params->allocation_buffer;
    const smallsize node_stride = params->node_stride;
    const smallsize node_size = params->node_size;
    const smallsize partition_size = params->partition_size;
    const key_type *maxbuf = static_cast<const key_type *>(params->maxvalues);
    const smallsize partition_count = params->partition_count;
    const smallsize partition_count_with_overflow = params->partition_count_with_overflow;
    const smallsize allocation_buffer_count = params->allocation_buffer_count;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    void *curr_node = initial_node;
    key_type curr_max = cg::extract<key_type>(curr_node, 0);
    smallsize thread_id = tile.thread_rank();

    //printf("TOP OF PROCESS_LOOKUP_TILE TILE ID id %u, thread id %u, probe_key %u\n", tile.meta_group_rank(), thread_id, bucket_index, probe_key);

    if (thread_id == 0)
    {
        while (curr_max < probe_key)
        {
           // if (probe_key ==3200461485 && tile.meta_group_rank() ==1)  printf("probe_key and currmax tid: probe_key: %u curr_max: %u \n", probe_key, curr_max);
               
            smallsize last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes) - 1;
            if (last_position_value >= allocation_buffer_count)
            {
                printf("Error: last position value out of bounds, tid: %u, probe_key %u lastposition value %u \n", thread_id, probe_key, last_position_value);
                print_node<key_type>(curr_node, node_size);
                return;
            } 
            /*
            if (last_position_value < 0)
            {
                printf("Error: last position value negative lastposition value %d probe_key %u PRINT NODE \n", last_position_value, probe_key);
               // if (thread_id == 0)
               //     print_node<key_type>(curr_node, node_size);
                return;
            }
            */
            curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            curr_max = cg::extract<key_type>(curr_node, 0);

           // if (probe_key ==3200461485 && tile.meta_group_rank() ==1 ) printf("probe_key and NEW currmax tid: probe_key: %u curr_max: %u \n", probe_key, curr_max);
           
        }

       // if (probe_key ==3200461485 && tile.meta_group_rank() ==1 ) printf("Done Loop probe_key and nFINAL currmax tid: probe_key: %u curr_max: %u \n", probe_key, curr_max);
        //printf("Looking for curr node, tile_id thread id %u, probe_key %u\n", tile.meta_group_rank(), thread_id, bucket_index, probe_key);
       // if (thread_id ==0) print_node<key_type>(curr_node, node_size);

    }

    uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
    raw_ptr = tile.shfl(raw_ptr, 0); // Broadcast pointer value from thread 0
    curr_node = reinterpret_cast<void *>(raw_ptr);

    // curr_node = tile.shfl(reinterpret_cast<uintptr_t>(curr_node), 0);
    // curr_node = reinterpret_cast<void *>(curr_node);
    //if(thread_id ==0) printf("Got Curr Node, tile_id %u  thread id %u, probe_key %u\n", tile.meta_group_rank(), thread_id, probe_key);

    smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
    smallsize found_index = 0;

   // if (thread_id == 0)
        DEBUG_LOOKUP_DEV("BEFORE search ", tile.meta_group_rank(), probe_key, curr_size);

    bool found = search_only_cuda_buffer_with_tile<key_type>(tile, curr_node, probe_key, curr_size, found_index);

  //  if (thread_id == 0)
        DEBUG_LOOKUP_DEV("after search ", found, tile.meta_group_rank(), probe_key);
   // if (thread_id == 0)
        DEBUG_LOOKUP_DEV("2nd after search ", probe_key, tile.meta_group_rank(), found_index, query_id);

    // int tid = tile.meta_group_rank() * TILE_SIZE + tile.thread_rank();
    if (!found)
    {
        params->result[query_id] = not_found;
        return;
    }

    smallsize result = extract_offset_node<key_type>(curr_node, found_index);
    params->result[query_id] = result;
}


// file: kernels/lookup_kernel_tile_static_tree.cuh

// NOTE: assumes the following helpers exist and are correct:
// - cg::extract<T>(void* node, size_t byte_offset)
// - extract_key_node<key_type>(const void* node, smallsize index)   // keys start at logical index 1
// - extract_offset_node<key_type>(const void* node, smallsize index) // returns payload/offset for matched key
// - get_lastposition_bytes<key_type>(smallsize node_size)
// - constants: TILE_SIZE, tombstone, not_found
// - types: smallsize, updatable_cg_params, coop_g alias in scope as `coop_g`

template <typename key_type>
DEVICEQUALIFIER void process_lookups_tile_ldg_inlined(
    key_type curr_node_max,
    smallsize bucket_index,
    key_type probe_key,
    updatable_cg_params* __restrict__ params,
    void* initial_node,
    key_type /*curr_node_size*/,
    smallsize query_id,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    // --- Read-only scalars via __ldg (avoid on pointers)
    const smallsize node_stride                     = __ldg(&params->node_stride);
    const smallsize allocation_buffer_count         = __ldg(&params->allocation_buffer_count);

    // Buffers (no __ldg on pointers)
    void*           __restrict__ allocation_buffer  = params->allocation_buffer;

    const smallsize lane = tile.thread_rank();

    // --- Follow 'last position' chain only on lane 0; broadcast final node ptr
    void* curr_node = initial_node;
    if (lane == 0) {
        const smallsize lastpos_bytes = get_lastposition_bytes<key_type>(__ldg(&params->node_size));
        key_type curr_max = cg::extract<key_type>(curr_node, 0);

        // Walk forward while node max < probe_key
        while (curr_max < probe_key) {
            // last_position is 1-based count of allocated nodes; convert to index and bounds-check
            const smallsize last_position_value = cg::extract<smallsize>(curr_node, lastpos_bytes) - 1;
            if (last_position_value >= allocation_buffer_count) {
                // Out-of-bounds => bail out early; mark as not found
                params->result[query_id] = not_found;
                // Broadcast a null ptr so peers don't read garbage
                uintptr_t null_raw = 0;
                uintptr_t raw      = tile.shfl(reinterpret_cast<uintptr_t&>(null_raw), 0);
                (void)raw;
                return;
            }
            curr_node = reinterpret_cast<uint8_t*>(allocation_buffer) + static_cast<size_t>(node_stride) * last_position_value;
            curr_max  = cg::extract<key_type>(curr_node, 0);
        }
    }

    // Broadcast node pointer from lane 0 to the full tile (no shared memory needed)
    uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
    raw_ptr           = tile.shfl(raw_ptr, 0);
    curr_node         = reinterpret_cast<void*>(raw_ptr);

    // If traversal failed above and returned early, we won't get here.
    // Extract current node's logical element count (lane-agnostic)
    const smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

    // --- Inline tile search (replaces search_only_cuda_buffer_with_tile)
    // Each lane probes its own logical slot (1..curr_size)
    const smallsize my_index  = lane;
    key_type        my_key    = key_type{};
    bool            active    = (my_index < curr_size);

    if (active) {
        // Keys are stored starting at logical index 1
        const smallsize logical_index = static_cast<smallsize>(my_index + 1);
        my_key = extract_key_node<key_type>(curr_node, logical_index);

        // Filter out empty/tombstoned slots
        if (my_key == key_type{} || my_key == static_cast<key_type>(tombstone)) {
            active = false;
        }
    }

    // Determine which lanes matched probe_key
    const bool is_match = (active && (my_key == probe_key));

    // Fast-exit: if no lane matches, write not_found and return
    const unsigned match_mask = tile.ballot(is_match);
    if (match_mask == 0u) {
        params->result[query_id] = not_found;
        return;
    }

    // Pick the first match within the tile and broadcast its index
    const int       first_active_lane = __ffs(match_mask) - 1; // ffs is 1-based
    smallsize       match_index_local = 0;
    if (lane == first_active_lane) {
        // +1 because logical indices start at 1 in node layout
        match_index_local = static_cast<smallsize>(lane + 1);
    }
    const smallsize found_index = tile.shfl(match_index_local, first_active_lane);

    // Map logical key index -> payload/offset stored in node
    const smallsize result = extract_offset_node<key_type>(curr_node, found_index);
    params->result[query_id] = result;
}


// Branch prediction hint (CUDA accepts GNU builtins).
#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

// Faster, register-lean version (restores original control flow shape)
template <typename key_type>
DEVICEQUALIFIER void process_lookups_tile_ldg_optimized2(
    key_type /*curr_node_max*/,
    smallsize /*bucket_index*/,
    key_type probe_key,
    updatable_cg_params* __restrict__ params,
    void* initial_node,
    key_type /*curr_node_size*/,
    smallsize query_id,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    const smallsize lane = tile.thread_rank();

    // Only scalars actually used; keep __ldg here.
    const smallsize node_stride             = __ldg(&params->node_stride);
    const smallsize allocation_buffer_count = __ldg(&params->allocation_buffer_count);
    const smallsize node_size_scalar        = __ldg(&params->node_size);

    // Pointers: no __ldg; let compiler choose RO path if possible.
    void* allocation_buffer                 = params->allocation_buffer;

    const smallsize lastposition_bytes      = get_lastposition_bytes<key_type>(node_size_scalar);

    // Lane 0 walks the "last-position" chain; no warp votes on the hot path.
    void*   curr_node = initial_node;
    key_type curr_max;

    if (lane == 0) {
        curr_max = cg::extract<key_type>(curr_node, 0);
        while (curr_max < probe_key) {
            const smallsize last_pos = cg::extract<smallsize>(curr_node, lastposition_bytes) - 1;
            if (UNLIKELY(last_pos >= allocation_buffer_count)) {
                // Coherent early-out: write and return (same semantics as original on error).
                params->result[query_id] = not_found;
                // broadcast a harmless ptr to avoid UB if peers speculatively read (not required but safe)
                uintptr_t dummy = 0;
                (void)tile.shfl(dummy, 0);
                return;
            }
            curr_node = reinterpret_cast<uint8_t*>(allocation_buffer)
                        + static_cast<size_t>(node_stride) * last_pos;
            curr_max  = cg::extract<key_type>(curr_node, 0);
        }
    }

    // Broadcast the node pointer to all lanes; single shuffle like original.
    uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
    raw_ptr           = tile.shfl(raw_ptr, 0);
    curr_node         = reinterpret_cast<void*>(raw_ptr);

    // Read the node’s logical element count; trivial empty fast-exit.
    const smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
    if (UNLIKELY(curr_size == 0)) {
        params->result[query_id] = not_found;
        return;
    }

    // Delegate to the (faster) helper; keep it non-inlined to reduce register pressure here.
    smallsize found_index = 0;
    const bool found = search_only_cuda_buffer_with_tile<key_type>(
        tile, curr_node, probe_key, curr_size, found_index);

    if (UNLIKELY(!found)) {
        params->result[query_id] = not_found;
        return;
    }

    // Map logical key index -> payload/offset.
    const smallsize result = extract_offset_node<key_type>(curr_node, found_index);
    params->result[query_id] = result;
}
// Tile lookup: lean register footprint, tile-safe failure handling.
template <typename key_type>
DEVICEQUALIFIER void process_lookups_tile_ldg_optimized1(
    key_type /*curr_node_max*/,
    smallsize /*bucket_index*/,
    key_type probe_key,
    updatable_cg_params* __restrict__ params,
    void* initial_node,
    key_type /*curr_node_size*/,
    smallsize query_id,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    const smallsize lane = tile.thread_rank();

    // Only scalars we use; keep __ldg here (compiler can map to RO paths).
    const smallsize node_stride             = __ldg(&params->node_stride);
    const smallsize allocation_buffer_count = __ldg(&params->allocation_buffer_count);
    const smallsize node_size_scalar        = __ldg(&params->node_size);

    void* allocation_buffer                 = params->allocation_buffer; // no __ldg on pointers
    const smallsize lastpos_bytes           = get_lastposition_bytes<key_type>(node_size_scalar);

    // Lane-0 walks the chain; broadcast pointer + status (0=ok, 1=OOB/abort).
    void* curr_node = initial_node;
    int   status    = 0;

    if (lane == 0) {
        key_type curr_max = cg::extract<key_type>(curr_node, 0);
        while (curr_max < probe_key) {
            const smallsize last_pos = cg::extract<smallsize>(curr_node, lastpos_bytes) - 1;
            if (last_pos >= allocation_buffer_count) {
                status = 1;            // abort: out-of-bounds
                break;
            }
            curr_node = reinterpret_cast<uint8_t*>(allocation_buffer)
                        + static_cast<size_t>(node_stride) * last_pos;
            curr_max  = cg::extract<key_type>(curr_node, 0);
        }
    }

    // Broadcast results to the tile (avoid ballot cost here).
    uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
    raw_ptr           = tile.shfl(raw_ptr, 0);
    curr_node         = reinterpret_cast<void*>(raw_ptr);

    const int bcast_status = tile.shfl(status, 0);
    if (bcast_status != 0) {
        params->result[query_id] = not_found;
        return;
    }

    // Read current node’s logical size; guard the trivial empty case.
    const smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
    if (curr_size == 0) {
        params->result[query_id] = not_found;
        return;
    }

    // Delegate to faster helper (kept non-inlined to reduce reg pressure here).
    smallsize found_index = 0;
    const bool found = search_only_cuda_buffer_with_tile<key_type>(
        tile, curr_node, probe_key, curr_size, found_index);

    if (!found) {
        params->result[query_id] = not_found;
        return;
    }

    const smallsize result = extract_offset_node<key_type>(curr_node, found_index);
    params->result[query_id] = result;
}

template <typename key_type>
DEVICEQUALIFIER void process_lookups_tile_ldg(
    key_type curr_node_max,
    smallsize bucket_index,
    key_type probe_key,
    updatable_cg_params *params,
    void *initial_node,
    key_type curr_node_size,
    smallsize query_id,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    // Keep __ldg for primitive scalars only, remove for pointers
    const auto allocation_buffer = params->allocation_buffer;
    const smallsize node_stride = __ldg(&params->node_stride);
    const smallsize node_size = __ldg(&params->node_size);
    const smallsize partition_size = __ldg(&params->partition_size);
    const key_type *maxbuf = static_cast<const key_type *>(params->maxvalues);
    const smallsize partition_count = __ldg(&params->partition_count);
    const smallsize partition_count_with_overflow = __ldg(&params->partition_count_with_overflow);
    const smallsize allocation_buffer_count = __ldg(&params->allocation_buffer_count);
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    void *curr_node = initial_node;
    key_type curr_max = cg::extract<key_type>(curr_node, 0);
    smallsize thread_id = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    if (thread_id == 0)
    {
        while (curr_max < probe_key)
        {
            smallsize last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes) - 1;
            if (last_position_value >= allocation_buffer_count)
            {
                printf("Error: last position value out of bounds, tid: %u,tile_id %u, probe_key %u, bucket_index %u, curr_max %u, lastposition value %u \n", thread_id, tile_id, probe_key, bucket_index, curr_max, last_position_value);
                //print_node<key_type>(curr_node, node_size);
                return;
            }
            curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            curr_max = cg::extract<key_type>(curr_node, 0);
        }
    }

    uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
    raw_ptr = tile.shfl(raw_ptr, 0);
    curr_node = reinterpret_cast<void *>(raw_ptr);

    //if(thread_id == 0) printf(" tid: %u, tile_id %u, probe_key %u, bucket_index %u, max_curr_node %u \n", thread_id, tile_id, probe_key, bucket_index, curr_max);


    smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
    smallsize found_index = 0;

    DEBUG_LOOKUP_DEV("BEFORE search ", tile.meta_group_rank(), probe_key, curr_size);

    bool found = search_only_cuda_buffer_with_tile<key_type>(tile, curr_node, probe_key, curr_size, found_index);

    DEBUG_LOOKUP_DEV("after search ", found, tile.meta_group_rank(), probe_key);
    DEBUG_LOOKUP_DEV("2nd after search ", probe_key, tile.meta_group_rank(), found_index, query_id);

    if (!found)
    {
        params->result[query_id] = not_found;
        return;
    }

    smallsize result = extract_offset_node<key_type>(curr_node, found_index);
    params->result[query_id] = result;
}

// file: kernels/lookup_kernel_tile_static_tree.cuh


template <typename key_type, int16_t node_size_log, uint16_t cg_size_log, bool caching_enabled>
GLOBALQUALIFIER void lookup_kernel_tile_static_tree(updatable_cg_params* __restrict__ launch_params,
                                                    bool perform_range_query_lookup)
{
    namespace cg = coop_g;
/*
    cg::thread_block block                = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    const int       global_thread_id      = blockIdx.x * blockDim.x + threadIdx.x;
    const smallsize lane                  = tile.thread_rank();

    // Read-only inputs (encourage RO caching)
    const smallsize query_size            = launch_params->query_size;
    const key_type* __restrict__ query   = static_cast<const key_type*>(launch_params->query_lower);
    const key_type* __restrict__ tree    = static_cast<const key_type*>(launch_params->tree_buffer);
    const auto      metadata_out          = launch_params->metadata;
    const smallsize num_buckets_with_of   = launch_params->partition_count_with_overflow;
    const smallsize node_stride           = launch_params->node_stride;

    // Mutables
    uint8_t* __restrict__ node_buf        = static_cast<uint8_t*>(launch_params->ordered_node_pairs);

    // No shared-tree caching variant in this path
    constexpr bool use_row_layout = false;
    const key_type* shmem_tree    = nullptr;
    const smallsize cached_entries_count = 0;

    // Compute tile-aligned base id; ensures coalesced key loads
    const int base_idx = global_thread_id - static_cast<int>(lane);

    // Each lane pulls its candidate key (coalesced), tracks validity
    key_type my_key     = key_type{};
    smallsize my_qindex = 0;
    bool in_range       = false;

    if (base_idx >= static_cast<int>(query_size)) {
    // All lanes in this tile are beyond the query range
        return;
    }
    if (base_idx >= 0) {
        const int qidx = base_idx + lane;
        in_range       = (qidx < static_cast<int>(query_size));
        if (in_range) {
            my_qindex = static_cast<smallsize>(qidx);
            // Modern CC: const global -> RO path; __ldg unnecessary on recent arch, left to compiler
            my_key    = query[qidx];
        }
    }

    // Tile work-queue: process one key per iteration cooperatively
    unsigned mask = tile.ballot(in_range);
    while (mask) {
        const int leader = __ffs(mask) - 1;

        // Broadcast the leader's query to the whole tile
        const key_type  probe_key   = tile.shfl(my_key, leader);
        const smallsize query_index = tile.shfl(my_qindex, leader);

        // Cooperative static-tree routing
        const smallsize found_index = static_tree_search<key_type, node_size_log, cg_size_log, use_row_layout>(
            tile, probe_key, tree, metadata_out, shmem_tree, cached_entries_count);

        // Guard: skip invalid bucket without killing the tile
        if (found_index < num_buckets_with_of) {
            uint8_t* __restrict__ curr_node = node_buf + static_cast<size_t>(node_stride) * found_index;

            // Local placeholders (kept for API parity)
            smallsize currnodesize = 0;
            key_type  currnodeMax  = 0;

            if (perform_range_query_lookup) {
                // NOTE: enable when range-query path is implemented for static-tree routing
                // process_lookups_tile_rq<key_type>(currnodeMax, found_index, probe_key,
                //     launch_params, curr_node, currnodesize, query_index, tile);
            } else {
                process_lookups_tile_ldg<key_type>(currnodeMax, found_index, probe_key,
                    launch_params, curr_node, currnodesize, query_index, tile);
            }
        }

        // Only the leader lane marks its work as done
        if (lane == leader) in_range = false;

        // Recompute remaining work
        mask = tile.ballot(in_range);
    }

#ifdef PRINT_LOOKUPS_END
    __syncthreads();
    print_set_nodes_and_links<key_type>(launch_params, global_thread_id);  // diagnostics only
#endif

*/
}


// file: kernels/lookup_kernel_tile_static_tree.cuh

// 3) Kernel: consume precomputed bucket indices
template <typename key_type, int16_t node_size_log, uint16_t cg_size_log, bool caching_enabled>
GLOBALQUALIFIER void lookup_kernel_tile_static_tree_clean(updatable_cg_params* launch_params, bool perform_range_query_lookup)
{

    /*
    namespace cg = coop_g;

    cg::thread_block block                 = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile  = cg::tiled_partition<TILE_SIZE>(block);

    const int       global_thread_id       = blockIdx.x * blockDim.x + threadIdx.x;
    const smallsize tid                    = tile.thread_rank();
    smallsize       tile_id                = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    const smallsize query_size             = launch_params->query_size;
    smallsize       total_tiles_required   = SDIV(query_size, TILE_SIZE);

    if (global_thread_id >= total_tiles_required * TILE_SIZE) return;

    auto*          node_buf                       = static_cast<uint8_t*>(launch_params->ordered_node_pairs);
    const key_type* query_list                    = static_cast<const key_type*>(launch_params->query_lower);
    const smallsize num_partitions_with_overflow  = launch_params->partition_count_with_overflow;
    const smallsize node_stride                   = launch_params->node_stride;

    const smallsize cached_entries_count          = 0; // no cache in this path
    smallsize       tree_entries_count            = launch_params->tree_entries_count; // kept for parity if needed downstream
    const auto      metadata_out                  = launch_params->metadata;

    const key_type* shmem_tree                    = nullptr; // no dynamic smem tree in this variant
    const key_type* tree_buf                      = static_cast<const key_type*>(launch_params->tree_buffer); // callee expects key_type*

    key_type  probe_key    = 0;
    smallsize query_index  = 0;

    // Each tile iterates over TILE_SIZE broadcast steps
    for (int i = 0; i < TILE_SIZE; ++i) {
        if (tid == i) {
            query_index = static_cast<smallsize>(global_thread_id);
            probe_key   = query_list[global_thread_id];
        }
        probe_key   = tile.shfl(probe_key, i);
        query_index = tile.shfl(query_index, i);

        // Route via static tree
        smallsize found_index = static_tree_search<key_type, node_size_log, cg_size_log, false>(
            tile, probe_key, tree_buf, metadata_out, shmem_tree, cached_entries_count);

        const smallsize bucket_index = found_index;
        if (bucket_index >= num_partitions_with_overflow) return; // early-out if routed to overflow

        auto*     curr_node     = node_buf + static_cast<size_t>(node_stride) * bucket_index;
        smallsize currnodesize  = 0;
        key_type  currnodeMax   = 0;

        if (perform_range_query_lookup) {
            // process_lookups_tile_rq<key_type>(currnodeMax, bucket_index, probe_key, launch_params, curr_node, currnodesize, query_index, tile);
            // Intentional no-op: range-query path disabled in this build.
        } else {
            process_lookups_tile_ldg<key_type>(currnodeMax, bucket_index, probe_key, launch_params, curr_node, currnodesize, query_index, tile);
        }
    }

#ifdef PRINT_LOOKUPS_END
    __syncthreads();
    print_set_nodes_and_links<key_type>(launch_params, global_thread_id); // only if diagnostic flag enabled
#endif

    */
}


// file: kernels/lookup_kernel_tile_static_tree.cuh

// file: kernels/lookup_kernel_tile_static_tree.cuh

// -----------------------------------------------------------------------------
// Helper: leader-only search; NO global writes; returns result via out_result.
// -----------------------------------------------------------------------------
template <typename key_type>
DEVICEQUALIFIER bool process_lookups_tile_ldg_results(
    key_type /*curr_node_max*/,
    smallsize /*bucket_index*/,
    key_type probe_key,
    updatable_cg_params* __restrict__ params,
    void* initial_node,
    key_type /*curr_node_size*/,
    smallsize /*query_id*/, // kept for parity; not used inside (no global writes here)
    coop_g::thread_block_tile<TILE_SIZE> tile,
    smallsize& out_result)  // <-- leader's result (valid only on leader)
{
    // Only needed scalars; never __ldg on pointers
    void*        allocation_buffer        = params->allocation_buffer;
    const smallsize node_stride           = __ldg(&params->node_stride);
    const smallsize node_size_scalar      = __ldg(&params->node_size);
    const smallsize allocation_buf_count  = __ldg(&params->allocation_buffer_count);

    const smallsize lastpos_bytes         = get_lastposition_bytes<key_type>(node_size_scalar);

    // Lane 0 traverses; broadcast node pointer
    const smallsize lane = tile.thread_rank();

    void*     curr_node = initial_node;
    key_type  curr_max  = cg::extract<key_type>(curr_node, 0);

    if (lane == 0) {
        while (curr_max < probe_key) {
            const smallsize last_pos = cg::extract<smallsize>(curr_node, lastpos_bytes) - 1;
            if (last_pos >= allocation_buf_count) {
                out_result = not_found;  // report failure via leader's register
                return false;
            }
            curr_node = reinterpret_cast<uint8_t*>(allocation_buffer)
                        + static_cast<size_t>(node_stride) * last_pos;
            curr_max  = cg::extract<key_type>(curr_node, 0);
        }
    }

    // Broadcast ptr
    uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
    raw_ptr           = tile.shfl(raw_ptr, 0);
    curr_node         = reinterpret_cast<void*>(raw_ptr);

    // Search inside node (tile-cooperative); leader consumes the answer
    const smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
    smallsize found_index     = 0;

    const bool found = (curr_size != 0) &&
        search_only_cuda_buffer_with_tile<key_type>(tile, curr_node, probe_key, curr_size, found_index);

    if (!found) {
        out_result = not_found;
        return false;
    }

    out_result = extract_offset_node<key_type>(curr_node, found_index);
    return true;
}
#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

//------------------------------------------------------------------------------
// Core kernel (compile-time switch for range-query path)
//------------------------------------------------------------------------------
template <bool DoRangeQuery,
          typename key_type, int16_t node_size_log, uint16_t cg_size_log, bool caching_enabled>
GLOBALQUALIFIER void lookup_kernel_tile_static_tree_params_core(
    updatable_cg_params* __restrict__ launch_params,
    const key_type* __restrict__ query_list,
    smallsize* __restrict__ results,
    smallsize query_size)
{
   /* namespace cg = coop_g;

    cg::thread_block block                = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    const int       global_thread_id      = blockIdx.x * blockDim.x + threadIdx.x;
    const smallsize lane                  = tile.thread_rank();

    // Routing data (const/read-mostly)
    const key_type* __restrict__ tree     = static_cast<const key_type*>(launch_params->tree_buffer);
    const auto      metadata_out          = launch_params->metadata;
    const smallsize num_buckets_with_of   = launch_params->partition_count_with_overflow;
    const size_t    node_stride_bytes     = static_cast<size_t>(launch_params->node_stride);
    uint8_t* __restrict__ node_buf        = static_cast<uint8_t*>(launch_params->ordered_node_pairs);

    constexpr bool use_row_layout = false;
    const key_type* shmem_tree    = nullptr;
    const smallsize cached_entries_count = 0;

    // Tile-aligned base index for coalesced loads
    const int base_idx = global_thread_id - static_cast<int>(lane);

    key_type  my_key    = key_type{};
    smallsize my_qindex = 0;
    bool      in_range  = false;

    if (base_idx >= static_cast<int>(query_size)) return;

    if (base_idx >= 0) {
        const int qidx = base_idx + lane;
        in_range       = (qidx < static_cast<int>(query_size));
        if (in_range) {
            my_qindex = static_cast<smallsize>(qidx);
            my_key    = query_list[qidx];
        }
    }

    const bool had_work = in_range;
    smallsize  my_out   = not_found;

    // Initial work mask from active lanes
    unsigned mask = tile.ballot(in_range);

    // Process active lanes; avoid re-balloting each iteration
    while (mask) {
        const int leader = __ffs(mask) - 1;      // lane id of next work item

        // Broadcast leader's inputs
        const key_type  probe_key   = tile.shfl(my_key,    leader);
        const smallsize query_index = tile.shfl(my_qindex, leader);

        // Cooperative static-tree routing
        const smallsize bucket_idx = static_tree_search<key_type, node_size_log, cg_size_log, use_row_layout>(
            tile, probe_key, tree, metadata_out, shmem_tree, cached_entries_count);

        if (LIKELY(bucket_idx < num_buckets_with_of)) {
            uint8_t* __restrict__ curr_node =
                node_buf + node_stride_bytes * static_cast<size_t>(bucket_idx);

            if constexpr (DoRangeQuery) {
                // TODO: range-query handler (accumulate to leader then write later)
                if (lane == leader) my_out = not_found;
            } else {
                // Point lookup path: leader-only local result, no global write here
                smallsize leader_result = not_found;
                bool ok = process_lookups_tile_ldg_results<key_type>(
                    key_type{}, bucket_idx, probe_key,
                    launch_params, curr_node,  key_type{},
                     query_index, tile, leader_result);

                if (lane == leader) {
                    my_out = ok ? leader_result : not_found;
                }
            }
        } else {
            if (lane == leader) my_out = not_found;
        }

        // Clear the leader bit locally: mask &= (mask - 1)
        mask &= (mask - 1);
    }

    // Coalesced writeback (each lane that had work writes its element)
    if (had_work) {
        results[my_qindex] = my_out;
    }

#ifdef PRINT_LOOKUPS_END
    __syncthreads();
    print_set_nodes_and_links<key_type>(launch_params, global_thread_id);
#endif

*/
}


// -----------------------------------------------------------------------------
// Kernel: accumulates per-lane results in registers; coalesced write at the end
// -----------------------------------------------------------------------------
template <typename key_type, int16_t node_size_log, uint16_t cg_size_log, bool caching_enabled>
GLOBALQUALIFIER void lookup_kernel_tile_static_tree_params(
    updatable_cg_params* __restrict__ launch_params,
   // -- not used for now bool perform_range_query_lookup,
    const key_type* __restrict__ query_list,
    smallsize* __restrict__ results,
    smallsize query_size)
{

/*
    namespace cg = coop_g;

    cg::thread_block block                = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    const int       global_thread_id      = blockIdx.x * blockDim.x + threadIdx.x;
    const smallsize lane                  = tile.thread_rank();

    // Routing data from params
    const key_type* __restrict__ tree     = static_cast<const key_type*>(launch_params->tree_buffer);
    const auto      metadata_out          = launch_params->metadata;
    const smallsize num_buckets_with_of   = launch_params->partition_count_with_overflow;
    const smallsize node_stride           = launch_params->node_stride;
    uint8_t* __restrict__ node_buf        = static_cast<uint8_t*>(launch_params->ordered_node_pairs);

    constexpr bool use_row_layout = false;
    const key_type* shmem_tree    = nullptr;
    const smallsize cached_entries_count = 0;

    // Tile-aligned base index for coalesced loads
    const int base_idx = global_thread_id - static_cast<int>(lane);

    key_type  my_key    = key_type{};
    smallsize my_qindex = 0;
    bool      in_range  = false;

    if (base_idx >= static_cast<int>(query_size)) return;

    if (base_idx >= 0) {
        const int qidx = base_idx + lane;
        in_range       = (qidx < static_cast<int>(query_size));
        if (in_range) {
            my_qindex = static_cast<smallsize>(qidx);
            my_key    = query_list[qidx];
        }
    }

    // Keep original "had work" for final coalesced write
    const bool had_work = in_range;

    // Per-lane result kept in register; initialized to not_found
    smallsize my_out = not_found;

    // Tile work-queue
    unsigned mask = tile.ballot(in_range);
    while (mask) {
        const int leader = __ffs(mask) - 1;

        const key_type  probe_key   = tile.shfl(my_key,    leader);
        const smallsize query_index = tile.shfl(my_qindex, leader);

        // Route via static tree cooperatively
        const smallsize found_index = static_tree_search<key_type, node_size_log, cg_size_log, use_row_layout>(
            tile, probe_key, tree, metadata_out, shmem_tree, cached_entries_count);

        if (found_index < num_buckets_with_of) {
            uint8_t* __restrict__ curr_node = node_buf + static_cast<size_t>(node_stride) * found_index;

            // Only the leader consumes the produced result in its register
           //---> bring back for RQ if (!perform_range_query_lookup) {
                smallsize leader_result = not_found;
                bool ok = process_lookups_tile_ldg_results<key_type>(
                    key_type{}, found_index, probe_key,
                    launch_params, curr_node, key_type{},
                    query_index, tile, leader_result);

                if (lane == leader) {
                    my_out = ok ? leader_result : not_found;
                }
           // } else {
                // Range-query path would set leader_result similarly (omitted here)
             //   if (lane == leader) my_out = not_found;
            //}
         } else {
            if (lane == leader) my_out = not_found;
        }

        // Mark served lane done
        if (lane == leader) in_range = false;
        mask = tile.ballot(in_range);
    }

    // Final coalesced write: lanes write their own contiguous results
    if (had_work) {
        results[my_qindex] = my_out;
    }

#ifdef PRINT_LOOKUPS_END
    __syncthreads();
    print_set_nodes_and_links<key_type>(launch_params, global_thread_id);
#endif

*/
}


// ============================================================================
// 3) Kernel: consume precomputed bucket indices
template <typename key_type, int16_t node_size_log, uint16_t cg_size_log, bool caching_enabled>
GLOBALQUALIFIER void lookup_kernel_tile_static_tree_debug(updatable_cg_params* launch_params, bool perform_range_query_lookup)
{

/*
    namespace cg = coop_g;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const smallsize tid = tile.thread_rank();   
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    //if (tile_id ==0 && tid==0) printf("TOP Lookup Static Tree:global thread id %d \n", global_thread_id);
    const smallsize query_size  = launch_params->query_size;
    smallsize total_tiles_required = SDIV(query_size, TILE_SIZE);

    //if (global_thread_id >= query_size) return;
    if (global_thread_id >= total_tiles_required * TILE_SIZE) return;

    auto* node_buf = static_cast<uint8_t*>(launch_params->ordered_node_pairs);
   // auto* tree_buf = static_cast<key_type*>(launch_params->tree_buffer);
    const key_type* query_list = static_cast<const key_type*>(launch_params->query_lower);

   // const smallsize* bucket_indices = launch_params->bucket_indices;   // <-- NEW

    const smallsize num_partitions_with_overflow = launch_params->partition_count_with_overflow;
    const smallsize node_stride = launch_params->node_stride;
    const smallsize node_size   = launch_params->node_size;
    
   
    const smallsize cached_entries_count = 0; // 
    smallsize tree_entries_count = launch_params->tree_entries_count;
    const auto metadata_out  = launch_params->metadata;
    // extern __shared__ uint8_t shmem_tree[]; // Dynamic shared memory for tree
    const key_type* shmem_tree =nullptr;    //reinterpret_cast<const key_type*>(shmem_tree_bytes);
    const key_type* tree_buf   = (const key_type*)launch_params->tree_buffer; // why: callee expects key_type*
*/
    
    /*
// print contents of metadata here
    // print contents of metadata here
   if (tile_id == 0 && tid == 0) {
    std::printf("Static tree metadata:\n");
    std::printf("  level_count: %u\n", static_cast<unsigned>(metadata_out.level_count));
    std::printf("  total_nodes: %u\n", static_cast<unsigned>(metadata_out.total_nodes));
    std::printf("  nodes_on_level: [");
    for (unsigned i = 0; i < static_cast<unsigned>(metadata_out.level_count); ++i) {
        std::printf("%u%s", static_cast<unsigned>(metadata_out.nodes_on_level[i]),
                    (i + 1 == static_cast<unsigned>(metadata_out.level_count)) ? "" : ", ");
    }
    std::printf("]\n");
   }


if (tile_id == 0 && tid == 0) {
    printf("[tree dump] ptr=%p, count=%u\n",
           (const void*)tree_buf, (unsigned)tree_entries_count);

    if (!tree_buf) {
        printf("[tree dump] ERROR: tree_buf is NULL\n");
    } else {
        // WARNING: this can be very large; uncomment the clamp if needed
        // const unsigned max_dump = 1024u;
        // const unsigned dump_n   = min((unsigned)cached_entries_count, max_dump);
        const unsigned dump_n = (unsigned)tree_entries_count;

        for (unsigned i = 0; i < dump_n; ++i) {
            // Works for 32/64-bit key_type
            unsigned long long v = (unsigned long long)tree_buf[i];
            printf("tree[%u] = %llu\n", i, v);
        }

        // if (dump_n < (unsigned)cached_entries_count) {
        //     printf("[tree dump] ... truncated at %u of %u\n",
        //            dump_n, (unsigned)cached_entries_count);
        // }
    }
}
__syncthreads();    
*/

/*

    key_type  probe_key = 0;
    smallsize query_index = 0;

   // if (tid==0) printf("START Lookup Static Tree: processing query size %d \n", query_size);

    // Each tile iterates over its TILE_SIZE local-broadcast steps
    for (int i = 0; i < TILE_SIZE; ++i) {

        if (tid == i) {
            query_index = static_cast<smallsize>(global_thread_id);
            probe_key   = query_list[global_thread_id];
        }
        probe_key   = tile.shfl(probe_key, i);
        query_index = tile.shfl(query_index, i);

        // ---- ROUTING via static tree result (no device binary search) ----
        smallsize found_index = 0;
        //if (tid == 0) {
        //---  found_index = bucket_indices[query_index];  // <-- uses precomputed routing
       // if (tid==0) printf("Before BUCKET Search for thread id %d and global thread id %u, probe key is %u node_size_log %u, cg_size_log %u\n", tid, global_thread_id, probe_key, node_size_log, cg_size_log);
       
        //found_index = static_tree_find_single<key_type, node_size_log, cg_size_log, use_row_layout>(tile, cur_key, offset, sorted_entries);
        found_index = static_tree_search<key_type, node_size_log, cg_size_log, false >(
                            tile, probe_key, tree_buf, metadata_out, shmem_tree, cached_entries_count);
        //}
        //found_index = tile.shfl(found_index, 0);
       // if (tid==0) printf("After BUCKET index for thread id %d and global thread id %u, tile id %u,  bucket index is %u  and probe key is %u \n", tid, global_thread_id, tile_id, found_index, probe_key);
        
        const smallsize bucket_index = found_index;
        if (bucket_index >= num_partitions_with_overflow) return;

        auto* curr_node = node_buf + static_cast<size_t>(node_stride) * bucket_index;

        smallsize currnodesize = 0;  // if needed, load alongside curr node header here
        key_type  currnodeMax  = 0;  // if needed

        if (perform_range_query_lookup) {
            //process_lookups_tile_rq<key_type>(currnodeMax, found_index, probe_key, launch_params, curr_node, currnodesize, query_index, tile);
        } else {
            process_lookups_tile_ldg<key_type>(currnodeMax, bucket_index, probe_key, launch_params, curr_node, currnodesize, query_index, tile);
        }
    }

 //print_set_nodes_and_links<key_type>(launch_params, global_thread_id);
#ifdef PRINT_LOOKUPS_END
    __syncthreads();
    //if (tile.meta_group_rank() == 1 && tid == 0) {
        print_set_nodes_and_links<key_type>(launch_params, global_thread_id);
   // }
#endif

*/


}


template <typename key_type>
GLOBALQUALIFIER void lookup_kernel_tile_loop(updatable_cg_params *launch_params, bool perform_group_lookup)
{

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    smallsize thread_id = tile.thread_rank();
    smallsize query_size = launch_params->query_size;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    key_type probe_key = 0;
    smallsize query_index = 0;

    if (global_thread_id >= query_size)
        return; // Ensure we don't go out of bounds

    auto buf = launch_params->ordered_node_pairs;
    const key_type *maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    smallsize num_partitions_with_overflow = launch_params->partition_count_with_overflow;
    const key_type *query_list = static_cast<const key_type *>(launch_params->query_lower);


    // HERE 
    // COPY to local Thread shared array.. copy all probe keys here first
    // Each thread will handle its own probe key
    for (int i = 0; i < TILE_SIZE; i++)
    {
        // int query_index = global_thread_id + i * blockDim.x;
        if (thread_id == i)
        {
            query_index = global_thread_id;
            probe_key = query_list[global_thread_id];
        }

        // braodcast the probe key to all threads in the tile
        probe_key = tile.shfl(probe_key, i);
        query_index = tile.shfl(query_index, i);

        // tile.sync();
        // printf("MAIN LOOP this tile id %u, thread id %u, probe_key %u, query_index %u\n", tile.meta_group_rank(), thread_id, probe_key, query_index);

        smallsize found_index = 0;
        // bool probe_key_found = search_sorted_array_with_tile<key_type>(maxbuf, num_partitions_with_overflow, probe_key, found_index, tile);
        if (thread_id == 0)
        {
            bool probe_key_found = binary_search_equal_or_greater<key_type>(maxbuf, num_partitions_with_overflow, probe_key, found_index, thread_id);
            // probe_key_found = binary_search_in_array<key_type>(maxbuf, maxbufsize, probe_key, found_index, tid);
        }
        //tile.sync();

        found_index = tile.shfl(found_index, 0);

        //tile.sync();
        smallsize bucket_index = found_index; // is the maxval of the bucket where this key belongs... so it is the i'th bucket

        if (bucket_index >= num_partitions_with_overflow)
        {
            printf("Error: bucket index out of bounds, tid: %d\n", thread_id);
            return;
        }

      
        auto curr_node = reinterpret_cast<uint8_t *>(buf) + launch_params->node_stride* bucket_index;

       // smallsize currnodesize = cg::extract<smallsize>(curr_node, sizeof(key_type));
       // key_type currnodeMax = cg::extract<key_type>(curr_node, 0);
       
        smallsize currnodesize = 0;
        key_type currnodeMax = 0;
       
    
           // DEBUG_LOOKUP_DEV("Before Calling ProcessLookupTiles this tile id", tile.meta_group_rank(), thread_id, probe_key);
           // DEBUG_LOOKUP_DEV("2nd Before Calling ProcessLookupTiles", query_index, currnodeMax, bucket_index);
            // smallsize node_size = launch_params->node_size;
            // print_node<key_type>(curr_node, node_size);
       

        if (perform_group_lookup)
        {
            printf("Performing Group Lookup with TILE at global_thread_id %d\n", global_thread_id);
        }
        else
        {
            process_lookups_tile_ldg<key_type>(currnodeMax, found_index, probe_key, launch_params, curr_node, currnodesize, query_index, tile);
        }
    }

#ifdef PRINT_LOOKUPS_END
    __syncthreads();
    if (tile.meta_group_rank() == 0 && thread_id == 0)
    {
        printf("END LOOKUPS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, global_thread_id);
    }

#endif
}


template <typename key_type>
GLOBALQUALIFIER void lookup_kernel_tile_loop_SM(updatable_cg_params *launch_params, bool perform_group_lookup)
{
    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    smallsize thread_id = tile.thread_rank();
    smallsize query_size = launch_params->query_size;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    key_type my_tile_probe_keys[TILE_SIZE];
    smallsize my_tile_query_indices[TILE_SIZE];

    auto buf = launch_params->ordered_node_pairs;
    const key_type *maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    smallsize num_partitions_with_overflow = launch_params->partition_count_with_overflow;
    const key_type *query_list = static_cast<const key_type *>(launch_params->query_lower);

    if (global_thread_id >= query_size)
        return; // Ensure we don't go out of bounds

    int tile_start_index = (blockIdx.x * (blockDim.x / TILE_SIZE) + tile.meta_group_rank()) * TILE_SIZE;

    for (int i = 0; i < TILE_SIZE; i++) {
        int load_index = tile_start_index + i;
        if (load_index < query_size) {
            my_tile_probe_keys[i] = query_list[load_index];
            my_tile_query_indices[i] = load_index;
        } else {
            my_tile_probe_keys[i] = 0;
            my_tile_query_indices[i] = static_cast<smallsize>(-1);
        }
    }

    tile.sync();

    for (int i = 0; i < TILE_SIZE; i++) {
        key_type probe_key = my_tile_probe_keys[i];
        smallsize query_index = my_tile_query_indices[i];

        if (query_index == static_cast<smallsize>(-1)) continue;

       // printf("MAIN LOOP this tile id %u, thread id %u, probe_key %u, query_index %u\n", tile.meta_group_rank(), thread_id, probe_key, query_index);

        smallsize found_index = 0;
        if (thread_id == 0) {
            bool probe_key_found = binary_search_equal_or_greater<key_type>(maxbuf, num_partitions_with_overflow, probe_key, found_index, thread_id);
        }
        found_index = tile.shfl(found_index, 0);

        smallsize bucket_index = found_index;
        if (bucket_index >= num_partitions_with_overflow) {
            if (thread_id == 0)
                printf("Error: bucket index out of bounds, tid: %d\n", thread_id);
            continue;
        }

        auto curr_node = reinterpret_cast<uint8_t *>(buf) + launch_params->node_stride * bucket_index;

        smallsize currnodesize = cg::extract<smallsize>(curr_node, sizeof(key_type));
        key_type currnodeMax = cg::extract<key_type>(curr_node, 0);

       //printf("Looking for curr node, tile_id %u  thread id %u, probe_key %u bucket_index %u \n", tile.meta_group_rank(), thread_id, probe_key, bucket_index);

        if (perform_group_lookup) {
            printf("Performing Group Lookup with TILE at global_thread_id %d\n", global_thread_id);
        } else {
            process_lookups_tile<key_type>(currnodeMax, found_index, probe_key, launch_params, curr_node, node_size, query_index, tile);
        }
    }

#ifdef PRINT_LOOKUPS_END
    __syncthreads();
    if (tile.meta_group_rank() == 0 && thread_id == 0)
    {
        printf("END LOOKUPS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, global_thread_id);
    }
#endif
}

template <typename key_type>
GLOBALQUALIFIER void lookup_kernel_tile_loop_sharedMem(updatable_cg_params *launch_params, bool perform_group_lookup)
{

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    smallsize thread_id = tile.thread_rank();
    smallsize query_size = launch_params->query_size;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    key_type probe_key = 0;
    smallsize query_index = 0;

    if (global_thread_id >= query_size)
        return; // Ensure we don't go out of bounds

    auto buf = launch_params->ordered_node_pairs;
    const key_type *maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    smallsize num_partitions_with_overflow = launch_params->partition_count_with_overflow;
    const key_type *query_list = static_cast<const key_type *>(launch_params->query_lower);


    //HERE 
    // COPY to local Thread shared array.. copy all probe keys here first

    __shared__ key_type tile_probe_keys[TILE_SIZE];
    __shared__ smallsize tile_query_indices[TILE_SIZE];
    //if (global_thread_id < query_size) {
        tile_probe_keys[thread_id] = query_list[global_thread_id];
        tile_query_indices[thread_id] = global_thread_id;

        tile.sync();

    // Each thread will handle its own probe key
    for (int i = 0; i < TILE_SIZE; i++)
    {

        // int query_index = global_thread_id + i * blockDim.x;

       // if (thread_id == i)
       // {
       //     query_index = global_thread_id;
       //     probe_key = query_list[global_thread_id];
       // }

        // braodcast the probe key to all threads in the tile
        //--> probe_key = tile.shfl(probe_key, i);
        // ---> query_index = tile.shfl(query_index, i);

        probe_key = tile_probe_keys[i];
        query_index = tile_query_indices[i];


       // tile.sync();
        // printf("MAIN LOOP this tile id %u, thread id %u, probe_key %u, query_index %u\n", tile.meta_group_rank(), thread_id, probe_key, query_index);

        smallsize found_index = 0;
        // bool probe_key_found = search_sorted_array_with_tile<key_type>(maxbuf, num_partitions_with_overflow, probe_key, found_index, tile);
        if (thread_id == 0)
        {
            bool probe_key_found = binary_search_equal_or_greater<key_type>(maxbuf, num_partitions_with_overflow, probe_key, found_index, thread_id);
            // probe_key_found = binary_search_in_array<key_type>(maxbuf, maxbufsize, probe_key, found_index, tid);
        }
        //tile.sync();

        found_index = tile.shfl(found_index, 0);

        //tile.sync();
        smallsize bucket_index = found_index; // is the maxval of the bucket where this key belongs... so it is the i'th bucket

        if (bucket_index >= num_partitions_with_overflow)
        {
            printf("Error: bucket index out of bounds, tid: %d\n", thread_id);
            return;
        }

      
        auto curr_node = reinterpret_cast<uint8_t *>(buf) + launch_params->node_stride* bucket_index;

        smallsize currnodesize = cg::extract<smallsize>(curr_node, sizeof(key_type));
        key_type currnodeMax = cg::extract<key_type>(curr_node, 0);

        //catch ERRORs
        /*
        if(currnodeMax < probe_key)
        {
            if (thread_id ==1 && probe_key ==3200461485 && tile.meta_group_rank() == 1)
            {
               // printf("Error Thread01: currnodeMax < probe_key, currnodeMax %u currsize %u probe_key %u tile_id %u  thread_id %u, bucket_index %u \n", currnodeMax, currnodesize, probe_key, tile.meta_group_rank(), thread_id, bucket_index);
               // print_node<key_type>(curr_node, node_size);
            }

            //if (thread_id ==0) 
           // printf("Error: currnodeMax < probe_key, currnodeMax %u currsize %u probe_key %u tile_id %u  thread_id %u, bucket_index %u \n", currnodeMax, currnodesize, probe_key, tile.meta_group_rank(), thread_id, bucket_index);
            //if (thread_id == 0 && tile.meta_group_rank() == 3  )
              //  print_node<key_type>(curr_node, node_size);
            //return;
        }

                tile.sync();
            */



    
           // DEBUG_LOOKUP_DEV("Before Calling ProcessLookupTiles this tile id", tile.meta_group_rank(), thread_id, probe_key);
           // DEBUG_LOOKUP_DEV("2nd Before Calling ProcessLookupTiles", query_index, currnodeMax, bucket_index);
            // smallsize node_size = launch_params->node_size;
            // print_node<key_type>(curr_node, node_size);
       

        if (perform_group_lookup)
        {
            printf("Performing Group Lookup with TILE at global_thread_id %d\n", global_thread_id);
        }
        else
        {
            process_lookups_tile<key_type>(currnodeMax, found_index, probe_key, launch_params, curr_node, currnodesize, query_index, tile);
        }
    }

#ifdef PRINT_LOOKUPS_END
    __syncthreads();
    if (tile.meta_group_rank() == 0 && thread_id == 0)
    {
        printf("END LOOKUPS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, global_thread_id);
    }

#endif
}

/*
template <typename key_type>
GLOBALQUALIFIER
void lookup_kernel_loop(updatable_cg_params* launch_params, bool perform_group_lookup) {
    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id = tile.thread_rank();

    const key_type* query_list = static_cast<const key_type*>(launch_params->query_lower);
    smallsize query_size = launch_params->query_size;

    // Each thread will handle its own probe key
    for (int query_index = global_thread_id; query_index < query_size; query_index += gridDim.x * blockDim.x) {

        key_type probe_key = query_list[query_index];

        const key_type* maxbuf = static_cast<const key_type*>(launch_params->maxvalues);
        smallsize num_partitions_with_overflow = launch_params->partition_count_with_overflow;

        smallsize found_index = 0;
        bool probe_key_found = search_sorted_array_with_tile<key_type>(maxbuf, num_partitions_with_overflow, probe_key, found_index, tile);

        if (found_index >= num_partitions_with_overflow) continue;

        auto buf = launch_params->ordered_node_pairs;
        auto curr_node = reinterpret_cast<uint8_t*>(buf) + launch_params->node_stride * found_index;

        smallsize currnodesize = cg::extract<smallsize>(curr_node, sizeof(key_type));
        key_type currnodeMax = cg::extract<key_type>(curr_node, 0);

        if (perform_group_lookup) {
            printf("Performing Group Lookup with TILE at global_thread_id %d\n", global_thread_id);
        } else {
            process_tile<key_type>(currnodeMax, found_index, probe_key, launch_params, curr_node, currnodesize, query_index, tile);
        }
    }
}
    */

// ------------------------------
// DEVICE: bucket precompute pass
// ------------------------------
template <typename key_type, int16_t node_size_log, uint16_t cg_size_log, bool caching_enabled>
__global__ void bucket_lookup_kernel_params(
    updatable_cg_params* __restrict__ launch_params,
    const key_type* __restrict__ query_list,
    smallsize* __restrict__ bucket_values,
    smallsize query_size)
{
    /* namespace cg = coop_g; 
    cg::thread_block block                = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    const int       global_thread_id      = blockIdx.x * blockDim.x + threadIdx.x;
    const smallsize lane                  = tile.thread_rank();

    // Route params
    const key_type* __restrict__ tree     = static_cast<const key_type*>(launch_params->tree_buffer);
    const auto      metadata_out          = launch_params->metadata;
    const smallsize num_buckets_with_of   = launch_params->partition_count_with_overflow;

    // Unused here; kept for signature parity
    constexpr bool  use_row_layout        = false;
    const key_type* shmem_tree            = nullptr;
    const smallsize cached_entries_count  = 0;

    const int base_idx = global_thread_id - static_cast<int>(lane);

    key_type  my_key    = key_type{};
    smallsize my_qindex = 0;
    bool      in_range  = false;

    if (base_idx >= static_cast<int>(query_size)) return;

    if (base_idx >= 0) {
        const int qidx = base_idx + lane;
        in_range       = (qidx < static_cast<int>(query_size));
        if (in_range) {
            my_qindex = static_cast<smallsize>(qidx);
            my_key    = query_list[qidx];
        }
    }

    const bool had_work = in_range;
    smallsize  my_bucket = not_found;

    unsigned mask = tile.ballot(in_range);
    while (mask) {
        const int leader = __ffs(mask) - 1;

        const key_type  probe_key   = tile.shfl(my_key,    leader);
        const smallsize query_index = tile.shfl(my_qindex, leader);

        const smallsize found_index = static_tree_search<key_type, node_size_log, cg_size_log, use_row_layout>(
            tile, probe_key, tree, metadata_out, shmem_tree, cached_entries_count);

        const smallsize out_idx = (found_index < num_buckets_with_of) ? found_index : not_found;

        if (lane == leader) {
            my_bucket = out_idx;
        }

        if (lane == leader) in_range = false;
        mask = tile.ballot(in_range);
    }

    if (had_work) {
        bucket_values[my_qindex] = my_bucket;
    }
        */
}

// ----------------------------------------------------
// DEVICE: lookup using precomputed bucket index buffer
// ----------------------------------------------------

template <typename key_type, int16_t node_size_log, uint16_t cg_size_log, bool caching_enabled>
__global__ void lookup_kernel_tile_static_tree_params_with_buffer(
    updatable_cg_params* __restrict__ launch_params,
    const key_type* __restrict__ query_list,
    const smallsize* __restrict__ precomputed_buckets,
    smallsize* __restrict__ results,
    smallsize query_size)
{

    /*

     namespace cg = coop_g; 

    cg::thread_block block                = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    const int       global_thread_id      = blockIdx.x * blockDim.x + threadIdx.x;
    const smallsize lane                  = tile.thread_rank();

    // Route params
    const auto      metadata_out          = launch_params->metadata; // kept for debug parity
    const smallsize num_buckets_with_of   = launch_params->partition_count_with_overflow;
    const smallsize node_stride           = launch_params->node_stride;
    uint8_t* __restrict__ node_buf        = static_cast<uint8_t*>(launch_params->ordered_node_pairs);

    const int base_idx = global_thread_id - static_cast<int>(lane);

    key_type  my_key    = key_type{};
    smallsize my_qindex = 0;
    bool      in_range  = false;

    if (base_idx >= static_cast<int>(query_size)) return;

    if (base_idx >= 0) {
        const int qidx = base_idx + lane;
        in_range       = (qidx < static_cast<int>(query_size));
        if (in_range) {
            my_qindex = static_cast<smallsize>(qidx);
            my_key    = query_list[qidx];
        }
    }

    const bool had_work = in_range;
    smallsize  my_out   = not_found;

    unsigned mask = tile.ballot(in_range);
    while (mask) {
        const int leader = __ffs(mask) - 1;

        const key_type  probe_key   = tile.shfl(my_key,    leader);
        const smallsize query_index = tile.shfl(my_qindex, leader);

        const smallsize found_index = tile.shfl(precomputed_buckets[my_qindex], leader);

        if (found_index < num_buckets_with_of) {
            uint8_t* __restrict__ curr_node = node_buf + static_cast<size_t>(node_stride) * found_index;

            smallsize leader_result = not_found;
            const bool ok = process_lookups_tile_ldg_results<key_type>(
                key_type{}, found_index, probe_key,
                launch_params, curr_node, key_type{},
                query_index, tile, leader_result);

            if (lane == leader) {
                my_out = ok ? leader_result : not_found;
            }
        } else {
            if (lane == leader) my_out = not_found;
        }

        if (lane == leader) in_range = false;
        mask = tile.ballot(in_range);
    }

    if (had_work) {
        results[my_qindex] = my_out;
    }

#ifdef PRINT_LOOKUPS_END
    __syncthreads();
    print_set_nodes_and_links<key_type>(launch_params, global_thread_id);
#endif

*/
}


#endif