// =============================================================================
// File: coarse_granular_inserts_tiles_bulk.cuh
// Author: Rosina Kharal
// Description: Implements coarse_granular_inserts_tiles_bulk
//              Primary files for tiled Bulk Insertions in FliX
//              Different versions of functions exist - used in testing and optimizing the functions for the final version of FliX. 
// =============================================================================

#ifndef INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_INSERTS_TILES_BULK_CUH
#define INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_INSERTS_TILES_BULK_CUH

// #include <cooperative_groups.h>
#include "coarse_granular_deletes.cuh"
#include "coarse_granular_inserts.cuh"
#include "coarse_granular_deletes_tiles.cuh"
#include "definitions_updates.cuh"
#include "tile_utils.cuh"

template <typename key_type>
DEVICEQUALIFIER void process_inserts_tile_bulk(
    key_type maxkey, smallsize minindex, smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    smallsize partition_count = launch_params->partition_count_with_overflow;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize my_tid = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    smallsize total_keys = maxindex - minindex + 1;
    smallsize total_insert_count = 0;
    void *curr_node = starting_node;
    smallsize update_idx = minindex;
    key_type curr_max = cg::extract<key_type>(curr_node, 0);

    // if (tile_id == 1) DEBUG_PI_TILE_BULK("Before Loop", curr_max, minindex, maxindex);
    // tile.sync();

    while (update_idx <= maxindex)
    {
        key_type key = update_list[update_idx];
        smallsize offset = offset_list[update_idx];

        // if (key ==1248854015) DEBUG_PI_TILE_BULK("Insert Key", key, offset, update_idx);
        // if (key ==1248854015 ) DEBUG_PI_TILE_BULK("Thread Info", my_tid, tile_id, curr_max);

        if (tile.thread_rank() == 0)
        {
            while (curr_max < key)
            {
                smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count)
                {
                    ERROR_INSERTS("Invalid link in chain", key, tile_id, my_tid);
                }
                /// if (tile_id == 1) DEBUG_PI_TILE_BULK("Next Node", key, next_ptr, my_tid);
                curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + (next_ptr - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        curr_max = tile.shfl(curr_max, 0);
        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr = tile.shfl(raw_ptr, 0);
        curr_node = reinterpret_cast<void *>(raw_ptr);

        // tile.sync();

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // need to check if space in the node for new keys
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("Current Node Size", curr_size, node_size, my_tid);
        if (curr_size >= node_size)
        {
            // DEBUG_PI_TILE_BULK("Node size exceeded", key, curr_size, node_size);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);

            continue;
        }

        smallsize insert_index = 1;
        smallsize insert_count = 0;
        key_type testkey = 0;
        key_type my_key = 0;
        smallsize my_offset = 0;
        bool active = false;
        bool found = false;
        bool max_testkey = false;

        smallsize free_space = node_size - curr_size;

        if (my_tid < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, my_tid + 1);
            my_offset = extract_offset_node<key_type>(curr_node, my_tid + 1);
            if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
            {
                active = true;
                found = (my_key == key);
            }
        }
        tile.sync();

        bool key_found = tile.any(found);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("Post DupCheck", my_tid, found, key_found);
        // if (my_key ==2030567423 ) DEBUG_PI_TILE_BULK("MyKey", my_key, tile_id, key);
        // tile.sync();

        if (key_found)
        {
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP FOUND", key, my_tid, tile_id);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP MATCH", key, my_key, testkey);
            update_idx++;
            continue;
        }

        bool is_gt = active && (my_key > key);
        unsigned mask = tile.ballot(is_gt);

        if (mask != 0)
        {
            insert_index = __ffs(mask);
            testkey = tile.shfl(my_key, insert_index - 1);
        }
        else
        {
            insert_index = curr_size + 1;
            max_testkey = true;
            testkey = curr_max;
        }

        tile.sync();
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("InsertPos", insert_index, my_tid, is_gt);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("TestKey", testkey, curr_max, mask);

        if (my_tid == 0)
        {
            set_key_node<key_type>(curr_node, insert_index, key);
            set_offset_node<key_type>(curr_node, insert_index, offset);
            insert_count++;
            update_idx++;
            smallsize next_index = insert_index + 1;
            key_type next_key = 0;

            while (update_idx <= maxindex && insert_count < free_space && next_index <= node_size)
            {
                next_key = update_list[update_idx];
                // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", update_idx, key, testkey);
                // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", next_key, my_tid, tile_id);

                if (next_key == testkey && !max_testkey)
                {
                    // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                    update_idx++;
                    break;
                }
                else if (next_key > testkey)
                {
                    break;
                }
                else
                {
                    key = next_key;
                    offset = offset_list[update_idx];
                    // if (next_key ==1248854015) DEBUG_PI_TILE_BULK(" Performing INsert", key, offset, next_index);

                    set_key_node<key_type>(curr_node, next_index, key);
                    set_offset_node<key_type>(curr_node, next_index, offset);

                    // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                    insert_count++;
                    update_idx++;
                    next_index++;
                }
            }
            // if (next_key ==1248854015) DEBUG_PI_TILE_BULK("InsertDone", insert_count, curr_size, next_index);
        }

        tile.sync();
        insert_count = tile.shfl(insert_count, 0);
        update_idx = tile.shfl(update_idx, 0);
        total_insert_count += insert_count;
        curr_size += insert_count;
        if (my_tid == 0)
        {
            cg::set<smallsize>(curr_node, sizeof(key_type), curr_size);
            // if (tile_id == 1) print_node<key_type>(curr_node, node_size);
        }

        curr_size = tile.shfl(curr_size, 0);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("SizePost", my_tid, curr_size, insert_count);

        if (active && my_tid >= (insert_index - 1))
        {
            //  if (tile_id == 1) DEBUG_PI_TILE_BULK("Shift", my_tid, insert_index, node_size);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("ShiftVal", my_key, my_offset, curr_size);

            set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
            set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);
        }
        // if (my_tid==0 && tile_id == 1) DEBUG_PI_TILE_BULK("Post Full Update to Node", my_tid, curr_size, insert_count);
        // if (my_tid==0 && tile_id == 1) print_node<key_type>(curr_node, node_size);

        tile.sync();

        // if (curr_size > node_size || insert_count > free_space)
        // {
        //     ERROR_INSERTS("Overflow", key, curr_size, node_size);
        //}

        if (curr_size == node_size && update_idx < maxindex &&
            (total_keys - total_insert_count > 0) &&
            (update_list[update_idx] <= curr_max))
        {
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("SplitTrigger", curr_size, update_list[update_idx], curr_max);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);
        }

        // tile.sync();
        // if (my_tid==0 && tile_id == 1) DEBUG_PI_TILE_BULK("END ONE ROUND", my_tid, tile_id);
    }

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();
    if (my_tid == 0 && tile_id == 1)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, my_tid);
    }
#endif
}

template <typename key_type>
DEVICEQUALIFIER void process_inserts_tile_bulk_hybrid(
    key_type maxkey, smallsize minindex, smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    smallsize partition_count = launch_params->partition_count_with_overflow;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize my_tid = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    smallsize total_keys = maxindex - minindex + 1;
    smallsize total_insert_count = 0;
    void *curr_node = starting_node;
    smallsize update_idx = minindex;
    key_type curr_max = cg::extract<key_type>(curr_node, 0);

    // if (tile_id == 1) DEBUG_PI_TILE_BULK("Before Loop", curr_max, minindex, maxindex);
    // tile.sync();

    while (update_idx <= maxindex)
    {
        key_type key = update_list[update_idx];
        smallsize offset = offset_list[update_idx];

        // if (key ==1248854015) DEBUG_PI_TILE_BULK("Insert Key", key, offset, update_idx);
        // if (key ==1248854015 ) DEBUG_PI_TILE_BULK("Thread Info", my_tid, tile_id, curr_max);

        if (tile.thread_rank() == 0)
        {
            while (curr_max < key)
            {
                smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count)
                {
                    ERROR_INSERTS("Invalid link in chain", key, tile_id, my_tid);
                }
                /// if (tile_id == 1) DEBUG_PI_TILE_BULK("Next Node", key, next_ptr, my_tid);
                curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + (next_ptr - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        curr_max = tile.shfl(curr_max, 0);
        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr = tile.shfl(raw_ptr, 0);
        curr_node = reinterpret_cast<void *>(raw_ptr);

        // tile.sync();

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // need to check if space in the node for new keys
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("Current Node Size", curr_size, node_size, my_tid);
        if (curr_size >= node_size)
        {
            // DEBUG_PI_TILE_BULK("Node size exceeded", key, curr_size, node_size);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);

            continue;
        }

        smallsize insert_index = 1;
        smallsize insert_count = 0;
        key_type testkey = 0;
        key_type my_key = 0;
        smallsize my_offset = 0;
        bool active = false;
        bool found = false;
        bool max_testkey = false;

        smallsize free_space = node_size - curr_size;
        //--------------------------------- REGULAR TILE SHIFT RIGHT LOGIC --------------------------------

        if (free_space < 2 || (total_keys - total_insert_count) < 2)
        {
            smallsize insert_idx = 0;
            bool search_found = single_search_cuda_buffer_tile<key_type>(tile, curr_node, key, curr_size, insert_idx);

            if (search_found)
            {
                update_idx++;
                continue;
            }

            key_type key_at_index = extract_key_node<key_type>(curr_node, insert_idx);

            // if (tile.thread_rank() == 0) DEBUG_PI_TILE("Not Found", key, key_at_index, insert_index);

            if (key_at_index < key)
            {
                // insert_index = (curr_size == 0) ? 1 : insert_index + 1;
                insert_idx = (curr_size == 0) ? 1 : insert_idx;
            }

            if (curr_size < node_size)
            {
                /* if (tile.thread_rank() == 0 && key==1235571559 ) {
                    printf("DEBUG_PI_TILE: Before shift insert | Key: %llu, Insert Index: %u, Tile ID: %u, Thread ID: %u\n",static_cast<unsigned long long>(key), insert_index, tile_id, this_tid);
                }
                */
                perform_shift_insert_tile<key_type>(tile, curr_node, insert_idx, curr_size, key, offset);
                update_idx++;
                /* if (tile.thread_rank() == 0 && key==1235571559 ) {
                    printf("DEBUG_PI_TILE: After shift insert | Key: %llu, Insert Index: %u, Tile ID: %u, Thread ID: %u\n",static_cast<unsigned long long>(key), insert_index, tile_id, this_tid);
                    print_node<key_type>(curr_node, node_size);
                }*/
            }
            total_insert_count += 1;
            continue;
        }

        //--------------------------------- BULK INSERTION LOGIC --------------------------------

        if (my_tid < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, my_tid + 1);
            my_offset = extract_offset_node<key_type>(curr_node, my_tid + 1);
            if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
            {
                active = true;
                found = (my_key == key);
            }
        }
        tile.sync();

        bool key_found = tile.any(found);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("Post DupCheck", my_tid, found, key_found);
        // if (my_key ==2030567423 ) DEBUG_PI_TILE_BULK("MyKey", my_key, tile_id, key);
        // tile.sync();

        if (key_found)
        {
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP FOUND", key, my_tid, tile_id);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP MATCH", key, my_key, testkey);
            update_idx++;
            continue;
        }

        bool is_gt = active && (my_key > key);
        unsigned mask = tile.ballot(is_gt);

        if (mask != 0)
        {
            insert_index = __ffs(mask);
            testkey = tile.shfl(my_key, insert_index - 1);
        }
        else
        {
            insert_index = curr_size + 1;
            max_testkey = true;
            testkey = curr_max;
        }

        tile.sync();
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("InsertPos", insert_index, my_tid, is_gt);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("TestKey", testkey, curr_max, mask);

        if (my_tid == 0)
        {
            set_key_node<key_type>(curr_node, insert_index, key);
            set_offset_node<key_type>(curr_node, insert_index, offset);
            insert_count++;
            update_idx++;
            smallsize next_index = insert_index + 1;
            key_type next_key = 0;

            while (update_idx <= maxindex && insert_count < free_space && next_index < node_size)
            {
                next_key = update_list[update_idx];
                // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", update_idx, key, testkey);
                // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", next_key, my_tid, tile_id);

                if (next_key == testkey && !max_testkey)
                {
                    // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                    update_idx++;
                    break;
                }
                else if (next_key > testkey)
                {
                    break;
                }
                else
                {
                    key = next_key;
                    offset = offset_list[update_idx];
                    // if (next_key ==1248854015) DEBUG_PI_TILE_BULK(" Performing INsert", key, offset, next_index);

                    set_key_node<key_type>(curr_node, next_index, key);
                    set_offset_node<key_type>(curr_node, next_index, offset);

                    // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                    insert_count++;
                    update_idx++;
                    next_index++;
                }
            }
            // if (next_key ==1248854015) DEBUG_PI_TILE_BULK("InsertDone", insert_count, curr_size, next_index);
        }

        tile.sync();
        insert_count = tile.shfl(insert_count, 0);
        update_idx = tile.shfl(update_idx, 0);
        total_insert_count += insert_count;
        curr_size += insert_count;
        if (my_tid == 0)
        {
            cg::set<smallsize>(curr_node, sizeof(key_type), curr_size);
            // if (tile_id == 1) print_node<key_type>(curr_node, node_size);
        }

        curr_size = tile.shfl(curr_size, 0);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("SizePost", my_tid, curr_size, insert_count);

        if (active && my_tid >= (insert_index - 1))
        {
            //  if (tile_id == 1) DEBUG_PI_TILE_BULK("Shift", my_tid, insert_index, node_size);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("ShiftVal", my_key, my_offset, curr_size);

            set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
            set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);
        }
        // if (my_tid==0 && tile_id == 1) DEBUG_PI_TILE_BULK("Post Full Update to Node", my_tid, curr_size, insert_count);
        // if (my_tid==0 && tile_id == 1) print_node<key_type>(curr_node, node_size);

        tile.sync();

        // if (curr_size > node_size || insert_count > free_space)
        // {
        //     ERROR_INSERTS("Overflow", key, curr_size, node_size);
        //}

        if (curr_size == node_size && update_idx < maxindex &&
            (total_keys - total_insert_count > 0) &&
            (update_list[update_idx] < curr_max))
        {
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("SplitTrigger", curr_size, update_list[update_idx], curr_max);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);
        }

        // tile.sync();
        // if (my_tid==0 && tile_id == 1) DEBUG_PI_TILE_BULK("END ONE ROUND", my_tid, tile_id);
    }

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();
    if (my_tid == 0 && tile_id == 1)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, my_tid);
    }
#endif
}

template <typename key_type>
DEVICEQUALIFIER void process_INSERTS_TILE_BULK_ONLY(
    key_type maxkey, smallsize minindex, smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile, const key_type *__restrict__ update_list, const smallsize *__restrict__ offset_list,
    smallsize update_size)

{
    // const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    // const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    smallsize partition_count = launch_params->partition_count_with_overflow;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize my_tid = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    smallsize total_keys = maxindex - minindex + 1;
    smallsize total_insert_count = 0;
    void *curr_node = starting_node;
    smallsize update_idx = minindex;
    key_type curr_max = cg::extract<key_type>(curr_node, 0);

    // if (tile_id == 1) DEBUG_PI_TILE_BULK("Before Loop", curr_max, minindex, maxindex);
    // tile.sync();
    smallsize total_insert_count_across_all_nodes = 0;
    while (update_idx <= maxindex)
    {

        key_type key = update_list[update_idx];
        smallsize offset = offset_list[update_idx];

        if (tile.thread_rank() == 0)
        {
            while (curr_max < key)
            {
                smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count)
                {
                    ERROR_INSERTS("Invalid link in chain", key, tile_id, my_tid);
                }
                /// if (tile_id == 1) DEBUG_PI_TILE_BULK("Next Node", key, next_ptr, my_tid);
                curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + (next_ptr - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        curr_max = tile.shfl(curr_max, 0);
        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr = tile.shfl(raw_ptr, 0);
        curr_node = reinterpret_cast<void *>(raw_ptr);

        // if (tile_id == 0 && my_tid == 0)  DEBUG_PI_TILE_BULK("Insert Key", key, offset, update_idx);
        // if (tile_id == 0 && my_tid == 0) DEBUG_PI_TILE_BULK("Thread Info", my_tid, tile_id, curr_max);

        // tile.sync();

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // need to check if space in the node for new keys
        if (tile_id == 0 && my_tid == 0)
            DEBUG_PI_TILE_BULK("Current Node Size", curr_size, node_size, my_tid);
        if (curr_size >= node_size)
        {
            // DEBUG_PI_TILE_BULK("Node size exceeded", key, curr_size, node_size);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);

            continue;
        }

        smallsize insert_index = 1;
        smallsize insert_count = 0;
        key_type testkey = 0;
        smallsize testoffset = 0;
        key_type my_key = 0;
        smallsize my_offset = 0;
        bool active = false;
        bool found = false;
        bool max_testkey = false;
        smallsize tid_test_key = node_size + 1;
        smallsize tid_old_test_key = 0;

        smallsize free_space = node_size - curr_size;
        // NOTE try the other way first - then use FREE_SPACE

        //--------------------------------- REGULAR TILE SHIFT RIGHT LOGIC --------------------------------

        // Changes: **------------------------------------
        smallsize num_left_to_insert = maxindex - update_idx + 1;

        if (hybrid || all_shift_insertions)
        {
            // **--------------------------------------------
            //-- PRIOR WAYif (free_space < 1 || (total_keys - total_insert_count) < 1)
            smallsize node_max_fill_state = SDIV(node_size * node_threshold, 100); // variable argument at command line
            smallsize min_remaining_to_insert = min_keys_insert;

            // --- 2 conditinos for HYBRD:  if (all_shift_insertions || (hybrid && ((curr_size >= node_max_fill_state) || (num_left_to_insert <= min_remaining_to_insert)))) // hybrid5

            // 1 condition only
            if (all_shift_insertions || (hybrid && (num_left_to_insert <= min_remaining_to_insert))) // hybrid5

            {
                // printf("performing regular insertions in tile hybrid \n ");

                smallsize insert_idx = 0;
                bool search_found = single_search_cuda_buffer_tile<key_type>(tile, curr_node, key, curr_size, insert_idx);

                if (search_found)
                {
                    update_idx++;
                    continue;
                }

                key_type key_at_index = extract_key_node<key_type>(curr_node, insert_idx);

                // if (tile.thread_rank() == 0) DEBUG_PI_TILE("Not Found", key, key_at_index, insert_index);

                if (key_at_index < key)
                {
                    // insert_index = (curr_size == 0) ? 1 : insert_index + 1;
                    insert_idx = (curr_size == 0) ? 1 : insert_idx;
                }

                if (curr_size < node_size)
                {
                    /* if (tile.thread_rank() == 0 && key==1235571559 ) {
                        printf("DEBUG_PI_TILE: Before shift insert | Key: %llu, Insert Index: %u, Tile ID: %u, Thread ID: %u\n",static_cast<unsigned long long>(key), insert_index, tile_id, this_tid);
                    }
                    */
                    perform_shift_insert_tile<key_type>(tile, curr_node, insert_idx, curr_size, key, offset);
                    update_idx++;
                    /* if (tile.thread_rank() == 0 && key==1235571559 ) {
                        printf("DEBUG_PI_TILE: After shift insert | Key: %llu, Insert Index: %u, Tile ID: %u, Thread ID: %u\n",static_cast<unsigned long long>(key), insert_index, tile_id, this_tid);
                        print_node<key_type>(curr_node, node_size);
                    }*/
                }
                total_insert_count += 1;
                continue;
            }
        }
        //--------------------------------- BULK INSERTION LOGIC --------------------------------
        // printf("performing BULK insertions in tile hybrid \n ");

        if (my_tid < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, my_tid + 1);
            my_offset = extract_offset_node<key_type>(curr_node, my_tid + 1);
            if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
            {
                active = true;
                found = (my_key == key);
            }
        }

        bool key_found = tile.any(found);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("Post DupCheck", my_tid, found, key_found);
        // if (my_key ==2030567423 ) DEBUG_PI_TILE_BULK("MyKey", my_key, tile_id, key);
        // tile.sync();

        if (key_found)
        {
            if (my_tid == 0)
                DEBUG_PI_TILE_BULK("DUP FOUND", key, my_tid, tile_id);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP MATCH", key, my_key, testkey);
            update_idx++;
            continue;
        }
        tile.sync();
        bool insert_curr_node = true;
        key_found = false;
        smallsize added_keys = 0;

        //***************************************************** */
        // LOOP AROUND AND KEEP TRYING TO ADD INSERT KEYS TO CURR_NODE as LONG AS We CAN
        //***************************************************** */

        // if (tile_id == 0 && my_tid == 0) {
        //             DEBUG_PI_TILE_BULK("CurrNode Before Insert Loop", my_key, update_idx, maxindex);
        //             print_node<key_type>(curr_node, node_size);
        //
        // }

        while (insert_curr_node)
        {
            max_testkey = false;

            added_keys = 0;
            bool is_gt = active && (my_key > key);
            unsigned mask = tile.ballot(is_gt);

            if (mask != 0)
            {
                insert_index = __ffs(mask);
                testkey = tile.shfl(my_key, insert_index - 1);
                testoffset = tile.shfl(my_offset, insert_index - 1);
                tid_test_key = insert_index - 1;
            }
            else
            {

                insert_index = curr_size + 1;
                max_testkey = true;
                testkey = curr_max;
                tid_test_key = node_size + 1;
            }

            tile.sync();
            // if (tile_id == 0 && my_tid == 0)DEBUG_PI_TILE_BULK("InsertPos", key, testkey, is_gt);
            // if (tile_id == 0 && my_tid == 0)DEBUG_PI_TILE_BULK("TestKey", testkey, tid_test_key, insert_index);

            // before we insert the next key. update the current node with the original keys that need to come into place
            smallsize added_keys = 0;
            // if (tid_old_test_key != tid_test_key && tid_old_test_key >= 0 && insert_count > 0)
            if (tid_old_test_key != tid_test_key && insert_count > 0)

            {
                // if (tile_id == 0) DEBUG_PI_TILE_BULK("Add originals", my_tid, my_key, my_offset);
                // if (tile_id == 0) DEBUG_PI_TILE_BULK("Add originals2", insert_index, tid_old_test_key, tid_test_key);

                smallsize max_thread_id = (tid_test_key <= node_size) ? tid_test_key : node_size;
                if (active && my_tid >= tid_old_test_key && my_tid < max_thread_id)

                {
                    set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
                    set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);
                    added_keys = tid_test_key - tid_old_test_key;
                }
                //*******DEBUGGING*/
                /*
                if (tile_id == 0 && my_tid == tid_old_test_key) {
                    DEBUG_PI_TILE_BULK("Done Adding Originals", my_key, tid_old_test_key, tid_test_key);
                    print_node<key_type>(curr_node, node_size);

                } */
                //*******DEBUGGING*/
            }

            insert_index = insert_index + insert_count;
            smallsize next_index = insert_index + 1; // next place where we can begin inserting
            // insert_count = 0;
            if (my_tid == 0)
            {
                set_key_node<key_type>(curr_node, insert_index, key);
                set_offset_node<key_type>(curr_node, insert_index, offset);
                insert_count++;
                update_idx++;

                //*******DEBUGGING*/
                /* if (tile_id == 0 && my_tid == 0) {
                    DEBUG_PI_TILE_BULK("Done inserting one", key, insert_index, insert_index);
                    print_node<key_type>(curr_node, node_size);

                } */
                //*******DEBUGGING*/

                key_type next_key = 0;

                // try to see if any more keys can fit in before TEST KEY
                while (update_idx <= maxindex && insert_count < free_space && next_index <= node_size)
                {
                    next_key = update_list[update_idx];
                    offset = offset_list[update_idx];
                    // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", update_idx, key, testkey);
                    // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", next_key, my_tid, tile_id);

                    if (next_key == testkey && !max_testkey)
                    {
                        // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                        update_idx++;
                        break;
                    }
                    else if (next_key > testkey)
                    {
                        break;
                    }
                    else
                    {
                        key = next_key;
                        // offset = offset_list[update_idx];
                        //  if (next_key ==1248854015) DEBUG_PI_TILE_BULK(" Performing INsert", key, offset, next_index);

                        set_key_node<key_type>(curr_node, next_index, key);
                        set_offset_node<key_type>(curr_node, next_index, offset);

                        // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                        insert_count++;
                        update_idx++;
                        next_index++;
                    }
                }
                // if (next_key ==1248854015) DEBUG_PI_TILE_BULK("InsertDone", insert_count, curr_size, next_index);
            }
            // INSERTED ONE FOR SURE and POSSIBLY MORE THAT WERE LESS THAN TEST KEY

            tile.sync();
            insert_count = tile.shfl(insert_count, 0);
            update_idx = tile.shfl(update_idx, 0);
            next_index = tile.shfl(next_index, 0);
            total_insert_count += insert_count;

            if (update_idx > maxindex || next_index > node_size || insert_count >= free_space)
            {
                insert_curr_node = false; // no more keys to insert in this node
                break;
            }
            key = update_list[update_idx];
            offset = offset_list[update_idx];

            if (key > curr_max)
            {
                insert_curr_node = false;
                break;
            } // no more keys in this node

            found = (active && my_key == key);
            key_found = tile.any(found);

            // exit if key found and re do loop
            while (key_found && key <= curr_max && update_idx <= maxindex)
            {
                // if (tile_id == 0) DEBUG_PI_TILE_BULK("DUP FOUND", key, my_tid, tile_id);
                //  if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP MATCH", key, my_key, testkey);
                update_idx++;
                key = update_list[update_idx];
                offset = offset_list[update_idx];
                found = (active && my_key == key);
                key_found = tile.any(found);
            }

            if (key_found || key > curr_max || update_idx > maxindex)
                insert_curr_node = false;

            // --- go to top
            tid_old_test_key = tid_test_key;

            tile.sync();
        }

        // Update TOTAL SIZE
        total_insert_count_across_all_nodes += insert_count;
        curr_size += insert_count;
        if (my_tid == 0)
        {
            cg::set<smallsize>(curr_node, sizeof(key_type), curr_size);
            // if (tile_id == 1) print_node<key_type>(curr_node, node_size);
        }

        curr_size = tile.shfl(curr_size, 0);

        // if (active && my_tid >= (insert_index - 1))
        if (active && my_tid >= (tid_test_key))
        {
            //  if (tile_id == 1) DEBUG_PI_TILE_BULK("Shift", my_tid, insert_index, node_size);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("ShiftVal", my_key, my_offset, curr_size);
            //  DEBUG_PI_TILE_BULK("Bottom After While Before ADDING oRIGNS",my_tid, my_key, update_idx, insert_index);
            set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
            set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);

            /* if (tile_id == 0 && my_tid == insert_index - 1) {
                  DEBUG_PI_TILE_BULK("Bottom After While ADDED oRIGNS", my_key, update_idx, maxindex);
                  print_node<key_type>(curr_node, node_size);

              } */
        }

        tile.sync();

        if (curr_size == node_size && update_idx <= maxindex &&
            (total_keys - total_insert_count_across_all_nodes > 0) &&
            (update_list[update_idx] <= curr_max))
        {
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("SplitTrigger", curr_size, update_list[update_idx], curr_max);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);
        }
    }

    // tile.sync();
    // if (my_tid==0 && tile_id == 1) DEBUG_PI_TILE_BULK("END ONE ROUND", my_tid, tile_id);

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();
    if (my_tid == 0 && tile_id == 1)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, my_tid);
    }
#endif
}

template <typename key_type>
DEVICEQUALIFIER void process_inserts_tile_bulk_only(
    key_type bucket_max, smallsize minindex, smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile, const key_type *__restrict__ update_list, const smallsize *__restrict__ offset_list,
    smallsize update_size)

{
    // const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    // const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    smallsize partition_count = launch_params->partition_count_with_overflow;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize my_tid = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    smallsize total_keys = maxindex - minindex + 1;
    smallsize total_insert_count = 0;
    void *curr_node = starting_node;
    smallsize update_idx = minindex;
    key_type curr_max = cg::extract<key_type>(curr_node, 0);

    // if (tile_id == 1) DEBUG_PI_TILE_BULK("Before Loop", curr_max, minindex, maxindex);
    // tile.sync();
    smallsize total_insert_count_across_all_nodes = 0;
    while (update_idx <= maxindex)
    {

        key_type key = update_list[update_idx];
        smallsize offset = offset_list[update_idx];

        if (tile.thread_rank() == 0)
        {
            while (curr_max < key)
            {
                smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count)
                {
                    ERROR_INSERTS("Invalid link in chain", key, tile_id, my_tid);
                }
                /// if (tile_id == 1) DEBUG_PI_TILE_BULK("Next Node", key, next_ptr, my_tid);
                curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + (next_ptr - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        curr_max = tile.shfl(curr_max, 0);
        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr = tile.shfl(raw_ptr, 0);
        curr_node = reinterpret_cast<void *>(raw_ptr);

        // if (tile_id == 0 && my_tid == 0)  DEBUG_PI_TILE_BULK("Insert Key", key, offset, update_idx);
        // if (tile_id == 0 && my_tid == 0) DEBUG_PI_TILE_BULK("Thread Info", my_tid, tile_id, curr_max);

        // tile.sync();

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // need to check if space in the node for new keys
        if (tile_id == 0 && my_tid == 0)
            DEBUG_PI_TILE_BULK("Current Node Size", curr_size, node_size, my_tid);
        if (curr_size >= node_size)
        {
            // DEBUG_PI_TILE_BULK("Node size exceeded", key, curr_size, node_size);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);

            continue;
        }

        smallsize insert_index = 1;
        smallsize insert_count = 0;
        key_type testkey = 0;
        smallsize testoffset = 0;
        key_type my_key = 0;
        smallsize my_offset = 0;
        bool active = false;
        bool found = false;
        bool max_testkey = false;
        smallsize tid_test_key = node_size + 1;
        smallsize tid_old_test_key = 0;

        smallsize free_space = node_size - curr_size;
        // NOTE try the other way first - then use FREE_SPACE

        //--------------------------------- REGULAR TILE SHIFT RIGHT LOGIC --------------------------------

        /*
         // Changes: **------------------------------------
         smallsize num_left_to_insert = maxindex - update_idx + 1;

         if (hybrid || all_shift_insertions)
         {
             // **--------------------------------------------
             //-- PRIOR WAYif (free_space < 1 || (total_keys - total_insert_count) < 1)
             smallsize node_max_fill_state = SDIV(node_size * node_threshold, 100); // variable argument at command line
             smallsize min_remaining_to_insert = min_keys_insert;

             // --- 2 conditinos for HYBRD:  if (all_shift_insertions || (hybrid && ((curr_size >= node_max_fill_state) || (num_left_to_insert <= min_remaining_to_insert)))) // hybrid5

             // 1 condition only
             if (all_shift_insertions || (hybrid && (num_left_to_insert <= min_remaining_to_insert))) // hybrid5

             {
                 // printf("performing regular insertions in tile hybrid \n ");

                 smallsize insert_idx = 0;
                 bool search_found = single_search_cuda_buffer_tile<key_type>(tile, curr_node, key, curr_size, insert_idx);

                 if (search_found)
                 {
                     update_idx++;
                     continue;
                 }

                 key_type key_at_index = extract_key_node<key_type>(curr_node, insert_idx);

                 // if (tile.thread_rank() == 0) DEBUG_PI_TILE("Not Found", key, key_at_index, insert_index);

                 if (key_at_index < key)
                 {
                     // insert_index = (curr_size == 0) ? 1 : insert_index + 1;
                     insert_idx = (curr_size == 0) ? 1 : insert_idx;
                 }

                 if (curr_size < node_size)
                 {

                     perform_shift_insert_tile<key_type>(tile, curr_node, insert_idx, curr_size, key, offset);
                     update_idx++;

                 }
                 total_insert_count += 1;
                 continue;
             }
         }

         */
        //--------------------------------- BULK INSERTION LOGIC --------------------------------
        // printf("performing BULK insertions in tile hybrid \n ");

        if (my_tid < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, my_tid + 1);
            my_offset = extract_offset_node<key_type>(curr_node, my_tid + 1);
            if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
            {
                active = true;
                found = (my_key == key);
            }
        }

        bool key_found = tile.any(found);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("Post DupCheck", my_tid, found, key_found);
        // if (my_key ==2030567423 ) DEBUG_PI_TILE_BULK("MyKey", my_key, tile_id, key);
        // tile.sync();

        if (key_found)
        {
            // if (my_tid == 0)
            //  DEBUG_PI_TILE_BULK("DUP FOUND", key, my_tid, tile_id);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP MATCH", key, my_key, testkey);
            update_idx++;
            continue;
        }
        tile.sync();
        bool insert_curr_node = true;
        key_found = false;
        smallsize added_keys = 0;

        //***************************************************** */
        // LOOP AROUND AND KEEP TRYING TO ADD INSERT KEYS TO CURR_NODE as LONG AS We CAN
        //***************************************************** */

        // if (tile_id == 0 && my_tid == 0) {
        //             DEBUG_PI_TILE_BULK("CurrNode Before Insert Loop", my_key, update_idx, maxindex);
        //             print_node<key_type>(curr_node, node_size);
        //
        // }

        while (insert_curr_node)
        {
            max_testkey = false;

            added_keys = 0;
            bool is_gt = active && (my_key > key);
            unsigned mask = tile.ballot(is_gt);

            if (mask != 0)
            {
                insert_index = __ffs(mask);
                testkey = tile.shfl(my_key, insert_index - 1);
                testoffset = tile.shfl(my_offset, insert_index - 1);
                tid_test_key = insert_index - 1;
            }
            else
            {

                insert_index = curr_size + 1;
                max_testkey = true;
                testkey = curr_max;
                tid_test_key = node_size + 1;
            }

            tile.sync();
            // if (tile_id == 0 && my_tid == 0)DEBUG_PI_TILE_BULK("InsertPos", key, testkey, is_gt);
            // if (tile_id == 0 && my_tid == 0)DEBUG_PI_TILE_BULK("TestKey", testkey, tid_test_key, insert_index);

            // before we insert the next key. update the current node with the original keys that need to come into place
            smallsize added_keys = 0;
            // if (tid_old_test_key != tid_test_key && tid_old_test_key >= 0 && insert_count > 0)
            if (tid_old_test_key != tid_test_key && insert_count > 0)

            {
                // if (tile_id == 0) DEBUG_PI_TILE_BULK("Add originals", my_tid, my_key, my_offset);
                // if (tile_id == 0) DEBUG_PI_TILE_BULK("Add originals2", insert_index, tid_old_test_key, tid_test_key);

                smallsize max_thread_id = (tid_test_key <= node_size) ? tid_test_key : node_size;
                if (active && my_tid >= tid_old_test_key && my_tid < max_thread_id)

                {
                    set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
                    set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);
                    added_keys = tid_test_key - tid_old_test_key;
                }
                //*******DEBUGGING*/
                /*
                if (tile_id == 0 && my_tid == tid_old_test_key) {
                    DEBUG_PI_TILE_BULK("Done Adding Originals", my_key, tid_old_test_key, tid_test_key);
                    print_node<key_type>(curr_node, node_size);

                } */
                //*******DEBUGGING*/
            }

            insert_index = insert_index + insert_count;
            smallsize next_index = insert_index + 1; // next place where we can begin inserting
            // insert_count = 0;
            if (my_tid == 0)
            {
                set_key_node<key_type>(curr_node, insert_index, key);
                set_offset_node<key_type>(curr_node, insert_index, offset);
                insert_count++;
                update_idx++;

                //*******DEBUGGING*/
                /* if (tile_id == 0 && my_tid == 0) {
                    DEBUG_PI_TILE_BULK("Done inserting one", key, insert_index, insert_index);
                    print_node<key_type>(curr_node, node_size);

                } */
                //*******DEBUGGING*/

                key_type next_key = 0;

                // try to see if any more keys can fit in before TEST KEY
                while (update_idx <= maxindex && insert_count < free_space && next_index <= node_size)
                {
                    next_key = update_list[update_idx];
                    offset = offset_list[update_idx];
                    // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", update_idx, key, testkey);
                    // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", next_key, my_tid, tile_id);

                    if (next_key == testkey && !max_testkey)
                    {
                        // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                        update_idx++;
                        break;
                    }
                    else if (next_key > testkey)
                    {
                        break;
                    }
                    else
                    {
                        key = next_key;
                        // offset = offset_list[update_idx];
                        //  if (next_key ==1248854015) DEBUG_PI_TILE_BULK(" Performing INsert", key, offset, next_index);

                        set_key_node<key_type>(curr_node, next_index, key);
                        set_offset_node<key_type>(curr_node, next_index, offset);

                        // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                        insert_count++;
                        update_idx++;
                        next_index++;
                    }
                }
                // if (next_key ==1248854015) DEBUG_PI_TILE_BULK("InsertDone", insert_count, curr_size, next_index);
            }
            // INSERTED ONE FOR SURE and POSSIBLY MORE THAT WERE LESS THAN TEST KEY

            tile.sync();
            insert_count = tile.shfl(insert_count, 0);
            update_idx = tile.shfl(update_idx, 0);
            next_index = tile.shfl(next_index, 0);
            total_insert_count += insert_count;

            if (update_idx > maxindex || next_index > node_size || insert_count >= free_space)
            {
                insert_curr_node = false; // no more keys to insert in this node
                break;
            }
            key = update_list[update_idx];
            offset = offset_list[update_idx];

            if (key > curr_max)
            {
                insert_curr_node = false;
                break;
            } // no more keys in this node

            found = (active && my_key == key);
            key_found = tile.any(found);

            // exit if key found and re do loop
            while (key_found && key <= curr_max && update_idx <= maxindex)
            {
                // if (tile_id == 0) DEBUG_PI_TILE_BULK("DUP FOUND", key, my_tid, tile_id);
                //  if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP MATCH", key, my_key, testkey);
                update_idx++;
                key = update_list[update_idx];
                offset = offset_list[update_idx];
                found = (active && my_key == key);
                key_found = tile.any(found);
            }

            if (key_found || key > curr_max || update_idx > maxindex)
                insert_curr_node = false;

            // --- go to top
            tid_old_test_key = tid_test_key;

            tile.sync();
        }

        // Update TOTAL SIZE
        total_insert_count_across_all_nodes += insert_count;
        curr_size += insert_count;
        if (my_tid == 0)
        {
            cg::set<smallsize>(curr_node, sizeof(key_type), curr_size);
            // if (tile_id == 1) print_node<key_type>(curr_node, node_size);
        }

        curr_size = tile.shfl(curr_size, 0);

        // if (active && my_tid >= (insert_index - 1))
        if (active && my_tid >= (tid_test_key))
        {
            //  if (tile_id == 1) DEBUG_PI_TILE_BULK("Shift", my_tid, insert_index, node_size);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("ShiftVal", my_key, my_offset, curr_size);
            //  DEBUG_PI_TILE_BULK("Bottom After While Before ADDING oRIGNS",my_tid, my_key, update_idx, insert_index);
            set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
            set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);

            /* if (tile_id == 0 && my_tid == insert_index - 1) {
                  DEBUG_PI_TILE_BULK("Bottom After While ADDED oRIGNS", my_key, update_idx, maxindex);
                  print_node<key_type>(curr_node, node_size);

              } */
        }

        tile.sync();

        if (curr_size == node_size && update_idx <= maxindex &&
            (total_keys - total_insert_count_across_all_nodes > 0) &&
            (update_list[update_idx] <= curr_max))
        {
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("SplitTrigger", curr_size, update_list[update_idx], curr_max);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);
        }
    }

    // tile.sync();
    // if (my_tid==0 && tile_id == 1) DEBUG_PI_TILE_BULK("END ONE ROUND", my_tid, tile_id);

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();
    if (my_tid == 0 && tile_id == 1)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, my_tid);
    }
#endif
}


template <typename key_type>
DEVICEQUALIFIER void process_inserts_tile_bulk_only_optimized_2(
    key_type bucket_max,
    smallsize minindex,
    smallsize maxindex_old,
    updatable_cg_params *__restrict__ launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const key_type *__restrict__ update_list,
    const smallsize *__restrict__ offset_list,
    smallsize update_size)
{
    //(void)bucket_max;
    ///(void)update_size;

    const smallsize maxindex = update_size - 1;  // now use this everywhere in the function

    uint8_t *__restrict__ allocation_buffer =
        static_cast<uint8_t *>(launch_params->allocation_buffer);

    const smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    const smallsize node_stride = launch_params->node_stride;
    const smallsize node_size = launch_params->node_size;
    const smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    const smallsize my_tid = tile.thread_rank();
    const smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    const smallsize total_keys = maxindex - minindex + 1;

    void *curr_node = starting_node;
    smallsize update_idx = minindex;
    key_type curr_max = cg::extract<key_type>(curr_node, 0);

    smallsize total_insert_count_across_all_nodes = 0;

    //**** PERFORM BULK INSERTIONS ONLY -- NO HYBRID LOGIC ****/

    // here print out all keys in the update_list begining at mindex
#ifdef PRINT_INSERT_VALUES
    if (my_tid == 0 )
    {
        printf("In Bulk Insert Function Update List Keys: bucket_max = %llu\n", static_cast<unsigned long long>(bucket_max) );
        for (smallsize i = minindex; i <= maxindex; ++i)
        {
            printf("Key: %llu, Offset: %u\n", static_cast<unsigned long long>(update_list[i]), static_cast<unsigned>(offset_list[i]));
        }
    }
#endif

    while (update_idx <= maxindex && update_list[update_idx] <= bucket_max)
    {
        key_type key = update_list[update_idx];
        smallsize offset = offset_list[update_idx];

        if ( (curr_max < key) && (my_tid == 0) )
        {
            while (curr_max < key)
            {
                const smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
                // if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count)

                if (next_ptr == 0) // || (next_ptr - 1) >= allocation_buffer_count)

                {
                    ERROR_INSERTS("Invalid link in chain", key, tile_id, my_tid);
                }
                curr_node = allocation_buffer + static_cast<size_t>(next_ptr - 1) * static_cast<size_t>(node_stride);
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        curr_max = tile.shfl(curr_max, 0);
        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr = tile.shfl(raw_ptr, 0);
        curr_node = reinterpret_cast<void *>(raw_ptr);

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        //if (tile_id == 0 && my_tid == 0)
          //  DEBUG_PI_TILE_BULK("Current Node Size", curr_size, node_size, my_tid);

        if (curr_size >= node_size)
        {
            process_split_tile<key_type>(tile, curr_node, launch_params);
            curr_max = cg::extract<key_type>(curr_node, 0);
            continue;
        }

        smallsize insert_index = 1;
        smallsize insert_count = 0;
        key_type testkey = 0;
        smallsize testoffset = 0;
        key_type my_key = 0;
        smallsize my_offset = 0;
        bool active = false;
        bool found = false;
        bool max_testkey = false;
        smallsize tid_test_key = node_size + 1;
        smallsize tid_old_test_key = 0;

        const smallsize free_space = node_size - curr_size;

        // ---------------- BULK INSERTION LOGIC ----------------
        if (my_tid < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, my_tid + 1);
            my_offset = extract_offset_node<key_type>(curr_node, my_tid + 1);

           // if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
            if (my_key != 0 )
            {
                active = true;
                found = (my_key == key);
            }
        }

        bool key_found = tile.any(found);

        if (key_found)
        {
            update_idx++;
            continue;
        }

      // -- dont think we need this  tile.sync();

        bool insert_curr_node = true;
        key_found = false;
        smallsize added_keys = 0;

        while (insert_curr_node)
        {
            max_testkey = false;
            added_keys = 0;

            const bool is_gt = active && (my_key > key);
            const unsigned mask = tile.ballot(is_gt);

            if (mask != 0)
            {
                insert_index = __ffs(mask);
                testkey = tile.shfl(my_key, insert_index - 1);
                testoffset = tile.shfl(my_offset, insert_index - 1);
                tid_test_key = insert_index - 1;
            }
            else
            {
                insert_index = curr_size + 1;
                max_testkey = true;
                testkey = curr_max;
                tid_test_key = node_size + 1;
            }

            tile.sync();

            // before we insert the next key, place original keys that need to move
            if (tid_old_test_key != tid_test_key && insert_count > 0)
            {
                const smallsize max_thread_id = (tid_test_key <= node_size) ? tid_test_key : node_size;

                if (active && my_tid >= tid_old_test_key && my_tid < max_thread_id)
                {
                    set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
                    set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);
                    added_keys = tid_test_key - tid_old_test_key;
                }
            }

            insert_index = insert_index + insert_count;
            smallsize next_index = insert_index + 1;

            if (my_tid == 0)
            {
                set_key_node<key_type>(curr_node, insert_index, key);
                set_offset_node<key_type>(curr_node, insert_index, offset);
                insert_count++;
                update_idx++;

                key_type next_key = 0;

                while (update_idx <= maxindex && insert_count < free_space && next_index <= node_size)
                {
                    next_key = update_list[update_idx];
                    offset = offset_list[update_idx];

                    if (next_key == testkey && !max_testkey)
                    {
                        update_idx++;
                        break;
                    }
                    else if (next_key > testkey)
                    {
                        break;
                    }
                    else
                    {
                        key = next_key;
                        set_key_node<key_type>(curr_node, next_index, key);
                        set_offset_node<key_type>(curr_node, next_index, offset);
                        insert_count++;
                        update_idx++;
                        next_index++;
                    }
                }
            }

            tile.sync();

            insert_count = tile.shfl(insert_count, 0);
            update_idx = tile.shfl(update_idx, 0);
            next_index = tile.shfl(next_index, 0);


           // update_idx <= maxindex && update_list[update_idx] <= bucket_max)
            if (update_idx > maxindex || update_list[update_idx] > bucket_max || next_index > node_size || insert_count >= free_space)
            {
                insert_curr_node = false;
                break;
            }

            key = update_list[update_idx];
            offset = offset_list[update_idx];

            if (key > curr_max)
            {
                insert_curr_node = false;
                break;
            }

            found = (active && my_key == key);
            key_found = tile.any(found);

            while (key_found ) // do not need to check again  && key <= curr_max && update_idx <= maxindex)
            {
                update_idx++;
                if (update_idx > maxindex || update_list[update_idx] > curr_max)
                {
                    insert_curr_node = false;
                    key_found = false;
                    break;
                }
                key = update_list[update_idx];
                offset = offset_list[update_idx];

                found = (active && my_key == key);
                key_found = tile.any(found);
            }

           // if (key_found || key > curr_max || update_idx > maxindex)
           //     insert_curr_node = false;

            tid_old_test_key = tid_test_key;

            tile.sync();
        }

        total_insert_count_across_all_nodes += insert_count;

        curr_size += insert_count;
        if (my_tid == 0)
        {
            cg::set<smallsize>(curr_node, sizeof(key_type), curr_size);
        }

        curr_size = tile.shfl(curr_size, 0);

        if (active && my_tid >= (tid_test_key))
        {
            set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
            set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);
        }

        tile.sync();

        //*** TOO MANY CONDITIONS CHECK THIS ***/
        //if (curr_size == node_size && update_idx <= maxindex &&
        //    (total_keys - total_insert_count_across_all_nodes > 0) &&
         //   (update_list[update_idx] <= curr_max))

        if (curr_size == node_size && update_idx <= maxindex && (update_list[update_idx] <= curr_max))
       
        {
            process_split_tile<key_type>(tile, curr_node, launch_params);
            curr_max = cg::extract<key_type>(curr_node, 0);
        }
    }

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();
    if (my_tid == 0 && tile_id == 1)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, my_tid);
    }
#endif
}
// file: process_inserts_tile_bulk_only.cuh

template <typename key_type>
DEVICEQUALIFIER void process_inserts_tile_bulk_only_optimized(
    key_type bucket_max,
    smallsize minindex,
    smallsize maxindex,
    updatable_cg_params *__restrict__ launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const key_type *__restrict__ update_list,
    const smallsize *__restrict__ offset_list,
    smallsize update_size)
{
    (void)bucket_max;
    (void)update_size;

    uint8_t *__restrict__ allocation_buffer =
        static_cast<uint8_t *>(launch_params->allocation_buffer);

    const smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    const smallsize node_stride = launch_params->node_stride;
    const smallsize node_size = launch_params->node_size;
    const smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    const smallsize my_tid = tile.thread_rank();
    const smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    const smallsize total_keys = maxindex - minindex + 1;

    void *curr_node = starting_node;
    smallsize update_idx = minindex;
    key_type curr_max = cg::extract<key_type>(curr_node, 0);

    smallsize total_insert_count_across_all_nodes = 0;

    while (update_idx <= maxindex)
    {
        key_type key = update_list[update_idx];
        smallsize offset = offset_list[update_idx];

        if (my_tid == 0)
        {
            while (curr_max < key)
            {
                const smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
                // if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count)

                if (next_ptr == 0) // || (next_ptr - 1) >= allocation_buffer_count)

                {
                    ERROR_INSERTS("Invalid link in chain", key, tile_id, my_tid);
                }
                curr_node = allocation_buffer + static_cast<size_t>(next_ptr - 1) * static_cast<size_t>(node_stride);
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        curr_max = tile.shfl(curr_max, 0);
        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr = tile.shfl(raw_ptr, 0);
        curr_node = reinterpret_cast<void *>(raw_ptr);

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        if (tile_id == 0 && my_tid == 0)
            DEBUG_PI_TILE_BULK("Current Node Size", curr_size, node_size, my_tid);

        if (curr_size >= node_size)
        {
            process_split_tile<key_type>(tile, curr_node, launch_params);
            curr_max = cg::extract<key_type>(curr_node, 0);
            continue;
        }

        smallsize insert_index = 1;
        smallsize insert_count = 0;
        key_type testkey = 0;
        smallsize testoffset = 0;
        key_type my_key = 0;
        smallsize my_offset = 0;
        bool active = false;
        bool found = false;
        bool max_testkey = false;
        smallsize tid_test_key = node_size + 1;
        smallsize tid_old_test_key = 0;

        const smallsize free_space = node_size - curr_size;

        // ---------------- BULK INSERTION LOGIC ----------------
        if (my_tid < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, my_tid + 1);
            my_offset = extract_offset_node<key_type>(curr_node, my_tid + 1);

            if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
            {
                active = true;
                found = (my_key == key);
            }
        }

        bool key_found = tile.any(found);

        if (key_found)
        {
            update_idx++;
            continue;
        }

        tile.sync();

        bool insert_curr_node = true;
        key_found = false;
        smallsize added_keys = 0;

        while (insert_curr_node)
        {
            max_testkey = false;
            added_keys = 0;

            const bool is_gt = active && (my_key > key);
            const unsigned mask = tile.ballot(is_gt);

            if (mask != 0)
            {
                insert_index = __ffs(mask);
                testkey = tile.shfl(my_key, insert_index - 1);
                testoffset = tile.shfl(my_offset, insert_index - 1);
                tid_test_key = insert_index - 1;
            }
            else
            {
                insert_index = curr_size + 1;
                max_testkey = true;
                testkey = curr_max;
                tid_test_key = node_size + 1;
            }

            tile.sync();

            // before we insert the next key, place original keys that need to move
            if (tid_old_test_key != tid_test_key && insert_count > 0)
            {
                const smallsize max_thread_id = (tid_test_key <= node_size) ? tid_test_key : node_size;

                if (active && my_tid >= tid_old_test_key && my_tid < max_thread_id)
                {
                    set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
                    set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);
                    added_keys = tid_test_key - tid_old_test_key;
                }
            }

            insert_index = insert_index + insert_count;
            smallsize next_index = insert_index + 1;

            if (my_tid == 0)
            {
                set_key_node<key_type>(curr_node, insert_index, key);
                set_offset_node<key_type>(curr_node, insert_index, offset);
                insert_count++;
                update_idx++;

                key_type next_key = 0;

                while (update_idx <= maxindex && insert_count < free_space && next_index <= node_size)
                {
                    next_key = update_list[update_idx];
                    offset = offset_list[update_idx];

                    if (next_key == testkey && !max_testkey)
                    {
                        update_idx++;
                        break;
                    }
                    else if (next_key > testkey)
                    {
                        break;
                    }
                    else
                    {
                        key = next_key;
                        set_key_node<key_type>(curr_node, next_index, key);
                        set_offset_node<key_type>(curr_node, next_index, offset);
                        insert_count++;
                        update_idx++;
                        next_index++;
                    }
                }
            }

            tile.sync();

            insert_count = tile.shfl(insert_count, 0);
            update_idx = tile.shfl(update_idx, 0);
            next_index = tile.shfl(next_index, 0);

            if (update_idx > maxindex || next_index > node_size || insert_count >= free_space)
            {
                insert_curr_node = false;
                break;
            }

            key = update_list[update_idx];
            offset = offset_list[update_idx];

            if (key > curr_max)
            {
                insert_curr_node = false;
                break;
            }

            found = (active && my_key == key);
            key_found = tile.any(found);

            while (key_found && key <= curr_max && update_idx <= maxindex)
            {
                update_idx++;
                key = update_list[update_idx];
                offset = offset_list[update_idx];

                found = (active && my_key == key);
                key_found = tile.any(found);
            }

            if (key_found || key > curr_max || update_idx > maxindex)
                insert_curr_node = false;

            tid_old_test_key = tid_test_key;

            tile.sync();
        }

        total_insert_count_across_all_nodes += insert_count;

        curr_size += insert_count;
        if (my_tid == 0)
        {
            cg::set<smallsize>(curr_node, sizeof(key_type), curr_size);
        }

        curr_size = tile.shfl(curr_size, 0);

        if (active && my_tid >= (tid_test_key))
        {
            set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
            set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);
        }

        tile.sync();

        if (curr_size == node_size && update_idx <= maxindex &&
            (total_keys - total_insert_count_across_all_nodes > 0) &&
            (update_list[update_idx] <= curr_max))
        {
            process_split_tile<key_type>(tile, curr_node, launch_params);
            curr_max = cg::extract<key_type>(curr_node, 0);
        }
    }

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();
    if (my_tid == 0 && tile_id == 1)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, my_tid);
    }
#endif
}

// File: inserts_tile_bulk_mergepath.cuh (suggested)

template <typename T>
DEVICEQUALIFIER __forceinline__ smallsize upper_bound_device(
    const T *__restrict__ a,
    smallsize lo, smallsize hi, // [lo, hi)
    T value)
{
    while (lo < hi)
    {
        smallsize mid = lo + ((hi - lo) >> 1);
        if (a[mid] <= value)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// Merge-path partition for two sorted arrays A (len m) and B (len n).
// Returns i such that i from A and (diag - i) from B are taken for merged prefix length diag.
// Guarantees boundaries and ordering for stable merge.
template <typename key_type>
DEVICEQUALIFIER __forceinline__ smallsize merge_path_partition(
    const key_type *__restrict__ A, smallsize m,
    const key_type *__restrict__ B, smallsize n,
    smallsize diag)
{
    smallsize i_min = (diag > n) ? (diag - n) : 0;
    smallsize i_max = (diag < m) ? diag : m;

    while (i_min < i_max)
    {
        smallsize i = i_min + ((i_max - i_min) >> 1);
        smallsize j = diag - i;

        // We want A[i-1] <= B[j] and B[j-1] < A[i] for strict tie-breaking.
        key_type A_im1 = (i > 0) ? A[i - 1] : static_cast<key_type>(0);
        key_type B_j = (j < n) ? B[j] : static_cast<key_type>(~key_type(0)); // +inf
        key_type B_jm1 = (j > 0) ? B[j - 1] : static_cast<key_type>(0);
        key_type A_i = (i < m) ? A[i] : static_cast<key_type>(~key_type(0)); // +inf

        // If A[i-1] > B[j], move left (decrease i)
        if (i > 0 && j < n && A_im1 > B_j)
        {
            i_max = i;
            continue;
        }
        // If B[j-1] >= A[i], move right (increase i)
        if (j > 0 && i < m && B_jm1 >= A_i)
        {
            i_min = i + 1;
            continue;
        }
        return i;
    }
    return i_min;
}

// Fast single-node merge insertion: node_size <= TILE_SIZE.
// Merges existing keys [1..curr_size] with update keys in [update_idx..run_end) (sorted).
// Writes back compacted sorted result, returns new_size and advances update_idx.
template <typename key_type>
DEVICEQUALIFIER __forceinline__ void merge_insert_node_tile(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    void *__restrict__ curr_node,
    smallsize node_size,
    smallsize curr_size,
    const key_type *__restrict__ update_list,
    const smallsize *__restrict__ offset_list,
    smallsize &update_idx,
    smallsize run_end)
{
    const smallsize lane = tile.thread_rank();

    // Existing arrays (length m)
    const smallsize m = curr_size;
    const smallsize n = run_end - update_idx;

    // Shared scratch (keys+offsets) sized for TILE_SIZE (common fast path).
    // If TILE_SIZE is 32/64, this is cheap.
    __shared__ key_type shA_keys[1024]; // MUST be >= TILE_SIZE * max tiles per block if multiple tiles
    __shared__ smallsize shA_offs[1024];
    __shared__ key_type shB_keys[1024];
    __shared__ smallsize shB_offs[1024];
    __shared__ key_type shOut_keys[1024];
    __shared__ smallsize shOut_offs[1024];

    // Compute per-tile base into shared (avoid cross-tile clobber).
    // Assumes block contains multiple tiles; allocate per-tile slice.
    const int tiles_per_block = blockDim.x / tile.size();
    const int tile_rank_in_block = tile.meta_group_rank();
    const int base = tile_rank_in_block * TILE_SIZE;

    key_type *A_keys = shA_keys + base;
    smallsize *A_offs = shA_offs + base;
    key_type *B_keys = shB_keys + base;
    smallsize *B_offs = shB_offs + base;
    key_type *O_keys = shOut_keys + base;
    smallsize *O_offs = shOut_offs + base;

    // Load existing keys/offs into shared (lanes < m)
    if (lane < m)
    {
        A_keys[lane] = extract_key_node<key_type>(curr_node, lane + 1);
        A_offs[lane] = extract_offset_node<key_type>(curr_node, lane + 1);
    }
    tile.sync();

    // Load update run into shared (lanes < n) - if n can exceed TILE_SIZE, you need a loop.
    // Fast-path assumes n <= TILE_SIZE. If n > TILE_SIZE, either loop or fall back to your current method.
    if (lane < n)
    {
        const smallsize idx = update_idx + lane;
        B_keys[lane] = update_list[idx];
        B_offs[lane] = offset_list[idx];
    }
    tile.sync();

    // Merge lengths: output length can be up to m + n, but must fit node_size.
    // Fast-path: require (m + n) <= node_size and <= TILE_SIZE for single-pass.
    const smallsize out_cap = node_size;
    const smallsize merged_len = (m + n <= out_cap) ? (m + n) : out_cap;

    // Each lane produces output element at position k = lane if lane < merged_len.
    // Use merge-path to select whether output[k] comes from A or B.
    if (lane < merged_len)
    {
        const smallsize diag = lane;
        const smallsize i = merge_path_partition<key_type>(A_keys, m, B_keys, n, diag);
        const smallsize j = diag - i;

        const bool takeA =
            (i < m) &&
            (j >= n || A_keys[i] <= B_keys[j]);

        key_type k = takeA ? A_keys[i] : B_keys[j];
        smallsize o = takeA ? A_offs[i] : B_offs[j];

        // Drop duplicates vs existing keys (conservative):
        // If B key equals adjacent A key, we prefer A (stable) because takeA uses <=.
        // But duplicates inside B itself are not removed here.
        O_keys[lane] = k;
        O_offs[lane] = o;
    }
    tile.sync();

    // Write back merged prefix into node
    if (lane < merged_len)
    {
        set_key_node<key_type>(curr_node, lane + 1, O_keys[lane]);
        set_offset_node<key_type>(curr_node, lane + 1, O_offs[lane]);
    }

    // Update size and advance update_idx (lane0)
    if (lane == 0)
    {
        const smallsize new_size = merged_len;
        cg::set<smallsize>(curr_node, sizeof(key_type), new_size);
        update_idx += n; // consumed this run
    }
    tile.sync();
}

// Main idea: per-bucket, per-node do "merge insert" instead of repeated ballot+shift.
// Hook this into your existing outer chain traversal.
template <typename key_type>
DEVICEQUALIFIER void process_inserts_tile_bulk_only_optimized_merge(
    key_type bucket_max,
    smallsize minindex,
    smallsize maxindex,
    updatable_cg_params *__restrict__ launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const key_type *__restrict__ update_list,
    const smallsize *__restrict__ offset_list,
    smallsize update_size)
{
    (void)bucket_max;
    (void)update_size;

    uint8_t *__restrict__ allocation_buffer =
        static_cast<uint8_t *>(launch_params->allocation_buffer);

    const smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    const smallsize node_stride = launch_params->node_stride;
    const smallsize node_size = launch_params->node_size;
    const smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    const smallsize lane = tile.thread_rank();

    void *curr_node = starting_node;
    smallsize update_idx = minindex;

    key_type curr_max = cg::extract<key_type>(curr_node, 0);

    while (update_idx <= maxindex)
    {
        // Lane0 walks to the node that can contain update_list[update_idx]
        if (lane == 0)
        {
            key_type key = update_list[update_idx];
            while (curr_max < key)
            {
                const smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
                if (next_ptr == 0)
                {
                    // Keep your ERROR macro
                    ERROR_INSERTS("Invalid link in chain", key, /*tile_id*/ 0, lane);
                }
                curr_node = allocation_buffer + static_cast<size_t>(next_ptr - 1) * static_cast<size_t>(node_stride);
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        curr_max = tile.shfl(curr_max, 0);
        {
            uintptr_t raw = reinterpret_cast<uintptr_t>(curr_node);
            raw = tile.shfl(raw, 0);
            curr_node = reinterpret_cast<void *>(raw);
        }

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // If full, split then retry
        if (curr_size >= node_size)
        {
            process_split_tile<key_type>(tile, curr_node, launch_params);
            curr_max = cg::extract<key_type>(curr_node, 0);
            continue;
        }

        // Compute this node's update run end: keys <= curr_max
        smallsize run_end = update_idx;
        if (lane == 0)
        {
            run_end = upper_bound_device<key_type>(
                update_list,
                update_idx,
                static_cast<smallsize>(maxindex + 1),
                curr_max);
        }
        run_end = tile.shfl(run_end, 0);

        const smallsize run_len = run_end - update_idx;
        const smallsize free_space = node_size - curr_size;

        // Fast merge path only if it fits in one go and node_size <= TILE_SIZE and run_len <= TILE_SIZE.
        // Otherwise fall back to your current method (keep it as a slow path).
        const bool can_fast =
            (node_size <= TILE_SIZE) &&
            (run_len <= TILE_SIZE) &&
            (run_len <= free_space) &&
            (curr_size <= TILE_SIZE);

        if (!can_fast)
        {
            process_inserts_tile_bulk_only_optimized<key_type>(
                bucket_max,
                update_idx,
                maxindex,
                launch_params,
                curr_node,
                tile,
                update_list,
                offset_list,
                update_size);
            return;
        }

        // Merge insert for this node
        merge_insert_node_tile<key_type>(
            tile,
            curr_node,
            node_size,
            curr_size,
            update_list,
            offset_list,
            update_idx,
            run_end);

        // If node became full and more keys remain for this node, split.
        curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        if (curr_size == node_size && update_idx <= maxindex && update_list[update_idx] <= curr_max)
        {
            process_split_tile<key_type>(tile, curr_node, launch_params);
            curr_max = cg::extract<key_type>(curr_node, 0);
        }
    }
}

#endif

/*


    template <typename key_type>
    DEVICEQUALIFIER void process_inserts_tile_bulk_opt(
        key_type maxkey, smallsize minindex, smallsize maxindex,
        updatable_cg_params *launch_params,
        void *starting_node,
        coop_g::thread_block_tile<TILE_SIZE> tile)
{
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    smallsize partition_count = launch_params->partition_count_with_overflow;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize my_tid = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    smallsize total_keys = maxindex - minindex + 1;
    smallsize total_insert_count = 0;
    void *curr_node = starting_node;
    smallsize update_idx = minindex;
    key_type curr_max = cg::extract<key_type>(curr_node, 0);

    // if (tile_id == 1) DEBUG_PI_TILE_BULK("Before Loop", curr_max, minindex, maxindex);
    // tile.sync();

    while (update_idx <= maxindex)
    {
        key_type key = update_list[update_idx];
        smallsize offset = offset_list[update_idx];

        // if (key ==1248854015) DEBUG_PI_TILE_BULK("Insert Key", key, offset, update_idx);
        // if (key ==1248854015 ) DEBUG_PI_TILE_BULK("Thread Info", my_tid, tile_id, curr_max);

        if (tile.thread_rank() == 0)
        {
            while (curr_max < key)
            {
                smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count)
                {
                    ERROR_INSERTS("Invalid link in chain", key, tile_id, my_tid);
                }
                /// if (tile_id == 1) DEBUG_PI_TILE_BULK("Next Node", key, next_ptr, my_tid);
                curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + (next_ptr - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        curr_max = tile.shfl(curr_max, 0);
        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr = tile.shfl(raw_ptr, 0);
        curr_node = reinterpret_cast<void *>(raw_ptr);

        // tile.sync();

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // need to check if space in the node for new keys
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("Current Node Size", curr_size, node_size, my_tid);
        if (curr_size >= node_size)
        {
            // DEBUG_PI_TILE_BULK("Node size exceeded", key, curr_size, node_size);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);

            continue;
        }

        smallsize insert_index = 1;
        smallsize insert_count = 0;
        key_type testkey = 0;
        smallsize testoffset = 0 key_type my_key = 0;
        smallsize my_offset = 0;
        bool active = false;
        bool found = false;
        bool max_testkey = false;
        smallsize tid_test_key = node_size + 1;

        smallsize free_space = node_size - curr_size;

        if (my_tid < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, my_tid + 1);
            my_offset = extract_offset_node<key_type>(curr_node, my_tid + 1);
            if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
            {
                active = true;
                found = (my_key == key);
            }
        }
        tile.sync();

        bool key_found = tile.any(found);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("Post DupCheck", my_tid, found, key_found);
        // if (my_key ==2030567423 ) DEBUG_PI_TILE_BULK("MyKey", my_key, tile_id, key);
        // tile.sync();

        if (key_found)
        {
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP FOUND", key, my_tid, tile_id);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP MATCH", key, my_key, testkey);
            update_idx++;
            continue;
        }

        bool is_gt = active && (my_key > key);
        unsigned mask = tile.ballot(is_gt);

        if (mask != 0)
        {
            insert_index = __ffs(mask);
            testkey = tile.shfl(my_key, insert_index - 1);
            testoffset = tile.shfl(my_offset, insert_index - 1);
            tid_test_key = insert_index - 1;
        }
        else
        {
            insert_index = curr_size + 1;
            max_testkey = true;
            testkey = curr_max;
        }

        tile.sync();
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("InsertPos", insert_index, my_tid, is_gt);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("TestKey", testkey, curr_max, mask);

        if (my_tid == 0) // insert the first key
        {
            set_key_node<key_type>(curr_node, insert_index, key);
            set_offset_node<key_type>(curr_node, insert_index, offset);
            insert_count++;
            update_idx++;
            smallsize next_index = insert_index + 1;
        }
        tile.sync();
        next_index = tile.shfl(next_index, 0);
        update_idx = tile.shfl(update_idx, 0);
        insert_count = tile.shfl(insert_count, 0);
        key_type next_key = 0;
        bool need_new_testkey = false;

        while (update_idx <= maxindex && insert_count < free_space && next_index <= node_size)
        {
            next_key = update_list[update_idx];
            offset = offset_list[update_idx];
            // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", update_idx, key, testkey);
            // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", next_key, my_tid, tile_id);

            if (next_key == testkey && !max_testkey)
            {
                set_key_node<key_type>(curr_node, next_index, testkey); // write the old values
                set_offset_node<key_type>(curr_node, next_index, testoffset);

                tid_test_key++;
                next_index++;
                update_idx++;
                need_new_testkey = true;
            }
            else if (next_key > testkey)
            {
                // new testkey - next one over
                if (max_testkey)
                    break; // means all keys in the node are less than next_key. next_key belongs in next node.
                set_key_node<key_type>(curr_node, next_index, testkey);
                set_offset_node<key_type>(curr_node, next_index, testoffset);
                next_index++;
                tid_test_key++;
                need_new_testkey = true;
                // use same update_idx key.
            }
            else
            {
                key = next_key;

                // if (next_key ==1248854015) DEBUG_PI_TILE_BULK(" Performing INsert", key, offset, next_index);

                set_key_node<key_type>(curr_node, next_index, key);
                set_offset_node<key_type>(curr_node, next_index, offset);

                // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                insert_count++;
                update_idx++;
                next_index++;
                need_new_testkey = false;
            }

            if (!need_new_testkey)
                continue; // back to top

            if (update_idx > maxindex || next_index > node_size || tid_test_key >= node_size)
                break; // no more space in the node

            next_key = update_list[update_idx];
            if (next_key > curr_max)
                break;

            if (my_tid >= tid_test_key)
            {
                found = (my_key == next_key);
                is_gt = active && (my_key > next_key);
                mask = tile.ballot(is_gt);
            }

            if found --figure out what to do :) \\

            if (mask != 0)
                {
                    insert_index = __ffs(mask);
                    testkey = tile.shfl(my_key, insert_index - 1);
                    testoffset = tile.shfl(my_offset, insert_index - 1);
                    tid_test_key = insert_index - 1;
                }
            else
            {
                insert_index = curr_size + insert_count;
                max_testkey = true;
                testkey = curr_max;
            }

            testkey = tile.shfl(my_key, tid_test_key);
            testoffset = tile.shfl(my_offset, tid_test_key);

            while (next_index <= node_size && testkey > 0 && next_key > testkey)
            {
                set_key_node<key_type>(curr_node, next_index, testkey);
                set_offset_node<key_type>(curr_node, next_index, testoffset);

                tid_test_key++;
                testkey = tile.shfl(my_key, tid_test_key);
                testoffset = tile.shfl(my_offset, tid_test_key);
                next_index++;
            }
            // if (update_idx > maxindex) break;
            //     next_key = update_list[update_idx];

            // index_test_key++;
            // continue;
        }
        // if (next_key ==1248854015) DEBUG_PI_TILE_BULK("InsertDone", insert_count, curr_size, next_index);

        tile.sync();

        insert_count = tile.shfl(insert_count, 0);
        update_idx = tile.shfl(update_idx, 0);
        total_insert_count += insert_count;
        curr_size += insert_count;
        if (my_tid == 0)
        {
            cg::set<smallsize>(curr_node, sizeof(key_type), curr_size);
            // if (tile_id == 1) print_node<key_type>(curr_node, node_size);
        }

        curr_size = tile.shfl(curr_size, 0);

        // HEREE
        //---->  update_more_keys<key_type>(tile, curr_node, next_index, free_space, insert_index, insert_count, my_key, my_offset, curr_size, node_size);

        // if (tile_id == 1) DEBUG_PI_TILE_BULK("SizePost", my_tid, curr_size, insert_count);

        if (active && my_tid >= (insert_index - 1))
        {
            //  if (tile_id == 1) DEBUG_PI_TILE_BULK("Shift", my_tid, insert_index, node_size);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("ShiftVal", my_key, my_offset, curr_size);

            set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
            set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);
        }
        // if (my_tid==0 && tile_id == 1) DEBUG_PI_TILE_BULK("Post Full Update to Node", my_tid, curr_size, insert_count);
        // if (my_tid==0 && tile_id == 1) print_node<key_type>(curr_node, node_size);

        tile.sync();

        // if (curr_size > node_size || insert_count > free_space)
        // {
        //     ERROR_INSERTS("Overflow", key, curr_size, node_size);
        //}

        if (curr_size == node_size && update_idx < maxindex &&
            (total_keys - total_insert_count > 0) &&
            (update_list[update_idx] < curr_max))
        {
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("SplitTrigger", curr_size, update_list[update_idx], curr_max);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);
        }

        // tile.sync();
        // if (my_tid==0 && tile_id == 1) DEBUG_PI_TILE_BULK("END ONE ROUND", my_tid, tile_id);
    }

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();
    if (my_tid == 0 && tile_id == 1)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, my_tid);
    }
#endif
}

*/