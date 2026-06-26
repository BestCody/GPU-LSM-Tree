// =============================================================================
// File: coarse_granular_inserts_tiles.cuh
// Author: Rosina Kharal
// Description: Implements coarse_granular_inserts_tiles
//              One of the primary files for tiled insertions in FliX 
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_INSERTS_TILES_CUH
#define INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_INSERTS_TILES_CUH

//#include <cooperative_groups.h>
#include "coarse_granular_deletes.cuh"
#include "coarse_granular_inserts.cuh"
#include "coarse_granular_deletes_tiles.cuh"
#include "coarse_granular_inserts_tiles_bulk.cuh"
#include "definitions_updates.cuh"
#include "tile_utils.cuh"

//#define TILE_SIZE 8 //

//namespace coop_g = cooperative_groups;


// New: Single-threaded perform_shift_insert (no tile)
template <typename key_type>
DEVICEQUALIFIER void perform_shift_insert_single(
    void *curr_node,
    smallsize insert_index,
    smallsize num_elements,
    key_type insertkey,
    smallsize thisoffset)
{
    // Shift elements up by 1 position starting from the end
    for (smallsize i = num_elements; i >= insert_index; --i) {
        key_type key = extract_key_node<key_type>(curr_node, i);
        smallsize offset = extract_offset_node<key_type>(curr_node, i);

        set_key_node<key_type>(curr_node, i + 1, key);
        set_offset_node<key_type>(curr_node, i + 1, offset);
    }

    // Insert the new key and offset
    set_key_node<key_type>(curr_node, insert_index, static_cast<key_type>(insertkey));
    set_offset_node<key_type>(curr_node, insert_index, thisoffset);

    // Increment node size
    smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
    cg::set<smallsize>(curr_node, sizeof(key_type), curr_size + 1);
}
template <typename key_type>
DEVICEQUALIFIER void perform_shift_insert_tile_prevs(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    void *curr_node,
    smallsize insert_index,
    smallsize num_elements,
    key_type insertkey,
    smallsize thisoffset)
{
    smallsize tid = tile.thread_rank();

    key_type key = 0;
    smallsize offset = 0;

    // Each thread checks whether it needs to shift
    bool do_shift = (tid >= insert_index && tid <= num_elements);
    if (do_shift)
    {
        key = extract_key_node<key_type>(curr_node, tid);
        offset = extract_offset_node<key_type>(curr_node, tid);
    }

    tile.sync(); // All threads must reach this

    if (do_shift)
    {
        set_key_node<key_type>(curr_node, tid + 1, key);
        set_offset_node<key_type>(curr_node, tid + 1, offset);
    }

    tile.sync(); // Ensure shifts are done

    if (tid == insert_index)
    {
        set_key_node<key_type>(curr_node, insert_index, static_cast<key_type>(insertkey));
        set_offset_node<key_type>(curr_node, insert_index, thisoffset);

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        cg::set<smallsize>(curr_node, sizeof(key_type), curr_size + 1);
    }
}


template <typename key_type>
DEVICEQUALIFIER void process_insert_split_tile_B(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    void *curr_node,
    smallsize insert_index,
    key_type insertkey,
    smallsize thisoffset,
    updatable_cg_params *launch_params)
{
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    // static_assert(TILE_SIZE >= 16, "Tile size must be at least large enough to handle nodes.");

    smallsize t = tile.thread_rank();

    // Thread 0 allocates new node
    smallsize free_index = 0;
    if (t == 0)
    {
        free_index = atomicAdd(&(launch_params->free_node), 1ULL);
    }
    free_index = tile.shfl(free_index, 0); // Broadcast allocation result
    if (free_index >= allocation_buffer_count)
        return;

    auto linked_node = reinterpret_cast<uint8_t *>(allocation_buffer) + node_stride * free_index;

    smallsize half = node_size / 2;
    bool uneven = (node_size % 2 != 0);
    smallsize middle = half + (uneven ? 1 : 0);
    smallsize move_count = node_size - middle;

    // Copy from curr_node[middle + t] → linked_node[t + 1]
    key_type key = 0;
    smallsize offset = 0;
    bool copy_valid = (t < move_count);
    if (copy_valid)
    {
        key = extract_key_node<key_type>(curr_node, middle + t);
        offset = extract_offset_node<key_type>(curr_node, middle + t);
    }
    tile.sync();

    if (copy_valid)
    {
        set_key_node<key_type>(linked_node, t + 1, key);
        set_offset_node<key_type>(linked_node, t + 1, offset);
    }
    tile.sync();

    // Clear the slots from curr_node
    if (copy_valid)
    {
        set_key_node<key_type>(curr_node, middle + t, 0);
        set_offset_node<key_type>(curr_node, middle + t, 0);
    }
    tile.sync();

    if (t == 0)
    {
        cg::set<smallsize>(curr_node, sizeof(key_type), middle);
        cg::set<smallsize>(linked_node, sizeof(key_type), move_count);

        key_type new_curr_max = extract_key_node<key_type>(curr_node, middle);
        key_type old_curr_max = cg::extract<key_type>(curr_node, 0);

        printf("setting maxes in split tid:%d %llu %llu\n", t, new_curr_max, old_curr_max);

        print_node<key_type>(curr_node, node_size);
        print_node<key_type>(linked_node, node_size);

        cg::set<key_type>(curr_node, 0, new_curr_max);
        cg::set<key_type>(linked_node, 0, old_curr_max);

        smallsize existing_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
        cg::set<smallsize>(linked_node, lastposition_bytes, existing_ptr);
        cg::set<smallsize>(curr_node, lastposition_bytes, free_index + 1);
    }
    tile.sync();

    // Perform insert into correct node
    if (insert_index <= middle)
    {
        perform_shift_insert_tile<key_type>(tile, curr_node, insert_index, middle, insertkey, thisoffset);
    }
    else
    {
        smallsize adjusted_index = insert_index - middle;
        perform_shift_insert_tile<key_type>(tile, linked_node, adjusted_index, move_count, insertkey, thisoffset);
    }
    tile.sync();
}

template <typename key_type>
DEVICEQUALIFIER void process_insert_split_tile(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    void *curr_node,
    smallsize insert_index,
    key_type insertkey,
    smallsize thisoffset,
    updatable_cg_params *launch_params)
{
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    // static_assert(TILE_SIZE >= node_size, "Tile size must be at least large enough to handle nodes.");

    smallsize t = tile.thread_rank();

    // Thread 0 allocates new node
    smallsize free_index = 0;
    if (t == 0)
    {
        free_index = atomicAdd(&(launch_params->free_node), 1ULL);
    }
    free_index = tile.shfl(free_index, 0);
    if (free_index >= allocation_buffer_count)
        return;

    auto linked_node = reinterpret_cast<uint8_t *>(allocation_buffer) + node_stride * free_index;

    smallsize half = node_size / 2;          // e.g., 7 if node_size = 14
    smallsize move_count = node_size - half; // e.g., 7
    smallsize t_index = t;

    // Copy curr_node[half + 1 + t] → linked_node[t + 1]
    if (t_index < move_count)
    {
        key_type k = extract_key_node<key_type>(curr_node, half + 1 + t_index);
        smallsize off = extract_offset_node<key_type>(curr_node, half + 1 + t_index);
        set_key_node<key_type>(linked_node, t_index + 1, k);
        set_offset_node<key_type>(linked_node, t_index + 1, off);
    }
    tile.sync();

    // Clear those slots from curr_node
    if (t_index < move_count)
    {
        set_key_node<key_type>(curr_node, half + 1 + t_index, 0);
        set_offset_node<key_type>(curr_node, half + 1 + t_index, 0);
    }
    tile.sync();

    // Thread 0 sets sizes, maxes, and links
    if (t == 0)
    {
        cg::set<smallsize>(curr_node, sizeof(key_type), half);         // new size
        cg::set<smallsize>(linked_node, sizeof(key_type), move_count); // new size

        key_type new_curr_max = extract_key_node<key_type>(curr_node, half);
        key_type old_max = cg::extract<key_type>(curr_node, 0);

        cg::set<key_type>(curr_node, 0, new_curr_max);
        cg::set<key_type>(linked_node, 0, old_max);

        smallsize link_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
        cg::set<smallsize>(linked_node, lastposition_bytes, link_ptr);
        cg::set<smallsize>(curr_node, lastposition_bytes, free_index + 1);
    }
    tile.sync();

   // if (insertkey ==2377150882 && t == 0){
           // print_node<key_type>(curr_node, node_size);
      //    printf("SPlit FUnction FOUND KEY %llu and insertindex is %d \n", insertkey, insert_index);
      //    print_node<key_type>(curr_node, node_size);

      //  }
    // Insert in correct node
    if (insert_index <= half)
    { 
        
        perform_shift_insert_tile<key_type>(tile, curr_node, insert_index, half, insertkey, thisoffset);
    }
    else
    {
       // if (insertkey ==2377150882 && t == 0){
       //    // print_node<key_type>(curr_node, node_size);
       //   printf("FOUND KEY %llu\n", insertkey);
       //   print_node<key_type>(linked_node, node_size);

      //  }
        smallsize adjusted_index = insert_index - half; // (half + 1);  // skip over moved half and the middle key
        // perform_shift_insert_tile<key_type>(tile, linked_node, adjusted_index + 1, move_count, insertkey, thisoffset);
        perform_shift_insert_tile<key_type>(tile, linked_node, adjusted_index, move_count, insertkey, thisoffset);
    }
    tile.sync();
}


//---------------------------------------------------------------------
template <typename key_type>
DEVICEQUALIFIER bool single_search_cuda_buffer_tile_01(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const void *buffer,
    key_type search_key,
    smallsize num_elements,
    smallsize &insert_index)
{
    smallsize my_index = tile.thread_rank();
    key_type my_key = 0;
    bool active = false;
    bool found = false;

    if (my_index >= 1 && my_index <= num_elements)
    {
        my_key = extract_key_node<key_type>(buffer, my_index);
        if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
        {
            active = true;
            if (my_key == search_key)
            {
                found = true;
            }
        }
    }

    bool key_found = tile.any(found);

    if (key_found)
    {
        if (found)
        {
            insert_index = my_index;
        }

        // Broadcast the found insert_index to all tile threads
        smallsize result = tile.shfl(insert_index, tile.match_any(found));
        insert_index = result;
        return true;
    }

    bool is_gt = active && (my_key > search_key);
    unsigned mask = tile.ballot(is_gt);

    int first_gt_index = -1;
    for (int i = 0; i < tile.size(); ++i)
    {
        if (mask & (1 << i))
        {
            first_gt_index = i;
            break;
        }
    }

    insert_index = (first_gt_index != -1) ? first_gt_index : num_elements + 1;
    return false;
}

// Corrected Single-Threaded Split with Tiled Shift Insert

template <typename key_type>
DEVICEQUALIFIER void process_insert_split_tile_one_thread(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    void* curr_node,
    smallsize insert_index,
    key_type insertkey,
    smallsize thisoffset,
    updatable_cg_params* launch_params)
{
    uintptr_t linked_node_ptr = 0;
    bool early_exit = false;

    if (tile.thread_rank() == 0) {
        void* allocation_buffer = launch_params->allocation_buffer;
        smallsize node_stride = launch_params->node_stride;
        smallsize node_size = launch_params->node_size;
        smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
        smallsize partition_size = launch_params->partition_size;
        smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

        smallsize free_index = atomicAdd(&(launch_params->free_node), 1ULL);

        if (free_index >= allocation_buffer_count) {
            early_exit = true;
        } else {
            auto linked_node = reinterpret_cast<uint8_t*>(allocation_buffer) + node_stride * free_index;
            linked_node_ptr = reinterpret_cast<uintptr_t>(linked_node);

            const smallsize start_idx = partition_size + 1;
            const smallsize end_idx = partition_size * 2;

            // Move key-offset pairs from curr_node to linked_node
            for (smallsize i = start_idx; i <= end_idx; ++i) {
                key_type curr_key = extract_key_node<key_type>(curr_node, i);
                smallsize curr_offset = extract_offset_node<key_type>(curr_node, i);

                set_key_node<key_type>(curr_node, i, 0);
                set_offset_node<key_type>(curr_node, i, 0);

                set_key_node<key_type>(linked_node, i - start_idx + 1, curr_key);
                set_offset_node<key_type>(linked_node, i - start_idx + 1, curr_offset);
            }

            // Set new sizes
            cg::set<smallsize>(curr_node, sizeof(key_type), partition_size);
            cg::set<smallsize>(linked_node, sizeof(key_type), partition_size);

            // Update max values
            key_type curr_max = extract_key_node<key_type>(curr_node, partition_size);
            key_type original_max = cg::extract<key_type>(curr_node, 0);

            cg::set<key_type>(curr_node, 0, curr_max);
            cg::set<key_type>(linked_node, 0, original_max);

            // Handle linking
            smallsize last_link = cg::extract<smallsize>(curr_node, lastposition_bytes);
            cg::set<smallsize>(linked_node, lastposition_bytes, last_link);
            cg::set<smallsize>(curr_node, lastposition_bytes, free_index + 1);
        }
    }

    early_exit = tile.shfl(early_exit, 0);
    if (early_exit) return;

    linked_node_ptr = tile.shfl(linked_node_ptr, 0);
    void* linked_node = reinterpret_cast<void*>(linked_node_ptr);

    smallsize partition_size = launch_params->partition_size;

    if (insert_index < partition_size + 1) {
        perform_shift_insert_tile<key_type>(tile, curr_node, insert_index, partition_size, insertkey, thisoffset);
    } else {
        smallsize adjusted_index = insert_index - partition_size;
        perform_shift_insert_tile<key_type>(tile, linked_node, adjusted_index, partition_size, insertkey, thisoffset);
    }
}


template <typename key_type>
DEVICEQUALIFIER void process_inserts_tile(
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
    smallsize this_tid = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    int total_keys = maxindex - minindex + 1;
    void *curr_node = starting_node;
    //--> for (int k = tile.thread_rank(); k < total_keys; k += tile.size()) {
    for (smallsize i = minindex; i <= maxindex; ++i)
    {

        // print the tile_id and this_tid and print the key and minindex and maxindex using a printf statement
        // printf("TOP FOR LOOP: tile_id = %d, this_tid = %d, key = %llu, minindex = %llu, maxindex = %llu\n",
        //   tile_id, this_tid,
        //  static_cast<unsigned long long>(update_list[i]),
        //  static_cast<unsigned long long>(minindex),
        //  static_cast<unsigned long long>(maxindex));

        // --> smallsize i = minindex + k;
        // smallsize i = minindex + k;

        // ---> smallsize i = minindex + k;
        key_type key = update_list[i];
        smallsize offset = offset_list[i];

        // if (key ==2737346476 && this_tid ==0) print_node<key_type>(curr_node, node_size);

        // --> if (this_tid == 0 && tile_id==0) DEBUG_PI_TILE("TOP process_inserts_tile", key, offset);
        // --> if (this_tid == 0 && tile_id==0) print_node<key_type>(curr_node, node_size);
        // void* curr_node = starting_node;

        if (tile.thread_rank() == 0)
        {
            key_type curr_max = cg::extract<key_type>(curr_node, 0);
            while (curr_max < key)
            {
                smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);

                if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count)
                {
                    ERROR_INSERTS("ERROR: invalid link in node chain during insert", key, tile_id, this_tid);
                    // return;
                }
                // DEBUG_PI_TILE("Next Node", key, next_ptr, this_tid);
                curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + (next_ptr - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        // curr_node = tile.shfl(reinterpret_cast<uintptr_t>(curr_node), 0);
        // curr_node = reinterpret_cast<void*>(curr_node);

        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr = tile.shfl(raw_ptr, 0); // Broadcast pointer value from thread 0
        curr_node = reinterpret_cast<void *>(raw_ptr);

        tile.sync();

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        smallsize insert_index = 1;

        /// if (tile.thread_rank() == 0 ) DEBUG_PI_TILE("Before single search: ", key, curr_size ,tile_id);
        // if (tile.thread_rank() == 0 )print_node<key_type>(curr_node, node_size);

        bool found = single_search_cuda_buffer_tile<key_type>(tile, curr_node, key, curr_size, insert_index);
        
        /*
        if (tile.thread_rank() == 0 && key==1235571559 ) {
            printf("DEBUG_PI_TILE: After single search: %llu, Insert Index : %u, Tile ID: %u, found %u,  size of node %u \n",
            static_cast<unsigned long long>(key), insert_index, tile_id, found, curr_size);
            print_node<key_type>(curr_node, node_size);
        }
        */
       
        if (found)
        {
            continue;
        }

        key_type key_at_index = extract_key_node<key_type>(curr_node, insert_index);

        // if (tile.thread_rank() == 0) DEBUG_PI_TILE("Not Found", key, key_at_index, insert_index);

        if (key_at_index < key)
        {
            // insert_index = (curr_size == 0) ? 1 : insert_index + 1;
            insert_index = (curr_size == 0) ? 1 : insert_index;
        }

        if (curr_size < node_size)
        {
            /* if (tile.thread_rank() == 0 && key==1235571559 ) {
                printf("DEBUG_PI_TILE: Before shift insert | Key: %llu, Insert Index: %u, Tile ID: %u, Thread ID: %u\n",static_cast<unsigned long long>(key), insert_index, tile_id, this_tid);
            } 
            */
            perform_shift_insert_tile<key_type>(tile, curr_node, insert_index, curr_size, key, offset);
           
            /* if (tile.thread_rank() == 0 && key==1235571559 ) {
                printf("DEBUG_PI_TILE: After shift insert | Key: %llu, Insert Index: %u, Tile ID: %u, Thread ID: %u\n",static_cast<unsigned long long>(key), insert_index, tile_id, this_tid);
                print_node<key_type>(curr_node, node_size);
            }*/
        }
        else
        {
            
            process_insert_split_tile<key_type>(tile, curr_node, insert_index, key, offset, launch_params);
    
        }

        // tile.sync();
    }

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads(); // Only valid if ALL threads in block reach this point
    if (this_tid == 0 && tile_id == 0 && blockIdx.x == 0)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, this_tid);
    }
#endif
}
// file: update_kernel_tile.cuh




template <typename key_type>
GLOBALQUALIFIER void update_kernel_tile_inserts_new(
    updatable_cg_params* __restrict__ launch_params,
    const key_type* __restrict__ update_list,
    const smallsize* __restrict__ offsets,
    smallsize update_size)
{
    const key_type* __restrict__ maxbuf =
        static_cast<const key_type*>(launch_params->maxvalues);

    const smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    const int tile_id = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / static_cast<int>(tile.size())) +
                        static_cast<int>(tile.meta_group_rank());
    const smallsize this_tid = tile.thread_rank();


    if (tile_id >= (partition_count_with_overflow)) return;

    const key_type maxkey = maxbuf[tile_id];
    const key_type minkey = (tile_id > 0) ? static_cast<key_type>(maxbuf[tile_id - 1] + 1) : static_cast<key_type>(1);

    int minindex = -1;
    int maxindex = -1;



    if (tile.thread_rank() == 0)
    {

        minindex = binarySearchIndex<key_type>(update_list, minkey, 0, update_size, false);
        //maxindex = binarySearchIndex<key_type>(update_list,
    }

                minindex = tile.shfl(minindex, 0);
                const bool skip_tile = (minindex >= update_size || minindex == -1);


    // After shfl, all lanes observe the same values -> tile.any() adds overhead.
   // const bool skip_tile = (minindex > maxindex || minindex == -1 || maxindex == -1);
    if (skip_tile) return;

    uint8_t* __restrict__ curr_node =
        static_cast<uint8_t*>(launch_params->ordered_node_pairs) +
        static_cast<size_t>(tile_id) * static_cast<size_t>(launch_params->node_stride);

//#ifdef defined(INSERTS_TILE_BULK_ONLY)
//#pragma message "TILE INSERTS_BULK_ONLY_OPTIMIZED_2=YES"

//printf("Launching process_inserts_tile_bulk_only_optimized_2 for tile_id %d, minindex %d, maxindex %d, update_size %d\n",
//    tile_id, minindex, maxindex, update_size);


   process_inserts_tile_bulk_only_optimized_2<key_type>(
      //process_INSERTS_TILE_BULK_ONLY_hybrid<key_type>(
        maxkey,
        (smallsize)minindex,
        (smallsize)maxindex,
        launch_params,
        curr_node,
        tile,
        update_list,
        offsets,
        update_size);

//#else
//#pragma message "TILE INSERTS_BULK_=NO"
//    process_inserts_tile<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile);
//#endif

}


template <typename key_type>
GLOBALQUALIFIER void update_kernel_tile_inserts(
    updatable_cg_params* __restrict__ launch_params,
    const key_type* __restrict__ update_list,
    const smallsize* __restrict__ offsets,
    smallsize update_size)
{
    const key_type* __restrict__ maxbuf =
        static_cast<const key_type*>(launch_params->maxvalues);

    const smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    const int tile_id = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / static_cast<int>(tile.size())) +
                        static_cast<int>(tile.meta_group_rank());
    const smallsize this_tid = tile.thread_rank();

#ifdef PRINT_UPDATE_VALUES
    if ((this_tid == 0) && (tile_id == 0))
    {
        DEBUG_PI_TILE(" In Update Kernel TILES ", tile_id, this_tid);

        printf("in UKT : Update list values: [");
        for (int i = 0; i < update_size; ++i)
        {
            if (i > 0) printf(", ");
            printf("%llu", static_cast<unsigned long long>(update_list[i]));
        }
        printf("]\n");
    }
#endif

    if (tile_id >= (partition_count_with_overflow)) return;

    const key_type maxkey = maxbuf[tile_id];
    const key_type minkey = (tile_id > 0) ? static_cast<key_type>(maxbuf[tile_id - 1] + 1) : static_cast<key_type>(1);

    int minindex = -1;
    int maxindex = -1;



    if (tile.thread_rank() == 0)
    {
    #if !defined(INSERTS_TILE_BULK_NOT_NEEDED)
        minindex = binarySearchIndex<key_type>(update_list, minkey, 0, update_size, false);
        maxindex = binarySearchIndex<key_type>(update_list, maxkey, 0, update_size, true);
    #else
        minindex = binarySearchIndex<key_type>(update_list, minkey, 0, update_size, false);
        //maxindex = binarySearchIndex<key_type>(update_list,

    #endif
    }

    #if !defined(INSERTS_TILE_BULK_NOT_NEEDED)
                minindex = tile.shfl(minindex, 0);
                maxindex = tile.shfl(maxindex, 0);
                const bool skip_tile = (minindex > maxindex || minindex == -1 || maxindex == -1);

    #else
                minindex = tile.shfl(minindex, 0);
                const bool skip_tile = (minindex >= update_size || minindex == -1);

                // maxindex = tile.shfl(maxindex, 0);
    #endif      // Perform a modified search to find maxindex


    // After shfl, all lanes observe the same values -> tile.any() adds overhead.
   // const bool skip_tile = (minindex > maxindex || minindex == -1 || maxindex == -1);
    if (skip_tile) return;

    uint8_t* __restrict__ curr_node =
        static_cast<uint8_t*>(launch_params->ordered_node_pairs) +
        static_cast<size_t>(tile_id) * static_cast<size_t>(launch_params->node_stride);

#ifdef INSERTS_TILE_BULK
#pragma message "TILE INSERTS_BULK_=YES"
    process_inserts_tile_bulk<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile);
#elif defined(INSERTS_TILE_BULK_HYBRID)
#pragma message "TILE INSERTS_BULK_HYBRID=YES"
    process_inserts_tile_bulk_hybrid<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile);
#elif defined(INSERTS_TILE_BULK_ONLY)
#pragma message "TILE INSERTS_BULK_ONLY=YES"
   process_inserts_tile_bulk_only_optimized<key_type>(
      //process_INSERTS_TILE_BULK_ONLY_hybrid<key_type>(
        maxkey,
        (smallsize)minindex,
        (smallsize)maxindex,
        launch_params,
        curr_node,
        tile,
        update_list,
        offsets,
        update_size);
#else
#pragma message "TILE INSERTS_BULK_=NO"
    process_inserts_tile<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile);
#endif
}


template <typename key_type>
GLOBALQUALIFIER void update_kernel_tile(updatable_cg_params *launch_params, const key_type *__restrict__ update_list,const smallsize *__restrict__ offsets,
     smallsize update_size)
{

  //  smallsize update_size = launch_params->update_size;
   // const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const key_type *maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    const smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    int tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();
    smallsize this_tid = tile.thread_rank();

   

    // DEBUG_PI_TILE("UKT:TOP OF Kernel: TIles and IDs",tile_id, this_tid);

#ifdef PRINT_UPDATE_VALUES
    if ((this_tid == 0) && (tile_id == 0))
    {
        DEBUG_PI_TILE(" In Update Kernel TILES ", tile_id, this_tid);

        printf("in UKT : Update list values: [");
        for (int i = 0; i < update_size; ++i)
        {
            if (i > 0)
            {
                printf(", ");
            }
            printf("%llu", static_cast<unsigned long long>(update_list[i]));
        }
        printf("]\n");
    }
#endif

    if (tile_id >= partition_count_with_overflow)
        return;

    key_type maxkey = maxbuf[tile_id];
    key_type minkey = (tile_id > 0)
                          ? maxbuf[tile_id - 1] + 1
                          : 1;
    // DEBUG_PI_TILE("UKT:After minkey maxkey", maxkey, tile_id, this_tid);

    int minindex = -1;
    int maxindex = -1;
    //  DEBUG_PI_TILE("UKT Before binary search Index", minindex, tile_id, this_tid);

    if (tile.thread_rank() == 0)
    {
        minindex = binarySearchIndex<key_type>(update_list, minkey, 0, update_size, false);
        maxindex = binarySearchIndex<key_type>(update_list, maxkey, 0, update_size, true);
    }

    minindex = tile.shfl(minindex, 0);
    maxindex = tile.shfl(maxindex, 0);

    // DEBUG_PI_TILE("UKT:After binary search", maxindex, tile_id, this_tid);

    // tile.any();

    bool skip_tile = (minindex > maxindex || minindex == -1 || maxindex == -1);
    if (tile.any(skip_tile))
    {
        return; // Exit early if any thread in the tile determines the tile should be skipped
    }
    // if (coop_g::any(tile, skip_tile)) return;
    //  -->  unsigned mask = __ballot_sync(0xffffffff, skip_tile);
    //  --->if (mask != 0) return;

    //if (minindex > maxindex || minindex == -1 || maxindex == -1)
      //  return;

    auto curr_node = reinterpret_cast<uint8_t *>(launch_params->ordered_node_pairs) + tile_id * launch_params->node_stride;
   // smallsize currnodesize = cg::extract<smallsize>(curr_node, sizeof(key_type));
   // key_type currnodeMax = cg::extract<key_type>(curr_node, 0);

    //if (tile_id == 0) DEBUG_PI_TILE_DELS("Current node max:, minindex , maxindex  ", currnodeMax, minindex, maxindex);
   /* if (perform_dels)
    {
       // DEBUG_PI_TILE("Before process_del_tile", maxkey, minkey, tile_id);
       #ifdef DELETES_TILE_BULK
       #pragma message "TILE DELETES_TILE_BULK=YES"
        process_deletes_tile_bulk_clean<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile);
       #else
       #pragma message "TILE DELETES REGULAR MERGE=YES"
        process_deletes_tile_merge<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile);
       #endif
    }
    else
    { */
        // DEBUG_PI_TILE("Before process_inserts_tile", minindex, maxindex, tile_id);
        //DEBUG_PI_TILE("Before process_inserts_tile", maxkey, minkey, tile_id);
        #ifdef INSERTS_TILE_BULK
        #pragma message "TILE INSERTS_BULK_=YES"
            process_inserts_tile_bulk<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile);
        #elif defined(INSERTS_TILE_BULK_HYBRID)
        #pragma message "TILE INSERTS_BULK_HYBRID=YES"
            process_inserts_tile_bulk_hybrid<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile);
        #elif defined(INSERTS_TILE_BULK_ONLY)
        #pragma message "TILE INSERTS_BULK_ONLY=YES"
        //printf("going into hybrid programs \n");
            process_INSERTS_TILE_BULK_ONLY<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile, update_list, offsets, update_size);
       
        #else
         #pragma message "TILE INSERTS_BULK_=NO"
            process_inserts_tile<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile);
        #endif
    //}
}


// removed merged insert - delete kernels 
template <typename key_type>
GLOBALQUALIFIER void update_kernel_tile_deletes(updatable_cg_params *launch_params)
{

    smallsize update_size = launch_params->update_size;
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const key_type *maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    const smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    int tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();
    smallsize this_tid = tile.thread_rank();

    // printf(" TOP Reuse list values at start of UKT: [");
    //smallsize *reuse_list = static_cast<smallsize *>(launch_params->reuse_list);



#ifdef PRINT_UPDATE_VALUES
    if ((this_tid == 0) && (tile_id == 0))
    {
        DEBUG_PI_TILE(" In Update Kernel TILES ", tile_id, this_tid);

        printf("in UKT : Update list values: [");
        for (int i = 0; i < update_size; ++i)
        {
            if (i > 0)
            {
                printf(", ");
            }
            printf("%llu", static_cast<unsigned long long>(update_list[i]));
        }
        printf("]\n");
    }
#endif

    if (tile_id >= partition_count_with_overflow)
        return;

    key_type maxkey = maxbuf[tile_id];
    key_type minkey = (tile_id > 0)
                          ? maxbuf[tile_id - 1] + 1
                          : 1;
    // DEBUG_PI_TILE("UKT:After minkey maxkey", maxkey, tile_id, this_tid);

    int minindex = -1;
    int maxindex = -1;
    //  DEBUG_PI_TILE("UKT Before binary search Index", minindex, tile_id, this_tid);

    if (tile.thread_rank() == 0)
    {
        minindex = binarySearchIndex<key_type>(update_list, minkey, 0, update_size, false);
       // maxindex = binarySearchIndex<key_type>(update_list, maxkey, 0, update_size, true);
    }

    minindex = tile.shfl(minindex, 0);
   // maxindex = tile.shfl(maxindex, 0);

    // DEBUG_PI_TILE("UKT:After binary search", maxindex, tile_id, this_tid);

    // tile.any();

    bool skip_tile = (minindex > (update_size -1) || minindex == -1 );
    if (tile.any(skip_tile))
    {
        return; // Exit early if any thread in the tile determines the tile should be skipped
    }
    

    auto curr_node = reinterpret_cast<uint8_t *>(launch_params->ordered_node_pairs) + tile_id * launch_params->node_stride;
   // smallsize currnodesize = cg::extract<smallsize>(curr_node, sizeof(key_type));
   // key_type currnodeMax = cg::extract<key_type>(curr_node, 0);

    //if (tile_id == 0) DEBUG_PI_TILE_DELS("Current node max:, minindex , maxindex  ", currnodeMax, minindex, maxindex);
   // if (perform_dels)
   // {
       // DEBUG_PI_TILE("Before process_del_tile", maxkey, minkey, tile_id);
       #ifdef DELETES_TILE_BULK
       #pragma message "TILE DELETES_TILE_BULK=YES"
        process_deletes_tile_bulk_clean_BM<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile);
       #else
       #pragma message "TILE DELETES REGULAR MERGE=YES"
        process_deletes_tile_merge<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile);
       #endif
   // }
     /* else
    {
        // DEBUG_PI_TILE("Before process_inserts_tile", minindex, maxindex, tile_id);
        DEBUG_PI_TILE("Before process_inserts_tile", maxkey, minkey, tile_id);
        #ifdef INSERTS_TILE_BULK
        #pragma message "TILE INSERTS_BULK_=YES"
            process_inserts_tile_bulk<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile);
        #elif defined(INSERTS_TILE_BULK_HYBRID)
        #pragma message "TILE INSERTS_BULK_HYBRID=YES"
            process_inserts_tile_bulk_hybrid<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile);
        #elif defined(INSERTS_TILE_BULK_ONLY)
        #pragma message "TILE INSERTS_BULK_OPT2=YES"
        //printf("going into hybrid programs \n");
            process_INSERTS_TILE_BULK_ONLY_hybrid<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile);
       
        #else
         #pragma message "TILE INSERTS_BULK_=NO"
            process_inserts_tile<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, tile);
        #endif
    } */
}

// file: update_kernel_tile_deletes.cuh

//#include <cooperative_groups.h>
//namespace coop_g = cooperative_groups;

template <typename key_type>
GLOBALQUALIFIER void update_kernel_tile_deletes_new(updatable_cg_params* __restrict__ launch_params,const key_type *__restrict__ update_list, smallsize update_size)
{
   // const smallsize update_size_ss = launch_params->update_size;
    if (update_size <= 0) return;

    //const int update_size = static_cast<int>(update_size_ss);

   // const key_type* __restrict__ update_list =
   //     static_cast<const key_type*>(launch_params->update_list);
    const key_type* __restrict__ maxbuf =
        static_cast<const key_type*>(launch_params->maxvalues);

    const smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    const int tiles_per_block = static_cast<int>(blockDim.x) / static_cast<int>(tile.size());
    const int tile_id = static_cast<int>(blockIdx.x) * tiles_per_block + static_cast<int>(tile.meta_group_rank());
    const int lane = static_cast<int>(tile.thread_rank());

#ifdef PRINT_UPDATE_VALUES
    if (lane == 0 && tile_id == 0) {
        DEBUG_PI_TILE(" In Update Kernel TILES ", tile_id, static_cast<smallsize>(lane));
        printf("in UKT : Update list values: [");
        for (int i = 0; i < update_size; ++i) {
            if (i) printf(", ");
            printf("%llu", static_cast<unsigned long long>(update_list[i]));
        }
        printf("]\n");
    }
#endif

    if (tile_id >= static_cast<int>(partition_count_with_overflow)) return;

    const key_type maxkey = maxbuf[tile_id];
    const key_type minkey =
        (tile_id > 0)
            ? static_cast<key_type>(maxbuf[tile_id - 1] + static_cast<key_type>(1))
            : static_cast<key_type>(1);

    int minindex = -1;
    int maxindex = -1;

    if (lane == 0) {
        minindex = binarySearchIndex<key_type>(update_list, minkey, 0, update_size, false);
       // maxindex = binarySearchIndex<key_type>(update_list, maxkey, 0, update_size_ss, true);
    }

    minindex = tile.shfl(minindex, 0);
   // maxindex = tile.shfl(maxindex, 0);

    // After shfl, all lanes see identical minindex/maxindex -> no need for tile.any().
    const bool skip_tile = (minindex > (update_size -1) || minindex == -1 );

    if (skip_tile) return;

    uint8_t* __restrict__ curr_node =
        static_cast<uint8_t*>(launch_params->ordered_node_pairs) +
        static_cast<size_t>(launch_params->node_stride) * static_cast<size_t>(tile_id);

#ifdef DELETES_TILE_BULK
#pragma message "TILE DELETES_TILE_BULK=YES"
    process_deletes_tile_bulk_clean_BM_new<key_type>(
        update_list,
        update_size,
        maxkey,
        static_cast<smallsize>(minindex),
        static_cast<smallsize>(maxindex),
        launch_params,
        curr_node,
        tile
    );
#else
#pragma message "TILE DELETES REGULAR MERGE=YES"
   
//printf("Tile --Shift Left going into regular merge deletes \n");
process_deletes_tile_merge<key_type>(
        update_list,
        update_size,
        maxkey,
        static_cast<smallsize>(minindex),
        static_cast<smallsize>(maxindex),
        launch_params,
        curr_node,
        tile
    );
#endif
}

#endif

/*
unsigned tile_mask = (1u << tile.size()) - 1;  // e.g., 0xFFFF for 16 threads
unsigned full_mask = __activemask(); // active threads in warp (usually 0xFFFFFFFF)
unsigned ballot = __ballot_sync(full_mask, skip_tile);
unsigned tile_ballot = ballot & tile_mask;

if (tile_ballot != 0) return;



template <typename key_type>
DEVICEQUALIFIER void process_insert_split_tile_01(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    void* curr_node,
    smallsize insert_index,
    key_type insertkey,
    smallsize thisoffset,
    updatable_cg_params* launch_params)
{
    void* allocation_buffer = launch_params->allocation_buffer;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    static_assert(TILE_SIZE >= 16, "Tile size must be at least large enough to handle nodes.");

    // Thread 0 allocates new node
    smallsize free_index = 0;
    if (tile.thread_rank() == 0) {
        free_index = atomicAdd(&(launch_params->free_node), 1ULL);
    }
    free_index = tile.shfl(free_index, 0);
    if (free_index >= allocation_buffer_count) return;

    auto linked_node = reinterpret_cast<uint8_t*>(allocation_buffer) + node_stride * free_index;

    smallsize half = node_size / 2;
    bool uneven = (node_size % 2 != 0);
    smallsize curr_size = node_size;

    // Copy from curr_node[half + 1 + t] → linked_node[t + 1]
    smallsize t = tile.thread_rank();
    if (t < node_size - half) {
        key_type key = extract_key_node<key_type>(curr_node, half + 1 + t);
        smallsize offset = extract_offset_node<key_type>(curr_node, half + 1 + t);
        set_key_node<key_type>(linked_node, t + 1, key);
        set_offset_node<key_type>(linked_node, t + 1, offset);
    }
    tile.sync();

    // Clear from curr_node[half + 1 + t]
    if (t < node_size - half) {
        set_key_node<key_type>(curr_node, half + 1 + t, 0);
        set_offset_node<key_type>(curr_node, half + 1 + t, 0);
    }
    tile.sync();

    // Thread 0 updates sizes, maxes, link ptrs
    if (tile.thread_rank() == 0) {
        cg::set<smallsize>(curr_node, sizeof(key_type), half + (uneven ? 1 : 0));
        cg::set<smallsize>(linked_node, sizeof(key_type), node_size - (half + (uneven ? 1 : 0)));

        key_type new_curr_max = extract_key_node<key_type>(curr_node, half + (uneven ? 1 : 0));
        key_type old_curr_max = cg::extract<key_type>(curr_node, 0);

        cg::set<key_type>(curr_node, 0, new_curr_max);
        cg::set<key_type>(linked_node, 0, old_curr_max);

        smallsize existing_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
        cg::set<smallsize>(linked_node, lastposition_bytes, existing_ptr);
        cg::set<smallsize>(curr_node, lastposition_bytes, free_index + 1);
    }
    tile.sync();

    // Determine if insert should go in curr_node or linked_node
    if (insert_index <= half + (uneven ? 1 : 0)) {
        perform_shift_insert_tile<key_type>(tile, curr_node, insert_index, half + (uneven ? 1 : 0), insertkey, thisoffset);
    } else {
        smallsize adjusted_index = insert_index - (half + (uneven ? 1 : 0));
        perform_shift_insert_tile<key_type>(tile, linked_node, adjusted_index, node_size - (half + (uneven ? 1 : 0)), insertkey, thisoffset);
    }
    tile.sync();
}
*/
/*

// New version: split handled by a single thread, but use tile for shift insert
template <typename key_type>
DEVICEQUALIFIER void process_insert_split_tile_one_thread(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    void* curr_node,
    smallsize insert_index,
    key_type insertkey,
    smallsize thisoffset,
    updatable_cg_params* launch_params)
{
    if (tile.thread_rank() == 0) {
        void* allocation_buffer = launch_params->allocation_buffer;
        smallsize node_stride = launch_params->node_stride;
        smallsize node_size = launch_params->node_size;
        smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
        smallsize lastposition_bytes = node_size - sizeof(smallsize);

        smallsize free_index = atomicAdd(&(launch_params->free_node), 1ULL);
        if (free_index >= allocation_buffer_count) return;

        auto linked_node = reinterpret_cast<uint8_t*>(allocation_buffer) + node_stride * free_index;

        smallsize half = node_size / 2;
        smallsize move_count = node_size - half;

        for (smallsize i = 0; i < move_count; ++i) {
            key_type k = cg::extract<key_type>(curr_node, (half + 1 + i) * sizeof(key_type));
            smallsize off = cg::extract<smallsize>(curr_node, (half + 1 + i) * sizeof(key_type) + sizeof(key_type));

            cg::set<key_type>(linked_node, (i + 1) * sizeof(key_type), k);
            cg::set<smallsize>(linked_node, (i + 1) * sizeof(key_type) + sizeof(key_type), off);

            cg::set<key_type>(curr_node, (half + 1 + i) * sizeof(key_type), 0);
            cg::set<smallsize>(curr_node, (half + 1 + i) * sizeof(key_type) + sizeof(key_type), 0);
        }

        cg::set<smallsize>(curr_node, sizeof(key_type), half);
        cg::set<smallsize>(linked_node, sizeof(key_type), move_count);

        key_type new_curr_max = cg::extract<key_type>(curr_node, half * sizeof(key_type));
        key_type old_max = cg::extract<key_type>(curr_node, 0);

        cg::set<key_type>(curr_node, 0, new_curr_max);
        cg::set<key_type>(linked_node, 0, old_max);

        smallsize link_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
        cg::set<smallsize>(linked_node, lastposition_bytes, link_ptr);
        cg::set<smallsize>(curr_node, lastposition_bytes, free_index + 1);
    }
    tile.sync();

    void* allocation_buffer = launch_params->allocation_buffer;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;
    smallsize half = node_size / 2;
    void* linked_node = reinterpret_cast<uint8_t*>(allocation_buffer) + node_stride * (tile.thread_rank() / TILE_SIZE);

if (insert_index <= half)
{
    perform_shift_insert_tile<key_type>(tile, curr_node, insert_index, half, insertkey, thisoffset);
}
else
{
    smallsize adjusted_index = insert_index - half;
    perform_shift_insert_tile<key_type>(tile, linked_node, adjusted_index, node_size - half, insertkey, thisoffset);
}
}
//////****** ONE THREAD */

// New version: split handled by a single thread, but use tile for shift insert