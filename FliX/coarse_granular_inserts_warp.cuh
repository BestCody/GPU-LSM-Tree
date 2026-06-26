// =============================================================================
// File: coarse_granular_inserts_warp.cuh
// Author: Rosina Kharal
// Description: Implements coarse_granular_inserts_warp
//              Testing purposes
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_INSERTS_WARP_CUH
#define INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_INSERTS_WARP_CUH

#include <cooperative_groups.h>
#include "coarse_granular_deletes.cuh"
#include "coarse_granular_inserts.cuh"
// Warp-level key search (no tile)


template <typename key_type>
DEVICEQUALIFIER void perform_shift_insert_warp(
    int lane_id,
    void *curr_node,
    smallsize insert_index,
    smallsize num_elements,
    key_type insertkey,
    smallsize thisoffset)
{
    key_type key = 0;
    smallsize offset = 0;

    // Early Exit
    if(insert_index == WARP_SIZE )
    {
        if (lane_id == WARP_SIZE -1){
            set_key_node<key_type>(curr_node, insert_index, static_cast<key_type>(insertkey));
            set_offset_node<key_type>(curr_node, insert_index, thisoffset);
    
            smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
            cg::set<smallsize>(curr_node, sizeof(key_type), curr_size + 1);
        }
            __syncwarp();
            return;
    }


    bool do_shift = (lane_id >= insert_index && lane_id <= num_elements);

    if (do_shift)
    {
        key = extract_key_node<key_type>(curr_node, lane_id);
        offset = extract_offset_node<key_type>(curr_node, lane_id);
    }

    __syncwarp();

    if (do_shift)
    {
        set_key_node<key_type>(curr_node, lane_id + 1, key);
        set_offset_node<key_type>(curr_node, lane_id + 1, offset);
    }

    __syncwarp();

    if (lane_id == insert_index)
    {
        set_key_node<key_type>(curr_node, insert_index, static_cast<key_type>(insertkey));
        set_offset_node<key_type>(curr_node, insert_index, thisoffset);

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        cg::set<smallsize>(curr_node, sizeof(key_type), curr_size + 1);
    }
}

// Warp-level split
template <typename key_type>
DEVICEQUALIFIER void process_insert_split_warp(
    int lane_id,
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

    smallsize free_index = 0;
    uintptr_t linked_node_ptr = 0;

    if (lane_id == 0)
    {
        free_index = atomicAdd(&(launch_params->free_node), 1ULL);
        if (free_index >= allocation_buffer_count)
            return;
        void *linked_node = reinterpret_cast<uint8_t *>(allocation_buffer) + node_stride * free_index;
        linked_node_ptr = reinterpret_cast<uintptr_t>(linked_node);
    }

    linked_node_ptr = __shfl_sync(0xffffffff, linked_node_ptr, 0);
    void *linked_node = reinterpret_cast<void *>(linked_node_ptr);

    smallsize half = node_size / 2;
    smallsize move_count = node_size - half;

    if (lane_id < move_count)
    {
        key_type k = extract_key_node<key_type>(curr_node, half + 1 + lane_id);
        smallsize off = extract_offset_node<key_type>(curr_node, half + 1 + lane_id);
        set_key_node<key_type>(linked_node, lane_id + 1, k);
        set_offset_node<key_type>(linked_node, lane_id + 1, off);
    }

    __syncwarp();

    if (lane_id < move_count)
    {
        set_key_node<key_type>(curr_node, half + 1 + lane_id, 0);
        set_offset_node<key_type>(curr_node, half + 1 + lane_id, 0);
    }

    __syncwarp();

    if (lane_id == 0)
    {
        cg::set<smallsize>(curr_node, sizeof(key_type), half);
        cg::set<smallsize>(linked_node, sizeof(key_type), move_count);

        key_type new_curr_max = extract_key_node<key_type>(curr_node, half);
        key_type old_max = cg::extract<key_type>(curr_node, 0);

        cg::set<key_type>(curr_node, 0, new_curr_max);
        cg::set<key_type>(linked_node, 0, old_max);

        smallsize link_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
        cg::set<smallsize>(linked_node, lastposition_bytes, link_ptr);
        cg::set<smallsize>(curr_node, lastposition_bytes, free_index + 1);
    }

    __syncwarp();

    //if (insertkey == 2377150882 && lane_id == 0)
    //{
    //    printf("Split Function FOUND KEY %u and insertindex is: %u \n", insertkey, insert_index);
       //print_node<key_type>(curr_node, node_size);
   // }

    if (insert_index <= half)
    {

        //if (insertkey == 3504328163 && lane_id == 0)
        //{
        //         printf("Split: Going to CURRe FOUND KEY %u and insert_index is: %u \n", insertkey, insert_index);
        //         print_node<key_type>(curr_node, node_size);
        //}
        perform_shift_insert_warp<key_type>(lane_id, curr_node, insert_index, half, insertkey, thisoffset);
    }
    else
    {
        smallsize adjusted_index = insert_index - half;
        // if (insertkey == 3504328163 && lane_id == 0)
       // {
       //          printf("Split: Going to Linked Node FOUND KEY %u and adjsuted_index is: %u \n", insertkey, adjusted_index);
       //          print_node<key_type>(linked_node, node_size);
       // }

        perform_shift_insert_warp<key_type>(lane_id, linked_node, adjusted_index, node_size - half, insertkey, thisoffset);
    }
}

// Warp-level split

template <typename key_type>
DEVICEQUALIFIER void process_insert_split_warp_single_thread(
    int lane_id,
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

    smallsize free_index = 0;
    uintptr_t linked_node_ptr = 0;

    if (lane_id == 0)
    {
        free_index = atomicAdd(&(launch_params->free_node), 1ULL);
        if (free_index >= allocation_buffer_count)
            return;
        void *linked_node = reinterpret_cast<uint8_t *>(allocation_buffer) + node_stride * free_index;
        linked_node_ptr = reinterpret_cast<uintptr_t>(linked_node);

        smallsize half = node_size / 2;
        smallsize move_count = node_size - half;

        for (smallsize i = 0; i < move_count; ++i)
        {
            key_type k = extract_key_node<key_type>(curr_node, half + 1 + i);
            smallsize off = extract_offset_node<key_type>(curr_node, half + 1 + i);

            set_key_node<key_type>(linked_node, i + 1, k);
            set_offset_node<key_type>(linked_node, i + 1, off);

            set_key_node<key_type>(curr_node, half + 1 + i, 0);
            set_offset_node<key_type>(curr_node, half + 1 + i, 0);
        }

        cg::set<smallsize>(curr_node, sizeof(key_type), half);
        cg::set<smallsize>(linked_node, sizeof(key_type), move_count);

        key_type new_curr_max = extract_key_node<key_type>(curr_node, half);
        key_type old_max = cg::extract<key_type>(curr_node, 0);

        cg::set<key_type>(curr_node, 0, new_curr_max);
        cg::set<key_type>(linked_node, 0, old_max);

        smallsize link_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
        cg::set<smallsize>(linked_node, lastposition_bytes, link_ptr);
        cg::set<smallsize>(curr_node, lastposition_bytes, free_index + 1);
    }

    linked_node_ptr = __shfl_sync(0xffffffff, linked_node_ptr, 0);
    void *linked_node = reinterpret_cast<void *>(linked_node_ptr);

    smallsize half = node_size / 2;

    if (insert_index <= half)
    {
        perform_shift_insert_warp<key_type>(lane_id, curr_node, insert_index, half, insertkey, thisoffset);
    }
    else
    {
        smallsize adjusted_index = insert_index - half;
        perform_shift_insert_warp<key_type>(lane_id, linked_node, adjusted_index, node_size - half, insertkey, thisoffset);
    }
}

// Used to find if key exists or the insertion point if key does not exist
// may be used Just for the search operation for Query and Deletion
template <typename key_type>
DEVICEQUALIFIER bool single_search_cuda_buffer_warp(
    int lane_id,
    const void *buffer,
    key_type search_key,
    smallsize num_elements,
    smallsize &insert_index)
{
    key_type my_key = 0;
    bool active = false;
    bool found = false;

    if (lane_id >= 0 && lane_id < num_elements)
    {
        my_key = extract_key_node<key_type>(buffer, lane_id+1);
        if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
        {
            active = true;
            found = (my_key == search_key);
        }
    }

    unsigned ballot_mask = __ballot_sync(0xffffffff, found);
    if (ballot_mask != 0)
    {
        int found_lane = __ffs(ballot_mask); //not needd + 1;
        insert_index = found_lane;
        return true;
    }

    bool is_gt = active && (my_key > search_key);
    unsigned mask = __ballot_sync(0xffffffff, is_gt);
    if (mask != 0)
    {
        insert_index = __ffs(mask) ; //-1 ;
    }
    else
    {
        insert_index = num_elements + 1;
    }
    //if (search_key == 3504328163 && lane_id == 0)
    //{
    //     printf("In Search NOT FOUND KEY %u and insert_index is: %u num_elements are %u \n", search_key, insert_index, num_elements);
    //     print_node<key_type>(buffer, num_elements );
    //}

    return false;
}

// Warp version of insert processor

template <typename key_type>
DEVICEQUALIFIER void process_inserts_warp(
    key_type maxkey, smallsize minindex, smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    int lane_id)
{
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    void *curr_node = starting_node;

    for (smallsize i = minindex; i <= maxindex; ++i)
    {
        key_type key = update_list[i];
        smallsize offset = offset_list[i];

        if (lane_id == 0)
        {
            key_type curr_max = cg::extract<key_type>(curr_node, 0);
            while (curr_max < key)
            {
                smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);

                if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count)
                    return;

                curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + (next_ptr - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr = __shfl_sync(0xffffffff, raw_ptr, 0);
        curr_node = reinterpret_cast<void *>(raw_ptr);

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        smallsize insert_index = 1;

        bool found = single_search_cuda_buffer_warp<key_type>(lane_id, curr_node, key, curr_size, insert_index);
        if (found)
            continue;

        key_type key_at_index = extract_key_node<key_type>(curr_node, insert_index);
        if (key_at_index < key)
            insert_index = (curr_size == 0) ? 1 : insert_index;
            

       // if (key == 3504328163 && lane_id == 0)
       // {
       //      printf("PIW NOT FOUND KEY %u and insert_index is: %u \n", key, insert_index);
       //      print_node<key_type>(curr_node, node_size);
       // }

     

        if (curr_size < node_size)
        {   
            
            perform_shift_insert_warp<key_type>(lane_id, curr_node, insert_index, curr_size, key, offset);
        }
        else
        {
            process_insert_split_warp<key_type>(lane_id, curr_node, insert_index, key, offset, launch_params);
        }
    }

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();
    if (lane_id == 0 && blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("END INSERTIONS (warp): PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, lane_id);
    }
#endif
}



template <typename key_type>
GLOBALQUALIFIER void update_kernel_warp(updatable_cg_params *launch_params, bool perform_dels)
{
    const smallsize update_size = launch_params->update_size;
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const key_type *maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    const smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;

    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = global_tid / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= partition_count_with_overflow)
        return;

    key_type maxkey = maxbuf[warp_id];
    key_type minkey = (warp_id > 0) ? maxbuf[warp_id - 1] + 1 : 1;

    int minindex = -1;
    int maxindex = -1;

    if (lane_id == 0)
    {
        minindex = binarySearchIndex<key_type>(update_list, minkey, 0, update_size, false);
        maxindex = binarySearchIndex<key_type>(update_list, maxkey, 0, update_size, true);
    }

    minindex = __shfl_sync(0xffffffff, minindex, 0);
    maxindex = __shfl_sync(0xffffffff, maxindex, 0);

    bool skip = (minindex > maxindex || minindex == -1 || maxindex == -1);
    unsigned mask = __ballot_sync(0xffffffff, skip);
    if (mask != 0)
        return;

    void *curr_node = reinterpret_cast<uint8_t *>(launch_params->ordered_node_pairs) + warp_id * launch_params->node_stride;

    if (perform_dels)
    {
        // process_deletes_warp<key_type>(...);  // Define a warp version if needed
    }
    else
    {
        process_inserts_warp<key_type>(maxkey, static_cast<smallsize>(minindex), static_cast<smallsize>(maxindex),
                                       launch_params, curr_node, lane_id);
    }
}



#endif