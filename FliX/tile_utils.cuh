// =============================================================================
// File: tile_utils.cuh
// Author: Rosina Kharal
// Description: Implements tile_utils  
//              tile and non tile based operations used for investigation. Not all are used.
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef TILE_SEARCH_UTILS_CUH
#define TILE_SEARCH_UTILS_CUH

#include "definitions_updates.cuh"  // includes TILE_SIZE, coop_g alias

// Rightmost occurrence  Returns -1 if key not found.

template <typename key_type>
DEVICEQUALIFIER int
binarySearchIndex_rightmost(const key_type* array, key_type key, int left, int right)
{
    // Empty range fast path
    if (left >= right) return -1;

    int lo = left;
    int hi = right; // half-open [lo, hi)

   
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (array[mid] <= key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // lo is first > key; candidate rightmost is lo - 1
    int idx = lo - 1;
    if (idx >= left && idx < right && array[idx] == key) {
        return idx;  // rightmost duplicate
    }
    return -1;       // not found
}

template <typename key_type>
DEVICEQUALIFIER int
binarySearchIndex_leftmost_ge_orig(const key_type* __restrict__ array,
                              key_type key,
                              int left,
                              int right /* size: half-open [left, right) */)
{
    // Binary search for lower_bound(key): first index with array[idx] >= key
    int lo = left;
    int hi = right; // half-open

    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        // Move left on equality to converge to the leftmost position
        if (array[mid] >= key) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    // No element >= key
    if (lo >= right) return -1;

   
    if (array[lo] == key) {
        while (lo > left && array[lo - 1] == key) {
            --lo;
        }
    }
    return lo;
}

// file: binary_search.cuh

template <typename Key>
DEVICEQUALIFIER int binarySearchIndex_leftmost_ge(
    const Key* __restrict__ array,
    Key key,
    int left,
    int right /* half-open: [left, right) */)
{

   // printf(" 0 Binary Search Leftmost GE: Called with left=%d right=%d key=%llu \n", left, right, static_cast<unsigned long long>(key));
    // 
    if (!array) return -80;
    if (right <= left) return -70;
    if (left < 0) left = 0;               // optional clamp; remove if you want strictness
    if (right < 0) return -50;             // invalid
    // NOTE: cannot clamp 'right' to allocation size here because we don't know it.

    int lo = left;
    int hi = right -1;

   // printf(" 0 Binary Search Leftmost GE: Initial lo=%d hi=%d \n", lo, hi);

    while (lo < hi) {
        // Safe mid; avoids any UB from shifting negatives (hi-lo is non-negative here)

       /// printf("1 Binary Search Leftmost GE: lo=%d hi=%d mid calc \n", lo, hi);
        int mid = lo + (hi - lo) / 2;
       /// printf("2 Binary Search Leftmost GE: lo=%d hi=%d mid=%d array[mid]=%llu key=%llu \n", lo, hi, mid, static_cast<unsigned long long>(array[mid]), static_cast<unsigned long long>(key));
        const Key v = array[mid];
        if (v >= key) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    // If insertion point is at 'right', then no element >= key in [left,right)
    //return (lo < right) ? lo : -99;

    return (lo < right) ? lo : -1;
}



template <typename key_type>
DEVICEQUALIFIER int
binarySearchIndex_first_gt_leftmost_dup(const key_type* __restrict__ array,
                                        key_type key,
                                        int left,
                                        int right /* size: half-open [left, right) */)
{
    // 1) upper_bound(key): first index with array[idx] > key
    int lo = left;
    int hi = right;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (array[mid] > key) {
            hi = mid;       // keep left half (including mid)
        } else {
            lo = mid + 1;   // skip <= key
        }
    }

    
    if (lo >= right) return -1;

    // 2) Walk left to the first duplicate of array[lo]
    const key_type target = array[lo];
    while (lo > left && array[lo - 1] == target) {
        --lo;
    }
    return lo;
}


template <typename key_type>
DEVICEQUALIFIER void perform_shift_insert_tile(
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

     // Early Exit if inserting at last position
     if(insert_index == TILE_SIZE )
     {
         if (tid == TILE_SIZE -1){
             set_key_node<key_type>(curr_node, insert_index, static_cast<key_type>(insertkey));
             set_offset_node<key_type>(curr_node, insert_index, thisoffset);
     
             smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
             cg::set<smallsize>(curr_node, sizeof(key_type), curr_size + 1);
         }
             tile.sync();
             return;
     }
 


    bool do_shift = (tid >= insert_index && tid <= num_elements);

    if (do_shift) {
        key = extract_key_node<key_type>(curr_node, tid);
        offset = extract_offset_node<key_type>(curr_node, tid);
    }

    tile.sync();  // <-- everyone syncs here safely

    if (do_shift) {
        set_key_node<key_type>(curr_node, tid + 1, key);
        set_offset_node<key_type>(curr_node, tid + 1, offset);
    }

    tile.sync();  //
    if (tid == insert_index) {
        set_key_node<key_type>(curr_node, insert_index, static_cast<key_type>(insertkey));
        set_offset_node<key_type>(curr_node, insert_index, thisoffset);

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        cg::set<smallsize>(curr_node, sizeof(key_type), curr_size + 1);
    }
}



// Used to find if key exists or the insertion point if key does not exist
// may be used Just for the search operation for Query and Deletion
template <typename key_type>
DEVICEQUALIFIER bool search_only_cuda_buffer_with_tile(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const void *buffer,
    key_type search_key,
    smallsize num_elements,
    smallsize &found_index)
{
    smallsize my_index = tile.thread_rank();
    key_type my_key = 0;
    bool active = false;
    bool found = false;

   // printf("TILE ID id %u, Search key %u my_index %u, num_elements %u\n", tile.meta_group_rank(),tile.thread_rank(),search_key,  my_index, num_elements);
    if (my_index < num_elements)
    {
        my_key = extract_key_node<key_type>(buffer, my_index + 1); // keys begin at position 1 to node_size
        if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
        {
            active = true;
            found = (my_key == search_key);
        }
    }
    //printf("Search key %u my_index %u, active %u found %u \n", search_key,  my_index, active, found);

    if (tile.any(found))
    {
        unsigned mask = tile.ballot(found);

        int first_active = __ffs(mask)- 1; // THIS (()#($*)#@(*$)) FUNCTION is 1 BASED !
        smallsize thread_found_index = 0;

        if (tile.thread_rank() == first_active)
            thread_found_index = tile.thread_rank() + 1; // + 1; // real index is +1

        found_index = tile.shfl(thread_found_index, first_active);
        // tile.sync();
        return true;
    }

    return false;
}

// Used to find if key exists or the insertion point if key does not exist
// may be used Just for the search operation for Query and Deletion
template <typename key_type>
DEVICEQUALIFIER bool single_search_cuda_buffer_tile(
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

    if ( my_index < num_elements)
    {
        my_key = extract_key_node<key_type>(buffer, my_index+1);  // keys begin at position 1 to node_size
        if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
        {
            active = true;
            found = (my_key == search_key);
        }
    }

    // Fast exit if found
    bool key_found = tile.any(found);
    if (key_found)
    {
        smallsize thread_found_index = my_index;
        insert_index = tile.shfl(thread_found_index, tile.match_any(found));
        insert_index += 1; // Adjust for 1-based indexing
        return true;
    }

    // Otherwise find insertion point
    bool is_gt = active && (my_key > search_key);
    unsigned mask = tile.ballot(is_gt);

    if (mask != 0)
    {
        insert_index = __ffs(mask);  // __ffs is 1-based
    }
    else
    {
        insert_index = num_elements + 1;
    }

    return false;
}

template <typename key_type>
DEVICEQUALIFIER void process_split_tile(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    void *curr_node,
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
    int reuse_num = 0;
    if (t == 0)
    {
        free_index = atomicAdd(&(launch_params->free_node), 1ULL);

        if (free_index >= allocation_buffer_count) {
            reuse_num = atomicSub(reinterpret_cast<int*>(launch_params->reuse_list_count), 1);
            if (reuse_num < 0) {
                printf("ERROR: no more nodes available\n");
                
            } else {
                free_index = (launch_params->reuse_list[reuse_num]);
            }
        }
    }
    free_index = tile.shfl(free_index, 0);


    if (free_index >= allocation_buffer_count) 
        return;

    auto linked_node = reinterpret_cast<uint8_t *>(allocation_buffer) + node_stride * free_index;

    smallsize half = node_size / 2;          //  7 if node_size = 14
    smallsize move_count = node_size - half; // ie 7
    smallsize t_index = t;

   
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
   // if (insertkey ==2377150882 && t == 0){
           // print_node<key_type>(curr_node, node_size);
      //    printf("SPlit FUnction FOUND KEY %llu and insertindex is %d \n", insertkey, insert_index);
      //    print_node<key_type>(curr_node, node_size);

    
    tile.sync();
}

#endif