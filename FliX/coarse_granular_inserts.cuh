// =============================================================================
// File: coarse_granular_inserts.cuh
// Author: Rosina Kharal
// Description: Implements coarse_granular_inserts
//              Single Threaded Shift_Right Insertions
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_INSERTS_CUH
#define INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_INSERTS_CUH

#include <cstdint>
#include <cstdio>
#include "launch_parameters.cuh"
#include "definitions.cuh"
#include "definitions_coarse_granular.cuh"
#include "definitions_updates.cuh"

/*
#define NODESPLIT
#define ERROR_CHECK
#define DEBUG_TID_0
#define DEBUG_LOOKUP
#define DEBUG_COMMENTS

*/
// #define ERROR_CHECK
// #define DEBUG_SELECTIVE

// #define ERROR_CHECK(msg, count, ...) PRINT_VARS()
//  Wrapper macro for ERROR_CHECK
#define ERROR_CHECK2(msg, count, ...) PRINT_VARS(msg, count, __VA_ARGS__)

// Define the device constant
//__device__ __constant__ smallsize key_offset_bytes = key_offset_bytes_value;
//----------------------------------------------------------------
// NOTE: These functions extract KEYS in the NODE only
template <typename key_type>
DEVICEQUALIFIER
    key_type
    extract_key_node(const void *curr_node, smallsize i)
{

    return cg::extract<key_type>(curr_node, (i)*compute_key_offset_bytes<key_type>());
};

template <typename key_type>
DEVICEQUALIFIER
    smallsize
    extract_offset_node(const void *curr_node, smallsize i)
{
    //
    return cg::extract<smallsize>(curr_node, (i)*compute_key_offset_bytes<key_type>() + sizeof(key_type));
};

template <typename key_type>
DEVICEQUALIFIER void set_key_node(void *curr_node, smallsize i, key_type key)
{

    cg::set<key_type>(curr_node, (i)*compute_key_offset_bytes<key_type>(), key);
};

template <typename key_type>
DEVICEQUALIFIER void set_offset_node(void *curr_node, smallsize i, smallsize offset)
{

    cg::set<smallsize>(curr_node, (i)*compute_key_offset_bytes<key_type>() + sizeof(key_type), offset);
};

template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER bool linear_search_in_cuda_buffer_with_tombstones(const void *buffer, smallsize num_elements, key_type key, smallsize &found_index, int tid)
{
    constexpr key_type tombstone = 1;

    for (smallsize i = 1; i <= num_elements; ++i)
    {
        key_type current_key = extract_key_node<key_type>(buffer, i);

        // Skip over tombstones
        if (current_key == tombstone)
        {
            continue;
        }

        // Check if the current key matches the search key
        if (current_key == key)
        {
            found_index = i;
            return true;
        }
    }

    // Key not found
    found_index = num_elements + 1; // Indicate that the key is not present
    return false;
}

template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER bool linear_search_in_cuda_buffer_with_tombstones_full(const void *buffer, smallsize num_elements, smallsize node_size, key_type key, smallsize &found_index, int tid)
{
    // search full node until 0 is reached

    constexpr key_type tombstone = 1;

    for (smallsize i = 1; i <= node_size; ++i)
    {
        key_type current_key = extract_key_node<key_type>(buffer, i);

        // if (key == 195250822) DEBUB_LOOKUP_LINEAR_SEARCH("looking for key", tid, key, current_key);
        //  Skip over tombstones
        if (current_key == tombstone)
        {
            continue;
        }
        if (current_key == 0)
        {                                   // reached the end of the node's keys
            found_index = num_elements + 1; // Indicate that the key is not present
            return false;
        }

        // Check if the current key matches the search key
        if (current_key == key)
        {
            found_index = i;
            return true;
        }
    }

    // Key not found
    found_index = num_elements + 1; // Indicate that the key is not present
    return false;
}

template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER bool binary_search_in_cuda_buffer_with_tombstones(const void *buffer, smallsize num_elements, key_type key, smallsize &found_index, int tid)
{
    constexpr key_type tombstone = 1;
    smallsize left = 1;             // Start at 1 because position 0 is metadata
    smallsize right = num_elements; // Valid range of elements

    while (left <= right)
    {
        smallsize mid = left + (right - left) / 2;

        // Extract key at the mid position
        key_type mid_key = extract_key_node<key_type>(buffer, mid);

        // Skip tombstones by adjusting the search bounds
        if (mid_key == tombstone)
        {
            // Check both sides for valid keys
            smallsize temp_left = mid, temp_right = mid;

            // Look for the next non-tombstone key on the right
            while (temp_right <= right)
            {
                mid_key = extract_key_node<key_type>(buffer, temp_right);
                if (mid_key != tombstone)
                {
                    mid = temp_right;
                    break;
                }
                temp_right++;
            }

            // If no valid key was found to the right, search left
            if (mid_key == tombstone)
            {
                while (temp_left >= left)
                {
                    mid_key = extract_key_node<key_type>(buffer, temp_left);
                    if (mid_key != tombstone)
                    {
                        mid = temp_left;
                        break;
                    }
                    temp_left--;
                }
            }

            // If no valid key is found on either side, terminate the search
            if (mid_key == tombstone)
            {
                break;
            }
        }

        // Perform standard binary search comparison
        if (mid_key < key)
        {
            left = mid + 1;
        }
        else if (mid_key > key)
        {
            right = mid - 1;
        }
        else
        {
            found_index = mid;
            return true; // Key found
        }
    }

    // Key not found, set found_index to insertion point
    found_index = left;
    return false; // Key not found
}

template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER
    // pass in num_elements is the size - num keys in node
    bool
    binary_search_in_cuda_buffer(const void *buffer, smallsize num_elements, key_type key, smallsize &insert_index, int tid)
{

    smallsize left = 1;             // First Two positions are dedicated 0;
    smallsize right = num_elements; // since first two positions are dedicated

    // if( key ==3280486733)
    //   printf("Binary Search: Tid: %d, key %u, num_elements %d  \n", tid, key, num_elements);

    while (left < right)
    {
        smallsize mid = left + (right - left) / 2;

        // Extract key at the mid position
        key_type mid_key = extract_key_node<key_type>(buffer, mid);
        // print all data, mid_key, key, left, right, mid
        // if( key ==3280486733)  printf("IN LOOP Binary Search: Tid: %d, mid_key %u, key %u, left %d, right %d, mid %d \n", tid, mid_key, key, left, right, mid);
        if (mid_key < key)
        {
            // if( key ==3280486733) printf("Binary Search: Tid: %d, mid_key: %u is < key: %u, left %d and right %d , mid: %d \n", tid, mid_key,key, left, right, mid);
            left = mid + 1;
        }
        else
        {
            // if( key ==3280486733) printf("Binary Search: Tid: %d, mid_key: %u is >= key: %u, left %d and right %d , mid: %d \n", tid, mid_key,key, left, right, mid);
            right = mid;
        }
    }

    insert_index = left;

#define ERROR_CHECK_Search
#ifdef ERROR_CHECK_Search
    if (insert_index > num_elements)
    {
        // printf("ERROR Binary Search: Tid: %d, key %u, insert_index %d, num_elements %d \n", tid, key, insert_index, num_elements);
    }
#endif

    key_type found_key = extract_key_node<key_type>(buffer, insert_index);
    if (found_key == key)
    {
        return true;
    }
    // if( key ==3280486733) printf("Binary Search returing FALSE Tid: %d, key %u, num_elements %d \n", tid, key, num_elements);
    return false;
}

//-----------------------------------------------
template <typename key_type>
DEVICEQUALIFIER void print_node_and_links(const void *curr_node, smallsize node_size, const void *allocation_buffer, smallsize allocation_buffer_count, smallsize node_stride)
{
    // Print max key and size
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    key_type max_key = cg::extract<key_type>(curr_node, 0);
    smallsize size = cg::extract<smallsize>(curr_node, sizeof(key_type));
    printf("[ {%llu}, {%u} ", static_cast<unsigned long long>(max_key), static_cast<smallsize>(size));

    const smallsize element_size = compute_key_offset_bytes<key_type>(); // 8-byte key + 4-byte offset = 12 bytes

    // Print all key-offset pairs
    for (smallsize i = 1; i <= node_size; ++i)
    {
        key_type key = extract_key_node<key_type>(curr_node, i);
        smallsize offset = extract_offset_node<key_type>(curr_node, i);
        printf(", (%llu, %u)", static_cast<unsigned long long>(key), static_cast<smallsize>(offset));
    }

    // Extract and print one more value from the node
    smallsize next_offset = cg::extract<smallsize>(curr_node, lastposition_bytes);
    printf(", %u", static_cast<smallsize>(next_offset));
    printf(" ]\n");

    // Traverse and print linked nodes
    while (next_offset != 0)
    {
        next_offset--; // Adjust for +1 offset in storage
        if (next_offset >= allocation_buffer_count)
        {
            printf("ERROR: LAST PTR EXCEEDS NUM PARTITIONS\n");
            return;
        }

        auto next_node = reinterpret_cast<const uint8_t *>(allocation_buffer) + next_offset * node_stride;
        printf("   ---> Linked Node: ");
        print_node_and_links<key_type>(next_node, node_size, allocation_buffer, allocation_buffer_count, node_stride);

        // Extract next last_position_value
        next_offset = cg::extract<smallsize>(next_node, lastposition_bytes);
    }
}
//-----------------------------------------------
template <typename key_type>
DEVICEQUALIFIER void print_node(const void *curr_node, smallsize node_size)
{
    // Print max key and size
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    key_type max_key = cg::extract<key_type>(curr_node, 0);
    smallsize size = cg::extract<smallsize>(curr_node, sizeof(key_type));
    printf("[ {%llu}, {%u} ", static_cast<unsigned long long>(max_key), static_cast<smallsize>(size));

    const smallsize element_size = compute_key_offset_bytes<key_type>(); // 8-byte key + 4-byte offset = 12 bytes

    // Print all key-offset pairs
    for (smallsize i = 1; i <= node_size; ++i)
    {
        // if (i % 2 == 0) {
        key_type key = extract_key_node<key_type>(curr_node, i);
        smallsize offset = extract_offset_node<key_type>(curr_node, i);
        printf(", (%llu, %u)", static_cast<unsigned long long>(key), static_cast<smallsize>(offset));
        // }
    }

    // Extract and print one more value from the node

    smallsize next_offset = cg::extract<smallsize>(curr_node, lastposition_bytes);

    printf(", %u", static_cast<smallsize>(next_offset));

    printf(" ]\n");
}

//----------------------------------------------------------------
template <typename key_type>
DEVICEQUALIFIER void perform_insert_shift(void *curr_node, smallsize insert_index, smallsize num_elements, key64 insertkey, smallsize thisoffset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // smallsize idx = tid + insert_index;

    // assert(insert_index <= num_elements + 1 && insert_index >= 2);
    // assert(insert_index >= 1 && insert_index <= num_elements);

#ifdef ERROR_CHECK
    // check this
    if (insert_index > num_elements)
    { //
        printf("ERROR INSERT INDEX > SIZE: tid:%d insertkey %llu, insert_index %d  num_elements % d\n", tid, insertkey, insert_index, num_elements);

        return; // no shifting, Error Case, Key is not inserted
    }
#endif

    const smallsize num_shifts = num_elements - insert_index + 1; // need to also shift out th
    smallsize curr_index = num_elements;

#ifdef DEBUG_TID_0
    if (tid == 0)
    {
        printf("PIS: B4 Shifting: Tid: %d, num_shifts %d, insert_index %d, num_elements %d\n", tid, num_shifts, insert_index, num_elements);
        print_node<key_type>(curr_node, node_size);
    }
#endif
    for (smallsize i = 0; i < num_shifts; ++i)
    {
        // Extract key and offset from curr_node at index curr_index
        key_type curr_key = extract_key_node<key_type>(curr_node, curr_index);
        smallsize curr_offset = extract_offset_node<key_type>(curr_node, curr_index);
#ifdef DEBUG_TID_0
        if (tid == 0)
            printf("Tid: %d, key that is read and will move to one over: curr_key %llu, curr offset: %u read from curr_index %d \n", tid, curr_key, curr_offset, curr_index);
#endif
        // Set key and offset at index (curr_index - 1)
        set_key_node<key_type>(curr_node, (curr_index + 1), curr_key);
        set_offset_node<key_type>(curr_node, (curr_index + 1), curr_offset);

        // Decrement curr_index
        --curr_index;
    }

    // set postion to be inserted into to 0:
    // should not be necessary
    set_key_node<key_type>(curr_node, insert_index, static_cast<key_type>(0));
    set_offset_node<key_type>(curr_node, insert_index, static_cast<smallsize>(0));

    // printf("After Perform Insert Shift: Tid: %d, num_shifts %d, insert_index %d, num_elements %d \n", tid, num_shifts, insert_index, num_elements);
#ifdef DEBUG_TID_0
    if (tid == 0)
    {
        printf("PIS: After shifting: Tid: %d, num_shifts %d, insert_index %d, num_elements %d\n", tid, num_shifts, insert_index, num_elements);
        print_node<key_type>(curr_node, node_size);
    }
#endif
    /******/

#ifdef DEBUG_TID_0
    printf("PIS: Tid: %d, Attempt to insert key %llu and offset %u at index %d \n", tid, insertkey, thisoffset, insert_index);
#endif

    /*** Insert Key and offset pair and Update Size ***/
    // set_key_node<key_type>(curr_node, insert_index, key);
    set_key_node<key_type>(curr_node, insert_index, static_cast<key_type>(insertkey));
    //----> I was setting the insert key twice
    //----> cg::set<key_type>(curr_node, insert_index * key_offset_bytes, insertkey);
    set_offset_node<key_type>(curr_node, insert_index, thisoffset);
    // update the size of the node
    smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
    cg::set<smallsize>(curr_node, sizeof(key_type), curr_size + 1);

#ifdef DEBUG_TID_0
    // if (tid == 0)
    //{
    printf("PIS: Complete Insertion: Tid: %d, num_shifts %d, insert_index %d, key %llu and offset %u \n", tid, num_shifts, insert_index, insertkey, thisoffset);
    print_node<key_type>(curr_node, node_size);
    //}
#endif
}

template <typename key_type>
DEVICEQUALIFIER int
binarySearchIndex(const key_type *array, key_type key, int left, int right, bool max)
{
    int result = 0;
    smallsize total_size = right;
    int mid;
    right = right - 1; // index is 1 less than size of list

    while (left <= right)
    {
        mid = left + (right - left) / 2;
        if (array[mid] == key)
            return mid;
        else if (array[mid] < key)
        {
            left = mid + 1;
            if (max)
                result = mid; // Update result if max is true and mid element is less than key
        }
        else
        {
            right = mid - 1;
            if (!max)
                result = mid; // Update result if max is false and mid element is greater than key
        }
    }

    if (!max && array[mid] > key && (mid == total_size - 1))
        result = mid;
    else if (!max && array[mid] < key && (mid == total_size - 1))
        result = -1;

    if (mid == 0 && max && array[mid] > key) // case where array[0] > max search range... another thread will catch this key
        result = -1;

    return result; // Return the closest index found, or -1 if no suitable index was foun
}

// This function is called only when we know  there is room in the node
template <typename key_type>
DEVICEQUALIFIER void
perform_insert(void *curr_node, smallsize insert_index, key_type insertkey, smallsize thisoffset, int tid)
{

    set_key_node<key_type>(curr_node, insert_index, static_cast<key_type>(insertkey));
    //----> cg::set<key_type>(curr_node, insert_index * key_offset_bytes, insertkey);
    set_offset_node<key_type>(curr_node, insert_index, thisoffset);
    // update the size of the node
    smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
    cg::set<smallsize>(curr_node, sizeof(key_type), curr_size + 1);
}

// first curr_node must be split into two nodes.
// Next the insertkey can be inserted into one of the two nodes.
template <typename key_type>
DEVICEQUALIFIER void insert_split(void *curr_node, smallsize insert_index, key64 insertkey, smallsize thisoffset, updatable_cg_params *launch_params, int tid)
{
    // Read necessary parameters from launch_params
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize node_stride = launch_params->node_stride;
    smallsize free_index = launch_params->free_node;
    smallsize partition_size = launch_params->partition_size;
    smallsize node_size = launch_params->node_size;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;

    if (free_index >= allocation_buffer_count)
    {
#ifdef ERROR_CHECK
        printf("Tid: %d ERROR: Insert Split: Allocation Buffer is Full: free.node: %u \n", tid, free_index);
#endif
        return;
    }
    // Increment the free_node index atomically
    smallsize free_value = atomicAdd((&launch_params->free_node), 1ULL);

#ifdef NODESPLIT
    printf("Tid: %d, Free Value: %d\n", tid, free_value);
#endif
    // smallsize free_value = atomicAdd(reinterpret_cast<unsigned long long int*>(&launch_params->free_node), 1ULL);
    auto linked_node = reinterpret_cast<uint8_t *>(allocation_buffer) + node_stride * free_value;

    const smallsize start_idx = partition_size + 1;
    const smallsize end_idx = partition_size * 2;
    const smallsize lastpositionptr = partition_size * 2 + 1;

    /******* Check for available space in Allocation Buffer **********/
    if (free_value >= allocation_buffer_count)
    {
#ifdef ERROR_INSERTS
        printf("Tid: %d ERROR: Insert Split: Allocation Buffer is Full: free.node: %u \n", tid, free_value);
#endif
        return;
    }

    //****** CHECKS
#ifdef NODESPLIT
    printf("Before SPLIT loop: Tid: %d, Print Both Nodes\n", tid);
    print_node<key_type>(curr_node, partition_size);
    printf("tid: %d Linked Node\n", tid);
    print_node<key_type>(linked_node, node_size);
#endif

    //}
    //******* SPLITTING
    // Move every key-offset pair from curr_node[start_idx] to curr_node[end_idx]
    for (smallsize i = start_idx; i <= end_idx; ++i)
    {
        key_type curr_key = extract_key_node<key_type>(curr_node, i);
        smallsize curr_offset = extract_offset_node<key_type>(curr_node, i);
        // Set the original position in curr_node to 0
        set_key_node<key_type>(curr_node, i, static_cast<key_type>(0));
        set_offset_node<key_type>(curr_node, i, static_cast<smallsize>(0));

        set_key_node<key_type>(linked_node, i - start_idx + 1, curr_key);
        set_offset_node<key_type>(linked_node, i - start_idx + 1, curr_offset);
    }

#ifdef ERROR_CHECK
    //****** CHECKS
    // check postion partition_size+1 in both nodes, it should be zero
    key_type test_curr_key = extract_key_node<key_type>(curr_node, partition_size + 1);
    key_type test_linked_key = extract_key_node<key_type>(linked_node, partition_size + 1);
    // printf(" check values of test_curr_key %llu, test_linked_key %llu\n", test_curr_key, test_linked_key);
    if (test_curr_key != 0 || test_linked_key != 0)
    {
        printf("ERROR: INSERT SPLIT NON ZERO: Tid: %d, Curr_Node || Linked Node [partition_size +1] Non Zero\n", tid);
        print_node<key_type>(curr_node, partition_size);
        print_node<key_type>(linked_node, node_size);
    }
#endif

#ifdef NODESPLIT
    printf(" AFTER  Split: Tid: %d, After Copy Half Curr Node to Linked Node\n", tid);
    print_node<key_type>(curr_node, node_size);
    print_node<key_type>(linked_node, node_size);
#endif
    //*********** Set sizes and Max for curr_node and linked_node
    cg::set<smallsize>(curr_node, sizeof(key_type), partition_size);
    cg::set<smallsize>(linked_node, sizeof(key_type), partition_size);
    // printf("Set Sizes and Maxes, \n"); //currnodes original max %llu, new curr node ma %llu \n", extract_key_node<key_type>(curr_node, 0), extract_key_node<key_type>(curr_node, partition_size));

    // Set max values for curr_node and linked_node

    // Ensure MAX VALUES for REPRESNATIVE TRIANGLES ARE IN TACT
    //  get current max of curr_node
    key_type original_curr_max = cg::extract<key_type>(curr_node, 0);
    key_type curr_max = extract_key_node<key_type>(curr_node, partition_size);
    // key_type link_max = extract_key_node<key_type>(linked_node, partition_size);
    cg::set<key_type>(curr_node, 0, curr_max);
    cg::set<key_type>(linked_node, 0, original_curr_max);

#ifdef NODESPLIT
    printf("After Set Maxes and Sizes, currnodes original max %llu, new curr node max %llu \n", original_curr_max, curr_max);
    print_node<key_type>(curr_node, node_size);
    print_node<key_type>(linked_node, node_size);
#endif

    //*********** Set Linked PTR
    // if lastptr in curr_node is non zero,this value must be assigned now to linked node
    // printf("Tid: %d Set Linked Ptr \n",tid);
    smallsize last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
    // printf("tid %d Last Position Value: %u print lastpostionsBytes %u \n",tid, last_position_value, lastposition_bytes);

    if (last_position_value != 0)
    {
        cg::set<smallsize>(linked_node, lastposition_bytes, last_position_value);
        cg::set<smallsize>(curr_node, lastposition_bytes, free_value + 1); // insert at plus 1 to avoid 0
    }
    else
    {
        cg::set<smallsize>(curr_node, lastposition_bytes, free_value + 1); // insert free_value+1 into last position since we
                                                                           //  do not allow 0 in the last position
        // last position in lined node should be 0
        //  check this
        smallsize linked_ptr = cg::extract<smallsize>(linked_node, lastposition_bytes);
#ifdef NODESPLIT
        if (linked_ptr != 0)
        {
            printf("ERROR: INSERT SPLIT: Tid: %d Linked Ptr is Non Zero\n", tid);
            print_node<key_type>(linked_node, node_size);
        }
#endif
    }

    // Must Insert the Insertkey in the correct node, curr_node or linked_node
    // Perform the insert after resizing
    if (insert_index < partition_size + 1)
    {
#ifdef NODESPLIT
        printf("tid: %d Insert Split:  key %llu will go in Curr Node, insert_index %d, partitionsize %d \n", tid, insertkey, insert_index, partition_size);
#endif
        perform_insert_shift<key_type>(curr_node, insert_index, partition_size, insertkey, thisoffset);

        // Adjust max values to account for inserted key
        key_type curr_max = extract_key_node<key_type>(curr_node, partition_size + 1);

        cg::set<key_type>(curr_node, 0, curr_max);
    }
    else // insert in linked node
    {

#ifdef NODESPLIT
        printf("tid: %d Insert Split key %llu will go in Linked Node, insert_index %d, partitionsize %d \n", tid, insertkey, insert_index, partition_size);
#endif
        // perform_insert_shift<key_type>(linked_node, insert_index - partition_size, partition_size);
        // perform_insert<key_type>(linked_node, insert_index - partition_size, key, partition_size,launch_params, tid);
        //  take key and find it's insert index in linked node
        smallsize new_index = 1;
        binary_search_in_cuda_buffer<key_type>(linked_node, partition_size, insertkey, new_index, tid);
        key_type key_at_index = extract_key_node<key_type>(linked_node, new_index);

        if (key_at_index < insertkey) // This also is the case when size of list is 0
        {
#ifdef NODESPLIT
            printf("INSERT SPLIT UNIQUE CASE: keyatindex < key Tid: %d All keys were less than new Insertkey: %llu, key_at_index: %llu, new_index %d curr size is Partition size: %d \n", tid, insertkey, key_at_index, new_index, partition_size);
            printf("tid %d, Printing, key_at_index< insertkey \n", tid);
            print_node<key_type>(linked_node, node_size);
#endif
            new_index = new_index + 1;
            perform_insert<key_type>(linked_node, new_index, static_cast<key64>(insertkey), thisoffset, tid);
            // Note: if node is full, this will be caught below when we check for size
        }
        else
        {
            // shift over and insert
            perform_insert_shift<key_type>(linked_node, new_index, partition_size, insertkey, thisoffset);
        }

        // Max is correct in linked node
    }
#ifdef NODESPLIT
    printf("tid %d, Complete Splitting and Inserting Printing\n", tid);
    print_node<key_type>(curr_node, node_size);
    print_node<key_type>(linked_node, node_size);
#endif
}

template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER void print_set_nodes_and_links(void *node_buffer, void *allocation_buffer, smallsize node_size, smallsize node_stride, smallsize partition_count_with_overflow, smallsize allocation_buffer_count)
{
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    for (int k = 0; k < partition_count_with_overflow; ++k)
    {
        auto curr_node = reinterpret_cast<uint8_t *>(node_buffer) + (node_stride)*k;
        printf("Printing Node # %d: NodeStride %d\n", k, node_stride);
        print_node<key_type>(curr_node, node_size);

        smallsize last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);

        while (last_position_value != 0)
        {
            last_position_value--;
            if (last_position_value >= allocation_buffer_count)
            {
                printf("ERROR: LAST PTR EXCEEDS NUM PARTITIONS\n");
                return;
            }
            auto next_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            printf("   ---> Linked Node: ");
            print_node<key_type>(next_node, node_size);

            last_position_value = cg::extract<smallsize>(next_node, lastposition_bytes);
        }
    }
}

//----------------------------------
template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER void print_set_nodes_and_links(updatable_cg_params *launch_params, int tid)
{
    smallsize node_stride = launch_params->node_stride;
    auto buf = launch_params->ordered_node_pairs;
    smallsize node_size = launch_params->node_size;
    smallsize num_buckets_with_overflow = launch_params->partition_count_with_overflow;
    auto allocation_buffer = launch_params->allocation_buffer;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    for (int k = 0; k < num_buckets_with_overflow; ++k)
    {
        auto curr_node = reinterpret_cast<uint8_t *>(buf) + (node_stride)*k;
        printf("tid:%d Printing Node # %d: NodeStride %d \n", tid, k, node_stride);
        print_node<key_type>(curr_node, node_size);

        // Extract initial last_position_value
        smallsize last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
        // last_position_value--; // Adjust for +1 offset in storage

        // Traverse and print linked nodes
        while (last_position_value != 0)
        {
            last_position_value--; // Adjust for +1 offset in storage
            if (last_position_value >= allocation_buffer_count)
            {
                ERROR_INSERTS("ERROR: LAST PTR EXCEEDS NUM PARTITIONS", tid, last_position_value, num_buckets_with_overflow);
                return;
            }

            auto next_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            printf("   ---> Linked Node: ");
            print_node<key_type>(next_node, node_size);

            // Extract next last_position_value
            last_position_value = cg::extract<smallsize>(next_node, lastposition_bytes);
        }
    }
}

// ----------------------------------

template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER void print_set_nodes(updatable_cg_params *launch_params, int tid)
{
    // Print the first 50 nodes
    smallsize node_stride = launch_params->node_stride;
    auto buf = launch_params->ordered_node_pairs;
    smallsize node_size = launch_params->node_size;
    smallsize numnodes_wiithoverflow = launch_params->partition_count_with_overflow;

    // for (int k = 0; k < numnodes_wiithoverflow; ++k)
    for (int k = 0; k < numnodes_wiithoverflow; ++k)

    {
        auto curr_node = reinterpret_cast<uint8_t *>(buf) + (node_stride)*k;
        printf("tid:%d Printing Node # %d: NodeStride %d \n ", tid, k, node_stride);
        // smallsize currnodesize = cg::extract<smallsize>(curr_node, sizeof(key_type));
        // Print the node
        print_node<key_type>(curr_node, node_size);
        // printf("Size of node %d: %u\n", idx, static_cast<smallsize>(currnodesize));
    }
}

template <typename key_type>
// GLOBALQUALIFIER
DEVICEQUALIFIER void process_inserts(key_type maxkey, smallsize minindex, smallsize maxindex, updatable_cg_params *launch_params, void *curr_node, key_type num_elements, int tid)
{
    //
    // compute tid here
    smallsize idx = blockIdx.x * blockDim.x + threadIdx.x;
    smallsize totalstoredsize = launch_params->stored_size;
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize partition_size = launch_params->partition_size;
    smallsize node_size = launch_params->node_size;
    smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;
    smallsize node_stride = launch_params->node_stride;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize returnval = 0; // 0 not inserted was present
                             // 1 successfully inserted
                             // 2 error of some type occured
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    auto buf = launch_params->ordered_node_pairs;
    smallsize maxkeys_pernode = node_size; // allow 2xpartionsize keys per node
                                           // last index is used for next pointer

    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);

#ifdef PRINT_PROCESS_INSERTS_DEBUG
    const key_type *maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    // print all max values in the maxbuf
    if (tid == 0)
    {
        printf("Tid: %d, PRINTING MAX VALUES and All DS TOP OF INS SHFT RIGHT \n", tid);
        for (int i = 0; i < partition_count_with_overflow; i++)
        {
            printf("Tid: %d, maxbuf[%d] %llu\n", tid, i, maxbuf[i]);
        }
    }

    print_set_nodes_and_links<key_type>(launch_params, tid);

#endif

    for (smallsize i = minindex; i <= maxindex; ++i)
    {
        // Get the key from the update list
        // key_type key = static_cast<const key_type*>(launch_params->update_list)[i];
        key_type key = update_list[i];

        DEBUG_PI("Process_Inserts ", tid, key, i);
        DEBUG_PI("Process_Inserts ", tid, minindex, maxindex);

        smallsize insert_index;
        // printf("ProcessUpdates: TID: %d, insert key %llu for i: %d :  minindex %d, maxindex %d \n", tid, key, i, minindex, maxindex);
        // printf("processUpdates: Tid: %d, minindex %d, maxindex %d\n", tid, minindex, maxindex);
        //  Perform binary search
        // FOR LOOP HERE UNTIL WE FIND THE CORRECT CURR_NODE
        smallsize last_position_value = 0; // extract_key_node<key_type>(curr_node, lastpositionptr);
        key_type curr_max = cg::extract<key_type>(curr_node, 0);
        while (curr_max < key)
        {

            last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
            last_position_value--; // decrement b/c it is inserted into node as +1 to avoid 0
                                   // NOTE: When we enter this code in error it is usually because list is unsorted
#ifdef PRINT_PROCESS_INSERTS
            printf("Process Inserts: tid:%d, Traverse Links: curr_max %llu is less than key %llu, last_position_value %u \n", tid, curr_max, key, last_position_value);
            print_node<key_type>(curr_node, node_size);
#endif
            // check for any error
            if (last_position_value >= allocation_buffer_count)
            {

                ERROR_INSERTS("ERROR: LAST PTR EXCEEDS NUM PARTITIONS", tid, curr_max, partition_count_with_overflow);

                return;
            }
            curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            curr_max = cg::extract<key_type>(curr_node, 0);
        }

#ifdef DEBUG_TID_0
        if (tid == 0)
        {
            printf("PU: Tid: %d, Going to look for key: %llu in Binary Search in Cuda Buffer \n", tid, key);
            print_node<key_type>(curr_node, node_size);
        }
#endif
        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        insert_index = 1;
        // if (tid==3072)
        //   {
        //       DEBUG_PI_BUCKET("Before Binary Search", tid, key, curr_size);
        //       print_node<key_type>(curr_node, node_size);
        //   }

        bool found = binary_search_in_cuda_buffer<key_type>(curr_node, curr_size, key, insert_index, tid);

        if (found)
        {
            // printf("PU: FOUND: Tid: %d, key %llu , insert_ index %d already present in node\n", tid, key, insert_index);
            //  launch_params->result[i]= 0; //key already present
            continue; // move on to next key to insert
        }

        // if (tid==3072)
        //     {
        //        DEBUG_PI_BUCKET("Not Found Before shift Insertion", tid, key, insert_index, curr_size);
        //        print_node<key_type>(curr_node, node_size);
        //    }
        // This check may not be necessary
        // extract key from curr_node at insert_index
        //  get offset corresponding to this key
        smallsize thisoffset = offset_list[i];
        key_type key_at_index = extract_key_node<key_type>(curr_node, insert_index);
        smallsize lastkeyindex = node_size;

        if (key_at_index < key) // This also is the case when size of list is 0
        {
#ifdef DEBUG_COMMENTS
            printf("Process Inserts: Key Not Found UNIQUE CASE: keyatindex < key Tid: %d All keys were less than new insertkey: %llu, key_at_index: %llu, insert_index %d curr_size: %d \n", tid, key, key_at_index, insert_index, curr_size);
            print_node<key_type>(curr_node, node_size);
#endif
            insert_index = (curr_size == 0) ? 1 : insert_index + 1;
            if (curr_size < maxkeys_pernode)
            {

                perform_insert<key_type>(curr_node, insert_index, static_cast<key64>(key), thisoffset, tid);

                /* 
                if (tid == 3072)
                {
                    DEBUG_PI_BUCKET("Unque Case PerformED Insert", tid, key, insert_index, curr_size);
                    print_node<key_type>(curr_node, node_size);
                }
                */
                continue;
            } // Note: if node is full, this will be caught below when we check for size
        }

        // If key is not found, perform insertion key at insert_index is > key
        if (curr_size < maxkeys_pernode)
        {
            // there is space in the node
#ifdef DEBUG_TID_0
            if (tid == 0)
                printf("PU NOT FOUND: Tid: %d, TRYING INSERT key: %llu Room in Non Full Node, size is:%d PRINTING BEFORE for tid =0 \n", tid, key, curr_size);
#endif
            perform_insert_shift<key_type>(curr_node, insert_index, curr_size, static_cast<key64>(key), thisoffset);
            //      if (tid==3072)
            //{
            //    DEBUG_PI_BUCKET("After Shift Perform Insert", tid, key, insert_index, curr_size);
            //    print_node<key_type>(curr_node, node_size);
            // }
            // perform_insert<key_type>(curr_node, insert_index, key, thisoffset, curr_size, launch_params, tid);
            // returnval = 1;
        }
        else
        { // need to resize node - split
#ifdef NODESPLIT
            printf("PU: Tid: %d, Need to RESIZE Node, for insertion key %llu \n", tid, key);
#endif
            insert_split<key_type>(curr_node, insert_index, static_cast<key64>(key), thisoffset, launch_params, tid); // up
        }

        // launch_params->result[i]= returnval;
    }
#ifdef DEBUG_SELECTIVE
    __syncthreads(); // Synchronize all threads in the block
    if (tid == 4)
    {
        printf("END INSERTIONS PRINT ALL NODES\n    ");
        print_set_nodes<key_type>(launch_params, tid);
    }
#endif

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();
    if (tid == 3)
    {
        printf("END Normal ShfRight INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, tid);
    }

#endif
}

//------------------------------------------------
// new Experiments Jan 2025 No Dynamic Allocation
//--------------- Node_Split
template <typename key_type>
DEVICEQUALIFIER void split_node(void *curr_node, key_type *keys_array, smallsize *offsets_array, smallsize num_elements, smallsize node_size, updatable_cg_params *launch_params, int tid)
{
    // Read necessary parameters from launch_params
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize node_stride = launch_params->node_stride;
    smallsize free_index = launch_params->free_node;
    smallsize partition_size = launch_params->partition_size;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;

    if (free_index >= allocation_buffer_count)
    {
        ERROR_INSERTS("ERROR: Node Split: Allocation Buffer is Full:", tid, free_index);
        return;
    }

    // Increment the free_node index atomically
    smallsize free_value = atomicAdd((&launch_params->free_node), 1ULL);
    // printf("Tid: %d, Free Value: %d\n", tid, free_value);

    auto linked_node = reinterpret_cast<uint8_t *>(allocation_buffer) + node_stride * free_value;

    /******* Check for available space in Allocation Buffer **********/
    if (free_value >= allocation_buffer_count)
    {
        ERROR_INSERTS("ERROR: Node Split: Allocation Buffer is Full:", tid, free_index);
        return;
    }

    // printf("Before SPLIT loop: Tid: %d, Print Both Nodes\n", tid);
    // print_node<key_type>(curr_node, partition_size);
    // printf("tid: %d Linked Node\n", tid);
    // print_node<key_type>(linked_node, node_size);

    // Split the keys and offsets between curr_node and linked_node
    smallsize mid = num_elements / 2;
    for (smallsize i = 0; i < mid; ++i)
    {
        set_key_node<key_type>(curr_node, i + 1, keys_array[i]);
        set_offset_node<key_type>(curr_node, i + 1, offsets_array[i]);
    }
    for (smallsize i = mid; i < num_elements; ++i)
    {
        set_key_node<key_type>(linked_node, i - mid + 1, keys_array[i]);
        set_offset_node<key_type>(linked_node, i - mid + 1, offsets_array[i]);
        set_key_node<key_type>(curr_node, i + 1, 0);
        set_offset_node<key_type>(curr_node, i + 1, 0);
    }

    // printf("CHECK: SPLIT NODES: Tid: \n", tid);
    // print_node<key_type>(curr_node, partition_size);
    // print_node<key_type>(linked_node, node_size);

    // #ifdef ERROR_CHECK
    //  Check position partition_size+1 in both nodes, it should be zero
    key_type test_curr_key = extract_key_node<key_type>(curr_node, partition_size + 1);
    key_type test_linked_key = extract_key_node<key_type>(linked_node, partition_size + 1);
    if (test_curr_key != 0 || test_linked_key != 0)
    {
        printf("ERROR: SPLIT NODE NON ZERO: Tid: %d, Curr_Node || Linked Node [partition_size +1] Non Zero\n", tid);
        // print_node<key_type>(curr_node, node_size);
        // print_node<key_type>(linked_node, node_size);
    }
    // #endif

#ifdef NODESPLIT
    printf("AFTER Split: Tid: %d, After Copy Half Curr Node to Linked Node\n", tid);
    print_node<key_type>(curr_node, node_size);
    print_node<key_type>(linked_node, node_size);
#endif

    // Set sizes and max values for curr_node and linked_node
    cg::set<smallsize>(curr_node, sizeof(key_type), mid);
    cg::set<smallsize>(linked_node, sizeof(key_type), num_elements - mid);

    // Set max values for curr_node and linked_node
    key_type original_max = cg::extract<key_type>(curr_node, 0);
    key_type curr_max = extract_key_node<key_type>(curr_node, mid);
    cg::set<key_type>(curr_node, 0, curr_max);
    cg::set<key_type>(linked_node, 0, original_max);

    // #ifdef NODESPLIT
    DEBUG_PI_BUCKET_SPLIT("After Set Maxes and Sizes", curr_max, original_max);
    // print_node<key_type>(curr_node, node_size);
    // print_node<key_type>(linked_node, node_size);
    //  #endif

    // Set linked pointer
    smallsize last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
    if (last_position_value != 0)
    {
        cg::set<smallsize>(linked_node, lastposition_bytes, last_position_value);
        cg::set<smallsize>(curr_node, lastposition_bytes, free_value + 1); // insert at plus 1 to avoid 0
    }
    else
    {
        cg::set<smallsize>(curr_node, lastposition_bytes, free_value + 1); // insert free_value+1 into last position since we do not allow 0 in the last position
        smallsize linked_ptr = cg::extract<smallsize>(linked_node, lastposition_bytes);
        // #ifdef NODESPLIT
        if (linked_ptr != 0)
        {
            printf("ERROR: SPLIT NODE: Tid: %d Linked Ptr is Non Zero\n", tid);
            print_node<key_type>(linked_node, node_size);
        }
        // #endif
    }

    DEBUG_PI_BUCKET_SPLIT("After Links Set", tid, last_position_value);
    // print_node<key_type>(curr_node, node_size);
    // print_node<key_type>(linked_node, node_size);
}
//-------------------------------

template <typename key_type>
DEVICEQUALIFIER void copy_arrays_to_node(void *curr_node, key_type *keys_array, smallsize *offsets_array, smallsize num_elements, smallsize node_size, updatable_cg_params *launch_params, int tid)
{
    // check if a split is requried. If so, split the node into two nodes and place keys in the correct node
    //  Copy all the keys and offsets from arrays into the curr_node and next_node pointed to by curr_node
    //  Update the size of curr_node
    // cg::set<smallsize>(curr_node, sizeof(key_type), num_elements);  //update the size
    //  size will be correct set in the split function
    // DEBUG_PI_BUCKET("IN Copy Function ", tid, num_elements, node_size);

    if (num_elements >= node_size)
    {
        // printf("Tid: %d, Copy Arrays to Node:  Splitting Node\n", tid);
        // print value in keys_array and offsets_array side by side on one line, and then print the curr_node using print_node function
        for (smallsize i = 0; i < num_elements; ++i)
        {
            // DEBUG_PI_BUCKET("keys/offsets ", tid, keys_array[i], offsets_array[i]); // %u\n", tid, i, keys_array[i], i, offsets_array[i]);
        }
        // print_node<key_type>(curr_node, node_size);

        split_node<key_type>(curr_node, keys_array, offsets_array, num_elements, node_size, launch_params, tid);
        return;
    }

    for (smallsize i = 0; i < num_elements; ++i)
    { // DEBUG_PI_BUCKET("Just copying: keys/offsets ", tid, keys_array[i], offsets_array[i]); // %u\n", tid, i, keys_array[i], i, offsets_array[i]);

        set_key_node<key_type>(curr_node, i + 1, keys_array[i]);
        set_offset_node<key_type>(curr_node, i + 1, offsets_array[i]);
    }
    // Update the size of curr_node
    cg::set<smallsize>(curr_node, sizeof(key_type), num_elements); // update the size
}

//---------------------------------
template <typename key_type>
DEVICEQUALIFIER void split_node_cudabuffer(
    void *curr_node, void *copy_node, smallsize num_elements,
    smallsize node_size, updatable_cg_params *launch_params, int tid)
{
    // Read necessary parameters from launch_params
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize node_stride = launch_params->node_stride;
    smallsize free_index = launch_params->free_node;
    smallsize partition_size = launch_params->partition_size;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;

    if (free_index >= allocation_buffer_count)
    {
        ERROR_INSERTS("ERROR: Node Split: Allocation Buffer is Full:", tid, free_index);
        return;
    }

    // Increment the free_node index atomically
    smallsize free_value = atomicAdd((&launch_params->free_node), 1ULL);

    auto linked_node = reinterpret_cast<uint8_t *>(allocation_buffer) + node_stride * free_value;

    if (free_value >= allocation_buffer_count)
    {
        ERROR_INSERTS("ERROR: Node Split: Allocation Buffer is Full:", tid, free_index);
        return;
    }

    // Split the keys and offsets between curr_node and linked_node
    smallsize mid = num_elements / 2;

    for (smallsize i = 1; i <= mid; ++i) // Start at index 1
    {
        key_type key = extract_key_node<key_type>(copy_node, i);
        smallsize offset = extract_offset_node<key_type>(copy_node, i);

        set_key_node<key_type>(curr_node, i, key);
        set_offset_node<key_type>(curr_node, i, offset);
    }

    for (smallsize i = mid + 1; i <= num_elements; ++i) // Continue from mid + 1
    {
        key_type key = extract_key_node<key_type>(copy_node, i);
        smallsize offset = extract_offset_node<key_type>(copy_node, i);

        set_key_node<key_type>(linked_node, i - mid, key); // Populate linked_node
        set_offset_node<key_type>(linked_node, i - mid, offset);

        set_key_node<key_type>(curr_node, i, 0); // Clear from curr_node
        set_offset_node<key_type>(curr_node, i, 0);
    }
#ifdef ERROR_CHECK
    //  Check position partition_size+1 in both nodes, it should be zero
    key_type test_curr_key = extract_key_node<key_type>(curr_node, partition_size + 1);
    key_type test_linked_key = extract_key_node<key_type>(linked_node, partition_size + 1);
    if (test_curr_key != 0 || test_linked_key != 0)
    {
        printf("ERROR: SPLIT NODE NON ZERO: Tid: %d, Curr_Node || Linked Node [partition_size +1] Non Zero\n", tid);
        // print_node<key_type>(curr_node, node_size);
        // print_node<key_type>(linked_node, node_size);
    }
#endif
    // Debugging to verify correct split
#ifdef NODESPLIT
    printf("AFTER Split: Tid: %d, After Copy Half Curr Node to Linked Node\n", tid);
    print_node<key_type>(curr_node, node_size);
    print_node<key_type>(linked_node, node_size);
#endif

    // Set sizes and max values for curr_node and linked_node
    cg::set<smallsize>(curr_node, sizeof(key_type), mid);
    cg::set<smallsize>(linked_node, sizeof(key_type), num_elements - mid);

    // Update max values for curr_node and linked_node
    key_type original_max = cg::extract<key_type>(curr_node, 0);
    key_type curr_max = extract_key_node<key_type>(curr_node, mid);
    cg::set<key_type>(curr_node, 0, curr_max);
    cg::set<key_type>(linked_node, 0, original_max);

    // DEBUG_PI_BUCKET_SPLIT("After Set Maxes and Sizes", curr_max, original_max);
    // print_node<key_type>(curr_node, node_size);
    // print_node<key_type>(linked_node, node_size);

    // Set linked pointer
    smallsize last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
    if (last_position_value != 0)
    {
        cg::set<smallsize>(linked_node, lastposition_bytes, last_position_value);
        cg::set<smallsize>(curr_node, lastposition_bytes, free_value + 1);
    }
    else
    {
        cg::set<smallsize>(curr_node, lastposition_bytes, free_value + 1);
        smallsize linked_ptr = cg::extract<smallsize>(linked_node, lastposition_bytes);
#ifdef NODESPLIT
        if (linked_ptr != 0)
        {
            printf("ERROR: SPLIT NODE: Tid: %d Linked Ptr is Non Zero\n", tid);
            print_node<key_type>(linked_node, node_size);
        }
#endif
    }
    // if (tid ==3){
    // DEBUG_PI_BUCKET_SPLIT("After Links Set", tid, last_position_value);
    // print_node<key_type>(curr_node, node_size);
    // print_node<key_type>(linked_node, node_size);
    // }
}

template <typename key_type>
DEVICEQUALIFIER void copy_arrays_to_node_cudabuffer(
    void *curr_node, void *copy_node, smallsize num_elements,
    smallsize node_size, smallsize tomb_stone_count, updatable_cg_params *launch_params, bool completed_inserts, int tid)
{
    // Check if a split is required. If so, split the node into two nodes and place keys in the correct node.
    // Copy all the keys and offsets from copy_node into curr_node and possibly a next_node.
    // Update the size of curr_node if no split is required.

    // if (num_elements >= node_size)
    if (num_elements >= node_size && !completed_inserts) // not causing extra node splits
    {

        /*
        if (tid==34)
        {
            DEBUG_PI_BUCKET("Tid: Copy Arrays to Node: Splitting Node", tid, num_elements, node_size);

        // Print debug info about the keys/offsets in copy_node before splitting.

                for (smallsize i = 1; i <= num_elements; ++i)
                {
                    key_type key = extract_key_node<key_type>(copy_node, i);
                    smallsize offset = extract_offset_node<key_type>(copy_node, i);
                    DEBUG_PI_BUCKET("Before Split: keys/offsets", tid, key, offset);
                }
                DEBUG_PI_BUCKET("Before Split: Print both curr_node and copy_node", tid);
                print_node<key_type>(curr_node, node_size);
                print_node<key_type>(copy_node, node_size);
        }
        */

        // Split the node and return
        /* if (tid==0)
        {
        DEBUG_PI_BUCKET_SPLIT("Before Split: Print both curr_node and copy_node", tid);
                print_node<key_type>(curr_node, node_size);
                print_node<key_type>(copy_node, node_size);
        } */
        split_node_cudabuffer<key_type>(curr_node, copy_node, num_elements, node_size, launch_params, tid);
        /* if (tid==0)
        {
        DEBUG_PI_BUCKET_SPLIT( "After Split print curr_node", tid);
        print_node<key_type>(curr_node, node_size);
        } */
        return;
    }

    // if (tid==34) DEBUG_PI_BUCKET(" Prior to Copying copy_node to curr_node:", tid, num_elements, node_size);
    // if (tid==34) print_node<key_type>(curr_node, node_size);
    // if (tid==34) print_node<key_type>(copy_node, node_size);

    // Copy values from copy_node to curr_node
    smallsize m = 1;
    for (m = 1; m <= num_elements; ++m)
    {
        key_type key = extract_key_node<key_type>(copy_node, m);
        smallsize offset = extract_offset_node<key_type>(copy_node, m);

        // if (tid==34) DEBUG_PI_BUCKET("Copying: keys/offsets", tid, key, offset);

        set_key_node<key_type>(curr_node, m, key);
        set_offset_node<key_type>(curr_node, m, offset);
    }
    // if (tid==34) DEBUG_PI_BUCKET("Complete Copy: ", tid,m, tomb_stone_count);
    // if (tid==34) print_node<key_type>(curr_node, node_size);

    // Update the size of curr_node
    cg::set<smallsize>(curr_node, sizeof(key_type), num_elements);

    for (int i = 1; i <= tomb_stone_count; ++i)
    {
        if (m > node_size)
        {
            break;
        }
        set_key_node<key_type>(curr_node, m, 0);
        set_offset_node<key_type>(curr_node, m, 0);
        m++;
    }

    // if (tid==34) DEBUG_PI_BUCKET("Complete Clear Tombstones: ", tid);
    // if (tid==34) print_node<key_type>(curr_node, node_size);
}

// OLDER VERSION WHERE CURR_SIZE did not decrement tombstones
template <typename key_type>
DEVICEQUALIFIER void A_process_inserts_per_bucket_tombstones_cudabuffers_A(key_type maxkey, smallsize minindex, smallsize maxindex, updatable_cg_params *launch_params, void *curr_node, key_type num_elements, int tid)
{
    // Compute tid here
    smallsize idx = blockIdx.x * blockDim.x + threadIdx.x;
    // smallsize totalstoredsize = launch_params->stored_size;
    void *allocation_buffer = launch_params->allocation_buffer;
    void *copy_buffer = launch_params->copy_buffer;

    smallsize partition_size = launch_params->partition_size;
    smallsize node_size = launch_params->node_size;
    smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;
    smallsize node_stride = launch_params->node_stride;

    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    // smallsize returnval = 0; // 0 not inserted was present
    //  1 successfully inserted
    //  2 error of some type occurred
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    // auto buf = launch_params->ordered_node_pairs;
    // smallsize maxkeys_pernode = node_size; // allow 2xpartition_size keys per node
    //  last index is used for next pointer
#ifdef PRINT_PROCESS_INSERTS
    const key_type *maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    key_type maxkey_forthisthread = maxbuf[idx]; // curr_node[0]; // Assuming the key to search is at position 2
#endif
    // Initialize local variables for update_list and offset_list
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);

    // Use thread-specific sections of copy_update_list and copy_offset_list
    // key_type *copy_update_list = static_cast<key_type *>(launch_params->copy_update_list);
    // smallsize *copy_offset_list = static_cast<smallsize *>(launch_params->copy_offset_list);

    void *copy_node = reinterpret_cast<uint8_t *>(copy_buffer) + tid * node_stride;

    // Clear the thread-specific sections
    for (smallsize m = 1; m <= node_size; ++m)
    {
        set_key_node<key_type>(copy_node, m, static_cast<key_type>(0));
        set_offset_node<key_type>(copy_node, m, static_cast<smallsize>(0));
    }

    // DEBUG_PI_BUCKET("IN Process Inserts BUCKETS: Tid: ", tid, minindex, maxindex);

    smallsize k = minindex;

    while (k <= maxindex)
    {
        key_type next_insert_key = update_list[k];

        smallsize last_position_value = 0; // extract_key_node<key_type>(curr_node, lastpositionptr);
        key_type curr_max = cg::extract<key_type>(curr_node, 0);

        if (next_insert_key == 4004458884)
            DEBUG_PI_BUCKET("While Loop Process Inserts per Bucket", tid, curr_max, next_insert_key);

        while (curr_max < next_insert_key)
        {
            last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
            last_position_value--; // decrement b/c it is inserted into node as +1 to avoid 0
#ifdef PRINT_PROCESS_INSERTS_FIND_NODE
            DEBUG_PI_BUCKET("Find the correct Node in Bucket", tid, curr_max, next_insert_key);
            print_node<key_type>(curr_node, node_size);
#endif

            if (last_position_value >= allocation_buffer_count)
            {
                ERROR_INSERTS("ERROR: LAST PTR EXCEEDS NUM PARTITIONS", tid, curr_max, partition_count_with_overflow);
                return;
            }
            curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            curr_max = cg::extract<key_type>(curr_node, 0);
        }

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        smallsize ts_count = 0;
        smallsize j = 1;            // Index for keys and offsets
        smallsize update_idx = k;   // Index for update_list
        smallsize insert_count = 0; // Counter for new insertions

        // ----> if(next_insert_key > curr_max) printf("ERROR in while loop: Tid: %d, curr_max %u, next_insert_key %u\n", tid, curr_max, next_insert_key);

        for (smallsize i = 1; i <= curr_size; ++i)
        {
            key_type curr_key = extract_key_node<key_type>(curr_node, i);
            smallsize curr_offset = extract_offset_node<key_type>(curr_node, i);

            // ERROR  you may go off the end of the LIST... Exhaust the Size of the node and grab a 0,0

            /* while (curr_key == tombstone){
                 i++;
                 curr_key = extract_key_node<key_type>(curr_node, i);
                 curr_offset = extract_offset_node<key_type>(curr_node, i);
                 ts_count++;
             } */

            if (curr_key == tombstone)
            {
                ts_count++;
                continue;
            }

            if (next_insert_key == 4004458884)
                DEBUG_PI_BUCKET("Extracted Key/Offset from Curr_Node", tid, curr_key, curr_offset);

            // Insert all keys from update_list that are less than the current key
            while (insert_count + curr_size < (node_size + ts_count) && update_idx <= maxindex && update_list[update_idx] < curr_key && j <= node_size)
            {
                set_key_node<key_type>(copy_node, j, update_list[update_idx]);
                set_offset_node<key_type>(copy_node, j, offset_list[update_idx]);

                if (update_list[update_idx] == 4004458884)
                    DEBUG_PI_BUCKET("Inserted Update List into Copy Node", tid, update_list[update_idx], offset_list[update_idx]);

                j++;
                update_idx++;
                insert_count++;
            }

            // If the current key matches a key in update_list, update the offset
            if (update_idx <= maxindex && update_list[update_idx] == curr_key)
            {
                curr_offset = offset_list[update_idx];

                if (update_list[update_idx] == 4004458884)
                    DEBUG_PI_BUCKET("Updated Offset", tid, curr_key, curr_offset);

                update_idx++;
            }

            // Copy current key and offset
            ///--->  if(j > node_size) printf("ERROR in J : Tid: %d, curr_max %u, curr_size %u j= %d \n", tid, curr_max, curr_size, j);

            set_key_node<key_type>(copy_node, j, curr_key);
            set_offset_node<key_type>(copy_node, j, curr_offset);

            if (next_insert_key == 4004458884)
                DEBUG_PI_BUCKET("Copied Curr_Node into Copy Node", tid, curr_key, curr_offset);

            j++;
        }

        // Insert any remaining keys from update_list
        while (insert_count + curr_size < (node_size + ts_count) && update_idx <= maxindex && update_list[update_idx] <= curr_max && j <= node_size)
        {
            set_key_node<key_type>(copy_node, j, update_list[update_idx]);
            set_offset_node<key_type>(copy_node, j, offset_list[update_idx]);

            if (next_insert_key == 4004458884)
                DEBUG_PI_BUCKET("Inserted Remaining Update List into Copy Node", tid, update_list[update_idx], offset_list[update_idx]);

            j++;
            update_idx++;
            insert_count++;
        }

        // Copy the arrays back to curr_no
        // DEBUG_PI_BUCKET("Before Copy Arrays to Node", tid, curr_size, j);
        // if( j > node_size+1) printf("ERROR in J : Tid: %d, curr_max %u, curr_size %u j= %d \n", tid, curr_max, curr_size, j);
        copy_arrays_to_node_cudabuffer<key_type>(curr_node, copy_node, j - 1, node_size, launch_params, tid);

        k = update_idx;
    }

#ifdef PRINT_PROCESS_INSERTS
    __syncthreads();
    if (tid == 19)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes<key_type>(launch_params, tid);
    }

    // Loop to print all values in maxbuf
    if (tid == 5)
    {
        printf("END INSERTIONS number of partitions count with overflow %u \n", partition_count_with_overflow);
        for (smallsize i = 0; i < partition_count_with_overflow; ++i)
        {
            key_type value = maxbuf[i];
            printf("MAX BUFFER Index: %d, Value: %u \n", i, value);
        }
        printf("TID: %d, Max Key for this thread: %u\n", tid, maxkey_forthisthread);
        printf("************************************\n");
    }

#endif
}

// -- process Inserts without TOMBSTONES
template <typename key_type>
DEVICEQUALIFIER void process_inserts_per_bucket(key_type maxkey, smallsize minindex, smallsize maxindex, updatable_cg_params *launch_params, void *curr_node, key_type num_elements, int tid)
{
    // Compute tid here
    smallsize idx = blockIdx.x * blockDim.x + threadIdx.x;
    smallsize totalstoredsize = launch_params->stored_size;
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize partition_size = launch_params->partition_size;
    smallsize node_size = launch_params->node_size;
    smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;
    smallsize node_stride = launch_params->node_stride;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize returnval = 0; // 0 not inserted was present
                             // 1 successfully inserted
                             // 2 error of some type occurred
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    auto buf = launch_params->ordered_node_pairs;
    smallsize maxkeys_pernode = node_size; // allow 2xpartition_size keys per node
                                           // last index is used for next pointer

    const key_type *maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    key_type maxkey_forthisthread = maxbuf[idx]; // curr_node[0]; // Assuming the key to search is at position 2

    // Initialize local variables for update_list and offset_list
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);

    // Use thread-specific sections of copy_update_list and copy_offset_list
    key_type *copy_update_list = static_cast<key_type *>(launch_params->copy_update_list);
    smallsize *copy_offset_list = static_cast<smallsize *>(launch_params->copy_offset_list);

    // Thread-specific sections
    key_type *keys_array = copy_update_list + tid * node_size;
    smallsize *offsets_array = copy_offset_list + tid * node_size;

    // Clear the thread-specific sections
    for (smallsize m = 0; m < node_size; ++m)
    {
        keys_array[m] = 0;
        offsets_array[m] = 0;
    }
    DEBUG_PI_BUCKET("IN Process Inserts BUCKETS: Tid: ", tid, minindex, maxindex);

    smallsize k = minindex;

    while (k <= maxindex)
    {
        key_type next_insert_key = update_list[k];

        smallsize last_position_value = 0; // extract_key_node<key_type>(curr_node, lastpositionptr);
        key_type curr_max = cg::extract<key_type>(curr_node, 0);

        DEBUG_PI_BUCKET("While Loop Process Inserts per Bucket", tid, curr_max, next_insert_key);
        while (curr_max < next_insert_key)
        {
            last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
            last_position_value--; // decrement b/c it is inserted into node as +1 to avoid 0
#ifdef PRINT_PROCESS_INSERTS
            DEBUG_PI_BUCKET("Find the correct Node in Bucket", tid, curr_max, next_insert_key);
            print_node<key_type>(curr_node, node_size);
#endif

            if (last_position_value >= allocation_buffer_count)
            {
                ERROR_INSERTS("ERROR: LAST PTR EXCEEDS NUM PARTITIONS", tid, curr_max, partition_count_with_overflow);
                return;
            }
            curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            curr_max = cg::extract<key_type>(curr_node, 0);
        }

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        smallsize j = 0;            // Index for keys and offsets
        smallsize update_idx = k;   // Index for update_list
        smallsize insert_count = 0; // Counter for new insertions

        if (next_insert_key > curr_max)
            printf("ERROR in while loop: Tid: %d, curr_max %u, next_insert_key %u\n", tid, curr_max, next_insert_key);

        for (smallsize i = 1; i <= curr_size; ++i)
        {
            key_type curr_key = extract_key_node<key_type>(curr_node, i);
            smallsize curr_offset = extract_offset_node<key_type>(curr_node, i);

            // if (tid == 5)
            //{
            DEBUG_PI_BUCKET("Extracted Key/Offset from Curr_Node", tid, curr_key, curr_offset);
            //}

            // Insert all keys from update_list that are less than the current key
            while ((insert_count + curr_size < node_size) && update_idx <= maxindex && update_list[update_idx] < curr_key && j < node_size)
            {
                keys_array[j] = update_list[update_idx];
                offsets_array[j] = offset_list[update_idx];

                // if (tid == 5)
                // {
                DEBUG_PI_BUCKET("Inserted Update List into Copy Arrays", tid, update_list[update_idx], offset_list[update_idx]);
                // }

                j++;
                update_idx++;
                insert_count++;
            }

            // If the current key matches a key in update_list, update the offset
            if (update_idx <= maxindex && update_list[update_idx] == curr_key)
            {
                curr_offset = offset_list[update_idx];

                // if (tid == 5)
                // {
                DEBUG_PI_BUCKET("Updated Offset", tid, curr_key, curr_offset);
                // }

                update_idx++;
            }

            // Copy current key and offset
            if (j >= node_size)
                printf("ERROR in J : Tid: %d, curr_max %u, curr_size %u j= %d \n", tid, curr_max, curr_size, j);

            keys_array[j] = curr_key;
            offsets_array[j] = curr_offset;

            // if (tid == 5)
            // {
            DEBUG_PI_BUCKET("Copied Curr_Node into Copy Arrays", tid, curr_key, curr_offset);
            // }

            j++;
        }

        // Insert any remaining keys from update_list
        // while ((insert_count + curr_size < node_size) && update_idx <= maxindex)
        while ((insert_count + curr_size < node_size) && update_idx <= maxindex && update_list[update_idx] <= curr_max && j < node_size)
        {
            keys_array[j] = update_list[update_idx];
            offsets_array[j] = offset_list[update_idx];

            // if (tid ==5)
            // {
            DEBUG_PI_BUCKET("Inserted Remaining Update List into Copy Arrays", tid, update_list[update_idx], offset_list[update_idx]);
            // }

            j++;
            update_idx++;
            insert_count++;
        }

        // Debugging: print the final state of keys_array and offsets_array
        /*

         if (tid == 5)
         {
             printf("TID %d: Final state of keys and offsets before copy:\n", tid);
             print_node<key_type>(curr_node, node_size);
             for (smallsize m = 0; m < node_size; ++m)
             {
                 DEBUG_PI_BUCKET("Final Keys/Offsets", tid, keys_array[m], offsets_array[m]);
             }
         }

        */

        // Copy the arrays back to curr_node
        // assert (j <= node_size);
        // if(j >= node_size) printf("ERROR in J : Tid: %d, curr_max %u, curr_size %u j= %d \n", tid, curr_max, curr_size, j);
        copy_arrays_to_node<key_type>(curr_node, keys_array, offsets_array, j, node_size, launch_params, tid);

        k = update_idx;
    }

#ifdef PRINT_PROCESS_INSERTS_TOMBSTONES
    __syncthreads();
    if (tid == 5)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes<key_type>(launch_params, tid);
    }

    // Loop to print all values in maxbuf
    if (tid == 5)
    {
        printf("END INSERTIONS number of partitions count with overflow %u \n", partition_count_with_overflow);
        for (smallsize i = 0; i < partition_count_with_overflow; ++i)
        {
            key_type value = maxbuf[i];
            printf("MAX BUFFER Index: %d, Value: %u \n", i, value);
        }
        printf("TID: %d, Max Key for this thread: %u\n", tid, maxkey_forthisthread);
        printf("************************************\n");
    }

#endif
}

// #endif

#endif