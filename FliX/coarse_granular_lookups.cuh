// =============================================================================
// File: coarse_granular_lookups.cuh
// Author: Rosina Kharal
// Description: Implements coarse_granular_lookups
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INDEX_PROTOTYPE_COARSE_GRANULAR_LOOKUPS_CUH
#define INDEX_PROTOTYPE_COARSE_GRANULAR_LOOKUPS_CUH

#include <cstdint>
#include <cstdio>
#include "launch_parameters.cuh"
#include "definitions.cuh"
#include "definitions_coarse_granular.cuh"
#include "definitions_updates.cuh"
#include "coarse_granular_inserts.cuh"
#include "debug_definitions_updates.cuh"

//#define LOOKUP_KERNEL_DEBUG


template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER
bool binary_search_equal_or_greater(const key_type* maxvalues, smallsize num_elements, key_type key, smallsize &found_index, int tid) {
    smallsize left = 0;
    smallsize right = num_elements-1;

    while (left < right) {
        smallsize mid = left + (right - left) / 2;
        key_type mid_key = maxvalues[mid];

        if (mid_key == key) {
            found_index = mid;
            return true;
        }
        if (mid_key < key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    found_index = left;


    // Case: key > all elements
    if (found_index == num_elements-1) {
        //printf("Search key %u not found, found_index %u, num_elements %u key at found_index %u \n", key, found_index, num_elements, maxvalues[found_index]);
        return false;
    }

    return true;  // maxvalues[found_index] >= key
}

//----------------------------------------
//  Binary Search in CUDA Buffer for MAXVALUES
//----------------------------------------------

template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER
bool binary_search_in_array_old(const key_type* maxvalues, smallsize num_elements, key_type key, smallsize &found_index, int tid) {
    smallsize left = 0;                // Start from the first element in array
    smallsize right = num_elements-1;  // End at the last element in array
 

    //print all of maxvalues using %d 
    for (int i = 0; i < num_elements; i++)
       if (key ==2249298376 )printf("BF Bin Search Tid: %d, maxvalues[%d] %u\n", tid, i, maxvalues[i]);


    while (left < right) {
        smallsize mid = left + (right - left) / 2;
         // print all above values
         //if (key ==2249298376 ) printf(" LOOP Tid: %d, key %u, left %d, right %d, mid %d\n", tid, key, left, right, mid);

        // Extract key at the mid position
        key_type mid_key = maxvalues[mid];
        if (mid_key == key) {
            found_index = mid;
            //print "found it" , print above values
            //if (key ==2249298376 )printf("Found exact key Tid: %d, found_index %d, key found in maxvalues %u\n", tid, found_index, key);
            return true;
        }

        if (mid_key < key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    found_index = left;
    // print - out of loop what are the above values
   // printf("Tid: %d, key %u, left %d, right %d, mid %d found_index %d \n", tid, key, left, right, left, found_index);


    if (found_index > num_elements-1) {
        printf("ERROR insert_index %d, num_elements %d\n", found_index, num_elements);
    }

    if (found_index > 0 && maxvalues[found_index - 1] == key) {
        //print "found it" , print above values
       // printf("Found It Tid: %d, key %u, found_index %d, key found in maxvalues %u\n", tid, key, found_index, key);
        found_index = found_index - 1;
        return true;
    }

    // If the key is not found, return the smallest index to the left
    if (found_index == 0) {
        //print "not found it" , print above values
        if (key ==2249298376 ) printf("Not Found It Tid: %d, key %u, found_index %d, key not found in maxvalues \n", tid, key, found_index);
        return false;
    }

    found_index = found_index;
    return false;
}


template <typename key_type>
// GLOBALQUALIFIER
DEVICEQUALIFIER void process_lookups(key_type curr_node_max, smallsize bucket_index, key_type probe_key, updatable_cg_params *params, void *curr_node, key_type num_elements, int tid)
{

    const auto allocation_buffer = params->allocation_buffer;
    const smallsize node_stride = params->node_stride;
    const smallsize partition_size = params->partition_size;
    const key_type *maxbuf = static_cast<const key_type *>(params->maxvalues);
    const smallsize partition_count = params->partition_count;
    const smallsize partition_count_with_overflow = params->partition_count_with_overflow;
    const smallsize allocation_buffer_count = params->allocation_buffer_count;
    smallsize node_size = params->node_size;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    smallsize ix = tid;


    DEBUG_LOOKUP_DEV(" process_lookups", tid, probe_key, curr_node_max);

    //--------------------------------
    smallsize last_position_value = 0; // extract_key_node<key_type>(curr_node, lastpositionptr);
    key_type curr_max = cg::extract<key_type>(curr_node, 0);
    while (curr_max < probe_key)
    {
        last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
        last_position_value--; // decrement b/c it is inserted into node as +1 to avoid 0
//#ifdef LOOKUP_KERNEL_DEBUG
       // printf("While Loop Lookups: tid:%d, Traverse Links: curr_max %u is less than probe_key %u, last_position_value %u \n", tid, curr_max, probe_key, last_position_value);
//#endif
        // print_node<key_type>(curr_node, partition_size);

        // check for error
        if (last_position_value >= allocation_buffer_count)
        {

            DEBUG_LOOKUP_DEV("LOOKUP ERROR: LAST PTR EXCEEDS NUM PARTITIONS", last_position_value, allocation_buffer_count, probe_key);

            return;
        }
        curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
        curr_max = cg::extract<key_type>(curr_node, 0);
    }
    smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));




    smallsize found_index = 1;
    bool probe_key_found = binary_search_in_cuda_buffer<key_type>(curr_node, curr_size, probe_key, found_index, tid);

    if (!probe_key_found)
    {
        params->result[ix] = not_found;

     //if (probe_key ==2249298376) {
     //   DEBUG_LOOKUP_DEV("Not Found key", tid, probe_key);
     //}
        return;
        // ---> no for looop   continue; // move on to next key to insert
    }
    // found the key
#ifdef LOOKUP_KERNEL_DEBUG
    // if (probe_key ==13730996833869725462)
    printf("Tid: %d LOOKUP: WAS FOUND: key %u Present in node, found INDEX is %d \n", tid, probe_key, found_index);
    // print_node<key_type>(curr_node, partition_size);
#endif
    smallsize result;
    result = extract_offset_node<key_type>(curr_node, found_index);
    params->result[ix] = result;

    //--------------------------------



 
#ifdef PRINT_LOOKUPS_END
 __syncthreads();
 if (tid == 0)
 {
     printf("END LOOKUPS: PRINT ALL NODES\n");
     print_set_nodes_and_links<key_type>(params, tid);
 }

#endif

}

//---------------------------------------------------

template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER 
smallsize find_bucket_offset_for_probekey(key_type probe_key, const smallsize partition_count_with_overflow, const key_type* maxbuf) {

    smallsize final_bucket_offset = 0;
   
    // Extract max value at partition_offset
    key_type max_value_at_offset = maxbuf[final_bucket_offset];
    //if the probe_key is correctly less than the max value at the offset, return the partition offset. 
    //It is already Correct
    if (probe_key <= max_value_at_offset) {
        return final_bucket_offset;
    }
    while (probe_key > max_value_at_offset) {
        #ifdef DEBUG_LOOKUP
            printf("need to search for partition offset, probe_key %u, max_value_at_offset %u, final_partition_offset %u \n", probe_key, max_value_at_offset, final_partition_offset);
        #endif
        final_bucket_offset++;
       
        // ERROR CHECK
        if (final_bucket_offset >= partition_count_with_overflow) {
            printf("ERROR: While Loop Partition Offset is greater than Overflow, setting to Overflow node \n");
            return partition_count_with_overflow - 1;
        }
    
        max_value_at_offset = maxbuf[final_bucket_offset];
       // max_value_at_offset = (partition_offset < partition_count_with_overflow - 1) ? maxbuf[partition_offset + 1] : max_value_at_offset;
    }
    return final_bucket_offset;
}
//--------------------------------------------------

template <typename key_type>
// GLOBALQUALIFIER
DEVICEQUALIFIER void process_group_lookups(key_type curr_node_max, smallsize bucket_index, key_type probe_key, updatable_cg_params *params, void *curr_node, key_type num_elements, int tid)
{

    const auto allocation_buffer = params->allocation_buffer;
    const auto buf = params->ordered_node_pairs;
    const smallsize node_stride = params->node_stride;
    const smallsize partition_size = params->partition_size;
    const smallsize node_size = partition_size * 2;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(partition_size);
    const key_type *maxbuf = static_cast<const key_type *>(params->maxvalues);
    const smallsize partition_count = params->partition_count;
    const smallsize partition_count_with_overflow = params->partition_count_with_overflow;
    const smallsize allocation_buffer_count = params->allocation_buffer_count;
    smallsize ix = tid;
    smallsize lookup_count = 0;
    smallsize bucket_offset = not_found;
    smallsize query_size = params->query_size;
    const key_type* query_list = static_cast<const key_type*>(params->query_lower);

    key_type curr_max = curr_node_max;
   // void* curr_node = curr_node;
    
    assert( probe_key == query_list[ix] );
    key_type next_probe_key = probe_key;

    smallsize k  =ix;
    while( k<(node_size + ix) )
    {
        //--------------------------------
#ifdef LOOKUP_KERNEL_DEBUG
        printf("Process Lookups: tid:%d, Here to do Lookup for key %u Node_size is %d  k value is %d\n", tid, next_probe_key, node_size, k);
#endif
        //--------------------------------
        smallsize last_position_value = 0; // extract_key_node<key_type>(curr_node, lastpositionptr);
        curr_max = cg::extract<key_type>(curr_node, 0);
       // printf(" next_probe_key %u, curr_max %u \n", next_probe_key, curr_max);
        while (curr_max < next_probe_key)
        {
            last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
            last_position_value--; // decrement b/c it is inserted into node as +1 to avoid 0
#ifdef LOOKUP_KERNEL_DEBUG
            printf("Lookups: tid:%d, Traverse Links: curr_max %u is less than next_probe_key %u, last_position_value %u \n", tid, curr_max, next_probe_key, last_position_value);
#endif
            // print_node<key_type>(curr_node, partition_size);

            // check for error
            if (last_position_value >= allocation_buffer_count)
            {
//#ifdef LOOKUP_KERNEL_DEBUG
               // printf("Tid: %d LOOKUP ERROR: LAST PTR EXCEEDS NUM PARTITIONS: last_position_value: %u, AllocationBufferCount: %u Search Key %u \n", tid, last_position_value, allocation_buffer_count, next_probe_key);
//#endif        
                print_node<key_type>(curr_node, partition_size);
                return;
            }
            curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            curr_max = cg::extract<key_type>(curr_node, 0);
        }
        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));


        smallsize found_index = 1;
        bool probe_key_found = binary_search_in_cuda_buffer<key_type>(curr_node, curr_size, next_probe_key, found_index, ix);
        //if (next_probe_key == 641317930) {
       // printf("After Search : Tid: %d, key: %u in Binary Search, probe key found is: %d and found_index %d \n", tid, next_probe_key, probe_key_found, found_index);
            //print_node<key_type>(curr_node, partition_size);
        if (!probe_key_found)
        {
           // printf(" NOT FOUND Tid: %d, key: %u and k %d , \n", tid, next_probe_key, k);
            params->result[k] = not_found;
#ifdef LOOKUP_KERNEL_DEBUG
            // if (probe_key ==13730996833869725462)
            printf("Tid: %d LOOKUP: NOT FOUND: key %u Not present in node, found value is %u \n", tid, next_probe_key, not_found);
            // print_node<key_type>(curr_node, partition_size);
#endif
        } else { //Found

        smallsize result;
        result = extract_offset_node<key_type>(curr_node, found_index);
        //print next_probe_key and result in one line please 

       // printf(" FOUND Tid: %d, key: %u, result: %u and k %d \n", tid, next_probe_key, result, k);
        params->result[k] = result;
        }

        //next probe key
        k++;
        if (k >= query_size || k >= (node_size +ix)) return;  
        

        next_probe_key = query_list[k];
       // printf("End of loop, next probe key is %u k is %d \n", next_probe_key, k);
         if (next_probe_key > curr_max)
        {
            // Find the correct partition to search
              //---- Computer the Correct Partition Offset for this query key.
            bucket_offset = find_bucket_offset_for_probekey<key_type>(next_probe_key, partition_count_with_overflow, maxbuf);
            //printf(" Looked for new bucket offset Tid: %d,  next_probe_key %u, bucket_offset %d \n ", tid, probe_key, next_probe_key, bucket_offset);
            curr_node = reinterpret_cast<uint8_t *>(buf) + node_stride * bucket_offset;

        }
    }

    //--------------------------------

#ifdef LOOKUP_KERNEL_DEBUG
    if (tid == 0)
    {
        printf("COMPLETED ProcessLookups: Tid: %d, Completed ALl LOOKUPS and printing node\n", tid);
        print_node<key_type>(curr_node, partition_size);
    }
#endif

}

template <typename key_type>
// GLOBALQUALIFIER
DEVICEQUALIFIER void process_group_lookups_opt_duplicates(key_type curr_node_max, smallsize bucket_index, key_type probe_key, updatable_cg_params *params, void *curr_node, key_type num_elements, int tid)
{

    const auto allocation_buffer = params->allocation_buffer;
    const auto buf = params->ordered_node_pairs;
    const smallsize node_stride = params->node_stride;
    const smallsize partition_size = params->partition_size;
    const smallsize node_size = partition_size * 2;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(partition_size);
    const key_type *maxbuf = static_cast<const key_type *>(params->maxvalues);
    const smallsize partition_count = params->partition_count;
    const smallsize partition_count_with_overflow = params->partition_count_with_overflow;
    const smallsize allocation_buffer_count = params->allocation_buffer_count;
    smallsize ix = tid;
    smallsize lookup_count = 0;
    smallsize bucket_offset = not_found;
    smallsize query_size = params->query_size;
    const key_type* query_list = static_cast<const key_type*>(params->query_lower);

    key_type curr_max = curr_node_max;
   // void* curr_node = curr_node;
    
    assert( probe_key == query_list[ix] );
    key_type next_probe_key = probe_key;
    key_type prev_probe_key = 0;

    smallsize k  =ix;
    while( k<(node_size + ix) )
    {
        //--------------------------------
#ifdef LOOKUP_KERNEL_DEBUG
        printf("Process Lookups: tid:%d, Here to do Lookup for key %u Node_size is %d  k value is %d\n", tid, next_probe_key, node_size, k);
#endif
        //--------------------------------
        smallsize last_position_value = 0; // extract_key_node<key_type>(curr_node, lastpositionptr);
        curr_max = cg::extract<key_type>(curr_node, 0);
       // printf(" next_probe_key %u, curr_max %u \n", next_probe_key, curr_max);
        while (curr_max < next_probe_key)
        {
            last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
            last_position_value--; // decrement b/c it is inserted into node as +1 to avoid 0
#ifdef LOOKUP_KERNEL_DEBUG
            printf("Lookups: tid:%d, Traverse Links: curr_max %u is less than next_probe_key %u, last_position_value %u \n", tid, curr_max, next_probe_key, last_position_value);
#endif
            // print_node<key_type>(curr_node, partition_size);

            // check for error
            if (last_position_value >= allocation_buffer_count)
            {
//#ifdef LOOKUP_KERNEL_DEBUG
               // printf("Tid: %d LOOKUP ERROR: LAST PTR EXCEEDS NUM PARTITIONS: last_position_value: %u, AllocationBufferCount: %u Search Key %u \n", tid, last_position_value, allocation_buffer_count, next_probe_key);
//#endif        
                print_node<key_type>(curr_node, partition_size);
                return;
            }
            curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            curr_max = cg::extract<key_type>(curr_node, 0);
        }
        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        

        smallsize found_index = 1;
        bool probe_key_found = binary_search_in_cuda_buffer<key_type>(curr_node, curr_size, next_probe_key, found_index, ix);
        //if (next_probe_key == 641317930) {
       // printf("After Search : Tid: %d, key: %u in Binary Search, probe key found is: %d and found_index %d \n", tid, next_probe_key, probe_key_found, found_index);
            //print_node<key_type>(curr_node, partition_size);
        if (!probe_key_found)
        {
           // printf(" NOT FOUND Tid: %d, key: %u and k %d , \n", tid, next_probe_key, k);
            params->result[k] = not_found;
#ifdef LOOKUP_KERNEL_DEBUG
            // if (probe_key ==13730996833869725462)
            printf("Tid: %d LOOKUP: NOT FOUND: key %u Not present in node, found value is %u \n", tid, next_probe_key, not_found);
            // print_node<key_type>(curr_node, partition_size);
#endif
        } else { //Found

        smallsize result;
        result = extract_offset_node<key_type>(curr_node, found_index);
        //print next_probe_key and result in one line please 

       // printf(" FOUND Tid: %d, key: %u, result: %u and k %d \n", tid, next_probe_key, result, k);
        params->result[k] = result;
        }

        //next probe key
        k++;
        if (k >= query_size || k >= (node_size +ix)) return;  
        
        prev_probe_key = next_probe_key;
        next_probe_key = query_list[k];

        //optimization to skip duplicate keys
        while (prev_probe_key == next_probe_key) {
            //printf("Duplicate key found in query list, skipping it \n");
            params->result[k] = params->result[k-1];
            k++;
            if (k >= query_size || k >= (node_size +ix)) return;
            next_probe_key = query_list[k];
        }
       // printf("End of loop, next probe key is %u k is %d \n", next_probe_key, k);
         if (next_probe_key > curr_max)
        {
            // Find the correct partition to search
              //---- Computer the Correct Partition Offset for this query key.
            bucket_offset = find_bucket_offset_for_probekey<key_type>(next_probe_key, partition_count_with_overflow, maxbuf);
            //printf(" Looked for new bucket offset Tid: %d,  next_probe_key %u, bucket_offset %d \n ", tid, probe_key, next_probe_key, bucket_offset);
            curr_node = reinterpret_cast<uint8_t *>(buf) + node_stride * bucket_offset;

        }
    }

    //--------------------------------

#ifdef LOOKUP_KERNEL_DEBUG
    if (tid == 0)
    {
        printf("COMPLETED ProcessLookups: Tid: %d, Completed ALl LOOKUPS and printing node\n", tid);
        print_node<key_type>(curr_node, partition_size);
    }
#endif

}

//---------------------------------------------------

template <typename key_type>
// GLOBALQUALIFIER
DEVICEQUALIFIER void process_group_lookups_opt_twofingerwalk(key_type curr_node_max, smallsize bucket_index, key_type probe_key, updatable_cg_params *params, void *curr_node, key_type num_elements, int tid)
{

    const auto allocation_buffer = params->allocation_buffer;
    const auto buf = params->ordered_node_pairs;
    const smallsize node_stride = params->node_stride;
    const smallsize partition_size = params->partition_size;
    const smallsize node_size = partition_size * 2;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(partition_size);
    const key_type *maxbuf = static_cast<const key_type *>(params->maxvalues);
    const smallsize partition_count = params->partition_count;
    const smallsize partition_count_with_overflow = params->partition_count_with_overflow;
    const smallsize allocation_buffer_count = params->allocation_buffer_count;
    smallsize ix = tid;
    smallsize lookup_count = 0;
    smallsize bucket_offset = not_found;
    smallsize query_size = params->query_size;
    const key_type* query_list = static_cast<const key_type*>(params->query_lower);
    key_type prev_probe_key = 0;

    key_type curr_max = curr_node_max;
   // void* curr_node = curr_node;
    
    assert( probe_key == query_list[ix] );
    key_type next_probe_key = probe_key;

    smallsize k  =ix;
    while( k<(node_size + ix) )
    {
        //--------------------------------
#ifdef LOOKUP_KERNEL_DEBUG
        printf("Process Lookups: tid:%d, Here to do Lookup for key %u Node_size is %d  k value is %d\n", tid, next_probe_key, node_size, k);
#endif
        //--------------------------------
        smallsize last_position_value = 0; // extract_key_node<key_type>(curr_node, lastpositionptr);
        curr_max = cg::extract<key_type>(curr_node, 0);
       // printf(" next_probe_key %u, curr_max %u \n", next_probe_key, curr_max);
        while (curr_max < next_probe_key)
        {
            last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
            last_position_value--; // decrement b/c it is inserted into node as +1 to avoid 0
#ifdef LOOKUP_KERNEL_DEBUG
            printf("Lookups: tid:%d, Traverse Links: curr_max %u is less than next_probe_key %u, last_position_value %u \n", tid, curr_max, next_probe_key, last_position_value);
#endif
            // print_node<key_type>(curr_node, partition_size);

            // check for error
            if (last_position_value >= allocation_buffer_count)
            {
//#ifdef LOOKUP_KERNEL_DEBUG
               // printf("Tid: %d LOOKUP ERROR: LAST PTR EXCEEDS NUM PARTITIONS: last_position_value: %u, AllocationBufferCount: %u Search Key %u \n", tid, last_position_value, allocation_buffer_count, next_probe_key);
//#endif        
                print_node<key_type>(curr_node, partition_size);
                return;
            }
            curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            curr_max = cg::extract<key_type>(curr_node, 0);
        }
        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        
//#ifdef LOOKUP_KERNEL_DEBUG
       // if (next_probe_key == 641317930) {
          //  printf("GOING FOR SEARCH -Process Lookups: Tid: %d, Going to look for key: %u in Binary Search, Node max is: %u and node size is %u \n", tid, next_probe_key, curr_max, curr_size);
           // print_node<key_type>(curr_node, partition_size);
       // }
//#endif
        smallsize found_index = 1;
        bool probe_key_found = binary_search_in_cuda_buffer<key_type>(curr_node, curr_size, next_probe_key, found_index, ix);
        //if (next_probe_key == 641317930) {
       // printf("After Search : Tid: %d, key: %u in Binary Search, probe key found is: %d and found_index %d \n", tid, next_probe_key, probe_key_found, found_index);
            //print_node<key_type>(curr_node, partition_size);
        if (!probe_key_found)
        {
           // printf(" NOT FOUND Tid: %d, key: %u and k %d , \n", tid, next_probe_key, k);
            params->result[k] = not_found;
#ifdef LOOKUP_KERNEL_DEBUG
            // if (probe_key ==13730996833869725462)
            printf("Tid: %d LOOKUP: NOT FOUND: key %u Not present in node, found value is %u \n", tid, next_probe_key, not_found);
            // print_node<key_type>(curr_node, partition_size);
#endif
        } else { //Found

        smallsize result;
        result = extract_offset_node<key_type>(curr_node, found_index);
        //print next_probe_key and result in one line please 

       // printf(" FOUND Tid: %d, key: %u, result: %u and k %d \n", tid, next_probe_key, result, k);
        params->result[k] = result;
        }

        //next probe key
        k++;
        if (k >= query_size || k >= (node_size +ix)) return;  
        
        prev_probe_key = next_probe_key;
        next_probe_key = query_list[k];

        //optimization to skip duplicate keys
        while (prev_probe_key == next_probe_key) {
            //printf("Duplicate key found in query list, skipping it \n");
            params->result[k] = params->result[k-1];
            k++;
            if (k >= query_size || k >= (node_size +ix)) return;
            next_probe_key = query_list[k];
        }
       // printf("End of loop, next probe key is %u k is %d \n", next_probe_key, k);
         if (next_probe_key > curr_max)
        {
            // Find the correct partition to search
              //---- Computer the Correct Partition Offset for this query key.
            bucket_offset = find_bucket_offset_for_probekey<key_type>(next_probe_key, partition_count_with_overflow, maxbuf);
            //printf(" Looked for new bucket offset Tid: %d,  next_probe_key %u, bucket_offset %d \n ", tid, probe_key, next_probe_key, bucket_offset);
            curr_node = reinterpret_cast<uint8_t *>(buf) + node_stride * bucket_offset;

        }
    }

    //--------------------------------

#ifdef LOOKUP_KERNEL_DEBUG
    if (tid == 0)
    {
        printf("COMPLETED ProcessLookups: Tid: %d, Completed ALl LOOKUPS and printing node\n", tid);
        print_node<key_type>(curr_node, partition_size);
    }
#endif

}


///---------------------------------------------- SHARED MEMORY VERSION -------------------------------------------- ///
/*
template <typename key_type>
GLOBALQUALIFIER
void process_group_lookups_opt_duplicates_sharedmem(
    key_type curr_node_max,
    smallsize bucket_index,
    key_type probe_key,
    updatable_cg_params *params,
    void *curr_node,
    key_type num_elements,
    int tid) 
{

    const smallsize partition_count_with_overflow = params->partition_count_with_overflow;
    const smallsize allocation_buffer_count = params->allocation_buffer_count;
    smallsize ix = tid;
    smallsize lookup_count = 0;
    smallsize bucket_offset = not_found;
    smallsize query_size = params->query_size;

    // Declare shared memory
    __shared__ smallsize shared_partition_size;
    __shared__ smallsize shared_node_stride;
    __shared__ smallsize shared_query_size;
    __shared__ key_type shared_maxbuf[partition_count_with_overflow];  // Assuming MAX_PARTITIONS is known
    __shared__ key_type shared_query_list[query_size];  // Assuming MAX_QUERY_SIZE is known

    // First thread initializes shared memory
    if (threadIdx.x == 0) {
        shared_partition_size = params->partition_size;
        shared_node_stride = params->node_stride;
        shared_query_size = params->query_size;

        // Load max values (assumes partition_count_with_overflow is small)
        for (int i = 0; i < params->partition_count_with_overflow; i++) {
            shared_maxbuf[i] = static_cast<const key_type *>(params->maxvalues)[i];
        }

        // Load query list (assumes query size is small)
        for (int i = 0; i < shared_query_size; i++) {
            shared_query_list[i] = static_cast<const key_type *>(params->query_lower)[i];
        }
    }

    // Synchronize threads to ensure shared memory is initialized
    __syncthreads();

    // Variables stored in registers for fast access
    const key_type *maxbuf = shared_maxbuf;
    const key_type *query_list = shared_query_list;
    smallsize partition_size = shared_partition_size;
    smallsize node_stride = shared_node_stride;
    smallsize query_size = shared_query_size;

    key_type curr_max = curr_node_max;
    assert(probe_key == query_list[tid]);

    key_type next_probe_key = probe_key;
    key_type prev_probe_key = 0;
    smallsize k = tid;

    while (k < (partition_size * 2 + tid)) {
        smallsize last_position_value = 0;
        curr_max = cg::extract<key_type>(curr_node, 0);

        while (curr_max < next_probe_key) {
            last_position_value = cg::extract<smallsize>(curr_node, get_lastposition_bytes<key_type>(partition_size));
            last_position_value--;

            if (last_position_value >= params->allocation_buffer_count) {
                print_node<key_type>(curr_node, partition_size);
                return;
            }

            curr_node = reinterpret_cast<uint8_t *>(params->allocation_buffer) + last_position_value * node_stride;
            curr_max = cg::extract<key_type>(curr_node, 0);
        }

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        smallsize found_index = 1;
        bool probe_key_found = binary_search_in_cuda_buffer<key_type>(curr_node, curr_size, next_probe_key, found_index, tid);

        if (!probe_key_found) {
            params->result[k] = not_found;
        } else {
            params->result[k] = extract_offset_node<key_type>(curr_node, found_index);
        }

        // Move to next query key
        k++;
        if (k >= query_size || k >= (partition_size * 2 + tid)) return;

        key_type prev_probe_key = next_probe_key;
        next_probe_key = query_list[k];

        // Skip duplicate keys
        while (prev_probe_key == next_probe_key) {
            params->result[k] = params->result[k - 1];
            k++;
            if (k >= query_size || k >= (partition_size * 2 + tid)) return;
            next_probe_key = query_list[k];
        }

        if (next_probe_key > curr_max) {
            smallsize bucket_offset = find_bucket_offset_for_probekey<key_type>(next_probe_key, params->partition_count_with_overflow, maxbuf);
            curr_node = reinterpret_cast<uint8_t *>(params->ordered_node_pairs) + node_stride * bucket_offset;
        }
    }
}
*/

#endif

/* LINEAR TWO FINDER

template <typename key_type>
DEVICEQUALIFIER void process_group_lookups_opt_duplicates_TWO FINGEr(
    key_type curr_node_max,
    smallsize bucket_index,
    key_type probe_key,
    updatable_cg_params *params,
    void *curr_node,
    key_type num_elements,
    int tid) 
{
    const auto allocation_buffer = params->allocation_buffer;
    const auto buf = params->ordered_node_pairs;
    const smallsize node_stride = params->node_stride;
    const smallsize partition_size = params->partition_size;
    const smallsize node_size = partition_size * 2;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(partition_size);
    const key_type *maxbuf = static_cast<const key_type *>(params->maxvalues);
    const smallsize partition_count = params->partition_count;
    const smallsize partition_count_with_overflow = params->partition_count_with_overflow;
    const smallsize allocation_buffer_count = params->allocation_buffer_count;
    smallsize ix = tid;
    smallsize lookup_count = 0;
    smallsize bucket_offset = not_found;
    smallsize query_size = params->query_size;
    const key_type* query_list = static_cast<const key_type*>(params->query_lower);

    key_type curr_max = curr_node_max;
    key_type next_probe_key = probe_key;
    key_type prev_probe_key = 0;

    smallsize k = ix;
    while (k < (node_size + ix)) {
        smallsize last_position_value = 0;
        curr_max = cg::extract<key_type>(curr_node, 0);

        while (curr_max < next_probe_key) {
            last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes) - 1;
            if (last_position_value >= allocation_buffer_count) {
                print_node<key_type>(curr_node, partition_size);
                return;
            }
            curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            curr_max = cg::extract<key_type>(curr_node, 0);
        }

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        smallsize found_index = 1;
        bool found = false;

        for (smallsize i = 0; i < curr_size; i++) {
            key_type key_in_node = cg::extract<key_type>(curr_node, i * sizeof(key_type));
            
            if (key_in_node > next_probe_key) {
                params->result[k] = not_found;
                k++;
                if (k >= query_size) return;
                continue;
            }
            if (key_in_node == next_probe_key) {
                params->result[k] = extract_offset_node<key_type>(curr_node, found_index);
                found = true;
                k++;
                if (k >= query_size) return;
                continue;
            }
        }

        if (!found) {
            params->result[k] = not_found;
            k++;
            if (k >= query_size) return;
        }

        prev_probe_key = next_probe_key;
        next_probe_key = query_list[k];
        
        while (prev_probe_key == next_probe_key) {
            params->result[k] = params->result[k - 1];
            k++;
            if (k >= query_size) return;
            next_probe_key = query_list[k];
        }

        if (next_probe_key < curr_max) {
            continue;
        }

        bucket_offset = find_bucket_offset_for_probekey<key_type>(next_probe_key, partition_count_with_overflow, maxbuf);
        curr_node = reinterpret_cast<uint8_t *>(buf) + node_stride * bucket_offset;
    }
}

*/