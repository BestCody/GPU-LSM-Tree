// =============================================================================
// File: device_code_coarse_granular_updates.cu
// Author: Justus Henneberg, Rosina Kharal
// Description: Launch experiments     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#include "launch_parameters.cuh"

#include "coarse_granular_inserts.cuh"
#include "definitions_coarse_granular.cuh"
#include "definitions.cuh"
#include "definitions_updates.cuh"

extern "C" __constant__ updatable_cg_params params;
//__constant__ key64 MAX_KEY = ~0ULL; // All bits set to 1
//__constant__ key64 MIN_KEY = 1ULL;  // Minimum key value set to 1/*

//#define DEBUG_LOOKUP_NODES
//---------------------------------------
// These Functions are Un Used-
// See extract_key_node and extract_offset_node
template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER
    key_type
    extract_key_stride(const void *buf, smallsize i, smallsize partition_offset)
{

    return cg::extract<key_type>(buf, partition_offset*params.nodestride + i * key_offset_bytes));
}

// //MAKE SURE THIS IS KEY_TYPE PASSED IN
template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER
    smallsize
    extract_offset_stride(const void *buf, smallsize i, smallsize partition_offset)
{
    return cg::extract<smallsize>(buf, partition_offset * params.nodestride + i * key_offset_bytes + sizeof(key_type));
}
//----------------------------------------

template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER
    // Note: Need to ensure we are taking keys from i =1 onward
    smallsize
    find_range_start_in_partition_nodes(const updatable_cg_params &params, smallsize partition_offset, key_type lower_bound)
{
    const auto buf = params.ordered_node_pairs; // change rk

    const smallsize first_offset = params.partition_size * partition_offset;
    const smallsize last_offset = min(first_offset + params.partition_size, params.stored_size);

    smallsize search_offset = last_offset - 1;
    smallsize initial_skip = 31u - __clz(last_offset - first_offset);
    for (smallsize skip = uint32_t(1) << initial_skip; skip > 0; skip >>= 1u)
    {
        if (search_offset < first_offset + skip)
            continue;
        if (extract_key<key_type>(buf, search_offset - skip) >= lower_bound)
            search_offset -= skip;
    }
    return search_offset;
}

template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER void perform_lookup_nodes_tombstones()
{
    constexpr smallsize rays_per_thread = 1;

    const smallsize num_threads = optixGetLaunchDimensions().x;
    const smallsize thread_offset = optixGetLaunchIndex().x;
    const smallsize last_thread_offset = rays_per_thread * num_threads;
    const smallsize thread_increment = num_threads;
    const smallsize tid = thread_offset;

    const auto buf = params.ordered_node_pairs;
    const auto allocation_buffer = params.allocation_buffer;
    const smallsize node_stride = params.node_stride;
    const smallsize partition_size = params.partition_size;
    const smallsize node_size = params.node_size;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    const key_type* maxbuf = static_cast<const key_type*>(params.maxvalues);
    const smallsize partition_count = params.partition_count;
    const smallsize partition_count_with_overflow = params.partition_count_with_overflow;
    const smallsize allocation_buffer_count = params.allocation_buffer_count;

    const key_type max_of_first_node = maxbuf[0];
    const key_type max_of_last_node = maxbuf[partition_count-1]; 

    // print the two values above
     #ifdef DEBUG_LOOKUP
             if (tid ==0)
                 printf("Tid: %d, max_of_first_node %u, max_of_last_node %u \n", tid, max_of_first_node, max_of_last_node);
     #endif 
   // const bool more_than_one_row = (min_inserted_key >> cg::x_bits) != (max_inserted_key >> cg::x_bits);
   // const bool more_than_one_plane = (min_inserted_key >> (cg::x_bits + cg::y_bits)) != (max_inserted_key >> (cg::x_bits + cg::y_bits));

#ifdef DEBUG_LOOKUP
    //printf("Tid: %d, LOOKUP FUNCTION \n", tid);
#endif
   
    // my version--->  for (smallsize ix = thread_offset; ix < last_thread_offset; ix++)
    for (smallsize ix = thread_offset; ix < last_thread_offset; ix += thread_increment)
    {
        // ---> testing with small ranges ---> smallsize ix = thread_offset; 
        {

            key_type probe_key = reinterpret_cast<const key_type *>(params.query_lower)[ix];
            key_type lower_bound = reinterpret_cast<const key_type *>(params.query_lower)[ix];
            //key_type upper_bound = reinterpret_cast<const key_type *>(params.query_upper)[ix];

            smallsize partition_offset = not_found;
            if ( probe_key > max_of_last_node)
            {
            #ifdef DEBUG_LOOKUP
             if (probe_key ==1928079863)
                printf("Tid: %d Probe key %u GREATER than maxlastnode %u \n", tid, probe_key, max_of_last_node);
            #endif 
                partition_offset= partition_count_with_overflow - 1; //use the overflow node
            }
            else if (probe_key < max_of_first_node)
            {
                // range starts before the smallest key, start searching in partition 0
            #ifdef DEBUG_LOOKUP
             if (probe_key ==1928079863)
                printf("Tid: %d Probe key %u LESS than max_of_first %u \n", tid, probe_key, max_of_first_node);
            #endif 
                partition_offset = 0;
            }
            else
            {
                // perform tracing to find the partition

            #ifdef DEBUG_LOOKUP
             if (probe_key ==1928079863)
                printf("Tid: %d Probe key %u GO SEARCH WITH RAYS \n", tid, probe_key);
            #endif 
            
                partition_offset = cg::find_partition_offset_with_rays(params.traversable, params.partition_count, lower_bound);

             #ifdef DEBUG_LOOKUP
             if (probe_key == 195250822)
                printf("Tid: %d FIRED RAYS Probe key %u partition_offset %d \n", tid, partition_offset);
            #endif 
            }

            // added Rosina
            if (partition_offset == not_found)
            {
#ifdef DEBUG_LOOKUP
                //printf("Tid: %d LOOKUP PARTITION_OFFSET NOT FOUND: probe_key %u:  \n", tid, probe_key);
#endif
                params.result[ix] = not_found;
                return;
             // --no for loop               continue;
            }
 
              //if (probe_key == 195250822)DEBUB_LOOKUP_TOMBSTONES("Tid:partition found", tid, probe_key, partition_offset );

            // use partition offset to find the first entry node in the Bucket
            auto curr_node = reinterpret_cast<uint8_t *>(buf) + node_stride * partition_offset;
            auto next_curr_node = reinterpret_cast<uint8_t *>(buf) + node_stride * (partition_offset+1);

            smallsize last_position_value = 0; // extract_key_node<key_type>(curr_node, lastpositionptr);
            key_type curr_max = cg::extract<key_type>(curr_node, 0);
           // #ifdef DEBUG_LOOKUP
            //if (probe_key == 1594331703 && tid == 327){
            //               DEBUB_LOOKUP_TOMBSTONES("Tid: after curr_node found Top Loop", tid, probe_key, curr_max);
            //               print_node<key_type>(curr_node, node_size);
            //               printf("Tid: %d, next_curr_node Top Loop \n", tid);
            //               print_node<key_type>(next_curr_node, node_size);
            //            
            // }

            //#endif 
            while (curr_max < probe_key)
            {
                last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
                last_position_value--; // decrement b/c it is inserted into node as +1 to avoid 0
#ifdef DEBUG_LOOKUP
                 if (probe_key == 195250822) printf("Lookups: tid:%d, Traverse Links: curr_max %llu is less than probe_key %llu, last_position_value %u \n", tid, curr_max, probe_key, last_position_value);
#endif
                                       // print_node<key_type>(curr_node, partition_size);

                // check for error
                if (last_position_value >= allocation_buffer_count)
                {
//#ifdef ERROR_CHECK
                 // if (probe_key == 195250822) printf("Tid: %d LOOKUP ERROR: LAST PTR EXCEEDS NUM PARTITIONS: last_position_value: %u, node max: %llu Search Key %llu \n", tid, last_position_value, curr_max, probe_key);
//#endif
                    return;
                }
                curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
            smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

#ifdef DEBUG_LOOKUP
            if (probe_key ==1928079863) {
                 printf("After While loop LOOKUP: Tid: %d, Going to look for key: %u in Binary Search, curr max is: %u and node size is %u \n", tid, probe_key, curr_max, curr_size);
                 print_node<key_type>(curr_node, partition_size*2);
            }
#endif
            smallsize found_index = 1;
            //bool probe_key_found = binary_search_in_cuda_buffer<key_type>(curr_node, curr_size, probe_key, found_index, tid);
            //bool probe_key_found = binary_search_in_cuda_buffer_with_tombstones<key_type>(curr_node, curr_size, probe_key, found_index, tid);
            //bool probe_key_found = linear_search_in_cuda_buffer_with_tombstones<key_type>(curr_node, curr_size, node_size, probe_key, found_index, tid);
            bool probe_key_found = linear_search_in_cuda_buffer_with_tombstones_full<key_type>(curr_node, curr_size, node_size, probe_key, found_index, tid);

           // if (probe_key == 912857181 && tid == 445)
          //  {
          DEBUB_LOOKUP_TOMBSTONES("Tid:LOOKUP after linear search", tid, probe_key,probe_key_found);
                //print_node<key_type>(curr_node, node_size);
           // }

            if (!probe_key_found)
            {
                params.result[ix] = not_found;
#ifdef DEBUG_LOOKUP
                if (probe_key ==1928079863)
                    printf("Tid: %d LOOKUP: NOT FOUND: key %u Not present in node\n", tid, probe_key);
                    //print_node<key_type>(curr_node, partition_size);
#endif          
                return;
                // ---> no for looop   continue; // move on to next key to insert
            }
            // found the key
            smallsize result;
            result = extract_offset_node<key_type>(curr_node, found_index);
            params.result[ix] = result;
#ifdef DEBUG_LOOKUP
            if (probe_key ==1928079863)
                printf("Tid: %d LOOKUP FOUND: key %u and Offset Result: %u \n", tid, probe_key, result);
            // print_node<key_type>(curr_node, partition_size);
#endif
        }
        
   }

  /*   __syncthreads();
   if (tid == 0)
   {
    printf("END LOOKUPS: PRINT ALL NODES\n");
    print_set_nodes_and_links<key_type>(&params, tid);
   }
  */

}


template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER void perform_lookup_nodes()
{
    constexpr smallsize rays_per_thread = 1;

    const smallsize num_threads = optixGetLaunchDimensions().x;
    const smallsize thread_offset = optixGetLaunchIndex().x;
    const smallsize last_thread_offset = rays_per_thread * num_threads;
    const smallsize thread_increment = num_threads;
    const smallsize tid = thread_offset;

    const auto buf = params.ordered_node_pairs;
    const auto allocation_buffer = params.allocation_buffer;
    const smallsize node_stride = params.node_stride;
    const smallsize partition_size = params.partition_size;
    const smallsize node_size = params.node_size;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    const key_type* maxbuf = static_cast<const key_type*>(params.maxvalues);
    const smallsize partition_count = params.partition_count;
    const smallsize partition_count_with_overflow = params.partition_count_with_overflow;
    const smallsize allocation_buffer_count = params.allocation_buffer_count;

    const key_type max_of_first_node = maxbuf[0];
    const key_type max_of_last_node = maxbuf[partition_count-1]; 

    // print the two values above
     #ifdef DEBUG_LOOKUP
             if (tid ==0)
                 printf("Tid: %d, max_of_first_node %u, max_of_last_node %u \n", tid, max_of_first_node, max_of_last_node);
     #endif 
   // const bool more_than_one_row = (min_inserted_key >> cg::x_bits) != (max_inserted_key >> cg::x_bits);
   // const bool more_than_one_plane = (min_inserted_key >> (cg::x_bits + cg::y_bits)) != (max_inserted_key >> (cg::x_bits + cg::y_bits));

#ifdef DEBUG_LOOKUP
    //printf("Tid: %d, LOOKUP FUNCTION \n", tid);
#endif
   
    // my version--->  for (smallsize ix = thread_offset; ix < last_thread_offset; ix++)
    for (smallsize ix = thread_offset; ix < last_thread_offset; ix += thread_increment)
    {
        // ---> testing with small ranges ---> smallsize ix = thread_offset; 
        {

            key_type probe_key = reinterpret_cast<const key_type *>(params.query_lower)[ix];
            key_type lower_bound = reinterpret_cast<const key_type *>(params.query_lower)[ix];
            //key_type upper_bound = reinterpret_cast<const key_type *>(params.query_upper)[ix];

            smallsize partition_offset = not_found;
            if ( probe_key > max_of_last_node)
            {
            #ifdef DEBUG_LOOKUP
             if (probe_key ==1928079863)
                printf("Tid: %d Probe key %u GREATER than maxlastnode %u \n", tid, probe_key, max_of_last_node);
            #endif 
                partition_offset= partition_count_with_overflow - 1; //use the overflow node
            }
            else if (probe_key < max_of_first_node)
            {
                // range starts before the smallest key, start searching in partition 0
            #ifdef DEBUG_LOOKUP
             if (probe_key ==1928079863)
                printf("Tid: %d Probe key %u LESS than max_of_first %u \n", tid, probe_key, max_of_first_node);
            #endif 
                partition_offset = 0;
            }
            else
            {
                // perform tracing to find the partition

            #ifdef DEBUG_LOOKUP
             if (probe_key ==1928079863)
                printf("Tid: %d Probe key %u GO SEARCH WITH RAYS \n", tid, probe_key);
            #endif 
            
                partition_offset = cg::find_partition_offset_with_rays(params.traversable, params.partition_count, lower_bound);

             #ifdef DEBUG_LOOKUP
             if (probe_key ==1928079863)
                printf("Tid: %d FIRED RAYS Probe key %u partition_offset %d \n", tid, partition_offset);
            #endif 
            }

            // added Rosina
            if (partition_offset == not_found)
            {
#ifdef DEBUG_LOOKUP
                //printf("Tid: %d LOOKUP PARTITION_OFFSET NOT FOUND: probe_key %u:  \n", tid, probe_key);
#endif
                params.result[ix] = not_found;
                return;
             // --no for loop               continue;
            }

            // use partition offset to find the first entry node in the Bucket
            auto curr_node = reinterpret_cast<uint8_t *>(buf) + node_stride * partition_offset;

            smallsize last_position_value = 0; // extract_key_node<key_type>(curr_node, lastpositionptr);
            key_type curr_max = cg::extract<key_type>(curr_node, 0);
            #ifdef DEBUG_LOOKUP
            if (probe_key ==1928079863)
                printf("Tid: %d LOOKUP FOR key %u curr_max %u \n", tid, probe_key, curr_max);
            #endif 
            while (curr_max < probe_key)
            {
                last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
                last_position_value--; // decrement b/c it is inserted into node as +1 to avoid 0
#ifdef DEBUG_LOOKUP
                // printf("Lookups: tid:%d, Traverse Links: curr_max %llu is less than probe_key %llu, last_position_value %u \n", tid, curr_max, probe_key, last_position_value);
#endif
                                       // print_node<key_type>(curr_node, partition_size);

                // check for error
                if (last_position_value >= allocation_buffer_count)
                {
#ifdef ERROR_CHECK
                  printf("Tid: %d LOOKUP ERROR: LAST PTR EXCEEDS NUM PARTITIONS: last_position_value: %u, node max: %llu Search Key %llu \n", tid, last_position_value, curr_max, probe_key);
#endif
                    return;
                }
                curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
            smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));


            /* if (probe_key ==1248854015) {
                DEBUG_LOOKUP_NODES("After While loop LOOKUP: Tid: Going to look for key ",probe_key, curr_max, curr_size);
                 print_node<key_type>(curr_node, partition_size*2);
            } */

            smallsize found_index = 1;
            //bool probe_key_found = binary_search_in_cuda_buffer<key_type>(curr_node, curr_size, probe_key, found_index, tid);
           // bool probe_key_found = linear_search_in_cuda_buffer_with_tombstones_full<key_type>(curr_node, curr_size, probe_key, found_index, tid);
            bool probe_key_found = linear_search_in_cuda_buffer_with_tombstones_full<key_type>(curr_node, curr_size, node_size, probe_key, found_index, tid);

            if (!probe_key_found)
            {
                params.result[ix] = not_found;

               // if (probe_key ==1248854015)DEBUG_LOOKUP_NODES("Tid: LOOKUP: NOT FOUND: Not present in node", tid, probe_key);
                    //print_node<key_type>(curr_node, partition_size);
         
                return;
                // ---> no for looop   continue; // move on to next key to insert
            }
            // found the key
            smallsize result;
            result = extract_offset_node<key_type>(curr_node, found_index);
            params.result[ix] = result;

        }
        
   }

   
 #ifdef PRINT_LOOKUPS_END
   // __syncthreads();
    if (tid == 0)
    {
        printf("END LOOKUPS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(&params, tid);
    }

#endif

}

extern "C" GLOBALQUALIFIER void __raygen__lookup_nodes()
{
    if (params.long_keys)
    {
#ifdef TOMBSTONES
#pragma message "TOMBSTONES=YES"
        perform_lookup_nodes_tombstones<key64>();
#else 
#pragma message "TOMBSTONES=NO"  
        perform_lookup_nodes<key64>();
#endif
    }
    else
    {

#ifdef TOMBSTONES
#pragma message "TOMBSTONES=YES"
        perform_lookup_nodes_tombstones<key32>();
#else   
#pragma message "TOMBSTONES=NO"
        perform_lookup_nodes<key32>();
#endif

    }
}
