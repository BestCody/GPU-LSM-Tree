// =============================================================================
// File: coarse_granular_inserts_additions.cuh
// Author: Rosina Kharal
// Description: Implements coarse_granular_inserts_additions - this is the first file we added for the inserts additions, 
// it includes the split_curr_node function and process_inserts_per_bucket_tombstones function which are called in the combined kernel.
// This file also contains variations of insertion functions that were used for testing, but are not called in the final version of FliX.
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_INSERTS_ADDITIONS_CUH
#define INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_INSERTS_ADDITIONS_CUH

#include "coarse_granular_inserts.cuh"
#include "definitions_updates.cuh"

//#define MAX_NODE 16 // 14 keys per node

//------------------------ LOCAL MEM
template <typename key_type>
DEVICEQUALIFIER void split_curr_node(
    void *curr_node, updatable_cg_params *launch_params, int tid)
{
    // Read parameters from launch_params
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize node_stride = launch_params->node_stride;
    smallsize free_index = launch_params->free_node;
    smallsize node_size = launch_params->node_size;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;

    if (free_index >= allocation_buffer_count)
    {
        ERROR_INSERTS("ERROR: Node Split: Allocation Buffer is Full:", tid, free_index);
        return;
    }

    // Atomically allocate a new node
    smallsize free_value = atomicAdd(&launch_params->free_node, 1ULL);
    auto linked_node = reinterpret_cast<uint8_t *>(allocation_buffer) + node_stride * free_value;

    if (free_value >= allocation_buffer_count)
    {
        ERROR_INSERTS("ERROR: Node Split: Allocation Buffer is Full:", tid, free_value);
        return;
    }

    // Determine the split point
    smallsize mid = node_size / 2;

    // Move half of the elements from curr_node to linked_node
    for (smallsize i = mid + 1; i <= node_size; ++i)
    {
        key_type key = extract_key_node<key_type>(curr_node, i);
        smallsize offset = extract_offset_node<key_type>(curr_node, i);

        set_key_node<key_type>(linked_node, i - mid, key);
        set_offset_node<key_type>(linked_node, i - mid, offset);

        // Clear old entries in curr_node
        set_key_node<key_type>(curr_node, i, static_cast<key_type>(0));
        set_offset_node<key_type>(curr_node, i, static_cast<smallsize>(0));
    }

    // Update size metadata
    cg::set<smallsize>(curr_node, sizeof(key_type), mid);
    cg::set<smallsize>(linked_node, sizeof(key_type), node_size - mid);

    // Update max values for curr_node and linked_node
    key_type original_max = cg::extract<key_type>(curr_node, 0);
    key_type curr_max = extract_key_node<key_type>(curr_node, mid);
    cg::set<key_type>(curr_node, 0, curr_max);
    cg::set<key_type>(linked_node, 0, original_max);

    // Update linked pointer
    smallsize last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
    if (last_position_value != 0)
    {
        cg::set<smallsize>(linked_node, lastposition_bytes, last_position_value);
        cg::set<smallsize>(curr_node, lastposition_bytes, free_value + 1);
    }
    else
    {
        cg::set<smallsize>(curr_node, lastposition_bytes, free_value + 1);
    }
}

template <typename key_type>
DEVICEQUALIFIER void process_inserts_per_bucket_tombstones(key_type maxkey, smallsize minindex, smallsize maxindex, updatable_cg_params *launch_params, void *curr_node, key_type num_elements, int tid)
{
    // Compute tid here
    smallsize idx = blockIdx.x * blockDim.x + threadIdx.x;
    // smallsize totalstoredsize = launch_params->stored_size;
    void *allocation_buffer = launch_params->allocation_buffer;
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
    // last index is used for next pointer
#ifdef PRINT_PROCESS_INSERTS
    const key_type *maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    key_type maxkey_forthisthread = maxbuf[idx]; // curr_node[0]; // Assuming the key to search is at position 2
#endif

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
        smallsize ts_count = 0;
        smallsize j = 0;            // Index for keys and offsets
        smallsize update_idx = k;   // Index for update_list
        smallsize insert_count = 0; // Counter for new insertions

        if (next_insert_key > curr_max)
            printf("ERROR in while loop: Tid: %d, curr_max %u, next_insert_key %u\n", tid, curr_max, next_insert_key);

        for (smallsize i = 1; i <= curr_size; ++i)
        {
            key_type curr_key = extract_key_node<key_type>(curr_node, i);
            smallsize curr_offset = extract_offset_node<key_type>(curr_node, i);

            while (curr_key == tombstone)
            {
                i++;
                curr_key = extract_key_node<key_type>(curr_node, i);
                curr_offset = extract_offset_node<key_type>(curr_node, i);
                ts_count++;
            }
            // if (tid == 5)
            //{
            DEBUG_PI_BUCKET("Extracted Key/Offset from Curr_Node", tid, curr_key, curr_offset);
            //}

            // Insert all keys from update_list that are less than the current key
            while (insert_count + curr_size < (node_size + ts_count) && update_idx <= maxindex && update_list[update_idx] < curr_key && j < node_size)
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
        while ((insert_count + curr_size < (node_size + ts_count)) && update_idx <= maxindex && update_list[update_idx] <= curr_max && j < node_size)
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

      
        copy_arrays_to_node<key_type>(curr_node, keys_array, offsets_array, j, node_size, launch_params, tid);

        k = update_idx;
    }

#ifdef PRINT_PROCESS_INSERTS
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

//---------------------- Process Inserts With Buckets With Tombstones With CUDA BUFFER

template <typename key_type>
DEVICEQUALIFIER void process_inserts_per_bucket_tombstones_cudabuffers(key_type maxkey, smallsize minindex, smallsize maxindex, updatable_cg_params *launch_params, void *curr_node, key_type num_elements, int tid)
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

#ifdef PRINT_PROCESS_INSERTS
    if (tid == 0)
    {
        printf("Tid: %d, PRINTING UPDATE LIST\n", tid);
        for (int i = minindex; i <= maxindex; i++)
        {
            printf("Tid: %d, update_list[%d] %llu\n", tid, i, update_list[i]);
        }
    }
#endif

    smallsize k = minindex;

    while (k <= maxindex)
    {
        key_type next_insert_key = update_list[k];

        smallsize last_position_value = 0; // extract_key_node<key_type>(curr_node, lastpositionptr);
        key_type curr_max = cg::extract<key_type>(curr_node, 0);

        // if (next_insert_key == 4004458884) DEBUG_PI_BUCKET("While Loop Process Inserts per Bucket", tid, curr_max, next_insert_key);

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
        smallsize init_size = curr_size;
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
                curr_size++;
                ts_count++;
                continue;
            }

            // if (next_insert_key == 4004458884) DEBUG_PI_BUCKET("Extracted Key/Offset from Curr_Node", tid, curr_key, curr_offset);

            // Insert all keys from update_list that are less than the current key
            while (insert_count + init_size < (node_size) && update_idx <= maxindex && update_list[update_idx] < curr_key && j <= node_size)
            {
                set_key_node<key_type>(copy_node, j, update_list[update_idx]);
                set_offset_node<key_type>(copy_node, j, offset_list[update_idx]);

                // if (update_list[update_idx] == 4004458884) DEBUG_PI_BUCKET("Inserted Update List into Copy Node", tid, update_list[update_idx], offset_list[update_idx]);

                j++;
                update_idx++;
                insert_count++;
            }

            // If the current key matches a key in update_list, update the offset
            if (update_idx <= maxindex && update_list[update_idx] == curr_key)
            {
                curr_offset = offset_list[update_idx];

                // if (update_list[update_idx] == 4004458884) DEBUG_PI_BUCKET("Updated Offset", tid, curr_key, curr_offset);

                update_idx++;
            }

            // Copy current key and offset
            ///--->  if(j > node_size) printf("ERROR in J : Tid: %d, curr_max %u, curr_size %u j= %d \n", tid, curr_max, curr_size, j);

            set_key_node<key_type>(copy_node, j, curr_key);
            set_offset_node<key_type>(copy_node, j, curr_offset);

            // if (next_insert_key == 4004458884) DEBUG_PI_BUCKET("Copied Curr_Node into Copy Node", tid, curr_key, curr_offset);

            j++;
        }

        // Insert any remaining keys from update_list
        while (insert_count + init_size < (node_size) && update_idx <= maxindex && update_list[update_idx] <= curr_max && j <= node_size)
        {
            set_key_node<key_type>(copy_node, j, update_list[update_idx]);
            set_offset_node<key_type>(copy_node, j, offset_list[update_idx]);

            // if (next_insert_key == 4004458884) DEBUG_PI_BUCKET("Inserted Remaining Update List into Copy Node", tid, update_list[update_idx], offset_list[update_idx]);

            j++;
            update_idx++;
            insert_count++;
        }

        // Copy the arrays back to curr_no
        // DEBUG_PI_BUCKET("Before Copy Arrays to Node", tid, curr_size, j);
        // if( j > node_size+1) printf("ERROR in J : Tid: %d, curr_max %u, curr_size %u j= %d \n", tid, curr_max, curr_size, j);
        copy_arrays_to_node_cudabuffer<key_type>(curr_node, copy_node, j - 1, node_size, ts_count, launch_params, tid);

        k = update_idx;
    }

#ifdef PRINT_PROCESS_INSERTS
    __syncthreads();
    if (tid == 19)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes<key_type>(launch_params, tid);
    }

    /*
        // Loop to print all values in maxbuf
        if (tid == 34) {
        printf("END INSERTIONS number of partitions count with overflow %u \n", partition_count_with_overflow);
        for (smallsize i = 0; i < partition_count_with_overflow; ++i) {
            key_type value = maxbuf[i];
            printf("MAX BUFFER Index: %d, Value: %u \n", i, value);
        }
        printf("TID: %d, Max Key for this thread: %u\n", tid, maxkey_forthisthread);
        printf("************************************\n");
        }
        */

#endif
}



//----------------------------------------------- Process Inserts With Buckets With Tombstones With CUDA BUFFER UPDATED ----------------------------
template <typename key_type>
DEVICEQUALIFIER void process_inserts_per_bucket_tombstones_cudabuffers_updated(key_type maxkey, smallsize minindex, smallsize maxindex, updatable_cg_params *launch_params, void *curr_node, key_type num_elements, int tid)
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

#ifdef PRINT_PROCESS_INSERTS
    if (tid == 0)
    {
        printf("Tid: %d, PRINTING UPDATE LIST\n", tid);
        for (int i = minindex; i <= maxindex; i++)
        {
            printf("Tid: %d, update_list[%d] %llu\n", tid, i, update_list[i]);
        }
    }
#endif

    smallsize k = minindex;
    while (k <= maxindex)
    {
        key_type next_insert_key = update_list[k];
        smallsize last_position_value = 0; // extract_key_node<key_type>(curr_node, lastpositionptr);
        key_type curr_max = cg::extract<key_type>(curr_node, 0);

        // if (next_insert_key == 4004458884) DEBUG_PI_BUCKET("While Loop Process Inserts per Bucket", tid, curr_max, next_insert_key);

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

        if (curr_max < next_insert_key)
            printf("ERROR after while loop: Tid: %d, curr_max %u, next_insert_key %u\n", tid, curr_max, next_insert_key);
        // DEBUG_PI_BUCKET("Found the correct bucket", tid, curr_max, next_insert_key);

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        smallsize ts_count = 0;
        smallsize j = 1;            // Index for keys and offsets
        smallsize update_idx = k;   // Index for update_list
        smallsize insert_count = 0; // Counter for new insertions

        // ----> if(next_insert_key > curr_max) printf("ERROR in while loop: Tid: %d, curr_max %u, next_insert_key %u\n", tid, curr_max, next_insert_key);
        smallsize init_size = curr_size;
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
                curr_size++;
                ts_count++;
                continue;
            }

            // if (next_insert_key == 4004458884) DEBUG_PI_BUCKET("Extracted Key/Offset from Curr_Node", tid, curr_key, curr_offset);

            // Insert all keys from update_list that are less than the current key
            while (insert_count + init_size < (node_size) && update_idx <= maxindex && update_list[update_idx] < curr_key && j <= node_size)
            {
                set_key_node<key_type>(copy_node, j, update_list[update_idx]);
                set_offset_node<key_type>(copy_node, j, offset_list[update_idx]);

                // if (update_list[update_idx] == 4004458884) DEBUG_PI_BUCKET("Inserted Update List into Copy Node", tid, update_list[update_idx], offset_list[update_idx]);

                j++;
                update_idx++;
                insert_count++;
            }

            // If the current key matches a key in update_list, update the offset
            if (update_idx <= maxindex && update_list[update_idx] == curr_key)
            {
                curr_offset = offset_list[update_idx];

                // if (update_list[update_idx] == 4004458884) DEBUG_PI_BUCKET("Updated Offset", tid, curr_key, curr_offset);

                update_idx++;
            }

            // Copy current key and offset
            ///--->  if(j > node_size) printf("ERROR in J : Tid: %d, curr_max %u, curr_size %u j= %d \n", tid, curr_max, curr_size, j);

            set_key_node<key_type>(copy_node, j, curr_key);
            set_offset_node<key_type>(copy_node, j, curr_offset);

            // if (next_insert_key == 4004458884) DEBUG_PI_BUCKET("Copied Curr_Node into Copy Node", tid, curr_key, curr_offset);

            j++;
        }

        // Insert any remaining keys from update_list
        while (insert_count + init_size < (node_size) && update_idx <= maxindex && update_list[update_idx] <= curr_max && j <= node_size)
        {
            set_key_node<key_type>(copy_node, j, update_list[update_idx]);
            set_offset_node<key_type>(copy_node, j, offset_list[update_idx]);

            // if (next_insert_key == 4004458884) DEBUG_PI_BUCKET("Inserted Remaining Update List into Copy Node", tid, update_list[update_idx], offset_list[update_idx]);

            j++;
            update_idx++;
            insert_count++;
        }

        // Copy the arrays back to curr_no
        // DEBUG_PI_BUCKET("Before Copy Arrays to Node", tid, curr_size, j);
        // if( j > node_size+1) printf("ERROR in J : Tid: %d, curr_max %u, curr_size %u j= %d \n", tid, curr_max, curr_size, j);
        // DEBUG_PI_BUCKET("going to copy array", tid, j-1, insert_count);
        bool complete_inserts = (update_idx > maxindex);

        if ((insert_count == 0) && (init_size == node_size))
        {
            split_curr_node<key_type>(curr_node, launch_params, tid);
        }
        else if (insert_count > 0)
        {
            // if(insert_count > 0)
            copy_arrays_to_node_cudabuffer<key_type>(curr_node, copy_node, j - 1, node_size, ts_count, launch_params, complete_inserts, tid);
        }
        k = update_idx;
    }

#ifdef PRINT_PROCESS_INSERTS
    __syncthreads();
    if (tid == 19)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes<key_type>(launch_params, tid);
    }

    /*
        // Loop to print all values in maxbuf
        if (tid == 34) {
        printf("END INSERTIONS number of partitions count with overflow %u \n", partition_count_with_overflow);
        for (smallsize i = 0; i < partition_count_with_overflow; ++i) {
            key_type value = maxbuf[i];
            printf("MAX BUFFER Index: %d, Value: %u \n", i, value);
        }
        printf("TID: %d, Max Key for this thread: %u\n", tid, maxkey_forthisthread);
        printf("************************************\n");
        }
        */

#endif
}

template <typename key_type>
DEVICEQUALIFIER void split_node_localmem(
    void *curr_node, key_type *local_keys, smallsize *local_offsets, smallsize num_elements,
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
        key_type key = local_keys[i - 1];
        smallsize offset = local_offsets[i - 1];

        set_key_node<key_type>(curr_node, i, key);
        set_offset_node<key_type>(curr_node, i, offset);
    }

    for (smallsize i = mid + 1; i <= num_elements; ++i) // Continue from mid + 1
    {
        key_type key = local_keys[i - 1];
        smallsize offset = local_offsets[i - 1];

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
    }
#endif

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

    /* if (tid == 3)
     {
         DEBUG_PI_BUCKET_SPLIT("After Links Set", tid, last_position_value);
         print_node<key_type>(curr_node, node_size);
         print_node<key_type>(linked_node, node_size);
     } */
}

template <typename key_type>
DEVICEQUALIFIER void copy_arrays_to_node_localmem(
    void *curr_node, key_type *local_keys, smallsize *local_offsets, smallsize num_elements,
    smallsize node_size, smallsize tomb_stone_count, updatable_cg_params *launch_params, bool completed_inserts, int tid)
{
   

   // printf("TID: %d, CHECK SPLIT: num elements %u, completed_inserts %d\n", tid, num_elements, completed_inserts);

    if (num_elements >= node_size && !completed_inserts) // not causing extra node splits
    {
        split_node_localmem<key_type>(curr_node, local_keys, local_offsets, num_elements, node_size, launch_params, tid);
        return;
    }
    smallsize m;
    // Copy values from local memory to curr_node
    for (m = 0; m < num_elements; ++m)
    {
        key_type key = local_keys[m];
        smallsize offset = local_offsets[m];

        set_key_node<key_type>(curr_node, m + 1, key);
        set_offset_node<key_type>(curr_node, m + 1, offset);
    }

    // Update the size of curr_node
    cg::set<smallsize>(curr_node, sizeof(key_type), num_elements);

    for (int i = 1; i <= tomb_stone_count; ++i)
    {
        if (m >= node_size)
        {
            break;
        }
        set_key_node<key_type>(curr_node, m + 1, 0);
        set_offset_node<key_type>(curr_node, m + 1, 0);
        m++;
    }
}

//--------------------------------------- LOCAL MEM- BULK INSERTIONS ONLY --------------------------------------------

template <typename key_type>
DEVICEQUALIFIER void process_inserts_per_bucket_tombstones_localmem(
    key_type maxkey, smallsize minindex, smallsize maxindex,
    updatable_cg_params *launch_params, void *curr_node,
    key_type num_elements, int tid)
{
    smallsize idx = blockIdx.x * blockDim.x + threadIdx.x;
    void *allocation_buffer = launch_params->allocation_buffer;

    smallsize partition_size = launch_params->partition_size;
    smallsize node_size = launch_params->node_size;
    smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;
    smallsize node_stride = launch_params->node_stride;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);

    // Local thread memory for keys and offsets
    // key_type local_keys[max_node_size];
    // smallsize local_offsets[max_node_size];

    // Local thread memory for keys and offsets
    key_type local_keys[MAX_NODE];
    smallsize local_offsets[MAX_NODE];

    // Initialize local memory
    for (smallsize m = 0; m < node_size; ++m)
    {
        local_keys[m] = 0;
        local_offsets[m] = 0;
    }

    smallsize k = minindex;
    smallsize update_idx = minindex;

    while (update_idx <= maxindex)
    {
        // if (tid ==47) DEBUG_PI_BUCKET("TOP OF WHILE LOOP", update_idx, maxindex, minindex);
        key_type next_insert_key = update_list[update_idx];
        smallsize last_position_value = 0;
        key_type curr_max = cg::extract<key_type>(curr_node, 0);

        while (curr_max < next_insert_key)
        {
            // last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes) - 1;

            last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
            last_position_value--; // decrement b/c it is inserted into node as +1 to avoid 0

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
        smallsize j = 0, insert_count = 0;
        smallsize init_size = curr_size;

        /// update_idx = k,

        // if (tid ==47) {
        //  if (next_insert_key > curr_max) printf("ERROR in while loop: Tid: %d, curr_max %u, next_insert_key %u\n", tid, curr_max, next_insert_key);
        // }

        for (smallsize i = 1; i <= curr_size; ++i)
        {
            if (init_size == node_size)
                break; // normal split of this node required

            key_type curr_key = extract_key_node<key_type>(curr_node, i);
            smallsize curr_offset = extract_offset_node<key_type>(curr_node, i);

            // if (tid ==47) DEBUG_PI_BUCKET("Top for loop next key", tid, curr_key, curr_offset);

            if (curr_key == tombstone)
            {
                // if (tid ==47) DEBUG_PI_BUCKET("FOUND tombstone", tid, ts_count);
                curr_size++;
                ts_count++;
                continue;
            }

            while ((insert_count + init_size < node_size) && update_idx <= maxindex && update_list[update_idx] < curr_key && j < node_size)
            {
                // if (tid == 47) DEBUG_PI_BUCKET("Keep adding keys", tid, j, node_size);
                local_keys[j] = update_list[update_idx];
                local_offsets[j] = offset_list[update_idx];
                j++;
                update_idx++;
                insert_count++;
            }

            if (update_idx <= maxindex && update_list[update_idx] == curr_key)
            {
                // if (tid == 47) DEBUG_PI_BUCKET("Update Offset", tid, j, node_size);
                //curr_offset = offset_list[update_idx];
                update_idx++;
            }

            local_keys[j] = curr_key;
            local_offsets[j] = curr_offset;
            j++;
        }

        while ((insert_count + init_size < node_size) && update_idx <= maxindex && update_list[update_idx] <= curr_max && j < node_size)
        {

            // if (tid ==47) DEBUG_PI_BUCKET("keep adding update key", tid, update_list[update_idx], offset_list[update_idx]);
            local_keys[j] = update_list[update_idx];
            local_offsets[j] = offset_list[update_idx];
            j++;
            update_idx++;
            insert_count++;
        }

        bool complete_inserts = (update_idx > maxindex);

        if ((insert_count == 0) && (init_size == node_size))
        {
            split_curr_node<key_type>(curr_node, launch_params, tid);
        }
        else if (insert_count > 0)
        {
            // if (tid ==47) DEBUG_PI_BUCKET("Going to Copy", tid, j, node_size);
            copy_arrays_to_node_localmem<key_type>(curr_node, local_keys, local_offsets, j, node_size, ts_count, launch_params, complete_inserts, tid);
        }

        // printf("TID: %d, After Copying Arrays to Node, complete_inserts %d, j %u AND INSERT COUNT %D\n", tid, complete_inserts,j, insert_count);
    }

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();
    if (tid == 0)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, tid);
    }

#endif
}

//------ HYBRID

template <typename key_type>
DEVICEQUALIFIER void shift_insertion(
    void *curr_node, smallsize &curr_size, smallsize minindex, smallsize maxindex,
    updatable_cg_params *launch_params, smallsize &update_idx, int tid)
{
    smallsize node_size = launch_params->node_size; // Store in local memory for faster access
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);
    key_type curr_max = cg::extract<key_type>(curr_node, 0);

    while (update_idx <= maxindex)
    {
       // if (next_insert_key ==2658754282)

        key_type next_insert_key = update_list[update_idx];

       //if (next_insert_key == 2658754282){
       //      printf("IN SHIFT INSERT TID: %d, Next Insert Key: %u, Current Size: %u, Min Index: %u, Max Index: %u, Update Index: %u \n", tid, next_insert_key, curr_size, minindex, maxindex, update_idx);
       // }
        //if (tid == 555)DEBUG_PI_BUCKET(" Update Key Keep in Shift Insertion Loop", tid, update_idx, update_list[update_idx]);

        if (next_insert_key > curr_max)
            break; // Exit if next_insert_key is greater than curr_max

        // Case 1: If curr_size == node_size, split the node
        if (curr_size == node_size)
        {

            //if (tid == 555)DEBUG_PI_BUCKET("Before Split Insertion", tid, next_insert_key, curr_size);

            smallsize insert_index = 1;
            bool found = binary_search_in_cuda_buffer<key_type>(curr_node, curr_size, next_insert_key, insert_index, tid);

             //if (next_insert_key == 2658754282){
             //printf("Searched  TID: %d, Found %u, Next Insert Key: %u, Current Size: %u, Min Index: %u, Max Index: %u, Update Index: %u \n", tid,found, next_insert_key, curr_size, minindex, maxindex, update_idx);
            // }
            if (!found)
            { // insert the key, split node make and leave
                insert_split<key_type>(curr_node, insert_index, next_insert_key, offset_list[update_idx], launch_params, tid);
                update_idx++;
                return; // Exit after split
            }
            // otherwise continue to next key
            update_idx++;
        }
        else
        {

            // Case 2: If curr_size == node_size -1 or node_size -2

            smallsize insert_index = 1;
            //if (tid == 555)printf("TID: %d, Before Binary Search, Next Insert Key: %u, Current Size: %u, Min Index: %u, Max Index: %u, Update Index: %u\n", tid, next_insert_key, curr_size, minindex, maxindex, update_idx);
            
            bool found = binary_search_in_cuda_buffer<key_type>(curr_node, curr_size, next_insert_key, insert_index, tid);

            if (!found)
            {
                //if (tid == 555)printf("TID: %d, Not Found After Binary Search, Next Insert Key: %u, Current Size: %u, Currmax: %u, insert index: %u, Update Index: %u\n", tid, next_insert_key, curr_size, curr_max, insert_index, update_idx);
                    ///print_node<key_type>(curr_node, node_size);
               // }

                // Add Support for Unique Case
                smallsize thisoffset = offset_list[update_idx];
                key_type key_at_index = extract_key_node<key_type>(curr_node, insert_index);
                //smallsize lastkeyindex = node_size;

                if (key_at_index < next_insert_key) // This also is the case when size of list is 0
                {
                    insert_index = (curr_size == 0) ? 1 : insert_index + 1;
                    //if (tid ==555) printf( "going to perfrom _insert Key %u, Insert Index %u, Offset %u, TID %d \n", next_insert_key, insert_index, thisoffset, tid);
                    perform_insert<key_type>(curr_node, insert_index, static_cast<key64>(next_insert_key), thisoffset, tid);
                    curr_size++;
                    //if (tid == 555) DEBUG_PI_BUCKET("Unique Case PerformED Insert", tid, next_insert_key, insert_index, curr_size);
                    //{
                    //    DEBUG_PI_BUCKET("Unique Case PerformED Insert", tid, next_insert_key, insert_index, curr_size);
                        //print_node<key_type>(curr_node, node_size);
                   // }
                    // Note: if node is full, this will be caught below when we check for size
                }
                else
                {
                    perform_insert_shift<key_type>(curr_node, insert_index, curr_size, static_cast<key64>(next_insert_key), offset_list[update_idx]);
                    curr_size++;

                    //if (tid == 555)printf("TID: %d, Inserted Shift, Next Insert Key: %u, Current Size: %u, Currmax: %u, insert index: %u, Update Index: %u\n", tid, next_insert_key, curr_size, curr_max, insert_index, update_idx);
                    //print_node<key_type>(curr_node, node_size);
                
                   cg::set<smallsize>(curr_node, sizeof(key_type), curr_size);
                }
            }
        }
    
    update_idx++;

    }
    // if (curr_size == node_size)
    //  {
    //   return; // Exit after inserting max allowed keys
    //  }
}


//----------------------------------------------- Process Inserts With Buckets With Tombstones With Local Mem UPDATED HYBRID APPROACH ----------------------------
template <typename key_type>
DEVICEQUALIFIER
bool check_for_bulk_insertions(smallsize curr_size, smallsize node_size, smallsize minindex, smallsize maxindex) {
    if ( (curr_size <= node_size - 4) && (maxindex - minindex) >= 3) {
        return true;
    }
    return false;
}


//----------------------------------------------- Process Inserts With Buckets With Tombstones With Local Mem UPDATED HYBRID APPROACH ----------------------------
template <typename key_type>
DEVICEQUALIFIER
void check_shift_insertions (smallsize num_left_to_insert, void *curr_node, smallsize &curr_size, smallsize minindex, smallsize maxindex,
    updatable_cg_params *launch_params, smallsize &update_idx, int tid)
{

    smallsize node_size = launch_params->node_size;
//if (tid == 0) printf(" Next key to insert %u, curr_max %u, curr_size %u, num_left_to_insert %u, tid %d \n", next_insert_key, curr_max, curr_size, num_left_to_insert, tid);
        // }

        //if (curr_size >= (node_size - 2))      //hybrid1 ** USE THIS ONE
        smallsize node_max_fill_state = SDIV(node_size*node_threshold,100);  // variable argument at command line
        smallsize min_remaining_to_insert = min_keys_insert;
        //if (curr_size >= (node_size - 4))     //hybrid2
        //if  (curr_size > (node_size / 2)+2)   //hybrid3
        //if(bulk_insertion_condition)
        // if (num_left_to_insert <= 3)         //hybrid4
        if ( all_shift_insertions || (hybrid && ((curr_size > node_max_fill_state) || (num_left_to_insert <=min_remaining_to_insert) )  ))       //hybrid5
        //  if ( (curr_size > (node_size / 2)+2) || (num_left_to_insert <=3) )  //hybrid6
        // NOTE ADD SUPPORT FOR TOMBSTONES
        {
            
            shift_insertion<key_type>(curr_node, curr_size, minindex, maxindex, launch_params, update_idx, tid);
        }

        return;
    }

template <typename key_type>
DEVICEQUALIFIER void process_inserts_single_thread(
    key_type maxkey, smallsize minindex, smallsize maxindex,
    updatable_cg_params *launch_params, void *curr_node,
    key_type num_elements, int tid)
{
    smallsize idx = blockIdx.x * blockDim.x + threadIdx.x;
    void *allocation_buffer = launch_params->allocation_buffer;

    smallsize partition_size = launch_params->partition_size;
    smallsize node_size = launch_params->node_size;
    smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;
    smallsize node_stride = launch_params->node_stride;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);

    // Local thread memory for keys and offsets
    // key_type local_keys[max_node_size];
    // smallsize local_offsets[max_node_size];

    // Local thread memory for keys and offsets
    key_type local_keys[MAX_NODE];
    smallsize local_offsets[MAX_NODE];
    ///printf("TOP BULK insertions MAX NODE SIZE %u \n", MAX_NODE);

    // Initialize local memory
    for (smallsize m = 0; m < node_size; ++m)
    {
        local_keys[m] = 0;
        local_offsets[m] = 0;
    }

    smallsize k = minindex;
    smallsize update_idx = minindex;

    smallsize total_inserts = maxindex - minindex + 1;


    //DEBUG_PI_BUCKET_NUMINSERTS("My TOTAL IN Process Inserts BUCKETS: Tid: ", tid,total_inserts, maxindex);

    while (update_idx <= maxindex)
    {
        // if (tid ==47) DEBUG_PI_BUCKET("TOP OF WHILE LOOP", update_idx, maxindex, minindex);
        key_type next_insert_key = update_list[update_idx];
        smallsize last_position_value = 0;
        key_type curr_max = cg::extract<key_type>(curr_node, 0);

       // printf("Top WHILE TID: %d, Next Insert Key: %u, Current Max: %u, minindex %d, maxindex %d, update_idx \n", tid, next_insert_key, curr_max, minindex, maxindex, update_idx);
        while (curr_max < next_insert_key)
        {
            // last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes) - 1;

            last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
            last_position_value--; // decrement b/c it is inserted into node as +1 to avoid 0

            if (last_position_value >= allocation_buffer_count)
            {
                ERROR_INSERTS("ERROR: LAST PTR EXCEEDS NUM PARTITIONS", tid, curr_max, partition_count_with_overflow);
                return;
            }
            curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            curr_max = cg::extract<key_type>(curr_node, 0);
        }

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        smallsize num_left_to_insert = maxindex - update_idx + 1;

        //------------------ SHIFT RIGHT INSERTIONS ONLY --------------------------

        if( hybrid || all_shift_insertions) {
            
            //printf("perform shift right insertions \n");
                //check_shift_insertions<key_type>(num_left_to_insert, curr_node, curr_size, minindex, maxindex,launch_params, update_idx, tid);
                
                //if (curr_size >= (node_size - 2))      //hybrid1 ** USE THIS ONE
            smallsize node_max_fill_state = SDIV(node_size*node_threshold,100);  // variable argument at command line 
            smallsize min_remaining_to_insert = min_keys_insert;
         
            // ---  2 PARAM:   if ( all_shift_insertions || (hybrid && ((curr_size >= node_max_fill_state) || (num_left_to_insert <=min_remaining_to_insert) )  ))       //hybrid5
            if ( all_shift_insertions || (hybrid && (num_left_to_insert <=min_remaining_to_insert) )  )       //hybrid5
            //  if ( (curr_size > (node_size / 2)+2) || (num_left_to_insert <=3) )  //hybrid6
            // NOTE ADD SUPPORT FOR TOMBSTONES
            {
                
                shift_insertion<key_type>(curr_node, curr_size, minindex, maxindex, launch_params, update_idx, tid);
               // if (next_insert_key ==2658754282)printf("tid:  %d Functin call SHIFT RIGHT ONLY \n", tid);
                continue;
            }

          // if (next_insert_key ==2658754282)printf("tid:  %d Bottom SHIFT RIGHT ONLY \n", tid);
            
        }


        smallsize ts_count = 0;
        smallsize j = 0, insert_count = 0;
        smallsize init_size = curr_size;


       // printf("perform BULK insertions MAX NODE SIZE %u \n", MAX_NODE);

       
    //------------------ BULK INSERTIONS ONLY --------------------------
        for (smallsize i = 1; i <= curr_size; ++i)
        {

            if(init_size == node_size)
              break;
            //} //break; // normal split of this node required

            key_type curr_key = extract_key_node<key_type>(curr_node, i);
            smallsize curr_offset = extract_offset_node<key_type>(curr_node, i);

            //if (curr_key == 4088163018)
             DEBUG_PI_BUCKET("Top For Looop", tid, curr_key, curr_offset);
            //if (update_list[update_idx] == 300134)
             DEBUG_PI_BUCKET(" Top For Loop UpdateKey", tid, curr_key, update_list[update_idx]);

            // if (tid ==47) DEBUG_PI_BUCKET("Top for loop next key", tid, curr_key, curr_offset);

            if (curr_key == tombstone)
            {
                //if (curr_key == 2807674411)
                  //  DEBUG_PI_BUCKET("FOUND tombstone", tid, ts_count);
                curr_size++;
                ts_count++;
                continue;
            }

            while ((insert_count + init_size < node_size) && update_idx <= maxindex && update_list[update_idx] < curr_key && j < node_size)
            {
                //if (curr_key == 53183598)
                  // DEBUG_PI_BUCKET("Keep adding keys", tid, curr_key, update_list[update_idx]);
                //if (update_list[update_idx] == 2658754282)
                  //      DEBUG_PI_BUCKET(" Found Update Key Keep adding keys", tid, curr_key, update_list[update_idx]);

                local_keys[j] = update_list[update_idx];
                local_offsets[j] = offset_list[update_idx];
              //   printf("Inserted One from Insert List TID %d: local_keys[%d] = %u, local_offsets[%d] = %u\n", tid, j, (unsigned int)local_keys[j], j, (unsigned int)local_offsets[j]);
           
                j++;
                update_idx++;
                insert_count++;
            }

            if (update_idx <= maxindex && update_list[update_idx] == curr_key)
            {
                DEBUG_PI_BUCKET("Update Offset", tid, j, node_size);
                // ---> curr_offset = offset_list[update_idx];
                update_idx++;
            }

            local_keys[j] = curr_key;
            local_offsets[j] = curr_offset;

          //  printf("Done one round of For Loop TID %d: local_keys[%d] = %u, local_offsets[%d] = %u\n", tid, j, (unsigned int)local_keys[j], j, (unsigned int)local_offsets[j]);
            j++;
        }


       // AFTER WHILE LOOP
        // Print keys and offsets pair by pair
        //printf("TID %d: After For Loop, j = %u, node_size = %u, insert_count = %u, init_size = %u \n", tid, j, node_size, insert_count, init_size);
           //     for (int idx = 0; idx < j; ++idx)
           //     {
            //        printf("TID %d: local_keys[%d] = %u, local_offsets[%d] = %u \n", tid, idx, (unsigned int)local_keys[idx], idx, (unsigned int)local_offsets[idx]);
            //    }


        while ((insert_count + init_size < node_size) && update_idx <= maxindex && update_list[update_idx] <= curr_max && j < node_size)
        {
            //if (update_list[update_idx] == 4088163018)
          //  printf("ERROR in While Loop: TID %d, update_idx %u, maxindex %u, update_list[update_idx] = %u \n", tid, update_idx, maxindex, update_list[update_idx]);
            DEBUG_PI_BUCKET(" Lower While Loop Keep adding keys", tid, update_list[update_idx]);

            // if (tid ==47) DEBUG_PI_BUCKET("keep adding update key", tid, update_list[update_idx], offset_list[update_idx]);
            local_keys[j] = update_list[update_idx];
            local_offsets[j] = offset_list[update_idx];
            j++;
            update_idx++;
            insert_count++;
        }

        bool complete_inserts = (update_idx > maxindex);
        // printf("TID: %d, Bottom of While Loop: tid %d complete inserts %d update_idx %d \n", tid, complete_inserts, update_idx);

        if ((insert_count == 0 ) && (init_size == node_size))
        {
            split_curr_node<key_type>(curr_node, launch_params, tid);
        }
        else if (insert_count > 0)
        {
            DEBUG_PI_BUCKET("Going to Copy", tid, j, node_size);
            DEBUG_PI_BUCKET("Going to Copy", complete_inserts, ts_count);
            // Print keys and offsets pair by pair
              //  for (int idx = 0; idx < j; ++idx)
              //  {
              //      printf("TID %d: local_keys[%d] = %u, local_offsets[%d] = %u \n", tid, idx, (unsigned int)local_keys[idx], idx, (unsigned int)local_offsets[idx]);
              //  }
            copy_arrays_to_node_localmem<key_type>(curr_node, local_keys, local_offsets, j, node_size, ts_count, launch_params, complete_inserts, tid);
        }

        // printf("TID: %d, After Copying Arrays to Node, complete_inserts %d, j %u AND INSERT COUNT %D\n", tid, complete_inserts,j, insert_count);
    }

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();

   /*  if (tid == 0)
    {
        printf("END INSERTIONS: PRINT NODE 555\n");
        print_node<key_type>(curr_node, node_size);
    } */

    if (tid == 1)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, tid);
    }

#endif
}


template <typename key_type>
DEVICEQUALIFIER void process_inserts_per_bucket_tombstones_localmem_hybrid(
    key_type maxkey, smallsize minindex, smallsize maxindex,
    updatable_cg_params *launch_params, void *curr_node,
    key_type num_elements, int tid)
{
    smallsize idx = blockIdx.x * blockDim.x + threadIdx.x;
    void *allocation_buffer = launch_params->allocation_buffer;

    smallsize partition_size = launch_params->partition_size;
    smallsize node_size = launch_params->node_size;
    smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;
    smallsize node_stride = launch_params->node_stride;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);

    // Local thread memory for keys and offsets
    // key_type local_keys[max_node_size];
    // smallsize local_offsets[max_node_size];

    // Local thread memory for keys and offsets
    key_type local_keys[MAX_NODE];
    smallsize local_offsets[MAX_NODE];

    // Initialize local memory
    for (smallsize m = 0; m < node_size; ++m)
    {
        local_keys[m] = 0;
        local_offsets[m] = 0;
    }

    smallsize k = minindex;
    smallsize update_idx = minindex;

    smallsize total_inserts = maxindex - minindex + 1;


    //DEBUG_PI_BUCKET_NUMINSERTS("My TOTAL IN Process Inserts BUCKETS: Tid: ", tid,total_inserts, maxindex);

    while (update_idx <= maxindex)
    {
        // if (tid ==47) DEBUG_PI_BUCKET("TOP OF WHILE LOOP", update_idx, maxindex, minindex);
        key_type next_insert_key = update_list[update_idx];
        smallsize last_position_value = 0;
        key_type curr_max = cg::extract<key_type>(curr_node, 0);

        // printf("Top WHILE TID: %d, Next Insert Key: %u, Current Max: %u, minindex %d, maxindex %d, update_idx \n", tid, next_insert_key, curr_max, minindex, maxindex, update_idx);
        while (curr_max < next_insert_key)
        {
            // last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes) - 1;

            last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
            last_position_value--; // decrement b/c it is inserted into node as +1 to avoid 0

            if (last_position_value >= allocation_buffer_count)
            {
                ERROR_INSERTS("ERROR: LAST PTR EXCEEDS NUM PARTITIONS", tid, curr_max, partition_count_with_overflow);
                return;
            }
            curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            curr_max = cg::extract<key_type>(curr_node, 0);
        }

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // check if next_insert_key belongs in this curr_node, but the curr_node is full,
        // then split the node and insert the key in the correct node
        // ---->> HERE
        
       
       // bool bulk_insertion_condition = check_for_bulk_insertions<key_type>(curr_size, node_size, minindex, maxindex);
        smallsize num_left_to_insert = maxindex - update_idx + 1;

        //if (tid == 0) printf(" Next key to insert %u, curr_max %u, curr_size %u, num_left_to_insert %u, tid %d \n", next_insert_key, curr_max, curr_size, num_left_to_insert, tid);
        // }

        //if (curr_size >= (node_size - 2))      //hybrid1 ** USE THIS ONE
        //if (curr_size >= (node_size - 4))     //hybrid2
        //if  (curr_size > (node_size / 2)+2)   //hybrid3
        //if(bulk_insertion_condition)
        // if (num_left_to_insert <= 3)         //hybrid4

        /* *************
        if ( (curr_size > (node_size -4 )) || (num_left_to_insert <=3) )       //hybrid5
        //  if ( (curr_size > (node_size / 2)+2) || (num_left_to_insert <=3) )  //hybrid6
        // NOTE ADD SUPPORT FOR TOMBSTONES
        {
            // possibly changes curr_node to a new node and update_idx to another key.
            // this function also splits if needed
           // if (tid == 555)
           // {
            //    DEBUG_PI_BUCKET("Before shift Insertion", tid, update_list[update_idx], curr_size);
            //    print_node<key_type>(curr_node, node_size);
            //}

            shift_insertion<key_type>(curr_node, curr_size, minindex, maxindex, launch_params, update_idx, tid);
           
           // if (tid == 555)
           // {
            //    DEBUG_PI_BUCKET("AFTER shift Insertion", tid, update_list[update_idx], curr_size);
            //    print_node<key_type>(curr_node, node_size);
            //}
           // printf("after shift insertions tid %d  update_idx %u  maxindex %u \n", tid, update_idx, maxindex);
            continue; // Restart the loop from the top
        }
        ************** */


        smallsize ts_count = 0;
        smallsize j = 0, insert_count = 0;
        smallsize init_size = curr_size;

        /// update_idx = k,

        // if (tid ==47) {
        //  if (next_insert_key > curr_max) printf("ERROR in while loop: Tid: %d, curr_max %u, next_insert_key %u\n", tid, curr_max, next_insert_key);
        // }

        for (smallsize i = 1; i <= curr_size; ++i)
        {

            if(init_size == node_size)
              break;
            //} //break; // normal split of this node required

            key_type curr_key = extract_key_node<key_type>(curr_node, i);
            smallsize curr_offset = extract_offset_node<key_type>(curr_node, i);

            //if (curr_key == 4088163018)
             DEBUG_PI_BUCKET("Top For Looop", tid, curr_key, curr_offset);
            //if (update_list[update_idx] == 300134)
             DEBUG_PI_BUCKET(" Top For Loop UpdateKey", tid, curr_key, update_list[update_idx]);

            // if (tid ==47) DEBUG_PI_BUCKET("Top for loop next key", tid, curr_key, curr_offset);

            if (curr_key == tombstone)
            {
                //if (curr_key == 2807674411)
                  //  DEBUG_PI_BUCKET("FOUND tombstone", tid, ts_count);
                curr_size++;
                ts_count++;
                continue;
            }

            while ((insert_count + init_size < node_size) && update_idx <= maxindex && update_list[update_idx] < curr_key && j < node_size)
            {
                //if (curr_key == 53183598)
                  // DEBUG_PI_BUCKET("Keep adding keys", tid, curr_key, update_list[update_idx]);
                //if (update_list[update_idx] == 4088163018)
                DEBUG_PI_BUCKET(" Found Update Key Keep adding keys", tid, curr_key, update_list[update_idx]);

                local_keys[j] = update_list[update_idx];
                local_offsets[j] = offset_list[update_idx];
              //   printf("Inserted One from Insert List TID %d: local_keys[%d] = %u, local_offsets[%d] = %u\n", tid, j, (unsigned int)local_keys[j], j, (unsigned int)local_offsets[j]);
           
                j++;
                update_idx++;
                insert_count++;
            }

            if (update_idx <= maxindex && update_list[update_idx] == curr_key)
            {
                DEBUG_PI_BUCKET("Update Offset", tid, j, node_size);
                // ---> curr_offset = offset_list[update_idx];
                update_idx++;
            }

            local_keys[j] = curr_key;
            local_offsets[j] = curr_offset;

          //  printf("Done one round of For Loop TID %d: local_keys[%d] = %u, local_offsets[%d] = %u\n", tid, j, (unsigned int)local_keys[j], j, (unsigned int)local_offsets[j]);
            j++;
        }


     

        while ((insert_count + init_size < node_size) && update_idx <= maxindex && update_list[update_idx] <= curr_max && j < node_size)
        {
            //if (update_list[update_idx] == 4088163018)
          //  printf("ERROR in While Loop: TID %d, update_idx %u, maxindex %u, update_list[update_idx] = %u \n", tid, update_idx, maxindex, update_list[update_idx]);
            DEBUG_PI_BUCKET(" Lower While Loop Keep adding keys", tid, update_list[update_idx]);

            // if (tid ==47) DEBUG_PI_BUCKET("keep adding update key", tid, update_list[update_idx], offset_list[update_idx]);
            local_keys[j] = update_list[update_idx];
            local_offsets[j] = offset_list[update_idx];
            j++;
            update_idx++;
            insert_count++;
        }

      
         bool complete_inserts = (update_idx > maxindex);

        if ((insert_count == 0) && (init_size == node_size))
        {
            split_curr_node<key_type>(curr_node, launch_params, tid);
        }
        else if (insert_count > 0)
        {
            // if (tid ==47) DEBUG_PI_BUCKET("Going to Copy", tid, j, node_size);
            copy_arrays_to_node_localmem<key_type>(curr_node, local_keys, local_offsets, j, node_size, ts_count, launch_params, complete_inserts, tid);
        }
        // printf("TID: %d, After Copying Arrays to Node, complete_inserts %d, j %u AND INSERT COUNT %D\n", tid, complete_inserts,j, insert_count);
    }

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();

   /*  if (tid == 0)
    {
        printf("END INSERTIONS: PRINT NODE 555\n");
        print_node<key_type>(curr_node, node_size);
    } */

    if (tid == 0)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, tid);
    }

#endif
}

#endif