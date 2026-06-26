// =============================================================================
// File: coarse_granular_combined_updates.cuh
// Author: Rosina Kharal
// Description: Implements coarse_granular_combined_updates
//              Testing a combined kernel for inserts and deletes in FliX.
//              This is a prototype for testing and debugging, not optimized for performance.
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

// filepath: /home/rkharal/OPTIX/cgRX_LatestNov1124/rtx-index/index-prototype/src/coarse_granular_combined_updates.cuh
#ifndef COARSE_GRANULAR_COMBINED_UPDATES_CUH
#define COARSE_GRANULAR_COMBINED_UPDATES_CUH


#include <cstdint>
#include <cstdio>
#include "launch_parameters.cuh"
#include "definitions.cuh"
#include "definitions_updates.cuh"
#include "definitions_coarse_granular.cuh"
#include "coarse_granular_inserts.cuh"

//#define DEBUG_DELETION    //for debugging, print DELETION information and Shifting keys to Left '
//#define ERROR_CHECKS      //for debugging, check for errors, prints relevant error messages in INSERT of REMOVE

template <typename key_type>
GLOBALQUALIFIER void combined_update_kernel(updatable_cg_params *launch_params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = idx;
    smallsize lastkeyindex = (launch_params->partition_size * 2);
    smallsize num_partitions_with_overflow = launch_params->partition_count_with_overflow;
    smallsize nodesizebytes = launch_params->node_stride;
    smallsize update_size = launch_params->update_size;

    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);

    auto buf = launch_params->ordered_node_pairs;
    const key_type *maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
   
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const key_type *delete_list = static_cast<const key_type *>(launch_params->delete_list);

    if (idx >= 2 * num_partitions_with_overflow)
    {
        return;
    }

    bool is_insert = (idx < num_partitions_with_overflow);
    int partition_idx = idx % num_partitions_with_overflow;

    key_type maxkey = maxbuf[partition_idx];
    key_type minkey = 1;

    if (partition_idx > 0)
    {
        minkey = maxbuf[partition_idx - 1] + 1;
    }

    int minindex, maxindex;
    if (is_insert)
    {
        minindex = binarySearchIndex<key_type>(update_list, minkey, 0, update_size, false);
        maxindex = binarySearchIndex<key_type>(update_list, maxkey, 0, update_size, true);
    }
    else
    {
        minindex = binarySearchIndex<key_type>(delete_list, minkey, 0, update_size, false);
        maxindex = binarySearchIndex<key_type>(delete_list, maxkey, 0, update_size, true);
    }

    if (tid == 0)
    {
        DEBUG_UPDATES_DEVICE("UpdateKernel ", minindex, maxindex);
        DEBUG_UPDATES_DEVICE("UpdateKernel ", minkey, maxkey);
    }
    if (minindex > maxindex)
    {
        return;
    }
    if (minindex == -1 || maxindex == -1)
    {
        return;
    }

    auto curr_node = reinterpret_cast<uint8_t *>(buf) + (launch_params->node_stride) * idx;

    smallsize currnodesize = cg::extract<smallsize>(curr_node, sizeof(key_type));
    key_type currnodeMax = cg::extract<key_type>(curr_node, 0);


#ifdef PRINT_UPDATE_VALUES
    if (tid == 0)
    {
       DEBUG_UPDATES_DEVICE(" In Update Kernel Thread 0 ", maxkey, currnodesize, currnodeMax);
       DEBUG_UPDATES_DEVICE(" In Update Kernel Thread 0 ", minindex, maxindex);


        printf("Update list values: [");
        for (int i = minindex; i <= maxindex; ++i)
        {
            if (i > minindex)
            {
                printf(", ");
            }
            printf("%llu", static_cast<unsigned long long>(update_list[i]));
        }
        printf("]\n");
    }
#endif
    if (tid == 0)
    {
        DEBUG_UPDATES_DEVICE("In Update Kernel Calling Process Updates", tid, currnodesize, currnodeMax);
    }
    smallsize total_inserts = maxindex - minindex + 1;
     //if ((total_inserts >= 10) && !perform_dels)  DEBUG_PI_BUCKET_NUMINSERTS("Before Process Inserts: Num Inserts/Thread", tid, total_inserts, maxindex);
   // if (!perform_dels)  DEBUG_PI_BUCKET_NUMINSERTS("Before Process Inserts: Num Inserts/Thread", tid, total_inserts, maxindex);

   // Perform deletes first
    if (idx >= num_partitions_with_overflow)
    {
        process_deletes<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
    }

    // Synchronize all threads
    __syncthreads();

    // Perform inserts
    if (idx < num_partitions_with_overflow)
    {
        process_inserts<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
    }
      
}

#endif