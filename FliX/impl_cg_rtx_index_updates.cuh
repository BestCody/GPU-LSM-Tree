// =============================================================================
// File: impl_cg_rtx_index_updates.cuh
// Author: Rosina Kharal
// Description: Implements impl_cg_rtx_index_updates     
//              Previous implementations of FliX include a ray tracing index layer, or a static tree index layer
//              Support for all version of FliX exist in this file. 
//              
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INDEX_PROTOTYPE_IMPL_CG_RTX_INDEX_UPDATES_CUH
#define INDEX_PROTOTYPE_IMPL_CG_RTX_INDEX_UPDATES_CUH

#include "definitions.cuh"
#include "definitions_updates.cuh"
#include "definitions_coarse_granular.cuh"
#include "coarse_granular_inserts.cuh"
#include "coarse_granular_inserts_tiles.cuh"
// #include "coarse_granular_inserts_tiles_bulk.cuh"
#include "coarse_granular_inserts_warp.cuh"
#include "coarse_granular_inserts_additions.cuh"
#include "coarse_granular_deletes.cuh"
#include "coarse_granular_lookups.cuh"
// #include "coarse_granular_lookups.cuh"
#include "coarse_granular_lookups_tile.cuh"
#include "coarse_granular_lookups_tile_bulk.cuh"
// #include "coarse_granular_lookups_tile.cuh"

#include "coarse_granular_combined_updates.cuh"
#include "launch_parameters.cuh"

#include "utilities.cuh"
// ------#include "impl_opt_static_tree.cuh"
// #include "static_tree_index_layer.cuh"

namespace mem = memory_layout;

#ifdef REBUILD_GROUPS
#pragma message "Rebuild Groups include File=YES"
#include "coarse_granular_rebuild_groups.cuh"
#else
#pragma message "Rebuild Groups include File=NO"

#include "coarse_granular_rebuild.cuh"
#endif


#define OPTIMIZATION_ON // Optimization for maxvalues to minimize # of rays fired/search
extern "C" char coarse_granular_embedded_updates_ptx_code[];

extern optix_wrapper optix;

//--------------- STATIC TREE BUILD ---------------

template <typename key_type>
GLOBALQUALIFIER void count_keys_per_bucket_kernel(
    const void *node_buffer,
    const void *allocation_buffer,
    smallsize node_size,
    smallsize node_stride,
    smallsize allocation_buffer_count,
    smallsize partition_count_with_overflow,
    smallsize *keys_per_bucket)
{
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= partition_count_with_overflow)
        return;

    const uint8_t *curr_node = reinterpret_cast<const uint8_t *>(node_buffer) + tid * node_stride;

    size_t key_sum = 0;
    DEBUG_COMPUTE_TOTAL("  Tid per Bucket ", tid);

    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize next_offset = cg::extract<smallsize>(curr_node, lastposition_bytes);

    while (true)
    {
        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        key_sum += curr_size;
        DEBUG_COMPUTE_TOTAL("LOOP: ", tid, key_sum, curr_size);

        if (next_offset == 0)
            break;

        next_offset--;

        if (next_offset >= allocation_buffer_count)
        {
            printf("ERROR: NEXT OFFSET EXCEEDS ALLOCATION BUFFER COUNT\n");
            // if(tid==0) print_node(curr_node, node_size);
            break;
        }

        curr_node = reinterpret_cast<const uint8_t *>(allocation_buffer) + next_offset * node_stride;
        next_offset = cg::extract<smallsize>(curr_node, lastposition_bytes);
    }

    keys_per_bucket[tid] = static_cast<smallsize>(key_sum);
}

// Kernel to count total keys per bucket

// Kernel to count total keys per bucket
// Kernel to reduce key counts into one total
GLOBALQUALIFIER void reduce_total_keys_kernel(
    const smallsize *keys_per_bucket,
    size_t partition_count_with_overflow,
    size_t *total_keys)
{
    __shared__ size_t sdata[MAXBLOCKSIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    size_t my_sum = 0;
    if (i < partition_count_with_overflow)
        my_sum = keys_per_bucket[i];

    sdata[tid] = my_sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(total_keys, sdata[0]);
}
// Host function to compute total keys
template <typename key_type>
void compute_total_keys_in_structure(
    void *node_buffer,
    void *allocation_buffer,
    smallsize node_size,
    smallsize node_stride,
    smallsize allocation_buffer_count,
    smallsize partition_count_with_overflow,
    cudaStream_t stream,
    cuda_buffer<size_t> &d_total_keys)
{
    cuda_buffer<smallsize> d_keys_per_bucket;
    d_keys_per_bucket.alloc(partition_count_with_overflow);
    d_keys_per_bucket.zero();

    int threadsPerBlock = MAXBLOCKSIZE;
    int numBlocks = (partition_count_with_overflow + threadsPerBlock - 1) / threadsPerBlock;

    DEBUG_COMPUTE_TOTAL("Compute Total Keys Kernel: ", partition_count_with_overflow, node_size, node_stride);
    count_keys_per_bucket_kernel<key_type><<<numBlocks, threadsPerBlock, 0, stream>>>(
        node_buffer, allocation_buffer,
        node_size, node_stride,
        allocation_buffer_count, partition_count_with_overflow,
        d_keys_per_bucket.ptr());

    d_total_keys.zero();

    DEBUG_COMPUTE_TOTAL("Compute REDUCTION: ", partition_count_with_overflow, node_size, node_stride);

    reduce_total_keys_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_keys_per_bucket.ptr(),
        partition_count_with_overflow,
        d_total_keys.ptr());

    cudaStreamSynchronize(stream);
    CUERR;
}

// Host function to launch the kernel
template <typename key_type>
void launch_count_keys_per_bucket(
    const void *node_buffer,
    const void *allocation_buffer,
    smallsize node_size,
    smallsize node_stride,
    smallsize allocation_buffer_count,
    smallsize partition_count_with_overflow,
    smallsize *keys_per_bucket)
{
    cuda_buffer<smallsize> d_keys_per_bucket;
    d_keys_per_bucket.alloc(partition_count_with_overflow);
    d_keys_per_bucket.zero();

    int threadsPerBlock = MAXBLOCKSIZE;
    int numBlocks = (partition_count_with_overflow + threadsPerBlock - 1) / threadsPerBlock;

    count_keys_per_bucket_kernel<key_type><<<numBlocks, threadsPerBlock>>>(
        node_buffer, allocation_buffer,
        node_size, node_stride,
        allocation_buffer_count, partition_count_with_overflow,
        d_keys_per_bucket.ptr());

    cudaStreamSynchronize(0);
    CUERR;

    d_keys_per_bucket.download(keys_per_bucket, partition_count_with_overflow);
}

#ifdef OPTIMIZATION_ON

template <typename key_type>
GLOBALQUALIFIER void find_maxvalues_kernel(key_type *maxvalues, const uint8_t *buf, smallsize partition_count_with_overflow, smallsize node_stride)
{
    key_type successor = -1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    smallsize numpartitions = partition_count_with_overflow - 1;

    if (i < partition_count_with_overflow)
    {

        key_type curr_node_max = cg::extract<key_type>(buf, i * node_stride);

        if (i < (numpartitions - 1))
        {
            successor = cg::extract<key_type>(buf, (i + 1) * node_stride + compute_key_offset_bytes<key_type>());

            // check if y and z bits are all the same
            if (!(cg::is_same_row<key_type>(curr_node_max, successor)))
            {

                DEBUG_MAXVALUES("Not Same Row: Max Val Kernel tid", 3, i, curr_node_max, successor);

                key_type new_curr_node_max = (curr_node_max | cg::x_mask<key_type>);
                cg::set<key_type>((void *)buf, i * node_stride, new_curr_node_max);
                curr_node_max = new_curr_node_max;
            }
        }
        // set the max_value of the node
        maxvalues[i] = curr_node_max;
        DEBUG_MAXVALUES("Optimization: Max Val Kernel:", i, maxvalues[i]);
    }
}

#else

template <typename key_type>
GLOBALQUALIFIER void find_maxvalues_kernel(key_type *maxvalues, const uint8_t *buf, smallsize partition_count_withO, smallsize node_stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < partition_count_withO)
    {
        maxvalues[i] = cg::extract<key_type>(buf, i * node_stride);
        DEBUG_MAXVALUES("No Optimization: Max Val Kernel: tid and maxvalues ", 2, i, maxvalues[i]);
    }
}
#endif

template <typename key_type>
void find_maxvalues(
    cuda_buffer<key_type> &maxvalues_buffer,
    const cuda_buffer<uint8_t> &ordered_node_pairs_buffer,
    smallsize partition_count_with_overflow,
    smallsize node_stride)
{
    const auto buf = ordered_node_pairs_buffer.ptr();
    find_maxvalues_kernel<key_type><<<SDIV(partition_count_with_overflow, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(maxvalues_buffer.ptr(), buf, partition_count_with_overflow, node_stride);
    cudaStreamSynchronize(0);
    CUERR // Check for launch errors

        DEBUG_MAXVALUES("Test Maxval host", partition_count_with_overflow, node_stride);

#ifdef PRINT_MAXVALUES
    // Download maxvalues to host
    std::vector<key_type> maxvalues_host = maxvalues_buffer.download(partition_count_with_overflow);
    for (int i = 0; i < partition_count_with_overflow; ++i)
    {
        printf("Done Max Val Kernel: tid %d, maxvalues_buffer[%d] = %llu\n", i, i, static_cast<unsigned long long>(maxvalues_host[i]));
    }
#endif
}

template <typename key_type>
void find_minmax_function(
    const key_type *ordered_keys,
    size_t key_count,
    key_type &min_key,
    key_type &max_key)
{
    cudaMemcpy(&min_key, ordered_keys, sizeof(key_type), cudaMemcpyDeviceToHost);
    CUERR
    cudaMemcpy(&max_key, ordered_keys + (key_count - 1), sizeof(key_type), cudaMemcpyDeviceToHost);
    CUERR
}


//Single Threaded Insertions
template <typename key_type>
GLOBALQUALIFIER void update_kernel(const key_type *__restrict__ update_list, smallsize update_size, updatable_cg_params *launch_params, bool perform_dels)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = idx;
    smallsize lastkeyindex = (launch_params->partition_size * 2);
    smallsize num_partitions_with_overflow = launch_params->partition_count_with_overflow;
    smallsize nodesizebytes = launch_params->node_stride;
   // smallsize update_size = launch_params->update_size;

    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);

    auto buf = launch_params->ordered_node_pairs;
    const key_type *maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    key_type maxkey = maxbuf[idx]; // curr_node[0]; // Assuming the key to search is at position 2
    key_type minkey = 1;

    //const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);

    if (idx > num_partitions_with_overflow - 1)
    {
        return;
    }
    if (idx > 0)
    {
        minkey = maxbuf[idx - 1] + 1; // max of previous node plus 1
    }

    int minindex = binarySearchIndex<key_type>(update_list, minkey, 0, update_size, false);
    int maxindex = binarySearchIndex<key_type>(update_list, maxkey, 0, update_size, true);

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
    // if ((total_inserts >= 10) && !perform_dels)  DEBUG_PI_BUCKET_NUMINSERTS("Before Process Inserts: Num Inserts/Thread", tid, total_inserts, maxindex);
    // if (!perform_dels)  DEBUG_PI_BUCKET_NUMINSERTS("Before Process Inserts: Num Inserts/Thread", tid, total_inserts, maxindex);

    if (perform_dels)
    {
        // printf(" Going to process Deletes \n");
        //---#ifdef ST_Delete_Tombstones
#ifdef ST_DELETE_TOMBSTONES
#pragma message "Process Deletes Using Tombstones=YES"
        // printf(" Going to process TOMSTONES Deletes \n");
        process_deletes_tombstones<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
#else
        process_deletes<key_type>(update_list, update_size, maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
#endif
    }
    else
    {

/********************* */
#ifdef PROCESS_INSERTS_SFTRIGHT
#pragma message "Process Inserts SFTRIGHT=YES"
        // process_inserts<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
        // process_inserts_shift_right<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
        process_inserts<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
#elif defined(PROCESS_INSERTS_BULK)
#pragma message "Process Inserts Per BuckeLMt=YES"
        // process_inserts_per_bucket<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
        // process_inserts_per_bucket_tombstones<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);

        // process_inserts_per_bucket_tombstones_cudabuffers_updated<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
        process_inserts_per_bucket_tombstones_localmem<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
#elif defined(PROCESS_INSERTS_BULK_HYBRID)
#pragma message "Process Inserts Per Bucket Hybrid=YES"
        // process_inserts_per_bucket_tombstones_localmem_hybrid<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
        process_inserts_per_bucket_tombstones_localmem_hybrid<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
#endif
        /********************* */

#ifdef PROCESS_INSERTS_SINGLETHREAD
#pragma message "Process Inserts SINGLE THREADED"
        process_inserts_single_thread<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
        // process_inserts_per_bucket_tombstones_localmem_hybrid<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
        // process_inserts_per_bucket_tombstones_localmem<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);

#else
#pragma message "No Insert Single Threaded"

#endif

        return;
    }
    /*

    Description on the above algorithms for Insertions:
    --> we removed support for tombstones in final version of FliX

    The key ones we use are as follows:

    process_inserts: original based on shifting right
    process_inserts_per_bucket_tombstones_cudabuffers_updated:
        - using cuda buffers, updated to minimize the node splitting
        do not split node if no more update keys remain to be inserted
    process_inserts_per_bucket_tombstones_localmem:
        - using thread local memory for copy space instead of cuda buffer.
        also updated to minimize the node splitting- in case of full node only split if more insert keys remain
    process_inserts_per_bucket_tombstones_localmem_hybrid:
        -same as above using thread local memory.
        However, insertions are performed using shift right method when node_size is close to full (node_size-2)
        Bulk Insertions only used for more empty nodesls *.

    */
}

template <typename key_type>
GLOBALQUALIFIER void transform_into_node_layout_kernel(const key_type *keys, const smallsize *offsets, void *node_buffer, smallsize key_count, smallsize node_size, smallsize node_stride, smallsize initial_num_keys_each_node, smallsize partition_count_with_overflow)
{

    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= key_count)
        return;

    smallsize total_nodes = partition_count_with_overflow - 1; // total nodes not including the overflow node
    smallsize key_offset_bytes = compute_key_offset_bytes<key_type>();
    smallsize my_node = tid / initial_num_keys_each_node; // which node this tid belongs to
    smallsize my_place_in_node = (tid + 1) % initial_num_keys_each_node;

    smallsize byte_offset = (node_stride * my_node) + ((tid % initial_num_keys_each_node) + 1) * key_offset_bytes;

    cg::set<key_type>(node_buffer, byte_offset, keys[tid]);
    cg::set<smallsize>(node_buffer, byte_offset + sizeof(key_type), offsets[tid]);

    // extra work if the thread is the last member in the group
    // insert maxkey and size at the front of the node
    if (my_place_in_node == 0)
    {
        // insert maxkey
        cg::set<key_type>(node_buffer, my_node * node_stride, keys[tid]);
        // insert size
        cg::set<smallsize>(node_buffer, my_node * node_stride + sizeof(key_type), initial_num_keys_each_node);
        // clear the lastptr to 0
        cg::set<smallsize>(node_buffer, my_node * node_stride + (node_size + 1) * key_offset_bytes, 0);
    }

    if (my_node == total_nodes - 1)
    { // last node not including overflow node

        smallsize remainder = key_count % initial_num_keys_each_node;
        bool has_remainder = (key_count % initial_num_keys_each_node != 0);
        if (has_remainder)
            my_place_in_node = ((tid + 1) % remainder);

        if (my_place_in_node == 0)
        {

            // this code block repeated in case there is a remainder. last node max needs to be set
            if (has_remainder)
            { // update size correctly in last actual node (not overflow node)

                DEBUG_BUILD("Node Layout Kernel: Has Remainder true", 5, tid, remainder, my_node, my_place_in_node, keys[key_count - 1]);

                // set the size to be remainder and the max_key to be the last key in the dataset
                cg::set<smallsize>(node_buffer, (total_nodes - 1) * node_stride + sizeof(key_type), remainder);
                cg::set<key_type>(node_buffer, my_node * node_stride, keys[key_count - 1]);
                // clear the lastptr to 0
                cg::set<smallsize>(node_buffer, my_node * node_stride + (node_size + 1) * key_offset_bytes, 0);
            }

            key_type datamax; // datamax is the max value for the overflow node
            if (sizeof(key_type) == 8)
            {
                datamax = static_cast<key_type>(std::numeric_limits<key64>::max());
            }
            else if (sizeof(key_type) == 4)
            {
                datamax = static_cast<key_type>(std::numeric_limits<key32>::max());
            }

            // set max for OVERFLOW NODE ONLY (overflow node size is 0 at initial build time)
            cg::set<key_type>(node_buffer, (total_nodes)*node_stride, datamax);
        }
    }
}

template <typename key_type>
void transform_into_node_layout(
    const key_type *keys,
    const smallsize *offsets,
    void *node_buffer,
    size_t key_count,
    size_t node_size,
    size_t node_stride,
    double *time_ms,
    size_t initial_num_keys_each_node,
    size_t partition_count_with_overflow)
{
    scoped_cuda_timer timer(0, time_ms);
    transform_into_node_layout_kernel<<<SDIV(key_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(keys, offsets, node_buffer, key_count, node_size, node_stride, initial_num_keys_each_node, partition_count_with_overflow);
    cudaStreamSynchronize(0);
    CUERR
}

template <typename key_type>
OptixTraversableHandle rebuild_structures(
    const optix_wrapper &optix,
    size_t partition_size,
    smallsize node_size,
    smallsize partition_count,
    smallsize partition_count_with_overflow,
    cuda_buffer<uint8_t> &as_buffer,
    cuda_buffer<uint8_t> &ordered_node_pairs_buffer,
    cuda_buffer<uint8_t> &allocation_buffer,
    cuda_buffer<key_type> &maxvalues_buffer,
    cuda_buffer<uint8_t> &copy_buffer,
    // cuda_buffer<smallsize> &copy_offset_list,
    smallsize node_stride,
    double *rebuild_time_ms,
    size_t *build_bytes,
    smallsize total_nodes_used_from_AR,
    smallsize &new_bucket_count_with_overflow) // Pass by reference)
{
    // this is required to find the maximum space needed for the build

    // Allocate a buffer for the representative nodes
    cuda_buffer<uint8_t> representative_temp_buffer;
    // determine size of representative_temp_buffer
    size_t representative_temp_buffer_size_bytes = (partition_count_with_overflow + total_nodes_used_from_AR) * node_stride;
    representative_temp_buffer.alloc(representative_temp_buffer_size_bytes);
    CUERR
    representative_temp_buffer.zero();
    CUERR

    // to understand the cost in bytes of the rebuild step
    auto size_snapshot = [&]
    {
        // return  ordered_node_pairs_buffer.size_in_bytes() + allocation_buffer.size_in_bytes() + maxvalues_buffer.size_in_bytes() + ordered_key_buffer.size_in_bytes() + offset_buffer.size_in_bytes() + ordered_offset_buffer.size_in_bytes() + sort_temp_buffer.size_in_bytes();
        return representative_temp_buffer.size_in_bytes() + ordered_node_pairs_buffer.size_in_bytes() + allocation_buffer.size_in_bytes() + maxvalues_buffer.size_in_bytes();
    };

    DEBUG_REBUILD("Rebuild Structures Original Sizes: ", 5, partition_size, node_size, partition_count_with_overflow, node_stride);

#ifdef REBUILD_GROUPS
#pragma message "Rebuild Groups=YES"
    printf(" Rebuild Groups=YES \n");
    // Call rebuild data on the GPU kernel
    smallsize new_node_count = rebuild_gpu_structures_tile<key_type>(
        ordered_node_pairs_buffer.ptr(),
        representative_temp_buffer.ptr(),
        allocation_buffer.ptr(),
        maxvalues_buffer.ptr(),
        node_size,
        node_stride,
        rebuild_time_ms,
        total_nodes_used_from_AR,
        partition_count_with_overflow);
    // (smallsize)(partition_count_with_overflow));

#else
#pragma message "Rebuild Groups=NO"

    // Call rebuild data on the GPU kernel
    smallsize new_node_count = rebuild_gpu_structures<key_type>(
        ordered_node_pairs_buffer.ptr(),
        representative_temp_buffer.ptr(),
        allocation_buffer.ptr(),
        maxvalues_buffer.ptr(),
        node_size,
        node_stride,
        rebuild_time_ms,
        total_nodes_used_from_AR,
        partition_count_with_overflow);
    // (smallsize)(partition_count_with_overflow));

#endif
    size_t max_bytes_during_gpu_rebuild = size_snapshot();

    new_bucket_count_with_overflow = new_node_count;

    // Free space and Swap the pointers
    ordered_node_pairs_buffer.free();
    ordered_node_pairs_buffer.swap(representative_temp_buffer);
    allocation_buffer.zero();
    CUERR                                                               // clear Allocation Buffer after copying the  nodes
                                                                        // update partition counts
        partition_count_with_overflow = new_bucket_count_with_overflow; // total_nodes_used_from_AR + partition_count_with_overflow;
    partition_count = partition_count_with_overflow - 1;

    DEBUG_REBUILD(" New Node Count: ", 1, new_node_count);

    DEBUG_REBUILD("After Rebuild Structures Sizes: ", 5, partition_size, node_size, partition_count_with_overflow, node_stride);

    // set maxvalues
    maxvalues_buffer.free();
    maxvalues_buffer.alloc(partition_count_with_overflow);
    CUERR
    maxvalues_buffer.zero();
    CUERR
    find_maxvalues<key_type>(maxvalues_buffer, ordered_node_pairs_buffer, partition_count_with_overflow, node_stride);
    size_t numnodes = ordered_node_pairs_buffer.size_in_bytes() / node_stride;

    assert(partition_count_with_overflow == numnodes);

    DEBUG_REBUILD("Rebuilt on GPU: ", 5, partition_size, partition_count, partition_count_with_overflow, numnodes, node_stride);
    size_t max_bytes_during_rebuild;

    // Call rebuild AS structure
    // auto compacted_as = nullptr;
    // create the compacted AS based on the representatives (not including the overflow node)

    auto compacted_as = cg::build_compacted_as_from_representatives<key_type>(
        optix,
        ordered_node_pairs_buffer.ptr(),
        partition_count,
        0,                                   // byte offset first entry
        node_stride,                         // stride in bytes
        node_stride * (partition_count - 1), // last rep offset

#ifdef OPTIMIZATION_ON
        node_stride + compute_key_offset_bytes<key_type>(), // offset to first key (successor)
#else
        0, //// respresentative_successor_offset =0 disable optimization,
#endif
        as_buffer,
        rebuild_time_ms,
        &max_bytes_during_rebuild);

    max_bytes_during_rebuild += size_snapshot();

    DEBUG_REBUILD("----Complete REBUILD-----", 3, max_bytes_during_rebuild, ordered_node_pairs_buffer.size_in_bytes(), node_stride);

    if (build_bytes)
        *build_bytes = std::max(max_bytes_during_rebuild, max_bytes_during_gpu_rebuild);

    return compacted_as;
}

template <typename key_type>
void build_structures_bucket_layer(
    // const optix_wrapper &optix,
    const key_type *keys_device_pointer,
    size_t key_count,
    size_t partition_size,
    size_t node_size,
    size_t partition_count,
    size_t partition_count_with_overflow,
    // cuda_buffer<uint8_t> &as_buffer,
    cuda_buffer<uint8_t> &ordered_node_pairs_buffer,
    cuda_buffer<uint8_t> &allocation_buffer,
    cuda_buffer<key_type> &maxvalues_buffer,
    cuda_buffer<uint8_t> &copy_buffer,
    cuda_buffer<smallsize> &reuse_list_buffer,
    smallsize node_stride,
    double *build_time_ms,
    size_t *build_bytes,
    smallsize initial_num_keys_each_node,
    smallsize extra_allocated_nodes)
{
    cuda_buffer<key_type> ordered_key_buffer;
    cuda_buffer<smallsize> offset_buffer;
    cuda_buffer<smallsize> ordered_offset_buffer;
    cuda_buffer<uint8_t> sort_temp_buffer;

    // this is required to find the maximum space needed for the build
    auto size_snapshot = [&]
    {
        // return  ordered_node_pairs_buffer.size_in_bytes() + allocation_buffer.size_in_bytes() + maxvalues_buffer.size_in_bytes() + ordered_key_buffer.size_in_bytes() + offset_buffer.size_in_bytes() + ordered_offset_buffer.size_in_bytes() + sort_temp_buffer.size_in_bytes();

        return copy_buffer.size_in_bytes() + ordered_node_pairs_buffer.size_in_bytes() + allocation_buffer.size_in_bytes() + maxvalues_buffer.size_in_bytes() + ordered_key_buffer.size_in_bytes() + offset_buffer.size_in_bytes() + ordered_offset_buffer.size_in_bytes() + sort_temp_buffer.size_in_bytes();
    };

    offset_buffer.alloc(key_count);
    CUERR
    init_offsets(offset_buffer, key_count, build_time_ms);

    // sort keys and offsets
    size_t sort_temp_buffer_bytes = find_pair_sort_buffer_size<key_type, smallsize>(key_count);
    sort_temp_buffer.alloc(sort_temp_buffer_bytes);
    CUERR
    ordered_key_buffer.alloc(key_count);
    CUERR
    ordered_offset_buffer.alloc(key_count);
    CUERR
    timed_pair_sort(
        sort_temp_buffer.raw_ptr,
        sort_temp_buffer_bytes,
        keys_device_pointer,
        ordered_key_buffer.ptr(),
        offset_buffer.ptr(),
        ordered_offset_buffer.ptr(),
        key_count,
        build_time_ms);

    size_t max_bytes_during_sort = size_snapshot();
    sort_temp_buffer.free();
    offset_buffer.free();

#ifdef PRINT_BUILD_VALUES
    printf("Build Structures: Print All key-offset pairs in the keys list\n");
    std::vector<key_type> keys_host = ordered_key_buffer.download(key_count);
    std::vector<smallsize> offsets_host = ordered_offset_buffer.download(key_count); // Assuming offset_type is the correct type
    // for (int i = 0; i < key_count; ++i) {
    for (int i = 0; i < key_count; ++i)
    {
        printf("Key %d: %llu, Offset %d: %u\n", i, static_cast<unsigned long long>(keys_host[i]), i, (offsets_host[i]));
    }
#endif

    /*------------------------------- NODE CONFIG ------------------------------*/
    // [max,size][key0,offset0][key1,offset1],[key2,offset2],.....[last_ptr]
    /*--------------------------------------------------------------------------*/

    ordered_node_pairs_buffer.alloc(node_stride * partition_count_with_overflow);
    CUERR
    ordered_node_pairs_buffer.zero();
    CUERR
    allocation_buffer.alloc(node_stride * extra_allocated_nodes);
    CUERR
    allocation_buffer.zero();
    CUERR
    maxvalues_buffer.alloc(partition_count_with_overflow);
    CUERR
    maxvalues_buffer.zero();
    CUERR
    copy_buffer.alloc(node_stride * partition_count_with_overflow);
    CUERR
    // copy_offset_list.alloc(partition_count_with_overflow*node_size);
    // CUERR
    // copy_offset_list.zero();
    // CUERR
    copy_buffer.zero();

    reuse_list_buffer.alloc(partition_count_with_overflow);
    CUERR
    reuse_list_buffer.zero();
    CUERR

    transform_into_node_layout(
        ordered_key_buffer.ptr(),
        ordered_offset_buffer.ptr(),
        ordered_node_pairs_buffer.ptr(),
        key_count,
        node_size,
        node_stride,
        build_time_ms,
        initial_num_keys_each_node,
        partition_count_with_overflow);
    size_t max_bytes_during_transform = size_snapshot();
    ordered_offset_buffer.free();
    ordered_key_buffer.free();

    find_maxvalues<key_type>(maxvalues_buffer, ordered_node_pairs_buffer, partition_count_with_overflow, node_stride);
    size_t numnodes = ordered_node_pairs_buffer.size_in_bytes() / node_stride;

    DEBUG_BUILD("Going to Build AS: ", 5, key_count, partition_size, partition_count, numnodes, node_stride);
    size_t max_bytes_during_build;

    // create the compacted AS based on the representatives (not including the overflow node)

    /*

    auto compacted_as = cg::build_compacted_as_from_representatives<key_type>(
        optix,
        ordered_node_pairs_buffer.ptr(),
        partition_count,
        0,                                   // byte offset first entry
        node_stride,                         // stride in bytes
        node_stride * (partition_count - 1), // last rep offset

#ifdef OPTIMIZATION_ON
        node_stride + compute_key_offset_bytes<key_type>(), // offset to first key (successor)
#else
        0, //// respresentative_successor_offset =0 disable optimization,
#endif

        as_buffer,
        build_time_ms,
        &max_bytes_during_build);
    max_bytes_during_build += size_snapshot();

    DEBUG_BUILD("----Complete Build-----", 2, ordered_node_pairs_buffer.size_in_bytes(), node_stride);

    if (build_bytes)
        *build_bytes = std::max(max_bytes_during_sort, std::max(max_bytes_during_build, max_bytes_during_transform));

    return compacted_as;
    */
}

template <typename key_type>
// OptixTraversableHandle rebuild_structures_bucket_layer(
void rebuild_structures_bucket_layer(
    // const optix_wrapper &optix,
    size_t partition_size,
    smallsize node_size,
    smallsize partition_count,
    smallsize partition_count_with_overflow,
    cuda_buffer<uint8_t> &as_buffer,
    cuda_buffer<uint8_t> &ordered_node_pairs_buffer,
    cuda_buffer<uint8_t> &allocation_buffer,
    cuda_buffer<key_type> &maxvalues_buffer,
    cuda_buffer<uint8_t> &copy_buffer,
    cuda_buffer<smallsize> &reuse_list_buffer,
    // cuda_buffer<smallsize> &copy_offset_list,
    smallsize node_stride,
    double *rebuild_time_ms,
    size_t *build_bytes,
    smallsize total_nodes_used_from_AR,
    smallsize &new_bucket_count_with_overflow) // Pass by reference)
{
    // this is required to find the maximum space needed for the build

    // Allocate a buffer for the representative nodes
    cuda_buffer<uint8_t> representative_temp_buffer;
    // determine size of representative_temp_buffer
    size_t representative_temp_buffer_size_bytes = (partition_count_with_overflow + total_nodes_used_from_AR) * node_stride;
    representative_temp_buffer.alloc(representative_temp_buffer_size_bytes);
    CUERR
    representative_temp_buffer.zero();
    CUERR
  

   
    // to understand the cost in bytes of the rebuild step
    auto size_snapshot = [&]
    {
        // return  ordered_node_pairs_buffer.size_in_bytes() + allocation_buffer.size_in_bytes() + maxvalues_buffer.size_in_bytes() + ordered_key_buffer.size_in_bytes() + offset_buffer.size_in_bytes() + ordered_offset_buffer.size_in_bytes() + sort_temp_buffer.size_in_bytes();
        return representative_temp_buffer.size_in_bytes() + ordered_node_pairs_buffer.size_in_bytes() + allocation_buffer.size_in_bytes() + maxvalues_buffer.size_in_bytes();
    };

    DEBUG_REBUILD("Rebuild Structures Original Sizes: ", 5, partition_size, node_size, partition_count_with_overflow, node_stride);

#ifdef REBUILD_GROUPS
#pragma message "Rebuild Groups=YES"

    // Call rebuild data on the GPU kernel
    smallsize new_node_count = rebuild_gpu_structures_compact_one_tile<key_type>(
    //smallsize new_node_count = rebuild_gpu_structures_tile<key_type>(
        ordered_node_pairs_buffer.ptr(),
        representative_temp_buffer.ptr(),
        allocation_buffer.ptr(),
        maxvalues_buffer.ptr(),
        node_size,
        node_stride,
        rebuild_time_ms,
        total_nodes_used_from_AR,
        partition_count_with_overflow);
    // (smallsize)(partition_count_with_overflow));

#else
#pragma message "Rebuild Groups=NO"

    // Call rebuild data on the GPU kernel
    smallsize new_node_count = rebuild_gpu_structures<key_type>(
        ordered_node_pairs_buffer.ptr(),
        representative_temp_buffer.ptr(),
        allocation_buffer.ptr(),
        maxvalues_buffer.ptr(),
        node_size,
        node_stride,
        rebuild_time_ms,
        total_nodes_used_from_AR,
        partition_count_with_overflow);
    // (smallsize)(partition_count_with_overflow));

#endif
    size_t max_bytes_during_gpu_rebuild = size_snapshot();

    new_bucket_count_with_overflow = new_node_count;

    // Free space and Swap the pointers
    ordered_node_pairs_buffer.free();
    ordered_node_pairs_buffer.swap(representative_temp_buffer);
    allocation_buffer.zero();
    CUERR                                                               // clear Allocation Buffer after copying the  nodes
                                                                        // update partition counts
        partition_count_with_overflow = new_bucket_count_with_overflow; // total_nodes_used_from_AR + partition_count_with_overflow;
    partition_count = partition_count_with_overflow - 1;

    reuse_list_buffer.alloc(partition_count_with_overflow);
    CUERR
    reuse_list_buffer.zero();
    CUERR

    DEBUG_REBUILD(" New Node Count: ", 1, new_node_count);

    DEBUG_REBUILD("After Rebuild Structures Sizes: ", 5, partition_size, node_size, partition_count_with_overflow, node_stride);

    // set maxvalues
    maxvalues_buffer.free();
    maxvalues_buffer.alloc(partition_count_with_overflow);
    CUERR
    maxvalues_buffer.zero();
    CUERR
    find_maxvalues<key_type>(maxvalues_buffer, ordered_node_pairs_buffer, partition_count_with_overflow, node_stride);
    size_t numnodes = ordered_node_pairs_buffer.size_in_bytes() / node_stride;

    assert(partition_count_with_overflow == numnodes);

    DEBUG_REBUILD("Rebuilt on GPU: ", 5, partition_size, partition_count, partition_count_with_overflow, numnodes, node_stride);
    size_t max_bytes_during_rebuild;


}


template <typename key_type>
OptixTraversableHandle build_structures(
    const optix_wrapper &optix,
    const key_type *keys_device_pointer,
    size_t key_count,
    size_t partition_size,
    size_t node_size,
    size_t partition_count,
    size_t partition_count_with_overflow,
    cuda_buffer<uint8_t> &as_buffer,
    cuda_buffer<uint8_t> &ordered_node_pairs_buffer,
    cuda_buffer<uint8_t> &allocation_buffer,
    cuda_buffer<key_type> &maxvalues_buffer,
    cuda_buffer<uint8_t> &copy_buffer,
    // cuda_buffer<smallsize> &copy_offset_list,
    smallsize node_stride,
    double *build_time_ms,
    size_t *build_bytes,
    smallsize initial_num_keys_each_node,
    smallsize extra_allocated_nodes)
{
    cuda_buffer<key_type> ordered_key_buffer;
    cuda_buffer<smallsize> offset_buffer;
    cuda_buffer<smallsize> ordered_offset_buffer;
    cuda_buffer<uint8_t> sort_temp_buffer;

    // this is required to find the maximum space needed for the build
    auto size_snapshot = [&]
    {
        // return  ordered_node_pairs_buffer.size_in_bytes() + allocation_buffer.size_in_bytes() + maxvalues_buffer.size_in_bytes() + ordered_key_buffer.size_in_bytes() + offset_buffer.size_in_bytes() + ordered_offset_buffer.size_in_bytes() + sort_temp_buffer.size_in_bytes();

        return copy_buffer.size_in_bytes() + ordered_node_pairs_buffer.size_in_bytes() + allocation_buffer.size_in_bytes() + maxvalues_buffer.size_in_bytes() + ordered_key_buffer.size_in_bytes() + offset_buffer.size_in_bytes() + ordered_offset_buffer.size_in_bytes() + sort_temp_buffer.size_in_bytes();
    };

    offset_buffer.alloc(key_count);
    CUERR
    init_offsets(offset_buffer, key_count, build_time_ms);

    // sort keys and offsets
    size_t sort_temp_buffer_bytes = find_pair_sort_buffer_size<key_type, smallsize>(key_count);
    sort_temp_buffer.alloc(sort_temp_buffer_bytes);
    CUERR
    ordered_key_buffer.alloc(key_count);
    CUERR
    ordered_offset_buffer.alloc(key_count);
    CUERR
    timed_pair_sort(
        sort_temp_buffer.raw_ptr,
        sort_temp_buffer_bytes,
        keys_device_pointer,
        ordered_key_buffer.ptr(),
        offset_buffer.ptr(),
        ordered_offset_buffer.ptr(),
        key_count,
        build_time_ms);

    size_t max_bytes_during_sort = size_snapshot();
    sort_temp_buffer.free();
    offset_buffer.free();

#ifdef PRINT_BUILD_VALUES
    printf("Build Structures: Print All key-offset pairs in the keys list\n");
    std::vector<key_type> keys_host = ordered_key_buffer.download(key_count);
    std::vector<smallsize> offsets_host = ordered_offset_buffer.download(key_count); // Assuming offset_type is the correct type
    // for (int i = 0; i < key_count; ++i) {
    for (int i = 0; i < key_count; ++i)
    {
        printf("Key %d: %llu, Offset %d: %u\n", i, static_cast<unsigned long long>(keys_host[i]), i, (offsets_host[i]));
    }
#endif

    /*------------------------------- NODE CONFIG ------------------------------*/
    // [max,size][key0,offset0][key1,offset1],[key2,offset2],.....[last_ptr]
    /*--------------------------------------------------------------------------*/

    ordered_node_pairs_buffer.alloc(node_stride * partition_count_with_overflow);
    CUERR
    ordered_node_pairs_buffer.zero();
    CUERR
    allocation_buffer.alloc(node_stride * extra_allocated_nodes);
    CUERR
    allocation_buffer.zero();
    CUERR
    maxvalues_buffer.alloc(partition_count_with_overflow);
    CUERR
    maxvalues_buffer.zero();
    CUERR
    copy_buffer.alloc(node_stride * partition_count_with_overflow);
    CUERR
    // copy_offset_list.alloc(partition_count_with_overflow*node_size);
    // CUERR
    // copy_offset_list.zero();
    // CUERR
    copy_buffer.zero();
    CUERR

    transform_into_node_layout(
        ordered_key_buffer.ptr(),
        ordered_offset_buffer.ptr(),
        ordered_node_pairs_buffer.ptr(),
        key_count,
        node_size,
        node_stride,
        build_time_ms,
        initial_num_keys_each_node,
        partition_count_with_overflow);
    size_t max_bytes_during_transform = size_snapshot();
    ordered_offset_buffer.free();
    ordered_key_buffer.free();

    find_maxvalues<key_type>(maxvalues_buffer, ordered_node_pairs_buffer, partition_count_with_overflow, node_stride);
    size_t numnodes = ordered_node_pairs_buffer.size_in_bytes() / node_stride;

    DEBUG_BUILD("Going to Build AS: ", 5, key_count, partition_size, partition_count, numnodes, node_stride);
    size_t max_bytes_during_build;

    // create the compacted AS based on the representatives (not including the overflow node)

    auto compacted_as = cg::build_compacted_as_from_representatives<key_type>(
        optix,
        ordered_node_pairs_buffer.ptr(),
        partition_count,
        0,                                   // byte offset first entry
        node_stride,                         // stride in bytes
        node_stride * (partition_count - 1), // last rep offset

#ifdef OPTIMIZATION_ON
        node_stride + compute_key_offset_bytes<key_type>(), // offset to first key (successor)
#else
        0, //// respresentative_successor_offset =0 disable optimization,
#endif

        as_buffer,
        build_time_ms,
        &max_bytes_during_build);
    max_bytes_during_build += size_snapshot();

    DEBUG_BUILD("----Complete Build-----", 2, ordered_node_pairs_buffer.size_in_bytes(), node_stride);

    if (build_bytes)
        *build_bytes = std::max(max_bytes_during_sort, std::max(max_bytes_during_build, max_bytes_during_transform));

    return compacted_as;
}

template <typename key_type>
GLOBALQUALIFIER void setup_combine_update_data(
    updatable_cg_params *launch_params,
    bool long_keys,
    bool aggregate_results,
    const key_type *update_list,
    const smallsize *offset_list,
    smallsize size,
    const key_type *delete_list,
    smallsize *result)
{
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1)
        return;

    launch_params->long_keys = long_keys;
    launch_params->aggregate_results = aggregate_results;
    launch_params->query_lower = nullptr;
    launch_params->query_upper = nullptr;
    launch_params->update_list = update_list;
    launch_params->offset_list = offset_list;
    launch_params->update_size = size;
    launch_params->delete_list = delete_list;
    launch_params->result = nullptr;
}

template <typename key_type>
GLOBALQUALIFIER void setup_update_data(
    updatable_cg_params *launch_params,
    bool long_keys,
    bool aggregate_results,
    const key_type *update_list,
    const smallsize *offset_list,
    smallsize size,
    smallsize *result)
{
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1)
        return;

    launch_params->long_keys = long_keys;
    launch_params->aggregate_results = aggregate_results;
    launch_params->query_lower = nullptr;
    launch_params->query_upper = nullptr;
    launch_params->update_list = update_list;
    launch_params->offset_list = offset_list;
    launch_params->update_size = size;
    launch_params->result = nullptr;
}

template <typename key_type>
GLOBALQUALIFIER void setup_lookup_data_static_tree(
    updatable_cg_params *launch_params,
    bool long_keys,
    bool aggregate_results,
    const key_type *query_lower,
    const key_type *query_upper,
    smallsize *result,
    smallsize query_size)
{
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1)
        return;

    launch_params->long_keys = long_keys;
    launch_params->aggregate_results = aggregate_results;
    launch_params->query_lower = query_lower;
    launch_params->query_upper = query_upper;
    launch_params->result = result;
    launch_params->query_size = query_size;
    // launch_params->bucket_indices = bucket_indices;
}

template <typename key_type>
GLOBALQUALIFIER void setup_lookup_data(
    updatable_cg_params *launch_params,
    bool long_keys,
    bool aggregate_results,
    const key_type *query_lower,
    const key_type *query_upper,
    smallsize *result,
    smallsize query_size)
{
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1)
        return;

    launch_params->long_keys = long_keys;
    launch_params->aggregate_results = aggregate_results;
    launch_params->query_lower = query_lower;
    launch_params->query_upper = query_upper;
    launch_params->result = result;
    launch_params->query_size = query_size;
}

template <typename key_type>
GLOBALQUALIFIER void setup_static_tree_build_data(
    updatable_cg_params *launch_params,
    const key_type *tree_buffer,
    //    tree_metadata metadata,
    smallsize tree_entries_count,
    void *ordered_node_pairs,
    void *allocation_buffer,
    const key_type *maxvalue_buffer,
    smallsize node_stride,
    smallsize stored_size,
    smallsize partition_size,
    smallsize node_size,
    void *copy_buffer,
    // smallsize* copy_offset_list,
    smallsize partition_count,
    smallsize partition_count_with_overflow,
    smallsize extra_allocated_nodes)
{
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1)
        return;

    // launch_params->tree_buffer = tree_buffer;
    // launch_params->metadata = metadata;
    // launch_params->tree_entries_count = tree_entries_count;
    launch_params->ordered_node_pairs = ordered_node_pairs;
    launch_params->allocation_buffer = allocation_buffer;
    launch_params->maxvalues = maxvalue_buffer;
    launch_params->node_stride = node_stride;
    launch_params->stored_size = stored_size;
    launch_params->partition_size = partition_size;
    launch_params->node_size = node_size;
    launch_params->copy_buffer = copy_buffer;
    // launch_params->copy_offset_list = copy_offset_list;
    launch_params->partition_count = partition_count;
    launch_params->partition_count_with_overflow = partition_count_with_overflow;
    launch_params->free_node = 0;
    launch_params->allocation_buffer_count = extra_allocated_nodes;
}

template <typename key_type>
GLOBALQUALIFIER void setup_bucket_only_build_data(
    updatable_cg_params *launch_params,
    // const key_type* tree_buffer,
    // tree_metadata metadata,
    // smallsize tree_entries_count,
    void *ordered_node_pairs,
    void *allocation_buffer,
    const key_type *maxvalue_buffer,
    smallsize node_stride,
    smallsize stored_size,
    smallsize partition_size,
    smallsize node_size,
    // void *copy_buffer,
    smallsize *reuse_list_buffer,
    smallsize partition_count,
    smallsize partition_count_with_overflow,
    smallsize extra_allocated_nodes)
{
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1)
        return;

    // launch_params->tree_buffer = tree_buffer;
    // launch_params->metadata = metadata;
    // launch_params->tree_entries_count = tree_entries_count;
    launch_params->ordered_node_pairs = ordered_node_pairs;
    launch_params->allocation_buffer = allocation_buffer;
    launch_params->maxvalues = maxvalue_buffer;
    launch_params->node_stride = node_stride;
    launch_params->stored_size = stored_size;
    launch_params->partition_size = partition_size;
    launch_params->node_size = node_size;
    // launch_params->copy_buffer = copy_buffer;
    // launch_params->copy_offset_list = copy_offset_list;
    launch_params->partition_count = partition_count;
    launch_params->partition_count_with_overflow = partition_count_with_overflow;
    launch_params->free_node = 0;
    launch_params->allocation_buffer_count = extra_allocated_nodes;
    launch_params->reuse_list_count = 0;
    launch_params->reuse_list = reuse_list_buffer;
}

template <typename key_type>
GLOBALQUALIFIER void setup_build_data(
    updatable_cg_params *launch_params,
    OptixTraversableHandle traversable,
    void *ordered_node_pairs,
    void *allocation_buffer,
    const key_type *maxvalue_buffer,
    smallsize node_stride,
    smallsize stored_size,
    smallsize partition_size,
    smallsize node_size,
    void *copy_buffer,
    // smallsize* copy_offset_list,
    smallsize partition_count,
    smallsize partition_count_with_overflow,
    smallsize extra_allocated_nodes)
{
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1)
        return;

    launch_params->traversable = traversable;
    launch_params->ordered_node_pairs = ordered_node_pairs;
    launch_params->allocation_buffer = allocation_buffer;
    launch_params->maxvalues = maxvalue_buffer;
    launch_params->node_stride = node_stride;
    launch_params->stored_size = stored_size;
    launch_params->partition_size = partition_size;
    launch_params->node_size = node_size;
    launch_params->copy_buffer = copy_buffer;
    // launch_params->copy_offset_list = copy_offset_list;
    launch_params->partition_count = partition_count;
    launch_params->partition_count_with_overflow = partition_count_with_overflow;
    launch_params->free_node = 0;
    launch_params->allocation_buffer_count = extra_allocated_nodes;
}
template <typename key_type>
GLOBALQUALIFIER void setup_rebuild_data(
    updatable_cg_params *launch_params,
    OptixTraversableHandle traversable,
    void *ordered_node_pairs,
    void *allocation_buffer,
    const key_type *maxvalue_buffer,
    smallsize node_stride,
    smallsize partition_size,
    smallsize node_size,
    void *copy_buffer,
    // smallsize* copy_offset_list,
    smallsize partition_count,
    smallsize partition_count_with_overflow,
    smallsize extra_allocated_nodes)
{
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1)
        return;

    launch_params->traversable = traversable;
    launch_params->ordered_node_pairs = ordered_node_pairs;
    launch_params->allocation_buffer = allocation_buffer;
    launch_params->maxvalues = maxvalue_buffer;
    launch_params->node_stride = node_stride;
    // launch_params->stored_size = stored_size;
    launch_params->partition_size = partition_size;
    launch_params->node_size = node_size;
    launch_params->copy_buffer = copy_buffer;
    // launch_params->copy_offset_list = copy_offset_list;
    launch_params->partition_count = partition_count;
    launch_params->partition_count_with_overflow = partition_count_with_overflow;
    launch_params->free_node = 0;
    launch_params->allocation_buffer_count = extra_allocated_nodes;
}

//******************* LOOKUP KERNELS */

//-------------------LOOKUP KERNEL
template <typename key_type>
GLOBALQUALIFIER void lookup_kernel(updatable_cg_params *launch_params, bool perform_group_lookup)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = idx;
    smallsize lastkeyindex = (launch_params->partition_size * 2);
    smallsize num_partitions_with_overflow = launch_params->partition_count_with_overflow;
    smallsize nodesizebytes = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    // const smallsize* offset_list = static_cast<const smallsize*>(launch_params->offset_list);

    auto buf = launch_params->ordered_node_pairs;
    const key_type *maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    //  key_type maxkey = maxbuf[idx];       // curr_node[0]; // Assuming the key to search is at position 2
    key_type maxbufsize = num_partitions_with_overflow;

    const key_type *query_list = static_cast<const key_type *>(launch_params->query_lower);
    smallsize query_size = launch_params->query_size;

#ifdef LOOKUP_KERNEL_DEBUG
    if (tid == 0)
    {
        printf("Tid: %d, PRINTING MAXBUF VALUES\n", tid);
        for (smallsize i = 0; i < num_partitions_with_overflow; i++)
        {
            printf("maxbuf[%d] = %u\n", i, maxbuf[i]);
        }
    }
#endif

    if (idx > query_size - 1)
    {
        // printf("Error: idx > query_size, tid: %d\n", tid);
        return;
    }

    if (perform_group_lookup)
    {
        idx = idx * node_size;
        if (idx > query_size - 1)
            return;
        // printf("Group lookup tid: %d and New Group idx %d \n", tid, idx);
    }
    key_type probe_key = query_list[idx];
    smallsize found_index = 0;
    bool probe_key_found = false;

    probe_key_found = binary_search_equal_or_greater<key_type>(maxbuf, maxbufsize, probe_key, found_index, tid);

    // Replace with new SEARCH ---
    // probe_key_found = binary_search_in_array<key_type>(maxbuf, maxbufsize, probe_key, found_index, tid);

#ifdef LOOKUP_KERNEL_DEBUG

    printf("Thread %d myprobekey %u was probkeyfound %d, and found_index %d \n ", idx, probe_key, probe_key_found, found_index);
    // print_maxbuf(maxbuf, maxbufsize);
#endif
    smallsize bucket_index = found_index; // is the maxval of the bucket where this key belongs... so it is the i'th bucket

    if (bucket_index >= maxbufsize)
    {
        printf("Error: bucket index out of bounds, tid: %d\n", tid);
        return;
    }

    auto curr_node = reinterpret_cast<uint8_t *>(buf) + (launch_params->node_stride) * bucket_index;

    smallsize currnodesize = cg::extract<smallsize>(curr_node, sizeof(key_type));
    key_type currnodeMax = cg::extract<key_type>(curr_node, 0);

#ifdef LOOKUP_KERNEL_DEBUG
    // if (tid == 0){
    // printf("Going to Process Lookups:Thread %d, myprobekey %u, currnodesize %u and currmax %u \n ", idx, probe_key  ,currnodesize,currnodeMax);
    // PRINT_KEY(idx, currnodeMax, key_type);
    // printf(" going to process a LOOKUP in CUDA Kernal\n");
    // }
#endif

    if (perform_group_lookup)
        printf("Performing Group Lookup in CUDA Kernel\n");
    // process_group_lookups<key_type>(currnodeMax, bucket_index, probe_key, launch_params, curr_node, currnodesize, idx);
    else
        // process the Lookup One Key at a time
        process_lookups<key_type>(currnodeMax, bucket_index, probe_key, launch_params, curr_node, currnodesize, idx);
}

//******************* LOOKUP KERNELS */

template <typename key_type_, uint8_t cache_line = 0, uint8_t node_size_log = 0, uint8_t percentage_initial_fill = 50> // uint32_t additional_inserts_percentage = 0>
class cg_rtx_index_updates
{
    static_assert(sizeof(key_type_) == 4 || sizeof(key_type_) == 8, "key_type must be 4 or 8 bytes");
    static_assert(cache_line == 0 || cache_line == 5 || cache_line == 10 || cache_line == 20, "cache_line must be encoded as 0 (0.0), 5 (0.5), 10 (1.0), or 20 (2.0)");
    static_assert(percentage_initial_fill <= 100, "percentage_initial_fill cannot exceed 100");
    static_assert(node_size_log >= 0 && node_size_log <= 8, "node_size_log must be between 0 and 8 (inclusive)");
    // static_assert(additional_inserts_percentage <= 500, "additional_inserts_percentage cannot exceed 200");
    static_assert((cache_line == 0) + (node_size_log == 0) == 1, "Exactly one of cache_line or node_size_log must be 0");
    static_assert(((cache_line == 10) && (sizeof(key_type_) == 4)) || cache_line != 10, "Cache_line == 10 supported only for 32-bit keys in this update");
    // static_assert(TILE_SIZE >= node_size, "Tile size must be at least large enough to handle nodes.");

public:
    using key_type = key_type_;

private:
    cuda_buffer<uint8_t> as_buffer;
    cuda_buffer<uint8_t> ordered_node_pairs_buffer;
    cuda_buffer<uint8_t> allocation_buffer;
    cuda_buffer<uint8_t> copy_buffer;
    cuda_buffer<key_type> maxvalues_buffer;
    cuda_buffer<smallsize> reuse_list_buffer;

    cuda_buffer<smallsize> bucket_values_buffer;

    // add a buffer for static tree
    //  Meta data

    cuda_buffer<updatable_cg_params> launch_params_buffer;
    std::optional<optix_pipeline> pipeline;

    key_type *maxvalues;

    //--------------static tree private data

    cuda_buffer<key_type> sorted_keys_buffer;
    cuda_buffer<smallsize> sorted_offsets_buffer;
    cuda_buffer<uint8_t> sorted_key_offset_pairs_buffer;
    // size_t base_entries_count = 0;

    // buffer stores metadata and all tree levels in sequence
    static constexpr uint16_t rq_threads_per_block = 512;
    static constexpr uint16_t rq_registers_per_thread = 1;
    cuda_buffer<key_type> tree_buffer;

    size_t shared_entries_count = 0;
    size_t base_entries_count = 0;

    size_t shared_bytes_to_load = 0;
    size_t shared_bytes_for_shuffle = 0;

    uint32_t number_of_sms = 0;
    size_t max_shared_memory_bytes = 0;


public:
    static constexpr const char *name = "cg_rtx_index_updates";
    static constexpr operation_support can_lookup = operation_support::async;
    static constexpr operation_support can_multi_lookup = operation_support::none;
    static constexpr operation_support can_range_lookup = operation_support::none;
    static constexpr operation_support can_update = operation_support::async;

    static constexpr smallsize key_offset_pair = compute_key_offset_bytes<key_type>();
    static constexpr smallsize node_size = compute_node_size<key_type>(cache_line, node_size_log);
    static constexpr smallsize node_stride = compute_nodestride_bytes<key_type>(node_size, cache_line); // size of the node in bytes

    static constexpr smallsize partition_size = SDIV(node_size, 2);
    static constexpr smallsize initial_fill = percentage_initial_fill;

    static constexpr uint16_t node_size_log_static_tree = (node_size_log > 0 ? node_size_log : 4);
    static constexpr uint16_t cg_size_log_static_tree = node_size_log_static_tree;

    // static constexpr smallsize additional_inserts = additional_inserts_percentage;
    // static constexpr uint16_t node_size_log_static_tree = 4;
    // static constexpr uint16_t cg_size_log_static_tree = node_size_log_static_tree;

    // smallsize partition_count = 0;
    // smallsize partition_count_with_overflow = 0;
    smallsize partition_count = 0;
    smallsize partition_count_with_overflow = 0;

    smallsize total_nodes_used_from_AR = 0;
    size_t total_allocation_region_nodes = 0;
    static constexpr smallsize initial_num_keys_each_node = (initial_fill * node_size + 99) / 100;

    static std::string short_description()
    {
        return std::string("cg_rtx_index_updates_") + std::to_string(node_size_log);
    }

    // TO DO
    static parameters_type parameters()
    {
        return {
            {"node_size", std::to_string(node_size)},
            {"tile size log", std::to_string(TILE_SIZE)},
            {"cache line", std::to_string(cache_line)},
            {"partition size", std::to_string(partition_size)},
            {"initial fill", std::to_string(initial_fill)},
            {"div factor", std::to_string(DIV_FACTOR)},
            {"num rounds", std::to_string(rounds)},
            {"percentage_distribution_dense_keys", std::to_string(percentage_distribution_dense_keys)},
            {"percentage_new_keys_from_dense_region", std::to_string(percentage_new_keys_from_dense_region)},

            //{"num_replicas", std::to_string(num_replicas)}
        };
    }
    static size_t additional_nodes_required(size_t maxsize, size_t initial_nodes_with_overflow)
    {

        DEBUG_EXTRA_NODES("Additional Nodes Calculation", 2, size, initial_nodes_with_overflow);

        size_t total_keys = maxsize;                           // size * (100 + additional_inserts) / 100;
        size_t total_nodes = SDIV(total_keys, partition_size); // Assuming half full nodes
        size_t total_nodes_without_initial = total_nodes > initial_nodes_with_overflow ? total_nodes - initial_nodes_with_overflow : 0;

        // Increase by 1.75x using integer arithmetic
        // HERE ---> The increased number of nodes is No Longer Needed
        size_t total_nodes_without_initial_increased = total_nodes_without_initial * 175 / 100;

        return total_nodes_without_initial;
        // return total_nodes_without_initial*2;
    }

    static size_t estimate_build_bytes(size_t size, size_t maxsize)
    {

        size_t num_partitions_plus_overflow = SDIV(size, initial_num_keys_each_node) + 1;
        size_t num_triangles_required_in_scene = num_partitions_plus_overflow - 1;
        // size_t permutation_bytes = (sizeof(key_type) + 4) * size;
        size_t permutation_bytes = node_stride * num_partitions_plus_overflow;
        size_t sort_aux_bytes = sizeof(smallsize) * size + find_pair_sort_buffer_size<key_type, smallsize>(size);

        // size_t triangle_bytes = 3 * 9 * SDIV(size, partition_size) * sizeof(float);
        size_t triangle_bytes = 3 * 9 * num_triangles_required_in_scene * sizeof(float);

        size_t bvh_bytes = triangle_bytes * 13 / 10;
        size_t aux_bytes = triangle_bytes * 2 / 10;

        size_t total_allocation_nodes = additional_nodes_required(maxsize, num_partitions_plus_overflow);

        size_t allocation_region_bytes = node_stride * total_allocation_nodes;
        size_t max_value_buffer_bytes = num_partitions_plus_overflow * sizeof(key_type);
        // ADDED copy_buffer_bytes
        size_t copy_buffer_bytes = node_stride * num_partitions_plus_overflow;

        size_t additional_gpu_buffers_bytes = allocation_region_bytes + max_value_buffer_bytes + copy_buffer_bytes;
        // ? Did not add copy_buffer here for

        size_t phase_1_bytes = permutation_bytes + sort_aux_bytes + additional_gpu_buffers_bytes;
        size_t phase_2_bytes = permutation_bytes + triangle_bytes + bvh_bytes + aux_bytes + additional_gpu_buffers_bytes;
        size_t phase_3_bytes = permutation_bytes + 2 * bvh_bytes + additional_gpu_buffers_bytes;
        return std::max(phase_1_bytes, std::max(phase_2_bytes, phase_3_bytes));
    }

    /*
    // UPDATE with copy arrays and update build structures
    size_t gpu_resident_bytes()
    {

        updatable_cg_params params = launch_params_buffer.download_first_item();
        total_nodes_used_from_AR = params.free_node;
        //return as_buffer.size_in_bytes() + ordered_node_pairs_buffer.size_in_bytes() + (total_nodes_used_from_AR)*node_stride + maxvalues_buffer.size_in_bytes() + launch_params_buffer.size_in_bytes();
        size_t sum_all = ordered_node_pairs_buffer.size_in_bytes() + (total_nodes_used_from_AR)*node_stride + maxvalues_buffer.size_in_bytes() + launch_params_buffer.size_in_bytes();

        printf("GPU RESIDENT BYTES CALCULATION: \n");
        printf("AS BUFFER SIZE: %zu \n", as_buffer.size_in_bytes());
        printf("ORDERED NODE PAIRS BUFFER SIZE: %zu \n", ordered_node_pairs_buffer.size_in_bytes());
        printf("TOTAL NODES USED FROM AR: %zu \n", total_nodes_used_from_AR);
        printf("NODE STRIDE: %zu \n", node_stride);
        printf("MAXVALUES BUFFER SIZE: %zu \n", maxvalues_buffer.size_in_bytes());
        printf("LAUNCH PARAMS BUFFER SIZE: %zu \n", launch_params_buffer.size_in_bytes());
        printf("TOTAL GPU RESIDENT BYTES: %zu \n", sum_all);
        return sum_all;

    } */

    inline size_t gpu_resident_bytes()
    {
        // why: snapshot launch params once for consistency with the debug output
        const updatable_cg_params params = launch_params_buffer.download_first_item();
        total_nodes_used_from_AR = params.free_node;

        smallsize total_reuse_list_nodes = params.reuse_list_count;

        if (total_nodes_used_from_AR > total_reuse_list_nodes)
        {

            total_nodes_used_from_AR = total_nodes_used_from_AR - total_reuse_list_nodes;
        }
        else
        {

            total_nodes_used_from_AR = 0;
        }
        DEBUG_GPU_RESIDENT_BYTES("Reuse List Nodes, Total Used from AR", 2, total_reuse_list_nodes, total_nodes_used_from_AR);

        // total_nodes_used_from_AR = total_nodes_used_from_AR - total_reuse_list_nodes;
        // total_nodes_used_from_AR = total_nodes_used_from_AR - total_reuse_list_nodes;

        // Component sizes
        const size_t as_sz = as_buffer.size_in_bytes();
        const size_t pairs_sz = ordered_node_pairs_buffer.size_in_bytes();
        const size_t maxvals_sz = maxvalues_buffer.size_in_bytes();
        const size_t launch_params_sz = launch_params_buffer.size_in_bytes();
        const size_t nodes_bytes = static_cast<size_t>(total_nodes_used_from_AR) * node_stride;

        // Totals (preserve original: exclude AS buffer from returned total)
        const size_t sum_all = pairs_sz + nodes_bytes + maxvals_sz + launch_params_sz;
        const size_t sum_with_as = sum_all + as_sz;

        //  snapshot
      /* 
        DEBUG_GPU_RESIDENT_BYTES("GPU Resident Byte Components", 7,
                                 as_sz,                                         // AS BUFFER SIZE
                                 pairs_sz,                                      // ORDERED NODE PAIRS BUFFER SIZE
                                 static_cast<size_t>(total_nodes_used_from_AR), // TOTAL NODES USED FROM AR
                                 node_stride,                                   // NODE STRIDE
                                 maxvals_sz,                                    // MAXVALUES BUFFER SIZE
                                 total_reuse_list_nodes,                        // TOTAL REUSE LIST NODES
                                 // ---------total_reuse_list_nodes, // TOTAL REUSE LIST NODES
                                 launch_params_sz // LAUNCH PARAMS BUFFER SIZE
        );
        */

        //  (excl/with AS buffer)
        DEBUG_GPU_RESIDENT_BYTES("GPU Resident Byte Totals (excl AS, incl AS)", 2,
                                 sum_all,    // TOTAL GPU RESIDENT BYTES (original behavior)
                                 sum_with_as // TOTAL incl. AS buffer (for visibility)
        );

        return sum_all;
    }

    size_t compute_total_size(cudaStream_t stream)
    {
        void *allocation_region = allocation_buffer.ptr();
        void *ordered_nodes = ordered_node_pairs_buffer.ptr();

        smallsize nodesize = node_size;
        smallsize nodestride = node_stride;
        smallsize partitioncount = partition_count;
        smallsize partitioncount_with_overflow = partition_count_with_overflow;
        smallsize total_allocation_region_nodes = total_nodes_used_from_AR;

        cuda_buffer<size_t> d_total_keys;
        d_total_keys.alloc(1);

        compute_total_keys_in_structure<key_type>(
            ordered_nodes,                 // node_buffer
            allocation_region,             // allocation_buffer
            nodesize,                      // node_size
            nodestride,                    // node_stride
            total_allocation_region_nodes, // allocation_buffer_count
            partitioncount_with_overflow,  // partition_count_with_overflow
            stream,                        // cuda stream
            d_total_keys);                 // output buffer

        size_t total_keys_in_DS = d_total_keys.download_first_item();

        return static_cast<smallsize>(total_keys_in_DS);
    }

    bool is_allocation_buffer_exceeded()
    {
        return total_nodes_used_from_AR > total_allocation_region_nodes;
    }
    smallsize allocation_buffer_total_nodes()
    {
        return total_allocation_region_nodes;
    }
    smallsize allocation_buffer_next_free()
    {
        updatable_cg_params params = launch_params_buffer.download_first_item();
        DEBUG_UPDATES("Next AR FREE_NODE: ", 2, params.free_node, total_allocation_region_nodes);
        total_nodes_used_from_AR = params.free_node;
        return total_nodes_used_from_AR; // total_allocation_region_nodes;
    }

    void data_structure_stats()
    {
        // updatable_cg_params params = launch_params_buffer.download_first_item();

        DATA_STRUCTURE_STATS("Stats Build params", 5, cache_line, node_size, partition_size, node_stride, percentage_initial_fill);
        return; // total_nodes_used_from_AR > total_allocation_region_nodes;
    }

    // void build(const key_type *keys, size_t keysize, double *build_time_ms, size_t *build_bytes)
    void build_bucket_layer_only(const key_type *keys, size_t size, size_t max_size, size_t available_memory_bytes, double *build_time_ms, size_t *build_bytes)
    {
        // DEBUG_BUILD_PARAMS("START BUILD: Fill Size", 1, keysize);
        smallsize tile_size = TILE_SIZE;
        smallsize warp_size = WARP_SIZE;

        launch_params_buffer.alloc(1);
        C2EX

            if (build_bytes)
                *build_bytes += launch_params_buffer.size_in_bytes();

        //DEBUG_BUILD_PARAMS("Output Initial Build Paramaters", 1, short_description().c_str());
        //DEBUG_BUILD_PARAMS("Output Initial Build Paramaters", 0);
        DEBUG_BUILD_PARAMS("Output Build params:", 3, initial_fill, size, key_offset_pair);

        DEBUG_BUILD_PARAMS("More Build params", 5, cache_line, node_size, partition_size, node_stride, percentage_initial_fill);

        assert(initial_num_keys_each_node <= node_size); // Ensure not exceeding max capacity
        partition_count = SDIV(size, initial_num_keys_each_node);
        partition_count_with_overflow = partition_count + 1;

        DEBUG_BUILD("BUILD Partition Count", 3, partition_count, partition_count_with_overflow, initial_num_keys_each_node);
        DEBUG_BUILD("BUILD node params", 4, cache_line, node_size_log, percentage_initial_fill, additional_inserts);

        total_allocation_region_nodes = additional_nodes_required(max_size, partition_count_with_overflow);

        DEBUG_BUILD("BUILD Allocated ", 1, total_allocation_region_nodes);
        DEBUG_BUILD_PARAMS("More Build params", 3, partition_count, partition_count_with_overflow, total_allocation_region_nodes);
        DEBUG_BUILD_PARAMS("More Build params", 2, tile_size, warp_size);

        // Inserts line
        DEBUG_BUILD_PARAMS(
            "Compile flags INSERTS: BULK_ONLY BULK_HYBRID TILE_INSERTS TILE_INSERTS_C",
            4,
            compile_flags::inserts_tile_bulk_only,
            compile_flags::inserts_tile_bulk_hybrid,
            compile_flags::tile_inserts,
            compile_flags::tile_inserts_c);

        // Deletes line
        DEBUG_BUILD_PARAMS(
            "Compile flags DELETES: TILE_DELETES DELETES_TILE_BULK",
            2,
            compile_flags::tile_deletes,
            compile_flags::deletes_tile_bulk);

        DEBUG_BUILD_PARAMS(
            "Compile flags OTHER: BASELINES HASHTABLE_WARPCORE HASHTABLE_SLAB GPU_BTREE LSM_TREE",
            5,
            compile_flags::baselines,
            compile_flags::hashtable_warpcore,
            compile_flags::hashtable_slab,
            compile_flags::gpu_btree,
            compile_flags::lsm_tree);

        build_structures_bucket_layer(
            // *pipeline->optix,
            keys,
            size,
            partition_size,
            node_size,
            partition_count,
            partition_count_with_overflow,
            // as_buffer,
            ordered_node_pairs_buffer,
            allocation_buffer,
            maxvalues_buffer,
            copy_buffer,
            reuse_list_buffer,
            node_stride,
            build_time_ms,
            build_bytes, initial_num_keys_each_node, total_allocation_region_nodes);

       /*. Prior Version
        const size_t tree_n = static_cast<size_t>(partition_count_with_overflow);
        smallsize tree_entries_count = tree_n; // not used 
        base_entries_count = tree_n;  // not used 
        const size_t reuse_capacity = total_allocation_region_nodes; // must match allocation_buffer_count
        smallsize *d_reuse_list = nullptr;
        cudaError_t st = cudaSuccess;
        st = cudaMalloc(reinterpret_cast<void **>(&d_reuse_list),
                        reuse_capacity * sizeof(smallsize));
        assert(st == cudaSuccess && "cudaMalloc(reuse_list) failed");
        // Zero the entire reuse_list (byte 0 is fine for integer zero)
        st = cudaMemset(d_reuse_list, 0, reuse_capacity * sizeof(smallsize));
        assert(st == cudaSuccess && "cudaMemset(reuse_list) failed");
        */


        setup_bucket_only_build_data<key_type><<<1, 1, 0, 0>>>(
            launch_params_buffer.ptr(),
            ordered_node_pairs_buffer.ptr(),
            allocation_buffer.ptr(),
            maxvalues_buffer.ptr(),
            node_stride,
            size,
            partition_size,
            node_size,
            reuse_list_buffer.ptr(),
            partition_count,
            partition_count_with_overflow,
            total_allocation_region_nodes);

        cudaDeviceSynchronize();
        C2EX

 
    }

    void rebuild_bucket_layer_only(double *build_time_ms, size_t *build_bytes)
    {
        updatable_cg_params params = launch_params_buffer.download_first_item();
        DEBUG_REBUILD("Rebuild AR FREE_NODE: ", 2, params.free_node, total_allocation_region_nodes);
        total_nodes_used_from_AR = params.free_node;
        smallsize size = params.stored_size;

        tree_buffer.free();

        smallsize new_bucket_count_with_overflow = 0; // Initialize it before passing

        rebuild_structures_bucket_layer(
            //*pipeline->optix,
            partition_size,
            node_size,
            partition_count,
            partition_count_with_overflow,
            as_buffer,
            ordered_node_pairs_buffer,
            allocation_buffer,
            maxvalues_buffer,
            copy_buffer,
            reuse_list_buffer,
            node_stride,
            build_time_ms,
            build_bytes,
            total_nodes_used_from_AR,
            new_bucket_count_with_overflow // Pass by reference
        );
        // partition_count_with_overflow = total_nodes_used_from_AR + partition_count_with_overflow;
        partition_count_with_overflow = new_bucket_count_with_overflow;
        partition_count = partition_count_with_overflow - 1;
        DEBUG_REBUILD("After Rebuild Partition Count", 3, partition_count, partition_count_with_overflow, initial_num_keys_each_node);
        
        /*. prior version of static tree
        const size_t tree_n = static_cast<size_t>(partition_count_with_overflow);
        // printf("REBuilding Static Tree with %u entries\n", tree_n);
        smallsize tree_entries_count = tree_n; // not used
        base_entries_count = tree_n; 
        */

        setup_bucket_only_build_data<key_type><<<1, 1, 0, 0>>>(
            launch_params_buffer.ptr(),
           
            ordered_node_pairs_buffer.ptr(),
            allocation_buffer.ptr(),
            maxvalues_buffer.ptr(),
            node_stride,
            size,
            partition_size,
            node_size,
            reuse_list_buffer.ptr(),
            // copy_offset_list,
            partition_count,
            partition_count_with_overflow,
            total_allocation_region_nodes);

        cudaDeviceSynchronize();
        C2EX

    }

    // void build(const key_type *keys, size_t keysize, double *build_time_ms, size_t *build_bytes)
    void build_static_tree(const key_type *keys, size_t size, size_t max_size, size_t available_memory_bytes, double *build_time_ms, size_t *build_bytes)
    {
        // DEBUG_BUILD_PARAMS("START BUILD: Fill Size", 1, keysize);
        smallsize tile_size = TILE_SIZE;
        smallsize warp_size = WARP_SIZE;

        launch_params_buffer.alloc(1);
        C2EX

            if (build_bytes)
                *build_bytes += launch_params_buffer.size_in_bytes();

        DEBUG_BUILD_PARAMS("Initial Build Fill Size", 1, size);
        DEBUG_BUILD_PARAMS("Build params", 3, initial_fill, size, key_offset_pair);

        DEBUG_BUILD_PARAMS("More Build params", 5, cache_line, node_size, partition_size, node_stride, percentage_initial_fill);

        assert(initial_num_keys_each_node <= node_size); // Ensure not exceeding max capacity
        partition_count = SDIV(size, initial_num_keys_each_node);
        partition_count_with_overflow = partition_count + 1;

        DEBUG_BUILD("BUILD Partition Count", 3, partition_count, partition_count_with_overflow, initial_num_keys_each_node);
        DEBUG_BUILD("BUILD node params", 4, cache_line, node_size_log, percentage_initial_fill, additional_inserts);

        total_allocation_region_nodes = additional_nodes_required(max_size, partition_count_with_overflow);

        DEBUG_BUILD("BUILD Allocated ", 1, total_allocation_region_nodes);
        DEBUG_BUILD_PARAMS("More Build params", 3, partition_count, partition_count_with_overflow, total_allocation_region_nodes);
        DEBUG_BUILD_PARAMS("More Build params", 2, tile_size, warp_size);

        build_structures_bucket_layer(
            // *pipeline->optix,
            keys,
            size,
            partition_size,
            node_size,
            partition_count,
            partition_count_with_overflow,
            // as_buffer,
            ordered_node_pairs_buffer,
            allocation_buffer,
            maxvalues_buffer,
            copy_buffer,
            // copy_offset_list,
            node_stride,
            build_time_ms,
            build_bytes, initial_num_keys_each_node, total_allocation_region_nodes);

        const size_t tree_n = static_cast<size_t>(partition_count_with_overflow);
        // const size_t leaf_n = static_cast<size_t>(partition_count);
        //---------------------------------------------------------------------

        // Build tree using the per-partition maxima as keys.
        // printf("Building Static Tree with %zu entries\n and size %zu\n", tree_n , size);

        // build tree
        // build_static_tree<key_type, node_size_log>(
        //         (const uint8_t*) maxvalues_buffer.ptr(), tree_n, 0, sizeof(key_type),
        //     metadata, tree_buffer, build_time_ms);
        smallsize tree_entries_count = tree_n; // CHECK THE LEAF NODES FOR MAX VALUES

        base_entries_count = tree_n; // if you still rely on this in wrapper
                                     // ----build_static_tree_from_pivots<key_type, node_size_log, false>(
                                     // ---    (const uint8_t*)maxvalues_buffer.ptr(), tree_n, 0, sizeof(key_type), sizeof(key_type) * (tree_entries_count - 1),
                                     // ---     metadata, tree_buffer, build_time_ms);
        C2EX

            setup_static_tree_build_data<key_type><<<1, 1, 0, 0>>>(
                launch_params_buffer.ptr(),
                tree_buffer.ptr(),
                // metadata,
                tree_entries_count,
                ordered_node_pairs_buffer.ptr(),
                allocation_buffer.ptr(),
                maxvalues_buffer.ptr(),
                node_stride,
                size,
                partition_size,
                node_size,
                copy_buffer,
                // copy_offset_list,
                partition_count,
                partition_count_with_overflow,
                total_allocation_region_nodes);

        cudaDeviceSynchronize();
        C2EX

        // CHECK THE LEAF NODES FOR MAX VALUES
        // static_tree.dump_tree_csv("debug/static_tree");      // writes two CSVs

        // Free the allocated buffers after use
        // cudaFree(copy_update_list);
        // cudaFree(copy_offset_list);
    }

    void rebuild_static_tree(double *build_time_ms, size_t *build_bytes)
    // void rebuild_statictree(const key_type* keys, size_t size, size_t max_size, size_t available_memory_bytes, double* build_time_ms, size_t* build_bytes)
    {
        updatable_cg_params params = launch_params_buffer.download_first_item();
        DEBUG_REBUILD("Rebuild AR FREE_NODE: ", 2, params.free_node, total_allocation_region_nodes);
        total_nodes_used_from_AR = params.free_node;
        smallsize size = params.stored_size;

        tree_buffer.free();

        smallsize new_bucket_count_with_overflow = 0; // Initialize it before passing

        rebuild_structures_bucket_layer(
            //*pipeline->optix,
            partition_size,
            node_size,
            partition_count,
            partition_count_with_overflow,
            as_buffer,
            ordered_node_pairs_buffer,
            allocation_buffer,
            maxvalues_buffer,
            copy_buffer,
            // copy_offset_list,
            node_stride,
            build_time_ms,
            build_bytes,
            total_nodes_used_from_AR,
            new_bucket_count_with_overflow // Pass by reference
        );
        // partition_count_with_overflow = total_nodes_used_from_AR + partition_count_with_overflow;
        partition_count_with_overflow = new_bucket_count_with_overflow;
        partition_count = partition_count_with_overflow - 1;
        DEBUG_REBUILD("After Rebuild Partition Count", 3, partition_count, partition_count_with_overflow, initial_num_keys_each_node);
        const size_t tree_n = static_cast<size_t>(partition_count_with_overflow);

        // printf("REBuilding Static Tree with %u entries\n", tree_n);
        smallsize tree_entries_count = tree_n; // CHECK THE LEAF NODES FOR MAX VALUES

        base_entries_count = tree_n; // if you still rely on this in wrapper
                                     //   build_static_tree_from_pivots<key_type, node_size_log, false>(
                                     //       (const uint8_t*)maxvalues_buffer.ptr(), tree_n, 0, sizeof(key_type), sizeof(key_type) * (tree_entries_count - 1),
                                     //       metadata, tree_buffer, build_time_ms);
        C2EX

            setup_static_tree_build_data<key_type><<<1, 1, 0, 0>>>(
                launch_params_buffer.ptr(),
                tree_buffer.ptr(),
                //    metadata,
                tree_entries_count,
                ordered_node_pairs_buffer.ptr(),
                allocation_buffer.ptr(),
                maxvalues_buffer.ptr(),
                node_stride,
                size,
                partition_size,
                node_size,
                copy_buffer,
                // copy_offset_list,
                partition_count,
                partition_count_with_overflow,
                total_allocation_region_nodes);

        cudaDeviceSynchronize();
        C2EX
    }

    // void build(const key_type *keys, size_t keysize, double *build_time_ms, size_t *build_bytes)
    void build(const key_type *keys, size_t size, size_t max_size, size_t available_memory_bytes, double *build_time_ms, size_t *build_bytes)
    {
        // DEBUG_BUILD_PARAMS("START BUILD: Fill Size", 1, keysize);
        smallsize tile_size = TILE_SIZE;
        smallsize warp_size = WARP_SIZE;

        if (!pipeline)
        {
            pipeline.emplace(
                &optix,
                coarse_granular_embedded_updates_ptx_code,
                "params",
                "__raygen__lookup_nodes");
        }

        launch_params_buffer.alloc(1);
        CUERR

        if (build_bytes)
            *build_bytes += launch_params_buffer.size_in_bytes();

        DEBUG_BUILD_PARAMS("Initial Build Fill Size", 1, size);
        DEBUG_BUILD_PARAMS("Build params", 3, initial_fill, size, key_offset_pair);

        DEBUG_BUILD_PARAMS("More Build params", 5, cache_line, node_size, partition_size, node_stride, percentage_initial_fill);

        assert(initial_num_keys_each_node <= node_size); // Ensure not exceeding max capacity
        partition_count = SDIV(size, initial_num_keys_each_node);
        partition_count_with_overflow = partition_count + 1;

        DEBUG_BUILD("BUILD Partition Count", 3, partition_count, partition_count_with_overflow, initial_num_keys_each_node);
        DEBUG_BUILD("BUILD node params", 4, cache_line, node_size_log, percentage_initial_fill, additional_inserts);

        total_allocation_region_nodes = additional_nodes_required(max_size, partition_count_with_overflow);

        DEBUG_BUILD("BUILD Allocated ", 1, total_allocation_region_nodes);
        DEBUG_BUILD_PARAMS("More Build params", 3, partition_count, partition_count_with_overflow, total_allocation_region_nodes);
        DEBUG_BUILD_PARAMS("More Build params", 2, tile_size, warp_size);

        /*    // ------------------ BULK_INSERTS
        // Allocate and zero out the copy_update_list and copy_offset_list buffers
        key_type *copy_update_list;
        smallsize *copy_offset_list;
        size_t buffer_size = partition_count_with_overflow * sizeof(key_type);

        cudaMalloc(&copy_update_list, buffer_size);
        cudaMemset(copy_update_list, 0, buffer_size);

        buffer_size = partition_count_with_overflow * sizeof(smallsize);
        cudaMalloc(&copy_offset_list, buffer_size);
        cudaMemset(copy_offset_list, 0, buffer_size);
        //------------------ BULK_INSERTS
        */
        OptixTraversableHandle traversable = build_structures(
            *pipeline->optix,
            keys,
            size,
            partition_size,
            node_size,
            partition_count,
            partition_count_with_overflow,
            as_buffer,
            ordered_node_pairs_buffer,
            allocation_buffer,
            maxvalues_buffer,
            copy_buffer,
            // copy_offset_list,
            node_stride,
            build_time_ms,
            build_bytes, initial_num_keys_each_node, total_allocation_region_nodes);

        setup_build_data<key_type><<<1, 1, 0, 0>>>(
            launch_params_buffer.ptr(),
            traversable,
            ordered_node_pairs_buffer.ptr(),
            allocation_buffer.ptr(),
            maxvalues_buffer.ptr(),
            node_stride,
            size,
            partition_size,
            node_size,
            copy_buffer,
            // copy_offset_list,
            partition_count,
            partition_count_with_overflow,
            total_allocation_region_nodes);

        cudaDeviceSynchronize();
        CUERR
        // Free the allocated buffers after use
        // cudaFree(copy_update_list);
        // cudaFree(copy_offset_list);
    }

    void rebuild(double *build_time_ms, size_t *build_bytes)
    {

        updatable_cg_params params = launch_params_buffer.download_first_item();
        DEBUG_REBUILD("Rebuild AR FREE_NODE: ", 2, params.free_node, total_allocation_region_nodes);
        total_nodes_used_from_AR = params.free_node;

        as_buffer.free();

        smallsize new_bucket_count_with_overflow = 0; // Initialize it before passing

        OptixTraversableHandle traversable = rebuild_structures(
            *pipeline->optix,
            partition_size,
            node_size,
            partition_count,
            partition_count_with_overflow,
            as_buffer,
            ordered_node_pairs_buffer,
            allocation_buffer,
            maxvalues_buffer,
            copy_buffer,
            // copy_offset_list,
            node_stride,
            build_time_ms,
            build_bytes,
            total_nodes_used_from_AR,
            new_bucket_count_with_overflow // Pass by reference
        );

        // partition_count_with_overflow = total_nodes_used_from_AR + partition_count_with_overflow;
        partition_count_with_overflow = new_bucket_count_with_overflow;
        partition_count = partition_count_with_overflow - 1;

        setup_rebuild_data<key_type><<<1, 1, 0, 0>>>(
            launch_params_buffer.ptr(),
            traversable,
            ordered_node_pairs_buffer.ptr(),
            allocation_buffer.ptr(),
            maxvalues_buffer.ptr(),
            node_stride,
            partition_size,
            node_size,
            copy_buffer,
            // copy_offset_list,
            partition_count,
            partition_count_with_overflow,
            total_allocation_region_nodes);

        cudaDeviceSynchronize();
        CUERR
    }

    void insert(const key_type *update_list, const smallsize *offsets, size_t size, cudaStream_t stream)
    {

#ifdef PRINT_INSERT_VALUES
        printf("\n ");
        printf("****************************************************************\n");
        printf("Insertions Latest File, Size of Insert List: %d \n", size);
        printf("****************************************************************\n");
        printf("\n ");
        printf("Insert Function Print all new key-offset pairs in the update list\n");
        assert(ordered_node_pairs_buffer.size_in_bytes() % node_stride == 0);

        std::vector<key_type> keys_host(size);
        std::vector<smallsize> offsets_host(size);

        cudaMemcpy(keys_host.data(), update_list, size * sizeof(key_type), cudaMemcpyDeviceToHost);
        CUERR
        cudaMemcpy(offsets_host.data(), offsets, size * sizeof(smallsize), cudaMemcpyDeviceToHost);
        CUERR

        for (int i = 0; i < size; ++i)
        {
            key_type key = keys_host[i];
            printf("Key %d: %llu, Offset: %u\n", i, static_cast<unsigned long long>(key), static_cast<unsigned>(offsets_host[i]));
        }
#endif

#ifdef TILE_INSERTS
#pragma message "TILE INSERTS_UPDATED=YES"

#ifdef TILE_INSERTS_C
#pragma message "TILE INSERTS_C_UPDATED=YES"
        // NUMTHREADS = MAXBLOCKSIZE;
        //  total_nodes = 1000000;

        smallsize threadsPerBlock = TILE_SIZE;                               // e.g., 8 or 16
        smallsize blocksPerGrid = partition_count_with_overflow * TILE_SIZE; // NOT divided by anything
        // smallsize maxblocksplus1 = SDIV(blocksPerGrid, MAXBLOCKSIZE /DIV_FACTOR) + 1;

        smallsize maxblocks_required = SDIV(blocksPerGrid, MAXBLOCKSIZE /DIV_FACTOR);


        update_kernel_tile_inserts_new<key_type><<<maxblocks_required, MAXBLOCKSIZE /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), update_list, offsets, size);
        //update_kernel_tile_inserts<key_type><<<maxblocks_required, MAXBLOCKSIZE /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), update_list, offsets, size);

#endif

#else
#pragma message "TILE INSERTS=NO"

        // smallsize total_nodes = partition_count_with_overflow; // taking into account overflow node
        smallsize total_nodes = partition_count_with_overflow; // taking into account overflow node
        smallsize NUMTHREADS = (total_nodes > MAXBLOCKSIZE) ? MAXBLOCKSIZE : total_nodes;
        update_kernel<key_type><<<SDIV(total_nodes, NUMTHREADS /DIV_FACTOR), NUMTHREADS /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), false);
#endif
    }

   
    void remove(const key_type *update_list, size_t size, cudaStream_t stream)
    {

        smallsize *offsets = nullptr;
        smallsize *result = nullptr;
        // key_type *copy_update_list = nullptr;
        // smallsize *copy_offset_list = nullptr;

#ifdef PRINT_REMOVE_VALUES
        printf("\n ");
        printf("****************************************************************\n");
        printf("REMOVE, Size of Delete List:\n", size);
        printf("****************************************************************\n");
        printf("\n ");

        assert(ordered_node_pairs_buffer.size_in_bytes() % node_stride == 0);
        std::vector<key_type> keys_host(size);

        cudaMemcpy(keys_host.data(), update_list, size * sizeof(key_type), cudaMemcpyDeviceToHost);
        CUERR
        for (int i = 0; i < size; ++i)
        {
            printf("Key %d: %llu \n", i, static_cast<unsigned long long>(keys_host[i]));
        }
        printf("Remove: partition_count %d, partition_count_with_overflow %d\n", partition_count, partition_count_with_overflow);
#endif

        /*  {
             nvtx3::scoped_range_in<nvtx_rtx_domain> upload{"upload-params"};
             setup_update_data<<<1, 1, 0, stream>>>(
                 launch_params_buffer.ptr(),
                 sizeof(key_type) == 8,
                 false,
                 update_list,
                 offsets,
                 size,
                 result);
         } */

        {
            smallsize total_nodes = partition_count_with_overflow; // taking into account overflow node
            smallsize NUMTHREADS = (total_nodes > MAXBLOCKSIZE) ? MAXBLOCKSIZE : total_nodes;
            DEBUG_UPDATES("Remove: Calling Update Kernel", 2, partition_count_with_overflow, size);
#ifdef TILE_DELETES
#pragma message "PERFORM TILE DELETES=YES"
            // printf("TILE DELETES=YES\n");
            //  #ifdef TILE_DELETES_A#pragma message "TILE INSERTS_A_=YES"
            NUMTHREADS = MAXBLOCKSIZE;

            smallsize threadsPerBlock = TILE_SIZE;                               // e.g., 8 or 16
            smallsize blocksPerGrid = partition_count_with_overflow * TILE_SIZE; // NOT divided by anything
            smallsize maxblocksplus1 = SDIV(blocksPerGrid, MAXBLOCKSIZE /DIV_FACTOR);

            update_kernel_tile_deletes_new<key_type><<<maxblocksplus1, MAXBLOCKSIZE /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), update_list, size);

            // update_kernel_tile<key_type><<<maxblocksplus1, MAXBLOCKSIZE /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), true);

            // update_kernel_tile<key_type><<<maxblocksplus1, MAXBLOCKSIZE / 8, 0, stream>>>(launch_params_buffer.ptr(), true);

#else
#pragma message "TILE DELETES=NO"
            //printf("SINGLE THREADED Shift Left  TILE DELETES=NO\n");
            update_kernel<key_type><<<SDIV(total_nodes, NUMTHREADS /DIV_FACTOR), NUMTHREADS /DIV_FACTOR, 0, stream>>>(update_list, size,launch_params_buffer.ptr(), true);
#endif
        }
    }

    //--------------------- COMBINE INSERT AND DEL ---------------------
    void insert_and_remove(const key_type *update_list, const smallsize *offsets, size_t size, const key_type *delete_list, cudaStream_t stream)
    {

        smallsize *result = nullptr;

#ifdef PRINT_COMBINE_UPDATE_VALUES
        printf("\n ");
        printf("****************************************************************\n");
        printf("COMBINE: FIRST DELETIONS, then INSERTIONS Size of Lists is equal: %d \n", size);
        printf("****************************************************************\n");
        printf("\n ");
        printf("COMBINE Function Print all new key-offset pairs in the insert list\n");
        assert(ordered_node_pairs_buffer.size_in_bytes() % node_stride == 0);

        std::vector<key_type> keys_host(size);
        std::vector<smallsize> offsets_host(size);

        cudaMemcpy(keys_host.data(), update_list, size * sizeof(key_type), cudaMemcpyDeviceToHost);
        CUERR
        cudaMemcpy(offsets_host.data(), offsets, size * sizeof(smallsize), cudaMemcpyDeviceToHost);
        CUERR

        for (int i = 0; i < size; ++i)
        {
            printf("Insert Key %d: %llu, Offset: %u\n", i, static_cast<unsigned long long>(keys_host[i]), static_cast<unsigned>(offsets_host[i]));
        }

        printf("****************************************************************\n");
        printf("\n ");

        printf("COMBINE Function Print all values in the Delete list\n");

        // clear keys_host and offsets_host
        keys_host.clear();

        cudaMemcpy(keys_host.data(), delete_list, size * sizeof(key_type), cudaMemcpyDeviceToHost);
        CUERR

        for (int i = 0; i < size; ++i)
        {
            printf("Delete Key %d: %llu \n", i, static_cast<unsigned long long>(keys_host[i]));
        }

#endif
        DEBUG_COMBINED("Combine Function ", 2, partition_count, partition_count_with_overflow);

        {
            nvtx3::scoped_range_in<nvtx_rtx_domain> upload{"upload-params"};
            setup_combine_update_data<<<1, 1, 0, stream>>>(
                launch_params_buffer.ptr(),
                sizeof(key_type) == 8,
                false,
                update_list,
                offsets,
                size,
                delete_list,
                result);
        }

        {

            smallsize total_nodes = partition_count_with_overflow; // taking into account overflow node
            total_nodes = total_nodes * 2;                         // Double the size for insert and delete operations

            smallsize NUMTHREADS = (total_nodes > MAXBLOCKSIZE) ? MAXBLOCKSIZE : total_nodes;
            DEBUG_COMBINED(" Combine Calling Update Kernel", 2, partition_count_with_overflow, size);

            combined_update_kernel<key_type><<<SDIV(total_nodes, NUMTHREADS), NUMTHREADS, 0, stream>>>(launch_params_buffer.ptr());

#ifdef PRINT_COMBINE_UPDATE_VALUES
            cudaStreamSynchronize(0);
            CUERR
            updatable_cg_params params = launch_params_buffer.download_first_item();
            // Download and print the free_node value
            std::cout << "Done Insertions free_node: " << params.free_node << std::endl;

            total_nodes_used_from_AR = params.free_node;
#endif
        }
    }

    void lookups_static_tree(const key_type *keys, smallsize *result, size_t size, cudaStream_t stream)
    {

        DEBUG_LOOKUPS("Performing Lookups in Static Tree, Size of Probe List: ", 1, size);
        smallsize query_size = size;

#ifdef PRINT_LOOKUP_VALUES
        printf("\n ");
        printf("****************************************************************\n");
        printf("Lookups, Size of Probe List: %d \n", size);
        printf("****************************************************************\n");
        printf("\n ");
        printf("Lookup Function Print all keys in the probe list\n");

        std::vector<key_type> keys_host(size);

        cudaMemcpy(keys_host.data(), keys, size * sizeof(key_type), cudaMemcpyDeviceToHost);
        CUERR

        for (int i = 0; i < size; ++i)
        {
            printf("Lookup Key %d: %llu\n", i, static_cast<unsigned long long>(keys_host[i]));
        }
#endif
        /*
       {
           nvtx3::scoped_range_in<nvtx_rtx_domain> upload{"upload-params"};
           setup_lookup_data_static_tree<<<1, 1, 0, stream>>>(
               launch_params_buffer.ptr(),
               sizeof(key_type) == 8,
               false,
               keys,
               keys,
               result,
               size);
       }
       */

        // smallsize totalnodes = partition_count_with_overflow; // taking into account overflow node
        smallsize NUMTHREADS = (size > MAXBLOCKSIZE) ? MAXBLOCKSIZE : size;

        bool group_lookup_kernel = false;

        // lookup_kernel_tile_static_tree<key_type, node_size_log_static_tree, cg_size_log_static_tree, false><<<SDIV(size, NUMTHREADS /DIV_FACTOR), NUMTHREADS /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), group_lookup_kernel);
        lookup_kernel_tile_static_tree_params<key_type, node_size_log_static_tree, cg_size_log_static_tree, false><<<SDIV(size, NUMTHREADS /DIV_FACTOR), NUMTHREADS /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), keys, result, size);
        // lookup_kernel_tile_static_tree_params_core<false, key_type, node_size_log_static_tree, cg_size_log_static_tree, false><<<SDIV(size, NUMTHREADS /DIV_FACTOR), NUMTHREADS /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(),  keys, result, size);
    }

    void index_layer_lookups(const key_type *keys, size_t size, cudaStream_t stream)
    {
        DEBUG_LOOKUPS("Performing Lookups in Buckets, Size of Probe List: ", 1, size);

        // cuda_buffer<smallsize> bucket_values_buffer;
        smallsize NUMTHREADS = (size > MAXBLOCKSIZE) ? MAXBLOCKSIZE : size;

        {
            bucket_values_buffer.alloc(size);
            bucket_values_buffer.zero();
            // Launch the bucket lookup kernel
            bucket_lookup_kernel_params<key_type, node_size_log_static_tree, cg_size_log_static_tree, false>
                <<<SDIV(size, NUMTHREADS /DIV_FACTOR), NUMTHREADS /DIV_FACTOR, 0, stream>>>(
                    launch_params_buffer.ptr(),
                    keys,
                    bucket_values_buffer.ptr(),
                    static_cast<smallsize>(size));
        }

        // class_bucket_values_buffer = std::move(bucket_values_buffer);
    }

    void bucket_layer_lookups(const key_type *keys, smallsize *result, size_t size, cudaStream_t stream)
    {

        smallsize NUMTHREADS = (size > MAXBLOCKSIZE) ? MAXBLOCKSIZE : size;
        lookup_kernel_tile_static_tree_params_with_buffer<key_type, node_size_log_static_tree, cg_size_log_static_tree, false>
            <<<SDIV(size, NUMTHREADS /DIV_FACTOR), NUMTHREADS /DIV_FACTOR, 0, stream>>>(
                launch_params_buffer.ptr(),
                keys,
                bucket_values_buffer.ptr(),
                result,
                static_cast<smallsize>(size));
    }

    void lookups_ordered(const key_type *keys, smallsize *result, size_t size, cudaStream_t stream)
    {

        DEBUG_LOOKUPS("Performing Lookups No Index Layer -Ordered, Size of Probe List: ", 1, size);
        //xsmallsize query_size = size;

#ifdef PRINT_LOOKUP_VALUES
        printf("\n ");
        printf("****************************************************************\n");
        printf("Lookups, Size of Probe List: %d \n", size);
        printf("****************************************************************\n");
        printf("\n ");
        printf("Lookup Function Print all keys in the probe list\n");

        std::vector<key_type> keys_host(size);

        cudaMemcpy(keys_host.data(), keys, size * sizeof(key_type), cudaMemcpyDeviceToHost);
        CUERR

        for (int i = 0; i < size; ++i)
        {
            printf("Lookup Key %d: %llu\n", i, static_cast<unsigned long long>(keys_host[i]));
        }
#endif
        /*
       {
           nvtx3::scoped_range_in<nvtx_rtx_domain> upload{"upload-params"};
           setup_lookup_data_static_tree<<<1, 1, 0, stream>>>(
               launch_params_buffer.ptr(),
               sizeof(key_type) == 8,
               false,
               keys,
               keys,
               result,
               size);
       }
       */
        // smallsize totalnodes = partition_count_with_overflow; // taking into account overflow node
        // smallsize NUMTHREADS = (size > MAXBLOCKSIZE) ? MAXBLOCKSIZE : size;

        // bool single_threads_per_bucket = false; // use later
        // smallsize threadsPerTile = TILE_SIZE;                               // e.g., 8 or 16
        smallsize total_threads_required = partition_count_with_overflow * TILE_SIZE; // NOT divided by anything
                                                                                      // smallsize maxblocksplus1 = SDIV(blocksPerGrid, MAXBLOCKSIZE /DIV_FACTOR) + 1;
        smallsize max_blocks_required = SDIV(total_threads_required, MAXBLOCKSIZE /DIV_FACTOR);
        // lookup_kernel_tile_static_tree<key_type, node_size_log_static_tree, cg_size_log_static_tree, false><<<SDIV(size, NUMTHREADS /DIV_FACTOR), NUMTHREADS /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), group_lookup_kernel);
        // printf("Going to LOOKUP TILE ORDERED:  div is:%d,  max_blocks_required=%d and MAXBLOCKSIZE/DIV %d \n",DIV, max_blocks_required, MAXBLOCKSIZE /DIV_FACTOR);

        lookup_kernel_tile_ordered<key_type><<<max_blocks_required, MAXBLOCKSIZE /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), keys, result, size);
        // lookup_kernel_tile_static_tree_params_core<false, key_type, node_size_log_static_tree, cg_size_log_static_tree, false><<<SDIV(size, NUMTHREADS /DIV_FACTOR), NUMTHREADS /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(),  keys, result, size);
    }

    void lookup_cuda(const key_type *keys, smallsize *result, size_t size, cudaStream_t stream)
    {
        {

            DEBUG_LOOKUPS("Performing Lookups, Size of Probe List: ", 1, size);

            nvtx3::scoped_range_in<nvtx_rtx_domain> upload{"upload-params"};
            setup_lookup_data<<<1, 1, 0, stream>>>(
                launch_params_buffer.ptr(),
                sizeof(key_type) == 8,
                false,
                keys,
                keys,
                result,
                size);
        }

        // smallsize totalnodes = partition_count_with_overflow; // taking into account overflow node
        smallsize NUMTHREADS = (size > MAXBLOCKSIZE) ? MAXBLOCKSIZE : size;
        // printf("No Rays Reg LOOKUP KERNEL: size %d, NUMTHREADS %d \n", size, NUMTHREADS);
        bool group_lookup_kernel = false;

#ifdef LOOKUP_KERNEL_TILE
#pragma message "LOOKUP_KERNEL_TILE=YES"

        // lookup_kernel_tile_loop<key_type><<<SDIV(size, NUMTHREADS/8),  NUMTHREADS/8, 0, stream>>>(launch_params_buffer.ptr(), group_lookup_kernel);

        lookup_kernel_tile_loop<key_type><<<SDIV(size, NUMTHREADS /DIV_FACTOR), NUMTHREADS /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), group_lookup_kernel);

        // lookup_kernel_tile_loop_sharedMem<key_type><<<SDIV(size, NUMTHREADS),  NUMTHREADS, 0, stream>>>(launch_params_buffer.ptr(), group_lookup_kernel);
        // lookup_kernel_tile_loop_SM<key_type><<<SDIV(size, NUMTHREADS/4),  NUMTHREADS/4, 0, stream>>>(launch_params_buffer.ptr(), group_lookup_kernel);

#else
        // #pragma message "LOOKUP_KERNEL_REGULAR=YES"

        lookup_kernel<key_type><<<SDIV(size, NUMTHREADS /DIV_FACTOR), NUMTHREADS /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), group_lookup_kernel);
#endif
    }

    //--------------------- END COMBINE INSERT AND DEL ---------------------
    void lookup(const key_type *keys, smallsize *result, size_t size, cudaStream_t stream)
    {

#ifdef PRINT_LOOKUP_VALUES
        printf("\n ");
        printf("****************************************************************\n");
        printf("Lookups, Size of Probe List: %d \n", size);
        printf("****************************************************************\n");
        printf("\n ");
        printf("Lookup Function Print all keys in the probe list\n");

        std::vector<key_type> keys_host(size);

        cudaMemcpy(keys_host.data(), keys, size * sizeof(key_type), cudaMemcpyDeviceToHost);
        CUERR

        for (int i = 0; i < size; ++i)
        {
            printf("Lookup Key %d: %llu\n", i, static_cast<unsigned long long>(keys_host[i]));
        }
#endif

#ifdef LOOKUPS_CUDA
#pragma message "LOOKUPS_CUDA=YES"
        lookup_cuda(keys, result, size, stream);
        return;
#endif

#pragma message "LOOKUPS_CUDA=NO"
        {

            DEBUG_LOOKUPS("Performing Lookups, Size of Probe List: ", 1, size);

            nvtx3::scoped_range_in<nvtx_rtx_domain> upload{"upload-params"};
            setup_lookup_data<<<1, 1, 0, stream>>>(
                launch_params_buffer.ptr(),
                sizeof(key_type) == 8,
                false,
                keys,
                keys,
                result, size);
        }

        {
            nvtx3::scoped_range_in<nvtx_rtx_domain> launch{"launch"};
            OPTIX_CHECK(optixLaunch(
                pipeline->pipeline,
                stream,
                launch_params_buffer.cu_ptr(),
                launch_params_buffer.size_in_bytes(),
                &pipeline->sbt,
                size,
                1,
                1))
        }
    }

    void multi_lookup_sum(const key_type *keys, smallsize *result, size_t size, cudaStream_t stream)
    {

        // todo
    }

    void range_lookup_sum(const key_type *lower, const key_type *upper, smallsize *result, size_t size, cudaStream_t stream)
    {

        // todo
    }


    void lookups_successor(const key_type *keys, key_type *result, size_t size, cudaStream_t stream)
    {

        DEBUG_LOOKUPS("Performing Successor Lookups- Ordered, Size of Probe List: ", 1, size);
        //smallsize query_size = size;

#ifdef PRINT_SUCCESSOR_VALUES
        printf("\n ");
        printf("****************************************************************\n");
        printf("Lookups Successor, Size of Probe List: %d \n", size);
        printf("****************************************************************\n");
        printf("\n ");
        printf("Lookup Successor Function Print all keys in the probe list\n");

        std::vector<key_type> keys_host(size);

        cudaMemcpy(keys_host.data(), keys, size * sizeof(key_type), cudaMemcpyDeviceToHost);
        CUERR

        for (int i = 0; i < size; ++i)
        {
            printf("Successor Key %d: %llu\n", i, static_cast<unsigned long long>(keys_host[i]));
        }
#endif
       
        smallsize total_threads_required = partition_count_with_overflow * TILE_SIZE; // NOT divided by anything
                                                                                      // smallsize maxblocksplus1 = SDIV(blocksPerGrid, MAXBLOCKSIZE /DIV_FACTOR) + 1;
        smallsize max_blocks_required = SDIV(total_threads_required, MAXBLOCKSIZE /DIV_FACTOR);
       
        lookup_kernel_successor_tile_ordered<key_type><<<max_blocks_required, MAXBLOCKSIZE /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), keys, result, size);
    }

    void destroy()
    {
        as_buffer.free();
        ordered_node_pairs_buffer.free();
        launch_params_buffer.free();
        allocation_buffer.free();
        maxvalues_buffer.free();
        // reuse_list_buffer.free();
        copy_buffer.free();
        tree_buffer.free();
        bucket_values_buffer.free();
        // copy_update_list.free();
        // copy_offset_list.free();
    }
};

#endif



/* /iffalse 
 void insert_previous1(const key_type *update_list, const smallsize *offsets, size_t size, cudaStream_t stream)
    {

        smallsize *result = nullptr;

#ifdef PRINT_INSERT_VALUES
        printf("\n ");
        printf("****************************************************************\n");
        printf("Insertions, Size of Insert List: %d \n", size);
        printf("****************************************************************\n");
        printf("\n ");
        printf("Insert Function Print all new key-offset pairs in the update list\n");
        assert(ordered_node_pairs_buffer.size_in_bytes() % node_stride == 0);

        std::vector<key_type> keys_host(size);
        std::vector<smallsize> offsets_host(size);

        cudaMemcpy(keys_host.data(), update_list, size * sizeof(key_type), cudaMemcpyDeviceToHost);
        CUERR
        cudaMemcpy(offsets_host.data(), offsets, size * sizeof(smallsize), cudaMemcpyDeviceToHost);
        CUERR

        for (int i = 0; i < size; ++i)
        {
            key_type key = keys_host[i];
            printf("Key %d: %llu, Offset: %u\n", i, static_cast<unsigned long long>(key), static_cast<unsigned>(offsets_host[i]));
        }
#endif
        // DEBUG_UPDATES("Insert Function ", 2, partition_count, partition_count_with_overflow);

          {
              nvtx3::scoped_range_in<nvtx_rtx_domain> upload{"upload-params"};
              setup_update_data<<<1, 1, 0, stream>>>(
                  launch_params_buffer.ptr(),
                  sizeof(key_type) == 8,
                  false,
                  update_list,
                  offsets,
                  size,
                  result);
          } 

        {

            // DEBUG_UPDATES("Insert:Calling Update Kernel", 2, partition_count_with_overflow, size);

#ifdef PRINT_INSERT_VALUES
            cudaFuncAttributes attr;
            cudaError_t err = cudaFuncGetAttributes(&attr, update_kernel<key_type>);

            if (err != cudaSuccess)
            {
                printf(" Insert: cudaFuncGetAttributes failed with error: %s\n", cudaGetErrorString(err));
            }
            else
            {
                printf("Shared memory per block: %zu\n", attr.sharedSizeBytes);
                printf("Registers per block: %d\n", attr.numRegs);
            }

            DEBUG_UPDATES("Inserts Call to Update kernel", 3, total_nodes, MAXBLOCKSIZE, NUMTHREADS);

#endif

#ifdef TILE_INSERTS
#pragma message "TILE INSERTS=YES"

#ifdef TILE_INSERTS_A
#pragma message "TILE INSERTS_A_=YES"
            smallsize total_nodes = partition_count_with_overflow; // taking into account overflow node
            smallsize NUMTHREADS = (total_nodes > MAXBLOCKSIZE) ? MAXBLOCKSIZE : total_nodes;
            const int total_tiles = partition_count_with_overflow * TILE_SIZE;
            const int tiles_per_block = 8; //
            const int threads_per_block = TILE_SIZE * tiles_per_block;
            const int num_blocks = (total_tiles + tiles_per_block - 1) / tiles_per_block;

            // printf("Tile Insert: num_blocks %d, threads_per_block %d\n", num_blocks, threads_per_block);

            update_kernel_tile<key_type><<<num_blocks, threads_per_block, 0, stream>>>(launch_params_buffer.ptr(), false);
            // update_kernel_tile<key_type><<<SDIV(total_nodes, NUMTHREADS), NUMTHREADS, 0, stream>>>(launch_params_buffer.ptr(), false);
#elif defined(TILE_INSERTS_B)
#pragma message "TILE INSERTS_B_=YES"
            smallsize total_nodes = partition_count_with_overflow; // taking into account overflow node
            smallsize NUMTHREADS = (total_nodes > MAXBLOCKSIZE) ? MAXBLOCKSIZE : total_nodes;
            total_nodes = partition_count_with_overflow * TILE_SIZE; // taking into account overflow node
            NUMTHREADS = (total_nodes > MAXBLOCKSIZE) ? MAXBLOCKSIZE : total_nodes;
            update_kernel_tile<key_type><<<SDIV(total_nodes, NUMTHREADS), NUMTHREADS, 0, stream>>>(launch_params_buffer.ptr(), false);

#elif defined(TILE_INSERTS_C)
#pragma message "TILE INSERTS_C_=YES"
            // NUMTHREADS = MAXBLOCKSIZE;
            //  total_nodes = 1000000;

            smallsize threadsPerBlock = TILE_SIZE;                               // e.g., 8 or 16
            smallsize blocksPerGrid = partition_count_with_overflow * TILE_SIZE; // NOT divided by anything
            // smallsize maxblocksplus1 = SDIV(blocksPerGrid, MAXBLOCKSIZE /DIV_FACTOR) + 1;

            smallsize maxblocks_required = SDIV(blocksPerGrid, MAXBLOCKSIZE /DIV_FACTOR);

            update_kernel_tile_inserts<key_type><<<maxblocks_required, MAXBLOCKSIZE /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), update_list, offsets, size);

#else
#pragma message "WARP_INSERTS_D_=YES"
            const int threadsPerBlock = MAXBLOCKSIZE;              // e.g., 256
            const int warpsPerBlock = threadsPerBlock / WARP_SIZE; // e.g., 8
            const int blocksPerGrid = SDIV(partition_count_with_overflow, warpsPerBlock);

            update_kernel_warp<key_type><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                launch_params_buffer.ptr(), false);

#endif

#else
#pragma message "TILE INSERTS=NO"

            // smallsize total_nodes = partition_count_with_overflow; // taking into account overflow node
            smallsize total_nodes = partition_count_with_overflow; // taking into account overflow node
            smallsize NUMTHREADS = (total_nodes > MAXBLOCKSIZE) ? MAXBLOCKSIZE : total_nodes;
            update_kernel<key_type><<<SDIV(total_nodes, NUMTHREADS /DIV_FACTOR), NUMTHREADS /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), false);
#endif

#ifdef PRINT_INSERT_VALUES
            cudaStreamSynchronize(0);
            CUERR
            updatable_cg_params params = launch_params_buffer.download_first_item();
            // Download and print the free_node value
            std::cout << "Done Insertions free_node: " << params.free_node << std::endl;

            total_nodes_used_from_AR = params.free_node;
#endif
        }
    }

    void insert_previous2(const key_type *update_list, const smallsize *offsets, size_t size, cudaStream_t stream)
    {

        smallsize *result = nullptr;

#ifdef PRINT_INSERT_VALUES
        printf("\n ");
        printf("****************************************************************\n");
        printf("Insertions, Size of Insert List: %d \n", size);
        printf("****************************************************************\n");
        printf("\n ");
        printf("Insert Function Print all new key-offset pairs in the update list\n");
        assert(ordered_node_pairs_buffer.size_in_bytes() % node_stride == 0);

        std::vector<key_type> keys_host(size);
        std::vector<smallsize> offsets_host(size);

        cudaMemcpy(keys_host.data(), update_list, size * sizeof(key_type), cudaMemcpyDeviceToHost);
        CUERR
        cudaMemcpy(offsets_host.data(), offsets, size * sizeof(smallsize), cudaMemcpyDeviceToHost);
        CUERR

        for (int i = 0; i < size; ++i)
        {
            key_type key = keys_host[i];
            printf("Key %d: %llu, Offset: %u\n", i, static_cast<unsigned long long>(key), static_cast<unsigned>(offsets_host[i]));
        }
#endif
        DEBUG_UPDATES("Insert Function ", 2, partition_count, partition_count_with_overflow);

        {
            nvtx3::scoped_range_in<nvtx_rtx_domain> upload{"upload-params"};
            setup_update_data<<<1, 1, 0, stream>>>(
                launch_params_buffer.ptr(),
                sizeof(key_type) == 8,
                false,
                update_list,
                offsets,
                size,
                result);
        }

        {

            smallsize total_nodes = partition_count_with_overflow; // taking into account overflow node
            smallsize NUMTHREADS = (total_nodes > MAXBLOCKSIZE) ? MAXBLOCKSIZE : total_nodes;
            DEBUG_UPDATES("Insert:Calling Update Kernel", 2, partition_count_with_overflow, size);

#ifdef PRINT_INSERT_VALUES
            cudaFuncAttributes attr;
            cudaError_t err = cudaFuncGetAttributes(&attr, update_kernel<key_type>);

            if (err != cudaSuccess)
            {
                printf(" Insert: cudaFuncGetAttributes failed with error: %s\n", cudaGetErrorString(err));
            }
            else
            {
                printf("Shared memory per block: %zu\n", attr.sharedSizeBytes);
                printf("Registers per block: %d\n", attr.numRegs);
            }

            DEBUG_UPDATES("Inserts Call to Update kernel", 3, total_nodes, MAXBLOCKSIZE, NUMTHREADS);

#endif

#ifdef TILE_INSERTS
#pragma message "TILE INSERTS=YES"

#ifdef TILE_INSERTS_A
#pragma message "TILE INSERTS_A_=YES"

            const int total_tiles = partition_count_with_overflow * TILE_SIZE;
            const int tiles_per_block = 8; //
            const int threads_per_block = TILE_SIZE * tiles_per_block;
            const int num_blocks = (total_tiles + tiles_per_block - 1) / tiles_per_block;

            // printf("Tile Insert: num_blocks %d, threads_per_block %d\n", num_blocks, threads_per_block);

            update_kernel_tile<key_type><<<num_blocks, threads_per_block, 0, stream>>>(launch_params_buffer.ptr(), false);
            // update_kernel_tile<key_type><<<SDIV(total_nodes, NUMTHREADS), NUMTHREADS, 0, stream>>>(launch_params_buffer.ptr(), false);
#elif defined(TILE_INSERTS_B)
#pragma message "TILE INSERTS_B_=YES"

            total_nodes = partition_count_with_overflow * TILE_SIZE; // taking into account overflow node
            NUMTHREADS = (total_nodes > MAXBLOCKSIZE) ? MAXBLOCKSIZE : total_nodes;
            update_kernel_tile<key_type><<<SDIV(total_nodes, NUMTHREADS), NUMTHREADS, 0, stream>>>(launch_params_buffer.ptr(), false);

#elif defined(TILE_INSERTS_C)
#pragma message "TILE INSERTS_C_=YES"
            NUMTHREADS = MAXBLOCKSIZE;
            //  total_nodes = 1000000;

            smallsize threadsPerBlock = TILE_SIZE;                               // e.g., 8 or 16
            smallsize blocksPerGrid = partition_count_with_overflow * TILE_SIZE; // NOT divided by anything
            // smallsize maxblocksplus1 = SDIV(blocksPerGrid, MAXBLOCKSIZE /DIV_FACTOR) + 1;

            smallsize maxblocks_required = SDIV(blocksPerGrid, MAXBLOCKSIZE /DIV_FACTOR);

            update_kernel_tile<key_type><<<maxblocks_required, MAXBLOCKSIZE /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), false);

            // update_kernel_tile<key_type><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(launch_params_buffer.ptr(), false);

            //   #elif defined(TILE_INSERTS_BULK)
            //   #pragma message "TILE INSERTS_BULK_=YES"
            //   // Bulk inserts with tile size
            //   smallsize threadsPerBlock = TILE_SIZE;  // e.g., 8 or 16
            //   smallsize blocksPerGrid = partition_count_with_overflow*TILE_SIZE;  // NOT divided by anything
            //   smallsize maxblocksplus1 = SDIV(blocksPerGrid, MAXBLOCKSIZE/DIV) + 1;
            //   update_kernel_tile_bulk<key_type><<<maxblocksplus1, MAXBLOCKSIZE/DIV, 0, stream>>>(launch_params_buffer.ptr(), false);
            //    update_kernel_tile<key_type><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(launch_params_buffer.ptr(), false);

#else
#pragma message "WARP_INSERTS_D_=YES"
            const int threadsPerBlock = MAXBLOCKSIZE;              // e.g., 256
            const int warpsPerBlock = threadsPerBlock / WARP_SIZE; // e.g., 8
            const int blocksPerGrid = SDIV(partition_count_with_overflow, warpsPerBlock);

            update_kernel_warp<key_type><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                launch_params_buffer.ptr(), false);

#endif

#else
#pragma message "TILE INSERTS=NO"
            update_kernel<key_type><<<SDIV(total_nodes, NUMTHREADS /DIV_FACTOR), NUMTHREADS /DIV_FACTOR, 0, stream>>>(launch_params_buffer.ptr(), false);
#endif

#ifdef PRINT_INSERT_VALUES
            cudaStreamSynchronize(0);
            CUERR
            updatable_cg_params params = launch_params_buffer.download_first_item();
            // Download and print the free_node value
            std::cout << "Done Insertions free_node: " << params.free_node << std::endl;

            total_nodes_used_from_AR = params.free_node;
#endif
        }
    }


    */