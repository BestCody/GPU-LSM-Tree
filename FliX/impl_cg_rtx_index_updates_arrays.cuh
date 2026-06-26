// =============================================================================
// File: impl_cg_rtx_index_updates_arrays.cuh
// Author: Justus Henneberg
// Description: Implements impl_cg_rtx_index_updates_arrays     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INDEX_PROTOTYPE_IMPL_CG_RTX_INDEX_UPDATES_CUH
#define INDEX_PROTOTYPE_IMPL_CG_RTX_INDEX_UPDATES_CUH

#include "definitions.cuh"
#include "definitions_updates.cuh"
#include "definitions_coarse_granular.cuh"
#include "coarse_granular_inserts.cuh"
#include "coarse_granular_deletes.cuh"
#include "launch_parameters.cuh"



//#define BUCKET_INSERTS
#define OPTIMIZATION_ON // Optimization for maxvalues to minimize # of rays fired/search
extern "C" char coarse_granular_embedded_updates_ptx_code[];
extern optix_wrapper optix;

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

        if (i < numpartitions - 1)
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
        DEBUG_MAXVALUE("No Optimization: Max Val Kernel: tid and maxvalues ", 2, i, maxvalues[i]);
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

template <typename key_type>
GLOBALQUALIFIER void update_kernel(updatable_cg_params *launch_params, bool perform_dels)
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
    key_type maxkey = maxbuf[idx]; // curr_node[0]; // Assuming the key to search is at position 2
    key_type minkey = 1;

    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);

    if (idx > num_partitions_with_overflow - 1)
    {
        return;
    }
    if (idx > 0)
    {
        minkey = maxbuf[idx - 1] + 1; // max of previous node plus 1
    }

    int minindex = binarySearchIndex<key_type>(static_cast<const key_type *>(launch_params->update_list), minkey, 0, update_size, false);
    int maxindex = binarySearchIndex<key_type>(static_cast<const key_type *>(launch_params->update_list), maxkey, 0, update_size, true);

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

    if (perform_dels)
        // process_deletes<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
        process_deletes_tombstones<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
 
    else
        //processInserts<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
        //process_inserts_per_bucket<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);
        process_inserts_per_bucket_tombstones<key_type>(maxkey, (smallsize)minindex, (smallsize)maxindex, launch_params, curr_node, currnodesize, tid);

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
    cuda_buffer<key_type> &copy_update_list,
    cuda_buffer<smallsize> &copy_offset_list,
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

     size_t copy_update_list_size_bytes = partition_count_with_overflow * node_size * sizeof(key_type);
     size_t copy_offset_list_size_bytes = partition_count_with_overflow * node_size * sizeof(smallsize);
    // this is required to find the maximum space needed for the build
    auto size_snapshot = [&]
    {
        //return ordered_node_pairs_buffer.size_in_bytes() + allocation_buffer.size_in_bytes() + maxvalues_buffer.size_in_bytes() + ordered_key_buffer.size_in_bytes() + offset_buffer.size_in_bytes() + ordered_offset_buffer.size_in_bytes() + sort_temp_buffer.size_in_bytes();

        return   copy_update_list_size_bytes + copy_offset_list_size_bytes +ordered_node_pairs_buffer.size_in_bytes() + allocation_buffer.size_in_bytes() + maxvalues_buffer.size_in_bytes() + ordered_key_buffer.size_in_bytes() + offset_buffer.size_in_bytes() + ordered_offset_buffer.size_in_bytes() + sort_temp_buffer.size_in_bytes();
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
    copy_update_list.alloc(partition_count_with_overflow*node_size);
    CUERR
    copy_offset_list.alloc(partition_count_with_overflow*node_size);
    CUERR
    copy_offset_list.zero();
    CUERR
    copy_update_list.zero();
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
GLOBALQUALIFIER void setup_update_data(
    updatable_cg_params *launch_params,
    bool long_keys,
    bool aggregate_results,
    const key_type *update_list,
    const smallsize *offset_list,
    smallsize size,
    smallsize *result
)
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
GLOBALQUALIFIER void setup_lookup_data(
    updatable_cg_params *launch_params,
    bool long_keys,
    bool aggregate_results,
    const key_type *query_lower,
    const key_type *query_upper,
    smallsize *result)
{
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1)
        return;

    launch_params->long_keys = long_keys;
    launch_params->aggregate_results = aggregate_results;
    launch_params->query_lower = query_lower;
    launch_params->query_upper = query_upper;
    launch_params->result = result;
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
    key_type* copy_update_list,
    smallsize* copy_offset_list,
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
    launch_params->copy_update_list = copy_update_list;
    launch_params->copy_offset_list = copy_offset_list;
    launch_params->partition_count = partition_count;
    launch_params->partition_count_with_overflow = partition_count_with_overflow;
    launch_params->free_node = 0;
    launch_params->allocation_buffer_count = extra_allocated_nodes;
}

template <typename key_type_, uint8_t cache_line = 0, uint8_t node_size_log = 0, uint8_t percentage_initial_fill = 50, uint32_t additional_inserts_percentage = 0>
class cg_rtx_index_updates
{
    static_assert(sizeof(key_type_) == 4 || sizeof(key_type_) == 8, "key_type must be 4 or 8 bytes");
    static_assert(cache_line == 0 || cache_line == 5 || cache_line == 10 || cache_line == 20, "cache_line must be encoded as 0 (0.0), 5 (0.5), 10 (1.0), or 20 (2.0)");
    static_assert(percentage_initial_fill <= 100, "percentage_initial_fill cannot exceed 100");
    static_assert(node_size_log >= 0 && node_size_log <= 8, "node_size_log must be between 0 and 8 (inclusive)");
    static_assert(additional_inserts_percentage <= 200, "additional_inserts_percentage cannot exceed 200");
    static_assert((cache_line == 0) + (node_size_log == 0) == 1, "Exactly one of cache_line or node_size_log must be 0");
    static_assert(((cache_line == 10) && (sizeof(key_type_) == 4)) || cache_line != 10, "Cache_line == 10 supported only for 32-bit keys in this update");

public:
    using key_type = key_type_;

private:
    cuda_buffer<uint8_t> as_buffer;
    cuda_buffer<uint8_t> ordered_node_pairs_buffer;
    cuda_buffer<uint8_t> allocation_buffer;
    cuda_buffer<key_type> maxvalues_buffer;
    cuda_buffer<key_type> copy_update_list;
    cuda_buffer<smallsize> copy_offset_list;    
    cuda_buffer<updatable_cg_params> launch_params_buffer;
    std::optional<optix_pipeline> pipeline;

    key_type *maxvalues;

public:
    static constexpr bool can_lookup = true;
    static constexpr bool can_multi_lookup = false;
    static constexpr bool can_range_lookup = false;
    static constexpr bool can_update = true;

    static constexpr smallsize key_offset_pair = compute_key_offset_bytes<key_type>();
    static constexpr smallsize node_size = compute_node_size<key_type>(cache_line, node_size_log);
    static constexpr smallsize node_stride = compute_nodestride_bytes<key_type>(node_size, cache_line); // size of the node in bytes

    static constexpr smallsize partition_size = SDIV(node_size, 2);
    static constexpr smallsize initial_fill = percentage_initial_fill;
    static constexpr smallsize additional_inserts = additional_inserts_percentage;

    smallsize partition_count = 0;
    smallsize partition_count_with_overflow = 0;
    smallsize total_nodes_used_from_AR = 0;
    size_t total_allocation_region_nodes = 0;
    static constexpr smallsize initial_num_keys_each_node = (initial_fill * node_size + 99) / 100;

    static std::string short_description()
    {
        return std::string("cg_rtx_index_updates_") + std::to_string(node_size_log);
    }

    static size_t additional_nodes_required(size_t size, size_t initial_nodes_with_overflow)
    {

        DEBUG_EXTRA_NODES("Additional Nodes Calculation", 2, size, initial_nodes_with_overflow);

        size_t total_keys = size * (100 + additional_inserts) / 100;
        size_t total_nodes = SDIV(total_keys, partition_size); // Assuming half full nodes
        size_t total_nodes_without_initial = total_nodes > initial_nodes_with_overflow ? total_nodes - initial_nodes_with_overflow : 0;

        // Increase by 1.75x using integer arithmetic
        //---> The increased number of nodes is No Longer Needed
        size_t total_nodes_without_initial_increased = total_nodes_without_initial * 175 / 100;

        return total_nodes_without_initial;
    }

    static size_t estimate_build_bytes(size_t size)
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

        size_t total_allocation_nodes = additional_nodes_required(size, num_partitions_plus_overflow);

        size_t allocation_region_bytes = node_stride * total_allocation_nodes;
        size_t max_value_buffer_bytes = num_partitions_plus_overflow * sizeof(key_type);
        size_t copy_update_list_size_bytes = num_partitions_plus_overflow * node_size * sizeof(key_type);
        size_t copy_offset_list_size_bytes = num_partitions_plus_overflow * node_size * sizeof(smallsize);
        size_t additional_gpu_buffers_bytes = allocation_region_bytes + max_value_buffer_bytes + copy_update_list_size_bytes + copy_offset_list_size_bytes;

        size_t phase_1_bytes = permutation_bytes + sort_aux_bytes + additional_gpu_buffers_bytes;
        size_t phase_2_bytes = permutation_bytes + triangle_bytes + bvh_bytes + aux_bytes + additional_gpu_buffers_bytes;
        size_t phase_3_bytes = permutation_bytes + 2 * bvh_bytes + additional_gpu_buffers_bytes;
        return std::max(phase_1_bytes, std::max(phase_2_bytes, phase_3_bytes));
    }


    //UPDATE with copy arrays and update build structures
    size_t gpu_resident_bytes()
    {

        updatable_cg_params params = launch_params_buffer.download_first_item();
        total_nodes_used_from_AR = params.free_node;
        size_t copy_update_list_size_bytes = partition_count_with_overflow * node_size * sizeof(key_type);
        size_t copy_offset_list_size_bytes = partition_count_with_overflow * node_size * sizeof(smallsize);
        
       // return as_buffer.size_in_bytes() + ordered_node_pairs_buffer.size_in_bytes() + (total_nodes_used_from_AR)*node_stride + maxvalues_buffer.size_in_bytes() + launch_params_buffer.size_in_bytes();
        return as_buffer.size_in_bytes() + ordered_node_pairs_buffer.size_in_bytes() + (total_nodes_used_from_AR)*node_stride + maxvalues_buffer.size_in_bytes() + launch_params_buffer.size_in_bytes();

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
        return total_nodes_used_from_AR; //total_allocation_region_nodes;
    }

    void build(const key_type *keys, size_t keysize, double *build_time_ms, size_t *build_bytes)
    {

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

        DEBUG_BUILD_PARAMS("New Build params", 3, initial_fill, keysize, key_offset_pair);
        DEBUG_BUILD_PARAMS("More Build params", 5, cache_line, node_size, partition_size, node_stride, percentage_initial_fill);

        assert(initial_num_keys_each_node <= node_size); // Ensure not exceeding max capacity
        partition_count = SDIV(keysize, initial_num_keys_each_node);
        partition_count_with_overflow = partition_count + 1;

        DEBUG_BUILD("BUILD Partition Count", 3, partition_count, partition_count_with_overflow, initial_num_keys_each_node);
        DEBUG_BUILD("BUILD node params", 4, cache_line, node_size_log, percentage_initial_fill, additional_inserts);

        total_allocation_region_nodes = additional_nodes_required(keysize, partition_count_with_overflow);

        DEBUG_BUILD("BUILD Allocated ", 1, total_allocation_region_nodes);


    /*    // ------------------ BUCKET_INSERTS
    // Allocate and zero out the copy_update_list and copy_offset_list buffers
    key_type *copy_update_list;
    smallsize *copy_offset_list;
    size_t buffer_size = partition_count_with_overflow * sizeof(key_type);

    cudaMalloc(&copy_update_list, buffer_size);
    cudaMemset(copy_update_list, 0, buffer_size);

    buffer_size = partition_count_with_overflow * sizeof(smallsize);
    cudaMalloc(&copy_offset_list, buffer_size);
    cudaMemset(copy_offset_list, 0, buffer_size);
    //------------------ BUCKET_INSERTS
    */
        OptixTraversableHandle traversable = build_structures(
            *pipeline->optix,
            keys,
            keysize,
            partition_size,
            node_size,
            partition_count,
            partition_count_with_overflow,
            as_buffer,
            ordered_node_pairs_buffer,
            allocation_buffer,
            maxvalues_buffer,
            copy_update_list,
            copy_offset_list,
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
            keysize,
            partition_size,
            node_size,
            copy_update_list,
            copy_offset_list,
            partition_count,
            partition_count_with_overflow,
            total_allocation_region_nodes);

        cudaDeviceSynchronize();
        CUERR
         // Free the allocated buffers after use
        // cudaFree(copy_update_list);
        //cudaFree(copy_offset_list);

    }

    void insert(const key_type *update_list, const smallsize *offsets, size_t size, cudaStream_t stream)
    {

        smallsize *result = nullptr;

#ifdef PRINT_INSERT_VALUES
        printf("\n ");
        printf("****************************************************************\n");
        printf("Insertions, Size of Insert List: %d \n", size);
        printf("****************************************************************\n");
        printf("\n ");
        printf("Inset Function Print all new key-offset pairs in the update list\n");
        assert(ordered_node_pairs_buffer.size_in_bytes() % node_stride == 0);

        std::vector<key_type> keys_host(size);
        std::vector<smallsize> offsets_host(size);

        cudaMemcpy(keys_host.data(), update_list, size * sizeof(key_type), cudaMemcpyDeviceToHost);
        CUERR
        cudaMemcpy(offsets_host.data(), offsets, size * sizeof(smallsize), cudaMemcpyDeviceToHost);
        CUERR

        for (int i = 0; i < size; ++i)
        {
            printf("Key %d: %llu, Offset: %u\n", i, static_cast<unsigned long long>(keys_host[i]), static_cast<unsigned>(offsets_host[i]));
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

            update_kernel<key_type><<<SDIV(total_nodes, NUMTHREADS), NUMTHREADS, 0, stream>>>(launch_params_buffer.ptr(), false);

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

    void remove(const key_type *update_list, size_t size, cudaStream_t stream)
    {

        smallsize *offsets = nullptr;
        smallsize *result = nullptr;
        //key_type *copy_update_list = nullptr;
        //smallsize *copy_offset_list = nullptr;

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
            DEBUG_UPDATES("Remove: Calling Update Kernel", 2, partition_count_with_overflow, size);

            update_kernel<key_type><<<SDIV(total_nodes, NUMTHREADS), NUMTHREADS, 0, stream>>>(launch_params_buffer.ptr(), true);
        }
    }

    void lookup(const key_type *keys, smallsize *result, size_t size, cudaStream_t stream)
    {
        {

            DEBUG_LOOKUPS("Performing Lookups, Size of Probe List: ",1, size);
    
            nvtx3::scoped_range_in<nvtx_rtx_domain> upload{"upload-params"};
            setup_lookup_data<<<1, 1, 0, stream>>>(
                launch_params_buffer.ptr(),
                sizeof(key_type) == 8,
                false,
                keys,
                keys,
                result);
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

    void destroy()
    {
        as_buffer.free();
        ordered_node_pairs_buffer.free();
        launch_params_buffer.free();
        allocation_buffer.free();
        maxvalues_buffer.free();
        copy_update_list.free();
        copy_offset_list.free();
    }
};

#endif
