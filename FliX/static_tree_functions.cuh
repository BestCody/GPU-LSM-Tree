// =============================================================================
// File: static_tree_functions.cuh
// Author: Justus Henneberg
// Description: Implements static_tree_functions     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================


template <typename key_type>
DEVICEQUALIFIER void process_lookups_tile_ldg(
    key_type curr_node_max,
    smallsize bucket_index,
    key_type probe_key,
    updatable_cg_params *params,
    void *initial_node,
    key_type curr_node_size,
    smallsize query_id,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    // Keep __ldg for primitive scalars only, remove for pointers
    const auto allocation_buffer = params->allocation_buffer;
    const smallsize node_stride = __ldg(&params->node_stride);
    const smallsize node_size = __ldg(&params->node_size);
    const smallsize partition_size = __ldg(&params->partition_size);
    const key_type *maxbuf = static_cast<const key_type *>(params->maxvalues);
    const smallsize partition_count = __ldg(&params->partition_count);
    const smallsize partition_count_with_overflow = __ldg(&params->partition_count_with_overflow);
    const smallsize allocation_buffer_count = __ldg(&params->allocation_buffer_count);
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

    void *curr_node = initial_node;
    key_type curr_max = cg::extract<key_type>(curr_node, 0);
    smallsize thread_id = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    if (thread_id == 0)
    {
        while (curr_max < probe_key)
        {
            smallsize last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes) - 1;
            if (last_position_value >= allocation_buffer_count)
            {
                printf("Error: last position value out of bounds, tid: %u,tile_id %u, probe_key %u, bucket_index %u, curr_max %u, lastposition value %u \n", thread_id, tile_id, probe_key, bucket_index, curr_max, last_position_value);
                //print_node<key_type>(curr_node, node_size);
                return;
            }
            curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            curr_max = cg::extract<key_type>(curr_node, 0);
        }
    }

    uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
    raw_ptr = tile.shfl(raw_ptr, 0);
    curr_node = reinterpret_cast<void *>(raw_ptr);

    //if(thread_id == 0) printf(" tid: %u, tile_id %u, probe_key %u, bucket_index %u, max_curr_node %u \n", thread_id, tile_id, probe_key, bucket_index, curr_max);


    smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
    smallsize found_index = 0;

    DEBUG_LOOKUP_DEV("BEFORE search ", tile.meta_group_rank(), probe_key, curr_size);

    bool found = search_only_cuda_buffer_with_tile<key_type>(tile, curr_node, probe_key, curr_size, found_index);

    DEBUG_LOOKUP_DEV("after search ", found, tile.meta_group_rank(), probe_key);
    DEBUG_LOOKUP_DEV("2nd after search ", probe_key, tile.meta_group_rank(), found_index, query_id);

    if (!found)
    {
        params->result[query_id] = not_found;
        return;
    }

    smallsize result = extract_offset_node<key_type>(curr_node, found_index);
    params->result[query_id] = result;
}

template <typename key_type, int16_t node_size_log, uint16_t cg_size_log, bool caching_enabled>
GLOBALQUALIFIER void lookup_kernel_tile_static_tree(updatable_cg_params* __restrict__ launch_params,
                                                    bool perform_range_query_lookup)
{
    namespace cg = coop_g;

    cg::thread_block block                = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    const int       global_thread_id      = blockIdx.x * blockDim.x + threadIdx.x;
    const smallsize lane                  = tile.thread_rank();

    // RO inputs (maybe use __ldg()): ? 
    const smallsize query_size            = launch_params->query_size;
    const key_type* __restrict__ query   = static_cast<const key_type*>(launch_params->query_lower);
    const key_type* __restrict__ tree    = static_cast<const key_type*>(launch_params->tree_buffer);
    const auto      metadata_out          = launch_params->metadata;
    const smallsize num_buckets_with_of   = launch_params->partition_count_with_overflow;
    const smallsize node_stride           = launch_params->node_stride;

 
    uint8_t* __restrict__ node_buf        = static_cast<uint8_t*>(launch_params->ordered_node_pairs);

    
    constexpr bool use_row_layout = false;
    const key_type* shmem_tree    = nullptr;
    const smallsize cached_entries_count = 0;

    // Compute the base index according to tile and thread
    const int base_idx = global_thread_id - static_cast<int>(lane);

    // Each thread reads its candidate key
    key_type my_key     = key_type{};
    smallsize my_qindex = 0;
    bool in_range       = false;

    if (base_idx >= static_cast<int>(query_size)) {
        return;
    }
    if (base_idx >= 0) {
        const int qidx = base_idx + lane;
        in_range       = (qidx < static_cast<int>(query_size));
        if (in_range) {
            my_qindex = static_cast<smallsize>(qidx);
            
            my_key    = query[qidx];
        }
    }

    // Tile work together until all done
    unsigned mask = tile.ballot(in_range);
    while (mask) {
        const int leader = __ffs(mask) - 1;

        // leader query key to the whole tile
        const key_type  probe_key   = tile.shfl(my_key, leader);
        const smallsize query_index = tile.shfl(my_qindex, leader);

            //find correct bucket for the probe_key
        const smallsize found_index = static_tree_search<key_type, node_size_log, cg_size_log, use_row_layout>(
            tile, probe_key, tree, metadata_out, shmem_tree, cached_entries_count);

       
        if (found_index < num_buckets_with_of) {
            uint8_t* __restrict__ curr_node = node_buf + static_cast<size_t>(node_stride) * found_index;

        
            smallsize currnodesize = 0;
            key_type  currnodeMax  = 0;

            if (perform_range_query_lookup) {
                //implemented but not tested
                // process_lookups_tile_rq<key_type>(currnodeMax, found_index, probe_key,
                //     launch_params, curr_node, currnodesize, query_index, tile);
            } else {
                process_lookups_tile_ldg<key_type>(currnodeMax, found_index, probe_key,
                    launch_params, curr_node, currnodesize, query_index, tile);
            }
        }

        // -Leader lane marks its work as done
        if (lane == leader) in_range = false;

        // Recompute remaining work
        mask = tile.ballot(in_range);
    }

#ifdef PRINT_LOOKUPS_END
    __syncthreads();
    print_set_nodes_and_links<key_type>(launch_params, global_thread_id);  // diagnostics only
#endif
}


     void lookup_static_tree(const key_type *keys, smallsize *result, size_t size, cudaStream_t stream)
    {
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

        // smallsize totalnodes = partition_count_with_overflow; // taking into account overflow node
        smallsize NUMTHREADS = (size > MAXBLOCKSIZE) ? MAXBLOCKSIZE : size;
        // printf("No Rays Reg LOOKUP KERNEL: size %d, NUMTHREADS %d \n", size, NUMTHREADS);
        bool group_lookup_kernel = false;

        ///----> TILED VERSION ----lookup_kernel_tile_static_tree<key_type, node_size_log_static_tree, cg_size_log_static_tree, false><<<SDIV(size, NUMTHREADS / DIV), NUMTHREADS / DIV, 0, stream>>>(launch_params_buffer.ptr(), group_lookup_kernel);
        lookup_kernel_tile_static_tree<key_type, node_size_log_static_tree, cg_size_log_static_tree, false><<<SDIV(size, NUMTHREADS / DIV), NUMTHREADS / DIV, 0, stream>>>(launch_params_buffer.ptr(), group_lookup_kernel);

        // lookup_kernel_tile_loop_sharedMem<key_type><<<SDIV(size, NUMTHREADS),  NUMTHREADS, 0, stream>>>(launch_params_buffer.ptr(), group_lookup_kernel);
        // lookup_kernel_tile_loop_SM<key_type><<<SDIV(size, NUMTHREADS/4),  NUMTHREADS/4, 0, stream>>>(launch_params_buffer.ptr(), group_lookup_kernel);

        
    }


