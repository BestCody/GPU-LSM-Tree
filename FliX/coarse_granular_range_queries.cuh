// file: ordered_range_lookup.cuh
#ifndef COARSE_GRANULAR_RANGE_QUERIES_CUH
#define COARSE_GRANULAR_RANGE_QUERIES_CUH

#include "coarse_granular_lookups_tile_bulk.cuh"


template <typename key_type>
DEVICEQUALIFIER void process_lookup_tile_bulk_ordered_dup_rq(
    key_type bucket_min,
    key_type bucket_max,
    int minindex,
  //  int maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const key_type *__restrict__ lowest_list,
    const key_type *__restrict__ highest_list,
    smallsize query_size,
    smallsize *__restrict__ results)
{
    void *allocation_buffer = launch_params->allocation_buffer;
    const smallsize allocation_count = launch_params->allocation_buffer_count;
    const smallsize node_stride = launch_params->node_stride;
    const smallsize node_size = launch_params->node_size;

    const smallsize lastpos_off = get_lastposition_bytes<key_type>(node_size);
    const smallsize lane = tile.thread_rank();
    const smallsize tile_id =
        blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    void *base_node = starting_node;

    for (int qi = minindex; qi < static_cast<int>(query_size); ++qi)
    {
        const key_type qlow = lowest_list[qi];
        const key_type qhigh = highest_list[qi];

        if (qlow > bucket_max)
            break;

        if (qhigh < bucket_min)
            continue;

        const key_type search_key = (qlow > bucket_min) ? qlow : bucket_min;
        if (lane == 0)
        {
            key_type base_max = cg::extract<key_type>(base_node, 0);
            while (base_max < search_key)
            {
                const smallsize next_ptr = cg::extract<smallsize>(base_node, lastpos_off);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_count)
                {
                    ERROR_INSERTS(
                        "ERROR: invalid link in node chain during tile bulk ordered range lookup",
                        search_key, lane, tile_id);
                }

                base_node = static_cast<uint8_t *>(allocation_buffer) +
                            static_cast<size_t>(next_ptr - 1) * node_stride;
                base_max = cg::extract<key_type>(base_node, 0);
            }
        }

        uintptr_t p = reinterpret_cast<uintptr_t>(base_node);
        p = tile.shfl(p, 0);
        void *scan_node = reinterpret_cast<void *>(p);

        smallsize query_sum = 0;
        while (true)
        {
            key_type curr_max = key_type(0);
            key_type node_min = key_type(0);
            smallsize curr_size = 0;
            smallsize next_ptr = 0;

            if (lane == 0)
            {
                curr_max = cg::extract<key_type>(scan_node, 0);
                curr_size = cg::extract<smallsize>(scan_node, sizeof(key_type));
                next_ptr = cg::extract<smallsize>(scan_node, lastpos_off);
                node_min = (curr_size > 0)
                               ? extract_key_node<key_type>(scan_node, 1)
                               : curr_max;
            }

            curr_max = tile.shfl(curr_max, 0);
            node_min = tile.shfl(node_min, 0);
            curr_size = tile.shfl(curr_size, 0);
            next_ptr = tile.shfl(next_ptr, 0);

            if (qhigh < node_min)
                break;

            key_type my_key = key_type(0);
            smallsize my_offset = 0;
            const bool lane_live = (lane < curr_size);

            if (lane_live)
            {
                my_key = extract_key_node<key_type>(scan_node, lane + 1);
                my_offset = extract_offset_node<key_type>(scan_node, lane + 1);
            }

            smallsize partial_sum = 0;
            if (lane_live && my_key >= qlow && my_key <= qhigh)
                partial_sum = my_offset;

#pragma unroll
            for (int delta = tile.size() / 2; delta > 0; delta >>= 1)
                partial_sum += tile.shfl_down(partial_sum, delta);

            if (lane == 0)
                query_sum += partial_sum;

            if (curr_max >= bucket_max || curr_max >= qhigh)
                break;

            if (lane == 0)
            {
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_count)
                {
                    ERROR_INSERTS(
                        "ERROR: invalid link in node chain during tile bulk ordered range lookup",
                        qhigh, lane, tile_id);
                }

                scan_node = static_cast<uint8_t *>(allocation_buffer) +
                            static_cast<size_t>(next_ptr - 1) * node_stride;
            }

            p = reinterpret_cast<uintptr_t>(scan_node);
            p = tile.shfl(p, 0);
            scan_node = reinterpret_cast<void *>(p);
            tile.sync();
        }

        if (lane == 0 && query_sum != 0)
            atomicAdd(results + qi, query_sum);

        tile.sync();
    }
}

template <typename key_type>
GLOBALQUALIFIER void lookup_kernel_tile_ordered_rq(
    updatable_cg_params *__restrict__ launch_params,
    const key_type *__restrict__ lowest_list,   // sorted by lower bound
    const key_type *__restrict__ highest_list,  // paired with lowest_list
    const smallsize *__restrict__ bucket_offset_sums,
    smallsize *__restrict__ results,            // must be zero-initialized
    smallsize query_size)
{
    (void)bucket_offset_sums;

    const key_type *__restrict__ maxbuf =
        static_cast<const key_type *>(launch_params->maxvalues);
    const smallsize partition_count_with_overflow =
        launch_params->partition_count_with_overflow;

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile =
        coop_g::tiled_partition<TILE_SIZE>(block);

    const int tiles_per_block = blockDim.x / tile.size();
    const int tile_id = blockIdx.x * tiles_per_block + tile.meta_group_rank();

    if (tile_id >= partition_count_with_overflow || query_size == 0)
        return;

    const key_type maxkey = maxbuf[tile_id];
    const key_type minkey =
        (tile_id > 0)
            ? (maxbuf[tile_id - 1] + static_cast<key_type>(1))
            : static_cast<key_type>(1);

    const int minindex = 0;

    uint8_t *__restrict__ curr_node =
        static_cast<uint8_t *>(launch_params->ordered_node_pairs) +
        static_cast<size_t>(launch_params->node_stride) *
            static_cast<size_t>(tile_id);

    process_lookup_tile_bulk_ordered_dup_rq<key_type>(
        /*bucket_min*/   minkey,
        /*bucket_max*/   maxkey,
        /*minindex*/     minindex,
        // /-*maxindex*-/ maxindex,
                         launch_params,
        /*curr_node*/    curr_node,
        /*tile*/         tile,
        /*lowest_list*/  lowest_list,
        /*highest_list*/ highest_list,
        /*query_size*/   query_size,
        /*writeback*/    results);
}

template <typename key_type>
DEVICEQUALIFIER int find_range_bucket_from_maxvalues(
    const key_type *__restrict__ maxbuf,
    key_type key,
    smallsize partition_count_with_overflow)
{
    if (partition_count_with_overflow == 0)
        return -1;

    int lo = 0;
    int hi = static_cast<int>(partition_count_with_overflow);

    while (lo < hi)
    {
        const int mid = lo + ((hi - lo) >> 1);
        if (maxbuf[mid] >= key)
            hi = mid;
        else
            lo = mid + 1;
    }

    if (lo >= static_cast<int>(partition_count_with_overflow))
        return static_cast<int>(partition_count_with_overflow) - 1;

    return lo;
}

template <typename key_type>
DEVICEQUALIFIER void *get_bucket_head_node(
    updatable_cg_params *__restrict__ launch_params,
    smallsize bucket_index)
{
    return static_cast<uint8_t *>(launch_params->ordered_node_pairs) +
           static_cast<size_t>(bucket_index) *
               static_cast<size_t>(launch_params->node_stride);
}

template <typename key_type>
DEVICEQUALIFIER void *get_next_range_node(
    updatable_cg_params *__restrict__ launch_params,
    void *curr_node,
    smallsize lastpos_off,
    smallsize query_index)
{
    const smallsize next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
    if (next_ptr == 0)
        return nullptr;

    if ((next_ptr - 1) >= launch_params->allocation_buffer_count)
    {
        ERROR_INSERTS(
            "ERROR: invalid link in node chain during single-thread range lookup",
            query_index, next_ptr, launch_params->allocation_buffer_count);
        return nullptr;
    }

    return static_cast<uint8_t *>(launch_params->allocation_buffer) +
           (static_cast<size_t>(next_ptr) - 1) *
               static_cast<size_t>(launch_params->node_stride);
}

template <typename key_type>
DEVICEQUALIFIER unsigned long long sum_offsets_in_bucket_range_single_thread(
    updatable_cg_params *__restrict__ launch_params,
    smallsize bucket_index,
    key_type lower,
    key_type upper,
    smallsize query_index)
{
    const smallsize node_size = launch_params->node_size;
    const smallsize lastpos_off = get_lastposition_bytes<key_type>(node_size);

    void *curr_node = get_bucket_head_node<key_type>(launch_params, bucket_index);

    while (curr_node != nullptr)
    {
        const key_type curr_max = cg::extract<key_type>(curr_node, 0);
        if (curr_max >= lower)
            break;

        curr_node = get_next_range_node<key_type>(
            launch_params, curr_node, lastpos_off, query_index);
    }

    unsigned long long sum = 0;

    while (curr_node != nullptr)
    {
        const key_type curr_max = cg::extract<key_type>(curr_node, 0);
        const smallsize curr_size =
            cg::extract<smallsize>(curr_node, sizeof(key_type));

        for (smallsize slot = 1; slot <= curr_size; ++slot)
        {
            const key_type key = extract_key_node<key_type>(curr_node, slot);

            if (key > upper)
                return sum;

            if (key >= lower)
            {
                sum += static_cast<unsigned long long>(
                    extract_offset_node<key_type>(curr_node, slot));
            }
        }

        if (curr_max >= upper)
            break;

        curr_node = get_next_range_node<key_type>(
            launch_params, curr_node, lastpos_off, query_index);
    }

    return sum;
}

template <typename key_type>
GLOBALQUALIFIER void lookup_kernel_tile_ordered_rq_sums(
    updatable_cg_params *__restrict__ launch_params,
    const key_type *__restrict__ lowest_list,
    const key_type *__restrict__ highest_list,
    const smallsize *__restrict__ bucket_offset_sums,
    smallsize *__restrict__ results,
    smallsize query_size)
{
    const smallsize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= query_size)
        return;

    if (launch_params == nullptr || bucket_offset_sums == nullptr ||
        lowest_list == nullptr || highest_list == nullptr || results == nullptr)
        return;

    const key_type lower = lowest_list[tid];
    const key_type upper = highest_list[tid];

    if (lower > upper)
    {
        results[tid] = 0;
        return;
    }

    const key_type *__restrict__ maxbuf =
        static_cast<const key_type *>(launch_params->maxvalues);
    const smallsize partition_count_with_overflow =
        launch_params->partition_count_with_overflow;

    const int lower_bucket = find_range_bucket_from_maxvalues<key_type>(
        maxbuf, lower, partition_count_with_overflow);
    const int upper_bucket = find_range_bucket_from_maxvalues<key_type>(
        maxbuf, upper, partition_count_with_overflow);

    if (lower_bucket < 0 || upper_bucket < 0)
    {
        results[tid] = 0;
        return;
    }

    unsigned long long sum = 0;

    if (lower_bucket == upper_bucket)
    {
        sum = sum_offsets_in_bucket_range_single_thread<key_type>(
            launch_params,
            static_cast<smallsize>(lower_bucket),
            lower,
            upper,
            tid);
    }
    else
    {
        sum += sum_offsets_in_bucket_range_single_thread<key_type>(
            launch_params,
            static_cast<smallsize>(lower_bucket),
            lower,
            maxbuf[lower_bucket],
            tid);

        for (int bucket = lower_bucket + 1; bucket < upper_bucket; ++bucket)
        {
            sum += static_cast<unsigned long long>(bucket_offset_sums[bucket]);
        }

        const key_type last_bucket_min =
            (upper_bucket > 0)
                ? (maxbuf[upper_bucket - 1] + static_cast<key_type>(1))
                : static_cast<key_type>(1);

        sum += sum_offsets_in_bucket_range_single_thread<key_type>(
            launch_params,
            static_cast<smallsize>(upper_bucket),
            last_bucket_min,
            upper,
            tid);
    }

    results[tid] = static_cast<smallsize>(sum);
}



// path: gpu_range_lookup.cuh

template <typename Tile>
DEVICEQUALIFIER unsigned long long tile_reduce_sum_ull(Tile tile, unsigned long long value)
{
    for (int delta = tile.size() / 2; delta > 0; delta >>= 1)
    {
        value += tile.shfl_down(value, delta);
    }
    return value;
}

template <typename key_type>
GLOBALQUALIFIER void precompute_bucket_offset_sums_kernel(
    updatable_cg_params *__restrict__ launch_params,
    smallsize *__restrict__ bucket_offset_sums)
{
    if (launch_params == nullptr || bucket_offset_sums == nullptr)
        return;

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    const int tiles_per_block = blockDim.x / tile.size();
    const int tile_id = blockIdx.x * tiles_per_block + tile.meta_group_rank();
    const smallsize lane = tile.thread_rank();

    if (tile_id >= launch_params->partition_count_with_overflow)
        return;

    const smallsize node_stride = launch_params->node_stride;
    const smallsize node_size = launch_params->node_size;
    const smallsize allocation_count = launch_params->allocation_buffer_count;
    const smallsize lastpos_off = get_lastposition_bytes<key_type>(node_size);

    void *curr_node =
        static_cast<uint8_t *>(launch_params->ordered_node_pairs) +
        static_cast<size_t>(tile_id) * static_cast<size_t>(node_stride);

    unsigned long long bucket_sum = 0;

    while (true)
    {
        const smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        unsigned long long lane_sum = 0;
        for (smallsize slot = static_cast<smallsize>(lane) + 1; slot <= curr_size; slot += tile.size())
        {
            lane_sum += static_cast<unsigned long long>(
                extract_offset_node<key_type>(curr_node, slot));
        }

        const unsigned long long node_sum = tile_reduce_sum_ull(tile, lane_sum);

        if (lane == 0)
        {
            bucket_sum += node_sum;
        }

        tile.sync();

        smallsize next_ptr = 0;
        if (lane == 0)
        {
            next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
        }
        next_ptr = tile.shfl(next_ptr, 0);

        if (next_ptr == 0)
            break;

        if (lane == 0)
        {
            if ((next_ptr - 1) >= allocation_count)
            {
                printf("ERROR: invalid link in precompute_bucket_offset_sums_kernel, tile_id=%d, next_ptr=%u, allocation_count=%u\n",
                       tile_id,
                       static_cast<unsigned>(next_ptr),
                       static_cast<unsigned>(allocation_count));
                curr_node = nullptr;
            }
            else
            {
                curr_node =
                    static_cast<uint8_t *>(launch_params->allocation_buffer) +
                    (static_cast<size_t>(next_ptr) - 1) * static_cast<size_t>(node_stride);
            }
        }

        uintptr_t p = reinterpret_cast<uintptr_t>(curr_node);
        p = tile.shfl(p, 0);
        curr_node = reinterpret_cast<void *>(p);

        if (curr_node == nullptr)
            break;

        tile.sync();
    }

    if (lane == 0)
    {
        bucket_offset_sums[tile_id] = static_cast<smallsize>(bucket_sum);
    }
}

/*
template <typename key_type>
void precompute_bucket_offset_sums(cudaStream_t stream, smallsize partition_count_with_overflow)
{
    const smallsize total_threads_required = partition_count_with_overflow * TILE_SIZE;
    const smallsize max_blocks_required = SDIV(total_threads_required, MAXBLOCKSIZE / DIV_FACTOR);

    precompute_bucket_offset_sums_kernel<key_type>
        <<<max_blocks_required, MAXBLOCKSIZE / DIV_FACTOR, 0, stream>>>(
            launch_params_buffer.ptr());

    CUERR;
}

void range_lookup_sum( launch_params_buffer_type launch_params_buffer,
                      smallsize partition_count_with_overflow, cudaStream_t stream)
{
    const smallsize total_threads_required = partition_count_with_overflow * TILE_SIZE;
    const smallsize max_blocks_required = SDIV(total_threads_required, MAXBLOCKSIZE / DIV_FACTOR);

    precompute_bucket_offset_sums<key_type>(stream);

    printf("RANGE QUERIES LOOKUP SUM TILE ORDERED: div is:%d, max_blocks_required=%d and MAXBLOCKSIZE/DIV %d\n",
           DIV_FACTOR,
           max_blocks_required,
           MAXBLOCKSIZE / DIV_FACTOR);

    lookup_kernel_tile_ordered_rq<key_type>
        <<<max_blocks_required, MAXBLOCKSIZE / DIV_FACTOR, 0, stream>>>(
            launch_params_buffer.ptr(),
            lower,
            upper,
            result,
            size);

    CUERR;
}
*/

#endif
