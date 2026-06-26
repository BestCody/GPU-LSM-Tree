// =============================================================================
// File: impl_hashtable_slab.cuh
// Author: Justus Henneberg
// Description: Implements impl_hashtable_slab     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef IMPL_HASHTABLE_SLAB_CUH
#define IMPL_HASHTABLE_SLAB_CUH

#include "definitions.cuh"
#include "../slabhash/src/slab_hash.cuh"


template <typename key_type, typename table_context_type>
__global__
void hashtable_slab_search_kernel(
    const key_type* queries,
    smallsize* results,
    smallsize num_queries,
    table_context_type slab_hash
) {
    smallsize tid = threadIdx.x + blockIdx.x * blockDim.x;
    smallsize lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_queries) {
        return;
    }

    key_type query = 0;
    smallsize result = not_found;
    smallsize bucket = 0;
    bool to_search = false;
    if (tid < num_queries) {
        query = queries[tid];
        bucket = slab_hash.computeBucket(query);
        to_search = true;
    }

    // todo make this a template parameter
    constexpr bool use_bulk_search = false;
    // use bulk search if every thread has a query
    if (use_bulk_search && (tid - lane_id + 31) < num_queries) {
        slab_hash.searchKeyBulk(lane_id, query, result, bucket);
    } else {
        slab_hash.searchKey(to_search, lane_id, query, result, bucket);
    }

    if (tid < num_queries) {
        results[tid] = result;
    }
}


template <typename key_type, typename table_context_type>
__global__
void hashtable_slab_multi_search_kernel(
    const key_type* queries,
    smallsize* results,
    smallsize num_queries,
    table_context_type slab_hash
) {
    smallsize tid = threadIdx.x + blockIdx.x * blockDim.x;
    smallsize lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_queries) {
        return;
    }

    key_type query = 0;
    smallsize result = 0;
    smallsize bucket = 0;
    bool to_search = false;
    if (tid < num_queries) {
        query = queries[tid];
        bucket = slab_hash.computeBucket(query);
        to_search = true;
    }

    slab_hash.searchKeySum(to_search, lane_id, query, result, bucket);

    if (tid < num_queries) {
        results[tid] = result;
    }
}


template <typename key_type, typename table_context_type, typename allocator_context_type>
__global__
void hashtable_slab_insert_kernel(
    const key_type* keys,
    const smallsize* values,
    smallsize num_keys,
    table_context_type slab_hash
) {
    smallsize tid = threadIdx.x + blockIdx.x * blockDim.x;
    smallsize lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_keys) {
        return;
    }

    allocator_context_type local_allocator_ctx(slab_hash.getAllocatorContext());
    local_allocator_ctx.initAllocator(tid, lane_id);

    key_type key = 0;
    smallsize value = 0;
    smallsize bucket = 0;
    bool to_insert = false;

    if (tid < num_keys) {
        key = keys[tid];
        value = values ? values[tid] : tid;
        bucket = slab_hash.computeBucket(key);
        to_insert = true;
    }

    // todo make this a template parameter
    constexpr bool unique_keys = false;
    if constexpr (unique_keys) {
        slab_hash.insertPairUnique(to_insert, lane_id, key, value, bucket, local_allocator_ctx);
    } else {
        slab_hash.insertPair(to_insert, lane_id, key, value, bucket, local_allocator_ctx);
    }
}


template <typename key_type, typename table_context_type>
__global__
void hashtable_slab_delete_kernel(
    const key_type* deletes,
    smallsize num_deletes,
    table_context_type slab_hash
) {
    smallsize tid = threadIdx.x + blockIdx.x * blockDim.x;
    smallsize lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_deletes) {
        return;
    }

    key_type key = 0;
    smallsize bucket = 0;
    bool to_delete = false;

    if (tid < num_deletes) {
        key = deletes[tid];
        bucket = slab_hash.computeBucket(key);
        to_delete = true;
    }

    slab_hash.deleteKey(to_delete, lane_id, key, bucket);
}


template <typename table_context_type>
__global__
void hashtable_slab_count_kernel(
    bigsize* allocated_slab_count_ptr,
    smallsize num_buckets,
    table_context_type slab_hash
) {
    smallsize tid = threadIdx.x + blockIdx.x * blockDim.x;
    smallsize lane_id = threadIdx.x & 0x1F;
    smallsize wid = tid >> 5;
    if (wid >= num_buckets) {
        return;
    }

    slab_hash.getAllocatorContext().initAllocator(tid, lane_id);

    smallsize slabs_count = 1;

    uint32_t src_unit_data = *slab_hash.getPointerFromBucket(wid, lane_id);
    uint32_t next = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
    while (next != ConcurrentMapT<key32, smallsize>::EMPTY_INDEX_POINTER) {
        src_unit_data = *slab_hash.getPointerFromSlab(next, lane_id);
        next = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
        slabs_count++;
    }
    if (lane_id == 0) {
        atomicAdd(allocated_slab_count_ptr, (bigsize)slabs_count);
    }
}


template <typename key_type_, uint8_t initial_load_percent = 100>
class hashtable_slab {
    static_assert(std::is_same<key_type_, key32>::value, "key must be 32 bits wide");

public:
    using key_type = key_type_;

private:
    // value corresponds to GpuSlabHash::BLOCKSIZE_
    static constexpr uint32_t threads_per_block = 128;

    // todo these are hard-coded elsewhere!!!! this is a problem
    // taken from slab_hash_global.cuh
    static constexpr uint32_t log_num_mem_blocks = 8;
    static constexpr uint32_t max_num_super_blocks = 128;
    static constexpr uint32_t num_replicas = 1;

    using allocator_type = SlabAllocLight<log_num_mem_blocks, max_num_super_blocks, num_replicas>;
    using allocator_context_type = SlabAllocLightContext<log_num_mem_blocks, num_replicas>;
    using table_type = GpuSlabHash<key_type, smallsize, SlabHashTypeT::ConcurrentMap, allocator_type>;
    using table_context_type = GpuSlabHashContext<key_type, smallsize, SlabHashTypeT::ConcurrentMap, allocator_context_type>;

    std::optional<allocator_type> wrapped_allocator;
    std::optional<table_type> wrapped_table;

    size_t num_buckets = 0;

public:
    static constexpr const char* name = "hashtable_slab";
    static constexpr operation_support can_lookup = operation_support::async;
    static constexpr operation_support can_multi_lookup = operation_support::async;
    static constexpr operation_support can_range_lookup = operation_support::none;
    static constexpr operation_support can_update = operation_support::async;

    static std::string short_description() {
        return std::string("hashtable_slab");
    }

    static parameters_type parameters() {
        return {
            {"initial_load_percent", std::to_string(initial_load_percent)},
            {"threads_per_block", std::to_string(threads_per_block)},
            {"log_num_mem_blocks", std::to_string(log_num_mem_blocks)},
            {"max_num_super_blocks", std::to_string(max_num_super_blocks)},
            {"num_replicas", std::to_string(num_replicas)}
        };
    }

    size_t gpu_resident_bytes() {
        cuda_buffer<bigsize> allocated_slab_count_buffer;
        allocated_slab_count_buffer.alloc(1);
        allocated_slab_count_buffer.zero();
        hashtable_slab_count_kernel<<<SDIV(num_buckets * 32, threads_per_block), threads_per_block>>>(
            allocated_slab_count_buffer.ptr(),
            num_buckets,
            wrapped_table->gpu_context_);
        return 32 * sizeof(uint32_t) * allocated_slab_count_buffer.download_first_item();
    }

    void build(const key_type* keys, size_t size, size_t max_size, size_t available_memory_bytes, double* build_time_ms, size_t* build_bytes) {

        if (max_size > (size_t(1) << 28)) {
            throw std::runtime_error("size unsupported");
        }

        constexpr uint32_t device_id = 0;

        // for each key, we lose another slot for the value, and 1 per 16 for the next pointer
        // in total, we can store 15 keys in 32 slots (1 slab)
        num_buckets = 1 + size * 100 / (15 * initial_load_percent);
        size_t slots_per_superblock = allocator_context_type::MEM_BLOCK_SIZE_ * allocator_context_type::NUM_MEM_BLOCKS_PER_SUPER_BLOCK_;
        // since we do not know the distribution of keys across buckets, we should allocate an extra slab per bucket
        // however, the table itself pre-allocates one slab per bucket, so we can ignore this
        size_t required_key_capacity = max_size;
        // final number of additional slots is ceil(required_key_capacity / (slots_per_superblock * 15/32))
        // add one more superblock for performance gains
        size_t required_superblocks = 1 + (1 + required_key_capacity * 32 / (slots_per_superblock * 15));

        // SUPER_BLOCK_SIZE_ includes the per-block bitmaps
        size_t expected_size_bytes = (num_buckets + required_superblocks * allocator_context_type::SUPER_BLOCK_SIZE_) * sizeof(key_type);
        if (expected_size_bytes > available_memory_bytes) {
            throw std::runtime_error("not enough memory to build");
        }

        wrapped_allocator.emplace(required_superblocks);
        wrapped_table.emplace(num_buckets, &wrapped_allocator.value(), device_id);

        {
            scoped_cuda_timer timer(0, build_time_ms);
            hashtable_slab_insert_kernel<key_type, table_context_type, allocator_context_type><<<SDIV(size, threads_per_block), threads_per_block>>>(
                keys,
                nullptr,
                size,
                wrapped_table->gpu_context_);
        }
        if (build_bytes) *build_bytes += gpu_resident_bytes();
    }

    void destroy() {
        wrapped_allocator.reset();
        wrapped_table.reset();
    }

    void lookup(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        hashtable_slab_search_kernel<<<SDIV(size, threads_per_block), threads_per_block, 0, stream>>>(
            keys,
            result,
            size,
            wrapped_table->gpu_context_);
    }

    void multi_lookup_sum(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        hashtable_slab_multi_search_kernel<<<SDIV(size, threads_per_block), threads_per_block, 0, stream>>>(
            keys,
            result,
            size,
            wrapped_table->gpu_context_);
    }

    void range_lookup_sum(const key_type* lower, const key_type* upper, smallsize* result, size_t size, cudaStream_t stream) {}

    void insert(const key_type* update_list, const smallsize* positions, size_t size, cudaStream_t stream) {
        hashtable_slab_insert_kernel<key_type, table_context_type, allocator_context_type><<<SDIV(size, threads_per_block), threads_per_block, 0, stream>>>(
            update_list,
            positions,
            size,
            wrapped_table->gpu_context_);
    }

    void remove(const key_type* update_list, size_t size, cudaStream_t stream) {
        hashtable_slab_delete_kernel<<<SDIV(size, threads_per_block), threads_per_block, 0, stream>>>(
            update_list,
            size,
            wrapped_table->gpu_context_);
    }

     void lookups_successor(const key_type* keys, key_type* result, size_t size, cudaStream_t stream) {
        return;
    }
};

#endif
