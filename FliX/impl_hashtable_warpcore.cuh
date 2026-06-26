// =============================================================================
// File: impl_hashtable_warpcore.cuh
// Author: Justus Henneberg
// Description: Implements impl_hashtable_warpcore     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef IMPL_HASHTABLE_WARPCORE_CUH
#define IMPL_HASHTABLE_WARPCORE_CUH

#include <optional>

#include "../warpcore/warpcore.cuh"
#include "definitions.cuh"


template<class table_type>
GLOBALQUALIFIER
void warpcore_build_kernel(table_type hash_table, const typename table_type::key_type* keys, const smallsize num_in) {
    const size_t tid = global_thread_id();
    const smallsize gid = tid / table_type::cg_size();
    const auto group = cooperative_groups::tiled_partition<table_type::cg_size()>(cooperative_groups::this_thread_block());
    if (gid >= num_in) return;

    hash_table.insert(keys[gid], gid, group);
}


template<class table_type>
GLOBALQUALIFIER
void warpcore_insert_kernel(table_type hash_table, const typename table_type::key_type* keys, const smallsize* offsets, const smallsize num_in) {
    const size_t tid = global_thread_id();
    const smallsize gid = tid / table_type::cg_size();
    const auto group = cooperative_groups::tiled_partition<table_type::cg_size()>(cooperative_groups::this_thread_block());
    if (gid >= num_in) return;

    hash_table.insert(keys[gid], offsets[gid], group);
}


template<class table_type>
GLOBALQUALIFIER
void warpcore_delete_kernel(table_type hash_table, const typename table_type::key_type* keys, const smallsize num_in) {
    const size_t tid = global_thread_id();
    const smallsize gid = tid / table_type::cg_size();
    const auto group = cooperative_groups::tiled_partition<table_type::cg_size()>(cooperative_groups::this_thread_block());
    if (gid >= num_in) return;

    // note: this function is not part of warpcore proper, it was added later on
    hash_table.erase_all(keys[gid], group);
}


template<class table_type>
GLOBALQUALIFIER
void warpcore_lookup_kernel(table_type hash_table, const typename table_type::key_type* keys, smallsize* result, const smallsize num_in) {
    const size_t tid = global_thread_id();
    const smallsize gid = tid / table_type::cg_size();
    const auto group = cooperative_groups::tiled_partition<table_type::cg_size()>(cooperative_groups::this_thread_block());
    if (gid >= num_in) return;

    typename table_type::key_type key = keys[gid];
    smallsize retrieved_index;

    // note: this function is not part of warpcore proper, it was added later on
    const auto status = hash_table.retrieve(key, retrieved_index, group);
    if (group.thread_rank() == 0) {
        if (status.has_key_not_found() || status.has_probing_length_exceeded()) {
            result[gid] = not_found;
        } else {
            result[gid] = retrieved_index;
        }
    }
}


template<class table_type>
GLOBALQUALIFIER
void warpcore_multi_lookup_kernel(table_type hash_table, const typename table_type::key_type* keys, smallsize* result, const smallsize num_in) {
    const size_t tid = global_thread_id();
    const smallsize gid = tid / table_type::cg_size();
    const auto group = cooperative_groups::tiled_partition<table_type::cg_size()>(cooperative_groups::this_thread_block());
    if (gid >= num_in) return;

    typename table_type::key_type key = keys[gid];
    typename table_type::index_type num_out; // ignored
    smallsize agg = 0;

    hash_table.for_each(
        [=, &agg] DEVICEQUALIFIER (const typename table_type::key_type, const typename table_type::value_type& retrieved_index, const typename table_type::index_type) {
            agg += retrieved_index;
        },
        key,
        num_out,
        group
    );

    for (size_t i = table_type::cg_size() >> 1u; i > 0; i >>= 1u) {
        agg += group.shfl_down(agg, i);
    }
    if (group.thread_rank() == 0) {
        result[gid] = agg;
    }
}


template <typename key_type_, uint8_t max_load_percent = 80>
class hashtable_warpcore {
public:
    using key_type = key_type_;

private:
    using table_type = warpcore::MultiValueHashTable<key_type, smallsize, key_type(-2), key_type(-1)>;

    std::optional<table_type> wrapped_table;

public:
    static constexpr const char* name = "hashtable_warpcore";
    static constexpr operation_support can_lookup = operation_support::async;
    static constexpr operation_support can_multi_lookup = operation_support::async;
    static constexpr operation_support can_range_lookup = operation_support::none;
    static constexpr operation_support can_update = operation_support::async;

    static std::string short_description() {
        std::string desc = "hashtable_warpcore";
        return desc;
    }

    static parameters_type parameters() {
        return {
                {"max_load_percent", std::to_string(max_load_percent)}
        };
    }

    static size_t estimate_build_bytes(size_t size) {
        // hash table size will always be the next higher prime, use factor 1.2 as an approximation
        // https://projecteuclid.org/journals/proceedings-of-the-japan-academy-series-a-mathematical-sciences/volume-28/issue-4/On-the-interval-containing-at-least-one-prime-number/10.3792/pja/1195570997.full
        return static_cast<size_t>(1.2 * size * 100.0 / max_load_percent) * (sizeof(key_type) + sizeof(smallsize));
    }

    size_t gpu_resident_bytes() {
        return wrapped_table.value().bytes_total();
    }

    void build(const key_type* keys, size_t size, size_t max_size, size_t available_memory_bytes, double* build_time_ms, size_t* build_bytes) {
        if (estimate_build_bytes(max_size) > available_memory_bytes) throw std::runtime_error("insufficient memory");

        const size_t capacity = static_cast<size_t>(max_size * 100.0 / max_load_percent);
        wrapped_table.emplace(capacity);

        cuda_timer timer(0);
        timer.start();

        warpcore_build_kernel<<<SDIV(size * table_type::cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE>>>(
            wrapped_table.value(),
            keys,
            size
        );

        timer.stop();
        if (build_time_ms) *build_time_ms += timer.time_ms();
        if (build_bytes) *build_bytes += gpu_resident_bytes();

        cudaDeviceSynchronize(); C2EX
    }

    void lookup(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {

        warpcore_lookup_kernel<<<SDIV(size * table_type::cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
            wrapped_table.value(),
            keys,
            result,
            size
        );
    }

    void multi_lookup_sum(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {

        warpcore_multi_lookup_kernel<<<SDIV(size * table_type::cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
            wrapped_table.value(),
            keys,
            result,
            size
        );
    }

    void range_lookup_sum(const key_type* lower, const key_type* upper, smallsize* result, size_t size, cudaStream_t stream) {}

    void destroy() {
        wrapped_table.reset();
    }

    void insert(const key_type* update_list, const smallsize* offsets, size_t size, cudaStream_t stream) {

        warpcore_insert_kernel<<<SDIV(size * table_type::cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
            wrapped_table.value(),
            update_list,
            offsets,
            size
        );
    }

    void remove(const key_type* update_list, size_t size, cudaStream_t stream) {

        warpcore_delete_kernel<<<SDIV(size * table_type::cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
            wrapped_table.value(),
            update_list,
            size
        );
    }

     void lookups_successor(const key_type* keys, key_type* result, size_t size, cudaStream_t stream) {
        return;
    }
};


#endif
