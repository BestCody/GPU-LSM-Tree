// =============================================================================
// File: impl_tree_awad.cuh
// Author: Justus Henneberg
// Description: Implements impl_tree_awad     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef IMPL_TREE_AWAD_CUH
#define IMPL_TREE_AWAD_CUH

#include "gpu_btree.h"

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <limits>

using GpuBTree::gpu_blink_tree;

// for nvtx
struct nvtx_tree_awad_domain{ static constexpr char const* name{"tree_awad"}; };


template <typename key_type>
GLOBALQUALIFIER void tree_awad_make_upper_exclusive(const key_type* upper,
                                                    key_type* upper_exclusive,
                                                    std::size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  auto value = upper[i];
  upper_exclusive[i] = value == std::numeric_limits<key_type>::max()
                           ? value
                           : static_cast<key_type>(value + 1);
}

template <typename pair_type>
GLOBALQUALIFIER void tree_awad_sum_pairs_kernel(const pair_type* pairs,
                                                const smallsize* counts,
                                                smallsize* result,
                                                std::size_t stride,
                                                std::size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  smallsize sum = 0;
  const auto count = counts[i];
  const auto base = i * stride;
  for (smallsize j = 0; j < count; ++j) {
    sum += pairs[base + j].second;
  }
  result[i] = sum;
}

template <typename key_type_, bool bulk_load_full = false>
class tree_awad {
    static_assert(std::is_same<key_type_, key32>::value, "key must be 32 bits wide");

public:
    using key_type = key_type_;

private:
    using tree_type = gpu_blink_tree<key_type, smallsize, 16>;
    using pair_type = typename tree_type::pair_type;
    std::optional<tree_type> wrapped_tree;

public:
    static constexpr const char* name = "tree_awad";
    static constexpr operation_support can_lookup = operation_support::async;
    static constexpr operation_support can_lower_bound_rank = operation_support::none;
    static constexpr operation_support can_multi_lookup = operation_support::none;
    static constexpr operation_support can_range_lookup = operation_support::async;
    static constexpr operation_support can_insert = operation_support::async;
    static constexpr operation_support can_delete = operation_support::async;
    static constexpr operation_support can_update = operation_support::async;
    static constexpr operation_support can_successor = operation_support::async;

    static std::string short_description() {
        std::string desc = "tree_awad";
        if constexpr (bulk_load_full) {
            desc += "_full";
        } else {
            desc += "_half";
        }
        return desc;
    }

    static parameters_type parameters() {
        return {
                {"node_size", "16"},
                {"bulk_load", bulk_load_full ? "full" : "half"}
        };
    }

    static size_t estimate_build_bytes(size_t size) {
        return sizeof(key_type) * size + sizeof(smallsize) * size +
               find_pair_sort_buffer_size<key_type, smallsize>(size) +
               sizeof(pair_type) * size;
    }

    size_t gpu_resident_bytes() {
        if (!wrapped_tree) return 0;
        return static_cast<size_t>(wrapped_tree.value().get_num_tree_node()) *
               tree_type::branching_factor * sizeof(pair_type);
    }

    void build(const key_type* keys, size_t size, size_t max_size, size_t available_memory_bytes, double* build_time_ms, size_t* build_bytes) {
        (void)max_size;
        (void)available_memory_bytes;
        cuda_buffer<key_type> sorted_keys_buffer;
        cuda_buffer<smallsize> sorted_offsets_buffer;

        {
            cuda_buffer<uint8_t> temp_buffer;
            cuda_buffer<smallsize> offsets_buffer;
            sorted_keys_buffer.alloc(size);
            offsets_buffer.alloc(size);
            sorted_offsets_buffer.alloc(size);
            init_offsets(offsets_buffer, size, build_time_ms);

            cudaDeviceSynchronize(); C2EX

            size_t temp_storage_bytes = find_pair_sort_buffer_size<key_type, smallsize>(size);
            temp_buffer.alloc(temp_storage_bytes);
            timed_pair_sort(
                temp_buffer.raw_ptr, temp_storage_bytes,
                keys, sorted_keys_buffer.ptr(), offsets_buffer.ptr(), sorted_offsets_buffer.ptr(), size, build_time_ms);

            if (build_bytes) *build_bytes += sorted_keys_buffer.size_in_bytes() + sorted_offsets_buffer.size_in_bytes() + temp_buffer.size_in_bytes() + offsets_buffer.size_in_bytes();

            cudaDeviceSynchronize(); C2EX
        }

        {
            scoped_cuda_timer timer(0, build_time_ms);
            auto built_tree = tree_type(sorted_keys_buffer.ptr(), sorted_offsets_buffer.ptr(), static_cast<typename tree_type::size_type>(size), true, cudaStream_t{});
            wrapped_tree.emplace(built_tree);
        }
        if (build_bytes) *build_bytes += gpu_resident_bytes();

        cudaDeviceSynchronize(); C2EX
    }

    void lookup(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        nvtx3::scoped_range_in<nvtx_tree_awad_domain> launch{"launch"};
        (void)launch;
        wrapped_tree.value().find(keys, result, static_cast<typename tree_type::size_type>(size), stream, false);
    }

    void multi_lookup_sum(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        (void)keys;
        (void)result;
        (void)size;
        (void)stream;
    }

    void range_lookup_sum(const key_type* lower, const key_type* upper, smallsize* result, size_t size, cudaStream_t stream) {
        if (size == 0) return;

        auto upper_exclusive = cuda_buffer<key_type>{};
        upper_exclusive.alloc(size);
        auto block_size = 512;
        auto grid_size = SDIV(size, block_size);
        tree_awad_make_upper_exclusive<<<grid_size, block_size, 0, stream>>>(upper, upper_exclusive.ptr(), size);
        C2EX

        auto counts_buffer = cuda_buffer<smallsize>{};
        counts_buffer.alloc(size);
        wrapped_tree.value().range_query(lower, upper_exclusive.ptr(), nullptr, counts_buffer.ptr(), 0, static_cast<typename tree_type::size_type>(size), stream, false);
        cudaStreamSynchronize(stream); C2EX

        auto counts = counts_buffer.download(size);
        auto max_count = smallsize{0};
        for (auto count : counts) {
            max_count = std::max(max_count, count);
        }

        if (max_count == 0) {
            cudaMemsetAsync(result, 0, size * sizeof(smallsize), stream); C2EX
            return;
        }

        auto pair_buffer = cuda_buffer<pair_type>{};
        pair_buffer.alloc(static_cast<size_t>(size) * max_count);
        wrapped_tree.value().range_query(lower, upper_exclusive.ptr(), pair_buffer.ptr(), counts_buffer.ptr(), max_count, static_cast<typename tree_type::size_type>(size), stream, false);
        tree_awad_sum_pairs_kernel<<<grid_size, block_size, 0, stream>>>(pair_buffer.ptr(), counts_buffer.ptr(), result, max_count, size);
        C2EX
    }

    void destroy() {
        wrapped_tree.reset();
    }

    void insert(const key_type* update_list, const smallsize* offsets, size_t size, cudaStream_t stream) {
        wrapped_tree.value().insert(update_list, offsets, static_cast<typename tree_type::size_type>(size), stream);
    }

    void remove(const key_type* update_list, size_t size, cudaStream_t stream) {
        wrapped_tree.value().erase(update_list, static_cast<typename tree_type::size_type>(size), stream);
    }

    void lookups_successor(const key_type* keys, key_type* result, size_t size, cudaStream_t stream) {
        nvtx3::scoped_range_in<nvtx_tree_awad_domain> launch{"launch"};
        (void)launch;
        wrapped_tree.value().successor(keys, result, static_cast<typename tree_type::size_type>(size), stream, false);
    }
};

#endif
