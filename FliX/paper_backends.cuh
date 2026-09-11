#ifndef FLIX_PAPER_BACKENDS_CUH
#define FLIX_PAPER_BACKENDS_CUH

#include "benchmark_lookup.cuh"
#include "benchmark_inputs.cuh"
#include "benchmark_range.cuh"
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#if defined(GPULSMOPT)
#include "impl_gpulsmopt.cuh"
using selected_paper_backend = gpulsmopt<key32>;
constexpr const char *paper_backend_name = "GPULSMOpt";
constexpr bool paper_dynamic = true, paper_tombstones = true;
#elif defined(LSM_TREE)
#include "impl_lsm_tree.cuh"
using selected_paper_backend = lsm_tree_ashkiani<key32, PAPER_LSM_BATCH_LOG>;
constexpr const char *paper_backend_name = "LSMu";
constexpr bool paper_dynamic = true, paper_tombstones = true;
#elif defined(GPU_BTREE)
#include <nvtx3/nvtx3.hpp>
#include "impl_tree_awad.cuh"
using selected_paper_backend = tree_awad<key32, true>;
constexpr const char *paper_backend_name = "GPU B-tree";
constexpr bool paper_dynamic = true, paper_tombstones = false;
#elif defined(HASHTABLE_SLAB)
#include <nvtx3/nvtx3.hpp>
#include "impl_hashtable_slab.cuh"
using selected_paper_backend = hashtable_slab<key32>;
constexpr const char *paper_backend_name = "SlabHash";
constexpr bool paper_dynamic = true, paper_tombstones = false;
#elif defined(HASHTABLE_WARPCORE)
#include <nvtx3/nvtx3.hpp>
#include "impl_hashtable_warpcore.cuh"
using selected_paper_backend = hashtable_warpcore<key32, 80>;
constexpr const char *paper_backend_name = "WarpCore";
constexpr bool paper_dynamic = true, paper_tombstones = false;
#elif defined(SORTED_ARRAY)
#include <nvtx3/nvtx3.hpp>
#include "impl_binsearch.cuh"
using selected_paper_backend = sorted_array<key32>;
constexpr const char *paper_backend_name = "Sorted array";
constexpr bool paper_dynamic = true, paper_tombstones = false;
#else
#include <nvtx3/nvtx3.hpp>
#include "impl_rtx_index.cuh"
#include "impl_cg_rtx_index.cuh"
#include "impl_cg_rtx_index_updates.cuh"
using selected_paper_backend = cg_rtx_index_updates<key32, 0, 5, 50>;
constexpr const char *paper_backend_name = "FliX";
constexpr bool paper_dynamic = true, paper_tombstones = false;
#endif

template <typename Index>
constexpr bool paper_supports_ranges() {
    if constexpr (std::is_same_v<decltype(Index::can_range_lookup), const bool>)
        return Index::can_range_lookup;
    else
        return Index::can_range_lookup != operation_support::none;
}
constexpr bool paper_range = paper_supports_ranges<selected_paper_backend>();
static_assert(!paper_range ||
              flix_benchmark::enumerates_range_records<selected_paper_backend>::value,
              "Paper ranges must enumerate records before summing values");

template <typename Index>
void paper_build(Index &index, const key32 *keys, size_t size, size_t capacity,
                 size_t free_bytes) {
    if constexpr (!paper_dynamic)
        index.build(keys, size, nullptr, nullptr);
    else if constexpr (flix_benchmark::requires_ordered_lookup<Index>::value)
        index.build_bucket_layer_only(keys, size, capacity, free_bytes, nullptr, nullptr);
    else
        index.build(keys, size, capacity, free_bytes, nullptr, nullptr);
}

template <typename Index>
void paper_prepare_updates(key32 *keys, smallsize *values, size_t size) {
    if constexpr (flix_benchmark::requires_ordered_lookup<Index>::value) {
        if (values)
            thrust::sort_by_key(thrust::cuda::par.on(0), keys, keys + size, values);
        else
            thrust::sort(thrust::cuda::par.on(0), keys, keys + size);
    }
}
#endif
