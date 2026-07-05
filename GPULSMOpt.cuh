#pragma once
#include "gpu_dictionary_adapter.cuh"

#include <cub/cub.cuh>
#include <cub/device/device_merge.cuh>
#include <cuda_runtime.h>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t err__ = (stmt);                                                \
    if (err__ != cudaSuccess) {                                                \
      throw std::runtime_error(cudaGetErrorString(err__));                     \
    }                                                                          \
  } while (false)
#endif

namespace gpulsmopt_detail {

#ifndef GPULSMOPT_SEGMENT_BUCKETS
#define GPULSMOPT_SEGMENT_BUCKETS 256
#endif
#ifndef GPULSMOPT_TARGET_FILL
#define GPULSMOPT_TARGET_FILL 23
#endif
#ifndef GPULSMOPT_INPLACE_MAX_INCOMING
#define GPULSMOPT_INPLACE_MAX_INCOMING 1024
#endif
#ifndef GPULSMOPT_RANGE_MAX_CANDIDATES
#define GPULSMOPT_RANGE_MAX_CANDIDATES 16777216
#endif
#ifndef GPULSMOPT_RUN_DRAIN_FLOOR
#define GPULSMOPT_RUN_DRAIN_FLOOR 65536
#endif
#ifndef GPULSMOPT_RUN_DRAIN_DIVISOR
#define GPULSMOPT_RUN_DRAIN_DIVISOR 20
#endif
#ifndef GPULSMOPT_RUN_DRAIN_SHEET_NUMERATOR
#define GPULSMOPT_RUN_DRAIN_SHEET_NUMERATOR 0
#endif
#ifndef GPULSMOPT_RUN_DRAIN_SHEET_DENOMINATOR
#define GPULSMOPT_RUN_DRAIN_SHEET_DENOMINATOR 1
#endif
#ifndef GPULSMOPT_DISTINCT_KEYS
#define GPULSMOPT_DISTINCT_KEYS 1
#endif

// Run-layer tuning.
#ifndef GPULSMOPT_RUN_CAPACITY
#define GPULSMOPT_RUN_CAPACITY (1 << 22)
#endif
// Tier-2 generations before merge-down.
#ifndef GPULSMOPT_TIER2_FANOUT
#define GPULSMOPT_TIER2_FANOUT 4
#endif
// Run-layer flush threshold.
#ifndef GPULSMOPT_RUN_FLUSH_BUDGET
#define GPULSMOPT_RUN_FLUSH_BUDGET (1 << 26)
#endif
// 0 drains the run layer directly to the sheet.
#ifndef GPULSMOPT_ENABLE_TIER2
#define GPULSMOPT_ENABLE_TIER2 0
#endif
// 0 forces Method A (scan per run); 1 forces Method B (sort once + slice);
// 2 (default) auto-selects per batch: A when count < GPULSMOPT_ROUTE_AUTO_LIMIT
// (small batch -> skip the sort startup), B otherwise (large batch -> parallel
// sort+slice beats A's few-block scan).
#ifndef GPULSMOPT_ROUTE_METHOD
#define GPULSMOPT_ROUTE_METHOD 2
#endif
#ifndef GPULSMOPT_ROUTE_AUTO_LIMIT
#define GPULSMOPT_ROUTE_AUTO_LIMIT (1 << 20)
#endif
// ST-FliX: drain the sheet with thread-per-key deletes (not warp-per-key) once
// the tombstone batch has at least this many keys; the warp kernel routes each
// key with a lane-0-only binary search (31 idle lanes), which loses badly at
// scale. Small batches keep the warp for better occupancy.
#ifndef GPULSMOPT_ST_DELETE_LIMIT
#define GPULSMOPT_ST_DELETE_LIMIT 32768
#endif

constexpr int kBucketSlots = 32;
constexpr int kSegmentBuckets = GPULSMOPT_SEGMENT_BUCKETS;
constexpr int kSegmentSlots = kSegmentBuckets * kBucketSlots;
static_assert(GPULSMOPT_TARGET_FILL >= 1 &&
                  GPULSMOPT_TARGET_FILL <= kBucketSlots,
              "target fill must fit a bucket");
constexpr std::uint32_t kEmptyKey = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kInsert = 1;
constexpr std::uint32_t kTombstone = 0;
constexpr std::size_t kRunsPerClass = 4;
constexpr int kRunCapacity = GPULSMOPT_RUN_CAPACITY;
constexpr std::size_t kStDeleteLimit = GPULSMOPT_ST_DELETE_LIMIT;
constexpr std::size_t kTier2Fanout = GPULSMOPT_TIER2_FANOUT;
constexpr std::size_t kRunDrainDivisor = GPULSMOPT_RUN_DRAIN_DIVISOR;
constexpr std::size_t kRunDrainSheetNumerator =
    GPULSMOPT_RUN_DRAIN_SHEET_NUMERATOR;
constexpr std::size_t kRunDrainSheetDenominator =
    GPULSMOPT_RUN_DRAIN_SHEET_DENOMINATOR;
static_assert(kRunDrainDivisor >= 1, "drain divisor must be positive");
static_assert(kRunDrainSheetDenominator >= 1,
              "sheet drain denominator must be positive");

struct DeviceUpdateBatch {
  const std::uint32_t *keys = nullptr;
  const std::uint32_t *values = nullptr;
  const std::uint8_t *ops = nullptr;
  std::size_t count = 0;
};

struct DeviceKeyBatch {
  const std::uint32_t *keys = nullptr;
  std::size_t count = 0;
};

struct LessU32 {
  __host__ __device__ bool operator()(std::uint32_t a,
                                      std::uint32_t b) const {
    return a < b;
  }
};

__host__ __device__ inline std::uint32_t ceil_div_u32(std::uint32_t a,
                                                      std::uint32_t b) {
  return (a + b - 1) / b;
}


__host__ __device__ inline std::uint64_t drain_sort_key(std::uint32_t key,
                                                        std::uint32_t seq) {
  return (static_cast<std::uint64_t>(key) << 32) |
         static_cast<std::uint64_t>(UINT32_MAX - seq);
}

__host__ __device__ inline std::size_t
lower_bound_u32(const std::uint32_t *data, std::size_t n, std::uint32_t key) {
  std::size_t first = 0;
  while (n > 0) {
    const std::size_t step = n >> 1;
    const std::size_t mid = first + step;
    if (data[mid] < key) {
      first = mid + 1;
      n -= step + 1;
    } else {
      n = step;
    }
  }
  return first;
}

__host__ __device__ inline std::size_t
upper_bound_u32(const std::uint32_t *data, std::size_t n, std::uint32_t key) {
  std::size_t first = 0;
  while (n > 0) {
    const std::size_t step = n >> 1;
    const std::size_t mid = first + step;
    if (!(key < data[mid])) {
      first = mid + 1;
      n -= step + 1;
    } else {
      n = step;
    }
  }
  return first;
}





__global__ void fill_drain_sort_keys_kernel(const std::uint32_t *keys,
                                            const std::uint32_t *seqs,
                                            std::uint64_t *sort_keys,
                                            std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    sort_keys[i] = drain_sort_key(keys[i], seqs[i]);
  }
}





__global__ void seg_route_keys_kernel(const std::uint32_t *keys, std::size_t n,
                                      const std::uint32_t *dir_boundary,
                                      std::size_t dir_count,
                                      std::uint32_t *out_ordinal) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  std::size_t ord = lower_bound_u32(dir_boundary, dir_count, keys[i]);
  if (ord >= dir_count)
    ord = dir_count - 1;
  out_ordinal[i] = static_cast<std::uint32_t>(ord);
}

__device__ inline bool seg_point_lookup(std::uint32_t seg_id, std::uint32_t key,
                                        const std::uint32_t *pool_keys,
                                        const std::uint32_t *pool_values,
                                        const std::uint8_t *pool_valid,
                                        const std::uint32_t *seg_bucket_max,
                                        std::uint32_t *out_value) {
  const std::size_t meta_base =
      static_cast<std::size_t>(seg_id) * kSegmentBuckets;
  const std::size_t bucket =
      lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, key);
  if (bucket >= kSegmentBuckets)
    return false;
  const std::size_t start =
      static_cast<std::size_t>(seg_id) * kSegmentSlots + bucket * kBucketSlots;
  for (int lane = 0; lane < kBucketSlots; ++lane) {
    if (pool_valid[start + lane] && pool_keys[start + lane] == key) {
      if (out_value)
        *out_value = pool_values[start + lane];
      return true;
    }
  }
  return false;
}

__device__ inline bool seg_sheet_contains(std::uint32_t key,
                                          const std::uint32_t *dir_boundary,
                                          const std::uint32_t *dir_seg_id,
                                          std::size_t dir_count,
                                          const std::uint32_t *pool_keys,
                                          const std::uint8_t *pool_valid,
                                          const std::uint32_t *seg_bucket_max) {
  std::size_t ord = lower_bound_u32(dir_boundary, dir_count, key);
  if (ord >= dir_count)
    ord = dir_count - 1;
  const std::uint32_t seg_id = dir_seg_id[ord];
  const std::size_t meta_base =
      static_cast<std::size_t>(seg_id) * kSegmentBuckets;
  const std::size_t bucket =
      lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, key);
  if (bucket >= kSegmentBuckets)
    return false;
  const std::size_t start =
      static_cast<std::size_t>(seg_id) * kSegmentSlots + bucket * kBucketSlots;
  for (int lane = 0; lane < kBucketSlots; ++lane) {
    if (pool_valid[start + lane] && pool_keys[start + lane] == key)
      return true;
  }
  return false;
}

__device__ inline std::uint32_t seg_range_count_one(
    std::uint32_t seg_id, std::uint32_t lo, std::uint32_t hi,
    const std::uint32_t *pool_keys, const std::uint8_t *pool_valid,
    const std::uint32_t *seg_bucket_max, const std::uint32_t *seg_bucket_live) {
  const std::size_t meta_base =
      static_cast<std::size_t>(seg_id) * kSegmentBuckets;
  const std::size_t slot_base =
      static_cast<std::size_t>(seg_id) * kSegmentSlots;
  const std::size_t first =
      lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, lo);
  if (first >= kSegmentBuckets)
    return 0;
  const std::size_t last =
      lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, hi);
  std::uint32_t count = 0;
  auto scan_bucket = [&](std::size_t b) {
    const std::size_t start = slot_base + b * kBucketSlots;
    for (int lane = 0; lane < kBucketSlots; ++lane) {
      const std::size_t pos = start + lane;
      if (pool_valid[pos] && pool_keys[pos] >= lo && pool_keys[pos] <= hi)
        ++count;
    }
  };
  if (first == last) {
    scan_bucket(first);
    return count;
  }
  scan_bucket(first);
  const std::size_t full_end = last < kSegmentBuckets ? last : kSegmentBuckets;
  if (last < kSegmentBuckets)
    scan_bucket(last);
  for (std::size_t b = first + 1; b < full_end; ++b) {
    count += seg_bucket_live[meta_base + b];
  }
  return count;
}

__device__ inline std::uint32_t seg_range_sum_one(
    std::uint32_t seg_id, std::uint32_t lo, std::uint32_t hi,
    const std::uint32_t *pool_keys, const std::uint32_t *pool_values,
    const std::uint8_t *pool_valid, const std::uint32_t *seg_bucket_max,
    const std::uint32_t *seg_bucket_value_sum) {
  const std::size_t meta_base =
      static_cast<std::size_t>(seg_id) * kSegmentBuckets;
  const std::size_t slot_base =
      static_cast<std::size_t>(seg_id) * kSegmentSlots;
  const std::size_t first =
      lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, lo);
  if (first >= kSegmentBuckets)
    return 0;
  const std::size_t last =
      lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, hi);
  std::uint32_t sum = 0;
  auto scan_bucket = [&](std::size_t b) {
    const std::size_t start = slot_base + b * kBucketSlots;
    for (int lane = 0; lane < kBucketSlots; ++lane) {
      const std::size_t pos = start + lane;
      if (pool_valid[pos] && pool_keys[pos] >= lo && pool_keys[pos] <= hi)
        sum += pool_values[pos];
    }
  };
  if (first == last) {
    scan_bucket(first);
    return sum;
  }
  scan_bucket(first);
  const std::size_t full_end = last < kSegmentBuckets ? last : kSegmentBuckets;
  if (last < kSegmentBuckets)
    scan_bucket(last);
  for (std::size_t b = first + 1; b < full_end; ++b) {
    sum += seg_bucket_value_sum[meta_base + b];
  }
  return sum;
}

__device__ inline std::uint32_t seg_sheet_range_sum(
    std::uint32_t lo, std::uint32_t hi, const std::uint32_t *dir_boundary,
    const std::uint32_t *dir_seg_id, const std::uint32_t *dir_value_prefix,
    std::size_t dir_count, const std::uint32_t *pool_keys,
    const std::uint32_t *pool_values, const std::uint8_t *pool_valid,
    const std::uint32_t *seg_bucket_max,
    const std::uint32_t *seg_bucket_value_sum) {
  std::size_t pl = lower_bound_u32(dir_boundary, dir_count, lo);
  if (pl >= dir_count)
    return 0;
  std::size_t pr = lower_bound_u32(dir_boundary, dir_count, hi);
  if (pr >= dir_count)
    pr = dir_count - 1;
  if (pl == pr) {
    return seg_range_sum_one(dir_seg_id[pl], lo, hi, pool_keys, pool_values,
                             pool_valid, seg_bucket_max, seg_bucket_value_sum);
  }
  std::uint32_t sum =
      seg_range_sum_one(dir_seg_id[pl], lo, hi, pool_keys, pool_values,
                        pool_valid, seg_bucket_max, seg_bucket_value_sum) +
      seg_range_sum_one(dir_seg_id[pr], lo, hi, pool_keys, pool_values,
                        pool_valid, seg_bucket_max, seg_bucket_value_sum);
  if (pr > pl + 1)
    sum += dir_value_prefix[pr] - dir_value_prefix[pl + 1];
  return sum;
}

__device__ inline std::uint32_t seg_sheet_range_count(
    std::uint32_t lo, std::uint32_t hi, const std::uint32_t *dir_boundary,
    const std::uint32_t *dir_seg_id, const std::uint32_t *dir_prefix,
    std::size_t dir_count, const std::uint32_t *pool_keys,
    const std::uint8_t *pool_valid, const std::uint32_t *seg_bucket_max,
    const std::uint32_t *seg_bucket_live) {
  std::size_t pl = lower_bound_u32(dir_boundary, dir_count, lo);
  if (pl >= dir_count)
    return 0;
  std::size_t pr = lower_bound_u32(dir_boundary, dir_count, hi);
  if (pr >= dir_count)
    pr = dir_count - 1;
  if (pl == pr) {
    return seg_range_count_one(dir_seg_id[pl], lo, hi, pool_keys, pool_valid,
                               seg_bucket_max, seg_bucket_live);
  }
  std::uint32_t total =
      seg_range_count_one(dir_seg_id[pl], lo, hi, pool_keys, pool_valid,
                          seg_bucket_max, seg_bucket_live) +
      seg_range_count_one(dir_seg_id[pr], lo, hi, pool_keys, pool_valid,
                          seg_bucket_max, seg_bucket_live);
  if (pr > pl + 1)
    total += dir_prefix[pr] - dir_prefix[pl + 1];
  return total;
}

__global__ void seg_gather_candidates_kernel(
    const std::uint32_t *pool_keys, const std::uint32_t *pool_values,
    const std::uint8_t *pool_valid, const std::uint32_t *dirty_seg_id,
    const std::uint32_t *candidate_offset, const std::uint32_t *dirty_old_live,
    const std::uint32_t *dirty_incoming_begin,
    const std::uint32_t *dirty_incoming_end, const std::uint32_t *incoming_keys,
    const std::uint32_t *incoming_values, std::size_t dirty_count,
    std::uint32_t *cand_seg, std::uint32_t *cand_key, std::uint32_t *cand_value,
    std::uint32_t *cand_op, std::uint32_t *cand_seq,
    std::uint32_t *sheet_cursor) {
  const std::size_t m = blockIdx.x;
  if (m >= dirty_count)
    return;
  const std::size_t seg_base =
      static_cast<std::size_t>(dirty_seg_id[m]) * kSegmentSlots;
  const std::uint32_t base = candidate_offset[m];

  for (std::size_t p = threadIdx.x; p < kSegmentSlots; p += blockDim.x) {
    if (!pool_valid[seg_base + p])
      continue;
    const std::uint32_t out = base + atomicAdd(&sheet_cursor[m], 1u);
    cand_seg[out] = static_cast<std::uint32_t>(m);
    cand_key[out] = pool_keys[seg_base + p];
    cand_value[out] = pool_values[seg_base + p];
    cand_op[out] = kInsert;
    cand_seq[out] = 0u;
  }

  const std::size_t begin = dirty_incoming_begin[m];
  const std::size_t end = dirty_incoming_end[m];
  const std::size_t inc_out =
      static_cast<std::size_t>(base) + dirty_old_live[m];
  for (std::size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    const std::size_t out = inc_out + (i - begin);
    cand_seg[out] = static_cast<std::uint32_t>(m);
    cand_key[out] = incoming_keys[i];
    cand_value[out] = incoming_values[i];
    cand_op[out] = kInsert;
    cand_seq[out] = 1u;
  }
}

__global__ void seg_make_resolve_keys_kernel(const std::uint32_t *cand_seg,
                                             const std::uint32_t *cand_key,
                                             const std::uint32_t *cand_seq,
                                             std::size_t count,
                                             std::uint32_t *seq_sort,
                                             std::uint64_t *group_sort) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  seq_sort[i] = UINT32_MAX - cand_seq[i];
  group_sort[i] = (static_cast<std::uint64_t>(cand_seg[i]) << 32) | cand_key[i];
}

__global__ void seg_classify_inplace_kernel(
    const std::uint32_t *dirty_seg_id, const std::uint32_t *dirty_in_begin,
    const std::uint32_t *dirty_in_end, std::size_t dirty_count,
    const std::uint32_t *incoming_keys, const std::uint32_t *pool_keys,
    const std::uint8_t *pool_valid, const std::uint32_t *seg_bucket_max,
    const std::uint32_t *seg_bucket_live, std::uint32_t *seg_slow,
    std::uint32_t *seg_new_live) {
  const std::size_t m = blockIdx.x;
  if (m >= dirty_count)
    return;
  const std::uint32_t seg = dirty_seg_id[m];
  const std::size_t meta = static_cast<std::size_t>(seg) * kSegmentBuckets;
  const std::size_t slot_base = static_cast<std::size_t>(seg) * kSegmentSlots;
  const std::uint32_t in_begin = dirty_in_begin[m];
  const std::uint32_t in_count = dirty_in_end[m] - in_begin;

  if (in_count > GPULSMOPT_INPLACE_MAX_INCOMING) {
    if (threadIdx.x == 0) {
      seg_slow[m] = 1;
      seg_new_live[m] = 0;
    }
    return;
  }

  __shared__ int s_slow;
  __shared__ unsigned long long s_total;
  if (threadIdx.x == 0) {
    s_slow = 0;
    s_total = 0ull;
  }
  __syncthreads();

  for (int b = threadIdx.x; b < kSegmentBuckets; b += blockDim.x) {
    const std::size_t begin =
        b == 0 ? 0
               : upper_bound_u32(incoming_keys + in_begin, in_count,
                                 seg_bucket_max[meta + b - 1]);
    const std::size_t end =
        b == kSegmentBuckets - 1
            ? in_count
            : upper_bound_u32(incoming_keys + in_begin, in_count,
                              seg_bucket_max[meta + b]);
    const std::size_t base =
        slot_base + static_cast<std::size_t>(b) * kBucketSlots;
    int inserts_new = 0;
    for (std::size_t i = in_begin + begin; i < in_begin + end; ++i) {
      const std::uint32_t key = incoming_keys[i];
      bool exists = false;
      for (int lane = 0; lane < kBucketSlots; ++lane) {
        if (pool_valid[base + lane] && pool_keys[base + lane] == key) {
          exists = true;
          break;
        }
      }
      if (!exists)
        ++inserts_new;
    }
    const int new_live =
        static_cast<int>(seg_bucket_live[meta + b]) + inserts_new;
    if (new_live > kBucketSlots)
      atomicOr(&s_slow, 1);
    atomicAdd(&s_total, static_cast<unsigned long long>(new_live));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    seg_slow[m] = static_cast<std::uint32_t>(s_slow);
    seg_new_live[m] = static_cast<std::uint32_t>(s_total);
  }
}

__global__ void seg_apply_inplace_kernel(
    const std::uint32_t *fast_seg_id, const std::uint32_t *fast_in_begin,
    const std::uint32_t *fast_in_end, std::size_t fast_count,
    const std::uint32_t *incoming_keys, const std::uint32_t *incoming_values,
    std::uint32_t *pool_keys, std::uint32_t *pool_values,
    std::uint8_t *pool_valid, std::uint32_t *seg_bucket_max,
    std::uint32_t *seg_bucket_live, std::uint32_t *seg_bucket_value_sum) {
  const std::size_t f = blockIdx.x;
  if (f >= fast_count)
    return;
  const std::uint32_t seg = fast_seg_id[f];
  const std::size_t meta = static_cast<std::size_t>(seg) * kSegmentBuckets;
  const std::size_t slot_base = static_cast<std::size_t>(seg) * kSegmentSlots;
  const std::uint32_t in_begin = fast_in_begin[f];
  const std::uint32_t in_count = fast_in_end[f] - in_begin;

  for (int b = threadIdx.x; b < kSegmentBuckets; b += blockDim.x) {
    const std::size_t begin =
        b == 0 ? 0
               : upper_bound_u32(incoming_keys + in_begin, in_count,
                                 seg_bucket_max[meta + b - 1]);
    const std::size_t end =
        b == kSegmentBuckets - 1
            ? in_count
            : upper_bound_u32(incoming_keys + in_begin, in_count,
                              seg_bucket_max[meta + b]);
    const std::size_t base =
        slot_base + static_cast<std::size_t>(b) * kBucketSlots;

    for (std::size_t i = in_begin + begin; i < in_begin + end; ++i) {
      const std::uint32_t key = incoming_keys[i];
      int found = -1;
      for (int lane = 0; lane < kBucketSlots; ++lane) {
        if (pool_valid[base + lane] && pool_keys[base + lane] == key) {
          found = lane;
          break;
        }
      }
      if (found < 0)
        continue;
      pool_values[base + found] = incoming_values[i];
    }
    for (std::size_t i = in_begin + begin; i < in_begin + end; ++i) {
      const std::uint32_t key = incoming_keys[i];
      bool exists = false;
      for (int lane = 0; lane < kBucketSlots; ++lane) {
        if (pool_valid[base + lane] && pool_keys[base + lane] == key) {
          exists = true;
          break;
        }
      }
      if (exists)
        continue;
      for (int lane = 0; lane < kBucketSlots; ++lane) {
        if (!pool_valid[base + lane]) {
          pool_keys[base + lane] = key;
          pool_values[base + lane] = incoming_values[i];
          pool_valid[base + lane] = 1;
          break;
        }
      }
    }
  }
  __syncthreads();
  for (int b = threadIdx.x; b < kSegmentBuckets; b += blockDim.x) {
    const std::size_t base =
        slot_base + static_cast<std::size_t>(b) * kBucketSlots;
    std::uint32_t live = 0, sum = 0, mx = 0;
    bool any = false;
    for (int lane = 0; lane < kBucketSlots; ++lane) {
      if (pool_valid[base + lane]) {
        ++live;
        sum += pool_values[base + lane];
        const std::uint32_t k = pool_keys[base + lane];
        if (!any || k > mx) {
          mx = k;
          any = true;
        }
      }
    }
    seg_bucket_live[meta + b] = live;
    if (any)
      seg_bucket_max[meta + b] = mx;
    seg_bucket_value_sum[meta + b] = sum;
  }
}





__global__ void seg_delete_keys_kernel(
    const std::uint32_t *delete_keys, std::size_t delete_count,
    const std::uint32_t *dir_boundary, const std::uint32_t *dir_seg_id,
    std::size_t dir_count, std::uint32_t *pool_keys, std::uint32_t *pool_values,
    std::uint8_t *pool_valid, const std::uint32_t *seg_bucket_max,
    std::uint32_t *seg_bucket_live, std::uint32_t *seg_bucket_value_sum) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= delete_count || dir_count == 0)
    return;
  const std::uint32_t key = delete_keys[i];
  std::size_t ord = lower_bound_u32(dir_boundary, dir_count, key);
  if (ord >= dir_count)
    ord = dir_count - 1;
  const std::uint32_t seg = dir_seg_id[ord];
  const std::size_t meta = static_cast<std::size_t>(seg) * kSegmentBuckets;
  const std::size_t bucket =
      lower_bound_u32(seg_bucket_max + meta, kSegmentBuckets, key);
  if (bucket >= kSegmentBuckets)
    return;
  const std::size_t base =
      static_cast<std::size_t>(seg) * kSegmentSlots + bucket * kBucketSlots;
  for (int lane = 0; lane < kBucketSlots; ++lane) {
    const std::size_t pos = base + lane;
    if (!pool_valid[pos] || pool_keys[pos] != key)
      continue;
    const std::uint32_t value = pool_values[pos];
    pool_valid[pos] = 0u;
    pool_values[pos] = 0u;
    atomicSub(seg_bucket_live + meta + bucket, 1u);
    atomicSub(seg_bucket_value_sum + meta + bucket, value);
    return;
  }
}

__global__ void seg_delete_keys_warp_kernel(
    const std::uint32_t *delete_keys, std::size_t delete_count,
    const std::uint32_t *dir_boundary, const std::uint32_t *dir_seg_id,
    std::size_t dir_count, std::uint32_t *pool_keys, std::uint32_t *pool_values,
    std::uint8_t *pool_valid, const std::uint32_t *seg_bucket_max,
    std::uint32_t *seg_bucket_live, std::uint32_t *seg_bucket_value_sum) {
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t warp = tid / warpSize;
  const unsigned lane = threadIdx.x & (warpSize - 1);
  if (warp >= delete_count || dir_count == 0)
    return;

  const std::uint32_t key = delete_keys[warp];
  std::uint32_t seg = 0;
  std::uint32_t bucket = kSegmentBuckets;
  if (lane == 0) {
    std::size_t ord = lower_bound_u32(dir_boundary, dir_count, key);
    if (ord >= dir_count)
      ord = dir_count - 1;
    seg = dir_seg_id[ord];
    const std::size_t meta = static_cast<std::size_t>(seg) * kSegmentBuckets;
    bucket = static_cast<std::uint32_t>(
        lower_bound_u32(seg_bucket_max + meta, kSegmentBuckets, key));
  }
  seg = __shfl_sync(UINT32_MAX, seg, 0);
  bucket = __shfl_sync(UINT32_MAX, bucket, 0);
  if (bucket >= kSegmentBuckets)
    return;

  const std::size_t meta = static_cast<std::size_t>(seg) * kSegmentBuckets;
  const std::size_t pos = static_cast<std::size_t>(seg) * kSegmentSlots +
                          bucket * kBucketSlots + lane;
  const bool match = pool_valid[pos] && pool_keys[pos] == key;
  const unsigned mask = __ballot_sync(UINT32_MAX, match);
  if (mask == 0)
    return;
  const unsigned first = __ffs(mask) - 1;
  if (lane == first) {
    const std::uint32_t value = pool_values[pos];
    pool_valid[pos] = 0u;
    pool_values[pos] = 0u;
    atomicSub(seg_bucket_live + meta + bucket, 1u);
    atomicSub(seg_bucket_value_sum + meta + bucket, value);
  }
}

__global__ void directory_live_counts_kernel(
    const std::uint32_t *dir_seg_id, std::size_t dir_count,
    const std::uint32_t *seg_bucket_live, std::uint32_t *out_live) {
  const std::size_t ord = blockIdx.x * blockDim.x + threadIdx.x;
  if (ord >= dir_count)
    return;
  const std::size_t meta =
      static_cast<std::size_t>(dir_seg_id[ord]) * kSegmentBuckets;
  std::uint32_t live = 0;
  for (std::size_t b = 0; b < kSegmentBuckets; ++b) {
    live += seg_bucket_live[meta + b];
  }
  out_live[ord] = live;
}

__global__ void seg_count_survivors_kernel(const std::uint32_t *cand_seg,
                                           std::size_t live_count,
                                           std::uint32_t *dirty_live) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < live_count)
    atomicAdd(&dirty_live[cand_seg[i]], 1u);
}

__global__ void seg_output_boundaries_kernel(
    const std::uint32_t *out_dirty, const std::uint32_t *out_local,
    const std::uint32_t *out_k, std::size_t output_count,
    const std::uint32_t *dirty_live, const std::uint32_t *dirty_live_offset,
    const std::uint32_t *dirty_old_boundary, const std::uint32_t *survivor_keys,
    std::uint32_t *out_boundary) {
  const std::size_t o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= output_count)
    return;
  const std::uint32_t m = out_dirty[o];
  const std::uint32_t local = out_local[o];
  const std::uint32_t k = out_k[o];
  if (local + 1 >= k) {
    out_boundary[o] = dirty_old_boundary[m];
    return;
  }
  const std::uint32_t live = dirty_live[m];
  const std::uint32_t per = ceil_div_u32(live, k);
  const std::uint32_t idx = dirty_live_offset[m] + (local + 1u) * per - 1u;
  out_boundary[o] = survivor_keys[idx];
}

__global__ void seg_clear_segments_kernel(const std::uint32_t *out_seg_id,
                                          std::size_t output_count,
                                          std::uint32_t *pool_keys,
                                          std::uint32_t *pool_values,
                                          std::uint8_t *pool_valid) {
  const std::size_t o = blockIdx.x;
  if (o >= output_count)
    return;
  const std::size_t seg_base =
      static_cast<std::size_t>(out_seg_id[o]) * kSegmentSlots;
  for (std::size_t p = threadIdx.x; p < kSegmentSlots; p += blockDim.x) {
    pool_keys[seg_base + p] = kEmptyKey;
    pool_values[seg_base + p] = 0u;
    pool_valid[seg_base + p] = 0u;
  }
}

__global__ void seg_scatter_survivors_kernel(
    const std::uint32_t *cand_seg, const std::uint32_t *cand_key,
    const std::uint32_t *cand_value, std::size_t live_count,
    const std::uint32_t *dirty_live, const std::uint32_t *dirty_live_offset,
    const std::uint32_t *dirty_k, const std::uint32_t *dirty_output_base,
    const std::uint32_t *output_seg_id, std::uint32_t target_fill,
    std::uint32_t *pool_keys, std::uint32_t *pool_values,
    std::uint8_t *pool_valid) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= live_count)
    return;
  const std::uint32_t m = cand_seg[i];
  const std::uint32_t live = dirty_live[m];
  const std::uint32_t k = dirty_k[m];
  const std::uint32_t per = live == 0 ? 1u : ceil_div_u32(live, k);
  const std::uint32_t rank =
      static_cast<std::uint32_t>(i) - dirty_live_offset[m];
  const std::uint32_t local = rank / per;
  const std::uint32_t rank_in_out = rank - local * per;
  const std::uint32_t seg_id = output_seg_id[dirty_output_base[m] + local];
  const std::uint32_t bucket = rank_in_out / target_fill;
  const std::uint32_t slot = rank_in_out % target_fill;
  const std::size_t pos = static_cast<std::size_t>(seg_id) * kSegmentSlots +
                          static_cast<std::size_t>(bucket) * kBucketSlots +
                          slot;
  pool_keys[pos] = cand_key[i];
  pool_values[pos] = cand_value[i];
  pool_valid[pos] = 1u;
}

__global__ void seg_build_segment_metadata_kernel(
    const std::uint32_t *out_seg_id, const std::uint32_t *out_boundary,
    std::size_t output_count, const std::uint32_t *pool_keys,
    const std::uint32_t *pool_values, const std::uint8_t *pool_valid,
    std::uint32_t *seg_bucket_max, std::uint32_t *seg_bucket_live,
    std::uint32_t *seg_bucket_value_sum) {
  const std::size_t o = blockIdx.x;
  if (o >= output_count)
    return;
  const std::uint32_t seg_id = out_seg_id[o];
  const std::uint32_t boundary = out_boundary[o];
  const std::size_t slot_base =
      static_cast<std::size_t>(seg_id) * kSegmentSlots;
  const std::size_t meta_base =
      static_cast<std::size_t>(seg_id) * kSegmentBuckets;
  for (std::size_t b = threadIdx.x; b < kSegmentBuckets; b += blockDim.x) {
    const std::size_t start = slot_base + b * kBucketSlots;
    std::uint32_t live = 0;
    std::uint32_t sum = 0;
    std::uint32_t max_key = boundary;
    for (int lane = 0; lane < kBucketSlots; ++lane) {
      if (pool_valid[start + lane]) {
        ++live;
        sum += pool_values[start + lane];
        max_key = pool_keys[start + lane];
      }
    }
    seg_bucket_live[meta_base + b] = live;
    seg_bucket_max[meta_base + b] = live > 0 ? max_key : boundary;
    seg_bucket_value_sum[meta_base + b] = sum;
  }
}

__global__ void directory_value_sums_kernel(
    const std::uint32_t *dir_seg_id, std::size_t dir_count,
    const std::uint32_t *seg_bucket_value_sum, std::uint32_t *out_sums) {
  const std::size_t ord = blockIdx.x * blockDim.x + threadIdx.x;
  if (ord >= dir_count)
    return;
  const std::size_t meta_base =
      static_cast<std::size_t>(dir_seg_id[ord]) * kSegmentBuckets;
  std::uint32_t sum = 0;
  for (std::size_t b = 0; b < kSegmentBuckets; ++b) {
    sum += seg_bucket_value_sum[meta_base + b];
  }
  out_sums[ord] = sum;
}

__global__ void seg_merge_gather_kernel(
    const std::uint32_t *pool_keys, const std::uint32_t *pool_values,
    const std::uint8_t *pool_valid, const std::uint32_t *group_src_seg,
    const std::uint32_t *group_src_begin, const std::uint32_t *group_src_end,
    const std::uint32_t *candidate_offset, std::size_t group_count,
    std::uint32_t *cand_seg, std::uint32_t *cand_key, std::uint32_t *cand_value,
    std::uint32_t *cursor) {
  const std::size_t g = blockIdx.x;
  if (g >= group_count)
    return;
  const std::uint32_t base = candidate_offset[g];
  const std::uint32_t sb = group_src_begin[g];
  const std::uint32_t se = group_src_end[g];
  for (std::uint32_t s = sb; s < se; ++s) {
    const std::size_t seg_base =
        static_cast<std::size_t>(group_src_seg[s]) * kSegmentSlots;
    for (std::size_t p = threadIdx.x; p < kSegmentSlots; p += blockDim.x) {
      if (!pool_valid[seg_base + p])
        continue;
      const std::uint32_t out = base + atomicAdd(&cursor[g], 1u);
      cand_seg[out] = static_cast<std::uint32_t>(g);
      cand_key[out] = pool_keys[seg_base + p];
      cand_value[out] = pool_values[seg_base + p];
    }
  }
}

__global__ void seg_merge_sort_keys_kernel(const std::uint32_t *cand_seg,
                                           const std::uint32_t *cand_key,
                                           std::size_t count,
                                           std::uint64_t *group_sort) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  group_sort[i] = (static_cast<std::uint64_t>(cand_seg[i]) << 32) | cand_key[i];
}










// Segment ceiling key.
__device__ inline std::uint32_t seg_segment_ceiling(
    std::uint32_t seg_id, std::uint32_t q, const std::uint32_t *pool_keys,
    const std::uint8_t *pool_valid, const std::uint32_t *seg_bucket_max) {
  const std::size_t meta_base =
      static_cast<std::size_t>(seg_id) * kSegmentBuckets;
  const std::size_t slot_base =
      static_cast<std::size_t>(seg_id) * kSegmentSlots;
  std::uint32_t best = kEmptyKey;
  for (std::size_t b =
           lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, q);
       b < kSegmentBuckets; ++b) {
    const std::size_t start = slot_base + b * kBucketSlots;
    for (int lane = 0; lane < kBucketSlots; ++lane) {
      const std::size_t pos = start + lane;
      if (pool_valid[pos] && pool_keys[pos] >= q && pool_keys[pos] < best)
        best = pool_keys[pos];
    }
    if (best != kEmptyKey)
      break;
  }
  return best;
}

// Sheet ceiling key.
__device__ inline std::uint32_t
seg_sheet_ceiling(std::uint32_t q, const std::uint32_t *dir_boundary,
                  const std::uint32_t *dir_seg_id, std::size_t dir_count,
                  const std::uint32_t *pool_keys,
                  const std::uint8_t *pool_valid,
                  const std::uint32_t *seg_bucket_max) {
  std::uint32_t best = kEmptyKey;
  for (std::size_t ord = lower_bound_u32(dir_boundary, dir_count, q);
       ord < dir_count && best == kEmptyKey; ++ord) {
    best = seg_segment_ceiling(dir_seg_id[ord], q, pool_keys, pool_valid,
                               seg_bucket_max);
  }
  return best;
}

// Route key to a run ordinal.
__device__ __host__ inline std::size_t
rl_route_key(const std::uint32_t *run_boundary, std::size_t run_count,
             std::uint32_t key) {
  std::size_t ord = lower_bound_u32(run_boundary, run_count, key);
  if (ord >= run_count && run_count > 0)
    ord = run_count - 1;
  return ord;
}

// Tuple predicate for live inserts.
struct TupleOpIsInsert {
  template <class Tuple>
  __host__ __device__ bool operator()(const Tuple &t) const {
    return thrust::get<2>(t) == kInsert;
  }
};

// Count batch elements per run ordinal.
__global__ void rl_histogram_kernel(const std::uint32_t *ordinal, std::size_t n,
                                    std::uint32_t *counts) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    atomicAdd(&counts[ordinal[i]], 1u);
}

// Method A scans each run against the batch.
__device__ inline bool rl_in_run_range(std::uint32_t key, std::uint32_t hi,
                                       bool has_lo, std::uint32_t lo) {
  return key <= hi && (!has_lo || key > lo);
}

// Method A count pass.
__global__ void rl_run_count_kernel(const std::uint32_t *keys, std::size_t n,
                                    const std::uint32_t *boundary, std::size_t r,
                                    std::uint32_t *counts) {
  const std::size_t run = blockIdx.x;
  if (run >= r)
    return;
  const std::uint32_t hi = boundary[run];
  const bool has_lo = run > 0;
  const std::uint32_t lo = has_lo ? boundary[run - 1] : 0u;
  std::uint32_t local = 0;
  for (std::size_t i = threadIdx.x; i < n; i += blockDim.x)
    if (rl_in_run_range(keys[i], hi, has_lo, lo))
      ++local;
  __shared__ unsigned s_cnt;
  if (threadIdx.x == 0)
    s_cnt = 0;
  __syncthreads();
  atomicAdd(&s_cnt, local);
  __syncthreads();
  if (threadIdx.x == 0)
    counts[run] = s_cnt;
}

// Method A fast append pass.
__global__ void rl_run_append_kernel(
    const std::uint8_t *is_fast, const std::uint32_t *boundary,
    const std::uint32_t *run_start, const std::uint32_t *old_size, std::size_t r,
    const std::uint32_t *keys, const std::uint32_t *values, std::uint8_t op,
    std::size_t n, std::uint32_t *pool_keys, std::uint32_t *pool_values,
    std::uint8_t *pool_ops) {
  const std::size_t run = blockIdx.x;
  if (run >= r || !is_fast[run])
    return;
  const std::uint32_t hi = boundary[run];
  const bool has_lo = run > 0;
  const std::uint32_t lo = has_lo ? boundary[run - 1] : 0u;
  const std::size_t dst =
      static_cast<std::size_t>(run_start[run]) + old_size[run];
  __shared__ unsigned s_pos;
  if (threadIdx.x == 0)
    s_pos = 0;
  __syncthreads();
  for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (rl_in_run_range(keys[i], hi, has_lo, lo)) {
      const std::size_t o = dst + atomicAdd(&s_pos, 1u);
      pool_keys[o] = keys[i];
      pool_values[o] = values[i];
      pool_ops[o] = op;
    }
  }
}

// Method A overflow gather.
__global__ void rl_run_gather_kernel(const std::uint32_t *keys,
                                     const std::uint32_t *values, std::size_t n,
                                     std::uint32_t lo, std::uint32_t hi,
                                     std::uint8_t has_lo, std::uint32_t *out_keys,
                                     std::uint32_t *out_vals) {
  __shared__ unsigned s_pos;
  if (threadIdx.x == 0)
    s_pos = 0;
  __syncthreads();
  for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (rl_in_run_range(keys[i], hi, has_lo != 0, lo)) {
      const unsigned slot = atomicAdd(&s_pos, 1u);
      out_keys[slot] = keys[i];
      out_vals[slot] = values[i];
    }
  }
}

// Fused point lookup. The overlay (runs + tier-2) is pre-flattened newest-wins
// into a sorted [inserts | tombstones] list (see resolve_overlay); one thread
// per query binary-searches the inserts block, then the tombstones block, then
// falls through to the sorted sheet. An overlay hit (insert OR tombstone)
// shadows the sheet, exactly like the old newest->oldest layer walk, but at
// O(log overlay) per query instead of scanning the unsorted runs. Every thread
// writes its own answer, so no answered-flag / pre-zeroing is needed.
__global__ void point_lookup_kernel(
    const std::uint32_t *queries, std::size_t n, std::uint32_t *out_value,
    std::uint8_t *out_found, const std::uint32_t *ins_keys,
    const std::uint32_t *ins_values, std::size_t ins_count,
    const std::uint32_t *tomb_keys, std::size_t tomb_count,
    const std::uint32_t *dir_boundary, const std::uint32_t *dir_seg_id,
    std::size_t dir_count, const std::uint32_t *pool_keys,
    const std::uint32_t *pool_values, const std::uint8_t *pool_valid,
    const std::uint32_t *seg_bucket_max) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint32_t key = queries[i];
  // 1) overlay inserts (newest already resolved) shadow the sheet.
  if (ins_count > 0) {
    const std::size_t p = lower_bound_u32(ins_keys, ins_count, key);
    if (p < ins_count && ins_keys[p] == key) {
      out_found[i] = 1u;
      out_value[i] = ins_values[p];
      return;
    }
  }
  // 2) overlay tombstones shadow the sheet (key deleted).
  if (tomb_count > 0) {
    const std::size_t p = lower_bound_u32(tomb_keys, tomb_count, key);
    if (p < tomb_count && tomb_keys[p] == key) {
      out_found[i] = 0u;
      out_value[i] = 0u;
      return;
    }
  }
  // 3) not in overlay: consult the sorted sheet.
  std::uint32_t value = 0;
  bool found = false;
  if (dir_count > 0) {
    std::size_t ord = lower_bound_u32(dir_boundary, dir_count, key);
    if (ord >= dir_count)
      ord = dir_count - 1;
    found = seg_point_lookup(dir_seg_id[ord], key, pool_keys, pool_values,
                             pool_valid, seg_bucket_max, &value);
  }
  out_found[i] = found ? 1u : 0u;
  out_value[i] = found ? value : 0u;
}

// Overlay range helpers.
__device__ inline std::uint32_t
overlay_prefix_range(const std::uint32_t *prefix, const std::uint32_t *keys,
                     std::size_t n, std::uint32_t lo, std::uint32_t hi) {
  if (n == 0)
    return 0;
  const std::size_t b = lower_bound_u32(keys, n, lo);
  const std::size_t e = upper_bound_u32(keys, n, hi);
  return prefix[e] - prefix[b];
}
__device__ inline std::uint32_t overlay_count_range(const std::uint32_t *keys,
                                                    std::size_t n,
                                                    std::uint32_t lo,
                                                    std::uint32_t hi) {
  if (n == 0)
    return 0;
  const std::size_t b = lower_bound_u32(keys, n, lo);
  const std::size_t e = upper_bound_u32(keys, n, hi);
  return static_cast<std::uint32_t>(e - b);
}

// Sheet values for tombstone accounting.
__global__ void sheet_point_values_kernel(
    const std::uint32_t *keys, std::size_t n, const std::uint32_t *dir_boundary,
    const std::uint32_t *dir_seg_id, std::size_t dir_count,
    const std::uint32_t *pool_keys, const std::uint32_t *pool_values,
    const std::uint8_t *pool_valid, const std::uint32_t *seg_bucket_max,
    std::uint32_t *out_val, std::uint32_t *out_flag) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  std::uint32_t v = 0;
  bool f = false;
  if (dir_count > 0) {
    std::size_t ord = lower_bound_u32(dir_boundary, dir_count, keys[i]);
    if (ord >= dir_count)
      ord = dir_count - 1;
    f = seg_point_lookup(dir_seg_id[ord], keys[i], pool_keys, pool_values,
                         pool_valid, seg_bucket_max, &v);
  }
  out_val[i] = f ? v : 0u;
  out_flag[i] = f ? 1u : 0u;
}

// Range sum/count over sheet plus overlay.
__global__ void range_overlay_kernel(
    const std::uint32_t *lo, const std::uint32_t *hi, std::uint32_t *out_sums,
    std::uint32_t *out_counts, std::size_t query_count,
    const std::uint32_t *dir_boundary, const std::uint32_t *dir_seg_id,
    const std::uint32_t *dir_prefix, const std::uint32_t *dir_value_prefix,
    std::size_t dir_count, const std::uint32_t *pool_keys,
    const std::uint32_t *pool_values, const std::uint8_t *pool_valid,
    const std::uint32_t *seg_bucket_max, const std::uint32_t *seg_bucket_live,
    const std::uint32_t *seg_bucket_value_sum, const std::uint32_t *ins_keys,
    const std::uint32_t *ins_prefix, std::size_t ins_count,
    const std::uint32_t *tomb_keys, const std::uint32_t *tomb_val_prefix,
    const std::uint32_t *tomb_cnt_prefix, std::size_t tomb_count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= query_count)
    return;
  const std::uint32_t l = lo[i], h = hi[i];
  if (l > h) {
    out_sums[i] = 0;
    if (out_counts)
      out_counts[i] = 0;
    return;
  }
  std::uint32_t sum = seg_sheet_range_sum(
      l, h, dir_boundary, dir_seg_id, dir_value_prefix, dir_count, pool_keys,
      pool_values, pool_valid, seg_bucket_max, seg_bucket_value_sum);
  sum += overlay_prefix_range(ins_prefix, ins_keys, ins_count, l, h);
  sum -= overlay_prefix_range(tomb_val_prefix, tomb_keys, tomb_count, l, h);
  out_sums[i] = sum;
  if (out_counts) {
    std::uint32_t c = seg_sheet_range_count(l, h, dir_boundary, dir_seg_id,
                                            dir_prefix, dir_count, pool_keys,
                                            pool_valid, seg_bucket_max,
                                            seg_bucket_live);
    c += overlay_count_range(ins_keys, ins_count, l, h);
    c -= overlay_prefix_range(tomb_cnt_prefix, tomb_keys, tomb_count, l, h);
    out_counts[i] = c;
  }
}

// Distinct range count over sheet plus overlay.
__global__ void count_overlay_kernel(
    const std::uint32_t *lo, const std::uint32_t *hi, std::uint32_t *out_counts,
    std::size_t query_count, const std::uint32_t *dir_boundary,
    const std::uint32_t *dir_seg_id, const std::uint32_t *dir_prefix,
    std::size_t dir_count, const std::uint32_t *pool_keys,
    const std::uint8_t *pool_valid, const std::uint32_t *seg_bucket_max,
    const std::uint32_t *seg_bucket_live, const std::uint32_t *ins_keys,
    std::size_t ins_count, const std::uint32_t *tomb_keys,
    const std::uint32_t *tomb_cnt_prefix, std::size_t tomb_count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= query_count)
    return;
  const std::uint32_t l = lo[i], h = hi[i];
  if (l > h) {
    out_counts[i] = 0;
    return;
  }
  std::uint32_t c = seg_sheet_range_count(l, h, dir_boundary, dir_seg_id,
                                          dir_prefix, dir_count, pool_keys,
                                          pool_valid, seg_bucket_max,
                                          seg_bucket_live);
  c += overlay_count_range(ins_keys, ins_count, l, h);
  c -= overlay_prefix_range(tomb_cnt_prefix, tomb_keys, tomb_count, l, h);
  out_counts[i] = c;
}

// Successor over sheet plus overlay.
__global__ void successor_overlay_kernel(
    const std::uint32_t *queries, std::size_t query_count,
    std::uint32_t *out_keys, const std::uint32_t *dir_boundary,
    const std::uint32_t *dir_seg_id, std::size_t dir_count,
    const std::uint32_t *pool_keys, const std::uint8_t *pool_valid,
    const std::uint32_t *seg_bucket_max, const std::uint32_t *ins_keys,
    std::size_t ins_count, const std::uint32_t *tomb_keys,
    std::size_t tomb_count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= query_count)
    return;
  const std::uint32_t q = queries[i];
  std::uint32_t best = kEmptyKey;
  if (ins_count) {
    const std::size_t p = lower_bound_u32(ins_keys, ins_count, q);
    if (p < ins_count && ins_keys[p] < best)
      best = ins_keys[p];
  }
  std::uint32_t c = seg_sheet_ceiling(q, dir_boundary, dir_seg_id, dir_count,
                                      pool_keys, pool_valid, seg_bucket_max);
  while (c != kEmptyKey) {
    if (c >= best)
      break; // overlay insert already the answer
    bool dead = false;
    if (tomb_count) {
      const std::size_t p = lower_bound_u32(tomb_keys, tomb_count, c);
      dead = (p < tomb_count && tomb_keys[p] == c);
    }
    if (!dead) {
      best = c;
      break;
    }
    if (c >= kEmptyKey - 1)
      break;
    c = seg_sheet_ceiling(c + 1, dir_boundary, dir_seg_id, dir_count, pool_keys,
                          pool_valid, seg_bucket_max);
  }
  out_keys[i] = (best == kEmptyKey) ? 0u : best;
}

// Bucket split and method-B helpers.

// Gather a strided sample for quantile pivots.
__global__ void sample_gather_kernel(const std::uint32_t *src, std::size_t m,
                                     std::uint32_t *out, std::size_t s) {
  const std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= s)
    return;
  std::size_t idx = (j * m) / s;
  if (idx >= m)
    idx = m - 1;
  out[j] = src[idx];
}

// Assign each entry to a pivot bucket.
__global__ void bucket_assign_kernel(const std::uint32_t *keys, std::size_t m,
                                     const std::uint32_t *B, std::size_t k,
                                     std::uint32_t *out_bucket) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m)
    return;
  std::size_t c = lower_bound_u32(B, k, keys[i]);
  if (c >= k)
    c = k - 1;
  out_bucket[i] = static_cast<std::uint32_t>(c);
}

// Find each run's slice in the sorted batch.
__global__ void run_slice_bounds_kernel(const std::uint32_t *keys,
                                        std::size_t m,
                                        const std::uint32_t *boundary,
                                        std::size_t r, std::uint32_t *out_begin,
                                        std::uint32_t *out_end) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= r)
    return;
  const std::size_t b = (i == 0) ? 0 : upper_bound_u32(keys, m, boundary[i - 1]);
  const std::size_t e =
      (i + 1 == r) ? m : upper_bound_u32(keys, m, boundary[i]);
  out_begin[i] = static_cast<std::uint32_t>(b);
  out_end[i] = static_cast<std::uint32_t>(e);
}

// Copy each sorted slice into its run block.
__global__ void rl_pull_fast_kernel(
    const std::uint8_t *is_fast, const std::uint32_t *run_begin,
    const std::uint32_t *run_end, const std::uint32_t *run_start,
    const std::uint32_t *old_size, const std::uint32_t *sorted_keys,
    const std::uint32_t *sorted_vals, std::uint8_t op, std::uint32_t *pool_keys,
    std::uint32_t *pool_values, std::uint8_t *pool_ops) {
  const std::uint32_t i = blockIdx.x;
  if (!is_fast[i])
    return;
  const std::uint32_t b = run_begin[i];
  const std::uint32_t e = run_end[i];
  const std::size_t dst = static_cast<std::size_t>(run_start[i]) + old_size[i];
  for (std::uint32_t p = b + threadIdx.x; p < e; p += blockDim.x) {
    const std::size_t o = dst + (p - b);
    pool_keys[o] = sorted_keys[p];
    pool_values[o] = sorted_vals[p];
    pool_ops[o] = op;
  }
}

} // namespace gpulsmopt_detail

class GPULSMOpt {
public:
  using DeviceUpdateBatch = gpulsmopt_detail::DeviceUpdateBatch;
  using DeviceKeyBatch = gpulsmopt_detail::DeviceKeyBatch;

  struct DirectorySnapshot {
    std::vector<std::uint32_t> seg_id;
    std::vector<std::uint32_t> boundary;
    std::vector<std::uint32_t> live;
    std::vector<std::uint32_t> prefix;
  };

  explicit GPULSMOpt(const DictionaryConfig &config)
      : max_elements_(config.max_elements), batch_size_(config.batch_size),
        target_fill_(GPULSMOPT_TARGET_FILL) {
    if (max_elements_ > 0x7fffffffu) {
      throw std::invalid_argument(
          "GPULSMOpt currently supports at most 2^31-1 records");
    }
    target_segment_live_ =
        static_cast<std::uint32_t>(gpulsmopt_detail::kSegmentBuckets) *
        target_fill_;
    initialize_segmented_storage(0);
    init_run_layer(0);
    CUDA_CHECK(cudaStreamSynchronize(0));
  }

  ~GPULSMOpt() {
    // Finish pending drain before freeing device memory.
    join_pending_drain();
  }

  void clear(cudaStream_t stream) {
    join_pending_drain();
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    reset_directory_to_root(stream);
    init_run_layer(stream);
    overlay_dirty_ = true; // sheet + overlay reset; cached read index stale
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void insert(const DeviceKeyValueBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    join_pending_drain(); // finish any prior background drain before mutating
    bool need_drain = false;
    {
      std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
#ifdef GPULSMOPT_PROFILE_INSERT
      cudaEvent_t ia, ib;
      cudaEventCreate(&ia);
      cudaEventCreate(&ib);
      cudaEventRecord(ia, stream);
#endif
      insert_records(batch.keys, batch.values,
                     static_cast<std::uint8_t>(gpulsmopt_detail::kInsert),
                     batch.count, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
#ifdef GPULSMOPT_PROFILE_INSERT
      cudaEventRecord(ib, stream);
      cudaEventSynchronize(ib);
      float t_records = 0.f;
      cudaEventElapsedTime(&t_records, ia, ib);
      printf("[prof] insert %zu keys: insert_records=%.3f ms  runs_after=%zu\n",
             batch.count, t_records, h_run_block_.size());
      cudaEventDestroy(ia);
      cudaEventDestroy(ib);
#endif
      need_drain = drain_needed_locked(stream);
    }
    // Launch drain after append.
    if (need_drain)
      launch_background_drain();
  }

  void erase(const DeviceKeyBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    join_pending_drain();
    bool need_drain = false;
    {
      std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
      // Append tombstones through the insert path.
      insert_records(batch.keys, batch.keys,
                     static_cast<std::uint8_t>(gpulsmopt_detail::kTombstone),
                     batch.count, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      need_drain = drain_needed_locked(stream);
    }
    if (need_drain)
      launch_background_drain();
  }

  void lookup(const DeviceLookupBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    join_pending_drain(); // absorb any backgrounded drain here, not on inserts
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    lookup_layered(batch, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Range APIs read through a flattened overlay.
  void count(const DeviceRangeBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    join_pending_drain();
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    const OverlayReadIndex &ix = overlay_index(stream);
    const int block = 128;
    const int grid = static_cast<int>((batch.count + block - 1) / block);
    gpulsmopt_detail::count_overlay_kernel<<<grid, block, 0, stream>>>(
        batch.lo, batch.hi, batch.out_counts, batch.count,
        raw_or_null(d_dir_boundary_), raw_or_null(d_dir_seg_id_),
        raw_or_null(d_dir_prefix_), h_dir_seg_id_.size(),
        raw_or_null(pool_keys_), raw_or_null(pool_valid_),
        raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_),
        raw_or_null(ix.gk), ix.ins, raw_or_null(ix.gk) + ix.ins,
        raw_or_null(ix.tomb_cnt_prefix), ix.u - ix.ins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void successor(const DeviceSuccessorBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    join_pending_drain();
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    const OverlayReadIndex &ix = overlay_index(stream);
    const int block = 128;
    const int grid = static_cast<int>((batch.count + block - 1) / block);
    gpulsmopt_detail::successor_overlay_kernel<<<grid, block, 0, stream>>>(
        batch.queries, batch.count, batch.out_keys,
        raw_or_null(d_dir_boundary_), raw_or_null(d_dir_seg_id_),
        h_dir_seg_id_.size(), raw_or_null(pool_keys_), raw_or_null(pool_valid_),
        raw_or_null(seg_bucket_max_), raw_or_null(ix.gk), ix.ins,
        raw_or_null(ix.gk) + ix.ins, ix.u - ix.ins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void range(const DeviceRangeOutputBatch &batch, cudaStream_t stream) {
    if (batch.query_count == 0)
      return;
    join_pending_drain();
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    const OverlayReadIndex &ix = overlay_index(stream);
    const int block = 128;
    const int grid = static_cast<int>((batch.query_count + block - 1) / block);
    gpulsmopt_detail::range_overlay_kernel<<<grid, block, 0, stream>>>(
        batch.lo, batch.hi, batch.out_sums, batch.out_counts,
        batch.query_count, raw_or_null(d_dir_boundary_),
        raw_or_null(d_dir_seg_id_), raw_or_null(d_dir_prefix_),
        raw_or_null(d_dir_value_prefix_), h_dir_seg_id_.size(),
        raw_or_null(pool_keys_), raw_or_null(pool_values_),
        raw_or_null(pool_valid_), raw_or_null(seg_bucket_max_),
        raw_or_null(seg_bucket_live_), raw_or_null(seg_bucket_value_sum_),
        raw_or_null(ix.gk), raw_or_null(ix.ins_prefix), ix.ins,
        raw_or_null(ix.gk) + ix.ins, raw_or_null(ix.tomb_val_prefix),
        raw_or_null(ix.tomb_cnt_prefix), ix.u - ix.ins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void cleanup(cudaStream_t stream) {
    join_pending_drain();
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    merge_down(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void maintain(cudaStream_t stream) {
    join_pending_drain();
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    merge_down(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void drain_to_sheet(cudaStream_t stream) {
    join_pending_drain();
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    merge_down(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  std::size_t sheet_live_count() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    return sheet_live_count_;
  }
  std::size_t run_count() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    return h_run_block_.size();
  }
  std::size_t segment_count() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    return h_dir_seg_id_.size();
  }
  std::size_t sheet_bucket_count() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    return h_dir_seg_id_.size() *
           static_cast<std::size_t>(gpulsmopt_detail::kSegmentBuckets);
  }
  std::size_t sheet_capacity() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    return pool_capacity_ *
           static_cast<std::size_t>(gpulsmopt_detail::kSegmentSlots);
  }

  std::size_t gpu_resident_bytes() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    std::size_t total = device_bytes_all(
        run_pool_keys_, run_pool_values_, run_pool_ops_, d_run_block_,
        d_run_boundary_, d_run_size_, d_run_start_, pool_keys_, pool_values_,
        pool_valid_, seg_bucket_max_, seg_bucket_live_, seg_bucket_value_sum_,
        d_dir_seg_id_, d_dir_boundary_, d_dir_prefix_, d_dir_value_sum_,
        d_dir_value_prefix_, scratch_incoming_keys_, scratch_incoming_values_,
        scratch_delete_keys_, scratch_query_found_, seg_cand_seg_, seg_cand_key_,
        seg_cand_value_, seg_cand_op_, seg_cand_seq_, seg_cand_group_sort_);
    for (const auto &g : tier2_)
      total += device_bytes_all(g.keys, g.values, g.ops, g.block_boundary,
                                g.block_start, g.block_size);
    return total;
  }

  DirectorySnapshot directory_snapshot() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    DirectorySnapshot snap;
    snap.seg_id = h_dir_seg_id_;
    snap.boundary = h_dir_boundary_;
    snap.live = h_dir_live_;
    snap.prefix.resize(h_dir_live_.size() + 1);
    std::uint32_t acc = 0;
    snap.prefix[0] = 0;
    for (std::size_t i = 0; i < h_dir_live_.size(); ++i) {
      acc += h_dir_live_[i];
      snap.prefix[i + 1] = acc;
    }
    return snap;
  }

private:
  template <class T> static T *raw_or_null(thrust::device_vector<T> &v) {
    return v.empty() ? nullptr : thrust::raw_pointer_cast(v.data());
  }
  template <class T>
  static const T *raw_or_null(const thrust::device_vector<T> &v) {
    return v.empty() ? nullptr : thrust::raw_pointer_cast(v.data());
  }
  template <class T>
  static std::size_t device_bytes(const thrust::device_vector<T> &v) {
    return v.size() * sizeof(T);
  }
  template <class... Vecs>
  static std::size_t device_bytes_all(const Vecs &...vecs) {
    return (std::size_t{0} + ... + device_bytes(vecs));
  }

  void fill_drain_sort_keys(const thrust::device_vector<std::uint32_t> &keys,
                            const thrust::device_vector<std::uint32_t> &seqs,
                            thrust::device_vector<std::uint64_t> &sort_keys,
                            std::size_t offset, std::size_t count,
                            cudaStream_t stream) {
    if (count == 0)
      return;
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    gpulsmopt_detail::fill_drain_sort_keys_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(keys) + offset, raw_or_null(seqs) + offset,
        raw_or_null(sort_keys) + offset, count);
    CUDA_CHECK(cudaGetLastError());
  }

  void refresh_directory_live_counts(cudaStream_t stream) {
    const std::size_t d = h_dir_seg_id_.size();
    if (d == 0)
      return;
    seg_new_live_.resize(d);
    const int block = 256;
    const int grid = static_cast<int>((d + block - 1) / block);
    gpulsmopt_detail::directory_live_counts_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(d_dir_seg_id_), d, raw_or_null(seg_bucket_live_),
        raw_or_null(seg_new_live_));
    CUDA_CHECK(cudaGetLastError());
    h_dir_live_.resize(d);
    CUDA_CHECK(cudaMemcpyAsync(h_dir_live_.data(), raw_or_null(seg_new_live_),
                               d * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    recompute_sheet_live_count();
    upload_directory(stream);
  }

  void initialize_segmented_storage(cudaStream_t stream) {
    pool_capacity_ = 0;
    free_ids_.clear();
    h_dir_seg_id_.clear();
    h_dir_boundary_.clear();
    h_dir_live_.clear();
    const std::size_t initial =
        max_elements_ == 0 ? 4
                           : 2 * ((max_elements_ + target_segment_live_ - 1) /
                                  target_segment_live_) +
                                 4;
    grow_pool(std::max<std::size_t>(initial, 4));
    reset_directory_to_root(stream);
  }

  void reset_directory_to_root(cudaStream_t stream) {
    free_ids_.clear();
    for (std::size_t id = 0; id < pool_capacity_; ++id) {
      free_ids_.push_back(static_cast<std::uint32_t>(id));
    }
    const std::uint32_t root = alloc_segment();
    reset_segment_storage(root, gpulsmopt_detail::kEmptyKey, stream);
    h_dir_seg_id_ = {root};
    h_dir_boundary_ = {gpulsmopt_detail::kEmptyKey};
    h_dir_live_ = {0u};
    sheet_live_count_ = 0;
    upload_directory(stream);
  }

  void reset_segment_storage(std::uint32_t seg_id, std::uint32_t boundary,
                             cudaStream_t stream) {
    auto policy = thrust::cuda::par.on(stream);
    const std::size_t slot_base =
        static_cast<std::size_t>(seg_id) * gpulsmopt_detail::kSegmentSlots;
    const std::size_t meta_base =
        static_cast<std::size_t>(seg_id) * gpulsmopt_detail::kSegmentBuckets;
    thrust::fill(policy, pool_keys_.begin() + slot_base,
                 pool_keys_.begin() + slot_base +
                     gpulsmopt_detail::kSegmentSlots,
                 gpulsmopt_detail::kEmptyKey);
    thrust::fill(
        policy, pool_values_.begin() + slot_base,
        pool_values_.begin() + slot_base + gpulsmopt_detail::kSegmentSlots, 0u);
    thrust::fill(policy, pool_valid_.begin() + slot_base,
                 pool_valid_.begin() + slot_base +
                     gpulsmopt_detail::kSegmentSlots,
                 std::uint8_t{0});
    thrust::fill(policy, seg_bucket_max_.begin() + meta_base,
                 seg_bucket_max_.begin() + meta_base +
                     gpulsmopt_detail::kSegmentBuckets,
                 boundary);
    thrust::fill(policy, seg_bucket_live_.begin() + meta_base,
                 seg_bucket_live_.begin() + meta_base +
                     gpulsmopt_detail::kSegmentBuckets,
                 0u);
    thrust::fill(policy, seg_bucket_value_sum_.begin() + meta_base,
                 seg_bucket_value_sum_.begin() + meta_base +
                     gpulsmopt_detail::kSegmentBuckets,
                 0u);
  }

  void grow_pool(std::size_t new_capacity) {
    if (new_capacity <= pool_capacity_)
      return;
    pool_keys_.resize(new_capacity * gpulsmopt_detail::kSegmentSlots,
                      gpulsmopt_detail::kEmptyKey);
    pool_values_.resize(new_capacity * gpulsmopt_detail::kSegmentSlots, 0u);
    pool_valid_.resize(new_capacity * gpulsmopt_detail::kSegmentSlots, 0u);
    seg_bucket_max_.resize(new_capacity * gpulsmopt_detail::kSegmentBuckets,
                           gpulsmopt_detail::kEmptyKey);
    seg_bucket_live_.resize(new_capacity * gpulsmopt_detail::kSegmentBuckets,
                            0u);
    seg_bucket_value_sum_.resize(
        new_capacity * gpulsmopt_detail::kSegmentBuckets, 0u);
    for (std::size_t id = pool_capacity_; id < new_capacity; ++id) {
      free_ids_.push_back(static_cast<std::uint32_t>(id));
    }
    pool_capacity_ = new_capacity;
  }

  std::uint32_t alloc_segment() {
    if (free_ids_.empty())
      grow_pool(std::max<std::size_t>(pool_capacity_ * 2, pool_capacity_ + 1));
    const std::uint32_t id = free_ids_.back();
    free_ids_.pop_back();
    return id;
  }

  void free_segment(std::uint32_t id) { free_ids_.push_back(id); }

  void upload_directory(cudaStream_t stream) {
    const std::size_t d = h_dir_seg_id_.size();
    d_dir_seg_id_.resize(d);
    d_dir_boundary_.resize(d);
    d_dir_prefix_.resize(d + 1);
    d_dir_value_sum_.resize(d);
    d_dir_value_prefix_.resize(d + 1);
    std::vector<std::uint32_t> prefix(d + 1);
    std::uint32_t acc = 0;
    prefix[0] = 0;
    for (std::size_t i = 0; i < d; ++i) {
      acc += h_dir_live_[i];
      prefix[i + 1] = acc;
    }
    auto copy = [&](thrust::device_vector<std::uint32_t> &dst,
                    const std::vector<std::uint32_t> &src) {
      if (src.empty())
        return;
      CUDA_CHECK(cudaMemcpyAsync(raw_or_null(dst), src.data(),
                                 src.size() * sizeof(std::uint32_t),
                                 cudaMemcpyHostToDevice, stream));
    };
    copy(d_dir_seg_id_, h_dir_seg_id_);
    copy(d_dir_boundary_, h_dir_boundary_);
    copy(d_dir_prefix_, prefix);
    auto policy = thrust::cuda::par.on(stream);
    thrust::fill(policy, d_dir_value_prefix_.begin(), d_dir_value_prefix_.end(),
                 0u);
    if (d == 0)
      return;
    const int block = 256;
    const int grid = static_cast<int>((d + block - 1) / block);
    gpulsmopt_detail::directory_value_sums_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(d_dir_seg_id_), d, raw_or_null(seg_bucket_value_sum_),
        raw_or_null(d_dir_value_sum_));
    CUDA_CHECK(cudaGetLastError());
    thrust::inclusive_scan(policy, d_dir_value_sum_.begin(),
                           d_dir_value_sum_.end(),
                           d_dir_value_prefix_.begin() + 1);
  }

  template <class T>
  void upload_vec(thrust::device_vector<T> &dst, const std::vector<T> &src,
                  cudaStream_t stream) {
    dst.resize(src.size());
    if (!src.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(raw_or_null(dst), src.data(),
                                 src.size() * sizeof(T), cudaMemcpyHostToDevice,
                                 stream));
    }
  }

  void resize_seg_candidates(std::size_t count) {
    seg_cand_seg_.resize(count);
    seg_cand_key_.resize(count);
    seg_cand_value_.resize(count);
    seg_cand_op_.resize(count);
    seg_cand_seq_.resize(count);
    seg_cand_seq_sort_.resize(count);
    seg_cand_group_sort_.resize(count);
  }

  // Merge insert survivors into the segmented sheet.
  void merge_incoming_into_sheet(std::size_t incoming_count,
                                 cudaStream_t stream) {
    if (incoming_count == 0)
      return;
    auto policy = thrust::cuda::par.on(stream);

    seg_inc_ordinal_.resize(incoming_count);
    {
      const int block = 256;
      const int grid = static_cast<int>((incoming_count + block - 1) / block);
      gpulsmopt_detail::seg_route_keys_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(scratch_incoming_keys_), incoming_count,
          raw_or_null(d_dir_boundary_), h_dir_seg_id_.size(),
          raw_or_null(seg_inc_ordinal_));
      CUDA_CHECK(cudaGetLastError());
    }

    seg_dirty_ord_.resize(incoming_count);
    seg_dirty_count_.resize(incoming_count);
    auto rle_end =
        thrust::reduce_by_key(policy, seg_inc_ordinal_.begin(),
                              seg_inc_ordinal_.begin() + incoming_count,
                              thrust::make_constant_iterator<std::uint32_t>(1u),
                              seg_dirty_ord_.begin(), seg_dirty_count_.begin());
    const std::size_t m =
        static_cast<std::size_t>(rle_end.first - seg_dirty_ord_.begin());

    std::vector<std::uint32_t> dirty_ord(m), dirty_in_count(m);
    CUDA_CHECK(cudaMemcpyAsync(dirty_ord.data(), raw_or_null(seg_dirty_ord_),
                               m * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        dirty_in_count.data(), raw_or_null(seg_dirty_count_),
        m * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<std::uint32_t> dirty_seg_id(m), dirty_old_boundary(m),
        dirty_old_live(m), dirty_in_begin(m), dirty_in_end(m);
    std::uint32_t in_acc = 0;
    for (std::size_t j = 0; j < m; ++j) {
      const std::uint32_t ord = dirty_ord[j];
      dirty_seg_id[j] = h_dir_seg_id_[ord];
      dirty_old_boundary[j] = h_dir_boundary_[ord];
      dirty_old_live[j] = h_dir_live_[ord];
      dirty_in_begin[j] = in_acc;
      in_acc += dirty_in_count[j];
      dirty_in_end[j] = in_acc;
    }
    upload_vec(seg_d_dirty_seg_id_, dirty_seg_id, stream);
    upload_vec(seg_d_dirty_in_begin_, dirty_in_begin, stream);
    upload_vec(seg_d_dirty_in_end_, dirty_in_end, stream);

    seg_slow_.resize(m);
    seg_new_live_.resize(m);
    gpulsmopt_detail::seg_classify_inplace_kernel<<<static_cast<unsigned>(m),
                                                    256, 0, stream>>>(
        raw_or_null(seg_d_dirty_seg_id_), raw_or_null(seg_d_dirty_in_begin_),
        raw_or_null(seg_d_dirty_in_end_), m,
        raw_or_null(scratch_incoming_keys_), raw_or_null(pool_keys_),
        raw_or_null(pool_valid_), raw_or_null(seg_bucket_max_),
        raw_or_null(seg_bucket_live_), raw_or_null(seg_slow_),
        raw_or_null(seg_new_live_));
    CUDA_CHECK(cudaGetLastError());
    std::vector<std::uint32_t> seg_slow(m), seg_new_live(m);
    CUDA_CHECK(cudaMemcpyAsync(seg_slow.data(), raw_or_null(seg_slow_),
                               m * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(seg_new_live.data(), raw_or_null(seg_new_live_),
                               m * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int> is_fast(m), slow_index(m, -1);
    std::vector<std::uint32_t> fast_seg_id, fast_in_begin, fast_in_end;
    std::vector<std::size_t> slow_j;
    for (std::size_t j = 0; j < m; ++j) {
      if (seg_slow[j] == 0) {
        is_fast[j] = 1;
        fast_seg_id.push_back(dirty_seg_id[j]);
        fast_in_begin.push_back(dirty_in_begin[j]);
        fast_in_end.push_back(dirty_in_end[j]);
      } else {
        is_fast[j] = 0;
        slow_index[j] = static_cast<int>(slow_j.size());
        slow_j.push_back(j);
      }
    }

    if (!fast_seg_id.empty()) {
      upload_vec(seg_fast_seg_id_, fast_seg_id, stream);
      upload_vec(seg_fast_in_begin_, fast_in_begin, stream);
      upload_vec(seg_fast_in_end_, fast_in_end, stream);
      gpulsmopt_detail::seg_apply_inplace_kernel<<<
          static_cast<unsigned>(fast_seg_id.size()), 256, 0, stream>>>(
          raw_or_null(seg_fast_seg_id_), raw_or_null(seg_fast_in_begin_),
          raw_or_null(seg_fast_in_end_), fast_seg_id.size(),
          raw_or_null(scratch_incoming_keys_),
          raw_or_null(scratch_incoming_values_), raw_or_null(pool_keys_),
          raw_or_null(pool_values_), raw_or_null(pool_valid_),
          raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_),
          raw_or_null(seg_bucket_value_sum_));
      CUDA_CHECK(cudaGetLastError());
    }

    const std::size_t ms = slow_j.size();
#ifdef GPULSMOPT_PROFILE_INSERT
    printf("[prof]     sheet_merge segs: dirty=%zu fast=%zu slow=%zu "
           "(incoming=%zu)\n",
           m, fast_seg_id.size(), ms, incoming_count);
#endif
    std::vector<std::uint32_t> dirty_k, dirty_output_base, output_seg_id,
        out_boundary, out_live;
    if (ms > 0) {
      std::vector<std::uint32_t> s_seg_id(ms), s_old_boundary(ms),
          s_old_live(ms), s_in_begin(ms), s_in_end(ms), s_cand_off(ms);
      std::uint32_t cand_acc = 0;
      for (std::size_t s = 0; s < ms; ++s) {
        const std::size_t j = slow_j[s];
        s_seg_id[s] = dirty_seg_id[j];
        s_old_boundary[s] = dirty_old_boundary[j];
        s_old_live[s] = dirty_old_live[j];
        s_in_begin[s] = dirty_in_begin[j];
        s_in_end[s] = dirty_in_end[j];
        s_cand_off[s] = cand_acc;
        cand_acc += dirty_old_live[j] + (dirty_in_end[j] - dirty_in_begin[j]);
      }
      const std::size_t candidate_count = cand_acc;
      upload_vec(seg_d_dirty_seg_id_, s_seg_id, stream);
      upload_vec(seg_d_dirty_old_boundary_, s_old_boundary, stream);
      upload_vec(seg_d_dirty_old_live_, s_old_live, stream);
      upload_vec(seg_d_dirty_in_begin_, s_in_begin, stream);
      upload_vec(seg_d_dirty_in_end_, s_in_end, stream);
      upload_vec(seg_d_candidate_offset_, s_cand_off, stream);

      // Gather old and incoming segment records.
      resize_seg_candidates(candidate_count);
      seg_sheet_cursor_.resize(ms);
      thrust::fill(policy, seg_sheet_cursor_.begin(), seg_sheet_cursor_.end(),
                   0u);
      gpulsmopt_detail::seg_gather_candidates_kernel<<<
          static_cast<unsigned>(ms), 256, 0, stream>>>(
          raw_or_null(pool_keys_), raw_or_null(pool_values_),
          raw_or_null(pool_valid_), raw_or_null(seg_d_dirty_seg_id_),
          raw_or_null(seg_d_candidate_offset_),
          raw_or_null(seg_d_dirty_old_live_),
          raw_or_null(seg_d_dirty_in_begin_), raw_or_null(seg_d_dirty_in_end_),
          raw_or_null(scratch_incoming_keys_),
          raw_or_null(scratch_incoming_values_), ms, raw_or_null(seg_cand_seg_),
          raw_or_null(seg_cand_key_), raw_or_null(seg_cand_value_),
          raw_or_null(seg_cand_op_), raw_or_null(seg_cand_seq_),
          raw_or_null(seg_sheet_cursor_));
      CUDA_CHECK(cudaGetLastError());

      std::size_t live_count = 0;
      if (candidate_count > 0) {
        const int block = 256;
        const int grid =
            static_cast<int>((candidate_count + block - 1) / block);
#if GPULSMOPT_DISTINCT_KEYS
#ifdef GPULSMOPT_PROFILE_INSERT
        cudaEvent_t se0, se1;
        cudaEventCreate(&se0);
        cudaEventCreate(&se1);
        cudaEventRecord(se0, stream);
#endif
        gpulsmopt_detail::
            seg_merge_sort_keys_kernel<<<grid, block, 0, stream>>>(
                raw_or_null(seg_cand_seg_), raw_or_null(seg_cand_key_),
                candidate_count, raw_or_null(seg_cand_group_sort_));
        CUDA_CHECK(cudaGetLastError());
        auto payload = thrust::make_zip_iterator(
            thrust::make_tuple(seg_cand_seg_.begin(), seg_cand_key_.begin(),
                               seg_cand_value_.begin()));
        thrust::stable_sort_by_key(
            policy, seg_cand_group_sort_.begin(),
            seg_cand_group_sort_.begin() + candidate_count, payload);
        live_count = candidate_count;
#ifdef GPULSMOPT_PROFILE_INSERT
        cudaEventRecord(se1, stream);
        cudaEventSynchronize(se1);
        float t_slowsort = 0.f;
        cudaEventElapsedTime(&t_slowsort, se0, se1);
        printf("[prof]     seg slow-rebuild sort=%.3f ms (candidates=%zu)\n",
               t_slowsort, candidate_count);
        cudaEventDestroy(se0);
        cudaEventDestroy(se1);
#endif
#else
        gpulsmopt_detail::
            seg_make_resolve_keys_kernel<<<grid, block, 0, stream>>>(
                raw_or_null(seg_cand_seg_), raw_or_null(seg_cand_key_),
                raw_or_null(seg_cand_seq_), candidate_count,
                raw_or_null(seg_cand_seq_sort_),
                raw_or_null(seg_cand_group_sort_));
        CUDA_CHECK(cudaGetLastError());

        auto payload1 = thrust::make_zip_iterator(
            thrust::make_tuple(seg_cand_group_sort_.begin(),
                               seg_cand_seg_.begin(), seg_cand_key_.begin(),
                               seg_cand_value_.begin(), seg_cand_op_.begin()));
        thrust::stable_sort_by_key(policy, seg_cand_seq_sort_.begin(),
                                   seg_cand_seq_sort_.begin() + candidate_count,
                                   payload1);
        auto payload2 = thrust::make_zip_iterator(
            thrust::make_tuple(seg_cand_seg_.begin(), seg_cand_key_.begin(),
                               seg_cand_value_.begin(), seg_cand_op_.begin()));
        thrust::stable_sort_by_key(
            policy, seg_cand_group_sort_.begin(),
            seg_cand_group_sort_.begin() + candidate_count, payload2);
        auto uend = thrust::unique_by_key(
            policy, seg_cand_group_sort_.begin(),
            seg_cand_group_sort_.begin() + candidate_count, payload2);
        const std::size_t unique_count =
            static_cast<std::size_t>(uend.first - seg_cand_group_sort_.begin());
        auto begin = thrust::make_zip_iterator(
            thrust::make_tuple(seg_cand_seg_.begin(), seg_cand_key_.begin(),
                               seg_cand_value_.begin(), seg_cand_op_.begin()));
        auto fend =
            thrust::remove_if(policy, begin, begin + unique_count,
                              gpulsmopt_detail::DrainTombstonePredicate{});
        live_count = static_cast<std::size_t>(fend - begin);
#endif
      }

      seg_dirty_live_.resize(ms);
      thrust::fill(policy, seg_dirty_live_.begin(), seg_dirty_live_.end(), 0u);
      if (live_count > 0) {
        const int block = 256;
        const int grid = static_cast<int>((live_count + block - 1) / block);
        gpulsmopt_detail::
            seg_count_survivors_kernel<<<grid, block, 0, stream>>>(
                raw_or_null(seg_cand_seg_), live_count,
                raw_or_null(seg_dirty_live_));
        CUDA_CHECK(cudaGetLastError());
      }
      std::vector<std::uint32_t> dirty_live(ms);
      CUDA_CHECK(cudaMemcpyAsync(
          dirty_live.data(), raw_or_null(seg_dirty_live_),
          ms * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      dirty_k.resize(ms);
      dirty_output_base.resize(ms);
      std::vector<std::uint32_t> dirty_live_offset(ms);
      std::uint32_t out_acc = 0, live_off = 0;
      for (std::size_t s = 0; s < ms; ++s) {
        const std::uint32_t L = dirty_live[s];
        const std::uint32_t k = std::max<std::uint32_t>(
            1, (L + target_segment_live_ - 1) / target_segment_live_);
        dirty_k[s] = k;
        dirty_output_base[s] = out_acc;
        out_acc += k;
        dirty_live_offset[s] = live_off;
        live_off += L;
      }
      const std::size_t output_count = out_acc;
      output_seg_id.resize(output_count);
      out_boundary.assign(output_count, 0u);
      out_live.resize(output_count);
      std::vector<std::uint32_t> out_dirty(output_count),
          out_local(output_count), out_k(output_count);
      for (std::size_t s = 0; s < ms; ++s) {
        const std::uint32_t k = dirty_k[s];
        const std::uint32_t L = dirty_live[s];
        const std::uint32_t per = k == 0 ? 1u : (L + k - 1) / k;
        for (std::uint32_t local = 0; local < k; ++local) {
          const std::size_t o = dirty_output_base[s] + local;
          output_seg_id[o] = alloc_segment();
          out_dirty[o] = static_cast<std::uint32_t>(s);
          out_local[o] = local;
          out_k[o] = k;
          const std::uint32_t lo = std::min<std::uint32_t>(local * per, L);
          const std::uint32_t hi =
              std::min<std::uint32_t>((local + 1) * per, L);
          out_live[o] = hi - lo;
        }
      }

      upload_vec(seg_d_dirty_live_offset_, dirty_live_offset, stream);
      upload_vec(seg_d_dirty_k_, dirty_k, stream);
      upload_vec(seg_d_dirty_output_base_, dirty_output_base, stream);
      upload_vec(seg_d_output_seg_id_, output_seg_id, stream);
      upload_vec(seg_d_out_dirty_, out_dirty, stream);
      upload_vec(seg_d_out_local_, out_local, stream);
      upload_vec(seg_d_out_k_, out_k, stream);

      seg_d_out_boundary_.resize(output_count);
      if (output_count > 0) {
        const int block = 256;
        const int grid = static_cast<int>((output_count + block - 1) / block);
        gpulsmopt_detail::
            seg_output_boundaries_kernel<<<grid, block, 0, stream>>>(
                raw_or_null(seg_d_out_dirty_), raw_or_null(seg_d_out_local_),
                raw_or_null(seg_d_out_k_), output_count,
                raw_or_null(seg_dirty_live_),
                raw_or_null(seg_d_dirty_live_offset_),
                raw_or_null(seg_d_dirty_old_boundary_),
                raw_or_null(seg_cand_key_), raw_or_null(seg_d_out_boundary_));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(out_boundary.data(),
                                   raw_or_null(seg_d_out_boundary_),
                                   output_count * sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }

      rebuild_output_segments(live_count, output_count, stream);
    }

    std::vector<char> is_dirty(h_dir_seg_id_.size(), 0);
    std::vector<std::size_t> ord_to_j(h_dir_seg_id_.size(), 0);
    for (std::size_t j = 0; j < m; ++j) {
      is_dirty[dirty_ord[j]] = 1;
      ord_to_j[dirty_ord[j]] = j;
    }
    std::vector<std::uint32_t> new_seg_id, new_boundary, new_live;
    new_seg_id.reserve(h_dir_seg_id_.size() + output_seg_id.size());
    new_boundary.reserve(new_seg_id.capacity());
    new_live.reserve(new_seg_id.capacity());
    for (std::size_t ord = 0; ord < h_dir_seg_id_.size(); ++ord) {
      if (!is_dirty[ord]) {
        new_seg_id.push_back(h_dir_seg_id_[ord]);
        new_boundary.push_back(h_dir_boundary_[ord]);
        new_live.push_back(h_dir_live_[ord]);
        continue;
      }
      const std::size_t j = ord_to_j[ord];
      if (is_fast[j]) {
        new_seg_id.push_back(h_dir_seg_id_[ord]);
        new_boundary.push_back(h_dir_boundary_[ord]);
        new_live.push_back(seg_new_live[j]);
        continue;
      }
      const std::size_t s = static_cast<std::size_t>(slow_index[j]);
      for (std::uint32_t local = 0; local < dirty_k[s]; ++local) {
        const std::size_t o = dirty_output_base[s] + local;
        new_seg_id.push_back(output_seg_id[o]);
        new_boundary.push_back(out_boundary[o]);
        new_live.push_back(out_live[o]);
      }
      free_segment(h_dir_seg_id_[ord]);
    }
    h_dir_seg_id_ = std::move(new_seg_id);
    h_dir_boundary_ = std::move(new_boundary);
    h_dir_live_ = std::move(new_live);
    recompute_sheet_live_count();
    upload_directory(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    merge_underfull_segments(stream);
  }

  void rebuild_output_segments(std::size_t live_count, std::size_t output_count,
                               cudaStream_t stream) {
    if (output_count > 0) {
      gpulsmopt_detail::seg_clear_segments_kernel<<<
          static_cast<unsigned>(output_count), 256, 0, stream>>>(
          raw_or_null(seg_d_output_seg_id_), output_count,
          raw_or_null(pool_keys_), raw_or_null(pool_values_),
          raw_or_null(pool_valid_));
      CUDA_CHECK(cudaGetLastError());
    }
    if (live_count > 0) {
      const int block = 256;
      const int grid = static_cast<int>((live_count + block - 1) / block);
      gpulsmopt_detail::
          seg_scatter_survivors_kernel<<<grid, block, 0, stream>>>(
              raw_or_null(seg_cand_seg_), raw_or_null(seg_cand_key_),
              raw_or_null(seg_cand_value_), live_count,
              raw_or_null(seg_dirty_live_),
              raw_or_null(seg_d_dirty_live_offset_),
              raw_or_null(seg_d_dirty_k_),
              raw_or_null(seg_d_dirty_output_base_),
              raw_or_null(seg_d_output_seg_id_), target_fill_,
              raw_or_null(pool_keys_), raw_or_null(pool_values_),
              raw_or_null(pool_valid_));
      CUDA_CHECK(cudaGetLastError());
    }
    if (output_count > 0) {
      gpulsmopt_detail::seg_build_segment_metadata_kernel<<<
          static_cast<unsigned>(output_count), 256, 0, stream>>>(
          raw_or_null(seg_d_output_seg_id_), raw_or_null(seg_d_out_boundary_),
          output_count, raw_or_null(pool_keys_), raw_or_null(pool_values_),
          raw_or_null(pool_valid_), raw_or_null(seg_bucket_max_),
          raw_or_null(seg_bucket_live_), raw_or_null(seg_bucket_value_sum_));
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void merge_underfull_segments(cudaStream_t stream) {
    const std::size_t n = h_dir_seg_id_.size();
    if (n < 2)
      return;
    const std::uint32_t watermark = target_segment_live_ / 2;

    struct Group {
      std::size_t begin; // inclusive
      std::size_t end;   // exclusive
      std::uint32_t live;
    };
    std::vector<Group> groups;
    std::size_t i = 0;
    while (i < n) {
      if (h_dir_live_[i] >= watermark) {
        ++i;
        continue;
      }
      std::size_t j = i;
      std::uint32_t sum = h_dir_live_[i];
      while (j + 1 < n && sum + h_dir_live_[j + 1] <= target_segment_live_) {
        sum += h_dir_live_[j + 1];
        ++j;
      }
      if (j > i) {
        groups.push_back({i, j + 1, sum});
        i = j + 1;
      } else {
        ++i;
      }
    }
    if (groups.empty())
      return;

    auto policy = thrust::cuda::par.on(stream);
    const std::size_t group_count = groups.size();

    std::vector<std::uint32_t> group_src_seg, group_src_begin(group_count),
        group_src_end(group_count), candidate_offset(group_count);
    std::vector<std::uint32_t> dirty_live(group_count),
        dirty_live_offset(group_count), dirty_k(group_count, 1u),
        dirty_output_base(group_count);
    std::vector<std::uint32_t> output_seg_id(group_count),
        out_boundary(group_count);
    std::uint32_t cand_acc = 0, live_off = 0;
    for (std::size_t g = 0; g < group_count; ++g) {
      group_src_begin[g] = static_cast<std::uint32_t>(group_src_seg.size());
      for (std::size_t ord = groups[g].begin; ord < groups[g].end; ++ord) {
        group_src_seg.push_back(h_dir_seg_id_[ord]);
      }
      group_src_end[g] = static_cast<std::uint32_t>(group_src_seg.size());
      candidate_offset[g] = cand_acc;
      cand_acc += groups[g].live;
      dirty_live[g] = groups[g].live;
      dirty_live_offset[g] = live_off;
      live_off += groups[g].live;
      dirty_output_base[g] = static_cast<std::uint32_t>(g);
      output_seg_id[g] = alloc_segment();
      out_boundary[g] = h_dir_boundary_[groups[g].end - 1];
    }
    const std::size_t candidate_count = cand_acc;

    upload_vec(seg_group_src_seg_, group_src_seg, stream);
    upload_vec(seg_group_src_begin_, group_src_begin, stream);
    upload_vec(seg_group_src_end_, group_src_end, stream);
    upload_vec(seg_d_candidate_offset_, candidate_offset, stream);
    upload_vec(seg_dirty_live_, dirty_live, stream);
    upload_vec(seg_d_dirty_live_offset_, dirty_live_offset, stream);
    upload_vec(seg_d_dirty_k_, dirty_k, stream);
    upload_vec(seg_d_dirty_output_base_, dirty_output_base, stream);
    upload_vec(seg_d_output_seg_id_, output_seg_id, stream);
    upload_vec(seg_d_out_boundary_, out_boundary, stream);

    // Gather live records from source segments.
    resize_seg_candidates(candidate_count);
    seg_sheet_cursor_.resize(group_count);
    thrust::fill(policy, seg_sheet_cursor_.begin(), seg_sheet_cursor_.end(),
                 0u);
    if (group_count > 0 && candidate_count > 0) {
      gpulsmopt_detail::seg_merge_gather_kernel<<<
          static_cast<unsigned>(group_count), 256, 0, stream>>>(
          raw_or_null(pool_keys_), raw_or_null(pool_values_),
          raw_or_null(pool_valid_), raw_or_null(seg_group_src_seg_),
          raw_or_null(seg_group_src_begin_), raw_or_null(seg_group_src_end_),
          raw_or_null(seg_d_candidate_offset_), group_count,
          raw_or_null(seg_cand_seg_), raw_or_null(seg_cand_key_),
          raw_or_null(seg_cand_value_), raw_or_null(seg_sheet_cursor_));
      CUDA_CHECK(cudaGetLastError());

      // Order each group by key.
      const int block = 256;
      const int grid = static_cast<int>((candidate_count + block - 1) / block);
      gpulsmopt_detail::seg_merge_sort_keys_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(seg_cand_seg_), raw_or_null(seg_cand_key_),
          candidate_count, raw_or_null(seg_cand_group_sort_));
      CUDA_CHECK(cudaGetLastError());
      auto payload = thrust::make_zip_iterator(
          thrust::make_tuple(seg_cand_seg_.begin(), seg_cand_key_.begin(),
                             seg_cand_value_.begin()));
      thrust::stable_sort_by_key(policy, seg_cand_group_sort_.begin(),
                                 seg_cand_group_sort_.begin() + candidate_count,
                                 payload);
    }

    // Repack into one segment per group
    rebuild_output_segments(candidate_count, group_count, stream);

    std::vector<int> group_of_ord(n, -1);
    for (std::size_t g = 0; g < group_count; ++g) {
      for (std::size_t ord = groups[g].begin; ord < groups[g].end; ++ord) {
        group_of_ord[ord] = static_cast<int>(g);
      }
    }
    std::vector<std::uint32_t> new_seg_id, new_boundary, new_live;
    new_seg_id.reserve(n);
    new_boundary.reserve(n);
    new_live.reserve(n);
    for (std::size_t ord = 0; ord < n; ++ord) {
      const int g = group_of_ord[ord];
      if (g < 0) {
        new_seg_id.push_back(h_dir_seg_id_[ord]);
        new_boundary.push_back(h_dir_boundary_[ord]);
        new_live.push_back(h_dir_live_[ord]);
        continue;
      }
      if (ord == groups[g].begin) {
        new_seg_id.push_back(output_seg_id[g]);
        new_boundary.push_back(out_boundary[g]);
        new_live.push_back(groups[g].live);
        for (std::size_t src = groups[g].begin; src < groups[g].end; ++src) {
          free_segment(h_dir_seg_id_[src]);
        }
      }
    }
    h_dir_seg_id_ = std::move(new_seg_id);
    h_dir_boundary_ = std::move(new_boundary);
    h_dir_live_ = std::move(new_live);
    recompute_sheet_live_count();
    upload_directory(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void recompute_sheet_live_count() {
    std::size_t total = 0;
    for (std::uint32_t v : h_dir_live_)
      total += v;
    sheet_live_count_ = total;
  }

  // Run layer and merge-down state.

  // Run flush budget.
  std::size_t run_flush_budget() const {
    // Keep the threshold below total run capacity.
    const std::size_t cap_total =
        h_run_block_.size() *
        static_cast<std::size_t>(gpulsmopt_detail::kRunCapacity);
    std::size_t budget = std::min<std::size_t>(
        static_cast<std::size_t>(GPULSMOPT_RUN_FLUSH_BUDGET), cap_total * 3 / 4);
    if (max_elements_ > 0)
      budget = std::min(budget, max_elements_);
    return std::max<std::size_t>(budget, std::size_t{1} << 16);
  }

  void grow_run_pool(std::size_t new_capacity) {
    if (new_capacity <= run_pool_capacity_)
      return;
    run_pool_keys_.resize(new_capacity * gpulsmopt_detail::kRunCapacity,
                          gpulsmopt_detail::kEmptyKey);
    run_pool_values_.resize(new_capacity * gpulsmopt_detail::kRunCapacity, 0u);
    run_pool_ops_.resize(new_capacity * gpulsmopt_detail::kRunCapacity,
                         static_cast<std::uint8_t>(gpulsmopt_detail::kInsert));
    for (std::size_t id = run_pool_capacity_; id < new_capacity; ++id)
      run_free_ids_.push_back(static_cast<std::uint32_t>(id));
    run_pool_capacity_ = new_capacity;
  }

  std::uint32_t alloc_run_block() {
    if (run_free_ids_.empty())
      grow_run_pool(
          std::max<std::size_t>(run_pool_capacity_ * 2, run_pool_capacity_ + 1));
    const std::uint32_t id = run_free_ids_.back();
    run_free_ids_.pop_back();
    return id;
  }

  void free_run_block(std::uint32_t id) { run_free_ids_.push_back(id); }

  void upload_run_directory(cudaStream_t stream) {
    const std::size_t r = h_run_block_.size();
    std::vector<std::uint32_t> starts(r);
    for (std::size_t i = 0; i < r; ++i)
      starts[i] = h_run_block_[i] *
                  static_cast<std::uint32_t>(gpulsmopt_detail::kRunCapacity);
    upload_vec(d_run_block_, h_run_block_, stream);
    upload_vec(d_run_boundary_, h_run_boundary_, stream);
    upload_vec(d_run_size_, h_run_size_, stream);
    upload_vec(d_run_start_, starts, stream);
  }

  void init_run_layer(cudaStream_t stream) {
    run_target_fill_ = std::max<std::uint32_t>(
        1u, static_cast<std::uint32_t>(gpulsmopt_detail::kRunCapacity / 2));
    run_pool_capacity_ = 0;
    run_free_ids_.clear();
    run_pool_keys_.clear();
    run_pool_values_.clear();
    run_pool_ops_.clear();
    tier2_.clear();
    grow_run_pool(4);
    const std::uint32_t block = alloc_run_block();
    h_run_block_ = {block};
    h_run_boundary_ = {gpulsmopt_detail::kEmptyKey};
    h_run_size_ = {0u};
    run_live_total_ = 0;
    upload_run_directory(stream);
  }

  // Append records to the run layer. UseB=true is Method B (sort once + slice),
  // false is Method A (scan the batch per run). Both are always compiled; the
  // auto route picks per batch size (see insert_records).
  template <bool UseB>
  void insert_records_impl(const std::uint32_t *keys_in,
                           const std::uint32_t *values_in, std::uint8_t op,
                           std::size_t count, cudaStream_t stream) {
    if (count == 0)
      return;
    const std::uint32_t *keys = keys_in;
    const std::uint32_t *vals = values_in ? values_in : keys_in; // tomb val unused
    const std::size_t r = h_run_block_.size();
    std::vector<std::uint32_t> h_counts(r);

    // Cross-section state (only the chosen path populates its own).
    [[maybe_unused]] thrust::device_vector<std::uint32_t> routeb_keys,
        routeb_vals;                                            // Method B batch
    [[maybe_unused]] thrust::device_vector<std::uint32_t> d_begin, d_end; // slices
    [[maybe_unused]] std::vector<std::uint32_t> h_begin;        // Method B starts
    [[maybe_unused]] thrust::device_vector<std::uint32_t> gk_tmp,
        gv_tmp; // Method A gather buffer

    if constexpr (UseB) {
      // Method B: sort once, then compute run slices.
      {
        auto pol = thrust::cuda::par.on(stream);
        routeb_keys.assign(thrust::device_pointer_cast(keys),
                           thrust::device_pointer_cast(keys) + count);
        routeb_vals.assign(thrust::device_pointer_cast(vals),
                           thrust::device_pointer_cast(vals) + count);
        thrust::sort_by_key(pol, routeb_keys.begin(), routeb_keys.end(),
                            routeb_vals.begin());
        keys = raw_or_null(routeb_keys);
        vals = raw_or_null(routeb_vals);
      }
      d_begin.resize(r);
      d_end.resize(r);
      {
        const int block = 256;
        const int grid = static_cast<int>((r + block - 1) / block);
        gpulsmopt_detail::run_slice_bounds_kernel<<<grid, block, 0, stream>>>(
            keys, count, raw_or_null(d_run_boundary_), r, raw_or_null(d_begin),
            raw_or_null(d_end));
        CUDA_CHECK(cudaGetLastError());
      }
      h_begin.resize(r);
      std::vector<std::uint32_t> h_end(r);
      CUDA_CHECK(cudaMemcpyAsync(h_begin.data(), raw_or_null(d_begin),
                                 r * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_end.data(), raw_or_null(d_end),
                                 r * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      for (std::size_t i = 0; i < r; ++i)
        h_counts[i] = h_end[i] - h_begin[i];
    } else {
      // Method A: scan the whole batch per run.
      ins_counts_.assign(r, 0u);
      gpulsmopt_detail::rl_run_count_kernel<<<static_cast<int>(r), 256, 0,
                                              stream>>>(
          keys, count, raw_or_null(d_run_boundary_), r,
          raw_or_null(ins_counts_));
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpyAsync(h_counts.data(), raw_or_null(ins_counts_),
                                 r * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<std::uint8_t> is_fast(r);
    for (std::size_t i = 0; i < r; ++i)
      is_fast[i] =
          (h_run_size_[i] + h_counts[i] <=
           static_cast<std::uint32_t>(gpulsmopt_detail::kRunCapacity))
              ? 1u
              : 0u;

    // Fast path: append to runs with room.
    {
      ins_is_fast_.assign(is_fast.begin(), is_fast.end());
      ins_oldsize_.assign(h_run_size_.begin(), h_run_size_.end());
      if constexpr (UseB) {
        gpulsmopt_detail::rl_pull_fast_kernel<<<static_cast<int>(r), 128, 0,
                                                stream>>>(
            raw_or_null(ins_is_fast_), raw_or_null(d_begin), raw_or_null(d_end),
            raw_or_null(d_run_start_), raw_or_null(ins_oldsize_), keys, vals, op,
            raw_or_null(run_pool_keys_), raw_or_null(run_pool_values_),
            raw_or_null(run_pool_ops_));
      } else {
        gpulsmopt_detail::rl_run_append_kernel<<<static_cast<int>(r), 256, 0,
                                                 stream>>>(
            raw_or_null(ins_is_fast_), raw_or_null(d_run_boundary_),
            raw_or_null(d_run_start_), raw_or_null(ins_oldsize_), r, keys, vals,
            op, count, raw_or_null(run_pool_keys_), raw_or_null(run_pool_values_),
            raw_or_null(run_pool_ops_));
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    for (std::size_t i = 0; i < r; ++i)
      if (is_fast[i])
        h_run_size_[i] += h_counts[i];

    // Slow path: rebuild overflowing runs.
    std::vector<std::uint32_t> nb, nbd, nsz;
    nb.reserve(r);
    nbd.reserve(r);
    nsz.reserve(r);
    bool any_slow = false; // did any run split/rebuild (blocks/boundaries move)?
    for (std::size_t i = 0; i < r; ++i) {
      if (is_fast[i]) {
        nb.push_back(h_run_block_[i]);
        nbd.push_back(h_run_boundary_[i]);
        nsz.push_back(h_run_size_[i]);
        continue;
      }
      any_slow = true;
      if constexpr (UseB) {
        rebuild_run_contiguous(i, keys + h_begin[i], vals + h_begin[i], op,
                               h_counts[i], stream, nb, nbd, nsz);
      } else {
        // Method A overflow rebuild.
        const std::uint32_t inc = h_counts[i];
        gk_tmp.resize(inc);
        gv_tmp.resize(inc);
        if (inc > 0) {
          const std::uint32_t hi = h_run_boundary_[i];
          const bool has_lo = i > 0;
          const std::uint32_t lo = has_lo ? h_run_boundary_[i - 1] : 0u;
          gpulsmopt_detail::rl_run_gather_kernel<<<1, 256, 0, stream>>>(
              keys, vals, count, lo, hi, has_lo ? 1u : 0u, raw_or_null(gk_tmp),
              raw_or_null(gv_tmp));
          CUDA_CHECK(cudaGetLastError());
        }
        rebuild_run_contiguous(i, raw_or_null(gk_tmp), raw_or_null(gv_tmp), op,
                               inc, stream, nb, nbd, nsz);
      }
      free_run_block(h_run_block_[i]);
    }
    h_run_block_ = std::move(nb);
    h_run_boundary_ = std::move(nbd);
    h_run_size_ = std::move(nsz);
    run_live_total_ = 0;
    for (auto s : h_run_size_)
      run_live_total_ += s;
    // Fast path (nothing split): blocks/boundaries/starts are unchanged on the
    // device — only sizes moved, so skip 3 of the 4 directory H->D copies.
    if (any_slow)
      upload_run_directory(stream);
    else
      upload_vec(d_run_size_, h_run_size_, stream);
    overlay_dirty_ = true; // overlay changed; invalidate the cached read index
  }

  // Route by batch size: Method A for small batches (skip the sort), Method B
  // for large. GPULSMOPT_ROUTE_METHOD=0/1 forces A/B; default (2) is auto at
  // the GPULSMOPT_ROUTE_AUTO_LIMIT (2^20) crossover.
  void insert_records(const std::uint32_t *keys_in,
                      const std::uint32_t *values_in, std::uint8_t op,
                      std::size_t count, cudaStream_t stream) {
#if GPULSMOPT_ROUTE_METHOD == 1
    insert_records_impl<true>(keys_in, values_in, op, count, stream);
#elif GPULSMOPT_ROUTE_METHOD == 0
    insert_records_impl<false>(keys_in, values_in, op, count, stream);
#else
    if (count >= static_cast<std::size_t>(GPULSMOPT_ROUTE_AUTO_LIMIT))
      insert_records_impl<true>(keys_in, values_in, op, count, stream);
    else
      insert_records_impl<false>(keys_in, values_in, op, count, stream);
#endif
  }

  // Copy one run into a fresh block.
  void emit_run(thrust::device_vector<std::uint32_t> &gk,
                thrust::device_vector<std::uint32_t> &gv,
                thrust::device_vector<std::uint8_t> &gop, std::size_t off,
                std::uint32_t len, std::uint32_t boundary,
                std::vector<std::uint32_t> &nb, std::vector<std::uint32_t> &nbd,
                std::vector<std::uint32_t> &nsz, cudaStream_t stream) {
    auto policy = thrust::cuda::par.on(stream);
    const std::uint32_t block = alloc_run_block();
    const std::size_t dst =
        static_cast<std::size_t>(block) * gpulsmopt_detail::kRunCapacity;
    thrust::copy(policy, gk.begin() + off, gk.begin() + off + len,
                 run_pool_keys_.begin() + dst);
    thrust::copy(policy, gv.begin() + off, gv.begin() + off + len,
                 run_pool_values_.begin() + dst);
    thrust::copy(policy, gop.begin() + off, gop.begin() + off + len,
                 run_pool_ops_.begin() + dst);
    nb.push_back(block);
    nbd.push_back(boundary);
    nsz.push_back(len);
  }

  // Split combined entries into bounded runs.
  void split_entries_into_runs(thrust::device_vector<std::uint32_t> &gk,
                               thrust::device_vector<std::uint32_t> &gv,
                               thrust::device_vector<std::uint8_t> &gop,
                               std::size_t m, std::uint32_t top_boundary,
                               std::vector<std::uint32_t> &nb,
                               std::vector<std::uint32_t> &nbd,
                               std::vector<std::uint32_t> &nsz,
                               cudaStream_t stream) {
    if (m == 0)
      return;
    const std::uint32_t target = run_target_fill_;
    if (m <= target) {
      emit_run(gk, gv, gop, 0, static_cast<std::uint32_t>(m), top_boundary, nb,
               nbd, nsz, stream);
      return;
    }
    auto policy = thrust::cuda::par.on(stream);
    const std::size_t k = (m + target - 1) / target;

    // Sampled quantile pivots.
    const std::size_t s = std::min(m, std::max<std::size_t>(k * 32, 256));
    thrust::device_vector<std::uint32_t> samp(s);
    {
      const int block = 256;
      const int grid = static_cast<int>((s + block - 1) / block);
      gpulsmopt_detail::sample_gather_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(gk), m, raw_or_null(samp), s);
      CUDA_CHECK(cudaGetLastError());
    }
    thrust::sort(policy, samp.begin(), samp.end());
    std::vector<std::uint32_t> hsamp(s);
    CUDA_CHECK(cudaMemcpyAsync(hsamp.data(), raw_or_null(samp),
                               s * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::uint32_t> hB(k);
    for (std::size_t c = 0; c + 1 < k; ++c) {
      std::size_t pos = ((c + 1) * s) / k;
      if (pos >= s)
        pos = s - 1;
      std::uint32_t v = hsamp[pos];
      if (v > top_boundary)
        v = top_boundary;
      if (c > 0 && v < hB[c - 1])
        v = hB[c - 1];
      hB[c] = v;
    }
    hB[k - 1] = top_boundary;
    thrust::device_vector<std::uint32_t> dB(hB);

    thrust::device_vector<std::uint32_t> bucket(m);
    {
      const int block = 256;
      const int grid = static_cast<int>((m + block - 1) / block);
      gpulsmopt_detail::bucket_assign_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(gk), m, raw_or_null(dB), k, raw_or_null(bucket));
      CUDA_CHECK(cudaGetLastError());
    }
    thrust::device_vector<std::uint32_t> bcount(k, 0u);
    {
      const int block = 256;
      const int grid = static_cast<int>((m + block - 1) / block);
      gpulsmopt_detail::rl_histogram_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(bucket), m, raw_or_null(bcount));
      CUDA_CHECK(cudaGetLastError());
    }
    std::vector<std::uint32_t> hcount(k);
    CUDA_CHECK(cudaMemcpyAsync(hcount.data(), raw_or_null(bcount),
                               k * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::uint32_t maxc = 0;
    for (auto c : hcount)
      maxc = std::max(maxc, c);
    if (maxc > static_cast<std::uint32_t>(gpulsmopt_detail::kRunCapacity)) {
      sort_split_entries(gk, gv, gop, m, top_boundary, nb, nbd, nsz, stream);
      return;
    }

    // Stable group by bucket.
    auto payload = thrust::make_zip_iterator(
        thrust::make_tuple(gk.begin(), gv.begin(), gop.begin()));
    thrust::stable_sort_by_key(policy, bucket.begin(), bucket.end(), payload);

    std::size_t off = 0;
    std::size_t last_emitted = static_cast<std::size_t>(-1);
    for (std::size_t c = 0; c < k; ++c) {
      const std::uint32_t len = hcount[c];
      if (len == 0)
        continue;
      emit_run(gk, gv, gop, off, len, hB[c], nb, nbd, nsz, stream);
      off += len;
      last_emitted = nb.size() - 1;
    }
    if (last_emitted != static_cast<std::size_t>(-1))
      nbd[last_emitted] = top_boundary; // highest run is the catch-all
  }

  // Sort/dedup fallback split.
  void sort_split_entries(thrust::device_vector<std::uint32_t> &gk,
                          thrust::device_vector<std::uint32_t> &gv,
                          thrust::device_vector<std::uint8_t> &gop,
                          std::size_t m, std::uint32_t top_boundary,
                          std::vector<std::uint32_t> &nb,
                          std::vector<std::uint32_t> &nbd,
                          std::vector<std::uint32_t> &nsz, cudaStream_t stream) {
    auto policy = thrust::cuda::par.on(stream);
    thrust::device_vector<std::uint32_t> arr(m);
    thrust::sequence(policy, arr.begin(), arr.end());
    thrust::device_vector<std::uint64_t> sortk(m);
    fill_drain_sort_keys(gk, arr, sortk, 0, m, stream);
    auto payload = thrust::make_zip_iterator(
        thrust::make_tuple(gk.begin(), gv.begin(), gop.begin()));
    thrust::sort_by_key(policy, sortk.begin(), sortk.end(), payload);
    auto uend = thrust::unique_by_key(
        policy, gk.begin(), gk.begin() + m,
        thrust::make_zip_iterator(thrust::make_tuple(gv.begin(), gop.begin())));
    const std::size_t u = static_cast<std::size_t>(uend.first - gk.begin());
    if (u == 0)
      return;
    const std::uint32_t target = run_target_fill_;
    const std::size_t k = (u + target - 1) / target;
    std::vector<std::uint32_t> host_keys(u);
    CUDA_CHECK(cudaMemcpyAsync(host_keys.data(), raw_or_null(gk),
                               u * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t c = 0; c < k; ++c) {
      const std::size_t cs = c * target;
      const std::size_t ce =
          std::min((c + 1) * static_cast<std::size_t>(target), u);
      const std::uint32_t bnd = (c + 1 == k) ? top_boundary : host_keys[ce - 1];
      emit_run(gk, gv, gop, cs, static_cast<std::uint32_t>(ce - cs), bnd, nb,
               nbd, nsz, stream);
    }
  }

  // Rebuild one overflowing method-B run.
  void rebuild_run_contiguous(std::size_t i, const std::uint32_t *in_keys,
                              const std::uint32_t *in_vals, std::uint8_t op,
                              std::uint32_t inc, cudaStream_t stream,
                              std::vector<std::uint32_t> &nb,
                              std::vector<std::uint32_t> &nbd,
                              std::vector<std::uint32_t> &nsz) {
    auto policy = thrust::cuda::par.on(stream);
    const std::uint32_t old_size = h_run_size_[i];
    const std::uint32_t old_boundary = h_run_boundary_[i];
    const std::size_t m = static_cast<std::size_t>(old_size) + inc;
    const std::size_t base =
        static_cast<std::size_t>(h_run_block_[i]) * gpulsmopt_detail::kRunCapacity;
    thrust::device_vector<std::uint32_t> gk(m), gv(m);
    thrust::device_vector<std::uint8_t> gop(m);
    if (old_size > 0) {
      thrust::copy(policy, run_pool_keys_.begin() + base,
                   run_pool_keys_.begin() + base + old_size, gk.begin());
      thrust::copy(policy, run_pool_values_.begin() + base,
                   run_pool_values_.begin() + base + old_size, gv.begin());
      thrust::copy(policy, run_pool_ops_.begin() + base,
                   run_pool_ops_.begin() + base + old_size, gop.begin());
    }
    if (inc > 0) {
      auto kptr = thrust::device_pointer_cast(in_keys);
      auto vptr = thrust::device_pointer_cast(in_vals);
      thrust::copy(policy, kptr, kptr + inc, gk.begin() + old_size);
      thrust::copy(policy, vptr, vptr + inc, gv.begin() + old_size);
      thrust::fill(policy, gop.begin() + old_size, gop.end(), op);
    }
    split_entries_into_runs(gk, gv, gop, m, old_boundary, nb, nbd, nsz, stream);
  }

  // Move live runs into a tier-2 generation.
  void flush_to_tier2(cudaStream_t stream) {
    const std::size_t r = h_run_block_.size();
    std::size_t total = 0;
    for (auto s : h_run_size_)
      total += s;
    if (total == 0)
      return;
    auto policy = thrust::cuda::par.on(stream);
    Tier2Gen gen;
    gen.keys.resize(total);
    gen.values.resize(total);
    gen.ops.resize(total);
    std::vector<std::uint32_t> bnd, start, sz;
    std::size_t off = 0;
    for (std::size_t i = 0; i < r; ++i) {
      const std::uint32_t s = h_run_size_[i];
      if (s == 0)
        continue;
      const std::size_t bpos =
          static_cast<std::size_t>(h_run_block_[i]) * gpulsmopt_detail::kRunCapacity;
      thrust::copy(policy, run_pool_keys_.begin() + bpos,
                   run_pool_keys_.begin() + bpos + s, gen.keys.begin() + off);
      thrust::copy(policy, run_pool_values_.begin() + bpos,
                   run_pool_values_.begin() + bpos + s, gen.values.begin() + off);
      thrust::copy(policy, run_pool_ops_.begin() + bpos,
                   run_pool_ops_.begin() + bpos + s, gen.ops.begin() + off);
      bnd.push_back(h_run_boundary_[i]);
      start.push_back(static_cast<std::uint32_t>(off));
      sz.push_back(s);
      off += s;
    }
    upload_vec(gen.block_boundary, bnd, stream);
    upload_vec(gen.block_start, start, stream);
    upload_vec(gen.block_size, sz, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    tier2_.push_back(std::move(gen));
    std::fill(h_run_size_.begin(), h_run_size_.end(), 0u);
    run_live_total_ = 0;
    upload_run_directory(stream);
  }

  // Resolved overlay snapshot for direct reads.
  struct OverlayReadIndex {
    thrust::device_vector<std::uint32_t> gk, gv;
    thrust::device_vector<std::uint8_t> gop;
    std::size_t u = 0, ins = 0;
    thrust::device_vector<std::uint32_t> ins_prefix;      // size ins+1
    thrust::device_vector<std::uint32_t> tomb_val_prefix; // size (u-ins)+1
    thrust::device_vector<std::uint32_t> tomb_cnt_prefix; // size (u-ins)+1
  };

  // Resolve overlay entries newest-first by key.
  void resolve_overlay(thrust::device_vector<std::uint32_t> &gk,
                       thrust::device_vector<std::uint32_t> &gv,
                       thrust::device_vector<std::uint8_t> &gop, std::size_t &u,
                       std::size_t &ins, cudaStream_t stream) {
    u = 0;
    ins = 0;
    std::size_t total = run_live_total_;
    for (auto &g : tier2_)
      total += g.keys.size();
    if (total == 0) {
      gk.clear();
      gv.clear();
      gop.clear();
      return;
    }
    auto policy = thrust::cuda::par.on(stream);
    gk.resize(total);
    gv.resize(total);
    gop.resize(total);
    std::size_t off = 0;
    for (auto &g : tier2_) { // oldest generation first -> lowest arrival
      const std::size_t s = g.keys.size();
      if (s) {
        thrust::copy(policy, g.keys.begin(), g.keys.end(), gk.begin() + off);
        thrust::copy(policy, g.values.begin(), g.values.end(), gv.begin() + off);
        thrust::copy(policy, g.ops.begin(), g.ops.end(), gop.begin() + off);
      }
      off += s;
    }
    const std::size_t r = h_run_block_.size(); // run layer newest -> highest arr
    for (std::size_t i = 0; i < r; ++i) {
      const std::uint32_t s = h_run_size_[i];
      if (s == 0)
        continue;
      const std::size_t bpos =
          static_cast<std::size_t>(h_run_block_[i]) * gpulsmopt_detail::kRunCapacity;
      thrust::copy(policy, run_pool_keys_.begin() + bpos,
                   run_pool_keys_.begin() + bpos + s, gk.begin() + off);
      thrust::copy(policy, run_pool_values_.begin() + bpos,
                   run_pool_values_.begin() + bpos + s, gv.begin() + off);
      thrust::copy(policy, run_pool_ops_.begin() + bpos,
                   run_pool_ops_.begin() + bpos + s, gop.begin() + off);
      off += s;
    }
    thrust::device_vector<std::uint32_t> garr(total);
    thrust::sequence(policy, garr.begin(), garr.end());
    thrust::device_vector<std::uint64_t> sortk(total);
    fill_drain_sort_keys(gk, garr, sortk, 0, total, stream);
    auto payload = thrust::make_zip_iterator(
        thrust::make_tuple(gk.begin(), gv.begin(), gop.begin()));
    thrust::sort_by_key(policy, sortk.begin(), sortk.end(), payload);
    auto uend = thrust::unique_by_key(
        policy, gk.begin(), gk.end(),
        thrust::make_zip_iterator(thrust::make_tuple(gv.begin(), gop.begin())));
    u = static_cast<std::size_t>(uend.first - gk.begin());
    auto beg = thrust::make_zip_iterator(
        thrust::make_tuple(gk.begin(), gv.begin(), gop.begin()));
    auto mid = thrust::stable_partition(policy, beg, beg + u,
                                        gpulsmopt_detail::TupleOpIsInsert{});
    ins = static_cast<std::size_t>(mid - beg);
  }

  // Build the overlay read index.
  void build_overlay_read_index(OverlayReadIndex &ix, cudaStream_t stream) {
    resolve_overlay(ix.gk, ix.gv, ix.gop, ix.u, ix.ins, stream);
    auto policy = thrust::cuda::par.on(stream);
    const std::size_t ins = ix.ins, u = ix.u, tomb = u - ins;
    ix.ins_prefix.assign(ins + 1, 0u);
    if (ins > 0)
      thrust::inclusive_scan(policy, ix.gv.begin(), ix.gv.begin() + ins,
                             ix.ins_prefix.begin() + 1);
    ix.tomb_val_prefix.assign(tomb + 1, 0u);
    ix.tomb_cnt_prefix.assign(tomb + 1, 0u);
    if (tomb > 0 && h_dir_seg_id_.size() > 0) {
      thrust::device_vector<std::uint32_t> tval(tomb), tflag(tomb);
      const int block = 256;
      const int grid = static_cast<int>((tomb + block - 1) / block);
      gpulsmopt_detail::sheet_point_values_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(ix.gk) + ins, tomb, raw_or_null(d_dir_boundary_),
          raw_or_null(d_dir_seg_id_), h_dir_seg_id_.size(),
          raw_or_null(pool_keys_), raw_or_null(pool_values_),
          raw_or_null(pool_valid_), raw_or_null(seg_bucket_max_),
          raw_or_null(tval), raw_or_null(tflag));
      CUDA_CHECK(cudaGetLastError());
      thrust::inclusive_scan(policy, tval.begin(), tval.end(),
                             ix.tomb_val_prefix.begin() + 1);
      thrust::inclusive_scan(policy, tflag.begin(), tflag.end(),
                             ix.tomb_cnt_prefix.begin() + 1);
    }
  }

  // Lazily-rebuilt resolved overlay, shared by every read (lookup/count/range/
  // successor) between mutations. Any insert/delete/drain sets overlay_dirty_,
  // so the expensive resolve+index (sort of the whole overlay) runs at most once
  // per mutation instead of once per read. The rebuild happens here on the read
  // side; mutators pay only a bool store. NOTE: assumes reads are sequential
  // (single-reader) — like the rest of the read scratch, concurrent readers
  // would race the rebuild; the mutation lock already serializes writers.
  OverlayReadIndex &overlay_index(cudaStream_t stream) {
    if (overlay_dirty_) {
      build_overlay_read_index(cached_overlay_, stream);
      overlay_dirty_ = false;
    }
    return cached_overlay_;
  }

  // Reset the run layer to one empty root run.
  void reset_run_layer_to_root(cudaStream_t stream) {
    for (std::uint32_t b : h_run_block_)
      run_free_ids_.push_back(b);
    const std::uint32_t block = alloc_run_block();
    h_run_block_ = {block};
    h_run_boundary_ = {gpulsmopt_detail::kEmptyKey};
    h_run_size_ = {0u};
    run_live_total_ = 0;
    upload_run_directory(stream);
  }

  // Clear run sizes while keeping boundaries.
  void clear_run_layer_keep_boundaries(cudaStream_t stream) {
    if (h_run_block_.empty()) {
      reset_run_layer_to_root(stream);
      return;
    }
    std::fill(h_run_size_.begin(), h_run_size_.end(), 0u);
    run_live_total_ = 0;
    upload_run_directory(stream);
  }

  // Merge overlay entries into the segmented sheet.
  void merge_down(cudaStream_t stream) {
#ifdef GPULSMOPT_PROFILE_INSERT
    cudaEvent_t pe0, pe1, pe2, pe3;
    cudaEventCreate(&pe0);
    cudaEventCreate(&pe1);
    cudaEventCreate(&pe2);
    cudaEventCreate(&pe3);
    cudaEventRecord(pe0, stream);
#endif
    thrust::device_vector<std::uint32_t> gk, gv;
    thrust::device_vector<std::uint8_t> gop;
    std::size_t u = 0, ins = 0;
    resolve_overlay(gk, gv, gop, u, ins, stream);
#ifdef GPULSMOPT_PROFILE_INSERT
    cudaEventRecord(pe1, stream);
    cudaEventRecord(pe2, stream); // default: tomb phase empty
    cudaEventRecord(pe3, stream); // default: incoming phase empty
#endif
    if (u > 0) {
      auto policy = thrust::cuda::par.on(stream);
      const std::size_t tomb = u - ins;
      if (tomb > 0 && h_dir_seg_id_.size() > 0) {
        scratch_delete_keys_.resize(tomb);
        thrust::copy(policy, gk.begin() + ins, gk.begin() + u,
                     scratch_delete_keys_.begin());
        const int block = 256;
        if (tomb >= gpulsmopt_detail::kStDeleteLimit) {
          // ST (thread/key): no lane-0 routing waste; wins for large tombstone
          // batches (the usual drain case).
          const int grid = static_cast<int>((tomb + block - 1) / block);
          gpulsmopt_detail::seg_delete_keys_kernel<<<grid, block, 0, stream>>>(
              raw_or_null(scratch_delete_keys_), tomb,
              raw_or_null(d_dir_boundary_), raw_or_null(d_dir_seg_id_),
              h_dir_seg_id_.size(), raw_or_null(pool_keys_),
              raw_or_null(pool_values_), raw_or_null(pool_valid_),
              raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_),
              raw_or_null(seg_bucket_value_sum_));
        } else {
          // Warp/key: better occupancy for small tombstone batches.
          const int warps_per_block = block / 32;
          const int grid =
              static_cast<int>((tomb + warps_per_block - 1) / warps_per_block);
          gpulsmopt_detail::seg_delete_keys_warp_kernel<<<grid, block, 0,
                                                          stream>>>(
              raw_or_null(scratch_delete_keys_), tomb,
              raw_or_null(d_dir_boundary_), raw_or_null(d_dir_seg_id_),
              h_dir_seg_id_.size(), raw_or_null(pool_keys_),
              raw_or_null(pool_values_), raw_or_null(pool_valid_),
              raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_),
              raw_or_null(seg_bucket_value_sum_));
        }
        CUDA_CHECK(cudaGetLastError());
        refresh_directory_live_counts(stream);
      }
#ifdef GPULSMOPT_PROFILE_INSERT
      cudaEventRecord(pe2, stream);
#endif
      if (ins > 0) {
        scratch_incoming_keys_.resize(ins);
        scratch_incoming_values_.resize(ins);
        thrust::copy(policy, gk.begin(), gk.begin() + ins,
                     scratch_incoming_keys_.begin());
        thrust::copy(policy, gv.begin(), gv.begin() + ins,
                     scratch_incoming_values_.begin());
        merge_incoming_into_sheet(ins, stream);
      }
#ifdef GPULSMOPT_PROFILE_INSERT
      cudaEventRecord(pe3, stream);
#endif
    }
    tier2_.clear();
    clear_run_layer_keep_boundaries(stream);
    overlay_dirty_ = true; // overlay drained to sheet; cached read index stale
#ifdef GPULSMOPT_PROFILE_INSERT
    cudaEventSynchronize(pe3);
    float t_res = 0.f, t_tomb = 0.f, t_inc = 0.f;
    cudaEventElapsedTime(&t_res, pe0, pe1);
    cudaEventElapsedTime(&t_tomb, pe1, pe2);
    cudaEventElapsedTime(&t_inc, pe2, pe3);
    printf("[prof]   merge_down phases: resolve=%.3f tomb_delete=%.3f "
           "sheet_merge=%.3f ms (u=%zu ins=%zu)\n",
           t_res, t_tomb, t_inc, u, ins);
    cudaEventDestroy(pe0);
    cudaEventDestroy(pe1);
    cudaEventDestroy(pe2);
    cudaEventDestroy(pe3);
#endif
  }

  // Wait for any in-flight drain.
  void join_pending_drain() const {
    if (pending_drain_.valid())
      pending_drain_.get(); // waits and rethrows any worker-side exception
  }

  // Launch merge_down on a worker thread.
  void launch_background_drain() {
    pending_drain_ = std::async(std::launch::async, [this]() {
      std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
      // Serialize worker work on the default stream.
      merge_down(0);
      CUDA_CHECK(cudaStreamSynchronize(0));
    });
  }

  // Decide whether the run layer should drain.
  bool drain_needed_locked(cudaStream_t stream) {
#if GPULSMOPT_ENABLE_TIER2
    maybe_flush_and_merge(stream);
    return false;
#else
    (void)stream;
    return run_live_total_ >= run_flush_budget();
#endif
  }

  void maybe_flush_and_merge(cudaStream_t stream) {
#if GPULSMOPT_ENABLE_TIER2
    // Tier-2 path.
    if (run_live_total_ >= run_flush_budget())
      flush_to_tier2(stream);
    if (tier2_.size() >= gpulsmopt_detail::kTier2Fanout)
      merge_down(stream);
#else
    // Direct sheet drain.
    if (run_live_total_ < run_flush_budget())
      return;
#ifdef GPULSMOPT_PROFILE_INSERT
    cudaEvent_t pa, pb;
    cudaEventCreate(&pa);
    cudaEventCreate(&pb);
    cudaEventRecord(pa, stream);
    merge_down(stream);
    cudaEventRecord(pb, stream);
    cudaEventSynchronize(pb);
    float t_merge = 0.f;
    cudaEventElapsedTime(&t_merge, pa, pb);
    printf("[prof]   merge_down (L1->sheet, tier2 off)=%.3f ms\n", t_merge);
    cudaEventDestroy(pa);
    cudaEventDestroy(pb);
#else
    merge_down(stream);
#endif
#endif
  }

  // Layered point lookup.
  void lookup_layered(const DeviceLookupBatch &batch, cudaStream_t stream) {
    const std::size_t n = batch.count;
    // Binary-search the cached resolved overlay (sorted [inserts | tombstones])
    // per query instead of scanning the unsorted runs; misses fall to the sheet.
    // The overlay is rebuilt only when a mutation dirtied it (see overlay_index),
    // so back-to-back reads share one sort. Empty overlay -> straight to sheet.
    const OverlayReadIndex &ix = overlay_index(stream);
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    gpulsmopt_detail::point_lookup_kernel<<<grid, block, 0, stream>>>(
        batch.queries, n, batch.out_values, batch.out_found, raw_or_null(ix.gk),
        raw_or_null(ix.gv), ix.ins, raw_or_null(ix.gk) + ix.ins, ix.u - ix.ins,
        raw_or_null(d_dir_boundary_), raw_or_null(d_dir_seg_id_),
        h_dir_seg_id_.size(), raw_or_null(pool_keys_), raw_or_null(pool_values_),
        raw_or_null(pool_valid_), raw_or_null(seg_bucket_max_));
    CUDA_CHECK(cudaGetLastError());
  }

  std::size_t run_layer_live() const { return run_live_total_; }

  // Configuration.
  std::size_t max_elements_ = 0;
  std::size_t batch_size_ = 0;
  std::uint32_t target_fill_ = 23;
  std::uint32_t target_segment_live_ = 0;
  std::size_t sheet_live_count_ = 0;
  mutable std::shared_mutex snapshot_mutex_;
  // Background drain handle.
  mutable std::future<void> pending_drain_;

  // Run layer state.
  std::size_t run_pool_capacity_ = 0; // number of allocatable run blocks
  std::vector<std::uint32_t> run_free_ids_;
  thrust::device_vector<std::uint32_t> run_pool_keys_;
  thrust::device_vector<std::uint32_t> run_pool_values_;
  thrust::device_vector<std::uint8_t> run_pool_ops_; // kInsert / kTombstone
  std::vector<std::uint32_t> h_run_block_;    // pool block id per run ordinal
  std::vector<std::uint32_t> h_run_boundary_; // inclusive upper key bound
  std::vector<std::uint32_t> h_run_size_;     // live entry count per run
  thrust::device_vector<std::uint32_t> d_run_block_;
  thrust::device_vector<std::uint32_t> d_run_boundary_;
  thrust::device_vector<std::uint32_t> d_run_size_;
  thrust::device_vector<std::uint32_t> d_run_start_; // block * kRunCapacity
  // Reused insert scratch (sized to run count r ~ 4-9) — avoids a per-insert
  // cudaMalloc/free triple on the hot small-batch path. resize/assign no-op
  // once capacity is warm.
  thrust::device_vector<std::uint32_t> ins_counts_;  // Method A per-run counts
  thrust::device_vector<std::uint8_t> ins_is_fast_;  // per-run "has room" flag
  thrust::device_vector<std::uint32_t> ins_oldsize_; // per-run pre-append size
  std::size_t run_live_total_ = 0;                   // live entries across runs
  std::uint32_t run_target_fill_ = 0;                // fill per run after split

  // Tier-2 generation state.
  struct Tier2Gen {
    thrust::device_vector<std::uint32_t> keys;
    thrust::device_vector<std::uint32_t> values;
    thrust::device_vector<std::uint8_t> ops;
    thrust::device_vector<std::uint32_t> block_boundary; // per sub-block max key
    thrust::device_vector<std::uint32_t> block_start;    // per sub-block offset
    thrust::device_vector<std::uint32_t> block_size;     // per sub-block live
    std::uint32_t range_lo = 0;
    std::uint32_t range_hi = 0;
  };
  std::vector<Tier2Gen> tier2_;

  // Segmented bottom layer state.
  std::size_t pool_capacity_ = 0;
  std::vector<std::uint32_t> free_ids_;
  thrust::device_vector<std::uint32_t> pool_keys_;
  thrust::device_vector<std::uint32_t> pool_values_;
  thrust::device_vector<std::uint8_t> pool_valid_;
  thrust::device_vector<std::uint32_t> seg_bucket_max_;
  thrust::device_vector<std::uint32_t> seg_bucket_live_;
  thrust::device_vector<std::uint32_t> seg_bucket_value_sum_;

  std::vector<std::uint32_t> h_dir_seg_id_;
  std::vector<std::uint32_t> h_dir_boundary_;
  std::vector<std::uint32_t> h_dir_live_;
  thrust::device_vector<std::uint32_t> d_dir_seg_id_;
  thrust::device_vector<std::uint32_t> d_dir_boundary_;
  thrust::device_vector<std::uint32_t> d_dir_prefix_;
  thrust::device_vector<std::uint32_t> d_dir_value_sum_;
  thrust::device_vector<std::uint32_t> d_dir_value_prefix_;

  // Merge-down scratch.
  thrust::device_vector<std::uint32_t> scratch_incoming_keys_;
  thrust::device_vector<std::uint32_t> scratch_incoming_values_;
  thrust::device_vector<std::uint32_t> scratch_delete_keys_;

  // Drain and merge scratch.
  thrust::device_vector<std::uint32_t> seg_inc_ordinal_;
  thrust::device_vector<std::uint32_t> seg_dirty_ord_;
  thrust::device_vector<std::uint32_t> seg_dirty_count_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_seg_id_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_old_boundary_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_old_live_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_in_begin_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_in_end_;
  thrust::device_vector<std::uint32_t> seg_d_candidate_offset_;
  thrust::device_vector<std::uint32_t> seg_sheet_cursor_;
  thrust::device_vector<std::uint32_t> seg_dirty_live_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_live_offset_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_k_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_output_base_;
  thrust::device_vector<std::uint32_t> seg_d_output_seg_id_;
  thrust::device_vector<std::uint32_t> seg_d_out_dirty_;
  thrust::device_vector<std::uint32_t> seg_d_out_local_;
  thrust::device_vector<std::uint32_t> seg_d_out_k_;
  thrust::device_vector<std::uint32_t> seg_d_out_boundary_;
  thrust::device_vector<std::uint32_t> seg_group_src_seg_;
  thrust::device_vector<std::uint32_t> seg_group_src_begin_;
  thrust::device_vector<std::uint32_t> seg_group_src_end_;
  thrust::device_vector<std::uint32_t> seg_slow_;
  thrust::device_vector<std::uint32_t> seg_new_live_;
  thrust::device_vector<std::uint32_t> seg_fast_seg_id_;
  thrust::device_vector<std::uint32_t> seg_fast_in_begin_;
  thrust::device_vector<std::uint32_t> seg_fast_in_end_;

  thrust::device_vector<std::uint32_t> seg_cand_seg_;
  thrust::device_vector<std::uint32_t> seg_cand_key_;
  thrust::device_vector<std::uint32_t> seg_cand_value_;
  thrust::device_vector<std::uint32_t> seg_cand_op_;
  thrust::device_vector<std::uint32_t> seg_cand_seq_;
  thrust::device_vector<std::uint32_t> seg_cand_seq_sort_;
  thrust::device_vector<std::uint64_t> seg_cand_group_sort_;

  // Lookup scratch.
  thrust::device_vector<std::uint8_t> scratch_query_found_;

  // Cached resolved overlay (see overlay_index): reused across reads, rebuilt
  // only when a mutation flips overlay_dirty_.
  OverlayReadIndex cached_overlay_;
  bool overlay_dirty_ = true;
};
