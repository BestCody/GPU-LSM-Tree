#pragma once
#include "gpu_dictionary_adapter.cuh"

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                        \
  do {                                                                         \
    cudaError_t err__ = (stmt);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      throw std::runtime_error(cudaGetErrorString(err__));                      \
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

constexpr int kBucketSlots = 32;
constexpr int kSegmentBuckets = GPULSMOPT_SEGMENT_BUCKETS;
constexpr int kSegmentSlots = kSegmentBuckets * kBucketSlots;
static_assert(GPULSMOPT_TARGET_FILL >= 1 && GPULSMOPT_TARGET_FILL <= kBucketSlots,
              "target fill must fit a bucket");
constexpr std::uint32_t kEmptyKey = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kInsert = 1;
constexpr std::uint32_t kTombstone = 0;
constexpr std::size_t kRunsPerClass = 4;

struct DeviceUpdateBatch {
  const std::uint32_t* keys = nullptr;
  const std::uint32_t* values = nullptr;
  const std::uint8_t* ops = nullptr;
  std::size_t count = 0;
};

struct DeviceKeyBatch {
  const std::uint32_t* keys = nullptr;
  std::size_t count = 0;
};

__host__ __device__ inline std::uint32_t ceil_div_u32(std::uint32_t a,
                                                      std::uint32_t b) {
  return (a + b - 1) / b;
}

__host__ __device__ inline std::uint64_t batch_sort_key(std::uint32_t key,
                                                        std::uint32_t arrival) {
  return (static_cast<std::uint64_t>(key) << 32) |
         static_cast<std::uint64_t>(UINT32_MAX - arrival);
}

__host__ __device__ inline std::uint64_t drain_sort_key(std::uint32_t key,
                                                        std::uint32_t seq) {
  return (static_cast<std::uint64_t>(key) << 32) |
         static_cast<std::uint64_t>(UINT32_MAX - seq);
}

__device__ inline std::size_t lower_bound_u32(const std::uint32_t* data,
                                              std::size_t n,
                                              std::uint32_t key) {
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

__device__ inline std::size_t upper_bound_u32(const std::uint32_t* data,
                                              std::size_t n,
                                              std::uint32_t key) {
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

__global__ void fill_constant_op_kernel(std::uint32_t* ops, std::size_t n,
                                        std::uint32_t op) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) ops[i] = op;
}

__global__ void widen_ops_kernel(const std::uint8_t* input,
                                 std::uint32_t* output, std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = input[i];
}

__global__ void fill_constant_u32_kernel(std::uint32_t* values,
                                         std::size_t n,
                                         std::uint32_t value) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) values[i] = value;
}

__global__ void fill_batch_sort_keys_kernel(const std::uint32_t* keys,
                                            std::uint64_t* sort_keys,
                                            std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    sort_keys[i] = batch_sort_key(keys[i], static_cast<std::uint32_t>(i));
  }
}

__global__ void fill_drain_sort_keys_kernel(const std::uint32_t* keys,
                                            const std::uint32_t* seqs,
                                            std::uint64_t* sort_keys,
                                            std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    sort_keys[i] = drain_sort_key(keys[i], seqs[i]);
  }
}

__global__ void build_run_bucket_max_kernel(
    const std::uint32_t* keys, std::size_t count,
    std::uint32_t* bucket_max) {
  const std::size_t bucket = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t bucket_count =
      (count + kBucketSlots - 1) / kBucketSlots;
  if (bucket >= bucket_count) return;
  const std::size_t raw_end =
      (bucket + 1) * static_cast<std::size_t>(kBucketSlots);
  const std::size_t end = raw_end < count ? raw_end : count;
  bucket_max[bucket] = keys[end - 1];
}

__host__ __device__ inline std::uint64_t pack_winner(
    std::uint32_t seq, std::uint8_t op, std::uint32_t value) {
  return (static_cast<std::uint64_t>(seq) << 33) |
         (static_cast<std::uint64_t>(op & 1u) << 32) |
         static_cast<std::uint64_t>(value);
}

__global__ void flix_run_pull_kernel(
    const std::uint32_t* sorted_queries, std::size_t query_count,
    unsigned long long* winners,
    const std::uint32_t* const* run_keys,
    const std::uint32_t* const* run_values,
    const std::uint32_t* const* run_ops,
    const std::uint32_t* const* run_seqs,
    const std::size_t* run_sizes,
    const std::uint32_t* const* run_bucket_max,
    const std::uint32_t* descriptor_run,
    const std::uint32_t* descriptor_bucket,
    std::size_t descriptor_count) {
  const std::size_t thread = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t warp = thread / warpSize;
  const unsigned lane = thread & (warpSize - 1);
  if (warp >= descriptor_count) return;

  const std::uint32_t run = descriptor_run[warp];
  const std::uint32_t bucket = descriptor_bucket[warp];
  const std::size_t start =
      static_cast<std::size_t>(bucket) * kBucketSlots;
  const std::size_t pos = start + lane;
  const bool valid = pos < run_sizes[run];
  const std::uint32_t lane_key =
      valid ? run_keys[run][pos] : kEmptyKey;
  const std::uint32_t max_key = run_bucket_max[run][bucket];
  const std::size_t begin =
      bucket == 0
          ? 0
          : upper_bound_u32(sorted_queries, query_count,
                            run_bucket_max[run][bucket - 1]);
  const std::size_t end =
      upper_bound_u32(sorted_queries, query_count, max_key);

  for (std::size_t q = begin; q < end; ++q) {
    const std::uint32_t query = sorted_queries[q];
    const unsigned mask =
        __ballot_sync(UINT32_MAX, valid && lane_key == query);
    if (lane == 0 && mask != 0) {
      const unsigned match_lane = __ffs(mask) - 1;
      const std::size_t match_pos = start + match_lane;
      const std::uint64_t candidate =
          pack_winner(run_seqs[run][match_pos],
                      run_ops[run][match_pos],
                      run_values[run][match_pos]);
      atomicMax(&winners[q],
                static_cast<unsigned long long>(candidate));
    }
  }
}

__global__ void resolve_flix_lookup_kernel(
    const unsigned long long* winners,
    const std::uint32_t* sorted_values,
    const std::uint8_t* sorted_found,
    const std::uint32_t* original_indices,
    std::size_t query_count, std::uint32_t* out_values,
    std::uint8_t* out_found) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= query_count) return;

  bool found = sorted_found[i] != 0;
  std::uint32_t value = sorted_values[i];
  const std::uint64_t winner = winners[i];
  if (winner != 0) {
    const std::uint8_t op = static_cast<std::uint8_t>((winner >> 32) & 1u);
    found = op == kInsert;
    value = found ? static_cast<std::uint32_t>(winner) : 0;
  }

  const std::uint32_t original = original_indices[i];
  out_found[original] = found ? 1 : 0;
  out_values[original] = found ? value : 0;
}

__global__ void seg_route_keys_kernel(const std::uint32_t* keys, std::size_t n,
                                      const std::uint32_t* dir_boundary,
                                      std::size_t dir_count,
                                      std::uint32_t* out_ordinal) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  std::size_t ord = lower_bound_u32(dir_boundary, dir_count, keys[i]);
  if (ord >= dir_count) ord = dir_count - 1;
  out_ordinal[i] = static_cast<std::uint32_t>(ord);
}

__device__ inline bool seg_point_lookup(std::uint32_t seg_id, std::uint32_t key,
                                        const std::uint32_t* pool_keys,
                                        const std::uint32_t* pool_values,
                                        const std::uint8_t* pool_valid,
                                        const std::uint32_t* seg_bucket_max,
                                        std::uint32_t* out_value) {
  const std::size_t meta_base =
      static_cast<std::size_t>(seg_id) * kSegmentBuckets;
  const std::size_t bucket =
      lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, key);
  if (bucket >= kSegmentBuckets) return false;
  const std::size_t start =
      static_cast<std::size_t>(seg_id) * kSegmentSlots + bucket * kBucketSlots;
  for (int lane = 0; lane < kBucketSlots; ++lane) {
    if (pool_valid[start + lane] && pool_keys[start + lane] == key) {
      if (out_value) *out_value = pool_values[start + lane];
      return true;
    }
  }
  return false;
}

__device__ inline bool seg_sheet_contains(
    std::uint32_t key, const std::uint32_t* dir_boundary,
    const std::uint32_t* dir_seg_id, std::size_t dir_count,
    const std::uint32_t* pool_keys, const std::uint8_t* pool_valid,
    const std::uint32_t* seg_bucket_max) {
  std::size_t ord = lower_bound_u32(dir_boundary, dir_count, key);
  if (ord >= dir_count) ord = dir_count - 1;
  const std::uint32_t seg_id = dir_seg_id[ord];
  const std::size_t meta_base =
      static_cast<std::size_t>(seg_id) * kSegmentBuckets;
  const std::size_t bucket =
      lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, key);
  if (bucket >= kSegmentBuckets) return false;
  const std::size_t start =
      static_cast<std::size_t>(seg_id) * kSegmentSlots + bucket * kBucketSlots;
  for (int lane = 0; lane < kBucketSlots; ++lane) {
    if (pool_valid[start + lane] && pool_keys[start + lane] == key) return true;
  }
  return false;
}

__device__ inline std::uint32_t seg_range_count_one(
    std::uint32_t seg_id, std::uint32_t lo, std::uint32_t hi,
    const std::uint32_t* pool_keys, const std::uint8_t* pool_valid,
    const std::uint32_t* seg_bucket_max, const std::uint32_t* seg_bucket_live) {
  const std::size_t meta_base =
      static_cast<std::size_t>(seg_id) * kSegmentBuckets;
  const std::size_t slot_base =
      static_cast<std::size_t>(seg_id) * kSegmentSlots;
  const std::size_t first =
      lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, lo);
  if (first >= kSegmentBuckets) return 0;
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
  if (last < kSegmentBuckets) scan_bucket(last);
  for (std::size_t b = first + 1; b < full_end; ++b) {
    count += seg_bucket_live[meta_base + b];
  }
  return count;
}

__device__ inline std::uint32_t seg_sheet_range_count(
    std::uint32_t lo, std::uint32_t hi, const std::uint32_t* dir_boundary,
    const std::uint32_t* dir_seg_id, const std::uint32_t* dir_prefix,
    std::size_t dir_count, const std::uint32_t* pool_keys,
    const std::uint8_t* pool_valid, const std::uint32_t* seg_bucket_max,
    const std::uint32_t* seg_bucket_live) {
  std::size_t pl = lower_bound_u32(dir_boundary, dir_count, lo);
  if (pl >= dir_count) return 0;
  std::size_t pr = lower_bound_u32(dir_boundary, dir_count, hi);
  if (pr >= dir_count) pr = dir_count - 1;
  if (pl == pr) {
    return seg_range_count_one(dir_seg_id[pl], lo, hi, pool_keys, pool_valid,
                               seg_bucket_max, seg_bucket_live);
  }
  std::uint32_t total =
      seg_range_count_one(dir_seg_id[pl], lo, hi, pool_keys, pool_valid,
                          seg_bucket_max, seg_bucket_live) +
      seg_range_count_one(dir_seg_id[pr], lo, hi, pool_keys, pool_valid,
                          seg_bucket_max, seg_bucket_live);
  if (pr > pl + 1) total += dir_prefix[pr] - dir_prefix[pl + 1];
  return total;
}

__global__ void seg_gather_candidates_kernel(
    const std::uint32_t* pool_keys, const std::uint32_t* pool_values,
    const std::uint8_t* pool_valid, const std::uint32_t* dirty_seg_id,
    const std::uint32_t* candidate_offset, const std::uint32_t* dirty_old_live,
    const std::uint32_t* dirty_incoming_begin,
    const std::uint32_t* dirty_incoming_end, const std::uint32_t* incoming_keys,
    const std::uint32_t* incoming_values, const std::uint32_t* incoming_ops,
    std::size_t dirty_count, std::uint32_t* cand_seg, std::uint32_t* cand_key,
    std::uint32_t* cand_value, std::uint32_t* cand_op, std::uint32_t* cand_seq,
    std::uint32_t* sheet_cursor) {
  const std::size_t m = blockIdx.x;
  if (m >= dirty_count) return;
  const std::size_t seg_base =
      static_cast<std::size_t>(dirty_seg_id[m]) * kSegmentSlots;
  const std::uint32_t base = candidate_offset[m];

  for (std::size_t p = threadIdx.x; p < kSegmentSlots; p += blockDim.x) {
    if (!pool_valid[seg_base + p]) continue;
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
    cand_op[out] = incoming_ops[i];
    cand_seq[out] = 1u;
  }
}

__global__ void seg_make_resolve_keys_kernel(const std::uint32_t* cand_seg,
                                             const std::uint32_t* cand_key,
                                             const std::uint32_t* cand_seq,
                                             std::size_t count,
                                             std::uint32_t* seq_sort,
                                             std::uint64_t* group_sort) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  seq_sort[i] = UINT32_MAX - cand_seq[i];
  group_sort[i] =
      (static_cast<std::uint64_t>(cand_seg[i]) << 32) | cand_key[i];
}

__global__ void seg_classify_inplace_kernel(
    const std::uint32_t* dirty_seg_id, const std::uint32_t* dirty_in_begin,
    const std::uint32_t* dirty_in_end, std::size_t dirty_count,
    const std::uint32_t* incoming_keys, const std::uint32_t* incoming_ops,
    const std::uint32_t* pool_keys, const std::uint8_t* pool_valid,
    const std::uint32_t* seg_bucket_max, const std::uint32_t* seg_bucket_live,
    std::uint32_t* seg_slow, std::uint32_t* seg_new_live) {
  const std::size_t m = blockIdx.x;
  if (m >= dirty_count) return;
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
    int inserts_new = 0, deletes_ex = 0;
    for (std::size_t i = in_begin + begin; i < in_begin + end; ++i) {
      const std::uint32_t key = incoming_keys[i];
      bool exists = false;
      for (int lane = 0; lane < kBucketSlots; ++lane) {
        if (pool_valid[base + lane] && pool_keys[base + lane] == key) {
          exists = true;
          break;
        }
      }
      if (incoming_ops[i] == kInsert) {
        if (!exists) ++inserts_new;
      } else if (exists) {
        ++deletes_ex;
      }
    }
    const int new_live =
        static_cast<int>(seg_bucket_live[meta + b]) + inserts_new - deletes_ex;
    if (new_live > kBucketSlots) atomicOr(&s_slow, 1);
    atomicAdd(&s_total, static_cast<unsigned long long>(new_live));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    seg_slow[m] = static_cast<std::uint32_t>(s_slow);
    seg_new_live[m] = static_cast<std::uint32_t>(s_total);
  }
}

__global__ void seg_apply_inplace_kernel(
    const std::uint32_t* fast_seg_id, const std::uint32_t* fast_in_begin,
    const std::uint32_t* fast_in_end, std::size_t fast_count,
    const std::uint32_t* incoming_keys, const std::uint32_t* incoming_values,
    const std::uint32_t* incoming_ops, std::uint32_t* pool_keys,
    std::uint32_t* pool_values, std::uint8_t* pool_valid,
    std::uint32_t* seg_bucket_max, std::uint32_t* seg_bucket_live) {
  const std::size_t f = blockIdx.x;
  if (f >= fast_count) return;
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
      if (found < 0) continue;
      if (incoming_ops[i] == kInsert) {
        pool_values[base + found] = incoming_values[i];
      } else {
        pool_valid[base + found] = 0;
      }
    }
    for (std::size_t i = in_begin + begin; i < in_begin + end; ++i) {
      if (incoming_ops[i] != kInsert) continue;
      const std::uint32_t key = incoming_keys[i];
      bool exists = false;
      for (int lane = 0; lane < kBucketSlots; ++lane) {
        if (pool_valid[base + lane] && pool_keys[base + lane] == key) {
          exists = true;
          break;
        }
      }
      if (exists) continue;
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
    std::uint32_t live = 0, mx = 0;
    bool any = false;
    for (int lane = 0; lane < kBucketSlots; ++lane) {
      if (pool_valid[base + lane]) {
        ++live;
        const std::uint32_t k = pool_keys[base + lane];
        if (!any || k > mx) {
          mx = k;
          any = true;
        }
      }
    }
    seg_bucket_live[meta + b] = live;
    if (any) seg_bucket_max[meta + b] = mx;
  }
}

struct DrainTombstonePredicate {
  template <class Tuple>
  __host__ __device__ bool operator()(const Tuple& t) const {
    return thrust::get<3>(t) == kTombstone;
  }
};

__global__ void seg_count_survivors_kernel(const std::uint32_t* cand_seg,
                                           std::size_t live_count,
                                           std::uint32_t* dirty_live) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < live_count) atomicAdd(&dirty_live[cand_seg[i]], 1u);
}

__global__ void seg_output_boundaries_kernel(
    const std::uint32_t* out_dirty, const std::uint32_t* out_local,
    const std::uint32_t* out_k, std::size_t output_count,
    const std::uint32_t* dirty_live, const std::uint32_t* dirty_live_offset,
    const std::uint32_t* dirty_old_boundary,
    const std::uint32_t* survivor_keys, std::uint32_t* out_boundary) {
  const std::size_t o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= output_count) return;
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

__global__ void seg_clear_segments_kernel(const std::uint32_t* out_seg_id,
                                          std::size_t output_count,
                                          std::uint32_t* pool_keys,
                                          std::uint32_t* pool_values,
                                          std::uint8_t* pool_valid) {
  const std::size_t o = blockIdx.x;
  if (o >= output_count) return;
  const std::size_t seg_base =
      static_cast<std::size_t>(out_seg_id[o]) * kSegmentSlots;
  for (std::size_t p = threadIdx.x; p < kSegmentSlots; p += blockDim.x) {
    pool_keys[seg_base + p] = kEmptyKey;
    pool_values[seg_base + p] = 0u;
    pool_valid[seg_base + p] = 0u;
  }
}

__global__ void seg_scatter_survivors_kernel(
    const std::uint32_t* cand_seg, const std::uint32_t* cand_key,
    const std::uint32_t* cand_value, std::size_t live_count,
    const std::uint32_t* dirty_live, const std::uint32_t* dirty_live_offset,
    const std::uint32_t* dirty_k, const std::uint32_t* dirty_output_base,
    const std::uint32_t* output_seg_id, std::uint32_t target_fill,
    std::uint32_t* pool_keys, std::uint32_t* pool_values,
    std::uint8_t* pool_valid) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= live_count) return;
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
                          static_cast<std::size_t>(bucket) * kBucketSlots + slot;
  pool_keys[pos] = cand_key[i];
  pool_values[pos] = cand_value[i];
  pool_valid[pos] = 1u;
}

__global__ void seg_build_segment_metadata_kernel(
    const std::uint32_t* out_seg_id, const std::uint32_t* out_boundary,
    std::size_t output_count, const std::uint32_t* pool_keys,
    const std::uint8_t* pool_valid, std::uint32_t* seg_bucket_max,
    std::uint32_t* seg_bucket_live) {
  const std::size_t o = blockIdx.x;
  if (o >= output_count) return;
  const std::uint32_t seg_id = out_seg_id[o];
  const std::uint32_t boundary = out_boundary[o];
  const std::size_t slot_base = static_cast<std::size_t>(seg_id) * kSegmentSlots;
  const std::size_t meta_base =
      static_cast<std::size_t>(seg_id) * kSegmentBuckets;
  for (std::size_t b = threadIdx.x; b < kSegmentBuckets; b += blockDim.x) {
    const std::size_t start = slot_base + b * kBucketSlots;
    std::uint32_t live = 0;
    std::uint32_t max_key = boundary;
    for (int lane = 0; lane < kBucketSlots; ++lane) {
      if (pool_valid[start + lane]) {
        ++live;
        max_key = pool_keys[start + lane];
      }
    }
    seg_bucket_live[meta_base + b] = live;
    seg_bucket_max[meta_base + b] = live > 0 ? max_key : boundary;
  }
}

// ---------------------------------------------------------------------------
// Merge / repack kernels (net-new; not present in segmented_sheet.cuh).
// ---------------------------------------------------------------------------

// One block per merge group: gather all valid (seq 0) records from each of the
// group's source segments into its candidate slice. Order within the slice is
// arbitrary; a stable key sort follows.
__global__ void seg_merge_gather_kernel(
    const std::uint32_t* pool_keys, const std::uint32_t* pool_values,
    const std::uint8_t* pool_valid, const std::uint32_t* group_src_seg,
    const std::uint32_t* group_src_begin, const std::uint32_t* group_src_end,
    const std::uint32_t* candidate_offset, std::size_t group_count,
    std::uint32_t* cand_seg, std::uint32_t* cand_key,
    std::uint32_t* cand_value, std::uint32_t* cursor) {
  const std::size_t g = blockIdx.x;
  if (g >= group_count) return;
  const std::uint32_t base = candidate_offset[g];
  const std::uint32_t sb = group_src_begin[g];
  const std::uint32_t se = group_src_end[g];
  for (std::uint32_t s = sb; s < se; ++s) {
    const std::size_t seg_base =
        static_cast<std::size_t>(group_src_seg[s]) * kSegmentSlots;
    for (std::size_t p = threadIdx.x; p < kSegmentSlots; p += blockDim.x) {
      if (!pool_valid[seg_base + p]) continue;
      const std::uint32_t out = base + atomicAdd(&cursor[g], 1u);
      cand_seg[out] = static_cast<std::uint32_t>(g);
      cand_key[out] = pool_keys[seg_base + p];
      cand_value[out] = pool_values[seg_base + p];
    }
  }
}

__global__ void seg_merge_sort_keys_kernel(const std::uint32_t* cand_seg,
                                           const std::uint32_t* cand_key,
                                           std::size_t count,
                                           std::uint64_t* group_sort) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  group_sort[i] =
      (static_cast<std::uint64_t>(cand_seg[i]) << 32) | cand_key[i];
}

__global__ void segment_sheet_pull_kernel(
    const std::uint32_t* sorted_queries, std::size_t query_count,
    const unsigned long long* winners, const std::uint32_t* dir_boundary,
    const std::uint32_t* dir_seg_id, std::size_t dir_count,
    const std::uint32_t* pool_keys, const std::uint32_t* pool_values,
    const std::uint8_t* pool_valid, const std::uint32_t* seg_bucket_max,
    std::uint32_t* sorted_values, std::uint8_t* sorted_found) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= query_count) return;
  if (winners[i] != 0) return; 
  const std::uint32_t key = sorted_queries[i];
  std::size_t ord = lower_bound_u32(dir_boundary, dir_count, key);
  if (ord >= dir_count) ord = dir_count - 1;
  std::uint32_t value = 0;
  if (seg_point_lookup(dir_seg_id[ord], key, pool_keys, pool_values, pool_valid,
                       seg_bucket_max, &value)) {
    sorted_values[i] = value;
    sorted_found[i] = 1;
  }
}

__global__ void count_combined_kernel(
    const std::uint32_t* lo, const std::uint32_t* hi,
    std::uint32_t* out_counts, std::size_t query_count,
    const std::uint32_t* dir_boundary, const std::uint32_t* dir_seg_id,
    const std::uint32_t* dir_prefix, std::size_t dir_count,
    const std::uint32_t* pool_keys, const std::uint8_t* pool_valid,
    const std::uint32_t* seg_bucket_max, const std::uint32_t* seg_bucket_live,
    const std::uint32_t* const* run_keys, const std::uint32_t* const* run_ops,
    const std::uint32_t* const* run_seqs, const std::size_t* run_sizes,
    std::size_t run_count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= query_count) return;
  if (lo[i] > hi[i]) {
    out_counts[i] = 0;
    return;
  }

  std::int64_t count = static_cast<std::int64_t>(seg_sheet_range_count(
      lo[i], hi[i], dir_boundary, dir_seg_id, dir_prefix, dir_count, pool_keys,
      pool_valid, seg_bucket_max, seg_bucket_live));

  for (std::size_t rr = run_count; rr > 0; --rr) {
    const std::size_t r = rr - 1;
    const std::uint32_t* keys = run_keys[r];
    const std::size_t n = run_sizes[r];
    const std::size_t begin = lower_bound_u32(keys, n, lo[i]);
    const std::size_t end = upper_bound_u32(keys, n, hi[i]);
    for (std::size_t p = begin; p < end; ++p) {
      const std::uint32_t key = keys[p];
      const std::uint32_t seq = run_seqs[r][p];
      bool shadowed = false;
      for (std::size_t other = 0; other < run_count; ++other) {
        if (other == r) continue;
        const std::size_t pos =
            lower_bound_u32(run_keys[other], run_sizes[other], key);
        if (pos < run_sizes[other] && run_keys[other][pos] == key &&
            run_seqs[other][pos] > seq) {
          shadowed = true;
          break;
        }
      }
      if (shadowed) continue;

      const bool in_sheet =
          seg_sheet_contains(key, dir_boundary, dir_seg_id, dir_count,
                             pool_keys, pool_valid, seg_bucket_max);
      if (run_ops[r][p] == kTombstone) {
        if (in_sheet) --count;
      } else if (!in_sheet) {
        ++count;
      }
    }
  }

  out_counts[i] = count > 0 ? static_cast<std::uint32_t>(count) : 0;
}

__global__ void range_source_counts_kernel(
    const std::uint32_t* lo, const std::uint32_t* hi, std::size_t query_count,
    const std::uint32_t* dir_boundary, const std::uint32_t* dir_seg_id,
    const std::uint32_t* dir_prefix, std::size_t dir_count,
    const std::uint32_t* pool_keys, const std::uint8_t* pool_valid,
    const std::uint32_t* seg_bucket_max, const std::uint32_t* seg_bucket_live,
    const std::uint32_t* const* run_keys, const std::size_t* run_sizes,
    std::size_t run_count, std::uint32_t* source_counts) {
  const std::size_t source_count = run_count + 1;
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= query_count * source_count) return;
  const std::size_t query = index / source_count;
  const std::size_t source = index % source_count;
  if (lo[query] > hi[query]) {
    source_counts[index] = 0;
    return;
  }
  if (source == 0) {
    source_counts[index] = seg_sheet_range_count(
        lo[query], hi[query], dir_boundary, dir_seg_id, dir_prefix, dir_count,
        pool_keys, pool_valid, seg_bucket_max, seg_bucket_live);
  } else {
    const std::uint32_t* keys = run_keys[source - 1];
    const std::size_t count = run_sizes[source - 1];
    const std::size_t begin = lower_bound_u32(keys, count, lo[query]);
    const std::size_t end = upper_bound_u32(keys, count, hi[query]);
    source_counts[index] = static_cast<std::uint32_t>(end - begin);
  }
}

__global__ void range_query_totals_kernel(
    const std::uint32_t* source_counts, std::size_t query_count,
    std::size_t source_count, std::uint32_t* query_totals) {
  const std::size_t query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= query_count) return;
  std::uint32_t total = 0;
  const std::size_t base = query * source_count;
  for (std::size_t source = 0; source < source_count; ++source) {
    total += source_counts[base + source];
  }
  query_totals[query] = total;
}

__global__ void range_source_offsets_kernel(
    const std::uint32_t* source_counts, const std::uint32_t* query_offsets,
    std::size_t query_count, std::size_t source_count,
    std::uint32_t* source_offsets) {
  const std::size_t query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= query_count) return;
  std::uint32_t offset = query_offsets[query];
  const std::size_t base = query * source_count;
  for (std::size_t source = 0; source < source_count; ++source) {
    source_offsets[base + source] = offset;
    offset += source_counts[base + source];
  }
}

__global__ void range_copy_candidates_kernel(
    const std::uint32_t* lo, const std::uint32_t* hi, std::size_t query_count,
    const std::uint32_t* dir_boundary, const std::uint32_t* dir_seg_id,
    std::size_t dir_count, const std::uint32_t* pool_keys,
    const std::uint32_t* pool_values, const std::uint8_t* pool_valid,
    const std::uint32_t* seg_bucket_max, const std::uint32_t* const* run_keys,
    const std::uint32_t* const* run_values, const std::uint32_t* const* run_ops,
    const std::uint32_t* const* run_seqs, const std::size_t* run_sizes,
    std::size_t run_count, const std::uint32_t* source_offsets,
    std::uint32_t* source_cursors, std::uint32_t* candidate_queries,
    std::uint32_t* candidate_keys, std::uint32_t* candidate_values,
    std::uint32_t* candidate_ops, std::uint32_t* candidate_seqs) {
  const std::size_t source_count = run_count + 1;
  const std::size_t descriptor = blockIdx.x;
  if (descriptor >= query_count * source_count) return;
  const std::size_t query = descriptor / source_count;
  const std::size_t source = descriptor % source_count;

  if (source == 0) {
    if (lo[query] > hi[query]) return;
    std::size_t pl = lower_bound_u32(dir_boundary, dir_count, lo[query]);
    if (pl >= dir_count) return;
    std::size_t pr = lower_bound_u32(dir_boundary, dir_count, hi[query]);
    if (pr >= dir_count) pr = dir_count - 1;
    for (std::size_t ord = pl; ord <= pr; ++ord) {
      const std::uint32_t seg_id = dir_seg_id[ord];
      const std::size_t slot_base =
          static_cast<std::size_t>(seg_id) * kSegmentSlots;
      const std::size_t meta_base =
          static_cast<std::size_t>(seg_id) * kSegmentBuckets;
      const std::size_t first_b =
          lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, lo[query]);
      std::size_t last_b =
          lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, hi[query]);
      if (last_b >= kSegmentBuckets) last_b = kSegmentBuckets - 1;
      if (first_b > last_b) continue;
      const std::size_t begin_pos = slot_base + first_b * kBucketSlots;
      const std::size_t end_pos = slot_base + (last_b + 1) * kBucketSlots;
      for (std::size_t p = begin_pos + threadIdx.x; p < end_pos;
           p += blockDim.x) {
        if (!pool_valid[p]) continue;
        const std::uint32_t key = pool_keys[p];
        if (key < lo[query] || key > hi[query]) continue;
        const std::uint32_t out = atomicAdd(&source_cursors[descriptor], 1u);
        candidate_queries[out] = static_cast<std::uint32_t>(query);
        candidate_keys[out] = key;
        candidate_values[out] = pool_values[p];
        candidate_ops[out] = kInsert;
        candidate_seqs[out] = 0u;
      }
    }
  } else {
    const std::uint32_t* keys = run_keys[source - 1];
    const std::uint32_t* values = run_values[source - 1];
    const std::size_t count = run_sizes[source - 1];
    const std::size_t begin = lower_bound_u32(keys, count, lo[query]);
    const std::size_t end = upper_bound_u32(keys, count, hi[query]);
    const std::uint32_t output = source_offsets[descriptor];
    for (std::size_t p = begin + threadIdx.x; p < end; p += blockDim.x) {
      const std::size_t out = output + (p - begin);
      candidate_queries[out] = static_cast<std::uint32_t>(query);
      candidate_keys[out] = keys[p];
      candidate_values[out] = values[p];
      candidate_ops[out] = run_ops[source - 1][p];
      candidate_seqs[out] = run_seqs[source - 1][p];
    }
  }
}

__global__ void range_candidate_sort_keys_kernel(
    const std::uint32_t* queries, const std::uint32_t* keys,
    const std::uint32_t* seqs, std::size_t count,
    std::uint32_t* seq_sort_keys, std::uint64_t* group_sort_keys) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  seq_sort_keys[i] = UINT32_MAX - seqs[i];
  group_sort_keys[i] =
      (static_cast<std::uint64_t>(queries[i]) << 32) | keys[i];
}

__global__ void range_count_live_kernel(
    const std::uint32_t* candidate_queries, std::size_t live_count,
    std::uint32_t* query_counts) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < live_count) atomicAdd(&query_counts[candidate_queries[i]], 1u);
}

__global__ void range_scatter_live_kernel(
    const std::uint32_t* candidate_keys,
    const std::uint32_t* candidate_values, std::size_t live_count,
    std::size_t output_capacity, std::uint32_t* out_keys,
    std::uint32_t* out_values) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= live_count || i >= output_capacity) return;
  out_keys[i] = candidate_keys[i];
  out_values[i] = candidate_values[i];
}

struct RangeTombstonePredicate {
  template <class Tuple>
  __host__ __device__ bool operator()(const Tuple& t) const {
    return thrust::get<3>(t) == kTombstone;  // (query, key, value, op, seq)
  }
};

}  // namespace gpulsmopt_detail

class GPULSMOpt {
 public:
  using DeviceUpdateBatch = gpulsmopt_detail::DeviceUpdateBatch;
  using DeviceKeyBatch = gpulsmopt_detail::DeviceKeyBatch;

  struct DirectorySnapshot {
    std::vector<std::uint32_t> seg_id;
    std::vector<std::uint32_t> boundary;
    std::vector<std::uint32_t> live;
    std::vector<std::uint32_t> prefix;  // size == seg_id.size() + 1
  };

  explicit GPULSMOpt(const DictionaryConfig& config)
      : max_elements_(config.max_elements),
        batch_size_(config.batch_size),
        target_fill_(GPULSMOPT_TARGET_FILL) {
    if (max_elements_ > 0x7fffffffu) {
      throw std::invalid_argument(
          "GPULSMOpt currently supports at most 2^31-1 records");
    }
    target_segment_live_ =
        static_cast<std::uint32_t>(gpulsmopt_detail::kSegmentBuckets) * target_fill_;
    compute_max_size_class();
    initialize_segmented_storage(0);
    CUDA_CHECK(cudaStreamSynchronize(0));
  }

  void clear(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    runs_.clear();
    next_seq_ = 1;
    refresh_run_pointer_cache(stream);
    reset_directory_to_root(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void insert(const DeviceKeyValueBatch& batch, cudaStream_t stream) {
    if (batch.count == 0) return;
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    append_constant_op_run(batch.keys, batch.values, batch.count,
                           gpulsmopt_detail::kInsert, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void update(const DeviceUpdateBatch& batch, cudaStream_t stream) {
    if (batch.count == 0) return;
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);

    Run run;
    const std::uint32_t seq = allocate_sequence();
    run.size_class = 0;
    run.keys.resize(batch.count);
    run.values.resize(batch.count);
    run.ops.resize(batch.count);
    run.seqs.resize(batch.count);

    CUDA_CHECK(cudaMemcpyAsync(raw_or_null(run.keys), batch.keys,
                               batch.count * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(raw_or_null(run.values), batch.values,
                               batch.count * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToDevice, stream));
    widen_ops(batch.ops, run.ops, stream);
    fill_constant_u32(run.seqs, seq, stream);

    dedup_run_by_last_arrival(&run, stream);
    runs_.push_back(std::move(run));
    refresh_run_pointer_cache(stream);
    compact_overflow_classes(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void erase(const DeviceKeyBatch& batch, cudaStream_t stream) {
    if (batch.count == 0) return;
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    append_constant_op_run(batch.keys, nullptr, batch.count,
                           gpulsmopt_detail::kTombstone, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void lookup(const DeviceLookupBatch& batch, cudaStream_t stream) {
    if (batch.count == 0) return;
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    auto policy = thrust::cuda::par.on(stream);
    scratch_query_keys_.resize(batch.count);
    scratch_query_indices_.resize(batch.count);
    scratch_query_winners_.resize(batch.count);
    scratch_query_values_.resize(batch.count);
    scratch_query_found_.resize(batch.count);

    CUDA_CHECK(cudaMemcpyAsync(raw_or_null(scratch_query_keys_), batch.queries,
                               batch.count * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToDevice, stream));
    thrust::sequence(policy, scratch_query_indices_.begin(),
                     scratch_query_indices_.end());
    thrust::sort_by_key(policy, scratch_query_keys_.begin(),
                        scratch_query_keys_.end(),
                        scratch_query_indices_.begin());
    thrust::fill(policy, scratch_query_winners_.begin(),
                 scratch_query_winners_.end(), 0ull);
    thrust::fill(policy, scratch_query_values_.begin(),
                 scratch_query_values_.end(), 0u);
    thrust::fill(policy, scratch_query_found_.begin(),
                 scratch_query_found_.end(), 0u);

    if (!run_bucket_descriptor_run_.empty()) {
      constexpr int block = 256;
      constexpr int warps_per_block = block / 32;
      const int grid = static_cast<int>(
          (run_bucket_descriptor_run_.size() + warps_per_block - 1) /
          warps_per_block);
      gpulsmopt_detail::flix_run_pull_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(scratch_query_keys_), batch.count,
          raw_or_null(scratch_query_winners_), raw_or_null(run_key_ptrs_),
          raw_or_null(run_value_ptrs_), raw_or_null(run_op_ptrs_),
          raw_or_null(run_seq_ptrs_), raw_or_null(run_sizes_),
          raw_or_null(run_bucket_max_ptrs_),
          raw_or_null(run_bucket_descriptor_run_),
          raw_or_null(run_bucket_descriptor_bucket_),
          run_bucket_descriptor_run_.size());
      CUDA_CHECK(cudaGetLastError());
    }

    {
      const int block = 256;
      const int grid = static_cast<int>((batch.count + block - 1) / block);
      gpulsmopt_detail::segment_sheet_pull_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(scratch_query_keys_), batch.count,
          raw_or_null(scratch_query_winners_), raw_or_null(d_dir_boundary_),
          raw_or_null(d_dir_seg_id_), h_dir_seg_id_.size(),
          raw_or_null(pool_keys_), raw_or_null(pool_values_),
          raw_or_null(pool_valid_), raw_or_null(seg_bucket_max_),
          raw_or_null(scratch_query_values_),
          raw_or_null(scratch_query_found_));
      CUDA_CHECK(cudaGetLastError());
    }

    const int block = 256;
    const int grid = static_cast<int>((batch.count + block - 1) / block);
    gpulsmopt_detail::resolve_flix_lookup_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(scratch_query_winners_),
        raw_or_null(scratch_query_values_), raw_or_null(scratch_query_found_),
        raw_or_null(scratch_query_indices_), batch.count, batch.out_values,
        batch.out_found);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void count(const DeviceRangeBatch& batch, cudaStream_t stream) {
    if (batch.count == 0) return;
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    const int block = 128;
    const int grid = static_cast<int>((batch.count + block - 1) / block);
    gpulsmopt_detail::count_combined_kernel<<<grid, block, 0, stream>>>(
        batch.lo, batch.hi, batch.out_counts, batch.count,
        raw_or_null(d_dir_boundary_), raw_or_null(d_dir_seg_id_),
        raw_or_null(d_dir_prefix_), h_dir_seg_id_.size(),
        raw_or_null(pool_keys_), raw_or_null(pool_valid_),
        raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_),
        raw_or_null(run_key_ptrs_), raw_or_null(run_op_ptrs_),
        raw_or_null(run_seq_ptrs_), raw_or_null(run_sizes_), runs_.size());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void range(const DeviceRangeOutputBatch& batch, cudaStream_t stream) {
    if (batch.query_count == 0) return;
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    auto policy = thrust::cuda::par.on(stream);
    const std::size_t source_count = runs_.size() + 1;
    const std::size_t descriptor_count = batch.query_count * source_count;
    scratch_range_source_counts_.resize(descriptor_count);
    scratch_range_source_offsets_.resize(descriptor_count);
    scratch_range_source_cursors_.resize(descriptor_count);
    scratch_range_query_totals_.resize(batch.query_count);
    scratch_range_query_offsets_.resize(batch.query_count);

    const int block = 256;
    const int count_grid =
        static_cast<int>((descriptor_count + block - 1) / block);
    gpulsmopt_detail::range_source_counts_kernel<<<count_grid, block, 0, stream>>>(
        batch.lo, batch.hi, batch.query_count, raw_or_null(d_dir_boundary_),
        raw_or_null(d_dir_seg_id_), raw_or_null(d_dir_prefix_),
        h_dir_seg_id_.size(), raw_or_null(pool_keys_), raw_or_null(pool_valid_),
        raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_),
        raw_or_null(run_key_ptrs_), raw_or_null(run_sizes_), runs_.size(),
        raw_or_null(scratch_range_source_counts_));
    CUDA_CHECK(cudaGetLastError());

    const int query_grid =
        static_cast<int>((batch.query_count + block - 1) / block);
    gpulsmopt_detail::range_query_totals_kernel<<<query_grid, block, 0, stream>>>(
        raw_or_null(scratch_range_source_counts_), batch.query_count,
        source_count, raw_or_null(scratch_range_query_totals_));
    CUDA_CHECK(cudaGetLastError());
    thrust::exclusive_scan(policy, scratch_range_query_totals_.begin(),
                           scratch_range_query_totals_.end(),
                           scratch_range_query_offsets_.begin());

    std::uint32_t last_offset = 0;
    std::uint32_t last_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &last_offset,
        raw_or_null(scratch_range_query_offsets_) + batch.query_count - 1,
        sizeof(last_offset), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &last_count,
        raw_or_null(scratch_range_query_totals_) + batch.query_count - 1,
        sizeof(last_count), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::size_t candidate_count =
        static_cast<std::size_t>(last_offset) + last_count;

    gpulsmopt_detail::range_source_offsets_kernel<<<query_grid, block, 0, stream>>>(
        raw_or_null(scratch_range_source_counts_),
        raw_or_null(scratch_range_query_offsets_), batch.query_count,
        source_count, raw_or_null(scratch_range_source_offsets_));
    CUDA_CHECK(cudaGetLastError());

    resize_range_candidates(candidate_count);
    if (candidate_count > 0) {
      // The sheet-source gather uses atomic cursors seeded at the source
      // offsets; runs index deterministically from the offsets directly.
      CUDA_CHECK(cudaMemcpyAsync(
          raw_or_null(scratch_range_source_cursors_),
          raw_or_null(scratch_range_source_offsets_),
          descriptor_count * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice,
          stream));
      gpulsmopt_detail::range_copy_candidates_kernel
          <<<static_cast<unsigned>(descriptor_count), block, 0, stream>>>(
              batch.lo, batch.hi, batch.query_count,
              raw_or_null(d_dir_boundary_), raw_or_null(d_dir_seg_id_),
              h_dir_seg_id_.size(), raw_or_null(pool_keys_),
              raw_or_null(pool_values_), raw_or_null(pool_valid_),
              raw_or_null(seg_bucket_max_), raw_or_null(run_key_ptrs_),
              raw_or_null(run_value_ptrs_), raw_or_null(run_op_ptrs_),
              raw_or_null(run_seq_ptrs_), raw_or_null(run_sizes_), runs_.size(),
              raw_or_null(scratch_range_source_offsets_),
              raw_or_null(scratch_range_source_cursors_),
              raw_or_null(scratch_range_candidate_queries_),
              raw_or_null(scratch_range_candidate_keys_),
              raw_or_null(scratch_range_candidate_values_),
              raw_or_null(scratch_range_candidate_ops_),
              raw_or_null(scratch_range_candidate_seqs_));
      CUDA_CHECK(cudaGetLastError());

      const int candidate_grid =
          static_cast<int>((candidate_count + block - 1) / block);
      gpulsmopt_detail::range_candidate_sort_keys_kernel
          <<<candidate_grid, block, 0, stream>>>(
              raw_or_null(scratch_range_candidate_queries_),
              raw_or_null(scratch_range_candidate_keys_),
              raw_or_null(scratch_range_candidate_seqs_), candidate_count,
              raw_or_null(scratch_range_seq_sort_keys_),
              raw_or_null(scratch_range_group_sort_keys_));
      CUDA_CHECK(cudaGetLastError());

      auto first_payload = thrust::make_zip_iterator(thrust::make_tuple(
          scratch_range_group_sort_keys_.begin(),
          scratch_range_candidate_queries_.begin(),
          scratch_range_candidate_keys_.begin(),
          scratch_range_candidate_values_.begin(),
          scratch_range_candidate_ops_.begin(),
          scratch_range_candidate_seqs_.begin()));
      thrust::stable_sort_by_key(policy, scratch_range_seq_sort_keys_.begin(),
                                 scratch_range_seq_sort_keys_.end(),
                                 first_payload);

      auto second_payload = thrust::make_zip_iterator(thrust::make_tuple(
          scratch_range_candidate_queries_.begin(),
          scratch_range_candidate_keys_.begin(),
          scratch_range_candidate_values_.begin(),
          scratch_range_candidate_ops_.begin(),
          scratch_range_candidate_seqs_.begin()));
      thrust::stable_sort_by_key(policy, scratch_range_group_sort_keys_.begin(),
                                 scratch_range_group_sort_keys_.end(),
                                 second_payload);

      auto unique_end = thrust::unique_by_key(
          policy, scratch_range_group_sort_keys_.begin(),
          scratch_range_group_sort_keys_.end(), second_payload);
      const std::size_t unique_count = static_cast<std::size_t>(
          unique_end.first - scratch_range_group_sort_keys_.begin());
      resize_range_candidates(unique_count);

      auto live_begin = thrust::make_zip_iterator(thrust::make_tuple(
          scratch_range_candidate_queries_.begin(),
          scratch_range_candidate_keys_.begin(),
          scratch_range_candidate_values_.begin(),
          scratch_range_candidate_ops_.begin(),
          scratch_range_candidate_seqs_.begin()));
      auto filtered_end = thrust::remove_if(policy, live_begin,
                                            live_begin + unique_count,
                                            gpulsmopt_detail::RangeTombstonePredicate{});
      const std::size_t live_count =
          static_cast<std::size_t>(filtered_end - live_begin);
      resize_range_candidates(live_count);

      thrust::fill(policy, scratch_range_query_totals_.begin(),
                   scratch_range_query_totals_.end(), 0u);
      const int live_grid = static_cast<int>((live_count + block - 1) / block);
      if (live_count > 0) {
        gpulsmopt_detail::range_count_live_kernel<<<live_grid, block, 0, stream>>>(
            raw_or_null(scratch_range_candidate_queries_), live_count,
            raw_or_null(scratch_range_query_totals_));
        CUDA_CHECK(cudaGetLastError());
      }
      thrust::exclusive_scan(policy, scratch_range_query_totals_.begin(),
                             scratch_range_query_totals_.end(),
                             scratch_range_query_offsets_.begin());
      CUDA_CHECK(cudaMemcpyAsync(
          batch.out_counts, raw_or_null(scratch_range_query_totals_),
          batch.query_count * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice,
          stream));
      CUDA_CHECK(cudaMemcpyAsync(
          batch.out_offsets, raw_or_null(scratch_range_query_offsets_),
          batch.query_count * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice,
          stream));
      if (live_count > 0) {
        gpulsmopt_detail::range_scatter_live_kernel<<<live_grid, block, 0, stream>>>(
            raw_or_null(scratch_range_candidate_keys_),
            raw_or_null(scratch_range_candidate_values_), live_count,
            batch.output_capacity, batch.out_keys, batch.out_values);
        CUDA_CHECK(cudaGetLastError());
      }
    } else {
      CUDA_CHECK(cudaMemsetAsync(batch.out_counts, 0,
                                 batch.query_count * sizeof(std::uint32_t),
                                 stream));
      CUDA_CHECK(cudaMemsetAsync(batch.out_offsets, 0,
                                 batch.query_count * sizeof(std::uint32_t),
                                 stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void cleanup(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    drain_runs_into_segments(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void maintain(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    while (compact_one_ready_class(stream, gpulsmopt_detail::kRunsPerClass)) {
    }
    merge_underfull_segments(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void drain_to_sheet(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    drain_runs_into_segments(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  std::size_t sheet_live_count() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    return sheet_live_count_;
  }
  std::size_t run_count() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    return runs_.size();
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
    return pool_capacity_ * static_cast<std::size_t>(gpulsmopt_detail::kSegmentSlots);
  }

  std::size_t gpu_resident_bytes() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    std::size_t total = 0;
    for (const auto& run : runs_) {
      total += device_bytes_all(run.keys, run.values, run.ops, run.seqs,
                                run.bucket_max);
    }
    total += device_bytes_all(
        pool_keys_, pool_values_, pool_valid_, seg_bucket_max_, seg_bucket_live_,
        d_dir_seg_id_, d_dir_boundary_, d_dir_prefix_, scratch_sort_keys_,
        scratch_incoming_keys_, scratch_incoming_values_, scratch_incoming_ops_,
        scratch_incoming_seqs_, seg_inc_ordinal_, seg_dirty_ord_,
        seg_dirty_count_, seg_d_dirty_seg_id_, seg_d_dirty_old_boundary_,
        seg_d_dirty_old_live_, seg_d_dirty_in_begin_, seg_d_dirty_in_end_,
        seg_d_candidate_offset_, seg_sheet_cursor_, seg_dirty_live_,
        seg_d_dirty_live_offset_, seg_d_dirty_k_, seg_d_dirty_output_base_,
        seg_d_output_seg_id_, seg_d_out_dirty_, seg_d_out_local_, seg_d_out_k_,
        seg_d_out_boundary_, seg_group_src_seg_, seg_group_src_begin_,
        seg_group_src_end_, seg_slow_, seg_new_live_, seg_fast_seg_id_,
        seg_fast_in_begin_, seg_fast_in_end_, seg_cand_seg_, seg_cand_key_,
        seg_cand_value_, seg_cand_op_, seg_cand_seq_, seg_cand_seq_sort_,
        seg_cand_group_sort_, scratch_query_keys_, scratch_query_indices_,
        scratch_query_winners_, scratch_query_values_, scratch_query_found_,
        scratch_range_source_counts_, scratch_range_source_offsets_,
        scratch_range_source_cursors_, scratch_range_query_totals_,
        scratch_range_query_offsets_, scratch_range_candidate_queries_,
        scratch_range_candidate_keys_, scratch_range_candidate_values_,
        scratch_range_candidate_ops_, scratch_range_candidate_seqs_,
        scratch_range_seq_sort_keys_, scratch_range_group_sort_keys_);
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
  struct Run {
    thrust::device_vector<std::uint32_t> keys;
    thrust::device_vector<std::uint32_t> values;
    thrust::device_vector<std::uint32_t> ops;
    thrust::device_vector<std::uint32_t> seqs;
    thrust::device_vector<std::uint32_t> bucket_max;
    std::size_t size_class = 0;
  };

  template <class T>
  static T* raw_or_null(thrust::device_vector<T>& v) {
    return v.empty() ? nullptr : thrust::raw_pointer_cast(v.data());
  }
  template <class T>
  static const T* raw_or_null(const thrust::device_vector<T>& v) {
    return v.empty() ? nullptr : thrust::raw_pointer_cast(v.data());
  }
  template <class T>
  static std::size_t device_bytes(const thrust::device_vector<T>& v) {
    return v.size() * sizeof(T);
  }
  template <class... Vecs>
  static std::size_t device_bytes_all(const Vecs&... vecs) {
    return (std::size_t{0} + ... + device_bytes(vecs));
  }

  // ---- Run-layer batch helpers (unchanged) -------------------------------

  void fill_constant_ops(thrust::device_vector<std::uint32_t>& ops,
                         std::uint32_t op, cudaStream_t stream) {
    if (ops.empty()) return;
    const int block = 256;
    const int grid = static_cast<int>((ops.size() + block - 1) / block);
    gpulsmopt_detail::fill_constant_op_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(ops), ops.size(), op);
    CUDA_CHECK(cudaGetLastError());
  }

  void widen_ops(const std::uint8_t* input,
                 thrust::device_vector<std::uint32_t>& output,
                 cudaStream_t stream) {
    if (output.empty()) return;
    const int block = 256;
    const int grid = static_cast<int>((output.size() + block - 1) / block);
    gpulsmopt_detail::widen_ops_kernel<<<grid, block, 0, stream>>>(
        input, raw_or_null(output), output.size());
    CUDA_CHECK(cudaGetLastError());
  }

  void fill_constant_u32(thrust::device_vector<std::uint32_t>& values,
                         std::uint32_t value, cudaStream_t stream) {
    if (values.empty()) return;
    const int block = 256;
    const int grid = static_cast<int>((values.size() + block - 1) / block);
    gpulsmopt_detail::fill_constant_u32_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(values), values.size(), value);
    CUDA_CHECK(cudaGetLastError());
  }

  std::uint32_t allocate_sequence() {
    if (next_seq_ >= 0x7fffffffu) {
      throw std::overflow_error(
          "Design B sequence space exhausted; rebuild with wider packing");
    }
    return next_seq_++;
  }

  void resize_range_candidates(std::size_t count) {
    scratch_range_candidate_queries_.resize(count);
    scratch_range_candidate_keys_.resize(count);
    scratch_range_candidate_values_.resize(count);
    scratch_range_candidate_ops_.resize(count);
    scratch_range_candidate_seqs_.resize(count);
    scratch_range_seq_sort_keys_.resize(count);
    scratch_range_group_sort_keys_.resize(count);
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

  void fill_batch_sort_keys(const thrust::device_vector<std::uint32_t>& keys,
                            thrust::device_vector<std::uint64_t>& sort_keys,
                            cudaStream_t stream) {
    if (keys.empty()) return;
    const int block = 256;
    const int grid = static_cast<int>((keys.size() + block - 1) / block);
    gpulsmopt_detail::fill_batch_sort_keys_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(keys), raw_or_null(sort_keys), keys.size());
    CUDA_CHECK(cudaGetLastError());
  }

  void fill_drain_sort_keys(const thrust::device_vector<std::uint32_t>& keys,
                            const thrust::device_vector<std::uint32_t>& seqs,
                            thrust::device_vector<std::uint64_t>& sort_keys,
                            std::size_t offset, std::size_t count,
                            cudaStream_t stream) {
    if (count == 0) return;
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    gpulsmopt_detail::fill_drain_sort_keys_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(keys) + offset, raw_or_null(seqs) + offset,
        raw_or_null(sort_keys) + offset, count);
    CUDA_CHECK(cudaGetLastError());
  }

  void dedup_run_by_last_arrival(Run* run, cudaStream_t stream) {
    auto policy = thrust::cuda::par.on(stream);
    scratch_sort_keys_.resize(run->keys.size());
    fill_batch_sort_keys(run->keys, scratch_sort_keys_, stream);

    auto records = thrust::make_zip_iterator(
        thrust::make_tuple(run->keys.begin(), run->values.begin(),
                           run->ops.begin(), run->seqs.begin()));
    thrust::sort_by_key(policy, scratch_sort_keys_.begin(),
                        scratch_sort_keys_.end(), records);

    auto unique_end = thrust::unique_by_key(
        policy, run->keys.begin(), run->keys.end(),
        thrust::make_zip_iterator(thrust::make_tuple(
            run->values.begin(), run->ops.begin(), run->seqs.begin())));
    const std::size_t n =
        static_cast<std::size_t>(unique_end.first - run->keys.begin());
    run->keys.resize(n);
    run->values.resize(n);
    run->ops.resize(n);
    run->seqs.resize(n);
    build_run_metadata(run, stream);
  }

  void build_run_metadata(Run* run, cudaStream_t stream) {
    const std::size_t bucket_count =
        (run->keys.size() + gpulsmopt_detail::kBucketSlots - 1) /
        gpulsmopt_detail::kBucketSlots;
    run->bucket_max.resize(bucket_count);
    if (bucket_count == 0) return;
    const int block = 256;
    const int grid = static_cast<int>((bucket_count + block - 1) / block);
    gpulsmopt_detail::build_run_bucket_max_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(run->keys), run->keys.size(), raw_or_null(run->bucket_max));
    CUDA_CHECK(cudaGetLastError());
  }

  void append_constant_op_run(const std::uint32_t* keys,
                              const std::uint32_t* values, std::size_t count,
                              std::uint32_t op, cudaStream_t stream) {
    Run run;
    const std::uint32_t seq = allocate_sequence();
    run.size_class = 0;
    run.keys.resize(count);
    run.values.resize(count);
    run.ops.resize(count);
    run.seqs.resize(count);

    CUDA_CHECK(cudaMemcpyAsync(raw_or_null(run.keys), keys,
                               count * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToDevice, stream));
    if (values) {
      CUDA_CHECK(cudaMemcpyAsync(raw_or_null(run.values), values,
                                 count * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToDevice, stream));
    } else {
      auto policy = thrust::cuda::par.on(stream);
      thrust::fill(policy, run.values.begin(), run.values.end(), 0);
    }
    fill_constant_ops(run.ops, op, stream);
    fill_constant_u32(run.seqs, seq, stream);

    dedup_run_by_last_arrival(&run, stream);
    runs_.push_back(std::move(run));
    refresh_run_pointer_cache(stream);
    compact_overflow_classes(stream);
  }

  void compute_max_size_class() {
    const std::size_t run_budget =
        std::max<std::size_t>(batch_size_ * gpulsmopt_detail::kRunsPerClass,
                              max_elements_ / 20);
    std::size_t class_capacity = std::max<std::size_t>(1, batch_size_);
    max_size_class_ = 0;
    while (class_capacity < run_budget &&
           class_capacity <= std::numeric_limits<std::size_t>::max() /
                                 gpulsmopt_detail::kRunsPerClass) {
      class_capacity *= gpulsmopt_detail::kRunsPerClass;
      ++max_size_class_;
    }
  }

  std::size_t class_run_count(std::size_t size_class) const {
    std::size_t count = 0;
    for (const auto& run : runs_) {
      if (run.size_class == size_class) ++count;
    }
    return count;
  }

  void compact_overflow_classes(cudaStream_t stream) {
    while (compact_one_ready_class(stream, 2 * gpulsmopt_detail::kRunsPerClass)) {
    }
  }

  bool compact_one_ready_class(cudaStream_t stream,
                               std::size_t trigger_count) {
    for (std::size_t size_class = 0; size_class <= max_size_class_;
         ++size_class) {
      if (class_run_count(size_class) < trigger_count) continue;

      if (size_class == max_size_class_) {
        drain_runs_into_segments(stream);
        return false;
      }

      std::vector<std::size_t> selected;
      selected.reserve(gpulsmopt_detail::kRunsPerClass);
      for (std::size_t i = 0; i < runs_.size() &&
                              selected.size() < gpulsmopt_detail::kRunsPerClass;
           ++i) {
        if (runs_[i].size_class == size_class) selected.push_back(i);
      }
      merge_selected_runs(selected, size_class + 1, stream);
      return true;
    }
    return false;
  }

  void merge_selected_runs(const std::vector<std::size_t>& selected,
                           std::size_t output_class, cudaStream_t stream) {
    if (selected.empty()) return;

    std::vector<std::uint8_t> is_selected(runs_.size(), 0);
    std::size_t total = 0;
    for (std::size_t index : selected) {
      is_selected[index] = 1;
      total += runs_[index].keys.size();
    }

    Run output;
    output.size_class = output_class;
    output.keys.resize(total);
    output.values.resize(total);
    output.ops.resize(total);
    output.seqs.resize(total);
    scratch_sort_keys_.resize(total);

    auto policy = thrust::cuda::par.on(stream);
    std::size_t offset = 0;
    for (std::size_t index : selected) {
      const Run& run = runs_[index];
      const std::size_t n = run.keys.size();
      thrust::copy(policy, run.keys.begin(), run.keys.end(),
                   output.keys.begin() + offset);
      thrust::copy(policy, run.values.begin(), run.values.end(),
                   output.values.begin() + offset);
      thrust::copy(policy, run.ops.begin(), run.ops.end(),
                   output.ops.begin() + offset);
      thrust::copy(policy, run.seqs.begin(), run.seqs.end(),
                   output.seqs.begin() + offset);
      fill_drain_sort_keys(output.keys, output.seqs, scratch_sort_keys_, offset,
                           n, stream);
      offset += n;
    }

    auto records = thrust::make_zip_iterator(
        thrust::make_tuple(output.keys.begin(), output.values.begin(),
                           output.ops.begin(), output.seqs.begin()));
    thrust::sort_by_key(policy, scratch_sort_keys_.begin(),
                        scratch_sort_keys_.end(), records);
    auto unique_end = thrust::unique_by_key(
        policy, output.keys.begin(), output.keys.end(),
        thrust::make_zip_iterator(thrust::make_tuple(
            output.values.begin(), output.ops.begin(), output.seqs.begin())));
    const std::size_t output_count =
        static_cast<std::size_t>(unique_end.first - output.keys.begin());
    output.keys.resize(output_count);
    output.values.resize(output_count);
    output.ops.resize(output_count);
    output.seqs.resize(output_count);
    build_run_metadata(&output, stream);

    std::vector<Run> kept;
    kept.reserve(runs_.size() - selected.size() + 1);
    for (std::size_t i = 0; i < runs_.size(); ++i) {
      if (!is_selected[i]) kept.push_back(std::move(runs_[i]));
    }
    kept.push_back(std::move(output));
    runs_ = std::move(kept);
    refresh_run_pointer_cache(stream);
  }

  std::size_t build_resolved_incoming_device(cudaStream_t stream) {
    const std::size_t total = total_run_elements();
    if (total == 0) return 0;

    auto policy = thrust::cuda::par.on(stream);
    scratch_sort_keys_.resize(total);
    scratch_incoming_keys_.resize(total);
    scratch_incoming_values_.resize(total);
    scratch_incoming_ops_.resize(total);
    scratch_incoming_seqs_.resize(total);

    std::size_t offset = 0;
    for (const auto& run : runs_) {
      const std::size_t n = run.keys.size();
      thrust::copy(policy, run.keys.begin(), run.keys.end(),
                   scratch_incoming_keys_.begin() + offset);
      thrust::copy(policy, run.values.begin(), run.values.end(),
                   scratch_incoming_values_.begin() + offset);
      thrust::copy(policy, run.ops.begin(), run.ops.end(),
                   scratch_incoming_ops_.begin() + offset);
      thrust::copy(policy, run.seqs.begin(), run.seqs.end(),
                   scratch_incoming_seqs_.begin() + offset);
      fill_drain_sort_keys(scratch_incoming_keys_, scratch_incoming_seqs_,
                           scratch_sort_keys_, offset, n, stream);
      offset += n;
    }

    auto records = thrust::make_zip_iterator(
        thrust::make_tuple(scratch_incoming_keys_.begin(),
                           scratch_incoming_values_.begin(),
                           scratch_incoming_ops_.begin(),
                           scratch_incoming_seqs_.begin()));
    thrust::sort_by_key(policy, scratch_sort_keys_.begin(),
                        scratch_sort_keys_.end(), records);

    auto unique_end = thrust::unique_by_key(
        policy, scratch_incoming_keys_.begin(), scratch_incoming_keys_.end(),
        thrust::make_zip_iterator(thrust::make_tuple(
            scratch_incoming_values_.begin(), scratch_incoming_ops_.begin(),
            scratch_incoming_seqs_.begin())));
    const std::size_t count = static_cast<std::size_t>(
        unique_end.first - scratch_incoming_keys_.begin());
    scratch_incoming_keys_.resize(count);
    scratch_incoming_values_.resize(count);
    scratch_incoming_ops_.resize(count);
    scratch_incoming_seqs_.resize(count);
    return count;
  }

  // ---- Segmented bottom layer -------------------------------------------

  void initialize_segmented_storage(cudaStream_t stream) {
    pool_capacity_ = 0;
    free_ids_.clear();
    h_dir_seg_id_.clear();
    h_dir_boundary_.clear();
    h_dir_live_.clear();
    const std::size_t initial =
        max_elements_ == 0
            ? 4
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
                 pool_keys_.begin() + slot_base + gpulsmopt_detail::kSegmentSlots,
                 gpulsmopt_detail::kEmptyKey);
    thrust::fill(policy, pool_values_.begin() + slot_base,
                 pool_values_.begin() + slot_base + gpulsmopt_detail::kSegmentSlots,
                 0u);
    thrust::fill(policy, pool_valid_.begin() + slot_base,
                 pool_valid_.begin() + slot_base + gpulsmopt_detail::kSegmentSlots,
                 std::uint8_t{0});
    thrust::fill(
        policy, seg_bucket_max_.begin() + meta_base,
        seg_bucket_max_.begin() + meta_base + gpulsmopt_detail::kSegmentBuckets,
        boundary);
    thrust::fill(
        policy, seg_bucket_live_.begin() + meta_base,
        seg_bucket_live_.begin() + meta_base + gpulsmopt_detail::kSegmentBuckets,
        0u);
  }

  void grow_pool(std::size_t new_capacity) {
    if (new_capacity <= pool_capacity_) return;
    pool_keys_.resize(new_capacity * gpulsmopt_detail::kSegmentSlots,
                      gpulsmopt_detail::kEmptyKey);
    pool_values_.resize(new_capacity * gpulsmopt_detail::kSegmentSlots, 0u);
    pool_valid_.resize(new_capacity * gpulsmopt_detail::kSegmentSlots, 0u);
    seg_bucket_max_.resize(new_capacity * gpulsmopt_detail::kSegmentBuckets,
                           gpulsmopt_detail::kEmptyKey);
    seg_bucket_live_.resize(new_capacity * gpulsmopt_detail::kSegmentBuckets, 0u);
    for (std::size_t id = pool_capacity_; id < new_capacity; ++id) {
      free_ids_.push_back(static_cast<std::uint32_t>(id));
    }
    pool_capacity_ = new_capacity;
  }

  std::uint32_t alloc_segment() {
    if (free_ids_.empty()) grow_pool(std::max<std::size_t>(pool_capacity_ * 2,
                                                           pool_capacity_ + 1));
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
    std::vector<std::uint32_t> prefix(d + 1);
    std::uint32_t acc = 0;
    prefix[0] = 0;
    for (std::size_t i = 0; i < d; ++i) {
      acc += h_dir_live_[i];
      prefix[i + 1] = acc;
    }
    auto copy = [&](thrust::device_vector<std::uint32_t>& dst,
                    const std::vector<std::uint32_t>& src) {
      if (src.empty()) return;
      CUDA_CHECK(cudaMemcpyAsync(raw_or_null(dst), src.data(),
                                 src.size() * sizeof(std::uint32_t),
                                 cudaMemcpyHostToDevice, stream));
    };
    copy(d_dir_seg_id_, h_dir_seg_id_);
    copy(d_dir_boundary_, h_dir_boundary_);
    copy(d_dir_prefix_, prefix);
  }

  template <class T>
  void upload_vec(thrust::device_vector<T>& dst, const std::vector<T>& src,
                  cudaStream_t stream) {
    dst.resize(src.size());
    if (!src.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(raw_or_null(dst), src.data(),
                                 src.size() * sizeof(T), cudaMemcpyHostToDevice,
                                 stream));
    }
  }

  void drain_runs_into_segments(cudaStream_t stream) {
    if (runs_.empty()) return;
    const std::size_t incoming_count = build_resolved_incoming_device(stream);
    if (incoming_count == 0) {
      runs_.clear();
      refresh_run_pointer_cache(stream);
      return;
    }
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

    // 2. Run-length-encode ordinals -> (dirty ordinal, incoming count).
    seg_dirty_ord_.resize(incoming_count);
    seg_dirty_count_.resize(incoming_count);
    auto rle_end = thrust::reduce_by_key(
        policy, seg_inc_ordinal_.begin(),
        seg_inc_ordinal_.begin() + incoming_count,
        thrust::make_constant_iterator<std::uint32_t>(1u),
        seg_dirty_ord_.begin(), seg_dirty_count_.begin());
    const std::size_t m =
        static_cast<std::size_t>(rle_end.first - seg_dirty_ord_.begin());

    std::vector<std::uint32_t> dirty_ord(m), dirty_in_count(m);
    CUDA_CHECK(cudaMemcpyAsync(dirty_ord.data(), raw_or_null(seg_dirty_ord_),
                               m * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(dirty_in_count.data(),
                               raw_or_null(seg_dirty_count_),
                               m * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 3. Per-dirty-segment host metadata: incoming slice into the resolved
    //    incoming array, plus the segment's current directory entry.
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

    // 3b. Classify each dirty segment: in-place fast path (no bucket would
    //     overflow its 32 slots) vs. full rebuild + 1->k split.
    seg_slow_.resize(m);
    seg_new_live_.resize(m);
    gpulsmopt_detail::seg_classify_inplace_kernel<<<static_cast<unsigned>(m), 256, 0,
                                              stream>>>(
        raw_or_null(seg_d_dirty_seg_id_), raw_or_null(seg_d_dirty_in_begin_),
        raw_or_null(seg_d_dirty_in_end_), m, raw_or_null(scratch_incoming_keys_),
        raw_or_null(scratch_incoming_ops_), raw_or_null(pool_keys_),
        raw_or_null(pool_valid_), raw_or_null(seg_bucket_max_),
        raw_or_null(seg_bucket_live_), raw_or_null(seg_slow_),
        raw_or_null(seg_new_live_));
    CUDA_CHECK(cudaGetLastError());
    std::vector<std::uint32_t> seg_slow(m), seg_new_live(m);
    CUDA_CHECK(cudaMemcpyAsync(seg_slow.data(), raw_or_null(seg_slow_),
                               m * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(seg_new_live.data(), raw_or_null(seg_new_live_),
                               m * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 3c. Partition dirty segments into fast (edit in place) and slow (rebuild).
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

    // 4. Fast path: edit segments in place using existing bucket gaps. Fast
    //    segments keep their physical id and routing boundary.
    if (!fast_seg_id.empty()) {
      upload_vec(seg_fast_seg_id_, fast_seg_id, stream);
      upload_vec(seg_fast_in_begin_, fast_in_begin, stream);
      upload_vec(seg_fast_in_end_, fast_in_end, stream);
      gpulsmopt_detail::seg_apply_inplace_kernel<<<
          static_cast<unsigned>(fast_seg_id.size()), 256, 0, stream>>>(
          raw_or_null(seg_fast_seg_id_), raw_or_null(seg_fast_in_begin_),
          raw_or_null(seg_fast_in_end_), fast_seg_id.size(),
          raw_or_null(scratch_incoming_keys_),
          raw_or_null(scratch_incoming_values_),
          raw_or_null(scratch_incoming_ops_), raw_or_null(pool_keys_),
          raw_or_null(pool_values_), raw_or_null(pool_valid_),
          raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_));
      CUDA_CHECK(cudaGetLastError());
    }

    // 5. Slow path: full rebuild + 1->k split for overflowing segments only.
    const std::size_t ms = slow_j.size();
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

      // Gather candidates (old sheet records seq 0 + incoming seq 1).
      resize_seg_candidates(candidate_count);
      seg_sheet_cursor_.resize(ms);
      thrust::fill(policy, seg_sheet_cursor_.begin(), seg_sheet_cursor_.end(),
                   0u);
      gpulsmopt_detail::seg_gather_candidates_kernel<<<static_cast<unsigned>(ms), 256,
                                                  0, stream>>>(
          raw_or_null(pool_keys_), raw_or_null(pool_values_),
          raw_or_null(pool_valid_), raw_or_null(seg_d_dirty_seg_id_),
          raw_or_null(seg_d_candidate_offset_),
          raw_or_null(seg_d_dirty_old_live_), raw_or_null(seg_d_dirty_in_begin_),
          raw_or_null(seg_d_dirty_in_end_), raw_or_null(scratch_incoming_keys_),
          raw_or_null(scratch_incoming_values_),
          raw_or_null(scratch_incoming_ops_), ms, raw_or_null(seg_cand_seg_),
          raw_or_null(seg_cand_key_), raw_or_null(seg_cand_value_),
          raw_or_null(seg_cand_op_), raw_or_null(seg_cand_seq_),
          raw_or_null(seg_sheet_cursor_));
      CUDA_CHECK(cudaGetLastError());

      // Resolve newest per (segment, key); drop tombstone winners.
      std::size_t live_count = 0;
      if (candidate_count > 0) {
        const int block = 256;
        const int grid =
            static_cast<int>((candidate_count + block - 1) / block);
        gpulsmopt_detail::seg_make_resolve_keys_kernel<<<grid, block, 0, stream>>>(
            raw_or_null(seg_cand_seg_), raw_or_null(seg_cand_key_),
            raw_or_null(seg_cand_seq_), candidate_count,
            raw_or_null(seg_cand_seq_sort_), raw_or_null(seg_cand_group_sort_));
        CUDA_CHECK(cudaGetLastError());

        auto payload1 = thrust::make_zip_iterator(thrust::make_tuple(
            seg_cand_group_sort_.begin(), seg_cand_seg_.begin(),
            seg_cand_key_.begin(), seg_cand_value_.begin(),
            seg_cand_op_.begin()));
        thrust::stable_sort_by_key(policy, seg_cand_seq_sort_.begin(),
                                   seg_cand_seq_sort_.begin() + candidate_count,
                                   payload1);
        auto payload2 = thrust::make_zip_iterator(thrust::make_tuple(
            seg_cand_seg_.begin(), seg_cand_key_.begin(),
            seg_cand_value_.begin(), seg_cand_op_.begin()));
        thrust::stable_sort_by_key(policy, seg_cand_group_sort_.begin(),
                                   seg_cand_group_sort_.begin() + candidate_count,
                                   payload2);
        auto uend = thrust::unique_by_key(
            policy, seg_cand_group_sort_.begin(),
            seg_cand_group_sort_.begin() + candidate_count, payload2);
        const std::size_t unique_count =
            static_cast<std::size_t>(uend.first - seg_cand_group_sort_.begin());
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(
            seg_cand_seg_.begin(), seg_cand_key_.begin(),
            seg_cand_value_.begin(), seg_cand_op_.begin()));
        auto fend = thrust::remove_if(policy, begin, begin + unique_count,
                                      gpulsmopt_detail::DrainTombstonePredicate{});
        live_count = static_cast<std::size_t>(fend - begin);
      }

      // Count survivors per slow segment.
      seg_dirty_live_.resize(ms);
      thrust::fill(policy, seg_dirty_live_.begin(), seg_dirty_live_.end(), 0u);
      if (live_count > 0) {
        const int block = 256;
        const int grid = static_cast<int>((live_count + block - 1) / block);
        gpulsmopt_detail::seg_count_survivors_kernel<<<grid, block, 0, stream>>>(
            raw_or_null(seg_cand_seg_), live_count,
            raw_or_null(seg_dirty_live_));
        CUDA_CHECK(cudaGetLastError());
      }
      std::vector<std::uint32_t> dirty_live(ms);
      CUDA_CHECK(cudaMemcpyAsync(dirty_live.data(), raw_or_null(seg_dirty_live_),
                                 ms * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      // Decide split factor k; allocate output ids.
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
          const std::uint32_t hi = std::min<std::uint32_t>((local + 1) * per, L);
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
        gpulsmopt_detail::seg_output_boundaries_kernel<<<grid, block, 0, stream>>>(
            raw_or_null(seg_d_out_dirty_), raw_or_null(seg_d_out_local_),
            raw_or_null(seg_d_out_k_), output_count,
            raw_or_null(seg_dirty_live_), raw_or_null(seg_d_dirty_live_offset_),
            raw_or_null(seg_d_dirty_old_boundary_), raw_or_null(seg_cand_key_),
            raw_or_null(seg_d_out_boundary_));
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

    runs_.clear();
    refresh_run_pointer_cache(stream);
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
      gpulsmopt_detail::seg_scatter_survivors_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(seg_cand_seg_), raw_or_null(seg_cand_key_),
          raw_or_null(seg_cand_value_), live_count, raw_or_null(seg_dirty_live_),
          raw_or_null(seg_d_dirty_live_offset_), raw_or_null(seg_d_dirty_k_),
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
          output_count, raw_or_null(pool_keys_), raw_or_null(pool_valid_),
          raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_));
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Greedily merge maximal runs of adjacent directory entries that start at an
  // underfull segment and whose combined live still fits one segment. Each
  // group is gathered (seq 0, key order) and repacked into a single segment.
  void merge_underfull_segments(cudaStream_t stream) {
    const std::size_t n = h_dir_seg_id_.size();
    if (n < 2) return;
    const std::uint32_t watermark = target_segment_live_ / 2;

    struct Group {
      std::size_t begin;  // first ord (inclusive)
      std::size_t end;    // last ord (exclusive)
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
      while (j + 1 < n &&
             sum + h_dir_live_[j + 1] <= target_segment_live_) {
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
    if (groups.empty()) return;

    auto policy = thrust::cuda::par.on(stream);
    const std::size_t group_count = groups.size();

    // Flatten the source-segment lists and build per-group accounting.
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
      out_boundary[g] = h_dir_boundary_[groups[g].end - 1];  // rightmost
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

    // Gather the live records from each group's source segments.
    resize_seg_candidates(candidate_count);
    seg_sheet_cursor_.resize(group_count);
    thrust::fill(policy, seg_sheet_cursor_.begin(), seg_sheet_cursor_.end(), 0u);
    if (group_count > 0 && candidate_count > 0) {
      gpulsmopt_detail::seg_merge_gather_kernel<<<static_cast<unsigned>(group_count),
                                             256, 0, stream>>>(
          raw_or_null(pool_keys_), raw_or_null(pool_values_),
          raw_or_null(pool_valid_), raw_or_null(seg_group_src_seg_),
          raw_or_null(seg_group_src_begin_), raw_or_null(seg_group_src_end_),
          raw_or_null(seg_d_candidate_offset_), group_count,
          raw_or_null(seg_cand_seg_), raw_or_null(seg_cand_key_),
          raw_or_null(seg_cand_value_), raw_or_null(seg_sheet_cursor_));
      CUDA_CHECK(cudaGetLastError());

      // Order each group's slice by key (keys are unique across the group).
      const int block = 256;
      const int grid = static_cast<int>((candidate_count + block - 1) / block);
      gpulsmopt_detail::seg_merge_sort_keys_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(seg_cand_seg_), raw_or_null(seg_cand_key_),
          candidate_count, raw_or_null(seg_cand_group_sort_));
      CUDA_CHECK(cudaGetLastError());
      auto payload = thrust::make_zip_iterator(thrust::make_tuple(
          seg_cand_seg_.begin(), seg_cand_key_.begin(),
          seg_cand_value_.begin()));
      thrust::stable_sort_by_key(policy, seg_cand_group_sort_.begin(),
                                 seg_cand_group_sort_.begin() + candidate_count,
                                 payload);
    }

    // Repack into one segment per group (k == 1, boundaries already set).
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
    for (std::uint32_t v : h_dir_live_) total += v;
    sheet_live_count_ = total;
  }

  std::size_t total_run_elements() const {
    std::size_t total = 0;
    for (const auto& run : runs_) total += run.keys.size();
    return total;
  }

  void refresh_run_pointer_cache(cudaStream_t stream) {
    std::vector<const std::uint32_t*> key_ptrs;
    std::vector<const std::uint32_t*> value_ptrs;
    std::vector<const std::uint32_t*> op_ptrs;
    std::vector<const std::uint32_t*> seq_ptrs;
    std::vector<const std::uint32_t*> bucket_max_ptrs;
    std::vector<std::size_t> sizes;
    std::vector<std::uint32_t> descriptor_run;
    std::vector<std::uint32_t> descriptor_bucket;
    key_ptrs.reserve(runs_.size());
    value_ptrs.reserve(runs_.size());
    op_ptrs.reserve(runs_.size());
    seq_ptrs.reserve(runs_.size());
    bucket_max_ptrs.reserve(runs_.size());
    sizes.reserve(runs_.size());

    for (std::size_t run_index = 0; run_index < runs_.size(); ++run_index) {
      const auto& run = runs_[run_index];
      key_ptrs.push_back(raw_or_null(run.keys));
      value_ptrs.push_back(raw_or_null(run.values));
      op_ptrs.push_back(raw_or_null(run.ops));
      seq_ptrs.push_back(raw_or_null(run.seqs));
      bucket_max_ptrs.push_back(raw_or_null(run.bucket_max));
      sizes.push_back(run.keys.size());
      for (std::size_t bucket = 0; bucket < run.bucket_max.size(); ++bucket) {
        descriptor_run.push_back(static_cast<std::uint32_t>(run_index));
        descriptor_bucket.push_back(static_cast<std::uint32_t>(bucket));
      }
    }

    run_key_ptrs_.resize(key_ptrs.size());
    run_value_ptrs_.resize(value_ptrs.size());
    run_op_ptrs_.resize(op_ptrs.size());
    run_seq_ptrs_.resize(seq_ptrs.size());
    run_bucket_max_ptrs_.resize(bucket_max_ptrs.size());
    run_sizes_.resize(sizes.size());
    run_bucket_descriptor_run_.resize(descriptor_run.size());
    run_bucket_descriptor_bucket_.resize(descriptor_bucket.size());
    if (!key_ptrs.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(raw_or_null(run_key_ptrs_), key_ptrs.data(),
                                 key_ptrs.size() * sizeof(key_ptrs[0]),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(raw_or_null(run_value_ptrs_), value_ptrs.data(),
                                 value_ptrs.size() * sizeof(value_ptrs[0]),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(raw_or_null(run_op_ptrs_), op_ptrs.data(),
                                 op_ptrs.size() * sizeof(op_ptrs[0]),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(raw_or_null(run_seq_ptrs_), seq_ptrs.data(),
                                 seq_ptrs.size() * sizeof(seq_ptrs[0]),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          raw_or_null(run_bucket_max_ptrs_), bucket_max_ptrs.data(),
          bucket_max_ptrs.size() * sizeof(bucket_max_ptrs[0]),
          cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(raw_or_null(run_sizes_), sizes.data(),
                                 sizes.size() * sizeof(sizes[0]),
                                 cudaMemcpyHostToDevice, stream));
    }
    if (!descriptor_run.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(
          raw_or_null(run_bucket_descriptor_run_), descriptor_run.data(),
          descriptor_run.size() * sizeof(descriptor_run[0]),
          cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          raw_or_null(run_bucket_descriptor_bucket_), descriptor_bucket.data(),
          descriptor_bucket.size() * sizeof(descriptor_bucket[0]),
          cudaMemcpyHostToDevice, stream));
    }
  }

  // ---- Configuration / run-layer state -----------------------------------
  std::size_t max_elements_ = 0;
  std::size_t batch_size_ = 0;
  std::uint32_t target_fill_ = 23;
  std::uint32_t target_segment_live_ = 0;
  std::size_t max_size_class_ = 0;
  std::size_t sheet_live_count_ = 0;
  std::uint32_t next_seq_ = 1;
  mutable std::shared_mutex snapshot_mutex_;

  std::vector<Run> runs_;
  thrust::device_vector<const std::uint32_t*> run_key_ptrs_;
  thrust::device_vector<const std::uint32_t*> run_value_ptrs_;
  thrust::device_vector<const std::uint32_t*> run_op_ptrs_;
  thrust::device_vector<const std::uint32_t*> run_seq_ptrs_;
  thrust::device_vector<const std::uint32_t*> run_bucket_max_ptrs_;
  thrust::device_vector<std::size_t> run_sizes_;
  thrust::device_vector<std::uint32_t> run_bucket_descriptor_run_;
  thrust::device_vector<std::uint32_t> run_bucket_descriptor_bucket_;

  // ---- Segmented bottom layer state --------------------------------------
  std::size_t pool_capacity_ = 0;
  std::vector<std::uint32_t> free_ids_;
  thrust::device_vector<std::uint32_t> pool_keys_;
  thrust::device_vector<std::uint32_t> pool_values_;
  thrust::device_vector<std::uint8_t> pool_valid_;
  thrust::device_vector<std::uint32_t> seg_bucket_max_;
  thrust::device_vector<std::uint32_t> seg_bucket_live_;

  std::vector<std::uint32_t> h_dir_seg_id_;
  std::vector<std::uint32_t> h_dir_boundary_;
  std::vector<std::uint32_t> h_dir_live_;
  thrust::device_vector<std::uint32_t> d_dir_seg_id_;
  thrust::device_vector<std::uint32_t> d_dir_boundary_;
  thrust::device_vector<std::uint32_t> d_dir_prefix_;

  // ---- Run-layer / drain scratch -----------------------------------------
  thrust::device_vector<std::uint64_t> scratch_sort_keys_;
  thrust::device_vector<std::uint32_t> scratch_incoming_keys_;
  thrust::device_vector<std::uint32_t> scratch_incoming_values_;
  thrust::device_vector<std::uint32_t> scratch_incoming_ops_;
  thrust::device_vector<std::uint32_t> scratch_incoming_seqs_;

  // ---- Drain / merge device scratch --------------------------------------
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
  // In-place fast-path scratch (improvement #1).
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

  // ---- Query scratch ------------------------------------------------------
  thrust::device_vector<std::uint32_t> scratch_query_keys_;
  thrust::device_vector<std::uint32_t> scratch_query_indices_;
  thrust::device_vector<unsigned long long> scratch_query_winners_;
  thrust::device_vector<std::uint32_t> scratch_query_values_;
  thrust::device_vector<std::uint8_t> scratch_query_found_;
  thrust::device_vector<std::uint32_t> scratch_range_source_counts_;
  thrust::device_vector<std::uint32_t> scratch_range_source_offsets_;
  thrust::device_vector<std::uint32_t> scratch_range_source_cursors_;
  thrust::device_vector<std::uint32_t> scratch_range_query_totals_;
  thrust::device_vector<std::uint32_t> scratch_range_query_offsets_;
  thrust::device_vector<std::uint32_t> scratch_range_candidate_queries_;
  thrust::device_vector<std::uint32_t> scratch_range_candidate_keys_;
  thrust::device_vector<std::uint32_t> scratch_range_candidate_values_;
  thrust::device_vector<std::uint32_t> scratch_range_candidate_ops_;
  thrust::device_vector<std::uint32_t> scratch_range_candidate_seqs_;
  thrust::device_vector<std::uint32_t> scratch_range_seq_sort_keys_;
  thrust::device_vector<std::uint64_t> scratch_range_group_sort_keys_;
};
