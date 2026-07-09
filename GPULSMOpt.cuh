#pragma once
#include "gpu_dictionary_adapter.cuh"

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef GPULSMOPT_PROFILE_INSERT
#include <chrono>
#include <cstdio>
#endif

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
#ifndef GPULSMOPT_C0_FLUSH_BUDGET
#define GPULSMOPT_C0_FLUSH_BUDGET (1 << 18)
#endif
#ifndef GPULSMOPT_LOOKUP_FLIX_MIN_BATCH
#define GPULSMOPT_LOOKUP_FLIX_MIN_BATCH (1 << 22)
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
constexpr std::uint32_t kC0LogMaxIndex = 0x00fffffeu;

#ifdef GPULSMOPT_PROFILE_INSERT
struct ScopedInsertPhaseTimer {
  cudaStream_t stream_;
  double *acc_;
  std::chrono::high_resolution_clock::time_point t0_;
  ScopedInsertPhaseTimer(cudaStream_t stream, double *acc)
      : stream_(stream), acc_(acc),
        t0_(std::chrono::high_resolution_clock::now()) {}
  ~ScopedInsertPhaseTimer() {
    cudaStreamSynchronize(stream_);
    const auto t1 = std::chrono::high_resolution_clock::now();
    *acc_ += std::chrono::duration<double, std::milli>(t1 - t0_).count();
  }
};
#define GPULSMOPT_PROF_CAT2(a, b) a##b
#define GPULSMOPT_PROF_CAT(a, b) GPULSMOPT_PROF_CAT2(a, b)
#define GPULSMOPT_PROF_PHASE(acc)                                              \
  gpulsmopt_detail::ScopedInsertPhaseTimer GPULSMOPT_PROF_CAT(prof_phase_,     \
                                                              __LINE__)(       \
      stream, &(acc))
#else
#define GPULSMOPT_PROF_PHASE(acc)                                              \
  do {                                                                         \
  } while (false)
#endif

struct DeviceKeyBatch {
  const std::uint32_t *keys = nullptr;
  std::size_t count = 0;
};

__host__ __device__ inline std::uint32_t ceil_div_u32(std::uint32_t a,
                                                      std::uint32_t b) {
  return (a + b - 1) / b;
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

__global__ void seg_route_buckets_kernel(
    const std::uint32_t *keys, const std::uint32_t *seg_ordinal,
    std::size_t n, const std::uint32_t *dir_seg_id,
    const std::uint32_t *seg_bucket_max, std::uint32_t *out_bucket) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint32_t ord = seg_ordinal[i];
  const std::uint32_t seg = dir_seg_id[ord];
  const std::size_t meta =
      static_cast<std::size_t>(seg) * kSegmentBuckets;
  std::size_t bucket =
      lower_bound_u32(seg_bucket_max + meta, kSegmentBuckets, keys[i]);
  if (bucket >= kSegmentBuckets)
    bucket = kSegmentBuckets - 1;
  out_bucket[i] =
      ord * static_cast<std::uint32_t>(kSegmentBuckets) +
      static_cast<std::uint32_t>(bucket);
}

__global__ void seg_pull_dirty_segments_kernel(
    const std::uint32_t *keys, std::size_t n,
    const std::uint32_t *dir_boundary, std::size_t dir_count,
    const std::uint32_t *dir_seg_id, const std::uint32_t *dir_prefix,
    const std::uint32_t *dir_value_sum, std::uint32_t *dirty_ord,
    std::uint32_t *dirty_count, std::uint32_t *dirty_seg_id,
    std::uint32_t *dirty_old_boundary, std::uint32_t *dirty_old_live,
    std::uint32_t *dirty_old_value_sum, std::uint32_t *dirty_in_begin,
    std::uint32_t *dirty_in_end, std::uint32_t *slow,
    std::uint32_t *new_live, std::uint32_t *new_value_sum,
    std::uint32_t *counters) {
  const std::size_t ord = blockIdx.x * blockDim.x + threadIdx.x;
  if (ord >= dir_count)
    return;
  const std::size_t begin =
      ord == 0 ? 0 : upper_bound_u32(keys, n, dir_boundary[ord - 1]);
  const std::size_t end =
      ord + 1 == dir_count ? n : upper_bound_u32(keys, n, dir_boundary[ord]);
  if (begin == end)
    return;
  const std::uint32_t out = atomicAdd(counters, 1u);
  const std::uint32_t incoming = static_cast<std::uint32_t>(end - begin);
  const std::uint32_t old_live = dir_prefix[ord + 1] - dir_prefix[ord];
  dirty_ord[out] = static_cast<std::uint32_t>(ord);
  dirty_count[out] = incoming;
  dirty_seg_id[out] = dir_seg_id[ord];
  dirty_old_boundary[out] = dir_boundary[ord];
  dirty_old_live[out] = old_live;
  dirty_old_value_sum[out] = dir_value_sum[ord];
  dirty_in_begin[out] = static_cast<std::uint32_t>(begin);
  dirty_in_end[out] = static_cast<std::uint32_t>(end);
  slow[out] = incoming > GPULSMOPT_INPLACE_MAX_INCOMING ? 1u : 0u;
  new_live[out] = old_live + incoming;
  new_value_sum[out] = dir_value_sum[ord];
}

__global__ void seg_pull_dirty_buckets_kernel(
    const std::uint32_t *keys, const std::uint32_t *dirty_ord,
    const std::uint32_t *dirty_count, const std::uint32_t *dirty_seg_id,
    const std::uint32_t *dirty_in_begin, std::size_t dirty_count_len,
    const std::uint32_t *seg_bucket_max,
    const std::uint32_t *seg_bucket_live, std::uint32_t *dirty_bucket,
    std::uint32_t *dirty_bucket_count, std::uint32_t *dirty_bucket_begin,
    std::uint32_t *dirty_bucket_dirty, std::uint32_t *slow,
    std::uint32_t *counters) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total =
      dirty_count_len * static_cast<std::size_t>(kSegmentBuckets);
  if (i >= total)
    return;
  const std::uint32_t dirty = static_cast<std::uint32_t>(i / kSegmentBuckets);
  const std::uint32_t bucket = static_cast<std::uint32_t>(i % kSegmentBuckets);
  const std::uint32_t ord = dirty_ord[dirty];
  const std::uint32_t seg = dirty_seg_id[dirty];
  const std::size_t meta =
      static_cast<std::size_t>(seg) * kSegmentBuckets;
  const std::uint32_t begin = dirty_in_begin[dirty];
  const std::uint32_t count = dirty_count[dirty];
  const std::uint32_t *base = keys + begin;
  const std::uint32_t b =
      bucket == 0
          ? 0u
          : static_cast<std::uint32_t>(
                upper_bound_u32(base, count, seg_bucket_max[meta + bucket - 1]));
  const std::uint32_t e =
      bucket + 1 == kSegmentBuckets
          ? count
          : static_cast<std::uint32_t>(
                upper_bound_u32(base, count, seg_bucket_max[meta + bucket]));
  if (b == e)
    return;
  const std::uint32_t out = atomicAdd(counters + 1, 1u);
  const std::uint32_t bucket_count = e - b;
  dirty_bucket[out] =
      ord * static_cast<std::uint32_t>(kSegmentBuckets) + bucket;
  dirty_bucket_count[out] = bucket_count;
  dirty_bucket_begin[out] = begin + b;
  dirty_bucket_dirty[out] = dirty;
  if (seg_bucket_live[meta + bucket] + bucket_count > kBucketSlots)
    atomicExch(slow + dirty, 1u);
}

__global__ void seg_emit_fast_slow_kernel(const std::uint32_t *slow,
                                          std::size_t count,
                                          std::uint32_t *slow_index,
                                          std::uint32_t *counters) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  if (slow[i]) {
    const std::uint32_t out = atomicAdd(counters + 1, 1u);
    slow_index[out] = static_cast<std::uint32_t>(i);
  } else {
    atomicAdd(counters, 1u);
  }
}

__global__ void seg_init_dirty_segments_kernel(
    const std::uint32_t *old_live, const std::uint32_t *old_value_sum,
    const std::uint32_t *in_begin, const std::uint32_t *in_end,
    std::size_t dirty_count, std::uint32_t *slow,
    std::uint32_t *new_live, std::uint32_t *new_value_sum) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dirty_count)
    return;
  const std::uint32_t incoming = in_end[i] - in_begin[i];
  slow[i] = incoming > GPULSMOPT_INPLACE_MAX_INCOMING ? 1u : 0u;
  new_live[i] = old_live[i] + incoming;
  new_value_sum[i] = old_value_sum[i];
}

__global__ void seg_classify_dirty_buckets_kernel(
    const std::uint32_t *dirty_bucket,
    const std::uint32_t *dirty_bucket_count, std::size_t bucket_count,
    const std::uint32_t *dirty_ord, std::size_t dirty_count,
    const std::uint32_t *dir_seg_id,
    const std::uint32_t *seg_bucket_live, std::uint32_t *slow) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= bucket_count)
    return;
  const std::uint32_t packed = dirty_bucket[i];
  const std::uint32_t ord = packed / kSegmentBuckets;
  const std::uint32_t bucket = packed % kSegmentBuckets;
  const std::size_t dirty =
      lower_bound_u32(dirty_ord, dirty_count, ord);
  const std::uint32_t seg = dir_seg_id[ord];
  const std::size_t meta =
      static_cast<std::size_t>(seg) * kSegmentBuckets;
  if (seg_bucket_live[meta + bucket] + dirty_bucket_count[i] >
      kBucketSlots)
    atomicExch(slow + dirty, 1u);
}

__global__ void seg_apply_dirty_buckets_kernel(
    const std::uint32_t *dirty_bucket,
    const std::uint32_t *dirty_bucket_count,
    const std::uint32_t *dirty_bucket_begin,
    const std::uint32_t *dirty_bucket_dirty, std::size_t bucket_count,
    const std::uint32_t *dirty_ord, std::size_t dirty_count,
    const std::uint32_t *slow, const std::uint32_t *dir_seg_id,
    const std::uint32_t *incoming_keys,
    const std::uint32_t *incoming_values, std::uint32_t *pool_keys,
    std::uint32_t *pool_values, std::uint8_t *pool_valid,
    std::uint32_t *seg_bucket_max, std::uint32_t *seg_bucket_live,
    std::uint32_t *seg_bucket_value_sum,
    std::uint32_t *new_value_sum) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= bucket_count)
    return;
  const std::uint32_t packed = dirty_bucket[i];
  const std::uint32_t ord = packed / kSegmentBuckets;
  const std::uint32_t bucket = packed % kSegmentBuckets;
  const std::size_t dirty =
      dirty_bucket_dirty ? dirty_bucket_dirty[i]
                         : lower_bound_u32(dirty_ord, dirty_count, ord);
  if (slow[dirty])
    return;
  const std::uint32_t seg = dir_seg_id[ord];
  const std::size_t meta =
      static_cast<std::size_t>(seg) * kSegmentBuckets + bucket;
  const std::size_t base =
      static_cast<std::size_t>(seg) * kSegmentSlots +
      static_cast<std::size_t>(bucket) * kBucketSlots;
  const std::uint32_t live = seg_bucket_live[meta];
  const std::uint32_t in_begin = dirty_bucket_begin[i];
  const std::uint32_t in_end = in_begin + dirty_bucket_count[i];
  std::uint32_t old_keys[kBucketSlots];
  std::uint32_t old_values[kBucketSlots];
  for (std::uint32_t j = 0; j < live; ++j) {
    old_keys[j] = pool_keys[base + j];
    old_values[j] = pool_values[base + j];
  }
  std::uint32_t old = 0;
  std::uint32_t incoming = in_begin;
  std::uint32_t out = 0;
  std::uint32_t sum = 0;
  std::uint32_t inserted_sum = 0;
  while (old < live && incoming < in_end) {
    std::uint32_t key;
    std::uint32_t value;
    if (old_keys[old] < incoming_keys[incoming]) {
      key = old_keys[old];
      value = old_values[old++];
    } else {
      key = incoming_keys[incoming];
      value = incoming_values[incoming++];
      inserted_sum += value;
    }
    pool_keys[base + out] = key;
    pool_values[base + out] = value;
    pool_valid[base + out] = 1u;
    sum += value;
    ++out;
  }
  while (old < live) {
    const std::uint32_t value = old_values[old];
    pool_keys[base + out] = old_keys[old++];
    pool_values[base + out] = value;
    pool_valid[base + out] = 1u;
    sum += value;
    ++out;
  }
  while (incoming < in_end) {
    const std::uint32_t value = incoming_values[incoming];
    pool_keys[base + out] = incoming_keys[incoming++];
    pool_values[base + out] = value;
    pool_valid[base + out] = 1u;
    sum += value;
    inserted_sum += value;
    ++out;
  }
  for (std::uint32_t j = out; j < kBucketSlots; ++j) {
    pool_keys[base + j] = kEmptyKey;
    pool_values[base + j] = 0u;
    pool_valid[base + j] = 0u;
  }
  seg_bucket_live[meta] = out;
  seg_bucket_max[meta] = pool_keys[base + out - 1];
  seg_bucket_value_sum[meta] = sum;
  atomicAdd(new_value_sum + dirty, inserted_sum);
}

__global__ void seg_init_delete_totals_kernel(
    const std::uint32_t *old_live, const std::uint32_t *old_value_sum,
    std::size_t dirty_count, std::uint32_t *new_live,
    std::uint32_t *new_value_sum) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dirty_count)
    return;
  new_live[i] = old_live[i];
  new_value_sum[i] = old_value_sum[i];
}

__global__ void seg_delete_dirty_buckets_kernel(
    const std::uint32_t *del_keys, const std::uint32_t *dirty_bucket,
    const std::uint32_t *dirty_bucket_count,
    const std::uint32_t *dirty_bucket_begin, std::size_t bucket_count,
    const std::uint32_t *dirty_ord, std::size_t dirty_count,
    const std::uint32_t *dir_seg_id, std::uint32_t *pool_keys,
    std::uint32_t *pool_values, std::uint8_t *pool_valid,
    std::uint32_t *seg_bucket_max, std::uint32_t *seg_bucket_live,
    std::uint32_t *seg_bucket_value_sum, std::uint32_t *new_live,
    std::uint32_t *new_value_sum) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= bucket_count)
    return;
  const std::uint32_t packed = dirty_bucket[i];
  const std::uint32_t ord = packed / kSegmentBuckets;
  const std::uint32_t bucket = packed % kSegmentBuckets;
  const std::size_t dirty =
      lower_bound_u32(dirty_ord, dirty_count, ord);
  const std::uint32_t seg = dir_seg_id[ord];
  const std::size_t meta =
      static_cast<std::size_t>(seg) * kSegmentBuckets + bucket;
  const std::size_t base =
      static_cast<std::size_t>(seg) * kSegmentSlots +
      static_cast<std::size_t>(bucket) * kBucketSlots;
  const std::uint32_t live = seg_bucket_live[meta];
  const std::uint32_t del_begin = dirty_bucket_begin[i];
  const std::uint32_t del_end = del_begin + dirty_bucket_count[i];
  std::uint32_t del = del_begin;
  std::uint32_t out = 0;
  std::uint32_t sum = 0;
  std::uint32_t removed = 0;
  std::uint32_t removed_sum = 0;
  for (std::uint32_t j = 0; j < live; ++j) {
    const std::uint32_t key = pool_keys[base + j];
    const std::uint32_t value = pool_values[base + j];
    while (del < del_end && del_keys[del] < key)
      ++del;
    if (del < del_end && del_keys[del] == key) {
      ++del;
      ++removed;
      removed_sum += value;
      continue;
    }
    pool_keys[base + out] = key;
    pool_values[base + out] = value;
    ++out;
    sum += value;
  }
  for (std::uint32_t j = out; j < kBucketSlots; ++j) {
    pool_keys[base + j] = kEmptyKey;
    pool_values[base + j] = 0u;
    pool_valid[base + j] = 0u;
  }
  seg_bucket_live[meta] = out;
  if (out > 0)
    seg_bucket_max[meta] = pool_keys[base + out - 1];
  seg_bucket_value_sum[meta] = sum;
  if (removed > 0) {
    atomicSub(new_live + dirty, removed);
    atomicSub(new_value_sum + dirty, removed_sum);
  }
}

__global__ void seg_make_dirty_plan_kernel(
    const std::uint32_t *dirty_ord, const std::uint32_t *dirty_count,
    std::size_t dirty_len, const std::uint32_t *dir_seg_id,
    const std::uint32_t *dir_boundary, const std::uint32_t *dir_prefix,
    const std::uint32_t *dir_value_sum,
    std::uint32_t *dirty_seg_id, std::uint32_t *dirty_old_boundary,
    std::uint32_t *dirty_old_live, std::uint32_t *dirty_old_value_sum,
    std::uint32_t *dirty_in_begin, std::uint32_t *dirty_in_end) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dirty_len)
    return;
  const std::uint32_t ord = dirty_ord[i];
  const std::uint32_t begin = dirty_in_begin[i];
  dirty_in_end[i] = begin + dirty_count[i];
  dirty_seg_id[i] = dir_seg_id[ord];
  dirty_old_boundary[i] = dir_boundary[ord];
  dirty_old_live[i] = dir_prefix[ord + 1] - dir_prefix[ord];
  dirty_old_value_sum[i] = dir_value_sum[ord];
}

__global__ void seg_gather_dirty_plan_kernel(
    const std::uint32_t *indices, std::size_t count,
    const std::uint32_t *dirty_seg_id,
    const std::uint32_t *dirty_old_boundary,
    const std::uint32_t *dirty_old_live,
    const std::uint32_t *dirty_in_begin,
    const std::uint32_t *dirty_in_end, std::uint32_t *out_seg_id,
    std::uint32_t *out_old_boundary, std::uint32_t *out_old_live,
    std::uint32_t *out_in_begin, std::uint32_t *out_in_end) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::uint32_t j = indices[i];
  out_seg_id[i] = dirty_seg_id[j];
  out_in_begin[i] = dirty_in_begin[j];
  out_in_end[i] = dirty_in_end[j];
  if (out_old_boundary)
    out_old_boundary[i] = dirty_old_boundary[j];
  if (out_old_live)
    out_old_live[i] = dirty_old_live[j];
}

__global__ void seg_make_slow_counts_kernel(
    const std::uint32_t *slow_old_live, const std::uint32_t *slow_in_begin,
    const std::uint32_t *slow_in_end, std::size_t slow_count,
    std::uint32_t *slow_inc_count, std::uint32_t *slow_cand_count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= slow_count)
    return;
  const std::uint32_t inc = slow_in_end[i] - slow_in_begin[i];
  slow_inc_count[i] = inc;
  slow_cand_count[i] = slow_old_live[i] + inc;
}

__global__ void seg_make_dirty_k_kernel(const std::uint32_t *dirty_live,
                                        std::size_t dirty_count,
                                        std::uint32_t target_live,
                                        std::uint32_t *dirty_k) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dirty_count)
    return;
  const std::uint32_t live = dirty_live[i];
  const std::uint32_t k =
      live <= kSegmentSlots ? 1u : ceil_div_u32(live, target_live);
  dirty_k[i] = k == 0u ? 1u : k;
}

__global__ void seg_output_total_kernel(const std::uint32_t *output_base,
                                        const std::uint32_t *dirty_k,
                                        std::size_t dirty_count,
                                        std::uint32_t *total) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  if (dirty_count == 0) {
    total[0] = 0u;
    return;
  }
  const std::size_t i = dirty_count - 1;
  total[0] = output_base[i] + dirty_k[i];
}

__global__ void seg_output_plan_kernel(
    const std::uint32_t *dirty_live, const std::uint32_t *dirty_k,
    const std::uint32_t *dirty_output_base, std::size_t dirty_count,
    std::uint32_t *out_dirty, std::uint32_t *out_local,
    std::uint32_t *out_k, std::uint32_t *out_live) {
  const std::size_t s = blockIdx.x;
  if (s >= dirty_count)
    return;
  const std::uint32_t live = dirty_live[s];
  const std::uint32_t k = dirty_k[s];
  const std::uint32_t base = dirty_output_base[s];
  const std::uint32_t per = k == 0 ? 1u : ceil_div_u32(live, k);
  for (std::uint32_t local = threadIdx.x; local < k; local += blockDim.x) {
    const std::uint32_t o = base + local;
    const std::uint32_t lo_raw = local * per;
    const std::uint32_t hi_raw = (local + 1u) * per;
    const std::uint32_t lo = lo_raw < live ? lo_raw : live;
    const std::uint32_t hi = hi_raw < live ? hi_raw : live;
    out_dirty[o] = static_cast<std::uint32_t>(s);
    out_local[o] = local;
    out_k[o] = k;
    out_live[o] = hi - lo;
  }
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

  const std::size_t p = lower_bound_u32(pool_keys + start, kBucketSlots, key);
  if (p < kBucketSlots && pool_keys[start + p] == key && pool_valid[start + p]) {
    if (out_value)
      *out_value = pool_values[start + p];
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
    const std::uint32_t live = seg_bucket_live[meta_base + b];
    const std::size_t lb = lower_bound_u32(pool_keys + start, live, lo);
    const std::size_t ub = upper_bound_u32(pool_keys + start, live, hi);
    count += static_cast<std::uint32_t>(ub - lb);
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

    const std::size_t lb = lower_bound_u32(pool_keys + start, kBucketSlots, lo);
    const std::size_t ub = upper_bound_u32(pool_keys + start, kBucketSlots, hi);
    for (std::size_t j = lb; j < ub; ++j)
      sum += pool_values[start + j];
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
    std::uint32_t *seg_bucket_value_sum, std::uint32_t *out_value_sum) {
  const std::size_t o = blockIdx.x;
  if (o >= output_count)
    return;
  const std::uint32_t seg_id = out_seg_id[o];
  const std::uint32_t boundary = out_boundary[o];
  const std::size_t slot_base =
      static_cast<std::size_t>(seg_id) * kSegmentSlots;
  const std::size_t meta_base =
      static_cast<std::size_t>(seg_id) * kSegmentBuckets;
  __shared__ unsigned long long s_value_sum;
  if (threadIdx.x == 0)
    s_value_sum = 0;
  __syncthreads();
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
    atomicAdd(&s_value_sum, static_cast<unsigned long long>(sum));
  }
  __syncthreads();
  if (threadIdx.x == 0)
    out_value_sum[o] = static_cast<std::uint32_t>(s_value_sum);
}

__global__ void seg_gather_old_ordered_kernel(
    const std::uint32_t *pool_keys, const std::uint32_t *pool_values,
    const std::uint32_t *seg_bucket_live, const std::uint32_t *dirty_seg_id,
    const std::uint32_t *old_offset, std::size_t dirty_count,
    std::uint32_t *old_seg, std::uint32_t *old_key, std::uint32_t *old_value) {
  const std::size_t m = blockIdx.x;
  if (m >= dirty_count)
    return;
  const std::uint32_t seg = dirty_seg_id[m];
  const std::size_t meta = static_cast<std::size_t>(seg) * kSegmentBuckets;
  const std::size_t slot_base = static_cast<std::size_t>(seg) * kSegmentSlots;
  const std::uint32_t base = old_offset[m];
  __shared__ std::uint32_t prefix[kSegmentBuckets];
  if (threadIdx.x == 0) {
    std::uint32_t acc = 0;
    for (int b = 0; b < kSegmentBuckets; ++b) {
      prefix[b] = acc;
      acc += seg_bucket_live[meta + b];
    }
  }
  __syncthreads();
  for (int b = threadIdx.x; b < kSegmentBuckets; b += blockDim.x) {
    const std::uint32_t live = seg_bucket_live[meta + b];
    const std::size_t src = slot_base + static_cast<std::size_t>(b) * kBucketSlots;
    const std::uint32_t dst = base + prefix[b];
    for (std::uint32_t j = 0; j < live; ++j) {
      if (old_seg)
        old_seg[dst + j] = static_cast<std::uint32_t>(m);
      old_key[dst + j] = pool_keys[src + j];
      old_value[dst + j] = pool_values[src + j];
    }
  }
}

__device__ inline std::uint32_t seg_merge_partition(
    const std::uint32_t *incoming, std::uint32_t incoming_count,
    const std::uint32_t *old_keys, std::uint32_t old_count,
    std::uint32_t diagonal) {
  std::uint32_t low =
      diagonal > old_count ? diagonal - old_count : 0u;
  std::uint32_t high =
      diagonal < incoming_count ? diagonal : incoming_count;
  while (low <= high) {
    const std::uint32_t i = low + ((high - low) >> 1);
    const std::uint32_t j = diagonal - i;
    if (i > 0 && j < old_count && incoming[i - 1] > old_keys[j]) {
      high = i - 1;
      continue;
    }
    if (j > 0 && i < incoming_count && old_keys[j - 1] >= incoming[i]) {
      low = i + 1;
      continue;
    }
    return i;
  }
  return low;
}

__global__ void seg_merge_pack_output_kernel(
    const std::uint32_t *out_dirty, const std::uint32_t *out_local,
    const std::uint32_t *out_k, const std::uint32_t *out_live_plan,
    const std::uint32_t *output_total,
    const std::uint32_t *slow_seg_id,
    const std::uint32_t *slow_old_boundary,
    const std::uint32_t *slow_old_live,
    const std::uint32_t *slow_in_begin,
    const std::uint32_t *slow_in_end,
    const std::uint32_t *old_offset,
    const std::uint32_t *reserved_seg_id,
    const std::uint32_t *incoming_keys,
    const std::uint32_t *incoming_values,
    const std::uint32_t *old_keys, const std::uint32_t *old_values,
    std::uint32_t *output_seg_id,
    std::uint32_t *out_boundary, std::uint32_t *out_live,
    std::uint32_t *out_value_sum, std::uint32_t *pool_keys,
    std::uint32_t *pool_values, std::uint8_t *pool_valid,
    std::uint32_t *seg_bucket_max, std::uint32_t *seg_bucket_live,
    std::uint32_t *seg_bucket_value_sum) {
  const std::size_t output = blockIdx.x;
  if (output >= output_total[0])
    return;
  const std::uint32_t s = out_dirty[output];
  const std::uint32_t local = out_local[output];
  const std::uint32_t k = out_k[output];
  const std::uint32_t output_live = out_live_plan[output];
  const std::uint32_t old_count = slow_old_live[s];
  const std::uint32_t in_begin = slow_in_begin[s];
  const std::uint32_t in_count = slow_in_end[s] - in_begin;
  const std::uint32_t live = old_count + in_count;
  const std::uint32_t per = ceil_div_u32(live, k);
  const std::uint32_t rank_begin = local * per;
  const std::uint32_t pack_fill =
      ceil_div_u32(output_live, kSegmentBuckets);
  const std::uint32_t seg =
      local == 0 ? slow_seg_id[s] : reserved_seg_id[output];
  const std::size_t slot_base =
      static_cast<std::size_t>(seg) * kSegmentSlots;
  const std::size_t meta_base =
      static_cast<std::size_t>(seg) * kSegmentBuckets;
  if (threadIdx.x == 0) {
    output_seg_id[output] = seg;
    out_live[output] = output_live;
    out_value_sum[output] = 0u;
  }
  for (std::size_t p = threadIdx.x; p < kSegmentSlots;
       p += blockDim.x) {
    pool_keys[slot_base + p] = kEmptyKey;
    pool_values[slot_base + p] = 0u;
    pool_valid[slot_base + p] = 0u;
  }
  __syncthreads();

  const std::uint32_t local_begin =
      static_cast<std::uint32_t>(
          (static_cast<unsigned long long>(output_live) * threadIdx.x) /
          blockDim.x);
  const std::uint32_t local_end =
      static_cast<std::uint32_t>(
          (static_cast<unsigned long long>(output_live) *
           (threadIdx.x + 1u)) /
          blockDim.x);
  const std::uint32_t diagonal_begin = rank_begin + local_begin;
  const std::uint32_t diagonal_end = rank_begin + local_end;
  const std::uint32_t *in_keys = incoming_keys + in_begin;
  const std::uint32_t *in_values = incoming_values + in_begin;
  const std::uint32_t *old_key = old_keys + old_offset[s];
  const std::uint32_t *old_value = old_values + old_offset[s];
  std::uint32_t incoming =
      seg_merge_partition(in_keys, in_count, old_key, old_count,
                          diagonal_begin);
  std::uint32_t old = diagonal_begin - incoming;
  for (std::uint32_t rank = diagonal_begin; rank < diagonal_end; ++rank) {
    std::uint32_t key;
    std::uint32_t value;
    if (incoming < in_count &&
        (old >= old_count || in_keys[incoming] <= old_key[old])) {
      key = in_keys[incoming];
      value = in_values[incoming++];
    } else {
      key = old_key[old];
      value = old_value[old++];
    }
    const std::uint32_t rank_in_output = rank - rank_begin;
    const std::uint32_t bucket = rank_in_output / pack_fill;
    const std::uint32_t slot = rank_in_output % pack_fill;
    const std::size_t position =
        slot_base + static_cast<std::size_t>(bucket) * kBucketSlots + slot;
    pool_keys[position] = key;
    pool_values[position] = value;
    pool_valid[position] = 1u;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (local + 1 == k) {
      out_boundary[output] = slow_old_boundary[s];
    } else {
      const std::uint32_t rank_last = output_live - 1;
      const std::uint32_t bucket = rank_last / pack_fill;
      const std::uint32_t slot = rank_last % pack_fill;
      out_boundary[output] =
          pool_keys[slot_base +
                    static_cast<std::size_t>(bucket) * kBucketSlots +
                    slot];
    }
  }
  __syncthreads();
  const std::uint32_t boundary = out_boundary[output];
  for (std::uint32_t bucket = threadIdx.x;
       bucket < kSegmentBuckets; bucket += blockDim.x) {
    const std::uint32_t first = bucket * pack_fill;
    const std::uint32_t count =
        first >= output_live
            ? 0u
            : ((output_live - first) < pack_fill
                   ? output_live - first
                   : pack_fill);
    const std::size_t start =
        slot_base + static_cast<std::size_t>(bucket) * kBucketSlots;
    std::uint32_t sum = 0;
    for (std::uint32_t lane = 0; lane < count; ++lane)
      sum += pool_values[start + lane];
    seg_bucket_live[meta_base + bucket] = count;
    seg_bucket_max[meta_base + bucket] =
        count > 0 ? pool_keys[start + count - 1] : boundary;
    seg_bucket_value_sum[meta_base + bucket] = sum;
    if (sum > 0)
      atomicAdd(out_value_sum + output, sum);
  }
}

__global__ void seg_gather_group_ordered_kernel(
    const std::uint32_t *pool_keys, const std::uint32_t *pool_values,
    const std::uint32_t *seg_bucket_live, const std::uint32_t *src_seg_id,
    const std::uint32_t *src_out_base, const std::uint32_t *src_group,
    std::size_t src_count, std::uint32_t *cand_seg, std::uint32_t *cand_key,
    std::uint32_t *cand_value) {
  const std::size_t m = blockIdx.x;
  if (m >= src_count)
    return;
  const std::uint32_t seg = src_seg_id[m];
  const std::uint32_t g = src_group[m];
  const std::size_t meta = static_cast<std::size_t>(seg) * kSegmentBuckets;
  const std::size_t slot_base = static_cast<std::size_t>(seg) * kSegmentSlots;
  const std::uint32_t base = src_out_base[m];
  __shared__ std::uint32_t prefix[kSegmentBuckets];
  if (threadIdx.x == 0) {
    std::uint32_t acc = 0;
    for (int b = 0; b < kSegmentBuckets; ++b) {
      prefix[b] = acc;
      acc += seg_bucket_live[meta + b];
    }
  }
  __syncthreads();
  for (int b = threadIdx.x; b < kSegmentBuckets; b += blockDim.x) {
    const std::uint32_t live = seg_bucket_live[meta + b];
    const std::size_t src =
        slot_base + static_cast<std::size_t>(b) * kBucketSlots;
    const std::uint32_t dst = base + prefix[b];
    for (std::uint32_t j = 0; j < live; ++j) {
      cand_seg[dst + j] = g;
      cand_key[dst + j] = pool_keys[src + j];
      cand_value[dst + j] = pool_values[src + j];
    }
  }
}

struct TupleOpIsInsert {
  template <class Tuple>
  __host__ __device__ bool operator()(const Tuple &t) const {
    return thrust::get<2>(t) == kInsert;
  }
};

struct NonZeroU32 {
  __host__ __device__ bool operator()(std::uint32_t x) const {
    return x != 0u;
  }
};

__global__ void mark_live_inserts_kernel(const std::uint32_t *ins_keys,
                                         std::size_t ins_count,
                                         const std::uint32_t *tomb_keys,
                                         std::size_t tomb_count,
                                         std::uint32_t *out_flag) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ins_count)
    return;
  const std::uint32_t k = ins_keys[i];
  const std::size_t p = lower_bound_u32(tomb_keys, tomb_count, k);
  out_flag[i] = (p < tomb_count && tomb_keys[p] == k) ? 0u : 1u;
}

__global__ void c0_log_append_kernel(
    const std::uint32_t *keys, const std::uint32_t *values, std::uint8_t op,
    std::uint32_t old_count, std::size_t n, std::uint32_t *log_keys,
    std::uint32_t *log_values, std::uint8_t *log_ops) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint32_t idx = old_count + static_cast<std::uint32_t>(i);
  log_keys[idx] = keys[i];
  log_values[idx] = values ? values[i] : keys[i];
  log_ops[idx] = op;
}

#ifndef GPULSMOPT_DIR_SMEM_MAX
#define GPULSMOPT_DIR_SMEM_MAX 6144
#endif
__global__ void point_lookup_walk_kernel(
    const std::uint32_t *queries, std::size_t n, std::uint32_t *out_value,
    std::uint8_t *out_found, const std::uint32_t *perm,
    const std::uint32_t *const *run_keys, const std::uint32_t *const *run_vals,
    const std::uint8_t *const *run_ops, const std::uint32_t *run_cnt,
    int num_runs, const std::uint32_t *dir_boundary,
    const std::uint32_t *dir_seg_id, std::size_t dir_count,
    const std::uint32_t *pool_keys, const std::uint32_t *pool_values,
    const std::uint8_t *pool_valid, const std::uint32_t *seg_bucket_max) {
  __shared__ std::uint32_t s_dir[GPULSMOPT_DIR_SMEM_MAX];
  const bool cache_dir = dir_count > 0 && dir_count <= GPULSMOPT_DIR_SMEM_MAX;
  if (cache_dir) {
    for (std::size_t t = threadIdx.x; t < dir_count; t += blockDim.x)
      s_dir[t] = dir_boundary[t];
    __syncthreads();
  }
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint32_t key = queries[i];
  const std::size_t o = perm ? perm[i] : i;
  for (int r = 0; r < num_runs; ++r) {
    const std::uint32_t *rk = run_keys[r];
    const std::uint32_t rc = run_cnt[r];
    const std::size_t p = lower_bound_u32(rk, rc, key);
    if (p < rc && rk[p] == key) {
      bool has_ins = false, tomb = false;
      std::uint32_t val = 0;
      for (std::size_t q = p; q < rc && rk[q] == key; ++q) {
        if (run_ops[r][q] == kInsert) {
          has_ins = true;
          val = run_vals[r][q];
        } else {
          tomb = true;
        }
      }
      const bool live = has_ins && !tomb;
      out_found[o] = live ? 1u : 0u;
      out_value[o] = live ? val : 0u;
      return;
    }
  }
  std::uint32_t value = 0;
  bool found = false;
  if (dir_count > 0) {
    const std::uint32_t *dir = cache_dir ? s_dir : dir_boundary;
    std::size_t ord = lower_bound_u32(dir, dir_count, key);
    if (ord >= dir_count)
      ord = dir_count - 1;
    found = seg_point_lookup(dir_seg_id[ord], key, pool_keys, pool_values,
                             pool_valid, seg_bucket_max, &value);
  }
  out_found[o] = found ? 1u : 0u;
  out_value[o] = found ? value : 0u;
}

__global__ void seg_flip_lookup_kernel(
    const std::uint32_t *sorted_q, const std::uint32_t *seg_id_arr,
    const std::uint32_t *q_begin_arr, const std::uint32_t *q_end_arr,
    std::size_t dirty_count, std::uint32_t *out_value, std::uint8_t *out_found,
    const std::uint32_t *pool_keys, const std::uint32_t *pool_values,
    const std::uint8_t *pool_valid, const std::uint32_t *seg_bucket_max,
    const std::uint32_t *seg_bucket_live) {
  const std::size_t m = blockIdx.x;
  if (m >= dirty_count)
    return;
  const std::uint32_t seg = seg_id_arr[m];
  const std::size_t meta = static_cast<std::size_t>(seg) * kSegmentBuckets;
  const std::size_t slot_base = static_cast<std::size_t>(seg) * kSegmentSlots;
  const std::uint32_t qb = q_begin_arr[m];
  const std::uint32_t qn = q_end_arr[m] - qb;
  for (int b = threadIdx.x; b < kSegmentBuckets; b += blockDim.x) {
    const std::size_t qbb =
        b == 0 ? 0
               : upper_bound_u32(sorted_q + qb, qn,
                                 seg_bucket_max[meta + b - 1]);
    const std::size_t qee =
        b == kSegmentBuckets - 1
            ? qn
            : upper_bound_u32(sorted_q + qb, qn, seg_bucket_max[meta + b]);
    if (qbb >= qee)
      continue;
    const std::uint32_t live = seg_bucket_live[meta + b];
    const std::size_t base =
        slot_base + static_cast<std::size_t>(b) * kBucketSlots;
    std::uint32_t ki = 0;
    for (std::size_t qi = qb + qbb; qi < qb + qee; ++qi) {
      const std::uint32_t q = sorted_q[qi];
      while (ki < live && pool_keys[base + ki] < q)
        ++ki;
      if (ki < live && pool_keys[base + ki] == q && pool_valid[base + ki]) {
        out_found[qi] = 1u;
        out_value[qi] = pool_values[base + ki];
      } else {
        out_found[qi] = 0u;
        out_value[qi] = 0u;
      }
    }
  }
}

__global__ void overlay_walk_override_kernel(
    const std::uint32_t *sorted_q, std::size_t n, std::uint32_t *out_value,
    std::uint8_t *out_found, const std::uint32_t *const *run_keys,
    const std::uint32_t *const *run_vals, const std::uint8_t *const *run_ops,
    const std::uint32_t *run_cnt, int num_runs) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint32_t q = sorted_q[i];
  for (int r = 0; r < num_runs; ++r) {
    const std::uint32_t *rk = run_keys[r];
    const std::uint32_t rc = run_cnt[r];
    const std::size_t p = lower_bound_u32(rk, rc, q);
    if (p < rc && rk[p] == q) {
      bool has_ins = false, tomb = false;
      std::uint32_t val = 0;
      for (std::size_t s = p; s < rc && rk[s] == q; ++s) {
        if (run_ops[r][s] == kInsert) {
          has_ins = true;
          val = run_vals[r][s];
        } else {
          tomb = true;
        }
      }
      const bool live = has_ins && !tomb;
      out_found[i] = live ? 1u : 0u;
      out_value[i] = live ? val : 0u;
      return;
    }
  }
}

__global__ void successor_index_kernel(
    const std::uint32_t *queries, std::size_t n, std::uint32_t *out_keys,
    const std::uint32_t *dir_boundary, const std::uint32_t *dir_seg_id,
    const std::uint32_t *dir_prefix, std::size_t dir_count,
    const std::uint32_t *pool_keys,
    const std::uint32_t *seg_bucket_max, const std::uint32_t *seg_bucket_live,
    const std::uint32_t *killed_keys, std::size_t killed_count,
    const std::uint32_t *live_ins_keys, std::size_t live_ins_count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint32_t q = queries[i];
  std::uint32_t best = kEmptyKey;
  if (live_ins_count > 0) {
    const std::size_t p = lower_bound_u32(live_ins_keys, live_ins_count, q);
    if (p < live_ins_count)
      best = live_ins_keys[p];
  }
  if (dir_count > 0 && q < best) {
    std::size_t start_ord = lower_bound_u32(dir_boundary, dir_count, q);
    if (start_ord >= dir_count)
      start_ord = dir_count - 1;
    std::size_t killed =
        killed_count ? lower_bound_u32(killed_keys, killed_count, q) : 0;
    for (std::size_t ord = start_ord; ord < dir_count; ++ord) {
      if (dir_prefix[ord + 1] == dir_prefix[ord])
        continue;
      const std::uint32_t seg = dir_seg_id[ord];
      const std::size_t meta =
          static_cast<std::size_t>(seg) * kSegmentBuckets;
      const std::size_t slots =
          static_cast<std::size_t>(seg) * kSegmentSlots;
      const std::size_t start_bucket =
          ord == start_ord
              ? lower_bound_u32(seg_bucket_max + meta, kSegmentBuckets, q)
              : 0;
      std::size_t bucket = start_bucket;
      for (; bucket < kSegmentBuckets; ++bucket) {
        const std::uint32_t live = seg_bucket_live[meta + bucket];
        if (live == 0)
          continue;
        const std::size_t base = slots + bucket * kBucketSlots;
        const std::size_t first =
            ord == start_ord && bucket == start_bucket
                ? lower_bound_u32(pool_keys + base, live, q)
                : 0;
        for (std::size_t p = first; p < live; ++p) {
          const std::uint32_t candidate = pool_keys[base + p];
          if (candidate >= best) {
            out_keys[i] = best == kEmptyKey ? 0u : best;
            return;
          }
          while (killed < killed_count && killed_keys[killed] < candidate)
            ++killed;
          if (killed == killed_count || killed_keys[killed] != candidate) {
            out_keys[i] = candidate;
            return;
          }
        }
      }
    }
  }
  out_keys[i] = (best == kEmptyKey) ? 0u : best;
}

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

}

class GPULSMOpt {
public:
  using DeviceKeyBatch = gpulsmopt_detail::DeviceKeyBatch;

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
    CUDA_CHECK(cudaStreamSynchronize(0));
  }

  void clear(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    reset_directory_to_root(stream);
    recycle_active_runs();
    clear_c0_log(stream);
    overlay_dirty_ = true;
    read_view_dirty_ = true;
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void insert(const DeviceKeyValueBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    {
      std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
#ifdef GPULSMOPT_PROFILE_INSERT
      reset_insert_prof_();
      const auto prof_t0 = std::chrono::high_resolution_clock::now();
#endif
      insert_records(batch.keys, batch.values,
                     static_cast<std::uint8_t>(gpulsmopt_detail::kInsert),
                     batch.count, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      maybe_flush_and_merge(stream);
#ifdef GPULSMOPT_PROFILE_INSERT
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const auto prof_t1 = std::chrono::high_resolution_clock::now();
      const double total =
          std::chrono::duration<double, std::milli>(prof_t1 - prof_t0).count();
      const double measured = prof_append_ms_ + prof_flushsort_ms_ +
                              prof_runmerge_ms_ + prof_resolve_ms_ +
                              prof_delete_ms_ + prof_sheetmerge_ms_;
      const double other = total - measured;
      auto pct = [total](double x) {
        return total > 0.0 ? 100.0 * x / total : 0.0;
      };
      printf("[prof] insert %zu keys: total=%.3f ms\n", batch.count, total);
      printf("[prof]   append      = %.3f ms (%5.1f%%)\n", prof_append_ms_,
             pct(prof_append_ms_));
      printf("[prof]   flush_sort  = %.3f ms (%5.1f%%)\n", prof_flushsort_ms_,
             pct(prof_flushsort_ms_));
      printf("[prof]   run_merge   = %.3f ms (%5.1f%%)\n", prof_runmerge_ms_,
             pct(prof_runmerge_ms_));
      printf("[prof]   resolve     = %.3f ms (%5.1f%%)\n", prof_resolve_ms_,
             pct(prof_resolve_ms_));
      printf("[prof]   tomb_delete = %.3f ms (%5.1f%%)\n", prof_delete_ms_,
             pct(prof_delete_ms_));
      printf("[prof]   sheet_merge = %.3f ms (%5.1f%%)\n", prof_sheetmerge_ms_,
             pct(prof_sheetmerge_ms_));
      printf("[prof]   other/host  = %.3f ms (%5.1f%%)\n", other, pct(other));
#endif
    }
  }

  void erase(const DeviceKeyBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    {
      std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);

      insert_records(batch.keys, batch.keys,
                     static_cast<std::uint8_t>(gpulsmopt_detail::kTombstone),
                     batch.count, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      maybe_flush_and_merge(stream);
    }
  }

  void lookup(const DeviceLookupBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    lookup_layered(batch, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void count(const DeviceRangeBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    const bool no_overlay = c0_log_count_ == 0 && runs_.empty();
    const OverlayReadIndex *ix = no_overlay ? nullptr : &overlay_index(stream);
    const std::uint32_t *ins_keys = no_overlay ? nullptr : raw_or_null(ix->gk);
    const std::uint32_t *tomb_keys =
        no_overlay ? nullptr : raw_or_null(ix->gk) + ix->ins;
    const std::uint32_t *tomb_cnt =
        no_overlay ? nullptr : raw_or_null(ix->tomb_cnt_prefix);
    const int block = 128;
    const int grid = static_cast<int>((batch.count + block - 1) / block);
    gpulsmopt_detail::count_overlay_kernel<<<grid, block, 0, stream>>>(
        batch.lo, batch.hi, batch.out_counts, batch.count,
        raw_or_null(d_dir_boundary_), raw_or_null(d_dir_seg_id_),
        raw_or_null(d_dir_prefix_), h_dir_seg_id_.size(),
        raw_or_null(pool_keys_), raw_or_null(pool_valid_),
        raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_),
        ins_keys, no_overlay ? 0 : ix->ins, tomb_keys, tomb_cnt,
        no_overlay ? 0 : ix->u - ix->ins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void successor(const DeviceSuccessorBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    const bool no_overlay = c0_log_count_ == 0 && runs_.empty();
    const OverlayReadIndex *ix = no_overlay ? nullptr : &overlay_index(stream);
    const std::uint32_t *killed =
        no_overlay ? nullptr : raw_or_null(ix->killed_keys);
    const std::size_t killed_count = no_overlay ? 0 : ix->killed_count;
    const std::uint32_t *live_ins =
        no_overlay ? nullptr : raw_or_null(ix->live_ins_keys);
    const std::size_t live_ins_count = no_overlay ? 0 : ix->live_ins_count;
    const int block = 128;
    const int grid = static_cast<int>((batch.count + block - 1) / block);
    gpulsmopt_detail::successor_index_kernel<<<grid, block, 0, stream>>>(
        batch.queries, batch.count, batch.out_keys,
        raw_or_null(d_dir_boundary_), raw_or_null(d_dir_seg_id_),
        raw_or_null(d_dir_prefix_), h_dir_seg_id_.size(),
        raw_or_null(pool_keys_), raw_or_null(seg_bucket_max_),
        raw_or_null(seg_bucket_live_), killed, killed_count, live_ins,
        live_ins_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void range(const DeviceRangeOutputBatch &batch, cudaStream_t stream) {
    if (batch.query_count == 0)
      return;
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    const bool no_overlay = c0_log_count_ == 0 && runs_.empty();
    const OverlayReadIndex *ix = no_overlay ? nullptr : &overlay_index(stream);
    const std::uint32_t *ins_keys = no_overlay ? nullptr : raw_or_null(ix->gk);
    const std::uint32_t *ins_prefix =
        no_overlay ? nullptr : raw_or_null(ix->ins_prefix);
    const std::uint32_t *tomb_keys =
        no_overlay ? nullptr : raw_or_null(ix->gk) + ix->ins;
    const std::uint32_t *tomb_val =
        no_overlay ? nullptr : raw_or_null(ix->tomb_val_prefix);
    const std::uint32_t *tomb_cnt =
        no_overlay ? nullptr : raw_or_null(ix->tomb_cnt_prefix);
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
        ins_keys, ins_prefix, no_overlay ? 0 : ix->ins, tomb_keys, tomb_val,
        tomb_cnt, no_overlay ? 0 : ix->u - ix->ins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void drain_to_sheet(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    merge_down(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void bulk_build(const std::uint32_t *keys, const std::uint32_t *values,
                  std::size_t n, cudaStream_t stream) {
    if (n == 0)
      return;
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    auto policy = thrust::cuda::par.on(stream);
    scratch_incoming_keys_.resize(n);
    scratch_incoming_values_.resize(n);
    thrust::copy(policy, thrust::device_pointer_cast(keys),
                 thrust::device_pointer_cast(keys) + n,
                 scratch_incoming_keys_.begin());
    thrust::copy(policy, thrust::device_pointer_cast(values),
                 thrust::device_pointer_cast(values) + n,
                 scratch_incoming_values_.begin());
    thrust::sort_by_key(policy, scratch_incoming_keys_.begin(),
                        scratch_incoming_keys_.end(),
                        scratch_incoming_values_.begin());
    merge_incoming_into_sheet(n, stream);
    overlay_dirty_ = true;
    read_view_dirty_ = true;
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  std::size_t sheet_live_count() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    return sheet_live_count_;
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
        pool_keys_, pool_values_, pool_valid_, seg_bucket_max_,
        seg_bucket_live_, seg_bucket_value_sum_, d_dir_seg_id_,
        d_dir_boundary_, d_dir_prefix_, d_dir_value_sum_, d_dir_value_prefix_,
        scratch_incoming_keys_, scratch_incoming_values_, scratch_delete_keys_,
        seg_pull_counter_, seg_dirty_bucket_dirty_,
        seg_cand_seg_, seg_cand_key_, seg_cand_value_,
        c0_log_keys_, c0_log_values_, c0_log_ops_, seg_new_value_sum_,
        seg_d_out_value_sum_);
    for (const auto &g : runs_)
      total += device_bytes_all(g.keys, g.values, g.ops);
    for (const auto &g : run_buffer_pool_)
      total += device_bytes_all(g.keys, g.values, g.ops);
    return total;
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

  void initialize_segmented_storage(cudaStream_t stream) {
    pool_capacity_ = 0;
    free_ids_.clear();
    h_dir_seg_id_.clear();
    h_dir_boundary_.clear();
    h_dir_live_.clear();
    h_dir_value_sum_.clear();
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
    h_dir_value_sum_ = {0u};
    sheet_live_count_ = 0;
    sheet_fragmented_ = false;
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
    upload_directory_metadata(stream);
  }

  void upload_directory_metadata(cudaStream_t stream) {
    const std::size_t d = h_dir_seg_id_.size();
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
    if (!prefix.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(
          raw_or_null(d_dir_prefix_), prefix.data(),
          prefix.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
          stream));
    }
    if (!h_dir_value_sum_.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(
          raw_or_null(d_dir_value_sum_), h_dir_value_sum_.data(),
          h_dir_value_sum_.size() * sizeof(std::uint32_t),
          cudaMemcpyHostToDevice, stream));
    }
    auto policy = thrust::cuda::par.on(stream);
    thrust::fill(policy, d_dir_value_prefix_.begin(), d_dir_value_prefix_.end(),
                 0u);
    if (d == 0)
      return;
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
  }

  void prepare_dirty_plan(std::size_t dirty_count, cudaStream_t stream) {
    auto policy = thrust::cuda::par.on(stream);
    seg_d_dirty_seg_id_.resize(dirty_count);
    seg_d_dirty_old_boundary_.resize(dirty_count);
    seg_d_dirty_old_live_.resize(dirty_count);
    seg_d_dirty_old_value_sum_.resize(dirty_count);
    seg_d_dirty_in_begin_.resize(dirty_count);
    seg_d_dirty_in_end_.resize(dirty_count);
    thrust::exclusive_scan(policy, seg_dirty_count_.begin(),
                           seg_dirty_count_.begin() + dirty_count,
                           seg_d_dirty_in_begin_.begin());
    const int block = 256;
    const int grid = static_cast<int>((dirty_count + block - 1) / block);
    gpulsmopt_detail::seg_make_dirty_plan_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(seg_dirty_ord_), raw_or_null(seg_dirty_count_),
        dirty_count, raw_or_null(d_dir_seg_id_), raw_or_null(d_dir_boundary_),
        raw_or_null(d_dir_prefix_), raw_or_null(d_dir_value_sum_),
        raw_or_null(seg_d_dirty_seg_id_),
        raw_or_null(seg_d_dirty_old_boundary_),
        raw_or_null(seg_d_dirty_old_live_),
        raw_or_null(seg_d_dirty_old_value_sum_),
        raw_or_null(seg_d_dirty_in_begin_),
        raw_or_null(seg_d_dirty_in_end_));
    CUDA_CHECK(cudaGetLastError());
  }

  void build_key_routed_dirty_plan(std::size_t incoming_count,
                                   std::size_t dir_count,
                                   cudaStream_t stream, std::size_t &m,
                                   std::size_t &mb) {
    auto policy = thrust::cuda::par.on(stream);
    seg_inc_ordinal_.resize(incoming_count);
    {
      const int block = 256;
      const int grid = static_cast<int>((incoming_count + block - 1) / block);
      gpulsmopt_detail::seg_route_keys_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(scratch_incoming_keys_), incoming_count,
          raw_or_null(d_dir_boundary_), dir_count, raw_or_null(seg_inc_ordinal_));
      CUDA_CHECK(cudaGetLastError());
    }
    seg_dirty_ord_.resize(incoming_count);
    seg_dirty_count_.resize(incoming_count);
    auto rle_end = thrust::reduce_by_key(
        policy, seg_inc_ordinal_.begin(), seg_inc_ordinal_.begin() + incoming_count,
        thrust::make_constant_iterator<std::uint32_t>(1u),
        seg_dirty_ord_.begin(), seg_dirty_count_.begin());
    m = static_cast<std::size_t>(rle_end.first - seg_dirty_ord_.begin());
    if (m == 0) {
      mb = 0;
      return;
    }
    prepare_dirty_plan(m, stream);
    seg_inc_bucket_.resize(incoming_count);
    {
      const int block = 256;
      const int grid = static_cast<int>((incoming_count + block - 1) / block);
      gpulsmopt_detail::seg_route_buckets_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(scratch_incoming_keys_), raw_or_null(seg_inc_ordinal_),
          incoming_count, raw_or_null(d_dir_seg_id_),
          raw_or_null(seg_bucket_max_), raw_or_null(seg_inc_bucket_));
      CUDA_CHECK(cudaGetLastError());
    }
    seg_dirty_bucket_.resize(incoming_count);
    seg_dirty_bucket_count_.resize(incoming_count);
    auto bucket_rle_end = thrust::reduce_by_key(
        policy, seg_inc_bucket_.begin(), seg_inc_bucket_.begin() + incoming_count,
        thrust::make_constant_iterator<std::uint32_t>(1u),
        seg_dirty_bucket_.begin(), seg_dirty_bucket_count_.begin());
    mb = static_cast<std::size_t>(
        bucket_rle_end.first - seg_dirty_bucket_.begin());
    seg_dirty_bucket_begin_.resize(mb);
    thrust::exclusive_scan(policy, seg_dirty_bucket_count_.begin(),
                           seg_dirty_bucket_count_.begin() + mb,
                           seg_dirty_bucket_begin_.begin());
  }

  bool build_pull_dirty_plan(std::size_t incoming_count, std::size_t dir_count,
                             cudaStream_t stream, std::size_t &m,
                             std::size_t &mb) {
    seg_pull_counter_.resize(2);
    CUDA_CHECK(cudaMemsetAsync(raw_or_null(seg_pull_counter_), 0,
                               2 * sizeof(std::uint32_t), stream));
    seg_dirty_ord_.resize(dir_count);
    seg_dirty_count_.resize(dir_count);
    seg_d_dirty_seg_id_.resize(dir_count);
    seg_d_dirty_old_boundary_.resize(dir_count);
    seg_d_dirty_old_live_.resize(dir_count);
    seg_d_dirty_old_value_sum_.resize(dir_count);
    seg_d_dirty_in_begin_.resize(dir_count);
    seg_d_dirty_in_end_.resize(dir_count);
    seg_slow_.resize(dir_count);
    seg_new_live_.resize(dir_count);
    seg_new_value_sum_.resize(dir_count);
    {
      const int block = 256;
      const int grid = static_cast<int>((dir_count + block - 1) / block);
      gpulsmopt_detail::seg_pull_dirty_segments_kernel<<<grid, block, 0,
                                                         stream>>>(
          raw_or_null(scratch_incoming_keys_), incoming_count,
          raw_or_null(d_dir_boundary_), dir_count, raw_or_null(d_dir_seg_id_),
          raw_or_null(d_dir_prefix_), raw_or_null(d_dir_value_sum_),
          raw_or_null(seg_dirty_ord_), raw_or_null(seg_dirty_count_),
          raw_or_null(seg_d_dirty_seg_id_),
          raw_or_null(seg_d_dirty_old_boundary_),
          raw_or_null(seg_d_dirty_old_live_),
          raw_or_null(seg_d_dirty_old_value_sum_),
          raw_or_null(seg_d_dirty_in_begin_),
          raw_or_null(seg_d_dirty_in_end_), raw_or_null(seg_slow_),
          raw_or_null(seg_new_live_), raw_or_null(seg_new_value_sum_),
          raw_or_null(seg_pull_counter_));
      CUDA_CHECK(cudaGetLastError());
    }
    std::uint32_t dirty_total = 0;
    CUDA_CHECK(cudaMemcpyAsync(&dirty_total, raw_or_null(seg_pull_counter_),
                               sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    m = dirty_total;
    if (m == 0) {
      mb = 0;
      return true;
    }

    const std::size_t bucket_slots =
        m * static_cast<std::size_t>(gpulsmopt_detail::kSegmentBuckets);
    if (bucket_slots > incoming_count * 2)
      return false;
    seg_dirty_bucket_.resize(bucket_slots);
    seg_dirty_bucket_count_.resize(bucket_slots);
    seg_dirty_bucket_begin_.resize(bucket_slots);
    seg_dirty_bucket_dirty_.resize(bucket_slots);
    CUDA_CHECK(cudaMemsetAsync(raw_or_null(seg_pull_counter_) + 1, 0,
                               sizeof(std::uint32_t), stream));
    {
      const int block = 256;
      const int grid = static_cast<int>((bucket_slots + block - 1) / block);
      gpulsmopt_detail::seg_pull_dirty_buckets_kernel<<<grid, block, 0,
                                                        stream>>>(
          raw_or_null(scratch_incoming_keys_), raw_or_null(seg_dirty_ord_),
          raw_or_null(seg_dirty_count_), raw_or_null(seg_d_dirty_seg_id_),
          raw_or_null(seg_d_dirty_in_begin_), m,
          raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_),
          raw_or_null(seg_dirty_bucket_),
          raw_or_null(seg_dirty_bucket_count_),
          raw_or_null(seg_dirty_bucket_begin_),
          raw_or_null(seg_dirty_bucket_dirty_), raw_or_null(seg_slow_),
          raw_or_null(seg_pull_counter_));
      CUDA_CHECK(cudaGetLastError());
    }
    std::uint32_t bucket_total = 0;
    CUDA_CHECK(cudaMemcpyAsync(&bucket_total, raw_or_null(seg_pull_counter_) + 1,
                               sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    mb = bucket_total;
    return true;
  }

  void merge_incoming_into_sheet(std::size_t incoming_count,
                                 cudaStream_t stream) {
    if (incoming_count == 0)
      return;
    auto policy = thrust::cuda::par.on(stream);

    const std::size_t dir_count = h_dir_seg_id_.size();
    if (dir_count == 0)
      return;
    std::size_t m = 0, mb = 0;
    bool fused_plan = false;
    if (incoming_count < dir_count * 16) {
      build_key_routed_dirty_plan(incoming_count, dir_count, stream, m, mb);
    } else {
      fused_plan = build_pull_dirty_plan(incoming_count, dir_count, stream, m, mb);
      if (!fused_plan)
        build_key_routed_dirty_plan(incoming_count, dir_count, stream, m, mb);
    }
    if (m == 0)
      return;
    if (!fused_plan) {
      seg_slow_.resize(m);
      seg_new_live_.resize(m);
      seg_new_value_sum_.resize(m);
      {
        const int block = 256;
        const int grid = static_cast<int>((m + block - 1) / block);
        gpulsmopt_detail::seg_init_dirty_segments_kernel<<<grid, block, 0,
                                                          stream>>>(
            raw_or_null(seg_d_dirty_old_live_),
            raw_or_null(seg_d_dirty_old_value_sum_),
            raw_or_null(seg_d_dirty_in_begin_),
            raw_or_null(seg_d_dirty_in_end_), m, raw_or_null(seg_slow_),
            raw_or_null(seg_new_live_), raw_or_null(seg_new_value_sum_));
        CUDA_CHECK(cudaGetLastError());
      }
      {
        const int block = 256;
        const int grid = static_cast<int>((mb + block - 1) / block);
        gpulsmopt_detail::seg_classify_dirty_buckets_kernel<<<
            grid, block, 0, stream>>>(
            raw_or_null(seg_dirty_bucket_),
            raw_or_null(seg_dirty_bucket_count_), mb,
            raw_or_null(seg_dirty_ord_), m, raw_or_null(d_dir_seg_id_),
            raw_or_null(seg_bucket_live_), raw_or_null(seg_slow_));
        CUDA_CHECK(cudaGetLastError());
      }
    }

    seg_slow_index_.resize(m);
    seg_pull_counter_.resize(2);
    CUDA_CHECK(cudaMemsetAsync(raw_or_null(seg_pull_counter_), 0,
                               2 * sizeof(std::uint32_t), stream));
    {
      const int block = 256;
      const int grid = static_cast<int>((m + block - 1) / block);
      gpulsmopt_detail::seg_emit_fast_slow_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(seg_slow_), m, raw_or_null(seg_slow_index_),
          raw_or_null(seg_pull_counter_));
      CUDA_CHECK(cudaGetLastError());
    }
    std::uint32_t split_count[2] = {0u, 0u};
    CUDA_CHECK(cudaMemcpyAsync(split_count, raw_or_null(seg_pull_counter_),
                               2 * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::size_t fast_count = split_count[0];
    const std::size_t ms = split_count[1];

    if (fast_count > 0 && mb > 0) {
      const int block = 256;
      const int grid = static_cast<int>((mb + block - 1) / block);
      gpulsmopt_detail::seg_apply_dirty_buckets_kernel<<<
          grid, block, 0, stream>>>(
          raw_or_null(seg_dirty_bucket_), raw_or_null(seg_dirty_bucket_count_),
          raw_or_null(seg_dirty_bucket_begin_),
          fused_plan ? raw_or_null(seg_dirty_bucket_dirty_) : nullptr, mb,
          raw_or_null(seg_dirty_ord_), m, raw_or_null(seg_slow_),
          raw_or_null(d_dir_seg_id_),
          raw_or_null(scratch_incoming_keys_),
          raw_or_null(scratch_incoming_values_), raw_or_null(pool_keys_),
          raw_or_null(pool_values_), raw_or_null(pool_valid_),
          raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_),
          raw_or_null(seg_bucket_value_sum_),
          raw_or_null(seg_new_value_sum_));
      CUDA_CHECK(cudaGetLastError());
    }

#ifdef GPULSMOPT_PROFILE_INSERT
    printf("[prof]     sheet_merge segs: dirty=%zu fast=%zu slow=%zu "
           "(incoming=%zu)\n",
           m, fast_count, ms, incoming_count);
#endif
    std::vector<std::uint32_t> dirty_k, dirty_output_base, output_seg_id,
        out_boundary, out_live, out_value_sum, reserved_seg_id;
    std::uint32_t output_total = 0;
    if (ms > 0) {
      seg_slow_seg_id_.resize(ms);
      seg_slow_old_boundary_.resize(ms);
      seg_slow_old_live_.resize(ms);
      seg_slow_in_begin_.resize(ms);
      seg_slow_in_end_.resize(ms);
      const int block = 256;
      const int grid = static_cast<int>((ms + block - 1) / block);
      gpulsmopt_detail::seg_gather_dirty_plan_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(seg_slow_index_), ms,
          raw_or_null(seg_d_dirty_seg_id_),
          raw_or_null(seg_d_dirty_old_boundary_),
          raw_or_null(seg_d_dirty_old_live_),
          raw_or_null(seg_d_dirty_in_begin_),
          raw_or_null(seg_d_dirty_in_end_), raw_or_null(seg_slow_seg_id_),
          raw_or_null(seg_slow_old_boundary_),
          raw_or_null(seg_slow_old_live_),
          raw_or_null(seg_slow_in_begin_),
          raw_or_null(seg_slow_in_end_));
      CUDA_CHECK(cudaGetLastError());

      seg_slow_inc_count_.resize(ms);
      seg_slow_cand_count_.resize(ms);
      gpulsmopt_detail::seg_make_slow_counts_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(seg_slow_old_live_), raw_or_null(seg_slow_in_begin_),
          raw_or_null(seg_slow_in_end_), ms,
          raw_or_null(seg_slow_inc_count_),
          raw_or_null(seg_slow_cand_count_));
      CUDA_CHECK(cudaGetLastError());

      seg_d_old_offset_.resize(ms);
      thrust::exclusive_scan(policy, seg_slow_old_live_.begin(),
                             seg_slow_old_live_.begin() + ms,
                             seg_d_old_offset_.begin());
      seg_dirty_live_.resize(ms);
      thrust::copy(policy, seg_slow_cand_count_.begin(),
                   seg_slow_cand_count_.begin() + ms,
                   seg_dirty_live_.begin());
      seg_d_dirty_k_.resize(ms);
      gpulsmopt_detail::seg_make_dirty_k_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(seg_dirty_live_), ms, target_segment_live_,
          raw_or_null(seg_d_dirty_k_));
      CUDA_CHECK(cudaGetLastError());
      seg_d_dirty_output_base_.resize(ms);
      thrust::exclusive_scan(policy, seg_d_dirty_k_.begin(),
                             seg_d_dirty_k_.begin() + ms,
                             seg_d_dirty_output_base_.begin());
      seg_plan_total_.resize(1);
      gpulsmopt_detail::seg_output_total_kernel<<<1, 1, 0, stream>>>(
          raw_or_null(seg_d_dirty_output_base_),
          raw_or_null(seg_d_dirty_k_), ms, raw_or_null(seg_plan_total_));
      CUDA_CHECK(cudaGetLastError());
      const std::size_t max_output =
          3 * ms + gpulsmopt_detail::ceil_div_u32(
                       static_cast<std::uint32_t>(incoming_count),
                       target_segment_live_);
      reserved_seg_id.resize(max_output);
      for (std::size_t o = 0; o < max_output; ++o)
        reserved_seg_id[o] = alloc_segment();
      upload_vec(seg_d_reserved_seg_id_, reserved_seg_id, stream);
      seg_d_output_seg_id_.resize(max_output);
      seg_d_out_dirty_.resize(max_output);
      seg_d_out_local_.resize(max_output);
      seg_d_out_k_.resize(max_output);
      seg_d_out_live_.resize(max_output);
      seg_d_out_boundary_.resize(max_output);
      seg_d_out_value_sum_.resize(max_output);
      gpulsmopt_detail::seg_output_plan_kernel<<<
          static_cast<unsigned>(ms), 256, 0, stream>>>(
          raw_or_null(seg_dirty_live_), raw_or_null(seg_d_dirty_k_),
          raw_or_null(seg_d_dirty_output_base_), ms,
          raw_or_null(seg_d_out_dirty_), raw_or_null(seg_d_out_local_),
          raw_or_null(seg_d_out_k_), raw_or_null(seg_d_out_live_));
      CUDA_CHECK(cudaGetLastError());

      const std::size_t old_capacity =
          ms * static_cast<std::size_t>(gpulsmopt_detail::kSegmentSlots);
      seg_old_key_.resize(old_capacity);
      seg_old_value_.resize(old_capacity);
      gpulsmopt_detail::seg_gather_old_ordered_kernel<<<
          static_cast<unsigned>(ms), 256, 0, stream>>>(
          raw_or_null(pool_keys_), raw_or_null(pool_values_),
          raw_or_null(seg_bucket_live_), raw_or_null(seg_slow_seg_id_),
          raw_or_null(seg_d_old_offset_), ms, nullptr,
          raw_or_null(seg_old_key_), raw_or_null(seg_old_value_));
      CUDA_CHECK(cudaGetLastError());
      gpulsmopt_detail::seg_merge_pack_output_kernel<<<
          static_cast<unsigned>(max_output), 256, 0, stream>>>(
          raw_or_null(seg_d_out_dirty_), raw_or_null(seg_d_out_local_),
          raw_or_null(seg_d_out_k_), raw_or_null(seg_d_out_live_),
          raw_or_null(seg_plan_total_), raw_or_null(seg_slow_seg_id_),
          raw_or_null(seg_slow_old_boundary_),
          raw_or_null(seg_slow_old_live_),
          raw_or_null(seg_slow_in_begin_), raw_or_null(seg_slow_in_end_),
          raw_or_null(seg_d_old_offset_),
          raw_or_null(seg_d_reserved_seg_id_),
          raw_or_null(scratch_incoming_keys_),
          raw_or_null(scratch_incoming_values_), raw_or_null(seg_old_key_),
          raw_or_null(seg_old_value_),
          raw_or_null(seg_d_output_seg_id_),
          raw_or_null(seg_d_out_boundary_), raw_or_null(seg_d_out_live_),
          raw_or_null(seg_d_out_value_sum_), raw_or_null(pool_keys_),
          raw_or_null(pool_values_), raw_or_null(pool_valid_),
          raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_),
          raw_or_null(seg_bucket_value_sum_));
      CUDA_CHECK(cudaGetLastError());

      dirty_k.resize(ms);
      dirty_output_base.resize(ms);
      output_seg_id.resize(max_output);
      out_boundary.resize(max_output);
      out_live.resize(max_output);
      out_value_sum.resize(max_output);
      CUDA_CHECK(cudaMemcpyAsync(&output_total, raw_or_null(seg_plan_total_),
                                 sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(dirty_k.data(), raw_or_null(seg_d_dirty_k_),
                                 ms * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          dirty_output_base.data(), raw_or_null(seg_d_dirty_output_base_),
          ms * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          output_seg_id.data(), raw_or_null(seg_d_output_seg_id_),
          max_output * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
          stream));
      CUDA_CHECK(cudaMemcpyAsync(out_boundary.data(),
                                 raw_or_null(seg_d_out_boundary_),
                                 max_output * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(out_live.data(), raw_or_null(seg_d_out_live_),
                                 max_output * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          out_value_sum.data(), raw_or_null(seg_d_out_value_sum_),
          max_output * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
          stream));
    }

    std::vector<std::uint32_t> dirty_ord(m), seg_slow(m), seg_new_live(m),
        seg_new_value_sum(m);
    CUDA_CHECK(cudaMemcpyAsync(dirty_ord.data(), raw_or_null(seg_dirty_ord_),
                               m * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(seg_slow.data(), raw_or_null(seg_slow_),
                               m * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(seg_new_live.data(), raw_or_null(seg_new_live_),
                               m * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        seg_new_value_sum.data(), raw_or_null(seg_new_value_sum_),
        m * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
    std::vector<std::uint32_t> slow_index_device(ms);
    if (ms > 0) {
      CUDA_CHECK(cudaMemcpyAsync(slow_index_device.data(),
                                 raw_or_null(seg_slow_index_),
                                 ms * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (ms > 0) {
      if (output_total > output_seg_id.size())
        throw std::runtime_error("segment output plan overflow");
      for (std::size_t o = 0; o < reserved_seg_id.size(); ++o) {
        const bool used =
            o < output_total && output_seg_id[o] == reserved_seg_id[o];
        if (!used)
          free_segment(reserved_seg_id[o]);
      }
      output_seg_id.resize(output_total);
      out_boundary.resize(output_total);
      out_live.resize(output_total);
      out_value_sum.resize(output_total);
    }

    std::vector<int> is_fast(m), slow_index(m, -1);
    for (std::size_t j = 0; j < m; ++j)
      is_fast[j] = seg_slow[j] == 0u ? 1 : 0;
    for (std::size_t s = 0; s < ms; ++s)
      slow_index[slow_index_device[s]] = static_cast<int>(s);

    std::vector<char> is_dirty(h_dir_seg_id_.size(), 0);
    std::vector<std::size_t> ord_to_j(h_dir_seg_id_.size(), 0);
    for (std::size_t j = 0; j < m; ++j) {
      is_dirty[dirty_ord[j]] = 1;
      ord_to_j[dirty_ord[j]] = j;
    }
    std::vector<std::uint32_t> new_seg_id, new_boundary, new_live,
        new_value_sum;
    new_seg_id.reserve(h_dir_seg_id_.size() + output_seg_id.size());
    new_boundary.reserve(new_seg_id.capacity());
    new_live.reserve(new_seg_id.capacity());
    new_value_sum.reserve(new_seg_id.capacity());
    for (std::size_t ord = 0; ord < h_dir_seg_id_.size(); ++ord) {
      if (!is_dirty[ord]) {
        new_seg_id.push_back(h_dir_seg_id_[ord]);
        new_boundary.push_back(h_dir_boundary_[ord]);
        new_live.push_back(h_dir_live_[ord]);
        new_value_sum.push_back(h_dir_value_sum_[ord]);
        continue;
      }
      const std::size_t j = ord_to_j[ord];
      if (is_fast[j]) {
        new_seg_id.push_back(h_dir_seg_id_[ord]);
        new_boundary.push_back(h_dir_boundary_[ord]);
        new_live.push_back(seg_new_live[j]);
        new_value_sum.push_back(seg_new_value_sum[j]);
        continue;
      }
      const std::size_t s = static_cast<std::size_t>(slow_index[j]);
      for (std::uint32_t local = 0; local < dirty_k[s]; ++local) {
        const std::size_t o = dirty_output_base[s] + local;
        new_seg_id.push_back(output_seg_id[o]);
        new_boundary.push_back(out_boundary[o]);
        new_live.push_back(out_live[o]);
        new_value_sum.push_back(out_value_sum[o]);
      }
    }
    h_dir_seg_id_ = std::move(new_seg_id);
    h_dir_boundary_ = std::move(new_boundary);
    h_dir_live_ = std::move(new_live);
    h_dir_value_sum_ = std::move(new_value_sum);
    recompute_sheet_live_count();
    upload_directory(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (sheet_fragmented_) {
      merge_underfull_segments(stream);
      sheet_fragmented_ = false;
    }
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
          raw_or_null(seg_bucket_live_), raw_or_null(seg_bucket_value_sum_),
          raw_or_null(seg_d_out_value_sum_));
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
      std::size_t begin;
      std::size_t end;
      std::uint32_t live;
      std::uint32_t value_sum;
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
      std::uint32_t value_sum = h_dir_value_sum_[i];
      while (j + 1 < n && sum + h_dir_live_[j + 1] <= target_segment_live_) {
        sum += h_dir_live_[j + 1];
        value_sum += h_dir_value_sum_[j + 1];
        ++j;
      }
      if (j > i) {
        groups.push_back({i, j + 1, sum, value_sum});
        i = j + 1;
      } else {
        ++i;
      }
    }
    if (groups.empty())
      return;

    auto policy = thrust::cuda::par.on(stream);
    const std::size_t group_count = groups.size();

    std::vector<std::uint32_t> dirty_live(group_count),
        dirty_live_offset(group_count), dirty_k(group_count, 1u),
        dirty_output_base(group_count);
    std::vector<std::uint32_t> output_seg_id(group_count),
        out_boundary(group_count);

    std::vector<std::uint32_t> src_seg_id, src_out_base, src_group;
    std::uint32_t cand_acc = 0, live_off = 0;
    for (std::size_t g = 0; g < group_count; ++g) {
      std::uint32_t within = 0;
      for (std::size_t ord = groups[g].begin; ord < groups[g].end; ++ord) {
        src_seg_id.push_back(h_dir_seg_id_[ord]);
        src_out_base.push_back(cand_acc + within);
        src_group.push_back(static_cast<std::uint32_t>(g));
        within += h_dir_live_[ord];
      }
      cand_acc += groups[g].live;
      dirty_live[g] = groups[g].live;
      dirty_live_offset[g] = live_off;
      live_off += groups[g].live;
      dirty_output_base[g] = static_cast<std::uint32_t>(g);
      output_seg_id[g] = alloc_segment();
      out_boundary[g] = h_dir_boundary_[groups[g].end - 1];
    }
    const std::size_t candidate_count = cand_acc;
    const std::size_t src_count = src_seg_id.size();

    upload_vec(seg_src_seg_id_, src_seg_id, stream);
    upload_vec(seg_src_out_base_, src_out_base, stream);
    upload_vec(seg_src_group_, src_group, stream);
    upload_vec(seg_dirty_live_, dirty_live, stream);
    upload_vec(seg_d_dirty_live_offset_, dirty_live_offset, stream);
    upload_vec(seg_d_dirty_k_, dirty_k, stream);
    upload_vec(seg_d_dirty_output_base_, dirty_output_base, stream);
    upload_vec(seg_d_output_seg_id_, output_seg_id, stream);
    upload_vec(seg_d_out_boundary_, out_boundary, stream);
    seg_d_out_value_sum_.resize(group_count);

    resize_seg_candidates(candidate_count);
    if (src_count > 0 && candidate_count > 0) {
      gpulsmopt_detail::seg_gather_group_ordered_kernel<<<
          static_cast<unsigned>(src_count), 256, 0, stream>>>(
          raw_or_null(pool_keys_), raw_or_null(pool_values_),
          raw_or_null(seg_bucket_live_), raw_or_null(seg_src_seg_id_),
          raw_or_null(seg_src_out_base_), raw_or_null(seg_src_group_), src_count,
          raw_or_null(seg_cand_seg_), raw_or_null(seg_cand_key_),
          raw_or_null(seg_cand_value_));
      CUDA_CHECK(cudaGetLastError());
    }

    rebuild_output_segments(candidate_count, group_count, stream);

    std::vector<int> group_of_ord(n, -1);
    for (std::size_t g = 0; g < group_count; ++g) {
      for (std::size_t ord = groups[g].begin; ord < groups[g].end; ++ord) {
        group_of_ord[ord] = static_cast<int>(g);
      }
    }
    std::vector<std::uint32_t> new_seg_id, new_boundary, new_live,
        new_value_sum;
    new_seg_id.reserve(n);
    new_boundary.reserve(n);
    new_live.reserve(n);
    new_value_sum.reserve(n);
    for (std::size_t ord = 0; ord < n; ++ord) {
      const int g = group_of_ord[ord];
      if (g < 0) {
        new_seg_id.push_back(h_dir_seg_id_[ord]);
        new_boundary.push_back(h_dir_boundary_[ord]);
        new_live.push_back(h_dir_live_[ord]);
        new_value_sum.push_back(h_dir_value_sum_[ord]);
        continue;
      }
      if (ord == groups[g].begin) {
        new_seg_id.push_back(output_seg_id[g]);
        new_boundary.push_back(out_boundary[g]);
        new_live.push_back(groups[g].live);
        new_value_sum.push_back(groups[g].value_sum);
        for (std::size_t src = groups[g].begin; src < groups[g].end; ++src) {
          free_segment(h_dir_seg_id_[src]);
        }
      }
    }
    h_dir_seg_id_ = std::move(new_seg_id);
    h_dir_boundary_ = std::move(new_boundary);
    h_dir_live_ = std::move(new_live);
    h_dir_value_sum_ = std::move(new_value_sum);
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

  std::size_t c0_log_live() const {
    return c0_log_count_;
  }

  std::size_t c0_flush_budget() const {
    std::size_t budget = std::min<std::size_t>(
        static_cast<std::size_t>(GPULSMOPT_C0_FLUSH_BUDGET),
        gpulsmopt_detail::kC0LogMaxIndex);
    if (max_elements_ > 0)
      budget = std::min(budget, max_elements_);
    return std::max<std::size_t>(budget, 1);
  }

  struct SortedRun {
    std::uint32_t log_total = 0;
    std::uint32_t level = 0;
    thrust::device_vector<std::uint32_t> keys;
    thrust::device_vector<std::uint32_t> values;
    thrust::device_vector<std::uint8_t> ops;
  };

  void ensure_c0_log(cudaStream_t stream) {
    (void)stream;
    const std::size_t reserve_count = c0_flush_budget();
    if (c0_log_keys_.capacity() < reserve_count) {
      for (std::size_t i = 0; i < run_buffer_pool_.size(); ++i) {
        SortedRun &run = run_buffer_pool_[i];
        if (run.keys.capacity() < reserve_count ||
            run.values.capacity() < reserve_count ||
            run.ops.capacity() < reserve_count)
          continue;
        c0_log_keys_ = std::move(run.keys);
        c0_log_values_ = std::move(run.values);
        c0_log_ops_ = std::move(run.ops);
        if (i + 1 != run_buffer_pool_.size())
          run_buffer_pool_[i] = std::move(run_buffer_pool_.back());
        run_buffer_pool_.pop_back();
        c0_log_keys_.clear();
        c0_log_values_.clear();
        c0_log_ops_.clear();
        return;
      }
    }
    c0_log_keys_.reserve(reserve_count);
    c0_log_values_.reserve(reserve_count);
    c0_log_ops_.reserve(reserve_count);
  }

  bool append_c0_log(const std::uint32_t *keys_in,
                     const std::uint32_t *values_in, std::uint8_t op,
                     std::size_t count, cudaStream_t stream) {
    if (count == 0)
      return true;
    if (count > gpulsmopt_detail::kC0LogMaxIndex - c0_log_count_)
      return false;
    ensure_c0_log(stream);
    const std::uint32_t old_total = c0_log_count_;
    const std::size_t new_total = static_cast<std::size_t>(old_total) + count;
    {
      GPULSMOPT_PROF_PHASE(prof_append_ms_);
      c0_log_keys_.resize(new_total);
      c0_log_values_.resize(new_total);
      c0_log_ops_.resize(new_total);
      const int block = 256;
      const int grid = static_cast<int>((count + block - 1) / block);
      gpulsmopt_detail::c0_log_append_kernel<<<grid, block, 0, stream>>>(
          keys_in, values_in, op, old_total, count, raw_or_null(c0_log_keys_),
          raw_or_null(c0_log_values_), raw_or_null(c0_log_ops_));
      CUDA_CHECK(cudaGetLastError());
    }
    c0_log_count_ = static_cast<std::uint32_t>(new_total);
    overlay_dirty_ = true;
    read_view_dirty_ = true;
    return true;
  }

  void clear_c0_log(cudaStream_t stream) {
    if (c0_log_count_ == 0)
      return;
    c0_log_count_ = 0;
    c0_log_keys_.clear();
    c0_log_values_.clear();
    c0_log_ops_.clear();
    (void)stream;
  }

  bool runs_should_drain_to_sheet() const {
    std::size_t total = sheet_live_count_ + c0_log_count_;
    std::size_t largest = 0;
    for (const auto &g : runs_) {
      total += g.log_total;
      if (g.log_total > largest)
        largest = g.log_total;
    }
    const std::size_t floor = 4 * c0_flush_budget();
    const std::size_t five_pct = total / 20;
    return largest >= (floor > five_pct ? floor : five_pct);
  }

  void drain_c0_for_space(cudaStream_t stream) {
    if (c0_log_count_ == 0)
      return;
    flush_c0_to_run(stream);
    if (runs_should_drain_to_sheet())
      merge_down(stream);
  }

  void insert_records(const std::uint32_t *keys_in,
                      const std::uint32_t *values_in, std::uint8_t op,
                      std::size_t count, cudaStream_t stream) {
    if (try_direct_run(keys_in, values_in, op, count, stream))
      return;
    std::size_t off = 0;
    while (off < count) {
      const std::size_t budget = c0_flush_budget();
      if (c0_log_count_ >= budget)
        drain_c0_for_space(stream);
      if (c0_log_count_ >= gpulsmopt_detail::kC0LogMaxIndex)
        drain_c0_for_space(stream);
      const std::size_t cap_left =
          gpulsmopt_detail::kC0LogMaxIndex - c0_log_count_;
      const std::size_t budget_left =
          budget > c0_log_count_ ? budget - c0_log_count_ : cap_left;
      std::size_t chunk = count - off;
      chunk = std::min(chunk, cap_left);
      chunk = std::min(chunk, budget_left);
      if (chunk == 0) {
        drain_c0_for_space(stream);
        continue;
      }
      const std::uint32_t *vals = values_in ? values_in + off : nullptr;
      if (!append_c0_log(keys_in + off, vals, op, chunk, stream))
        throw std::runtime_error("C0 log append failed");
      off += chunk;
    }
  }

  SortedRun acquire_run_storage(std::size_t count, std::uint32_t level) {
    std::size_t selected = run_buffer_pool_.size();
    std::size_t selected_capacity = std::numeric_limits<std::size_t>::max();
    for (std::size_t i = 0; i < run_buffer_pool_.size(); ++i) {
      const SortedRun &run = run_buffer_pool_[i];
      const std::size_t capacity = run.keys.capacity();
      if (capacity >= count && run.values.capacity() >= count &&
          run.ops.capacity() >= count && capacity < selected_capacity) {
        selected = i;
        selected_capacity = capacity;
      }
    }
    SortedRun run;
    if (selected != run_buffer_pool_.size()) {
      run = std::move(run_buffer_pool_[selected]);
      if (selected + 1 != run_buffer_pool_.size())
        run_buffer_pool_[selected] = std::move(run_buffer_pool_.back());
      run_buffer_pool_.pop_back();
    }
    run.keys.resize(count);
    run.values.resize(count);
    run.ops.resize(count);
    run.log_total = static_cast<std::uint32_t>(count);
    run.level = level;
    return run;
  }

  void release_run_storage(SortedRun &&run) {
    run.log_total = 0;
    run.level = 0;
    run.keys.clear();
    run.values.clear();
    run.ops.clear();
    run_buffer_pool_.push_back(std::move(run));
  }

  void recycle_active_runs() {
    for (auto &run : runs_)
      release_run_storage(std::move(run));
    runs_.clear();
  }

  void sort_log(thrust::device_vector<std::uint32_t> &k,
                thrust::device_vector<std::uint32_t> &v,
                thrust::device_vector<std::uint8_t> &op, std::size_t n,
                cudaStream_t stream) {
    if (n == 0) {
      k.clear();
      v.clear();
      op.clear();
      return;
    }
    auto policy = thrust::cuda::par.on(stream);
    auto payload = thrust::make_zip_iterator(
        thrust::make_tuple(v.begin(), op.begin()));
    thrust::sort_by_key(policy, k.begin(), k.begin() + n, payload);
    k.resize(n);
    v.resize(n);
    op.resize(n);
  }

  SortedRun merge_two_runs(SortedRun &a, SortedRun &b, cudaStream_t stream) {
    auto policy = thrust::cuda::par.on(stream);
    const std::size_t na = a.log_total, nb = b.log_total;
    SortedRun out = acquire_run_storage(na + nb, a.level + 1);
    thrust::merge_by_key(
        policy, b.keys.begin(), b.keys.begin() + nb, a.keys.begin(),
        a.keys.begin() + na,
        thrust::make_zip_iterator(
            thrust::make_tuple(b.values.begin(), b.ops.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(a.values.begin(), a.ops.begin())),
        out.keys.begin(),
        thrust::make_zip_iterator(
            thrust::make_tuple(out.values.begin(), out.ops.begin())));
    return out;
  }

  void append_sorted_run(SortedRun &&run, cudaStream_t stream) {
    runs_.push_back(std::move(run));
    while (runs_.size() >= 2 &&
           runs_[runs_.size() - 1].level == runs_[runs_.size() - 2].level) {
      SortedRun newer = std::move(runs_.back());
      runs_.pop_back();
      SortedRun older = std::move(runs_.back());
      runs_.pop_back();
      SortedRun merged;
      {
        GPULSMOPT_PROF_PHASE(prof_runmerge_ms_);
        merged = merge_two_runs(older, newer, stream);
      }
      release_run_storage(std::move(older));
      release_run_storage(std::move(newer));
      runs_.push_back(std::move(merged));
    }
    overlay_dirty_ = true;
    read_view_dirty_ = true;
  }

  bool try_direct_run(const std::uint32_t *keys_in,
                      const std::uint32_t *values_in, std::uint8_t op,
                      std::size_t count, cudaStream_t stream) {
    const std::size_t budget = c0_flush_budget();
    if (c0_log_count_ != 0 || count < budget || count % budget != 0 ||
        count > std::numeric_limits<std::uint32_t>::max())
      return false;
    const std::size_t units = count / budget;
    if ((units & (units - 1)) != 0)
      return false;
    std::uint32_t level = 0;
    for (std::size_t n = units; n > 1; n >>= 1)
      ++level;
    if (!runs_.empty() && runs_.back().level < level)
      return false;
    SortedRun run = acquire_run_storage(count, level);
    {
      GPULSMOPT_PROF_PHASE(prof_append_ms_);
      const int block = 256;
      const int grid = static_cast<int>((count + block - 1) / block);
      gpulsmopt_detail::c0_log_append_kernel<<<grid, block, 0, stream>>>(
          keys_in, values_in, op, 0u, count, raw_or_null(run.keys),
          raw_or_null(run.values), raw_or_null(run.ops));
      CUDA_CHECK(cudaGetLastError());
    }
    {
      GPULSMOPT_PROF_PHASE(prof_flushsort_ms_);
      sort_log(run.keys, run.values, run.ops, count, stream);
    }
    append_sorted_run(std::move(run), stream);
    return true;
  }

  void flush_c0_to_run(cudaStream_t stream) {
    const std::size_t c0_total = c0_log_count_;
    if (c0_total == 0)
      return;
    SortedRun run;
    run.keys = std::move(c0_log_keys_);
    run.values = std::move(c0_log_values_);
    run.ops = std::move(c0_log_ops_);
    {
      GPULSMOPT_PROF_PHASE(prof_flushsort_ms_);
      sort_log(run.keys, run.values, run.ops, c0_total, stream);
    }
    run.log_total = static_cast<std::uint32_t>(c0_total);
    run.level = 0;
    c0_log_count_ = 0;
    append_sorted_run(std::move(run), stream);
  }

  struct OverlayReadIndex {
    thrust::device_vector<std::uint32_t> gk, gv;
    thrust::device_vector<std::uint8_t> gop;
    std::size_t u = 0, ins = 0;
    thrust::device_vector<std::uint32_t> ins_prefix;
    thrust::device_vector<std::uint32_t> tomb_val_prefix;
    thrust::device_vector<std::uint32_t> tomb_cnt_prefix;
    thrust::device_vector<std::uint32_t> live_ins_keys;
    std::size_t live_ins_count = 0;
    thrust::device_vector<std::uint32_t> killed_keys;
    std::size_t killed_count = 0;
  };

  void resolve_overlay(thrust::device_vector<std::uint32_t> &gk,
                       thrust::device_vector<std::uint32_t> &gv,
                       thrust::device_vector<std::uint8_t> &gop, std::size_t &u,
                       std::size_t &ins, cudaStream_t stream,
                       bool consume = false) {
    u = 0;
    ins = 0;
    std::size_t total = c0_log_count_;
    for (auto &g : runs_)
      total += g.log_total;
    if (total == 0) {
      gk.clear();
      gv.clear();
      gop.clear();
      return;
    }
    auto policy = thrust::cuda::par.on(stream);
    if (consume && c0_log_count_ == 0 && runs_.size() == 1) {
      SortedRun run = std::move(runs_.back());
      runs_.clear();
      gk = std::move(run.keys);
      gv = std::move(run.values);
      gop = std::move(run.ops);
      u = run.log_total;
      auto beg = thrust::make_zip_iterator(
          thrust::make_tuple(gk.begin(), gv.begin(), gop.begin()));
      auto mid = thrust::stable_partition(
          policy, beg, beg + u, gpulsmopt_detail::TupleOpIsInsert{});
      ins = static_cast<std::size_t>(mid - beg);
      return;
    }
    if (consume && runs_.empty() && c0_log_count_ > 0) {
      gk = std::move(c0_log_keys_);
      gv = std::move(c0_log_values_);
      gop = std::move(c0_log_ops_);
      u = c0_log_count_;
      c0_log_count_ = 0;
      sort_log(gk, gv, gop, u, stream);
      auto beg = thrust::make_zip_iterator(
          thrust::make_tuple(gk.begin(), gv.begin(), gop.begin()));
      auto mid = thrust::stable_partition(
          policy, beg, beg + u, gpulsmopt_detail::TupleOpIsInsert{});
      ins = static_cast<std::size_t>(mid - beg);
      return;
    }
    gk.resize(total);
    gv.resize(total);
    gop.resize(total);

    struct Src {
      const std::uint32_t *k;
      const std::uint32_t *v;
      const std::uint8_t *op;
      std::size_t n;
    };
    std::vector<Src> src;
    src.reserve(runs_.size() + 1);
    if (c0_log_count_ > 0) {
      const std::size_t s = c0_log_count_;
      ov_c0k_.resize(s);
      ov_c0v_.resize(s);
      ov_c0op_.resize(s);
      thrust::copy(policy, c0_log_keys_.begin(), c0_log_keys_.begin() + s,
                   ov_c0k_.begin());
      thrust::copy(policy, c0_log_values_.begin(), c0_log_values_.begin() + s,
                   ov_c0v_.begin());
      thrust::copy(policy, c0_log_ops_.begin(), c0_log_ops_.begin() + s,
                   ov_c0op_.begin());
      sort_log(ov_c0k_, ov_c0v_, ov_c0op_, s, stream);
      src.push_back({raw_or_null(ov_c0k_), raw_or_null(ov_c0v_),
                     raw_or_null(ov_c0op_), s});
    }
    for (auto &g : runs_)
      if (g.log_total)
        src.push_back({raw_or_null(g.keys), raw_or_null(g.values),
                       raw_or_null(g.ops), g.log_total});

    if (src.size() == 1) {
      const Src &s = src[0];
      thrust::copy(policy, thrust::device_pointer_cast(s.k),
                   thrust::device_pointer_cast(s.k) + s.n, gk.begin());
      thrust::copy(policy, thrust::device_pointer_cast(s.v),
                   thrust::device_pointer_cast(s.v) + s.n, gv.begin());
      thrust::copy(policy, thrust::device_pointer_cast(s.op),
                   thrust::device_pointer_cast(s.op) + s.n, gop.begin());
    } else {
      const std::size_t nmerge = src.size() - 1;
      if (nmerge >= 2) {
        ov_mk_.resize(total);
        ov_mv_.resize(total);
        ov_mop_.resize(total);
      }
      const std::uint32_t *curK = src[0].k;
      const std::uint32_t *curV = src[0].v;
      const std::uint8_t *curOp = src[0].op;
      std::size_t curN = src[0].n;
      for (std::size_t j = 0; j < nmerge; ++j) {
        const bool to_g = (((nmerge - 1 - j) & 1u) == 0u);
        std::uint32_t *dK = to_g ? raw_or_null(gk) : raw_or_null(ov_mk_);
        std::uint32_t *dV = to_g ? raw_or_null(gv) : raw_or_null(ov_mv_);
        std::uint8_t *dOp = to_g ? raw_or_null(gop) : raw_or_null(ov_mop_);
        const Src &s = src[j + 1];
        auto aval = thrust::make_zip_iterator(
            thrust::make_tuple(thrust::device_pointer_cast(curV),
                               thrust::device_pointer_cast(curOp)));
        auto bval = thrust::make_zip_iterator(
            thrust::make_tuple(thrust::device_pointer_cast(s.v),
                               thrust::device_pointer_cast(s.op)));
        auto oval = thrust::make_zip_iterator(
            thrust::make_tuple(thrust::device_pointer_cast(dV),
                               thrust::device_pointer_cast(dOp)));
        thrust::merge_by_key(policy, thrust::device_pointer_cast(curK),
                             thrust::device_pointer_cast(curK) + curN,
                             thrust::device_pointer_cast(s.k),
                             thrust::device_pointer_cast(s.k) + s.n, aval, bval,
                             thrust::device_pointer_cast(dK), oval);
        curK = dK;
        curV = dV;
        curOp = dOp;
        curN += s.n;
      }
    }
    u = total;
    auto beg = thrust::make_zip_iterator(
        thrust::make_tuple(gk.begin(), gv.begin(), gop.begin()));
    auto mid = thrust::stable_partition(policy, beg, beg + u,
                                        gpulsmopt_detail::TupleOpIsInsert{});
    ins = static_cast<std::size_t>(mid - beg);
  }

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
    ix.killed_keys.clear();
    ix.killed_count = 0;
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
      ix.killed_keys.resize(tomb);
      auto kend = thrust::copy_if(policy, ix.gk.begin() + ins, ix.gk.begin() + u,
                                  tflag.begin(), ix.killed_keys.begin(),
                                  gpulsmopt_detail::NonZeroU32{});
      ix.killed_count =
          static_cast<std::size_t>(kend - ix.killed_keys.begin());
      ix.killed_keys.resize(ix.killed_count);
    }
    ix.live_ins_keys.resize(ins);
    ix.live_ins_count = 0;
    if (ins > 0) {
      thrust::device_vector<std::uint32_t> live_flag(ins);
      const int lblock = 256;
      const int lgrid = static_cast<int>((ins + lblock - 1) / lblock);
      gpulsmopt_detail::mark_live_inserts_kernel<<<lgrid, lblock, 0, stream>>>(
          raw_or_null(ix.gk), ins, raw_or_null(ix.gk) + ins, u - ins,
          raw_or_null(live_flag));
      CUDA_CHECK(cudaGetLastError());
      auto lend = thrust::copy_if(policy, ix.gk.begin(), ix.gk.begin() + ins,
                                  live_flag.begin(), ix.live_ins_keys.begin(),
                                  gpulsmopt_detail::NonZeroU32{});
      ix.live_ins_count =
          static_cast<std::size_t>(lend - ix.live_ins_keys.begin());
    }
    ix.live_ins_keys.resize(ix.live_ins_count);
  }

  OverlayReadIndex &overlay_index(cudaStream_t stream) {
    if (overlay_dirty_) {
      build_overlay_read_index(cached_overlay_, stream);
      overlay_dirty_ = false;
    }
    return cached_overlay_;
  }

  void apply_sheet_deletes(std::size_t tomb, cudaStream_t stream) {
    if (tomb == 0 || h_dir_seg_id_.empty())
      return;
    auto policy = thrust::cuda::par.on(stream);
    seg_inc_ordinal_.resize(tomb);
    {
      const int block = 256;
      const int grid = static_cast<int>((tomb + block - 1) / block);
      gpulsmopt_detail::seg_route_keys_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(scratch_delete_keys_), tomb, raw_or_null(d_dir_boundary_),
          h_dir_seg_id_.size(), raw_or_null(seg_inc_ordinal_));
      CUDA_CHECK(cudaGetLastError());
    }
    seg_dirty_ord_.resize(tomb);
    seg_dirty_count_.resize(tomb);
    auto rle_end = thrust::reduce_by_key(
        policy, seg_inc_ordinal_.begin(), seg_inc_ordinal_.begin() + tomb,
        thrust::make_constant_iterator<std::uint32_t>(1u),
        seg_dirty_ord_.begin(), seg_dirty_count_.begin());
    const std::size_t m =
        static_cast<std::size_t>(rle_end.first - seg_dirty_ord_.begin());
    if (m == 0)
      return;
    prepare_dirty_plan(m, stream);
    seg_inc_bucket_.resize(tomb);
    {
      const int block = 256;
      const int grid = static_cast<int>((tomb + block - 1) / block);
      gpulsmopt_detail::seg_route_buckets_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(scratch_delete_keys_),
          raw_or_null(seg_inc_ordinal_), tomb,
          raw_or_null(d_dir_seg_id_), raw_or_null(seg_bucket_max_),
          raw_or_null(seg_inc_bucket_));
      CUDA_CHECK(cudaGetLastError());
    }
    seg_dirty_bucket_.resize(tomb);
    seg_dirty_bucket_count_.resize(tomb);
    auto bucket_rle_end = thrust::reduce_by_key(
        policy, seg_inc_bucket_.begin(), seg_inc_bucket_.begin() + tomb,
        thrust::make_constant_iterator<std::uint32_t>(1u),
        seg_dirty_bucket_.begin(), seg_dirty_bucket_count_.begin());
    const std::size_t mb = static_cast<std::size_t>(
        bucket_rle_end.first - seg_dirty_bucket_.begin());
    seg_dirty_bucket_begin_.resize(mb);
    thrust::exclusive_scan(policy, seg_dirty_bucket_count_.begin(),
                           seg_dirty_bucket_count_.begin() + mb,
                           seg_dirty_bucket_begin_.begin());
    seg_new_live_.resize(m);
    seg_new_value_sum_.resize(m);
    {
      const int block = 256;
      const int grid = static_cast<int>((m + block - 1) / block);
      gpulsmopt_detail::seg_init_delete_totals_kernel<<<grid, block, 0,
                                                       stream>>>(
        raw_or_null(seg_d_dirty_old_live_),
          raw_or_null(seg_d_dirty_old_value_sum_), m,
          raw_or_null(seg_new_live_), raw_or_null(seg_new_value_sum_));
      CUDA_CHECK(cudaGetLastError());
    }
    {
      const int block = 256;
      const int grid = static_cast<int>((mb + block - 1) / block);
      gpulsmopt_detail::seg_delete_dirty_buckets_kernel<<<
          grid, block, 0, stream>>>(
          raw_or_null(scratch_delete_keys_),
          raw_or_null(seg_dirty_bucket_),
          raw_or_null(seg_dirty_bucket_count_),
          raw_or_null(seg_dirty_bucket_begin_), mb,
          raw_or_null(seg_dirty_ord_), m, raw_or_null(d_dir_seg_id_),
          raw_or_null(pool_keys_), raw_or_null(pool_values_),
          raw_or_null(pool_valid_), raw_or_null(seg_bucket_max_),
          raw_or_null(seg_bucket_live_), raw_or_null(seg_bucket_value_sum_),
          raw_or_null(seg_new_live_), raw_or_null(seg_new_value_sum_));
      CUDA_CHECK(cudaGetLastError());
    }
    std::vector<std::uint32_t> dirty_ord(m), new_live(m), new_value_sum(m);
    CUDA_CHECK(cudaMemcpyAsync(dirty_ord.data(), raw_or_null(seg_dirty_ord_),
                               m * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(new_live.data(), raw_or_null(seg_new_live_),
                               m * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        new_value_sum.data(), raw_or_null(seg_new_value_sum_),
        m * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < m; ++i) {
      const std::size_t ord = dirty_ord[i];
      if (new_live[i] < h_dir_live_[ord])
        sheet_fragmented_ = true;
      h_dir_live_[ord] = new_live[i];
      h_dir_value_sum_[ord] = new_value_sum[i];
    }
    recompute_sheet_live_count();
    upload_directory_metadata(stream);
  }

  void merge_down(cudaStream_t stream) {
    thrust::device_vector<std::uint32_t> gk, gv;
    thrust::device_vector<std::uint8_t> gop;
    std::size_t u = 0, ins = 0;
    {
      GPULSMOPT_PROF_PHASE(prof_resolve_ms_);
      resolve_overlay(gk, gv, gop, u, ins, stream, true);
    }
    if (u > 0) {
      auto policy = thrust::cuda::par.on(stream);
      const std::size_t tomb = u - ins;
      if (tomb > 0 && h_dir_seg_id_.size() > 0) {
        GPULSMOPT_PROF_PHASE(prof_delete_ms_);
        scratch_delete_keys_.resize(tomb);
        thrust::copy(policy, gk.begin() + ins, gk.begin() + u,
                     scratch_delete_keys_.begin());
        apply_sheet_deletes(tomb, stream);
      }
      if (ins > 0) {
        GPULSMOPT_PROF_PHASE(prof_sheetmerge_ms_);
        scratch_incoming_keys_.resize(ins);
        scratch_incoming_values_.resize(ins);
        std::size_t live = ins;
        if (tomb > 0) {
          thrust::device_vector<std::uint32_t> live_flag(ins);
          const int lblock = 256;
          const int lgrid = static_cast<int>((ins + lblock - 1) / lblock);
          gpulsmopt_detail::mark_live_inserts_kernel<<<lgrid, lblock, 0,
                                                       stream>>>(
              raw_or_null(gk), ins, raw_or_null(gk) + ins, tomb,
              raw_or_null(live_flag));
          CUDA_CHECK(cudaGetLastError());
          auto in_begin = thrust::make_zip_iterator(
              thrust::make_tuple(gk.begin(), gv.begin()));
          auto out_begin = thrust::make_zip_iterator(thrust::make_tuple(
              scratch_incoming_keys_.begin(), scratch_incoming_values_.begin()));
          auto out_end =
              thrust::copy_if(policy, in_begin, in_begin + ins, live_flag.begin(),
                              out_begin, gpulsmopt_detail::NonZeroU32{});
          live = static_cast<std::size_t>(out_end - out_begin);
        } else {
          thrust::copy(policy, gk.begin(), gk.begin() + ins,
                       scratch_incoming_keys_.begin());
          thrust::copy(policy, gv.begin(), gv.begin() + ins,
                       scratch_incoming_values_.begin());
        }
        if (live > 0)
          merge_incoming_into_sheet(live, stream);
      }
    }
    recycle_active_runs();
    clear_c0_log(stream);
    SortedRun recycled;
    recycled.keys = std::move(gk);
    recycled.values = std::move(gv);
    recycled.ops = std::move(gop);
    if (recycled.keys.capacity() > 0)
      release_run_storage(std::move(recycled));
    overlay_dirty_ = true;
    read_view_dirty_ = true;
  }

  void maybe_flush_and_merge(cudaStream_t stream) {
    if (c0_log_live() >= c0_flush_budget())
      flush_c0_to_run(stream);
    if (runs_should_drain_to_sheet())
      merge_down(stream);
  }

  void refresh_read_view(cudaStream_t stream) {
    if (!read_view_dirty_)
      return;
    const std::size_t n = c0_log_count_;
    if (n > 0) {
      c0_sorted_keys_ = c0_log_keys_;
      c0_sorted_values_ = c0_log_values_;
      c0_sorted_ops_ = c0_log_ops_;
      sort_log(c0_sorted_keys_, c0_sorted_values_, c0_sorted_ops_, n, stream);
      c0_sorted_count_ = static_cast<std::uint32_t>(n);
    } else {
      c0_sorted_count_ = 0;
    }

    std::vector<const std::uint32_t *> hk, hv;
    std::vector<const std::uint8_t *> ho;
    std::vector<std::uint32_t> hc;
    if (c0_sorted_count_ > 0) {
      hk.push_back(raw_or_null(c0_sorted_keys_));
      hv.push_back(raw_or_null(c0_sorted_values_));
      ho.push_back(raw_or_null(c0_sorted_ops_));
      hc.push_back(c0_sorted_count_);
    }
    for (auto it = runs_.rbegin(); it != runs_.rend(); ++it) {
      if (it->log_total == 0)
        continue;
      hk.push_back(raw_or_null(it->keys));
      hv.push_back(raw_or_null(it->values));
      ho.push_back(raw_or_null(it->ops));
      hc.push_back(it->log_total);
    }
    num_read_runs_ = static_cast<int>(hk.size());
    read_run_keys_ = hk;
    read_run_vals_ = hv;
    read_run_ops_ = ho;
    read_run_cnt_ = hc;
    read_view_dirty_ = false;
  }

  void lookup_flip(std::size_t n, const std::uint32_t *const *rk,
                   const std::uint32_t *const *rv, const std::uint8_t *const *ro,
                   const std::uint32_t *rc, int num_runs, cudaStream_t stream) {
    auto policy = thrust::cuda::par.on(stream);
    seg_inc_ordinal_.resize(n);
    {
      const int block = 256;
      const int grid = static_cast<int>((n + block - 1) / block);
      gpulsmopt_detail::seg_route_keys_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(lookup_sorted_queries_), n, raw_or_null(d_dir_boundary_),
          h_dir_seg_id_.size(), raw_or_null(seg_inc_ordinal_));
      CUDA_CHECK(cudaGetLastError());
    }
    seg_dirty_ord_.resize(n);
    seg_dirty_count_.resize(n);
    auto rle = thrust::reduce_by_key(
        policy, seg_inc_ordinal_.begin(), seg_inc_ordinal_.begin() + n,
        thrust::make_constant_iterator<std::uint32_t>(1u),
        seg_dirty_ord_.begin(), seg_dirty_count_.begin());
    const std::size_t m =
        static_cast<std::size_t>(rle.first - seg_dirty_ord_.begin());
    std::vector<std::uint32_t> dord(m), dcnt(m);
    CUDA_CHECK(cudaMemcpyAsync(dord.data(), raw_or_null(seg_dirty_ord_),
                               m * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(dcnt.data(), raw_or_null(seg_dirty_count_),
                               m * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::uint32_t> dseg(m), dbeg(m), dend(m);
    std::uint32_t acc = 0;
    for (std::size_t j = 0; j < m; ++j) {
      dseg[j] = h_dir_seg_id_[dord[j]];
      dbeg[j] = acc;
      acc += dcnt[j];
      dend[j] = acc;
    }
    upload_vec(seg_d_dirty_seg_id_, dseg, stream);
    upload_vec(seg_d_dirty_in_begin_, dbeg, stream);
    upload_vec(seg_d_dirty_in_end_, dend, stream);
    gpulsmopt_detail::seg_flip_lookup_kernel<<<static_cast<unsigned>(m), 256, 0,
                                               stream>>>(
        raw_or_null(lookup_sorted_queries_), raw_or_null(seg_d_dirty_seg_id_),
        raw_or_null(seg_d_dirty_in_begin_), raw_or_null(seg_d_dirty_in_end_), m,
        raw_or_null(lookup_temp_values_), raw_or_null(lookup_temp_found_),
        raw_or_null(pool_keys_), raw_or_null(pool_values_),
        raw_or_null(pool_valid_), raw_or_null(seg_bucket_max_),
        raw_or_null(seg_bucket_live_));
    CUDA_CHECK(cudaGetLastError());
    if (num_runs > 0) {
      const int block = 256;
      const int grid = static_cast<int>((n + block - 1) / block);
      gpulsmopt_detail::overlay_walk_override_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(lookup_sorted_queries_), n, raw_or_null(lookup_temp_values_),
          raw_or_null(lookup_temp_found_), rk, rv, ro, rc, num_runs);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  void lookup_layered(const DeviceLookupBatch &batch, cudaStream_t stream) {
    const std::size_t n = batch.count;
    refresh_read_view(stream);
    const std::uint32_t *const *rk = raw_or_null(read_run_keys_);
    const std::uint32_t *const *rv = raw_or_null(read_run_vals_);
    const std::uint8_t *const *ro = raw_or_null(read_run_ops_);
    const std::uint32_t *rc = raw_or_null(read_run_cnt_);
    const int num_runs = num_read_runs_;

    const std::size_t nbuckets =
        h_dir_seg_id_.size() *
        static_cast<std::size_t>(gpulsmopt_detail::kSegmentBuckets);
    const bool use_flip =
        !h_dir_seg_id_.empty() && n >= nbuckets &&
        n >= static_cast<std::size_t>(GPULSMOPT_LOOKUP_FLIX_MIN_BATCH);

    if (!use_flip && num_runs == 0) {
      const int block = 256;
      const int grid = static_cast<int>((n + block - 1) / block);
      gpulsmopt_detail::point_lookup_walk_kernel<<<grid, block, 0, stream>>>(
          batch.queries, n, batch.out_values, batch.out_found, nullptr, rk, rv,
          ro, rc, 0, raw_or_null(d_dir_boundary_), raw_or_null(d_dir_seg_id_),
          h_dir_seg_id_.size(), raw_or_null(pool_keys_),
          raw_or_null(pool_values_), raw_or_null(pool_valid_),
          raw_or_null(seg_bucket_max_));
      CUDA_CHECK(cudaGetLastError());
      return;
    }

    auto policy = thrust::cuda::par.on(stream);
    lookup_sorted_queries_.resize(n);
    lookup_permutation_.resize(n);
    lookup_temp_values_.resize(n);
    lookup_temp_found_.resize(n);
    thrust::copy(policy, batch.queries, batch.queries + n, lookup_sorted_queries_.begin());
    thrust::sequence(policy, lookup_permutation_.begin(), lookup_permutation_.end());
    thrust::sort_by_key(policy, lookup_sorted_queries_.begin(), lookup_sorted_queries_.end(),
                        lookup_permutation_.begin());

    if (use_flip) {
      lookup_flip(n, rk, rv, ro, rc, num_runs, stream);
      thrust::scatter(policy, lookup_temp_values_.begin(), lookup_temp_values_.end(),
                      lookup_permutation_.begin(),
                      thrust::device_pointer_cast(batch.out_values));
      thrust::scatter(policy, lookup_temp_found_.begin(), lookup_temp_found_.end(),
                      lookup_permutation_.begin(),
                      thrust::device_pointer_cast(batch.out_found));
      CUDA_CHECK(cudaGetLastError());
    } else {
      const int block = 256;
      const int grid = static_cast<int>((n + block - 1) / block);
      gpulsmopt_detail::point_lookup_walk_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(lookup_sorted_queries_), n, batch.out_values, batch.out_found,
          raw_or_null(lookup_permutation_), rk, rv, ro, rc, num_runs,
          raw_or_null(d_dir_boundary_), raw_or_null(d_dir_seg_id_),
          h_dir_seg_id_.size(), raw_or_null(pool_keys_),
          raw_or_null(pool_values_), raw_or_null(pool_valid_),
          raw_or_null(seg_bucket_max_));
      CUDA_CHECK(cudaGetLastError());
    }
  }

  std::size_t max_elements_ = 0;
  std::size_t batch_size_ = 0;
  std::uint32_t target_fill_ = 23;
  std::uint32_t target_segment_live_ = 0;
  std::size_t sheet_live_count_ = 0;
  bool sheet_fragmented_ = false;
  mutable std::shared_mutex snapshot_mutex_;

  std::uint32_t c0_log_count_ = 0;
  thrust::device_vector<std::uint32_t> c0_log_keys_;
  thrust::device_vector<std::uint32_t> c0_log_values_;
  thrust::device_vector<std::uint8_t> c0_log_ops_;

#ifdef GPULSMOPT_PROFILE_INSERT
  double prof_append_ms_ = 0.0;
  double prof_flushsort_ms_ = 0.0;
  double prof_runmerge_ms_ = 0.0;
  double prof_resolve_ms_ = 0.0;
  double prof_delete_ms_ = 0.0;
  double prof_sheetmerge_ms_ = 0.0;
  void reset_insert_prof_() {
    prof_append_ms_ = prof_flushsort_ms_ = prof_runmerge_ms_ = 0.0;
    prof_resolve_ms_ = prof_delete_ms_ = prof_sheetmerge_ms_ = 0.0;
  }
#endif

  thrust::device_vector<std::uint32_t> lookup_sorted_queries_;
  thrust::device_vector<std::uint32_t> lookup_permutation_;
  thrust::device_vector<std::uint32_t> lookup_temp_values_;
  thrust::device_vector<std::uint8_t> lookup_temp_found_;

  thrust::device_vector<std::uint32_t> c0_sorted_keys_;
  thrust::device_vector<std::uint32_t> c0_sorted_values_;
  thrust::device_vector<std::uint8_t> c0_sorted_ops_;
  std::uint32_t c0_sorted_count_ = 0;
  thrust::device_vector<const std::uint32_t *> read_run_keys_;
  thrust::device_vector<const std::uint32_t *> read_run_vals_;
  thrust::device_vector<const std::uint8_t *> read_run_ops_;
  thrust::device_vector<std::uint32_t> read_run_cnt_;
  int num_read_runs_ = 0;
  bool read_view_dirty_ = true;

  std::vector<SortedRun> runs_;
  std::vector<SortedRun> run_buffer_pool_;

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
  std::vector<std::uint32_t> h_dir_value_sum_;
  thrust::device_vector<std::uint32_t> d_dir_seg_id_;
  thrust::device_vector<std::uint32_t> d_dir_boundary_;
  thrust::device_vector<std::uint32_t> d_dir_prefix_;
  thrust::device_vector<std::uint32_t> d_dir_value_sum_;
  thrust::device_vector<std::uint32_t> d_dir_value_prefix_;

  thrust::device_vector<std::uint32_t> scratch_incoming_keys_;
  thrust::device_vector<std::uint32_t> scratch_incoming_values_;
  thrust::device_vector<std::uint32_t> scratch_delete_keys_;

  thrust::device_vector<std::uint32_t> ov_c0k_;
  thrust::device_vector<std::uint32_t> ov_c0v_;
  thrust::device_vector<std::uint8_t> ov_c0op_;
  thrust::device_vector<std::uint32_t> ov_mk_;
  thrust::device_vector<std::uint32_t> ov_mv_;
  thrust::device_vector<std::uint8_t> ov_mop_;

  thrust::device_vector<std::uint32_t> seg_inc_ordinal_;
  thrust::device_vector<std::uint32_t> seg_inc_bucket_;
  thrust::device_vector<std::uint32_t> seg_pull_counter_;
  thrust::device_vector<std::uint32_t> seg_dirty_ord_;
  thrust::device_vector<std::uint32_t> seg_dirty_count_;
  thrust::device_vector<std::uint32_t> seg_dirty_bucket_;
  thrust::device_vector<std::uint32_t> seg_dirty_bucket_count_;
  thrust::device_vector<std::uint32_t> seg_dirty_bucket_begin_;
  thrust::device_vector<std::uint32_t> seg_dirty_bucket_dirty_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_seg_id_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_old_boundary_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_old_live_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_old_value_sum_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_in_begin_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_in_end_;
  thrust::device_vector<std::uint32_t> seg_dirty_live_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_live_offset_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_k_;
  thrust::device_vector<std::uint32_t> seg_d_dirty_output_base_;
  thrust::device_vector<std::uint32_t> seg_d_output_seg_id_;
  thrust::device_vector<std::uint32_t> seg_d_reserved_seg_id_;
  thrust::device_vector<std::uint32_t> seg_d_out_dirty_;
  thrust::device_vector<std::uint32_t> seg_d_out_local_;
  thrust::device_vector<std::uint32_t> seg_d_out_k_;
  thrust::device_vector<std::uint32_t> seg_d_out_live_;
  thrust::device_vector<std::uint32_t> seg_d_out_boundary_;
  thrust::device_vector<std::uint32_t> seg_d_out_value_sum_;
  thrust::device_vector<std::uint32_t> seg_src_seg_id_;
  thrust::device_vector<std::uint32_t> seg_src_out_base_;
  thrust::device_vector<std::uint32_t> seg_src_group_;
  thrust::device_vector<std::uint32_t> seg_slow_;
  thrust::device_vector<std::uint32_t> seg_new_live_;
  thrust::device_vector<std::uint32_t> seg_new_value_sum_;
  thrust::device_vector<std::uint32_t> seg_slow_index_;
  thrust::device_vector<std::uint32_t> seg_slow_seg_id_;
  thrust::device_vector<std::uint32_t> seg_slow_old_boundary_;
  thrust::device_vector<std::uint32_t> seg_slow_old_live_;
  thrust::device_vector<std::uint32_t> seg_slow_in_begin_;
  thrust::device_vector<std::uint32_t> seg_slow_in_end_;
  thrust::device_vector<std::uint32_t> seg_slow_inc_count_;
  thrust::device_vector<std::uint32_t> seg_slow_cand_count_;
  thrust::device_vector<std::uint32_t> seg_plan_total_;
  thrust::device_vector<std::uint32_t> seg_cand_seg_;
  thrust::device_vector<std::uint32_t> seg_cand_key_;
  thrust::device_vector<std::uint32_t> seg_cand_value_;

  thrust::device_vector<std::uint32_t> seg_old_key_;
  thrust::device_vector<std::uint32_t> seg_old_value_;
  thrust::device_vector<std::uint32_t> seg_d_old_offset_;

  OverlayReadIndex cached_overlay_;
  bool overlay_dirty_ = true;
};
