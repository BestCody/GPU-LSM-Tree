#pragma once
#include "gpu_dictionary_adapter.cuh"

#include <cooperative_groups.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
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
#define GPULSMOPT_TARGET_FILL 17
#endif
#ifndef GPULSMOPT_INPLACE_MAX_INCOMING
#define GPULSMOPT_INPLACE_MAX_INCOMING 1024
#endif
#ifndef GPULSMOPT_C0_FLUSH_BUDGET
#define GPULSMOPT_C0_FLUSH_BUDGET (1 << 19)
#endif
#ifndef GPULSMOPT_LOOKUP_FLIX_MIN_BATCH
#define GPULSMOPT_LOOKUP_FLIX_MIN_BATCH (1 << 22)
#endif
// Large batches bypass C0 and enter the Sheet.
#ifndef GPULSMOPT_SCATTER_MIN_BATCH
#define GPULSMOPT_SCATTER_MIN_BATCH (1 << 18)
#endif
// Maximum sorted delta entries per segment.
#ifndef GPULSMOPT_DELTA_CAP
#define GPULSMOPT_DELTA_CAP 4096
#endif
#ifndef GPULSMOPT_DELTA_CAP_JITTER
#define GPULSMOPT_DELTA_CAP_JITTER 1024
#endif
#ifndef GPULSMOPT_SPLIT_FILL
#define GPULSMOPT_SPLIT_FILL 32
#endif
// Split-fill jitter is disabled by default.
#ifndef GPULSMOPT_SPLIT_FILL_JITTER
#define GPULSMOPT_SPLIT_FILL_JITTER 0
#endif

constexpr int kBucketSlots = 32;
constexpr int kSegmentBuckets = GPULSMOPT_SEGMENT_BUCKETS;
constexpr int kSegmentSlots = kSegmentBuckets * kBucketSlots;
static_assert(GPULSMOPT_TARGET_FILL >= 1 &&
                  GPULSMOPT_TARGET_FILL <= kBucketSlots,
              "target fill must fit a bucket");
static_assert(GPULSMOPT_SPLIT_FILL >= GPULSMOPT_TARGET_FILL &&
                  GPULSMOPT_SPLIT_FILL <= kBucketSlots,
              "split trigger must sit between target fill and bucket size");
static_assert(GPULSMOPT_SPLIT_FILL_JITTER >= 0 &&
                  GPULSMOPT_SPLIT_FILL - GPULSMOPT_SPLIT_FILL_JITTER >=
                      GPULSMOPT_TARGET_FILL,
              "jittered split triggers must stay above the target fill");
constexpr int kDeltaCap = GPULSMOPT_DELTA_CAP;
constexpr int kDeltaFence = 16;
constexpr int kDeltaPrefixStride = 8;
static_assert(GPULSMOPT_DELTA_CAP > GPULSMOPT_DELTA_CAP_JITTER &&
                  GPULSMOPT_DELTA_CAP % kDeltaFence == 0,
              "delta capacity must dominate its jitter");
static_assert(GPULSMOPT_DELTA_CAP % kDeltaPrefixStride == 0,
              "delta prefix stride must divide capacity");
static_assert(GPULSMOPT_DELTA_CAP % 256 == 0,
              "delta merge tiling needs a 256-divisible capacity");
// Merged entries each thread handles in one delta rewrite.
constexpr int kDeltaLocal = GPULSMOPT_DELTA_CAP / 256;
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

template <class T> class RawDeviceBuffer {
public:
  RawDeviceBuffer() = default;
  RawDeviceBuffer(const RawDeviceBuffer &) = delete;
  RawDeviceBuffer &operator=(const RawDeviceBuffer &) = delete;
  ~RawDeviceBuffer() {
    if (data_)
      cudaFree(data_);
  }

  void resize_discard(std::size_t count) {
    if (count > capacity_) {
      std::size_t next_capacity = capacity_ == 0 ? 1 : capacity_;
      while (next_capacity < count)
        next_capacity *= 2;
      T *next = nullptr;
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&next),
                            next_capacity * sizeof(T)));
      if (data_)
        CUDA_CHECK(cudaFree(data_));
      data_ = next;
      capacity_ = next_capacity;
    }
    size_ = count;
  }

  T *data() { return data_; }
  const T *data() const { return data_; }
  std::size_t size() const { return size_; }
  std::size_t capacity() const { return capacity_; }

private:
  T *data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

__host__ __device__ inline std::uint32_t ceil_div_u32(std::uint32_t a,
                                                      std::uint32_t b) {
  return (a + b - 1) / b;
}

// Choose a stable split threshold per segment.
__host__ __device__ inline std::uint32_t seg_split_trigger(std::uint32_t seg) {
  const std::uint32_t jitter =
      (seg * 2654435761u >> 16) % (GPULSMOPT_SPLIT_FILL_JITTER + 1u);
  return static_cast<std::uint32_t>(kSegmentBuckets) *
         (static_cast<std::uint32_t>(GPULSMOPT_SPLIT_FILL) - jitter);
}

// Choose a stable delta capacity per segment.
__host__ __device__ inline std::uint32_t seg_delta_cap(std::uint32_t seg) {
  return static_cast<std::uint32_t>(kDeltaCap) -
         ((seg * 2654435761u >> 12) %
          (GPULSMOPT_DELTA_CAP_JITTER + 1u));
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

// Select the active ping-pong delta buffer for a segment.
__device__ inline std::size_t seg_delta_base(std::uint32_t seg,
                                             const std::uint8_t *active) {
  return (2u * static_cast<std::size_t>(seg) +
          static_cast<std::size_t>(active[seg])) *
         static_cast<std::size_t>(kDeltaCap);
}

// Segment directory radix.
constexpr int kDirRadixBits = 14;
constexpr std::size_t kDirRadixSize = std::size_t{1} << kDirRadixBits;

__global__ void dir_build_radix_kernel(const std::uint32_t *dir_boundary,
                                       std::size_t dir_count,
                                       std::uint32_t *radix_first) {
  const std::size_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g > dir_count)
    return;
  const std::size_t bin_hi =
      g == dir_count ? kDirRadixSize
                     : static_cast<std::size_t>(dir_boundary[g] >>
                                                (32 - kDirRadixBits));
  std::size_t bin_lo;
  if (g == 0) {
    bin_lo = 0;
    radix_first[0] = 0u;
  } else {
    bin_lo = static_cast<std::size_t>(dir_boundary[g - 1] >>
                                      (32 - kDirRadixBits)) +
             1;
  }
  for (std::size_t bin = bin_lo; bin <= bin_hi; ++bin)
    radix_first[bin] = static_cast<std::uint32_t>(g);
}

// lower_bound over directory boundaries via one radix load.
__device__ inline std::size_t dir_route(std::uint32_t key,
                                        const std::uint32_t *dir_radix_first,
                                        const std::uint32_t *dir_boundary,
                                        std::size_t dir_count) {
  const std::uint32_t bin = key >> (32 - kDirRadixBits);
  std::size_t lo = dir_radix_first[bin];
  std::size_t hi = dir_radix_first[bin + 1];
  while (lo < hi) {
    const std::size_t mid = (lo + hi) >> 1;
    if (dir_boundary[mid] < key)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

__device__ inline std::uint32_t
block_scan_exclusive(std::uint32_t value, std::uint32_t *scratch);

__global__ void directory_prefix_total_kernel(
    const std::uint32_t *live, const std::uint32_t *value_sum,
    std::size_t count, std::uint32_t *live_prefix,
    std::uint32_t *value_prefix) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  if (count == 0) {
    live_prefix[0] = 0u;
    value_prefix[0] = 0u;
    return;
  }
  live_prefix[count] = live_prefix[count - 1] + live[count - 1];
  value_prefix[count] =
      value_prefix[count - 1] + value_sum[count - 1];
}

__global__ void seg_bucket_prefix_kernel(
    const std::uint32_t *index, const std::uint32_t *dir_seg_id,
    std::size_t count, const std::uint32_t *seg_bucket_live,
    const std::uint32_t *seg_bucket_value_sum,
    std::uint32_t *seg_bucket_live_prefix,
    std::uint32_t *seg_bucket_value_prefix) {
  const std::size_t i = blockIdx.x;
  if (i >= count)
    return;
  const std::uint32_t item = index ? index[i] : static_cast<std::uint32_t>(i);
  const std::uint32_t seg = dir_seg_id ? dir_seg_id[item] : item;
  const std::size_t meta = static_cast<std::size_t>(seg) * kSegmentBuckets;
  const std::size_t out =
      static_cast<std::size_t>(seg) * (kSegmentBuckets + 1);
  const std::uint32_t b = threadIdx.x;
  const std::uint32_t live = seg_bucket_live[meta + b];
  const std::uint32_t sum = seg_bucket_value_sum[meta + b];
  __shared__ std::uint32_t scratch[256];
  const std::uint32_t live_prefix = block_scan_exclusive(live, scratch);
  seg_bucket_live_prefix[out + b] = live_prefix;
  if (b == kSegmentBuckets - 1)
    seg_bucket_live_prefix[out + kSegmentBuckets] = live_prefix + live;
  const std::uint32_t value_prefix = block_scan_exclusive(sum, scratch);
  seg_bucket_value_prefix[out + b] = value_prefix;
  if (b == kSegmentBuckets - 1)
    seg_bucket_value_prefix[out + kSegmentBuckets] = value_prefix + sum;
}

// Fenced search over a segment delta.

__device__ inline void seg_delta_window(const std::uint32_t *fence,
                                        std::uint32_t count, std::uint32_t key,
                                        std::uint32_t &lo, std::uint32_t &hi) {
  std::uint32_t j = 0;
#pragma unroll
  for (int t = 1; t < kDeltaFence; ++t)
    j += fence[t] <= key ? 1u : 0u;
  lo = (j * count) / kDeltaFence;
  hi = j + 1 == kDeltaFence ? count : ((j + 1) * count) / kDeltaFence;
}

__device__ inline std::uint32_t
seg_delta_lower_bound(const std::uint32_t *delta_keys,
                      const std::uint32_t *fence, std::uint32_t count,
                      std::uint32_t key) {
  if (count == 0)
    return 0;
  std::uint32_t lo, hi;
  seg_delta_window(fence, count, key, lo, hi);
  return lo + static_cast<std::uint32_t>(
                  lower_bound_u32(delta_keys + lo, hi - lo, key));
}

__device__ inline std::uint32_t
seg_delta_upper_bound(const std::uint32_t *delta_keys,
                      const std::uint32_t *fence, std::uint32_t count,
                      std::uint32_t key) {
  if (count == 0)
    return 0;
  std::uint32_t lo, hi;
  seg_delta_window(fence, count, key, lo, hi);
  return lo + static_cast<std::uint32_t>(
                  upper_bound_u32(delta_keys + lo, hi - lo, key));
}

__device__ inline bool
seg_delta_find(const std::uint32_t *delta_keys, const std::uint32_t *delta_values,
               const std::uint32_t *fence, std::uint32_t count,
               std::uint32_t key, std::uint32_t *out_value) {
  const std::uint32_t p = seg_delta_lower_bound(delta_keys, fence, count, key);
  if (p < count && delta_keys[p] == key) {
    if (out_value)
      *out_value = delta_values[p];
    return true;
  }
  return false;
}

// Bucket routing used by the direct delete path.

constexpr int kRadixRouteBits = 20;
constexpr std::size_t kRadixRouteSize = std::size_t{1} << kRadixRouteBits;
static_assert(kBucketSlots == 32, "warp bucket merge needs 32-slot buckets");

// Flat bucket maxima support radix routing.
__global__ void ds_build_flat_max_kernel(const std::uint32_t *dir_seg_id,
                                         const std::uint32_t *dir_boundary,
                                         std::size_t dir_count,
                                         const std::uint32_t *seg_bucket_max,
                                         std::uint32_t *flat_max) {
  const std::size_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= dir_count * kSegmentBuckets)
    return;
  const std::size_t ord = g / kSegmentBuckets;
  const std::size_t bucket = g % kSegmentBuckets;
  flat_max[g] =
      bucket + 1 == kSegmentBuckets
          ? dir_boundary[ord]
          : seg_bucket_max[static_cast<std::size_t>(dir_seg_id[ord]) *
                               kSegmentBuckets +
                           bucket];
}

// Map radix bins to flat bucket ranges.
__global__ void ds_build_radix_kernel(const std::uint32_t *sorted_max,
                                      std::size_t count,
                                      std::uint32_t *radix_first) {
  const std::size_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g > count)
    return;
  const std::size_t bin_hi =
      g == count ? kRadixRouteSize
                 : static_cast<std::size_t>(sorted_max[g] >>
                                            (32 - kRadixRouteBits));
  std::size_t bin_lo;
  if (g == 0) {
    bin_lo = 0;
    radix_first[0] = 0u;
  } else {
    bin_lo = static_cast<std::size_t>(sorted_max[g - 1] >>
                                      (32 - kRadixRouteBits)) +
             1;
  }
  for (std::size_t bin = bin_lo; bin <= bin_hi; ++bin)
    radix_first[bin] = static_cast<std::uint32_t>(g);
}

__global__ void ds_route_kernel(const std::uint32_t *keys, std::size_t n,
                                const std::uint32_t *radix_first,
                                const std::uint32_t *flat_max,
                                std::size_t total_buckets, std::uint32_t *dest,
                                std::uint32_t *cnt, std::uint32_t *dirty_list,
                                std::uint32_t *counters) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint32_t key = keys[i];
  const std::uint32_t bin = key >> (32 - kRadixRouteBits);
  std::size_t lo = radix_first[bin];
  std::size_t hi = radix_first[bin + 1];
  while (lo < hi) {
    const std::size_t mid = (lo + hi) >> 1;
    if (flat_max[mid] < key)
      lo = mid + 1;
    else
      hi = mid;
  }
  std::uint32_t g = static_cast<std::uint32_t>(lo);
  if (g >= total_buckets)
    g = static_cast<std::uint32_t>(total_buckets - 1);
  dest[i] = g;
  if (atomicAdd(cnt + g, 1u) == 0u)
    dirty_list[atomicAdd(counters, 1u)] = g;
}

__device__ inline std::uint32_t warp_add_u32(std::uint32_t x) {
#pragma unroll
  for (int offset = kBucketSlots / 2; offset > 0; offset >>= 1)
    x += __shfl_xor_sync(0xffffffffu, x, offset);
  return x;
}

// Sort one key/value pair per warp lane.
__device__ inline void warp_bitonic_sort_pair(std::uint32_t &key,
                                              std::uint32_t &val, int lane) {
#pragma unroll
  for (int k = 2; k <= kBucketSlots; k <<= 1) {
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
      const std::uint32_t partner_key = __shfl_xor_sync(0xffffffffu, key, j);
      const std::uint32_t partner_val = __shfl_xor_sync(0xffffffffu, val, j);
      const bool keep_min = ((lane & k) == 0) == ((lane & j) == 0);
      const bool take = keep_min ? partner_key < key : partner_key > key;
      if (take) {
        key = partner_key;
        val = partner_val;
      }
    }
  }
}





__global__ void ds_scatter_kernel(const std::uint32_t *keys,
                                  const std::uint32_t *values, std::size_t n,
                                  const std::uint32_t *dest,
                                  const std::uint32_t *base,
                                  std::uint32_t *cursor,
                                  std::uint32_t *staged_keys,
                                  std::uint32_t *staged_values) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint32_t g = dest[i];
  const std::uint32_t p = base[g] + atomicAdd(cursor + g, 1u);
  staged_keys[p] = keys[i];
  staged_values[p] = values[i];
}




// Remove one bucket's staged delete keys.
__global__ void ds_bucket_delete_kernel(
    const std::uint32_t *dirty_list, const std::uint32_t *counters,
    const std::uint32_t *cnt, const std::uint32_t *base,
    const std::uint32_t *staged_keys, const std::uint32_t *dir_seg_id,
    std::uint32_t *pool_keys, std::uint32_t *pool_values,
    std::uint8_t *pool_valid, std::uint32_t *seg_bucket_max,
    std::uint32_t *seg_bucket_live, std::uint32_t *seg_bucket_value_sum,
    std::uint32_t *ord_live_delta, std::uint32_t *ord_value_delta) {
  const std::size_t w =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
      kBucketSlots;
  if (w >= counters[0])
    return;
  const int lane = threadIdx.x & (kBucketSlots - 1);
  const std::uint32_t g = dirty_list[w];
  const std::uint32_t ord = g / kSegmentBuckets;
  const std::uint32_t bucket = g % kSegmentBuckets;
  const std::uint32_t seg = dir_seg_id[ord];
  const std::size_t meta =
      static_cast<std::size_t>(seg) * kSegmentBuckets + bucket;
  const std::size_t slot =
      static_cast<std::size_t>(seg) * kSegmentSlots +
      static_cast<std::size_t>(bucket) * kBucketSlots;
  const int live = static_cast<int>(seg_bucket_live[meta]);
  const std::uint32_t del_n = cnt[g];
  const std::uint32_t del0 = base[g];
  std::uint32_t key = kEmptyKey;
  std::uint32_t val = 0u;
  if (lane < live) {
    key = pool_keys[slot + lane];
    val = pool_values[slot + lane];
  }
  bool matched = false;
  for (std::uint32_t chunk = 0; chunk < del_n;
       chunk += static_cast<std::uint32_t>(kBucketSlots)) {
    const std::uint32_t idx = chunk + static_cast<std::uint32_t>(lane);
    const std::uint32_t del_key =
        idx < del_n ? staged_keys[del0 + idx] : kEmptyKey;
#pragma unroll
    for (int j = 0; j < kBucketSlots; ++j) {
      const std::uint32_t broadcast = __shfl_sync(0xffffffffu, del_key, j);
      matched |= (lane < live && key == broadcast);
    }
  }
  const bool survive = lane < live && !matched;
  const unsigned ballot = __ballot_sync(0xffffffffu, survive);
  const int new_live = __popc(ballot);
  const int removed = live - new_live;
  if (removed == 0)
    return;
  const std::uint32_t removed_sum =
      warp_add_u32(matched && lane < live ? val : 0u);
  const int pos = __popc(ballot & ((1u << lane) - 1u));
  if (survive) {
    pool_keys[slot + pos] = key;
    pool_values[slot + pos] = val;
  }
  if (lane >= new_live && lane < live) {
    pool_keys[slot + lane] = kEmptyKey;
    pool_values[slot + lane] = 0u;
    pool_valid[slot + lane] = 0u;
  }
  if (lane == 0) {
    seg_bucket_live[meta] = static_cast<std::uint32_t>(new_live);
    seg_bucket_value_sum[meta] -= removed_sum;
    atomicAdd(ord_live_delta + ord,
              0u - static_cast<std::uint32_t>(removed));
    atomicAdd(ord_value_delta + ord, 0u - removed_sum);
  }
  if (new_live > 0) {
    // Reduce the maximum across unordered survivors.
    std::uint32_t cand = survive ? key : 0u;
#pragma unroll
    for (int offset = kBucketSlots / 2; offset > 0; offset >>= 1) {
      const std::uint32_t other = __shfl_xor_sync(0xffffffffu, cand, offset);
      cand = other > cand ? other : cand;
    }
    if (lane == 0)
      seg_bucket_max[meta] = cand;
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
  // Buckets are stored sorted.
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

__global__ void seg_pull_apply_fast_kernel(
    const std::uint32_t *keys, const std::uint32_t *values, std::size_t n,
    const std::uint32_t *dir_boundary, std::size_t dir_count,
    const std::uint32_t *dir_seg_id, const std::uint32_t *dir_prefix,
    const std::uint32_t *dir_value_sum, std::uint32_t *dirty_ord,
    std::uint32_t *dirty_count, std::uint32_t *dirty_seg_id,
    std::uint32_t *dirty_old_boundary, std::uint32_t *dirty_old_live,
    std::uint32_t *dirty_old_value_sum, std::uint32_t *dirty_in_begin,
    std::uint32_t *dirty_in_end, std::uint32_t *slow,
    std::uint32_t *new_live, std::uint32_t *new_value_sum,
    std::uint32_t *slow_index, std::uint32_t *counters,
    std::uint32_t *pool_keys, std::uint32_t *pool_values,
    std::uint8_t *pool_valid, std::uint32_t *seg_bucket_max,
    std::uint32_t *seg_bucket_live,
    std::uint32_t *seg_bucket_value_sum) {
  namespace cg = cooperative_groups;
  constexpr int tile_size = 8;
  constexpr int tile_count = 256 / tile_size;
  const std::uint32_t ord = blockIdx.x;
  if (ord >= dir_count)
    return;
  const cg::thread_block block = cg::this_thread_block();
  const auto tile = cg::tiled_partition<tile_size>(block);
  const std::uint32_t tile_id = threadIdx.x / tile_size;
  const std::uint32_t lane = tile.thread_rank();
  __shared__ std::uint32_t s_begin;
  __shared__ std::uint32_t s_count;
  __shared__ std::uint32_t s_dirty;
  __shared__ std::uint32_t s_seg;
  __shared__ std::uint32_t s_slow;
  __shared__ std::uint32_t s_bucket_begin[kSegmentBuckets];
  __shared__ std::uint32_t s_bucket_end[kSegmentBuckets];
  __shared__ std::uint32_t s_merge_key[tile_count * kBucketSlots];
  __shared__ std::uint32_t s_merge_value[tile_count * kBucketSlots];
  __shared__ std::uint32_t s_tile_sum[tile_count];
  if (threadIdx.x == 0) {
    const std::size_t begin =
        ord == 0 ? 0 : upper_bound_u32(keys, n, dir_boundary[ord - 1]);
    const std::size_t end =
        ord + 1 == dir_count ? n : upper_bound_u32(keys, n, dir_boundary[ord]);
    s_begin = static_cast<std::uint32_t>(begin);
    s_count = static_cast<std::uint32_t>(end - begin);
    if (s_count != 0) {
      const std::uint32_t out = atomicAdd(counters, 1u);
      const std::uint32_t old_live = dir_prefix[ord + 1] - dir_prefix[ord];
      s_dirty = out;
      s_seg = dir_seg_id[ord];
      s_slow = s_count > GPULSMOPT_INPLACE_MAX_INCOMING ? 1u : 0u;
      dirty_ord[out] = ord;
      dirty_count[out] = s_count;
      dirty_seg_id[out] = s_seg;
      dirty_old_boundary[out] = dir_boundary[ord];
      dirty_old_live[out] = old_live;
      dirty_old_value_sum[out] = dir_value_sum[ord];
      dirty_in_begin[out] = s_begin;
      dirty_in_end[out] = static_cast<std::uint32_t>(end);
      slow[out] = s_slow;
      new_live[out] = old_live + s_count;
      new_value_sum[out] = dir_value_sum[ord];
    }
  }
  block.sync();
  if (s_count == 0)
    return;
  const std::uint32_t *base_keys = keys + s_begin;
  for (std::uint32_t bucket = tile_id; bucket < kSegmentBuckets;
       bucket += tile_count) {
    if (lane == 0) {
      const std::size_t meta =
          static_cast<std::size_t>(s_seg) * kSegmentBuckets + bucket;
      const std::uint32_t b =
          bucket == 0
              ? 0u
              : static_cast<std::uint32_t>(upper_bound_u32(
                    base_keys, s_count, seg_bucket_max[meta - 1]));
      const std::uint32_t e =
          bucket + 1 == kSegmentBuckets
              ? s_count
              : static_cast<std::uint32_t>(upper_bound_u32(
                    base_keys, s_count, seg_bucket_max[meta]));
      s_bucket_begin[bucket] = b;
      s_bucket_end[bucket] = e;
      if (seg_bucket_live[meta] + e - b > kBucketSlots)
        atomicExch(&s_slow, 1u);
    }
  }
  block.sync();
  if (threadIdx.x == 0) {
    slow[s_dirty] = s_slow;
    if (s_slow) {
      const std::uint32_t out = atomicAdd(counters + 1, 1u);
      slow_index[out] = s_dirty;
    }
  }
  block.sync();
  if (s_slow)
    return;
  std::uint32_t tile_insert_sum = 0;
  for (std::uint32_t wave = 0; wave < kSegmentBuckets / tile_count;
       ++wave) {
    const std::uint32_t bucket = wave * tile_count + tile_id;
    const std::uint32_t b = s_bucket_begin[bucket];
    const std::uint32_t e = s_bucket_end[bucket];
    const std::uint32_t incoming_count = e - b;
    if (incoming_count != 0) {
      const std::size_t meta =
          static_cast<std::size_t>(s_seg) * kSegmentBuckets + bucket;
      const std::size_t slot_base =
          static_cast<std::size_t>(s_seg) * kSegmentSlots +
          static_cast<std::size_t>(bucket) * kBucketSlots;
      const std::uint32_t live = seg_bucket_live[meta];
      const std::uint32_t output_count = live + incoming_count;
      const std::uint32_t incoming_begin = s_begin + b;
      const std::size_t merge_base = tile_id * kBucketSlots;
      // Rank the entry within its bucket.
      for (std::uint32_t old = lane; old < live; old += tile_size) {
        const std::uint32_t key = pool_keys[slot_base + old];
        std::uint32_t self_rank = 0;
        for (std::uint32_t j = 0; j < live; ++j)
          self_rank += pool_keys[slot_base + j] < key ? 1u : 0u;
        const std::uint32_t rank =
            self_rank + static_cast<std::uint32_t>(lower_bound_u32(
                            keys + incoming_begin, incoming_count, key));
        s_merge_key[merge_base + rank] = key;
        s_merge_value[merge_base + rank] = pool_values[slot_base + old];
      }
      std::uint32_t inserted_sum = 0;
      for (std::uint32_t incoming = lane; incoming < incoming_count;
           incoming += tile_size) {
        const std::uint32_t key = keys[incoming_begin + incoming];
        std::uint32_t below = 0;
        for (std::uint32_t j = 0; j < live; ++j)
          below += pool_keys[slot_base + j] < key ? 1u : 0u;
        const std::uint32_t rank = incoming + below;
        const std::uint32_t value = values[incoming_begin + incoming];
        s_merge_key[merge_base + rank] = key;
        s_merge_value[merge_base + rank] = value;
        inserted_sum += value;
      }
      tile.sync();
      std::uint32_t sum = 0;
      for (std::uint32_t out = lane; out < output_count; out += tile_size) {
        const std::uint32_t value = s_merge_value[merge_base + out];
        pool_keys[slot_base + out] = s_merge_key[merge_base + out];
        pool_values[slot_base + out] = value;
        pool_valid[slot_base + out] = 1u;
        sum += value;
      }
      for (std::uint32_t out = output_count + lane; out < kBucketSlots;
           out += tile_size) {
        pool_keys[slot_base + out] = kEmptyKey;
        pool_values[slot_base + out] = 0u;
        pool_valid[slot_base + out] = 0u;
      }
      for (int offset = tile_size / 2; offset > 0; offset /= 2) {
        sum += tile.shfl_down(sum, offset);
        inserted_sum += tile.shfl_down(inserted_sum, offset);
      }
      if (lane == 0) {
        seg_bucket_live[meta] = output_count;
        seg_bucket_max[meta] = s_merge_key[merge_base + output_count - 1];
        seg_bucket_value_sum[meta] = sum;
        tile_insert_sum += inserted_sum;
      }
    }
    block.sync();
  }
  if (lane == 0)
    s_tile_sum[tile_id] = tile_insert_sum;
  block.sync();
  if (threadIdx.x == 0) {
    std::uint32_t inserted_sum = 0;
    for (int tile_index = 0; tile_index < tile_count; ++tile_index)
      inserted_sum += s_tile_sum[tile_index];
    new_value_sum[s_dirty] = dir_value_sum[ord] + inserted_sum;
  }
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
  const std::uint32_t del_count = dirty_bucket_count[i];
  std::uint32_t out = 0;
  std::uint32_t sum = 0;
  std::uint32_t removed = 0;
  std::uint32_t removed_sum = 0;
  std::uint32_t max_key = 0;
  // Match the bucket against sorted deletes.
  for (std::uint32_t j = 0; j < live; ++j) {
    const std::uint32_t key = pool_keys[base + j];
    const std::uint32_t value = pool_values[base + j];
    const std::size_t p =
        lower_bound_u32(del_keys + del_begin, del_count, key);
    if (p < del_count && del_keys[del_begin + p] == key) {
      ++removed;
      removed_sum += value;
      continue;
    }
    pool_keys[base + out] = key;
    pool_values[base + out] = value;
    if (key > max_key)
      max_key = key;
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
    seg_bucket_max[meta] = max_key;
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
                                        const std::uint32_t *dirty_seg_id,
                                        std::size_t dirty_count,
                                        std::uint32_t target_live,
                                        std::uint32_t *dirty_k) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dirty_count)
    return;
  const std::uint32_t live = dirty_live[i];
  // Split before buckets lose all headroom.
  const std::uint32_t k = live <= seg_split_trigger(dirty_seg_id[i])
                              ? 1u
                              : ceil_div_u32(live, target_live);
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

// Probe the larger of buckets/delta first; both are sorted.
__device__ inline bool seg_point_lookup(
    std::uint32_t seg_id, std::uint32_t key, const std::uint32_t *pool_keys,
    const std::uint32_t *pool_values, const std::uint8_t *pool_valid,
    const std::uint32_t *seg_bucket_max, const std::uint32_t *seg_bucket_live,
    const std::uint32_t *seg_bucket_live_prefix,
    const std::uint32_t *delta_keys, const std::uint32_t *delta_values,
    const std::uint8_t *delta_active, const std::uint32_t *delta_fence,
    const std::uint32_t *delta_count, std::uint32_t *out_value) {
  const std::uint32_t dc = delta_count[seg_id];
  const std::size_t dbase = seg_delta_base(seg_id, delta_active);
  const std::uint32_t *fence =
      delta_fence + static_cast<std::size_t>(seg_id) * kDeltaFence;
  const std::uint32_t bucket_total =
      seg_bucket_live_prefix[static_cast<std::size_t>(seg_id) *
                                 (kSegmentBuckets + 1) +
                             kSegmentBuckets];
  const bool delta_first = dc > bucket_total;
  if (delta_first && dc > 0 &&
      seg_delta_find(delta_keys + dbase, delta_values + dbase, fence, dc, key,
                     out_value))
    return true;
  const std::size_t meta_base =
      static_cast<std::size_t>(seg_id) * kSegmentBuckets;
  const std::size_t bucket =
      lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, key);
  if (bucket < kSegmentBuckets) {
    const std::size_t start =
        static_cast<std::size_t>(seg_id) * kSegmentSlots +
        bucket * kBucketSlots;
    const std::uint32_t live = seg_bucket_live[meta_base + bucket];
    const std::size_t p = lower_bound_u32(pool_keys + start, live, key);
    if (p < live && pool_keys[start + p] == key) {
      if (out_value)
        *out_value = pool_values[start + p];
      return true;
    }
  }
  if (!delta_first && dc > 0 &&
      seg_delta_find(delta_keys + dbase, delta_values + dbase, fence, dc, key,
                     out_value))
    return true;
  return false;
}

__device__ inline std::uint32_t seg_delta_range_count(
    std::uint32_t seg_id, std::uint32_t lo, std::uint32_t hi,
    const std::uint32_t *delta_keys, const std::uint8_t *delta_active,
    const std::uint32_t *delta_fence, const std::uint32_t *delta_count) {
  const std::uint32_t dc = delta_count[seg_id];
  if (dc == 0)
    return 0;
  const std::uint32_t *dk = delta_keys + seg_delta_base(seg_id, delta_active);
  const std::uint32_t *fence =
      delta_fence + static_cast<std::size_t>(seg_id) * kDeltaFence;
  return seg_delta_upper_bound(dk, fence, dc, hi) -
         seg_delta_lower_bound(dk, fence, dc, lo);
}

__device__ inline std::uint32_t seg_delta_range_sum(
    std::uint32_t seg_id, std::uint32_t lo, std::uint32_t hi,
    const std::uint32_t *delta_keys, const std::uint32_t *delta_values,
    const std::uint8_t *delta_active,
    const std::uint32_t *delta_fence, const std::uint32_t *delta_count,
    const std::uint32_t *delta_prefix) {
  const std::uint32_t dc = delta_count[seg_id];
  if (dc == 0)
    return 0;
  const std::uint32_t *dk = delta_keys + seg_delta_base(seg_id, delta_active);
  const std::uint32_t *fence =
      delta_fence + static_cast<std::size_t>(seg_id) * kDeltaFence;
  const std::uint32_t *prefix =
      delta_prefix + static_cast<std::size_t>(seg_id) * (kDeltaCap + 1);
  const std::uint32_t *dv =
      delta_values + seg_delta_base(seg_id, delta_active);
  auto prefix_at = [&](std::uint32_t pos) {
    const std::uint32_t base =
        (pos / kDeltaPrefixStride) * kDeltaPrefixStride;
    std::uint32_t sum = prefix[base];
    for (std::uint32_t p = base; p < pos; ++p)
      sum += dv[p];
    return sum;
  };
  return prefix_at(seg_delta_upper_bound(dk, fence, dc, hi)) -
         prefix_at(seg_delta_lower_bound(dk, fence, dc, lo));
}

__device__ inline std::uint32_t seg_range_count_one(
    std::uint32_t seg_id, std::uint32_t lo, std::uint32_t hi,
    const std::uint32_t *pool_keys, const std::uint8_t *pool_valid,
    const std::uint32_t *seg_bucket_max, const std::uint32_t *seg_bucket_live,
    const std::uint32_t *seg_bucket_live_prefix, const std::uint32_t *delta_keys,
    const std::uint8_t *delta_active, const std::uint32_t *delta_fence,
    const std::uint32_t *delta_count) {
  const std::uint32_t from_delta =
      seg_delta_range_count(seg_id, lo, hi, delta_keys, delta_active,
                            delta_fence, delta_count);
  const std::size_t meta_base =
      static_cast<std::size_t>(seg_id) * kSegmentBuckets;
  const std::size_t slot_base =
      static_cast<std::size_t>(seg_id) * kSegmentSlots;
  const std::size_t first =
      lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, lo);
  if (first >= kSegmentBuckets)
    return from_delta;
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
    return count + from_delta;
  }
  scan_bucket(first);
  const std::size_t full_end = last < kSegmentBuckets ? last : kSegmentBuckets;
  if (last < kSegmentBuckets)
    scan_bucket(last);
  if (full_end > first + 1) {
    const std::size_t pbase =
        static_cast<std::size_t>(seg_id) * (kSegmentBuckets + 1);
    count += seg_bucket_live_prefix[pbase + full_end] -
             seg_bucket_live_prefix[pbase + first + 1];
  }
  return count + from_delta;
}

__device__ inline std::uint32_t seg_range_sum_one(
    std::uint32_t seg_id, std::uint32_t lo, std::uint32_t hi,
    const std::uint32_t *pool_keys, const std::uint32_t *pool_values,
    const std::uint8_t *pool_valid, const std::uint32_t *seg_bucket_max,
    const std::uint32_t *seg_bucket_live,
    const std::uint32_t *seg_bucket_value_prefix,
    const std::uint32_t *delta_keys, const std::uint32_t *delta_values,
    const std::uint8_t *delta_active,
    const std::uint32_t *delta_fence, const std::uint32_t *delta_count,
    const std::uint32_t *delta_prefix) {
  const std::uint32_t from_delta =
      seg_delta_range_sum(seg_id, lo, hi, delta_keys, delta_values, delta_active,
                          delta_fence, delta_count, delta_prefix);
  const std::size_t meta_base =
      static_cast<std::size_t>(seg_id) * kSegmentBuckets;
  const std::size_t slot_base =
      static_cast<std::size_t>(seg_id) * kSegmentSlots;
  const std::size_t first =
      lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, lo);
  if (first >= kSegmentBuckets)
    return from_delta;
  const std::size_t last =
      lower_bound_u32(seg_bucket_max + meta_base, kSegmentBuckets, hi);
  std::uint32_t sum = 0;
  auto scan_bucket = [&](std::size_t b) {
    const std::size_t start = slot_base + b * kBucketSlots;
    const std::uint32_t live = seg_bucket_live[meta_base + b];
    const std::size_t lb = lower_bound_u32(pool_keys + start, live, lo);
    const std::size_t ub = upper_bound_u32(pool_keys + start, live, hi);
    for (std::size_t p = lb; p < ub; ++p)
      sum += pool_values[start + p];
  };
  if (first == last) {
    scan_bucket(first);
    return sum + from_delta;
  }
  scan_bucket(first);
  const std::size_t full_end = last < kSegmentBuckets ? last : kSegmentBuckets;
  if (last < kSegmentBuckets)
    scan_bucket(last);
  if (full_end > first + 1) {
    const std::size_t pbase =
        static_cast<std::size_t>(seg_id) * (kSegmentBuckets + 1);
    sum += seg_bucket_value_prefix[pbase + full_end] -
           seg_bucket_value_prefix[pbase + first + 1];
  }
  return sum + from_delta;
}

__device__ inline std::uint32_t seg_sheet_range_sum(
    std::uint32_t lo, std::uint32_t hi, const std::uint32_t *dir_radix_first,
    const std::uint32_t *dir_boundary,
    const std::uint32_t *dir_seg_id, const std::uint32_t *dir_value_prefix,
    std::size_t dir_count, const std::uint32_t *pool_keys,
    const std::uint32_t *pool_values, const std::uint8_t *pool_valid,
    const std::uint32_t *seg_bucket_max, const std::uint32_t *seg_bucket_live,
    const std::uint32_t *seg_bucket_value_prefix,
    const std::uint32_t *delta_keys, const std::uint32_t *delta_values,
    const std::uint8_t *delta_active,
    const std::uint32_t *delta_fence,
    const std::uint32_t *delta_count, const std::uint32_t *delta_prefix) {
  std::size_t pl = dir_route(lo, dir_radix_first, dir_boundary, dir_count);
  if (pl >= dir_count)
    return 0;
  std::size_t pr = dir_route(hi, dir_radix_first, dir_boundary, dir_count);
  if (pr >= dir_count)
    pr = dir_count - 1;
  if (pl == pr) {
    return seg_range_sum_one(dir_seg_id[pl], lo, hi, pool_keys, pool_values,
                             pool_valid, seg_bucket_max, seg_bucket_live,
                             seg_bucket_value_prefix, delta_keys, delta_values,
                             delta_active, delta_fence, delta_count,
                             delta_prefix);
  }
  std::uint32_t sum =
      seg_range_sum_one(dir_seg_id[pl], lo, hi, pool_keys, pool_values,
                        pool_valid, seg_bucket_max, seg_bucket_live,
                        seg_bucket_value_prefix, delta_keys, delta_values,
                        delta_active, delta_fence, delta_count, delta_prefix) +
      seg_range_sum_one(dir_seg_id[pr], lo, hi, pool_keys, pool_values,
                        pool_valid, seg_bucket_max, seg_bucket_live,
                        seg_bucket_value_prefix, delta_keys, delta_values,
                        delta_active, delta_fence, delta_count, delta_prefix);
  if (pr > pl + 1)
    sum += dir_value_prefix[pr] - dir_value_prefix[pl + 1];
  return sum;
}

__device__ inline std::uint32_t seg_sheet_range_count(
    std::uint32_t lo, std::uint32_t hi, const std::uint32_t *dir_radix_first,
    const std::uint32_t *dir_boundary,
    const std::uint32_t *dir_seg_id, const std::uint32_t *dir_prefix,
    std::size_t dir_count, const std::uint32_t *pool_keys,
    const std::uint8_t *pool_valid, const std::uint32_t *seg_bucket_max,
    const std::uint32_t *seg_bucket_live,
    const std::uint32_t *seg_bucket_live_prefix, const std::uint32_t *delta_keys,
    const std::uint8_t *delta_active,
    const std::uint32_t *delta_fence, const std::uint32_t *delta_count) {
  std::size_t pl = dir_route(lo, dir_radix_first, dir_boundary, dir_count);
  if (pl >= dir_count)
    return 0;
  std::size_t pr = dir_route(hi, dir_radix_first, dir_boundary, dir_count);
  if (pr >= dir_count)
    pr = dir_count - 1;
  if (pl == pr) {
    return seg_range_count_one(dir_seg_id[pl], lo, hi, pool_keys, pool_valid,
                               seg_bucket_max, seg_bucket_live,
                               seg_bucket_live_prefix, delta_keys,
                               delta_active, delta_fence, delta_count);
  }
  std::uint32_t total =
      seg_range_count_one(dir_seg_id[pl], lo, hi, pool_keys, pool_valid,
                          seg_bucket_max, seg_bucket_live,
                          seg_bucket_live_prefix, delta_keys, delta_active,
                          delta_fence, delta_count) +
      seg_range_count_one(dir_seg_id[pr], lo, hi, pool_keys, pool_valid,
                          seg_bucket_max, seg_bucket_live,
                          seg_bucket_live_prefix, delta_keys, delta_active,
                          delta_fence, delta_count);
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
    // Buckets are stored sorted; copy through.
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

// Gather and repack consolidating segments.

constexpr int kDsGatherChunks = 8;
constexpr int kDsAbsorbChunks = 8;

__global__ void ds_gather_old_kernel(
    const std::uint32_t *seg_ids, const std::uint32_t *old_offset,
    std::size_t seg_count, const std::uint32_t *pool_keys,
    const std::uint32_t *pool_values, const std::uint32_t *seg_bucket_live,
    std::uint32_t *old_keys, std::uint32_t *old_values) {
  const std::size_t i = blockIdx.x / kDsGatherChunks;
  if (i >= seg_count)
    return;
  const std::uint32_t chunk = blockIdx.x % kDsGatherChunks;
  const std::uint32_t seg = seg_ids[i];
  const std::size_t meta = static_cast<std::size_t>(seg) * kSegmentBuckets;
  const std::size_t slot_base = static_cast<std::size_t>(seg) * kSegmentSlots;
  __shared__ std::uint32_t s_live[kSegmentBuckets];
  __shared__ std::uint32_t s_prefix[kSegmentBuckets];
  for (int b = threadIdx.x; b < kSegmentBuckets; b += blockDim.x) {
    // Counts above 32 use the overflow list.
    const std::uint32_t live = seg_bucket_live[meta + b];
    s_live[b] = live < static_cast<std::uint32_t>(kBucketSlots)
                    ? live
                    : static_cast<std::uint32_t>(kBucketSlots);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::uint32_t acc = 0;
    for (int b = 0; b < kSegmentBuckets; ++b) {
      s_prefix[b] = acc;
      acc += s_live[b];
    }
  }
  __syncthreads();
  // Buckets are stored sorted; copy them through unchanged.
  constexpr int warps_per_block = 256 / kBucketSlots;
  constexpr std::uint32_t chunk_buckets = kSegmentBuckets / kDsGatherChunks;
  const int warp = threadIdx.x / kBucketSlots;
  const int lane = threadIdx.x & (kBucketSlots - 1);
  const std::uint32_t out_base = old_offset[i];
  for (std::uint32_t b = chunk * chunk_buckets + warp;
       b < (chunk + 1) * chunk_buckets; b += warps_per_block) {
    const std::uint32_t live = s_live[b];
    if (lane < live) {
      const std::size_t src =
          slot_base + static_cast<std::size_t>(b) * kBucketSlots;
      const std::size_t dst =
          static_cast<std::size_t>(out_base) + s_prefix[b] + lane;
      old_keys[dst] = pool_keys[src + lane];
      old_values[dst] = pool_values[src + lane];
    }
  }
}

__device__ inline void ds_merge_pick(const std::uint32_t *inc_k,
                                     const std::uint32_t *inc_v,
                                     std::uint32_t inc_n,
                                     const std::uint32_t *old_k,
                                     const std::uint32_t *old_v,
                                     std::uint32_t old_n, std::uint32_t rank,
                                     std::uint32_t &key, std::uint32_t &value) {
  const std::uint32_t p = seg_merge_partition(inc_k, inc_n, old_k, old_n, rank);
  const std::uint32_t q = rank - p;
  if (p < inc_n && (q >= old_n || inc_k[p] <= old_k[q])) {
    key = inc_k[p];
    value = inc_v[p];
  } else {
    key = old_k[q];
    value = old_v[q];
  }
}

__device__ inline std::uint32_t
block_scan_exclusive(std::uint32_t value, std::uint32_t *scratch) {
  const unsigned mask = 0xffffffffu;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5;
  std::uint32_t inclusive = value;
#pragma unroll
  for (std::uint32_t offset = 1; offset < 32; offset <<= 1) {
    const std::uint32_t other =
        __shfl_up_sync(mask, inclusive, static_cast<int>(offset));
    if (lane >= offset)
      inclusive += other;
  }
  if (lane == 31u)
    scratch[warp] = inclusive;
  __syncthreads();
  if (warp == 0u) {
    std::uint32_t warp_sum = lane < (blockDim.x >> 5) ? scratch[lane] : 0u;
#pragma unroll
    for (std::uint32_t offset = 1; offset < 32; offset <<= 1) {
      const std::uint32_t other =
          __shfl_up_sync(mask, warp_sum, static_cast<int>(offset));
      if (lane >= offset)
        warp_sum += other;
    }
    if (lane < (blockDim.x >> 5))
      scratch[lane] = warp_sum;
  }
  __syncthreads();
  const std::uint32_t warp_prefix = warp == 0u ? 0u : scratch[warp - 1];
  const std::uint32_t result = warp_prefix + inclusive - value;
  __syncthreads();
  return result;
}

__global__ void delta_slice_kernel(
    const std::uint32_t *batch_keys, std::size_t n,
    const std::uint32_t *dir_boundary, std::size_t dir_count,
    std::uint32_t *cuts) {
  const std::size_t ord = blockIdx.x * blockDim.x + threadIdx.x;
  if (ord == 0)
    cuts[0] = 0u;
  if (ord >= dir_count)
    return;
  cuts[ord + 1] =
      ord + 1 == dir_count
          ? static_cast<std::uint32_t>(n)
          : static_cast<std::uint32_t>(
                upper_bound_u32(batch_keys, n, dir_boundary[ord]));
}

__global__ void delta_append_kernel(
    const std::uint32_t *batch_keys, const std::uint32_t *batch_values,
    const std::uint32_t *cuts, const std::uint32_t *dir_seg_id,
    std::size_t dir_count, std::uint32_t *delta_keys,
    std::uint32_t *delta_values, std::uint8_t *delta_active,
    std::uint32_t *delta_count, std::uint32_t *delta_fence,
    std::uint32_t *delta_prefix, std::uint32_t *dir_live,
    std::uint32_t *dir_value_sum,
    std::uint32_t *absorb_ord, std::uint32_t *absorb_begin,
    std::uint32_t *absorb_end, std::uint32_t *split_ord,
    std::uint32_t *split_begin, std::uint32_t *split_end,
    std::uint32_t *split_dcount,
    std::uint32_t *counters) {
  const std::size_t ord = blockIdx.x;
  if (ord >= dir_count)
    return;
  __shared__ std::uint32_t s_scan[256];
  __shared__ std::uint32_t s_class;
  const std::uint32_t begin = cuts[ord];
  const std::uint32_t end = cuts[ord + 1];
  if (threadIdx.x == 0) {
    std::uint32_t cls = 0u;
    if (end > begin) {
      const std::uint32_t seg = dir_seg_id[ord];
      const std::uint32_t dc = delta_count[seg];
      if (dc + (end - begin) <= seg_delta_cap(seg)) {
        cls = 1u;
      } else if (dir_live[ord] + (end - begin) <=
                 seg_split_trigger(seg)) {
        cls = 2u;
        const std::uint32_t i = atomicAdd(counters, 1u);
        absorb_ord[i] = static_cast<std::uint32_t>(ord);
        absorb_begin[i] = begin;
        absorb_end[i] = end;
      } else {
        cls = 3u;
        const std::uint32_t i = atomicAdd(counters + 1, 1u);
        split_ord[i] = static_cast<std::uint32_t>(ord);
        split_begin[i] = begin;
        split_end[i] = end;
        split_dcount[i] = dc;
      }
    }
    s_class = cls;
  }
  __syncthreads();
  if (s_class != 1u)
    return;
  const std::uint32_t inc_n = end - begin;
  const std::uint32_t seg = dir_seg_id[ord];
  const std::uint32_t act = delta_active[seg];
  const std::uint32_t old_n = delta_count[seg];
  const std::uint32_t total = old_n + inc_n;
  const std::size_t src_base =
      (2u * static_cast<std::size_t>(seg) + act) * kDeltaCap;
  const std::size_t dst_base =
      (2u * static_cast<std::size_t>(seg) + (act ^ 1u)) * kDeltaCap;
  const std::uint32_t *sk = delta_keys + src_base;
  const std::uint32_t *sv = delta_values + src_base;
  std::uint32_t *dk = delta_keys + dst_base;
  std::uint32_t *dv = delta_values + dst_base;
  std::uint32_t *prefix =
      delta_prefix + static_cast<std::size_t>(seg) * (kDeltaCap + 1);
  const std::uint32_t old_sum = prefix[old_n];
  const std::uint32_t chunk = ceil_div_u32(total, blockDim.x);
  const std::uint32_t r0 = threadIdx.x * chunk;
  const std::uint32_t r1 = r0 + chunk < total ? r0 + chunk : total;
  std::uint32_t vals[kDeltaLocal];
  std::uint32_t local_sum = 0u;
  if (r0 < total) {
    std::uint32_t incoming =
        seg_merge_partition(batch_keys + begin, inc_n, sk, old_n, r0);
    std::uint32_t previous = r0 - incoming;
    for (std::uint32_t rank = r0; rank < r1; ++rank) {
      std::uint32_t key;
      std::uint32_t value;
      if (incoming < inc_n &&
          (previous >= old_n ||
           batch_keys[begin + incoming] <= sk[previous])) {
        key = batch_keys[begin + incoming];
        value = batch_values[begin + incoming++];
      } else {
        key = sk[previous];
        value = sv[previous++];
      }
      dk[rank] = key;
      dv[rank] = value;
      vals[rank - r0] = value;
      local_sum += value;
    }
  }
  const std::uint32_t tile_prefix =
      block_scan_exclusive(local_sum, s_scan);
  std::uint32_t run = tile_prefix;
  for (std::uint32_t rank = r0; rank < r1; ++rank) {
    run += vals[rank - r0];
    if ((rank + 1) % kDeltaPrefixStride == 0 || rank + 1 == total)
      prefix[rank + 1] = run;
  }
  __syncthreads();
  if (threadIdx.x < kDeltaFence)
    delta_fence[static_cast<std::size_t>(seg) * kDeltaFence + threadIdx.x] =
        dk[(threadIdx.x * total) / kDeltaFence];
  if (threadIdx.x == 0) {
    prefix[0] = 0u;
    delta_count[seg] = total;
    delta_active[seg] = static_cast<std::uint8_t>(act ^ 1u);
    dir_live[ord] += inc_n;
    dir_value_sum[ord] += prefix[total] - old_sum;
  }
}

// Merge a full delta into consolidation scratch.
__global__ void delta_inc_merge_kernel(
    const std::uint32_t *cons_seg, const std::uint32_t *cons_begin,
    const std::uint32_t *cons_end, const std::uint32_t *cons_out_off,
    std::size_t cons_count, const std::uint32_t *batch_keys,
    const std::uint32_t *batch_values, const std::uint32_t *delta_keys,
    const std::uint32_t *delta_values, const std::uint8_t *delta_active,
    const std::uint32_t *delta_count,
    std::uint32_t *out_keys, std::uint32_t *out_values) {
  const std::size_t i = blockIdx.x;
  if (i >= cons_count)
    return;
  const std::uint32_t seg = cons_seg[i];
  const std::uint32_t b = cons_begin[i];
  const std::uint32_t inc_n = cons_end[i] - b;
  const std::uint32_t old_n = delta_count[seg];
  const std::uint32_t total = old_n + inc_n;
  const std::uint32_t off = cons_out_off[i];
  const std::size_t dbase = seg_delta_base(seg, delta_active);
  const std::uint32_t *dk = delta_keys + dbase;
  const std::uint32_t *dv = delta_values + dbase;
  const std::uint32_t chunk = ceil_div_u32(total, blockDim.x);
  const std::uint32_t r0 = threadIdx.x * chunk;
  const std::uint32_t r1 = r0 + chunk < total ? r0 + chunk : total;
  if (r0 >= total)
    return;
  // One merge-path partition per tile, then a serial merge.
  std::uint32_t p = seg_merge_partition(batch_keys + b, inc_n, dk, old_n, r0);
  std::uint32_t q = r0 - p;
  for (std::uint32_t r = r0; r < r1; ++r) {
    std::uint32_t key, val;
    if (p < inc_n && (q >= old_n || batch_keys[b + p] <= dk[q])) {
      key = batch_keys[b + p];
      val = batch_values[b + p];
      ++p;
    } else {
      key = dk[q];
      val = dv[q];
      ++q;
    }
    out_keys[off + r] = key;
    out_values[off + r] = val;
  }
}

__global__ void delta_reset_kernel(const std::uint32_t *seg_list,
                                   std::size_t count,
                                   std::uint8_t *delta_active,
                                   std::uint32_t *delta_count,
                                   std::uint32_t *delta_prefix) {
  const std::size_t i = blockIdx.x;
  if (i >= count)
    return;
  const std::uint32_t seg = seg_list[i];
  if (threadIdx.x == 0) {
    delta_active[seg] = 0u;
    delta_count[seg] = 0u;
    delta_prefix[static_cast<std::size_t>(seg) * (kDeltaCap + 1)] = 0u;
  }
}

__global__ void delta_clear_dead_kernel(
    const std::uint32_t *dir_seg_id, std::size_t dir_count,
    const std::uint8_t *delta_active, const std::uint32_t *delta_count,
    std::uint8_t *delta_dead) {
  const std::size_t ord = blockIdx.x;
  if (ord >= dir_count)
    return;
  const std::uint32_t seg = dir_seg_id[ord];
  const std::uint32_t count = delta_count[seg];
  const std::size_t base = seg_delta_base(seg, delta_active);
  for (std::uint32_t p = threadIdx.x; p < count; p += blockDim.x)
    delta_dead[base + p] = 0u;
}

// Mark delta entries removed by a large delete.
__global__ void delta_mark_deleted_kernel(
    const std::uint32_t *keys, std::size_t n, const std::uint32_t *dest,
    std::uint32_t ord_divisor, const std::uint32_t *dir_seg_id,
    const std::uint32_t *delta_keys,
    const std::uint32_t *delta_values, const std::uint8_t *delta_active,
    const std::uint32_t *delta_fence,
    const std::uint32_t *delta_count, std::uint8_t *delta_dead,
    std::uint32_t *ord_flag, std::uint32_t *touched_list,
    std::uint32_t *ord_live_delta, std::uint32_t *ord_value_delta,
    std::uint32_t *counters) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint32_t ord = dest[i] / ord_divisor;
  const std::uint32_t seg = dir_seg_id[ord];
  const std::uint32_t cnt = delta_count[seg];
  if (cnt == 0)
    return;
  const std::size_t base = seg_delta_base(seg, delta_active);
  const std::uint32_t key = keys[i];
  const std::uint32_t p = seg_delta_lower_bound(
      delta_keys + base, delta_fence + static_cast<std::size_t>(seg) * kDeltaFence,
      cnt, key);
  if (p < cnt && delta_keys[base + p] == key) {
    delta_dead[base + p] = 1u;
    atomicAdd(ord_live_delta + ord, 0u - 1u);
    atomicAdd(ord_value_delta + ord, 0u - delta_values[base + p]);
    if (atomicExch(ord_flag + ord, 1u) == 0u)
      touched_list[atomicAdd(counters + 1, 1u)] = ord;
  }
}

// Compact one touched delta into its inactive buffer.
__global__ void delta_compact_kernel(
    const std::uint32_t *touched_list, const std::uint32_t *counters,
    const std::uint32_t *dir_seg_id, std::uint32_t *delta_keys,
    std::uint32_t *delta_values, std::uint8_t *delta_active,
    std::uint32_t *delta_count,
    std::uint32_t *delta_fence, std::uint32_t *delta_prefix,
    std::uint8_t *delta_dead) {
  const std::uint32_t touched = counters[1];
  if (blockIdx.x >= touched)
    return;
  const std::uint32_t seg = dir_seg_id[touched_list[blockIdx.x]];
  const std::uint32_t act = delta_active[seg];
  const std::size_t src_base =
      (2u * static_cast<std::size_t>(seg) + act) * kDeltaCap;
  const std::size_t dst_base =
      (2u * static_cast<std::size_t>(seg) + (act ^ 1u)) * kDeltaCap;
  const std::uint32_t old_n = delta_count[seg];
  __shared__ std::uint32_t s_scan[256];
  __shared__ std::uint32_t s_total;
  const std::uint32_t chunk = ceil_div_u32(old_n, blockDim.x);
  const std::uint32_t r0 = threadIdx.x * chunk;
  const std::uint32_t r1 = r0 + chunk < old_n ? r0 + chunk : old_n;
  std::uint32_t keep = 0;
  for (std::uint32_t r = r0; r < r1; ++r)
    keep += delta_dead[src_base + r] ? 0u : 1u;
  const std::uint32_t pos0 = block_scan_exclusive(keep, s_scan);
  if (threadIdx.x == blockDim.x - 1)
    s_total = pos0 + keep;
  std::uint32_t vals[kDeltaLocal];
  std::uint32_t w = 0;
  std::uint32_t vsum = 0;
  for (std::uint32_t r = r0; r < r1; ++r) {
    if (!delta_dead[src_base + r]) {
      const std::uint32_t v = delta_values[src_base + r];
      delta_keys[dst_base + pos0 + w] = delta_keys[src_base + r];
      delta_values[dst_base + pos0 + w] = v;
      vals[w] = v;
      vsum += v;
      ++w;
    }
  }
  const std::uint32_t vbase = block_scan_exclusive(vsum, s_scan);
  std::uint32_t *prefix =
      delta_prefix + static_cast<std::size_t>(seg) * (kDeltaCap + 1);
  std::uint32_t run = vbase;
  for (std::uint32_t j = 0; j < w; ++j) {
    run += vals[j];
    const std::uint32_t pos = pos0 + j + 1;
    if (pos % kDeltaPrefixStride == 0 || pos == s_total)
      prefix[pos] = run;
  }
  __syncthreads();
  const std::uint32_t total = s_total;
  if (total > 0 && threadIdx.x < kDeltaFence)
    delta_fence[static_cast<std::size_t>(seg) * kDeltaFence + threadIdx.x] =
        delta_keys[dst_base + (threadIdx.x * total) / kDeltaFence];
  if (threadIdx.x == 0) {
    prefix[0] = 0u;
    delta_count[seg] = total;
    delta_active[seg] = static_cast<std::uint8_t>(act ^ 1u);
  }
}

__global__ void delta_gather_counts_kernel(const std::uint32_t *dir_seg_id,
                                           std::size_t dir_count,
                                           const std::uint32_t *delta_count,
                                           std::uint32_t *out) {
  const std::size_t ord = blockIdx.x * blockDim.x + threadIdx.x;
  if (ord < dir_count)
    out[ord] = delta_count[dir_seg_id[ord]];
}

__global__ void ds_absorb_write_kernel(
    const std::uint32_t *seg_ids, const std::uint32_t *boundaries,
    const std::uint32_t *totals, const std::uint32_t *old_offset,
    const std::uint32_t *in_begin, const std::uint32_t *in_end,
    std::size_t absorb_count, const std::uint32_t *inc_keys,
    const std::uint32_t *inc_values, const std::uint32_t *old_keys,
    const std::uint32_t *old_values, std::uint32_t *pool_keys,
    std::uint32_t *pool_values, std::uint8_t *pool_valid,
    std::uint32_t *seg_bucket_max, std::uint32_t *seg_bucket_live,
    std::uint32_t *seg_bucket_value_sum, std::uint32_t *out_value_sum) {
  const std::size_t i = blockIdx.x / kDsAbsorbChunks;
  if (i >= absorb_count)
    return;
  const std::uint32_t chunk = blockIdx.x % kDsAbsorbChunks;
  constexpr std::uint32_t chunk_buckets = kSegmentBuckets / kDsAbsorbChunks;
  const std::uint32_t seg = seg_ids[i];
  const std::uint32_t boundary = boundaries[i];
  const std::uint32_t total = totals[i];
  const std::uint32_t ib = in_begin[i];
  const std::uint32_t inc_n = in_end[i] - ib;
  const std::uint32_t old_n = total - inc_n;
  const std::uint32_t fill = ceil_div_u32(total, kSegmentBuckets);
  const std::size_t meta = static_cast<std::size_t>(seg) * kSegmentBuckets;
  const std::size_t slot_base = static_cast<std::size_t>(seg) * kSegmentSlots;
  const std::uint32_t *inc_k = inc_keys + ib;
  const std::uint32_t *inc_v = inc_values + ib;
  const std::uint32_t *old_k = old_keys + old_offset[i];
  const std::uint32_t *old_v = old_values + old_offset[i];
  const std::uint32_t bucket_lo = chunk * chunk_buckets;
  const std::uint32_t rank_lo = bucket_lo * fill;
  const std::uint32_t rank_hi_raw = (bucket_lo + chunk_buckets) * fill;
  const std::uint32_t rank_hi = rank_hi_raw < total ? rank_hi_raw : total;
  __shared__ unsigned long long s_sum;
  if (threadIdx.x == 0)
    s_sum = 0;
  __syncthreads();
  std::uint32_t my_sum = 0;
  for (std::uint32_t r = rank_lo + threadIdx.x; r < rank_hi;
       r += blockDim.x) {
    std::uint32_t key;
    std::uint32_t value;
    ds_merge_pick(inc_k, inc_v, inc_n, old_k, old_v, old_n, r, key, value);
    const std::uint32_t bucket = r / fill;
    const std::uint32_t slot = r % fill;
    const std::size_t pos =
        slot_base + static_cast<std::size_t>(bucket) * kBucketSlots + slot;
    pool_keys[pos] = key;
    pool_values[pos] = value;
    pool_valid[pos] = 1u;
    my_sum += value;
  }
  atomicAdd(&s_sum, static_cast<unsigned long long>(my_sum));
  __syncthreads();
  const std::uint32_t slots_lo = bucket_lo * kBucketSlots;
  for (std::uint32_t p = slots_lo + threadIdx.x;
       p < slots_lo + chunk_buckets * kBucketSlots; p += blockDim.x) {
    const std::uint32_t b = p / kBucketSlots;
    const std::uint32_t slot = p % kBucketSlots;
    const std::uint32_t first = b * fill;
    const std::uint32_t cnt_b =
        first >= total ? 0u : (total - first < fill ? total - first : fill);
    if (slot >= cnt_b) {
      pool_keys[slot_base + p] = kEmptyKey;
      pool_values[slot_base + p] = 0u;
      pool_valid[slot_base + p] = 0u;
    }
  }
  __syncthreads();
  for (std::uint32_t b = bucket_lo + threadIdx.x;
       b < bucket_lo + chunk_buckets; b += blockDim.x) {
    const std::uint32_t first = b * fill;
    const std::uint32_t cnt_b =
        first >= total ? 0u : (total - first < fill ? total - first : fill);
    const std::size_t start =
        slot_base + static_cast<std::size_t>(b) * kBucketSlots;
    std::uint32_t bucket_sum = 0;
    for (std::uint32_t l = 0; l < cnt_b; ++l)
      bucket_sum += pool_values[start + l];
    seg_bucket_live[meta + b] = cnt_b;
    seg_bucket_max[meta + b] =
        cnt_b ? pool_keys[start + cnt_b - 1] : boundary;
    seg_bucket_value_sum[meta + b] = bucket_sum;
  }
  if (threadIdx.x == 0 && s_sum)
    atomicAdd(out_value_sum + i, static_cast<std::uint32_t>(s_sum));
}

__global__ void delta_absorb_fused_kernel(
    const std::uint32_t *ord, const std::uint32_t *batch_begin,
    const std::uint32_t *batch_end, std::size_t absorb_count,
    const std::uint32_t *batch_keys, const std::uint32_t *batch_values,
    const std::uint32_t *dir_seg_id, const std::uint32_t *dir_boundary,
    std::uint32_t *dir_live, std::uint32_t *dir_value_sum,
    const std::uint32_t *delta_keys, const std::uint32_t *delta_values,
    const std::uint8_t *delta_active, std::uint32_t *delta_count,
    std::uint32_t *delta_prefix, std::uint32_t *pool_keys,
    std::uint32_t *pool_values, std::uint32_t *seg_bucket_max,
    std::uint32_t *seg_bucket_live,
    std::uint32_t *seg_bucket_value_sum,
    std::uint32_t *seg_bucket_live_prefix,
    std::uint32_t *seg_bucket_value_prefix) {
  const std::size_t i = blockIdx.x;
  if (i >= absorb_count)
    return;
  extern __shared__ std::uint32_t shared[];
  std::uint32_t *s_keys = shared;
  std::uint32_t *s_values = s_keys + kSegmentSlots;
  std::uint32_t *s_old_prefix = s_values + kSegmentSlots;
  std::uint32_t *s_in_prefix = s_old_prefix + kSegmentBuckets + 1;
  std::uint32_t *s_window = s_in_prefix + kSegmentBuckets + 1;
  std::uint32_t *s_state = s_window + kSegmentBuckets / 8;
  const std::uint32_t o = ord[i];
  const std::uint32_t seg = dir_seg_id[o];
  const std::uint32_t boundary = dir_boundary[o];
  const std::size_t meta =
      static_cast<std::size_t>(seg) * kSegmentBuckets;
  const std::size_t slot_base =
      static_cast<std::size_t>(seg) * kSegmentSlots;
  const std::uint32_t begin = batch_begin[i];
  const std::uint32_t inc_n = batch_end[i] - begin;
  const std::uint32_t dc = delta_count[seg];
  const std::size_t dbase = seg_delta_base(seg, delta_active);
  const std::uint32_t merged_n = dc + inc_n;
  const std::uint32_t merge_chunk = ceil_div_u32(merged_n, blockDim.x);
  const std::uint32_t merge_begin = threadIdx.x * merge_chunk;
  const std::uint32_t merge_end =
      merge_begin + merge_chunk < merged_n ? merge_begin + merge_chunk
                                           : merged_n;
  std::uint32_t inserted_sum = 0;
  if (merge_begin < merged_n) {
    std::uint32_t incoming = seg_merge_partition(
        batch_keys + begin, inc_n, delta_keys + dbase, dc, merge_begin);
    std::uint32_t previous = merge_begin - incoming;
    for (std::uint32_t rank = merge_begin; rank < merge_end; ++rank) {
      std::uint32_t key;
      std::uint32_t value;
      if (incoming < inc_n &&
          (previous >= dc ||
           batch_keys[begin + incoming] <= delta_keys[dbase + previous])) {
        key = batch_keys[begin + incoming];
        value = batch_values[begin + incoming++];
        inserted_sum += value;
      } else {
        key = delta_keys[dbase + previous];
        value = delta_values[dbase + previous++];
      }
      s_keys[rank] = key;
      s_values[rank] = value;
    }
  }
  const std::uint32_t inserted_before =
      block_scan_exclusive(inserted_sum, s_old_prefix);
  if (threadIdx.x == blockDim.x - 1)
    s_state[1] = inserted_before + inserted_sum;
  const std::uint32_t bucket_live = seg_bucket_live[meta + threadIdx.x];
  const std::uint32_t old_before =
      block_scan_exclusive(bucket_live, s_old_prefix);
  s_old_prefix[threadIdx.x] = old_before;
  if (threadIdx.x == blockDim.x - 1)
    s_old_prefix[kSegmentBuckets] = old_before + bucket_live;
  __syncthreads();
  const std::uint32_t old_n = s_old_prefix[kSegmentBuckets];
  const int bucket = threadIdx.x;
  s_in_prefix[bucket] =
      bucket == 0
          ? 0u
          : static_cast<std::uint32_t>(upper_bound_u32(
                s_keys, merged_n, seg_bucket_max[meta + bucket - 1]));
  if (threadIdx.x == 0)
    s_in_prefix[kSegmentBuckets] = merged_n;
  __syncthreads();
  if (threadIdx.x == 0)
    s_state[0] = 1u;
  __syncthreads();
  const std::uint32_t old_bucket =
      s_old_prefix[threadIdx.x + 1] - s_old_prefix[threadIdx.x];
  const std::uint32_t in_bucket =
      s_in_prefix[threadIdx.x + 1] - s_in_prefix[threadIdx.x];
  const unsigned overflow_mask =
      __ballot_sync(0xffffffffu, old_bucket + in_bucket > kBucketSlots);
  const int warp_lane = threadIdx.x & 31;
  if ((warp_lane & 7) == 0) {
    const int window = threadIdx.x / 8;
    const bool overflow = ((overflow_mask >> warp_lane) & 0xffu) != 0u;
    s_window[window] = overflow ? 1u : 0u;
    const int lo = window * 8;
    const int hi = lo + 8;
    const std::uint32_t window_total =
        s_old_prefix[hi] - s_old_prefix[lo] +
        s_in_prefix[hi] - s_in_prefix[lo];
    if (overflow && window_total > 8u * kBucketSlots)
      atomicExch(s_state, 0u);
  }
  __syncthreads();

  const bool selective = s_state[0] != 0u;
  if (!selective || s_window[bucket / 8]) {
    const std::uint32_t live = s_old_prefix[bucket + 1] - s_old_prefix[bucket];
    const std::size_t src =
        slot_base + static_cast<std::size_t>(bucket) * kBucketSlots;
    const std::uint32_t dst = merged_n + s_old_prefix[bucket];
    for (std::uint32_t lane = 0; lane < live; ++lane) {
      s_keys[dst + lane] = pool_keys[src + lane];
      s_values[dst + lane] = pool_values[src + lane];
    }
  }
  __syncthreads();

  const std::uint32_t total = old_n + merged_n;
  if (selective) {
    const int warp = threadIdx.x / kBucketSlots;
    const int lane = threadIdx.x & (kBucketSlots - 1);
    for (int wave = 0; wave < kSegmentBuckets / 8; ++wave) {
      const int b = wave * 8 + warp;
      const std::uint32_t in_begin = s_in_prefix[b];
      const std::uint32_t in_end = s_in_prefix[b + 1];
      const std::uint32_t incoming = in_end - in_begin;
      if (s_window[b / 8] == 0u && incoming > 0) {
        const std::uint32_t live = s_old_prefix[b + 1] - s_old_prefix[b];
        const std::uint32_t output = live + incoming;
        const std::size_t start =
            slot_base + static_cast<std::size_t>(b) * kBucketSlots;
        std::uint32_t key = kEmptyKey;
        std::uint32_t value = 0u;
        if (lane < static_cast<int>(live)) {
          key = pool_keys[start + lane];
          value = pool_values[start + lane];
        } else if (lane < static_cast<int>(output)) {
          const std::uint32_t p = in_begin + lane - live;
          key = s_keys[p];
          value = s_values[p];
        }
        warp_bitonic_sort_pair(key, value, lane);
        if (lane < static_cast<int>(output)) {
          pool_keys[start + lane] = key;
          pool_values[start + lane] = value;
        }
      }
    }
    __syncthreads();
    for (int window = warp; window < kSegmentBuckets / 8; window += 8) {
      if (s_window[window] == 0u)
        continue;
      const int lo = window * 8;
      const int hi = lo + 8;
      const std::uint32_t old_begin = s_old_prefix[lo];
      const std::uint32_t old_count = s_old_prefix[hi] - old_begin;
      const std::uint32_t in_begin = s_in_prefix[lo];
      const std::uint32_t in_count = s_in_prefix[hi] - in_begin;
      const std::uint32_t window_total = old_count + in_count;
      const std::uint32_t fill = ceil_div_u32(window_total, 8u);
      const std::uint32_t chunk = ceil_div_u32(window_total, kBucketSlots);
      const std::uint32_t rank_begin = lane * chunk;
      const std::uint32_t rank_end =
          rank_begin + chunk < window_total ? rank_begin + chunk : window_total;
      if (rank_begin < window_total) {
        const std::uint32_t *old_keys = s_keys + merged_n + old_begin;
        const std::uint32_t *old_values = s_values + merged_n + old_begin;
        const std::uint32_t *incoming_keys = s_keys + in_begin;
        const std::uint32_t *incoming_values = s_values + in_begin;
        std::uint32_t incoming = seg_merge_partition(
            incoming_keys, in_count, old_keys, old_count, rank_begin);
        std::uint32_t previous = rank_begin - incoming;
        for (std::uint32_t rank = rank_begin; rank < rank_end; ++rank) {
          std::uint32_t key;
          std::uint32_t value;
          if (incoming < in_count &&
              (previous >= old_count ||
               incoming_keys[incoming] <= old_keys[previous])) {
            key = incoming_keys[incoming];
            value = incoming_values[incoming++];
          } else {
            key = old_keys[previous];
            value = old_values[previous++];
          }
          const std::uint32_t out_bucket = lo + rank / fill;
          const std::uint32_t out_lane = rank % fill;
          const std::size_t dst =
              slot_base + static_cast<std::size_t>(out_bucket) * kBucketSlots +
              out_lane;
          pool_keys[dst] = key;
          pool_values[dst] = value;
        }
      }
    }
  } else {
    const std::uint32_t pack_fill = ceil_div_u32(total, kSegmentBuckets);
    const std::uint32_t output_chunk = ceil_div_u32(total, blockDim.x);
    const std::uint32_t output_begin = threadIdx.x * output_chunk;
    const std::uint32_t output_end =
        output_begin + output_chunk < total ? output_begin + output_chunk
                                            : total;
    if (output_begin < total) {
      const std::uint32_t *old_keys = s_keys + merged_n;
      const std::uint32_t *old_values = s_values + merged_n;
      std::uint32_t incoming = seg_merge_partition(
          s_keys, merged_n, old_keys, old_n, output_begin);
      std::uint32_t previous = output_begin - incoming;
      for (std::uint32_t rank = output_begin; rank < output_end; ++rank) {
        std::uint32_t key;
        std::uint32_t value;
        if (incoming < merged_n &&
            (previous >= old_n || s_keys[incoming] <= old_keys[previous])) {
          key = s_keys[incoming];
          value = s_values[incoming++];
        } else {
          key = old_keys[previous];
          value = old_values[previous++];
        }
        const std::uint32_t out_bucket = rank / pack_fill;
        const std::uint32_t out_lane = rank % pack_fill;
        const std::size_t dst =
            slot_base + static_cast<std::size_t>(out_bucket) * kBucketSlots +
            out_lane;
        pool_keys[dst] = key;
        pool_values[dst] = value;
      }
    }
  }
  __syncthreads();

  const bool full_repack = !selective;
  const bool window_repack = selective && s_window[bucket / 8] != 0u;
  const std::uint32_t incoming = s_in_prefix[bucket + 1] - s_in_prefix[bucket];
  const bool changed = full_repack || window_repack || incoming != 0u;
  if (changed) {
    std::uint32_t live;
    if (full_repack) {
      const std::uint32_t fill = ceil_div_u32(total, kSegmentBuckets);
      const std::uint32_t first = bucket * fill;
      live = first >= total ? 0u
                            : (total - first < fill ? total - first : fill);
    } else if (window_repack) {
      const int lo = (bucket / 8) * 8;
      const int hi = lo + 8;
      const std::uint32_t window_total =
          s_old_prefix[hi] - s_old_prefix[lo] +
          s_in_prefix[hi] - s_in_prefix[lo];
      const std::uint32_t fill = ceil_div_u32(window_total, 8u);
      const std::uint32_t first = (bucket - lo) * fill;
      live = first >= window_total
                 ? 0u
                 : (window_total - first < fill ? window_total - first : fill);
    } else {
      live = s_old_prefix[bucket + 1] - s_old_prefix[bucket] + incoming;
    }
    const std::size_t start =
        slot_base + static_cast<std::size_t>(bucket) * kBucketSlots;
    std::uint32_t sum = 0;
    for (std::uint32_t lane = 0; lane < live; ++lane)
      sum += pool_values[start + lane];
    seg_bucket_live[meta + bucket] = live;
    seg_bucket_max[meta + bucket] =
        live ? pool_keys[start + live - 1] : boundary;
    seg_bucket_value_sum[meta + bucket] = sum;
  }
  __syncthreads();
  const std::uint32_t live = seg_bucket_live[meta + bucket];
  const std::uint32_t live_before =
      block_scan_exclusive(live, s_old_prefix);
  const std::uint32_t value = seg_bucket_value_sum[meta + bucket];
  const std::uint32_t value_before =
      block_scan_exclusive(value, s_old_prefix);
  const std::size_t prefix_base =
      static_cast<std::size_t>(seg) * (kSegmentBuckets + 1);
  seg_bucket_live_prefix[prefix_base + bucket + 1] = live_before + live;
  seg_bucket_value_prefix[prefix_base + bucket + 1] = value_before + value;
  if (threadIdx.x == 0) {
    seg_bucket_live_prefix[prefix_base] = 0u;
    seg_bucket_value_prefix[prefix_base] = 0u;
    dir_live[o] = total;
    dir_value_sum[o] += s_state[1];
    delta_count[seg] = 0u;
    delta_prefix[static_cast<std::size_t>(seg) * (kDeltaCap + 1)] = 0u;
  }
}

// Pack one split child from merged input.
__global__ void ds_split_write_kernel(
    const std::uint32_t *o_src, const std::uint32_t *o_local,
    const std::uint32_t *o_k, const std::uint32_t *o_child_seg,
    const std::uint32_t *o_old_live, const std::uint32_t *o_old_boundary,
    std::size_t output_count, const std::uint32_t *in_begin,
    const std::uint32_t *in_end, const std::uint32_t *old_offset,
    const std::uint32_t *inc_keys, const std::uint32_t *inc_values,
    const std::uint32_t *old_keys, const std::uint32_t *old_values,
    std::uint32_t *pool_keys, std::uint32_t *pool_values,
    std::uint8_t *pool_valid, std::uint32_t *seg_bucket_max,
    std::uint32_t *seg_bucket_live, std::uint32_t *seg_bucket_value_sum,
    std::uint32_t *out_boundary, std::uint32_t *out_value_sum) {
  const std::size_t o = blockIdx.x;
  if (o >= output_count)
    return;
  const std::uint32_t i = o_src[o];
  const std::uint32_t local = o_local[o];
  const std::uint32_t k = o_k[o];
  const std::uint32_t seg = o_child_seg[o];
  const std::uint32_t old_n = o_old_live[o];
  const std::uint32_t ib = in_begin[i];
  const std::uint32_t inc_n = in_end[i] - ib;
  const std::uint32_t total = old_n + inc_n;
  const std::uint32_t per = ceil_div_u32(total, k);
  const std::uint32_t begin = local * per;
  const std::uint32_t end = begin + per < total ? begin + per : total;
  const std::uint32_t cnt_child = end - begin;
  const std::uint32_t fill = ceil_div_u32(cnt_child, kSegmentBuckets);
  const std::size_t meta = static_cast<std::size_t>(seg) * kSegmentBuckets;
  const std::size_t slot_base = static_cast<std::size_t>(seg) * kSegmentSlots;
  const std::uint32_t *inc_k = inc_keys + ib;
  const std::uint32_t *inc_v = inc_values + ib;
  const std::uint32_t *old_k = old_keys + old_offset[i];
  const std::uint32_t *old_v = old_values + old_offset[i];
  __shared__ std::uint32_t s_boundary;
  __shared__ unsigned long long s_sum;
  if (threadIdx.x == 0) {
    s_sum = 0;
    if (local + 1 == k) {
      s_boundary = o_old_boundary[o];
    } else {
      std::uint32_t key;
      std::uint32_t value;
      ds_merge_pick(inc_k, inc_v, inc_n, old_k, old_v, old_n, end - 1, key,
                    value);
      s_boundary = key;
    }
  }
  __syncthreads();
  std::uint32_t my_sum = 0;
  for (std::uint32_t r = begin + threadIdx.x; r < end; r += blockDim.x) {
    std::uint32_t key;
    std::uint32_t value;
    ds_merge_pick(inc_k, inc_v, inc_n, old_k, old_v, old_n, r, key, value);
    const std::uint32_t rr = r - begin;
    const std::uint32_t bucket = rr / fill;
    const std::uint32_t slot = rr % fill;
    const std::size_t pos =
        slot_base + static_cast<std::size_t>(bucket) * kBucketSlots + slot;
    pool_keys[pos] = key;
    pool_values[pos] = value;
    pool_valid[pos] = 1u;
    my_sum += value;
  }
  atomicAdd(&s_sum, static_cast<unsigned long long>(my_sum));
  __syncthreads();
  for (std::size_t p = threadIdx.x; p < kSegmentSlots; p += blockDim.x) {
    const std::uint32_t b = static_cast<std::uint32_t>(p / kBucketSlots);
    const std::uint32_t slot = static_cast<std::uint32_t>(p % kBucketSlots);
    const std::uint32_t first = b * fill;
    const std::uint32_t cnt_b =
        first >= cnt_child
            ? 0u
            : (cnt_child - first < fill ? cnt_child - first : fill);
    if (slot >= cnt_b) {
      pool_keys[slot_base + p] = kEmptyKey;
      pool_values[slot_base + p] = 0u;
      pool_valid[slot_base + p] = 0u;
    }
  }
  __syncthreads();
  for (int b = threadIdx.x; b < kSegmentBuckets; b += blockDim.x) {
    const std::uint32_t first = b * fill;
    const std::uint32_t cnt_b =
        first >= cnt_child
            ? 0u
            : (cnt_child - first < fill ? cnt_child - first : fill);
    const std::size_t start =
        slot_base + static_cast<std::size_t>(b) * kBucketSlots;
    std::uint32_t bucket_sum = 0;
    for (std::uint32_t l = 0; l < cnt_b; ++l)
      bucket_sum += pool_values[start + l];
    seg_bucket_live[meta + b] = cnt_b;
    seg_bucket_max[meta + b] =
        cnt_b ? pool_keys[start + cnt_b - 1] : s_boundary;
    seg_bucket_value_sum[meta + b] = bucket_sum;
  }
  if (threadIdx.x == 0) {
    out_boundary[o] = s_boundary;
    out_value_sum[o] = static_cast<std::uint32_t>(s_sum);
  }
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
    // Buckets are stored sorted; copy through.
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

__global__ void pack_sort_payload_kernel(
    const std::uint32_t *values, const std::uint8_t *ops, std::size_t n,
    std::uint64_t *payload) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  payload[i] = (static_cast<std::uint64_t>(values[i]) << 8) | ops[i];
}

__global__ void unpack_sort_payload_kernel(
    const std::uint64_t *payload, std::size_t n, std::uint32_t *values,
    std::uint8_t *ops) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  values[i] = static_cast<std::uint32_t>(payload[i] >> 8);
  ops[i] = static_cast<std::uint8_t>(payload[i]);
}

__global__ void point_lookup_walk_kernel(
    const std::uint32_t *queries, std::size_t n, std::uint32_t *out_value,
    std::uint8_t *out_found, const std::uint32_t *perm,
    const std::uint32_t *const *run_keys, const std::uint32_t *const *run_vals,
    const std::uint8_t *const *run_ops, const std::uint32_t *run_cnt,
    int num_runs, const std::uint32_t *dir_boundary,
    const std::uint32_t *dir_radix_first, const std::uint32_t *dir_seg_id,
    std::size_t dir_count,
    const std::uint32_t *pool_keys, const std::uint32_t *pool_values,
    const std::uint8_t *pool_valid, const std::uint32_t *seg_bucket_max,
    const std::uint32_t *seg_bucket_live,
    const std::uint32_t *seg_bucket_live_prefix,
    const std::uint32_t *delta_keys, const std::uint32_t *delta_values,
    const std::uint8_t *delta_active, const std::uint32_t *delta_fence,
    const std::uint32_t *delta_count) {
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
    std::size_t ord =
        dir_route(key, dir_radix_first, dir_boundary, dir_count);
    if (ord >= dir_count)
      ord = dir_count - 1;
    found = seg_point_lookup(dir_seg_id[ord], key, pool_keys, pool_values,
                             pool_valid, seg_bucket_max, seg_bucket_live,
                             seg_bucket_live_prefix, delta_keys, delta_values,
                             delta_active, delta_fence, delta_count, &value);
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
    const std::uint32_t *seg_bucket_live, const std::uint32_t *delta_keys,
    const std::uint32_t *delta_values, const std::uint8_t *delta_active,
    const std::uint32_t *delta_fence, const std::uint32_t *delta_count) {
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
    for (std::size_t qi = qb + qbb; qi < qb + qee; ++qi) {
      const std::uint32_t q = sorted_q[qi];
      const std::size_t ki = lower_bound_u32(pool_keys + base, live, q);
      const std::uint32_t found =
          ki < live && pool_keys[base + ki] == q ? 1u : 0u;
      const std::uint32_t value = found ? pool_values[base + ki] : 0u;
      out_found[qi] = found;
      out_value[qi] = value;
    }
  }
  __syncthreads();
  // Probe deltas for queries missed by buckets.
  const std::uint32_t dc = delta_count[seg];
  if (dc == 0)
    return;
  const std::size_t dbase = seg_delta_base(seg, delta_active);
  const std::uint32_t *dk = delta_keys + dbase;
  const std::uint32_t *dv = delta_values + dbase;
  const std::uint32_t *fence =
      delta_fence + static_cast<std::size_t>(seg) * kDeltaFence;
  for (std::uint32_t qi = qb + threadIdx.x; qi < qb + qn; qi += blockDim.x) {
    if (out_found[qi])
      continue;
    std::uint32_t value;
    if (seg_delta_find(dk, dv, fence, dc, sorted_q[qi], &value)) {
      out_found[qi] = 1u;
      out_value[qi] = value;
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
    const std::uint32_t *dir_boundary, const std::uint32_t *dir_radix_first,
    const std::uint32_t *dir_seg_id, const std::uint32_t *dir_prefix,
    std::size_t dir_count,
    const std::uint32_t *pool_keys,
    const std::uint32_t *seg_bucket_max, const std::uint32_t *seg_bucket_live,
    const std::uint32_t *delta_keys, const std::uint8_t *delta_active,
    const std::uint32_t *delta_fence, const std::uint32_t *delta_count,
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
    std::size_t start_ord =
        dir_route(q, dir_radix_first, dir_boundary, dir_count);
    if (start_ord >= dir_count)
      start_ord = dir_count - 1;
    std::size_t killed =
        killed_count ? lower_bound_u32(killed_keys, killed_count, q) : 0;
    for (std::size_t ord = start_ord; ord < dir_count; ++ord) {
      if (dir_prefix[ord + 1] == dir_prefix[ord])
        continue;
      const std::uint32_t seg = dir_seg_id[ord];
      // Find the first live delta key at or above q.
      std::uint32_t dcand = kEmptyKey;
      const std::uint32_t dc = delta_count[seg];
      if (dc > 0) {
        const std::uint32_t *dk =
            delta_keys + seg_delta_base(seg, delta_active);
        std::uint32_t p = seg_delta_lower_bound(
            dk, delta_fence + static_cast<std::size_t>(seg) * kDeltaFence, dc,
            q);
        for (; p < dc; ++p) {
          const std::uint32_t cand = dk[p];
          const std::size_t kp =
              killed_count ? lower_bound_u32(killed_keys, killed_count, cand)
                           : killed_count;
          if (kp == killed_count || killed_keys[kp] != cand) {
            dcand = cand;
            break;
          }
        }
      }
      const std::uint32_t limit = dcand < best ? dcand : best;
      const std::size_t meta =
          static_cast<std::size_t>(seg) * kSegmentBuckets;
      const std::size_t slots =
          static_cast<std::size_t>(seg) * kSegmentSlots;
      const std::size_t start_bucket =
          ord == start_ord
              ? lower_bound_u32(seg_bucket_max + meta, kSegmentBuckets, q)
              : 0;
      std::uint32_t bcand = kEmptyKey;
      bool capped = false;
      for (std::size_t bucket = start_bucket;
           bucket < kSegmentBuckets && bcand == kEmptyKey && !capped;
           ++bucket) {
        const std::uint32_t live = seg_bucket_live[meta + bucket];
        if (live == 0)
          continue;
        const std::size_t base = slots + bucket * kBucketSlots;
        const std::uint32_t floor =
            ord == start_ord && bucket == start_bucket ? q : 0u;
        std::size_t p = lower_bound_u32(pool_keys + base, live, floor);
        for (; p < live; ++p) {
          const std::uint32_t candidate = pool_keys[base + p];
          if (candidate >= limit) {
            capped = true;
            break;
          }
          while (killed < killed_count && killed_keys[killed] < candidate)
            ++killed;
          if (killed == killed_count || killed_keys[killed] != candidate) {
            bcand = candidate;
            break;
          }
        }
      }
      const std::uint32_t cand = bcand < dcand ? bcand : dcand;
      if (cand != kEmptyKey) {
        out_keys[i] = cand < best ? cand : (best == kEmptyKey ? 0u : best);
        return;
      }
      if (capped) {
        // Later segments cannot improve this candidate.
        out_keys[i] = best == kEmptyKey ? 0u : best;
        return;
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
    const std::uint32_t *dir_radix_first, const std::uint32_t *dir_seg_id,
    std::size_t dir_count,
    const std::uint32_t *pool_keys, const std::uint32_t *pool_values,
    const std::uint8_t *pool_valid, const std::uint32_t *seg_bucket_max,
    const std::uint32_t *seg_bucket_live,
    const std::uint32_t *seg_bucket_live_prefix,
    const std::uint32_t *delta_keys, const std::uint32_t *delta_values,
    const std::uint8_t *delta_active, const std::uint32_t *delta_fence,
    const std::uint32_t *delta_count, std::uint32_t *out_val,
    std::uint32_t *out_flag) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  std::uint32_t v = 0;
  bool f = false;
  if (dir_count > 0) {
    std::size_t ord =
        dir_route(keys[i], dir_radix_first, dir_boundary, dir_count);
    if (ord >= dir_count)
      ord = dir_count - 1;
    f = seg_point_lookup(dir_seg_id[ord], keys[i], pool_keys, pool_values,
                         pool_valid, seg_bucket_max, seg_bucket_live,
                         seg_bucket_live_prefix, delta_keys, delta_values,
                         delta_active, delta_fence, delta_count, &v);
  }
  out_val[i] = f ? v : 0u;
  out_flag[i] = f ? 1u : 0u;
}

__global__ void range_overlay_kernel(
    const std::uint32_t *lo, const std::uint32_t *hi, std::uint32_t *out_sums,
    std::uint32_t *out_counts, std::size_t query_count,
    const std::uint32_t *dir_boundary, const std::uint32_t *dir_radix_first,
    const std::uint32_t *dir_seg_id,
    const std::uint32_t *dir_prefix, const std::uint32_t *dir_value_prefix,
    std::size_t dir_count, const std::uint32_t *pool_keys,
    const std::uint32_t *pool_values, const std::uint8_t *pool_valid,
    const std::uint32_t *seg_bucket_max, const std::uint32_t *seg_bucket_live,
    const std::uint32_t *seg_bucket_live_prefix,
    const std::uint32_t *seg_bucket_value_prefix,
    const std::uint32_t *delta_keys, const std::uint32_t *delta_values,
    const std::uint8_t *delta_active,
    const std::uint32_t *delta_fence, const std::uint32_t *delta_count,
    const std::uint32_t *delta_prefix,
    const std::uint32_t *ins_keys,
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
      l, h, dir_radix_first, dir_boundary, dir_seg_id, dir_value_prefix,
      dir_count, pool_keys, pool_values, pool_valid, seg_bucket_max,
      seg_bucket_live, seg_bucket_value_prefix, delta_keys, delta_values,
      delta_active, delta_fence, delta_count, delta_prefix);
  sum += overlay_prefix_range(ins_prefix, ins_keys, ins_count, l, h);
  sum -= overlay_prefix_range(tomb_val_prefix, tomb_keys, tomb_count, l, h);
  out_sums[i] = sum;
  if (out_counts) {
    std::uint32_t c = seg_sheet_range_count(
        l, h, dir_radix_first, dir_boundary, dir_seg_id, dir_prefix, dir_count,
        pool_keys, pool_valid, seg_bucket_max, seg_bucket_live,
        seg_bucket_live_prefix, delta_keys, delta_active, delta_fence,
        delta_count);
    c += overlay_count_range(ins_keys, ins_count, l, h);
    c -= overlay_prefix_range(tomb_cnt_prefix, tomb_keys, tomb_count, l, h);
    out_counts[i] = c;
  }
}

__global__ void count_overlay_kernel(
    const std::uint32_t *lo, const std::uint32_t *hi, std::uint32_t *out_counts,
    std::size_t query_count, const std::uint32_t *dir_boundary,
    const std::uint32_t *dir_radix_first, const std::uint32_t *dir_seg_id,
    const std::uint32_t *dir_prefix,
    std::size_t dir_count, const std::uint32_t *pool_keys,
    const std::uint8_t *pool_valid, const std::uint32_t *seg_bucket_max,
    const std::uint32_t *seg_bucket_live,
    const std::uint32_t *seg_bucket_live_prefix,
    const std::uint32_t *delta_keys, const std::uint8_t *delta_active,
    const std::uint32_t *delta_fence, const std::uint32_t *delta_count,
    const std::uint32_t *ins_keys,
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
  std::uint32_t c = seg_sheet_range_count(
      l, h, dir_radix_first, dir_boundary, dir_seg_id, dir_prefix, dir_count,
      pool_keys, pool_valid, seg_bucket_max, seg_bucket_live,
      seg_bucket_live_prefix, delta_keys, delta_active, delta_fence,
      delta_count);
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
                              prof_delete_ms_ + prof_sheetmerge_ms_ +
                              prof_route_ms_ + prof_bucket_ms_ +
                              prof_window_ms_ + prof_delta_sort_ms_ +
                              prof_delta_ingest_ms_ +
                              prof_delta_consolidate_ms_;
      const double other = total - measured;
      auto pct = [total](double x) {
        return total > 0.0 ? 100.0 * x / total : 0.0;
      };
      printf("[prof] insert %zu keys: total=%.3f ms\n", batch.count, total);
      printf("[prof]   delta_sort  = %.3f ms (%5.1f%%)\n",
             prof_delta_sort_ms_, pct(prof_delta_sort_ms_));
      printf("[prof]   delta_write = %.3f ms (%5.1f%%)\n",
             prof_delta_ingest_ms_, pct(prof_delta_ingest_ms_));
      printf("[prof]   consolidate = %.3f ms (%5.1f%%)\n",
             prof_delta_consolidate_ms_, pct(prof_delta_consolidate_ms_));
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
#ifdef GPULSMOPT_PROFILE_INSERT
      reset_insert_prof_();
      const auto prof_t0 = std::chrono::high_resolution_clock::now();
#endif
      insert_records(batch.keys, batch.keys,
                     static_cast<std::uint8_t>(gpulsmopt_detail::kTombstone),
                     batch.count, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      maybe_flush_and_merge(stream);
#ifdef GPULSMOPT_PROFILE_INSERT
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const auto prof_t1 = std::chrono::high_resolution_clock::now();
      const double total =
          std::chrono::duration<double, std::milli>(prof_t1 - prof_t0).count();
      printf("[prof] delete %zu keys: total=%.3f ms (route=%.3f bucket=%.3f "
             "flush_sort=%.3f resolve=%.3f tomb=%.3f sheet=%.3f)\n",
             batch.count, total, prof_route_ms_, prof_bucket_ms_,
             prof_flushsort_ms_, prof_resolve_ms_, prof_delete_ms_,
             prof_sheetmerge_ms_);
#endif
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
        raw_or_null(d_dir_boundary_), raw_or_null(d_dir_radix_first_),
        raw_or_null(d_dir_seg_id_), raw_or_null(d_dir_prefix_),
        h_dir_seg_id_.size(),
        raw_or_null(pool_keys_), raw_or_null(pool_valid_),
        raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_),
        raw_or_null(seg_bucket_live_prefix_), raw_or_null(delta_keys_),
        raw_or_null(seg_delta_active_), raw_or_null(seg_delta_fence_),
        raw_or_null(seg_delta_count_),
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
        raw_or_null(d_dir_boundary_), raw_or_null(d_dir_radix_first_),
        raw_or_null(d_dir_seg_id_), raw_or_null(d_dir_prefix_),
        h_dir_seg_id_.size(),
        raw_or_null(pool_keys_), raw_or_null(seg_bucket_max_),
        raw_or_null(seg_bucket_live_), raw_or_null(delta_keys_),
        raw_or_null(seg_delta_active_), raw_or_null(seg_delta_fence_),
        raw_or_null(seg_delta_count_), killed, killed_count, live_ins,
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
        raw_or_null(d_dir_radix_first_), raw_or_null(d_dir_seg_id_),
        raw_or_null(d_dir_prefix_), raw_or_null(d_dir_value_prefix_),
        h_dir_seg_id_.size(),
        raw_or_null(pool_keys_), raw_or_null(pool_values_),
        raw_or_null(pool_valid_), raw_or_null(seg_bucket_max_),
        raw_or_null(seg_bucket_live_), raw_or_null(seg_bucket_live_prefix_),
        raw_or_null(seg_bucket_value_prefix_), raw_or_null(delta_keys_),
        raw_or_null(delta_values_), raw_or_null(seg_delta_active_),
        raw_or_null(seg_delta_fence_), raw_or_null(seg_delta_count_),
        raw_or_null(seg_delta_prefix_),
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
    resize_reuse(scratch_incoming_keys_, n);
    resize_reuse(scratch_incoming_values_, n);
    thrust::copy(policy, thrust::device_pointer_cast(keys),
                 thrust::device_pointer_cast(keys) + n,
                 scratch_incoming_keys_.begin());
    thrust::copy(policy, thrust::device_pointer_cast(values),
                 thrust::device_pointer_cast(values) + n,
                 scratch_incoming_values_.begin());
    thrust::sort_by_key(policy, scratch_incoming_keys_.begin(),
                        scratch_incoming_keys_.end(),
                        scratch_incoming_values_.begin());
    merge_incoming_into_sheet(raw_or_null(scratch_incoming_keys_),
                              raw_or_null(scratch_incoming_values_), n,
                              stream);
    prepare_for_insert(stream);
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
        seg_bucket_live_, seg_bucket_value_sum_, seg_bucket_live_prefix_,
        seg_bucket_value_prefix_, delta_keys_, delta_values_,
        seg_delta_active_, seg_delta_count_, seg_delta_fence_,
        seg_delta_prefix_, delta_dead_, d_dir_seg_id_, d_dir_boundary_,
        d_dir_live_, d_dir_prefix_, d_dir_value_sum_, d_dir_value_prefix_,
        d_dir_radix_first_,
        scratch_incoming_keys_, scratch_incoming_values_, scratch_delete_keys_,
        seg_pull_counter_, seg_dirty_bucket_dirty_,
        seg_cand_seg_, seg_cand_key_, seg_cand_value_,
        c0_log_keys_, c0_log_values_, c0_log_ops_, seg_new_value_sum_,
        seg_d_out_value_sum_, direct_sort_keys_, direct_sort_values_,
        sort_key_output_, sort_payload_input_, sort_payload_output_,
        sort_temp_storage_, delta_inc_keys_, delta_inc_values_, radix_first_,
        flat_bucket_max_, ds_dest_, ds_cnt_cursor_, ds_base_,
        ds_dirty_, ds_staged_keys_, ds_staged_values_, ds_meta_,
        ds_slow_list_, ds_residue_, ds_window_);
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
    return v.capacity() * sizeof(T);
  }
  template <class T>
  static std::size_t
  device_bytes(const gpulsmopt_detail::RawDeviceBuffer<T> &v) {
    return v.capacity() * sizeof(T);
  }
  template <class... Vecs>
  static std::size_t device_bytes_all(const Vecs &...vecs) {
    return (std::size_t{0} + ... + device_bytes(vecs));
  }
  static std::size_t reserve_goal(std::size_t current, std::size_t need) {
    if (current >= need)
      return current;
    std::size_t goal = current == 0 ? 1 : current;
    while (goal < need) {
      if (goal > std::numeric_limits<std::size_t>::max() / 2)
        return need;
      goal *= 2;
    }
    return goal;
  }
  template <class T>
  void resize_reuse(thrust::device_vector<T> &v, std::size_t n) {
    if (v.capacity() < n)
      v.reserve(reserve_goal(v.capacity(), n));
    v.resize(n);
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
    if (!seg_delta_count_.empty()) {
      CUDA_CHECK(cudaMemsetAsync(raw_or_null(seg_delta_count_), 0,
                                 seg_delta_count_.size() *
                                     sizeof(std::uint32_t),
                                 stream));
    }
    if (!seg_delta_active_.empty()) {
      CUDA_CHECK(cudaMemsetAsync(raw_or_null(seg_delta_active_), 0,
                                 seg_delta_active_.size(), stream));
    }
    if (!seg_delta_prefix_.empty()) {
      CUDA_CHECK(cudaMemsetAsync(raw_or_null(seg_delta_prefix_), 0,
                                 seg_delta_prefix_.size() *
                                     sizeof(std::uint32_t),
                                 stream));
    }
    if (!delta_dead_.empty()) {
      CUDA_CHECK(cudaMemsetAsync(raw_or_null(delta_dead_), 0,
                                 delta_dead_.size() * sizeof(std::uint8_t),
                                 stream));
    }
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
    CUDA_CHECK(cudaMemsetAsync(raw_or_null(seg_delta_count_) + seg_id, 0,
                               sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(raw_or_null(seg_delta_active_) + seg_id, 0,
                               sizeof(std::uint8_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        raw_or_null(seg_delta_prefix_) +
            static_cast<std::size_t>(seg_id) *
                (gpulsmopt_detail::kDeltaCap + 1),
        0, sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        raw_or_null(delta_dead_) +
            2u * static_cast<std::size_t>(seg_id) *
                gpulsmopt_detail::kDeltaCap,
        0, 2u * gpulsmopt_detail::kDeltaCap * sizeof(std::uint8_t), stream));
    const std::size_t prefix_base =
        static_cast<std::size_t>(seg_id) *
        (gpulsmopt_detail::kSegmentBuckets + 1);
    CUDA_CHECK(cudaMemsetAsync(
        raw_or_null(seg_bucket_live_prefix_) + prefix_base, 0,
        (gpulsmopt_detail::kSegmentBuckets + 1) * sizeof(std::uint32_t),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        raw_or_null(seg_bucket_value_prefix_) + prefix_base, 0,
        (gpulsmopt_detail::kSegmentBuckets + 1) * sizeof(std::uint32_t),
        stream));
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
    seg_bucket_live_prefix_.resize(
        new_capacity * (gpulsmopt_detail::kSegmentBuckets + 1), 0u);
    seg_bucket_value_prefix_.resize(
        new_capacity * (gpulsmopt_detail::kSegmentBuckets + 1), 0u);
    delta_keys_.resize(2u * new_capacity * gpulsmopt_detail::kDeltaCap,
                       gpulsmopt_detail::kEmptyKey);
    delta_values_.resize(2u * new_capacity * gpulsmopt_detail::kDeltaCap, 0u);
    seg_delta_active_.resize(new_capacity, 0u);
    seg_delta_count_.resize(new_capacity, 0u);
    seg_delta_fence_.resize(new_capacity * gpulsmopt_detail::kDeltaFence,
                            gpulsmopt_detail::kEmptyKey);
    seg_delta_prefix_.resize(
        new_capacity * (gpulsmopt_detail::kDeltaCap + 1), 0u);
    delta_dead_.resize(2u * new_capacity * gpulsmopt_detail::kDeltaCap, 0u);
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
    resize_reuse(d_dir_seg_id_, d);
    resize_reuse(d_dir_boundary_, d);
    resize_reuse(d_dir_radix_first_, gpulsmopt_detail::kDirRadixSize + 1);
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
    const int block = 256;
    const int grid = static_cast<int>((d + 1 + block - 1) / block);
    gpulsmopt_detail::dir_build_radix_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(d_dir_boundary_), d, raw_or_null(d_dir_radix_first_));
    CUDA_CHECK(cudaGetLastError());
    flat_route_dirty_ = true;
    upload_directory_metadata(stream);
    rebuild_all_bucket_prefixes(stream);
  }

  // Rebuild flat bucket and radix delete routing.
  void rebuild_flat_route(cudaStream_t stream) {
    if (!flat_route_dirty_)
      return;
    flat_route_dirty_ = false;
    const std::size_t dir_count = h_dir_seg_id_.size();
    const std::size_t total_buckets =
        dir_count * static_cast<std::size_t>(gpulsmopt_detail::kSegmentBuckets);
    flat_bucket_max_.resize_discard(total_buckets);
    radix_first_.resize_discard(gpulsmopt_detail::kRadixRouteSize + 1);
    const int block = 256;
    const int flat_grid =
        static_cast<int>((total_buckets + block - 1) / block);
    gpulsmopt_detail::ds_build_flat_max_kernel<<<flat_grid, block, 0,
                                                 stream>>>(
        raw_or_null(d_dir_seg_id_), raw_or_null(d_dir_boundary_), dir_count,
        raw_or_null(seg_bucket_max_), flat_bucket_max_.data());
    CUDA_CHECK(cudaGetLastError());
    const int radix_grid =
        static_cast<int>((total_buckets + 1 + block - 1) / block);
    gpulsmopt_detail::ds_build_radix_kernel<<<radix_grid, block, 0, stream>>>(
        flat_bucket_max_.data(), total_buckets, radix_first_.data());
    CUDA_CHECK(cudaGetLastError());
  }

  void upload_directory_metadata(cudaStream_t stream) {
    const std::size_t d = h_dir_seg_id_.size();
    resize_reuse(d_dir_live_, d);
    resize_reuse(d_dir_prefix_, d + 1);
    resize_reuse(d_dir_value_sum_, d);
    resize_reuse(d_dir_value_prefix_, d + 1);
    if (d > 0) {
      CUDA_CHECK(cudaMemcpyAsync(
          raw_or_null(d_dir_live_), h_dir_live_.data(),
          d * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          raw_or_null(d_dir_value_sum_), h_dir_value_sum_.data(),
          d * sizeof(std::uint32_t),
          cudaMemcpyHostToDevice, stream));
    }
    host_metadata_dirty_ = false;
    rebuild_directory_prefixes_device(stream);
  }

  void rebuild_directory_prefixes_device(cudaStream_t stream) {
    const std::size_t d = h_dir_seg_id_.size();
    if (d == 0) {
      CUDA_CHECK(cudaMemsetAsync(raw_or_null(d_dir_prefix_), 0,
                                 sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(raw_or_null(d_dir_value_prefix_), 0,
                                 sizeof(std::uint32_t), stream));
      return;
    }
    if (d > dir_scan_count_) {
      dir_scan_live_bytes_ = 0;
      dir_scan_value_bytes_ = 0;
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          nullptr, dir_scan_live_bytes_, raw_or_null(d_dir_live_),
          raw_or_null(d_dir_prefix_), static_cast<int>(d), stream));
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          nullptr, dir_scan_value_bytes_, raw_or_null(d_dir_value_sum_),
          raw_or_null(d_dir_value_prefix_), static_cast<int>(d), stream));
      dir_scan_count_ = d;
    }
    std::size_t live_bytes = dir_scan_live_bytes_;
    std::size_t value_bytes = dir_scan_value_bytes_;
    ensure_sort_temp(std::max(live_bytes, value_bytes));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        sort_temp_storage_.data(), live_bytes, raw_or_null(d_dir_live_),
        raw_or_null(d_dir_prefix_), static_cast<int>(d), stream));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        sort_temp_storage_.data(), value_bytes, raw_or_null(d_dir_value_sum_),
        raw_or_null(d_dir_value_prefix_), static_cast<int>(d), stream));
    gpulsmopt_detail::directory_prefix_total_kernel<<<1, 1, 0, stream>>>(
        raw_or_null(d_dir_live_), raw_or_null(d_dir_value_sum_), d,
        raw_or_null(d_dir_prefix_), raw_or_null(d_dir_value_prefix_));
    CUDA_CHECK(cudaGetLastError());
  }

  void ensure_host_directory_metadata(cudaStream_t stream) {
    if (!host_metadata_dirty_)
      return;
    const std::size_t d = h_dir_seg_id_.size();
    h_dir_live_.resize(d);
    h_dir_value_sum_.resize(d);
    if (d > 0) {
      CUDA_CHECK(cudaMemcpyAsync(
          h_dir_live_.data(), raw_or_null(d_dir_live_),
          d * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          h_dir_value_sum_.data(), raw_or_null(d_dir_value_sum_),
          d * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    host_metadata_dirty_ = false;
  }

  void rebuild_all_bucket_prefixes(cudaStream_t stream) {
    const std::size_t count = h_dir_seg_id_.size();
    if (count == 0)
      return;
    gpulsmopt_detail::seg_bucket_prefix_kernel<<<count, 256, 0, stream>>>(
        nullptr, raw_or_null(d_dir_seg_id_), count,
        raw_or_null(seg_bucket_live_), raw_or_null(seg_bucket_value_sum_),
        raw_or_null(seg_bucket_live_prefix_),
        raw_or_null(seg_bucket_value_prefix_));
    CUDA_CHECK(cudaGetLastError());
  }

  void rebuild_direct_bucket_prefixes(const std::uint32_t *seg_ids,
                                      std::size_t count,
                                      cudaStream_t stream) {
    if (count == 0)
      return;
    gpulsmopt_detail::seg_bucket_prefix_kernel<<<count, 256, 0, stream>>>(
        seg_ids, nullptr, count, raw_or_null(seg_bucket_live_),
        raw_or_null(seg_bucket_value_sum_),
        raw_or_null(seg_bucket_live_prefix_),
        raw_or_null(seg_bucket_value_prefix_));
    CUDA_CHECK(cudaGetLastError());
  }

  void rebuild_ordinal_bucket_prefixes(const std::uint32_t *ordinals,
                                       std::size_t count,
                                       cudaStream_t stream) {
    if (count == 0)
      return;
    gpulsmopt_detail::seg_bucket_prefix_kernel<<<count, 256, 0, stream>>>(
        ordinals, raw_or_null(d_dir_seg_id_), count,
        raw_or_null(seg_bucket_live_), raw_or_null(seg_bucket_value_sum_),
        raw_or_null(seg_bucket_live_prefix_),
        raw_or_null(seg_bucket_value_prefix_));
    CUDA_CHECK(cudaGetLastError());
  }

  template <class T>
  void upload_vec(thrust::device_vector<T> &dst, const std::vector<T> &src,
                  cudaStream_t stream) {
    resize_reuse(dst, src.size());
    if (!src.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(raw_or_null(dst), src.data(),
                                 src.size() * sizeof(T), cudaMemcpyHostToDevice,
                                 stream));
    }
  }

  void resize_seg_candidates(std::size_t count) {
    resize_reuse(seg_cand_seg_, count);
    resize_reuse(seg_cand_key_, count);
    resize_reuse(seg_cand_value_, count);
  }

  void prepare_dirty_plan(std::size_t dirty_count, cudaStream_t stream) {
    auto policy = thrust::cuda::par.on(stream);
    resize_reuse(seg_d_dirty_seg_id_, dirty_count);
    resize_reuse(seg_d_dirty_old_boundary_, dirty_count);
    resize_reuse(seg_d_dirty_old_live_, dirty_count);
    resize_reuse(seg_d_dirty_old_value_sum_, dirty_count);
    resize_reuse(seg_d_dirty_in_begin_, dirty_count);
    resize_reuse(seg_d_dirty_in_end_, dirty_count);
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

  void build_key_routed_dirty_plan(const std::uint32_t *incoming_keys,
                                   std::size_t incoming_count,
                                   std::size_t dir_count,
                                   cudaStream_t stream, std::size_t &m,
                                   std::size_t &mb) {
    auto policy = thrust::cuda::par.on(stream);
    resize_reuse(seg_inc_ordinal_, incoming_count);
    {
      const int block = 256;
      const int grid = static_cast<int>((incoming_count + block - 1) / block);
      gpulsmopt_detail::seg_route_keys_kernel<<<grid, block, 0, stream>>>(
          incoming_keys, incoming_count,
          raw_or_null(d_dir_boundary_), dir_count, raw_or_null(seg_inc_ordinal_));
      CUDA_CHECK(cudaGetLastError());
    }
    resize_reuse(seg_dirty_ord_, incoming_count);
    resize_reuse(seg_dirty_count_, incoming_count);
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
    resize_reuse(seg_inc_bucket_, incoming_count);
    {
      const int block = 256;
      const int grid = static_cast<int>((incoming_count + block - 1) / block);
      gpulsmopt_detail::seg_route_buckets_kernel<<<grid, block, 0, stream>>>(
          incoming_keys, raw_or_null(seg_inc_ordinal_),
          incoming_count, raw_or_null(d_dir_seg_id_),
          raw_or_null(seg_bucket_max_), raw_or_null(seg_inc_bucket_));
      CUDA_CHECK(cudaGetLastError());
    }
    resize_reuse(seg_dirty_bucket_, incoming_count);
    resize_reuse(seg_dirty_bucket_count_, incoming_count);
    auto bucket_rle_end = thrust::reduce_by_key(
        policy, seg_inc_bucket_.begin(), seg_inc_bucket_.begin() + incoming_count,
        thrust::make_constant_iterator<std::uint32_t>(1u),
        seg_dirty_bucket_.begin(), seg_dirty_bucket_count_.begin());
    mb = static_cast<std::size_t>(
        bucket_rle_end.first - seg_dirty_bucket_.begin());
    resize_reuse(seg_dirty_bucket_begin_, mb);
    thrust::exclusive_scan(policy, seg_dirty_bucket_count_.begin(),
                           seg_dirty_bucket_count_.begin() + mb,
                           seg_dirty_bucket_begin_.begin());
  }

  bool build_pull_dirty_plan(const std::uint32_t *incoming_keys,
                             std::size_t incoming_count, std::size_t dir_count,
                             cudaStream_t stream, std::size_t &m,
                             std::size_t &mb) {
    resize_reuse(seg_pull_counter_, 2);
    CUDA_CHECK(cudaMemsetAsync(raw_or_null(seg_pull_counter_), 0,
                               2 * sizeof(std::uint32_t), stream));
    resize_reuse(seg_dirty_ord_, dir_count);
    resize_reuse(seg_dirty_count_, dir_count);
    resize_reuse(seg_d_dirty_seg_id_, dir_count);
    resize_reuse(seg_d_dirty_old_boundary_, dir_count);
    resize_reuse(seg_d_dirty_old_live_, dir_count);
    resize_reuse(seg_d_dirty_old_value_sum_, dir_count);
    resize_reuse(seg_d_dirty_in_begin_, dir_count);
    resize_reuse(seg_d_dirty_in_end_, dir_count);
    resize_reuse(seg_slow_, dir_count);
    resize_reuse(seg_new_live_, dir_count);
    resize_reuse(seg_new_value_sum_, dir_count);
    {
      const int block = 256;
      const int grid = static_cast<int>((dir_count + block - 1) / block);
      gpulsmopt_detail::seg_pull_dirty_segments_kernel<<<grid, block, 0,
                                                         stream>>>(
          incoming_keys, incoming_count,
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
    resize_reuse(seg_dirty_bucket_, bucket_slots);
    resize_reuse(seg_dirty_bucket_count_, bucket_slots);
    resize_reuse(seg_dirty_bucket_begin_, bucket_slots);
    resize_reuse(seg_dirty_bucket_dirty_, bucket_slots);
    CUDA_CHECK(cudaMemsetAsync(raw_or_null(seg_pull_counter_) + 1, 0,
                               sizeof(std::uint32_t), stream));
    {
      const int block = 256;
      const int grid = static_cast<int>((bucket_slots + block - 1) / block);
      gpulsmopt_detail::seg_pull_dirty_buckets_kernel<<<grid, block, 0,
                                                        stream>>>(
          incoming_keys, raw_or_null(seg_dirty_ord_),
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

  void build_pull_apply_plan(const std::uint32_t *incoming_keys,
                             const std::uint32_t *incoming_values,
                             std::size_t incoming_count, std::size_t dir_count,
                             cudaStream_t stream, std::size_t &m,
                             std::size_t &ms) {
    resize_reuse(seg_pull_counter_, 2);
    CUDA_CHECK(cudaMemsetAsync(raw_or_null(seg_pull_counter_), 0,
                               2 * sizeof(std::uint32_t), stream));
    resize_reuse(seg_dirty_ord_, dir_count);
    resize_reuse(seg_dirty_count_, dir_count);
    resize_reuse(seg_d_dirty_seg_id_, dir_count);
    resize_reuse(seg_d_dirty_old_boundary_, dir_count);
    resize_reuse(seg_d_dirty_old_live_, dir_count);
    resize_reuse(seg_d_dirty_old_value_sum_, dir_count);
    resize_reuse(seg_d_dirty_in_begin_, dir_count);
    resize_reuse(seg_d_dirty_in_end_, dir_count);
    resize_reuse(seg_slow_, dir_count);
    resize_reuse(seg_new_live_, dir_count);
    resize_reuse(seg_new_value_sum_, dir_count);
    resize_reuse(seg_slow_index_, dir_count);
    const unsigned block =
        static_cast<unsigned>(gpulsmopt_detail::kSegmentBuckets);
    gpulsmopt_detail::seg_pull_apply_fast_kernel<<<
        static_cast<unsigned>(dir_count), block, 0, stream>>>(
        incoming_keys, incoming_values, incoming_count,
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
        raw_or_null(seg_slow_index_), raw_or_null(seg_pull_counter_),
        raw_or_null(pool_keys_), raw_or_null(pool_values_),
        raw_or_null(pool_valid_), raw_or_null(seg_bucket_max_),
        raw_or_null(seg_bucket_live_), raw_or_null(seg_bucket_value_sum_));
    CUDA_CHECK(cudaGetLastError());
    std::uint32_t counts[2] = {0u, 0u};
    CUDA_CHECK(cudaMemcpyAsync(counts, raw_or_null(seg_pull_counter_),
                               2 * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    m = counts[0];
    ms = counts[1];
  }

  void merge_incoming_into_sheet(const std::uint32_t *incoming_keys,
                                 const std::uint32_t *incoming_values,
                                 std::size_t incoming_count,
                                 cudaStream_t stream) {
    if (incoming_count == 0)
      return;
    consolidate_all_deltas(stream);
    ensure_host_directory_metadata(stream);
    auto policy = thrust::cuda::par.on(stream);

    const std::size_t dir_count = h_dir_seg_id_.size();
    if (dir_count == 0)
      return;
    std::size_t m = 0, mb = 0;
    bool fused_plan = false;
    bool fused_apply = false;
    if (incoming_count < dir_count * 16) {
      build_key_routed_dirty_plan(incoming_keys, incoming_count, dir_count,
                                  stream, m, mb);
    } else if (dir_count * static_cast<std::size_t>(
                            gpulsmopt_detail::kSegmentBuckets) <=
               incoming_count * 2) {
      build_pull_apply_plan(incoming_keys, incoming_values, incoming_count,
                            dir_count, stream, m, mb);
      fused_plan = true;
      fused_apply = true;
    } else {
      fused_plan = build_pull_dirty_plan(incoming_keys, incoming_count,
                                         dir_count, stream, m, mb);
      if (!fused_plan)
        build_key_routed_dirty_plan(incoming_keys, incoming_count, dir_count,
                                    stream, m, mb);
    }
    if (m == 0)
      return;
    if (!fused_plan) {
      resize_reuse(seg_slow_, m);
      resize_reuse(seg_new_live_, m);
      resize_reuse(seg_new_value_sum_, m);
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

    std::size_t fast_count = fused_apply ? m - mb : 0;
    std::size_t ms = fused_apply ? mb : 0;
    if (!fused_apply) {
      resize_reuse(seg_slow_index_, m);
      resize_reuse(seg_pull_counter_, 2);
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
      fast_count = split_count[0];
      ms = split_count[1];
    }

    if (!fused_apply && fast_count > 0 && mb > 0) {
      const int block = 256;
      const int grid = static_cast<int>((mb + block - 1) / block);
      gpulsmopt_detail::seg_apply_dirty_buckets_kernel<<<
          grid, block, 0, stream>>>(
          raw_or_null(seg_dirty_bucket_), raw_or_null(seg_dirty_bucket_count_),
          raw_or_null(seg_dirty_bucket_begin_),
          fused_plan ? raw_or_null(seg_dirty_bucket_dirty_) : nullptr, mb,
          raw_or_null(seg_dirty_ord_), m, raw_or_null(seg_slow_),
          raw_or_null(d_dir_seg_id_), incoming_keys, incoming_values,
          raw_or_null(pool_keys_),
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
      resize_reuse(seg_slow_seg_id_, ms);
      resize_reuse(seg_slow_old_boundary_, ms);
      resize_reuse(seg_slow_old_live_, ms);
      resize_reuse(seg_slow_in_begin_, ms);
      resize_reuse(seg_slow_in_end_, ms);
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

      resize_reuse(seg_slow_inc_count_, ms);
      resize_reuse(seg_slow_cand_count_, ms);
      gpulsmopt_detail::seg_make_slow_counts_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(seg_slow_old_live_), raw_or_null(seg_slow_in_begin_),
          raw_or_null(seg_slow_in_end_), ms,
          raw_or_null(seg_slow_inc_count_),
          raw_or_null(seg_slow_cand_count_));
      CUDA_CHECK(cudaGetLastError());

      resize_reuse(seg_d_old_offset_, ms);
      thrust::exclusive_scan(policy, seg_slow_old_live_.begin(),
                             seg_slow_old_live_.begin() + ms,
                             seg_d_old_offset_.begin());
      resize_reuse(seg_dirty_live_, ms);
      thrust::copy(policy, seg_slow_cand_count_.begin(),
                   seg_slow_cand_count_.begin() + ms,
                   seg_dirty_live_.begin());
      resize_reuse(seg_d_dirty_k_, ms);
      gpulsmopt_detail::seg_make_dirty_k_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(seg_dirty_live_), raw_or_null(seg_slow_seg_id_), ms,
          target_segment_live_, raw_or_null(seg_d_dirty_k_));
      CUDA_CHECK(cudaGetLastError());
      resize_reuse(seg_d_dirty_output_base_, ms);
      thrust::exclusive_scan(policy, seg_d_dirty_k_.begin(),
                             seg_d_dirty_k_.begin() + ms,
                             seg_d_dirty_output_base_.begin());
      resize_reuse(seg_plan_total_, 1);
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
      resize_reuse(seg_d_output_seg_id_, max_output);
      resize_reuse(seg_d_out_dirty_, max_output);
      resize_reuse(seg_d_out_local_, max_output);
      resize_reuse(seg_d_out_k_, max_output);
      resize_reuse(seg_d_out_live_, max_output);
      resize_reuse(seg_d_out_boundary_, max_output);
      resize_reuse(seg_d_out_value_sum_, max_output);
      gpulsmopt_detail::seg_output_plan_kernel<<<
          static_cast<unsigned>(ms), 256, 0, stream>>>(
          raw_or_null(seg_dirty_live_), raw_or_null(seg_d_dirty_k_),
          raw_or_null(seg_d_dirty_output_base_), ms,
          raw_or_null(seg_d_out_dirty_), raw_or_null(seg_d_out_local_),
          raw_or_null(seg_d_out_k_), raw_or_null(seg_d_out_live_));
      CUDA_CHECK(cudaGetLastError());

      const std::size_t old_capacity =
          ms * static_cast<std::size_t>(gpulsmopt_detail::kSegmentSlots);
      seg_old_key_.resize_discard(old_capacity);
      seg_old_value_.resize_discard(old_capacity);
      gpulsmopt_detail::seg_gather_old_ordered_kernel<<<
          static_cast<unsigned>(ms), 256, 0, stream>>>(
          raw_or_null(pool_keys_), raw_or_null(pool_values_),
          raw_or_null(seg_bucket_live_), raw_or_null(seg_slow_seg_id_),
          raw_or_null(seg_d_old_offset_), ms, nullptr,
          seg_old_key_.data(), seg_old_value_.data());
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
          incoming_keys, incoming_values, seg_old_key_.data(),
          seg_old_value_.data(),
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
    // This gather reads bucket storage only.
    consolidate_all_deltas(stream);
    ensure_host_directory_metadata(stream);
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
    resize_reuse(seg_d_out_value_sum_, group_count);

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
        break;
      }
    }
    c0_log_keys_.reserve(reserve_count);
    c0_log_values_.reserve(reserve_count);
    c0_log_ops_.reserve(reserve_count);
    if (c0_log_keys_.size() < reserve_count)
      c0_log_keys_.resize(reserve_count);
    if (c0_log_values_.size() < reserve_count)
      c0_log_values_.resize(reserve_count);
    if (c0_log_ops_.size() < reserve_count)
      c0_log_ops_.resize(reserve_count);
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
    if (op == static_cast<std::uint8_t>(gpulsmopt_detail::kTombstone) &&
        try_scatter_sheet_delete(keys_in, count, stream))
      return;
    if (try_scatter_sheet_insert(keys_in, values_in, op, count, stream))
      return;
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
    if (run.keys.size() < count)
      run.keys.resize(count);
    if (run.values.size() < count)
      run.values.resize(count);
    if (run.ops.size() < count)
      run.ops.resize(count);
    run.log_total = static_cast<std::uint32_t>(count);
    run.level = level;
    return run;
  }

  void release_run_storage(SortedRun &&run) {
    run.log_total = 0;
    run.level = 0;
    run_buffer_pool_.push_back(std::move(run));
  }

  void reserve_run_storage(std::size_t count) {
    SortedRun run;
    run.keys.resize(count);
    run.values.resize(count);
    run.ops.resize(count);
    release_run_storage(std::move(run));
  }

  void prepare_run_storage() {
    if (!runs_.empty() || !run_buffer_pool_.empty())
      return;
    const std::size_t budget = c0_flush_budget();
    const std::size_t limit = std::min(max_elements_, 4 * budget);
    run_buffer_pool_.reserve(5);
    if (budget <= limit) {
      reserve_run_storage(budget);
      reserve_run_storage(budget);
    }
    if (2 * budget <= limit) {
      reserve_run_storage(2 * budget);
      reserve_run_storage(2 * budget);
    }
    if (4 * budget <= limit)
      reserve_run_storage(4 * budget);
  }

  void recycle_active_runs() {
    for (auto &run : runs_)
      release_run_storage(std::move(run));
    runs_.clear();
  }

  void ensure_sort_temp(std::size_t bytes) {
    if (sort_temp_storage_.capacity() < bytes)
      sort_temp_storage_.resize_discard(bytes);
  }

  void exclusive_scan_u32(const std::uint32_t *input,
                          std::uint32_t *output, std::size_t count,
                          cudaStream_t stream) {
    if (count == 0)
      return;
    std::size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_bytes, input, output, static_cast<int>(count), stream));
    ensure_sort_temp(temp_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        sort_temp_storage_.data(), temp_bytes, input, output,
        static_cast<int>(count), stream));
  }

  void prepare_sort_storage(std::size_t direct_count,
                            std::size_t log_count,
                            cudaStream_t stream) {
    direct_sort_keys_.resize_discard(direct_count);
    direct_sort_values_.resize_discard(direct_count);
    sort_key_output_.resize_discard(log_count);
    sort_payload_input_.resize_discard(log_count);
    sort_payload_output_.resize_discard(log_count);
    std::size_t direct_bytes = 0;
    if (direct_count > 0) {
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, direct_bytes, raw_or_null(scratch_incoming_keys_),
          direct_sort_keys_.data(), raw_or_null(scratch_incoming_values_),
          direct_sort_values_.data(), direct_count, 0, 32, stream));
      direct_sort_count_ = direct_count;
      direct_sort_temp_bytes_ = direct_bytes;
    }
    std::size_t log_bytes = 0;
    if (log_count > 0) {
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, log_bytes, raw_or_null(scratch_incoming_keys_),
          sort_key_output_.data(), sort_payload_input_.data(),
          sort_payload_output_.data(), log_count, 0, 32, stream));
    }
    ensure_sort_temp(std::max(direct_bytes, log_bytes));
  }

  // Reserve direct-path buffers before timed updates.
  void prepare_scatter_storage() {
    const std::size_t batch_hint =
        std::max<std::size_t>(4 * c0_flush_budget(), batch_size_);
    // Leave headroom for directory growth.
    const std::size_t dir_count = 2 * h_dir_seg_id_.size();
    const std::size_t total_buckets =
        dir_count * static_cast<std::size_t>(gpulsmopt_detail::kSegmentBuckets);
    ds_dest_.resize_discard(batch_hint);
    ds_cnt_cursor_.resize_discard(2 * total_buckets);
    ds_base_.resize_discard(total_buckets);
    ds_dirty_.resize_discard(std::min(batch_hint, total_buckets));
    ds_staged_keys_.resize_discard(batch_hint);
    ds_staged_values_.resize_discard(batch_hint);
    ds_meta_.resize_discard(3 * dir_count + 4);
    ds_slow_list_.resize_discard(dir_count);
    ds_residue_.resize_discard(dir_count);
    ds_window_.resize_discard(3 * std::min(batch_hint, total_buckets));
    flat_bucket_max_.resize_discard(total_buckets);
    radix_first_.resize_discard(gpulsmopt_detail::kRadixRouteSize + 1);
    delta_inc_keys_.resize_discard(
        batch_hint + dir_count * gpulsmopt_detail::kDeltaCap);
    delta_inc_values_.resize_discard(
        batch_hint + dir_count * gpulsmopt_detail::kDeltaCap);
    resize_reuse(seg_slow_old_live_, dir_count);
    seg_old_key_.resize_discard(max_elements_ / 2 + 1);
    seg_old_value_.resize_discard(max_elements_ / 2 + 1);
    resize_reuse(seg_dirty_ord_, dir_count);
    resize_reuse(seg_slow_seg_id_, dir_count);
    resize_reuse(seg_d_old_offset_, dir_count);
    resize_reuse(seg_slow_old_boundary_, dir_count);
    resize_reuse(seg_dirty_live_, dir_count);
    resize_reuse(seg_slow_in_begin_, dir_count);
    resize_reuse(seg_slow_in_end_, dir_count);
    resize_reuse(seg_d_out_value_sum_, dir_count);
    resize_reuse(seg_new_value_sum_, dir_count);
    resize_reuse(seg_d_out_boundary_, dir_count);
    resize_reuse(seg_d_out_dirty_, dir_count);
    resize_reuse(seg_d_out_local_, dir_count);
    resize_reuse(seg_d_out_k_, dir_count);
    resize_reuse(seg_d_output_seg_id_, dir_count);
    resize_reuse(seg_src_out_base_, dir_count);
    resize_reuse(seg_src_seg_id_, dir_count);
    prepare_absorb_kernel();
  }

  void prepare_for_insert(cudaStream_t stream) {
    const std::size_t direct_count = std::min(
        max_elements_, std::max(4 * c0_flush_budget(), batch_size_));
    ensure_c0_log(stream);
    prepare_sort_storage(direct_count, c0_flush_budget(), stream);
    prepare_scatter_storage();
    prepare_run_storage();
  }

  static constexpr std::size_t absorb_shared_bytes() {
    return (2 * gpulsmopt_detail::kSegmentSlots +
            2 * (gpulsmopt_detail::kSegmentBuckets + 1) +
            gpulsmopt_detail::kSegmentBuckets / 8 + 2) *
           sizeof(std::uint32_t);
  }

  void prepare_absorb_kernel() {
    if (absorb_kernel_ready_)
      return;
    CUDA_CHECK(cudaFuncSetAttribute(
        gpulsmopt_detail::delta_absorb_fused_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(absorb_shared_bytes())));
    absorb_kernel_ready_ = true;
  }

  void sort_direct_batch(const std::uint32_t *keys,
                         const std::uint32_t *values, std::size_t n,
                         cudaStream_t stream) {
    direct_sort_keys_.resize_discard(n);
    direct_sort_values_.resize_discard(n);
    if (n > direct_sort_count_) {
      direct_sort_temp_bytes_ = 0;
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, direct_sort_temp_bytes_, keys, direct_sort_keys_.data(),
          values, direct_sort_values_.data(), n, 0, 32, stream));
      direct_sort_count_ = n;
    }
    std::size_t temp_bytes = direct_sort_temp_bytes_;
    ensure_sort_temp(temp_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sort_temp_storage_.data(), temp_bytes, keys, direct_sort_keys_.data(),
        values, direct_sort_values_.data(), n, 0, 32, stream));
  }

  void sort_log(thrust::device_vector<std::uint32_t> &k,
                thrust::device_vector<std::uint32_t> &v,
                thrust::device_vector<std::uint8_t> &op, std::size_t n,
                cudaStream_t stream) {
    if (n == 0) {
      return;
    }
    sort_key_output_.resize_discard(n);
    sort_payload_input_.resize_discard(n);
    sort_payload_output_.resize_discard(n);
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    gpulsmopt_detail::pack_sort_payload_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(v), raw_or_null(op), n, sort_payload_input_.data());
    CUDA_CHECK(cudaGetLastError());
    std::size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes, raw_or_null(k), sort_key_output_.data(),
        sort_payload_input_.data(), sort_payload_output_.data(), n, 0, 32,
        stream));
    ensure_sort_temp(temp_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sort_temp_storage_.data(), temp_bytes, raw_or_null(k),
        sort_key_output_.data(), sort_payload_input_.data(),
        sort_payload_output_.data(), n, 0, 32, stream));
    CUDA_CHECK(cudaMemcpyAsync(raw_or_null(k), sort_key_output_.data(),
                               n * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToDevice, stream));
    gpulsmopt_detail::unpack_sort_payload_kernel<<<grid, block, 0, stream>>>(
        sort_payload_output_.data(), n, raw_or_null(v), raw_or_null(op));
    CUDA_CHECK(cudaGetLastError());
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

  struct DeltaConsolidation {
    std::uint32_t ord;
    std::uint32_t inc_begin;
    std::uint32_t inc_len;
    std::uint32_t dcount;
  };

  // Consolidate full deltas into buckets or splits.
  void run_consolidations(std::vector<DeltaConsolidation> &entries,
                          const std::uint32_t *batch_keys,
                          const std::uint32_t *batch_values,
                          cudaStream_t stream) {
    if (entries.empty())
      return;
    ensure_host_directory_metadata(stream);
    const std::size_t dir_count = h_dir_seg_id_.size();
    const int block = 256;
    std::sort(entries.begin(), entries.end(),
              [](const DeltaConsolidation &a, const DeltaConsolidation &b) {
                return a.ord < b.ord;
              });
    std::vector<DeltaConsolidation> absorb, split;
    for (const auto &e : entries) {
      if (h_dir_live_[e.ord] + e.inc_len <=
          gpulsmopt_detail::seg_split_trigger(h_dir_seg_id_[e.ord]))
        absorb.push_back(e);
      else
        split.push_back(e);
    }
    const std::size_t absorb_count = absorb.size();
    const std::size_t split_count = split.size();
    const std::size_t residue_count = absorb_count + split_count;

    // Allocate before taking pool pointers.
    std::vector<std::uint32_t> split_k(split_count);
    std::vector<std::uint32_t> child_seg;
    std::size_t outputs_total = 0;
    for (std::size_t s = 0; s < split_count; ++s) {
      const std::uint32_t total = h_dir_live_[split[s].ord] + split[s].inc_len;
      std::uint32_t k =
          gpulsmopt_detail::ceil_div_u32(total, target_segment_live_);
      if (k < 2)
        k = 2;
      split_k[s] = k;
      outputs_total += k;
    }
    child_seg.reserve(outputs_total);
    for (std::size_t s = 0; s < split_count; ++s) {
      child_seg.push_back(h_dir_seg_id_[split[s].ord]);
      for (std::uint32_t c = 1; c < split_k[s]; ++c)
        child_seg.push_back(alloc_segment());
    }

    std::vector<std::uint32_t> r_ord(residue_count), r_seg(residue_count),
        r_old_offset(residue_count), r_in_begin(residue_count),
        r_in_end(residue_count), r_slice_begin(residue_count),
        r_slice_end(residue_count);
    std::vector<std::uint32_t> a_bound(absorb_count), a_total(absorb_count);
    std::vector<std::uint32_t> bucket_live(residue_count);
    std::uint32_t old_acc = 0;
    std::uint32_t inc_acc = 0;
    auto fill_entry = [&](std::size_t i, const DeltaConsolidation &e) {
      r_ord[i] = e.ord;
      r_seg[i] = h_dir_seg_id_[e.ord];
      bucket_live[i] = h_dir_live_[e.ord] - e.dcount;
      r_old_offset[i] = old_acc;
      old_acc += bucket_live[i];
      const std::uint32_t inc_total = e.dcount + e.inc_len;
      r_in_begin[i] = inc_acc;
      r_in_end[i] = inc_acc + inc_total;
      inc_acc += inc_total;
      r_slice_begin[i] = e.inc_begin;
      r_slice_end[i] = e.inc_begin + e.inc_len;
    };
    for (std::size_t i = 0; i < absorb_count; ++i) {
      fill_entry(i, absorb[i]);
      a_bound[i] = h_dir_boundary_[absorb[i].ord];
      a_total[i] = h_dir_live_[absorb[i].ord] + absorb[i].inc_len;
    }
    for (std::size_t s = 0; s < split_count; ++s)
      fill_entry(absorb_count + s, split[s]);

    upload_vec(seg_dirty_ord_, r_ord, stream);
    upload_vec(seg_slow_seg_id_, r_seg, stream);
    upload_vec(seg_d_old_offset_, r_old_offset, stream);
    upload_vec(seg_slow_old_boundary_, a_bound, stream);
    upload_vec(seg_dirty_live_, a_total, stream);
    upload_vec(seg_slow_in_begin_, r_in_begin, stream);
    upload_vec(seg_slow_in_end_, r_in_end, stream);
    upload_vec(seg_slow_old_live_, r_slice_begin, stream);
    upload_vec(seg_slow_inc_count_, r_slice_end, stream);
    delta_inc_keys_.resize_discard(std::max<std::size_t>(inc_acc, 1));
    delta_inc_values_.resize_discard(std::max<std::size_t>(inc_acc, 1));
    seg_old_key_.resize_discard(std::max<std::size_t>(old_acc, 1));
    seg_old_value_.resize_discard(std::max<std::size_t>(old_acc, 1));

    gpulsmopt_detail::delta_inc_merge_kernel<<<
        static_cast<unsigned>(residue_count), block, 0, stream>>>(
        raw_or_null(seg_slow_seg_id_), raw_or_null(seg_slow_old_live_),
        raw_or_null(seg_slow_inc_count_), raw_or_null(seg_slow_in_begin_),
        residue_count, batch_keys, batch_values, raw_or_null(delta_keys_),
        raw_or_null(delta_values_), raw_or_null(seg_delta_active_),
        raw_or_null(seg_delta_count_), delta_inc_keys_.data(),
        delta_inc_values_.data());
    CUDA_CHECK(cudaGetLastError());
    gpulsmopt_detail::delta_reset_kernel<<<
        static_cast<unsigned>(residue_count), block, 0, stream>>>(
        raw_or_null(seg_slow_seg_id_), residue_count,
        raw_or_null(seg_delta_active_), raw_or_null(seg_delta_count_),
        raw_or_null(seg_delta_prefix_));
    CUDA_CHECK(cudaGetLastError());
    gpulsmopt_detail::ds_gather_old_kernel<<<
        static_cast<unsigned>(residue_count *
                              gpulsmopt_detail::kDsGatherChunks),
        block, 0, stream>>>(
        raw_or_null(seg_slow_seg_id_), raw_or_null(seg_d_old_offset_),
        residue_count, raw_or_null(pool_keys_), raw_or_null(pool_values_),
        raw_or_null(seg_bucket_live_), seg_old_key_.data(),
        seg_old_value_.data());
    CUDA_CHECK(cudaGetLastError());

    std::vector<std::uint32_t> a_value_sum(absorb_count);
    if (absorb_count > 0) {
      resize_reuse(seg_d_out_value_sum_, absorb_count);
      CUDA_CHECK(cudaMemsetAsync(raw_or_null(seg_d_out_value_sum_), 0,
                                 absorb_count * sizeof(std::uint32_t),
                                 stream));
      gpulsmopt_detail::ds_absorb_write_kernel<<<
          static_cast<unsigned>(absorb_count *
                                gpulsmopt_detail::kDsAbsorbChunks),
          block, 0, stream>>>(
          raw_or_null(seg_slow_seg_id_),
          raw_or_null(seg_slow_old_boundary_), raw_or_null(seg_dirty_live_),
          raw_or_null(seg_d_old_offset_), raw_or_null(seg_slow_in_begin_),
          raw_or_null(seg_slow_in_end_), absorb_count,
          delta_inc_keys_.data(), delta_inc_values_.data(),
          seg_old_key_.data(),
          seg_old_value_.data(), raw_or_null(pool_keys_),
          raw_or_null(pool_values_), raw_or_null(pool_valid_),
          raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_),
          raw_or_null(seg_bucket_value_sum_),
          raw_or_null(seg_d_out_value_sum_));
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpyAsync(a_value_sum.data(),
                                 raw_or_null(seg_d_out_value_sum_),
                                 absorb_count * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
    }

    std::vector<std::uint32_t> o_boundary(outputs_total),
        o_value_sum(outputs_total);
    if (split_count > 0) {
      std::vector<std::uint32_t> o_src(outputs_total),
          o_local(outputs_total), o_kvec(outputs_total),
          o_old_live(outputs_total), o_old_bound(outputs_total);
      std::size_t o = 0;
      for (std::size_t s = 0; s < split_count; ++s) {
        const std::uint32_t ord = split[s].ord;
        for (std::uint32_t local = 0; local < split_k[s]; ++local, ++o) {
          o_src[o] = static_cast<std::uint32_t>(absorb_count + s);
          o_local[o] = local;
          o_kvec[o] = split_k[s];
          o_old_live[o] = bucket_live[absorb_count + s];
          o_old_bound[o] = h_dir_boundary_[ord];
        }
      }
      upload_vec(seg_d_out_dirty_, o_src, stream);
      upload_vec(seg_d_out_local_, o_local, stream);
      upload_vec(seg_d_out_k_, o_kvec, stream);
      upload_vec(seg_d_output_seg_id_, child_seg, stream);
      gpulsmopt_detail::delta_reset_kernel<<<
          static_cast<unsigned>(outputs_total), block, 0, stream>>>(
          raw_or_null(seg_d_output_seg_id_), outputs_total,
          raw_or_null(seg_delta_active_), raw_or_null(seg_delta_count_),
          raw_or_null(seg_delta_prefix_));
      CUDA_CHECK(cudaGetLastError());
      upload_vec(seg_src_out_base_, o_old_live, stream);
      upload_vec(seg_src_seg_id_, o_old_bound, stream);
      resize_reuse(seg_d_out_boundary_, outputs_total);
      resize_reuse(seg_new_value_sum_, outputs_total);
      gpulsmopt_detail::ds_split_write_kernel<<<
          static_cast<unsigned>(outputs_total), block, 0, stream>>>(
          raw_or_null(seg_d_out_dirty_), raw_or_null(seg_d_out_local_),
          raw_or_null(seg_d_out_k_), raw_or_null(seg_d_output_seg_id_),
          raw_or_null(seg_src_out_base_), raw_or_null(seg_src_seg_id_),
          outputs_total, raw_or_null(seg_slow_in_begin_),
          raw_or_null(seg_slow_in_end_), raw_or_null(seg_d_old_offset_),
          delta_inc_keys_.data(), delta_inc_values_.data(),
          seg_old_key_.data(),
          seg_old_value_.data(), raw_or_null(pool_keys_),
          raw_or_null(pool_values_), raw_or_null(pool_valid_),
          raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_),
          raw_or_null(seg_bucket_value_sum_),
          raw_or_null(seg_d_out_boundary_),
          raw_or_null(seg_new_value_sum_));
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpyAsync(o_boundary.data(),
                                 raw_or_null(seg_d_out_boundary_),
                                 outputs_total * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(o_value_sum.data(),
                                 raw_or_null(seg_new_value_sum_),
                                 outputs_total * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (std::size_t i = 0; i < absorb_count; ++i) {
      const std::uint32_t ord = absorb[i].ord;
      h_dir_live_[ord] = a_total[i];
      h_dir_value_sum_[ord] = a_value_sum[i];
    }
    if (split_count > 0) {
      // Splice split children into the directory.
      std::vector<std::uint32_t> new_seg_id, new_boundary, new_live,
          new_value_sum;
      new_seg_id.reserve(dir_count + outputs_total);
      new_boundary.reserve(dir_count + outputs_total);
      new_live.reserve(dir_count + outputs_total);
      new_value_sum.reserve(dir_count + outputs_total);
      std::size_t s = 0, o = 0;
      for (std::size_t ord = 0; ord < dir_count; ++ord) {
        if (s < split_count && split[s].ord == ord) {
          const std::uint32_t total = h_dir_live_[ord] + split[s].inc_len;
          const std::uint32_t per =
              gpulsmopt_detail::ceil_div_u32(total, split_k[s]);
          for (std::uint32_t local = 0; local < split_k[s]; ++local, ++o) {
            const std::uint32_t begin = local * per;
            const std::uint32_t end =
                begin + per < total ? begin + per : total;
            new_seg_id.push_back(child_seg[o]);
            new_boundary.push_back(o_boundary[o]);
            new_live.push_back(end - begin);
            new_value_sum.push_back(o_value_sum[o]);
          }
          ++s;
          continue;
        }
        new_seg_id.push_back(h_dir_seg_id_[ord]);
        new_boundary.push_back(h_dir_boundary_[ord]);
        new_live.push_back(h_dir_live_[ord]);
        new_value_sum.push_back(h_dir_value_sum_[ord]);
      }
      h_dir_seg_id_ = std::move(new_seg_id);
      h_dir_boundary_ = std::move(new_boundary);
      h_dir_live_ = std::move(new_live);
      h_dir_value_sum_ = std::move(new_value_sum);
      recompute_sheet_live_count();
      upload_directory(stream);
    } else {
      rebuild_direct_bucket_prefixes(raw_or_null(seg_slow_seg_id_),
                                     absorb_count, stream);
      recompute_sheet_live_count();
      upload_directory_metadata(stream);
    }
    // Repack invalidates bucket routing metadata.
    flat_route_dirty_ = true;
  }

  void run_absorb_consolidations_device(
      const std::uint32_t *ord, const std::uint32_t *batch_begin,
      const std::uint32_t *batch_end, std::size_t absorb_count,
      const std::uint32_t *batch_keys, const std::uint32_t *batch_values,
      cudaStream_t stream) {
    if (absorb_count == 0)
      return;
    const int block = 256;
    prepare_absorb_kernel();
    const std::size_t shared_bytes = absorb_shared_bytes();
    gpulsmopt_detail::delta_absorb_fused_kernel<<<
        static_cast<unsigned>(absorb_count), block, shared_bytes, stream>>>(
        ord, batch_begin, batch_end, absorb_count, batch_keys, batch_values,
        raw_or_null(d_dir_seg_id_), raw_or_null(d_dir_boundary_),
        raw_or_null(d_dir_live_), raw_or_null(d_dir_value_sum_),
        raw_or_null(delta_keys_), raw_or_null(delta_values_),
        raw_or_null(seg_delta_active_), raw_or_null(seg_delta_count_),
        raw_or_null(seg_delta_prefix_), raw_or_null(pool_keys_),
        raw_or_null(pool_values_), raw_or_null(seg_bucket_max_),
        raw_or_null(seg_bucket_live_),
        raw_or_null(seg_bucket_value_sum_),
        raw_or_null(seg_bucket_live_prefix_),
        raw_or_null(seg_bucket_value_prefix_));
    CUDA_CHECK(cudaGetLastError());
    host_metadata_dirty_ = true;
    flat_route_dirty_ = true;
  }

  // Consolidate deltas before whole-segment gathers.
  void consolidate_all_deltas(cudaStream_t stream) {
    const std::size_t dir_count = h_dir_seg_id_.size();
    if (dir_count == 0)
      return;
    ds_base_.resize_discard(dir_count);
    const int block = 256;
    const int grid = static_cast<int>((dir_count + block - 1) / block);
    gpulsmopt_detail::delta_gather_counts_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(d_dir_seg_id_), dir_count,
        raw_or_null(seg_delta_count_), ds_base_.data());
    CUDA_CHECK(cudaGetLastError());
    std::vector<std::uint32_t> counts(dir_count);
    CUDA_CHECK(cudaMemcpyAsync(counts.data(), ds_base_.data(),
                               dir_count * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<DeltaConsolidation> entries;
    for (std::size_t ord = 0; ord < dir_count; ++ord) {
      if (counts[ord] > 0)
        entries.push_back({static_cast<std::uint32_t>(ord), 0u, 0u,
                           counts[ord]});
    }
    // Empty slices still require valid batch pointers.
    run_consolidations(entries, direct_sort_keys_.data(),
                       direct_sort_values_.data(), stream);
  }

  bool try_scatter_sheet_insert(const std::uint32_t *keys_in,
                                const std::uint32_t *values_in,
                                std::uint8_t op, std::size_t count,
                                cudaStream_t stream) {
    const std::size_t threshold = GPULSMOPT_SCATTER_MIN_BATCH;
    if (op != gpulsmopt_detail::kInsert || values_in == nullptr ||
        count < threshold ||
        count > std::numeric_limits<std::uint32_t>::max())
      return false;
    if (h_dir_seg_id_.empty())
      return false;
    {
      GPULSMOPT_PROF_PHASE(prof_delta_sort_ms_);
      sort_direct_batch(keys_in, values_in, count, stream);
    }
    sheet_delta_insert(direct_sort_keys_.data(), direct_sort_values_.data(),
                       count, stream);
    return true;
  }

  // Insert a sorted batch through segment deltas.
  void sheet_delta_insert(const std::uint32_t *sorted_keys,
                          const std::uint32_t *sorted_values,
                          std::size_t count, cudaStream_t stream) {
    const std::size_t dir_count = h_dir_seg_id_.size();
    if (dir_count == 0 || count == 0)
      return;
    ds_meta_.resize_discard(2);
    ds_slow_list_.resize_discard(dir_count);
    ds_residue_.resize_discard(dir_count);
    ds_base_.resize_discard(3 * dir_count + 1);
    ds_window_.resize_discard(3 * dir_count);
    std::uint32_t *counters = ds_meta_.data();
    std::uint32_t *cuts = ds_base_.data();
    std::uint32_t *absorb_begin = cuts + dir_count + 1;
    std::uint32_t *absorb_end = absorb_begin + dir_count;
    std::uint32_t *split_begin = ds_window_.data();
    std::uint32_t *split_end = split_begin + dir_count;
    std::uint32_t *split_dcount = split_end + dir_count;

    const int block = 256;
    std::uint32_t counters_host[2] = {0u, 0u};
    {
      GPULSMOPT_PROF_PHASE(prof_delta_ingest_ms_);
      CUDA_CHECK(
          cudaMemsetAsync(counters, 0, 2 * sizeof(std::uint32_t), stream));
      const int slice_grid =
          static_cast<int>((dir_count + block - 1) / block);
      gpulsmopt_detail::delta_slice_kernel<<<slice_grid, block, 0, stream>>>(
          sorted_keys, count, raw_or_null(d_dir_boundary_), dir_count, cuts);
      CUDA_CHECK(cudaGetLastError());
      gpulsmopt_detail::delta_append_kernel<<<
          static_cast<unsigned>(dir_count), block, 0, stream>>>(
          sorted_keys, sorted_values, cuts, raw_or_null(d_dir_seg_id_),
          dir_count, raw_or_null(delta_keys_), raw_or_null(delta_values_),
          raw_or_null(seg_delta_active_), raw_or_null(seg_delta_count_),
          raw_or_null(seg_delta_fence_), raw_or_null(seg_delta_prefix_),
          raw_or_null(d_dir_live_), raw_or_null(d_dir_value_sum_),
          ds_slow_list_.data(), absorb_begin, absorb_end, ds_residue_.data(),
          split_begin, split_end, split_dcount, counters);
      CUDA_CHECK(cudaGetLastError());
    }
    {
      CUDA_CHECK(cudaMemcpyAsync(counters_host, counters,
                                 2 * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const std::uint32_t n_absorb = counters_host[0];
    const std::uint32_t n_split = counters_host[1];
    host_metadata_dirty_ = true;
    overlay_dirty_ = true;
    read_view_dirty_ = true;
#ifdef GPULSMOPT_PROFILE_INSERT
    printf("[prof]     scatter: absorb=%u split=%u (incoming=%zu)\n",
           n_absorb, n_split, count);
#endif
    if (n_absorb > 0) {
      GPULSMOPT_PROF_PHASE(prof_delta_consolidate_ms_);
      run_absorb_consolidations_device(
          ds_slow_list_.data(), absorb_begin, absorb_end, n_absorb,
          sorted_keys, sorted_values, stream);
    }
    if (n_split > 0) {
      GPULSMOPT_PROF_PHASE(prof_delta_consolidate_ms_);
      std::vector<std::uint32_t> h_split_ord(n_split);
      std::vector<std::uint32_t> h_split_begin(n_split);
      std::vector<std::uint32_t> h_split_end(n_split);
      std::vector<std::uint32_t> h_split_dcount(n_split);
      const std::size_t bytes = n_split * sizeof(std::uint32_t);
      CUDA_CHECK(cudaMemcpyAsync(h_split_ord.data(), ds_residue_.data(), bytes,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_split_begin.data(), split_begin, bytes,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_split_end.data(), split_end, bytes,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_split_dcount.data(), split_dcount, bytes,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      std::vector<DeltaConsolidation> entries(n_split);
      for (std::uint32_t i = 0; i < n_split; ++i) {
        entries[i] = {h_split_ord[i], h_split_begin[i],
                      h_split_end[i] - h_split_begin[i],
                      h_split_dcount[i]};
      }
      run_consolidations(entries, sorted_keys, sorted_values, stream);
      return;
    }
    sheet_live_count_ += count;
    rebuild_directory_prefixes_device(stream);
  }

  bool try_scatter_sheet_delete(const std::uint32_t *keys_in,
                                std::size_t count, cudaStream_t stream) {
    const std::size_t threshold = GPULSMOPT_SCATTER_MIN_BATCH;
    if (count < threshold ||
        count > std::numeric_limits<std::uint32_t>::max())
      return false;
    if (h_dir_seg_id_.empty())
      return false;
    // Drain overlays before deleting from the Sheet.
    if (c0_log_count_ > 0 || !runs_.empty())
      merge_down(stream);
    ensure_host_directory_metadata(stream);
    const std::size_t dir_count = h_dir_seg_id_.size();
    const std::size_t total_buckets =
        dir_count * static_cast<std::size_t>(gpulsmopt_detail::kSegmentBuckets);

    ds_dest_.resize_discard(count);
    ds_cnt_cursor_.resize_discard(2 * total_buckets);
    ds_base_.resize_discard(total_buckets);
    ds_dirty_.resize_discard(std::min(count, total_buckets));
    ds_staged_keys_.resize_discard(count);
    ds_staged_values_.resize_discard(count);
    ds_meta_.resize_discard(3 * dir_count + 4);
    std::uint32_t *cnt = ds_cnt_cursor_.data();
    std::uint32_t *cursor = cnt + total_buckets;
    std::uint32_t *live_delta = ds_meta_.data() + dir_count;
    std::uint32_t *value_delta = live_delta + dir_count;
    std::uint32_t *counters = value_delta + dir_count;

    const int block = 256;
    const int key_grid = static_cast<int>((count + block - 1) / block);
    const std::size_t dirty_cap = std::min(count, total_buckets);
    {
      GPULSMOPT_PROF_PHASE(prof_route_ms_);
      CUDA_CHECK(cudaMemsetAsync(
          cnt, 0, 2 * total_buckets * sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(
          ds_meta_.data(), 0, (3 * dir_count + 4) * sizeof(std::uint32_t),
          stream));
      rebuild_flat_route(stream);
      gpulsmopt_detail::ds_route_kernel<<<key_grid, block, 0, stream>>>(
          keys_in, count, radix_first_.data(), flat_bucket_max_.data(),
          total_buckets, ds_dest_.data(), cnt, ds_dirty_.data(), counters);
      CUDA_CHECK(cudaGetLastError());
      std::size_t scan_bytes = 0;
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          nullptr, scan_bytes, cnt, ds_base_.data(),
          static_cast<int>(total_buckets), stream));
      ensure_sort_temp(scan_bytes);
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          sort_temp_storage_.data(), scan_bytes, cnt, ds_base_.data(),
          static_cast<int>(total_buckets), stream));
      gpulsmopt_detail::ds_scatter_kernel<<<key_grid, block, 0, stream>>>(
          keys_in, keys_in, count, ds_dest_.data(), ds_base_.data(),
          cursor, ds_staged_keys_.data(), ds_staged_values_.data());
      CUDA_CHECK(cudaGetLastError());
    }
    {
      GPULSMOPT_PROF_PHASE(prof_bucket_ms_);
      const std::size_t warps_per_block =
          block / gpulsmopt_detail::kBucketSlots;
      const int delete_grid = static_cast<int>(
          (dirty_cap + warps_per_block - 1) / warps_per_block);
      gpulsmopt_detail::ds_bucket_delete_kernel<<<delete_grid, block, 0,
                                                  stream>>>(
          ds_dirty_.data(), counters, cnt, ds_base_.data(),
          ds_staged_keys_.data(), raw_or_null(d_dir_seg_id_),
          raw_or_null(pool_keys_), raw_or_null(pool_values_),
          raw_or_null(pool_valid_), raw_or_null(seg_bucket_max_),
          raw_or_null(seg_bucket_live_), raw_or_null(seg_bucket_value_sum_),
          live_delta, value_delta);
      CUDA_CHECK(cudaGetLastError());
      // Remove matching entries from segment deltas.
      ds_slow_list_.resize_discard(dir_count);
      gpulsmopt_detail::delta_clear_dead_kernel<<<
          static_cast<unsigned>(dir_count), block, 0, stream>>>(
          raw_or_null(d_dir_seg_id_), dir_count,
          raw_or_null(seg_delta_active_), raw_or_null(seg_delta_count_),
          raw_or_null(delta_dead_));
      CUDA_CHECK(cudaGetLastError());
      gpulsmopt_detail::delta_mark_deleted_kernel<<<key_grid, block, 0,
                                                    stream>>>(
          keys_in, count, ds_dest_.data(),
          static_cast<std::uint32_t>(gpulsmopt_detail::kSegmentBuckets),
          raw_or_null(d_dir_seg_id_),
          raw_or_null(delta_keys_), raw_or_null(delta_values_),
          raw_or_null(seg_delta_active_), raw_or_null(seg_delta_fence_),
          raw_or_null(seg_delta_count_), raw_or_null(delta_dead_),
          ds_meta_.data(), ds_slow_list_.data(), live_delta, value_delta,
          counters);
      CUDA_CHECK(cudaGetLastError());
      gpulsmopt_detail::delta_compact_kernel<<<
          static_cast<unsigned>(dir_count), block, 0, stream>>>(
          ds_slow_list_.data(), counters, raw_or_null(d_dir_seg_id_),
          raw_or_null(delta_keys_), raw_or_null(delta_values_),
          raw_or_null(seg_delta_active_), raw_or_null(seg_delta_count_),
          raw_or_null(seg_delta_fence_), raw_or_null(seg_delta_prefix_),
          raw_or_null(delta_dead_));
      CUDA_CHECK(cudaGetLastError());
    }
    std::vector<std::uint32_t> meta_host(2 * dir_count + 1);
    CUDA_CHECK(cudaMemcpyAsync(meta_host.data(), live_delta,
                               (2 * dir_count + 1) * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::uint32_t num_dirty = meta_host[2 * dir_count];
    (void)num_dirty;
    // Bucket deletes invalidate routing metadata.
    flat_route_dirty_ = true;
    const std::uint32_t watermark = target_segment_live_ / 2;
    for (std::size_t ord = 0; ord < dir_count; ++ord) {
      const std::uint32_t live_delta_v = meta_host[ord];
      if (live_delta_v == 0)
        continue;
      h_dir_live_[ord] += live_delta_v;
      h_dir_value_sum_[ord] += meta_host[dir_count + ord];
      if (h_dir_live_[ord] < watermark)
        sheet_fragmented_ = true;
    }
    recompute_sheet_live_count();
    rebuild_all_bucket_prefixes(stream);
    upload_directory_metadata(stream);
    overlay_dirty_ = true;
    read_view_dirty_ = true;
#ifdef GPULSMOPT_PROFILE_INSERT
    printf("[prof]     scatter_delete: dirty_buckets=%u (batch=%zu)\n",
           num_dirty, count);
#endif
    if (sheet_fragmented_) {
      merge_underfull_segments(stream);
      sheet_fragmented_ = false;
    }
    return true;
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
      resize_reuse(ov_c0k_, s);
      resize_reuse(ov_c0v_, s);
      resize_reuse(ov_c0op_, s);
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
        resize_reuse(ov_mk_, total);
        resize_reuse(ov_mv_, total);
        resize_reuse(ov_mop_, total);
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
          raw_or_null(d_dir_radix_first_), raw_or_null(d_dir_seg_id_),
          h_dir_seg_id_.size(),
          raw_or_null(pool_keys_), raw_or_null(pool_values_),
          raw_or_null(pool_valid_), raw_or_null(seg_bucket_max_),
          raw_or_null(seg_bucket_live_),
          raw_or_null(seg_bucket_live_prefix_), raw_or_null(delta_keys_),
          raw_or_null(delta_values_), raw_or_null(seg_delta_active_),
          raw_or_null(seg_delta_fence_), raw_or_null(seg_delta_count_),
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
    ensure_host_directory_metadata(stream);
    auto policy = thrust::cuda::par.on(stream);
    resize_reuse(seg_inc_ordinal_, tomb);
    {
      const int block = 256;
      const int grid = static_cast<int>((tomb + block - 1) / block);
      gpulsmopt_detail::seg_route_keys_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(scratch_delete_keys_), tomb, raw_or_null(d_dir_boundary_),
          h_dir_seg_id_.size(), raw_or_null(seg_inc_ordinal_));
      CUDA_CHECK(cudaGetLastError());
    }
    resize_reuse(seg_dirty_ord_, tomb);
    resize_reuse(seg_dirty_count_, tomb);
    auto rle_end = thrust::reduce_by_key(
        policy, seg_inc_ordinal_.begin(), seg_inc_ordinal_.begin() + tomb,
        thrust::make_constant_iterator<std::uint32_t>(1u),
        seg_dirty_ord_.begin(), seg_dirty_count_.begin());
    const std::size_t m =
        static_cast<std::size_t>(rle_end.first - seg_dirty_ord_.begin());
    if (m == 0)
      return;
    prepare_dirty_plan(m, stream);
    resize_reuse(seg_inc_bucket_, tomb);
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
    resize_reuse(seg_dirty_bucket_, tomb);
    resize_reuse(seg_dirty_bucket_count_, tomb);
    auto bucket_rle_end = thrust::reduce_by_key(
        policy, seg_inc_bucket_.begin(), seg_inc_bucket_.begin() + tomb,
        thrust::make_constant_iterator<std::uint32_t>(1u),
        seg_dirty_bucket_.begin(), seg_dirty_bucket_count_.begin());
    const std::size_t mb = static_cast<std::size_t>(
        bucket_rle_end.first - seg_dirty_bucket_.begin());
    resize_reuse(seg_dirty_bucket_begin_, mb);
    thrust::exclusive_scan(policy, seg_dirty_bucket_count_.begin(),
                           seg_dirty_bucket_count_.begin() + mb,
                           seg_dirty_bucket_begin_.begin());
    resize_reuse(seg_new_live_, m);
    resize_reuse(seg_new_value_sum_, m);
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
    // Remove tombstoned entries from segment deltas.
    const std::size_t dir_count = h_dir_seg_id_.size();
    ds_meta_.resize_discard(3 * dir_count + 4);
    ds_slow_list_.resize_discard(dir_count);
    std::uint32_t *d_ord_flag = ds_meta_.data();
    std::uint32_t *d_live_delta = d_ord_flag + dir_count;
    std::uint32_t *d_value_delta = d_live_delta + dir_count;
    std::uint32_t *d_counters = d_value_delta + dir_count;
    CUDA_CHECK(cudaMemsetAsync(ds_meta_.data(), 0,
                               (3 * dir_count + 4) * sizeof(std::uint32_t),
                               stream));
    {
      const int block = 256;
      const int grid = static_cast<int>((tomb + block - 1) / block);
      gpulsmopt_detail::delta_clear_dead_kernel<<<
          static_cast<unsigned>(dir_count), block, 0, stream>>>(
          raw_or_null(d_dir_seg_id_), dir_count,
          raw_or_null(seg_delta_active_), raw_or_null(seg_delta_count_),
          raw_or_null(delta_dead_));
      CUDA_CHECK(cudaGetLastError());
      gpulsmopt_detail::delta_mark_deleted_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(scratch_delete_keys_), tomb,
          raw_or_null(seg_inc_ordinal_), 1u, raw_or_null(d_dir_seg_id_),
          raw_or_null(delta_keys_), raw_or_null(delta_values_),
          raw_or_null(seg_delta_active_), raw_or_null(seg_delta_fence_),
          raw_or_null(seg_delta_count_), raw_or_null(delta_dead_), d_ord_flag,
          ds_slow_list_.data(), d_live_delta, d_value_delta, d_counters);
      CUDA_CHECK(cudaGetLastError());
      gpulsmopt_detail::delta_compact_kernel<<<
          static_cast<unsigned>(dir_count), block, 0, stream>>>(
          ds_slow_list_.data(), d_counters, raw_or_null(d_dir_seg_id_),
          raw_or_null(delta_keys_), raw_or_null(delta_values_),
          raw_or_null(seg_delta_active_), raw_or_null(seg_delta_count_),
          raw_or_null(seg_delta_fence_), raw_or_null(seg_delta_prefix_),
          raw_or_null(delta_dead_));
      CUDA_CHECK(cudaGetLastError());
    }
    std::vector<std::uint32_t> delta_meta(2 * dir_count);
    CUDA_CHECK(cudaMemcpyAsync(delta_meta.data(), d_live_delta,
                               2 * dir_count * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
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
    for (std::size_t ord = 0; ord < dir_count; ++ord) {
      if (delta_meta[ord] == 0)
        continue;
      h_dir_live_[ord] += delta_meta[ord];
      h_dir_value_sum_[ord] += delta_meta[dir_count + ord];
      if (h_dir_live_[ord] < target_segment_live_ / 2)
        sheet_fragmented_ = true;
    }
    // Bucket deletes invalidate routing metadata.
    flat_route_dirty_ = true;
    recompute_sheet_live_count();
    rebuild_ordinal_bucket_prefixes(raw_or_null(seg_dirty_ord_), m, stream);
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
        resize_reuse(scratch_delete_keys_, tomb);
        thrust::copy(policy, gk.begin() + ins, gk.begin() + u,
                     scratch_delete_keys_.begin());
        apply_sheet_deletes(tomb, stream);
      }
      if (ins > 0) {
        GPULSMOPT_PROF_PHASE(prof_sheetmerge_ms_);
        resize_reuse(scratch_incoming_keys_, ins);
        resize_reuse(scratch_incoming_values_, ins);
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
          sheet_delta_insert(raw_or_null(scratch_incoming_keys_),
                             raw_or_null(scratch_incoming_values_), live,
                             stream);
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
      auto policy = thrust::cuda::par.on(stream);
      resize_reuse(c0_sorted_keys_, n);
      resize_reuse(c0_sorted_values_, n);
      resize_reuse(c0_sorted_ops_, n);
      thrust::copy(policy, c0_log_keys_.begin(), c0_log_keys_.begin() + n,
                   c0_sorted_keys_.begin());
      thrust::copy(policy, c0_log_values_.begin(), c0_log_values_.begin() + n,
                   c0_sorted_values_.begin());
      thrust::copy(policy, c0_log_ops_.begin(), c0_log_ops_.begin() + n,
                   c0_sorted_ops_.begin());
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
    resize_reuse(seg_inc_ordinal_, n);
    {
      const int block = 256;
      const int grid = static_cast<int>((n + block - 1) / block);
      gpulsmopt_detail::seg_route_keys_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(lookup_sorted_queries_), n, raw_or_null(d_dir_boundary_),
          h_dir_seg_id_.size(), raw_or_null(seg_inc_ordinal_));
      CUDA_CHECK(cudaGetLastError());
    }
    resize_reuse(seg_dirty_ord_, n);
    resize_reuse(seg_dirty_count_, n);
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
        raw_or_null(seg_bucket_live_), raw_or_null(delta_keys_),
        raw_or_null(delta_values_), raw_or_null(seg_delta_active_),
        raw_or_null(seg_delta_fence_), raw_or_null(seg_delta_count_));
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
          ro, rc, 0, raw_or_null(d_dir_boundary_),
          raw_or_null(d_dir_radix_first_), raw_or_null(d_dir_seg_id_),
          h_dir_seg_id_.size(),
          raw_or_null(pool_keys_),
          raw_or_null(pool_values_), raw_or_null(pool_valid_),
          raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_),
          raw_or_null(seg_bucket_live_prefix_), raw_or_null(delta_keys_),
          raw_or_null(delta_values_), raw_or_null(seg_delta_active_),
          raw_or_null(seg_delta_fence_), raw_or_null(seg_delta_count_));
      CUDA_CHECK(cudaGetLastError());
      return;
    }

    auto policy = thrust::cuda::par.on(stream);
    resize_reuse(lookup_sorted_queries_, n);
    resize_reuse(lookup_permutation_, n);
    resize_reuse(lookup_temp_values_, n);
    resize_reuse(lookup_temp_found_, n);
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
          raw_or_null(d_dir_boundary_), raw_or_null(d_dir_radix_first_),
          raw_or_null(d_dir_seg_id_), h_dir_seg_id_.size(),
          raw_or_null(pool_keys_),
          raw_or_null(pool_values_), raw_or_null(pool_valid_),
          raw_or_null(seg_bucket_max_), raw_or_null(seg_bucket_live_),
          raw_or_null(seg_bucket_live_prefix_), raw_or_null(delta_keys_),
          raw_or_null(delta_values_), raw_or_null(seg_delta_active_),
          raw_or_null(seg_delta_fence_), raw_or_null(seg_delta_count_));
      CUDA_CHECK(cudaGetLastError());
    }
  }

  std::size_t max_elements_ = 0;
  std::size_t batch_size_ = 0;
  std::uint32_t target_fill_ = GPULSMOPT_TARGET_FILL;
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
  double prof_route_ms_ = 0.0;
  double prof_bucket_ms_ = 0.0;
  double prof_window_ms_ = 0.0;
  double prof_delta_sort_ms_ = 0.0;
  double prof_delta_ingest_ms_ = 0.0;
  double prof_delta_consolidate_ms_ = 0.0;
  void reset_insert_prof_() {
    prof_append_ms_ = prof_flushsort_ms_ = prof_runmerge_ms_ = 0.0;
    prof_resolve_ms_ = prof_delete_ms_ = prof_sheetmerge_ms_ = 0.0;
    prof_route_ms_ = prof_bucket_ms_ = prof_window_ms_ = 0.0;
    prof_delta_sort_ms_ = prof_delta_ingest_ms_ = 0.0;
    prof_delta_consolidate_ms_ = 0.0;
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
  thrust::device_vector<std::uint32_t> seg_bucket_live_prefix_;
  thrust::device_vector<std::uint32_t> seg_bucket_value_prefix_;
  thrust::device_vector<std::uint32_t> delta_keys_;
  thrust::device_vector<std::uint32_t> delta_values_;
  thrust::device_vector<std::uint8_t> seg_delta_active_;
  thrust::device_vector<std::uint32_t> seg_delta_count_;
  thrust::device_vector<std::uint32_t> seg_delta_fence_;
  thrust::device_vector<std::uint32_t> seg_delta_prefix_;
  thrust::device_vector<std::uint8_t> delta_dead_;

  std::vector<std::uint32_t> h_dir_seg_id_;
  std::vector<std::uint32_t> h_dir_boundary_;
  std::vector<std::uint32_t> h_dir_live_;
  std::vector<std::uint32_t> h_dir_value_sum_;
  thrust::device_vector<std::uint32_t> d_dir_seg_id_;
  thrust::device_vector<std::uint32_t> d_dir_boundary_;
  thrust::device_vector<std::uint32_t> d_dir_live_;
  thrust::device_vector<std::uint32_t> d_dir_prefix_;
  thrust::device_vector<std::uint32_t> d_dir_value_sum_;
  thrust::device_vector<std::uint32_t> d_dir_value_prefix_;
  thrust::device_vector<std::uint32_t> d_dir_radix_first_;
  bool host_metadata_dirty_ = false;

  thrust::device_vector<std::uint32_t> scratch_incoming_keys_;
  thrust::device_vector<std::uint32_t> scratch_incoming_values_;
  thrust::device_vector<std::uint32_t> scratch_delete_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> direct_sort_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> direct_sort_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> sort_key_output_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> sort_payload_input_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> sort_payload_output_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> sort_temp_storage_;
  std::size_t direct_sort_count_ = 0;
  std::size_t direct_sort_temp_bytes_ = 0;
  std::size_t dir_scan_count_ = 0;
  std::size_t dir_scan_live_bytes_ = 0;
  std::size_t dir_scan_value_bytes_ = 0;

  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> radix_first_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> flat_bucket_max_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> delta_inc_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> delta_inc_values_;
  bool absorb_kernel_ready_ = false;
  bool flat_route_dirty_ = true;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> ds_dest_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> ds_cnt_cursor_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> ds_base_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> ds_dirty_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> ds_staged_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> ds_staged_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> ds_meta_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> ds_slow_list_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> ds_residue_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> ds_window_;

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

  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> seg_old_key_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> seg_old_value_;
  thrust::device_vector<std::uint32_t> seg_d_old_offset_;

  OverlayReadIndex cached_overlay_;
  bool overlay_dirty_ = true;
};
