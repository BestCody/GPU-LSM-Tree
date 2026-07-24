#pragma once
#include "gpu_dictionary_adapter.cuh"

#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_merge.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
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
      throw std::runtime_error(                                                \
          std::string(cudaGetErrorString(err__)) + " at " + __FILE__ + ":" +   \
          std::to_string(__LINE__) + " (" #stmt ")");                          \
    }                                                                          \
  } while (false)
#endif

namespace gpulsmopt_detail {
#ifndef GPULSMOPT_SM120_RADIX
#define GPULSMOPT_SM120_RADIX 1
#endif
#ifndef GPULSMOPT_RADIX_THREADS
#define GPULSMOPT_RADIX_THREADS 256
#endif
#ifndef GPULSMOPT_RADIX_ITEMS
#define GPULSMOPT_RADIX_ITEMS 22
#endif
#ifndef GPULSMOPT_RANGE_CDF_MAX_RATIO
#define GPULSMOPT_RANGE_CDF_MAX_RATIO 4
#endif
#ifndef GPULSMOPT_LOOKUP_RUN_PARALLEL_MAX_QUERIES
#define GPULSMOPT_LOOKUP_RUN_PARALLEL_MAX_QUERIES (1 << 16)
#endif
#ifndef GPULSMOPT_LOOKUP_RUN_PARALLEL_MIN_RUNS
#define GPULSMOPT_LOOKUP_RUN_PARALLEL_MIN_RUNS 8
#endif
#ifndef GPULSMOPT_NARROW_RANGE_MAX_QUERIES
#define GPULSMOPT_NARROW_RANGE_MAX_QUERIES 4096
#endif
#ifndef GPULSMOPT_PREWARM_LEAVES
#define GPULSMOPT_PREWARM_LEAVES 64
#endif
constexpr int kRunCapacity = 128;
constexpr std::size_t kCompactionTileRecords = std::size_t{1} << 22;
#ifdef GPULSMOPT_EPOCH_MAX
static_assert(GPULSMOPT_EPOCH_MAX == kRunCapacity,
              "GPULSMOPT_EPOCH_MAX must be 128");
#endif
constexpr int kEpochQuotientBits = 16;
constexpr int kEpochSubgroupBits = 4;
constexpr int kEpochQuotients = 1 << kEpochQuotientBits;
constexpr std::size_t kAdaptiveTransitionMaxRecords =
    2u * static_cast<std::size_t>(kEpochQuotients);
constexpr int kEpochSubgroups = 1 << kEpochSubgroupBits;
constexpr int kEpochSubgroupPlanes = kEpochSubgroupBits;
constexpr int kEpochSubgroupPrefixStride = kEpochSubgroups;
constexpr int kEpochHeavySortCap = 128;
constexpr int kRunStride = kRunCapacity;
constexpr int kGpuResidentWarps = 9024;
// Flat BaseRun rank directory.
constexpr int kBaseRank23Bits = 23;
constexpr int kBaseRank23Shift = 32 - kBaseRank23Bits;
constexpr std::size_t kBaseRank23Size = std::size_t{1} << kBaseRank23Bits;
constexpr std::size_t kSortedRunMinRecords = 1u << 22;
// Rank23 canonical fold geometry.
constexpr int kRawFoldWidth = 64;
constexpr int kStableFanout = 4;
constexpr int kColdArenaSlots = kStableFanout;
constexpr int kStableLevels = 16;
constexpr int kRank23BinsPerQuotient = 1 << (kBaseRank23Bits - kEpochQuotientBits);
constexpr std::uint32_t kRank23LocalBinMask = 0x7f;
static_assert(kRawFoldWidth == 64, "raw fold width must be 64");
static_assert(kStableFanout == 4, "stable fanout must be 4");
static_assert(kStableLevels <= 16, "stable levels bounded to 16");
static_assert(1 + kRawFoldWidth + 3 * kStableLevels + 1 <= kRunCapacity,
              "descriptor occupancy must fit run capacity");
static_assert(kRank23BinsPerQuotient == 128, "128 rank23 bins per quotient");
constexpr std::uint64_t kRangeCdfMaxRatio = GPULSMOPT_RANGE_CDF_MAX_RATIO;
static_assert(GPULSMOPT_RADIX_THREADS % 32 == 0,
              "radix block size must be warp aligned");
static_assert(kRunCapacity == 128, "run kernels require 128 physical slots");
constexpr std::uint32_t kEmptyKey = std::numeric_limits<std::uint32_t>::max();

constexpr std::uint32_t kInsert = 1;
constexpr std::uint32_t kTombstone = 0;

// Assignment runs avoid eager owner transitions.
enum class RunOperation : std::uint8_t { Insert, Delete };
// Raw = pending batch run; ColdStable = folded non-base run.
enum class AssignmentClass : std::uint8_t { Raw, ColdStable };
// Canonical overlay state per BaseRun position.
constexpr std::uint8_t kCanonBase = 0;     // immutable BaseRun value
constexpr std::uint8_t kCanonOverride = 1; // override value
constexpr std::uint8_t kCanonDead = 2;     // deleted
#if GPULSMOPT_SM120_RADIX
struct Sm120RadixPolicy
    : cub::DeviceRadixSortPolicy<std::uint32_t, std::uint32_t, std::uint32_t> {
  using Base =
      cub::DeviceRadixSortPolicy<std::uint32_t, std::uint32_t, std::uint32_t>;
  using BasePolicy = typename Base::Policy900;

  struct Policy1200 : cub::ChainedPolicy<1200, Policy1200, BasePolicy> {
    enum {
      PRIMARY_RADIX_BITS = 7,
      SINGLE_TILE_RADIX_BITS = 6,
      SEGMENTED_RADIX_BITS = 6,
      ONESWEEP = true,
      ONESWEEP_RADIX_BITS = 8,
    };

    using HistogramPolicy = typename BasePolicy::HistogramPolicy;
    using ExclusiveSumPolicy = typename BasePolicy::ExclusiveSumPolicy;
    using OnesweepPolicy = cub::AgentRadixSortOnesweepPolicy<
        GPULSMOPT_RADIX_THREADS, GPULSMOPT_RADIX_ITEMS, std::uint32_t, 1,
        cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY, cub::BLOCK_SCAN_RAKING_MEMOIZE,
        cub::RADIX_SORT_STORE_DIRECT, ONESWEEP_RADIX_BITS>;
    using ScanPolicy = typename BasePolicy::ScanPolicy;
    using DownsweepPolicy = typename BasePolicy::DownsweepPolicy;
    using AltDownsweepPolicy = typename BasePolicy::AltDownsweepPolicy;
    using UpsweepPolicy = typename BasePolicy::UpsweepPolicy;
    using AltUpsweepPolicy = typename BasePolicy::AltUpsweepPolicy;
    using SingleTilePolicy = typename BasePolicy::SingleTilePolicy;
    using SegmentedPolicy = typename BasePolicy::SegmentedPolicy;
    using AltSegmentedPolicy = typename BasePolicy::AltSegmentedPolicy;
  };

  using MaxPolicy = Policy1200;
};

inline cudaError_t
epoch_radix_sort_pairs(void *temp_storage, std::size_t &temp_bytes,
                       const std::uint32_t *keys_in, std::uint32_t *keys_out,
                       const std::uint32_t *values_in,
                       std::uint32_t *values_out, std::uint32_t count,
                       int begin_bit, int end_bit, cudaStream_t stream) {
  cub::DoubleBuffer<std::uint32_t> keys(const_cast<std::uint32_t *>(keys_in),
                                        keys_out);
  cub::DoubleBuffer<std::uint32_t> values(
      const_cast<std::uint32_t *>(values_in), values_out);
  return cub::DispatchRadixSort<
      false, std::uint32_t, std::uint32_t, std::uint32_t,
      Sm120RadixPolicy>::Dispatch(temp_storage, temp_bytes, keys, values, count,
                                  begin_bit, end_bit, false, stream);
}
#else
inline cudaError_t
epoch_radix_sort_pairs(void *temp_storage, std::size_t &temp_bytes,
                       const std::uint32_t *keys_in, std::uint32_t *keys_out,
                       const std::uint32_t *values_in,
                       std::uint32_t *values_out, std::uint32_t count,
                       int begin_bit, int end_bit, cudaStream_t stream) {
  return cub::DeviceRadixSort::SortPairs(temp_storage, temp_bytes, keys_in,
                                         keys_out, values_in, values_out, count,
                                         begin_bit, end_bit, stream);
}
#endif

#if defined(GPULSMOPT_PROFILE_INSERT) || defined(GPULSMOPT_PROFILE_FOLD)
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
  gpulsmopt_detail::ScopedInsertPhaseTimer GPULSMOPT_PROF_CAT(                 \
      prof_phase_, __LINE__)(stream, &(acc))
#else
#define GPULSMOPT_PROF_PHASE(acc)                                              \
  do {                                                                         \
  } while (false)
#endif
#ifndef GPULSMOPT_PROFILE_INSERT
#if defined(GPULSMOPT_PROFILE_FOLD)
#undef GPULSMOPT_PROF_PHASE
#define GPULSMOPT_PROF_PHASE(acc)                                              \
  do {                                                                         \
  } while (false)
#endif
#endif
// Fold-phase timing; diagnostic builds only (syncs per phase).
#ifdef GPULSMOPT_PROFILE_FOLD
#define GPULSMOPT_FOLD_PHASE(acc)                                              \
  gpulsmopt_detail::ScopedInsertPhaseTimer GPULSMOPT_PROF_CAT(                 \
      fold_phase_, __LINE__)(stream, &(acc))
#else
#define GPULSMOPT_FOLD_PHASE(acc)                                              \
  do {                                                                         \
  } while (false)
#endif

struct DeviceKeyBatch {
  const std::uint32_t *keys = nullptr;
  std::size_t count = 0;
  bool sorted = false;
};

template <class T> class RawDeviceBuffer {
public:
  RawDeviceBuffer() = default;
  RawDeviceBuffer(const RawDeviceBuffer &) = delete;
  RawDeviceBuffer &operator=(const RawDeviceBuffer &) = delete;
  RawDeviceBuffer(RawDeviceBuffer &&other) noexcept
      : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }
  RawDeviceBuffer &operator=(RawDeviceBuffer &&other) noexcept {
    if (this == &other)
      return *this;
    if (data_)
      cudaFree(data_);
    data_ = other.data_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
    return *this;
  }
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

  void resize_discard_exact(std::size_t count) {
    if (count > capacity_) {
      T *next = nullptr;
      CUDA_CHECK(
          cudaMalloc(reinterpret_cast<void **>(&next), count * sizeof(T)));
      if (data_)
        CUDA_CHECK(cudaFree(data_));
      data_ = next;
      capacity_ = count;
    }
    size_ = count;
  }

  void release() {
    if (data_)
      CUDA_CHECK(cudaFree(data_));
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
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

struct RunView {
  const std::uint32_t *keys;
  const std::uint32_t *quotient_off;
};

struct SortedRunView {
  const std::uint32_t *keys;
  const std::uint32_t *values;
  const std::uint32_t *rank23;
  std::size_t count;
  std::uint32_t unit_counts;
};

struct SortedRunRangeView {
  const std::uint32_t *cdf;
  std::uint32_t min_key;
  std::uint64_t span;
};


;

;









// Transposed reads keep adjacent accesses together.


__device__ inline void
sorted_search_bounds(const SortedRunView &sorted, std::uint32_t key,
                     std::size_t *begin, std::size_t *end) {
  const std::uint32_t bin = key >> kBaseRank23Shift;
  *begin = sorted.rank23[bin];
  *end = sorted.rank23[bin + 1];
}

__device__ inline bool sorted_find_value(const SortedRunView &sorted,
                                         std::uint32_t key,
                                         std::uint32_t *value) {
  if (sorted.count == 0)
    return false;
  std::size_t begin = 0;
  std::size_t end = 0;
  sorted_search_bounds(sorted, key, &begin, &end);
  const std::size_t position =
      begin + lower_bound_u32(sorted.keys + begin, end - begin, key);
  if (position >= end || sorted.keys[position] != key)
    return false;
  *value = sorted.values[position];
  return true;
}

__device__ inline void sorted_range_ranks(const SortedRunView &sorted,
                                          std::uint32_t lo, std::uint32_t hi,
                                          std::size_t *lower,
                                          std::size_t *upper) {
  const std::uint32_t lo_bin = lo >> kBaseRank23Shift;
  const std::uint32_t hi_bin = hi >> kBaseRank23Shift;
  const std::size_t lo_begin = sorted.rank23[lo_bin];
  const std::size_t lo_end = sorted.rank23[lo_bin + 1];
  const std::size_t hi_begin = sorted.rank23[hi_bin];
  const std::size_t hi_end = sorted.rank23[hi_bin + 1];
  *lower =
      lo_begin + lower_bound_u32(sorted.keys + lo_begin, lo_end - lo_begin, lo);
  *upper =
      hi_begin + upper_bound_u32(sorted.keys + hi_begin, hi_end - hi_begin, hi);
}

__device__ inline std::uint32_t
sorted_range_cdf_prefix(const SortedRunRangeView &range, std::uint32_t key,
                        bool upper) {
  if (key < range.min_key)
    return 0u;
  std::uint64_t index = static_cast<std::uint64_t>(key) - range.min_key +
                        static_cast<unsigned>(upper);
  if (index > range.span)
    index = range.span;
  return range.cdf[index];
}

__device__ inline std::uint32_t
sorted_range_count(const SortedRunView &sorted,
                   const std::uint32_t *count_prefix, std::uint32_t lo,
                   std::uint32_t hi) {
  if (sorted.count == 0)
    return 0u;
  std::size_t begin = 0, end = 0;
  sorted_range_ranks(sorted, lo, hi, &begin, &end);
  if (sorted.unit_counts)
    return static_cast<std::uint32_t>(end - begin);
  return count_prefix[end] - count_prefix[begin];
}

__device__ inline std::uint32_t
sorted_range_sum(const SortedRunView &sorted, const SortedRunRangeView &range,
                 const std::uint32_t *value_prefix, std::uint32_t lo,
                 std::uint32_t hi) {
  if (sorted.count == 0)
    return 0u;
  if (range.cdf) {
    return sorted_range_cdf_prefix(range, hi, true) -
           sorted_range_cdf_prefix(range, lo, false);
  }
  std::size_t begin = 0, end = 0;
  sorted_range_ranks(sorted, lo, hi, &begin, &end);
  return value_prefix[end] - value_prefix[begin];
}

// BaseRun plus the canonical overlay it is folded against.
struct CanonicalBaseView {
  SortedRunView base;
  const std::uint8_t *state;
  const std::uint32_t *override_values;
  const std::uint32_t *rank23_value_prefix;
  const std::uint32_t *rank23_count_prefix;
  std::uint8_t active;
};

// Locate a BaseRun key; return its position or base.count on miss.
__device__ inline std::size_t canonical_find_position(const SortedRunView &base,
                                                      std::uint32_t key) {
  if (base.count == 0)
    return 0;
  std::size_t begin = 0, end = 0;
  sorted_search_bounds(base, key, &begin, &end);
  const std::size_t p =
      begin + lower_bound_u32(base.keys + begin, end - begin, key);
  return (p < end && base.keys[p] == key) ? p : base.count;
}

// Visible value at a BaseRun position under the overlay.
__device__ inline std::uint32_t
canonical_value_at(const CanonicalBaseView &v, std::size_t p) {
  if (!v.active || v.state[p] == kCanonBase)
    return v.base.values[p];
  return v.override_values[p]; // kCanonOverride
}

// Visible liveness at a BaseRun position under the overlay.
__device__ inline bool canonical_live_at(const CanonicalBaseView &v,
                                         std::size_t p) {
  return !v.active || v.state[p] != kCanonDead;
}

// Overlay value correction at a BaseRun position (mod 2^32).
__device__ inline std::uint32_t
canonical_value_delta_at(const CanonicalBaseView &v, std::size_t p) {
  if (!v.active)
    return 0u;
  const std::uint8_t s = v.state[p];
  if (s == kCanonOverride)
    return v.override_values[p] - v.base.values[p];
  if (s == kCanonDead)
    return 0u - v.base.values[p];
  return 0u;
}

// Overlay count correction at a BaseRun position.
__device__ inline std::int32_t
canonical_count_delta_at(const CanonicalBaseView &v, std::size_t p) {
  return (v.active && v.state[p] == kCanonDead) ? -1 : 0;
}

// Find a BaseRun key and its visible state in one probe.
__device__ inline bool canonical_find_value(const CanonicalBaseView &v,
                                            std::uint32_t key,
                                            std::uint32_t *value) {
  const std::size_t p = canonical_find_position(v.base, key);
  if (p >= v.base.count)
    return false;
  if (!canonical_live_at(v, p))
    return false;
  *value = canonical_value_at(v, p);
  return true;
}

// Overlay value correction over one Rank23 bin within [lo,hi].
__device__ inline std::uint32_t
canonical_bin_value_delta(const CanonicalBaseView &v, std::uint32_t bin,
                          std::uint32_t lo, std::uint32_t hi) {
  const std::uint32_t begin = v.base.rank23[bin];
  const std::uint32_t end = v.base.rank23[bin + 1u];
  std::uint32_t s = 0u;
  for (std::uint32_t p = begin; p < end; ++p) {
    const std::uint32_t k = v.base.keys[p];
    if (k >= lo && k <= hi)
      s += canonical_value_delta_at(v, p);
  }
  return s;
}

// Overlay count correction over one Rank23 bin within [lo,hi].
__device__ inline std::int32_t
canonical_bin_count_delta(const CanonicalBaseView &v, std::uint32_t bin,
                          std::uint32_t lo, std::uint32_t hi) {
  const std::uint32_t begin = v.base.rank23[bin];
  const std::uint32_t end = v.base.rank23[bin + 1u];
  std::int32_t s = 0;
  for (std::uint32_t p = begin; p < end; ++p) {
    const std::uint32_t k = v.base.keys[p];
    if (k >= lo && k <= hi)
      s += canonical_count_delta_at(v, p);
  }
  return s;
}

// Overlay value correction over [lo,hi]: prefix interior + boundary.
__device__ inline std::uint32_t
canonical_range_value_delta(const CanonicalBaseView &v, std::uint32_t lo,
                            std::uint32_t hi) {
  if (!v.active)
    return 0u;
  const std::uint32_t lo_bin = lo >> kBaseRank23Shift;
  const std::uint32_t hi_bin = hi >> kBaseRank23Shift;
  if (lo_bin == hi_bin)
    return canonical_bin_value_delta(v, lo_bin, lo, hi);
  std::uint32_t s = canonical_bin_value_delta(v, lo_bin, lo, hi) +
                    canonical_bin_value_delta(v, hi_bin, lo, hi);
  if (hi_bin > lo_bin + 1u)
    s += v.rank23_value_prefix[hi_bin] - v.rank23_value_prefix[lo_bin + 1u];
  return s;
}

// Overlay count correction over [lo,hi].
__device__ inline std::int32_t
canonical_range_count_delta(const CanonicalBaseView &v, std::uint32_t lo,
                            std::uint32_t hi) {
  if (!v.active)
    return 0;
  const std::uint32_t lo_bin = lo >> kBaseRank23Shift;
  const std::uint32_t hi_bin = hi >> kBaseRank23Shift;
  if (lo_bin == hi_bin)
    return canonical_bin_count_delta(v, lo_bin, lo, hi);
  std::int32_t s = canonical_bin_count_delta(v, lo_bin, lo, hi) +
                   canonical_bin_count_delta(v, hi_bin, lo, hi);
  if (hi_bin > lo_bin + 1u)
    s += static_cast<std::int32_t>(v.rank23_count_prefix[hi_bin] -
                                   v.rank23_count_prefix[lo_bin + 1u]);
  return s;
}

// Canonical base range sum: immutable BaseRun sum + overlay delta.
__device__ inline std::uint32_t
canonical_range_sum(const CanonicalBaseView &v, const SortedRunRangeView &range,
                    const std::uint32_t *base_value_prefix, std::uint32_t lo,
                    std::uint32_t hi) {
  return sorted_range_sum(v.base, range, base_value_prefix, lo, hi) +
         canonical_range_value_delta(v, lo, hi);
}

// Canonical base range count: BaseRun count + overlay delta.
__device__ inline std::uint32_t
canonical_range_count(const CanonicalBaseView &v,
                      const std::uint32_t *base_count_prefix, std::uint32_t lo,
                      std::uint32_t hi) {
  return sorted_range_count(v.base, base_count_prefix, lo, hi) +
         static_cast<std::uint32_t>(canonical_range_count_delta(v, lo, hi));
}

// Shared fold record; source_rank indexes the 64 chronological views.
struct FoldRecord {
  std::uint16_t low_key;
  std::uint8_t operation;
  std::uint8_t source_rank;
  std::uint32_t value;
  std::uint32_t source_position;
};



__global__ void sorted_range_cdf_scatter_kernel(const std::uint32_t *keys,
                                                const std::uint32_t *values,
                                                std::size_t count,
                                                std::uint32_t min_key,
                                                std::uint32_t *cdf) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::uint64_t slot = static_cast<std::uint64_t>(keys[i]) - min_key + 1u;
  cdf[slot] = values[i];
}

struct TakeLastU32 {
  __host__ __device__ std::uint32_t operator()(std::uint32_t,
                                               std::uint32_t newer) const {
    return newer;
  }
};


// Scalar op for a leaf, or one packed bit for a mixed run.
__device__ inline int assignment_op_at(const std::uint32_t *op_words,
                                       std::uint8_t constant_op,
                                       std::uint8_t mixed, std::uint32_t p) {
  if (!mixed)
    return constant_op;
  return (op_words[p >> 5] >> (p & 31u)) & 1u;
}

// Successor over the BaseRun alone: every base key is live.
__global__ void base_successor_kernel(CanonicalBaseView base,
                                      const std::uint32_t *queries,
                                      std::size_t query_count,
                                      std::uint32_t *results) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= query_count)
    return;
  if (base.base.count == 0) {
    results[i] = 0u;
    return;
  }
  const std::uint32_t key = queries[i];
  std::size_t begin = 0;
  std::size_t end = 0;
  sorted_search_bounds(base.base, key, &begin, &end);
  // Empty bins start at the next nonempty bin.
  std::size_t position =
      begin + lower_bound_u32(base.base.keys + begin, end - begin, key);
  // Skip canonical-dead BaseRun positions.
  while (position < base.base.count && !canonical_live_at(base, position))
    ++position;
  results[i] = position < base.base.count ? base.base.keys[position] : 0u;
}

// Immutable assignment run seen by device readers.
struct AssignmentRunView {
  const std::uint32_t *keys;
  const std::uint32_t *values;
  const std::uint32_t *offsets;
  const std::uint32_t *page_counts; // null unless paged
  const std::uint32_t *op_words;
  std::uint8_t constant_op;      // 1 insert, 0 delete
  std::uint8_t mixed;            // 1 if op_words is used
  std::uint8_t paged;            // 1 if cold paged output
};

// Quotient record range: packed offsets, or paged begin+count.
__device__ inline void assignment_bounds(const AssignmentRunView &run,
                                         std::uint32_t q,
                                         std::uint32_t *begin,
                                         std::uint32_t *end) {
  const std::uint32_t b = run.offsets[q];
  *begin = b;
  *end = run.paged ? b + run.page_counts[q] : run.offsets[q + 1u];
}

struct PinnedHostState {
  AssignmentRunView views[kRunCapacity];
  std::uint32_t narrow_overflow;
  std::uint32_t resolved_count;
  std::uint32_t successor_miss_count;
  std::uint32_t gathered_count;
};

__global__ void assignment_group_count_kernel(
    const AssignmentRunView *runs, int run_count,
    std::uint32_t *counts) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kEpochQuotients)
    return;
  if (q == kEpochQuotients) {
    counts[q] = 0u;
    return;
  }
  std::uint32_t total = 0u;
  for (int r = 0; r < run_count; ++r) {
    std::uint32_t b, e;
    assignment_bounds(runs[r], q, &b, &e);
    total += e - b;
  }
  counts[q] = total;
}

__global__ void assignment_group_gather_kernel(
    const AssignmentRunView *runs, int run_count,
    const std::uint32_t *offsets, std::uint32_t *out_keys,
    std::uint64_t *out_payload) {
  const std::uint32_t q = blockIdx.x;
  if (q >= kEpochQuotients)
    return;
  std::uint32_t cursor = offsets[q];
  for (int r = 0; r < run_count; ++r) {
    const AssignmentRunView run = runs[r];
    std::uint32_t begin, end;
    assignment_bounds(run, q, &begin, &end);
    const std::uint32_t count = end - begin;
    for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
      const std::uint32_t p = begin + i;
      const int op =
          assignment_op_at(run.op_words, run.constant_op, run.mixed, p);
      const std::uint32_t value = op != 0 ? run.values[p] : 0u;
      out_keys[cursor + i] = run.keys[p];
      out_payload[cursor + i] =
          (static_cast<std::uint64_t>(value) << 32) |
          static_cast<std::uint64_t>(op != 0);
    }
    cursor += count;
  }
}

__global__ void assignment_group_gather_range_kernel(
    const AssignmentRunView *runs, int run_count,
    std::uint32_t first_quotient, const std::uint32_t *tile_offsets,
    std::uint32_t *out_keys, std::uint64_t *out_payload) {
  const std::uint32_t local_q = blockIdx.x;
  const std::uint32_t q = first_quotient + local_q;
  std::uint32_t cursor = tile_offsets[local_q];
  for (int r = 0; r < run_count; ++r) {
    const AssignmentRunView run = runs[r];
    std::uint32_t begin, end;
    assignment_bounds(run, q, &begin, &end);
    const std::uint32_t count = end - begin;
    for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
      const std::uint32_t p = begin + i;
      const int op =
          assignment_op_at(run.op_words, run.constant_op, run.mixed, p);
      const std::uint32_t value = op != 0 ? run.values[p] : 0u;
      out_keys[cursor + i] = run.keys[p];
      out_payload[cursor + i] =
          (static_cast<std::uint64_t>(value) << 32) |
          static_cast<std::uint64_t>(op != 0);
    }
    cursor += count;
  }
}

__global__ void compaction_unique_count_kernel(
    const std::uint32_t *keys, std::size_t count,
    std::uint32_t *quotient_counts) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::uint32_t key = keys[i];
  if (i + 1u < count && keys[i + 1u] == key)
    return;
  atomicAdd(quotient_counts + (key >> kEpochQuotientBits), 1u);
}

__global__ void compaction_tile_offsets_kernel(
    const std::uint32_t *global_offsets, std::uint32_t first,
    std::uint32_t segments, std::uint32_t *tile_offsets) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > segments)
    return;
  tile_offsets[i] = global_offsets[first + i] - global_offsets[first];
}

__global__ void compaction_unique_scatter_kernel(
    const std::uint32_t *keys, const std::uint64_t *payload,
    std::size_t count, std::uint32_t *quotient_cursors,
    std::uint32_t *out_keys, std::uint32_t *out_values,
    std::uint32_t *out_op_words) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::uint32_t key = keys[i];
  if (i + 1u < count && keys[i + 1u] == key)
    return;
  const std::uint32_t q = key >> kEpochQuotientBits;
  const std::uint32_t out = atomicAdd(quotient_cursors + q, 1u);
  const std::uint64_t packed = payload[i];
  out_keys[out] = key;
  out_values[out] = static_cast<std::uint32_t>(packed >> 32);
  if (packed & 1u)
    atomicOr(out_op_words + (out >> 5), 1u << (out & 31u));
}

// Scan newest first, then fall through to the canonical base.
__global__ void temporal_lookup_kernel(const AssignmentRunView *runs,
                                       int run_count, CanonicalBaseView base,
                                       const std::uint32_t *queries,
                                       std::size_t n,
                                       std::uint32_t *out_values,
                                       std::uint8_t *out_found) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint32_t key = queries[i];
  const std::uint32_t q = key >> kEpochQuotientBits;
  for (int r = run_count - 1; r >= 0; --r) {
    const AssignmentRunView run = runs[r];
    std::uint32_t begin, position;
    assignment_bounds(run, q, &begin, &position);
    while (position-- > begin) {
      if (run.keys[position] != key)
        continue;
      const bool live = assignment_op_at(
                            run.op_words, run.constant_op,
                            run.mixed, position) != 0;
      out_values[i] = live ? run.values[position] : kEmptyKey;
      if (out_found)
        out_found[i] = live ? 1u : 0u;
      return;
    }
  }
  std::uint32_t value = 0u;
  if (canonical_find_value(base, key, &value)) {
    out_values[i] = value;
    if (out_found)
      out_found[i] = 1u;
    return;
  }
  out_values[i] = kEmptyKey;
  if (out_found)
    out_found[i] = 0u;
}

// Sparse successor state for the base and corrections.
struct SuccessorSparseView {
  SortedRunView base;
  const std::uint32_t *deleted_base_words;
  const std::uint32_t *base_live_l1;
  const std::uint32_t *base_live_l2;
  const std::uint32_t *base_live_l3;
  const std::uint32_t *correction_keys;
  const std::uint32_t *correction_offsets;
  const std::uint32_t *positive_words;
  const std::uint32_t *positive_l1;
  const std::uint32_t *positive_l2;
  const std::uint32_t *positive_l3;
  std::uint32_t correction_count;
  std::uint32_t base_l0_words;
  std::uint32_t base_l3_words;
  std::uint32_t positive_l0_words;
  std::uint32_t positive_l3_words;
};

__device__ inline std::uint32_t succ_ffs0(std::uint32_t bits) {
  return static_cast<std::uint32_t>(__ffs(static_cast<int>(bits))) - 1u;
}

// Bits at or above `bit`.
__device__ inline std::uint32_t succ_mask_from(std::uint32_t bit) {
  return 0xffffffffu << bit;
}

// Bits strictly above `bit` (bit 31 leaves nothing).
__device__ inline std::uint32_t succ_mask_above(std::uint32_t bit) {
  return bit >= 31u ? 0u : (0xffffffffu << (bit + 1u));
}

// Finds the next nonempty L0 word.
__device__ inline std::uint32_t
successor_next_set_word(const std::uint32_t *l1,
                        const std::uint32_t *l2,
                        const std::uint32_t *l3,
                        std::uint32_t l0_words,
                        std::uint32_t l3_words,
                        std::uint32_t w) {
  if (w >= l0_words)
    return l0_words;
  std::uint32_t i1 = w >> 5;
  std::uint32_t b1 = l1[i1] & succ_mask_from(w & 31u);
  if (b1)
    return (i1 << 5) | succ_ffs0(b1);
  std::uint32_t i2 = i1 >> 5;
  std::uint32_t b2 = l2[i2] & succ_mask_above(i1 & 31u);
  if (!b2) {
    std::uint32_t i3 = i2 >> 5;
    std::uint32_t b3 = l3[i3] & succ_mask_above(i2 & 31u);
    while (!b3) {
      if (++i3 >= l3_words)
        return l0_words;
      b3 = l3[i3];
    }
    i2 = (i3 << 5) | succ_ffs0(b3);
    b2 = l2[i2];
  }
  i1 = (i2 << 5) | succ_ffs0(b2);
  b1 = l1[i1];
  return (i1 << 5) | succ_ffs0(b1);
}

// Finds the first surviving BaseRun position.
__device__ inline std::size_t
successor_first_base_live(const SuccessorSparseView &view,
                          std::size_t position) {
  if (position >= view.base.count)
    return view.base.count;
  std::uint32_t w = static_cast<std::uint32_t>(position >> 5);
  std::uint32_t bits = ~view.deleted_base_words[w] &
                       succ_mask_from(static_cast<std::uint32_t>(position) & 31u);
  if (!bits) {
    w = successor_next_set_word(
        view.base_live_l1, view.base_live_l2, view.base_live_l3,
        view.base_l0_words, view.base_l3_words, w + 1u);
    if (w >= view.base_l0_words)
      return view.base.count;
    bits = ~view.deleted_base_words[w];
  }
  const std::size_t found =
      (static_cast<std::size_t>(w) << 5) | succ_ffs0(bits);
  return found < view.base.count ? found : view.base.count;
}

// Finds the first positive correction position.
__device__ inline std::size_t
successor_first_positive(const SuccessorSparseView &view,
                         std::size_t position) {
  if (position >= view.correction_count)
    return view.correction_count;
  std::uint32_t w = static_cast<std::uint32_t>(position >> 5);
  std::uint32_t bits =
      view.positive_words[w] &
      succ_mask_from(static_cast<std::uint32_t>(position) & 31u);
  if (!bits) {
    w = successor_next_set_word(
        view.positive_l1, view.positive_l2, view.positive_l3,
        view.positive_l0_words, view.positive_l3_words, w + 1u);
    if (w >= view.positive_l0_words)
      return view.correction_count;
    bits = view.positive_words[w];
  }
  const std::size_t found =
      (static_cast<std::size_t>(w) << 5) | succ_ffs0(bits);
  return found < view.correction_count ? found
                                       : view.correction_count;
}

// Resolves hits and compacts misses once per block.
__global__ void successor_live_or_miss_kernel(
    const AssignmentRunView *runs, int run_count, CanonicalBaseView base,
    const std::uint32_t *queries, std::size_t query_count,
    std::uint32_t *results, std::uint32_t *miss_indices,
    std::uint32_t *miss_count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  bool missing = false;
  if (i < query_count) {
    const std::uint32_t key = queries[i];
    const std::uint32_t q = key >> kEpochQuotientBits;
    int live = -1;
    for (int r = run_count - 1; r >= 0 && live < 0; --r) {
      const AssignmentRunView run = runs[r];
      std::uint32_t begin, position;
      assignment_bounds(run, q, &begin, &position);
      while (position-- > begin) {
        if (run.keys[position] != key)
          continue;
        live = assignment_op_at(run.op_words, run.constant_op, run.mixed,
                                position) != 0
                   ? 1
                   : 0;
        break;
      }
    }
    if (live < 0) {
      std::uint32_t value = 0u;
      live = canonical_find_value(base, key, &value) ? 1 : 0;
    }
    if (live > 0)
      results[i] = key;
    else
      missing = true;
  }
  constexpr unsigned kWarps = 8u;
  __shared__ unsigned warp_counts[kWarps];
  __shared__ unsigned warp_offsets[kWarps];
  __shared__ unsigned block_slot;
  const unsigned lane = threadIdx.x & 31u;
  const unsigned warp = threadIdx.x >> 5;
  const unsigned mask = __ballot_sync(0xffffffffu, missing);
  const unsigned warp_count = static_cast<unsigned>(__popc(mask));
  if (lane == 0u)
    warp_counts[warp] = warp_count;
  if (!__syncthreads_or(missing))
    return;
  if (warp == 0u) {
    const unsigned value = lane < kWarps ? warp_counts[lane] : 0u;
    unsigned prefix = value;
#pragma unroll
    for (unsigned offset = 1u; offset < kWarps; offset <<= 1u) {
      const unsigned add =
          __shfl_up_sync(0xffffffffu, prefix, offset);
      if (lane >= offset)
        prefix += add;
    }
    if (lane < kWarps)
      warp_offsets[lane] = prefix - value;
    if (lane == kWarps - 1u)
      block_slot = atomicAdd(miss_count, prefix);
  }
  __syncthreads();
  if (missing) {
    const unsigned rank =
        static_cast<unsigned>(__popc(mask & ((1u << lane) - 1u)));
    miss_indices[block_slot + warp_offsets[warp] + rank] =
        static_cast<std::uint32_t>(i);
  }
}

// Classifies resolved corrections into two bitmaps.
__global__ void successor_classify_kernel(const std::uint32_t *keys,
                                          const std::int8_t *count_delta,
                                          std::size_t count, SortedRunView base,
                                          std::uint32_t *positive_words,
                                          std::uint32_t *deleted_base_words) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const int delta = static_cast<int>(count_delta[i]);
  if (delta > 0) {
    atomicOr(positive_words + (i >> 5),
             1u << (static_cast<std::uint32_t>(i) & 31u));
    return;
  }
  if (delta >= 0 || base.count == 0)
    return;
  const std::uint32_t key = keys[i];
  std::size_t begin = 0;
  std::size_t end = 0;
  sorted_search_bounds(base, key, &begin, &end);
  const std::size_t position =
      begin + lower_bound_u32(base.keys + begin, end - begin, key);
  if (position >= end || base.keys[position] != key)
    return;
  // Shared words require atomic updates.
  atomicOr(deleted_base_words + (position >> 5),
           1u << (static_cast<std::uint32_t>(position) & 31u));
}

// Seed deleted-base bits from canonical dead positions (sec 23).
__global__ void succ_seed_canonical_dead_kernel(const std::uint8_t *state,
                                                std::uint32_t base_count,
                                                std::uint32_t *deleted_words) {
  const std::uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= base_count)
    return;
  if (state[p] == kCanonDead)
    atomicOr(deleted_words + (p >> 5), 1u << (p & 31u));
}

// Masks unused BaseRun tail positions.
__global__ void successor_tail_mask_kernel(std::uint32_t *words,
                                           std::uint32_t word_count,
                                           std::uint32_t base_count) {
  if (word_count == 0u)
    return;
  const std::uint32_t used = base_count & 31u;
  if (used)
    words[word_count - 1u] |= ~((1u << used) - 1u);
}

// Marks nonempty lower-level words.
__global__ void successor_live_level_kernel(const std::uint32_t *lower,
                                            std::uint32_t lower_words,
                                            std::uint32_t *upper,
                                            std::uint32_t upper_words,
                                            int deleted_bits) {
  const std::uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= upper_words)
    return;
  const std::uint32_t first = j << 5;
  const std::uint32_t limit =
      lower_words > first ? min(32u, lower_words - first) : 0u;
  std::uint32_t bits = 0u;
  for (std::uint32_t k = 0; k < limit; ++k) {
    const std::uint32_t word = lower[first + k];
    const bool live = deleted_bits ? (~word) != 0u : word != 0u;
    if (live)
      bits |= 1u << k;
  }
  upper[j] = bits;
}

// Returns the smaller base or positive correction key.
__global__ void sparse_successor_kernel(SuccessorSparseView view,
                                        const std::uint32_t *queries,
                                        const std::uint32_t *indices,
                                        std::size_t count,
                                        std::uint32_t *results) {
  const std::size_t t =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= count)
    return;
  const std::size_t i =
      indices ? static_cast<std::size_t>(indices[t]) : t;
  const std::uint32_t key = queries[i];
  std::uint32_t best = 0u;
  bool found = false;
  if (view.base.count != 0) {
    std::size_t begin = 0;
    std::size_t end = 0;
    sorted_search_bounds(view.base, key, &begin, &end);
    std::size_t position =
        begin + lower_bound_u32(view.base.keys + begin, end - begin, key);
    position = successor_first_base_live(view, position);
    if (position < view.base.count) {
      best = view.base.keys[position];
      found = true;
    }
  }
  if (view.correction_count != 0u) {
    const std::uint32_t q = key >> kEpochQuotientBits;
    const std::uint32_t begin = view.correction_offsets[q];
    const std::uint32_t end = view.correction_offsets[q + 1u];
    std::size_t position =
        begin + lower_bound_u32(view.correction_keys + begin,
                                end - begin, key);
    position = successor_first_positive(view, position);
    if (position < view.correction_count) {
      const std::uint32_t candidate =
          view.correction_keys[position];
      if (!found || candidate < best) {
        best = candidate;
        found = true;
      }
    }
  }
  results[i] = found ? best : 0u;
}

// Packs one run in chronological order.
__global__ void resolve_pack_run_kernel(const std::uint32_t *keys,
                                        const std::uint32_t *values,
                                        const std::uint32_t *op_words,
                                        std::uint8_t constant_op,
                                        std::uint8_t mixed, std::size_t n,
                                        std::uint32_t *out_keys,
                                        std::uint64_t *out_payload) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint32_t v = values ? values[i] : 0u;
  const int op = assignment_op_at(op_words, constant_op, mixed,
                                  static_cast<std::uint32_t>(i));
  out_keys[i] = keys[i];
  out_payload[i] =
      (static_cast<std::uint64_t>(v) << 32) | (op != 0 ? 1u : 0u);
}

// Builds assignment quotient offsets.
__global__ void assignment_boundary_kernel(const std::uint32_t *keys,
                                           std::uint32_t count,
                                           std::uint32_t *offsets) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  if (i == 0u)
    offsets[kEpochQuotients] = count;
  const std::uint32_t q = keys[i] >> kEpochQuotientBits;
  const std::uint32_t prev = i == 0u ? q : keys[i - 1u] >> kEpochQuotientBits;
  if (i == 0u || q != prev)
    offsets[q] = i;
}

// Flat directory boundary: rank23[key>>9] = first position.
__global__ void base_rank23_boundary_kernel(const std::uint32_t *keys,
                                            std::uint32_t count,
                                            std::uint32_t *rank23) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  if (i == 0u)
    rank23[kBaseRank23Size] = count;
  const std::uint32_t bin = keys[i] >> kBaseRank23Shift;
  const std::uint32_t prev =
      i == 0u ? bin : keys[i - 1u] >> kBaseRank23Shift;
  if (i == 0u || bin != prev)
    rank23[bin] = i;
}

// Keeps the newest equal-key assignment.
__global__ void resolve_flag_last_kernel(const std::uint32_t *keys,
                                         std::size_t n,
                                         std::uint8_t *flags) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  flags[i] = (i + 1u == n || keys[i] != keys[i + 1u]) ? 1u : 0u;
}

// Corrections use modulo-32-bit value arithmetic.
__host__ __device__ inline std::uint64_t
corr_pack(std::uint32_t value_delta, std::int8_t count_delta) {
  return (static_cast<std::uint64_t>(value_delta) << 32) |
         static_cast<std::uint32_t>(static_cast<std::int32_t>(count_delta));
}

struct CacheTaggedKey {
  __host__ __device__ std::uint64_t operator()(std::uint32_t key) const {
    return static_cast<std::uint64_t>(key) << 32;
  }
};

struct CountDeltaToU32 {
  __host__ __device__ std::uint32_t operator()(std::int8_t delta) const {
    return static_cast<std::uint32_t>(
        static_cast<std::int32_t>(delta));
  }
};

// Converts assignments to base corrections.
__global__ void normalize_correction_kernel(
    const std::uint32_t *keys, const std::uint64_t *assignment,
    std::size_t count, CanonicalBaseView base,
    std::uint64_t *tagged_keys, std::uint64_t *corrections) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::uint32_t key = keys[i];
  const std::uint32_t value =
      static_cast<std::uint32_t>(assignment[i] >> 32);
  const bool insert = (assignment[i] & 1u) != 0u;
  // Baseline is the canonical visible state, not the raw BaseRun.
  std::uint32_t base_value = 0u;
  bool base_live = false;
  if (base.base.count != 0u && canonical_find_value(base, key, &base_value))
    base_live = true;
  std::uint32_t value_delta = 0u;
  std::int8_t count_delta = 0;
  if (insert) {
    value_delta = base_live ? value - base_value : value;
    count_delta = base_live ? 0 : 1;
  } else if (base_live) {
    value_delta = 0u - base_value;
    count_delta = -1;
  }
  const std::uint64_t rank = static_cast<std::uint32_t>(i + 1u);
  tagged_keys[i] = (static_cast<std::uint64_t>(key) << 32) | rank;
  corrections[i] = corr_pack(value_delta, count_delta);
}

__global__ void corr_pack_kernel(const std::uint32_t *value_delta,
                                 const std::int8_t *count_delta, std::size_t n,
                                 std::uint64_t *out_payload) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  out_payload[i] = corr_pack(value_delta[i], count_delta[i]);
}

__global__ void corr_unpack_kernel(const std::uint64_t *tagged_keys,
                                   const std::uint64_t *payload, std::size_t n,
                                   std::uint32_t *out_keys,
                                   std::uint32_t *out_value_delta,
                                   std::int8_t *out_count_delta) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  out_keys[i] = static_cast<std::uint32_t>(tagged_keys[i] >> 32);
  out_value_delta[i] = static_cast<std::uint32_t>(payload[i] >> 32);
  out_count_delta[i] = static_cast<std::int8_t>(
      static_cast<std::uint32_t>(payload[i]));
}

// The greatest temporal rank is the visible assignment.
__global__ void resolve_merge_flag_kernel(const std::uint64_t *tagged_keys,
                                          const std::uint64_t *payload,
                                          std::size_t total,
                                          std::uint8_t *flags) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total)
    return;
  const std::uint32_t key =
      static_cast<std::uint32_t>(tagged_keys[i] >> 32);
  const bool last =
      i + 1u == total ||
      static_cast<std::uint32_t>(tagged_keys[i + 1u] >> 32) != key;
  flags[i] = (last && payload[i] != 0u) ? 1u : 0u;
}

// Splits newest live assignments from deletes.
__global__ void resolve_base_extract_kernel(const std::uint64_t *payload,
                                            std::size_t m,
                                            std::uint32_t *out_values,
                                            std::uint8_t *out_keep) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= m)
    return;
  out_values[i] = static_cast<std::uint32_t>(payload[i] >> 32);
  out_keep[i] = (payload[i] & 1u) != 0u ? 1u : 0u;
}

__device__ inline void resolved_range_ranks(
    const RunView &resolved, std::uint32_t lo, std::uint32_t hi,
    std::size_t *lower, std::size_t *upper) {
  const std::uint32_t lo_q = lo >> kEpochQuotientBits;
  const std::uint32_t hi_q = hi >> kEpochQuotientBits;
  const std::size_t lo_begin = resolved.quotient_off[lo_q];
  const std::size_t lo_end = resolved.quotient_off[lo_q + 1u];
  const std::size_t hi_begin = resolved.quotient_off[hi_q];
  const std::size_t hi_end = resolved.quotient_off[hi_q + 1u];
  *lower = lo_begin + lower_bound_u32(
      resolved.keys + lo_begin, lo_end - lo_begin, lo);
  *upper = hi_begin + upper_bound_u32(
      resolved.keys + hi_begin, hi_end - hi_begin, hi);
}

template <bool WithCounts>
__global__ void resolved_range_kernel(
    const std::uint32_t *lo, const std::uint32_t *hi,
    std::uint32_t *out_sums, std::uint32_t *out_counts,
    std::size_t query_count, CanonicalBaseView base,
    SortedRunRangeView base_range,
    const std::uint32_t *base_value_prefix,
    const std::uint32_t *base_count_prefix, RunView resolved,
    const std::uint32_t *resolved_value_prefix,
    const std::uint32_t *resolved_count_prefix, int resolved_ready) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= query_count)
    return;
  const std::uint32_t l = lo[i];
  const std::uint32_t h = hi[i];
  if (l > h) {
    out_sums[i] = 0u;
    if constexpr (WithCounts)
      out_counts[i] = 0u;
    return;
  }
  std::size_t delta_lower = 0u;
  std::size_t delta_upper = 0u;
  if (resolved_ready)
    resolved_range_ranks(
        resolved, l, h, &delta_lower, &delta_upper);
  // BaseRun sum + canonical overlay + resolved pending/cold.
  std::uint32_t sum =
      canonical_range_sum(base, base_range, base_value_prefix, l, h);
  if (resolved_ready)
    sum += resolved_value_prefix[delta_upper] -
           resolved_value_prefix[delta_lower];
  out_sums[i] = sum;
  if constexpr (WithCounts) {
    std::uint32_t count =
        canonical_range_count(base, base_count_prefix, l, h);
    if (resolved_ready)
      count += resolved_count_prefix[delta_upper] -
               resolved_count_prefix[delta_lower];
    out_counts[i] = count;
  }
}

// One block maps its lanes directly to chronological runs.
__global__ void temporal_lookup_run_parallel_kernel(
    const AssignmentRunView *runs, int run_count, CanonicalBaseView base,
    const std::uint32_t *queries, std::size_t n,
    std::uint32_t *out_values, std::uint8_t *out_found) {
  const std::size_t query = blockIdx.x;
  if (query >= n)
    return;
  __shared__ unsigned long long candidates[kRunCapacity];
  const int run_index = threadIdx.x;
  const std::uint32_t key = queries[query];
  const std::uint32_t q = key >> kEpochQuotientBits;
  unsigned long long candidate = 0u;
  if (run_index < run_count) {
    const AssignmentRunView run = runs[run_index];
    std::uint32_t begin, position;
    assignment_bounds(run, q, &begin, &position);
    while (position-- > begin) {
      if (run.keys[position] != key)
        continue;
      const int op = assignment_op_at(
          run.op_words, run.constant_op, run.mixed, position);
      const std::uint32_t value = op != 0 ? run.values[position] : kEmptyKey;
      candidate =
          (static_cast<unsigned long long>(run_index + 1) << 33) |
          (static_cast<unsigned long long>(op != 0) << 32) | value;
      break;
    }
  }
  candidates[run_index] = candidate;
  __syncthreads();
  for (int stride = kRunCapacity / 2; stride > 0; stride >>= 1) {
    if (run_index < stride) {
      const unsigned long long other = candidates[run_index + stride];
      if (other > candidates[run_index])
        candidates[run_index] = other;
    }
    __syncthreads();
  }
  if (run_index != 0)
    return;
  const unsigned long long best = candidates[0];
  if (best != 0u) {
    const bool live = ((best >> 32) & 1u) != 0u;
    out_values[query] =
        live ? static_cast<std::uint32_t>(best) : kEmptyKey;
    if (out_found)
      out_found[query] = live ? 1u : 0u;
    return;
  }
  std::uint32_t value = kEmptyKey;
  std::uint8_t found = 0u;
  found = canonical_find_value(base, key, &value) ? 1u : 0u;
  if (!found)
    value = kEmptyKey;
  out_values[query] = value;
  if (out_found)
    out_found[query] = found;
}

constexpr int kNarrowSeenCap = 128;
constexpr int kNarrowHashSlots = 2 * kNarrowSeenCap;
static_assert((kNarrowHashSlots & (kNarrowHashSlots - 1)) == 0);

// Hash for narrow-range sets.
__host__ __device__ inline std::uint64_t hash_mix_slot(std::uint32_t key,
                                                       std::uint64_t mask) {
  std::uint64_t x = key;
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x & mask;
}

__device__ inline void narrow_record_latest(
    std::uint32_t *keys, unsigned long long *references,
    std::uint32_t key, unsigned long long reference) {
  std::uint32_t slot =
      static_cast<std::uint32_t>(hash_mix_slot(key, kNarrowHashSlots - 1u));
  for (int probe = 0; probe < kNarrowHashSlots; ++probe) {
    const std::uint32_t previous =
        atomicCAS(keys + slot, kEmptyKey, key);
    if (previous == kEmptyKey || previous == key) {
      atomicMax(references + slot, reference);
      return;
    }
    slot = (slot + 1u) & (kNarrowHashSlots - 1u);
  }
}

// One block resolves one bounded range in shared memory.
__global__ void narrow_range_kernel(
    const std::uint32_t *lo, const std::uint32_t *hi,
    std::uint32_t *out_sums, std::uint32_t *out_counts,
    std::size_t query_count, const AssignmentRunView *runs,
    int run_count, CanonicalBaseView base_view,
    SortedRunRangeView base_range,
    const std::uint32_t *base_value_prefix,
    const std::uint32_t *base_count_prefix,
    std::uint32_t *overflow) {
  const std::size_t query = blockIdx.x;
  if (query >= query_count)
    return;
  const int lane = threadIdx.x;
  const std::uint32_t l = lo[query];
  const std::uint32_t h = hi[query];
  if (l > h) {
    if (lane == 0) {
      out_sums[query] = 0u;
      if (out_counts)
        out_counts[query] = 0u;
    }
    return;
  }

  __shared__ std::uint32_t hash_keys[kNarrowHashSlots];
  __shared__ unsigned long long hash_refs[kNarrowHashSlots];
  __shared__ std::uint32_t reduce_sum[kNarrowSeenCap];
  __shared__ std::int32_t reduce_count[kNarrowSeenCap];
  for (int slot = lane; slot < kNarrowHashSlots; slot += blockDim.x) {
    hash_keys[slot] = kEmptyKey;
    hash_refs[slot] = 0u;
  }

  const std::uint32_t qlo = l >> kEpochQuotientBits;
  const std::uint32_t qhi = h >> kEpochQuotientBits;
  // A wide quotient span defers to the whole-batch wide path.
  if (qhi - qlo > static_cast<std::uint32_t>(kNarrowSeenCap)) {
    if (lane == 0)
      atomicExch(overflow, 1u);
    return;
  }
  std::uint32_t candidates = 0u;
  for (int r = lane; r < run_count; r += blockDim.x) {
    const AssignmentRunView run = runs[r];
    for (std::uint32_t qq = qlo; qq <= qhi; ++qq) {
      std::uint32_t b, e;
      assignment_bounds(run, qq, &b, &e);
      candidates += e - b;
    }
  }
  reduce_sum[lane] = candidates;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (lane < stride)
      reduce_sum[lane] += reduce_sum[lane + stride];
    __syncthreads();
  }
  if (reduce_sum[0] > kNarrowSeenCap) {
    if (lane == 0)
      atomicExch(overflow, 1u);
    return;
  }

  if (lane < run_count) {
    const AssignmentRunView run = runs[lane];
    for (std::uint32_t qq = qlo; qq <= qhi; ++qq) {
      std::uint32_t begin, end;
      assignment_bounds(run, qq, &begin, &end);
      for (std::uint32_t p = begin; p < end; ++p) {
        const std::uint32_t key = run.keys[p];
        if (key < l || key > h)
          continue;
        const unsigned long long reference =
            (static_cast<unsigned long long>(lane + 1) << 32) | p;
        narrow_record_latest(hash_keys, hash_refs, key, reference);
      }
    }
  }
  __syncthreads();

  std::uint32_t correction = 0u;
  std::int32_t count_correction = 0;
  for (int slot = lane; slot < kNarrowHashSlots; slot += blockDim.x) {
    const std::uint32_t key = hash_keys[slot];
    if (key == kEmptyKey)
      continue;
    const unsigned long long reference = hash_refs[slot];
    const int r = static_cast<int>(reference >> 32) - 1;
    const std::uint32_t p = static_cast<std::uint32_t>(reference);
    const AssignmentRunView run = runs[r];
    const int op =
        assignment_op_at(run.op_words, run.constant_op, run.mixed, p);
    // Baseline is the canonical visible state, not the raw BaseRun.
    std::uint32_t base_value = 0u;
    const bool base_live = canonical_find_value(base_view, key, &base_value);
    if (op != 0) {
      const std::uint32_t value = run.values[p];
      correction += base_live ? value - base_value : value;
      count_correction += base_live ? 0 : 1;
    } else if (base_live) {
      correction += 0u - base_value;
      --count_correction;
    }
  }
  reduce_sum[lane] = correction;
  reduce_count[lane] = count_correction;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      reduce_sum[lane] += reduce_sum[lane + stride];
      reduce_count[lane] += reduce_count[lane + stride];
    }
    __syncthreads();
  }
  if (lane == 0) {
    out_sums[query] =
        canonical_range_sum(base_view, base_range, base_value_prefix, l, h) +
        reduce_sum[0];
    if (out_counts) {
      const std::uint32_t base_count =
          canonical_range_count(base_view, base_count_prefix, l, h);
      out_counts[query] =
          base_count + static_cast<std::uint32_t>(reduce_count[0]);
    }
  }
}

template <bool WithCounts>
__global__ void base_only_range_kernel(
    const std::uint32_t *lo, const std::uint32_t *hi,
    std::uint32_t *out_sums, std::uint32_t *out_counts,
    std::size_t query_count, CanonicalBaseView base,
    SortedRunRangeView base_range,
    const std::uint32_t *base_value_prefix,
    const std::uint32_t *base_count_prefix) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= query_count)
    return;
  const std::uint32_t l = lo[i];
  const std::uint32_t h = hi[i];
  if (l > h) {
    out_sums[i] = 0u;
    if constexpr (WithCounts)
      out_counts[i] = 0u;
    return;
  }
  out_sums[i] =
      canonical_range_sum(base, base_range, base_value_prefix, l, h);
  if constexpr (WithCounts)
    out_counts[i] =
      canonical_range_count(base, base_count_prefix, l, h);
}

// --- Rank23 canonical fold (sec 13) ---
constexpr int kFoldQuotientCap = 1280;
constexpr int kFoldBinCap = 32;
constexpr int kFoldThreads = 512;
// Base keys per quotient (~512 uniform); window caps the shared
// preload of state/override/base-value so apply hits on-chip memory.
constexpr int kWindowCap = 768;
// Device fold stat cells (sec 28).
constexpr int kFoldStatMatched = 0;
constexpr int kFoldStatUnmatched = 1;
constexpr int kFoldStatFallback = 2;
constexpr int kFoldStatOverflow = 3;
constexpr int kFoldStatCells = 4;
static_assert(sizeof(FoldRecord) == 12, "fold record layout");
static_assert(kFoldQuotientCap *
                      (sizeof(FoldRecord) + 2 * sizeof(std::uint16_t) + 1) +
                  kWindowCap * (3 * sizeof(std::uint32_t) + 1) +
                  12 * 1024 <=
              48 * 1024,
              "fold shared memory exceeds the 48 KiB static limit");
static_assert(kFoldBinCap <= 32, "fast bins must fit one warp");

// Apply one winner to a matched BaseRun position; the caller
// accumulates the returned correction change into the bin totals.
__device__ inline void
canonical_apply_one(std::uint32_t bv, std::uint8_t *state,
                    std::uint32_t *override_values, std::size_t p, int op,
                    std::uint32_t value, std::uint32_t *out_vdelta,
                    std::int32_t *out_cdelta) {
  const std::uint8_t os = state[p];
  std::uint32_t old_vc = 0u;
  std::int32_t old_cc = 0;
  if (os == kCanonOverride)
    old_vc = override_values[p] - bv;
  else if (os == kCanonDead) {
    old_vc = 0u - bv;
    old_cc = -1;
  }
  std::uint8_t ns;
  std::uint32_t nv = bv;
  if (op == 0)
    ns = kCanonDead;
  else if (value == bv)
    ns = kCanonBase;
  else {
    ns = kCanonOverride;
    nv = value;
  }
  std::uint32_t new_vc = 0u;
  std::int32_t new_cc = 0;
  if (ns == kCanonOverride)
    new_vc = nv - bv;
  else if (ns == kCanonDead) {
    new_vc = 0u - bv;
    new_cc = -1;
  }
  state[p] = ns;
  if (ns == kCanonOverride)
    override_values[p] = nv;
  *out_vdelta = new_vc - old_vc;
  *out_cdelta = new_cc - old_cc;
}

__device__ inline unsigned long long
fold_record_order(const FoldRecord &record) {
  return (static_cast<unsigned long long>(record.low_key) << 40) |
         (static_cast<unsigned long long>(record.source_rank) << 32) |
         record.source_position;
}

__device__ inline int fold_sort_width(int count) {
  int width = 1;
  while (width < count)
    width <<= 1;
  return width;
}

__device__ inline void
fold_warp_bitonic(unsigned long long &key, std::uint16_t &record,
                  int width) {
  constexpr unsigned mask = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  for (int length = 2; length <= width; length <<= 1) {
    for (int stride = length >> 1; stride > 0; stride >>= 1) {
      const unsigned long long other_key =
          __shfl_xor_sync(mask, key, stride, width);
      const std::uint16_t other_record =
          __shfl_xor_sync(mask, record, stride, width);
      const bool ascending = (lane & length) == 0;
      const bool lower_lane = (lane & stride) == 0;
      const bool take_min = ascending == lower_lane;
      const bool take_other =
          take_min ? other_key < key : other_key > key;
      if (take_other) {
        key = other_key;
        record = other_record;
      }
    }
  }
}

// One block per quotient; fold 64 raw runs into canonical state
// plus a cold page of unmatched keys. Overflow -> fallback list.
__global__ void canonical_fold_rank23_kernel(
    const AssignmentRunView *runs, int run_count, SortedRunView base,
    std::uint8_t *state, std::uint32_t *override_values,
    std::uint32_t *rank23_value_delta, std::int32_t *rank23_count_delta,
    std::uint32_t *cold_page_begin, std::uint32_t *cold_page_count,
    std::uint32_t *cold_page_capacity,
    std::uint32_t *cold_arena_tail, std::uint32_t arena_capacity,
    std::uint32_t *cold_keys, std::uint32_t *cold_values,
    std::uint32_t *cold_ops, std::uint32_t *fallback_quotients,
    std::uint32_t *fallback_count, std::uint32_t *fold_stats) {
  const std::uint32_t q = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ FoldRecord s_rec[kFoldQuotientCap];
  __shared__ std::uint16_t s_order[kFoldQuotientCap];
  __shared__ std::uint16_t s_cold[kFoldQuotientCap];
  __shared__ std::uint8_t s_coldflag[kFoldQuotientCap];
  __shared__ std::uint32_t s_runoff[kRawFoldWidth + 1];
  __shared__ std::uint32_t s_runbegin[kRawFoldWidth];
  __shared__ AssignmentRunView s_views[kRawFoldWidth];
  __shared__ std::uint32_t s_base_b[128];
  __shared__ std::uint32_t s_base_e[128];
  __shared__ std::uint32_t s_win_key[kWindowCap];
  __shared__ std::uint32_t s_win_val[kWindowCap];
  __shared__ std::uint32_t s_win_ovr[kWindowCap];
  __shared__ std::uint8_t s_win_state[kWindowCap];
  __shared__ std::uint32_t s_wb;
  __shared__ int s_wlen;
  __shared__ int s_bincount[128];
  __shared__ int s_binoff[129];
  __shared__ int s_bincur[128];
  __shared__ int s_coldcount[128];
  __shared__ int s_coldoff[129];
  __shared__ std::uint32_t s_rank_dv[128];
  __shared__ std::int32_t s_rank_dc[128];
  __shared__ int s_total;
  __shared__ int s_overflow;
  __shared__ int s_coldn;
  __shared__ int s_matched;
  __shared__ std::uint32_t s_page;

  if (tid == 0) {
    s_overflow = 0;
    s_coldn = 0;
    s_matched = 0;
  }
  for (int r = tid; r < run_count; r += blockDim.x) {
    s_views[r] = runs[r];
    std::uint32_t b, e;
    assignment_bounds(s_views[r], q, &b, &e);
    s_runbegin[r] = b;
    s_runoff[r] = e - b;
  }
  // One parallel pass caches every base bin boundary.
  if (tid < 128) {
    const std::uint32_t gbin =
        (q << (kBaseRank23Bits - kEpochQuotientBits)) |
        static_cast<std::uint32_t>(tid);
    s_base_b[tid] = base.rank23[gbin];
    s_base_e[tid] = base.rank23[gbin + 1u];
  }
  __syncthreads();
  if (tid == 0) {
    std::uint32_t c = 0;
    for (int r = 0; r < run_count; ++r) {
      const std::uint32_t cc = s_runoff[r];
      s_runoff[r] = c;
      c += cc;
    }
    s_runoff[run_count] = c;
    s_total = static_cast<int>(c);
    if (c > static_cast<std::uint32_t>(kFoldQuotientCap))
      s_overflow = 1;
    // Contiguous base window this block will touch (all 128 bins).
    s_wb = s_base_b[0];
    s_wlen = static_cast<int>(s_base_e[127] - s_base_b[0]);
    if (s_wlen > kWindowCap)
      s_overflow = 1;
  }
  __syncthreads();
  if (s_total == 0) {
    if (tid == 0) {
      cold_page_begin[q] = 0;
      cold_page_count[q] = 0;
      cold_page_capacity[q] = 0;
    }
    return;
  }
  if (s_overflow) {
    if (tid == 0) {
      const std::uint32_t s = atomicAdd(fallback_count, 1u);
      fallback_quotients[s] = q;
      cold_page_begin[q] = 0;
      cold_page_count[q] = 0;
      cold_page_capacity[q] = 0;
      atomicAdd(fold_stats + kFoldStatFallback, 1u);
    }
    return;
  }
  // Zero histogram + delta accumulators before the fused gather.
  for (int b = tid; b < 128; b += blockDim.x) {
    s_bincount[b] = 0;
    s_rank_dv[b] = 0u;
    s_rank_dc[b] = 0;
  }
  __syncthreads();
  // Flat gather: locate each record's run and tally its bin in one
  // pass (no separate re-read of s_rec for the histogram).
  for (int i = tid; i < s_total; i += blockDim.x) {
    int lo = 0, hi = run_count - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (s_runoff[mid] <= static_cast<std::uint32_t>(i))
        lo = mid;
      else
        hi = mid - 1;
    }
    const AssignmentRunView &run = s_views[lo];
    const std::uint32_t p =
        s_runbegin[lo] + (static_cast<std::uint32_t>(i) - s_runoff[lo]);
    const int op =
        assignment_op_at(run.op_words, run.constant_op, run.mixed, p);
    FoldRecord rec;
    rec.low_key = static_cast<std::uint16_t>(run.keys[p] & 0xffffu);
    rec.operation = static_cast<std::uint8_t>(op != 0);
    rec.source_rank = static_cast<std::uint8_t>(lo);
    rec.value = (op != 0 && run.values) ? run.values[p] : 0u;
    rec.source_position = p;
    s_rec[i] = rec;
    atomicAdd(&s_bincount[rec.low_key >> 9], 1);
  }
  // Coalesced preload of this quotient's base window into shared so
  // apply reads/writes on-chip instead of scattered global memory.
  for (int i = tid; i < s_wlen; i += blockDim.x) {
    const std::uint32_t gp = s_wb + static_cast<std::uint32_t>(i);
    s_win_key[i] = base.keys[gp];
    s_win_val[i] = base.values[gp];
    s_win_ovr[i] = override_values[gp];
    s_win_state[i] = state[gp];
  }
  __syncthreads();
  if (tid == 0) {
    int off = 0;
    for (int b = 0; b < 128; ++b) {
      s_binoff[b] = off;
      if (s_bincount[b] > kFoldBinCap)
        s_overflow = 1;
      off += s_bincount[b];
    }
    s_binoff[128] = off;
  }
  __syncthreads();
  if (s_overflow) {
    if (tid == 0) {
      const std::uint32_t s = atomicAdd(fallback_count, 1u);
      fallback_quotients[s] = q;
      cold_page_begin[q] = 0;
      cold_page_count[q] = 0;
      cold_page_capacity[q] = 0;
      atomicAdd(fold_stats + kFoldStatFallback, 1u);
    }
    return;
  }
  for (int b = tid; b < 128; b += blockDim.x)
    s_bincur[b] = s_binoff[b];
  __syncthreads();
  for (int i = tid; i < s_total; i += blockDim.x) {
    const int lb = s_rec[i].low_key >> 9;
    s_order[atomicAdd(&s_bincur[lb], 1)] = static_cast<std::uint16_t>(i);
  }
  __syncthreads();
  const int warp = tid >> 5;
  const int lane = tid & 31;
  constexpr unsigned warp_mask = 0xffffffffu;
  for (int bin = warp; bin < 128; bin += (blockDim.x >> 5)) {
    const int begin = s_binoff[bin];
    const int end = s_binoff[bin + 1];
    const int count = end - begin;
    if (count == 0) {
      if (lane == 0)
        s_coldcount[bin] = 0;
      continue;
    }
    std::uint16_t record = 0xffffu;
    FoldRecord me{};
    unsigned long long order = 0ull;
    if (lane < count) {
      record = s_order[begin + lane];
      me = s_rec[record];
      order = fold_record_order(me);
    }
    // All-pairs winner: this lane owns its key iff no other lane
    // holds the same key with a higher order. No sort needed; the
    // cold path is re-sorted by the downstream merge.
    bool newest = lane < count;
    for (int src = 0; src < count; ++src) {
      const int okey =
          __shfl_sync(warp_mask, static_cast<int>(me.low_key), src);
      const unsigned long long oord =
          __shfl_sync(warp_mask, order, src);
      if (src != lane && okey == static_cast<int>(me.low_key) &&
          oord > order)
        newest = false;
    }
    const std::uint32_t base_b = s_base_b[bin];
    const std::uint32_t base_e = s_base_e[bin];
    std::uint32_t dv = 0u;
    std::int32_t dc = 0;
    bool unmatched = false;
    // Search base.keys (windowed in shared); apply hits shared.
    if (newest) {
      const std::uint32_t full_key =
          (q << kEpochQuotientBits) | me.low_key;
      const std::uint32_t rel_b = base_b - s_wb;
      const std::uint32_t p =
          base_b + static_cast<std::uint32_t>(lower_bound_u32(
                       s_win_key + rel_b, base_e - base_b, full_key));
      const std::size_t j = p - s_wb;
      if (p < base_e && s_win_key[j] == full_key) {
        canonical_apply_one(s_win_val[j], s_win_state, s_win_ovr, j,
                            me.operation, me.value, &dv, &dc);
      } else {
        unmatched = true;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      dv += __shfl_down_sync(warp_mask, dv, offset);
      dc += __shfl_down_sync(warp_mask, dc, offset);
    }
    if (lane == 0) {
      s_rank_dv[bin] = dv;
      s_rank_dc[bin] = dc;
    }
    if (lane < count)
      s_coldflag[begin + lane] =
          static_cast<std::uint8_t>(unmatched);
    const unsigned cold = __ballot_sync(warp_mask, unmatched);
    const unsigned applied =
        __ballot_sync(warp_mask, newest && !unmatched);
    if (lane == 0) {
      s_coldcount[bin] = __popc(cold);
      if (applied)
        atomicAdd(&s_matched, __popc(applied));
    }
  }
  __syncthreads();
  // Coalesced write-back of the modified base window.
  for (int i = tid; i < s_wlen; i += blockDim.x) {
    const std::uint32_t gp = s_wb + static_cast<std::uint32_t>(i);
    state[gp] = s_win_state[i];
    override_values[gp] = s_win_ovr[i];
  }
  // Coalesced write-back of this quotient's 128 rank23 deltas.
  const std::uint32_t rank_base =
      q << (kBaseRank23Bits - kEpochQuotientBits);
  for (int b = tid; b < 128; b += blockDim.x) {
    if (s_rank_dv[b] != 0u)
      rank23_value_delta[rank_base + b] += s_rank_dv[b];
    if (s_rank_dc[b] != 0)
      rank23_count_delta[rank_base + b] += s_rank_dc[b];
  }
  if (tid == 0) {
    int off = 0;
    for (int bin = 0; bin < 128; ++bin) {
      s_coldoff[bin] = off;
      off += s_coldcount[bin];
    }
    s_coldoff[128] = off;
    s_coldn = off;
  }
  __syncthreads();
  for (int bin = warp; bin < 128; bin += (blockDim.x >> 5)) {
    const int begin = s_binoff[bin];
    const int count = s_binoff[bin + 1] - begin;
    const bool keep =
        lane < count && s_coldflag[begin + lane] != 0;
    const unsigned kept = __ballot_sync(warp_mask, keep);
    if (keep) {
      const unsigned before =
          lane == 0 ? 0u : kept & ((1u << lane) - 1u);
      s_cold[s_coldoff[bin] + __popc(before)] =
          s_order[begin + lane];
    }
  }
  __syncthreads();
  if (tid == 0) {
    const int n = s_coldn;
    cold_page_count[q] = static_cast<std::uint32_t>(n);
    if (n > 0) {
      const std::uint32_t cap =
          static_cast<std::uint32_t>((n + 31) & ~31);
      const std::uint32_t begin = atomicAdd(cold_arena_tail, cap);
      s_page = (begin + cap <= arena_capacity) ? begin : 0xffffffffu;
      cold_page_begin[q] = (s_page == 0xffffffffu) ? 0u : begin;
      cold_page_capacity[q] =
          (s_page == 0xffffffffu) ? 0u : cap;
      if (s_page == 0xffffffffu) {
        cold_page_count[q] = 0u;
        atomicAdd(fold_stats + kFoldStatOverflow,
                  static_cast<std::uint32_t>(n));
      } else {
        atomicAdd(fold_stats + kFoldStatUnmatched,
                  static_cast<std::uint32_t>(n));
      }
    } else {
      cold_page_begin[q] = 0u;
      cold_page_capacity[q] = 0u;
      s_page = 0xffffffffu;
    }
    if (s_matched)
      atomicAdd(fold_stats + kFoldStatMatched,
                static_cast<std::uint32_t>(s_matched));
  }
  __syncthreads();
  if (s_page != 0xffffffffu) {
    const std::uint32_t words = cold_page_capacity[q] >> 5;
    for (std::uint32_t word = tid; word < words;
         word += blockDim.x)
      cold_ops[(s_page >> 5) + word] = 0u;
  }
  __syncthreads();
  if (s_page != 0xffffffffu) {
    for (int i = tid; i < s_coldn; i += blockDim.x) {
      const FoldRecord r = s_rec[s_cold[i]];
      const std::uint32_t dst = s_page + static_cast<std::uint32_t>(i);
      cold_keys[dst] = (q << kEpochQuotientBits) | r.low_key;
      cold_values[dst] = r.operation ? r.value : 0u;
      if (r.operation)
        atomicOr(&cold_ops[dst >> 5], 1u << (dst & 31u));
    }
  }
}

constexpr int kFoldFallbackBlocks = 256;
constexpr int kFoldFallbackThreads = 64;

__global__ void canonical_fold_fallback_kernel(
    const AssignmentRunView *runs, int run_count,
    const std::uint32_t *quotient_list,
    const std::uint32_t *quotient_count,
    std::uint32_t *work_head, SortedRunView base,
    std::uint8_t *state, std::uint32_t *override_values,
    std::uint32_t *rank23_value_delta,
    std::int32_t *rank23_count_delta,
    std::uint32_t *cold_page_begin,
    std::uint32_t *cold_page_count,
    std::uint32_t *cold_page_capacity,
    std::uint32_t *cold_arena_tail,
    std::uint32_t arena_capacity,
    std::uint32_t *cold_keys,
    std::uint32_t *cold_values,
    std::uint32_t *cold_ops,
    std::uint32_t *fold_stats) {
  const int tid = threadIdx.x;
  __shared__ std::uint32_t s_pos[kRawFoldWidth];
  __shared__ std::uint32_t s_end[kRawFoldWidth];
  __shared__ std::uint32_t s_last[kRawFoldWidth];
  __shared__ std::uint32_t s_reduce[kRawFoldWidth];
  __shared__ int s_rank[kRawFoldWidth];
  __shared__ std::uint32_t s_work;
  __shared__ std::uint32_t s_q;
  __shared__ std::uint32_t s_page;
  __shared__ std::uint32_t s_coldn;
  __shared__ std::uint32_t s_matchn;

  while (true) {
    if (tid == 0)
      s_work = atomicAdd(work_head, 1u);
    __syncthreads();
    if (s_work >= *quotient_count)
      return;
    if (tid == 0)
      s_q = quotient_list[s_work];
    __syncthreads();
    const std::uint32_t q = s_q;
    std::uint32_t begin = 0u;
    std::uint32_t end = 0u;
    if (tid < run_count)
      assignment_bounds(runs[tid], q, &begin, &end);
    s_pos[tid] = begin;
    s_end[tid] = end;
    s_reduce[tid] = end - begin;
    __syncthreads();
    for (int stride = kRawFoldWidth / 2; stride > 0;
         stride >>= 1) {
      if (tid < stride)
        s_reduce[tid] += s_reduce[tid + stride];
      __syncthreads();
    }
    if (tid == 0) {
      const std::uint32_t total = s_reduce[0];
      const std::uint32_t cap = (total + 31u) & ~31u;
      const std::uint32_t page =
          atomicAdd(cold_arena_tail, cap);
      if (page > arena_capacity ||
          cap > arena_capacity - page)
        asm("trap;");
      s_page = page;
      s_coldn = 0u;
      s_matchn = 0u;
      cold_page_begin[q] = page;
      cold_page_count[q] = 0u;
      cold_page_capacity[q] = cap;
    }
    __syncthreads();
    const std::uint32_t words =
        cold_page_capacity[q] >> 5;
    for (std::uint32_t word = tid; word < words;
         word += blockDim.x)
      cold_ops[(s_page >> 5) + word] = 0u;
    __syncthreads();

    while (true) {
      const std::uint32_t pos = s_pos[tid];
      const bool active = pos < s_end[tid];
      s_reduce[tid] = active ? runs[tid].keys[pos]
                             : 0xffffffffu;
      if (__syncthreads_count(active) == 0)
        break;
      for (int stride = kRawFoldWidth / 2; stride > 0;
           stride >>= 1) {
        if (tid < stride)
          s_reduce[tid] =
              min(s_reduce[tid], s_reduce[tid + stride]);
        __syncthreads();
      }
      const std::uint32_t key = s_reduce[0];

      int rank = -1;
      if (tid < run_count) {
        std::uint32_t p = s_pos[tid];
        if (p < s_end[tid] && runs[tid].keys[p] == key) {
          do {
            s_last[tid] = p;
            ++p;
          } while (p < s_end[tid] &&
                   runs[tid].keys[p] == key);
          s_pos[tid] = p;
          rank = tid;
        }
      }
      s_rank[tid] = rank;
      __syncthreads();
      for (int stride = kRawFoldWidth / 2; stride > 0;
           stride >>= 1) {
        if (tid < stride)
          s_rank[tid] = max(s_rank[tid],
                            s_rank[tid + stride]);
        __syncthreads();
      }
      const int winner = s_rank[0];
      if (tid == winner) {
        const AssignmentRunView run = runs[winner];
        const std::uint32_t p = s_last[winner];
        const int op = assignment_op_at(
            run.op_words, run.constant_op, run.mixed, p);
        const std::uint32_t value =
            op != 0 && run.values ? run.values[p] : 0u;
        const std::uint32_t gbin =
            key >> kBaseRank23Shift;
        const std::uint32_t base_b = base.rank23[gbin];
        const std::uint32_t base_e = base.rank23[gbin + 1u];
        const std::uint32_t bp =
            base_b + static_cast<std::uint32_t>(
                         lower_bound_u32(base.keys + base_b,
                                         base_e - base_b, key));
        if (bp < base_e && base.keys[bp] == key) {
          std::uint32_t dv;
          std::int32_t dc;
          canonical_apply_one(base.values[bp], state,
                              override_values, bp, op,
                              value, &dv, &dc);
          rank23_value_delta[gbin] += dv;
          rank23_count_delta[gbin] += dc;
          ++s_matchn;
        } else {
          const std::uint32_t slot = s_page + s_coldn++;
          cold_keys[slot] = key;
          cold_values[slot] = op != 0 ? value : 0u;
          if (op != 0)
            atomicOr(&cold_ops[slot >> 5],
                     1u << (slot & 31u));
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      cold_page_count[q] = s_coldn;
      if (s_matchn)
        atomicAdd(fold_stats + kFoldStatMatched, s_matchn);
      if (s_coldn)
        atomicAdd(fold_stats + kFoldStatUnmatched, s_coldn);
    }
    __syncthreads();
  }
}

// Bake the overlay into the base: adjusted value + keep flag.
__global__ void canonical_bake_kernel(const std::uint8_t *state,
                                      const std::uint32_t *override_values,
                                      const std::uint32_t *base_values,
                                      std::size_t n, std::uint32_t *out_values,
                                      std::uint8_t *out_keep) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint8_t s = state[i];
  out_keep[i] = (s == kCanonDead) ? 0u : 1u;
  out_values[i] = (s == kCanonOverride) ? override_values[i] : base_values[i];
}

} // namespace gpulsmopt_detail

class GPULSMOpt {
public:
  using DeviceKeyBatch = gpulsmopt_detail::DeviceKeyBatch;

  // Host-side maintenance counters for sustained workloads.
  struct MaintenanceStats {
    std::uint64_t compaction_count = 0;
    std::uint64_t compacted_input_records = 0;
    std::uint64_t compacted_output_records = 0;
    std::size_t physical_runs = 0;
    std::size_t assignment_runs = 0;
    // Canonical Rank23 fold counters (sec 28).
    std::uint64_t canonical_fold_count = 0;
    std::uint64_t canonical_input_records = 0;
    std::uint64_t canonical_matched_records = 0;
    std::uint64_t canonical_unmatched_records = 0;
    std::uint64_t canonical_fallback_quotients = 0;
    double canonical_fold_time = 0.0;
    std::uint64_t cold_tier_compaction_count = 0;
    std::uint64_t cold_tier_input_records = 0;
    std::uint64_t cold_tier_output_records = 0;
    std::uint64_t cold_arena_overflow_records = 0;
    std::size_t raw_runs = 0;
    std::size_t stable_levels_occupied = 0;
  };

  // Read-only snapshot; run counts reflect current state.
  // Device fold counters are copied here, outside timed updates.
  MaintenanceStats maintenance_stats() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    MaintenanceStats stats = maintenance_stats_;
    if (fold_stats_.size() >= gpulsmopt_detail::kFoldStatCells) {
      std::uint32_t cells[gpulsmopt_detail::kFoldStatCells] = {};
      CUDA_CHECK(cudaMemcpy(cells, fold_stats_.data(), sizeof(cells),
                            cudaMemcpyDeviceToHost));
      stats.canonical_matched_records =
          cells[gpulsmopt_detail::kFoldStatMatched];
      stats.canonical_unmatched_records =
          cells[gpulsmopt_detail::kFoldStatUnmatched];
      stats.canonical_fallback_quotients =
          cells[gpulsmopt_detail::kFoldStatFallback];
      stats.cold_arena_overflow_records =
          cells[gpulsmopt_detail::kFoldStatOverflow];
    }
    stats.physical_runs = runs_.size();
    std::size_t assignment = 0, raw = 0;
    bool level_seen[gpulsmopt_detail::kStableLevels] = {};
    for (const auto &run : runs_) {
      if (!run.assignment)
        continue;
      ++assignment;
      if (run.assignment_class == gpulsmopt_detail::AssignmentClass::Raw)
        ++raw;
      else if (run.stable_level >= 0 &&
               run.stable_level < gpulsmopt_detail::kStableLevels)
        level_seen[run.stable_level] = true;
    }
    stats.assignment_runs = assignment;
    stats.raw_runs = raw;
    std::size_t occupied = 0;
    for (bool s : level_seen)
      occupied += s ? 1u : 0u;
    stats.stable_levels_occupied = occupied;
    return stats;
  }

explicit GPULSMOpt(const DictionaryConfig &config)
      : max_elements_(config.max_elements),
        batch_capacity_(config.batch_capacity) {
    if (max_elements_ > 0x7fffffffu) {
      throw std::invalid_argument(
          "GPULSMOpt currently supports at most 2^31-1 records");
    }
    runs_.reserve(gpulsmopt_detail::kRunCapacity);
    run_pool_.reserve(gpulsmopt_detail::kRunCapacity);
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void **>(&host_state_), sizeof(*host_state_)));
    const cudaError_t event_error =
        cudaEventCreateWithFlags(&stream_handoff_, cudaEventDisableTiming);
    if (event_error != cudaSuccess) {
      cudaFreeHost(host_state_);
      host_state_ = nullptr;
      CUDA_CHECK(event_error);
    }
  }

  ~GPULSMOpt() {
    if (stream_handoff_)
      cudaEventDestroy(stream_handoff_);
    if (host_state_)
      cudaFreeHost(host_state_);
  }

  void clear(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    order_stream_locked(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    clear_run_state();
    live_count_ = 0;
    run_sequence_ = 0;
    chrono_views_.clear();
    invalidate_resolved();
    succ_sparse_ready_ = false;
    reset_canonical_overlay(stream);
    reset_cold_arena(stream);
    ++base_generation_;
  }

  void insert(const DeviceKeyValueBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    {
      std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
      order_stream_locked(stream);
#ifdef GPULSMOPT_PROFILE_INSERT
      reset_insert_prof_();
      const auto prof_t0 = std::chrono::high_resolution_clock::now();
#endif
      insert_records(batch.keys, batch.values,
                     static_cast<std::uint8_t>(gpulsmopt_detail::kInsert),
                     batch.count, false, stream);
#ifdef GPULSMOPT_PROFILE_INSERT
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const auto prof_t1 = std::chrono::high_resolution_clock::now();
      const double total =
          std::chrono::duration<double, std::milli>(prof_t1 - prof_t0).count();
      const double measured = prof_delta_sort_ms_ + prof_delta_ingest_ms_;
      const double other = total - measured;
      auto pct = [total](double x) {
        return total > 0.0 ? 100.0 * x / total : 0.0;
      };
      printf("[prof] insert %zu keys: total=%.3f ms\n", batch.count, total);
      printf("[prof]   delta_sort  = %.3f ms (%5.1f%%)\n", prof_delta_sort_ms_,
             pct(prof_delta_sort_ms_));
      printf("[prof]   delta_write = %.3f ms (%5.1f%%)\n",
             prof_delta_ingest_ms_, pct(prof_delta_ingest_ms_));
      printf("[prof]   other/host  = %.3f ms (%5.1f%%)\n", other, pct(other));
#endif
    }
  }

  void erase(const DeviceKeyBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    {
      std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
      order_stream_locked(stream);
#ifdef GPULSMOPT_PROFILE_INSERT
      reset_insert_prof_();
      const auto prof_t0 = std::chrono::high_resolution_clock::now();
#endif
      insert_records(batch.keys, batch.keys,
                     static_cast<std::uint8_t>(gpulsmopt_detail::kTombstone),
                     batch.count, batch.sorted, stream);
#ifdef GPULSMOPT_PROFILE_INSERT
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const auto prof_t1 = std::chrono::high_resolution_clock::now();
      const double total =
          std::chrono::duration<double, std::milli>(prof_t1 - prof_t0).count();
      printf("[prof] delete %zu keys: total=%.3f ms\n", batch.count, total);
#endif
    }
  }

  void lookup(const DeviceLookupBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    order_stream_locked(stream);
    ensure_sorted_run_cache(stream);
    const int run_count = static_cast<int>(chrono_views_.size());
    const bool run_parallel =
        batch.count <=
            static_cast<std::size_t>(
                GPULSMOPT_LOOKUP_RUN_PARALLEL_MAX_QUERIES) &&
        run_count >= GPULSMOPT_LOOKUP_RUN_PARALLEL_MIN_RUNS;
    if (run_parallel) {
      constexpr int block = gpulsmopt_detail::kRunCapacity;
      const int grid = static_cast<int>(batch.count);
      gpulsmopt_detail::temporal_lookup_run_parallel_kernel<<<
          grid, block, 0, stream>>>(
          assignment_views_.data(), run_count, make_canonical_base_view(),
          batch.queries, batch.count,
          batch.out_values, batch.out_found);
    } else {
      const int block = 256;
      const int grid = static_cast<int>((batch.count + block - 1) / block);
      gpulsmopt_detail::temporal_lookup_kernel<<<grid, block, 0, stream>>>(
          assignment_views_.data(), run_count, make_canonical_base_view(),
          batch.queries, batch.count,
          batch.out_values, batch.out_found);
    }
    CUDA_CHECK(cudaGetLastError());
  }

  void successor(const DeviceSuccessorBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    order_stream_locked(stream);
    ensure_sorted_run_cache(stream);
    constexpr int block = 256;
    const int grid = static_cast<int>((batch.count + block - 1) / block);
    const int run_count = static_cast<int>(chrono_views_.size());
    if (run_count == 0) {
      gpulsmopt_detail::base_successor_kernel<<<grid, block, 0, stream>>>(
          make_canonical_base_view(), batch.queries, batch.count,
          batch.out_keys);
      CUDA_CHECK(cudaGetLastError());
      return;
    }
    if (sparse_view_is_current()) {
      gpulsmopt_detail::sparse_successor_kernel<<<grid, block, 0, stream>>>(
          make_sparse_successor_view(), batch.queries, nullptr, batch.count,
          batch.out_keys);
      CUDA_CHECK(cudaGetLastError());
      return;
    }
    succ_miss_indices_.resize_discard(batch.count);
    succ_miss_count_.resize_discard(1);
    CUDA_CHECK(cudaMemsetAsync(succ_miss_count_.data(), 0,
                               sizeof(std::uint32_t), stream));
    gpulsmopt_detail::successor_live_or_miss_kernel<<<grid, block, 0, stream>>>(
        assignment_views_.data(), run_count, make_canonical_base_view(),
        batch.queries, batch.count, batch.out_keys, succ_miss_indices_.data(),
        succ_miss_count_.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(&host_state_->successor_miss_count,
                               succ_miss_count_.data(), sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::uint32_t misses = host_state_->successor_miss_count;
    if (misses == 0u)
      return;
    ensure_sparse_successor_view(stream);
    const int miss_grid = static_cast<int>((misses + block - 1u) / block);
    gpulsmopt_detail::sparse_successor_kernel<<<miss_grid, block, 0, stream>>>(
        make_sparse_successor_view(), batch.queries, succ_miss_indices_.data(),
        misses, batch.out_keys);
    CUDA_CHECK(cudaGetLastError());
  }

  void range(const DeviceRangeOutputBatch &batch, cudaStream_t stream) {
    if (batch.query_count == 0)
      return;
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    order_stream_locked(stream);
    ensure_sorted_run_cache(stream);
    ensure_canonical_value_prefix(stream);
    if (batch.out_counts)
      ensure_canonical_count_prefix(stream);
    const int run_count = static_cast<int>(chrono_views_.size());
    if (run_count == 0) {
      const int block = 128;
      const int grid =
          static_cast<int>((batch.query_count + block - 1) / block);
      if (batch.out_counts) {
        gpulsmopt_detail::base_only_range_kernel<true><<<
            grid, block, 0, stream>>>(
            batch.lo, batch.hi, batch.out_sums, batch.out_counts,
            batch.query_count, make_canonical_base_view(),
            make_sorted_range_view(), sorted_value_prefix_.data(),
            sorted_count_prefix_.data());
      } else {
        gpulsmopt_detail::base_only_range_kernel<false><<<
            grid, block, 0, stream>>>(
            batch.lo, batch.hi, batch.out_sums, nullptr,
            batch.query_count, make_canonical_base_view(),
            make_sorted_range_view(), sorted_value_prefix_.data(),
            nullptr);
      }
      CUDA_CHECK(cudaGetLastError());
      return;
    }
    const bool try_narrow =
        !resolved_ready_ &&
        batch.query_count <=
            static_cast<std::size_t>(
                GPULSMOPT_NARROW_RANGE_MAX_QUERIES) &&
        run_count > 0;
    if (try_narrow) {
      narrow_overflow_.resize_discard(1);
      CUDA_CHECK(cudaMemsetAsync(narrow_overflow_.data(), 0,
                                 sizeof(std::uint32_t), stream));
      constexpr int block = gpulsmopt_detail::kNarrowSeenCap;
      const int grid = static_cast<int>(batch.query_count);
      gpulsmopt_detail::narrow_range_kernel<<<grid, block, 0, stream>>>(
          batch.lo, batch.hi, batch.out_sums, batch.out_counts,
          batch.query_count, assignment_views_.data(), run_count,
          make_canonical_base_view(), make_sorted_range_view(),
          sorted_value_prefix_.data(),
          sorted_count_prefix_.data(), narrow_overflow_.data());
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpyAsync(&host_state_->narrow_overflow,
                                 narrow_overflow_.data(),
                                 sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      if (host_state_->narrow_overflow == 0u)
        return;
    }
    ensure_resolved(stream);
    if (batch.out_counts)
      ensure_resolved_count_prefix(stream);
    ensure_resolved_value_prefix(stream);
    const int block = 128;
    const int grid =
        static_cast<int>((batch.query_count + block - 1) / block);
    if (batch.out_counts) {
      gpulsmopt_detail::resolved_range_kernel<true><<<
          grid, block, 0, stream>>>(
          batch.lo, batch.hi, batch.out_sums, batch.out_counts,
          batch.query_count, make_canonical_base_view(),
          make_sorted_range_view(), sorted_value_prefix_.data(),
          sorted_count_prefix_.data(), make_run_view(resolved_),
          resolved_value_prefix_.data(),
          resolved_count_prefix_.data(),
          resolved_.count > 0 ? 1 : 0);
    } else {
      gpulsmopt_detail::resolved_range_kernel<false><<<
          grid, block, 0, stream>>>(
          batch.lo, batch.hi, batch.out_sums, nullptr,
          batch.query_count, make_canonical_base_view(),
          make_sorted_range_view(), sorted_value_prefix_.data(),
          nullptr, make_run_view(resolved_),
          resolved_value_prefix_.data(), nullptr,
          resolved_.count > 0 ? 1 : 0);
    }
    CUDA_CHECK(cudaGetLastError());
  }

  void consolidate(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    order_stream_locked(stream);
    fold_into_base(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void bulk_build(const std::uint32_t *keys, const std::uint32_t *values,
                  std::size_t n, cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    order_stream_locked(stream);
    clear_run_state();
    live_count_ = 0;
    run_sequence_ = 0;
    chrono_views_.clear();
    invalidate_resolved();
    succ_sparse_ready_ = false;
    ++base_generation_;
    maintenance_stats_ = MaintenanceStats{};
    if (n == 0) {
      prepare_for_insert(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      return;
    }
    sort_direct_batch(keys, values, n, stream);
    acquire_run_slot();
    RunStorage &run = runs_.back();
    run.count = n;
    run.fully_sorted = true;
    run.unit_counts = true;
    run.unique_keys = true;
    run.keys.resize_discard(n);
    run.values.resize_discard(n);
    auto policy = thrust::cuda::par.on(stream);
    auto unique_end = thrust::reduce_by_key(
        policy, direct_sort_keys_.data(), direct_sort_keys_.data() + n,
        direct_sort_values_.data(), run.keys.data(), run.values.data(),
        thrust::equal_to<std::uint32_t>(), gpulsmopt_detail::TakeLastU32{});
    run.count = static_cast<std::size_t>(unique_end.first - run.keys.data());
    run.keys.resize_discard(run.count);
    run.values.resize_discard(run.count);
    build_assignment_offsets(run, static_cast<std::uint32_t>(run.count), stream);
    build_sorted_run_cache(0u, stream);
    live_count_ = run.count;
    allocate_canonical_overlay(run.count, stream);
    prepare_for_insert(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Allocate + zero the canonical overlay for a fresh BaseRun (sec 24).
  void allocate_canonical_overlay(std::size_t base_count,
                                  cudaStream_t stream) {
    base_override_state_.resize_discard(std::max<std::size_t>(base_count, 1));
    base_override_values_.resize_discard(std::max<std::size_t>(base_count, 1));
    CUDA_CHECK(cudaMemsetAsync(base_override_state_.data(), 0,
                               base_count * sizeof(std::uint8_t), stream));
    rank23_value_delta_.resize_discard(gpulsmopt_detail::kBaseRank23Size);
    rank23_count_delta_.resize_discard(gpulsmopt_detail::kBaseRank23Size);
    CUDA_CHECK(cudaMemsetAsync(
        rank23_value_delta_.data(), 0,
        gpulsmopt_detail::kBaseRank23Size * sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        rank23_count_delta_.data(), 0,
        gpulsmopt_detail::kBaseRank23Size * sizeof(std::int32_t), stream));
    rank23_value_prefix_.resize_discard(gpulsmopt_detail::kBaseRank23Size + 1u);
    rank23_count_prefix_.resize_discard(gpulsmopt_detail::kBaseRank23Size + 1u);
    rank23_value_prefix_ready_ = false;
    rank23_count_prefix_ready_ = false;
    canonical_overlay_active_ = false;
    ++canonical_generation_;
    canonical_through_sequence_ = 0;
  }

  // Reset the overlay to identity without freeing capacity (sec 25).
  void reset_canonical_overlay(cudaStream_t stream) {
    const std::size_t base_count =
        sorted_run_ready() ? sorted_run().count : 0u;
    if (base_count > 0 && base_override_state_.size() >= base_count)
      CUDA_CHECK(cudaMemsetAsync(base_override_state_.data(), 0,
                                 base_count * sizeof(std::uint8_t), stream));
    if (rank23_value_delta_.size() >= gpulsmopt_detail::kBaseRank23Size) {
      CUDA_CHECK(cudaMemsetAsync(
          rank23_value_delta_.data(), 0,
          gpulsmopt_detail::kBaseRank23Size * sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(
          rank23_count_delta_.data(), 0,
          gpulsmopt_detail::kBaseRank23Size * sizeof(std::int32_t), stream));
    }
    rank23_value_prefix_ready_ = false;
    rank23_count_prefix_ready_ = false;
    canonical_overlay_active_ = false;
    ++canonical_generation_;
    canonical_through_sequence_ = 0;
  }

  std::size_t live_count() const {
    auto *self = const_cast<GPULSMOpt *>(this);
    std::unique_lock<std::shared_mutex> guard(self->snapshot_mutex_);
    const cudaStream_t stream = self->operation_stream_;
    self->ensure_resolved(stream);
    std::int64_t live = self->sorted_run_ready()
                            ? static_cast<std::int64_t>(
                                  self->sorted_run().count)
                            : 0;
    auto policy = thrust::cuda::par.on(stream);
    // BaseRun count + canonical count delta + pending count delta.
    if (self->canonical_overlay_active_)
      live += thrust::reduce(
          policy, self->rank23_count_delta_.data(),
          self->rank23_count_delta_.data() +
              gpulsmopt_detail::kBaseRank23Size,
          std::int64_t{0}, thrust::plus<std::int64_t>());
    if (self->resolved_.count > 0)
      live += thrust::reduce(
          policy, self->resolved_.count_delta.data(),
          self->resolved_.count_delta.data() + self->resolved_.count,
          std::int64_t{0}, thrust::plus<std::int64_t>());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    self->live_count_ = static_cast<std::size_t>(std::max<std::int64_t>(0, live));
    return self->live_count_;
  }

  std::size_t gpu_resident_bytes() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    std::size_t total = device_bytes_all(
        resolve_keys_, resolve_payload_, resolve_alt_keys_, resolve_alt_payload_,
        resolve_flags_, resolve_sel_vdelta_, resolve_count_,
        normalize_views_, norm_keys_, norm_pay_,
        cache_pay_, merge_out_keys_, merge_out_pay_, merge_flags_,
        merge_sel_keys_, merge_sel_pay_, compaction_counts_,
        compaction_offsets_, compaction_tile_offsets_,
        compaction_unique_counts_, compaction_unique_offsets_,
        compaction_unique_cursors_, narrow_overflow_, assignment_views_,
        direct_sort_keys_, direct_sort_values_, sort_temp_storage_,
        sorted_value_prefix_, sorted_count_prefix_, base_rank23_,
        sorted_range_cdf_, succ_miss_indices_, succ_miss_count_,
        succ_deleted_base_words_, succ_live_word_l1_, succ_live_word_l2_,
        succ_live_word_l3_, succ_positive_words_, succ_positive_l1_,
        succ_positive_l2_, succ_positive_l3_);
    total += device_bytes_all(
        resolved_.keys, resolved_.values, resolved_.count_delta,
        resolved_value_prefix_, resolved_count_prefix_, resolved_.quotient_off,
        resolved_.op_words);
    // Canonical overlay + fold + cold-arena scratch.
    total += device_bytes_all(
        base_override_state_, base_override_values_, rank23_value_delta_,
        rank23_count_delta_, rank23_value_prefix_, rank23_count_prefix_,
        fold_source_views_, fold_fallback_quotients_,
        fold_fallback_count_, fold_fallback_head_, fold_stats_,
        cold_arena_keys_, cold_arena_values_, cold_arena_ops_,
        cold_arena_tail_);
    for (const auto &epoch : runs_)
      total += device_bytes_all(
          epoch.keys, epoch.values, epoch.count_delta, epoch.quotient_off,
          epoch.op_words, epoch.page_counts, epoch.page_capacity);
    for (const auto &epoch : run_pool_)
      total += device_bytes_all(
          epoch.keys, epoch.values, epoch.count_delta, epoch.quotient_off,
          epoch.op_words, epoch.page_counts, epoch.page_capacity);
    return total;
  }

private:
  struct AssignmentLeafStorage {
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> keys;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> values;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_off;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> op_words;
    // Paged cold-run metadata (empty for packed runs).
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> page_counts;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> page_capacity;
  };

  struct RunStorage : AssignmentLeafStorage {
    gpulsmopt_detail::RawDeviceBuffer<std::int8_t> count_delta;
    std::size_t count = 0;
    // Temporal identity of an immutable assignment run.
    std::uint64_t sequence = 0;
    std::uint64_t sequence_begin = 0;
    std::uint64_t sequence_end = 0;
    int stable_level = -1;
    gpulsmopt_detail::RunOperation operation =
        gpulsmopt_detail::RunOperation::Insert;
    gpulsmopt_detail::AssignmentClass assignment_class =
        gpulsmopt_detail::AssignmentClass::Raw;
    bool mixed = false;
    bool assignment = false;
    bool fully_sorted = false;
    bool unit_counts = false;
    bool unique_keys = false;
    bool paged = false;
    int cold_arena_slot = -1;
  };

  void order_stream_locked(cudaStream_t stream) {
    if (!operation_stream_valid_) {
      operation_stream_ = stream;
      operation_stream_valid_ = true;
      return;
    }
    if (stream == operation_stream_)
      return;
    CUDA_CHECK(cudaEventRecord(stream_handoff_, operation_stream_));
    CUDA_CHECK(cudaStreamWaitEvent(stream, stream_handoff_, 0));
    operation_stream_ = stream;
  }

  template <class T>
  static T *raw_or_null(gpulsmopt_detail::RawDeviceBuffer<T> &v) {
    return v.size() == 0 ? nullptr : v.data();
  }
  template <class T>
  static const T *raw_or_null(const gpulsmopt_detail::RawDeviceBuffer<T> &v) {
    return v.size() == 0 ? nullptr : v.data();
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
  bool sorted_run_ready() const { return sorted_run_index_ < runs_.size(); }

  const RunStorage &sorted_run() const { return runs_[sorted_run_index_]; }

  gpulsmopt_detail::SortedRunView make_sorted_view() const {
    if (!sorted_run_ready())
      return {};
    const RunStorage &run = sorted_run();
    return {run.keys.data(),
            run.values.data(),
            base_rank23_.data(),
            run.count,
            run.unit_counts ? 1u : 0u};
  }

  // BaseRun wrapped with the canonical overlay for readers.
  gpulsmopt_detail::CanonicalBaseView make_canonical_base_view() const {
    return {make_sorted_view(),
            canonical_overlay_active_ ? base_override_state_.data() : nullptr,
            canonical_overlay_active_ ? base_override_values_.data() : nullptr,
            canonical_overlay_active_ ? rank23_value_prefix_.data() : nullptr,
            canonical_overlay_active_ ? rank23_count_prefix_.data() : nullptr,
            static_cast<std::uint8_t>(canonical_overlay_active_ ? 1u : 0u)};
  }

  gpulsmopt_detail::SortedRunRangeView make_sorted_range_view() const {
    return {sorted_range_cdf_ready_ ? sorted_range_cdf_.data() : nullptr,
            sorted_range_min_key_, sorted_range_span_};
  }





  void clear_sorted_state() {
    sorted_run_index_ = std::numeric_limits<std::size_t>::max();
    sorted_value_prefix_.resize_discard(0);
    sorted_count_prefix_.resize_discard(0);
    base_rank23_.resize_discard(0);
    sorted_range_cdf_.release();
    sorted_range_min_key_ = 0u;
    sorted_range_span_ = 0u;
    sorted_range_cdf_ready_ = false;
  }

  // Builds the flat rank directory per base.
  void build_base_rank23(RunStorage &base, cudaStream_t stream) {
    base_rank23_.resize_discard(gpulsmopt_detail::kBaseRank23Size + 1);
    std::uint32_t *dir = base_rank23_.data();
    const std::uint32_t count = static_cast<std::uint32_t>(base.count);
    if (count == 0u) {
      CUDA_CHECK(cudaMemsetAsync(dir, 0,
                                 (gpulsmopt_detail::kBaseRank23Size + 1) *
                                     sizeof(std::uint32_t),
                                 stream));
      return;
    }
    CUDA_CHECK(cudaMemsetAsync(dir, 0xff,
                               (gpulsmopt_detail::kBaseRank23Size + 1) *
                                   sizeof(std::uint32_t),
                               stream));
    constexpr int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    gpulsmopt_detail::base_rank23_boundary_kernel<<<grid, block, 0, stream>>>(
        base.keys.data(), count, dir);
    CUDA_CHECK(cudaGetLastError());
    // Empty bins inherit the next start.
    constexpr int items = gpulsmopt_detail::kBaseRank23Size + 1;
    auto rev = thrust::make_reverse_iterator(
        thrust::device_pointer_cast(dir + items));
    std::size_t bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveScan(
        nullptr, bytes, rev, rev, thrust::minimum<std::uint32_t>(), items,
        stream));
    ensure_sort_temp(bytes);
    CUDA_CHECK(cub::DeviceScan::InclusiveScan(
        sort_temp_storage_.data(), bytes, rev, rev,
        thrust::minimum<std::uint32_t>(), items, stream));
  }

  void build_sorted_metadata(cudaStream_t stream) {
    RunStorage &run = runs_[sorted_run_index_];
    const std::size_t count = run.count;
    build_base_rank23(run, stream);
    // BaseRun needs only a value prefix.
    sorted_value_prefix_.resize_discard(count + 1u);
    sorted_count_prefix_.resize_discard(0u);
    CUDA_CHECK(cudaMemsetAsync(sorted_value_prefix_.data(), 0,
                               sizeof(std::uint32_t), stream));
    if (count == 0)
      return;
    auto policy = thrust::cuda::par.on(stream);
    thrust::inclusive_scan(policy, run.values.data(), run.values.data() + count,
                           sorted_value_prefix_.data() + 1u);
  }

  void build_sorted_range_cdf(cudaStream_t stream) {
    RunStorage &run = runs_[sorted_run_index_];
    sorted_range_cdf_ready_ = false;
    sorted_range_min_key_ = 0u;
    sorted_range_span_ = 0u;
    if (run.count == 0) {
      sorted_range_cdf_.release();
      return;
    }
    std::uint32_t endpoints[2]{};
    CUDA_CHECK(cudaMemcpyAsync(endpoints, run.keys.data(),
                               sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(endpoints + 1, run.keys.data() + run.count - 1u,
                               sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::uint64_t span =
        static_cast<std::uint64_t>(endpoints[1]) - endpoints[0] + 1u;
    const std::uint64_t entries = span + 1u;
    const std::uint64_t bytes = entries * sizeof(std::uint32_t);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    (void)total_bytes;
    const bool dense_enough = span <= static_cast<std::uint64_t>(run.count) *
                                          gpulsmopt_detail::kRangeCdfMaxRatio;
    const bool reuses_storage = entries <= sorted_range_cdf_.capacity();
    const bool memory_ok =
        reuses_storage || bytes <= static_cast<std::uint64_t>(free_bytes) / 4u;
    if (!dense_enough || !memory_ok ||
        entries > std::numeric_limits<std::size_t>::max()) {
      sorted_range_cdf_.release();
      return;
    }
    const std::size_t count = static_cast<std::size_t>(entries);
    sorted_range_cdf_.resize_discard_exact(count);
    CUDA_CHECK(cudaMemsetAsync(sorted_range_cdf_.data(), 0,
                               count * sizeof(std::uint32_t), stream));
    constexpr int block = 256;
    const int grid = static_cast<int>((run.count + block - 1u) / block);
    gpulsmopt_detail::
        sorted_range_cdf_scatter_kernel<<<grid, block, 0, stream>>>(
            run.keys.data(), run.values.data(), run.count, endpoints[0],
            sorted_range_cdf_.data());
    CUDA_CHECK(cudaGetLastError());
    auto policy = thrust::cuda::par.on(stream);
    thrust::inclusive_scan(policy, sorted_range_cdf_.data(),
                           sorted_range_cdf_.data() + count,
                           sorted_range_cdf_.data());
    sorted_range_min_key_ = endpoints[0];
    sorted_range_span_ = span;
    sorted_range_cdf_ready_ = true;
  }

  void build_sorted_run_cache(std::size_t index, cudaStream_t stream) {
    clear_sorted_state();
    if (index >= runs_.size() || !runs_[index].fully_sorted ||
        !runs_[index].unique_keys)
      return;
    sorted_run_index_ = index;
    build_sorted_metadata(stream);
    build_sorted_range_cdf(stream);
  }

  void ensure_sorted_run_cache(cudaStream_t stream) {
    if (sorted_run_ready())
      return;
    std::size_t best = runs_.size();
    std::size_t best_count = 0;
    for (std::size_t r = 0; r < runs_.size(); ++r) {
      if (!runs_[r].fully_sorted || !runs_[r].unique_keys ||
          runs_[r].count <= best_count)
        continue;
      best = r;
      best_count = runs_[r].count;
    }
    if (best_count >= gpulsmopt_detail::kSortedRunMinRecords)
      build_sorted_run_cache(best, stream);
  }

  std::size_t run_count() const { return runs_.size(); }

  gpulsmopt_detail::RunView make_run_view(RunStorage &epoch) {
    return {raw_or_null(epoch.keys), raw_or_null(epoch.quotient_off)};
  }









  void reverse_min_scan_offsets(std::uint32_t *offsets, cudaStream_t stream) {
    constexpr int items = gpulsmopt_detail::kEpochQuotients + 1;
    auto reverse = thrust::make_reverse_iterator(
        thrust::device_pointer_cast(offsets + items));
    if (metadata_scan_temp_bytes_ == 0u) {
      CUDA_CHECK(cub::DeviceScan::InclusiveScan(
          nullptr, metadata_scan_temp_bytes_, reverse, reverse,
          thrust::minimum<std::uint32_t>(), items, stream));
      ensure_sort_temp(metadata_scan_temp_bytes_);
    }
    std::size_t temp_bytes = metadata_scan_temp_bytes_;
    CUDA_CHECK(cub::DeviceScan::InclusiveScan(
        sort_temp_storage_.data(), temp_bytes, reverse, reverse,
        thrust::minimum<std::uint32_t>(), items, stream));
  }







  // Builds an unresolved assignment run.
  void create_assignment_run(bool is_insert, const std::uint32_t *keys,
                             const std::uint32_t *values, std::size_t count,
                             bool keys_sorted, cudaStream_t stream) {
    acquire_run_slot();
    RunStorage &run = runs_.back();
    run.count = count;
    run.assignment = true;
    run.assignment_class = gpulsmopt_detail::AssignmentClass::Raw;
    run.paged = false;
    run.mixed = false;
    run.stable_level = -1;
    run.operation = is_insert ? gpulsmopt_detail::RunOperation::Insert
                              : gpulsmopt_detail::RunOperation::Delete;
    run.sequence = ++run_sequence_;
    run.sequence_begin = run.sequence;
    run.sequence_end = run.sequence;
    run.fully_sorted = false;
    run.unit_counts = false;
    run.unique_keys = false;
    run.keys.resize_discard(count);
    {
      GPULSMOPT_PROF_PHASE(prof_delta_sort_ms_);
      // Stable sort preserves quotient input order.
      if (is_insert) {
        run.values.resize_discard(count);
        sort_run_batch(keys, values, count, run.keys.data(),
                       run.values.data(), stream);
      } else if (keys_sorted) {
        // Fully sorted keys need only a copy.
        run.values.resize_discard(0);
        CUDA_CHECK(cudaMemcpyAsync(run.keys.data(), keys,
                                   count * sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToDevice, stream));
      } else {
        // Deletion leaves are key-only: no value traffic.
        run.values.resize_discard(0);
        sort_delete_batch(keys, count, run.keys.data(), stream);
      }
    }
    {
      GPULSMOPT_PROF_PHASE(prof_delta_ingest_ms_);
      // Leaf metadata stores quotient offsets only.
      build_assignment_offsets(run, static_cast<std::uint32_t>(count), stream);
    }
    publish_assignment_view(run, stream);
    invalidate_resolved();
    // Fold immediately after the 64th raw run is published (sec 10).
    if (count_raw_runs() >= gpulsmopt_detail::kRawFoldWidth)
      canonical_fold(stream);
  }

  std::size_t count_raw_runs() const {
    std::size_t n = 0;
    for (const auto &run : runs_)
      if (run.assignment &&
          run.assignment_class == gpulsmopt_detail::AssignmentClass::Raw)
        ++n;
    return n;
  }

  void build_assignment_offsets(RunStorage &run, std::uint32_t count,
                                 cudaStream_t stream) {
    run.quotient_off.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1);
    std::uint32_t *offsets = run.quotient_off.data();
    if (count == 0u) {
      CUDA_CHECK(cudaMemsetAsync(offsets, 0,
                                 (gpulsmopt_detail::kEpochQuotients + 1u) *
                                     sizeof(std::uint32_t),
                                 stream));
      return;
    }
    CUDA_CHECK(cudaMemsetAsync(offsets, 0xff,
                               (gpulsmopt_detail::kEpochQuotients + 1u) *
                                   sizeof(std::uint32_t),
                               stream));
    constexpr int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    gpulsmopt_detail::assignment_boundary_kernel<<<grid, block, 0, stream>>>(
        run.keys.data(), count, offsets);
    CUDA_CHECK(cudaGetLastError());
    reverse_min_scan_offsets(offsets, stream);
  }

  gpulsmopt_detail::AssignmentRunView
  make_assignment_view(RunStorage &run) {
    const bool insert =
        run.operation == gpulsmopt_detail::RunOperation::Insert;
    const bool paged = run.paged;
    if (paged && (run.cold_arena_slot < 0 ||
                  run.cold_arena_slot >=
                      gpulsmopt_detail::kColdArenaSlots))
      throw std::runtime_error("paged run has no cold arena slot");
    const std::size_t slot = paged
                                 ? static_cast<std::size_t>(
                                       run.cold_arena_slot)
                                 : 0u;
    std::uint32_t *keys = paged
                              ? cold_arena_keys_.data() +
                                    slot * cold_arena_slot_capacity_
                              : run.keys.data();
    std::uint32_t *values = paged
                                ? cold_arena_values_.data() +
                                      slot * cold_arena_slot_capacity_
                                : run.values.data();
    std::uint32_t *ops = paged
                             ? cold_arena_ops_.data() +
                                   slot * cold_arena_slot_words_
                             : run.op_words.data();
    return {keys,
            insert ? values : nullptr,
            run.quotient_off.data(),
            paged ? run.page_counts.data() : nullptr,
            run.mixed ? ops : nullptr,
            static_cast<std::uint8_t>(insert ? 1u : 0u),
            static_cast<std::uint8_t>(run.mixed ? 1u : 0u),
            static_cast<std::uint8_t>(paged ? 1u : 0u)};
  }

  // Publishes one chronological run descriptor.
  void publish_assignment_view(RunStorage &run, cudaStream_t stream) {
    gpulsmopt_detail::AssignmentRunView view =
        make_assignment_view(run);
    const std::size_t slot = chrono_views_.size();
    if (slot >= static_cast<std::size_t>(gpulsmopt_detail::kRunCapacity))
      throw std::runtime_error("assignment descriptor capacity exceeded");
    chrono_views_.push_back(view);
    host_state_->views[slot] = view;
    if (assignment_views_.size() < chrono_views_.size())
      assignment_views_.resize_discard(gpulsmopt_detail::kRunCapacity);
    CUDA_CHECK(cudaMemcpyAsync(assignment_views_.data() + slot,
                               host_state_->views + slot, sizeof(view),
                               cudaMemcpyHostToDevice, stream));
  }

  void acquire_run_slot() {
    if (run_pool_.empty()) {
      runs_.emplace_back();
      return;
    }
    runs_.push_back(std::move(run_pool_.back()));
    run_pool_.pop_back();
  }

  void acquire_compaction_slot() {
    if (run_pool_.empty()) {
      runs_.emplace_back();
      return;
    }
    auto best = std::max_element(
        run_pool_.begin(), run_pool_.end(),
        [](const RunStorage &left, const RunStorage &right) {
          return left.keys.capacity() < right.keys.capacity();
        });
    runs_.push_back(std::move(*best));
    run_pool_.erase(best);
  }

  void reserve_leaf_storage(std::size_t count) {
    while (runs_.size() + run_pool_.size() <
           static_cast<std::size_t>(gpulsmopt_detail::kRunCapacity)) {
      run_pool_.emplace_back();
    }
    // A full fold horizon of raw leaves stays warm (sec 10).
    const std::size_t warm = std::min(
        run_pool_.size(),
        std::max<std::size_t>(
            GPULSMOPT_PREWARM_LEAVES,
            gpulsmopt_detail::kRawFoldWidth + 2u));
    const std::size_t first_warm = run_pool_.size() - warm;
    for (std::size_t i = 0; i < run_pool_.size(); ++i) {
      RunStorage &leaf = run_pool_[i];
      if (i < first_warm) {
        leaf.keys.release();
        leaf.values.release();
        leaf.quotient_off.release();
        leaf.op_words.release();
        leaf.count_delta.release();
        leaf.page_counts.release();
        leaf.page_capacity.release();
        continue;
      }
      leaf.keys.resize_discard(count);
      leaf.values.resize_discard(count);
      leaf.quotient_off.resize_discard_exact(
          gpulsmopt_detail::kEpochQuotients + 1);
    }
  }

  void clear_run_state() {
    clear_sorted_state();
    for (auto &epoch : runs_)
      run_pool_.push_back(std::move(epoch));
    runs_.clear();
  }

  void insert_records(const std::uint32_t *keys_in,
                      const std::uint32_t *values_in, std::uint8_t op,
                      std::size_t count, bool keys_sorted,
                      cudaStream_t stream) {
    if (count == 0)
      return;
    const bool is_insert =
        op == static_cast<std::uint8_t>(gpulsmopt_detail::kInsert);
    create_assignment_run(is_insert, keys_in, is_insert ? values_in : nullptr,
                          count, keys_sorted, stream);
  }

  void ensure_sort_temp(std::size_t bytes) {
    if (sort_temp_storage_.capacity() < bytes)
      sort_temp_storage_.resize_discard(bytes);
  }

  void exclusive_scan_u32(const std::uint32_t *input, std::uint32_t *output,
                          std::size_t count, cudaStream_t stream) {
    if (count == 0)
      return;
    if (count > scan_u32_count_) {
      scan_u32_temp_bytes_ = 0;
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          nullptr, scan_u32_temp_bytes_, input, output, static_cast<int>(count),
          stream));
      scan_u32_count_ = count;
    }
    std::size_t temp_bytes = scan_u32_temp_bytes_;
    ensure_sort_temp(temp_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(sort_temp_storage_.data(),
                                             temp_bytes, input, output,
                                             static_cast<int>(count), stream));
  }

  void prepare_sort_storage(std::size_t direct_count, cudaStream_t stream) {
    direct_sort_keys_.resize_discard(direct_count);
    direct_sort_values_.resize_discard(direct_count);
    RunStorage &sample = run_pool_.back();
    std::size_t direct_bytes = 0;
    if (direct_count > 0) {
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, direct_bytes, sample.keys.data(), direct_sort_keys_.data(),
          sample.values.data(), direct_sort_values_.data(), direct_count, 0, 32,
          stream));
      direct_sort_count_ = direct_count;
      direct_sort_temp_bytes_ = direct_bytes;
    }
    std::size_t epoch_bytes = 0;
    if (direct_count > 0) {
      CUDA_CHECK(gpulsmopt_detail::epoch_radix_sort_pairs(
          nullptr, epoch_bytes, sample.keys.data(), direct_sort_keys_.data(),
          sample.values.data(), direct_sort_values_.data(),
          static_cast<std::uint32_t>(direct_count), 16, 32, stream));
      run_sort_count_ = direct_count;
      run_sort_temp_bytes_ = epoch_bytes;
    }
    // Resolve staging reuses the same high-water sizing.
    resolve_keys_.resize_discard(direct_count);
    resolve_payload_.resize_discard(direct_count);
    resolve_alt_keys_.resize_discard(direct_count);
    resolve_alt_payload_.resize_discard(direct_count);
    resolve_flags_.resize_discard(direct_count);
    resolve_sel_vdelta_.resize_discard(direct_count);
    resolve_count_.resize_discard(1);
    std::size_t scan_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, direct_sort_keys_.data(),
        direct_sort_values_.data(), gpulsmopt_detail::kEpochQuotients, stream));
    scan_u32_count_ = gpulsmopt_detail::kEpochQuotients;
    scan_u32_temp_bytes_ = scan_bytes;
    ensure_sort_temp(std::max({direct_bytes, epoch_bytes, scan_bytes}));
  }

  void reserve_temporal_compaction_storage(
      std::size_t run_capacity, cudaStream_t stream) {
    (void)run_capacity;
    (void)stream;
    compaction_counts_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    compaction_offsets_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    compaction_tile_offsets_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    compaction_unique_counts_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    compaction_unique_offsets_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    compaction_unique_cursors_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    host_compaction_offsets_.resize(
        gpulsmopt_detail::kEpochQuotients + 1u);
  }

  // Reserve successor storage before timed updates.
  void reserve_successor_storage() {
    const std::size_t queries = std::max<std::size_t>(1, batch_capacity_);
    succ_miss_indices_.resize_discard(queries);
    succ_miss_count_.resize_discard(1);
    const std::size_t base_capacity = std::max<std::size_t>(max_elements_, 1);
    const std::size_t l0 = (base_capacity + 31u) >> 5;
    const std::size_t l1 = (l0 + 31u) >> 5;
    const std::size_t l2 = (l1 + 31u) >> 5;
    const std::size_t l3 = (l2 + 31u) >> 5;
    succ_deleted_base_words_.resize_discard(l0);
    succ_live_word_l1_.resize_discard(l1);
    succ_live_word_l2_.resize_discard(l2);
    succ_live_word_l3_.resize_discard(l3);
    succ_positive_words_.resize_discard(l0);
    succ_positive_l1_.resize_discard(l1);
    succ_positive_l2_.resize_discard(l2);
    succ_positive_l3_.resize_discard(l3);
  }

  void reserve_fold_storage(std::size_t batch_count) {
    fold_source_views_.resize_discard(gpulsmopt_detail::kRawFoldWidth);
    fold_fallback_quotients_.resize_discard(
        gpulsmopt_detail::kEpochQuotients);
    fold_fallback_count_.resize_discard(1);
    fold_fallback_head_.resize_discard(1);
    // Build-time reset matches the host-side stats reset.
    fold_stats_.resize_discard(gpulsmopt_detail::kFoldStatCells);
    CUDA_CHECK(cudaMemset(fold_stats_.data(), 0,
                          gpulsmopt_detail::kFoldStatCells *
                              sizeof(std::uint32_t)));
    cold_arena_tail_.resize_discard(
        gpulsmopt_detail::kColdArenaSlots);
    const std::size_t batch =
        std::max<std::size_t>(batch_count, 1);
    const std::size_t fold_records =
        batch * gpulsmopt_detail::kRawFoldWidth;
    const std::size_t page_padding =
        31u * gpulsmopt_detail::kEpochQuotients;
    const std::size_t slot =
        (fold_records + page_padding + 31u) & ~std::size_t{31u};
    if (slot > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("cold fold slot exceeds 32-bit offsets");
    cold_arena_slot_capacity_ =
        static_cast<std::uint32_t>(slot);
    cold_arena_slot_words_ = slot >> 5;
    const std::size_t arena =
        slot * gpulsmopt_detail::kColdArenaSlots;
    cold_arena_keys_.resize_discard(arena);
    cold_arena_values_.resize_discard(arena);
    cold_arena_ops_.resize_discard(
        cold_arena_slot_words_ * gpulsmopt_detail::kColdArenaSlots);
  }

  void reset_cold_arena(cudaStream_t stream) {
    if (cold_arena_tail_.size() == 0)
      return;
    CUDA_CHECK(cudaMemsetAsync(
        cold_arena_tail_.data(), 0,
        gpulsmopt_detail::kColdArenaSlots * sizeof(std::uint32_t),
        stream));
    for (int slot = 0; slot < gpulsmopt_detail::kColdArenaSlots;
         ++slot)
      cold_arena_slot_used_[slot] = false;
  }

  void prepare_for_insert(cudaStream_t stream) {
    const std::size_t direct_count =
        std::min(max_elements_,
                 std::max<std::size_t>(1, batch_capacity_));
    reserve_leaf_storage(direct_count);
    assignment_views_.resize_discard_exact(gpulsmopt_detail::kRunCapacity);
    chrono_views_.reserve(gpulsmopt_detail::kRunCapacity);
    if (direct_count > 0) {
      delete_sort_temp_bytes_ = 0;
      CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
          nullptr, delete_sort_temp_bytes_, direct_sort_keys_.data(),
          direct_sort_keys_.data(), static_cast<int>(direct_count), 16, 32,
          stream));
      delete_sort_count_ = direct_count;
      ensure_sort_temp(delete_sort_temp_bytes_);
    }
    prepare_sort_storage(direct_count, stream);
    reserve_temporal_compaction_storage(direct_count, stream);
    reserve_successor_storage();
    reserve_fold_storage(direct_count);
    reset_cold_arena(stream);
    if (direct_count > 0 && !run_pool_.empty()) {
      RunStorage &sample = run_pool_.back();
      sort_run_batch(
          direct_sort_keys_.data(), direct_sort_values_.data(),
          direct_count, sample.keys.data(), sample.values.data(), stream);
      sort_delete_batch(
          direct_sort_keys_.data(), direct_count, sample.keys.data(), stream);
    }
  }

  void sort_direct_batch(const std::uint32_t *keys, const std::uint32_t *values,
                         std::size_t n, cudaStream_t stream) {
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

  // Key-only upper-16 sort for deletion leaves.
  void sort_delete_batch(const std::uint32_t *keys, std::size_t n,
                         std::uint32_t *out_keys, cudaStream_t stream) {
    if (n > delete_sort_count_) {
      delete_sort_temp_bytes_ = 0;
      CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, delete_sort_temp_bytes_,
                                                keys, out_keys,
                                                static_cast<int>(n), 16, 32,
                                                stream));
      delete_sort_count_ = n;
    }
    std::size_t temp_bytes = delete_sort_temp_bytes_;
    ensure_sort_temp(temp_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(sort_temp_storage_.data(),
                                              temp_bytes, keys, out_keys,
                                              static_cast<int>(n), 16, 32,
                                              stream));
  }

  void sort_run_batch(const std::uint32_t *keys, const std::uint32_t *values,
                      std::size_t n, std::uint32_t *out_keys,
                      std::uint32_t *out_values, cudaStream_t stream) {
    if (n > run_sort_count_) {
      run_sort_temp_bytes_ = 0;
      CUDA_CHECK(gpulsmopt_detail::epoch_radix_sort_pairs(
          nullptr, run_sort_temp_bytes_, keys, out_keys, values, out_values,
          static_cast<std::uint32_t>(n), 16, 32, stream));
      run_sort_count_ = n;
    }
    std::size_t temp_bytes = run_sort_temp_bytes_;
    ensure_sort_temp(temp_bytes);
    CUDA_CHECK(gpulsmopt_detail::epoch_radix_sort_pairs(
        sort_temp_storage_.data(), temp_bytes, keys, out_keys, values,
        out_values, static_cast<std::uint32_t>(n), 16, 32, stream));
  }




  bool sparse_view_is_current() const {
    return succ_sparse_ready_ &&
           succ_sparse_base_generation_ == base_generation_ &&
           succ_sparse_run_sequence_ == run_sequence_;
  }

  gpulsmopt_detail::SuccessorSparseView make_sparse_successor_view() const {
    return {make_sorted_view(),
            succ_deleted_base_words_.data(),
            succ_live_word_l1_.data(),
            succ_live_word_l2_.data(),
            succ_live_word_l3_.data(),
            resolved_.keys.data(),
            resolved_.quotient_off.data(),
            succ_positive_words_.data(),
            succ_positive_l1_.data(),
            succ_positive_l2_.data(),
            succ_positive_l3_.data(),
            static_cast<std::uint32_t>(resolved_.count),
            succ_sparse_l0_words_,
            succ_sparse_l3_words_,
            succ_sparse_positive_l0_words_,
            succ_sparse_positive_l3_words_};
  }

  // Builds sparse state only after a successor miss.
  void ensure_sparse_successor_view(cudaStream_t stream) {
    ensure_resolved(stream);
    constexpr int block = 256;
    const std::uint32_t base_count =
        sorted_run_ready() ? static_cast<std::uint32_t>(sorted_run().count) : 0u;
    const std::uint32_t l0 = (base_count + 31u) >> 5;
    const std::uint32_t l1 = (l0 + 31u) >> 5;
    const std::uint32_t l2 = (l1 + 31u) >> 5;
    const std::uint32_t l3 = (l2 + 31u) >> 5;
    const std::size_t corrections = resolved_.count;
    const std::uint32_t p0 =
        (static_cast<std::uint32_t>(corrections) + 31u) >> 5;
    const std::uint32_t p1 = (p0 + 31u) >> 5;
    const std::uint32_t p2 = (p1 + 31u) >> 5;
    const std::uint32_t p3 = (p2 + 31u) >> 5;
    succ_sparse_l0_words_ = l0;
    succ_sparse_l3_words_ = l3;
    succ_sparse_positive_l0_words_ = p0;
    succ_sparse_positive_l3_words_ = p3;
    succ_deleted_base_words_.resize_discard(std::max<std::uint32_t>(l0, 1u));
    succ_live_word_l1_.resize_discard(std::max<std::uint32_t>(l1, 1u));
    succ_live_word_l2_.resize_discard(std::max<std::uint32_t>(l2, 1u));
    succ_live_word_l3_.resize_discard(std::max<std::uint32_t>(l3, 1u));
    succ_positive_words_.resize_discard(std::max<std::uint32_t>(p0, 1u));
    succ_positive_l1_.resize_discard(std::max<std::uint32_t>(p1, 1u));
    succ_positive_l2_.resize_discard(std::max<std::uint32_t>(p2, 1u));
    succ_positive_l3_.resize_discard(std::max<std::uint32_t>(p3, 1u));
    if (l0 != 0u)
      CUDA_CHECK(cudaMemsetAsync(succ_deleted_base_words_.data(), 0,
                                 l0 * sizeof(std::uint32_t), stream));
    if (p0 != 0u)
      CUDA_CHECK(cudaMemsetAsync(succ_positive_words_.data(), 0,
                                 p0 * sizeof(std::uint32_t), stream));
    // Canonical-dead BaseRun positions are also successor-invisible.
    if (canonical_overlay_active_ && base_count > 0) {
      const int grid = static_cast<int>((base_count + block - 1u) / block);
      gpulsmopt_detail::succ_seed_canonical_dead_kernel<<<
          grid, block, 0, stream>>>(base_override_state_.data(), base_count,
                                    succ_deleted_base_words_.data());
      CUDA_CHECK(cudaGetLastError());
    }
    if (corrections > 0) {
      const int grid = static_cast<int>((corrections + block - 1u) / block);
      gpulsmopt_detail::successor_classify_kernel<<<grid, block, 0, stream>>>(
          resolved_.keys.data(), resolved_.count_delta.data(), corrections,
          make_sorted_view(), succ_positive_words_.data(),
          succ_deleted_base_words_.data());
      CUDA_CHECK(cudaGetLastError());
    }
    if (l0 != 0u) {
      gpulsmopt_detail::successor_tail_mask_kernel<<<1, 1, 0, stream>>>(
          succ_deleted_base_words_.data(), l0, base_count);
      CUDA_CHECK(cudaGetLastError());
      gpulsmopt_detail::successor_live_level_kernel<<<
          static_cast<int>((l1 + block - 1u) / block), block, 0, stream>>>(
          succ_deleted_base_words_.data(), l0, succ_live_word_l1_.data(), l1, 1);
      gpulsmopt_detail::successor_live_level_kernel<<<
          static_cast<int>((l2 + block - 1u) / block), block, 0, stream>>>(
          succ_live_word_l1_.data(), l1, succ_live_word_l2_.data(), l2, 0);
      gpulsmopt_detail::successor_live_level_kernel<<<
          static_cast<int>((l3 + block - 1u) / block), block, 0, stream>>>(
          succ_live_word_l2_.data(), l2, succ_live_word_l3_.data(), l3, 0);
      CUDA_CHECK(cudaGetLastError());
    }
    if (p0 != 0u) {
      gpulsmopt_detail::successor_live_level_kernel<<<
          static_cast<int>((p1 + block - 1u) / block), block, 0, stream>>>(
          succ_positive_words_.data(), p0, succ_positive_l1_.data(), p1, 0);
      gpulsmopt_detail::successor_live_level_kernel<<<
          static_cast<int>((p2 + block - 1u) / block), block, 0, stream>>>(
          succ_positive_l1_.data(), p1, succ_positive_l2_.data(), p2, 0);
      gpulsmopt_detail::successor_live_level_kernel<<<
          static_cast<int>((p3 + block - 1u) / block), block, 0, stream>>>(
          succ_positive_l2_.data(), p2, succ_positive_l3_.data(), p3, 0);
      CUDA_CHECK(cudaGetLastError());
    }
    succ_sparse_base_generation_ = base_generation_;
    succ_sparse_run_sequence_ = run_sequence_;
    succ_sparse_ready_ = true;
  }

  void invalidate_resolved() {
    resolved_ready_ = false;
    resolved_value_prefix_ready_ = false;
    resolved_count_prefix_ready_ = false;
  }

  // Gather by quotient and sort only unseen low bits. Paged run
  // counts are upper bounds; the scan yields the exact total.
  std::size_t normalize_runs(const std::vector<std::size_t> &idx,
                             cudaStream_t stream) {
    constexpr int block = 256;
    normalize_views_.resize_discard(
        gpulsmopt_detail::kRunCapacity);
    for (std::size_t slot = 0; slot < idx.size(); ++slot)
      host_state_->views[slot] = make_assignment_view(runs_[idx[slot]]);
    CUDA_CHECK(cudaMemcpyAsync(
        normalize_views_.data(), host_state_->views,
        idx.size() * sizeof(gpulsmopt_detail::AssignmentRunView),
        cudaMemcpyHostToDevice, stream));

    compaction_counts_.resize_discard(
        gpulsmopt_detail::kEpochQuotients + 1u);
    compaction_offsets_.resize_discard(
        gpulsmopt_detail::kEpochQuotients + 1u);
    constexpr int rows = gpulsmopt_detail::kEpochQuotients + 1;
    constexpr int count_grid = (rows + block - 1) / block;
    gpulsmopt_detail::assignment_group_count_kernel<<<
        count_grid, block, 0, stream>>>(
        normalize_views_.data(), static_cast<int>(idx.size()),
        compaction_counts_.data());
    CUDA_CHECK(cudaGetLastError());
    exclusive_scan_u32(
        compaction_counts_.data(), compaction_offsets_.data(),
        gpulsmopt_detail::kEpochQuotients + 1u, stream);
    CUDA_CHECK(cudaMemcpyAsync(
        &host_state_->gathered_count,
        compaction_offsets_.data() + gpulsmopt_detail::kEpochQuotients,
        sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::size_t total = host_state_->gathered_count;
    if (total == 0)
      return 0;
    if (total > static_cast<std::size_t>(
                    std::numeric_limits<int>::max()))
      throw std::runtime_error("resolved cache exceeds CUB limits");

    resolve_keys_.resize_discard(total);
    resolve_payload_.resize_discard(total);
    resolve_alt_keys_.resize_discard(total);
    resolve_alt_payload_.resize_discard(total);
    gpulsmopt_detail::assignment_group_gather_kernel<<<
        gpulsmopt_detail::kEpochQuotients, block, 0, stream>>>(
        normalize_views_.data(), static_cast<int>(idx.size()),
        compaction_offsets_.data(), resolve_keys_.data(),
        resolve_payload_.data());
    CUDA_CHECK(cudaGetLastError());

    if (total > resolved_sort_count_) {
      resolved_sort_temp_bytes_ = 0u;
      CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
          nullptr, resolved_sort_temp_bytes_, resolve_keys_.data(),
          resolve_alt_keys_.data(), resolve_payload_.data(),
          resolve_alt_payload_.data(), static_cast<int>(total),
          gpulsmopt_detail::kEpochQuotients,
          compaction_offsets_.data(),
          compaction_offsets_.data() + 1u, 0, 16, stream));
      resolved_sort_count_ = total;
    }
    std::size_t sort_bytes = resolved_sort_temp_bytes_;
    ensure_sort_temp(sort_bytes);
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
        sort_temp_storage_.data(), sort_bytes, resolve_keys_.data(),
        resolve_alt_keys_.data(), resolve_payload_.data(),
        resolve_alt_payload_.data(), static_cast<int>(total),
        gpulsmopt_detail::kEpochQuotients,
        compaction_offsets_.data(),
        compaction_offsets_.data() + 1u, 0, 16, stream));

    norm_keys_.resize_discard(total);
    norm_pay_.resize_discard(total);
    const int grid =
        static_cast<int>((total + block - 1u) / block);
    gpulsmopt_detail::normalize_correction_kernel<<<
        grid, block, 0, stream>>>(
        resolve_alt_keys_.data(), resolve_alt_payload_.data(), total,
        make_canonical_base_view(), norm_keys_.data(),
        norm_pay_.data());
    CUDA_CHECK(cudaGetLastError());
    return total;
  }

  std::uint32_t select_resolved(const std::uint64_t *keys,
                                const std::uint64_t *pay,
                                std::size_t total,
                                cudaStream_t stream) {
    if (total == 0)
      return 0u;
    if (total > static_cast<std::size_t>(
                    std::numeric_limits<int>::max()))
      throw std::runtime_error("resolved selection exceeds CUB limits");
    constexpr int block = 256;
    const int grid =
        static_cast<int>((total + block - 1u) / block);
    merge_flags_.resize_discard(total);
    merge_sel_keys_.resize_discard(total);
    merge_sel_pay_.resize_discard(total);
    resolve_count_.resize_discard(1);
    gpulsmopt_detail::resolve_merge_flag_kernel<<<
        grid, block, 0, stream>>>(
        keys, pay, total, merge_flags_.data());
    CUDA_CHECK(cudaGetLastError());

    std::size_t key_bytes = 0u;
    std::size_t pay_bytes = 0u;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, key_bytes, keys, merge_flags_.data(),
        merge_sel_keys_.data(), resolve_count_.data(),
        static_cast<int>(total), stream));
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, pay_bytes, pay, merge_flags_.data(),
        merge_sel_pay_.data(), resolve_count_.data(),
        static_cast<int>(total), stream));
    ensure_sort_temp(std::max(key_bytes, pay_bytes));
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        sort_temp_storage_.data(), key_bytes, keys,
        merge_flags_.data(), merge_sel_keys_.data(),
        resolve_count_.data(), static_cast<int>(total), stream));
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        sort_temp_storage_.data(), pay_bytes, pay,
        merge_flags_.data(), merge_sel_pay_.data(),
        resolve_count_.data(), static_cast<int>(total), stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &host_state_->resolved_count, resolve_count_.data(),
        sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host_state_->resolved_count;
  }

  void store_resolved(const std::uint64_t *keys,
                      const std::uint64_t *pay,
                      std::uint32_t changed,
                      cudaStream_t stream) {
    constexpr int block = 256;
    resolved_.count = changed;
    resolved_.assignment = false;
    resolved_.fully_sorted = true;
    resolved_.unit_counts = false;
    resolved_.unique_keys = true;
    resolved_.keys.resize_discard(changed);
    resolved_.values.resize_discard(changed);
    resolved_.count_delta.resize_discard(changed);
    if (changed > 0) {
      const int grid =
          static_cast<int>((changed + block - 1u) / block);
      gpulsmopt_detail::corr_unpack_kernel<<<
          grid, block, 0, stream>>>(
          keys, pay, changed, resolved_.keys.data(),
          resolved_.values.data(), resolved_.count_delta.data());
      CUDA_CHECK(cudaGetLastError());
    }
    build_assignment_offsets(resolved_, changed, stream);
    resolved_value_prefix_ready_ = false;
    resolved_count_prefix_ready_ = false;
    resolved_ready_ = true;
  }

  // Extend the cache only through unseen run sequences.
  void ensure_resolved(cudaStream_t stream) {
    // Canonical baseline moved: rebuild from surviving runs (sec 20.4).
    if (resolved_base_generation_ != base_generation_ ||
        resolved_canonical_generation_ != canonical_generation_) {
      resolved_.count = 0;
      resolved_through_sequence_ = 0;
      resolved_base_generation_ = base_generation_;
      resolved_canonical_generation_ = canonical_generation_;
      resolved_ready_ = false;
    }
    std::vector<std::size_t> nidx;
    std::size_t upper = 0;
    for (std::size_t r = 0; r < runs_.size(); ++r) {
      if (runs_[r].assignment &&
          runs_[r].sequence_end > resolved_through_sequence_) {
        nidx.push_back(r);
        upper += runs_[r].count;
      }
    }
    std::sort(nidx.begin(), nidx.end(),
              [this](std::size_t a, std::size_t b) {
                return runs_[a].sequence_end < runs_[b].sequence_end;
              });
    // Paged counts are upper bounds; the gather reports the truth.
    const std::size_t total =
        (nidx.empty() || upper == 0) ? 0 : normalize_runs(nidx, stream);
    if (total == 0) {
      resolved_through_sequence_ = run_sequence_;
      if (!resolved_ready_) {
        build_assignment_offsets(resolved_, resolved_.count, stream);
        resolved_value_prefix_ready_ = false;
        resolved_count_prefix_ready_ = false;
        resolved_ready_ = true;
      }
      return;
    }
    const std::uint64_t *candidate_keys = norm_keys_.data();
    const std::uint64_t *candidate_pay = norm_pay_.data();
    std::size_t candidate_count = total;
    if (resolved_.count > 0) {
      const std::uint32_t old_count =
          static_cast<std::uint32_t>(resolved_.count);
      cache_pay_.resize_discard(old_count);
      constexpr int block = 256;
      const int grid =
          static_cast<int>((old_count + block - 1u) / block);
      gpulsmopt_detail::corr_pack_kernel<<<
          grid, block, 0, stream>>>(
          resolved_.values.data(), resolved_.count_delta.data(),
          old_count, cache_pay_.data());
      CUDA_CHECK(cudaGetLastError());

      candidate_count += old_count;
      if (candidate_count > static_cast<std::size_t>(
                                std::numeric_limits<int>::max()))
        throw std::runtime_error("resolved merge exceeds CUB limits");
      merge_out_keys_.resize_discard(candidate_count);
      merge_out_pay_.resize_discard(candidate_count);
      using TaggedIterator = cub::TransformInputIterator<
          std::uint64_t, gpulsmopt_detail::CacheTaggedKey,
          const std::uint32_t *>;
      TaggedIterator old_keys(
          resolved_.keys.data(),
          gpulsmopt_detail::CacheTaggedKey{});
      std::size_t merge_bytes = 0u;
      CUDA_CHECK(cub::DeviceMerge::MergePairs(
          nullptr, merge_bytes, old_keys, cache_pay_.data(),
          static_cast<int>(old_count), norm_keys_.data(),
          norm_pay_.data(), static_cast<int>(total),
          merge_out_keys_.data(), merge_out_pay_.data(),
          thrust::less<std::uint64_t>{}, stream));
      ensure_sort_temp(merge_bytes);
      CUDA_CHECK(cub::DeviceMerge::MergePairs(
          sort_temp_storage_.data(), merge_bytes, old_keys,
          cache_pay_.data(), static_cast<int>(old_count),
          norm_keys_.data(), norm_pay_.data(),
          static_cast<int>(total), merge_out_keys_.data(),
          merge_out_pay_.data(), thrust::less<std::uint64_t>{},
          stream));
      candidate_keys = merge_out_keys_.data();
      candidate_pay = merge_out_pay_.data();
    }

    const std::uint32_t changed = select_resolved(
        candidate_keys, candidate_pay, candidate_count, stream);
    store_resolved(merge_sel_keys_.data(),
                   merge_sel_pay_.data(), changed, stream);
    resolved_through_sequence_ = run_sequence_;
  }

  void ensure_resolved_value_prefix(cudaStream_t stream) {
    if (resolved_value_prefix_ready_)
      return;
    const std::size_t count = resolved_.count;
    resolved_value_prefix_.resize_discard(count + 1u);
    CUDA_CHECK(cudaMemsetAsync(
        resolved_value_prefix_.data(), 0, sizeof(std::uint32_t), stream));
    if (count > 0u) {
      if (count > resolved_value_scan_count_) {
        resolved_value_scan_temp_bytes_ = 0u;
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            nullptr, resolved_value_scan_temp_bytes_,
            resolved_.values.data(),
            resolved_value_prefix_.data() + 1u,
            static_cast<int>(count), stream));
        resolved_value_scan_count_ = count;
      }
      std::size_t temp_bytes = resolved_value_scan_temp_bytes_;
      ensure_sort_temp(temp_bytes);
      CUDA_CHECK(cub::DeviceScan::InclusiveSum(
          sort_temp_storage_.data(), temp_bytes,
          resolved_.values.data(),
          resolved_value_prefix_.data() + 1u,
          static_cast<int>(count), stream));
    }
    resolved_value_prefix_ready_ = true;
  }

  void ensure_resolved_count_prefix(cudaStream_t stream) {
    if (resolved_count_prefix_ready_)
      return;
    const std::size_t count = resolved_.count;
    resolved_count_prefix_.resize_discard(count + 1u);
    CUDA_CHECK(cudaMemsetAsync(
        resolved_count_prefix_.data(), 0, sizeof(std::uint32_t), stream));
    if (count > 0u) {
      using CountIterator = cub::TransformInputIterator<
          std::uint32_t, gpulsmopt_detail::CountDeltaToU32,
          const std::int8_t *>;
      CountIterator input(
          resolved_.count_delta.data(),
          gpulsmopt_detail::CountDeltaToU32{});
      if (count > resolved_count_scan_count_) {
        resolved_count_scan_temp_bytes_ = 0u;
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            nullptr, resolved_count_scan_temp_bytes_, input,
            resolved_count_prefix_.data() + 1u,
            static_cast<int>(count), stream));
        resolved_count_scan_count_ = count;
      }
      std::size_t temp_bytes = resolved_count_scan_temp_bytes_;
      ensure_sort_temp(temp_bytes);
      CUDA_CHECK(cub::DeviceScan::InclusiveSum(
          sort_temp_storage_.data(), temp_bytes, input,
          resolved_count_prefix_.data() + 1u,
          static_cast<int>(count), stream));
    }
    resolved_count_prefix_ready_ = true;
  }

  // Rank23 value-delta prefix over 2^23 bins (sec 20.1).
  void ensure_canonical_value_prefix(cudaStream_t stream) {
    if (!canonical_overlay_active_ || rank23_value_prefix_ready_)
      return;
    const int bins = static_cast<int>(gpulsmopt_detail::kBaseRank23Size);
    rank23_value_prefix_.resize_discard(gpulsmopt_detail::kBaseRank23Size + 1u);
    CUDA_CHECK(cudaMemsetAsync(rank23_value_prefix_.data(), 0,
                               sizeof(std::uint32_t), stream));
    if (rank23_value_scan_bytes_ == 0u) {
      CUDA_CHECK(cub::DeviceScan::InclusiveSum(
          nullptr, rank23_value_scan_bytes_, rank23_value_delta_.data(),
          rank23_value_prefix_.data() + 1u, bins, stream));
    }
    ensure_sort_temp(rank23_value_scan_bytes_);
    std::size_t bytes = rank23_value_scan_bytes_;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        sort_temp_storage_.data(), bytes, rank23_value_delta_.data(),
        rank23_value_prefix_.data() + 1u, bins, stream));
    rank23_value_prefix_ready_ = true;
  }

  // Rank23 count-delta prefix (signed, stored mod 2^32).
  void ensure_canonical_count_prefix(cudaStream_t stream) {
    if (!canonical_overlay_active_ || rank23_count_prefix_ready_)
      return;
    const int bins = static_cast<int>(gpulsmopt_detail::kBaseRank23Size);
    rank23_count_prefix_.resize_discard(gpulsmopt_detail::kBaseRank23Size + 1u);
    CUDA_CHECK(cudaMemsetAsync(rank23_count_prefix_.data(), 0,
                               sizeof(std::uint32_t), stream));
    auto *in = reinterpret_cast<const std::uint32_t *>(
        rank23_count_delta_.data());
    if (rank23_count_scan_bytes_ == 0u) {
      CUDA_CHECK(cub::DeviceScan::InclusiveSum(
          nullptr, rank23_count_scan_bytes_, in,
          rank23_count_prefix_.data() + 1u, bins, stream));
    }
    ensure_sort_temp(rank23_count_scan_bytes_);
    std::size_t bytes = rank23_count_scan_bytes_;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        sort_temp_storage_.data(), bytes, in,
        rank23_count_prefix_.data() + 1u, bins, stream));
    rank23_count_prefix_ready_ = true;
  }

  // Rewrite the BaseRun to its canonical visible state, then reset
  // the overlay to identity (used only by full consolidation).
  void bake_overlay_into_base(cudaStream_t stream) {
    if (!canonical_overlay_active_ || !sorted_run_ready())
      return;
    RunStorage &base = runs_[sorted_run_index_];
    const std::size_t n = base.count;
    if (n == 0) {
      reset_canonical_overlay(stream);
      return;
    }
    resolve_alt_keys_.resize_discard(n);
    resolve_sel_vdelta_.resize_discard(n);
    resolve_flags_.resize_discard(n);
    resolve_count_.resize_discard(1);
    constexpr int block = 256;
    const int grid = static_cast<int>((n + block - 1u) / block);
    gpulsmopt_detail::canonical_bake_kernel<<<grid, block, 0, stream>>>(
        base_override_state_.data(), base_override_values_.data(),
        base.values.data(), n, resolve_sel_vdelta_.data(),
        reinterpret_cast<std::uint8_t *>(resolve_flags_.data()));
    CUDA_CHECK(cudaGetLastError());
    std::uint32_t kept = 0;
    std::size_t bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, bytes, base.keys.data(), resolve_flags_.data(),
        resolve_alt_keys_.data(), resolve_count_.data(), static_cast<int>(n),
        stream));
    ensure_sort_temp(bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        sort_temp_storage_.data(), bytes, base.keys.data(),
        resolve_flags_.data(), resolve_alt_keys_.data(), resolve_count_.data(),
        static_cast<int>(n), stream));
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        sort_temp_storage_.data(), bytes, resolve_sel_vdelta_.data(),
        resolve_flags_.data(), base.values.data(), resolve_count_.data(),
        static_cast<int>(n), stream));
    CUDA_CHECK(cudaMemcpyAsync(&kept, resolve_count_.data(), sizeof(kept),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(base.keys.data(), resolve_alt_keys_.data(),
                               kept * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToDevice, stream));
    base.count = kept;
    base.keys.resize_discard(kept);
    base.values.resize_discard(kept);
    build_assignment_offsets(base, kept, stream);
    build_sorted_run_cache(sorted_run_index_, stream);
    reset_canonical_overlay(stream);
  }

  // Folds the base and assignments with last-wins.
  void fold_into_base(cudaStream_t stream) {
    // Materialize the overlay and collapse paged/cold runs first.
    bake_overlay_into_base(stream);
    std::vector<std::size_t> to_merge;
    bool any_paged = false;
    for (std::size_t r = 0; r < runs_.size(); ++r)
      if (runs_[r].assignment) {
        to_merge.push_back(r);
        any_paged = any_paged || runs_[r].paged;
      }
    // Paged counts are upper bounds; merging packs them exactly.
    if (to_merge.size() > 1 || (to_merge.size() == 1 && any_paged))
      merge_run_group(to_merge, gpulsmopt_detail::AssignmentClass::ColdStable,
                      0, stream);
    std::vector<std::size_t> idx;
    std::size_t total = 0;
    if (sorted_run_ready()) {
      idx.push_back(sorted_run_index_);
      total += runs_[sorted_run_index_].count;
    }
    std::vector<std::size_t> updates;
    for (std::size_t r = 0; r < runs_.size(); ++r)
      if (runs_[r].assignment) {
        updates.push_back(r);
        total += runs_[r].count;
      }
    std::sort(updates.begin(), updates.end(),
              [this](std::size_t a, std::size_t b) {
                return runs_[a].sequence < runs_[b].sequence;
              });
    for (const std::size_t r : updates)
      idx.push_back(r);
    if (total == 0)
      return;
    resolve_keys_.resize_discard(total);
    resolve_payload_.resize_discard(total);
    resolve_alt_keys_.resize_discard(total);
    resolve_alt_payload_.resize_discard(total);
    resolve_flags_.resize_discard(total);
    resolve_count_.resize_discard(1);
    constexpr int block = 256;
    std::size_t off = 0;
    for (std::size_t j = 0; j < idx.size(); ++j) {
      RunStorage &run = runs_[idx[j]];
      if (run.count == 0)
        continue;
      // Packs the base as an insert run.
      const bool is_insert =
          !run.assignment ||
          run.operation == gpulsmopt_detail::RunOperation::Insert;
      const int grid = static_cast<int>((run.count + block - 1) / block);
      gpulsmopt_detail::resolve_pack_run_kernel<<<grid, block, 0, stream>>>(
          run.keys.data(), is_insert ? run.values.data() : nullptr,
          run.mixed ? run.op_words.data() : nullptr,
          static_cast<std::uint8_t>(is_insert ? 1u : 0u),
          static_cast<std::uint8_t>(run.mixed ? 1u : 0u), run.count,
          resolve_keys_.data() + off, resolve_payload_.data() + off);
      CUDA_CHECK(cudaGetLastError());
      off += run.count;
    }
    std::size_t sort_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, resolve_keys_.data(), resolve_alt_keys_.data(),
        resolve_payload_.data(), resolve_alt_payload_.data(),
        static_cast<int>(total), 0, 32, stream));
    ensure_sort_temp(sort_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sort_temp_storage_.data(), sort_bytes, resolve_keys_.data(),
        resolve_alt_keys_.data(), resolve_payload_.data(),
        resolve_alt_payload_.data(), static_cast<int>(total), 0, 32, stream));
    const int grid = static_cast<int>((total + block - 1) / block);
    gpulsmopt_detail::resolve_flag_last_kernel<<<grid, block, 0, stream>>>(
        resolve_alt_keys_.data(), total, resolve_flags_.data());
    CUDA_CHECK(cudaGetLastError());
    std::size_t key_sel_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, key_sel_bytes, resolve_alt_keys_.data(),
        resolve_flags_.data(),
        resolve_keys_.data(), resolve_count_.data(), static_cast<int>(total),
        stream));
    std::size_t payload_sel_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, payload_sel_bytes, resolve_alt_payload_.data(),
        resolve_flags_.data(), resolve_payload_.data(), resolve_count_.data(),
        static_cast<int>(total), stream));
    std::size_t sel_bytes =
        std::max(key_sel_bytes, payload_sel_bytes);
    ensure_sort_temp(sel_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        sort_temp_storage_.data(), sel_bytes, resolve_alt_keys_.data(),
        resolve_flags_.data(), resolve_keys_.data(), resolve_count_.data(),
        static_cast<int>(total), stream));
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        sort_temp_storage_.data(), sel_bytes, resolve_alt_payload_.data(),
        resolve_flags_.data(), resolve_payload_.data(), resolve_count_.data(),
        static_cast<int>(total), stream));
    std::uint32_t latest = 0u;
    CUDA_CHECK(cudaMemcpyAsync(&latest, resolve_count_.data(), sizeof(latest),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    resolve_sel_vdelta_.resize_discard(latest);
    resolve_flags_.resize_discard(latest);
    if (latest > 0) {
      const int egrid = static_cast<int>((latest + block - 1) / block);
      gpulsmopt_detail::resolve_base_extract_kernel<<<egrid, block, 0,
                                                      stream>>>(
          resolve_payload_.data(), latest, resolve_sel_vdelta_.data(),
          resolve_flags_.data());
      CUDA_CHECK(cudaGetLastError());
    }
    clear_sorted_state();
    for (auto &run : runs_)
      run_pool_.push_back(std::move(run));
    runs_.clear();
    acquire_run_slot();
    RunStorage &base = runs_.back();
    base.assignment = false;
    base.sequence = 0;
    base.fully_sorted = true;
    base.unit_counts = true;
    base.unique_keys = true;
    base.keys.resize_discard(latest);
    base.values.resize_discard(latest);
    std::uint32_t kept = 0u;
    if (latest > 0) {
      std::size_t b_bytes = 0;
      CUDA_CHECK(cub::DeviceSelect::Flagged(
          nullptr, b_bytes, resolve_keys_.data(), resolve_flags_.data(),
          base.keys.data(), resolve_count_.data(), static_cast<int>(latest),
          stream));
      ensure_sort_temp(b_bytes);
      CUDA_CHECK(cub::DeviceSelect::Flagged(
          sort_temp_storage_.data(), b_bytes, resolve_keys_.data(),
          resolve_flags_.data(), base.keys.data(), resolve_count_.data(),
          static_cast<int>(latest), stream));
      CUDA_CHECK(cub::DeviceSelect::Flagged(
          sort_temp_storage_.data(), b_bytes, resolve_sel_vdelta_.data(),
          resolve_flags_.data(), base.values.data(), resolve_count_.data(),
          static_cast<int>(latest), stream));
      CUDA_CHECK(cudaMemcpyAsync(&kept, resolve_count_.data(), sizeof(kept),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    base.count = kept;
    base.keys.resize_discard(kept);
    base.values.resize_discard(kept);
    build_assignment_offsets(base, kept, stream);
    build_sorted_run_cache(0u, stream);
    run_sequence_ = 0;
    chrono_views_.clear();
    invalidate_resolved();
    succ_sparse_ready_ = false;
    allocate_canonical_overlay(kept, stream);
    reset_cold_arena(stream);
    ++base_generation_;
  }

  // Republishes descriptors newest-last, ordered by sequence_end.
  void rebuild_chrono_views(cudaStream_t stream) {
    std::vector<std::size_t> idx;
    for (std::size_t r = 0; r < runs_.size(); ++r)
      if (runs_[r].assignment)
        idx.push_back(r);
    std::sort(idx.begin(), idx.end(), [this](std::size_t a, std::size_t b) {
      return runs_[a].sequence_end < runs_[b].sequence_end;
    });
    chrono_views_.clear();
    for (const std::size_t r : idx)
      publish_assignment_view(runs_[r], stream);
  }

  std::size_t acquire_cold_run(cudaStream_t stream) {
    int slot = -1;
    for (int i = 0; i < gpulsmopt_detail::kColdArenaSlots;
         ++i) {
      if (!cold_arena_slot_used_[i]) {
        slot = i;
        break;
      }
    }
    if (slot < 0)
      throw std::runtime_error("no free cold arena slot");
    cold_arena_slot_used_[slot] = true;
    CUDA_CHECK(cudaMemsetAsync(cold_arena_tail_.data() + slot, 0,
                               sizeof(std::uint32_t), stream));
    acquire_compaction_slot();
    const std::size_t idx = runs_.size() - 1u;
    RunStorage &cold = runs_[idx];
    cold.assignment = true;
    cold.assignment_class =
        gpulsmopt_detail::AssignmentClass::ColdStable;
    cold.stable_level = 0;
    cold.paged = true;
    cold.cold_arena_slot = slot;
    cold.mixed = true;
    cold.operation = gpulsmopt_detail::RunOperation::Insert;
    cold.unique_keys = true;
    cold.fully_sorted = false;
    cold.unit_counts = false;
    cold.count = 0;
    cold.keys.resize_discard(0);
    cold.values.resize_discard(0);
    cold.op_words.resize_discard(0);
    const std::size_t rows =
        gpulsmopt_detail::kEpochQuotients + 1u;
    cold.quotient_off.resize_discard(rows);
    cold.page_counts.resize_discard(rows);
    cold.page_capacity.resize_discard(rows);
    CUDA_CHECK(cudaMemsetAsync(
        cold.quotient_off.data(), 0,
        rows * sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        cold.page_counts.data(), 0,
        rows * sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        cold.page_capacity.data(), 0,
        rows * sizeof(std::uint32_t), stream));
    return idx;
  }

  // Fold the 64 oldest raw runs into the canonical overlay (sec 10-13).
  void canonical_fold(cudaStream_t stream) {
    std::vector<std::size_t> raw;
    for (std::size_t r = 0; r < runs_.size(); ++r)
      if (runs_[r].assignment &&
          runs_[r].assignment_class == gpulsmopt_detail::AssignmentClass::Raw)
        raw.push_back(r);
    std::sort(raw.begin(), raw.end(), [this](std::size_t a, std::size_t b) {
      return runs_[a].sequence_end < runs_[b].sequence_end;
    });
    const std::size_t group_size =
        std::min<std::size_t>(raw.size(), gpulsmopt_detail::kRawFoldWidth);
    if (group_size == 0)
      return;
    std::vector<std::size_t> group(raw.begin(), raw.begin() + group_size);
    std::uint64_t seq_end = 0, seq_begin = ~std::uint64_t{0};
    std::size_t input_records = 0;
    std::size_t cold_idx = 0;
    {
      GPULSMOPT_FOLD_PHASE(prof_fold_publish_ms_);
      for (std::size_t slot = 0; slot < group_size; ++slot) {
        const std::size_t r = group[slot];
        seq_end = std::max(seq_end, runs_[r].sequence_end);
        seq_begin = std::min(seq_begin, runs_[r].sequence_begin);
        input_records += runs_[r].count;
        host_state_->views[slot] = make_assignment_view(runs_[r]);
      }
      fold_source_views_.resize_discard(group_size);
      CUDA_CHECK(cudaMemcpyAsync(
          fold_source_views_.data(), host_state_->views,
          group_size * sizeof(gpulsmopt_detail::AssignmentRunView),
          cudaMemcpyHostToDevice, stream));

      cold_idx = acquire_cold_run(stream);
      // Overlay becomes active; readers consult canonical state.
      canonical_overlay_active_ = true;
      CUDA_CHECK(cudaMemsetAsync(fold_fallback_count_.data(), 0,
                                 sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(fold_fallback_head_.data(), 0,
                                 sizeof(std::uint32_t), stream));
    }

    RunStorage &cold = runs_[cold_idx];
    const std::size_t arena_slot =
        static_cast<std::size_t>(cold.cold_arena_slot);
    std::uint32_t *arena_keys =
        cold_arena_keys_.data() +
        arena_slot * cold_arena_slot_capacity_;
    std::uint32_t *arena_values =
        cold_arena_values_.data() +
        arena_slot * cold_arena_slot_capacity_;
    std::uint32_t *arena_ops =
        cold_arena_ops_.data() +
        arena_slot * cold_arena_slot_words_;
    std::uint32_t *arena_tail =
        cold_arena_tail_.data() + arena_slot;
    {
      GPULSMOPT_FOLD_PHASE(prof_fold_fast_ms_);
      gpulsmopt_detail::canonical_fold_rank23_kernel<<<
          gpulsmopt_detail::kEpochQuotients,
          gpulsmopt_detail::kFoldThreads, 0, stream>>>(
          fold_source_views_.data(), static_cast<int>(group_size),
          make_sorted_view(), base_override_state_.data(),
          base_override_values_.data(), rank23_value_delta_.data(),
          rank23_count_delta_.data(), cold.quotient_off.data(),
          cold.page_counts.data(), cold.page_capacity.data(),
          arena_tail, cold_arena_slot_capacity_, arena_keys,
          arena_values, arena_ops, fold_fallback_quotients_.data(),
          fold_fallback_count_.data(), fold_stats_.data());
      CUDA_CHECK(cudaGetLastError());
    }

    {
      GPULSMOPT_FOLD_PHASE(prof_fold_fallback_ms_);
      // Skip the fallback kernel entirely when nothing overflowed
      // (the common case): a one-word D2H beats a flat empty scan.
      std::uint32_t fallback_n = 0;
      CUDA_CHECK(cudaMemcpyAsync(&fallback_n, fold_fallback_count_.data(),
                                 sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      if (fallback_n > 0)
        run_fold_fallback(group_size, cold, stream);
    }

    {
      GPULSMOPT_FOLD_PHASE(prof_fold_book_ms_);
      cold.count = input_records;
      cold.sequence = seq_end;
      cold.sequence_begin = seq_begin;
      cold.sequence_end = seq_end;
      cold.stable_level = 0;

      // Retire the folded raw runs (cold run stays in runs_).
      RunStorage cold_moved = std::move(runs_[cold_idx]);
      runs_.erase(runs_.begin() + static_cast<std::ptrdiff_t>(cold_idx));
      retire_run_group(group, stream);
      runs_.push_back(std::move(cold_moved));

      ++canonical_generation_;
      canonical_through_sequence_ = seq_end;
      rank23_value_prefix_ready_ = false;
      rank23_count_prefix_ready_ = false;
      invalidate_resolved();
      succ_sparse_ready_ = false;
      ++maintenance_stats_.canonical_fold_count;
      maintenance_stats_.canonical_input_records += input_records;
      // The fold is the compaction event the harness observes.
      ++maintenance_stats_.compaction_count;
      maintenance_stats_.compacted_input_records += input_records;
    }

    {
      GPULSMOPT_FOLD_PHASE(prof_fold_carry_ms_);
      carry_cold_run(runs_.size() - 1u, stream);
      rebuild_chrono_views(stream);
    }
#ifdef GPULSMOPT_PROFILE_FOLD
    maintenance_stats_.canonical_fold_time =
        prof_fold_publish_ms_ + prof_fold_fast_ms_ +
        prof_fold_fallback_ms_ + prof_fold_book_ms_ +
        prof_fold_carry_ms_;
    printf("[fold] publish=%.3f fast=%.3f fallback=%.3f "
           "book=%.3f carry=%.3f ms\n",
           prof_fold_publish_ms_, prof_fold_fast_ms_,
           prof_fold_fallback_ms_, prof_fold_book_ms_,
           prof_fold_carry_ms_);
#endif
  }

  void run_fold_fallback(std::size_t group_size,
                         RunStorage &cold,
                         cudaStream_t stream) {
    const std::size_t slot =
        static_cast<std::size_t>(cold.cold_arena_slot);
    gpulsmopt_detail::canonical_fold_fallback_kernel<<<
        gpulsmopt_detail::kFoldFallbackBlocks,
        gpulsmopt_detail::kFoldFallbackThreads, 0, stream>>>(
        fold_source_views_.data(), static_cast<int>(group_size),
        fold_fallback_quotients_.data(),
        fold_fallback_count_.data(), fold_fallback_head_.data(),
        make_sorted_view(), base_override_state_.data(),
        base_override_values_.data(), rank23_value_delta_.data(),
        rank23_count_delta_.data(), cold.quotient_off.data(),
        cold.page_counts.data(), cold.page_capacity.data(),
        cold_arena_tail_.data() + slot,
        cold_arena_slot_capacity_,
        cold_arena_keys_.data() +
            slot * cold_arena_slot_capacity_,
        cold_arena_values_.data() +
            slot * cold_arena_slot_capacity_,
        cold_arena_ops_.data() +
            slot * cold_arena_slot_words_,
        fold_stats_.data());
    CUDA_CHECK(cudaGetLastError());
  }

  // Four-way cold stable tier (sec 18). Full levels are absorbed
  // into ONE merge that lands at the first non-full level; no
  // intermediate carry run is materialized.
  void carry_cold_run(std::size_t cold_idx, cudaStream_t stream) {
    (void)cold_idx;
    std::vector<std::size_t> fused;
    int out_level = 0;
    for (int level = 0; level < gpulsmopt_detail::kStableLevels; ++level) {
      std::vector<std::size_t> at_level;
      for (std::size_t r = 0; r < runs_.size(); ++r)
        if (runs_[r].assignment &&
            runs_[r].assignment_class ==
                gpulsmopt_detail::AssignmentClass::ColdStable &&
            runs_[r].stable_level == level)
          at_level.push_back(r);
      if (fused.empty()) {
        // A level triggers only once it holds a fourth run.
        if (at_level.size() <
            static_cast<std::size_t>(gpulsmopt_detail::kStableFanout))
          return;
        fused = at_level;
      } else if (at_level.size() >= static_cast<std::size_t>(
                                        gpulsmopt_detail::kStableFanout - 1)) {
        // The carried unit would fill this level too; absorb it.
        fused.insert(fused.end(), at_level.begin(), at_level.end());
      } else {
        out_level = level;
        break;
      }
      out_level =
          std::min(level + 1, gpulsmopt_detail::kStableLevels - 1);
    }
    if (fused.empty())
      return;
    // Merge sources oldest to newest for last-wins resolution.
    std::sort(fused.begin(), fused.end(),
              [this](std::size_t a, std::size_t b) {
                return runs_[a].sequence_end < runs_[b].sequence_end;
              });
    merge_run_group(fused, gpulsmopt_detail::AssignmentClass::ColdStable,
                    out_level, stream);
  }

  // Merge a set of assignment runs into one packed mixed run with
  // last-wins; retire the sources; return the merged run index.
  std::size_t merge_run_group(std::vector<std::size_t> group,
                              gpulsmopt_detail::AssignmentClass out_class,
                              int out_level, cudaStream_t stream) {
    const std::size_t group_size = group.size();
    std::uint64_t group_seq = 0;
    std::uint64_t seq_begin = ~std::uint64_t{0};
    std::size_t total = 0;
    for (const std::size_t r : group) {
      group_seq = std::max(group_seq, runs_[r].sequence_end);
      seq_begin = std::min(seq_begin, runs_[r].sequence_begin);
      total += runs_[r].count; // exact for packed and paged cold runs
    }
    if (total == 0) {
      // Nothing to merge; drop the empty sources.
      retire_run_group(group, stream);
      rebuild_chrono_views(stream);
      invalidate_resolved();
      return runs_.size();
    }
    if (total > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("temporal compaction exceeds CUB limits");

    normalize_views_.resize_discard(group_size);
    for (std::size_t slot = 0; slot < group_size; ++slot)
      host_state_->views[slot] = make_assignment_view(runs_[group[slot]]);
    CUDA_CHECK(cudaMemcpyAsync(
        normalize_views_.data(), host_state_->views,
        group_size * sizeof(gpulsmopt_detail::AssignmentRunView),
        cudaMemcpyHostToDevice, stream));

    constexpr int block = 256;
    constexpr int quotient_rows = gpulsmopt_detail::kEpochQuotients + 1;
    constexpr int count_grid = (quotient_rows + block - 1) / block;
    gpulsmopt_detail::assignment_group_count_kernel<<<
        count_grid, block, 0, stream>>>(
        normalize_views_.data(), static_cast<int>(group_size),
        compaction_counts_.data());
    CUDA_CHECK(cudaGetLastError());
    exclusive_scan_u32(
        compaction_counts_.data(), compaction_offsets_.data(),
        gpulsmopt_detail::kEpochQuotients + 1u, stream);
    CUDA_CHECK(cudaMemcpyAsync(
        host_compaction_offsets_.data(), compaction_offsets_.data(),
        (gpulsmopt_detail::kEpochQuotients + 1u) * sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // Paged sources carry upper-bound counts; actual may be less.
    const std::size_t actual = host_compaction_offsets_.back();
    if (actual > total)
      throw std::runtime_error("temporal compaction count mismatch");
    if (actual == 0) {
      retire_run_group(group, stream);
      rebuild_chrono_views(stream);
      invalidate_resolved();
      return runs_.size();
    }

    using QuotientTile = std::pair<std::uint32_t, std::uint32_t>;
    std::vector<QuotientTile> tiles;
    std::uint32_t first = 0u;
    while (first < gpulsmopt_detail::kEpochQuotients) {
      std::uint32_t last = first;
      while (last < gpulsmopt_detail::kEpochQuotients) {
        const std::size_t candidate =
            host_compaction_offsets_[last + 1u] -
            host_compaction_offsets_[first];
        if (candidate > gpulsmopt_detail::kCompactionTileRecords &&
            last > first)
          break;
        ++last;
        if (candidate >= gpulsmopt_detail::kCompactionTileRecords)
          break;
      }
      tiles.emplace_back(first, last);
      first = last;
    }

    auto stage_tile = [&](const QuotientTile &tile) -> std::size_t {
      const std::uint32_t tile_first = tile.first;
      const std::uint32_t segments = tile.second - tile.first;
      const std::size_t tile_count =
          host_compaction_offsets_[tile.second] -
          host_compaction_offsets_[tile.first];
      if (tile_count == 0u)
        return 0u;
      resolve_keys_.resize_discard(tile_count);
      resolve_payload_.resize_discard(tile_count);
      resolve_alt_keys_.resize_discard(tile_count);
      resolve_alt_payload_.resize_discard(tile_count);
      const int offset_grid = static_cast<int>((segments + 1u + block - 1u) /
                                               block);
      gpulsmopt_detail::compaction_tile_offsets_kernel<<<
          offset_grid, block, 0, stream>>>(
          compaction_offsets_.data(), tile_first, segments,
          compaction_tile_offsets_.data());
      CUDA_CHECK(cudaGetLastError());
      gpulsmopt_detail::assignment_group_gather_range_kernel<<<
          segments, block, 0, stream>>>(
          normalize_views_.data(), static_cast<int>(group_size), tile_first,
          compaction_tile_offsets_.data(), resolve_keys_.data(),
          resolve_payload_.data());
      CUDA_CHECK(cudaGetLastError());
      if (tile_count > compaction_sort_count_ ||
          segments > compaction_sort_segments_) {
        std::size_t required = 0u;
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, required, resolve_keys_.data(),
            resolve_alt_keys_.data(), resolve_payload_.data(),
            resolve_alt_payload_.data(), static_cast<int>(tile_count),
            static_cast<int>(segments), compaction_tile_offsets_.data(),
            compaction_tile_offsets_.data() + 1u, 0, 16, stream));
        compaction_sort_count_ =
            std::max(compaction_sort_count_, tile_count);
        compaction_sort_segments_ =
            std::max(compaction_sort_segments_,
                     static_cast<std::size_t>(segments));
        compaction_sort_temp_bytes_ =
            std::max(compaction_sort_temp_bytes_, required);
      }
      std::size_t sort_bytes = compaction_sort_temp_bytes_;
      ensure_sort_temp(sort_bytes);
      CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
          sort_temp_storage_.data(), sort_bytes, resolve_keys_.data(),
          resolve_alt_keys_.data(), resolve_payload_.data(),
          resolve_alt_payload_.data(), static_cast<int>(tile_count),
          static_cast<int>(segments), compaction_tile_offsets_.data(),
          compaction_tile_offsets_.data() + 1u, 0, 16, stream));
      return tile_count;
    };

    CUDA_CHECK(cudaMemsetAsync(
        compaction_unique_counts_.data(), 0,
        (gpulsmopt_detail::kEpochQuotients + 1u) * sizeof(std::uint32_t),
        stream));
    for (const QuotientTile &tile : tiles) {
      const std::size_t tile_count = stage_tile(tile);
      if (tile_count == 0u)
        continue;
      const int grid =
          static_cast<int>((tile_count + block - 1u) / block);
      gpulsmopt_detail::compaction_unique_count_kernel<<<
          grid, block, 0, stream>>>(
          resolve_alt_keys_.data(), tile_count,
          compaction_unique_counts_.data());
      CUDA_CHECK(cudaGetLastError());
    }
    exclusive_scan_u32(
        compaction_unique_counts_.data(),
        compaction_unique_offsets_.data(),
        gpulsmopt_detail::kEpochQuotients + 1u, stream);
    std::uint32_t compact_count = 0u;
    CUDA_CHECK(cudaMemcpyAsync(
        &compact_count,
        compaction_unique_offsets_.data() +
            gpulsmopt_detail::kEpochQuotients,
        sizeof(compact_count), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    acquire_compaction_slot();
    RunStorage &merged = runs_.back();
    merged.assignment = true;
    merged.assignment_class = out_class;
    merged.stable_level = out_level;
    merged.paged = false;
    merged.mixed = true;
    merged.operation = gpulsmopt_detail::RunOperation::Insert;
    merged.sequence = group_seq;
    merged.sequence_begin = seq_begin;
    merged.sequence_end = group_seq;
    merged.count = compact_count;
    merged.unique_keys = true;
    merged.fully_sorted = false;
    merged.unit_counts = false;
    merged.keys.resize_discard(compact_count);
    merged.values.resize_discard(compact_count);
    merged.quotient_off.resize_discard(
        gpulsmopt_detail::kEpochQuotients + 1u);
    merged.op_words.resize_discard(
        (compact_count + 31u) / 32u + 1u);
    CUDA_CHECK(cudaMemsetAsync(
        merged.op_words.data(), 0,
        ((compact_count + 31u) / 32u + 1u) *
            sizeof(std::uint32_t),
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        merged.quotient_off.data(), compaction_unique_offsets_.data(),
        (gpulsmopt_detail::kEpochQuotients + 1u) * sizeof(std::uint32_t),
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        compaction_unique_cursors_.data(),
        compaction_unique_offsets_.data(),
        gpulsmopt_detail::kEpochQuotients * sizeof(std::uint32_t),
        cudaMemcpyDeviceToDevice, stream));
    for (const QuotientTile &tile : tiles) {
      const std::size_t tile_count = stage_tile(tile);
      if (tile_count == 0u)
        continue;
      const int grid =
          static_cast<int>((tile_count + block - 1u) / block);
      gpulsmopt_detail::compaction_unique_scatter_kernel<<<
          grid, block, 0, stream>>>(
          resolve_alt_keys_.data(), resolve_alt_payload_.data(), tile_count,
          compaction_unique_cursors_.data(), merged.keys.data(),
          merged.values.data(), merged.op_words.data());
      CUDA_CHECK(cudaGetLastError());
    }

    RunStorage compacted = std::move(runs_.back());
    runs_.pop_back();
    retire_run_group(group, stream);
    runs_.push_back(std::move(compacted));
    rebuild_chrono_views(stream);
    invalidate_resolved();
    // The successor watermark handles the merge.
    ++maintenance_stats_.cold_tier_compaction_count;
    maintenance_stats_.cold_tier_input_records += actual;
    maintenance_stats_.cold_tier_output_records += compact_count;
    return runs_.size() - 1u;
  }

  // Move a set of source runs back to the pool, fixing indices.
  // Paged sources release their cold arena slot for reuse.
  void retire_run_group(std::vector<std::size_t> group, cudaStream_t stream) {
    (void)stream;
    std::sort(group.begin(), group.end());
    for (auto it = group.rbegin(); it != group.rend(); ++it) {
      RunStorage &run = runs_[*it];
      if (run.paged && run.cold_arena_slot >= 0 &&
          run.cold_arena_slot < gpulsmopt_detail::kColdArenaSlots)
        cold_arena_slot_used_[run.cold_arena_slot] = false;
      run.paged = false;
      run.cold_arena_slot = -1;
      if (sorted_run_ready() && sorted_run_index_ > *it)
        --sorted_run_index_;
      run_pool_.push_back(std::move(runs_[*it]));
      runs_.erase(runs_.begin() + static_cast<std::ptrdiff_t>(*it));
    }
  }


  std::size_t max_elements_ = 0;
  std::size_t batch_capacity_ = 0;
  std::size_t live_count_ = 0;
  std::size_t sorted_run_index_ = std::numeric_limits<std::size_t>::max();
  mutable std::shared_mutex snapshot_mutex_;
  MaintenanceStats maintenance_stats_{};

  // Temporal assignment-run state.
  std::uint64_t run_sequence_ = 0;
  RunStorage resolved_;
  bool resolved_ready_ = false;
  // Incremental cache and merge scratch.
  std::uint64_t resolved_through_sequence_ = 0;
  std::uint64_t resolved_base_generation_ = ~std::uint64_t{0};
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::AssignmentRunView>
      normalize_views_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> norm_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> norm_pay_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> cache_pay_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> merge_out_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> merge_out_pay_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> merge_flags_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> merge_sel_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> merge_sel_pay_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> resolved_value_prefix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> resolved_count_prefix_;
  std::size_t resolved_value_scan_count_ = 0;
  std::size_t resolved_value_scan_temp_bytes_ = 0;
  std::size_t resolved_count_scan_count_ = 0;
  std::size_t resolved_count_scan_temp_bytes_ = 0;
  std::size_t resolved_sort_count_ = 0;
  std::size_t resolved_sort_temp_bytes_ = 0;
  bool resolved_value_prefix_ready_ = false;
  bool resolved_count_prefix_ready_ = false;
  // Lazy successor sidecar.
  bool succ_sparse_ready_ = false;
  std::uint64_t succ_sparse_base_generation_ = ~std::uint64_t{0};
  std::uint64_t succ_sparse_run_sequence_ = 0;
  std::uint32_t succ_sparse_l0_words_ = 0;
  std::uint32_t succ_sparse_l3_words_ = 0;
  std::uint32_t succ_sparse_positive_l0_words_ = 0;
  std::uint32_t succ_sparse_positive_l3_words_ = 0;
  std::uint64_t base_generation_ = 0;
  // Canonical overlay: per-BaseRun-position state + Rank23 deltas.
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> base_override_state_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> base_override_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> rank23_value_delta_;
  gpulsmopt_detail::RawDeviceBuffer<std::int32_t> rank23_count_delta_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> rank23_value_prefix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> rank23_count_prefix_;
  bool rank23_value_prefix_ready_ = false;
  bool rank23_count_prefix_ready_ = false;
  bool canonical_overlay_active_ = false;
  std::uint32_t cold_arena_slot_capacity_ = 0;
  std::size_t cold_arena_slot_words_ = 0;
  bool cold_arena_slot_used_[gpulsmopt_detail::kColdArenaSlots] = {};
  std::uint64_t canonical_generation_ = 0;
  std::uint64_t canonical_through_sequence_ = 0;
  std::uint64_t resolved_canonical_generation_ = ~std::uint64_t{0};
  std::size_t rank23_value_scan_bytes_ = 0;
  std::size_t rank23_count_scan_bytes_ = 0;
  // Fold scratch (sec 27).
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::AssignmentRunView>
      fold_source_views_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> fold_fallback_quotients_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> fold_stats_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> fold_fallback_count_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> fold_fallback_head_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> cold_arena_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> cold_arena_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> cold_arena_ops_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> cold_arena_tail_;
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::AssignmentRunView>
      assignment_views_;
  // Oldest-to-newest descriptor mirror.
  std::vector<gpulsmopt_detail::AssignmentRunView> chrono_views_;
  std::size_t delete_sort_count_ = 0;
  std::size_t delete_sort_temp_bytes_ = 0;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> narrow_overflow_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> resolve_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> resolve_payload_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> resolve_alt_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> resolve_alt_payload_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> resolve_flags_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> resolve_sel_vdelta_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> resolve_count_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compaction_counts_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compaction_offsets_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compaction_tile_offsets_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compaction_unique_counts_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compaction_unique_offsets_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compaction_unique_cursors_;
  std::vector<std::uint32_t> host_compaction_offsets_;
  std::size_t compaction_sort_count_ = 0;
  std::size_t compaction_sort_segments_ = 0;
  std::size_t compaction_sort_temp_bytes_ = 0;

#ifdef GPULSMOPT_PROFILE_INSERT
  double prof_delta_sort_ms_ = 0.0;
  double prof_delta_ingest_ms_ = 0.0;
  void reset_insert_prof_() {
    prof_delta_sort_ms_ = prof_delta_ingest_ms_ = 0.0;
  }
#endif
#ifdef GPULSMOPT_PROFILE_FOLD
  double prof_fold_publish_ms_ = 0.0;
  double prof_fold_fast_ms_ = 0.0;
  double prof_fold_fallback_ms_ = 0.0;
  double prof_fold_book_ms_ = 0.0;
  double prof_fold_carry_ms_ = 0.0;
#endif

  std::vector<RunStorage> runs_;
  std::vector<RunStorage> run_pool_;

  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> direct_sort_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> direct_sort_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> sort_temp_storage_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> sorted_value_prefix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> sorted_count_prefix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> sorted_range_cdf_;
  std::uint32_t sorted_range_min_key_ = 0u;
  std::uint64_t sorted_range_span_ = 0u;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> base_rank23_;
  gpulsmopt_detail::PinnedHostState *host_state_ = nullptr;
  cudaEvent_t stream_handoff_ = nullptr;
  cudaStream_t operation_stream_ = nullptr;
  bool operation_stream_valid_ = false;
  std::size_t direct_sort_count_ = 0;
  std::size_t direct_sort_temp_bytes_ = 0;
  std::size_t run_sort_count_ = 0;
  std::size_t run_sort_temp_bytes_ = 0;
  std::size_t scan_u32_count_ = 0;
  std::size_t scan_u32_temp_bytes_ = 0;
  std::size_t metadata_scan_temp_bytes_ = 0;
  bool sorted_range_cdf_ready_ = false;

  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> succ_miss_indices_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> succ_miss_count_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> succ_deleted_base_words_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> succ_live_word_l1_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> succ_live_word_l2_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> succ_live_word_l3_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> succ_positive_words_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> succ_positive_l1_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> succ_positive_l2_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> succ_positive_l3_;
};
