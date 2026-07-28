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
#define GPULSMOPT_PREWARM_LEAVES 576
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
// Flat BaseRun rank directory.
constexpr int kBaseRank23Bits = 23;
constexpr int kBaseRank23Shift = 32 - kBaseRank23Bits;
constexpr std::size_t kBaseRank23Size = std::size_t{1} << kBaseRank23Bits;
constexpr std::size_t kSortedRunMinRecords = 1u << 22;
// Rank23 canonical fold geometry.
constexpr int kRawFoldWidth = 64;
constexpr int kStableFanout = 4;
constexpr int kColdArenaSlots = 8;
constexpr int kParentSlots = kRunCapacity;
constexpr int kHashFoldMaxRecords = 1280;
constexpr int kHashRouteBits = 6;
constexpr int kHashRouteBins = 1 << kHashRouteBits;
constexpr int kHashRouteWordsPerBin = 2;
constexpr std::size_t kHashRouteWordsPerParent =
    static_cast<std::size_t>(kEpochQuotients) *
    kHashRouteBins * kHashRouteWordsPerBin;
constexpr int kLookupLeafCapacity =
    kColdArenaSlots * kRawFoldWidth + kRawFoldWidth;
constexpr std::uint16_t kNoLookupParent = 0xffffu;
constexpr std::uint32_t kHashHeavyFlag = 1u << 31;
constexpr std::uint32_t kHashScanFlag = 1u << 30;
constexpr std::uint32_t kHashCountMask =
    ~(kHashHeavyFlag | kHashScanFlag);
constexpr std::size_t kWarmLeafCount =
    GPULSMOPT_PREWARM_LEAVES;
constexpr int kStableLevels = 16;
constexpr int kRank23BinsPerQuotient = 1 << (kBaseRank23Bits - kEpochQuotientBits);
constexpr std::uint32_t kRank23LocalBinMask = 0x7f;
static_assert(kRawFoldWidth == 64, "raw fold width must be 64");
static_assert(kWarmLeafCount >= kRawFoldWidth,
              "prewarm must hold one raw fold");
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
  const std::uint32_t *page_counts;
  const std::uint32_t *op_words;
  const std::uint64_t *quotient_mask;
  const std::uint32_t *hash_table;
  const AssignmentRunView *hash_children;
  const std::uint32_t *child_router;
  const AssignmentRunView *group_children;
  std::uint16_t hash_slots;
  std::uint16_t child_count;
  std::uint8_t constant_op;
  std::uint8_t mixed;
  std::uint8_t paged;
  std::uint8_t hashed;
  std::uint8_t grouped;
};

__device__ inline void assignment_bounds(const AssignmentRunView &run,
                                         std::uint32_t q,
                                         std::uint32_t *begin,
                                         std::uint32_t *end) {
  const std::uint32_t b = run.offsets[q];
  *begin = b;
  *end = run.paged ? b + run.page_counts[q] : run.offsets[q + 1u];
}

__device__ inline std::uint32_t hash_fold_slot(std::uint32_t low,
                                              std::uint32_t slots) {
  return __umulhi(low * 0x9e3779b1u, slots);
}

__device__ inline std::uint64_t raw_quotient_mask_bits(
    std::uint32_t low) {
  const std::uint32_t first =
      (low * 0x9e3779b1u) >> 27;
  std::uint32_t mixed = low ^ (low >> 7);
  mixed *= 0x85ebca6bu;
  const std::uint32_t second = mixed >> 27;
  return (std::uint64_t{1} << first) |
         (std::uint64_t{1} << (32u + second));
}

__device__ inline std::uint32_t
assignment_record_count(const AssignmentRunView &run, std::uint32_t q) {
  if (run.grouped) {
    std::uint32_t total = 0u;
    for (std::uint32_t child = 0; child < run.child_count; ++child)
      total += assignment_record_count(run.group_children[child], q);
    return total;
  }
  if (run.hashed)
    return run.page_counts[q] & kHashCountMask;
  std::uint32_t begin = 0u, end = 0u;
  assignment_bounds(run, q, &begin, &end);
  return end - begin;
}

__device__ inline bool assignment_find(const AssignmentRunView &run,
                                       std::uint32_t key,
                                       std::uint32_t *value,
                                       bool *live) {
  if (run.grouped) {
    for (int child = static_cast<int>(run.child_count) - 1;
         child >= 0; --child) {
      if (assignment_find(run.group_children[child], key, value, live))
        return true;
    }
    return false;
  }
  const std::uint32_t q = key >> kEpochQuotientBits;
  if (run.hashed) {
    const std::uint32_t meta = run.page_counts[q];
    const std::uint32_t count = meta & kHashCountMask;
    if (count == 0u)
      return false;
    if (meta & kHashScanFlag) {
      if (run.child_router != nullptr) {
        const std::uint32_t route =
            (key >> (kEpochQuotientBits - kHashRouteBits)) &
            (kHashRouteBins - 1u);
        const std::size_t base =
            (static_cast<std::size_t>(q) * kHashRouteBins + route) *
            kHashRouteWordsPerBin;
        for (int word = kHashRouteWordsPerBin - 1;
             word >= 0; --word) {
          std::uint32_t candidates = run.child_router[base + word];
          while (candidates != 0u) {
            const int bit = 31 - __clz(candidates);
            const int child = word * 32 + bit;
            if (assignment_find(
                    run.hash_children[child], key, value, live))
              return true;
            candidates &= ~(1u << bit);
          }
        }
        return false;
      }
      for (int child = static_cast<int>(run.child_count) - 1;
           child >= 0; --child) {
        if (assignment_find(run.hash_children[child], key, value, live))
          return true;
      }
      return false;
    }
    const bool heavy = (meta & kHashHeavyFlag) != 0u;
    const std::uint32_t low = key & 0xffffu;
    const std::uint32_t slots = heavy
        ? run.offsets[q + 1u] - run.offsets[q]
        : static_cast<std::uint32_t>(run.hash_slots);
    if (slots == 0u)
      return false;
    const std::uint32_t *table = heavy
        ? run.keys + run.offsets[q]
        : run.hash_table + static_cast<std::size_t>(q) * run.hash_slots;
    std::uint32_t slot = slots == 65536u
                             ? low
                             : hash_fold_slot(low, slots);
    for (std::uint32_t probe = 0; probe < slots; ++probe) {
      const std::uint32_t entry = table[slot];
      if (entry == 0u)
        return false;
      if ((entry & 0xffffu) == low) {
        const std::uint32_t child = (entry >> 16) & 0x7fffu;
        return assignment_find(
            run.hash_children[child], key, value, live);
      }
      if (++slot == slots)
        slot = 0u;
    }
    return false;
  }
  if (run.quotient_mask != nullptr) {
    const std::uint64_t required =
        raw_quotient_mask_bits(key & 0xffffu);
    if ((run.quotient_mask[q] & required) != required)
      return false;
  }
  std::uint32_t begin = 0u, position = 0u;
  assignment_bounds(run, q, &begin, &position);
  while (position-- > begin) {
    if (run.keys[position] != key)
      continue;
    const bool is_live = assignment_op_at(
                             run.op_words, run.constant_op,
                             run.mixed, position) != 0;
    *live = is_live;
    *value = is_live ? run.values[position] : 0u;
    return true;
  }
  return false;
}

__device__ inline void assignment_decode_hash(
    const AssignmentRunView &run, std::uint32_t q,
    std::uint32_t entry, std::uint32_t *key,
    std::uint32_t *value, int *op) {
  const std::uint32_t child = (entry >> 16) & 0x7fffu;
  *key = (q << kEpochQuotientBits) | (entry & 0xffffu);
  bool live = false;
  if (!assignment_find(run.hash_children[child], *key, value, &live)) {
    *value = 0u;
    *op = 0;
    return;
  }
  *op = live ? 1 : 0;
}

__device__ inline void assignment_gather_quotient(
    const AssignmentRunView &run, std::uint32_t q,
    std::uint32_t *cursor, std::uint32_t *shared_cursor,
    std::uint32_t *out_keys, std::uint64_t *out_payload) {
  if (run.grouped) {
    for (std::uint32_t child = 0; child < run.child_count; ++child)
      assignment_gather_quotient(
          run.group_children[child], q, cursor, shared_cursor,
          out_keys, out_payload);
    return;
  }
  if (run.hashed) {
    const std::uint32_t meta = run.page_counts[q];
    const std::uint32_t count = meta & kHashCountMask;
    if (count == 0u)
      return;
    if (meta & kHashScanFlag) {
      for (std::uint32_t child = 0; child < run.child_count; ++child)
        assignment_gather_quotient(
            run.hash_children[child], q, cursor, shared_cursor,
            out_keys, out_payload);
      return;
    }
    const bool heavy = (meta & kHashHeavyFlag) != 0u;
    const std::uint32_t slots = heavy
        ? run.offsets[q + 1u] - run.offsets[q]
        : static_cast<std::uint32_t>(run.hash_slots);
    const std::uint32_t *table = heavy
        ? run.keys + run.offsets[q]
        : run.hash_table + static_cast<std::size_t>(q) * run.hash_slots;
    if (threadIdx.x == 0)
      *shared_cursor = *cursor;
    __syncthreads();
    for (std::uint32_t slot = threadIdx.x; slot < slots;
         slot += blockDim.x) {
      const std::uint32_t entry = table[slot];
      if (entry == 0u)
        continue;
      const std::uint32_t output = atomicAdd(shared_cursor, 1u);
      std::uint32_t key = 0u, value = 0u;
      int op = 0;
      assignment_decode_hash(run, q, entry, &key, &value, &op);
      out_keys[output] = key;
      out_payload[output] =
          (static_cast<std::uint64_t>(value) << 32) |
          static_cast<std::uint64_t>(op != 0);
    }
    __syncthreads();
    *cursor += count;
    return;
  }
  std::uint32_t begin = 0u, end = 0u;
  assignment_bounds(run, q, &begin, &end);
  const std::uint32_t count = end - begin;
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    const std::uint32_t position = begin + i;
    const int op = assignment_op_at(
        run.op_words, run.constant_op, run.mixed, position);
    const std::uint32_t value = op ? run.values[position] : 0u;
    out_keys[*cursor + i] = run.keys[position];
    out_payload[*cursor + i] =
        (static_cast<std::uint64_t>(value) << 32) |
        static_cast<std::uint64_t>(op != 0);
  }
  *cursor += count;
}

struct LookupLeafView {
  AssignmentRunView leaf;
  std::uint32_t rank;
  std::uint16_t parent;
  std::uint8_t parent_leader;
};

struct LookupPublication {
  LookupLeafView leaves[kLookupLeafCapacity];
  AssignmentRunView parents[kColdArenaSlots];
  std::uint32_t parent_ranks[kColdArenaSlots];
};

struct PinnedHostState {
  AssignmentRunView views[kRunCapacity];
  AssignmentRunView scratch_views[kRunCapacity];
  LookupPublication lookup;
  std::uint32_t narrow_overflow;
  std::uint32_t resolved_count;
  std::uint32_t successor_miss_count;
  std::uint32_t gathered_count;
  std::uint32_t fold_fallback_count;
  std::uint32_t fold_cold_count;
  std::uint32_t fold_matched_count;
  std::uint32_t hash_heavy_count;
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
  for (int r = 0; r < run_count; ++r)
    total += assignment_record_count(runs[r], q);
  counts[q] = total;
}

__global__ void assignment_group_gather_kernel(
    const AssignmentRunView *runs, int run_count,
    const std::uint32_t *offsets, std::uint32_t *out_keys,
    std::uint64_t *out_payload) {
  __shared__ std::uint32_t hash_cursor;
  const std::uint32_t q = blockIdx.x;
  if (q >= kEpochQuotients)
    return;
  std::uint32_t cursor = offsets[q];
  for (int r = 0; r < run_count; ++r)
    assignment_gather_quotient(
        runs[r], q, &cursor, &hash_cursor,
        out_keys, out_payload);
}

__global__ void assignment_group_gather_range_kernel(
    const AssignmentRunView *runs, int run_count,
    std::uint32_t first_quotient, const std::uint32_t *tile_offsets,
    std::uint32_t *out_keys,
    std::uint64_t *out_payload) {
  __shared__ std::uint32_t hash_cursor;
  const std::uint32_t local_q = blockIdx.x;
  const std::uint32_t q = first_quotient + local_q;
  std::uint32_t cursor = tile_offsets[local_q];
  for (int r = 0; r < run_count; ++r)
    assignment_gather_quotient(
        runs[r], q, &cursor, &hash_cursor,
        out_keys, out_payload);
}

__global__ void compaction_stage_unique_kernel(
    const std::uint32_t *keys, const std::uint64_t *payload,
    std::size_t count, std::size_t output_begin,
    std::uint32_t *staged_keys, std::uint64_t *staged_payload,
    std::uint32_t *keep_flags) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::uint32_t key = keys[i];
  const bool keep = i + 1u == count || keys[i + 1u] != key;
  const std::size_t output = output_begin + i;
  keep_flags[output] = keep ? 1u : 0u;
  if (!keep)
    return;
  staged_keys[output] = key;
  staged_payload[output] = payload[i];
}

__global__ void compaction_tile_offsets_kernel(
    const std::uint32_t *global_offsets, std::uint32_t first,
    std::uint32_t segments, std::uint32_t *tile_offsets) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > segments)
    return;
  tile_offsets[i] = global_offsets[first + i] - global_offsets[first];
}

__global__ void compaction_output_offsets_kernel(
    const std::uint32_t *input_offsets,
    const std::uint32_t *positions,
    const std::uint32_t *keep_flags, std::uint32_t count,
    std::uint32_t *output_offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kEpochQuotients)
    return;
  const std::uint32_t compact_count =
      positions[count - 1u] + keep_flags[count - 1u];
  const std::uint32_t input = input_offsets[q];
  output_offsets[q] = input < count ? positions[input] : compact_count;
}

__global__ void compaction_unique_scatter_kernel(
    const std::uint32_t *keys, const std::uint64_t *payload,
    const std::uint32_t *keep_flags,
    const std::uint32_t *positions, std::size_t count,
    std::uint32_t *out_keys, std::uint32_t *out_values,
    std::uint8_t *out_ops) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count || keep_flags[i] == 0u)
    return;
  const std::uint32_t out = positions[i];
  const std::uint64_t packed = payload[i];
  out_keys[out] = keys[i];
  out_values[out] = static_cast<std::uint32_t>(packed >> 32);
  out_ops[out] = static_cast<std::uint8_t>(packed & 1u);
}

__global__ void compaction_pack_ops_kernel(
    const std::uint8_t *ops, std::size_t count,
    std::uint32_t *out_op_words) {
  const std::size_t word =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t begin = word << 5;
  if (begin >= count)
    return;
  std::uint32_t packed = 0u;
  const std::size_t end = begin + 32u < count ? begin + 32u : count;
  for (std::size_t i = begin; i < end; ++i)
    packed |= static_cast<std::uint32_t>(ops[i] != 0u) << (i - begin);
  out_op_words[word] = packed;
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
  for (int r = run_count - 1; r >= 0; --r) {
    const AssignmentRunView run = runs[r];
    std::uint32_t value = 0u;
    bool live = false;
    if (!assignment_find(run, key, &value, &live))
      continue;
    out_values[i] = live ? value : kEmptyKey;
    if (out_found)
      out_found[i] = live ? 1u : 0u;
    return;
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
    int live = -1;
    for (int r = run_count - 1; r >= 0 && live < 0; --r) {
      const AssignmentRunView run = runs[r];
      std::uint32_t value = 0u;
      bool found_live = false;
      if (assignment_find(run, key, &value, &found_live))
        live = found_live ? 1 : 0;
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

// Packs the visible canonical BaseRun.
__global__ void resolve_pack_canonical_base_kernel(
    CanonicalBaseView base, std::uint32_t *out_keys,
    std::uint64_t *out_payload) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= base.base.count)
    return;
  const bool live = canonical_live_at(base, i);
  const std::uint32_t value = live ? canonical_value_at(base, i) : 0u;
  out_keys[i] = base.base.keys[i];
  out_payload[i] =
      (static_cast<std::uint64_t>(value) << 32) | (live ? 1u : 0u);
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

// Dense quotient offsets in one pass over the SORTED keys:
// offsets[q] = lower_bound(keys, q << 16) = count of records whose
// quotient is < q. Empty quotients collapse for free; no histogram
// atomics, no boundary marking, no reverse-min repair, no scan.
__global__ void quotient_lower_bound_kernel(const std::uint32_t *sorted_keys,
                                            std::uint32_t count,
                                            std::uint32_t *offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kEpochQuotients)
    return;
  if (q == kEpochQuotients) {
    offsets[q] = count;
    return;
  }
  const std::uint32_t target = q << kEpochQuotientBits;
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + ((hi - lo) >> 1);
    if (sorted_keys[mid] < target)
      lo = mid + 1u;
    else
      hi = mid;
  }
  offsets[q] = lo;
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

// Keeps only the newest live record for each key.
__global__ void resolve_live_last_kernel(
    const std::uint32_t *keys, const std::uint64_t *payload,
    std::size_t n, std::uint32_t *out_values,
    std::uint8_t *flags) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const bool last = i + 1u == n || keys[i] != keys[i + 1u];
  out_values[i] = static_cast<std::uint32_t>(payload[i] >> 32);
  flags[i] = last && (payload[i] & 1u) != 0u ? 1u : 0u;
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

__device__ __forceinline__ bool lookup_raw_leaf(
    const AssignmentRunView &run, std::uint32_t key,
    std::uint32_t *value, bool *live) {
  const std::uint32_t q = key >> kEpochQuotientBits;
  if (run.quotient_mask != nullptr) {
    const std::uint64_t required =
        raw_quotient_mask_bits(key & 0xffffu);
    if ((run.quotient_mask[q] & required) != required)
      return false;
  }
  std::uint32_t begin = 0u, position = 0u;
  assignment_bounds(run, q, &begin, &position);
  while (position-- > begin) {
    if (run.keys[position] != key)
      continue;
    const bool is_live = assignment_op_at(
                             run.op_words, run.constant_op,
                             run.mixed, position) != 0;
    *live = is_live;
    *value = is_live ? run.values[position] : 0u;
    return true;
  }
  return false;
}

__device__ __forceinline__ bool lookup_heavy_parent(
    const AssignmentRunView &parent, std::uint32_t key,
    std::uint32_t *value, bool *live,
    std::uint32_t *found_child) {
  const std::uint32_t q = key >> kEpochQuotientBits;
  const std::uint32_t slots =
      parent.offsets[q + 1u] - parent.offsets[q];
  const std::uint32_t low = key & 0xffffu;
  const std::uint32_t *table =
      parent.keys + parent.offsets[q];
  std::uint32_t slot =
      slots == 65536u ? low : hash_fold_slot(low, slots);
  for (std::uint32_t probe = 0; probe < slots; ++probe) {
    const std::uint32_t entry = table[slot];
    if (entry == 0u)
      return false;
    if ((entry & 0xffffu) == low) {
      const std::uint32_t child =
          (entry >> 16) & 0x7fffu;
      *found_child = child;
      return lookup_raw_leaf(
          parent.hash_children[child], key, value, live);
    }
    if (++slot == slots)
      slot = 0u;
  }
  return false;
}

__device__ __forceinline__ bool lookup_routed_parent(
    const AssignmentRunView &parent, std::uint32_t key,
    std::uint32_t *value, bool *live,
    std::uint32_t *found_child) {
  const std::uint32_t q = key >> kEpochQuotientBits;
  if (parent.child_router == nullptr) {
    for (int child = static_cast<int>(parent.child_count) - 1;
         child >= 0; --child) {
      if (lookup_raw_leaf(
              parent.hash_children[child], key, value, live)) {
        *found_child = static_cast<std::uint32_t>(child);
        return true;
      }
    }
    return false;
  }
  const std::uint32_t route =
      (key >> (kEpochQuotientBits - kHashRouteBits)) &
      (kHashRouteBins - 1u);
  const std::size_t base =
      (static_cast<std::size_t>(q) * kHashRouteBins + route) *
      kHashRouteWordsPerBin;
  for (int word = kHashRouteWordsPerBin - 1;
       word >= 0; --word) {
    std::uint32_t candidates = parent.child_router[base + word];
    while (candidates != 0u) {
      const int bit = 31 - __clz(candidates);
      const std::uint32_t child =
          static_cast<std::uint32_t>(word * 32 + bit);
      if (lookup_raw_leaf(
              parent.hash_children[child], key, value, live)) {
        *found_child = child;
        return true;
      }
      candidates &= ~(1u << bit);
    }
  }
  return false;
}

template <bool RouteParent>
__device__ __forceinline__ unsigned long long
lookup_leaf_candidate(
    const LookupLeafView &item,
    const LookupPublication *publication,
    std::uint32_t key) {
  std::uint32_t rank = item.rank;
  std::uint32_t value = 0u;
  bool live = false;
  bool found = false;
  if (item.parent != kNoLookupParent) {
    const AssignmentRunView parent =
        publication->parents[item.parent];
    const std::uint32_t q = key >> kEpochQuotientBits;
    const std::uint32_t meta = parent.page_counts[q];
    const bool indexed =
        (meta & kHashHeavyFlag) != 0u &&
        (meta & kHashScanFlag) == 0u;
    if (indexed || RouteParent) {
      if (!item.parent_leader)
        return 0u;
      std::uint32_t child = 0u;
      if (indexed) {
        found = lookup_heavy_parent(
            parent, key, &value, &live, &child);
      } else {
        found = lookup_routed_parent(
            parent, key, &value, &live, &child);
      }
      rank = publication->parent_ranks[item.parent] -
             (parent.child_count - 1u - child);
    } else {
      found = lookup_raw_leaf(
          item.leaf, key, &value, &live);
    }
  } else {
    found = lookup_raw_leaf(
        item.leaf, key, &value, &live);
  }
  if (!found)
    return 0u;
  return (static_cast<unsigned long long>(rank) << 33) |
         (static_cast<unsigned long long>(live) << 32) |
         (live ? value : kEmptyKey);
}

__device__ inline void write_lookup_candidate(
    unsigned long long best, CanonicalBaseView base,
    std::uint32_t key, std::size_t query,
    std::uint32_t *out_values, std::uint8_t *out_found) {
  if (best != 0u) {
    const bool live = ((best >> 32) & 1u) != 0u;
    out_values[query] =
        live ? static_cast<std::uint32_t>(best) : kEmptyKey;
    if (out_found)
      out_found[query] = live ? 1u : 0u;
    return;
  }
  std::uint32_t value = kEmptyKey;
  const bool found = canonical_find_value(base, key, &value);
  if (!found)
    value = kEmptyKey;
  out_values[query] = value;
  if (out_found)
    out_found[query] = found ? 1u : 0u;
}

// One block evaluates all chronological leaves.
__global__ void temporal_lookup_leaf_block_kernel(
    const LookupPublication *publication, int leaf_count,
    CanonicalBaseView base, const std::uint32_t *queries,
    std::size_t n, std::uint32_t *out_values,
    std::uint8_t *out_found) {
  const std::size_t query = blockIdx.x;
  if (query >= n)
    return;
  __shared__ unsigned long long candidates[256];
  const int tid = threadIdx.x;
  const std::uint32_t key = queries[query];
  unsigned long long best = 0u;
  for (int leaf = tid; leaf < leaf_count;
       leaf += blockDim.x) {
    const unsigned long long candidate =
        lookup_leaf_candidate<false>(
            publication->leaves[leaf], publication, key);
    if (candidate > best)
      best = candidate;
  }
  candidates[tid] = best;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const unsigned long long other = candidates[tid + stride];
      if (other > candidates[tid])
        candidates[tid] = other;
    }
    __syncthreads();
  }
  if (tid == 0)
    write_lookup_candidate(
        candidates[0], base, key, query,
        out_values, out_found);
}

// Large batches map one thread to each query.
__global__ void temporal_lookup_leaf_thread_kernel(
    const LookupPublication *publication, int leaf_count,
    CanonicalBaseView base, const std::uint32_t *queries,
    std::size_t n, std::uint32_t *out_values,
    std::uint8_t *out_found) {
  const std::size_t query =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (query >= n)
    return;
  const std::uint32_t key = queries[query];
  unsigned long long best = 0u;
  for (int leaf = leaf_count - 1; leaf >= 0; --leaf) {
    const LookupLeafView item = publication->leaves[leaf];
    best = lookup_leaf_candidate<true>(
        item, publication, key);
    if (best != 0u)
      break;
    if (item.parent != kNoLookupParent && item.parent_leader)
      leaf -= publication->parents[item.parent].child_count - 1u;
  }
  write_lookup_candidate(
      best, base, key, query, out_values, out_found);
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
    if (run.hashed || run.grouped) {
      candidates += kNarrowSeenCap + 1u;
      continue;
    }
    for (std::uint32_t qq = qlo; qq <= qhi; ++qq)
      candidates += assignment_record_count(run, qq);
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

__global__ void raw_quotient_mask_kernel(
    const std::uint32_t *keys,
    const std::uint32_t *offsets,
    std::uint64_t *masks) {
  const std::uint32_t warp = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t q =
      blockIdx.x * (blockDim.x >> 5) + warp;
  if (q >= kEpochQuotients)
    return;
  const std::uint32_t begin = offsets[q];
  const std::uint32_t end = offsets[q + 1u];
  std::uint32_t low_mask = 0u;
  std::uint32_t high_mask = 0u;
  for (std::uint32_t position = begin + lane;
       position < end; position += 32u) {
    const std::uint64_t bits =
        raw_quotient_mask_bits(keys[position] & 0xffffu);
    low_mask |= static_cast<std::uint32_t>(bits);
    high_mask |= static_cast<std::uint32_t>(bits >> 32);
  }
  for (int delta = 16; delta != 0; delta >>= 1) {
    low_mask |= __shfl_down_sync(
        0xffffffffu, low_mask, delta);
    high_mask |= __shfl_down_sync(
        0xffffffffu, high_mask, delta);
  }
  if (lane == 0u)
    masks[q] = (static_cast<std::uint64_t>(high_mask) << 32) |
               low_mask;
}

// Plans pointer scans and indexes only heavy quotients.
__global__ void temporal_hash_plan_kernel(
    const AssignmentRunView *runs, int run_count,
    std::uint32_t *capacities,
    std::uint32_t *quotient_counts,
    std::uint32_t *heavy_quotients,
    std::uint32_t *heavy_count) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kEpochQuotients)
    return;
  if (q == kEpochQuotients) {
    capacities[q] = 0u;
    quotient_counts[q] = 0u;
    return;
  }

  std::uint32_t total = 0u;
  for (int child = 0; child < run_count; ++child)
    total += runs[child].offsets[q + 1u] - runs[child].offsets[q];

  std::uint32_t capacity = 0u;
  if (total > kHashFoldMaxRecords) {
    if (total >= 32768u) {
      capacity = 65536u;
    } else {
      const std::uint32_t target = total + (total >> 2) + 1u;
      capacity = 2048u;
      while (capacity < target)
        capacity <<= 1;
    }
  }
  capacities[q] = capacity;
  quotient_counts[q] =
      capacity == 0u && total != 0u ? kHashScanFlag | total : 0u;

  const unsigned mask =
      __ballot_sync(0xffffffffu, capacity != 0u);
  if (capacity == 0u)
    return;
  const unsigned lane = threadIdx.x & 31u;
  const unsigned leader = __ffs(mask) - 1u;
  std::uint32_t base = 0u;
  if (lane == leader)
    base = atomicAdd(heavy_count, __popc(mask));
  base = __shfl_sync(mask, base, leader);
  const unsigned lower =
      lane == 0u ? 0u : mask & ((1u << lane) - 1u);
  heavy_quotients[base + __popc(lower)] = q;
}

// Builds indexes only for heavy quotients.
__global__ __launch_bounds__(256)
void temporal_hash_heavy_kernel(
    const AssignmentRunView *runs, int run_count,
    const std::uint32_t *heavy_quotients,
    const std::uint32_t *heavy_offsets,
    std::uint32_t *quotient_counts,
    std::uint32_t heavy_capacity,
    std::uint32_t *heavy_table) {
  __shared__ AssignmentRunView views[kRawFoldWidth];
  __shared__ std::uint32_t prefix[kRawFoldWidth + 1];
  __shared__ std::uint32_t occupied_count;
  const std::uint32_t q = heavy_quotients[blockIdx.x];
  const int tid = threadIdx.x;
  if (tid < run_count)
    views[tid] = runs[tid];
  if (tid == 0) {
    occupied_count = 0u;
    prefix[0] = 0u;
  }
  __syncthreads();

  if (tid == 0) {
    for (int child = 0; child < run_count; ++child) {
      const std::uint32_t begin = views[child].offsets[q];
      const std::uint32_t count =
          views[child].offsets[q + 1u] - begin;
      prefix[child + 1] = prefix[child] + count;
    }
  }
  __syncthreads();

  const std::uint32_t total = prefix[run_count];
  const std::uint32_t table_begin = heavy_offsets[q];
  const std::uint32_t table_end = heavy_offsets[q + 1u];
  const std::uint32_t slots = table_end - table_begin;
  if (table_end > heavy_capacity) {
    if (tid == 0)
      quotient_counts[q] = kHashHeavyFlag | kHashScanFlag | total;
    return;
  }

  std::uint32_t *table = heavy_table + table_begin;
  for (std::uint32_t slot = tid; slot < slots;
       slot += blockDim.x)
    table[slot] = 0u;
  __syncthreads();

  if (slots == 65536u) {
    for (int child = 0; child < run_count; ++child) {
      const std::uint32_t begin = views[child].offsets[q];
      const std::uint32_t end = views[child].offsets[q + 1u];
      const std::uint32_t tag =
          0x80000000u |
          (static_cast<std::uint32_t>(child) << 16);
      for (std::uint32_t position = begin + tid;
           position < end; position += blockDim.x) {
        const std::uint32_t low =
            views[child].keys[position] & 0xffffu;
        table[low] = tag | low;
      }
      __syncthreads();
    }
  } else {
    for (std::uint32_t ordinal = tid; ordinal < total;
         ordinal += blockDim.x) {
      int child = 0;
      while (ordinal >= prefix[child + 1])
        ++child;
      const std::uint32_t position =
          views[child].offsets[q] + ordinal - prefix[child];
      const std::uint32_t low =
          views[child].keys[position] & 0xffffu;
      const std::uint32_t entry =
          0x80000000u |
          (static_cast<std::uint32_t>(child) << 16) | low;
      std::uint32_t slot = hash_fold_slot(low, slots);
      while (true) {
        const std::uint32_t old =
            atomicCAS(table + slot, 0u, entry);
        if (old == 0u)
          break;
        if ((old & 0xffffu) == low) {
          atomicMax(table + slot, entry);
          break;
        }
        if (++slot == slots)
          slot = 0u;
      }
    }
  }
  __syncthreads();

  std::uint32_t occupied = 0u;
  for (std::uint32_t slot = tid; slot < slots;
       slot += blockDim.x)
    occupied += table[slot] != 0u ? 1u : 0u;
  occupied = __reduce_add_sync(0xffffffffu, occupied);
  if ((tid & 31) == 0)
    atomicAdd(&occupied_count, occupied);
  __syncthreads();
  if (tid == 0)
    quotient_counts[q] = kHashHeavyFlag | occupied_count;
}

// Routes one key-prefix bin to candidate children.
__global__ __launch_bounds__(128)
void temporal_hash_route_kernel(
    const AssignmentRunView *runs, int run_count,
    const std::uint32_t *quotient_counts,
    std::uint32_t *child_router) {
  constexpr int words =
      kHashRouteBins * kHashRouteWordsPerBin;
  __shared__ std::uint32_t route_masks[words];
  const std::uint32_t q = blockIdx.x;
  const int tid = threadIdx.x;
  if (tid < words)
    route_masks[tid] = 0u;
  __syncthreads();

  const std::uint32_t meta = quotient_counts[q];
  if ((meta & kHashScanFlag) != 0u) {
    for (int child = tid; child < run_count;
         child += blockDim.x) {
      const std::uint32_t begin = runs[child].offsets[q];
      const std::uint32_t end = runs[child].offsets[q + 1u];
      for (std::uint32_t position = begin; position < end;
           ++position) {
        const std::uint32_t route =
            (runs[child].keys[position] >>
             (kEpochQuotientBits - kHashRouteBits)) &
            (kHashRouteBins - 1u);
        const int word = child >> 5;
        atomicOr(route_masks +
                     route * kHashRouteWordsPerBin + word,
                 1u << (child & 31));
      }
    }
  }
  __syncthreads();

  const std::size_t base =
      static_cast<std::size_t>(q) * words;
  if (tid < words)
    child_router[base + tid] = route_masks[tid];
}

// --- Rank23 canonical fold (sec 13) ---
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
static_assert(kWindowCap * (3 * sizeof(std::uint32_t) + 1 +
                            sizeof(unsigned long long)) +
                  8 * 1024 <=
              48 * 1024,
              "fold shared memory exceeds the 48 KiB static limit");

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

// One block per quotient; fold 64 raw runs into canonical state
// plus a cold page of unmatched keys. Overflow -> fallback list.
__global__ void canonical_fold_rank23_kernel(
    const AssignmentRunView *runs, int run_count, SortedRunView base,
    std::uint8_t *state, std::uint32_t *override_values,
    std::uint32_t *rank23_value_delta, std::int32_t *rank23_count_delta,
    std::uint32_t *cold_page_begin, std::uint32_t *cold_page_count,
    std::uint32_t *cold_arena_tail, std::uint32_t arena_capacity,
    std::uint32_t *cold_keys, std::uint32_t *cold_values,
    std::uint32_t *cold_ops, std::uint32_t *fallback_quotients,
    std::uint32_t *fallback_count, std::uint32_t *cold_output_count,
    std::uint32_t *matched_output_count, std::uint32_t *fold_stats) {
  const std::uint32_t q = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ std::uint32_t s_runoff[kRawFoldWidth + 1];
  __shared__ std::uint32_t s_runbegin[kRawFoldWidth];
  __shared__ AssignmentRunView s_views[kRawFoldWidth];
  __shared__ std::uint32_t s_base_b[128];
  __shared__ std::uint32_t s_base_e[128];
  __shared__ std::uint32_t s_win_key[kWindowCap];
  __shared__ std::uint32_t s_win_val[kWindowCap];
  __shared__ std::uint32_t s_win_ovr[kWindowCap];
  __shared__ std::uint8_t s_win_state[kWindowCap];
  __shared__ unsigned long long s_winner[kWindowCap];
  __shared__ std::uint32_t s_rank_dv[128];
  __shared__ std::int32_t s_rank_dc[128];
  __shared__ std::uint32_t s_warp_count[16];
  __shared__ std::uint32_t s_warp_offset[16];
  __shared__ std::uint32_t s_wb;
  __shared__ std::uint32_t s_page;
  __shared__ std::uint32_t s_emit_cursor;
  __shared__ std::uint32_t s_chunk_base;
  __shared__ int s_wlen;
  __shared__ int s_total;
  __shared__ int s_overflow;
  __shared__ int s_matched;
  __shared__ int s_coldn;

  if (tid == 0) {
    s_overflow = 0;
    s_matched = 0;
    s_coldn = 0;
  }
  for (int r = tid; r < run_count; r += blockDim.x) {
    s_views[r] = runs[r];
    std::uint32_t b, e;
    assignment_bounds(s_views[r], q, &b, &e);
    s_runbegin[r] = b;
    s_runoff[r] = e - b;
  }
  if (tid < 128) {
    if (base.count == 0u) {
      s_base_b[tid] = 0u;
      s_base_e[tid] = 0u;
    } else {
      const std::uint32_t gbin =
          (q << (kBaseRank23Bits - kEpochQuotientBits)) |
          static_cast<std::uint32_t>(tid);
      s_base_b[tid] = base.rank23[gbin];
      s_base_e[tid] = base.rank23[gbin + 1u];
    }
  }
  __syncthreads();
  if (tid == 0) {
    std::uint32_t count = 0;
    for (int r = 0; r < run_count; ++r) {
      const std::uint32_t records = s_runoff[r];
      s_runoff[r] = count;
      count += records;
    }
    s_runoff[run_count] = count;
    s_total = static_cast<int>(count);
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
    }
    return;
  }
  if (s_overflow) {
    if (tid == 0) {
      const std::uint32_t slot = atomicAdd(fallback_count, 1u);
      fallback_quotients[slot] = q;
      cold_page_begin[q] = 0;
      cold_page_count[q] = 0;
      atomicAdd(fold_stats + kFoldStatFallback, 1u);
    }
    return;
  }

  for (int i = tid; i < s_wlen; i += blockDim.x) {
    const std::uint32_t p = s_wb + static_cast<std::uint32_t>(i);
    s_win_key[i] = base.keys[p];
    s_winner[i] = 0ull;
  }
  for (int b = tid; b < 128; b += blockDim.x) {
    s_rank_dv[b] = 0u;
    s_rank_dc[b] = 0;
  }
  __syncthreads();

  for (int chunk = 0; chunk < s_total; chunk += blockDim.x) {
    const int i = chunk + tid;
    bool miss = false;
    if (i < s_total) {
      int lo = 0;
      int hi = run_count - 1;
      while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (s_runoff[mid] <= static_cast<std::uint32_t>(i))
          lo = mid;
        else
          hi = mid - 1;
      }
      const AssignmentRunView run = s_views[lo];
      const std::uint32_t p =
          s_runbegin[lo] +
          (static_cast<std::uint32_t>(i) - s_runoff[lo]);
      const std::uint32_t key = run.keys[p];
      const int lbin = static_cast<int>((key >> 9) & 0x7fu);
      const std::uint32_t rb = s_base_b[lbin] - s_wb;
      const std::uint32_t bn = s_base_e[lbin] - s_base_b[lbin];
      const std::uint32_t off = static_cast<std::uint32_t>(
          lower_bound_u32(s_win_key + rb, bn, key));
      if (off < bn && s_win_key[rb + off] == key) {
        const unsigned long long priority =
            (static_cast<unsigned long long>(lo + 1) << 32) | p;
        atomicMax(&s_winner[rb + off], priority);
      } else {
        miss = true;
      }
    }
    const unsigned mask = __ballot_sync(0xffffffffu, miss);
    if (lane == 0)
      s_warp_count[warp] = static_cast<std::uint32_t>(__popc(mask));
    __syncthreads();
    if (tid == 0) {
      for (int w = 0; w < 16; ++w)
        s_coldn += static_cast<int>(s_warp_count[w]);
    }
    __syncthreads();
  }

  int local_matched = 0;
  for (int j = tid; j < s_wlen; j += blockDim.x)
    local_matched += s_winner[j] != 0ull ? 1 : 0;
  if (local_matched)
    atomicAdd(&s_matched, local_matched);
  __syncthreads();

  const bool dense_apply = s_wlen > 0 && s_matched * 2 >= s_wlen;
  if (dense_apply) {
    for (int i = tid; i < s_wlen; i += blockDim.x) {
      const std::uint32_t p = s_wb + static_cast<std::uint32_t>(i);
      s_win_val[i] = base.values[p];
      s_win_ovr[i] = override_values[p];
      s_win_state[i] = state[p];
    }
    __syncthreads();
  }

  for (int j = tid; j < s_wlen; j += blockDim.x) {
    const unsigned long long winner = s_winner[j];
    if (winner == 0ull)
      continue;
    const int run_index = static_cast<int>(winner >> 32) - 1;
    const std::uint32_t p = static_cast<std::uint32_t>(winner);
    const AssignmentRunView run = s_views[run_index];
    const int op =
        assignment_op_at(run.op_words, run.constant_op, run.mixed, p);
    const std::uint32_t value =
        op != 0 && run.values ? run.values[p] : 0u;
    const std::uint32_t global_p = s_wb + static_cast<std::uint32_t>(j);
    std::uint32_t dv = 0u;
    std::int32_t dc = 0;
    if (dense_apply) {
      canonical_apply_one(s_win_val[j], s_win_state, s_win_ovr,
                          static_cast<std::size_t>(j), op, value, &dv, &dc);
    } else {
      canonical_apply_one(base.values[global_p], state, override_values,
                          global_p, op, value, &dv, &dc);
    }
    const int lbin = static_cast<int>((s_win_key[j] >> 9) & 0x7fu);
    if (dv != 0u)
      atomicAdd(&s_rank_dv[lbin], dv);
    if (dc != 0)
      atomicAdd(&s_rank_dc[lbin], dc);
  }
  __syncthreads();

  if (dense_apply) {
    for (int i = tid; i < s_wlen; i += blockDim.x) {
      const std::uint32_t p = s_wb + static_cast<std::uint32_t>(i);
      state[p] = s_win_state[i];
      override_values[p] = s_win_ovr[i];
    }
  }

  const std::uint32_t rank_base =
      q << (kBaseRank23Bits - kEpochQuotientBits);
  for (int b = tid; b < 128; b += blockDim.x) {
    if (s_rank_dv[b] != 0u)
      rank23_value_delta[rank_base + b] += s_rank_dv[b];
    if (s_rank_dc[b] != 0)
      rank23_count_delta[rank_base + b] += s_rank_dc[b];
  }

  if (s_coldn == 0) {
    if (tid == 0) {
      cold_page_begin[q] = 0u;
      cold_page_count[q] = 0u;
    }
  } else {
    if (tid == 0) {
      const std::uint32_t cap =
          (static_cast<std::uint32_t>(s_coldn) + 31u) & ~31u;
      const std::uint32_t page = atomicAdd(cold_arena_tail, cap);
      if (page > arena_capacity || cap > arena_capacity - page)
        asm("trap;");
      s_page = page;
      s_emit_cursor = 0u;
      cold_page_begin[q] = page;
      cold_page_count[q] = static_cast<std::uint32_t>(s_coldn);
      atomicAdd(cold_output_count, static_cast<std::uint32_t>(s_coldn));
      atomicAdd(fold_stats + kFoldStatUnmatched,
                static_cast<std::uint32_t>(s_coldn));
    }
    __syncthreads();
    const std::uint32_t words =
        (static_cast<std::uint32_t>(s_coldn) + 31u) >> 5;
    for (std::uint32_t word = tid; word < words; word += blockDim.x)
      cold_ops[(s_page >> 5) + word] = 0u;
    __syncthreads();

    const bool all_miss = s_coldn == s_total;
    for (int chunk = 0; chunk < s_total; chunk += blockDim.x) {
      const int i = chunk + tid;
      bool miss = false;
      std::uint32_t key = 0u;
      std::uint32_t value = 0u;
      int op = 0;
      if (i < s_total) {
        int lo = 0;
        int hi = run_count - 1;
        while (lo < hi) {
          const int mid = (lo + hi + 1) >> 1;
          if (s_runoff[mid] <= static_cast<std::uint32_t>(i))
            lo = mid;
          else
            hi = mid - 1;
        }
        const AssignmentRunView run = s_views[lo];
        const std::uint32_t p =
            s_runbegin[lo] +
            (static_cast<std::uint32_t>(i) - s_runoff[lo]);
        key = run.keys[p];
        if (all_miss) {
          miss = true;
        } else {
          const int lbin = static_cast<int>((key >> 9) & 0x7fu);
          const std::uint32_t rb = s_base_b[lbin] - s_wb;
          const std::uint32_t bn = s_base_e[lbin] - s_base_b[lbin];
          const std::uint32_t off = static_cast<std::uint32_t>(
              lower_bound_u32(s_win_key + rb, bn, key));
          miss = off >= bn || s_win_key[rb + off] != key;
        }
        if (miss) {
          op = assignment_op_at(run.op_words, run.constant_op,
                                run.mixed, p);
          value = op != 0 && run.values ? run.values[p] : 0u;
        }
      }
      const unsigned mask = __ballot_sync(0xffffffffu, miss);
      if (lane == 0)
        s_warp_count[warp] = static_cast<std::uint32_t>(__popc(mask));
      __syncthreads();
      if (tid == 0) {
        std::uint32_t count = 0u;
        for (int w = 0; w < 16; ++w) {
          s_warp_offset[w] = count;
          count += s_warp_count[w];
        }
        s_chunk_base = s_emit_cursor;
        s_emit_cursor += count;
      }
      __syncthreads();
      if (miss) {
        const unsigned before =
            lane == 0 ? 0u : mask & ((1u << lane) - 1u);
        const std::uint32_t rank =
            static_cast<std::uint32_t>(__popc(before));
        const std::uint32_t out =
            s_page + s_chunk_base + s_warp_offset[warp] + rank;
        cold_keys[out] = key;
        cold_values[out] = value;
        if (op != 0)
          atomicOr(cold_ops + (out >> 5), 1u << (out & 31u));
      }
      __syncthreads();
    }
  }

  if (tid == 0 && s_matched != 0) {
    atomicAdd(matched_output_count,
              static_cast<std::uint32_t>(s_matched));
    atomicAdd(fold_stats + kFoldStatMatched,
              static_cast<std::uint32_t>(s_matched));
  }
}

constexpr int kFoldFallbackBlocks = 256;
constexpr int kFoldFallbackThreads = 64;

__global__ void canonical_fold_fallback_kernel(
    const AssignmentRunView *runs, int run_count,
    const std::uint32_t *quotient_list,
    const std::uint32_t *quotient_count,
    std::uint32_t *work_head,
    std::uint32_t *cold_page_begin,
    std::uint32_t *cold_page_count,
    std::uint32_t *cold_arena_tail,
    std::uint32_t arena_capacity,
    std::uint32_t *cold_keys,
    std::uint32_t *cold_values,
    std::uint32_t *cold_ops,
    std::uint32_t *cold_output_count,
    std::uint32_t *fold_stats) {
  const int tid = threadIdx.x;
  __shared__ std::uint32_t s_begin[kRawFoldWidth];
  __shared__ std::uint32_t s_count[kRawFoldWidth];
  __shared__ std::uint32_t s_offset[kRawFoldWidth + 1];
  __shared__ std::uint32_t s_work;
  __shared__ std::uint32_t s_q;
  __shared__ std::uint32_t s_page;
  __shared__ std::uint32_t s_total;

  if (*quotient_count == 0u)
    return;

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
    if (tid < run_count) {
      std::uint32_t begin = 0u;
      std::uint32_t end = 0u;
      assignment_bounds(runs[tid], q, &begin, &end);
      s_begin[tid] = begin;
      s_count[tid] = end - begin;
    } else {
      s_begin[tid] = 0u;
      s_count[tid] = 0u;
    }
    __syncthreads();
    if (tid == 0) {
      std::uint32_t total = 0u;
      for (int r = 0; r < run_count; ++r) {
        s_offset[r] = total;
        total += s_count[r];
      }
      s_offset[run_count] = total;
      s_total = total;
      const std::uint32_t cap = (total + 31u) & ~31u;
      const std::uint32_t page = atomicAdd(cold_arena_tail, cap);
      if (page > arena_capacity || cap > arena_capacity - page)
        asm("trap;");
      s_page = page;
      cold_page_begin[q] = page;
      cold_page_count[q] = total;
      atomicAdd(cold_output_count, total);
      atomicAdd(fold_stats + kFoldStatUnmatched, total);
    }
    __syncthreads();

    const std::uint32_t words = (s_total + 31u) >> 5;
    for (std::uint32_t word = tid; word < words;
         word += blockDim.x)
      cold_ops[(s_page >> 5) + word] = 0u;
    __syncthreads();

    for (int r = 0; r < run_count; ++r) {
      const AssignmentRunView run = runs[r];
      const std::uint32_t begin = s_begin[r];
      const std::uint32_t count = s_count[r];
      const std::uint32_t out_begin = s_page + s_offset[r];
      for (std::uint32_t i = tid; i < count; i += blockDim.x) {
        const std::uint32_t p = begin + i;
        const std::uint32_t out = out_begin + i;
        const int op = assignment_op_at(
            run.op_words, run.constant_op, run.mixed, p);
        cold_keys[out] = run.keys[p];
        cold_values[out] = op != 0 && run.values ? run.values[p] : 0u;
        if (op != 0)
          atomicOr(cold_ops + (out >> 5), 1u << (out & 31u));
      }
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
    run_pool_.reserve(gpulsmopt_detail::kWarmLeafCount +
                      gpulsmopt_detail::kRunCapacity);
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
                     batch.count, stream);
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
    if (!batch.sorted)
      throw std::invalid_argument(
          "GPULSMOpt deletion requires sorted keys");
    {
      std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
      order_stream_locked(stream);
#ifdef GPULSMOPT_PROFILE_INSERT
      reset_insert_prof_();
      const auto prof_t0 = std::chrono::high_resolution_clock::now();
#endif
      insert_records(batch.keys, batch.keys,
                     static_cast<std::uint8_t>(gpulsmopt_detail::kTombstone),
                     batch.count, stream);
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
#ifdef GPULSMOPT_PROFILE_FOLD
    double prof_masks = 0.0;
    double prof_publish = 0.0;
    double prof_kernel = 0.0;
#endif
    {
      GPULSMOPT_FOLD_PHASE(prof_masks);
      ensure_raw_lookup_masks(stream);
    }
    int leaf_count = 0;
    {
      GPULSMOPT_FOLD_PHASE(prof_publish);
      leaf_count = prepare_lookup_leaf_views(stream);
      if (leaf_count <
          GPULSMOPT_LOOKUP_RUN_PARALLEL_MIN_RUNS)
        flush_pending_views(stream);
    }
    const int run_count = static_cast<int>(chrono_views_.size());
    const bool leaf_parallel =
        leaf_count >= GPULSMOPT_LOOKUP_RUN_PARALLEL_MIN_RUNS;
    {
      GPULSMOPT_FOLD_PHASE(prof_kernel);
      if (leaf_parallel &&
          batch.count <=
              static_cast<std::size_t>(
                  GPULSMOPT_LOOKUP_RUN_PARALLEL_MAX_QUERIES)) {
        constexpr int block = 256;
        const int grid = static_cast<int>(batch.count);
        gpulsmopt_detail::temporal_lookup_leaf_block_kernel<<<
            grid, block, 0, stream>>>(
            lookup_publication_.data(), leaf_count,
            make_canonical_base_view(), batch.queries,
            batch.count, batch.out_values, batch.out_found);
      } else if (leaf_parallel) {
        constexpr int block = 256;
        const int grid = static_cast<int>(
            (batch.count + block - 1u) / block);
        gpulsmopt_detail::temporal_lookup_leaf_thread_kernel<<<
            grid, block, 0, stream>>>(
            lookup_publication_.data(), leaf_count,
            make_canonical_base_view(), batch.queries,
            batch.count, batch.out_values, batch.out_found);
      } else {
        constexpr int block = 256;
        const int grid = static_cast<int>(
            (batch.count + block - 1u) / block);
        gpulsmopt_detail::temporal_lookup_kernel<<<
            grid, block, 0, stream>>>(
            assignment_views_.data(), run_count,
            make_canonical_base_view(), batch.queries,
            batch.count, batch.out_values, batch.out_found);
      }
      CUDA_CHECK(cudaGetLastError());
    }
#ifdef GPULSMOPT_PROFILE_FOLD
    printf("[lookup] leaves=%d masks=%.3f publish=%.3f "
           "kernel=%.3f ms\n",
           leaf_count, prof_masks, prof_publish, prof_kernel);
#endif
  }

  void successor(const DeviceSuccessorBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    order_stream_locked(stream);
    ensure_sorted_run_cache(stream);
    flush_pending_views(stream);
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
    flush_pending_views(stream);
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
    run.sequence = 0;
    run.sequence_begin = 0;
    run.sequence_end = 0;
    run.stable_level = -1;
    run.operation = gpulsmopt_detail::RunOperation::Insert;
    run.assignment_class = gpulsmopt_detail::AssignmentClass::Raw;
    run.mixed = false;
    run.assignment = false;
    run.paged = false;
    run.hashed = false;
    run.grouped = false;
    run.nested = false;
    run.lookup_mask_ready = false;
    run.group_child_count = 0;
    run.cold_arena_slot = -1;
    run.parent_slot = -1;
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
    run.quotient_off.release();
    run.op_words.release();
    run.count_delta.release();
    run.page_counts.release();
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
    // Range-only prefixes (2 x Rank23) are allocated lazily by
    // ensure_canonical_*_prefix on the first range query; lookup/successor-only
    // workloads never pay for them.
    rank23_value_prefix_ready_ = false;
    rank23_count_prefix_ready_ = false;
    canonical_overlay_active_ = false;
    ++canonical_generation_;
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
        compaction_stage_keys_, compaction_stage_payload_,
        compaction_keep_flags_, compaction_positions_,
        compaction_unique_offsets_, compaction_output_ops_,
        narrow_overflow_, assignment_views_,
        lookup_publication_,
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
        fold_fallback_count_, fold_fallback_head_, fold_cold_count_,
        fold_matched_count_, fold_stats_,
        cold_arena_keys_, cold_arena_values_, cold_arena_ops_,
        cold_arena_tail_, hash_overflow_offsets_arena_,
        hash_counts_arena_, hash_heavy_capacities_,
        hash_active_quotients_, hash_heavy_count_,
        hash_child_views_arena_, hash_child_router_arena_,
        parent_child_views_arena_);
    for (const auto &epoch : runs_) {
      total += device_bytes_all(
          epoch.keys, epoch.values, epoch.count_delta, epoch.quotient_off,
          epoch.op_words, epoch.lookup_mask, epoch.page_counts);
      for (const auto &child : epoch.hash_children)
        total += device_bytes_all(
            child.keys, child.values, child.quotient_off,
            child.op_words, child.lookup_mask, child.page_counts);
    }
    for (const auto &epoch : run_pool_)
      total += device_bytes_all(
          epoch.keys, epoch.values, epoch.count_delta, epoch.quotient_off,
          epoch.op_words, epoch.lookup_mask, epoch.page_counts);
    return total;
  }

private:
  struct AssignmentLeafStorage {
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> keys;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> values;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_off;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> op_words;
    gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> lookup_mask;
    // Paged cold-run metadata (empty for packed runs).
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> page_counts;
  };

  struct RetainedLeaf : AssignmentLeafStorage {
    std::size_t count = 0;
    std::uint64_t sequence = 0;
    gpulsmopt_detail::RunOperation operation =
        gpulsmopt_detail::RunOperation::Insert;
    bool lookup_mask_ready = false;
  };

  struct HostLookupLeaf {
    std::uint64_t sequence = 0;
    gpulsmopt_detail::LookupLeafView view{};
  };

  struct RunStorage : AssignmentLeafStorage {
    gpulsmopt_detail::RawDeviceBuffer<std::int8_t> count_delta;
    std::vector<RetainedLeaf> hash_children;
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
    bool hashed = false;
    bool grouped = false;
    bool nested = false;
    bool lookup_mask_ready = false;
    std::uint16_t group_child_count = 0;
    int cold_arena_slot = -1;
    int parent_slot = -1;
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
                             cudaStream_t stream) {
    acquire_run_slot();
    RunStorage &run = runs_.back();
    run.count = count;
    run.assignment = true;
    run.assignment_class = gpulsmopt_detail::AssignmentClass::Raw;
    run.paged = false;
    run.hashed = false;
    run.grouped = false;
    run.nested = false;
    run.lookup_mask_ready = false;
    run.group_child_count = 0;
    run.cold_arena_slot = -1;
    run.parent_slot = -1;
    run.hash_children.clear();
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
    if (is_insert) {
      // One fused pass: stable sort + dense quotient offsets.
      run.values.resize_discard(count);
      run.quotient_off.resize_discard_exact(
          gpulsmopt_detail::kEpochQuotients + 1);
      GPULSMOPT_PROF_PHASE(prof_delta_sort_ms_);
      build_raw_assignment_run(keys, values, count, run.keys.data(),
                               run.values.data(), run.quotient_off.data(),
                               stream);
    } else {
      {
        GPULSMOPT_PROF_PHASE(prof_delta_sort_ms_);
        run.values.resize_discard(0);
        CUDA_CHECK(cudaMemcpyAsync(run.keys.data(), keys,
                                   count * sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToDevice, stream));
      }
      {
        GPULSMOPT_PROF_PHASE(prof_delta_ingest_ms_);
        // Build quotient offsets from sorted deletion keys.
        run.quotient_off.resize_discard_exact(
            gpulsmopt_detail::kEpochQuotients + 1);
        build_quotient_offsets(run.keys.data(),
                               static_cast<std::uint32_t>(count),
                               run.quotient_off.data(), stream);
      }
    }
    publish_assignment_view(run, stream);
    invalidate_resolved();
    if (count_raw_runs() >= gpulsmopt_detail::kRawFoldWidth)
      temporal_hash_fold(stream);
  }

  std::size_t count_raw_runs() const {
    std::size_t n = 0;
    for (const auto &run : runs_)
      if (run.assignment && !run.nested &&
          run.assignment_class == gpulsmopt_detail::AssignmentClass::Raw)
        ++n;
    return n;
  }

  // Dense quotient offsets over quotient-ascending sorted keys in one kernel:
  // offsets[q] = lower_bound(keys, q<<16). Shared by insert and delete raw
  // leaves; replaces the boundary + reverse-min-scan + sentinel-memset path.
  void build_quotient_offsets(const std::uint32_t *sorted_keys,
                              std::uint32_t count, std::uint32_t *out_off,
                              cudaStream_t stream) {
    constexpr std::uint32_t Q = gpulsmopt_detail::kEpochQuotients;
    if (count == 0u) {
      CUDA_CHECK(cudaMemsetAsync(out_off, 0,
                                 (Q + 1u) * sizeof(std::uint32_t), stream));
      return;
    }
    constexpr int block = 256;
    const int grid = static_cast<int>((Q + 1u + block - 1) / block);
    gpulsmopt_detail::quotient_lower_bound_kernel<<<grid, block, 0, stream>>>(
        sorted_keys, count, out_off);
    CUDA_CHECK(cudaGetLastError());
  }

  // One fused operation for an ordinary insert raw run: stable
  // upper-16 sort + dense quotient offsets, no boundary kernel, no
  // reverse-min repair, no sentinel memset, no per-insert allocation.
  void build_raw_assignment_run(const std::uint32_t *keys,
                                const std::uint32_t *values,
                                std::size_t count, std::uint32_t *out_keys,
                                std::uint32_t *out_values,
                                std::uint32_t *out_quotient_off,
                                cudaStream_t stream) {
    if (count == 0u) {
      build_quotient_offsets(out_keys, 0u, out_quotient_off, stream);
      return;
    }
    // Stable sort preserves same-quotient input order (last-wins).
    sort_run_batch(keys, values, count, out_keys, out_values, stream);
    // Dense offsets in a single kernel over the sorted keys.
    build_quotient_offsets(out_keys, static_cast<std::uint32_t>(count),
                           out_quotient_off, stream);
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
  make_assignment_view(RunStorage &run,
                       bool with_lookup_mask = true) {
    const bool insert =
        run.operation == gpulsmopt_detail::RunOperation::Insert;
    const bool paged = run.paged;
    if (paged && (run.cold_arena_slot < 0 ||
                  run.cold_arena_slot >=
                      gpulsmopt_detail::kColdArenaSlots))
      throw std::runtime_error("paged run has no cold arena slot");
    if (run.grouped &&
        (run.parent_slot < 0 ||
         run.parent_slot >= gpulsmopt_detail::kParentSlots))
      throw std::runtime_error("grouped run has no parent slot");
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
    const bool hashed = run.hashed;
    const std::size_t hash_slot = hashed
                                      ? static_cast<std::size_t>(
                                            run.cold_arena_slot)
                                      : 0u;
    const std::uint32_t *offsets = hashed
        ? hash_overflow_offsets_arena_.data() +
              hash_slot * (gpulsmopt_detail::kEpochQuotients + 1u)
        : run.quotient_off.data();
    const std::uint32_t *counts = hashed
        ? hash_counts_arena_.data() +
              hash_slot * (gpulsmopt_detail::kEpochQuotients + 1u)
        : (paged ? run.page_counts.data() : nullptr);
    const std::uint32_t *hash_table = nullptr;
    const gpulsmopt_detail::AssignmentRunView *children = hashed
        ? hash_child_views_arena_.data() +
              hash_slot * gpulsmopt_detail::kRawFoldWidth
        : nullptr;
    const std::uint32_t *child_router = hashed
        ? hash_child_router_arena_.data() +
              hash_slot * gpulsmopt_detail::kHashRouteWordsPerParent
        : nullptr;
    const gpulsmopt_detail::AssignmentRunView *group_children = run.grouped
        ? parent_child_views_arena_.data() +
              static_cast<std::size_t>(run.parent_slot) *
                  gpulsmopt_detail::kStableFanout
        : nullptr;
    const std::uint64_t *lookup_mask =
        with_lookup_mask &&
                run.assignment_class ==
                    gpulsmopt_detail::AssignmentClass::Raw &&
                run.lookup_mask_ready
            ? run.lookup_mask.data()
            : nullptr;
    return {run.grouped ? nullptr : keys,
            run.grouped ? nullptr : (insert ? values : nullptr),
            run.grouped ? nullptr : offsets,
            run.grouped ? nullptr : counts,
            run.grouped ? nullptr : (run.mixed ? ops : nullptr),
            run.grouped ? nullptr : lookup_mask,
            run.grouped ? nullptr : hash_table,
            run.grouped ? nullptr : children,
            run.grouped ? nullptr : child_router,
            group_children,
            static_cast<std::uint16_t>(0u),
            static_cast<std::uint16_t>(
                run.grouped ? run.group_child_count
                            : (hashed ? run.hash_children.size() : 0u)),
            static_cast<std::uint8_t>(insert ? 1u : 0u),
            static_cast<std::uint8_t>(run.mixed ? 1u : 0u),
            static_cast<std::uint8_t>(paged ? 1u : 0u),
            static_cast<std::uint8_t>(hashed ? 1u : 0u),
            static_cast<std::uint8_t>(run.grouped ? 1u : 0u)};
  }

  // Publishes one chronological run descriptor.
  void publish_assignment_view(RunStorage &run, cudaStream_t stream) {
    (void)stream;
    gpulsmopt_detail::AssignmentRunView view =
        make_assignment_view(run);
    const std::size_t slot = chrono_views_.size();
    if (slot >= static_cast<std::size_t>(gpulsmopt_detail::kRunCapacity))
      throw std::runtime_error("assignment descriptor capacity exceeded");
    chrono_views_.push_back(view);
    host_state_->views[slot] = view;
    if (assignment_views_.size() < chrono_views_.size())
      assignment_views_.resize_discard(gpulsmopt_detail::kRunCapacity);
    // Device descriptors are materialized lazily in one batched copy right
    // before a read consumes them (flush_pending_views). Inserts never read
    // the device array, so this drops an H2D copy + stream op from every
    // insert; the fold uses its own fold_source_views_ buffer.
    views_dirty_ = true;
  }

  // Sync the device descriptor array with the host run views. Only the read
  // paths (lookup/successor/range) consume it, and they call this after
  // ensure_sorted_run_cache. host_state_->views[0..chrono_views_) always
  // mirrors the live runs (publish writes both in lockstep; clears zero the
  // count), so one batched copy of the current run count is exact.
  gpulsmopt_detail::AssignmentRunView
  make_retained_leaf_view(RetainedLeaf &leaf) {
    const bool insert =
        leaf.operation == gpulsmopt_detail::RunOperation::Insert;
    return {leaf.keys.data(),
            insert ? leaf.values.data() : nullptr,
            leaf.quotient_off.data(),
            nullptr,
            nullptr,
            leaf.lookup_mask_ready ? leaf.lookup_mask.data() : nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            0u,
            0u,
            static_cast<std::uint8_t>(insert ? 1u : 0u),
            0u,
            0u,
            0u,
            0u};
  }

  void ensure_raw_lookup_masks(cudaStream_t stream) {
    bool changed = false;
    constexpr int block = 256;
    constexpr int warps = block / 32;
    constexpr int grid =
        (gpulsmopt_detail::kEpochQuotients + warps - 1) /
        warps;
    for (RunStorage &run : runs_) {
      if (!run.assignment)
        continue;
      if (run.assignment_class ==
              gpulsmopt_detail::AssignmentClass::Raw &&
          !run.lookup_mask_ready) {
        run.lookup_mask.resize_discard_exact(
            gpulsmopt_detail::kEpochQuotients);
        gpulsmopt_detail::raw_quotient_mask_kernel<<<
            grid, block, 0, stream>>>(
            run.keys.data(), run.quotient_off.data(),
            run.lookup_mask.data());
        CUDA_CHECK(cudaGetLastError());
        run.lookup_mask_ready = true;
        changed = true;
      }
      if (!run.hashed || run.hash_children.empty())
        continue;
      bool refresh = false;
      for (RetainedLeaf &child : run.hash_children) {
        if (!child.lookup_mask_ready) {
          child.lookup_mask.resize_discard_exact(
              gpulsmopt_detail::kEpochQuotients);
          gpulsmopt_detail::raw_quotient_mask_kernel<<<
              grid, block, 0, stream>>>(
              child.keys.data(), child.quotient_off.data(),
              child.lookup_mask.data());
          CUDA_CHECK(cudaGetLastError());
          child.lookup_mask_ready = true;
          refresh = true;
        }
      }
      if (!refresh)
        continue;
      for (std::size_t child = 0;
           child < run.hash_children.size(); ++child)
        host_state_->scratch_views[child] =
            make_retained_leaf_view(run.hash_children[child]);
      const std::size_t view_base =
          static_cast<std::size_t>(run.cold_arena_slot) *
          gpulsmopt_detail::kRawFoldWidth;
      CUDA_CHECK(cudaMemcpyAsync(
          hash_child_views_arena_.data() + view_base,
          host_state_->scratch_views,
          run.hash_children.size() *
              sizeof(gpulsmopt_detail::AssignmentRunView),
          cudaMemcpyHostToDevice, stream));
      changed = true;
    }
    if (changed) {
      lookup_publication_ready_ = false;
      rebuild_chrono_views(stream);
    }
  }

  int prepare_lookup_leaf_views(cudaStream_t stream) {
    if (lookup_publication_ready_)
      return lookup_leaf_count_;
    lookup_host_leaves_.clear();
    int parent_count = 0;

    for (RunStorage &run : runs_) {
      if (!run.assignment || run.grouped)
        continue;
      if (!run.hashed) {
        lookup_host_leaves_.push_back(
            {run.sequence_end,
             {make_assignment_view(run),
              0u,
              gpulsmopt_detail::kNoLookupParent,
              0u}});
        continue;
      }
      if (run.hash_children.empty())
        continue;
      if (parent_count >= gpulsmopt_detail::kColdArenaSlots)
        throw std::runtime_error("lookup parent capacity exceeded");
      host_state_->lookup.parents[parent_count] =
          make_assignment_view(run);
      host_state_->lookup.parent_ranks[parent_count] = 0u;
      for (std::size_t child = 0;
           child < run.hash_children.size(); ++child) {
        RetainedLeaf &leaf = run.hash_children[child];
        lookup_host_leaves_.push_back(
            {leaf.sequence,
             {make_retained_leaf_view(leaf),
              0u,
              static_cast<std::uint16_t>(parent_count),
              static_cast<std::uint8_t>(
                  child + 1u == run.hash_children.size())}});
      }
      ++parent_count;
    }

    std::stable_sort(
        lookup_host_leaves_.begin(), lookup_host_leaves_.end(),
        [](const HostLookupLeaf &left,
           const HostLookupLeaf &right) {
          return left.sequence < right.sequence;
        });
    if (lookup_host_leaves_.size() >
        static_cast<std::size_t>(
            gpulsmopt_detail::kLookupLeafCapacity))
      throw std::runtime_error("lookup leaf capacity exceeded");

    for (std::size_t i = 0;
         i < lookup_host_leaves_.size(); ++i) {
      gpulsmopt_detail::LookupLeafView view =
          lookup_host_leaves_[i].view;
      view.rank = static_cast<std::uint32_t>(i + 1u);
      host_state_->lookup.leaves[i] = view;
      if (view.parent != gpulsmopt_detail::kNoLookupParent)
        host_state_->lookup.parent_ranks[view.parent] =
            std::max(host_state_->lookup.parent_ranks[view.parent],
                     view.rank);
    }

    CUDA_CHECK(cudaMemcpyAsync(
        lookup_publication_.data(), &host_state_->lookup,
        sizeof(gpulsmopt_detail::LookupPublication),
        cudaMemcpyHostToDevice, stream));
    lookup_leaf_count_ =
        static_cast<int>(lookup_host_leaves_.size());
    lookup_publication_ready_ = true;
    return lookup_leaf_count_;
  }

  void flush_pending_views(cudaStream_t stream) {
    if (!views_dirty_)
      return;
    const std::size_t n = chrono_views_.size();
    if (n != 0)
      CUDA_CHECK(cudaMemcpyAsync(
          assignment_views_.data(), host_state_->views,
          n * sizeof(gpulsmopt_detail::AssignmentRunView),
          cudaMemcpyHostToDevice, stream));
    views_dirty_ = false;
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
    while (run_pool_.size() < gpulsmopt_detail::kWarmLeafCount)
      run_pool_.emplace_back();
    for (RunStorage &leaf : run_pool_) {
      leaf.keys.resize_discard(count);
      leaf.values.resize_discard(count);
      leaf.quotient_off.resize_discard_exact(
          gpulsmopt_detail::kEpochQuotients + 1);
      leaf.lookup_mask.resize_discard_exact(
          gpulsmopt_detail::kEpochQuotients);
    }
  }

  void release_hash_children(RunStorage &run) {
    for (RetainedLeaf &child : run.hash_children) {
      run_pool_.emplace_back();
      RunStorage &leaf = run_pool_.back();
      leaf.keys = std::move(child.keys);
      leaf.values = std::move(child.values);
      leaf.quotient_off = std::move(child.quotient_off);
      leaf.op_words = std::move(child.op_words);
      leaf.lookup_mask = std::move(child.lookup_mask);
      leaf.page_counts = std::move(child.page_counts);
    }
    run.hash_children.clear();
  }

  void clear_run_state() {
    clear_sorted_state();
    for (auto &epoch : runs_) {
      if (epoch.hashed)
        release_hash_children(epoch);
      if (epoch.grouped && epoch.parent_slot >= 0 &&
          epoch.parent_slot < gpulsmopt_detail::kParentSlots)
        parent_slot_used_[epoch.parent_slot] = false;
      if (epoch.cold_arena_slot >= 0 &&
          epoch.cold_arena_slot < gpulsmopt_detail::kColdArenaSlots)
        cold_arena_slot_used_[epoch.cold_arena_slot] = false;
      epoch.hashed = false;
      epoch.grouped = false;
      epoch.nested = false;
      epoch.group_child_count = 0;
      epoch.paged = false;
      epoch.cold_arena_slot = -1;
      epoch.parent_slot = -1;
      run_pool_.push_back(std::move(epoch));
    }
    runs_.clear();
  }

  void insert_records(const std::uint32_t *keys_in,
                      const std::uint32_t *values_in, std::uint8_t op,
                      std::size_t count, cudaStream_t stream) {
    if (count == 0)
      return;
    const bool is_insert =
        op == static_cast<std::uint8_t>(gpulsmopt_detail::kInsert);
    create_assignment_run(is_insert, keys_in, is_insert ? values_in : nullptr,
                          count, stream);
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
    (void)stream;
    compaction_counts_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    compaction_offsets_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    compaction_tile_offsets_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    compaction_unique_offsets_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    compaction_stage_keys_.resize_discard(run_capacity);
    compaction_stage_payload_.resize_discard(run_capacity);
    compaction_keep_flags_.resize_discard(run_capacity);
    compaction_positions_.resize_discard(run_capacity);
    compaction_output_ops_.resize_discard(run_capacity);
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
    lookup_host_leaves_.reserve(
        gpulsmopt_detail::kLookupLeafCapacity);
    fold_source_views_.resize_discard(gpulsmopt_detail::kRawFoldWidth);
    fold_fallback_quotients_.resize_discard(
        gpulsmopt_detail::kEpochQuotients);
    fold_fallback_count_.resize_discard(1);
    fold_fallback_head_.resize_discard(1);
    fold_cold_count_.resize_discard(1);
    fold_matched_count_.resize_discard(1);
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
    hash_heavy_capacities_.resize_discard(
        gpulsmopt_detail::kEpochQuotients + 1u);
    const std::size_t hash_rows =
        gpulsmopt_detail::kEpochQuotients + 1u;
    hash_overflow_offsets_arena_.resize_discard(
        hash_rows * gpulsmopt_detail::kColdArenaSlots);
    hash_counts_arena_.resize_discard(
        hash_rows * gpulsmopt_detail::kColdArenaSlots);
    hash_active_quotients_.resize_discard(
        gpulsmopt_detail::kEpochQuotients);
    hash_heavy_count_.resize_discard(1);
    hash_child_views_arena_.resize_discard(
        gpulsmopt_detail::kRawFoldWidth *
        gpulsmopt_detail::kColdArenaSlots);
    hash_child_router_arena_.resize_discard(
        gpulsmopt_detail::kHashRouteWordsPerParent *
        gpulsmopt_detail::kColdArenaSlots);
    lookup_publication_.resize_discard(1);
    parent_child_views_arena_.resize_discard(
        gpulsmopt_detail::kStableFanout *
        gpulsmopt_detail::kParentSlots);
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
    for (int slot = 0; slot < gpulsmopt_detail::kParentSlots;
         ++slot)
      parent_slot_used_[slot] = false;
  }

  void prepare_for_insert(cudaStream_t stream) {
    const std::size_t direct_count =
        std::min(max_elements_,
                 std::max<std::size_t>(1, batch_capacity_));
    reserve_leaf_storage(direct_count);
    assignment_views_.resize_discard_exact(gpulsmopt_detail::kRunCapacity);
    chrono_views_.reserve(gpulsmopt_detail::kRunCapacity);
    prepare_sort_storage(direct_count, stream);
    reserve_temporal_compaction_storage(direct_count, stream);
    reserve_successor_storage();
    reserve_fold_storage(direct_count);
    reset_cold_arena(stream);
    constexpr int lookup_block = 256;
    const auto base = make_canonical_base_view();
    gpulsmopt_detail::temporal_lookup_leaf_block_kernel<<<
        1, lookup_block, 0, stream>>>(
        lookup_publication_.data(), 0, base, nullptr, 0,
        nullptr, nullptr);
    gpulsmopt_detail::temporal_lookup_leaf_thread_kernel<<<
        1, lookup_block, 0, stream>>>(
        lookup_publication_.data(), 0, base, nullptr, 0,
        nullptr, nullptr);
    CUDA_CHECK(cudaGetLastError());
    if (direct_count > 0 && !run_pool_.empty()) {
      RunStorage &sample = run_pool_.back();
      sort_run_batch(
          direct_sort_keys_.data(), direct_sort_values_.data(),
          direct_count, sample.keys.data(), sample.values.data(), stream);
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
    lookup_publication_ready_ = false;
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
      host_state_->scratch_views[slot] =
          make_assignment_view(runs_[idx[slot]]);
    CUDA_CHECK(cudaMemcpyAsync(
        normalize_views_.data(), host_state_->scratch_views,
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
      if (runs_[r].assignment && !runs_[r].nested &&
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
    base.quotient_off.release();
    build_sorted_run_cache(sorted_run_index_, stream);
    reset_canonical_overlay(stream);
  }

  // Folds the base and assignments with last-wins.
  void fold_into_base(cudaStream_t stream) {
    std::vector<std::size_t> updates;
    std::size_t assignment_total = 0;
    for (std::size_t r = 0; r < runs_.size(); ++r) {
      if (!runs_[r].assignment || runs_[r].nested)
        continue;
      updates.push_back(r);
      assignment_total += runs_[r].count;
    }
    if (updates.empty()) {
      bake_overlay_into_base(stream);
      return;
    }
    std::stable_sort(updates.begin(), updates.end(),
                     [this](std::size_t a, std::size_t b) {
                       if (runs_[a].sequence_begin !=
                           runs_[b].sequence_begin)
                         return runs_[a].sequence_begin <
                                runs_[b].sequence_begin;
                       return runs_[a].sequence_end < runs_[b].sequence_end;
                     });

    const std::size_t base_count =
        sorted_run_ready() ? sorted_run().count : 0u;
    const std::size_t total = base_count + assignment_total;
    if (total == 0)
      return;
    if (total > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("full consolidation exceeds CUB limits");

    resolve_keys_.resize_discard(total);
    resolve_payload_.resize_discard(total);
    resolve_alt_keys_.resize_discard(total);
    resolve_alt_payload_.resize_discard(total);
    resolve_flags_.resize_discard(total);
    resolve_sel_vdelta_.resize_discard(total);
    resolve_count_.resize_discard(1);
    constexpr int block = 256;

    if (base_count > 0) {
      const int base_grid =
          static_cast<int>((base_count + block - 1u) / block);
      gpulsmopt_detail::resolve_pack_canonical_base_kernel<<<
          base_grid, block, 0, stream>>>(
          make_canonical_base_view(), resolve_keys_.data(),
          resolve_payload_.data());
      CUDA_CHECK(cudaGetLastError());
    }

    normalize_views_.resize_discard(updates.size());
    for (std::size_t slot = 0; slot < updates.size(); ++slot)
      host_state_->scratch_views[slot] =
          make_assignment_view(runs_[updates[slot]]);
    CUDA_CHECK(cudaMemcpyAsync(
        normalize_views_.data(), host_state_->scratch_views,
        updates.size() * sizeof(gpulsmopt_detail::AssignmentRunView),
        cudaMemcpyHostToDevice, stream));
    constexpr int quotient_rows = gpulsmopt_detail::kEpochQuotients + 1;
    constexpr int count_grid = (quotient_rows + block - 1) / block;
    gpulsmopt_detail::assignment_group_count_kernel<<<
        count_grid, block, 0, stream>>>(
        normalize_views_.data(), static_cast<int>(updates.size()),
        compaction_counts_.data());
    CUDA_CHECK(cudaGetLastError());
    exclusive_scan_u32(
        compaction_counts_.data(), compaction_offsets_.data(),
        gpulsmopt_detail::kEpochQuotients + 1u, stream);
    gpulsmopt_detail::assignment_group_gather_kernel<<<
        gpulsmopt_detail::kEpochQuotients, block, 0, stream>>>(
        normalize_views_.data(), static_cast<int>(updates.size()),
        compaction_offsets_.data(), resolve_keys_.data() + base_count,
        resolve_payload_.data() + base_count);
    CUDA_CHECK(cudaGetLastError());

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
    const int grid = static_cast<int>((total + block - 1u) / block);
    gpulsmopt_detail::resolve_live_last_kernel<<<grid, block, 0, stream>>>(
        resolve_alt_keys_.data(), resolve_alt_payload_.data(), total,
        resolve_sel_vdelta_.data(), resolve_flags_.data());
    CUDA_CHECK(cudaGetLastError());

    clear_sorted_state();
    for (auto &run : runs_)
      run_pool_.push_back(std::move(run));
    runs_.clear();
    acquire_compaction_slot();
    RunStorage &base = runs_.back();
    base.count = 0;
    base.sequence = 0;
    base.sequence_begin = 0;
    base.sequence_end = 0;
    base.stable_level = -1;
    base.operation = gpulsmopt_detail::RunOperation::Insert;
    base.assignment_class = gpulsmopt_detail::AssignmentClass::Raw;
    base.mixed = false;
    base.assignment = false;
    base.fully_sorted = true;
    base.unit_counts = true;
    base.unique_keys = true;
    base.paged = false;
    base.hashed = false;
    base.grouped = false;
    base.nested = false;
    base.group_child_count = 0;
    base.cold_arena_slot = -1;
    base.parent_slot = -1;
    base.keys.resize_discard(total);
    base.values.resize_discard(total);

    std::size_t select_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, select_bytes, resolve_alt_keys_.data(),
        resolve_flags_.data(), base.keys.data(), resolve_count_.data(),
        static_cast<int>(total), stream));
    ensure_sort_temp(select_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        sort_temp_storage_.data(), select_bytes, resolve_alt_keys_.data(),
        resolve_flags_.data(), base.keys.data(), resolve_count_.data(),
        static_cast<int>(total), stream));
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        sort_temp_storage_.data(), select_bytes, resolve_sel_vdelta_.data(),
        resolve_flags_.data(), base.values.data(), resolve_count_.data(),
        static_cast<int>(total), stream));
    std::uint32_t kept = 0u;
    CUDA_CHECK(cudaMemcpyAsync(&kept, resolve_count_.data(), sizeof(kept),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    base.quotient_off.release();
    base.op_words.release();
    base.count_delta.release();
    base.page_counts.release();
    base.count = kept;
    base.keys.resize_discard(kept);
    base.values.resize_discard(kept);
    build_sorted_run_cache(0u, stream);
    run_sequence_ = 0;
    chrono_views_.clear();
    views_dirty_ = false;
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
      if (runs_[r].assignment && !runs_[r].nested)
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
    cold.hashed = false;
    cold.grouped = false;
    cold.nested = false;
    cold.group_child_count = 0;
    cold.cold_arena_slot = slot;
    cold.parent_slot = -1;
    cold.mixed = true;
    cold.operation = gpulsmopt_detail::RunOperation::Insert;
    cold.unique_keys = false;
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
    return idx;
  }

  void temporal_hash_fold(cudaStream_t stream) {
#ifdef GPULSMOPT_PROFILE_FOLD
    double prof_hash_publish_ms = 0.0;
    double prof_hash_plan_ms = 0.0;
    double prof_hash_scan_ms = 0.0;
    double prof_hash_build_ms = 0.0;
#endif
    std::vector<std::size_t> raw;
    for (std::size_t r = 0; r < runs_.size(); ++r) {
      if (runs_[r].assignment && !runs_[r].nested &&
          runs_[r].assignment_class ==
              gpulsmopt_detail::AssignmentClass::Raw)
        raw.push_back(r);
    }
    std::sort(raw.begin(), raw.end(), [this](std::size_t a,
                                             std::size_t b) {
      return runs_[a].sequence_end < runs_[b].sequence_end;
    });
    if (raw.size() < gpulsmopt_detail::kRawFoldWidth)
      return;
    raw.resize(gpulsmopt_detail::kRawFoldWidth);

    int arena_slot = -1;
    for (int slot = 0; slot < gpulsmopt_detail::kColdArenaSlots; ++slot) {
      if (!cold_arena_slot_used_[slot]) {
        arena_slot = slot;
        break;
      }
    }
    if (arena_slot < 0)
      throw std::runtime_error("no free hash-fold arena slot");
    cold_arena_slot_used_[arena_slot] = true;

    std::uint64_t sequence_begin = ~std::uint64_t{0};
    std::uint64_t sequence_end = 0u;
    std::size_t input_records = 0u;
    for (std::size_t child = 0; child < raw.size(); ++child) {
      RunStorage &run = runs_[raw[child]];
      sequence_begin = std::min(sequence_begin, run.sequence_begin);
      sequence_end = std::max(sequence_end, run.sequence_end);
      input_records += run.count;
      host_state_->scratch_views[child] =
          make_assignment_view(run);
    }

    const std::size_t view_base =
        static_cast<std::size_t>(arena_slot) *
        gpulsmopt_detail::kRawFoldWidth;
    auto *child_views = hash_child_views_arena_.data() + view_base;
    {
      GPULSMOPT_FOLD_PHASE(prof_hash_publish_ms);
      CUDA_CHECK(cudaMemcpyAsync(
          child_views, host_state_->scratch_views,
          raw.size() * sizeof(gpulsmopt_detail::AssignmentRunView),
          cudaMemcpyHostToDevice, stream));
    }
    const std::size_t arena_base =
        static_cast<std::size_t>(arena_slot) *
        cold_arena_slot_capacity_;
    const std::size_t row_base =
        static_cast<std::size_t>(arena_slot) *
        (gpulsmopt_detail::kEpochQuotients + 1u);
    std::uint32_t *counts = hash_counts_arena_.data() + row_base;
    std::uint32_t *heavy_offsets =
        hash_overflow_offsets_arena_.data() + row_base;
    constexpr int block = 256;
    constexpr int rows = gpulsmopt_detail::kEpochQuotients + 1;
    constexpr int grid = (rows + block - 1) / block;
    {
      GPULSMOPT_FOLD_PHASE(prof_hash_plan_ms);
      CUDA_CHECK(cudaMemsetAsync(
          hash_heavy_count_.data(), 0,
          sizeof(std::uint32_t), stream));
      gpulsmopt_detail::temporal_hash_plan_kernel<<<
          grid, block, 0, stream>>>(
          child_views, static_cast<int>(raw.size()),
          hash_heavy_capacities_.data(), counts,
          hash_active_quotients_.data(),
          hash_heavy_count_.data());
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpyAsync(
          &host_state_->hash_heavy_count,
          hash_heavy_count_.data(), sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const std::uint32_t heavy_count =
        host_state_->hash_heavy_count;
    {
      GPULSMOPT_FOLD_PHASE(prof_hash_scan_ms);
      if (heavy_count != 0u) {
        exclusive_scan_u32(
            hash_heavy_capacities_.data(), heavy_offsets,
            gpulsmopt_detail::kEpochQuotients + 1u, stream);
      }
    }
    {
      GPULSMOPT_FOLD_PHASE(prof_hash_build_ms);
      if (heavy_count != 0u) {
        gpulsmopt_detail::temporal_hash_heavy_kernel<<<
            heavy_count, block, 0, stream>>>(
            child_views, static_cast<int>(raw.size()),
            hash_active_quotients_.data(), heavy_offsets,
            counts, cold_arena_slot_capacity_,
            cold_arena_keys_.data() + arena_base);
        CUDA_CHECK(cudaGetLastError());
      }
      std::uint32_t *child_router =
          hash_child_router_arena_.data() +
          static_cast<std::size_t>(arena_slot) *
              gpulsmopt_detail::kHashRouteWordsPerParent;
      gpulsmopt_detail::temporal_hash_route_kernel<<<
          gpulsmopt_detail::kEpochQuotients, 128, 0, stream>>>(
          child_views, static_cast<int>(raw.size()), counts,
          child_router);
      CUDA_CHECK(cudaGetLastError());
    }
#ifdef GPULSMOPT_PROFILE_FOLD
    printf("[hash-fold] active=%u heavy=%u publish=%.3f "
           "plan=%.3f scan=%.3f build=%.3f ms\n",
           heavy_count, heavy_count, prof_hash_publish_ms,
           prof_hash_plan_ms, prof_hash_scan_ms,
           prof_hash_build_ms);
#endif

    RunStorage folded;
    folded.assignment = true;
    folded.assignment_class =
        gpulsmopt_detail::AssignmentClass::ColdStable;
    folded.stable_level = 0;
    folded.paged = true;
    folded.hashed = true;
    folded.grouped = false;
    folded.nested = false;
    folded.cold_arena_slot = arena_slot;
    folded.mixed = true;
    folded.operation = gpulsmopt_detail::RunOperation::Insert;
    folded.sequence = sequence_end;
    folded.sequence_begin = sequence_begin;
    folded.sequence_end = sequence_end;
    folded.count = input_records;
    folded.unique_keys = false;
    folded.fully_sorted = false;
    folded.unit_counts = false;
    folded.hash_children.reserve(raw.size());
    for (const std::size_t index : raw) {
      RunStorage &source = runs_[index];
      folded.hash_children.emplace_back();
      RetainedLeaf &child = folded.hash_children.back();
      child.keys = std::move(source.keys);
      child.values = std::move(source.values);
      child.quotient_off = std::move(source.quotient_off);
      child.op_words = std::move(source.op_words);
      child.lookup_mask = std::move(source.lookup_mask);
      child.page_counts = std::move(source.page_counts);
      child.count = source.count;
      child.sequence = source.sequence_end;
      child.operation = source.operation;
      child.lookup_mask_ready = source.lookup_mask_ready;
    }
    std::sort(raw.begin(), raw.end());
    for (auto it = raw.rbegin(); it != raw.rend(); ++it) {
      if (sorted_run_ready() && sorted_run_index_ > *it)
        --sorted_run_index_;
      runs_.erase(runs_.begin() + static_cast<std::ptrdiff_t>(*it));
    }
    runs_.push_back(std::move(folded));
    invalidate_resolved();
    succ_sparse_ready_ = false;
    ++maintenance_stats_.canonical_fold_count;
    maintenance_stats_.canonical_input_records += input_records;
    ++maintenance_stats_.compaction_count;
    maintenance_stats_.compacted_input_records += input_records;
    maintenance_stats_.compacted_output_records += input_records;
    carry_cold_run(runs_.size() - 1u, stream);
    rebuild_chrono_views(stream);
  }

  // Fold the 64 oldest raw runs.
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
    std::uint64_t seq_end = 0;
    std::uint64_t seq_begin = ~std::uint64_t{0};
    std::size_t input_records = 0;
    std::size_t cold_idx = 0;
    {
      GPULSMOPT_FOLD_PHASE(prof_fold_publish_ms_);
      for (std::size_t slot = 0; slot < group_size; ++slot) {
        const std::size_t r = group[slot];
        seq_end = std::max(seq_end, runs_[r].sequence_end);
        seq_begin = std::min(seq_begin, runs_[r].sequence_begin);
        input_records += runs_[r].count;
        host_state_->scratch_views[slot] =
            make_assignment_view(runs_[r]);
      }
      fold_source_views_.resize_discard(group_size);
      CUDA_CHECK(cudaMemcpyAsync(
          fold_source_views_.data(), host_state_->scratch_views,
          group_size * sizeof(gpulsmopt_detail::AssignmentRunView),
          cudaMemcpyHostToDevice, stream));

      cold_idx = acquire_cold_run(stream);
      CUDA_CHECK(cudaMemsetAsync(fold_fallback_count_.data(), 0,
                                 sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(fold_fallback_head_.data(), 0,
                                 sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(fold_cold_count_.data(), 0,
                                 sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(fold_matched_count_.data(), 0,
                                 sizeof(std::uint32_t), stream));
    }

    RunStorage &cold = runs_[cold_idx];
    const std::size_t arena_slot =
        static_cast<std::size_t>(cold.cold_arena_slot);
    std::uint32_t *arena_keys =
        cold_arena_keys_.data() + arena_slot * cold_arena_slot_capacity_;
    std::uint32_t *arena_values =
        cold_arena_values_.data() + arena_slot * cold_arena_slot_capacity_;
    std::uint32_t *arena_ops =
        cold_arena_ops_.data() + arena_slot * cold_arena_slot_words_;
    std::uint32_t *arena_tail = cold_arena_tail_.data() + arena_slot;
    {
      GPULSMOPT_FOLD_PHASE(prof_fold_fast_ms_);
      gpulsmopt_detail::canonical_fold_rank23_kernel<<<
          gpulsmopt_detail::kEpochQuotients,
          gpulsmopt_detail::kFoldThreads, 0, stream>>>(
          fold_source_views_.data(), static_cast<int>(group_size),
          make_sorted_view(), base_override_state_.data(),
          base_override_values_.data(), rank23_value_delta_.data(),
          rank23_count_delta_.data(), cold.quotient_off.data(),
          cold.page_counts.data(), arena_tail, cold_arena_slot_capacity_,
          arena_keys, arena_values, arena_ops,
          fold_fallback_quotients_.data(), fold_fallback_count_.data(),
          fold_cold_count_.data(), fold_matched_count_.data(),
          fold_stats_.data());
      CUDA_CHECK(cudaGetLastError());
    }

    {
      GPULSMOPT_FOLD_PHASE(prof_fold_fallback_ms_);
      run_fold_fallback(group_size, cold, stream);
      CUDA_CHECK(cudaMemcpyAsync(
          &host_state_->fold_fallback_count,
          fold_fallback_count_.data(), sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          &host_state_->fold_cold_count,
          fold_cold_count_.data(), sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          &host_state_->fold_matched_count,
          fold_matched_count_.data(), sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    {
      GPULSMOPT_FOLD_PHASE(prof_fold_book_ms_);
      const std::size_t cold_records = host_state_->fold_cold_count;
      const std::size_t matched_records = host_state_->fold_matched_count;
      cold.count = cold_records;
      cold.sequence = seq_end;
      cold.sequence_begin = seq_begin;
      cold.sequence_end = seq_end;
      cold.stable_level = 0;

      RunStorage cold_moved = std::move(runs_[cold_idx]);
      runs_.erase(runs_.begin() + static_cast<std::ptrdiff_t>(cold_idx));
      retire_run_group(group, stream);
      if (cold_records == 0) {
        const int slot = cold_moved.cold_arena_slot;
        if (slot >= 0 && slot < gpulsmopt_detail::kColdArenaSlots)
          cold_arena_slot_used_[slot] = false;
        cold_moved.paged = false;
        cold_moved.cold_arena_slot = -1;
        run_pool_.push_back(std::move(cold_moved));
      } else {
        runs_.push_back(std::move(cold_moved));
      }

      if (matched_records != 0) {
        canonical_overlay_active_ = true;
        ++canonical_generation_;
        rank23_value_prefix_ready_ = false;
        rank23_count_prefix_ready_ = false;
      }
      invalidate_resolved();
      succ_sparse_ready_ = false;
      ++maintenance_stats_.canonical_fold_count;
      maintenance_stats_.canonical_input_records += input_records;
      ++maintenance_stats_.compaction_count;
      maintenance_stats_.compacted_input_records += input_records;
      maintenance_stats_.compacted_output_records += cold_records;
    }

    {
      GPULSMOPT_FOLD_PHASE(prof_fold_carry_ms_);
      if (host_state_->fold_cold_count != 0)
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
        fold_fallback_quotients_.data(), fold_fallback_count_.data(),
        fold_fallback_head_.data(), cold.quotient_off.data(),
        cold.page_counts.data(), cold_arena_tail_.data() + slot,
        cold_arena_slot_capacity_,
        cold_arena_keys_.data() + slot * cold_arena_slot_capacity_,
        cold_arena_values_.data() + slot * cold_arena_slot_capacity_,
        cold_arena_ops_.data() + slot * cold_arena_slot_words_,
        fold_cold_count_.data(), fold_stats_.data());
    CUDA_CHECK(cudaGetLastError());
  }

  int acquire_parent_slot() {
    for (int slot = 0; slot < gpulsmopt_detail::kParentSlots; ++slot) {
      if (parent_slot_used_[slot])
        continue;
      parent_slot_used_[slot] = true;
      return slot;
    }
    throw std::runtime_error("parent descriptor capacity exceeded");
  }

  void make_parent_group(std::vector<std::size_t> group,
                         int out_level,
                         cudaStream_t stream) {
    std::sort(group.begin(), group.end(),
              [this](std::size_t a, std::size_t b) {
                return runs_[a].sequence_end < runs_[b].sequence_end;
              });
    const int parent_slot = acquire_parent_slot();
    std::uint64_t sequence_begin = ~std::uint64_t{0};
    std::uint64_t sequence_end = 0u;
    std::size_t records = 0u;
    for (std::size_t child = 0; child < group.size(); ++child) {
      RunStorage &run = runs_[group[child]];
      sequence_begin = std::min(sequence_begin, run.sequence_begin);
      sequence_end = std::max(sequence_end, run.sequence_end);
      records += run.count;
      host_state_->scratch_views[child] = make_assignment_view(run);
    }
    CUDA_CHECK(cudaMemcpyAsync(
        parent_child_views_arena_.data() +
            static_cast<std::size_t>(parent_slot) *
                gpulsmopt_detail::kStableFanout,
        host_state_->scratch_views,
        group.size() * sizeof(gpulsmopt_detail::AssignmentRunView),
        cudaMemcpyHostToDevice, stream));
    for (const std::size_t child : group)
      runs_[child].nested = true;

    RunStorage parent;
    parent.assignment = true;
    parent.assignment_class =
        gpulsmopt_detail::AssignmentClass::ColdStable;
    parent.stable_level = out_level;
    parent.operation = gpulsmopt_detail::RunOperation::Insert;
    parent.sequence = sequence_end;
    parent.sequence_begin = sequence_begin;
    parent.sequence_end = sequence_end;
    parent.count = records;
    parent.mixed = true;
    parent.grouped = true;
    parent.group_child_count =
        static_cast<std::uint16_t>(group.size());
    parent.parent_slot = parent_slot;
    runs_.push_back(std::move(parent));

    ++maintenance_stats_.cold_tier_compaction_count;
    maintenance_stats_.cold_tier_input_records += records;
    maintenance_stats_.cold_tier_output_records += records;
  }

  // Four children become one pointer-only parent.
  void carry_cold_run(std::size_t cold_idx, cudaStream_t stream) {
    (void)cold_idx;
    for (int level = 0; level < gpulsmopt_detail::kStableLevels - 1;
         ++level) {
      std::vector<std::size_t> group;
      for (std::size_t r = 0; r < runs_.size(); ++r) {
        const RunStorage &run = runs_[r];
        if (!run.assignment || run.nested ||
            run.assignment_class !=
                gpulsmopt_detail::AssignmentClass::ColdStable ||
            run.stable_level != level)
          continue;
        group.push_back(r);
      }
      if (group.size() <
          static_cast<std::size_t>(gpulsmopt_detail::kStableFanout))
        return;
      std::sort(group.begin(), group.end(),
                [this](std::size_t a, std::size_t b) {
                  return runs_[a].sequence_end < runs_[b].sequence_end;
                });
      group.resize(gpulsmopt_detail::kStableFanout);
      make_parent_group(group, level + 1, stream);
    }
  }

  // Merge a set of assignment runs into one packed mixed run with
  // last-wins; retire the sources; return the merged run index.
  std::size_t merge_run_group(std::vector<std::size_t> group,
                              gpulsmopt_detail::AssignmentClass out_class,
                              int out_level, cudaStream_t stream) {
    std::stable_sort(group.begin(), group.end(),
                     [this](std::size_t a, std::size_t b) {
                       if (runs_[a].sequence_begin !=
                           runs_[b].sequence_begin)
                         return runs_[a].sequence_begin <
                                runs_[b].sequence_begin;
                       return runs_[a].sequence_end < runs_[b].sequence_end;
                     });
    const std::size_t group_size = group.size();
    std::uint64_t group_seq = 0;
    std::uint64_t seq_begin = ~std::uint64_t{0};
    std::size_t total = 0;
    for (const std::size_t r : group) {
      group_seq = std::max(group_seq, runs_[r].sequence_end);
      seq_begin = std::min(seq_begin, runs_[r].sequence_begin);
      total += runs_[r].count;
    }
    if (total == 0) {
      retire_run_group(group, stream);
      rebuild_chrono_views(stream);
      invalidate_resolved();
      return runs_.size();
    }
    if (total > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("temporal compaction exceeds CUB limits");

    normalize_views_.resize_discard(group_size);
    for (std::size_t slot = 0; slot < group_size; ++slot)
      host_state_->scratch_views[slot] =
          make_assignment_view(runs_[group[slot]]);
    CUDA_CHECK(cudaMemcpyAsync(
        normalize_views_.data(), host_state_->scratch_views,
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
    const std::size_t actual = host_compaction_offsets_.back();
    if (actual > total)
      throw std::runtime_error("temporal compaction count mismatch");
    if (actual == 0) {
      retire_run_group(group, stream);
      rebuild_chrono_views(stream);
      invalidate_resolved();
      return runs_.size();
    }

    compaction_stage_keys_.resize_discard(actual);
    compaction_stage_payload_.resize_discard(actual);
    compaction_keep_flags_.resize_discard(actual);
    compaction_positions_.resize_discard(actual);

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

    for (const QuotientTile &tile : tiles) {
      const std::uint32_t tile_first = tile.first;
      const std::uint32_t segments = tile.second - tile.first;
      const std::size_t tile_begin =
          host_compaction_offsets_[tile.first];
      const std::size_t tile_count =
          host_compaction_offsets_[tile.second] - tile_begin;
      if (tile_count == 0u)
        continue;
      resolve_keys_.resize_discard(tile_count);
      resolve_payload_.resize_discard(tile_count);
      resolve_alt_keys_.resize_discard(tile_count);
      resolve_alt_payload_.resize_discard(tile_count);
      const int offset_grid = static_cast<int>(
          (segments + 1u + block - 1u) / block);
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
      const int grid =
          static_cast<int>((tile_count + block - 1u) / block);
      gpulsmopt_detail::compaction_stage_unique_kernel<<<
          grid, block, 0, stream>>>(
          resolve_alt_keys_.data(), resolve_alt_payload_.data(), tile_count,
          tile_begin, compaction_stage_keys_.data(),
          compaction_stage_payload_.data(),
          compaction_keep_flags_.data());
      CUDA_CHECK(cudaGetLastError());
    }

    exclusive_scan_u32(
        compaction_keep_flags_.data(), compaction_positions_.data(),
        actual, stream);
    gpulsmopt_detail::compaction_output_offsets_kernel<<<
        count_grid, block, 0, stream>>>(
        compaction_offsets_.data(), compaction_positions_.data(),
        compaction_keep_flags_.data(), static_cast<std::uint32_t>(actual),
        compaction_unique_offsets_.data());
    CUDA_CHECK(cudaGetLastError());
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
    merged.hashed = false;
    merged.grouped = false;
    merged.nested = false;
    merged.group_child_count = 0;
    merged.parent_slot = -1;
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
    const std::size_t op_word_count = (compact_count + 31u) / 32u;
    merged.op_words.resize_discard(op_word_count);
    compaction_output_ops_.resize_discard(compact_count);
    CUDA_CHECK(cudaMemcpyAsync(
        merged.quotient_off.data(), compaction_unique_offsets_.data(),
        (gpulsmopt_detail::kEpochQuotients + 1u) * sizeof(std::uint32_t),
        cudaMemcpyDeviceToDevice, stream));
    const int scatter_grid =
        static_cast<int>((actual + block - 1u) / block);
    gpulsmopt_detail::compaction_unique_scatter_kernel<<<
        scatter_grid, block, 0, stream>>>(
        compaction_stage_keys_.data(), compaction_stage_payload_.data(),
        compaction_keep_flags_.data(), compaction_positions_.data(), actual,
        merged.keys.data(), merged.values.data(),
        compaction_output_ops_.data());
    CUDA_CHECK(cudaGetLastError());
    const int word_grid =
        static_cast<int>((op_word_count + block - 1u) / block);
    gpulsmopt_detail::compaction_pack_ops_kernel<<<
        word_grid, block, 0, stream>>>(
        compaction_output_ops_.data(), compact_count,
        merged.op_words.data());
    CUDA_CHECK(cudaGetLastError());

    RunStorage compacted = std::move(runs_.back());
    runs_.pop_back();
    retire_run_group(group, stream);
    runs_.push_back(std::move(compacted));
    rebuild_chrono_views(stream);
    invalidate_resolved();
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
      if (run.hashed)
        release_hash_children(run);
      if ((run.paged || run.hashed) && run.cold_arena_slot >= 0 &&
          run.cold_arena_slot < gpulsmopt_detail::kColdArenaSlots)
        cold_arena_slot_used_[run.cold_arena_slot] = false;
      run.paged = false;
      run.hashed = false;
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
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> fold_cold_count_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> fold_matched_count_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> cold_arena_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> cold_arena_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> cold_arena_ops_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> cold_arena_tail_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      hash_overflow_offsets_arena_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      hash_heavy_capacities_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> hash_counts_arena_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      hash_active_quotients_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> hash_heavy_count_;
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::AssignmentRunView>
      hash_child_views_arena_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      hash_child_router_arena_;
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::AssignmentRunView>
      parent_child_views_arena_;
  bool parent_slot_used_[gpulsmopt_detail::kParentSlots] = {};
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::AssignmentRunView>
      assignment_views_;
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::LookupPublication>
      lookup_publication_;
  // Oldest-to-newest descriptor mirror.
  std::vector<gpulsmopt_detail::AssignmentRunView> chrono_views_;
  std::vector<HostLookupLeaf> lookup_host_leaves_;
  // Set when a run is published; cleared when the device descriptor array is
  // batch-synced before a read (flush_pending_views).
  bool views_dirty_ = false;
  bool lookup_publication_ready_ = false;
  int lookup_leaf_count_ = 0;
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
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compaction_stage_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> compaction_stage_payload_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compaction_keep_flags_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compaction_positions_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compaction_unique_offsets_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> compaction_output_ops_;
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
