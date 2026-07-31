#pragma once
#include "gpu_dictionary_adapter.cuh"

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
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
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
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
#ifndef GPULSMOPT_LOOKUP_RUN_PARALLEL_MAX_QUERIES
#define GPULSMOPT_LOOKUP_RUN_PARALLEL_MAX_QUERIES (1 << 16)
#endif
#ifndef GPULSMOPT_LOOKUP_RUN_PARALLEL_MIN_RUNS
#define GPULSMOPT_LOOKUP_RUN_PARALLEL_MIN_RUNS 8
#endif
#ifndef GPULSMOPT_PREWARM_LEAVES
#define GPULSMOPT_PREWARM_LEAVES 576
#endif
constexpr int kRunCapacity = 128;
#ifdef GPULSMOPT_EPOCH_MAX
static_assert(GPULSMOPT_EPOCH_MAX == kRunCapacity,
              "GPULSMOPT_EPOCH_MAX must be 128");
#endif
constexpr int kEpochQuotientBits = 16;
constexpr int kEpochQuotients = 1 << kEpochQuotientBits;
// Flat BaseRun rank directory.
constexpr int kBaseRank23Bits = 23;
constexpr int kBaseRank23Shift = 32 - kBaseRank23Bits;
constexpr std::size_t kBaseRank23Size = std::size_t{1} << kBaseRank23Bits;
constexpr int kBaseBinsPerQuotient =
    1 << (kBaseRank23Bits - kEpochQuotientBits);
constexpr std::size_t kSortedRunMinRecords = 1u << 22;
// Temporal hash-fold geometry.
constexpr int kRawFoldWidth = 64;
constexpr int kStableFanout = 4;
constexpr int kColdArenaSlots = 8;
constexpr int kParentSlots = kRunCapacity;
constexpr int kHashFoldMaxRecords = 1280;
constexpr int kHashRouteBits = 6;
constexpr int kHashRouteBins = 1 << kHashRouteBits;
constexpr int kHashRouteWordsPerBin = 2;
constexpr int kHashRouteMaskBuildMax = 4;
constexpr int kResolveSmallMax = 128;
constexpr int kResolveMediumMax = 512;
constexpr int kResolveLocalMax = 2048;
constexpr std::size_t kHashRouteWordsPerParent =
    static_cast<std::size_t>(kEpochQuotients) *
    kHashRouteBins * kHashRouteWordsPerBin;
constexpr int kLookupLeafCapacity =
    kColdArenaSlots * kRawFoldWidth + kRawFoldWidth;
constexpr std::uint16_t kNoLookupParent = 0xffffu;
constexpr std::uint32_t kNoOwnerLeaf = 0xffffffffu;
constexpr std::uint32_t kLookupOwnerMaxProbe = 1024u;
constexpr std::size_t kLookupOwnerMaxSlots = std::size_t{1} << 29;
constexpr std::uint32_t kHashHeavyFlag = 1u << 31;
constexpr std::uint32_t kHashScanFlag = 1u << 30;
constexpr std::uint32_t kHashCountMask =
    ~(kHashHeavyFlag | kHashScanFlag);
constexpr std::size_t kWarmLeafCount =
    GPULSMOPT_PREWARM_LEAVES;
constexpr std::size_t kLeafSlabRuns = 64;
constexpr std::size_t kInitialLeafSlabs =
    (kWarmLeafCount + kLeafSlabRuns - 1u) /
    kLeafSlabRuns;
constexpr std::size_t kInitialInsertLeafSlabs =
    (kInitialLeafSlabs + 1u) / 2u;
constexpr std::size_t kInitialDeleteLeafSlabs =
    kInitialLeafSlabs / 2u;
constexpr int kStableLevels = 16;
static_assert(kRawFoldWidth == 64, "raw fold width must be 64");
static_assert(kWarmLeafCount >= kRawFoldWidth,
              "prewarm must hold one raw fold");
static_assert(kStableFanout == 4, "stable fanout must be 4");
static_assert(kStableLevels <= 16, "stable levels bounded to 16");
static_assert(1 + kRawFoldWidth + 3 * kStableLevels + 1 <= kRunCapacity,
              "descriptor occupancy must fit run capacity");
static_assert(GPULSMOPT_RADIX_THREADS % 32 == 0,
              "radix block size must be warp aligned");
static_assert(kRunCapacity == 128, "run kernels require 128 physical slots");
constexpr std::uint32_t kEmptyKey = std::numeric_limits<std::uint32_t>::max();

// QVRF uses immutable, quotient-aligned Base leaves and sparse visible-cell
// overrides.  This is a hardware/coalescing granularity, not a query-size
// heuristic.  Each rank23 cell still bounds visibility reconstruction to 512
// possible keys.
constexpr std::uint32_t kQvrfBaseLeafRecords = 128u;
constexpr std::uint32_t kQvrfTopFanout = 16u;
constexpr std::uint32_t kQvrfTopL1 = kEpochQuotients / kQvrfTopFanout;
constexpr std::uint32_t kQvrfTopL2 = kQvrfTopL1 / kQvrfTopFanout;
constexpr std::uint32_t kQvrfTopL3 = kQvrfTopL2 / kQvrfTopFanout;
constexpr std::uint32_t kQvrfTopL4 = kQvrfTopL3 / kQvrfTopFanout;
constexpr std::uint32_t kQvrfMaterializationFreeDivisor = 8u;
static_assert(kQvrfTopL4 == 1u, "QVRF top geometry must reach one root");

constexpr std::uint32_t kInsert = 1;
constexpr std::uint32_t kTombstone = 0;

// Assignment runs avoid eager owner transitions.
enum class RunOperation : std::uint8_t { Insert, Delete };
// Raw = pending batch run; ColdStable = folded non-base run.
enum class AssignmentClass : std::uint8_t { Raw, ColdStable };
// Canonical overlay state per BaseRun position.
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
      : data_(other.data_), size_(other.size_), capacity_(other.capacity_),
        owns_(other.owns_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
    other.owns_ = false;
  }
  RawDeviceBuffer &operator=(RawDeviceBuffer &&other) noexcept {
    if (this == &other)
      return *this;
    if (data_ && owns_)
      cudaFree(data_);
    data_ = other.data_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    owns_ = other.owns_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
    other.owns_ = false;
    return *this;
  }
  ~RawDeviceBuffer() {
    if (data_ && owns_)
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
      if (data_ && owns_)
        CUDA_CHECK(cudaFree(data_));
      data_ = next;
      capacity_ = next_capacity;
      owns_ = true;
    }
    size_ = count;
  }

  void resize_discard_exact(std::size_t count) {
    if (count > capacity_) {
      T *next = nullptr;
      CUDA_CHECK(
          cudaMalloc(reinterpret_cast<void **>(&next), count * sizeof(T)));
      if (data_ && owns_)
        CUDA_CHECK(cudaFree(data_));
      data_ = next;
      capacity_ = count;
      owns_ = true;
    }
    size_ = count;
  }

  void release() {
    if (data_ && owns_)
      CUDA_CHECK(cudaFree(data_));
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
    owns_ = false;
  }

  void bind_external(T *data, std::size_t capacity) {
    if (data_ && owns_)
      CUDA_CHECK(cudaFree(data_));
    data_ = data;
    size_ = 0;
    capacity_ = capacity;
    owns_ = false;
  }

  void adopt(T *data, std::size_t count) {
    if (data_ && owns_)
      CUDA_CHECK(cudaFree(data_));
    data_ = data;
    size_ = count;
    capacity_ = count;
    owns_ = true;
  }

  T *data() { return data_; }
  const T *data() const { return data_; }
  std::size_t size() const { return size_; }
  std::size_t capacity() const { return capacity_; }
  bool owns() const { return owns_; }
  bool external() const { return data_ != nullptr && !owns_; }

private:
  T *data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
  bool owns_ = false;
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
  if (sorted.count == 0u) {
    *lower = 0u;
    *upper = 0u;
    return;
  }
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

struct BaseRunView {
  SortedRunView base;
};

__device__ inline bool base_find_value(
    const BaseRunView &v, std::uint32_t key,
    std::uint32_t *value) {
  return sorted_find_value(v.base, key, value);
}

struct alignas(16) QvrfSummary {
  std::uint32_t sum;
  std::uint32_t count;
  std::uint32_t minimum;
  std::uint32_t maximum;
};

__host__ __device__ inline QvrfSummary qvrf_identity() {
  return {0u, 0u, std::numeric_limits<std::uint32_t>::max(), 0u};
}

__host__ __device__ inline QvrfSummary
qvrf_lift(std::uint32_t key, std::uint32_t value) {
  (void)key;
  return {value, 1u, value, value};
}

// combine is deliberately ordered.  Sum/count/min/max happen to commute, but
// the forest traversal uses this same left-to-right shape for custom monoids.
__host__ __device__ inline QvrfSummary
qvrf_combine(const QvrfSummary &left, const QvrfSummary &right) {
  if (left.count == 0u)
    return right;
  if (right.count == 0u)
    return left;
  return {left.sum + right.sum,
          left.count + right.count,
          left.minimum < right.minimum ? left.minimum : right.minimum,
          left.maximum > right.maximum ? left.maximum : right.maximum};
}

struct QvrfView {
  SortedRunView base;
  RunView resolved;
  const std::uint32_t *resolved_values;
  const std::int8_t *resolved_counts;
  const std::uint32_t *base_q_block_offsets;
  const QvrfSummary *base_block_summaries;
  const std::uint32_t *cell_ids;
  const std::uint32_t *cell_offsets;
  const std::uint32_t *cell_keys;
  const std::uint32_t *cell_values;
  const QvrfSummary *cell_summaries;
  const std::uint32_t *q_cell_offsets;
  const QvrfSummary *q_summaries;
  const QvrfSummary *top_l1;
  const QvrfSummary *top_l2;
  const QvrfSummary *top_l3;
  const QvrfSummary *top_l4;
  std::uint32_t cell_count;
  std::uint32_t materialized_cells;
};

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
__global__ void base_successor_kernel(BaseRunView base,
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
  results[i] = position < base.base.count ? base.base.keys[position] : 0u;
}

// Immutable assignment run seen by device readers.
struct AssignmentRunView {
  const std::uint32_t *keys;
  const std::uint32_t *values;
  const std::uint32_t *offsets;
  const std::uint32_t *quotient_meta;
  const std::uint32_t *op_words;
  const std::uint64_t *quotient_mask;
  const AssignmentRunView *hash_children;
  const std::uint32_t *child_router;
  const AssignmentRunView *group_children;
  std::uint16_t child_count;
  std::uint8_t constant_op;
  std::uint8_t mixed;
  std::uint8_t hashed;
  std::uint8_t grouped;
};

struct FlatAssignmentLeafView {
  const std::uint32_t *keys;
  const std::uint32_t *values;
  const std::uint32_t *offsets;
  std::uint32_t count;
  std::uint32_t rank_base;
  std::uint8_t constant_op;
};

__global__ void flat_assignment_count_kernel(
    const FlatAssignmentLeafView *leaves, int leaf_count,
    std::uint32_t *counts) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kEpochQuotients)
    return;
  if (q == kEpochQuotients) {
    counts[q] = 0u;
    return;
  }
  std::uint32_t total = 0u;
  for (int leaf = 0; leaf < leaf_count; ++leaf)
    total += leaves[leaf].offsets[q + 1u] -
             leaves[leaf].offsets[q];
  counts[q] = total;
}

__global__ void flat_assignment_base_kernel(
    const FlatAssignmentLeafView *leaves, int leaf_count,
    const std::uint32_t *quotient_offsets,
    std::uint32_t *leaf_bases) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kEpochQuotients)
    return;
  std::uint32_t cursor = quotient_offsets[q];
  for (int leaf = 0; leaf < leaf_count; ++leaf) {
    leaf_bases[static_cast<std::size_t>(leaf) *
                   kEpochQuotients +
               q] = cursor;
    cursor += leaves[leaf].offsets[q + 1u] -
              leaves[leaf].offsets[q];
  }
}

__global__ void flat_assignment_gather_kernel(
    const FlatAssignmentLeafView *leaves, int leaf_count,
    std::uint32_t max_count, const std::uint32_t *leaf_bases,
    std::uint32_t *out_keys, std::uint64_t *out_payload) {
  const std::uint32_t leaf = blockIdx.y;
  const std::uint32_t position =
      blockIdx.x * blockDim.x + threadIdx.x;
  if (leaf >= static_cast<std::uint32_t>(leaf_count) ||
      position >= max_count)
    return;
  const FlatAssignmentLeafView view = leaves[leaf];
  if (position >= view.count)
    return;
  const std::uint32_t key = view.keys[position];
  const std::uint32_t q = key >> kEpochQuotientBits;
  const std::uint32_t output =
      leaf_bases[static_cast<std::size_t>(leaf) *
                      kEpochQuotients +
                  q] +
      position - view.offsets[q];
  const std::uint32_t value =
      view.constant_op != 0u ? view.values[position] : 0u;
  out_keys[output] = key;
  out_payload[output] =
      (static_cast<std::uint64_t>(value) << 32) |
      static_cast<std::uint64_t>(view.constant_op != 0u);
}

__device__ inline void assignment_bounds(const AssignmentRunView &run,
                                         std::uint32_t q,
                                         std::uint32_t *begin,
                                         std::uint32_t *end) {
  *begin = run.offsets[q];
  *end = run.offsets[q + 1u];
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
    return run.quotient_meta[q] & kHashCountMask;
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
    const std::uint32_t meta = run.quotient_meta[q];
    const std::uint32_t count = meta & kHashCountMask;
    if (count == 0u)
      return false;
    if (meta & kHashScanFlag) {
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
    const std::uint32_t low = key & 0xffffu;
    const std::uint32_t slots =
        run.offsets[q + 1u] - run.offsets[q];
    const std::uint32_t *table = run.keys + run.offsets[q];
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
    const std::uint32_t meta = run.quotient_meta[q];
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
    const std::uint32_t slots =
        run.offsets[q + 1u] - run.offsets[q];
    const std::uint32_t *table = run.keys + run.offsets[q];
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

struct LookupOwnerLeafView {
  const std::uint32_t *keys;
  const std::uint32_t *values;
  const std::uint32_t *offsets;
  std::uint64_t *route_mask;
  std::uint32_t count;
  std::uint8_t constant_op;
};

struct PinnedHostState {
  AssignmentRunView views[kRunCapacity];
  AssignmentRunView scratch_views[kRunCapacity];
  const std::uint64_t *route_masks[kRawFoldWidth];
  LookupPublication lookup;
  LookupOwnerLeafView owner_leaves[kLookupLeafCapacity];
  std::uint32_t qvrf_base_block_count;
  std::uint32_t qvrf_visible_count;
  std::uint64_t qvrf_report_count;
  std::uint32_t resolved_count;
  std::uint32_t successor_miss_count;
  std::uint32_t gathered_count;
  std::uint32_t resolve_class_counts[4];
  std::uint32_t fused_winner_count;
  std::uint32_t hash_heavy_count;
  std::uint32_t hash_capacity;
};

__host__ __device__ inline std::uint32_t
lookup_owner_hash(std::uint32_t key) {
  key ^= key >> 16;
  key *= 0x7feb352du;
  key ^= key >> 15;
  key *= 0x846ca68bu;
  return key ^ (key >> 16);
}

__device__ __forceinline__ std::uint32_t lookup_owner_start(
    std::uint32_t key, std::uint32_t table_mask) {
  const std::uint32_t table_bits = 32u - __clz(table_mask);
  if (table_bits <= 16u)
    return lookup_owner_hash(key) & table_mask;
  const std::uint32_t local_bits = table_bits - 16u;
  const std::uint32_t local_mask = (1u << local_bits) - 1u;
  std::uint32_t low = key & 0xffffu;
  low ^= low >> 7;
  low *= 0x9e3779b1u;
  low ^= low >> 11;
  return ((key >> 16) << local_bits) | (low & local_mask);
}

__device__ __forceinline__ std::uint32_t lookup_owner_next(
    std::uint32_t slot, std::uint32_t table_mask) {
  const std::uint32_t table_bits = 32u - __clz(table_mask);
  if (table_bits <= 16u)
    return (slot + 1u) & table_mask;
  const std::uint32_t local_bits = table_bits - 16u;
  const std::uint32_t local_mask = (1u << local_bits) - 1u;
  return (slot & ~local_mask) | ((slot + 1u) & local_mask);
}


__device__ __forceinline__ bool lookup_owner_find(
    const unsigned long long *slots, std::uint32_t table_mask,
    std::uint32_t key, std::uint32_t *token) {
  std::uint32_t slot = lookup_owner_start(key, table_mask);
  for (std::uint32_t probe = 0; probe < kLookupOwnerMaxProbe; ++probe) {
    const unsigned long long entry = slots[slot];
    if (entry == 0u) {
      *token = 0u;
      return true;
    }
    if (static_cast<std::uint32_t>(entry >> 32) == key) {
      *token = static_cast<std::uint32_t>(entry);
      return true;
    }
    slot = lookup_owner_next(slot, table_mask);
  }
  return false;
}

__device__ __forceinline__ void lookup_owner_legacy(
    const AssignmentRunView *runs, int run_count, BaseRunView base,
    std::uint32_t key, std::size_t query, std::uint32_t *out_values,
    std::uint8_t *out_found) {
  for (int r = run_count - 1; r >= 0; --r) {
    std::uint32_t value = 0u;
    bool live = false;
    if (!assignment_find(runs[r], key, &value, &live))
      continue;
    out_values[query] = live ? value : kEmptyKey;
    if (out_found)
      out_found[query] = live ? 1u : 0u;
    return;
  }
  std::uint32_t value = kEmptyKey;
  const bool found = base_find_value(base, key, &value);
  out_values[query] = found ? value : kEmptyKey;
  if (out_found)
    out_found[query] = found ? 1u : 0u;
}

// Applies leaves with one cooperative warp per quotient.
__global__ void lookup_owner_apply_kernel(
    const LookupOwnerLeafView *leaves, std::uint32_t first_leaf,
    std::uint32_t leaf_count, std::uint32_t position_bits,
    unsigned long long *slots, std::uint32_t table_mask,
    std::uint32_t *overflow) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned warp = threadIdx.x >> 5u;
  const unsigned warps_per_block = blockDim.x >> 5u;
  const std::uint32_t quotient =
      blockIdx.x * warps_per_block + warp;
  if (quotient >= kEpochQuotients || *overflow != 0u)
    return;
  const unsigned lane_bit = 1u << lane;
  for (std::uint32_t offset = 0; offset < leaf_count; ++offset) {
    const std::uint32_t leaf_id = first_leaf + offset;
    const LookupOwnerLeafView leaf = leaves[leaf_id];
    const std::uint32_t begin = leaf.offsets[quotient];
    const std::uint32_t end = leaf.offsets[quotient + 1u];
    std::uint64_t route_mask = 0u;
    for (std::uint32_t base = begin; base < end; base += 32u) {
      const std::uint32_t position = base + lane;
      const unsigned full = __activemask();
      unsigned active = __ballot_sync(full, position < end);
      std::uint32_t key = 0u;
      std::uint32_t token = 0u;
      std::uint32_t slot = 0u;
      std::uint32_t probes = 0u;
      if (position < end) {
        key = leaf.keys[position];
        token = ((leaf_id + 1u) << position_bits) | position;
        slot = lookup_owner_start(key, table_mask);
      }
      std::uint64_t route_bits =
          position < end
              ? std::uint64_t{1}
                    << ((key & 0xffffu) >>
                        (kEpochQuotientBits - kHashRouteBits))
              : 0u;
      for (int delta = 16; delta != 0; delta >>= 1)
        route_bits |= __shfl_down_sync(full, route_bits, delta);
      if (lane == 0u)
        route_mask |= route_bits;
      while ((active & lane_bit) != 0u) {
        const unsigned group = __match_any_sync(active, slot);
        const int leader = __ffs(group) - 1;
        unsigned long long entry = 0u;
        if (static_cast<int>(lane) == leader)
          entry = slots[slot];
        entry = __shfl_sync(group, entry, leader);
        const std::uint32_t entry_key =
            static_cast<std::uint32_t>(entry >> 32);
        const std::uint32_t selected_key =
            __shfl_sync(group, key, leader);
        const bool claim =
            entry == 0u ? key == selected_key : entry_key == key;
        const unsigned claim_mask =
            __ballot_sync(active, claim) & group;
        if (claim_mask != 0u) {
          const int writer = 31 - __clz(claim_mask);
          if (static_cast<int>(lane) == writer) {
            slots[slot] =
                (static_cast<unsigned long long>(key) << 32) | token;
          }
        }
        const bool done = (claim_mask & lane_bit) != 0u;
        __syncwarp(active);
        if (!done) {
          slot = lookup_owner_next(slot, table_mask);
          ++probes;
        }
        const unsigned failed = __ballot_sync(
            active, !done && probes >= kLookupOwnerMaxProbe);
        if (failed != 0u) {
          if (lane == static_cast<unsigned>(__ffs(failed) - 1))
            atomicExch(overflow, 1u);
          return;
        }
        active = __ballot_sync(active, !done);
      }
    }
    if (lane == 0u)
      leaf.route_mask[quotient] = route_mask;
  }
}


__global__ void lookup_owner_kernel(
    const unsigned long long *slots, std::uint32_t table_mask,
    const std::uint32_t *overflow, const LookupOwnerLeafView *leaves,
    std::uint32_t leaf_count, std::uint32_t position_bits,
    std::uint32_t position_mask, const AssignmentRunView *runs,
    int run_count, BaseRunView base, const std::uint32_t *queries,
    std::size_t n, std::uint32_t *out_values,
    std::uint8_t *out_found) {
  const std::size_t query =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (query >= n)
    return;
  const std::uint32_t key = queries[query];
  std::uint32_t token = 0u;
  if (*overflow != 0u ||
      !lookup_owner_find(slots, table_mask, key, &token)) {
    lookup_owner_legacy(runs, run_count, base, key, query,
                        out_values, out_found);
    return;
  }
  if (token != 0u) {
    const std::uint32_t encoded_leaf = token >> position_bits;
    const std::uint32_t position = token & position_mask;
    if (encoded_leaf == 0u || encoded_leaf > leaf_count) {
      lookup_owner_legacy(runs, run_count, base, key, query,
                          out_values, out_found);
      return;
    }
    const LookupOwnerLeafView leaf = leaves[encoded_leaf - 1u];
    if (position >= leaf.count || leaf.keys[position] != key) {
      lookup_owner_legacy(runs, run_count, base, key, query,
                          out_values, out_found);
      return;
    }
    const bool live = leaf.constant_op != 0u;
    out_values[query] = live ? leaf.values[position] : kEmptyKey;
    if (out_found)
      out_found[query] = live ? 1u : 0u;
    return;
  }
  std::uint32_t value = kEmptyKey;
  const bool found = base_find_value(base, key, &value);
  out_values[query] = found ? value : kEmptyKey;
  if (out_found)
    out_found[query] = found ? 1u : 0u;
}

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

// Scan newest first, then fall through to BaseRun.
__global__ void temporal_lookup_kernel(const AssignmentRunView *runs,
                                       int run_count, BaseRunView base,
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
  if (base_find_value(base, key, &value)) {
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
    const AssignmentRunView *runs, int run_count, BaseRunView base,
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
      live = base_find_value(base, key, &value) ? 1 : 0;
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

// Pack the visible BaseRun.
__global__ void resolve_pack_base_kernel(
    BaseRunView base, std::uint32_t *out_keys,
    std::uint64_t *out_payload) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= base.base.count)
    return;
  out_keys[i] = base.base.keys[i];
  out_payload[i] =
      (static_cast<std::uint64_t>(base.base.values[i]) << 32) | 1u;
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

struct QvrfKeyToCell {
  __host__ __device__ std::uint32_t operator()(std::uint32_t key) const {
    return key >> kBaseRank23Shift;
  }
};

__global__ void qvrf_base_block_count_kernel(const std::uint32_t *rank23,
                                              std::uint32_t *counts) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kEpochQuotients)
    return;
  const std::uint32_t begin = rank23[q * kBaseBinsPerQuotient];
  const std::uint32_t end = rank23[(q + 1u) * kBaseBinsPerQuotient];
  counts[q] = (end - begin + kQvrfBaseLeafRecords - 1u) /
              kQvrfBaseLeafRecords;
}

__global__ void qvrf_base_block_layout_kernel(
    const std::uint32_t *rank23, const std::uint32_t *block_offsets,
    std::uint32_t *block_begins, std::uint16_t *block_counts) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kEpochQuotients)
    return;
  const std::uint32_t q_begin = rank23[q * kBaseBinsPerQuotient];
  const std::uint32_t q_end = rank23[(q + 1u) * kBaseBinsPerQuotient];
  const std::uint32_t first = block_offsets[q];
  const std::uint32_t blocks = block_offsets[q + 1u] - first;
  for (std::uint32_t local = 0; local < blocks; ++local) {
    const std::uint32_t begin = q_begin + local * kQvrfBaseLeafRecords;
    const std::uint32_t count =
        min(kQvrfBaseLeafRecords, q_end - begin);
    block_begins[first + local] = begin;
    block_counts[first + local] = static_cast<std::uint16_t>(count);
  }
}

__global__ __launch_bounds__(kQvrfBaseLeafRecords)
void qvrf_base_block_summary_kernel(
    SortedRunView base, const std::uint32_t *block_begins,
    const std::uint16_t *block_counts, const std::uint32_t *block_offsets,
    QvrfSummary *summaries) {
  const std::uint32_t block_id = blockIdx.x;
  if (block_id >= block_offsets[kEpochQuotients])
    return;
  __shared__ QvrfSummary state[kQvrfBaseLeafRecords];
  const std::uint32_t begin = block_begins[block_id];
  const std::uint32_t count = block_counts[block_id];
  const std::uint32_t lane = threadIdx.x;
  state[lane] = lane < count
                    ? qvrf_lift(base.keys[begin + lane],
                                base.values[begin + lane])
                             : qvrf_identity();
  __syncthreads();
  for (std::uint32_t stride = 1u; stride < kQvrfBaseLeafRecords;
       stride <<= 1u) {
    const std::uint32_t left = lane * (stride << 1u);
    if (left + stride < kQvrfBaseLeafRecords)
      state[left] = qvrf_combine(state[left], state[left + stride]);
    __syncthreads();
  }
  if (lane == 0u)
    summaries[block_id] = state[0];
}

__global__ void qvrf_base_q_summary_kernel(
    const std::uint32_t *block_offsets,
    const QvrfSummary *block_summaries, QvrfSummary *q_summaries) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kEpochQuotients)
    return;
  QvrfSummary state = qvrf_identity();
  for (std::uint32_t i = block_offsets[q]; i < block_offsets[q + 1u]; ++i)
    state = qvrf_combine(state, block_summaries[i]);
  q_summaries[q] = state;
}

__global__ void qvrf_reduce16_kernel(const QvrfSummary *input,
                                     QvrfSummary *output,
                                     std::uint32_t output_count) {
  const std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= output_count)
    return;
  QvrfSummary summary = qvrf_identity();
  const std::uint32_t begin = item * kQvrfTopFanout;
  for (std::uint32_t i = 0u; i < kQvrfTopFanout; ++i)
    summary = qvrf_combine(summary, input[begin + i]);
  output[item] = summary;
}

__device__ inline QvrfSummary qvrf_base_rank_summary(
    const QvrfView &view, std::uint32_t q, std::uint32_t begin,
    std::uint32_t end) {
  if (begin >= end)
    return qvrf_identity();
  const std::uint32_t q_begin =
      view.base.rank23[q * kBaseBinsPerQuotient];
  const std::uint32_t first_local =
      (begin - q_begin) / kQvrfBaseLeafRecords;
  const std::uint32_t last_local =
      (end - 1u - q_begin) / kQvrfBaseLeafRecords;
  QvrfSummary state = qvrf_identity();
  if (first_local == last_local) {
    for (std::uint32_t i = begin; i < end; ++i)
      state = qvrf_combine(
          state, qvrf_lift(view.base.keys[i], view.base.values[i]));
    return state;
  }
  const std::uint32_t first_end =
      q_begin + (first_local + 1u) * kQvrfBaseLeafRecords;
  for (std::uint32_t i = begin; i < first_end; ++i)
    state = qvrf_combine(
        state, qvrf_lift(view.base.keys[i], view.base.values[i]));
  const std::uint32_t q_block = view.base_q_block_offsets[q];
  for (std::uint32_t local = first_local + 1u; local < last_local; ++local)
    state = qvrf_combine(state,
                         view.base_block_summaries[q_block + local]);
  const std::uint32_t last_begin =
      q_begin + last_local * kQvrfBaseLeafRecords;
  for (std::uint32_t i = last_begin; i < end; ++i)
    state = qvrf_combine(
        state, qvrf_lift(view.base.keys[i], view.base.values[i]));
  return state;
}

__device__ inline void qvrf_resolved_cell_bounds(
    RunView resolved, std::uint32_t cell, std::uint32_t *begin,
    std::uint32_t *end) {
  const std::uint32_t q = cell >>
      (kBaseRank23Bits - kEpochQuotientBits);
  const std::uint32_t q_begin = resolved.quotient_off[q];
  const std::uint32_t q_end = resolved.quotient_off[q + 1u];
  const std::uint32_t key_begin = cell << kBaseRank23Shift;
  *begin = q_begin + static_cast<std::uint32_t>(lower_bound_u32(
      resolved.keys + q_begin, q_end - q_begin, key_begin));
  if ((cell & (kBaseBinsPerQuotient - 1u)) ==
      kBaseBinsPerQuotient - 1u) {
    *end = q_end;
  } else {
    const std::uint32_t key_end = (cell + 1u) << kBaseRank23Shift;
    *end = q_begin + static_cast<std::uint32_t>(lower_bound_u32(
        resolved.keys + q_begin, q_end - q_begin, key_end));
  }
}

__global__ void qvrf_cell_summary_kernel(QvrfView view,
                                          QvrfSummary *summaries) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= view.cell_count)
    return;
  const std::uint32_t cell = view.cell_ids[i];
  std::uint32_t bi = view.base.rank23[cell];
  const std::uint32_t be = view.base.rank23[cell + 1u];
  std::uint32_t ci = 0u, ce = 0u;
  qvrf_resolved_cell_bounds(view.resolved, cell, &ci, &ce);
  QvrfSummary summary = qvrf_identity();
  while (bi < be || ci < ce) {
    std::uint32_t key = 0u, value = 0u;
    bool live = false;
    if (ci < ce &&
        (bi == be || view.resolved.keys[ci] < view.base.keys[bi])) {
      key = view.resolved.keys[ci];
      value = view.resolved_values[ci];
      live = view.resolved_counts[ci] > 0;
      ++ci;
    } else if (bi < be &&
               (ci == ce || view.base.keys[bi] < view.resolved.keys[ci])) {
      key = view.base.keys[bi];
      value = view.base.values[bi];
      live = true;
      ++bi;
    } else {
      key = view.base.keys[bi];
      if (view.resolved_counts[ci] >= 0) {
        value = view.resolved_counts[ci] > 0
                    ? view.resolved_values[ci]
                    : view.base.values[bi] + view.resolved_values[ci];
        live = true;
      }
      ++bi;
      ++ci;
    }
    if (live)
      summary = qvrf_combine(summary, qvrf_lift(key, value));
  }
  summaries[i] = summary;
}

__global__ void qvrf_cell_visible_count_kernel(QvrfView view,
                                                std::uint32_t *counts) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= view.cell_count)
    return;
  const std::uint32_t cell = view.cell_ids[i];
  std::uint32_t bi = view.base.rank23[cell];
  const std::uint32_t be = view.base.rank23[cell + 1u];
  std::uint32_t ci = 0u, ce = 0u;
  qvrf_resolved_cell_bounds(view.resolved, cell, &ci, &ce);
  std::uint32_t count = 0u;
  while (bi < be || ci < ce) {
    if (ci < ce &&
        (bi == be || view.resolved.keys[ci] < view.base.keys[bi])) {
      count += view.resolved_counts[ci] > 0 ? 1u : 0u;
      ++ci;
    } else if (bi < be &&
               (ci == ce || view.base.keys[bi] < view.resolved.keys[ci])) {
      ++count;
      ++bi;
    } else {
      count += view.resolved_counts[ci] < 0 ? 0u : 1u;
      ++bi;
      ++ci;
    }
  }
  counts[i] = count;
}

__global__ void qvrf_finish_offsets_kernel(const std::uint32_t *counts,
                                            std::uint32_t *offsets,
                                            std::uint32_t count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  offsets[count] = count == 0u ? 0u : offsets[count - 1u] + counts[count - 1u];
}

__global__ void qvrf_cell_emit_kernel(QvrfView view,
                                       const std::uint32_t *offsets,
                                       std::uint32_t *out_keys,
                                       std::uint32_t *out_values,
                                       QvrfSummary *summaries) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= view.cell_count)
    return;
  const std::uint32_t cell = view.cell_ids[i];
  std::uint32_t bi = view.base.rank23[cell];
  const std::uint32_t be = view.base.rank23[cell + 1u];
  std::uint32_t ci = 0u, ce = 0u;
  qvrf_resolved_cell_bounds(view.resolved, cell, &ci, &ce);
  std::uint32_t output = offsets[i];
  QvrfSummary summary = qvrf_identity();
  while (bi < be || ci < ce) {
    std::uint32_t key = 0u, value = 0u;
    bool live = false;
    if (ci < ce &&
        (bi == be || view.resolved.keys[ci] < view.base.keys[bi])) {
      key = view.resolved.keys[ci];
      value = view.resolved_values[ci];
      live = view.resolved_counts[ci] > 0;
      ++ci;
    } else if (bi < be &&
               (ci == ce || view.base.keys[bi] < view.resolved.keys[ci])) {
      key = view.base.keys[bi];
      value = view.base.values[bi];
      live = true;
      ++bi;
    } else {
      key = view.base.keys[bi];
      if (view.resolved_counts[ci] >= 0) {
        value = view.resolved_counts[ci] > 0
                    ? view.resolved_values[ci]
                    : view.base.values[bi] + view.resolved_values[ci];
        live = true;
      }
      ++bi;
      ++ci;
    }
    if (live) {
      out_keys[output] = key;
      out_values[output] = value;
      summary = qvrf_combine(summary, qvrf_lift(key, value));
      ++output;
    }
  }
  summaries[i] = summary;
}

__global__ void qvrf_q_cell_offsets_kernel(const std::uint32_t *cell_ids,
                                            std::uint32_t cell_count,
                                            std::uint32_t *offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kEpochQuotients)
    return;
  if (cell_count == 0u) {
    offsets[q] = 0u;
    return;
  }
  const std::uint32_t target = q * kBaseBinsPerQuotient;
  offsets[q] = static_cast<std::uint32_t>(
      lower_bound_u32(cell_ids, cell_count, target));
}

__global__ void qvrf_dirty_q_summary_kernel(QvrfView view,
                                             QvrfSummary *q_summaries) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kEpochQuotients)
    return;
  const std::uint32_t first = view.q_cell_offsets[q];
  const std::uint32_t last = view.q_cell_offsets[q + 1u];
  if (first == last)
    return;
  std::uint32_t cursor =
      view.base.rank23[q * kBaseBinsPerQuotient];
  const std::uint32_t q_end =
      view.base.rank23[(q + 1u) * kBaseBinsPerQuotient];
  QvrfSummary summary = qvrf_identity();
  for (std::uint32_t i = first; i < last; ++i) {
    const std::uint32_t cell = view.cell_ids[i];
    const std::uint32_t cell_begin = view.base.rank23[cell];
    const std::uint32_t cell_end = view.base.rank23[cell + 1u];
    summary = qvrf_combine(
        summary, qvrf_base_rank_summary(view, q, cursor, cell_begin));
    summary = qvrf_combine(summary, view.cell_summaries[i]);
    cursor = cell_end;
  }
  summary = qvrf_combine(
      summary, qvrf_base_rank_summary(view, q, cursor, q_end));
  q_summaries[q] = summary;
}

__device__ inline QvrfSummary qvrf_page_range_summary(
    const QvrfView &view, std::uint32_t page, std::uint32_t lo,
    std::uint32_t hi) {
  const std::uint32_t cell = view.cell_ids[page];
  const std::uint32_t cell_lo = cell << kBaseRank23Shift;
  const std::uint32_t cell_hi =
      cell_lo | ((1u << kBaseRank23Shift) - 1u);
  if (lo <= cell_lo && hi >= cell_hi)
    return view.cell_summaries[page];
  if (view.materialized_cells != 0u) {
    const std::uint32_t begin = view.cell_offsets[page];
    const std::uint32_t end = view.cell_offsets[page + 1u];
    if (begin == end)
      return qvrf_identity();
    const std::uint32_t lower = begin + static_cast<std::uint32_t>(
        lower_bound_u32(view.cell_keys + begin, end - begin, lo));
    const std::uint32_t upper = begin + static_cast<std::uint32_t>(
        upper_bound_u32(view.cell_keys + begin, end - begin, hi));
    QvrfSummary summary = qvrf_identity();
    for (std::uint32_t i = lower; i < upper; ++i)
      summary = qvrf_combine(
          summary, qvrf_lift(view.cell_keys[i], view.cell_values[i]));
    return summary;
  }

  std::uint32_t bi = view.base.rank23[cell];
  const std::uint32_t be = view.base.rank23[cell + 1u];
  std::uint32_t ci = 0u, ce = 0u;
  qvrf_resolved_cell_bounds(view.resolved, cell, &ci, &ce);
  QvrfSummary summary = qvrf_identity();
  while (bi < be || ci < ce) {
    std::uint32_t key = 0u, value = 0u;
    bool live = false;
    if (ci < ce &&
        (bi == be || view.resolved.keys[ci] < view.base.keys[bi])) {
      key = view.resolved.keys[ci];
      value = view.resolved_values[ci];
      live = view.resolved_counts[ci] > 0;
      ++ci;
    } else if (bi < be &&
               (ci == ce || view.base.keys[bi] < view.resolved.keys[ci])) {
      key = view.base.keys[bi];
      value = view.base.values[bi];
      live = true;
      ++bi;
    } else {
      key = view.base.keys[bi];
      if (view.resolved_counts[ci] >= 0) {
        value = view.resolved_counts[ci] > 0
                    ? view.resolved_values[ci]
                    : view.base.values[bi] + view.resolved_values[ci];
        live = true;
      }
      ++bi;
      ++ci;
    }
    if (live && key >= lo && key <= hi)
      summary = qvrf_combine(summary, qvrf_lift(key, value));
  }
  return summary;
}

__device__ inline QvrfSummary qvrf_q_range_summary(
    const QvrfView &view, std::uint32_t q, std::uint32_t lo,
    std::uint32_t hi) {
  std::size_t lower_size = 0u, upper_size = 0u;
  sorted_range_ranks(view.base, lo, hi, &lower_size, &upper_size);
  std::uint32_t cursor = static_cast<std::uint32_t>(lower_size);
  const std::uint32_t upper = static_cast<std::uint32_t>(upper_size);
  const std::uint32_t first = view.q_cell_offsets[q];
  const std::uint32_t last = view.q_cell_offsets[q + 1u];
  const std::uint32_t lo_cell = lo >> kBaseRank23Shift;
  const std::uint32_t hi_cell = hi >> kBaseRank23Shift;
  std::uint32_t page = first;
  if (first < last)
    page += static_cast<std::uint32_t>(lower_bound_u32(
        view.cell_ids + first, last - first, lo_cell));
  QvrfSummary summary = qvrf_identity();
  for (; page < last && view.cell_ids[page] <= hi_cell; ++page) {
    const std::uint32_t cell = view.cell_ids[page];
    const std::uint32_t base_begin = view.base.rank23[cell];
    const std::uint32_t base_end = view.base.rank23[cell + 1u];
    if (cursor < base_begin)
      summary = qvrf_combine(
          summary, qvrf_base_rank_summary(view, q, cursor,
                                           min(base_begin, upper)));
    summary = qvrf_combine(summary,
                           qvrf_page_range_summary(view, page, lo, hi));
    if (base_end > cursor)
      cursor = base_end;
  }
  if (cursor < upper)
    summary = qvrf_combine(
        summary, qvrf_base_rank_summary(view, q, cursor, upper));
  return summary;
}

__device__ inline QvrfSummary qvrf_top_range_summary(
    const QvrfView &view, std::uint32_t begin, std::uint32_t end) {
  QvrfSummary left = qvrf_identity();
  QvrfSummary right = qvrf_identity();
  const QvrfSummary *level = view.q_summaries;
  for (std::uint32_t depth = 0u; depth < 5u && begin < end; ++depth) {
    while (begin < end && (begin & (kQvrfTopFanout - 1u)) != 0u)
      left = qvrf_combine(left, level[begin++]);
    while (begin < end && (end & (kQvrfTopFanout - 1u)) != 0u)
      right = qvrf_combine(level[--end], right);
    if (begin == end)
      break;
    begin /= kQvrfTopFanout;
    end /= kQvrfTopFanout;
    level = depth == 0u ? view.top_l1
          : depth == 1u ? view.top_l2
          : depth == 2u ? view.top_l3
                        : view.top_l4;
  }
  return qvrf_combine(left, right);
}

__device__ inline QvrfSummary qvrf_range_summary(
    const QvrfView &view, std::uint32_t lower, std::uint32_t upper) {
  QvrfSummary summary = qvrf_identity();
  if (lower > upper)
    return summary;
  const std::uint32_t qlo = lower >> kEpochQuotientBits;
  const std::uint32_t qhi = upper >> kEpochQuotientBits;
  if (qlo == qhi)
    return qvrf_q_range_summary(view, qlo, lower, upper);
  const std::uint32_t left_hi =
      (qlo << kEpochQuotientBits) | 0xffffu;
  const std::uint32_t right_lo = qhi << kEpochQuotientBits;
  summary = qvrf_q_range_summary(view, qlo, lower, left_hi);
  if (qlo + 1u < qhi)
    summary = qvrf_combine(
        summary, qvrf_top_range_summary(view, qlo + 1u, qhi));
  return qvrf_combine(
      summary, qvrf_q_range_summary(view, qhi, right_lo, upper));
}

__global__ void qvrf_range_kernel(
    QvrfView view, const std::uint32_t *lo, const std::uint32_t *hi,
    std::size_t query_count, std::uint32_t *out_sums,
    std::uint32_t *out_counts, std::uint32_t *out_mins,
    std::uint32_t *out_maxs) {
  const std::size_t query =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (query >= query_count)
    return;
  const QvrfSummary summary =
      qvrf_range_summary(view, lo[query], hi[query]);
  if (out_sums)
    out_sums[query] = summary.sum;
  if (out_counts)
    out_counts[query] = summary.count;
  if (out_mins)
    out_mins[query] = summary.minimum;
  if (out_maxs)
    out_maxs[query] = summary.maximum;
}

__global__ void qvrf_report_count_kernel(
    QvrfView view, const std::uint32_t *lo, const std::uint32_t *hi,
    std::size_t query_count, std::uint64_t *counts) {
  const std::size_t query =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (query < query_count)
    counts[query] = qvrf_range_summary(view, lo[query], hi[query]).count;
}

__global__ void qvrf_finish_report_offsets_kernel(
    const std::uint64_t *counts, std::uint64_t *offsets,
    std::size_t count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  offsets[count] = count == 0u ? 0u : offsets[count - 1u] + counts[count - 1u];
}

__global__ void qvrf_report_kernel(
    QvrfView view, const std::uint32_t *lo, const std::uint32_t *hi,
    std::size_t query_count, const std::uint64_t *offsets,
    std::uint32_t *out_keys, std::uint32_t *out_values) {
  const std::size_t query =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (query >= query_count || lo[query] > hi[query])
    return;
  const std::uint32_t lower_key = lo[query];
  const std::uint32_t upper_key = hi[query];
  std::size_t lower_size = 0u, upper_size = 0u;
  sorted_range_ranks(view.base, lower_key, upper_key,
                     &lower_size, &upper_size);
  std::uint32_t cursor = static_cast<std::uint32_t>(lower_size);
  const std::uint32_t base_upper = static_cast<std::uint32_t>(upper_size);
  std::uint64_t output = offsets[query];

  const std::uint32_t qlo = lower_key >> kEpochQuotientBits;
  const std::uint32_t qhi = upper_key >> kEpochQuotientBits;
  const std::uint32_t lo_cell = lower_key >> kBaseRank23Shift;
  const std::uint32_t hi_cell = upper_key >> kBaseRank23Shift;
  const std::uint32_t first = view.q_cell_offsets[qlo];
  const std::uint32_t last = view.q_cell_offsets[qhi + 1u];
  std::uint32_t page = first;
  if (first < last)
    page += static_cast<std::uint32_t>(lower_bound_u32(
        view.cell_ids + first, last - first, lo_cell));

  for (; page < last && view.cell_ids[page] <= hi_cell; ++page) {
    const std::uint32_t cell = view.cell_ids[page];
    const std::uint32_t cell_begin = view.base.rank23[cell];
    const std::uint32_t cell_end = view.base.rank23[cell + 1u];
    const std::uint32_t gap_end = min(cell_begin, base_upper);
    while (cursor < gap_end) {
      out_keys[output] = view.base.keys[cursor];
      out_values[output] = view.base.values[cursor];
      ++output;
      ++cursor;
    }

    if (view.materialized_cells != 0u) {
      const std::uint32_t begin = view.cell_offsets[page];
      const std::uint32_t end = view.cell_offsets[page + 1u];
      if (begin < end) {
        const std::uint32_t emit_begin = begin +
            static_cast<std::uint32_t>(lower_bound_u32(
                view.cell_keys + begin, end - begin, lower_key));
        const std::uint32_t emit_end = begin +
            static_cast<std::uint32_t>(upper_bound_u32(
                view.cell_keys + begin, end - begin, upper_key));
        for (std::uint32_t i = emit_begin; i < emit_end; ++i) {
          out_keys[output] = view.cell_keys[i];
          out_values[output] = view.cell_values[i];
          ++output;
        }
      }
    } else {
      std::uint32_t bi = cell_begin;
      std::uint32_t ci = 0u, ce = 0u;
      qvrf_resolved_cell_bounds(view.resolved, cell, &ci, &ce);
      while (bi < cell_end || ci < ce) {
        std::uint32_t key = 0u, value = 0u;
        bool live = false;
        if (ci < ce &&
            (bi == cell_end || view.resolved.keys[ci] < view.base.keys[bi])) {
          key = view.resolved.keys[ci];
          value = view.resolved_values[ci];
          live = view.resolved_counts[ci] > 0;
          ++ci;
        } else if (bi < cell_end &&
                   (ci == ce || view.base.keys[bi] < view.resolved.keys[ci])) {
          key = view.base.keys[bi];
          value = view.base.values[bi];
          live = true;
          ++bi;
        } else {
          key = view.base.keys[bi];
          if (view.resolved_counts[ci] >= 0) {
            value = view.resolved_counts[ci] > 0
                        ? view.resolved_values[ci]
                        : view.base.values[bi] + view.resolved_values[ci];
            live = true;
          }
          ++bi;
          ++ci;
        }
        if (live && key >= lower_key && key <= upper_key) {
          out_keys[output] = key;
          out_values[output] = value;
          ++output;
        }
      }
    }
    if (cell_end > cursor)
      cursor = cell_end;
  }
  while (cursor < base_upper) {
    out_keys[output] = view.base.keys[cursor];
    out_values[output] = view.base.values[cursor];
    ++output;
    ++cursor;
  }
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

// Converts assignments to base corrections.
__global__ void normalize_correction_kernel(
    const std::uint32_t *keys, const std::uint64_t *assignment,
    std::size_t count, BaseRunView base,
    std::uint64_t *tagged_keys, std::uint64_t *corrections) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::uint32_t key = keys[i];
  const std::uint32_t value =
      static_cast<std::uint32_t>(assignment[i] >> 32);
  const bool insert = (assignment[i] & 1u) != 0u;
  // Baseline is the immutable BaseRun state.
  std::uint32_t base_value = 0u;
  bool base_live = false;
  if (base.base.count != 0u && base_find_value(base, key, &base_value))
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

__global__ void resolve_classify_kernel(
    const std::uint32_t *counts, std::uint32_t *lists,
    std::uint32_t *class_counts,
    std::uint32_t *winner_counts) {
  const std::uint32_t q =
      blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kEpochQuotients)
    return;
  winner_counts[q] = 0u;
  if (q == kEpochQuotients || counts[q] == 0u)
    return;
  const std::uint32_t count = counts[q];
  const int cls = count <= kResolveSmallMax
                      ? 0
                      : count <= kResolveMediumMax
                            ? 1
                            : count <= kResolveLocalMax ? 2 : 3;
  const std::uint32_t slot =
      atomicAdd(class_counts + cls, 1u);
  lists[static_cast<std::size_t>(cls) *
            kEpochQuotients +
        slot] = q;
}

template <int Capacity> struct ResolveRecordStorage {
  std::uint32_t keys[Capacity];
  std::uint64_t payload[Capacity];
};

template <int Threads, int Items, bool KeepZero,
          bool CacheBaseOffsets>
__global__ __launch_bounds__(Threads)
void quotient_local_resolve_kernel(
    const FlatAssignmentLeafView *leaves, int leaf_count,
    const std::uint32_t *quotient_list,
    const std::uint32_t *input_counts,
    const std::uint32_t *input_offsets, BaseRunView base,
    std::uint32_t *out_keys, std::uint32_t *out_values,
    std::int8_t *out_counts,
    std::uint32_t *winner_counts) {
  constexpr int capacity = Threads * Items;
  constexpr int warps = Threads / 32;
  using Sort = cub::BlockRadixSort<
      std::uint32_t, Threads, Items, std::uint64_t>;
  using Scan = cub::BlockScan<std::uint32_t, Threads>;
  union SharedStorage {
    typename Sort::TempStorage sort;
    typename Scan::TempStorage scan;
    ResolveRecordStorage<capacity> records;
  };
  __shared__ SharedStorage shared;
  __shared__ std::uint32_t leaf_bases[kLookupLeafCapacity + 1];
  __shared__ std::uint32_t base_offsets[
      CacheBaseOffsets ? kBaseBinsPerQuotient + 1 : 1];

  const std::uint32_t q = quotient_list[blockIdx.x];
  const std::uint32_t count = input_counts[q];
  if (threadIdx.x == 0) {
    std::uint32_t cursor = 0u;
    for (int leaf = 0; leaf < leaf_count; ++leaf) {
      leaf_bases[leaf] = cursor;
      cursor += leaves[leaf].offsets[q + 1u] -
                leaves[leaf].offsets[q];
    }
    leaf_bases[leaf_count] = cursor;
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  for (int leaf = warp; leaf < leaf_count; leaf += warps) {
    const FlatAssignmentLeafView view = leaves[leaf];
    const std::uint32_t begin = view.offsets[q];
    const std::uint32_t end = view.offsets[q + 1u];
    const std::uint32_t output = leaf_bases[leaf];
    for (std::uint32_t i = lane; i < end - begin; i += 32u) {
      const std::uint32_t position = begin + i;
      const std::uint32_t key = view.keys[position];
      const std::uint32_t value =
          view.constant_op != 0u ? view.values[position] : 0u;
      const std::uint32_t rank =
          view.rank_base + position + 1u;
      shared.records.keys[output + i] = key & 0xffffu;
      shared.records.payload[output + i] =
          (static_cast<std::uint64_t>(view.constant_op != 0u)
           << 63) |
          (static_cast<std::uint64_t>(rank) << 32) |
          value;
    }
  }
  __syncthreads();

  std::uint32_t keys[Items];
  std::uint64_t payload[Items];
  bool keep[Items];
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    const int local = threadIdx.x * Items + item;
    if (local < static_cast<int>(count)) {
      keys[item] = shared.records.keys[local];
      payload[item] = shared.records.payload[local];
    } else {
      keys[item] = 0xffffffffu;
      payload[item] = 0u;
    }
    keep[item] = false;
  }
  __syncthreads();
  Sort(shared.sort).Sort(keys, payload, 0, 16);
  __syncthreads();
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    const int local = threadIdx.x * Items + item;
    shared.records.keys[local] = keys[item];
    shared.records.payload[local] = payload[item];
  }
  __syncthreads();

  if constexpr (CacheBaseOffsets) {
    const std::size_t first_bin =
        static_cast<std::size_t>(q) *
        kBaseBinsPerQuotient;
    for (int bin = threadIdx.x;
         bin <= kBaseBinsPerQuotient;
         bin += blockDim.x) {
      base_offsets[bin] =
          base.base.count == 0u
              ? 0u
              : base.base.rank23[first_bin + bin];
    }
    __syncthreads();
  }

#pragma unroll
  for (int item = 0; item < Items; ++item) {
    const int local = threadIdx.x * Items + item;
    if (local >= static_cast<int>(count))
      continue;
    const std::uint32_t low = shared.records.keys[local];
    if (local != 0 &&
        shared.records.keys[local - 1] == low)
      continue;
    std::uint64_t newest = shared.records.payload[local];
    std::uint32_t newest_rank =
        static_cast<std::uint32_t>(
            (newest >> 32) & 0x7fffffffu);
    int next = local + 1;
    while (next < static_cast<int>(count) &&
           shared.records.keys[next] == low) {
      const std::uint64_t candidate =
          shared.records.payload[next];
      const std::uint32_t rank =
          static_cast<std::uint32_t>(
              (candidate >> 32) & 0x7fffffffu);
      if (rank > newest_rank) {
        newest = candidate;
        newest_rank = rank;
      }
      ++next;
    }

    const std::uint32_t key =
        (q << kEpochQuotientBits) | low;
    const bool insert = (newest >> 63) != 0u;
    const std::uint32_t value =
        static_cast<std::uint32_t>(newest);
    std::uint32_t base_value = 0u;
    bool base_live = false;
    if (base.base.count != 0u) {
      if constexpr (CacheBaseOffsets) {
        const std::uint32_t base_bin =
            low >> kBaseRank23Shift;
        const std::uint32_t base_begin =
            base_offsets[base_bin];
        const std::uint32_t base_end =
            base_offsets[base_bin + 1u];
        const std::uint32_t base_position =
            base_begin + static_cast<std::uint32_t>(
                lower_bound_u32(
                    base.base.keys + base_begin,
                    base_end - base_begin, key));
        base_live =
            base_position < base_end &&
            base.base.keys[base_position] == key;
        if (base_live)
          base_value = base.base.values[base_position];
      } else {
        base_live = base_find_value(
            base, key, &base_value);
      }
    }

    std::uint32_t value_delta = 0u;
    std::int8_t count_delta = 0;
    if (insert) {
      value_delta = base_live ? value - base_value : value;
      count_delta = base_live ? 0 : 1;
    } else if (base_live) {
      value_delta = 0u - base_value;
      count_delta = -1;
    }
    const std::uint64_t correction =
        corr_pack(value_delta, count_delta);
    keep[item] = KeepZero || correction != 0u;
    keys[item] = key;
    payload[item] = correction;
  }
  __syncthreads();

  std::uint32_t thread_count = 0u;
#pragma unroll
  for (int item = 0; item < Items; ++item)
    thread_count += keep[item] ? 1u : 0u;
  std::uint32_t thread_offset = 0u;
  std::uint32_t block_total = 0u;
  Scan(shared.scan).ExclusiveSum(
      thread_count, thread_offset, block_total);
  std::uint32_t local_offset = 0u;
  const std::uint32_t output_base = input_offsets[q];
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    if (!keep[item])
      continue;
    const std::uint32_t output =
        output_base + thread_offset + local_offset++;
    out_keys[output] = keys[item];
    out_values[output] =
        static_cast<std::uint32_t>(payload[item] >> 32);
    out_counts[output] = static_cast<std::int8_t>(
        static_cast<std::uint32_t>(payload[item]));
  }
  if (threadIdx.x == 0)
    winner_counts[q] = block_total;
}
__global__ void fused_resolve_pack_kernel(
    const std::uint32_t *input_offsets,
    const std::uint32_t *winner_counts,
    const std::uint32_t *winner_offsets,
    const std::uint32_t *input_keys,
    const std::uint32_t *input_values,
    const std::int8_t *input_counts,
    std::uint64_t *output_keys,
    std::uint64_t *output_payload) {
  const std::uint32_t q = blockIdx.x;
  const std::uint32_t count = winner_counts[q];
  const std::uint32_t input = input_offsets[q];
  const std::uint32_t output = winner_offsets[q];
  for (std::uint32_t i = threadIdx.x; i < count;
       i += blockDim.x) {
    output_keys[output + i] =
        (static_cast<std::uint64_t>(input_keys[input + i]) << 32) |
        1u;
    output_payload[output + i] =
        corr_pack(input_values[input + i], input_counts[input + i]);
  }
}

__global__ void fused_resolve_unpack_kernel(
    const std::uint32_t *input_offsets,
    const std::uint32_t *winner_counts,
    const std::uint32_t *winner_offsets,
    const std::uint32_t *input_keys,
    const std::uint32_t *input_values,
    const std::int8_t *input_counts,
    std::uint32_t *output_keys,
    std::uint32_t *output_values,
    std::int8_t *output_counts,
    std::uint32_t *output_offsets) {
  const std::uint32_t q = blockIdx.x;
  const std::uint32_t count = winner_counts[q];
  const std::uint32_t input = input_offsets[q];
  const std::uint32_t output = winner_offsets[q];
  for (std::uint32_t i = threadIdx.x; i < count;
       i += blockDim.x) {
    output_keys[output + i] = input_keys[input + i];
    output_values[output + i] = input_values[input + i];
    output_counts[output + i] = input_counts[input + i];
  }
  if (threadIdx.x == 0) {
    output_offsets[q] = output;
    if (q + 1u == kEpochQuotients)
      output_offsets[q + 1u] = winner_offsets[q + 1u];
  }
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
    const std::uint32_t meta = parent.quotient_meta[q];
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
    unsigned long long best, BaseRunView base,
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
  const bool found = base_find_value(base, key, &value);
  if (!found)
    value = kEmptyKey;
  out_values[query] = value;
  if (out_found)
    out_found[query] = found ? 1u : 0u;
}

// One block evaluates all chronological leaves.
__global__ void temporal_lookup_leaf_block_kernel(
    const LookupPublication *publication, int leaf_count,
    BaseRunView base, const std::uint32_t *queries,
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
    BaseRunView base, const std::uint32_t *queries,
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

template <bool WithLookup>
__global__ void raw_quotient_masks_kernel(
    const std::uint32_t *keys,
    const std::uint32_t *offsets,
    std::uint64_t *lookup_masks,
    std::uint64_t *route_masks) {
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
  std::uint64_t route_mask = 0u;
  for (std::uint32_t position = begin + lane;
       position < end; position += 32u) {
    const std::uint32_t low = keys[position] & 0xffffu;
    if constexpr (WithLookup) {
      const std::uint64_t bits = raw_quotient_mask_bits(low);
      low_mask |= static_cast<std::uint32_t>(bits);
      high_mask |= static_cast<std::uint32_t>(bits >> 32);
    }
    route_mask |= std::uint64_t{1}
                  << (low >> (kEpochQuotientBits - kHashRouteBits));
  }
  for (int delta = 16; delta != 0; delta >>= 1) {
    if constexpr (WithLookup) {
      low_mask |= __shfl_down_sync(
          0xffffffffu, low_mask, delta);
      high_mask |= __shfl_down_sync(
          0xffffffffu, high_mask, delta);
    }
    route_mask |= __shfl_down_sync(
        0xffffffffu, route_mask, delta);
  }
  if (lane == 0u) {
    if constexpr (WithLookup)
      lookup_masks[q] =
          (static_cast<std::uint64_t>(high_mask) << 32) | low_mask;
    route_masks[q] = route_mask;
  }
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

__global__ __launch_bounds__(64)
void temporal_hash_route_mask_kernel(
    const std::uint64_t *const *route_masks, int run_count,
    const std::uint32_t *quotient_counts,
    std::uint32_t *child_router) {
  const std::uint32_t q = blockIdx.x;
  const std::uint32_t child = threadIdx.x;
  const std::uint32_t lane = child & 31u;
  const std::uint32_t word = child >> 5;
  const std::size_t base =
      static_cast<std::size_t>(q) * kHashRouteBins *
      kHashRouteWordsPerBin;
  if ((quotient_counts[q] & kHashScanFlag) == 0u) {
    if (child < kHashRouteBins) {
      child_router[base + child * kHashRouteWordsPerBin] = 0u;
      child_router[base + child * kHashRouteWordsPerBin + 1u] = 0u;
    }
    return;
  }
  const std::uint64_t mask =
      child < static_cast<std::uint32_t>(run_count)
          ? route_masks[child][q]
          : 0u;
  for (int route = 0; route < kHashRouteBins; ++route) {
    const std::uint32_t candidates = __ballot_sync(
        0xffffffffu,
        (mask & (std::uint64_t{1} << route)) != 0u);
    if (lane == 0u)
      child_router[base + route * kHashRouteWordsPerBin + word] =
          candidates;
  }
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
    // Temporal hash-fold counters.
    std::uint64_t hash_fold_count = 0;
    std::uint64_t hash_fold_input_records = 0;
    std::uint64_t cold_tier_compaction_count = 0;
    std::uint64_t cold_tier_input_records = 0;
    std::uint64_t cold_tier_output_records = 0;
    std::size_t raw_runs = 0;
    std::size_t stable_levels_occupied = 0;
  };

  // Read-only snapshot; run counts reflect current state.
  // Device fold counters are copied here, outside timed updates.
  MaintenanceStats maintenance_stats() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    MaintenanceStats stats = maintenance_stats_;
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
    insert_leaf_pool_.reserve(
        (gpulsmopt_detail::kInitialInsertLeafSlabs + 1u) *
        gpulsmopt_detail::kLeafSlabRuns);
    delete_leaf_pool_.reserve(
        (gpulsmopt_detail::kInitialDeleteLeafSlabs + 1u) *
        gpulsmopt_detail::kLeafSlabRuns);
    hash_storage_pool_.reserve(
        gpulsmopt_detail::kColdArenaSlots + 1u);
    hash_child_pool_.reserve(
        gpulsmopt_detail::kColdArenaSlots + 1u);
    for (int slot = 0;
         slot < gpulsmopt_detail::kColdArenaSlots + 1; ++slot) {
      hash_child_pool_.emplace_back();
      hash_child_pool_.back().resize(
          gpulsmopt_detail::kRawFoldWidth);
    }
    leaf_slabs_.reserve(
        gpulsmopt_detail::kLookupLeafCapacity /
        gpulsmopt_detail::kLeafSlabRuns + 4u);
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void **>(&host_state_), sizeof(*host_state_)));
    const cudaError_t event_error =
        cudaEventCreateWithFlags(&stream_handoff_, cudaEventDisableTiming);
    if (event_error != cudaSuccess) {
      cudaFreeHost(host_state_);
      host_state_ = nullptr;
      CUDA_CHECK(event_error);
    }
    const cudaError_t stream_error =
        cudaStreamCreateWithFlags(&leaf_alloc_stream_,
                                  cudaStreamNonBlocking);
    if (stream_error != cudaSuccess) {
      cudaEventDestroy(stream_handoff_);
      stream_handoff_ = nullptr;
      cudaFreeHost(host_state_);
      host_state_ = nullptr;
      CUDA_CHECK(stream_error);
    }
    const cudaError_t leaf_event_error =
        cudaEventCreateWithFlags(&leaf_alloc_ready_,
                                 cudaEventDisableTiming);
    if (leaf_event_error != cudaSuccess) {
      cudaStreamDestroy(leaf_alloc_stream_);
      leaf_alloc_stream_ = nullptr;
      cudaEventDestroy(stream_handoff_);
      stream_handoff_ = nullptr;
      cudaFreeHost(host_state_);
      host_state_ = nullptr;
      CUDA_CHECK(leaf_event_error);
    }
    const cudaError_t device_error =
        cudaGetDevice(&leaf_alloc_device_);
    if (device_error != cudaSuccess) {
      cudaEventDestroy(leaf_alloc_ready_);
      leaf_alloc_ready_ = nullptr;
      cudaStreamDestroy(leaf_alloc_stream_);
      leaf_alloc_stream_ = nullptr;
      cudaEventDestroy(stream_handoff_);
      stream_handoff_ = nullptr;
      cudaFreeHost(host_state_);
      host_state_ = nullptr;
      CUDA_CHECK(device_error);
    }
    try {
      leaf_alloc_worker_ = std::thread(
          [this] { leaf_alloc_worker_loop(); });
    } catch (...) {
      cudaEventDestroy(leaf_alloc_ready_);
      leaf_alloc_ready_ = nullptr;
      cudaStreamDestroy(leaf_alloc_stream_);
      leaf_alloc_stream_ = nullptr;
      cudaEventDestroy(stream_handoff_);
      stream_handoff_ = nullptr;
      cudaFreeHost(host_state_);
      host_state_ = nullptr;
      throw;
    }
  }

  ~GPULSMOpt() {
    {
      std::lock_guard<std::mutex> guard(leaf_alloc_mutex_);
      leaf_alloc_stop_ = true;
    }
    leaf_alloc_cv_.notify_one();
    if (leaf_alloc_worker_.joinable())
      leaf_alloc_worker_.join();
    if (leaf_alloc_stream_)
      cudaStreamSynchronize(leaf_alloc_stream_);
    if (leaf_alloc_result_)
      cudaFree(leaf_alloc_result_);
    if (leaf_alloc_ready_)
      cudaEventDestroy(leaf_alloc_ready_);
    if (leaf_alloc_stream_)
      cudaStreamDestroy(leaf_alloc_stream_);
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
    reset_parent_storage();
    reset_lookup_owner(stream);
    ++base_generation_;
    initialize_empty_base(stream);
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
    {
      std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
      order_stream_locked(stream);
#ifdef GPULSMOPT_PROFILE_INSERT
      reset_insert_prof_();
      const auto prof_t0 = std::chrono::high_resolution_clock::now();
#endif
      insert_records(batch.keys, batch.keys,
                     static_cast<std::uint8_t>(gpulsmopt_detail::kTombstone),
                     batch.count, stream, batch.sorted);
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
    if (should_use_lookup_owner(batch.count)) {
      flush_pending_views(stream);
      prepare_lookup_owner(stream);
      constexpr int block = 256;
      const int grid = static_cast<int>(
          (batch.count + block - 1u) / block);
      const int owner_run_count =
          static_cast<int>(chrono_views_.size());
      gpulsmopt_detail::lookup_owner_kernel<<<grid, block, 0, stream>>>(
          lookup_owner_slots_.data(), lookup_owner_table_mask_,
          lookup_owner_overflow_.data(), lookup_owner_leaves_.data(),
          lookup_owner_leaf_count_, lookup_owner_position_bits_,
          lookup_owner_position_mask_, assignment_views_.data(),
          owner_run_count, make_base_view(), batch.queries, batch.count,
          batch.out_values, batch.out_found);
      CUDA_CHECK(cudaGetLastError());
      return;
    }
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
            make_base_view(), batch.queries,
            batch.count, batch.out_values, batch.out_found);
      } else if (leaf_parallel) {
        constexpr int block = 256;
        const int grid = static_cast<int>(
            (batch.count + block - 1u) / block);
        gpulsmopt_detail::temporal_lookup_leaf_thread_kernel<<<
            grid, block, 0, stream>>>(
            lookup_publication_.data(), leaf_count,
            make_base_view(), batch.queries,
            batch.count, batch.out_values, batch.out_found);
      } else {
        constexpr int block = 256;
        const int grid = static_cast<int>(
            (batch.count + block - 1u) / block);
        gpulsmopt_detail::temporal_lookup_kernel<<<
            grid, block, 0, stream>>>(
            assignment_views_.data(), run_count,
            make_base_view(), batch.queries,
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
          make_base_view(), batch.queries, batch.count,
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
        assignment_views_.data(), run_count, make_base_view(),
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
    ensure_qvrf_visible(stream);
    const int block = 128;
    const int grid =
        static_cast<int>((batch.query_count + block - 1) / block);
    gpulsmopt_detail::qvrf_range_kernel<<<grid, block, 0, stream>>>(
        make_qvrf_view(), batch.lo, batch.hi, batch.query_count,
        batch.out_sums, batch.out_counts, batch.out_mins, batch.out_maxs);
    CUDA_CHECK(cudaGetLastError());
  }

  // Returns the required row count. When both output arrays are present and
  // output_capacity is sufficient, rows are emitted in sorted-key order per
  // query. Callers may pass null row arrays for a sizing-only pass.
  std::size_t range_report(const DeviceRangeReportBatch &batch,
                           cudaStream_t stream) {
    if (batch.out_offsets == nullptr)
      throw std::invalid_argument("range_report requires out_offsets");
    if (batch.query_count == 0u) {
      CUDA_CHECK(cudaMemsetAsync(batch.out_offsets, 0,
                                 sizeof(std::uint64_t), stream));
      return 0u;
    }
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    order_stream_locked(stream);
    ensure_sorted_run_cache(stream);
    ensure_qvrf_visible(stream);

    qvrf_report_counts_.resize_discard(batch.query_count);
    constexpr int block = 128;
    const int grid = static_cast<int>(
        (batch.query_count + block - 1u) / block);
    gpulsmopt_detail::qvrf_report_count_kernel<<<grid, block, 0, stream>>>(
        make_qvrf_view(), batch.lo, batch.hi, batch.query_count,
        qvrf_report_counts_.data());
    CUDA_CHECK(cudaGetLastError());
    auto policy = thrust::cuda::par.on(stream);
    thrust::exclusive_scan(policy, qvrf_report_counts_.data(),
                           qvrf_report_counts_.data() + batch.query_count,
                           batch.out_offsets);
    gpulsmopt_detail::qvrf_finish_report_offsets_kernel<<<1, 1, 0, stream>>>(
        qvrf_report_counts_.data(), batch.out_offsets, batch.query_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(
        &host_state_->qvrf_report_count,
        batch.out_offsets + batch.query_count,
        sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::uint64_t required = host_state_->qvrf_report_count;
    if (required > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("range_report output exceeds size_t");
    if (required > batch.output_capacity || batch.out_keys == nullptr ||
        batch.out_values == nullptr)
      return static_cast<std::size_t>(required);

    gpulsmopt_detail::qvrf_report_kernel<<<grid, block, 0, stream>>>(
        make_qvrf_view(), batch.lo, batch.hi, batch.query_count,
        batch.out_offsets, batch.out_keys, batch.out_values);
    CUDA_CHECK(cudaGetLastError());
    return static_cast<std::size_t>(required);
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
    reserve_lookup_owner_storage(stream);
    if (n == 0) {
      initialize_empty_base(stream);
      prepare_for_insert(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      return;
    }
    sort_direct_batch(keys, values, n, stream);
    acquire_compaction_slot();
    RunStorage &run = runs_.back();
    run.sequence_begin = 0;
    run.sequence_end = 0;
    run.stable_level = -1;
    run.operation = gpulsmopt_detail::RunOperation::Insert;
    run.assignment_class = gpulsmopt_detail::AssignmentClass::Raw;
    run.mixed = false;
    run.assignment = false;
    run.hashed = false;
    run.grouped = false;
    run.nested = false;
    run.lookup_mask_ready = false;
    run.route_mask_ready = false;
    run.group_child_count = 0;
    run.parent_slot = -1;
    run.owner_leaf_id = gpulsmopt_detail::kNoOwnerLeaf;
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
    build_sorted_run_cache(0u, stream);
    live_count_ = run.count;
    release_build_sort_storage();
    prepare_for_insert(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
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
    // BaseRun count plus pending assignment corrections.
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
        resolve_quotient_lists_, resolve_class_counts_,
        resolve_winner_counts_, resolve_winner_offsets_,
        normalize_views_, flat_normalize_views_,
        flat_normalize_bases_, norm_keys_, norm_pay_,
        resolve_winner_keys_data_, resolve_winner_values_,
        resolve_winner_deltas_, winner_workspace_arena_,
        resolved_workspace_arena_,
        cache_pay_, merge_out_keys_, merge_out_pay_, merge_flags_,
        merge_sel_keys_, merge_sel_pay_, compaction_counts_,
        compaction_offsets_,
        assignment_views_,
        lookup_publication_, lookup_owner_slots_,
        lookup_owner_leaves_, lookup_owner_overflow_,
        hash_route_mask_views_,
        direct_sort_keys_, direct_sort_values_, sort_temp_storage_,
        base_rank23_, qvrf_base_block_counts_,
        qvrf_base_q_block_offsets_, qvrf_base_block_begins_,
        qvrf_base_block_sizes_, qvrf_base_block_summaries_,
        qvrf_base_q_summaries_, qvrf_current_q_summaries_,
        qvrf_top_l1_, qvrf_top_l2_, qvrf_top_l3_, qvrf_top_l4_,
        qvrf_cell_ids_, qvrf_cell_counts_, qvrf_cell_offsets_,
        qvrf_cell_keys_, qvrf_cell_values_, qvrf_cell_summaries_,
        qvrf_q_cell_offsets_,
        qvrf_report_counts_,
        succ_miss_indices_, succ_miss_count_,
        succ_deleted_base_words_, succ_live_word_l1_, succ_live_word_l2_,
        succ_live_word_l3_, succ_positive_words_, succ_positive_l1_,
        succ_positive_l2_, succ_positive_l3_);
    total += device_bytes_all(
        resolved_.keys, resolved_.values, resolved_.count_delta,
        resolved_.quotient_off, resolved_.op_words);
    // Hash-fold scratch and pointer-only parent storage.
    total += device_bytes_all(
        hash_heavy_capacities_, hash_active_quotients_,
        hash_heavy_count_, parent_child_views_arena_);
    for (const auto &epoch : runs_) {
      total += device_bytes_all(
          epoch.keys, epoch.values, epoch.count_delta, epoch.quotient_off,
          epoch.op_words, epoch.lookup_mask, epoch.route_mask,
          epoch.hash_storage.keys, epoch.hash_storage.offsets,
          epoch.hash_storage.counts, epoch.hash_storage.child_views,
          epoch.hash_storage.child_router);
      for (const auto &child : epoch.hash_children)
        total += device_bytes_all(
            child.keys, child.values, child.quotient_off,
            child.op_words, child.lookup_mask, child.route_mask);
    }
    for (const auto &epoch : run_pool_)
      total += device_bytes_all(
          epoch.keys, epoch.values, epoch.count_delta, epoch.quotient_off,
          epoch.op_words, epoch.lookup_mask, epoch.route_mask);
    for (const auto &epoch : insert_leaf_pool_)
      total += device_bytes_all(
          epoch.keys, epoch.values, epoch.count_delta, epoch.quotient_off,
          epoch.op_words, epoch.lookup_mask, epoch.route_mask);
    for (const auto &epoch : delete_leaf_pool_)
      total += device_bytes_all(
          epoch.keys, epoch.values, epoch.count_delta, epoch.quotient_off,
          epoch.op_words, epoch.lookup_mask, epoch.route_mask);
    for (const auto &storage : hash_storage_pool_)
      total += device_bytes_all(
          storage.keys, storage.offsets, storage.counts,
          storage.child_views, storage.child_router);
    for (const auto &slab : leaf_slabs_)
      total += device_bytes(slab.arena);
    if (pending_leaf_slab_valid_) {
      const std::size_t owned =
          device_bytes(pending_leaf_slab_.arena);
      total += owned != 0u ? owned : pending_leaf_slab_bytes_;
    }
    return total;
  }

private:
  struct AssignmentLeafStorage {
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> keys;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> values;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_off;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> op_words;
    gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> lookup_mask;
    gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> route_mask;
    // Paged cold-run metadata (empty for packed runs).
  };

  struct HashFoldStorage {
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> keys;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> offsets;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> counts;
    gpulsmopt_detail::RawDeviceBuffer<
        gpulsmopt_detail::AssignmentRunView> child_views;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> child_router;
  };

  struct LeafSlab {
    gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> arena;
    std::uint32_t *keys = nullptr;
    std::uint32_t *values = nullptr;
    std::uint32_t *quotient_off = nullptr;
    std::uint64_t *lookup_mask = nullptr;
    std::uint64_t *route_mask = nullptr;
    std::size_t records_per_leaf = 0;
    bool with_values = false;
  };

  struct LeafSlabLayout {
    std::size_t bytes = 0;
    std::size_t keys_offset = 0;
    std::size_t values_offset = 0;
    std::size_t quotient_offset = 0;
    std::size_t lookup_offset = 0;
    std::size_t route_offset = 0;
  };

  struct RetainedLeaf : AssignmentLeafStorage {
    std::size_t count = 0;
    std::uint64_t sequence = 0;
    gpulsmopt_detail::RunOperation operation =
        gpulsmopt_detail::RunOperation::Insert;
    bool lookup_mask_ready = false;
    bool route_mask_ready = false;
    std::uint32_t owner_leaf_id = gpulsmopt_detail::kNoOwnerLeaf;
  };

  struct HostLookupLeaf {
    std::uint64_t sequence = 0;
    gpulsmopt_detail::LookupLeafView view{};
  };

  struct RunStorage : AssignmentLeafStorage {
    gpulsmopt_detail::RawDeviceBuffer<std::int8_t> count_delta;
    HashFoldStorage hash_storage;
    std::vector<RetainedLeaf> hash_children;
    std::size_t count = 0;
    // Temporal identity of an immutable assignment run.
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
    bool hashed = false;
    bool grouped = false;
    bool nested = false;
    bool lookup_mask_ready = false;
    bool route_mask_ready = false;
    std::uint16_t group_child_count = 0;
    int parent_slot = -1;
    std::uint32_t owner_leaf_id = gpulsmopt_detail::kNoOwnerLeaf;
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
    return v.owns() ? v.capacity() * sizeof(T) : 0u;
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

  gpulsmopt_detail::BaseRunView make_base_view() const {
    return {make_sorted_view()};
  }

  gpulsmopt_detail::QvrfView make_qvrf_view() const {
    return {make_sorted_view(),
            {raw_or_null(resolved_.keys),
             raw_or_null(resolved_.quotient_off)},
            raw_or_null(resolved_.values),
            raw_or_null(resolved_.count_delta),
            qvrf_base_q_block_offsets_.data(),
            qvrf_base_block_summaries_.data(),
            qvrf_cell_ids_.data(),
            qvrf_cell_offsets_.data(),
            qvrf_cell_keys_.data(),
            qvrf_cell_values_.data(),
            qvrf_cell_summaries_.data(),
            qvrf_q_cell_offsets_.data(),
            qvrf_current_q_summaries_.data(),
            qvrf_top_l1_.data(),
            qvrf_top_l2_.data(),
            qvrf_top_l3_.data(),
            qvrf_top_l4_.data(),
            qvrf_cell_count_,
            qvrf_materialized_cells_ ? 1u : 0u};
  }

  void clear_qvrf_state() {
    qvrf_base_block_counts_.resize_discard(0);
    qvrf_base_q_block_offsets_.resize_discard(0);
    qvrf_base_block_begins_.resize_discard(0);
    qvrf_base_block_sizes_.resize_discard(0);
    qvrf_base_block_summaries_.resize_discard(0);
    qvrf_base_q_summaries_.resize_discard(0);
    qvrf_current_q_summaries_.resize_discard(0);
    qvrf_top_l1_.resize_discard(0);
    qvrf_top_l2_.resize_discard(0);
    qvrf_top_l3_.resize_discard(0);
    qvrf_top_l4_.resize_discard(0);
    qvrf_cell_ids_.resize_discard(0);
    qvrf_cell_counts_.resize_discard(0);
    qvrf_cell_offsets_.resize_discard(0);
    qvrf_cell_keys_.release();
    qvrf_cell_values_.release();
    qvrf_cell_summaries_.resize_discard(0);
    qvrf_q_cell_offsets_.resize_discard(0);
    qvrf_cell_count_ = 0u;
    qvrf_materialized_cells_ = false;
    qvrf_base_ready_ = false;
    qvrf_ready_ = false;
    qvrf_base_generation_ = ~std::uint64_t{0};
    qvrf_run_sequence_ = 0u;
  }

  void clear_sorted_state() {
    sorted_run_index_ = std::numeric_limits<std::size_t>::max();
    base_rank23_.resize_discard(0);
    clear_qvrf_state();
  }

  // Builds the flat rank directory per base.
  void build_base_rank23(RunStorage &base, cudaStream_t stream) {
    base_rank23_.resize_discard_exact(
        gpulsmopt_detail::kBaseRank23Size + 1);
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

  void build_qvrf_top_tree(cudaStream_t stream) {
    constexpr int block = 256;
    gpulsmopt_detail::qvrf_reduce16_kernel<<<
        (gpulsmopt_detail::kQvrfTopL1 + block - 1u) / block,
        block, 0, stream>>>(qvrf_current_q_summaries_.data(),
                            qvrf_top_l1_.data(),
                            gpulsmopt_detail::kQvrfTopL1);
    gpulsmopt_detail::qvrf_reduce16_kernel<<<
        (gpulsmopt_detail::kQvrfTopL2 + block - 1u) / block,
        block, 0, stream>>>(qvrf_top_l1_.data(), qvrf_top_l2_.data(),
                            gpulsmopt_detail::kQvrfTopL2);
    gpulsmopt_detail::qvrf_reduce16_kernel<<<
        (gpulsmopt_detail::kQvrfTopL3 + block - 1u) / block,
        block, 0, stream>>>(qvrf_top_l2_.data(), qvrf_top_l3_.data(),
                            gpulsmopt_detail::kQvrfTopL3);
    gpulsmopt_detail::qvrf_reduce16_kernel<<<1, block, 0, stream>>>(
        qvrf_top_l3_.data(), qvrf_top_l4_.data(),
        gpulsmopt_detail::kQvrfTopL4);
    CUDA_CHECK(cudaGetLastError());
  }

  void build_qvrf_base_metadata(cudaStream_t stream) {
    constexpr int block = 256;
    constexpr std::uint32_t q_count = gpulsmopt_detail::kEpochQuotients;
    constexpr int q_grid = (q_count + block - 1) / block;
    qvrf_base_block_counts_.resize_discard_exact(q_count + 1u);
    qvrf_base_q_block_offsets_.resize_discard_exact(q_count + 1u);
    CUDA_CHECK(cudaMemsetAsync(qvrf_base_block_counts_.data(), 0,
                               (q_count + 1u) * sizeof(std::uint32_t),
                               stream));
    gpulsmopt_detail::qvrf_base_block_count_kernel<<<q_grid, block, 0,
                                                     stream>>>(
        base_rank23_.data(), qvrf_base_block_counts_.data());
    CUDA_CHECK(cudaGetLastError());
    exclusive_scan_u32(qvrf_base_block_counts_.data(),
                       qvrf_base_q_block_offsets_.data(), q_count + 1u,
                       stream);
    CUDA_CHECK(cudaMemcpyAsync(
        &host_state_->qvrf_base_block_count,
        qvrf_base_q_block_offsets_.data() + q_count,
        sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::uint32_t block_count =
        host_state_->qvrf_base_block_count;

    qvrf_base_block_begins_.resize_discard_exact(block_count);
    qvrf_base_block_sizes_.resize_discard_exact(block_count);
    qvrf_base_block_summaries_.resize_discard_exact(block_count);
    qvrf_base_q_summaries_.resize_discard_exact(q_count);
    qvrf_current_q_summaries_.resize_discard_exact(q_count);
    qvrf_top_l1_.resize_discard_exact(gpulsmopt_detail::kQvrfTopL1);
    qvrf_top_l2_.resize_discard_exact(gpulsmopt_detail::kQvrfTopL2);
    qvrf_top_l3_.resize_discard_exact(gpulsmopt_detail::kQvrfTopL3);
    qvrf_top_l4_.resize_discard_exact(gpulsmopt_detail::kQvrfTopL4);
    qvrf_q_cell_offsets_.resize_discard_exact(q_count + 1u);
    CUDA_CHECK(cudaMemsetAsync(qvrf_q_cell_offsets_.data(), 0,
                               (q_count + 1u) * sizeof(std::uint32_t),
                               stream));

    gpulsmopt_detail::qvrf_base_block_layout_kernel<<<q_grid, block, 0,
                                                      stream>>>(
        base_rank23_.data(), qvrf_base_q_block_offsets_.data(),
        qvrf_base_block_begins_.data(), qvrf_base_block_sizes_.data());
    CUDA_CHECK(cudaGetLastError());
    if (block_count > 0u) {
      gpulsmopt_detail::qvrf_base_block_summary_kernel<<<
          block_count, gpulsmopt_detail::kQvrfBaseLeafRecords, 0, stream>>>(
          make_sorted_view(), qvrf_base_block_begins_.data(),
          qvrf_base_block_sizes_.data(),
          qvrf_base_q_block_offsets_.data(),
          qvrf_base_block_summaries_.data());
      CUDA_CHECK(cudaGetLastError());
    }
    gpulsmopt_detail::qvrf_base_q_summary_kernel<<<q_grid, block, 0, stream>>>(
        qvrf_base_q_block_offsets_.data(),
        qvrf_base_block_summaries_.data(),
        qvrf_base_q_summaries_.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(
        qvrf_current_q_summaries_.data(), qvrf_base_q_summaries_.data(),
        q_count * sizeof(gpulsmopt_detail::QvrfSummary),
        cudaMemcpyDeviceToDevice, stream));
    build_qvrf_top_tree(stream);

    qvrf_cell_ids_.resize_discard(0);
    qvrf_cell_counts_.resize_discard(0);
    qvrf_cell_offsets_.resize_discard(0);
    qvrf_cell_keys_.release();
    qvrf_cell_values_.release();
    qvrf_cell_summaries_.resize_discard(0);
    qvrf_cell_count_ = 0u;
    qvrf_materialized_cells_ = false;
    qvrf_base_generation_ = base_generation_;
    qvrf_base_ready_ = true;
    qvrf_run_sequence_ = run_sequence_;
    qvrf_ready_ = run_sequence_ == 0u;
  }

  void ensure_qvrf_visible(cudaStream_t stream) {
    if (!qvrf_base_ready_ || qvrf_base_generation_ != base_generation_)
      build_qvrf_base_metadata(stream);
    if (qvrf_ready_ && qvrf_base_generation_ == base_generation_ &&
        qvrf_run_sequence_ == run_sequence_)
      return;

    constexpr int block = 256;
    constexpr std::uint32_t q_count = gpulsmopt_detail::kEpochQuotients;
    constexpr int q_grid = (q_count + block - 1) / block;

    // Restore the immutable snapshot before overlaying the current dirty
    // cells. Rebuilding from the complete resolved cache is what makes a
    // correction that returns to Base disappear rather than leave stale state.
    CUDA_CHECK(cudaMemcpyAsync(
        qvrf_current_q_summaries_.data(), qvrf_base_q_summaries_.data(),
        q_count * sizeof(gpulsmopt_detail::QvrfSummary),
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(qvrf_q_cell_offsets_.data(), 0,
                               (q_count + 1u) * sizeof(std::uint32_t),
                               stream));
    qvrf_cell_count_ = 0u;
    qvrf_cell_ids_.resize_discard(0);
    qvrf_cell_counts_.resize_discard(0);
    qvrf_cell_offsets_.resize_discard(0);
    qvrf_cell_keys_.resize_discard(0);
    qvrf_cell_values_.resize_discard(0);
    qvrf_cell_summaries_.resize_discard(0);
    qvrf_materialized_cells_ = false;

    if (run_sequence_ != 0u)
      ensure_resolved(stream);

    const std::uint32_t correction_count =
        static_cast<std::uint32_t>(resolved_.count);
    if (run_sequence_ != 0u && correction_count > 0u) {
      qvrf_cell_ids_.resize_discard(correction_count);
      auto policy = thrust::cuda::par.on(stream);
      thrust::transform(policy, resolved_.keys.data(),
                        resolved_.keys.data() + correction_count,
                        qvrf_cell_ids_.data(),
                        gpulsmopt_detail::QvrfKeyToCell{});
      auto unique_end = thrust::unique(
          policy, qvrf_cell_ids_.data(),
          qvrf_cell_ids_.data() + correction_count);
      qvrf_cell_count_ = static_cast<std::uint32_t>(
          unique_end - qvrf_cell_ids_.data());
      qvrf_cell_ids_.resize_discard(qvrf_cell_count_);
      qvrf_cell_summaries_.resize_discard(qvrf_cell_count_);

      const int cell_grid = static_cast<int>(
          (qvrf_cell_count_ + block - 1u) / block);
      const std::uint64_t row_upper =
          static_cast<std::uint64_t>(qvrf_cell_count_) *
              (std::uint64_t{1} << gpulsmopt_detail::kBaseRank23Shift) +
          correction_count;
      const std::uint64_t byte_upper =
          row_upper * 2u * sizeof(std::uint32_t) +
          static_cast<std::uint64_t>(qvrf_cell_count_) *
              2u * sizeof(std::uint32_t);
      std::size_t free_bytes = 0u, total_bytes = 0u;
      CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
      (void)total_bytes;
      const bool reuses_rows =
          row_upper <= qvrf_cell_keys_.capacity() &&
          row_upper <= qvrf_cell_values_.capacity();
      bool materialize = reuses_rows ||
          byte_upper <= static_cast<std::uint64_t>(free_bytes) /
                            gpulsmopt_detail::
                                kQvrfMaterializationFreeDivisor;
#ifdef GPULSMOPT_QVRF_FORCE_VIRTUAL
      materialize = false;
#endif
      if (materialize) {
        qvrf_cell_counts_.resize_discard(qvrf_cell_count_);
        qvrf_cell_offsets_.resize_discard(qvrf_cell_count_ + 1u);
        gpulsmopt_detail::qvrf_cell_visible_count_kernel<<<
            cell_grid, block, 0, stream>>>(make_qvrf_view(),
                                          qvrf_cell_counts_.data());
        CUDA_CHECK(cudaGetLastError());
        exclusive_scan_u32(qvrf_cell_counts_.data(),
                           qvrf_cell_offsets_.data(), qvrf_cell_count_,
                           stream);
        gpulsmopt_detail::qvrf_finish_offsets_kernel<<<1, 1, 0, stream>>>(
            qvrf_cell_counts_.data(), qvrf_cell_offsets_.data(),
            qvrf_cell_count_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(
            &host_state_->qvrf_visible_count,
            qvrf_cell_offsets_.data() + qvrf_cell_count_,
            sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::uint32_t visible_count =
            host_state_->qvrf_visible_count;
        qvrf_cell_keys_.resize_discard_exact(visible_count);
        qvrf_cell_values_.resize_discard_exact(visible_count);
        qvrf_materialized_cells_ = true;
        gpulsmopt_detail::qvrf_cell_emit_kernel<<<
            cell_grid, block, 0, stream>>>(
            make_qvrf_view(), qvrf_cell_offsets_.data(),
            qvrf_cell_keys_.data(), qvrf_cell_values_.data(),
            qvrf_cell_summaries_.data());
        CUDA_CHECK(cudaGetLastError());
      } else {
        qvrf_cell_counts_.resize_discard(0);
        qvrf_cell_offsets_.resize_discard(0);
        qvrf_cell_keys_.release();
        qvrf_cell_values_.release();
        gpulsmopt_detail::qvrf_cell_summary_kernel<<<
            cell_grid, block, 0, stream>>>(make_qvrf_view(),
                                          qvrf_cell_summaries_.data());
        CUDA_CHECK(cudaGetLastError());
      }
      constexpr int q_offset_grid =
          (gpulsmopt_detail::kEpochQuotients + 1 + block - 1) / block;
      gpulsmopt_detail::qvrf_q_cell_offsets_kernel<<<
          q_offset_grid, block, 0, stream>>>(
          qvrf_cell_ids_.data(), qvrf_cell_count_,
          qvrf_q_cell_offsets_.data());
      CUDA_CHECK(cudaGetLastError());
      gpulsmopt_detail::qvrf_dirty_q_summary_kernel<<<
          q_grid, block, 0, stream>>>(make_qvrf_view(),
                                     qvrf_current_q_summaries_.data());
      CUDA_CHECK(cudaGetLastError());
    }

    if (qvrf_cell_count_ == 0u) {
      qvrf_cell_counts_.resize_discard(0);
      qvrf_cell_offsets_.resize_discard(0);
      qvrf_cell_keys_.release();
      qvrf_cell_values_.release();
    }

    build_qvrf_top_tree(stream);
    qvrf_base_generation_ = base_generation_;
    qvrf_run_sequence_ = run_sequence_;
    qvrf_ready_ = true;
  }

  void build_sorted_metadata(cudaStream_t stream) {
    RunStorage &run = runs_[sorted_run_index_];
    build_base_rank23(run, stream);
    // The forest is intentionally lazy. Point lookup, successor, and ordinary
    // updates pay no range-sidecar construction cost.
    qvrf_base_ready_ = false;
    qvrf_ready_ = false;
  }

  void build_sorted_run_cache(std::size_t index, cudaStream_t stream) {
    clear_sorted_state();
    if (index >= runs_.size() || !runs_[index].fully_sorted ||
        !runs_[index].unique_keys)
      return;
    sorted_run_index_ = index;
    build_sorted_metadata(stream);
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
                             cudaStream_t stream,
                             bool input_sorted_by_quotient = false) {
    acquire_run_slot(is_insert, count, stream);
    RunStorage &run = runs_.back();
    run.count = count;
    run.assignment = true;
    run.assignment_class = gpulsmopt_detail::AssignmentClass::Raw;
    run.hashed = false;
    run.grouped = false;
    run.nested = false;
    run.lookup_mask_ready = false;
    run.route_mask_ready = false;
    run.group_child_count = 0;
    run.parent_slot = -1;
    run.hash_children.clear();
    run.mixed = false;
    run.stable_level = -1;
    run.operation = is_insert ? gpulsmopt_detail::RunOperation::Insert
                              : gpulsmopt_detail::RunOperation::Delete;
    run.sequence_begin = ++run_sequence_;
    run.sequence_end = run.sequence_begin;
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
      GPULSMOPT_PROF_PHASE(prof_delta_sort_ms_);
      run.values.resize_discard(0);
      if (input_sorted_by_quotient) {
        CUDA_CHECK(cudaMemcpyAsync(run.keys.data(), keys,
                                   count * sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToDevice, stream));
      } else {
        // Deletions have no payload, but the stable quotient sort still needs
        // a value lane. Reuse the ordinary sort scratch; the dummy values are
        // never published in the run descriptor.
        direct_sort_values_.resize_discard(count);
        sort_run_batch(keys, keys, count, run.keys.data(),
                       direct_sort_values_.data(), stream);
      }
      {
        GPULSMOPT_PROF_PHASE(prof_delta_ingest_ms_);
        // Build quotient offsets from quotient-ascending deletion keys.
        run.quotient_off.resize_discard_exact(
            gpulsmopt_detail::kEpochQuotients + 1);
        build_quotient_offsets(run.keys.data(),
                               static_cast<std::uint32_t>(count),
                               run.quotient_off.data(), stream);
      }
    }
    register_lookup_owner_leaf(run);
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
    const bool hashed = run.hashed;
    if (hashed &&
        (run.hash_storage.offsets.data() == nullptr ||
         run.hash_storage.counts.data() == nullptr ||
         run.hash_storage.child_views.data() == nullptr ||
         run.hash_storage.child_router.data() == nullptr))
      throw std::runtime_error("hashed run has no storage");
    if (run.grouped &&
        (run.parent_slot < 0 ||
         run.parent_slot >= gpulsmopt_detail::kParentSlots))
      throw std::runtime_error("grouped run has no parent slot");
    std::uint32_t *keys = hashed
        ? run.hash_storage.keys.data() : run.keys.data();
    const std::uint32_t *offsets = hashed
        ? run.hash_storage.offsets.data() : run.quotient_off.data();
    const std::uint32_t *quotient_meta = hashed
        ? run.hash_storage.counts.data() : nullptr;
    const gpulsmopt_detail::AssignmentRunView *children = hashed
        ? run.hash_storage.child_views.data() : nullptr;
    const std::uint32_t *child_router = hashed
        ? run.hash_storage.child_router.data() : nullptr;
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
            run.grouped ? nullptr :
                (insert ? run.values.data() : nullptr),
            run.grouped ? nullptr : offsets,
            run.grouped ? nullptr : quotient_meta,
            run.grouped ? nullptr :
                (run.mixed ? run.op_words.data() : nullptr),
            run.grouped ? nullptr : lookup_mask,
            run.grouped ? nullptr : children,
            run.grouped ? nullptr : child_router,
            group_children,
            static_cast<std::uint16_t>(
                run.grouped ? run.group_child_count
                            : (hashed ? run.hash_children.size() : 0u)),
            static_cast<std::uint8_t>(insert ? 1u : 0u),
            static_cast<std::uint8_t>(run.mixed ? 1u : 0u),
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
    // insert; hash folds publish directly into their arena.
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
            0u,
            static_cast<std::uint8_t>(insert ? 1u : 0u),
            0u,
            0u,
            0u};
  }

  gpulsmopt_detail::LookupOwnerLeafView
  make_lookup_owner_view(RunStorage &run) {
    const bool insert =
        run.operation == gpulsmopt_detail::RunOperation::Insert;
    return {run.keys.data(), insert ? run.values.data() : nullptr,
            run.quotient_off.data(), run.route_mask.data(),
            static_cast<std::uint32_t>(run.count),
            static_cast<std::uint8_t>(insert ? 1u : 0u)};
  }

  void register_lookup_owner_leaf(RunStorage &run) {
    run.owner_leaf_id = gpulsmopt_detail::kNoOwnerLeaf;
    if (!lookup_owner_usable_)
      return;
    const std::size_t position_capacity =
        lookup_owner_position_bits_ == 0u
            ? 1u
            : std::size_t{1} << lookup_owner_position_bits_;
    if (run.count > position_capacity ||
        lookup_owner_leaf_count_ >=
            gpulsmopt_detail::kLookupLeafCapacity) {
      lookup_owner_usable_ = false;
      return;
    }
    const std::uint32_t leaf_id = lookup_owner_leaf_count_++;
    run.owner_leaf_id = leaf_id;
    host_state_->owner_leaves[leaf_id] = make_lookup_owner_view(run);
  }

  void mark_lookup_owner_routes(std::uint32_t first,
                                std::uint32_t end) {
    for (RunStorage &run : runs_) {
      if (run.owner_leaf_id >= first && run.owner_leaf_id < end)
        run.route_mask_ready = true;
      for (RetainedLeaf &leaf : run.hash_children) {
        if (leaf.owner_leaf_id >= first && leaf.owner_leaf_id < end)
          leaf.route_mask_ready = true;
      }
    }
  }


  bool should_use_lookup_owner(std::size_t queries) const {
    if (!lookup_owner_usable_ || lookup_owner_leaf_count_ == 0u)
      return false;
    const std::size_t threshold =
        std::max<std::size_t>(1u, batch_capacity_ / 4u);
    return queries >= threshold;
  }

  void prepare_lookup_owner(cudaStream_t stream) {
    if (lookup_owner_published_leaf_count_ < lookup_owner_leaf_count_) {
      const std::uint32_t first = lookup_owner_published_leaf_count_;
      const std::uint32_t count = lookup_owner_leaf_count_ - first;
      CUDA_CHECK(cudaMemcpyAsync(
          lookup_owner_leaves_.data() + first,
          host_state_->owner_leaves + first,
          count * sizeof(gpulsmopt_detail::LookupOwnerLeafView),
          cudaMemcpyHostToDevice, stream));
      lookup_owner_published_leaf_count_ = lookup_owner_leaf_count_;
    }
    if (lookup_owner_applied_leaf_count_ >= lookup_owner_leaf_count_)
      return;
    const std::uint32_t first = lookup_owner_applied_leaf_count_;
    const std::uint32_t pending = lookup_owner_leaf_count_ - first;
    constexpr int block = 256;
    constexpr int warps = block / 32;
    constexpr int grid =
        (gpulsmopt_detail::kEpochQuotients + warps - 1) / warps;
    gpulsmopt_detail::lookup_owner_apply_kernel<<<
        grid, block, 0, stream>>>(
        lookup_owner_leaves_.data(), first, pending,
        lookup_owner_position_bits_, lookup_owner_slots_.data(),
        lookup_owner_table_mask_, lookup_owner_overflow_.data());
    CUDA_CHECK(cudaGetLastError());
    mark_lookup_owner_routes(first, first + pending);
    lookup_owner_applied_leaf_count_ = lookup_owner_leaf_count_;
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
          (!run.lookup_mask_ready || !run.route_mask_ready)) {
        run.lookup_mask.resize_discard_exact(
            gpulsmopt_detail::kEpochQuotients);
        run.route_mask.resize_discard_exact(
            gpulsmopt_detail::kEpochQuotients);
        gpulsmopt_detail::raw_quotient_masks_kernel<true><<<
            grid, block, 0, stream>>>(
            run.keys.data(), run.quotient_off.data(),
            run.lookup_mask.data(), run.route_mask.data());
        CUDA_CHECK(cudaGetLastError());
        run.lookup_mask_ready = true;
        run.route_mask_ready = true;
        changed = true;
      }
      if (!run.hashed || run.hash_children.empty())
        continue;
      bool refresh = false;
      for (RetainedLeaf &child : run.hash_children) {
        if (!child.lookup_mask_ready || !child.route_mask_ready) {
          child.lookup_mask.resize_discard_exact(
              gpulsmopt_detail::kEpochQuotients);
          child.route_mask.resize_discard_exact(
              gpulsmopt_detail::kEpochQuotients);
          gpulsmopt_detail::raw_quotient_masks_kernel<true><<<
              grid, block, 0, stream>>>(
              child.keys.data(), child.quotient_off.data(),
              child.lookup_mask.data(), child.route_mask.data());
          CUDA_CHECK(cudaGetLastError());
          child.lookup_mask_ready = true;
          child.route_mask_ready = true;
          refresh = true;
        }
      }
      if (!refresh)
        continue;
      for (std::size_t child = 0;
           child < run.hash_children.size(); ++child)
        host_state_->scratch_views[child] =
            make_retained_leaf_view(run.hash_children[child]);
      CUDA_CHECK(cudaMemcpyAsync(
          run.hash_storage.child_views.data(),
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

  static std::size_t checked_product(
      std::size_t left, std::size_t right) {
    if (right != 0u &&
        left > std::numeric_limits<std::size_t>::max() / right)
      throw std::overflow_error("leaf slab size overflow");
    return left * right;
  }

  static std::size_t checked_sum(
      std::size_t left, std::size_t right) {
    if (left > std::numeric_limits<std::size_t>::max() - right)
      throw std::overflow_error("leaf slab size overflow");
    return left + right;
  }

  static std::size_t append_leaf_region(
      std::size_t &cursor, std::size_t bytes,
      std::size_t alignment) {
    const std::size_t remainder = cursor % alignment;
    if (remainder != 0u)
      cursor = checked_sum(cursor, alignment - remainder);
    const std::size_t offset = cursor;
    cursor = checked_sum(cursor, bytes);
    return offset;
  }

  static LeafSlabLayout leaf_slab_layout(
      std::size_t count, bool with_values) {
    count = std::max<std::size_t>(count, 1u);
    constexpr std::size_t slots =
        gpulsmopt_detail::kLeafSlabRuns;
    const std::size_t records = checked_product(slots, count);
    const std::size_t quotient_entries = checked_product(
        slots, gpulsmopt_detail::kEpochQuotients + 1u);
    const std::size_t mask_entries = checked_product(
        slots, gpulsmopt_detail::kEpochQuotients);
    LeafSlabLayout layout;
    layout.keys_offset = append_leaf_region(
        layout.bytes,
        checked_product(records, sizeof(std::uint32_t)),
        alignof(std::uint32_t));
    if (with_values)
      layout.values_offset = append_leaf_region(
          layout.bytes,
          checked_product(records, sizeof(std::uint32_t)),
          alignof(std::uint32_t));
    layout.quotient_offset = append_leaf_region(
        layout.bytes,
        checked_product(quotient_entries, sizeof(std::uint32_t)),
        alignof(std::uint32_t));
    layout.lookup_offset = append_leaf_region(
        layout.bytes,
        checked_product(mask_entries, sizeof(std::uint64_t)),
        alignof(std::uint64_t));
    layout.route_offset = append_leaf_region(
        layout.bytes,
        checked_product(mask_entries, sizeof(std::uint64_t)),
        alignof(std::uint64_t));
    return layout;
  }

  static void bind_leaf_slab(
      LeafSlab &slab, const LeafSlabLayout &layout) {
    std::uint8_t *base = slab.arena.data();
    slab.keys = reinterpret_cast<std::uint32_t *>(
        base + layout.keys_offset);
    if (slab.with_values)
      slab.values = reinterpret_cast<std::uint32_t *>(
          base + layout.values_offset);
    slab.quotient_off = reinterpret_cast<std::uint32_t *>(
        base + layout.quotient_offset);
    slab.lookup_mask = reinterpret_cast<std::uint64_t *>(
        base + layout.lookup_offset);
    slab.route_mask = reinterpret_cast<std::uint64_t *>(
        base + layout.route_offset);
  }

  LeafSlab make_leaf_slab(
      std::size_t count, bool with_values) {
    count = std::max<std::size_t>(count, 1u);
    const LeafSlabLayout layout =
        leaf_slab_layout(count, with_values);
    LeafSlab slab;
    slab.records_per_leaf = count;
    slab.with_values = with_values;
    slab.arena.resize_discard_exact(layout.bytes);
    bind_leaf_slab(slab, layout);
    return slab;
  }

  void leaf_alloc_worker_loop() {
    const cudaError_t device_status =
        cudaSetDevice(leaf_alloc_device_);
    for (;;) {
      std::size_t bytes = 0u;
      {
        std::unique_lock<std::mutex> guard(leaf_alloc_mutex_);
        leaf_alloc_cv_.wait(guard, [this] {
          return leaf_alloc_stop_ || leaf_alloc_request_;
        });
        if (leaf_alloc_stop_ && !leaf_alloc_request_)
          return;
        bytes = leaf_alloc_request_bytes_;
        leaf_alloc_request_ = false;
      }

      std::uint8_t *arena = nullptr;
      cudaError_t status = device_status;
      if (status == cudaSuccess)
        status = cudaMallocAsync(
            reinterpret_cast<void **>(&arena), bytes,
            leaf_alloc_stream_);
      if (status == cudaSuccess)
        status = cudaEventRecord(
            leaf_alloc_ready_, leaf_alloc_stream_);

      {
        std::lock_guard<std::mutex> guard(leaf_alloc_mutex_);
        leaf_alloc_result_ = arena;
        leaf_alloc_status_ = status;
        leaf_alloc_result_ready_ = true;
      }
      leaf_alloc_cv_.notify_all();
    }
  }

  void publish_leaf_slab(LeafSlab &&slab) {
    leaf_slabs_.push_back(std::move(slab));
    LeafSlab &owner = leaf_slabs_.back();
    const std::size_t count = owner.records_per_leaf;
    const bool with_values = owner.with_values;
    auto &pool = leaf_pool(with_values);

    for (std::size_t slot = 0;
         slot < gpulsmopt_detail::kLeafSlabRuns; ++slot) {
      RunStorage leaf;
      leaf.keys.bind_external(owner.keys + slot * count, count);
      if (with_values)
        leaf.values.bind_external(
            owner.values + slot * count, count);
      leaf.quotient_off.bind_external(
          owner.quotient_off + slot *
              (gpulsmopt_detail::kEpochQuotients + 1u),
          gpulsmopt_detail::kEpochQuotients + 1u);
      leaf.lookup_mask.bind_external(
          owner.lookup_mask +
              slot * gpulsmopt_detail::kEpochQuotients,
          gpulsmopt_detail::kEpochQuotients);
      leaf.route_mask.bind_external(
          owner.route_mask +
              slot * gpulsmopt_detail::kEpochQuotients,
          gpulsmopt_detail::kEpochQuotients);
      pool.push_back(std::move(leaf));
    }
  }

  void grow_leaf_pool(std::size_t count, bool with_values) {
    publish_leaf_slab(
        make_leaf_slab(count, with_values));
  }

  void start_pending_leaf_slab(
      std::size_t count, bool with_values) {
    if (pending_leaf_slab_valid_)
      throw std::runtime_error("leaf slab request already pending");
    count = std::max<std::size_t>(count, 1u);
    const LeafSlabLayout layout =
        leaf_slab_layout(count, with_values);
    pending_leaf_slab_ = LeafSlab{};
    pending_leaf_slab_.records_per_leaf = count;
    pending_leaf_slab_.with_values = with_values;
    pending_leaf_slab_bytes_ = layout.bytes;
    {
      std::lock_guard<std::mutex> guard(leaf_alloc_mutex_);
      if (leaf_alloc_request_ || leaf_alloc_result_ready_)
        throw std::runtime_error("leaf allocator is busy");
      leaf_alloc_request_bytes_ = layout.bytes;
      leaf_alloc_request_ = true;
    }
    pending_leaf_slab_valid_ = true;
    leaf_alloc_cv_.notify_one();
  }

  bool promote_pending_leaf_slab(
      cudaStream_t stream, bool wait) {
    if (!pending_leaf_slab_valid_)
      return false;
    std::uint8_t *arena = nullptr;
    cudaError_t allocation_status = cudaSuccess;
    {
      std::unique_lock<std::mutex> guard(leaf_alloc_mutex_);
      if (!leaf_alloc_result_ready_) {
        if (!wait)
          return false;
        leaf_alloc_cv_.wait(guard, [this] {
          return leaf_alloc_result_ready_;
        });
      }
      allocation_status = leaf_alloc_status_;
      if (allocation_status == cudaSuccess) {
        const cudaError_t ready =
            cudaEventQuery(leaf_alloc_ready_);
        if (ready == cudaErrorNotReady) {
          if (!wait)
            return false;
          CUDA_CHECK(cudaStreamWaitEvent(
              stream, leaf_alloc_ready_, 0));
        } else {
          CUDA_CHECK(ready);
        }
      }
      arena = leaf_alloc_result_;
      leaf_alloc_result_ = nullptr;
      leaf_alloc_result_ready_ = false;
    }
    if (allocation_status != cudaSuccess) {
      cudaStreamSynchronize(leaf_alloc_stream_);
      if (arena)
        cudaFree(arena);
      pending_leaf_slab_ = LeafSlab{};
      pending_leaf_slab_bytes_ = 0u;
      pending_leaf_slab_valid_ = false;
      CUDA_CHECK(allocation_status);
    }
    const LeafSlabLayout layout = leaf_slab_layout(
        pending_leaf_slab_.records_per_leaf,
        pending_leaf_slab_.with_values);
    pending_leaf_slab_.arena.adopt(arena, layout.bytes);
    bind_leaf_slab(pending_leaf_slab_, layout);
    LeafSlab slab = std::move(pending_leaf_slab_);
    pending_leaf_slab_ = LeafSlab{};
    pending_leaf_slab_bytes_ = 0u;
    pending_leaf_slab_valid_ = false;
    publish_leaf_slab(std::move(slab));
    return true;
  }

  static bool leaf_slot_matches(
      const RunStorage &run, bool with_values,
      std::size_t count) {
    if (!run.keys.external() ||
        run.keys.capacity() < count ||
        !run.quotient_off.external() ||
        run.quotient_off.capacity() <
            gpulsmopt_detail::kEpochQuotients + 1u ||
        !run.lookup_mask.external() ||
        !run.route_mask.external())
      return false;
    if (with_values)
      return run.values.external() &&
             run.values.capacity() >= count;
    return run.values.data() == nullptr;
  }

  std::vector<RunStorage> &leaf_pool(bool with_values) {
    return with_values ? insert_leaf_pool_ : delete_leaf_pool_;
  }

  const std::vector<RunStorage> &leaf_pool(
      bool with_values) const {
    return with_values ? insert_leaf_pool_ : delete_leaf_pool_;
  }

  std::size_t available_leaf_slots(
      bool with_values, std::size_t count) const {
    const auto &pool = leaf_pool(with_values);
    return static_cast<std::size_t>(std::count_if(
        pool.begin(), pool.end(),
        [with_values, count](const RunStorage &run) {
          return leaf_slot_matches(run, with_values, count);
        }));
  }

  RunStorage &available_insert_leaf(std::size_t count) {
    if (!insert_leaf_pool_.empty() &&
        leaf_slot_matches(insert_leaf_pool_.back(), true, count))
      return insert_leaf_pool_.back();
    auto slot = std::find_if(
        insert_leaf_pool_.begin(), insert_leaf_pool_.end(),
        [count](const RunStorage &run) {
          return leaf_slot_matches(run, true, count);
        });
    if (slot == insert_leaf_pool_.end())
      throw std::runtime_error("no insert leaf slab available");
    return *slot;
  }

  void reserve_leaf_storage(
      std::size_t count, cudaStream_t stream) {
    if (pending_leaf_slab_valid_)
      promote_pending_leaf_slab(stream, true);
    const std::size_t insert_target =
        gpulsmopt_detail::kInitialInsertLeafSlabs *
        gpulsmopt_detail::kLeafSlabRuns;
    const std::size_t delete_target =
        gpulsmopt_detail::kInitialDeleteLeafSlabs *
        gpulsmopt_detail::kLeafSlabRuns;
    while (available_leaf_slots(true, count) < insert_target)
      grow_leaf_pool(count, true);
    while (available_leaf_slots(false, count) < delete_target)
      grow_leaf_pool(count, false);
  }

  void maintain_leaf_spare(
      std::size_t count, cudaStream_t stream) {
    count = std::max<std::size_t>(count, 1u);
    constexpr std::size_t low =
        gpulsmopt_detail::kLeafSlabRuns;
    std::size_t insert_free = available_leaf_slots(true, count);
    std::size_t delete_free = available_leaf_slots(false, count);

    if (pending_leaf_slab_valid_) {
      const bool pending_insert =
          pending_leaf_slab_.with_values;
      const std::size_t pending_free =
          pending_insert ? insert_free : delete_free;
      const std::size_t other_free =
          pending_insert ? delete_free : insert_free;
      if (pending_free == 0u || other_free <= low) {
        promote_pending_leaf_slab(stream, true);
        insert_free = available_leaf_slots(true, count);
        delete_free = available_leaf_slots(false, count);
      }
    }

    if (!pending_leaf_slab_valid_) {
      const bool insert_low = insert_free <= low;
      const bool delete_low = delete_free <= low;
      if (insert_low || delete_low) {
        const bool with_values =
            insert_low && (!delete_low || insert_free <= delete_free);
        start_pending_leaf_slab(count, with_values);
        const std::size_t free =
            with_values ? insert_free : delete_free;
        if (free == 0u)
          promote_pending_leaf_slab(stream, true);
      }
    }
  }

  void acquire_run_slot(bool with_values, std::size_t count,
                        cudaStream_t stream) {
    auto &pool = leaf_pool(with_values);
    if (pool.empty() ||
        !leaf_slot_matches(pool.back(), with_values, count)) {
      auto slot = std::find_if(
          pool.begin(), pool.end(),
          [with_values, count](const RunStorage &run) {
            return leaf_slot_matches(run, with_values, count);
          });
      if (slot != pool.end())
        std::iter_swap(slot, pool.end() - 1);
    }
    if (pool.empty() ||
        !leaf_slot_matches(pool.back(), with_values, count)) {
      if (pending_leaf_slab_valid_)
        promote_pending_leaf_slab(stream, true);
      if (pool.empty() ||
          !leaf_slot_matches(pool.back(), with_values, count)) {
        start_pending_leaf_slab(count, with_values);
        promote_pending_leaf_slab(stream, true);
      }
    }
    if (pool.empty() ||
        !leaf_slot_matches(pool.back(), with_values, count))
      throw std::runtime_error("leaf slab allocation failed");
    runs_.push_back(std::move(pool.back()));
    pool.pop_back();
  }

  void recycle_run_storage(RunStorage &&run) {
    if (!run.keys.external()) {
      run_pool_.push_back(std::move(run));
      return;
    }
    if (run.values.external())
      insert_leaf_pool_.push_back(std::move(run));
    else
      delete_leaf_pool_.push_back(std::move(run));
  }

  void acquire_compaction_slot() {
    auto best = run_pool_.end();
    for (auto it = run_pool_.begin();
         it != run_pool_.end(); ++it) {
      if (!it->keys.owns())
        continue;
      if (best == run_pool_.end() ||
          it->keys.capacity() > best->keys.capacity())
        best = it;
    }
    if (best == run_pool_.end()) {
      runs_.emplace_back();
      return;
    }
    runs_.push_back(std::move(*best));
    run_pool_.erase(best);
  }

  void release_hash_children(RunStorage &run) {
    for (RetainedLeaf &child : run.hash_children) {
      RunStorage leaf;
      leaf.operation = child.operation;
      leaf.keys = std::move(child.keys);
      leaf.values = std::move(child.values);
      leaf.quotient_off = std::move(child.quotient_off);
      leaf.op_words = std::move(child.op_words);
      leaf.lookup_mask = std::move(child.lookup_mask);
      leaf.route_mask = std::move(child.route_mask);
      recycle_run_storage(std::move(leaf));
    }
    hash_child_pool_.push_back(
        std::move(run.hash_children));
  }

  std::vector<RetainedLeaf> acquire_hash_children() {
    if (hash_child_pool_.empty())
      throw std::runtime_error("no free hash-child container");
    std::vector<RetainedLeaf> children =
        std::move(hash_child_pool_.back());
    hash_child_pool_.pop_back();
    return children;
  }

  void initialize_empty_base(cudaStream_t stream) {
    acquire_compaction_slot();
    RunStorage &base = runs_.back();
    base.sequence_begin = 0u;
    base.sequence_end = 0u;
    base.stable_level = -1;
    base.operation = gpulsmopt_detail::RunOperation::Insert;
    base.assignment_class = gpulsmopt_detail::AssignmentClass::Raw;
    base.mixed = false;
    base.assignment = false;
    base.hashed = false;
    base.grouped = false;
    base.nested = false;
    base.lookup_mask_ready = false;
    base.route_mask_ready = false;
    base.group_child_count = 0;
    base.parent_slot = -1;
    base.owner_leaf_id = gpulsmopt_detail::kNoOwnerLeaf;
    base.count = 0u;
    base.fully_sorted = true;
    base.unit_counts = true;
    base.unique_keys = true;
    base.hash_children.clear();
    base.keys.resize_discard(0);
    base.values.resize_discard(0);
    base.quotient_off.release();
    base.op_words.release();
    base.count_delta.release();
    build_sorted_run_cache(runs_.size() - 1u, stream);
  }

  void clear_run_state() {
    clear_sorted_state();
    for (auto &epoch : runs_) {
      if (epoch.hashed) {
        release_hash_children(epoch);
        recycle_hash_storage(epoch);
      }
      if (epoch.grouped && epoch.parent_slot >= 0 &&
          epoch.parent_slot < gpulsmopt_detail::kParentSlots)
        parent_slot_used_[epoch.parent_slot] = false;
      epoch.hashed = false;
      epoch.grouped = false;
      epoch.nested = false;
      epoch.group_child_count = 0;
      epoch.parent_slot = -1;
      recycle_run_storage(std::move(epoch));
    }
    runs_.clear();
  }

  void insert_records(const std::uint32_t *keys_in,
                      const std::uint32_t *values_in, std::uint8_t op,
                      std::size_t count, cudaStream_t stream,
                      bool input_sorted_by_quotient = false) {
    if (count == 0)
      return;
    const bool is_insert =
        op == static_cast<std::uint8_t>(gpulsmopt_detail::kInsert);
    create_assignment_run(is_insert, keys_in, is_insert ? values_in : nullptr,
                          count, stream, input_sorted_by_quotient);
  }

  void ensure_sort_temp(std::size_t bytes) {
    if (sort_temp_storage_.capacity() < bytes)
      sort_temp_storage_.resize_discard(bytes);
  }

  static std::size_t range_arena_records(
      std::size_t count) {
    std::size_t capacity = 1u;
    while (capacity < count) {
      if (capacity >
          std::numeric_limits<std::size_t>::max() / 2u)
        throw std::overflow_error("range arena overflow");
      capacity *= 2u;
    }
    return capacity;
  }

  static std::size_t range_arena_bytes(
      std::size_t capacity) {
    if (capacity >
        std::numeric_limits<std::size_t>::max() / 9u)
      throw std::overflow_error("range arena overflow");
    return capacity * 9u;
  }

  void reserve_winner_workspace(std::size_t count) {
    if (count > winner_workspace_capacity_) {
      winner_workspace_capacity_ = range_arena_records(count);
      winner_workspace_arena_.resize_discard_exact(
          range_arena_bytes(winner_workspace_capacity_));
      std::uint8_t *base = winner_workspace_arena_.data();
      resolve_winner_keys_data_.bind_external(
          reinterpret_cast<std::uint32_t *>(base),
          winner_workspace_capacity_);
      resolve_winner_values_.bind_external(
          reinterpret_cast<std::uint32_t *>(
              base + 4u * winner_workspace_capacity_),
          winner_workspace_capacity_);
      resolve_winner_deltas_.bind_external(
          reinterpret_cast<std::int8_t *>(
              base + 8u * winner_workspace_capacity_),
          winner_workspace_capacity_);
    }
    resolve_winner_keys_data_.resize_discard(count);
    resolve_winner_values_.resize_discard(count);
    resolve_winner_deltas_.resize_discard(count);
  }

  void reserve_resolved_workspace(std::size_t count) {
    if (count > resolved_workspace_capacity_) {
      resolved_workspace_capacity_ = range_arena_records(count);
      resolved_workspace_arena_.resize_discard_exact(
          range_arena_bytes(resolved_workspace_capacity_));
      std::uint8_t *base = resolved_workspace_arena_.data();
      resolved_.keys.bind_external(
          reinterpret_cast<std::uint32_t *>(base),
          resolved_workspace_capacity_);
      resolved_.values.bind_external(
          reinterpret_cast<std::uint32_t *>(
              base + 4u * resolved_workspace_capacity_),
          resolved_workspace_capacity_);
      resolved_.count_delta.bind_external(
          reinterpret_cast<std::int8_t *>(
              base + 8u * resolved_workspace_capacity_),
          resolved_workspace_capacity_);
    }
    resolved_.keys.resize_discard(count);
    resolved_.values.resize_discard(count);
    resolved_.count_delta.resize_discard(count);
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

  void release_build_sort_storage() {
    direct_sort_keys_.release();
    direct_sort_values_.release();
    sort_temp_storage_.release();
    direct_sort_count_ = 0;
    direct_sort_temp_bytes_ = 0;
    run_sort_count_ = 0;
    run_sort_temp_bytes_ = 0;
    scan_u32_count_ = 0;
    scan_u32_temp_bytes_ = 0;
    metadata_scan_temp_bytes_ = 0;
  }

  void prepare_sort_storage(std::size_t direct_count, cudaStream_t stream) {
    direct_sort_keys_.resize_discard(direct_count);
    direct_sort_values_.resize_discard(direct_count);
    RunStorage &sample = available_insert_leaf(direct_count);
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

  void reserve_assignment_gather_storage() {
    compaction_counts_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    compaction_offsets_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    resolve_quotient_lists_.resize_discard_exact(
        4u * gpulsmopt_detail::kEpochQuotients);
    resolve_class_counts_.resize_discard_exact(4u);
    resolve_winner_counts_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    resolve_winner_offsets_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
  }

  static std::uint32_t lookup_owner_position_bits(std::size_t count) {
    std::uint32_t bits = 0u;
    std::size_t capacity = 1u;
    while (capacity < std::max<std::size_t>(count, 1u)) {
      capacity <<= 1u;
      ++bits;
    }
    return bits;
  }

  void reset_lookup_owner(cudaStream_t stream) {
    lookup_owner_leaf_count_ = 0u;
    lookup_owner_published_leaf_count_ = 0u;
    lookup_owner_applied_leaf_count_ = 0u;
    lookup_owner_usable_ = lookup_owner_enabled_;
    if (!lookup_owner_enabled_)
      return;
    CUDA_CHECK(cudaMemsetAsync(
        lookup_owner_slots_.data(), 0,
        lookup_owner_slots_.size() * sizeof(unsigned long long), stream));
    CUDA_CHECK(cudaMemsetAsync(lookup_owner_overflow_.data(), 0,
                               sizeof(std::uint32_t), stream));
  }

  void reserve_lookup_owner_storage(cudaStream_t stream) {
    lookup_owner_position_bits_ =
        lookup_owner_position_bits(batch_capacity_);
    if (lookup_owner_position_bits_ > 22u) {
      lookup_owner_enabled_ = false;
      lookup_owner_usable_ = false;
      return;
    }
    lookup_owner_position_mask_ =
        lookup_owner_position_bits_ == 0u
            ? 0u
            : (std::uint32_t{1} << lookup_owner_position_bits_) - 1u;
    if (max_elements_ >
        std::numeric_limits<std::size_t>::max() / 2u) {
      lookup_owner_enabled_ = false;
      lookup_owner_usable_ = false;
      return;
    }
    const std::size_t required =
        std::max<std::size_t>(2u, max_elements_ * 2u);
    std::size_t slots = 1u;
    while (slots < required &&
           slots < gpulsmopt_detail::kLookupOwnerMaxSlots)
      slots <<= 1u;
    if (slots < required ||
        slots > gpulsmopt_detail::kLookupOwnerMaxSlots) {
      lookup_owner_enabled_ = false;
      lookup_owner_usable_ = false;
      return;
    }
    if (lookup_owner_slots_.capacity() < slots) {
      std::size_t free_bytes = 0u;
      std::size_t total_bytes = 0u;
      CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
      (void)total_bytes;
      const std::size_t bytes =
          slots * sizeof(unsigned long long);
      constexpr std::size_t reserve = std::size_t{256} << 20;
      if (bytes > free_bytes || free_bytes - bytes < reserve) {
        lookup_owner_enabled_ = false;
        lookup_owner_usable_ = false;
        return;
      }
    }
    lookup_owner_slots_.resize_discard_exact(slots);
    lookup_owner_leaves_.resize_discard_exact(
        gpulsmopt_detail::kLookupLeafCapacity);
    lookup_owner_overflow_.resize_discard_exact(1u);
    lookup_owner_table_mask_ = static_cast<std::uint32_t>(slots - 1u);
    lookup_owner_enabled_ = true;
    reset_lookup_owner(stream);
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

  void prepare_hash_storage(HashFoldStorage &storage) {
    storage.offsets.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    storage.counts.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    storage.child_views.resize_discard_exact(
        gpulsmopt_detail::kRawFoldWidth);
    storage.child_router.resize_discard_exact(
        gpulsmopt_detail::kHashRouteWordsPerParent);
  }

  void reserve_hash_storage() {
    while (hash_storage_pool_.size() <
           gpulsmopt_detail::kColdArenaSlots) {
      hash_storage_pool_.emplace_back();
      prepare_hash_storage(hash_storage_pool_.back());
    }
  }

  HashFoldStorage acquire_hash_storage() {
    if (hash_storage_pool_.empty())
      throw std::runtime_error("hash fold capacity exceeded");
    HashFoldStorage storage =
        std::move(hash_storage_pool_.back());
    hash_storage_pool_.pop_back();
    return storage;
  }

  void recycle_hash_storage(RunStorage &run) {
    run.hash_storage.keys.release();
    if (hash_storage_pool_.size() <
        gpulsmopt_detail::kColdArenaSlots) {
      hash_storage_pool_.push_back(
          std::move(run.hash_storage));
    } else {
      run.hash_storage = HashFoldStorage{};
    }
  }

  void reserve_fold_storage(std::size_t batch_count) {
    const std::size_t batch =
        std::max<std::size_t>(batch_count, 1u);
    if (batch > std::numeric_limits<std::uint32_t>::max() /
                    gpulsmopt_detail::kRawFoldWidth)
      throw std::runtime_error("hash fold exceeds 32-bit offsets");
    lookup_host_leaves_.reserve(
        gpulsmopt_detail::kLookupLeafCapacity);
    hash_heavy_capacities_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients + 1u);
    hash_active_quotients_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients);
    hash_heavy_count_.resize_discard_exact(1u);
    hash_route_mask_views_.resize_discard_exact(
        gpulsmopt_detail::kRawFoldWidth);
    reserve_hash_storage();
    lookup_publication_.resize_discard_exact(1u);
    parent_child_views_arena_.resize_discard_exact(
        gpulsmopt_detail::kStableFanout *
        gpulsmopt_detail::kParentSlots);
  }

  void reset_parent_storage() {
    for (int slot = 0; slot < gpulsmopt_detail::kParentSlots;
         ++slot)
      parent_slot_used_[slot] = false;
  }

  void prepare_for_insert(cudaStream_t stream) {
    const std::size_t direct_count =
        std::min(max_elements_,
                 std::max<std::size_t>(1, batch_capacity_));
    reserve_leaf_storage(direct_count, stream);
    assignment_views_.resize_discard_exact(gpulsmopt_detail::kRunCapacity);
    chrono_views_.reserve(gpulsmopt_detail::kRunCapacity);
    prepare_sort_storage(direct_count, stream);
    reserve_assignment_gather_storage();
    reserve_successor_storage();
    reserve_fold_storage(direct_count);
    reset_parent_storage();
    constexpr int lookup_block = 256;
    const auto base = make_base_view();
    gpulsmopt_detail::temporal_lookup_leaf_block_kernel<<<
        1, lookup_block, 0, stream>>>(
        lookup_publication_.data(), 0, base, nullptr, 0,
        nullptr, nullptr);
    gpulsmopt_detail::temporal_lookup_leaf_thread_kernel<<<
        1, lookup_block, 0, stream>>>(
        lookup_publication_.data(), 0, base, nullptr, 0,
        nullptr, nullptr);
    CUDA_CHECK(cudaGetLastError());
    if (direct_count > 0) {
      // Warm-up must not feed uninitialized staging records to CUB. The
      // outputs are discarded, but initcheck correctly treats the reads as
      // undefined behavior.
      CUDA_CHECK(cudaMemsetAsync(direct_sort_keys_.data(), 0,
                                 direct_count * sizeof(std::uint32_t),
                                 stream));
      CUDA_CHECK(cudaMemsetAsync(direct_sort_values_.data(), 0,
                                 direct_count * sizeof(std::uint32_t),
                                 stream));
      RunStorage &sample = available_insert_leaf(direct_count);
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
    qvrf_ready_ = false;
  }

  bool prepare_flat_normalize_leaves(
      const std::vector<std::size_t> &idx,
      std::uint32_t *max_count, cudaStream_t stream) {
    flat_normalize_host_.clear();
    *max_count = 0u;
    std::uint32_t rank_base = 0u;
    for (const std::size_t index : idx) {
      RunStorage &run = runs_[index];
      if (run.grouped)
        return false;
      if (run.hashed) {
        if (run.hash_children.empty())
          return false;
        for (RetainedLeaf &leaf : run.hash_children) {
          const bool insert =
              leaf.operation ==
              gpulsmopt_detail::RunOperation::Insert;
          if (leaf.count >
              std::numeric_limits<std::uint32_t>::max())
            return false;
          flat_normalize_host_.push_back(
              {leaf.keys.data(),
               insert ? leaf.values.data() : nullptr,
               leaf.quotient_off.data(),
               static_cast<std::uint32_t>(leaf.count),
               rank_base,
               static_cast<std::uint8_t>(insert ? 1u : 0u)});
          rank_base += static_cast<std::uint32_t>(leaf.count);
          *max_count = std::max(
              *max_count,
              static_cast<std::uint32_t>(leaf.count));
        }
        continue;
      }
      if (run.mixed ||
          run.count > std::numeric_limits<std::uint32_t>::max())
        return false;
      const bool insert =
          run.operation == gpulsmopt_detail::RunOperation::Insert;
      flat_normalize_host_.push_back(
          {run.keys.data(), insert ? run.values.data() : nullptr,
           run.quotient_off.data(),
           static_cast<std::uint32_t>(run.count),
           rank_base,
           static_cast<std::uint8_t>(insert ? 1u : 0u)});
      rank_base += static_cast<std::uint32_t>(run.count);
      *max_count = std::max(
          *max_count, static_cast<std::uint32_t>(run.count));
    }
    if (flat_normalize_host_.empty() ||
        flat_normalize_host_.size() > 65535u)
      return false;
    flat_normalize_views_.resize_discard(
        flat_normalize_host_.size());
    flat_normalize_bases_.resize_discard(
        flat_normalize_host_.size() *
        gpulsmopt_detail::kEpochQuotients);
    CUDA_CHECK(cudaMemcpyAsync(
        flat_normalize_views_.data(), flat_normalize_host_.data(),
        flat_normalize_host_.size() *
            sizeof(gpulsmopt_detail::FlatAssignmentLeafView),
        cudaMemcpyHostToDevice, stream));
    return true;
  }

  // Gather by quotient and sort only unseen low bits.
  std::size_t normalize_runs_legacy(
      const std::vector<std::size_t> &idx,
      cudaStream_t stream) {
    constexpr int block = 256;
    std::uint32_t flat_max_count = 0u;
    const bool use_flat = prepare_flat_normalize_leaves(
        idx, &flat_max_count, stream);
    const int flat_leaf_count = use_flat
        ? static_cast<int>(flat_normalize_host_.size())
        : 0;
    if (!use_flat) {
      normalize_views_.resize_discard(
          gpulsmopt_detail::kRunCapacity);
      for (std::size_t slot = 0; slot < idx.size(); ++slot)
        host_state_->scratch_views[slot] =
            make_assignment_view(runs_[idx[slot]]);
      CUDA_CHECK(cudaMemcpyAsync(
          normalize_views_.data(), host_state_->scratch_views,
          idx.size() * sizeof(gpulsmopt_detail::AssignmentRunView),
          cudaMemcpyHostToDevice, stream));
    }

    compaction_counts_.resize_discard(
        gpulsmopt_detail::kEpochQuotients + 1u);
    compaction_offsets_.resize_discard(
        gpulsmopt_detail::kEpochQuotients + 1u);
    constexpr int rows = gpulsmopt_detail::kEpochQuotients + 1;
    constexpr int count_grid = (rows + block - 1) / block;
    if (use_flat) {
      gpulsmopt_detail::flat_assignment_count_kernel<<<
          count_grid, block, 0, stream>>>(
          flat_normalize_views_.data(), flat_leaf_count,
          compaction_counts_.data());
    } else {
      gpulsmopt_detail::assignment_group_count_kernel<<<
          count_grid, block, 0, stream>>>(
          normalize_views_.data(), static_cast<int>(idx.size()),
          compaction_counts_.data());
    }
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
    if (use_flat) {
      gpulsmopt_detail::flat_assignment_base_kernel<<<
          count_grid, block, 0, stream>>>(
          flat_normalize_views_.data(), flat_leaf_count,
          compaction_offsets_.data(), flat_normalize_bases_.data());
      CUDA_CHECK(cudaGetLastError());
      const dim3 gather_grid(
          (flat_max_count + block - 1u) / block,
          static_cast<unsigned>(flat_leaf_count));
      gpulsmopt_detail::flat_assignment_gather_kernel<<<
          gather_grid, block, 0, stream>>>(
          flat_normalize_views_.data(), flat_leaf_count,
          flat_max_count, flat_normalize_bases_.data(),
          resolve_keys_.data(), resolve_payload_.data());
      CUDA_CHECK(cudaGetLastError());
    } else {
      gpulsmopt_detail::assignment_group_gather_kernel<<<
          gpulsmopt_detail::kEpochQuotients, block, 0, stream>>>(
          normalize_views_.data(), static_cast<int>(idx.size()),
          compaction_offsets_.data(), resolve_keys_.data(),
          resolve_payload_.data());
      CUDA_CHECK(cudaGetLastError());
    }

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
        make_base_view(), norm_keys_.data(),
        norm_pay_.data());
    CUDA_CHECK(cudaGetLastError());
    return total;
  }

  std::size_t normalize_runs(
      const std::vector<std::size_t> &idx,
      cudaStream_t stream) {
    constexpr int block = 256;
    last_normalize_fused_ = false;
    std::uint32_t max_leaf_count = 0u;
    const bool flat = prepare_flat_normalize_leaves(
        idx, &max_leaf_count, stream);
    const std::size_t leaf_count =
        flat_normalize_host_.size();
    if (!flat || leaf_count > static_cast<std::size_t>(
                                  gpulsmopt_detail::kLookupLeafCapacity))
      return normalize_runs_legacy(idx, stream);

    std::size_t total = 0u;
    for (const auto &leaf : flat_normalize_host_)
      total += leaf.count;
    if (total == 0u)
      return 0u;
    if (total > static_cast<std::size_t>(
                    std::numeric_limits<int>::max()))
      throw std::runtime_error("resolved cache exceeds local limits");

    constexpr int rows = gpulsmopt_detail::kEpochQuotients + 1;
    constexpr int count_grid = (rows + block - 1) / block;
    compaction_counts_.resize_discard(rows);
    compaction_offsets_.resize_discard(rows);
    gpulsmopt_detail::flat_assignment_count_kernel<<<
        count_grid, block, 0, stream>>>(
        flat_normalize_views_.data(),
        static_cast<int>(leaf_count),
        compaction_counts_.data());
    CUDA_CHECK(cudaGetLastError());
    exclusive_scan_u32(
        compaction_counts_.data(), compaction_offsets_.data(),
        rows, stream);

    CUDA_CHECK(cudaMemsetAsync(
        resolve_class_counts_.data(), 0,
        4u * sizeof(std::uint32_t), stream));
    gpulsmopt_detail::resolve_classify_kernel<<<
        count_grid, block, 0, stream>>>(
        compaction_counts_.data(),
        resolve_quotient_lists_.data(),
        resolve_class_counts_.data(),
        resolve_winner_counts_.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(
        host_state_->resolve_class_counts,
        resolve_class_counts_.data(),
        4u * sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::uint32_t small =
        host_state_->resolve_class_counts[0];
    const std::uint32_t medium =
        host_state_->resolve_class_counts[1];
    const std::uint32_t large =
        host_state_->resolve_class_counts[2];
    if (host_state_->resolve_class_counts[3] != 0u)
      return normalize_runs_legacy(idx, stream);

    reserve_winner_workspace(total);
    const bool keep_zero = resolved_.count != 0u;
    auto *lists = resolve_quotient_lists_.data();
    if (keep_zero) {
      if (small != 0u)
        gpulsmopt_detail::quotient_local_resolve_kernel<
            32, 4, true, false><<<small, 32, 0, stream>>>(
            flat_normalize_views_.data(),
            static_cast<int>(leaf_count), lists,
            compaction_counts_.data(),
            compaction_offsets_.data(), make_base_view(),
            resolve_winner_keys_data_.data(),
            resolve_winner_values_.data(),
            resolve_winner_deltas_.data(),
            resolve_winner_counts_.data());
      if (medium != 0u)
        gpulsmopt_detail::quotient_local_resolve_kernel<
            128, 4, true, true><<<medium, 128, 0, stream>>>(
            flat_normalize_views_.data(),
            static_cast<int>(leaf_count),
            lists + gpulsmopt_detail::kEpochQuotients,
            compaction_counts_.data(),
            compaction_offsets_.data(), make_base_view(),
            resolve_winner_keys_data_.data(),
            resolve_winner_values_.data(),
            resolve_winner_deltas_.data(),
            resolve_winner_counts_.data());
      if (large != 0u)
        gpulsmopt_detail::quotient_local_resolve_kernel<
            256, 8, true, true><<<large, 256, 0, stream>>>(
            flat_normalize_views_.data(),
            static_cast<int>(leaf_count),
            lists + 2u * gpulsmopt_detail::kEpochQuotients,
            compaction_counts_.data(),
            compaction_offsets_.data(), make_base_view(),
            resolve_winner_keys_data_.data(),
            resolve_winner_values_.data(),
            resolve_winner_deltas_.data(),
            resolve_winner_counts_.data());
    } else {
      if (small != 0u)
        gpulsmopt_detail::quotient_local_resolve_kernel<
            32, 4, false, false><<<small, 32, 0, stream>>>(
            flat_normalize_views_.data(),
            static_cast<int>(leaf_count), lists,
            compaction_counts_.data(),
            compaction_offsets_.data(), make_base_view(),
            resolve_winner_keys_data_.data(),
            resolve_winner_values_.data(),
            resolve_winner_deltas_.data(),
            resolve_winner_counts_.data());
      if (medium != 0u)
        gpulsmopt_detail::quotient_local_resolve_kernel<
            128, 4, false, true><<<medium, 128, 0, stream>>>(
            flat_normalize_views_.data(),
            static_cast<int>(leaf_count),
            lists + gpulsmopt_detail::kEpochQuotients,
            compaction_counts_.data(),
            compaction_offsets_.data(), make_base_view(),
            resolve_winner_keys_data_.data(),
            resolve_winner_values_.data(),
            resolve_winner_deltas_.data(),
            resolve_winner_counts_.data());
      if (large != 0u)
        gpulsmopt_detail::quotient_local_resolve_kernel<
            256, 8, false, true><<<large, 256, 0, stream>>>(
            flat_normalize_views_.data(),
            static_cast<int>(leaf_count),
            lists + 2u * gpulsmopt_detail::kEpochQuotients,
            compaction_counts_.data(),
            compaction_offsets_.data(), make_base_view(),
            resolve_winner_keys_data_.data(),
            resolve_winner_values_.data(),
            resolve_winner_deltas_.data(),
            resolve_winner_counts_.data());
    }
    CUDA_CHECK(cudaGetLastError());

    exclusive_scan_u32(
        resolve_winner_counts_.data(),
        resolve_winner_offsets_.data(), rows, stream);
    CUDA_CHECK(cudaMemcpyAsync(
        &host_state_->fused_winner_count,
        resolve_winner_offsets_.data() +
            gpulsmopt_detail::kEpochQuotients,
        sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::uint32_t winners =
        host_state_->fused_winner_count;
    last_normalize_fused_ = true;
    if (resolved_.count != 0u && winners != 0u) {
      norm_keys_.resize_discard(winners);
      norm_pay_.resize_discard(winners);
      gpulsmopt_detail::fused_resolve_pack_kernel<<<
          gpulsmopt_detail::kEpochQuotients,
          block, 0, stream>>>(
          compaction_offsets_.data(),
          resolve_winner_counts_.data(),
          resolve_winner_offsets_.data(),
          resolve_winner_keys_data_.data(),
          resolve_winner_values_.data(),
          resolve_winner_deltas_.data(),
          norm_keys_.data(), norm_pay_.data());
      CUDA_CHECK(cudaGetLastError());
    }
    return winners;
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
    reserve_resolved_workspace(changed);
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
    qvrf_ready_ = false;
    resolved_ready_ = true;
  }

  // Extend the cache only through unseen run sequences.
  void ensure_resolved(cudaStream_t stream) {
    // BaseRun changed, so rebuild from surviving runs.
    if (resolved_base_generation_ != base_generation_) {
      resolved_.count = 0;
      resolved_through_sequence_ = 0;
      resolved_base_generation_ = base_generation_;
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
    const std::size_t total =
        (nidx.empty() || upper == 0) ? 0 : normalize_runs(nidx, stream);
    if (total == 0) {
      resolved_through_sequence_ = run_sequence_;
      if (!resolved_ready_) {
        build_assignment_offsets(resolved_, resolved_.count, stream);
        qvrf_ready_ = false;
        resolved_ready_ = true;
      }
      return;
    }
    if (last_normalize_fused_ && resolved_.count == 0u) {
      const std::uint32_t changed =
          static_cast<std::uint32_t>(total);
      resolved_.count = changed;
      reserve_resolved_workspace(changed);
      resolved_.quotient_off.resize_discard_exact(
          gpulsmopt_detail::kEpochQuotients + 1u);
      gpulsmopt_detail::fused_resolve_unpack_kernel<<<
          gpulsmopt_detail::kEpochQuotients,
          256, 0, stream>>>(
          compaction_offsets_.data(),
          resolve_winner_counts_.data(),
          resolve_winner_offsets_.data(),
          resolve_winner_keys_data_.data(),
          resolve_winner_values_.data(),
          resolve_winner_deltas_.data(),
          resolved_.keys.data(), resolved_.values.data(),
          resolved_.count_delta.data(),
          resolved_.quotient_off.data());
      CUDA_CHECK(cudaGetLastError());
      qvrf_ready_ = false;
      resolved_ready_ = true;
      resolved_through_sequence_ = run_sequence_;
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
    if (updates.empty())
      return;
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
    const std::size_t capacity_total = base_count + assignment_total;
    if (capacity_total == 0)
      return;
    if (capacity_total >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("full consolidation exceeds CUB limits");

    resolve_keys_.resize_discard(capacity_total);
    resolve_payload_.resize_discard(capacity_total);
    resolve_alt_keys_.resize_discard(capacity_total);
    resolve_alt_payload_.resize_discard(capacity_total);
    resolve_flags_.resize_discard(capacity_total);
    resolve_sel_vdelta_.resize_discard(capacity_total);
    resolve_count_.resize_discard(1);
    constexpr int block = 256;

    if (base_count > 0) {
      const int base_grid =
          static_cast<int>((base_count + block - 1u) / block);
      gpulsmopt_detail::resolve_pack_base_kernel<<<
          base_grid, block, 0, stream>>>(
          make_base_view(), resolve_keys_.data(),
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
    CUDA_CHECK(cudaMemcpyAsync(
        &host_state_->gathered_count,
        compaction_offsets_.data() + gpulsmopt_detail::kEpochQuotients,
        sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::size_t total =
        base_count + static_cast<std::size_t>(host_state_->gathered_count);
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
    for (auto &run : runs_) {
      if (run.hashed) {
        release_hash_children(run);
        recycle_hash_storage(run);
      }
      if (run.grouped && run.parent_slot >= 0 &&
          run.parent_slot < gpulsmopt_detail::kParentSlots)
        parent_slot_used_[run.parent_slot] = false;
      recycle_run_storage(std::move(run));
    }
    runs_.clear();
    acquire_compaction_slot();
    RunStorage &base = runs_.back();
    base.count = 0;
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
    base.hashed = false;
    base.grouped = false;
    base.nested = false;
    base.group_child_count = 0;
    base.parent_slot = -1;
    base.owner_leaf_id = gpulsmopt_detail::kNoOwnerLeaf;
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
    base.count = kept;
    base.keys.resize_discard(kept);
    base.values.resize_discard(kept);
    build_sorted_run_cache(0u, stream);
    run_sequence_ = 0;
    chrono_views_.clear();
    views_dirty_ = false;
    invalidate_resolved();
    succ_sparse_ready_ = false;
    reset_parent_storage();
    reset_lookup_owner(stream);
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

    HashFoldStorage hash_storage = acquire_hash_storage();
    std::uint64_t sequence_begin = ~std::uint64_t{0};
    std::uint64_t sequence_end = 0u;
    std::size_t input_records = 0u;
    int missing_route_masks = 0;
    for (std::size_t child = 0; child < raw.size(); ++child) {
      RunStorage &run = runs_[raw[child]];
      sequence_begin = std::min(sequence_begin, run.sequence_begin);
      sequence_end = std::max(sequence_end, run.sequence_end);
      input_records += run.count;
      host_state_->scratch_views[child] =
          make_assignment_view(run);
      missing_route_masks += run.route_mask_ready ? 0 : 1;
    }
    const bool use_route_masks =
        missing_route_masks <=
        gpulsmopt_detail::kHashRouteMaskBuildMax;
    if (use_route_masks) {
      for (std::size_t child = 0; child < raw.size(); ++child) {
        RunStorage &run = runs_[raw[child]];
        run.route_mask.resize_discard_exact(
            gpulsmopt_detail::kEpochQuotients);
        host_state_->route_masks[child] = run.route_mask.data();
      }
    }

    auto *child_views = hash_storage.child_views.data();
    {
      GPULSMOPT_FOLD_PHASE(prof_hash_publish_ms);
      CUDA_CHECK(cudaMemcpyAsync(
          child_views, host_state_->scratch_views,
          raw.size() * sizeof(gpulsmopt_detail::AssignmentRunView),
          cudaMemcpyHostToDevice, stream));
      if (use_route_masks) {
        CUDA_CHECK(cudaMemcpyAsync(
            hash_route_mask_views_.data(), host_state_->route_masks,
            raw.size() * sizeof(std::uint64_t *),
            cudaMemcpyHostToDevice, stream));
      }
    }
    std::uint32_t *counts = hash_storage.counts.data();
    std::uint32_t *heavy_offsets = hash_storage.offsets.data();
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
    }
    {
      GPULSMOPT_FOLD_PHASE(prof_hash_scan_ms);
      exclusive_scan_u32(
          hash_heavy_capacities_.data(), heavy_offsets,
          gpulsmopt_detail::kEpochQuotients + 1u, stream);
      CUDA_CHECK(cudaMemcpyAsync(
          &host_state_->hash_heavy_count,
          hash_heavy_count_.data(), sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          &host_state_->hash_capacity,
          heavy_offsets + gpulsmopt_detail::kEpochQuotients,
          sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const std::uint32_t heavy_count =
        host_state_->hash_heavy_count;
    const std::uint32_t hash_capacity =
        host_state_->hash_capacity;
    hash_storage.keys.resize_discard_exact(hash_capacity);
    {
      GPULSMOPT_FOLD_PHASE(prof_hash_build_ms);
      if (heavy_count != 0u) {
        gpulsmopt_detail::temporal_hash_heavy_kernel<<<
            heavy_count, block, 0, stream>>>(
            child_views, static_cast<int>(raw.size()),
            hash_active_quotients_.data(), heavy_offsets,
            counts, hash_capacity, hash_storage.keys.data());
        CUDA_CHECK(cudaGetLastError());
      }
      if (use_route_masks) {
        constexpr int mask_block = 256;
        constexpr int mask_warps = mask_block / 32;
        constexpr int mask_grid =
            (gpulsmopt_detail::kEpochQuotients +
             mask_warps - 1) /
            mask_warps;
        for (const std::size_t index : raw) {
          RunStorage &run = runs_[index];
          if (run.route_mask_ready)
            continue;
          gpulsmopt_detail::raw_quotient_masks_kernel<false><<<
              mask_grid, mask_block, 0, stream>>>(
              run.keys.data(), run.quotient_off.data(), nullptr,
              run.route_mask.data());
          CUDA_CHECK(cudaGetLastError());
          run.route_mask_ready = true;
        }
        gpulsmopt_detail::temporal_hash_route_mask_kernel<<<
            gpulsmopt_detail::kEpochQuotients, 64, 0, stream>>>(
            hash_route_mask_views_.data(),
            static_cast<int>(raw.size()), counts,
            hash_storage.child_router.data());
      } else {
        gpulsmopt_detail::temporal_hash_route_kernel<<<
            gpulsmopt_detail::kEpochQuotients, 128, 0, stream>>>(
            child_views, static_cast<int>(raw.size()), counts,
            hash_storage.child_router.data());
      }
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
    folded.hashed = true;
    folded.grouped = false;
    folded.nested = false;
    folded.mixed = true;
    folded.operation = gpulsmopt_detail::RunOperation::Insert;
    folded.sequence_begin = sequence_begin;
    folded.sequence_end = sequence_end;
    folded.count = input_records;
    folded.unique_keys = false;
    folded.fully_sorted = false;
    folded.unit_counts = false;
    folded.hash_storage = std::move(hash_storage);
    folded.hash_children = acquire_hash_children();
    for (std::size_t child_slot = 0;
         child_slot < raw.size(); ++child_slot) {
      const std::size_t index = raw[child_slot];
      RunStorage &source = runs_[index];
      RetainedLeaf &child =
          folded.hash_children[child_slot];
      child.keys = std::move(source.keys);
      child.values = std::move(source.values);
      child.quotient_off = std::move(source.quotient_off);
      child.op_words = std::move(source.op_words);
      child.lookup_mask = std::move(source.lookup_mask);
      child.route_mask = std::move(source.route_mask);
      child.count = source.count;
      child.sequence = source.sequence_end;
      child.operation = source.operation;
      child.lookup_mask_ready = source.lookup_mask_ready;
      child.route_mask_ready = source.route_mask_ready;
      child.owner_leaf_id = source.owner_leaf_id;
    }
    std::sort(raw.begin(), raw.end());
    for (auto it = raw.rbegin(); it != raw.rend(); ++it) {
      if (sorted_run_ready() && sorted_run_index_ > *it)
        --sorted_run_index_;
      runs_.erase(runs_.begin() + static_cast<std::ptrdiff_t>(*it));
    }
    runs_.push_back(std::move(folded));
    maintain_leaf_spare(
        std::max<std::size_t>(batch_capacity_, 1u), stream);
    invalidate_resolved();
    succ_sparse_ready_ = false;
    ++maintenance_stats_.hash_fold_count;
    maintenance_stats_.hash_fold_input_records += input_records;
    ++maintenance_stats_.compaction_count;
    maintenance_stats_.compacted_input_records += input_records;
    maintenance_stats_.compacted_output_records += input_records;
    carry_cold_run(stream);
    rebuild_chrono_views(stream);
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
  void carry_cold_run(cudaStream_t stream) {
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
  bool last_normalize_fused_ = false;
  // Incremental cache and merge scratch.
  std::uint64_t resolved_through_sequence_ = 0;
  std::uint64_t resolved_base_generation_ = ~std::uint64_t{0};
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::AssignmentRunView>
      normalize_views_;
  gpulsmopt_detail::RawDeviceBuffer<
      gpulsmopt_detail::FlatAssignmentLeafView>
      flat_normalize_views_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      flat_normalize_bases_;
  std::vector<gpulsmopt_detail::FlatAssignmentLeafView>
      flat_normalize_host_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> norm_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> norm_pay_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      resolve_winner_keys_data_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      resolve_winner_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::int8_t>
      resolve_winner_deltas_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t>
      winner_workspace_arena_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t>
      resolved_workspace_arena_;
  std::size_t winner_workspace_capacity_ = 0u;
  std::size_t resolved_workspace_capacity_ = 0u;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> cache_pay_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> merge_out_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> merge_out_pay_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> merge_flags_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> merge_sel_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> merge_sel_pay_;
  std::size_t resolved_sort_count_ = 0;
  std::size_t resolved_sort_temp_bytes_ = 0;
  // Lazy successor sidecar.
  bool succ_sparse_ready_ = false;
  std::uint64_t succ_sparse_base_generation_ = ~std::uint64_t{0};
  std::uint64_t succ_sparse_run_sequence_ = 0;
  std::uint32_t succ_sparse_l0_words_ = 0;
  std::uint32_t succ_sparse_l3_words_ = 0;
  std::uint32_t succ_sparse_positive_l0_words_ = 0;
  std::uint32_t succ_sparse_positive_l3_words_ = 0;
  std::uint64_t base_generation_ = 0;
  // Hash-fold scratch and one ready spare.
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      hash_heavy_capacities_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      hash_active_quotients_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> hash_heavy_count_;
  gpulsmopt_detail::RawDeviceBuffer<const std::uint64_t *>
      hash_route_mask_views_;
  std::vector<HashFoldStorage> hash_storage_pool_;
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::AssignmentRunView>
      parent_child_views_arena_;
  bool parent_slot_used_[gpulsmopt_detail::kParentSlots] = {};
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::AssignmentRunView>
      assignment_views_;
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::LookupPublication>
      lookup_publication_;
  gpulsmopt_detail::RawDeviceBuffer<unsigned long long>
      lookup_owner_slots_;
  gpulsmopt_detail::RawDeviceBuffer<
      gpulsmopt_detail::LookupOwnerLeafView>
      lookup_owner_leaves_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      lookup_owner_overflow_;
  std::uint32_t lookup_owner_table_mask_ = 0u;
  std::uint32_t lookup_owner_position_bits_ = 0u;
  std::uint32_t lookup_owner_position_mask_ = 0u;
  std::uint32_t lookup_owner_leaf_count_ = 0u;
  std::uint32_t lookup_owner_published_leaf_count_ = 0u;
  std::uint32_t lookup_owner_applied_leaf_count_ = 0u;
  bool lookup_owner_enabled_ = false;
  bool lookup_owner_usable_ = false;
  // Oldest-to-newest descriptor mirror.
  std::vector<gpulsmopt_detail::AssignmentRunView> chrono_views_;
  std::vector<HostLookupLeaf> lookup_host_leaves_;
  // Set when a run is published; cleared when the device descriptor array is
  // batch-synced before a read (flush_pending_views).
  bool views_dirty_ = false;
  bool lookup_publication_ready_ = false;
  int lookup_leaf_count_ = 0;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> resolve_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> resolve_payload_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> resolve_alt_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> resolve_alt_payload_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> resolve_flags_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> resolve_sel_vdelta_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> resolve_count_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      resolve_quotient_lists_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      resolve_class_counts_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      resolve_winner_counts_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      resolve_winner_offsets_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compaction_counts_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compaction_offsets_;

#ifdef GPULSMOPT_PROFILE_INSERT
  double prof_delta_sort_ms_ = 0.0;
  double prof_delta_ingest_ms_ = 0.0;
  void reset_insert_prof_() {
    prof_delta_sort_ms_ = prof_delta_ingest_ms_ = 0.0;
  }
#endif
  std::vector<RunStorage> runs_;
  std::vector<RunStorage> run_pool_;
  std::vector<RunStorage> insert_leaf_pool_;
  std::vector<RunStorage> delete_leaf_pool_;
  std::vector<LeafSlab> leaf_slabs_;
  LeafSlab pending_leaf_slab_;
  std::size_t pending_leaf_slab_bytes_ = 0;
  bool pending_leaf_slab_valid_ = false;
  std::vector<std::vector<RetainedLeaf>> hash_child_pool_;

  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> direct_sort_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> direct_sort_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> sort_temp_storage_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> base_rank23_;
  // Lazy Quotient-Visible Range Forest. Base blocks are packed within each
  // upper-16 quotient; current resolved cells shadow their complete rank23
  // slices and materialize visible rows only after a range is requested.
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      qvrf_base_block_counts_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      qvrf_base_q_block_offsets_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      qvrf_base_block_begins_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint16_t>
      qvrf_base_block_sizes_;
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::QvrfSummary>
      qvrf_base_block_summaries_;
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::QvrfSummary>
      qvrf_base_q_summaries_;
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::QvrfSummary>
      qvrf_current_q_summaries_;
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::QvrfSummary>
      qvrf_top_l1_;
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::QvrfSummary>
      qvrf_top_l2_;
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::QvrfSummary>
      qvrf_top_l3_;
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::QvrfSummary>
      qvrf_top_l4_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> qvrf_cell_ids_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> qvrf_cell_counts_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> qvrf_cell_offsets_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> qvrf_cell_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> qvrf_cell_values_;
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::QvrfSummary>
      qvrf_cell_summaries_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> qvrf_q_cell_offsets_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> qvrf_report_counts_;
  bool qvrf_base_ready_ = false;
  bool qvrf_ready_ = false;
  std::uint64_t qvrf_base_generation_ = ~std::uint64_t{0};
  std::uint64_t qvrf_run_sequence_ = 0u;
  std::uint32_t qvrf_cell_count_ = 0u;
  bool qvrf_materialized_cells_ = false;
  gpulsmopt_detail::PinnedHostState *host_state_ = nullptr;
  cudaEvent_t stream_handoff_ = nullptr;
  cudaEvent_t leaf_alloc_ready_ = nullptr;
  cudaStream_t leaf_alloc_stream_ = nullptr;
  std::thread leaf_alloc_worker_;
  std::mutex leaf_alloc_mutex_;
  std::condition_variable leaf_alloc_cv_;
  std::uint8_t *leaf_alloc_result_ = nullptr;
  std::size_t leaf_alloc_request_bytes_ = 0;
  cudaError_t leaf_alloc_status_ = cudaSuccess;
  int leaf_alloc_device_ = 0;
  bool leaf_alloc_stop_ = false;
  bool leaf_alloc_request_ = false;
  bool leaf_alloc_result_ready_ = false;
  cudaStream_t operation_stream_ = nullptr;
  bool operation_stream_valid_ = false;
  std::size_t direct_sort_count_ = 0;
  std::size_t direct_sort_temp_bytes_ = 0;
  std::size_t run_sort_count_ = 0;
  std::size_t run_sort_temp_bytes_ = 0;
  std::size_t scan_u32_count_ = 0;
  std::size_t scan_u32_temp_bytes_ = 0;
  std::size_t metadata_scan_temp_bytes_ = 0;

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
