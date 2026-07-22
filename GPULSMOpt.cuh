#pragma once
#include "gpu_dictionary_adapter.cuh"

#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_merge.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
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

#ifndef GPULSMOPT_C0_FLUSH_BUDGET
#define GPULSMOPT_C0_FLUSH_BUDGET (1 << 19)
#endif
// Large batches bypass C0 and become runs.
#ifndef GPULSMOPT_SCATTER_MIN_BATCH
#define GPULSMOPT_SCATTER_MIN_BATCH (1 << 18)
#endif
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
constexpr int kRunCapacity = 128;
// Assignment runs merged per contiguous compaction.
constexpr int kCompactGroup = 64;
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
constexpr int kOwnerSlotBits = 11;
constexpr int kOwnerSlots = 1 << kOwnerSlotBits;
constexpr int kOwnerSlotMask = kOwnerSlots - 1;
constexpr std::uint32_t kOwnerNoPage = 0xffffffffu;
constexpr int kRunStride = kRunCapacity;
constexpr int kGpuResidentWarps = 9024;
// Flat BaseRun rank directory.
constexpr int kBaseRank23Bits = 23;
constexpr int kBaseRank23Shift = 32 - kBaseRank23Bits;
constexpr std::size_t kBaseRank23Size = std::size_t{1} << kBaseRank23Bits;
constexpr std::size_t kSortedRunMinRecords = 1u << 22;
constexpr int kRangeProjectionBits = 20;
constexpr int kRangeProjectionBins = 1 << kRangeProjectionBits;
constexpr int kRangeProjectionShift = 32 - kRangeProjectionBits;
constexpr std::size_t kRangeProjectionMinQueries = 1 << 18;
constexpr std::uint64_t kRangeCdfMaxRatio = GPULSMOPT_RANGE_CDF_MAX_RATIO;
static_assert(GPULSMOPT_RADIX_THREADS % 32 == 0,
              "radix block size must be warp aligned");
static_assert(kRunCapacity == 128, "run kernels require 128 physical slots");
constexpr std::uint32_t kEmptyKey = std::numeric_limits<std::uint32_t>::max();

// Successor bitmap: L0 is one bit per 32-bit key.
__host__ __device__ inline std::uint32_t succ_level_words(int level) {
  return level == 0   ? 1u << 27
         : level == 1 ? 1u << 22
         : level == 2 ? 1u << 17
         : level == 3 ? 1u << 12
         : level == 4 ? 1u << 7
                      : 4u;
}

__host__ __device__ inline std::uint32_t succ_level_off(int level) {
  std::uint32_t off = 0u;
  for (int l = 0; l < level; ++l)
    off += succ_level_words(l);
  return off;
}

// Sum of all six level sizes (~528.5 MiB of bits).
constexpr std::size_t kSuccTotalWords =
    (std::size_t{1} << 27) + (std::size_t{1} << 22) + (std::size_t{1} << 17) +
    (std::size_t{1} << 12) + (std::size_t{1} << 7) + 4u;
constexpr std::uint32_t kInsert = 1;
constexpr std::uint32_t kTombstone = 0;
constexpr std::uint32_t kC0LogMaxIndex = 0x00fffffeu;

// Immutable assignment runs replace the owner-transition path.
enum class RunOperation : std::uint8_t { Insert, Delete };
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
  gpulsmopt_detail::ScopedInsertPhaseTimer GPULSMOPT_PROF_CAT(                 \
      prof_phase_, __LINE__)(stream, &(acc))
#else
#define GPULSMOPT_PROF_PHASE(acc)                                              \
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
  const std::uint32_t *values;
  const std::int8_t *count_delta;
  const std::uint32_t *quotient_off;
  const std::uint32_t *subgroup_masks;
  std::uint32_t *quotient_live;
  std::uint32_t *quotient_count_sum;
  std::uint32_t *quotient_value_sum;
  const std::uint32_t *quotient_count_prefix;
  const std::uint32_t *quotient_value_prefix;
  const std::uint32_t *subgroup_value_prefix;
  std::uint32_t *heavy_list;
  std::uint32_t *heavy_count;
  std::uint32_t fully_sorted;
};

struct SortedRunView {
  const std::uint32_t *keys;
  const std::uint32_t *values;
  const std::uint32_t *rank23; // flat directory, 2^23 + 1 entries
  std::size_t count;
  std::uint32_t unit_counts;
};

struct SortedRunRangeView {
  const std::uint32_t *cdf;
  std::uint32_t min_key;
  std::uint64_t span;
};

constexpr std::uint32_t kSortedRunCache = 0u;
constexpr std::uint32_t kQuotientRun = 1u;

struct LogicalRunView {
  SortedRunView sorted;
  SortedRunRangeView sorted_range;
  const std::uint32_t *sorted_value_prefix;
  const std::uint32_t *sorted_count_prefix;
  RunView delta;
  std::uint32_t kind;
};

struct RangeDeltaView {
  const std::uint32_t *offsets;
  const std::uint32_t *value_prefix;
  const std::uint32_t *keys;
  const std::uint32_t *values;
};

struct OwnerView {
  std::uint64_t *primary;
  std::uint64_t *spill_keys;
  std::uint32_t *spill_values;
  std::uint32_t *spill_count;
  std::uint32_t *quotient_live;
  std::uint64_t spill_mask;
};

constexpr std::uint32_t kOwnerEmpty = 0u;
constexpr std::uint32_t kOwnerLive = 1u;
constexpr std::uint32_t kOwnerTomb = 2u;
// Terminal insertion failure: publish no delta.
constexpr std::uint32_t kOwnerFail = 3u;
constexpr std::uint64_t kSpillEmpty = 0u;
constexpr std::uint64_t kSpillTomb = 1u;
constexpr std::uint64_t kSpillLock = std::uint64_t{1} << 63;
__host__ __device__ inline std::uint64_t
owner_pack(std::uint32_t low, std::uint32_t value, std::uint32_t state) {
  return static_cast<std::uint64_t>(value) |
         (static_cast<std::uint64_t>(low) << 32) |
         (static_cast<std::uint64_t>(state) << 62);
}

__host__ __device__ inline std::uint32_t owner_state(std::uint64_t slot) {
  return static_cast<std::uint32_t>(slot >> 62);
}

__host__ __device__ inline std::uint32_t owner_low(std::uint64_t slot) {
  return static_cast<std::uint32_t>((slot >> 32) & 0xffffu);
}

__host__ __device__ inline std::uint32_t owner_value(std::uint64_t slot) {
  return static_cast<std::uint32_t>(slot);
}

__host__ __device__ inline std::uint32_t owner_slot(std::uint32_t low) {
  return (low * 0x9e3779b1u) >> (32 - kOwnerSlotBits);
}

__host__ __device__ inline std::uint64_t spill_token(std::uint32_t key) {
  return static_cast<std::uint64_t>(key) + 2u;
}

__host__ __device__ inline std::uint64_t
spill_slot(std::uint32_t key, std::uint64_t mask) {
  std::uint64_t x = key;
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x & mask;
}

__device__ inline void owner_add_live(OwnerView owner, std::uint32_t quotient,
                                      std::int32_t delta) {
  const std::int64_t live =
      static_cast<std::int64_t>(owner.quotient_live[quotient]) + delta;
  owner.quotient_live[quotient] = static_cast<std::uint32_t>(live);
}

__device__ inline bool owner_find_in_page(const std::uint64_t *page,
                                          std::uint32_t low,
                                          std::uint64_t *found) {
  std::uint32_t slot = owner_slot(low);
  for (int probe = 0; probe < kOwnerSlots; ++probe) {
    const std::uint64_t packed = page[slot];
    const std::uint32_t state = owner_state(packed);
    if (state == kOwnerEmpty)
      return false;
    if (owner_low(packed) == low) {
      *found = packed;
      return true;
    }
    slot = (slot + 1u) & kOwnerSlotMask;
  }
  return false;
}

// Update the value of a key already resident in spill.
__device__ inline bool spill_try_update(OwnerView owner, std::uint32_t key,
                                        std::uint32_t value,
                                        std::uint64_t *previous) {
  const std::uint32_t low = key & 0xffffu;
  const std::uint64_t token = spill_token(key);
  std::uint64_t slot = spill_slot(key, owner.spill_mask);
  for (std::uint64_t probe = 0; probe <= owner.spill_mask; ++probe) {
    std::uint64_t state = *reinterpret_cast<volatile const std::uint64_t *>(
        owner.spill_keys + slot);
    if (state == kSpillEmpty)
      return false;
    if ((state & ~kSpillLock) == token) {
      while (*reinterpret_cast<volatile const std::uint64_t *>(
                 owner.spill_keys + slot) == (token | kSpillLock)) {
      }
      const std::uint32_t old = atomicExch(owner.spill_values + slot, value);
      *previous = owner_pack(low, old, kOwnerLive);
      return true;
    }
    slot = (slot + 1u) & owner.spill_mask;
  }
  return false;
}

// Insert a key proven absent; claims empty or tomb slots.
__device__ inline bool spill_insert_fresh(OwnerView owner, std::uint32_t key,
                                          std::uint32_t value) {
  const std::uint32_t quotient = key >> kEpochQuotientBits;
  const std::uint64_t token = spill_token(key);
  for (;;) {
    std::uint64_t slot = spill_slot(key, owner.spill_mask);
    std::uint64_t reusable = ~std::uint64_t{0};
    bool retry = false;
    for (std::uint64_t probe = 0; probe <= owner.spill_mask; ++probe) {
      const std::uint64_t state =
          *reinterpret_cast<volatile const std::uint64_t *>(owner.spill_keys +
                                                            slot);
      if (state == kSpillEmpty) {
        const std::uint64_t target =
            reusable != ~std::uint64_t{0} ? reusable : slot;
        const std::uint64_t expected =
            reusable != ~std::uint64_t{0} ? kSpillTomb : kSpillEmpty;
        const std::uint64_t old = atomicCAS(
            reinterpret_cast<unsigned long long *>(owner.spill_keys + target),
            expected, token | kSpillLock);
        if (old == expected) {
          owner.spill_values[target] = value;
          __threadfence();
          *reinterpret_cast<volatile std::uint64_t *>(owner.spill_keys +
                                                      target) = token;
          atomicAdd(owner.spill_count + quotient, 1u);
          return true;
        }
        retry = true;
        break;
      }
      if (state == kSpillTomb && reusable == ~std::uint64_t{0})
        reusable = slot;
      slot = (slot + 1u) & owner.spill_mask;
    }
    if (retry)
      continue;
    if (reusable != ~std::uint64_t{0}) {
      const std::uint64_t old = atomicCAS(
          reinterpret_cast<unsigned long long *>(owner.spill_keys + reusable),
          kSpillTomb, token | kSpillLock);
      if (old == kSpillTomb) {
        owner.spill_values[reusable] = value;
        __threadfence();
        *reinterpret_cast<volatile std::uint64_t *>(owner.spill_keys +
                                                    reusable) = token;
        atomicAdd(owner.spill_count + quotient, 1u);
        return true;
      }
      continue;
    }
    return false;
  }
}

// Remove a spilled key; returns its packed live entry or 0.
__device__ inline std::uint64_t spill_erase(OwnerView owner,
                                            std::uint32_t key) {
  const std::uint32_t quotient = key >> kEpochQuotientBits;
  const std::uint32_t low = key & 0xffffu;
  const std::uint64_t token = spill_token(key);
  std::uint64_t slot = spill_slot(key, owner.spill_mask);
  for (std::uint64_t probe = 0; probe <= owner.spill_mask; ++probe) {
    const std::uint64_t state =
        *reinterpret_cast<volatile const std::uint64_t *>(owner.spill_keys +
                                                          slot);
    if (state == kSpillEmpty)
      return 0u;
    if ((state & ~kSpillLock) == token) {
      while (*reinterpret_cast<volatile const std::uint64_t *>(
                 owner.spill_keys + slot) == (token | kSpillLock)) {
      }
      const std::uint32_t value = owner.spill_values[slot];
      atomicExch(reinterpret_cast<unsigned long long *>(owner.spill_keys +
                                                        slot),
                 kSpillTomb);
      atomicSub(owner.spill_count + quotient, 1u);
      return owner_pack(low, value, kOwnerLive);
    }
    slot = (slot + 1u) & owner.spill_mask;
  }
  return 0u;
}

__device__ inline bool owner_find(OwnerView owner, std::uint32_t key,
                                  std::uint64_t *found) {
  const std::uint32_t quotient = key >> kEpochQuotientBits;
  const std::uint32_t low = key & 0xffffu;
  const std::uint64_t *page =
      owner.primary + static_cast<std::size_t>(quotient) * kOwnerSlots;
  if (owner_find_in_page(page, low, found))
    return true;
  if (owner.spill_count[quotient] == 0u)
    return false;
  const std::uint64_t token = spill_token(key);
  std::uint64_t slot = spill_slot(key, owner.spill_mask);
  for (std::uint64_t probe = 0; probe <= owner.spill_mask; ++probe) {
    std::uint64_t state =
        *reinterpret_cast<volatile const std::uint64_t *>(owner.spill_keys + slot);
    if (state == kSpillEmpty)
      return false;
    if (state == (token | kSpillLock)) {
      do {
        state = *reinterpret_cast<volatile const std::uint64_t *>(
            owner.spill_keys + slot);
      } while (state == (token | kSpillLock));
    }
    if (state == token) {
      const std::uint32_t value =
          *reinterpret_cast<volatile const std::uint32_t *>(owner.spill_values +
                                                            slot);
      *found = owner_pack(low, value, kOwnerLive);
      return true;
    }
    slot = (slot + 1u) & owner.spill_mask;
  }
  return false;
}
// Build each primary page in shared memory; write coalesced.
__global__ void owner_build_kernel(OwnerView owner, const std::uint32_t *keys,
                                   const std::uint32_t *values,
                                   const std::uint32_t *offsets) {
  __shared__ std::uint64_t page[kOwnerSlots];
  const std::uint32_t quotient = blockIdx.x;
  if (quotient >= kEpochQuotients)
    return;
  const std::uint32_t begin = offsets[quotient];
  const std::uint32_t end = offsets[quotient + 1u];
  for (int s = threadIdx.x; s < kOwnerSlots; s += blockDim.x)
    page[s] = 0u;
  if (threadIdx.x == 0)
    owner.quotient_live[quotient] = end - begin;
  __syncthreads();
  for (std::uint32_t position = begin + threadIdx.x; position < end;
       position += blockDim.x) {
    const std::uint32_t key = keys[position];
    const std::uint32_t low = key & 0xffffu;
    const std::uint64_t packed = owner_pack(low, values[position], kOwnerLive);
    std::uint32_t slot = owner_slot(low);
    bool placed = false;
    for (int probe = 0; probe < kOwnerSlots; ++probe) {
      const std::uint64_t old = atomicCAS(
          reinterpret_cast<unsigned long long *>(page + slot), 0ull, packed);
      if (old == 0ull) {
        placed = true;
        break;
      }
      if (owner_low(old) == low) {
        atomicExch(reinterpret_cast<unsigned long long *>(page + slot),
                   packed);
        placed = true;
        break;
      }
      slot = (slot + 1u) & kOwnerSlotMask;
    }
    if (!placed)
      spill_insert_fresh(owner, key, values[position]);
  }
  __syncthreads();
  std::uint64_t *destination =
      owner.primary + static_cast<std::size_t>(quotient) * kOwnerSlots;
  for (int s = threadIdx.x; s < kOwnerSlots; s += blockDim.x)
    destination[s] = page[s];
}

// The page may live in global or staged shared memory.
__device__ inline bool owner_atomic_upsert(OwnerView owner,
                                           std::uint64_t *page,
                                           std::uint32_t key,
                                           std::uint32_t value,
                                           std::uint64_t *previous) {
  const std::uint32_t quotient = key >> kEpochQuotientBits;
  const std::uint32_t low = key & 0xffffu;
  for (;;) {
    std::uint32_t slot = owner_slot(low);
    std::uint32_t reusable = kOwnerNoPage;
    std::uint64_t reusable_value = 0u;
    bool full_scan = true;
    for (int probe = 0; probe < kOwnerSlots; ++probe) {
      const std::uint64_t packed =
          *reinterpret_cast<volatile std::uint64_t *>(page + slot);
      const std::uint32_t state = owner_state(packed);
      if (state == kOwnerEmpty) {
        const std::uint32_t target = reusable == kOwnerNoPage ? slot : reusable;
        const std::uint64_t expected =
            reusable == kOwnerNoPage ? packed : reusable_value;
        auto *atomic_slot =
            reinterpret_cast<unsigned long long *>(page + target);
        const std::uint64_t old = atomicCAS(atomic_slot, expected,
                                            owner_pack(low, value, kOwnerLive));
        if (old == expected) {
          *previous = old;
          return true;
        }
        full_scan = false;
        break;
      }
      if (owner_low(packed) == low) {
        auto *atomic_slot = reinterpret_cast<unsigned long long *>(page + slot);
        const std::uint64_t old =
            atomicCAS(atomic_slot, packed, owner_pack(low, value, kOwnerLive));
        if (old == packed) {
          *previous = old;
          return true;
        }
        full_scan = false;
        break;
      }
      if (state == kOwnerTomb && reusable == kOwnerNoPage) {
        reusable = slot;
        reusable_value = packed;
      }
      slot = (slot + 1u) & kOwnerSlotMask;
    }
    if (!full_scan)
      continue;
    // No empty in the page: the key may live in spill.
    if (owner.spill_count[quotient] != 0u &&
        spill_try_update(owner, key, value, previous))
      return true;
    if (reusable != kOwnerNoPage) {
      auto *atomic_slot =
          reinterpret_cast<unsigned long long *>(page + reusable);
      const std::uint64_t old = atomicCAS(atomic_slot, reusable_value,
                                          owner_pack(low, value, kOwnerLive));
      if (old == reusable_value) {
        *previous = old;
        return true;
      }
      continue;
    }
    if (spill_insert_fresh(owner, key, value)) {
      *previous = 0u;
      return true;
    }
    *previous = owner_pack(0u, 0u, kOwnerFail);
    return true;
  }
}

__device__ inline std::uint64_t owner_atomic_erase(OwnerView owner,
                                                   std::uint64_t *page,
                                                   std::uint32_t key) {
  const std::uint32_t quotient = key >> kEpochQuotientBits;
  const std::uint32_t low = key & 0xffffu;
  std::uint32_t slot = owner_slot(low);
  for (int probe = 0; probe < kOwnerSlots; ++probe) {
    const std::uint64_t packed =
        *reinterpret_cast<volatile std::uint64_t *>(page + slot);
    const std::uint32_t state = owner_state(packed);
    if (state == kOwnerEmpty)
      return 0u;
    if (owner_low(packed) == low) {
      if (state != kOwnerLive)
        return packed;
      auto *atomic_slot = reinterpret_cast<unsigned long long *>(page + slot);
      return atomicExch(atomic_slot, owner_pack(low, 0u, kOwnerTomb));
    }
    slot = (slot + 1u) & kOwnerSlotMask;
  }
  if (owner.spill_count[quotient] != 0u)
    return spill_erase(owner, key);
  return 0u;
}

__device__ inline void
owner_apply_transition(OwnerView owner, std::uint64_t *page, std::uint32_t key,
                       std::uint8_t op, std::uint32_t input_value,
                       std::uint32_t *value_delta, std::int8_t *count_delta) {
  std::uint64_t previous = 0u;
  if (op == kInsert) {
    owner_atomic_upsert(owner, page, key, input_value, &previous);
    if (owner_state(previous) == kOwnerFail) {
      // Publish no delta for a key that was not stored.
      *value_delta = 0u;
      *count_delta = 0;
      return;
    }
    if (owner_state(previous) == kOwnerLive) {
      *value_delta = input_value - owner_value(previous);
      *count_delta = 0;
    } else {
      *value_delta = input_value;
      *count_delta = 1;
    }
    return;
  }
  previous = owner_atomic_erase(owner, page, key);
  if (owner_state(previous) == kOwnerLive) {
    *value_delta = 0u - owner_value(previous);
    *count_delta = -1;
  } else {
    *value_delta = 0u;
    *count_delta = 0;
  }
}

__global__ void owner_transition_quotient_kernel(
    OwnerView owner, const std::uint32_t *keys, std::uint32_t *values,
    const std::uint8_t *ops, std::uint8_t constant_op,
    const std::uint32_t *offsets, std::int8_t *count_delta,
    std::uint32_t *quotient_count_sum) {
  const std::uint32_t quotient = blockIdx.x;
  if (quotient >= kEpochQuotients)
    return;
  const int lane = threadIdx.x & 31;
  const std::uint32_t begin = offsets[quotient];
  const std::uint32_t end = offsets[quotient + 1u];
  std::uint64_t *page =
      owner.primary + static_cast<std::size_t>(quotient) * kOwnerSlots;
  std::int32_t local_sum = 0;
  for (std::uint32_t base = begin; base < end; base += 32u) {
    const std::uint32_t position = base + static_cast<std::uint32_t>(lane);
    const bool active = position < end;
    const unsigned active_mask = __ballot_sync(0xffffffffu, active);
    std::int8_t delta = 0;
    if (active) {
      const std::uint32_t key = keys[position];
      const unsigned peers = __match_any_sync(active_mask, key);
      const bool final_occurrence = lane == 31 - __clz(peers);
      if (final_occurrence) {
        const std::uint8_t op = ops ? ops[position] : constant_op;
        std::uint32_t value_delta = 0u;
        owner_apply_transition(owner, page, key, op, values[position],
                               &value_delta, &delta);
        values[position] = value_delta;
        count_delta[position] = delta;
      } else {
        values[position] = 0u;
        count_delta[position] = 0;
      }
    }
    local_sum += static_cast<std::int32_t>(delta);
    __syncwarp(0xffffffffu);
  }
  for (int offset = 16; offset > 0; offset >>= 1)
    local_sum += __shfl_down_sync(0xffffffffu, local_sum, offset);
  if (lane == 0) {
    quotient_count_sum[quotient] = static_cast<std::uint32_t>(local_sum);
    if (local_sum != 0)
      owner_add_live(owner, quotient, local_sum);
  }
}

// Occupancy-adaptive transitions: W lanes per quotient.
template <int W>
__global__ void owner_transition_subwarp_kernel(
    OwnerView owner, const std::uint32_t *keys, std::uint32_t *values,
    const std::uint8_t *ops, std::uint8_t constant_op,
    const std::uint32_t *offsets, std::int8_t *count_delta,
    std::uint32_t *quotient_count_sum, const std::uint32_t *class_list,
    const std::uint32_t *class_count) {
  const std::uint32_t items = class_count[0];
  const int lane = threadIdx.x & 31;
  const int sub = lane / W;
  const int s = lane % W;
  const unsigned sub_bits = W == 32 ? 0xffffffffu : (1u << W) - 1u;
  const unsigned sub_mask = sub_bits << (sub * W);
  const std::uint32_t groups =
      static_cast<std::uint32_t>(gridDim.x) * blockDim.x / W;
  for (std::uint32_t group =
           (blockIdx.x * blockDim.x + threadIdx.x) / W;
       group < items; group += groups) {
    const std::uint32_t quotient = class_list[group];
    const std::uint32_t begin = offsets[quotient];
    const std::uint32_t len = offsets[quotient + 1u] - begin;
    const bool active = static_cast<std::uint32_t>(s) < len;
    const std::uint32_t position = begin + static_cast<std::uint32_t>(s);
    std::int8_t delta = 0;
    const unsigned voters = __ballot_sync(sub_mask, active) & sub_mask;
    if (active) {
      std::uint64_t *page =
          owner.primary + static_cast<std::size_t>(quotient) * kOwnerSlots;
      const std::uint32_t key = keys[position];
      const unsigned peers = __match_any_sync(voters, key);
      const bool final_occurrence = lane == 31 - __clz(peers);
      if (!final_occurrence) {
        values[position] = 0u;
        count_delta[position] = 0;
      } else {
        const std::uint8_t op = ops ? ops[position] : constant_op;
        std::uint32_t value_delta = 0u;
        owner_apply_transition(owner, page, key, op, values[position],
                               &value_delta, &delta);
        values[position] = value_delta;
        count_delta[position] = delta;
      }
    }
    std::int32_t sum = static_cast<std::int32_t>(delta);
#pragma unroll
    for (int offset = W / 2; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(sub_mask, sum, offset, W);
    if (s == 0) {
      quotient_count_sum[quotient] = static_cast<std::uint32_t>(sum);
      if (sum != 0)
        owner_add_live(owner, quotient, sum);
    }
  }
}

// Dense quotients: sort low bits with sequence numbers so
// the last write wins, then apply distinct keys in parallel.
constexpr int kDenseChunk = 2048;
constexpr int kDenseThreads = 256;
constexpr int kDenseItems = kDenseChunk / kDenseThreads;
__global__ void owner_transition_dense_kernel(
    OwnerView owner, const std::uint32_t *keys, std::uint32_t *values,
    const std::uint8_t *ops, std::uint8_t constant_op,
    const std::uint32_t *offsets, std::int8_t *count_delta,
    std::uint32_t *quotient_count_sum, const std::uint32_t *class_list,
    const std::uint32_t *class_count) {
  using BlockSort =
      cub::BlockRadixSort<std::uint32_t, kDenseThreads, kDenseItems>;
  __shared__ union DenseShared {
    typename BlockSort::TempStorage sort;
    std::uint32_t sorted[kDenseChunk];
  } shared;
  // Stage the 16 KiB primary page for dense updates.
  __shared__ std::uint64_t page[kOwnerSlots];
  __shared__ std::int32_t block_sum;
  const std::uint32_t items = class_count[0];
  for (std::uint32_t item = blockIdx.x; item < items; item += gridDim.x) {
    const std::uint32_t quotient = class_list[item];
    const std::uint32_t begin = offsets[quotient];
    const std::uint32_t end = offsets[quotient + 1u];
    std::uint64_t *global_page =
        owner.primary + static_cast<std::size_t>(quotient) * kOwnerSlots;
    for (int s = threadIdx.x; s < kOwnerSlots; s += blockDim.x)
      page[s] = global_page[s];
    if (threadIdx.x == 0)
      block_sum = 0;
    __syncthreads();
    for (std::uint32_t chunk = begin; chunk < end; chunk += kDenseChunk) {
      const std::uint32_t n =
          min(static_cast<std::uint32_t>(kDenseChunk), end - chunk);
      std::uint32_t local[kDenseItems];
#pragma unroll
      for (int r = 0; r < kDenseItems; ++r) {
        const std::uint32_t seq =
            static_cast<std::uint32_t>(threadIdx.x) * kDenseItems + r;
        local[r] = seq < n ? ((keys[chunk + seq] & 0xffffu) << 11) | seq
                           : 0x0fffffffu;
      }
      BlockSort(shared.sort).Sort(local, 0, 28);
      __syncthreads();
#pragma unroll
      for (int r = 0; r < kDenseItems; ++r)
        shared.sorted[threadIdx.x * kDenseItems + r] = local[r];
      __syncthreads();
      std::int32_t sum = 0;
      for (std::uint32_t slot = threadIdx.x; slot < n; slot += blockDim.x) {
        const std::uint32_t packed = shared.sorted[slot];
        const std::uint32_t position = chunk + (packed & 0x7ffu);
        const bool applier =
            slot + 1u >= n ||
            (shared.sorted[slot + 1u] >> 11) != (packed >> 11);
        if (!applier) {
          values[position] = 0u;
          count_delta[position] = 0;
          continue;
        }
        const std::uint8_t op = ops ? ops[position] : constant_op;
        std::uint32_t value_delta = 0u;
        std::int8_t delta = 0;
        owner_apply_transition(owner, page, keys[position], op,
                               values[position], &value_delta, &delta);
        values[position] = value_delta;
        count_delta[position] = delta;
        sum += static_cast<std::int32_t>(delta);
      }
      for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
      if ((threadIdx.x & 31) == 0 && sum != 0)
        atomicAdd(&block_sum, sum);
      __syncthreads();
    }
    for (int s = threadIdx.x; s < kOwnerSlots; s += blockDim.x)
      global_page[s] = page[s];
    if (threadIdx.x == 0) {
      quotient_count_sum[quotient] = static_cast<std::uint32_t>(block_sum);
      if (block_sum != 0)
        owner_add_live(owner, quotient, block_sum);
    }
    __syncthreads();
  }
}

__global__ void owner_lookup_kernel(OwnerView owner,
                                    const std::uint32_t *queries,
                                    std::size_t count,
                                    std::uint32_t *out_values,
                                    std::uint8_t *out_found) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  std::uint64_t packed = 0u;
  const bool found = owner_find(owner, queries[i], &packed) &&
                     owner_state(packed) == kOwnerLive;
  out_values[i] = found ? owner_value(packed) : kEmptyKey;
  if (out_found)
    out_found[i] = found ? 1u : 0u;
}

__device__ inline std::uint32_t epoch_subgroup_mask(const RunView &ev,
                                                    std::uint32_t quotient,
                                                    std::uint32_t subgroup,
                                                    std::uint32_t count) {
  const std::uint32_t valid = count == 32u ? 0xffffffffu : ((1u << count) - 1u);
  const std::uint32_t *planes =
      ev.subgroup_masks + quotient * kEpochSubgroupPlanes;
  std::uint32_t mask = valid;
#pragma unroll
  for (int bit = 0; bit < kEpochSubgroupBits; ++bit)
    mask &= ((subgroup >> bit) & 1u) ? planes[bit] : ~planes[bit];
  return mask;
}


__device__ inline std::uint32_t epoch_quotient_count_bounds(const RunView &ev,
                                                            std::uint32_t begin,
                                                            std::uint32_t end,
                                                            std::uint32_t lo,
                                                            std::uint32_t hi) {
  std::int32_t count = 0;
  for (std::uint32_t p = begin; p < end; ++p) {
    if (ev.keys[p] >= lo && ev.keys[p] <= hi)
      count += static_cast<std::int32_t>(ev.count_delta[p]);
  }
  return static_cast<std::uint32_t>(count);
}

__device__ inline std::uint32_t epoch_quotient_sum_bounds(const RunView &ev,
                                                          std::uint32_t begin,
                                                          std::uint32_t end,
                                                          std::uint32_t lo,
                                                          std::uint32_t hi) {
  std::uint32_t sum = 0u;
  const std::uint32_t physical_count = end - begin;
  if (physical_count > 32u &&
      (ev.fully_sorted || physical_count <= kEpochHeavySortCap)) {
    const std::uint32_t lb = begin + static_cast<std::uint32_t>(lower_bound_u32(
                                         ev.keys + begin, end - begin, lo));
    const std::uint32_t ub = begin + static_cast<std::uint32_t>(upper_bound_u32(
                                         ev.keys + begin, end - begin, hi));
    for (std::uint32_t p = lb; p < ub; ++p)
      sum += ev.values[p];
    return sum;
  }
  for (std::uint32_t p = begin; p < end; ++p)
    if (ev.keys[p] >= lo && ev.keys[p] <= hi)
      sum += ev.values[p];
  return sum;
}

__device__ inline std::uint32_t
epoch_subgroup_edge_sum(const RunView &ev, std::uint32_t quotient,
                        std::uint32_t begin, std::uint32_t count,
                        std::uint32_t subgroup, std::uint32_t lo,
                        std::uint32_t hi) {
  std::uint32_t mask = epoch_subgroup_mask(ev, quotient, subgroup, count);
  std::uint32_t sum = 0u;
  while (mask != 0u) {
    const std::uint32_t bit = __ffs(mask) - 1;
    const std::uint32_t position = begin + bit;
    const std::uint32_t key = ev.keys[position];
    if (key >= lo && key <= hi)
      sum += ev.values[position];
    mask &= mask - 1;
  }
  return sum;
}

__device__ inline std::uint32_t
epoch_indexed_quotient_sum_bounds(const RunView &ev, std::uint32_t quotient,
                                  std::uint32_t begin, std::uint32_t end,
                                  std::uint32_t lo, std::uint32_t hi) {
  const std::uint32_t count = end - begin;
  if (!ev.subgroup_value_prefix || count > 32u)
    return epoch_quotient_sum_bounds(ev, begin, end, lo, hi);
  const std::uint32_t first =
      (lo >> (kEpochQuotientBits - kEpochSubgroupBits)) &
      (kEpochSubgroups - 1u);
  const std::uint32_t last = (hi >> (kEpochQuotientBits - kEpochSubgroupBits)) &
                             (kEpochSubgroups - 1u);
  if (first == last)
    return epoch_subgroup_edge_sum(ev, quotient, begin, count, first, lo, hi);
  std::uint32_t sum =
      epoch_subgroup_edge_sum(ev, quotient, begin, count, first, lo, hi);
  sum += epoch_subgroup_edge_sum(ev, quotient, begin, count, last, lo, hi);
  if (last > first + 1u) {
    const std::uint32_t *prefix =
        ev.subgroup_value_prefix + quotient * kEpochSubgroupPrefixStride;
    sum += prefix[last] - prefix[first + 1u];
  }
  return sum;
}

__device__ inline std::uint32_t
epoch_range_sum_one(const RunView &ev, std::uint32_t lo, std::uint32_t hi) {
  const std::uint32_t first = lo >> kEpochQuotientBits;
  const std::uint32_t last = hi >> kEpochQuotientBits;
  const std::uint32_t fb = ev.quotient_off[first];
  const std::uint32_t fe = ev.quotient_off[first + 1u];
  if (first == last)
    return epoch_indexed_quotient_sum_bounds(ev, first, fb, fe, lo, hi);
  std::uint32_t sum =
      epoch_indexed_quotient_sum_bounds(ev, first, fb, fe, lo, 0xffffffffu);
  const std::uint32_t lb = ev.quotient_off[last];
  const std::uint32_t le = ev.quotient_off[last + 1u];
  sum += epoch_indexed_quotient_sum_bounds(ev, last, lb, le, 0u, hi);
  if (last > first + 1u)
    sum +=
        ev.quotient_value_prefix[last] - ev.quotient_value_prefix[first + 1u];
  return sum;
}

__device__ inline std::uint32_t
epoch_range_count_one(const RunView &ev, std::uint32_t lo, std::uint32_t hi) {
  const std::uint32_t first = lo >> kEpochQuotientBits;
  const std::uint32_t last = hi >> kEpochQuotientBits;
  const std::uint32_t fb = ev.quotient_off[first];
  const std::uint32_t fe = ev.quotient_off[first + 1u];
  if (first == last)
    return epoch_quotient_count_bounds(ev, fb, fe, lo, hi);
  std::uint32_t count =
      epoch_quotient_count_bounds(ev, fb, fe, lo, 0xffffffffu);
  const std::uint32_t lb = ev.quotient_off[last];
  const std::uint32_t le = ev.quotient_off[last + 1u];
  count += epoch_quotient_count_bounds(ev, lb, le, 0u, hi);
  if (last > first + 1u)
    count +=
        ev.quotient_count_prefix[last] - ev.quotient_count_prefix[first + 1u];
  return count;
}

// Transposed reads keep adjacent accesses together.
__device__ inline std::uint32_t
run_range_sum_transposed(const RunView &ev, const std::uint32_t *off_t,
                         const std::uint32_t *vp_t, std::uint32_t run,
                         std::uint32_t lo, std::uint32_t hi) {
  const std::uint32_t first = lo >> kEpochQuotientBits;
  const std::uint32_t last = hi >> kEpochQuotientBits;
  const std::uint32_t fb = off_t[first * kRunStride + run];
  const std::uint32_t fe = off_t[(first + 1u) * kRunStride + run];
  if (first == last)
    return epoch_indexed_quotient_sum_bounds(ev, first, fb, fe, lo, hi);
  std::uint32_t sum =
      epoch_indexed_quotient_sum_bounds(ev, first, fb, fe, lo, 0xffffffffu);
  const std::uint32_t lb = off_t[last * kRunStride + run];
  const std::uint32_t le = off_t[(last + 1u) * kRunStride + run];
  sum += epoch_indexed_quotient_sum_bounds(ev, last, lb, le, 0u, hi);
  if (last > first + 1u)
    sum +=
        vp_t[last * kRunStride + run] - vp_t[(first + 1u) * kRunStride + run];
  return sum;
}

__device__ inline std::uint32_t
run_range_count_transposed(const RunView &ev, const std::uint32_t *off_t,
                           const std::uint32_t *cp_t, std::uint32_t run,
                           std::uint32_t lo, std::uint32_t hi) {
  const std::uint32_t first = lo >> kEpochQuotientBits;
  const std::uint32_t last = hi >> kEpochQuotientBits;
  const std::uint32_t fb = off_t[first * kRunStride + run];
  const std::uint32_t fe = off_t[(first + 1u) * kRunStride + run];
  if (first == last)
    return epoch_quotient_count_bounds(ev, fb, fe, lo, hi);
  std::uint32_t count =
      epoch_quotient_count_bounds(ev, fb, fe, lo, 0xffffffffu);
  const std::uint32_t lb = off_t[last * kRunStride + run];
  const std::uint32_t le = off_t[(last + 1u) * kRunStride + run];
  count += epoch_quotient_count_bounds(ev, lb, le, 0u, hi);
  if (last > first + 1u)
    count +=
        cp_t[last * kRunStride + run] - cp_t[(first + 1u) * kRunStride + run];
  return count;
}

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

__device__ inline std::uint32_t
logical_run_sum(const LogicalRunView &run, std::uint32_t lo, std::uint32_t hi) {
  if (run.kind == kSortedRunCache) {
    return sorted_range_sum(run.sorted, run.sorted_range,
                            run.sorted_value_prefix, lo, hi);
  }
  return epoch_range_sum_one(run.delta, lo, hi);
}

__device__ inline std::uint32_t logical_run_count(const LogicalRunView &run,
                                                  std::uint32_t lo,
                                                  std::uint32_t hi) {
  if (run.kind == kSortedRunCache)
    return sorted_range_count(run.sorted, run.sorted_count_prefix, lo, hi);
  return epoch_range_count_one(run.delta, lo, hi);
}

__device__ inline std::uint32_t
logical_run_sum_transposed(const LogicalRunView &run,
                           const std::uint32_t *off_t,
                           const std::uint32_t *vp_t, std::uint32_t slot,
                           std::uint32_t lo, std::uint32_t hi) {
  if (run.kind == kSortedRunCache) {
    return sorted_range_sum(run.sorted, run.sorted_range,
                            run.sorted_value_prefix, lo, hi);
  }
  return run_range_sum_transposed(run.delta, off_t, vp_t, slot, lo, hi);
}

__device__ inline std::uint32_t
logical_run_count_transposed(const LogicalRunView &run,
                             const std::uint32_t *off_t,
                             const std::uint32_t *cp_t, std::uint32_t slot,
                             std::uint32_t lo, std::uint32_t hi) {
  if (run.kind == kSortedRunCache)
    return sorted_range_count(run.sorted, run.sorted_count_prefix, lo, hi);
  return run_range_count_transposed(run.delta, off_t, cp_t, slot, lo, hi);
}

__global__ void run_count_prefix_input_kernel(const std::int8_t *counts,
                                              std::size_t count,
                                              std::uint32_t *prefix) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  prefix[i + 1u] =
      static_cast<std::uint32_t>(static_cast<std::int32_t>(counts[i]));
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

__device__ inline std::uint32_t
range_delta_edge_sum(const RangeDeltaView &projection, std::uint32_t bin,
                     std::uint32_t lo, std::uint32_t hi) {
  const std::uint32_t begin = projection.offsets[bin];
  const std::uint32_t end = projection.offsets[bin + 1u];
  std::uint32_t sum = 0u;
  for (std::uint32_t position = begin; position < end; ++position) {
    const std::uint32_t key = projection.keys[position];
    if (key >= lo && key <= hi)
      sum += projection.values[position];
  }
  return sum;
}

__device__ inline std::uint32_t
range_delta_sum(const RangeDeltaView &projection, std::uint32_t lo,
                std::uint32_t hi) {
  const std::uint32_t first = lo >> kRangeProjectionShift;
  const std::uint32_t last = hi >> kRangeProjectionShift;
  if (first == last)
    return range_delta_edge_sum(projection, first, lo, hi);
  std::uint32_t sum = range_delta_edge_sum(projection, first, lo, 0xffffffffu);
  sum += range_delta_edge_sum(projection, last, 0u, hi);
  if (last > first + 1u) {
    sum += projection.value_prefix[last] - projection.value_prefix[first + 1u];
  }
  return sum;
}

__device__ inline void range_delta_accumulate_warp(
    const std::uint32_t *keys, const std::uint32_t *values, std::uint32_t begin,
    std::uint32_t end, std::uint32_t *counts, std::uint32_t *sums) {
  const unsigned full = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  for (std::uint32_t base = begin; base < end; base += 32u) {
    const std::uint32_t position = base + lane;
    const bool included = position < end;
    const unsigned active = __ballot_sync(full, included);
    if (included) {
      const std::uint32_t subgroup =
          (keys[position] >> kRangeProjectionShift) & (kEpochSubgroups - 1u);
      const unsigned peers = __match_any_sync(active, subgroup);
      const std::uint32_t group_sum =
          __reduce_add_sync(peers, values[position]);
      const int leader = __ffs(peers) - 1;
      if (lane == leader) {
        counts[subgroup] += __popc(peers);
        sums[subgroup] += group_sum;
      }
    }
    __syncwarp(full);
  }
}


__global__ void range_delta_plan_kernel(const RunView *epochs, int epoch_count,
                                        int skip_run, std::uint32_t *bin_counts,
                                        std::uint32_t *bin_sums) {
  constexpr int warps = 8;
  __shared__ std::uint32_t counts[warps][kEpochSubgroups];
  __shared__ std::uint32_t sums[warps][kEpochSubgroups];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const std::uint32_t quotient =
      blockIdx.x * warps + static_cast<std::uint32_t>(warp);
  if (quotient >= kEpochQuotients)
    return;
  if (lane < kEpochSubgroups) {
    counts[warp][lane] = 0u;
    sums[warp][lane] = 0u;
  }
  __syncwarp();
  for (int e = 0; e < epoch_count; ++e) {
    if (e == skip_run)
      continue;
    const RunView epoch = epochs[e];
    const std::uint32_t begin = epoch.quotient_off[quotient];
    const std::uint32_t end = epoch.quotient_off[quotient + 1u];
    range_delta_accumulate_warp(epoch.keys, epoch.values, begin, end,
                                counts[warp], sums[warp]);
  }
  __syncwarp();
  if (lane < kEpochSubgroups) {
    const std::uint32_t bin = quotient * kEpochSubgroups + lane;
    bin_counts[bin] = counts[warp][lane];
    bin_sums[bin] = sums[warp][lane];
  }
}

// Repack records quotient-locally, then write each bin
// as one contiguous segment.
constexpr int kPackChunk = 512;
constexpr int kPackThreads = 256;
constexpr int kPackPerThread = kPackChunk / kPackThreads;
__global__ void range_delta_pack_kernel(const RunView *epochs, int epoch_count,
                                        int skip_run,
                                        const std::uint32_t *bin_offsets,
                                        std::uint32_t *out_keys,
                                        std::uint32_t *out_values) {
  __shared__ std::uint32_t slice_prefix[kRunCapacity + 1];
  __shared__ std::uint32_t staged_keys[kPackChunk];
  __shared__ std::uint32_t staged_values[kPackChunk];
  __shared__ std::uint32_t bin_count[kEpochSubgroups];
  __shared__ std::uint32_t bin_start[kEpochSubgroups];
  __shared__ std::uint32_t bin_fill[kEpochSubgroups];
  __shared__ std::uint32_t cursors[kEpochSubgroups];
  const std::uint32_t quotient = blockIdx.x;
  if (quotient >= kEpochQuotients)
    return;
  if (threadIdx.x == 0) {
    std::uint32_t running = 0u;
    slice_prefix[0] = 0u;
    for (int e = 0; e < epoch_count; ++e) {
      if (e != skip_run)
        running += epochs[e].quotient_off[quotient + 1u] -
                   epochs[e].quotient_off[quotient];
      slice_prefix[e + 1] = running;
    }
  }
  if (threadIdx.x < kEpochSubgroups)
    cursors[threadIdx.x] =
        bin_offsets[quotient * kEpochSubgroups + threadIdx.x];
  __syncthreads();
  const std::uint32_t total = slice_prefix[epoch_count];
  for (std::uint32_t chunk = 0; chunk < total; chunk += kPackChunk) {
    const std::uint32_t n =
        min(static_cast<std::uint32_t>(kPackChunk), total - chunk);
    if (threadIdx.x < kEpochSubgroups) {
      bin_count[threadIdx.x] = 0u;
      bin_fill[threadIdx.x] = 0u;
    }
    __syncthreads();
    std::uint32_t local_keys[kPackPerThread];
    std::uint32_t local_values[kPackPerThread];
    std::uint32_t local_subs[kPackPerThread];
#pragma unroll
    for (int r = 0; r < kPackPerThread; ++r) {
      const std::uint32_t i =
          static_cast<std::uint32_t>(r) * blockDim.x + threadIdx.x;
      if (i >= n)
        continue;
      const std::uint32_t g = chunk + i;
      int run = 0;
      int hi_run = epoch_count;
      while (run + 1 < hi_run) {
        const int mid = (run + hi_run) >> 1;
        if (slice_prefix[mid] <= g)
          run = mid;
        else
          hi_run = mid;
      }
      const RunView ev = epochs[run];
      const std::uint32_t p =
          ev.quotient_off[quotient] + (g - slice_prefix[run]);
      local_keys[r] = ev.keys[p];
      local_values[r] = ev.values[p];
      local_subs[r] =
          (local_keys[r] >> kRangeProjectionShift) & (kEpochSubgroups - 1u);
      atomicAdd(bin_count + local_subs[r], 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      std::uint32_t running = 0u;
      for (int b = 0; b < kEpochSubgroups; ++b) {
        bin_start[b] = running;
        running += bin_count[b];
      }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kPackPerThread; ++r) {
      const std::uint32_t i =
          static_cast<std::uint32_t>(r) * blockDim.x + threadIdx.x;
      if (i >= n)
        continue;
      const std::uint32_t position =
          bin_start[local_subs[r]] + atomicAdd(bin_fill + local_subs[r], 1u);
      staged_keys[position] = local_keys[r];
      staged_values[position] = local_values[r];
    }
    __syncthreads();
    for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
      const std::uint32_t key = staged_keys[i];
      const std::uint32_t sub =
          (key >> kRangeProjectionShift) & (kEpochSubgroups - 1u);
      const std::uint32_t destination = cursors[sub] + (i - bin_start[sub]);
      out_keys[destination] = key;
      out_values[destination] = staged_values[i];
    }
    __syncthreads();
    if (threadIdx.x < kEpochSubgroups)
      cursors[threadIdx.x] += bin_count[threadIdx.x];
    __syncthreads();
  }
}

// Scatter quotient starts; mask warp-contained segments.
__global__ void run_boundary_scatter_kernel(
    const std::uint32_t *keys, std::uint32_t record_count,
    std::uint32_t *offsets, std::uint32_t *subgroup_masks,
    std::uint32_t *pending, std::uint32_t *pending_count) {
  constexpr unsigned full = 0xffffffffu;
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t warp_begin = i & ~31u;
  const int lane = threadIdx.x & 31;
  const bool valid = i < record_count;
  const std::uint32_t key = valid ? keys[i] : 0u;
  const std::uint32_t quotient = key >> kEpochQuotientBits;
  std::uint32_t previous = __shfl_up_sync(full, quotient, 1);
  if (lane == 0 && valid)
    previous = i == 0u ? quotient : keys[i - 1] >> kEpochQuotientBits;
  const bool starts = valid && (i == 0u || quotient != previous);
  if (starts)
    offsets[quotient] = i;
  if (i == 0u)
    offsets[kEpochQuotients] = record_count;
  if (warp_begin >= record_count)
    return;
  const std::uint32_t warp_end = min(warp_begin + 32u, record_count);
  std::uint32_t next_quotient = kEpochQuotients;
  if (lane == 0 && warp_end < record_count)
    next_quotient = keys[warp_end] >> kEpochQuotientBits;
  next_quotient = __shfl_sync(full, next_quotient, 0);
  unsigned starts_mask = __ballot_sync(full, starts);
  while (starts_mask != 0u) {
    const int start_lane = __ffs(starts_mask) - 1;
    const std::uint32_t segment_quotient =
        __shfl_sync(full, quotient, start_lane);
    const unsigned later =
        start_lane == 31 ? 0u : starts_mask & (0xffffffffu << (start_lane + 1));
    const int end_lane = later == 0u
                             ? static_cast<int>(warp_end - warp_begin)
                             : __ffs(later) - 1;
    const bool contained = later != 0u || segment_quotient != next_quotient;
    if (!contained) {
      // Crosses the warp boundary: finish in completion pass.
      if (lane == 0)
        pending[atomicAdd(pending_count, 1u)] = segment_quotient;
      starts_mask &= starts_mask - 1u;
      continue;
    }
    const unsigned low = 0xffffffffu << start_lane;
    const unsigned high =
        end_lane >= 32 ? 0xffffffffu : (1u << end_lane) - 1u;
    const unsigned segment = low & high;
    const std::uint32_t subgroup =
        (key >> (kEpochQuotientBits - kEpochSubgroupBits)) &
        (kEpochSubgroups - 1u);
    const std::uint32_t base = segment_quotient * kEpochSubgroupPlanes;
#pragma unroll
    for (int bit = 0; bit < kEpochSubgroupPlanes; ++bit) {
      const unsigned local =
          __ballot_sync(full, valid && ((subgroup >> bit) & 1u));
      if (lane == 0)
        subgroup_masks[base + bit] = (local & segment) >> start_lane;
    }
    starts_mask &= starts_mask - 1u;
  }
}

// Lengths, live counts, and transition size classes.
__global__ void run_finalize_metadata_kernel(const std::uint32_t *offsets,
                                             std::uint32_t *quotient_live,
                                             std::uint32_t *quotient_count_sum,
                                             std::uint32_t *class_list,
                                             std::uint32_t *class_count) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kEpochQuotients)
    return;
  const std::uint32_t len = offsets[q + 1u] - offsets[q];
  quotient_live[q] = len;
  quotient_count_sum[q] = 0u;
  if (len == 0u || !class_list || !class_count)
    return;
  const int cls = len <= 8u ? 0 : len <= 16u ? 1 : len <= 32u ? 2 : 3;
  class_list[cls * kEpochQuotients + atomicAdd(class_count + cls, 1u)] = q;
}

// Finish masks for quotients that crossed a warp boundary.
__global__ void run_pending_masks_kernel(const std::uint32_t *keys,
                                         const std::uint32_t *offsets,
                                         const std::uint32_t *pending,
                                         const std::uint32_t *pending_count,
                                         std::uint32_t *subgroup_masks) {
  const std::uint32_t items = pending_count[0];
  for (std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
       item < items; item += gridDim.x * blockDim.x) {
    const std::uint32_t q = pending[item];
    const std::uint32_t begin = offsets[q];
    const std::uint32_t len = offsets[q + 1u] - begin;
    if (len > 32u)
      continue;
    std::uint32_t planes[kEpochSubgroupPlanes] = {};
    for (std::uint32_t p = 0; p < len; ++p) {
      const std::uint32_t subgroup =
          (keys[begin + p] >> (kEpochQuotientBits - kEpochSubgroupBits)) &
          (kEpochSubgroups - 1u);
      const std::uint32_t bit_position = 1u << p;
#pragma unroll
      for (int bit = 0; bit < kEpochSubgroupPlanes; ++bit)
        planes[bit] |= (0u - ((subgroup >> bit) & 1u)) & bit_position;
    }
    const std::uint32_t base = q * kEpochSubgroupPlanes;
#pragma unroll
    for (int bit = 0; bit < kEpochSubgroupPlanes; ++bit)
      subgroup_masks[base + bit] = planes[bit];
  }
}

__global__ void
epoch_subgroup_value_prefix_kernel(RunView ev, std::uint32_t *subgroup_prefix) {
  const std::uint32_t quotient = blockIdx.x * blockDim.x + threadIdx.x;
  if (quotient >= kEpochQuotients)
    return;
  const std::uint32_t begin = ev.quotient_off[quotient];
  const std::uint32_t end = ev.quotient_off[quotient + 1];
  std::uint32_t sums[kEpochSubgroups] = {};
  for (std::uint32_t position = begin; position < end; ++position) {
    const std::uint32_t subgroup =
        (ev.keys[position] >> (kEpochQuotientBits - kEpochSubgroupBits)) &
        (kEpochSubgroups - 1u);
    sums[subgroup] += ev.values[position];
  }
  const std::uint32_t base = quotient * kEpochSubgroupPrefixStride;
  std::uint32_t prefix = 0u;
#pragma unroll
  for (int subgroup = 0; subgroup < kEpochSubgroups; ++subgroup) {
    subgroup_prefix[base + subgroup] = prefix;
    prefix += sums[subgroup];
  }
  ev.quotient_value_sum[quotient] = prefix;
}

__global__ void epoch_classify_heavy_quotients_kernel(RunView ev) {
  const std::uint32_t quotient = blockIdx.x * blockDim.x + threadIdx.x;
  if (quotient >= kEpochQuotients)
    return;
  const std::uint32_t count =
      ev.quotient_off[quotient + 1] - ev.quotient_off[quotient];
  if (count > 32u && count <= kEpochHeavySortCap)
    ev.heavy_list[atomicAdd(ev.heavy_count, 1u)] = quotient;
}

__global__ void epoch_sort_heavy_quotients_kernel(RunView ev) {
  __shared__ std::uint32_t shared_keys[kEpochHeavySortCap];
  __shared__ std::uint32_t shared_values[kEpochHeavySortCap];
  __shared__ std::int8_t shared_counts[kEpochHeavySortCap];
  constexpr int shards = 64;
  const int shard = blockIdx.x;
  std::uint32_t *keys = const_cast<std::uint32_t *>(ev.keys);
  std::uint32_t *values = const_cast<std::uint32_t *>(ev.values);
  std::int8_t *counts = const_cast<std::int8_t *>(ev.count_delta);
  const int tid = threadIdx.x;
  const std::uint32_t count = ev.heavy_count[0];
  for (std::uint32_t item = shard; item < count; item += shards) {
    const std::uint32_t quotient = ev.heavy_list[item];
    const std::uint32_t begin = ev.quotient_off[quotient];
    const std::uint32_t length = ev.quotient_off[quotient + 1] - begin;
    shared_keys[tid] = tid < length ? keys[begin + tid] : kEmptyKey;
    shared_values[tid] = tid < length ? values[begin + tid] : 0u;
    shared_counts[tid] = tid < length ? counts[begin + tid] : 0;
    __syncthreads();
    for (int size = 2; size <= kEpochHeavySortCap; size <<= 1) {
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        const int peer = tid ^ stride;
        if (peer > tid) {
          const bool ascending = (tid & size) == 0;
          const std::uint32_t a = shared_keys[tid];
          const std::uint32_t b = shared_keys[peer];
          if ((ascending && a > b) || (!ascending && a < b)) {
            shared_keys[tid] = b;
            shared_keys[peer] = a;
            const std::uint32_t av = shared_values[tid];
            shared_values[tid] = shared_values[peer];
            shared_values[peer] = av;
            const std::int8_t ac = shared_counts[tid];
            shared_counts[tid] = shared_counts[peer];
            shared_counts[peer] = ac;
          }
        }
        __syncthreads();
      }
    }
    if (tid < length) {
      keys[begin + tid] = shared_keys[tid];
      values[begin + tid] = shared_values[tid];
      counts[begin + tid] = shared_counts[tid];
    }
    __syncthreads();
  }
}

struct TakeLastU32 {
  __host__ __device__ std::uint32_t operator()(std::uint32_t,
                                               std::uint32_t newer) const {
    return newer;
  }
};


__global__ void epoch_count_delta_sum_kernel(RunView epoch) {
  const std::uint32_t quotient = blockIdx.x * blockDim.x + threadIdx.x;
  if (quotient >= kEpochQuotients)
    return;
  std::int32_t total = 0;
  const std::uint32_t begin = epoch.quotient_off[quotient];
  const std::uint32_t end = epoch.quotient_off[quotient + 1u];
  for (std::uint32_t position = begin; position < end; ++position)
    total += static_cast<std::int32_t>(epoch.count_delta[position]);
  epoch.quotient_count_sum[quotient] = static_cast<std::uint32_t>(total);
}
// Rebuild live-key bits by scanning the owner directory.
// Scalar op for a leaf, or one packed bit for a mixed run.
__device__ inline int assignment_op_at(const std::uint32_t *op_words,
                                       std::uint8_t constant_op,
                                       std::uint8_t mixed, std::uint32_t p) {
  if (!mixed)
    return constant_op;
  return (op_words[p >> 5] >> (p & 31u)) & 1u;
}

// Seed L0 from the sorted BaseRun (all keys live).
__global__ void succ_seed_base_kernel(const std::uint32_t *keys,
                                      std::size_t count, std::uint32_t *bits) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::uint32_t key = keys[i];
  atomicOr(bits + (key >> 5), 1u << (key & 31u));
}

// Propagate until the containing word stays nonempty.
__device__ inline void succ_apply_hierarchy(std::uint32_t *bits,
                                            std::uint32_t key, bool insert) {
  std::uint32_t index = key;
#pragma unroll
  for (int level = 0; level < 6; ++level) {
    std::uint32_t *words = bits + succ_level_off(level);
    const std::uint32_t word = index >> 5;
    const std::uint32_t mask = 1u << (index & 31u);
    if (insert) {
      const std::uint32_t old = atomicOr(words + word, mask);
      if ((old & mask) != 0u || old != 0u)
        return;
    } else {
      const std::uint32_t old = atomicAnd(words + word, ~mask);
      if ((old & mask) == 0u || (old & ~mask) != 0u)
        return;
    }
    index = word;
  }
}

__global__ void succ_apply_homogeneous_kernel(const std::uint32_t *keys,
                                              std::size_t count, bool insert,
                                              std::uint32_t *bits) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  succ_apply_hierarchy(bits, keys[i], insert);
}

// Mixed runs settle L0 before sparse hierarchy repair.
__global__ void succ_apply_mixed_kernel(const std::uint32_t *keys,
                                        const std::uint32_t *op_words,
                                        std::size_t count,
                                        std::uint32_t *bits) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::uint32_t key = keys[i];
  if (i + 1 < count && keys[i + 1] == key)
    return;
  const std::uint32_t mask = 1u << (key & 31u);
  if (assignment_op_at(op_words, 0u, 1u,
                       static_cast<std::uint32_t>(i)) != 0)
    atomicOr(bits + (key >> 5), mask);
  else
    atomicAnd(bits + (key >> 5), ~mask);
}

// One upper bit summarizes 32 lower words.
__global__ void succ_build_level_kernel(const std::uint32_t *lower,
                                        std::uint32_t *upper,
                                        std::uint32_t upper_words) {
  const std::uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= upper_words)
    return;
  const std::uint32_t base = w << 5;
  std::uint32_t m = 0u;
#pragma unroll
  for (int b = 0; b < 32; ++b)
    m |= lower[base + b] != 0u ? (1u << b) : 0u;
  upper[w] = m;
}

// Propagate touched words into one upper level.
__global__ void succ_update_level_kernel(const std::uint32_t *keys,
                                         std::size_t count,
                                         const std::uint32_t *lower,
                                         std::uint32_t *upper, int key_shift) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool valid = i < count;
  const std::uint32_t index = valid ? keys[i] >> key_shift : 0u;
  const unsigned active = __ballot_sync(0xffffffffu, valid);
  if (!valid)
    return;
  const unsigned peers = __match_any_sync(active, index);
  const int lane = threadIdx.x & 31;
  if (lane != __ffs(peers) - 1)
    return;
  const std::uint32_t mask = 1u << (index & 31u);
  if (lower[index] != 0u)
    atomicOr(upper + (index >> 5), mask);
  else
    atomicAnd(upper + (index >> 5), ~mask);
}

// Smallest live key >= query via bounded traversal.
__device__ inline std::uint32_t succ_find(const std::uint32_t *bits,
                                          std::uint32_t key) {
  const std::uint32_t word = key >> 5;
  std::uint32_t m = bits[word] & (0xffffffffu << (key & 31u));
  if (m != 0u)
    return (word << 5) | (__ffs(m) - 1u);
  std::uint32_t pos = word;
  for (int level = 1; level < 6; ++level) {
    const std::uint32_t *lv = bits + succ_level_off(level);
    std::uint32_t w = pos >> 5;
    m = lv[w] & (0xfffffffeu << (pos & 31u));
    if (level == 5) {
      while (m == 0u && w + 1u < succ_level_words(5))
        m = lv[++w];
    }
    if (m != 0u) {
      std::uint32_t index = (w << 5) | (__ffs(m) - 1u);
      for (int down = level - 1; down >= 0; --down) {
        const std::uint32_t child = bits[succ_level_off(down) + index];
        index = (index << 5) | (__ffs(child) - 1u);
      }
      return index;
    }
    pos = w;
  }
  return kEmptyKey;
}

__global__ void succ_query_kernel(const std::uint32_t *queries,
                                  std::size_t count, std::uint32_t *out_keys,
                                  const std::uint32_t *bits) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::uint32_t best = succ_find(bits, queries[i]);
  out_keys[i] = best == kEmptyKey ? 0u : best;
}

// 8 runs x 32 quotients tiles: both sides coalesce.
constexpr int kTransposeRuns = 8;
constexpr int kTransposeQuots = 32;
__global__ void run_transpose_meta_kernel(const RunView *epochs,
                                          int epoch_count, int skip_run,
                                          std::uint32_t *off_t,
                                          std::uint32_t *vp_t,
                                          std::uint32_t *cp_t) {
  __shared__ std::uint32_t t_off[kTransposeRuns][kTransposeQuots + 1];
  __shared__ std::uint32_t t_vp[kTransposeRuns][kTransposeQuots + 1];
  __shared__ std::uint32_t t_cp[kTransposeRuns][kTransposeQuots + 1];
  constexpr std::uint32_t rows = kEpochQuotients + 1;
  const std::uint32_t run_tiles =
      (static_cast<std::uint32_t>(epoch_count) + kTransposeRuns - 1u) /
      kTransposeRuns;
  const std::uint32_t q_base =
      (blockIdx.x / run_tiles) * kTransposeQuots;
  const std::uint32_t r_base =
      (blockIdx.x % run_tiles) * kTransposeRuns;
  const int rr = threadIdx.x / kTransposeQuots;
  const int qq = threadIdx.x % kTransposeQuots;
  const std::uint32_t run = r_base + static_cast<std::uint32_t>(rr);
  const std::uint32_t q = q_base + static_cast<std::uint32_t>(qq);
  std::uint32_t off = 0u;
  std::uint32_t vp = 0u;
  std::uint32_t cp = 0u;
  if (run < static_cast<std::uint32_t>(epoch_count) &&
      static_cast<int>(run) != skip_run && q < rows) {
    off = epochs[run].quotient_off[q];
    if (q < kEpochQuotients) {
      vp = epochs[run].quotient_value_prefix[q];
      if (cp_t)
        cp = epochs[run].quotient_count_prefix[q];
    }
  }
  t_off[rr][qq] = off;
  t_vp[rr][qq] = vp;
  t_cp[rr][qq] = cp;
  __syncthreads();
  const int wq = threadIdx.x / kTransposeRuns;
  const int wr = threadIdx.x % kTransposeRuns;
  const std::uint32_t out_q = q_base + static_cast<std::uint32_t>(wq);
  const std::uint32_t out_run = r_base + static_cast<std::uint32_t>(wr);
  if (out_q >= rows || out_run >= static_cast<std::uint32_t>(epoch_count))
    return;
  const std::size_t dst =
      static_cast<std::size_t>(out_q) * kRunStride + out_run;
  off_t[dst] = t_off[wr][wq];
  if (out_q >= kEpochQuotients)
    return;
  vp_t[dst] = t_vp[wr][wq];
  if (cp_t)
    cp_t[dst] = t_cp[wr][wq];
}

__global__ void run_transpose_count_kernel(const RunView *epochs,
                                           int epoch_count, int skip_run,
                                           std::uint32_t *cp_t) {
  __shared__ std::uint32_t t_cp[kTransposeRuns][kTransposeQuots + 1];
  const std::uint32_t run_tiles =
      (static_cast<std::uint32_t>(epoch_count) + kTransposeRuns - 1u) /
      kTransposeRuns;
  const std::uint32_t q_base =
      (blockIdx.x / run_tiles) * kTransposeQuots;
  const std::uint32_t r_base =
      (blockIdx.x % run_tiles) * kTransposeRuns;
  const int rr = threadIdx.x / kTransposeQuots;
  const int qq = threadIdx.x % kTransposeQuots;
  const std::uint32_t run = r_base + static_cast<std::uint32_t>(rr);
  const std::uint32_t q = q_base + static_cast<std::uint32_t>(qq);
  std::uint32_t cp = 0u;
  if (run < static_cast<std::uint32_t>(epoch_count) &&
      static_cast<int>(run) != skip_run && q < kEpochQuotients)
    cp = epochs[run].quotient_count_prefix[q];
  t_cp[rr][qq] = cp;
  __syncthreads();
  const int wq = threadIdx.x / kTransposeRuns;
  const int wr = threadIdx.x % kTransposeRuns;
  const std::uint32_t out_q = q_base + static_cast<std::uint32_t>(wq);
  const std::uint32_t out_run = r_base + static_cast<std::uint32_t>(wr);
  if (out_q >= kEpochQuotients ||
      out_run >= static_cast<std::uint32_t>(epoch_count))
    return;
  cp_t[static_cast<std::size_t>(out_q) * kRunStride + out_run] =
      t_cp[wr][wq];
}

// Pack 32/W queries per warp for small run counts.
__global__ void run_subwarp_range_kernel(
    const std::uint32_t *lo, const std::uint32_t *hi, std::uint32_t *out_sums,
    std::uint32_t *out_counts, std::size_t query_count,
    const LogicalRunView *runs, int run_count, int width,
    const std::uint32_t *off_t, const std::uint32_t *vp_t,
    const std::uint32_t *cp_t) {
  const int lane = threadIdx.x & 31;
  const int sub = lane / width;
  const int s = lane % width;
  const unsigned sub_bits =
      width == 32 ? 0xffffffffu : (1u << width) - 1u;
  const unsigned sub_mask = sub_bits << (sub * width);
  const std::size_t query =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
      static_cast<std::size_t>(width);
  if (query >= query_count)
    return;
  const std::uint32_t l = lo[query];
  const std::uint32_t h = hi[query];
  std::uint32_t sum = 0u;
  std::uint32_t count = 0u;
  if (l <= h && s < run_count) {
    const LogicalRunView run = runs[s];
    sum = logical_run_sum_transposed(run, off_t, vp_t,
                                     static_cast<std::uint32_t>(s), l, h);
    if (out_counts) {
      count = logical_run_count_transposed(
          run, off_t, cp_t, static_cast<std::uint32_t>(s), l, h);
    }
  }
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(sub_mask, sum, offset, width);
    count += __shfl_down_sync(sub_mask, count, offset, width);
  }
  if (s == 0) {
    out_sums[query] = sum;
    if (out_counts)
      out_counts[query] = count;
  }
}

// One block per query; one thread per logical run.
__global__ void run_parallel_range_kernel(
    const std::uint32_t *lo, const std::uint32_t *hi, std::uint32_t *out_sums,
    std::uint32_t *out_counts, std::size_t query_count,
    const LogicalRunView *runs, int run_count, const std::uint32_t *off_t,
    const std::uint32_t *vp_t, const std::uint32_t *cp_t) {
  __shared__ std::uint32_t warp_sums[4];
  __shared__ std::uint32_t warp_counts[4];
  const std::size_t query = blockIdx.x;
  if (query >= query_count)
    return;
  const std::uint32_t l = lo[query];
  const std::uint32_t h = hi[query];
  const int t = threadIdx.x;
  std::uint32_t sum = 0u;
  std::uint32_t count = 0u;
  if (l <= h && t < run_count) {
    const LogicalRunView run = runs[t];
    sum = logical_run_sum_transposed(run, off_t, vp_t,
                                     static_cast<std::uint32_t>(t), l, h);
    if (out_counts) {
      count = logical_run_count_transposed(run, off_t, cp_t,
                                           static_cast<std::uint32_t>(t), l, h);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
    count += __shfl_down_sync(0xffffffffu, count, offset);
  }
  if ((t & 31) == 0) {
    warp_sums[t >> 5] = sum;
    warp_counts[t >> 5] = count;
  }
  __syncthreads();
  if (t == 0) {
    const int warps = blockDim.x >> 5;
    for (int w = 1; w < warps; ++w) {
      sum += warp_sums[w];
      count += warp_counts[w];
    }
    out_sums[query] = sum;
    if (out_counts)
      out_counts[query] = count;
  }
}

__global__ void
run_sequential_range_kernel(const std::uint32_t *lo, const std::uint32_t *hi,
                            std::uint32_t *out_sums, std::uint32_t *out_counts,
                            std::size_t query_count, const LogicalRunView *runs,
                            int run_count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= query_count)
    return;
  const std::uint32_t l = lo[i];
  const std::uint32_t h = hi[i];
  if (l > h) {
    out_sums[i] = 0u;
    if (out_counts)
      out_counts[i] = 0u;
    return;
  }
  std::uint32_t sum = 0u;
  std::uint32_t count = 0u;
  for (int run = 0; run < run_count; ++run) {
    const LogicalRunView view = runs[run];
    sum += logical_run_sum(view, l, h);
    if (out_counts)
      count += logical_run_count(view, l, h);
  }
  out_sums[i] = sum;
  if (out_counts)
    out_counts[i] = count;
}

__global__ void range_projection_query_kernel(
    const std::uint32_t *lo, const std::uint32_t *hi, std::uint32_t *out_sums,
    std::size_t query_count, SortedRunView sorted,
    SortedRunRangeView sorted_range, const std::uint32_t *sorted_prefix,
    RangeDeltaView projection) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= query_count)
    return;
  const std::uint32_t l = lo[i];
  const std::uint32_t h = hi[i];
  if (l > h) {
    out_sums[i] = 0u;
    return;
  }
  out_sums[i] = sorted_range_sum(sorted, sorted_range, sorted_prefix, l, h) +
                range_delta_sum(projection, l, h);
}

// Immutable assignment run seen by device readers.
struct AssignmentRunView {
  const std::uint32_t *keys;
  const std::uint32_t *values;
  const std::uint32_t *offsets;
  const std::uint32_t *op_words; // null for homogeneous leaves
  std::uint8_t constant_op;      // 1 insert, 0 delete
  std::uint8_t mixed;            // 1 if op_words is used
};

// Scalar op for a leaf, or one packed bit for a mixed run.
struct PinnedHostState {
  AssignmentRunView views[kRunCapacity];
  std::uint32_t narrow_overflow;
  std::uint32_t resolved_count;
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
    total += runs[r].offsets[q + 1u] - runs[r].offsets[q];
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
    const std::uint32_t begin = run.offsets[q];
    const std::uint32_t count = run.offsets[q + 1u] - begin;
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
    const std::uint32_t begin = run.offsets[q];
    const std::uint32_t count = run.offsets[q + 1u] - begin;
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

// Scan newest first, then fall through to BaseRun.
__global__ void temporal_lookup_kernel(const AssignmentRunView *runs,
                                       int run_count, SortedRunView base,
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
    const std::uint32_t b = run.offsets[q];
    std::uint32_t p = run.offsets[q + 1u];
    while (p-- > b) {
      if (run.keys[p] == key) {
        const bool live =
            assignment_op_at(run.op_words, run.constant_op, run.mixed, p) != 0;
        out_values[i] = live ? run.values[p] : kEmptyKey;
        if (out_found)
          out_found[i] = live ? 1u : 0u;
        return;
      }
    }
  }
  std::uint32_t value = 0u;
  if (sorted_find_value(base, key, &value)) {
    out_values[i] = value;
    if (out_found)
      out_found[i] = 1u;
    return;
  }
  out_values[i] = kEmptyKey;
  if (out_found)
    out_found[i] = 0u;
}

// Pack one run into the resolve staging arrays. Concatenation
// order (oldest run first, input order within) is the timestamp.
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

// Boundary-only offsets for an assignment leaf: no masks, no
// pending list, no per-quotient counts.
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

// After a stable full-key sort, the last record of each equal-key
// group is the newest assignment.
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

// Convert sorted assignments to timestamped base corrections.
__global__ void normalize_correction_kernel(
    const std::uint32_t *keys, const std::uint64_t *assignment,
    std::size_t count, SortedRunView base,
    std::uint64_t *tagged_keys, std::uint64_t *corrections) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::uint32_t key = keys[i];
  const std::uint32_t value =
      static_cast<std::uint32_t>(assignment[i] >> 32);
  const bool insert = (assignment[i] & 1u) != 0u;
  std::uint32_t base_value = 0u;
  bool base_live = false;
  if (base.count != 0u) {
    std::size_t begin = 0u;
    std::size_t end = 0u;
    sorted_search_bounds(base, key, &begin, &end);
    const std::size_t local =
        lower_bound_u32(base.keys + begin, end - begin, key);
    const std::size_t position = begin + local;
    if (position < end && base.keys[position] == key) {
      base_live = true;
      base_value = base.values[position];
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

// Split latest assignments into absolute value + keep flag.
// A key whose latest op is delete is dropped from the new base.
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
    std::size_t query_count, SortedRunView base,
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
  std::uint32_t sum =
      sorted_range_sum(base, base_range, base_value_prefix, l, h);
  if (resolved_ready)
    sum += resolved_value_prefix[delta_upper] -
           resolved_value_prefix[delta_lower];
  out_sums[i] = sum;
  if constexpr (WithCounts) {
    std::uint32_t count =
        sorted_range_count(base, base_count_prefix, l, h);
    if (resolved_ready)
      count += resolved_count_prefix[delta_upper] -
               resolved_count_prefix[delta_lower];
    out_counts[i] = count;
  }
}

// One block maps its lanes directly to chronological runs.
__global__ void temporal_lookup_run_parallel_kernel(
    const AssignmentRunView *runs, int run_count, SortedRunView base,
    const std::uint32_t *queries, std::size_t n,
    std::uint32_t *out_values, std::uint8_t *out_found) {
  const std::size_t query = blockIdx.x;
  if (query >= n)
    return;
  __shared__ unsigned long long candidates[kRunCapacity];
  const int run_index = threadIdx.x;
  const std::uint32_t key = queries[query];
  unsigned long long candidate = 0u;
  if (run_index < run_count) {
    const AssignmentRunView run = runs[run_index];
    const std::uint32_t q = key >> kEpochQuotientBits;
    const std::uint32_t begin = run.offsets[q];
    std::uint32_t position = run.offsets[q + 1u];
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
  found = sorted_find_value(base, key, &value) ? 1u : 0u;
  if (!found)
    value = kEmptyKey;
  out_values[query] = value;
  if (out_found)
    out_found[query] = found;
}

constexpr int kNarrowSeenCap = 128;
constexpr int kNarrowHashSlots = 2 * kNarrowSeenCap;
static_assert((kNarrowHashSlots & (kNarrowHashSlots - 1)) == 0);

__device__ inline void narrow_record_latest(
    std::uint32_t *keys, unsigned long long *references,
    std::uint32_t key, unsigned long long reference) {
  std::uint32_t slot =
      static_cast<std::uint32_t>(spill_slot(key, kNarrowHashSlots - 1u));
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
    int run_count, SortedRunView base_view,
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
  std::uint32_t candidates = 0u;
  for (int r = lane; r < run_count; r += blockDim.x) {
    const AssignmentRunView run = runs[r];
    candidates += run.offsets[qhi + 1u] - run.offsets[qlo];
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
    const std::uint32_t begin = run.offsets[qlo];
    const std::uint32_t end = run.offsets[qhi + 1u];
    for (std::uint32_t p = begin; p < end; ++p) {
      const std::uint32_t key = run.keys[p];
      if (key < l || key > h)
        continue;
      const unsigned long long reference =
          (static_cast<unsigned long long>(lane + 1) << 32) | p;
      narrow_record_latest(hash_keys, hash_refs, key, reference);
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
    std::uint32_t base_value = 0u;
    const bool base_live = sorted_find_value(base_view, key, &base_value);
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
        sorted_range_sum(base_view, base_range, base_value_prefix, l, h) +
        reduce_sum[0];
    if (out_counts) {
      const std::uint32_t base_count =
          sorted_range_count(base_view, base_count_prefix, l, h);
      out_counts[query] =
          base_count + static_cast<std::uint32_t>(reduce_count[0]);
    }
  }
}

template <bool WithCounts>
__global__ void base_only_range_kernel(
    const std::uint32_t *lo, const std::uint32_t *hi,
    std::uint32_t *out_sums, std::uint32_t *out_counts,
    std::size_t query_count, SortedRunView base,
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
      sorted_range_sum(base, base_range, base_value_prefix, l, h);
  if constexpr (WithCounts)
    out_counts[i] =
        sorted_range_count(base, base_count_prefix, l, h);
}

} // namespace gpulsmopt_detail

class GPULSMOpt {
public:
  using DeviceKeyBatch = gpulsmopt_detail::DeviceKeyBatch;

explicit GPULSMOpt(const DictionaryConfig &config)
      : max_elements_(config.max_elements),
        batch_capacity_(config.batch_capacity) {
    if (max_elements_ > 0x7fffffffu) {
      throw std::invalid_argument(
          "GPULSMOpt currently supports at most 2^31-1 records");
    }
    runs_.reserve(gpulsmopt_detail::kRunCapacity);
    run_pool_.reserve(gpulsmopt_detail::kRunCapacity);
    run_views_.reserve(gpulsmopt_detail::kRunCapacity);
    run_views_.resize(gpulsmopt_detail::kRunCapacity);
    bound_run_views_.resize(gpulsmopt_detail::kRunCapacity);
    bound_run_view_valid_.resize(gpulsmopt_detail::kRunCapacity);
    logical_run_views_.resize_discard_exact(gpulsmopt_detail::kRunCapacity);
    bound_logical_run_views_.resize(gpulsmopt_detail::kRunCapacity);
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
    succ_built_ = false;
    ++base_generation_;
    succ_bits_.release();
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
      const double measured =
          prof_append_ms_ + prof_delta_sort_ms_ + prof_delta_ingest_ms_;
      const double other = total - measured;
      auto pct = [total](double x) {
        return total > 0.0 ? 100.0 * x / total : 0.0;
      };
      printf("[prof] insert %zu keys: total=%.3f ms\n", batch.count, total);
      printf("[prof]   delta_sort  = %.3f ms (%5.1f%%)\n", prof_delta_sort_ms_,
             pct(prof_delta_sort_ms_));
      printf("[prof]   delta_write = %.3f ms (%5.1f%%)\n",
             prof_delta_ingest_ms_, pct(prof_delta_ingest_ms_));
      printf("[prof]   append      = %.3f ms (%5.1f%%)\n", prof_append_ms_,
             pct(prof_append_ms_));
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
          assignment_views_.data(), run_count, make_sorted_view(),
          batch.queries, batch.count,
          batch.out_values, batch.out_found);
    } else {
      const int block = 256;
      const int grid = static_cast<int>((batch.count + block - 1) / block);
      gpulsmopt_detail::temporal_lookup_kernel<<<grid, block, 0, stream>>>(
          assignment_views_.data(), run_count, make_sorted_view(),
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
    ensure_successor_bitmap(stream);
    const int block = 256;
    const int grid = static_cast<int>((batch.count + block - 1) / block);
    gpulsmopt_detail::succ_query_kernel<<<grid, block, 0, stream>>>(
        batch.queries, batch.count, batch.out_keys, succ_bits_.data());
    CUDA_CHECK(cudaGetLastError());
  }

  void range(const DeviceRangeOutputBatch &batch, cudaStream_t stream) {
    if (batch.query_count == 0)
      return;
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    order_stream_locked(stream);
    ensure_sorted_run_cache(stream);
    const int run_count = static_cast<int>(chrono_views_.size());
    // No updates: BaseRun-only, no resolve/metadata/count-prefix work.
    if (run_count == 0) {
      const int block = 128;
      const int grid =
          static_cast<int>((batch.query_count + block - 1) / block);
      if (batch.out_counts) {
        gpulsmopt_detail::base_only_range_kernel<true><<<
            grid, block, 0, stream>>>(
            batch.lo, batch.hi, batch.out_sums, batch.out_counts,
            batch.query_count, make_sorted_view(),
            make_sorted_range_view(), sorted_value_prefix_.data(),
            sorted_count_prefix_.data());
      } else {
        gpulsmopt_detail::base_only_range_kernel<false><<<
            grid, block, 0, stream>>>(
            batch.lo, batch.hi, batch.out_sums, nullptr,
            batch.query_count, make_sorted_view(),
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
          make_sorted_view(), make_sorted_range_view(),
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
          batch.query_count, make_sorted_view(),
          make_sorted_range_view(), sorted_value_prefix_.data(),
          sorted_count_prefix_.data(), make_run_view(resolved_),
          resolved_value_prefix_.data(),
          resolved_count_prefix_.data(),
          resolved_.count > 0 ? 1 : 0);
    } else {
      gpulsmopt_detail::resolved_range_kernel<false><<<
          grid, block, 0, stream>>>(
          batch.lo, batch.hi, batch.out_sums, nullptr,
          batch.query_count, make_sorted_view(),
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
    succ_built_ = false;
    ++base_generation_;
    succ_bits_.release();
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
        run_views_, range_delta_counts_, range_delta_sums_,
        range_delta_offsets_, range_delta_value_prefix_,
        range_delta_keys_, range_delta_values_, resolve_keys_,
        resolve_payload_, resolve_alt_keys_, resolve_alt_payload_,
        resolve_flags_, resolve_sel_vdelta_, resolve_count_,
        normalize_views_, norm_keys_, norm_pay_,
        cache_pay_, merge_out_keys_, merge_out_pay_, merge_flags_,
        merge_sel_keys_, merge_sel_pay_, compaction_counts_,
        compaction_offsets_, compaction_tile_offsets_,
        compaction_unique_counts_, compaction_unique_offsets_,
        compaction_unique_cursors_, narrow_overflow_, assignment_views_,
        direct_sort_keys_,
        direct_sort_values_, sort_key_output_, sort_temp_storage_,
        sorted_value_prefix_, sorted_count_prefix_, base_rank23_,
        sorted_range_cdf_, owner_primary_, owner_spill_keys_,
        owner_spill_values_, owner_spill_count_, owner_quotient_live_,
        meta_pending_list_, meta_pending_count_, owner_class_list_,
        owner_class_count_, logical_run_views_, run_off_t_, run_vp_t_,
        run_cp_t_, succ_bits_);
    total += device_bytes_all(
        resolved_.keys, resolved_.values, resolved_.count_delta,
        resolved_value_prefix_, resolved_count_prefix_,
        resolved_.quotient_count_sum, resolved_.quotient_off,
        resolved_.subgroup_masks, resolved_.quotient_live,
        resolved_.quotient_value_sum, resolved_.quotient_count_prefix,
        resolved_.quotient_value_prefix,
        resolved_.subgroup_value_prefix, resolved_.heavy_list,
        resolved_.heavy_count, resolved_.op_words);
    for (const auto &epoch : runs_)
      total += device_bytes_all(
          epoch.keys, epoch.values, epoch.count_delta,
          epoch.quotient_count_sum, epoch.quotient_off,
          epoch.subgroup_masks, epoch.quotient_live,
          epoch.quotient_value_sum, epoch.quotient_count_prefix,
          epoch.quotient_value_prefix, epoch.subgroup_value_prefix,
          epoch.heavy_list, epoch.heavy_count, epoch.op_words);
    for (const auto &epoch : run_pool_)
      total += device_bytes_all(
          epoch.keys, epoch.values, epoch.count_delta,
          epoch.quotient_count_sum, epoch.quotient_off,
          epoch.subgroup_masks, epoch.quotient_live,
          epoch.quotient_value_sum, epoch.quotient_count_prefix,
          epoch.quotient_value_prefix, epoch.subgroup_value_prefix,
          epoch.heavy_list, epoch.heavy_count, epoch.op_words);
    return total;
  }

private:
  struct AssignmentLeafStorage {
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> keys;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> values;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_off;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> op_words;
  };

  struct RunStorage : AssignmentLeafStorage {
    gpulsmopt_detail::RawDeviceBuffer<std::int8_t> count_delta;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> subgroup_masks;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_live;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_count_sum;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_value_sum;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_count_prefix;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_value_prefix;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> subgroup_value_prefix;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> heavy_list;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> heavy_count;
    std::size_t count = 0;
    // Temporal identity of an immutable assignment run.
    std::uint64_t sequence = 0;
    gpulsmopt_detail::RunOperation operation =
        gpulsmopt_detail::RunOperation::Insert;
    bool mixed = false;
    bool assignment = false;
    bool heavy_sorted = true;
    bool value_sums_ready = true;
    bool value_prefix_ready = true;
    bool subgroup_value_prefix_ready = true;
    bool count_prefix_ready = true;
    bool fully_sorted = false;
    bool unit_counts = false;
    bool unique_keys = false;
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

  template <class T> static T *raw_or_null(thrust::device_vector<T> &v) {
    return v.empty() ? nullptr : thrust::raw_pointer_cast(v.data());
  }
  template <class T>
  static const T *raw_or_null(const thrust::device_vector<T> &v) {
    return v.empty() ? nullptr : thrust::raw_pointer_cast(v.data());
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
    if (v.size() < n)
      v.resize(n);
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

  gpulsmopt_detail::SortedRunRangeView make_sorted_range_view() const {
    return {sorted_range_cdf_ready_ ? sorted_range_cdf_.data() : nullptr,
            sorted_range_min_key_, sorted_range_span_};
  }

  gpulsmopt_detail::OwnerView make_owner_view() {
    return {owner_primary_.data(),     owner_spill_keys_.data(),
            owner_spill_values_.data(), owner_spill_count_.data(),
            owner_quotient_live_.data(),
            static_cast<std::uint64_t>(owner_spill_slots_ - 1u)};
  }

  void initialize_owner_storage(cudaStream_t stream,
                                bool clear_primary = true) {
    const std::size_t primary_count =
        static_cast<std::size_t>(gpulsmopt_detail::kEpochQuotients) *
        gpulsmopt_detail::kOwnerSlots;
    const std::size_t capacity =
        std::max({std::size_t{1}, max_elements_, batch_capacity_});
    // Below 50% load: two spill slots per possible key.
    std::size_t spill_slots = 1;
    while (spill_slots < capacity * 2u)
      spill_slots <<= 1;
    owner_spill_slots_ = spill_slots;
    owner_primary_.resize_discard_exact(primary_count);
    owner_spill_keys_.resize_discard_exact(spill_slots);
    owner_spill_values_.resize_discard_exact(spill_slots);
    owner_spill_count_.resize_discard_exact(gpulsmopt_detail::kEpochQuotients);
    owner_quotient_live_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients);
    if (clear_primary) {
      CUDA_CHECK(cudaMemsetAsync(owner_primary_.data(), 0,
                                 owner_primary_.size() * sizeof(std::uint64_t),
                                 stream));
    }
    CUDA_CHECK(cudaMemsetAsync(owner_spill_keys_.data(), 0,
                               spill_slots * sizeof(std::uint64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(owner_spill_count_.data(), 0,
                               gpulsmopt_detail::kEpochQuotients *
                                   sizeof(std::uint32_t),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(
        owner_quotient_live_.data(), 0,
        owner_quotient_live_.size() * sizeof(std::uint32_t), stream));
    owner_ready_ = true;
  }

  void build_owner_index(cudaStream_t stream) {
    // The build kernel writes every primary slot itself.
    const bool have_run = !runs_.empty();
    initialize_owner_storage(stream, !have_run);
    if (!have_run)
      return;
    RunStorage &run = runs_.front();
    gpulsmopt_detail::owner_build_kernel<<<gpulsmopt_detail::kEpochQuotients,
                                           256, 0, stream>>>(
        make_owner_view(), run.keys.data(), run.values.data(),
        run.quotient_off.data());
    CUDA_CHECK(cudaGetLastError());
  }

  void clear_owner_state(cudaStream_t stream) {
    if (owner_primary_.size() > 0) {
      CUDA_CHECK(cudaMemsetAsync(owner_primary_.data(), 0,
                                 owner_primary_.size() * sizeof(std::uint64_t),
                                 stream));
      CUDA_CHECK(cudaMemsetAsync(owner_spill_keys_.data(), 0,
                                 owner_spill_keys_.size() *
                                     sizeof(std::uint64_t),
                                 stream));
      CUDA_CHECK(cudaMemsetAsync(owner_spill_count_.data(), 0,
                                 owner_spill_count_.size() *
                                     sizeof(std::uint32_t),
                                 stream));
      CUDA_CHECK(cudaMemsetAsync(
          owner_quotient_live_.data(), 0,
          owner_quotient_live_.size() * sizeof(std::uint32_t), stream));
    }
    owner_ready_ = false;
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
    run_meta_ready_ = false;
    run_meta_counts_ready_ = false;
    logical_run_views_ready_ = false;
  }

  // Flat 23-bit rank directory, rebuilt only when the base changes.
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
    // Reverse-min inclusive scan fills empty bins with the next start.
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
    constexpr int block = 256;
    sorted_value_prefix_.resize_discard(count + 1u);
    sorted_count_prefix_.resize_discard(run.unit_counts ? 0u : count + 1u);
    CUDA_CHECK(cudaMemsetAsync(sorted_value_prefix_.data(), 0,
                               sizeof(std::uint32_t), stream));
    if (!run.unit_counts) {
      CUDA_CHECK(cudaMemsetAsync(sorted_count_prefix_.data(), 0,
                                 sizeof(std::uint32_t), stream));
    }
    if (count == 0)
      return;
    auto policy = thrust::cuda::par.on(stream);
    thrust::inclusive_scan(policy, run.values.data(), run.values.data() + count,
                           sorted_value_prefix_.data() + 1u);
    if (!run.unit_counts) {
      const int grid = static_cast<int>((count + block - 1u) / block);
      gpulsmopt_detail::
          run_count_prefix_input_kernel<<<grid, block, 0, stream>>>(
              run.count_delta.data(), count, sorted_count_prefix_.data());
      CUDA_CHECK(cudaGetLastError());
      thrust::inclusive_scan(policy, sorted_count_prefix_.data() + 1u,
                             sorted_count_prefix_.data() + count + 1u,
                             sorted_count_prefix_.data() + 1u);
    }
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
    invalidate_range_delta_projection();
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

  const gpulsmopt_detail::RunView *run_view_ptr() const {
    return raw_or_null(run_views_);
  }

  int active_run_count() const { return static_cast<int>(runs_.size()); }

  std::size_t run_count() const { return runs_.size(); }

  gpulsmopt_detail::RunView make_run_view(RunStorage &epoch) {
    return {raw_or_null(epoch.keys),
            raw_or_null(epoch.values),
            raw_or_null(epoch.count_delta),
            raw_or_null(epoch.quotient_off),
            raw_or_null(epoch.subgroup_masks),
            raw_or_null(epoch.quotient_live),
            raw_or_null(epoch.quotient_count_sum),
            raw_or_null(epoch.quotient_value_sum),
            raw_or_null(epoch.quotient_count_prefix),
            raw_or_null(epoch.quotient_value_prefix),
            epoch.subgroup_value_prefix_ready
                ? raw_or_null(epoch.subgroup_value_prefix)
                : nullptr,
            raw_or_null(epoch.heavy_list),
            raw_or_null(epoch.heavy_count),
            epoch.fully_sorted ? 1u : 0u};
  }

  gpulsmopt_detail::LogicalRunView make_sorted_run_view() const {
    return {make_sorted_view(),
            make_sorted_range_view(),
            sorted_value_prefix_.data(),
            sorted_count_prefix_.data(),
            {},
            gpulsmopt_detail::kSortedRunCache};
  }

  gpulsmopt_detail::LogicalRunView make_quotient_run_view(RunStorage &run) {
    return {{},
            {},
            nullptr,
            nullptr,
            make_run_view(run),
            gpulsmopt_detail::kQuotientRun};
  }

  int logical_run_count() const { return active_run_count(); }

  void ensure_logical_run_views(cudaStream_t stream) {
    if (logical_run_views_ready_)
      return;
    for (std::size_t r = 0; r < runs_.size(); ++r) {
      bound_logical_run_views_[r] = r == sorted_run_index_
                                        ? make_sorted_run_view()
                                        : make_quotient_run_view(runs_[r]);
    }
    const std::size_t count = runs_.size();
    if (count > 0) {
      CUDA_CHECK(cudaMemcpyAsync(logical_run_views_.data(),
                                 bound_logical_run_views_.data(),
                                 count * sizeof(bound_logical_run_views_[0]),
                                 cudaMemcpyHostToDevice, stream));
    }
    logical_run_views_ready_ = true;
  }

  bool should_use_run_parallel(std::size_t queries) const {
    const std::size_t runs = static_cast<std::size_t>(logical_run_count());
    const std::size_t sequential_warps = (queries + 31u) / 32u;
    const std::size_t parallel_warps = queries * ((runs + 31u) / 32u);
    const std::size_t resident = gpulsmopt_detail::kGpuResidentWarps;
    const std::size_t sequential_waves =
        (sequential_warps + resident - 1u) / resident;
    const std::size_t parallel_waves =
        (parallel_warps + resident - 1u) / resident;
    const std::size_t sequential_cost = sequential_waves * runs;
    const std::size_t parallel_cost = parallel_waves * 3u;
    return parallel_cost < sequential_cost;
  }
  gpulsmopt_detail::RangeDeltaView make_range_delta_view() const {
    return {range_delta_offsets_.data(), range_delta_value_prefix_.data(),
            range_delta_keys_.data(), range_delta_values_.data()};
  }

  bool should_use_range_delta_projection(std::size_t queries,
                                         bool wants_counts) const {
    const std::size_t projected_runs =
        runs_.size() - static_cast<std::size_t>(sorted_run_ready());
    return !wants_counts && projected_runs >= 2u &&
           queries >= gpulsmopt_detail::kRangeProjectionMinQueries;
  }

  void invalidate_range_delta_projection() {
    range_delta_projection_ready_ = false;
  }

  void build_range_delta_projection(cudaStream_t stream) {
    if (range_delta_projection_ready_)
      return;
    constexpr int bins = gpulsmopt_detail::kRangeProjectionBins;
    constexpr int block = 256;
    constexpr int warps = block / 32;
    constexpr int quotient_grid = gpulsmopt_detail::kEpochQuotients / warps;
    range_delta_counts_.resize_discard(bins + 1u);
    range_delta_sums_.resize_discard(bins + 1u);
    range_delta_offsets_.resize_discard(bins + 1u);
    range_delta_value_prefix_.resize_discard(bins + 1u);
    CUDA_CHECK(cudaMemsetAsync(range_delta_counts_.data() + bins, 0,
                               sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(range_delta_sums_.data() + bins, 0,
                               sizeof(std::uint32_t), stream));
    const int skip_run =
        sorted_run_ready() ? static_cast<int>(sorted_run_index_) : -1;
    gpulsmopt_detail::
        range_delta_plan_kernel<<<quotient_grid, block, 0, stream>>>(
            run_view_ptr(), active_run_count(), skip_run,
            range_delta_counts_.data(), range_delta_sums_.data());
    CUDA_CHECK(cudaGetLastError());
    exclusive_scan_u32(range_delta_counts_.data(), range_delta_offsets_.data(),
                       bins + 1u, stream);
    exclusive_scan_u32(range_delta_sums_.data(),
                       range_delta_value_prefix_.data(), bins + 1u, stream);
    std::size_t physical_count = 0;
    for (std::size_t r = 0; r < runs_.size(); ++r)
      if (static_cast<int>(r) != skip_run)
        physical_count += runs_[r].count;
    range_delta_keys_.resize_discard(physical_count);
    range_delta_values_.resize_discard(physical_count);
    if (physical_count > 0) {
      // Block per quotient: bins write contiguously.
      gpulsmopt_detail::
          range_delta_pack_kernel<<<gpulsmopt_detail::kEpochQuotients,
                                    gpulsmopt_detail::kPackThreads, 0,
                                    stream>>>(
              run_view_ptr(), active_run_count(), skip_run,
              range_delta_offsets_.data(), range_delta_keys_.data(),
              range_delta_values_.data());
      CUDA_CHECK(cudaGetLastError());
    }
    range_delta_projection_ready_ = true;
  }

  static bool same_run_view(const gpulsmopt_detail::RunView &a,
                            const gpulsmopt_detail::RunView &b) {
    return a.keys == b.keys && a.values == b.values &&
           a.count_delta == b.count_delta && a.quotient_off == b.quotient_off &&
           a.subgroup_masks == b.subgroup_masks &&
           a.quotient_live == b.quotient_live &&
           a.quotient_count_sum == b.quotient_count_sum &&
           a.quotient_value_sum == b.quotient_value_sum &&
           a.quotient_count_prefix == b.quotient_count_prefix &&
           a.quotient_value_prefix == b.quotient_value_prefix &&
           a.subgroup_value_prefix == b.subgroup_value_prefix &&
           a.heavy_list == b.heavy_list && a.heavy_count == b.heavy_count &&
           a.fully_sorted == b.fully_sorted;
  }

  void upload_run_view(std::size_t index, const gpulsmopt_detail::RunView &view,
                       cudaStream_t stream) {
    bound_run_views_[index] = view;
    bound_run_view_valid_[index] = 1u;
    CUDA_CHECK(cudaMemcpyAsync(raw_or_null(run_views_) + index,
                               bound_run_views_.data() + index, sizeof(view),
                               cudaMemcpyHostToDevice, stream));
  }

  void append_run_view(RunStorage &epoch, cudaStream_t stream) {
    const std::size_t index = runs_.size() - 1;
    const gpulsmopt_detail::RunView view = make_run_view(epoch);
    if (bound_run_view_valid_[index] &&
        same_run_view(bound_run_views_[index], view))
      return;
    upload_run_view(index, view, stream);
  }

  void prebind_run_views(cudaStream_t stream) {
    std::size_t index = 0;
    for (auto &epoch : runs_) {
      bound_run_views_[index] = make_run_view(epoch);
      bound_run_view_valid_[index] = 1u;
      ++index;
    }
    for (auto it = run_pool_.rbegin(); it != run_pool_.rend(); ++it) {
      bound_run_views_[index] = make_run_view(*it);
      bound_run_view_valid_[index] = 1u;
      ++index;
    }
    std::fill(bound_run_view_valid_.begin() + index,
              bound_run_view_valid_.end(), 0u);
    if (index == 0)
      return;
    CUDA_CHECK(cudaMemcpyAsync(raw_or_null(run_views_), bound_run_views_.data(),
                               index * sizeof(bound_run_views_[0]),
                               cudaMemcpyHostToDevice, stream));
  }

  void refresh_active_run_views(cudaStream_t stream) {
    for (std::size_t i = 0; i < runs_.size(); ++i) {
      const auto view = make_run_view(runs_[i]);
      if (bound_run_view_valid_[i] && same_run_view(bound_run_views_[i], view))
        continue;
      upload_run_view(i, view, stream);
    }
  }

  void prepare_run_metadata_storage(RunStorage &epoch) {
    epoch.heavy_sorted = epoch.fully_sorted;
    epoch.value_sums_ready = false;
    epoch.value_prefix_ready = false;
    epoch.subgroup_value_prefix_ready = false;
    epoch.count_prefix_ready = false;
    epoch.count_delta.resize_discard(epoch.count);
    epoch.quotient_off.resize_discard(gpulsmopt_detail::kEpochQuotients + 1);
    epoch.subgroup_masks.resize_discard(gpulsmopt_detail::kEpochQuotients *
                                        gpulsmopt_detail::kEpochSubgroupPlanes);
    epoch.quotient_live.resize_discard(gpulsmopt_detail::kEpochQuotients);
    epoch.quotient_count_sum.resize_discard(gpulsmopt_detail::kEpochQuotients);
    epoch.quotient_value_sum.resize_discard(gpulsmopt_detail::kEpochQuotients);
    epoch.quotient_count_prefix.resize_discard(
        gpulsmopt_detail::kEpochQuotients + 1);
    epoch.quotient_value_prefix.resize_discard(
        gpulsmopt_detail::kEpochQuotients + 1);
    epoch.subgroup_value_prefix.resize_discard(0);
    epoch.heavy_list.resize_discard(gpulsmopt_detail::kEpochQuotients);
    epoch.heavy_count.resize_discard(1);
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

  void launch_run_metadata_kernels(RunStorage &epoch, std::size_t count,
                                   cudaStream_t stream,
                                   bool build_owner_classes) {
    constexpr int block = 256;
    const std::uint32_t *keys = epoch.keys.data();
    std::uint32_t record_count = static_cast<std::uint32_t>(count);
    std::uint32_t *offsets = epoch.quotient_off.data();
    std::uint32_t *subgroup_masks = epoch.subgroup_masks.data();
    std::uint32_t *quotient_live = epoch.quotient_live.data();
    if (build_owner_classes) {
      owner_class_list_.resize_discard(4u * gpulsmopt_detail::kEpochQuotients);
      owner_class_count_.resize_discard(4);
      CUDA_CHECK(cudaMemsetAsync(owner_class_count_.data(), 0,
                                 4u * sizeof(std::uint32_t), stream));
    }
    if (record_count == 0u) {
      CUDA_CHECK(cudaMemsetAsync(offsets, 0,
                                 (gpulsmopt_detail::kEpochQuotients + 1u) *
                                     sizeof(std::uint32_t),
                                 stream));
      CUDA_CHECK(cudaMemsetAsync(subgroup_masks, 0,
                                 gpulsmopt_detail::kEpochQuotients *
                                     gpulsmopt_detail::kEpochSubgroupPlanes *
                                     sizeof(std::uint32_t),
                                 stream));
      CUDA_CHECK(cudaMemsetAsync(
          quotient_live, 0,
          gpulsmopt_detail::kEpochQuotients * sizeof(std::uint32_t), stream));
      return;
    }
    meta_pending_list_.resize_discard((count + 31u) / 32u + 1u);
    meta_pending_count_.resize_discard(1);
    CUDA_CHECK(cudaMemsetAsync(offsets, 0xff,
                               (gpulsmopt_detail::kEpochQuotients + 1u) *
                                   sizeof(std::uint32_t),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(meta_pending_count_.data(), 0,
                               sizeof(std::uint32_t), stream));
    const int grid = static_cast<int>((record_count + block - 1) / block);
    gpulsmopt_detail::run_boundary_scatter_kernel<<<grid, block, 0, stream>>>(
        keys, record_count, offsets, subgroup_masks, meta_pending_list_.data(),
        meta_pending_count_.data());
    CUDA_CHECK(cudaGetLastError());
    reverse_min_scan_offsets(offsets, stream);
    constexpr int finalize_grid = gpulsmopt_detail::kEpochQuotients / block;
    gpulsmopt_detail::
        run_finalize_metadata_kernel<<<finalize_grid, block, 0, stream>>>(
            offsets, quotient_live, epoch.quotient_count_sum.data(),
            build_owner_classes ? owner_class_list_.data() : nullptr,
            build_owner_classes ? owner_class_count_.data() : nullptr);
    CUDA_CHECK(cudaGetLastError());
    gpulsmopt_detail::run_pending_masks_kernel<<<256, block, 0, stream>>>(
        keys, offsets, meta_pending_list_.data(), meta_pending_count_.data(),
        subgroup_masks);
    CUDA_CHECK(cudaGetLastError());
  }

  void commit_run_metadata(RunStorage &epoch, cudaStream_t stream,
                           const std::uint8_t *ops = nullptr,
                           std::uint8_t constant_op = static_cast<std::uint8_t>(
                               gpulsmopt_detail::kInsert),
                           bool apply_owner = true) {
    prepare_run_metadata_storage(epoch);
    const bool adaptive_owner =
        apply_owner &&
        epoch.count <= gpulsmopt_detail::kAdaptiveTransitionMaxRecords;
    launch_run_metadata_kernels(epoch, epoch.count, stream, adaptive_owner);
    if (apply_owner) {
      if (!owner_ready_)
        throw std::runtime_error("owner directory is not initialized");
      const auto owner_view = make_owner_view();
      if (!adaptive_owner) {
        gpulsmopt_detail::owner_transition_quotient_kernel<<<
            gpulsmopt_detail::kEpochQuotients, 32, 0, stream>>>(
            owner_view, epoch.keys.data(), epoch.values.data(), ops,
            constant_op, epoch.quotient_off.data(), epoch.count_delta.data(),
            epoch.quotient_count_sum.data());
        CUDA_CHECK(cudaGetLastError());
      } else {
        const std::uint32_t *cls = owner_class_list_.data();
        const std::uint32_t *cnt = owner_class_count_.data();
        constexpr int block = 256;
        constexpr int quotients = gpulsmopt_detail::kEpochQuotients;
        gpulsmopt_detail::owner_transition_subwarp_kernel<8>
            <<<1024, block, 0, stream>>>(
                owner_view, epoch.keys.data(), epoch.values.data(), ops,
                constant_op, epoch.quotient_off.data(),
                epoch.count_delta.data(), epoch.quotient_count_sum.data(), cls,
                cnt);
        CUDA_CHECK(cudaGetLastError());
        gpulsmopt_detail::owner_transition_subwarp_kernel<16>
            <<<1024, block, 0, stream>>>(
                owner_view, epoch.keys.data(), epoch.values.data(), ops,
                constant_op, epoch.quotient_off.data(),
                epoch.count_delta.data(), epoch.quotient_count_sum.data(),
                cls + quotients, cnt + 1);
        CUDA_CHECK(cudaGetLastError());
        gpulsmopt_detail::owner_transition_subwarp_kernel<32>
            <<<1024, block, 0, stream>>>(
                owner_view, epoch.keys.data(), epoch.values.data(), ops,
                constant_op, epoch.quotient_off.data(),
                epoch.count_delta.data(), epoch.quotient_count_sum.data(),
                cls + 2 * quotients, cnt + 2);
        CUDA_CHECK(cudaGetLastError());
        gpulsmopt_detail::
            owner_transition_dense_kernel<<<1024, block, 0, stream>>>(
                owner_view, epoch.keys.data(), epoch.values.data(), ops,
                constant_op, epoch.quotient_off.data(),
                epoch.count_delta.data(), epoch.quotient_count_sum.data(),
                cls + 3 * quotients, cnt + 3);
        CUDA_CHECK(cudaGetLastError());
      }
    } else {
      constexpr int block = 256;
      constexpr int grid = gpulsmopt_detail::kEpochQuotients / block;
      gpulsmopt_detail::
          epoch_count_delta_sum_kernel<<<grid, block, 0, stream>>>(
              make_run_view(epoch));
      CUDA_CHECK(cudaGetLastError());
    }
    append_run_view(epoch, stream);
    invalidate_range_delta_projection();
    run_meta_ready_ = false;
    run_meta_counts_ready_ = false;
    logical_run_views_ready_ = false;
  }

  void ensure_run_heavy_sorted(cudaStream_t stream) {
    constexpr int block = 256;
    constexpr int classify_grid = gpulsmopt_detail::kEpochQuotients / block;
    constexpr int shards = 64;
    for (auto &epoch : runs_) {
      if (epoch.heavy_sorted || epoch.fully_sorted) {
        epoch.heavy_sorted = true;
        continue;
      }
      CUDA_CHECK(cudaMemsetAsync(epoch.heavy_count.data(), 0,
                                 sizeof(std::uint32_t), stream));
      const auto view = make_run_view(epoch);
      gpulsmopt_detail::epoch_classify_heavy_quotients_kernel<<<
          classify_grid, block, 0, stream>>>(view);
      CUDA_CHECK(cudaGetLastError());
      gpulsmopt_detail::
          epoch_sort_heavy_quotients_kernel<<<shards, 128, 0, stream>>>(view);
      CUDA_CHECK(cudaGetLastError());
      epoch.heavy_sorted = true;
    }
  }

  void ensure_run_value_prefixes(cudaStream_t stream) {
    ensure_run_value_sums(stream);
    for (std::size_t r = 0; r < runs_.size(); ++r) {
      if (r == sorted_run_index_)
        continue;
      RunStorage &epoch = runs_[r];
      if (epoch.value_prefix_ready)
        continue;
      exclusive_scan_u32(epoch.quotient_value_sum.data(),
                         epoch.quotient_value_prefix.data(),
                         gpulsmopt_detail::kEpochQuotients, stream);
      epoch.value_prefix_ready = true;
    }
  }

  void ensure_run_value_sums(cudaStream_t stream) {
    constexpr int block = 256;
    constexpr int grid = gpulsmopt_detail::kEpochQuotients / block;
    bool views_changed = false;
    for (std::size_t r = 0; r < runs_.size(); ++r) {
      if (r == sorted_run_index_)
        continue;
      RunStorage &epoch = runs_[r];
      if (epoch.value_sums_ready && epoch.subgroup_value_prefix_ready)
        continue;
      epoch.subgroup_value_prefix.resize_discard(
          gpulsmopt_detail::kEpochQuotients *
          gpulsmopt_detail::kEpochSubgroupPrefixStride);
      gpulsmopt_detail::
          epoch_subgroup_value_prefix_kernel<<<grid, block, 0, stream>>>(
              make_run_view(epoch), epoch.subgroup_value_prefix.data());
      CUDA_CHECK(cudaGetLastError());
      epoch.value_sums_ready = true;
      epoch.subgroup_value_prefix_ready = true;
      views_changed = true;
    }
    if (views_changed) {
      refresh_active_run_views(stream);
      logical_run_views_ready_ = false;
    }
  }

  void ensure_run_count_prefixes(cudaStream_t stream) {
    for (std::size_t r = 0; r < runs_.size(); ++r) {
      if (r == sorted_run_index_)
        continue;
      RunStorage &epoch = runs_[r];
      if (epoch.count_prefix_ready)
        continue;
      exclusive_scan_u32(epoch.quotient_count_sum.data(),
                         epoch.quotient_count_prefix.data(),
                         gpulsmopt_detail::kEpochQuotients, stream);
      epoch.count_prefix_ready = true;
    }
  }

  // Immutable assignment run: upper-16 sort + quotient offsets,
  // NO previous-state resolution. This is the insert/delete path.
  void create_assignment_run(bool is_insert, const std::uint32_t *keys,
                             const std::uint32_t *values, std::size_t count,
                             cudaStream_t stream) {
    if (run_count() >=
        static_cast<std::size_t>(gpulsmopt_detail::kRunCapacity) - 1u)
      compact_contiguous(stream);
    acquire_run_slot();
    RunStorage &run = runs_.back();
    run.count = count;
    run.assignment = true;
    run.mixed = false;
    run.operation = is_insert ? gpulsmopt_detail::RunOperation::Insert
                              : gpulsmopt_detail::RunOperation::Delete;
    run.sequence = ++run_sequence_;
    run.fully_sorted = false;
    run.unit_counts = false;
    run.unique_keys = false;
    run.keys.resize_discard(count);
    {
      GPULSMOPT_PROF_PHASE(prof_delta_sort_ms_);
      // Stable upper-16 sort keeps each quotient in input order.
      if (is_insert) {
        run.values.resize_discard(count);
        sort_run_batch(keys, values, count, run.keys.data(),
                       run.values.data(), stream);
      } else {
        // Deletion leaves are key-only: no value traffic.
        run.values.resize_discard(0);
        sort_delete_batch(keys, count, run.keys.data(), stream);
      }
    }
    {
      GPULSMOPT_PROF_PHASE(prof_delta_ingest_ms_);
      // Leaf metadata is quotient offsets only: no masks, no counts.
      build_assignment_offsets(run, static_cast<std::uint32_t>(count), stream);
    }
    publish_assignment_view(run, stream);
    invalidate_resolved();
    logical_run_views_ready_ = false;
    ++structure_generation_;
  }

  // Offsets-only leaf metadata; richer metadata is built only for
  // resolved delta runs and the BaseRun.
  void build_assignment_offsets(RunStorage &run, std::uint32_t count,
                                cudaStream_t stream) {
    run.quotient_off.resize_discard(gpulsmopt_detail::kEpochQuotients + 1);
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

  // Publish one chronological descriptor so the run is queryable.
  void publish_assignment_view(RunStorage &run, cudaStream_t stream) {
    gpulsmopt_detail::AssignmentRunView view{
        run.keys.data(),
        run.operation == gpulsmopt_detail::RunOperation::Insert
            ? run.values.data()
            : nullptr,
        run.quotient_off.data(),
        run.mixed ? run.op_words.data() : nullptr,
        static_cast<std::uint8_t>(
            run.operation == gpulsmopt_detail::RunOperation::Insert ? 1u : 0u),
        static_cast<std::uint8_t>(run.mixed ? 1u : 0u)};
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
    for (auto &leaf : run_pool_) {
      leaf.keys.resize_discard(count);
      leaf.values.resize_discard(count);
      leaf.quotient_off.resize_discard(
          gpulsmopt_detail::kEpochQuotients + 1);
    }
  }

  void clear_run_state() {
    clear_sorted_state();
    for (auto &epoch : runs_)
      run_pool_.push_back(std::move(epoch));
    runs_.clear();
    range_delta_counts_.resize_discard(0);
    range_delta_sums_.resize_discard(0);
    range_delta_offsets_.resize_discard(0);
    range_delta_value_prefix_.resize_discard(0);
    range_delta_keys_.resize_discard(0);
    range_delta_values_.resize_discard(0);
    range_delta_projection_ready_ = false;
    run_meta_ready_ = false;
    run_meta_counts_ready_ = false;
    logical_run_views_ready_ = false;
    ++structure_generation_;
  }

  void insert_records(const std::uint32_t *keys_in,
                      const std::uint32_t *values_in, std::uint8_t op,
                      std::size_t count, bool keys_sorted,
                      cudaStream_t stream) {
    (void)keys_sorted;
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
    if (run_capacity == 0u || max_elements_ == 0u || run_pool_.empty())
      return;
    const std::size_t run_limit =
        run_capacity > std::numeric_limits<std::size_t>::max() /
                           gpulsmopt_detail::kCompactGroup
            ? std::numeric_limits<std::size_t>::max()
            : run_capacity * gpulsmopt_detail::kCompactGroup;
    const std::size_t live_limit =
        max_elements_ > std::numeric_limits<std::size_t>::max() / 2u
            ? std::numeric_limits<std::size_t>::max()
            : max_elements_ * 2u;
    const std::size_t capacity = std::min(
        {run_limit, live_limit,
         static_cast<std::size_t>(std::numeric_limits<int>::max())});
    if (capacity == 0u)
      return;

    const std::size_t tile_capacity =
        std::min(capacity, gpulsmopt_detail::kCompactionTileRecords);
    resolve_keys_.resize_discard_exact(tile_capacity);
    resolve_payload_.resize_discard_exact(tile_capacity);
    resolve_alt_keys_.resize_discard_exact(tile_capacity);
    resolve_alt_payload_.resize_discard_exact(tile_capacity);
    RunStorage &destination = run_pool_.front();
    destination.keys.resize_discard_exact(capacity);
    destination.values.resize_discard_exact(capacity);
    destination.op_words.resize_discard_exact(
        (capacity + 31u) / 32u + 1u);

    std::size_t sort_bytes = 0u;
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, sort_bytes, resolve_keys_.data(),
        resolve_alt_keys_.data(), resolve_payload_.data(),
        resolve_alt_payload_.data(), static_cast<int>(tile_capacity),
        gpulsmopt_detail::kEpochQuotients,
        compaction_tile_offsets_.data(),
        compaction_tile_offsets_.data() + 1u,
        0, 16, stream));
    compaction_sort_count_ = tile_capacity;
    compaction_sort_segments_ = gpulsmopt_detail::kEpochQuotients;
    compaction_sort_temp_bytes_ = sort_bytes;
    ensure_sort_temp(sort_bytes);
  }

  // Reserve direct-path buffers before timed updates.
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
    prebind_run_views(stream);
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

  void ensure_run_transposed_metadata(bool wants_counts, cudaStream_t stream) {
    const bool need_counts = wants_counts && !run_meta_counts_ready_;
    if (run_meta_ready_ && !need_counts)
      return;
    const std::size_t rows = gpulsmopt_detail::kEpochQuotients + 1;
    const std::size_t capacity = rows * gpulsmopt_detail::kRunStride;
    run_off_t_.resize_discard(capacity);
    run_vp_t_.resize_discard(capacity);
    if (wants_counts)
      run_cp_t_.resize_discard(capacity);
    constexpr int block =
        gpulsmopt_detail::kTransposeRuns * gpulsmopt_detail::kTransposeQuots;
    const int skip_run =
        sorted_run_ready() ? static_cast<int>(sorted_run_index_) : -1;
    const int run_tiles =
        (active_run_count() + gpulsmopt_detail::kTransposeRuns - 1) /
        gpulsmopt_detail::kTransposeRuns;
    if (!run_meta_ready_) {
      const int q_tiles = static_cast<int>(
          (rows + gpulsmopt_detail::kTransposeQuots - 1) /
          gpulsmopt_detail::kTransposeQuots);
      gpulsmopt_detail::run_transpose_meta_kernel<<<q_tiles * run_tiles, block,
                                                    0, stream>>>(
          run_view_ptr(), active_run_count(), skip_run, run_off_t_.data(),
          run_vp_t_.data(), wants_counts ? run_cp_t_.data() : nullptr);
      CUDA_CHECK(cudaGetLastError());
      run_meta_ready_ = true;
      run_meta_counts_ready_ = wants_counts;
      return;
    }
    constexpr int count_q_tiles = gpulsmopt_detail::kEpochQuotients /
                                  gpulsmopt_detail::kTransposeQuots;
    gpulsmopt_detail::run_transpose_count_kernel<<<count_q_tiles * run_tiles,
                                                   block, 0, stream>>>(
        run_view_ptr(), active_run_count(), skip_run, run_cp_t_.data());
    CUDA_CHECK(cudaGetLastError());
    run_meta_counts_ready_ = true;
  }

  void launch_run_parallel_range(const DeviceRangeOutputBatch &batch,
                                 cudaStream_t stream) {
    const int runs = logical_run_count();
    if (runs <= 32) {
      // Subgroup width: 32/W queries share each warp.
      int width = 1;
      while (width < runs)
        width <<= 1;
      constexpr int block = 256;
      const std::size_t queries_per_block =
          static_cast<std::size_t>(block / width);
      const int grid = static_cast<int>(
          (batch.query_count + queries_per_block - 1u) / queries_per_block);
      gpulsmopt_detail::run_subwarp_range_kernel<<<grid, block, 0, stream>>>(
          batch.lo, batch.hi, batch.out_sums, batch.out_counts,
          batch.query_count, logical_run_views_.data(), runs, width,
          run_off_t_.data(), run_vp_t_.data(),
          batch.out_counts ? run_cp_t_.data() : nullptr);
      CUDA_CHECK(cudaGetLastError());
      return;
    }
    const int block = runs <= 64 ? 64 : 128;
    const int grid = static_cast<int>(batch.query_count);
    gpulsmopt_detail::run_parallel_range_kernel<<<grid, block, 0, stream>>>(
        batch.lo, batch.hi, batch.out_sums, batch.out_counts, batch.query_count,
        logical_run_views_.data(), runs, run_off_t_.data(), run_vp_t_.data(),
        batch.out_counts ? run_cp_t_.data() : nullptr);
    CUDA_CHECK(cudaGetLastError());
  }

  void launch_run_sequential_range(const DeviceRangeOutputBatch &batch,
                                   cudaStream_t stream) {
    constexpr int block = 128;
    const int grid = static_cast<int>((batch.query_count + block - 1) / block);
    gpulsmopt_detail::run_sequential_range_kernel<<<grid, block, 0, stream>>>(
        batch.lo, batch.hi, batch.out_sums, batch.out_counts, batch.query_count,
        logical_run_views_.data(), logical_run_count());
    CUDA_CHECK(cudaGetLastError());
  }

  // Successor bitmap = base live keys + resolved corrections.
  // Persistent bitmap: built once from the BaseRun, then advanced
  // incrementally by assignment runs past the applied watermark.
  void ensure_successor_bitmap(cudaStream_t stream) {
    if (succ_bits_.size() == 0)
      succ_bits_.resize_discard_exact(gpulsmopt_detail::kSuccTotalWords);
    constexpr int block = 256;
    const bool rebuild =
        !succ_built_ || succ_base_generation_ != base_generation_;
    if (rebuild) {
      // Clear L0 and seed live keys from the sorted BaseRun only.
      CUDA_CHECK(cudaMemsetAsync(
          succ_bits_.data(), 0,
          static_cast<std::size_t>(gpulsmopt_detail::succ_level_words(0)) *
              sizeof(std::uint32_t),
          stream));
      if (sorted_run_ready() && sorted_run().count > 0) {
        const RunStorage &base = sorted_run();
        const int grid = static_cast<int>((base.count + block - 1) / block);
        gpulsmopt_detail::succ_seed_base_kernel<<<grid, block, 0, stream>>>(
            base.keys.data(), base.count, succ_bits_.data());
        CUDA_CHECK(cudaGetLastError());
      }
      for (int level = 1; level < 6; ++level) {
        const std::uint32_t upper_words =
            gpulsmopt_detail::succ_level_words(level);
        const int grid = static_cast<int>((upper_words + block - 1) / block);
        gpulsmopt_detail::succ_build_level_kernel<<<grid, block, 0, stream>>>(
            succ_bits_.data() + gpulsmopt_detail::succ_level_off(level - 1),
            succ_bits_.data() + gpulsmopt_detail::succ_level_off(level),
            upper_words);
        CUDA_CHECK(cudaGetLastError());
      }
      succ_applied_sequence_ = 0;
      succ_base_generation_ = base_generation_;
      succ_built_ = true;
    }
    // Apply assignment runs past the watermark, oldest first.
    std::vector<std::size_t> pending;
    for (std::size_t r = 0; r < runs_.size(); ++r)
      if (runs_[r].assignment && runs_[r].sequence > succ_applied_sequence_)
        pending.push_back(r);
    std::sort(pending.begin(), pending.end(),
              [this](std::size_t a, std::size_t b) {
                return runs_[a].sequence < runs_[b].sequence;
              });
    for (const std::size_t r : pending) {
      RunStorage &run = runs_[r];
      if (run.count == 0)
        continue;
      const int grid =
          static_cast<int>((run.count + block - 1) / block);
      if (!run.mixed) {
        const bool insert =
            run.operation == gpulsmopt_detail::RunOperation::Insert;
        gpulsmopt_detail::succ_apply_homogeneous_kernel<<<
            grid, block, 0, stream>>>(
            run.keys.data(), run.count, insert, succ_bits_.data());
        CUDA_CHECK(cudaGetLastError());
        continue;
      }
      gpulsmopt_detail::succ_apply_mixed_kernel<<<
          grid, block, 0, stream>>>(
          run.keys.data(), run.op_words.data(), run.count,
          succ_bits_.data());
      CUDA_CHECK(cudaGetLastError());
      for (int level = 1; level < 6; ++level) {
        gpulsmopt_detail::succ_update_level_kernel<<<
            grid, block, 0, stream>>>(
            run.keys.data(), run.count,
            succ_bits_.data() +
                gpulsmopt_detail::succ_level_off(level - 1),
            succ_bits_.data() +
                gpulsmopt_detail::succ_level_off(level),
            5 * level);
        CUDA_CHECK(cudaGetLastError());
      }
    }
    succ_applied_sequence_ = run_sequence_;
  }

  void invalidate_resolved() {
    resolved_ready_ = false;
    resolved_value_prefix_ready_ = false;
    resolved_count_prefix_ready_ = false;
  }

  // Gather by quotient and sort only unseen low bits.
  void normalize_runs(const std::vector<std::size_t> &idx,
                      std::size_t total, cudaStream_t stream) {
    if (total > static_cast<std::size_t>(
                    std::numeric_limits<int>::max()))
      throw std::runtime_error("resolved cache exceeds CUB limits");
    constexpr int block = 256;
    normalize_views_.resize_discard(
        gpulsmopt_detail::kRunCapacity);
    for (std::size_t slot = 0; slot < idx.size(); ++slot) {
      RunStorage &run = runs_[idx[slot]];
      const bool insert =
          run.operation == gpulsmopt_detail::RunOperation::Insert;
      host_state_->views[slot] = {
          run.keys.data(),
          insert ? run.values.data() : nullptr,
          run.quotient_off.data(),
          run.mixed ? run.op_words.data() : nullptr,
          static_cast<std::uint8_t>(insert ? 1u : 0u),
          static_cast<std::uint8_t>(run.mixed ? 1u : 0u)};
    }
    CUDA_CHECK(cudaMemcpyAsync(
        normalize_views_.data(), host_state_->views,
        idx.size() * sizeof(gpulsmopt_detail::AssignmentRunView),
        cudaMemcpyHostToDevice, stream));

    resolve_keys_.resize_discard(total);
    resolve_payload_.resize_discard(total);
    resolve_alt_keys_.resize_discard(total);
    resolve_alt_payload_.resize_discard(total);
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
        make_sorted_view(), norm_keys_.data(),
        norm_pay_.data());
    CUDA_CHECK(cudaGetLastError());
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
    if (resolved_base_generation_ != base_generation_) {
      resolved_.count = 0;
      resolved_through_sequence_ = 0;
      resolved_base_generation_ = base_generation_;
      resolved_ready_ = false;
    }
    std::vector<std::size_t> nidx;
    std::size_t total = 0;
    for (std::size_t r = 0; r < runs_.size(); ++r) {
      if (runs_[r].assignment &&
          runs_[r].sequence > resolved_through_sequence_) {
        nidx.push_back(r);
        total += runs_[r].count;
      }
    }
    std::sort(nidx.begin(), nidx.end(),
              [this](std::size_t a, std::size_t b) {
                return runs_[a].sequence < runs_[b].sequence;
              });
    if (nidx.empty() || total == 0) {
      resolved_through_sequence_ = run_sequence_;
      if (!resolved_ready_) {
        build_assignment_offsets(resolved_, resolved_.count, stream);
        resolved_value_prefix_ready_ = false;
        resolved_count_prefix_ready_ = false;
        resolved_ready_ = true;
      }
      return;
    }

    normalize_runs(nidx, total, stream);
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

  // Fold the base snapshot and every assignment run into one
  // fresh sorted base via last-wins; drops resolved tombstones.
  void fold_into_base(cudaStream_t stream) {
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
      // Base run packs as all-insert; assignment leaves use their op.
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
    prebind_run_views(stream);
    run_sequence_ = 0;
    chrono_views_.clear();
    invalidate_resolved();
    succ_built_ = false;
    ++base_generation_;
    ++structure_generation_;
  }

  // Republish the chronological descriptor array after the run set
  // changes (compaction). Assignment runs, oldest sequence first.
  void rebuild_chrono_views(cudaStream_t stream) {
    std::vector<std::size_t> idx;
    for (std::size_t r = 0; r < runs_.size(); ++r)
      if (runs_[r].assignment)
        idx.push_back(r);
    std::sort(idx.begin(), idx.end(), [this](std::size_t a, std::size_t b) {
      return runs_[a].sequence < runs_[b].sequence;
    });
    chrono_views_.clear();
    for (const std::size_t r : idx)
      publish_assignment_view(runs_[r], stream);
  }

  // Temporally-contiguous last-write-wins compaction. Merges the
  // oldest group of assignment runs into ONE mixed run, retaining
  // tombstones (unlike fold_into_base). Falls back to fold when a
  // group cannot be formed.
  void compact_contiguous(cudaStream_t stream) {
    std::vector<std::size_t> idx;
    for (std::size_t r = 0; r < runs_.size(); ++r)
      if (runs_[r].assignment)
        idx.push_back(r);
    if (idx.size() < 2) {
      fold_into_base(stream);
      return;
    }
    std::sort(idx.begin(), idx.end(), [this](std::size_t a, std::size_t b) {
      return runs_[a].sequence < runs_[b].sequence;
    });
    const std::size_t group_size =
        std::min<std::size_t>(idx.size(), gpulsmopt_detail::kCompactGroup);
    std::vector<std::size_t> group(idx.begin(), idx.begin() + group_size);
    std::uint64_t group_seq = 0;
    std::size_t total = 0;
    for (const std::size_t r : group) {
      group_seq = std::max(group_seq, runs_[r].sequence);
      total += runs_[r].count;
    }
    if (total == 0) {
      fold_into_base(stream);
      return;
    }
    if (total > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("temporal compaction exceeds CUB limits");

    normalize_views_.resize_discard(group_size);
    for (std::size_t slot = 0; slot < group_size; ++slot) {
      RunStorage &run = runs_[group[slot]];
      const bool insert =
          run.operation == gpulsmopt_detail::RunOperation::Insert;
      host_state_->views[slot] = {
          run.keys.data(), insert ? run.values.data() : nullptr,
          run.quotient_off.data(), run.mixed ? run.op_words.data() : nullptr,
          static_cast<std::uint8_t>(insert ? 1u : 0u),
          static_cast<std::uint8_t>(run.mixed ? 1u : 0u)};
    }
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
    if (host_compaction_offsets_.back() != total)
      throw std::runtime_error("temporal compaction count mismatch");

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
    merged.mixed = true;
    merged.operation = gpulsmopt_detail::RunOperation::Insert;
    merged.sequence = group_seq;
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
    std::sort(group.begin(), group.end());
    for (auto it = group.rbegin(); it != group.rend(); ++it) {
      if (sorted_run_ready() && sorted_run_index_ > *it)
        --sorted_run_index_;
      run_pool_.push_back(std::move(runs_[*it]));
      runs_.erase(runs_.begin() + static_cast<std::ptrdiff_t>(*it));
    }
    runs_.push_back(std::move(compacted));
    rebuild_chrono_views(stream);
    invalidate_resolved();
    // Base is unchanged: the successor watermark handles the merge,
    // so the bitmap is NOT invalidated here.
    ++structure_generation_;
  }

  void launch_range_projection(const DeviceRangeOutputBatch &batch,
                               cudaStream_t stream) {
    const int block = 128;
    const int grid = static_cast<int>((batch.query_count + block - 1) / block);
    gpulsmopt_detail::range_projection_query_kernel<<<grid, block, 0, stream>>>(
        batch.lo, batch.hi, batch.out_sums, batch.query_count,
        make_sorted_view(), make_sorted_range_view(),
        sorted_value_prefix_.data(), make_range_delta_view());
    CUDA_CHECK(cudaGetLastError());
  }

  std::size_t max_elements_ = 0;
  std::size_t batch_capacity_ = 0;
  std::size_t live_count_ = 0;
  std::size_t sorted_run_index_ = std::numeric_limits<std::size_t>::max();
  mutable std::shared_mutex snapshot_mutex_;

  // Temporal assignment-run state.
  std::uint64_t run_sequence_ = 0;
  RunStorage resolved_;
  bool resolved_ready_ = false;
  // Incremental resolved-cache state + linear merge scratch.
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
  bool succ_built_ = false;
  std::uint64_t succ_applied_sequence_ = 0;
  std::uint64_t succ_base_generation_ = ~std::uint64_t{0};
  std::uint64_t base_generation_ = 0;
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::AssignmentRunView>
      assignment_views_;
  // Host mirror of the chronological descriptor array (oldest->newest).
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
  double prof_append_ms_ = 0.0;
  double prof_delta_sort_ms_ = 0.0;
  double prof_delta_ingest_ms_ = 0.0;
  void reset_insert_prof_() {
    prof_append_ms_ = prof_delta_sort_ms_ = prof_delta_ingest_ms_ = 0.0;
  }
#endif

  std::vector<RunStorage> runs_;
  std::vector<RunStorage> run_pool_;
  thrust::device_vector<gpulsmopt_detail::RunView> run_views_;
  std::vector<gpulsmopt_detail::RunView> bound_run_views_;
  std::vector<std::uint8_t> bound_run_view_valid_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> range_delta_counts_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> range_delta_sums_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> range_delta_offsets_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> range_delta_value_prefix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> range_delta_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> range_delta_values_;
  bool range_delta_projection_ready_ = false;

  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> direct_sort_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> direct_sort_values_;
  thrust::device_vector<std::uint32_t> sort_key_output_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> sort_temp_storage_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> sorted_value_prefix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> sorted_count_prefix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> sorted_range_cdf_;
  std::uint32_t sorted_range_min_key_ = 0u;
  std::uint64_t sorted_range_span_ = 0u;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> base_rank23_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> owner_primary_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> owner_spill_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> owner_spill_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> owner_spill_count_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> owner_quotient_live_;
  std::size_t owner_spill_slots_ = 0;
  // Boundary-scatter metadata scratch.
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> meta_pending_list_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> meta_pending_count_;
  // Per-commit transition size classes (fix 5 consumers).
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> owner_class_list_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> owner_class_count_;
  bool owner_ready_ = false;
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

  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::LogicalRunView>
      logical_run_views_;
  std::vector<gpulsmopt_detail::LogicalRunView> bound_logical_run_views_;
  bool logical_run_views_ready_ = false;

  // Transposed [quotient][run] read metadata, built lazily.
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> run_off_t_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> run_vp_t_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> run_cp_t_;
  bool run_meta_ready_ = false;
  bool run_meta_counts_ready_ = false;

  // Lazy hierarchical live-key bitmap for successor().
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> succ_bits_;
  std::uint64_t structure_generation_ = 0;
  std::uint64_t succ_generation_ = 0;
  std::size_t succ_synced_runs_ = 0;

};
