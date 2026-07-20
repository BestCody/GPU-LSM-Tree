#pragma once
#include "gpu_dictionary_adapter.cuh"

#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
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
constexpr int kRunCapacity = 128;
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
constexpr int kCompactTargetRuns = 64;
constexpr int kCompactMergeRuns = kRunCapacity - kCompactTargetRuns + 1;
constexpr int kGpuResidentWarps = 9024;
static_assert(kCompactMergeRuns == 65);
static_assert(kRunCapacity - kCompactMergeRuns + 1 == kCompactTargetRuns);
// Shards up to this size sort in shared memory.
constexpr std::uint32_t kCompactSharedCap = 2048;
constexpr int kSortedRunRadixBits = 20;
constexpr int kSortedRunRadixShift = 32 - kSortedRunRadixBits;
constexpr int kSortedRunMicroTarget = 4;
constexpr std::size_t kSortedRunMinRecords = 1u << 22;
constexpr int kRangeProjectionBits = 20;
constexpr int kRangeProjectionBins = 1 << kRangeProjectionBits;
constexpr int kRangeProjectionShift = 32 - kRangeProjectionBits;
constexpr std::size_t kRangeProjectionMinQueries = 1 << 18;
constexpr std::uint64_t kRangeCdfMaxRatio = GPULSMOPT_RANGE_CDF_MAX_RATIO;
constexpr std::size_t kSortedRunRadixSize = std::size_t{1}
                                            << kSortedRunRadixBits;
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
  const std::uint32_t *radix;
  const std::uint32_t *micro_base;
  const std::uint16_t *micro_offsets;
  const std::uint8_t *micro_bits;
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
sorted_refined_bounds(const SortedRunView &sorted, std::uint32_t key,
                      std::uint32_t bin, std::size_t parent_begin,
                      std::size_t parent_end, std::size_t *begin,
                      std::size_t *end) {
  *begin = parent_begin;
  *end = parent_end;
  if (!sorted.micro_bits)
    return;
  const std::uint32_t bits = sorted.micro_bits[bin];
  if (bits == 0u)
    return;
  const std::uint32_t slots = 1u << bits;
  const std::uint32_t low = key & ((1u << kSortedRunRadixShift) - 1u);
  const std::uint32_t slot = low >> (kSortedRunRadixShift - bits);
  const std::uint32_t pool = sorted.micro_base[bin];
  const std::size_t count = parent_end - parent_begin;
  const std::size_t local_begin = sorted.micro_offsets[pool + slot];
  const std::size_t local_end =
      slot + 1u < slots ? sorted.micro_offsets[pool + slot + 1u] : count;
  *begin = parent_begin + local_begin;
  *end = parent_begin + local_end;
}

__device__ inline void sorted_search_bounds(const SortedRunView &sorted,
                                            std::uint32_t key,
                                            std::size_t *begin,
                                            std::size_t *end) {
  const std::uint32_t bin = key >> kSortedRunRadixShift;
  const std::size_t parent_begin = sorted.radix[bin];
  const std::size_t parent_end = sorted.radix[bin + 1];
  sorted_refined_bounds(sorted, key, bin, parent_begin, parent_end, begin, end);
}

__device__ inline void sorted_range_ranks(const SortedRunView &sorted,
                                          std::uint32_t lo, std::uint32_t hi,
                                          std::size_t *lower,
                                          std::size_t *upper) {
  const std::uint32_t lo_bin = lo >> kSortedRunRadixShift;
  const std::uint32_t hi_bin = hi >> kSortedRunRadixShift;
  std::size_t lo_begin = 0, lo_end = 0;
  std::size_t hi_begin = 0, hi_end = 0;
  if (lo_bin == hi_bin) {
    const std::size_t parent_begin = sorted.radix[lo_bin];
    const std::size_t parent_end = sorted.radix[lo_bin + 1];
    sorted_refined_bounds(sorted, lo, lo_bin, parent_begin, parent_end,
                          &lo_begin, &lo_end);
    sorted_refined_bounds(sorted, hi, hi_bin, parent_begin, parent_end,
                          &hi_begin, &hi_end);
  } else {
    sorted_search_bounds(sorted, lo, &lo_begin, &lo_end);
    sorted_search_bounds(sorted, hi, &hi_begin, &hi_end);
  }
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

__global__ void sorted_build_radix_kernel(const std::uint32_t *keys,
                                          std::size_t count,
                                          std::uint32_t *radix) {
  const std::size_t bin = blockIdx.x * blockDim.x + threadIdx.x;
  if (bin > kSortedRunRadixSize)
    return;
  if (bin == kSortedRunRadixSize) {
    radix[bin] = static_cast<std::uint32_t>(count);
    return;
  }
  const std::uint32_t key =
      static_cast<std::uint32_t>(bin << kSortedRunRadixShift);
  radix[bin] = static_cast<std::uint32_t>(lower_bound_u32(keys, count, key));
}

__global__ void sorted_micro_plan_kernel(const std::uint32_t *radix,
                                         std::uint8_t *micro_bits,
                                         std::uint32_t *micro_base) {
  const std::uint32_t bin = blockIdx.x * blockDim.x + threadIdx.x;
  if (bin >= kSortedRunRadixSize)
    return;
  const std::uint32_t count = radix[bin + 1] - radix[bin];
  if (count <= kSortedRunMicroTarget) {
    micro_bits[bin] = 0u;
    micro_base[bin] = 0u;
    return;
  }
  const std::uint32_t needed =
      (count + kSortedRunMicroTarget - 1u) / kSortedRunMicroTarget;
  std::uint32_t slots = 1u;
  std::uint8_t bits = 0u;
  while (slots < needed) {
    slots <<= 1;
    ++bits;
  }
  micro_bits[bin] = bits;
  micro_base[bin] = slots;
}

__global__ void sorted_micro_fill_kernel(const std::uint32_t *keys,
                                         const std::uint32_t *radix,
                                         const std::uint8_t *micro_bits,
                                         const std::uint32_t *micro_base,
                                         std::uint16_t *micro_offsets) {
  const std::uint32_t bin = blockIdx.x * blockDim.x + threadIdx.x;
  if (bin >= kSortedRunRadixSize)
    return;
  const std::uint32_t bits = micro_bits[bin];
  if (bits == 0u)
    return;
  const std::uint32_t begin = radix[bin];
  const std::uint32_t end = radix[bin + 1];
  const std::uint32_t slots = 1u << bits;
  const std::uint32_t pool = micro_base[bin];
  std::uint32_t next_slot = 0u;
  for (std::uint32_t position = begin; position < end; ++position) {
    const std::uint32_t low =
        keys[position] & ((1u << kSortedRunRadixShift) - 1u);
    const std::uint32_t slot = low >> (kSortedRunRadixShift - bits);
    while (next_slot <= slot)
      micro_offsets[pool + next_slot++] =
          static_cast<std::uint16_t>(position - begin);
  }
  while (next_slot < slots)
    micro_offsets[pool + next_slot++] = static_cast<std::uint16_t>(end - begin);
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

__global__ void c0_log_append_kernel(const std::uint32_t *keys,
                                     const std::uint32_t *values,
                                     std::uint8_t op, std::uint32_t old_count,
                                     std::size_t n, std::uint32_t *log_keys,
                                     std::uint32_t *log_values,
                                     std::uint8_t *log_ops) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint32_t idx = old_count + static_cast<std::uint32_t>(i);
  log_keys[idx] = keys[i];
  log_values[idx] = values ? values[i] : keys[i];
  if (log_ops)
    log_ops[idx] = op;
}

struct TakeLatestPayload {
  __host__ __device__ std::uint64_t operator()(std::uint64_t older,
                                               std::uint64_t newer) const {
    const std::uint32_t older_sequence = static_cast<std::uint32_t>(older) >> 1;
    const std::uint32_t newer_sequence = static_cast<std::uint32_t>(newer) >> 1;
    return newer_sequence >= older_sequence ? newer : older;
  }
};

__global__ void pack_latest_payload_kernel(const std::uint32_t *values,
                                           const std::uint8_t *ops,
                                           std::uint8_t constant_op,
                                           std::size_t count,
                                           std::uint64_t *payload) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::uint8_t op = ops ? ops[i] : constant_op;
  const std::uint32_t sequence = static_cast<std::uint32_t>(i);
  payload[i] = (static_cast<std::uint64_t>(values[i]) << 32) |
               (static_cast<std::uint64_t>(sequence) << 1) | op;
}

__global__ void unpack_latest_payload_kernel(const std::uint64_t *payload,
                                             std::size_t count,
                                             std::uint32_t *values,
                                             std::uint8_t *ops) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  values[i] = static_cast<std::uint32_t>(payload[i] >> 32);
  ops[i] = static_cast<std::uint8_t>(payload[i] & 1u);
}
struct SumDeltaPayload {
  __host__ __device__ std::uint64_t operator()(std::uint64_t left,
                                               std::uint64_t right) const {
    const std::uint32_t value = static_cast<std::uint32_t>(left >> 32) +
                                static_cast<std::uint32_t>(right >> 32);
    const std::int32_t count =
        static_cast<std::int32_t>(static_cast<std::uint32_t>(left)) +
        static_cast<std::int32_t>(static_cast<std::uint32_t>(right));
    return (static_cast<std::uint64_t>(value) << 32) |
           static_cast<std::uint32_t>(count);
  }
};

__global__ void pack_delta_payload_kernel(const std::uint32_t *values,
                                          const std::int8_t *counts,
                                          std::size_t count,
                                          std::uint64_t *payload) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::int32_t count_delta = static_cast<std::int32_t>(counts[i]);
  payload[i] = (static_cast<std::uint64_t>(values[i]) << 32) |
               static_cast<std::uint32_t>(count_delta);
}

__global__ void unpack_delta_payload_kernel(const std::uint64_t *payload,
                                            std::size_t count,
                                            std::uint32_t *values,
                                            std::int8_t *counts) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  values[i] = static_cast<std::uint32_t>(payload[i] >> 32);
  counts[i] = static_cast<std::int8_t>(
      static_cast<std::int32_t>(static_cast<std::uint32_t>(payload[i])));
}

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
__global__ void succ_rebuild_from_owner_kernel(OwnerView owner,
                                               std::uint32_t *bits) {
  const std::uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= kEpochQuotients)
    return;
  const std::uint64_t *page =
      owner.primary + static_cast<std::size_t>(warp) * kOwnerSlots;
  for (int slot = lane; slot < kOwnerSlots; slot += 32) {
    const std::uint64_t packed = page[slot];
    if (owner_state(packed) != kOwnerLive)
      continue;
    const std::uint32_t key = (warp << kEpochQuotientBits) | owner_low(packed);
    atomicOr(bits + (key >> 5), 1u << (key & 31u));
  }
}

// Spilled keys are always live: scan the table flat.
__global__ void succ_rebuild_from_spill_kernel(OwnerView owner,
                                               std::uint32_t *bits) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i > owner.spill_mask)
    return;
  const std::uint64_t token = owner.spill_keys[i];
  if (token < 2u)
    return;
  const std::uint32_t key = static_cast<std::uint32_t>(token - 2u);
  atomicOr(bits + (key >> 5), 1u << (key & 31u));
}

// Apply one delta run; owner state decides liveness.
// Count deltas encode liveness transitions directly.
// Relies on runs never holding both +1 and -1 for a key.
__global__ void succ_apply_epoch_kernel(const std::uint32_t *keys,
                                        const std::int8_t *count_delta,
                                        std::size_t count,
                                        std::uint32_t *bits) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::int8_t delta = count_delta[i];
  if (delta == 0)
    return;
  const std::uint32_t key = keys[i];
  const std::uint32_t mask = 1u << (key & 31u);
  if (delta > 0)
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

// Combined and padded record counts per quotient shard.
__global__ void compact_plan_kernel(const RunView *views, int group_count,
                                    std::uint32_t *combined) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kEpochQuotients)
    return;
  if (q == kEpochQuotients) {
    combined[q] = 0u;
    return;
  }
  std::uint32_t total = 0u;
  for (int e = 0; e < group_count; ++e)
    total += views[e].quotient_off[q + 1u] - views[e].quotient_off[q];
  combined[q] = total;
}

__host__ __device__ inline std::uint64_t
compact_pack_payload(std::uint32_t value, std::int8_t count) {
  return (static_cast<std::uint64_t>(value) << 32) |
         static_cast<std::uint32_t>(static_cast<std::int32_t>(count));
}

// Stage heavy shards with packed payloads for the sorter.
__global__ void compact_gather_heavy_kernel(
    const RunView *views, int group_count, const std::uint32_t *heavy_list,
    const std::uint32_t *heavy_count, const std::uint32_t *segoff,
    std::uint32_t *skeys, std::uint64_t *spayload) {
  const std::uint32_t items = heavy_count[0];
  for (std::uint32_t item = blockIdx.x; item < items; item += gridDim.x) {
    const std::uint32_t q = heavy_list[item];
    std::uint32_t cursor = segoff[q];
    for (int e = 0; e < group_count; ++e) {
      const std::uint32_t begin = views[e].quotient_off[q];
      const std::uint32_t n = views[e].quotient_off[q + 1u] - begin;
      for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        skeys[cursor + i] = views[e].keys[begin + i];
        spayload[cursor + i] = compact_pack_payload(
            views[e].values[begin + i], views[e].count_delta[begin + i]);
      }
      cursor += n;
    }
  }
}

// Segment bounds of heavy shards for the segmented sort.
__global__ void compact_heavy_bounds_kernel(const std::uint32_t *heavy_list,
                                            const std::uint32_t *heavy_count,
                                            const std::uint32_t *segoff,
                                            const std::uint32_t *combined,
                                            std::uint32_t *begins,
                                            std::uint32_t *ends) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= heavy_count[0])
    return;
  const std::uint32_t q = heavy_list[i];
  begins[i] = segoff[q];
  ends[i] = segoff[q] + combined[q];
}

// Return sorted heavy segments into primary staging.
__global__ void compact_heavy_copyback_kernel(
    const std::uint32_t *heavy_list, const std::uint32_t *heavy_count,
    const std::uint32_t *segoff, const std::uint32_t *combined,
    const std::uint32_t *alt_keys, const std::uint64_t *alt_payload,
    std::uint32_t *skeys, std::uint64_t *spayload) {
  const std::uint32_t items = heavy_count[0];
  for (std::uint32_t item = blockIdx.x; item < items; item += gridDim.x) {
    const std::uint32_t q = heavy_list[item];
    const std::uint32_t base = segoff[q];
    const std::uint32_t n = combined[q];
    for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
      skeys[base + i] = alt_keys[base + i];
      spayload[base + i] = alt_payload[base + i];
    }
  }
}

// Fused gather plus 16-bit block radix sort per shard.
constexpr int kCompactThreads = 256;
constexpr int kCompactItems =
    static_cast<int>(kCompactSharedCap) / kCompactThreads;
__global__ void compact_sort_light_kernel(const RunView *views,
                                          int group_count,
                                          const std::uint32_t *combined,
                                          const std::uint32_t *segoff,
                                          std::uint32_t *skeys,
                                          std::uint64_t *spayload) {
  using BlockSort = cub::BlockRadixSort<std::uint32_t, kCompactThreads,
                                        kCompactItems, std::uint64_t>;
  __shared__ union LightShared {
    struct {
      std::uint32_t keys[kCompactSharedCap];
      std::uint64_t payload[kCompactSharedCap];
    } stage;
    typename BlockSort::TempStorage sort;
  } shared;
  const std::uint32_t q = blockIdx.x;
  const std::uint32_t n = combined[q];
  if (n == 0u || n > kCompactSharedCap)
    return;
  std::uint32_t cursor = 0u;
  for (int e = 0; e < group_count; ++e) {
    const std::uint32_t begin = views[e].quotient_off[q];
    const std::uint32_t len = views[e].quotient_off[q + 1u] - begin;
    for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
      shared.stage.keys[cursor + i] = views[e].keys[begin + i];
      shared.stage.payload[cursor + i] = compact_pack_payload(
          views[e].values[begin + i], views[e].count_delta[begin + i]);
    }
    cursor += len;
  }
  __syncthreads();
  // Bit 16 pushes padding past every real low key.
  std::uint32_t sort_keys[kCompactItems];
  std::uint64_t sort_payload[kCompactItems];
#pragma unroll
  for (int r = 0; r < kCompactItems; ++r) {
    const std::uint32_t slot =
        static_cast<std::uint32_t>(threadIdx.x) * kCompactItems + r;
    sort_keys[r] = slot < n ? shared.stage.keys[slot] & 0xffffu : 0x10000u;
    sort_payload[r] = slot < n ? shared.stage.payload[slot] : 0u;
  }
  __syncthreads();
  BlockSort(shared.sort).Sort(sort_keys, sort_payload, 0, 17);
  const std::uint32_t base = segoff[q];
  const std::uint32_t high = q << kEpochQuotientBits;
#pragma unroll
  for (int r = 0; r < kCompactItems; ++r) {
    const std::uint32_t slot =
        static_cast<std::uint32_t>(threadIdx.x) * kCompactItems + r;
    if (slot < n) {
      skeys[base + slot] = high | sort_keys[r];
      spayload[base + slot] = sort_payload[r];
    }
  }
}

__global__ void compact_classify_heavy_kernel(const std::uint32_t *combined,
                                              std::uint32_t *heavy_list,
                                              std::uint32_t *heavy_count) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kEpochQuotients)
    return;
  if (combined[q] > kCompactSharedCap)
    heavy_list[atomicAdd(heavy_count, 1u)] = q;
}

// Zero-net filter over the reduced duplicate groups.
__global__ void compact_filter_flags_kernel(const std::uint64_t *payload,
                                            const int *num_runs,
                                            std::size_t total,
                                            std::uint32_t *flags) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total)
    return;
  const std::size_t runs = static_cast<std::size_t>(num_runs[0]);
  flags[i] = i < runs && payload[i] != 0u ? 1u : 0u;
}

// Unpack surviving records into the merged delta run.
__global__ void
compact_scatter_kernel(const std::uint32_t *unique_keys,
                       const std::uint64_t *unique_payload,
                       const std::uint32_t *flags, const std::uint32_t *pos,
                       std::size_t total, std::uint32_t *out_keys,
                       std::uint32_t *out_values, std::int8_t *out_counts) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total || flags[i] == 0u)
    return;
  const std::uint64_t payload = unique_payload[i];
  const std::uint32_t p = pos[i];
  out_keys[p] = unique_keys[i];
  out_values[p] = static_cast<std::uint32_t>(payload >> 32);
  out_counts[p] = static_cast<std::int8_t>(
      static_cast<std::int32_t>(static_cast<std::uint32_t>(payload)));
}

__global__ void compact_unique_count_kernel(const std::uint32_t *flags,
                                            const std::uint32_t *pos,
                                            std::size_t capacity,
                                            std::uint32_t *counts,
                                            std::uint32_t group) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  if (capacity == 0) {
    counts[group] = 0u;
    return;
  }
  counts[group] = pos[capacity - 1u] + flags[capacity - 1u];
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
  }

  void clear(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    last_update_stream_ = stream;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    clear_run_state();
    clear_owner_state(stream);
    live_count_ = 0;
    clear_c0_log(stream);
  }

  void insert(const DeviceKeyValueBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    {
      std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
      last_update_stream_ = stream;
      if (!owner_ready_)
        initialize_owner_storage(stream);
#ifdef GPULSMOPT_PROFILE_INSERT
      reset_insert_prof_();
      const auto prof_t0 = std::chrono::high_resolution_clock::now();
#endif
      insert_records(batch.keys, batch.values,
                     static_cast<std::uint8_t>(gpulsmopt_detail::kInsert),
                     batch.count, false, stream);
      maybe_flush_and_merge(stream);
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
      last_update_stream_ = stream;
      if (!owner_ready_)
        initialize_owner_storage(stream);
#ifdef GPULSMOPT_PROFILE_INSERT
      reset_insert_prof_();
      const auto prof_t0 = std::chrono::high_resolution_clock::now();
#endif
      insert_records(batch.keys, batch.keys,
                     static_cast<std::uint8_t>(gpulsmopt_detail::kTombstone),
                     batch.count, batch.sorted, stream);
      maybe_flush_and_merge(stream);
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
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    std::unique_lock<std::shared_mutex> exclusive(snapshot_mutex_,
                                                  std::defer_lock);
    if (c0_log_count_ != 0) {
      guard.unlock();
      exclusive.lock();
      if (c0_log_count_ != 0)
        merge_down(stream);
    }
    if (!owner_ready_) {
      CUDA_CHECK(cudaMemsetAsync(batch.out_values, 0xff,
                                 batch.count * sizeof(std::uint32_t), stream));
      if (batch.out_found) {
        CUDA_CHECK(cudaMemsetAsync(batch.out_found, 0,
                                   batch.count * sizeof(std::uint8_t), stream));
      }
      return;
    }
    const int block = 256;
    const int grid = static_cast<int>((batch.count + block - 1) / block);
    gpulsmopt_detail::owner_lookup_kernel<<<grid, block, 0, stream>>>(
        make_owner_view(), batch.queries, batch.count, batch.out_values,
        batch.out_found);
    CUDA_CHECK(cudaGetLastError());
  }

  void successor(const DeviceSuccessorBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    if (c0_log_count_ != 0)
      merge_down(stream);
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
    if (c0_log_count_ != 0)
      merge_down(stream);
    if (runs_.empty()) {
      CUDA_CHECK(cudaMemsetAsync(batch.out_sums, 0,
                                 batch.query_count * sizeof(std::uint32_t),
                                 stream));
      if (batch.out_counts) {
        CUDA_CHECK(cudaMemsetAsync(batch.out_counts, 0,
                                   batch.query_count * sizeof(std::uint32_t),
                                   stream));
      }
      return;
    }
    ensure_sorted_run_cache(stream);
    const bool use_projection = should_use_range_delta_projection(
        batch.query_count, batch.out_counts != nullptr);
    if (!use_projection) {
      ensure_run_heavy_sorted(stream);
      ensure_run_value_prefixes(stream);
    }
    if (batch.out_counts)
      ensure_run_count_prefixes(stream);
    if (use_projection) {
      if (!range_delta_projection_ready_)
        build_range_delta_projection(stream);
      launch_range_projection(batch, stream);
      return;
    }
    ensure_logical_run_views(stream);
    if (should_use_run_parallel(batch.query_count)) {
      ensure_run_transposed_metadata(batch.out_counts != nullptr, stream);
      launch_run_parallel_range(batch, stream);
    } else {
      launch_run_sequential_range(batch, stream);
    }
  }

  void consolidate(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    merge_down(stream);
    consolidate_all_runs(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void bulk_build(const std::uint32_t *keys, const std::uint32_t *values,
                  std::size_t n, cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    last_update_stream_ = stream;
    clear_run_state();
    clear_c0_log(stream);
    live_count_ = 0;
    if (n == 0) {
      clear_owner_state(stream);
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
    run.count_delta.resize_discard(run.count);
    CUDA_CHECK(cudaMemsetAsync(run.count_delta.data(), 1,
                               run.count * sizeof(std::int8_t), stream));
    commit_run_metadata(run, stream, nullptr,
                        static_cast<std::uint8_t>(gpulsmopt_detail::kInsert),
                        false);
    build_sorted_run_cache(0u, stream);
    build_owner_index(stream);
    live_count_ = run.count;
    prepare_for_insert(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  std::size_t live_count() const {
    auto *self = const_cast<GPULSMOpt *>(this);
    std::unique_lock<std::shared_mutex> guard(self->snapshot_mutex_);
    const cudaStream_t stream = self->last_update_stream_;
    if (self->c0_log_count_ != 0)
      self->merge_down(stream);
    if (!self->owner_ready_)
      return 0;
    auto policy = thrust::cuda::par.on(stream);
    const std::size_t live = thrust::reduce(
        policy, self->owner_quotient_live_.data(),
        self->owner_quotient_live_.data() + gpulsmopt_detail::kEpochQuotients,
        std::size_t{0}, thrust::plus<std::size_t>());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    self->live_count_ = live;
    return self->live_count_;
  }

  std::size_t gpu_resident_bytes() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    std::size_t total = device_bytes_all(
        run_views_, range_delta_counts_, range_delta_sums_,
        range_delta_offsets_, range_delta_value_prefix_, range_delta_keys_,
        range_delta_values_, c0_log_keys_, c0_log_values_, c0_log_ops_,
        direct_sort_keys_, direct_sort_values_, sort_key_output_,
        sort_payload_input_, sort_payload_output_, sort_temp_storage_,
        run_merge_keys_, sorted_value_prefix_, sorted_count_prefix_,
        sorted_radix_, sorted_range_cdf_, sorted_micro_base_,
        sorted_micro_offsets_, sorted_micro_bits_, owner_primary_,
        owner_spill_keys_, owner_spill_values_, owner_spill_count_,
        owner_quotient_live_, merge_ops_, logical_run_views_,
        run_off_t_, run_vp_t_, run_cp_t_, succ_bits_, compact_views_,
        compact_combined_, compact_segoff_, compact_heavy_list_,
        compact_heavy_count_, compact_heavy_begin_, compact_heavy_end_,
        compact_keys_, compact_payload_, compact_alt_keys_,
        compact_alt_payload_, compact_num_runs_, compact_flags_, compact_pos_,
        compact_group_unique_, compact_output_keys_, compact_output_values_,
        compact_output_counts_);
    for (const auto &epoch : runs_)
      total += device_bytes_all(
          epoch.keys, epoch.values, epoch.count_delta, epoch.quotient_count_sum,
          epoch.quotient_off, epoch.subgroup_masks, epoch.quotient_live,
          epoch.quotient_value_sum, epoch.quotient_count_prefix,
          epoch.quotient_value_prefix, epoch.subgroup_value_prefix,
          epoch.heavy_list, epoch.heavy_count);
    for (const auto &epoch : run_pool_)
      total += device_bytes_all(
          epoch.keys, epoch.values, epoch.count_delta, epoch.quotient_count_sum,
          epoch.quotient_off, epoch.subgroup_masks, epoch.quotient_live,
          epoch.quotient_value_sum, epoch.quotient_count_prefix,
          epoch.quotient_value_prefix, epoch.subgroup_value_prefix,
          epoch.heavy_list, epoch.heavy_count);
    return total;
  }

private:
  struct RunStorage {
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> keys;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> values;
    gpulsmopt_detail::RawDeviceBuffer<std::int8_t> count_delta;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_off;
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
    bool heavy_sorted = true;
    bool value_sums_ready = true;
    bool value_prefix_ready = true;
    bool subgroup_value_prefix_ready = true;
    bool count_prefix_ready = true;
    bool fully_sorted = false;
    bool unit_counts = false;
    bool unique_keys = false;
  };

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
            sorted_radix_.data(),
            sorted_micro_ready_ ? sorted_micro_base_.data() : nullptr,
            sorted_micro_ready_ ? sorted_micro_offsets_.data() : nullptr,
            sorted_micro_ready_ ? sorted_micro_bits_.data() : nullptr,
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
    sorted_radix_.resize_discard(0);
    sorted_micro_base_.resize_discard(0);
    sorted_micro_offsets_.resize_discard(0);
    sorted_micro_bits_.resize_discard(0);
    sorted_range_cdf_.release();
    sorted_range_min_key_ = 0u;
    sorted_range_span_ = 0u;
    sorted_range_cdf_ready_ = false;
    sorted_micro_ready_ = false;
    run_meta_ready_ = false;
    run_meta_counts_ready_ = false;
    logical_run_views_ready_ = false;
  }

  void build_sorted_metadata(cudaStream_t stream) {
    RunStorage &run = runs_[sorted_run_index_];
    const std::size_t count = run.count;
    sorted_radix_.resize_discard(gpulsmopt_detail::kSortedRunRadixSize + 1);
    constexpr int block = 256;
    const int radix_grid = static_cast<int>(
        (gpulsmopt_detail::kSortedRunRadixSize + 1 + block - 1) / block);
    gpulsmopt_detail::
        sorted_build_radix_kernel<<<radix_grid, block, 0, stream>>>(
            run.keys.data(), count, sorted_radix_.data());
    CUDA_CHECK(cudaGetLastError());
    sorted_micro_ready_ = false;
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

  void ensure_sorted_microdirectory(cudaStream_t stream) {
    if (sorted_micro_ready_)
      return;
    if (!sorted_run_ready() || sorted_run().count == 0) {
      sorted_micro_ready_ = true;
      return;
    }
    const RunStorage &run = sorted_run();
    sorted_micro_base_.resize_discard(gpulsmopt_detail::kSortedRunRadixSize);
    sorted_micro_bits_.resize_discard(gpulsmopt_detail::kSortedRunRadixSize);
    sorted_micro_offsets_.resize_discard((run.count + 1u) / 2u);
    constexpr int block = 256;
    constexpr int grid = gpulsmopt_detail::kSortedRunRadixSize / block;
    gpulsmopt_detail::sorted_micro_plan_kernel<<<grid, block, 0, stream>>>(
        sorted_radix_.data(), sorted_micro_bits_.data(),
        sorted_micro_base_.data());
    CUDA_CHECK(cudaGetLastError());
    auto policy = thrust::cuda::par.on(stream);
    thrust::exclusive_scan(policy, sorted_micro_base_.data(),
                           sorted_micro_base_.data() +
                               gpulsmopt_detail::kSortedRunRadixSize,
                           sorted_micro_base_.data());
    gpulsmopt_detail::sorted_micro_fill_kernel<<<grid, block, 0, stream>>>(
        run.keys.data(), sorted_radix_.data(), sorted_micro_bits_.data(),
        sorted_micro_base_.data(), sorted_micro_offsets_.data());
    CUDA_CHECK(cudaGetLastError());
    sorted_micro_ready_ = true;
  }

  void build_sorted_run_cache(std::size_t index, cudaStream_t stream) {
    clear_sorted_state();
    if (index >= runs_.size() || !runs_[index].fully_sorted ||
        !runs_[index].unique_keys)
      return;
    sorted_run_index_ = index;
    build_sorted_metadata(stream);
    build_sorted_range_cdf(stream);
    ensure_sorted_microdirectory(stream);
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
    return {epoch.keys.data(),
            epoch.values.data(),
            epoch.count_delta.data(),
            epoch.quotient_off.data(),
            epoch.subgroup_masks.data(),
            epoch.quotient_live.data(),
            epoch.quotient_count_sum.data(),
            epoch.quotient_value_sum.data(),
            epoch.quotient_count_prefix.data(),
            epoch.quotient_value_prefix.data(),
            epoch.subgroup_value_prefix_ready
                ? epoch.subgroup_value_prefix.data()
                : nullptr,
            epoch.heavy_list.data(),
            epoch.heavy_count.data(),
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

  void create_unsorted_run(const std::uint32_t *keys,
                           const std::uint32_t *values, std::size_t count,
                           cudaStream_t stream) {
    if (run_count() >= static_cast<std::size_t>(gpulsmopt_detail::kRunCapacity))
      compact_runs(stream);
    acquire_run_slot();
    RunStorage &epoch = runs_.back();
    epoch.count = count;
    epoch.fully_sorted = false;
    epoch.unit_counts = false;
    epoch.unique_keys = false;
    epoch.keys.resize_discard(count);
    epoch.values.resize_discard(count);
    {
      GPULSMOPT_PROF_PHASE(prof_delta_sort_ms_);
      sort_run_batch(keys, values, count, epoch.keys.data(),
                     epoch.values.data(), stream);
    }
    {
      GPULSMOPT_PROF_PHASE(prof_delta_ingest_ms_);
      commit_run_metadata(epoch, stream);
    }
  }

  void create_delete_run(const std::uint32_t *keys, std::size_t count,
                         bool sorted, cudaStream_t stream) {
    if (run_count() >= static_cast<std::size_t>(gpulsmopt_detail::kRunCapacity))
      compact_runs(stream);
    acquire_run_slot();
    RunStorage &epoch = runs_.back();
    epoch.count = count;
    epoch.fully_sorted = sorted;
    epoch.unit_counts = false;
    epoch.unique_keys = false;
    epoch.keys.resize_discard(count);
    epoch.values.resize_discard(count);
    if (sorted) {
      CUDA_CHECK(cudaMemcpyAsync(epoch.keys.data(), keys,
                                 count * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToDevice, stream));
    } else {
      sort_run_batch(keys, keys, count, epoch.keys.data(), epoch.values.data(),
                     stream);
    }
    commit_run_metadata(
        epoch, stream, nullptr,
        static_cast<std::uint8_t>(gpulsmopt_detail::kTombstone));
  }
  void acquire_run_slot() {
    if (run_pool_.empty()) {
      runs_.emplace_back();
      return;
    }
    runs_.push_back(std::move(run_pool_.back()));
    run_pool_.pop_back();
  }

  void reserve_run_storage(std::size_t count) {
    while (runs_.size() + run_pool_.size() <
           static_cast<std::size_t>(gpulsmopt_detail::kRunCapacity)) {
      run_pool_.emplace_back();
    }
    for (auto &epoch : run_pool_) {
      epoch.keys.resize_discard(count);
      epoch.values.resize_discard(count);
      epoch.count_delta.resize_discard(count);
      epoch.quotient_off.resize_discard(gpulsmopt_detail::kEpochQuotients + 1);
      epoch.subgroup_masks.resize_discard(
          gpulsmopt_detail::kEpochQuotients *
          gpulsmopt_detail::kEpochSubgroupPlanes);
      epoch.quotient_live.resize_discard(gpulsmopt_detail::kEpochQuotients);
      epoch.quotient_count_sum.resize_discard(
          gpulsmopt_detail::kEpochQuotients);
      epoch.quotient_value_sum.resize_discard(
          gpulsmopt_detail::kEpochQuotients);
      epoch.quotient_count_prefix.resize_discard(
          gpulsmopt_detail::kEpochQuotients + 1);
      epoch.quotient_value_prefix.resize_discard(
          gpulsmopt_detail::kEpochQuotients + 1);
      epoch.subgroup_value_prefix.resize_discard(0);
      epoch.heavy_list.resize_discard(gpulsmopt_detail::kEpochQuotients);
      epoch.heavy_count.resize_discard(1);
    }
    const std::size_t pending_capacity =
        count * static_cast<std::size_t>(gpulsmopt_detail::kRunCapacity);
    run_merge_keys_.resize_discard(pending_capacity);
  }

  void reserve_compaction_storage(std::size_t run_capacity,
                                  cudaStream_t stream) {
    if (max_elements_ == 0u || run_pool_.empty())
      return;
    compact_output_capacity_ = max_elements_ * 2u;
    constexpr std::size_t tail_runs = gpulsmopt_detail::kCompactMergeRuns - 1u;
    const std::size_t scan_limit =
        static_cast<std::size_t>(std::numeric_limits<int>::max() / 2);
    if (compact_output_capacity_ > scan_limit ||
        run_capacity > (scan_limit - compact_output_capacity_) / tail_runs) {
      compact_output_capacity_ = 0u;
      return;
    }

    compact_output_keys_.resize_discard_exact(compact_output_capacity_);
    compact_output_values_.resize_discard_exact(compact_output_capacity_);
    compact_output_counts_.resize_discard_exact(compact_output_capacity_);

    auto first_slot = run_pool_.begin();
    for (auto it = run_pool_.begin(); it != run_pool_.end(); ++it) {
      if (it->keys.capacity() > first_slot->keys.capacity())
        first_slot = it;
    }
    const std::size_t large_index = std::min<std::size_t>(
        gpulsmopt_detail::kCompactTargetRuns, run_pool_.size() - 1u);
    auto large_slot = run_pool_.begin() + large_index;
    if (first_slot != large_slot)
      std::iter_swap(first_slot, large_slot);
    RunStorage &first = *large_slot;
    first.keys.resize_discard_exact(compact_output_capacity_);
    first.values.resize_discard_exact(compact_output_capacity_);
    first.count_delta.resize_discard_exact(compact_output_capacity_);

    const std::size_t input_capacity =
        compact_output_capacity_ + tail_runs * run_capacity;
    compact_views_.resize_discard_exact(gpulsmopt_detail::kCompactMergeRuns);
    compact_combined_.resize_discard_exact(gpulsmopt_detail::kEpochQuotients +
                                           1u);
    compact_segoff_.resize_discard_exact(gpulsmopt_detail::kEpochQuotients +
                                         1u);
    compact_keys_.resize_discard_exact(input_capacity);
    compact_payload_.resize_discard_exact(input_capacity);
    compact_alt_keys_.resize_discard_exact(input_capacity);
    compact_alt_payload_.resize_discard_exact(input_capacity);
    compact_flags_.resize_discard_exact(input_capacity);
    compact_pos_.resize_discard_exact(input_capacity);
    compact_heavy_list_.resize_discard_exact(gpulsmopt_detail::kEpochQuotients);
    compact_heavy_count_.resize_discard_exact(1u);
    compact_heavy_begin_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients);
    compact_heavy_end_.resize_discard_exact(gpulsmopt_detail::kEpochQuotients);
    compact_num_runs_.resize_discard_exact(1u);
    compact_group_unique_.resize_discard_exact(1u);

    // Prewarm CUB temp storage for the compaction pipeline.
    std::size_t seg_bytes = 0u;
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, seg_bytes, compact_keys_.data(), compact_alt_keys_.data(),
        compact_payload_.data(), compact_alt_payload_.data(),
        static_cast<int>(input_capacity),
        gpulsmopt_detail::kEpochQuotients, compact_heavy_begin_.data(),
        compact_heavy_end_.data(), 0, 16, stream));
    std::size_t reduce_bytes = 0u;
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        nullptr, reduce_bytes, compact_keys_.data(), compact_alt_keys_.data(),
        compact_payload_.data(), compact_alt_payload_.data(),
        compact_num_runs_.data(), gpulsmopt_detail::SumDeltaPayload{},
        static_cast<int>(input_capacity), stream));
    ensure_sort_temp(std::max(seg_bytes, reduce_bytes));

    if (input_capacity <= scan_u32_count_)
      return;
    std::size_t temp_bytes = 0u;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_bytes, compact_flags_.data(), compact_pos_.data(),
        static_cast<int>(input_capacity), stream));
    scan_u32_count_ = input_capacity;
    scan_u32_temp_bytes_ = temp_bytes;
    ensure_sort_temp(temp_bytes);
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

  std::size_t c0_log_live() const { return c0_log_count_; }

  std::size_t c0_flush_budget() const {
    std::size_t budget = std::min<std::size_t>(
        static_cast<std::size_t>(GPULSMOPT_C0_FLUSH_BUDGET),
        gpulsmopt_detail::kC0LogMaxIndex);
    if (max_elements_ > 0)
      budget = std::min(budget, max_elements_);
    return std::max<std::size_t>(budget, 1);
  }

  enum class RunKind : std::uint8_t {
    empty,
    inserts,
    tombstones,
    mixed,
  };

  static RunKind operation_kind(std::uint8_t op) {
    return op == gpulsmopt_detail::kInsert ? RunKind::inserts
                                           : RunKind::tombstones;
  }

  static std::uint8_t operation_value(RunKind kind) {
    return kind == RunKind::inserts
               ? static_cast<std::uint8_t>(gpulsmopt_detail::kInsert)
               : static_cast<std::uint8_t>(gpulsmopt_detail::kTombstone);
  }

  void ensure_c0_log(cudaStream_t stream) {
    (void)stream;
    const std::size_t reserve_count = c0_flush_budget();
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
    const RunKind incoming_kind = operation_kind(op);
    if (c0_kind_ == RunKind::empty) {
      c0_kind_ = incoming_kind;
    } else if (c0_kind_ != incoming_kind && c0_kind_ != RunKind::mixed) {
      CUDA_CHECK(cudaMemsetAsync(raw_or_null(c0_log_ops_),
                                 operation_value(c0_kind_), old_total, stream));
      c0_kind_ = RunKind::mixed;
    }
    std::uint8_t *ops =
        c0_kind_ == RunKind::mixed ? raw_or_null(c0_log_ops_) : nullptr;
    {
      GPULSMOPT_PROF_PHASE(prof_append_ms_);
      const int block = 256;
      const int grid = static_cast<int>((count + block - 1) / block);
      gpulsmopt_detail::c0_log_append_kernel<<<grid, block, 0, stream>>>(
          keys_in, values_in, op, old_total, count, raw_or_null(c0_log_keys_),
          raw_or_null(c0_log_values_), ops);
      CUDA_CHECK(cudaGetLastError());
    }
    c0_log_count_ = static_cast<std::uint32_t>(new_total);
    return true;
  }

  void clear_c0_log(cudaStream_t stream) {
    if (c0_log_count_ == 0)
      return;
    c0_log_count_ = 0;
    c0_kind_ = RunKind::empty;
    (void)stream;
  }

  void drain_c0_for_space(cudaStream_t stream) {
    if (c0_log_count_ != 0)
      merge_down(stream);
  }

  void insert_records(const std::uint32_t *keys_in,
                      const std::uint32_t *values_in, std::uint8_t op,
                      std::size_t count, bool keys_sorted,
                      cudaStream_t stream) {
    if (op == static_cast<std::uint8_t>(gpulsmopt_detail::kTombstone) &&
        try_direct_delete(keys_in, count, keys_sorted, stream))
      return;
    if (try_create_epoch(keys_in, values_in, op, count, stream))
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

  void prepare_sort_storage(std::size_t direct_count, std::size_t log_count,
                            cudaStream_t stream) {
    direct_sort_keys_.resize_discard(direct_count);
    direct_sort_values_.resize_discard(direct_count);
    resize_reuse(sort_key_output_, log_count);
    sort_payload_input_.resize_discard(log_count);
    sort_payload_output_.resize_discard(log_count);
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
    std::size_t log_bytes = 0;
    if (log_count > 0) {
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, log_bytes, raw_or_null(c0_log_keys_),
          raw_or_null(sort_key_output_), sort_payload_input_.data(),
          sort_payload_output_.data(), log_count, 0, 32, stream));
      log_sort_count_ = log_count;
      log_sort_temp_bytes_ = log_bytes;
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
    std::size_t scan_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, direct_sort_keys_.data(),
        direct_sort_values_.data(), gpulsmopt_detail::kEpochQuotients, stream));
    scan_u32_count_ = gpulsmopt_detail::kEpochQuotients;
    scan_u32_temp_bytes_ = scan_bytes;
    ensure_sort_temp(
        std::max({direct_bytes, log_bytes, epoch_bytes, scan_bytes}));
  }

  // Reserve direct-path buffers before timed updates.
  void prepare_for_insert(cudaStream_t stream) {
    const std::size_t direct_count = std::min(
        max_elements_, std::max(4 * c0_flush_budget(), batch_capacity_));
    ensure_c0_log(stream);
    reserve_run_storage(direct_count);
    owner_class_list_.resize_discard_exact(4u *
                                           gpulsmopt_detail::kEpochQuotients);
    owner_class_count_.resize_discard_exact(4u);
    meta_pending_list_.resize_discard_exact((direct_count + 31u) / 32u + 1u);
    meta_pending_count_.resize_discard_exact(1u);
    prepare_sort_storage(direct_count, c0_flush_budget(), stream);
    reserve_compaction_storage(direct_count, stream);
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

  void consolidate_all_runs(cudaStream_t stream) {
    if (runs_.size() < 2)
      return;
    std::size_t total = 0;
    for (const auto &epoch : runs_)
      total += epoch.count;
    if (total == 0) {
      clear_run_state();
      return;
    }
    run_merge_keys_.resize_discard(total);
    sort_payload_input_.resize_discard(total);
    sort_payload_output_.resize_discard(total);
    direct_sort_keys_.resize_discard(total);
    std::size_t offset = 0;
    constexpr int block = 256;
    for (const auto &epoch : runs_) {
      CUDA_CHECK(cudaMemcpyAsync(run_merge_keys_.data() + offset,
                                 epoch.keys.data(),
                                 epoch.count * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToDevice, stream));
      const int grid = static_cast<int>((epoch.count + block - 1) / block);
      gpulsmopt_detail::pack_delta_payload_kernel<<<grid, block, 0, stream>>>(
          epoch.values.data(), epoch.count_delta.data(), epoch.count,
          sort_payload_input_.data() + offset);
      CUDA_CHECK(cudaGetLastError());
      offset += epoch.count;
    }
    std::size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes, run_merge_keys_.data(), direct_sort_keys_.data(),
        sort_payload_input_.data(), sort_payload_output_.data(), total, 0, 32,
        stream));
    ensure_sort_temp(temp_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sort_temp_storage_.data(), temp_bytes, run_merge_keys_.data(),
        direct_sort_keys_.data(), sort_payload_input_.data(),
        sort_payload_output_.data(), total, 0, 32, stream));
    auto policy = thrust::cuda::par.on(stream);
    auto unique_end = thrust::reduce_by_key(
        policy, direct_sort_keys_.data(), direct_sort_keys_.data() + total,
        sort_payload_output_.data(), run_merge_keys_.data(),
        sort_payload_input_.data(), thrust::equal_to<std::uint32_t>(),
        gpulsmopt_detail::SumDeltaPayload{});
    const std::size_t unique_count =
        static_cast<std::size_t>(unique_end.first - run_merge_keys_.data());
    clear_run_state();
    acquire_run_slot();
    RunStorage &epoch = runs_.back();
    epoch.count = unique_count;
    epoch.fully_sorted = true;
    epoch.unit_counts = false;
    epoch.unique_keys = true;
    epoch.keys.resize_discard(unique_count);
    epoch.values.resize_discard(unique_count);
    epoch.count_delta.resize_discard(unique_count);
    CUDA_CHECK(cudaMemcpyAsync(epoch.keys.data(), run_merge_keys_.data(),
                               unique_count * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToDevice, stream));
    const int grid = static_cast<int>((unique_count + block - 1) / block);
    gpulsmopt_detail::unpack_delta_payload_kernel<<<grid, block, 0, stream>>>(
        sort_payload_input_.data(), unique_count, epoch.values.data(),
        epoch.count_delta.data());
    CUDA_CHECK(cudaGetLastError());
    commit_run_metadata(epoch, stream, nullptr,
                        static_cast<std::uint8_t>(gpulsmopt_detail::kInsert),
                        false);
  }

  struct CompactRange {
    std::size_t begin;
    std::size_t end;
    std::size_t total;
  };

  CompactRange plan_compaction_group() const {
    const std::size_t n = runs_.size();
    const std::size_t target = gpulsmopt_detail::kCompactTargetRuns;
    if (n <= target)
      return {0u, 0u, 0u};
    const std::size_t merge_count = n - target + 1u;
    if (merge_count >
        static_cast<std::size_t>(gpulsmopt_detail::kCompactMergeRuns)) {
      throw std::runtime_error("run capacity exceeded");
    }
    std::size_t window = 0u;
    for (std::size_t r = 0; r < merge_count; ++r)
      window += runs_[r].count;
    std::size_t best_begin = 0u;
    std::size_t best_total = window;
    for (std::size_t begin = 1u; begin + merge_count <= n; ++begin) {
      window -= runs_[begin - 1u].count;
      window += runs_[begin + merge_count - 1u].count;
      if (window <= best_total) {
        best_begin = begin;
        best_total = window;
      }
    }
    return {best_begin, best_begin + merge_count, best_total};
  }

  void compact_runs(cudaStream_t stream) {
    const CompactRange group = plan_compaction_group();
    if (group.end == 0u)
      return;
    if (group.total >
        static_cast<std::size_t>(std::numeric_limits<int>::max() / 2)) {
      throw std::runtime_error("compaction group is too large");
    }
    // Exact record count: no padded tail work anywhere.
    const std::size_t total = group.total;
    if (total > compact_keys_.capacity()) {
      throw std::runtime_error("compaction workspace was not preallocated");
    }
    constexpr int block = 256;
    const std::size_t rows = gpulsmopt_detail::kEpochQuotients + 1u;
    const std::size_t group_count = group.end - group.begin;
    compact_views_.resize_discard(group_count);
    compact_combined_.resize_discard(rows);
    compact_segoff_.resize_discard(rows);
    compact_keys_.resize_discard(total);
    compact_payload_.resize_discard(total);
    compact_alt_keys_.resize_discard(total);
    compact_alt_payload_.resize_discard(total);
    compact_flags_.resize_discard(total);
    compact_pos_.resize_discard(total);
    compact_heavy_list_.resize_discard(gpulsmopt_detail::kEpochQuotients);
    compact_heavy_count_.resize_discard(1u);
    compact_heavy_begin_.resize_discard(gpulsmopt_detail::kEpochQuotients);
    compact_heavy_end_.resize_discard(gpulsmopt_detail::kEpochQuotients);
    compact_num_runs_.resize_discard(1u);
    compact_group_unique_.resize_discard(1u);

    std::vector<gpulsmopt_detail::RunView> views;
    views.reserve(group_count);
    for (std::size_t r = group.begin; r < group.end; ++r)
      views.push_back(make_run_view(runs_[r]));
    CUDA_CHECK(cudaMemcpyAsync(compact_views_.data(), views.data(),
                               views.size() * sizeof(views[0]),
                               cudaMemcpyHostToDevice, stream));

    const int plan_grid = static_cast<int>((rows + block - 1u) / block);
    gpulsmopt_detail::compact_plan_kernel<<<plan_grid, block, 0, stream>>>(
        compact_views_.data(), static_cast<int>(views.size()),
        compact_combined_.data());
    CUDA_CHECK(cudaGetLastError());
    exclusive_scan_u32(compact_combined_.data(), compact_segoff_.data(), rows,
                       stream);
    CUDA_CHECK(cudaMemsetAsync(compact_heavy_count_.data(), 0,
                               sizeof(std::uint32_t), stream));
    const int classify_grid = gpulsmopt_detail::kEpochQuotients / block;
    gpulsmopt_detail::
        compact_classify_heavy_kernel<<<classify_grid, block, 0, stream>>>(
            compact_combined_.data(), compact_heavy_list_.data(),
            compact_heavy_count_.data());
    CUDA_CHECK(cudaGetLastError());
    std::uint32_t heavy_items = 0u;
    CUDA_CHECK(cudaMemcpyAsync(&heavy_items, compact_heavy_count_.data(),
                               sizeof(heavy_items), cudaMemcpyDeviceToHost,
                               stream));
    gpulsmopt_detail::compact_sort_light_kernel<<<
        gpulsmopt_detail::kEpochQuotients, block, 0, stream>>>(
        compact_views_.data(), static_cast<int>(views.size()),
        compact_combined_.data(), compact_segoff_.data(), compact_keys_.data(),
        compact_payload_.data());
    CUDA_CHECK(cudaGetLastError());
    gpulsmopt_detail::compact_gather_heavy_kernel<<<64, block, 0, stream>>>(
        compact_views_.data(), static_cast<int>(views.size()),
        compact_heavy_list_.data(), compact_heavy_count_.data(),
        compact_segoff_.data(), compact_keys_.data(),
        compact_payload_.data());
    CUDA_CHECK(cudaGetLastError());
    gpulsmopt_detail::
        compact_heavy_bounds_kernel<<<classify_grid, block, 0, stream>>>(
            compact_heavy_list_.data(), compact_heavy_count_.data(),
            compact_segoff_.data(), compact_combined_.data(),
            compact_heavy_begin_.data(), compact_heavy_end_.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (heavy_items > 0u) {
      // Installed segmented radix fallback, low 16 bits only.
      std::size_t seg_bytes = 0u;
      CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
          nullptr, seg_bytes, compact_keys_.data(), compact_alt_keys_.data(),
          compact_payload_.data(), compact_alt_payload_.data(),
          static_cast<int>(total), static_cast<int>(heavy_items),
          compact_heavy_begin_.data(), compact_heavy_end_.data(), 0, 16,
          stream));
      ensure_sort_temp(seg_bytes);
      CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
          sort_temp_storage_.data(), seg_bytes, compact_keys_.data(),
          compact_alt_keys_.data(), compact_payload_.data(),
          compact_alt_payload_.data(), static_cast<int>(total),
          static_cast<int>(heavy_items), compact_heavy_begin_.data(),
          compact_heavy_end_.data(), 0, 16, stream));
      gpulsmopt_detail::
          compact_heavy_copyback_kernel<<<64, block, 0, stream>>>(
              compact_heavy_list_.data(), compact_heavy_count_.data(),
              compact_segoff_.data(), compact_combined_.data(),
              compact_alt_keys_.data(), compact_alt_payload_.data(),
              compact_keys_.data(), compact_payload_.data());
      CUDA_CHECK(cudaGetLastError());
    }
    // Sum duplicate deltas across the group in one pass.
    std::size_t reduce_bytes = 0u;
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        nullptr, reduce_bytes, compact_keys_.data(), compact_alt_keys_.data(),
        compact_payload_.data(), compact_alt_payload_.data(),
        compact_num_runs_.data(), gpulsmopt_detail::SumDeltaPayload{},
        static_cast<int>(total), stream));
    ensure_sort_temp(reduce_bytes);
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        sort_temp_storage_.data(), reduce_bytes, compact_keys_.data(),
        compact_alt_keys_.data(), compact_payload_.data(),
        compact_alt_payload_.data(), compact_num_runs_.data(),
        gpulsmopt_detail::SumDeltaPayload{}, static_cast<int>(total), stream));
    const int item_grid = static_cast<int>((total + block - 1u) / block);
    gpulsmopt_detail::
        compact_filter_flags_kernel<<<item_grid, block, 0, stream>>>(
            compact_alt_payload_.data(), compact_num_runs_.data(), total,
            compact_flags_.data());
    CUDA_CHECK(cudaGetLastError());
    exclusive_scan_u32(compact_flags_.data(), compact_pos_.data(), total,
                       stream);
    gpulsmopt_detail::compact_unique_count_kernel<<<1, 1, 0, stream>>>(
        compact_flags_.data(), compact_pos_.data(), total,
        compact_group_unique_.data(), 0u);
    CUDA_CHECK(cudaGetLastError());

    std::uint32_t unique_count = 0u;
    CUDA_CHECK(cudaMemcpyAsync(&unique_count, compact_group_unique_.data(),
                               sizeof(unique_count), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (unique_count > compact_output_keys_.capacity()) {
      throw std::runtime_error("compaction output exceeds configured capacity");
    }

    compact_output_keys_.resize_discard(unique_count);
    compact_output_values_.resize_discard(unique_count);
    compact_output_counts_.resize_discard(unique_count);
    gpulsmopt_detail::compact_scatter_kernel<<<item_grid, block, 0, stream>>>(
        compact_alt_keys_.data(), compact_alt_payload_.data(),
        compact_flags_.data(), compact_pos_.data(), total,
        compact_output_keys_.data(), compact_output_values_.data(),
        compact_output_counts_.data());
    CUDA_CHECK(cudaGetLastError());

    const std::size_t old_sorted = sorted_run_index_;
    const bool cache_selected =
        old_sorted >= group.begin && old_sorted < group.end;
    std::size_t next_sorted = std::numeric_limits<std::size_t>::max();
    if (sorted_run_ready() && !cache_selected) {
      next_sorted = old_sorted < group.begin ? old_sorted
                                             : old_sorted - (group_count - 1u);
    }
    if (cache_selected)
      clear_sorted_state();

    RunStorage merged = std::move(runs_[group.begin]);
    std::swap(merged.keys, compact_output_keys_);
    std::swap(merged.values, compact_output_values_);
    std::swap(merged.count_delta, compact_output_counts_);
    merged.count = unique_count;
    merged.fully_sorted = true;
    merged.unit_counts = false;
    merged.unique_keys = true;
    merged.keys.resize_discard(unique_count);
    merged.values.resize_discard(unique_count);
    merged.count_delta.resize_discard(unique_count);
    compact_output_keys_.resize_discard(compact_output_capacity_);
    compact_output_values_.resize_discard(compact_output_capacity_);
    compact_output_counts_.resize_discard(compact_output_capacity_);

    std::vector<RunStorage> next;
    next.reserve(gpulsmopt_detail::kRunCapacity);
    for (std::size_t r = 0; r < group.begin; ++r)
      next.push_back(std::move(runs_[r]));
    next.push_back(std::move(merged));
    for (std::size_t r = group.begin + 1u; r < group.end; ++r)
      run_pool_.push_back(std::move(runs_[r]));
    for (std::size_t r = group.end; r < runs_.size(); ++r)
      next.push_back(std::move(runs_[r]));
    runs_ = std::move(next);
    runs_.reserve(gpulsmopt_detail::kRunCapacity);
    if (!cache_selected)
      sorted_run_index_ = next_sorted;

    RunStorage &out = runs_[group.begin];
    prepare_run_metadata_storage(out);
    launch_run_metadata_kernels(out, out.count, stream, false);
    constexpr int count_grid = gpulsmopt_detail::kEpochQuotients / block;
    gpulsmopt_detail::
        epoch_count_delta_sum_kernel<<<count_grid, block, 0, stream>>>(
            make_run_view(out));
    CUDA_CHECK(cudaGetLastError());

    prebind_run_views(stream);
    invalidate_range_delta_projection();
    run_meta_ready_ = false;
    run_meta_counts_ready_ = false;
    logical_run_views_ready_ = false;
    ++structure_generation_;
  }

  // Lazy [quotient][run] tables for run-parallel reads.
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

  // Refresh the successor bitmap only on reads.
  void ensure_successor_bitmap(cudaStream_t stream) {
    if (!owner_ready_)
      initialize_owner_storage(stream);
    const bool fresh = succ_bits_.size() == 0;
    if (fresh)
      succ_bits_.resize_discard_exact(gpulsmopt_detail::kSuccTotalWords);
    const bool rebuild = fresh || succ_generation_ != structure_generation_;
    if (!rebuild && succ_synced_runs_ == runs_.size())
      return;
    constexpr int block = 256;
    if (rebuild) {
      CUDA_CHECK(cudaMemsetAsync(
          succ_bits_.data(), 0,
          static_cast<std::size_t>(gpulsmopt_detail::succ_level_words(0)) *
              sizeof(std::uint32_t),
          stream));
      const int grid = gpulsmopt_detail::kEpochQuotients * 32 / block;
      gpulsmopt_detail::
          succ_rebuild_from_owner_kernel<<<grid, block, 0, stream>>>(
              make_owner_view(), succ_bits_.data());
      CUDA_CHECK(cudaGetLastError());
      const int spill_grid =
          static_cast<int>((owner_spill_slots_ + block - 1) / block);
      gpulsmopt_detail::
          succ_rebuild_from_spill_kernel<<<spill_grid, block, 0, stream>>>(
              make_owner_view(), succ_bits_.data());
      CUDA_CHECK(cudaGetLastError());
    } else {
      for (std::size_t e = succ_synced_runs_; e < runs_.size(); ++e) {
        const std::size_t n = runs_[e].count;
        if (n == 0)
          continue;
        const int grid = static_cast<int>((n + block - 1) / block);
        gpulsmopt_detail::succ_apply_epoch_kernel<<<grid, block, 0, stream>>>(
            runs_[e].keys.data(), runs_[e].count_delta.data(), n,
            succ_bits_.data());
        CUDA_CHECK(cudaGetLastError());
      }
    }
    if (rebuild) {
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
    } else {
      for (int level = 1; level < 6; ++level) {
        const std::uint32_t *lower =
            succ_bits_.data() + gpulsmopt_detail::succ_level_off(level - 1);
        std::uint32_t *upper =
            succ_bits_.data() + gpulsmopt_detail::succ_level_off(level);
        for (std::size_t e = succ_synced_runs_; e < runs_.size(); ++e) {
          const std::size_t n = runs_[e].count;
          if (n == 0)
            continue;
          const int grid = static_cast<int>((n + block - 1) / block);
          gpulsmopt_detail::
              succ_update_level_kernel<<<grid, block, 0, stream>>>(
                  runs_[e].keys.data(), n, lower, upper, level * 5);
          CUDA_CHECK(cudaGetLastError());
        }
      }
    }
    succ_generation_ = structure_generation_;
    succ_synced_runs_ = runs_.size();
  }

  bool try_create_epoch(const std::uint32_t *keys_in,
                        const std::uint32_t *values_in, std::uint8_t op,
                        std::size_t count, cudaStream_t stream) {
    const std::size_t threshold = GPULSMOPT_SCATTER_MIN_BATCH;
    if (op != gpulsmopt_detail::kInsert || values_in == nullptr ||
        count < threshold || count > std::numeric_limits<std::uint32_t>::max())
      return false;
    if (c0_log_count_ != 0)
      merge_down(stream);
    create_unsorted_run(keys_in, values_in, count, stream);
    return true;
  }

  bool try_direct_delete(const std::uint32_t *keys, std::size_t count,
                         bool sorted, cudaStream_t stream) {
    if (count < GPULSMOPT_SCATTER_MIN_BATCH ||
        count > std::numeric_limits<std::uint32_t>::max())
      return false;
    if (c0_log_count_ > 0)
      merge_down(stream);
    create_delete_run(keys, count, sorted, stream);
    return true;
  }

  void merge_down(cudaStream_t stream) {
    if (c0_log_count_ == 0)
      return;
    last_update_stream_ = stream;
    const std::size_t count = c0_log_count_;
    if (run_count() >= static_cast<std::size_t>(gpulsmopt_detail::kRunCapacity))
      compact_runs(stream);
    resize_reuse(sort_key_output_, count);
    sort_payload_input_.resize_discard(count);
    sort_payload_output_.resize_discard(count);
    const std::uint8_t *ops =
        c0_kind_ == RunKind::mixed ? raw_or_null(c0_log_ops_) : nullptr;
    const std::uint8_t constant_op = operation_value(c0_kind_);
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    gpulsmopt_detail::pack_latest_payload_kernel<<<grid, block, 0, stream>>>(
        raw_or_null(c0_log_values_), ops, constant_op, count,
        sort_payload_input_.data());
    CUDA_CHECK(cudaGetLastError());
    if (count > log_sort_count_) {
      log_sort_temp_bytes_ = 0;
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, log_sort_temp_bytes_, raw_or_null(c0_log_keys_),
          raw_or_null(sort_key_output_), sort_payload_input_.data(),
          sort_payload_output_.data(), count, 0, 32, stream));
      log_sort_count_ = count;
    }
    std::size_t temp_bytes = log_sort_temp_bytes_;
    ensure_sort_temp(temp_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sort_temp_storage_.data(), temp_bytes, raw_or_null(c0_log_keys_),
        raw_or_null(sort_key_output_), sort_payload_input_.data(),
        sort_payload_output_.data(), count, 0, 32, stream));
    acquire_run_slot();
    RunStorage &epoch = runs_.back();
    epoch.keys.resize_discard(count);
    epoch.values.resize_discard(count);
    resize_reuse(merge_ops_, count);
    auto policy = thrust::cuda::par.on(stream);
    auto unique_end = thrust::reduce_by_key(
        policy, sort_key_output_.begin(), sort_key_output_.begin() + count,
        thrust::device_pointer_cast(sort_payload_output_.data()),
        thrust::device_pointer_cast(epoch.keys.data()),
        thrust::device_pointer_cast(sort_payload_input_.data()),
        thrust::equal_to<std::uint32_t>(),
        gpulsmopt_detail::TakeLatestPayload{});
    const std::size_t unique_count = static_cast<std::size_t>(
        unique_end.first - thrust::device_pointer_cast(epoch.keys.data()));
    epoch.count = unique_count;
    epoch.fully_sorted = true;
    epoch.unit_counts = false;
    epoch.unique_keys = true;
    epoch.keys.resize_discard(unique_count);
    epoch.values.resize_discard(unique_count);
    gpulsmopt_detail::unpack_latest_payload_kernel<<<
        static_cast<int>((unique_count + block - 1) / block), block, 0,
        stream>>>(sort_payload_input_.data(), unique_count, epoch.values.data(),
                  raw_or_null(merge_ops_));
    CUDA_CHECK(cudaGetLastError());
    commit_run_metadata(epoch, stream, raw_or_null(merge_ops_),
                        static_cast<std::uint8_t>(gpulsmopt_detail::kInsert));
    clear_c0_log(stream);
  }

  void maybe_flush_and_merge(cudaStream_t stream) {
    if (c0_log_live() >= c0_flush_budget())
      merge_down(stream);
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

  std::uint32_t c0_log_count_ = 0;
  RunKind c0_kind_ = RunKind::empty;
  thrust::device_vector<std::uint32_t> c0_log_keys_;
  thrust::device_vector<std::uint32_t> c0_log_values_;
  thrust::device_vector<std::uint8_t> c0_log_ops_;

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
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> sort_payload_input_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> sort_payload_output_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> sort_temp_storage_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> run_merge_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> sorted_value_prefix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> sorted_count_prefix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> sorted_range_cdf_;
  std::uint32_t sorted_range_min_key_ = 0u;
  std::uint64_t sorted_range_span_ = 0u;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> sorted_radix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> sorted_micro_base_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint16_t> sorted_micro_offsets_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> sorted_micro_bits_;
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
  cudaStream_t last_update_stream_ = nullptr;
  std::size_t direct_sort_count_ = 0;
  std::size_t direct_sort_temp_bytes_ = 0;
  std::size_t run_sort_count_ = 0;
  std::size_t run_sort_temp_bytes_ = 0;
  std::size_t log_sort_count_ = 0;
  std::size_t log_sort_temp_bytes_ = 0;
  std::size_t scan_u32_count_ = 0;
  std::size_t scan_u32_temp_bytes_ = 0;
  std::size_t metadata_scan_temp_bytes_ = 0;
  bool sorted_micro_ready_ = false;
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

  // Shard-parallel compaction staging.
  gpulsmopt_detail::RawDeviceBuffer<gpulsmopt_detail::RunView> compact_views_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compact_combined_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compact_segoff_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compact_heavy_list_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compact_heavy_count_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compact_heavy_begin_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compact_heavy_end_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compact_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> compact_payload_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compact_alt_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> compact_alt_payload_;
  gpulsmopt_detail::RawDeviceBuffer<int> compact_num_runs_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compact_flags_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compact_pos_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compact_group_unique_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compact_output_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> compact_output_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::int8_t> compact_output_counts_;
  std::size_t compact_output_capacity_ = 0u;

  thrust::device_vector<std::uint8_t> merge_ops_;
};
