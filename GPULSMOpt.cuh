#pragma once
#include "gpu_dictionary_adapter.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
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
// Large batches bypass C0 and become epochs.
#ifndef GPULSMOPT_SCATTER_MIN_BATCH
#define GPULSMOPT_SCATTER_MIN_BATCH (1 << 18)
#endif
// Immutable sorted-batch epochs held before consolidation.
#ifndef GPULSMOPT_EPOCH_MAX
#define GPULSMOPT_EPOCH_MAX 16
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
constexpr int kEpochMax = GPULSMOPT_EPOCH_MAX;
constexpr int kEpochQuotientBits = 16;
constexpr int kEpochSubgroupBits = 4;
constexpr int kEpochQuotients = 1 << kEpochQuotientBits;
constexpr int kEpochSubgroups = 1 << kEpochSubgroupBits;
constexpr int kEpochSubgroupPlanes = kEpochSubgroupBits;
constexpr int kEpochSubgroupPrefixStride = kEpochSubgroups;
constexpr int kEpochHeavySortCap = 128;
constexpr int kEpochQuotientBitmapWords = kEpochQuotients / 32;
constexpr int kOwnerSlotBits = 11;
constexpr int kOwnerSlots = 1 << kOwnerSlotBits;
constexpr int kOwnerSlotMask = kOwnerSlots - 1;
constexpr std::uint32_t kOwnerNoPage = 0xffffffffu;
constexpr int kEpochStorageMax = kEpochMax + 1;
constexpr int kSpineRadixBits = 20;
constexpr int kSpineRadixShift = 32 - kSpineRadixBits;
constexpr int kSpineMicroTarget = 4;
constexpr int kRangeProjectionBits = 20;
constexpr int kRangeProjectionBins = 1 << kRangeProjectionBits;
constexpr int kRangeProjectionShift = 32 - kRangeProjectionBits;
constexpr std::size_t kRangeProjectionMinQueries = 1 << 18;
constexpr std::uint64_t kRangeCdfMaxRatio =
    GPULSMOPT_RANGE_CDF_MAX_RATIO;
constexpr std::size_t kSpineRadixSize =
    std::size_t{1} << kSpineRadixBits;
static_assert(GPULSMOPT_RADIX_THREADS % 32 == 0,
              "radix block size must be warp aligned");
constexpr std::uint32_t kEmptyKey = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kInsert = 1;
constexpr std::uint32_t kTombstone = 0;
constexpr std::uint32_t kC0LogMaxIndex = 0x00fffffeu;

#if GPULSMOPT_SM120_RADIX
struct Sm120RadixPolicy
    : cub::DeviceRadixSortPolicy<std::uint32_t, std::uint32_t,
                                 std::uint32_t> {
  using Base = cub::DeviceRadixSortPolicy<std::uint32_t, std::uint32_t,
                                           std::uint32_t>;
  using BasePolicy = typename Base::Policy900;

  struct Policy1200
      : cub::ChainedPolicy<1200, Policy1200, BasePolicy> {
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
        cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        cub::BLOCK_SCAN_RAKING_MEMOIZE, cub::RADIX_SORT_STORE_DIRECT,
        ONESWEEP_RADIX_BITS>;
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

inline cudaError_t epoch_radix_sort_pairs(
    void *temp_storage, std::size_t &temp_bytes,
    const std::uint32_t *keys_in, std::uint32_t *keys_out,
    const std::uint32_t *values_in, std::uint32_t *values_out,
    std::uint32_t count, int begin_bit, int end_bit,
    cudaStream_t stream) {
  cub::DoubleBuffer<std::uint32_t> keys(
      const_cast<std::uint32_t *>(keys_in), keys_out);
  cub::DoubleBuffer<std::uint32_t> values(
      const_cast<std::uint32_t *>(values_in), values_out);
  return cub::DispatchRadixSort<false, std::uint32_t, std::uint32_t,
                                std::uint32_t, Sm120RadixPolicy>::Dispatch(
      temp_storage, temp_bytes, keys, values, count, begin_bit, end_bit,
      false, stream);
}
#else
inline cudaError_t epoch_radix_sort_pairs(
    void *temp_storage, std::size_t &temp_bytes,
    const std::uint32_t *keys_in, std::uint32_t *keys_out,
    const std::uint32_t *values_in, std::uint32_t *values_out,
    std::uint32_t count, int begin_bit, int end_bit,
    cudaStream_t stream) {
  return cub::DeviceRadixSort::SortPairs(
      temp_storage, temp_bytes, keys_in, keys_out, values_in, values_out,
      count, begin_bit, end_bit, stream);
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
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&next),
                            count * sizeof(T)));
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

struct EpochView {
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
  std::uint32_t *quotient_bitmap;
  std::uint32_t *heavy_list;
  std::uint32_t *heavy_count;
  std::uint32_t fully_sorted;
};

struct SpineView {
  const std::uint32_t *keys;
  const std::uint32_t *values;
  const std::uint32_t *radix;
  const std::uint32_t *micro_base;
  const std::uint16_t *micro_offsets;
  const std::uint8_t *micro_bits;
  std::size_t count;
};

struct SpineRangeView {
  const std::uint32_t *cdf;
  std::uint32_t min_key;
  std::uint64_t span;
};

struct RangeDeltaView {
  const std::uint32_t *offsets;
  const std::uint32_t *value_prefix;
  const std::uint32_t *keys;
  const std::uint32_t *values;
};

struct OwnerView {
  std::uint64_t *primary;
  std::uint64_t *overflow;
  std::uint32_t *heads;
  std::uint32_t *next;
  std::uint32_t *page_alloc;
  std::uint32_t *error;
  std::uint32_t *quotient_live;
  std::uint32_t page_capacity;
};

constexpr std::uint32_t kOwnerEmpty = 0u;
constexpr std::uint32_t kOwnerLive = 1u;
constexpr std::uint32_t kOwnerTomb = 2u;
__host__ __device__ inline std::uint64_t owner_pack(
    std::uint32_t low, std::uint32_t value, std::uint32_t state) {
  return static_cast<std::uint64_t>(value) |
         (static_cast<std::uint64_t>(low) << 32) |
         (static_cast<std::uint64_t>(state) << 62);
}

__host__ __device__ inline std::uint32_t owner_state(
    std::uint64_t slot) {
  return static_cast<std::uint32_t>(slot >> 62);
}

__host__ __device__ inline std::uint32_t owner_low(
    std::uint64_t slot) {
  return static_cast<std::uint32_t>((slot >> 32) & 0xffffu);
}

__host__ __device__ inline std::uint32_t owner_value(
    std::uint64_t slot) {
  return static_cast<std::uint32_t>(slot);
}

__host__ __device__ inline std::uint32_t owner_slot(
    std::uint32_t low) {
  return (low * 0x9e3779b1u) >> (32 - kOwnerSlotBits);
}

__device__ inline void owner_add_live(
    OwnerView owner, std::uint32_t quotient,
    std::int32_t delta) {
  const std::int64_t live =
      static_cast<std::int64_t>(owner.quotient_live[quotient]) + delta;
  owner.quotient_live[quotient] = static_cast<std::uint32_t>(live);
}

__device__ inline bool owner_find_in_page(
    const std::uint64_t *page, std::uint32_t low,
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

__device__ inline bool owner_find(OwnerView owner, std::uint32_t key,
                                  std::uint64_t *found) {
  const std::uint32_t quotient = key >> kEpochQuotientBits;
  const std::uint32_t low = key & 0xffffu;
  const std::uint64_t *page =
      owner.primary + static_cast<std::size_t>(quotient) * kOwnerSlots;
  if (owner_find_in_page(page, low, found))
    return true;
  std::uint32_t link = owner.heads[quotient];
  while (link != kOwnerNoPage) {
    page = owner.overflow +
           static_cast<std::size_t>(link) * kOwnerSlots;
    if (owner_find_in_page(page, low, found))
      return true;
    link = owner.next[link];
  }
  return false;
}
__device__ inline bool owner_insert_in_page(
    std::uint64_t *page, std::uint32_t low, std::uint32_t value,
    std::uint64_t *previous) {
  std::uint32_t slot = owner_slot(low);
  std::uint32_t reusable = kOwnerNoPage;
  for (int probe = 0; probe < kOwnerSlots; ++probe) {
    const std::uint64_t packed = page[slot];
    const std::uint32_t state = owner_state(packed);
    if (state == kOwnerEmpty) {
      const std::uint32_t target =
          reusable == kOwnerNoPage ? slot : reusable;
      *previous = page[target];
      page[target] = owner_pack(low, value, kOwnerLive);
      return true;
    }
    if (owner_low(packed) == low) {
      *previous = packed;
      page[slot] = owner_pack(low, value, kOwnerLive);
      return true;
    }
    if (state == kOwnerTomb && reusable == kOwnerNoPage)
      reusable = slot;
    slot = (slot + 1u) & kOwnerSlotMask;
  }
  if (reusable != kOwnerNoPage) {
    *previous = page[reusable];
    page[reusable] = owner_pack(low, value, kOwnerLive);
    return true;
  }
  return false;
}

__device__ inline std::uint32_t owner_allocate_page(OwnerView owner) {
  const std::uint32_t page = atomicAdd(owner.page_alloc, 1u);
  if (page < owner.page_capacity)
    return page;
  atomicExch(owner.error, 1u);
  return kOwnerNoPage;
}

__device__ inline bool owner_update_in_page(
    std::uint64_t *page, std::uint32_t low, std::uint32_t value,
    std::uint64_t *previous) {
  std::uint32_t slot = owner_slot(low);
  for (int probe = 0; probe < kOwnerSlots; ++probe) {
    const std::uint64_t packed = page[slot];
    const std::uint32_t state = owner_state(packed);
    if (state == kOwnerEmpty)
      return false;
    if (owner_low(packed) == low) {
      *previous = packed;
      page[slot] = owner_pack(low, value, kOwnerLive);
      return true;
    }
    slot = (slot + 1u) & kOwnerSlotMask;
  }
  return false;
}

__device__ inline std::uint64_t owner_upsert(
    OwnerView owner, std::uint32_t key, std::uint32_t value) {
  const std::uint32_t quotient = key >> kEpochQuotientBits;
  const std::uint32_t low = key & 0xffffu;
  std::uint64_t previous = 0u;
  std::uint64_t *primary =
      owner.primary + static_cast<std::size_t>(quotient) * kOwnerSlots;
  if (owner.heads[quotient] == kOwnerNoPage) {
    if (owner_insert_in_page(primary, low, value, &previous))
      return previous;
    const std::uint32_t index = owner_allocate_page(owner);
    if (index == kOwnerNoPage)
      return 0u;
    owner.heads[quotient] = index;
    std::uint64_t *page =
        owner.overflow +
        static_cast<std::size_t>(index) * kOwnerSlots;
    owner_insert_in_page(page, low, value, &previous);
    return previous;
  }
  if (owner_update_in_page(primary, low, value, &previous))
    return previous;
  std::uint32_t page_index = owner.heads[quotient];
  while (page_index != kOwnerNoPage) {
    std::uint64_t *page =
        owner.overflow +
        static_cast<std::size_t>(page_index) * kOwnerSlots;
    if (owner_update_in_page(page, low, value, &previous))
      return previous;
    page_index = owner.next[page_index];
  }
  if (owner_insert_in_page(primary, low, value, &previous))
    return previous;
  std::uint32_t *link = owner.heads + quotient;
  while (*link != kOwnerNoPage) {
    const std::uint32_t index = *link;
    std::uint64_t *page =
        owner.overflow +
        static_cast<std::size_t>(index) * kOwnerSlots;
    if (owner_insert_in_page(page, low, value, &previous))
      return previous;
    link = owner.next + index;
  }
  const std::uint32_t index = owner_allocate_page(owner);
  if (index == kOwnerNoPage)
    return 0u;
  *link = index;
  std::uint64_t *page =
      owner.overflow +
      static_cast<std::size_t>(index) * kOwnerSlots;
  owner_insert_in_page(page, low, value, &previous);
  return previous;
}

__device__ inline std::uint64_t owner_erase(
    OwnerView owner, std::uint32_t key) {
  const std::uint32_t quotient = key >> kEpochQuotientBits;
  const std::uint32_t low = key & 0xffffu;
  std::uint32_t page_index = kOwnerNoPage;
  std::uint64_t *page =
      owner.primary + static_cast<std::size_t>(quotient) * kOwnerSlots;
  for (;;) {
    std::uint32_t slot = owner_slot(low);
    for (int probe = 0; probe < kOwnerSlots; ++probe) {
      const std::uint64_t packed = page[slot];
      const std::uint32_t state = owner_state(packed);
      if (state == kOwnerEmpty)
        return 0u;
      if (owner_low(packed) == low) {
        if (state == kOwnerLive)
          page[slot] = owner_pack(low, 0u, kOwnerTomb);
        return packed;
      }
      slot = (slot + 1u) & kOwnerSlotMask;
    }
    page_index = page_index == kOwnerNoPage
                     ? owner.heads[quotient]
                     : owner.next[page_index];
    if (page_index == kOwnerNoPage)
      return 0u;
    page = owner.overflow +
           static_cast<std::size_t>(page_index) * kOwnerSlots;
  }
}
__global__ void owner_build_kernel(
    OwnerView owner, const std::uint32_t *keys,
    const std::uint32_t *values, const std::uint32_t *radix) {
  const std::uint32_t quotient = blockIdx.x;
  if (quotient >= kEpochQuotients || threadIdx.x != 0)
    return;
  constexpr int shift = kSpineRadixBits - kEpochQuotientBits;
  const std::uint32_t begin = radix[quotient << shift];
  const std::uint32_t end = radix[(quotient + 1u) << shift];
  owner.quotient_live[quotient] = end - begin;
  for (std::uint32_t position = begin; position < end; ++position)
    owner_upsert(owner, keys[position], values[position]);
}

__device__ inline bool owner_atomic_upsert_primary(
    OwnerView owner, std::uint32_t key, std::uint32_t value,
    std::uint64_t *previous) {
  const std::uint32_t quotient = key >> kEpochQuotientBits;
  const std::uint32_t low = key & 0xffffu;
  std::uint64_t *page =
      owner.primary + static_cast<std::size_t>(quotient) * kOwnerSlots;
  for (;;) {
    std::uint32_t slot = owner_slot(low);
    std::uint32_t reusable = kOwnerNoPage;
    std::uint64_t reusable_value = 0u;
    for (int probe = 0; probe < kOwnerSlots; ++probe) {
      const std::uint64_t packed =
          *reinterpret_cast<volatile std::uint64_t *>(page + slot);
      const std::uint32_t state = owner_state(packed);
      if (state == kOwnerEmpty) {
        const std::uint32_t target =
            reusable == kOwnerNoPage ? slot : reusable;
        const std::uint64_t expected =
            reusable == kOwnerNoPage ? packed : reusable_value;
        auto *atomic_slot =
            reinterpret_cast<unsigned long long *>(page + target);
        const std::uint64_t old = atomicCAS(
            atomic_slot, expected,
            owner_pack(low, value, kOwnerLive));
        if (old == expected) {
          *previous = old;
          return true;
        }
        break;
      }
      if (owner_low(packed) == low) {
        auto *atomic_slot =
            reinterpret_cast<unsigned long long *>(page + slot);
        const std::uint64_t old = atomicCAS(
            atomic_slot, packed,
            owner_pack(low, value, kOwnerLive));
        if (old == packed) {
          *previous = old;
          return true;
        }
        break;
      }
      if (state == kOwnerTomb && reusable == kOwnerNoPage) {
        reusable = slot;
        reusable_value = packed;
      }
      slot = (slot + 1u) & kOwnerSlotMask;
    }
    if (reusable != kOwnerNoPage) {
      auto *atomic_slot =
          reinterpret_cast<unsigned long long *>(page + reusable);
      const std::uint64_t old = atomicCAS(
          atomic_slot, reusable_value,
          owner_pack(low, value, kOwnerLive));
      if (old == reusable_value) {
        *previous = old;
        return true;
      }
      continue;
    }
    return false;
  }
}

__device__ inline std::uint64_t owner_atomic_erase_primary(
    OwnerView owner, std::uint32_t key) {
  const std::uint32_t quotient = key >> kEpochQuotientBits;
  const std::uint32_t low = key & 0xffffu;
  std::uint64_t *page =
      owner.primary + static_cast<std::size_t>(quotient) * kOwnerSlots;
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
      auto *atomic_slot =
          reinterpret_cast<unsigned long long *>(page + slot);
      return atomicExch(
          atomic_slot, owner_pack(low, 0u, kOwnerTomb));
    }
    slot = (slot + 1u) & kOwnerSlotMask;
  }
  return 0u;
}

__device__ inline void owner_apply_transition(
    OwnerView owner, std::uint32_t key, std::uint8_t op,
    std::uint32_t input_value, bool atomic_primary,
    std::uint32_t *value_delta, std::int8_t *count_delta) {
  std::uint64_t previous = 0u;
  if (op == kInsert) {
    const bool applied =
        !atomic_primary ||
        owner_atomic_upsert_primary(
            owner, key, input_value, &previous);
    if (!applied) {
      *count_delta = 127;
      return;
    }
    if (!atomic_primary)
      previous = owner_upsert(owner, key, input_value);
    if (owner_state(previous) == kOwnerLive) {
      *value_delta = input_value - owner_value(previous);
      *count_delta = 0;
    } else {
      *value_delta = input_value;
      *count_delta = 1;
    }
    return;
  }
  previous = atomic_primary
                 ? owner_atomic_erase_primary(owner, key)
                 : owner_erase(owner, key);
  if (owner_state(previous) == kOwnerLive) {
    *value_delta = 0u - owner_value(previous);
    *count_delta = -1;
  } else {
    *value_delta = 0u;
    *count_delta = 0;
  }
}
__global__ void owner_transition_kernel(
    OwnerView owner, const std::uint32_t *keys,
    std::uint32_t *values, const std::uint8_t *ops,
    std::uint8_t constant_op, const std::uint32_t *offsets,
    std::int8_t *count_delta,
    std::uint32_t *quotient_count_sum) {
  const std::uint32_t quotient = blockIdx.x;
  if (quotient >= kEpochQuotients)
    return;
  const std::uint32_t begin = offsets[quotient];
  const std::uint32_t end = offsets[quotient + 1u];
  const int lane = threadIdx.x & 31;
  const bool serial = owner.heads[quotient] != kOwnerNoPage ||
                      end - begin > 32u;
  if (serial) {
    if (lane != 0)
      return;
    std::int32_t count_sum = 0;
    for (std::uint32_t position = begin;
         position < end; ++position) {
      const std::uint8_t op =
          ops ? ops[position] : constant_op;
      std::uint32_t value_delta = 0u;
      std::int8_t delta = 0;
      owner_apply_transition(
          owner, keys[position], op, values[position], false,
          &value_delta, &delta);
      values[position] = value_delta;
      count_delta[position] = delta;
      count_sum += static_cast<std::int32_t>(delta);
    }
    quotient_count_sum[quotient] =
        static_cast<std::uint32_t>(count_sum);
    if (count_sum != 0)
      owner_add_live(owner, quotient, count_sum);
    return;
  }
  for (std::uint32_t position = begin + lane;
       position < end; position += 32u) {
    const std::uint32_t key = keys[position];
    const unsigned active = __activemask();
    const unsigned peers = __match_any_sync(active, key);
    const bool final_occurrence =
        lane == 31 - __clz(peers);
    if (!final_occurrence) {
      values[position] = 0u;
      count_delta[position] = 0;
      continue;
    }
    const std::uint8_t op =
        ops ? ops[position] : constant_op;
    std::uint32_t value_delta = 0u;
    std::int8_t delta = 0;
    owner_apply_transition(
        owner, key, op, values[position], true,
        &value_delta, &delta);
    if (delta != 127)
      values[position] = value_delta;
    count_delta[position] = delta;
  }
  __syncwarp();
  if (lane == 0) {
    for (std::uint32_t position = begin;
         position < end; ++position) {
      if (count_delta[position] != 127)
        continue;
      const std::uint8_t op =
          ops ? ops[position] : constant_op;
      std::uint32_t value_delta = 0u;
      std::int8_t delta = 0;
      owner_apply_transition(
          owner, keys[position], op, values[position], false,
          &value_delta, &delta);
      values[position] = value_delta;
      count_delta[position] = delta;
    }
  }
  __syncwarp();
  std::int32_t local_sum = 0;
  for (std::uint32_t position = begin + lane;
       position < end; position += 32u) {
    local_sum += static_cast<std::int32_t>(
        count_delta[position]);
  }
  for (int offset = 16; offset > 0; offset >>= 1)
    local_sum += __shfl_down_sync(0xffffffffu, local_sum, offset);
  if (lane == 0) {
    quotient_count_sum[quotient] =
        static_cast<std::uint32_t>(local_sum);
    if (local_sum != 0)
      owner_add_live(owner, quotient, local_sum);
  }
}

__global__ void owner_lookup_kernel(
    OwnerView owner, const std::uint32_t *queries, std::size_t count,
    std::uint32_t *out_values, std::uint8_t *out_found) {
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

__device__ inline bool owner_key_live(
    OwnerView owner, std::uint32_t key) {
  std::uint64_t packed = 0u;
  return owner_find(owner, key, &packed) &&
         owner_state(packed) == kOwnerLive;
}

__device__ inline std::uint32_t epoch_subgroup_mask(
    const EpochView &ev, std::uint32_t quotient, std::uint32_t subgroup,
    std::uint32_t count) {
  const std::uint32_t valid =
      count == 32u ? 0xffffffffu : ((1u << count) - 1u);
  const std::uint32_t *planes =
      ev.subgroup_masks + quotient * kEpochSubgroupPlanes;
  std::uint32_t mask = valid;
#pragma unroll
  for (int bit = 0; bit < kEpochSubgroupBits; ++bit)
    mask &= ((subgroup >> bit) & 1u) ? planes[bit] : ~planes[bit];
  return mask;
}

__device__ inline bool epoch_point_position_bounds(
    const EpochView &ev, std::uint32_t key, std::uint32_t quotient,
    std::uint32_t begin, std::uint32_t end, std::uint32_t *position) {
  const std::uint32_t count = end - begin;
  if (count <= 32u) {
    const std::uint32_t subgroup =
        (key >> (kEpochQuotientBits - kEpochSubgroupBits)) &
        (kEpochSubgroups - 1);
    std::uint32_t mask =
        epoch_subgroup_mask(ev, quotient, subgroup, count);
    while (mask != 0u) {
      const std::uint32_t bit = __ffs(mask) - 1;
      const std::uint32_t p = begin + bit;
      if (ev.keys[p] == key) {
        *position = p;
        return true;
      }
      mask &= mask - 1;
    }
    return false;
  }
  if (ev.fully_sorted || count <= kEpochHeavySortCap) {
    const std::uint32_t p = begin + static_cast<std::uint32_t>(
                                        lower_bound_u32(ev.keys + begin,
                                                        end - begin, key));
    if (p < end && ev.keys[p] == key) {
      *position = p;
      return true;
    }
    return false;
  }
  for (std::uint32_t p = begin; p < end; ++p) {
    if (ev.keys[p] == key) {
      *position = p;
      return true;
    }
  }
  return false;
}

__device__ inline bool epoch_point_position(
    const EpochView &ev, std::uint32_t key, std::uint32_t *position) {
  const std::uint32_t quotient = key >> kEpochQuotientBits;
  const std::uint32_t begin = ev.quotient_off[quotient];
  const std::uint32_t end = ev.quotient_off[quotient + 1];
  return epoch_point_position_bounds(ev, key, quotient, begin, end,
                                     position);
}

__device__ inline bool epoch_point_find(const EpochView &ev,
                                        std::uint32_t key,
                                        std::uint32_t *out_value) {
  std::uint32_t position;
  if (!epoch_point_position(ev, key, &position))
    return false;
  if (out_value)
    *out_value = ev.values[position];
  return true;
}

__device__ inline std::uint32_t epoch_quotient_count(
    const EpochView &ev, std::uint32_t quotient, std::uint32_t lo,
    std::uint32_t hi) {
  const std::uint32_t begin = ev.quotient_off[quotient];
  const std::uint32_t end = ev.quotient_off[quotient + 1];
  std::int32_t count = 0;
  for (std::uint32_t p = begin; p < end; ++p) {
    if (ev.keys[p] >= lo && ev.keys[p] <= hi)
      count += static_cast<std::int32_t>(ev.count_delta[p]);
  }
  return static_cast<std::uint32_t>(count);
}

__device__ inline std::uint32_t epoch_quotient_sum(
    const EpochView &ev, std::uint32_t quotient, std::uint32_t lo,
    std::uint32_t hi) {
  const std::uint32_t begin = ev.quotient_off[quotient];
  const std::uint32_t end = ev.quotient_off[quotient + 1];
  std::uint32_t sum = 0u;
  const std::uint32_t physical_count = end - begin;
  if (physical_count > 32u &&
      (ev.fully_sorted || physical_count <= kEpochHeavySortCap)) {
    const std::uint32_t lb = begin + static_cast<std::uint32_t>(
                                         lower_bound_u32(ev.keys + begin,
                                                         end - begin, lo));
    const std::uint32_t ub = begin + static_cast<std::uint32_t>(
                                         upper_bound_u32(ev.keys + begin,
                                                         end - begin, hi));
    for (std::uint32_t p = lb; p < ub; ++p)
      sum += ev.values[p];
    return sum;
  }
  for (std::uint32_t p = begin; p < end; ++p)
    if (ev.keys[p] >= lo && ev.keys[p] <= hi)
      sum += ev.values[p];
  return sum;
}

__device__ inline std::uint32_t epoch_subgroup_edge_sum(
    const EpochView &ev, std::uint32_t quotient, std::uint32_t subgroup,
    std::uint32_t lo, std::uint32_t hi) {
  const std::uint32_t begin = ev.quotient_off[quotient];
  const std::uint32_t count =
      ev.quotient_off[quotient + 1] - begin;
  std::uint32_t mask =
      epoch_subgroup_mask(ev, quotient, subgroup, count);
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

__device__ inline std::uint32_t epoch_indexed_quotient_sum(
    const EpochView &ev, std::uint32_t quotient, std::uint32_t lo,
    std::uint32_t hi) {
  const std::uint32_t begin = ev.quotient_off[quotient];
  const std::uint32_t count =
      ev.quotient_off[quotient + 1] - begin;
  if (!ev.subgroup_value_prefix || count > 32u)
    return epoch_quotient_sum(ev, quotient, lo, hi);
  const std::uint32_t first =
      (lo >> (kEpochQuotientBits - kEpochSubgroupBits)) &
      (kEpochSubgroups - 1u);
  const std::uint32_t last =
      (hi >> (kEpochQuotientBits - kEpochSubgroupBits)) &
      (kEpochSubgroups - 1u);
  if (first == last)
    return epoch_subgroup_edge_sum(ev, quotient, first, lo, hi);
  std::uint32_t sum =
      epoch_subgroup_edge_sum(ev, quotient, first, lo, hi);
  sum += epoch_subgroup_edge_sum(ev, quotient, last, lo, hi);
  if (last > first + 1u) {
    const std::uint32_t *prefix =
        ev.subgroup_value_prefix +
        quotient * kEpochSubgroupPrefixStride;
    sum += prefix[last] - prefix[first + 1u];
  }
  return sum;
}

__device__ inline std::uint32_t epoch_range_count_one(
    const EpochView &ev, std::uint32_t lo, std::uint32_t hi) {
  const std::uint32_t first = lo >> kEpochQuotientBits;
  const std::uint32_t last = hi >> kEpochQuotientBits;
  if (first == last)
    return epoch_quotient_count(ev, first, lo, hi);
  std::uint32_t count = epoch_quotient_count(ev, first, lo, 0xffffffffu);
  count += epoch_quotient_count(ev, last, 0u, hi);
  if (last > first + 1) {
    const std::uint32_t *prefix = ev.quotient_count_prefix;
    count += prefix[last] - prefix[first + 1];
  }
  return count;
}

__device__ inline std::uint32_t epoch_range_sum_one(
    const EpochView &ev, std::uint32_t lo, std::uint32_t hi) {
  const std::uint32_t first = lo >> kEpochQuotientBits;
  const std::uint32_t last = hi >> kEpochQuotientBits;
  if (first == last)
    return epoch_indexed_quotient_sum(ev, first, lo, hi);
  std::uint32_t sum =
      epoch_indexed_quotient_sum(ev, first, lo, 0xffffffffu);
  sum += epoch_indexed_quotient_sum(ev, last, 0u, hi);
  if (last > first + 1)
    sum += ev.quotient_value_prefix[last] -
           ev.quotient_value_prefix[first + 1];
  return sum;
}

__device__ inline std::uint32_t epoch_quotient_successor(
    const EpochView &ev, std::uint32_t quotient, std::uint32_t floor) {
  const std::uint32_t begin = ev.quotient_off[quotient];
  const std::uint32_t end = ev.quotient_off[quotient + 1];
  const std::uint32_t physical_count = end - begin;
  if (physical_count > 32u &&
      (ev.fully_sorted || physical_count <= kEpochHeavySortCap)) {
    const std::uint32_t p = begin + static_cast<std::uint32_t>(
                                        lower_bound_u32(ev.keys + begin,
                                                        end - begin, floor));
    return p < end ? ev.keys[p] : kEmptyKey;
  }
  std::uint32_t best = kEmptyKey;
  for (std::uint32_t p = begin; p < end; ++p) {
    const std::uint32_t key = ev.keys[p];
    if (key >= floor && key < best)
      best = key;
  }
  return best;
}

__device__ inline std::uint32_t epoch_successor_candidate(
    const EpochView &ev, std::uint32_t key) {
  std::uint32_t quotient = key >> kEpochQuotientBits;
  std::uint32_t candidate = epoch_quotient_successor(ev, quotient, key);
  if (candidate != kEmptyKey)
    return candidate;
  ++quotient;
  while (quotient < kEpochQuotients) {
    const std::uint32_t word = quotient >> 5;
    const std::uint32_t bit = quotient & 31u;
    std::uint32_t mask = ev.quotient_bitmap[word] & (0xffffffffu << bit);
    if (mask == 0u) {
      quotient = (word + 1u) * 32u;
      continue;
    }
    quotient = word * 32u + static_cast<std::uint32_t>(__ffs(mask) - 1);
    candidate = epoch_quotient_successor(ev, quotient, 0u);
    if (candidate != kEmptyKey)
      return candidate;
    ++quotient;
  }
  return kEmptyKey;
}

__device__ inline void spine_refined_bounds(
    const SpineView &spine, std::uint32_t key, std::uint32_t bin,
    std::size_t parent_begin, std::size_t parent_end,
    std::size_t *begin, std::size_t *end) {
  *begin = parent_begin;
  *end = parent_end;
  if (!spine.micro_bits)
    return;
  const std::uint32_t bits = spine.micro_bits[bin];
  if (bits == 0u)
    return;
  const std::uint32_t slots = 1u << bits;
  const std::uint32_t low = key & ((1u << kSpineRadixShift) - 1u);
  const std::uint32_t slot = low >> (kSpineRadixShift - bits);
  const std::uint32_t pool = spine.micro_base[bin];
  const std::size_t count = parent_end - parent_begin;
  const std::size_t local_begin = spine.micro_offsets[pool + slot];
  const std::size_t local_end =
      slot + 1u < slots ? spine.micro_offsets[pool + slot + 1u] : count;
  *begin = parent_begin + local_begin;
  *end = parent_begin + local_end;
}

__device__ inline void spine_search_bounds(
    const SpineView &spine, std::uint32_t key,
    std::size_t *begin, std::size_t *end) {
  const std::uint32_t bin = key >> kSpineRadixShift;
  const std::size_t parent_begin = spine.radix[bin];
  const std::size_t parent_end = spine.radix[bin + 1];
  spine_refined_bounds(spine, key, bin, parent_begin,
                       parent_end, begin, end);
}

__device__ inline std::size_t spine_lower_rank(
    const SpineView &spine, std::uint32_t key) {
  std::size_t begin = 0, end = 0;
  spine_search_bounds(spine, key, &begin, &end);
  return begin + lower_bound_u32(spine.keys + begin, end - begin, key);
}

__device__ inline std::size_t spine_upper_rank(
    const SpineView &spine, std::uint32_t key) {
  std::size_t begin = 0, end = 0;
  spine_search_bounds(spine, key, &begin, &end);
  return begin + upper_bound_u32(spine.keys + begin, end - begin, key);
}

__device__ inline void spine_range_ranks(
    const SpineView &spine, std::uint32_t lo, std::uint32_t hi,
    std::size_t *lower, std::size_t *upper) {
  const std::uint32_t lo_bin = lo >> kSpineRadixShift;
  const std::uint32_t hi_bin = hi >> kSpineRadixShift;
  std::size_t lo_begin = 0, lo_end = 0;
  std::size_t hi_begin = 0, hi_end = 0;
  if (lo_bin == hi_bin) {
    const std::size_t parent_begin = spine.radix[lo_bin];
    const std::size_t parent_end = spine.radix[lo_bin + 1];
    spine_refined_bounds(spine, lo, lo_bin, parent_begin,
                         parent_end, &lo_begin, &lo_end);
    spine_refined_bounds(spine, hi, hi_bin, parent_begin,
                         parent_end, &hi_begin, &hi_end);
  } else {
    spine_search_bounds(spine, lo, &lo_begin, &lo_end);
    spine_search_bounds(spine, hi, &hi_begin, &hi_end);
  }
  *lower = lo_begin +
           lower_bound_u32(spine.keys + lo_begin, lo_end - lo_begin, lo);
  *upper = hi_begin +
           upper_bound_u32(spine.keys + hi_begin, hi_end - hi_begin, hi);
}

__device__ inline std::uint32_t spine_range_cdf_prefix(
    const SpineRangeView &range, std::uint32_t key, bool upper) {
  if (key < range.min_key)
    return 0u;
  std::uint64_t index = static_cast<std::uint64_t>(key) -
                        range.min_key + static_cast<unsigned>(upper);
  if (index > range.span)
    index = range.span;
  return range.cdf[index];
}

__device__ inline bool spine_point_find(
    const SpineView &spine, std::uint32_t key,
    std::uint32_t *out_value) {
  if (spine.count == 0)
    return false;
  const std::size_t position = spine_lower_rank(spine, key);
  if (position >= spine.count || spine.keys[position] != key)
    return false;
  if (out_value)
    *out_value = spine.values[position];
  return true;
}

__device__ inline std::uint32_t spine_range_count(
    const SpineView &spine, std::uint32_t lo, std::uint32_t hi) {
  if (spine.count == 0)
    return 0u;
  std::size_t begin = 0, end = 0;
  spine_range_ranks(spine, lo, hi, &begin, &end);
  return static_cast<std::uint32_t>(end - begin);
}

__device__ inline std::uint32_t spine_range_sum(
    const SpineView &spine, const SpineRangeView &range,
    const std::uint32_t *value_prefix, std::uint32_t lo,
    std::uint32_t hi) {
  if (spine.count == 0)
    return 0u;
  if (range.cdf) {
    return spine_range_cdf_prefix(range, hi, true) -
           spine_range_cdf_prefix(range, lo, false);
  }
  std::size_t begin = 0, end = 0;
  spine_range_ranks(spine, lo, hi, &begin, &end);
  return value_prefix[end] - value_prefix[begin];
}

__device__ inline std::uint32_t spine_successor_candidate(
    const SpineView &spine, std::uint32_t key) {
  if (spine.count == 0)
    return kEmptyKey;
  const std::size_t position = spine_lower_rank(spine, key);
  return position < spine.count ? spine.keys[position] : kEmptyKey;
}

__device__ inline std::uint32_t spine_owner_successor(
    const SpineView &spine, std::uint32_t key, OwnerView owner) {
  std::uint32_t floor = key;
  for (;;) {
    const std::uint32_t candidate =
        spine_successor_candidate(spine, floor);
    if (candidate == kEmptyKey || owner_key_live(owner, candidate))
      return candidate;
    if (candidate == 0xffffffffu)
      return kEmptyKey;
    floor = candidate + 1u;
  }
}

__device__ inline std::uint32_t epoch_owner_successor(
    const EpochView &epoch, std::uint32_t key, OwnerView owner) {
  std::uint32_t floor = key;
  for (;;) {
    const std::uint32_t candidate =
        epoch_successor_candidate(epoch, floor);
    if (candidate == kEmptyKey || owner_key_live(owner, candidate))
      return candidate;
    if (candidate == 0xffffffffu)
      return kEmptyKey;
    floor = candidate + 1u;
  }
}

__global__ void spine_build_radix_kernel(
    const std::uint32_t *keys, std::size_t count,
    std::uint32_t *radix) {
  const std::size_t bin = blockIdx.x * blockDim.x + threadIdx.x;
  if (bin > kSpineRadixSize)
    return;
  if (bin == kSpineRadixSize) {
    radix[bin] = static_cast<std::uint32_t>(count);
    return;
  }
  const std::uint32_t key =
      static_cast<std::uint32_t>(bin << kSpineRadixShift);
  radix[bin] = static_cast<std::uint32_t>(
      lower_bound_u32(keys, count, key));
}

__global__ void spine_micro_plan_kernel(
    const std::uint32_t *radix, std::uint8_t *micro_bits,
    std::uint32_t *micro_base) {
  const std::uint32_t bin = blockIdx.x * blockDim.x + threadIdx.x;
  if (bin >= kSpineRadixSize)
    return;
  const std::uint32_t count = radix[bin + 1] - radix[bin];
  if (count <= kSpineMicroTarget) {
    micro_bits[bin] = 0u;
    micro_base[bin] = 0u;
    return;
  }
  const std::uint32_t needed =
      (count + kSpineMicroTarget - 1u) / kSpineMicroTarget;
  std::uint32_t slots = 1u;
  std::uint8_t bits = 0u;
  while (slots < needed) {
    slots <<= 1;
    ++bits;
  }
  micro_bits[bin] = bits;
  micro_base[bin] = slots;
}

__global__ void spine_micro_fill_kernel(
    const std::uint32_t *keys, const std::uint32_t *radix,
    const std::uint8_t *micro_bits, const std::uint32_t *micro_base,
    std::uint16_t *micro_offsets) {
  const std::uint32_t bin = blockIdx.x * blockDim.x + threadIdx.x;
  if (bin >= kSpineRadixSize)
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
        keys[position] & ((1u << kSpineRadixShift) - 1u);
    const std::uint32_t slot = low >> (kSpineRadixShift - bits);
    while (next_slot <= slot)
      micro_offsets[pool + next_slot++] =
          static_cast<std::uint16_t>(position - begin);
  }
  while (next_slot < slots)
    micro_offsets[pool + next_slot++] =
        static_cast<std::uint16_t>(end - begin);
}
__global__ void spine_range_cdf_scatter_kernel(
    const std::uint32_t *keys, const std::uint32_t *values,
    std::size_t count, std::uint32_t min_key,
    std::uint32_t *cdf) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::uint64_t slot =
      static_cast<std::uint64_t>(keys[i]) - min_key + 1u;
  cdf[slot] = values[i];
}

__device__ inline std::uint32_t range_delta_edge_sum(
    const RangeDeltaView &projection, std::uint32_t bin,
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

__device__ inline std::uint32_t range_delta_sum(
    const RangeDeltaView &projection, std::uint32_t lo,
    std::uint32_t hi) {
  const std::uint32_t first = lo >> kRangeProjectionShift;
  const std::uint32_t last = hi >> kRangeProjectionShift;
  if (first == last)
    return range_delta_edge_sum(projection, first, lo, hi);
  std::uint32_t sum =
      range_delta_edge_sum(projection, first, lo, 0xffffffffu);
  sum += range_delta_edge_sum(projection, last, 0u, hi);
  if (last > first + 1u) {
    sum += projection.value_prefix[last] -
           projection.value_prefix[first + 1u];
  }
  return sum;
}

__device__ inline void range_delta_accumulate_warp(
    const std::uint32_t *keys, const std::uint32_t *values,
    std::uint32_t begin, std::uint32_t end,
    std::uint32_t *counts, std::uint32_t *sums) {
  const unsigned full = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  for (std::uint32_t base = begin; base < end; base += 32u) {
    const std::uint32_t position = base + lane;
    const bool included = position < end;
    const unsigned active = __ballot_sync(full, included);
    if (included) {
      const std::uint32_t subgroup =
          (keys[position] >> kRangeProjectionShift) &
          (kEpochSubgroups - 1u);
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

__device__ inline void range_delta_pack_warp(
    const std::uint32_t *keys, const std::uint32_t *values,
    std::uint32_t begin, std::uint32_t end,
    std::uint32_t *cursors, std::uint32_t *out_keys,
    std::uint32_t *out_values) {
  const unsigned full = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  for (std::uint32_t base = begin; base < end; base += 32u) {
    const std::uint32_t position = base + lane;
    const bool included = position < end;
    const unsigned active = __ballot_sync(full, included);
    if (included) {
      const std::uint32_t key = keys[position];
      const std::uint32_t subgroup =
          (key >> kRangeProjectionShift) &
          (kEpochSubgroups - 1u);
      const unsigned peers = __match_any_sync(active, subgroup);
      const int leader = __ffs(peers) - 1;
      std::uint32_t output = 0u;
      if (lane == leader) {
        output = cursors[subgroup];
        cursors[subgroup] += __popc(peers);
      }
      output = __shfl_sync(peers, output, leader);
      const unsigned lower =
          lane == 0 ? 0u : (1u << lane) - 1u;
      output += __popc(peers & lower);
      out_keys[output] = key;
      out_values[output] = values[position];
    }
    __syncwarp(full);
  }
}

__global__ void range_delta_plan_kernel(
    const EpochView *epochs, int epoch_count,
    std::uint32_t *bin_counts, std::uint32_t *bin_sums) {
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
    const EpochView epoch = epochs[e];
    const std::uint32_t begin = epoch.quotient_off[quotient];
    const std::uint32_t end = epoch.quotient_off[quotient + 1u];
    range_delta_accumulate_warp(
        epoch.keys, epoch.values, begin, end,
        counts[warp], sums[warp]);
  }
  __syncwarp();
  if (lane < kEpochSubgroups) {
    const std::uint32_t bin =
        quotient * kEpochSubgroups + lane;
    bin_counts[bin] = counts[warp][lane];
    bin_sums[bin] = sums[warp][lane];
  }
}

__global__ void range_delta_pack_kernel(
    const EpochView *epochs, int epoch_count,
    const std::uint32_t *bin_offsets,
    std::uint32_t *out_keys, std::uint32_t *out_values) {
  constexpr int warps = 8;
  __shared__ std::uint32_t cursors[warps][kEpochSubgroups];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const std::uint32_t quotient =
      blockIdx.x * warps + static_cast<std::uint32_t>(warp);
  if (quotient >= kEpochQuotients)
    return;
  if (lane < kEpochSubgroups) {
    const std::uint32_t bin =
        quotient * kEpochSubgroups + lane;
    cursors[warp][lane] = bin_offsets[bin];
  }
  __syncwarp();
  for (int e = 0; e < epoch_count; ++e) {
    const EpochView epoch = epochs[e];
    const std::uint32_t begin = epoch.quotient_off[quotient];
    const std::uint32_t end = epoch.quotient_off[quotient + 1u];
    range_delta_pack_warp(
        epoch.keys, epoch.values, begin, end,
        cursors[warp], out_keys, out_values);
  }
}

__global__ void epoch_quotient_metadata_kernel(
    const std::uint32_t *keys, std::uint32_t record_count,
    std::uint32_t *offsets, std::uint32_t *subgroup_masks,
    std::uint32_t *quotient_live) {
  constexpr unsigned full = 0xffffffffu;
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t warp_begin = i & ~31u;
  const int lane = threadIdx.x & 31;
  const bool valid = i < record_count;
  const std::uint32_t key = valid ? keys[i] : 0u;
  const std::uint32_t quotient = key >> kEpochQuotientBits;
  std::uint32_t previous = __shfl_up_sync(full, quotient, 1);
  if (lane == 0 && valid)
    previous = i == 0u ? quotient
                       : keys[i - 1] >> kEpochQuotientBits;
  const bool starts = valid &&
                      (i == 0u || quotient != previous);
  unsigned starts_mask = __ballot_sync(full, starts);
  while (starts_mask != 0u) {
    const int start_lane = __ffs(starts_mask) - 1;
    const std::uint32_t start = warp_begin + start_lane;
    const std::uint32_t segment_quotient =
        __shfl_sync(full, quotient, start_lane);
    const std::uint32_t segment_previous =
        __shfl_sync(full, previous, start_lane);
    const unsigned later = start_lane == 31
                               ? 0u
                               : starts_mask &
                                     (0xffffffffu << (start_lane + 1));
    const int end_lane = later == 0u ? 32 : __ffs(later) - 1;
    std::uint32_t end = warp_begin + end_lane;
    const std::uint32_t warp_end =
        min(warp_begin + 32u, record_count);
    if (later == 0u) {
      if (lane == 0) {
        end = warp_end;
        while (end < record_count &&
               (keys[end] >> kEpochQuotientBits) == segment_quotient)
          ++end;
      }
      end = __shfl_sync(full, end, 0);
    }
    const std::uint32_t count = end - start;
    std::uint32_t tail_planes[kEpochSubgroupPlanes] = {};
    if (lane == 0 && count <= 32u && end > warp_end) {
      for (std::uint32_t position = warp_end; position < end; ++position) {
        const std::uint32_t subgroup =
            (keys[position] >>
             (kEpochQuotientBits - kEpochSubgroupBits)) &
            (kEpochSubgroups - 1u);
        const std::uint32_t bit_position = 1u << (position - start);
#pragma unroll
        for (int bit = 0; bit < kEpochSubgroupPlanes; ++bit)
          tail_planes[bit] |=
              (0u - ((subgroup >> bit) & 1u)) & bit_position;
      }
    }
    if (count <= 32u) {
      const unsigned low = 0xffffffffu << start_lane;
      const unsigned high = end_lane == 32
                                ? 0xffffffffu
                                : (1u << end_lane) - 1u;
      const unsigned segment = low & high;
      const std::uint32_t subgroup =
          (key >> (kEpochQuotientBits - kEpochSubgroupBits)) &
          (kEpochSubgroups - 1u);
      const std::uint32_t base =
          segment_quotient * kEpochSubgroupPlanes;
#pragma unroll
      for (int bit = 0; bit < kEpochSubgroupPlanes; ++bit) {
        const unsigned local = __ballot_sync(
            full, valid && ((subgroup >> bit) & 1u));
        const std::uint32_t tail =
            __shfl_sync(full, tail_planes[bit], 0);
        if (lane == 0)
          subgroup_masks[base + bit] =
              ((local & segment) >> start_lane) | tail;
      }
    }
    if (lane == 0) {
      if (start == 0u) {
        for (std::uint32_t q = 0; q <= segment_quotient; ++q)
          offsets[q] = 0u;
        for (std::uint32_t q = 0; q < segment_quotient; ++q)
          quotient_live[q] = 0u;
      } else {
        for (std::uint32_t q = segment_previous + 1u;
             q <= segment_quotient; ++q)
          offsets[q] = start;
        for (std::uint32_t q = segment_previous + 1u;
             q < segment_quotient; ++q)
          quotient_live[q] = 0u;
      }
      if (end == record_count) {
        for (std::uint32_t q = segment_quotient + 1u;
             q <= kEpochQuotients; ++q)
          offsets[q] = record_count;
        for (std::uint32_t q = segment_quotient + 1u;
             q < kEpochQuotients; ++q)
          quotient_live[q] = 0u;
      }
      quotient_live[segment_quotient] = count;
    }
    starts_mask &= starts_mask - 1u;
  }
}

__global__ void epoch_subgroup_value_prefix_kernel(
    EpochView ev, std::uint32_t *subgroup_prefix) {
  const std::uint32_t quotient = blockIdx.x * blockDim.x + threadIdx.x;
  if (quotient >= kEpochQuotients)
    return;
  const std::uint32_t begin = ev.quotient_off[quotient];
  const std::uint32_t end = ev.quotient_off[quotient + 1];
  std::uint32_t sums[kEpochSubgroups] = {};
  for (std::uint32_t position = begin; position < end; ++position) {
    const std::uint32_t subgroup =
        (ev.keys[position] >>
         (kEpochQuotientBits - kEpochSubgroupBits)) &
        (kEpochSubgroups - 1u);
    sums[subgroup] += ev.values[position];
  }
  const std::uint32_t base =
      quotient * kEpochSubgroupPrefixStride;
  std::uint32_t prefix = 0u;
#pragma unroll
  for (int subgroup = 0; subgroup < kEpochSubgroups; ++subgroup) {
    subgroup_prefix[base + subgroup] = prefix;
    prefix += sums[subgroup];
  }
  ev.quotient_value_sum[quotient] = prefix;
}

__global__ void epoch_classify_heavy_quotients_kernel(EpochView ev) {
  const std::uint32_t quotient = blockIdx.x * blockDim.x + threadIdx.x;
  if (quotient >= kEpochQuotients)
    return;
  const std::uint32_t count =
      ev.quotient_off[quotient + 1] - ev.quotient_off[quotient];
  if (count > 32u && count <= kEpochHeavySortCap)
    ev.heavy_list[atomicAdd(ev.heavy_count, 1u)] = quotient;
}

__global__ void epoch_sort_heavy_quotients_kernel(EpochView ev) {
  __shared__ std::uint32_t shared_keys[kEpochHeavySortCap];
  __shared__ std::uint32_t shared_values[kEpochHeavySortCap];
  __shared__ std::int8_t shared_counts[kEpochHeavySortCap];
  constexpr int shards = 64;
  const int shard = blockIdx.x;
  std::uint32_t *keys = const_cast<std::uint32_t *>(ev.keys);
  std::uint32_t *values = const_cast<std::uint32_t *>(ev.values);
  std::int8_t *counts =
      const_cast<std::int8_t *>(ev.count_delta);
  const int tid = threadIdx.x;
  const std::uint32_t count = ev.heavy_count[0];
  for (std::uint32_t item = shard; item < count; item += shards) {
    const std::uint32_t quotient = ev.heavy_list[item];
    const std::uint32_t begin = ev.quotient_off[quotient];
    const std::uint32_t length = ev.quotient_off[quotient + 1] - begin;
    shared_keys[tid] =
        tid < length ? keys[begin + tid] : kEmptyKey;
    shared_values[tid] = tid < length ? values[begin + tid] : 0u;
    shared_counts[tid] =
        tid < length ? counts[begin + tid] : 0;
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

__global__ void epoch_quotient_bitmap_kernel(
    const std::uint32_t *quotient_live, std::uint32_t *quotient_bitmap) {
  const std::uint32_t quotient = blockIdx.x * blockDim.x + threadIdx.x;
  if (quotient >= kEpochQuotients)
    return;
  const std::uint32_t occupied =
      __ballot_sync(0xffffffffu, quotient_live[quotient] != 0u);
  if ((threadIdx.x & 31) == 0)
    quotient_bitmap[quotient >> 5] = occupied;
}

struct TakeLastU32 {
  __host__ __device__ std::uint32_t operator()(
      std::uint32_t, std::uint32_t newer) const {
    return newer;
  }
};

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
  if (log_ops)
    log_ops[idx] = op;
}

struct TakeLatestPayload {
  __host__ __device__ std::uint64_t operator()(
      std::uint64_t older, std::uint64_t newer) const {
    const std::uint32_t older_sequence =
        static_cast<std::uint32_t>(older) >> 1;
    const std::uint32_t newer_sequence =
        static_cast<std::uint32_t>(newer) >> 1;
    return newer_sequence >= older_sequence ? newer : older;
  }
};

__global__ void pack_latest_payload_kernel(
    const std::uint32_t *values, const std::uint8_t *ops,
    std::uint8_t constant_op, std::size_t count,
    std::uint64_t *payload) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::uint8_t op = ops ? ops[i] : constant_op;
  const std::uint32_t sequence =
      static_cast<std::uint32_t>(i);
  payload[i] =
      (static_cast<std::uint64_t>(values[i]) << 32) |
      (static_cast<std::uint64_t>(sequence) << 1) | op;
}

__global__ void unpack_latest_payload_kernel(
    const std::uint64_t *payload, std::size_t count,
    std::uint32_t *values, std::uint8_t *ops) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  values[i] = static_cast<std::uint32_t>(payload[i] >> 32);
  ops[i] = static_cast<std::uint8_t>(payload[i] & 1u);
}
struct SumDeltaPayload {
  __host__ __device__ std::uint64_t operator()(
      std::uint64_t left, std::uint64_t right) const {
    const std::uint32_t value =
        static_cast<std::uint32_t>(left >> 32) +
        static_cast<std::uint32_t>(right >> 32);
    const std::int32_t count =
        static_cast<std::int32_t>(
            static_cast<std::uint32_t>(left)) +
        static_cast<std::int32_t>(
            static_cast<std::uint32_t>(right));
    return (static_cast<std::uint64_t>(value) << 32) |
           static_cast<std::uint32_t>(count);
  }
};

__global__ void pack_delta_payload_kernel(
    const std::uint32_t *values, const std::int8_t *counts,
    std::size_t count, std::uint64_t *payload) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::int32_t count_delta =
      static_cast<std::int32_t>(counts[i]);
  payload[i] =
      (static_cast<std::uint64_t>(values[i]) << 32) |
      static_cast<std::uint32_t>(count_delta);
}

__global__ void unpack_delta_payload_kernel(
    const std::uint64_t *payload, std::size_t count,
    std::uint32_t *values, std::int8_t *counts) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  values[i] = static_cast<std::uint32_t>(payload[i] >> 32);
  counts[i] = static_cast<std::int8_t>(
      static_cast<std::int32_t>(
          static_cast<std::uint32_t>(payload[i])));
}

__global__ void epoch_count_delta_sum_kernel(EpochView epoch) {
  const std::uint32_t quotient =
      blockIdx.x * blockDim.x + threadIdx.x;
  if (quotient >= kEpochQuotients)
    return;
  std::int32_t total = 0;
  const std::uint32_t begin = epoch.quotient_off[quotient];
  const std::uint32_t end = epoch.quotient_off[quotient + 1u];
  for (std::uint32_t position = begin; position < end; ++position)
    total += static_cast<std::int32_t>(
        epoch.count_delta[position]);
  epoch.quotient_count_sum[quotient] =
      static_cast<std::uint32_t>(total);
}
__global__ void owner_successor_index_kernel(
    const std::uint32_t *queries, std::size_t count,
    std::uint32_t *out_keys, OwnerView owner, SpineView spine,
    const EpochView *epochs, int epoch_count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::uint32_t query = queries[i];
  std::uint32_t best = spine_owner_successor(spine, query, owner);
  for (int epoch = epoch_count - 1; epoch >= 0; --epoch) {
    const std::uint32_t candidate =
        epoch_owner_successor(epochs[epoch], query, owner);
    if (candidate < best)
      best = candidate;
  }
  out_keys[i] = best == kEmptyKey ? 0u : best;
}
template <bool HasEpochs, bool HasProjection>
__global__ void range_query_kernel(
    const std::uint32_t *lo, const std::uint32_t *hi, std::uint32_t *out_sums,
    std::uint32_t *out_counts, std::size_t query_count, SpineView spine,
    SpineRangeView range,
    const std::uint32_t *spine_value_prefix,
    const EpochView *epochs, int epoch_count,
    RangeDeltaView projection) {
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
  std::uint32_t sum = spine_range_sum(
      spine, range, spine_value_prefix, l, h);
  if constexpr (HasProjection) {
    sum += range_delta_sum(projection, l, h);
  } else if constexpr (HasEpochs) {
    for (int e = 0; e < epoch_count; ++e)
      sum += epoch_range_sum_one(epochs[e], l, h);
  }
  out_sums[i] = sum;
  if (out_counts) {
    std::uint32_t c = spine_range_count(spine, l, h);
    if constexpr (HasEpochs) {
      for (int e = 0; e < epoch_count; ++e)
        c += epoch_range_count_one(epochs[e], l, h);
    }
    out_counts[i] = c;
  }
}

}

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
    epochs_.reserve(gpulsmopt_detail::kEpochStorageMax);
    epoch_pool_.reserve(gpulsmopt_detail::kEpochStorageMax);
    epoch_views_.reserve(gpulsmopt_detail::kEpochStorageMax);
    epoch_views_.resize(gpulsmopt_detail::kEpochStorageMax);
    bound_epoch_views_.resize(gpulsmopt_detail::kEpochStorageMax);
    bound_epoch_view_valid_.resize(gpulsmopt_detail::kEpochStorageMax);
  }

  void clear(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    last_update_stream_ = stream;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    clear_epoch_state();
    clear_spine_state();
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
      const double measured = prof_append_ms_ + prof_delta_sort_ms_ +
                              prof_delta_ingest_ms_;
      const double other = total - measured;
      auto pct = [total](double x) {
        return total > 0.0 ? 100.0 * x / total : 0.0;
      };
      printf("[prof] insert %zu keys: total=%.3f ms\n", batch.count, total);
      printf("[prof]   delta_sort  = %.3f ms (%5.1f%%)\n",
             prof_delta_sort_ms_, pct(prof_delta_sort_ms_));
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

  void finish_pending_delete(cudaStream_t) {}

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
      CUDA_CHECK(cudaMemsetAsync(
          batch.out_values, 0xff,
          batch.count * sizeof(std::uint32_t), stream));
      if (batch.out_found) {
        CUDA_CHECK(cudaMemsetAsync(
            batch.out_found, 0,
            batch.count * sizeof(std::uint8_t), stream));
      }
      return;
    }
    const int block = 256;
    const int grid =
        static_cast<int>((batch.count + block - 1) / block);
    gpulsmopt_detail::owner_lookup_kernel<<<grid, block, 0, stream>>>(
        make_owner_view(), batch.queries, batch.count,
        batch.out_values, batch.out_found);
    CUDA_CHECK(cudaGetLastError());
  }

  void successor(const DeviceSuccessorBatch &batch,
                 cudaStream_t stream) {
    if (batch.count == 0)
      return;
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    if (c0_log_count_ != 0)
      merge_down(stream);
    ensure_spine_microdirectory(stream);
    ensure_epoch_heavy_sorted(stream);
    ensure_epoch_bitmaps(stream);
    const int block = 128;
    const int grid =
        static_cast<int>((batch.count + block - 1) / block);
    gpulsmopt_detail::owner_successor_index_kernel<<<
        grid, block, 0, stream>>>(
        batch.queries, batch.count, batch.out_keys,
        make_owner_view(), make_spine_view(), epoch_view_ptr(),
        epoch_count());
    CUDA_CHECK(cudaGetLastError());
  }

  void range(const DeviceRangeOutputBatch &batch,
             cudaStream_t stream) {
    if (batch.query_count == 0)
      return;
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    if (c0_log_count_ != 0)
      merge_down(stream);
    const bool has_epochs = !epochs_.empty();
    const bool use_projection =
        should_use_range_delta_projection(
            batch.query_count, batch.out_counts != nullptr);
    ensure_spine_microdirectory(stream);
    if (has_epochs && !use_projection) {
      ensure_epoch_heavy_sorted(stream);
      ensure_epoch_value_prefixes(stream);
    }
    if (batch.out_counts)
      ensure_epoch_count_prefixes(stream);
    if (use_projection && !range_delta_projection_ready_)
      build_range_delta_projection(stream);
    if (use_projection) {
      launch_range_query<false, true>(batch, stream);
    } else if (has_epochs) {
      launch_range_query<true, false>(batch, stream);
    } else {
      launch_range_query<false, false>(batch, stream);
    }
  }

  void consolidate(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    merge_down(stream);
    consolidate_all_epochs(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void bulk_build(const std::uint32_t *keys,
                  const std::uint32_t *values, std::size_t n,
                  cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    last_update_stream_ = stream;
    clear_epoch_state();
    clear_spine_state();
    clear_c0_log(stream);
    live_count_ = 0;
    if (n == 0) {
      clear_owner_state(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      return;
    }
    sort_direct_batch(keys, values, n, stream);
    spine_keys_.resize_discard(n);
    spine_values_.resize_discard(n);
    auto policy = thrust::cuda::par.on(stream);
    auto unique_end = thrust::reduce_by_key(
        policy, direct_sort_keys_.data(), direct_sort_keys_.data() + n,
        direct_sort_values_.data(), spine_keys_.data(),
        spine_values_.data(), thrust::equal_to<std::uint32_t>(),
        gpulsmopt_detail::TakeLastU32{});
    spine_count_ = static_cast<std::size_t>(
        unique_end.first - spine_keys_.data());
    spine_keys_.resize_discard(spine_count_);
    spine_values_.resize_discard(spine_count_);
    build_spine_metadata(stream);
    build_spine_range_cdf(stream);
    ensure_spine_microdirectory(stream);
    build_owner_index(stream);
    live_count_ = spine_count_;
    prepare_for_insert(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  std::size_t live_count() const {
    auto *self = const_cast<GPULSMOpt *>(this);
    std::unique_lock<std::shared_mutex> guard(
        self->snapshot_mutex_);
    const cudaStream_t stream = self->last_update_stream_;
    if (self->c0_log_count_ != 0)
      self->merge_down(stream);
    if (!self->owner_ready_)
      return 0;
    auto policy = thrust::cuda::par.on(stream);
    const std::size_t live = thrust::reduce(
        policy, self->owner_quotient_live_.data(),
        self->owner_quotient_live_.data() +
            gpulsmopt_detail::kEpochQuotients,
        std::size_t{0}, thrust::plus<std::size_t>());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    self->live_count_ = live;
    return self->live_count_;
  }

  std::size_t gpu_resident_bytes() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    std::size_t total = device_bytes_all(
        epoch_views_,
        range_delta_counts_, range_delta_sums_,
        range_delta_offsets_, range_delta_value_prefix_,
        range_delta_keys_, range_delta_values_,
        c0_log_keys_, c0_log_values_, c0_log_ops_, direct_sort_keys_,
        direct_sort_values_,
        sort_key_output_, sort_payload_input_,
        sort_payload_output_,
        sort_temp_storage_, epoch_merge_keys_,
        spine_keys_, spine_values_, spine_value_prefix_, spine_radix_,
        spine_range_cdf_,
        spine_micro_base_, spine_micro_offsets_, spine_micro_bits_,
        owner_primary_, owner_overflow_, owner_heads_, owner_next_,
        owner_page_alloc_, owner_error_, owner_quotient_live_,
        merge_ops_);
    for (const auto &epoch : epochs_)
      total += device_bytes_all(
          epoch.keys, epoch.values,
          epoch.count_delta, epoch.quotient_count_sum,
          epoch.quotient_off, epoch.subgroup_masks, epoch.quotient_live,
          epoch.quotient_value_sum, epoch.quotient_count_prefix,
          epoch.quotient_value_prefix, epoch.subgroup_value_prefix,
          epoch.quotient_bitmap,
          epoch.heavy_list, epoch.heavy_count);
    for (const auto &epoch : epoch_pool_)
      total += device_bytes_all(
          epoch.keys, epoch.values,
          epoch.count_delta, epoch.quotient_count_sum,
          epoch.quotient_off, epoch.subgroup_masks, epoch.quotient_live,
          epoch.quotient_value_sum, epoch.quotient_count_prefix,
          epoch.quotient_value_prefix, epoch.subgroup_value_prefix,
          epoch.quotient_bitmap,
          epoch.heavy_list, epoch.heavy_count);
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
  static T *raw_or_null(gpulsmopt_detail::RawDeviceBuffer<T> &v) {
    return v.size() == 0 ? nullptr : v.data();
  }
  template <class T>
  static const T *
  raw_or_null(const gpulsmopt_detail::RawDeviceBuffer<T> &v) {
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

  gpulsmopt_detail::SpineView make_spine_view() const {
    return {spine_keys_.data(),
            spine_values_.data(),
            spine_radix_.data(),
            spine_micro_ready_ ? spine_micro_base_.data() : nullptr,
            spine_micro_ready_ ? spine_micro_offsets_.data() : nullptr,
            spine_micro_ready_ ? spine_micro_bits_.data() : nullptr,
            spine_count_};
  }

  gpulsmopt_detail::SpineRangeView make_spine_range_view() const {
    return {spine_range_cdf_ready_ ? spine_range_cdf_.data() : nullptr,
            spine_range_min_key_, spine_range_span_};
  }

  gpulsmopt_detail::OwnerView make_owner_view() {
    return {owner_primary_.data(),
            owner_overflow_.data(),
            owner_heads_.data(),
            owner_next_.data(),
            owner_page_alloc_.data(),
            owner_error_.data(),
            owner_quotient_live_.data(),
            owner_page_capacity_};
  }

  void initialize_owner_storage(cudaStream_t stream) {
    const std::size_t primary_count =
        static_cast<std::size_t>(gpulsmopt_detail::kEpochQuotients) *
        gpulsmopt_detail::kOwnerSlots;
    const std::size_t capacity =
        std::max({std::size_t{1}, max_elements_, spine_count_,
                  batch_capacity_});
    const std::size_t pages =
        (capacity + gpulsmopt_detail::kOwnerSlots - 1u) /
        gpulsmopt_detail::kOwnerSlots;
    if (pages >
        std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("owner page capacity overflow");
    owner_page_capacity_ = static_cast<std::uint32_t>(pages);
    owner_primary_.resize_discard_exact(primary_count);
    owner_overflow_.resize_discard_exact(
        pages * gpulsmopt_detail::kOwnerSlots);
    owner_heads_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients);
    owner_next_.resize_discard_exact(pages);
    owner_page_alloc_.resize_discard_exact(1);
    owner_error_.resize_discard_exact(1);
    owner_quotient_live_.resize_discard_exact(
        gpulsmopt_detail::kEpochQuotients);
    CUDA_CHECK(cudaMemsetAsync(
        owner_primary_.data(), 0,
        owner_primary_.size() * sizeof(std::uint64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        owner_overflow_.data(), 0,
        owner_overflow_.size() * sizeof(std::uint64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        owner_heads_.data(), 0xff,
        owner_heads_.size() * sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        owner_next_.data(), 0xff,
        owner_next_.size() * sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        owner_page_alloc_.data(), 0, sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        owner_error_.data(), 0, sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        owner_quotient_live_.data(), 0,
        owner_quotient_live_.size() * sizeof(std::uint32_t), stream));
    owner_ready_ = true;
  }

  void build_owner_index(cudaStream_t stream) {
    initialize_owner_storage(stream);
    if (spine_count_ > 0) {
      gpulsmopt_detail::owner_build_kernel<<<
          gpulsmopt_detail::kEpochQuotients, 1, 0, stream>>>(
          make_owner_view(), spine_keys_.data(), spine_values_.data(),
          spine_radix_.data());
      CUDA_CHECK(cudaGetLastError());
    }
  }

  void clear_owner_state(cudaStream_t stream) {
    if (owner_primary_.size() > 0) {
      CUDA_CHECK(cudaMemsetAsync(
          owner_primary_.data(), 0,
          owner_primary_.size() * sizeof(std::uint64_t), stream));
      CUDA_CHECK(cudaMemsetAsync(
          owner_overflow_.data(), 0,
          owner_overflow_.size() * sizeof(std::uint64_t), stream));
      CUDA_CHECK(cudaMemsetAsync(
          owner_heads_.data(), 0xff,
          owner_heads_.size() * sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(
          owner_next_.data(), 0xff,
          owner_next_.size() * sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(
          owner_page_alloc_.data(), 0, sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(
          owner_error_.data(), 0, sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(
          owner_quotient_live_.data(), 0,
          owner_quotient_live_.size() * sizeof(std::uint32_t), stream));
    }
    owner_ready_ = false;
  }

  void clear_spine_state() {
    spine_count_ = 0;
    spine_keys_.resize_discard(0);
    spine_values_.resize_discard(0);
    spine_value_prefix_.resize_discard(0);
    spine_radix_.resize_discard(0);
    spine_micro_base_.resize_discard(0);
    spine_micro_offsets_.resize_discard(0);
    spine_micro_bits_.resize_discard(0);
    spine_range_cdf_.release();
    spine_range_min_key_ = 0u;
    spine_range_span_ = 0u;
    spine_range_cdf_ready_ = false;
    spine_micro_ready_ = false;
  }

  void build_spine_metadata(cudaStream_t stream) {
    const std::size_t count = spine_count_;
    spine_radix_.resize_discard(gpulsmopt_detail::kSpineRadixSize + 1);
    constexpr int block = 256;
    const int radix_grid = static_cast<int>(
        (gpulsmopt_detail::kSpineRadixSize + 1 + block - 1) / block);
    gpulsmopt_detail::spine_build_radix_kernel<<<radix_grid, block, 0,
                                                 stream>>>(
        spine_keys_.data(), count, spine_radix_.data());
    CUDA_CHECK(cudaGetLastError());
    spine_micro_ready_ = false;
    spine_value_prefix_.resize_discard(count + 1);
    CUDA_CHECK(cudaMemsetAsync(spine_value_prefix_.data(), 0,
                               sizeof(std::uint32_t), stream));
    if (count > 0) {
      auto policy = thrust::cuda::par.on(stream);
      thrust::inclusive_scan(policy, spine_values_.data(),
                             spine_values_.data() + count,
                             spine_value_prefix_.data() + 1);
    }
  }

  void build_spine_range_cdf(cudaStream_t stream) {
    spine_range_cdf_ready_ = false;
    spine_range_min_key_ = 0u;
    spine_range_span_ = 0u;
    if (spine_count_ == 0) {
      spine_range_cdf_.release();
      return;
    }
    std::uint32_t endpoints[2]{};
    CUDA_CHECK(cudaMemcpyAsync(
        endpoints, spine_keys_.data(), sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        endpoints + 1, spine_keys_.data() + spine_count_ - 1u,
        sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::uint64_t span =
        static_cast<std::uint64_t>(endpoints[1]) - endpoints[0] + 1u;
    const std::uint64_t entries = span + 1u;
    const std::uint64_t bytes =
        entries * sizeof(std::uint32_t);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    (void)total_bytes;
    const bool dense_enough =
        span <= static_cast<std::uint64_t>(spine_count_) *
                    gpulsmopt_detail::kRangeCdfMaxRatio;
    const bool reuses_storage =
        entries <= spine_range_cdf_.capacity();
    const bool memory_ok =
        reuses_storage ||
        bytes <= static_cast<std::uint64_t>(free_bytes) / 4u;
    if (!dense_enough || !memory_ok ||
        entries > std::numeric_limits<std::size_t>::max()) {
      spine_range_cdf_.release();
      return;
    }
    const std::size_t count = static_cast<std::size_t>(entries);
    spine_range_cdf_.resize_discard_exact(count);
    CUDA_CHECK(cudaMemsetAsync(
        spine_range_cdf_.data(), 0,
        count * sizeof(std::uint32_t), stream));
    constexpr int block = 256;
    const int grid = static_cast<int>(
        (spine_count_ + block - 1u) / block);
    gpulsmopt_detail::spine_range_cdf_scatter_kernel<<<
        grid, block, 0, stream>>>(
        spine_keys_.data(), spine_values_.data(), spine_count_,
        endpoints[0], spine_range_cdf_.data());
    CUDA_CHECK(cudaGetLastError());
    auto policy = thrust::cuda::par.on(stream);
    thrust::inclusive_scan(
        policy, spine_range_cdf_.data(),
        spine_range_cdf_.data() + count,
        spine_range_cdf_.data());
    spine_range_min_key_ = endpoints[0];
    spine_range_span_ = span;
    spine_range_cdf_ready_ = true;
  }

  void ensure_spine_microdirectory(cudaStream_t stream) {
    if (spine_micro_ready_)
      return;
    if (spine_count_ == 0) {
      spine_micro_ready_ = true;
      return;
    }
    spine_micro_base_.resize_discard(gpulsmopt_detail::kSpineRadixSize);
    spine_micro_bits_.resize_discard(gpulsmopt_detail::kSpineRadixSize);
    spine_micro_offsets_.resize_discard((spine_count_ + 1u) / 2u);
    constexpr int block = 256;
    constexpr int grid =
        gpulsmopt_detail::kSpineRadixSize / block;
    gpulsmopt_detail::spine_micro_plan_kernel<<<grid, block, 0, stream>>>(
        spine_radix_.data(), spine_micro_bits_.data(),
        spine_micro_base_.data());
    CUDA_CHECK(cudaGetLastError());
    auto policy = thrust::cuda::par.on(stream);
    thrust::exclusive_scan(policy, spine_micro_base_.data(),
                           spine_micro_base_.data() +
                               gpulsmopt_detail::kSpineRadixSize,
                           spine_micro_base_.data());
    gpulsmopt_detail::spine_micro_fill_kernel<<<grid, block, 0, stream>>>(
        spine_keys_.data(), spine_radix_.data(), spine_micro_bits_.data(),
        spine_micro_base_.data(), spine_micro_offsets_.data());
    CUDA_CHECK(cudaGetLastError());
    spine_micro_ready_ = true;
  }

  struct EpochStorage {
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
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_bitmap;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> heavy_list;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> heavy_count;
    std::size_t count = 0;
    bool heavy_sorted = true;
    bool value_sums_ready = true;
    bool value_prefix_ready = true;
    bool subgroup_value_prefix_ready = true;
    bool count_prefix_ready = true;
    bool bitmap_ready = true;
    bool fully_sorted = false;
  };

  const gpulsmopt_detail::EpochView *epoch_view_ptr() const {
    return raw_or_null(epoch_views_);
  }

  int epoch_count() const { return static_cast<int>(epochs_.size()); }

  std::size_t recent_epoch_count() const {
    return epochs_.size();
  }

  gpulsmopt_detail::EpochView make_epoch_view(EpochStorage &epoch) {
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
            epoch.quotient_bitmap.data(),
            epoch.heavy_list.data(),
            epoch.heavy_count.data(),
            epoch.fully_sorted ? 1u : 0u};
  }
  gpulsmopt_detail::RangeDeltaView make_range_delta_view() const {
    return {range_delta_offsets_.data(),
            range_delta_value_prefix_.data(),
            range_delta_keys_.data(), range_delta_values_.data()};
  }

  bool should_use_range_delta_projection(
      std::size_t queries, bool wants_counts) const {
    return !wants_counts && epochs_.size() >= 2u &&
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
    constexpr int quotient_grid =
        gpulsmopt_detail::kEpochQuotients / warps;
    range_delta_counts_.resize_discard(bins + 1u);
    range_delta_sums_.resize_discard(bins + 1u);
    range_delta_offsets_.resize_discard(bins + 1u);
    range_delta_value_prefix_.resize_discard(bins + 1u);
    CUDA_CHECK(cudaMemsetAsync(
        range_delta_counts_.data() + bins, 0,
        sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        range_delta_sums_.data() + bins, 0,
        sizeof(std::uint32_t), stream));
    gpulsmopt_detail::range_delta_plan_kernel<<<
        quotient_grid, block, 0, stream>>>(
        epoch_view_ptr(), epoch_count(), range_delta_counts_.data(),
        range_delta_sums_.data());
    CUDA_CHECK(cudaGetLastError());
    exclusive_scan_u32(
        range_delta_counts_.data(), range_delta_offsets_.data(),
        bins + 1u, stream);
    exclusive_scan_u32(
        range_delta_sums_.data(), range_delta_value_prefix_.data(),
        bins + 1u, stream);
    std::size_t physical_count = 0;
    for (const auto &epoch : epochs_)
      physical_count += epoch.count;
    range_delta_keys_.resize_discard(physical_count);
    range_delta_values_.resize_discard(physical_count);
    if (physical_count > 0) {
      gpulsmopt_detail::range_delta_pack_kernel<<<
          quotient_grid, block, 0, stream>>>(
          epoch_view_ptr(), epoch_count(), range_delta_offsets_.data(),
          range_delta_keys_.data(), range_delta_values_.data());
      CUDA_CHECK(cudaGetLastError());
    }
    range_delta_projection_ready_ = true;
  }

  static bool same_epoch_view(const gpulsmopt_detail::EpochView &a,
                              const gpulsmopt_detail::EpochView &b) {
    return a.keys == b.keys && a.values == b.values &&
           a.count_delta == b.count_delta &&
           a.quotient_off == b.quotient_off &&
           a.subgroup_masks == b.subgroup_masks &&
           a.quotient_live == b.quotient_live &&
           a.quotient_count_sum == b.quotient_count_sum &&
           a.quotient_value_sum == b.quotient_value_sum &&
           a.quotient_count_prefix == b.quotient_count_prefix &&
           a.quotient_value_prefix == b.quotient_value_prefix &&
           a.subgroup_value_prefix == b.subgroup_value_prefix &&
           a.quotient_bitmap == b.quotient_bitmap &&
           a.heavy_list == b.heavy_list &&
           a.heavy_count == b.heavy_count &&
           a.fully_sorted == b.fully_sorted;
  }

  void upload_epoch_view(std::size_t index,
                         const gpulsmopt_detail::EpochView &view,
                         cudaStream_t stream) {
    bound_epoch_views_[index] = view;
    bound_epoch_view_valid_[index] = 1u;
    CUDA_CHECK(cudaMemcpyAsync(raw_or_null(epoch_views_) + index,
                               bound_epoch_views_.data() + index,
                               sizeof(view), cudaMemcpyHostToDevice, stream));
  }

  void append_epoch_view(EpochStorage &epoch, cudaStream_t stream) {
    const std::size_t index = epochs_.size() - 1;
    const gpulsmopt_detail::EpochView view = make_epoch_view(epoch);
    if (bound_epoch_view_valid_[index] &&
        same_epoch_view(bound_epoch_views_[index], view))
      return;
    upload_epoch_view(index, view, stream);
  }

  void prebind_epoch_views(cudaStream_t stream) {
    std::size_t index = 0;
    for (auto &epoch : epochs_) {
      bound_epoch_views_[index] = make_epoch_view(epoch);
      bound_epoch_view_valid_[index] = 1u;
      ++index;
    }
    for (auto it = epoch_pool_.rbegin(); it != epoch_pool_.rend(); ++it) {
      bound_epoch_views_[index] = make_epoch_view(*it);
      bound_epoch_view_valid_[index] = 1u;
      ++index;
    }
    std::fill(bound_epoch_view_valid_.begin() + index,
              bound_epoch_view_valid_.end(), 0u);
    if (index == 0)
      return;
    CUDA_CHECK(cudaMemcpyAsync(raw_or_null(epoch_views_),
                               bound_epoch_views_.data(),
                               index * sizeof(bound_epoch_views_[0]),
                               cudaMemcpyHostToDevice, stream));
  }

  void refresh_active_epoch_views(cudaStream_t stream) {
    for (std::size_t i = 0; i < epochs_.size(); ++i) {
      const auto view = make_epoch_view(epochs_[i]);
      if (bound_epoch_view_valid_[i] &&
          same_epoch_view(bound_epoch_views_[i], view))
        continue;
      upload_epoch_view(i, view, stream);
    }
  }

  void prepare_epoch_metadata_storage(EpochStorage &epoch) {
    epoch.heavy_sorted = epoch.fully_sorted;
    epoch.value_sums_ready = false;
    epoch.value_prefix_ready = false;
    epoch.subgroup_value_prefix_ready = false;
    epoch.count_prefix_ready = false;
    epoch.bitmap_ready = false;
    epoch.count_delta.resize_discard(epoch.count);
    epoch.quotient_off.resize_discard(
        gpulsmopt_detail::kEpochQuotients + 1);
    epoch.subgroup_masks.resize_discard(
        gpulsmopt_detail::kEpochQuotients *
        gpulsmopt_detail::kEpochSubgroupPlanes);
    epoch.quotient_live.resize_discard(
        gpulsmopt_detail::kEpochQuotients);
    epoch.quotient_count_sum.resize_discard(
        gpulsmopt_detail::kEpochQuotients);
    epoch.quotient_value_sum.resize_discard(
        gpulsmopt_detail::kEpochQuotients);
    epoch.quotient_count_prefix.resize_discard(
        gpulsmopt_detail::kEpochQuotients + 1);
    epoch.quotient_value_prefix.resize_discard(
        gpulsmopt_detail::kEpochQuotients + 1);
    epoch.subgroup_value_prefix.resize_discard(0);
    epoch.quotient_bitmap.resize_discard(
        gpulsmopt_detail::kEpochQuotientBitmapWords);
    epoch.heavy_list.resize_discard(gpulsmopt_detail::kEpochQuotients);
    epoch.heavy_count.resize_discard(1);
  }

  void launch_epoch_metadata_kernels(EpochStorage &epoch, std::size_t count,
                                     cudaStream_t stream) {
    constexpr int block = 256;
    const std::uint32_t *keys = epoch.keys.data();
    std::uint32_t record_count = static_cast<std::uint32_t>(count);
    std::uint32_t *offsets = epoch.quotient_off.data();
    std::uint32_t *subgroup_masks = epoch.subgroup_masks.data();
    std::uint32_t *quotient_live = epoch.quotient_live.data();
    const int grid = static_cast<int>((record_count + block - 1) / block);
    gpulsmopt_detail::epoch_quotient_metadata_kernel<<<grid, block, 0,
                                                       stream>>>(
        keys, record_count, offsets, subgroup_masks, quotient_live);
    CUDA_CHECK(cudaGetLastError());
  }

  void commit_epoch_metadata(
      EpochStorage &epoch, cudaStream_t stream,
      const std::uint8_t *ops = nullptr,
      std::uint8_t constant_op =
          static_cast<std::uint8_t>(gpulsmopt_detail::kInsert),
      bool apply_owner = true) {
    prepare_epoch_metadata_storage(epoch);
    launch_epoch_metadata_kernels(epoch, epoch.count, stream);
    if (apply_owner) {
      if (!owner_ready_)
        throw std::runtime_error("owner directory is not initialized");
      gpulsmopt_detail::owner_transition_kernel<<<
          gpulsmopt_detail::kEpochQuotients, 32, 0, stream>>>(
          make_owner_view(), epoch.keys.data(), epoch.values.data(),
          ops, constant_op, epoch.quotient_off.data(),
          epoch.count_delta.data(), epoch.quotient_count_sum.data());
      CUDA_CHECK(cudaGetLastError());
    } else {
      constexpr int block = 256;
      constexpr int grid =
          gpulsmopt_detail::kEpochQuotients / block;
      gpulsmopt_detail::epoch_count_delta_sum_kernel<<<
          grid, block, 0, stream>>>(make_epoch_view(epoch));
      CUDA_CHECK(cudaGetLastError());
    }
    append_epoch_view(epoch, stream);
    invalidate_range_delta_projection();
  }

  void ensure_epoch_heavy_sorted(cudaStream_t stream) {
    constexpr int block = 256;
    constexpr int classify_grid =
        gpulsmopt_detail::kEpochQuotients / block;
    constexpr int shards = 64;
    for (auto &epoch : epochs_) {
      if (epoch.heavy_sorted || epoch.fully_sorted) {
        epoch.heavy_sorted = true;
        continue;
      }
      CUDA_CHECK(cudaMemsetAsync(epoch.heavy_count.data(), 0,
                                 sizeof(std::uint32_t), stream));
      const auto view = make_epoch_view(epoch);
      gpulsmopt_detail::epoch_classify_heavy_quotients_kernel<<<
          classify_grid, block, 0, stream>>>(view);
      CUDA_CHECK(cudaGetLastError());
      gpulsmopt_detail::epoch_sort_heavy_quotients_kernel<<<shards, 128, 0,
                                                            stream>>>(view);
      CUDA_CHECK(cudaGetLastError());
      epoch.heavy_sorted = true;
    }
  }

  void ensure_epoch_value_prefixes(cudaStream_t stream) {
    ensure_epoch_value_sums(stream);
    for (auto &epoch : epochs_) {
      if (epoch.value_prefix_ready)
        continue;
      exclusive_scan_u32(epoch.quotient_value_sum.data(),
                         epoch.quotient_value_prefix.data(),
                         gpulsmopt_detail::kEpochQuotients, stream);
      epoch.value_prefix_ready = true;
    }
  }

  void ensure_epoch_value_sums(cudaStream_t stream) {
    constexpr int block = 256;
    constexpr int grid = gpulsmopt_detail::kEpochQuotients / block;
    bool views_changed = false;
    for (auto &epoch : epochs_) {
      if (epoch.value_sums_ready &&
          epoch.subgroup_value_prefix_ready)
        continue;
      epoch.subgroup_value_prefix.resize_discard(
          gpulsmopt_detail::kEpochQuotients *
          gpulsmopt_detail::kEpochSubgroupPrefixStride);
      gpulsmopt_detail::epoch_subgroup_value_prefix_kernel<<<
          grid, block, 0, stream>>>(
          make_epoch_view(epoch), epoch.subgroup_value_prefix.data());
      CUDA_CHECK(cudaGetLastError());
      epoch.value_sums_ready = true;
      epoch.subgroup_value_prefix_ready = true;
      views_changed = true;
    }
    if (views_changed)
      refresh_active_epoch_views(stream);
  }

  void ensure_epoch_count_prefixes(cudaStream_t stream) {
    for (auto &epoch : epochs_) {
      if (epoch.count_prefix_ready)
        continue;
      exclusive_scan_u32(epoch.quotient_count_sum.data(),
                         epoch.quotient_count_prefix.data(),
                         gpulsmopt_detail::kEpochQuotients, stream);
      epoch.count_prefix_ready = true;
    }
  }

  void ensure_epoch_bitmaps(cudaStream_t stream) {
    for (auto &epoch : epochs_) {
      if (epoch.bitmap_ready)
        continue;
      gpulsmopt_detail::epoch_quotient_bitmap_kernel<<<256, 256, 0,
                                                       stream>>>(
          epoch.quotient_live.data(), epoch.quotient_bitmap.data());
      CUDA_CHECK(cudaGetLastError());
      epoch.bitmap_ready = true;
    }
  }

  void create_unsorted_epoch(
      const std::uint32_t *keys, const std::uint32_t *values,
      std::size_t count, cudaStream_t stream) {
    if (recent_epoch_count() >=
        static_cast<std::size_t>(gpulsmopt_detail::kEpochMax))
      consolidate_all_epochs(stream);
    acquire_epoch_slot();
    EpochStorage &epoch = epochs_.back();
    epoch.count = count;
    epoch.fully_sorted = false;
    epoch.keys.resize_discard(count);
    epoch.values.resize_discard(count);
    {
      GPULSMOPT_PROF_PHASE(prof_delta_sort_ms_);
      sort_epoch_batch(
          keys, values, count, epoch.keys.data(),
          epoch.values.data(), stream);
    }
    {
      GPULSMOPT_PROF_PHASE(prof_delta_ingest_ms_);
      commit_epoch_metadata(epoch, stream);
    }
  }

  void create_delete_epoch(
      const std::uint32_t *keys, std::size_t count,
      bool sorted, cudaStream_t stream) {
    if (recent_epoch_count() >=
        static_cast<std::size_t>(gpulsmopt_detail::kEpochMax))
      consolidate_all_epochs(stream);
    acquire_epoch_slot();
    EpochStorage &epoch = epochs_.back();
    epoch.count = count;
    epoch.fully_sorted = sorted;
    epoch.keys.resize_discard(count);
    epoch.values.resize_discard(count);
    if (sorted) {
      CUDA_CHECK(cudaMemcpyAsync(
          epoch.keys.data(), keys,
          count * sizeof(std::uint32_t),
          cudaMemcpyDeviceToDevice, stream));
    } else {
      sort_epoch_batch(
          keys, keys, count, epoch.keys.data(),
          epoch.values.data(), stream);
    }
    commit_epoch_metadata(
        epoch, stream, nullptr,
        static_cast<std::uint8_t>(
            gpulsmopt_detail::kTombstone));
  }
  void acquire_epoch_slot() {
    if (epoch_pool_.empty()) {
      epochs_.emplace_back();
      return;
    }
    epochs_.push_back(std::move(epoch_pool_.back()));
    epoch_pool_.pop_back();
  }

  void reserve_epoch_storage(std::size_t count) {
    while (epochs_.size() + epoch_pool_.size() <
           static_cast<std::size_t>(gpulsmopt_detail::kEpochStorageMax)) {
      epoch_pool_.emplace_back();
    }
    for (auto &epoch : epoch_pool_) {
      epoch.keys.resize_discard(count);
      epoch.values.resize_discard(count);
      epoch.count_delta.resize_discard(count);
      epoch.quotient_off.resize_discard(
          gpulsmopt_detail::kEpochQuotients + 1);
      epoch.subgroup_masks.resize_discard(
          gpulsmopt_detail::kEpochQuotients *
          gpulsmopt_detail::kEpochSubgroupPlanes);
      epoch.quotient_live.resize_discard(
          gpulsmopt_detail::kEpochQuotients);
      epoch.quotient_count_sum.resize_discard(
          gpulsmopt_detail::kEpochQuotients);
      epoch.quotient_value_sum.resize_discard(
          gpulsmopt_detail::kEpochQuotients);
      epoch.quotient_count_prefix.resize_discard(
          gpulsmopt_detail::kEpochQuotients + 1);
      epoch.quotient_value_prefix.resize_discard(
          gpulsmopt_detail::kEpochQuotients + 1);
      epoch.subgroup_value_prefix.resize_discard(0);
      epoch.quotient_bitmap.resize_discard(
          gpulsmopt_detail::kEpochQuotientBitmapWords);
      epoch.heavy_list.resize_discard(
          gpulsmopt_detail::kEpochQuotients);
      epoch.heavy_count.resize_discard(1);
    }
    const std::size_t pending_capacity =
        count * static_cast<std::size_t>(gpulsmopt_detail::kEpochStorageMax);
    epoch_merge_keys_.resize_discard(pending_capacity);
  }

  void clear_epoch_state() {
    for (auto &epoch : epochs_)
      epoch_pool_.push_back(std::move(epoch));
    epochs_.clear();
    range_delta_counts_.resize_discard(0);
    range_delta_sums_.resize_discard(0);
    range_delta_offsets_.resize_discard(0);
    range_delta_value_prefix_.resize_discard(0);
    range_delta_keys_.resize_discard(0);
    range_delta_values_.resize_discard(0);
    range_delta_projection_ready_ = false;
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

  enum class RunKind : std::uint8_t {
    empty,
    inserts,
    tombstones,
    mixed,
  };

  static RunKind operation_kind(std::uint8_t op) {
    return op == gpulsmopt_detail::kInsert
               ? RunKind::inserts
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
                                 operation_value(c0_kind_), old_total,
                                 stream));
      c0_kind_ = RunKind::mixed;
    }
    std::uint8_t *ops = c0_kind_ == RunKind::mixed
                            ? raw_or_null(c0_log_ops_)
                            : nullptr;
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
    if (op == static_cast<std::uint8_t>(gpulsmopt_detail::kInsert))
      c0_insert_count_ += static_cast<std::uint32_t>(count);
    return true;
  }

  void clear_c0_log(cudaStream_t stream) {
    if (c0_log_count_ == 0)
      return;
    c0_log_count_ = 0;
    c0_insert_count_ = 0;
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

  void exclusive_scan_u32(const std::uint32_t *input,
                          std::uint32_t *output, std::size_t count,
                          cudaStream_t stream) {
    if (count == 0)
      return;
    if (count > scan_u32_count_) {
      scan_u32_temp_bytes_ = 0;
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          nullptr, scan_u32_temp_bytes_, input, output,
          static_cast<int>(count), stream));
      scan_u32_count_ = count;
    }
    std::size_t temp_bytes = scan_u32_temp_bytes_;
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
    resize_reuse(sort_key_output_, log_count);
    sort_payload_input_.resize_discard(log_count);
    sort_payload_output_.resize_discard(log_count);
    std::size_t direct_bytes = 0;
    if (direct_count > 0) {
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, direct_bytes, spine_keys_.data(),
          direct_sort_keys_.data(), spine_values_.data(),
          direct_sort_values_.data(), direct_count, 0, 32, stream));
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
          nullptr, epoch_bytes, spine_keys_.data(),
          direct_sort_keys_.data(), spine_values_.data(),
          direct_sort_values_.data(), static_cast<std::uint32_t>(direct_count),
          16, 32, stream));
      epoch_sort_count_ = direct_count;
      epoch_sort_temp_bytes_ = epoch_bytes;
    }
    std::size_t scan_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, direct_sort_keys_.data(),
        direct_sort_values_.data(), gpulsmopt_detail::kEpochQuotients,
        stream));
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
    prepare_sort_storage(direct_count, c0_flush_budget(), stream);
    reserve_epoch_storage(direct_count);
    prebind_epoch_views(stream);
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

  void sort_epoch_batch(const std::uint32_t *keys,
                        const std::uint32_t *values, std::size_t n,
                        std::uint32_t *out_keys, std::uint32_t *out_values,
                        cudaStream_t stream) {
    if (n > epoch_sort_count_) {
      epoch_sort_temp_bytes_ = 0;
      CUDA_CHECK(gpulsmopt_detail::epoch_radix_sort_pairs(
          nullptr, epoch_sort_temp_bytes_, keys, out_keys, values, out_values,
          static_cast<std::uint32_t>(n), 16, 32, stream));
      epoch_sort_count_ = n;
    }
    std::size_t temp_bytes = epoch_sort_temp_bytes_;
    ensure_sort_temp(temp_bytes);
    CUDA_CHECK(gpulsmopt_detail::epoch_radix_sort_pairs(
        sort_temp_storage_.data(), temp_bytes, keys, out_keys, values,
        out_values, static_cast<std::uint32_t>(n), 16, 32, stream));
  }

  void consolidate_all_epochs(cudaStream_t stream) {
    if (epochs_.size() < 2)
      return;
    std::size_t total = 0;
    for (const auto &epoch : epochs_)
      total += epoch.count;
    if (total == 0) {
      clear_epoch_state();
      return;
    }
    epoch_merge_keys_.resize_discard(total);
    sort_payload_input_.resize_discard(total);
    sort_payload_output_.resize_discard(total);
    direct_sort_keys_.resize_discard(total);
    std::size_t offset = 0;
    constexpr int block = 256;
    for (const auto &epoch : epochs_) {
      CUDA_CHECK(cudaMemcpyAsync(
          epoch_merge_keys_.data() + offset, epoch.keys.data(),
          epoch.count * sizeof(std::uint32_t),
          cudaMemcpyDeviceToDevice, stream));
      const int grid = static_cast<int>(
          (epoch.count + block - 1) / block);
      gpulsmopt_detail::pack_delta_payload_kernel<<<
          grid, block, 0, stream>>>(
          epoch.values.data(), epoch.count_delta.data(),
          epoch.count, sort_payload_input_.data() + offset);
      CUDA_CHECK(cudaGetLastError());
      offset += epoch.count;
    }
    std::size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes, epoch_merge_keys_.data(),
        direct_sort_keys_.data(), sort_payload_input_.data(),
        sort_payload_output_.data(), total, 0, 32, stream));
    ensure_sort_temp(temp_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sort_temp_storage_.data(), temp_bytes,
        epoch_merge_keys_.data(), direct_sort_keys_.data(),
        sort_payload_input_.data(), sort_payload_output_.data(),
        total, 0, 32, stream));
    auto policy = thrust::cuda::par.on(stream);
    auto unique_end = thrust::reduce_by_key(
        policy, direct_sort_keys_.data(),
        direct_sort_keys_.data() + total,
        sort_payload_output_.data(), epoch_merge_keys_.data(),
        sort_payload_input_.data(), thrust::equal_to<std::uint32_t>(),
        gpulsmopt_detail::SumDeltaPayload{});
    const std::size_t unique_count =
        static_cast<std::size_t>(
            unique_end.first - epoch_merge_keys_.data());
    clear_epoch_state();
    acquire_epoch_slot();
    EpochStorage &epoch = epochs_.back();
    epoch.count = unique_count;
    epoch.fully_sorted = true;
    epoch.keys.resize_discard(unique_count);
    epoch.values.resize_discard(unique_count);
    epoch.count_delta.resize_discard(unique_count);
    CUDA_CHECK(cudaMemcpyAsync(
        epoch.keys.data(), epoch_merge_keys_.data(),
        unique_count * sizeof(std::uint32_t),
        cudaMemcpyDeviceToDevice, stream));
    const int grid = static_cast<int>(
        (unique_count + block - 1) / block);
    gpulsmopt_detail::unpack_delta_payload_kernel<<<
        grid, block, 0, stream>>>(
        sort_payload_input_.data(), unique_count,
        epoch.values.data(), epoch.count_delta.data());
    CUDA_CHECK(cudaGetLastError());
    commit_epoch_metadata(
        epoch, stream, nullptr,
        static_cast<std::uint8_t>(gpulsmopt_detail::kInsert),
        false);
  }

  bool try_create_epoch(const std::uint32_t *keys_in,
                        const std::uint32_t *values_in, std::uint8_t op,
                        std::size_t count, cudaStream_t stream) {
    const std::size_t threshold = GPULSMOPT_SCATTER_MIN_BATCH;
    if (op != gpulsmopt_detail::kInsert || values_in == nullptr ||
        count < threshold ||
        count > std::numeric_limits<std::uint32_t>::max())
      return false;
    if (c0_log_count_ != 0)
      merge_down(stream);
    create_unsorted_epoch(keys_in, values_in, count, stream);
    return true;
  }

  bool try_direct_delete(const std::uint32_t *keys,
                         std::size_t count, bool sorted,
                         cudaStream_t stream) {
    if (count < GPULSMOPT_SCATTER_MIN_BATCH ||
        count > std::numeric_limits<std::uint32_t>::max())
      return false;
    if (c0_log_count_ > 0)
      merge_down(stream);
    create_delete_epoch(keys, count, sorted, stream);
    return true;
  }

  void merge_down(cudaStream_t stream) {
    if (c0_log_count_ == 0)
      return;
    last_update_stream_ = stream;
    const std::size_t count = c0_log_count_;
    if (recent_epoch_count() >=
        static_cast<std::size_t>(gpulsmopt_detail::kEpochMax))
      consolidate_all_epochs(stream);
    resize_reuse(sort_key_output_, count);
    sort_payload_input_.resize_discard(count);
    sort_payload_output_.resize_discard(count);
    const std::uint8_t *ops =
        c0_kind_ == RunKind::mixed
            ? raw_or_null(c0_log_ops_)
            : nullptr;
    const std::uint8_t constant_op =
        operation_value(c0_kind_);
    const int block = 256;
    const int grid =
        static_cast<int>((count + block - 1) / block);
    gpulsmopt_detail::pack_latest_payload_kernel<<<
        grid, block, 0, stream>>>(
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
        sort_temp_storage_.data(), temp_bytes,
        raw_or_null(c0_log_keys_), raw_or_null(sort_key_output_),
        sort_payload_input_.data(), sort_payload_output_.data(),
        count, 0, 32, stream));
    acquire_epoch_slot();
    EpochStorage &epoch = epochs_.back();
    epoch.keys.resize_discard(count);
    epoch.values.resize_discard(count);
    resize_reuse(merge_ops_, count);
    auto policy = thrust::cuda::par.on(stream);
    auto unique_end = thrust::reduce_by_key(
        policy, sort_key_output_.begin(),
        sort_key_output_.begin() + count,
        thrust::device_pointer_cast(sort_payload_output_.data()),
        thrust::device_pointer_cast(epoch.keys.data()),
        thrust::device_pointer_cast(sort_payload_input_.data()),
        thrust::equal_to<std::uint32_t>(),
        gpulsmopt_detail::TakeLatestPayload{});
    const std::size_t unique_count =
        static_cast<std::size_t>(
            unique_end.first -
            thrust::device_pointer_cast(epoch.keys.data()));
    epoch.count = unique_count;
    epoch.fully_sorted = true;
    epoch.keys.resize_discard(unique_count);
    epoch.values.resize_discard(unique_count);
    gpulsmopt_detail::unpack_latest_payload_kernel<<<
        static_cast<int>((unique_count + block - 1) / block),
        block, 0, stream>>>(
        sort_payload_input_.data(), unique_count,
        epoch.values.data(), raw_or_null(merge_ops_));
    CUDA_CHECK(cudaGetLastError());
    commit_epoch_metadata(
        epoch, stream, raw_or_null(merge_ops_),
        static_cast<std::uint8_t>(gpulsmopt_detail::kInsert));
    clear_c0_log(stream);
  }

  void maybe_flush_and_merge(cudaStream_t stream) {
    if (c0_log_live() >= c0_flush_budget())
      merge_down(stream);
  }

  template <bool HasEpochs, bool HasProjection>
  void launch_range_query(const DeviceRangeOutputBatch &batch,
                          cudaStream_t stream) {
    const int block = 128;
    const int grid = static_cast<int>(
        (batch.query_count + block - 1) / block);
    gpulsmopt_detail::range_query_kernel<HasEpochs, HasProjection>
        <<<grid, block, 0, stream>>>(
            batch.lo, batch.hi, batch.out_sums, batch.out_counts,
            batch.query_count, make_spine_view(), make_spine_range_view(),
            spine_value_prefix_.data(),
            HasEpochs ? epoch_view_ptr() : nullptr,
            HasEpochs ? epoch_count() : 0,
            HasProjection ? make_range_delta_view()
                          : gpulsmopt_detail::RangeDeltaView{});
    CUDA_CHECK(cudaGetLastError());
  }

  std::size_t max_elements_ = 0;
  std::size_t batch_capacity_ = 0;
  std::size_t live_count_ = 0;
  std::size_t spine_count_ = 0;
  mutable std::shared_mutex snapshot_mutex_;

  std::uint32_t c0_log_count_ = 0;
  std::uint32_t c0_insert_count_ = 0;
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

  std::vector<EpochStorage> epochs_;
  std::vector<EpochStorage> epoch_pool_;
  thrust::device_vector<gpulsmopt_detail::EpochView> epoch_views_;
  std::vector<gpulsmopt_detail::EpochView> bound_epoch_views_;
  std::vector<std::uint8_t> bound_epoch_view_valid_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> range_delta_counts_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> range_delta_sums_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> range_delta_offsets_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t>
      range_delta_value_prefix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> range_delta_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> range_delta_values_;
  bool range_delta_projection_ready_ = false;

  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> direct_sort_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> direct_sort_values_;
  thrust::device_vector<std::uint32_t> sort_key_output_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> sort_payload_input_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> sort_payload_output_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> sort_temp_storage_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> epoch_merge_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_value_prefix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_range_cdf_;
  std::uint32_t spine_range_min_key_ = 0u;
  std::uint64_t spine_range_span_ = 0u;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_radix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_micro_base_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint16_t> spine_micro_offsets_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> spine_micro_bits_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> owner_primary_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> owner_overflow_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> owner_heads_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> owner_next_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> owner_page_alloc_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> owner_error_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> owner_quotient_live_;
  std::uint32_t owner_page_capacity_ = 0;
  bool owner_ready_ = false;
  cudaStream_t last_update_stream_ = nullptr;
  std::size_t direct_sort_count_ = 0;
  std::size_t direct_sort_temp_bytes_ = 0;
  std::size_t epoch_sort_count_ = 0;
  std::size_t epoch_sort_temp_bytes_ = 0;
  std::size_t log_sort_count_ = 0;
  std::size_t log_sort_temp_bytes_ = 0;
  std::size_t scan_u32_count_ = 0;
  std::size_t scan_u32_temp_bytes_ = 0;
  bool spine_micro_ready_ = false;
  bool spine_range_cdf_ready_ = false;

  thrust::device_vector<std::uint8_t> merge_ops_;
};
