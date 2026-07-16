#pragma once
#include "gpu_dictionary_adapter.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/scan.h>
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
#ifndef GPULSMOPT_DELETE_OWNER_SAMPLES
#define GPULSMOPT_DELETE_OWNER_SAMPLES 2048
#endif
#ifndef GPULSMOPT_DELETE_ADAPTIVE_MIN_BATCH
#define GPULSMOPT_DELETE_ADAPTIVE_MIN_BATCH (1 << 22)
#endif
#ifndef GPULSMOPT_DELETE_RANGE_MIN_BATCH
#define GPULSMOPT_DELETE_RANGE_MIN_BATCH (1 << 18)
#endif
#ifndef GPULSMOPT_DELETE_RANGE_TARGET
#define GPULSMOPT_DELETE_RANGE_TARGET 128
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
constexpr int kEpochMax = GPULSMOPT_EPOCH_MAX;
constexpr int kEpochQuotientBits = 16;
constexpr int kEpochSubgroupBits = 4;
constexpr int kEpochQuotients = 1 << kEpochQuotientBits;
constexpr int kEpochSubgroups = 1 << kEpochSubgroupBits;
constexpr int kEpochSubgroupPlanes = kEpochSubgroupBits;
constexpr int kEpochSubgroupPrefixStride = kEpochSubgroups + 1;
constexpr int kEpochHeavySortCap = 128;
constexpr int kEpochQuotientBitmapWords = kEpochQuotients / 32;
constexpr int kDeleteLayerMax = kEpochMax + 1;
constexpr int kDeleteOwnerMinBits = 10;
constexpr int kSpineRadixBits = 20;
constexpr int kSpineRadixShift = 32 - kSpineRadixBits;
constexpr int kSpineMicroTarget = 4;
constexpr int kSpineDeadBlockShift = 3;
constexpr int kSpineDeadBlockSize = 1 << kSpineDeadBlockShift;
constexpr std::size_t kSpineRadixSize =
    std::size_t{1} << kSpineRadixBits;
static_assert(kEpochMax <= 32, "epoch delete mask requires <= 32 epochs");
static_assert(GPULSMOPT_DELETE_OWNER_SAMPLES > 0,
              "delete sample count must be positive");
static_assert(GPULSMOPT_DELETE_RANGE_MIN_BATCH > 0,
              "delete owner minimum must be positive");
static_assert(GPULSMOPT_DELETE_RANGE_TARGET > 0,
              "delete owner target must be positive");
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
  const std::uint8_t *dead;
  const std::uint32_t *quotient_off;
  const std::uint32_t *subgroup_masks;
  std::uint32_t *quotient_live;
  std::uint32_t *quotient_value_sum;
  const std::uint32_t *quotient_count_prefix;
  const std::uint32_t *quotient_value_prefix;
  const std::uint32_t *subgroup_value_prefix;
  std::uint32_t *quotient_bitmap;
  std::uint32_t *heavy_list;
  std::uint32_t *heavy_count;
  std::uint32_t has_dead;
};

struct SpineView {
  const std::uint32_t *keys;
  const std::uint32_t *values;
  const std::uint32_t *radix;
  const std::uint32_t *micro_base;
  const std::uint16_t *micro_offsets;
  const std::uint8_t *micro_bits;
  const std::uint32_t *dead;
  std::size_t count;
};

__device__ inline bool epoch_key_killed(
    std::uint32_t key, const std::uint32_t *killed_keys,
    std::size_t killed_count) {
  if (killed_count == 0)
    return false;
  const std::size_t p = lower_bound_u32(killed_keys, killed_count, key);
  return p < killed_count && killed_keys[p] == key;
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
      if (ev.keys[p] == key && (!ev.has_dead || !ev.dead[p])) {
        *position = p;
        return true;
      }
      mask &= mask - 1;
    }
    return false;
  }
  if (count <= kEpochHeavySortCap) {
    std::uint32_t p = begin + static_cast<std::uint32_t>(
                                  lower_bound_u32(ev.keys + begin,
                                                  end - begin, key));
    for (; p < end && ev.keys[p] == key; ++p) {
      if (!ev.has_dead || !ev.dead[p]) {
        *position = p;
        return true;
      }
    }
    return false;
  }
  for (std::uint32_t p = begin; p < end; ++p) {
    if (ev.keys[p] == key && (!ev.has_dead || !ev.dead[p])) {
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
  std::uint32_t count = 0u;
  const std::uint32_t physical_count = end - begin;
  if (physical_count > 32u && physical_count <= kEpochHeavySortCap) {
    const std::uint32_t lb = begin + static_cast<std::uint32_t>(
                                         lower_bound_u32(ev.keys + begin,
                                                         end - begin, lo));
    const std::uint32_t ub = begin + static_cast<std::uint32_t>(
                                         upper_bound_u32(ev.keys + begin,
                                                         end - begin, hi));
    if (!ev.has_dead)
      return ub - lb;
    for (std::uint32_t p = lb; p < ub; ++p)
      count += ev.dead[p] == 0u;
    return count;
  }
  for (std::uint32_t p = begin; p < end; ++p)
    count += ev.keys[p] >= lo && ev.keys[p] <= hi &&
             (!ev.has_dead || !ev.dead[p]);
  return count;
}

__device__ inline std::uint32_t epoch_quotient_sum(
    const EpochView &ev, std::uint32_t quotient, std::uint32_t lo,
    std::uint32_t hi) {
  const std::uint32_t begin = ev.quotient_off[quotient];
  const std::uint32_t end = ev.quotient_off[quotient + 1];
  std::uint32_t sum = 0u;
  const std::uint32_t physical_count = end - begin;
  if (physical_count > 32u && physical_count <= kEpochHeavySortCap) {
    const std::uint32_t lb = begin + static_cast<std::uint32_t>(
                                         lower_bound_u32(ev.keys + begin,
                                                         end - begin, lo));
    const std::uint32_t ub = begin + static_cast<std::uint32_t>(
                                         upper_bound_u32(ev.keys + begin,
                                                         end - begin, hi));
    for (std::uint32_t p = lb; p < ub; ++p)
      if (!ev.has_dead || !ev.dead[p])
        sum += ev.values[p];
    return sum;
  }
  for (std::uint32_t p = begin; p < end; ++p)
    if (ev.keys[p] >= lo && ev.keys[p] <= hi &&
        (!ev.has_dead || !ev.dead[p]))
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
    if (key >= lo && key <= hi &&
        (!ev.has_dead || !ev.dead[position]))
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
    const std::uint32_t *prefix =
        ev.has_dead ? ev.quotient_count_prefix : ev.quotient_off;
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
    const EpochView &ev, std::uint32_t quotient, std::uint32_t floor,
    const std::uint32_t *killed_keys, std::size_t killed_count) {
  const std::uint32_t begin = ev.quotient_off[quotient];
  const std::uint32_t end = ev.quotient_off[quotient + 1];
  const std::uint32_t physical_count = end - begin;
  if (physical_count > 32u && physical_count <= kEpochHeavySortCap) {
    std::uint32_t p = begin + static_cast<std::uint32_t>(
                                  lower_bound_u32(ev.keys + begin,
                                                  end - begin, floor));
    for (; p < end; ++p) {
      const std::uint32_t key = ev.keys[p];
      if ((!ev.has_dead || !ev.dead[p]) &&
          !epoch_key_killed(key, killed_keys, killed_count))
        return key;
    }
    return kEmptyKey;
  }
  std::uint32_t best = kEmptyKey;
  for (std::uint32_t p = begin; p < end; ++p) {
    const std::uint32_t key = ev.keys[p];
    if (key >= floor && key < best && (!ev.has_dead || !ev.dead[p]) &&
        !epoch_key_killed(key, killed_keys, killed_count))
      best = key;
  }
  return best;
}

__device__ inline std::uint32_t epoch_successor_candidate(
    const EpochView &ev, std::uint32_t key,
    const std::uint32_t *killed_keys, std::size_t killed_count) {
  std::uint32_t quotient = key >> kEpochQuotientBits;
  std::uint32_t candidate = epoch_quotient_successor(
      ev, quotient, key, killed_keys, killed_count);
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
    candidate = epoch_quotient_successor(
        ev, quotient, 0u, killed_keys, killed_count);
    if (candidate != kEmptyKey)
      return candidate;
    ++quotient;
  }
  return kEmptyKey;
}

__device__ inline bool spine_dead(const std::uint32_t *dead,
                                  std::size_t position) {
  return dead && ((dead[position >> 5] >> (position & 31u)) & 1u);
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

__device__ inline bool spine_point_find(
    const SpineView &spine, std::uint32_t key,
    std::uint32_t *out_value) {
  if (spine.count == 0)
    return false;
  const std::size_t position = spine_lower_rank(spine, key);
  if (position >= spine.count || spine.keys[position] != key ||
      spine_dead(spine.dead, position))
    return false;
  if (out_value)
    *out_value = spine.values[position];
  return true;
}

__device__ inline std::uint32_t spine_dead_count_range(
    const std::uint32_t *dead, const std::uint32_t *dead_prefix,
    std::size_t begin, std::size_t end) {
  if (!dead || begin >= end)
    return 0u;
  const std::size_t first = begin >> kSpineDeadBlockShift;
  const std::size_t last = (end - 1) >> kSpineDeadBlockShift;
  std::uint32_t total = 0u;
  if (first == last) {
    for (std::size_t p = begin; p < end; ++p)
      total += spine_dead(dead, p);
    return total;
  }
  const std::size_t first_end = (first + 1) << kSpineDeadBlockShift;
  for (std::size_t p = begin; p < first_end; ++p)
    total += spine_dead(dead, p);
  const std::size_t last_begin = last << kSpineDeadBlockShift;
  for (std::size_t p = last_begin; p < end; ++p)
    total += spine_dead(dead, p);
  if (last > first + 1)
    total += dead_prefix[last] - dead_prefix[first + 1];
  return total;
}

__device__ inline std::uint32_t spine_dead_sum_range(
    const std::uint32_t *values, const std::uint32_t *dead,
    const std::uint32_t *dead_prefix, std::size_t begin,
    std::size_t end) {
  if (!dead || begin >= end)
    return 0u;
  const std::size_t first = begin >> kSpineDeadBlockShift;
  const std::size_t last = (end - 1) >> kSpineDeadBlockShift;
  std::uint32_t total = 0u;
  if (first == last) {
    for (std::size_t p = begin; p < end; ++p)
      if (spine_dead(dead, p))
        total += values[p];
    return total;
  }
  const std::size_t first_end = (first + 1) << kSpineDeadBlockShift;
  for (std::size_t p = begin; p < first_end; ++p)
    if (spine_dead(dead, p))
      total += values[p];
  const std::size_t last_begin = last << kSpineDeadBlockShift;
  for (std::size_t p = last_begin; p < end; ++p)
    if (spine_dead(dead, p))
      total += values[p];
  if (last > first + 1)
    total += dead_prefix[last] - dead_prefix[first + 1];
  return total;
}

__device__ inline std::uint32_t spine_range_count(
    const SpineView &spine, const std::uint32_t *dead_prefix,
    std::uint32_t lo, std::uint32_t hi) {
  if (spine.count == 0)
    return 0u;
  std::size_t begin = 0, end = 0;
  spine_range_ranks(spine, lo, hi, &begin, &end);
  return static_cast<std::uint32_t>(end - begin) -
         spine_dead_count_range(spine.dead, dead_prefix, begin, end);
}

__device__ inline std::uint32_t spine_range_sum(
    const SpineView &spine, const std::uint32_t *value_prefix,
    const std::uint32_t *dead_value_prefix, std::uint32_t lo,
    std::uint32_t hi) {
  if (spine.count == 0)
    return 0u;
  std::size_t begin = 0, end = 0;
  spine_range_ranks(spine, lo, hi, &begin, &end);
  return value_prefix[end] - value_prefix[begin] -
         spine_dead_sum_range(spine.values, spine.dead,
                              dead_value_prefix, begin, end);
}

__device__ inline std::uint32_t spine_successor_candidate(
    const SpineView &spine, std::uint32_t key,
    const std::uint32_t *killed_keys, std::size_t killed_count) {
  if (spine.count == 0)
    return kEmptyKey;
  std::size_t position = spine_lower_rank(spine, key);
  for (; position < spine.count; ++position) {
    const std::uint32_t candidate = spine.keys[position];
    if (!spine_dead(spine.dead, position) &&
        !epoch_key_killed(candidate, killed_keys, killed_count))
      return candidate;
  }
  return kEmptyKey;
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

__global__ void spine_live_input_kernel(
    const std::uint32_t *dead, std::size_t count,
    std::uint32_t *live) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    live[i] = spine_dead(dead, i) ? 0u : 1u;
}

__global__ void spine_gather_live_kernel(
    const std::uint32_t *keys, const std::uint32_t *values,
    const std::uint32_t *dead, const std::uint32_t *prefix,
    std::size_t count, std::uint32_t *out_keys,
    std::uint32_t *out_values) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || spine_dead(dead, i))
    return;
  const std::size_t output = prefix[i];
  out_keys[output] = keys[i];
  out_values[output] = values[i];
}

__global__ void epoch_quotient_metadata_kernel(
    const std::uint32_t *keys, std::uint32_t record_count,
    std::uint32_t *offsets, std::uint32_t *subgroup_masks,
    std::uint32_t *quotient_live) {
  const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t stride = gridDim.x * blockDim.x;
  for (std::uint32_t i = tid; i < record_count; i += stride) {
    const std::uint32_t quotient = keys[i] >> kEpochQuotientBits;
    const unsigned active = __activemask();
    std::uint32_t previous = __shfl_up_sync(active, quotient, 1);
    if ((threadIdx.x & 31) == 0 && i != 0u)
      previous = keys[i - 1] >> kEpochQuotientBits;
    if (i != 0u && quotient == previous)
      continue;
    std::uint32_t planes[kEpochSubgroupPlanes] = {};
    std::uint32_t end = i;
    while (end < record_count) {
      const std::uint32_t current_key = end == i ? keys[i] : keys[end];
      if ((current_key >> kEpochQuotientBits) != quotient)
        break;
      const std::uint32_t local = end - i;
      if (local < 32u) {
        const std::uint32_t subgroup =
            (current_key >>
             (kEpochQuotientBits - kEpochSubgroupBits)) &
            (kEpochSubgroups - 1);
        const std::uint32_t position = 1u << local;
#pragma unroll
        for (int bit = 0; bit < kEpochSubgroupPlanes; ++bit)
          planes[bit] |= (0u - ((subgroup >> bit) & 1u)) & position;
      }
      ++end;
    }
    if (i == 0u) {
      for (std::uint32_t q = 0; q <= quotient; ++q)
        offsets[q] = 0u;
      for (std::uint32_t q = 0; q < quotient; ++q)
        quotient_live[q] = 0u;
    } else {
      for (std::uint32_t q = previous + 1u; q <= quotient; ++q)
        offsets[q] = i;
      for (std::uint32_t q = previous + 1u; q < quotient; ++q)
        quotient_live[q] = 0u;
    }
    if (end == record_count) {
      for (std::uint32_t q = quotient + 1u; q <= kEpochQuotients; ++q)
        offsets[q] = record_count;
      for (std::uint32_t q = quotient + 1u; q < kEpochQuotients; ++q)
        quotient_live[q] = 0u;
    }
    const std::uint32_t count = end - i;
    if (count <= 32u) {
      const std::uint32_t base = quotient * kEpochSubgroupPlanes;
#pragma unroll
      for (int bit = 0; bit < kEpochSubgroupPlanes; ++bit)
        subgroup_masks[base + bit] = planes[bit];
    }
    quotient_live[quotient] = count;
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
    if (ev.has_dead && ev.dead[position])
      continue;
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
  subgroup_prefix[base + kEpochSubgroups] = prefix;
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
  constexpr int shards = 64;
  const int shard = blockIdx.x;
  std::uint32_t *keys = const_cast<std::uint32_t *>(ev.keys);
  std::uint32_t *values = const_cast<std::uint32_t *>(ev.values);
  const int tid = threadIdx.x;
  const std::uint32_t count = ev.heavy_count[0];
  for (std::uint32_t item = shard; item < count; item += shards) {
    const std::uint32_t quotient = ev.heavy_list[item];
    const std::uint32_t begin = ev.quotient_off[quotient];
    const std::uint32_t length = ev.quotient_off[quotient + 1] - begin;
    shared_keys[tid] =
        tid < length ? keys[begin + tid] : kEmptyKey;
    shared_values[tid] = tid < length ? values[begin + tid] : 0u;
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
          }
        }
        __syncthreads();
      }
    }
    if (tid < length) {
      keys[begin + tid] = shared_keys[tid];
      values[begin + tid] = shared_values[tid];
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

__device__ inline std::uint32_t
epoch_delete_probe_work(std::uint32_t count) {
  if (count <= 32u)
    return 6u + (count + 15u) / 16u;
  if (count <= kEpochHeavySortCap)
    return 4u + 32u - __clz(count - 1u);
  return 2u + (count < 4096u ? count : 4096u);
}

__device__ inline bool delete_layer_precedes(
    int lhs, int rhs, const std::uint32_t *hits,
    const std::uint32_t *work) {
  if (hits[lhs] == 0u)
    return false;
  if (hits[rhs] == 0u)
    return true;
  const unsigned long long left =
      static_cast<unsigned long long>(hits[lhs]) * work[rhs];
  const unsigned long long right =
      static_cast<unsigned long long>(hits[rhs]) * work[lhs];
  return left > right;
}

__global__ void delete_layer_order_kernel(
    const std::uint32_t *keys, std::size_t n, const EpochView *epochs,
    int epoch_count, SpineView spine, std::uint32_t *global_hits,
    std::uint32_t *global_work, std::uint32_t *block_counter,
    std::uint8_t *layer_order) {
  __shared__ std::uint32_t owner_hits[kDeleteLayerMax];
  __shared__ std::uint32_t probe_work[kDeleteLayerMax];
  const int layer_count = epoch_count + 1;
  for (int layer = threadIdx.x; layer < layer_count;
       layer += blockDim.x) {
    owner_hits[layer] = 0u;
    probe_work[layer] = 0u;
  }
  __syncthreads();

  const std::uint32_t sample_count =
      n < static_cast<std::size_t>(GPULSMOPT_DELETE_OWNER_SAMPLES)
          ? static_cast<std::uint32_t>(n)
          : static_cast<std::uint32_t>(GPULSMOPT_DELETE_OWNER_SAMPLES);
  for (std::uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
       sample < sample_count; sample += gridDim.x * blockDim.x) {
    const unsigned long long scaled =
        (2ull * sample + 1ull) * static_cast<unsigned long long>(n);
    const std::size_t index = scaled / (2ull * sample_count);
    const std::uint32_t key = keys[index];
    const std::uint32_t quotient = key >> kEpochQuotientBits;
    int owner = -1;
    for (int rank = 0; rank < epoch_count; ++rank) {
      const int epoch = epoch_count - rank - 1;
      const EpochView &ev = epochs[epoch];
      const std::uint32_t begin = ev.quotient_off[quotient];
      const std::uint32_t end = ev.quotient_off[quotient + 1];
      atomicAdd(probe_work + epoch,
                epoch_delete_probe_work(end - begin));
      if (owner >= 0)
        continue;
      std::uint32_t position;
      if (epoch_point_position_bounds(ev, key, quotient, begin, end,
                                      &position))
        owner = epoch;
    }
    atomicAdd(probe_work + epoch_count,
              spine.count == 0 ? 1u : (spine.micro_bits ? 8u : 12u));
    if (owner < 0 && spine_point_find(spine, key, nullptr))
      owner = epoch_count;
    if (owner >= 0)
      atomicAdd(owner_hits + owner, 1u);
  }
  __syncthreads();

  for (int layer = threadIdx.x; layer < layer_count;
       layer += blockDim.x) {
    atomicAdd(global_hits + layer, owner_hits[layer]);
    atomicAdd(global_work + layer, probe_work[layer]);
  }
  __syncthreads();
  if (threadIdx.x != 0)
    return;
  __threadfence();
  const std::uint32_t ticket = atomicAdd(block_counter, 1u);
  if (ticket + 1u != gridDim.x)
    return;
  for (int rank = 0; rank < epoch_count; ++rank)
    layer_order[rank] = static_cast<std::uint8_t>(epoch_count - rank - 1);
  layer_order[epoch_count] = static_cast<std::uint8_t>(epoch_count);
  for (int rank = 1; rank < layer_count; ++rank) {
    const std::uint8_t layer = layer_order[rank];
    int position = rank;
    while (position > 0 &&
           delete_layer_precedes(layer, layer_order[position - 1],
                                 global_hits, global_work)) {
      layer_order[position] = layer_order[position - 1];
      --position;
    }
    layer_order[position] = layer;
  }
}

__global__ void epoch_spine_delete_kernel(
    const std::uint32_t *keys, std::size_t n, const EpochView *epochs,
    int epoch_count, const std::uint8_t *layer_order,
    std::uint32_t *epoch_removed,
    std::uint32_t value_sums_ready_mask, SpineView spine,
    std::uint32_t *spine_dead_count, std::uint32_t *spine_dead_value,
    std::uint32_t *spine_removed) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint32_t key = keys[i];
  const std::uint32_t quotient = key >> kEpochQuotientBits;
  int found_epoch = -1;
  bool found_spine = false;
  std::size_t found_position = 0;
  std::uint32_t found_value = 0u;
  for (int rank = 0; rank <= epoch_count; ++rank) {
    const int layer = layer_order
                          ? static_cast<int>(layer_order[rank])
                          : (rank < epoch_count ? epoch_count - rank - 1
                                                : epoch_count);
    if (layer == epoch_count) {
      if (spine.count == 0)
        continue;
      const std::size_t position = spine_lower_rank(spine, key);
      if (position >= spine.count || spine.keys[position] != key ||
          spine_dead(spine.dead, position))
        continue;
      found_spine = true;
      found_position = position;
      break;
    }
    const EpochView &ev = epochs[layer];
    std::uint32_t position = 0u;
    if (!epoch_point_position(ev, key, &position))
      continue;
    const_cast<std::uint8_t *>(ev.dead)[position] = 1u;
    found_value = ev.values[position];
    found_epoch = layer;
    break;
  }
  const unsigned active = __activemask();
  const unsigned found_mask =
      __ballot_sync(active, found_epoch >= 0);
  if (found_epoch >= 0) {
    const int lane = threadIdx.x & 31;
    const std::uint32_t group_key =
        (static_cast<std::uint32_t>(found_epoch) << 16) | quotient;
    const unsigned quotient_peers =
        __match_any_sync(found_mask, group_key);
    const int quotient_leader = __ffs(quotient_peers) - 1;
    const std::uint32_t removed =
        static_cast<std::uint32_t>(__popc(quotient_peers));
    const std::uint32_t removed_sum =
        __reduce_add_sync(quotient_peers, found_value);
    if (lane == quotient_leader) {
      EpochView ev = epochs[found_epoch];
      const std::uint32_t old =
          atomicSub(ev.quotient_live + quotient, removed);
      if ((value_sums_ready_mask >> found_epoch) & 1u)
        atomicSub(ev.quotient_value_sum + quotient, removed_sum);
      if (old == removed)
        atomicAnd(ev.quotient_bitmap + (quotient >> 5),
                  ~(1u << (quotient & 31u)));
    }
    const unsigned epoch_peers =
        __match_any_sync(found_mask, found_epoch);
    if (lane == __ffs(epoch_peers) - 1) {
      atomicAdd(epoch_removed + found_epoch,
                static_cast<std::uint32_t>(__popc(epoch_peers)));
    }
    return;
  }
  const unsigned spine_active = active & ~found_mask;
  const unsigned spine_mask = __ballot_sync(spine_active, found_spine);
  if (!found_spine)
    return;
  const std::size_t position = found_position;
  const std::size_t word = position >> 5;
  const std::uint32_t bit = 1u << (position & 31u);
  const unsigned word_peers =
      __match_any_sync(spine_mask, static_cast<std::uint32_t>(word));
  const int lane = threadIdx.x & 31;
  const int word_leader = __ffs(word_peers) - 1;
  const std::uint32_t word_bits = __reduce_or_sync(word_peers, bit);
  std::uint32_t old_word = 0u;
  if (lane == word_leader) {
    old_word = atomicOr(const_cast<std::uint32_t *>(spine.dead) + word,
                        word_bits);
  }
  old_word = __shfl_sync(word_peers, old_word, word_leader);
  const unsigned bit_peers = __match_any_sync(word_peers, bit);
  const bool newly_dead =
      lane == __ffs(bit_peers) - 1 && (old_word & bit) == 0u;
  const unsigned new_mask = __ballot_sync(spine_mask, newly_dead);
  if (!newly_dead)
    return;
  const std::size_t dead_block = position >> kSpineDeadBlockShift;
  const unsigned block_peers = __match_any_sync(
      new_mask, static_cast<std::uint32_t>(dead_block));
  const int block_leader = __ffs(block_peers) - 1;
  const std::uint32_t value_sum =
      __reduce_add_sync(block_peers, spine.values[position]);
  if (lane == block_leader) {
    atomicAdd(spine_dead_count + dead_block,
              static_cast<std::uint32_t>(__popc(block_peers)));
    atomicAdd(spine_dead_value + dead_block, value_sum);
  }
  if (lane == __ffs(new_mask) - 1)
    atomicAdd(spine_removed, static_cast<std::uint32_t>(__popc(new_mask)));
}

__device__ inline bool delete_rank_group_owned(
    std::size_t group, int shift, std::size_t owner_begin,
    std::size_t owner_end, std::size_t total) {
  const std::size_t begin = group << shift;
  const std::size_t end = min(begin + (std::size_t{1} << shift), total);
  return begin >= owner_begin && end <= owner_end;
}

__global__ void epoch_spine_owner_delete_kernel(
    const std::uint32_t *keys, std::uint32_t n, const EpochView *epochs,
    int epoch_count, const std::uint8_t *layer_order,
    std::uint32_t *epoch_removed,
    std::uint32_t value_sums_ready_mask, SpineView spine,
    std::uint32_t *spine_dead_count, std::uint32_t *spine_dead_value,
    int owner_bits, std::uint32_t owner_count) {
  constexpr int kWarps = 8;
  __shared__ std::uint32_t owner_bounds[kWarps + 1];
  __shared__ std::uint32_t block_removed[kDeleteLayerMax];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const std::uint32_t owner_base = blockIdx.x * kWarps;
  const int owner_shift = 32 - owner_bits;
  if (threadIdx.x < kWarps + 1) {
    const std::uint32_t boundary = owner_base + threadIdx.x;
    if (boundary == 0u) {
      owner_bounds[threadIdx.x] = 0u;
    } else if (boundary >= owner_count) {
      owner_bounds[threadIdx.x] = n;
    } else {
      const std::uint32_t key = boundary << owner_shift;
      owner_bounds[threadIdx.x] = static_cast<std::uint32_t>(
          lower_bound_u32(keys, n, key));
    }
  }
  if (threadIdx.x < kDeleteLayerMax)
    block_removed[threadIdx.x] = 0u;
  __syncthreads();

  const std::uint32_t owner = owner_base + warp;
  if (owner < owner_count) {
    const std::uint32_t input_begin = owner_bounds[warp];
    const std::uint32_t input_end = owner_bounds[warp + 1];
    std::size_t spine_begin = 0;
    std::size_t spine_end = 0;
    if (spine.count != 0) {
      const int radix_shift = kSpineRadixBits - owner_bits;
      const std::uint32_t radix_begin = owner << radix_shift;
      const std::uint32_t radix_end = (owner + 1u) << radix_shift;
      spine_begin = spine.radix[radix_begin];
      spine_end = spine.radix[radix_end];
    }
    constexpr unsigned full = 0xffffffffu;
    for (std::uint32_t base = input_begin; base < input_end; base += 32u) {
      const std::uint32_t input = base + lane;
      const bool valid = input < input_end;
      std::uint32_t key = 0u;
      std::uint32_t quotient = 0u;
      int found_epoch = -1;
      bool found_spine = false;
      std::size_t found_position = 0;
      std::uint32_t found_value = 0u;
      if (valid) {
        key = keys[input];
        quotient = key >> kEpochQuotientBits;
        for (int rank = 0; rank <= epoch_count; ++rank) {
          const int layer = layer_order
                                ? static_cast<int>(layer_order[rank])
                                : (rank < epoch_count
                                       ? epoch_count - rank - 1
                                       : epoch_count);
          if (layer == epoch_count) {
            if (spine.count == 0)
              continue;
            const std::size_t position = spine_lower_rank(spine, key);
            if (position >= spine.count || spine.keys[position] != key ||
                spine_dead(spine.dead, position))
              continue;
            found_spine = true;
            found_position = position;
            break;
          }
          const EpochView &ev = epochs[layer];
          std::uint32_t position = 0u;
          if (!epoch_point_position(ev, key, &position))
            continue;
          const_cast<std::uint8_t *>(ev.dead)[position] = 1u;
          found_value = ev.values[position];
          found_epoch = layer;
          found_position = position;
          break;
        }
      }

      const unsigned epoch_mask =
          __ballot_sync(full, valid && found_epoch >= 0);
      if (found_epoch >= 0) {
        const std::uint32_t group_key =
            (static_cast<std::uint32_t>(found_epoch) << 16) | quotient;
        const unsigned quotient_peers =
            __match_any_sync(epoch_mask, group_key);
        const int quotient_leader = __ffs(quotient_peers) - 1;
        const std::uint32_t removed =
            static_cast<std::uint32_t>(__popc(quotient_peers));
        const std::uint32_t removed_sum =
            __reduce_add_sync(quotient_peers, found_value);
        if (lane == quotient_leader) {
          EpochView ev = epochs[found_epoch];
          const std::uint32_t old = ev.quotient_live[quotient];
          ev.quotient_live[quotient] = old - removed;
          if ((value_sums_ready_mask >> found_epoch) & 1u)
            ev.quotient_value_sum[quotient] -= removed_sum;
          if (old == removed)
            atomicAnd(ev.quotient_bitmap + (quotient >> 5),
                      ~(1u << (quotient & 31u)));
        }
        const unsigned layer_peers =
            __match_any_sync(epoch_mask, found_epoch);
        if (lane == __ffs(layer_peers) - 1) {
          atomicAdd(block_removed + found_epoch,
                    static_cast<std::uint32_t>(__popc(layer_peers)));
        }
      }

      const unsigned spine_mask =
          __ballot_sync(full, valid && found_spine);
      bool newly_spine_dead = false;
      if (found_spine) {
        const std::size_t word = found_position >> 5;
        const std::uint32_t bit = 1u << (found_position & 31u);
        const unsigned word_peers = __match_any_sync(
            spine_mask, static_cast<std::uint32_t>(word));
        const int word_leader = __ffs(word_peers) - 1;
        const std::uint32_t word_bits =
            __reduce_or_sync(word_peers, bit);
        std::uint32_t old_word = 0u;
        if (lane == word_leader) {
          auto *dead = const_cast<std::uint32_t *>(spine.dead) + word;
          if (delete_rank_group_owned(word, 5, spine_begin,
                                      spine_end, spine.count)) {
            old_word = *dead;
            *dead = old_word | word_bits;
          } else {
            old_word = atomicOr(dead, word_bits);
          }
        }
        old_word = __shfl_sync(word_peers, old_word, word_leader);
        const unsigned bit_peers = __match_any_sync(word_peers, bit);
        newly_spine_dead =
            lane == __ffs(bit_peers) - 1 && (old_word & bit) == 0u;
      }
      const unsigned new_mask =
          __ballot_sync(full, newly_spine_dead);
      if (newly_spine_dead) {
        const std::size_t dead_block =
            found_position >> kSpineDeadBlockShift;
        const unsigned block_peers = __match_any_sync(
            new_mask, static_cast<std::uint32_t>(dead_block));
        const int block_leader = __ffs(block_peers) - 1;
        const std::uint32_t removed =
            static_cast<std::uint32_t>(__popc(block_peers));
        const std::uint32_t removed_sum =
            __reduce_add_sync(block_peers,
                              spine.values[found_position]);
        if (lane == block_leader) {
          if (delete_rank_group_owned(
                  dead_block, kSpineDeadBlockShift,
                  spine_begin, spine_end, spine.count)) {
            spine_dead_count[dead_block] += removed;
            spine_dead_value[dead_block] += removed_sum;
          } else {
            atomicAdd(spine_dead_count + dead_block, removed);
            atomicAdd(spine_dead_value + dead_block, removed_sum);
          }
        }
      }
      if (new_mask != 0u && lane == __ffs(new_mask) - 1) {
        atomicAdd(block_removed + epoch_count,
                  static_cast<std::uint32_t>(__popc(new_mask)));
      }
      __syncwarp();
    }
  }
  __syncthreads();
  if (threadIdx.x <= epoch_count) {
    const std::uint32_t removed = block_removed[threadIdx.x];
    if (removed != 0u)
      atomicAdd(epoch_removed + threadIdx.x, removed);
  }
}

__global__ void epoch_combined_quotient_counts_kernel(
    const EpochView *epochs, int epoch_count,
    std::uint32_t *combined_counts) {
  const std::uint32_t quotient = blockIdx.x * blockDim.x + threadIdx.x;
  if (quotient >= kEpochQuotients)
    return;
  std::uint32_t count = 0u;
  for (int e = 0; e < epoch_count; ++e)
    count += epochs[e].quotient_live[quotient];
  combined_counts[quotient] = count;
}

__global__ void set_last_quotient_offset_kernel(
    std::uint32_t *offsets, std::uint32_t total) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    offsets[kEpochQuotients] = total;
}

__global__ void epoch_pack_quotients_kernel(
    const EpochView *epochs, int epoch_count,
    std::uint32_t *write_cursor, std::uint32_t *out_keys,
    std::uint32_t *out_values) {
  const std::uint32_t quotient = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warp_count = blockDim.x >> 5;
  for (int e = 0; e < epoch_count; ++e) {
    const EpochView &ev = epochs[e];
    const std::uint32_t begin = ev.quotient_off[quotient];
    const std::uint32_t end = ev.quotient_off[quotient + 1];
    for (std::uint32_t chunk = begin + warp * 32u; chunk < end;
         chunk += warp_count * 32u) {
      const std::uint32_t position = chunk + lane;
      const bool live = position < end &&
                        (!ev.has_dead || !ev.dead[position]);
      const unsigned live_mask = __ballot_sync(0xffffffffu, live);
      if (live_mask == 0u)
        continue;
      const int leader = __ffs(live_mask) - 1;
      std::uint32_t output_base = 0u;
      if (lane == leader)
        output_base = atomicAdd(write_cursor + quotient,
                                static_cast<std::uint32_t>(__popc(live_mask)));
      output_base = __shfl_sync(0xffffffffu, output_base, leader);
      if (live) {
        const std::uint32_t rank = static_cast<std::uint32_t>(
            __popc(live_mask & ((1u << lane) - 1u)));
        const std::uint32_t output = output_base + rank;
        out_keys[output] = ev.keys[position];
        out_values[output] = ev.values[position];
      }
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

template <bool HasOverlay, bool HasEpochs>
__global__ void point_lookup_kernel(
    const std::uint32_t *queries, std::size_t n, std::uint32_t *out_value,
    std::uint8_t *out_found, const std::uint32_t *ins_keys,
    const std::uint32_t *ins_values, std::size_t ins_count,
    const std::uint32_t *tomb_keys, std::size_t tomb_count,
    SpineView spine,
    const EpochView *epochs, int epoch_count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint32_t key = queries[i];
  if constexpr (HasOverlay) {
    if (tomb_count > 0) {
      const std::size_t p = lower_bound_u32(tomb_keys, tomb_count, key);
      if (p < tomb_count && tomb_keys[p] == key) {
        if (out_found)
          out_found[i] = 0u;
        out_value[i] = kEmptyKey;
        return;
      }
    }
    if (ins_count > 0) {
      const std::size_t p = lower_bound_u32(ins_keys, ins_count, key);
      if (p < ins_count && ins_keys[p] == key) {
        if (out_found)
          out_found[i] = 1u;
        out_value[i] = ins_values[p];
        return;
      }
    }
  }
  std::uint32_t value = 0;
  bool found = spine_point_find(spine, key, &value);
  if constexpr (HasEpochs) {
    for (int e = epoch_count - 1; !found && e >= 0; --e)
      found = epoch_point_find(epochs[e], key, &value);
  }
  if (out_found)
    out_found[i] = found ? 1u : 0u;
  out_value[i] = found ? value : kEmptyKey;
}

template <bool HasOverlay, bool HasEpochs>
__global__ void successor_index_kernel(
    const std::uint32_t *queries, std::size_t n, std::uint32_t *out_keys,
    SpineView spine, const EpochView *epochs, int epoch_count,
    const std::uint32_t *killed_keys, std::size_t killed_count,
    const std::uint32_t *live_ins_keys, std::size_t live_ins_count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::uint32_t q = queries[i];
  std::uint32_t best = kEmptyKey;
  if constexpr (HasOverlay) {
    if (live_ins_count > 0) {
      const std::size_t p = lower_bound_u32(live_ins_keys, live_ins_count, q);
      if (p < live_ins_count)
        best = live_ins_keys[p];
    }
  }
  const std::uint32_t spine_candidate =
      spine_successor_candidate(
          spine, q, HasOverlay ? killed_keys : nullptr,
          HasOverlay ? killed_count : 0);
  if (spine_candidate < best)
    best = spine_candidate;
  if constexpr (HasEpochs) {
    for (int e = epoch_count - 1; e >= 0; --e) {
      const std::uint32_t candidate = epoch_successor_candidate(
          epochs[e], q, HasOverlay ? killed_keys : nullptr,
          HasOverlay ? killed_count : 0);
      if (candidate < best)
        best = candidate;
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

__global__ void base_point_values_kernel(
    const std::uint32_t *keys, std::size_t n, SpineView spine,
    const EpochView *epochs, int epoch_count,
    std::uint32_t *out_val,
    std::uint32_t *out_flag) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  std::uint32_t v = 0;
  bool f = spine_point_find(spine, keys[i], &v);
  for (int e = epoch_count - 1; !f && e >= 0; --e)
    f = epoch_point_find(epochs[e], keys[i], &v);
  out_val[i] = f ? v : 0u;
  out_flag[i] = f ? 1u : 0u;
}

template <bool HasOverlay, bool HasEpochs>
__global__ void range_query_kernel(
    const std::uint32_t *lo, const std::uint32_t *hi, std::uint32_t *out_sums,
    std::uint32_t *out_counts, std::size_t query_count, SpineView spine,
    const std::uint32_t *spine_value_prefix,
    const std::uint32_t *spine_dead_count_prefix,
    const std::uint32_t *spine_dead_value_prefix,
    const EpochView *epochs, int epoch_count,
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
  std::uint32_t sum = spine_range_sum(
      spine, spine_value_prefix, spine_dead_value_prefix, l, h);
  if constexpr (HasEpochs) {
    for (int e = 0; e < epoch_count; ++e)
      sum += epoch_range_sum_one(epochs[e], l, h);
  }
  if constexpr (HasOverlay) {
    sum += overlay_prefix_range(ins_prefix, ins_keys, ins_count, l, h);
    sum -= overlay_prefix_range(tomb_val_prefix, tomb_keys, tomb_count, l, h);
  }
  out_sums[i] = sum;
  if (out_counts) {
    std::uint32_t c =
        spine_range_count(spine, spine_dead_count_prefix, l, h);
    if constexpr (HasEpochs) {
      for (int e = 0; e < epoch_count; ++e)
        c += epoch_range_count_one(epochs[e], l, h);
    }
    if constexpr (HasOverlay) {
      c += overlay_count_range(ins_keys, ins_count, l, h);
      c -= overlay_prefix_range(tomb_cnt_prefix, tomb_keys, tomb_count, l, h);
    }
    out_counts[i] = c;
  }
}

template <bool HasOverlay, bool HasEpochs>
__global__ void count_query_kernel(
    const std::uint32_t *lo, const std::uint32_t *hi, std::uint32_t *out_counts,
    std::size_t query_count, SpineView spine,
    const std::uint32_t *spine_dead_count_prefix,
    const EpochView *epochs, int epoch_count,
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
  std::uint32_t c =
      spine_range_count(spine, spine_dead_count_prefix, l, h);
  if constexpr (HasEpochs) {
    for (int e = 0; e < epoch_count; ++e)
      c += epoch_range_count_one(epochs[e], l, h);
  }
  if constexpr (HasOverlay) {
    c += overlay_count_range(ins_keys, ins_count, l, h);
    c -= overlay_prefix_range(tomb_cnt_prefix, tomb_keys, tomb_count, l, h);
  }
  out_counts[i] = c;
}

}

class GPULSMOpt {
public:
  using DeviceKeyBatch = gpulsmopt_detail::DeviceKeyBatch;

  explicit GPULSMOpt(const DictionaryConfig &config)
      : max_elements_(config.max_elements),
        batch_capacity_(config.batch_capacity),
        delete_order_policy_(config.delete_order) {
    if (max_elements_ > 0x7fffffffu) {
      throw std::invalid_argument(
          "GPULSMOpt currently supports at most 2^31-1 records");
    }
    epochs_.reserve(gpulsmopt_detail::kEpochMax);
    epoch_pool_.reserve(gpulsmopt_detail::kEpochMax);
    epoch_views_.reserve(gpulsmopt_detail::kEpochMax);
    epoch_views_.resize(gpulsmopt_detail::kEpochMax);
    bound_epoch_views_.resize(gpulsmopt_detail::kEpochMax);
    bound_epoch_view_valid_.resize(gpulsmopt_detail::kEpochMax);
    try {
      CUDA_CHECK(cudaMallocHost(
          reinterpret_cast<void **>(&epoch_removed_host_),
          (gpulsmopt_detail::kEpochMax + 1) * sizeof(std::uint32_t)));
      CUDA_CHECK(cudaMallocHost(
          reinterpret_cast<void **>(&delete_layer_order_host_),
          gpulsmopt_detail::kDeleteLayerMax * sizeof(std::uint8_t)));
      CUDA_CHECK(cudaEventCreateWithFlags(
          &epoch_delete_ready_, cudaEventDisableTiming));
    } catch (...) {
      if (epoch_delete_ready_)
        cudaEventDestroy(epoch_delete_ready_);
      if (epoch_removed_host_)
        cudaFreeHost(epoch_removed_host_);
      if (delete_layer_order_host_)
        cudaFreeHost(delete_layer_order_host_);
      throw;
    }
  }

  ~GPULSMOpt() {
    if (epoch_delete_pending_ && epoch_delete_ready_)
      cudaEventSynchronize(epoch_delete_ready_);
    if (epoch_delete_ready_)
      cudaEventDestroy(epoch_delete_ready_);
    if (epoch_removed_host_)
      cudaFreeHost(epoch_removed_host_);
    if (delete_layer_order_host_)
      cudaFreeHost(delete_layer_order_host_);
  }

  void clear(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    reap_pending_deletes(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    clear_epoch_state();
    clear_spine_state();
    invalidate_delete_layer_order();
    live_count_ = 0;
    recycle_active_runs();
    clear_c0_log(stream);
    invalidate_overlay_records();
  }

  void insert(const DeviceKeyValueBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    {
      std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
      reap_pending_deletes(stream);
      invalidate_delete_layer_order();
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
      const double measured = prof_append_ms_ + prof_flushsort_ms_ +
                              prof_runmerge_ms_ + prof_resolve_ms_ +
                              prof_delete_ms_ + prof_epoch_ingest_ms_ +
                              prof_delta_sort_ms_ +
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
      printf("[prof]   epoch_ingest = %.3f ms (%5.1f%%)\n",
             prof_epoch_ingest_ms_, pct(prof_epoch_ingest_ms_));
      printf("[prof]   other/host  = %.3f ms (%5.1f%%)\n", other, pct(other));
#endif
    }
  }

  void erase(const DeviceKeyBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    {
      std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
      reap_pending_deletes(stream);
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
      printf("[prof] delete %zu keys: total=%.3f ms "
             "(flush_sort=%.3f resolve=%.3f tomb=%.3f epoch=%.3f)\n",
             batch.count, total, prof_flushsort_ms_, prof_resolve_ms_,
             prof_delete_ms_, prof_epoch_ingest_ms_);
#endif
    }
  }

  void finish_pending_delete(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    reap_pending_deletes(stream);
  }

  void lookup(const DeviceLookupBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    std::unique_lock<std::shared_mutex> exclusive(snapshot_mutex_,
                                                   std::defer_lock);
    if (epoch_delete_pending_) {
      guard.unlock();
      exclusive.lock();
      reap_pending_deletes(stream);
    }
    const bool has_overlay = c0_log_count_ != 0 || !runs_.empty();
    if (!spine_micro_ready_ || epoch_heavy_pending() ||
        (has_overlay && overlay_dirty_)) {
      if (!exclusive.owns_lock()) {
        guard.unlock();
        exclusive.lock();
      }
      ensure_spine_microdirectory(stream);
      ensure_epoch_heavy_sorted(stream);
      if (has_overlay)
        resolved_overlay(stream);
    }
    lookup_layered(batch, stream);
  }

  void count(const DeviceRangeBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    std::unique_lock<std::shared_mutex> exclusive(snapshot_mutex_,
                                                   std::defer_lock);
    if (epoch_delete_pending_) {
      guard.unlock();
      exclusive.lock();
      reap_pending_deletes(stream);
    }
    const bool no_overlay = c0_log_count_ == 0 && runs_.empty();
    const bool overlay_pending =
        !no_overlay &&
        (overlay_dirty_ || !cached_overlay_.count_prefix_ready);
    if (!spine_micro_ready_ || epoch_heavy_pending() ||
        epoch_count_prefix_pending() ||
        (spine_has_dead_ && !spine_dead_count_prefix_ready_) ||
        overlay_pending) {
      if (!exclusive.owns_lock()) {
        guard.unlock();
        exclusive.lock();
      }
      ensure_spine_microdirectory(stream);
      ensure_epoch_heavy_sorted(stream);
      ensure_epoch_count_prefixes(stream);
      ensure_spine_dead_count_prefix(stream);
      if (!no_overlay) {
        auto &overlay = resolved_overlay(stream);
        ensure_overlay_count_prefix(overlay, stream);
      }
    }
    const OverlayReadIndex *ix =
        no_overlay ? nullptr : &resolved_overlay(stream);
    const std::uint32_t *ins_keys = no_overlay ? nullptr : raw_or_null(ix->gk);
    const std::uint32_t *tomb_keys =
        no_overlay ? nullptr : raw_or_null(ix->gk) + ix->ins;
    const std::uint32_t *tomb_cnt =
        no_overlay ? nullptr : raw_or_null(ix->tomb_cnt_prefix);
    const int block = 128;
    const int grid = static_cast<int>((batch.count + block - 1) / block);
    const auto spine = make_spine_view();
    gpulsmopt_detail::count_query_kernel<true, true>
        <<<grid, block, 0, stream>>>(
            batch.lo, batch.hi, batch.out_counts, batch.count, spine,
            spine_dead_count_prefix_.data(), epoch_view_ptr(), epoch_count(),
            ins_keys, no_overlay ? 0 : ix->ins, tomb_keys, tomb_cnt,
            no_overlay ? 0 : ix->u - ix->ins);
    CUDA_CHECK(cudaGetLastError());
  }

  void successor(const DeviceSuccessorBatch &batch, cudaStream_t stream) {
    if (batch.count == 0)
      return;
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    std::unique_lock<std::shared_mutex> exclusive(snapshot_mutex_,
                                                   std::defer_lock);
    if (epoch_delete_pending_) {
      guard.unlock();
      exclusive.lock();
      reap_pending_deletes(stream);
    }
    const bool no_overlay = c0_log_count_ == 0 && runs_.empty();
    const bool overlay_pending =
        !no_overlay &&
        (overlay_dirty_ || !cached_overlay_.successor_ready);
    if (!spine_micro_ready_ || epoch_heavy_pending() ||
        epoch_bitmap_pending() || overlay_pending) {
      if (!exclusive.owns_lock()) {
        guard.unlock();
        exclusive.lock();
      }
      ensure_spine_microdirectory(stream);
      ensure_epoch_heavy_sorted(stream);
      ensure_epoch_bitmaps(stream);
      if (!no_overlay) {
        auto &overlay = resolved_overlay(stream);
        ensure_overlay_successor(overlay, stream);
      }
    }
    const OverlayReadIndex *ix =
        no_overlay ? nullptr : &resolved_overlay(stream);
    const std::uint32_t *killed =
        no_overlay ? nullptr : raw_or_null(ix->killed_keys);
    const std::size_t killed_count = no_overlay ? 0 : ix->killed_count;
    const std::uint32_t *live_ins =
        no_overlay ? nullptr : raw_or_null(ix->live_ins_keys);
    const std::size_t live_ins_count = no_overlay ? 0 : ix->live_ins_count;
    const int block = 128;
    const int grid = static_cast<int>((batch.count + block - 1) / block);
    gpulsmopt_detail::successor_index_kernel<true, true>
        <<<grid, block, 0, stream>>>(
            batch.queries, batch.count, batch.out_keys, make_spine_view(),
            epoch_view_ptr(), epoch_count(), killed, killed_count, live_ins,
            live_ins_count);
    CUDA_CHECK(cudaGetLastError());
  }

  void range(const DeviceRangeOutputBatch &batch, cudaStream_t stream) {
    if (batch.query_count == 0)
      return;
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    std::unique_lock<std::shared_mutex> exclusive(snapshot_mutex_,
                                                   std::defer_lock);
    if (epoch_delete_pending_) {
      guard.unlock();
      exclusive.lock();
      reap_pending_deletes(stream);
    }
    const bool no_overlay = c0_log_count_ == 0 && runs_.empty();
    const bool overlay_pending =
        !no_overlay &&
        (overlay_dirty_ || !cached_overlay_.value_prefix_ready ||
         (batch.out_counts && !cached_overlay_.count_prefix_ready));
    if (!spine_micro_ready_ || epoch_heavy_pending() ||
        epoch_value_prefix_pending() ||
        (batch.out_counts && epoch_count_prefix_pending()) ||
        (spine_has_dead_ &&
         (!spine_dead_value_prefix_ready_ ||
          (batch.out_counts && !spine_dead_count_prefix_ready_))) ||
        overlay_pending) {
      if (!exclusive.owns_lock()) {
        guard.unlock();
        exclusive.lock();
      }
      ensure_spine_microdirectory(stream);
      ensure_epoch_heavy_sorted(stream);
      ensure_epoch_value_prefixes(stream);
      if (batch.out_counts)
        ensure_epoch_count_prefixes(stream);
      ensure_spine_dead_value_prefix(stream);
      if (batch.out_counts)
        ensure_spine_dead_count_prefix(stream);
      if (!no_overlay) {
        auto &overlay = resolved_overlay(stream);
        ensure_overlay_value_prefixes(overlay, stream);
        if (batch.out_counts)
          ensure_overlay_count_prefix(overlay, stream);
      }
    }
    const OverlayReadIndex *ix =
        no_overlay ? nullptr : &resolved_overlay(stream);
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
    const auto spine = make_spine_view();
    gpulsmopt_detail::range_query_kernel<true, true>
        <<<grid, block, 0, stream>>>(
            batch.lo, batch.hi, batch.out_sums, batch.out_counts,
            batch.query_count, spine, spine_value_prefix_.data(),
            spine_dead_count_prefix_.data(),
            spine_dead_value_prefix_.data(), epoch_view_ptr(), epoch_count(),
            ins_keys, ins_prefix, no_overlay ? 0 : ix->ins, tomb_keys,
            tomb_val, tomb_cnt, no_overlay ? 0 : ix->u - ix->ins);
    CUDA_CHECK(cudaGetLastError());
  }

  void consolidate(cudaStream_t stream) {
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    reap_pending_deletes(stream);
    invalidate_delete_layer_order();
    merge_down(stream);
    consolidate_all_epochs(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void bulk_build(const std::uint32_t *keys, const std::uint32_t *values,
                  std::size_t n, cudaStream_t stream) {
    if (n == 0)
      return;
    std::unique_lock<std::shared_mutex> guard(snapshot_mutex_);
    reap_pending_deletes(stream);
    invalidate_delete_layer_order();
    sort_direct_batch(keys, values, n, stream);
    std::swap(spine_keys_, direct_sort_keys_);
    std::swap(spine_values_, direct_sort_values_);
    spine_count_ = n;
    build_spine_metadata(stream);
    ensure_spine_microdirectory(stream);
    live_count_ = n;
    prepare_for_insert(stream);
    invalidate_overlay_records();
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  std::size_t live_count() const {
    auto *self = const_cast<GPULSMOpt *>(this);
    std::unique_lock<std::shared_mutex> guard(self->snapshot_mutex_);
    if (self->epoch_delete_pending_) {
      const cudaStream_t stream = self->epoch_delete_stream_;
      self->reap_pending_deletes(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return self->live_count_;
  }

  std::size_t gpu_resident_bytes() const {
    std::shared_lock<std::shared_mutex> guard(snapshot_mutex_);
    std::size_t total = device_bytes_all(
        epoch_views_, epoch_removed_,
        c0_log_keys_, c0_log_values_, c0_log_ops_, direct_sort_keys_,
        direct_sort_values_,
        sort_key_output_, sort_payload_input_, sort_payload_output_,
        sort_temp_storage_, epoch_merge_keys_, epoch_merge_values_,
        epoch_quotient_counts_, epoch_quotient_offsets_,
        epoch_quotient_cursor_,
        spine_keys_, spine_values_, spine_value_prefix_, spine_radix_,
        spine_micro_base_, spine_micro_offsets_, spine_micro_bits_,
        spine_dead_words_, spine_dead_count_sum_, spine_dead_value_sum_,
        spine_dead_count_prefix_, spine_dead_value_prefix_,
        spine_live_counts_, spine_live_prefix_, spine_gather_keys_,
        spine_gather_values_, spine_merge_keys_, spine_merge_values_,
        delete_sort_keys_, delete_layer_stats_, delete_layer_order_, ov_c0k_,
        ov_c0v_, ov_c0op_, ov_mk_, ov_mv_,
        ov_mop_, overlay_tomb_values_, overlay_tomb_flags_,
        overlay_live_flags_, cached_overlay_.gk, cached_overlay_.gv,
        cached_overlay_.gop, cached_overlay_.ins_prefix,
        cached_overlay_.tomb_val_prefix,
        cached_overlay_.tomb_cnt_prefix, cached_overlay_.live_ins_keys,
        cached_overlay_.killed_keys);
    for (const auto &g : runs_)
      total += device_bytes_all(g.keys, g.values, g.ops);
    for (const auto &g : run_buffer_pool_)
      total += device_bytes_all(g.keys, g.values, g.ops);
    for (const auto &epoch : epochs_)
      total += device_bytes_all(
          epoch.keys, epoch.values, epoch.dead,
          epoch.quotient_off, epoch.subgroup_masks, epoch.quotient_live,
          epoch.quotient_value_sum, epoch.quotient_count_prefix,
          epoch.quotient_value_prefix, epoch.subgroup_value_prefix,
          epoch.quotient_bitmap,
          epoch.heavy_list, epoch.heavy_count);
    for (const auto &epoch : epoch_pool_)
      total += device_bytes_all(
          epoch.keys, epoch.values, epoch.dead,
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
            spine_has_dead_ ? spine_dead_words_.data() : nullptr,
            spine_count_};
  }

  gpulsmopt_detail::SpineView make_spine_delete_view() const {
    auto view = make_spine_view();
    view.dead = spine_dead_words_.data();
    return view;
  }

  void clear_spine_state() {
    spine_count_ = 0;
    spine_live_count_ = 0;
    spine_has_dead_ = false;
    spine_dead_count_prefix_ready_ = true;
    spine_dead_value_prefix_ready_ = true;
    spine_keys_.resize_discard(0);
    spine_values_.resize_discard(0);
    spine_value_prefix_.resize_discard(0);
    spine_radix_.resize_discard(0);
    spine_micro_base_.resize_discard(0);
    spine_micro_offsets_.resize_discard(0);
    spine_micro_bits_.resize_discard(0);
    spine_micro_ready_ = false;
    spine_dead_words_.resize_discard(0);
    spine_dead_count_sum_.resize_discard(0);
    spine_dead_value_sum_.resize_discard(0);
    spine_dead_count_prefix_.resize_discard(0);
    spine_dead_value_prefix_.resize_discard(0);
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
    const std::size_t words = (count + 31) / 32;
    spine_dead_words_.resize_discard(words);
    if (words > 0)
      CUDA_CHECK(cudaMemsetAsync(spine_dead_words_.data(), 0,
                                 words * sizeof(std::uint32_t), stream));
    const std::size_t dead_blocks =
        (count + gpulsmopt_detail::kSpineDeadBlockSize - 1) /
        gpulsmopt_detail::kSpineDeadBlockSize;
    spine_dead_count_sum_.resize_discard(dead_blocks + 1);
    spine_dead_value_sum_.resize_discard(dead_blocks + 1);
    spine_dead_count_prefix_.resize_discard(dead_blocks + 1);
    spine_dead_value_prefix_.resize_discard(dead_blocks + 1);
    CUDA_CHECK(cudaMemsetAsync(
        spine_dead_count_sum_.data(), 0,
        (dead_blocks + 1) * sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        spine_dead_value_sum_.data(), 0,
        (dead_blocks + 1) * sizeof(std::uint32_t), stream));
    spine_live_count_ = count;
    spine_has_dead_ = false;
    spine_dead_count_prefix_ready_ = true;
    spine_dead_value_prefix_ready_ = true;
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

  void ensure_spine_dead_count_prefix(cudaStream_t stream) {
    if (!spine_has_dead_ || spine_dead_count_prefix_ready_)
      return;
    const std::size_t blocks = spine_dead_count_sum_.size();
    exclusive_scan_u32(spine_dead_count_sum_.data(),
                       spine_dead_count_prefix_.data(), blocks, stream);
    spine_dead_count_prefix_ready_ = true;
  }

  void ensure_spine_dead_value_prefix(cudaStream_t stream) {
    if (!spine_has_dead_ || spine_dead_value_prefix_ready_)
      return;
    const std::size_t blocks = spine_dead_value_sum_.size();
    exclusive_scan_u32(spine_dead_value_sum_.data(),
                       spine_dead_value_prefix_.data(), blocks, stream);
    spine_dead_value_prefix_ready_ = true;
  }

  struct EpochStorage {
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> keys;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> values;
    gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> dead;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_off;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> subgroup_masks;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_live;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_value_sum;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_count_prefix;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_value_prefix;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> subgroup_value_prefix;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> quotient_bitmap;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> heavy_list;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> heavy_count;
    std::size_t count = 0;
    std::size_t live_count = 0;
    bool has_dead = false;
    bool dead_initialized = false;
    bool heavy_sorted = true;
    bool value_sums_ready = true;
    bool value_prefix_ready = true;
    bool subgroup_value_prefix_ready = true;
    bool count_prefix_ready = true;
    bool bitmap_ready = true;
  };

  const gpulsmopt_detail::EpochView *epoch_view_ptr() const {
    return raw_or_null(epoch_views_);
  }

  int epoch_count() const { return static_cast<int>(epochs_.size()); }

  gpulsmopt_detail::EpochView make_epoch_view(EpochStorage &epoch) {
    return {epoch.keys.data(),
            epoch.values.data(),
            epoch.dead.data(),
            epoch.quotient_off.data(),
            epoch.subgroup_masks.data(),
            epoch.quotient_live.data(),
            epoch.quotient_value_sum.data(),
            epoch.quotient_count_prefix.data(),
            epoch.quotient_value_prefix.data(),
            epoch.subgroup_value_prefix_ready
                ? epoch.subgroup_value_prefix.data()
                : nullptr,
            epoch.quotient_bitmap.data(),
            epoch.heavy_list.data(),
            epoch.heavy_count.data(),
            epoch.has_dead ? 1u : 0u};
  }

  static bool same_epoch_view(const gpulsmopt_detail::EpochView &a,
                              const gpulsmopt_detail::EpochView &b) {
    return a.keys == b.keys && a.values == b.values && a.dead == b.dead &&
           a.quotient_off == b.quotient_off &&
           a.subgroup_masks == b.subgroup_masks &&
           a.quotient_live == b.quotient_live &&
           a.quotient_value_sum == b.quotient_value_sum &&
           a.quotient_count_prefix == b.quotient_count_prefix &&
           a.quotient_value_prefix == b.quotient_value_prefix &&
           a.subgroup_value_prefix == b.subgroup_value_prefix &&
           a.quotient_bitmap == b.quotient_bitmap &&
           a.heavy_list == b.heavy_list &&
           a.heavy_count == b.heavy_count &&
           a.has_dead == b.has_dead;
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
      auto view = make_epoch_view(*it);
      view.has_dead = 0u;
      bound_epoch_views_[index] = view;
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
    epoch.has_dead = false;
    epoch.dead_initialized = false;
    epoch.heavy_sorted = false;
    epoch.value_sums_ready = false;
    epoch.value_prefix_ready = false;
    epoch.subgroup_value_prefix_ready = false;
    epoch.count_prefix_ready = true;
    epoch.bitmap_ready = false;
    epoch.dead.resize_discard(epoch.count);
    epoch.quotient_off.resize_discard(
        gpulsmopt_detail::kEpochQuotients + 1);
    epoch.subgroup_masks.resize_discard(
        gpulsmopt_detail::kEpochQuotients *
        gpulsmopt_detail::kEpochSubgroupPlanes);
    epoch.quotient_live.resize_discard(
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

  void commit_epoch_metadata(EpochStorage &epoch, cudaStream_t stream) {
    prepare_epoch_metadata_storage(epoch);
    launch_epoch_metadata_kernels(epoch, epoch.count, stream);
    append_epoch_view(epoch, stream);
    live_count_ += epoch.count;
    invalidate_overlay_derivatives();
  }

  bool epoch_heavy_pending() const {
    for (const auto &epoch : epochs_)
      if (!epoch.heavy_sorted)
        return true;
    return false;
  }

  bool epoch_value_prefix_pending() const {
    for (const auto &epoch : epochs_)
      if (!epoch.value_prefix_ready ||
          !epoch.subgroup_value_prefix_ready)
        return true;
    return false;
  }

  bool epoch_bitmap_pending() const {
    for (const auto &epoch : epochs_)
      if (!epoch.bitmap_ready)
        return true;
    return false;
  }

  bool epoch_count_prefix_pending() const {
    for (const auto &epoch : epochs_)
      if (epoch.has_dead && !epoch.count_prefix_ready)
        return true;
    return false;
  }

  void ensure_epoch_heavy_sorted(cudaStream_t stream) {
    constexpr int block = 256;
    constexpr int classify_grid =
        gpulsmopt_detail::kEpochQuotients / block;
    constexpr int shards = 64;
    for (auto &epoch : epochs_) {
      if (epoch.heavy_sorted)
        continue;
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
      if (!epoch.has_dead || epoch.count_prefix_ready)
        continue;
      exclusive_scan_u32(epoch.quotient_live.data(),
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

  void create_sorted_epoch(const std::uint32_t *keys,
                           const std::uint32_t *values, std::size_t count,
                           cudaStream_t stream) {
    if (epochs_.size() >= static_cast<std::size_t>(gpulsmopt_detail::kEpochMax))
      consolidate_all_epochs(stream);
    acquire_epoch_slot();
    EpochStorage &epoch = epochs_.back();
    epoch.count = count;
    epoch.live_count = count;
    epoch.keys.resize_discard(count);
    epoch.values.resize_discard(count);
    CUDA_CHECK(cudaMemcpyAsync(epoch.keys.data(), keys,
                               count * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(epoch.values.data(), values,
                               count * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToDevice, stream));
    commit_epoch_metadata(epoch, stream);
  }

  void create_unsorted_epoch(const std::uint32_t *keys,
                             const std::uint32_t *values, std::size_t count,
                             cudaStream_t stream) {
    if (epochs_.size() >= static_cast<std::size_t>(gpulsmopt_detail::kEpochMax))
      consolidate_all_epochs(stream);
    acquire_epoch_slot();
    EpochStorage &epoch = epochs_.back();
    epoch.count = count;
    epoch.live_count = count;
    epoch.keys.resize_discard(count);
    epoch.values.resize_discard(count);
    {
      GPULSMOPT_PROF_PHASE(prof_delta_sort_ms_);
      sort_epoch_batch(keys, values, count, epoch.keys.data(),
                       epoch.values.data(), stream);
    }
    {
      GPULSMOPT_PROF_PHASE(prof_delta_ingest_ms_);
      commit_epoch_metadata(epoch, stream);
    }
  }

  static std::uint32_t host_delete_probe_work(std::size_t count) {
    const std::uint32_t bounded = static_cast<std::uint32_t>(count);
    if (bounded <= 32u)
      return 6u + (bounded + 15u) / 16u;
    if (bounded <= gpulsmopt_detail::kEpochHeavySortCap) {
      std::uint32_t value = bounded - 1u;
      std::uint32_t bits = 0u;
      while (value != 0u) {
        value >>= 1;
        ++bits;
      }
      return 4u + bits;
    }
    return 2u + (bounded < 4096u ? bounded : 4096u);
  }

  void invalidate_delete_layer_order() {
    delete_layer_order_host_count_ = 0;
    delete_layer_order_host_valid_ = false;
    delete_layer_order_device_count_ = 0;
    delete_layer_order_device_matches_host_ = false;
  }

  void make_delete_layer_order(const std::uint32_t *weights,
                               std::size_t layer_count,
                               std::uint8_t *order) const {
    const std::size_t ecount = layer_count - 1;
    std::uint32_t work[gpulsmopt_detail::kDeleteLayerMax]{};
    for (std::size_t epoch = 0; epoch < ecount; ++epoch) {
      const std::size_t average =
          (epochs_[epoch].count + gpulsmopt_detail::kEpochQuotients - 1) /
          gpulsmopt_detail::kEpochQuotients;
      work[epoch] = host_delete_probe_work(average);
      order[epoch] = static_cast<std::uint8_t>(ecount - epoch - 1);
    }
    work[ecount] = spine_count_ == 0 ? 1u :
                   (spine_micro_ready_ ? 8u : 12u);
    order[ecount] = static_cast<std::uint8_t>(ecount);
    for (std::size_t rank = 1; rank < layer_count; ++rank) {
      const std::uint8_t layer = order[rank];
      std::size_t position = rank;
      while (position > 0) {
        const std::uint8_t previous = order[position - 1];
        const unsigned long long left =
            static_cast<unsigned long long>(weights[layer]) * work[previous];
        const unsigned long long right =
            static_cast<unsigned long long>(weights[previous]) * work[layer];
        if (weights[layer] == 0u || left <= right)
          break;
        order[position] = previous;
        --position;
      }
      order[position] = layer;
    }
  }

  bool store_delete_layer_order(const std::uint8_t *order,
                                std::size_t layer_count) {
    if (!delete_layer_order_host_ || layer_count < 2 ||
        layer_count > gpulsmopt_detail::kDeleteLayerMax) {
      invalidate_delete_layer_order();
      return false;
    }
    bool same = delete_layer_order_host_count_ == layer_count;
    for (std::size_t rank = 0; same && rank < layer_count; ++rank)
      same = delete_layer_order_host_[rank] == order[rank];
    if (!same) {
      for (std::size_t rank = 0; rank < layer_count; ++rank)
        delete_layer_order_host_[rank] = order[rank];
      delete_layer_order_device_matches_host_ = false;
    }
    delete_layer_order_host_count_ = layer_count;
    const std::size_t ecount = layer_count - 1;
    bool differs = false;
    for (std::size_t rank = 0; rank < layer_count; ++rank) {
      const std::uint8_t fallback = static_cast<std::uint8_t>(
          rank < ecount ? ecount - rank - 1 : ecount);
      differs |= order[rank] != fallback;
    }
    delete_layer_order_host_valid_ = differs;
    return differs;
  }

  const std::uint8_t *upload_delete_layer_order(cudaStream_t stream) {
    if (!delete_layer_order_host_valid_)
      return nullptr;
    delete_layer_order_.resize_discard(gpulsmopt_detail::kDeleteLayerMax);
    if (!delete_layer_order_device_matches_host_ ||
        delete_layer_order_device_count_ != delete_layer_order_host_count_) {
      CUDA_CHECK(cudaMemcpyAsync(
          delete_layer_order_.data(), delete_layer_order_host_,
          delete_layer_order_host_count_, cudaMemcpyHostToDevice, stream));
      delete_layer_order_device_count_ = delete_layer_order_host_count_;
      delete_layer_order_device_matches_host_ = true;
    }
    return delete_layer_order_.data();
  }

  const std::uint8_t *prepare_uniform_live_delete_order(
      cudaStream_t stream) {
    const std::size_t ecount = epochs_.size();
    const std::size_t layer_count = ecount + 1;
    std::uint32_t live[gpulsmopt_detail::kDeleteLayerMax]{};
    std::uint8_t order[gpulsmopt_detail::kDeleteLayerMax]{};
    for (std::size_t epoch = 0; epoch < ecount; ++epoch)
      live[epoch] = static_cast<std::uint32_t>(epochs_[epoch].live_count);
    live[ecount] = static_cast<std::uint32_t>(spine_live_count_);
    make_delete_layer_order(live, layer_count, order);
    if (!store_delete_layer_order(order, layer_count))
      return nullptr;
    return upload_delete_layer_order(stream);
  }

  static int delete_owner_bits(std::size_t count) {
    const std::size_t target = GPULSMOPT_DELETE_RANGE_TARGET;
    const std::size_t desired = (count + target - 1) / target;
    int bits = gpulsmopt_detail::kDeleteOwnerMinBits;
    while (bits < gpulsmopt_detail::kEpochQuotientBits &&
           (std::size_t{1} << bits) < desired) {
      ++bits;
    }
    return bits;
  }

  void cache_delete_layer_order(
      const std::vector<std::uint32_t> &owner_hits) {
    const std::size_t ecount = epochs_.size();
    const std::size_t layer_count = ecount + 1;
    if (!delete_layer_order_host_ || owner_hits.size() != layer_count ||
        layer_count > gpulsmopt_detail::kDeleteLayerMax) {
      invalidate_delete_layer_order();
      return;
    }
    std::uint8_t order[gpulsmopt_detail::kDeleteLayerMax]{};
    make_delete_layer_order(owner_hits.data(), layer_count, order);
    store_delete_layer_order(order, layer_count);
  }

  void begin_epoch_deletes(const std::uint32_t *keys, std::size_t count,
                           bool keys_sorted, cudaStream_t stream) {
    if (count == 0)
      return;
    reap_pending_deletes(stream);
    if (!epochs_.empty()) {
      ensure_epoch_heavy_sorted(stream);
      ensure_epoch_bitmaps(stream);
    }
    const std::size_t ecount = epochs_.size();
    for (auto &epoch : epochs_) {
      if (epoch.dead_initialized)
        continue;
      if (epoch.count > 0)
        CUDA_CHECK(cudaMemsetAsync(epoch.dead.data(), 0, epoch.count, stream));
      epoch.dead_initialized = true;
    }
    resize_reuse(epoch_removed_, ecount + 1);
    CUDA_CHECK(cudaMemsetAsync(raw_or_null(epoch_removed_), 0,
                               (ecount + 1) * sizeof(std::uint32_t), stream));
    std::uint32_t value_sums_ready_mask = 0u;
    for (std::size_t e = 0; e < ecount; ++e)
      if (epochs_[e].value_sums_ready)
        value_sums_ready_mask |= 1u << e;
    const std::uint32_t *delete_keys = keys;
    if (!keys_sorted) {
      sort_delete_batch(keys, count, stream);
      delete_keys = delete_sort_keys_.data();
    }
    const std::uint8_t *layer_order = nullptr;
    const std::size_t live_layers =
        ecount + static_cast<std::size_t>(spine_live_count_ != 0);
    if (delete_order_policy_ == DeleteOrderPolicy::uniform_live &&
        live_layers > 1) {
      layer_order = prepare_uniform_live_delete_order(stream);
    } else if (count >= GPULSMOPT_DELETE_ADAPTIVE_MIN_BATCH &&
               live_layers > 1) {
      constexpr std::size_t stat_count =
          2 * gpulsmopt_detail::kDeleteLayerMax + 1;
      delete_layer_stats_.resize_discard(stat_count);
      delete_layer_order_.resize_discard(gpulsmopt_detail::kDeleteLayerMax);
      CUDA_CHECK(cudaMemsetAsync(delete_layer_stats_.data(), 0,
                                 stat_count * sizeof(std::uint32_t), stream));
      const std::size_t samples = std::min<std::size_t>(
          count, GPULSMOPT_DELETE_OWNER_SAMPLES);
      const int sample_grid = static_cast<int>((samples + 255) / 256);
      gpulsmopt_detail::delete_layer_order_kernel<<<sample_grid, 256, 0,
                                                    stream>>>(
          delete_keys, count, epoch_view_ptr(), epoch_count(),
          make_spine_delete_view(), delete_layer_stats_.data(),
          delete_layer_stats_.data() + gpulsmopt_detail::kDeleteLayerMax,
          delete_layer_stats_.data() +
              2 * gpulsmopt_detail::kDeleteLayerMax,
          delete_layer_order_.data());
      CUDA_CHECK(cudaGetLastError());
      delete_layer_order_device_matches_host_ = false;
      layer_order = delete_layer_order_.data();
    } else if (live_layers > 1 && delete_layer_order_host_valid_ &&
               delete_layer_order_host_count_ == ecount + 1) {
      layer_order = upload_delete_layer_order(stream);
    }
    constexpr int block = 256;
    if (delete_order_policy_ == DeleteOrderPolicy::uniform_live &&
        count >= GPULSMOPT_DELETE_RANGE_MIN_BATCH) {
      const int owner_bits = delete_owner_bits(count);
      const std::uint32_t owner_count = 1u << owner_bits;
      constexpr int warps = block / 32;
      const int grid = static_cast<int>((owner_count + warps - 1) / warps);
      gpulsmopt_detail::epoch_spine_owner_delete_kernel<<<
          grid, block, 0, stream>>>(
          delete_keys, static_cast<std::uint32_t>(count),
          epoch_view_ptr(), epoch_count(), layer_order,
          raw_or_null(epoch_removed_), value_sums_ready_mask,
          make_spine_delete_view(), spine_dead_count_sum_.data(),
          spine_dead_value_sum_.data(), owner_bits, owner_count);
    } else {
      const int grid = static_cast<int>((count + block - 1) / block);
      gpulsmopt_detail::epoch_spine_delete_kernel<<<grid, block, 0, stream>>>(
          delete_keys, count, epoch_view_ptr(), epoch_count(), layer_order,
          raw_or_null(epoch_removed_), value_sums_ready_mask,
          make_spine_delete_view(), spine_dead_count_sum_.data(),
          spine_dead_value_sum_.data(), raw_or_null(epoch_removed_) + ecount);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(epoch_removed_host_,
                               raw_or_null(epoch_removed_),
                               (ecount + 1) * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(epoch_delete_ready_, stream));
    epoch_delete_epoch_count_ = ecount;
    epoch_delete_stream_ = stream;
    epoch_delete_pending_ = true;
  }

  void finish_epoch_deletes(cudaStream_t stream, std::size_t ecount) {
    if (ecount != epochs_.size())
      throw std::runtime_error("epoch state changed during delete");
    const std::uint32_t spine_removed = epoch_removed_host_[ecount];
    bool epochs_changed = false;
    for (std::size_t i = 0; i < ecount; ++i) {
      if (epoch_removed_host_[i] == 0)
        continue;
      epochs_changed = true;
      EpochStorage &epoch = epochs_[i];
      if (epoch_removed_host_[i] > epoch.live_count)
        throw std::runtime_error("epoch delete count overflow");
      epoch.live_count -= epoch_removed_host_[i];
      if (epoch.live_count == 0)
        continue;
      epoch.value_prefix_ready = false;
      epoch.subgroup_value_prefix_ready = false;
      epoch.count_prefix_ready = false;
      epoch.bitmap_ready = true;
      epoch.has_dead = true;
    }
    const bool cache_observed_order =
        delete_order_policy_ == DeleteOrderPolicy::adaptive;
    std::vector<std::uint32_t> surviving_hits;
    if (cache_observed_order)
      surviving_hits.reserve(ecount + 1);
    std::size_t output = 0;
    for (std::size_t i = 0; i < epochs_.size(); ++i) {
      if (epochs_[i].live_count == 0) {
        epoch_pool_.push_back(std::move(epochs_[i]));
        continue;
      }
      if (cache_observed_order)
        surviving_hits.push_back(epoch_removed_host_[i]);
      if (output != i)
        epochs_[output] = std::move(epochs_[i]);
      ++output;
    }
    epochs_.resize(output);
    if (spine_removed > spine_live_count_)
      throw std::runtime_error("spine delete count overflow");
    if (spine_removed > 0) {
      spine_live_count_ -= spine_removed;
      spine_has_dead_ = true;
      spine_dead_count_prefix_ready_ = false;
      spine_dead_value_prefix_ready_ = false;
    }
    if (epochs_changed)
      refresh_active_epoch_views(stream);
    if (cache_observed_order) {
      surviving_hits.push_back(spine_removed);
      cache_delete_layer_order(surviving_hits);
    }
  }

  void reap_pending_deletes(cudaStream_t stream) {
    if (!epoch_delete_pending_)
      return;
    const cudaError_t status = cudaEventQuery(epoch_delete_ready_);
    if (status == cudaErrorNotReady) {
      CUDA_CHECK(cudaEventSynchronize(epoch_delete_ready_));
    } else {
      CUDA_CHECK(status);
    }
    finish_epoch_deletes(stream, epoch_delete_epoch_count_);
    recompute_live_count();
    epoch_delete_pending_ = false;
    epoch_delete_epoch_count_ = 0;
    epoch_delete_stream_ = nullptr;
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
           static_cast<std::size_t>(gpulsmopt_detail::kEpochMax)) {
      epoch_pool_.emplace_back();
    }
    for (auto &epoch : epoch_pool_) {
      epoch.keys.resize_discard(count);
      epoch.values.resize_discard(count);
      epoch.dead.resize_discard(count);
      epoch.quotient_off.resize_discard(
          gpulsmopt_detail::kEpochQuotients + 1);
      epoch.subgroup_masks.resize_discard(
          gpulsmopt_detail::kEpochQuotients *
          gpulsmopt_detail::kEpochSubgroupPlanes);
      epoch.quotient_live.resize_discard(
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
    resize_reuse(epoch_removed_, gpulsmopt_detail::kEpochMax + 1);
    epoch_quotient_counts_.resize_discard(
        gpulsmopt_detail::kEpochQuotients);
    epoch_quotient_offsets_.resize_discard(
        gpulsmopt_detail::kEpochQuotients + 1);
    epoch_quotient_cursor_.resize_discard(
        gpulsmopt_detail::kEpochQuotients);
    delete_layer_stats_.resize_discard(
        2 * gpulsmopt_detail::kDeleteLayerMax + 1);
    delete_layer_order_.resize_discard(gpulsmopt_detail::kDeleteLayerMax);
    const std::size_t pending_capacity =
        count * static_cast<std::size_t>(gpulsmopt_detail::kEpochMax);
    epoch_merge_keys_.resize_discard(pending_capacity);
    epoch_merge_values_.resize_discard(pending_capacity);
  }

  void clear_epoch_state() {
    for (auto &epoch : epochs_)
      epoch_pool_.push_back(std::move(epoch));
    epochs_.clear();
    epoch_removed_.clear();
  }

  void recompute_live_count() {
    std::size_t total = spine_live_count_;
    for (const auto &epoch : epochs_)
      total += epoch.live_count;
    live_count_ = total;
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
    std::uint32_t insert_total = 0;
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
    if (op == static_cast<std::uint8_t>(gpulsmopt_detail::kInsert))
      c0_insert_count_ += static_cast<std::uint32_t>(count);
    note_overlay_append(op, count);
    return true;
  }

  void clear_c0_log(cudaStream_t stream) {
    if (c0_log_count_ == 0)
      return;
    c0_log_count_ = 0;
    c0_insert_count_ = 0;
    (void)stream;
  }

  bool runs_should_drain() const {
    std::size_t total = live_count_ + c0_log_count_;
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
    if (runs_should_drain())
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
    run.insert_total = 0;
    run.level = level;
    return run;
  }

  void release_run_storage(SortedRun &&run) {
    run.log_total = 0;
    run.insert_total = 0;
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

  void prepare_overlay_storage(std::size_t capacity) {
    const std::size_t c0_capacity =
        std::min(capacity, c0_flush_budget());
    resize_reuse(ov_c0k_, c0_capacity);
    resize_reuse(ov_c0v_, c0_capacity);
    resize_reuse(ov_c0op_, c0_capacity);
    resize_reuse(ov_mk_, capacity);
    resize_reuse(ov_mv_, capacity);
    resize_reuse(ov_mop_, capacity);
    resize_reuse(cached_overlay_.gk, capacity);
    resize_reuse(cached_overlay_.gv, capacity);
    resize_reuse(cached_overlay_.gop, capacity);
    cached_overlay_.ins_prefix.resize_discard(capacity + 1);
    cached_overlay_.tomb_val_prefix.resize_discard(capacity + 1);
    cached_overlay_.tomb_cnt_prefix.resize_discard(capacity + 1);
    cached_overlay_.live_ins_keys.resize_discard(capacity);
    cached_overlay_.killed_keys.resize_discard(capacity);
    overlay_tomb_values_.resize_discard(capacity);
    overlay_tomb_flags_.resize_discard(capacity);
    overlay_live_flags_.resize_discard(capacity);
    cached_overlay_.ins_prefix.resize_discard(0);
    cached_overlay_.tomb_val_prefix.resize_discard(0);
    cached_overlay_.tomb_cnt_prefix.resize_discard(0);
    cached_overlay_.live_ins_keys.resize_discard(0);
    cached_overlay_.killed_keys.resize_discard(0);
    overlay_tomb_values_.resize_discard(0);
    overlay_tomb_flags_.resize_discard(0);
    overlay_live_flags_.resize_discard(0);
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
    sort_key_output_.resize_discard(log_count);
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
          sort_key_output_.data(), sort_payload_input_.data(),
          sort_payload_output_.data(), log_count, 0, 32, stream));
      log_sort_count_ = log_count;
      log_sort_temp_bytes_ = log_bytes;
    }
    std::size_t epoch_bytes = 0;
    std::size_t delete_bytes = 0;
    if (direct_count > 0) {
      CUDA_CHECK(gpulsmopt_detail::epoch_radix_sort_pairs(
          nullptr, epoch_bytes, spine_keys_.data(),
          direct_sort_keys_.data(), spine_values_.data(),
          direct_sort_values_.data(), static_cast<std::uint32_t>(direct_count),
          16, 32, stream));
      epoch_sort_count_ = direct_count;
      epoch_sort_temp_bytes_ = epoch_bytes;
      delete_sort_keys_.resize_discard(direct_count);
      CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
          nullptr, delete_bytes, spine_keys_.data(),
          delete_sort_keys_.data(), direct_count, 16, 32, stream));
      delete_sort_count_ = direct_count;
      delete_sort_temp_bytes_ = delete_bytes;
    }
    std::size_t scan_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, direct_sort_keys_.data(),
        direct_sort_values_.data(), gpulsmopt_detail::kEpochQuotients,
        stream));
    scan_u32_count_ = gpulsmopt_detail::kEpochQuotients;
    scan_u32_temp_bytes_ = scan_bytes;
    ensure_sort_temp(
        std::max({direct_bytes, log_bytes, epoch_bytes, delete_bytes,
                  scan_bytes}));
  }

  // Reserve direct-path buffers before timed updates.
  void prepare_for_insert(cudaStream_t stream) {
    const std::size_t direct_count = std::min(
        max_elements_, std::max(4 * c0_flush_budget(), batch_capacity_));
    ensure_c0_log(stream);
    prepare_sort_storage(direct_count, c0_flush_budget(), stream);
    reserve_epoch_storage(direct_count);
    prebind_epoch_views(stream);
    prepare_run_storage();
    prepare_overlay_storage(direct_count);
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

  void sort_delete_batch(const std::uint32_t *keys, std::size_t n,
                         cudaStream_t stream) {
    delete_sort_keys_.resize_discard(n);
    if (n > delete_sort_count_) {
      delete_sort_temp_bytes_ = 0;
      CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
          nullptr, delete_sort_temp_bytes_, keys, delete_sort_keys_.data(), n,
          16, 32, stream));
      delete_sort_count_ = n;
    }
    std::size_t temp_bytes = delete_sort_temp_bytes_;
    ensure_sort_temp(temp_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        sort_temp_storage_.data(), temp_bytes, keys, delete_sort_keys_.data(),
        n, 16, 32, stream));
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
    if (n > log_sort_count_) {
      log_sort_temp_bytes_ = 0;
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, log_sort_temp_bytes_, raw_or_null(k),
          sort_key_output_.data(), sort_payload_input_.data(),
          sort_payload_output_.data(), n, 0, 32, stream));
      log_sort_count_ = n;
    }
    std::size_t temp_bytes = log_sort_temp_bytes_;
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
    out.insert_total = a.insert_total + b.insert_total;
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
  }

  void sort_packed_epoch_quotients(std::size_t count,
                                   cudaStream_t stream) {
    if (count == 0)
      return;
    direct_sort_keys_.resize_discard(count);
    direct_sort_values_.resize_discard(count);
    if (count > segmented_sort_count_) {
      segmented_sort_temp_bytes_ = 0;
      CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
          nullptr, segmented_sort_temp_bytes_, epoch_merge_keys_.data(),
          direct_sort_keys_.data(), epoch_merge_values_.data(),
          direct_sort_values_.data(), static_cast<int>(count),
          gpulsmopt_detail::kEpochQuotients,
          epoch_quotient_offsets_.data(),
          epoch_quotient_offsets_.data() + 1, 0, 16, stream));
      segmented_sort_count_ = count;
    }
    std::size_t temp_bytes = segmented_sort_temp_bytes_;
    ensure_sort_temp(temp_bytes);
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
        sort_temp_storage_.data(), temp_bytes, epoch_merge_keys_.data(),
        direct_sort_keys_.data(), epoch_merge_values_.data(),
        direct_sort_values_.data(), static_cast<int>(count),
        gpulsmopt_detail::kEpochQuotients,
        epoch_quotient_offsets_.data(),
        epoch_quotient_offsets_.data() + 1, 0, 16, stream));
  }

  void consolidate_all_epochs(cudaStream_t stream) {
    if (epochs_.empty())
      return;
    std::size_t live_total = 0;
    for (const auto &epoch : epochs_)
      live_total += epoch.live_count;
    epoch_merge_keys_.resize_discard(live_total);
    epoch_merge_values_.resize_discard(live_total);
    if (live_total > 0) {
      epoch_quotient_counts_.resize_discard(
          gpulsmopt_detail::kEpochQuotients);
      epoch_quotient_offsets_.resize_discard(
          gpulsmopt_detail::kEpochQuotients + 1);
      epoch_quotient_cursor_.resize_discard(
          gpulsmopt_detail::kEpochQuotients);
      constexpr int metadata_block = 256;
      constexpr int metadata_grid =
          gpulsmopt_detail::kEpochQuotients / metadata_block;
      gpulsmopt_detail::epoch_combined_quotient_counts_kernel<<<
          metadata_grid, metadata_block, 0, stream>>>(
          epoch_view_ptr(), epoch_count(), epoch_quotient_counts_.data());
      CUDA_CHECK(cudaGetLastError());
      exclusive_scan_u32(epoch_quotient_counts_.data(),
                         epoch_quotient_offsets_.data(),
                         gpulsmopt_detail::kEpochQuotients, stream);
      gpulsmopt_detail::set_last_quotient_offset_kernel<<<1, 1, 0, stream>>>(
          epoch_quotient_offsets_.data(),
          static_cast<std::uint32_t>(live_total));
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpyAsync(
          epoch_quotient_cursor_.data(), epoch_quotient_offsets_.data(),
          gpulsmopt_detail::kEpochQuotients * sizeof(std::uint32_t),
          cudaMemcpyDeviceToDevice, stream));
      gpulsmopt_detail::epoch_pack_quotients_kernel<<<
          gpulsmopt_detail::kEpochQuotients, 128, 0, stream>>>(
          epoch_view_ptr(), epoch_count(), epoch_quotient_cursor_.data(),
          epoch_merge_keys_.data(), epoch_merge_values_.data());
      CUDA_CHECK(cudaGetLastError());
      sort_packed_epoch_quotients(live_total, stream);
    }
    const int block = 256;
    const std::uint32_t *base_keys = spine_keys_.data();
    const std::uint32_t *base_values = spine_values_.data();
    if (spine_has_dead_ && spine_count_ > 0) {
      spine_live_counts_.resize_discard(spine_count_);
      spine_live_prefix_.resize_discard(spine_count_);
      spine_gather_keys_.resize_discard(spine_live_count_);
      spine_gather_values_.resize_discard(spine_live_count_);
      const int grid =
          static_cast<int>((spine_count_ + block - 1) / block);
      gpulsmopt_detail::spine_live_input_kernel<<<grid, block, 0, stream>>>(
          spine_dead_words_.data(), spine_count_, spine_live_counts_.data());
      exclusive_scan_u32(spine_live_counts_.data(),
                         spine_live_prefix_.data(), spine_count_, stream);
      gpulsmopt_detail::spine_gather_live_kernel<<<grid, block, 0, stream>>>(
          spine_keys_.data(), spine_values_.data(), spine_dead_words_.data(),
          spine_live_prefix_.data(), spine_count_, spine_gather_keys_.data(),
          spine_gather_values_.data());
      CUDA_CHECK(cudaGetLastError());
      base_keys = spine_gather_keys_.data();
      base_values = spine_gather_values_.data();
    }
    const std::size_t merged_count = spine_live_count_ + live_total;
    if (spine_live_count_ == 0 && live_total > 0) {
      clear_epoch_state();
      std::swap(spine_keys_, direct_sort_keys_);
      std::swap(spine_values_, direct_sort_values_);
      spine_count_ = live_total;
      build_spine_metadata(stream);
      recompute_live_count();
      invalidate_overlay_derivatives();
      return;
    }
    spine_merge_keys_.resize_discard(merged_count);
    spine_merge_values_.resize_discard(merged_count);
    if (live_total == 0 && spine_live_count_ > 0) {
      CUDA_CHECK(cudaMemcpyAsync(
          spine_merge_keys_.data(), base_keys,
          spine_live_count_ * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice,
          stream));
      CUDA_CHECK(cudaMemcpyAsync(
          spine_merge_values_.data(), base_values,
          spine_live_count_ * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice,
          stream));
    } else if (merged_count > 0) {
      auto policy = thrust::cuda::par.on(stream);
      thrust::merge_by_key(
          policy, base_keys, base_keys + spine_live_count_,
          direct_sort_keys_.data(), direct_sort_keys_.data() + live_total,
          base_values, direct_sort_values_.data(), spine_merge_keys_.data(),
          spine_merge_values_.data());
    }
    clear_epoch_state();
    if (merged_count == 0) {
      clear_spine_state();
    } else {
      std::swap(spine_keys_, spine_merge_keys_);
      std::swap(spine_values_, spine_merge_values_);
      spine_count_ = merged_count;
      build_spine_metadata(stream);
    }
    recompute_live_count();
    invalidate_overlay_derivatives();
  }

  bool try_create_epoch(const std::uint32_t *keys_in,
                        const std::uint32_t *values_in, std::uint8_t op,
                        std::size_t count, cudaStream_t stream) {
    const std::size_t threshold = GPULSMOPT_SCATTER_MIN_BATCH;
    if (op != gpulsmopt_detail::kInsert || values_in == nullptr ||
        count < threshold ||
        count > std::numeric_limits<std::uint32_t>::max())
      return false;
    create_unsorted_epoch(keys_in, values_in, count, stream);
    return true;
  }

  void ingest_sorted_epoch(const std::uint32_t *sorted_keys,
                           const std::uint32_t *sorted_values,
                           std::size_t count, cudaStream_t stream) {
    if (count == 0)
      return;
    {
      GPULSMOPT_PROF_PHASE(prof_delta_ingest_ms_);
      create_sorted_epoch(sorted_keys, sorted_values, count, stream);
    }
  }

  void ingest_filtered_sorted_epoch(
      const std::uint32_t *sorted_keys,
      const std::uint32_t *sorted_values,
      const std::uint32_t *live_flags, std::size_t count,
      cudaStream_t stream) {
    if (count == 0)
      return;
    if (epochs_.size() >=
        static_cast<std::size_t>(gpulsmopt_detail::kEpochMax))
      consolidate_all_epochs(stream);
    acquire_epoch_slot();
    EpochStorage &epoch = epochs_.back();
    epoch.keys.resize_discard(count);
    epoch.values.resize_discard(count);
    auto policy = thrust::cuda::par.on(stream);
    auto input = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_pointer_cast(sorted_keys),
        thrust::device_pointer_cast(sorted_values)));
    auto output = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_pointer_cast(epoch.keys.data()),
        thrust::device_pointer_cast(epoch.values.data())));
    auto output_end = thrust::copy_if(
        policy, input, input + count,
        thrust::device_pointer_cast(live_flags), output,
        gpulsmopt_detail::NonZeroU32{});
    const std::size_t live = static_cast<std::size_t>(output_end - output);
    if (live == 0) {
      epoch_pool_.push_back(std::move(epoch));
      epochs_.pop_back();
      return;
    }
    epoch.count = live;
    epoch.live_count = live;
    epoch.keys.resize_discard(live);
    epoch.values.resize_discard(live);
    commit_epoch_metadata(epoch, stream);
  }

  bool try_direct_delete(const std::uint32_t *keys, std::size_t count,
                         bool sorted, cudaStream_t stream) {
    if (count < GPULSMOPT_SCATTER_MIN_BATCH ||
        count > std::numeric_limits<std::uint32_t>::max())
      return false;
    if (c0_log_count_ > 0 || !runs_.empty())
      merge_down(stream);
    begin_epoch_deletes(keys, count, sorted, stream);
    invalidate_overlay_derivatives();
    return true;
  }

  void flush_c0_to_run(cudaStream_t stream) {
    const std::size_t c0_total = c0_log_count_;
    if (c0_total == 0)
      return;
    if (overlay_cache_valid_ && overlay_pending_count_ != 0)
      invalidate_overlay_records();
    SortedRun run;
    run.keys = std::move(c0_log_keys_);
    run.values = std::move(c0_log_values_);
    run.ops = std::move(c0_log_ops_);
    {
      GPULSMOPT_PROF_PHASE(prof_flushsort_ms_);
      sort_log(run.keys, run.values, run.ops, c0_total, stream);
    }
    run.log_total = static_cast<std::uint32_t>(c0_total);
    run.insert_total = c0_insert_count_;
    run.level = 0;
    c0_log_count_ = 0;
    c0_insert_count_ = 0;
    append_sorted_run(std::move(run), stream);
  }

  struct OverlayReadIndex {
    thrust::device_vector<std::uint32_t> gk, gv;
    thrust::device_vector<std::uint8_t> gop;
    std::size_t u = 0, ins = 0;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> ins_prefix;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> tomb_val_prefix;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> tomb_cnt_prefix;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> live_ins_keys;
    std::size_t live_ins_count = 0;
    gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> killed_keys;
    std::size_t killed_count = 0;
    bool base_values_ready = false;
    bool value_prefix_ready = false;
    bool count_prefix_ready = false;
    bool successor_ready = false;
  };

  void invalidate_overlay_derivatives() {
    cached_overlay_.live_ins_count = 0;
    cached_overlay_.killed_count = 0;
    cached_overlay_.base_values_ready = false;
    cached_overlay_.value_prefix_ready = false;
    cached_overlay_.count_prefix_ready = false;
    cached_overlay_.successor_ready = false;
  }

  void invalidate_overlay_records() {
    overlay_dirty_ = true;
    overlay_cache_valid_ = false;
    overlay_incremental_possible_ = false;
    overlay_pending_count_ = 0;
    overlay_pending_insert_count_ = 0;
    invalidate_overlay_derivatives();
  }

  void note_overlay_append(std::uint8_t op, std::size_t count) {
    if (overlay_cache_valid_) {
      overlay_incremental_possible_ = true;
      overlay_pending_count_ += count;
      if (op == static_cast<std::uint8_t>(gpulsmopt_detail::kInsert))
        overlay_pending_insert_count_ += count;
    }
    overlay_dirty_ = true;
    invalidate_overlay_derivatives();
  }

  std::size_t partition_overlay(
      thrust::device_vector<std::uint32_t> &keys,
      thrust::device_vector<std::uint32_t> &values,
      thrust::device_vector<std::uint8_t> &ops, std::size_t count,
      std::size_t insert_count, cudaStream_t stream) {
    if (insert_count == 0 || insert_count == count)
      return insert_count;
    auto policy = thrust::cuda::par.on(stream);
    auto begin = thrust::make_zip_iterator(
        thrust::make_tuple(keys.begin(), values.begin(), ops.begin()));
    auto middle = thrust::stable_partition(
        policy, begin, begin + count, gpulsmopt_detail::TupleOpIsInsert{});
    return static_cast<std::size_t>(middle - begin);
  }

  void extend_resolved_overlay(OverlayReadIndex &ix, cudaStream_t stream) {
    const std::size_t delta = overlay_pending_count_;
    if (delta == 0)
      return;
    if (!overlay_incremental_possible_ || delta > c0_log_count_)
      throw std::runtime_error("invalid incremental overlay state");
    const std::size_t suffix_begin = c0_log_count_ - delta;
    auto policy = thrust::cuda::par.on(stream);
    resize_reuse(ov_c0k_, delta);
    resize_reuse(ov_c0v_, delta);
    resize_reuse(ov_c0op_, delta);
    thrust::copy(policy, c0_log_keys_.begin() + suffix_begin,
                 c0_log_keys_.begin() + c0_log_count_, ov_c0k_.begin());
    thrust::copy(policy, c0_log_values_.begin() + suffix_begin,
                 c0_log_values_.begin() + c0_log_count_, ov_c0v_.begin());
    thrust::copy(policy, c0_log_ops_.begin() + suffix_begin,
                 c0_log_ops_.begin() + c0_log_count_, ov_c0op_.begin());
    sort_log(ov_c0k_, ov_c0v_, ov_c0op_, delta, stream);
    const std::size_t suffix_inserts = partition_overlay(
        ov_c0k_, ov_c0v_, ov_c0op_, delta,
        overlay_pending_insert_count_, stream);
    const std::size_t old_inserts = ix.ins;
    const std::size_t old_tombs = ix.u - ix.ins;
    const std::size_t suffix_tombs = delta - suffix_inserts;
    const std::size_t total = ix.u + delta;
    const std::size_t total_inserts = old_inserts + suffix_inserts;
    resize_reuse(ov_mk_, total);
    resize_reuse(ov_mv_, total);
    resize_reuse(ov_mop_, total);
    auto merge_partition = [&](std::size_t old_begin, std::size_t old_count,
                               std::size_t suffix_offset,
                               std::size_t suffix_count,
                               std::size_t output) {
      if (old_count == 0) {
        thrust::copy(policy, ov_c0k_.begin() + suffix_offset,
                     ov_c0k_.begin() + suffix_offset + suffix_count,
                     ov_mk_.begin() + output);
        thrust::copy(policy, ov_c0v_.begin() + suffix_offset,
                     ov_c0v_.begin() + suffix_offset + suffix_count,
                     ov_mv_.begin() + output);
      } else if (suffix_count == 0) {
        thrust::copy(policy, ix.gk.begin() + old_begin,
                     ix.gk.begin() + old_begin + old_count,
                     ov_mk_.begin() + output);
        thrust::copy(policy, ix.gv.begin() + old_begin,
                     ix.gv.begin() + old_begin + old_count,
                     ov_mv_.begin() + output);
      } else {
        thrust::merge_by_key(
            policy, ix.gk.begin() + old_begin,
            ix.gk.begin() + old_begin + old_count,
            ov_c0k_.begin() + suffix_offset,
            ov_c0k_.begin() + suffix_offset + suffix_count,
            ix.gv.begin() + old_begin, ov_c0v_.begin() + suffix_offset,
            ov_mk_.begin() + output, ov_mv_.begin() + output);
      }
    };
    merge_partition(0, old_inserts, 0, suffix_inserts, 0);
    merge_partition(ix.ins, old_tombs, suffix_inserts, suffix_tombs,
                    total_inserts);
    if (total_inserts > 0)
      CUDA_CHECK(cudaMemsetAsync(raw_or_null(ov_mop_),
                                 gpulsmopt_detail::kInsert,
                                 total_inserts, stream));
    if (total > total_inserts)
      CUDA_CHECK(cudaMemsetAsync(raw_or_null(ov_mop_) + total_inserts,
                                 gpulsmopt_detail::kTombstone,
                                 total - total_inserts, stream));
    ix.gk.swap(ov_mk_);
    ix.gv.swap(ov_mv_);
    ix.gop.swap(ov_mop_);
    ix.u = total;
    ix.ins = total_inserts;
    overlay_pending_count_ = 0;
    overlay_pending_insert_count_ = 0;
    overlay_incremental_possible_ = false;
    overlay_dirty_ = false;
    invalidate_overlay_derivatives();
  }

  void resolve_overlay(thrust::device_vector<std::uint32_t> &gk,
                       thrust::device_vector<std::uint32_t> &gv,
                       thrust::device_vector<std::uint8_t> &gop, std::size_t &u,
                       std::size_t &ins, cudaStream_t stream,
                       bool consume = false) {
    u = 0;
    ins = 0;
    std::size_t total = c0_log_count_;
    std::size_t insert_total = c0_insert_count_;
    for (auto &g : runs_) {
      total += g.log_total;
      insert_total += g.insert_total;
    }
    if (total == 0) {
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
      ins = partition_overlay(gk, gv, gop, u, run.insert_total, stream);
      return;
    }
    if (consume && runs_.empty() && c0_log_count_ > 0) {
      const std::size_t c0_inserts = c0_insert_count_;
      gk = std::move(c0_log_keys_);
      gv = std::move(c0_log_values_);
      gop = std::move(c0_log_ops_);
      u = c0_log_count_;
      c0_log_count_ = 0;
      c0_insert_count_ = 0;
      sort_log(gk, gv, gop, u, stream);
      ins = partition_overlay(gk, gv, gop, u, c0_inserts, stream);
      return;
    }
    if (!consume && runs_.empty() && c0_log_count_ > 0) {
      const std::size_t count = c0_log_count_;
      resize_reuse(gk, count);
      resize_reuse(gv, count);
      resize_reuse(gop, count);
      thrust::copy(policy, c0_log_keys_.begin(),
                   c0_log_keys_.begin() + count, gk.begin());
      thrust::copy(policy, c0_log_values_.begin(),
                   c0_log_values_.begin() + count, gv.begin());
      thrust::copy(policy, c0_log_ops_.begin(),
                   c0_log_ops_.begin() + count, gop.begin());
      sort_log(gk, gv, gop, count, stream);
      u = count;
      ins = partition_overlay(gk, gv, gop, u, c0_insert_count_, stream);
      return;
    }
    resize_reuse(gk, total);
    resize_reuse(gv, total);
    resize_reuse(gop, total);

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
    ins = partition_overlay(gk, gv, gop, u, insert_total, stream);
  }

  OverlayReadIndex &resolved_overlay(cudaStream_t stream) {
    if (!overlay_dirty_ && overlay_cache_valid_)
      return cached_overlay_;
    auto &ix = cached_overlay_;
    if (overlay_cache_valid_ && overlay_incremental_possible_) {
      extend_resolved_overlay(ix, stream);
      return ix;
    }
    resolve_overlay(ix.gk, ix.gv, ix.gop, ix.u, ix.ins, stream);
    overlay_cache_valid_ = true;
    overlay_incremental_possible_ = false;
    overlay_pending_count_ = 0;
    overlay_pending_insert_count_ = 0;
    overlay_dirty_ = false;
    invalidate_overlay_derivatives();
    return ix;
  }

  void ensure_overlay_base_values(OverlayReadIndex &ix,
                                  cudaStream_t stream) {
    if (ix.base_values_ready)
      return;
    const std::size_t tomb = ix.u - ix.ins;
    overlay_tomb_values_.resize_discard(tomb);
    overlay_tomb_flags_.resize_discard(tomb);
    if (tomb > 0) {
      const int block = 256;
      const int grid = static_cast<int>((tomb + block - 1) / block);
      gpulsmopt_detail::base_point_values_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(ix.gk) + ix.ins, tomb, make_spine_view(),
          epoch_view_ptr(), epoch_count(), overlay_tomb_values_.data(),
          overlay_tomb_flags_.data());
      CUDA_CHECK(cudaGetLastError());
    }
    ix.base_values_ready = true;
  }

  void ensure_overlay_value_prefixes(OverlayReadIndex &ix,
                                     cudaStream_t stream) {
    if (ix.value_prefix_ready)
      return;
    auto policy = thrust::cuda::par.on(stream);
    ix.ins_prefix.resize_discard(ix.ins + 1);
    CUDA_CHECK(cudaMemsetAsync(raw_or_null(ix.ins_prefix), 0,
                               sizeof(std::uint32_t), stream));
    if (ix.ins > 0)
      thrust::inclusive_scan(policy, ix.gv.begin(), ix.gv.begin() + ix.ins,
          thrust::device_pointer_cast(ix.ins_prefix.data()) + 1);
    ensure_overlay_base_values(ix, stream);
    const std::size_t tomb = ix.u - ix.ins;
    ix.tomb_val_prefix.resize_discard(tomb + 1);
    CUDA_CHECK(cudaMemsetAsync(raw_or_null(ix.tomb_val_prefix), 0,
                               sizeof(std::uint32_t), stream));
    if (tomb > 0)
      thrust::inclusive_scan(
          policy, thrust::device_pointer_cast(overlay_tomb_values_.data()),
          thrust::device_pointer_cast(overlay_tomb_values_.data()) + tomb,
          thrust::device_pointer_cast(ix.tomb_val_prefix.data()) + 1);
    ix.value_prefix_ready = true;
  }

  void ensure_overlay_count_prefix(OverlayReadIndex &ix,
                                   cudaStream_t stream) {
    if (ix.count_prefix_ready)
      return;
    ensure_overlay_base_values(ix, stream);
    const std::size_t tomb = ix.u - ix.ins;
    ix.tomb_cnt_prefix.resize_discard(tomb + 1);
    CUDA_CHECK(cudaMemsetAsync(raw_or_null(ix.tomb_cnt_prefix), 0,
                               sizeof(std::uint32_t), stream));
    if (tomb > 0) {
      auto policy = thrust::cuda::par.on(stream);
      thrust::inclusive_scan(
          policy, thrust::device_pointer_cast(overlay_tomb_flags_.data()),
          thrust::device_pointer_cast(overlay_tomb_flags_.data()) + tomb,
          thrust::device_pointer_cast(ix.tomb_cnt_prefix.data()) + 1);
    }
    ix.count_prefix_ready = true;
  }

  void ensure_overlay_successor(OverlayReadIndex &ix,
                                cudaStream_t stream) {
    if (ix.successor_ready)
      return;
    auto policy = thrust::cuda::par.on(stream);
    ensure_overlay_base_values(ix, stream);
    const std::size_t tomb = ix.u - ix.ins;
    ix.killed_keys.resize_discard(tomb);
    ix.killed_count = 0;
    if (tomb > 0) {
      auto end = thrust::copy_if(
          policy, ix.gk.begin() + ix.ins, ix.gk.begin() + ix.u,
          thrust::device_pointer_cast(overlay_tomb_flags_.data()),
          thrust::device_pointer_cast(ix.killed_keys.data()),
          gpulsmopt_detail::NonZeroU32{});
      ix.killed_count = static_cast<std::size_t>(
          end - thrust::device_pointer_cast(ix.killed_keys.data()));
    }
    ix.live_ins_keys.resize_discard(ix.ins);
    ix.live_ins_count = 0;
    if (ix.ins > 0) {
      overlay_live_flags_.resize_discard(ix.ins);
      const int block = 256;
      const int grid = static_cast<int>((ix.ins + block - 1) / block);
      gpulsmopt_detail::mark_live_inserts_kernel<<<grid, block, 0, stream>>>(
          raw_or_null(ix.gk), ix.ins, raw_or_null(ix.gk) + ix.ins, tomb,
          overlay_live_flags_.data());
      CUDA_CHECK(cudaGetLastError());
      auto end = thrust::copy_if(
          policy, ix.gk.begin(), ix.gk.begin() + ix.ins,
          thrust::device_pointer_cast(overlay_live_flags_.data()),
          thrust::device_pointer_cast(ix.live_ins_keys.data()),
          gpulsmopt_detail::NonZeroU32{});
      ix.live_ins_count = static_cast<std::size_t>(
          end - thrust::device_pointer_cast(ix.live_ins_keys.data()));
    }
    ix.successor_ready = true;
  }

  void apply_sorted_deletes(const std::uint32_t *keys,
                            std::size_t count, cudaStream_t stream) {
    if (count == 0)
      return;
    begin_epoch_deletes(keys, count, true, stream);
    reap_pending_deletes(stream);
  }

  void merge_down(cudaStream_t stream) {
    thrust::device_vector<std::uint32_t> gk, gv;
    thrust::device_vector<std::uint8_t> gop;
    std::size_t total = c0_log_count_;
    for (const auto &run : runs_)
      total += run.log_total;
    const bool takes_source =
        (c0_log_count_ == 0 && runs_.size() == 1) ||
        (runs_.empty() && c0_log_count_ > 0);
    if (total > 0 && !takes_source) {
      SortedRun scratch = acquire_run_storage(total, 0);
      gk = std::move(scratch.keys);
      gv = std::move(scratch.values);
      gop = std::move(scratch.ops);
    }
    std::size_t u = 0, ins = 0;
    {
      GPULSMOPT_PROF_PHASE(prof_resolve_ms_);
      resolve_overlay(gk, gv, gop, u, ins, stream, true);
    }
    if (u > 0) {
      const std::size_t tomb = u - ins;
      if (tomb > 0) {
        GPULSMOPT_PROF_PHASE(prof_delete_ms_);
        apply_sorted_deletes(raw_or_null(gk) + ins, tomb, stream);
      }
      if (ins > 0) {
        GPULSMOPT_PROF_PHASE(prof_epoch_ingest_ms_);
        if (tomb > 0) {
          overlay_live_flags_.resize_discard(ins);
          const int lblock = 256;
          const int lgrid = static_cast<int>((ins + lblock - 1) / lblock);
          gpulsmopt_detail::mark_live_inserts_kernel<<<lgrid, lblock, 0,
                                                       stream>>>(
              raw_or_null(gk), ins, raw_or_null(gk) + ins, tomb,
              overlay_live_flags_.data());
          CUDA_CHECK(cudaGetLastError());
          ingest_filtered_sorted_epoch(
              raw_or_null(gk), raw_or_null(gv),
              overlay_live_flags_.data(), ins, stream);
        } else {
          ingest_sorted_epoch(raw_or_null(gk), raw_or_null(gv), ins,
                              stream);
        }
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
    invalidate_overlay_records();
  }

  void maybe_flush_and_merge(cudaStream_t stream) {
    if (c0_log_live() >= c0_flush_budget())
      flush_c0_to_run(stream);
    if (runs_should_drain())
      merge_down(stream);
  }

  void lookup_layered(const DeviceLookupBatch &batch, cudaStream_t stream) {
    const std::size_t n = batch.count;
    const bool no_overlay = c0_log_count_ == 0 && runs_.empty();
    const OverlayReadIndex *ix =
        no_overlay ? nullptr : &resolved_overlay(stream);
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    const auto spine = make_spine_view();
    gpulsmopt_detail::point_lookup_kernel<true, true>
        <<<grid, block, 0, stream>>>(
            batch.queries, n, batch.out_values, batch.out_found,
            no_overlay ? nullptr : raw_or_null(ix->gk),
            no_overlay ? nullptr : raw_or_null(ix->gv),
            no_overlay ? 0 : ix->ins,
            no_overlay ? nullptr : raw_or_null(ix->gk) + ix->ins,
            no_overlay ? 0 : ix->u - ix->ins, spine,
            epoch_view_ptr(), epoch_count());
    CUDA_CHECK(cudaGetLastError());
  }

  std::size_t max_elements_ = 0;
  std::size_t batch_capacity_ = 0;
  DeleteOrderPolicy delete_order_policy_ =
      DeleteOrderPolicy::adaptive;
  std::size_t live_count_ = 0;
  std::size_t spine_count_ = 0;
  std::size_t spine_live_count_ = 0;
  bool spine_has_dead_ = false;
  bool spine_dead_count_prefix_ready_ = true;
  bool spine_dead_value_prefix_ready_ = true;
  mutable std::shared_mutex snapshot_mutex_;

  std::uint32_t c0_log_count_ = 0;
  std::uint32_t c0_insert_count_ = 0;
  thrust::device_vector<std::uint32_t> c0_log_keys_;
  thrust::device_vector<std::uint32_t> c0_log_values_;
  thrust::device_vector<std::uint8_t> c0_log_ops_;

#ifdef GPULSMOPT_PROFILE_INSERT
  double prof_append_ms_ = 0.0;
  double prof_flushsort_ms_ = 0.0;
  double prof_runmerge_ms_ = 0.0;
  double prof_resolve_ms_ = 0.0;
  double prof_delete_ms_ = 0.0;
  double prof_epoch_ingest_ms_ = 0.0;
  double prof_delta_sort_ms_ = 0.0;
  double prof_delta_ingest_ms_ = 0.0;
  double prof_delta_consolidate_ms_ = 0.0;
  void reset_insert_prof_() {
    prof_append_ms_ = prof_flushsort_ms_ = prof_runmerge_ms_ = 0.0;
    prof_resolve_ms_ = prof_delete_ms_ = prof_epoch_ingest_ms_ = 0.0;
    prof_delta_sort_ms_ = prof_delta_ingest_ms_ = 0.0;
    prof_delta_consolidate_ms_ = 0.0;
  }
#endif

  std::vector<SortedRun> runs_;
  std::vector<SortedRun> run_buffer_pool_;

  std::vector<EpochStorage> epochs_;
  std::vector<EpochStorage> epoch_pool_;
  thrust::device_vector<gpulsmopt_detail::EpochView> epoch_views_;
  std::vector<gpulsmopt_detail::EpochView> bound_epoch_views_;
  std::vector<std::uint8_t> bound_epoch_view_valid_;
  thrust::device_vector<std::uint32_t> epoch_removed_;
  std::uint32_t *epoch_removed_host_ = nullptr;
  std::uint8_t *delete_layer_order_host_ = nullptr;
  std::size_t delete_layer_order_host_count_ = 0;
  bool delete_layer_order_host_valid_ = false;
  std::size_t delete_layer_order_device_count_ = 0;
  bool delete_layer_order_device_matches_host_ = false;
  cudaEvent_t epoch_delete_ready_ = nullptr;
  cudaStream_t epoch_delete_stream_ = nullptr;
  std::size_t epoch_delete_epoch_count_ = 0;
  bool epoch_delete_pending_ = false;

  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> direct_sort_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> direct_sort_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> sort_key_output_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> sort_payload_input_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint64_t> sort_payload_output_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> sort_temp_storage_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> epoch_merge_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> epoch_merge_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> epoch_quotient_counts_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> epoch_quotient_offsets_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> epoch_quotient_cursor_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_value_prefix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_radix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_micro_base_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint16_t> spine_micro_offsets_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> spine_micro_bits_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_dead_words_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_dead_count_sum_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_dead_value_sum_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_dead_count_prefix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_dead_value_prefix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_live_counts_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_live_prefix_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_gather_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_gather_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_merge_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> spine_merge_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> delete_sort_keys_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> delete_layer_stats_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint8_t> delete_layer_order_;
  std::size_t direct_sort_count_ = 0;
  std::size_t direct_sort_temp_bytes_ = 0;
  std::size_t epoch_sort_count_ = 0;
  std::size_t epoch_sort_temp_bytes_ = 0;
  std::size_t segmented_sort_count_ = 0;
  std::size_t segmented_sort_temp_bytes_ = 0;
  std::size_t delete_sort_count_ = 0;
  std::size_t delete_sort_temp_bytes_ = 0;
  std::size_t log_sort_count_ = 0;
  std::size_t log_sort_temp_bytes_ = 0;
  std::size_t scan_u32_count_ = 0;
  std::size_t scan_u32_temp_bytes_ = 0;
  bool spine_micro_ready_ = false;

  thrust::device_vector<std::uint32_t> ov_c0k_;
  thrust::device_vector<std::uint32_t> ov_c0v_;
  thrust::device_vector<std::uint8_t> ov_c0op_;
  thrust::device_vector<std::uint32_t> ov_mk_;
  thrust::device_vector<std::uint32_t> ov_mv_;
  thrust::device_vector<std::uint8_t> ov_mop_;

  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> overlay_tomb_values_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> overlay_tomb_flags_;
  gpulsmopt_detail::RawDeviceBuffer<std::uint32_t> overlay_live_flags_;

  OverlayReadIndex cached_overlay_;
  bool overlay_dirty_ = true;
  bool overlay_cache_valid_ = false;
  bool overlay_incremental_possible_ = false;
  std::size_t overlay_pending_count_ = 0;
  std::size_t overlay_pending_insert_count_ = 0;
};
