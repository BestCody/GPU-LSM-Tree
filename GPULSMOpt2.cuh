#pragma once

// GPULSMOpt2: update-driven quotient-local epoch LSM.
// This implementation is independent of the legacy GPULSMOpt structure.

#include "gpu_dictionary_adapter.cuh"
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t e__ = (call);                                             \
    if (e__ != cudaSuccess)                                                     \
      throw std::runtime_error(std::string(cudaGetErrorString(e__)) +           \
                               " at " + __FILE__ + ":" +                     \
                               std::to_string(__LINE__));                       \
  } while (false)
#endif

namespace gpulsmopt2_detail {

constexpr std::uint32_t kQuotients = 1u << 16u;
constexpr std::uint32_t kLevels = 4u;
#ifndef GPULSMOPT2_BATCHES_PER_EPOCH
#define GPULSMOPT2_BATCHES_PER_EPOCH 16
#endif
constexpr std::uint32_t kBatchesPerEpoch = GPULSMOPT2_BATCHES_PER_EPOCH;
static_assert(kBatchesPerEpoch == 8u || kBatchesPerEpoch == 16u,
              "GPULSMOpt2 currently supports eight- or sixteen-batch epochs");
constexpr std::uint32_t kBatchPositionBits =
    kBatchesPerEpoch == 16u ? 28u : 29u;
constexpr std::uint32_t kRankBits = 23u;
constexpr std::uint32_t kRankEntries = (1u << kRankBits) + 1u;
constexpr std::uint32_t kThreads = 256u;
constexpr std::uint32_t kWarpsPerBlock = kThreads / 32u;
constexpr std::uint32_t kSmallMaximum = 256u;
constexpr std::uint32_t kRangeCapacity = 1024u;
constexpr std::uint32_t kMaximumSizeClass = 16u;
constexpr std::uint32_t kInvalid = 0xffffffffu;
constexpr std::uint32_t kTombstone = 1u;
constexpr std::uint32_t kMaximumArenaRows = 1u << 26u;

struct Row {
  std::uint32_t key;
  std::uint32_t value;
  std::uint32_t flags;
};

struct RawAssignment {
  Row row;
  std::uint32_t logical_position;
};

static_assert(sizeof(Row) == 12u);

struct Descriptor {
  std::uint64_t bits{};
  __host__ __device__ static Descriptor make(std::uint32_t offset,
                                             std::uint32_t count,
                                             std::uint32_t size_class,
                                             std::uint32_t generation) {
    return {std::uint64_t{offset} |
            (std::uint64_t{count} << 26u) |
            (std::uint64_t{size_class} << 43u) |
            (std::uint64_t{generation} << 48u)};
  }
  __host__ __device__ std::uint32_t offset() const {
    return static_cast<std::uint32_t>(bits & ((1ull << 26u) - 1ull));
  }
  __host__ __device__ std::uint32_t count() const {
    return static_cast<std::uint32_t>((bits >> 26u) & ((1ull << 17u) - 1ull));
  }
  __host__ __device__ std::uint32_t size_class() const {
    return static_cast<std::uint32_t>((bits >> 43u) & 31ull);
  }
  __host__ __device__ std::uint32_t generation() const {
    return static_cast<std::uint32_t>(bits >> 48u);
  }
};

static_assert(sizeof(Descriptor) == 8u);

__host__ __device__ __forceinline__ std::size_t descriptor_index(
    std::uint32_t q, std::uint32_t level) {
  return std::size_t{q} * kLevels + level;
}

struct Pending {
  std::uint32_t offset;
  std::uint32_t count;
  std::uint16_t size_class;
  std::uint8_t level;
  std::uint8_t valid;
};

struct TaggedRow {
  Row row;
  std::uint32_t age;
};

struct RangeFragment {
  std::uint32_t query;
  std::uint32_t quotient;
};

static_assert(sizeof(RangeFragment) == 8u);

template <class T> class Buffer {
public:
  Buffer() = default;
  explicit Buffer(std::size_t count) { resize(count); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept : pointer_(other.pointer_), count_(other.count_) {
    other.pointer_ = nullptr;
    other.count_ = 0u;
  }
  Buffer &operator=(Buffer &&other) noexcept {
    if (this != &other) {
      if (pointer_) cudaFree(pointer_);
      pointer_ = other.pointer_;
      count_ = other.count_;
      other.pointer_ = nullptr;
      other.count_ = 0u;
    }
    return *this;
  }
  ~Buffer() { if (pointer_) cudaFree(pointer_); }
  void resize(std::size_t count) {
    if (pointer_) CUDA_CHECK(cudaFree(pointer_));
    pointer_ = nullptr;
    count_ = count;
    if (count)
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pointer_),
                            count * sizeof(T)));
  }
  T *data() { return pointer_; }
  const T *data() const { return pointer_; }
  std::size_t size() const { return count_; }
private:
  T *pointer_{};
  std::size_t count_{};
};

__global__ void initialize_query_ids_kernel(std::uint32_t *ids,
                                            std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) ids[i] = i;
}

__global__ void scatter_lookup_results_kernel(
    const std::uint32_t *sorted_ids, const std::uint64_t *sorted_results,
    std::uint64_t *results, std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) results[sorted_ids[i]] = sorted_results[i];
}

__global__ void scatter_lookup_output_kernel(
    const std::uint32_t *sorted_ids, const std::uint32_t *sorted_values,
    const std::uint8_t *sorted_found, std::uint32_t *values,
    std::uint8_t *found, std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t destination = sorted_ids[i];
  values[destination] = sorted_values[i];
  if (found) found[destination] = sorted_found[i];
}

__device__ __forceinline__ std::uint32_t size_class_for(
    std::uint32_t count) {
  if (count <= 1u) return 0u;
  return 32u - static_cast<std::uint32_t>(__clz(count - 1u));
}

__device__ void release_extent(Row *arena, std::uint32_t *local_free_heads,
                               std::uint32_t q, std::uint32_t offset,
                               std::uint32_t size_class) {
  std::uint32_t &local_head =
      local_free_heads[q * (kMaximumSizeClass + 1u) + size_class];
  arena[offset].key = local_head;
  local_head = offset;
}

__device__ __forceinline__ bool tagged_less(const TaggedRow &a,
                                            const TaggedRow &b) {
  const bool ai = a.age == kInvalid;
  const bool bi = b.age == kInvalid;
  if (ai != bi) return !ai;
  if (ai) return false;
  if (a.row.key != b.row.key) return a.row.key < b.row.key;
  return a.age < b.age;
}

// Allocate one power-of-two extent per touched quotient, but reserve all fresh
// storage required by a 256-thread block with one global atomic. Recycled pages
// remain quotient-local and therefore need no contested allocator operation.
__global__ void plan_extents_kernel(
    const std::uint32_t *__restrict__ epoch_counts,
    Row *__restrict__ arena, std::uint32_t arena_capacity,
    const Descriptor *__restrict__ descriptors,
    const std::uint8_t *__restrict__ masks,
    std::uint32_t *__restrict__ local_free_heads,
    std::uint32_t *__restrict__ cursor, Pending *__restrict__ pending,
    std::uint32_t *__restrict__ dense_list,
    std::uint32_t *__restrict__ dense_count,
    unsigned long long *__restrict__ reuse_count,
    std::uint32_t *__restrict__ overflow) {
  __shared__ std::uint32_t fresh_capacity[kThreads];
  __shared__ std::uint32_t fresh_prefix[kThreads];
  __shared__ std::uint32_t recycled_offset[kThreads];
  __shared__ std::uint32_t block_base;
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t capacity = 0u, recycled = kInvalid;
  Pending plan{kInvalid, 0u, 0u, 0u, 0u};
  if (q < kQuotients && epoch_counts[q] != 0u) {
    const std::uint32_t depth =
        static_cast<std::uint32_t>(__ffs(~masks[q]) - 1);
    if (depth >= kLevels) {
      atomicAdd(overflow, 1u);
    } else {
      std::uint32_t total = epoch_counts[q];
      for (std::uint32_t level = 0u; level < depth; ++level)
        total += descriptors[descriptor_index(q, level)].count();
      const std::uint32_t size_class = size_class_for(total);
      if (total >= (1u << 17u) || size_class > kMaximumSizeClass) {
        atomicAdd(overflow, 1u);
      } else {
        std::uint32_t &head =
            local_free_heads[q * (kMaximumSizeClass + 1u) + size_class];
        recycled = head;
        if (recycled != kInvalid) head = arena[recycled].key;
        else capacity = 1u << size_class;
        plan = {kInvalid, total, static_cast<std::uint16_t>(size_class),
                static_cast<std::uint8_t>(depth), 1u};
      }
    }
  }
  fresh_capacity[threadIdx.x] = capacity;
  recycled_offset[threadIdx.x] = recycled;
  __syncthreads();
  if (threadIdx.x == 0u) {
    std::uint32_t total = 0u, reused = 0u;
    for (std::uint32_t i = 0u; i < blockDim.x; ++i) {
      fresh_prefix[i] = total;
      total += fresh_capacity[i];
      reused += recycled_offset[i] != kInvalid;
    }
    block_base = total ? atomicAdd(cursor, total) : 0u;
    if (reused) atomicAdd(reuse_count, static_cast<unsigned long long>(reused));
    if (total && (block_base > arena_capacity ||
                  total > arena_capacity - block_base))
      atomicAdd(overflow, 1u);
  }
  __syncthreads();
  if (q >= kQuotients) return;
  if (!plan.valid) {
    pending[q] = plan;
    return;
  }
  plan.offset = recycled != kInvalid
      ? recycled : block_base + fresh_prefix[threadIdx.x];
  pending[q] = plan;
  if (plan.count > kSmallMaximum)
    dense_list[atomicAdd(dense_count, 1u)] = q;
}

__device__ TaggedRow load_stream_ordinal(
    std::uint32_t ordinal, const Row *epoch_rows, std::uint32_t epoch_begin,
    std::uint32_t epoch_count, const Row *arena,
    const Descriptor *descriptors, std::uint32_t q, std::uint32_t depth) {
  if (ordinal < epoch_count)
    return {epoch_rows[epoch_begin + ordinal], kLevels + 1u};
  ordinal -= epoch_count;
  for (std::uint32_t level = 0u; level < depth; ++level) {
    const Descriptor descriptor = descriptors[descriptor_index(q, level)];
    if (ordinal < descriptor.count())
      return {arena[descriptor.offset() + ordinal], kLevels - level};
    ordinal -= descriptor.count();
  }
  return {{0u, 0u, 0u}, kInvalid};
}

__device__ RawAssignment load_raw_ordinal(
    const RawAssignment *assignments, const std::uint32_t *offsets,
    const std::uint32_t *batch_bases, std::uint32_t q,
    std::uint32_t ordinal) {
#pragma unroll
  for (std::uint32_t batch = 0u; batch < kBatchesPerEpoch; ++batch) {
    const std::size_t oi = std::size_t{batch} * (kQuotients + 1u) + q;
    const std::uint32_t begin = offsets[oi];
    const std::uint32_t count = offsets[oi + 1u] - begin;
    if (ordinal < count)
      return assignments[batch_bases[batch] + begin + ordinal];
    ordinal -= count;
  }
  return {{0u, 0u, 0u}, 0u};
}

__global__ void fused_raw_epoch_carry_kernel(
    const RawAssignment *__restrict__ assignments,
    const std::uint32_t *__restrict__ raw_offsets,
    const std::uint32_t *__restrict__ batch_bases,
    const std::uint32_t *__restrict__ raw_counts,
    Row *__restrict__ arena, const Descriptor *__restrict__ descriptors,
    Pending *__restrict__ pending, std::uint32_t *__restrict__ overflow) {
  __shared__ TaggedRow items[kWarpsPerBlock][kSmallMaximum];
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t q = blockIdx.x * kWarpsPerBlock + warp;
  if (q >= kQuotients || !pending[q].valid) return;
  Pending allocation = pending[q];
  const std::uint32_t total = allocation.count;
  if (total <= 128u) return;
  if (total > kSmallMaximum) {
    return;
  }
  const std::uint32_t raw_count = raw_counts[q];
  const std::uint32_t sort_size = max(32u, 1u << size_class_for(total));
  TaggedRow *local = items[warp];
  for (std::uint32_t i = lane; i < sort_size; i += 32u) {
    if (i < raw_count) {
      const RawAssignment raw = load_raw_ordinal(
          assignments, raw_offsets, batch_bases, q, i);
      local[i] = {raw.row, 0x10000000u | raw.logical_position};
    } else if (i < total) {
      std::uint32_t ordinal = i - raw_count;
      bool loaded = false;
      for (std::uint32_t level = 0u; level < allocation.level; ++level) {
        const Descriptor descriptor = descriptors[descriptor_index(q, level)];
        if (ordinal < descriptor.count()) {
          local[i] = {arena[descriptor.offset() + ordinal], kLevels - level};
          loaded = true;
          break;
        }
        ordinal -= descriptor.count();
      }
      if (!loaded) local[i] = {{0u, 0u, 0u}, kInvalid};
    } else {
      local[i] = {{0u, 0u, 0u}, kInvalid};
    }
  }
  __syncwarp();
  for (std::uint32_t width = 2u; width <= sort_size; width <<= 1u) {
    for (std::uint32_t stride = width >> 1u; stride; stride >>= 1u) {
      for (std::uint32_t group = 0u; group < sort_size / 32u; ++group) {
        const std::uint32_t index = lane + group * 32u;
        const std::uint32_t other_index = index ^ stride;
        if (other_index > index) {
          const TaggedRow x = local[index], y = local[other_index];
          const bool ascending = (index & width) == 0u;
          const bool swap = ascending ? tagged_less(y, x) : tagged_less(x, y);
          if (swap) {
            local[index] = y;
            local[other_index] = x;
          }
        }
      }
      __syncwarp();
    }
  }
  bool winner[kSmallMaximum / 32u]{};
  unsigned masks[kSmallMaximum / 32u]{};
  std::uint32_t winner_count = 0u;
  for (std::uint32_t group = 0u; group < sort_size / 32u; ++group) {
    const std::uint32_t index = lane + group * 32u;
    winner[group] = local[index].age != kInvalid &&
        (index + 1u == sort_size || local[index + 1u].age == kInvalid ||
         local[index].row.key != local[index + 1u].row.key);
    masks[group] = __ballot_sync(0xffffffffu, winner[group]);
    winner_count += __popc(masks[group]);
  }
  const unsigned before = lane == 0u ? 0u : ((1u << lane) - 1u);
  std::uint32_t base = 0u;
  for (std::uint32_t group = 0u; group < sort_size / 32u; ++group) {
    if (winner[group])
      arena[allocation.offset + base + __popc(masks[group] & before)] =
          local[lane + group * 32u].row;
    base += __popc(masks[group]);
  }
  if (lane == 0u) pending[q].count = winner_count;
}

__global__ void fused_raw_rank_carry_kernel(
    const RawAssignment *__restrict__ assignments,
    const std::uint32_t *__restrict__ raw_offsets,
    const std::uint32_t *__restrict__ batch_bases,
    const std::uint32_t *__restrict__ raw_counts,
    Row *__restrict__ arena, const Descriptor *__restrict__ descriptors,
    Pending *__restrict__ pending, std::uint32_t *__restrict__ overflow) {
  constexpr std::uint32_t warps_per_block = 4u;
  constexpr std::uint32_t raw_capacity = 64u;
  __shared__ TaggedRow raw_shared[warps_per_block][raw_capacity];
  __shared__ Row current_shared[warps_per_block][128];
  __shared__ Row older_shared[warps_per_block][128];
  __shared__ Row merged_shared[warps_per_block][128];
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t q = blockIdx.x * warps_per_block + warp;
  if (q >= kQuotients || !pending[q].valid) return;
  const Pending allocation = pending[q];
  if (allocation.level == 0u) return;
  const std::uint32_t raw_count = raw_counts[q];
  if (raw_count > raw_capacity) {
    return;
  }
  if (allocation.count > 128u) return;
  TaggedRow *raw = raw_shared[warp];
  Row *current = current_shared[warp];
  Row *older = older_shared[warp];
  Row *merged = merged_shared[warp];
  const std::uint32_t raw_sort_size = max(32u, 1u << size_class_for(raw_count));
  for (std::uint32_t i = lane; i < raw_sort_size; i += 32u) {
    if (i < raw_count) {
      const RawAssignment item = load_raw_ordinal(
          assignments, raw_offsets, batch_bases, q, i);
      raw[i] = {item.row, item.logical_position};
    } else {
      raw[i] = {{0u, 0u, 0u}, kInvalid};
    }
  }
  __syncwarp();
  for (std::uint32_t width = 2u; width <= raw_sort_size; width <<= 1u) {
    for (std::uint32_t stride = width >> 1u; stride; stride >>= 1u) {
      for (std::uint32_t group = 0u; group < raw_sort_size / 32u; ++group) {
        const std::uint32_t index = lane + group * 32u;
        const std::uint32_t other_index = index ^ stride;
        if (other_index > index) {
          const TaggedRow x = raw[index], y = raw[other_index];
          const bool ascending = (index & width) == 0u;
          const bool swap = ascending ? tagged_less(y, x) : tagged_less(x, y);
          if (swap) {
            raw[index] = y;
            raw[other_index] = x;
          }
        }
      }
      __syncwarp();
    }
  }
  bool raw_winner[2]{};
  unsigned raw_masks[2]{};
  std::uint32_t current_count = 0u;
  for (std::uint32_t group = 0u; group < raw_sort_size / 32u; ++group) {
    const std::uint32_t index = lane + group * 32u;
    raw_winner[group] = raw[index].age != kInvalid &&
        (index + 1u == raw_sort_size || raw[index + 1u].age == kInvalid ||
         raw[index].row.key != raw[index + 1u].row.key);
    raw_masks[group] = __ballot_sync(0xffffffffu, raw_winner[group]);
    current_count += __popc(raw_masks[group]);
  }
  const unsigned before = lane == 0u ? 0u : ((1u << lane) - 1u);
  std::uint32_t raw_base = 0u;
  for (std::uint32_t group = 0u; group < raw_sort_size / 32u; ++group) {
    if (raw_winner[group])
      current[raw_base + __popc(raw_masks[group] & before)] =
          raw[lane + group * 32u].row;
    raw_base += __popc(raw_masks[group]);
  }
  __syncwarp();

  for (std::uint32_t level = 0u; level < allocation.level; ++level) {
    const Descriptor descriptor = descriptors[descriptor_index(q, level)];
    const std::uint32_t older_count = descriptor.count();
    if (current_count + older_count > 128u) {
      if (lane == 0u) atomicAdd(overflow, 1u);
      return;
    }
    for (std::uint32_t i = lane; i < older_count; i += 32u)
      older[i] = arena[descriptor.offset() + i];
    __syncwarp();
    for (std::uint32_t i = lane; i < older_count; i += 32u) {
      const std::uint32_t key = older[i].key;
      std::uint32_t lo = 0u, hi = current_count;
      while (lo < hi) {
        const std::uint32_t mid = (lo + hi) >> 1u;
        if (current[mid].key < key) lo = mid + 1u;
        else hi = mid;
      }
      merged[i + lo] = older[i];
    }
    for (std::uint32_t i = lane; i < current_count; i += 32u) {
      const std::uint32_t key = current[i].key;
      std::uint32_t lo = 0u, hi = older_count;
      while (lo < hi) {
        const std::uint32_t mid = (lo + hi) >> 1u;
        if (older[mid].key <= key) lo = mid + 1u;
        else hi = mid;
      }
      merged[i + lo] = current[i];
    }
    __syncwarp();
    const std::uint32_t merged_count = current_count + older_count;
    bool winner[4]{};
    unsigned masks[4]{};
    std::uint32_t next_count = 0u;
    for (std::uint32_t group = 0u; group < 4u; ++group) {
      const std::uint32_t index = lane + group * 32u;
      winner[group] = index < merged_count &&
          (index + 1u == merged_count ||
           merged[index].key != merged[index + 1u].key);
      masks[group] = __ballot_sync(0xffffffffu, winner[group]);
      next_count += __popc(masks[group]);
    }
    std::uint32_t base = 0u;
    for (std::uint32_t group = 0u; group < 4u; ++group) {
      if (winner[group])
        current[base + __popc(masks[group] & before)] =
            merged[lane + group * 32u];
      base += __popc(masks[group]);
    }
    __syncwarp();
    current_count = next_count;
  }
  for (std::uint32_t i = lane; i < current_count; i += 32u)
    arena[allocation.offset + i] = current[i];
  if (lane == 0u) pending[q].count = current_count;
}

// Two-buffer stable merge-path variant. One diagonal partition produces a
// short contiguous output tile per lane; older equal keys are emitted before
// newer equal keys, so adjacent duplicate reduction retains newest-wins.
__global__ void fused_raw_merge_path_carry_kernel(
    const RawAssignment *__restrict__ assignments,
    const std::uint32_t *__restrict__ raw_offsets,
    const std::uint32_t *__restrict__ batch_bases,
    const std::uint32_t *__restrict__ raw_counts,
    Row *__restrict__ arena, const Descriptor *__restrict__ descriptors,
    Pending *__restrict__ pending, std::uint32_t *__restrict__ overflow) {
  constexpr std::uint32_t warps_per_block = 4u;
  constexpr std::uint32_t raw_capacity = 64u;
  __shared__ TaggedRow raw_shared[warps_per_block][raw_capacity];
  __shared__ Row current_shared[warps_per_block][128];
  __shared__ Row merged_shared[warps_per_block][128];
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t q = blockIdx.x * warps_per_block + warp;
  if (q >= kQuotients || !pending[q].valid) return;
  const Pending allocation = pending[q];
  if (allocation.level == 0u) return;
  const std::uint32_t raw_count = raw_counts[q];
  if (raw_count > raw_capacity) {
    return;
  }
  if (allocation.count > 128u) return;
  TaggedRow *raw = raw_shared[warp];
  Row *current = current_shared[warp];
  Row *merged = merged_shared[warp];
  const std::uint32_t raw_sort_size = max(32u, 1u << size_class_for(raw_count));
  for (std::uint32_t index = lane; index < raw_sort_size; index += 32u) {
    if (index < raw_count) {
      const RawAssignment item = load_raw_ordinal(
          assignments, raw_offsets, batch_bases, q, index);
      raw[index] = {item.row, item.logical_position};
    } else {
      raw[index] = {{0u, 0u, 0u}, kInvalid};
    }
  }
  __syncwarp();
  for (std::uint32_t width = 2u; width <= raw_sort_size; width <<= 1u) {
    for (std::uint32_t stride = width >> 1u; stride; stride >>= 1u) {
      for (std::uint32_t group = 0u; group < raw_sort_size / 32u; ++group) {
        const std::uint32_t index = lane + group * 32u;
        const std::uint32_t other_index = index ^ stride;
        if (other_index > index) {
          const TaggedRow x = raw[index], y = raw[other_index];
          const bool ascending = (index & width) == 0u;
          const bool swap = ascending ? tagged_less(y, x) : tagged_less(x, y);
          if (swap) {
            raw[index] = y;
            raw[other_index] = x;
          }
        }
      }
      __syncwarp();
    }
  }
  bool raw_winner[2]{};
  unsigned raw_masks[2]{};
  std::uint32_t current_count = 0u;
  for (std::uint32_t group = 0u; group < raw_sort_size / 32u; ++group) {
    const std::uint32_t index = lane + group * 32u;
    raw_winner[group] = raw[index].age != kInvalid &&
        (index + 1u == raw_sort_size || raw[index + 1u].age == kInvalid ||
         raw[index].row.key != raw[index + 1u].row.key);
    raw_masks[group] = __ballot_sync(0xffffffffu, raw_winner[group]);
    current_count += __popc(raw_masks[group]);
  }
  const unsigned before = lane == 0u ? 0u : ((1u << lane) - 1u);
  std::uint32_t raw_base = 0u;
  for (std::uint32_t group = 0u; group < raw_sort_size / 32u; ++group) {
    if (raw_winner[group])
      current[raw_base + __popc(raw_masks[group] & before)] =
          raw[lane + group * 32u].row;
    raw_base += __popc(raw_masks[group]);
  }
  __syncwarp();

  for (std::uint32_t level = 0u; level < allocation.level; ++level) {
    const Descriptor descriptor = descriptors[descriptor_index(q, level)];
    const Row *older = arena + descriptor.offset();
    const std::uint32_t older_count = descriptor.count();
    const std::uint32_t merged_count = current_count + older_count;
    if (merged_count > 128u) {
      if (lane == 0u) atomicAdd(overflow, 1u);
      return;
    }
    const std::uint32_t tile = merged_count <= 32u ? 1u :
                               merged_count <= 64u ? 2u : 4u;
    const std::uint32_t diagonal = min(lane * tile, merged_count);
    std::uint32_t low = diagonal > current_count
        ? diagonal - current_count : 0u;
    std::uint32_t high = min(diagonal, older_count);
    // Find how many older rows occur in the prefix. Equality moves right so
    // older equal keys are chosen before the newer current stream.
    while (low < high) {
      const std::uint32_t older_index = (low + high) >> 1u;
      const std::uint32_t current_index = diagonal - older_index;
      if (older_index < older_count && current_index > 0u &&
          current[current_index - 1u].key >= older[older_index].key)
        low = older_index + 1u;
      else
        high = older_index;
    }
    std::uint32_t older_index = low;
    std::uint32_t current_index = diagonal - older_index;
#pragma unroll
    for (std::uint32_t item = 0u; item < 4u; ++item) {
      const std::uint32_t output_index = diagonal + item;
      if (item >= tile || output_index >= merged_count) break;
      const bool choose_older = older_index < older_count &&
          (current_index >= current_count ||
           older[older_index].key <= current[current_index].key);
      if (choose_older) merged[output_index] = older[older_index++];
      else merged[output_index] = current[current_index++];
    }
    __syncwarp();
    bool winner[4]{};
    unsigned masks[4]{};
    std::uint32_t next_count = 0u;
    for (std::uint32_t group = 0u; group < 4u; ++group) {
      const std::uint32_t index = lane + group * 32u;
      winner[group] = index < merged_count &&
          (index + 1u == merged_count ||
           merged[index].key != merged[index + 1u].key);
      masks[group] = __ballot_sync(0xffffffffu, winner[group]);
      next_count += __popc(masks[group]);
    }
    std::uint32_t base = 0u;
    for (std::uint32_t group = 0u; group < 4u; ++group) {
      if (winner[group])
        current[base + __popc(masks[group] & before)] =
            merged[lane + group * 32u];
      base += __popc(masks[group]);
    }
    __syncwarp();
    current_count = next_count;
  }
  for (std::uint32_t index = lane; index < current_count; index += 32u)
    arena[allocation.offset + index] = current[index];
  if (lane == 0u) pending[q].count = current_count;
}

__global__ void fused_raw_publish_kernel(
    const RawAssignment *__restrict__ assignments,
    const std::uint32_t *__restrict__ raw_offsets,
    const std::uint32_t *__restrict__ batch_bases,
    const std::uint32_t *__restrict__ raw_counts,
    Row *__restrict__ arena, Pending *__restrict__ pending,
    std::uint32_t *__restrict__ overflow) {
  constexpr std::uint32_t raw_capacity = 64u;
  __shared__ TaggedRow raw_shared[kWarpsPerBlock][raw_capacity];
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t q = blockIdx.x * kWarpsPerBlock + warp;
  if (q >= kQuotients || !pending[q].valid || pending[q].level != 0u) return;
  const std::uint32_t raw_count = raw_counts[q];
  if (raw_count > raw_capacity) {
    return;
  }
  TaggedRow *raw = raw_shared[warp];
  const std::uint32_t sort_size = max(32u, 1u << size_class_for(raw_count));
  for (std::uint32_t i = lane; i < sort_size; i += 32u) {
    if (i < raw_count) {
      const RawAssignment item = load_raw_ordinal(
          assignments, raw_offsets, batch_bases, q, i);
      raw[i] = {item.row, item.logical_position};
    } else {
      raw[i] = {{0u, 0u, 0u}, kInvalid};
    }
  }
  __syncwarp();
  for (std::uint32_t width = 2u; width <= sort_size; width <<= 1u) {
    for (std::uint32_t stride = width >> 1u; stride; stride >>= 1u) {
      for (std::uint32_t group = 0u; group < sort_size / 32u; ++group) {
        const std::uint32_t index = lane + group * 32u;
        const std::uint32_t other_index = index ^ stride;
        if (other_index > index) {
          const TaggedRow x = raw[index], y = raw[other_index];
          const bool ascending = (index & width) == 0u;
          const bool swap = ascending ? tagged_less(y, x) : tagged_less(x, y);
          if (swap) {
            raw[index] = y;
            raw[other_index] = x;
          }
        }
      }
      __syncwarp();
    }
  }
  bool winner[2]{};
  unsigned masks[2]{};
  std::uint32_t winner_count = 0u;
  for (std::uint32_t group = 0u; group < sort_size / 32u; ++group) {
    const std::uint32_t index = lane + group * 32u;
    winner[group] = raw[index].age != kInvalid &&
        (index + 1u == sort_size || raw[index + 1u].age == kInvalid ||
         raw[index].row.key != raw[index + 1u].row.key);
    masks[group] = __ballot_sync(0xffffffffu, winner[group]);
    winner_count += __popc(masks[group]);
  }
  const unsigned before = lane == 0u ? 0u : ((1u << lane) - 1u);
  std::uint32_t base = 0u;
  const Pending allocation = pending[q];
  for (std::uint32_t group = 0u; group < sort_size / 32u; ++group) {
    if (winner[group])
      arena[allocation.offset + base + __popc(masks[group] & before)] =
          raw[lane + group * 32u].row;
    base += __popc(masks[group]);
  }
  if (lane == 0u) pending[q].count = winner_count;
}

// One warp owns a quotient.  It sorts at most 256 records by full key and age.
// Older rows sort before newer rows for equal keys, so the final equal row wins.
__global__ void merge_small_kernel(
    const Row *__restrict__ epoch_rows,
    const std::uint32_t *__restrict__ epoch_offsets,
    const std::uint32_t *__restrict__ epoch_counts,
    Row *__restrict__ arena,
    const Descriptor *__restrict__ descriptors,
    const std::uint8_t *__restrict__ masks,
    Pending *__restrict__ pending, std::uint32_t *__restrict__ dense_list,
    std::uint32_t *__restrict__ dense_count,
    std::uint32_t *__restrict__ overflow) {
  __shared__ TaggedRow items[kWarpsPerBlock][kSmallMaximum];
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t q = blockIdx.x * kWarpsPerBlock + warp;
  if (q >= kQuotients) return;
  const std::uint32_t incoming = epoch_counts[q];
  if (incoming == 0u || !pending[q].valid) return;
  const Pending allocation = pending[q];
  const std::uint32_t depth = allocation.level;
  // An admitted epoch stream is already exact and sorted. With no occupied
  // class below it, publication needs only allocation and a coalesced copy.
  if (depth == 0u) {
    for (std::uint32_t i = lane; i < incoming; i += 32u)
      arena[allocation.offset + i] = epoch_rows[epoch_offsets[q] + i];
    return;
  }
  const std::uint32_t total = allocation.count;
  // Ordinary carries are handled by the iterative rank-merge kernel below.
  if (total <= 128u) return;
  if (total > kSmallMaximum) {
    return;
  }
  TaggedRow *local = items[warp];
  const std::uint32_t sort_size = max(32u, 1u << size_class_for(total));
  for (std::uint32_t i = lane; i < sort_size; i += 32u)
    local[i] = i < total
        ? load_stream_ordinal(i, epoch_rows, epoch_offsets[q], incoming,
                              arena, descriptors, q, depth)
        : TaggedRow{{0u, 0u, 0u}, kInvalid};
  __syncwarp();
  for (std::uint32_t width = 2u; width <= sort_size; width <<= 1u) {
    for (std::uint32_t stride = width >> 1u; stride; stride >>= 1u) {
      for (std::uint32_t group = 0u; group < sort_size / 32u; ++group) {
        const std::uint32_t index = lane + group * 32u;
        const std::uint32_t other_index = index ^ stride;
        if (other_index > index) {
          const TaggedRow x = local[index];
          const TaggedRow y = local[other_index];
          const bool ascending = (index & width) == 0u;
          const bool swap = ascending ? tagged_less(y, x) : tagged_less(x, y);
          if (swap) {
            local[index] = y;
            local[other_index] = x;
          }
        }
      }
      __syncwarp();
    }
  }
  bool winner[kSmallMaximum / 32u];
  unsigned winner_masks[kSmallMaximum / 32u];
  std::uint32_t winner_count = 0u;
  for (std::uint32_t group = 0u; group < sort_size / 32u; ++group) {
    const std::uint32_t index = lane + group * 32u;
    const TaggedRow item = local[index];
    winner[group] = item.age != kInvalid &&
        (index + 1u == sort_size || local[index + 1u].age == kInvalid ||
         item.row.key != local[index + 1u].row.key);
    winner_masks[group] = __ballot_sync(0xffffffffu, winner[group]);
    winner_count += __popc(winner_masks[group]);
  }
  if (lane == 0u) pending[q].count = winner_count;
  __syncwarp();
  const Pending output = pending[q];
  if (!output.valid) return;
  const unsigned before = lane == 0u ? 0u : ((1u << lane) - 1u);
  std::uint32_t base = 0u;
  for (std::uint32_t group = 0u; group < sort_size / 32u; ++group) {
    if (winner[group])
      arena[output.offset + base + __popc(winner_masks[group] & before)] =
          local[lane + group * 32u].row;
    base += __popc(winner_masks[group]);
  }
}

// Iteratively merge the already-sorted incoming stream with each consumed
// older class. For equal keys, older rows receive the lower rank; retaining the
// final equal row therefore implements exact newest-wins without re-sorting.
__global__ void merge_rank_iterative_kernel(
    const Row *__restrict__ epoch_rows,
    const std::uint32_t *__restrict__ epoch_offsets,
    const std::uint32_t *__restrict__ epoch_counts,
    Row *__restrict__ arena, const Descriptor *__restrict__ descriptors,
    Pending *__restrict__ pending) {
  constexpr std::uint32_t warps_per_block = 4u;
  __shared__ Row current_shared[warps_per_block][128];
  __shared__ Row older_shared[warps_per_block][128];
  __shared__ Row merged_shared[warps_per_block][128];
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t q = blockIdx.x * warps_per_block + warp;
  if (q >= kQuotients || !pending[q].valid) return;
  const Pending allocation = pending[q];
  std::uint32_t raw_count = epoch_counts[q];
  for (std::uint32_t level = 0u; level < allocation.level; ++level)
    raw_count += descriptors[descriptor_index(q, level)].count();
  if (allocation.level == 0u || raw_count > 128u) return;
  Row *current = current_shared[warp];
  Row *older = older_shared[warp];
  Row *merged = merged_shared[warp];
  std::uint32_t current_count = epoch_counts[q];
  for (std::uint32_t i = lane; i < current_count; i += 32u)
    current[i] = epoch_rows[epoch_offsets[q] + i];
  __syncwarp();
  for (std::uint32_t level = 0u; level < allocation.level; ++level) {
    const Descriptor descriptor = descriptors[descriptor_index(q, level)];
    const std::uint32_t older_count = descriptor.count();
    for (std::uint32_t i = lane; i < older_count; i += 32u)
      older[i] = arena[descriptor.offset() + i];
    __syncwarp();
    for (std::uint32_t i = lane; i < older_count; i += 32u) {
      const std::uint32_t key = older[i].key;
      std::uint32_t lo = 0u, hi = current_count;
      while (lo < hi) {
        const std::uint32_t mid = (lo + hi) >> 1u;
        if (current[mid].key < key) lo = mid + 1u;
        else hi = mid;
      }
      merged[i + lo] = older[i];
    }
    for (std::uint32_t i = lane; i < current_count; i += 32u) {
      const std::uint32_t key = current[i].key;
      std::uint32_t lo = 0u, hi = older_count;
      while (lo < hi) {
        const std::uint32_t mid = (lo + hi) >> 1u;
        if (older[mid].key <= key) lo = mid + 1u;
        else hi = mid;
      }
      merged[i + lo] = current[i];
    }
    __syncwarp();
    const std::uint32_t merged_count = current_count + older_count;
    bool winner[4];
    unsigned masks[4];
    std::uint32_t next_count = 0u;
    for (std::uint32_t group = 0u; group < 4u; ++group) {
      const std::uint32_t index = lane + group * 32u;
      winner[group] = index < merged_count &&
          (index + 1u == merged_count ||
           merged[index].key != merged[index + 1u].key);
      masks[group] = __ballot_sync(0xffffffffu, winner[group]);
      next_count += __popc(masks[group]);
    }
    const unsigned before = lane == 0u ? 0u : ((1u << lane) - 1u);
    std::uint32_t base = 0u;
    for (std::uint32_t group = 0u; group < 4u; ++group) {
      if (winner[group])
        current[base + __popc(masks[group] & before)] =
            merged[lane + group * 32u];
      base += __popc(masks[group]);
    }
    __syncwarp();
    current_count = next_count;
  }
  for (std::uint32_t i = lane; i < current_count; i += 32u)
    arena[allocation.offset + i] = current[i];
  if (lane == 0u) pending[q].count = current_count;
}

__device__ __forceinline__ Row stream_head(
    std::uint32_t stream, const Row *epoch_rows, std::uint32_t epoch_begin,
    std::uint32_t epoch_count, const Row *arena,
    const Descriptor *descriptors, std::uint32_t q, std::uint32_t depth,
    const std::uint32_t *positions) {
  if (stream == 0u) return epoch_rows[epoch_begin + positions[0]];
  const Descriptor descriptor = descriptors[descriptor_index(q, stream - 1u)];
  return arena[descriptor.offset() + positions[stream]];
}

// Correct overflow path for dense quotients.  One lane performs a k-way merge;
// this is intentionally a correctness fallback whose cost is reported.
__global__ void merge_dense_kernel(
    const Row *__restrict__ epoch_rows,
    const std::uint32_t *__restrict__ epoch_offsets,
    const std::uint32_t *__restrict__ epoch_counts,
    Row *__restrict__ arena,
    const Descriptor *__restrict__ descriptors,
    const std::uint8_t *__restrict__ masks,
    Pending *__restrict__ pending, const std::uint32_t *__restrict__ dense_list,
    const std::uint32_t *__restrict__ dense_count,
    std::uint32_t *__restrict__ overflow) {
  const std::uint32_t task = blockIdx.x;
  if (task >= *dense_count || threadIdx.x != 0u) return;
  const std::uint32_t q = dense_list[task];
  const std::uint32_t incoming = epoch_counts[q];
  const std::uint32_t depth = static_cast<std::uint32_t>(__ffs(~masks[q]) - 1);
  std::uint32_t counts[kLevels + 1u]{};
  counts[0] = incoming;
  std::uint32_t total = incoming;
  for (std::uint32_t level = 0u; level < depth; ++level) {
    counts[level + 1u] = descriptors[descriptor_index(q, level)].count();
    total += counts[level + 1u];
  }
  const Pending allocation = pending[q];
  const std::uint32_t offset = allocation.offset;
  std::uint32_t positions[kLevels + 1u]{};
  std::uint32_t output_count = 0u;
  while (true) {
    std::uint32_t minimum = kInvalid;
    std::uint32_t chosen = kInvalid;
    for (std::uint32_t stream = 0u; stream <= depth; ++stream) {
      if (positions[stream] >= counts[stream]) continue;
      const Row row = stream_head(stream, epoch_rows, epoch_offsets[q], incoming,
                                  arena, descriptors, q, depth, positions);
      if (chosen == kInvalid || row.key < minimum) {
        minimum = row.key;
        chosen = stream;
      }
    }
    if (chosen == kInvalid) break;
    const Row winner = stream_head(chosen, epoch_rows, epoch_offsets[q],
                                   incoming, arena, descriptors, q, depth,
                                   positions);
    arena[offset + output_count++] = winner;
    for (std::uint32_t stream = 0u; stream <= depth; ++stream) {
      if (positions[stream] >= counts[stream]) continue;
      const Row row = stream_head(stream, epoch_rows, epoch_offsets[q], incoming,
                                  arena, descriptors, q, depth, positions);
      if (row.key == minimum) ++positions[stream];
    }
  }
  pending[q].count = output_count;
}

// Publication is separated from allocation/merge so no newly freed page can
// be reused while another owner is still reading it.
__global__ void publish_and_release_kernel(
    Row *arena, Descriptor *descriptors, std::uint8_t *masks,
    std::uint32_t *local_free_heads, const Pending *pending,
    std::uint16_t generation) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients || !pending[q].valid) return;
  const Pending item = pending[q];
  for (std::uint32_t level = 0u; level < item.level; ++level) {
    Descriptor &old = descriptors[descriptor_index(q, level)];
    release_extent(arena, local_free_heads, q, old.offset(), old.size_class());
    old = {};
  }
  descriptors[descriptor_index(q, item.level)] = Descriptor::make(
      item.offset, item.count, item.size_class, generation);
  const std::uint32_t lower = (1u << item.level) - 1u;
  masks[q] = static_cast<std::uint8_t>((masks[q] & ~lower) |
                                      (1u << item.level));
}

__device__ __forceinline__ std::uint32_t lower_bound_rows(
    const Row *rows, std::uint32_t count, std::uint32_t key) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    if (rows[mid].key < key) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

__device__ __forceinline__ std::uint32_t upper_bound_rows(
    const Row *rows, std::uint32_t count, std::uint32_t key) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    if (rows[mid].key <= key) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

__global__ void lookup_kernel(
    const std::uint32_t *__restrict__ queries, std::uint64_t *__restrict__ out,
    std::uint32_t query_count, const Row *__restrict__ arena,
    const Descriptor *__restrict__ descriptors,
    const Row *__restrict__ base,
    const std::uint32_t *__restrict__ rank23) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= query_count) return;
  const std::uint32_t key = queries[i];
  const std::uint32_t q = key >> 16u;
  Descriptor local_descriptors[kLevels];
#pragma unroll
  for (std::uint32_t level = 0u; level < kLevels; ++level)
    local_descriptors[level] = descriptors[descriptor_index(q, level)];
#pragma unroll
  for (std::uint32_t level = 0u; level < kLevels; ++level) {
    const Descriptor descriptor = local_descriptors[level];
    if (descriptor.count() == 0u) continue;
    const Row *rows = arena + descriptor.offset();
    const std::uint32_t position = lower_bound_rows(rows, descriptor.count(), key);
    if (position < descriptor.count() && rows[position].key == key) {
      const Row row = rows[position];
      out[i] = (row.flags & kTombstone)
          ? 0ull : (1ull << 63u) | std::uint64_t{row.value};
      return;
    }
  }
  const std::uint32_t rank_cell = key >> (32u - kRankBits);
  const std::uint32_t begin = rank23[rank_cell];
  const std::uint32_t end = rank23[rank_cell + 1u];
  const Row *base_rows = base + begin;
  const std::uint32_t count = end - begin;
  const std::uint32_t position = lower_bound_rows(base_rows, count, key);
  out[i] = position < count && base_rows[position].key == key
      ? (1ull << 63u) | std::uint64_t{base_rows[position].value} : 0ull;
}

// Preserve the newest-class early exit, then overlap the independent binary
// searches of every older immutable class and the rank23 BaseRun interval.
// Candidate priority is applied only after all independent searches finish.
__global__ void lookup_interleaved_kernel(
    const std::uint32_t *__restrict__ queries, std::uint64_t *__restrict__ out,
    std::uint32_t query_count, const Row *__restrict__ arena,
    const Descriptor *__restrict__ descriptors,
    const Row *__restrict__ base,
    const std::uint32_t *__restrict__ rank23) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= query_count) return;
  const std::uint32_t key = queries[i];
  const std::uint32_t q = key >> 16u;
  Descriptor local_descriptors[kLevels];
#pragma unroll
  for (std::uint32_t level = 0u; level < kLevels; ++level)
    local_descriptors[level] = descriptors[descriptor_index(q, level)];

  // Temporal locality makes this early exit valuable and it avoids reading
  // every older stream for a newest-class hit.
  const Descriptor newest = local_descriptors[0];
  if (newest.count()) {
    const Row *rows = arena + newest.offset();
    const std::uint32_t position = lower_bound_rows(rows, newest.count(), key);
    if (position < newest.count() && rows[position].key == key) {
      const Row row = rows[position];
      out[i] = (row.flags & kTombstone)
          ? 0ull : (1ull << 63u) | std::uint64_t{row.value};
      return;
    }
  }

  constexpr std::uint32_t kChains = kLevels; // levels 1..3 plus BaseRun
  const Row *chain_rows[kChains];
  std::uint32_t lo[kChains]{}, hi[kChains]{}, mid[kChains]{}, probe[kChains]{};
#pragma unroll
  for (std::uint32_t chain = 0u; chain + 1u < kChains; ++chain) {
    const Descriptor descriptor = local_descriptors[chain + 1u];
    chain_rows[chain] = arena + descriptor.offset();
    hi[chain] = descriptor.count();
  }
  const std::uint32_t rank_cell = key >> (32u - kRankBits);
  const std::uint32_t base_begin = rank23[rank_cell];
  const std::uint32_t base_end = rank23[rank_cell + 1u];
  chain_rows[kChains - 1u] = base + base_begin;
  hi[kChains - 1u] = base_end - base_begin;

  // Seventeen iterations cover the packed descriptor's exact 17-bit count.
  // Loads and comparisons are in separate unrolled loops so the compiler can
  // keep all active chains outstanding concurrently.
#pragma unroll
  for (std::uint32_t step = 0u; step < 17u; ++step) {
    bool any = false;
#pragma unroll
    for (std::uint32_t chain = 0u; chain < kChains; ++chain) {
      const bool active = lo[chain] < hi[chain];
      any |= active;
      mid[chain] = (lo[chain] + hi[chain]) >> 1u;
      probe[chain] = active ? chain_rows[chain][mid[chain]].key : key;
    }
    if (!any) break;
#pragma unroll
    for (std::uint32_t chain = 0u; chain < kChains; ++chain) {
      if (lo[chain] < hi[chain]) {
        if (probe[chain] < key) lo[chain] = mid[chain] + 1u;
        else hi[chain] = mid[chain];
      }
    }
  }

#pragma unroll
  for (std::uint32_t chain = 0u; chain + 1u < kChains; ++chain) {
    const std::uint32_t count = local_descriptors[chain + 1u].count();
    if (lo[chain] < count && chain_rows[chain][lo[chain]].key == key) {
      const Row row = chain_rows[chain][lo[chain]];
      out[i] = (row.flags & kTombstone)
          ? 0ull : (1ull << 63u) | std::uint64_t{row.value};
      return;
    }
  }
  const std::uint32_t base_count = base_end - base_begin;
  const std::uint32_t base_chain = kChains - 1u;
  out[i] = lo[base_chain] < base_count &&
           chain_rows[base_chain][lo[base_chain]].key == key
      ? (1ull << 63u) |
            std::uint64_t{chain_rows[base_chain][lo[base_chain]].value}
      : 0ull;
}

struct SourceCursor {
  const Row *rows;
  std::uint32_t position;
  std::uint32_t count;
};

__device__ std::uint32_t traverse_visible_quotient(
    std::uint32_t q, std::uint32_t low, std::uint32_t high,
    const Row *base, const std::uint32_t *base_offsets, const Row *arena,
    const Descriptor *descriptors, std::uint8_t mask, Row *output) {
  SourceCursor sources[kLevels + 1u]{};
  for (std::uint32_t level = 0u; level < kLevels; ++level) {
    if (mask & (1u << level)) {
      const Descriptor descriptor = descriptors[descriptor_index(q, level)];
      sources[level] = {arena + descriptor.offset(), 0u, descriptor.count()};
      sources[level].position =
          lower_bound_rows(sources[level].rows, sources[level].count, low);
    }
  }
  const std::uint32_t base_begin = base_offsets[q];
  const std::uint32_t base_count = base_offsets[q + 1u] - base_begin;
  sources[kLevels] = {base + base_begin, 0u, base_count};
  sources[kLevels].position = lower_bound_rows(
      sources[kLevels].rows, sources[kLevels].count, low);
  std::uint32_t produced = 0u;
  while (true) {
    std::uint32_t minimum = kInvalid;
    std::uint32_t chosen = kInvalid;
    for (std::uint32_t stream = 0u; stream <= kLevels; ++stream) {
      if (sources[stream].position >= sources[stream].count) continue;
      const Row row = sources[stream].rows[sources[stream].position];
      if (row.key > high) continue;
      if (chosen == kInvalid || row.key < minimum) {
        minimum = row.key;
        chosen = stream; // lower stream number is newer and wins equal keys
      }
    }
    if (chosen == kInvalid) break;
    const Row winner = sources[chosen].rows[sources[chosen].position];
    if ((winner.flags & kTombstone) == 0u && output)
      output[produced] = {winner.key, winner.value, 0u};
    if ((winner.flags & kTombstone) == 0u) ++produced;
    for (std::uint32_t stream = 0u; stream <= kLevels; ++stream) {
      if (sources[stream].position >= sources[stream].count) continue;
      if (sources[stream].rows[sources[stream].position].key == minimum)
        ++sources[stream].position;
    }
  }
  return produced;
}

struct NoAggregate {
  using State = unsigned long long;
  static constexpr bool enabled = false;
  __device__ static State identity() { return 0ull; }
  __device__ static State consume(State state, const Row &) { return state; }
  __device__ State operator()(State a, State b) const { return a + b; }
};

struct SumRowsAggregate {
  using State = unsigned long long;
  static constexpr bool enabled = true;
  __device__ static State identity() { return 0ull; }
  __device__ static State consume(State state, const Row &row) {
    return state + row.value;
  }
  __device__ State operator()(State a, State b) const { return a + b; }
};

// Exact serial fallback for an associative consumer when one severely skewed
// quotient cannot fit the cooperative shared-memory tile.
template <class Aggregate>
__device__ typename Aggregate::State aggregate_visible_quotient(
    std::uint32_t q, std::uint32_t low, std::uint32_t high,
    const Row *base, const std::uint32_t *base_offsets, const Row *arena,
    const Descriptor *descriptors, std::uint8_t mask) {
  SourceCursor sources[kLevels + 1u]{};
  for (std::uint32_t level = 0u; level < kLevels; ++level) {
    if (mask & (1u << level)) {
      const Descriptor descriptor = descriptors[descriptor_index(q, level)];
      sources[level] = {arena + descriptor.offset(), 0u, descriptor.count()};
      sources[level].position =
          lower_bound_rows(sources[level].rows, sources[level].count, low);
    }
  }
  const std::uint32_t base_begin = base_offsets[q];
  const std::uint32_t base_count = base_offsets[q + 1u] - base_begin;
  sources[kLevels] = {base + base_begin, 0u, base_count};
  sources[kLevels].position = lower_bound_rows(
      sources[kLevels].rows, sources[kLevels].count, low);
  typename Aggregate::State state = Aggregate::identity();
  while (true) {
    std::uint32_t minimum = kInvalid, chosen = kInvalid;
    for (std::uint32_t stream = 0u; stream <= kLevels; ++stream) {
      if (sources[stream].position >= sources[stream].count) continue;
      const Row row = sources[stream].rows[sources[stream].position];
      if (row.key > high) continue;
      if (chosen == kInvalid || row.key < minimum) {
        minimum = row.key;
        chosen = stream;
      }
    }
    if (chosen == kInvalid) break;
    const Row winner = sources[chosen].rows[sources[chosen].position];
    if ((winner.flags & kTombstone) == 0u)
      state = Aggregate::consume(state, winner);
    for (std::uint32_t stream = 0u; stream <= kLevels; ++stream) {
      if (sources[stream].position < sources[stream].count &&
          sources[stream].rows[sources[stream].position].key == minimum)
        ++sources[stream].position;
    }
  }
  return state;
}

__global__ void count_visible_kernel(
    std::uint32_t low, std::uint32_t high, const Row *base,
    const std::uint32_t *base_offsets, const Row *arena,
    const Descriptor *descriptors, const std::uint8_t *masks,
    std::uint32_t q_begin, std::uint32_t q_count, std::uint32_t *counts) {
  const std::uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local > q_count) return;
  if (local == q_count) {
    counts[local] = 0u;
    return;
  }
  const std::uint32_t q = q_begin + local;
  counts[local] = traverse_visible_quotient(q, low, high, base, base_offsets,
                                            arena, descriptors, masks[q],
                                            nullptr);
}

__global__ void emit_visible_kernel(
    std::uint32_t low, std::uint32_t high, const Row *base,
    const std::uint32_t *base_offsets, const Row *arena,
    const Descriptor *descriptors, const std::uint8_t *masks,
    std::uint32_t q_begin, std::uint32_t q_count,
    const std::uint32_t *offsets, Row *output) {
  const std::uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local >= q_count) return;
  const std::uint32_t q = q_begin + local;
  traverse_visible_quotient(q, low, high, base, base_offsets, arena,
                            descriptors, masks[q], output + offsets[local]);
}

template <bool Emit, class Aggregate = NoAggregate>
__global__ void cooperative_visible_kernel(
    std::uint32_t low, std::uint32_t high, const Row *base,
    const std::uint32_t *base_offsets, const Row *arena,
    const Descriptor *descriptors, const std::uint8_t *masks,
    std::uint32_t q_begin,
    std::uint32_t q_count, const std::uint32_t *offsets,
    std::uint32_t *counts, Row *output, std::uint32_t *overflow,
    typename Aggregate::State *aggregate_partials) {
  using BlockScan = cub::BlockScan<std::uint32_t, kThreads>;
  using BlockReduce =
      cub::BlockReduce<typename Aggregate::State, kThreads>;
  __shared__ Row current[kRangeCapacity];
  __shared__ Row older[kRangeCapacity];
  __shared__ Row merged[kRangeCapacity];
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ std::uint32_t current_count_shared;
  __shared__ std::uint32_t serial_fallback_shared;
  const std::uint32_t local_q = blockIdx.x;
  if (local_q >= q_count) return;
  const std::uint32_t q = q_begin + local_q;
  if (threadIdx.x == 0u) {
    current_count_shared = 0u;
    std::uint32_t physical_rows = base_offsets[q + 1u] - base_offsets[q];
    for (std::uint32_t level = 0u; level < kLevels; ++level)
      physical_rows += descriptors[descriptor_index(q, level)].count();
    serial_fallback_shared = physical_rows > kRangeCapacity;
  }
  __syncthreads();

  // Severe skew can exceed the fixed cooperative shared-memory tile. The CTA
  // retains exactness by assigning lane zero the existing generic traversal.
  // Ordinary locations still use the cooperative path below.
  if (serial_fallback_shared) {
    if (threadIdx.x == 0u) {
      if constexpr (Aggregate::enabled) {
        aggregate_partials[local_q] =
            aggregate_visible_quotient<Aggregate>(
                q, low, high, base, base_offsets, arena, descriptors,
                masks[q]);
      } else {
        Row *destination = nullptr;
        if constexpr (Emit) destination = output + offsets[local_q];
        const std::uint32_t produced = traverse_visible_quotient(
            q, low, high, base, base_offsets, arena, descriptors, masks[q],
            destination);
        if constexpr (!Emit) counts[local_q] = produced;
      }
    }
    return;
  }

  // Streams are visited newest to oldest: classes 0..3, then BaseRun.
  for (std::uint32_t source = 0u; source <= kLevels; ++source) {
    const Row *source_rows{};
    std::uint32_t source_count{};
    if (source < kLevels) {
      const Descriptor descriptor = descriptors[descriptor_index(q, source)];
      source_rows = arena + descriptor.offset();
      source_count = descriptor.count();
    } else {
      source_rows = base + base_offsets[q];
      source_count = base_offsets[q + 1u] - base_offsets[q];
    }
    if (source_count == 0u) continue;
    const std::uint32_t current_count = current_count_shared;
    if (current_count == 0u) {
      for (std::uint32_t i = threadIdx.x; i < source_count;
           i += blockDim.x)
        current[i] = source_rows[i];
      __syncthreads();
      if (threadIdx.x == 0u) current_count_shared = source_count;
      __syncthreads();
      continue;
    }
    if (current_count + source_count > kRangeCapacity) {
      if (threadIdx.x == 0u) atomicAdd(overflow, 1u);
      return;
    }
    for (std::uint32_t i = threadIdx.x; i < source_count; i += blockDim.x)
      older[i] = source_rows[i];
    __syncthreads();
    // Existing current rows are newer and therefore sort after older equal
    // rows. Keeping the final equal key selects the newest assignment.
    for (std::uint32_t i = threadIdx.x; i < source_count; i += blockDim.x) {
      const std::uint32_t key = older[i].key;
      std::uint32_t lo = 0u, hi = current_count;
      while (lo < hi) {
        const std::uint32_t mid = (lo + hi) >> 1u;
        if (current[mid].key < key) lo = mid + 1u;
        else hi = mid;
      }
      merged[i + lo] = older[i];
    }
    for (std::uint32_t i = threadIdx.x; i < current_count; i += blockDim.x) {
      const std::uint32_t key = current[i].key;
      std::uint32_t lo = 0u, hi = source_count;
      while (lo < hi) {
        const std::uint32_t mid = (lo + hi) >> 1u;
        if (older[mid].key <= key) lo = mid + 1u;
        else hi = mid;
      }
      merged[i + lo] = current[i];
    }
    __syncthreads();
    const std::uint32_t merged_count = current_count + source_count;
    const std::uint32_t chunk_begin = threadIdx.x * 4u;
    std::uint32_t local_winners = 0u;
    bool winner[4]{};
#pragma unroll
    for (std::uint32_t j = 0u; j < 4u; ++j) {
      const std::uint32_t index = chunk_begin + j;
      winner[j] = index < merged_count &&
          (index + 1u == merged_count ||
           merged[index].key != merged[index + 1u].key);
      local_winners += winner[j];
    }
    std::uint32_t thread_base{}, winner_count{};
    BlockScan(scan_storage).ExclusiveSum(local_winners, thread_base,
                                          winner_count);
    std::uint32_t local_rank = 0u;
#pragma unroll
    for (std::uint32_t j = 0u; j < 4u; ++j) {
      if (winner[j])
        current[thread_base + local_rank++] = merged[chunk_begin + j];
    }
    __syncthreads();
    if (threadIdx.x == 0u) current_count_shared = winner_count;
    __syncthreads();
  }

  const std::uint32_t current_count = current_count_shared;
  const std::uint32_t chunk_begin = threadIdx.x * 4u;
  std::uint32_t local_visible = 0u;
  typename Aggregate::State local_aggregate = Aggregate::identity();
  bool visible[4]{};
#pragma unroll
  for (std::uint32_t j = 0u; j < 4u; ++j) {
    const std::uint32_t index = chunk_begin + j;
    visible[j] = index < current_count && current[index].key >= low &&
        current[index].key <= high &&
        (current[index].flags & kTombstone) == 0u;
    local_visible += visible[j];
    if constexpr (Aggregate::enabled) {
      if (visible[j])
        local_aggregate =
            Aggregate::consume(local_aggregate, current[index]);
    }
  }
  if constexpr (Aggregate::enabled) {
    const typename Aggregate::State aggregate =
        BlockReduce(reduce_storage).Reduce(local_aggregate, Aggregate{});
    if (threadIdx.x == 0u) aggregate_partials[local_q] = aggregate;
    return;
  }
  std::uint32_t thread_base{}, visible_count{};
  BlockScan(scan_storage).ExclusiveSum(local_visible, thread_base,
                                        visible_count);
  if constexpr (Emit) {
    std::uint32_t local_rank = 0u;
    const std::uint32_t quotient_base = offsets[local_q];
#pragma unroll
    for (std::uint32_t j = 0u; j < 4u; ++j) {
      if (visible[j]) {
        const Row row = current[chunk_begin + j];
        output[quotient_base + thread_base + local_rank++] =
            {row.key, row.value, 0u};
      }
    }
  } else if (threadIdx.x == 0u) {
    counts[local_q] = visible_count;
  }
}

// Same exact visibility rule as cooperative_visible_kernel, but stable
// merge-path tiles read the immutable older stream directly and need only two
// shared row buffers.
__device__ unsigned long long sum_visible_quotient_with_pending(
    std::uint32_t q, std::uint32_t low, std::uint32_t high,
    const RawAssignment *raw, const std::uint32_t *raw_offsets,
    const std::uint32_t *batch_bases, std::uint32_t pending_batches,
    const Row *arena, const Descriptor *descriptors, const Row *base,
    const std::uint32_t *base_offsets);

template <bool Emit, class Aggregate = NoAggregate,
          bool PendingAlreadySorted = false>
__global__ void cooperative_visible_merge_path_kernel(
    std::uint32_t low, std::uint32_t high, const Row *base,
    const std::uint32_t *base_offsets, const Row *arena,
    const Descriptor *descriptors, const std::uint8_t *masks,
    const RawAssignment *raw, const std::uint32_t *raw_offsets,
    const std::uint32_t *batch_bases, std::uint32_t pending_batches,
    std::uint32_t q_begin, std::uint32_t q_count,
    const std::uint32_t *offsets, std::uint32_t *counts, Row *output,
    std::uint32_t *overflow,
    typename Aggregate::State *aggregate_partials) {
  using BlockScan = cub::BlockScan<std::uint32_t, kThreads>;
  using BlockReduce =
      cub::BlockReduce<typename Aggregate::State, kThreads>;
  union MergeWorkspace {
    Row merged[kRangeCapacity];
    TaggedRow pending_rows[kRangeCapacity];
  };
  __shared__ Row current[kRangeCapacity];
  __shared__ MergeWorkspace workspace;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ std::uint32_t current_count_shared;
  __shared__ std::uint32_t raw_count_shared;
  __shared__ std::uint32_t serial_fallback_shared;
  const std::uint32_t local_q = blockIdx.x;
  if (local_q >= q_count) return;
  const std::uint32_t q = q_begin + local_q;
  if (threadIdx.x == 0u) {
    current_count_shared = 0u;
    std::uint32_t physical_rows = base_offsets[q + 1u] - base_offsets[q];
    raw_count_shared = 0u;
    for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
      const std::size_t oi = std::size_t{batch} * (kQuotients + 1u) + q;
      raw_count_shared += raw_offsets[oi + 1u] - raw_offsets[oi];
    }
    physical_rows += raw_count_shared;
    for (std::uint32_t level = 0u; level < kLevels; ++level)
      physical_rows += descriptors[descriptor_index(q, level)].count();
    serial_fallback_shared = physical_rows > kRangeCapacity;
  }
  __syncthreads();
  if (serial_fallback_shared) {
    if (threadIdx.x == 0u) {
      if constexpr (Aggregate::enabled) {
        if (pending_batches)
          aggregate_partials[local_q] =
              sum_visible_quotient_with_pending(
                  q, low, high, raw, raw_offsets, batch_bases,
                  pending_batches, arena, descriptors, base, base_offsets);
        else
          aggregate_partials[local_q] =
              aggregate_visible_quotient<Aggregate>(
                  q, low, high, base, base_offsets, arena, descriptors,
                  masks[q]);
      } else {
        Row *destination = nullptr;
        if constexpr (Emit) destination = output + offsets[local_q];
        const std::uint32_t produced = traverse_visible_quotient(
            q, low, high, base, base_offsets, arena, descriptors, masks[q],
            destination);
        if constexpr (!Emit) counts[local_q] = produced;
      }
    }
    return;
  }

  // Reduce the open epoch inside the same location owner. Admission has
  // already grouped records by quotient; full-key and logical order are
  // resolved here only for sections touched by the range.
  const std::uint32_t raw_count = raw_count_shared;
  if (raw_count) {
    TaggedRow *pending_rows = workspace.pending_rows;
    if constexpr (PendingAlreadySorted) {
      // A wide operation may first stably sort the complete open epoch by
      // full key. Stable order preserves increasing logical order among
      // equal keys, so retaining the final equal record is exact.
      const std::uint32_t chunk_begin = threadIdx.x * 4u;
      const std::uint32_t pending_begin = raw_offsets[q];
      std::uint32_t local_winners = 0u;
      bool winner[4]{};
#pragma unroll
      for (std::uint32_t item = 0u; item < 4u; ++item) {
        const std::uint32_t index = chunk_begin + item;
        winner[item] = index < raw_count &&
            (index + 1u == raw_count ||
             raw[pending_begin + index].row.key !=
                 raw[pending_begin + index + 1u].row.key);
        local_winners += winner[item];
      }
      std::uint32_t thread_base{}, winner_count{};
      BlockScan(scan_storage).ExclusiveSum(local_winners, thread_base,
                                            winner_count);
      std::uint32_t local_rank = 0u;
#pragma unroll
      for (std::uint32_t item = 0u; item < 4u; ++item)
        if (winner[item])
          current[thread_base + local_rank++] =
              raw[pending_begin + chunk_begin + item].row;
      __syncthreads();
      if (threadIdx.x == 0u) current_count_shared = winner_count;
      __syncthreads();
    } else if (raw_count <= 32u) {
      // The common sparse case averages roughly one pending assignment per
      // admitted batch and quotient. Sort it entirely in the first warp's
      // registers, avoiding the block-wide barriers of a 256-lane bitonic
      // network while preserving exact key/order comparison.
      if (threadIdx.x < 32u) {
        const std::uint32_t lane = threadIdx.x;
        TaggedRow item = {{0u, 0u, 0u}, kInvalid};
        if (lane < raw_count) {
          const RawAssignment loaded = load_raw_ordinal(
              raw, raw_offsets, batch_bases, q, lane);
          item = {loaded.row, loaded.logical_position};
        }
        constexpr unsigned full_mask = 0xffffffffu;
        for (std::uint32_t width = 2u; width <= 32u; width <<= 1u) {
          for (std::uint32_t stride = width >> 1u; stride; stride >>= 1u) {
            TaggedRow other{};
            other.row.key = __shfl_xor_sync(full_mask, item.row.key, stride);
            other.row.value = __shfl_xor_sync(full_mask, item.row.value, stride);
            other.row.flags = __shfl_xor_sync(full_mask, item.row.flags, stride);
            other.age = __shfl_xor_sync(full_mask, item.age, stride);
            const bool ascending = (lane & width) == 0u;
            const bool take_min = ((lane & stride) == 0u) == ascending;
            if ((take_min && tagged_less(other, item)) ||
                (!take_min && tagged_less(item, other)))
              item = other;
          }
        }
        const std::uint32_t next_key =
            __shfl_down_sync(full_mask, item.row.key, 1u);
        const std::uint32_t next_age =
            __shfl_down_sync(full_mask, item.age, 1u);
        const bool winner = item.age != kInvalid &&
            (lane == 31u || next_age == kInvalid || item.row.key != next_key);
        const unsigned winner_mask = __ballot_sync(full_mask, winner);
        if (winner) {
          const unsigned before = lane ? ((1u << lane) - 1u) : 0u;
          current[__popc(winner_mask & before)] = item.row;
        }
        if (lane == 0u) current_count_shared = __popc(winner_mask);
      }
      __syncthreads();
    } else {
      const std::uint32_t sort_size = 1u << size_class_for(raw_count);
      for (std::uint32_t ordinal = threadIdx.x; ordinal < sort_size;
           ordinal += blockDim.x) {
        if (ordinal < raw_count) {
          const RawAssignment item = load_raw_ordinal(
              raw, raw_offsets, batch_bases, q, ordinal);
          pending_rows[ordinal] = {item.row, item.logical_position};
        } else {
          pending_rows[ordinal] = {{0u, 0u, 0u}, kInvalid};
        }
      }
      __syncthreads();
      for (std::uint32_t width = 2u; width <= sort_size; width <<= 1u) {
        for (std::uint32_t stride = width >> 1u; stride; stride >>= 1u) {
          for (std::uint32_t index = threadIdx.x; index < sort_size;
               index += blockDim.x) {
            const std::uint32_t other = index ^ stride;
            if (other > index) {
              const TaggedRow x = pending_rows[index], y = pending_rows[other];
              const bool ascending = (index & width) == 0u;
              const bool swap = ascending ? tagged_less(y, x)
                                          : tagged_less(x, y);
              if (swap) {
                pending_rows[index] = y;
                pending_rows[other] = x;
              }
            }
          }
          __syncthreads();
        }
      }
      const std::uint32_t chunk_begin = threadIdx.x * 4u;
      std::uint32_t local_winners = 0u;
      bool winner[4]{};
#pragma unroll
      for (std::uint32_t item = 0u; item < 4u; ++item) {
        const std::uint32_t index = chunk_begin + item;
        winner[item] = index < sort_size &&
            pending_rows[index].age != kInvalid &&
            (index + 1u == sort_size ||
             pending_rows[index + 1u].age == kInvalid ||
             pending_rows[index].row.key != pending_rows[index + 1u].row.key);
        local_winners += winner[item];
      }
      std::uint32_t thread_base{}, winner_count{};
      BlockScan(scan_storage).ExclusiveSum(local_winners, thread_base,
                                            winner_count);
      std::uint32_t local_rank = 0u;
#pragma unroll
      for (std::uint32_t item = 0u; item < 4u; ++item)
        if (winner[item])
          current[thread_base + local_rank++] =
              pending_rows[chunk_begin + item].row;
      __syncthreads();
      if (threadIdx.x == 0u) current_count_shared = winner_count;
      __syncthreads();
    }
  }

  for (std::uint32_t source = 0u; source <= kLevels; ++source) {
    const Row *source_rows{};
    std::uint32_t source_count{};
    if (source < kLevels) {
      const Descriptor descriptor = descriptors[descriptor_index(q, source)];
      source_rows = arena + descriptor.offset();
      source_count = descriptor.count();
    } else {
      source_rows = base + base_offsets[q];
      source_count = base_offsets[q + 1u] - base_offsets[q];
    }
    if (source_count == 0u) continue;
    const std::uint32_t current_count = current_count_shared;
    if (current_count == 0u) {
      for (std::uint32_t index = threadIdx.x; index < source_count;
           index += blockDim.x)
        current[index] = source_rows[index];
      __syncthreads();
      if (threadIdx.x == 0u) current_count_shared = source_count;
      __syncthreads();
      continue;
    }
    const std::uint32_t merged_count = current_count + source_count;
    if (merged_count > kRangeCapacity) {
      if (threadIdx.x == 0u) atomicAdd(overflow, 1u);
      return;
    }
    const std::uint32_t tile =
        (merged_count + blockDim.x - 1u) / blockDim.x;
    const std::uint32_t diagonal = min(threadIdx.x * tile, merged_count);
    std::uint32_t partition_low = diagonal > current_count
        ? diagonal - current_count : 0u;
    std::uint32_t partition_high = min(diagonal, source_count);
    while (partition_low < partition_high) {
      const std::uint32_t source_index =
          (partition_low + partition_high) >> 1u;
      const std::uint32_t current_index = diagonal - source_index;
      if (source_index < source_count && current_index > 0u &&
          current[current_index - 1u].key >= source_rows[source_index].key)
        partition_low = source_index + 1u;
      else
        partition_high = source_index;
    }
    std::uint32_t source_index = partition_low;
    std::uint32_t current_index = diagonal - source_index;
#pragma unroll
    for (std::uint32_t item = 0u; item < 4u; ++item) {
      const std::uint32_t output_index = diagonal + item;
      if (item >= tile || output_index >= merged_count) break;
      const bool choose_source = source_index < source_count &&
          (current_index >= current_count ||
           source_rows[source_index].key <= current[current_index].key);
      if (choose_source)
        workspace.merged[output_index] = source_rows[source_index++];
      else
        workspace.merged[output_index] = current[current_index++];
    }
    __syncthreads();
    const std::uint32_t chunk_begin = threadIdx.x * 4u;
    std::uint32_t local_winners = 0u;
    bool winner[4]{};
#pragma unroll
    for (std::uint32_t item = 0u; item < 4u; ++item) {
      const std::uint32_t index = chunk_begin + item;
      winner[item] = index < merged_count &&
          (index + 1u == merged_count ||
           workspace.merged[index].key != workspace.merged[index + 1u].key);
      local_winners += winner[item];
    }
    std::uint32_t thread_base{}, winner_count{};
    BlockScan(scan_storage).ExclusiveSum(local_winners, thread_base,
                                          winner_count);
    std::uint32_t local_rank = 0u;
#pragma unroll
    for (std::uint32_t item = 0u; item < 4u; ++item)
      if (winner[item])
        current[thread_base + local_rank++] =
            workspace.merged[chunk_begin + item];
    __syncthreads();
    if (threadIdx.x == 0u) current_count_shared = winner_count;
    __syncthreads();
  }

  const std::uint32_t current_count = current_count_shared;
  const std::uint32_t chunk_begin = threadIdx.x * 4u;
  std::uint32_t local_visible = 0u;
  typename Aggregate::State local_aggregate = Aggregate::identity();
  bool visible[4]{};
#pragma unroll
  for (std::uint32_t item = 0u; item < 4u; ++item) {
    const std::uint32_t index = chunk_begin + item;
    visible[item] = index < current_count && current[index].key >= low &&
        current[index].key <= high &&
        (current[index].flags & kTombstone) == 0u;
    local_visible += visible[item];
    if constexpr (Aggregate::enabled) {
      if (visible[item])
        local_aggregate =
            Aggregate::consume(local_aggregate, current[index]);
    }
  }
  if constexpr (Aggregate::enabled) {
    const typename Aggregate::State aggregate =
        BlockReduce(reduce_storage).Reduce(local_aggregate, Aggregate{});
    if (threadIdx.x == 0u) aggregate_partials[local_q] = aggregate;
    return;
  }
  std::uint32_t thread_base{}, visible_count{};
  BlockScan(scan_storage).ExclusiveSum(local_visible, thread_base,
                                        visible_count);
  if constexpr (Emit) {
    std::uint32_t local_rank = 0u;
    const std::uint32_t quotient_base = offsets[local_q];
#pragma unroll
    for (std::uint32_t item = 0u; item < 4u; ++item) {
      if (visible[item]) {
        const Row row = current[chunk_begin + item];
        output[quotient_base + thread_base + local_rank++] =
            {row.key, row.value, 0u};
      }
    }
  } else if (threadIdx.x == 0u) {
    counts[local_q] = visible_count;
  }
}

// Wide aggregate/report traversal specialized to GPULSMOpt2's shape: update
// streams are small compared with the immutable BaseRun section. Resolve only
// those streams into one exact newest-winner set, then visit Base rows once and
// suppress a Base row iff the winner set contains its full key. This avoids
// repeatedly copying the BaseRun through shared merge buffers while retaining
// the same row-wise visibility rule used for generic reporting.
template <class Aggregate>
__global__ void cooperative_update_winners_probe_base_kernel(
    std::uint32_t low, std::uint32_t high, const Row *base,
    const std::uint32_t *base_offsets, const Row *arena,
    const Descriptor *descriptors, const RawAssignment *sorted_pending,
    const std::uint32_t *pending_offsets, const std::uint32_t *batch_bases,
    std::uint32_t pending_batches, std::uint32_t q_begin,
    std::uint32_t q_count, typename Aggregate::State *aggregate_partials) {
  constexpr std::uint32_t kUpdateCapacity = 512u;
  using BlockScan = cub::BlockScan<std::uint32_t, kThreads>;
  using BlockReduce = cub::BlockReduce<typename Aggregate::State, kThreads>;
  __shared__ Row current[kUpdateCapacity];
  __shared__ Row merged[kUpdateCapacity];
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ std::uint32_t current_count_shared;
  __shared__ std::uint32_t fallback_shared;
  const std::uint32_t local_q = blockIdx.x;
  if (local_q >= q_count) return;
  const std::uint32_t q = q_begin + local_q;
  if (threadIdx.x == 0u) {
    std::uint32_t update_rows = 0u;
    if (pending_batches)
      update_rows += pending_offsets[q + 1u] - pending_offsets[q];
    for (std::uint32_t level = 0u; level < kLevels; ++level)
      update_rows += descriptors[descriptor_index(q, level)].count();
    fallback_shared = update_rows > kUpdateCapacity;
    current_count_shared = 0u;
  }
  __syncthreads();
  if (fallback_shared) {
    if (threadIdx.x == 0u) {
      aggregate_partials[local_q] = sum_visible_quotient_with_pending(
          q, low, high, sorted_pending, pending_offsets, batch_bases,
          pending_batches, arena, descriptors, base, base_offsets);
    }
    return;
  }

  if (pending_batches) {
    const std::uint32_t begin = pending_offsets[q];
    const std::uint32_t count = pending_offsets[q + 1u] - begin;
    const std::uint32_t chunk_begin = threadIdx.x * 2u;
    std::uint32_t local_winners = 0u;
    bool winner[2]{};
#pragma unroll
    for (std::uint32_t item = 0u; item < 2u; ++item) {
      const std::uint32_t index = chunk_begin + item;
      winner[item] = index < count &&
          (index + 1u == count ||
           sorted_pending[begin + index].row.key !=
               sorted_pending[begin + index + 1u].row.key);
      local_winners += winner[item];
    }
    std::uint32_t thread_base{}, winner_count{};
    BlockScan(scan_storage).ExclusiveSum(local_winners, thread_base,
                                          winner_count);
    std::uint32_t local_rank = 0u;
#pragma unroll
    for (std::uint32_t item = 0u; item < 2u; ++item)
      if (winner[item])
        current[thread_base + local_rank++] =
            sorted_pending[begin + chunk_begin + item].row;
    __syncthreads();
    if (threadIdx.x == 0u) current_count_shared = winner_count;
    __syncthreads();
  }

  // Classes are visited newest to oldest. Stable merging emits the older
  // class first on an equal key; adjacent reduction therefore retains the
  // already-resolved newer assignment.
  for (std::uint32_t level = 0u; level < kLevels; ++level) {
    const Descriptor descriptor = descriptors[descriptor_index(q, level)];
    const std::uint32_t source_count = descriptor.count();
    if (!source_count) continue;
    const Row *source = arena + descriptor.offset();
    const std::uint32_t current_count = current_count_shared;
    if (!current_count) {
      for (std::uint32_t index = threadIdx.x; index < source_count;
           index += blockDim.x)
        current[index] = source[index];
      __syncthreads();
      if (threadIdx.x == 0u) current_count_shared = source_count;
      __syncthreads();
      continue;
    }
    const std::uint32_t merged_count = current_count + source_count;
    const std::uint32_t tile =
        (merged_count + blockDim.x - 1u) / blockDim.x;
    const std::uint32_t diagonal = min(threadIdx.x * tile, merged_count);
    std::uint32_t partition_low = diagonal > current_count
        ? diagonal - current_count : 0u;
    std::uint32_t partition_high = min(diagonal, source_count);
    while (partition_low < partition_high) {
      const std::uint32_t source_index =
          (partition_low + partition_high) >> 1u;
      const std::uint32_t current_index = diagonal - source_index;
      if (source_index < source_count && current_index > 0u &&
          current[current_index - 1u].key >= source[source_index].key)
        partition_low = source_index + 1u;
      else
        partition_high = source_index;
    }
    std::uint32_t source_index = partition_low;
    std::uint32_t current_index = diagonal - source_index;
#pragma unroll
    for (std::uint32_t item = 0u; item < 2u; ++item) {
      const std::uint32_t output_index = diagonal + item;
      if (item >= tile || output_index >= merged_count) break;
      const bool choose_source = source_index < source_count &&
          (current_index >= current_count ||
           source[source_index].key <= current[current_index].key);
      merged[output_index] = choose_source
          ? source[source_index++] : current[current_index++];
    }
    __syncthreads();
    const std::uint32_t chunk_begin = threadIdx.x * 2u;
    std::uint32_t local_winners = 0u;
    bool winner[2]{};
#pragma unroll
    for (std::uint32_t item = 0u; item < 2u; ++item) {
      const std::uint32_t index = chunk_begin + item;
      winner[item] = index < merged_count &&
          (index + 1u == merged_count ||
           merged[index].key != merged[index + 1u].key);
      local_winners += winner[item];
    }
    std::uint32_t thread_base{}, winner_count{};
    BlockScan(scan_storage).ExclusiveSum(local_winners, thread_base,
                                          winner_count);
    std::uint32_t local_rank = 0u;
#pragma unroll
    for (std::uint32_t item = 0u; item < 2u; ++item)
      if (winner[item])
        current[thread_base + local_rank++] = merged[chunk_begin + item];
    __syncthreads();
    if (threadIdx.x == 0u) current_count_shared = winner_count;
    __syncthreads();
  }

  typename Aggregate::State local = Aggregate::identity();
  const std::uint32_t current_count = current_count_shared;
  for (std::uint32_t index = threadIdx.x; index < current_count;
       index += blockDim.x) {
    const Row row = current[index];
    if (row.key >= low && row.key <= high &&
        (row.flags & kTombstone) == 0u)
      local = Aggregate::consume(local, row);
  }
  const std::uint32_t base_begin = base_offsets[q];
  const std::uint32_t base_end = base_offsets[q + 1u];
  for (std::uint32_t index = base_begin + threadIdx.x; index < base_end;
       index += blockDim.x) {
    const Row row = base[index];
    if (row.key < low || row.key > high) continue;
    const std::uint32_t position =
        lower_bound_rows(current, current_count, row.key);
    if (position == current_count || current[position].key != row.key)
      local = Aggregate::consume(local, row);
  }
  const typename Aggregate::State total =
      BlockReduce(reduce_storage).Reduce(local, Aggregate{});
  if (threadIdx.x == 0u) aggregate_partials[local_q] = total;
}

// Exact linear fallback for a distribution whose update set cannot fit the
// warp workspace. Every input is sorted here: a wide operation stably sorts
// the open epoch, and persistent classes/BaseRun are immutable sorted streams.
__device__ unsigned long long sum_visible_sorted_update_quotient(
    std::uint32_t q, std::uint32_t low, std::uint32_t high,
    const RawAssignment *pending, const std::uint32_t *pending_offsets,
    bool has_pending, const Row *arena, const Descriptor *descriptors,
    const Row *base, const std::uint32_t *base_offsets) {
  std::uint32_t pending_position = has_pending ? pending_offsets[q] : 0u;
  const std::uint32_t pending_end =
      has_pending ? pending_offsets[q + 1u] : 0u;
  while (pending_position < pending_end &&
         pending[pending_position].row.key < low)
    ++pending_position;
  std::uint32_t class_position[kLevels]{}, class_end[kLevels]{};
  for (std::uint32_t level = 0u; level < kLevels; ++level) {
    const Descriptor descriptor = descriptors[descriptor_index(q, level)];
    class_end[level] = descriptor.count();
    class_position[level] = lower_bound_rows(
        arena + descriptor.offset(), descriptor.count(), low);
  }
  const std::uint32_t base_begin = base_offsets[q];
  const std::uint32_t base_end = base_offsets[q + 1u];
  std::uint32_t base_position = base_begin + lower_bound_rows(
      base + base_begin, base_end - base_begin, low);
  unsigned long long sum = 0ull;
  while (true) {
    std::uint32_t minimum = kInvalid;
    bool found = false;
    if (pending_position < pending_end) {
      minimum = pending[pending_position].row.key;
      found = true;
    }
    for (std::uint32_t level = 0u; level < kLevels; ++level) {
      if (class_position[level] >= class_end[level]) continue;
      const Descriptor descriptor = descriptors[descriptor_index(q, level)];
      const std::uint32_t key =
          arena[descriptor.offset() + class_position[level]].key;
      if (!found || key < minimum) minimum = key;
      found = true;
    }
    if (base_position < base_end) {
      const std::uint32_t key = base[base_position].key;
      if (!found || key < minimum) minimum = key;
      found = true;
    }
    if (!found || minimum > high) break;

    Row winner{};
    bool have_winner = false;
    if (pending_position < pending_end &&
        pending[pending_position].row.key == minimum) {
      // Stable radix order is chronological among equal keys; the final equal
      // assignment is therefore the exact newest pending winner.
      do {
        winner = pending[pending_position++].row;
      } while (pending_position < pending_end &&
               pending[pending_position].row.key == minimum);
      have_winner = true;
    }
    for (std::uint32_t level = 0u; level < kLevels; ++level) {
      if (class_position[level] >= class_end[level]) continue;
      const Descriptor descriptor = descriptors[descriptor_index(q, level)];
      const Row row = arena[descriptor.offset() + class_position[level]];
      if (row.key != minimum) continue;
      if (!have_winner) {
        winner = row;
        have_winner = true;
      }
      do {
        ++class_position[level];
      } while (class_position[level] < class_end[level] &&
               arena[descriptor.offset() + class_position[level]].key ==
                   minimum);
    }
    if (base_position < base_end && base[base_position].key == minimum) {
      if (!have_winner) {
        winner = base[base_position];
        have_winner = true;
      }
      do {
        ++base_position;
      } while (base_position < base_end &&
               base[base_position].key == minimum);
    }
    if (have_winner && (winner.flags & kTombstone) == 0u)
      sum += winner.value;
  }
  return sum;
}

__global__ void count_range_fragments_kernel(
    const std::uint32_t *low, const std::uint32_t *high,
    std::uint32_t query_count, std::uint32_t *counts) {
  const std::uint32_t query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query > query_count) return;
  if (query == query_count) {
    counts[query] = 0u;
    return;
  }
  const std::uint32_t lo = low[query], hi = high[query];
  counts[query] = lo <= hi ? (hi >> 16u) - (lo >> 16u) + 1u : 0u;
}

// One warp emits all section fragments of one query. This handles a single
// full-universe range without making one thread write 65,536 tasks, while
// retaining contiguous query-major and increasing-section order.
__global__ void emit_range_fragments_kernel(
    const std::uint32_t *low, const std::uint32_t *high,
    const std::uint32_t *offsets, std::uint32_t query_count,
    RangeFragment *fragments) {
  const std::uint32_t warp =
      (blockIdx.x * blockDim.x + threadIdx.x) >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  if (warp >= query_count || low[warp] > high[warp]) return;
  const std::uint32_t first = low[warp] >> 16u;
  const std::uint32_t count = offsets[warp + 1u] - offsets[warp];
  for (std::uint32_t local = lane; local < count; local += 32u)
    fragments[offsets[warp] + local] = {warp, first + local};
}

// Parallel form of the same query-major emission for one range.  A full
// 32-bit range contains 65,536 fragments; assigning those stores to one warp
// costs more than the visibility traversal's planning budget, whereas this
// flat emission remains fully coalesced.  This is only a planner variant: the
// emitted fragment representation and traversal are identical.
__global__ void emit_single_range_fragments_kernel(
    const std::uint32_t *low, const std::uint32_t *offsets,
    RangeFragment *fragments) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count = offsets[1u];
  if (index < count)
    fragments[index] = {0u, (low[0u] >> 16u) + index};
}

__global__ void reduce_range_fragment_partials_kernel(
    const std::uint32_t *offsets, std::uint32_t query_count,
    const unsigned long long *partials, std::uint32_t *out_sums) {
  const std::uint32_t warp =
      (blockIdx.x * blockDim.x + threadIdx.x) >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  if (warp >= query_count) return;
  unsigned long long local = 0ull;
  for (std::uint32_t index = offsets[warp] + lane;
       index < offsets[warp + 1u]; index += 32u)
    local += partials[index];
  for (std::uint32_t offset = 16u; offset; offset >>= 1u)
    local += __shfl_down_sync(0xffffffffu, local, offset);
  if (lane == 0u) out_sums[warp] = static_cast<std::uint32_t>(local);
}

// A single wide query can have 65,536 partials.  The ordinary one-warp/query
// reducer is ideal for narrow batched ranges but serializes that case.  These
// two kernels retain the same ordered partial stream and perform a bounded
// parallel reduction without reading values or precomputing an answer.
__global__ void reduce_single_range_partials_stage1_kernel(
    const unsigned long long *partials,
    const std::uint32_t *device_fragment_count,
    unsigned long long *block_partials) {
  __shared__ unsigned long long values[256];
  unsigned long long local = 0ull;
  const std::uint32_t count = *device_fragment_count;
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += gridDim.x * blockDim.x)
    local += partials[index];
  values[threadIdx.x] = local;
  __syncthreads();
  for (std::uint32_t stride = 128u; stride; stride >>= 1u) {
    if (threadIdx.x < stride)
      values[threadIdx.x] += values[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0u) block_partials[blockIdx.x] = values[0u];
}

__global__ void reduce_single_range_partials_stage2_kernel(
    const unsigned long long *block_partials, std::uint32_t *out_sum) {
  __shared__ unsigned long long values[256];
  values[threadIdx.x] = block_partials[threadIdx.x];
  __syncthreads();
  for (std::uint32_t stride = 128u; stride; stride >>= 1u) {
    if (threadIdx.x < stride)
      values[threadIdx.x] += values[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0u)
    out_sum[0u] = static_cast<std::uint32_t>(values[0u]);
}

// Universal batched-range schedule: one warp owns one (query, upper-16
// section) fragment. Pending batches may be arbitrary within a section, so the
// warp first rank-sorts their exact assignments by (full key, logical order)
// and retains the newest equal key. Persistent classes are already sorted and
// exact. The resolved update winners then suppress BaseRun rows by full key.
template <class Aggregate, bool HasPending>
__global__ void warp_range_fragment_kernel(
    const RangeFragment *fragments, std::uint32_t fragment_count,
    const std::uint32_t *device_fragment_count,
    const std::uint32_t *query_low, const std::uint32_t *query_high,
    const Row *base, const std::uint32_t *base_offsets, const Row *arena,
    const Descriptor *descriptors, const RawAssignment *raw,
    const std::uint32_t *raw_offsets, const std::uint32_t *batch_bases,
    std::uint32_t pending_batches,
    typename Aggregate::State *aggregate_partials) {
  constexpr std::uint32_t kWarps = 4u;
  constexpr std::uint32_t kUpdateCapacity = 128u;
  union WarpScratch {
    Row merged[kWarps][kUpdateCapacity];
    TaggedRow tagged[HasPending ? kWarps : 1u]
                    [HasPending ? kUpdateCapacity : 1u];
  };
  __shared__ Row current_shared[kWarps][kUpdateCapacity];
  __shared__ WarpScratch scratch;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t fragment_index = blockIdx.x * kWarps + warp;
  // fragment_count is normally the exact host-visible count.  A single query
  // instead launches its safe maximum of 65,536 fragments and guards it with
  // the scan result in device memory.  That removes a host synchronization
  // from the wide-range path without changing the fragment traversal.
  if (fragment_index >= fragment_count ||
      fragment_index >= *device_fragment_count) return;
  constexpr unsigned full_mask = 0xffffffffu;
  // One lane performs each uniform metadata read.  Apart from avoiding 32
  // redundant loads, this matters for wide ranges: their consecutive warps
  // visit consecutive sections, while every lane in a warp requests the same
  // fragment and query descriptor.
  std::uint32_t query = 0u, q = 0u;
  if (lane == 0u) {
    const RangeFragment fragment = fragments[fragment_index];
    query = fragment.query;
    q = fragment.quotient;
  }
  query = __shfl_sync(full_mask, query, 0u);
  q = __shfl_sync(full_mask, q, 0u);
  std::uint32_t query_begin = 0u, query_end = 0u;
  if (lane == 0u) {
    query_begin = query_low[query];
    query_end = query_high[query];
  }
  query_begin = __shfl_sync(full_mask, query_begin, 0u);
  query_end = __shfl_sync(full_mask, query_end, 0u);
  const std::uint32_t q_low = q << 16u;
  const std::uint32_t q_high = q_low | 0xffffu;
  const std::uint32_t low = max(query_begin, q_low);
  const std::uint32_t high = min(query_end, q_high);
  const unsigned before = lane ? ((1u << lane) - 1u) : 0u;
  Row *current = current_shared[warp];
  Row *merged = scratch.merged[warp];

  std::uint32_t current_count = 0u;
  if constexpr (HasPending) {
    // Compact all in-fragment pending assignments. Their logical-position
    // field contains batch slot and original input position, so numeric order
    // exactly implements later-batch and later-position wins.
    std::uint32_t pending_count = 0u;
    for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
      const std::size_t oi = std::size_t{batch} * (kQuotients + 1u) + q;
      const std::uint32_t begin = raw_offsets[oi];
      const std::uint32_t end = raw_offsets[oi + 1u];
      for (std::uint32_t chunk = begin; chunk < end; chunk += 32u) {
        const std::uint32_t position = chunk + lane;
        RawAssignment item{};
        const bool valid = position < end &&
            ((item = raw[batch_bases[batch] + position]).row.key >= low) &&
            item.row.key <= high;
        const unsigned selected = __ballot_sync(full_mask, valid);
        const std::uint32_t destination =
            pending_count + __popc(selected & before);
        if (valid && destination < kUpdateCapacity)
          scratch.tagged[warp][destination] =
              {item.row, item.logical_position};
        pending_count += __popc(selected);
      }
    }
    const bool pending_overflow = pending_count > kUpdateCapacity;
    if (__shfl_sync(full_mask, pending_overflow, 0u)) {
      if (lane == 0u)
        aggregate_partials[fragment_index] =
            sum_visible_quotient_with_pending(
                q, low, high, raw, raw_offsets, batch_bases,
                pending_batches, arena, descriptors, base, base_offsets);
      return;
    }

    // Exact small rank sort. Typical random histories contain only a few tens
    // of assignments per section; doing the comparisons cooperatively avoids
    // a global open-epoch sort and preserves arbitrary update semantics.
    for (std::uint32_t index = lane; index < pending_count; index += 32u) {
      const TaggedRow item = scratch.tagged[warp][index];
      std::uint32_t rank = 0u;
      for (std::uint32_t other_index = 0u; other_index < pending_count;
           ++other_index) {
        const TaggedRow other = scratch.tagged[warp][other_index];
        rank += other.row.key < item.row.key ||
            (other.row.key == item.row.key &&
             (other.age < item.age ||
              (other.age == item.age && other_index < index)));
      }
      current[rank] = item.row;
    }
    __syncwarp();
    for (std::uint32_t group = 0u; group < 4u; ++group) {
      const std::uint32_t index = lane + group * 32u;
      const bool winner = index < pending_count &&
          (index + 1u == pending_count ||
           current[index].key != current[index + 1u].key);
      const unsigned winners = __ballot_sync(full_mask, winner);
      if (winner)
        merged[current_count + __popc(winners & before)] = current[index];
      current_count += __popc(winners);
    }
    __syncwarp();
    Row *temporary = current;
    current = merged;
    merged = temporary;
  }

  // Persistent lower-numbered classes are newer. Insert each older equal row
  // before the current row and retain the final equal row after merging.
  bool class_overflow = false;
  for (std::uint32_t level = 0u; level < kLevels; ++level) {
    unsigned long long descriptor_bits = 0ull;
    if (lane == 0u)
      descriptor_bits = descriptors[descriptor_index(q, level)].bits;
    descriptor_bits = __shfl_sync(full_mask, descriptor_bits, 0u);
    const Descriptor descriptor{descriptor_bits};
    const Row *rows = arena + descriptor.offset();
    std::uint32_t older_begin = 0u, older_end = 0u;
    if (lane == 0u && descriptor.count()) {
      // Interior fragments cover a complete upper-16 section.  Its descriptor
      // already supplies exact boundaries, so binary searching it would add
      // two scattered dependent-read chains for no information.  Only the two
      // boundary fragments of an arbitrary range need searches.
      older_begin = low == q_low
          ? 0u
          : lower_bound_rows(rows, descriptor.count(), low);
      older_end = high == q_high
          ? descriptor.count()
          : upper_bound_rows(rows, descriptor.count(), high);
    }
    older_begin = __shfl_sync(full_mask, older_begin, 0u);
    older_end = __shfl_sync(full_mask, older_end, 0u);
    const std::uint32_t older_count = older_end - older_begin;
    if (!older_count) continue;
    if (current_count + older_count > kUpdateCapacity) {
      class_overflow = true;
      break;
    }
    const Row *older = rows + older_begin;
    if (!current_count) {
      for (std::uint32_t index = lane; index < older_count; index += 32u)
        current[index] = older[index];
      current_count = older_count;
      __syncwarp();
      continue;
    }
    for (std::uint32_t index = lane; index < older_count; index += 32u) {
      const Row row = older[index];
      std::uint32_t lo = 0u, hi = current_count;
      while (lo < hi) {
        const std::uint32_t mid = (lo + hi) >> 1u;
        if (current[mid].key < row.key) lo = mid + 1u;
        else hi = mid;
      }
      merged[index + lo] = row;
    }
    for (std::uint32_t index = lane; index < current_count; index += 32u) {
      const Row row = current[index];
      std::uint32_t lo = 0u, hi = older_count;
      while (lo < hi) {
        const std::uint32_t mid = (lo + hi) >> 1u;
        if (older[mid].key <= row.key) lo = mid + 1u;
        else hi = mid;
      }
      merged[index + lo] = row;
    }
    __syncwarp();
    const std::uint32_t merged_count = current_count + older_count;
    std::uint32_t output_count = 0u;
    for (std::uint32_t group = 0u; group < 4u; ++group) {
      const std::uint32_t index = lane + group * 32u;
      const bool winner = index < merged_count &&
          (index + 1u == merged_count ||
           merged[index].key != merged[index + 1u].key);
      const unsigned winners = __ballot_sync(full_mask, winner);
      if (winner)
        current[output_count + __popc(winners & before)] = merged[index];
      output_count += __popc(winners);
    }
    current_count = output_count;
    __syncwarp();
  }
  class_overflow = __shfl_sync(full_mask, class_overflow, 0u);
  if (class_overflow) {
    if (lane == 0u)
      aggregate_partials[fragment_index] =
          sum_visible_quotient_with_pending(
              q, low, high, raw, raw_offsets, batch_bases, pending_batches,
              arena, descriptors, base, base_offsets);
    return;
  }

  typename Aggregate::State local = Aggregate::identity();
  for (std::uint32_t index = lane; index < current_count; index += 32u) {
    const Row row = current[index];
    if ((row.flags & kTombstone) == 0u)
      local = Aggregate::consume(local, row);
  }
  const std::uint32_t base_section_begin = base_offsets[q];
  const std::uint32_t base_section_count =
      base_offsets[q + 1u] - base_section_begin;
  std::uint32_t base_begin = 0u, base_end = 0u;
  if (lane == 0u && base_section_count) {
    const Row *rows = base + base_section_begin;
    base_begin = low == q_low
        ? 0u
        : lower_bound_rows(rows, base_section_count, low);
    base_end = high == q_high
        ? base_section_count
        : upper_bound_rows(rows, base_section_count, high);
  }
  base_begin = __shfl_sync(full_mask, base_begin, 0u);
  base_end = __shfl_sync(full_mask, base_end, 0u);
  if (!current_count) {
    for (std::uint32_t index = base_begin + lane; index < base_end;
         index += 32u)
      local = Aggregate::consume(local, base[base_section_begin + index]);
  } else {
    for (std::uint32_t index = base_begin + lane; index < base_end;
         index += 32u) {
      const Row row = base[base_section_begin + index];
      const std::uint32_t position =
          lower_bound_rows(current, current_count, row.key);
      if (position == current_count || current[position].key != row.key)
        local = Aggregate::consume(local, row);
    }
  }
  for (std::uint32_t offset = 16u; offset; offset >>= 1u)
    local += __shfl_down_sync(full_mask, local, offset);
  if (lane == 0u) aggregate_partials[fragment_index] = local;
}

// Four independent warp owners per block remove all CTA-wide synchronization
// from the common range path. A warp resolves the bounded update history for
// one quotient, then visits that quotient's BaseRun rows exactly once. Only a
// quotient whose physical update set exceeds the warp workspace uses the exact
// serial fallback.
template <class Aggregate>
__global__ void warp_visible_probe_base_kernel(
    std::uint32_t low, std::uint32_t high, const Row *base,
    const std::uint32_t *base_offsets, const Row *arena,
    const Descriptor *descriptors, const RawAssignment *sorted_pending,
    const std::uint32_t *pending_offsets, const std::uint32_t *batch_bases,
    std::uint32_t pending_batches, std::uint32_t q_begin,
    std::uint32_t q_count, typename Aggregate::State *aggregate_partials) {
  constexpr std::uint32_t kWarps = 4u;
  constexpr std::uint32_t kUpdateCapacity = 128u;
  __shared__ Row current_shared[kWarps][kUpdateCapacity];
  __shared__ Row merged_shared[kWarps][kUpdateCapacity];
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t local_q = blockIdx.x * kWarps + warp;
  if (local_q >= q_count) return;
  const std::uint32_t q = q_begin + local_q;
  Row *current = current_shared[warp];
  Row *merged = merged_shared[warp];
  constexpr unsigned full_mask = 0xffffffffu;

  std::uint32_t physical_updates = 0u;
  if (lane == 0u) {
    if (pending_batches)
      physical_updates += pending_offsets[q + 1u] - pending_offsets[q];
    for (std::uint32_t level = 0u; level < kLevels; ++level)
      physical_updates += descriptors[descriptor_index(q, level)].count();
  }
  physical_updates = __shfl_sync(full_mask, physical_updates, 0u);
  if (physical_updates > kUpdateCapacity) {
    if (lane == 0u)
      aggregate_partials[local_q] = sum_visible_sorted_update_quotient(
          q, low, high, sorted_pending, pending_offsets,
          pending_batches != 0u, arena, descriptors, base, base_offsets);
    return;
  }

  std::uint32_t current_count = 0u;
  const unsigned before = lane ? ((1u << lane) - 1u) : 0u;
  if (pending_batches) {
    const std::uint32_t begin = pending_offsets[q];
    const std::uint32_t count = pending_offsets[q + 1u] - begin;
    std::uint32_t output_base = 0u;
    for (std::uint32_t group = 0u; group < 4u; ++group) {
      const std::uint32_t index = lane + group * 32u;
      const bool winner = index < count &&
          (index + 1u == count ||
           sorted_pending[begin + index].row.key !=
               sorted_pending[begin + index + 1u].row.key);
      const unsigned winners = __ballot_sync(full_mask, winner);
      if (winner)
        current[output_base + __popc(winners & before)] =
            sorted_pending[begin + index].row;
      output_base += __popc(winners);
    }
    current_count = output_base;
    __syncwarp();
  }

  // Lower class numbers are newer. Place every older equal key immediately
  // before its newer counterpart so adjacent reduction retains newest-wins.
  for (std::uint32_t level = 0u; level < kLevels; ++level) {
    const Descriptor descriptor = descriptors[descriptor_index(q, level)];
    const std::uint32_t older_count = descriptor.count();
    if (!older_count) continue;
    const Row *older = arena + descriptor.offset();
    if (!current_count) {
      for (std::uint32_t index = lane; index < older_count; index += 32u)
        current[index] = older[index];
      current_count = older_count;
      __syncwarp();
      continue;
    }
    for (std::uint32_t index = lane; index < older_count; index += 32u) {
      const Row row = older[index];
      std::uint32_t lo = 0u, hi = current_count;
      while (lo < hi) {
        const std::uint32_t mid = (lo + hi) >> 1u;
        if (current[mid].key < row.key) lo = mid + 1u;
        else hi = mid;
      }
      merged[index + lo] = row;
    }
    for (std::uint32_t index = lane; index < current_count; index += 32u) {
      const Row row = current[index];
      std::uint32_t lo = 0u, hi = older_count;
      while (lo < hi) {
        const std::uint32_t mid = (lo + hi) >> 1u;
        if (older[mid].key <= row.key) lo = mid + 1u;
        else hi = mid;
      }
      merged[index + lo] = row;
    }
    __syncwarp();
    const std::uint32_t merged_count = current_count + older_count;
    std::uint32_t output_base = 0u;
    for (std::uint32_t group = 0u; group < 4u; ++group) {
      const std::uint32_t index = lane + group * 32u;
      const bool winner = index < merged_count &&
          (index + 1u == merged_count ||
           merged[index].key != merged[index + 1u].key);
      const unsigned winners = __ballot_sync(full_mask, winner);
      if (winner)
        current[output_base + __popc(winners & before)] = merged[index];
      output_base += __popc(winners);
    }
    current_count = output_base;
    __syncwarp();
  }

  typename Aggregate::State local = Aggregate::identity();
  for (std::uint32_t index = lane; index < current_count; index += 32u) {
    const Row row = current[index];
    if (row.key >= low && row.key <= high &&
        (row.flags & kTombstone) == 0u)
      local = Aggregate::consume(local, row);
  }
  const std::uint32_t base_begin = base_offsets[q];
  const std::uint32_t base_end = base_offsets[q + 1u];
  for (std::uint32_t index = base_begin + lane; index < base_end;
       index += 32u) {
    const Row row = base[index];
    if (row.key < low || row.key > high) continue;
    const std::uint32_t position =
        lower_bound_rows(current, current_count, row.key);
    if (position == current_count || current[position].key != row.key)
      local = Aggregate::consume(local, row);
  }
  for (std::uint32_t offset = 16u; offset; offset >>= 1u)
    local += __shfl_down_sync(full_mask, local, offset);
  if (lane == 0u) aggregate_partials[local_q] = local;
}

// Production admission helpers. Input order is retained explicitly even
// though radix sorting groups a batch by full key.
__global__ void iota_kernel(std::uint32_t *ids, std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) ids[i] = i;
}

// Compact fixed-capacity admitted batch slots into one chronological stream.
// A subsequent stable full-key radix sort preserves batch and in-batch order
// among equal keys, making the final equal record the exact newest winner.
__global__ void gather_open_epoch_for_range_kernel(
    const RawAssignment *raw, std::uint32_t batch_capacity,
    const std::uint32_t *raw_offsets, const std::uint32_t *compact_bases,
    RawAssignment *rows, std::uint32_t *keys) {
  const std::uint32_t batch = blockIdx.y;
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count =
      raw_offsets[std::size_t{batch} * (kQuotients + 1u) + kQuotients];
  if (position >= count) return;
  const RawAssignment item =
      raw[std::size_t{batch} * batch_capacity + position];
  const std::uint32_t destination = compact_bases[batch] + position;
  rows[destination] = item;
  keys[destination] = item.row.key;
}

__global__ void gather_raw_batch_kernel(
    const std::uint32_t *sorted_keys, const std::uint32_t *sorted_ids,
    const std::uint32_t *values, std::uint32_t count, std::uint32_t batch_slot,
    bool tombstone, RawAssignment *destination) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t original = sorted_ids[i];
  destination[i] = {{sorted_keys[i], tombstone ? 0u : values[original],
                     tombstone ? kTombstone : 0u},
                    (batch_slot << kBatchPositionBits) | original};
}

__global__ void histogram_quotients_kernel(
    const std::uint32_t *sorted_keys, std::uint32_t count,
    std::uint32_t *shifted_counts) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) atomicAdd(shifted_counts + (sorted_keys[i] >> 16u) + 1u, 1u);
}

__global__ void find_quotient_offsets_kernel(
    const std::uint32_t *sorted_keys, std::uint32_t count,
    std::uint32_t *offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  if (q == kQuotients) {
    offsets[q] = count;
    return;
  }
  const std::uint32_t key = q << 16u;
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    if (sorted_keys[mid] < key) lo = mid + 1u;
    else hi = mid;
  }
  offsets[q] = lo;
}

__device__ __forceinline__ std::uint64_t pending_signature_bits(
    std::uint32_t key) {
  const std::uint32_t first = key * 0x9e3779b1u;
  const std::uint32_t second = (key ^ (key >> 16u)) * 0x85ebca6bu;
  return (1ull << (first >> 26u)) | (1ull << (second >> 26u));
}

__global__ void add_epoch_counts_kernel(
    const std::uint32_t *batch_offsets, const std::uint32_t *sorted_keys,
    std::uint32_t *epoch_counts, std::uint64_t *batch_signatures,
    std::uint64_t *epoch_signatures) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  const std::uint32_t begin = batch_offsets[q];
  const std::uint32_t end = batch_offsets[q + 1u];
  epoch_counts[q] += end - begin;
  std::uint64_t signature = 0ull;
  for (std::uint32_t index = begin; index < end; ++index)
    signature |= pending_signature_bits(sorted_keys[index]);
  batch_signatures[q] = signature;
  epoch_signatures[q] |= signature;
}

__global__ void prepare_base_candidates_kernel(
    const std::uint32_t *keys, const std::uint32_t *values,
    std::uint32_t count, Row *candidates, std::uint8_t *keep) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  candidates[i] = {keys[i], values[i], 0u};
  keep[i] = i + 1u == count || keys[i] != keys[i + 1u];
}

__global__ void histogram_base_kernel(
    const Row *base, std::uint32_t count, std::uint32_t *rank_shifted,
    std::uint32_t *quotient_shifted) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t key = base[i].key;
  atomicAdd(rank_shifted + (key >> (32u - kRankBits)) + 1u, 1u);
  atomicAdd(quotient_shifted + (key >> 16u) + 1u, 1u);
}

__device__ __forceinline__ std::uint32_t lower_bound_raw(
    const RawAssignment *rows, std::uint32_t begin, std::uint32_t end,
    std::uint32_t key) {
  std::uint32_t lo = begin, hi = end;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    if (rows[mid].row.key < key) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

// One deliberately slow but exact owner handles a quotient that cannot fit
// the shared-memory fast paths. Persistent class sources are sorted; open
// epoch assignments are quotient-grouped but may be in arbitrary full-key
// order, so this path selects their exact newest winners without assuming
// benchmark-provided ordering.
__global__ void fused_raw_dense_fallback_kernel(
    const RawAssignment *assignments, const std::uint32_t *raw_offsets,
    const std::uint32_t *batch_bases, const std::uint32_t *raw_counts,
    Row *arena, const Descriptor *descriptors, Pending *pending) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients || !pending[q].valid) return;
  Pending plan = pending[q];
  const std::uint32_t raw_count = raw_counts[q];
  const bool fast_publish = plan.level == 0u && raw_count <= 64u;
  const bool fast_merge = plan.level != 0u && raw_count <= 64u &&
                          plan.count <= 128u;
  const bool fast_epoch = plan.count > 128u && plan.count <= kSmallMaximum;
  if (fast_publish || fast_merge || fast_epoch) return;

  std::uint32_t raw_begin[kBatchesPerEpoch];
  std::uint32_t raw_end[kBatchesPerEpoch];
#pragma unroll
  for (std::uint32_t batch = 0u; batch < kBatchesPerEpoch; ++batch) {
    const std::size_t oi = std::size_t{batch} * (kQuotients + 1u) + q;
    raw_begin[batch] = raw_offsets[oi];
    raw_end[batch] = raw_offsets[oi + 1u];
  }
  std::uint32_t class_position[kLevels]{};
  std::uint32_t class_end[kLevels]{};
#pragma unroll
  for (std::uint32_t level = 0u; level < kLevels; ++level)
    if (level < plan.level)
      class_end[level] = descriptors[descriptor_index(q, level)].count();

  std::uint32_t produced = 0u, previous = 0u;
  bool have_previous = false;
  while (true) {
    std::uint32_t minimum = kInvalid;
    bool found = false;
#pragma unroll
    for (std::uint32_t batch = 0u; batch < kBatchesPerEpoch; ++batch) {
      for (std::uint32_t position = raw_begin[batch];
           position < raw_end[batch]; ++position) {
        const std::uint32_t key =
            assignments[batch_bases[batch] + position].row.key;
        if ((!have_previous || key > previous) &&
            (!found || key < minimum)) {
          minimum = key;
          found = true;
        }
      }
    }
#pragma unroll
    for (std::uint32_t level = 0u; level < kLevels; ++level) {
      if (class_position[level] < class_end[level]) {
        const Descriptor descriptor = descriptors[descriptor_index(q, level)];
        const std::uint32_t key =
            arena[descriptor.offset() + class_position[level]].key;
        if (!found || key < minimum) minimum = key;
        found = true;
      }
    }
    if (!found) break;

    Row winner{};
    bool have_winner = false;
    // Later batch slots are newer. Within one sorted batch, the final equal
    // record has the greatest original input position.
    for (int batch = int(kBatchesPerEpoch) - 1; batch >= 0; --batch) {
      const std::uint32_t batch_index = static_cast<std::uint32_t>(batch);
      Row candidate{};
      std::uint32_t newest_position = 0u;
      bool matched = false;
      for (std::uint32_t position = raw_begin[batch_index];
           position < raw_end[batch_index]; ++position) {
        const RawAssignment item =
            assignments[batch_bases[batch_index] + position];
        if (item.row.key == minimum &&
            (!matched || item.logical_position > newest_position)) {
          candidate = item.row;
          newest_position = item.logical_position;
          matched = true;
        }
      }
      if (!have_winner && matched) {
        winner = candidate;
        have_winner = true;
        arena[plan.offset + produced] = winner;
      }
    }
    if (!have_winner) {
#pragma unroll
      for (std::uint32_t level = 0u; level < kLevels; ++level) {
        if (class_position[level] < class_end[level]) {
          const Descriptor descriptor = descriptors[descriptor_index(q, level)];
          const Row row = arena[descriptor.offset() + class_position[level]];
          if (!have_winner && row.key == minimum) {
            winner = row;
            arena[plan.offset + produced] = winner;
            have_winner = true;
          }
        }
      }
    }
    ++produced;
    previous = minimum;
    have_previous = true;
#pragma unroll
    for (std::uint32_t level = 0u; level < kLevels; ++level)
      if (class_position[level] < class_end[level]) {
        const Descriptor descriptor = descriptors[descriptor_index(q, level)];
        if (arena[descriptor.offset() + class_position[level]].key == minimum)
          ++class_position[level];
      }
  }
  pending[q].count = produced;
}

__global__ void lookup_with_pending_kernel(
    const std::uint32_t *queries, std::uint32_t *out_values,
    std::uint8_t *out_found, std::uint32_t count,
    const RawAssignment *raw, const std::uint32_t *raw_offsets,
    const std::uint32_t *batch_bases, std::uint32_t pending_batches,
    const std::uint64_t *batch_signatures,
    const std::uint64_t *epoch_signatures,
    const Row *arena, const Descriptor *descriptors, const Row *base,
    const std::uint32_t *rank23) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t key = queries[i], q = key >> 16u;
  const std::uint64_t signature_bits = pending_signature_bits(key);
  if (pending_batches &&
      (epoch_signatures[q] & signature_bits) == signature_bits) {
    for (int batch = int(pending_batches) - 1; batch >= 0; --batch) {
      const std::uint32_t batch_index = static_cast<std::uint32_t>(batch);
      const std::uint64_t signature =
          batch_signatures[std::size_t{batch_index} * kQuotients + q];
      if ((signature & signature_bits) != signature_bits) continue;
      const std::size_t oi =
          std::size_t{batch_index} * (kQuotients + 1u) + q;
      const std::uint32_t begin = raw_offsets[oi], end = raw_offsets[oi + 1u];
      Row row{};
      std::uint32_t newest_position = 0u;
      bool matched = false;
      for (std::uint32_t position = begin; position < end; ++position) {
        const RawAssignment candidate =
            raw[batch_bases[batch_index] + position];
        if (candidate.row.key == key &&
            (!matched || candidate.logical_position > newest_position)) {
          row = candidate.row;
          newest_position = candidate.logical_position;
          matched = true;
        }
      }
      if (matched) {
        const bool live = (row.flags & kTombstone) == 0u;
        out_values[i] = live ? row.value : 0u;
        if (out_found) out_found[i] = live;
        return;
      }
    }
  }
  for (std::uint32_t level = 0u; level < kLevels; ++level) {
    const Descriptor descriptor = descriptors[descriptor_index(q, level)];
    if (!descriptor.count()) continue;
    const Row *rows = arena + descriptor.offset();
    const std::uint32_t position = lower_bound_rows(rows, descriptor.count(), key);
    if (position < descriptor.count() && rows[position].key == key) {
      const bool live = (rows[position].flags & kTombstone) == 0u;
      out_values[i] = live ? rows[position].value : 0u;
      if (out_found) out_found[i] = live;
      return;
    }
  }
  const std::uint32_t cell = key >> (32u - kRankBits);
  const std::uint32_t begin = rank23[cell], end = rank23[cell + 1u];
  const std::uint32_t position = lower_bound_rows(base + begin, end - begin, key);
  const bool live = position < end - begin && base[begin + position].key == key;
  out_values[i] = live ? base[begin + position].value : 0u;
  if (out_found) out_found[i] = live;
}

__device__ unsigned long long sum_visible_quotient_with_pending(
    std::uint32_t q, std::uint32_t low, std::uint32_t high,
    const RawAssignment *raw, const std::uint32_t *raw_offsets,
    const std::uint32_t *batch_bases, std::uint32_t pending_batches,
    const Row *arena, const Descriptor *descriptors, const Row *base,
    const std::uint32_t *base_offsets) {
  std::uint32_t raw_begin[kBatchesPerEpoch]{}, raw_end[kBatchesPerEpoch]{};
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t oi = std::size_t{batch} * (kQuotients + 1u) + q;
    raw_begin[batch] = raw_offsets[oi];
    raw_end[batch] = raw_offsets[oi + 1u];
  }
  std::uint32_t class_position[kLevels]{}, class_end[kLevels]{};
  for (std::uint32_t level = 0u; level < kLevels; ++level) {
    const Descriptor descriptor = descriptors[descriptor_index(q, level)];
    class_end[level] = descriptor.count();
    class_position[level] = lower_bound_rows(
        arena + descriptor.offset(), descriptor.count(), low);
  }
  const std::uint32_t base_begin = base_offsets[q];
  const std::uint32_t base_end = base_offsets[q + 1u];
  std::uint32_t base_position = base_begin +
      lower_bound_rows(base + base_begin, base_end - base_begin, low);
  unsigned long long sum = 0ull;
  std::uint32_t previous = 0u;
  bool have_previous = false;
  while (true) {
    std::uint32_t minimum = kInvalid;
    bool found = false;
    for (std::uint32_t batch = 0u; batch < pending_batches; ++batch)
      for (std::uint32_t position = raw_begin[batch];
           position < raw_end[batch]; ++position) {
        const std::uint32_t key =
            raw[batch_bases[batch] + position].row.key;
        if (key >= low && key <= high &&
            (!have_previous || key > previous) &&
            (!found || key < minimum)) {
          minimum = key;
          found = true;
        }
      }
    for (std::uint32_t level = 0u; level < kLevels; ++level)
      if (class_position[level] < class_end[level]) {
        const Descriptor descriptor = descriptors[descriptor_index(q, level)];
        const std::uint32_t key =
            arena[descriptor.offset() + class_position[level]].key;
        if (key <= high && (!found || key < minimum)) minimum = key;
        found |= key <= high;
      }
    if (base_position < base_end) {
      const std::uint32_t key = base[base_position].key;
      if (key <= high && (!found || key < minimum)) minimum = key;
      found |= key <= high;
    }
    if (!found) break;

    Row winner{};
    bool have_winner = false;
    for (int batch = int(pending_batches) - 1; batch >= 0; --batch) {
      const std::uint32_t batch_index = static_cast<std::uint32_t>(batch);
      Row candidate{};
      std::uint32_t newest_position = 0u;
      bool matched = false;
      for (std::uint32_t position = raw_begin[batch_index];
           position < raw_end[batch_index]; ++position) {
        const RawAssignment item = raw[batch_bases[batch_index] + position];
        if (item.row.key == minimum &&
            (!matched || item.logical_position > newest_position)) {
          candidate = item.row;
          newest_position = item.logical_position;
          matched = true;
        }
      }
      if (!have_winner && matched) {
        winner = candidate;
        have_winner = true;
      }
    }
    if (!have_winner)
      for (std::uint32_t level = 0u; level < kLevels; ++level) {
        if (class_position[level] >= class_end[level]) continue;
        const Descriptor descriptor = descriptors[descriptor_index(q, level)];
        const Row row = arena[descriptor.offset() + class_position[level]];
        if (!have_winner && row.key == minimum) {
          winner = row;
          have_winner = true;
        }
      }
    if (!have_winner && base_position < base_end &&
        base[base_position].key == minimum) {
      winner = base[base_position];
      have_winner = true;
    }
    if (have_winner && (winner.flags & kTombstone) == 0u) sum += winner.value;

    previous = minimum;
    have_previous = true;
    for (std::uint32_t level = 0u; level < kLevels; ++level)
      if (class_position[level] < class_end[level]) {
        const Descriptor descriptor = descriptors[descriptor_index(q, level)];
        if (arena[descriptor.offset() + class_position[level]].key == minimum)
          ++class_position[level];
      }
    if (base_position < base_end && base[base_position].key == minimum)
      ++base_position;
  }
  return sum;
}

__global__ void range_sum_with_pending_kernel(
    const std::uint32_t *low, const std::uint32_t *high,
    std::uint32_t query_count, std::uint32_t *out_sums,
    const RawAssignment *raw, const std::uint32_t *raw_offsets,
    const std::uint32_t *batch_bases, std::uint32_t pending_batches,
    const Row *arena, const Descriptor *descriptors, const Row *base,
    const std::uint32_t *base_offsets) {
  using BlockReduce = cub::BlockReduce<unsigned long long, kThreads>;
  __shared__ typename BlockReduce::TempStorage storage;
  const std::uint32_t query = blockIdx.x;
  if (query >= query_count) return;
  const std::uint32_t lo = low[query], hi = high[query];
  unsigned long long local = 0ull;
  if (lo <= hi) {
    const std::uint32_t first = lo >> 16u, last = hi >> 16u;
    for (std::uint32_t q = first + threadIdx.x; q <= last; q += blockDim.x)
      local += sum_visible_quotient_with_pending(
          q, lo, hi, raw, raw_offsets, batch_bases, pending_batches, arena,
          descriptors, base, base_offsets);
  }
  const unsigned long long total = BlockReduce(storage).Sum(local);
  if (threadIdx.x == 0u) out_sums[query] = static_cast<std::uint32_t>(total);
}

__global__ void narrow_sum_kernel(const unsigned long long *input,
                                  std::uint32_t *output) {
  if (blockIdx.x == 0u && threadIdx.x == 0u)
    output[0] = static_cast<std::uint32_t>(input[0]);
}

__device__ bool first_visible_in_quotient(
    std::uint32_t q, std::uint32_t lower, const RawAssignment *raw,
    const std::uint32_t *raw_offsets, const std::uint32_t *batch_bases,
    std::uint32_t pending_batches, const Row *arena,
    const Descriptor *descriptors, const Row *base,
    const std::uint32_t *base_offsets, std::uint32_t &result) {
  std::uint32_t raw_begin[kBatchesPerEpoch]{}, raw_end[kBatchesPerEpoch]{};
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t oi = std::size_t{batch} * (kQuotients + 1u) + q;
    raw_begin[batch] = raw_offsets[oi];
    raw_end[batch] = raw_offsets[oi + 1u];
  }
  std::uint32_t class_position[kLevels]{}, class_end[kLevels]{};
  for (std::uint32_t level = 0u; level < kLevels; ++level) {
    const Descriptor descriptor = descriptors[descriptor_index(q, level)];
    class_end[level] = descriptor.count();
    class_position[level] = lower_bound_rows(
        arena + descriptor.offset(), descriptor.count(), lower);
  }
  const std::uint32_t base_begin = base_offsets[q],
                      base_end = base_offsets[q + 1u];
  std::uint32_t base_position = base_begin +
      lower_bound_rows(base + base_begin, base_end - base_begin, lower);
  std::uint32_t previous{};
  bool have_previous = false;
  while (true) {
    std::uint32_t minimum = kInvalid;
    bool found = false;
    for (std::uint32_t batch = 0u; batch < pending_batches; ++batch)
      for (std::uint32_t position = raw_begin[batch];
           position < raw_end[batch]; ++position) {
        const std::uint32_t key = raw[batch_bases[batch] + position].row.key;
        if (key >= lower && (!have_previous || key > previous) &&
            (!found || key < minimum)) {
          minimum = key;
          found = true;
        }
      }
    for (std::uint32_t level = 0u; level < kLevels; ++level)
      if (class_position[level] < class_end[level]) {
        const Descriptor descriptor = descriptors[descriptor_index(q, level)];
        const std::uint32_t key =
            arena[descriptor.offset() + class_position[level]].key;
        if (!found || key < minimum) { minimum = key; found = true; }
      }
    if (base_position < base_end) {
      const std::uint32_t key = base[base_position].key;
      if (!found || key < minimum) { minimum = key; found = true; }
    }
    if (!found) return false;

    Row winner{};
    bool have_winner = false;
    for (int batch = int(pending_batches) - 1; batch >= 0; --batch) {
      const std::uint32_t batch_index = static_cast<std::uint32_t>(batch);
      Row candidate{};
      std::uint32_t newest_position{};
      bool matched = false;
      for (std::uint32_t position = raw_begin[batch_index];
           position < raw_end[batch_index]; ++position) {
        const RawAssignment item = raw[batch_bases[batch_index] + position];
        if (item.row.key == minimum &&
            (!matched || item.logical_position > newest_position)) {
          candidate = item.row;
          newest_position = item.logical_position;
          matched = true;
        }
      }
      if (!have_winner && matched) { winner = candidate; have_winner = true; }
    }
    if (!have_winner)
      for (std::uint32_t level = 0u; level < kLevels; ++level) {
        if (class_position[level] >= class_end[level]) continue;
        const Descriptor descriptor = descriptors[descriptor_index(q, level)];
        const Row row = arena[descriptor.offset() + class_position[level]];
        if (!have_winner && row.key == minimum) {
          winner = row; have_winner = true;
        }
      }
    if (!have_winner && base_position < base_end &&
        base[base_position].key == minimum) {
      winner = base[base_position]; have_winner = true;
    }
    if (have_winner && (winner.flags & kTombstone) == 0u) {
      result = winner.key;
      return true;
    }
    for (std::uint32_t level = 0u; level < kLevels; ++level)
      if (class_position[level] < class_end[level]) {
        const Descriptor descriptor = descriptors[descriptor_index(q, level)];
        if (arena[descriptor.offset() + class_position[level]].key == minimum)
          ++class_position[level];
      }
    if (base_position < base_end && base[base_position].key == minimum)
      ++base_position;
    previous = minimum;
    have_previous = true;
  }
}

__global__ void successor_with_pending_kernel(
    const std::uint32_t *queries, std::uint32_t count,
    std::uint32_t *out_keys, const RawAssignment *raw,
    const std::uint32_t *raw_offsets, const std::uint32_t *batch_bases,
    std::uint32_t pending_batches, const Row *arena,
    const Descriptor *descriptors, const Row *base,
    const std::uint32_t *base_offsets) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t query = queries[i];
  for (std::uint32_t q = query >> 16u; q < kQuotients; ++q) {
    const std::uint32_t lower = q == (query >> 16u) ? query : q << 16u;
    std::uint32_t result{};
    if (first_visible_in_quotient(
            q, lower, raw, raw_offsets, batch_bases, pending_batches, arena,
            descriptors, base, base_offsets, result)) {
      out_keys[i] = result;
      return;
    }
  }
  out_keys[i] = kInvalid;
}


} // namespace gpulsmopt2_detail

class GPULSMOpt2 {
public:
  struct DeviceKeyBatch {
    const std::uint32_t *keys = nullptr;
    std::size_t count = 0u;
    bool sorted = false;
  };

  struct MaintenanceStats {
    std::uint64_t epochs_published = 0u;
    std::uint64_t admitted_batches = 0u;
    std::uint64_t admitted_records = 0u;
    std::uint32_t pending_batches = 0u;
    std::uint32_t pending_records = 0u;
  };

  explicit GPULSMOpt2(const DictionaryConfig &config)
      : max_elements_(config.max_elements),
        batch_capacity_(std::max<std::size_t>(1u, config.batch_capacity)),
        arena_capacity_(static_cast<std::uint32_t>(std::min<std::size_t>(
            gpulsmopt2_detail::kMaximumArenaRows,
            std::max<std::size_t>(max_elements_, batch_capacity_ * 16u)))),
        rank23_(gpulsmopt2_detail::kRankEntries),
        base_offsets_(gpulsmopt2_detail::kQuotients + 1u),
        arena_(arena_capacity_),
        descriptors_(std::size_t{gpulsmopt2_detail::kQuotients} *
                     gpulsmopt2_detail::kLevels),
        masks_(gpulsmopt2_detail::kQuotients),
        free_heads_(std::size_t{gpulsmopt2_detail::kQuotients} *
                    (gpulsmopt2_detail::kMaximumSizeClass + 1u)),
        cursor_(1u), pending_(gpulsmopt2_detail::kQuotients),
        dense_list_(gpulsmopt2_detail::kQuotients), dense_count_(1u),
        reuse_count_(1u), overflow_(1u),
        raw_assignments_(gpulsmopt2_detail::kBatchesPerEpoch *
                         batch_capacity_),
        raw_offsets_(std::size_t{gpulsmopt2_detail::kBatchesPerEpoch} *
                     (gpulsmopt2_detail::kQuotients + 1u)),
        raw_batch_bases_(gpulsmopt2_detail::kBatchesPerEpoch),
        raw_signatures_(std::size_t{gpulsmopt2_detail::kBatchesPerEpoch} *
                        gpulsmopt2_detail::kQuotients),
        raw_epoch_signatures_(gpulsmopt2_detail::kQuotients),
        raw_counts_(gpulsmopt2_detail::kQuotients),
        batch_counts_(gpulsmopt2_detail::kQuotients + 1u),
        sort_keys_(batch_capacity_), sort_ids_in_(batch_capacity_),
        sort_ids_out_(batch_capacity_),
        range_open_offsets_(gpulsmopt2_detail::kQuotients + 1u),
        range_compact_bases_(gpulsmopt2_detail::kBatchesPerEpoch),
        range_partials_(gpulsmopt2_detail::kThreads) {
    if (batch_capacity_ >
        (std::size_t{1} << gpulsmopt2_detail::kBatchPositionBits))
      throw std::invalid_argument("GPULSMOpt2 batch capacity exceeds logical-position encoding");
    if (max_elements_ > gpulsmopt2_detail::kMaximumArenaRows)
      throw std::invalid_argument("GPULSMOpt2 arena currently supports 2^26 rows");
    CUDA_CHECK(cudaEventCreateWithFlags(&operation_done_,
                                         cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(operation_done_, 0));
    std::array<std::uint32_t, gpulsmopt2_detail::kBatchesPerEpoch> bases{};
    for (std::uint32_t batch = 0u;
         batch < gpulsmopt2_detail::kBatchesPerEpoch; ++batch)
      bases[batch] = static_cast<std::uint32_t>(batch * batch_capacity_);
    CUDA_CHECK(cudaMemcpy(raw_batch_bases_.data(), bases.data(),
                          sizeof(bases), cudaMemcpyHostToDevice));
    reset_updates(0);
    CUDA_CHECK(cudaMemset(rank23_.data(), 0,
                          rank23_.size() * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(base_offsets_.data(), 0,
                          base_offsets_.size() * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaEventRecord(operation_done_, 0));
  }

  GPULSMOpt2(const GPULSMOpt2 &) = delete;
  GPULSMOpt2 &operator=(const GPULSMOpt2 &) = delete;

  ~GPULSMOpt2() {
    if (operation_done_) {
      cudaEventSynchronize(operation_done_);
      cudaEventDestroy(operation_done_);
    }
  }

  void clear(cudaStream_t stream) {
    begin_operation(stream);
    base_ = {};
    base_count_ = 0u;
    CUDA_CHECK(cudaMemsetAsync(rank23_.data(), 0,
                               rank23_.size() * sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(base_offsets_.data(), 0,
                               base_offsets_.size() * sizeof(std::uint32_t),
                               stream));
    reset_updates(stream);
    end_operation(stream);
  }

  void bulk_build(const std::uint32_t *keys, const std::uint32_t *values,
                  std::size_t count, cudaStream_t stream) {
    if ((count && (!keys || !values)) || count > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("invalid GPULSMOpt2 BaseRun input");
    begin_operation(stream);
    reset_updates(stream);
    if (!count) {
      base_ = {};
      base_count_ = 0u;
      CUDA_CHECK(cudaMemsetAsync(rank23_.data(), 0,
                                 rank23_.size() * sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(base_offsets_.data(), 0,
                                 base_offsets_.size() * sizeof(std::uint32_t),
                                 stream));
      end_operation(stream);
      return;
    }
    const std::uint32_t n = static_cast<std::uint32_t>(count);
    gpulsmopt2_detail::Buffer<std::uint32_t> sorted_keys(n), sorted_values(n);
    gpulsmopt2_detail::Buffer<gpulsmopt2_detail::Row> candidates(n);
    gpulsmopt2_detail::Buffer<std::uint8_t> keep(n);
    gpulsmopt2_detail::Buffer<std::uint32_t> selected_count(1u);
    std::size_t sort_bytes{};
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, keys, sorted_keys.data(), values,
        sorted_values.data(), n, 0, 32, stream));
    gpulsmopt2_detail::Buffer<std::uint8_t> sort_temp(sort_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sort_temp.data(), sort_bytes, keys, sorted_keys.data(), values,
        sorted_values.data(), n, 0, 32, stream));
    gpulsmopt2_detail::prepare_base_candidates_kernel<<<
        blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
        sorted_keys.data(), sorted_values.data(), n, candidates.data(),
        keep.data());
    CUDA_CHECK(cudaGetLastError());
    base_.resize(n);
    std::size_t select_bytes{};
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, select_bytes, candidates.data(), keep.data(), base_.data(),
        selected_count.data(), n, stream));
    gpulsmopt2_detail::Buffer<std::uint8_t> select_temp(select_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        select_temp.data(), select_bytes, candidates.data(), keep.data(),
        base_.data(), selected_count.data(), n, stream));
    CUDA_CHECK(cudaMemcpyAsync(&base_count_, selected_count.data(),
                               sizeof(base_count_), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    gpulsmopt2_detail::Buffer<std::uint32_t> rank_counts(
        gpulsmopt2_detail::kRankEntries);
    gpulsmopt2_detail::Buffer<std::uint32_t> quotient_counts(
        gpulsmopt2_detail::kQuotients + 1u);
    CUDA_CHECK(cudaMemsetAsync(rank_counts.data(), 0,
                               rank_counts.size() * sizeof(std::uint32_t),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(quotient_counts.data(), 0,
                               quotient_counts.size() * sizeof(std::uint32_t),
                               stream));
    gpulsmopt2_detail::histogram_base_kernel<<<
        blocks(base_count_), gpulsmopt2_detail::kThreads, 0, stream>>>(
        base_.data(), base_count_, rank_counts.data(), quotient_counts.data());
    CUDA_CHECK(cudaGetLastError());
    std::size_t rank_scan_bytes{}, quotient_scan_bytes{};
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, rank_scan_bytes, rank_counts.data(), rank23_.data(),
        gpulsmopt2_detail::kRankEntries, stream));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, quotient_scan_bytes, quotient_counts.data(),
        base_offsets_.data(), gpulsmopt2_detail::kQuotients + 1u, stream));
    gpulsmopt2_detail::Buffer<std::uint8_t> scan_temp(
        std::max(rank_scan_bytes, quotient_scan_bytes));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        scan_temp.data(), rank_scan_bytes, rank_counts.data(), rank23_.data(),
        gpulsmopt2_detail::kRankEntries, stream));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        scan_temp.data(), quotient_scan_bytes, quotient_counts.data(),
        base_offsets_.data(), gpulsmopt2_detail::kQuotients + 1u, stream));
    end_operation(stream);
  }

  void insert(const DeviceKeyValueBatch &batch, cudaStream_t stream) {
    admit(batch.keys, batch.values, batch.count, false, stream);
  }

  void erase(const DeviceKeyBatch &batch, cudaStream_t stream) {
    admit(batch.keys, nullptr, batch.count, true, stream);
  }

  void lookup(const DeviceLookupBatch &batch, cudaStream_t stream) {
    if (!batch.count) return;
    if (!batch.queries || !batch.out_values ||
        batch.count > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("invalid GPULSMOpt2 lookup");
    begin_operation(stream);
    const std::uint32_t count = static_cast<std::uint32_t>(batch.count);
    if (count >= (1u << 20u)) {
      ensure_query_capacity(count);
      gpulsmopt2_detail::iota_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          query_ids_in_.data(), count);
      std::size_t bytes{};
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, bytes, batch.queries, query_keys_.data(),
          query_ids_in_.data(), query_ids_out_.data(), count, 16, 32,
          stream));
      ensure_query_temp(bytes);
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          query_temp_.data(), bytes, batch.queries, query_keys_.data(),
          query_ids_in_.data(), query_ids_out_.data(), count, 16, 32,
          stream));
      gpulsmopt2_detail::lookup_with_pending_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          query_keys_.data(), query_values_.data(),
          batch.out_found ? query_found_.data() : nullptr, count,
          raw_assignments_.data(), raw_offsets_.data(),
          raw_batch_bases_.data(), pending_batches_, raw_signatures_.data(),
          raw_epoch_signatures_.data(), arena_.data(), descriptors_.data(),
          base_.data(), rank23_.data());
      CUDA_CHECK(cudaGetLastError());
      gpulsmopt2_detail::scatter_lookup_output_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          query_ids_out_.data(), query_values_.data(),
          batch.out_found ? query_found_.data() : nullptr, batch.out_values,
          batch.out_found, count);
    } else {
      gpulsmopt2_detail::lookup_with_pending_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          batch.queries, batch.out_values, batch.out_found, count,
          raw_assignments_.data(), raw_offsets_.data(), raw_batch_bases_.data(),
          pending_batches_, raw_signatures_.data(),
          raw_epoch_signatures_.data(), arena_.data(), descriptors_.data(),
          base_.data(), rank23_.data());
    }
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

  void range(const DeviceRangeOutputBatch &batch, cudaStream_t stream) {
    if (!batch.query_count) return;
    if (!batch.lo || !batch.hi || !batch.out_sums ||
        batch.query_count > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("invalid GPULSMOpt2 range input");
    begin_operation(stream);
    const std::uint32_t query_count =
        static_cast<std::uint32_t>(batch.query_count);
    ensure_range_fragment_query_capacity(query_count);
    gpulsmopt2_detail::count_range_fragments_kernel<<<
        blocks(std::size_t{query_count} + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            batch.lo, batch.hi, query_count,
            range_fragment_counts_.data());
    std::size_t scan_bytes{};
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, range_fragment_counts_.data(),
        range_fragment_offsets_.data(), query_count + 1u, stream));
    ensure_range_temp(scan_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        range_temp_.data(), scan_bytes, range_fragment_counts_.data(),
        range_fragment_offsets_.data(), query_count + 1u, stream));
    std::uint32_t fragment_count{};
    if (query_count == 1u) {
      // The exact count is already the final scan cell on the device.  One
      // arbitrary 32-bit range can never span more than all upper-16 sections,
      // so reserve that small fixed maximum and let the fragment kernel guard
      // against the device count.  This removes result-independent host
      // synchronization from both narrow and full single-range operations.
      fragment_count = gpulsmopt2_detail::kQuotients;
    } else {
      CUDA_CHECK(cudaMemcpyAsync(
          &fragment_count, range_fragment_offsets_.data() + query_count,
          sizeof(fragment_count), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      if (!fragment_count) {
        CUDA_CHECK(cudaMemsetAsync(batch.out_sums, 0,
                                   std::size_t{query_count} *
                                       sizeof(std::uint32_t),
                                   stream));
        end_operation(stream);
        return;
      }
    }
    ensure_range_fragment_capacity(fragment_count);
    if (query_count == 1u) {
      gpulsmopt2_detail::emit_single_range_fragments_kernel<<<
          gpulsmopt2_detail::kQuotients / gpulsmopt2_detail::kThreads,
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              batch.lo, range_fragment_offsets_.data(),
              range_fragments_.data());
    } else {
      gpulsmopt2_detail::emit_range_fragments_kernel<<<
          (query_count + 7u) / 8u, gpulsmopt2_detail::kThreads, 0, stream>>>(
              batch.lo, batch.hi, range_fragment_offsets_.data(), query_count,
              range_fragments_.data());
    }
    if (pending_batches_) {
      gpulsmopt2_detail::warp_range_fragment_kernel<
          gpulsmopt2_detail::SumRowsAggregate, true>
          <<<(fragment_count + 3u) / 4u, 128, 0, stream>>>(
              range_fragments_.data(), fragment_count,
              range_fragment_offsets_.data() + query_count,
              batch.lo, batch.hi,
              base_.data(), base_offsets_.data(), arena_.data(),
              descriptors_.data(), raw_assignments_.data(),
              raw_offsets_.data(), raw_batch_bases_.data(), pending_batches_,
              range_fragment_partials_.data());
    } else {
      gpulsmopt2_detail::warp_range_fragment_kernel<
          gpulsmopt2_detail::SumRowsAggregate, false>
          <<<(fragment_count + 3u) / 4u, 128, 0, stream>>>(
              range_fragments_.data(), fragment_count,
              range_fragment_offsets_.data() + query_count,
              batch.lo, batch.hi,
              base_.data(), base_offsets_.data(), arena_.data(),
              descriptors_.data(), raw_assignments_.data(),
              raw_offsets_.data(), raw_batch_bases_.data(), 0u,
              range_fragment_partials_.data());
    }
    if (query_count == 1u) {
      gpulsmopt2_detail::reduce_single_range_partials_stage1_kernel<<<
          256, gpulsmopt2_detail::kThreads, 0, stream>>>(
              range_fragment_partials_.data(),
              range_fragment_offsets_.data() + 1u,
              range_partials_.data());
      gpulsmopt2_detail::reduce_single_range_partials_stage2_kernel<<<
          1, gpulsmopt2_detail::kThreads, 0, stream>>>(
              range_partials_.data(), batch.out_sums);
    } else {
      gpulsmopt2_detail::reduce_range_fragment_partials_kernel<<<
          (query_count + 7u) / 8u, gpulsmopt2_detail::kThreads, 0, stream>>>(
              range_fragment_offsets_.data(), query_count,
              range_fragment_partials_.data(), batch.out_sums);
    }
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

  void successor(const DeviceSuccessorBatch &batch, cudaStream_t stream) {
    if (!batch.count) return;
    if (!batch.queries || !batch.out_keys ||
        batch.count > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("invalid GPULSMOpt2 successor input");
    begin_operation(stream);
    gpulsmopt2_detail::successor_with_pending_kernel<<<
        blocks(batch.count), gpulsmopt2_detail::kThreads, 0, stream>>>(
        batch.queries, static_cast<std::uint32_t>(batch.count), batch.out_keys,
        raw_assignments_.data(), raw_offsets_.data(), raw_batch_bases_.data(),
        pending_batches_, arena_.data(), descriptors_.data(), base_.data(),
        base_offsets_.data());
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

  MaintenanceStats maintenance_stats() const {
    MaintenanceStats result = stats_;
    result.pending_batches = pending_batches_;
    result.pending_records = pending_records_;
    return result;
  }

  std::size_t gpu_resident_bytes() const {
    return base_.size() * sizeof(gpulsmopt2_detail::Row) +
        rank23_.size() * sizeof(std::uint32_t) +
        base_offsets_.size() * sizeof(std::uint32_t) +
        arena_.size() * sizeof(gpulsmopt2_detail::Row) +
        descriptors_.size() * sizeof(gpulsmopt2_detail::Descriptor) +
        masks_.size() * sizeof(std::uint8_t) +
        free_heads_.size() * sizeof(std::uint32_t) +
        cursor_.size() * sizeof(std::uint32_t) +
        pending_.size() * sizeof(gpulsmopt2_detail::Pending) +
        dense_list_.size() * sizeof(std::uint32_t) +
        dense_count_.size() * sizeof(std::uint32_t) +
        reuse_count_.size() * sizeof(unsigned long long) +
        overflow_.size() * sizeof(std::uint32_t) +
        raw_assignments_.size() * sizeof(gpulsmopt2_detail::RawAssignment) +
        raw_offsets_.size() * sizeof(std::uint32_t) +
        raw_batch_bases_.size() * sizeof(std::uint32_t) +
        raw_signatures_.size() * sizeof(std::uint64_t) +
        raw_epoch_signatures_.size() * sizeof(std::uint64_t) +
        raw_counts_.size() * sizeof(std::uint32_t) +
        batch_counts_.size() * sizeof(std::uint32_t) +
        sort_keys_.size() * sizeof(std::uint32_t) +
        sort_ids_in_.size() * sizeof(std::uint32_t) +
        sort_ids_out_.size() * sizeof(std::uint32_t) +
        sort_temp_.size() * sizeof(std::uint8_t) +
        scan_temp_.size() * sizeof(std::uint8_t) +
        query_keys_.size() * sizeof(std::uint32_t) +
        query_ids_in_.size() * sizeof(std::uint32_t) +
        query_ids_out_.size() * sizeof(std::uint32_t) +
        query_values_.size() * sizeof(std::uint32_t) +
        query_found_.size() * sizeof(std::uint8_t) +
        query_temp_.size() * sizeof(std::uint8_t) +
        range_open_keys_in_.size() * sizeof(std::uint32_t) +
        range_open_keys_out_.size() * sizeof(std::uint32_t) +
        range_open_rows_in_.size() * sizeof(gpulsmopt2_detail::RawAssignment) +
        range_open_rows_out_.size() * sizeof(gpulsmopt2_detail::RawAssignment) +
        range_open_offsets_.size() * sizeof(std::uint32_t) +
        range_compact_bases_.size() * sizeof(std::uint32_t) +
        range_open_sort_temp_.size() * sizeof(std::uint8_t) +
        range_partials_.size() * sizeof(unsigned long long) +
        range_fragment_counts_.size() * sizeof(std::uint32_t) +
        range_fragment_offsets_.size() * sizeof(std::uint32_t) +
        range_fragments_.size() * sizeof(gpulsmopt2_detail::RangeFragment) +
        range_fragment_partials_.size() * sizeof(unsigned long long) +
        range_temp_.size() * sizeof(std::uint8_t);
  }

private:
  static int blocks(std::size_t count) {
    return static_cast<int>((count + gpulsmopt2_detail::kThreads - 1u) /
                            gpulsmopt2_detail::kThreads);
  }

  void begin_operation(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamWaitEvent(stream, operation_done_, 0));
  }

  void end_operation(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(operation_done_, stream));
  }

  void reset_updates(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(descriptors_.data(), 0,
                               descriptors_.size() *
                                   sizeof(gpulsmopt2_detail::Descriptor),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(masks_.data(), 0,
                               masks_.size() * sizeof(std::uint8_t), stream));
    CUDA_CHECK(cudaMemsetAsync(free_heads_.data(), 0xff,
                               free_heads_.size() * sizeof(std::uint32_t),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(cursor_.data(), 0, sizeof(std::uint32_t),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(raw_offsets_.data(), 0,
                               raw_offsets_.size() * sizeof(std::uint32_t),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(raw_counts_.data(), 0,
                               raw_counts_.size() * sizeof(std::uint32_t),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(
        raw_epoch_signatures_.data(), 0,
        raw_epoch_signatures_.size() * sizeof(std::uint64_t), stream));
    pending_batches_ = 0u;
    pending_records_ = 0u;
    raw_host_counts_.fill(0u);
    generation_ = 0u;
    stats_ = {};
  }

  void admit(const std::uint32_t *keys, const std::uint32_t *values,
             std::size_t count, bool tombstone, cudaStream_t stream) {
    if (!count) return;
    if (!keys || (!tombstone && !values) || count > batch_capacity_)
      throw std::invalid_argument("invalid GPULSMOpt2 update batch");
    begin_operation(stream);
    const std::uint32_t n = static_cast<std::uint32_t>(count);
    gpulsmopt2_detail::iota_kernel<<<
        blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
        sort_ids_in_.data(), n);
    std::size_t sort_bytes{};
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, keys, sort_keys_.data(), sort_ids_in_.data(),
        sort_ids_out_.data(), n, 16, 32, stream));
    ensure_sort_temp(sort_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sort_temp_.data(), sort_bytes, keys, sort_keys_.data(),
        sort_ids_in_.data(), sort_ids_out_.data(), n, 16, 32, stream));
    const std::uint32_t slot = pending_batches_;
    raw_host_counts_[slot] = n;
    pending_records_ += n;
    gpulsmopt2_detail::gather_raw_batch_kernel<<<
        blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
        sort_keys_.data(), sort_ids_out_.data(), values, n, slot, tombstone,
        raw_assignments_.data() + std::size_t{slot} * batch_capacity_);
    std::uint32_t *batch_offsets = raw_offsets_.data() +
        std::size_t{slot} * (gpulsmopt2_detail::kQuotients + 1u);
    gpulsmopt2_detail::find_quotient_offsets_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(sort_keys_.data(), n,
                                                  batch_offsets);
    gpulsmopt2_detail::add_epoch_counts_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients), gpulsmopt2_detail::kThreads, 0,
        stream>>>(batch_offsets, sort_keys_.data(), raw_counts_.data(),
                  raw_signatures_.data() +
                      std::size_t{slot} * gpulsmopt2_detail::kQuotients,
                  raw_epoch_signatures_.data());
    CUDA_CHECK(cudaGetLastError());
    ++pending_batches_;
    ++stats_.admitted_batches;
    stats_.admitted_records += count;
    if (pending_batches_ == gpulsmopt2_detail::kBatchesPerEpoch)
      publish_epoch(stream);
    end_operation(stream);
  }

  void publish_epoch(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(dense_count_.data(), 0, sizeof(std::uint32_t),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(overflow_.data(), 0, sizeof(std::uint32_t),
                               stream));
    gpulsmopt2_detail::plan_extents_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients), gpulsmopt2_detail::kThreads, 0,
        stream>>>(raw_counts_.data(), arena_.data(), arena_capacity_,
                  descriptors_.data(), masks_.data(), free_heads_.data(),
                  cursor_.data(), pending_.data(), dense_list_.data(),
                  dense_count_.data(), reuse_count_.data(), overflow_.data());
    gpulsmopt2_detail::fused_raw_publish_kernel<<<
        (gpulsmopt2_detail::kQuotients +
         gpulsmopt2_detail::kWarpsPerBlock - 1u) /
            gpulsmopt2_detail::kWarpsPerBlock,
        gpulsmopt2_detail::kThreads, 0, stream>>>(
        raw_assignments_.data(), raw_offsets_.data(),
                  raw_batch_bases_.data(), raw_counts_.data(), arena_.data(),
                  pending_.data(), overflow_.data());
    gpulsmopt2_detail::fused_raw_merge_path_carry_kernel<<<
        (gpulsmopt2_detail::kQuotients + 3u) / 4u, 128, 0, stream>>>(
        raw_assignments_.data(), raw_offsets_.data(), raw_batch_bases_.data(),
        raw_counts_.data(), arena_.data(), descriptors_.data(), pending_.data(),
        overflow_.data());
    gpulsmopt2_detail::fused_raw_epoch_carry_kernel<<<
        (gpulsmopt2_detail::kQuotients +
         gpulsmopt2_detail::kWarpsPerBlock - 1u) /
            gpulsmopt2_detail::kWarpsPerBlock,
        gpulsmopt2_detail::kThreads, 0, stream>>>(
        raw_assignments_.data(), raw_offsets_.data(), raw_batch_bases_.data(),
        raw_counts_.data(), arena_.data(), descriptors_.data(), pending_.data(),
        overflow_.data());
    gpulsmopt2_detail::fused_raw_dense_fallback_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
        raw_assignments_.data(), raw_offsets_.data(), raw_batch_bases_.data(),
        raw_counts_.data(), arena_.data(), descriptors_.data(), pending_.data());
    gpulsmopt2_detail::publish_and_release_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients), gpulsmopt2_detail::kThreads, 0,
        stream>>>(arena_.data(), descriptors_.data(), masks_.data(),
                  free_heads_.data(), pending_.data(),
                  static_cast<std::uint16_t>(++generation_));
    CUDA_CHECK(cudaGetLastError());
    std::uint32_t overflow = 0u;
    CUDA_CHECK(cudaMemcpyAsync(&overflow, overflow_.data(), sizeof(overflow),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (overflow)
      throw std::runtime_error("GPULSMOpt2 epoch level/arena overflow");
    CUDA_CHECK(cudaMemsetAsync(raw_counts_.data(), 0,
                               raw_counts_.size() * sizeof(std::uint32_t),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(
        raw_epoch_signatures_.data(), 0,
        raw_epoch_signatures_.size() * sizeof(std::uint64_t), stream));
    pending_batches_ = 0u;
    pending_records_ = 0u;
    raw_host_counts_.fill(0u);
    ++stats_.epochs_published;
  }

  void ensure_sort_temp(std::size_t bytes) {
    if (sort_temp_.size() < bytes) sort_temp_.resize(bytes);
  }
  void ensure_scan_temp(std::size_t bytes) {
    if (scan_temp_.size() < bytes) scan_temp_.resize(bytes);
  }
  void ensure_query_temp(std::size_t bytes) {
    if (query_temp_.size() < bytes) query_temp_.resize(bytes);
  }
  void ensure_range_temp(std::size_t bytes) {
    if (range_temp_.size() < bytes) range_temp_.resize(bytes);
  }
  void ensure_range_open_capacity(std::size_t count) {
    if (range_open_keys_in_.size() >= count) return;
    range_open_keys_in_.resize(count);
    range_open_keys_out_.resize(count);
    range_open_rows_in_.resize(count);
    range_open_rows_out_.resize(count);
  }
  void ensure_range_open_sort_temp(std::size_t bytes) {
    if (range_open_sort_temp_.size() < bytes)
      range_open_sort_temp_.resize(bytes);
  }
  void ensure_range_fragment_query_capacity(std::size_t count) {
    if (range_fragment_counts_.size() >= count + 1u) return;
    range_fragment_counts_.resize(count + 1u);
    range_fragment_offsets_.resize(count + 1u);
  }
  void ensure_range_fragment_capacity(std::size_t count) {
    if (range_fragments_.size() >= count) return;
    range_fragments_.resize(count);
    range_fragment_partials_.resize(count);
  }
  void ensure_query_capacity(std::size_t count) {
    if (query_keys_.size() >= count) return;
    query_keys_.resize(count);
    query_ids_in_.resize(count);
    query_ids_out_.resize(count);
    query_values_.resize(count);
    query_found_.resize(count);
  }

  std::size_t max_elements_{};
  std::size_t batch_capacity_{};
  std::uint32_t arena_capacity_{};
  std::uint32_t base_count_{};
  std::uint32_t pending_batches_{};
  std::uint32_t pending_records_{};
  std::uint32_t generation_{};
  std::array<std::uint32_t, gpulsmopt2_detail::kBatchesPerEpoch>
      raw_host_counts_{};
  MaintenanceStats stats_{};
  cudaEvent_t operation_done_{};

  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::Row> base_;
  gpulsmopt2_detail::Buffer<std::uint32_t> rank23_, base_offsets_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::Row> arena_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::Descriptor> descriptors_;
  gpulsmopt2_detail::Buffer<std::uint8_t> masks_;
  gpulsmopt2_detail::Buffer<std::uint32_t> free_heads_, cursor_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::Pending> pending_;
  gpulsmopt2_detail::Buffer<std::uint32_t> dense_list_, dense_count_;
  gpulsmopt2_detail::Buffer<unsigned long long> reuse_count_;
  gpulsmopt2_detail::Buffer<std::uint32_t> overflow_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RawAssignment> raw_assignments_;
  gpulsmopt2_detail::Buffer<std::uint32_t> raw_offsets_, raw_batch_bases_,
      raw_counts_, batch_counts_;
  gpulsmopt2_detail::Buffer<std::uint64_t> raw_signatures_;
  gpulsmopt2_detail::Buffer<std::uint64_t> raw_epoch_signatures_;
  gpulsmopt2_detail::Buffer<std::uint32_t> sort_keys_, sort_ids_in_,
      sort_ids_out_;
  gpulsmopt2_detail::Buffer<std::uint8_t> sort_temp_, scan_temp_;
  gpulsmopt2_detail::Buffer<std::uint32_t> query_keys_, query_ids_in_,
      query_ids_out_, query_values_;
  gpulsmopt2_detail::Buffer<std::uint8_t> query_found_, query_temp_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_open_keys_in_,
      range_open_keys_out_, range_open_offsets_, range_compact_bases_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RawAssignment>
      range_open_rows_in_, range_open_rows_out_;
  gpulsmopt2_detail::Buffer<std::uint8_t> range_open_sort_temp_;
  gpulsmopt2_detail::Buffer<unsigned long long> range_partials_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_fragment_counts_,
      range_fragment_offsets_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RangeFragment> range_fragments_;
  gpulsmopt2_detail::Buffer<unsigned long long> range_fragment_partials_;
  gpulsmopt2_detail::Buffer<std::uint8_t> range_temp_;
};
