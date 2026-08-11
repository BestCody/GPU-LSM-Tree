#pragma once
#include "gpu_dictionary_adapter.cuh"
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_merge.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
constexpr std::uint32_t kMaximumLevels = 16u;
constexpr std::uint32_t kBatchesPerEpoch = 16u;
constexpr std::uint32_t kBatchPositionBits = 28u;
constexpr std::uint32_t kLocalRankBits = 7u;
constexpr std::uint32_t kLocalRankEntries =
    kQuotients * (1u << kLocalRankBits);
constexpr std::uint32_t kThreads = 256u;
constexpr std::uint32_t kWarpsPerBlock = kThreads / 32u;
constexpr std::uint32_t kInvalid = 0xffffffffu;
constexpr std::uint32_t kInvalidAge = 0xffffffffu;
constexpr std::uint32_t kTombstone = 1u;
constexpr std::uint32_t kMaximumArenaRows = 1u << 26u;
constexpr std::uint32_t kSectionOwnerMinimumReuse = 4u;

struct Row {
  std::uint32_t value;
  std::uint16_t key;
  std::uint16_t flags;
};

__host__ __device__ __forceinline__ Row make_row(
    std::uint32_t key, std::uint32_t value, std::uint32_t flags) {
  return {value, static_cast<std::uint16_t>(key),
          static_cast<std::uint16_t>(flags)};
}

__host__ __device__ __forceinline__ std::uint32_t full_key(
    std::uint32_t q, std::uint32_t suffix) {
  return (q << 16u) | suffix;
}

__host__ __device__ __forceinline__ std::uint32_t key_suffix(
    std::uint32_t key) {
  return key & 0xffffu;
}

__host__ __device__ __forceinline__ std::uint32_t raw_age(
    std::uint32_t logical_position, std::uint32_t batch_stride) {
  const std::uint32_t batch = logical_position >> kBatchPositionBits;
  const std::uint32_t position =
      logical_position & ((1u << kBatchPositionBits) - 1u);
  return kMaximumLevels + batch * batch_stride + position;
}

struct RawAssignment {
  Row row;
  std::uint32_t logical_position;
};

static_assert(sizeof(Row) == 8u);
static_assert(sizeof(RawAssignment) == 12u);

struct Descriptor {
  std::uint64_t bits{};
  __host__ __device__ static Descriptor make(std::uint32_t offset,
                                             std::uint32_t count) {
    return {std::uint64_t{offset} |
            (std::uint64_t{count} << 26u)};
  }
  __host__ __device__ std::uint32_t offset() const {
    return static_cast<std::uint32_t>(bits & ((1ull << 26u) - 1ull));
  }
  __host__ __device__ std::uint32_t count() const {
    return static_cast<std::uint32_t>((bits >> 26u) & ((1ull << 17u) - 1ull));
  }
};

static_assert(sizeof(Descriptor) == 8u);

__host__ __device__ __forceinline__ std::size_t descriptor_index(
    std::uint32_t q, std::uint32_t level) {
  return std::size_t{q} * kMaximumLevels + level;
}

struct TaggedRow {
  Row row;
  std::uint32_t age;
};

static_assert(sizeof(TaggedRow) == 12u);

struct SameFullKey {
  std::uint32_t age_bits = 32u;
  __host__ __device__ bool operator()(std::uint64_t first,
                                      std::uint64_t second) const {
    return (first >> age_bits) == (second >> age_bits);
  }
};

struct RangeFragment {
  std::uint32_t query;
  std::uint32_t quotient;
};

static_assert(sizeof(RangeFragment) == 8u);

struct SectionRangeFragment {
  std::uint32_t original_index;
  std::uint16_t low_suffix;
  std::uint16_t high_suffix;
};

static_assert(sizeof(SectionRangeFragment) == 8u);

struct SectionRangeTask {
  std::uint32_t quotient;
  std::uint32_t begin;
  std::uint32_t end;
};

static_assert(sizeof(SectionRangeTask) == 12u);
constexpr std::uint32_t kSectionTaskFragments = 256u;

template <class T> class Buffer {
public:
  Buffer() = default;
  explicit Buffer(std::size_t count) { resize(count); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
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
  std::size_t size() const { return count_; }
private:
  T *pointer_{};
  std::size_t count_{};
};

__global__ void iota_kernel(std::uint32_t *ids, std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) ids[i] = i;
}

__device__ __forceinline__ std::uint32_t size_class_for(
    std::uint32_t count) {
  if (count <= 1u) return 0u;
  return 32u - static_cast<std::uint32_t>(__clz(count - 1u));
}

__device__ __forceinline__ bool tagged_less(const TaggedRow &a,
                                            const TaggedRow &b) {
  const bool ai = a.age == kInvalidAge;
  const bool bi = b.age == kInvalidAge;
  if (ai != bi) return !ai;
  if (ai) return false;
  if (a.row.key != b.row.key) return a.row.key < b.row.key;
  return a.age < b.age;
}

__device__ RawAssignment load_pending_raw_ordinal(
    const RawAssignment *assignments, const std::uint32_t *offsets,
    std::uint32_t batch_stride, std::uint32_t pending_batches,
    std::uint32_t q, std::uint32_t ordinal) {
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t oi = std::size_t{batch} * (kQuotients + 1u) + q;
    const std::uint32_t begin = offsets[oi];
    const std::uint32_t count = offsets[oi + 1u] - begin;
    if (ordinal < count)
      return assignments[batch * batch_stride + begin + ordinal];
    ordinal -= count;
  }
  return {{0u, 0u, 0u}, 0u};
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

__device__ __forceinline__ std::uint32_t lower_bound_keys(
    const std::uint16_t *keys, std::uint32_t count, std::uint32_t key) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    if (keys[mid] < key) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

__device__ __forceinline__ std::uint32_t upper_bound_keys(
    const std::uint16_t *keys, std::uint32_t count, std::uint32_t key) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    if (keys[mid] <= key) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

struct SumRowsAggregate {
  using State = unsigned long long;
  __device__ static State identity() { return 0ull; }
  __device__ static State consume(State state, const Row &row) {
    return state + row.value;
  }
  __device__ State operator()(State a, State b) const { return a + b; }
};

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
  const std::uint32_t count =
      lo <= hi ? (hi >> 16u) - (lo >> 16u) + 1u : 0u;
  counts[query] = count;
}

template <bool SectionOwned>
__device__ __forceinline__ void emit_range_fragment(
    std::uint32_t index, std::uint32_t query, std::uint32_t quotient,
    std::uint32_t low, std::uint32_t high, RangeFragment *fragments,
    std::uint32_t *section_keys, SectionRangeFragment *section_fragments) {
  if constexpr (SectionOwned) {
    const std::uint32_t q_low = quotient << 16u;
    const std::uint32_t clipped_low = max(low, q_low);
    const std::uint32_t clipped_high = min(high, q_low | 0xffffu);
    section_keys[index] = q_low | (clipped_low & 0xffffu);
    section_fragments[index] = {
        index, static_cast<std::uint16_t>(clipped_low),
        static_cast<std::uint16_t>(clipped_high)};
  } else {
    fragments[index] = {query, quotient};
  }
}

template <bool SectionOwned>
__global__ void emit_range_fragments_kernel(
    const std::uint32_t *low, const std::uint32_t *high,
    const std::uint32_t *offsets, std::uint32_t query_count,
    RangeFragment *fragments, std::uint32_t *section_keys,
    SectionRangeFragment *section_fragments) {
  const std::uint32_t warp =
      (blockIdx.x * blockDim.x + threadIdx.x) >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  if (warp >= query_count || low[warp] > high[warp]) return;
  const std::uint32_t first = low[warp] >> 16u;
  const std::uint32_t count = offsets[warp + 1u] - offsets[warp];
  for (std::uint32_t local = lane; local < count; local += 32u)
    emit_range_fragment<SectionOwned>(
        offsets[warp] + local, warp, first + local, low[warp], high[warp],
        fragments, section_keys, section_fragments);
}

template <bool SectionOwned>
__global__ void emit_range_fragments_thread_kernel(
    const std::uint32_t *low, const std::uint32_t *high,
    const std::uint32_t *offsets, std::uint32_t query_count,
    RangeFragment *fragments, std::uint32_t *section_keys,
    SectionRangeFragment *section_fragments) {
  const std::uint32_t query =
      blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= query_count || low[query] > high[query]) return;
  const std::uint32_t first = low[query] >> 16u;
  const std::uint32_t count = offsets[query + 1u] - offsets[query];
  for (std::uint32_t local = 0u; local < count; ++local)
    emit_range_fragment<SectionOwned>(
        offsets[query] + local, query, first + local, low[query], high[query],
        fragments, section_keys, section_fragments);
}
template <bool SectionOwned>
__global__ void emit_single_range_fragments_kernel(
    const std::uint32_t *low, const std::uint32_t *high,
    const std::uint32_t *offsets, RangeFragment *fragments,
    std::uint32_t *section_keys,
    SectionRangeFragment *section_fragments) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count = offsets[1u];
  if (index < count) {
    const std::uint32_t quotient = (low[0u] >> 16u) + index;
    emit_range_fragment<SectionOwned>(
        index, 0u, quotient, low[0u], high[0u], fragments, section_keys,
        section_fragments);
  }
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

__global__ void reduce_range_fragment_partials_thread_kernel(
    const std::uint32_t *offsets, std::uint32_t query_count,
    const unsigned long long *partials, std::uint32_t *out_sums) {
  const std::uint32_t query =
      blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= query_count) return;
  unsigned long long local = 0ull;
  for (std::uint32_t index = offsets[query];
       index < offsets[query + 1u]; ++index)
    local += partials[index];
  out_sums[query] = static_cast<std::uint32_t>(local);
}
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

__device__ __noinline__ unsigned long long warp_sum_visible_by_verification(
    std::uint32_t q, std::uint32_t low, std::uint32_t high,
    const RawAssignment *raw, const std::uint32_t *raw_offsets,
    std::uint32_t batch_stride, std::uint32_t pending_batches,
    const Row *arena, const Descriptor *descriptors,
    const std::uint16_t *base_keys, const std::uint32_t *base_values,
    const std::uint32_t *base_offsets,
    std::uint32_t active_levels) {
  constexpr unsigned mask = 0xffffffffu;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t low_suffix = key_suffix(low);
  const std::uint32_t high_suffix = key_suffix(high);
  unsigned long long local = 0ull;

  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t oi = std::size_t{batch} * (kQuotients + 1u) + q;
    const std::uint32_t section_begin = raw_offsets[oi];
    const std::uint32_t section_end = raw_offsets[oi + 1u];
    const RawAssignment *rows = raw + batch * batch_stride;
    for (std::uint32_t index = section_begin + lane; index < section_end;
         index += 32u) {
      const RawAssignment item = rows[index];
      if (item.row.key < low_suffix || item.row.key > high_suffix) continue;
      bool covered = false;
      for (std::uint32_t other_batch = 0u;
           other_batch < pending_batches && !covered; ++other_batch) {
        const std::size_t noi =
            std::size_t{other_batch} * (kQuotients + 1u) + q;
        const std::uint32_t nb = raw_offsets[noi], ne = raw_offsets[noi + 1u];
        const RawAssignment *other_rows = raw + other_batch * batch_stride;
        for (std::uint32_t other = nb; other < ne; ++other) {
          const RawAssignment candidate = other_rows[other];
          if (candidate.row.key == item.row.key &&
              candidate.logical_position > item.logical_position) {
            covered = true;
            break;
          }
        }
      }
      if (!covered && (item.row.flags & kTombstone) == 0u)
        local += item.row.value;
    }
  }

  for (std::uint32_t level = 0u; level < active_levels; ++level) {
    const Descriptor descriptor = descriptors[descriptor_index(q, level)];
    const Row *rows = arena + descriptor.offset();
    const std::uint32_t begin =
        lower_bound_rows(rows, descriptor.count(), low_suffix);
    const std::uint32_t end = high == kInvalid
        ? descriptor.count()
        : upper_bound_rows(rows, descriptor.count(), high_suffix);
    for (std::uint32_t index = begin + lane; index < end; index += 32u) {
      const Row item = rows[index];
      bool covered = false;
      for (std::uint32_t batch = 0u;
           batch < pending_batches && !covered; ++batch) {
        const std::size_t oi =
            std::size_t{batch} * (kQuotients + 1u) + q;
        const std::uint32_t rb = raw_offsets[oi], re = raw_offsets[oi + 1u];
        const RawAssignment *pending_rows = raw + batch * batch_stride;
        for (std::uint32_t position = rb; position < re; ++position)
          if (pending_rows[position].row.key == item.key) {
            covered = true;
            break;
          }
      }
      for (std::uint32_t newer = 0u; newer < level && !covered; ++newer) {
        const Descriptor newer_descriptor =
            descriptors[descriptor_index(q, newer)];
        const Row *newer_rows = arena + newer_descriptor.offset();
        const std::uint32_t position =
            lower_bound_rows(newer_rows, newer_descriptor.count(), item.key);
        covered = position < newer_descriptor.count() &&
            newer_rows[position].key == item.key;
      }
      if (!covered && (item.flags & kTombstone) == 0u) local += item.value;
    }
  }

  const std::uint32_t section_begin = base_offsets[q];
  const std::uint32_t section_end = base_offsets[q + 1u];
  const std::uint32_t begin = section_begin +
      lower_bound_keys(base_keys + section_begin,
                       section_end - section_begin, low_suffix);
  const std::uint32_t end = high == kInvalid
      ? section_end
      : section_begin + upper_bound_keys(base_keys + section_begin,
                                         section_end - section_begin,
                                         high_suffix);
  for (std::uint32_t index = begin + lane; index < end; index += 32u) {
    const std::uint32_t key = base_keys[index];
    const std::uint32_t suffix = key_suffix(key);
    bool covered = false;
    for (std::uint32_t batch = 0u;
         batch < pending_batches && !covered; ++batch) {
      const std::size_t oi =
          std::size_t{batch} * (kQuotients + 1u) + q;
      const std::uint32_t rb = raw_offsets[oi], re = raw_offsets[oi + 1u];
      const RawAssignment *pending_rows = raw + batch * batch_stride;
      for (std::uint32_t position = rb; position < re; ++position)
        if (pending_rows[position].row.key == suffix) {
          covered = true;
          break;
        }
    }
    for (std::uint32_t level = 0u;
         level < active_levels && !covered; ++level) {
      const Descriptor descriptor = descriptors[descriptor_index(q, level)];
      const Row *rows = arena + descriptor.offset();
      const std::uint32_t position =
          lower_bound_rows(rows, descriptor.count(), suffix);
      covered = position < descriptor.count() &&
          rows[position].key == suffix;
    }
    if (!covered) local += base_values[index];
  }
  for (std::uint32_t offset = 16u; offset; offset >>= 1u)
    local += __shfl_down_sync(mask, local, offset);
  return __shfl_sync(mask, local, 0u);
}

__global__ void find_section_fragment_offsets_kernel(
    const std::uint32_t *sorted_sections, std::uint32_t count,
    std::uint32_t *offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  if (q == kQuotients) {
    offsets[q] = count;
    return;
  }
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    const std::uint32_t section = sorted_sections[mid] >> 16u;
    if (section < q) lo = mid + 1u;
    else hi = mid;
  }
  offsets[q] = lo;
}

__global__ void summarize_section_fragment_density_kernel(
    const std::uint32_t *section_offsets,
    std::uint32_t *maximum_fragments) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  atomicMax(maximum_fragments,
            section_offsets[q + 1u] - section_offsets[q]);
}

__global__ void count_section_range_tasks_kernel(
    const std::uint32_t *section_offsets, std::uint32_t *task_counts) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  if (q == kQuotients) {
    task_counts[q] = 0u;
    return;
  }
  const std::uint32_t begin = section_offsets[q];
  const std::uint32_t end = section_offsets[q + 1u];
  const std::uint32_t count = end - begin;
  task_counts[q] =
      (count + kSectionTaskFragments - 1u) / kSectionTaskFragments;
}

__global__ void emit_section_range_tasks_kernel(
    const std::uint32_t *section_offsets,
    const std::uint32_t *task_offsets, SectionRangeTask *tasks) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  const std::uint32_t begin = section_offsets[q];
  const std::uint32_t end = section_offsets[q + 1u];
  const std::uint32_t local_tasks = task_offsets[q + 1u] - task_offsets[q];
  const std::uint32_t task_base = task_offsets[q];
  for (std::uint32_t tile = 0u; tile < local_tasks; ++tile) {
    const std::uint32_t tile_begin =
        begin + tile * kSectionTaskFragments;
    tasks[task_base + tile] =
        {q, tile_begin, min(tile_begin + kSectionTaskFragments, end)};
  }
}

template <class Aggregate, bool HasPending, bool Tiled>
__global__ void cooperative_section_owned_range_kernel(
    const SectionRangeFragment *fragments,
    const std::uint32_t *section_offsets, const SectionRangeTask *tasks,
    const std::uint32_t *task_count,
    const std::uint16_t *base_keys, const std::uint32_t *base_values,
    const std::uint32_t *base_offsets, const Row *arena,
    const Descriptor *descriptors, const RawAssignment *raw,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, std::uint32_t active_levels,
    typename Aggregate::State *aggregate_partials) {
  constexpr std::uint32_t kCapacity = 512u;
  using BlockScan = cub::BlockScan<std::uint32_t, kThreads>;
  union Workspace {
    Row merged[kCapacity];
    TaggedRow tagged[kCapacity];
    uint2 base_rows[kWarpsPerBlock][128u];
  };
  __shared__ Row current[kCapacity];
  __shared__ Workspace workspace;
  __shared__ typename BlockScan::TempStorage scan_storage;
  constexpr std::uint32_t kFragmentBaseMaskRows = 2048u;
  constexpr std::uint32_t kFragmentBaseMaskWords =
      kFragmentBaseMaskRows / 32u;
  constexpr std::uint32_t kSectionBaseMaskRows = 16384u;
  constexpr std::uint32_t kSectionBaseMaskWords =
      kSectionBaseMaskRows / 32u;
  union BaseMaskWorkspace {
    std::uint32_t section[kSectionBaseMaskWords];
    std::uint32_t fragments[kWarpsPerBlock][kFragmentBaseMaskWords];
  };
  __shared__ BaseMaskWorkspace base_mask_workspace;
  __shared__ std::uint32_t current_count_shared;
  __shared__ std::uint32_t pending_count_shared;
  __shared__ std::uint32_t overflow_shared;
  __shared__ std::uint32_t next_fragment_shared;
  __shared__ std::uint32_t dynamic_queue_shared;
  __shared__ std::uint32_t section_base_mask_valid_shared;
  __shared__ std::uint32_t quotient_shared;
  __shared__ std::uint32_t fragment_begin_shared;
  __shared__ std::uint32_t fragment_end_shared;
  __shared__ std::uint32_t task_valid_shared;
  __shared__ std::uint32_t base_section_begin_shared;
  __shared__ std::uint32_t base_section_count_shared;
  __shared__ Descriptor section_descriptors[kMaximumLevels];

  if (threadIdx.x == 0u) {
    task_valid_shared = 1u;
    if constexpr (Tiled) {
      const std::uint32_t task_index = blockIdx.x;
      if (task_index >= *task_count) {
        task_valid_shared = 0u;
      } else {
        const SectionRangeTask task = tasks[task_index];
        quotient_shared = task.quotient;
        fragment_begin_shared = task.begin;
        fragment_end_shared = task.end;
      }
    } else {
      quotient_shared = blockIdx.x;
      fragment_begin_shared = section_offsets[blockIdx.x];
      fragment_end_shared = section_offsets[blockIdx.x + 1u];
      task_valid_shared =
          fragment_begin_shared != fragment_end_shared;
    }
  }
  __syncthreads();
  if (!task_valid_shared) return;
  const std::uint32_t q = quotient_shared;
  const std::uint32_t fragment_begin = fragment_begin_shared;
  const std::uint32_t fragment_end = fragment_end_shared;
  if (threadIdx.x < active_levels)
    section_descriptors[threadIdx.x] =
        descriptors[descriptor_index(q, threadIdx.x)];
  if (threadIdx.x == 0u) {
    base_section_begin_shared = base_offsets[q];
    base_section_count_shared =
        base_offsets[q + 1u] - base_section_begin_shared;
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    std::uint32_t physical = 0u;
    pending_count_shared = 0u;
    if constexpr (HasPending) {
      for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
        const std::size_t oi =
            std::size_t{batch} * (kQuotients + 1u) + q;
        pending_count_shared += raw_offsets[oi + 1u] - raw_offsets[oi];
      }
      physical += pending_count_shared;
    }
    for (std::uint32_t level = 0u; level < active_levels; ++level)
      physical += section_descriptors[level].count();
    overflow_shared = physical > kCapacity;
    current_count_shared = 0u;
  }
  __syncthreads();

  if (!overflow_shared && pending_count_shared) {
    const std::uint32_t pending_count = pending_count_shared;
    if (pending_count <= 32u) {
      if (threadIdx.x < 32u) {
        const std::uint32_t lane = threadIdx.x;
        TaggedRow item{{0u, 0u, 0u}, kInvalidAge};
        if (lane < pending_count) {
          const RawAssignment loaded = load_pending_raw_ordinal(
              raw, raw_offsets, batch_stride, pending_batches, q, lane);
          item = {loaded.row,
                  raw_age(loaded.logical_position, batch_stride)};
        }
        constexpr unsigned full_mask = 0xffffffffu;
        for (std::uint32_t width = 2u; width <= 32u; width <<= 1u) {
          for (std::uint32_t stride = width >> 1u; stride; stride >>= 1u) {
            TaggedRow other{};
            other.row.key = __shfl_xor_sync(full_mask, item.row.key, stride);
            other.row.value =
                __shfl_xor_sync(full_mask, item.row.value, stride);
            other.row.flags =
                __shfl_xor_sync(full_mask, item.row.flags, stride);
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
        const bool winner = item.age != kInvalidAge &&
            (lane == 31u || next_age == kInvalid ||
             item.row.key != next_key);
        const unsigned winners = __ballot_sync(full_mask, winner);
        if (winner) {
          const unsigned before = lane ? ((1u << lane) - 1u) : 0u;
          current[__popc(winners & before)] = item.row;
        }
        if (lane == 0u) current_count_shared = __popc(winners);
      }
      __syncthreads();
    } else {
      const std::uint32_t sort_size = 1u << size_class_for(pending_count);
      for (std::uint32_t ordinal = threadIdx.x; ordinal < sort_size;
           ordinal += blockDim.x) {
        if (ordinal < pending_count) {
          const RawAssignment loaded = load_pending_raw_ordinal(
              raw, raw_offsets, batch_stride, pending_batches, q, ordinal);
          workspace.tagged[ordinal] =
              {loaded.row,
               raw_age(loaded.logical_position, batch_stride)};
        } else {
          workspace.tagged[ordinal] = {{0u, 0u, 0u}, kInvalidAge};
        }
      }
      __syncthreads();
      for (std::uint32_t width = 2u; width <= sort_size; width <<= 1u) {
        for (std::uint32_t stride = width >> 1u; stride; stride >>= 1u) {
          for (std::uint32_t index = threadIdx.x; index < sort_size;
               index += blockDim.x) {
            const std::uint32_t other_index = index ^ stride;
            if (other_index > index) {
              const TaggedRow x = workspace.tagged[index];
              const TaggedRow y = workspace.tagged[other_index];
              const bool ascending = (index & width) == 0u;
              const bool swap = ascending ? tagged_less(y, x)
                                          : tagged_less(x, y);
              if (swap) {
                workspace.tagged[index] = y;
                workspace.tagged[other_index] = x;
              }
            }
          }
          __syncthreads();
        }
      }
      const std::uint32_t chunk_begin = threadIdx.x * 2u;
      std::uint32_t local_winners = 0u;
      bool winner[2]{};
#pragma unroll
      for (std::uint32_t item = 0u; item < 2u; ++item) {
        const std::uint32_t index = chunk_begin + item;
        winner[item] = index < sort_size &&
            workspace.tagged[index].age != kInvalidAge &&
            (index + 1u == sort_size ||
             workspace.tagged[index + 1u].age == kInvalidAge ||
             workspace.tagged[index].row.key !=
                 workspace.tagged[index + 1u].row.key);
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
              workspace.tagged[chunk_begin + item].row;
      __syncthreads();
      if (threadIdx.x == 0u) current_count_shared = winner_count;
      __syncthreads();
    }
  }

  if (!overflow_shared) {
    for (std::uint32_t level = 0u; level < active_levels; ++level) {
      const Descriptor descriptor = section_descriptors[level];
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
      const std::uint32_t diagonal =
          min(threadIdx.x * tile, merged_count);
      std::uint32_t partition_low =
          diagonal > current_count ? diagonal - current_count : 0u;
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
        workspace.merged[output_index] = choose_source
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
             workspace.merged[index].key !=
                 workspace.merged[index + 1u].key);
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
              workspace.merged[chunk_begin + item];
      __syncthreads();
      if (threadIdx.x == 0u) current_count_shared = winner_count;
      __syncthreads();
    }
  }

  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  constexpr unsigned full_mask = 0xffffffffu;
  const std::uint32_t current_count = current_count_shared;
  const std::uint32_t base_section_begin = base_section_begin_shared;
  const std::uint32_t base_section_count = base_section_count_shared;

  if (!overflow_shared && current_count &&
      base_section_count <= kSectionBaseMaskRows) {
    const std::uint32_t words = (base_section_count + 31u) >> 5u;
    for (std::uint32_t word = threadIdx.x; word < words;
         word += blockDim.x)
      base_mask_workspace.section[word] = 0u;
    __syncthreads();
    for (std::uint32_t update = threadIdx.x; update < current_count;
         update += blockDim.x) {
      const std::uint32_t key = current[update].key;
      const std::uint32_t position = lower_bound_keys(
          base_keys + base_section_begin, base_section_count, key);
      if (position < base_section_count &&
          base_keys[base_section_begin + position] == key)
        atomicOr(base_mask_workspace.section + (position >> 5u),
                 1u << (position & 31u));
    }
    __syncthreads();
    if (threadIdx.x == 0u) section_base_mask_valid_shared = 1u;
  } else {
    if (threadIdx.x == 0u) section_base_mask_valid_shared = 0u;
  }
  __syncthreads();
  if constexpr (Tiled) {
    if (threadIdx.x == 0u) {
      const std::uint32_t count = fragment_end - fragment_begin;
      const std::uint32_t samples = min(count, 8u);
      std::uint32_t widths[8]{};
      for (std::uint32_t sample = 0u; sample < samples; ++sample) {
        const std::uint32_t local =
            (std::uint64_t{sample} * count) / samples;
        const SectionRangeFragment item = fragments[fragment_begin + local];
        widths[sample] =
            std::uint32_t{item.high_suffix} - item.low_suffix + 1u;
        for (std::uint32_t position = sample; position > 0u &&
             widths[position] < widths[position - 1u]; --position) {
          const std::uint32_t temporary = widths[position];
          widths[position] = widths[position - 1u];
          widths[position - 1u] = temporary;
        }
      }
      const std::uint32_t lower = widths[samples / 4u];
      const std::uint32_t upper = widths[(3u * samples) / 4u];
      const bool variable = 10u * upper > 11u * lower;
      dynamic_queue_shared = variable;
      next_fragment_shared = fragment_begin;
    }
    __syncthreads();
  }
  if (!overflow_shared && (!Tiled || !dynamic_queue_shared) &&
      (section_base_mask_valid_shared || current_count == 0u)) {
      constexpr std::uint32_t kGroup = 4u;
      constexpr std::uint32_t kSubgroup = 32u / kGroup;
      constexpr std::uint32_t kTileRows = 128u;
      const std::uint32_t subgroup = lane / kSubgroup;
      const std::uint32_t subgroup_lane = lane & (kSubgroup - 1u);
      const unsigned subgroup_mask =
          ((1u << kSubgroup) - 1u) << (subgroup * kSubgroup);
      std::uint32_t group_begin =
          fragment_begin + warp * kGroup;
      while (true) {
        if constexpr (Tiled) {
          if (dynamic_queue_shared) {
            if (lane == 0u)
              group_begin = atomicAdd(&next_fragment_shared, kGroup);
            group_begin = __shfl_sync(full_mask, group_begin, 0u);
          }
        }
        if (group_begin >= fragment_end) break;
        const std::uint32_t fragment_index = group_begin + subgroup;
        const bool active = fragment_index < fragment_end;
        SectionRangeFragment fragment{};
        if (active && subgroup_lane == 0u)
          fragment = fragments[fragment_index];
        std::uint32_t low_suffix = fragment.low_suffix;
        std::uint32_t high_suffix = fragment.high_suffix;
        low_suffix = __shfl_sync(full_mask, low_suffix,
                                 subgroup * kSubgroup);
        high_suffix = __shfl_sync(full_mask, high_suffix,
                                  subgroup * kSubgroup);
        std::uint32_t update_begin = 0u, update_end = 0u;
        std::uint32_t base_begin = 0u, base_end = 0u;
        if (active && subgroup_lane == 0u) {
          if (current_count) {
            update_begin = lower_bound_rows(
                current, current_count, low_suffix);
            update_end = upper_bound_rows(
                current, current_count, high_suffix);
          }
          if (base_section_count) {
            base_begin = lower_bound_keys(base_keys + base_section_begin,
                                          base_section_count, low_suffix);
            base_end = upper_bound_keys(base_keys + base_section_begin,
                                        base_section_count, high_suffix);
          }
        }
        const std::uint32_t leader = subgroup * kSubgroup;
        update_begin = __shfl_sync(full_mask, update_begin, leader);
        update_end = __shfl_sync(full_mask, update_end, leader);
        base_begin = __shfl_sync(full_mask, base_begin, leader);
        base_end = __shfl_sync(full_mask, base_end, leader);

        unsigned long long local = 0ull;
        if (active) {
          for (std::uint32_t index = update_begin + subgroup_lane;
               index < update_end; index += kSubgroup) {
            const Row row = current[index];
            if ((row.flags & kTombstone) == 0u)
              local = Aggregate::consume(local, row);
          }
        }

        std::uint32_t union_begin = base_section_count;
        std::uint32_t union_end = 0u;
        std::uint32_t direct_rows = 0u;
#pragma unroll
        for (std::uint32_t member = 0u; member < kGroup; ++member) {
          const std::uint32_t member_lane = member * kSubgroup;
          const bool member_active =
              __shfl_sync(full_mask, active, member_lane);
          const std::uint32_t member_begin =
              __shfl_sync(full_mask, base_begin, member_lane);
          const std::uint32_t member_end =
              __shfl_sync(full_mask, base_end, member_lane);
          if (member_active) {
            union_begin = min(union_begin, member_begin);
            union_end = max(union_end, member_end);
            direct_rows += member_end - member_begin;
          }
        }
        const std::uint32_t union_rows = union_end - union_begin;
        const bool share_rows = union_rows != 0u &&
            direct_rows >= union_rows + (union_rows >> 1u);

        if (share_rows) {
          for (std::uint32_t tile_begin = union_begin;
               tile_begin < union_end; tile_begin += kTileRows) {
            const std::uint32_t tile_end =
                min(tile_begin + kTileRows, union_end);
            for (std::uint32_t index = lane;
                 index < tile_end - tile_begin; index += 32u) {
              const std::uint32_t position =
                  base_section_begin + tile_begin + index;
              workspace.base_rows[warp][index] =
                  {base_keys[position], base_values[position]};
            }
            __syncwarp();
            if (active) {
              const std::uint32_t consume_begin =
                  max(base_begin, tile_begin);
              const std::uint32_t consume_end =
                  min(base_end, tile_end);
              for (std::uint32_t index =
                       consume_begin + subgroup_lane;
                   index < consume_end; index += kSubgroup) {
                const uint2 row =
                    workspace.base_rows[warp][index - tile_begin];
                bool covered = false;
                if (section_base_mask_valid_shared) {
                  covered =
                      (base_mask_workspace.section[index >> 5u] &
                       (1u << (index & 31u))) != 0u;
                } else if (current_count) {
                  const std::uint32_t update_position =
                      lower_bound_rows(current, current_count,
                                       key_suffix(row.x));
                  covered = update_position < current_count &&
                      current[update_position].key == key_suffix(row.x);
                }
                if (!covered)
                  local = Aggregate::consume(
                      local, make_row(row.x, row.y, 0u));
              }
            }
            __syncwarp();
          }
        } else if (active) {
          for (std::uint32_t index = base_begin + subgroup_lane;
               index < base_end; index += kSubgroup) {
            const std::uint32_t position =
                base_section_begin + index;
            const std::uint32_t key = base_keys[position];
            bool covered = false;
            if (section_base_mask_valid_shared) {
              covered =
                  (base_mask_workspace.section[index >> 5u] &
                   (1u << (index & 31u))) != 0u;
            } else if (current_count) {
              const std::uint32_t suffix = key_suffix(key);
              const std::uint32_t update_position =
                  lower_bound_rows(current, current_count, suffix);
              covered = update_position < current_count &&
                  current[update_position].key == suffix;
            }
            if (!covered)
              local = Aggregate::consume(
                  local, make_row(key, base_values[position], 0u));
          }
        }

        for (std::uint32_t offset = kSubgroup / 2u; offset;
             offset >>= 1u) {
          const unsigned long long other = __shfl_down_sync(
              subgroup_mask, local, offset, kSubgroup);
          local = Aggregate{}(local, other);
        }
        if (active && subgroup_lane == 0u)
          aggregate_partials[fragment.original_index] = local;
        if constexpr (Tiled) {
          if (!dynamic_queue_shared)
            group_begin += kWarpsPerBlock * kGroup;
        } else {
          group_begin += kWarpsPerBlock * kGroup;
        }
      }
      return;
  }
  std::uint32_t fragment_index = fragment_begin + warp;
  while (true) {
    if constexpr (Tiled) {
      if (dynamic_queue_shared) {
        if (lane == 0u)
          fragment_index = atomicAdd(&next_fragment_shared, 1u);
        fragment_index = __shfl_sync(full_mask, fragment_index, 0u);
      }
    }
    if (fragment_index >= fragment_end) break;
    const SectionRangeFragment fragment = fragments[fragment_index];
    const std::uint32_t q_low = q << 16u;
    const std::uint32_t low = q_low | fragment.low_suffix;
    const std::uint32_t high = q_low | fragment.high_suffix;
    const std::uint32_t low_suffix = fragment.low_suffix;
    const std::uint32_t high_suffix = fragment.high_suffix;
    unsigned long long local = 0ull;
    if (overflow_shared) {
      local = warp_sum_visible_by_verification(
          q, low, high, HasPending ? raw : nullptr,
          HasPending ? raw_offsets : nullptr,
          batch_stride,
          HasPending ? pending_batches : 0u, arena, descriptors,
          base_keys, base_values, base_offsets, active_levels);
    } else {
      std::uint32_t update_begin = 0u, update_end = 0u;
      if (lane == 0u && current_count) {
        update_begin = lower_bound_rows(
            current, current_count, low_suffix);
        update_end = upper_bound_rows(
            current, current_count, high_suffix);
      }
      update_begin = __shfl_sync(full_mask, update_begin, 0u);
      update_end = __shfl_sync(full_mask, update_end, 0u);
      const std::uint32_t update_count = update_end - update_begin;
      for (std::uint32_t index = update_begin + lane; index < update_end;
           index += 32u) {
        const Row row = current[index];
        if ((row.flags & kTombstone) == 0u)
          local = Aggregate::consume(local, row);
      }

      std::uint32_t begin = 0u, end = 0u;
      if (lane == 0u && base_section_count) {
        begin = lower_bound_keys(base_keys + base_section_begin,
                                 base_section_count, low_suffix);
        end = upper_bound_keys(base_keys + base_section_begin,
                               base_section_count, high_suffix);
      }
      begin = __shfl_sync(full_mask, begin, 0u);
      end = __shfl_sync(full_mask, end, 0u);
      if (!update_count) {
        for (std::uint32_t index = begin + lane; index < end; index += 32u)
          local += base_values[base_section_begin + index];
      } else {
        const Row *updates = current + update_begin;
        const std::uint32_t base_count = end - begin;
        if (section_base_mask_valid_shared) {
          for (std::uint32_t index = begin + lane; index < end;
               index += 32u)
            if ((base_mask_workspace.section[index >> 5u] &
                 (1u << (index & 31u))) == 0u)
              local += base_values[base_section_begin + index];
        } else if (base_count <= kFragmentBaseMaskRows) {
          const std::uint32_t mask_words = (base_count + 31u) >> 5u;
          for (std::uint32_t word = lane; word < mask_words; word += 32u)
            base_mask_workspace.fragments[warp][word] = 0u;
          __syncwarp();

          const std::uint32_t base_absolute = base_section_begin + begin;
          for (std::uint32_t update = lane; update < update_count;
               update += 32u) {
            const std::uint32_t key = updates[update].key;
            const std::uint32_t position =
                lower_bound_keys(base_keys + base_absolute, base_count, key);
            if (position < base_count &&
                base_keys[base_absolute + position] == key)
              atomicOr(base_mask_workspace.fragments[warp] +
                           (position >> 5u),
                       1u << (position & 31u));
          }
          __syncwarp();
          for (std::uint32_t index = lane; index < base_count; index += 32u)
            if ((base_mask_workspace.fragments[warp][index >> 5u] &
                 (1u << (index & 31u))) == 0u)
              local += base_values[base_absolute + index];
        } else {
          for (std::uint32_t index = begin + lane; index < end;
               index += 32u) {
            const std::uint32_t position = base_section_begin + index;
            const std::uint32_t key = base_keys[position];
            const std::uint32_t suffix = key_suffix(key);
            const std::uint32_t update_position =
                lower_bound_rows(updates, update_count, suffix);
            if (update_position == update_count ||
                updates[update_position].key != suffix)
              local += base_values[position];
          }
        }
      }
      for (std::uint32_t offset = 16u; offset; offset >>= 1u)
        local += __shfl_down_sync(full_mask, local, offset);
    }
    if (lane == 0u)
      aggregate_partials[fragment.original_index] = local;
    if constexpr (Tiled) {
      if (!dynamic_queue_shared) fragment_index += kWarpsPerBlock;
    } else {
      fragment_index += kWarpsPerBlock;
    }
  }
}

template <class Aggregate, bool HasPending>
__global__ void warp_range_fragment_kernel(
    const RangeFragment *fragments, std::uint32_t fragment_count,
    const std::uint32_t *device_fragment_count,
    const std::uint32_t *query_low, const std::uint32_t *query_high,
    const std::uint16_t *base_keys, const std::uint32_t *base_values,
    const std::uint32_t *base_offsets, const Row *arena,
    const Descriptor *descriptors, const RawAssignment *raw,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, std::uint32_t active_levels,
    typename Aggregate::State *aggregate_partials,
    std::uint32_t *overflow_fragments, std::uint32_t *overflow_count) {
  constexpr std::uint32_t kWarps = 4u;
  constexpr std::uint32_t kUpdateCapacity = 128u;
  union WarpScratch {
    Row merged[kUpdateCapacity];
    TaggedRow tagged[HasPending ? kUpdateCapacity : 1u];
  };
  __shared__ Row current_shared[kWarps][kUpdateCapacity];
  __shared__ WarpScratch scratch[kWarps];
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t fragment_index = blockIdx.x * kWarps + warp;
  if (fragment_index >= fragment_count ||
      fragment_index >= *device_fragment_count) return;
  constexpr unsigned full_mask = 0xffffffffu;
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
  const std::uint32_t low_suffix = key_suffix(low);
  const std::uint32_t high_suffix = key_suffix(high);
  const unsigned before = lane ? ((1u << lane) - 1u) : 0u;
  Row *current = current_shared[warp];
  Row *merged = scratch[warp].merged;

  std::uint32_t current_count = 0u;
  if constexpr (HasPending) {
    std::uint32_t pending_count = 0u;
    for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
      const std::size_t oi = std::size_t{batch} * (kQuotients + 1u) + q;
      const std::uint32_t begin = raw_offsets[oi];
      const std::uint32_t end = raw_offsets[oi + 1u];
      for (std::uint32_t chunk = begin; chunk < end; chunk += 32u) {
        const std::uint32_t position = chunk + lane;
        RawAssignment item{};
        const bool valid = position < end &&
            ((item = raw[batch * batch_stride + position]).row.key >=
             low_suffix) && item.row.key <= high_suffix;
        const unsigned selected = __ballot_sync(full_mask, valid);
        const std::uint32_t destination =
            pending_count + __popc(selected & before);
        if (valid && destination < kUpdateCapacity)
          scratch[warp].tagged[destination] =
              {item.row, raw_age(item.logical_position, batch_stride)};
        pending_count += __popc(selected);
      }
    }
    const bool pending_overflow = pending_count > kUpdateCapacity;
    if (__shfl_sync(full_mask, pending_overflow, 0u)) {
      if (lane == 0u) {
        aggregate_partials[fragment_index] = 0ull;
        overflow_fragments[atomicAdd(overflow_count, 1u)] = fragment_index;
      }
      return;
    }

    for (std::uint32_t index = lane; index < pending_count; index += 32u) {
      const TaggedRow item = scratch[warp].tagged[index];
      std::uint32_t rank = 0u;
      for (std::uint32_t other_index = 0u; other_index < pending_count;
           ++other_index) {
        const TaggedRow other = scratch[warp].tagged[other_index];
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

  bool class_overflow = false;
  for (std::uint32_t level = 0u; level < active_levels; ++level) {
    unsigned long long descriptor_bits = 0ull;
    if (lane == 0u)
      descriptor_bits = descriptors[descriptor_index(q, level)].bits;
    descriptor_bits = __shfl_sync(full_mask, descriptor_bits, 0u);
    const Descriptor descriptor{descriptor_bits};
    const Row *rows = arena + descriptor.offset();
    std::uint32_t older_begin = 0u, older_end = 0u;
    if (lane == 0u && descriptor.count()) {
      older_begin = low == q_low
          ? 0u
          : lower_bound_rows(rows, descriptor.count(), low_suffix);
      older_end = high == q_high
          ? descriptor.count()
          : upper_bound_rows(rows, descriptor.count(), high_suffix);
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
    if (lane == 0u) {
      aggregate_partials[fragment_index] = 0ull;
      overflow_fragments[atomicAdd(overflow_count, 1u)] = fragment_index;
    }
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
    base_begin = low == q_low
        ? 0u
        : lower_bound_keys(base_keys + base_section_begin,
                           base_section_count, low_suffix);
    base_end = high == q_high
        ? base_section_count
        : upper_bound_keys(base_keys + base_section_begin,
                           base_section_count, high_suffix);
  }
  base_begin = __shfl_sync(full_mask, base_begin, 0u);
  base_end = __shfl_sync(full_mask, base_end, 0u);
  if (!current_count) {
    for (std::uint32_t index = base_begin + lane; index < base_end;
         index += 32u)
      local += base_values[base_section_begin + index];
  } else {
    for (std::uint32_t index = base_begin + lane; index < base_end;
         index += 32u) {
      const std::uint32_t position_in_base = base_section_begin + index;
      const Row row = make_row(base_keys[position_in_base],
                               base_values[position_in_base], 0u);
      const std::uint32_t position =
          lower_bound_rows(current, current_count, row.key);
      if (position == current_count || current[position].key != row.key)
        local = Aggregate::consume(local, row);
    }
  }
  for (std::uint32_t offset = 16u; offset; offset >>= 1u)
    local += __shfl_down_sync(full_mask, local, offset);
  if (lane == 0u) {
    aggregate_partials[fragment_index] = local;
  }
}

template <bool HasPending>
__global__ void overflow_range_fragment_kernel(
    const std::uint32_t *overflow_fragments,
    const std::uint32_t *overflow_count,
    const RangeFragment *fragments,
    const std::uint32_t *query_low, const std::uint32_t *query_high,
    const std::uint16_t *base_keys, const std::uint32_t *base_values,
    const std::uint32_t *base_offsets, const Row *arena,
    const Descriptor *descriptors, const RawAssignment *raw,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, std::uint32_t active_levels,
    unsigned long long *aggregate_partials) {
  constexpr std::uint32_t kWarps = 4u;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp_in_block = threadIdx.x >> 5u;
  const std::uint32_t worker = blockIdx.x * kWarps + warp_in_block;
  const std::uint32_t workers = gridDim.x * kWarps;
  const std::uint32_t count = *overflow_count;
  for (std::uint32_t overflow_index = worker; overflow_index < count;
       overflow_index += workers) {
    std::uint32_t fragment_index = 0u, query = 0u, q = 0u;
    if (lane == 0u) {
      fragment_index = overflow_fragments[overflow_index];
      const RangeFragment fragment = fragments[fragment_index];
      query = fragment.query;
      q = fragment.quotient;
    }
    constexpr unsigned full_mask = 0xffffffffu;
    fragment_index = __shfl_sync(full_mask, fragment_index, 0u);
    query = __shfl_sync(full_mask, query, 0u);
    q = __shfl_sync(full_mask, q, 0u);
    const std::uint32_t q_low = q << 16u;
    const std::uint32_t q_high = q_low | 0xffffu;
    const std::uint32_t low = max(query_low[query], q_low);
    const std::uint32_t high = min(query_high[query], q_high);
    const unsigned long long value = warp_sum_visible_by_verification(
        q, low, high, raw, raw_offsets, batch_stride,
        HasPending ? pending_batches : 0u, arena, descriptors, base_keys,
        base_values, base_offsets, active_levels);
    if (lane == 0u) aggregate_partials[fragment_index] = value;
  }
}

__global__ void gather_raw_batch_kernel(
    const std::uint32_t *sorted_keys, const std::uint32_t *sorted_ids,
    const std::uint32_t *values, std::uint32_t count, std::uint32_t batch_slot,
    bool tombstone, RawAssignment *destination,
    std::uint32_t *destination_full_keys,
    std::uint64_t *batch_signatures,
    std::uint64_t *epoch_signatures) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t original = sorted_ids[i];
  destination[i] = {
      make_row(sorted_keys[i], tombstone ? 0u : values[original],
               tombstone ? kTombstone : 0u),
      (batch_slot << kBatchPositionBits) | original};
  destination_full_keys[i] = sorted_keys[i];
  const std::uint32_t key = sorted_keys[i];
  const std::uint32_t first = key * 0x9e3779b1u;
  const std::uint32_t second = (key ^ (key >> 16u)) * 0x85ebca6bu;
  const std::uint64_t bits =
      (1ull << (first >> 26u)) | (1ull << (second >> 26u));
  const unsigned active = __activemask();
  const std::uint32_t quotient = key >> 16u;
  const unsigned peers = __match_any_sync(active, quotient);
  const std::uint32_t low = __reduce_or_sync(
      peers, static_cast<std::uint32_t>(bits));
  const std::uint32_t high = __reduce_or_sync(
      peers, static_cast<std::uint32_t>(bits >> 32u));
  if ((threadIdx.x & 31u) == static_cast<std::uint32_t>(__ffs(peers) - 1)) {
    const std::uint64_t aggregate =
        std::uint64_t{low} | (std::uint64_t{high} << 32u);
    atomicOr(reinterpret_cast<unsigned long long *>(
                 batch_signatures + quotient),
             static_cast<unsigned long long>(aggregate));
    atomicOr(reinterpret_cast<unsigned long long *>(
                 epoch_signatures + quotient),
             static_cast<unsigned long long>(aggregate));
  }
}

__device__ __forceinline__ std::uint64_t pending_signature_bits(
    std::uint32_t key) {
  const std::uint32_t first = key * 0x9e3779b1u;
  const std::uint32_t second = (key ^ (key >> 16u)) * 0x85ebca6bu;
  return (1ull << (first >> 26u)) | (1ull << (second >> 26u));
}

__global__ void finalize_quotient_metadata_kernel(
    const std::uint32_t *sorted_keys, std::uint32_t count,
    std::uint32_t *offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  const std::uint32_t key = q << 16u;
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    if (sorted_keys[mid] < key) lo = mid + 1u;
    else hi = mid;
  }
  offsets[q] = lo;
  if (q + 1u == kQuotients) offsets[kQuotients] = count;
}

__global__ void pack_publication_epoch_kernel(
    const RawAssignment *assignments, const std::uint32_t *full_keys,
    std::uint32_t batch_stride, const std::uint32_t *batch_offsets,
    std::uint32_t age_bits, std::uint64_t *keys, Row *rows) {
  const std::uint32_t batch = blockIdx.y;
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count =
      batch_offsets[batch + 1u] - batch_offsets[batch];
  if (position >= count) return;
  const std::uint32_t source = batch * batch_stride + position;
  const std::uint32_t output = batch_offsets[batch] + position;
  const RawAssignment item = assignments[source];
  const std::uint32_t age =
      raw_age(item.logical_position, batch_stride) - kMaximumLevels;
  const std::uint64_t inverse_age =
      ((std::uint64_t{1} << age_bits) - 1u) - age;
  keys[output] =
      (std::uint64_t{full_keys[source]} << age_bits) | inverse_age;
  rows[output] = item.row;
}

__global__ void normalize_publication_keys_kernel(
    const std::uint64_t *composite_keys, std::uint32_t *keys,
    const std::uint32_t *count, std::uint32_t age_bits) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < *count)
    keys[i] = static_cast<std::uint32_t>(composite_keys[i] >> age_bits);
}

__device__ __forceinline__ std::uint32_t lower_bound_full_keys(
    const std::uint32_t *keys, std::uint32_t count, std::uint32_t target) {
  std::uint32_t low = 0u, high = count;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (keys[middle] < target) low = middle + 1u;
    else high = middle;
  }
  return low;
}

__global__ void publish_global_level_descriptors_kernel(
    const std::uint32_t *keys, std::uint32_t count,
    std::uint32_t arena_offset, std::uint32_t level,
    Descriptor *descriptors) {
  __shared__ std::uint32_t boundaries[kThreads + 1u];
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  boundaries[threadIdx.x] = q < kQuotients
      ? lower_bound_full_keys(keys, count, q << 16u) : count;
  if (threadIdx.x + 1u == blockDim.x) {
    const std::uint32_t next = min(q + 1u, kQuotients);
    boundaries[blockDim.x] = next < kQuotients
        ? lower_bound_full_keys(keys, count, next << 16u) : count;
  }
  __syncthreads();
  if (q >= kQuotients) return;
  for (std::uint32_t old = 0u; old < level; ++old)
    descriptors[descriptor_index(q, old)] = {};
  const std::uint32_t begin = boundaries[threadIdx.x];
  const std::uint32_t end = boundaries[threadIdx.x + 1u];
  descriptors[descriptor_index(q, level)] = begin == end
      ? Descriptor{} : Descriptor::make(arena_offset + begin, end - begin);
}

__global__ void mark_base_winners_kernel(
    const std::uint32_t *keys, std::uint32_t count, std::uint8_t *keep) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  keep[i] = i + 1u == count || keys[i] != keys[i + 1u];
}

__global__ void gather_base_winners_kernel(
    const std::uint32_t *sorted_keys, const std::uint32_t *sorted_values,
    const std::uint32_t *selected, std::uint32_t count, std::uint16_t *keys,
    std::uint32_t *values, std::uint32_t *quotient_shifted) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t source = selected[i];
  const std::uint32_t key = sorted_keys[source];
  keys[i] = static_cast<std::uint16_t>(key);
  values[i] = sorted_values[source];
  atomicAdd(quotient_shifted + (key >> 16u) + 1u, 1u);
}

__global__ void build_local_rank_directory_kernel(
    const std::uint16_t *keys, const std::uint32_t *base_offsets,
    std::uint16_t *local_rank) {
  const std::uint32_t q = blockIdx.x;
  const std::uint32_t cell = threadIdx.x;
  const std::uint32_t section_begin = base_offsets[q];
  const std::uint32_t section_end = base_offsets[q + 1u];
  const std::uint32_t target = cell << 9u;
  const std::uint32_t position = lower_bound_keys(
      keys + section_begin, section_end - section_begin, target);
  local_rank[std::size_t{q} * 128u + cell] =
      static_cast<std::uint16_t>(position);
}

__global__ void lookup_with_pending_kernel(
    const std::uint32_t *queries, std::uint32_t *out_values,
    std::uint8_t *out_found, std::uint32_t count,
    const RawAssignment *raw, const std::uint32_t *raw_offsets,
    std::uint32_t batch_stride, std::uint32_t pending_batches,
    const std::uint64_t *batch_signatures,
    const std::uint64_t *epoch_signatures,
    const Row *arena, const Descriptor *descriptors,
    const std::uint16_t *base_keys, const std::uint32_t *base_values,
    const std::uint16_t *local_rank, const std::uint32_t *base_offsets,
    std::uint32_t active_levels,
    const std::uint32_t *query_ids,
    std::uint32_t *final_values, std::uint8_t *final_found) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t key = queries[i], q = key >> 16u;
  const std::uint32_t suffix = key_suffix(key);
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
            raw[batch_index * batch_stride + position];
        if (candidate.row.key == suffix &&
            (!matched || candidate.logical_position > newest_position)) {
          row = candidate.row;
          newest_position = candidate.logical_position;
          matched = true;
        }
      }
      if (matched) {
        const bool live = (row.flags & kTombstone) == 0u;
        const std::uint32_t destination = query_ids ? query_ids[i] : i;
        std::uint32_t *values = final_values ? final_values : out_values;
        std::uint8_t *found = final_found ? final_found : out_found;
        values[destination] = live ? row.value : found ? 0u : kInvalid;
        if (found) found[destination] = live;
        return;
      }
    }
  }
  for (std::uint32_t level = 0u; level < active_levels; ++level) {
    const Descriptor descriptor = descriptors[descriptor_index(q, level)];
    if (!descriptor.count()) continue;
    const Row *rows = arena + descriptor.offset();
    const std::uint32_t position =
        lower_bound_rows(rows, descriptor.count(), suffix);
    if (position < descriptor.count() && rows[position].key == suffix) {
      const bool live = (rows[position].flags & kTombstone) == 0u;
      const std::uint32_t destination = query_ids ? query_ids[i] : i;
      std::uint32_t *values = final_values ? final_values : out_values;
      std::uint8_t *found = final_found ? final_found : out_found;
      values[destination] =
          live ? rows[position].value : found ? 0u : kInvalid;
      if (found) found[destination] = live;
      return;
    }
  }
  const std::uint32_t cell = (key >> 9u) & 127u;
  const std::uint32_t section_begin = base_offsets[q];
  const std::size_t local_index = std::size_t{q} * 128u + cell;
  const std::uint32_t begin = section_begin + local_rank[local_index];
  const std::uint32_t end = cell == 127u
      ? base_offsets[q + 1u]
      : section_begin + local_rank[local_index + 1u];
  const std::uint32_t position =
      lower_bound_keys(base_keys + begin, end - begin, suffix);
  const bool live = position < end - begin &&
      base_keys[begin + position] == suffix;
  const std::uint32_t destination = query_ids ? query_ids[i] : i;
  std::uint32_t *values = final_values ? final_values : out_values;
  std::uint8_t *found = final_found ? final_found : out_found;
  values[destination] =
      live ? base_values[begin + position] : found ? 0u : kInvalid;
  if (found) found[destination] = live;
}

__device__ bool first_visible_in_quotient(
    std::uint32_t q, std::uint32_t lower, const RawAssignment *raw,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const Row *arena,
    const Descriptor *descriptors, const std::uint16_t *base_keys,
    const std::uint32_t *base_values,
    const std::uint32_t *base_offsets, std::uint32_t active_levels,
    std::uint32_t &result) {
  const std::uint32_t lower_suffix = key_suffix(lower);
  std::uint32_t raw_begin[kBatchesPerEpoch]{}, raw_end[kBatchesPerEpoch]{};
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t oi = std::size_t{batch} * (kQuotients + 1u) + q;
    raw_begin[batch] = raw_offsets[oi];
    raw_end[batch] = raw_offsets[oi + 1u];
  }
  std::uint32_t class_position[kMaximumLevels]{}, class_end[kMaximumLevels]{};
  for (std::uint32_t level = 0u; level < active_levels; ++level) {
    const Descriptor descriptor = descriptors[descriptor_index(q, level)];
    class_end[level] = descriptor.count();
    class_position[level] = lower_bound_rows(
        arena + descriptor.offset(), descriptor.count(), lower_suffix);
  }
  const std::uint32_t base_begin = base_offsets[q],
                      base_end = base_offsets[q + 1u];
  std::uint32_t base_position = base_begin +
      lower_bound_keys(base_keys + base_begin, base_end - base_begin,
                       lower_suffix);
  std::uint32_t previous{};
  bool have_previous = false;
  while (true) {
    std::uint32_t minimum = kInvalid;
    bool found = false;
    for (std::uint32_t batch = 0u; batch < pending_batches; ++batch)
      for (std::uint32_t position = raw_begin[batch];
           position < raw_end[batch]; ++position) {
        const std::uint32_t key =
            raw[batch * batch_stride + position].row.key;
        if (key >= lower_suffix && (!have_previous || key > previous) &&
            (!found || key < minimum)) {
          minimum = key;
          found = true;
        }
      }
    for (std::uint32_t level = 0u; level < active_levels; ++level)
      if (class_position[level] < class_end[level]) {
        const Descriptor descriptor = descriptors[descriptor_index(q, level)];
        const std::uint32_t key =
            arena[descriptor.offset() + class_position[level]].key;
        if (!found || key < minimum) { minimum = key; found = true; }
    }
    if (base_position < base_end) {
      const std::uint32_t key = key_suffix(base_keys[base_position]);
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
        const RawAssignment item =
            raw[batch_index * batch_stride + position];
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
      for (std::uint32_t level = 0u; level < active_levels; ++level) {
        if (class_position[level] >= class_end[level]) continue;
        const Descriptor descriptor = descriptors[descriptor_index(q, level)];
        const Row row = arena[descriptor.offset() + class_position[level]];
        if (!have_winner && row.key == minimum) {
          winner = row; have_winner = true;
        }
      }
    if (!have_winner && base_position < base_end &&
        key_suffix(base_keys[base_position]) == minimum) {
      winner = make_row(base_keys[base_position],
                        base_values[base_position], 0u);
      have_winner = true;
    }
    if (have_winner && (winner.flags & kTombstone) == 0u) {
      result = full_key(q, winner.key);
      return true;
    }
    for (std::uint32_t level = 0u; level < active_levels; ++level)
      if (class_position[level] < class_end[level]) {
        const Descriptor descriptor = descriptors[descriptor_index(q, level)];
        if (arena[descriptor.offset() + class_position[level]].key == minimum)
          ++class_position[level];
      }
    if (base_position < base_end &&
        key_suffix(base_keys[base_position]) == minimum)
      ++base_position;
    previous = minimum;
    have_previous = true;
  }
}

__global__ void successor_with_pending_kernel(
    const std::uint32_t *queries, std::uint32_t count,
    std::uint32_t *out_keys, const RawAssignment *raw,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const Row *arena,
    const Descriptor *descriptors, const std::uint16_t *base_keys,
    const std::uint32_t *base_values,
    const std::uint32_t *base_offsets,
    std::uint32_t active_levels) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t query = queries[i];
  for (std::uint32_t q = query >> 16u; q < kQuotients; ++q) {
    const std::uint32_t lower = q == (query >> 16u) ? query : q << 16u;
    std::uint32_t result{};
    if (first_visible_in_quotient(
            q, lower, raw, raw_offsets, batch_stride, pending_batches, arena,
            descriptors, base_keys, base_values, base_offsets,
            active_levels, result)) {
      out_keys[i] = result;
      return;
    }
  }
  out_keys[i] = kInvalid;
}


}

class GPULSMOpt {
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
    std::uint32_t active_levels = 0u;
  };

  explicit GPULSMOpt(const DictionaryConfig &config)
      : batch_capacity_(std::max<std::size_t>(1u, config.batch_capacity)),
        publication_capacity_(std::min<std::size_t>(
            gpulsmopt2_detail::kMaximumArenaRows,
            std::max<std::size_t>(
                config.max_elements,
                batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch))),
        local_rank_(gpulsmopt2_detail::kLocalRankEntries),
        base_offsets_(gpulsmopt2_detail::kQuotients + 1u),
        arena_(std::min<std::size_t>(
            gpulsmopt2_detail::kMaximumArenaRows,
            std::max<std::size_t>(
                config.max_elements,
                batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch))),
        descriptors_(std::size_t{gpulsmopt2_detail::kQuotients} *
                     gpulsmopt2_detail::kMaximumLevels),
        raw_assignments_(gpulsmopt2_detail::kBatchesPerEpoch *
                         batch_capacity_),
        raw_full_keys_(gpulsmopt2_detail::kBatchesPerEpoch *
                       batch_capacity_),
        raw_offsets_(std::size_t{gpulsmopt2_detail::kBatchesPerEpoch} *
                     (gpulsmopt2_detail::kQuotients + 1u)),
        raw_signatures_(std::size_t{gpulsmopt2_detail::kBatchesPerEpoch} *
                        gpulsmopt2_detail::kQuotients),
        raw_epoch_signatures_(gpulsmopt2_detail::kQuotients),
        level_keys_(publication_capacity_),
        publication_composite_keys_a_(
            batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch),
        publication_composite_keys_b_(
            batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch),
        publication_raw_rows_a_(
            batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch),
        publication_raw_rows_b_(
            batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch),
        publication_keys_a_(publication_capacity_),
        publication_keys_b_(publication_capacity_),
        publication_rows_a_(publication_capacity_),
        publication_rows_b_(publication_capacity_),
        publication_selected_count_(1u),
        publication_batch_offsets_(gpulsmopt2_detail::kBatchesPerEpoch + 1u),
        radix_keys_(batch_capacity_), radix_ids_out_(batch_capacity_),
        range_partials_(gpulsmopt2_detail::kThreads),
        range_overflow_count_(1u) {
    if (batch_capacity_ >
        (std::size_t{1} << gpulsmopt2_detail::kBatchPositionBits) - 2u)
      throw std::invalid_argument("GPULSMOpt batch capacity exceeds logical-position encoding");
    if (batch_capacity_ >
        gpulsmopt2_detail::kMaximumArenaRows /
            gpulsmopt2_detail::kBatchesPerEpoch)
      throw std::invalid_argument("GPULSMOpt epoch exceeds publication capacity");
    if (config.max_elements > gpulsmopt2_detail::kMaximumArenaRows)
      throw std::invalid_argument("GPULSMOpt arena currently supports 2^26 rows");
    std::size_t epoch_capacity =
        batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch;
    publication_age_bits_ = 1u;
    while ((std::size_t{1} << publication_age_bits_) < epoch_capacity)
      ++publication_age_bits_;
    CUDA_CHECK(cudaEventCreateWithFlags(&operation_done_,
                                         cudaEventDisableTiming));
    std::size_t initial_sort_bytes{};
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, initial_sort_bytes,
        reinterpret_cast<const std::uint32_t *>(raw_assignments_.data()),
        radix_keys_.data(), radix_ids_out_.data(), radix_ids_out_.data(),
        static_cast<std::uint32_t>(batch_capacity_), 16, 32, 0));
    ensure_radix_workspace(initial_sort_bytes, batch_capacity_, 0);
    initialize_publication_workspace();
    CUDA_CHECK(cudaEventRecord(operation_done_, 0));
    reset_updates(0);
    CUDA_CHECK(cudaMemset(local_rank_.data(), 0,
                          local_rank_.size() * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMemset(base_offsets_.data(), 0,
                          base_offsets_.size() * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaEventRecord(operation_done_, 0));
  }

  GPULSMOpt(const GPULSMOpt &) = delete;
  GPULSMOpt &operator=(const GPULSMOpt &) = delete;

  ~GPULSMOpt() {
    if (operation_done_) {
      cudaEventSynchronize(operation_done_);
      cudaEventDestroy(operation_done_);
    }
  }

  void clear(cudaStream_t stream) {
    begin_operation(stream);
    base_keys_ = {};
    base_values_ = {};
    CUDA_CHECK(cudaMemsetAsync(local_rank_.data(), 0,
                               local_rank_.size() * sizeof(std::uint16_t),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(base_offsets_.data(), 0,
                               base_offsets_.size() * sizeof(std::uint32_t),
                               stream));
    reset_updates(stream);
    end_operation(stream);
  }

  void bulk_build(const std::uint32_t *keys, const std::uint32_t *values,
                  std::size_t count, cudaStream_t stream) {
    if ((count && (!keys || !values)) || count > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("invalid GPULSMOpt BaseRun input");
    begin_operation(stream);
    reset_updates(stream);
    if (!count) {
      base_keys_ = {};
      base_values_ = {};
      CUDA_CHECK(cudaMemsetAsync(local_rank_.data(), 0,
                                 local_rank_.size() * sizeof(std::uint16_t),
                                 stream));
      CUDA_CHECK(cudaMemsetAsync(base_offsets_.data(), 0,
                                 base_offsets_.size() * sizeof(std::uint32_t),
                                 stream));
      end_operation(stream);
      return;
    }
    const std::uint32_t n = static_cast<std::uint32_t>(count);
    gpulsmopt2_detail::Buffer<std::uint32_t> sorted_keys(n), sorted_values(n);
    gpulsmopt2_detail::Buffer<std::uint8_t> keep(n);
    gpulsmopt2_detail::Buffer<std::uint32_t> selected_ids(n);
    gpulsmopt2_detail::Buffer<std::uint32_t> selected_count(1u);
    std::size_t sort_bytes{};
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, keys, sorted_keys.data(), values,
        sorted_values.data(), n, 0, 32, stream));
    gpulsmopt2_detail::Buffer<std::uint8_t> sort_temp(sort_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sort_temp.data(), sort_bytes, keys, sorted_keys.data(), values,
        sorted_values.data(), n, 0, 32, stream));
    gpulsmopt2_detail::mark_base_winners_kernel<<<
        blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
        sorted_keys.data(), n, keep.data());
    CUDA_CHECK(cudaGetLastError());
    cub::CountingInputIterator<std::uint32_t> input_ids(0u);
    std::size_t select_bytes{};
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, select_bytes, input_ids, keep.data(), selected_ids.data(),
        selected_count.data(), n, stream));
    gpulsmopt2_detail::Buffer<std::uint8_t> select_temp(select_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        select_temp.data(), select_bytes, input_ids, keep.data(),
        selected_ids.data(), selected_count.data(), n, stream));
    std::uint32_t base_count{};
    CUDA_CHECK(cudaMemcpyAsync(&base_count, selected_count.data(),
                               sizeof(base_count), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    base_keys_.resize(base_count);
    base_values_.resize(base_count);
    gpulsmopt2_detail::Buffer<std::uint32_t> quotient_counts(
        gpulsmopt2_detail::kQuotients + 1u);
    CUDA_CHECK(cudaMemsetAsync(quotient_counts.data(), 0,
                               quotient_counts.size() * sizeof(std::uint32_t),
                               stream));
    gpulsmopt2_detail::gather_base_winners_kernel<<<
        blocks(base_count), gpulsmopt2_detail::kThreads, 0, stream>>>(
            sorted_keys.data(), sorted_values.data(), selected_ids.data(),
            base_count, base_keys_.data(), base_values_.data(),
            quotient_counts.data());
    CUDA_CHECK(cudaGetLastError());
    std::size_t quotient_scan_bytes{};
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, quotient_scan_bytes, quotient_counts.data(),
        base_offsets_.data(), gpulsmopt2_detail::kQuotients + 1u, stream));
    gpulsmopt2_detail::Buffer<std::uint8_t> scan_temp(quotient_scan_bytes);
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        scan_temp.data(), quotient_scan_bytes, quotient_counts.data(),
        base_offsets_.data(), gpulsmopt2_detail::kQuotients + 1u, stream));
    gpulsmopt2_detail::build_local_rank_directory_kernel<<<
        gpulsmopt2_detail::kQuotients, 128u, 0, stream>>>(
            base_keys_.data(), base_offsets_.data(), local_rank_.data());
    CUDA_CHECK(cudaGetLastError());
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
      throw std::invalid_argument("invalid GPULSMOpt lookup");
    begin_operation(stream);
    const std::uint32_t count = static_cast<std::uint32_t>(batch.count);
    if (count >= gpulsmopt2_detail::kQuotients * 4u) {
      ensure_radix_buffers(count);
      std::size_t bytes{};
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, bytes, batch.queries, radix_keys_.data(),
          radix_ids_out_.data(), radix_ids_out_.data(), count, 16, 32,
          stream));
      ensure_radix_workspace(bytes, count, stream);
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          radix_workspace(), bytes, batch.queries, radix_keys_.data(),
          radix_input_ids(), radix_ids_out_.data(), count, 16, 32, stream));
      gpulsmopt2_detail::lookup_with_pending_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          radix_keys_.data(), nullptr, nullptr, count,
          raw_assignments_.data(), raw_offsets_.data(),
          static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
          raw_signatures_.data(),
          raw_epoch_signatures_.data(), arena_.data(), descriptors_.data(),
          base_keys_.data(), base_values_.data(), local_rank_.data(),
          base_offsets_.data(), active_levels_, radix_ids_out_.data(),
          batch.out_values, batch.out_found);
      CUDA_CHECK(cudaGetLastError());
    } else {
      gpulsmopt2_detail::lookup_with_pending_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          batch.queries, batch.out_values, batch.out_found, count,
          raw_assignments_.data(), raw_offsets_.data(),
          static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
          raw_signatures_.data(),
          raw_epoch_signatures_.data(), arena_.data(), descriptors_.data(),
          base_keys_.data(), base_values_.data(), local_rank_.data(),
          base_offsets_.data(), active_levels_, nullptr, nullptr, nullptr);
    }
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

  void range(const DeviceRangeOutputBatch &batch, cudaStream_t stream) {
    if (!batch.query_count) return;
    if (!batch.lo || !batch.hi || !batch.out_sums ||
        batch.query_count > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("invalid GPULSMOpt range input");
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
    const bool use_section_owners = query_count > 1u &&
        (stats_.epochs_published != 0u || pending_batches_ != 0u) &&
        std::uint64_t{fragment_count} >=
            std::uint64_t{gpulsmopt2_detail::kQuotients} *
                gpulsmopt2_detail::kSectionOwnerMinimumReuse;
    ensure_range_partial_capacity(fragment_count);
    if (use_section_owners)
      ensure_range_section_capacity(fragment_count);
    else
      ensure_range_fragment_capacity(fragment_count);
    if (query_count == 1u) {
      gpulsmopt2_detail::emit_single_range_fragments_kernel<false><<<
          gpulsmopt2_detail::kQuotients / gpulsmopt2_detail::kThreads,
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              batch.lo, batch.hi, range_fragment_offsets_.data(),
              range_fragments_.data(), nullptr, nullptr);
    } else if (std::uint64_t{fragment_count} <=
               std::uint64_t{query_count} * 4u) {
      if (use_section_owners)
        gpulsmopt2_detail::emit_range_fragments_thread_kernel<true><<<
            blocks(query_count), gpulsmopt2_detail::kThreads, 0, stream>>>(
                batch.lo, batch.hi, range_fragment_offsets_.data(), query_count,
                nullptr, range_section_keys_in_.data(),
                range_section_fragments_in_.data());
      else
        gpulsmopt2_detail::emit_range_fragments_thread_kernel<false><<<
            blocks(query_count), gpulsmopt2_detail::kThreads, 0, stream>>>(
                batch.lo, batch.hi, range_fragment_offsets_.data(), query_count,
                range_fragments_.data(), nullptr, nullptr);
    } else {
      if (use_section_owners)
        gpulsmopt2_detail::emit_range_fragments_kernel<true><<<
            (query_count + 7u) / 8u, gpulsmopt2_detail::kThreads, 0, stream>>>(
                batch.lo, batch.hi, range_fragment_offsets_.data(), query_count,
                nullptr, range_section_keys_in_.data(),
                range_section_fragments_in_.data());
      else
        gpulsmopt2_detail::emit_range_fragments_kernel<false><<<
            (query_count + 7u) / 8u, gpulsmopt2_detail::kThreads, 0, stream>>>(
                batch.lo, batch.hi, range_fragment_offsets_.data(), query_count,
                range_fragments_.data(), nullptr, nullptr);
    }
    if (use_section_owners) {
      std::size_t section_sort_bytes{};
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, section_sort_bytes, range_section_keys_in_.data(),
          range_section_keys_out_.data(),
          range_section_fragments_in_.data(),
          range_section_fragments_out_.data(), fragment_count, 0,
          32, stream));
      ensure_range_temp(section_sort_bytes);
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          range_temp_.data(), section_sort_bytes,
          range_section_keys_in_.data(), range_section_keys_out_.data(),
          range_section_fragments_in_.data(),
          range_section_fragments_out_.data(), fragment_count, 0,
          32, stream));
      gpulsmopt2_detail::find_section_fragment_offsets_kernel<<<
          blocks(gpulsmopt2_detail::kQuotients + 1u),
          gpulsmopt2_detail::kThreads, 0, stream>>>(
          range_section_keys_out_.data(), fragment_count,
              range_section_offsets_.data());
      CUDA_CHECK(cudaMemsetAsync(range_section_max_fragments_.data(), 0,
                                 sizeof(std::uint32_t), stream));
      gpulsmopt2_detail::summarize_section_fragment_density_kernel<<<
          blocks(gpulsmopt2_detail::kQuotients),
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              range_section_offsets_.data(),
              range_section_max_fragments_.data());
      std::uint32_t maximum_section_fragments{};
      CUDA_CHECK(cudaMemcpyAsync(
          &maximum_section_fragments, range_section_max_fragments_.data(),
          sizeof(maximum_section_fragments), cudaMemcpyDeviceToHost,
          stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const bool tile_sections = maximum_section_fragments >
          gpulsmopt2_detail::kSectionTaskFragments;
      const std::uint32_t maximum_section_tasks =
          gpulsmopt2_detail::kQuotients +
          (fragment_count + gpulsmopt2_detail::kSectionTaskFragments - 1u) /
              gpulsmopt2_detail::kSectionTaskFragments;
      if (tile_sections) {
        gpulsmopt2_detail::count_section_range_tasks_kernel<<<
            blocks(gpulsmopt2_detail::kQuotients + 1u),
            gpulsmopt2_detail::kThreads, 0, stream>>>(
                range_section_offsets_.data(),
                range_section_task_counts_.data());
        std::size_t task_scan_bytes{};
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr, task_scan_bytes, range_section_task_counts_.data(),
            range_section_task_offsets_.data(),
            gpulsmopt2_detail::kQuotients + 1u, stream));
        ensure_range_temp(task_scan_bytes);
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            range_temp_.data(), task_scan_bytes,
            range_section_task_counts_.data(),
            range_section_task_offsets_.data(),
            gpulsmopt2_detail::kQuotients + 1u, stream));
        gpulsmopt2_detail::emit_section_range_tasks_kernel<<<
            blocks(gpulsmopt2_detail::kQuotients),
            gpulsmopt2_detail::kThreads, 0, stream>>>(
                range_section_offsets_.data(),
                range_section_task_offsets_.data(),
                range_section_tasks_.data());
      }
      if (pending_batches_)
        tile_sections
            ? launch_section_ranges<true, true>(maximum_section_tasks, stream)
            : launch_section_ranges<true, false>(maximum_section_tasks,
                                                 stream);
      else
        tile_sections
            ? launch_section_ranges<false, true>(maximum_section_tasks,
                                                 stream)
            : launch_section_ranges<false, false>(maximum_section_tasks,
                                                  stream);
    } else if (pending_batches_) {
      launch_fragment_ranges<true>(fragment_count, query_count, batch, stream);
    } else {
      launch_fragment_ranges<false>(fragment_count, query_count, batch,
                                    stream);
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
    } else if (std::uint64_t{fragment_count} <=
               std::uint64_t{query_count} * 4u) {
      gpulsmopt2_detail::reduce_range_fragment_partials_thread_kernel<<<
          blocks(query_count), gpulsmopt2_detail::kThreads, 0, stream>>>(
              range_fragment_offsets_.data(), query_count,
              range_fragment_partials_.data(), batch.out_sums);
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
      throw std::invalid_argument("invalid GPULSMOpt successor input");
    begin_operation(stream);
    gpulsmopt2_detail::successor_with_pending_kernel<<<
        blocks(batch.count), gpulsmopt2_detail::kThreads, 0, stream>>>(
        batch.queries, static_cast<std::uint32_t>(batch.count), batch.out_keys,
        raw_assignments_.data(), raw_offsets_.data(),
        static_cast<std::uint32_t>(batch_capacity_),
        pending_batches_, arena_.data(), descriptors_.data(),
        base_keys_.data(), base_values_.data(), base_offsets_.data(),
        active_levels_);
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

  MaintenanceStats maintenance_stats() const {
    MaintenanceStats result = stats_;
    result.pending_batches = pending_batches_;
    result.pending_records = pending_records_;
    result.active_levels = active_levels_;
    return result;
  }

  std::size_t gpu_resident_bytes() const {
    return base_keys_.size() * sizeof(std::uint16_t) +
        base_values_.size() * sizeof(std::uint32_t) +
        local_rank_.size() * sizeof(std::uint16_t) +
        base_offsets_.size() * sizeof(std::uint32_t) +
        arena_.size() * sizeof(gpulsmopt2_detail::Row) +
        descriptors_.size() * sizeof(gpulsmopt2_detail::Descriptor) +
        raw_assignments_.size() * sizeof(gpulsmopt2_detail::RawAssignment) +
        raw_full_keys_.size() * sizeof(std::uint32_t) +
        raw_offsets_.size() * sizeof(std::uint32_t) +
        raw_signatures_.size() * sizeof(std::uint64_t) +
        raw_epoch_signatures_.size() * sizeof(std::uint64_t) +
        level_keys_.size() * sizeof(std::uint32_t) +
        (publication_composite_keys_a_.size() +
         publication_composite_keys_b_.size()) * sizeof(std::uint64_t) +
        (publication_raw_rows_a_.size() +
         publication_raw_rows_b_.size() + publication_rows_a_.size() +
         publication_rows_b_.size()) * sizeof(gpulsmopt2_detail::Row) +
        (publication_keys_a_.size() + publication_keys_b_.size() +
         publication_selected_count_.size() +
         publication_batch_offsets_.size()) * sizeof(std::uint32_t) +
        publication_temp_.size() * sizeof(std::uint8_t) +
        radix_keys_.size() * sizeof(std::uint32_t) +
        radix_ids_out_.size() * sizeof(std::uint32_t) +
        radix_temp_.size() * sizeof(std::uint8_t) +
        range_partials_.size() * sizeof(unsigned long long) +
        range_fragment_counts_.size() * sizeof(std::uint32_t) +
        range_fragment_offsets_.size() * sizeof(std::uint32_t) +
        range_fragments_.size() * sizeof(gpulsmopt2_detail::RangeFragment) +
        range_fragment_partials_.size() * sizeof(unsigned long long) +
        (range_overflow_fragments_.size() + range_overflow_count_.size()) *
            sizeof(std::uint32_t) +
        (range_section_keys_in_.size() + range_section_keys_out_.size() +
         range_section_offsets_.size()) * sizeof(std::uint32_t) +
        (range_section_fragments_in_.size() +
         range_section_fragments_out_.size()) *
            sizeof(gpulsmopt2_detail::SectionRangeFragment) +
        range_section_tasks_.size() *
            sizeof(gpulsmopt2_detail::SectionRangeTask) +
        (range_section_task_counts_.size() +
         range_section_task_offsets_.size()) * sizeof(std::uint32_t) +
        range_section_max_fragments_.size() * sizeof(std::uint32_t) +
        range_temp_.size() * sizeof(std::uint8_t);
  }

private:
  static int blocks(std::size_t count) {
    return static_cast<int>((count + gpulsmopt2_detail::kThreads - 1u) /
                            gpulsmopt2_detail::kThreads);
  }

  static std::uint32_t levels_for_epochs(std::uint64_t epochs) {
    std::uint32_t levels = 0u;
    while (epochs) {
      ++levels;
      epochs >>= 1u;
    }
    return levels;
  }

  void initialize_publication_workspace() {
    const std::uint32_t epoch_capacity = static_cast<std::uint32_t>(
        batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch);
    const std::uint32_t capacity =
        static_cast<std::uint32_t>(publication_capacity_);
    std::size_t sort_bytes{}, raw_unique_bytes{}, merge_bytes{};
    std::size_t merge_unique_bytes{};
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, publication_composite_keys_a_.data(),
        publication_composite_keys_b_.data(),
        publication_raw_rows_a_.data(), publication_raw_rows_b_.data(),
        epoch_capacity, 0, 32 + publication_age_bits_, 0));
    CUDA_CHECK(cub::DeviceSelect::UniqueByKey(
        nullptr, raw_unique_bytes,
        publication_composite_keys_b_.data(),
        publication_raw_rows_b_.data(),
        publication_composite_keys_a_.data(),
        publication_rows_a_.data(), publication_selected_count_.data(),
        epoch_capacity,
        gpulsmopt2_detail::SameFullKey{publication_age_bits_}, 0));
    const std::uint32_t first = capacity / 2u;
    const std::uint32_t second = capacity - first;
    CUDA_CHECK(cub::DeviceMerge::MergePairs(
        nullptr, merge_bytes, publication_keys_a_.data(),
        publication_rows_a_.data(), first, level_keys_.data(),
        arena_.data(), second, publication_keys_b_.data(),
        publication_rows_b_.data(), cuda::std::less<>{}, 0));
    CUDA_CHECK(cub::DeviceSelect::UniqueByKey(
        nullptr, merge_unique_bytes, publication_keys_b_.data(),
        publication_rows_b_.data(), publication_keys_a_.data(),
        publication_rows_a_.data(), publication_selected_count_.data(),
        capacity, cuda::std::equal_to<>{}, 0));
    publication_temp_.resize(std::max({sort_bytes, raw_unique_bytes,
                                       merge_bytes, merge_unique_bytes}));
  }

  std::uint32_t allocate_level_span(std::uint32_t count) const {
    struct Interval { std::uint32_t begin, end; };
    std::vector<Interval> occupied;
    for (std::uint32_t level = 0u;
         level < gpulsmopt2_detail::kMaximumLevels; ++level) {
      if (level_counts_[level])
        occupied.push_back({level_offsets_[level],
                            level_offsets_[level] + level_counts_[level]});
    }
    std::sort(occupied.begin(), occupied.end(),
              [](const Interval &a, const Interval &b) {
                return a.begin < b.begin;
              });
    std::uint32_t cursor = 0u;
    for (const Interval &interval : occupied) {
      if (interval.begin >= cursor && interval.begin - cursor >= count)
        return cursor;
      cursor = std::max(cursor, interval.end);
    }
    if (std::uint64_t{cursor} + count <= arena_.size()) return cursor;
    throw std::runtime_error("GPULSMOpt publication arena exhausted");
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
    CUDA_CHECK(cudaMemsetAsync(raw_offsets_.data(), 0,
                               raw_offsets_.size() * sizeof(std::uint32_t),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(
        raw_epoch_signatures_.data(), 0,
        raw_epoch_signatures_.size() * sizeof(std::uint64_t), stream));
    pending_batches_ = 0u;
    pending_records_ = 0u;
    active_levels_ = 0u;
    std::fill_n(level_counts_, gpulsmopt2_detail::kMaximumLevels, 0u);
    std::fill_n(level_offsets_, gpulsmopt2_detail::kMaximumLevels, 0u);
    std::fill_n(raw_batch_counts_, gpulsmopt2_detail::kBatchesPerEpoch, 0u);
    stats_ = {};
  }

  void admit(const std::uint32_t *keys, const std::uint32_t *values,
             std::size_t count, bool tombstone, cudaStream_t stream) {
    if (!count) return;
    if (!keys || (!tombstone && !values) || count > batch_capacity_)
      throw std::invalid_argument("invalid GPULSMOpt update batch");
    begin_operation(stream);
    const std::uint32_t n = static_cast<std::uint32_t>(count);
    std::size_t sort_bytes{};
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, keys, radix_keys_.data(), radix_ids_out_.data(),
        radix_ids_out_.data(), n, 16, 32, stream));
    ensure_radix_workspace(sort_bytes, n, stream);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        radix_workspace(), sort_bytes, keys, radix_keys_.data(),
        radix_input_ids(), radix_ids_out_.data(), n, 16, 32, stream));
    const std::uint32_t slot = pending_batches_;
    pending_records_ += n;
    std::uint64_t *batch_signatures = raw_signatures_.data() +
        std::size_t{slot} * gpulsmopt2_detail::kQuotients;
    CUDA_CHECK(cudaMemsetAsync(
        batch_signatures, 0,
        gpulsmopt2_detail::kQuotients * sizeof(std::uint64_t), stream));
    gpulsmopt2_detail::gather_raw_batch_kernel<<<
        blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
        radix_keys_.data(), radix_ids_out_.data(), values, n, slot, tombstone,
        raw_assignments_.data() + std::size_t{slot} * batch_capacity_,
        raw_full_keys_.data() + std::size_t{slot} * batch_capacity_,
        batch_signatures, raw_epoch_signatures_.data());
    std::uint32_t *batch_offsets = raw_offsets_.data() +
        std::size_t{slot} * (gpulsmopt2_detail::kQuotients + 1u);
    gpulsmopt2_detail::finalize_quotient_metadata_kernel<<<
        gpulsmopt2_detail::kQuotients / gpulsmopt2_detail::kThreads,
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            radix_keys_.data(), n, batch_offsets);
    CUDA_CHECK(cudaGetLastError());
    raw_batch_counts_[slot] = n;
    ++pending_batches_;
    ++stats_.admitted_batches;
    stats_.admitted_records += count;
    if (pending_batches_ == gpulsmopt2_detail::kBatchesPerEpoch)
      publish_epoch(stream);
    end_operation(stream);
  }

  void publish_epoch(cudaStream_t stream) {
    const std::uint32_t publication_levels =
        levels_for_epochs(stats_.epochs_published + 1u);
    if (publication_levels > gpulsmopt2_detail::kMaximumLevels)
      throw std::runtime_error("GPULSMOpt descriptor capacity exhausted");
    if (pending_records_ > publication_capacity_)
      throw std::runtime_error("GPULSMOpt publication workspace exhausted");
    std::uint32_t carry_levels = 0u;
    while (carry_levels < gpulsmopt2_detail::kMaximumLevels &&
           level_counts_[carry_levels])
      ++carry_levels;
    if (carry_levels == gpulsmopt2_detail::kMaximumLevels)
      throw std::runtime_error("GPULSMOpt descriptor capacity exhausted");
    std::uint32_t destination = gpulsmopt2_detail::kInvalid;
    if (carry_levels == 0u)
      destination = allocate_level_span(pending_records_);
    std::uint32_t batch_offsets[gpulsmopt2_detail::kBatchesPerEpoch + 1u]{};
    for (std::uint32_t batch = 0u;
         batch < gpulsmopt2_detail::kBatchesPerEpoch; ++batch)
      batch_offsets[batch + 1u] =
          batch_offsets[batch] + raw_batch_counts_[batch];
    CUDA_CHECK(cudaMemcpyAsync(
        publication_batch_offsets_.data(), batch_offsets,
        sizeof(batch_offsets), cudaMemcpyHostToDevice, stream));
    const dim3 publication_grid(
        blocks(batch_capacity_), gpulsmopt2_detail::kBatchesPerEpoch);
    gpulsmopt2_detail::pack_publication_epoch_kernel<<<
        publication_grid, gpulsmopt2_detail::kThreads, 0, stream>>>(
            raw_assignments_.data(), raw_full_keys_.data(),
            static_cast<std::uint32_t>(batch_capacity_),
            publication_batch_offsets_.data(), publication_age_bits_,
            publication_composite_keys_a_.data(),
            publication_raw_rows_a_.data());
    std::size_t workspace_bytes = publication_temp_.size();
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        publication_temp_.data(), workspace_bytes,
        publication_composite_keys_a_.data(),
        publication_composite_keys_b_.data(),
        publication_raw_rows_a_.data(), publication_raw_rows_b_.data(),
        pending_records_, 0, 32 + publication_age_bits_, stream));
    workspace_bytes = publication_temp_.size();
    CUDA_CHECK(cub::DeviceSelect::UniqueByKey(
        publication_temp_.data(), workspace_bytes,
        publication_composite_keys_b_.data(),
        publication_raw_rows_b_.data(),
        publication_composite_keys_a_.data(),
        carry_levels ? publication_rows_a_.data()
                     : arena_.data() + destination,
        publication_selected_count_.data(), pending_records_,
        gpulsmopt2_detail::SameFullKey{publication_age_bits_}, stream));
    std::uint32_t current_count{};
    CUDA_CHECK(cudaMemcpyAsync(
        &current_count, publication_selected_count_.data(),
        sizeof(current_count), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    gpulsmopt2_detail::normalize_publication_keys_kernel<<<
        blocks(current_count), gpulsmopt2_detail::kThreads, 0, stream>>>(
            publication_composite_keys_a_.data(),
            carry_levels ? publication_keys_a_.data()
                         : level_keys_.data() + destination,
            publication_selected_count_.data(), publication_age_bits_);

    const std::uint32_t destination_level = carry_levels;
    if (carry_levels) {
      std::uint32_t merged_count = current_count;
      for (std::uint32_t level = 0u; level < carry_levels; ++level) {
        merged_count += level_counts_[level];
      }
      if (merged_count > publication_capacity_)
        throw std::runtime_error("GPULSMOpt publication workspace exhausted");
      bool current_is_a = true;
      std::uint32_t raw_count = current_count;
      for (std::uint32_t level = 0u; level < carry_levels; ++level) {
        const std::uint32_t count = level_counts_[level];
        workspace_bytes = publication_temp_.size();
        CUDA_CHECK(cub::DeviceMerge::MergePairs(
            publication_temp_.data(), workspace_bytes,
            current_is_a ? publication_keys_a_.data()
                         : publication_keys_b_.data(),
            current_is_a ? publication_rows_a_.data()
                         : publication_rows_b_.data(),
            raw_count, level_keys_.data() + level_offsets_[level],
            arena_.data() + level_offsets_[level], count,
            current_is_a ? publication_keys_b_.data()
                         : publication_keys_a_.data(),
            current_is_a ? publication_rows_b_.data()
                         : publication_rows_a_.data(),
            cuda::std::less<>{}, stream));
        raw_count += count;
        current_is_a = !current_is_a;
      }
      for (std::uint32_t level = 0u; level < carry_levels; ++level)
        level_counts_[level] = 0u;
      destination = allocate_level_span(merged_count);
      workspace_bytes = publication_temp_.size();
      CUDA_CHECK(cub::DeviceSelect::UniqueByKey(
          publication_temp_.data(), workspace_bytes,
          current_is_a ? publication_keys_a_.data()
                       : publication_keys_b_.data(),
          current_is_a ? publication_rows_a_.data()
                       : publication_rows_b_.data(),
          level_keys_.data() + destination, arena_.data() + destination,
          publication_selected_count_.data(), merged_count,
          cuda::std::equal_to<>{}, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          &current_count, publication_selected_count_.data(),
          sizeof(current_count), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    gpulsmopt2_detail::publish_global_level_descriptors_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            level_keys_.data() + destination, current_count, destination,
            destination_level, descriptors_.data());
    level_offsets_[destination_level] = destination;
    level_counts_[destination_level] = current_count;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(
        raw_epoch_signatures_.data(), 0,
        raw_epoch_signatures_.size() * sizeof(std::uint64_t), stream));
    pending_batches_ = 0u;
    pending_records_ = 0u;
    active_levels_ = publication_levels;
    ++stats_.epochs_published;
  }

  static std::size_t aligned_id_bytes(std::size_t count) {
    return (count * sizeof(std::uint32_t) + 255u) & ~std::size_t{255u};
  }
  void ensure_radix_workspace(std::size_t bytes, std::size_t count,
                              cudaStream_t stream) {
    ensure_radix_buffers(count);
    const std::size_t capacity = std::max(radix_id_capacity_, count);
    const std::size_t required = aligned_id_bytes(capacity) + bytes;
    const bool resized = radix_temp_.size() < required;
    if (resized) radix_temp_.resize(required);
    if (resized || count > radix_id_capacity_) {
      gpulsmopt2_detail::iota_kernel<<<
          blocks(capacity), gpulsmopt2_detail::kThreads, 0, stream>>>(
              reinterpret_cast<std::uint32_t *>(radix_temp_.data()),
              static_cast<std::uint32_t>(capacity));
      CUDA_CHECK(cudaGetLastError());
    }
    radix_id_capacity_ = capacity;
  }
  std::uint32_t *radix_input_ids() {
    return reinterpret_cast<std::uint32_t *>(radix_temp_.data());
  }
  void *radix_workspace() {
    return radix_temp_.data() + aligned_id_bytes(radix_id_capacity_);
  }
  template <bool HasPending, bool Tiled>
  void launch_section_ranges(std::uint32_t maximum_tasks,
                             cudaStream_t stream) {
    const std::uint32_t grid =
        Tiled ? maximum_tasks : gpulsmopt2_detail::kQuotients;
    const auto *tasks = Tiled ? range_section_tasks_.data() : nullptr;
    const auto *task_count = Tiled
        ? range_section_task_offsets_.data() + gpulsmopt2_detail::kQuotients
        : nullptr;
    gpulsmopt2_detail::cooperative_section_owned_range_kernel<
        gpulsmopt2_detail::SumRowsAggregate, HasPending, Tiled>
        <<<grid, gpulsmopt2_detail::kThreads, 0, stream>>>(
            range_section_fragments_out_.data(),
            range_section_offsets_.data(), tasks, task_count,
            base_keys_.data(), base_values_.data(), base_offsets_.data(),
            arena_.data(), descriptors_.data(), raw_assignments_.data(),
            raw_offsets_.data(), static_cast<std::uint32_t>(batch_capacity_),
            HasPending ? pending_batches_ : 0u,
            active_levels_,
            range_fragment_partials_.data());
  }
  template <bool HasPending>
  void launch_fragment_ranges(std::uint32_t fragment_count,
                              std::uint32_t query_count,
                              const DeviceRangeOutputBatch &batch,
                              cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(range_overflow_count_.data(), 0,
                               sizeof(std::uint32_t), stream));
    gpulsmopt2_detail::warp_range_fragment_kernel<
        gpulsmopt2_detail::SumRowsAggregate, HasPending>
        <<<(fragment_count + 3u) / 4u, 128, 0, stream>>>(
            range_fragments_.data(), fragment_count,
            range_fragment_offsets_.data() + query_count,
            batch.lo, batch.hi, base_keys_.data(), base_values_.data(),
            base_offsets_.data(), arena_.data(), descriptors_.data(),
            raw_assignments_.data(), raw_offsets_.data(),
            static_cast<std::uint32_t>(batch_capacity_),
            HasPending ? pending_batches_ : 0u,
            active_levels_,
            range_fragment_partials_.data(),
            range_overflow_fragments_.data(), range_overflow_count_.data());
    gpulsmopt2_detail::overflow_range_fragment_kernel<HasPending>
        <<<256, 128, 0, stream>>>(
            range_overflow_fragments_.data(), range_overflow_count_.data(),
            range_fragments_.data(), batch.lo, batch.hi,
            base_keys_.data(), base_values_.data(), base_offsets_.data(),
            arena_.data(), descriptors_.data(), raw_assignments_.data(),
            raw_offsets_.data(), static_cast<std::uint32_t>(batch_capacity_),
            HasPending ? pending_batches_ : 0u,
            active_levels_,
            range_fragment_partials_.data());
  }
  void ensure_range_temp(std::size_t bytes) {
    if (range_temp_.size() < bytes) range_temp_.resize(bytes);
  }
  void ensure_range_fragment_query_capacity(std::size_t count) {
    if (range_fragment_counts_.size() >= count + 1u) return;
    range_fragment_counts_.resize(count + 1u);
    range_fragment_offsets_.resize(count + 1u);
  }
  void ensure_range_fragment_capacity(std::size_t count) {
    if (range_fragments_.size() >= count) return;
    range_fragments_.resize(count);
    range_overflow_fragments_.resize(count);
  }
  void ensure_range_partial_capacity(std::size_t count) {
    if (range_fragment_partials_.size() < count)
      range_fragment_partials_.resize(count);
  }
  void ensure_range_section_capacity(std::size_t count) {
    if (range_section_keys_in_.size() < count) {
      range_section_keys_in_.resize(count);
      range_section_keys_out_.resize(count);
      range_section_fragments_in_.resize(count);
      range_section_fragments_out_.resize(count);
    }
    if (range_section_offsets_.size() < gpulsmopt2_detail::kQuotients + 1u)
      range_section_offsets_.resize(gpulsmopt2_detail::kQuotients + 1u);
    const std::size_t maximum_tasks = gpulsmopt2_detail::kQuotients +
        (count + gpulsmopt2_detail::kSectionTaskFragments - 1u) /
            gpulsmopt2_detail::kSectionTaskFragments;
    if (range_section_tasks_.size() < maximum_tasks)
      range_section_tasks_.resize(maximum_tasks);
    if (range_section_task_offsets_.size() <
        gpulsmopt2_detail::kQuotients + 1u)
      range_section_task_offsets_.resize(
          gpulsmopt2_detail::kQuotients + 1u);
    if (range_section_task_counts_.size() <
        gpulsmopt2_detail::kQuotients + 1u)
      range_section_task_counts_.resize(
          gpulsmopt2_detail::kQuotients + 1u);
    if (!range_section_max_fragments_.size())
      range_section_max_fragments_.resize(1u);
  }
  void ensure_radix_buffers(std::size_t count) {
    if (radix_keys_.size() >= count) return;
    radix_keys_.resize(count);
    radix_ids_out_.resize(count);
  }

  std::size_t batch_capacity_{};
  std::size_t publication_capacity_{};
  std::uint32_t publication_age_bits_{};
  std::uint32_t pending_batches_{};
  std::uint32_t pending_records_{};
  std::uint32_t active_levels_{};
  std::size_t radix_id_capacity_{};
  MaintenanceStats stats_{};
  cudaEvent_t operation_done_{};

  gpulsmopt2_detail::Buffer<std::uint16_t> base_keys_;
  gpulsmopt2_detail::Buffer<std::uint32_t> base_values_, base_offsets_;
  gpulsmopt2_detail::Buffer<std::uint16_t> local_rank_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::Row> arena_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::Descriptor> descriptors_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RawAssignment> raw_assignments_;
  gpulsmopt2_detail::Buffer<std::uint32_t> raw_full_keys_;
  gpulsmopt2_detail::Buffer<std::uint32_t> raw_offsets_;
  gpulsmopt2_detail::Buffer<std::uint64_t> raw_signatures_;
  gpulsmopt2_detail::Buffer<std::uint64_t> raw_epoch_signatures_;
  gpulsmopt2_detail::Buffer<std::uint32_t> level_keys_;
  gpulsmopt2_detail::Buffer<std::uint64_t> publication_composite_keys_a_,
      publication_composite_keys_b_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::Row> publication_raw_rows_a_,
      publication_raw_rows_b_;
  gpulsmopt2_detail::Buffer<std::uint32_t> publication_keys_a_,
      publication_keys_b_, publication_selected_count_,
      publication_batch_offsets_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::Row> publication_rows_a_,
      publication_rows_b_;
  gpulsmopt2_detail::Buffer<std::uint8_t> publication_temp_;
  std::uint32_t level_counts_[gpulsmopt2_detail::kMaximumLevels]{};
  std::uint32_t level_offsets_[gpulsmopt2_detail::kMaximumLevels]{};
  std::uint32_t raw_batch_counts_[gpulsmopt2_detail::kBatchesPerEpoch]{};

  gpulsmopt2_detail::Buffer<std::uint32_t> radix_keys_, radix_ids_out_;
  gpulsmopt2_detail::Buffer<std::uint8_t> radix_temp_;

  gpulsmopt2_detail::Buffer<unsigned long long> range_partials_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_fragment_counts_,
      range_fragment_offsets_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RangeFragment> range_fragments_;
  gpulsmopt2_detail::Buffer<unsigned long long> range_fragment_partials_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_overflow_fragments_,
      range_overflow_count_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_section_keys_in_,
      range_section_keys_out_, range_section_offsets_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::SectionRangeFragment>
      range_section_fragments_in_, range_section_fragments_out_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::SectionRangeTask>
      range_section_tasks_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_section_task_counts_,
      range_section_task_offsets_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_section_max_fragments_;
  gpulsmopt2_detail::Buffer<std::uint8_t> range_temp_;
};
