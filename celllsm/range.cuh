#pragma once

// Range sum: per-section pieces, newer rows resolved on chip or in a
// global pass for crowded sections, then summed against the oldest run.

#include "pending.cuh"

namespace gpulsmopt2_detail {

constexpr std::uint32_t kRangeSchedulerBlocks = 256u;
constexpr std::uint32_t kSectionRangeThreads = 128u;

constexpr std::uint32_t kSectionOwnerMinimumReuse = 4u;
constexpr std::uint32_t kRangeThreadWork = 8u;
constexpr std::uint32_t kRangeSubgroupWork = 512u;
constexpr std::uint32_t kRangeOnChipNewerRows = 128u;
constexpr std::uint32_t kRangeHotWindowRows = 1u << 26u;

struct TaggedRow {
  Row row;
  std::uint32_t age;
};

static_assert(sizeof(TaggedRow) == 12u);

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
constexpr std::uint32_t kSectionTaskFragments = kSectionRangeThreads;

struct RangeFragmentBounds {
  std::uint32_t update_begin;
  std::uint32_t update_end;
  std::uint32_t base_begin;
  std::uint32_t base_end;
};

static_assert(sizeof(RangeFragmentBounds) == 16u);

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

constexpr std::uint32_t kRangeHotRawLocator = 1u << 31u;

__host__ __device__ __forceinline__ std::uint64_t range_hot_token(
    std::uint32_t key, std::uint32_t locator) {
  return (std::uint64_t{key} << 32u) | locator;
}

__host__ __device__ __forceinline__ std::uint32_t range_hot_key(
    std::uint64_t token) {
  return static_cast<std::uint32_t>(token >> 32u);
}

__host__ __device__ __forceinline__ std::uint32_t range_hot_locator(
    std::uint64_t token) {
  return static_cast<std::uint32_t>(token);
}

struct RangeHotTokenKey {
  __host__ __device__ std::uint32_t operator()(std::uint64_t token) const {
    return range_hot_key(token);
  }
};

struct RangeHotNewestToken {
  const RawPayload *raw_payloads{};
  std::uint32_t raw_record_capacity{};

  __device__ bool valid(std::uint64_t token) const {
    const std::uint32_t locator = range_hot_locator(token);
    return locator & kRangeHotRawLocator
        ? (locator & ~kRangeHotRawLocator) < raw_record_capacity
        : (locator >> 16u) < kMaximumLevels;
  }

  __device__ std::uint32_t age(std::uint64_t token) const {
    const std::uint32_t locator = range_hot_locator(token);
    if (locator & kRangeHotRawLocator)
      return kMaximumLevels +
          raw_position(raw_payloads[locator & ~kRangeHotRawLocator]);
    return kMaximumLevels - 1u - (locator >> 16u);
  }

  __device__ std::uint64_t operator()(std::uint64_t first,
                                      std::uint64_t second) const {
    const bool first_valid = valid(first), second_valid = valid(second);
    if (first_valid != second_valid) return second_valid ? second : first;
    if (!first_valid) return 0u;
    return age(second) > age(first) ? second : first;
  }
};

struct RangeHotTokenRow {
  const RawPayload *raw_payloads{};
  ResidentRows arena{};
  const Descriptor *descriptors{};

  __device__ Row operator()(std::uint64_t token) const {
    const std::uint32_t key = range_hot_key(token);
    const std::uint32_t locator = range_hot_locator(token);
    if (locator & kRangeHotRawLocator) {
      const std::uint32_t record = locator & ~kRangeHotRawLocator;
      return raw_row(key, raw_payloads[record]);
    }
    const std::uint32_t level = locator >> 16u;
    const std::uint32_t position = locator & 0xffffu;
    const Descriptor section = descriptors[descriptor_index(key >> 16u, level)];
    return arena[section.offset() + position];
  }
};

__global__ void count_range_hot_newer_rows_kernel(
    const std::uint32_t *raw_offsets, std::uint32_t pending_batches,
    const Descriptor *descriptors, const std::uint64_t *occupied_mask,
    std::uint64_t *counts) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  if (q == kQuotients) {
    counts[q] = 0u;
    return;
  }
  const DeviceManifestSnapshot manifest = load_query_manifest(occupied_mask);
  std::uint64_t physical = 0u;
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t index =
        std::size_t{batch} * (kQuotients + 1u) + q;
    physical += raw_offsets[index + 1u] - raw_offsets[index];
  }
  for (std::uint32_t level = 0u; level < manifest.active_levels; ++level)
    if (level != manifest.foundation_level &&
        level_is_occupied(manifest.occupied_level_mask, level))
      physical += descriptors[descriptor_index(q, level)].count();
  counts[q] = physical > kRangeOnChipNewerRows ? physical : 0u;
}

__global__ void make_range_hot_window_offsets_kernel(
    const std::uint64_t *global_offsets, std::uint32_t quotient_begin,
    std::uint32_t quotient_count, std::uint64_t window_base,
    std::uint32_t *window_offsets) {
  const std::uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local > quotient_count) return;
  window_offsets[local] = static_cast<std::uint32_t>(
      global_offsets[quotient_begin + local] - window_base);
}

__global__ void emit_range_hot_tokens_kernel(
    const std::uint32_t *raw_keys, const std::uint32_t *raw_offsets,
    std::uint32_t batch_stride, std::uint32_t pending_batches,
    ResidentRows arena, const Descriptor *descriptors,
    const std::uint64_t *occupied_mask, const std::uint64_t *hot_counts,
    const std::uint64_t *hot_offsets, std::uint32_t quotient_begin,
    std::uint32_t quotient_end, std::uint64_t window_base,
    std::uint64_t *tokens) {
  const std::uint32_t q = quotient_begin + blockIdx.x;
  if (q >= quotient_end || !hot_counts[q]) return;
  const DeviceManifestSnapshot manifest = load_query_manifest(occupied_mask);
  std::uint64_t cursor = hot_offsets[q] - window_base;
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t offset_index =
        std::size_t{batch} * (kQuotients + 1u) + q;
    const std::uint32_t begin = raw_offsets[offset_index];
    const std::uint32_t end = raw_offsets[offset_index + 1u];
    for (std::uint32_t position = begin + threadIdx.x;
         position < end; position += blockDim.x) {
      const std::uint32_t record = batch * batch_stride + position;
      tokens[cursor + position - begin] = range_hot_token(
          raw_keys[record], kRangeHotRawLocator | record);
    }
    cursor += end - begin;
  }
  for (std::uint32_t level = 0u; level < manifest.active_levels; ++level) {
    if (level == manifest.foundation_level ||
        !level_is_occupied(manifest.occupied_level_mask, level))
      continue;
    const Descriptor section = descriptors[descriptor_index(q, level)];
    const ResidentRows rows = arena + section.offset();
    for (std::uint32_t position = threadIdx.x;
         position < section.count(); position += blockDim.x) {
      const std::uint32_t locator = (level << 16u) | position;
      tokens[cursor + position] = range_hot_token(
          full_key(q, rows[position].key), locator);
    }
    cursor += descriptors[descriptor_index(q, level)].count();
  }
}

__global__ void build_range_hot_descriptors_kernel(
    const std::uint32_t *keys, const std::uint32_t *selected_count,
    std::uint32_t quotient_begin, std::uint32_t quotient_end,
    std::uint64_t output_base, Descriptor *descriptors) {
  const std::uint32_t q = quotient_begin +
      blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= quotient_end) return;
  const std::uint32_t count = *selected_count;
  const std::uint64_t low_key = std::uint64_t{q} << 16u;
  const std::uint64_t high_key = std::uint64_t{q + 1u} << 16u;
  std::uint32_t low = 0u, high = count;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (std::uint64_t{keys[middle]} < low_key) low = middle + 1u;
    else high = middle;
  }
  const std::uint32_t begin = low;
  high = count;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (std::uint64_t{keys[middle]} < high_key) low = middle + 1u;
    else high = middle;
  }
  descriptors[q] = Descriptor::make(output_base + begin, low - begin);
}

__global__ void materialize_range_hot_winners_kernel(
    Row *rows, const std::uint32_t *selected_count,
    const RawPayload *raw_payloads, ResidentRows arena,
    const Descriptor *descriptors) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= *selected_count) return;
  std::uint64_t token{};
  memcpy(&token, rows + index, sizeof(token));
  const RangeHotTokenRow transform{
      raw_payloads, arena, descriptors};
  rows[index] = transform(token);
}

// The enumeration visits records before this output sink.
struct RangeSumSink {
  using State = unsigned long long;
  __device__ static State identity() { return 0ull; }
  __device__ static State emit(State state, const Row &row) {
    return state + row.value;
  }
  __device__ State operator()(State a, State b) const { return a + b; }
};

__device__ __forceinline__ std::uint32_t section_range_row_count(
    std::uint32_t low, std::uint32_t high, ResidentRows arena,
    Descriptor section) {
  const ResidentRows rows = arena + section.offset();
  return upper_bound_rows(rows, section.count(), high) -
      lower_bound_rows(rows, section.count(), low);
}

// Scan base rows against the resolved newer run.
template <class Sink>
__device__ __forceinline__ typename Sink::State
enumerate_visible_section(
    std::uint32_t low, std::uint32_t high,
    const Row *current, std::uint32_t current_count,
    ResidentRows arena, Descriptor base,
    std::uint32_t group_lane, std::uint32_t group_size) {
  typename Sink::State result = Sink::identity();
  const std::uint32_t update_begin =
      lower_bound_rows(current, current_count, low);
  const std::uint32_t update_end =
      upper_bound_rows(current, current_count, high);
  for (std::uint32_t index = update_begin + group_lane;
       index < update_end; index += group_size) {
    const Row row = current[index];
    if ((row.flags & kTombstone) == 0u)
      result = Sink::emit(result, row);
  }

  const ResidentRows rows = arena + base.offset();
  const std::uint32_t begin = lower_bound_rows(rows, base.count(), low);
  const std::uint32_t end =
      upper_bound_rows(rows, base.count(), high);
  const std::uint32_t count = end - begin;
  const std::uint32_t lane_begin =
      begin + (std::uint64_t{count} * group_lane) / group_size;
  const std::uint32_t lane_end =
      begin + (std::uint64_t{count} * (group_lane + 1u)) / group_size;
  if (lane_begin == lane_end) return result;

  // Merge one base interval per worker.
  std::uint32_t update = update_begin + lower_bound_rows(
      current + update_begin, update_end - update_begin,
      rows[lane_begin].key);
  for (std::uint32_t index = lane_begin; index < lane_end; ++index) {
    const Row row = rows[index];
    while (update < update_end && current[update].key < row.key) ++update;
    const bool covered =
        update < update_end && current[update].key == row.key;
    if (!covered && (row.flags & kTombstone) == 0u)
      result = Sink::emit(result, row);
  }
  return result;
}

struct RangeTileSource {
  const Row *plain{};
  ResidentRows resident{};
  bool plain_storage{};

  __device__ Row load(std::uint32_t position) const {
    return plain_storage ? plain[position] : resident[position];
  }
};

template <class Sink>
__device__ __forceinline__ void enumerate_range_source_tiles(
    std::uint32_t fragment_count, RangeTileSource source,
    const Row *newer, std::uint32_t newer_count, bool hide_with_newer,
    const std::uint32_t *source_begins,
    const std::uint32_t *source_ends,
    const std::uint8_t *fragment_slots,
    const std::uint8_t *fragment_widths,
    const std::uint16_t *wave_fragment_begins,
    std::uint32_t wave_count,
    std::uint32_t *union_begins, std::uint32_t *union_ends,
    std::uint32_t *union_count, Row *row_tile,
    typename Sink::State *fragment_sums) {
  if (threadIdx.x == 0u) {
    std::uint32_t count = 0u;
    for (std::uint32_t fragment = 0u; fragment < fragment_count;
         ++fragment) {
      const std::uint32_t begin = source_begins[fragment];
      const std::uint32_t end = source_ends[fragment];
      if (begin == end) continue;
      if (!count || begin > union_ends[count - 1u]) {
        union_begins[count] = begin;
        union_ends[count] = end;
        ++count;
      } else {
        union_ends[count - 1u] = max(union_ends[count - 1u], end);
      }
    }
    *union_count = count;
  }
  __syncthreads();

  for (std::uint32_t interval = 0u; interval < *union_count; ++interval) {
    const std::uint32_t interval_begin = union_begins[interval];
    const std::uint32_t interval_end = union_ends[interval];
    for (std::uint32_t tile_begin = interval_begin;
         tile_begin < interval_end; tile_begin += kSectionRangeThreads) {
      const std::uint32_t tile_end =
          min(tile_begin + kSectionRangeThreads, interval_end);
      if (tile_begin + threadIdx.x < tile_end) {
        Row row = source.load(tile_begin + threadIdx.x);
        if (hide_with_newer && newer_count) {
          const std::uint32_t position =
              lower_bound_rows(newer, newer_count, row.key);
          if (position < newer_count && newer[position].key == row.key)
            row.flags |= kTombstone;
        }
        row_tile[threadIdx.x] = row;
      }
      __syncthreads();

      for (std::uint32_t wave = 0u; wave < wave_count; ++wave) {
        const std::uint32_t first = wave_fragment_begins[wave];
        const std::uint32_t last = wave_fragment_begins[wave + 1u];
        std::uint32_t low = first, high = last;
        while (low < high) {
          const std::uint32_t middle = (low + high) >> 1u;
          if (fragment_slots[middle] <= threadIdx.x) low = middle + 1u;
          else high = middle;
        }
        const std::uint32_t fragment = low ? low - 1u : last;
        const bool active = fragment >= first && fragment < last &&
            threadIdx.x >= fragment_slots[fragment] &&
            threadIdx.x < std::uint32_t{fragment_slots[fragment]} +
                fragment_widths[fragment];
        if (active) {
          const std::uint32_t width = fragment_widths[fragment];
          const std::uint32_t slot = fragment_slots[fragment];
          const std::uint32_t group_lane = threadIdx.x - slot;
          const std::uint32_t consume_begin =
              max(source_begins[fragment], tile_begin);
          const std::uint32_t consume_end =
              min(source_ends[fragment], tile_end);
          typename Sink::State local = Sink::identity();
          for (std::uint32_t position = consume_begin + group_lane;
               position < consume_end; position += width) {
            const Row row = row_tile[position - tile_begin];
            if ((row.flags & kTombstone) == 0u)
              local = Sink::emit(local, row);
          }
          const std::uint32_t warp_slot = slot & 31u;
          const unsigned mask = width == 32u ? 0xffffffffu
              : ((1u << width) - 1u) << warp_slot;
          for (std::uint32_t offset = width >> 1u; offset;
               offset >>= 1u)
            local = Sink{}(local, __shfl_down_sync(
                mask, local, offset, width));
          if (!group_lane)
            fragment_sums[fragment] = Sink{}(
                fragment_sums[fragment], local);
        }
      }
      __syncthreads();
    }
  }
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
  const std::uint32_t count =
      lo <= hi ? (hi >> 16u) - (lo >> 16u) + 1u : 0u;
  counts[query] = count;
}

struct WidenFragmentCount {
  __host__ __device__ std::uint64_t operator()(
      std::uint32_t count) const {
    return count;
  }
};

__device__ __forceinline__ void emit_range_fragment(
    std::uint32_t index, std::uint32_t query, std::uint32_t quotient,
    std::uint32_t low, std::uint32_t high, RangeFragment *fragments,
    std::uint32_t *section_keys, SectionRangeFragment *section_fragments,
    bool section_owned) {
  if (section_owned) {
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

__global__ void adaptive_emit_range_fragments_kernel(
    const std::uint32_t *low, const std::uint32_t *high,
    const std::uint32_t *offsets, std::uint32_t query_count,
    RangeFragment *fragments, std::uint32_t *section_keys,
    SectionRangeFragment *section_fragments, bool section_owned) {
  constexpr std::uint32_t kWarps = kThreads / 32u;
  constexpr std::uint32_t kThreadFragments = 4u;
  const std::uint32_t lane = threadIdx.x & 31u;
  constexpr unsigned full_mask = 0xffffffffu;
  if (query_count == 1u) {
    const std::uint32_t count = offsets[1u];
    const std::uint32_t first = low[0u] >> 16u;
    for (std::uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
         local < count; local += gridDim.x * blockDim.x)
      emit_range_fragment(
          local, 0u, first + local, low[0u], high[0u], fragments,
          section_keys, section_fragments, section_owned);
    return;
  }

  const std::uint32_t warp_in_block = threadIdx.x >> 5u;
  const std::uint32_t global_warp = blockIdx.x * kWarps + warp_in_block;
  const std::uint32_t warp_count = gridDim.x * kWarps;
  for (std::uint32_t query_base = global_warp * 32u;
       query_base < query_count; query_base += warp_count * 32u) {
    const std::uint32_t query = query_base + lane;
    const bool valid = query < query_count && low[query] <= high[query];
    const std::uint32_t query_offset = valid ? offsets[query] : 0u;
    const std::uint32_t query_low = valid ? low[query] : 0u;
    const std::uint32_t query_high = valid ? high[query] : 0u;
    const std::uint32_t count = valid
        ? offsets[query + 1u] - offsets[query] : 0u;
    if (valid && count <= kThreadFragments) {
      const std::uint32_t first = query_low >> 16u;
      for (std::uint32_t local = 0u; local < count; ++local)
        emit_range_fragment(
            query_offset + local, query, first + local, query_low,
            query_high, fragments, section_keys, section_fragments,
            section_owned);
    }

    unsigned wide = __ballot_sync(
        full_mask, valid && count > kThreadFragments);
    while (wide) {
      const std::uint32_t owner = __ffs(wide) - 1u;
      const std::uint32_t wide_query = query_base + owner;
      const std::uint32_t wide_offset =
          __shfl_sync(full_mask, query_offset, owner);
      const std::uint32_t wide_count =
          __shfl_sync(full_mask, count, owner);
      const std::uint32_t wide_low =
          __shfl_sync(full_mask, query_low, owner);
      const std::uint32_t wide_high =
          __shfl_sync(full_mask, query_high, owner);
      const std::uint32_t first = wide_low >> 16u;
      for (std::uint32_t local = lane; local < wide_count; local += 32u)
        emit_range_fragment(
            wide_offset + local, wide_query, first + local, wide_low,
            wide_high, fragments, section_keys, section_fragments,
            section_owned);
      wide &= ~(1u << owner);
    }
  }
}

__global__ void adaptive_reduce_range_partials_kernel(
    const std::uint32_t *offsets, std::uint32_t query_count,
    const unsigned long long *partials, std::uint32_t *out_sums,
    unsigned long long *block_partials, std::uint32_t *completion_count) {
  constexpr std::uint32_t kWarps = kThreads / 32u;
  constexpr std::uint32_t kThreadPartials = 4u;
  constexpr unsigned full_mask = 0xffffffffu;
  __shared__ unsigned long long values[kThreads];
  __shared__ std::uint32_t last_block;

  if (query_count == 1u) {
    unsigned long long local = 0ull;
    const std::uint32_t count = offsets[1u];
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += gridDim.x * blockDim.x)
      local += partials[index];
    values[threadIdx.x] = local;
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2u; stride; stride >>= 1u) {
      if (threadIdx.x < stride)
        values[threadIdx.x] += values[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0u) {
      block_partials[blockIdx.x] = values[0u];
      __threadfence();
      last_block = atomicInc(completion_count, gridDim.x - 1u) ==
          gridDim.x - 1u;
    }
    __syncthreads();
    if (!last_block) return;

    local = 0ull;
    for (std::uint32_t block = threadIdx.x; block < gridDim.x;
         block += blockDim.x)
      local += block_partials[block];
    values[threadIdx.x] = local;
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2u; stride; stride >>= 1u) {
      if (threadIdx.x < stride)
        values[threadIdx.x] += values[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0u)
      out_sums[0u] = static_cast<std::uint32_t>(values[0u]);
    return;
  }

  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp_in_block = threadIdx.x >> 5u;
  const std::uint32_t global_warp = blockIdx.x * kWarps + warp_in_block;
  const std::uint32_t warp_count = gridDim.x * kWarps;
  for (std::uint32_t query_base = global_warp * 32u;
       query_base < query_count; query_base += warp_count * 32u) {
    const std::uint32_t query = query_base + lane;
    const bool valid = query < query_count;
    const std::uint32_t begin = valid ? offsets[query] : 0u;
    const std::uint32_t end = valid ? offsets[query + 1u] : 0u;
    const std::uint32_t count = end - begin;
    if (valid && count <= kThreadPartials) {
      unsigned long long local = 0ull;
      for (std::uint32_t index = begin; index < end; ++index)
        local += partials[index];
      out_sums[query] = static_cast<std::uint32_t>(local);
    }

    unsigned wide = __ballot_sync(
        full_mask, valid && count > kThreadPartials);
    while (wide) {
      const std::uint32_t owner = __ffs(wide) - 1u;
      const std::uint32_t wide_query = query_base + owner;
      const std::uint32_t wide_begin =
          __shfl_sync(full_mask, begin, owner);
      const std::uint32_t wide_end = __shfl_sync(full_mask, end, owner);
      unsigned long long local = 0ull;
      for (std::uint32_t index = wide_begin + lane;
           index < wide_end; index += 32u)
        local += partials[index];
      for (std::uint32_t offset = 16u; offset; offset >>= 1u)
        local += __shfl_down_sync(full_mask, local, offset);
      if (lane == 0u)
        out_sums[wide_query] = static_cast<std::uint32_t>(local);
      wide &= ~(1u << owner);
    }
  }
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

template <class Sink>
__global__ void cooperative_section_owned_range_kernel(
    const SectionRangeFragment *fragments,
    const SectionRangeTask *tasks,
    const std::uint32_t *task_count, ResidentRows arena,
    const Descriptor *descriptors, const std::uint16_t *local_rank,
    const Row *hot_rows, const Descriptor *hot_descriptors, bool hot_ready,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches,
    typename Sink::State *aggregate_partials,
    const std::uint64_t *query_occupied_level_mask) {
  constexpr std::uint32_t kCapacity = kRangeOnChipNewerRows;
  using BlockScan = cub::BlockScan<std::uint32_t, kSectionRangeThreads>;
  union Workspace {
    Row merged[kCapacity];
    TaggedRow tagged[kCapacity];
    Row tile[kSectionRangeThreads];
  };
  __shared__ Row current[kCapacity];
  __shared__ Workspace workspace;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ Descriptor section_descriptors[kMaximumLevels];
  __shared__ std::uint32_t foundation_cell_ranks[kFoundationCells + 1u];
  __shared__ RangeFragmentBounds fragment_bounds[kSectionTaskFragments];
  __shared__ std::uint32_t fragment_work[kSectionTaskFragments];
  __shared__ typename Sink::State fragment_sums[kSectionTaskFragments];
  __shared__ std::uint32_t source_begins[kSectionTaskFragments];
  __shared__ std::uint32_t source_ends[kSectionTaskFragments];
  __shared__ std::uint32_t union_begins[kSectionTaskFragments];
  __shared__ std::uint32_t union_ends[kSectionTaskFragments];
  __shared__ std::uint8_t fragment_slots[kSectionTaskFragments];
  __shared__ std::uint8_t fragment_widths[kSectionTaskFragments];
  __shared__ std::uint16_t wave_fragment_begins[33u];
  __shared__ std::uint32_t quotient_shared;
  __shared__ std::uint32_t fragment_begin_shared;
  __shared__ std::uint32_t fragment_end_shared;
  __shared__ std::uint32_t current_count_shared;
  __shared__ std::uint32_t pending_count_shared;
  __shared__ std::uint32_t wave_count_shared;
  __shared__ std::uint32_t union_count_shared;
  __shared__ std::uint32_t ranks_valid_shared;
  __shared__ std::uint32_t task_valid_shared;
  __shared__ Descriptor hot_descriptor_shared;

  const DeviceManifestSnapshot manifest =
      load_query_manifest(query_occupied_level_mask);
  const std::uint32_t active_levels = manifest.active_levels;
  const std::uint32_t foundation_level = manifest.foundation_level;
  const std::uint64_t occupied_levels = manifest.occupied_level_mask;

  for (std::uint32_t task_index = blockIdx.x;;
       task_index += gridDim.x) {
    if (threadIdx.x == 0u) {
      task_valid_shared = task_index < *task_count;
      if (task_valid_shared) {
        const SectionRangeTask task = tasks[task_index];
        quotient_shared = task.quotient;
        fragment_begin_shared = task.begin;
        fragment_end_shared = task.end;
      }
    }
    __syncthreads();
    if (!task_valid_shared) return;

    const std::uint32_t q = quotient_shared;
    const std::uint32_t fragment_begin = fragment_begin_shared;
    const std::uint32_t fragment_end = fragment_end_shared;
    const std::uint32_t fragment_count = fragment_end - fragment_begin;
    if (threadIdx.x < active_levels) {
      section_descriptors[threadIdx.x] =
          level_is_occupied(occupied_levels, threadIdx.x)
          ? descriptors[descriptor_index(q, threadIdx.x)] : Descriptor{};
    }
    if (threadIdx.x == 0u) {
      hot_descriptor_shared = hot_ready ? hot_descriptors[q] : Descriptor{};
      pending_count_shared = 0u;
      current_count_shared = 0u;
      for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
        const std::size_t offset =
            std::size_t{batch} * (kQuotients + 1u) + q;
        pending_count_shared += raw_offsets[offset + 1u] - raw_offsets[offset];
      }
    }
    if (threadIdx.x < 32u) {
      bool ranked = false;
      Descriptor descriptor{};
      if (local_rank && foundation_level < active_levels) {
        descriptor = descriptors[descriptor_index(q, foundation_level)];
        ranked = descriptor.count() && cell_rank_supported(descriptor.count());
      }
      if (threadIdx.x == 0u) {
        ranks_valid_shared = ranked;
        foundation_cell_ranks[kFoundationCells] = descriptor.count();
      }
      if (ranked)
        for (std::uint32_t cell = threadIdx.x; cell < kFoundationCells;
             cell += 32u)
          foundation_cell_ranks[cell] =
              local_rank[std::size_t{q} * kFoundationCells + cell];
    }
    __syncthreads();

    const bool use_hot = hot_descriptor_shared.count() != 0u;
    if (!use_hot && pending_count_shared) {
      const std::uint32_t pending_count = pending_count_shared;
      if (pending_count <= 32u) {
        if (threadIdx.x < 32u) {
          const std::uint32_t lane = threadIdx.x;
          TaggedRow item{{0u, 0u, 0u}, kInvalidAge};
          if (lane < pending_count) {
            const RawAssignment loaded = load_pending_raw_ordinal(
                raw_keys, raw_payloads, raw_offsets, batch_stride,
                pending_batches, q, lane);
            item = {raw_row(loaded),
                    raw_age(raw_position(loaded), batch_stride)};
          }
          constexpr unsigned mask = 0xffffffffu;
          for (std::uint32_t width = 2u; width <= 32u; width <<= 1u)
            for (std::uint32_t stride = width >> 1u; stride;
                 stride >>= 1u) {
              TaggedRow other{};
              other.row.key = __shfl_xor_sync(mask, item.row.key, stride);
              other.row.value = __shfl_xor_sync(mask, item.row.value, stride);
              other.row.flags = __shfl_xor_sync(mask, item.row.flags, stride);
              other.age = __shfl_xor_sync(mask, item.age, stride);
              const bool ascending = (lane & width) == 0u;
              const bool take_min = ((lane & stride) == 0u) == ascending;
              if ((take_min && tagged_less(other, item)) ||
                  (!take_min && tagged_less(item, other)))
                item = other;
            }
          const std::uint32_t next_key =
              __shfl_down_sync(mask, item.row.key, 1u);
          const std::uint32_t next_age =
              __shfl_down_sync(mask, item.age, 1u);
          const bool winner = item.age != kInvalidAge &&
              (lane == 31u || next_age == kInvalidAge ||
               item.row.key != next_key);
          const unsigned winners = __ballot_sync(mask, winner);
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
                raw_keys, raw_payloads, raw_offsets, batch_stride,
                pending_batches, q, ordinal);
            workspace.tagged[ordinal] =
                {raw_row(loaded), raw_age(raw_position(loaded), batch_stride)};
          } else {
            workspace.tagged[ordinal] = {{0u, 0u, 0u}, kInvalidAge};
          }
        }
        __syncthreads();
        for (std::uint32_t width = 2u; width <= sort_size; width <<= 1u)
          for (std::uint32_t stride = width >> 1u; stride;
               stride >>= 1u) {
            for (std::uint32_t index = threadIdx.x; index < sort_size;
                 index += blockDim.x) {
              const std::uint32_t other_index = index ^ stride;
              if (other_index > index) {
                const TaggedRow first = workspace.tagged[index];
                const TaggedRow second = workspace.tagged[other_index];
                const bool ascending = (index & width) == 0u;
                const bool swap = ascending ? tagged_less(second, first)
                                            : tagged_less(first, second);
                if (swap) {
                  workspace.tagged[index] = second;
                  workspace.tagged[other_index] = first;
                }
              }
            }
            __syncthreads();
          }
        const std::uint32_t index = threadIdx.x;
        const bool winner = index < sort_size &&
            workspace.tagged[index].age != kInvalidAge &&
            (index + 1u == sort_size ||
             workspace.tagged[index + 1u].age == kInvalidAge ||
             workspace.tagged[index].row.key !=
                 workspace.tagged[index + 1u].row.key);
        std::uint32_t destination{}, winner_count{};
        BlockScan(scan_storage).ExclusiveSum(
            std::uint32_t{winner}, destination, winner_count);
        if (winner) current[destination] = workspace.tagged[index].row;
        __syncthreads();
        if (threadIdx.x == 0u) current_count_shared = winner_count;
        __syncthreads();
      }
    }

    if (!use_hot) {
      for (std::uint32_t level = 0u; level < active_levels; ++level) {
        if (level == foundation_level ||
            !level_is_occupied(occupied_levels, level)) continue;
        const Descriptor descriptor = section_descriptors[level];
        const std::uint32_t source_count = descriptor.count();
        if (!source_count) continue;
        const ResidentRows source = arena + descriptor.offset();
        const std::uint32_t old_count = current_count_shared;
        if (!old_count) {
          for (std::uint32_t index = threadIdx.x; index < source_count;
               index += blockDim.x)
            current[index] = source[index];
          __syncthreads();
          if (threadIdx.x == 0u) current_count_shared = source_count;
          __syncthreads();
          continue;
        }
        const std::uint32_t merged_count = old_count + source_count;
        const std::uint32_t diagonal = min(threadIdx.x, merged_count);
        std::uint32_t low = diagonal > old_count ? diagonal - old_count : 0u;
        std::uint32_t high = min(diagonal, source_count);
        while (low < high) {
          const std::uint32_t source_index = (low + high) >> 1u;
          const std::uint32_t current_index = diagonal - source_index;
          if (source_index < source_count && current_index > 0u &&
              current[current_index - 1u].key >= source[source_index].key)
            low = source_index + 1u;
          else
            high = source_index;
        }
        if (threadIdx.x < merged_count) {
          const std::uint32_t source_index = low;
          const std::uint32_t current_index = diagonal - source_index;
          const bool choose_source = source_index < source_count &&
              (current_index >= old_count ||
               source[source_index].key <= current[current_index].key);
          workspace.merged[threadIdx.x] = choose_source
              ? source[source_index] : current[current_index];
        }
        __syncthreads();
        const std::uint32_t index = threadIdx.x;
        const bool winner = index < merged_count &&
            (index + 1u == merged_count ||
             workspace.merged[index].key != workspace.merged[index + 1u].key);
        std::uint32_t destination{}, winner_count{};
        BlockScan(scan_storage).ExclusiveSum(
            std::uint32_t{winner}, destination, winner_count);
        if (winner) current[destination] = workspace.merged[index];
        __syncthreads();
        if (threadIdx.x == 0u) current_count_shared = winner_count;
        __syncthreads();
      }
    }

    const Row *newer = use_hot
        ? hot_rows + hot_descriptor_shared.offset() : current;
    const std::uint32_t newer_count = use_hot
        ? hot_descriptor_shared.count() : current_count_shared;
    const Descriptor foundation = foundation_level < active_levels
        ? section_descriptors[foundation_level] : Descriptor{};

    for (std::uint32_t local = threadIdx.x; local < fragment_count;
         local += blockDim.x) {
      const SectionRangeFragment fragment = fragments[fragment_begin + local];
      RangeFragmentBounds bounds{};
      bounds.update_begin = lower_bound_rows(
          newer, newer_count, fragment.low_suffix);
      bounds.update_end = upper_bound_rows(
          newer, newer_count, fragment.high_suffix);
      std::uint32_t base_work = 0u;
      if (foundation.count()) {
        const Descriptor descriptor = foundation;
        const ResidentRows rows = arena + descriptor.offset();
        if (ranks_valid_shared) {
          const std::uint32_t low_cell =
              std::uint32_t{fragment.low_suffix} / kFoundationCellKeys;
          const std::uint32_t low_begin = foundation_cell_ranks[low_cell];
          const std::uint32_t low_end = foundation_cell_ranks[low_cell + 1u];
          bounds.base_begin = low_begin + lower_bound_rows(
              rows + low_begin, low_end - low_begin, fragment.low_suffix);
          const std::uint32_t high_cell =
              std::uint32_t{fragment.high_suffix} / kFoundationCellKeys;
          const std::uint32_t high_begin = foundation_cell_ranks[high_cell];
          const std::uint32_t high_end = foundation_cell_ranks[high_cell + 1u];
          bounds.base_end = high_begin + upper_bound_rows(
              rows + high_begin, high_end - high_begin, fragment.high_suffix);
        } else {
          bounds.base_begin = lower_bound_rows(
              rows, descriptor.count(), fragment.low_suffix);
          bounds.base_end = upper_bound_rows(
              rows, descriptor.count(), fragment.high_suffix);
        }
        base_work = bounds.base_end - bounds.base_begin;
      }
      fragment_bounds[local] = bounds;
      fragment_work[local] = bounds.update_end - bounds.update_begin + base_work;
      fragment_sums[local] = Sink::identity();
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
      std::uint32_t wave = 0u, cursor = 0u;
      wave_fragment_begins[0u] = 0u;
      for (std::uint32_t fragment = 0u; fragment < fragment_count;
           ++fragment) {
        const std::uint32_t work = fragment_work[fragment];
        const std::uint32_t width = work <= kRangeThreadWork ? 1u
            : work <= kRangeSubgroupWork ? 8u : 32u;
        std::uint32_t slot = (cursor + width - 1u) & ~(width - 1u);
        if ((slot >> 5u) != ((slot + width - 1u) >> 5u))
          slot = (slot + 31u) & ~31u;
        if (slot + width > kSectionRangeThreads) {
          ++wave;
          wave_fragment_begins[wave] = fragment;
          slot = 0u;
        }
        fragment_slots[fragment] = static_cast<std::uint8_t>(slot);
        fragment_widths[fragment] = static_cast<std::uint8_t>(width);
        cursor = slot + width;
      }
      wave_count_shared = wave + 1u;
      wave_fragment_begins[wave_count_shared] = fragment_count;
    }
    __syncthreads();

    for (std::uint32_t local = threadIdx.x; local < fragment_count;
         local += blockDim.x) {
      source_begins[local] = fragment_bounds[local].update_begin;
      source_ends[local] = fragment_bounds[local].update_end;
    }
    __syncthreads();
    enumerate_range_source_tiles<Sink>(
        fragment_count,
        RangeTileSource{newer, {}, true}, newer, newer_count, false,
        source_begins, source_ends, fragment_slots, fragment_widths,
        wave_fragment_begins, wave_count_shared, union_begins, union_ends,
        &union_count_shared, workspace.tile, fragment_sums);

    if (foundation.count()) {
      const ResidentRows rows = arena + foundation.offset();
      for (std::uint32_t local = threadIdx.x; local < fragment_count;
           local += blockDim.x) {
        source_begins[local] = fragment_bounds[local].base_begin;
        source_ends[local] = fragment_bounds[local].base_end;
      }
      __syncthreads();
      enumerate_range_source_tiles<Sink>(
          fragment_count,
          RangeTileSource{nullptr, rows, false}, newer, newer_count, true,
          source_begins, source_ends, fragment_slots, fragment_widths,
          wave_fragment_begins, wave_count_shared, union_begins, union_ends,
          &union_count_shared, workspace.tile, fragment_sums);
    }

    for (std::uint32_t local = threadIdx.x; local < fragment_count;
         local += blockDim.x) {
      const SectionRangeFragment fragment = fragments[fragment_begin + local];
      aggregate_partials[fragment.original_index] = fragment_sums[local];
    }
    __syncthreads();
  }
}

template <class Sink>
__global__ void warp_range_fragment_kernel(
    const RangeFragment *fragments, std::uint32_t fragment_count,
    const std::uint32_t *device_fragment_count,
    const std::uint32_t *query_low, const std::uint32_t *query_high,
    ResidentRows arena,
    const Descriptor *descriptors, const Row *hot_rows,
    const Descriptor *hot_descriptors, bool hot_ready,
    const std::uint32_t *raw_keys,
    const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches,
    typename Sink::State *aggregate_partials,
    const std::uint64_t *query_occupied_level_mask) {
  constexpr std::uint32_t kWarps = 4u;
  constexpr std::uint32_t kUpdateCapacity = 128u;
  union WarpScratch {
    Row merged[kUpdateCapacity];
    TaggedRow tagged[kUpdateCapacity];
  };
  __shared__ Row current_shared[kWarps][kUpdateCapacity];
  __shared__ WarpScratch scratch[kWarps];
  const DeviceManifestSnapshot manifest =
      load_query_manifest(query_occupied_level_mask);
  const std::uint32_t active_levels = manifest.active_levels;
  const std::uint32_t foundation_level = manifest.foundation_level;
  const std::uint64_t occupied_levels = manifest.occupied_level_mask;
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

  unsigned long long hot_bits = 0u;
  if (lane == 0u && hot_ready) hot_bits = hot_descriptors[q].bits;
  hot_bits = __shfl_sync(full_mask, hot_bits, 0u);
  const Descriptor hot_descriptor{hot_bits};
  if (hot_descriptor.count()) {
    const Descriptor foundation = foundation_level < active_levels
        ? descriptors[descriptor_index(q, foundation_level)] : Descriptor{};
    unsigned long long value = enumerate_visible_section<Sink>(
        low_suffix, high_suffix, hot_rows + hot_descriptor.offset(),
        hot_descriptor.count(), arena, foundation,
        lane, 32u);
    for (std::uint32_t offset = 16u; offset; offset >>= 1u)
      value = Sink{}(value,
          __shfl_down_sync(full_mask, value, offset));
    if (lane == 0u) aggregate_partials[fragment_index] = value;
    return;
  }

  std::uint32_t current_count = 0u;
  if (pending_batches) {
    std::uint32_t pending_count = 0u;
    for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
      const std::size_t oi = std::size_t{batch} * (kQuotients + 1u) + q;
      const std::uint32_t begin = raw_offsets[oi];
      const std::uint32_t end = raw_offsets[oi + 1u];
      for (std::uint32_t chunk = begin; chunk < end; chunk += 32u) {
        const std::uint32_t position = chunk + lane;
        RawAssignment item{};
        const std::uint32_t record = batch * batch_stride + position;
        const std::uint32_t item_key =
            position < end ? raw_keys[record] : 0u;
        const bool valid = position < end &&
            key_suffix(item_key) >= low_suffix &&
            key_suffix(item_key) <= high_suffix;
        if (valid)
          item = load_raw_assignment(raw_keys, raw_payloads, record);
        const unsigned selected = __ballot_sync(full_mask, valid);
        const std::uint32_t destination =
            pending_count + __popc(selected & before);
        if (valid && destination < kUpdateCapacity)
          scratch[warp].tagged[destination] =
              {raw_row(item), raw_age(raw_position(item), batch_stride)};
        pending_count += __popc(selected);
      }
    }
    const bool pending_overflow = pending_count > kUpdateCapacity;
    if (__shfl_sync(full_mask, pending_overflow, 0u)) {
      asm volatile("trap;");
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
    if (level == foundation_level ||
        !level_is_occupied(occupied_levels, level)) continue;
    unsigned long long descriptor_bits = 0ull;
    if (lane == 0u)
      descriptor_bits = descriptors[descriptor_index(q, level)].bits;
    descriptor_bits = __shfl_sync(full_mask, descriptor_bits, 0u);
    const Descriptor descriptor{descriptor_bits};
    const ResidentRows rows = arena + descriptor.offset();
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
    const ResidentRows older = rows + older_begin;
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
    asm volatile("trap;");
    return;
  }

  const Descriptor foundation = foundation_level < active_levels
      ? descriptors[descriptor_index(q, foundation_level)] : Descriptor{};
  std::uint32_t work = 0u;
  if (lane == 0u)
    work = current_count + section_range_row_count(
        low_suffix, high_suffix, arena, foundation);
  work = __shfl_sync(full_mask, work, 0u);
  const std::uint32_t worker_width = work <= kRangeThreadWork ? 1u
      : work <= kRangeSubgroupWork ? 8u : 32u;
  const unsigned worker_mask = __ballot_sync(full_mask, lane < worker_width);
  typename Sink::State local = Sink::identity();
  if (lane < worker_width) {
    local = enumerate_visible_section<Sink>(
        low_suffix, high_suffix, current, current_count, arena,
        foundation, lane, worker_width);
    for (std::uint32_t offset = worker_width / 2u; offset; offset >>= 1u)
      local += __shfl_down_sync(worker_mask, local, offset, worker_width);
  }
  if (lane == 0u) {
    aggregate_partials[fragment_index] = local;
  }
}

}  // namespace gpulsmopt2_detail
