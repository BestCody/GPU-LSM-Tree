#pragma once

#include "range_common.cuh"

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>

namespace gpulsm_sparse::rank_range {

using namespace range_gate;

constexpr std::uint32_t kTileRows = 1024u;
constexpr std::uint32_t kMaximumSources = kLevels + 1u;
constexpr std::uint32_t kResidentOrdinarySpan = 0u;
constexpr std::uint32_t kResidentExactSpan = 1u;
constexpr std::uint32_t kPendingSpan = 2u;
constexpr std::uint32_t kTaskFinal = 0u;
constexpr std::uint32_t kTaskDirect = 1u;
constexpr std::uint32_t kTaskSplitLower = 2u;
constexpr std::uint32_t kTaskSplitUpper = 3u;

struct OrderSpan {
  std::uint64_t physical_begin{};
  std::uint64_t logical_begin{};
  std::uint32_t count{};
  std::uint32_t fixed_head{};
  std::uint32_t kind{};
  std::uint32_t reserved{};
};

struct OrderSource {
  std::uint32_t span_begin{};
  std::uint32_t span_count{};
  std::uint32_t row_count{};
  std::uint32_t recency{};
  std::uint32_t level{};
  std::uint32_t reserved{};
};

struct RankTask {
  std::uint32_t component{};
  std::uint32_t candidate_count{};
  std::uint32_t mode{};
  std::uint32_t active_source{};
};

struct RankView {
  gpulsmopt2_detail::ResidentRows arena{};
  const RootBuildState *roots{};
  PendingReadView pending{};
  const std::uint32_t *pending_refs{};
  const OrderSource *sources{};
  const OrderSpan *spans{};
  std::uint32_t source_count{};
};

struct EndpointSource {
  RangeQueryBatch queries{};
};

__device__ inline DeviceKeyCursor source_key(
    EndpointSource source, std::uint32_t ref) {
  const bool upper = ref >= source.queries.count;
  const std::uint32_t query = upper ? ref - source.queries.count : ref;
  const RecordBatchView batch = upper
      ? source.queries.upper : source.queries.lower;
  return {batch.keys, query, batch.head4_words};
}

struct EndpointValue {
  __device__ std::uint64_t length() const { return 0u; }
  __device__ std::uint8_t byte(std::uint64_t) const { return 0u; }
  __device__ std::uint32_t inline_word() const { return 0u; }
};

__device__ inline EndpointValue source_value(
    EndpointSource, std::uint32_t) {
  return {};
}

__device__ inline bool source_tombstone(
    EndpointSource, std::uint32_t) {
  return false;
}

__device__ inline std::uint32_t source_summary(
    EndpointSource, std::uint32_t) {
  return 0u;
}

struct LocatedRecord {
  RankView view{};
  std::uint32_t source{};
  std::uint32_t logical{};
  OrderSpan span{};
  std::uint32_t local{};

  __device__ bool pending_record() const {
    return span.kind == kPendingSpan;
  }

  __device__ std::uint32_t pending_ref() const {
    return view.pending_refs[span.physical_begin + local];
  }

  __device__ std::uint64_t physical() const {
    return span.physical_begin + local;
  }

  __device__ PendingRecordCursor pending_cursor() const {
    return pending_from_encoded(view.pending, pending_ref());
  }

  __device__ DeviceCapsulePlane plane() const {
    if (pending_record()) return {};
    const RootBuildState &root = view.roots[view.sources[source].level];
    return {reinterpret_cast<const CapsulePage *>(root.capsule_pages),
            reinterpret_cast<const CapsuleIndex *>(root.capsule_indexes),
            root.capsule_page_count, root.capsule_count};
  }

  __device__ gpulsmopt2_detail::Row row() const {
    if (span.kind != kResidentExactSpan) return view.arena[physical()];
    const RootBuildState &root = view.roots[view.sources[source].level];
    const auto *rows = reinterpret_cast<const gpulsmopt2_detail::Row *>(
        root.exact_rows);
    return rows[exact_physical_offset(physical())];
  }

  __device__ ResidentKeyCursor resident_cursor() const {
    const auto stored = row();
    const std::uint32_t head = span.kind == kResidentExactSpan
        ? span.fixed_head : span.fixed_head | stored.key;
    return {head, physical(), stored, plane()};
  }
};

__device__ inline OrderSpan locate_span(
    RankView view, std::uint32_t source, std::uint32_t logical) {
  const OrderSource entry = view.sources[source];
  std::uint32_t low = 0u;
  std::uint32_t high = entry.span_count;
  while (low + 1u < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (view.spans[entry.span_begin + middle].logical_begin <= logical)
      low = middle;
    else
      high = middle;
  }
  return view.spans[entry.span_begin + low];
}

__device__ inline LocatedRecord locate_record(
    RankView view, std::uint32_t source, std::uint32_t logical) {
  const OrderSpan span = locate_span(view, source, logical);
  return {view, source, logical, span,
          logical - static_cast<std::uint32_t>(span.logical_begin)};
}

struct RankKeyCursor {
  RankView view{};
  std::uint32_t source{};
  std::uint32_t logical{};

  __device__ LocatedRecord record() const {
    return locate_record(view, source, logical);
  }

  __device__ std::uint32_t head4() const {
    const auto found = record();
    return found.pending_record() ? found.pending_cursor().head4()
                                  : found.resident_cursor().head4();
  }

  __device__ std::uint64_t length() const {
    const auto found = record();
    return found.pending_record() ? found.pending_cursor().length()
                                  : found.resident_cursor().length();
  }

  __device__ std::uint8_t byte(std::uint64_t position) const {
    const auto found = record();
    return found.pending_record() ? found.pending_cursor().byte(position)
                                  : found.resident_cursor().byte(position);
  }
};

struct SpanKeyCursor {
  RankView view{};
  std::uint32_t source{};
  OrderSpan span{};
  std::uint32_t local{};

  __device__ LocatedRecord record() const {
    return {view, source,
            static_cast<std::uint32_t>(span.logical_begin) + local,
            span, local};
  }

  __device__ std::uint32_t head4() const {
    const auto found = record();
    return found.pending_record() ? found.pending_cursor().head4()
                                  : found.resident_cursor().head4();
  }

  __device__ std::uint64_t length() const {
    const auto found = record();
    return found.pending_record() ? found.pending_cursor().length()
                                  : found.resident_cursor().length();
  }

  __device__ std::uint8_t byte(std::uint64_t position) const {
    const auto found = record();
    return found.pending_record() ? found.pending_cursor().byte(position)
                                  : found.resident_cursor().byte(position);
  }
};

__device__ inline std::uint32_t source_span_index(
    RankView view, std::uint32_t source, std::uint32_t logical) {
  const OrderSource entry = view.sources[source];
  std::uint32_t low = 0u;
  std::uint32_t high = entry.span_count;
  while (low + 1u < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (view.spans[entry.span_begin + middle].logical_begin <= logical)
      low = middle;
    else
      high = middle;
  }
  return entry.span_begin + low;
}

__device__ inline SpanKeyCursor direct_key(
    RankView view, std::uint32_t source, std::uint32_t logical) {
  const std::uint32_t span_index = source_span_index(view, source, logical);
  const OrderSpan span = view.spans[span_index];
  return {view, source, span,
          logical - static_cast<std::uint32_t>(span.logical_begin)};
}

struct RankValueCursor {
  LocatedRecord record{};

  __device__ const CapsuleHeader *header() const {
    if (record.pending_record()) return record.pending_cursor().header();
    const auto stored = record.row();
    return capsule_header(anchor_has_capsule(stored)
        ? find_capsule(record.plane(), record.physical(),
                       anchor_capsule_rank(stored))
        : nullptr);
  }

  __device__ bool tombstone() const {
    return record.pending_record()
        ? (record.pending_cursor().payload().metadata &
           gpulsmopt2_detail::kRawTombstone) != 0u
        : (record.row().flags & gpulsmopt2_detail::kTombstone) != 0u;
  }

  __device__ std::uint64_t length() const {
    if (tombstone()) return 0u;
    const CapsuleHeader *object = header();
    if (object) return object->value_length;
    return record.pending_record() ? 4u : anchor_value_length(record.row());
  }

  __device__ std::uint8_t byte(std::uint64_t position) const {
    const CapsuleHeader *object = header();
    if (object) {
      const auto *bytes = reinterpret_cast<const std::uint8_t *>(object + 1u);
      return bytes[object->key_length + position];
    }
    const std::uint32_t word = record.pending_record()
        ? record.pending_cursor().payload().value : record.row().value;
    return static_cast<std::uint8_t>(word >> (8u * position));
  }

  __device__ std::uint32_t inline_word() const {
    return record.pending_record()
        ? record.pending_cursor().payload().value : record.row().value;
  }
};

__device__ inline RankValueCursor rank_value(
    RankView view, std::uint32_t source, std::uint32_t logical) {
  return {locate_record(view, source, logical)};
}

__device__ inline int compare_source_keys(
    RankView view, std::uint32_t left_source, std::uint32_t left_row,
    std::uint32_t right_source, std::uint32_t right_row) {
  return compare_keys(RankKeyCursor{view, left_source, left_row},
                      RankKeyCursor{view, right_source, right_row});
}

__device__ inline bool same_source_key(
    RankView view, std::uint32_t left_source, std::uint32_t left_row,
    std::uint32_t right_source, std::uint32_t right_row) {
  const RankKeyCursor left{view, left_source, left_row};
  const RankKeyCursor right{view, right_source, right_row};
  if (left.head4() != right.head4() || left.length() != right.length())
    return false;
  return left.length() == 4u || compare_keys(left, right, 4u) == 0;
}

__device__ inline std::uint32_t lower_bound_query(
    RankView view, std::uint32_t source, std::uint32_t begin,
    std::uint32_t end, const DeviceKeyCursor &query, bool strict) {
  if (begin >= end) return begin;
  const int first_comparison = compare_keys(
      direct_key(view, source, begin), query);
  if (first_comparison > 0 || (!strict && first_comparison == 0))
    return begin;
  const int last_comparison = compare_keys(
      direct_key(view, source, end - 1u), query);
  if (last_comparison < 0 || (strict && last_comparison == 0))
    return end;
  const OrderSource entry = view.sources[source];
  std::uint32_t span_low = source_span_index(view, source, begin);
  std::uint32_t span_high = source_span_index(view, source, end - 1u) + 1u;
  while (span_low < span_high) {
    const std::uint32_t middle = span_low + ((span_high - span_low) >> 1u);
    const OrderSpan span = view.spans[middle];
    const std::uint32_t last = span.count - 1u;
    const int comparison = compare_keys(
        SpanKeyCursor{view, source, span, last}, query);
    if (comparison < 0 || (strict && comparison == 0)) span_low = middle + 1u;
    else span_high = middle;
  }
  if (span_low >= entry.span_begin + entry.span_count) return end;
  const OrderSpan span = view.spans[span_low];
  std::uint32_t low = max(
      begin, static_cast<std::uint32_t>(span.logical_begin));
  std::uint32_t high = min(
      end, static_cast<std::uint32_t>(span.logical_begin) + span.count);
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    const int comparison = compare_keys(
        SpanKeyCursor{view, source, span,
                      middle - static_cast<std::uint32_t>(span.logical_begin)},
        query);
    if (comparison < 0 || (strict && comparison == 0)) low = middle + 1u;
    else high = middle;
  }
  return low;
}

__device__ inline std::uint32_t lower_bound_pivot(
    RankView view, std::uint32_t source, std::uint32_t begin,
    std::uint32_t end, std::uint32_t pivot_source,
    std::uint32_t pivot_row, bool strict) {
  if (begin >= end) return begin;
  const SpanKeyCursor pivot = direct_key(view, pivot_source, pivot_row);
  const OrderSource entry = view.sources[source];
  std::uint32_t span_low = source_span_index(view, source, begin);
  std::uint32_t span_high = source_span_index(view, source, end - 1u) + 1u;
  while (span_low < span_high) {
    const std::uint32_t middle = span_low + ((span_high - span_low) >> 1u);
    const OrderSpan span = view.spans[middle];
    const int comparison = compare_keys(
        SpanKeyCursor{view, source, span, span.count - 1u}, pivot);
    if (comparison < 0 || (strict && comparison == 0)) span_low = middle + 1u;
    else span_high = middle;
  }
  if (span_low >= entry.span_begin + entry.span_count) return end;
  const OrderSpan span = view.spans[span_low];
  std::uint32_t low = max(
      begin, static_cast<std::uint32_t>(span.logical_begin));
  std::uint32_t high = min(
      end, static_cast<std::uint32_t>(span.logical_begin) + span.count);
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    const int comparison = compare_keys(
        SpanKeyCursor{view, source, span,
                      middle - static_cast<std::uint32_t>(span.logical_begin)},
        pivot);
    if (comparison < 0 || (strict && comparison == 0)) low = middle + 1u;
    else high = middle;
  }
  return low;
}

__global__ void count_root_spans(
    const std::uint32_t *sections, std::uint32_t section_count,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const RootBuildState *roots, const std::uint32_t *levels,
    std::uint32_t root_count, std::uint32_t *span_counts,
    std::uint32_t *row_counts) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t entries = root_count * section_count;
  if (index >= entries) return;
  const std::uint32_t root_slot = index / section_count;
  const std::uint32_t section = sections[index % section_count];
  const std::uint32_t level = levels[root_slot];
  const RootBuildState root = roots[level];
  const auto ordinary = descriptors[
      gpulsmopt2_detail::descriptor_index(section, level)];
  std::uint32_t exact_begin = 0u;
  std::uint32_t exact_end = 0u;
  exact_head_bounds(root, section, exact_begin, exact_end);
  const auto *heads = reinterpret_cast<const ExactHeadDescriptor *>(
      root.exact_heads);
  std::uint32_t ordinary_position = 0u;
  std::uint32_t spans = 0u;
  std::uint32_t rows = 0u;
  for (std::uint32_t exact = exact_begin; exact < exact_end; ++exact) {
    const std::uint32_t position = lower_bound_ordinary_head(
        arena, ordinary, static_cast<std::uint16_t>(heads[exact].head));
    if (position > ordinary_position) ++spans;
    if (heads[exact].count) ++spans;
    rows += position - ordinary_position + heads[exact].count;
    // The specialized projection retains one sentinel row for this exact
    // head.  The logical range cursor expands the sidecar in its place.
    ordinary_position = position + 1u;
  }
  if (ordinary_position < ordinary.count()) ++spans;
  rows += ordinary.count() - ordinary_position;
  span_counts[index] = spans;
  row_counts[index] = rows;
}

__global__ void fill_root_spans(
    const std::uint32_t *sections, std::uint32_t section_count,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const RootBuildState *roots, const std::uint32_t *levels,
    std::uint32_t root_count, const std::uint32_t *span_offsets,
    const std::uint32_t *row_offsets, OrderSpan *spans,
    unsigned long long *errors) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t entries = root_count * section_count;
  if (index >= entries) return;
  const std::uint32_t root_slot = index / section_count;
  const std::uint32_t section = sections[index % section_count];
  const std::uint32_t level = levels[root_slot];
  const RootBuildState root = roots[level];
  const auto ordinary = descriptors[
      gpulsmopt2_detail::descriptor_index(section, level)];
  std::uint32_t exact_begin = 0u;
  std::uint32_t exact_end = 0u;
  exact_head_bounds(root, section, exact_begin, exact_end);
  const auto *heads = reinterpret_cast<const ExactHeadDescriptor *>(
      root.exact_heads);
  std::uint32_t ordinary_position = 0u;
  std::uint64_t logical = row_offsets[index];
  std::uint32_t output = span_offsets[index];
  for (std::uint32_t exact = exact_begin; exact < exact_end; ++exact) {
    const std::uint32_t position = lower_bound_ordinary_head(
        arena, ordinary, static_cast<std::uint16_t>(heads[exact].head));
    if (position > ordinary_position) {
      const std::uint32_t count = position - ordinary_position;
      spans[output++] = {ordinary.offset() + ordinary_position, logical,
                         count, section << 16u,
                         kResidentOrdinarySpan, 0u};
      logical += count;
    }
    if (heads[exact].count) {
      spans[output++] = {heads[exact].physical_begin, logical,
                         heads[exact].count, heads[exact].head,
                         kResidentExactSpan, 0u};
      logical += heads[exact].count;
    }
    ordinary_position = position + 1u;
  }
  if (ordinary_position < ordinary.count()) {
    const std::uint32_t count = ordinary.count() - ordinary_position;
    spans[output++] = {ordinary.offset() + ordinary_position, logical,
                       count, section << 16u,
                       kResidentOrdinarySpan, 0u};
  }
  const std::uint32_t expected = index + 1u < entries
      ? span_offsets[index + 1u] : output;
  if (index + 1u < entries && output != expected)
    atomicAdd(errors, 1ull);
}

__global__ void finish_roster_sources(
    std::uint32_t section_count, const std::uint32_t *span_counts,
    const std::uint32_t *span_offsets, const std::uint32_t *row_counts,
    const std::uint32_t *row_offsets, const std::uint32_t *levels,
    std::uint32_t root_count, std::uint32_t root_span_total,
    std::uint32_t pending_rows, OrderSource *sources, OrderSpan *spans) {
  const std::uint32_t source = blockIdx.x * blockDim.x + threadIdx.x;
  if (source < root_count) {
    const std::uint32_t first = source * section_count;
    const std::uint32_t last = first + section_count - 1u;
    const std::uint32_t span_begin = span_offsets[first];
    const std::uint32_t span_end = span_offsets[last] + span_counts[last];
    const std::uint32_t row_count = row_offsets[last] + row_counts[last];
    sources[source] = {span_begin, span_end - span_begin, row_count,
                       kLevels - levels[source], levels[source], 0u};
  }
  if (source == root_count && pending_rows) {
    spans[root_span_total] = {0u, 0u, pending_rows, 0u, kPendingSpan, 0u};
    sources[source] = {root_span_total, 1u, pending_rows,
                       std::numeric_limits<std::uint32_t>::max(),
                       kLevels, 0u};
  }
}

__global__ void append_pending_source(
    std::uint32_t root_count, std::uint32_t root_span_total,
    std::uint32_t pending_rows, OrderSource *sources, OrderSpan *spans) {
  if (blockIdx.x || threadIdx.x || !pending_rows) return;
  spans[root_span_total] = {
      0u, 0u, pending_rows, 0u, kPendingSpan, 0u};
  sources[root_count] = {
      root_span_total, 1u, pending_rows,
      std::numeric_limits<std::uint32_t>::max(), kLevels, 0u};
}

__global__ void initialize_rank_tasks(
    RangeQueryBatch queries, const RangeComponent *components,
    std::uint32_t component_count, RankView view, RankTask *tasks,
    std::uint32_t *begins, std::uint32_t *ends) {
  const std::uint32_t component = blockIdx.x;
  if (component >= component_count) return;
  const RangeComponent interval = components[component];
  const DeviceKeyCursor lower{
      queries.lower.keys, interval.lower_ref, queries.lower.head4_words};
  const DeviceKeyCursor upper{
      queries.upper.keys, interval.upper_ref, queries.upper.head4_words};
  __shared__ std::uint32_t total;
  if (threadIdx.x == 0u) total = 0u;
  __syncthreads();
  for (std::uint32_t source = threadIdx.x; source < view.source_count;
       source += blockDim.x) {
    const std::uint32_t count = view.sources[source].row_count;
    const std::uint32_t begin = lower_bound_query(
        view, source, 0u, count, lower, false);
    const std::uint32_t end = lower_bound_query(
        view, source, begin, count, upper, true);
    const std::uint64_t position =
        std::uint64_t{component} * view.source_count + source;
    begins[position] = begin;
    ends[position] = end;
    if (end > begin) atomicAdd(&total, end - begin);
  }
  __syncthreads();
  if (threadIdx.x == 0u)
    tasks[component] = {component, total, kTaskFinal, 0u};
}

__global__ void classify_rank_tasks(
    const RankTask *tasks, const std::uint32_t *task_count,
    const std::uint32_t *begins, const std::uint32_t *ends,
    RankView view, std::uint32_t *child_counts,
    std::uint32_t *split_lowers, std::uint32_t *split_uppers,
    RankTask *classified, std::uint32_t *oversized_count) {
  __shared__ std::uint32_t order[kMaximumSources];
  __shared__ std::uint32_t medians[kMaximumSources];
  __shared__ std::uint32_t active_count;
  __shared__ std::uint32_t active_source;
  __shared__ std::uint32_t pivot_source;
  __shared__ std::uint32_t pivot_row;
  __shared__ std::uint32_t lower_total;
  __shared__ std::uint32_t upper_total;
  const std::uint32_t task = blockIdx.x;
  if (task >= *task_count) return;
  RankTask meta = tasks[task];
  const std::uint64_t base = std::uint64_t{task} * view.source_count;
  if (threadIdx.x == 0u) {
    active_count = 0u;
    active_source = 0u;
    lower_total = 0u;
    upper_total = 0u;
    for (std::uint32_t source = 0u; source < view.source_count; ++source) {
      const std::uint32_t count = ends[base + source] - begins[base + source];
      if (!count) continue;
      active_source = source;
      order[active_count] = source;
      medians[active_count] = begins[base + source] + count / 2u;
      ++active_count;
    }
    meta.active_source = active_source;
    if (meta.candidate_count <= kTileRows) {
      meta.mode = kTaskFinal;
      child_counts[task] = 1u;
    } else if (active_count == 1u) {
      meta.mode = kTaskDirect;
      child_counts[task] =
          (meta.candidate_count + kTileRows - 1u) / kTileRows;
      atomicAdd(oversized_count, 1u);
    } else {
      for (std::uint32_t index = 1u; index < active_count; ++index) {
        const std::uint32_t source = order[index];
        const std::uint32_t row = medians[index];
        std::uint32_t position = index;
        while (position && compare_keys(
                   direct_key(view, source, row),
                   direct_key(view, order[position - 1u],
                              medians[position - 1u])) < 0) {
          order[position] = order[position - 1u];
          medians[position] = medians[position - 1u];
          --position;
        }
        order[position] = source;
        medians[position] = row;
      }
      std::uint64_t prefix = 0u;
      const std::uint64_t half =
          (static_cast<std::uint64_t>(meta.candidate_count) + 1u) / 2u;
      pivot_source = order[active_count - 1u];
      pivot_row = medians[active_count - 1u];
      for (std::uint32_t index = 0u; index < active_count; ++index) {
        const std::uint32_t source = order[index];
        prefix += ends[base + source] - begins[base + source];
        if (prefix >= half) {
          pivot_source = source;
          pivot_row = medians[index];
          break;
        }
      }
      child_counts[task] = 2u;
      atomicAdd(oversized_count, 1u);
    }
    classified[task] = meta;
  }
  __syncthreads();
  if (meta.candidate_count <= kTileRows || active_count == 1u) return;
  for (std::uint32_t source = threadIdx.x; source < view.source_count;
       source += blockDim.x) {
    const std::uint32_t begin = begins[base + source];
    const std::uint32_t end = ends[base + source];
    const std::uint32_t low = lower_bound_pivot(
        view, source, begin, end, pivot_source, pivot_row, false);
    const std::uint32_t high = lower_bound_pivot(
        view, source, low, end, pivot_source, pivot_row, true);
    split_lowers[base + source] = low;
    split_uppers[base + source] = high;
    atomicAdd(&lower_total, low - begin);
    atomicAdd(&upper_total, high - begin);
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    const std::uint32_t lower_distance = lower_total > meta.candidate_count / 2u
        ? lower_total - meta.candidate_count / 2u
        : meta.candidate_count / 2u - lower_total;
    const std::uint32_t upper_distance = upper_total > meta.candidate_count / 2u
        ? upper_total - meta.candidate_count / 2u
        : meta.candidate_count / 2u - upper_total;
    classified[task].mode = upper_distance < lower_distance
        ? kTaskSplitUpper : kTaskSplitLower;
  }
}

__global__ void emit_rank_children(
    const RankTask *tasks, const RankTask *classified,
    const std::uint32_t *task_count, const std::uint32_t *child_offsets,
    const std::uint32_t *begins, const std::uint32_t *ends,
    const std::uint32_t *split_lowers, const std::uint32_t *split_uppers,
    std::uint32_t source_count, RankTask *children,
    std::uint32_t *child_begins, std::uint32_t *child_ends) {
  const std::uint32_t task = blockIdx.x;
  if (task >= *task_count) return;
  const RankTask input = tasks[task];
  const RankTask plan = classified[task];
  const std::uint32_t output = child_offsets[task];
  const std::uint64_t input_base = std::uint64_t{task} * source_count;
  if (plan.mode == kTaskFinal) {
    if (threadIdx.x == 0u) children[output] = input;
    for (std::uint32_t source = threadIdx.x; source < source_count;
         source += blockDim.x) {
      const std::uint64_t destination =
          std::uint64_t{output} * source_count + source;
      child_begins[destination] = begins[input_base + source];
      child_ends[destination] = ends[input_base + source];
    }
    return;
  }
  if (plan.mode == kTaskDirect) {
    const std::uint32_t source = plan.active_source;
    const std::uint32_t first = begins[input_base + source];
    const std::uint32_t count = ends[input_base + source] - first;
    const std::uint32_t pieces = (count + kTileRows - 1u) / kTileRows;
    for (std::uint32_t piece = threadIdx.x; piece < pieces;
         piece += blockDim.x) {
      const std::uint32_t child = output + piece;
      const std::uint32_t begin = first + piece * kTileRows;
      const std::uint32_t end = min(first + count, begin + kTileRows);
      children[child] = {input.component, end - begin, kTaskFinal, source};
    }
    for (std::uint32_t flat = threadIdx.x;
         flat < pieces * source_count; flat += blockDim.x) {
      const std::uint32_t piece = flat / source_count;
      const std::uint32_t slot = flat % source_count;
      const std::uint32_t child = output + piece;
      const std::uint64_t destination =
          std::uint64_t{child} * source_count + slot;
      if (slot == source) {
        const std::uint32_t begin = first + piece * kTileRows;
        child_begins[destination] = begin;
        child_ends[destination] = min(first + count, begin + kTileRows);
      } else {
        child_begins[destination] = begins[input_base + slot];
        child_ends[destination] = begins[input_base + slot];
      }
    }
    return;
  }
  const bool upper = plan.mode == kTaskSplitUpper;
  std::uint32_t left_count = 0u;
  std::uint32_t right_count = 0u;
  for (std::uint32_t source = threadIdx.x; source < source_count;
       source += blockDim.x) {
    const std::uint32_t begin = begins[input_base + source];
    const std::uint32_t end = ends[input_base + source];
    const std::uint32_t cut = upper ? split_uppers[input_base + source]
                                    : split_lowers[input_base + source];
    const std::uint64_t left = std::uint64_t{output} * source_count + source;
    const std::uint64_t right = left + source_count;
    child_begins[left] = begin;
    child_ends[left] = cut;
    child_begins[right] = cut;
    child_ends[right] = end;
    atomicAdd(reinterpret_cast<unsigned int *>(&children[output].candidate_count),
              cut - begin);
    atomicAdd(reinterpret_cast<unsigned int *>(
                  &children[output + 1u].candidate_count), end - cut);
    left_count += cut - begin;
    right_count += end - cut;
  }
  if (threadIdx.x == 0u) {
    children[output].component = input.component;
    children[output].mode = kTaskFinal;
    children[output].active_source = 0u;
    children[output + 1u].component = input.component;
    children[output + 1u].mode = kTaskFinal;
    children[output + 1u].active_source = 0u;
  }
}

__global__ void clear_rank_tasks(RankTask *tasks, std::uint32_t begin,
                                 std::uint32_t count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) tasks[begin + index] = {};
}

__device__ inline std::uint32_t token_source(std::uint64_t token) {
  return static_cast<std::uint32_t>(token >> 32u);
}

__device__ inline std::uint32_t token_row(std::uint64_t token) {
  return static_cast<std::uint32_t>(token);
}

__device__ inline bool token_less(
    RankView view, std::uint64_t left, std::uint64_t right) {
  const std::uint32_t left_source = token_source(left);
  const std::uint32_t right_source = token_source(right);
  const int comparison = compare_source_keys(
      view, left_source, token_row(left), right_source, token_row(right));
  if (comparison) return comparison < 0;
  const std::uint32_t left_age = view.sources[left_source].recency;
  const std::uint32_t right_age = view.sources[right_source].recency;
  if (left_age != right_age) return left_age > right_age;
  return left_source < right_source;
}

__device__ inline std::uint32_t merge_partition(
    RankView view, const std::uint64_t *left, std::uint32_t left_count,
    const std::uint64_t *right, std::uint32_t right_count,
    std::uint32_t diagonal) {
  std::uint32_t low = diagonal > right_count ? diagonal - right_count : 0u;
  std::uint32_t high = min(diagonal, left_count);
  while (low <= high) {
    const std::uint32_t i = (low + high) >> 1u;
    const std::uint32_t j = diagonal - i;
    if (i && j < right_count && token_less(view, right[j], left[i - 1u]))
      high = i - 1u;
    else if (j && i < left_count &&
             !token_less(view, right[j - 1u], left[i]))
      low = i + 1u;
    else
      return i;
  }
  return low;
}

__device__ inline std::uint64_t *merge_rank_task(
    RankView view, const std::uint32_t *begins,
    const std::uint32_t *ends, std::uint64_t *tokens_a,
    std::uint64_t *tokens_b, std::uint32_t *segments,
    std::uint32_t &total, unsigned long long *errors) {
  if (threadIdx.x == 0u) {
    total = 0u;
    segments[0] = 0u;
    for (std::uint32_t source = 0u; source < view.source_count; ++source) {
      total += ends[source] - begins[source];
      segments[source + 1u] = total;
    }
    if (total > kTileRows) atomicAdd(errors, 1ull);
  }
  __syncthreads();
  if (total > kTileRows) return tokens_a;
  for (std::uint32_t source = 0u; source < view.source_count; ++source) {
    const std::uint32_t count = ends[source] - begins[source];
    for (std::uint32_t local = threadIdx.x; local < count;
         local += blockDim.x)
      tokens_a[segments[source] + local] =
          (std::uint64_t{source} << 32u) | (begins[source] + local);
  }
  __syncthreads();
  std::uint32_t segment_count = view.source_count;
  bool ping = false;
  while (segment_count > 1u) {
    const std::uint64_t *input = ping ? tokens_b : tokens_a;
    std::uint64_t *output = ping ? tokens_a : tokens_b;
    const std::uint32_t pairs = segment_count / 2u;
    for (std::uint32_t pair = 0u; pair < pairs; ++pair) {
      const std::uint32_t first = segments[pair * 2u];
      const std::uint32_t middle = segments[pair * 2u + 1u];
      const std::uint32_t last = segments[pair * 2u + 2u];
      const std::uint32_t left_count = middle - first;
      const std::uint32_t right_count = last - middle;
      for (std::uint32_t diagonal = threadIdx.x;
           diagonal < left_count + right_count;
           diagonal += blockDim.x) {
        const std::uint32_t i = merge_partition(
            view, input + first, left_count, input + middle,
            right_count, diagonal);
        const std::uint32_t j = diagonal - i;
        const bool take_left = i < left_count &&
            (j >= right_count ||
             !token_less(view, input[middle + j], input[first + i]));
        output[first + diagonal] = take_left
            ? input[first + i] : input[middle + j];
      }
    }
    if (segment_count & 1u) {
      const std::uint32_t begin = segments[segment_count - 1u];
      const std::uint32_t end = segments[segment_count];
      for (std::uint32_t index = begin + threadIdx.x;
           index < end; index += blockDim.x)
        output[index] = input[index];
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
      const std::uint32_t next = (segment_count + 1u) / 2u;
      for (std::uint32_t segment = 1u; segment < next; ++segment)
        segments[segment] = segments[segment * 2u];
      segments[next] = total;
    }
    __syncthreads();
    segment_count = (segment_count + 1u) / 2u;
    ping = !ping;
  }
  return ping ? tokens_b : tokens_a;
}

__device__ inline bool token_live(
    RankView view, const std::uint64_t *tokens, std::uint32_t index) {
  const std::uint64_t token = tokens[index];
  if (index && same_source_key(
          view, token_source(tokens[index - 1u]),
          token_row(tokens[index - 1u]), token_source(token),
          token_row(token)))
    return false;
  return !rank_value(
      view, token_source(token), token_row(token)).tombstone();
}

struct OutputAnchor {
  std::uint32_t head{};
  std::uint32_t value{};
};

struct OutputCapsule {
  std::uint64_t anchor_row{};
  CapsuleIndex index{};
};

struct CopyObject {
  std::uint32_t source{};
  std::uint32_t row{};
  std::uint64_t destination{};
};

struct OutputChunk {
  std::uint64_t physical_begin{};
  std::uint64_t logical_begin{};
  std::uint64_t capsule_begin{};
  std::uint32_t count{};
  std::uint32_t capsule_count{};
  std::uint32_t component{};
  std::uint32_t task{};
};

struct OutputView {
  const OutputAnchor *anchors{};
  const OutputCapsule *capsules{};
  const OutputChunk *chunks{};
  const std::uint64_t *logical_offsets{};
  std::uint32_t chunk_count{};
};

__device__ inline std::uint32_t output_chunk(
    OutputView output, std::uint64_t logical) {
  std::uint32_t low = 0u;
  std::uint32_t high = output.chunk_count;
  while (low + 1u < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (output.logical_offsets[middle] <= logical) low = middle;
    else high = middle;
  }
  return low;
}

struct OutputKeyCursor {
  OutputView output{};
  std::uint64_t logical{};

  __device__ OutputChunk chunk() const {
    return output.chunks[output_chunk(output, logical)];
  }

  __device__ std::uint64_t physical() const {
    const OutputChunk part = chunk();
    return part.physical_begin + logical - part.logical_begin;
  }

  __device__ OutputAnchor anchor() const {
    return output.anchors[physical()];
  }

  __device__ const CapsuleHeader *header() const {
    const OutputChunk part = chunk();
    const std::uint64_t row = physical();
    std::uint32_t low = 0u;
    std::uint32_t high = part.capsule_count;
    while (low < high) {
      const std::uint32_t middle = low + ((high - low) >> 1u);
      if (output.capsules[part.capsule_begin + middle].anchor_row < row)
        low = middle + 1u;
      else
        high = middle;
    }
    if (low == part.capsule_count) return nullptr;
    const OutputCapsule capsule = output.capsules[part.capsule_begin + low];
    return capsule.anchor_row == row
        ? reinterpret_cast<const CapsuleHeader *>(capsule.index.address)
        : nullptr;
  }

  __device__ std::uint32_t head4() const { return anchor().head; }

  __device__ std::uint64_t length() const {
    const CapsuleHeader *object = header();
    return object ? object->key_length : 4u;
  }

  __device__ std::uint8_t byte(std::uint64_t position) const {
    const CapsuleHeader *object = header();
    if (object)
      return reinterpret_cast<const std::uint8_t *>(object + 1u)[position];
    return static_cast<std::uint8_t>(
        head4() >> (24u - static_cast<std::uint32_t>(position) * 8u));
  }
};

struct OutputValueCursor {
  OutputKeyCursor key{};

  __device__ std::uint64_t length() const {
    const CapsuleHeader *object = key.header();
    return object ? object->value_length : 4u;
  }

  __device__ std::uint8_t byte(std::uint64_t position) const {
    const CapsuleHeader *object = key.header();
    if (object) {
      const auto *bytes = reinterpret_cast<const std::uint8_t *>(object + 1u);
      return bytes[object->key_length + position];
    }
    return static_cast<std::uint8_t>(key.anchor().value >> (8u * position));
  }

  __device__ std::uint32_t inline_word() const {
    return key.anchor().value;
  }
};

__global__ void emit_rank_tasks(
    const RankTask *tasks, const std::uint32_t *task_count,
    const std::uint32_t *begins, const std::uint32_t *ends,
    RankView view, OutputAnchor *output_anchors,
    std::uint64_t anchor_capacity, OutputCapsule *output_capsules,
    std::uint64_t capsule_capacity, std::uint8_t *output_bytes,
    std::uint64_t byte_capacity, CopyObject *copy_objects,
    OutputChunk *chunks, std::uint64_t *chunk_counts,
    RangeCounters *counters, unsigned long long *errors) {
  using BlockScan = cub::BlockScan<RangePrefix, kThreads>;
  __shared__ std::uint64_t tokens_a[kTileRows];
  __shared__ std::uint64_t tokens_b[kTileRows];
  __shared__ std::uint32_t segments[kMaximumSources + 1u];
  __shared__ typename BlockScan::TempStorage scan;
  __shared__ std::uint32_t total;
  __shared__ unsigned long long record_total;
  __shared__ unsigned long long capsule_total;
  __shared__ unsigned long long byte_total;
  __shared__ unsigned long long record_base;
  __shared__ unsigned long long capsule_base;
  __shared__ unsigned long long byte_base;
  __shared__ unsigned long long running_records;
  __shared__ unsigned long long running_capsules;
  __shared__ unsigned long long running_bytes;
  const std::uint32_t task = blockIdx.x;
  if (task >= *task_count) return;
  const std::uint64_t bounds = std::uint64_t{task} * view.source_count;
  std::uint64_t *tokens = merge_rank_task(
      view, begins + bounds, ends + bounds, tokens_a, tokens_b,
      segments, total, errors);
  if (total > kTileRows) return;
  if (threadIdx.x == 0u) {
    record_total = 0u;
    capsule_total = 0u;
    byte_total = 0u;
  }
  __syncthreads();
  std::uint64_t local_records = 0u;
  std::uint64_t local_capsules = 0u;
  std::uint64_t local_bytes = 0u;
  for (std::uint32_t index = threadIdx.x; index < total;
       index += blockDim.x) {
    if (!token_live(view, tokens, index)) continue;
    const std::uint64_t token = tokens[index];
    const RankKeyCursor key{
        view, token_source(token), token_row(token)};
    const RankValueCursor value = rank_value(
        view, token_source(token), token_row(token));
    const bool capsule = key.length() != 4u || value.length() != 4u;
    ++local_records;
    local_capsules += capsule;
    if (capsule) {
      std::uint64_t object_bytes = 0u;
      if (!capsule_object_bytes(key.length(), value.length(), object_bytes))
        atomicAdd(errors, 1ull);
      local_bytes += object_bytes;
    }
  }
  if (local_records) atomicAdd(&record_total, local_records);
  if (local_capsules) atomicAdd(&capsule_total, local_capsules);
  if (local_bytes) atomicAdd(&byte_total, local_bytes);
  __syncthreads();
  if (threadIdx.x == 0u) {
    record_base = atomicAdd(&counters->records, record_total);
    capsule_base = atomicAdd(&counters->capsule_indexes, capsule_total);
    byte_base = atomicAdd(&counters->capsule_bytes, byte_total);
    running_records = 0u;
    running_capsules = 0u;
    running_bytes = 0u;
    if (record_base + record_total > anchor_capacity ||
        capsule_base + capsule_total > capsule_capacity ||
        byte_base + byte_total > byte_capacity)
      atomicOr(&counters->status, kOutputOverflow);
    chunks[task] = {record_base, 0u, capsule_base,
                    static_cast<std::uint32_t>(record_total),
                    static_cast<std::uint32_t>(capsule_total),
                    tasks[task].component, task};
    chunk_counts[task] = record_total;
  }
  __syncthreads();
  const bool writable = record_base + record_total <= anchor_capacity &&
      capsule_base + capsule_total <= capsule_capacity &&
      byte_base + byte_total <= byte_capacity;
  for (std::uint32_t chunk = 0u; chunk < total; chunk += blockDim.x) {
    const std::uint32_t index = chunk + threadIdx.x;
    const bool live = index < total && token_live(view, tokens, index);
    std::uint64_t token = 0u;
    RangePrefix item{};
    if (live) {
      token = tokens[index];
      const RankKeyCursor key{
          view, token_source(token), token_row(token)};
      const RankValueCursor value = rank_value(
          view, token_source(token), token_row(token));
      const bool capsule = key.length() != 4u || value.length() != 4u;
      item.records = 1u;
      item.capsules = capsule;
      if (capsule) capsule_object_bytes(
          key.length(), value.length(), item.bytes);
    }
    RangePrefix prefix{};
    RangePrefix aggregate{};
    BlockScan(scan).ExclusiveScan(
        item, prefix, RangePrefix{}, AddRangePrefix{}, aggregate);
    __syncthreads();
    if (live && writable) {
      const std::uint32_t source = token_source(token);
      const std::uint32_t row = token_row(token);
      const RankKeyCursor key{view, source, row};
      const RankValueCursor value = rank_value(view, source, row);
      const std::uint64_t output = record_base + running_records +
          prefix.records;
      output_anchors[output] = {key.head4(), value.inline_word()};
      if (item.capsules) {
        const std::uint64_t capsule = capsule_base + running_capsules +
            prefix.capsules;
        const std::uint64_t object_offset = byte_base + running_bytes +
            prefix.bytes;
        const CapsuleIndex index_entry{
            reinterpret_cast<std::uint64_t>(output_bytes + object_offset),
            rejection_hash(key), 0u};
        output_capsules[capsule] = {output, index_entry};
        copy_objects[capsule] = {source, row,
                                index_entry.address};
      }
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
      running_records += aggregate.records;
      running_capsules += aggregate.capsules;
      running_bytes += aggregate.bytes;
    }
    __syncthreads();
  }
}

__global__ void copy_output_capsules(
    const CopyObject *objects, std::uint64_t capacity,
    const RangeCounters *counters, RankView view) {
  const std::uint64_t object = blockIdx.x;
  if (object >= capacity || object >= counters->capsule_indexes ||
      counters->status)
    return;
  const CopyObject copy = objects[object];
  const RankKeyCursor key{view, copy.source, copy.row};
  const RankValueCursor value = rank_value(view, copy.source, copy.row);
  auto *header = reinterpret_cast<CapsuleHeader *>(copy.destination);
  if (threadIdx.x == 0u)
    *header = {key.length(), value.length(), 0u, 0u};
  auto *bytes = reinterpret_cast<std::uint8_t *>(header + 1u);
  for (std::uint64_t position = threadIdx.x;
       position < key.length() + value.length(); position += blockDim.x) {
    bytes[position] = position < key.length()
        ? key.byte(position) : value.byte(position - key.length());
  }
}

__global__ void finish_output(
    OutputChunk *chunks, const std::uint64_t *logical_offsets,
    const std::uint32_t *chunk_count, const RangeCounters *counters,
    RangeReceipt *receipt) {
  const std::uint32_t chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk < *chunk_count)
    chunks[chunk].logical_begin = logical_offsets[chunk];
  if (!chunk) {
    receipt->records = counters->records;
    receipt->capsule_indexes = counters->capsule_indexes;
    receipt->capsule_bytes = counters->capsule_bytes;
    receipt->chunks = *chunk_count;
    receipt->status = counters->status;
  }
}

__device__ inline std::uint64_t lower_bound_output(
    OutputView output, std::uint64_t begin, std::uint64_t end,
    const DeviceKeyCursor &query, bool strict) {
  while (begin < end) {
    const std::uint64_t middle = begin + ((end - begin) >> 1u);
    const int comparison = compare_keys(OutputKeyCursor{output, middle}, query);
    if (comparison < 0 || (strict && comparison == 0)) begin = middle + 1u;
    else end = middle;
  }
  return begin;
}

__global__ void mark_endpoint_groups(
    EndpointSource source, const std::uint32_t *ordered,
    std::uint32_t count, std::uint32_t *flags) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= count) return;
  if (!position) {
    flags[position] = 1u;
    return;
  }
  const std::uint32_t current = ordered[position];
  const std::uint32_t previous = ordered[position - 1u];
  const bool current_upper = current >= source.queries.count;
  const bool previous_upper = previous >= source.queries.count;
  flags[position] = static_cast<std::uint32_t>(
      current_upper != previous_upper ||
      compare_keys(source_key(source, current),
                   source_key(source, previous)) != 0);
}

__global__ void scatter_unique_endpoints(
    const std::uint32_t *ordered, const std::uint32_t *flags,
    const std::uint32_t *group_ids, std::uint32_t count,
    std::uint32_t *unique_refs) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position < count && flags[position])
    unique_refs[group_ids[position]] = ordered[position];
}

__global__ void map_unique_endpoints(
    EndpointSource source, const std::uint32_t *unique_refs,
    const std::uint32_t *unique_count, OutputView output,
    const RangeReceipt *receipt, std::uint64_t *unique_ranks) {
  const std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= *unique_count) return;
  const std::uint32_t ref = unique_refs[group];
  const bool upper = ref >= source.queries.count;
  unique_ranks[group] = receipt->status ? 0u : lower_bound_output(
      output, 0u, receipt->records, source_key(source, ref), upper);
}

__global__ void scatter_endpoint_slices(
    EndpointSource source, const std::uint32_t *ordered,
    const std::uint32_t *flags, const std::uint32_t *group_ids,
    const std::uint64_t *unique_ranks,
    std::uint32_t count, std::uint64_t *endpoint_ranks) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= count) return;
  const std::uint32_t ref = ordered[position];
  const std::uint32_t group =
      group_ids[position] + flags[position] - 1u;
  endpoint_ranks[ref] = unique_ranks[group];
}

__global__ void assemble_endpoint_slices(
    std::uint32_t query_count, const std::uint64_t *endpoint_ranks,
    const RangeReceipt *receipt, RangeSlice *slices) {
  const std::uint32_t query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= query_count) return;
  slices[query] = receipt->status
      ? RangeSlice{0u, 0u, 0u, receipt->status}
      : RangeSlice{endpoint_ranks[query],
                   endpoint_ranks[query_count + query], 0u, 0u};
}

class RosterStorage {
 public:
  RosterStorage(std::uint32_t roots, std::uint32_t sections,
                std::uint32_t span_capacity)
      : root_count_(roots), section_count_(sections),
        span_counts_(std::uint64_t{roots} * sections),
        span_offsets_(std::uint64_t{roots} * sections),
        row_counts_(std::uint64_t{roots} * sections),
        row_offsets_(std::uint64_t{roots} * sections),
        spans_(span_capacity + 1u), sources_(roots + 1u), errors_(1u) {
    std::size_t span_bytes = 0u;
    std::size_t row_bytes = 0u;
    const std::uint32_t entries = roots * sections;
    check(cub::DeviceScan::ExclusiveSum(
              nullptr, span_bytes, span_counts_.data(),
              span_offsets_.data(), entries),
          "size root span scan");
    check(cub::DeviceScan::ExclusiveSum(
              nullptr, row_bytes, row_counts_.data(), row_offsets_.data(),
              sections),
          "size root row scan");
    scan_.reset(std::max(span_bytes, row_bytes));
    span_scan_bytes_ = span_bytes;
    row_scan_bytes_ = row_bytes;
  }

  std::pair<float, double> build(
      gpulsmopt2_detail::ResidentRows arena,
      const gpulsmopt2_detail::Descriptor *descriptors,
      ActiveSectionRoster &active, const RootBuildState *roots,
      cudaStream_t stream) {
    const std::uint32_t entries = root_count_ * section_count_;
    check(cudaMemsetAsync(errors_.data(), 0, errors_.bytes(), stream),
          "clear roster errors");
    const auto timing = time_once([&] {
      count_root_spans<<<blocks(entries), kThreads, 0, stream>>>(
          active.sections(), section_count_, arena, descriptors,
          roots, active.levels(),
          root_count_, span_counts_.data(), row_counts_.data());
      std::size_t bytes = span_scan_bytes_;
      check(cub::DeviceScan::ExclusiveSum(
                scan_.data(), bytes, span_counts_.data(),
                span_offsets_.data(), entries, stream),
            "scan root spans");
      for (std::uint32_t root = 0u; root < root_count_; ++root) {
        bytes = row_scan_bytes_;
        check(cub::DeviceScan::ExclusiveSum(
                  scan_.data(), bytes,
                  row_counts_.data() + std::uint64_t{root} * section_count_,
                  row_offsets_.data() + std::uint64_t{root} * section_count_,
                  section_count_, stream),
              "scan root logical rows");
      }
      fill_root_spans<<<blocks(entries), kThreads, 0, stream>>>(
          active.sections(), section_count_, arena, descriptors,
          roots, active.levels(),
          root_count_, span_offsets_.data(), row_offsets_.data(),
          spans_.data(), errors_.data());
    });
    const std::uint32_t last = entries - 1u;
    std::uint32_t last_offset = 0u;
    std::uint32_t last_count = 0u;
    check(cudaMemcpy(&last_offset, span_offsets_.data() + last,
                     sizeof(last_offset), cudaMemcpyDeviceToHost),
          "copy root span offset");
    check(cudaMemcpy(&last_count, span_counts_.data() + last,
                     sizeof(last_count), cudaMemcpyDeviceToHost),
          "copy root span count");
    root_span_total_ = last_offset + last_count;
    if (root_span_total_ + 1u > spans_.size())
      throw std::length_error("root roster span capacity");
    finish_roster_sources<<<blocks(root_count_ + 1u), kThreads, 0, stream>>>(
        section_count_, span_counts_.data(), span_offsets_.data(),
        row_counts_.data(), row_offsets_.data(), active.levels(), root_count_,
        root_span_total_, 0u, sources_.data(), spans_.data());
    check(cudaStreamSynchronize(stream), "finish root roster");
    unsigned long long errors = 0u;
    check(cudaMemcpy(&errors, errors_.data(), sizeof(errors),
                     cudaMemcpyDeviceToHost), "copy roster errors");
    if (errors) throw std::runtime_error("root roster validation");
    return timing;
  }

  void append_pending(std::uint32_t rows, cudaStream_t stream) {
    if (rows)
      append_pending_source<<<1u, 1u, 0, stream>>>(
          root_count_, root_span_total_, rows, sources_.data(), spans_.data());
  }

  const OrderSource *sources() const { return sources_.data(); }
  const OrderSpan *spans() const { return spans_.data(); }
  std::uint32_t root_count() const { return root_count_; }
  std::uint32_t root_span_total() const { return root_span_total_; }
  std::size_t bytes() const {
    return span_counts_.bytes() + span_offsets_.bytes() +
        row_counts_.bytes() + row_offsets_.bytes() + spans_.bytes() +
        sources_.bytes() + scan_.bytes() + errors_.bytes();
  }

 private:
  std::uint32_t root_count_{};
  std::uint32_t section_count_{};
  std::uint32_t root_span_total_{};
  Buffer<std::uint32_t> span_counts_, span_offsets_;
  Buffer<std::uint32_t> row_counts_, row_offsets_;
  Buffer<OrderSpan> spans_;
  Buffer<OrderSource> sources_;
  Buffer<std::uint8_t> scan_;
  Buffer<unsigned long long> errors_;
  std::size_t span_scan_bytes_{};
  std::size_t row_scan_bytes_{};
};

class OutputStorage {
 public:
  OutputStorage(std::uint64_t rows, std::uint64_t capsules,
                std::uint64_t bytes, std::uint32_t tasks,
                std::uint32_t queries)
      : anchors_(rows), capsules_(capsules), bytes_(bytes),
        copy_objects_(capsules), chunks_(tasks), chunk_counts_(tasks),
        logical_offsets_(tasks), counters_(1u), receipt_(1u),
        slices_(queries) {
    std::size_t scan_bytes = 0u;
    check(cub::DeviceScan::ExclusiveSum(
              nullptr, scan_bytes, chunk_counts_.data(),
              logical_offsets_.data(), tasks),
          "size rank output scan");
    scan_bytes_ = scan_bytes;
    scan_.reset(scan_bytes);
  }

  void clear(cudaStream_t stream) {
    check(cudaMemsetAsync(counters_.data(), 0, counters_.bytes(), stream),
          "clear rank output counters");
    check(cudaMemsetAsync(receipt_.data(), 0, receipt_.bytes(), stream),
          "clear rank output receipt");
  }

  OutputView view(std::uint32_t tasks) const {
    return {anchors_.data(), capsules_.data(), chunks_.data(),
            logical_offsets_.data(), tasks};
  }

  std::size_t bytes() const {
    return anchors_.bytes() + capsules_.bytes() + bytes_.bytes() +
        copy_objects_.bytes() + chunks_.bytes() + chunk_counts_.bytes() +
        logical_offsets_.bytes() + counters_.bytes() + receipt_.bytes() +
        slices_.bytes() + scan_.bytes();
  }

  Buffer<OutputAnchor> anchors_;
  Buffer<OutputCapsule> capsules_;
  Buffer<std::uint8_t> bytes_;
  Buffer<CopyObject> copy_objects_;
  Buffer<OutputChunk> chunks_;
  Buffer<std::uint64_t> chunk_counts_, logical_offsets_;
  Buffer<RangeCounters> counters_;
  Buffer<RangeReceipt> receipt_;
  Buffer<RangeSlice> slices_;
  Buffer<std::uint8_t> scan_;
  std::size_t scan_bytes_{};
};

struct EnumerationResult {
  float gpu_ms{};
  double wall_ms{};
  RangeReceipt receipt{};
  std::uint32_t components{};
  std::uint32_t tasks{};
  std::uint32_t pending_winners{};
  std::uint32_t planner_rounds{};
  unsigned long long errors{};
};

class Workspace {
 public:
  Workspace(std::uint32_t max_rows, std::uint32_t max_queries,
            std::uint32_t max_tasks, std::uint32_t source_count)
      : max_rows_(max_rows), max_queries_(max_queries),
        max_tasks_(max_tasks), source_count_(source_count),
        lower_order_(max_queries), endpoint_order_(2u * max_queries),
        pending_order_(max_rows),
        pending_age_refs_(max_rows), pending_slot_offsets_(
            gpulsmopt2_detail::kBatchesPerEpoch + 1u),
        components_(max_queries), component_count_(1u),
        tasks_a_(max_tasks), tasks_b_(max_tasks),
        classified_(max_tasks), task_count_a_(1u), task_count_b_(1u),
        child_counts_(max_tasks), child_offsets_(max_tasks),
        begins_a_(std::uint64_t{max_tasks} * source_count),
        ends_a_(std::uint64_t{max_tasks} * source_count),
        begins_b_(std::uint64_t{max_tasks} * source_count),
        ends_b_(std::uint64_t{max_tasks} * source_count),
        split_lowers_(std::uint64_t{max_tasks} * source_count),
        split_uppers_(std::uint64_t{max_tasks} * source_count),
        oversized_count_(1u), endpoint_flags_(2u * max_queries),
        endpoint_group_ids_(2u * max_queries),
        unique_endpoint_refs_(2u * max_queries),
        unique_endpoint_count_(1u), unique_endpoint_ranks_(2u * max_queries),
        endpoint_ranks_(2u * max_queries),
        errors_(1u) {
    std::size_t bytes = 0u;
    check(cub::DeviceScan::ExclusiveSum(
              nullptr, bytes, child_counts_.data(), child_offsets_.data(),
              max_tasks),
          "size rank task scan");
    task_scan_bytes_ = bytes;
    task_scan_.reset(bytes);
    bytes = 0u;
    check(cub::DeviceScan::ExclusiveSum(
              nullptr, bytes, endpoint_flags_.data(),
              endpoint_group_ids_.data(), 2u * max_queries),
          "size endpoint group scan");
    endpoint_scan_bytes_ = bytes;
    endpoint_scan_.reset(bytes);
  }

  EnumerationResult enumerate(
      RosterStorage &roster, RangeQueryBatch queries,
      OutputStorage &output, PendingReadView pending,
      const std::uint32_t *host_batch_counts,
      gpulsmopt2_detail::ResidentRows arena,
      const RootBuildState *roots, cudaStream_t stream) {
    if (queries.count > max_queries_)
      throw std::length_error("rank query capacity");
    std::array<std::uint32_t,
               gpulsmopt2_detail::kBatchesPerEpoch + 1u> slot_offsets{};
    for (std::uint32_t slot = 0u; slot < pending.batch_count; ++slot)
      slot_offsets[slot + 1u] = slot_offsets[slot] +
          host_batch_counts[slot];
    const std::uint32_t pending_rows = slot_offsets[pending.batch_count];
    if (pending_rows > max_rows_)
      throw std::length_error("rank pending row capacity");
    check(cudaMemcpyAsync(pending_slot_offsets_.data(), slot_offsets.data(),
                          sizeof(slot_offsets), cudaMemcpyHostToDevice, stream),
          "copy rank pending offsets");
    std::uint32_t pending_winners = 0u;
    std::uint32_t components = 0u;
    std::uint32_t task_count = 0u;
    std::uint32_t rounds = 0u;
    check(cudaMemsetAsync(errors_.data(), 0, errors_.bytes(), stream),
          "clear rank errors");
    output.clear(stream);
    const auto timing = time_once([&] {
      if (pending_rows) {
        const dim3 grid(128u, pending.batch_count);
        build_pending_age_refs<<<grid, kThreads, 0, stream>>>(
            pending, pending_slot_offsets_.data(), pending_age_refs_.data());
        const auto ordered = pending_order_.order(
            PendingOrderSource{pending}, pending_rows, stream, nullptr,
            pending_age_refs_.data());
        pending_winners = ordered.winner_count;
      }
      roster.append_pending(pending_winners, stream);
      const std::uint32_t actual_sources = roster.root_count() +
          (pending_winners ? 1u : 0u);
      if (actual_sources != source_count_)
        throw std::runtime_error("rank source count changed");
      RankView view{arena, roots, pending,
                    pending_order_.winners(), roster.sources(),
                    roster.spans(), actual_sources};
      lower_order_.order(
          QueryOrderSource{queries.lower}, queries.count, stream);
      coalesce_queries<<<1u, 1u, 0, stream>>>(
          queries, lower_order_.ordered(), components_.data(),
          component_count_.data(), errors_.data());
      check(cudaMemcpyAsync(&components, component_count_.data(),
                            sizeof(components), cudaMemcpyDeviceToHost,
                            stream), "copy rank component count");
      check(cudaStreamSynchronize(stream), "wait rank components");
      if (!components || components > max_tasks_)
        throw std::length_error("rank component capacity");
      task_count = components;
      check(cudaMemcpyAsync(task_count_a_.data(), &task_count,
                            sizeof(task_count), cudaMemcpyHostToDevice,
                            stream), "set initial rank task count");
      initialize_rank_tasks<<<components, kThreads, 0, stream>>>(
          queries, components_.data(), components, view, tasks_a_.data(),
          begins_a_.data(), ends_a_.data());
      RankTask *current_tasks = tasks_a_.data();
      RankTask *next_tasks = tasks_b_.data();
      std::uint32_t *current_count = task_count_a_.data();
      std::uint32_t *next_count = task_count_b_.data();
      std::uint32_t *current_begins = begins_a_.data();
      std::uint32_t *current_ends = ends_a_.data();
      std::uint32_t *next_begins = begins_b_.data();
      std::uint32_t *next_ends = ends_b_.data();
      for (rounds = 0u; rounds < 32u; ++rounds) {
        check(cudaMemsetAsync(oversized_count_.data(), 0,
                              oversized_count_.bytes(), stream),
              "clear rank oversized count");
        classify_rank_tasks<<<task_count, kThreads, 0, stream>>>(
            current_tasks, current_count, current_begins, current_ends,
            view, child_counts_.data(), split_lowers_.data(),
            split_uppers_.data(), classified_.data(),
            oversized_count_.data());
        std::uint32_t oversized = 0u;
        check(cudaMemcpyAsync(&oversized, oversized_count_.data(),
                              sizeof(oversized), cudaMemcpyDeviceToHost,
                              stream), "copy rank oversized count");
        check(cudaStreamSynchronize(stream), "wait rank classification");
        if (!oversized) break;
        std::size_t bytes = task_scan_bytes_;
        check(cub::DeviceScan::ExclusiveSum(
                  task_scan_.data(), bytes, child_counts_.data(),
                  child_offsets_.data(), task_count, stream),
              "scan rank children");
        finish_count_scan<<<1u, 1u, 0, stream>>>(
            child_counts_.data(), child_offsets_.data(), task_count,
            next_count);
        std::uint32_t next_host = 0u;
        check(cudaMemcpyAsync(&next_host, next_count, sizeof(next_host),
                              cudaMemcpyDeviceToHost, stream),
              "copy rank child count");
        check(cudaStreamSynchronize(stream), "wait rank child count");
        if (!next_host || next_host > max_tasks_)
          throw std::length_error("rank planner task capacity");
        clear_rank_tasks<<<blocks(next_host), kThreads, 0, stream>>>(
            next_tasks, 0u, next_host);
        emit_rank_children<<<task_count, kThreads, 0, stream>>>(
            current_tasks, classified_.data(), current_count,
            child_offsets_.data(), current_begins, current_ends,
            split_lowers_.data(), split_uppers_.data(), source_count_,
            next_tasks, next_begins, next_ends);
        std::swap(current_tasks, next_tasks);
        std::swap(current_count, next_count);
        std::swap(current_begins, next_begins);
        std::swap(current_ends, next_ends);
        task_count = next_host;
      }
      if (rounds == 32u)
        throw std::runtime_error("rank planner did not converge");
      emit_rank_tasks<<<task_count, kThreads, 0, stream>>>(
          current_tasks, current_count, current_begins, current_ends, view,
          output.anchors_.data(), output.anchors_.size(),
          output.capsules_.data(), output.capsules_.size(),
          output.bytes_.data(), output.bytes_.size(),
          output.copy_objects_.data(), output.chunks_.data(),
          output.chunk_counts_.data(), output.counters_.data(),
          errors_.data());
      if (output.capsules_.size())
        copy_output_capsules<<<output.capsules_.size(), kThreads, 0, stream>>>(
            output.copy_objects_.data(), output.capsules_.size(),
            output.counters_.data(), view);
      std::size_t bytes = output.scan_bytes_;
      check(cub::DeviceScan::ExclusiveSum(
                output.scan_.data(), bytes, output.chunk_counts_.data(),
                output.logical_offsets_.data(), task_count, stream),
            "scan rank output chunks");
      finish_output<<<blocks(task_count), kThreads, 0, stream>>>(
          output.chunks_.data(), output.logical_offsets_.data(),
          current_count, output.counters_.data(), output.receipt_.data());
      const EndpointSource endpoints{queries};
      const std::uint32_t endpoint_count = 2u * queries.count;
      endpoint_order_.order(endpoints, endpoint_count, stream);
      mark_endpoint_groups<<<blocks(endpoint_count), kThreads, 0, stream>>>(
          endpoints, endpoint_order_.ordered(), endpoint_count,
          endpoint_flags_.data());
      bytes = endpoint_scan_bytes_;
      check(cub::DeviceScan::ExclusiveSum(
                endpoint_scan_.data(), bytes, endpoint_flags_.data(),
                endpoint_group_ids_.data(), endpoint_count, stream),
            "scan endpoint groups");
      finish_count_scan<<<1u, 1u, 0, stream>>>(
          endpoint_flags_.data(), endpoint_group_ids_.data(),
          endpoint_count, unique_endpoint_count_.data());
      scatter_unique_endpoints<<<blocks(endpoint_count), kThreads, 0,
          stream>>>(endpoint_order_.ordered(), endpoint_flags_.data(),
                    endpoint_group_ids_.data(), endpoint_count,
                    unique_endpoint_refs_.data());
      map_unique_endpoints<<<blocks(endpoint_count), kThreads, 0, stream>>>(
          endpoints, unique_endpoint_refs_.data(),
          unique_endpoint_count_.data(), output.view(task_count),
          output.receipt_.data(), unique_endpoint_ranks_.data());
      scatter_endpoint_slices<<<blocks(endpoint_count), kThreads, 0,
          stream>>>(endpoints, endpoint_order_.ordered(),
                    endpoint_flags_.data(), endpoint_group_ids_.data(),
                    unique_endpoint_ranks_.data(), endpoint_count,
                    endpoint_ranks_.data());
      assemble_endpoint_slices<<<blocks(queries.count), kThreads, 0,
          stream>>>(queries.count, endpoint_ranks_.data(),
                    output.receipt_.data(), output.slices_.data());
    });
    RangeReceipt receipt{};
    unsigned long long errors = 0u;
    check(cudaMemcpy(&receipt, output.receipt_.data(), sizeof(receipt),
                     cudaMemcpyDeviceToHost), "copy rank receipt");
    check(cudaMemcpy(&errors, errors_.data(), sizeof(errors),
                     cudaMemcpyDeviceToHost), "copy rank errors");
    return {timing.first, timing.second, receipt, components, task_count,
            pending_winners, rounds, errors};
  }

  std::size_t bytes() const {
    return lower_order_.bytes() + endpoint_order_.bytes() +
        pending_order_.bytes() +
        pending_age_refs_.bytes() + pending_slot_offsets_.bytes() +
        components_.bytes() + component_count_.bytes() + tasks_a_.bytes() +
        tasks_b_.bytes() + classified_.bytes() + task_count_a_.bytes() +
        task_count_b_.bytes() + child_counts_.bytes() +
        child_offsets_.bytes() + begins_a_.bytes() + ends_a_.bytes() +
        begins_b_.bytes() + ends_b_.bytes() + split_lowers_.bytes() +
        split_uppers_.bytes() + oversized_count_.bytes() +
        endpoint_flags_.bytes() + endpoint_group_ids_.bytes() +
        unique_endpoint_refs_.bytes() + unique_endpoint_count_.bytes() +
        unique_endpoint_ranks_.bytes() + endpoint_ranks_.bytes() +
        task_scan_.bytes() +
        endpoint_scan_.bytes() + errors_.bytes();
  }

 private:
  std::uint32_t max_rows_{};
  std::uint32_t max_queries_{};
  std::uint32_t max_tasks_{};
  std::uint32_t source_count_{};
  ExactOrderWorkspace lower_order_, endpoint_order_;
  ExactOrderWorkspace pending_order_;
  Buffer<std::uint32_t> pending_age_refs_, pending_slot_offsets_;
  Buffer<RangeComponent> components_;
  Buffer<std::uint32_t> component_count_;
  Buffer<RankTask> tasks_a_, tasks_b_, classified_;
  Buffer<std::uint32_t> task_count_a_, task_count_b_;
  Buffer<std::uint32_t> child_counts_, child_offsets_;
  Buffer<std::uint32_t> begins_a_, ends_a_, begins_b_, ends_b_;
  Buffer<std::uint32_t> split_lowers_, split_uppers_;
  Buffer<std::uint32_t> oversized_count_;
  Buffer<std::uint32_t> endpoint_flags_, endpoint_group_ids_;
  Buffer<std::uint32_t> unique_endpoint_refs_, unique_endpoint_count_;
  Buffer<std::uint64_t> unique_endpoint_ranks_, endpoint_ranks_;
  Buffer<std::uint8_t> task_scan_, endpoint_scan_;
  std::size_t task_scan_bytes_{};
  std::size_t endpoint_scan_bytes_{};
  Buffer<unsigned long long> errors_;
};

struct PreparedReadState {
  std::vector<std::uint32_t> levels;
  std::array<RootBuildState, kLevels> roots{};
  std::array<std::uint32_t,
             gpulsmopt2_detail::kBatchesPerEpoch> batch_counts{};
  PendingReadView pending{};
  std::uint32_t pending_rows{};
  std::uint64_t candidate_rows{};
  std::uint64_t capsule_count{};
  std::uint64_t capsule_bytes{};
};

inline std::uint32_t roster_capacity(
    const PreparedReadState &state, std::uint32_t active_sections) {
  std::uint64_t exact = 0u;
  for (const std::uint32_t level : state.levels)
    exact += state.roots[level].exact_head_count;
  const std::uint64_t capacity =
      std::uint64_t{state.levels.size()} * active_sections + 2u * exact + 1u;
  if (capacity > std::numeric_limits<std::uint32_t>::max())
    throw std::length_error("range roster capacity overflow");
  return static_cast<std::uint32_t>(capacity);
}

__global__ void reduce_range_slices(
    OutputView output, const RangeSlice *slices,
    std::uint32_t query_count, std::uint32_t *sums) {
  using BlockReduce = cub::BlockReduce<std::uint32_t, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduction;
  const std::uint32_t query = blockIdx.x;
  if (query >= query_count) return;
  const RangeSlice slice = slices[query];
  std::uint32_t local = 0u;
  if (!slice.status) {
    for (std::uint64_t row = slice.begin + threadIdx.x;
         row < slice.end; row += blockDim.x)
      local += OutputValueCursor{OutputKeyCursor{output, row}}.inline_word();
  }
  const std::uint32_t total = BlockReduce(reduction).Sum(local);
  if (!threadIdx.x) sums[query] = slice.status ? 0u : total;
}


}  // namespace gpulsm_sparse::rank_range
