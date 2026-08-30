#pragma once

#include "ordered_read.cuh"

#include <cub/device/device_select.cuh>

namespace gpulsm_sparse::range_gate {

constexpr std::uint32_t kOutputOverflow = 1u;

template <class Function>
inline std::pair<float, double> time_once(Function &&function) {
  function();
  return {};
}

struct RangeQueryBatch {
  RecordBatchView lower{};
  RecordBatchView upper{};
  std::uint32_t count{};
};

struct QueryOrderSource {
  RecordBatchView records{};
};

__host__ __device__ inline DeviceKeyCursor source_key(
    QueryOrderSource source, std::uint32_t ref) {
  return {source.records.keys, ref, source.records.head4_words};
}

struct QueryOrderValue {
  __host__ __device__ std::uint64_t length() const { return 0u; }
  __host__ __device__ std::uint8_t byte(std::uint64_t) const { return 0u; }
  __host__ __device__ std::uint32_t inline_word() const { return 0u; }
};

__host__ __device__ inline QueryOrderValue source_value(
    QueryOrderSource, std::uint32_t) {
  return {};
}

__host__ __device__ inline bool source_tombstone(
    QueryOrderSource, std::uint32_t) {
  return false;
}

__host__ __device__ inline std::uint32_t source_summary(
    QueryOrderSource, std::uint32_t) {
  return 0u;
}

struct RangeComponent {
  std::uint32_t lower_ref{};
  std::uint32_t upper_ref{};
  std::uint32_t task_begin{};
  std::uint32_t task_count{};
};

struct RangeSlice {
  std::uint64_t begin{};
  std::uint64_t end{};
  std::uint32_t component{};
  std::uint32_t status{};
};

struct RangeReceipt {
  std::uint64_t records{};
  std::uint64_t capsule_indexes{};
  std::uint64_t capsule_bytes{};
  std::uint32_t chunks{};
  std::uint32_t status{};
};

struct RangeCounters {
  unsigned long long records{};
  unsigned long long capsule_indexes{};
  unsigned long long capsule_bytes{};
  unsigned int status{};
};

struct RangePrefix {
  std::uint64_t records{};
  std::uint64_t capsules{};
  std::uint64_t bytes{};
};

struct AddRangePrefix {
  __host__ __device__ RangePrefix operator()(
      const RangePrefix &left, const RangePrefix &right) const {
    return {left.records + right.records,
            left.capsules + right.capsules,
            left.bytes + right.bytes};
  }
};

struct PendingOrderSource {
  PendingReadView pending{};
};

__device__ inline PendingRecordCursor pending_from_encoded(
    PendingReadView pending, std::uint32_t ref) {
  return {pending,
          ref >> gpulsmopt2_detail::kBatchPositionBits,
          ref & ((1u << gpulsmopt2_detail::kBatchPositionBits) - 1u)};
}

__device__ inline PendingRecordCursor source_key(
    PendingOrderSource source, std::uint32_t ref) {
  return pending_from_encoded(source.pending, ref);
}

struct PendingOrderValue {
  PendingRecordCursor record{};

  __device__ std::uint64_t length() const {
    if (record.payload().metadata & gpulsmopt2_detail::kRawTombstone)
      return 0u;
    const CapsuleHeader *object = record.header();
    return object ? object->value_length : 4u;
  }

  __device__ std::uint8_t byte(std::uint64_t position) const {
    const CapsuleHeader *object = record.header();
    if (object) {
      const auto *bytes = reinterpret_cast<const std::uint8_t *>(object + 1u);
      return bytes[object->key_length + position];
    }
    return static_cast<std::uint8_t>(
        record.payload().value >> (8u * position));
  }

  __device__ std::uint32_t inline_word() const {
    const CapsuleHeader *object = record.header();
    if (!object) return record.payload().value;
    std::uint32_t result = 0u;
    const std::uint64_t count = object->value_length < 4u
        ? object->value_length : 4u;
    for (std::uint64_t position = 0u; position < count; ++position)
      result |= std::uint32_t{byte(position)} << (8u * position);
    return result;
  }
};

__device__ inline PendingOrderValue source_value(
    PendingOrderSource source, std::uint32_t ref) {
  return {pending_from_encoded(source.pending, ref)};
}

__device__ inline bool source_tombstone(
    PendingOrderSource source, std::uint32_t ref) {
  return (pending_from_encoded(source.pending, ref).payload().metadata &
          gpulsmopt2_detail::kRawTombstone) != 0u;
}

__device__ inline std::uint32_t source_summary(
    PendingOrderSource source, std::uint32_t ref) {
  return pending_from_encoded(source.pending, ref).payload().value;
}

__global__ void build_pending_age_refs(
    PendingReadView pending, const std::uint32_t *slot_offsets,
    std::uint32_t *refs) {
  const std::uint32_t slot = blockIdx.y;
  if (slot >= pending.batch_count) return;
  const std::uint32_t count =
      slot_offsets[slot + 1u] - slot_offsets[slot];
  for (std::uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
       local < count; local += blockDim.x * gridDim.x) {
    const std::uint64_t physical =
        std::uint64_t{slot} * pending.batch_capacity + local;
    const auto payload = pending.payloads[physical];
    const std::uint32_t position = payload.metadata &
        ((1u << gpulsmopt2_detail::kBatchPositionBits) - 1u);
    refs[slot_offsets[slot] + position] =
        (slot << gpulsmopt2_detail::kBatchPositionBits) | local;
  }
}

__device__ inline bool root_has_exact_section(
    const RootBuildState &root, std::uint32_t section) {
  const auto *heads = reinterpret_cast<const ExactHeadDescriptor *>(
      root.exact_heads);
  std::uint32_t low = 0u;
  std::uint32_t high = root.exact_head_count;
  const std::uint32_t target = section << 16u;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (heads[middle].head < target) low = middle + 1u;
    else high = middle;
  }
  return low < root.exact_head_count &&
      (heads[low].head >> 16u) == section;
}

__global__ void mark_active_sections(
    const gpulsmopt2_detail::Descriptor *descriptors,
    const RootBuildState *roots, const std::uint32_t *levels,
    std::uint32_t root_count, PendingReadView pending,
    std::uint8_t *flags) {
  const std::uint32_t section = blockIdx.x * blockDim.x + threadIdx.x;
  if (section >= kSections) return;
  bool active = false;
  for (std::uint32_t source = 0u; source < root_count && !active; ++source) {
    const std::uint32_t level = levels[source];
    active = descriptors[
        gpulsmopt2_detail::descriptor_index(section, level)].count() ||
        root_has_exact_section(roots[level], section);
  }
  for (std::uint32_t slot = 0u;
       slot < pending.batch_count && !active; ++slot) {
    const std::uint64_t base = std::uint64_t{slot} * (kSections + 1u);
    active = pending.offsets[base + section + 1u] !=
        pending.offsets[base + section];
  }
  flags[section] = static_cast<std::uint8_t>(active);
}

__global__ void coalesce_queries(
    RangeQueryBatch queries, const std::uint32_t *ordered,
    RangeComponent *components, std::uint32_t *component_count,
    unsigned long long *errors) {
  if (blockIdx.x || threadIdx.x) return;
  *component_count = 0u;
  if (!queries.count) return;
  std::uint32_t current_lower = ordered[0];
  std::uint32_t current_upper = current_lower;
  const DeviceKeyCursor first_lower{
      queries.lower.keys, current_lower, queries.lower.head4_words};
  const DeviceKeyCursor first_upper{
      queries.upper.keys, current_upper, queries.upper.head4_words};
  if (compare_keys(first_lower, first_upper) > 0) {
    atomicAdd(errors, 1ull);
    return;
  }
  for (std::uint32_t position = 1u; position < queries.count; ++position) {
    const std::uint32_t query = ordered[position];
    const DeviceKeyCursor lower{
        queries.lower.keys, query, queries.lower.head4_words};
    const DeviceKeyCursor upper{
        queries.upper.keys, query, queries.upper.head4_words};
    if (compare_keys(lower, upper) > 0) {
      atomicAdd(errors, 1ull);
      return;
    }
    const DeviceKeyCursor active_upper{
        queries.upper.keys, current_upper, queries.upper.head4_words};
    if (compare_keys(lower, active_upper) <= 0) {
      if (compare_keys(active_upper, upper) < 0) current_upper = query;
      continue;
    }
    components[*component_count] = {current_lower, current_upper, 0u, 0u};
    ++*component_count;
    current_lower = query;
    current_upper = query;
  }
  components[*component_count] = {current_lower, current_upper, 0u, 0u};
  ++*component_count;
}

__global__ void finish_count_scan(
    const std::uint32_t *counts, const std::uint32_t *offsets,
    std::uint32_t count, std::uint32_t *total) {
  if (blockIdx.x || threadIdx.x) return;
  *total = count ? offsets[count - 1u] + counts[count - 1u] : 0u;
}

__device__ inline void exact_head_bounds(
    const RootBuildState &root, std::uint32_t section,
    std::uint32_t &begin, std::uint32_t &end) {
  const auto *heads = reinterpret_cast<const ExactHeadDescriptor *>(
      root.exact_heads);
  const std::uint32_t low_head = section << 16u;
  const std::uint32_t high_head = section == 0xffffu
      ? std::numeric_limits<std::uint32_t>::max()
      : (section + 1u) << 16u;
  begin = 0u;
  end = root.exact_head_count;
  while (begin < end) {
    const std::uint32_t middle = begin + ((end - begin) >> 1u);
    if (heads[middle].head < low_head) begin = middle + 1u;
    else end = middle;
  }
  const std::uint32_t first = begin;
  end = root.exact_head_count;
  while (begin < end) {
    const std::uint32_t middle = begin + ((end - begin) >> 1u);
    const bool before = section == 0xffffu
        ? true : heads[middle].head < high_head;
    if (before) begin = middle + 1u;
    else end = middle;
  }
  end = begin;
  begin = first;
}

__device__ inline std::uint32_t lower_bound_ordinary_head(
    gpulsmopt2_detail::ResidentRows arena,
    gpulsmopt2_detail::Descriptor rows, std::uint16_t suffix) {
  return gpulsmopt2_detail::lower_bound_rows(
      arena + rows.offset(), rows.count(), suffix);
}

class ActiveSectionRoster {
 public:
  ActiveSectionRoster()
      : flags_(kSections), sections_(kSections), count_(1u) {
    cub::CountingInputIterator<std::uint32_t> ids(0u);
    check(cub::DeviceSelect::Flagged(
              nullptr, temporary_bytes_, ids, flags_.data(),
              sections_.data(), count_.data(), kSections),
          "size active section select");
    temporary_.reset(temporary_bytes_);
  }

  std::pair<float, double> rebuild(
      const gpulsmopt2_detail::Descriptor *descriptors,
      const RootBuildState *roots,
      const std::vector<std::uint32_t> &levels,
      PendingReadView pending, cudaStream_t stream) {
    levels_.reset(levels.size());
    if (!levels.empty())
      check(cudaMemcpyAsync(levels_.data(), levels.data(), levels_.bytes(),
                            cudaMemcpyHostToDevice, stream),
            "copy active root levels");
    const auto timing = time_once([&] {
      mark_active_sections<<<blocks(kSections), kThreads, 0, stream>>>(
          descriptors, roots, levels_.data(),
          static_cast<std::uint32_t>(levels.size()), pending, flags_.data());
      cub::CountingInputIterator<std::uint32_t> ids(0u);
      std::size_t bytes = temporary_bytes_;
      check(cub::DeviceSelect::Flagged(
                temporary_.data(), bytes, ids, flags_.data(), sections_.data(),
                count_.data(), kSections, stream),
            "select active sections");
    });
    check(cudaMemcpy(&host_count_, count_.data(), sizeof(host_count_),
                     cudaMemcpyDeviceToHost),
          "copy active section count");
    return timing;
  }

  const std::uint32_t *sections() const { return sections_.data(); }
  const std::uint32_t *levels() const { return levels_.data(); }
  std::uint32_t count() const { return host_count_; }
  std::size_t bytes() const {
    return flags_.bytes() + sections_.bytes() + count_.bytes() +
        levels_.bytes() + temporary_.bytes();
  }

 private:
  Buffer<std::uint8_t> flags_, temporary_;
  Buffer<std::uint32_t> sections_, count_, levels_;
  std::size_t temporary_bytes_{};
  std::uint32_t host_count_{};
};

}  // namespace gpulsm_sparse::range_gate
