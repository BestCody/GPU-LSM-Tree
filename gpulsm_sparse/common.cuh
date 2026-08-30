#pragma once

#include "../gpu_dictionary_adapter.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpulsm_sparse {

constexpr std::uint32_t kThreads = 256u;
constexpr std::uint32_t kSections = gpulsmopt2_detail::kQuotients;
constexpr std::uint32_t kLevels = gpulsmopt2_detail::kMaximumLevels;

inline void check(cudaError_t error, const char *where) {
  if (error != cudaSuccess)
    throw std::runtime_error(
        std::string(where) + ": " + cudaGetErrorString(error));
}

inline std::uint32_t blocks(std::uint64_t count) {
  return static_cast<std::uint32_t>(
      std::max<std::uint64_t>(1u, (count + kThreads - 1u) / kThreads));
}

template <class T>
class Buffer {
 public:
  Buffer() = default;
  explicit Buffer(std::size_t count) { reset(count); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept { swap(other); }
  Buffer &operator=(Buffer &&other) noexcept {
    if (this != &other) {
      clear();
      swap(other);
    }
    return *this;
  }
  ~Buffer() { clear(); }

  void reset(std::size_t count) {
    clear();
    count_ = count;
    if (count) check(cudaMalloc(&pointer_, count * sizeof(T)), "cudaMalloc");
  }

  void clear() noexcept {
    if (pointer_) cudaFree(pointer_);
    pointer_ = nullptr;
    count_ = 0u;
  }

  T *data() const { return pointer_; }
  std::size_t size() const { return count_; }
  std::size_t bytes() const { return count_ * sizeof(T); }

  void swap(Buffer &other) noexcept {
    std::swap(pointer_, other.pointer_);
    std::swap(count_, other.count_);
  }

 private:
  T *pointer_{};
  std::size_t count_{};
};

// This is the validated structural radix-4 forest planner.  It describes
// final roots symbolically so a sealed call materializes only those roots.
struct PlannedRoot {
  std::uint64_t resident_sources{};
  std::uint64_t raw_begin{};
  std::uint64_t raw_epochs{};
  std::uint32_t destination{};
};

struct ForestPlan {
  std::uint64_t old_mask{};
  std::uint64_t consumed_mask{};
  std::uint64_t output_mask{};
  std::uint64_t final_mask{};
  std::uint32_t output_count{};
  std::array<PlannedRoot, kLevels> outputs{};
};

struct PlannerNode {
  bool occupied{};
  bool created{};
  std::uint64_t resident_sources{};
  std::uint64_t raw_begin{};
  std::uint64_t raw_epochs{};
};

inline ForestPlan plan_forest(
    std::uint64_t old_mask, std::uint64_t complete_epochs,
    const std::vector<std::uint32_t> &tier_slots) {
  std::array<PlannerNode, kLevels> levels{};
  std::uint32_t level_base = 0u;
  for (std::uint32_t tier = 0u; tier < tier_slots.size(); ++tier) {
    const std::uint32_t slots = tier_slots[tier];
    const std::uint64_t mask = ((std::uint64_t{1u} << slots) - 1u)
        << level_base;
    const std::uint32_t count = static_cast<std::uint32_t>(
        __builtin_popcountll(old_mask & mask));
    const std::uint64_t expected = count
        ? ((std::uint64_t{1u} << count) - 1u)
              << (level_base + slots - count)
        : 0u;
    if ((old_mask & mask) != expected)
      throw std::invalid_argument("noncanonical forest mask");
    for (std::uint32_t local = 0u; local < slots; ++local) {
      const std::uint32_t level = level_base + local;
      if (old_mask & (std::uint64_t{1u} << level)) {
        levels[level].occupied = true;
        levels[level].resident_sources = std::uint64_t{1u} << level;
      }
    }
    level_base += slots;
  }

  for (std::uint64_t epoch = 0u; epoch < complete_epochs; ++epoch) {
    PlannerNode incoming{};
    incoming.occupied = true;
    incoming.created = true;
    incoming.raw_begin = epoch;
    incoming.raw_epochs = 1u;
    level_base = 0u;
    bool placed = false;
    for (std::uint32_t tier = 0u; tier < tier_slots.size(); ++tier) {
      const std::uint32_t slots = tier_slots[tier];
      std::uint32_t count = 0u;
      for (std::uint32_t local = 0u; local < slots; ++local)
        count += levels[level_base + local].occupied;
      if (count < slots) {
        const std::uint32_t destination = level_base + slots - 1u - count;
        levels[destination] = incoming;
        placed = true;
        break;
      }
      PlannerNode parent = incoming;
      std::uint64_t minimum = incoming.raw_epochs
          ? incoming.raw_begin : ~std::uint64_t{0};
      std::uint64_t maximum = incoming.raw_begin + incoming.raw_epochs;
      std::uint64_t total_raw = incoming.raw_epochs;
      for (std::uint32_t local = 0u; local < slots; ++local) {
        const PlannerNode &source = levels[level_base + local];
        parent.resident_sources |= source.resident_sources;
        if (source.raw_epochs) {
          minimum = std::min(minimum, source.raw_begin);
          maximum = std::max(
              maximum, source.raw_begin + source.raw_epochs);
          total_raw += source.raw_epochs;
        }
        levels[level_base + local] = {};
      }
      if (total_raw) {
        if (maximum - minimum != total_raw)
          throw std::logic_error("noncontiguous raw carry interval");
        parent.raw_begin = minimum;
        parent.raw_epochs = total_raw;
      }
      if (tier + 1u == tier_slots.size()) {
        if (slots != 1u)
          throw std::logic_error("terminal rollover requires one slot");
        levels[level_base] = parent;
        placed = true;
        break;
      }
      incoming = parent;
      level_base += slots;
    }
    if (!placed) throw std::overflow_error("forest planner overflow");
  }

  ForestPlan plan{};
  plan.old_mask = old_mask;
  level_base = 0u;
  for (std::uint32_t tier = 0u; tier < tier_slots.size(); ++tier) {
    for (std::uint32_t local = 0u; local < tier_slots[tier]; ++local) {
      const std::uint32_t level = level_base + local;
      const PlannerNode &node = levels[level];
      if (!node.occupied) continue;
      plan.final_mask |= std::uint64_t{1u} << level;
      if (!node.created) continue;
      plan.output_mask |= std::uint64_t{1u} << level;
      plan.consumed_mask |= node.resident_sources;
      plan.outputs[plan.output_count++] = {
          node.resident_sources, node.raw_begin, node.raw_epochs, level};
    }
    level_base += tier_slots[tier];
  }
  if (plan.final_mask !=
      ((old_mask & ~plan.consumed_mask) | plan.output_mask))
    throw std::logic_error("forest plan decomposition mismatch");
  return plan;
}

// Sparse descriptors hang off the authoritative GPULSMOpt level/pending
// state.  They describe only exceptional storage; ordinary masks, rows,
// offsets, and counts remain owned by GPULSMOpt.
struct RootBuildState {
  std::uint32_t level{};
  std::uint32_t generation{};
  std::uint32_t row_count{};
  std::uint32_t logical_count{};
  std::uint32_t exact_head_count{};
  std::uint32_t special_head_count{};
  std::uint64_t first_epoch{};
  std::uint64_t last_epoch{};
  std::uint64_t physical_begin{};
  std::uint64_t exact_heads{};
  std::uint64_t exact_rows{};
  std::uint64_t special_heads{};
  std::uint64_t capsule_pages{};
  std::uint64_t capsule_indexes{};
  std::uint32_t exact_row_count{};
  std::uint32_t capsule_page_count{};
  std::uint32_t capsule_count{};
  std::uint32_t reserved{};
  std::uint64_t capsule_live_bytes{};
  std::uint64_t capsule_garbage_bytes{};
};

struct PendingSlotBuildState {
  std::uint64_t pages{};
  std::uint64_t indexes{};
  std::uint64_t exact_heads{};
  std::uint64_t exact_refs{};
  std::uint64_t special_heads{};
  std::uint64_t exceptions{};
  std::uint32_t page_count{};
  std::uint32_t capsule_count{};
  std::uint32_t exact_head_count{};
  std::uint32_t exact_ref_count{};
  std::uint32_t special_head_count{};
  std::uint32_t exception_count{};
  std::uint32_t record_count{};
  std::uint32_t segment_ordinal{};
  std::uint32_t generation{};
  std::uint32_t reserved{};
  std::uint64_t live_bytes{};
  std::uint64_t garbage_bytes{};
};

struct PendingBuildState {
  PendingSlotBuildState
      slots[gpulsmopt2_detail::kBatchesPerEpoch]{};
  std::uint32_t generation{};
  std::uint32_t batch_count{};
  std::uint32_t record_count{};
  std::uint32_t reserved{};
  std::uint64_t live_bytes{};
  std::uint64_t garbage_bytes{};
};

struct DeviceSparseLevelState {
  std::uint64_t state{};
  std::uint32_t generation{};
  std::uint32_t flags{};
};

enum : std::uint32_t {
  kSparseHasExactHeads = 1u << 0u,
  kSparseHasCapsules = 1u << 1u,
};

// This table is indexed by the restored manifest generation.  It contains no
// ordinary mask, row, count, offset, or routing metadata.
struct DeviceSparseManifest {
  DeviceSparseLevelState levels[gpulsmopt2_detail::kMaximumLevels]{};
  std::uint64_t exact_level_mask{};
  std::uint64_t capsule_level_mask{};
  std::uint32_t generation{};
  std::uint32_t reserved{};
};

using Mutation = DeviceMutation;
using ByteSource = DeviceByteSource;

struct RecordBatchView {
  ByteSource keys{};
  ByteSource values{};
  const std::uint8_t *operations{};
  const std::uint32_t *range_contributions{};
  std::uint64_t count{};
  bool head4_words{};
  bool inline_u32_values{};
  Mutation uniform_operation{Mutation::put};
};

inline RecordBatchView record_batch_view(const DeviceRecordBatchView &source) {
  return {source.keys, source.values, source.operations,
          source.range_contributions, source.count,
          source.key_encoding == DeviceKeyEncoding::head4_words,
          !source.values.offsets &&
              source.values.stride == sizeof(std::uint32_t),
          source.uniform_operation};
}

inline RecordBatchView key_batch_view(const DeviceKeyBatchView &source) {
  return {source.keys, {}, nullptr, nullptr, source.count,
          source.key_encoding == DeviceKeyEncoding::head4_words,
          false, Mutation::put};
}

struct DeviceKeyCursor {
  ByteSource source{};
  std::uint64_t row{};
  bool head4_words{};

  __host__ __device__ std::uint64_t begin() const {
    return source.offsets ? source.offsets[row] : row * source.stride;
  }

  __host__ __device__ std::uint64_t length() const {
    return source.offsets ? source.offsets[row + 1u] - source.offsets[row]
                          : source.stride;
  }

  __host__ __device__ std::uint8_t byte(std::uint64_t position) const {
    if (head4_words) {
      const std::uint32_t key =
          reinterpret_cast<const std::uint32_t *>(source.bytes)[row];
      return static_cast<std::uint8_t>(
          key >> (24u - static_cast<std::uint32_t>(position) * 8u));
    }
    return source.bytes[begin() + position];
  }

  __host__ __device__ std::uint32_t head4() const {
    if (head4_words)
      return reinterpret_cast<const std::uint32_t *>(source.bytes)[row];
    const std::uint64_t size = length();
    std::uint32_t head = 0u;
#pragma unroll
    for (std::uint32_t position = 0u; position < 4u; ++position) {
      head <<= 8u;
      if (position < size) head |= byte(position);
    }
    return head;
  }
};

struct DeviceValueCursor {
  ByteSource source{};
  std::uint64_t row{};

  __host__ __device__ std::uint64_t begin() const {
    return source.offsets ? source.offsets[row] : row * source.stride;
  }

  __host__ __device__ std::uint64_t length() const {
    return source.offsets ? source.offsets[row + 1u] - source.offsets[row]
                          : source.stride;
  }

  __host__ __device__ std::uint8_t byte(std::uint64_t position) const {
    return source.bytes[begin() + position];
  }

  __host__ __device__ std::uint32_t inline_word() const {
    std::uint32_t value = 0u;
    const std::uint64_t size = length();
#pragma unroll
    for (std::uint32_t position = 0u; position < 4u; ++position)
      if (position < size)
        value |= std::uint32_t{byte(position)} << (8u * position);
    return value;
  }
};

__host__ __device__ inline bool record_is_tombstone(
    const RecordBatchView &source, std::uint64_t row) {
  if (source.operations)
    return source.operations[row] ==
        static_cast<std::uint8_t>(Mutation::erase);
  return source.uniform_operation == Mutation::erase;
}

template <class LeftCursor, class RightCursor>
__host__ __device__ inline int compare_keys(
    const LeftCursor &left, const RightCursor &right,
    std::uint64_t depth = 0u) {
  const std::uint64_t left_length = left.length();
  const std::uint64_t right_length = right.length();
  const std::uint64_t common =
      left_length < right_length ? left_length : right_length;
  for (std::uint64_t position = depth; position < common; ++position) {
    const std::uint8_t a = left.byte(position);
    const std::uint8_t b = right.byte(position);
    if (a != b) return a < b ? -1 : 1;
  }
  return left_length == right_length ? 0
      : left_length < right_length ? -1 : 1;
}

template <class KeyCursor>
__host__ __device__ inline std::uint32_t rejection_hash(
    const KeyCursor &key) {
  std::uint32_t hash = 2166136261u;
  for (std::uint64_t position = 0u; position < key.length(); ++position)
    hash = (hash ^ key.byte(position)) * 16777619u;
  return (hash ^ static_cast<std::uint32_t>(key.length())) * 16777619u;
}

inline RecordBatchView inline_u32_batch(
    const std::uint32_t *keys, const std::uint32_t *values,
    std::uint64_t count, Mutation operation = Mutation::put) {
  return {
      {reinterpret_cast<const std::uint8_t *>(keys), nullptr,
       sizeof(std::uint32_t)},
      {reinterpret_cast<const std::uint8_t *>(values), nullptr,
       sizeof(std::uint32_t)},
      nullptr, values, count, true, true, operation};
}

}  // namespace gpulsm_sparse
