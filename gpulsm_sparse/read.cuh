#pragma once

#include "capsule.cuh"

#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

namespace gpulsm_sparse {

// The restored FliX result plane and the optional complete-value sink share
// one final-store helper.  Sparse reads never allocate an intermediate
// `found` array.
struct SparseLookupOutput {
  std::uint32_t *summaries{};
  DeviceByteSink values{};
  std::uint64_t *value_lengths{};
  std::uint8_t *found{};
  std::uint8_t *overflow{};
};

struct InlineWordCursor {
  std::uint32_t word{};
  std::uint64_t bytes{4u};

  __device__ std::uint64_t length() const { return bytes; }
  __device__ std::uint8_t byte(std::uint64_t position) const {
    return static_cast<std::uint8_t>(word >> (8u * position));
  }
};

struct CapsuleValueCursor {
  const CapsuleHeader *object{};

  __device__ std::uint64_t length() const {
    return object ? object->value_length : 0u;
  }
  __device__ std::uint8_t byte(std::uint64_t position) const {
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(object + 1u);
    return bytes[object->key_length + position];
  }
};

__device__ __forceinline__ void write_lookup_missing(
    std::uint32_t output, SparseLookupOutput destination) {
  if (destination.summaries)
    destination.summaries[output] = destination.found
        ? 0u : gpulsmopt2_detail::kInvalid;
  if (destination.found) destination.found[output] = 0u;
  if (destination.value_lengths) destination.value_lengths[output] = 0u;
  if (destination.overflow) destination.overflow[output] = 0u;
}

template <class ValueCursor>
__device__ __forceinline__ void write_lookup_live(
    std::uint32_t output, std::uint32_t summary,
    const ValueCursor &value, SparseLookupOutput destination) {
  if (destination.summaries) destination.summaries[output] = summary;
  if (destination.found) destination.found[output] = 1u;
  const std::uint64_t length = value.length();
  if (destination.value_lengths) destination.value_lengths[output] = length;

  bool overflow = false;
  if (destination.values.bytes) {
    std::uint64_t begin = 0u;
    std::uint64_t capacity = 0u;
    if (destination.values.layout == DeviceSinkLayout::packed) {
      if (!destination.values.offsets) {
        overflow = true;
      } else {
        begin = destination.values.offsets[output];
        const std::uint64_t end = destination.values.offsets[output + 1u];
        overflow = end < begin || end > destination.values.capacity_bytes;
        if (!overflow) capacity = end - begin;
      }
    } else {
      begin = std::uint64_t{output} * destination.values.stride;
      overflow = begin > destination.values.capacity_bytes;
      if (!overflow) {
        const std::uint64_t remaining =
            destination.values.capacity_bytes - begin;
        capacity = remaining < destination.values.stride
            ? remaining : destination.values.stride;
      }
    }
    overflow = overflow || length > capacity;
    if (!overflow)
      for (std::uint64_t position = 0u; position < length; ++position)
        destination.values.bytes[begin + position] = value.byte(position);
  }
  if (destination.overflow)
    destination.overflow[output] = static_cast<std::uint8_t>(overflow);
}

struct PendingReadView {
  const std::uint32_t *keys{};
  const gpulsmopt2_detail::RawPayload *payloads{};
  const std::uint32_t *offsets{};
  const PendingSlotBuildState *slots{};
  std::uint32_t batch_capacity{};
  std::uint32_t batch_count{};
  std::uint32_t generation{};
};

struct PendingRecordCursor {
  PendingReadView source{};
  std::uint32_t batch{};
  std::uint32_t local{};

  __device__ std::uint64_t physical() const {
    return std::uint64_t{batch} * source.batch_capacity + local;
  }
  __device__ gpulsmopt2_detail::RawPayload payload() const {
    return source.payloads[physical()];
  }
  __device__ const PendingSlotBuildState *slot() const {
    const PendingSlotBuildState *state = source.slots + batch;
    return state->generation == source.generation ? state : nullptr;
  }
  __device__ const CapsuleIndex *capsule() const {
    const PendingSlotBuildState *state = slot();
    return state ? find_pending_capsule(*state, batch, local) : nullptr;
  }
  __device__ const CapsuleHeader *header() const {
    return capsule_header(capsule());
  }
  __device__ std::uint64_t length() const {
    const CapsuleHeader *object = header();
    return object ? object->key_length : 4u;
  }
  __device__ std::uint8_t byte(std::uint64_t position) const {
    const CapsuleHeader *object = header();
    if (object)
      return reinterpret_cast<const std::uint8_t *>(object + 1u)[position];
    const std::uint32_t head = source.keys[physical()];
    return static_cast<std::uint8_t>(
        head >> (24u - static_cast<std::uint32_t>(position) * 8u));
  }
  __device__ std::uint32_t head4() const {
    return source.keys[physical()];
  }
};

template <class Left, class Right>
__device__ __forceinline__ bool read_key_equal(
    const Left &left, const Right &right) {
  if (left.length() != right.length()) return false;
  if (left.head4() != right.head4()) return false;
  return left.length() == 4u || compare_keys(left, right, 4u) == 0;
}

template <class Cursor>
__device__ __forceinline__ std::uint64_t read_hash_key(
    const Cursor &key, std::uint64_t seed) {
  std::uint64_t hash = seed ^ (key.length() * 0x9e3779b97f4a7c15ull);
  for (std::uint64_t position = 0u; position < key.length(); ++position) {
    hash ^= std::uint64_t{key.byte(position)} + 0x9e3779b97f4a7c15ull +
        (hash << 6u) + (hash >> 2u);
  }
  return gpulsmopt2_detail::tqrj_hash_mix(hash);
}

constexpr std::uint64_t kPendingWinnerLive = std::uint64_t{1u} << 32u;
constexpr std::uint32_t kPendingWinnerOrderShift = 33u;

__device__ __forceinline__ std::uint64_t make_pending_winner(
    const PendingRecordCursor &candidate) {
  const auto payload = candidate.payload();
  const std::uint32_t age = payload.metadata & kPendingAgeMask;
  const std::uint64_t live =
      payload.metadata & gpulsmopt2_detail::kRawTombstone
      ? 0u : kPendingWinnerLive;
  return (std::uint64_t{age + 1u} << kPendingWinnerOrderShift) |
      live | payload.value;
}

__device__ __forceinline__ void write_pending_winner(
    std::uint32_t output, std::uint64_t winner,
    PendingReadView pending, SparseLookupOutput destination) {
  if (!winner || !(winner & kPendingWinnerLive)) {
    write_lookup_missing(output, destination);
    return;
  }
  const std::uint32_t age = static_cast<std::uint32_t>(
      (winner >> kPendingWinnerOrderShift) - 1u);
  const std::uint32_t batch =
      age >> gpulsmopt2_detail::kBatchPositionBits;
  const PendingSlotBuildState *slot = pending.slots + batch;
  if (slot->generation == pending.generation) {
    const auto *exception = find_pending_exception(
        reinterpret_cast<const PendingExceptionRef *>(slot->exceptions),
        slot->exception_count, age);
    if (exception) {
      const PendingRecordCursor candidate{
          pending, batch, exception->physical};
      const auto payload = candidate.payload();
      write_lookup_live(output, payload.value,
                        CapsuleValueCursor{candidate.header()}, destination);
      return;
    }
  }
  const std::uint32_t value = static_cast<std::uint32_t>(winner);
  write_lookup_live(
      output, value, InlineWordCursor{value, 4u}, destination);
}

struct ResidentMatch {
  bool found{};
  gpulsmopt2_detail::Row row{};
  std::uint64_t physical{};
  DeviceCapsulePlane capsules{};
};

__device__ __forceinline__ DeviceCapsulePlane root_capsule_plane(
    const RootBuildState *root) {
  return root ? DeviceCapsulePlane{
      reinterpret_cast<const CapsulePage *>(root->capsule_pages),
      reinterpret_cast<const CapsuleIndex *>(root->capsule_indexes),
      root->capsule_page_count, root->capsule_count} : DeviceCapsulePlane{};
}

__device__ __forceinline__ const RootBuildState *sparse_root(
    const DeviceSparseManifest &manifest, std::uint32_t level) {
  return reinterpret_cast<const RootBuildState *>(
      manifest.levels[level].state);
}

__device__ __forceinline__ const ExactHeadDescriptor *find_exact_head(
    const RootBuildState &root, std::uint32_t head) {
  const auto *descriptors = reinterpret_cast<const ExactHeadDescriptor *>(
      root.exact_heads);
  std::uint32_t low = 0u;
  std::uint32_t high = root.exact_head_count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (descriptors[middle].head < head) low = middle + 1u;
    else high = middle;
  }
  return low < root.exact_head_count && descriptors[low].head == head
      ? descriptors + low : nullptr;
}

__device__ __forceinline__ const PendingExactHeadDescriptor *
find_pending_exact_head(
    const PendingSlotBuildState &slot, std::uint32_t head) {
  const auto *descriptors =
      reinterpret_cast<const PendingExactHeadDescriptor *>(slot.exact_heads);
  std::uint32_t low = 0u;
  std::uint32_t high = slot.exact_head_count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (descriptors[middle].head < head) low = middle + 1u;
    else high = middle;
  }
  return low < slot.exact_head_count && descriptors[low].head == head
      ? descriptors + low : nullptr;
}

__device__ __forceinline__ ResidentMatch find_projection_match(
    std::uint32_t head, std::uint32_t level,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const std::uint16_t *cell_ranks, DeviceCapsulePlane capsules) {
  const std::uint32_t section = head >> 16u;
  const std::uint32_t suffix = head & 0xffffu;
  const auto rows = descriptors[
      gpulsmopt2_detail::descriptor_index(section, level)];
  if (!rows.count()) return {};
  const std::uint32_t cell =
      suffix / gpulsmopt2_detail::kFoundationCellKeys;
  const std::uint16_t *ranks = cell_ranks +
      std::size_t{level} * gpulsmopt2_detail::kLocalRankEntries +
      std::size_t{section} * gpulsmopt2_detail::kFoundationCells;
  const std::uint32_t begin = ranks[cell];
  const std::uint32_t end =
      cell + 1u < gpulsmopt2_detail::kFoundationCells
      ? ranks[cell + 1u] : rows.count();
  const std::uint32_t local = gpulsmopt2_detail::lower_bound_rows(
      arena + rows.offset() + begin, end - begin, suffix);
  if (local >= end - begin) return {};
  const std::uint64_t physical = rows.offset() + begin + local;
  const auto row = arena[physical];
  return row.key == suffix
      ? ResidentMatch{true, row, physical, capsules} : ResidentMatch{};
}

__device__ __forceinline__ ResidentMatch find_resident_ordinary(
    std::uint32_t head, gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const std::uint16_t *cell_ranks, std::uint64_t occupied_levels,
    const DeviceSparseManifest &sparse) {
  while (occupied_levels) {
    const std::uint32_t level = __ffsll(occupied_levels) - 1u;
    occupied_levels &= occupied_levels - 1u;
    const RootBuildState *root = sparse_root(sparse, level);
    const ResidentMatch match = find_projection_match(
        head, level, arena, descriptors, cell_ranks,
        root_capsule_plane(root));
    if (match.found && !anchor_is_exact_group(match.row)) return match;
  }
  return {};
}

template <class QueryCursor>
__device__ __forceinline__ ResidentMatch find_resident_exact(
    const QueryCursor &query, gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const std::uint16_t *cell_ranks, std::uint64_t occupied_levels,
    const DeviceSparseManifest &sparse) {
  const std::uint32_t head = query.head4();
  while (occupied_levels) {
    const std::uint32_t level = __ffsll(occupied_levels) - 1u;
    occupied_levels &= occupied_levels - 1u;
    const RootBuildState *root = sparse_root(sparse, level);
    const DeviceCapsulePlane capsules = root_capsule_plane(root);
    const ExactHeadDescriptor *chunk =
        root ? find_exact_head(*root, head) : nullptr;
    if (chunk) {
      const auto *exact_rows =
          reinterpret_cast<const gpulsmopt2_detail::Row *>(root->exact_rows);
      std::uint32_t low = 0u;
      std::uint32_t high = chunk->count;
      while (low < high) {
        const std::uint32_t middle = low + ((high - low) >> 1u);
        const std::uint64_t physical = chunk->physical_begin + middle;
        const auto row = exact_rows[exact_physical_offset(physical)];
        const ResidentKeyCursor stored{head, physical, row, capsules};
        if (compare_keys(stored, query) < 0) low = middle + 1u;
        else high = middle;
      }
      if (low < chunk->count) {
        const std::uint64_t physical = chunk->physical_begin + low;
        const auto row = exact_rows[exact_physical_offset(physical)];
        const ResidentKeyCursor stored{head, physical, row, capsules};
        if (read_key_equal(query, stored))
          return {true, row, physical, capsules};
      }
      continue;
    }
    const ResidentMatch ordinary = find_projection_match(
        head, level, arena, descriptors, cell_ranks, capsules);
    if (!ordinary.found || anchor_is_exact_group(ordinary.row)) continue;
    const ResidentKeyCursor stored{
        head, ordinary.physical, ordinary.row, capsules};
    if (read_key_equal(query, stored)) return ordinary;
  }
  return {};
}

__device__ __forceinline__ void write_resident_match(
    std::uint32_t output, const ResidentMatch &match,
    SparseLookupOutput destination) {
  if (!match.found ||
      (match.row.flags & gpulsmopt2_detail::kTombstone)) {
    write_lookup_missing(output, destination);
    return;
  }
  if (anchor_has_capsule(match.row)) {
    const CapsuleIndex *index = find_capsule(
        match.capsules, match.physical, anchor_capsule_rank(match.row));
    write_lookup_live(
        output, match.row.value,
        CapsuleValueCursor{capsule_header(index)}, destination);
  } else {
    write_lookup_live(
        output, match.row.value,
        InlineWordCursor{match.row.value, anchor_value_length(match.row)},
        destination);
  }
}

__device__ __forceinline__ bool query_head_requires_exact(
    std::uint32_t head, const PendingSlotBuildState *pending_slots,
    std::uint32_t pending_batches, std::uint32_t pending_generation,
    const DeviceSparseManifest &sparse) {
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const PendingSlotBuildState &slot = pending_slots[batch];
    if (slot.generation == pending_generation && slot.exact_head_count &&
        find_pending_exact_head(slot, head))
      return true;
  }
  std::uint64_t levels = sparse.exact_level_mask;
  while (levels) {
    const std::uint32_t level = __ffsll(levels) - 1u;
    levels &= levels - 1u;
    const RootBuildState *root = sparse_root(sparse, level);
    if (root && find_exact_head(*root, head)) return true;
  }
  return false;
}

__global__ void classify_lookup_queries(
    RecordBatchView queries, const PendingSlotBuildState *pending_slots,
    std::uint32_t pending_batches, std::uint32_t pending_generation,
    const DeviceSparseManifest *sparse_manifests,
    const std::uint32_t *active_manifest, std::uint8_t *exact_flags,
    std::uint32_t *heads) {
  const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= queries.count) return;
  const DeviceKeyCursor query{queries.keys, row, queries.head4_words};
  const std::uint32_t head = query.head4();
  if (heads) heads[row] = head;
  const std::uint32_t active = __ldg(active_manifest) & 1u;
  exact_flags[row] = static_cast<std::uint8_t>(
      query.length() != 4u || query_head_requires_exact(
          head, pending_slots, pending_batches, pending_generation,
          sparse_manifests[active]));
}

__global__ void sparse_canonical_ordinary_lookup_kernel(
    RecordBatchView queries, const std::uint32_t *ordered_heads,
    const std::uint32_t *query_ids, const std::uint8_t *exact_flags,
    std::uint32_t count, SparseLookupOutput output,
    PendingReadView pending, const std::uint64_t *batch_signatures,
    const std::uint64_t *epoch_signatures,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const std::uint16_t *cell_ranks,
    const std::uint64_t *occupied_mask,
    const DeviceSparseManifest *sparse_manifests,
    const std::uint32_t *active_manifest) {
  const std::uint32_t grouped = blockIdx.x * blockDim.x + threadIdx.x;
  if (grouped >= count) return;
  const std::uint32_t original = query_ids ? query_ids[grouped] : grouped;
  if (exact_flags[original]) return;
  const std::uint32_t head = ordered_heads[grouped];
  const std::uint32_t section = head >> 16u;
  const std::uint64_t signature_bits =
      gpulsmopt2_detail::pending_signature_bits(head);
  if (pending.batch_count &&
      (epoch_signatures[section] & signature_bits) == signature_bits) {
    for (int batch = static_cast<int>(pending.batch_count) - 1;
         batch >= 0; --batch) {
      const std::uint32_t slot = static_cast<std::uint32_t>(batch);
      const std::uint64_t signature = batch_signatures[
          std::size_t{slot} * kSections + section];
      if ((signature & signature_bits) != signature_bits) continue;
      const std::size_t offset =
          std::size_t{slot} * (kSections + 1u) + section;
      const std::uint32_t begin = pending.offsets[offset];
      const std::uint32_t end = pending.offsets[offset + 1u];
      bool found = false;
      std::uint32_t newest = 0u;
      std::uint64_t winner = 0u;
      for (std::uint32_t local = begin; local < end; ++local) {
        const PendingRecordCursor candidate{pending, slot, local};
        if (candidate.head4() != head) continue;
        const std::uint32_t age =
            candidate.payload().metadata & kPendingAgeMask;
        if (!found || age > newest) {
          found = true;
          newest = age;
          winner = make_pending_winner(candidate);
        }
      }
      if (found) {
        write_pending_winner(original, winner, pending, output);
        return;
      }
    }
  }
  const std::uint32_t active = __ldg(active_manifest) & 1u;
  write_resident_match(
      original, find_resident_ordinary(
          head, arena, descriptors, cell_ranks, __ldg(occupied_mask),
          sparse_manifests[active]), output);
}

__global__ void sparse_canonical_exact_lookup_kernel(
    RecordBatchView queries, const std::uint32_t *exact_ids,
    const std::uint32_t *exact_count, SparseLookupOutput output,
    PendingReadView pending, gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const std::uint16_t *cell_ranks,
    const std::uint64_t *occupied_mask,
    const DeviceSparseManifest *sparse_manifests,
    const std::uint32_t *active_manifest) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= *exact_count) return;
  const std::uint32_t original = exact_ids[position];
  const DeviceKeyCursor query{queries.keys, original, queries.head4_words};
  const std::uint32_t head = query.head4();
  const std::uint32_t section = head >> 16u;
  for (int batch = static_cast<int>(pending.batch_count) - 1;
       batch >= 0; --batch) {
    const std::uint32_t slot = static_cast<std::uint32_t>(batch);
    const std::size_t offset =
        std::size_t{slot} * (kSections + 1u) + section;
    const std::uint32_t begin = pending.offsets[offset];
    const std::uint32_t end = pending.offsets[offset + 1u];
    bool found = false;
    std::uint32_t newest = 0u;
    std::uint64_t winner = 0u;
    for (std::uint32_t local = begin; local < end; ++local) {
      const PendingRecordCursor candidate{pending, slot, local};
      if (candidate.head4() != head || !read_key_equal(query, candidate))
        continue;
      const std::uint32_t age =
          candidate.payload().metadata & kPendingAgeMask;
      if (!found || age > newest) {
        found = true;
        newest = age;
        winner = make_pending_winner(candidate);
      }
    }
    if (found) {
      write_pending_winner(original, winner, pending, output);
      return;
    }
  }
  const std::uint32_t active = __ldg(active_manifest) & 1u;
  write_resident_match(
      original, find_resident_exact(
          query, arena, descriptors, cell_ranks, __ldg(occupied_mask),
          sparse_manifests[active]), output);
}

template <class CandidateCursor>
__device__ __forceinline__ std::uint32_t sparse_direct_query_owner(
    const CandidateCursor &candidate, RecordBatchView queries,
    const std::uint8_t *exact_flags,
    const std::uint16_t *directory_offsets,
    const std::uint16_t *directory_suffixes,
    const std::uint16_t *directory_owners,
    const std::uint32_t *query_ids, std::uint32_t query_begin) {
  const std::uint32_t suffix = candidate.head4() & 0xffffu;
  const std::uint32_t interval = suffix >> 8u;
  const std::uint32_t end = directory_offsets[interval + 1u];
  for (std::uint32_t position = directory_offsets[interval];
       position < end; ++position) {
    if (directory_suffixes[position] != suffix) continue;
    const std::uint32_t owner = directory_owners[position];
    const std::uint32_t original = query_ids[query_begin + owner];
    if (!exact_flags[original]) return owner;
    const DeviceKeyCursor query{
        queries.keys, original, queries.head4_words};
    if (read_key_equal(candidate, query)) return owner;
  }
  return gpulsmopt2_detail::kInvalid;
}

template <class QueryCursor>
__device__ __forceinline__ void write_sparse_tqrj_result(
    const QueryCursor &query, std::uint32_t output,
    std::uint64_t winner, bool exact, PendingReadView pending,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const std::uint16_t *cell_ranks, std::uint64_t occupied_levels,
    const DeviceSparseManifest &sparse,
    SparseLookupOutput destination) {
  if (winner) {
    write_pending_winner(output, winner, pending, destination);
    return;
  }
  const ResidentMatch match = exact
      ? find_resident_exact(
            query, arena, descriptors, cell_ranks, occupied_levels, sparse)
      : find_resident_ordinary(
            query.head4(), arena, descriptors, cell_ranks,
            occupied_levels, sparse);
  write_resident_match(output, match, destination);
}

// This is the prototype's direct group owner adapted onto the restored TQRJ
// routing.  Exact owners are compared as complete keys before a pending
// winner is accepted; ordinary owners retain head-only matching.
__global__ void sparse_tqrj_direct_lookup_kernel(
    RecordBatchView queries, SparseLookupOutput output,
    const std::uint32_t *grouped_heads,
    const std::uint32_t *active_sections,
    const std::uint32_t *active_counts,
    const std::uint32_t *active_count,
    const std::uint32_t *query_bases,
    const std::uint32_t *query_ids,
    const std::uint8_t *exact_flags, PendingReadView pending,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const std::uint16_t *cell_ranks,
    const std::uint64_t *occupied_mask,
    const DeviceSparseManifest *sparse_manifests,
    const std::uint32_t *active_manifest,
    gpulsmopt2_detail::TqrjHashTask *hash_tasks,
    std::uint32_t *hash_task_count) {
  using BlockScan = cub::BlockScan<std::uint32_t,
                                   gpulsmopt2_detail::kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::uint32_t directory_counts[
      gpulsmopt2_detail::kTqrjDirectoryBins];
  __shared__ std::uint16_t directory_offsets[
      gpulsmopt2_detail::kTqrjDirectoryBins + 1u];
  __shared__ __align__(4) std::uint16_t directory_suffixes[
      gpulsmopt2_detail::kTqrjDirectCapacity];
  __shared__ std::uint16_t directory_owners[
      gpulsmopt2_detail::kTqrjDirectCapacity];
  __shared__ unsigned long long winners[
      gpulsmopt2_detail::kTqrjDirectCapacity];
  __shared__ std::uint32_t pending_rows;
  __shared__ std::uint64_t occupied_levels;
  __shared__ std::uint32_t manifest_index;

  const std::uint32_t task = blockIdx.x;
  if (task >= *active_count) return;
  const std::uint32_t section = active_sections[task];
  const std::uint32_t query_begin = query_bases[section];
  const std::uint32_t query_count = active_counts[task];
  if (threadIdx.x == 0u) {
    pending_rows = gpulsmopt2_detail::tqrj_pending_rows(
        section, pending.offsets, pending.batch_count);
    occupied_levels = __ldg(occupied_mask);
    manifest_index = __ldg(active_manifest) & 1u;
  }
  __syncthreads();
  if (query_count > gpulsmopt2_detail::kTqrjDirectCapacity ||
      pending_rows > gpulsmopt2_detail::kTqrjDirectPendingRows) {
    if (threadIdx.x == 0u)
      gpulsmopt2_detail::tqrj_enqueue_hash_task(
          section, pending_rows, hash_tasks, hash_task_count);
    return;
  }

  directory_counts[threadIdx.x] = 0u;
  __syncthreads();
  for (std::uint32_t local = threadIdx.x; local < query_count;
       local += blockDim.x) {
    const std::uint32_t suffix = grouped_heads[query_begin + local] & 0xffffu;
    atomicAdd(directory_counts + (suffix >> 8u), 1u);
  }
  __syncthreads();
  const std::uint32_t interval_count = directory_counts[threadIdx.x];
  std::uint32_t interval_begin = 0u;
  BlockScan(scan_storage).ExclusiveSum(interval_count, interval_begin);
  directory_offsets[threadIdx.x] =
      static_cast<std::uint16_t>(interval_begin);
  if (threadIdx.x + 1u == gpulsmopt2_detail::kTqrjDirectoryBins)
    directory_offsets[gpulsmopt2_detail::kTqrjDirectoryBins] =
        static_cast<std::uint16_t>(interval_begin + interval_count);
  directory_counts[threadIdx.x] = 0u;
  const bool crowded = __syncthreads_or(
      interval_count > gpulsmopt2_detail::kTqrjDirectIntervalLimit);
  if (crowded) {
    if (threadIdx.x == 0u)
      gpulsmopt2_detail::tqrj_enqueue_hash_task(
          section, pending_rows, hash_tasks, hash_task_count);
    return;
  }

  for (std::uint32_t local = threadIdx.x; local < query_count;
       local += blockDim.x) {
    const std::uint32_t suffix = grouped_heads[query_begin + local] & 0xffffu;
    const std::uint32_t interval = suffix >> 8u;
    const std::uint32_t rank = atomicAdd(directory_counts + interval, 1u);
    const std::uint32_t position = directory_offsets[interval] + rank;
    directory_suffixes[position] = static_cast<std::uint16_t>(suffix);
    directory_owners[position] = static_cast<std::uint16_t>(local);
    winners[local] = 0ull;
  }
  __syncthreads();

  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  constexpr std::uint32_t warps = gpulsmopt2_detail::kThreads / 32u;
  for (std::uint32_t batch = warp; batch < pending.batch_count;
       batch += warps) {
    const std::size_t oi =
        std::size_t{batch} * (kSections + 1u) + section;
    const std::uint32_t begin = pending.offsets[oi];
    const std::uint32_t end = pending.offsets[oi + 1u];
    for (std::uint32_t base = begin; base < end; base += 32u) {
      const std::uint32_t local = base + lane;
      bool matched = false;
      std::uint32_t owner = gpulsmopt2_detail::kInvalid;
      unsigned long long token = 0ull;
      if (local < end) {
        const PendingRecordCursor candidate{pending, batch, local};
        owner = sparse_direct_query_owner(
            candidate, queries, exact_flags, directory_offsets,
            directory_suffixes, directory_owners, query_ids, query_begin);
        matched = owner != gpulsmopt2_detail::kInvalid;
        if (matched) token = make_pending_winner(candidate);
      }
      gpulsmopt2_detail::tqrj_warp_atomic_max(
          matched, owner, token, winners);
    }
  }
  __syncthreads();

  for (std::uint32_t local = threadIdx.x; local < query_count;
       local += blockDim.x) {
    const std::uint32_t grouped = query_begin + local;
    const std::uint32_t original = query_ids[grouped];
    const DeviceKeyCursor query{
        queries.keys, original, queries.head4_words};
    const std::uint32_t owner = sparse_direct_query_owner(
        query, queries, exact_flags, directory_offsets, directory_suffixes,
        directory_owners, query_ids, query_begin);
    const std::uint64_t winner = owner == gpulsmopt2_detail::kInvalid
        ? 0ull : winners[owner];
    write_sparse_tqrj_result(
        query, original, winner, exact_flags[original] != 0u, pending,
        arena, descriptors, cell_ranks, occupied_levels,
        sparse_manifests[manifest_index], output);
  }
}

__device__ __forceinline__ std::uint64_t sparse_query_hash(
    RecordBatchView queries, const std::uint32_t *query_ids,
    const std::uint32_t *grouped_heads, const std::uint8_t *exact_flags,
    std::uint32_t grouped, std::uint64_t seed) {
  const std::uint32_t original = query_ids[grouped];
  if (!exact_flags[original])
    return gpulsmopt2_detail::tqrj_hash_key(grouped_heads[grouped], seed);
  return read_hash_key(
      DeviceKeyCursor{queries.keys, original, queries.head4_words}, seed);
}

__device__ __forceinline__ bool sparse_queries_equal(
    RecordBatchView queries, const std::uint32_t *query_ids,
    const std::uint32_t *grouped_heads, const std::uint8_t *exact_flags,
    std::uint32_t left, std::uint32_t right) {
  const std::uint32_t left_original = query_ids[left];
  const std::uint32_t right_original = query_ids[right];
  const bool exact = exact_flags[left_original] != 0u;
  if (exact != (exact_flags[right_original] != 0u)) return false;
  if (!exact) return grouped_heads[left] == grouped_heads[right];
  return read_key_equal(
      DeviceKeyCursor{queries.keys, left_original, queries.head4_words},
      DeviceKeyCursor{queries.keys, right_original, queries.head4_words});
}

__device__ __forceinline__ std::uint32_t sparse_hash_insert(
    RecordBatchView queries, const std::uint32_t *query_ids,
    const std::uint32_t *grouped_heads, const std::uint8_t *exact_flags,
    std::uint32_t query, std::uint32_t *table, std::uint32_t capacity,
    std::uint64_t seed, std::uint32_t maximum_probes,
    std::uint32_t *failure) {
  const std::uint64_t hash = sparse_query_hash(
      queries, query_ids, grouped_heads, exact_flags, query, seed);
  const std::uint32_t desired =
      gpulsmopt2_detail::tqrj_hash_entry(query, hash);
  const std::uint32_t fingerprint =
      gpulsmopt2_detail::tqrj_hash_entry_fingerprint(desired);
  std::uint32_t slot =
      gpulsmopt2_detail::tqrj_hash_slot(hash, capacity);
  for (std::uint32_t probe = 0u; probe < maximum_probes; ++probe) {
    const std::uint32_t previous = atomicCAS(table + slot, 0u, desired);
    if (!previous) return query;
    if (gpulsmopt2_detail::tqrj_hash_entry_fingerprint(previous) ==
        fingerprint) {
      const std::uint32_t owner =
          gpulsmopt2_detail::tqrj_hash_entry_owner(previous);
      if (sparse_queries_equal(
              queries, query_ids, grouped_heads, exact_flags,
              owner, query))
        return owner;
    }
    if (++slot == capacity) slot = 0u;
  }
  atomicExch(failure, 1u);
  return gpulsmopt2_detail::kInvalid;
}

template <class CandidateCursor>
__device__ __forceinline__ std::uint32_t sparse_hash_find_category(
    const CandidateCursor &candidate, bool exact, RecordBatchView queries,
    const std::uint32_t *query_ids, const std::uint32_t *grouped_heads,
    const std::uint8_t *exact_flags, const std::uint32_t *table,
    std::uint32_t capacity, std::uint64_t seed) {
  const std::uint64_t hash = exact
      ? read_hash_key(candidate, seed)
      : gpulsmopt2_detail::tqrj_hash_key(candidate.head4(), seed);
  const std::uint32_t fingerprint =
      gpulsmopt2_detail::tqrj_hash_entry_fingerprint(
          gpulsmopt2_detail::tqrj_hash_entry(0u, hash));
  std::uint32_t slot =
      gpulsmopt2_detail::tqrj_hash_slot(hash, capacity);
  for (std::uint32_t probe = 0u; probe < capacity; ++probe) {
    const std::uint32_t entry = table[slot];
    if (!entry) return gpulsmopt2_detail::kInvalid;
    if (gpulsmopt2_detail::tqrj_hash_entry_fingerprint(entry) ==
        fingerprint) {
      const std::uint32_t owner =
          gpulsmopt2_detail::tqrj_hash_entry_owner(entry);
      const std::uint32_t original = query_ids[owner];
      if ((exact_flags[original] != 0u) == exact) {
        if (!exact && grouped_heads[owner] == candidate.head4()) return owner;
        if (exact && read_key_equal(
                DeviceKeyCursor{
                    queries.keys, original, queries.head4_words},
                candidate))
          return owner;
      }
    }
    if (++slot == capacity) slot = 0u;
  }
  return gpulsmopt2_detail::kInvalid;
}

template <class CandidateCursor>
__device__ __forceinline__ std::uint32_t sparse_hash_find(
    const CandidateCursor &candidate, RecordBatchView queries,
    const std::uint32_t *query_ids, const std::uint32_t *grouped_heads,
    const std::uint8_t *exact_flags, const std::uint32_t *table,
    std::uint32_t capacity, std::uint64_t seed) {
  const std::uint32_t exact = sparse_hash_find_category(
      candidate, true, queries, query_ids, grouped_heads, exact_flags,
      table, capacity, seed);
  return exact != gpulsmopt2_detail::kInvalid ? exact
      : sparse_hash_find_category(
            candidate, false, queries, query_ids, grouped_heads,
            exact_flags, table, capacity, seed);
}

// The prototype's exact hash matcher is grafted into the restored overflow
// queue.  Ordinary entries remain keyed by four-byte heads; deferred entries
// are keyed and compared by complete key.
__global__ __launch_bounds__(gpulsmopt2_detail::kTqrjHashThreads, 1)
void sparse_tqrj_hash_lookup_kernel(
    gpulsmopt2_detail::TqrjHashTask *tasks,
    const std::uint32_t *task_count,
    const std::uint32_t *query_bases,
    const std::uint32_t *query_counts,
    const std::uint32_t *grouped_heads,
    const std::uint32_t *query_ids, RecordBatchView queries,
    SparseLookupOutput output, const std::uint8_t *exact_flags,
    std::uint32_t *query_owners,
    gpulsmopt2_detail::TqrjHashTile *query_tiles,
    gpulsmopt2_detail::TqrjHashTile *pending_tiles,
    std::uint32_t *counters, std::uint32_t *hash_table,
    std::uint32_t maximum_hash_entries,
    unsigned long long *winners, PendingReadView pending,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const std::uint16_t *cell_ranks,
    const std::uint64_t *occupied_mask,
    const DeviceSparseManifest *sparse_manifests,
    const std::uint32_t *active_manifest) {
  const cooperative_groups::grid_group grid =
      cooperative_groups::this_grid();
  __shared__ std::uint32_t section_prefixes[
      gpulsmopt2_detail::kBatchesPerEpoch + 1u];
  __shared__ gpulsmopt2_detail::TqrjHashTask shared_task;
  __shared__ std::uint32_t shared_a, shared_b;
  __shared__ std::uint64_t occupied_levels;
  __shared__ std::uint32_t manifest_index;
  const std::uint32_t tasks_in_queue = *task_count;
  if (!tasks_in_queue) return;
  const std::uint32_t global_thread =
      blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t global_stride = blockDim.x * gridDim.x;

  for (std::uint32_t task_index = blockIdx.x;
       task_index < tasks_in_queue; task_index += gridDim.x) {
    if (threadIdx.x == 0u) {
      const auto task = tasks[task_index];
      const std::uint32_t count = query_counts[task.quotient];
      shared_a = gpulsmopt2_detail::tqrj_hash_tile_count(count);
      shared_b = gpulsmopt2_detail::tqrj_hash_tile_count(task.pending_rows);
      tasks[task_index].query_tile_base = atomicAdd(counters, shared_a);
      tasks[task_index].pending_tile_base =
          atomicAdd(counters + 1u, shared_b);
      atomicAdd(counters + 2u, count);
    }
    __syncthreads();
    const auto task = tasks[task_index];
    for (std::uint32_t tile = threadIdx.x; tile < shared_a;
         tile += blockDim.x)
      query_tiles[task.query_tile_base + tile] = {
          task_index, tile * gpulsmopt2_detail::kTqrjHashTileRows};
    for (std::uint32_t tile = threadIdx.x; tile < shared_b;
         tile += blockDim.x)
      pending_tiles[task.pending_tile_base + tile] = {
          task_index, tile * gpulsmopt2_detail::kTqrjHashTileRows};
    __syncthreads();
  }
  grid.sync();

  const std::uint32_t query_tile_count = counters[0];
  const std::uint32_t pending_tile_count = counters[1];
  const std::uint32_t hash_capacity =
      gpulsmopt2_detail::tqrj_hash_capacity(counters[2]);
  if (global_thread == 0u) {
    counters[4] = 0u;
    counters[5] = hash_capacity > maximum_hash_entries ? 1u : 0u;
  }
  grid.sync();
  if (counters[5]) return;

  for (std::uint32_t attempt = 0u;
       attempt < gpulsmopt2_detail::kTqrjHashAttempts; ++attempt) {
    if (global_thread == 0u) counters[3] = 0u;
    for (std::uint32_t slot = global_thread; slot < hash_capacity;
         slot += global_stride)
      hash_table[slot] = 0u;
    grid.sync();
    const std::uint64_t seed =
        gpulsmopt2_detail::tqrj_hash_seed(attempt);
    const std::uint32_t probe_limit =
        attempt + 1u == gpulsmopt2_detail::kTqrjHashAttempts
        ? hash_capacity : gpulsmopt2_detail::kTqrjHashProbeLimit;
    for (std::uint32_t tile_index = blockIdx.x;
         tile_index < query_tile_count; tile_index += gridDim.x) {
      const auto tile = query_tiles[tile_index];
      const auto task = tasks[tile.task];
      const std::uint32_t query_begin = query_bases[task.quotient];
      const std::uint32_t query_count = query_counts[task.quotient];
      for (std::uint32_t local = threadIdx.x;
           local < gpulsmopt2_detail::kTqrjHashTileRows;
           local += blockDim.x) {
        const std::uint32_t row = tile.begin + local;
        if (row >= query_count) continue;
        const std::uint32_t query = query_begin + row;
        const std::uint32_t owner = sparse_hash_insert(
            queries, query_ids, grouped_heads, exact_flags, query,
            hash_table, hash_capacity, seed, probe_limit, counters + 3u);
        query_owners[query_ids[query]] = owner;
      }
    }
    grid.sync();
    if (!counters[3]) {
      if (global_thread == 0u) counters[4] = attempt;
      grid.sync();
      break;
    }
    grid.sync();
  }

  for (std::uint32_t tile_index = blockIdx.x;
       tile_index < query_tile_count; tile_index += gridDim.x) {
    const auto tile = query_tiles[tile_index];
    const auto task = tasks[tile.task];
    const std::uint32_t query_begin = query_bases[task.quotient];
    const std::uint32_t query_count = query_counts[task.quotient];
    for (std::uint32_t local = threadIdx.x;
         local < gpulsmopt2_detail::kTqrjHashTileRows;
         local += blockDim.x) {
      const std::uint32_t row = tile.begin + local;
      if (row < query_count) winners[query_begin + row] = 0ull;
    }
  }
  grid.sync();

  const std::uint64_t selected_seed =
      gpulsmopt2_detail::tqrj_hash_seed(counters[4]);
  for (std::uint32_t tile_index = blockIdx.x;
       tile_index < pending_tile_count; tile_index += gridDim.x) {
    const auto tile = pending_tiles[tile_index];
    if (threadIdx.x == 0u) {
      shared_task = tasks[tile.task];
      std::uint32_t prefix = 0u;
      section_prefixes[0] = 0u;
      for (std::uint32_t batch = 0u; batch < pending.batch_count; ++batch) {
        const std::size_t oi =
            std::size_t{batch} * (kSections + 1u) + shared_task.quotient;
        prefix += pending.offsets[oi + 1u] - pending.offsets[oi];
        section_prefixes[batch + 1u] = prefix;
      }
    }
    __syncthreads();
    for (std::uint32_t local = threadIdx.x;
         local < gpulsmopt2_detail::kTqrjHashTileRows;
         local += blockDim.x) {
      const std::uint32_t row = tile.begin + local;
      bool matched = false;
      std::uint32_t owner = gpulsmopt2_detail::kInvalid;
      unsigned long long token = 0ull;
      if (row < shared_task.pending_rows) {
        std::uint32_t low = 0u;
        std::uint32_t high = pending.batch_count;
        while (low + 1u < high) {
          const std::uint32_t middle = (low + high) >> 1u;
          if (section_prefixes[middle] <= row) low = middle;
          else high = middle;
        }
        const std::uint32_t batch = low;
        const std::size_t oi =
            std::size_t{batch} * (kSections + 1u) + shared_task.quotient;
        const std::uint32_t position =
            pending.offsets[oi] + row - section_prefixes[batch];
        const PendingRecordCursor candidate{pending, batch, position};
        owner = sparse_hash_find(
            candidate, queries, query_ids, grouped_heads, exact_flags,
            hash_table, hash_capacity, selected_seed);
        matched = owner != gpulsmopt2_detail::kInvalid;
        if (matched) token = make_pending_winner(candidate);
      }
      gpulsmopt2_detail::tqrj_warp_atomic_max(
          matched, owner, token, winners);
    }
    __syncthreads();
  }
  grid.sync();

  if (threadIdx.x == 0u) {
    occupied_levels = __ldg(occupied_mask);
    manifest_index = __ldg(active_manifest) & 1u;
  }
  __syncthreads();
  for (std::uint32_t tile_index = blockIdx.x;
       tile_index < query_tile_count; tile_index += gridDim.x) {
    const auto tile = query_tiles[tile_index];
    const auto task = tasks[tile.task];
    const std::uint32_t query_begin = query_bases[task.quotient];
    const std::uint32_t query_count = query_counts[task.quotient];
    for (std::uint32_t local = threadIdx.x;
         local < gpulsmopt2_detail::kTqrjHashTileRows;
         local += blockDim.x) {
      const std::uint32_t row = tile.begin + local;
      if (row >= query_count) continue;
      const std::uint32_t grouped = query_begin + row;
      const std::uint32_t original = query_ids[grouped];
      const std::uint32_t owner = query_owners[original];
      const DeviceKeyCursor query{
          queries.keys, original, queries.head4_words};
      write_sparse_tqrj_result(
          query, original, winners[owner], exact_flags[original] != 0u,
          pending, arena, descriptors, cell_ranks, occupied_levels,
          sparse_manifests[manifest_index], output);
    }
    __syncthreads();
  }
}

class SparseReadWorkspace {
 public:
  SparseReadWorkspace() {
    int blocks_per_sm = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks_per_sm, sparse_tqrj_hash_lookup_kernel,
              gpulsmopt2_detail::kTqrjHashThreads, 0u),
          "size sparse exact hash grid");
    int device = 0;
    check(cudaGetDevice(&device), "get sparse read device");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device),
          "get sparse read properties");
    hash_worker_blocks_ = static_cast<std::uint32_t>(
        blocks_per_sm * properties.multiProcessorCount);
    if (!hash_worker_blocks_)
      throw std::runtime_error("sparse exact hash has zero occupancy");
  }

  void ensure(std::uint32_t count, bool need_heads, bool need_owners) {
    if (count > capacity_) {
      capacity_ = count;
      exact_flags_.reset(capacity_);
      exact_ids_.reset(capacity_);
      exact_count_.reset(1u);
      std::size_t bytes = 0u;
      cub::CountingInputIterator<std::uint32_t> ids(0u);
      check(cub::DeviceSelect::Flagged(
                nullptr, bytes, ids, exact_flags_.data(),
                exact_ids_.data(), exact_count_.data(), capacity_),
            "size sparse exact query roster");
      select_temporary_.reset(bytes);
    }
    if (need_heads && heads_.size() < count) heads_.reset(count);
    if (need_owners && owners_.size() < count) owners_.reset(count);
  }

  void classify(
      RecordBatchView queries, const PendingSlotBuildState *pending_slots,
      std::uint32_t pending_batches, std::uint32_t pending_generation,
      const DeviceSparseManifest *sparse_manifests,
      const std::uint32_t *active_manifest, bool materialize_heads,
      cudaStream_t stream) {
    classify_lookup_queries<<<blocks(queries.count), kThreads, 0, stream>>>(
        queries, pending_slots, pending_batches, pending_generation,
        sparse_manifests, active_manifest, exact_flags_.data(),
        materialize_heads ? heads_.data() : nullptr);
    cub::CountingInputIterator<std::uint32_t> ids(0u);
    std::size_t bytes = select_temporary_.bytes();
    check(cub::DeviceSelect::Flagged(
              select_temporary_.data(), bytes, ids, exact_flags_.data(),
              exact_ids_.data(), exact_count_.data(),
              static_cast<std::uint32_t>(queries.count), stream),
          "compact sparse exact query roster");
  }

  std::uint8_t *exact_flags() const { return exact_flags_.data(); }
  std::uint32_t *exact_ids() const { return exact_ids_.data(); }
  std::uint32_t *exact_count() const { return exact_count_.data(); }
  std::uint32_t *heads() const { return heads_.data(); }
  std::uint32_t *owners() const { return owners_.data(); }
  std::uint32_t hash_worker_blocks() const { return hash_worker_blocks_; }
  std::size_t bytes() const {
    return exact_flags_.bytes() + exact_ids_.bytes() +
        exact_count_.bytes() + heads_.bytes() + owners_.bytes() +
        select_temporary_.bytes();
  }

 private:
  std::uint32_t capacity_{};
  std::uint32_t hash_worker_blocks_{};
  Buffer<std::uint8_t> exact_flags_;
  Buffer<std::uint32_t> exact_ids_, exact_count_, heads_, owners_;
  Buffer<std::uint8_t> select_temporary_;
};

}  // namespace gpulsm_sparse
