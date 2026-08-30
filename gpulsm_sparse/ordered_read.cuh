#pragma once

#include "read.cuh"

namespace gpulsm_sparse {

struct SparseSuccessorOutput {
  std::uint32_t *head4_words{};
  DeviceByteSink keys{};
  std::uint64_t *key_lengths{};
  std::uint8_t *found{};
  std::uint8_t *overflow{};
};

struct HeadKeyCursor {
  std::uint32_t head{};

  __device__ std::uint32_t head4() const { return head; }
  __device__ std::uint64_t length() const { return 4u; }
  __device__ std::uint8_t byte(std::uint64_t position) const {
    return static_cast<std::uint8_t>(
        head >> (24u - static_cast<std::uint32_t>(position) * 8u));
  }
};

struct SuccessorKeyCursor {
  // 1: ordinary four-byte head, 2: pending complete key,
  // 3: resident complete key.
  std::uint32_t kind{};
  std::uint32_t head{};
  PendingReadView pending{};
  std::uint32_t batch{};
  std::uint32_t local{};
  gpulsmopt2_detail::Row row{};
  std::uint64_t physical{};
  DeviceCapsulePlane capsules{};

  __device__ std::uint32_t head4() const { return head; }
  __device__ std::uint64_t length() const {
    if (kind == 1u) return 4u;
    if (kind == 2u)
      return PendingRecordCursor{pending, batch, local}.length();
    return ResidentKeyCursor{head, physical, row, capsules}.length();
  }
  __device__ std::uint8_t byte(std::uint64_t position) const {
    if (kind == 1u)
      return HeadKeyCursor{head}.byte(position);
    if (kind == 2u)
      return PendingRecordCursor{pending, batch, local}.byte(position);
    return ResidentKeyCursor{head, physical, row, capsules}.byte(position);
  }
};

__device__ __forceinline__ PendingRecordCursor pending_cursor_from_ref(
    PendingReadView pending, std::uint32_t ref) {
  return {
      pending, ref >> gpulsmopt2_detail::kBatchPositionBits,
      ref & ((1u << gpulsmopt2_detail::kBatchPositionBits) - 1u)};
}

__device__ __forceinline__ const std::uint32_t *pending_exact_refs(
    const PendingSlotBuildState &slot) {
  return reinterpret_cast<const std::uint32_t *>(slot.exact_refs);
}

template <class Candidate>
__device__ __forceinline__ bool successor_candidate_is_eligible(
    const Candidate &candidate, const DeviceKeyCursor &query,
    bool have_after, const SuccessorKeyCursor &after) {
  if (candidate.head4() == query.head4() &&
      compare_keys(candidate, query) < 0)
    return false;
  return !have_after || compare_keys(candidate, after) > 0;
}

template <class Candidate>
__device__ __forceinline__ void consider_exact_successor_candidate(
    const Candidate &candidate, const SuccessorKeyCursor &encoded,
    const DeviceKeyCursor &query, bool have_after,
    const SuccessorKeyCursor &after, bool &have_best,
    SuccessorKeyCursor &best) {
  if (!successor_candidate_is_eligible(
          candidate, query, have_after, after))
    return;
  if (!have_best || compare_keys(candidate, best) < 0) {
    best = encoded;
    have_best = true;
  }
}

__device__ __forceinline__ bool next_sparse_exact_head(
    std::uint32_t lower, PendingReadView pending,
    const DeviceSparseManifest &sparse, std::uint32_t &result) {
  bool found = false;
  std::uint32_t best = 0u;
  for (std::uint32_t slot = 0u; slot < pending.batch_count; ++slot) {
    const PendingSlotBuildState &state = pending.slots[slot];
    if (state.generation != pending.generation ||
        !state.exact_head_count)
      continue;
    const auto *heads =
        reinterpret_cast<const PendingExactHeadDescriptor *>(
            state.exact_heads);
    std::uint32_t low = 0u;
    std::uint32_t high = state.exact_head_count;
    while (low < high) {
      const std::uint32_t middle = low + ((high - low) >> 1u);
      if (heads[middle].head < lower) low = middle + 1u;
      else high = middle;
    }
    if (low < state.exact_head_count &&
        (!found || heads[low].head < best)) {
      best = heads[low].head;
      found = true;
    }
  }
  std::uint64_t levels = sparse.exact_level_mask;
  while (levels) {
    const std::uint32_t level = __ffsll(levels) - 1u;
    levels &= levels - 1u;
    const RootBuildState *root = sparse_root(sparse, level);
    if (!root || !root->exact_head_count) continue;
    const auto *heads = reinterpret_cast<const ExactHeadDescriptor *>(
        root->exact_heads);
    std::uint32_t low = 0u;
    std::uint32_t high = root->exact_head_count;
    while (low < high) {
      const std::uint32_t middle = low + ((high - low) >> 1u);
      if (heads[middle].head < lower) low = middle + 1u;
      else high = middle;
    }
    if (low < root->exact_head_count &&
        (!found || heads[low].head < best)) {
      best = heads[low].head;
      found = true;
    }
  }
  if (found) result = best;
  return found;
}

__device__ __forceinline__ bool first_original_successor_head(
    std::uint32_t lower, PendingReadView pending,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const gpulsmopt2_detail::RouteHeader *route_headers,
    const gpulsmopt2_detail::RouteSlice *route_slices,
    std::uint32_t active_levels, std::uint64_t occupied_levels,
    std::uint32_t &result) {
  const std::uint32_t first_section = lower >> 16u;
  for (std::uint32_t section = first_section;
       section < kSections; ++section) {
    const std::uint32_t section_lower = section == first_section
        ? lower : section << 16u;
    if (gpulsmopt2_detail::first_visible_in_quotient(
            section, section_lower, pending.keys, pending.payloads,
            pending.offsets, pending.batch_capacity, pending.batch_count,
            arena, descriptors, route_headers, route_slices, active_levels,
            occupied_levels, result))
      return true;
  }
  return false;
}

__device__ __forceinline__ std::uint32_t first_eligible_pending_key(
    PendingReadView pending, const std::uint32_t *refs,
    std::uint32_t ref_begin, std::uint32_t count,
    const DeviceKeyCursor &query, bool have_after,
    const SuccessorKeyCursor &after) {
  std::uint32_t low = 0u;
  std::uint32_t high = count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    const auto candidate = pending_cursor_from_ref(
        pending, refs[ref_begin + middle]);
    const bool before_query = candidate.head4() == query.head4() &&
        compare_keys(candidate, query) < 0;
    const bool before_after =
        have_after && compare_keys(candidate, after) <= 0;
    if (before_query || before_after) low = middle + 1u;
    else high = middle;
  }
  return low;
}

__device__ __forceinline__ std::uint32_t first_eligible_resident_key(
    std::uint32_t head, const gpulsmopt2_detail::Row *rows,
    const ExactHeadDescriptor &chunk, DeviceCapsulePlane capsules,
    const DeviceKeyCursor &query, bool have_after,
    const SuccessorKeyCursor &after) {
  std::uint32_t low = 0u;
  std::uint32_t high = chunk.count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    const std::uint64_t physical = chunk.physical_begin + middle;
    const auto row = rows[exact_physical_offset(physical)];
    const ResidentKeyCursor candidate{head, physical, row, capsules};
    const bool before_query = candidate.head4() == query.head4() &&
        compare_keys(candidate, query) < 0;
    const bool before_after =
        have_after && compare_keys(candidate, after) <= 0;
    if (before_query || before_after) low = middle + 1u;
    else high = middle;
  }
  return low;
}

__device__ __forceinline__ bool next_exact_key_at_head(
    std::uint32_t head, const DeviceKeyCursor &query,
    bool have_after, const SuccessorKeyCursor &after,
    PendingReadView pending, const DeviceSparseManifest &sparse,
    bool &have_best, SuccessorKeyCursor &best) {
  have_best = false;
  const HeadKeyCursor short_key{head};
  const SuccessorKeyCursor short_encoded{1u, head};
  consider_exact_successor_candidate(
      short_key, short_encoded, query, have_after, after,
      have_best, best);

  for (std::uint32_t slot = 0u; slot < pending.batch_count; ++slot) {
    const PendingSlotBuildState &state = pending.slots[slot];
    if (state.generation != pending.generation) continue;
    const PendingExactHeadDescriptor *chunk =
        find_pending_exact_head(state, head);
    if (!chunk) continue;
    const std::uint32_t *refs = pending_exact_refs(state);
    const std::uint32_t local = first_eligible_pending_key(
        pending, refs, chunk->ref_begin, chunk->count, query,
        have_after, after);
    if (local >= chunk->count) continue;
    const PendingRecordCursor candidate = pending_cursor_from_ref(
        pending, refs[chunk->ref_begin + local]);
    const SuccessorKeyCursor encoded{
        2u, head, pending, candidate.batch, candidate.local};
    consider_exact_successor_candidate(
        candidate, encoded, query, have_after, after, have_best, best);
  }

  std::uint64_t levels = sparse.exact_level_mask;
  while (levels) {
    const std::uint32_t level = __ffsll(levels) - 1u;
    levels &= levels - 1u;
    const RootBuildState *root = sparse_root(sparse, level);
    if (!root) continue;
    const ExactHeadDescriptor *chunk = find_exact_head(*root, head);
    if (!chunk) continue;
    const auto *rows = reinterpret_cast<const gpulsmopt2_detail::Row *>(
        root->exact_rows);
    const DeviceCapsulePlane capsules = root_capsule_plane(root);
    const std::uint32_t local = first_eligible_resident_key(
        head, rows, *chunk, capsules, query, have_after, after);
    if (local >= chunk->count) continue;
    const std::uint64_t physical = chunk->physical_begin + local;
    const auto row = rows[exact_physical_offset(physical)];
    const ResidentKeyCursor candidate{head, physical, row, capsules};
    const SuccessorKeyCursor encoded{
        3u, head, {}, 0u, 0u, row, physical, capsules};
    consider_exact_successor_candidate(
        candidate, encoded, query, have_after, after, have_best, best);
  }
  return have_best;
}

struct ExactSuccessorAssignment {
  bool found{};
  bool tombstone{};
  SuccessorKeyCursor key{};
};

template <class KeyCursor>
__device__ __forceinline__ ExactSuccessorAssignment
resolve_exact_successor_assignment(
    const KeyCursor &key, PendingReadView pending,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const std::uint16_t *cell_ranks, std::uint64_t occupied_levels,
    const DeviceSparseManifest &sparse) {
  const std::uint32_t section = key.head4() >> 16u;
  for (int batch = static_cast<int>(pending.batch_count) - 1;
       batch >= 0; --batch) {
    const std::uint32_t slot = static_cast<std::uint32_t>(batch);
    const std::size_t oi =
        std::size_t{slot} * (kSections + 1u) + section;
    const std::uint32_t begin = pending.offsets[oi];
    const std::uint32_t end = pending.offsets[oi + 1u];
    bool found = false;
    std::uint32_t newest = 0u;
    PendingRecordCursor winner{};
    for (std::uint32_t local = begin; local < end; ++local) {
      const PendingRecordCursor candidate{pending, slot, local};
      if (candidate.head4() != key.head4() ||
          !read_key_equal(candidate, key))
        continue;
      const std::uint32_t age =
          candidate.payload().metadata & kPendingAgeMask;
      if (!found || age > newest) {
        found = true;
        newest = age;
        winner = candidate;
      }
    }
    if (found) {
      const auto payload = winner.payload();
      return {
          true,
          (payload.metadata & gpulsmopt2_detail::kRawTombstone) != 0u,
          {2u, key.head4(), pending, winner.batch, winner.local}};
    }
  }
  const ResidentMatch resident = find_resident_exact(
      key, arena, descriptors, cell_ranks, occupied_levels, sparse);
  if (!resident.found) return {};
  return {
      true,
      (resident.row.flags & gpulsmopt2_detail::kTombstone) != 0u,
      {3u, key.head4(), {}, 0u, 0u, resident.row,
       resident.physical, resident.capsules}};
}

__device__ __forceinline__ bool refine_exact_successor_head(
    std::uint32_t head, const DeviceKeyCursor &query,
    PendingReadView pending, gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const std::uint16_t *cell_ranks, std::uint64_t occupied_levels,
    const DeviceSparseManifest &sparse, SuccessorKeyCursor &result) {
  bool have_after = false;
  SuccessorKeyCursor after{};
  for (;;) {
    bool have_candidate = false;
    SuccessorKeyCursor candidate{};
    next_exact_key_at_head(
        head, query, have_after, after, pending, sparse,
        have_candidate, candidate);
    if (!have_candidate) return false;
    const auto assignment = resolve_exact_successor_assignment(
        candidate, pending, arena, descriptors, cell_ranks,
        occupied_levels, sparse);
    if (assignment.found && !assignment.tombstone) {
      result = assignment.key;
      return true;
    }
    after = candidate;
    have_after = true;
  }
}

__device__ __forceinline__ void write_successor_missing(
    std::uint32_t output, SparseSuccessorOutput destination) {
  if (destination.head4_words)
    destination.head4_words[output] = gpulsmopt2_detail::kInvalid;
  if (destination.found) destination.found[output] = 0u;
  if (destination.key_lengths) destination.key_lengths[output] = 0u;
  if (destination.overflow) destination.overflow[output] = 0u;
}

template <class KeyCursor>
__device__ __forceinline__ void write_successor_key(
    std::uint32_t output, const KeyCursor &key,
    SparseSuccessorOutput destination) {
  if (destination.head4_words)
    destination.head4_words[output] = key.head4();
  if (destination.found) destination.found[output] = 1u;
  const std::uint64_t length = key.length();
  if (destination.key_lengths) destination.key_lengths[output] = length;
  bool overflow = false;
  if (destination.keys.bytes) {
    std::uint64_t begin = 0u;
    std::uint64_t capacity = 0u;
    if (destination.keys.layout == DeviceSinkLayout::packed) {
      if (!destination.keys.offsets) {
        overflow = true;
      } else {
        begin = destination.keys.offsets[output];
        const std::uint64_t end = destination.keys.offsets[output + 1u];
        overflow = end < begin || end > destination.keys.capacity_bytes;
        if (!overflow) capacity = end - begin;
      }
    } else {
      begin = std::uint64_t{output} * destination.keys.stride;
      overflow = begin > destination.keys.capacity_bytes;
      if (!overflow) {
        const std::uint64_t remaining =
            destination.keys.capacity_bytes - begin;
        capacity = remaining < destination.keys.stride
            ? remaining : destination.keys.stride;
      }
    }
    overflow = overflow || length > capacity;
    if (!overflow)
      for (std::uint64_t position = 0u; position < length; ++position)
        destination.keys.bytes[begin + position] = key.byte(position);
  }
  if (destination.overflow)
    destination.overflow[output] = static_cast<std::uint8_t>(overflow);
}

__global__ void sparse_successor_kernel(
    RecordBatchView queries, SparseSuccessorOutput output,
    PendingReadView pending, gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const gpulsmopt2_detail::RouteHeader *route_headers,
    const gpulsmopt2_detail::RouteSlice *route_slices,
    const std::uint16_t *cell_ranks,
    const std::uint64_t *occupied_mask,
    const DeviceSparseManifest *sparse_manifests,
    const std::uint32_t *active_manifest) {
  const std::uint32_t query_index =
      blockIdx.x * blockDim.x + threadIdx.x;
  if (query_index >= queries.count) return;
  const DeviceKeyCursor query{
      queries.keys, query_index, queries.head4_words};
  const std::uint32_t active = __ldg(active_manifest) & 1u;
  const DeviceSparseManifest &sparse = sparse_manifests[active];
  const std::uint64_t occupied_levels = __ldg(occupied_mask);
  const std::uint32_t active_levels = occupied_levels
      ? 64u - static_cast<std::uint32_t>(__clzll(occupied_levels)) : 0u;
  std::uint32_t lower = query.head4();

  for (;;) {
    std::uint32_t ordinary_head = 0u;
    const bool have_ordinary = first_original_successor_head(
        lower, pending, arena, descriptors, route_headers, route_slices,
        active_levels, occupied_levels, ordinary_head);
    std::uint32_t exact_head = 0u;
    const bool have_exact =
        next_sparse_exact_head(lower, pending, sparse, exact_head);
    if (!have_ordinary && !have_exact) {
      write_successor_missing(query_index, output);
      return;
    }
    const std::uint32_t candidate_head = !have_ordinary ? exact_head
        : !have_exact ? ordinary_head
        : ordinary_head < exact_head ? ordinary_head : exact_head;
    if (have_exact && exact_head == candidate_head) {
      SuccessorKeyCursor result{};
      if (refine_exact_successor_head(
              candidate_head, query, pending, arena, descriptors,
              cell_ranks, occupied_levels, sparse, result)) {
        write_successor_key(query_index, result, output);
        return;
      }
    } else {
      const HeadKeyCursor ordinary{candidate_head};
      if (compare_keys(ordinary, query) >= 0) {
        write_successor_key(query_index, ordinary, output);
        return;
      }
    }
    if (candidate_head == std::numeric_limits<std::uint32_t>::max()) {
      write_successor_missing(query_index, output);
      return;
    }
    lower = candidate_head + 1u;
  }
}

}  // namespace gpulsm_sparse
