#pragma once

#include "capsule_lifecycle.cuh"
#include "fan_in.cuh"

namespace gpulsm_sparse {

__device__ __forceinline__ const ExactHeadDescriptor *
find_root_exact_head(const RootBuildState &root, std::uint32_t head) {
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

__device__ __forceinline__ std::uint64_t find_projection_row(
    std::uint32_t head, std::uint32_t level,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors) {
  const std::uint32_t section = head >> 16u;
  const std::uint32_t suffix = head & 0xffffu;
  const auto rows = descriptors[
      gpulsmopt2_detail::descriptor_index(section, level)];
  const std::uint32_t local = gpulsmopt2_detail::lower_bound_rows(
      arena + rows.offset(), rows.count(), suffix);
  return local < rows.count() && arena[rows.offset() + local].key == suffix
      ? rows.offset() + local : ~std::uint64_t{0};
}

// Mixed epochs retain one non-tombstone placeholder for every observed
// exceptional head until exact refinement has resolved all complete keys.
// This is the required pre-winner deferral: a newer different long key may
// not erase the projection needed by an older surviving key.
__global__ void patch_epoch_sparse_placeholders(
    const std::uint32_t *roster, std::uint32_t roster_count,
    const std::uint32_t *section_offsets,
    const std::uint32_t *section_counts,
    gpulsmopt2_detail::Row *epoch_rows,
    gpulsmopt2_detail::ResidentRows resident_rows,
    std::uint64_t resident_begin, bool materialized) {
  const std::uint32_t task = blockIdx.x * blockDim.x + threadIdx.x;
  if (task >= roster_count) return;
  const std::uint32_t head = roster[task];
  const std::uint32_t section = head >> 16u;
  const std::uint32_t suffix = head & 0xffffu;
  const std::uint32_t begin = section_offsets[section];
  const std::uint32_t count = section_counts[section];
  const std::uint32_t local = materialized
      ? gpulsmopt2_detail::lower_bound_rows(
            resident_rows + resident_begin + begin, count, suffix)
      : gpulsmopt2_detail::lower_bound_rows(
            epoch_rows + begin, count, suffix);
  if (local >= count) return;
  const std::uint16_t found = materialized
      ? resident_rows.key_at(resident_begin + begin + local)
      : epoch_rows[begin + local].key;
  if (found != suffix) return;
  const gpulsmopt2_detail::Row sentinel = exact_group_row(head, 0u);
  if (materialized)
    resident_rows.store(resident_begin + begin + local, sentinel);
  else
    epoch_rows[begin + local] = sentinel;
}

__global__ void count_pending_roster_candidates(
    const std::uint32_t *roster, std::uint32_t roster_count,
    const std::uint32_t *incoming_sorted_heads,
    std::uint32_t incoming_records,
    const std::uint32_t *pending_keys,
    const std::uint32_t *pending_offsets,
    std::uint32_t batch_capacity, std::uint32_t batch_count,
    std::uint32_t *counts) {
  const std::uint32_t task = blockIdx.x * blockDim.x + threadIdx.x;
  if (task >= roster_count) return;
  const std::uint32_t head = roster[task];
  const std::uint32_t section = head >> 16u;
  std::uint32_t low = 0u;
  std::uint32_t high = incoming_records;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (incoming_sorted_heads[middle] < head) low = middle + 1u;
    else high = middle;
  }
  const std::uint32_t incoming_begin = low;
  high = incoming_records;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (incoming_sorted_heads[middle] <= head) low = middle + 1u;
    else high = middle;
  }
  std::uint32_t count = low - incoming_begin;
  for (std::uint32_t batch = 0u; batch < batch_count; ++batch) {
    const std::uint32_t *offsets = pending_offsets +
        std::size_t{batch} * (kSections + 1u);
    const std::uint32_t *keys = pending_keys +
        std::size_t{batch} * batch_capacity;
    for (std::uint32_t row = offsets[section];
         row < offsets[section + 1u]; ++row)
      count += keys[row] == head;
  }
  counts[task] = count;
}

__global__ void emit_pending_roster_candidates(
    const std::uint32_t *roster, std::uint32_t roster_count,
    const std::uint32_t *candidate_offsets,
    const std::uint32_t *incoming_sorted_heads,
    const std::uint32_t *incoming_sorted_refs,
    std::uint32_t incoming_records,
    const std::uint32_t *pending_keys,
    const gpulsmopt2_detail::RawPayload *pending_payloads,
    const std::uint32_t *pending_offsets,
    std::uint32_t batch_capacity, std::uint32_t batch_count,
    std::uint32_t *ages, std::uint32_t *refs) {
  const std::uint32_t task = blockIdx.x;
  if (task >= roster_count) return;
  const std::uint32_t head = roster[task];
  const std::uint32_t section = head >> 16u;
  __shared__ std::uint32_t cursor;
  if (!threadIdx.x) cursor = 0u;
  __syncthreads();
  std::uint32_t low = 0u;
  std::uint32_t high = incoming_records;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (incoming_sorted_heads[middle] < head) low = middle + 1u;
    else high = middle;
  }
  const std::uint32_t incoming_begin = low;
  high = incoming_records;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (incoming_sorted_heads[middle] <= head) low = middle + 1u;
    else high = middle;
  }
  const std::uint32_t incoming_end = low;
  for (std::uint32_t sorted = incoming_begin + threadIdx.x;
       sorted < incoming_end; sorted += blockDim.x) {
    const std::uint32_t ref = incoming_sorted_refs
        ? incoming_sorted_refs[sorted] : sorted;
    const std::uint32_t output = candidate_offsets[task] +
        atomicAdd(&cursor, 1u);
    ages[output] = ref;
    refs[output] = kCompletionIncoming | ref;
  }
  __syncthreads();
  for (std::uint32_t batch = 0u; batch < batch_count; ++batch) {
    const std::uint32_t *offsets = pending_offsets +
        std::size_t{batch} * (kSections + 1u);
    const std::uint32_t *keys = pending_keys +
        std::size_t{batch} * batch_capacity;
    const auto *payloads = pending_payloads +
        std::size_t{batch} * batch_capacity;
    for (std::uint32_t row = offsets[section] + threadIdx.x;
         row < offsets[section + 1u]; row += blockDim.x) {
      if (keys[row] != head) continue;
      const std::uint32_t output = candidate_offsets[task] +
          atomicAdd(&cursor, 1u);
      ages[output] = gpulsmopt2_detail::raw_position(payloads[row]);
      refs[output] =
          (batch << gpulsmopt2_detail::kBatchPositionBits) | row;
    }
    __syncthreads();
  }
}

__global__ void count_resident_candidate_streams(
    const std::uint32_t *roster, std::uint32_t roster_count,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const RootBuildState *roots, const std::uint32_t *levels,
    std::uint32_t source_count, unsigned long long *counts) {
  const std::uint32_t source = blockIdx.x;
  if (source >= source_count) return;
  const std::uint32_t level = levels[source];
  unsigned long long local = 0u;
  for (std::uint32_t task = threadIdx.x; task < roster_count;
       task += blockDim.x) {
    const std::uint32_t head = roster[task];
    const ExactHeadDescriptor *exact = find_root_exact_head(
        roots[level], head);
    if (exact)
      local += exact->count;
    else
      local += find_projection_row(head, level, arena, descriptors) !=
          ~std::uint64_t{0};
  }
  for (std::uint32_t offset = 16u; offset; offset >>= 1u)
    local += __shfl_down_sync(0xffffffffu, local, offset);
  if (!(threadIdx.x & 31u))
    atomicAdd(counts + source, local);
}

__global__ void emit_resident_candidate_streams(
    const std::uint32_t *roster, std::uint32_t roster_count,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    const RootBuildState *roots, const std::uint32_t *levels,
    std::uint32_t source_count, const std::uint32_t *stream_begins,
    TerminalCandidate *candidates) {
  const std::uint32_t source = blockIdx.x;
  if (source >= source_count) return;
  const std::uint32_t level = levels[source];
  __shared__ std::uint32_t cursor;
  if (!threadIdx.x) cursor = stream_begins[source];
  __syncthreads();
  for (std::uint32_t task = 0u; task < roster_count; ++task) {
    const std::uint32_t head = roster[task];
    const ExactHeadDescriptor *exact = find_root_exact_head(
        roots[level], head);
    if (exact) {
      const std::uint32_t begin = cursor;
      for (std::uint32_t local = threadIdx.x; local < exact->count;
           local += blockDim.x)
        candidates[begin + local] = {
            exact->physical_begin + local, head, level};
      __syncthreads();
      if (!threadIdx.x) cursor += exact->count;
      __syncthreads();
      continue;
    }
    if (!threadIdx.x) {
      const std::uint64_t ordinary = find_projection_row(
          head, level, arena, descriptors);
      if (ordinary != ~std::uint64_t{0})
        candidates[cursor++] = {ordinary, head, level};
    }
    __syncthreads();
  }
}

template <class Source>
__global__ void emit_incoming_candidate_stream(
    Source source, const std::uint32_t *refs, std::uint32_t count,
    TerminalCandidate *candidates) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= count) return;
  const std::uint32_t ref = refs[position];
  candidates[position] = {
      ref, source_key(source, ref).head4(), kTerminalIncomingSource};
}

template <class Source>
__global__ void mark_live_terminal_winners(
    Source source, const std::uint32_t *winners, std::uint32_t count,
    std::uint8_t *flags) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position < count) flags[position] = 1u;
}

template <class Source>
__global__ void make_winner_heads(
    Source source, const std::uint32_t *winners,
    std::uint32_t count, std::uint32_t *heads) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position < count)
    heads[position] = source_key(source, winners[position]).head4();
}

template <class Source>
__device__ __forceinline__ bool winner_requires_sparse(
    Source source, std::uint32_t ref) {
  const auto key = source_key(source, ref);
  const auto value = source_value(source, ref);
  const bool tombstone = source_tombstone(source, ref);
  return source_owned_capsule(source, ref) || key.length() != 4u ||
      (!tombstone && value.length() != 4u) ||
      (!tombstone && source_summary(source, ref) != value.inline_word());
}

template <class Source>
__global__ void classify_refined_groups(
    Source source, const std::uint32_t *winners,
    const std::uint32_t *counts, const std::uint32_t *starts,
    std::uint32_t group_count, std::uint32_t *exact_rows,
    std::uint8_t *exact_flags, std::uint8_t *special_flags) {
  const std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= group_count) return;
  const std::uint32_t count = counts[group];
  const std::uint32_t ref = winners[starts[group]];
  const bool recovered = count == 1u &&
      source_key(source, ref).length() == 4u &&
      !source_tombstone(source, ref);
  exact_rows[group] = recovered ? 0u : count;
  exact_flags[group] = static_cast<std::uint8_t>(!recovered);
  special_flags[group] = static_cast<std::uint8_t>(
      !recovered || winner_requires_sparse(source, ref));
}

__global__ void build_refined_exact_descriptors(
    const std::uint32_t *group_ids, std::uint32_t descriptor_count,
    const std::uint32_t *heads, const std::uint32_t *counts,
    const std::uint32_t *exact_offsets,
    ExactHeadDescriptor *descriptors) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= descriptor_count) return;
  const std::uint32_t group = group_ids[position];
  descriptors[position] = {
      heads[group], counts[group],
      kExactPhysicalBit | std::uint64_t{exact_offsets[group]}};
}

__global__ void resolve_refined_destinations(
    const std::uint32_t *heads, const std::uint32_t *counts,
    const std::uint32_t *starts, const std::uint32_t *exact_offsets,
    const std::uint8_t *exact_flags, std::uint32_t group_count,
    std::uint32_t destination_level,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    std::uint64_t *destinations, unsigned long long *errors) {
  const std::uint32_t group = blockIdx.x;
  if (group >= group_count) return;
  const std::uint32_t count = counts[group];
  const std::uint32_t begin = starts[group];
  if (exact_flags[group]) {
    for (std::uint32_t local = threadIdx.x; local < count;
         local += blockDim.x)
      destinations[begin + local] = kExactPhysicalBit |
          std::uint64_t{exact_offsets[group] + local};
    return;
  }
  if (threadIdx.x) return;
  const std::uint64_t physical = find_projection_row(
      heads[group], destination_level, arena, descriptors);
  if (physical == ~std::uint64_t{0}) {
    atomicAdd(errors, 1ull);
    return;
  }
  destinations[begin] = physical;
}

__global__ void patch_exact_projection_rows(
    const ExactHeadDescriptor *exact_heads, std::uint32_t exact_head_count,
    std::uint32_t destination_level,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    unsigned long long *errors) {
  const std::uint32_t descriptor = blockIdx.x * blockDim.x + threadIdx.x;
  if (descriptor >= exact_head_count) return;
  const std::uint32_t head = exact_heads[descriptor].head;
  const std::uint64_t physical = find_projection_row(
      head, destination_level, arena, descriptors);
  if (physical == ~std::uint64_t{0}) {
    atomicAdd(errors, 1ull);
    return;
  }
  arena.store(physical, exact_group_row(head, descriptor));
}

__global__ void patch_retired_roster_rows(
    const std::uint32_t *roster, std::uint32_t roster_count,
    const std::uint32_t *live_heads, std::uint32_t live_head_count,
    std::uint32_t destination_level,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    unsigned long long *errors) {
  const std::uint32_t task = blockIdx.x * blockDim.x + threadIdx.x;
  if (task >= roster_count) return;
  const std::uint32_t head = roster[task];
  std::uint32_t low = 0u;
  std::uint32_t high = live_head_count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (live_heads[middle] < head) low = middle + 1u;
    else high = middle;
  }
  if (low < live_head_count && live_heads[low] == head) return;
  const std::uint64_t physical = find_projection_row(
      head, destination_level, arena, descriptors);
  if (physical == ~std::uint64_t{0}) {
    atomicAdd(errors, 1ull);
    return;
  }
  arena.store(physical,
              {0u, static_cast<std::uint16_t>(head),
               gpulsmopt2_detail::kTombstone});
}

template <class Source>
__global__ void classify_refined_capsules(
    Source source, const std::uint32_t *winners,
    std::uint32_t count, std::uint8_t *flags) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= count) return;
  const std::uint32_t ref = winners[position];
  const auto key = source_key(source, ref);
  const auto value = source_value(source, ref);
  const bool tombstone = source_tombstone(source, ref);
  flags[position] = static_cast<std::uint8_t>(
      source_owned_capsule(source, ref) || key.length() != 4u ||
      (!tombstone && value.length() != 4u) ||
      (!tombstone && source_summary(source, ref) != value.inline_word()));
}

__global__ void gather_refined_capsule_sort_fields(
    const std::uint32_t *capsule_positions,
    const std::uint32_t *capsule_count,
    const std::uint64_t *destinations, std::uint64_t *physical_rows,
    std::uint32_t *winner_positions) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= *capsule_count) return;
  const std::uint32_t position = capsule_positions[ordinal];
  physical_rows[ordinal] = destinations[position];
  winner_positions[ordinal] = position;
}

__global__ void gather_sorted_refined_capsules(
    const std::uint32_t *winners, const std::uint32_t *winner_positions,
    const std::uint64_t *physical_rows, const std::uint32_t *capsule_count,
    std::uint32_t *refs, std::uint64_t *physical_pages) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= *capsule_count) return;
  refs[ordinal] = winners[winner_positions[ordinal]];
  physical_pages[ordinal] = physical_rows[ordinal] / kCapsulePageRows;
}

template <class Source>
__global__ void emit_refined_anchors(
    Source source, const std::uint32_t *winners,
    const std::uint64_t *destinations,
    const std::uint8_t *capsule_flags,
    const std::uint16_t *capsule_ranks, std::uint32_t count,
    gpulsmopt2_detail::ResidentRows arena,
    gpulsmopt2_detail::Row *exact_rows) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= count) return;
  const std::uint32_t ref = winners[position];
  const auto key = source_key(source, ref);
  const auto value = source_value(source, ref);
  const bool tombstone = source_tombstone(source, ref);
  const std::uint32_t inline_value = value.inline_word();
  const std::uint32_t summary = source_summary(source, ref);
  const bool summary_only = !tombstone && summary != inline_value;
  const std::uint16_t flags = capsule_flags[position]
      ? capsule_anchor_flags(
            capsule_ranks[position], tombstone, summary_only)
      : inline_anchor_flags(
            key.length(), tombstone ? 0u : value.length(), tombstone,
            summary_only);
  const gpulsmopt2_detail::Row row{
      summary, static_cast<std::uint16_t>(key.head4()), flags};
  const std::uint64_t destination = destinations[position];
  if (is_exact_physical(destination))
    exact_rows[exact_physical_offset(destination)] = row;
  else
    arena.store(destination, row);
}

struct SparseRefinementResult {
  std::uint32_t exact_heads{};
  std::uint32_t exact_rows{};
  std::uint32_t special_heads{};
  std::uint32_t capsules{};
  std::uint32_t pages{};
  std::uint64_t allocated_capsule_bytes{};
};

// Mixed publication is one receipt-backed commit.  The original ordinary
// manifest layout is preserved byte-for-byte; this companion kernel copies
// its publication rule and updates the sparse-only table before performing
// the same single active-index flip.
__global__ void publish_refined_manifest_kernel(
    gpulsmopt2_detail::ResidentPublicationPlan *plan,
    gpulsmopt2_detail::DeviceManifest *manifests,
    DeviceSparseManifest *sparse_manifests,
    std::uint32_t *active_manifest,
    std::uint64_t *query_occupied_level_mask,
    const RootBuildState *root, std::uint32_t sparse_flags) {
  if (blockIdx.x || plan->status) return;
  const auto *current = manifests + plan->active_manifest;
  auto *next = manifests + plan->inactive_manifest;
  const auto *current_sparse = sparse_manifests + plan->active_manifest;
  auto *next_sparse = sparse_manifests + plan->inactive_manifest;
  const std::uint32_t destination = plan->destination_level;
  const std::uint64_t consumed = destination == 64u
      ? ~std::uint64_t{0}
      : ((std::uint64_t{1u} << destination) - 1u);
  for (std::uint32_t level = threadIdx.x;
       level < gpulsmopt2_detail::kMaximumLevels; level += blockDim.x) {
    if (level == destination) continue;
    next->levels[level] = level < destination
        ? gpulsmopt2_detail::DeviceLevelState{}
        : current->levels[level];
    next_sparse->levels[level] = level < destination
        ? DeviceSparseLevelState{}
        : current_sparse->levels[level];
  }
  __syncthreads();
  if (!threadIdx.x) {
    next->occupied_level_mask = current->occupied_level_mask & ~consumed;
    gpulsmopt2_detail::DeviceLevelState ordinary{};
    ordinary.storage_generation = plan->output_generation;
    next->levels[destination] = ordinary;
    if (plan->survivor_count)
      next->occupied_level_mask |= std::uint64_t{1u} << destination;
    else
      next->occupied_level_mask &= ~(std::uint64_t{1u} << destination);
    next->active_levels = next->occupied_level_mask
        ? 64u - static_cast<std::uint32_t>(
                    __clzll(next->occupied_level_mask))
        : 0u;
    next->foundation_level = next->active_levels
        ? next->active_levels - 1u
        : gpulsmopt2_detail::kMaximumLevels;
    next->generation = current->generation + 1u;

    next_sparse->exact_level_mask =
        current_sparse->exact_level_mask & ~consumed;
    next_sparse->capsule_level_mask =
        current_sparse->capsule_level_mask & ~consumed;
    next_sparse->levels[destination] = {};
    next_sparse->exact_level_mask &=
        ~(std::uint64_t{1u} << destination);
    next_sparse->capsule_level_mask &=
        ~(std::uint64_t{1u} << destination);
    if (root && sparse_flags) {
      next_sparse->levels[destination] = {
          reinterpret_cast<std::uint64_t>(root), root->generation,
          sparse_flags};
      if (sparse_flags & kSparseHasExactHeads)
        next_sparse->exact_level_mask |=
            std::uint64_t{1u} << destination;
      if (sparse_flags & kSparseHasCapsules)
        next_sparse->capsule_level_mask |=
            std::uint64_t{1u} << destination;
    }
    next_sparse->generation = next->generation;
  }
  __syncthreads();
  if (!threadIdx.x) {
    __threadfence();
    atomicExch(active_manifest, plan->inactive_manifest);
    atomicExch(reinterpret_cast<unsigned long long *>(
                   query_occupied_level_mask),
               static_cast<unsigned long long>(
                   next->occupied_level_mask));
  }
}

class SparseRefinementWorkspace {
 public:
  struct PreparedRoster {
    const std::uint32_t *heads{};
    std::uint32_t count{};
    const std::uint32_t *source_levels{};
    std::uint32_t source_count{};
    const RootBuildState *device_roots{};
  };

  PreparedRoster prepare_roster(
      const std::array<std::shared_ptr<StagedPendingSlotOverlay>,
                       gpulsmopt2_detail::kBatchesPerEpoch> &pending,
      std::uint32_t pending_batches,
      const std::array<std::shared_ptr<StagedRootOverlay>,
                       gpulsmopt2_detail::kMaximumLevels> &roots,
      std::uint64_t source_mask, cudaStream_t stream,
      const std::uint32_t *direct_heads = nullptr,
      std::uint32_t direct_head_count = 0u) {
    host_source_levels_.clear();
    lifecycle_owners_.clear();
    std::array<RootBuildState, kLevels> host_roots{};
    std::uint64_t levels = source_mask;
    std::uint32_t occurrences = direct_head_count;
    for (std::uint32_t slot = 0u; slot < pending_batches; ++slot) {
      if (!pending[slot]) continue;
      occurrences += pending[slot]->state.special_head_count;
      for (auto &segment : pending[slot]->segments)
        if (segment.segment) lifecycle_owners_.push_back(&segment);
    }
    while (levels) {
      const std::uint32_t level = static_cast<std::uint32_t>(
          __builtin_ctzll(levels));
      levels &= levels - 1u;
      host_source_levels_.push_back(level);
      if (!roots[level]) continue;
      host_roots[level] = roots[level]->state;
      occurrences += roots[level]->state.special_head_count;
      for (auto &segment : roots[level]->segments)
        if (segment.segment) lifecycle_owners_.push_back(&segment);
    }
    device_roots_.reset(kLevels);
    check(cudaMemcpyAsync(
              device_roots_.data(), host_roots.data(),
              sizeof(host_roots), cudaMemcpyHostToDevice, stream),
          "stage sparse source roots");
    device_source_levels_.reset(host_source_levels_.size());
    if (!host_source_levels_.empty())
      check(cudaMemcpyAsync(
                device_source_levels_.data(), host_source_levels_.data(),
                host_source_levels_.size() * sizeof(std::uint32_t),
                cudaMemcpyHostToDevice, stream),
            "stage sparse source levels");
    if (!occurrences) {
      roster_count_host_ = 0u;
      return {nullptr, 0u, device_source_levels_.data(),
              static_cast<std::uint32_t>(host_source_levels_.size()),
              device_roots_.data()};
    }
    ensure_roster_capacity(occurrences);
    std::uint32_t cursor = 0u;
    if (direct_head_count) {
      check(cudaMemcpyAsync(
                roster_input_.data(), direct_heads,
                std::size_t{direct_head_count} * sizeof(std::uint32_t),
                cudaMemcpyDeviceToDevice, stream),
            "append direct sparse roster");
      cursor = direct_head_count;
    }
    for (std::uint32_t slot = 0u; slot < pending_batches; ++slot) {
      if (!pending[slot] || !pending[slot]->state.special_head_count)
        continue;
      const std::uint32_t count = pending[slot]->state.special_head_count;
      check(cudaMemcpyAsync(
                roster_input_.data() + cursor,
                pending[slot]->special_heads.data(),
                std::size_t{count} * sizeof(std::uint32_t),
                cudaMemcpyDeviceToDevice, stream),
            "append pending sparse roster");
      cursor += count;
    }
    for (std::uint32_t level : host_source_levels_) {
      if (!roots[level] || !roots[level]->state.special_head_count) continue;
      const std::uint32_t count = roots[level]->state.special_head_count;
      check(cudaMemcpyAsync(
                roster_input_.data() + cursor,
                roots[level]->special_heads.data(),
                std::size_t{count} * sizeof(std::uint32_t),
                cudaMemcpyDeviceToDevice, stream),
            "append resident sparse roster");
      cursor += count;
    }
    std::size_t bytes = roster_sort_bytes_;
    check(cub::DeviceRadixSort::SortKeys(
              roster_temporary_.data(), bytes, roster_input_.data(),
              roster_sorted_.data(), occurrences, 0, 32, stream),
          "sort sparse roster");
    bytes = roster_rle_bytes_;
    check(cub::DeviceRunLengthEncode::Encode(
              roster_temporary_.data(), bytes, roster_sorted_.data(),
              roster_.data(), roster_run_counts_.data(),
              roster_count_.data(), occurrences, stream),
          "deduplicate sparse roster");
    check(cudaMemcpyAsync(&roster_count_host_, roster_count_.data(),
                          sizeof(roster_count_host_),
                          cudaMemcpyDeviceToHost, stream),
          "copy sparse roster count");
    check(cudaStreamSynchronize(stream), "wait sparse roster");
    return {roster_.data(), roster_count_host_,
            device_source_levels_.data(),
            static_cast<std::uint32_t>(host_source_levels_.size()),
            device_roots_.data()};
  }

  // Direct bulk/sealed roots already have a sorted, deduplicated sparse-head
  // roster from the transplanted root builder.  Stage that sparse list only;
  // ordinary projection metadata remains exclusively in GPULSMOpt.
  PreparedRoster prepare_direct_roster(
      const std::uint32_t *heads, std::uint32_t count,
      cudaStream_t stream) {
    host_source_levels_.clear();
    lifecycle_owners_.clear();
    device_roots_.reset(kLevels);
    check(cudaMemsetAsync(device_roots_.data(), 0, device_roots_.bytes(),
                          stream),
          "clear direct sparse source roots");
    device_source_levels_.reset(0u);
    roster_count_host_ = count;
    if (!count)
      return {nullptr, 0u, nullptr, 0u, device_roots_.data()};
    ensure_roster_capacity(count);
    check(cudaMemcpyAsync(roster_.data(), heads,
                          std::size_t{count} * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToDevice, stream),
          "stage direct sparse roster");
    check(cudaMemcpyAsync(roster_count_.data(), &count, sizeof(count),
                          cudaMemcpyHostToDevice, stream),
          "stage direct sparse roster count");
    return {roster_.data(), count, nullptr, 0u, device_roots_.data()};
  }

  const std::vector<CapsuleSegmentOwnership *> &lifecycle_owners() const {
    return lifecycle_owners_;
  }

  std::size_t bytes() const;

  SparseRefinementResult refine(
      CompletionSourceView pending, std::uint32_t pending_batches,
      gpulsmopt2_detail::ResidentRows arena,
      const gpulsmopt2_detail::Descriptor *descriptors,
      std::uint32_t destination_level, std::uint64_t output_begin,
      std::uint32_t projection_count,
      bool keep_tombstones, std::uint32_t output_generation,
      std::uint32_t segment_ordinal, StagedRootOverlay &overlay,
      cudaStream_t stream);

 private:
  void ensure_roster_capacity(std::uint32_t count);
  void ensure_pending_capacity(std::uint32_t count);
  void ensure_candidate_capacity(std::uint32_t count);
  void ensure_group_capacity(std::uint32_t count);
  void ensure_capsule_capacity(std::uint32_t count);
  template <class Source>
  const CapsuleLifecycleSource *prepare_capsule_lifecycle(
      Source source, const std::uint32_t *refs, std::uint32_t count,
      StagedRootOverlay &overlay, cudaStream_t stream);

  std::uint32_t roster_capacity_{};
  std::uint32_t roster_count_host_{};
  Buffer<std::uint32_t> roster_input_, roster_sorted_, roster_;
  Buffer<std::uint32_t> roster_run_counts_, roster_count_;
  Buffer<std::uint8_t> roster_temporary_;
  std::size_t roster_sort_bytes_{}, roster_rle_bytes_{};
  Buffer<RootBuildState> device_roots_;
  Buffer<std::uint32_t> device_source_levels_;
  std::vector<std::uint32_t> host_source_levels_;
  std::vector<CapsuleSegmentOwnership *> lifecycle_owners_;

  std::uint32_t pending_capacity_{};
  Buffer<std::uint32_t> pending_counts_, pending_offsets_;
  Buffer<std::uint32_t> pending_ages_a_, pending_ages_b_;
  Buffer<std::uint32_t> pending_refs_a_, pending_refs_b_;
  Buffer<std::uint32_t> pending_total_;
  Buffer<std::uint8_t> pending_temporary_;
  std::size_t pending_scan_bytes_{}, pending_sort_bytes_{};
  std::unique_ptr<ExactOrderWorkspace> pending_order_;

  std::uint32_t candidate_capacity_{};
  Buffer<TerminalCandidate> candidates_;
  Buffer<unsigned long long> source_counts_;
  Buffer<std::uint32_t> source_begins_;
  ExactFanInWorkspace fan_in_;

  std::uint32_t group_capacity_{};
  Buffer<std::uint8_t> live_flags_;
  Buffer<std::uint32_t> live_winners_, live_count_;
  Buffer<std::uint32_t> winner_heads_, group_heads_, group_counts_;
  Buffer<std::uint32_t> group_starts_, group_count_;
  Buffer<std::uint32_t> exact_rows_, exact_offsets_;
  Buffer<std::uint8_t> exact_flags_, special_flags_;
  Buffer<std::uint32_t> exact_group_ids_, exact_group_count_;
  Buffer<std::uint32_t> selected_special_heads_, special_head_count_;
  Buffer<std::uint64_t> destinations_;
  Buffer<unsigned long long> errors_;
  Buffer<std::uint8_t> group_temporary_;
  std::size_t group_select_bytes_{}, group_rle_bytes_{};
  std::size_t group_scan_bytes_{};

  std::uint32_t capsule_capacity_{};
  Buffer<std::uint8_t> capsule_flags_;
  Buffer<std::uint16_t> capsule_ranks_;
  Buffer<std::uint32_t> capsule_positions_a_, capsule_positions_b_;
  Buffer<std::uint32_t> capsule_count_, capsule_refs_;
  Buffer<std::uint64_t> capsule_rows_a_, capsule_rows_b_;
  Buffer<std::uint64_t> capsule_pages_, unique_pages_;
  Buffer<std::uint32_t> page_counts_, page_starts_, page_count_;
  Buffer<std::uint64_t> capsule_sizes_, capsule_offsets_;
  Buffer<std::uint8_t> capsule_temporary_;
  std::size_t capsule_select_bytes_{}, capsule_sort_bytes_{};
  std::size_t capsule_rle_bytes_{}, capsule_scan32_bytes_{};
  std::size_t capsule_scan64_bytes_{};

  std::uint32_t lifecycle_capacity_{};
  std::uint32_t lifecycle_count_{};
  Buffer<CapsuleLifecycleSource> lifecycle_sources_;
  std::vector<CapsuleLifecycleSource> host_lifecycle_;
  Buffer<unsigned long long> lifecycle_errors_;
};

inline void SparseRefinementWorkspace::ensure_roster_capacity(
    std::uint32_t count) {
  if (count <= roster_capacity_) return;
  roster_capacity_ = count;
  roster_input_.reset(count);
  roster_sorted_.reset(count);
  roster_.reset(count);
  roster_run_counts_.reset(count);
  roster_count_.reset(1u);
  pending_counts_.reset(count);
  pending_offsets_.reset(count);
  pending_total_.reset(1u);
  std::size_t sizes[3]{};
  check(cub::DeviceRadixSort::SortKeys(
            nullptr, sizes[0], roster_input_.data(), roster_sorted_.data(),
            count),
        "size sparse roster sort");
  check(cub::DeviceRunLengthEncode::Encode(
            nullptr, sizes[1], roster_sorted_.data(), roster_.data(),
            roster_run_counts_.data(), roster_count_.data(), count),
        "size sparse roster encoding");
  check(cub::DeviceScan::ExclusiveSum(
            nullptr, sizes[2], pending_counts_.data(),
            pending_offsets_.data(), count),
        "size pending roster scan");
  roster_sort_bytes_ = sizes[0];
  roster_rle_bytes_ = sizes[1];
  pending_scan_bytes_ = sizes[2];
  roster_temporary_.reset(std::max(sizes[0], sizes[1]));
  pending_temporary_.reset(
      std::max(pending_scan_bytes_, pending_sort_bytes_));
}

inline void SparseRefinementWorkspace::ensure_pending_capacity(
    std::uint32_t count) {
  if (count <= pending_capacity_) return;
  pending_capacity_ = count;
  pending_ages_a_.reset(count);
  pending_ages_b_.reset(count);
  pending_refs_a_.reset(count);
  pending_refs_b_.reset(count);
  std::size_t bytes = 0u;
  check(cub::DeviceRadixSort::SortPairs(
            nullptr, bytes, pending_ages_a_.data(), pending_ages_b_.data(),
            pending_refs_a_.data(), pending_refs_b_.data(), count),
        "size pending age sort");
  pending_sort_bytes_ = bytes;
  pending_temporary_.reset(std::max(pending_scan_bytes_, bytes));
  pending_order_ = std::make_unique<ExactOrderWorkspace>(count);
}

inline void SparseRefinementWorkspace::ensure_candidate_capacity(
    std::uint32_t count) {
  if (count <= candidate_capacity_) return;
  candidate_capacity_ = count;
  candidates_.reset(count);
}

inline void SparseRefinementWorkspace::ensure_group_capacity(
    std::uint32_t count) {
  if (count <= group_capacity_) return;
  group_capacity_ = count;
  live_flags_.reset(count);
  live_winners_.reset(count);
  live_count_.reset(1u);
  winner_heads_.reset(count);
  group_heads_.reset(count);
  group_counts_.reset(count);
  group_starts_.reset(count);
  group_count_.reset(1u);
  exact_rows_.reset(count);
  exact_offsets_.reset(count);
  exact_flags_.reset(count);
  special_flags_.reset(count);
  exact_group_ids_.reset(count);
  exact_group_count_.reset(1u);
  selected_special_heads_.reset(count);
  special_head_count_.reset(1u);
  destinations_.reset(count);
  errors_.reset(1u);
  std::size_t sizes[4]{};
  check(cub::DeviceSelect::Flagged(
            nullptr, sizes[0], winner_heads_.data(), live_flags_.data(),
            live_winners_.data(), live_count_.data(), count),
        "size refined winner selection");
  check(cub::DeviceRunLengthEncode::Encode(
            nullptr, sizes[1], winner_heads_.data(), group_heads_.data(),
            group_counts_.data(), group_count_.data(), count),
        "size refined head encoding");
  check(cub::DeviceScan::ExclusiveSum(
            nullptr, sizes[2], group_counts_.data(), group_starts_.data(),
            count),
        "size refined group scan");
  cub::CountingInputIterator<std::uint32_t> ids(0u);
  check(cub::DeviceSelect::Flagged(
            nullptr, sizes[3], ids, exact_flags_.data(),
            exact_group_ids_.data(), exact_group_count_.data(), count),
        "size refined group selection");
  group_select_bytes_ = std::max(sizes[0], sizes[3]);
  group_rle_bytes_ = sizes[1];
  group_scan_bytes_ = sizes[2];
  group_temporary_.reset(*std::max_element(
      std::begin(sizes), std::end(sizes)));
}

inline void SparseRefinementWorkspace::ensure_capsule_capacity(
    std::uint32_t count) {
  if (count <= capsule_capacity_) return;
  capsule_capacity_ = count;
  capsule_flags_.reset(count);
  capsule_ranks_.reset(count);
  capsule_positions_a_.reset(count);
  capsule_positions_b_.reset(count);
  capsule_count_.reset(1u);
  capsule_refs_.reset(count);
  capsule_rows_a_.reset(count);
  capsule_rows_b_.reset(count);
  capsule_pages_.reset(count);
  unique_pages_.reset(count);
  page_counts_.reset(count);
  page_starts_.reset(count);
  page_count_.reset(1u);
  capsule_sizes_.reset(std::size_t{count} + 1u);
  capsule_offsets_.reset(std::size_t{count} + 1u);
  std::size_t sizes[5]{};
  cub::CountingInputIterator<std::uint32_t> ids(0u);
  check(cub::DeviceSelect::Flagged(
            nullptr, sizes[0], ids, capsule_flags_.data(),
            capsule_positions_a_.data(), capsule_count_.data(), count),
        "size refined capsule selection");
  check(cub::DeviceRadixSort::SortPairs(
            nullptr, sizes[1], capsule_rows_a_.data(),
            capsule_rows_b_.data(), capsule_positions_b_.data(),
            capsule_positions_a_.data(), count),
        "size refined capsule sort");
  check(cub::DeviceRunLengthEncode::Encode(
            nullptr, sizes[2], capsule_pages_.data(), unique_pages_.data(),
            page_counts_.data(), page_count_.data(), count),
        "size refined capsule pages");
  check(cub::DeviceScan::ExclusiveSum(
            nullptr, sizes[3], page_counts_.data(), page_starts_.data(),
            count),
        "size refined page scan");
  check(cub::DeviceScan::ExclusiveSum(
            nullptr, sizes[4], capsule_sizes_.data(),
            capsule_offsets_.data(), std::size_t{count} + 1u),
        "size refined capsule byte scan");
  capsule_select_bytes_ = sizes[0];
  capsule_sort_bytes_ = sizes[1];
  capsule_rle_bytes_ = sizes[2];
  capsule_scan32_bytes_ = sizes[3];
  capsule_scan64_bytes_ = sizes[4];
  capsule_temporary_.reset(*std::max_element(
      std::begin(sizes), std::end(sizes)));
}

template <class Source>
const CapsuleLifecycleSource *
SparseRefinementWorkspace::prepare_capsule_lifecycle(
    Source source, const std::uint32_t *refs, std::uint32_t count,
    StagedRootOverlay &overlay, cudaStream_t stream) {
  host_lifecycle_.clear();
  host_lifecycle_.reserve(lifecycle_owners_.size());
  for (CapsuleSegmentOwnership *owned : lifecycle_owners_) {
    if (!owned || !owned->segment) continue;
    host_lifecycle_.push_back({
        owned->ordinal, kCapsuleCopy, owned->live_bytes,
        owned->garbage_bytes, 0u});
  }
  std::sort(host_lifecycle_.begin(), host_lifecycle_.end(),
            [](const CapsuleLifecycleSource &left,
               const CapsuleLifecycleSource &right) {
              return left.ordinal < right.ordinal;
            });
  for (std::size_t index = 1u; index < host_lifecycle_.size(); ++index)
    if (host_lifecycle_[index - 1u].ordinal ==
        host_lifecycle_[index].ordinal)
      throw std::logic_error("capsule segment has multiple owners");
  lifecycle_count_ = static_cast<std::uint32_t>(host_lifecycle_.size());
  if (lifecycle_count_ > lifecycle_capacity_) {
    lifecycle_capacity_ = lifecycle_count_;
    lifecycle_sources_.reset(lifecycle_capacity_);
  }
  if (!lifecycle_errors_.size()) lifecycle_errors_.reset(1u);
  check(cudaMemsetAsync(lifecycle_errors_.data(), 0,
                        lifecycle_errors_.bytes(), stream),
        "clear capsule lifecycle errors");
  if (!lifecycle_count_) return nullptr;
  check(cudaMemcpyAsync(
            lifecycle_sources_.data(), host_lifecycle_.data(),
            std::size_t{lifecycle_count_} * sizeof(CapsuleLifecycleSource),
            cudaMemcpyHostToDevice, stream),
        "stage capsule lifecycle sources");
  count_capsule_survivors<<<blocks(count), kThreads, 0, stream>>>(
      source, refs, capsule_count_.data(), lifecycle_sources_.data(),
      lifecycle_count_, lifecycle_errors_.data());
  check(cudaMemcpyAsync(
            host_lifecycle_.data(), lifecycle_sources_.data(),
            std::size_t{lifecycle_count_} * sizeof(CapsuleLifecycleSource),
            cudaMemcpyDeviceToHost, stream),
        "copy capsule liveness");
  unsigned long long lifecycle_error = 0u;
  check(cudaMemcpyAsync(&lifecycle_error, lifecycle_errors_.data(),
                        sizeof(lifecycle_error), cudaMemcpyDeviceToHost,
                        stream),
        "copy capsule liveness error");
  check(cudaStreamSynchronize(stream), "wait capsule liveness");
  if (lifecycle_error)
    throw std::runtime_error(
        "capsule lifecycle error kind " +
        std::to_string(static_cast<std::uint32_t>(lifecycle_error >> 32u)) +
        " segment " + std::to_string(static_cast<std::uint32_t>(
            lifecycle_error)));

  const auto find_owner = [&](std::uint32_t ordinal) {
    CapsuleSegmentOwnership *found = nullptr;
    for (CapsuleSegmentOwnership *owned : lifecycle_owners_) {
      if (!owned || !owned->segment || owned->ordinal != ordinal) continue;
      if (found) throw std::logic_error(
          "capsule transfer source is duplicated");
      found = owned;
    }
    return found;
  };
  overlay.transfers.reserve(
      overlay.transfers.size() + lifecycle_count_);
  for (CapsuleLifecycleSource &state : host_lifecycle_) {
    if (state.survivor_bytes > state.live_bytes)
      throw std::runtime_error("capsule survivor accounting overflow");
    if (state.garbage_bytes >
        std::numeric_limits<std::uint64_t>::max() - state.live_bytes)
      throw std::overflow_error("capsule usage overflow");
    const std::uint64_t occupied = state.live_bytes + state.garbage_bytes;
    const std::uint64_t garbage = occupied - state.survivor_bytes;
    if (!state.survivor_bytes) {
      state.action = kCapsuleCompact;
    } else if (garbage < state.survivor_bytes) {
      state.action = kCapsuleTransfer;
      CapsuleSegmentOwnership *owner = find_owner(state.ordinal);
      if (!owner) throw std::logic_error(
          "capsule transfer source is absent");
      const std::uint64_t mapped = owner->segment->mapped_bytes();
      if (state.survivor_bytes > mapped ||
          garbage > mapped - state.survivor_bytes)
        throw std::overflow_error("capsule transfer usage is invalid");
      overlay.transfers.push_back({
          state.ordinal, 0u, state.survivor_bytes, garbage, owner});
      overlay.live_bytes += state.survivor_bytes;
      overlay.garbage_bytes += garbage;
    } else {
      state.action = kCapsuleCompact;
    }
  }
  check(cudaMemcpyAsync(
            lifecycle_sources_.data(), host_lifecycle_.data(),
            std::size_t{lifecycle_count_} * sizeof(CapsuleLifecycleSource),
            cudaMemcpyHostToDevice, stream),
        "stage capsule lifecycle actions");
  return lifecycle_sources_.data();
}

inline SparseRefinementResult SparseRefinementWorkspace::refine(
    CompletionSourceView pending, std::uint32_t pending_batches,
    gpulsmopt2_detail::ResidentRows arena,
    const gpulsmopt2_detail::Descriptor *descriptors,
    std::uint32_t destination_level, std::uint64_t output_begin,
    std::uint32_t projection_count, bool keep_tombstones,
    std::uint32_t output_generation, std::uint32_t segment_ordinal,
    StagedRootOverlay &overlay, cudaStream_t stream) {
  if (!roster_count_host_)
    throw std::logic_error("sparse refinement has no roster");
  overlay.state = {};
  overlay.live_bytes = 0u;
  overlay.garbage_bytes = 0u;
  overlay.transfers.clear();

  count_pending_roster_candidates<<<
      blocks(roster_count_host_), kThreads, 0, stream>>>(
      roster_.data(), roster_count_host_, pending.incoming_sorted_heads,
      pending.incoming_records, pending.pending_keys,
      pending.pending_offsets, pending.batch_capacity, pending_batches,
      pending_counts_.data());
  std::size_t bytes = pending_scan_bytes_;
  check(cub::DeviceScan::ExclusiveSum(
            pending_temporary_.data(), bytes, pending_counts_.data(),
            pending_offsets_.data(), roster_count_host_, stream),
        "scan pending sparse candidates");
  finish_scan_total<<<1, 1, 0, stream>>>(
      pending_counts_.data(), pending_offsets_.data(), roster_count_.data(),
      pending_total_.data());
  std::uint32_t pending_rows = 0u;
  check(cudaMemcpyAsync(&pending_rows, pending_total_.data(),
                        sizeof(pending_rows), cudaMemcpyDeviceToHost,
                        stream),
        "copy pending sparse candidate count");
  check(cudaStreamSynchronize(stream),
        "wait pending sparse candidate count");

  std::uint32_t pending_winners = 0u;
  if (pending_rows) {
    ensure_pending_capacity(pending_rows);
    emit_pending_roster_candidates<<<roster_count_host_, kThreads, 0,
                                     stream>>>(
        roster_.data(), roster_count_host_, pending_offsets_.data(),
        pending.incoming_sorted_heads, pending.incoming_sorted_refs,
        pending.incoming_records, pending.pending_keys,
        pending.pending_payloads,
        pending.pending_offsets, pending.batch_capacity, pending_batches,
        pending_ages_a_.data(), pending_refs_a_.data());
    bytes = pending_sort_bytes_;
    check(cub::DeviceRadixSort::SortPairs(
              pending_temporary_.data(), bytes, pending_ages_a_.data(),
              pending_ages_b_.data(), pending_refs_a_.data(),
              pending_refs_b_.data(), pending_rows, 0, 32, stream),
          "sort pending sparse candidates by age");
    const ExactOrderResult ordered = pending_order_->order(
        pending, pending_rows, stream, nullptr, pending_refs_b_.data());
    pending_winners = ordered.winner_count;
  }

  const std::uint32_t source_count = static_cast<std::uint32_t>(
      host_source_levels_.size());
  source_counts_.reset(source_count);
  source_begins_.reset(source_count);
  std::vector<unsigned long long> host_source_counts(source_count);
  if (source_count) {
    check(cudaMemsetAsync(source_counts_.data(), 0, source_counts_.bytes(),
                          stream),
          "clear resident sparse source counts");
    count_resident_candidate_streams<<<source_count, kThreads, 0, stream>>>(
        roster_.data(), roster_count_host_, arena, descriptors,
        device_roots_.data(), device_source_levels_.data(), source_count,
        source_counts_.data());
    check(cudaMemcpyAsync(
              host_source_counts.data(), source_counts_.data(),
              source_count * sizeof(unsigned long long),
              cudaMemcpyDeviceToHost, stream),
          "copy resident sparse source counts");
    check(cudaStreamSynchronize(stream),
          "wait resident sparse source counts");
  }

  std::uint64_t candidate_total = pending_winners;
  std::vector<std::uint32_t> host_source_begins(source_count);
  std::vector<ExactRun> runs;
  if (pending_winners) runs.push_back({0u, pending_winners});
  for (std::uint32_t source = 0u; source < source_count; ++source) {
    if (candidate_total > std::numeric_limits<std::uint32_t>::max() ||
        host_source_counts[source] >
            std::numeric_limits<std::uint32_t>::max() - candidate_total)
      throw std::length_error("sparse candidate set exceeds 32 bits");
    host_source_begins[source] = static_cast<std::uint32_t>(candidate_total);
    const std::uint32_t count = static_cast<std::uint32_t>(
        host_source_counts[source]);
    if (count) runs.push_back({static_cast<std::uint32_t>(candidate_total),
                               count});
    candidate_total += count;
  }
  if (!candidate_total)
    throw std::logic_error("sparse roster has no exact candidates");
  ensure_candidate_capacity(static_cast<std::uint32_t>(candidate_total));
  if (pending_winners)
    emit_incoming_candidate_stream<<<blocks(pending_winners), kThreads, 0,
                                     stream>>>(
        pending, pending_order_->winners(), pending_winners,
        candidates_.data());
  if (source_count) {
    check(cudaMemcpyAsync(
              source_begins_.data(), host_source_begins.data(),
              source_count * sizeof(std::uint32_t),
              cudaMemcpyHostToDevice, stream),
          "stage resident sparse stream begins");
    emit_resident_candidate_streams<<<source_count, kThreads, 0, stream>>>(
        roster_.data(), roster_count_host_, arena, descriptors,
        device_roots_.data(), device_source_levels_.data(), source_count,
        source_begins_.data(), candidates_.data());
  }

  const TerminalSourceView<CompletionSourceView> terminal{
      pending, arena, device_roots_.data(), candidates_.data()};
  const auto merged = fan_in_.merge(
      terminal, runs, static_cast<std::uint32_t>(candidate_total), stream);
  ensure_group_capacity(merged.winner_count);
  mark_live_terminal_winners<<<blocks(merged.winner_count), kThreads, 0,
                               stream>>>(
      terminal, merged.winners, merged.winner_count, live_flags_.data());
  bytes = group_select_bytes_;
  check(cub::DeviceSelect::Flagged(
            group_temporary_.data(), bytes, merged.winners,
            live_flags_.data(), live_winners_.data(), live_count_.data(),
            merged.winner_count, stream),
        "select live exact winners");
  std::uint32_t live_count = 0u;
  check(cudaMemcpyAsync(&live_count, live_count_.data(), sizeof(live_count),
                        cudaMemcpyDeviceToHost, stream),
        "copy live exact winner count");
  check(cudaStreamSynchronize(stream), "wait live exact winners");
  check(cudaMemsetAsync(errors_.data(), 0, errors_.bytes(), stream),
        "clear sparse refinement errors");

  std::uint32_t group_count = 0u;
  std::uint32_t exact_row_count = 0u;
  std::uint32_t exact_head_count = 0u;
  std::uint32_t special_head_count = 0u;
  if (live_count) {
    make_winner_heads<<<blocks(live_count), kThreads, 0, stream>>>(
        terminal, live_winners_.data(), live_count, winner_heads_.data());
    bytes = group_rle_bytes_;
    check(cub::DeviceRunLengthEncode::Encode(
              group_temporary_.data(), bytes, winner_heads_.data(),
              group_heads_.data(), group_counts_.data(), group_count_.data(),
              live_count, stream),
          "encode refined winner heads");
    check(cudaMemcpyAsync(&group_count, group_count_.data(),
                          sizeof(group_count), cudaMemcpyDeviceToHost,
                          stream),
          "copy refined head count");
    check(cudaStreamSynchronize(stream), "wait refined head count");
    bytes = group_scan_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              group_temporary_.data(), bytes, group_counts_.data(),
              group_starts_.data(), group_count, stream),
          "scan refined winner heads");
    classify_refined_groups<<<blocks(group_count), kThreads, 0, stream>>>(
        terminal, live_winners_.data(), group_counts_.data(),
        group_starts_.data(), group_count, exact_rows_.data(),
        exact_flags_.data(), special_flags_.data());
    bytes = group_scan_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              group_temporary_.data(), bytes, exact_rows_.data(),
              exact_offsets_.data(), group_count, stream),
          "scan refined exact rows");
    finish_scan_total<<<1, 1, 0, stream>>>(
        exact_rows_.data(), exact_offsets_.data(), group_count_.data(),
        pending_total_.data());
    cub::CountingInputIterator<std::uint32_t> ids(0u);
    bytes = group_select_bytes_;
    check(cub::DeviceSelect::Flagged(
              group_temporary_.data(), bytes, ids, exact_flags_.data(),
              exact_group_ids_.data(), exact_group_count_.data(),
              group_count, stream),
          "select refined exact groups");
    bytes = group_select_bytes_;
    check(cub::DeviceSelect::Flagged(
              group_temporary_.data(), bytes, group_heads_.data(),
              special_flags_.data(), selected_special_heads_.data(),
              special_head_count_.data(), group_count, stream),
          "select refined special heads");
    check(cudaMemcpyAsync(&exact_row_count, pending_total_.data(),
                          sizeof(exact_row_count), cudaMemcpyDeviceToHost,
                          stream),
          "copy refined exact row count");
    check(cudaMemcpyAsync(&exact_head_count, exact_group_count_.data(),
                          sizeof(exact_head_count), cudaMemcpyDeviceToHost,
                          stream),
          "copy refined exact head count");
    check(cudaMemcpyAsync(&special_head_count, special_head_count_.data(),
                          sizeof(special_head_count), cudaMemcpyDeviceToHost,
                          stream),
          "copy refined special head count");
    check(cudaStreamSynchronize(stream),
          "wait refined sparse metadata counts");
  }

  overlay.exact_heads.reset(exact_head_count);
  overlay.exact_rows.reset(exact_row_count);
  overlay.special_heads.reset(special_head_count);
  if (exact_head_count)
    build_refined_exact_descriptors<<<
        blocks(exact_head_count), kThreads, 0, stream>>>(
        exact_group_ids_.data(), exact_head_count, group_heads_.data(),
        group_counts_.data(), exact_offsets_.data(),
        overlay.exact_heads.data());
  if (special_head_count)
    check(cudaMemcpyAsync(
              overlay.special_heads.data(), selected_special_heads_.data(),
              std::size_t{special_head_count} * sizeof(std::uint32_t),
              cudaMemcpyDeviceToDevice, stream),
          "store refined special heads");
  if (group_count)
    resolve_refined_destinations<<<group_count, kThreads, 0, stream>>>(
        group_heads_.data(), group_counts_.data(), group_starts_.data(),
        exact_offsets_.data(), exact_flags_.data(), group_count,
        destination_level, arena, descriptors, destinations_.data(),
        errors_.data());

  std::uint32_t capsule_count = 0u;
  std::uint32_t page_count = 0u;
  std::uint64_t capsule_bytes = 0u;
  if (live_count) {
    ensure_capsule_capacity(live_count);
    check(cudaMemsetAsync(capsule_ranks_.data(), 0,
                          std::size_t{live_count} * sizeof(std::uint16_t),
                          stream),
          "clear refined capsule ranks");
    classify_refined_capsules<<<blocks(live_count), kThreads, 0, stream>>>(
        terminal, live_winners_.data(), live_count, capsule_flags_.data());
    cub::CountingInputIterator<std::uint32_t> ids(0u);
    bytes = capsule_select_bytes_;
    check(cub::DeviceSelect::Flagged(
              capsule_temporary_.data(), bytes, ids, capsule_flags_.data(),
              capsule_positions_a_.data(), capsule_count_.data(), live_count,
              stream),
          "select refined capsules");
    check(cudaMemcpyAsync(&capsule_count, capsule_count_.data(),
                          sizeof(capsule_count), cudaMemcpyDeviceToHost,
                          stream),
          "copy refined capsule count");
    check(cudaStreamSynchronize(stream), "wait refined capsule count");
  }
  if (capsule_count) {
    gather_refined_capsule_sort_fields<<<
        blocks(capsule_count), kThreads, 0, stream>>>(
        capsule_positions_a_.data(), capsule_count_.data(),
        destinations_.data(), capsule_rows_a_.data(),
        capsule_positions_b_.data());
    bytes = capsule_sort_bytes_;
    check(cub::DeviceRadixSort::SortPairs(
              capsule_temporary_.data(), bytes, capsule_rows_a_.data(),
              capsule_rows_b_.data(), capsule_positions_b_.data(),
              capsule_positions_a_.data(), capsule_count, 0, 64, stream),
          "sort refined capsules by destination");
    gather_sorted_refined_capsules<<<
        blocks(capsule_count), kThreads, 0, stream>>>(
        live_winners_.data(), capsule_positions_a_.data(),
        capsule_rows_b_.data(), capsule_count_.data(), capsule_refs_.data(),
        capsule_pages_.data());
    bytes = capsule_rle_bytes_;
    check(cub::DeviceRunLengthEncode::Encode(
              capsule_temporary_.data(), bytes, capsule_pages_.data(),
              unique_pages_.data(), page_counts_.data(), page_count_.data(),
              capsule_count, stream),
          "encode refined capsule pages");
    check(cudaMemcpyAsync(&page_count, page_count_.data(),
                          sizeof(page_count), cudaMemcpyDeviceToHost,
                          stream),
          "copy refined capsule page count");
    check(cudaStreamSynchronize(stream),
          "wait refined capsule page count");
    bytes = capsule_scan32_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              capsule_temporary_.data(), bytes, page_counts_.data(),
              page_starts_.data(), page_count, stream),
          "scan refined capsule pages");
    const CapsuleLifecycleSource *lifecycle = prepare_capsule_lifecycle(
        terminal, capsule_refs_.data(), capsule_count, overlay, stream);
    make_capsule_object_sizes<<<
        blocks(std::uint64_t{capsule_count} + 1u), kThreads, 0, stream>>>(
        terminal, capsule_refs_.data(), capsule_count_.data(), lifecycle,
        lifecycle_count_, capsule_sizes_.data(), lifecycle_errors_.data());
    bytes = capsule_scan64_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              capsule_temporary_.data(), bytes, capsule_sizes_.data(),
              capsule_offsets_.data(), std::size_t{capsule_count} + 1u,
              stream),
          "scan refined capsule bytes");
    check(cudaMemcpyAsync(&capsule_bytes,
                          capsule_offsets_.data() + capsule_count,
                          sizeof(capsule_bytes), cudaMemcpyDeviceToHost,
                          stream),
          "copy refined capsule bytes");
    check(cudaStreamSynchronize(stream), "wait refined capsule bytes");
    overlay.pages.reset(page_count);
    overlay.indexes.reset(capsule_count);
    std::shared_ptr<CapsuleSegment> new_segment;
    if (capsule_bytes) {
      new_segment = std::make_shared<CapsuleSegment>(capsule_bytes);
      overlay.segments.push_back({
          new_segment, segment_ordinal, 0u, capsule_bytes, 0u});
      overlay.live_bytes += capsule_bytes;
    }
    build_capsule_pages<<<blocks(page_count), kThreads, 0, stream>>>(
        unique_pages_.data(), page_counts_.data(), page_starts_.data(),
        page_count_.data(), overlay.pages.data());
    assign_capsule_ranks<<<blocks(capsule_count), kThreads, 0, stream>>>(
        capsule_pages_.data(), page_starts_.data(), page_count_.data(),
        capsule_positions_a_.data(), capsule_count_.data(),
        capsule_ranks_.data(), errors_.data());
    emit_capsule_objects<<<blocks(capsule_count), kThreads, 0, stream>>>(
        terminal, capsule_refs_.data(), capsule_count_.data(),
        capsule_offsets_.data(), new_segment ? new_segment->data() : nullptr,
        segment_ordinal, lifecycle_sources_.data(), lifecycle_count_,
        overlay.indexes.data(), lifecycle_errors_.data());
  }

  if (live_count)
    emit_refined_anchors<<<blocks(live_count), kThreads, 0, stream>>>(
        terminal, live_winners_.data(), destinations_.data(),
        capsule_flags_.data(), capsule_ranks_.data(), live_count, arena,
        overlay.exact_rows.data());
  if (exact_head_count)
    patch_exact_projection_rows<<<blocks(exact_head_count), kThreads, 0,
                                  stream>>>(
        overlay.exact_heads.data(), exact_head_count, destination_level,
        arena, descriptors, errors_.data());
  patch_retired_roster_rows<<<blocks(roster_count_host_), kThreads, 0,
                              stream>>>(
      roster_.data(), roster_count_host_, group_heads_.data(), group_count,
      destination_level, arena, descriptors, errors_.data());
  check(cudaGetLastError(), "refine sparse projection");

  unsigned long long errors = 0u;
  unsigned long long lifecycle_errors = 0u;
  check(cudaMemcpyAsync(&errors, errors_.data(), sizeof(errors),
                        cudaMemcpyDeviceToHost, stream),
        "copy sparse refinement errors");
  if (lifecycle_errors_.size())
    check(cudaMemcpyAsync(&lifecycle_errors, lifecycle_errors_.data(),
                          sizeof(lifecycle_errors), cudaMemcpyDeviceToHost,
                          stream),
          "copy sparse lifecycle errors");
  check(cudaStreamSynchronize(stream), "wait sparse refinement");
  if (lifecycle_errors)
    throw std::runtime_error(
        "capsule lifecycle error kind " +
        std::to_string(static_cast<std::uint32_t>(lifecycle_errors >> 32u)) +
        " segment " + std::to_string(static_cast<std::uint32_t>(
            lifecycle_errors)));
  if (errors)
    throw std::runtime_error("invalid sparse projection refinement");

  RootBuildState state{};
  state.level = destination_level;
  state.generation = output_generation;
  state.row_count = projection_count;
  state.logical_count = projection_count - exact_head_count + exact_row_count;
  state.exact_head_count = exact_head_count;
  state.special_head_count = special_head_count;
  state.physical_begin = output_begin;
  state.exact_heads = reinterpret_cast<std::uint64_t>(
      overlay.exact_heads.data());
  state.exact_rows = reinterpret_cast<std::uint64_t>(
      overlay.exact_rows.data());
  state.special_heads = reinterpret_cast<std::uint64_t>(
      overlay.special_heads.data());
  state.capsule_pages = reinterpret_cast<std::uint64_t>(
      overlay.pages.data());
  state.capsule_indexes = reinterpret_cast<std::uint64_t>(
      overlay.indexes.data());
  state.exact_row_count = exact_row_count;
  state.capsule_page_count = page_count;
  state.capsule_count = capsule_count;
  state.capsule_live_bytes = overlay.live_bytes;
  state.capsule_garbage_bytes = overlay.garbage_bytes;
  overlay.state = state;
  if (special_head_count) {
    overlay.device_state.reset(1u);
    check(cudaMemcpyAsync(overlay.device_state.data(), &overlay.state,
                          sizeof(overlay.state), cudaMemcpyHostToDevice,
                          stream),
          "stage refined root state");
  }
  return {exact_head_count, exact_row_count, special_head_count,
          capsule_count, page_count, capsule_bytes};
}

inline std::size_t SparseRefinementWorkspace::bytes() const {
  return roster_input_.bytes() + roster_sorted_.bytes() + roster_.bytes() +
      roster_run_counts_.bytes() + roster_count_.bytes() +
      roster_temporary_.bytes() + device_roots_.bytes() +
      device_source_levels_.bytes() + pending_counts_.bytes() +
      pending_offsets_.bytes() + pending_ages_a_.bytes() +
      pending_ages_b_.bytes() + pending_refs_a_.bytes() +
      pending_refs_b_.bytes() + pending_total_.bytes() +
      pending_temporary_.bytes() +
      (pending_order_ ? pending_order_->bytes() : 0u) + candidates_.bytes() +
      source_counts_.bytes() + source_begins_.bytes() + fan_in_.bytes() +
      live_flags_.bytes() + live_winners_.bytes() + live_count_.bytes() +
      winner_heads_.bytes() + group_heads_.bytes() + group_counts_.bytes() +
      group_starts_.bytes() + group_count_.bytes() + exact_rows_.bytes() +
      exact_offsets_.bytes() + exact_flags_.bytes() +
      special_flags_.bytes() + exact_group_ids_.bytes() +
      exact_group_count_.bytes() + selected_special_heads_.bytes() +
      special_head_count_.bytes() + destinations_.bytes() + errors_.bytes() +
      group_temporary_.bytes() + capsule_flags_.bytes() +
      capsule_ranks_.bytes() + capsule_positions_a_.bytes() +
      capsule_positions_b_.bytes() + capsule_count_.bytes() +
      capsule_refs_.bytes() + capsule_rows_a_.bytes() +
      capsule_rows_b_.bytes() + capsule_pages_.bytes() +
      unique_pages_.bytes() + page_counts_.bytes() + page_starts_.bytes() +
      page_count_.bytes() + capsule_sizes_.bytes() +
      capsule_offsets_.bytes() + capsule_temporary_.bytes() +
      lifecycle_sources_.bytes() + lifecycle_errors_.bytes();
}

}  // namespace gpulsm_sparse
