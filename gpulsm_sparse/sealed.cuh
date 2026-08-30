#pragma once

#include "bulk.cuh"

namespace gpulsm_sparse {

// Validated specialized sealed-anchor builder transplanted from
// closure_spine.cuh.  It remains an all-inline builder: mixed intervals use
// DirectRootWorkspace plus the same sparse refinement instead.
__global__ void mark_last_sealed_key(
    const std::uint32_t *keys, std::uint32_t count, std::uint8_t *keep) {
  const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < count)
    keep[row] = static_cast<std::uint8_t>(
        row + 1u == count || keys[row] != keys[row + 1u]);
}

__global__ void gather_sealed_to_arena(
    const std::uint32_t *keys, const std::uint32_t *values,
    const std::uint32_t *selected, const std::uint32_t *selected_count,
    gpulsmopt2_detail::ResidentRows arena, std::uint64_t destination) {
  for (std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
       row < *selected_count; row += blockDim.x * gridDim.x) {
    const std::uint32_t source = selected[row];
    arena.store(destination + row,
                gpulsmopt2_detail::make_row(
                    keys[source], values[source], 0u));
  }
}

__global__ void gather_sealed_epoch_rows(
    const std::uint32_t *keys, const std::uint32_t *values,
    const std::uint32_t *selected, const std::uint32_t *selected_count,
    gpulsmopt2_detail::Row *rows) {
  for (std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
       row < *selected_count; row += blockDim.x * gridDim.x) {
    const std::uint32_t source = selected[row];
    rows[row] = gpulsmopt2_detail::make_row(
        keys[source], values[source], 0u);
  }
}

__global__ void build_sealed_section_offsets(
    const std::uint32_t *keys, const std::uint32_t *selected,
    const std::uint32_t *selected_count, std::uint32_t *offsets) {
  const std::uint32_t section = blockIdx.x * blockDim.x + threadIdx.x;
  if (section > kSections) return;
  const std::uint32_t count = *selected_count;
  if (section == kSections) {
    offsets[section] = count;
    return;
  }
  const std::uint32_t target = section << 16u;
  std::uint32_t low = 0u;
  std::uint32_t high = count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (keys[selected[middle]] < target) low = middle + 1u;
    else high = middle;
  }
  offsets[section] = low;
}

__global__ void build_sealed_epoch_counts_and_ranks(
    const gpulsmopt2_detail::Row *rows,
    const std::uint32_t *section_offsets, std::uint32_t *section_counts,
    std::uint16_t *cell_ranks) {
  const std::uint32_t section = blockIdx.x;
  const std::uint32_t cell = threadIdx.x;
  const std::uint32_t begin = section_offsets[section];
  const std::uint32_t count =
      section_offsets[section + 1u] - begin;
  if (cell == 0u) {
    section_counts[section] = count;
    if (section == 0u) section_counts[kSections] = 0u;
  }
  cell_ranks[std::size_t{section} *
                 gpulsmopt2_detail::kFoundationCells + cell] =
      static_cast<std::uint16_t>(gpulsmopt2_detail::lower_bound_rows(
          rows + begin, count,
          cell * gpulsmopt2_detail::kFoundationCellKeys));
}

class SealedWorkspace {
 public:
  explicit SealedWorkspace(std::uint32_t capacity)
      : capacity_(capacity), sorted_keys_(capacity),
        sorted_values_(capacity), keep_(capacity), selected_(capacity),
        selected_count_(1u), section_offsets_(kSections + 1u),
        section_counts_(kSections + 1u), epoch_rows_(capacity),
        epoch_ranks_(gpulsmopt2_detail::kLocalRankEntries) {
    std::size_t bytes = 0u;
    check(cub::DeviceRadixSort::SortPairs(
              nullptr, bytes, sorted_keys_.data(), sorted_keys_.data(),
              sorted_values_.data(), sorted_values_.data(), capacity_,
              0, 32),
          "size sealed sort");
    sort_bytes_ = bytes;
    sort_temporary_.reset(bytes);
    cub::CountingInputIterator<std::uint32_t> ids(0u);
    bytes = 0u;
    check(cub::DeviceSelect::Flagged(
              nullptr, bytes, ids, keep_.data(), selected_.data(),
              selected_count_.data(), capacity_),
          "size sealed selection");
    select_bytes_ = bytes;
    select_temporary_.reset(bytes);
  }

  std::uint32_t capacity() const { return capacity_; }

  std::uint32_t prepare(
      const std::uint32_t *keys, const std::uint32_t *values,
      std::uint32_t count, bool materialize_epoch, cudaStream_t stream) {
    if (!count || count > capacity_)
      throw std::length_error("sealed workspace capacity");
    std::size_t bytes = sort_bytes_;
    check(cub::DeviceRadixSort::SortPairs(
              sort_temporary_.data(), bytes, keys, sorted_keys_.data(),
              values, sorted_values_.data(), count, 0, 32, stream),
          "sort sealed interval");
    mark_last_sealed_key<<<blocks(count), kThreads, 0, stream>>>(
        sorted_keys_.data(), count, keep_.data());
    cub::CountingInputIterator<std::uint32_t> ids(0u);
    bytes = select_bytes_;
    check(cub::DeviceSelect::Flagged(
              select_temporary_.data(), bytes, ids, keep_.data(),
              selected_.data(), selected_count_.data(), count, stream),
          "select sealed winners");
    build_sealed_section_offsets<<<blocks(kSections + 1u), kThreads, 0,
                                   stream>>>(
        sorted_keys_.data(), selected_.data(), selected_count_.data(),
        section_offsets_.data());
    if (materialize_epoch) {
      gather_sealed_epoch_rows<<<4096, kThreads, 0, stream>>>(
          sorted_keys_.data(), sorted_values_.data(), selected_.data(),
          selected_count_.data(), epoch_rows_.data());
      build_sealed_epoch_counts_and_ranks<<<
          kSections, gpulsmopt2_detail::kFoundationCells, 0, stream>>>(
          epoch_rows_.data(), section_offsets_.data(),
          section_counts_.data(), epoch_ranks_.data());
    }
    std::uint32_t selected = 0u;
    check(cudaMemcpyAsync(&selected, selected_count_.data(),
                          sizeof(selected), cudaMemcpyDeviceToHost,
                          stream),
          "copy sealed winner count");
    check(cudaStreamSynchronize(stream), "wait sealed preparation");
    return selected;
  }

  const std::uint32_t *sorted_keys() const { return sorted_keys_.data(); }
  const std::uint32_t *sorted_values() const {
    return sorted_values_.data();
  }
  const std::uint32_t *selected() const { return selected_.data(); }
  const std::uint32_t *selected_count() const {
    return selected_count_.data();
  }
  const std::uint32_t *section_offsets() const {
    return section_offsets_.data();
  }
  const std::uint32_t *section_counts() const {
    return section_counts_.data();
  }
  const gpulsmopt2_detail::Row *epoch_rows() const {
    return epoch_rows_.data();
  }
  const std::uint16_t *epoch_ranks() const { return epoch_ranks_.data(); }

  std::size_t bytes() const {
    return sorted_keys_.bytes() + sorted_values_.bytes() + keep_.bytes() +
        selected_.bytes() + selected_count_.bytes() +
        section_offsets_.bytes() + section_counts_.bytes() +
        epoch_rows_.bytes() + epoch_ranks_.bytes() +
        sort_temporary_.bytes() + select_temporary_.bytes();
  }

 private:
  std::uint32_t capacity_{};
  Buffer<std::uint32_t> sorted_keys_, sorted_values_;
  Buffer<std::uint8_t> keep_;
  Buffer<std::uint32_t> selected_, selected_count_;
  Buffer<std::uint32_t> section_offsets_, section_counts_;
  Buffer<gpulsmopt2_detail::Row> epoch_rows_;
  Buffer<std::uint16_t> epoch_ranks_;
  Buffer<std::uint8_t> sort_temporary_, select_temporary_;
  std::size_t sort_bytes_{}, select_bytes_{};
};

struct SealedForestCommand {
  std::uint64_t expected_mask{};
  std::uint64_t consumed_mask{};
  std::uint64_t output_mask{};
  std::uint64_t final_mask{};
  std::uint64_t output_generation_bits{};
  std::uint64_t sparse_states[kLevels]{};
  std::uint32_t sparse_flags[kLevels]{};
};

struct SealedForestReceipt {
  std::uint64_t occupied_mask{};
  std::uint32_t generation{};
  std::uint32_t status{};
};

// One publication for every final materialized root.  This is the prototype
// multi-root receipt rule grafted onto the original ordinary manifest and
// the sparse-only companion table; it introduces no wrapper manifest.
__global__ void publish_sealed_forest(
    const SealedForestCommand *command,
    gpulsmopt2_detail::DeviceManifest *manifests,
    DeviceSparseManifest *sparse_manifests,
    std::uint32_t *active_manifest, std::uint64_t *query_mask,
    SealedForestReceipt *receipt) {
  if (blockIdx.x) return;
  const std::uint32_t active = atomicAdd(active_manifest, 0u) & 1u;
  const auto *current = manifests + active;
  auto *next = manifests + (active ^ 1u);
  const auto *current_sparse = sparse_manifests + active;
  auto *next_sparse = sparse_manifests + (active ^ 1u);
  if (current->occupied_level_mask != command->expected_mask) {
    if (!threadIdx.x) {
      receipt->occupied_mask = current->occupied_level_mask;
      receipt->generation = current->generation;
      receipt->status = 1u;
    }
    return;
  }
  for (std::uint32_t level = threadIdx.x; level < kLevels;
       level += blockDim.x) {
    const std::uint64_t bit = std::uint64_t{1u} << level;
    if (command->output_mask & bit) {
      gpulsmopt2_detail::DeviceLevelState state{};
      state.storage_generation =
          (command->output_generation_bits & bit) != 0u;
      next->levels[level] = state;
      const std::uint32_t flags = command->sparse_flags[level];
      next_sparse->levels[level] = flags
          ? DeviceSparseLevelState{command->sparse_states[level],
                                   state.storage_generation, flags}
          : DeviceSparseLevelState{};
    } else if (command->consumed_mask & bit) {
      next->levels[level] = {};
      next_sparse->levels[level] = {};
    } else {
      next->levels[level] = current->levels[level];
      next_sparse->levels[level] = current_sparse->levels[level];
    }
  }
  __syncthreads();
  if (!threadIdx.x) {
    next->occupied_level_mask = command->final_mask;
    next->active_levels = command->final_mask
        ? 64u - static_cast<std::uint32_t>(__clzll(command->final_mask))
        : 0u;
    next->foundation_level = next->active_levels
        ? next->active_levels - 1u : kLevels;
    next->generation = current->generation + 1u;

    next_sparse->exact_level_mask = 0u;
    next_sparse->capsule_level_mask = 0u;
    for (std::uint32_t level = 0u; level < kLevels; ++level) {
      const std::uint64_t bit = std::uint64_t{1u} << level;
      if (!(command->final_mask & bit)) continue;
      const std::uint32_t flags = next_sparse->levels[level].flags;
      if (flags & kSparseHasExactHeads)
        next_sparse->exact_level_mask |= bit;
      if (flags & kSparseHasCapsules)
        next_sparse->capsule_level_mask |= bit;
    }
    next_sparse->generation = next->generation;
    receipt->occupied_mask = command->final_mask;
    receipt->generation = next->generation;
    receipt->status = 0u;
    __threadfence();
    atomicExch(active_manifest, active ^ 1u);
    atomicExch(reinterpret_cast<unsigned long long *>(query_mask),
               static_cast<unsigned long long>(command->final_mask));
  }
}

}  // namespace gpulsm_sparse
