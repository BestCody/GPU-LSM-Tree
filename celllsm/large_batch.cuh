#pragma once

// Building runs directly from a sorted input: bulk build and updates
// holding at least one full epoch with nothing pending.

#include "device_memory.cuh"
#include "layout.cuh"

namespace celllsm {

constexpr std::uint32_t kBulkRootSinkItemsPerThread = 8u;
constexpr std::uint32_t kBulkRootSinkTileRows =
    kThreads * kBulkRootSinkItemsPerThread;

__device__ __forceinline__ bool bulk_root_sink_terminal(
    const std::uint32_t *keys, std::uint32_t count,
    std::uint64_t position, std::uint32_t key) {
  return position + 1u == count || key != keys[position + 1u];
}

struct BulkRootSinkMaximum {
  __host__ __device__ __forceinline__ std::uint32_t operator()(
      std::uint32_t left, std::uint32_t right) const {
    return left > right ? left : right;
  }
};

// Count the newest row in each equal-key run.
__global__ void count_bulk_root_sink_tiles_kernel(
    const std::uint32_t *keys, std::uint32_t count,
    std::uint32_t *tile_counts) {
  using BlockScan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  const std::uint64_t tile_begin =
      std::uint64_t{blockIdx.x} * kBulkRootSinkTileRows;
  const std::uint64_t thread_begin = tile_begin +
      std::uint64_t{threadIdx.x} * kBulkRootSinkItemsPerThread;
  std::uint32_t local_count = 0u;
  for (std::uint32_t item = 0u;
       item < kBulkRootSinkItemsPerThread; ++item) {
    const std::uint64_t position = thread_begin + item;
    if (position < count) {
      const std::uint32_t key = keys[position];
      local_count += bulk_root_sink_terminal(
          keys, count, position, key);
    }
  }
  std::uint32_t unused_prefix = 0u, tile_count = 0u;
  BlockScan(scan_storage).ExclusiveSum(
      local_count, unused_prefix, tile_count);
  if (threadIdx.x == 0u) tile_counts[blockIdx.x] = tile_count;
}

// Write winners directly to the chosen row storage.
template <class Output>
__global__ void deposit_bulk_root_sink_kernel(
    const std::uint32_t *sorted_keys,
    const std::uint32_t *sorted_values,
    std::uint32_t count,
    const std::uint32_t *tile_offsets,
    std::uint32_t *region_end_markers,
    Output output_rows,
    std::uint64_t destination, bool tombstone) {
  constexpr std::uint32_t kWarpSize = 32u;
  constexpr std::uint32_t kWarps = kThreads / kWarpSize;
  constexpr std::uint32_t kGroups =
      kBulkRootSinkItemsPerThread * kWarps;
  __shared__ std::uint32_t group_offsets[kGroups + 1u];
  const std::uint64_t tile_begin =
      std::uint64_t{blockIdx.x} * kBulkRootSinkTileRows;
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  std::uint32_t keys[kBulkRootSinkItemsPerThread];
  bool terminal[kBulkRootSinkItemsPerThread];
  bool region_terminal[kBulkRootSinkItemsPerThread];
  if (threadIdx.x == 0u) group_offsets[0] = 0u;
  #pragma unroll
  for (std::uint32_t item = 0u;
       item < kBulkRootSinkItemsPerThread; ++item) {
    const std::uint64_t position = tile_begin +
        std::uint64_t{item} * kThreads + threadIdx.x;
    keys[item] = 0u;
    if (position < count) {
      keys[item] = sorted_keys[position];
    }
    std::uint32_t next = __shfl_down_sync(~0u, keys[item], 1u);
    if (lane + 1u == kWarpSize && position + 1u < count)
      next = sorted_keys[position + 1u];
    const bool final = position + 1u == count;
    terminal[item] = position < count &&
        (final || keys[item] != next);
    region_terminal[item] = terminal[item] &&
        (final || (keys[item] >> 16u) != (next >> 16u));
    const std::uint32_t mask = __ballot_sync(~0u, terminal[item]);
    if (lane == 0u) {
      const std::uint32_t group = item * kWarps + warp;
      group_offsets[group + 1u] = __popc(mask);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    #pragma unroll
    for (std::uint32_t group = 1u; group <= kGroups; ++group)
      group_offsets[group] += group_offsets[group - 1u];
  }
  __syncthreads();
  const std::uint32_t lower_lanes = lane
      ? ~0u >> (kWarpSize - lane) : 0u;
  #pragma unroll
  for (std::uint32_t item = 0u;
       item < kBulkRootSinkItemsPerThread; ++item) {
    const std::uint32_t mask = __ballot_sync(~0u, terminal[item]);
    if (!terminal[item]) continue;
    const std::uint64_t position = tile_begin +
        std::uint64_t{item} * kThreads + threadIdx.x;
    const std::uint32_t group = item * kWarps + warp;
    const std::uint32_t output_rank =
        tile_offsets[blockIdx.x] + group_offsets[group] +
        __popc(mask & lower_lanes);
    const std::uint64_t output = destination + output_rank;
    output_rows.store(output, make_row(
        keys[item], tombstone ? 0u : sorted_values[position],
        tombstone ? kTombstone : 0u));
    if (region_terminal[item]) {
      const std::uint32_t region = keys[item] >> 16u;
      region_end_markers[region + 1u] = output_rank + 1u;
    }
  }
}



// Plan final radix-4 roots before materialization.
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
  std::array<PlannedRoot, kMaximumLevels> outputs{};
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
  std::array<PlannerNode, kMaximumLevels> levels{};
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

__global__ void build_epoch_counts_and_ranks(
    const Row *rows,
    const std::uint32_t *region_offsets, std::uint32_t *region_counts,
    std::uint16_t *cell_ranks) {
  const std::uint32_t region = blockIdx.x;
  const std::uint32_t cell = threadIdx.x;
  const std::uint32_t begin = region_offsets[region];
  const std::uint32_t count =
      region_offsets[region + 1u] - begin;
  if (!cell) {
    region_counts[region] = count;
    if (!region) region_counts[kRegions] = 0u;
  }
  cell_ranks[std::size_t{region} *
                 kCellsPerRegion + cell] =
      static_cast<std::uint16_t>(lower_bound_rows(
          rows + begin, count,
          cell * kKeysPerCell));
}

// Sort, keep the last version of each key, and write a run directly.
template <bool WithEpoch> class DirectRunWorkspace {
 public:
  explicit DirectRunWorkspace(std::uint32_t capacity)
      : DirectRunWorkspace(capacity, Buffer<std::uint8_t>(required_bytes(capacity))) {}

  DirectRunWorkspace(std::uint32_t capacity, std::uint8_t *storage,
                      std::size_t storage_bytes)
      : capacity_(capacity), layout_(storage_layout(capacity)), storage_(storage) {
    if (!storage || storage_bytes < layout_.total_bytes)
      throw std::length_error("direct run workspace backing storage");
  }

  static std::size_t required_bytes(std::size_t capacity) {
    return storage_layout(capacity).total_bytes;
  }
  std::uint32_t capacity() const { return capacity_; }
  std::size_t bytes() const { return layout_.total_bytes; }
  std::uint32_t *sorted_keys() const { return view<std::uint32_t>(layout_.sorted_keys); }
  std::uint32_t *sorted_values() const { return view<std::uint32_t>(layout_.sorted_values); }
  std::uint32_t *region_offsets() const { return view<std::uint32_t>(layout_.region_offsets); }
  std::uint32_t *region_counts() const { return view<std::uint32_t>(layout_.region_counts); }
  Row *epoch_rows() const { return view<Row>(layout_.epoch_rows); }
  std::uint16_t *epoch_ranks() const { return view<std::uint16_t>(layout_.epoch_ranks); }

  std::uint32_t prepare(const std::uint32_t *keys, const std::uint32_t *values,
                        std::uint32_t count, bool tombstone,
                        bool materialize_epoch,
                        cudaStream_t stream) {
    if (!count || count > capacity_ || (materialize_epoch && !WithEpoch))
      throw std::length_error("direct run workspace capacity");
    count_ = count;
    tombstone_ = tombstone;
    const std::uint32_t tiles = tile_count(count);
    std::size_t bytes = layout_.sort_bytes;
    if (tombstone_) {
      check(cub::DeviceRadixSort::SortKeys(
          view<std::uint8_t>(layout_.temporary), bytes, keys, sorted_keys(),
          count, 0, 32, stream), "sort direct-run deletion keys");
    } else {
      check(cub::DeviceRadixSort::SortPairs(
          view<std::uint8_t>(layout_.temporary), bytes, keys, sorted_keys(),
          values, sorted_values(), count, 0, 32, stream), "sort direct run");
    }
    count_bulk_root_sink_tiles_kernel<<<
        tiles, kThreads, 0, stream>>>(
        sorted_keys(), count, view<std::uint32_t>(layout_.tile_counts));
    check(cudaMemsetAsync(view<std::uint32_t>(layout_.tile_counts) + tiles,
                          0, sizeof(std::uint32_t), stream), "clear tile sentinel");
    bytes = layout_.scan_bytes;
    check(cub::DeviceScan::ExclusiveSum(
        view<std::uint8_t>(layout_.temporary), bytes,
        view<std::uint32_t>(layout_.tile_counts),
        view<std::uint32_t>(layout_.tile_offsets), tiles + 1u, stream),
        "scan direct run winners");
    if constexpr (WithEpoch) {
      if (materialize_epoch) {
        deposit(EpochOutput{epoch_rows()}, 0u, region_offsets(), stream);
        build_epoch_counts_and_ranks<<<
            kRegions, kCellsPerRegion, 0, stream>>>(
            epoch_rows(), region_offsets(), region_counts(), epoch_ranks());
      }
    }
    std::uint32_t selected{};
    check(cudaMemcpyAsync(&selected,
        view<std::uint32_t>(layout_.tile_offsets) + tiles, sizeof(selected),
        cudaMemcpyDeviceToHost, stream), "read direct run winner count");
    check(cudaStreamSynchronize(stream), "wait direct run preparation");
    return selected;
  }

  template <class Output>
  void deposit(Output output, std::uint64_t destination,
               std::uint32_t *offsets, cudaStream_t stream) const {
    auto *markers = view<std::uint32_t>(layout_.temporary);
    check(cudaMemsetAsync(markers, 0, (kRegions + 1u) * sizeof(std::uint32_t),
                          stream), "clear direct run region markers");
    deposit_bulk_root_sink_kernel<<<
        tile_count(count_), kThreads, 0, stream>>>(
        sorted_keys(), sorted_values(), count_,
        view<std::uint32_t>(layout_.tile_offsets), markers, output, destination,
        tombstone_);
    std::size_t bytes = layout_.region_scan_bytes;
    check(cub::DeviceScan::InclusiveScan(
        view<std::uint8_t>(layout_.region_scan_temporary), bytes,
        markers, offsets, BulkRootSinkMaximum{},
        kRegions + 1u, stream), "scan direct run regions");
  }

  struct EpochOutput {
    Row *rows;
    __device__ void store(std::uint64_t position, Row row) const {
      rows[position] = row;
    }
  };
 private:
  DirectRunWorkspace(std::uint32_t capacity, Buffer<std::uint8_t> storage)
      : DirectRunWorkspace(capacity, storage.data(), storage.bytes()) {
    owned_storage_ = std::move(storage);
  }
  template <class T> T *view(std::size_t offset) const {
    return reinterpret_cast<T *>(storage_ + offset);
  }
  static std::uint32_t tile_count(std::size_t count) {
    return static_cast<std::uint32_t>((count + kBulkRootSinkTileRows - 1u) /
                                     kBulkRootSinkTileRows);
  }
  struct StorageLayout {
    std::size_t sorted_keys{}, sorted_values{}, temporary{};
    std::size_t region_scan_temporary{}, tile_counts{}, tile_offsets{};
    std::size_t region_offsets{}, region_counts{}, epoch_rows{}, epoch_ranks{};
    std::size_t sort_bytes{}, scan_bytes{}, region_scan_bytes{}, total_bytes{};
  };
  static StorageLayout storage_layout(std::size_t capacity) {
    if (!capacity || capacity > std::numeric_limits<std::uint32_t>::max())
      throw std::length_error("direct run workspace capacity");
    StorageLayout layout{};
    std::uint32_t *rows = nullptr;
    check(cub::DeviceRadixSort::SortPairs(
        nullptr, layout.sort_bytes, rows, rows, rows, rows,
        static_cast<std::uint32_t>(capacity), 0, 32), "size direct run sort");
    std::size_t key_sort_bytes = 0u;
    check(cub::DeviceRadixSort::SortKeys(
        nullptr, key_sort_bytes, rows, rows,
        static_cast<std::uint32_t>(capacity), 0, 32), "size direct-run deletion sort");
    layout.sort_bytes = std::max(layout.sort_bytes, key_sort_bytes);
    check(cub::DeviceScan::ExclusiveSum(
        nullptr, layout.scan_bytes, rows, rows, tile_count(capacity) + 1u),
        "size direct run scan");
    check(cub::DeviceScan::InclusiveScan(
        nullptr, layout.region_scan_bytes, rows, rows,
        BulkRootSinkMaximum{}, kRegions + 1u),
        "size direct run region scan");
    std::size_t offset = 0u;
    const auto place = [&](std::size_t bytes, std::size_t alignment = 256u) {
      offset = (offset + alignment - 1u) / alignment * alignment;
      const std::size_t result = offset;
      offset += bytes;
      return result;
    };
    layout.sorted_keys = place(capacity * sizeof(std::uint32_t));
    layout.sorted_values = place(capacity * sizeof(std::uint32_t));
    layout.temporary = place(0u);
    const std::size_t marker_bytes = (kRegions + 1u) * sizeof(std::uint32_t);
    const std::size_t scan_offset = (marker_bytes + 255u) & ~std::size_t{255u};
    layout.region_scan_temporary = layout.temporary + scan_offset;
    offset += std::max({layout.sort_bytes, layout.scan_bytes,
                       scan_offset + layout.region_scan_bytes});
    layout.tile_counts = place((tile_count(capacity) + 1u) * sizeof(std::uint32_t), 4u);
    layout.tile_offsets = place((tile_count(capacity) + 1u) * sizeof(std::uint32_t), 4u);
    if constexpr (WithEpoch) {
      layout.region_offsets = place(marker_bytes);
      layout.region_counts = place(marker_bytes);
      layout.epoch_rows = place(capacity * sizeof(Row));
      layout.epoch_ranks = place(kLocalRankEntries * sizeof(std::uint16_t));
    }
    layout.total_bytes = offset;
    return layout;
  }
  Buffer<std::uint8_t> owned_storage_;
  std::uint32_t capacity_{}, count_{};
  bool tombstone_{};
  StorageLayout layout_;
  std::uint8_t *storage_{};
};

struct LargeBatchCommand {
  std::uint64_t expected_mask{};
  std::uint64_t consumed_mask{};
  std::uint64_t output_mask{};
  std::uint64_t final_mask{};
  std::uint64_t output_generation_bits{};
};

struct LargeBatchReceipt {
  std::uint64_t occupied_mask{};
  std::uint32_t status{};
};

// Publish all materialized roots in one commit.
__global__ void publish_large_batch_kernel(
    const LargeBatchCommand *command,
    DeviceManifest *manifests,
    std::uint32_t *active_manifest, std::uint64_t *query_mask,
    LargeBatchReceipt *receipt) {
  if (blockIdx.x) return;
  const std::uint32_t active = atomicAdd(active_manifest, 0u) & 1u;
  const auto *current = manifests + active;
  auto *next = manifests + (active ^ 1u);
  if (current->occupied_level_mask != command->expected_mask) {
    if (!threadIdx.x) {
      receipt->occupied_mask = current->occupied_level_mask;
      receipt->status = 1u;
    }
    return;
  }
  for (std::uint32_t level = threadIdx.x; level < kMaximumLevels;
       level += blockDim.x) {
    const std::uint64_t bit = std::uint64_t{1u} << level;
    if (command->output_mask & bit) {
      DeviceLevelState state{};
      state.storage_generation =
          (command->output_generation_bits & bit) != 0u;
      next->levels[level] = state;
    } else if (command->consumed_mask & bit) {
      next->levels[level] = {};
    } else {
      next->levels[level] = current->levels[level];
    }
  }
  __syncthreads();
  if (!threadIdx.x) {
    next->occupied_level_mask = command->final_mask;
    next->active_levels = command->final_mask
        ? 64u - static_cast<std::uint32_t>(__clzll(command->final_mask))
        : 0u;
    next->oldest_level = next->active_levels
        ? next->active_levels - 1u : kMaximumLevels;
    next->generation = current->generation + 1u;
    receipt->occupied_mask = command->final_mask;
    receipt->status = 0u;
    __threadfence();
    atomicExch(active_manifest, active ^ 1u);
    atomicExch(reinterpret_cast<unsigned long long *>(query_mask),
               static_cast<unsigned long long>(command->final_mask));
  }
}

}  // namespace celllsm
