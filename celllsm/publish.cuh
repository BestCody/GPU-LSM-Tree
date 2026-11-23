#pragma once

// Epoch publication: per-region newest-wins dedupe of the pending
// batches, the publication plan, run metadata and the manifest switch.

#include "pending.cuh"

namespace celllsm {

constexpr std::uint32_t kLocalEpochItemsPerThread = 5u;
constexpr std::uint32_t kLocalEpochCapacity =
    kMergeThreads * kLocalEpochItemsPerThread;

constexpr std::uint32_t kCanonicalResolverSuffixes = 1u << 16u;

__device__ __forceinline__ std::uint64_t pack_canonical_epoch_job(
    std::uint32_t q, std::uint32_t count) {
  return (static_cast<std::uint64_t>(count) << 32u) | q;
}

__global__ void count_canonical_epoch_jobs_kernel(
    const std::uint32_t *raw_offsets, std::uint32_t pending_batches,
    std::uint32_t *region_counts, std::uint64_t *epoch_jobs,
    std::uint32_t *local_job_count,
    std::uint32_t *oversized_job_count) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kRegions) return;
  if (q == kRegions) {
    region_counts[q] = 0u;
    return;
  }
  std::uint32_t count = 0u;
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t base =
        std::size_t{batch} * (kRegions + 1u) + q;
    count += raw_offsets[base + 1u] - raw_offsets[base];
  }
  region_counts[q] = count;
  if (!count) return;
  const std::uint64_t job = pack_canonical_epoch_job(q, count);
  if (count <= kLocalEpochCapacity) {
    const std::uint32_t position = atomicAdd(local_job_count, 1u);
    epoch_jobs[position] = job;
  } else {
    const std::uint32_t position = atomicAdd(oversized_job_count, 1u);
    epoch_jobs[kRegions - 1u - position] = job;
  }
}

__global__ void sum_canonical_region_counts_kernel(
    const std::uint32_t *region_counts, std::uint32_t *total_count) {
  __shared__ std::uint32_t partials[kThreads];
  std::uint32_t total = 0u;
  for (std::uint32_t q = threadIdx.x; q < kRegions;
       q += blockDim.x)
    total += region_counts[q];
  partials[threadIdx.x] = total;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x >> 1u; stride; stride >>= 1u) {
    if (threadIdx.x < stride)
      partials[threadIdx.x] += partials[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0u) *total_count = partials[0];
}

template <bool ResidentOutput>
__device__ __forceinline__ void store_canonical_epoch_row(
    Row *output_rows, ResidentRows resident_rows,
    std::uint64_t resident_begin, std::uint32_t output,
    const Row &row) {
  if constexpr (ResidentOutput)
    resident_rows.store(resident_begin + output, row);
  else
    output_rows[output] = row;
}

template <std::uint32_t Items>
using ActiveEpochBlockSort = cub::BlockRadixSort<
    std::uint32_t, kMergeThreads, Items, std::uint32_t>;

using ActiveEpochBlockScan =
    cub::BlockScan<std::uint32_t, kMergeThreads>;

union ActiveEpochResolutionStorage {
  typename ActiveEpochBlockSort<1u>::TempStorage sort_1;
  typename ActiveEpochBlockSort<5u>::TempStorage sort_5;
  unsigned long long winners[kLocalEpochCapacity];
};

template <std::uint32_t Items, bool ResidentOutput>
__device__ __forceinline__ void resolve_canonical_epoch_active_job(
    std::uint32_t q, std::uint32_t raw_count,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const std::uint32_t *raw_region_offsets,
    Row *output_rows, ResidentRows resident_rows,
    std::uint64_t resident_begin, std::uint32_t *resolved_counts,
    std::uint16_t *cell_ranks,
    ActiveEpochResolutionStorage &resolution_storage,
    typename ActiveEpochBlockScan::TempStorage &scan_storage,
    const std::uint32_t *batch_prefix, std::uint16_t *sorted_suffixes,
    std::uint32_t *cell_counts, std::uint32_t &output_count_shared) {
  static_assert(Items >= 1u && Items <= kLocalEpochItemsPerThread);
  using BlockSort = ActiveEpochBlockSort<Items>;
  constexpr std::uint32_t kInvalidSortKey = 1u << 16u;
  std::uint32_t sort_keys[Items];
  std::uint32_t sort_sources[Items];
  for (std::uint32_t item = 0u; item < Items; ++item) {
    const std::uint32_t input_local = item * blockDim.x + threadIdx.x;
    sort_keys[item] = kInvalidSortKey;
    sort_sources[item] = 0u;
    if (input_local >= raw_count) continue;
    std::uint32_t batch = 0u;
    while (batch + 1u < pending_batches &&
           input_local >= batch_prefix[batch + 1u])
      ++batch;
    const std::size_t base =
        std::size_t{batch} * (kRegions + 1u) + q;
    const std::uint32_t source = static_cast<std::uint32_t>(
        std::size_t{batch} * batch_stride + raw_offsets[base] +
        input_local - batch_prefix[batch]);
    sort_keys[item] = key_suffix(raw_keys[source]);
    sort_sources[item] = source;
  }
  auto &sort_storage = *reinterpret_cast<typename BlockSort::TempStorage *>(
      &resolution_storage);
  BlockSort(sort_storage).Sort(sort_keys, sort_sources, 0, 17);
  for (std::uint32_t item = 0u; item < Items; ++item) {
    const std::uint32_t local = threadIdx.x * Items + item;
    sorted_suffixes[local] = static_cast<std::uint16_t>(sort_keys[item]);
  }
  __syncthreads();

  bool leaders[Items];
  std::uint32_t leader_count = 0u;
  for (std::uint32_t item = 0u; item < Items; ++item) {
    const std::uint32_t local = threadIdx.x * Items + item;
    const bool valid = sort_keys[item] != kInvalidSortKey;
    leaders[item] = valid &&
        (!local || sorted_suffixes[local - 1u] != sorted_suffixes[local]);
    leader_count += leaders[item];
  }
  std::uint32_t output_prefix = 0u, output_count = 0u;
  ActiveEpochBlockScan(scan_storage).ExclusiveSum(
      leader_count, output_prefix, output_count);
  if (threadIdx.x == 0u) {
    output_count_shared = output_count;
    resolved_counts[q] = output_count;
  }
  for (std::uint32_t group = threadIdx.x; group < output_count;
       group += blockDim.x)
    resolution_storage.winners[group] = 0ull;
  __syncthreads();

  std::uint32_t leaders_seen = 0u;
  for (std::uint32_t item = 0u; item < Items; ++item) {
    if (sort_keys[item] == kInvalidSortKey) continue;
    const std::uint32_t group = leaders[item]
        ? output_prefix + leaders_seen
        : output_prefix + leaders_seen - 1u;
    if (leaders[item]) {
      sorted_suffixes[group] = static_cast<std::uint16_t>(sort_keys[item]);
      ++leaders_seen;
    }
    const std::uint32_t source = sort_sources[item];
    const std::uint32_t age = raw_position(raw_payloads[source]);
    const unsigned long long token =
        (static_cast<unsigned long long>(age + 1u) << 32u) | source;
    atomicMax(resolution_storage.winners + group, token);
  }
  __syncthreads();

  const std::uint32_t output_begin = raw_region_offsets[q];
  for (std::uint32_t group = threadIdx.x;
       group < output_count_shared; group += blockDim.x) {
    const std::uint32_t source = static_cast<std::uint32_t>(
        resolution_storage.winners[group]);
    const std::uint32_t suffix = sorted_suffixes[group];
    const Row row = raw_row(full_key(q, suffix), raw_payloads[source]);
    store_canonical_epoch_row<ResidentOutput>(
        output_rows, resident_rows, resident_begin,
        output_begin + group, row);
    atomicAdd(cell_counts + suffix / kKeysPerCell, 1u);
  }
  __syncthreads();

  const std::uint32_t cell_count = threadIdx.x < kCellsPerRegion
      ? cell_counts[threadIdx.x] : 0u;
  std::uint32_t cell_prefix = 0u, ignored = 0u;
  ActiveEpochBlockScan(scan_storage).ExclusiveSum(
      cell_count, cell_prefix, ignored);
  if (cell_ranks && threadIdx.x < kCellsPerRegion)
    cell_ranks[std::size_t{q} * kCellsPerRegion + threadIdx.x] =
        static_cast<std::uint16_t>(cell_prefix);
}

template <bool ResidentOutput>
__global__ __launch_bounds__(kMergeThreads, 5)
void resolve_canonical_epoch_active_jobs_kernel(
    const std::uint64_t *epoch_jobs,
    const std::uint32_t *local_job_count, std::uint32_t *next_job,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const std::uint32_t *raw_region_offsets,
    Row *output_rows, ResidentRows resident_rows,
    std::uint64_t resident_begin, std::uint32_t *resolved_counts,
    std::uint16_t *cell_ranks) {
  using BlockScan = ActiveEpochBlockScan;
  using ResolutionStorage = ActiveEpochResolutionStorage;
  __shared__ ResolutionStorage resolution_storage;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::uint32_t batch_prefix[kBatchesPerEpoch + 1u];
  __shared__ std::uint16_t sorted_suffixes[kLocalEpochCapacity];
  __shared__ std::uint32_t cell_counts[kCellsPerRegion];
  __shared__ std::uint32_t output_count_shared;
  __shared__ std::uint64_t job_shared;

  std::uint32_t job_count = 0u;
  if (threadIdx.x == 0u) job_count = *local_job_count;
  for (;;) {
    if (threadIdx.x == 0u) {
      const std::uint32_t ticket = atomicAdd(next_job, 1u);
      if (ticket < job_count) {
        job_shared = epoch_jobs[ticket];
      } else {
        job_shared = pack_canonical_epoch_job(kInvalid, 0u);
      }
    }
    __syncthreads();
    const std::uint64_t job = job_shared;
    const std::uint32_t q = static_cast<std::uint32_t>(job);
    const std::uint32_t raw_count = static_cast<std::uint32_t>(job >> 32u);
    if (q == kInvalid) return;

    if (threadIdx.x == 0u) {
      std::uint32_t total = 0u;
      batch_prefix[0] = 0u;
      for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
        const std::size_t base =
            std::size_t{batch} * (kRegions + 1u) + q;
        total += raw_offsets[base + 1u] - raw_offsets[base];
        batch_prefix[batch + 1u] = total;
      }
      for (std::uint32_t batch = pending_batches;
           batch < kBatchesPerEpoch; ++batch)
        batch_prefix[batch + 1u] = total;
      output_count_shared = 0u;
    }
    if (threadIdx.x < kCellsPerRegion)
      cell_counts[threadIdx.x] = 0u;
    __syncthreads();

#define GPULSMOPT_RESOLVE_ACTIVE_EPOCH_JOB(Items)                         \
    resolve_canonical_epoch_active_job<Items, ResidentOutput>(            \
        q, raw_count, raw_keys, raw_payloads, raw_offsets, batch_stride,  \
        pending_batches, raw_region_offsets, output_rows, resident_rows, \
        resident_begin, resolved_counts, cell_ranks, resolution_storage,  \
        scan_storage, batch_prefix, sorted_suffixes, cell_counts,          \
        output_count_shared)
    if (raw_count <= kMergeThreads)
      GPULSMOPT_RESOLVE_ACTIVE_EPOCH_JOB(1u);
    else
      GPULSMOPT_RESOLVE_ACTIVE_EPOCH_JOB(5u);
#undef GPULSMOPT_RESOLVE_ACTIVE_EPOCH_JOB
    __syncthreads();
  }
}

template <bool ResidentOutput>
__global__ void resolve_canonical_epoch_oversized_kernel(
    const std::uint64_t *overflow_jobs,
    const std::uint32_t *overflow_count, std::uint32_t *next_overflow,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const std::uint32_t *raw_region_offsets,
    Row *output_rows, ResidentRows resident_rows,
    std::uint64_t resident_begin, std::uint32_t *resolved_counts,
    std::uint16_t *cell_ranks,
    unsigned long long *oversized_winner_workspace,
    std::uint32_t *oversized_workspace_locks,
    std::uint32_t oversized_workspace_slots) {
  using BlockScan = cub::BlockScan<
      std::uint32_t, kMergeThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::uint32_t q_shared;
  for (;;) {
    if (threadIdx.x == 0u) {
      const std::uint32_t ticket = atomicAdd(next_overflow, 1u);
      q_shared = ticket < *overflow_count
          ? static_cast<std::uint32_t>(
                overflow_jobs[kRegions - 1u - ticket])
          : kInvalid;
    }
    __syncthreads();
    const std::uint32_t q = q_shared;
    if (q == kInvalid) return;
    const std::uint32_t workspace_slot =
        blockIdx.x % oversized_workspace_slots;
    if (threadIdx.x == 0u)
      while (atomicCAS(
                 oversized_workspace_locks + workspace_slot, 0u, 1u))
        __nanosleep(64u);
    __syncthreads();
    unsigned long long *winners = oversized_winner_workspace +
        std::size_t{workspace_slot} * kCanonicalResolverSuffixes;
    for (std::uint32_t suffix = threadIdx.x;
         suffix < kCanonicalResolverSuffixes; suffix += blockDim.x)
      winners[suffix] = 0ull;
    __syncthreads();

    for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
      const std::size_t base =
          std::size_t{batch} * (kRegions + 1u) + q;
      const std::uint32_t begin = raw_offsets[base];
      const std::uint32_t end = raw_offsets[base + 1u];
      for (std::uint32_t position = begin + threadIdx.x;
           position < end; position += blockDim.x) {
        const std::uint32_t source = batch * batch_stride + position;
        const std::uint32_t suffix = key_suffix(raw_keys[source]);
        const std::uint32_t age = raw_position(raw_payloads[source]);
        const unsigned long long token =
            (static_cast<unsigned long long>(age + 1u) << 32u) | source;
        atomicMax(winners + suffix, token);
      }
    }
    __syncthreads();

    const std::uint32_t suffix_begin = threadIdx.x * 256u;
    std::uint32_t local_count = 0u;
    for (std::uint32_t suffix = suffix_begin;
         suffix < suffix_begin + 256u; ++suffix)
      local_count += winners[suffix] != 0ull;
    std::uint32_t output_prefix = 0u, output_count = 0u;
    BlockScan(scan_storage).ExclusiveSum(
        local_count, output_prefix, output_count);
    if (threadIdx.x == 0u) resolved_counts[q] = output_count;
    __syncthreads();

    const std::uint32_t output_begin = raw_region_offsets[q];
    std::uint32_t local = 0u;
    for (std::uint32_t suffix = suffix_begin;
         suffix < suffix_begin + 256u; ++suffix) {
      const unsigned long long token = winners[suffix];
      if (!token) continue;
      const std::uint32_t source = static_cast<std::uint32_t>(token);
      const Row row = raw_row(full_key(q, suffix), raw_payloads[source]);
      store_canonical_epoch_row<ResidentOutput>(
          output_rows, resident_rows, resident_begin,
          output_begin + output_prefix + local++, row);
    }
    __syncthreads();

    std::uint32_t cell_count = 0u;
    if (threadIdx.x < kCellsPerRegion) {
      const std::uint32_t cell_begin =
          threadIdx.x * kKeysPerCell;
      for (std::uint32_t suffix = cell_begin;
           suffix < cell_begin + kKeysPerCell; ++suffix)
        cell_count += winners[suffix] != 0ull;
    }
    std::uint32_t cell_prefix = 0u, ignored = 0u;
    BlockScan(scan_storage).ExclusiveSum(
        cell_count, cell_prefix, ignored);
    if (cell_ranks && threadIdx.x < kCellsPerRegion)
      cell_ranks[std::size_t{q} * kCellsPerRegion + threadIdx.x] =
          static_cast<std::uint16_t>(cell_prefix);
    __syncthreads();
    if (threadIdx.x == 0u)
      atomicExch(oversized_workspace_locks + workspace_slot, 0u);
    __syncthreads();
  }
}

__global__ void initialize_device_manifest_kernel(
    DeviceManifest *manifests, std::uint32_t *active_manifest,
    std::uint64_t *query_occupied_level_mask,
    std::uint32_t level, std::uint32_t count,
    std::uint32_t storage_generation) {
  if (blockIdx.x || threadIdx.x) return;
  DeviceManifest manifest{};
  if (count) {
    manifest.occupied_level_mask = std::uint64_t{1} << level;
    manifest.active_levels = level + 1u;
    manifest.oldest_level = level;
    manifest.generation = 1u;
    DeviceLevelState state{};
    state.storage_generation = storage_generation;
    manifest.levels[level] = state;
  }
  manifests[0] = manifest;
  manifests[1] = manifest;
  *active_manifest = 0u;
  *query_occupied_level_mask = manifest.occupied_level_mask;
}

__global__ void choose_canonical_publication_path_kernel(
    const std::uint32_t *selected_count,
    const DeviceManifest *manifests, const std::uint32_t *active_manifest,
    const LevelStorageSpan *level_spans, std::uint32_t regular_level_count,
    std::uint32_t level_count,
    std::uint32_t job_capacity, std::uint32_t tournament_workspace_bytes,
    bool top_level_rollover,
    ResidentPublicationPlan *plan) {
  if (blockIdx.x || threadIdx.x) return;
  const std::uint32_t active = atomicAdd(
      const_cast<std::uint32_t *>(active_manifest), 0u) & 1u;
  const DeviceManifest *manifest = manifests + active;
  const std::uint64_t occupied = manifest->occupied_level_mask;
  std::uint32_t natural_destination = kMaximumLevels;
  std::uint32_t destination_tier_begin = 0u;
  for (std::uint32_t tier_begin = 0u;
       tier_begin < regular_level_count; tier_begin += 3u) {
    const std::uint32_t slots = min(3u, regular_level_count - tier_begin);
    const std::uint64_t tier_mask =
        ((std::uint64_t{1} << slots) - 1u) << tier_begin;
    const std::uint32_t filled = static_cast<std::uint32_t>(
        __popcll(occupied & tier_mask));
    if (filled < slots) {
      // Lower slot IDs are newer.
      natural_destination = tier_begin + slots - 1u - filled;
      destination_tier_begin = tier_begin;
      break;
    }
  }
  // The final slot holds a full-capacity carry.
  if (natural_destination == kMaximumLevels &&
      regular_level_count < level_count &&
      !(occupied & (std::uint64_t{1} << regular_level_count))) {
    natural_destination = regular_level_count;
    destination_tier_begin = regular_level_count;
  }
  const bool valid_rollover = top_level_rollover && level_count &&
      level_count <= kMaximumLevels && natural_destination >= level_count;
  const std::uint32_t destination = valid_rollover
      ? level_count - 1u : natural_destination;
  ResidentPublicationPlan next{};
  next.selected_count = *selected_count;
  next.active_manifest = active;
  next.inactive_manifest = active ^ 1u;
  next.destination_level = destination;
  next.source_level_limit = valid_rollover
      ? destination
      : destination_tier_begin ? destination_tier_begin - 1u : 0u;
  const std::uint64_t source_mask = valid_rollover
      ? destination == kMaximumLevels - 1u
          ? ~std::uint64_t{0}
          : (std::uint64_t{1} << (destination + 1u)) - 1u
      : destination_tier_begin
          ? (std::uint64_t{1} << destination_tier_begin) - 1u : 0u;
  next.source_count = 1u + static_cast<std::uint32_t>(
      __popcll(occupied & source_mask));
  const bool destination_is_oldest = valid_rollover ||
      (destination < kMaximumLevels &&
      (manifest->oldest_level == kMaximumLevels ||
       destination > manifest->oldest_level));
  next.keep_tombstones = !destination_is_oldest;
  next.output_generation = valid_rollover
      ? (manifest->levels[destination].storage_generation ^ 1u) & 1u
      : 0u;
  next.job_capacity = job_capacity;
  next.tournament_workspace_bytes = tournament_workspace_bytes;
  const bool valid_destination = destination < level_count &&
      destination < kMaximumLevels;
  next.status = valid_destination
      ? kPublicationSuccess : kPublicationLevelOverflow;
  if (valid_destination) {
    const LevelStorageSpan span = level_spans[destination];
    next.output_begin = span.begin +
        std::uint64_t{next.output_generation} * span.capacity;
    next.output_capacity = span.capacity;
  }
  if (next.selected_count > next.output_capacity)
    next.status |= kPublicationOutputOverflow;
  *plan = next;
}

__global__ void build_canonical_rank_from_run_kernel(
    ResidentRows arena, std::uint64_t run_begin,
    const std::uint32_t *region_offsets, std::uint32_t level,
    std::uint16_t *cell_ranks) {
  const std::uint32_t q = blockIdx.x;
  const std::uint32_t cell = threadIdx.x;
  const std::uint32_t begin = region_offsets[q];
  const std::uint32_t count = region_offsets[q + 1u] - begin;
  const std::uint32_t rank = lower_bound_rows(
      arena + run_begin + begin, count, cell * kKeysPerCell);
  cell_ranks[std::size_t{level} * kLocalRankEntries +
             std::size_t{q} * kCellsPerRegion + cell] =
      static_cast<std::uint16_t>(rank);
}

template <bool SeparateCounts>
__global__ void finalize_canonical_metadata_kernel(
    const std::uint32_t *region_offsets,
    const std::uint32_t *region_counts, const std::uint32_t *selected_count,
    ResidentPublicationPlan *plan, Descriptor *descriptors) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kRegions || plan->status) return;
  const std::uint32_t level = plan->destination_level;
  const std::uint32_t offset = region_offsets[q];
  std::uint32_t count;
  if constexpr (SeparateCounts) count = region_counts[q];
  else count = region_offsets[q + 1u] - offset;
  const Descriptor descriptor = Descriptor::make(
      plan->output_begin + offset, count);
  const std::size_t mapping = descriptor_index(q, level);
  descriptors[mapping] = descriptor;
  if (q == 0u) {
    std::uint64_t total;
    if constexpr (SeparateCounts) total = *selected_count;
    else total = region_offsets[kRegions];
    plan->survivor_count = total;
    if (total > plan->output_capacity)
      atomicOr(&plan->status, kPublicationOutputOverflow);
  }
}


// Switch to the next manifest after an epoch publication.
__global__ void publish_manifest_kernel(
    ResidentPublicationPlan *plan,
    DeviceManifest *manifests,
    std::uint32_t *active_manifest,
    std::uint64_t *query_occupied_level_mask) {
  if (blockIdx.x || plan->status) return;
  const auto *current = manifests + plan->active_manifest;
  auto *next = manifests + plan->inactive_manifest;
  const std::uint32_t destination = plan->destination_level;
  const std::uint64_t consumed = destination == 64u
      ? ~std::uint64_t{0}
      : ((std::uint64_t{1u} << destination) - 1u);
  for (std::uint32_t level = threadIdx.x;
       level < kMaximumLevels; level += blockDim.x) {
    if (level == destination) continue;
    next->levels[level] = level < destination
        ? DeviceLevelState{}
        : current->levels[level];
  }
  __syncthreads();
  if (!threadIdx.x) {
    next->occupied_level_mask = current->occupied_level_mask & ~consumed;
    DeviceLevelState ordinary{};
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
    next->oldest_level = next->active_levels
        ? next->active_levels - 1u
        : kMaximumLevels;
    next->generation = current->generation + 1u;
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

}  // namespace celllsm
