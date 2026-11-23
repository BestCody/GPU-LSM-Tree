#pragma once

// Merging runs: job planning over regions and the per-cell tournament
// merge that keeps the newest version of each key.

#include "layout.cuh"

namespace celllsm {

constexpr std::uint32_t kPlanningTiles = 128u;
constexpr std::uint32_t kPlanningTileRegions =
    kRegions / kPlanningTiles;
constexpr std::uint32_t kMaximumMergeSources = kMaximumLevels + 1u;
constexpr std::uint32_t kCanonicalTournamentMinimumSources =
    1u;
static_assert(kCanonicalTournamentMinimumSources >= 1u &&
              kCanonicalTournamentMinimumSources <= kMaximumMergeSources);
constexpr std::uint32_t kCanonicalJobRegions = 16u;
static_assert(kCanonicalJobRegions == 1u << 4u);

constexpr std::uint32_t kCanonicalTournamentChains = 128u;
constexpr std::uint32_t kCanonicalTournamentTasks =
    kCanonicalJobRegions * kCellsPerRegion;
constexpr std::uint32_t kCanonicalTournamentReferenceBits = 9u;
constexpr std::uint32_t kCanonicalTournamentReferenceMask =
    (1u << kCanonicalTournamentReferenceBits) - 1u;
static_assert(kKeysPerCell ==
              (1u << kCanonicalTournamentReferenceBits));
using CanonicalTournamentReference = std::uint16_t;
// Tape indices limit each job to 65,535 rows.
constexpr std::uint32_t kCanonicalTournamentCapacityCeiling =
    std::numeric_limits<std::uint16_t>::max();
constexpr std::uint32_t kMergeSourceBits = 7u;
static_assert(kMaximumMergeSources <= (1u << kMergeSourceBits));
static_assert(kMergeSourceBits + kCanonicalTournamentReferenceBits <= 16u);

__host__ __device__ constexpr std::uint32_t canonical_next_power_of_two(
    std::uint32_t value) {
  std::uint32_t result = 1u;
  while (result < value) result <<= 1u;
  return result;
}

__host__ __device__ constexpr std::size_t canonical_align_bytes(
    std::size_t value, std::size_t alignment) {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

__host__ __device__ __forceinline__ std::size_t
canonical_tournament_body_bytes(
    std::uint32_t capacity, std::uint32_t source_count,
    std::uint32_t region_count) {
  if (!source_count || !region_count ||
      region_count > kCanonicalJobRegions)
    return ~std::size_t{0};
  const std::uint32_t leaves = canonical_next_power_of_two(source_count);
  std::size_t bytes = 0u;
  const std::size_t states =
      std::size_t{kCanonicalTournamentChains} * source_count;
  bytes = canonical_align_bytes(bytes, alignof(std::uint32_t));
  bytes += states * sizeof(std::uint32_t);
  bytes = canonical_align_bytes(bytes, alignof(std::uint32_t));
  bytes += states * sizeof(std::uint32_t);
  const std::size_t source_regions =
      std::size_t{region_count} * source_count;
  bytes = canonical_align_bytes(bytes, alignof(std::uint64_t));
  bytes += source_regions * sizeof(std::uint64_t);
  bytes = canonical_align_bytes(bytes, alignof(std::uint32_t));
  bytes += source_regions * sizeof(std::uint32_t);
  bytes += std::size_t{kCanonicalTournamentChains} * leaves;
  bytes = canonical_align_bytes(
      bytes, alignof(CanonicalTournamentReference));
  bytes += std::size_t{capacity} * sizeof(CanonicalTournamentReference);
  const std::size_t tasks =
      std::size_t{region_count} * kCellsPerRegion;
  bytes = canonical_align_bytes(bytes, alignof(std::uint16_t));
  bytes += tasks * sizeof(std::uint16_t);
  bytes = canonical_align_bytes(bytes, alignof(std::uint16_t));
  bytes += (tasks + 1u) * sizeof(std::uint16_t);
  return bytes;
}

__host__ __device__ __forceinline__ std::size_t
canonical_tournament_layout_bytes(
    std::uint32_t capacity, std::uint32_t source_count,
    std::uint32_t region_count) {
  const std::size_t bytes = canonical_tournament_body_bytes(
      capacity, source_count, region_count);
  return bytes == ~std::size_t{0}
      ? bytes : canonical_align_bytes(bytes, 16u);
}

__host__ __device__ __forceinline__ std::uint32_t
canonical_tournament_capacity(
    std::size_t shared_bytes, std::uint32_t source_count,
    std::uint32_t region_count) {
  // Tournament storage is 16-byte aligned.
  const std::size_t usable_bytes = shared_bytes & ~std::size_t{15u};
  const std::size_t fixed_bytes =
      canonical_tournament_body_bytes(
          0u, source_count, region_count);
  if (fixed_bytes == ~std::size_t{0} || fixed_bytes > usable_bytes)
    return 0u;
  const std::size_t capacity =
      (usable_bytes - fixed_bytes) /
      sizeof(CanonicalTournamentReference);
  return capacity > kCanonicalTournamentCapacityCeiling
      ? kCanonicalTournamentCapacityCeiling
      : static_cast<std::uint32_t>(capacity);
}

inline std::size_t canonical_tournament_workspace_bytes(
    std::uint32_t capacity, std::uint32_t source_count) {
  // Size shared memory for the widest job span.
  return canonical_tournament_layout_bytes(
      capacity, source_count, kCanonicalJobRegions);
}
constexpr std::uint64_t kCanonicalHotJobFlag = std::uint64_t{1} << 63u;

__host__ __device__ __forceinline__ std::uint64_t canonical_hot_job(
    std::uint32_t first_job, std::uint32_t pieces) {
  return kCanonicalHotJobFlag |
      (std::uint64_t{pieces} << 32u) | first_job;
}

__host__ __device__ __forceinline__ bool canonical_job_is_hot(
    std::uint64_t encoded) {
  return (encoded & kCanonicalHotJobFlag) != 0u;
}

__host__ __device__ __forceinline__ std::uint32_t
canonical_hot_first_job(std::uint64_t encoded) {
  return static_cast<std::uint32_t>(encoded);
}

__host__ __device__ __forceinline__ std::uint32_t
canonical_hot_pieces(std::uint64_t encoded) {
  return static_cast<std::uint32_t>((encoded >> 32u) & 0xffffu);
}

inline std::size_t maximum_resident_merge_jobs(
    std::size_t maximum_raw_rows, std::uint32_t merge_capacity) {
  // Reserve one job per region plus hot pieces.
  const std::size_t safe =
      merge_capacity - (kMaximumMergeSources - 1u);
  return std::size_t{kRegions} +
      (maximum_raw_rows + safe - 1u) / safe + 1u;
}

__device__ __forceinline__ std::uint32_t canonical_job_capacity(
    const ResidentPublicationPlan *plan, std::uint32_t region_count) {
  if (plan->tournament_workspace_bytes)
    return canonical_tournament_capacity(
        plan->tournament_workspace_bytes, plan->source_count,
        region_count);
  return plan->job_capacity;
}

struct BalancedMergeJob {
  std::uint64_t key_begin{};
  std::uint64_t key_end{};
  std::uint32_t region_begin;
  std::uint32_t region_end;
};

static_assert(sizeof(BalancedMergeJob) == 24u);

struct CanonicalJobPrefix {
  unsigned long long prefix{};
  std::uint32_t count{};
  std::uint32_t ready{};
};

static_assert(sizeof(CanonicalJobPrefix) == 16u);

__device__ __forceinline__ void emit_resident_job(
    BalancedMergeJob *jobs, std::uint64_t *job_raw_reservations,
    std::uint32_t global_index,
    std::uint64_t key_begin, std::uint64_t key_end,
    std::uint32_t q_begin, std::uint32_t q_end,
    std::uint64_t raw_count) {
  BalancedMergeJob job{};
  job.key_begin = key_begin;
  job.key_end = key_end;
  job.region_begin = q_begin;
  job.region_end = q_end;
  jobs[global_index] = job;
  job_raw_reservations[global_index] = raw_count;
}

__device__ __forceinline__ std::uint64_t canonical_source_level_mask(
    const ResidentPublicationPlan *plan) {
  if (plan->source_count <= 1u) return 0u;
  return plan->source_level_limit >= kMaximumLevels - 1u
      ? ~std::uint64_t{0}
      : (std::uint64_t{1} << (plan->source_level_limit + 1u)) - 1u;
}

__global__ void count_canonical_merge_work_kernel(
    const std::uint32_t *epoch_counts, const Descriptor *descriptors,
    const DeviceManifest *manifests,
    const std::uint32_t *active_manifest,
    const ResidentPublicationPlan *plan, std::uint64_t *counts) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kRegions || plan->status) return;
  const DeviceManifestSnapshot manifest = load_active_manifest(
      manifests, active_manifest);
  std::uint64_t total = epoch_counts[q];
  std::uint64_t levels = manifest.occupied_level_mask &
      canonical_source_level_mask(plan);
  while (levels) {
    const std::uint32_t level =
        static_cast<std::uint32_t>(__ffsll(levels) - 1);
    levels &= levels - 1u;
    total += descriptors[descriptor_index(q, level)].count();
  }
  counts[q] = total;
}

__device__ __forceinline__ std::uint32_t canonical_tile_job_count(
    const std::uint64_t *counts, const std::uint32_t *capacities,
    std::uint32_t source_count) {
  const std::uint32_t single_capacity = capacities[1u];
  const std::uint32_t safe_capacity = single_capacity > source_count
      ? single_capacity - (source_count - 1u) : 1u;
  std::uint32_t jobs = 0u;
  std::uint64_t run_rows = 0u;
  std::uint32_t run_begin = 0u;
  for (std::uint32_t local_q = 0u;
       local_q < kPlanningTileRegions; ++local_q) {
    const std::uint64_t count = counts[local_q];
    if (count > single_capacity) {
      if (run_rows) {
        ++jobs;
        run_rows = 0u;
      }
      jobs += static_cast<std::uint32_t>(
          (count + safe_capacity - 1u) / safe_capacity);
      continue;
    }
    if (!count) continue;
    if (!run_rows) run_begin = local_q;
    const std::uint32_t proposed_span = local_q - run_begin + 1u;
    if (run_rows &&
        (proposed_span > kCanonicalJobRegions ||
         run_rows + count > capacities[proposed_span])) {
      ++jobs;
      run_rows = 0u;
      run_begin = local_q;
    }
    run_rows += count;
  }
  return jobs + (run_rows != 0u);
}

__global__ void count_canonical_planning_jobs_kernel(
    const std::uint64_t *counts, const ResidentPublicationPlan *plan,
    std::uint32_t *tile_job_counts) {
  __shared__ std::uint64_t tile_counts[kPlanningTileRegions];
  __shared__ std::uint32_t capacities[kCanonicalJobRegions + 1u];
  const std::uint32_t tile = blockIdx.x;
  if (tile >= kPlanningTiles) return;
  const std::uint32_t first = tile * kPlanningTileRegions;
  tile_counts[threadIdx.x] = counts[first + threadIdx.x];
  tile_counts[threadIdx.x + blockDim.x] =
      counts[first + threadIdx.x + blockDim.x];
  if (threadIdx.x <= kCanonicalJobRegions)
    capacities[threadIdx.x] = threadIdx.x
        ? canonical_job_capacity(plan, threadIdx.x) : 0u;
  __syncthreads();
  if (threadIdx.x == 0u) {
    tile_job_counts[tile] = plan->status ? 0u :
        canonical_tile_job_count(tile_counts, capacities,
                                 plan->source_count);
    if (tile + 1u == kPlanningTiles)
      tile_job_counts[kPlanningTiles] = 0u;
  }
}

__global__ void emit_canonical_planning_jobs_kernel(
    std::uint64_t *counts, const std::uint32_t *tile_job_offsets,
    ResidentPublicationPlan *plan, std::uint32_t maximum_jobs,
    BalancedMergeJob *jobs, std::uint64_t *reservations) {
  __shared__ std::uint64_t tile_counts[kPlanningTileRegions];
  __shared__ std::uint32_t capacities[kCanonicalJobRegions + 1u];
  const std::uint32_t tile = blockIdx.x;
  if (tile >= kPlanningTiles) return;
  const std::uint32_t first = tile * kPlanningTileRegions;
  tile_counts[threadIdx.x] = counts[first + threadIdx.x];
  tile_counts[threadIdx.x + blockDim.x] =
      counts[first + threadIdx.x + blockDim.x];
  if (threadIdx.x <= kCanonicalJobRegions)
    capacities[threadIdx.x] = threadIdx.x
        ? canonical_job_capacity(plan, threadIdx.x) : 0u;
  __syncthreads();
  if (threadIdx.x != 0u || plan->status) return;
  const std::uint32_t total_jobs = tile_job_offsets[kPlanningTiles];
  if (tile == 0u) {
    plan->job_count = total_jobs;
    if (total_jobs > maximum_jobs)
      atomicOr(&plan->status, kPublicationJobOverflow);
  }
  if (total_jobs > maximum_jobs) return;
  const std::uint32_t single_capacity = capacities[1u];
  const std::uint32_t safe = single_capacity > plan->source_count
      ? single_capacity - (plan->source_count - 1u) : 1u;
  std::uint32_t global = tile_job_offsets[tile];
  std::uint64_t run_rows = 0u;
  std::uint32_t run_begin = first;
  std::uint32_t run_end = first;
  const auto flush = [&]() {
    if (!run_rows) return;
    emit_resident_job(
        jobs, reservations, global++,
        std::uint64_t{run_begin} << 16u,
        std::uint64_t{run_end} << 16u,
        run_begin, run_end, run_rows);
    run_rows = 0u;
  };
  // Reuse counts as the oversized-job directory.
  for (std::uint32_t local_q = 0u;
       local_q < kPlanningTileRegions; ++local_q) {
    const std::uint32_t q = first + local_q;
    const std::uint64_t count = tile_counts[local_q];
    counts[q] = 0u;
    if (count > single_capacity) {
      flush();
      const std::uint32_t pieces = static_cast<std::uint32_t>(
          (count + safe - 1u) / safe);
      const std::uint32_t first_job = global;
      for (std::uint32_t piece = 0u; piece < pieces; ++piece)
        emit_resident_job(
            jobs, reservations, global++,
            std::uint64_t{q} << 16u,
            std::uint64_t{q + 1u} << 16u,
            q, q + 1u, 0u);
      counts[q] = canonical_hot_job(first_job, pieces);
      continue;
    }
    if (!count) continue;
    if (!run_rows) {
      run_begin = q;
      run_end = q + 1u;
    }
    const std::uint32_t proposed_span = q - run_begin + 1u;
    if (run_rows &&
        (proposed_span > kCanonicalJobRegions ||
         run_rows + count > capacities[proposed_span])) {
      flush();
      run_begin = q;
    }
    run_rows += count;
    run_end = q + 1u;
  }
  flush();
}

__device__ __forceinline__ std::uint32_t canonical_combined_prefix_warp(
    std::uint32_t q, std::uint32_t suffix,
    const Row *epoch_rows, const std::uint32_t *epoch_offsets,
    const std::uint32_t *epoch_counts, ResidentRows arena,
    const Descriptor *descriptors,
    const DeviceManifestSnapshot &manifest,
    const ResidentPublicationPlan *plan) {
  constexpr unsigned mask = 0xffffffffu;
  const std::uint32_t lane = threadIdx.x & 31u;
  std::uint32_t total = 0u;
  if (lane == 0u) {
    const std::uint32_t begin = epoch_offsets[q];
    total = lower_bound_rows(
        epoch_rows + begin, epoch_counts[q], suffix);
  }
  const std::uint32_t source_end = plan->source_count > 1u
      ? min(kMaximumLevels, plan->source_level_limit + 1u) : 0u;
  for (std::uint32_t level = lane;
       level < source_end; level += 32u) {
    if (!level_is_occupied(manifest.occupied_level_mask, level)) continue;
    const Descriptor rows = descriptors[descriptor_index(q, level)];
    total += lower_bound_rows(
        arena + rows.offset(), rows.count(), suffix);
  }
  for (std::uint32_t offset = 16u; offset; offset >>= 1u)
    total += __shfl_down_sync(mask, total, offset);
  return __shfl_sync(mask, total, 0u);
}

__device__ __forceinline__ std::uint32_t canonical_cell_prefix_warp(
    std::uint32_t q, std::uint32_t cell,
    const std::uint32_t *epoch_counts,
    const std::uint16_t *epoch_cell_ranks,
    const Descriptor *descriptors, const std::uint16_t *cell_ranks,
    const DeviceManifestSnapshot &manifest,
    const ResidentPublicationPlan *plan) {
  constexpr unsigned mask = 0xffffffffu;
  const std::uint32_t lane = threadIdx.x & 31u;
  std::uint32_t total = 0u;
  if (lane == 0u) {
    const std::uint32_t count = epoch_counts[q];
    total = !count ? 0u : cell < kCellsPerRegion
        ? epoch_cell_ranks[
              std::size_t{q} * kCellsPerRegion + cell]
        : count;
  }
  const std::uint32_t source_end = plan->source_count > 1u
      ? min(kMaximumLevels, plan->source_level_limit + 1u) : 0u;
  for (std::uint32_t level = lane;
       level < source_end; level += 32u) {
    if (!level_is_occupied(manifest.occupied_level_mask, level)) continue;
    const Descriptor rows = descriptors[descriptor_index(q, level)];
    if (!rows.count()) continue;
    total += cell < kCellsPerRegion
        ? cell_ranks[
              std::size_t{level} * kLocalRankEntries +
              std::size_t{q} * kCellsPerRegion + cell]
        : rows.count();
  }
  for (std::uint32_t offset = 16u; offset; offset >>= 1u)
    total += __shfl_down_sync(mask, total, offset);
  return __shfl_sync(mask, total, 0u);
}

__device__ __forceinline__ std::uint32_t
canonical_combined_cell_prefix_warp(
    std::uint32_t q, std::uint32_t cell, std::uint32_t suffix,
    const Row *epoch_rows, const std::uint32_t *epoch_offsets,
    const std::uint32_t *epoch_counts,
    const std::uint16_t *epoch_cell_ranks, ResidentRows arena,
    const Descriptor *descriptors, const std::uint16_t *cell_ranks,
    const DeviceManifestSnapshot &manifest,
    const ResidentPublicationPlan *plan) {
  constexpr unsigned mask = 0xffffffffu;
  const std::uint32_t lane = threadIdx.x & 31u;
  std::uint32_t total = 0u;
  if (lane == 0u) {
    const std::uint32_t count = epoch_counts[q];
    if (count) {
      const std::uint32_t region_begin = epoch_offsets[q];
      const std::uint16_t *ranks = epoch_cell_ranks +
          std::size_t{q} * kCellsPerRegion;
      const std::uint32_t begin = ranks[cell];
      const std::uint32_t end = cell + 1u < kCellsPerRegion
          ? ranks[cell + 1u] : count;
      total = begin + lower_bound_rows(
          epoch_rows + region_begin + begin, end - begin, suffix);
    }
  }
  const std::uint32_t source_end = plan->source_count > 1u
      ? min(kMaximumLevels, plan->source_level_limit + 1u) : 0u;
  for (std::uint32_t level = lane;
       level < source_end; level += 32u) {
    if (!level_is_occupied(manifest.occupied_level_mask, level)) continue;
    const Descriptor rows = descriptors[descriptor_index(q, level)];
    if (!rows.count()) continue;
    const std::uint16_t *ranks = cell_ranks +
        std::size_t{level} * kLocalRankEntries +
        std::size_t{q} * kCellsPerRegion;
    const std::uint32_t begin = ranks[cell];
    const std::uint32_t end = cell + 1u < kCellsPerRegion
        ? ranks[cell + 1u] : rows.count();
    total += begin + lower_bound_rows(
        arena + rows.offset() + begin, end - begin, suffix);
  }
  for (std::uint32_t offset = 16u; offset; offset >>= 1u)
    total += __shfl_down_sync(mask, total, offset);
  return __shfl_sync(mask, total, 0u);
}

struct CanonicalBoundary {
  std::uint32_t suffix{};
  std::uint32_t prefix{};
};

__device__ __forceinline__ CanonicalBoundary canonical_boundary_warp(
    std::uint32_t q, std::uint32_t target,
    const Row *epoch_rows, const std::uint32_t *epoch_offsets,
    const std::uint32_t *epoch_counts,
    const std::uint16_t *epoch_cell_ranks, ResidentRows arena,
    const Descriptor *descriptors, const std::uint16_t *cell_ranks,
    const DeviceManifestSnapshot &manifest,
    const ResidentPublicationPlan *plan) {
  if (!target) return {};
  std::uint32_t low_cell = 1u, high_cell = kCellsPerRegion;
  while (low_cell < high_cell) {
    const std::uint32_t middle = (low_cell + high_cell) >> 1u;
    if (canonical_cell_prefix_warp(
            q, middle, epoch_counts, epoch_cell_ranks, descriptors,
            cell_ranks, manifest, plan) < target)
      low_cell = middle + 1u;
    else
      high_cell = middle;
  }
  const std::uint32_t end_cell = low_cell;
  const std::uint32_t begin_cell = end_cell - 1u;
  const std::uint32_t begin_prefix = canonical_cell_prefix_warp(
      q, begin_cell, epoch_counts, epoch_cell_ranks, descriptors,
      cell_ranks, manifest, plan);
  if (begin_prefix >= target)
    return {begin_cell * kKeysPerCell, begin_prefix};
  std::uint32_t low = begin_cell * kKeysPerCell + 1u;
  std::uint32_t high = end_cell * kKeysPerCell;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (canonical_combined_cell_prefix_warp(
            q, begin_cell, middle, epoch_rows, epoch_offsets,
            epoch_counts, epoch_cell_ranks, arena, descriptors,
            cell_ranks,
            manifest, plan) < target)
      low = middle + 1u;
    else
      high = middle;
  }
  const std::uint32_t prefix = canonical_combined_cell_prefix_warp(
      q, begin_cell, low, epoch_rows, epoch_offsets, epoch_counts,
      epoch_cell_ranks, arena, descriptors, cell_ranks, manifest, plan);
  return {low, prefix};
}

__global__ void resolve_canonical_job_boundaries_kernel(
    const std::uint64_t *hot_jobs, BalancedMergeJob *jobs,
    std::uint64_t *reservations,
    ResidentPublicationPlan *plan, const Row *epoch_rows,
    const std::uint32_t *epoch_offsets,
    const std::uint32_t *epoch_counts,
    const std::uint16_t *epoch_cell_ranks, ResidentRows arena,
    const Descriptor *descriptors, const std::uint16_t *cell_ranks,
    const DeviceManifest *manifests,
    const std::uint32_t *active_manifest) {
  const std::uint32_t lane = threadIdx.x & 31u;
  const DeviceManifestSnapshot manifest = load_active_manifest(
      manifests, active_manifest);
  // One warp partitions each oversized region.
  for (std::uint32_t q = blockIdx.x;
       q < kRegions && !plan->status; q += gridDim.x) {
    const std::uint64_t encoded = hot_jobs[q];
    if (!canonical_job_is_hot(encoded)) continue;
    const std::uint32_t first_job = canonical_hot_first_job(encoded);
    const std::uint32_t pieces = canonical_hot_pieces(encoded);
    if (!pieces || first_job + pieces > plan->job_count) {
      if (lane == 0u)
        atomicOr(&plan->status, kPublicationJobTooLarge);
      continue;
    }
    const std::uint32_t total = canonical_combined_prefix_warp(
        q, 1u << 16u, epoch_rows, epoch_offsets, epoch_counts,
        arena, descriptors,
        manifest, plan);
    std::uint32_t previous_suffix = 0u;
    std::uint32_t previous_prefix = 0u;
    for (std::uint32_t piece = 0u; piece < pieces; ++piece) {
      CanonicalBoundary high{1u << 16u, total};
      if (piece + 1u < pieces) {
        const std::uint32_t target = static_cast<std::uint32_t>(
            (std::uint64_t{total} * (piece + 1u) + pieces - 1u) /
            pieces);
        high = canonical_boundary_warp(
            q, target, epoch_rows, epoch_offsets, epoch_counts,
            epoch_cell_ranks, arena, descriptors, cell_ranks,
            manifest, plan);
      }
      if (lane == 0u) {
        const std::uint32_t index = first_job + piece;
        BalancedMergeJob job = jobs[index];
        const std::uint32_t exact = high.prefix - previous_prefix;
        job.key_begin = (std::uint64_t{q} << 16u) + previous_suffix;
        job.key_end = (std::uint64_t{q} << 16u) + high.suffix;
        reservations[index] = exact;
        jobs[index] = job;
        if (previous_suffix >= high.suffix ||
            high.prefix < previous_prefix ||
            exact > canonical_job_capacity(plan, 1u))
          atomicOr(&plan->status, kPublicationJobTooLarge);
      }
      previous_suffix = high.suffix;
      previous_prefix = high.prefix;
    }
  }
}

__global__ void validate_canonical_plan_kernel(
    ResidentPublicationPlan *plan,
    const std::uint32_t *tile_job_offsets,
    std::uint32_t maximum_jobs) {
  if (blockIdx.x || threadIdx.x) return;
  plan->job_count = tile_job_offsets[kPlanningTiles];
  plan->raw_reservation = 0u;
  if (plan->job_count > maximum_jobs)
    plan->status |= kPublicationJobOverflow;
}

// Jobs publish counts and claim output prefixes.
__device__ __forceinline__ unsigned long long canonical_job_prefix(
    std::uint32_t job_index, std::uint32_t count,
    CanonicalJobPrefix *prefixes) {
  CanonicalJobPrefix &state = prefixes[job_index];
  state.count = count;
  __threadfence();
  atomicExch(&state.ready, 1u);

  unsigned long long prefix = 0ull;
  std::uint32_t cursor = job_index;
  while (cursor) {
    CanonicalJobPrefix &previous = prefixes[cursor - 1u];
    const std::uint32_t ready = atomicAdd(&previous.ready, 0u);
    if (!ready) {
      __nanosleep(64u);
      continue;
    }
    prefix += previous.count;
    if (ready >= 2u) {
      prefix += previous.prefix;
      break;
    }
    --cursor;
  }
  state.prefix = prefix;
  __threadfence();
  atomicExch(&state.ready, 2u);
  return prefix;
}

struct CanonicalTournamentSlice {
  std::uint32_t begin{};
  std::uint32_t count{};
};

// Cache each source region's physical base.
__device__ __forceinline__ CanonicalTournamentSlice
canonical_tournament_cell_slice(
    std::uint32_t source, std::uint32_t level, std::uint32_t q,
    std::uint32_t cell, std::uint32_t suffix_begin,
    std::uint32_t suffix_end, std::uint64_t source_base,
    std::uint32_t region_count, const Row *epoch_rows,
    const std::uint16_t *epoch_cell_ranks, ResidentRows arena,
    const std::uint16_t *cell_ranks) {
  if (!region_count) return {};
  const std::uint32_t cell_suffix_begin = cell * kKeysPerCell;
  const std::uint32_t cell_suffix_end =
      cell_suffix_begin + kKeysPerCell;
  const std::uint16_t *ranks = source == 0u
      ? epoch_cell_ranks + std::size_t{q} * kCellsPerRegion
      : cell_ranks + std::size_t{level} * kLocalRankEntries +
            std::size_t{q} * kCellsPerRegion;
  std::uint32_t begin = ranks[cell];
  std::uint32_t end = cell + 1u < kCellsPerRegion
      ? ranks[cell + 1u] : region_count;
  if (source == 0u) {
    const Row *rows = epoch_rows + source_base;
    if (suffix_begin > cell_suffix_begin)
      begin += lower_bound_rows(
          rows + begin, end - begin, suffix_begin);
    if (suffix_end < cell_suffix_end)
      end = begin + lower_bound_rows(
          rows + begin, end - begin, suffix_end);
  } else {
    const ResidentRows rows = arena + source_base;
    if (suffix_begin > cell_suffix_begin)
      begin += lower_bound_rows(
          rows + begin, end - begin, suffix_begin);
    if (suffix_end < cell_suffix_end)
      end = begin + lower_bound_rows(
          rows + begin, end - begin, suffix_end);
  }
  return {begin, end - begin};
}

__device__ __forceinline__ std::uint32_t
canonical_tournament_slice_begin(
    std::uint32_t source, std::uint32_t level, std::uint32_t q,
    std::uint32_t cell, const BalancedMergeJob &job,
    std::uint64_t source_base, std::uint32_t region_count,
    const Row *epoch_rows, const std::uint16_t *epoch_cell_ranks,
    ResidentRows arena, const std::uint16_t *cell_ranks) {
  if (!region_count) return 0u;
  const std::uint64_t region_key = std::uint64_t{q} << 16u;
  const std::uint64_t cell_key_begin =
      region_key + std::uint64_t{cell} * kKeysPerCell;
  const std::uint64_t cell_key_end = cell_key_begin + kKeysPerCell;
  if (job.key_begin <= cell_key_begin && job.key_end >= cell_key_end) {
    if (source == 0u)
      return epoch_cell_ranks[
          std::size_t{q} * kCellsPerRegion + cell];
    return cell_ranks[
        std::size_t{level} * kLocalRankEntries +
        std::size_t{q} * kCellsPerRegion + cell];
  }
  const std::uint64_t clipped_begin = max(cell_key_begin, job.key_begin);
  const std::uint64_t clipped_end = min(cell_key_end, job.key_end);
  return canonical_tournament_cell_slice(
      source, level, q, cell,
      static_cast<std::uint32_t>(clipped_begin - region_key),
      static_cast<std::uint32_t>(clipped_end - region_key),
      source_base, region_count, epoch_rows, epoch_cell_ranks,
      arena, cell_ranks).begin;
}

__device__ __forceinline__ Row canonical_tournament_source_row(
    std::uint32_t source, std::uint64_t source_base,
    std::uint32_t region_position, const Row *epoch_rows,
    ResidentRows arena) {
  if (source == 0u)
    return epoch_rows[source_base + region_position];
  return arena[source_base + region_position];
}

__device__ __forceinline__ std::uint32_t
canonical_tournament_source_head(
    std::uint32_t source, std::uint64_t source_base,
    std::uint32_t region_position, const Row *epoch_rows,
    ResidentRows arena, std::uint32_t local_position) {
  std::uint32_t packed = 0u;
  if (source == 0u) {
    const Row *row = epoch_rows + source_base + region_position;
    packed = std::uint32_t{row->key} |
        (std::uint32_t{row->flags} << 16u);
  } else {
    packed = arena.key_flags[source_base + region_position];
  }
  return (packed & 0xffffu) |
      ((packed & (std::uint32_t{kTombstone} << 16u))
           ? (1u << 17u) : 0u) |
      (local_position << 18u);
}

__device__ __forceinline__ std::uint8_t canonical_tournament_choose(
    std::uint8_t left, std::uint8_t right,
    const std::uint32_t *heads, std::uint32_t chain) {
  if (left == 0xffu) return right;
  if (right == 0xffu) return left;
  const std::uint32_t left_key =
      heads[std::size_t{left} * kCanonicalTournamentChains + chain] &
      0x1ffffu;
  const std::uint32_t right_key =
      heads[std::size_t{right} * kCanonicalTournamentChains + chain] &
      0x1ffffu;
  return right_key < left_key ||
          (right_key == left_key && right < left)
      ? right : left;
}

__device__ __forceinline__ std::uint8_t canonical_tournament_child(
    std::uint32_t node, std::uint32_t leaves, std::uint32_t source_count,
    const std::uint8_t *tree, std::uint32_t chain) {
  if (node < leaves)
    return tree[std::size_t{node} * kCanonicalTournamentChains + chain];
  const std::uint32_t source = node - leaves;
  return source < source_count
      ? static_cast<std::uint8_t>(source) : 0xffu;
}

// Each chain merges one sorted key cell.
__global__ __launch_bounds__(kMergeThreads)
void canonical_tournament_carry_jobs_kernel(
    BalancedMergeJob *jobs, const std::uint64_t *reservations,
    ResidentPublicationPlan *plan, const Row *epoch_rows,
    const std::uint32_t *epoch_offsets,
    const std::uint32_t *epoch_counts,
    const std::uint16_t *epoch_cell_ranks, ResidentRows arena,
    const Descriptor *descriptors,
    const std::uint16_t *cell_ranks, const DeviceManifest *manifests,
    const std::uint32_t *active_manifest,
    CanonicalJobPrefix *prefixes, std::uint32_t *next_job,
    std::uint32_t *cell_counts) {
  constexpr std::uint32_t kThreads = kMergeThreads;
  constexpr std::uint32_t kScanItems =
      kCanonicalTournamentTasks / kThreads;
  static_assert(kScanItems * kThreads == kCanonicalTournamentTasks);
  using BlockScan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::uint16_t source_levels[kMaximumMergeSources];
  __shared__ std::uint32_t source_count_shared;
  __shared__ std::uint32_t leaves_shared;
  __shared__ std::uint32_t job_index_shared;
  __shared__ std::uint32_t task_count_shared;
  __shared__ std::uint32_t job_capacity_shared;
  __shared__ std::uint32_t next_task_shared;
  __shared__ std::uint32_t tape_cursor_shared;
  __shared__ std::uint32_t job_valid_shared;
  __shared__ unsigned long long output_prefix_shared;
  extern __shared__ __align__(16) unsigned char workspace[];

  const DeviceManifestSnapshot manifest = load_active_manifest(
      manifests, active_manifest);
  // Build the source list once per block.
  if (threadIdx.x == 0u) {
    std::uint32_t source_count = 0u;
    source_levels[source_count++] = kMaximumLevels;
    std::uint64_t levels = manifest.occupied_level_mask &
        canonical_source_level_mask(plan);
    while (levels && source_count < kMaximumMergeSources) {
      const std::uint32_t level =
          static_cast<std::uint32_t>(__ffsll(levels) - 1);
      levels &= levels - 1u;
      source_levels[source_count++] = static_cast<std::uint16_t>(level);
    }
    source_count_shared = source_count;
    leaves_shared = canonical_next_power_of_two(source_count);
  }
  __syncthreads();
  for (;;) {
    if (threadIdx.x == 0u)
      job_index_shared = atomicAdd(next_job, 1u);
    __syncthreads();
    const std::uint32_t job_index = job_index_shared;
    if (job_index >= plan->job_count) return;
    const BalancedMergeJob job = jobs[job_index];
    const std::uint32_t region_count =
        job.region_end - job.region_begin;

    if (threadIdx.x == 0u) {
      task_count_shared = region_count * kCellsPerRegion;
      job_capacity_shared = canonical_job_capacity(plan, region_count);
      next_task_shared = 0u;
      tape_cursor_shared = 0u;
      job_valid_shared = source_count_shared == plan->source_count &&
          source_count_shared <= kMaximumMergeSources && region_count &&
          region_count <= kCanonicalJobRegions && task_count_shared &&
          task_count_shared <= kCanonicalTournamentTasks &&
          job_capacity_shared &&
          reservations[job_index] <= job_capacity_shared;
    }
    __syncthreads();

    const std::uint32_t source_count = source_count_shared;
    const std::uint32_t leaves = leaves_shared;
    std::size_t offset = 0u;
    const std::size_t states =
        std::size_t{kCanonicalTournamentChains} * source_count;
    offset = canonical_align_bytes(offset, alignof(std::uint32_t));
    std::uint32_t *cursors =
        reinterpret_cast<std::uint32_t *>(workspace + offset);
    offset += states * sizeof(std::uint32_t);
    offset = canonical_align_bytes(offset, alignof(std::uint32_t));
    std::uint32_t *heads =
        reinterpret_cast<std::uint32_t *>(workspace + offset);
    offset += states * sizeof(std::uint32_t);
    const std::size_t source_regions =
        std::size_t{region_count} * source_count;
    offset = canonical_align_bytes(offset, alignof(std::uint64_t));
    std::uint64_t *source_bases =
        reinterpret_cast<std::uint64_t *>(workspace + offset);
    offset += source_regions * sizeof(std::uint64_t);
    offset = canonical_align_bytes(offset, alignof(std::uint32_t));
    std::uint32_t *source_region_counts =
        reinterpret_cast<std::uint32_t *>(workspace + offset);
    offset += source_regions * sizeof(std::uint32_t);
    std::uint8_t *tree = workspace + offset;
    offset += std::size_t{kCanonicalTournamentChains} * leaves;
    offset = canonical_align_bytes(
        offset, alignof(CanonicalTournamentReference));
    CanonicalTournamentReference *survivor_tape =
        reinterpret_cast<CanonicalTournamentReference *>(workspace + offset);
    offset += std::size_t{job_capacity_shared} *
        sizeof(CanonicalTournamentReference);
    offset = canonical_align_bytes(offset, alignof(std::uint16_t));
    std::uint16_t *task_tape_bases =
        reinterpret_cast<std::uint16_t *>(workspace + offset);
    offset += std::size_t{task_count_shared} *
        sizeof(std::uint16_t);
    offset = canonical_align_bytes(offset, alignof(std::uint16_t));
    std::uint16_t *task_output_offsets =
        reinterpret_cast<std::uint16_t *>(workspace + offset);
    offset += std::size_t{task_count_shared + 1u} *
        sizeof(std::uint16_t);
    const std::size_t workspace_bytes = canonical_align_bytes(offset, 16u);
    if (threadIdx.x == 0u)
      job_valid_shared &= workspace_bytes <=
          plan->tournament_workspace_bytes;
    __syncthreads();

    if (!job_valid_shared) {
      if (threadIdx.x == 0u) {
        atomicOr(&plan->status, kPublicationJobTooLarge);
        canonical_job_prefix(job_index, 0u, prefixes);
      }
      __syncthreads();
      continue;
    }

    const std::uint32_t active_source_regions =
        region_count * source_count;
    for (std::uint32_t entry = threadIdx.x;
         entry < active_source_regions; entry += blockDim.x) {
      const std::uint32_t local_q = entry / source_count;
      const std::uint32_t source = entry - local_q * source_count;
      const std::uint32_t q = job.region_begin + local_q;
      if (source == 0u) {
        source_bases[entry] = epoch_offsets[q];
        source_region_counts[entry] = epoch_counts[q];
      } else {
        const Descriptor descriptor = descriptors[
            descriptor_index(q, source_levels[source])];
        source_bases[entry] = descriptor.offset();
        source_region_counts[entry] = descriptor.count();
      }
    }
    __syncthreads();

    if (threadIdx.x < kCanonicalTournamentChains) {
      const std::uint32_t chain = threadIdx.x;
      for (;;) {
        const std::uint32_t task = atomicAdd(&next_task_shared, 1u);
        if (task >= task_count_shared) break;
        const std::uint32_t q =
            job.region_begin + task / kCellsPerRegion;
        const std::uint32_t cell = task % kCellsPerRegion;
        const std::uint64_t cell_key_begin =
            (std::uint64_t{q} << 16u) +
            std::uint64_t{cell} * kKeysPerCell;
        const std::uint64_t cell_key_end =
            cell_key_begin + kKeysPerCell;
        const std::uint64_t clipped_begin =
            max(cell_key_begin, job.key_begin);
        const std::uint64_t clipped_end = min(cell_key_end, job.key_end);
        const std::uint32_t suffix_begin = static_cast<std::uint32_t>(
            clipped_begin - (std::uint64_t{q} << 16u));
        const std::uint32_t suffix_end = static_cast<std::uint32_t>(
            clipped_end - (std::uint64_t{q} << 16u));

        std::uint32_t raw_count = 0u;
        for (std::uint32_t source = 0u;
             source < source_count; ++source) {
          const std::uint32_t level = source_levels[source];
          const std::uint32_t local_q = q - job.region_begin;
          const std::size_t entry =
              std::size_t{local_q} * source_count + source;
          const std::uint64_t source_base = source_bases[entry];
          const CanonicalTournamentSlice slice =
              canonical_tournament_cell_slice(
                  source, level, q, cell, suffix_begin, suffix_end,
                  source_base, source_region_counts[entry], epoch_rows,
                  epoch_cell_ranks, arena, cell_ranks);
          const std::size_t state =
              std::size_t{source} * kCanonicalTournamentChains + chain;
          cursors[state] = (slice.count << 16u) | slice.begin;
          raw_count += slice.count;
          if (slice.count) {
            heads[state] = canonical_tournament_source_head(
                source, source_base, slice.begin, epoch_rows, arena, 0u);
          } else {
            heads[state] = 1u << 16u;
          }
        }

        const std::uint32_t tape_begin =
            atomicAdd(&tape_cursor_shared, raw_count);
        task_tape_bases[task] = static_cast<std::uint16_t>(tape_begin);
        if (tape_begin + raw_count > job_capacity_shared) {
          atomicExch(&job_valid_shared, 0u);
          task_output_offsets[task] = 0u;
          continue;
        }

        if (leaves > 1u) {
          for (std::uint32_t node = leaves - 1u; node; --node) {
            const std::uint8_t left = canonical_tournament_child(
                node << 1u, leaves, source_count, tree, chain);
            const std::uint8_t right = canonical_tournament_child(
                (node << 1u) + 1u, leaves, source_count, tree, chain);
            tree[std::size_t{node} * kCanonicalTournamentChains + chain] =
                canonical_tournament_choose(left, right, heads, chain);
          }
        }

        std::uint32_t survivor_count = 0u;
        std::uint32_t previous_key = 1u << 16u;
        for (;;) {
          const std::uint32_t source = leaves == 1u ? 0u :
              tree[kCanonicalTournamentChains + chain];
          const std::size_t state =
              std::size_t{source} * kCanonicalTournamentChains + chain;
          const std::uint32_t head = heads[state];
          const std::uint32_t key = head & 0x1ffffu;
          if (key == (1u << 16u)) break;
          const std::uint32_t cursor = cursors[state];
          const std::uint32_t position = cursor & 0xffffu;
          const std::uint32_t count_before = cursor >> 16u;
          const std::uint32_t local_position =
              (head >> 18u) & kCanonicalTournamentReferenceMask;
          const std::uint32_t left = count_before - 1u;
          const std::uint32_t next_position = position + 1u;
          const std::uint32_t local_q = q - job.region_begin;
          const std::uint64_t source_base = source_bases[
              std::size_t{local_q} * source_count + source];

          // Prefetch before survivor bookkeeping.
          std::uint32_t advanced_head = 1u << 16u;
          if (left)
            advanced_head = canonical_tournament_source_head(
                source, source_base, next_position, epoch_rows, arena,
                local_position + 1u);

          if (key != previous_key) {
            if (plan->keep_tombstones || !(head & (1u << 17u))) {
              // Each source has one row per key.
              survivor_tape[tape_begin + survivor_count++] =
                  static_cast<std::uint16_t>(
                      (source << kCanonicalTournamentReferenceBits) |
                      local_position);
            }
            previous_key = key;
          }

          cursors[state] = (left << 16u) |
              (left ? next_position : position);
          heads[state] = advanced_head;

          if (leaves > 1u) {
            std::uint32_t node = (leaves + source) >> 1u;
            while (node) {
              const std::uint8_t left_source =
                  canonical_tournament_child(
                      node << 1u, leaves, source_count, tree, chain);
              const std::uint8_t right_source =
                  canonical_tournament_child(
                      (node << 1u) + 1u, leaves, source_count, tree, chain);
              tree[std::size_t{node} *
                       kCanonicalTournamentChains + chain] =
                  canonical_tournament_choose(
                      left_source, right_source, heads, chain);
              node >>= 1u;
            }
          }
        }
        task_output_offsets[task] =
            static_cast<std::uint16_t>(survivor_count);
        atomicAdd(cell_counts +
                      std::size_t{q} * kCellsPerRegion + cell,
                  survivor_count);
      }
    }
    __syncthreads();

    if (!job_valid_shared ||
        tape_cursor_shared != reservations[job_index]) {
      if (threadIdx.x == 0u) {
        atomicOr(&plan->status, kPublicationJobTooLarge);
        canonical_job_prefix(job_index, 0u, prefixes);
      }
      __syncthreads();
      continue;
    }

    std::uint32_t local_prefixes[kScanItems];
    std::uint32_t thread_total = 0u;
    for (std::uint32_t item = 0u; item < kScanItems; ++item) {
      const std::uint32_t task = threadIdx.x * kScanItems + item;
      const std::uint32_t count = task < task_count_shared
          ? task_output_offsets[task] : 0u;
      local_prefixes[item] = thread_total;
      thread_total += count;
    }
    std::uint32_t thread_prefix = 0u, job_output_count = 0u;
    BlockScan(scan_storage).ExclusiveSum(
        thread_total, thread_prefix, job_output_count);
    for (std::uint32_t item = 0u; item < kScanItems; ++item) {
      const std::uint32_t task = threadIdx.x * kScanItems + item;
      if (task < task_count_shared)
        task_output_offsets[task] = static_cast<std::uint16_t>(
            thread_prefix + local_prefixes[item]);
    }
    if (threadIdx.x == 0u) {
      task_output_offsets[task_count_shared] =
          static_cast<std::uint16_t>(job_output_count);
      const unsigned long long prefix = canonical_job_prefix(
          job_index, job_output_count, prefixes);
      output_prefix_shared = prefix;
      atomicAdd(reinterpret_cast<unsigned long long *>(
                    &plan->raw_reservation),
                static_cast<unsigned long long>(tape_cursor_shared));
      if (job_index + 1u == plan->job_count)
        plan->survivor_count = prefix + job_output_count;
      if (plan->source_level_limit == plan->destination_level &&
          prefix + job_output_count > plan->output_capacity) {
        job_valid_shared = 0u;
        atomicOr(&plan->status, kPublicationOutputOverflow);
      }
    }
    __syncthreads();

    if (!job_valid_shared) continue;

    // Materialize contiguous output by warp.
    constexpr unsigned kFullWarp = 0xffffffffu;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t warp = threadIdx.x >> 5u;
    std::uint32_t logical = warp * 32u + lane;
    std::uint32_t task = 0u;
    if (logical < job_output_count) {
      std::uint32_t low = 0u, high = task_count_shared;
      while (low < high) {
        const std::uint32_t middle = (low + high) >> 1u;
        if (task_output_offsets[middle + 1u] <= logical)
          low = middle + 1u;
        else
          high = middle;
      }
      task = low;
    }
    for (;;) {
      const unsigned active = __ballot_sync(
          kFullWarp, logical < job_output_count);
      if (!active) break;
      if (logical < job_output_count) {
        while (task + 1u < task_count_shared &&
               task_output_offsets[task + 1u] <= logical)
          ++task;
        const std::uint32_t reference = survivor_tape[
            task_tape_bases[task] + logical - task_output_offsets[task]];
        const std::uint32_t q =
            job.region_begin + task / kCellsPerRegion;
        const std::uint32_t cell = task % kCellsPerRegion;
        std::uint32_t source = 0u;
        std::uint32_t position = 0u;
        std::uint64_t source_base = 0u;
        std::uint32_t source_region_count = 0u;
        source = reference >> kCanonicalTournamentReferenceBits;
        const std::uint32_t local_q = q - job.region_begin;
        const std::size_t entry =
            std::size_t{local_q} * source_count + source;
        source_base = source_bases[entry];
        source_region_count = source_region_counts[entry];
        const unsigned peers = __match_any_sync(
            active, (task << kMergeSourceBits) | source);
        const std::uint32_t leader =
            static_cast<std::uint32_t>(__ffs(peers) - 1);
        std::uint32_t slice_begin = 0u;
        if (lane == leader)
          slice_begin = canonical_tournament_slice_begin(
              source, source_levels[source], q, cell, job,
              source_base, source_region_count, epoch_rows,
              epoch_cell_ranks, arena, cell_ranks);
        slice_begin = __shfl_sync(active, slice_begin, leader);
        position = slice_begin +
            (reference & kCanonicalTournamentReferenceMask);
        const Row row = canonical_tournament_source_row(
            source, source_base, position, epoch_rows, arena);
        arena.store(
            plan->output_begin + output_prefix_shared + logical, row);
      }
      logical += kThreads;
    }
    __syncthreads();
  }
}

__global__ void build_canonical_rank_from_counts_kernel(
    const std::uint32_t *cell_counts,
    const ResidentPublicationPlan *plan, std::uint32_t level_count,
    std::uint16_t *cell_ranks, std::uint32_t *region_counts) {
  using Scan = cub::BlockScan<std::uint32_t, kCellsPerRegion>;
  __shared__ typename Scan::TempStorage storage;
  const std::uint32_t q = blockIdx.x;
  const std::uint32_t cell = threadIdx.x;
  if (plan->status || plan->destination_level >= level_count) {
    if (threadIdx.x == 0u) region_counts[q] = 0u;
    if (q == 0u && threadIdx.x == 0u)
      region_counts[kRegions] = 0u;
    return;
  }
  const std::uint32_t level = plan->destination_level;
  const std::uint32_t count =
      cell_counts[std::size_t{q} * kCellsPerRegion + cell];
  std::uint32_t rank{}, total{};
  Scan(storage).ExclusiveSum(count, rank, total);
  cell_ranks[std::size_t{level} * kLocalRankEntries +
             std::size_t{q} * kCellsPerRegion + cell] =
      static_cast<std::uint16_t>(rank);
  if (threadIdx.x == 0u) region_counts[q] = total;
  if (q == 0u && threadIdx.x == 0u)
    region_counts[kRegions] = 0u;
}

inline std::uint32_t select_canonical_merge_capacity(
    std::uint32_t maximum_sources) {
  if (maximum_sources < kCanonicalTournamentMinimumSources ||
      maximum_sources > kMaximumMergeSources)
    throw std::invalid_argument("invalid tournament source count");
  int maximum_blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maximum_blocks, canonical_tournament_carry_jobs_kernel,
      kMergeThreads, 0u));
  const int desired_blocks = std::max(1, std::min(3, maximum_blocks));
  std::size_t shared_budget = 0u;
  CUDA_CHECK(cudaOccupancyAvailableDynamicSMemPerBlock(
      &shared_budget, canonical_tournament_carry_jobs_kernel,
      desired_blocks, kMergeThreads));
  std::uint32_t capacity = canonical_tournament_capacity(
      shared_budget, maximum_sources, kCanonicalJobRegions);
  while (capacity && canonical_tournament_workspace_bytes(
             capacity, maximum_sources) > shared_budget)
    --capacity;
  if (capacity <= kMaximumMergeSources)
    throw std::runtime_error(
        "insufficient shared memory for GPULSMOpt tournament");
  return capacity;
}

}  // namespace celllsm
