#pragma once

#include "gpu_dictionary_adapter.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t e__ = (call);                                             \
    if (e__ != cudaSuccess)                                                     \
      throw std::runtime_error(std::string(cudaGetErrorString(e__)) +           \
                               " at " + __FILE__ + ":" +                     \
                               std::to_string(__LINE__));                       \
  } while (false)
#endif

namespace gpulsmopt2_detail {

constexpr std::uint32_t kQuotients = 1u << 16u;
constexpr std::uint32_t kMaximumLevels = 64u;
constexpr std::uint32_t kBatchesPerEpoch = 16u;
constexpr std::uint32_t kBatchPositionBits = 20u;
constexpr std::uint32_t kLocalRankBits = 7u;
constexpr std::uint32_t kLocalRankEntries =
    kQuotients * (1u << kLocalRankBits);
constexpr std::uint32_t kThreads = 256u;
constexpr std::uint32_t kBulkRootSinkItemsPerThread = 8u;
constexpr std::uint32_t kBulkRootSinkTileRows =
    kThreads * kBulkRootSinkItemsPerThread;
constexpr std::uint32_t kTqrjHashThreads = 1024u;
static_assert(kTqrjHashThreads <= 1024u &&
              (kTqrjHashThreads & 31u) == 0u);
constexpr std::uint32_t kRangeSchedulerBlocks = 256u;
constexpr std::uint32_t kSectionRangeThreads = 128u;
constexpr std::uint32_t kInvalid = 0xffffffffu;
constexpr std::uint32_t kInvalidAge = 0xffffffffu;
constexpr std::uint32_t kTombstone = 1u;
constexpr std::size_t kMaximumOperationTile = std::size_t{1} << 20u;
static_assert(kMaximumOperationTile <=
              (std::size_t{1} << kBatchPositionBits));
constexpr std::size_t kMaximumPublicationRows =
    std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kDescriptorOffsetBits = 47u;
constexpr std::uint64_t kDescriptorOffsetMask =
    (std::uint64_t{1} << kDescriptorOffsetBits) - 1u;
static_assert(kMaximumPublicationRows <= kDescriptorOffsetMask);
constexpr std::uint32_t kSectionOwnerMinimumReuse = 4u;
constexpr std::uint32_t kRangeThreadWork = 8u;
constexpr std::uint32_t kRangeSubgroupWork = 512u;
constexpr std::uint32_t kRangeOnChipNewerRows = 128u;
constexpr std::uint32_t kRangeHotWindowRows = 1u << 26u;
constexpr std::uint32_t kAdmissionCtaGroupMaximum = 64u;
constexpr std::uint32_t kAdmissionCtaHashSlots = 128u;
static_assert((kAdmissionCtaHashSlots &
               (kAdmissionCtaHashSlots - 1u)) == 0u);
// TQRJ groups queries by the same sections already used by pending runs.
// Ordinary sections use a compact exact directory in shared memory; overflow
// sections share one phase-recycled exact-key hash table.  Both executors scan
// each owned pending-run section once.
constexpr std::uint32_t kTqrjDenseRowsPerSection = 8u;
constexpr std::uint32_t kTqrjMinimumBatches = 5u;
constexpr std::uint32_t kTqrjDirectCapacity = 1280u;
constexpr std::uint32_t kTqrjDirectoryBins = 256u;
constexpr std::uint32_t kTqrjDirectIntervalLimit = 64u;
constexpr std::uint32_t kTqrjPairFindThreshold = 16u;
constexpr std::uint32_t kTqrjDirectPendingRows = 1u << 16u;
constexpr std::uint32_t kTqrjHashTileRows = 2048u;
// One four-byte overflow-table slot stores an exact-query owner plus a small
// placement fingerprint.  The fingerprint can only reject candidates: every
// accepted match is confirmed against the complete query key.
constexpr std::uint32_t kTqrjHashOwnerBits = 25u;
constexpr std::uint32_t kTqrjHashOwnerMask =
    (1u << kTqrjHashOwnerBits) - 1u;
constexpr std::uint32_t kTqrjHashFingerprintBits =
    32u - kTqrjHashOwnerBits;
constexpr std::uint32_t kTqrjHashAttempts = 3u;
constexpr std::uint32_t kTqrjHashProbeLimit = 64u;
constexpr std::uint32_t kTqrjHashCapacityAlignment = 256u;
static_assert(kTqrjDirectCapacity == 1280u);
static_assert(kTqrjDirectCapacity <=
              std::numeric_limits<std::uint16_t>::max());
static_assert(kTqrjDirectoryBins == kThreads);
static_assert(kTqrjPairFindThreshold <= kTqrjDirectIntervalLimit);
static_assert(kTqrjDirectPendingRows > kTqrjDirectCapacity);
static_assert(kTqrjHashTileRows >= kTqrjHashThreads &&
              (kTqrjHashTileRows & (kTqrjHashTileRows - 1u)) == 0u);
static_assert(kMaximumOperationTile * kBatchesPerEpoch <=
              kTqrjHashOwnerMask);
constexpr std::uint32_t kFoundationCompactionThreads = 256u;
constexpr std::uint32_t kLocalEpochItemsPerThread = 5u;
constexpr std::uint32_t kLocalEpochCapacity =
    kFoundationCompactionThreads * kLocalEpochItemsPerThread;
constexpr std::uint32_t kFoundationCells = 128u;
constexpr std::uint32_t kFoundationCellKeys = 512u;
// A full section ends at the descriptor count.
__host__ __device__ constexpr bool cell_rank_supported(
    std::uint64_t count) {
  return count <= kQuotients;
}
static_assert((kFoundationCells - 1u) * kFoundationCellKeys <=
              std::numeric_limits<std::uint16_t>::max());
constexpr std::uint32_t kPlanningTiles = 128u;
constexpr std::uint32_t kPlanningTileQuotients =
    kQuotients / kPlanningTiles;
constexpr std::uint32_t kMaximumMergeSources = kMaximumLevels + 1u;
constexpr std::uint32_t kBalancedMergeCapacityCeiling =
    kFoundationCompactionThreads * 32u;
constexpr std::uint32_t kCanonicalTournamentMinimumSources =
    2u;
static_assert(kCanonicalTournamentMinimumSources >= 2u &&
              kCanonicalTournamentMinimumSources <= kMaximumMergeSources);
constexpr std::uint32_t kCanonicalJobQuotients = 16u;
constexpr std::uint32_t kCanonicalCandidateBits = 12u;
constexpr std::uint32_t kCanonicalCandidateLimit =
    1u << kCanonicalCandidateBits;
constexpr std::uint32_t kCanonicalTombstoneWords =
    kCanonicalCandidateLimit / 32u;
constexpr std::uint32_t kCanonicalCapacityAdjustment =
    (kCanonicalTombstoneWords * sizeof(std::uint32_t) +
     2u * sizeof(std::uint32_t) - 1u) /
    (2u * sizeof(std::uint32_t));
static_assert(kCanonicalJobQuotients == 1u << 4u);
static_assert(kBalancedMergeCapacityCeiling <= kCanonicalCandidateLimit * 2u);
constexpr std::uint32_t kCanonicalResolverSuffixes = 1u << 16u;
constexpr std::uint32_t kCanonicalTournamentChains = 128u;
constexpr std::uint32_t kCanonicalTournamentTasks =
    kCanonicalJobQuotients * kFoundationCells;
constexpr std::uint32_t kCanonicalTournamentReferenceBits = 9u;
constexpr std::uint32_t kCanonicalTournamentReferenceMask =
    (1u << kCanonicalTournamentReferenceBits) - 1u;
static_assert(kFoundationCellKeys ==
              (1u << kCanonicalTournamentReferenceBits));
using CanonicalTournamentReference = std::uint16_t;
// The actual capacity is selected per source count from the device's
// occupancy limits.  This is only the structural limit imposed by the
// 16-bit per-task offsets, not a workload-specific tuning constant.
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
    std::uint32_t quotient_count) {
  if (!source_count || !quotient_count ||
      quotient_count > kCanonicalJobQuotients)
    return ~std::size_t{0};
  const std::uint32_t leaves = canonical_next_power_of_two(source_count);
  std::size_t bytes = 0u;
  const std::size_t states =
      std::size_t{kCanonicalTournamentChains} * source_count;
  bytes = canonical_align_bytes(bytes, alignof(std::uint32_t));
  bytes += states * sizeof(std::uint32_t);  // packed cursors
  bytes = canonical_align_bytes(bytes, alignof(std::uint32_t));
  bytes += states * sizeof(std::uint32_t);  // source heads
  const std::size_t source_quotients =
      std::size_t{quotient_count} * source_count;
  bytes = canonical_align_bytes(bytes, alignof(std::uint64_t));
  bytes += source_quotients * sizeof(std::uint64_t);
  bytes = canonical_align_bytes(bytes, alignof(std::uint32_t));
  bytes += source_quotients * sizeof(std::uint32_t);
  bytes += std::size_t{kCanonicalTournamentChains} * leaves;
  bytes = canonical_align_bytes(
      bytes, alignof(CanonicalTournamentReference));
  bytes += std::size_t{capacity} * sizeof(CanonicalTournamentReference);
  const std::size_t tasks =
      std::size_t{quotient_count} * kFoundationCells;
  bytes = canonical_align_bytes(bytes, alignof(std::uint16_t));
  bytes += tasks * sizeof(std::uint16_t);
  bytes = canonical_align_bytes(bytes, alignof(std::uint16_t));
  bytes += (tasks + 1u) * sizeof(std::uint16_t);
  return bytes;
}

__host__ __device__ __forceinline__ std::size_t
canonical_tournament_layout_bytes(
    std::uint32_t capacity, std::uint32_t source_count,
    std::uint32_t quotient_count) {
  const std::size_t bytes = canonical_tournament_body_bytes(
      capacity, source_count, quotient_count);
  return bytes == ~std::size_t{0}
      ? bytes : canonical_align_bytes(bytes, 16u);
}

__host__ __device__ __forceinline__ std::uint32_t
canonical_tournament_capacity(
    std::size_t shared_bytes, std::uint32_t source_count,
    std::uint32_t quotient_count) {
  // All production tournament allocations are 16-byte aligned.  With the
  // tape placed before two-byte task arrays, its capacity is linear in the
  // remaining bytes; no search or workload-derived threshold is needed.
  const std::size_t usable_bytes = shared_bytes & ~std::size_t{15u};
  const std::size_t fixed_bytes =
      canonical_tournament_body_bytes(
          0u, source_count, quotient_count);
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
  // CUDA assigns one shared-memory budget to every block in a launch.  Size
  // that budget for the widest supported job; each block then lays out its
  // temporary arrays from the actual job span and gives the remainder to the
  // survivor tape.
  return canonical_tournament_layout_bytes(
      capacity, source_count, kCanonicalJobQuotients);
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

inline std::size_t initial_storage_capacity(
    std::size_t requested, std::size_t tile_capacity) {
  const std::size_t capacity = std::max(
      requested, tile_capacity * kBatchesPerEpoch);
  if (capacity > kMaximumPublicationRows)
    throw std::invalid_argument(
        "GPULSMOpt capacity exceeds 32-bit key space");
  return capacity;
}

inline std::size_t initial_level_capacity(
    std::size_t requested, std::size_t fallback,
    std::size_t maximum) {
  const std::size_t capacity = requested ? requested : fallback;
  return std::min(maximum, std::max<std::size_t>(1u, capacity));
}

inline std::size_t maximum_resident_merge_jobs(
    std::size_t maximum_raw_rows, std::uint32_t merge_capacity) {
  // Reserve one job per section plus hot pieces.
  const std::size_t safe =
      merge_capacity - (kMaximumMergeSources - 1u);
  return std::size_t{kQuotients} +
      (maximum_raw_rows + safe - 1u) / safe + 1u;
}

struct CanonicalLevelLayout {
  std::size_t initial_pool_capacity{};
  std::size_t highest_regular_capacity{};
  std::uint32_t regular_level_count{};
  std::uint32_t level_count{};
};

// A radix-4 tier owns up to three immutable, equal-capacity slots.  The
// fourth run is carried into the next tier.  A partially configured final
// tier may have fewer slots; when none of those slots can hold the complete
// dictionary, append one lazily mapped full-capacity consolidation slot.
// That final slot preserves indefinite update/rollover support without
// charging normal construction for an otherwise unused full-size bank.
inline CanonicalLevelLayout canonical_level_layout(
    std::size_t maximum_raw_rows, std::size_t epoch_capacity) {
  CanonicalLevelLayout layout{};
  std::size_t remaining = maximum_raw_rows;
  std::size_t capacity = std::min(maximum_raw_rows, epoch_capacity);
  while (remaining) {
    const std::size_t slots = std::min<std::size_t>(
        3u, (remaining + capacity - 1u) / capacity);
    if (slots > (std::numeric_limits<std::size_t>::max() -
                 layout.initial_pool_capacity) / capacity)
      throw std::bad_alloc();
    layout.initial_pool_capacity += slots * capacity;
    if (layout.regular_level_count > kMaximumLevels - slots)
      throw std::invalid_argument("GPULSMOpt radix-4 slot count overflow");
    layout.regular_level_count += static_cast<std::uint32_t>(slots);
    layout.highest_regular_capacity = capacity;
    const std::size_t covered = std::min(remaining, slots * capacity);
    remaining -= covered;
    if (!remaining) break;
    capacity = capacity > maximum_raw_rows / 4u
        ? maximum_raw_rows : capacity * 4u;
  }
  layout.level_count = layout.regular_level_count;
  if (layout.highest_regular_capacity < maximum_raw_rows) {
    if (layout.level_count == kMaximumLevels)
      throw std::invalid_argument("GPULSMOpt radix-4 slot count overflow");
    ++layout.level_count;
  }
  return layout;
}

inline std::size_t preassigned_level_pool_capacity(
    std::size_t maximum_raw_rows, std::size_t epoch_capacity) {
  return canonical_level_layout(
      maximum_raw_rows, epoch_capacity).initial_pool_capacity;
}

inline std::uint32_t canonical_regular_level_count(
    std::size_t maximum_raw_rows, std::size_t epoch_capacity) {
  return canonical_level_layout(
      maximum_raw_rows, epoch_capacity).regular_level_count;
}

inline std::uint32_t canonical_level_count(
    std::size_t maximum_raw_rows, std::size_t epoch_capacity) {
  return canonical_level_layout(maximum_raw_rows, epoch_capacity).level_count;
}


struct Row {
  std::uint32_t value;
  std::uint16_t key;
  std::uint16_t flags;
};

// Keys and values use parallel storage streams.
struct ResidentRows {
  std::uint32_t *key_flags{};
  std::uint32_t *values{};

  __host__ __device__ __forceinline__ ResidentRows operator+(
      std::uint64_t offset) const {
    return {key_flags + offset, values + offset};
  }

  __host__ __device__ __forceinline__ std::uint16_t key_at(
      std::uint64_t position) const {
    return static_cast<std::uint16_t>(key_flags[position]);
  }

  __host__ __device__ __forceinline__ Row operator[](
      std::uint64_t position) const {
    const std::uint32_t packed = key_flags[position];
    return {values[position], static_cast<std::uint16_t>(packed),
            static_cast<std::uint16_t>(packed >> 16u)};
  }

  __host__ __device__ __forceinline__ void store(
      std::uint64_t position, const Row &row) const {
    values[position] = row.value;
    key_flags[position] = std::uint32_t{row.key} |
        (std::uint32_t{row.flags} << 16u);
  }
};

static_assert(sizeof(ResidentRows) == 2u * sizeof(void *));

__host__ __device__ __forceinline__ Row make_row(
    std::uint32_t key, std::uint32_t value, std::uint32_t flags) {
  return {value, static_cast<std::uint16_t>(key),
          static_cast<std::uint16_t>(flags)};
}

__host__ __device__ __forceinline__ std::uint32_t full_key(
    std::uint32_t q, std::uint32_t suffix) {
  return (q << 16u) | suffix;
}

__host__ __device__ __forceinline__ std::uint32_t key_suffix(
    std::uint32_t key) {
  return key & 0xffffu;
}


__host__ __device__ __forceinline__ std::uint32_t raw_age(
    std::uint32_t logical_position, std::uint32_t batch_stride) {
  const std::uint32_t batch = logical_position >> kBatchPositionBits;
  const std::uint32_t position =
      logical_position & ((1u << kBatchPositionBits) - 1u);
  return kMaximumLevels + batch * batch_stride + position;
}

struct RawAssignment {
  std::uint32_t key;
  std::uint32_t value;
  std::uint32_t metadata;
};

struct alignas(8) RawPayload {
  std::uint32_t value;
  std::uint32_t metadata;
};

static_assert(sizeof(Row) == 8u);
static_assert(sizeof(RawAssignment) == 12u);
static_assert(sizeof(RawPayload) == 8u);
static_assert(alignof(RawPayload) == 8u);

inline std::size_t canonical_capacity_reservation_bytes(
    std::uint32_t capacity) {
  // Preserve the established fixed and per-record safety margin used when
  // sizing canonical jobs.  The production kernels allocate their actual
  // shared-memory layouts independently after this capacity is selected.
  constexpr std::uint32_t reservation_cells = 2u * kFoundationCells;
  constexpr std::size_t cell_words =
      (reservation_cells + 2u) + reservation_cells + reservation_cells +
      reservation_cells + (reservation_cells + 1u);
  const std::size_t tombstone_bytes =
      std::size_t{(capacity + 31u) / 32u} * sizeof(std::uint32_t);
  const std::size_t cell_bytes = cell_words * sizeof(std::uint16_t);
  return std::size_t{capacity} * sizeof(std::uint32_t) +
      std::size_t{capacity + 1u} * sizeof(std::uint16_t) * 2u +
      tombstone_bytes + cell_bytes;
}

constexpr std::uint32_t kRawTombstone = 0x80000000u;


__device__ __forceinline__ void store_raw_payload(
    RawPayload *__restrict__ destination, std::uint32_t position,
    std::uint32_t value, std::uint32_t metadata) {
  reinterpret_cast<uint2 *>(destination)[position] =
      make_uint2(value, metadata);
}

__host__ __device__ __forceinline__ std::uint32_t raw_position(
    const RawAssignment &assignment) {
  return assignment.metadata & ~kRawTombstone;
}

__host__ __device__ __forceinline__ std::uint32_t raw_position(
    const RawPayload &payload) {
  return payload.metadata & ~kRawTombstone;
}

__host__ __device__ __forceinline__ Row raw_row(
    const RawAssignment &assignment) {
  return make_row(assignment.key, assignment.value,
                  assignment.metadata & kRawTombstone
                      ? kTombstone : 0u);
}

__host__ __device__ __forceinline__ Row raw_row(
    std::uint32_t key, const RawPayload &payload) {
  return make_row(key, payload.value,
                  payload.metadata & kRawTombstone ? kTombstone : 0u);
}

__host__ __device__ __forceinline__ RawAssignment load_raw_assignment(
    const std::uint32_t *keys, const RawPayload *payloads,
    std::uint32_t index) {
  const RawPayload payload = payloads[index];
  return {keys[index], payload.value, payload.metadata};
}

template <class T> inline std::size_t maximum_resident_elements() {
  std::size_t free_bytes{}, total_bytes{};
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  (void)free_bytes;
  return total_bytes / sizeof(T);
}

struct Descriptor {
  std::uint64_t bits{};
  __host__ __device__ static Descriptor make(std::uint64_t offset,
                                             std::uint32_t count) {
    return {std::uint64_t{offset} |
            (std::uint64_t{count} << kDescriptorOffsetBits)};
  }
  __host__ __device__ std::uint64_t offset() const {
    return bits & kDescriptorOffsetMask;
  }
  __host__ __device__ std::uint32_t count() const {
    return static_cast<std::uint32_t>(bits >> kDescriptorOffsetBits);
  }
};

static_assert(sizeof(Descriptor) == 8u);

// Route slices form one logical sorted section.
struct RouteHeader {
  std::uint32_t begin{};
  std::uint32_t count{};
};

struct RouteSlice {
  Descriptor rows{};
  std::uint32_t suffix_begin{};
  std::uint32_t suffix_end{};
};

static_assert(sizeof(RouteHeader) == 8u);
static_assert(sizeof(RouteSlice) == 16u);

struct DeviceLevelState {
  std::uint32_t storage_generation{};
};

struct DeviceManifest {
  std::uint64_t occupied_level_mask{};
  std::uint32_t active_levels{};
  std::uint32_t foundation_level{kMaximumLevels};
  std::uint32_t generation{};
  DeviceLevelState levels[kMaximumLevels]{};
};

struct LevelStorageSpan {
  std::uint64_t begin{};
  std::uint64_t capacity{};
};

enum : std::uint32_t {
  kPublicationSuccess = 0u,
  kPublicationJobOverflow = 1u << 0u,
  kPublicationOutputOverflow = 1u << 3u,
  kPublicationJobTooLarge = 1u << 4u,
  kPublicationLevelOverflow = 1u << 5u,
};

struct ResidentPublicationPlan {
  std::uint32_t selected_count{};
  std::uint32_t active_manifest{};
  std::uint32_t inactive_manifest{};
  std::uint32_t destination_level{};
  std::uint32_t source_level_limit{};
  std::uint32_t source_count{};
  std::uint32_t keep_tombstones{};
  std::uint32_t output_generation{};
  std::uint32_t job_count{};
  std::uint32_t status{};
  std::uint32_t job_capacity{};
  std::uint32_t tournament_workspace_bytes{};
  std::uint64_t output_begin{};
  std::uint64_t output_capacity{};
  std::uint64_t raw_reservation{};
  std::uint64_t survivor_count{};
};

static_assert(sizeof(ResidentPublicationPlan) == 80u);

__device__ __forceinline__ std::uint32_t canonical_job_capacity(
    const ResidentPublicationPlan *plan, std::uint32_t quotient_count) {
  if (plan->tournament_workspace_bytes)
    return canonical_tournament_capacity(
        plan->tournament_workspace_bytes, plan->source_count,
        quotient_count);
  return plan->job_capacity;
}

struct DeviceManifestSnapshot {
  std::uint64_t occupied_level_mask{};
  std::uint32_t active_levels{};
  std::uint32_t foundation_level{kMaximumLevels};
};

__device__ __forceinline__ DeviceManifestSnapshot load_active_manifest(
    const DeviceManifest *manifests,
    const std::uint32_t *active_manifest) {
  // Read the published manifest once.
  const std::uint32_t index = __ldg(active_manifest) & 1u;
  const DeviceManifest *manifest = manifests + index;
  return {manifest->occupied_level_mask, manifest->active_levels,
          manifest->foundation_level};
}

__device__ __forceinline__ DeviceManifestSnapshot load_query_manifest(
    const std::uint64_t *query_occupied_level_mask) {
  const std::uint64_t occupied = __ldg(query_occupied_level_mask);
  const std::uint32_t active_levels = occupied
      ? 64u - static_cast<std::uint32_t>(__clzll(occupied)) : 0u;
  return {occupied, active_levels,
          active_levels ? active_levels - 1u : kMaximumLevels};
}

__device__ __forceinline__ bool level_is_occupied(
    std::uint64_t mask, std::uint32_t level) {
  return (mask & (std::uint64_t{1} << level)) != 0u;
}

__host__ __device__ __forceinline__ std::size_t descriptor_index(
    std::uint32_t q, std::uint32_t level) {
  return std::size_t{q} * kMaximumLevels + level;
}

struct TaggedRow {
  Row row;
  std::uint32_t age;
};

static_assert(sizeof(TaggedRow) == 12u);

struct RangeFragment {
  std::uint32_t query;
  std::uint32_t quotient;
};

static_assert(sizeof(RangeFragment) == 8u);

struct SectionRangeFragment {
  std::uint32_t original_index;
  std::uint16_t low_suffix;
  std::uint16_t high_suffix;
};

static_assert(sizeof(SectionRangeFragment) == 8u);

struct SectionRangeTask {
  std::uint32_t quotient;
  std::uint32_t begin;
  std::uint32_t end;
};

static_assert(sizeof(SectionRangeTask) == 12u);
constexpr std::uint32_t kSectionTaskFragments = kSectionRangeThreads;

struct RangeFragmentBounds {
  std::uint32_t update_begin;
  std::uint32_t update_end;
  std::uint32_t base_begin;
  std::uint32_t base_end;
};

static_assert(sizeof(RangeFragmentBounds) == 16u);

struct BalancedMergeJob {
  std::uint64_t key_begin{};
  std::uint64_t key_end{};
  std::uint32_t quotient_begin;
  std::uint32_t quotient_end;
};

static_assert(sizeof(BalancedMergeJob) == 24u);

struct CanonicalJobPrefix {
  unsigned long long prefix{};
  std::uint32_t count{};
  std::uint32_t ready{};
};

static_assert(sizeof(CanonicalJobPrefix) == 16u);

__device__ __forceinline__ unsigned long long canonical_job_prefix(
    std::uint32_t job_index, std::uint32_t count,
    CanonicalJobPrefix *prefixes);

template <class T> class Buffer {
public:
  Buffer() = default;
  explicit Buffer(std::size_t count) { resize(count); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  ~Buffer() { if (pointer_ && owns_) cudaFree(pointer_); }
  void resize(std::size_t count) {
    if (pointer_ && owns_) CUDA_CHECK(cudaFree(pointer_));
    pointer_ = nullptr;
    count_ = count;
    owns_ = true;
    if (count)
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pointer_),
                            count * sizeof(T)));
  }
  void attach(T *pointer, std::size_t count) {
    if (pointer_ && owns_) CUDA_CHECK(cudaFree(pointer_));
    pointer_ = pointer;
    count_ = count;
    owns_ = false;
  }
  T *data() { return pointer_; }
  std::size_t size() const { return count_; }
private:
  T *pointer_{};
  std::size_t count_{};
  bool owns_{true};
};

// Pinned receipts support asynchronous publication.
template <class T> class PinnedBuffer {
public:
  explicit PinnedBuffer(std::size_t count) {
    if (count)
      CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&pointer_),
                                count * sizeof(T)));
  }
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;
  ~PinnedBuffer() {
    if (pointer_) cudaFreeHost(pointer_);
  }
  T *data() { return pointer_; }

private:
  T *pointer_{};
};

inline void check_driver(CUresult result, const char *file, int line) {
  if (result == CUDA_SUCCESS) return;
  throw std::runtime_error(
      std::string("CUDA driver error ") + std::to_string(result) + " at " +
      file + ":" + std::to_string(line));
}

#define GPULSMOPT_CU_CHECK(call) \
  ::gpulsmopt2_detail::check_driver((call), __FILE__, __LINE__)

struct VmmFunctions {
  decltype(&cuMemAddressReserve) reserve{};
  decltype(&cuMemAddressFree) free_address{};
  decltype(&cuMemCreate) create{};
  decltype(&cuMemRelease) release{};
  decltype(&cuMemMap) map{};
  decltype(&cuMemUnmap) unmap{};
  decltype(&cuMemSetAccess) set_access{};
  decltype(&cuMemGetAllocationGranularity) granularity{};

  template <class Function>
  static Function load(const char *name) {
    void *pointer = nullptr;
    CUDA_CHECK(cudaGetDriverEntryPoint(
        name, &pointer, cudaEnableDefault, nullptr));
    if (!pointer)
      throw std::runtime_error(std::string("missing CUDA driver entry ") +
                               name);
    return reinterpret_cast<Function>(pointer);
  }

  VmmFunctions()
      : reserve(load<decltype(reserve)>("cuMemAddressReserve")),
        free_address(load<decltype(free_address)>("cuMemAddressFree")),
        create(load<decltype(create)>("cuMemCreate")),
        release(load<decltype(release)>("cuMemRelease")),
        map(load<decltype(map)>("cuMemMap")),
        unmap(load<decltype(unmap)>("cuMemUnmap")),
        set_access(load<decltype(set_access)>("cuMemSetAccess")),
        granularity(load<decltype(granularity)>(
            "cuMemGetAllocationGranularity")) {}
};

inline VmmFunctions &vmm_functions() {
  static VmmFunctions functions;
  return functions;
}

template <class T> class VirtualBuffer {
public:
  VirtualBuffer(std::size_t maximum_count, std::size_t initial_count) {
    reserve(maximum_count);
    grow(initial_count);
  }
  VirtualBuffer(const VirtualBuffer &) = delete;
  VirtualBuffer &operator=(const VirtualBuffer &) = delete;
  VirtualBuffer(VirtualBuffer &&) = delete;
  VirtualBuffer &operator=(VirtualBuffer &&) = delete;
  ~VirtualBuffer() { release(); }

  void grow(std::size_t requested_count) {
    if (requested_count <= size()) return;
    if (requested_count > maximum_count_)
      throw std::bad_alloc();
    std::size_t target_count = requested_count;
    if (mapped_bytes_) {
      const std::size_t doubled = std::min(
          maximum_count_, size() > maximum_count_ / 2u
              ? maximum_count_ : size() * 2u);
      target_count = std::max(target_count, doubled);
    }
    std::size_t target_bytes = align_up(target_count * sizeof(T));
    target_bytes = std::min(target_bytes, reserved_bytes_);
    const std::size_t extension = target_bytes - mapped_bytes_;
    auto &functions = vmm_functions();
    CUmemGenericAllocationHandle handle{};
    GPULSMOPT_CU_CHECK(functions.create(
        &handle, extension, &property_, 0u));
    bool mapped = false;
    try {
      GPULSMOPT_CU_CHECK(functions.map(
          address_ + mapped_bytes_, extension, 0u, handle, 0u));
      mapped = true;
      CUmemAccessDesc access{};
      access.location = property_.location;
      access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      GPULSMOPT_CU_CHECK(functions.set_access(
          address_ + mapped_bytes_, extension, &access, 1u));
    } catch (...) {
      if (mapped)
        functions.unmap(address_ + mapped_bytes_, extension);
      functions.release(handle);
      throw;
    }
    mappings_.push_back({mapped_bytes_, extension, handle});
    mapped_bytes_ = target_bytes;
  }

  T *data() {
    return reinterpret_cast<T *>(static_cast<std::uintptr_t>(address_));
  }
  std::size_t size() const { return mapped_bytes_ / sizeof(T); }
private:
  struct Mapping {
    std::size_t offset;
    std::size_t bytes;
    CUmemGenericAllocationHandle handle;
  };

  std::size_t align_up(std::size_t bytes) const {
    return (bytes + granularity_ - 1u) / granularity_ * granularity_;
  }

  void reserve(std::size_t maximum_count) {
    if (!maximum_count || maximum_count >
            std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    CUDA_CHECK(cudaFree(nullptr));
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    property_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    property_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    property_.location.id = device;
    auto &functions = vmm_functions();
    GPULSMOPT_CU_CHECK(functions.granularity(
        &granularity_, &property_, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    maximum_count_ = maximum_count;
    reserved_bytes_ = align_up(maximum_count * sizeof(T));
    GPULSMOPT_CU_CHECK(functions.reserve(
        &address_, reserved_bytes_, granularity_, 0u, 0u));
  }

  void release() noexcept {
    if (!address_) return;
    auto &functions = vmm_functions();
    for (auto it = mappings_.rbegin(); it != mappings_.rend(); ++it) {
      functions.unmap(address_ + it->offset, it->bytes);
      functions.release(it->handle);
    }
    functions.free_address(address_, reserved_bytes_);
    address_ = 0u;
    mapped_bytes_ = 0u;
  }

  CUdeviceptr address_{};
  std::size_t granularity_{};
  std::size_t reserved_bytes_{};
  std::size_t mapped_bytes_{};
  std::size_t maximum_count_{};
  CUmemAllocationProp property_{};
  std::vector<Mapping> mappings_;
};

__device__ __forceinline__ std::uint32_t size_class_for(
    std::uint32_t count) {
  if (count <= 1u) return 0u;
  return 32u - static_cast<std::uint32_t>(__clz(count - 1u));
}

__device__ __forceinline__ bool tagged_less(const TaggedRow &a,
                                            const TaggedRow &b) {
  const bool ai = a.age == kInvalidAge;
  const bool bi = b.age == kInvalidAge;
  if (ai != bi) return !ai;
  if (ai) return false;
  if (a.row.key != b.row.key) return a.row.key < b.row.key;
  return a.age < b.age;
}

__device__ RawAssignment load_pending_raw_ordinal(
    const std::uint32_t *keys, const RawPayload *payloads,
    const std::uint32_t *offsets,
    std::uint32_t batch_stride, std::uint32_t pending_batches,
    std::uint32_t q, std::uint32_t ordinal) {
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t oi = std::size_t{batch} * (kQuotients + 1u) + q;
    const std::uint32_t begin = offsets[oi];
    const std::uint32_t count = offsets[oi + 1u] - begin;
    if (ordinal < count) {
      const std::uint32_t index =
          batch * batch_stride + begin + ordinal;
      return load_raw_assignment(keys, payloads, index);
    }
    ordinal -= count;
  }
  return {};
}

__device__ __forceinline__ std::uint32_t lower_bound_rows(
    const Row *rows, std::uint32_t count, std::uint32_t key) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    if (rows[mid].key < key) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

__device__ __forceinline__ std::uint32_t lower_bound_rows(
    ResidentRows rows, std::uint32_t count, std::uint32_t key) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    if (rows.key_at(mid) < key) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

// Return the matching row with its position.
__device__ __forceinline__ bool find_unique_point_row(
    const Row *rows, std::uint32_t count, std::uint32_t key, Row &result) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    const Row candidate = rows[mid];
    if (candidate.key < key) lo = mid + 1u;
    else if (candidate.key > key) hi = mid;
    else {
      result = candidate;
      return true;
    }
  }
  return false;
}

__device__ __forceinline__ bool find_unique_point_row(
    ResidentRows rows, std::uint32_t count, std::uint32_t key, Row &result) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    const std::uint16_t candidate = rows.key_at(mid);
    if (candidate < key) lo = mid + 1u;
    else if (candidate > key) hi = mid;
    else {
      result = rows[mid];
      return true;
    }
  }
  return false;
}

__device__ __forceinline__ std::uint32_t upper_bound_rows(
    const Row *rows, std::uint32_t count, std::uint32_t key) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    if (rows[mid].key <= key) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

__device__ __forceinline__ std::uint32_t upper_bound_rows(
    ResidentRows rows, std::uint32_t count, std::uint32_t key) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    if (rows.key_at(mid) <= key) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

// Search split routes as one logical section.
__device__ __forceinline__ std::uint64_t logical_section_position(
    std::uint32_t q, std::uint32_t level, std::uint32_t position,
    const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets) {
  const RouteHeader header = route_headers[descriptor_index(q, level)];
  if (header.count == 1u)
    return route_slices[header.begin].rows.offset() + position;
  const std::uint32_t section_begin = level_q_logical_offsets[
      std::size_t{level} * (kQuotients + 1u) + q];
  const std::uint32_t logical = section_begin + position;
  std::uint32_t low = 0u, high = header.count;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (route_logical_begins[header.begin + middle] <= logical)
      low = middle + 1u;
    else
      high = middle;
  }
  if (low) {
    const std::uint32_t route = header.begin + low - 1u;
    const RouteSlice slice = route_slices[route];
    const std::uint32_t begin = route_logical_begins[route];
    if (logical < begin + slice.rows.count())
      return slice.rows.offset() + logical - begin;
  }
  return std::numeric_limits<std::uint64_t>::max();
}

__device__ __forceinline__ Row logical_section_row(
    std::uint32_t q, std::uint32_t level, std::uint32_t position,
    ResidentRows arena, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets) {
  const std::uint64_t physical = logical_section_position(
      q, level, position, route_headers, route_slices,
      route_logical_begins, level_q_logical_offsets);
  return physical == std::numeric_limits<std::uint64_t>::max()
      ? Row{} : arena[physical];
}

constexpr std::uint32_t kRangeHotRawLocator = 1u << 31u;

__host__ __device__ __forceinline__ std::uint64_t range_hot_token(
    std::uint32_t key, std::uint32_t locator) {
  return (std::uint64_t{key} << 32u) | locator;
}

__host__ __device__ __forceinline__ std::uint32_t range_hot_key(
    std::uint64_t token) {
  return static_cast<std::uint32_t>(token >> 32u);
}

__host__ __device__ __forceinline__ std::uint32_t range_hot_locator(
    std::uint64_t token) {
  return static_cast<std::uint32_t>(token);
}

struct RangeHotTokenKey {
  __host__ __device__ std::uint32_t operator()(std::uint64_t token) const {
    return range_hot_key(token);
  }
};

struct RangeHotNewestToken {
  const RawPayload *raw_payloads{};
  std::uint32_t raw_record_capacity{};

  __device__ bool valid(std::uint64_t token) const {
    const std::uint32_t locator = range_hot_locator(token);
    return locator & kRangeHotRawLocator
        ? (locator & ~kRangeHotRawLocator) < raw_record_capacity
        : (locator >> 16u) < kMaximumLevels;
  }

  __device__ std::uint32_t age(std::uint64_t token) const {
    const std::uint32_t locator = range_hot_locator(token);
    if (locator & kRangeHotRawLocator)
      return kMaximumLevels +
          raw_position(raw_payloads[locator & ~kRangeHotRawLocator]);
    return kMaximumLevels - 1u - (locator >> 16u);
  }

  __device__ std::uint64_t operator()(std::uint64_t first,
                                      std::uint64_t second) const {
    const bool first_valid = valid(first), second_valid = valid(second);
    if (first_valid != second_valid) return second_valid ? second : first;
    if (!first_valid) return 0u;
    return age(second) > age(first) ? second : first;
  }
};

struct RangeHotTokenRow {
  const RawPayload *raw_payloads{};
  ResidentRows arena{};
  const RouteHeader *route_headers{};
  const RouteSlice *route_slices{};
  const std::uint32_t *route_logical_begins{};
  const std::uint32_t *level_q_logical_offsets{};

  __device__ Row operator()(std::uint64_t token) const {
    const std::uint32_t key = range_hot_key(token);
    const std::uint32_t locator = range_hot_locator(token);
    if (locator & kRangeHotRawLocator) {
      const std::uint32_t record = locator & ~kRangeHotRawLocator;
      return raw_row(key, raw_payloads[record]);
    }
    const std::uint32_t level = locator >> 16u;
    const std::uint32_t position = locator & 0xffffu;
    return logical_section_row(
        key >> 16u, level, position, arena, route_headers, route_slices,
        route_logical_begins, level_q_logical_offsets);
  }
};

__global__ void count_range_hot_newer_rows_kernel(
    const std::uint32_t *raw_offsets, std::uint32_t pending_batches,
    const Descriptor *descriptors, const std::uint64_t *occupied_mask,
    std::uint64_t *counts) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  if (q == kQuotients) {
    counts[q] = 0u;
    return;
  }
  const DeviceManifestSnapshot manifest = load_query_manifest(occupied_mask);
  std::uint64_t physical = 0u;
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t index =
        std::size_t{batch} * (kQuotients + 1u) + q;
    physical += raw_offsets[index + 1u] - raw_offsets[index];
  }
  for (std::uint32_t level = 0u; level < manifest.active_levels; ++level)
    if (level != manifest.foundation_level &&
        level_is_occupied(manifest.occupied_level_mask, level))
      physical += descriptors[descriptor_index(q, level)].count();
  counts[q] = physical > kRangeOnChipNewerRows ? physical : 0u;
}

__global__ void make_range_hot_window_offsets_kernel(
    const std::uint64_t *global_offsets, std::uint32_t quotient_begin,
    std::uint32_t quotient_count, std::uint64_t window_base,
    std::uint32_t *window_offsets) {
  const std::uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local > quotient_count) return;
  window_offsets[local] = static_cast<std::uint32_t>(
      global_offsets[quotient_begin + local] - window_base);
}

__global__ void emit_range_hot_tokens_kernel(
    const std::uint32_t *raw_keys, const std::uint32_t *raw_offsets,
    std::uint32_t batch_stride, std::uint32_t pending_batches,
    ResidentRows arena, const Descriptor *descriptors,
    const RouteHeader *route_headers, const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    const std::uint64_t *occupied_mask, const std::uint64_t *hot_counts,
    const std::uint64_t *hot_offsets, std::uint32_t quotient_begin,
    std::uint32_t quotient_end, std::uint64_t window_base,
    std::uint64_t *tokens) {
  const std::uint32_t q = quotient_begin + blockIdx.x;
  if (q >= quotient_end || !hot_counts[q]) return;
  const DeviceManifestSnapshot manifest = load_query_manifest(occupied_mask);
  std::uint64_t cursor = hot_offsets[q] - window_base;
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t offset_index =
        std::size_t{batch} * (kQuotients + 1u) + q;
    const std::uint32_t begin = raw_offsets[offset_index];
    const std::uint32_t end = raw_offsets[offset_index + 1u];
    for (std::uint32_t position = begin + threadIdx.x;
         position < end; position += blockDim.x) {
      const std::uint32_t record = batch * batch_stride + position;
      tokens[cursor + position - begin] = range_hot_token(
          raw_keys[record], kRangeHotRawLocator | record);
    }
    cursor += end - begin;
  }
  for (std::uint32_t level = 0u; level < manifest.active_levels; ++level) {
    if (level == manifest.foundation_level ||
        !level_is_occupied(manifest.occupied_level_mask, level))
      continue;
    const RouteHeader header = route_headers[descriptor_index(q, level)];
    const std::uint32_t section_begin = level_q_logical_offsets[
        std::size_t{level} * (kQuotients + 1u) + q];
    for (std::uint32_t local = 0u; local < header.count; ++local) {
      const std::uint32_t route_index = header.begin + local;
      const RouteSlice route = route_slices[route_index];
      const std::uint32_t logical_begin =
          route_logical_begins[route_index] - section_begin;
      const ResidentRows rows = arena + route.rows.offset();
      for (std::uint32_t position = threadIdx.x;
           position < route.rows.count(); position += blockDim.x) {
        const Row row = rows[position];
        const std::uint32_t locator =
            (level << 16u) | (logical_begin + position);
        tokens[cursor + logical_begin + position] = range_hot_token(
            full_key(q, row.key), locator);
      }
    }
    cursor += descriptors[descriptor_index(q, level)].count();
  }
}

__global__ void build_range_hot_descriptors_kernel(
    const std::uint32_t *keys, const std::uint32_t *selected_count,
    std::uint32_t quotient_begin, std::uint32_t quotient_end,
    std::uint64_t output_base, Descriptor *descriptors) {
  const std::uint32_t q = quotient_begin +
      blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= quotient_end) return;
  const std::uint32_t count = *selected_count;
  const std::uint64_t low_key = std::uint64_t{q} << 16u;
  const std::uint64_t high_key = std::uint64_t{q + 1u} << 16u;
  std::uint32_t low = 0u, high = count;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (std::uint64_t{keys[middle]} < low_key) low = middle + 1u;
    else high = middle;
  }
  const std::uint32_t begin = low;
  high = count;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (std::uint64_t{keys[middle]} < high_key) low = middle + 1u;
    else high = middle;
  }
  descriptors[q] = Descriptor::make(output_base + begin, low - begin);
}

__global__ void materialize_range_hot_winners_kernel(
    Row *rows, const std::uint32_t *selected_count,
    const RawPayload *raw_payloads, ResidentRows arena,
    const RouteHeader *route_headers, const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= *selected_count) return;
  std::uint64_t token{};
  memcpy(&token, rows + index, sizeof(token));
  const RangeHotTokenRow transform{
      raw_payloads, arena, route_headers, route_slices,
      route_logical_begins, level_q_logical_offsets};
  rows[index] = transform(token);
}

struct SumRowsAggregate {
  using State = unsigned long long;
  __device__ static State identity() { return 0ull; }
  __device__ static State consume(State state, const Row &row) {
    return state + row.value;
  }
  __device__ State operator()(State a, State b) const { return a + b; }
};

__device__ __forceinline__ std::uint32_t route_range_row_count(
    std::uint32_t low, std::uint32_t high, ResidentRows arena,
    RouteHeader header, const RouteSlice *route_slices) {
  std::uint32_t count = 0u;
  for (std::uint32_t local = 0u; local < header.count; ++local) {
    const RouteSlice route = route_slices[header.begin + local];
    if (route.suffix_end <= low || route.suffix_begin > high) continue;
    const ResidentRows rows = arena + route.rows.offset();
    const std::uint32_t begin = lower_bound_rows(rows, route.rows.count(), low);
    const std::uint32_t end =
        upper_bound_rows(rows, route.rows.count(), high);
    count += end - begin;
  }
  return count;
}

// Scan base rows against the resolved newer run.
template <class Aggregate>
__device__ __forceinline__ typename Aggregate::State
cooperative_sum_visible_route_runs(
    std::uint32_t low, std::uint32_t high,
    const Row *current, std::uint32_t current_count,
    ResidentRows arena, RouteHeader base_header,
    const RouteSlice *route_slices,
    std::uint32_t group_lane, std::uint32_t group_size) {
  typename Aggregate::State result = Aggregate::identity();
  const std::uint32_t update_begin =
      lower_bound_rows(current, current_count, low);
  const std::uint32_t update_end =
      upper_bound_rows(current, current_count, high);
  for (std::uint32_t index = update_begin + group_lane;
       index < update_end; index += group_size) {
    const Row row = current[index];
    if ((row.flags & kTombstone) == 0u)
      result = Aggregate::consume(result, row);
  }

  for (std::uint32_t local = 0u; local < base_header.count; ++local) {
    const RouteSlice route = route_slices[base_header.begin + local];
    if (route.suffix_end <= low || route.suffix_begin > high) continue;
    const ResidentRows rows = arena + route.rows.offset();
    const std::uint32_t begin = lower_bound_rows(rows, route.rows.count(), low);
    const std::uint32_t end =
        upper_bound_rows(rows, route.rows.count(), high);
    const std::uint32_t count = end - begin;
    const std::uint32_t lane_begin =
        begin + (std::uint64_t{count} * group_lane) / group_size;
    const std::uint32_t lane_end =
        begin + (std::uint64_t{count} * (group_lane + 1u)) / group_size;
    if (lane_begin == lane_end) continue;

    // Merge one base interval per worker.
    std::uint32_t update = update_begin + lower_bound_rows(
        current + update_begin, update_end - update_begin,
        rows[lane_begin].key);
    for (std::uint32_t index = lane_begin; index < lane_end; ++index) {
      const Row row = rows[index];
      while (update < update_end && current[update].key < row.key) ++update;
      const bool covered =
          update < update_end && current[update].key == row.key;
      if (!covered && (row.flags & kTombstone) == 0u)
        result = Aggregate::consume(result, row);
    }
  }
  return result;
}

struct RangeTileSource {
  const Row *plain{};
  ResidentRows resident{};
  bool plain_storage{};

  __device__ Row load(std::uint32_t position) const {
    return plain_storage ? plain[position] : resident[position];
  }
};

template <class Aggregate>
__device__ __forceinline__ void enumerate_range_source_tiles(
    const SectionRangeFragment *fragments, std::uint32_t fragment_begin,
    std::uint32_t fragment_count, RangeTileSource source,
    const Row *newer, std::uint32_t newer_count, bool hide_with_newer,
    const std::uint32_t *source_begins,
    const std::uint32_t *source_ends,
    const std::uint8_t *fragment_slots,
    const std::uint8_t *fragment_widths,
    const std::uint16_t *wave_fragment_begins,
    std::uint32_t wave_count,
    std::uint32_t *union_begins, std::uint32_t *union_ends,
    std::uint32_t *union_count, Row *row_tile,
    typename Aggregate::State *fragment_sums) {
  if (threadIdx.x == 0u) {
    std::uint32_t count = 0u;
    for (std::uint32_t fragment = 0u; fragment < fragment_count;
         ++fragment) {
      const std::uint32_t begin = source_begins[fragment];
      const std::uint32_t end = source_ends[fragment];
      if (begin == end) continue;
      if (!count || begin > union_ends[count - 1u]) {
        union_begins[count] = begin;
        union_ends[count] = end;
        ++count;
      } else {
        union_ends[count - 1u] = max(union_ends[count - 1u], end);
      }
    }
    *union_count = count;
  }
  __syncthreads();

  for (std::uint32_t interval = 0u; interval < *union_count; ++interval) {
    const std::uint32_t interval_begin = union_begins[interval];
    const std::uint32_t interval_end = union_ends[interval];
    for (std::uint32_t tile_begin = interval_begin;
         tile_begin < interval_end; tile_begin += kSectionRangeThreads) {
      const std::uint32_t tile_end =
          min(tile_begin + kSectionRangeThreads, interval_end);
      if (tile_begin + threadIdx.x < tile_end) {
        Row row = source.load(tile_begin + threadIdx.x);
        if (hide_with_newer && newer_count) {
          const std::uint32_t position =
              lower_bound_rows(newer, newer_count, row.key);
          if (position < newer_count && newer[position].key == row.key)
            row.flags |= kTombstone;
        }
        row_tile[threadIdx.x] = row;
      }
      __syncthreads();

      for (std::uint32_t wave = 0u; wave < wave_count; ++wave) {
        const std::uint32_t first = wave_fragment_begins[wave];
        const std::uint32_t last = wave_fragment_begins[wave + 1u];
        std::uint32_t low = first, high = last;
        while (low < high) {
          const std::uint32_t middle = (low + high) >> 1u;
          if (fragment_slots[middle] <= threadIdx.x) low = middle + 1u;
          else high = middle;
        }
        const std::uint32_t fragment = low ? low - 1u : last;
        const bool active = fragment >= first && fragment < last &&
            threadIdx.x >= fragment_slots[fragment] &&
            threadIdx.x < std::uint32_t{fragment_slots[fragment]} +
                fragment_widths[fragment];
        if (active) {
          const std::uint32_t width = fragment_widths[fragment];
          const std::uint32_t slot = fragment_slots[fragment];
          const std::uint32_t group_lane = threadIdx.x - slot;
          const std::uint32_t consume_begin =
              max(source_begins[fragment], tile_begin);
          const std::uint32_t consume_end =
              min(source_ends[fragment], tile_end);
          typename Aggregate::State local = Aggregate::identity();
          for (std::uint32_t position = consume_begin + group_lane;
               position < consume_end; position += width) {
            const Row row = row_tile[position - tile_begin];
            if ((row.flags & kTombstone) == 0u)
              local = Aggregate::consume(local, row);
          }
          const std::uint32_t warp_slot = slot & 31u;
          const unsigned mask = width == 32u ? 0xffffffffu
              : ((1u << width) - 1u) << warp_slot;
          for (std::uint32_t offset = width >> 1u; offset;
               offset >>= 1u)
            local = Aggregate{}(local, __shfl_down_sync(
                mask, local, offset, width));
          if (!group_lane)
            fragment_sums[fragment] = Aggregate{}(
                fragment_sums[fragment], local);
        }
      }
      __syncthreads();
    }
  }
}

__global__ void count_range_fragments_kernel(
    const std::uint32_t *low, const std::uint32_t *high,
    std::uint32_t query_count, std::uint32_t *counts) {
  const std::uint32_t query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query > query_count) return;
  if (query == query_count) {
    counts[query] = 0u;
    return;
  }
  const std::uint32_t lo = low[query], hi = high[query];
  const std::uint32_t count =
      lo <= hi ? (hi >> 16u) - (lo >> 16u) + 1u : 0u;
  counts[query] = count;
}

struct WidenFragmentCount {
  __host__ __device__ std::uint64_t operator()(
      std::uint32_t count) const {
    return count;
  }
};

__device__ __forceinline__ void emit_range_fragment(
    std::uint32_t index, std::uint32_t query, std::uint32_t quotient,
    std::uint32_t low, std::uint32_t high, RangeFragment *fragments,
    std::uint32_t *section_keys, SectionRangeFragment *section_fragments,
    bool section_owned) {
  if (section_owned) {
    const std::uint32_t q_low = quotient << 16u;
    const std::uint32_t clipped_low = max(low, q_low);
    const std::uint32_t clipped_high = min(high, q_low | 0xffffu);
    section_keys[index] = q_low | (clipped_low & 0xffffu);
    section_fragments[index] = {
        index, static_cast<std::uint16_t>(clipped_low),
        static_cast<std::uint16_t>(clipped_high)};
  } else {
    fragments[index] = {query, quotient};
  }
}

__global__ void adaptive_emit_range_fragments_kernel(
    const std::uint32_t *low, const std::uint32_t *high,
    const std::uint32_t *offsets, std::uint32_t query_count,
    RangeFragment *fragments, std::uint32_t *section_keys,
    SectionRangeFragment *section_fragments, bool section_owned) {
  constexpr std::uint32_t kWarps = kThreads / 32u;
  constexpr std::uint32_t kThreadFragments = 4u;
  const std::uint32_t lane = threadIdx.x & 31u;
  constexpr unsigned full_mask = 0xffffffffu;
  if (query_count == 1u) {
    const std::uint32_t count = offsets[1u];
    const std::uint32_t first = low[0u] >> 16u;
    for (std::uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
         local < count; local += gridDim.x * blockDim.x)
      emit_range_fragment(
          local, 0u, first + local, low[0u], high[0u], fragments,
          section_keys, section_fragments, section_owned);
    return;
  }

  const std::uint32_t warp_in_block = threadIdx.x >> 5u;
  const std::uint32_t global_warp = blockIdx.x * kWarps + warp_in_block;
  const std::uint32_t warp_count = gridDim.x * kWarps;
  for (std::uint32_t query_base = global_warp * 32u;
       query_base < query_count; query_base += warp_count * 32u) {
    const std::uint32_t query = query_base + lane;
    const bool valid = query < query_count && low[query] <= high[query];
    const std::uint32_t query_offset = valid ? offsets[query] : 0u;
    const std::uint32_t query_low = valid ? low[query] : 0u;
    const std::uint32_t query_high = valid ? high[query] : 0u;
    const std::uint32_t count = valid
        ? offsets[query + 1u] - offsets[query] : 0u;
    if (valid && count <= kThreadFragments) {
      const std::uint32_t first = query_low >> 16u;
      for (std::uint32_t local = 0u; local < count; ++local)
        emit_range_fragment(
            query_offset + local, query, first + local, query_low,
            query_high, fragments, section_keys, section_fragments,
            section_owned);
    }

    unsigned wide = __ballot_sync(
        full_mask, valid && count > kThreadFragments);
    while (wide) {
      const std::uint32_t owner = __ffs(wide) - 1u;
      const std::uint32_t wide_query = query_base + owner;
      const std::uint32_t wide_offset =
          __shfl_sync(full_mask, query_offset, owner);
      const std::uint32_t wide_count =
          __shfl_sync(full_mask, count, owner);
      const std::uint32_t wide_low =
          __shfl_sync(full_mask, query_low, owner);
      const std::uint32_t wide_high =
          __shfl_sync(full_mask, query_high, owner);
      const std::uint32_t first = wide_low >> 16u;
      for (std::uint32_t local = lane; local < wide_count; local += 32u)
        emit_range_fragment(
            wide_offset + local, wide_query, first + local, wide_low,
            wide_high, fragments, section_keys, section_fragments,
            section_owned);
      wide &= ~(1u << owner);
    }
  }
}

__global__ void adaptive_reduce_range_partials_kernel(
    const std::uint32_t *offsets, std::uint32_t query_count,
    const unsigned long long *partials, std::uint32_t *out_sums,
    unsigned long long *block_partials, std::uint32_t *completion_count) {
  constexpr std::uint32_t kWarps = kThreads / 32u;
  constexpr std::uint32_t kThreadPartials = 4u;
  constexpr unsigned full_mask = 0xffffffffu;
  __shared__ unsigned long long values[kThreads];
  __shared__ std::uint32_t last_block;

  if (query_count == 1u) {
    unsigned long long local = 0ull;
    const std::uint32_t count = offsets[1u];
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += gridDim.x * blockDim.x)
      local += partials[index];
    values[threadIdx.x] = local;
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2u; stride; stride >>= 1u) {
      if (threadIdx.x < stride)
        values[threadIdx.x] += values[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0u) {
      block_partials[blockIdx.x] = values[0u];
      __threadfence();
      last_block = atomicInc(completion_count, gridDim.x - 1u) ==
          gridDim.x - 1u;
    }
    __syncthreads();
    if (!last_block) return;

    local = 0ull;
    for (std::uint32_t block = threadIdx.x; block < gridDim.x;
         block += blockDim.x)
      local += block_partials[block];
    values[threadIdx.x] = local;
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2u; stride; stride >>= 1u) {
      if (threadIdx.x < stride)
        values[threadIdx.x] += values[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0u)
      out_sums[0u] = static_cast<std::uint32_t>(values[0u]);
    return;
  }

  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp_in_block = threadIdx.x >> 5u;
  const std::uint32_t global_warp = blockIdx.x * kWarps + warp_in_block;
  const std::uint32_t warp_count = gridDim.x * kWarps;
  for (std::uint32_t query_base = global_warp * 32u;
       query_base < query_count; query_base += warp_count * 32u) {
    const std::uint32_t query = query_base + lane;
    const bool valid = query < query_count;
    const std::uint32_t begin = valid ? offsets[query] : 0u;
    const std::uint32_t end = valid ? offsets[query + 1u] : 0u;
    const std::uint32_t count = end - begin;
    if (valid && count <= kThreadPartials) {
      unsigned long long local = 0ull;
      for (std::uint32_t index = begin; index < end; ++index)
        local += partials[index];
      out_sums[query] = static_cast<std::uint32_t>(local);
    }

    unsigned wide = __ballot_sync(
        full_mask, valid && count > kThreadPartials);
    while (wide) {
      const std::uint32_t owner = __ffs(wide) - 1u;
      const std::uint32_t wide_query = query_base + owner;
      const std::uint32_t wide_begin =
          __shfl_sync(full_mask, begin, owner);
      const std::uint32_t wide_end = __shfl_sync(full_mask, end, owner);
      unsigned long long local = 0ull;
      for (std::uint32_t index = wide_begin + lane;
           index < wide_end; index += 32u)
        local += partials[index];
      for (std::uint32_t offset = 16u; offset; offset >>= 1u)
        local += __shfl_down_sync(full_mask, local, offset);
      if (lane == 0u)
        out_sums[wide_query] = static_cast<std::uint32_t>(local);
      wide &= ~(1u << owner);
    }
  }
}

__global__ void find_section_fragment_offsets_kernel(
    const std::uint32_t *sorted_sections, std::uint32_t count,
    std::uint32_t *offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  if (q == kQuotients) {
    offsets[q] = count;
    return;
  }
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    const std::uint32_t section = sorted_sections[mid] >> 16u;
    if (section < q) lo = mid + 1u;
    else hi = mid;
  }
  offsets[q] = lo;
}

__global__ void count_section_range_tasks_kernel(
    const std::uint32_t *section_offsets, std::uint32_t *task_counts) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  if (q == kQuotients) {
    task_counts[q] = 0u;
    return;
  }
  const std::uint32_t begin = section_offsets[q];
  const std::uint32_t end = section_offsets[q + 1u];
  const std::uint32_t count = end - begin;
  task_counts[q] =
      (count + kSectionTaskFragments - 1u) / kSectionTaskFragments;
}

__global__ void emit_section_range_tasks_kernel(
    const std::uint32_t *section_offsets,
    const std::uint32_t *task_offsets, SectionRangeTask *tasks) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  const std::uint32_t begin = section_offsets[q];
  const std::uint32_t end = section_offsets[q + 1u];
  const std::uint32_t local_tasks = task_offsets[q + 1u] - task_offsets[q];
  const std::uint32_t task_base = task_offsets[q];
  for (std::uint32_t tile = 0u; tile < local_tasks; ++tile) {
    const std::uint32_t tile_begin =
        begin + tile * kSectionTaskFragments;
    tasks[task_base + tile] =
        {q, tile_begin, min(tile_begin + kSectionTaskFragments, end)};
  }
}

template <class Aggregate>
__global__ void cooperative_section_owned_range_kernel(
    const SectionRangeFragment *fragments,
    const SectionRangeTask *tasks,
    const std::uint32_t *task_count, ResidentRows arena,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices, const std::uint16_t *local_rank,
    const Row *hot_rows, const Descriptor *hot_descriptors, bool hot_ready,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches,
    typename Aggregate::State *aggregate_partials,
    const std::uint64_t *query_occupied_level_mask) {
  constexpr std::uint32_t kCapacity = kRangeOnChipNewerRows;
  using BlockScan = cub::BlockScan<std::uint32_t, kSectionRangeThreads>;
  union Workspace {
    Row merged[kCapacity];
    TaggedRow tagged[kCapacity];
    Row tile[kSectionRangeThreads];
  };
  __shared__ Row current[kCapacity];
  __shared__ Workspace workspace;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ Descriptor section_descriptors[kMaximumLevels];
  __shared__ std::uint32_t foundation_cell_ranks[kFoundationCells + 1u];
  __shared__ RangeFragmentBounds fragment_bounds[kSectionTaskFragments];
  __shared__ std::uint32_t fragment_work[kSectionTaskFragments];
  __shared__ typename Aggregate::State fragment_sums[kSectionTaskFragments];
  __shared__ std::uint32_t source_begins[kSectionTaskFragments];
  __shared__ std::uint32_t source_ends[kSectionTaskFragments];
  __shared__ std::uint32_t union_begins[kSectionTaskFragments];
  __shared__ std::uint32_t union_ends[kSectionTaskFragments];
  __shared__ std::uint8_t fragment_slots[kSectionTaskFragments];
  __shared__ std::uint8_t fragment_widths[kSectionTaskFragments];
  __shared__ std::uint16_t wave_fragment_begins[33u];
  __shared__ std::uint32_t quotient_shared;
  __shared__ std::uint32_t fragment_begin_shared;
  __shared__ std::uint32_t fragment_end_shared;
  __shared__ std::uint32_t current_count_shared;
  __shared__ std::uint32_t pending_count_shared;
  __shared__ std::uint32_t wave_count_shared;
  __shared__ std::uint32_t union_count_shared;
  __shared__ std::uint32_t ranks_valid_shared;
  __shared__ std::uint32_t task_valid_shared;
  __shared__ Descriptor hot_descriptor_shared;

  const DeviceManifestSnapshot manifest =
      load_query_manifest(query_occupied_level_mask);
  const std::uint32_t active_levels = manifest.active_levels;
  const std::uint32_t foundation_level = manifest.foundation_level;
  const std::uint64_t occupied_levels = manifest.occupied_level_mask;

  for (std::uint32_t task_index = blockIdx.x;;
       task_index += gridDim.x) {
    if (threadIdx.x == 0u) {
      task_valid_shared = task_index < *task_count;
      if (task_valid_shared) {
        const SectionRangeTask task = tasks[task_index];
        quotient_shared = task.quotient;
        fragment_begin_shared = task.begin;
        fragment_end_shared = task.end;
      }
    }
    __syncthreads();
    if (!task_valid_shared) return;

    const std::uint32_t q = quotient_shared;
    const std::uint32_t fragment_begin = fragment_begin_shared;
    const std::uint32_t fragment_end = fragment_end_shared;
    const std::uint32_t fragment_count = fragment_end - fragment_begin;
    if (threadIdx.x < active_levels) {
      section_descriptors[threadIdx.x] =
          level_is_occupied(occupied_levels, threadIdx.x)
          ? descriptors[descriptor_index(q, threadIdx.x)] : Descriptor{};
    }
    if (threadIdx.x == 0u) {
      hot_descriptor_shared = hot_ready ? hot_descriptors[q] : Descriptor{};
      pending_count_shared = 0u;
      current_count_shared = 0u;
      for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
        const std::size_t offset =
            std::size_t{batch} * (kQuotients + 1u) + q;
        pending_count_shared += raw_offsets[offset + 1u] - raw_offsets[offset];
      }
    }
    if (threadIdx.x < 32u) {
      bool ranked = false;
      Descriptor descriptor{};
      if (local_rank && foundation_level < active_levels) {
        const RouteHeader header =
            route_headers[descriptor_index(q, foundation_level)];
        descriptor = descriptors[descriptor_index(q, foundation_level)];
        ranked = header.count == 1u && cell_rank_supported(descriptor.count());
      }
      if (threadIdx.x == 0u) {
        ranks_valid_shared = ranked;
        foundation_cell_ranks[kFoundationCells] = descriptor.count();
      }
      if (ranked)
        for (std::uint32_t cell = threadIdx.x; cell < kFoundationCells;
             cell += 32u)
          foundation_cell_ranks[cell] =
              local_rank[std::size_t{q} * kFoundationCells + cell];
    }
    __syncthreads();

    const bool use_hot = hot_descriptor_shared.count() != 0u;
    if (!use_hot && pending_count_shared) {
      const std::uint32_t pending_count = pending_count_shared;
      if (pending_count <= 32u) {
        if (threadIdx.x < 32u) {
          const std::uint32_t lane = threadIdx.x;
          TaggedRow item{{0u, 0u, 0u}, kInvalidAge};
          if (lane < pending_count) {
            const RawAssignment loaded = load_pending_raw_ordinal(
                raw_keys, raw_payloads, raw_offsets, batch_stride,
                pending_batches, q, lane);
            item = {raw_row(loaded),
                    raw_age(raw_position(loaded), batch_stride)};
          }
          constexpr unsigned mask = 0xffffffffu;
          for (std::uint32_t width = 2u; width <= 32u; width <<= 1u)
            for (std::uint32_t stride = width >> 1u; stride;
                 stride >>= 1u) {
              TaggedRow other{};
              other.row.key = __shfl_xor_sync(mask, item.row.key, stride);
              other.row.value = __shfl_xor_sync(mask, item.row.value, stride);
              other.row.flags = __shfl_xor_sync(mask, item.row.flags, stride);
              other.age = __shfl_xor_sync(mask, item.age, stride);
              const bool ascending = (lane & width) == 0u;
              const bool take_min = ((lane & stride) == 0u) == ascending;
              if ((take_min && tagged_less(other, item)) ||
                  (!take_min && tagged_less(item, other)))
                item = other;
            }
          const std::uint32_t next_key =
              __shfl_down_sync(mask, item.row.key, 1u);
          const std::uint32_t next_age =
              __shfl_down_sync(mask, item.age, 1u);
          const bool winner = item.age != kInvalidAge &&
              (lane == 31u || next_age == kInvalidAge ||
               item.row.key != next_key);
          const unsigned winners = __ballot_sync(mask, winner);
          if (winner) {
            const unsigned before = lane ? ((1u << lane) - 1u) : 0u;
            current[__popc(winners & before)] = item.row;
          }
          if (lane == 0u) current_count_shared = __popc(winners);
        }
        __syncthreads();
      } else {
        const std::uint32_t sort_size = 1u << size_class_for(pending_count);
        for (std::uint32_t ordinal = threadIdx.x; ordinal < sort_size;
             ordinal += blockDim.x) {
          if (ordinal < pending_count) {
            const RawAssignment loaded = load_pending_raw_ordinal(
                raw_keys, raw_payloads, raw_offsets, batch_stride,
                pending_batches, q, ordinal);
            workspace.tagged[ordinal] =
                {raw_row(loaded), raw_age(raw_position(loaded), batch_stride)};
          } else {
            workspace.tagged[ordinal] = {{0u, 0u, 0u}, kInvalidAge};
          }
        }
        __syncthreads();
        for (std::uint32_t width = 2u; width <= sort_size; width <<= 1u)
          for (std::uint32_t stride = width >> 1u; stride;
               stride >>= 1u) {
            for (std::uint32_t index = threadIdx.x; index < sort_size;
                 index += blockDim.x) {
              const std::uint32_t other_index = index ^ stride;
              if (other_index > index) {
                const TaggedRow first = workspace.tagged[index];
                const TaggedRow second = workspace.tagged[other_index];
                const bool ascending = (index & width) == 0u;
                const bool swap = ascending ? tagged_less(second, first)
                                            : tagged_less(first, second);
                if (swap) {
                  workspace.tagged[index] = second;
                  workspace.tagged[other_index] = first;
                }
              }
            }
            __syncthreads();
          }
        const std::uint32_t index = threadIdx.x;
        const bool winner = index < sort_size &&
            workspace.tagged[index].age != kInvalidAge &&
            (index + 1u == sort_size ||
             workspace.tagged[index + 1u].age == kInvalidAge ||
             workspace.tagged[index].row.key !=
                 workspace.tagged[index + 1u].row.key);
        std::uint32_t destination{}, winner_count{};
        BlockScan(scan_storage).ExclusiveSum(
            std::uint32_t{winner}, destination, winner_count);
        if (winner) current[destination] = workspace.tagged[index].row;
        __syncthreads();
        if (threadIdx.x == 0u) current_count_shared = winner_count;
        __syncthreads();
      }
    }

    if (!use_hot) {
      for (std::uint32_t level = 0u; level < active_levels; ++level) {
        if (level == foundation_level ||
            !level_is_occupied(occupied_levels, level)) continue;
        const RouteHeader header = route_headers[descriptor_index(q, level)];
        for (std::uint32_t local = 0u; local < header.count; ++local) {
          const Descriptor descriptor = route_slices[header.begin + local].rows;
          const std::uint32_t source_count = descriptor.count();
          if (!source_count) continue;
          const ResidentRows source = arena + descriptor.offset();
          const std::uint32_t old_count = current_count_shared;
          if (!old_count) {
            for (std::uint32_t index = threadIdx.x; index < source_count;
                 index += blockDim.x)
              current[index] = source[index];
            __syncthreads();
            if (threadIdx.x == 0u) current_count_shared = source_count;
            __syncthreads();
            continue;
          }
          const std::uint32_t merged_count = old_count + source_count;
          const std::uint32_t diagonal = min(threadIdx.x, merged_count);
          std::uint32_t low = diagonal > old_count ? diagonal - old_count : 0u;
          std::uint32_t high = min(diagonal, source_count);
          while (low < high) {
            const std::uint32_t source_index = (low + high) >> 1u;
            const std::uint32_t current_index = diagonal - source_index;
            if (source_index < source_count && current_index > 0u &&
                current[current_index - 1u].key >= source[source_index].key)
              low = source_index + 1u;
            else
              high = source_index;
          }
          if (threadIdx.x < merged_count) {
            const std::uint32_t source_index = low;
            const std::uint32_t current_index = diagonal - source_index;
            const bool choose_source = source_index < source_count &&
                (current_index >= old_count ||
                 source[source_index].key <= current[current_index].key);
            workspace.merged[threadIdx.x] = choose_source
                ? source[source_index] : current[current_index];
          }
          __syncthreads();
          const std::uint32_t index = threadIdx.x;
          const bool winner = index < merged_count &&
              (index + 1u == merged_count ||
               workspace.merged[index].key != workspace.merged[index + 1u].key);
          std::uint32_t destination{}, winner_count{};
          BlockScan(scan_storage).ExclusiveSum(
              std::uint32_t{winner}, destination, winner_count);
          if (winner) current[destination] = workspace.merged[index];
          __syncthreads();
          if (threadIdx.x == 0u) current_count_shared = winner_count;
          __syncthreads();
        }
      }
    }

    const Row *newer = use_hot
        ? hot_rows + hot_descriptor_shared.offset() : current;
    const std::uint32_t newer_count = use_hot
        ? hot_descriptor_shared.count() : current_count_shared;
    const RouteHeader foundation_header = foundation_level < active_levels
        ? route_headers[descriptor_index(q, foundation_level)] : RouteHeader{};

    for (std::uint32_t local = threadIdx.x; local < fragment_count;
         local += blockDim.x) {
      const SectionRangeFragment fragment = fragments[fragment_begin + local];
      RangeFragmentBounds bounds{};
      bounds.update_begin = lower_bound_rows(
          newer, newer_count, fragment.low_suffix);
      bounds.update_end = upper_bound_rows(
          newer, newer_count, fragment.high_suffix);
      std::uint32_t base_work = 0u;
      if (foundation_header.count == 1u) {
        const Descriptor descriptor = route_slices[foundation_header.begin].rows;
        const ResidentRows rows = arena + descriptor.offset();
        if (ranks_valid_shared) {
          const std::uint32_t low_cell =
              std::uint32_t{fragment.low_suffix} / kFoundationCellKeys;
          const std::uint32_t low_begin = foundation_cell_ranks[low_cell];
          const std::uint32_t low_end = foundation_cell_ranks[low_cell + 1u];
          bounds.base_begin = low_begin + lower_bound_rows(
              rows + low_begin, low_end - low_begin, fragment.low_suffix);
          const std::uint32_t high_cell =
              std::uint32_t{fragment.high_suffix} / kFoundationCellKeys;
          const std::uint32_t high_begin = foundation_cell_ranks[high_cell];
          const std::uint32_t high_end = foundation_cell_ranks[high_cell + 1u];
          bounds.base_end = high_begin + upper_bound_rows(
              rows + high_begin, high_end - high_begin, fragment.high_suffix);
        } else {
          bounds.base_begin = lower_bound_rows(
              rows, descriptor.count(), fragment.low_suffix);
          bounds.base_end = upper_bound_rows(
              rows, descriptor.count(), fragment.high_suffix);
        }
        base_work = bounds.base_end - bounds.base_begin;
      } else {
        base_work = route_range_row_count(
            fragment.low_suffix, fragment.high_suffix, arena,
            foundation_header, route_slices);
      }
      fragment_bounds[local] = bounds;
      fragment_work[local] = bounds.update_end - bounds.update_begin + base_work;
      fragment_sums[local] = Aggregate::identity();
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
      std::uint32_t wave = 0u, cursor = 0u;
      wave_fragment_begins[0u] = 0u;
      for (std::uint32_t fragment = 0u; fragment < fragment_count;
           ++fragment) {
        const std::uint32_t work = fragment_work[fragment];
        const std::uint32_t width = work <= kRangeThreadWork ? 1u
            : work <= kRangeSubgroupWork ? 8u : 32u;
        std::uint32_t slot = (cursor + width - 1u) & ~(width - 1u);
        if ((slot >> 5u) != ((slot + width - 1u) >> 5u))
          slot = (slot + 31u) & ~31u;
        if (slot + width > kSectionRangeThreads) {
          ++wave;
          wave_fragment_begins[wave] = fragment;
          slot = 0u;
        }
        fragment_slots[fragment] = static_cast<std::uint8_t>(slot);
        fragment_widths[fragment] = static_cast<std::uint8_t>(width);
        cursor = slot + width;
      }
      wave_count_shared = wave + 1u;
      wave_fragment_begins[wave_count_shared] = fragment_count;
    }
    __syncthreads();

    for (std::uint32_t local = threadIdx.x; local < fragment_count;
         local += blockDim.x) {
      source_begins[local] = fragment_bounds[local].update_begin;
      source_ends[local] = fragment_bounds[local].update_end;
    }
    __syncthreads();
    enumerate_range_source_tiles<Aggregate>(
        fragments, fragment_begin, fragment_count,
        RangeTileSource{newer, {}, true}, newer, newer_count, false,
        source_begins, source_ends, fragment_slots, fragment_widths,
        wave_fragment_begins, wave_count_shared, union_begins, union_ends,
        &union_count_shared, workspace.tile, fragment_sums);

    for (std::uint32_t route_index = 0u;
         route_index < foundation_header.count; ++route_index) {
      const RouteSlice route = route_slices[foundation_header.begin + route_index];
      const ResidentRows rows = arena + route.rows.offset();
      for (std::uint32_t local = threadIdx.x; local < fragment_count;
           local += blockDim.x) {
        const SectionRangeFragment fragment = fragments[fragment_begin + local];
        if (foundation_header.count == 1u) {
          source_begins[local] = fragment_bounds[local].base_begin;
          source_ends[local] = fragment_bounds[local].base_end;
        } else if (route.suffix_end > fragment.low_suffix &&
                   route.suffix_begin <= fragment.high_suffix) {
          source_begins[local] = lower_bound_rows(
              rows, route.rows.count(), fragment.low_suffix);
          source_ends[local] = upper_bound_rows(
              rows, route.rows.count(), fragment.high_suffix);
        } else {
          source_begins[local] = source_ends[local] = 0u;
        }
      }
      __syncthreads();
      enumerate_range_source_tiles<Aggregate>(
          fragments, fragment_begin, fragment_count,
          RangeTileSource{nullptr, rows, false}, newer, newer_count, true,
          source_begins, source_ends, fragment_slots, fragment_widths,
          wave_fragment_begins, wave_count_shared, union_begins, union_ends,
          &union_count_shared, workspace.tile, fragment_sums);
    }

    for (std::uint32_t local = threadIdx.x; local < fragment_count;
         local += blockDim.x) {
      const SectionRangeFragment fragment = fragments[fragment_begin + local];
      aggregate_partials[fragment.original_index] = fragment_sums[local];
    }
    __syncthreads();
  }
}

template <class Aggregate>
__global__ void warp_range_fragment_kernel(
    const RangeFragment *fragments, std::uint32_t fragment_count,
    const std::uint32_t *device_fragment_count,
    const std::uint32_t *query_low, const std::uint32_t *query_high,
    ResidentRows arena,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices, const Row *hot_rows,
    const Descriptor *hot_descriptors, bool hot_ready,
    const std::uint32_t *raw_keys,
    const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches,
    typename Aggregate::State *aggregate_partials,
    const std::uint64_t *query_occupied_level_mask) {
  constexpr std::uint32_t kWarps = 4u;
  constexpr std::uint32_t kUpdateCapacity = 128u;
  union WarpScratch {
    Row merged[kUpdateCapacity];
    TaggedRow tagged[kUpdateCapacity];
  };
  __shared__ Row current_shared[kWarps][kUpdateCapacity];
  __shared__ WarpScratch scratch[kWarps];
  const DeviceManifestSnapshot manifest =
      load_query_manifest(query_occupied_level_mask);
  const std::uint32_t active_levels = manifest.active_levels;
  const std::uint32_t foundation_level = manifest.foundation_level;
  const std::uint64_t occupied_levels = manifest.occupied_level_mask;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t fragment_index = blockIdx.x * kWarps + warp;
  if (fragment_index >= fragment_count ||
      fragment_index >= *device_fragment_count) return;
  constexpr unsigned full_mask = 0xffffffffu;
  std::uint32_t query = 0u, q = 0u;
  if (lane == 0u) {
    const RangeFragment fragment = fragments[fragment_index];
    query = fragment.query;
    q = fragment.quotient;
  }
  query = __shfl_sync(full_mask, query, 0u);
  q = __shfl_sync(full_mask, q, 0u);
  std::uint32_t query_begin = 0u, query_end = 0u;
  if (lane == 0u) {
    query_begin = query_low[query];
    query_end = query_high[query];
  }
  query_begin = __shfl_sync(full_mask, query_begin, 0u);
  query_end = __shfl_sync(full_mask, query_end, 0u);
  const std::uint32_t q_low = q << 16u;
  const std::uint32_t q_high = q_low | 0xffffu;
  const std::uint32_t low = max(query_begin, q_low);
  const std::uint32_t high = min(query_end, q_high);
  const std::uint32_t low_suffix = key_suffix(low);
  const std::uint32_t high_suffix = key_suffix(high);
  const unsigned before = lane ? ((1u << lane) - 1u) : 0u;
  Row *current = current_shared[warp];
  Row *merged = scratch[warp].merged;

  unsigned long long hot_bits = 0u;
  if (lane == 0u && hot_ready) hot_bits = hot_descriptors[q].bits;
  hot_bits = __shfl_sync(full_mask, hot_bits, 0u);
  const Descriptor hot_descriptor{hot_bits};
  if (hot_descriptor.count()) {
    const RouteHeader foundation_header = foundation_level < active_levels
        ? route_headers[descriptor_index(q, foundation_level)]
        : RouteHeader{};
    unsigned long long value = cooperative_sum_visible_route_runs<Aggregate>(
        low_suffix, high_suffix, hot_rows + hot_descriptor.offset(),
        hot_descriptor.count(), arena, foundation_header, route_slices,
        lane, 32u);
    for (std::uint32_t offset = 16u; offset; offset >>= 1u)
      value = Aggregate{}(value,
          __shfl_down_sync(full_mask, value, offset));
    if (lane == 0u) aggregate_partials[fragment_index] = value;
    return;
  }

  std::uint32_t current_count = 0u;
  if (pending_batches) {
    std::uint32_t pending_count = 0u;
    for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
      const std::size_t oi = std::size_t{batch} * (kQuotients + 1u) + q;
      const std::uint32_t begin = raw_offsets[oi];
      const std::uint32_t end = raw_offsets[oi + 1u];
      for (std::uint32_t chunk = begin; chunk < end; chunk += 32u) {
        const std::uint32_t position = chunk + lane;
        RawAssignment item{};
        const std::uint32_t record = batch * batch_stride + position;
        const std::uint32_t item_key =
            position < end ? raw_keys[record] : 0u;
        const bool valid = position < end &&
            key_suffix(item_key) >= low_suffix &&
            key_suffix(item_key) <= high_suffix;
        if (valid)
          item = load_raw_assignment(raw_keys, raw_payloads, record);
        const unsigned selected = __ballot_sync(full_mask, valid);
        const std::uint32_t destination =
            pending_count + __popc(selected & before);
        if (valid && destination < kUpdateCapacity)
          scratch[warp].tagged[destination] =
              {raw_row(item), raw_age(raw_position(item), batch_stride)};
        pending_count += __popc(selected);
      }
    }
    const bool pending_overflow = pending_count > kUpdateCapacity;
    if (__shfl_sync(full_mask, pending_overflow, 0u)) {
      asm volatile("trap;");
      return;
    }

    for (std::uint32_t index = lane; index < pending_count; index += 32u) {
      const TaggedRow item = scratch[warp].tagged[index];
      std::uint32_t rank = 0u;
      for (std::uint32_t other_index = 0u; other_index < pending_count;
           ++other_index) {
        const TaggedRow other = scratch[warp].tagged[other_index];
        rank += other.row.key < item.row.key ||
            (other.row.key == item.row.key &&
             (other.age < item.age ||
              (other.age == item.age && other_index < index)));
      }
      current[rank] = item.row;
    }
    __syncwarp();
    for (std::uint32_t group = 0u; group < 4u; ++group) {
      const std::uint32_t index = lane + group * 32u;
      const bool winner = index < pending_count &&
          (index + 1u == pending_count ||
           current[index].key != current[index + 1u].key);
      const unsigned winners = __ballot_sync(full_mask, winner);
      if (winner)
        merged[current_count + __popc(winners & before)] = current[index];
      current_count += __popc(winners);
    }
    __syncwarp();
    Row *temporary = current;
    current = merged;
    merged = temporary;
  }

  bool class_overflow = false;
  for (std::uint32_t level = 0u; level < active_levels; ++level) {
    if (level == foundation_level ||
        !level_is_occupied(occupied_levels, level)) continue;
    const RouteHeader source_header =
        route_headers[descriptor_index(q, level)];
    for (std::uint32_t source_route = 0u;
         source_route < source_header.count; ++source_route) {
    unsigned long long descriptor_bits = 0ull;
    if (lane == 0u)
      descriptor_bits =
          route_slices[source_header.begin + source_route].rows.bits;
    descriptor_bits = __shfl_sync(full_mask, descriptor_bits, 0u);
    const Descriptor descriptor{descriptor_bits};
    const ResidentRows rows = arena + descriptor.offset();
    std::uint32_t older_begin = 0u, older_end = 0u;
    if (lane == 0u && descriptor.count()) {
      older_begin = low == q_low
          ? 0u
          : lower_bound_rows(rows, descriptor.count(), low_suffix);
      older_end = high == q_high
          ? descriptor.count()
          : upper_bound_rows(rows, descriptor.count(), high_suffix);
    }
    older_begin = __shfl_sync(full_mask, older_begin, 0u);
    older_end = __shfl_sync(full_mask, older_end, 0u);
    const std::uint32_t older_count = older_end - older_begin;
    if (!older_count) continue;
    if (current_count + older_count > kUpdateCapacity) {
      class_overflow = true;
      break;
    }
    const ResidentRows older = rows + older_begin;
    if (!current_count) {
      for (std::uint32_t index = lane; index < older_count; index += 32u)
        current[index] = older[index];
      current_count = older_count;
      __syncwarp();
      continue;
    }
    for (std::uint32_t index = lane; index < older_count; index += 32u) {
      const Row row = older[index];
      std::uint32_t lo = 0u, hi = current_count;
      while (lo < hi) {
        const std::uint32_t mid = (lo + hi) >> 1u;
        if (current[mid].key < row.key) lo = mid + 1u;
        else hi = mid;
      }
      merged[index + lo] = row;
    }
    for (std::uint32_t index = lane; index < current_count; index += 32u) {
      const Row row = current[index];
      std::uint32_t lo = 0u, hi = older_count;
      while (lo < hi) {
        const std::uint32_t mid = (lo + hi) >> 1u;
        if (older[mid].key <= row.key) lo = mid + 1u;
        else hi = mid;
      }
      merged[index + lo] = row;
    }
    __syncwarp();
    const std::uint32_t merged_count = current_count + older_count;
    std::uint32_t output_count = 0u;
    for (std::uint32_t group = 0u; group < 4u; ++group) {
      const std::uint32_t index = lane + group * 32u;
      const bool winner = index < merged_count &&
          (index + 1u == merged_count ||
           merged[index].key != merged[index + 1u].key);
      const unsigned winners = __ballot_sync(full_mask, winner);
      if (winner)
        current[output_count + __popc(winners & before)] = merged[index];
      output_count += __popc(winners);
    }
    current_count = output_count;
    __syncwarp();
    }
    if (class_overflow) break;
  }
  class_overflow = __shfl_sync(full_mask, class_overflow, 0u);
  if (class_overflow) {
    asm volatile("trap;");
    return;
  }

  const RouteHeader foundation_header = foundation_level < active_levels
      ? route_headers[descriptor_index(q, foundation_level)]
      : RouteHeader{};
  std::uint32_t work = 0u;
  if (lane == 0u)
    work = current_count + route_range_row_count(
        low_suffix, high_suffix, arena, foundation_header, route_slices);
  work = __shfl_sync(full_mask, work, 0u);
  const std::uint32_t worker_width = work <= kRangeThreadWork ? 1u
      : work <= kRangeSubgroupWork ? 8u : 32u;
  const unsigned worker_mask = __ballot_sync(full_mask, lane < worker_width);
  typename Aggregate::State local = Aggregate::identity();
  if (lane < worker_width) {
    local = cooperative_sum_visible_route_runs<Aggregate>(
        low_suffix, high_suffix, current, current_count, arena,
        foundation_header, route_slices, lane, worker_width);
    for (std::uint32_t offset = worker_width / 2u; offset; offset >>= 1u)
      local += __shfl_down_sync(worker_mask, local, offset, worker_width);
  }
  if (lane == 0u) {
    aggregate_partials[fragment_index] = local;
  }
}

__device__ __forceinline__ std::uint64_t pending_signature_bits(
    std::uint32_t key) {
  const std::uint32_t first = key * 0x9e3779b1u;
  const std::uint32_t second = (key ^ (key >> 16u)) * 0x85ebca6bu;
  return (1ull << (first >> 26u)) | (1ull << (second >> 26u));
}

__global__ void count_admission_quotients_kernel(
    const std::uint32_t *keys, std::uint32_t count,
    std::uint32_t *quotient_counts,
    std::uint32_t *reservation_ranks) {
  constexpr std::uint32_t kEmpty = 0xffffffffu;
  __shared__ std::uint32_t local_keys[kAdmissionCtaHashSlots];
  __shared__ std::uint32_t local_counts[kAdmissionCtaHashSlots];
  __shared__ std::uint32_t global_bases[kAdmissionCtaHashSlots];
  __shared__ std::uint32_t warp_group_counts[kThreads / 32u];

  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = i < count;
  const std::uint32_t key = valid ? keys[i] : 0u;
  const std::uint32_t quotient = key >> 16u;
  const unsigned active = __ballot_sync(0xffffffffu, valid);
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  unsigned peers = 0u;
  std::uint32_t leader = 0u;
  bool group_leader = false;
  if (valid) {
    peers = __match_any_sync(active, quotient);
    leader = __ffs(peers) - 1u;
    group_leader = lane == leader;
  }
  const unsigned leaders = __ballot_sync(0xffffffffu, group_leader);
  if (lane == 0u) warp_group_counts[warp] = __popc(leaders);
  __syncthreads();
  std::uint32_t cta_group_count = 0u;
#pragma unroll
  for (std::uint32_t w = 0u; w < kThreads / 32u; ++w)
    cta_group_count += warp_group_counts[w];

  // Use CTA aggregation only for skewed input.
  if (cta_group_count > kAdmissionCtaGroupMaximum) {
    if (!valid) return;
    std::uint32_t base = 0u;
    if (group_leader)
      base = atomicAdd(quotient_counts + quotient, __popc(peers));
    base = __shfl_sync(peers, base, leader);
    const unsigned before = lane ? ((1u << lane) - 1u) : 0u;
    reservation_ranks[i] = base + __popc(peers & before);
    return;
  }

  for (std::uint32_t slot = threadIdx.x;
       slot < kAdmissionCtaHashSlots; slot += blockDim.x) {
    local_keys[slot] = kEmpty;
    local_counts[slot] = 0u;
  }
  __syncthreads();

  std::uint32_t local_slot = 0u;
  std::uint32_t local_rank = 0u;
  if (valid) {
    std::uint32_t group_base = 0u;
    if (group_leader) {
      local_slot = (quotient * 0x9e3779b1u) &
          (kAdmissionCtaHashSlots - 1u);
      while (true) {
        const std::uint32_t found = atomicCAS(
            local_keys + local_slot, kEmpty, quotient);
        if (found == kEmpty || found == quotient) break;
        local_slot = (local_slot + 1u) &
            (kAdmissionCtaHashSlots - 1u);
      }
      group_base = atomicAdd(local_counts + local_slot, __popc(peers));
    }
    local_slot = __shfl_sync(peers, local_slot, leader);
    group_base = __shfl_sync(peers, group_base, leader);
    const unsigned before = lane ? ((1u << lane) - 1u) : 0u;
    local_rank = group_base + __popc(peers & before);
  }
  __syncthreads();

  for (std::uint32_t slot = threadIdx.x;
       slot < kAdmissionCtaHashSlots; slot += blockDim.x) {
    const std::uint32_t local_quotient = local_keys[slot];
    if (local_quotient == kEmpty) continue;
    global_bases[slot] = atomicAdd(
        quotient_counts + local_quotient, local_counts[slot]);
  }
  __syncthreads();

  if (valid)
    reservation_ranks[i] = global_bases[local_slot] + local_rank;
}

// Lookup compilation mirrors the admission counter's aggregation, but also
// records the quotients that actually occur.  Subsequent TQRJ work is bounded
// by queries/active tasks rather than a launch over the quotient domain.
__global__ void count_lookup_quotients_kernel(
    const std::uint32_t *keys, std::uint32_t count,
    std::uint32_t *quotient_counts,
    std::uint32_t *reservation_ranks,
    std::uint32_t *active_quotients,
    std::uint32_t *active_count) {
  constexpr std::uint32_t kEmpty = 0xffffffffu;
  __shared__ std::uint32_t local_keys[kAdmissionCtaHashSlots];
  __shared__ std::uint32_t local_counts[kAdmissionCtaHashSlots];
  __shared__ std::uint32_t global_bases[kAdmissionCtaHashSlots];
  __shared__ std::uint32_t warp_group_counts[kThreads / 32u];

  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = i < count;
  const std::uint32_t key = valid ? keys[i] : 0u;
  const std::uint32_t quotient = key >> 16u;
  const unsigned active = __ballot_sync(0xffffffffu, valid);
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  unsigned peers = 0u;
  std::uint32_t leader = 0u;
  bool group_leader = false;
  if (valid) {
    peers = __match_any_sync(active, quotient);
    leader = __ffs(peers) - 1u;
    group_leader = lane == leader;
  }
  const unsigned leaders = __ballot_sync(0xffffffffu, group_leader);
  if (lane == 0u) warp_group_counts[warp] = __popc(leaders);
  __syncthreads();
  std::uint32_t cta_group_count = 0u;
#pragma unroll
  for (std::uint32_t w = 0u; w < kThreads / 32u; ++w)
    cta_group_count += warp_group_counts[w];

  if (cta_group_count > kAdmissionCtaGroupMaximum) {
    if (!valid) return;
    std::uint32_t base = 0u;
    if (group_leader) {
      base = atomicAdd(quotient_counts + quotient, __popc(peers));
      if (base == 0u) {
        const std::uint32_t ticket = atomicAdd(active_count, 1u);
        active_quotients[ticket] = quotient;
      }
    }
    base = __shfl_sync(peers, base, leader);
    const unsigned before = lane ? ((1u << lane) - 1u) : 0u;
    reservation_ranks[i] = base + __popc(peers & before);
    return;
  }

  for (std::uint32_t slot = threadIdx.x;
       slot < kAdmissionCtaHashSlots; slot += blockDim.x) {
    local_keys[slot] = kEmpty;
    local_counts[slot] = 0u;
  }
  __syncthreads();

  std::uint32_t local_slot = 0u;
  std::uint32_t local_rank = 0u;
  if (valid) {
    std::uint32_t group_base = 0u;
    if (group_leader) {
      local_slot = (quotient * 0x9e3779b1u) &
          (kAdmissionCtaHashSlots - 1u);
      while (true) {
        const std::uint32_t found = atomicCAS(
            local_keys + local_slot, kEmpty, quotient);
        if (found == kEmpty || found == quotient) break;
        local_slot = (local_slot + 1u) &
            (kAdmissionCtaHashSlots - 1u);
      }
      group_base = atomicAdd(local_counts + local_slot, __popc(peers));
    }
    local_slot = __shfl_sync(peers, local_slot, leader);
    group_base = __shfl_sync(peers, group_base, leader);
    const unsigned before = lane ? ((1u << lane) - 1u) : 0u;
    local_rank = group_base + __popc(peers & before);
  }
  __syncthreads();

  for (std::uint32_t slot = threadIdx.x;
       slot < kAdmissionCtaHashSlots; slot += blockDim.x) {
    const std::uint32_t local_quotient = local_keys[slot];
    if (local_quotient == kEmpty) continue;
    const std::uint32_t base = atomicAdd(
        quotient_counts + local_quotient, local_counts[slot]);
    global_bases[slot] = base;
    if (base == 0u) {
      const std::uint32_t ticket = atomicAdd(active_count, 1u);
      active_quotients[ticket] = local_quotient;
    }
  }
  __syncthreads();

  if (valid)
    reservation_ranks[i] = global_bases[local_slot] + local_rank;
}

__global__ void materialize_lookup_active_counts_kernel(
    const std::uint32_t *active_quotients,
    const std::uint32_t *active_count,
    const std::uint32_t *quotient_counts,
    std::uint32_t active_capacity,
    std::uint32_t *active_counts) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > active_capacity) return;
  const std::uint32_t count = *active_count;
  active_counts[i] = i < count
      ? quotient_counts[active_quotients[i]] : 0u;
}

__global__ void publish_lookup_active_bases_kernel(
    const std::uint32_t *active_quotients,
    const std::uint32_t *active_count,
    const std::uint32_t *active_offsets,
    std::uint32_t active_capacity,
    std::uint32_t *query_bases) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count = *active_count;
  if (i >= active_capacity || i >= count) return;
  const std::uint32_t q = active_quotients[i];
  query_bases[q] = active_offsets[i];
}

__global__ void reset_lookup_quotient_counts_kernel(
    const std::uint32_t *active_quotients,
    const std::uint32_t *active_count,
    std::uint32_t active_capacity,
    std::uint32_t *quotient_counts) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count = *active_count;
  if (i >= active_capacity || i >= count) return;
  quotient_counts[active_quotients[i]] = 0u;
}

__global__ void build_admission_signatures_kernel(
    const std::uint32_t *section_grouped_keys, std::uint32_t count,
    std::uint64_t *batch_signatures) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t key = section_grouped_keys[i];
  const std::uint32_t quotient = key >> 16u;
  const unsigned active = __activemask();
  const unsigned peers = __match_any_sync(active, quotient);
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t leader = __ffs(peers) - 1u;
  const std::uint64_t bits = pending_signature_bits(key);
  const std::uint32_t low = __reduce_or_sync(
      peers, static_cast<std::uint32_t>(bits));
  const std::uint32_t high = __reduce_or_sync(
      peers, static_cast<std::uint32_t>(bits >> 32u));
  if (lane == leader) {
    const unsigned long long aggregate =
        static_cast<unsigned long long>(low) |
        (static_cast<unsigned long long>(high) << 32u);
    atomicOr(reinterpret_cast<unsigned long long *>(
                 batch_signatures + quotient), aggregate);
  }
}

__global__ void commit_admission_metadata_kernel(
    std::uint32_t *counts, const std::uint64_t *batch_signatures,
    std::uint64_t *epoch_signatures) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  const std::uint64_t signature = batch_signatures[q];
  if (signature)
    atomicOr(reinterpret_cast<unsigned long long *>(epoch_signatures + q),
             static_cast<unsigned long long>(signature));
  counts[q] = 0u;
}

// Rebuild signatures after failed publication.
__global__ void rebuild_epoch_signatures_kernel(
    const std::uint64_t *batch_signatures,
    std::uint64_t *epoch_signatures) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  std::uint64_t aggregate = 0u;
  for (std::uint32_t batch = 0u; batch < kBatchesPerEpoch; ++batch)
    aggregate |= batch_signatures[
        std::size_t{batch} * kQuotients + q];
  epoch_signatures[q] = aggregate;
}

template <bool Tombstone>
__global__ void scatter_admission_records_kernel(
    const std::uint32_t *keys, const std::uint32_t *values,
    std::uint32_t count, std::uint32_t batch_slot,
    const std::uint32_t *offsets, const std::uint32_t *reservation_ranks,
    std::uint32_t *destination_keys, RawPayload *destination_payloads) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  // Expose all coalesced input loads before the quotient-dependent lookup.
  const std::uint32_t key = keys[i];
  const std::uint32_t rank = reservation_ranks[i];
  std::uint32_t value = 0u;
  if constexpr (!Tombstone) value = values[i];
  const std::uint32_t quotient = key >> 16u;
  const std::uint32_t output = offsets[quotient] + rank;
  destination_keys[output] = key;
  const std::uint32_t metadata =
      (batch_slot << kBatchPositionBits) | i |
      (Tombstone ? kRawTombstone : 0u);
  store_raw_payload(destination_payloads, output, value, metadata);
}

__global__ void scatter_query_records_kernel(
    const std::uint32_t *queries, std::uint32_t count,
    const std::uint32_t *offsets,
    const std::uint32_t *reservation_ranks,
    std::uint32_t *grouped_queries, std::uint32_t *original_ids) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t key = queries[i];
  const std::uint32_t q = key >> 16u;
  const std::uint32_t output = offsets[q] + reservation_ranks[i];
  grouped_queries[output] = key;
  original_ids[output] = i;
}

__global__ void build_query_quotient_offsets_kernel(
    const std::uint32_t *queries, std::uint32_t count,
    std::uint32_t *offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  if (q == kQuotients) {
    offsets[q] = count;
    return;
  }
  const std::uint32_t target = q << 16u;
  std::uint32_t low = 0u, high = count;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (queries[middle] < target) low = middle + 1u;
    else high = middle;
  }
  offsets[q] = low;
}

__device__ __forceinline__ bool bulk_root_sink_terminal(
    const std::uint32_t *keys, std::uint32_t count,
    std::uint64_t position, std::uint32_t key) {
  return position + 1u == count || key != keys[position + 1u];
}

// The immutable root is assigned in deterministic, contiguous tiles.  The
// terminal row of each equal-key run is the same winner selected by the old
// stable-sort + DeviceSelect pipeline, but only one count per tile survives.
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

// Repeat the terminal predicate after the small tile-count scan and deposit
// each winner directly into its final resident slot.  No selected-ID stream,
// publication-key stream, publication-row stream, or resident copy exists.
__global__ void deposit_bulk_root_sink_kernel(
    const std::uint32_t *sorted_keys,
    const std::uint32_t *sorted_values,
    std::uint32_t count,
    const std::uint32_t *tile_offsets,
    ResidentRows arena,
    std::uint64_t destination) {
  using BlockScan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  const std::uint64_t tile_begin =
      std::uint64_t{blockIdx.x} * kBulkRootSinkTileRows;
  const std::uint64_t thread_begin = tile_begin +
      std::uint64_t{threadIdx.x} * kBulkRootSinkItemsPerThread;
  std::uint32_t keys[kBulkRootSinkItemsPerThread];
  bool terminal[kBulkRootSinkItemsPerThread];
  std::uint32_t local_count = 0u;
  #pragma unroll
  for (std::uint32_t item = 0u;
       item < kBulkRootSinkItemsPerThread; ++item) {
    const std::uint64_t position = thread_begin + item;
    if (position < count) {
      keys[item] = sorted_keys[position];
      terminal[item] = bulk_root_sink_terminal(
          sorted_keys, count, position, keys[item]);
      local_count += terminal[item];
    } else {
      keys[item] = 0u;
      terminal[item] = false;
    }
  }
  std::uint32_t thread_prefix = 0u, unused_total = 0u;
  BlockScan(scan_storage).ExclusiveSum(
      local_count, thread_prefix, unused_total);
  std::uint32_t local_rank = 0u;
  #pragma unroll
  for (std::uint32_t item = 0u;
       item < kBulkRootSinkItemsPerThread; ++item) {
    if (!terminal[item]) continue;
    const std::uint64_t position = thread_begin + item;
    const std::uint64_t output = destination +
        tile_offsets[blockIdx.x] + thread_prefix + local_rank++;
    arena.store(output, make_row(
        keys[item], sorted_values[position], 0u));
  }
}

// Map each quotient boundary from raw sorted position to compacted resident
// rank.  Only the at-most-one partial tile before a boundary is inspected;
// completed tiles are represented by their scanned survivor counts.
__global__ void build_bulk_root_sink_quotient_offsets_kernel(
    const std::uint32_t *sorted_keys,
    std::uint32_t count,
    const std::uint32_t *tile_offsets,
    std::uint32_t tile_count,
    std::uint32_t *quotient_offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  if (q == kQuotients) {
    quotient_offsets[q] = tile_offsets[tile_count];
    return;
  }

  const std::uint32_t target = q << 16u;
  std::uint32_t low = 0u, high = count;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (sorted_keys[middle] < target) low = middle + 1u;
    else high = middle;
  }
  const std::uint32_t position = low;
  const std::uint32_t tile = position / kBulkRootSinkTileRows;
  std::uint32_t survivor_rank = tile_offsets[tile];
  const std::uint32_t tile_begin = tile * kBulkRootSinkTileRows;
  for (std::uint32_t i = tile_begin; i < position; ++i) {
    const std::uint32_t key = sorted_keys[i];
    survivor_rank += bulk_root_sink_terminal(
        sorted_keys, count, i, key);
  }
  quotient_offsets[q] = survivor_rank;
}

__device__ __forceinline__ std::uint64_t pack_canonical_epoch_job(
    std::uint32_t q, std::uint32_t count) {
  return (static_cast<std::uint64_t>(count) << 32u) | q;
}

__global__ void count_canonical_epoch_jobs_kernel(
    const std::uint32_t *raw_offsets, std::uint32_t pending_batches,
    std::uint32_t *section_counts, std::uint64_t *epoch_jobs,
    std::uint32_t *local_job_count,
    std::uint32_t *oversized_job_count) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  if (q == kQuotients) {
    section_counts[q] = 0u;
    return;
  }
  std::uint32_t count = 0u;
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t base =
        std::size_t{batch} * (kQuotients + 1u) + q;
    count += raw_offsets[base + 1u] - raw_offsets[base];
  }
  section_counts[q] = count;
  if (!count) return;
  const std::uint64_t job = pack_canonical_epoch_job(q, count);
  if (count <= kLocalEpochCapacity) {
    const std::uint32_t position = atomicAdd(local_job_count, 1u);
    epoch_jobs[position] = job;
  } else {
    const std::uint32_t position = atomicAdd(oversized_job_count, 1u);
    epoch_jobs[kQuotients - 1u - position] = job;
  }
}

__global__ void sum_canonical_section_counts_kernel(
    const std::uint32_t *section_counts, std::uint32_t *total_count) {
  __shared__ std::uint32_t partials[kThreads];
  std::uint32_t total = 0u;
  for (std::uint32_t q = threadIdx.x; q < kQuotients;
       q += blockDim.x)
    total += section_counts[q];
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
    std::uint32_t, kFoundationCompactionThreads, Items, std::uint32_t>;

using ActiveEpochBlockScan =
    cub::BlockScan<std::uint32_t, kFoundationCompactionThreads>;

union ActiveEpochResolutionStorage {
  typename ActiveEpochBlockSort<1u>::TempStorage sort_1;
  typename ActiveEpochBlockSort<2u>::TempStorage sort_2;
  typename ActiveEpochBlockSort<3u>::TempStorage sort_3;
  typename ActiveEpochBlockSort<4u>::TempStorage sort_4;
  typename ActiveEpochBlockSort<5u>::TempStorage sort_5;
  unsigned long long winners[kLocalEpochCapacity];
};

template <std::uint32_t Items, bool ResidentOutput>
__device__ __forceinline__ void resolve_canonical_epoch_active_job(
    std::uint32_t q, std::uint32_t raw_count,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const std::uint32_t *raw_section_offsets,
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
        std::size_t{batch} * (kQuotients + 1u) + q;
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

  const std::uint32_t output_begin = raw_section_offsets[q];
  for (std::uint32_t group = threadIdx.x;
       group < output_count_shared; group += blockDim.x) {
    const std::uint32_t source = static_cast<std::uint32_t>(
        resolution_storage.winners[group]);
    const std::uint32_t suffix = sorted_suffixes[group];
    const Row row = raw_row(full_key(q, suffix), raw_payloads[source]);
    store_canonical_epoch_row<ResidentOutput>(
        output_rows, resident_rows, resident_begin,
        output_begin + group, row);
    atomicAdd(cell_counts + suffix / kFoundationCellKeys, 1u);
  }
  __syncthreads();

  const std::uint32_t cell_count = threadIdx.x < kFoundationCells
      ? cell_counts[threadIdx.x] : 0u;
  std::uint32_t cell_prefix = 0u, ignored = 0u;
  ActiveEpochBlockScan(scan_storage).ExclusiveSum(
      cell_count, cell_prefix, ignored);
  if (threadIdx.x < kFoundationCells)
    cell_ranks[std::size_t{q} * kFoundationCells + threadIdx.x] =
        static_cast<std::uint16_t>(cell_prefix);
}

template <bool ResidentOutput>
__global__ __launch_bounds__(kFoundationCompactionThreads, 5)
void resolve_canonical_epoch_active_jobs_kernel(
    const std::uint64_t *epoch_jobs,
    const std::uint32_t *local_job_count, std::uint32_t *next_job,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const std::uint32_t *raw_section_offsets,
    Row *output_rows, ResidentRows resident_rows,
    std::uint64_t resident_begin, std::uint32_t *resolved_counts,
    std::uint16_t *cell_ranks) {
  using BlockScan = ActiveEpochBlockScan;
  using ResolutionStorage = ActiveEpochResolutionStorage;
  __shared__ ResolutionStorage resolution_storage;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::uint32_t batch_prefix[kBatchesPerEpoch + 1u];
  __shared__ std::uint16_t sorted_suffixes[kLocalEpochCapacity];
  __shared__ std::uint32_t cell_counts[kFoundationCells];
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
            std::size_t{batch} * (kQuotients + 1u) + q;
        total += raw_offsets[base + 1u] - raw_offsets[base];
        batch_prefix[batch + 1u] = total;
      }
      for (std::uint32_t batch = pending_batches;
           batch < kBatchesPerEpoch; ++batch)
        batch_prefix[batch + 1u] = total;
      output_count_shared = 0u;
    }
    if (threadIdx.x < kFoundationCells)
      cell_counts[threadIdx.x] = 0u;
    __syncthreads();

#define GPULSMOPT_RESOLVE_ACTIVE_EPOCH_JOB(Items)                         \
    resolve_canonical_epoch_active_job<Items, ResidentOutput>(            \
        q, raw_count, raw_keys, raw_payloads, raw_offsets, batch_stride,  \
        pending_batches, raw_section_offsets, output_rows, resident_rows, \
        resident_begin, resolved_counts, cell_ranks, resolution_storage,  \
        scan_storage, batch_prefix, sorted_suffixes, cell_counts,          \
        output_count_shared)
    if (raw_count <= kFoundationCompactionThreads)
      GPULSMOPT_RESOLVE_ACTIVE_EPOCH_JOB(1u);
    else if (raw_count <= kFoundationCompactionThreads * 2u)
      GPULSMOPT_RESOLVE_ACTIVE_EPOCH_JOB(2u);
    else if (raw_count <= kFoundationCompactionThreads * 3u)
      GPULSMOPT_RESOLVE_ACTIVE_EPOCH_JOB(3u);
    else if (raw_count <= kFoundationCompactionThreads * 4u)
      GPULSMOPT_RESOLVE_ACTIVE_EPOCH_JOB(4u);
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
    std::uint32_t pending_batches, const std::uint32_t *raw_section_offsets,
    Row *output_rows, ResidentRows resident_rows,
    std::uint64_t resident_begin, std::uint32_t *resolved_counts,
    std::uint16_t *cell_ranks,
    unsigned long long *oversized_winner_workspace,
    std::uint32_t *oversized_workspace_locks,
    std::uint32_t oversized_workspace_slots) {
  using BlockScan = cub::BlockScan<
      std::uint32_t, kFoundationCompactionThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::uint32_t q_shared;
  __shared__ std::uint32_t output_count_shared;
  for (;;) {
    if (threadIdx.x == 0u) {
      const std::uint32_t ticket = atomicAdd(next_overflow, 1u);
      q_shared = ticket < *overflow_count
          ? static_cast<std::uint32_t>(
                overflow_jobs[kQuotients - 1u - ticket])
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
          std::size_t{batch} * (kQuotients + 1u) + q;
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
    if (threadIdx.x == 0u) {
      output_count_shared = output_count;
      resolved_counts[q] = output_count;
    }
    __syncthreads();

    const std::uint32_t output_begin = raw_section_offsets[q];
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
    if (threadIdx.x < kFoundationCells) {
      const std::uint32_t cell_begin =
          threadIdx.x * kFoundationCellKeys;
      for (std::uint32_t suffix = cell_begin;
           suffix < cell_begin + kFoundationCellKeys; ++suffix)
        cell_count += winners[suffix] != 0ull;
    }
    std::uint32_t cell_prefix = 0u, ignored = 0u;
    BlockScan(scan_storage).ExclusiveSum(
        cell_count, cell_prefix, ignored);
    if (threadIdx.x < kFoundationCells)
      cell_ranks[std::size_t{q} * kFoundationCells + threadIdx.x] =
          static_cast<std::uint16_t>(cell_prefix);
    __syncthreads();
    if (threadIdx.x == 0u)
      atomicExch(oversized_workspace_locks + workspace_slot, 0u);
    __syncthreads();
  }
}

// Count grouped raw intervals without suffix sorting.
// GPU-resident publication.

__device__ __forceinline__ void emit_resident_job(
    BalancedMergeJob *jobs, std::uint64_t *job_raw_reservations,
    std::uint32_t global_index,
    std::uint64_t key_begin, std::uint64_t key_end,
    std::uint32_t q_begin, std::uint32_t q_end,
    std::uint64_t raw_count) {
  BalancedMergeJob job{};
  job.key_begin = key_begin;
  job.key_end = key_end;
  job.quotient_begin = q_begin;
  job.quotient_end = q_end;
  jobs[global_index] = job;
  job_raw_reservations[global_index] = raw_count;
}

__global__ void publish_resident_manifest_kernel(
    ResidentPublicationPlan *plan, DeviceManifest *manifests,
    std::uint32_t *active_manifest,
    std::uint64_t *query_occupied_level_mask) {
  if (blockIdx.x || plan->status) return;
  const DeviceManifest *current = manifests + plan->active_manifest;
  DeviceManifest *next = manifests + plan->inactive_manifest;
  const std::uint32_t destination = plan->destination_level;
  const std::uint64_t consumed = destination == 64u
      ? ~std::uint64_t{0}
      : ((std::uint64_t{1} << destination) - 1u);
  for (std::uint32_t level = threadIdx.x;
       level < kMaximumLevels; level += blockDim.x) {
    if (level == destination) continue;
    next->levels[level] = level < destination
        ? DeviceLevelState{} : current->levels[level];
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    next->occupied_level_mask = current->occupied_level_mask & ~consumed;
    DeviceLevelState state{};
    state.storage_generation = plan->output_generation;
    next->levels[destination] = state;
    if (plan->survivor_count)
      next->occupied_level_mask |= std::uint64_t{1} << destination;
    else
      next->occupied_level_mask &= ~(std::uint64_t{1} << destination);
    next->active_levels = next->occupied_level_mask
        ? 64u - static_cast<std::uint32_t>(
                    __clzll(next->occupied_level_mask)) : 0u;
    next->foundation_level = next->active_levels
        ? next->active_levels - 1u : kMaximumLevels;
    next->generation = current->generation + 1u;
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    __threadfence();
    atomicExch(active_manifest, plan->inactive_manifest);
    atomicExch(reinterpret_cast<unsigned long long *>(
                   query_occupied_level_mask),
               static_cast<unsigned long long>(next->occupied_level_mask));
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
    manifest.foundation_level = level;
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

// Canonical radix-4 quotient-run carry.  Three immutable quotient-major runs
// may coexist in each regular tier.  Slots fill from high id to low id so the
// existing low-to-high query traversal still observes newest data first.
// The fourth run carries into the next tier.  The persistent directory is a
// quotient prefix plus 128 exact cell starts per physical run.

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
      // The next lower id is newer than every occupied sibling.
      natural_destination = tier_begin + slots - 1u - filled;
      destination_tier_begin = tier_begin;
      break;
    }
  }
  // A partial final tier cannot necessarily hold every live row in one run.
  // Its optional full-capacity terminal slot is used only after all regular
  // slots fill, and is mapped lazily by the host before this graph launches.
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
  const bool destination_is_foundation = valid_rollover ||
      (destination < kMaximumLevels &&
      (manifest->foundation_level == kMaximumLevels ||
       destination > manifest->foundation_level));
  next.keep_tombstones = !destination_is_foundation;
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
  if (q >= kQuotients || plan->status) return;
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
       local_q < kPlanningTileQuotients; ++local_q) {
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
        (proposed_span > kCanonicalJobQuotients ||
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
  __shared__ std::uint64_t tile_counts[kPlanningTileQuotients];
  __shared__ std::uint32_t capacities[kCanonicalJobQuotients + 1u];
  const std::uint32_t tile = blockIdx.x;
  if (tile >= kPlanningTiles) return;
  const std::uint32_t first = tile * kPlanningTileQuotients;
  tile_counts[threadIdx.x] = counts[first + threadIdx.x];
  tile_counts[threadIdx.x + blockDim.x] =
      counts[first + threadIdx.x + blockDim.x];
  if (threadIdx.x <= kCanonicalJobQuotients)
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
  __shared__ std::uint64_t tile_counts[kPlanningTileQuotients];
  __shared__ std::uint32_t capacities[kCanonicalJobQuotients + 1u];
  const std::uint32_t tile = blockIdx.x;
  if (tile >= kPlanningTiles) return;
  const std::uint32_t first = tile * kPlanningTileQuotients;
  tile_counts[threadIdx.x] = counts[first + threadIdx.x];
  tile_counts[threadIdx.x + blockDim.x] =
      counts[first + threadIdx.x + blockDim.x];
  if (threadIdx.x <= kCanonicalJobQuotients)
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
  // The raw counts are no longer needed after the tile snapshot.  Reuse that
  // buffer as a sparse quotient-to-hot-job directory so boundary discovery
  // visits each oversized quotient once instead of rediscovering it per job.
  for (std::uint32_t local_q = 0u;
       local_q < kPlanningTileQuotients; ++local_q) {
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
        (proposed_span > kCanonicalJobQuotients ||
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
    total = !count ? 0u : cell < kFoundationCells
        ? epoch_cell_ranks[
              std::size_t{q} * kFoundationCells + cell]
        : count;
  }
  const std::uint32_t source_end = plan->source_count > 1u
      ? min(kMaximumLevels, plan->source_level_limit + 1u) : 0u;
  for (std::uint32_t level = lane;
       level < source_end; level += 32u) {
    if (!level_is_occupied(manifest.occupied_level_mask, level)) continue;
    const Descriptor rows = descriptors[descriptor_index(q, level)];
    total += cell < kFoundationCells
        ? cell_ranks[
              std::size_t{level} * kLocalRankEntries +
              std::size_t{q} * kFoundationCells + cell]
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
      const std::uint32_t section_begin = epoch_offsets[q];
      const std::uint16_t *ranks = epoch_cell_ranks +
          std::size_t{q} * kFoundationCells;
      const std::uint32_t begin = ranks[cell];
      const std::uint32_t end = cell + 1u < kFoundationCells
          ? ranks[cell + 1u] : count;
      total = begin + lower_bound_rows(
          epoch_rows + section_begin + begin, end - begin, suffix);
    }
  }
  const std::uint32_t source_end = plan->source_count > 1u
      ? min(kMaximumLevels, plan->source_level_limit + 1u) : 0u;
  for (std::uint32_t level = lane;
       level < source_end; level += 32u) {
    if (!level_is_occupied(manifest.occupied_level_mask, level)) continue;
    const Descriptor rows = descriptors[descriptor_index(q, level)];
    const std::uint16_t *ranks = cell_ranks +
        std::size_t{level} * kLocalRankEntries +
        std::size_t{q} * kFoundationCells;
    const std::uint32_t begin = ranks[cell];
    const std::uint32_t end = cell + 1u < kFoundationCells
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
  std::uint32_t low_cell = 1u, high_cell = kFoundationCells;
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
    return {begin_cell * kFoundationCellKeys, begin_prefix};
  std::uint32_t low = begin_cell * kFoundationCellKeys + 1u;
  std::uint32_t high = end_cell * kFoundationCellKeys;
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
  // One warp owns all pieces of an oversized quotient.  It computes the
  // total once and carries each exact boundary prefix into the next piece.
  for (std::uint32_t q = blockIdx.x;
       q < kQuotients && !plan->status; q += gridDim.x) {
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

__device__ __forceinline__ std::uint32_t canonical_merge_partition(
    const std::uint32_t *left, std::uint32_t left_count,
    const std::uint32_t *right, std::uint32_t right_count,
    std::uint32_t diagonal) {
  std::uint32_t low = diagonal > right_count
      ? diagonal - right_count : 0u;
  std::uint32_t high = min(diagonal, left_count);
  while (low <= high) {
    const std::uint32_t li = (low + high) >> 1u;
    const std::uint32_t ri = diagonal - li;
    if (li && ri < right_count && right[ri] < left[li - 1u]) {
      high = li - 1u;
    } else if (ri && li < left_count && left[li] < right[ri - 1u]) {
      low = li + 1u;
    } else {
      return li;
    }
  }
  return low;
}

__device__ __forceinline__ void canonical_merge_interval(
    const std::uint32_t *left, std::uint32_t left_count,
    const std::uint32_t *right, std::uint32_t right_count,
    std::uint32_t *output, std::uint32_t begin, std::uint32_t end) {
  std::uint32_t li = canonical_merge_partition(
      left, left_count, right, right_count, begin);
  std::uint32_t ri = begin - li;
  for (std::uint32_t position = begin; position < end; ++position) {
    const bool take_left = ri >= right_count ||
        (li < left_count && left[li] < right[ri]);
    output[position] = take_left ? left[li++] : right[ri++];
  }
}

__device__ __forceinline__ Row canonical_candidate_row(
    std::uint32_t candidate, std::uint32_t local_q,
    std::uint32_t source_count,
    const std::uint16_t *source_candidate_offsets,
    const std::uint16_t *source_q_offsets,
    const std::uint64_t *source_q_positions,
    const Row *epoch_rows, ResidentRows arena) {
  std::uint32_t low = 1u, high = source_count;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (source_candidate_offsets[middle] <= candidate)
      low = middle + 1u;
    else
      high = middle;
  }
  const std::uint32_t source = low - 1u;
  const std::uint32_t source_local =
      candidate - source_candidate_offsets[source];
  const std::uint16_t *q_offsets = source_q_offsets +
      source * (kCanonicalJobQuotients + 1u);
  const std::uint64_t position = source_q_positions[
      source * kCanonicalJobQuotients + local_q] +
      source_local - q_offsets[local_q];
  return source == 0u ? epoch_rows[position] : arena[position];
}

// Single-pass output allocation.  State 1 publishes a local count; state 2
// publishes the exclusive prefix.  A job may accumulate across any number of
// count-ready predecessors instead of waiting for every predecessor to finish
// its own prefix handoff.
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

__global__ void canonical_fallback_carry_jobs_kernel(
    BalancedMergeJob *jobs, ResidentPublicationPlan *plan,
    const Row *epoch_rows, const std::uint32_t *epoch_offsets,
    const std::uint32_t *epoch_counts,
    ResidentRows arena, const Descriptor *descriptors,
    const LevelStorageSpan *level_spans,
    const std::uint32_t *level_q_offsets,
    const DeviceManifest *manifests,
    const std::uint32_t *active_manifest,
    CanonicalJobPrefix *prefixes, std::uint32_t *next_job,
    std::uint32_t *cell_counts) {
  constexpr std::uint32_t kThreads = kFoundationCompactionThreads;
  constexpr std::uint32_t kSourceSlots = kMaximumMergeSources;
  constexpr std::uint32_t kMaximumItemsPerThread =
      (kCanonicalCandidateLimit + kThreads - 1u) / kThreads;
  using BlockScan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  extern __shared__ __align__(16) unsigned char workspace[];
  std::uint32_t *plane_a = reinterpret_cast<std::uint32_t *>(workspace);
  std::uint32_t *plane_b = plane_a + plan->job_capacity;

  __shared__ std::uint16_t source_candidate_offsets[kSourceSlots];
  __shared__ std::uint16_t source_lengths[kSourceSlots];
  __shared__ std::uint16_t source_levels[kSourceSlots];
  __shared__ std::uint16_t source_q_offsets[
      kSourceSlots * (kCanonicalJobQuotients + 1u)];
  __shared__ std::uint64_t source_q_positions[
      kSourceSlots * kCanonicalJobQuotients];
  __shared__ std::uint16_t physical_sources[kSourceSlots];
  __shared__ std::uint16_t run_offsets[kSourceSlots];
  __shared__ std::uint16_t run_lengths[kSourceSlots];
  __shared__ std::uint32_t tombstone_words[kCanonicalTombstoneWords];
  __shared__ std::uint32_t source_count_shared;
  __shared__ std::uint32_t physical_run_count_shared;
  __shared__ std::uint32_t run_count_shared;
  __shared__ std::uint32_t small_count_shared;
  __shared__ std::uint32_t largest_count_shared;
  __shared__ std::uint32_t task_rows_shared;
  __shared__ std::uint32_t job_valid_shared;
  __shared__ unsigned long long output_prefix_shared;

  const DeviceManifestSnapshot manifest = load_active_manifest(
      manifests, active_manifest);
  while (!plan->status) {
    __shared__ std::uint32_t job_index_shared;
    if (threadIdx.x == 0u)
      job_index_shared = atomicAdd(next_job, 1u);
    __syncthreads();
    const std::uint32_t job_index = job_index_shared;
    if (job_index >= plan->job_count) return;
    const BalancedMergeJob job = jobs[job_index];
    const std::uint32_t quotient_count =
        job.quotient_end - job.quotient_begin;
    for (std::uint32_t word = threadIdx.x;
         word < kCanonicalTombstoneWords;
         word += blockDim.x)
      tombstone_words[word] = 0u;

    if (threadIdx.x == 0u) {
      const std::uint32_t expected_sources = plan->source_count;
      std::uint32_t source_count = 0u;
      source_levels[source_count++] = kMaximumLevels;
      std::uint64_t levels = manifest.occupied_level_mask &
          canonical_source_level_mask(plan);
      while (levels && source_count < kSourceSlots) {
        const std::uint32_t level =
            static_cast<std::uint32_t>(__ffsll(levels) - 1);
        levels &= levels - 1u;
        source_levels[source_count++] = static_cast<std::uint16_t>(level);
      }
      source_count_shared = source_count;
      std::uint32_t candidate_cursor = 0u;
      std::uint32_t largest_source = 0u;
      std::uint32_t largest_count = 0u;
      for (std::uint32_t source = 0u; source < source_count; ++source) {
        const std::uint32_t level = source_levels[source];
        std::uint16_t *q_offsets = source_q_offsets +
            source * (kCanonicalJobQuotients + 1u);
        std::uint64_t *q_positions = source_q_positions +
            source * kCanonicalJobQuotients;
        std::uint32_t count = 0u;
        for (std::uint32_t local_q = 0u;
             local_q < quotient_count; ++local_q) {
          const std::uint32_t q = job.quotient_begin + local_q;
          const std::uint64_t key_base = std::uint64_t{q} << 16u;
          const std::uint32_t suffix_begin = local_q == 0u
              ? static_cast<std::uint32_t>(job.key_begin - key_base) : 0u;
          const std::uint32_t suffix_end = local_q + 1u == quotient_count
              ? static_cast<std::uint32_t>(job.key_end - key_base)
              : (1u << 16u);
          std::uint64_t section_begin = 0u;
          std::uint32_t section_count = 0u;
          if (source == 0u) {
            section_begin = epoch_offsets[q];
            section_count = epoch_counts[q];
          } else {
            const Descriptor rows =
                descriptors[descriptor_index(q, level)];
            section_begin = rows.offset();
            section_count = rows.count();
          }
          std::uint32_t begin = 0u, end = section_count;
          if (source == 0u) {
            begin = lower_bound_rows(
                epoch_rows + section_begin, section_count, suffix_begin);
            if (suffix_end != (1u << 16u))
              end = begin + lower_bound_rows(
                  epoch_rows + section_begin + begin,
                  section_count - begin, suffix_end);
          } else {
            begin = lower_bound_rows(
                arena + section_begin, section_count, suffix_begin);
            if (suffix_end != (1u << 16u))
              end = begin + lower_bound_rows(
                  arena + section_begin + begin,
                  section_count - begin, suffix_end);
          }
          q_offsets[local_q] = static_cast<std::uint16_t>(count);
          q_positions[local_q] = section_begin + begin;
          count += end - begin;
        }
        q_offsets[quotient_count] = static_cast<std::uint16_t>(count);
        source_candidate_offsets[source] =
            static_cast<std::uint16_t>(candidate_cursor);
        source_lengths[source] = static_cast<std::uint16_t>(count);
        candidate_cursor += count;
        if (count >= largest_count) {
          largest_count = count;
          largest_source = source;
        }
      }
      std::uint32_t physical_count = 0u;
      std::uint32_t physical_cursor = 0u;
      for (std::uint32_t source = 0u; source < source_count; ++source) {
        if (source == largest_source || !source_lengths[source]) continue;
        physical_sources[physical_count] = static_cast<std::uint16_t>(source);
        run_offsets[physical_count] =
            static_cast<std::uint16_t>(physical_cursor);
        run_lengths[physical_count] = source_lengths[source];
        physical_cursor += source_lengths[source];
        ++physical_count;
      }
      const std::uint32_t small_count = physical_cursor;
      if (largest_count) {
        physical_sources[physical_count] =
            static_cast<std::uint16_t>(largest_source);
        run_offsets[physical_count] =
            static_cast<std::uint16_t>(physical_cursor);
        run_lengths[physical_count] =
            static_cast<std::uint16_t>(largest_count);
        ++physical_count;
      }
      physical_run_count_shared = physical_count;
      run_count_shared = physical_count ? physical_count - 1u : 0u;
      small_count_shared = small_count;
      largest_count_shared = largest_count;
      task_rows_shared = candidate_cursor;
      job_valid_shared = source_count == expected_sources &&
          quotient_count && quotient_count <= kCanonicalJobQuotients &&
          candidate_cursor <= plan->job_capacity &&
          candidate_cursor < kCanonicalCandidateLimit;
      atomicAdd(reinterpret_cast<unsigned long long *>(
                    &plan->raw_reservation),
                static_cast<unsigned long long>(candidate_cursor));
      if (!job_valid_shared)
        atomicOr(&plan->status, kPublicationJobTooLarge);
    }
    __syncthreads();

    if (!job_valid_shared) {
      if (threadIdx.x == 0u) {
        canonical_job_prefix(job_index, 0u, prefixes);
      }
      __syncthreads();
      continue;
    }

    for (std::uint32_t physical = 0u;
         physical < physical_run_count_shared; ++physical) {
      const std::uint32_t source = physical_sources[physical];
      const std::uint32_t count = source_lengths[source];
      const std::uint32_t destination = run_offsets[physical];
      const std::uint16_t *q_offsets = source_q_offsets +
          source * (kCanonicalJobQuotients + 1u);
      std::uint32_t local_q = 0u;
      for (std::uint32_t position = threadIdx.x;
           position < count; position += blockDim.x) {
        while (local_q + 1u < quotient_count &&
               position >= q_offsets[local_q + 1u])
          ++local_q;
        const std::uint64_t physical_position = source_q_positions[
            source * kCanonicalJobQuotients + local_q] +
            position - q_offsets[local_q];
        const Row row = source == 0u
            ? epoch_rows[physical_position]
            : arena[physical_position];
        const std::uint32_t candidate =
            source_candidate_offsets[source] + position;
        const std::uint32_t record =
            (local_q << 28u) |
            (std::uint32_t{row.key} << kCanonicalCandidateBits) |
            candidate;
        plane_a[destination + position] = record;
        if (row.flags & kTombstone)
          atomicOr(tombstone_words + (candidate >> 5u),
                   1u << (candidate & 31u));
        if (physical + 1u == physical_run_count_shared)
          plane_b[destination + position] = record;
      }
    }
    __syncthreads();

    bool input_is_a = true;
    while (run_count_shared > 1u) {
      const std::uint32_t *input = input_is_a ? plane_a : plane_b;
      std::uint32_t *output = input_is_a ? plane_b : plane_a;
      const std::uint32_t items =
          (small_count_shared + kThreads - 1u) / kThreads;
      std::uint32_t position = threadIdx.x * items;
      const std::uint32_t thread_end = min(
          position + items, small_count_shared);
      while (position < thread_end) {
        std::uint32_t pair = 0u;
        while (pair * 2u < run_count_shared) {
          const std::uint32_t first = pair * 2u;
          const std::uint32_t pair_rows = run_lengths[first] +
              (first + 1u < run_count_shared
                   ? run_lengths[first + 1u] : 0u);
          if (position < run_offsets[first] + pair_rows) break;
          ++pair;
        }
        const std::uint32_t first = pair * 2u;
        const std::uint32_t begin = run_offsets[first];
        const std::uint32_t left_count = run_lengths[first];
        const std::uint32_t right_count = first + 1u < run_count_shared
            ? run_lengths[first + 1u] : 0u;
        const std::uint32_t pair_end = begin + left_count + right_count;
        const std::uint32_t output_end = min(thread_end, pair_end);
        if (!right_count) {
          while (position < output_end) {
            output[position] = input[position];
            ++position;
          }
        } else {
          canonical_merge_interval(
              input + begin, left_count, input + begin + left_count,
              right_count, output + begin, position - begin,
              output_end - begin);
          position = output_end;
        }
      }
      __syncthreads();
      if (threadIdx.x == 0u) {
        const std::uint32_t old_count = run_count_shared;
        const std::uint32_t next_count = (old_count + 1u) >> 1u;
        for (std::uint32_t next = 0u; next < next_count; ++next) {
          const std::uint32_t first = next * 2u;
          run_offsets[next] = run_offsets[first];
          run_lengths[next] = static_cast<std::uint16_t>(
              run_lengths[first] +
              (first + 1u < old_count ? run_lengths[first + 1u] : 0u));
        }
        run_count_shared = next_count;
      }
      input_is_a = !input_is_a;
      __syncthreads();
    }

    const std::uint32_t *small_input = input_is_a ? plane_a : plane_b;
    std::uint32_t *final_output = input_is_a ? plane_b : plane_a;
    const std::uint32_t *sorted = small_input;
    if (small_count_shared && largest_count_shared) {
      const std::uint32_t items =
          (task_rows_shared + kThreads - 1u) / kThreads;
      const std::uint32_t begin = threadIdx.x * items;
      const std::uint32_t end = min(begin + items, task_rows_shared);
      if (begin < end)
        canonical_merge_interval(
            small_input, small_count_shared,
            small_input + small_count_shared, largest_count_shared,
            final_output, begin, end);
      sorted = final_output;
    }
    __syncthreads();

    const std::uint32_t items_per_thread =
        (task_rows_shared + kThreads - 1u) / kThreads;
    const std::uint32_t thread_begin = threadIdx.x * items_per_thread;
    const std::uint32_t thread_end = min(
        thread_begin + items_per_thread, task_rows_shared);
    std::uint32_t local_live = 0u;
    std::uint32_t live_records[kMaximumItemsPerThread]{};
    std::uint32_t previous_logical_key =
        thread_begin && thread_begin < thread_end
        ? sorted[thread_begin - 1u] >> kCanonicalCandidateBits
        : std::numeric_limits<std::uint32_t>::max();
    for (std::uint32_t position = thread_begin;
         position < thread_end; ++position) {
      const std::uint32_t record = sorted[position];
      const std::uint32_t logical_key = record >> kCanonicalCandidateBits;
      const bool first = previous_logical_key != logical_key;
      previous_logical_key = logical_key;
      if (!first) continue;
      const std::uint32_t candidate =
          record & (kCanonicalCandidateLimit - 1u);
      if (!plan->keep_tombstones &&
          (tombstone_words[candidate >> 5u] &
           (1u << (candidate & 31u))))
        continue;
      live_records[local_live++] = record;
    }
    std::uint32_t thread_output{}, job_output_count{};
    BlockScan(scan_storage).ExclusiveSum(
        local_live, thread_output, job_output_count);
    if (threadIdx.x == 0u) {
      const unsigned long long prefix = canonical_job_prefix(
          job_index, job_output_count, prefixes);
      output_prefix_shared = prefix;
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

    constexpr unsigned full_warp = 0xffffffffu;
    const std::uint32_t lane = threadIdx.x & 31u;
    for (std::uint32_t local = 0u;
         local < kMaximumItemsPerThread; ++local) {
      const bool valid = local < local_live;
      const unsigned active = __ballot_sync(full_warp, valid);
      if (!valid) continue;
      const std::uint32_t record = live_records[local];
      const std::uint32_t candidate =
          record & (kCanonicalCandidateLimit - 1u);
      const std::uint32_t local_q = record >> 28u;
      const Row row = canonical_candidate_row(
          candidate, local_q, source_count_shared,
          source_candidate_offsets, source_q_offsets,
          source_q_positions, epoch_rows, arena);
      arena.store(
          plan->output_begin + output_prefix_shared + thread_output + local,
          row);
      const std::uint32_t q = job.quotient_begin + local_q;
      const std::uint32_t cell =
          q * kFoundationCells + row.key / kFoundationCellKeys;
      const unsigned peers = __match_any_sync(active, cell);
      if (lane == static_cast<std::uint32_t>(__ffs(peers) - 1u))
        atomicAdd(cell_counts + cell, __popc(peers));
    }
    __syncthreads();
  }
}

struct CanonicalTournamentSlice {
  std::uint32_t begin{};
  std::uint32_t count{};
};

// The cache overlay resolves the physical source location once for each
// source/quotient pair.  Cell discovery can then start directly from that
// base instead of following source -> level -> descriptor -> offset again.
__device__ __forceinline__ CanonicalTournamentSlice
canonical_tournament_cell_slice(
    std::uint32_t source, std::uint32_t level, std::uint32_t q,
    std::uint32_t cell, std::uint32_t suffix_begin,
    std::uint32_t suffix_end, std::uint64_t source_base,
    std::uint32_t section_count, const Row *epoch_rows,
    const std::uint16_t *epoch_cell_ranks, ResidentRows arena,
    const std::uint16_t *cell_ranks) {
  if (!section_count) return {};
  const std::uint32_t cell_suffix_begin = cell * kFoundationCellKeys;
  const std::uint32_t cell_suffix_end =
      cell_suffix_begin + kFoundationCellKeys;
  const std::uint16_t *ranks = source == 0u
      ? epoch_cell_ranks + std::size_t{q} * kFoundationCells
      : cell_ranks + std::size_t{level} * kLocalRankEntries +
            std::size_t{q} * kFoundationCells;
  std::uint32_t begin = ranks[cell];
  std::uint32_t end = cell + 1u < kFoundationCells
      ? ranks[cell + 1u] : section_count;
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
    std::uint64_t source_base, std::uint32_t section_count,
    const Row *epoch_rows, const std::uint16_t *epoch_cell_ranks,
    ResidentRows arena, const std::uint16_t *cell_ranks) {
  if (!section_count) return 0u;
  const std::uint64_t quotient_key = std::uint64_t{q} << 16u;
  const std::uint64_t cell_key_begin =
      quotient_key + std::uint64_t{cell} * kFoundationCellKeys;
  const std::uint64_t cell_key_end = cell_key_begin + kFoundationCellKeys;
  if (job.key_begin <= cell_key_begin && job.key_end >= cell_key_end) {
    if (source == 0u)
      return epoch_cell_ranks[
          std::size_t{q} * kFoundationCells + cell];
    return cell_ranks[
        std::size_t{level} * kLocalRankEntries +
        std::size_t{q} * kFoundationCells + cell];
  }
  const std::uint64_t clipped_begin = max(cell_key_begin, job.key_begin);
  const std::uint64_t clipped_end = min(cell_key_end, job.key_end);
  return canonical_tournament_cell_slice(
      source, level, q, cell,
      static_cast<std::uint32_t>(clipped_begin - quotient_key),
      static_cast<std::uint32_t>(clipped_end - quotient_key),
      source_base, section_count, epoch_rows, epoch_cell_ranks,
      arena, cell_ranks).begin;
}

__device__ __forceinline__ Row canonical_tournament_source_row(
    std::uint32_t source, std::uint64_t source_base,
    std::uint32_t section_position, const Row *epoch_rows,
    ResidentRows arena) {
  if (source == 0u)
    return epoch_rows[source_base + section_position];
  return arena[source_base + section_position];
}

__device__ __forceinline__ std::uint32_t
canonical_tournament_source_head(
    std::uint32_t source, std::uint64_t source_base,
    std::uint32_t section_position, const Row *epoch_rows,
    ResidentRows arena, std::uint32_t local_position) {
  std::uint32_t packed = 0u;
  if (source == 0u) {
    const Row *row = epoch_rows + source_base + section_position;
    packed = std::uint32_t{row->key} |
        (std::uint32_t{row->flags} << 16u);
  } else {
    packed = arena.key_flags[source_base + section_position];
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

// A cell is one independent sorted merge chain.  One hundred twenty-eight
// chains execute in
// parallel and dynamically claim cells.  Each chain keeps one head per source
// and an updateable tournament tree; advancing a source changes only O(log k)
// comparisons.  The algorithm is identical for every k up to the structural
// level bound, and reads each selected row only once after survivor offsets are
// known.
__global__ __launch_bounds__(kFoundationCompactionThreads)
void canonical_tournament_carry_jobs_kernel(
    BalancedMergeJob *jobs, const std::uint64_t *reservations,
    ResidentPublicationPlan *plan, const Row *epoch_rows,
    const std::uint32_t *epoch_offsets,
    const std::uint32_t *epoch_counts,
    const std::uint16_t *epoch_cell_ranks, ResidentRows arena,
    const Descriptor *descriptors, const LevelStorageSpan *level_spans,
    const std::uint16_t *cell_ranks, const DeviceManifest *manifests,
    const std::uint32_t *active_manifest,
    CanonicalJobPrefix *prefixes, std::uint32_t *next_job,
    std::uint32_t *cell_counts) {
  constexpr std::uint32_t kThreads = kFoundationCompactionThreads;
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
  // The source set is fixed for the publication, so build it once per block
  // instead of once for every dynamically claimed job.
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
    const std::uint32_t quotient_count =
        job.quotient_end - job.quotient_begin;

    if (threadIdx.x == 0u) {
      task_count_shared = quotient_count * kFoundationCells;
      job_capacity_shared = canonical_job_capacity(plan, quotient_count);
      next_task_shared = 0u;
      tape_cursor_shared = 0u;
      job_valid_shared = source_count_shared == plan->source_count &&
          source_count_shared <= kMaximumMergeSources && quotient_count &&
          quotient_count <= kCanonicalJobQuotients && task_count_shared &&
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
    const std::size_t source_quotients =
        std::size_t{quotient_count} * source_count;
    offset = canonical_align_bytes(offset, alignof(std::uint64_t));
    std::uint64_t *source_bases =
        reinterpret_cast<std::uint64_t *>(workspace + offset);
    offset += source_quotients * sizeof(std::uint64_t);
    offset = canonical_align_bytes(offset, alignof(std::uint32_t));
    std::uint32_t *source_section_counts =
        reinterpret_cast<std::uint32_t *>(workspace + offset);
    offset += source_quotients * sizeof(std::uint32_t);
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

    const std::uint32_t active_source_quotients =
        quotient_count * source_count;
    for (std::uint32_t entry = threadIdx.x;
         entry < active_source_quotients; entry += blockDim.x) {
      const std::uint32_t local_q = entry / source_count;
      const std::uint32_t source = entry - local_q * source_count;
      const std::uint32_t q = job.quotient_begin + local_q;
      if (source == 0u) {
        source_bases[entry] = epoch_offsets[q];
        source_section_counts[entry] = epoch_counts[q];
      } else {
        const Descriptor descriptor = descriptors[
            descriptor_index(q, source_levels[source])];
        source_bases[entry] = descriptor.offset();
        source_section_counts[entry] = descriptor.count();
      }
    }
    __syncthreads();

    if (threadIdx.x < kCanonicalTournamentChains) {
      const std::uint32_t chain = threadIdx.x;
      for (;;) {
        const std::uint32_t task = atomicAdd(&next_task_shared, 1u);
        if (task >= task_count_shared) break;
        const std::uint32_t q =
            job.quotient_begin + task / kFoundationCells;
        const std::uint32_t cell = task % kFoundationCells;
        const std::uint64_t cell_key_begin =
            (std::uint64_t{q} << 16u) +
            std::uint64_t{cell} * kFoundationCellKeys;
        const std::uint64_t cell_key_end =
            cell_key_begin + kFoundationCellKeys;
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
          const std::uint32_t local_q = q - job.quotient_begin;
          const std::size_t entry =
              std::size_t{local_q} * source_count + source;
          const std::uint64_t source_base = source_bases[entry];
          const CanonicalTournamentSlice slice =
              canonical_tournament_cell_slice(
                  source, level, q, cell, suffix_begin, suffix_end,
                  source_base, source_section_counts[entry], epoch_rows,
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
          const std::uint32_t local_q = q - job.quotient_begin;
          const std::uint64_t source_base = source_bases[
              std::size_t{local_q} * source_count + source];

          // This is the next required row, not a speculative read.  Issue it
          // before survivor bookkeeping so its latency can overlap the
          // duplicate and tombstone work below.
          std::uint32_t advanced_head = 1u << 16u;
          if (left)
            advanced_head = canonical_tournament_source_head(
                source, source_base, next_position, epoch_rows, arena,
                local_position + 1u);

          if (key != previous_key) {
            if (plan->keep_tombstones || !(head & (1u << 17u))) {
              // Canonical epoch and level runs contain at most one row for
              // each logical key.  Therefore a source contributes at most
              // the 512 key positions in one cell.  Duplicate user writes
              // are resolved before this merge; they are not an external
              // unique-key requirement.
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
                      std::size_t{q} * kFoundationCells + cell,
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

    // Each warp materializes contiguous output positions.  Besides coalescing
    // the final stores, lanes that need the same source slice share one rank
    // lookup.  This works for every source count and avoids a per-source
    // register array.
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
            job.quotient_begin + task / kFoundationCells;
        const std::uint32_t cell = task % kFoundationCells;
        std::uint32_t source = 0u;
        std::uint32_t position = 0u;
        std::uint64_t source_base = 0u;
        std::uint32_t source_section_count = 0u;
        source = reference >> kCanonicalTournamentReferenceBits;
        const std::uint32_t local_q = q - job.quotient_begin;
        const std::size_t entry =
            std::size_t{local_q} * source_count + source;
        source_base = source_bases[entry];
        source_section_count = source_section_counts[entry];
        const unsigned peers = __match_any_sync(
            active, (task << kMergeSourceBits) | source);
        const std::uint32_t leader =
            static_cast<std::uint32_t>(__ffs(peers) - 1);
        std::uint32_t slice_begin = 0u;
        if (lane == leader)
          slice_begin = canonical_tournament_slice_begin(
              source, source_levels[source], q, cell, job,
              source_base, source_section_count, epoch_rows,
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

__global__ void copy_canonical_epoch_kernel(
    const Row *rows, const std::uint32_t *count,
    ResidentRows arena, std::uint64_t destination) {
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < *count; i += gridDim.x * blockDim.x)
    arena.store(destination + i, rows[i]);
}

__global__ void build_canonical_rank_from_run_kernel(
    ResidentRows arena, std::uint64_t run_begin,
    const std::uint32_t *quotient_offsets, std::uint32_t level,
    std::uint16_t *cell_ranks) {
  const std::uint32_t q = blockIdx.x;
  const std::uint32_t cell = threadIdx.x;
  const std::uint32_t begin = quotient_offsets[q];
  const std::uint32_t count = quotient_offsets[q + 1u] - begin;
  const std::uint32_t rank = lower_bound_rows(
      arena + run_begin + begin, count, cell * kFoundationCellKeys);
  cell_ranks[std::size_t{level} * kLocalRankEntries +
             std::size_t{q} * kFoundationCells + cell] =
      static_cast<std::uint16_t>(rank);
}

__global__ void build_canonical_rank_from_counts_kernel(
    const std::uint32_t *cell_counts,
    const ResidentPublicationPlan *plan, std::uint32_t level_count,
    std::uint16_t *cell_ranks, std::uint32_t *quotient_counts) {
  using Scan = cub::BlockScan<std::uint32_t, kFoundationCells>;
  __shared__ typename Scan::TempStorage storage;
  const std::uint32_t q = blockIdx.x;
  const std::uint32_t cell = threadIdx.x;
  if (plan->status || plan->destination_level >= level_count) {
    if (threadIdx.x == 0u) quotient_counts[q] = 0u;
    if (q == 0u && threadIdx.x == 0u)
      quotient_counts[kQuotients] = 0u;
    return;
  }
  const std::uint32_t level = plan->destination_level;
  const std::uint32_t count =
      cell_counts[std::size_t{q} * kFoundationCells + cell];
  std::uint32_t rank{}, total{};
  Scan(storage).ExclusiveSum(count, rank, total);
  cell_ranks[std::size_t{level} * kLocalRankEntries +
             std::size_t{q} * kFoundationCells + cell] =
      static_cast<std::uint16_t>(rank);
  if (threadIdx.x == 0u) quotient_counts[q] = total;
  if (q == 0u && threadIdx.x == 0u)
    quotient_counts[kQuotients] = 0u;
}

__global__ void finalize_canonical_level_metadata_kernel(
    const std::uint32_t *quotient_offsets,
    const LevelStorageSpan *level_spans,
    ResidentPublicationPlan *plan, Descriptor *descriptors,
    std::uint32_t route_stride, RouteHeader *route_headers,
    RouteSlice *route_slices, std::uint32_t *route_logical_begins,
    std::uint16_t *route_quotients,
    std::uint32_t *level_q_logical_offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients || plan->status) return;
  const std::uint32_t level = plan->destination_level;
  const std::uint32_t offset = quotient_offsets[q];
  level_q_logical_offsets[
      std::size_t{level} * (kQuotients + 1u) + q] = offset;
  if (q == kQuotients) return;
  const std::uint32_t count = quotient_offsets[q + 1u] - offset;
  const Descriptor descriptor = Descriptor::make(
      plan->output_begin + offset, count);
  const std::size_t mapping = descriptor_index(q, level);
  descriptors[mapping] = descriptor;
  const std::uint32_t route = level * route_stride + q;
  route_headers[mapping] = {route, count ? 1u : 0u};
  route_slices[route] = {descriptor, 0u, 1u << 16u};
  route_logical_begins[route] = offset;
  route_quotients[route] = static_cast<std::uint16_t>(q);
  if (q == 0u) {
    const std::uint64_t total = quotient_offsets[kQuotients];
    plan->survivor_count = total;
    if (total > plan->output_capacity)
      atomicOr(&plan->status, kPublicationOutputOverflow);
  }
}

__global__ void finalize_canonical_section_metadata_kernel(
    const std::uint32_t *section_begins,
    const std::uint32_t *section_counts,
    const std::uint32_t *selected_count,
    const LevelStorageSpan *level_spans,
    ResidentPublicationPlan *plan, Descriptor *descriptors,
    std::uint32_t route_stride, RouteHeader *route_headers,
    RouteSlice *route_slices, std::uint32_t *route_logical_begins,
    std::uint16_t *route_quotients,
    std::uint32_t *level_q_logical_offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients || plan->status) return;
  const std::uint32_t level = plan->destination_level;
  const std::uint32_t begin = section_begins[q];
  level_q_logical_offsets[
      std::size_t{level} * (kQuotients + 1u) + q] = begin;
  if (q == kQuotients) return;
  const std::uint32_t count = section_counts[q];
  const Descriptor descriptor = Descriptor::make(
      plan->output_begin + begin, count);
  const std::size_t mapping = descriptor_index(q, level);
  descriptors[mapping] = descriptor;
  const std::uint32_t route = level * route_stride + q;
  route_headers[mapping] = {route, count ? 1u : 0u};
  route_slices[route] = {descriptor, 0u, 1u << 16u};
  route_logical_begins[route] = begin;
  route_quotients[route] = static_cast<std::uint16_t>(q);
  if (q == 0u) {
    plan->survivor_count = *selected_count;
    if (*selected_count > plan->output_capacity)
      atomicOr(&plan->status, kPublicationOutputOverflow);
  }
}

__device__ __forceinline__ void emit_lookup_result(
    std::uint32_t query_index, bool found, std::uint32_t value,
    std::uint32_t *out_values, std::uint8_t *out_found,
    const std::uint32_t *query_ids) {
  const std::uint32_t destination =
      query_ids ? query_ids[query_index] : query_index;
  out_values[destination] = found ? value : out_found ? 0u : kInvalid;
  if (out_found) out_found[destination] = found;
}

__device__ __forceinline__ void canonical_lookup_resident_only(
    std::uint32_t key, std::uint32_t query_index,
    std::uint32_t *out_values, std::uint8_t *out_found,
    ResidentRows arena, const Descriptor *descriptors,
    const std::uint16_t *cell_ranks, std::uint64_t levels,
    const std::uint32_t *query_ids) {
  const std::uint32_t q = key >> 16u;
  const std::uint32_t suffix = key_suffix(key);
  while (levels) {
    const std::uint32_t level =
        static_cast<std::uint32_t>(__ffsll(levels) - 1);
    levels &= levels - 1u;
    const Descriptor rows = descriptors[descriptor_index(q, level)];
    if (!rows.count()) continue;
    const std::uint32_t cell = suffix / kFoundationCellKeys;
    const std::uint16_t *ranks = cell_ranks +
        std::size_t{level} * kLocalRankEntries +
        std::size_t{q} * kFoundationCells;
    const std::uint32_t begin = ranks[cell];
    const std::uint32_t end = cell + 1u < kFoundationCells
        ? ranks[cell + 1u] : rows.count();
    Row winner{};
    if (!find_unique_point_row(
            arena + rows.offset() + begin, end - begin, suffix, winner))
      continue;
    const bool live = (winner.flags & kTombstone) == 0u;
    emit_lookup_result(
        query_index, live, winner.value, out_values, out_found, query_ids);
    return;
  }
  emit_lookup_result(
      query_index, false, 0u, out_values, out_found, query_ids);
}

__global__ void canonical_lookup_with_pending_kernel(
    const std::uint32_t *queries, std::uint32_t *out_values,
    std::uint8_t *out_found, std::uint32_t count,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const std::uint64_t *batch_signatures,
    const std::uint64_t *epoch_signatures, ResidentRows arena,
    const Descriptor *descriptors, const std::uint16_t *cell_ranks,
    const std::uint32_t *query_ids,
    const std::uint64_t *occupied_mask) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t key = queries[i];
  const std::uint32_t q = key >> 16u;
  const std::uint32_t suffix = key_suffix(key);
  const std::uint64_t signature_bits = pending_signature_bits(key);
  if (pending_batches &&
      (epoch_signatures[q] & signature_bits) == signature_bits) {
    for (int batch = static_cast<int>(pending_batches) - 1;
         batch >= 0; --batch) {
      const std::uint32_t b = static_cast<std::uint32_t>(batch);
      const std::uint64_t signature =
          batch_signatures[std::size_t{b} * kQuotients + q];
      if ((signature & signature_bits) != signature_bits) continue;
      const std::size_t oi = std::size_t{b} * (kQuotients + 1u) + q;
      const std::uint32_t begin = raw_offsets[oi];
      const std::uint32_t end = raw_offsets[oi + 1u];
      bool matched = false;
      std::uint32_t newest = 0u;
      Row winner{};
      for (std::uint32_t position = begin; position < end; ++position) {
        const std::uint32_t record = b * batch_stride + position;
        if (key_suffix(raw_keys[record]) != suffix) continue;
        const RawPayload payload = raw_payloads[record];
        const std::uint32_t age = raw_position(payload);
        if (!matched || age > newest) {
          winner = raw_row(key, payload);
          newest = age;
          matched = true;
        }
      }
      if (matched) {
        const bool live = (winner.flags & kTombstone) == 0u;
        emit_lookup_result(
            i, live, winner.value, out_values, out_found, query_ids);
        return;
      }
    }
  }
  canonical_lookup_resident_only(
      key, i, out_values, out_found, arena, descriptors, cell_ranks,
      __ldg(occupied_mask), query_ids);
}

inline std::uint32_t select_canonical_merge_capacity() {
  int maximum_blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maximum_blocks, canonical_fallback_carry_jobs_kernel,
      kFoundationCompactionThreads, 0u));
  // Three larger jobs per SM outperform a greater number of smaller jobs:
  // they reduce planning and boundary work while retaining enough warps to
  // hide the fallback merge's latency.
  const int desired_blocks = std::max(1, std::min(3, maximum_blocks));
  std::size_t shared_budget = 0u;
  CUDA_CHECK(cudaOccupancyAvailableDynamicSMemPerBlock(
      &shared_budget, canonical_fallback_carry_jobs_kernel, desired_blocks,
      kFoundationCompactionThreads));

  // Retain the established job-capacity calculation, deriving its shared
  // memory budget from the production canonical kernel.
  // Overlay cell cursors on the second index plane.
  const std::uint32_t minimum_capacity = std::max(
      kMaximumMergeSources, 4u * kFoundationCells + 1u);
  std::uint32_t block_low = minimum_capacity;
  std::uint32_t block_high = kBalancedMergeCapacityCeiling;
  while (block_low < block_high) {
    const std::uint32_t middle =
        block_low + (block_high - block_low + 1u) / 2u;
    if (canonical_capacity_reservation_bytes(middle) <= shared_budget)
      block_low = middle;
    else
      block_high = middle - 1u;
  }
  if (block_low < minimum_capacity)
    throw std::runtime_error("insufficient shared memory for GPULSMOpt merge");
  // Keep both index planes four-byte aligned.
  return (block_low & 1u) ? block_low : block_low - 1u;
}

__device__ __forceinline__ std::uint32_t tqrj_directory_find(
    const std::uint16_t *directory_offsets,
    const std::uint16_t *directory_suffixes,
    std::uint32_t target) {
  const std::uint32_t interval = target >> 8u;
  std::uint32_t position = directory_offsets[interval];
  const std::uint32_t end = directory_offsets[interval + 1u];
  if (end - position <= kTqrjPairFindThreshold) {
    for (; position < end; ++position)
      if (directory_suffixes[position] == target) return position;
    return kInvalid;
  }

  // Preserve first-owner order while aligning the remaining probe to one
  // 32-bit shared load for each pair of exact 16-bit suffixes.
  if (position < end) {
    if (directory_suffixes[position] == target) return position;
    ++position;
  }
  if ((position & 1u) && position < end) {
    if (directory_suffixes[position] == target) return position;
    ++position;
  }
  const std::uint32_t *packed =
      reinterpret_cast<const std::uint32_t *>(directory_suffixes);
  for (; position + 1u < end; position += 2u) {
    const std::uint32_t pair = packed[position >> 1u];
    if ((pair & 0xffffu) == target) return position;
    if ((pair >> 16u) == target) return position + 1u;
  }
  if (position < end && directory_suffixes[position] == target)
    return position;
  return kInvalid;
}

__device__ __forceinline__ unsigned long long tqrj_pending_token(
    const RawPayload &payload) {
  const std::uint64_t order =
      static_cast<std::uint64_t>(raw_position(payload)) + 1u;
  const std::uint64_t live =
      (payload.metadata & kRawTombstone) == 0u ? 1u : 0u;
  return static_cast<unsigned long long>(
      (order << 33u) | (live << 32u) | payload.value);
}

// All lanes call this helper.  Matching lanes with the same exact owner
// collapse to one atomic update, for either shared or global winner storage.
__device__ __forceinline__ void tqrj_warp_atomic_max(
    bool matched, std::uint32_t owner, unsigned long long token,
    unsigned long long *winners) {
  const unsigned matched_mask = __ballot_sync(0xffffffffu, matched);
  if (!matched) return;
  const unsigned peers = __match_any_sync(matched_mask, owner);
  unsigned long long aggregate = token;
  for (unsigned remaining = peers; remaining;
       remaining &= remaining - 1u) {
    const int source_lane = __ffs(remaining) - 1;
    const unsigned long long candidate =
        __shfl_sync(peers, token, source_lane);
    aggregate = candidate > aggregate ? candidate : aggregate;
  }
  const std::uint32_t lane = threadIdx.x & 31u;
  if (lane == static_cast<std::uint32_t>(__ffs(peers) - 1))
    atomicMax(winners + owner, aggregate);
}

struct TqrjHashTask {
  std::uint32_t quotient;
  std::uint32_t pending_rows;
  std::uint32_t query_tile_base;
  std::uint32_t pending_tile_base;
};

struct TqrjHashTile {
  std::uint32_t task;
  std::uint32_t begin;
};

// Typed names for storage borrowed by one dense lookup.  The owning buffers
// retain their other operation-specific names because publication, canonical,
// range, and TQRJ execute under the same operation lock.  These aliases add no
// storage; they only make TQRJ's phase lifetimes explicit at its launch site.
struct TqrjLookupWorkspace {
  std::uint32_t *grouped_queries;
  std::uint32_t *query_ids;
  std::uint32_t *reservation_ranks;
  std::uint32_t *active_quotients;
  std::uint32_t *active_query_counts;
  std::uint32_t *active_query_offsets;
  std::uint32_t *active_quotient_count;
  std::uint32_t *query_bases;

  TqrjHashTask *hash_tasks;
  std::uint32_t *hash_counters;
  std::uint32_t *hash_task_count;
  unsigned long long *hash_winners;
  TqrjHashTile *hash_query_tiles;
  TqrjHashTile *hash_pending_tiles;
  std::uint32_t *hash_table;
  std::uint32_t hash_table_capacity;
};

constexpr std::size_t kTqrjHashTaskBytes =
    std::size_t{kQuotients} * sizeof(TqrjHashTask);
constexpr std::size_t kTqrjHashCounterBytes =
    6u * sizeof(std::uint32_t);
static_assert(kTqrjHashTaskBytes + kTqrjHashCounterBytes <=
              std::size_t{kLocalRankEntries} * sizeof(std::uint32_t));

__device__ __forceinline__ void tqrj_enqueue_hash_task(
    std::uint32_t quotient, std::uint32_t pending_rows,
    TqrjHashTask *tasks, std::uint32_t *task_count) {
  const std::uint32_t ticket = atomicAdd(task_count, 1u);
  tasks[ticket] = {quotient, pending_rows, 0u, 0u};
}

__device__ __forceinline__ std::uint32_t tqrj_pending_rows(
    std::uint32_t q, const std::uint32_t *raw_offsets,
    std::uint32_t pending_batches) {
  std::uint32_t total = 0u;
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t oi =
        std::size_t{batch} * (kQuotients + 1u) + q;
    total += raw_offsets[oi + 1u] - raw_offsets[oi];
  }
  return total;
}

__device__ __forceinline__ void tqrj_write_result(
    std::uint32_t key, std::uint32_t query_index,
    unsigned long long winner, std::uint32_t *out_values,
    std::uint8_t *out_found, ResidentRows arena,
    const Descriptor *descriptors,
    const std::uint16_t *canonical_cell_ranks,
    std::uint64_t occupied_levels, const std::uint32_t *query_ids) {
  if (winner) {
    const bool live = ((winner >> 32u) & 1u) != 0u;
    emit_lookup_result(
        query_index, live, static_cast<std::uint32_t>(winner), out_values,
        out_found, query_ids);
    return;
  }
  canonical_lookup_resident_only(
      key, query_index, out_values, out_found, arena, descriptors,
      canonical_cell_ranks, occupied_levels, query_ids);
}

// One block owns one ordinary quotient.  Oversized, high-pending, or locally
// concentrated tasks are enqueued for the one size-independent skew executor
// below.
__global__ void tqrj_direct_lookup_kernel(
    const std::uint32_t *queries, std::uint32_t *out_values,
    std::uint8_t *out_found, const std::uint32_t *active_quotients,
    const std::uint32_t *active_counts,
    const std::uint32_t *active_count,
    const std::uint32_t *query_bases,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, ResidentRows arena,
    const Descriptor *descriptors, const std::uint32_t *query_ids,
    const std::uint16_t *canonical_cell_ranks,
    const std::uint64_t *query_occupied_level_mask,
    TqrjHashTask *hash_tasks, std::uint32_t *hash_task_count) {
  using BlockScan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::uint32_t directory_counts[kTqrjDirectoryBins];
  __shared__ std::uint16_t directory_offsets[kTqrjDirectoryBins + 1u];
  __shared__ __align__(4)
      std::uint16_t directory_suffixes[kTqrjDirectCapacity];
  __shared__ unsigned long long winners[kTqrjDirectCapacity];
  __shared__ std::uint64_t occupied_levels;
  __shared__ std::uint32_t pending_rows;

  const std::uint32_t task = blockIdx.x;
  if (task >= *active_count) return;
  const std::uint32_t q = active_quotients[task];
  const std::uint32_t query_begin = query_bases[q];
  const std::uint32_t query_count = active_counts[task];
  if (threadIdx.x == 0u) {
    pending_rows = tqrj_pending_rows(q, raw_offsets, pending_batches);
    occupied_levels =
        load_query_manifest(query_occupied_level_mask).occupied_level_mask;
  }
  __syncthreads();

  if (query_count > kTqrjDirectCapacity ||
      pending_rows > kTqrjDirectPendingRows) {
    if (threadIdx.x == 0u)
      tqrj_enqueue_hash_task(
          q, pending_rows, hash_tasks, hash_task_count);
    return;
  }
  directory_counts[threadIdx.x] = 0u;
  __syncthreads();

  // Direct radix addressing: the upper suffix byte chooses one of 256 exact
  // compact lists.  Only actual queries are counted and scattered; unlike a
  // fixed-capacity block sort, small direct tasks do no padded directory work.
  for (std::uint32_t local = threadIdx.x; local < query_count;
       local += blockDim.x) {
    const std::uint32_t suffix =
        key_suffix(queries[query_begin + local]);
    atomicAdd(directory_counts + (suffix >> 8u), 1u);
  }
  __syncthreads();
  const std::uint32_t interval_count = directory_counts[threadIdx.x];
  std::uint32_t interval_begin = 0u;
  BlockScan(scan_storage).ExclusiveSum(interval_count, interval_begin);
  directory_offsets[threadIdx.x] =
      static_cast<std::uint16_t>(interval_begin);
  if (threadIdx.x + 1u == kTqrjDirectoryBins)
    directory_offsets[kTqrjDirectoryBins] =
        static_cast<std::uint16_t>(interval_begin + interval_count);
  directory_counts[threadIdx.x] = 0u;
  const bool crowded = __syncthreads_or(
      interval_count > kTqrjDirectIntervalLimit);
  if (crowded) {
    if (threadIdx.x == 0u)
      tqrj_enqueue_hash_task(
          q, pending_rows, hash_tasks, hash_task_count);
    return;
  }

  for (std::uint32_t local = threadIdx.x; local < query_count;
       local += blockDim.x) {
    const std::uint32_t suffix =
        key_suffix(queries[query_begin + local]);
    const std::uint32_t interval = suffix >> 8u;
    const std::uint32_t rank = atomicAdd(
        directory_counts + interval, 1u);
    directory_suffixes[directory_offsets[interval] + rank] =
        static_cast<std::uint16_t>(suffix);
    winners[local] = 0ull;
  }
  __syncthreads();

  // The existing raw layout already gives this quotient one contiguous
  // section in every admission batch.  Assign batches to warps so the short
  // sections in the measured workload execute concurrently.
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  constexpr std::uint32_t kWarps = kThreads / 32u;
  for (std::uint32_t batch = warp; batch < pending_batches;
       batch += kWarps) {
    const std::size_t oi =
        std::size_t{batch} * (kQuotients + 1u) + q;
    const std::uint32_t begin = raw_offsets[oi];
    const std::uint32_t end = raw_offsets[oi + 1u];
    for (std::uint32_t base = begin; base < end; base += 32u) {
      const std::uint32_t position = base + lane;
      bool matched = false;
      std::uint32_t owner = kInvalid;
      unsigned long long token = 0ull;
      if (position < end) {
        const std::uint32_t record = batch * batch_stride + position;
        const std::uint32_t pending_suffix =
            key_suffix(raw_keys[record]);
        owner = tqrj_directory_find(
            directory_offsets, directory_suffixes, pending_suffix);
        matched = owner != kInvalid;
        if (matched) token = tqrj_pending_token(raw_payloads[record]);
      }
      tqrj_warp_atomic_max(matched, owner, token, winners);
    }
  }
  __syncthreads();

  for (std::uint32_t local = threadIdx.x; local < query_count;
       local += blockDim.x) {
    const std::uint32_t query_index = query_begin + local;
    const std::uint32_t key = queries[query_index];
    const std::uint32_t suffix = key_suffix(key);
    const std::uint32_t owner = tqrj_directory_find(
        directory_offsets, directory_suffixes, suffix);
    const unsigned long long winner = owner == kInvalid
        ? 0ull : winners[owner];
    tqrj_write_result(
        key, query_index, winner, out_values, out_found, arena,
        descriptors, canonical_cell_ranks, occupied_levels, query_ids);
  }
}

__host__ __device__ constexpr std::uint32_t tqrj_hash_tile_count(
    std::uint32_t rows) {
  return (rows + kTqrjHashTileRows - 1u) / kTqrjHashTileRows;
}

// Exact overflow directory.  Capacity is proportional to the number of
// queries that actually overflow compact CSR; it has no key-domain term and
// no power-of-two discontinuity.
__host__ __device__ constexpr std::uint64_t tqrj_hash_seed(
    std::uint32_t attempt) {
  return attempt == 0u ? 0x6a09e667f3bcc909ull
      : attempt == 1u ? 0x243f6a8885a308d3ull
      : 0x13198a2e03707344ull;
}

__host__ __device__ __forceinline__ std::uint64_t tqrj_hash_mix(
    std::uint64_t value) {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

__host__ __device__ __forceinline__ std::uint64_t tqrj_hash_key(
    std::uint32_t key, std::uint64_t seed) {
  return tqrj_hash_mix(static_cast<std::uint64_t>(key) ^ seed);
}


__host__ __device__ constexpr std::uint32_t tqrj_hash_capacity(
    std::uint32_t rows) {
  if (!rows) return 0u;
  const std::uint64_t proportional =
      (std::uint64_t{13u} * rows + 7u) / 8u;
  return static_cast<std::uint32_t>(
      (proportional + kTqrjHashCapacityAlignment - 1u) &
      ~std::uint64_t{kTqrjHashCapacityAlignment - 1u});
}

__host__ __device__ __forceinline__ std::uint32_t tqrj_hash_slot(
    std::uint64_t hash, std::uint32_t capacity) {
  return static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(hash)) *
       capacity) >> 32u);
}

__host__ __device__ __forceinline__ std::uint32_t tqrj_hash_entry(
    std::uint32_t owner, std::uint64_t hash) {
  const std::uint32_t fingerprint = static_cast<std::uint32_t>(
      hash >> (64u - kTqrjHashFingerprintBits));
  return (fingerprint << kTqrjHashOwnerBits) | (owner + 1u);
}

__host__ __device__ __forceinline__ std::uint32_t
tqrj_hash_entry_owner(std::uint32_t entry) {
  return (entry & kTqrjHashOwnerMask) - 1u;
}

__host__ __device__ __forceinline__ std::uint32_t
tqrj_hash_entry_fingerprint(std::uint32_t entry) {
  return entry >> kTqrjHashOwnerBits;
}

__device__ __forceinline__ std::uint32_t tqrj_hash_insert(
    const std::uint32_t *grouped_queries, std::uint32_t query,
    std::uint32_t *table, std::uint32_t capacity,
    std::uint64_t seed, std::uint32_t maximum_probes,
    std::uint32_t *failure) {
  const std::uint32_t key = grouped_queries[query];
  const std::uint64_t hash = tqrj_hash_key(key, seed);
  const std::uint32_t desired = tqrj_hash_entry(query, hash);
  const std::uint32_t fingerprint =
      tqrj_hash_entry_fingerprint(desired);
  std::uint32_t slot = tqrj_hash_slot(hash, capacity);
  for (std::uint32_t probe = 0u; probe < maximum_probes; ++probe) {
    const std::uint32_t previous = atomicCAS(table + slot, 0u, desired);
    if (!previous) return query;
    if (tqrj_hash_entry_fingerprint(previous) == fingerprint) {
      const std::uint32_t owner = tqrj_hash_entry_owner(previous);
      if (grouped_queries[owner] == key) return owner;
    }
    if (++slot == capacity) slot = 0u;
  }
  atomicExch(failure, 1u);
  return kInvalid;
}

__device__ __forceinline__ std::uint32_t tqrj_hash_find(
    std::uint32_t key, const std::uint32_t *grouped_queries,
    const std::uint32_t *table, std::uint32_t capacity,
    std::uint64_t seed) {
  const std::uint64_t hash = tqrj_hash_key(key, seed);
  const std::uint32_t fingerprint = tqrj_hash_entry_fingerprint(
      tqrj_hash_entry(0u, hash));
  std::uint32_t slot = tqrj_hash_slot(hash, capacity);
  for (std::uint32_t probe = 0u; probe < capacity; ++probe) {
    const std::uint32_t entry = table[slot];
    if (!entry) return kInvalid;
    if (tqrj_hash_entry_fingerprint(entry) == fingerprint) {
      const std::uint32_t owner = tqrj_hash_entry_owner(entry);
      if (grouped_queries[owner] == key) return owner;
    }
    if (++slot == capacity) slot = 0u;
  }
  return kInvalid;
}

__global__ void tqrj_hash_lookup_kernel(
    TqrjHashTask *tasks, const std::uint32_t *task_count,
    const std::uint32_t *query_bases,
    const std::uint32_t *query_counts,
    std::uint32_t *grouped_queries,
    TqrjHashTile *query_tiles, TqrjHashTile *pending_tiles,
    std::uint32_t *counters, std::uint32_t *hash_table,
    std::uint32_t maximum_hash_entries,
    unsigned long long *winners,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches,
    std::uint32_t *out_values, std::uint8_t *out_found,
    ResidentRows arena, const Descriptor *descriptors,
    const std::uint32_t *query_ids,
    const std::uint16_t *canonical_cell_ranks,
    const std::uint64_t *query_occupied_level_mask
    ) {
  const cooperative_groups::grid_group grid =
      cooperative_groups::this_grid();
  __shared__ std::uint32_t section_prefixes[kBatchesPerEpoch + 1u];
  __shared__ TqrjHashTask shared_task;
  __shared__ std::uint32_t shared_a, shared_b;
  __shared__ std::uint64_t occupied_levels;

  const std::uint32_t tasks_in_queue = *task_count;
  if (!tasks_in_queue) return;
  const std::uint32_t global_thread =
      blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t global_stride = blockDim.x * gridDim.x;
  const std::uint32_t lane = threadIdx.x & 31u;

  // First materialize balanced query and pending tiles and count only the
  // queries that actually overflowed direct matching.
  for (std::uint32_t task_index = blockIdx.x;
       task_index < tasks_in_queue; task_index += gridDim.x) {
    if (threadIdx.x == 0u) {
      const TqrjHashTask task = tasks[task_index];
      const std::uint32_t query_count = query_counts[task.quotient];
      shared_a = tqrj_hash_tile_count(query_count);
      shared_b = tqrj_hash_tile_count(task.pending_rows);
      tasks[task_index].query_tile_base = atomicAdd(counters, shared_a);
      tasks[task_index].pending_tile_base = atomicAdd(counters + 1u, shared_b);
      atomicAdd(counters + 2u, query_count);
    }
    __syncthreads();
    const TqrjHashTask task = tasks[task_index];
    for (std::uint32_t tile = threadIdx.x; tile < shared_a;
         tile += blockDim.x)
      query_tiles[task.query_tile_base + tile] =
          {task_index, tile * kTqrjHashTileRows};
    for (std::uint32_t tile = threadIdx.x; tile < shared_b;
         tile += blockDim.x)
      pending_tiles[task.pending_tile_base + tile] =
          {task_index, tile * kTqrjHashTileRows};
    __syncthreads();
  }
  grid.sync();

  const std::uint32_t query_tile_count = counters[0];
  const std::uint32_t pending_tile_count = counters[1];
  const std::uint32_t hash_capacity = tqrj_hash_capacity(counters[2]);
  if (global_thread == 0u) {
    counters[4] = 0u;
    counters[5] = hash_capacity > maximum_hash_entries ? 1u : 0u;
  }
  grid.sync();
  if (counters[5]) return;

  // Bounded retries protect normal inputs from probe clusters.  The final
  // attempt may probe the complete under-full table, preserving exact
  // completion independently of fingerprints and the bounded probe limit.
  for (std::uint32_t attempt = 0u; attempt < kTqrjHashAttempts; ++attempt) {
    if (global_thread == 0u) counters[3] = 0u;
    for (std::uint32_t slot = global_thread; slot < hash_capacity;
         slot += global_stride)
      hash_table[slot] = 0u;
    grid.sync();
    const std::uint64_t seed = tqrj_hash_seed(attempt);
    const std::uint32_t probe_limit =
        attempt + 1u == kTqrjHashAttempts
            ? hash_capacity : kTqrjHashProbeLimit;
    for (std::uint32_t tile_index = blockIdx.x;
         tile_index < query_tile_count; tile_index += gridDim.x) {
      const TqrjHashTile tile = query_tiles[tile_index];
      const TqrjHashTask task = tasks[tile.task];
      const std::uint32_t query_begin = query_bases[task.quotient];
      const std::uint32_t query_count = query_counts[task.quotient];
      for (std::uint32_t local = threadIdx.x; local < kTqrjHashTileRows;
           local += blockDim.x) {
        const std::uint32_t row = tile.begin + local;
        const bool valid = row < query_count;
        const std::uint32_t query_index = query_begin + row;
        const std::uint32_t key = valid ? grouped_queries[query_index] : 0u;
        const unsigned active = __ballot_sync(0xffffffffu, valid);
        if (!valid) continue;
        const unsigned peers = __match_any_sync(active, key);
        const std::uint32_t leader =
            static_cast<std::uint32_t>(__ffs(peers) - 1);
        std::uint32_t owner = kInvalid;
        if (lane == leader)
          owner = tqrj_hash_insert(
              grouped_queries, query_index, hash_table, hash_capacity, seed,
              probe_limit, counters + 3u);
        owner = __shfl_sync(peers, owner, leader);
        // The caller-provided result slot is dead until final emission.  It
        // holds this query's canonical owner so grouped full keys remain
        // available for exact pending probes without another owner array.
        out_values[query_ids[query_index]] = owner;
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

  // One existing 64-bit slot per grouped query becomes winner storage.  The
  // grouped key plane remains unchanged for exact table probes.
  for (std::uint32_t tile_index = blockIdx.x;
       tile_index < query_tile_count; tile_index += gridDim.x) {
    const TqrjHashTile tile = query_tiles[tile_index];
    const TqrjHashTask task = tasks[tile.task];
    const std::uint32_t query_begin = query_bases[task.quotient];
    const std::uint32_t query_count = query_counts[task.quotient];
    for (std::uint32_t local = threadIdx.x; local < kTqrjHashTileRows;
         local += blockDim.x) {
      const std::uint32_t row = tile.begin + local;
      if (row >= query_count) continue;
      const std::uint32_t query_index = query_begin + row;
      winners[query_index] = 0ull;
    }
  }
  grid.sync();

  const std::uint64_t selected_seed = tqrj_hash_seed(counters[4]);
  for (std::uint32_t tile_index = blockIdx.x;
       tile_index < pending_tile_count; tile_index += gridDim.x) {
    const TqrjHashTile tile = pending_tiles[tile_index];
    if (threadIdx.x == 0u) {
      shared_task = tasks[tile.task];
      std::uint32_t prefix = 0u;
      section_prefixes[0] = 0u;
      for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
        const std::size_t oi = std::size_t{batch} *
            (kQuotients + 1u) + shared_task.quotient;
        prefix += raw_offsets[oi + 1u] - raw_offsets[oi];
        section_prefixes[batch + 1u] = prefix;
      }
    }
    __syncthreads();
    for (std::uint32_t local = threadIdx.x; local < kTqrjHashTileRows;
         local += blockDim.x) {
      const std::uint32_t row = tile.begin + local;
      bool matched = false;
      std::uint32_t owner = kInvalid;
      unsigned long long token = 0ull;
      if (row < shared_task.pending_rows) {
        std::uint32_t low = 0u, high = pending_batches;
        while (low + 1u < high) {
          const std::uint32_t middle = (low + high) >> 1u;
          if (section_prefixes[middle] <= row)
            low = middle;
          else
            high = middle;
        }
        const std::uint32_t batch = low;
        const std::size_t oi = std::size_t{batch} *
            (kQuotients + 1u) + shared_task.quotient;
        const std::uint32_t position =
            raw_offsets[oi] + row - section_prefixes[batch];
        const std::uint32_t record = batch * batch_stride + position;
        const std::uint32_t key = raw_keys[record];
        owner = tqrj_hash_find(
            key, grouped_queries, hash_table, hash_capacity, selected_seed);
        matched = owner != kInvalid;
        if (matched) token = tqrj_pending_token(raw_payloads[record]);
      }
      tqrj_warp_atomic_max(matched, owner, token, winners);
    }
    __syncthreads();
  }
  grid.sync();

  if (threadIdx.x == 0u)
    occupied_levels =
        load_query_manifest(query_occupied_level_mask).occupied_level_mask;
  __syncthreads();
  for (std::uint32_t tile_index = blockIdx.x;
       tile_index < query_tile_count; tile_index += gridDim.x) {
    const TqrjHashTile tile = query_tiles[tile_index];
    const TqrjHashTask task = tasks[tile.task];
    const std::uint32_t query_begin = query_bases[task.quotient];
    const std::uint32_t query_count = query_counts[task.quotient];
    for (std::uint32_t local = threadIdx.x; local < kTqrjHashTileRows;
         local += blockDim.x) {
      const std::uint32_t row = tile.begin + local;
      if (row >= query_count) continue;
      const std::uint32_t query_index = query_begin + row;
      const std::uint32_t owner = out_values[query_ids[query_index]];
      const unsigned long long winner = winners[owner];
      const std::uint32_t key = grouped_queries[query_index];
      tqrj_write_result(
          key, query_index, winner, out_values, out_found, arena,
          descriptors, canonical_cell_ranks, occupied_levels, query_ids);
    }
    __syncthreads();
  }
}

__device__ bool first_visible_in_quotient(
    std::uint32_t q, std::uint32_t lower,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, ResidentRows arena,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices, std::uint32_t active_levels,
    std::uint64_t occupied_levels,
    std::uint32_t &result) {
  const std::uint32_t lower_suffix = key_suffix(lower);
  std::uint32_t raw_begin[kBatchesPerEpoch]{}, raw_end[kBatchesPerEpoch]{};
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t oi = std::size_t{batch} * (kQuotients + 1u) + q;
    raw_begin[batch] = raw_offsets[oi];
    raw_end[batch] = raw_offsets[oi + 1u];
  }
  std::uint32_t class_route[kMaximumLevels]{};
  std::uint32_t class_position[kMaximumLevels]{};
  std::uint32_t class_end[kMaximumLevels]{};
  for (std::uint32_t level = 0u; level < active_levels; ++level) {
    if (!level_is_occupied(occupied_levels, level)) continue;
    const RouteHeader header = route_headers[descriptor_index(q, level)];
    std::uint32_t route_index = 0u;
    while (route_index < header.count &&
           route_slices[header.begin + route_index].suffix_end <= lower_suffix)
      ++route_index;
    class_route[level] = route_index;
    while (route_index < header.count) {
      const Descriptor descriptor =
          route_slices[header.begin + route_index].rows;
      const std::uint32_t position = lower_bound_rows(
          arena + descriptor.offset(), descriptor.count(), lower_suffix);
      if (position < descriptor.count()) {
        class_route[level] = route_index;
        class_position[level] = position;
        class_end[level] = descriptor.count();
        break;
      }
      ++route_index;
      class_route[level] = route_index;
    }
  }
  std::uint32_t previous{};
  bool have_previous = false;
  while (true) {
    std::uint32_t minimum = kInvalid;
    bool found = false;
    for (std::uint32_t batch = 0u; batch < pending_batches; ++batch)
      for (std::uint32_t position = raw_begin[batch]; position < raw_end[batch];
           ++position) {
        const std::uint32_t key =
            key_suffix(raw_keys[batch * batch_stride + position]);
        if (key >= lower_suffix && (!have_previous || key > previous) &&
            (!found || key < minimum)) {
          minimum = key;
          found = true;
        }
      }
    for (std::uint32_t level = 0u; level < active_levels; ++level)
      if (level_is_occupied(occupied_levels, level))
      if (class_position[level] < class_end[level]) {
        const RouteHeader header =
            route_headers[descriptor_index(q, level)];
        const Descriptor descriptor = route_slices[
            header.begin + class_route[level]].rows;
        const std::uint32_t key =
            arena[descriptor.offset() + class_position[level]].key;
        if (!found || key < minimum) {
          minimum = key;
          found = true;
        }
      }
    if (!found) return false;

    Row winner{};
    bool have_winner = false;
    for (int batch = int(pending_batches) - 1; batch >= 0; --batch) {
      const std::uint32_t batch_index = static_cast<std::uint32_t>(batch);
      Row candidate{};
      std::uint32_t newest_position{};
      bool matched = false;
      for (std::uint32_t position = raw_begin[batch_index];
           position < raw_end[batch_index]; ++position) {
        const RawAssignment item = load_raw_assignment(
            raw_keys, raw_payloads,
            batch_index * batch_stride + position);
        const std::uint32_t item_position = raw_position(item);
        if (key_suffix(item.key) == minimum &&
            (!matched || item_position > newest_position)) {
          candidate = raw_row(item);
          newest_position = item_position;
          matched = true;
        }
      }
      if (!have_winner && matched) { winner = candidate; have_winner = true; }
    }
    if (!have_winner)
      for (std::uint32_t level = 0u; level < active_levels; ++level) {
        if (!level_is_occupied(occupied_levels, level)) continue;
        if (class_position[level] >= class_end[level]) continue;
        const RouteHeader header =
            route_headers[descriptor_index(q, level)];
        const Descriptor descriptor = route_slices[
            header.begin + class_route[level]].rows;
        const Row row = arena[descriptor.offset() + class_position[level]];
        if (!have_winner && row.key == minimum) {
          winner = row; have_winner = true;
        }
      }
    if (have_winner && (winner.flags & kTombstone) == 0u) {
      result = full_key(q, winner.key);
      return true;
    }
    for (std::uint32_t level = 0u; level < active_levels; ++level)
      if (level_is_occupied(occupied_levels, level))
      if (class_position[level] < class_end[level]) {
        const RouteHeader header =
            route_headers[descriptor_index(q, level)];
        Descriptor descriptor = route_slices[
            header.begin + class_route[level]].rows;
        if (arena[descriptor.offset() + class_position[level]].key == minimum) {
          ++class_position[level];
          while (class_position[level] == class_end[level] &&
                 class_route[level] + 1u < header.count) {
            ++class_route[level];
            descriptor = route_slices[
                header.begin + class_route[level]].rows;
            class_position[level] = 0u;
            class_end[level] = descriptor.count();
          }
        }
      }
    previous = minimum;
    have_previous = true;
  }
}

__global__ void successor_with_pending_kernel(
    const std::uint32_t *queries, std::uint32_t count,
    std::uint32_t *out_keys, const std::uint32_t *raw_keys,
    const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, ResidentRows arena,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint64_t *query_occupied_level_mask) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const DeviceManifestSnapshot manifest =
      load_query_manifest(query_occupied_level_mask);
  const std::uint32_t active_levels = manifest.active_levels;
  const std::uint64_t occupied_levels = manifest.occupied_level_mask;
  const std::uint32_t query = queries[i];
  for (std::uint32_t q = query >> 16u; q < kQuotients; ++q) {
    const std::uint32_t lower = q == (query >> 16u) ? query : q << 16u;
    std::uint32_t result{};
    if (first_visible_in_quotient(
            q, lower, raw_keys, raw_payloads, raw_offsets, batch_stride,
            pending_batches, arena, descriptors, route_headers,
            route_slices, active_levels, occupied_levels, result)) {
      out_keys[i] = result;
      return;
    }
  }
  out_keys[i] = kInvalid;
}


}

// BEGIN INLINED COMPONENT: gpulsm_sparse/pending.cuh

// BEGIN INLINED COMPONENT: gpulsm_sparse/capsule_lifecycle.cuh

// BEGIN INLINED COMPONENT: gpulsm_sparse/exact.cuh

// BEGIN INLINED COMPONENT: gpulsm_sparse/capsule.cuh

// BEGIN INLINED COMPONENT: gpulsm_sparse/common.cuh

#include <utility>

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
// END INLINED COMPONENT: gpulsm_sparse/common.cuh

namespace gpulsm_sparse {

constexpr std::uint16_t kAnchorCapsule = 1u << 1u;
constexpr std::uint16_t kAnchorLengthMask = 7u;
constexpr std::uint16_t kAnchorKeyLengthShift = 2u;
constexpr std::uint16_t kAnchorValueLengthShift = 5u;
constexpr std::uint16_t kAnchorCapsuleRankShift = 2u;
constexpr std::uint16_t kAnchorCapsuleRankMask = 0x0fffu;
constexpr std::uint16_t kAnchorSummaryValid = 1u << 14u;
constexpr std::uint16_t kExactGroupSentinel = 1u << 15u;
constexpr std::uint32_t kCapsulePageRows = 4096u;
constexpr std::uint32_t kPendingAgeMask = (1u << 24u) - 1u;
constexpr std::uint64_t kExactPhysicalBit = std::uint64_t{1u} << 63u;

__host__ __device__ constexpr bool is_exact_physical(
    std::uint64_t physical) {
  return (physical & kExactPhysicalBit) != 0u;
}

__host__ __device__ constexpr std::uint64_t exact_physical_offset(
    std::uint64_t physical) {
  return physical & ~kExactPhysicalBit;
}

__host__ __device__ constexpr bool anchor_is_exact_group(
    const gpulsmopt2_detail::Row &row) {
  return (row.flags & kExactGroupSentinel) != 0u;
}

__host__ __device__ constexpr gpulsmopt2_detail::Row exact_group_row(
    std::uint32_t head, std::uint32_t descriptor) {
  return {descriptor, static_cast<std::uint16_t>(head),
          kExactGroupSentinel};
}

struct CapsuleHeader {
  std::uint64_t key_length{};
  std::uint64_t value_length{};
  std::uint32_t flags{};
  std::uint32_t reserved{};
};

struct CapsuleIndex {
  std::uint64_t address{};
  std::uint32_t rejection{};
  std::uint32_t segment{};
};

struct CapsulePage {
  std::uint64_t physical_page{};
  std::uint32_t index_begin{};
  std::uint32_t index_count{};
};

struct ExactHeadDescriptor {
  std::uint32_t head{};
  std::uint32_t count{};
  std::uint64_t physical_begin{};
};

struct PendingExactHeadDescriptor {
  std::uint32_t head{};
  std::uint32_t count{};
  std::uint32_t ref_begin{};
  std::uint32_t batch_slot{};
};

struct PendingExceptionRef {
  std::uint32_t age{};
  std::uint32_t physical{};
};

struct PendingCapsulePage {
  std::uint32_t batch_slot{};
  std::uint32_t local_page{};
  std::uint32_t index_begin{};
  std::uint32_t index_count{};
  std::uint32_t bitmap[kCapsulePageRows / 32u]{};
  std::uint16_t prefixes[kCapsulePageRows / 32u + 1u]{};
};

static_assert(sizeof(CapsuleHeader) == 24u);
static_assert(sizeof(CapsuleIndex) == 16u);
static_assert(sizeof(CapsulePage) == 16u);
static_assert(sizeof(ExactHeadDescriptor) == 16u);
static_assert(sizeof(PendingExactHeadDescriptor) == 16u);
static_assert(sizeof(PendingExceptionRef) == 8u);

__host__ __device__ constexpr std::uint64_t align_capsule_bytes(
    std::uint64_t bytes) {
  return (bytes + 7u) & ~std::uint64_t{7u};
}

__host__ __device__ constexpr bool capsule_object_bytes(
    std::uint64_t key_bytes, std::uint64_t value_bytes,
    std::uint64_t &result) {
  if (key_bytes > std::numeric_limits<std::uint64_t>::max() -
                      sizeof(CapsuleHeader))
    return false;
  const std::uint64_t prefix = sizeof(CapsuleHeader) + key_bytes;
  if (value_bytes > std::numeric_limits<std::uint64_t>::max() - prefix)
    return false;
  const std::uint64_t raw = prefix + value_bytes;
  if (raw > std::numeric_limits<std::uint64_t>::max() - 7u) return false;
  result = align_capsule_bytes(raw);
  return true;
}

__host__ __device__ constexpr std::uint16_t anchor_length_code(
    std::uint64_t length) {
  return length == 4u ? 0u : static_cast<std::uint16_t>(length + 1u);
}

__host__ __device__ constexpr std::uint64_t anchor_length_from_code(
    std::uint16_t code) {
  return code ? static_cast<std::uint64_t>(code - 1u) : 4u;
}

__host__ __device__ constexpr bool anchor_has_capsule(
    const gpulsmopt2_detail::Row &row) {
  return (row.flags & kAnchorCapsule) != 0u;
}

__host__ __device__ constexpr std::uint32_t anchor_capsule_rank(
    const gpulsmopt2_detail::Row &row) {
  return (row.flags >> kAnchorCapsuleRankShift) & kAnchorCapsuleRankMask;
}

__host__ __device__ constexpr std::uint64_t anchor_key_length(
    const gpulsmopt2_detail::Row &row) {
  return anchor_has_capsule(row) ? 0u : anchor_length_from_code(
      static_cast<std::uint16_t>(
          (row.flags >> kAnchorKeyLengthShift) & kAnchorLengthMask));
}

__host__ __device__ constexpr std::uint64_t anchor_value_length(
    const gpulsmopt2_detail::Row &row) {
  return anchor_has_capsule(row) ? 0u : anchor_length_from_code(
      static_cast<std::uint16_t>(
          (row.flags >> kAnchorValueLengthShift) & kAnchorLengthMask));
}

__host__ __device__ constexpr std::uint16_t inline_anchor_flags(
    std::uint64_t key_length, std::uint64_t value_length,
    bool tombstone, bool summary) {
  return static_cast<std::uint16_t>(
      (tombstone ? gpulsmopt2_detail::kTombstone : 0u) |
      (anchor_length_code(key_length) << kAnchorKeyLengthShift) |
      (anchor_length_code(value_length) << kAnchorValueLengthShift) |
      (summary ? kAnchorSummaryValid : 0u));
}

__host__ __device__ constexpr std::uint16_t capsule_anchor_flags(
    std::uint32_t rank, bool tombstone, bool summary) {
  return static_cast<std::uint16_t>(
      (tombstone ? gpulsmopt2_detail::kTombstone : 0u) |
      kAnchorCapsule |
      ((rank & kAnchorCapsuleRankMask) << kAnchorCapsuleRankShift) |
      (summary ? kAnchorSummaryValid : 0u));
}

class CapsuleSegment {
 public:
  CapsuleSegment() = default;
  explicit CapsuleSegment(std::uint64_t bytes) { allocate(bytes); }
  CapsuleSegment(const CapsuleSegment &) = delete;
  CapsuleSegment &operator=(const CapsuleSegment &) = delete;
  CapsuleSegment(CapsuleSegment &&other) noexcept { move_from(other); }
  CapsuleSegment &operator=(CapsuleSegment &&other) noexcept {
    if (this != &other) {
      release();
      move_from(other);
    }
    return *this;
  }
  ~CapsuleSegment() { release(); }

  void allocate(std::uint64_t bytes) {
    if (!bytes) return;
    if (address_) throw std::logic_error("capsule already allocated");
    check(cudaFree(nullptr), "initialize CUDA for capsule VMM");
    int device = 0;
    check(cudaGetDevice(&device), "get capsule device");
    property_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    property_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    property_.location.id = device;
    auto &functions = gpulsmopt2_detail::vmm_functions();
    GPULSMOPT_CU_CHECK(functions.granularity(
        &granularity_, &property_, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    reserved_bytes_ = align(bytes, granularity_);
    GPULSMOPT_CU_CHECK(functions.reserve(
        &address_, reserved_bytes_, granularity_, 0u, 0u));
    try {
      constexpr std::uint64_t chunk_limit = std::uint64_t{1u} << 30u;
      std::uint64_t offset = 0u;
      while (offset < reserved_bytes_) {
        const std::uint64_t chunk = align(
            std::min(chunk_limit, reserved_bytes_ - offset), granularity_);
        CUmemGenericAllocationHandle handle{};
        GPULSMOPT_CU_CHECK(functions.create(
            &handle, chunk, &property_, 0u));
        bool mapped = false;
        try {
          GPULSMOPT_CU_CHECK(functions.map(
              address_ + offset, chunk, 0u, handle, 0u));
          mapped = true;
          CUmemAccessDesc access{};
          access.location = property_.location;
          access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
          GPULSMOPT_CU_CHECK(functions.set_access(
              address_ + offset, chunk, &access, 1u));
        } catch (...) {
          if (mapped) functions.unmap(address_ + offset, chunk);
          functions.release(handle);
          throw;
        }
        mappings_.push_back({offset, chunk, handle});
        mapped_bytes_ += chunk;
        offset += chunk;
      }
      live_bytes_ = bytes;
    } catch (...) {
      release();
      throw;
    }
  }

  std::uint8_t *data() const {
    return reinterpret_cast<std::uint8_t *>(
        static_cast<std::uintptr_t>(address_));
  }
  std::uint64_t address() const { return address_; }
  std::uint64_t reserved_bytes() const { return reserved_bytes_; }
  std::uint64_t mapped_bytes() const { return mapped_bytes_; }
  std::uint64_t live_bytes() const { return live_bytes_; }
  std::uint64_t garbage_bytes() const { return garbage_bytes_; }
  void set_usage(std::uint64_t live, std::uint64_t garbage) {
    if (live > mapped_bytes_ || garbage > mapped_bytes_ - live)
      throw std::overflow_error("invalid capsule usage");
    live_bytes_ = live;
    garbage_bytes_ = garbage;
  }
  void commit_usage(std::uint64_t live, std::uint64_t garbage) noexcept {
    live_bytes_ = live;
    garbage_bytes_ = garbage;
  }

 private:
  struct Mapping {
    std::uint64_t offset{};
    std::uint64_t bytes{};
    CUmemGenericAllocationHandle handle{};
  };

  static std::uint64_t align(std::uint64_t value,
                             std::uint64_t alignment) {
    if (value > std::numeric_limits<std::uint64_t>::max() - alignment + 1u)
      throw std::overflow_error("capsule size overflow");
    return (value + alignment - 1u) / alignment * alignment;
  }

  void move_from(CapsuleSegment &other) noexcept {
    address_ = other.address_;
    granularity_ = other.granularity_;
    reserved_bytes_ = other.reserved_bytes_;
    mapped_bytes_ = other.mapped_bytes_;
    live_bytes_ = other.live_bytes_;
    garbage_bytes_ = other.garbage_bytes_;
    property_ = other.property_;
    mappings_ = std::move(other.mappings_);
    other.address_ = 0u;
    other.granularity_ = 0u;
    other.reserved_bytes_ = 0u;
    other.mapped_bytes_ = 0u;
    other.live_bytes_ = 0u;
    other.garbage_bytes_ = 0u;
  }

  void release() noexcept {
    if (!address_) return;
    auto &functions = gpulsmopt2_detail::vmm_functions();
    for (auto it = mappings_.rbegin(); it != mappings_.rend(); ++it) {
      functions.unmap(address_ + it->offset, it->bytes);
      functions.release(it->handle);
    }
    functions.free_address(address_, reserved_bytes_);
    address_ = 0u;
    mappings_.clear();
    granularity_ = reserved_bytes_ = mapped_bytes_ = 0u;
    live_bytes_ = garbage_bytes_ = 0u;
  }

  CUdeviceptr address_{};
  std::uint64_t granularity_{};
  std::uint64_t reserved_bytes_{};
  std::uint64_t mapped_bytes_{};
  std::uint64_t live_bytes_{};
  std::uint64_t garbage_bytes_{};
  CUmemAllocationProp property_{};
  std::vector<Mapping> mappings_;
};

struct CapsuleSegmentOwnership {
  std::shared_ptr<CapsuleSegment> segment;
  std::uint32_t ordinal{};
  std::uint32_t reserved{};
  std::uint64_t live_bytes{};
  std::uint64_t garbage_bytes{};
};

struct CapsuleTransferIntent {
  std::uint32_t ordinal{};
  std::uint32_t reserved{};
  std::uint64_t live_bytes{};
  std::uint64_t garbage_bytes{};
  CapsuleSegmentOwnership *source{};
};

struct CapsuleLifecycleSource {
  std::uint32_t ordinal{};
  std::uint32_t action{};
  std::uint64_t live_bytes{};
  std::uint64_t garbage_bytes{};
  std::uint64_t survivor_bytes{};
};

constexpr std::uint32_t kCapsuleCopy = 0u;
constexpr std::uint32_t kCapsuleTransfer = 1u;
constexpr std::uint32_t kCapsuleCompact = 2u;

struct DeviceCapsulePlane {
  const CapsulePage *pages{};
  const CapsuleIndex *indexes{};
  std::uint32_t page_count{};
  std::uint32_t capsule_count{};
};

struct StagedRootOverlay {
  RootBuildState state{};
  Buffer<RootBuildState> device_state;
  Buffer<ExactHeadDescriptor> exact_heads;
  Buffer<gpulsmopt2_detail::Row> exact_rows;
  Buffer<std::uint32_t> special_heads;
  Buffer<CapsulePage> pages;
  Buffer<CapsuleIndex> indexes;
  std::vector<CapsuleSegmentOwnership> segments;
  std::vector<CapsuleTransferIntent> transfers;
  std::uint64_t live_bytes{};
  std::uint64_t garbage_bytes{};
};

struct StagedPendingSlotOverlay {
  PendingSlotBuildState state{};
  Buffer<PendingCapsulePage> pages;
  Buffer<CapsuleIndex> indexes;
  Buffer<PendingExactHeadDescriptor> exact_heads;
  Buffer<std::uint32_t> exact_refs;
  Buffer<std::uint32_t> special_heads;
  Buffer<PendingExceptionRef> exceptions;
  std::vector<CapsuleSegmentOwnership> segments;
  std::uint64_t live_bytes{};
  std::uint64_t garbage_bytes{};
};

__device__ __forceinline__ const CapsuleIndex *find_pending_capsule(
    const PendingCapsulePage *pages, std::uint32_t page_count,
    const CapsuleIndex *indexes, std::uint32_t batch_slot,
    std::uint32_t local_record) {
  const std::uint32_t local_page = local_record / kCapsulePageRows;
  const std::uint64_t target =
      std::uint64_t{batch_slot} * (1u << 20u) + local_page;
  std::uint32_t low = 0u;
  std::uint32_t high = page_count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    const std::uint64_t found =
        std::uint64_t{pages[middle].batch_slot} * (1u << 20u) +
        pages[middle].local_page;
    if (found < target) low = middle + 1u;
    else high = middle;
  }
  if (low == page_count || pages[low].batch_slot != batch_slot ||
      pages[low].local_page != local_page)
    return nullptr;
  const std::uint32_t local = local_record % kCapsulePageRows;
  const std::uint32_t word = local >> 5u;
  const std::uint32_t bit = local & 31u;
  if (!(pages[low].bitmap[word] & (1u << bit))) return nullptr;
  const std::uint32_t before = bit ? ((1u << bit) - 1u) : 0u;
  const std::uint32_t rank = pages[low].prefixes[word] +
      __popc(pages[low].bitmap[word] & before);
  return indexes + pages[low].index_begin + rank;
}

__device__ __forceinline__ const PendingExceptionRef *
find_pending_exception(
    const PendingExceptionRef *exceptions, std::uint32_t count,
    std::uint32_t age) {
  std::uint32_t low = 0u;
  std::uint32_t high = count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (exceptions[middle].age < age) low = middle + 1u;
    else high = middle;
  }
  return low < count && exceptions[low].age == age
      ? exceptions + low : nullptr;
}

__device__ __forceinline__ const CapsuleIndex *find_pending_capsule(
    const PendingSlotBuildState &slot, std::uint32_t batch_slot,
    std::uint32_t local_record) {
  return find_pending_capsule(
      reinterpret_cast<const PendingCapsulePage *>(slot.pages),
      slot.page_count, reinterpret_cast<const CapsuleIndex *>(slot.indexes),
      batch_slot, local_record);
}

constexpr std::uint32_t kCompletionIncoming = 1u << 31u;
constexpr std::uint32_t kCompletionSlotShift =
    gpulsmopt2_detail::kBatchPositionBits;
constexpr std::uint32_t kCompletionLocalMask =
    (1u << gpulsmopt2_detail::kBatchPositionBits) - 1u;

__device__ __forceinline__ const CapsuleHeader *capsule_header(
    const CapsuleIndex *index);

struct CompletionSourceView {
  RecordBatchView incoming{};
  // Direct root construction supplies the prototype's stable head-sorted
  // input order.  Sparse refinement can then gather only roster heads; it
  // never creates a second full incoming record bank.
  const std::uint32_t *incoming_sorted_heads{};
  const std::uint32_t *incoming_sorted_refs{};
  std::uint32_t incoming_records{};
  const std::uint32_t *pending_keys{};
  const gpulsmopt2_detail::RawPayload *pending_payloads{};
  const std::uint32_t *pending_offsets{};
  const PendingSlotBuildState *pending_slots{};
  std::uint32_t batch_capacity{};
  std::uint32_t pending_records{};
};

__host__ __device__ constexpr bool completion_is_incoming(
    std::uint32_t ref) {
  return (ref & kCompletionIncoming) != 0u;
}

__host__ __device__ constexpr std::uint32_t completion_slot(
    std::uint32_t ref) {
  return (ref >> kCompletionSlotShift) & 15u;
}

__host__ __device__ constexpr std::uint32_t completion_local(
    std::uint32_t ref) {
  return ref & kCompletionLocalMask;
}

struct CompletionKeyCursor {
  CompletionSourceView source{};
  std::uint32_t ref{};

  __device__ std::uint64_t pending_physical() const {
    return std::uint64_t{completion_slot(ref)} * source.batch_capacity +
        completion_local(ref);
  }

  __device__ gpulsmopt2_detail::RawPayload pending_payload() const {
    return source.pending_payloads[pending_physical()];
  }

  __device__ const CapsuleIndex *pending_capsule() const {
    if (completion_is_incoming(ref)) return nullptr;
    const std::uint32_t slot = completion_slot(ref);
    return find_pending_capsule(
        source.pending_slots[slot], slot, completion_local(ref));
  }

  __device__ const CapsuleHeader *pending_header() const {
    return capsule_header(pending_capsule());
  }

  __device__ std::uint64_t length() const {
    if (completion_is_incoming(ref))
      return DeviceKeyCursor{
          source.incoming.keys, ref & ~kCompletionIncoming,
          source.incoming.head4_words}.length();
    const CapsuleHeader *object = pending_header();
    return object ? object->key_length : 4u;
  }

  __device__ std::uint8_t byte(std::uint64_t position) const {
    if (completion_is_incoming(ref))
      return DeviceKeyCursor{
          source.incoming.keys, ref & ~kCompletionIncoming,
          source.incoming.head4_words}.byte(position);
    const CapsuleHeader *object = pending_header();
    if (object)
      return reinterpret_cast<const std::uint8_t *>(object + 1u)[position];
    const std::uint32_t head = source.pending_keys[pending_physical()];
    return static_cast<std::uint8_t>(
        head >> (24u - static_cast<std::uint32_t>(position) * 8u));
  }

  __device__ std::uint32_t head4() const {
    if (completion_is_incoming(ref))
      return DeviceKeyCursor{
          source.incoming.keys, ref & ~kCompletionIncoming,
          source.incoming.head4_words}.head4();
    return source.pending_keys[pending_physical()];
  }
};

struct CompletionValueCursor {
  CompletionSourceView source{};
  std::uint32_t ref{};

  __device__ std::uint64_t pending_physical() const {
    return std::uint64_t{completion_slot(ref)} * source.batch_capacity +
        completion_local(ref);
  }

  __device__ const CapsuleHeader *pending_header() const {
    const std::uint32_t slot = completion_slot(ref);
    return capsule_header(find_pending_capsule(
        source.pending_slots[slot], slot, completion_local(ref)));
  }

  __device__ std::uint64_t length() const {
    if (completion_is_incoming(ref))
      return DeviceValueCursor{
          source.incoming.values, ref & ~kCompletionIncoming}.length();
    const auto payload = source.pending_payloads[pending_physical()];
    if (payload.metadata & gpulsmopt2_detail::kRawTombstone) return 0u;
    const CapsuleHeader *object = pending_header();
    return object ? object->value_length : 4u;
  }

  __device__ std::uint8_t byte(std::uint64_t position) const {
    if (completion_is_incoming(ref))
      return DeviceValueCursor{
          source.incoming.values, ref & ~kCompletionIncoming}.byte(position);
    const CapsuleHeader *object = pending_header();
    if (object) {
      const auto *bytes = reinterpret_cast<const std::uint8_t *>(object + 1u);
      return bytes[object->key_length + position];
    }
    const std::uint32_t value =
        source.pending_payloads[pending_physical()].value;
    return static_cast<std::uint8_t>(value >> (8u * position));
  }

  __device__ std::uint32_t inline_word() const {
    if (!completion_is_incoming(ref)) {
      const CapsuleHeader *object = pending_header();
      if (!object)
        return source.pending_payloads[pending_physical()].value;
      std::uint32_t value = 0u;
      const std::uint64_t size = object->value_length < 4u
          ? object->value_length : 4u;
      for (std::uint32_t position = 0u; position < size; ++position)
        value |= std::uint32_t{byte(position)} << (8u * position);
      return value;
    }
    return DeviceValueCursor{
        source.incoming.values, ref & ~kCompletionIncoming}.inline_word();
  }
};

__host__ __device__ inline DeviceKeyCursor source_key(
    RecordBatchView source, std::uint32_t ref) {
  return {source.keys, ref, source.head4_words};
}

__device__ inline CompletionKeyCursor source_key(
    CompletionSourceView source, std::uint32_t ref) {
  return {source, ref};
}

__host__ __device__ inline DeviceValueCursor source_value(
    RecordBatchView source, std::uint32_t ref) {
  return {source.values, ref};
}

__device__ inline CompletionValueCursor source_value(
    CompletionSourceView source, std::uint32_t ref) {
  return {source, ref};
}

__host__ __device__ inline bool source_tombstone(
    RecordBatchView source, std::uint32_t ref) {
  return record_is_tombstone(source, ref);
}

__device__ inline bool source_tombstone(
    CompletionSourceView source, std::uint32_t ref) {
  if (completion_is_incoming(ref))
    return record_is_tombstone(source.incoming, ref & ~kCompletionIncoming);
  const std::uint64_t physical =
      std::uint64_t{completion_slot(ref)} * source.batch_capacity +
      completion_local(ref);
  return (source.pending_payloads[physical].metadata &
          gpulsmopt2_detail::kRawTombstone) != 0u;
}

__host__ __device__ inline std::uint32_t source_summary(
    RecordBatchView source, std::uint32_t ref) {
  return source.range_contributions ? source.range_contributions[ref]
      : source_value(source, ref).inline_word();
}

__device__ inline std::uint32_t source_summary(
    CompletionSourceView source, std::uint32_t ref) {
  if (completion_is_incoming(ref)) {
    const std::uint32_t row = ref & ~kCompletionIncoming;
    return source.incoming.range_contributions
        ? source.incoming.range_contributions[row]
        : source_value(source, ref).inline_word();
  }
  const std::uint64_t physical =
      std::uint64_t{completion_slot(ref)} * source.batch_capacity +
      completion_local(ref);
  return source.pending_payloads[physical].value;
}

__host__ __device__ inline const CapsuleIndex *source_owned_capsule(
    RecordBatchView, std::uint32_t) {
  return nullptr;
}

__device__ inline const CapsuleIndex *source_owned_capsule(
    CompletionSourceView source, std::uint32_t ref) {
  return CompletionKeyCursor{source, ref}.pending_capsule();
}

__device__ __forceinline__ const CapsuleIndex *find_capsule(
    DeviceCapsulePlane plane, std::uint64_t physical_row,
    std::uint32_t rank) {
  const std::uint64_t page = physical_row / kCapsulePageRows;
  std::uint32_t low = 0u;
  std::uint32_t high = plane.page_count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (plane.pages[middle].physical_page < page) low = middle + 1u;
    else high = middle;
  }
  if (low == plane.page_count || plane.pages[low].physical_page != page ||
      rank >= plane.pages[low].index_count)
    return nullptr;
  return plane.indexes + plane.pages[low].index_begin + rank;
}

__device__ __forceinline__ const CapsuleHeader *capsule_header(
    const CapsuleIndex *index) {
  return index ? reinterpret_cast<const CapsuleHeader *>(index->address)
               : nullptr;
}

struct ResidentKeyCursor {
  std::uint32_t head{};
  std::uint64_t physical_row{};
  gpulsmopt2_detail::Row row{};
  DeviceCapsulePlane plane{};

  __device__ const CapsuleIndex *capsule() const {
    return anchor_has_capsule(row)
        ? find_capsule(plane, physical_row, anchor_capsule_rank(row))
        : nullptr;
  }
  __device__ const CapsuleHeader *header() const {
    return capsule_header(capsule());
  }
  __device__ std::uint32_t head4() const { return head; }
  __device__ std::uint64_t length() const {
    const CapsuleHeader *object = header();
    return object ? object->key_length : anchor_key_length(row);
  }
  __device__ std::uint8_t byte(std::uint64_t position) const {
    const CapsuleHeader *object = header();
    if (object)
      return reinterpret_cast<const std::uint8_t *>(object + 1u)[position];
    return static_cast<std::uint8_t>(
        head >> (24u - static_cast<std::uint32_t>(position) * 8u));
  }
};

constexpr std::uint32_t kTerminalIncomingSource = 0xffffffffu;

struct TerminalCandidate {
  std::uint64_t locator{};
  std::uint32_t head{};
  std::uint32_t source{kTerminalIncomingSource};
};

static_assert(sizeof(TerminalCandidate) == 16u);

template <class IncomingSource>
struct TerminalSourceView {
  IncomingSource incoming{};
  gpulsmopt2_detail::ResidentRows arena{};
  const RootBuildState *roots{};
  const TerminalCandidate *candidates{};
};

template <class IncomingSource>
struct TerminalKeyCursor {
  TerminalSourceView<IncomingSource> source{};
  std::uint32_t ref{};

  __device__ const TerminalCandidate &candidate() const {
    return source.candidates[ref];
  }
  __device__ bool incoming() const {
    return candidate().source == kTerminalIncomingSource;
  }
  __device__ DeviceCapsulePlane plane() const {
    if (incoming()) return {};
    const RootBuildState &root = source.roots[candidate().source];
    return {reinterpret_cast<const CapsulePage *>(root.capsule_pages),
            reinterpret_cast<const CapsuleIndex *>(root.capsule_indexes),
            root.capsule_page_count, root.capsule_count};
  }
  __device__ gpulsmopt2_detail::Row row() const {
    if (!is_exact_physical(candidate().locator))
      return source.arena[candidate().locator];
    const RootBuildState &root = source.roots[candidate().source];
    const auto *rows = reinterpret_cast<const gpulsmopt2_detail::Row *>(
        root.exact_rows);
    return rows[exact_physical_offset(candidate().locator)];
  }
  __device__ ResidentKeyCursor resident() const {
    return {candidate().head, candidate().locator, row(), plane()};
  }
  __device__ std::uint32_t head4() const {
    return candidate().head;
  }
  __device__ std::uint64_t length() const {
    return incoming()
        ? source_key(source.incoming,
                     static_cast<std::uint32_t>(candidate().locator)).length()
        : resident().length();
  }
  __device__ std::uint8_t byte(std::uint64_t position) const {
    return incoming()
        ? source_key(source.incoming,
                     static_cast<std::uint32_t>(candidate().locator))
              .byte(position)
        : resident().byte(position);
  }
  __device__ const CapsuleIndex *capsule() const {
    return incoming()
        ? source_owned_capsule(
              source.incoming,
              static_cast<std::uint32_t>(candidate().locator))
        : resident().capsule();
  }
};

template <class IncomingSource>
struct TerminalValueCursor {
  TerminalSourceView<IncomingSource> source{};
  std::uint32_t ref{};

  __device__ const TerminalCandidate &candidate() const {
    return source.candidates[ref];
  }
  __device__ bool incoming() const {
    return candidate().source == kTerminalIncomingSource;
  }
  __device__ const CapsuleHeader *resident_header() const {
    const RootBuildState &root = source.roots[candidate().source];
    const auto row = TerminalKeyCursor<IncomingSource>{source, ref}.row();
    const DeviceCapsulePlane plane{
        reinterpret_cast<const CapsulePage *>(root.capsule_pages),
        reinterpret_cast<const CapsuleIndex *>(root.capsule_indexes),
        root.capsule_page_count, root.capsule_count};
    return capsule_header(anchor_has_capsule(row)
        ? find_capsule(plane, candidate().locator, anchor_capsule_rank(row))
        : nullptr);
  }
  __device__ std::uint64_t length() const {
    if (incoming())
      return source_value(
          source.incoming,
          static_cast<std::uint32_t>(candidate().locator)).length();
    const auto row = TerminalKeyCursor<IncomingSource>{source, ref}.row();
    if (row.flags & gpulsmopt2_detail::kTombstone) return 0u;
    const CapsuleHeader *object = resident_header();
    return object ? object->value_length : anchor_value_length(row);
  }
  __device__ std::uint8_t byte(std::uint64_t position) const {
    if (incoming())
      return source_value(
          source.incoming,
          static_cast<std::uint32_t>(candidate().locator)).byte(position);
    const CapsuleHeader *object = resident_header();
    if (object) {
      const auto *bytes = reinterpret_cast<const std::uint8_t *>(object + 1u);
      return bytes[object->key_length + position];
    }
    return static_cast<std::uint8_t>(
        TerminalKeyCursor<IncomingSource>{source, ref}.row().value >>
        (8u * position));
  }
  __device__ std::uint32_t inline_word() const {
    if (incoming())
      return source_value(
          source.incoming,
          static_cast<std::uint32_t>(candidate().locator)).inline_word();
    const CapsuleHeader *object = resident_header();
    if (!object)
      return TerminalKeyCursor<IncomingSource>{source, ref}.row().value;
    std::uint32_t result = 0u;
    const std::uint64_t count = object->value_length < 4u
        ? object->value_length : 4u;
    for (std::uint32_t position = 0u; position < count; ++position)
      result |= std::uint32_t{byte(position)} << (8u * position);
    return result;
  }
};

template <class IncomingSource>
__device__ inline TerminalKeyCursor<IncomingSource> source_key(
    TerminalSourceView<IncomingSource> source, std::uint32_t ref) {
  return {source, ref};
}

template <class IncomingSource>
__device__ inline TerminalValueCursor<IncomingSource> source_value(
    TerminalSourceView<IncomingSource> source, std::uint32_t ref) {
  return {source, ref};
}

template <class IncomingSource>
__device__ inline bool source_tombstone(
    TerminalSourceView<IncomingSource> source, std::uint32_t ref) {
  const TerminalCandidate &candidate = source.candidates[ref];
  return candidate.source == kTerminalIncomingSource
      ? source_tombstone(source.incoming,
                         static_cast<std::uint32_t>(candidate.locator))
      : (TerminalKeyCursor<IncomingSource>{source, ref}.row().flags &
         gpulsmopt2_detail::kTombstone) != 0u;
}

template <class IncomingSource>
__device__ inline std::uint32_t source_summary(
    TerminalSourceView<IncomingSource> source, std::uint32_t ref) {
  const TerminalCandidate &candidate = source.candidates[ref];
  return candidate.source == kTerminalIncomingSource
      ? source_summary(source.incoming,
                       static_cast<std::uint32_t>(candidate.locator))
      : TerminalKeyCursor<IncomingSource>{source, ref}.row().value;
}

template <class IncomingSource>
__device__ inline const CapsuleIndex *source_owned_capsule(
    TerminalSourceView<IncomingSource> source, std::uint32_t ref) {
  return TerminalKeyCursor<IncomingSource>{source, ref}.capsule();
}

__device__ __forceinline__ bool exact_key_equal(
    const DeviceKeyCursor &query, const ResidentKeyCursor &stored) {
  if (query.length() != stored.length()) return false;
  if (query.length() && rejection_hash(query) !=
      (stored.capsule() ? stored.capsule()->rejection
                        : rejection_hash(query)))
    return false;
  for (std::uint64_t position = 0u; position < query.length(); ++position)
    if (query.byte(position) != stored.byte(position)) return false;
  return true;
}

}  // namespace gpulsm_sparse
// END INLINED COMPONENT: gpulsm_sparse/capsule.cuh

namespace gpulsm_sparse {

constexpr std::uint32_t kContinuationTokenBits = 36u;
constexpr std::uint64_t kContinuationTokenMask =
    (std::uint64_t{1u} << kContinuationTokenBits) - 1u;

template <class Source>
__global__ void initialize_universal_records(
    Source source, const std::uint32_t *prepared_heads,
    const std::uint32_t *prepared_refs, std::uint32_t *heads,
    std::uint32_t *sort_heads, std::uint32_t *refs,
    std::uint32_t rows) {
  const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const std::uint32_t ref = prepared_refs ? prepared_refs[row] : row;
  const std::uint32_t head = prepared_heads
      ? prepared_heads[row] : source_key(source, ref).head4();
  heads[row] = head;
  sort_heads[row] = head;
  refs[row] = ref;
}

template <class Source>
__global__ void classify_primary_groups(
    Source source, const std::uint32_t *sorted_refs,
    const std::uint32_t *group_counts,
    const std::uint32_t *group_starts,
    const std::uint32_t *group_count,
    std::uint32_t *active_rows, std::uint32_t *active_groups,
    std::uint32_t *group_depths) {
  const std::uint32_t group = blockIdx.x;
  if (group >= *group_count) return;
  const std::uint32_t begin = group_starts[group];
  const std::uint32_t count = group_counts[group];
  __shared__ std::uint32_t ambiguous;
  __shared__ std::uint32_t short_key;
  if (threadIdx.x == 0u) {
    ambiguous = 0u;
    short_key = 0u;
  }
  __syncthreads();
  for (std::uint32_t local = threadIdx.x; local < count;
       local += blockDim.x) {
    const std::uint32_t ref = sorted_refs[begin + local];
    const std::uint64_t length = source_key(source, ref).length();
    if (length != 4u || source_tombstone(source, ref))
      atomicExch(&ambiguous, 1u);
    if (length < 4u) atomicExch(&short_key, 1u);
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    active_rows[group] = ambiguous ? count : 0u;
    active_groups[group] = ambiguous;
    group_depths[group] = short_key ? 0u : 4u;
  }
}

__global__ void finish_scan_total(
    const std::uint32_t *values, const std::uint32_t *offsets,
    const std::uint32_t *count, std::uint32_t *total) {
  if (blockIdx.x || threadIdx.x) return;
  const std::uint32_t size = *count;
  *total = size ? offsets[size - 1u] + values[size - 1u] : 0u;
}

__global__ void initialize_exact_tasks(
    const std::uint32_t *sorted_refs,
    const std::uint32_t *group_counts,
    const std::uint32_t *group_starts,
    const std::uint32_t *group_count,
    const std::uint32_t *active_offsets,
    const std::uint32_t *active_group_ids,
    const std::uint32_t *active_groups,
    const std::uint32_t *group_depths,
    std::uint32_t *active_refs, std::uint32_t *active_task_ids,
    std::uint32_t *ordered_refs, std::uint32_t *task_begins,
    std::uint32_t *task_bases, std::uint32_t *task_depths) {
  const std::uint32_t group = blockIdx.x;
  if (group >= *group_count) return;
  const std::uint32_t begin = group_starts[group];
  const std::uint32_t count = group_counts[group];
  if (!active_groups[group]) {
    for (std::uint32_t local = threadIdx.x; local < count;
         local += blockDim.x)
      ordered_refs[begin + local] = sorted_refs[begin + local];
    return;
  }
  const std::uint32_t output = active_offsets[group];
  const std::uint32_t task = active_group_ids[group];
  for (std::uint32_t local = threadIdx.x; local < count;
       local += blockDim.x) {
    active_refs[output + local] = sorted_refs[begin + local];
    active_task_ids[output + local] = task;
  }
  if (threadIdx.x == 0u) {
    task_begins[task] = output;
    task_bases[task] = begin;
    task_depths[task] = group_depths[group];
  }
}

template <class Source>
__global__ void make_continuation_composites(
    Source source, const std::uint32_t *refs,
    const std::uint32_t *task_ids,
    const std::uint32_t *task_depths,
    std::uint64_t *composites, std::uint32_t active_rows) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= active_rows) return;
  const std::uint32_t ref = refs[position];
  const std::uint32_t task = task_ids[position];
  const std::uint32_t depth = task_depths[task];
  const auto key = source_key(source, ref);
  std::uint64_t token = 0u;
  for (std::uint32_t symbol = 0u; symbol < 4u; ++symbol) {
    token <<= 9u;
    const std::uint64_t byte = std::uint64_t{depth} + symbol;
    if (byte < key.length()) token |= std::uint32_t{key.byte(byte)} + 1u;
  }
  composites[position] =
      (std::uint64_t{task} << kContinuationTokenBits) | token;
}

__device__ __forceinline__ bool continuation_token_has_end(
    std::uint64_t token) {
  for (std::uint32_t symbol = 0u; symbol < 4u; ++symbol) {
    if (!(token & 0x1ffu)) return true;
    token >>= 9u;
  }
  return false;
}

__global__ void classify_continuation_groups(
    const std::uint64_t *unique_composites,
    const std::uint32_t *group_counts,
    const std::uint32_t *group_count,
    std::uint32_t *continuation_rows,
    std::uint32_t *continuation_groups) {
  const std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= *group_count) return;
  const std::uint64_t token =
      unique_composites[group] & kContinuationTokenMask;
  const bool continuation =
      group_counts[group] > 1u && !continuation_token_has_end(token);
  continuation_rows[group] = continuation ? group_counts[group] : 0u;
  continuation_groups[group] = continuation;
}

__global__ void scatter_continuation_groups(
    const std::uint64_t *unique_composites,
    const std::uint32_t *sorted_refs,
    const std::uint32_t *group_counts,
    const std::uint32_t *group_starts,
    const std::uint32_t *group_count,
    const std::uint32_t *continuation_offsets,
    const std::uint32_t *continuation_group_ids,
    const std::uint32_t *continuation_groups,
    const std::uint32_t *task_begins,
    const std::uint32_t *task_bases,
    const std::uint32_t *task_depths,
    std::uint32_t *next_refs, std::uint32_t *next_task_ids,
    std::uint32_t *ordered_refs,
    std::uint32_t *next_task_begins,
    std::uint32_t *next_task_bases,
    std::uint32_t *next_task_depths) {
  const std::uint32_t group = blockIdx.x;
  if (group >= *group_count) return;
  const std::uint32_t count = group_counts[group];
  const std::uint32_t begin = group_starts[group];
  const std::uint32_t parent = static_cast<std::uint32_t>(
      unique_composites[group] >> kContinuationTokenBits);
  const std::uint32_t base =
      task_bases[parent] + begin - task_begins[parent];
  if (!continuation_groups[group]) {
    for (std::uint32_t local = threadIdx.x; local < count;
         local += blockDim.x)
      ordered_refs[base + local] = sorted_refs[begin + local];
    return;
  }
  const std::uint32_t output = continuation_offsets[group];
  const std::uint32_t child = continuation_group_ids[group];
  for (std::uint32_t local = threadIdx.x; local < count;
       local += blockDim.x) {
    next_refs[output + local] = sorted_refs[begin + local];
    next_task_ids[output + local] = child;
  }
  if (threadIdx.x == 0u) {
    next_task_begins[child] = output;
    next_task_bases[child] = base;
    next_task_depths[child] = task_depths[parent] + 4u;
  }
}

template <class Source>
__global__ void mark_last_universal_key(
    Source source, const std::uint32_t *ordered,
    std::uint8_t *last, std::uint32_t rows) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= rows) return;
  bool keep = position + 1u == rows;
  if (!keep) {
    const auto left = source_key(source, ordered[position]);
    const auto right = source_key(source, ordered[position + 1u]);
    keep = compare_keys(left, right) != 0;
  }
  last[position] = static_cast<std::uint8_t>(keep);
}

template <class Source>
__global__ void classify_winners(
    Source source, const std::uint32_t *winners,
    const std::uint32_t *winner_count,
    const std::uint32_t *unique_heads,
    const std::uint32_t *primary_group_flags,
    const std::uint32_t *primary_group_count,
    std::uint8_t *exact_flags, std::uint8_t *capsule_flags) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count = *winner_count;
  if (position >= count) return;
  const std::uint32_t ref = winners[position];
  const auto key = source_key(source, ref);
  const auto value = source_value(source, ref);
  const std::uint32_t head = key.head4();
  std::uint32_t low = 0u;
  std::uint32_t high = *primary_group_count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (unique_heads[middle] < head) low = middle + 1u;
    else high = middle;
  }
  exact_flags[position] =
      low < *primary_group_count && unique_heads[low] == head
          ? static_cast<std::uint8_t>(primary_group_flags[low] != 0u)
          : 0u;
  const bool tombstone = source_tombstone(source, ref);
  const std::uint32_t inline_value = value.inline_word();
  const std::uint32_t summary = source_summary(source, ref);
  capsule_flags[position] = static_cast<std::uint8_t>(
      key.length() != 4u || (!tombstone && value.length() != 4u) ||
      (!tombstone && summary != inline_value));
}

struct ExactOrderResult {
  std::uint32_t winner_count{};
  std::uint32_t primary_groups{};
  std::uint32_t active_rows{};
  std::uint32_t rounds{};
};

class ExactOrderWorkspace {
 public:
  explicit ExactOrderWorkspace(std::uint32_t capacity)
      : capacity_(capacity), heads_(capacity), sort_heads_a_(capacity),
        sort_heads_b_(capacity), refs_a_(capacity), refs_b_(capacity),
        task_ids_a_(capacity),
        unique_heads_(capacity), primary_counts_(capacity),
        primary_starts_(capacity), primary_count_(1u),
        primary_rows_(capacity), primary_flags_(capacity),
        primary_depths_(capacity), primary_offsets_(capacity),
        primary_ids_(capacity), composites_a_(capacity),
        composites_b_(capacity), unique_composites_(capacity),
        round_counts_(capacity), round_starts_(capacity), round_count_(1u),
        continuation_rows_(capacity), continuation_flags_(capacity),
        continuation_offsets_(capacity), continuation_ids_(capacity),
        task_begins_a_(capacity), task_begins_b_(capacity),
        task_bases_a_(capacity), task_bases_b_(capacity),
        task_depths_a_(capacity), task_depths_b_(capacity),
        ordered_(capacity), last_(capacity), winners_(capacity),
        winner_count_(1u), totals_(2u), exact_flags_(capacity),
        capsule_flags_(capacity) {
    size_temporary();
  }

  template <class Source>
  ExactOrderResult order(
      Source source, std::uint32_t rows, cudaStream_t stream,
      const std::uint32_t *prepared_heads = nullptr,
      const std::uint32_t *prepared_refs = nullptr) {
    if (rows > capacity_) throw std::length_error("exact order capacity");
    check(cudaMemsetAsync(
              ordered_.data(), 0xff, rows * sizeof(std::uint32_t), stream),
          "clear exact order");
    initialize_universal_records<<<blocks(rows), kThreads, 0, stream>>>(
        source, prepared_heads, prepared_refs, heads_.data(),
        sort_heads_a_.data(), refs_a_.data(), rows);
    std::size_t bytes = sort32_bytes_;
    check(cub::DeviceRadixSort::SortPairs(
              temporary_.data(), bytes, sort_heads_a_.data(),
              sort_heads_b_.data(), refs_a_.data(), refs_b_.data(), rows,
              0, 32, stream),
          "sort universal heads");
    bytes = rle32_bytes_;
    check(cub::DeviceRunLengthEncode::Encode(
              temporary_.data(), bytes, sort_heads_b_.data(),
              unique_heads_.data(), primary_counts_.data(),
              primary_count_.data(), rows, stream),
          "encode universal heads");
    std::uint32_t groups = 0u;
    check(cudaMemcpyAsync(&groups, primary_count_.data(), sizeof(groups),
                          cudaMemcpyDeviceToHost, stream),
          "copy head group count");
    check(cudaStreamSynchronize(stream), "wait head group count");
    bytes = scan32_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              temporary_.data(), bytes, primary_counts_.data(),
              primary_starts_.data(), groups, stream),
          "scan head groups");
    classify_primary_groups<<<groups, kThreads, 0, stream>>>(
        source, refs_b_.data(), primary_counts_.data(),
        primary_starts_.data(), primary_count_.data(), primary_rows_.data(),
        primary_flags_.data(), primary_depths_.data());
    bytes = scan32_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              temporary_.data(), bytes, primary_rows_.data(),
              primary_offsets_.data(), groups, stream),
          "scan active rows");
    bytes = scan32_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              temporary_.data(), bytes, primary_flags_.data(),
              primary_ids_.data(), groups, stream),
          "scan active groups");
    finish_scan_total<<<1, 1, 0, stream>>>(
        primary_rows_.data(), primary_offsets_.data(),
        primary_count_.data(), totals_.data());
    finish_scan_total<<<1, 1, 0, stream>>>(
        primary_flags_.data(), primary_ids_.data(),
        primary_count_.data(), totals_.data() + 1u);
    std::uint32_t initial[2]{};
    check(cudaMemcpyAsync(initial, totals_.data(), sizeof(initial),
                          cudaMemcpyDeviceToHost, stream),
          "copy initial exact totals");
    check(cudaStreamSynchronize(stream), "wait exact totals");
    initialize_exact_tasks<<<groups, kThreads, 0, stream>>>(
        refs_b_.data(), primary_counts_.data(), primary_starts_.data(),
        primary_count_.data(), primary_offsets_.data(), primary_ids_.data(),
        primary_flags_.data(), primary_depths_.data(), refs_a_.data(),
        task_ids_a_.data(), ordered_.data(), task_begins_a_.data(),
        task_bases_a_.data(), task_depths_a_.data());

    std::uint32_t active_rows = initial[0];
    std::uint32_t rounds = 0u;
    bool task_ping = false;
    while (active_rows) {
      if (++rounds > 64u)
        throw std::runtime_error("exact continuation did not converge");
      std::uint32_t *task_begins = task_ping
          ? task_begins_b_.data() : task_begins_a_.data();
      std::uint32_t *task_bases = task_ping
          ? task_bases_b_.data() : task_bases_a_.data();
      std::uint32_t *task_depths = task_ping
          ? task_depths_b_.data() : task_depths_a_.data();
      std::uint32_t *next_begins = task_ping
          ? task_begins_a_.data() : task_begins_b_.data();
      std::uint32_t *next_bases = task_ping
          ? task_bases_a_.data() : task_bases_b_.data();
      std::uint32_t *next_depths = task_ping
          ? task_depths_a_.data() : task_depths_b_.data();
      make_continuation_composites<<<
          blocks(active_rows), kThreads, 0, stream>>>(
          source, refs_a_.data(), task_ids_a_.data(), task_depths,
          composites_a_.data(), active_rows);
      bytes = sort64_bytes_;
      check(cub::DeviceRadixSort::SortPairs(
                temporary_.data(), bytes, composites_a_.data(),
                composites_b_.data(), refs_a_.data(), refs_b_.data(),
                active_rows, 0, 64, stream),
            "sort exact continuation");
      bytes = rle64_bytes_;
      check(cub::DeviceRunLengthEncode::Encode(
                temporary_.data(), bytes, composites_b_.data(),
                unique_composites_.data(), round_counts_.data(),
                round_count_.data(), active_rows, stream),
            "encode exact continuation");
      std::uint32_t round_groups = 0u;
      check(cudaMemcpyAsync(&round_groups, round_count_.data(),
                            sizeof(round_groups), cudaMemcpyDeviceToHost,
                            stream),
            "copy exact round groups");
      check(cudaStreamSynchronize(stream), "wait exact round groups");
      bytes = scan32_bytes_;
      check(cub::DeviceScan::ExclusiveSum(
                temporary_.data(), bytes, round_counts_.data(),
                round_starts_.data(), round_groups, stream),
            "scan exact round groups");
      classify_continuation_groups<<<
          blocks(round_groups), kThreads, 0, stream>>>(
          unique_composites_.data(), round_counts_.data(),
          round_count_.data(), continuation_rows_.data(),
          continuation_flags_.data());
      bytes = scan32_bytes_;
      check(cub::DeviceScan::ExclusiveSum(
                temporary_.data(), bytes, continuation_rows_.data(),
                continuation_offsets_.data(), round_groups, stream),
            "scan continuation rows");
      bytes = scan32_bytes_;
      check(cub::DeviceScan::ExclusiveSum(
                temporary_.data(), bytes, continuation_flags_.data(),
                continuation_ids_.data(), round_groups, stream),
            "scan continuation tasks");
      finish_scan_total<<<1, 1, 0, stream>>>(
          continuation_rows_.data(), continuation_offsets_.data(),
          round_count_.data(), totals_.data());
      finish_scan_total<<<1, 1, 0, stream>>>(
          continuation_flags_.data(), continuation_ids_.data(),
          round_count_.data(), totals_.data() + 1u);
      scatter_continuation_groups<<<round_groups, kThreads, 0, stream>>>(
          unique_composites_.data(), refs_b_.data(), round_counts_.data(),
          round_starts_.data(), round_count_.data(),
          continuation_offsets_.data(), continuation_ids_.data(),
          continuation_flags_.data(), task_begins, task_bases, task_depths,
          refs_a_.data(), task_ids_a_.data(), ordered_.data(), next_begins,
          next_bases, next_depths);
      std::uint32_t next[2]{};
      check(cudaMemcpyAsync(next, totals_.data(), sizeof(next),
                            cudaMemcpyDeviceToHost, stream),
            "copy continuation totals");
      check(cudaStreamSynchronize(stream), "wait continuation totals");
      active_rows = next[0];
      task_ping = !task_ping;
    }

    mark_last_universal_key<<<blocks(rows), kThreads, 0, stream>>>(
        source, ordered_.data(), last_.data(), rows);
    bytes = select_bytes_;
    check(cub::DeviceSelect::Flagged(
              temporary_.data(), bytes, ordered_.data(), last_.data(),
              winners_.data(), winner_count_.data(), rows, stream),
          "select exact winners");
    std::uint32_t winners = 0u;
    check(cudaMemcpyAsync(&winners, winner_count_.data(), sizeof(winners),
                          cudaMemcpyDeviceToHost, stream),
          "copy exact winner count");
    check(cudaStreamSynchronize(stream), "wait exact winner count");
    classify_winners<<<blocks(winners), kThreads, 0, stream>>>(
        source, winners_.data(), winner_count_.data(), unique_heads_.data(),
        primary_flags_.data(), primary_count_.data(), exact_flags_.data(),
        capsule_flags_.data());
    check(cudaGetLastError(), "classify exact winners");
    return {winners, groups, initial[0], rounds};
  }

  std::uint32_t *heads() const { return heads_.data(); }
  std::uint32_t *ordered() const { return ordered_.data(); }
  std::uint32_t *winners() const { return winners_.data(); }
  std::uint32_t *winner_count() const { return winner_count_.data(); }
  std::uint8_t *exact_flags() const { return exact_flags_.data(); }
  std::uint8_t *capsule_flags() const { return capsule_flags_.data(); }
  std::size_t bytes() const {
    return temporary_.bytes() +
        25u * std::size_t{capacity_} * sizeof(std::uint32_t) +
        3u * std::size_t{capacity_} * sizeof(std::uint64_t) +
        3u * std::size_t{capacity_} * sizeof(std::uint8_t);
  }

 private:
  void size_temporary() {
    std::size_t sizes[7]{};
    check(cub::DeviceRadixSort::SortPairs(
              nullptr, sizes[0], sort_heads_a_.data(), sort_heads_b_.data(),
              refs_a_.data(), refs_b_.data(), capacity_),
          "size head sort");
    check(cub::DeviceRadixSort::SortPairs(
              nullptr, sizes[1], composites_a_.data(), composites_b_.data(),
              refs_a_.data(), refs_b_.data(), capacity_),
          "size continuation sort");
    check(cub::DeviceRunLengthEncode::Encode(
              nullptr, sizes[2], sort_heads_b_.data(), unique_heads_.data(),
              primary_counts_.data(), primary_count_.data(), capacity_),
          "size head encoding");
    check(cub::DeviceRunLengthEncode::Encode(
              nullptr, sizes[3], composites_b_.data(),
              unique_composites_.data(), round_counts_.data(),
              round_count_.data(), capacity_),
          "size continuation encoding");
    check(cub::DeviceScan::ExclusiveSum(
              nullptr, sizes[4], primary_counts_.data(),
              primary_starts_.data(), capacity_),
          "size exact scan");
    check(cub::DeviceSelect::Flagged(
              nullptr, sizes[5], ordered_.data(), last_.data(),
              winners_.data(), winner_count_.data(), capacity_),
          "size winner selection");
    check(cub::DeviceSelect::Flagged(
              nullptr, sizes[6], winners_.data(), exact_flags_.data(),
              refs_b_.data(), totals_.data(), capacity_),
          "size exact selection");
    sort32_bytes_ = sizes[0];
    sort64_bytes_ = sizes[1];
    rle32_bytes_ = sizes[2];
    rle64_bytes_ = sizes[3];
    scan32_bytes_ = sizes[4];
    select_bytes_ = std::max(sizes[5], sizes[6]);
    temporary_.reset(*std::max_element(std::begin(sizes), std::end(sizes)));
  }

  std::uint32_t capacity_{};
  Buffer<std::uint32_t> heads_, sort_heads_a_, sort_heads_b_, refs_a_, refs_b_;
  Buffer<std::uint32_t> task_ids_a_;
  Buffer<std::uint32_t> unique_heads_, primary_counts_, primary_starts_;
  Buffer<std::uint32_t> primary_count_, primary_rows_, primary_flags_;
  Buffer<std::uint32_t> primary_depths_, primary_offsets_, primary_ids_;
  Buffer<std::uint64_t> composites_a_, composites_b_, unique_composites_;
  Buffer<std::uint32_t> round_counts_, round_starts_, round_count_;
  Buffer<std::uint32_t> continuation_rows_, continuation_flags_;
  Buffer<std::uint32_t> continuation_offsets_, continuation_ids_;
  Buffer<std::uint32_t> task_begins_a_, task_begins_b_;
  Buffer<std::uint32_t> task_bases_a_, task_bases_b_;
  Buffer<std::uint32_t> task_depths_a_, task_depths_b_;
  Buffer<std::uint32_t> ordered_, winners_, winner_count_, totals_;
  Buffer<std::uint8_t> last_, exact_flags_, capsule_flags_;
  Buffer<std::uint8_t> temporary_;
  std::size_t sort32_bytes_{}, sort64_bytes_{}, rle32_bytes_{};
  std::size_t rle64_bytes_{}, scan32_bytes_{}, select_bytes_{};
};

}  // namespace gpulsm_sparse
// END INLINED COMPONENT: gpulsm_sparse/exact.cuh

namespace gpulsm_sparse {

template <class Source>
__device__ __forceinline__ std::uint32_t find_capsule_lifecycle_source(
    const CapsuleLifecycleSource *sources, std::uint32_t source_count,
    std::uint32_t ordinal) {
  std::uint32_t low = 0u;
  std::uint32_t high = source_count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (sources[middle].ordinal < ordinal) low = middle + 1u;
    else high = middle;
  }
  return low < source_count && sources[low].ordinal == ordinal
      ? low : source_count;
}

template <class Source>
__global__ void count_capsule_survivors(
    Source source, const std::uint32_t *refs,
    const std::uint32_t *count, CapsuleLifecycleSource *sources,
    std::uint32_t source_count, unsigned long long *errors) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= *count) return;
  const CapsuleIndex *owned = source_owned_capsule(source, refs[ordinal]);
  if (!owned) return;
  const std::uint32_t source_index = find_capsule_lifecycle_source<Source>(
      sources, source_count, owned->segment);
  if (source_index == source_count) {
    atomicCAS(errors, 0ull,
              (1ull << 32u) | static_cast<unsigned long long>(owned->segment));
    return;
  }
  const CapsuleHeader *header = capsule_header(owned);
  std::uint64_t bytes = 0u;
  if (!header || !capsule_object_bytes(
          header->key_length, header->value_length, bytes)) {
    atomicCAS(errors, 0ull,
              (2ull << 32u) | static_cast<unsigned long long>(owned->segment));
    return;
  }
  atomicAdd(reinterpret_cast<unsigned long long *>(
                &sources[source_index].survivor_bytes),
            static_cast<unsigned long long>(bytes));
}

template <class Source>
__global__ void make_capsule_object_sizes(
    Source source, const std::uint32_t *refs,
    const std::uint32_t *count, const CapsuleLifecycleSource *sources,
    std::uint32_t source_count, std::uint64_t *sizes,
    unsigned long long *errors) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t size = *count;
  if (ordinal > size) return;
  if (ordinal == size) {
    sizes[ordinal] = 0u;
    return;
  }
  const std::uint32_t ref = refs[ordinal];
  const CapsuleIndex *owned = source_owned_capsule(source, ref);
  if (owned) {
    const std::uint32_t source_index = find_capsule_lifecycle_source<Source>(
        sources, source_count, owned->segment);
    if (source_index == source_count) {
      sizes[ordinal] = 0u;
      if (errors)
        atomicCAS(errors, 0ull,
                  (3ull << 32u) |
                      static_cast<unsigned long long>(owned->segment));
      return;
    }
    if (sources[source_index].action == kCapsuleTransfer) {
      sizes[ordinal] = 0u;
      return;
    }
  }
  const auto key = source_key(source, ref);
  const auto value = source_value(source, ref);
  std::uint64_t object = 0u;
  const std::uint64_t value_length =
      source_tombstone(source, ref) ? 0u : value.length();
  sizes[ordinal] = capsule_object_bytes(
      key.length(), value_length, object) ? object : 0u;
}

__global__ void build_capsule_pages(
    const std::uint64_t *unique_pages,
    const std::uint32_t *page_counts,
    const std::uint32_t *page_starts,
    const std::uint32_t *page_count,
    CapsulePage *pages) {
  const std::uint32_t page = blockIdx.x * blockDim.x + threadIdx.x;
  if (page < *page_count)
    pages[page] = {
        unique_pages[page], page_starts[page], page_counts[page]};
}

__global__ void assign_capsule_ranks(
    const std::uint64_t *physical_pages,
    const std::uint32_t *page_starts,
    const std::uint32_t *page_count,
    const std::uint32_t *capsule_positions,
    const std::uint32_t *capsule_count,
    std::uint16_t *combined_ranks,
    unsigned long long *errors) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= *capsule_count) return;
  std::uint32_t low = 0u;
  std::uint32_t high = *page_count;
  const std::uint64_t target = physical_pages[ordinal];
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (physical_pages[page_starts[middle]] < target) low = middle + 1u;
    else high = middle;
  }
  if (low >= *page_count || physical_pages[page_starts[low]] != target) {
    atomicAdd(errors, 1ull);
    return;
  }
  const std::uint32_t rank = ordinal - page_starts[low];
  if (rank > kAnchorCapsuleRankMask) {
    atomicAdd(errors, 1ull);
    return;
  }
  combined_ranks[capsule_positions[ordinal]] =
      static_cast<std::uint16_t>(rank);
}

template <class Source>
__global__ void emit_capsule_objects(
    Source source, const std::uint32_t *refs,
    const std::uint32_t *count, const std::uint64_t *offsets,
    std::uint8_t *segment, std::uint32_t segment_ordinal,
    const CapsuleLifecycleSource *sources, std::uint32_t source_count,
    CapsuleIndex *indexes, unsigned long long *errors) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= *count) return;
  const std::uint32_t ref = refs[ordinal];
  const CapsuleIndex *owned = source_owned_capsule(source, ref);
  if (owned) {
    const std::uint32_t source_index = find_capsule_lifecycle_source<Source>(
        sources, source_count, owned->segment);
    if (source_index == source_count) {
      if (errors)
        atomicCAS(errors, 0ull,
                  (4ull << 32u) |
                      static_cast<unsigned long long>(owned->segment));
      return;
    }
    if (sources[source_index].action == kCapsuleTransfer) {
      indexes[ordinal] = *owned;
      return;
    }
  }
  const auto key = source_key(source, ref);
  const auto value = source_value(source, ref);
  const bool tombstone = source_tombstone(source, ref);
  const std::uint64_t value_length = tombstone ? 0u : value.length();
  std::uint8_t *object = segment + offsets[ordinal];
  reinterpret_cast<CapsuleHeader *>(object)[0] = {
      key.length(), value_length, tombstone ? 1u : 0u, 0u};
  std::uint8_t *bytes = object + sizeof(CapsuleHeader);
  for (std::uint64_t position = 0u; position < key.length(); ++position)
    bytes[position] = key.byte(position);
  for (std::uint64_t position = 0u; position < value_length; ++position)
    bytes[key.length() + position] = value.byte(position);
  indexes[ordinal] = {
      reinterpret_cast<std::uint64_t>(object), rejection_hash(key),
      segment_ordinal};
}

}  // namespace gpulsm_sparse
// END INLINED COMPONENT: gpulsm_sparse/capsule_lifecycle.cuh

namespace gpulsm_sparse {

__global__ void normalize_pending_input(
    RecordBatchView source, std::uint32_t count, std::uint32_t batch_slot,
    std::uint32_t *heads, gpulsmopt2_detail::RawPayload *payloads,
    std::uint8_t *capsule_flags, std::uint8_t *extended_key_flags) {
  const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const DeviceKeyCursor key{source.keys, row, source.head4_words};
  const DeviceValueCursor value{source.values, row};
  const bool tombstone = record_is_tombstone(source, row);
  const std::uint32_t inline_value = value.inline_word();
  const std::uint32_t summary = source.range_contributions
      ? source.range_contributions[row] : inline_value;
  const bool capsule = key.length() != 4u ||
      (!tombstone && value.length() != 4u) ||
      (!tombstone && summary != inline_value);
  std::uint32_t metadata =
      ((batch_slot << gpulsmopt2_detail::kBatchPositionBits) | row) &
      kPendingAgeMask;
  if (tombstone) metadata |= gpulsmopt2_detail::kRawTombstone;
  heads[row] = key.head4();
  payloads[row] = {summary, metadata};
  capsule_flags[row] = capsule;
  extended_key_flags[row] = key.length() != 4u;
}

__global__ void make_pending_exact_entries(
    RecordBatchView source, const std::uint32_t *winners,
    std::uint32_t count, const std::uint32_t *destinations,
    std::uint32_t batch_slot, std::uint32_t *heads,
    std::uint32_t *physical_refs) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= count) return;
  const std::uint32_t input = winners[ordinal];
  heads[ordinal] = DeviceKeyCursor{
      source.keys, input, source.head4_words}.head4();
  physical_refs[ordinal] =
      (batch_slot << gpulsmopt2_detail::kBatchPositionBits) |
      destinations[input];
}

__global__ void make_pending_special_entries(
    RecordBatchView source, const std::uint32_t *selected,
    std::uint32_t count, std::uint32_t *heads, std::uint32_t *refs) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= count) return;
  const std::uint32_t ref = selected[ordinal];
  heads[ordinal] = DeviceKeyCursor{
      source.keys, ref, source.head4_words}.head4();
  refs[ordinal] = ref;
}

__global__ void build_pending_exact_descriptors(
    const std::uint32_t *heads, const std::uint32_t *counts,
    const std::uint32_t *starts, std::uint32_t descriptor_count,
    std::uint32_t ref_base, std::uint32_t batch_slot,
    PendingExactHeadDescriptor *descriptors) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < descriptor_count)
    descriptors[index] = {
        heads[index], counts[index], ref_base + starts[index], batch_slot};
}

__global__ void scatter_pending_input(
    const std::uint32_t *heads,
    const gpulsmopt2_detail::RawPayload *payloads,
    std::uint32_t count, const std::uint32_t *offsets,
    const std::uint32_t *reservation_ranks,
    std::uint32_t *destination_keys,
    gpulsmopt2_detail::RawPayload *destination_payloads,
    std::uint32_t *destinations) {
  const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const std::uint32_t head = heads[row];
  const std::uint32_t output =
      offsets[head >> 16u] + reservation_ranks[row];
  destination_keys[output] = head;
  destination_payloads[output] = payloads[row];
  destinations[row] = output;
}

__global__ void gather_pending_capsules(
    const std::uint32_t *selected_input,
    const std::uint32_t *selected_count,
    const std::uint32_t *destinations,
    std::uint32_t *physical_positions,
    std::uint32_t *input_refs) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= *selected_count) return;
  const std::uint32_t ref = selected_input[ordinal];
  physical_positions[ordinal] = destinations[ref];
  input_refs[ordinal] = ref;
}

__global__ void make_pending_exception_refs(
    const std::uint32_t *selected_input, std::uint32_t count,
    const std::uint32_t *destinations, std::uint32_t batch_slot,
    PendingExceptionRef *exceptions) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= count) return;
  const std::uint32_t input = selected_input[ordinal];
  exceptions[ordinal] = {
      (batch_slot << gpulsmopt2_detail::kBatchPositionBits) | input,
      destinations[input]};
}

__global__ void make_pending_page_keys(
    const std::uint32_t *positions, std::uint32_t count,
    std::uint32_t batch_slot, std::uint64_t *keys) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal < count)
    keys[ordinal] = (std::uint64_t{batch_slot} << 32u) |
        (positions[ordinal] / kCapsulePageRows);
}

__global__ void initialize_pending_pages(
    const std::uint64_t *keys, const std::uint32_t *counts,
    const std::uint32_t *starts, const std::uint32_t *page_count,
    std::uint32_t index_base, PendingCapsulePage *pages) {
  const std::uint32_t page = blockIdx.x;
  if (page >= *page_count) return;
  PendingCapsulePage &output = pages[page];
  if (threadIdx.x == 0u) {
    output.batch_slot = static_cast<std::uint32_t>(keys[page] >> 32u);
    output.local_page = static_cast<std::uint32_t>(keys[page]);
    output.index_begin = index_base + starts[page];
    output.index_count = counts[page];
  }
  for (std::uint32_t word = threadIdx.x;
       word < kCapsulePageRows / 32u; word += blockDim.x)
    output.bitmap[word] = 0u;
  if (threadIdx.x <= kCapsulePageRows / 32u)
    output.prefixes[threadIdx.x] = 0u;
}

__global__ void populate_pending_pages(
    const std::uint32_t *positions, std::uint32_t count,
    const std::uint32_t *page_starts,
    const std::uint32_t *page_count,
    PendingCapsulePage *pages) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= count) return;
  std::uint32_t low = 0u;
  std::uint32_t high = *page_count;
  const std::uint32_t target = positions[ordinal] / kCapsulePageRows;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    const std::uint32_t found =
        positions[page_starts[middle]] / kCapsulePageRows;
    if (found < target) low = middle + 1u;
    else high = middle;
  }
  if (low >= *page_count) return;
  const std::uint32_t local = positions[ordinal] % kCapsulePageRows;
  atomicOr(pages[low].bitmap + (local >> 5u), 1u << (local & 31u));
}

__global__ void prefix_pending_pages(
    PendingCapsulePage *pages, const std::uint32_t *page_count) {
  const std::uint32_t page = blockIdx.x;
  if (page >= *page_count || threadIdx.x) return;
  std::uint16_t prefix = 0u;
  pages[page].prefixes[0] = 0u;
  for (std::uint32_t word = 0u; word < kCapsulePageRows / 32u; ++word) {
    prefix = static_cast<std::uint16_t>(
        prefix + __popc(pages[page].bitmap[word]));
    pages[page].prefixes[word + 1u] = prefix;
  }
}

struct PendingBuildResult {
  std::uint32_t rows{};
  std::uint32_t capsules{};
  std::uint32_t pages{};
  std::uint32_t special_heads{};
  std::uint32_t exact_heads{};
  std::uint32_t exact_refs{};
  std::uint64_t capsule_bytes{};
};

class PendingWorkspace {
 public:
  explicit PendingWorkspace(std::uint32_t capacity)
      : capacity_(capacity), heads_(capacity), payloads_(capacity),
        reservation_ranks_(capacity), counts_(kSections + 1u),
        capsule_flags_(capacity), extended_key_flags_(capacity),
        capsule_inputs_(capacity), capsule_count_(1u),
        extended_inputs_(capacity), extended_count_(1u),
        destinations_(capacity), exact_order_(capacity),
        exact_heads_(capacity), unique_exact_heads_(capacity),
        exact_counts_(capacity), exact_starts_(capacity),
        exact_head_count_(1u), exact_physical_refs_(capacity),
        capsule_positions_a_(capacity), capsule_positions_b_(capacity),
        capsule_refs_a_(capacity), capsule_refs_b_(capacity),
        page_keys_(capacity), unique_page_keys_(capacity),
        page_counts_(capacity), page_starts_(capacity), page_count_(1u),
        object_sizes_(std::size_t{capacity} + 1u),
        object_offsets_(std::size_t{capacity} + 1u) {
    size_temporary();
  }

  PendingBuildResult build_batch(
      RecordBatchView source, std::uint32_t count,
      std::uint32_t batch_slot, std::uint32_t batch_capacity,
      std::uint32_t *staging_keys,
      gpulsmopt2_detail::RawPayload *staging_payloads,
      std::uint32_t *staging_offsets, std::uint64_t *staging_signatures,
      StagedPendingSlotOverlay &overlay, std::uint32_t segment_ordinal,
      cudaStream_t stream) {
    if (count > capacity_ || count > batch_capacity)
      throw std::length_error("pending batch capacity");
    check(cudaMemsetAsync(counts_.data(), 0, counts_.bytes(), stream),
          "clear pending counts");
    check(cudaMemsetAsync(staging_signatures, 0,
                          kSections * sizeof(std::uint64_t), stream),
          "clear pending signatures");
    normalize_pending_input<<<blocks(count), kThreads, 0, stream>>>(
        source, count, batch_slot, heads_.data(), payloads_.data(),
        capsule_flags_.data(), extended_key_flags_.data());
    gpulsmopt2_detail::count_admission_quotients_kernel<<<
        blocks(count), kThreads, 0, stream>>>(
        heads_.data(), count, counts_.data(), reservation_ranks_.data());
    std::size_t bytes = scan32_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              temporary_.data(), bytes, counts_.data(), staging_offsets,
              kSections + 1u, stream),
          "scan pending sections");
    scatter_pending_input<<<blocks(count), kThreads, 0, stream>>>(
        heads_.data(), payloads_.data(), count, staging_offsets,
        reservation_ranks_.data(), staging_keys, staging_payloads,
        destinations_.data());
    gpulsmopt2_detail::build_admission_signatures_kernel<<<
        blocks(count), kThreads, 0, stream>>>(
        staging_keys, count, staging_signatures);
    cub::CountingInputIterator<std::uint32_t> ids(0u);
    bytes = select_bytes_;
    check(cub::DeviceSelect::Flagged(
              temporary_.data(), bytes, ids, capsule_flags_.data(),
              capsule_inputs_.data(), capsule_count_.data(), count, stream),
          "select pending capsules");
    bytes = select_bytes_;
    check(cub::DeviceSelect::Flagged(
              temporary_.data(), bytes, ids, extended_key_flags_.data(),
              extended_inputs_.data(), extended_count_.data(), count,
              stream),
          "select pending extended keys");
    std::uint32_t capsules = 0u;
    std::uint32_t extended = 0u;
    check(cudaMemcpyAsync(&capsules, capsule_count_.data(), sizeof(capsules),
                          cudaMemcpyDeviceToHost, stream),
          "copy pending capsule count");
    check(cudaMemcpyAsync(&extended, extended_count_.data(),
                          sizeof(extended), cudaMemcpyDeviceToHost, stream),
          "copy pending extended count");
    check(cudaStreamSynchronize(stream), "wait pending capsule count");
    if (!capsules) return {count, 0u, 0u, 0u, 0u, 0u, 0u};

    gather_pending_capsules<<<blocks(capsules), kThreads, 0, stream>>>(
        capsule_inputs_.data(), capsule_count_.data(), destinations_.data(),
        capsule_positions_a_.data(), capsule_refs_a_.data());
    bytes = sort32_bytes_;
    check(cub::DeviceRadixSort::SortPairs(
              temporary_.data(), bytes, capsule_positions_a_.data(),
              capsule_positions_b_.data(), capsule_refs_a_.data(),
              capsule_refs_b_.data(), capsules, 0, 32, stream),
          "sort pending capsules");
    make_pending_page_keys<<<blocks(capsules), kThreads, 0, stream>>>(
        capsule_positions_b_.data(), capsules, batch_slot, page_keys_.data());
    bytes = rle64_bytes_;
    check(cub::DeviceRunLengthEncode::Encode(
              temporary_.data(), bytes, page_keys_.data(),
              unique_page_keys_.data(), page_counts_.data(),
              page_count_.data(), capsules, stream),
          "encode pending capsule pages");
    std::uint32_t pages = 0u;
    check(cudaMemcpyAsync(&pages, page_count_.data(), sizeof(pages),
                          cudaMemcpyDeviceToHost, stream),
          "copy pending page count");
    check(cudaStreamSynchronize(stream), "wait pending page count");
    bytes = scan32_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              temporary_.data(), bytes, page_counts_.data(),
              page_starts_.data(), pages, stream),
          "scan pending pages");
    make_capsule_object_sizes<<<blocks(std::uint64_t{capsules} + 1u),
                                kThreads, 0, stream>>>(
        source, capsule_refs_b_.data(), capsule_count_.data(),
        nullptr, 0u, object_sizes_.data(), nullptr);
    bytes = scan64_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              temporary_.data(), bytes, object_sizes_.data(),
              object_offsets_.data(), std::size_t{capsules} + 1u, stream),
          "scan pending capsule bytes");
    std::uint64_t capsule_bytes = 0u;
    check(cudaMemcpyAsync(&capsule_bytes, object_offsets_.data() + capsules,
                          sizeof(capsule_bytes), cudaMemcpyDeviceToHost,
                          stream),
          "copy pending capsule bytes");
    check(cudaStreamSynchronize(stream), "wait pending capsule bytes");

    overlay.pages.reset(pages);
    overlay.indexes.reset(capsules);
    overlay.exceptions.reset(capsules);
    auto segment = std::make_shared<CapsuleSegment>(capsule_bytes);
    make_pending_exception_refs<<<blocks(capsules), kThreads, 0, stream>>>(
        capsule_inputs_.data(), capsules, destinations_.data(), batch_slot,
        overlay.exceptions.data());
    initialize_pending_pages<<<pages, 256u, 0, stream>>>(
        unique_page_keys_.data(), page_counts_.data(), page_starts_.data(),
        page_count_.data(), 0u, overlay.pages.data());
    populate_pending_pages<<<blocks(capsules), kThreads, 0, stream>>>(
        capsule_positions_b_.data(), capsules, page_starts_.data(),
        page_count_.data(), overlay.pages.data());
    prefix_pending_pages<<<pages, 1u, 0, stream>>>(
        overlay.pages.data(), page_count_.data());
    emit_capsule_objects<<<blocks(capsules), kThreads, 0, stream>>>(
        source, capsule_refs_b_.data(), capsule_count_.data(),
        object_offsets_.data(), segment->data(), segment_ordinal,
        nullptr, 0u, overlay.indexes.data(), nullptr);
    overlay.segments.push_back(
        {std::move(segment), segment_ordinal, 0u, capsule_bytes, 0u});
    overlay.live_bytes = capsule_bytes;
    overlay.garbage_bytes = 0u;

    // Every capsule-valued head participates in sparse lifecycle work, even
    // when its key is an ordinary four-byte key and only its value is long.
    // This reuses the pending prototype's existing sort/RLE scratch after
    // capsule emission, so no capacity-sized permanent head plane is added.
    make_pending_special_entries<<<blocks(capsules), kThreads, 0, stream>>>(
        source, capsule_inputs_.data(), capsules, exact_heads_.data(),
        capsule_refs_a_.data());
    bytes = sort32_bytes_;
    check(cub::DeviceRadixSort::SortPairs(
              temporary_.data(), bytes, exact_heads_.data(),
              capsule_positions_a_.data(), capsule_refs_a_.data(),
              capsule_refs_b_.data(), capsules, 0, 32, stream),
          "sort pending special heads");
    bytes = rle32_bytes_;
    check(cub::DeviceRunLengthEncode::Encode(
              temporary_.data(), bytes, capsule_positions_a_.data(),
              unique_exact_heads_.data(), exact_counts_.data(),
              exact_head_count_.data(), capsules, stream),
          "encode pending special heads");
    std::uint32_t special_heads = 0u;
    check(cudaMemcpyAsync(&special_heads, exact_head_count_.data(),
                          sizeof(special_heads), cudaMemcpyDeviceToHost,
                          stream),
          "copy pending special head count");
    check(cudaStreamSynchronize(stream),
          "wait pending special head count");
    overlay.special_heads.reset(special_heads);
    check(cudaMemcpyAsync(
              overlay.special_heads.data(), unique_exact_heads_.data(),
              special_heads * sizeof(std::uint32_t),
              cudaMemcpyDeviceToDevice, stream),
          "store pending special heads");

    std::uint32_t exact_heads = 0u;
    std::uint32_t exact_refs = 0u;
    if (extended) {
      const ExactOrderResult order = exact_order_.order(
          source, extended, stream, nullptr, extended_inputs_.data());
      exact_refs = order.winner_count;
      make_pending_exact_entries<<<blocks(exact_refs), kThreads, 0, stream>>>(
          source, exact_order_.winners(), exact_refs, destinations_.data(),
          batch_slot, exact_heads_.data(), exact_physical_refs_.data());
      bytes = rle32_bytes_;
      check(cub::DeviceRunLengthEncode::Encode(
                temporary_.data(), bytes, exact_heads_.data(),
                unique_exact_heads_.data(), exact_counts_.data(),
                exact_head_count_.data(), exact_refs, stream),
            "encode pending exact heads");
      check(cudaMemcpyAsync(&exact_heads, exact_head_count_.data(),
                            sizeof(exact_heads), cudaMemcpyDeviceToHost,
                            stream),
            "copy pending exact head count");
      check(cudaStreamSynchronize(stream),
            "wait pending exact head count");
      bytes = scan32_bytes_;
      check(cub::DeviceScan::ExclusiveSum(
                temporary_.data(), bytes, exact_counts_.data(),
                exact_starts_.data(), exact_heads, stream),
            "scan pending exact heads");
      overlay.exact_heads.reset(exact_heads);
      overlay.exact_refs.reset(exact_refs);
      check(cudaMemcpyAsync(
                overlay.exact_refs.data(), exact_physical_refs_.data(),
                exact_refs * sizeof(std::uint32_t),
                cudaMemcpyDeviceToDevice, stream),
            "store pending exact refs");
      build_pending_exact_descriptors<<<
          blocks(exact_heads), kThreads, 0, stream>>>(
          unique_exact_heads_.data(), exact_counts_.data(),
          exact_starts_.data(), exact_heads, 0u, batch_slot,
          overlay.exact_heads.data());
    }
    check(cudaGetLastError(), "build pending capsules");
    return {count, capsules, pages, special_heads, exact_heads, exact_refs,
            capsule_bytes};
  }

  std::size_t bytes() const {
    return temporary_.bytes() +
        10u * std::size_t{capacity_} * sizeof(std::uint32_t) +
        5u * std::size_t{capacity_} * sizeof(std::uint64_t) +
        std::size_t{capacity_} * sizeof(gpulsmopt2_detail::RawPayload) +
        3u * std::size_t{capacity_} + exact_order_.bytes();
  }

 private:
  void size_temporary() {
    std::size_t sizes[7]{};
    cub::CountingInputIterator<std::uint32_t> ids(0u);
    check(cub::DeviceScan::ExclusiveSum(
              nullptr, sizes[0], counts_.data(), counts_.data(),
              kSections + 1u),
          "size pending scan");
    check(cub::DeviceSelect::Flagged(
              nullptr, sizes[1], ids, capsule_flags_.data(),
              capsule_inputs_.data(), capsule_count_.data(), capacity_),
          "size pending select");
    check(cub::DeviceRunLengthEncode::Encode(
              nullptr, sizes[5], exact_heads_.data(),
              unique_exact_heads_.data(), exact_counts_.data(),
              exact_head_count_.data(), capacity_),
          "size pending exact heads");
    check(cub::DeviceRadixSort::SortPairs(
              nullptr, sizes[2], capsule_positions_a_.data(),
              capsule_positions_b_.data(), capsule_refs_a_.data(),
              capsule_refs_b_.data(), capacity_),
          "size pending sort");
    check(cub::DeviceRunLengthEncode::Encode(
              nullptr, sizes[3], page_keys_.data(), unique_page_keys_.data(),
              page_counts_.data(), page_count_.data(), capacity_),
          "size pending pages");
    check(cub::DeviceScan::ExclusiveSum(
              nullptr, sizes[4], object_sizes_.data(), object_offsets_.data(),
              std::size_t{capacity_} + 1u),
          "size pending byte scan");
    check(cub::DeviceScan::ExclusiveSum(
              nullptr, sizes[6], exact_counts_.data(),
              exact_starts_.data(), capacity_),
          "size pending exact scan");
    scan32_bytes_ = std::max(sizes[0], sizes[6]);
    select_bytes_ = sizes[1];
    sort32_bytes_ = sizes[2];
    rle64_bytes_ = sizes[3];
    scan64_bytes_ = sizes[4];
    rle32_bytes_ = sizes[5];
    temporary_.reset(*std::max_element(std::begin(sizes), std::end(sizes)));
  }

  std::uint32_t capacity_{};
  Buffer<std::uint32_t> heads_;
  Buffer<gpulsmopt2_detail::RawPayload> payloads_;
  Buffer<std::uint32_t> reservation_ranks_, counts_;
  Buffer<std::uint8_t> capsule_flags_, extended_key_flags_;
  Buffer<std::uint32_t> capsule_inputs_, capsule_count_;
  Buffer<std::uint32_t> extended_inputs_, extended_count_, destinations_;
  ExactOrderWorkspace exact_order_;
  Buffer<std::uint32_t> exact_heads_, unique_exact_heads_;
  Buffer<std::uint32_t> exact_counts_, exact_starts_, exact_head_count_;
  Buffer<std::uint32_t> exact_physical_refs_;
  Buffer<std::uint32_t> capsule_positions_a_, capsule_positions_b_;
  Buffer<std::uint32_t> capsule_refs_a_, capsule_refs_b_;
  Buffer<std::uint64_t> page_keys_, unique_page_keys_;
  Buffer<std::uint32_t> page_counts_, page_starts_, page_count_;
  Buffer<std::uint64_t> object_sizes_, object_offsets_;
  Buffer<std::uint8_t> temporary_;
  std::size_t scan32_bytes_{}, select_bytes_{}, sort32_bytes_{};
  std::size_t rle64_bytes_{}, rle32_bytes_{}, scan64_bytes_{};
};

}  // namespace gpulsm_sparse
// END INLINED COMPONENT: gpulsm_sparse/pending.cuh
// BEGIN INLINED COMPONENT: gpulsm_sparse/refine.cuh

// BEGIN INLINED COMPONENT: gpulsm_sparse/fan_in.cuh


#include <thrust/execution_policy.h>
#include <thrust/merge.h>

namespace gpulsm_sparse {

struct ExactRun {
  std::uint32_t begin{};
  std::uint32_t count{};
};

template <class IncomingSource>
struct TerminalRefLess {
  TerminalSourceView<IncomingSource> source{};

  __device__ std::uint64_t age(std::uint32_t ref) const {
    const TerminalCandidate &candidate = source.candidates[ref];
    // This is the canonical GPULSMOpt source order: pending is source zero
    // and resident levels are visited in ascending level order, so a smaller
    // resident level is newer.  Express that order as an increasing age so
    // the prototype's mark-last rule selects the same winner as the original
    // carry without mirroring epoch metadata for ordinary roots.
    return candidate.source == kTerminalIncomingSource
        ? std::numeric_limits<std::uint64_t>::max()
        : std::numeric_limits<std::uint64_t>::max() - 1u -
              candidate.source;
  }

  __device__ bool operator()(std::uint32_t left,
                             std::uint32_t right) const {
    const int comparison = compare_keys(
        source_key(source, left), source_key(source, right));
    if (comparison != 0) return comparison < 0;
    const std::uint64_t left_age = age(left);
    const std::uint64_t right_age = age(right);
    if (left_age != right_age) return left_age < right_age;
    const TerminalCandidate &a = source.candidates[left];
    const TerminalCandidate &b = source.candidates[right];
    if (a.source != b.source) return a.source < b.source;
    if (a.locator != b.locator) return a.locator < b.locator;
    return left < right;
  }
};

__global__ void initialize_terminal_refs(
    std::uint32_t *refs, std::uint32_t count) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position < count) refs[position] = position;
}

template <class IncomingSource>
__global__ void mark_last_terminal_key(
    TerminalSourceView<IncomingSource> source,
    const std::uint32_t *merged, std::uint8_t *flags,
    std::uint32_t count) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= count) return;
  flags[position] = static_cast<std::uint8_t>(
      position + 1u == count ||
      compare_keys(source_key(source, merged[position]),
                   source_key(source, merged[position + 1u])) != 0);
}

// Direct production extraction of exact_fanin_gate.cu: sorted exact streams
// are merged pairwise with thrust::merge, then the last (newest) member of
// each equal-key group is selected.  The merge tree is allowed to have more
// than the benchmark's four leaves because a terminal radix carry can expose
// any number of already-sorted resident roots.
class ExactFanInWorkspace {
 public:
  struct Result {
    const std::uint32_t *winners{};
    std::uint32_t winner_count{};
  };

  template <class IncomingSource>
  Result merge(TerminalSourceView<IncomingSource> source,
               const std::vector<ExactRun> &runs,
               std::uint32_t candidate_count, cudaStream_t stream) {
    if (!candidate_count || runs.empty()) return {};
    ensure(candidate_count);
    std::uint64_t covered = 0u;
    for (const ExactRun &run : runs) {
      if (!run.count || run.begin != covered)
        throw std::invalid_argument("noncontiguous exact fan-in runs");
      covered += run.count;
    }
    if (covered != candidate_count)
      throw std::invalid_argument("exact fan-in coverage mismatch");

    initialize_terminal_refs<<<blocks(candidate_count), kThreads, 0,
                               stream>>>(refs_a_.data(), candidate_count);
    check(cudaGetLastError(), "initialize exact fan-in refs");
    std::vector<ExactRun> current = runs;
    bool input_a = true;
    TerminalRefLess<IncomingSource> less{source};
    auto policy = thrust::cuda::par_nosync.on(stream);
    while (current.size() > 1u) {
      const std::uint32_t *input = input_a
          ? refs_a_.data() : refs_b_.data();
      std::uint32_t *output = input_a
          ? refs_b_.data() : refs_a_.data();
      std::vector<ExactRun> next;
      next.reserve((current.size() + 1u) / 2u);
      for (std::size_t index = 0u; index < current.size(); index += 2u) {
        const ExactRun left = current[index];
        if (index + 1u == current.size()) {
          check(cudaMemcpyAsync(
                    output + left.begin, input + left.begin,
                    std::size_t{left.count} * sizeof(std::uint32_t),
                    cudaMemcpyDeviceToDevice, stream),
                "copy odd exact fan-in run");
          next.push_back(left);
          continue;
        }
        const ExactRun right = current[index + 1u];
        if (left.begin + left.count != right.begin)
          throw std::invalid_argument("disjoint exact fan-in pair");
        thrust::merge(policy,
                      input + left.begin,
                      input + left.begin + left.count,
                      input + right.begin,
                      input + right.begin + right.count,
                      output + left.begin, less);
        next.push_back({left.begin, left.count + right.count});
      }
      current = std::move(next);
      input_a = !input_a;
    }

    const std::uint32_t *merged = input_a
        ? refs_a_.data() : refs_b_.data();
    mark_last_terminal_key<<<blocks(candidate_count), kThreads, 0, stream>>>(
        source, merged, last_.data(), candidate_count);
    std::size_t bytes = select_bytes_;
    check(cub::DeviceSelect::Flagged(
              temporary_.data(), bytes, merged, last_.data(),
              winners_.data(), winner_count_.data(), candidate_count,
              stream),
          "select exact fan-in winners");
    std::uint32_t winners = 0u;
    check(cudaMemcpyAsync(&winners, winner_count_.data(), sizeof(winners),
                          cudaMemcpyDeviceToHost, stream),
          "copy exact fan-in winner count");
    check(cudaStreamSynchronize(stream), "wait exact fan-in winners");
    return {winners_.data(), winners};
  }

  std::size_t bytes() const {
    return refs_a_.bytes() + refs_b_.bytes() + last_.bytes() +
        winners_.bytes() + winner_count_.bytes() + temporary_.bytes();
  }

 private:
  void ensure(std::uint32_t capacity) {
    if (capacity <= capacity_) return;
    capacity_ = capacity;
    refs_a_.reset(capacity);
    refs_b_.reset(capacity);
    last_.reset(capacity);
    winners_.reset(capacity);
    winner_count_.reset(1u);
    std::size_t bytes = 0u;
    check(cub::DeviceSelect::Flagged(
              nullptr, bytes, refs_a_.data(), last_.data(),
              winners_.data(), winner_count_.data(), capacity),
          "size exact fan-in select");
    select_bytes_ = bytes;
    temporary_.reset(bytes);
  }

  std::uint32_t capacity_{};
  Buffer<std::uint32_t> refs_a_, refs_b_, winners_, winner_count_;
  Buffer<std::uint8_t> last_, temporary_;
  std::size_t select_bytes_{};
};

}  // namespace gpulsm_sparse
// END INLINED COMPONENT: gpulsm_sparse/fan_in.cuh

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
// END INLINED COMPONENT: gpulsm_sparse/refine.cuh
// BEGIN INLINED COMPONENT: gpulsm_sparse/bulk.cuh


namespace gpulsm_sparse {

// Direct extraction of the observed-head front of the closure prototype's
// UniversalRootWorkspace.  It produces exactly one restored projection row
// per head and compacts only exceptional head groups for exact refinement.
template <class Source>
__device__ __forceinline__ bool direct_source_requires_sparse(
    Source source, std::uint32_t ref) {
  const auto key = source_key(source, ref);
  const auto value = source_value(source, ref);
  const bool tombstone = source_tombstone(source, ref);
  const std::uint32_t inline_value = value.inline_word();
  return source_owned_capsule(source, ref) || key.length() != 4u ||
      (!tombstone && value.length() != 4u) ||
      (!tombstone && source_summary(source, ref) != inline_value);
}

template <class Source>
__global__ void classify_direct_head_groups(
    Source source, const std::uint32_t *sorted_refs,
    const std::uint32_t *group_counts, const std::uint32_t *group_starts,
    std::uint32_t group_count, std::uint32_t *sparse_rows,
    std::uint8_t *sparse_flags) {
  const std::uint32_t group = blockIdx.x;
  if (group >= group_count) return;
  const std::uint32_t begin = group_starts[group];
  const std::uint32_t count = group_counts[group];
  __shared__ std::uint32_t sparse;
  if (!threadIdx.x) sparse = 0u;
  __syncthreads();
  for (std::uint32_t local = threadIdx.x; local < count;
       local += blockDim.x)
    if (direct_source_requires_sparse(source, sorted_refs[begin + local]))
      atomicExch(&sparse, 1u);
  __syncthreads();
  if (!threadIdx.x) {
    sparse_rows[group] = sparse ? count : 0u;
    sparse_flags[group] = static_cast<std::uint8_t>(sparse);
  }
}

__global__ void scatter_direct_sparse_refs(
    const std::uint32_t *sorted_refs, const std::uint32_t *group_counts,
    const std::uint32_t *group_starts,
    const std::uint32_t *sparse_offsets,
    const std::uint8_t *sparse_flags, std::uint32_t group_count,
    std::uint32_t *sparse_refs) {
  const std::uint32_t group = blockIdx.x;
  if (group >= group_count || !sparse_flags[group]) return;
  const std::uint32_t count = group_counts[group];
  for (std::uint32_t local = threadIdx.x; local < count;
       local += blockDim.x)
    sparse_refs[sparse_offsets[group] + local] =
        sorted_refs[group_starts[group] + local];
}

template <class Source>
__global__ void emit_direct_projection(
    Source source, const std::uint32_t *heads,
    const std::uint32_t *sorted_refs,
    const std::uint32_t *group_counts, const std::uint32_t *group_starts,
    const std::uint8_t *sparse_flags, std::uint32_t group_count,
    std::uint32_t *projection_keys,
    gpulsmopt2_detail::Row *projection_rows) {
  const std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= group_count) return;
  const std::uint32_t head = heads[group];
  projection_keys[group] = head;
  if (sparse_flags[group]) {
    projection_rows[group] = exact_group_row(head, 0u);
    return;
  }
  // DeviceRadixSort is stable, so the last member of an ordinary head group
  // is the newest input update, matching restored bulk/admission chronology.
  const std::uint32_t ref = sorted_refs[
      group_starts[group] + group_counts[group] - 1u];
  const bool tombstone = source_tombstone(source, ref);
  projection_rows[group] = {
      tombstone ? gpulsmopt2_detail::kInvalid : source_summary(source, ref),
      static_cast<std::uint16_t>(head),
      static_cast<std::uint16_t>(
          tombstone ? gpulsmopt2_detail::kTombstone : 0u)};
}

__global__ void build_direct_section_offsets(
    const std::uint32_t *heads, std::uint32_t count,
    std::uint32_t *offsets) {
  const std::uint32_t section = blockIdx.x * blockDim.x + threadIdx.x;
  if (section > kSections) return;
  if (section == kSections) {
    offsets[section] = count;
    return;
  }
  const std::uint32_t target = section << 16u;
  std::uint32_t low = 0u;
  std::uint32_t high = count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (heads[middle] < target) low = middle + 1u;
    else high = middle;
  }
  offsets[section] = low;
}

__global__ void build_direct_epoch_counts_and_ranks(
    const gpulsmopt2_detail::Row *rows,
    const std::uint32_t *section_offsets, std::uint32_t *section_counts,
    std::uint16_t *cell_ranks) {
  const std::uint32_t section = blockIdx.x;
  const std::uint32_t cell = threadIdx.x;
  const std::uint32_t begin = section_offsets[section];
  const std::uint32_t count =
      section_offsets[section + 1u] - begin;
  if (!cell) {
    section_counts[section] = count;
    if (!section) section_counts[kSections] = 0u;
  }
  cell_ranks[std::size_t{section} *
                 gpulsmopt2_detail::kFoundationCells + cell] =
      static_cast<std::uint16_t>(gpulsmopt2_detail::lower_bound_rows(
          rows + begin, count,
          cell * gpulsmopt2_detail::kFoundationCellKeys));
}

struct DirectRootPreparation {
  std::uint32_t projection_rows{};
  std::uint32_t logical_rows{};
  std::uint32_t sparse_heads{};
  std::uint32_t sparse_input_rows{};
};

class DirectRootWorkspace {
 public:
  explicit DirectRootWorkspace(std::uint32_t capacity)
      : capacity_(capacity), heads_a_(capacity), heads_b_(capacity),
        refs_a_(capacity), refs_b_(capacity), group_heads_(capacity),
        group_counts_(capacity), group_starts_(capacity), group_count_(1u),
        sparse_rows_(capacity), sparse_offsets_(capacity),
        sparse_flags_(capacity), sparse_heads_(capacity),
        sparse_head_count_(1u), sparse_refs_(capacity), totals_(1u) {
    projection_keys_.reset(capacity);
    projection_rows_.reset(capacity);
    section_offsets_.reset(kSections + 1u);
    section_counts_.reset(kSections + 1u);
    epoch_ranks_.reset(gpulsmopt2_detail::kLocalRankEntries);
    std::size_t sizes[4]{};
    check(cub::DeviceRadixSort::SortPairs(
              nullptr, sizes[0], heads_a_.data(), heads_b_.data(),
              refs_a_.data(), refs_b_.data(), capacity),
          "size direct head sort");
    check(cub::DeviceRunLengthEncode::Encode(
              nullptr, sizes[1], heads_b_.data(), group_heads_.data(),
              group_counts_.data(), group_count_.data(), capacity),
          "size direct head encoding");
    check(cub::DeviceScan::ExclusiveSum(
              nullptr, sizes[2], sparse_rows_.data(),
              sparse_offsets_.data(), capacity),
          "size direct sparse scan");
    check(cub::DeviceSelect::Flagged(
              nullptr, sizes[3], group_heads_.data(), sparse_flags_.data(),
              sparse_heads_.data(), sparse_head_count_.data(), capacity),
          "size direct sparse-head selection");
    sort_bytes_ = sizes[0];
    rle_bytes_ = sizes[1];
    scan_bytes_ = sizes[2];
    select_bytes_ = sizes[3];
    temporary_.reset(*std::max_element(
        std::begin(sizes), std::end(sizes)));
  }

  DirectRootPreparation prepare(
      RecordBatchView source, std::uint32_t rows, cudaStream_t stream) {
    if (!rows || rows > capacity_)
      throw std::length_error("direct root capacity");
    initialize_universal_records<<<blocks(rows), kThreads, 0, stream>>>(
        source, nullptr, nullptr, heads_a_.data(), heads_a_.data(),
        refs_a_.data(), rows);
    std::size_t bytes = sort_bytes_;
    check(cub::DeviceRadixSort::SortPairs(
              temporary_.data(), bytes, heads_a_.data(), heads_b_.data(),
              refs_a_.data(), refs_b_.data(), rows, 0, 32, stream),
          "sort direct root heads");
    bytes = rle_bytes_;
    check(cub::DeviceRunLengthEncode::Encode(
              temporary_.data(), bytes, heads_b_.data(),
              group_heads_.data(), group_counts_.data(),
              group_count_.data(), rows, stream),
          "encode direct root heads");
    std::uint32_t groups = 0u;
    check(cudaMemcpyAsync(&groups, group_count_.data(), sizeof(groups),
                          cudaMemcpyDeviceToHost, stream),
          "copy direct projection count");
    check(cudaStreamSynchronize(stream),
          "wait direct projection count");
    bytes = scan_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              temporary_.data(), bytes, group_counts_.data(),
              group_starts_.data(), groups, stream),
          "scan direct head groups");
    classify_direct_head_groups<<<groups, kThreads, 0, stream>>>(
        source, refs_b_.data(), group_counts_.data(), group_starts_.data(),
        groups, sparse_rows_.data(), sparse_flags_.data());
    bytes = scan_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              temporary_.data(), bytes, sparse_rows_.data(),
              sparse_offsets_.data(), groups, stream),
          "scan direct sparse rows");
    finish_scan_total<<<1, 1, 0, stream>>>(
        sparse_rows_.data(), sparse_offsets_.data(), group_count_.data(),
        totals_.data());
    bytes = select_bytes_;
    check(cub::DeviceSelect::Flagged(
              temporary_.data(), bytes, group_heads_.data(),
              sparse_flags_.data(), sparse_heads_.data(),
              sparse_head_count_.data(), groups, stream),
          "select direct sparse heads");
    std::uint32_t sparse[2]{};
    check(cudaMemcpyAsync(sparse, sparse_head_count_.data(),
                          sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                          stream),
          "copy direct sparse head count");
    check(cudaMemcpyAsync(sparse + 1u, totals_.data(),
                          sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                          stream),
          "copy direct sparse row count");
    check(cudaStreamSynchronize(stream), "wait direct sparse counts");

    std::uint32_t sparse_winners = 0u;
    if (sparse[1]) {
      scatter_direct_sparse_refs<<<groups, kThreads, 0, stream>>>(
          refs_b_.data(), group_counts_.data(), group_starts_.data(),
          sparse_offsets_.data(), sparse_flags_.data(), groups,
          sparse_refs_.data());
      exact_order_ = std::make_unique<ExactOrderWorkspace>(sparse[1]);
      sparse_winners = exact_order_->order(
          source, sparse[1], stream, nullptr, sparse_refs_.data())
          .winner_count;
    }
    const std::uint64_t logical =
        std::uint64_t{groups} - sparse[0] + sparse_winners;
    if (logical > std::numeric_limits<std::uint32_t>::max())
      throw std::length_error("direct logical root exceeds 32 bits");
    return {groups, static_cast<std::uint32_t>(logical), sparse[0],
            sparse[1]};
  }

  void materialize_projection(
      RecordBatchView source, std::uint32_t groups,
      std::uint32_t *projection_keys,
      gpulsmopt2_detail::Row *projection_rows,
      cudaStream_t stream) const {
    emit_direct_projection<<<blocks(groups), kThreads, 0, stream>>>(
        source, group_heads_.data(), refs_b_.data(), group_counts_.data(),
        group_starts_.data(), sparse_flags_.data(), groups,
        projection_keys, projection_rows);
    check(cudaGetLastError(), "materialize direct root projection");
  }

  void materialize_epoch(
      RecordBatchView source, std::uint32_t groups,
      cudaStream_t stream) {
    materialize_projection(source, groups, projection_keys_.data(),
                           projection_rows_.data(), stream);
    build_direct_section_offsets<<<blocks(kSections + 1u), kThreads, 0,
                                   stream>>>(
        projection_keys_.data(), groups, section_offsets_.data());
    build_direct_epoch_counts_and_ranks<<<
        kSections, gpulsmopt2_detail::kFoundationCells, 0, stream>>>(
        projection_rows_.data(), section_offsets_.data(),
        section_counts_.data(), epoch_ranks_.data());
    check(cudaGetLastError(), "materialize direct projection epoch");
  }

  const std::uint32_t *sorted_heads() const { return heads_b_.data(); }
  const std::uint32_t *sorted_refs() const { return refs_b_.data(); }
  const std::uint32_t *sparse_heads() const {
    return sparse_heads_.data();
  }
  const std::uint32_t *projection_keys() const {
    return projection_keys_.data();
  }
  const gpulsmopt2_detail::Row *projection_rows() const {
    return projection_rows_.data();
  }
  const std::uint32_t *section_offsets() const {
    return section_offsets_.data();
  }
  const std::uint32_t *section_counts() const {
    return section_counts_.data();
  }
  const std::uint16_t *epoch_ranks() const { return epoch_ranks_.data(); }
  const std::uint32_t *projection_count_device() const {
    return group_count_.data();
  }

  std::size_t bytes() const {
    return heads_a_.bytes() + heads_b_.bytes() + refs_a_.bytes() +
        refs_b_.bytes() + group_heads_.bytes() + group_counts_.bytes() +
        group_starts_.bytes() + group_count_.bytes() +
        sparse_rows_.bytes() + sparse_offsets_.bytes() +
        sparse_flags_.bytes() + sparse_heads_.bytes() +
        sparse_head_count_.bytes() + sparse_refs_.bytes() + totals_.bytes() +
        projection_keys_.bytes() + projection_rows_.bytes() +
        section_offsets_.bytes() + section_counts_.bytes() +
        epoch_ranks_.bytes() + temporary_.bytes() +
        (exact_order_ ? exact_order_->bytes() : 0u);
  }

 private:
  std::uint32_t capacity_{};
  Buffer<std::uint32_t> heads_a_, heads_b_, refs_a_, refs_b_;
  Buffer<std::uint32_t> group_heads_, group_counts_, group_starts_;
  Buffer<std::uint32_t> group_count_, sparse_rows_, sparse_offsets_;
  Buffer<std::uint8_t> sparse_flags_;
  Buffer<std::uint32_t> sparse_heads_, sparse_head_count_, sparse_refs_;
  Buffer<std::uint32_t> totals_;
  Buffer<std::uint32_t> projection_keys_, section_offsets_,
      section_counts_;
  Buffer<gpulsmopt2_detail::Row> projection_rows_;
  Buffer<std::uint16_t> epoch_ranks_;
  Buffer<std::uint8_t> temporary_;
  std::unique_ptr<ExactOrderWorkspace> exact_order_;
  std::size_t sort_bytes_{}, rle_bytes_{}, scan_bytes_{}, select_bytes_{};
};

__global__ void initialize_direct_sparse_manifest(
    DeviceSparseManifest *manifests, const RootBuildState *root,
    std::uint32_t level, std::uint32_t flags) {
  if (blockIdx.x || threadIdx.x) return;
  DeviceSparseManifest manifest{};
  manifest.generation = 1u;
  if (root && flags) {
    manifest.levels[level] = {
        reinterpret_cast<std::uint64_t>(root), root->generation, flags};
    if (flags & kSparseHasExactHeads)
      manifest.exact_level_mask = std::uint64_t{1u} << level;
    if (flags & kSparseHasCapsules)
      manifest.capsule_level_mask = std::uint64_t{1u} << level;
  }
  manifests[0] = manifest;
  manifests[1] = manifest;
}

}  // namespace gpulsm_sparse
// END INLINED COMPONENT: gpulsm_sparse/bulk.cuh
// BEGIN INLINED COMPONENT: gpulsm_sparse/sealed.cuh


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
// END INLINED COMPONENT: gpulsm_sparse/sealed.cuh
// BEGIN INLINED COMPONENT: gpulsm_sparse/read.cuh


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
// END INLINED COMPONENT: gpulsm_sparse/read.cuh
// BEGIN INLINED COMPONENT: gpulsm_sparse/ordered_read.cuh


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
// END INLINED COMPONENT: gpulsm_sparse/ordered_read.cuh
// BEGIN INLINED COMPONENT: gpulsm_sparse/range.cuh

// BEGIN INLINED COMPONENT: gpulsm_sparse/range_common.cuh


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
// END INLINED COMPONENT: gpulsm_sparse/range_common.cuh

#include <cub/block/block_reduce.cuh>

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
// END INLINED COMPONENT: gpulsm_sparse/range.cuh

class GPULSMOpt {
public:
  struct DeviceKeyBatch {
    const std::uint32_t *keys = nullptr;
    std::size_t count = 0u;
  };

  struct SparseMemoryAccounting {
    std::uint64_t manifest_bytes{};
    std::uint64_t overlay_metadata_bytes{};
    std::uint64_t workspace_bytes{};
    std::uint64_t workspace_high_water_bytes{};
    std::uint64_t capsule_reserved_bytes{};
    std::uint64_t capsule_mapped_bytes{};
    std::uint64_t capsule_live_bytes{};
    std::uint64_t capsule_garbage_bytes{};
    std::uint32_t capsule_segments{};

    std::uint64_t physical_bytes() const {
      return manifest_bytes + overlay_metadata_bytes + workspace_bytes +
          capsule_mapped_bytes;
    }
  };

  explicit GPULSMOpt(const DictionaryConfig &config)
      : batch_capacity_(std::min(
            gpulsmopt2_detail::kMaximumOperationTile,
            std::max<std::size_t>(1u, config.batch_capacity))),
        publication_capacity_(gpulsmopt2_detail::initial_storage_capacity(
            config.max_elements, batch_capacity_)),
        level_zero_capacity_(gpulsmopt2_detail::initial_level_capacity(
            config.level_zero_capacity,
            batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch,
            publication_capacity_)),
        level_pool_capacity_(
            gpulsmopt2_detail::preassigned_level_pool_capacity(
                publication_capacity_, level_zero_capacity_)),
        canonical_regular_level_count_(
            gpulsmopt2_detail::canonical_regular_level_count(
                publication_capacity_, level_zero_capacity_)),
        canonical_level_count_(gpulsmopt2_detail::canonical_level_count(
            publication_capacity_, level_zero_capacity_)),
        resident_merge_capacity_(
            gpulsmopt2_detail::select_canonical_merge_capacity() -
            gpulsmopt2_detail::kCanonicalCapacityAdjustment),
        canonical_merge_workspace_bytes_(
            std::size_t{resident_merge_capacity_} *
            sizeof(std::uint32_t) * 2u),
        maximum_resident_jobs_(
            gpulsmopt2_detail::maximum_resident_merge_jobs(
                publication_capacity_, resident_merge_capacity_)),
        route_stride_(gpulsmopt2_detail::kQuotients),
        arena_key_flags_(gpulsmopt2_detail::maximum_resident_elements<
                             gpulsmopt2_detail::Row>(),
                         level_pool_capacity_),
        arena_values_(gpulsmopt2_detail::maximum_resident_elements<
                          gpulsmopt2_detail::Row>(),
                      level_pool_capacity_),
        descriptors_(std::size_t{gpulsmopt2_detail::kQuotients} *
                     gpulsmopt2_detail::kMaximumLevels),
        route_headers_(std::size_t{gpulsmopt2_detail::kQuotients} *
                       gpulsmopt2_detail::kMaximumLevels),
        route_slices_(route_stride_ * canonical_level_count_,
                      route_stride_ * canonical_level_count_),
        route_logical_begins_(
            route_stride_ * canonical_level_count_),
        route_quotients_(
            route_stride_ * canonical_level_count_),
        level_q_logical_offsets_(
            std::size_t{canonical_level_count_} *
            (gpulsmopt2_detail::kQuotients + 1u)),
        device_manifests_(2u),
        active_device_manifest_(1u),
        query_occupied_level_mask_(1u),
        device_sparse_manifests_(2u),
        device_pending_sparse_states_(
            gpulsmopt2_detail::kBatchesPerEpoch),
        sealed_device_command_(1u),
        sealed_device_receipt_(1u),
        resident_plan_(1u),
        publication_receipt_(1u),
        level_storage_spans_(gpulsmopt2_detail::kMaximumLevels),
        canonical_cell_ranks_(
            std::size_t{canonical_level_count_} *
                gpulsmopt2_detail::kLocalRankEntries),
        operation_workspace_(
            operation_workspace_maximum_bytes(batch_capacity_),
            operation_workspace_initial_bytes(batch_capacity_)),
        sealed_workspace_(make_sealed_workspace(
            publication_capacity_, level_zero_capacity_)),
        canonical_job_prefixes_(maximum_resident_jobs_),
        canonical_next_job_(1u),
        raw_keys_(gpulsmopt2_detail::kBatchesPerEpoch * batch_capacity_),
        raw_payloads_(gpulsmopt2_detail::kBatchesPerEpoch * batch_capacity_),
        raw_offsets_(std::size_t{gpulsmopt2_detail::kBatchesPerEpoch} *
                     (gpulsmopt2_detail::kQuotients + 1u)),
        raw_signatures_(std::size_t{gpulsmopt2_detail::kBatchesPerEpoch} *
                        gpulsmopt2_detail::kQuotients),
        raw_epoch_signatures_(gpulsmopt2_detail::kQuotients),
        publication_keys_a_(gpulsmopt2_detail::kMaximumPublicationRows,
            std::min(publication_capacity_,
                batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch)),
        publication_rows_a_(gpulsmopt2_detail::kMaximumPublicationRows,
            std::min(publication_capacity_,
                batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch)),
        publication_selected_count_(1u),
        foundation_source_offsets_(gpulsmopt2_detail::kQuotients + 1u),
        foundation_section_output_counts_(gpulsmopt2_detail::kQuotients + 1u),
        balanced_merge_raw_counts_(gpulsmopt2_detail::kQuotients),
        resident_tile_job_counts_(gpulsmopt2_detail::kPlanningTiles + 1u),
        resident_tile_job_offsets_(gpulsmopt2_detail::kPlanningTiles + 1u),
        resident_job_raw_reservations_(maximum_resident_jobs_ + 1u),
        balanced_merge_jobs_(maximum_resident_jobs_),
        local_epoch_overflow_flag_(1u),
        admission_counts_(gpulsmopt2_detail::kQuotients + 1u),
        range_partials_(gpulsmopt2_detail::kRangeSchedulerBlocks),
        range_reduction_completion_(1u),
        range_fragment_total_(1u),
        range_total_receipt_(1u),
        range_hot_counts_(gpulsmopt2_detail::kQuotients + 1u),
        range_hot_offsets_(gpulsmopt2_detail::kQuotients + 1u),
        range_hot_window_offsets_(gpulsmopt2_detail::kQuotients + 1u),
        range_hot_descriptors_(gpulsmopt2_detail::kQuotients),
        range_hot_selected_count_(1u),
        range_hot_total_receipt_(1u),
        range_hot_offsets_receipt_(gpulsmopt2_detail::kQuotients + 1u) {
    CUDA_CHECK(cudaEventCreateWithFlags(&operation_done_,
                                         cudaEventDisableTiming));
    initialize_operation_workspace_views();
    ensure_radix_workspace(batch_capacity_);
    std::size_t admission_scan_bytes{};
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, admission_scan_bytes, admission_counts_.data(),
        raw_offsets_.data(), gpulsmopt2_detail::kQuotients + 1u, 0));
    admission_temp_.resize(admission_scan_bytes);
    initialize_resident_workspace();
    initialize_canonical_workspace();
    initialize_canonical_publication_graphs();
    CUDA_CHECK(cudaEventRecord(operation_done_, 0));
    reset_updates(0);
    CUDA_CHECK(cudaEventRecord(operation_done_, 0));
  }

  GPULSMOpt(const GPULSMOpt &) = delete;
  GPULSMOpt &operator=(const GPULSMOpt &) = delete;

  ~GPULSMOpt() {
    if (operation_done_) {
      cudaEventSynchronize(operation_done_);
      cudaEventDestroy(operation_done_);
    }
    for (cudaGraphExec_t graph_exec : canonical_publication_graph_execs_)
      if (graph_exec) cudaGraphExecDestroy(graph_exec);
  }

  void clear(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    begin_operation(stream);
    reset_updates(stream);
    end_operation(stream);
  }

  void bulk_build(const std::uint32_t *keys, const std::uint32_t *values,
                  std::size_t count, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    if ((count && (!keys || !values)) || count > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("invalid GPULSMOpt initial input");
    begin_operation(stream);
    reset_updates(stream);
    if (!count) {
      end_operation(stream);
      return;
    }
    const std::uint32_t n = static_cast<std::uint32_t>(count);
    gpulsmopt2_detail::Buffer<std::uint32_t> sorted_keys(n), sorted_values(n);
    const std::uint32_t tile_count = static_cast<std::uint32_t>(
        (count + gpulsmopt2_detail::kBulkRootSinkTileRows - 1u) /
        gpulsmopt2_detail::kBulkRootSinkTileRows);
    gpulsmopt2_detail::Buffer<std::uint32_t> tile_counts(tile_count + 1u);
    gpulsmopt2_detail::Buffer<std::uint32_t> tile_offsets(tile_count + 1u);
    std::size_t sort_bytes{};
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, keys, sorted_keys.data(), values,
        sorted_values.data(), n, 0, 32, stream));
    std::size_t scan_bytes{};
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, tile_counts.data(), tile_offsets.data(),
        tile_count + 1u, stream));
    gpulsmopt2_detail::Buffer<std::uint8_t> temporary(
        std::max(sort_bytes, scan_bytes));
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        temporary.data(), sort_bytes, keys, sorted_keys.data(), values,
        sorted_values.data(), n, 0, 32, stream));
    gpulsmopt2_detail::count_bulk_root_sink_tiles_kernel<<<
        tile_count, gpulsmopt2_detail::kThreads, 0, stream>>>(
            sorted_keys.data(), n, tile_counts.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(
        tile_counts.data() + tile_count, 0, sizeof(std::uint32_t), stream));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        temporary.data(), scan_bytes, tile_counts.data(),
        tile_offsets.data(), tile_count + 1u, stream));
    std::uint32_t base_count{};
    CUDA_CHECK(cudaMemcpyAsync(&base_count, tile_offsets.data() + tile_count,
                               sizeof(base_count), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const std::uint32_t level = initial_level_for_records(base_count);
    ensure_level_storage_mapped(level, stream);
    const std::uint64_t destination = level_begin(level);
    const std::uint64_t capacity = level_capacity(level);
    if (base_count > capacity)
      throw std::bad_alloc();
    gpulsmopt2_detail::deposit_bulk_root_sink_kernel<<<
        tile_count, gpulsmopt2_detail::kThreads, 0, stream>>>(
            sorted_keys.data(), sorted_values.data(), n,
            tile_offsets.data(), resident_rows(), destination);
    gpulsmopt2_detail::build_bulk_root_sink_quotient_offsets_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            sorted_keys.data(), n, tile_offsets.data(), tile_count,
            foundation_source_offsets_.data());
    CUDA_CHECK(cudaGetLastError());
    gpulsmopt2_detail::ResidentPublicationPlan build_plan{};
    build_plan.destination_level = level;
    build_plan.output_begin = destination;
    build_plan.output_capacity = capacity;
    build_plan.status = gpulsmopt2_detail::kPublicationSuccess;
    CUDA_CHECK(cudaMemcpyAsync(
        resident_plan_.data(), &build_plan, sizeof(build_plan),
        cudaMemcpyHostToDevice, stream));
    gpulsmopt2_detail::build_canonical_rank_from_run_kernel<<<
        gpulsmopt2_detail::kQuotients,
        gpulsmopt2_detail::kFoundationCells, 0, stream>>>(
            resident_rows(), destination,
            foundation_source_offsets_.data(), level,
            canonical_cell_ranks_.data());
    gpulsmopt2_detail::finalize_canonical_level_metadata_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            foundation_source_offsets_.data(), level_storage_spans_.data(),
            resident_plan_.data(), descriptors_.data(),
            static_cast<std::uint32_t>(route_stride_),
            route_headers_.data(), route_slices_.data(),
            route_logical_begins_.data(), route_quotients_.data(),
            level_q_logical_offsets_.data());
    level_counts_[level] = base_count;
    host_occupied_level_mask_ = std::uint64_t{1} << level;
    gpulsmopt2_detail::initialize_device_manifest_kernel<<<1, 1, 0, stream>>>(
        device_manifests_.data(), active_device_manifest_.data(),
        query_occupied_level_mask_.data(), level, base_count, 0u);
    refresh_active_levels();
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

  void bulk_build(const DeviceRecordBatchView &batch,
                  cudaStream_t stream) {
    validate_record_batch(batch);
    // This is the protected representation-capability dispatch: call the
    // restored bulk_build entry itself, with no universal or sparse setup.
    if (ordinary_compatible(batch) &&
        batch.uniform_operation == DeviceMutation::put) {
      bulk_build(
          reinterpret_cast<const std::uint32_t *>(batch.keys.bytes),
          reinterpret_cast<const std::uint32_t *>(batch.values.bytes),
          static_cast<std::size_t>(batch.count), stream);
      return;
    }

    std::lock_guard<std::mutex> lock(operation_mutex_);
    if (batch.count >= gpulsm_sparse::kCompletionIncoming)
      throw std::length_error(
          "GPULSMOpt direct mixed root exceeds locator capacity");
    begin_operation(stream);
    reset_updates(stream);
    if (!batch.count) {
      end_operation(stream);
      return;
    }

    const std::uint32_t rows = static_cast<std::uint32_t>(batch.count);
    const gpulsm_sparse::RecordBatchView source =
        gpulsm_sparse::record_batch_view(batch);
    gpulsm_sparse::DirectRootWorkspace workspace(rows);
    const gpulsm_sparse::DirectRootPreparation prepared =
        workspace.prepare(source, rows, stream);
    const std::uint32_t level = initial_level_for_records(
        prepared.logical_rows);
    ensure_level_storage_mapped(level, stream);
    ensure_publication_capacity(prepared.projection_rows, stream);
    workspace.materialize_projection(
        source, prepared.projection_rows, publication_keys_a_.data(),
        publication_rows_a_.data(), stream);
    CUDA_CHECK(cudaMemcpyAsync(
        publication_selected_count_.data(), &prepared.projection_rows,
        sizeof(prepared.projection_rows), cudaMemcpyHostToDevice, stream));
    gpulsmopt2_detail::build_query_quotient_offsets_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            publication_keys_a_.data(), prepared.projection_rows,
            foundation_source_offsets_.data());

    const std::uint64_t destination = level_begin(level);
    const std::uint64_t capacity = level_capacity(level);
    if (prepared.projection_rows > capacity)
      throw std::bad_alloc();
    gpulsmopt2_detail::copy_canonical_epoch_kernel<<<
        blocks(prepared.projection_rows), gpulsmopt2_detail::kThreads, 0,
        stream>>>(publication_rows_a_.data(),
                  publication_selected_count_.data(), resident_rows(),
                  destination);
    gpulsmopt2_detail::ResidentPublicationPlan build_plan{};
    build_plan.destination_level = level;
    build_plan.output_begin = destination;
    build_plan.output_capacity = capacity;
    build_plan.survivor_count = prepared.projection_rows;
    build_plan.status = gpulsmopt2_detail::kPublicationSuccess;
    CUDA_CHECK(cudaMemcpyAsync(
        resident_plan_.data(), &build_plan, sizeof(build_plan),
        cudaMemcpyHostToDevice, stream));
    gpulsmopt2_detail::build_canonical_rank_from_run_kernel<<<
        gpulsmopt2_detail::kQuotients,
        gpulsmopt2_detail::kFoundationCells, 0, stream>>>(
            resident_rows(), destination,
            foundation_source_offsets_.data(), level,
            canonical_cell_ranks_.data());
    gpulsmopt2_detail::finalize_canonical_level_metadata_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            foundation_source_offsets_.data(), level_storage_spans_.data(),
            resident_plan_.data(), descriptors_.data(),
            static_cast<std::uint32_t>(route_stride_),
            route_headers_.data(), route_slices_.data(),
            route_logical_begins_.data(), route_quotients_.data(),
            level_q_logical_offsets_.data());

    std::uint32_t sparse_flags = 0u;
    std::shared_ptr<gpulsm_sparse::StagedRootOverlay> staged;
    if (prepared.sparse_heads) {
      if (!sparse_refinement_workspace_)
        sparse_refinement_workspace_ = std::make_unique<
            gpulsm_sparse::SparseRefinementWorkspace>();
      sparse_refinement_workspace_->prepare_direct_roster(
          workspace.sparse_heads(), prepared.sparse_heads, stream);
      gpulsm_sparse::CompletionSourceView completion{};
      completion.incoming = source;
      completion.incoming_sorted_heads = workspace.sorted_heads();
      completion.incoming_sorted_refs = workspace.sorted_refs();
      completion.incoming_records = rows;
      staged =
          std::make_shared<gpulsm_sparse::StagedRootOverlay>();
      const gpulsm_sparse::SparseRefinementResult refined =
          sparse_refinement_workspace_->refine(
              completion, 0u, resident_rows(), descriptors_.data(), level,
              destination, prepared.projection_rows, true, 0u,
              next_capsule_segment_ordinal_, *staged, stream);
      if (staged->state.logical_count != prepared.logical_rows)
        throw std::logic_error(
            "direct root logical/projection count mismatch");
      if (refined.allocated_capsule_bytes) {
        if (next_capsule_segment_ordinal_ ==
            std::numeric_limits<std::uint32_t>::max())
          throw std::overflow_error("GPULSMOpt capsule ordinal overflow");
        ++next_capsule_segment_ordinal_;
      }
      if (refined.exact_heads)
        sparse_flags |= gpulsm_sparse::kSparseHasExactHeads;
      if (refined.capsules)
        sparse_flags |= gpulsm_sparse::kSparseHasCapsules;
    }

    level_counts_[level] = prepared.projection_rows;
    host_occupied_level_mask_ = std::uint64_t{1u} << level;
    gpulsmopt2_detail::initialize_device_manifest_kernel<<<1, 1, 0,
                                                           stream>>>(
        device_manifests_.data(), active_device_manifest_.data(),
        query_occupied_level_mask_.data(), level,
        prepared.projection_rows, 0u);
    gpulsm_sparse::initialize_direct_sparse_manifest<<<1, 1, 0, stream>>>(
        device_sparse_manifests_.data(),
        sparse_flags ? staged->device_state.data() : nullptr, level,
        sparse_flags);
    if (sparse_flags) {
      sparse_roots_[level] = std::move(staged);
      const std::uint64_t bit = std::uint64_t{1u} << level;
      if (sparse_flags & gpulsm_sparse::kSparseHasExactHeads)
        sparse_exact_level_mask_ |= bit;
      if (sparse_flags & gpulsm_sparse::kSparseHasCapsules)
        sparse_capsule_level_mask_ |= bit;
    }
    note_sparse_workspace_high_water(workspace.bytes());
    refresh_active_levels();
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

  void insert(const DeviceKeyValueBatch &batch, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    reject_updates_after_publication_failure();
    admit(batch.keys, batch.values, batch.count, false, stream);
  }

  void insert(const DeviceRecordBatchView &batch, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    reject_updates_after_publication_failure();
    validate_record_batch(batch);
    if (!batch.count) return;
    if (ordinary_compatible(batch)) {
      const auto *keys = reinterpret_cast<const std::uint32_t *>(
          batch.keys.bytes);
      const bool tombstone =
          batch.uniform_operation == DeviceMutation::erase;
      const auto *values = tombstone ? nullptr
          : reinterpret_cast<const std::uint32_t *>(batch.values.bytes);
      admit(keys, values, static_cast<std::size_t>(batch.count),
            tombstone, stream);
      return;
    }
    admit_mixed(gpulsm_sparse::record_batch_view(batch), stream);
  }

  void erase(const DeviceKeyBatch &batch, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    reject_updates_after_publication_failure();
    admit(batch.keys, nullptr, batch.count, true, stream);
  }

  void lookup(const DeviceLookupBatch &batch, cudaStream_t stream,
              bool quotients_grouped = false) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    prepare_failed_epoch_for_reads(stream);
    if (!has_sparse_read_state()) {
      lookup_locked(batch, stream, quotients_grouped);
      return;
    }
    if (!batch.count) return;
    if (!batch.queries || !batch.out_values)
      throw std::invalid_argument("invalid GPULSMOpt lookup");
    gpulsm_sparse::RecordBatchView queries{};
    queries.keys = device_head4_source(batch.queries);
    queries.count = batch.count;
    queries.head4_words = true;
    gpulsm_sparse::SparseLookupOutput output{};
    output.summaries = batch.out_values;
    output.found = batch.out_found;
    lookup_sparse_locked(queries, output, stream, quotients_grouped);
  }

  void lookup(const DeviceLookupBatchView &batch, cudaStream_t stream,
              bool quotients_grouped = false) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    prepare_failed_epoch_for_reads(stream);
    validate_lookup_batch(batch);
    if (!batch.queries.count) return;

    const bool native_source =
        batch.queries.key_encoding == DeviceKeyEncoding::head4_words &&
        batch.queries.keys.bytes && !batch.queries.keys.offsets &&
        batch.queries.keys.stride == sizeof(std::uint32_t);
    const bool native_output =
        batch.output.values.layout == DeviceSinkLayout::fixed_stride &&
        batch.output.values.bytes && !batch.output.values.offsets &&
        batch.output.values.stride == sizeof(std::uint32_t) &&
        batch.queries.count <=
            batch.output.values.capacity_bytes / sizeof(std::uint32_t) &&
        !batch.output.required_value_lengths && !batch.output.overflow;
    if (!has_sparse_read_state() && native_source && native_output) {
      lookup_locked(
          DeviceLookupBatch{
              reinterpret_cast<const std::uint32_t *>(
                  batch.queries.keys.bytes),
              static_cast<std::size_t>(batch.queries.count),
              reinterpret_cast<std::uint32_t *>(batch.output.values.bytes),
              batch.output.found},
          stream, quotients_grouped);
      return;
    }

    gpulsm_sparse::SparseLookupOutput output{};
    output.values = batch.output.values;
    output.value_lengths = batch.output.required_value_lengths;
    output.found = batch.output.found;
    output.overflow = batch.output.overflow;
    lookup_sparse_locked(
        gpulsm_sparse::key_batch_view(batch.queries), output, stream,
        quotients_grouped);
  }

  void range(const DeviceRangeOutputBatch &batch, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    prepare_failed_epoch_for_reads(stream);
    if (!has_exact_read_state()) {
      range_locked(batch, stream);
      return;
    }
    if (!batch.query_count) return;
    if (!batch.lo || !batch.hi || !batch.out_sums)
      throw std::invalid_argument("invalid GPULSMOpt range input");
    if (batch.query_count > gpulsmopt2_detail::kMaximumOperationTile) {
      for (std::size_t begin = 0u; begin < batch.query_count;
           begin += gpulsmopt2_detail::kMaximumOperationTile) {
        const std::size_t count = std::min(
            batch.query_count - begin,
            gpulsmopt2_detail::kMaximumOperationTile);
        gpulsm_sparse::range_gate::RangeQueryBatch queries{};
        queries.lower.keys = device_head4_source(batch.lo + begin);
        queries.lower.count = count;
        queries.lower.head4_words = true;
        queries.upper.keys = device_head4_source(batch.hi + begin);
        queries.upper.count = count;
        queries.upper.head4_words = true;
        queries.count = static_cast<std::uint32_t>(count);
        range_sparse_locked(
            queries, batch.out_sums + begin, nullptr, stream);
      }
      return;
    }
    gpulsm_sparse::range_gate::RangeQueryBatch queries{};
    queries.lower.keys = device_head4_source(batch.lo);
    queries.lower.count = batch.query_count;
    queries.lower.head4_words = true;
    queries.upper.keys = device_head4_source(batch.hi);
    queries.upper.count = batch.query_count;
    queries.upper.head4_words = true;
    queries.count = static_cast<std::uint32_t>(batch.query_count);
    range_sparse_locked(queries, batch.out_sums, nullptr, stream);
  }

  void range(const DeviceRangeSumBatchView &batch, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    prepare_failed_epoch_for_reads(stream);
    validate_range_batch(batch);
    const std::uint64_t count = batch.queries.lower.count;
    if (!count) return;
    const auto native_source = [](const DeviceKeyBatchView &source) {
      return source.key_encoding == DeviceKeyEncoding::head4_words &&
          source.keys.bytes && !source.keys.offsets &&
          source.keys.stride == sizeof(std::uint32_t);
    };
    if (!has_exact_read_state() && native_source(batch.queries.lower) &&
        native_source(batch.queries.upper) && !batch.valid) {
      range_locked(
          DeviceRangeOutputBatch{
              reinterpret_cast<const std::uint32_t *>(
                  batch.queries.lower.keys.bytes),
              reinterpret_cast<const std::uint32_t *>(
                  batch.queries.upper.keys.bytes),
              static_cast<std::size_t>(count), batch.out_sums},
          stream);
      return;
    }
    gpulsm_sparse::range_gate::RangeQueryBatch queries{
        gpulsm_sparse::key_batch_view(batch.queries.lower),
        gpulsm_sparse::key_batch_view(batch.queries.upper),
        static_cast<std::uint32_t>(count)};
    range_sparse_locked(queries, batch.out_sums, batch.valid, stream);
  }

  void successor(const DeviceSuccessorBatch &batch, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    prepare_failed_epoch_for_reads(stream);
    if (!has_exact_read_state()) {
      successor_locked(batch, stream);
      return;
    }
    if (!batch.count) return;
    if (!batch.queries || !batch.out_keys)
      throw std::invalid_argument("invalid GPULSMOpt successor input");
    gpulsm_sparse::RecordBatchView queries{};
    queries.keys = device_head4_source(batch.queries);
    queries.count = batch.count;
    queries.head4_words = true;
    gpulsm_sparse::SparseSuccessorOutput output{};
    output.head4_words = batch.out_keys;
    successor_sparse_locked(queries, output, stream);
  }

  void successor(const DeviceSuccessorBatchView &batch,
                 cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    prepare_failed_epoch_for_reads(stream);
    validate_successor_batch(batch);
    if (!batch.queries.count) return;
    const bool native_source =
        batch.queries.key_encoding == DeviceKeyEncoding::head4_words &&
        batch.queries.keys.bytes && !batch.queries.keys.offsets &&
        batch.queries.keys.stride == sizeof(std::uint32_t);
    const bool native_output =
        batch.output.keys.layout == DeviceSinkLayout::fixed_stride &&
        batch.output.keys.bytes && !batch.output.keys.offsets &&
        batch.output.keys.stride == sizeof(std::uint32_t) &&
        batch.queries.count <=
            batch.output.keys.capacity_bytes / sizeof(std::uint32_t) &&
        !batch.output.required_key_lengths && !batch.output.found &&
        !batch.output.overflow;
    if (!has_exact_read_state() && native_source && native_output) {
      successor_locked(
          DeviceSuccessorBatch{
              reinterpret_cast<const std::uint32_t *>(
                  batch.queries.keys.bytes),
              static_cast<std::size_t>(batch.queries.count),
              reinterpret_cast<std::uint32_t *>(batch.output.keys.bytes)},
          stream);
      return;
    }
    gpulsm_sparse::SparseSuccessorOutput output{};
    output.keys = batch.output.keys;
    output.key_lengths = batch.output.required_key_lengths;
    output.found = batch.output.found;
    output.overflow = batch.output.overflow;
    successor_sparse_locked(
        gpulsm_sparse::key_batch_view(batch.queries), output, stream);
  }

  std::uint32_t canonical_carry_status() const {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    auto *self = const_cast<GPULSMOpt *>(this);
    self->resolve_publication_receipt();
    gpulsmopt2_detail::ResidentPublicationPlan plan{};
    CUDA_CHECK(cudaMemcpy(
        &plan, self->resident_plan_.data(), sizeof(plan),
        cudaMemcpyDeviceToHost));
    return plan.status;
  }

private:

  static void validate_record_batch(const DeviceRecordBatchView &batch) {
    if (batch.count > std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("GPULSMOpt record batch is too large");
    if (!batch.count) return;
    if (batch.key_encoding == DeviceKeyEncoding::head4_words) {
      if (!batch.keys.bytes || batch.keys.offsets ||
          batch.keys.stride != sizeof(std::uint32_t))
        throw std::invalid_argument(
            "invalid GPULSMOpt ordered-head source");
    } else if (!batch.keys.offsets && batch.keys.stride &&
               !batch.keys.bytes) {
      throw std::invalid_argument("invalid GPULSMOpt key source");
    }
    if (!batch.operations &&
        batch.uniform_operation == DeviceMutation::put &&
        !batch.values.offsets && batch.values.stride &&
        !batch.values.bytes)
      throw std::invalid_argument("invalid GPULSMOpt value source");
  }

  static bool ordinary_compatible(const DeviceRecordBatchView &batch) {
    if (batch.key_encoding != DeviceKeyEncoding::head4_words ||
        batch.keys.offsets ||
        batch.keys.stride != sizeof(std::uint32_t) ||
        batch.operations || batch.range_contributions)
      return false;
    if (batch.uniform_operation == DeviceMutation::erase) return true;
    return batch.values.bytes && !batch.values.offsets &&
        batch.values.stride == sizeof(std::uint32_t);
  }

  static std::uint64_t checked_byte_displacement(
      std::uint64_t begin, std::uint64_t stride) {
    if (begin && stride >
        std::numeric_limits<std::uint64_t>::max() / begin)
      throw std::overflow_error("GPULSMOpt byte displacement overflow");
    return begin * stride;
  }

  static gpulsm_sparse::ByteSource slice_byte_source(
      gpulsm_sparse::ByteSource source, std::uint64_t begin) {
    if (source.offsets) {
      source.offsets += begin;
    } else if (source.bytes) {
      source.bytes += checked_byte_displacement(begin, source.stride);
    }
    return source;
  }

  static gpulsm_sparse::RecordBatchView slice_record_batch(
      gpulsm_sparse::RecordBatchView source, std::uint64_t begin,
      std::uint64_t count) {
    source.keys = slice_byte_source(source.keys, begin);
    source.values = slice_byte_source(source.values, begin);
    if (source.operations) source.operations += begin;
    if (source.range_contributions) source.range_contributions += begin;
    source.count = count;
    return source;
  }

  static gpulsm_sparse::SparseLookupOutput slice_lookup_output(
      gpulsm_sparse::SparseLookupOutput output, std::uint64_t begin) {
    if (output.summaries) output.summaries += begin;
    if (output.value_lengths) output.value_lengths += begin;
    if (output.found) output.found += begin;
    if (output.overflow) output.overflow += begin;
    if (output.values.layout == DeviceSinkLayout::packed) {
      if (output.values.offsets) output.values.offsets += begin;
    } else if (output.values.bytes) {
      const std::uint64_t displacement = checked_byte_displacement(
          begin, output.values.stride);
      output.values.bytes += displacement;
      output.values.capacity_bytes =
          displacement <= output.values.capacity_bytes
          ? output.values.capacity_bytes - displacement : 0u;
    }
    return output;
  }

  static void validate_lookup_batch(const DeviceLookupBatchView &batch) {
    if (batch.queries.count > std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("GPULSMOpt lookup batch is too large");
    if (!batch.queries.count) return;
    const auto &keys = batch.queries.keys;
    if (batch.queries.key_encoding == DeviceKeyEncoding::head4_words) {
      if (!keys.bytes || keys.offsets ||
          keys.stride != sizeof(std::uint32_t))
        throw std::invalid_argument("invalid GPULSMOpt lookup heads");
    } else if (!keys.offsets && !keys.bytes) {
      throw std::invalid_argument("invalid GPULSMOpt lookup keys");
    }
    const auto &output = batch.output;
    if (!output.values.bytes && !output.required_value_lengths &&
        !output.found && !output.overflow)
      throw std::invalid_argument("GPULSMOpt lookup has no output");
    if (output.values.bytes &&
        output.values.layout == DeviceSinkLayout::packed &&
        !output.values.offsets)
      throw std::invalid_argument("packed GPULSMOpt output needs offsets");
  }

  bool has_sparse_read_state() const {
    return pending_sparse_exact_ || pending_sparse_capsules_ ||
        sparse_exact_level_mask_ || sparse_capsule_level_mask_;
  }

  bool has_exact_read_state() const {
    return pending_sparse_exact_ || sparse_exact_level_mask_;
  }

  static gpulsm_sparse::SparseSuccessorOutput slice_successor_output(
      gpulsm_sparse::SparseSuccessorOutput output, std::uint64_t begin) {
    if (output.head4_words) output.head4_words += begin;
    if (output.key_lengths) output.key_lengths += begin;
    if (output.found) output.found += begin;
    if (output.overflow) output.overflow += begin;
    if (output.keys.layout == DeviceSinkLayout::packed) {
      if (output.keys.offsets) output.keys.offsets += begin;
    } else if (output.keys.bytes) {
      const std::uint64_t displacement = checked_byte_displacement(
          begin, output.keys.stride);
      output.keys.bytes += displacement;
      output.keys.capacity_bytes = displacement <= output.keys.capacity_bytes
          ? output.keys.capacity_bytes - displacement : 0u;
    }
    return output;
  }

  static void validate_successor_batch(
      const DeviceSuccessorBatchView &batch) {
    if (batch.queries.count > std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("GPULSMOpt successor batch is too large");
    if (!batch.queries.count) return;
    const auto &keys = batch.queries.keys;
    if (batch.queries.key_encoding == DeviceKeyEncoding::head4_words) {
      if (!keys.bytes || keys.offsets ||
          keys.stride != sizeof(std::uint32_t))
        throw std::invalid_argument("invalid GPULSMOpt successor heads");
    } else if (!keys.offsets && !keys.bytes) {
      throw std::invalid_argument("invalid GPULSMOpt successor keys");
    }
    const auto &output = batch.output;
    if (!output.keys.bytes && !output.required_key_lengths &&
        !output.found && !output.overflow)
      throw std::invalid_argument("GPULSMOpt successor has no output");
    if (output.keys.bytes &&
        output.keys.layout == DeviceSinkLayout::packed &&
        !output.keys.offsets)
      throw std::invalid_argument(
          "packed GPULSMOpt successor output needs offsets");
  }

  static void validate_range_batch(const DeviceRangeSumBatchView &batch) {
    const std::uint64_t count = batch.queries.lower.count;
    if (count != batch.queries.upper.count ||
        count > std::numeric_limits<std::uint32_t>::max() ||
        (count && !batch.out_sums))
      throw std::invalid_argument("invalid GPULSMOpt range batch");
    if (!count) return;
    const auto validate = [](const DeviceKeyBatchView &source) {
      if (source.key_encoding == DeviceKeyEncoding::head4_words) {
        return source.keys.bytes && !source.keys.offsets &&
            source.keys.stride == sizeof(std::uint32_t);
      }
      return source.keys.offsets || source.keys.bytes;
    };
    if (!validate(batch.queries.lower) || !validate(batch.queries.upper))
      throw std::invalid_argument("invalid GPULSMOpt range endpoints");
  }

  void admit_mixed(gpulsm_sparse::RecordBatchView source,
                   cudaStream_t stream) {
    begin_operation(stream);
    std::uint64_t consumed = 0u;
    bool incomplete = false;
    if (pending_batches_) {
      const std::uint64_t slots =
          gpulsmopt2_detail::kBatchesPerEpoch - pending_batches_;
      const std::uint64_t prefix_rows = slots * batch_capacity_;
      if (source.count >= prefix_rows) {
        while (pending_batches_ <
               gpulsmopt2_detail::kBatchesPerEpoch) {
          admit_mixed_tile(
              slice_record_batch(source, consumed, batch_capacity_),
              static_cast<std::uint32_t>(batch_capacity_), stream);
          consumed += batch_capacity_;
        }
        if (publication_receipt_pending_)
          resolve_publication_receipt_on_stream(stream);
        if (publication_failed_) incomplete = true;
      }
    }
    if (!incomplete && consumed < source.count)
      consumed += try_admit_sealed_mixed(
          slice_record_batch(
              source, consumed, source.count - consumed),
          stream);
    while (!incomplete && consumed < source.count) {
      const std::uint32_t tile_count = static_cast<std::uint32_t>(
          std::min<std::uint64_t>(source.count - consumed,
                                 batch_capacity_));
      admit_mixed_tile(
          slice_record_batch(source, consumed, tile_count),
          tile_count, stream);
      consumed += tile_count;
      if (consumed < source.count && publication_receipt_pending_)
        resolve_publication_receipt_on_stream(stream);
      if (consumed < source.count && publication_failed_) {
        incomplete = true;
        break;
      }
    }
    end_operation(stream);
    if (incomplete)
      throw std::runtime_error(
          "GPULSMOpt publication failed while tiling a mixed update; "
          "accepted pending records were preserved");
  }

  void admit_mixed_tile(gpulsm_sparse::RecordBatchView source,
                        std::uint32_t count, cudaStream_t stream) {
    const std::uint32_t slot = pending_batches_;
    if (slot >= gpulsmopt2_detail::kBatchesPerEpoch)
      throw std::logic_error("GPULSMOpt pending slot overflow");
    // Ordinary epochs deliberately do not touch sparse state.  Clear the
    // tiny sparse slot table lazily when the first exceptional tile of a new
    // generation arrives, so untouched ordinary slots can never expose
    // pointers retained from an older epoch.
    if (pending_sparse_device_generation_ != pending_sparse_generation_) {
      CUDA_CHECK(cudaMemsetAsync(
          device_pending_sparse_states_.data(), 0,
          device_pending_sparse_states_.bytes(), stream));
      pending_sparse_device_generation_ = pending_sparse_generation_;
    }
    pending_records_ += count;
    pending_has_tombstones_ |= source.operations != nullptr ||
        source.uniform_operation == DeviceMutation::erase;
    if (!sparse_pending_workspace_)
      sparse_pending_workspace_ =
          std::make_unique<gpulsm_sparse::PendingWorkspace>(
              static_cast<std::uint32_t>(batch_capacity_));

    auto staged =
        std::make_shared<gpulsm_sparse::StagedPendingSlotOverlay>();
    std::uint32_t *batch_offsets = raw_offsets_.data() +
        std::size_t{slot} * (gpulsmopt2_detail::kQuotients + 1u);
    std::uint64_t *batch_signatures = raw_signatures_.data() +
        std::size_t{slot} * gpulsmopt2_detail::kQuotients;
    std::uint32_t *destination_keys = raw_keys_.data() +
        std::size_t{slot} * batch_capacity_;
    gpulsmopt2_detail::RawPayload *destination_payloads =
        raw_payloads_.data() + std::size_t{slot} * batch_capacity_;
    const std::uint32_t segment_ordinal = next_capsule_segment_ordinal_;
    const gpulsm_sparse::PendingBuildResult result =
        sparse_pending_workspace_->build_batch(
            source, count, slot, static_cast<std::uint32_t>(batch_capacity_),
            destination_keys, destination_payloads, batch_offsets,
            batch_signatures, *staged, segment_ordinal, stream);
    note_sparse_workspace_high_water();
    gpulsmopt2_detail::commit_admission_metadata_kernel<<<
        gpulsmopt2_detail::kQuotients / gpulsmopt2_detail::kThreads,
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            admission_counts_.data(), batch_signatures,
            raw_epoch_signatures_.data());

    if (result.capsules) {
      if (next_capsule_segment_ordinal_ ==
          std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("GPULSMOpt capsule ordinal overflow");
      ++next_capsule_segment_ordinal_;
      gpulsm_sparse::PendingSlotBuildState state{};
      state.pages = reinterpret_cast<std::uint64_t>(staged->pages.data());
      state.indexes = reinterpret_cast<std::uint64_t>(
          staged->indexes.data());
      state.exact_heads = reinterpret_cast<std::uint64_t>(
          staged->exact_heads.data());
      state.exact_refs = reinterpret_cast<std::uint64_t>(
          staged->exact_refs.data());
      state.special_heads = reinterpret_cast<std::uint64_t>(
          staged->special_heads.data());
      state.exceptions = reinterpret_cast<std::uint64_t>(
          staged->exceptions.data());
      state.page_count = result.pages;
      state.capsule_count = result.capsules;
      state.exact_head_count = result.exact_heads;
      state.exact_ref_count = result.exact_refs;
      state.special_head_count = result.special_heads;
      state.exception_count = result.capsules;
      state.record_count = count;
      state.segment_ordinal = segment_ordinal;
      state.generation = pending_sparse_generation_;
      state.live_bytes = result.capsule_bytes;
      staged->state = state;
      CUDA_CHECK(cudaMemcpyAsync(
          device_pending_sparse_states_.data() + slot, &staged->state,
          sizeof(staged->state), cudaMemcpyHostToDevice, stream));
      sparse_pending_slots_[slot] = std::move(staged);
      pending_sparse_exact_ |= result.exact_heads != 0u;
      pending_sparse_capsules_ = true;
    } else {
      sparse_pending_slots_[slot].reset();
    }
    CUDA_CHECK(cudaGetLastError());
    raw_batch_counts_[slot] = count;
    ++pending_batches_;
    if (pending_batches_ == gpulsmopt2_detail::kBatchesPerEpoch)
      publish_epoch(stream);
  }

  gpulsmopt2_detail::TqrjLookupWorkspace borrow_tqrj_lookup_workspace(
      std::uint32_t maximum_query_tiles) {
    std::uint8_t *operation_storage = operation_workspace_.data();
    std::uint32_t *query_ids = reinterpret_cast<std::uint32_t *>(
        operation_storage);
    // Reservation ranks are dead immediately after scatter.  Their phase
    // starts at the exact table's later phase address, so no copy or second
    // allocation is needed.
    std::uint32_t *reservation_ranks =
        reinterpret_cast<std::uint32_t *>(
            operation_storage + tqrj_hash_table_offset(batch_capacity_));
    std::uint8_t *hash_task_storage =
        operation_storage + tqrj_hash_task_offset(batch_capacity_);
    gpulsmopt2_detail::TqrjHashTile *query_tiles =
        reinterpret_cast<gpulsmopt2_detail::TqrjHashTile *>(
            operation_storage + tqrj_hash_tile_offset(batch_capacity_));

    gpulsmopt2_detail::TqrjLookupWorkspace workspace{};
    workspace.grouped_queries = publication_keys_a_.data();
    workspace.query_ids = query_ids;
    workspace.reservation_ranks = reservation_ranks;
    workspace.active_quotients = range_hot_window_offsets_.data();
    workspace.active_query_counts = foundation_section_output_counts_.data();
    workspace.active_query_offsets = foundation_source_offsets_.data();
    workspace.active_quotient_count = range_hot_selected_count_.data();
    workspace.query_bases = reinterpret_cast<std::uint32_t *>(
        operation_storage + tqrj_hash_query_bases_offset(batch_capacity_));
    workspace.hash_tasks =
        reinterpret_cast<gpulsmopt2_detail::TqrjHashTask *>(
            hash_task_storage);
    workspace.hash_counters = reinterpret_cast<std::uint32_t *>(
        hash_task_storage + gpulsmopt2_detail::kTqrjHashTaskBytes);
    workspace.hash_task_count = publication_selected_count_.data();
    workspace.hash_winners = reinterpret_cast<unsigned long long *>(
        publication_rows_a_.data());
    workspace.hash_query_tiles = query_tiles;
    workspace.hash_pending_tiles = query_tiles + maximum_query_tiles;
    workspace.hash_table = reinterpret_cast<std::uint32_t *>(
        operation_storage + tqrj_hash_table_offset(batch_capacity_));
    workspace.hash_table_capacity =
        tqrj_hash_maximum_entries(batch_capacity_);
    return workspace;
  }

  void lookup_locked(const DeviceLookupBatch &batch, cudaStream_t stream,
                     bool quotients_grouped) {
    if (!batch.count) return;
    if (!batch.queries || !batch.out_values)
      throw std::invalid_argument("invalid GPULSMOpt lookup");
    const std::uint64_t tqrj_dense_threshold =
        std::uint64_t{pending_batches_} * gpulsmopt2_detail::kQuotients *
        gpulsmopt2_detail::kTqrjDenseRowsPerSection;
    const bool use_tqrj =
        batch.count >= gpulsmopt2_detail::kQuotients * 4u &&
        batch.count <= batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch &&
        pending_batches_ >= gpulsmopt2_detail::kTqrjMinimumBatches &&
        std::uint64_t{pending_records_} >= tqrj_dense_threshold;
    if (use_tqrj) {
      if (publication_keys_a_.size() < batch.count ||
          publication_rows_a_.size() < batch.count) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        publication_keys_a_.grow(batch.count);
        publication_rows_a_.grow(batch.count);
      }
      if (operation_workspace_.size() <
              tqrj_hash_workspace_bytes(batch_capacity_) ||
          batch.count > tqrj_maximum_queries(batch_capacity_) ||
          publication_keys_a_.size() < batch.count ||
          publication_rows_a_.size() < batch.count) {
        throw std::length_error(
            "insufficient idle publication workspace for GPULSMOpt TQRJ");
      }
    }
    if (batch.count > gpulsmopt2_detail::kMaximumOperationTile &&
        !use_tqrj) {
      for (std::size_t begin = 0u; begin < batch.count;
           begin += gpulsmopt2_detail::kMaximumOperationTile) {
        const std::size_t count = std::min(
            batch.count - begin,
            gpulsmopt2_detail::kMaximumOperationTile);
        lookup_locked(DeviceLookupBatch{
            batch.queries + begin, count, batch.out_values + begin,
            batch.out_found ? batch.out_found + begin : nullptr}, stream,
            quotients_grouped);
      }
      return;
    }
    begin_operation(stream);
    const std::uint32_t count = static_cast<std::uint32_t>(batch.count);
    if (use_tqrj) {
      const std::uint32_t active_capacity = std::min(
          count, gpulsmopt2_detail::kQuotients);
      const std::uint32_t maximum_query_tiles =
          gpulsmopt2_detail::tqrj_hash_tile_count(count) + active_capacity;
      const std::uint32_t maximum_pending_tiles =
          gpulsmopt2_detail::tqrj_hash_tile_count(pending_records_) +
          active_capacity;
      if (maximum_query_tiles > tqrj_maximum_tiles(batch_capacity_) ||
          maximum_pending_tiles > tqrj_maximum_tiles(batch_capacity_))
        throw std::length_error(
            "insufficient recycled tile workspace for GPULSMOpt TQRJ");
      auto workspace = borrow_tqrj_lookup_workspace(maximum_query_tiles);

      CUDA_CHECK(cudaMemsetAsync(
          workspace.active_quotient_count, 0, sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(
          workspace.hash_task_count, 0, sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(
          workspace.hash_counters, 0,
          gpulsmopt2_detail::kTqrjHashCounterBytes, stream));
      gpulsmopt2_detail::count_lookup_quotients_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
              batch.queries, count, admission_counts_.data(),
              workspace.reservation_ranks, workspace.active_quotients,
              workspace.active_quotient_count);
      gpulsmopt2_detail::materialize_lookup_active_counts_kernel<<<
          blocks(active_capacity + 1u), gpulsmopt2_detail::kThreads,
          0, stream>>>(
              workspace.active_quotients, workspace.active_quotient_count,
              admission_counts_.data(), active_capacity,
              workspace.active_query_counts);
      std::size_t scan_bytes = admission_temp_.size();
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          admission_temp_.data(), scan_bytes, workspace.active_query_counts,
          workspace.active_query_offsets, active_capacity + 1u, stream));
      gpulsmopt2_detail::publish_lookup_active_bases_kernel<<<
          blocks(active_capacity), gpulsmopt2_detail::kThreads,
          0, stream>>>(
              workspace.active_quotients, workspace.active_quotient_count,
              workspace.active_query_offsets, active_capacity,
              workspace.query_bases);
      gpulsmopt2_detail::scatter_query_records_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
              batch.queries, count, workspace.query_bases,
              workspace.reservation_ranks, workspace.grouped_queries,
              workspace.query_ids);

      gpulsmopt2_detail::tqrj_direct_lookup_kernel<<<
          active_capacity, gpulsmopt2_detail::kThreads, 0, stream>>>(
              workspace.grouped_queries, batch.out_values, batch.out_found,
              workspace.active_quotients, workspace.active_query_counts,
              workspace.active_quotient_count, workspace.query_bases,
              raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
              static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
              resident_rows(), descriptors_.data(), workspace.query_ids,
              canonical_cell_ranks_.data(),
              query_occupied_level_mask_.data(), workspace.hash_tasks,
              workspace.hash_task_count);

      const std::uint32_t *hash_query_counts = admission_counts_.data();
      const std::uint32_t *hash_raw_keys = raw_keys_.data();
      const gpulsmopt2_detail::RawPayload *hash_raw_payloads =
          raw_payloads_.data();
      const std::uint32_t *hash_raw_offsets = raw_offsets_.data();
      const std::uint32_t hash_batch_stride =
          static_cast<std::uint32_t>(batch_capacity_);
      const std::uint32_t hash_pending_batches = pending_batches_;
      std::uint32_t *hash_out_values = batch.out_values;
      std::uint8_t *hash_out_found = batch.out_found;
      gpulsmopt2_detail::ResidentRows hash_arena = resident_rows();
      const gpulsmopt2_detail::Descriptor *hash_descriptors =
          descriptors_.data();
      const std::uint32_t *hash_query_ids = workspace.query_ids;
      const std::uint16_t *hash_cell_ranks =
          canonical_cell_ranks_.data();
      const std::uint64_t *hash_query_manifest =
          query_occupied_level_mask_.data();
      const std::uint32_t hash_capacity =
          workspace.hash_table_capacity;
      void *hash_arguments[] = {
          &workspace.hash_tasks, &workspace.hash_task_count,
          &workspace.query_bases, &hash_query_counts,
          &workspace.grouped_queries, &workspace.hash_query_tiles,
          &workspace.hash_pending_tiles, &workspace.hash_counters,
          &workspace.hash_table,
          const_cast<std::uint32_t *>(&hash_capacity),
          &workspace.hash_winners, &hash_raw_keys, &hash_raw_payloads,
          &hash_raw_offsets, const_cast<std::uint32_t *>(&hash_batch_stride),
          const_cast<std::uint32_t *>(&hash_pending_batches),
          &hash_out_values, &hash_out_found, &hash_arena,
          &hash_descriptors, &hash_query_ids, &hash_cell_ranks,
          &hash_query_manifest};
      CUDA_CHECK(cudaLaunchCooperativeKernel(
          reinterpret_cast<const void *>(
              gpulsmopt2_detail::tqrj_hash_lookup_kernel),
          dim3(tqrj_hash_worker_blocks_),
          dim3(gpulsmopt2_detail::kTqrjHashThreads), hash_arguments, 0u,
          stream));
      gpulsmopt2_detail::reset_lookup_quotient_counts_kernel<<<
          blocks(active_capacity), gpulsmopt2_detail::kThreads,
          0, stream>>>(
              workspace.active_quotients, workspace.active_quotient_count,
              active_capacity, admission_counts_.data());
      CUDA_CHECK(cudaGetLastError());
      end_operation(stream);
      return;
    }

    const bool grouped =
        count >= gpulsmopt2_detail::kQuotients * 4u;
    const std::uint32_t *queries = batch.queries;
    const std::uint32_t *query_ids = nullptr;
    if (grouped && !quotients_grouped) {
      ensure_radix_workspace(count);
      gpulsmopt2_detail::count_admission_quotients_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          batch.queries, count, admission_counts_.data(),
          radix_input_ids());
      std::size_t scan_bytes = admission_temp_.size();
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          admission_temp_.data(), scan_bytes, admission_counts_.data(),
          query_quotient_offsets(),
          gpulsmopt2_detail::kQuotients + 1u, stream));
      gpulsmopt2_detail::scatter_query_records_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          batch.queries, count, query_quotient_offsets(),
          radix_input_ids(), radix_keys_.data(), radix_ids_out_.data());
      CUDA_CHECK(cudaMemsetAsync(
          admission_counts_.data(), 0,
          admission_counts_.size() * sizeof(std::uint32_t), stream));
      queries = radix_keys_.data();
      query_ids = radix_ids_out_.data();
    }
    gpulsmopt2_detail::canonical_lookup_with_pending_kernel<<<
        blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
            queries, batch.out_values, batch.out_found, count,
            raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
            static_cast<std::uint32_t>(batch_capacity_),
            pending_batches_, raw_signatures_.data(),
            raw_epoch_signatures_.data(), resident_rows(),
            descriptors_.data(), canonical_cell_ranks_.data(),
            query_ids, query_occupied_level_mask_.data());
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

  void lookup_sparse_locked(
      gpulsm_sparse::RecordBatchView queries,
      gpulsm_sparse::SparseLookupOutput output, cudaStream_t stream,
      bool quotients_grouped) {
    if (!queries.count) return;
    const std::uint64_t tqrj_dense_threshold =
        std::uint64_t{pending_batches_} * gpulsmopt2_detail::kQuotients *
        gpulsmopt2_detail::kTqrjDenseRowsPerSection;
    const bool use_tqrj =
        queries.count >= gpulsmopt2_detail::kQuotients * 4u &&
        queries.count <=
            batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch &&
        pending_batches_ >= gpulsmopt2_detail::kTqrjMinimumBatches &&
        std::uint64_t{pending_records_} >= tqrj_dense_threshold;
    if (queries.count > std::numeric_limits<std::uint32_t>::max() ||
        (queries.count > gpulsmopt2_detail::kMaximumOperationTile &&
         !use_tqrj)) {
      for (std::uint64_t begin = 0u; begin < queries.count;
           begin += gpulsmopt2_detail::kMaximumOperationTile) {
        const std::uint64_t count = std::min<std::uint64_t>(
            queries.count - begin,
            gpulsmopt2_detail::kMaximumOperationTile);
        lookup_sparse_locked(
            slice_record_batch(queries, begin, count),
            slice_lookup_output(output, begin), stream,
            quotients_grouped);
      }
      return;
    }
    const std::uint32_t count = static_cast<std::uint32_t>(queries.count);
    const bool materialize_heads =
        !queries.head4_words || queries.keys.offsets ||
        queries.keys.stride != sizeof(std::uint32_t);
    if (!sparse_read_workspace_)
      sparse_read_workspace_ =
          std::make_unique<gpulsm_sparse::SparseReadWorkspace>();
    if (use_tqrj) {
      if (publication_keys_a_.size() < count ||
          publication_rows_a_.size() < count) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        publication_keys_a_.grow(count);
        publication_rows_a_.grow(count);
      }
      if (operation_workspace_.size() <
              tqrj_hash_workspace_bytes(batch_capacity_) ||
          count > tqrj_maximum_queries(batch_capacity_))
        throw std::length_error(
            "insufficient idle publication workspace for sparse TQRJ");
    }
    sparse_read_workspace_->ensure(
        count, materialize_heads, use_tqrj);

    begin_operation(stream);
    sparse_read_workspace_->classify(
        queries, device_pending_sparse_states_.data(), pending_batches_,
        pending_sparse_generation_, device_sparse_manifests_.data(),
        active_device_manifest_.data(), materialize_heads, stream);
    note_sparse_workspace_high_water();
    const std::uint32_t *heads = materialize_heads
        ? sparse_read_workspace_->heads()
        : reinterpret_cast<const std::uint32_t *>(queries.keys.bytes);
    const gpulsm_sparse::PendingReadView pending{
        raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
        device_pending_sparse_states_.data(),
        static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
        pending_sparse_generation_};

    if (use_tqrj) {
      const std::uint32_t active_capacity = std::min(
          count, gpulsmopt2_detail::kQuotients);
      const std::uint32_t maximum_query_tiles =
          gpulsmopt2_detail::tqrj_hash_tile_count(count) + active_capacity;
      const std::uint32_t maximum_pending_tiles =
          gpulsmopt2_detail::tqrj_hash_tile_count(pending_records_) +
          active_capacity;
      if (maximum_query_tiles > tqrj_maximum_tiles(batch_capacity_) ||
          maximum_pending_tiles > tqrj_maximum_tiles(batch_capacity_))
        throw std::length_error(
            "insufficient recycled tile workspace for sparse TQRJ");
      auto workspace = borrow_tqrj_lookup_workspace(maximum_query_tiles);
      CUDA_CHECK(cudaMemsetAsync(
          workspace.active_quotient_count, 0, sizeof(std::uint32_t),
          stream));
      CUDA_CHECK(cudaMemsetAsync(
          workspace.hash_task_count, 0, sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(
          workspace.hash_counters, 0,
          gpulsmopt2_detail::kTqrjHashCounterBytes, stream));
      gpulsmopt2_detail::count_lookup_quotients_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          heads, count, admission_counts_.data(),
          workspace.reservation_ranks, workspace.active_quotients,
          workspace.active_quotient_count);
      gpulsmopt2_detail::materialize_lookup_active_counts_kernel<<<
          blocks(active_capacity + 1u), gpulsmopt2_detail::kThreads,
          0, stream>>>(
          workspace.active_quotients, workspace.active_quotient_count,
          admission_counts_.data(), active_capacity,
          workspace.active_query_counts);
      std::size_t scan_bytes = admission_temp_.size();
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          admission_temp_.data(), scan_bytes,
          workspace.active_query_counts, workspace.active_query_offsets,
          active_capacity + 1u, stream));
      gpulsmopt2_detail::publish_lookup_active_bases_kernel<<<
          blocks(active_capacity), gpulsmopt2_detail::kThreads,
          0, stream>>>(
          workspace.active_quotients, workspace.active_quotient_count,
          workspace.active_query_offsets, active_capacity,
          workspace.query_bases);
      gpulsmopt2_detail::scatter_query_records_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          heads, count, workspace.query_bases, workspace.reservation_ranks,
          workspace.grouped_queries, workspace.query_ids);

      gpulsm_sparse::sparse_tqrj_direct_lookup_kernel<<<
          active_capacity, gpulsmopt2_detail::kThreads, 0, stream>>>(
          queries, output, workspace.grouped_queries,
          workspace.active_quotients, workspace.active_query_counts,
          workspace.active_quotient_count, workspace.query_bases,
          workspace.query_ids, sparse_read_workspace_->exact_flags(),
          pending, resident_rows(), descriptors_.data(),
          canonical_cell_ranks_.data(),
          query_occupied_level_mask_.data(),
          device_sparse_manifests_.data(), active_device_manifest_.data(),
          workspace.hash_tasks, workspace.hash_task_count);

      const std::uint32_t *hash_query_counts = admission_counts_.data();
      const std::uint32_t *hash_grouped_heads = workspace.grouped_queries;
      const std::uint32_t *hash_query_ids = workspace.query_ids;
      gpulsm_sparse::RecordBatchView hash_queries = queries;
      gpulsm_sparse::SparseLookupOutput hash_output = output;
      const std::uint8_t *hash_exact_flags =
          sparse_read_workspace_->exact_flags();
      std::uint32_t *hash_query_owners = sparse_read_workspace_->owners();
      const std::uint32_t hash_capacity = workspace.hash_table_capacity;
      gpulsm_sparse::PendingReadView hash_pending = pending;
      gpulsmopt2_detail::ResidentRows hash_arena = resident_rows();
      const gpulsmopt2_detail::Descriptor *hash_descriptors =
          descriptors_.data();
      const std::uint16_t *hash_cell_ranks =
          canonical_cell_ranks_.data();
      const std::uint64_t *hash_occupied_mask =
          query_occupied_level_mask_.data();
      const gpulsm_sparse::DeviceSparseManifest *hash_sparse_manifests =
          device_sparse_manifests_.data();
      const std::uint32_t *hash_active_manifest =
          active_device_manifest_.data();
      void *hash_arguments[] = {
          &workspace.hash_tasks, &workspace.hash_task_count,
          &workspace.query_bases, &hash_query_counts,
          &hash_grouped_heads, &hash_query_ids, &hash_queries,
          &hash_output, &hash_exact_flags, &hash_query_owners,
          &workspace.hash_query_tiles, &workspace.hash_pending_tiles,
          &workspace.hash_counters, &workspace.hash_table,
          const_cast<std::uint32_t *>(&hash_capacity),
          &workspace.hash_winners, &hash_pending, &hash_arena,
          &hash_descriptors, &hash_cell_ranks, &hash_occupied_mask,
          &hash_sparse_manifests, &hash_active_manifest};
      CUDA_CHECK(cudaLaunchCooperativeKernel(
          reinterpret_cast<const void *>(
              gpulsm_sparse::sparse_tqrj_hash_lookup_kernel),
          dim3(std::min(tqrj_hash_worker_blocks_,
                        sparse_read_workspace_->hash_worker_blocks())),
          dim3(gpulsmopt2_detail::kTqrjHashThreads), hash_arguments, 0u,
          stream));
      gpulsmopt2_detail::reset_lookup_quotient_counts_kernel<<<
          blocks(active_capacity), gpulsmopt2_detail::kThreads,
          0, stream>>>(
          workspace.active_quotients, workspace.active_quotient_count,
          active_capacity, admission_counts_.data());
      CUDA_CHECK(cudaGetLastError());
      end_operation(stream);
      return;
    }

    const bool grouped =
        count >= gpulsmopt2_detail::kQuotients * 4u;
    const std::uint32_t *ordered_heads = heads;
    const std::uint32_t *query_ids = nullptr;
    if (grouped && !quotients_grouped) {
      ensure_radix_workspace(count);
      gpulsmopt2_detail::count_admission_quotients_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          heads, count, admission_counts_.data(), radix_input_ids());
      std::size_t scan_bytes = admission_temp_.size();
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          admission_temp_.data(), scan_bytes, admission_counts_.data(),
          query_quotient_offsets(),
          gpulsmopt2_detail::kQuotients + 1u, stream));
      gpulsmopt2_detail::scatter_query_records_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          heads, count, query_quotient_offsets(), radix_input_ids(),
          radix_keys_.data(), radix_ids_out_.data());
      CUDA_CHECK(cudaMemsetAsync(
          admission_counts_.data(), 0,
          admission_counts_.size() * sizeof(std::uint32_t), stream));
      ordered_heads = radix_keys_.data();
      query_ids = radix_ids_out_.data();
    }

    gpulsm_sparse::sparse_canonical_ordinary_lookup_kernel<<<
        blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
        queries, ordered_heads, query_ids,
        sparse_read_workspace_->exact_flags(), count, output, pending,
        raw_signatures_.data(), raw_epoch_signatures_.data(),
        resident_rows(), descriptors_.data(), canonical_cell_ranks_.data(),
        query_occupied_level_mask_.data(), device_sparse_manifests_.data(),
        active_device_manifest_.data());
    gpulsm_sparse::sparse_canonical_exact_lookup_kernel<<<
        blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
        queries, sparse_read_workspace_->exact_ids(),
        sparse_read_workspace_->exact_count(), output, pending,
        resident_rows(), descriptors_.data(), canonical_cell_ranks_.data(),
        query_occupied_level_mask_.data(), device_sparse_manifests_.data(),
        active_device_manifest_.data());
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

  void range_sparse_locked(
      gpulsm_sparse::range_gate::RangeQueryBatch queries,
      std::uint32_t *sums, std::uint8_t *valid, cudaStream_t stream) {
    if (!queries.count) return;
    if (queries.count > gpulsmopt2_detail::kMaximumOperationTile) {
      for (std::uint32_t begin = 0u; begin < queries.count;
           begin += gpulsmopt2_detail::kMaximumOperationTile) {
        const std::uint32_t count = static_cast<std::uint32_t>(
            std::min<std::size_t>(
                queries.count - begin,
                gpulsmopt2_detail::kMaximumOperationTile));
        range_sparse_locked(
            {slice_record_batch(queries.lower, begin, count),
             slice_record_batch(queries.upper, begin, count), count},
            sums + begin, valid ? valid + begin : nullptr, stream);
      }
      return;
    }

    begin_operation(stream);
    try {
      gpulsm_sparse::rank_range::PreparedReadState state{};
      std::uint64_t levels = host_occupied_level_mask_;
      while (levels) {
        const std::uint32_t level = static_cast<std::uint32_t>(
            __builtin_ctzll(levels));
        levels &= levels - 1u;
        state.levels.push_back(level);
        gpulsm_sparse::RootBuildState root{};
        if (sparse_roots_[level]) {
          root = sparse_roots_[level]->state;
        } else {
          root.level = level;
          root.row_count = level_counts_[level];
          root.logical_count = level_counts_[level];
        }
        state.roots[level] = root;
        state.candidate_rows += root.logical_count;
        state.capsule_count += root.capsule_count;
        state.capsule_bytes += root.capsule_live_bytes;
      }
      for (std::uint32_t slot = 0u; slot < pending_batches_; ++slot) {
        const std::uint32_t count = raw_batch_counts_[slot];
        state.batch_counts[slot] = count;
        state.pending_rows += count;
        const auto &overlay = sparse_pending_slots_[slot];
        if (overlay &&
            overlay->state.generation == pending_sparse_generation_) {
          state.capsule_count += overlay->state.capsule_count;
          state.capsule_bytes += overlay->state.live_bytes;
        }
      }
      state.candidate_rows += state.pending_rows;
      state.pending = {
          raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
          device_pending_sparse_states_.data(),
          static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
          pending_sparse_generation_};
      if (!state.candidate_rows) {
        CUDA_CHECK(cudaMemsetAsync(
            sums, 0, std::size_t{queries.count} * sizeof(*sums), stream));
        if (valid)
          CUDA_CHECK(cudaMemsetAsync(valid, 1, queries.count, stream));
        end_operation(stream);
        return;
      }
      if (state.candidate_rows > std::numeric_limits<std::uint32_t>::max() ||
          state.capsule_count > std::numeric_limits<std::size_t>::max() ||
          state.capsule_bytes > std::numeric_limits<std::size_t>::max())
        throw std::length_error("range read state exceeds host capacity");

      gpulsm_sparse::Buffer<gpulsm_sparse::RootBuildState> roots(
          gpulsm_sparse::kLevels);
      gpulsm_sparse::check(cudaMemcpyAsync(
          roots.data(), state.roots.data(), roots.bytes(),
          cudaMemcpyHostToDevice, stream), "copy rank sparse roots");
      gpulsm_sparse::range_gate::ActiveSectionRoster active;
      active.rebuild(
          descriptors_.data(), roots.data(), state.levels,
          state.pending, stream);
      gpulsm_sparse::rank_range::RosterStorage roster(
          static_cast<std::uint32_t>(state.levels.size()), active.count(),
          gpulsm_sparse::rank_range::roster_capacity(
              state, active.count()));
      if (!state.levels.empty())
        roster.build(
            resident_rows(), descriptors_.data(), active, roots.data(),
            stream);
      const std::uint32_t source_count =
          static_cast<std::uint32_t>(state.levels.size()) +
          (state.pending_rows ? 1u : 0u);
      if (!source_count)
        throw std::logic_error("nonempty range has no rank source");
      const std::uint64_t leaves =
          (state.candidate_rows +
           gpulsm_sparse::rank_range::kTileRows - 1u) /
          gpulsm_sparse::rank_range::kTileRows;
      const std::uint64_t task_capacity64 =
          2u * leaves + queries.count + 64u;
      if (task_capacity64 > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("range task count exceeds 32 bits");
      const std::uint32_t task_capacity =
          static_cast<std::uint32_t>(task_capacity64);
      gpulsm_sparse::rank_range::Workspace workspace(
          std::max(1u, state.pending_rows), queries.count,
          task_capacity, source_count);
      gpulsm_sparse::rank_range::OutputStorage output(
          state.candidate_rows, state.capsule_count, state.capsule_bytes,
          task_capacity, queries.count);
      note_sparse_workspace_high_water(
          roots.bytes() + active.bytes() + roster.bytes() +
          workspace.bytes() + output.bytes());
      const auto result = workspace.enumerate(
          roster, queries, output, state.pending,
          state.batch_counts.data(), resident_rows(), roots.data(), stream);
      if (result.errors || result.receipt.status)
        throw std::runtime_error("rank-roster range enumeration failed");
      gpulsm_sparse::rank_range::reduce_range_slices<<<
          queries.count, gpulsm_sparse::kThreads, 0, stream>>>(
          output.view(result.tasks), output.slices_.data(), queries.count,
          sums);
      if (valid)
        CUDA_CHECK(cudaMemsetAsync(valid, 1, queries.count, stream));
      CUDA_CHECK(cudaGetLastError());
      end_operation(stream);
    } catch (...) {
      end_operation(stream);
      throw;
    }
  }

  void range_locked(const DeviceRangeOutputBatch &batch,
                    cudaStream_t stream) {
    if (!batch.query_count) return;
    if (!batch.lo || !batch.hi || !batch.out_sums)
      throw std::invalid_argument("invalid GPULSMOpt range input");
    if (batch.query_count > gpulsmopt2_detail::kMaximumOperationTile) {
      for (std::size_t begin = 0u; begin < batch.query_count;
           begin += gpulsmopt2_detail::kMaximumOperationTile) {
        const std::size_t count = std::min(
            batch.query_count - begin,
            gpulsmopt2_detail::kMaximumOperationTile);
        range_locked(DeviceRangeOutputBatch{
            batch.lo + begin, batch.hi + begin, count,
            batch.out_sums + begin}, stream);
      }
      return;
    }
    begin_operation(stream);
    const std::uint32_t query_count =
        static_cast<std::uint32_t>(batch.query_count);
    const bool may_have_crowded_newer = pending_batches_ != 0u ||
        (host_occupied_level_mask_ & (host_occupied_level_mask_ - 1u)) != 0u;
    if (may_have_crowded_newer) {
      std::size_t hot_scan_bytes{};
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          nullptr, hot_scan_bytes, range_hot_counts_.data(),
          range_hot_offsets_.data(),
          gpulsmopt2_detail::kQuotients + 1u, stream));
      if (range_hot_temp_.size() < hot_scan_bytes)
        range_hot_temp_.resize(hot_scan_bytes);
      CUDA_CHECK(cudaMemsetAsync(
          range_hot_descriptors_.data(), 0,
          range_hot_descriptors_.size() *
              sizeof(gpulsmopt2_detail::Descriptor), stream));
      gpulsmopt2_detail::count_range_hot_newer_rows_kernel<<<
          blocks(gpulsmopt2_detail::kQuotients + 1u),
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              raw_offsets_.data(), pending_batches_, descriptors_.data(),
              query_occupied_level_mask_.data(), range_hot_counts_.data());
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          range_hot_temp_.data(), hot_scan_bytes, range_hot_counts_.data(),
          range_hot_offsets_.data(),
          gpulsmopt2_detail::kQuotients + 1u, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          range_hot_total_receipt_.data(),
          range_hot_offsets_.data() + gpulsmopt2_detail::kQuotients,
          sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream));
    }
    const bool needs_wide_total = query_count >
        std::numeric_limits<std::uint32_t>::max() /
            gpulsmopt2_detail::kQuotients;
    std::size_t scan_bytes{};
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, range_fragment_counts_.data(),
        range_fragment_offsets_.data(), query_count + 1u, stream));
    std::size_t reduce_bytes{};
    if (needs_wide_total) {
      using WideCountIterator = cub::TransformInputIterator<
          std::uint64_t, gpulsmopt2_detail::WidenFragmentCount,
          const std::uint32_t *>;
      const WideCountIterator counts(
          range_fragment_counts_.data(),
          gpulsmopt2_detail::WidenFragmentCount{});
      CUDA_CHECK(cub::DeviceReduce::Sum(
          nullptr, reduce_bytes, counts, range_fragment_total_.data(),
          query_count, stream));
    }
    ensure_range_fragment_query_capacity(
        query_count, std::max(scan_bytes, reduce_bytes));
    gpulsmopt2_detail::count_range_fragments_kernel<<<
        blocks(std::size_t{query_count} + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            batch.lo, batch.hi, query_count,
            range_fragment_counts_.data());
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        range_query_temp_, scan_bytes, range_fragment_counts_.data(),
        range_fragment_offsets_.data(), query_count + 1u, stream));
    std::uint32_t fragment_count{};
    if (query_count == 1u) {
      fragment_count = gpulsmopt2_detail::kQuotients;
      if (may_have_crowded_newer) CUDA_CHECK(cudaStreamSynchronize(stream));
    } else if (needs_wide_total) {
      using WideCountIterator = cub::TransformInputIterator<
          std::uint64_t, gpulsmopt2_detail::WidenFragmentCount,
          const std::uint32_t *>;
      const WideCountIterator counts(
          range_fragment_counts_.data(),
          gpulsmopt2_detail::WidenFragmentCount{});
      CUDA_CHECK(cub::DeviceReduce::Sum(
          range_query_temp_, reduce_bytes, counts,
          range_fragment_total_.data(), query_count, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          range_total_receipt_.data(), range_fragment_total_.data(),
          sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const std::uint64_t total = range_total_receipt_.data()[0];
      if (total > std::numeric_limits<std::uint32_t>::max()) {
        // Reject fragment totals above 32 bits.
        end_operation(stream);
        throw std::length_error(
            "GPULSMOpt range produces more than 2^32-1 fragments");
      }
      fragment_count = static_cast<std::uint32_t>(total);
    } else {
      CUDA_CHECK(cudaMemcpyAsync(
          &fragment_count, range_fragment_offsets_.data() + query_count,
          sizeof(fragment_count), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const std::uint64_t hot_total = may_have_crowded_newer
        ? range_hot_total_receipt_.data()[0] : 0u;
    if (hot_total) materialize_range_hot_sections(hot_total, stream);
    const bool hot_ready = hot_total != 0u;
    if (!fragment_count) {
      CUDA_CHECK(cudaMemsetAsync(batch.out_sums, 0,
                                 std::size_t{query_count} *
                                     sizeof(std::uint32_t),
                                 stream));
      end_operation(stream);
      return;
    }
    const bool use_section_owners = query_count > 1u &&
        std::uint64_t{fragment_count} >=
            std::uint64_t{gpulsmopt2_detail::kQuotients} *
                gpulsmopt2_detail::kSectionOwnerMinimumReuse;
    std::size_t section_sort_bytes{}, task_scan_bytes{};
    if (use_section_owners) {
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, section_sort_bytes, range_section_keys_in_.data(),
          range_section_keys_out_.data(),
          range_section_fragments_in_.data(),
          range_section_fragments_out_.data(), fragment_count, 0,
          32, stream));
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          nullptr, task_scan_bytes, range_section_task_counts_.data(),
          range_section_task_offsets_.data(),
          gpulsmopt2_detail::kQuotients + 1u, stream));
      ensure_range_section_capacity(
          fragment_count, std::max(section_sort_bytes, task_scan_bytes));
    } else {
      ensure_range_fragment_capacity(fragment_count);
    }
    gpulsmopt2_detail::adaptive_emit_range_fragments_kernel<<<
        gpulsmopt2_detail::kRangeSchedulerBlocks,
        gpulsmopt2_detail::kThreads, 0,
        stream>>>(
            batch.lo, batch.hi, range_fragment_offsets_.data(), query_count,
            use_section_owners ? nullptr : range_fragments_.data(),
            use_section_owners ? range_section_keys_in_.data() : nullptr,
            use_section_owners ? range_section_fragments_in_.data() : nullptr,
            use_section_owners);
    if (use_section_owners) {
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          range_section_temp_, section_sort_bytes,
          range_section_keys_in_.data(), range_section_keys_out_.data(),
          range_section_fragments_in_.data(),
          range_section_fragments_out_.data(), fragment_count, 0,
          32, stream));
      gpulsmopt2_detail::find_section_fragment_offsets_kernel<<<
          blocks(gpulsmopt2_detail::kQuotients + 1u),
          gpulsmopt2_detail::kThreads, 0, stream>>>(
          range_section_keys_out_.data(), fragment_count,
              range_section_offsets_.data());
      gpulsmopt2_detail::count_section_range_tasks_kernel<<<
          blocks(gpulsmopt2_detail::kQuotients + 1u),
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              range_section_offsets_.data(),
              range_section_task_counts_.data());
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          range_section_temp_, task_scan_bytes,
          range_section_task_counts_.data(),
          range_section_task_offsets_.data(),
          gpulsmopt2_detail::kQuotients + 1u, stream));
      gpulsmopt2_detail::emit_section_range_tasks_kernel<<<
          blocks(gpulsmopt2_detail::kQuotients),
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              range_section_offsets_.data(),
              range_section_task_offsets_.data(),
              range_section_tasks_.data());
      launch_section_ranges(stream, hot_ready);
    } else {
      launch_fragment_ranges(
          fragment_count, query_count, batch, stream, hot_ready);
    }
    CUDA_CHECK(cudaMemsetAsync(range_reduction_completion_.data(), 0,
                               sizeof(std::uint32_t), stream));
    gpulsmopt2_detail::adaptive_reduce_range_partials_kernel<<<
        gpulsmopt2_detail::kRangeSchedulerBlocks,
        gpulsmopt2_detail::kThreads, 0,
        stream>>>(
            range_fragment_offsets_.data(), query_count,
            range_fragment_partials_.data(), batch.out_sums,
            range_partials_.data(), range_reduction_completion_.data());
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

  void successor_locked(const DeviceSuccessorBatch &batch,
                        cudaStream_t stream) {
    if (!batch.count) return;
    if (!batch.queries || !batch.out_keys)
      throw std::invalid_argument("invalid GPULSMOpt successor input");
    if (batch.count > gpulsmopt2_detail::kMaximumOperationTile) {
      for (std::size_t begin = 0u; begin < batch.count;
           begin += gpulsmopt2_detail::kMaximumOperationTile) {
        const std::size_t count = std::min(
            batch.count - begin,
            gpulsmopt2_detail::kMaximumOperationTile);
        successor_locked(DeviceSuccessorBatch{
            batch.queries + begin, count, batch.out_keys + begin}, stream);
      }
      return;
    }
    begin_operation(stream);
    gpulsmopt2_detail::successor_with_pending_kernel<<<
        blocks(batch.count), gpulsmopt2_detail::kThreads, 0, stream>>>(
        batch.queries, static_cast<std::uint32_t>(batch.count), batch.out_keys,
        raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
        static_cast<std::uint32_t>(batch_capacity_),
        pending_batches_, resident_rows(), descriptors_.data(),
        route_headers_.data(), route_slices_.data(),
        query_occupied_level_mask_.data());
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

  void successor_sparse_locked(
      gpulsm_sparse::RecordBatchView queries,
      gpulsm_sparse::SparseSuccessorOutput output,
      cudaStream_t stream) {
    if (!queries.count) return;
    if (queries.count > gpulsmopt2_detail::kMaximumOperationTile) {
      for (std::uint64_t begin = 0u; begin < queries.count;
           begin += gpulsmopt2_detail::kMaximumOperationTile) {
        const std::uint64_t count = std::min<std::uint64_t>(
            queries.count - begin,
            gpulsmopt2_detail::kMaximumOperationTile);
        successor_sparse_locked(
            slice_record_batch(queries, begin, count),
            slice_successor_output(output, begin), stream);
      }
      return;
    }
    begin_operation(stream);
    const gpulsm_sparse::PendingReadView pending{
        raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
        device_pending_sparse_states_.data(),
        static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
        pending_sparse_generation_};
    gpulsm_sparse::sparse_successor_kernel<<<
        blocks(queries.count), gpulsmopt2_detail::kThreads, 0, stream>>>(
        queries, output, pending, resident_rows(), descriptors_.data(),
        route_headers_.data(), route_slices_.data(),
        canonical_cell_ranks_.data(), query_occupied_level_mask_.data(),
        device_sparse_manifests_.data(), active_device_manifest_.data());
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

public:

  SparseMemoryAccounting sparse_memory_accounting() const {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    const_cast<GPULSMOpt *>(this)->resolve_publication_receipt();
    return sparse_memory_accounting_unlocked();
  }

  std::size_t gpu_resident_bytes() const {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    const_cast<GPULSMOpt *>(this)->resolve_publication_receipt();
    const SparseMemoryAccounting sparse =
        sparse_memory_accounting_unlocked();
    const std::size_t rollover_rank_bytes = canonical_rollover_epoch_ranks_
        ? canonical_rollover_epoch_ranks_->size() * sizeof(std::uint16_t)
        : 0u;
    const std::uint64_t restored = rollover_rank_bytes +
        arena_key_flags_.size() * sizeof(std::uint32_t) +
        arena_values_.size() * sizeof(std::uint32_t) +
        descriptors_.size() * sizeof(gpulsmopt2_detail::Descriptor) +
        route_headers_.size() * sizeof(gpulsmopt2_detail::RouteHeader) +
        route_slices_.size() * sizeof(gpulsmopt2_detail::RouteSlice) +
        route_logical_begins_.size() * sizeof(std::uint32_t) +
        route_quotients_.size() * sizeof(std::uint16_t) +
        level_q_logical_offsets_.size() * sizeof(std::uint32_t) +
        device_manifests_.size() *
            sizeof(gpulsmopt2_detail::DeviceManifest) +
        active_device_manifest_.size() * sizeof(std::uint32_t) +
        query_occupied_level_mask_.size() * sizeof(std::uint64_t) +
        sealed_device_command_.bytes() + sealed_device_receipt_.bytes() +
        resident_plan_.size() *
            sizeof(gpulsmopt2_detail::ResidentPublicationPlan) +
        level_storage_spans_.size() *
            sizeof(gpulsmopt2_detail::LevelStorageSpan) +
        canonical_cell_ranks_.size() * sizeof(std::uint16_t) +
        operation_workspace_.size() * sizeof(std::uint8_t) +
        (sealed_workspace_ ? sealed_workspace_->bytes() : 0u) +
        canonical_job_prefixes_.size() *
            sizeof(gpulsmopt2_detail::CanonicalJobPrefix) +
        canonical_next_job_.size() * sizeof(std::uint32_t) +
        raw_keys_.size() * sizeof(std::uint32_t) +
        raw_payloads_.size() * sizeof(gpulsmopt2_detail::RawPayload) +
        raw_offsets_.size() * sizeof(std::uint32_t) +
        raw_signatures_.size() * sizeof(std::uint64_t) +
        raw_epoch_signatures_.size() * sizeof(std::uint64_t) +
        publication_rows_a_.size() * sizeof(gpulsmopt2_detail::Row) +
        (publication_keys_a_.size() +
         publication_selected_count_.size()) * sizeof(std::uint32_t) +
        (foundation_source_offsets_.size() +
         foundation_section_output_counts_.size() +
         local_epoch_overflow_flag_.size()) * sizeof(std::uint32_t) +
        balanced_merge_raw_counts_.size() * sizeof(std::uint64_t) +
        (resident_tile_job_counts_.size() +
         resident_tile_job_offsets_.size()) * sizeof(std::uint32_t) +
        resident_job_raw_reservations_.size() * sizeof(std::uint64_t) +
        balanced_merge_jobs_.size() *
            sizeof(gpulsmopt2_detail::BalancedMergeJob) +
        resident_scan_temp_.size() * sizeof(std::uint8_t) +
        admission_counts_.size() * sizeof(std::uint32_t) +
        admission_temp_.size() * sizeof(std::uint8_t) +
        range_partials_.size() * sizeof(unsigned long long) +
        range_reduction_completion_.size() * sizeof(std::uint32_t) +
        range_fragment_total_.size() * sizeof(std::uint64_t) +
        (range_hot_counts_.size() + range_hot_offsets_.size() +
         range_hot_tokens_a_.size() + range_hot_tokens_b_.size()) *
            sizeof(std::uint64_t) +
        (range_hot_window_offsets_.size() +
         range_hot_selected_count_.size()) * sizeof(std::uint32_t) +
        range_hot_descriptors_.size() *
            sizeof(gpulsmopt2_detail::Descriptor) +
        range_hot_temp_.size() * sizeof(std::uint8_t) +
        range_query_storage_.size() + range_fragment_storage_.size() +
        range_section_storage_.size();
    if (sparse.physical_bytes() >
        std::numeric_limits<std::size_t>::max() - restored)
      throw std::overflow_error("GPULSMOpt memory accounting overflow");
    return static_cast<std::size_t>(restored + sparse.physical_bytes());
  }

private:
  gpulsmopt2_detail::ResidentRows resident_rows() {
    return {arena_key_flags_.data(), arena_values_.data()};
  }

  std::uint64_t current_sparse_workspace_bytes() const {
    return (sparse_pending_workspace_
                ? sparse_pending_workspace_->bytes() : 0u) +
        (sparse_refinement_workspace_
             ? sparse_refinement_workspace_->bytes() : 0u) +
        (sparse_read_workspace_ ? sparse_read_workspace_->bytes() : 0u);
  }

  void note_sparse_workspace_high_water(
      std::uint64_t operation_temporary_bytes = 0u) {
    const std::uint64_t retained = current_sparse_workspace_bytes();
    if (operation_temporary_bytes >
        std::numeric_limits<std::uint64_t>::max() - retained)
      throw std::overflow_error("sparse workspace accounting overflow");
    sparse_workspace_high_water_bytes_ = std::max(
        sparse_workspace_high_water_bytes_,
        retained + operation_temporary_bytes);
  }

  SparseMemoryAccounting sparse_memory_accounting_unlocked() const {
    SparseMemoryAccounting result{};
    result.manifest_bytes = device_sparse_manifests_.bytes() +
        device_pending_sparse_states_.bytes();
    result.workspace_bytes = current_sparse_workspace_bytes();
    result.workspace_high_water_bytes = std::max(
        sparse_workspace_high_water_bytes_, result.workspace_bytes);

    std::vector<const gpulsm_sparse::StagedRootOverlay *> root_seen;
    std::vector<const gpulsm_sparse::StagedPendingSlotOverlay *>
        pending_seen;
    std::vector<const gpulsm_sparse::CapsuleSegment *> segment_seen;
    const auto account_segment =
        [&](const gpulsm_sparse::CapsuleSegmentOwnership &owned) {
          if (!owned.segment) return;
          const auto *segment = owned.segment.get();
          if (std::find(segment_seen.begin(), segment_seen.end(), segment) !=
              segment_seen.end())
            return;
          segment_seen.push_back(segment);
          ++result.capsule_segments;
          result.capsule_reserved_bytes += segment->reserved_bytes();
          result.capsule_mapped_bytes += segment->mapped_bytes();
          result.capsule_live_bytes += owned.live_bytes;
          result.capsule_garbage_bytes += owned.garbage_bytes;
        };
    const auto account_root =
        [&](const std::shared_ptr<gpulsm_sparse::StagedRootOverlay> &root) {
          if (!root ||
              std::find(root_seen.begin(), root_seen.end(), root.get()) !=
                  root_seen.end())
            return;
          root_seen.push_back(root.get());
          result.overlay_metadata_bytes += root->device_state.bytes() +
              root->exact_heads.bytes() + root->exact_rows.bytes() +
              root->special_heads.bytes() + root->pages.bytes() +
              root->indexes.bytes();
          for (const auto &owned : root->segments) account_segment(owned);
        };
    const auto account_pending =
        [&](const std::shared_ptr<
                gpulsm_sparse::StagedPendingSlotOverlay> &slot) {
          if (!slot ||
              std::find(pending_seen.begin(), pending_seen.end(),
                        slot.get()) != pending_seen.end())
            return;
          pending_seen.push_back(slot.get());
          result.overlay_metadata_bytes += slot->pages.bytes() +
              slot->indexes.bytes() + slot->exact_heads.bytes() +
              slot->exact_refs.bytes() + slot->special_heads.bytes() +
              slot->exceptions.bytes();
          for (const auto &owned : slot->segments) account_segment(owned);
        };
    for (const auto &root : sparse_roots_) account_root(root);
    account_root(pending_sparse_publication_root_);
    for (const auto &slot : sparse_pending_slots_) account_pending(slot);
    return result;
  }

  static int blocks(std::size_t count) {
    return static_cast<int>((count + gpulsmopt2_detail::kThreads - 1u) /
                            gpulsmopt2_detail::kThreads);
  }


  std::uint32_t initial_level_for_records(std::size_t count) const {
    for (std::uint32_t tier_begin = 0u;
         tier_begin < canonical_regular_level_count_; tier_begin += 3u) {
      const std::uint32_t slots = std::min(
          3u, canonical_regular_level_count_ - tier_begin);
      if (count <= level_storage_spans_host_[tier_begin].capacity)
        return tier_begin + slots - 1u;
    }
    if (canonical_regular_level_count_ < canonical_level_count_ &&
        count <= level_storage_spans_host_[canonical_regular_level_count_]
                     .capacity)
      return canonical_regular_level_count_;
    throw std::bad_alloc();
  }

  std::uint64_t level_begin(std::uint32_t target) const {
    if (target >= canonical_level_count_) throw std::out_of_range(
        "GPULSMOpt physical run slot is out of range");
    return level_storage_spans_host_[target].begin;
  }

  std::uint64_t level_capacity(std::uint32_t target) const {
    if (target >= canonical_level_count_) throw std::out_of_range(
        "GPULSMOpt physical run slot is out of range");
    return level_storage_spans_host_[target].capacity;
  }

  void ensure_level_storage_mapped(
      std::uint32_t level, cudaStream_t stream) {
    const std::uint64_t required = level_begin(level) + level_capacity(level);
    if (required <= arena_key_flags_.size() &&
        required <= arena_values_.size())
      return;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    arena_key_flags_.grow(required);
    arena_values_.grow(required);
  }

  void ensure_canonical_top_rollover_bank(cudaStream_t stream) {
    if (!canonical_level_count_)
      throw std::logic_error("GPULSMOpt has no canonical levels");
    const std::uint32_t top = canonical_level_count_ - 1u;
    const std::uint64_t required = level_begin(top) +
        2u * level_capacity(top);
    const bool rows_ready = required <= arena_key_flags_.size() &&
        required <= arena_values_.size();
    if (rows_ready && canonical_rollover_epoch_ranks_)
      return;

    // VMM mapping is paid only on the first rollover. Normal construction and
    // all carries that still have an unused level keep their previous cost.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!rows_ready) {
      arena_key_flags_.grow(required);
      arena_values_.grow(required);
    }
    if (!canonical_rollover_epoch_ranks_)
      canonical_rollover_epoch_ranks_ = std::make_unique<
          gpulsmopt2_detail::Buffer<std::uint16_t>>(
              gpulsmopt2_detail::kLocalRankEntries);
  }

  void refresh_active_levels() {
    active_levels_ = 0u;
    for (std::uint32_t level = 0u;
         level < gpulsmopt2_detail::kMaximumLevels; ++level)
      if (level_counts_[level]) active_levels_ = level + 1u;
  }

  std::uint32_t foundation_level() const {
    return active_levels_ ? active_levels_ - 1u
                          : gpulsmopt2_detail::kMaximumLevels;
  }

  void initialize_resident_workspace() {
    std::array<gpulsmopt2_detail::LevelStorageSpan,
               gpulsmopt2_detail::kMaximumLevels> spans{};
    std::uint64_t cursor = 0u;
    std::size_t capacity = level_zero_capacity_;
    for (std::uint32_t level = 0u;
         level < canonical_regular_level_count_; ++level) {
      spans[level] = {cursor, capacity};
      cursor += capacity;
      if ((level + 1u) % 3u == 0u)
        capacity = capacity > publication_capacity_ / 4u
            ? publication_capacity_ : capacity * 4u;
    }
    if (cursor != level_pool_capacity_)
      throw std::logic_error("GPULSMOpt preassigned level spans overflow");
    if (canonical_regular_level_count_ < canonical_level_count_)
      spans[canonical_regular_level_count_] = {
          cursor, publication_capacity_};
    level_storage_spans_host_ = spans;
    CUDA_CHECK(cudaMemcpy(
        level_storage_spans_.data(), spans.data(), sizeof(spans),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(route_logical_begins_.data(), 0,
                          route_logical_begins_.size() *
                              sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(route_quotients_.data(), 0,
                          route_quotients_.size() *
                              sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMemset(level_q_logical_offsets_.data(), 0,
                          level_q_logical_offsets_.size() *
                              sizeof(std::uint32_t)));

    std::size_t maximum_scan_bytes = 0u, bytes = 0u;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, bytes, resident_tile_job_counts_.data(),
        resident_tile_job_offsets_.data(),
        gpulsmopt2_detail::kPlanningTiles + 1u, 0));
    maximum_scan_bytes = std::max(maximum_scan_bytes, bytes);
    bytes = 0u;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, bytes, foundation_section_output_counts_.data(),
        foundation_source_offsets_.data(),
        gpulsmopt2_detail::kQuotients + 1u, 0));
    maximum_scan_bytes = std::max(maximum_scan_bytes, bytes);
    resident_scan_temp_.resize(maximum_scan_bytes);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (!properties.cooperativeLaunch)
      throw std::runtime_error(
          "GPULSMOpt TQRJ requires cooperative kernel launch support");
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, gpulsmopt2_detail::tqrj_hash_lookup_kernel,
        gpulsmopt2_detail::kTqrjHashThreads, 0u));
    // The single hash executor derives its worker grid only from kernel
    // resources and hardware parallelism, independently of query count and
    // quotient skew.
    tqrj_hash_worker_blocks_ = static_cast<std::uint32_t>(
        std::max(1, std::min(4, blocks_per_sm)) *
        properties.multiProcessorCount);
    blocks_per_sm = 0;
    resident_planner_blocks_ = static_cast<std::uint32_t>(std::max<std::size_t>(
        1u, std::min<std::size_t>(maximum_resident_jobs_,
            static_cast<std::size_t>(properties.multiProcessorCount) * 4u)));
    blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        gpulsmopt2_detail::cooperative_section_owned_range_kernel<
            gpulsmopt2_detail::SumRowsAggregate>,
        gpulsmopt2_detail::kSectionRangeThreads, 0u));
    range_section_blocks_ = static_cast<std::uint32_t>(
        std::max(1, blocks_per_sm) * properties.multiProcessorCount);
  }

  void initialize_canonical_workspace() {
    if (resident_merge_capacity_ >=
        gpulsmopt2_detail::kCanonicalCandidateLimit)
      throw std::logic_error(
          "GPULSMOpt canonical job capacity does not fit 12 bits");
    CUDA_CHECK(cudaFuncSetAttribute(
        gpulsmopt2_detail::canonical_fallback_carry_jobs_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(canonical_merge_workspace_bytes_)));
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        gpulsmopt2_detail::resolve_canonical_epoch_oversized_kernel<false>,
        gpulsmopt2_detail::kFoundationCompactionThreads, 0u));
    canonical_epoch_resolver_blocks_ = static_cast<std::uint32_t>(
        std::max(1, blocks_per_sm) * properties.multiProcessorCount);
    canonical_epoch_workspace_slots_ = static_cast<std::uint32_t>(
        canonical_epoch_workspace_.size() /
        gpulsmopt2_detail::kCanonicalResolverSuffixes);
    canonical_epoch_workspace_slots_ =
        std::max(1u, canonical_epoch_workspace_slots_);

    const std::uint32_t maximum_sources = std::min(
        gpulsmopt2_detail::kMaximumMergeSources,
        std::max(1u, canonical_level_count_ + 1u));
    cudaFuncAttributes tournament_attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(
        &tournament_attributes,
        gpulsmopt2_detail::canonical_tournament_carry_jobs_kernel));
    const std::size_t optin_shared_bytes = std::max<std::size_t>(
        properties.sharedMemPerBlock,
        properties.sharedMemPerBlockOptin);
    const std::size_t maximum_dynamic_shared_bytes =
        optin_shared_bytes > tournament_attributes.sharedSizeBytes
            ? optin_shared_bytes - tournament_attributes.sharedSizeBytes
            : 0u;
    canonical_job_capacities_.fill(resident_merge_capacity_);
    std::size_t tournament_attribute_bytes = 0u;
    for (std::uint32_t source_count =
             gpulsmopt2_detail::kCanonicalTournamentMinimumSources;
         source_count <= maximum_sources; ++source_count) {
      const auto active_blocks = [&](std::uint32_t capacity) {
        const std::size_t shared_bytes =
            gpulsmopt2_detail::canonical_tournament_workspace_bytes(
                capacity, source_count);
        if (shared_bytes > maximum_dynamic_shared_bytes) return 0;
        int blocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks,
            gpulsmopt2_detail::canonical_tournament_carry_jobs_kernel,
            gpulsmopt2_detail::kFoundationCompactionThreads,
            shared_bytes));
        return blocks;
      };
      const int baseline_blocks = active_blocks(resident_merge_capacity_);
      if (!baseline_blocks) break;
      // Grow the widest job shape only while preserving the occupancy
      // supported by the general merge capacity on this device.
      const int desired_blocks = baseline_blocks;
      std::uint32_t low = resident_merge_capacity_;
      std::uint32_t high = std::max(
          low, gpulsmopt2_detail::kCanonicalTournamentCapacityCeiling);
      while (low < high) {
        const std::uint32_t middle =
            low + (high - low + 1u) / 2u;
        if (active_blocks(middle) >= desired_blocks)
          low = middle;
        else
          high = middle - 1u;
      }
      const std::uint32_t widest_job_capacity = low;
      const std::size_t shared_bytes =
          gpulsmopt2_detail::canonical_tournament_workspace_bytes(
              widest_job_capacity, source_count);
      const std::uint32_t capacity =
          gpulsmopt2_detail::canonical_tournament_capacity(
              shared_bytes, source_count, 1u);
      if (!capacity)
        throw std::logic_error(
            "GPULSMOpt tournament workspace has no job capacity");
      canonical_job_capacities_[source_count] = capacity;
      canonical_tournament_shared_bytes_[source_count] = shared_bytes;
      tournament_attribute_bytes =
          std::max(tournament_attribute_bytes, shared_bytes);
    }
    if (tournament_attribute_bytes)
      CUDA_CHECK(cudaFuncSetAttribute(
          gpulsmopt2_detail::canonical_tournament_carry_jobs_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(tournament_attribute_bytes)));
    for (std::uint32_t source_count =
             gpulsmopt2_detail::kCanonicalTournamentMinimumSources;
         source_count <= maximum_sources; ++source_count) {
      const std::size_t shared_bytes =
          canonical_tournament_shared_bytes_[source_count];
      if (!shared_bytes) break;
      blocks_per_sm = 0;
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm,
          gpulsmopt2_detail::canonical_tournament_carry_jobs_kernel,
          gpulsmopt2_detail::kFoundationCompactionThreads,
          shared_bytes));
      if (!blocks_per_sm) {
        canonical_job_capacities_[source_count] = resident_merge_capacity_;
        canonical_tournament_shared_bytes_[source_count] = 0u;
        continue;
      }
      canonical_tournament_blocks_[source_count] =
          static_cast<std::uint32_t>(
              blocks_per_sm * properties.multiProcessorCount);
    }

    blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        gpulsmopt2_detail::canonical_fallback_carry_jobs_kernel,
        gpulsmopt2_detail::kFoundationCompactionThreads,
        canonical_merge_workspace_bytes_));
    canonical_fallback_blocks_ = static_cast<std::uint32_t>(
        std::max(1, blocks_per_sm) * properties.multiProcessorCount);
    CUDA_CHECK(cudaMemset(
        canonical_cell_counts_.data(), 0,
        canonical_cell_counts_.size() * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(
        canonical_job_prefixes_.data(), 0,
        canonical_job_prefixes_.size() *
            sizeof(gpulsmopt2_detail::CanonicalJobPrefix)));
    CUDA_CHECK(cudaMemset(
        canonical_next_job_.data(), 0, sizeof(std::uint32_t)));
  }

  cudaGraphExec_t capture_canonical_publication_graph(
      cudaStream_t capture_stream, std::uint32_t destination,
      std::uint32_t source_count, bool direct_epoch) {
    CUDA_CHECK(cudaStreamBeginCapture(
        capture_stream, cudaStreamCaptureModeGlobal));
    launch_canonical_publication_commands(
        capture_stream, destination, source_count, direct_epoch, true,
        false);
    cudaGraph_t graph{};
    CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
    cudaGraphExec_t graph_exec{};
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, 0ull));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return graph_exec;
  }

  void initialize_canonical_publication_graphs() {
    cudaStream_t capture_stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &capture_stream, cudaStreamNonBlocking));
    for (std::uint32_t destination = 0u;
         destination < canonical_level_count_; ++destination) {
      const std::uint32_t tier_begin = destination <
              canonical_regular_level_count_
          ? (destination / 3u) * 3u : canonical_regular_level_count_;
      const bool direct_epoch = tier_begin == 0u;
      const std::uint32_t source_count = 1u + tier_begin;
      canonical_publication_graph_execs_[destination] =
          capture_canonical_publication_graph(
              capture_stream, destination, source_count, direct_epoch);
    }
    CUDA_CHECK(cudaStreamDestroy(capture_stream));
  }

  void ensure_publication_capacity(std::size_t count,
                                   cudaStream_t stream) {
    const std::size_t current =
        std::min(publication_keys_a_.size(), publication_rows_a_.size());
    if (count <= current) return;
    if (count > gpulsmopt2_detail::kMaximumPublicationRows)
      throw std::bad_alloc();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    publication_keys_a_.grow(count);
    publication_rows_a_.grow(count);
  }

  void finalize_sparse_publication_ownership() {
    if (!pending_sparse_publication_) return;
    auto destination = pending_sparse_publication_root_;
    if (!destination)
      throw std::logic_error("missing staged sparse publication root");
    destination->segments.reserve(
        destination->segments.size() + destination->transfers.size());
    for (gpulsm_sparse::CapsuleTransferIntent &intent :
         destination->transfers) {
      auto *source = intent.source;
      if (!source || !source->segment)
        throw std::logic_error("capsule transfer source disappeared");
      source->segment->commit_usage(
          intent.live_bytes, intent.garbage_bytes);
      gpulsm_sparse::CapsuleSegmentOwnership moved{
          std::move(source->segment), intent.ordinal, 0u,
          intent.live_bytes, intent.garbage_bytes};
      source->ordinal = 0u;
      source->live_bytes = 0u;
      source->garbage_bytes = 0u;
      destination->segments.push_back(std::move(moved));
    }
    destination->transfers.clear();

    const std::uint32_t destination_level =
        pending_sparse_publication_destination_;
    const std::uint64_t destination_bit =
        std::uint64_t{1u} << destination_level;
    const std::uint64_t replaced =
        pending_sparse_publication_consumed_mask_ | destination_bit;
    std::uint64_t levels = replaced;
    while (levels) {
      const std::uint32_t level = static_cast<std::uint32_t>(
          __builtin_ctzll(levels));
      levels &= levels - 1u;
      sparse_roots_[level].reset();
    }
    sparse_exact_level_mask_ &= ~replaced;
    sparse_capsule_level_mask_ &= ~replaced;
    if (pending_sparse_publication_flags_) {
      sparse_roots_[destination_level] = std::move(destination);
      if (pending_sparse_publication_flags_ &
          gpulsm_sparse::kSparseHasExactHeads)
        sparse_exact_level_mask_ |= destination_bit;
      if (pending_sparse_publication_flags_ &
          gpulsm_sparse::kSparseHasCapsules)
        sparse_capsule_level_mask_ |= destination_bit;
    }
    pending_sparse_publication_root_.reset();
    pending_sparse_publication_consumed_mask_ = 0u;
    pending_sparse_publication_destination_ = 0u;
    pending_sparse_publication_flags_ = 0u;
    pending_sparse_publication_ = false;
  }

  void apply_publication_receipt() {
    if (!publication_receipt_pending_) return;
    const gpulsmopt2_detail::ResidentPublicationPlan &receipt =
        publication_receipt_.data()[0];
    publication_receipt_pending_ = false;
    publication_failure_status_ = receipt.status;
    if (receipt.status != gpulsmopt2_detail::kPublicationSuccess) {
      // Keep raw batches when publication fails.
      publication_failure_receipt_ = receipt;
      publication_failed_ = true;
      failed_epoch_signatures_ready_ = false;
      return;
    }

    const std::uint32_t destination = receipt.destination_level;
    if (destination >= canonical_level_count_) {
      publication_failed_ = true;
      publication_failure_status_ =
          gpulsmopt2_detail::kPublicationLevelOverflow;
      publication_failure_receipt_ = receipt;
      publication_failure_receipt_.status |=
          gpulsmopt2_detail::kPublicationLevelOverflow;
      failed_epoch_signatures_ready_ = false;
      return;
    }
    const std::uint64_t consumed = destination
        ? ((std::uint64_t{1} << destination) - 1u) : 0u;
    host_occupied_level_mask_ &= ~consumed;
    for (std::uint32_t level = 0u; level < destination; ++level)
      level_counts_[level] = 0u;
    level_counts_[destination] =
        static_cast<std::uint32_t>(receipt.survivor_count);
    if (receipt.survivor_count)
      host_occupied_level_mask_ |= std::uint64_t{1} << destination;
    else
      host_occupied_level_mask_ &= ~(std::uint64_t{1} << destination);
    active_levels_ = host_occupied_level_mask_
        ? 64u - static_cast<std::uint32_t>(
                      __builtin_clzll(host_occupied_level_mask_))
        : 0u;

    finalize_sparse_publication_ownership();
    for (auto &slot : sparse_pending_slots_) slot.reset();
    ++pending_sparse_generation_;
    if (!pending_sparse_generation_) pending_sparse_generation_ = 1u;
    pending_sparse_exact_ = false;
    pending_sparse_capsules_ = false;

    pending_batches_ = 0u;
    pending_records_ = 0u;
    pending_has_tombstones_ = false;
    std::fill_n(raw_batch_counts_, gpulsmopt2_detail::kBatchesPerEpoch, 0u);
    publication_failed_ = false;
    publication_failure_status_ = gpulsmopt2_detail::kPublicationSuccess;
    publication_failure_receipt_ = {};
    failed_epoch_signatures_ready_ = false;
  }

  void resolve_publication_receipt() {
    if (!publication_receipt_pending_) return;
    // Resolve the asynchronous publication receipt.
    const cudaError_t ready = cudaEventQuery(operation_done_);
    if (ready == cudaErrorNotReady)
      CUDA_CHECK(cudaEventSynchronize(operation_done_));
    else
      CUDA_CHECK(ready);
    apply_publication_receipt();
  }

  void resolve_publication_receipt_on_stream(cudaStream_t stream) {
    if (!publication_receipt_pending_) return;
    // Synchronize between tiles of one large update.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    apply_publication_receipt();
  }

  void reject_updates_after_publication_failure() const {
    if (!publication_failed_) return;
    const auto &receipt = publication_failure_receipt_;
    const std::string reason =
        publication_failure_status_ &
                gpulsmopt2_detail::kPublicationLevelOverflow
            ? "canonical carry capacity exhausted"
            : "publication failed";
    throw std::runtime_error(
        "GPULSMOpt " + reason + " with status " +
        std::to_string(publication_failure_status_) +
        "; destination=" + std::to_string(receipt.destination_level) +
        ", selected=" + std::to_string(receipt.selected_count) +
        ", reservation=" + std::to_string(receipt.raw_reservation) +
        ", capacity=" + std::to_string(receipt.output_capacity) +
        ", survivors=" + std::to_string(receipt.survivor_count) +
        ", jobs=" + std::to_string(receipt.job_count) +
        "; pending updates were preserved");
  }

  void prepare_failed_epoch_for_reads(cudaStream_t stream) {
    if (!publication_failed_ || failed_epoch_signatures_ready_) return;
    begin_operation(stream);
    gpulsmopt2_detail::rebuild_epoch_signatures_kernel<<<
        gpulsmopt2_detail::kQuotients / gpulsmopt2_detail::kThreads,
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            raw_signatures_.data(), raw_epoch_signatures_.data());
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
    failed_epoch_signatures_ready_ = true;
  }


  void begin_operation(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamWaitEvent(stream, operation_done_, 0));
  }

  void end_operation(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(operation_done_, stream));
  }

  void reset_updates(cudaStream_t stream) {
    if (sparse_exact_level_mask_ || sparse_capsule_level_mask_ ||
        pending_sparse_exact_ || pending_sparse_capsules_)
      CUDA_CHECK(cudaEventSynchronize(operation_done_));
    CUDA_CHECK(cudaMemsetAsync(descriptors_.data(), 0,
                               descriptors_.size() *
                                   sizeof(gpulsmopt2_detail::Descriptor),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(route_headers_.data(), 0,
                               route_headers_.size() *
                                   sizeof(gpulsmopt2_detail::RouteHeader),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(raw_offsets_.data(), 0,
                               raw_offsets_.size() * sizeof(std::uint32_t),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(
        raw_epoch_signatures_.data(), 0,
        raw_epoch_signatures_.size() * sizeof(std::uint64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        admission_counts_.data(), 0,
        admission_counts_.size() * sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        device_sparse_manifests_.data(), 0,
        device_sparse_manifests_.bytes(), stream));
    CUDA_CHECK(cudaMemsetAsync(
        device_pending_sparse_states_.data(), 0,
        device_pending_sparse_states_.bytes(), stream));
    for (auto &root : sparse_roots_) root.reset();
    for (auto &slot : sparse_pending_slots_) slot.reset();
    pending_sparse_publication_root_.reset();
    pending_sparse_publication_consumed_mask_ = 0u;
    pending_sparse_publication_destination_ = 0u;
    pending_sparse_publication_flags_ = 0u;
    pending_sparse_publication_ = false;
    ++pending_sparse_generation_;
    if (!pending_sparse_generation_) pending_sparse_generation_ = 1u;
    pending_sparse_device_generation_ = pending_sparse_generation_;
    next_capsule_segment_ordinal_ = 1u;
    sparse_exact_level_mask_ = 0u;
    sparse_capsule_level_mask_ = 0u;
    pending_sparse_exact_ = false;
    pending_sparse_capsules_ = false;
    pending_batches_ = 0u;
    pending_records_ = 0u;
    pending_has_tombstones_ = false;
    publication_receipt_pending_ = false;
    publication_failed_ = false;
    publication_failure_status_ = gpulsmopt2_detail::kPublicationSuccess;
    publication_failure_receipt_ = {};
    failed_epoch_signatures_ready_ = false;
    active_levels_ = 0u;
    host_occupied_level_mask_ = 0u;
    std::fill_n(level_counts_, gpulsmopt2_detail::kMaximumLevels, 0u);
    gpulsmopt2_detail::initialize_device_manifest_kernel<<<1, 1, 0, stream>>>(
        device_manifests_.data(), active_device_manifest_.data(),
        query_occupied_level_mask_.data(), 0u, 0u, 0u);
    std::fill_n(raw_batch_counts_, gpulsmopt2_detail::kBatchesPerEpoch, 0u);
  }

  std::vector<std::uint32_t> sealed_tier_slots() const {
    std::vector<std::uint32_t> result;
    std::uint32_t remaining = canonical_regular_level_count_;
    while (remaining) {
      const std::uint32_t slots = std::min(3u, remaining);
      result.push_back(slots);
      remaining -= slots;
    }
    if (canonical_regular_level_count_ < canonical_level_count_)
      result.push_back(1u);
    return result;
  }

  static std::unique_ptr<gpulsm_sparse::SealedWorkspace>
  make_sealed_workspace(std::size_t publication_capacity,
                        std::size_t level_zero_capacity) {
    const std::size_t capacity =
        gpulsmopt2_detail::canonical_level_layout(
            publication_capacity, level_zero_capacity)
            .highest_regular_capacity;
    // A single epoch remains on the restored publication path.  Provision
    // reusable sealed storage only when the configured forest can fuse
    // multiple epochs, exactly as LSMu sizes insertion storage from max_size
    // before its insertion timer starts.
    if (capacity <= level_zero_capacity) return {};
    if (capacity > std::numeric_limits<std::uint32_t>::max())
      throw std::length_error("sealed workspace exceeds 32-bit capacity");
    return std::make_unique<gpulsm_sparse::SealedWorkspace>(
        static_cast<std::uint32_t>(capacity));
  }

  std::uint32_t build_sealed_inline_root(
      gpulsm_sparse::SealedWorkspace &workspace,
      const std::uint32_t *keys, const std::uint32_t *values,
      std::uint32_t count, std::uint32_t level,
      std::uint32_t generation, cudaStream_t stream) {
    const std::uint32_t selected = workspace.prepare(
        keys, values, count, false, stream);
    ensure_level_storage_mapped(level, stream);
    const std::uint64_t capacity = level_capacity(level);
    if (count > capacity || selected > capacity)
      throw std::length_error("sealed inline root capacity");
    const std::uint64_t destination =
        level_begin(level) + std::uint64_t{generation} * capacity;
    gpulsm_sparse::gather_sealed_to_arena<<<4096,
        gpulsmopt2_detail::kThreads, 0, stream>>>(
        workspace.sorted_keys(), workspace.sorted_values(),
        workspace.selected(), workspace.selected_count(), resident_rows(),
        destination);
    gpulsmopt2_detail::ResidentPublicationPlan plan{};
    plan.selected_count = selected;
    plan.survivor_count = selected;
    plan.destination_level = level;
    plan.output_generation = generation;
    plan.output_begin = destination;
    plan.output_capacity = capacity;
    plan.status = gpulsmopt2_detail::kPublicationSuccess;
    CUDA_CHECK(cudaMemcpyAsync(
        resident_plan_.data(), &plan, sizeof(plan),
        cudaMemcpyHostToDevice, stream));
    gpulsmopt2_detail::build_canonical_rank_from_run_kernel<<<
        gpulsmopt2_detail::kQuotients,
        gpulsmopt2_detail::kFoundationCells, 0, stream>>>(
        resident_rows(), destination, workspace.section_offsets(), level,
        canonical_cell_ranks_.data());
    gpulsmopt2_detail::finalize_canonical_level_metadata_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
        workspace.section_offsets(), level_storage_spans_.data(),
        resident_plan_.data(), descriptors_.data(),
        static_cast<std::uint32_t>(route_stride_), route_headers_.data(),
        route_slices_.data(), route_logical_begins_.data(),
        route_quotients_.data(), level_q_logical_offsets_.data());
    CUDA_CHECK(cudaGetLastError());
    return selected;
  }

  std::uint32_t build_sealed_projection_carry_root(
      const gpulsmopt2_detail::Row *epoch_rows,
      const std::uint32_t *section_offsets,
      const std::uint32_t *section_counts,
      const std::uint16_t *epoch_ranks, std::uint32_t selected,
      std::uint64_t resident_sources,
      std::uint32_t level, std::uint32_t generation,
      cudaStream_t stream) {
    const std::uint32_t source_count = 1u + static_cast<std::uint32_t>(
        __builtin_popcountll(resident_sources));
    if (!resident_sources ||
        source_count > gpulsmopt2_detail::kMaximumMergeSources)
      throw std::invalid_argument("invalid sealed carry sources");
    const std::uint32_t source_limit = 63u -
        static_cast<std::uint32_t>(__builtin_clzll(resident_sources));
    const std::uint64_t source_prefix = source_limit == 63u
        ? ~std::uint64_t{0}
        : (std::uint64_t{1u} << (source_limit + 1u)) - 1u;
    if ((host_occupied_level_mask_ & source_prefix) != resident_sources)
      throw std::invalid_argument("non-prefix sealed carry sources");

    ensure_level_storage_mapped(level, stream);
    const std::uint64_t capacity = level_capacity(level);
    const std::uint64_t destination =
        level_begin(level) + std::uint64_t{generation} * capacity;
    gpulsmopt2_detail::ResidentPublicationPlan plan{};
    plan.selected_count = selected;
    plan.destination_level = level;
    plan.source_level_limit = source_limit;
    plan.source_count = source_count;
    plan.keep_tombstones = 1u;
    plan.output_generation = generation;
    plan.output_begin = destination;
    plan.output_capacity = capacity;
    plan.job_capacity =
        source_count < canonical_job_capacities_.size() &&
                canonical_job_capacities_[source_count]
            ? canonical_job_capacities_[source_count]
            : resident_merge_capacity_;
    plan.tournament_workspace_bytes =
        source_count < canonical_tournament_shared_bytes_.size()
            ? static_cast<std::uint32_t>(
                  canonical_tournament_shared_bytes_[source_count])
            : 0u;
    plan.status = selected > capacity
        ? gpulsmopt2_detail::kPublicationOutputOverflow
        : gpulsmopt2_detail::kPublicationSuccess;
    CUDA_CHECK(cudaMemcpyAsync(
        resident_plan_.data(), &plan, sizeof(plan),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(
        canonical_cell_counts_.data(), 0,
        canonical_cell_counts_.size() * sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        canonical_job_prefixes_.data(), 0,
        canonical_job_prefixes_.size() *
            sizeof(gpulsmopt2_detail::CanonicalJobPrefix),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        canonical_next_job_.data(), 0, sizeof(std::uint32_t), stream));
    gpulsmopt2_detail::count_canonical_merge_work_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
        section_counts, descriptors_.data(),
        device_manifests_.data(), active_device_manifest_.data(),
        resident_plan_.data(), balanced_merge_raw_counts_.data());
    gpulsmopt2_detail::count_canonical_planning_jobs_kernel<<<
        gpulsmopt2_detail::kPlanningTiles,
        gpulsmopt2_detail::kThreads, 0, stream>>>(
        balanced_merge_raw_counts_.data(), resident_plan_.data(),
        resident_tile_job_counts_.data());
    std::size_t scan_bytes = resident_scan_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        resident_scan_temp_.data(), scan_bytes,
        resident_tile_job_counts_.data(),
        resident_tile_job_offsets_.data(),
        gpulsmopt2_detail::kPlanningTiles + 1u, stream));
    gpulsmopt2_detail::emit_canonical_planning_jobs_kernel<<<
        gpulsmopt2_detail::kPlanningTiles,
        gpulsmopt2_detail::kThreads, 0, stream>>>(
        balanced_merge_raw_counts_.data(),
        resident_tile_job_offsets_.data(), resident_plan_.data(),
        static_cast<std::uint32_t>(maximum_resident_jobs_),
        balanced_merge_jobs_.data(), resident_job_raw_reservations_.data());
    gpulsmopt2_detail::validate_canonical_plan_kernel<<<1, 1, 0, stream>>>(
        resident_plan_.data(), resident_tile_job_offsets_.data(),
        static_cast<std::uint32_t>(maximum_resident_jobs_));
    gpulsmopt2_detail::resolve_canonical_job_boundaries_kernel<<<
        resident_planner_blocks_, 32u, 0, stream>>>(
        balanced_merge_raw_counts_.data(), balanced_merge_jobs_.data(),
        resident_job_raw_reservations_.data(), resident_plan_.data(),
        epoch_rows, section_offsets, section_counts, epoch_ranks,
        resident_rows(), descriptors_.data(), canonical_cell_ranks_.data(),
        device_manifests_.data(), active_device_manifest_.data());
    if (source_count < canonical_tournament_blocks_.size() &&
        canonical_tournament_blocks_[source_count]) {
      gpulsmopt2_detail::canonical_tournament_carry_jobs_kernel<<<
          canonical_tournament_blocks_[source_count],
          gpulsmopt2_detail::kFoundationCompactionThreads,
          canonical_tournament_shared_bytes_[source_count], stream>>>(
          balanced_merge_jobs_.data(),
          resident_job_raw_reservations_.data(), resident_plan_.data(),
          epoch_rows, section_offsets, section_counts, epoch_ranks,
          resident_rows(), descriptors_.data(), level_storage_spans_.data(),
          canonical_cell_ranks_.data(), device_manifests_.data(),
          active_device_manifest_.data(), canonical_job_prefixes_.data(),
          canonical_next_job_.data(), canonical_cell_counts_.data());
    } else {
      gpulsmopt2_detail::canonical_fallback_carry_jobs_kernel<<<
          canonical_fallback_blocks_,
          gpulsmopt2_detail::kFoundationCompactionThreads,
          canonical_merge_workspace_bytes_, stream>>>(
          balanced_merge_jobs_.data(), resident_plan_.data(),
          epoch_rows, section_offsets, section_counts, resident_rows(),
          descriptors_.data(),
          level_storage_spans_.data(), level_q_logical_offsets_.data(),
          device_manifests_.data(), active_device_manifest_.data(),
          canonical_job_prefixes_.data(), canonical_next_job_.data(),
          canonical_cell_counts_.data());
    }
    gpulsmopt2_detail::build_canonical_rank_from_counts_kernel<<<
        gpulsmopt2_detail::kQuotients,
        gpulsmopt2_detail::kFoundationCells, 0, stream>>>(
        canonical_cell_counts_.data(), resident_plan_.data(),
        canonical_level_count_, canonical_cell_ranks_.data(),
        foundation_section_output_counts_.data());
    scan_bytes = resident_scan_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        resident_scan_temp_.data(), scan_bytes,
        foundation_section_output_counts_.data(),
        foundation_source_offsets_.data(),
        gpulsmopt2_detail::kQuotients + 1u, stream));
    gpulsmopt2_detail::finalize_canonical_level_metadata_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
        foundation_source_offsets_.data(), level_storage_spans_.data(),
        resident_plan_.data(), descriptors_.data(),
        static_cast<std::uint32_t>(route_stride_), route_headers_.data(),
        route_slices_.data(), route_logical_begins_.data(),
        route_quotients_.data(), level_q_logical_offsets_.data());
    std::uint32_t survivors = 0u;
    CUDA_CHECK(cudaMemcpyAsync(
        &survivors,
        foundation_source_offsets_.data() + gpulsmopt2_detail::kQuotients,
        sizeof(survivors), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &plan, resident_plan_.data(), sizeof(plan),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (plan.status)
      throw std::runtime_error("sealed canonical carry failed");
    return survivors;
  }

  std::uint32_t build_sealed_inline_carry_root(
      gpulsm_sparse::SealedWorkspace &workspace,
      const std::uint32_t *keys, const std::uint32_t *values,
      std::uint32_t count, std::uint64_t resident_sources,
      std::uint32_t level, std::uint32_t generation,
      cudaStream_t stream) {
    const std::uint32_t selected = workspace.prepare(
        keys, values, count, true, stream);
    return build_sealed_projection_carry_root(
        workspace.epoch_rows(), workspace.section_offsets(),
        workspace.section_counts(), workspace.epoch_ranks(), selected,
        resident_sources, level, generation, stream);
  }

  std::uint32_t build_sealed_direct_root(
      const gpulsm_sparse::DirectRootWorkspace &workspace,
      const gpulsm_sparse::DirectRootPreparation &prepared,
      std::uint32_t level, std::uint32_t generation,
      cudaStream_t stream) {
    ensure_level_storage_mapped(level, stream);
    const std::uint64_t capacity = level_capacity(level);
    if (prepared.projection_rows > capacity)
      throw std::length_error("sealed direct root capacity");
    const std::uint64_t destination =
        level_begin(level) + std::uint64_t{generation} * capacity;
    gpulsmopt2_detail::copy_canonical_epoch_kernel<<<
        blocks(prepared.projection_rows), gpulsmopt2_detail::kThreads, 0,
        stream>>>(workspace.projection_rows(),
                  workspace.projection_count_device(), resident_rows(),
                  destination);
    gpulsmopt2_detail::ResidentPublicationPlan plan{};
    plan.selected_count = prepared.projection_rows;
    plan.survivor_count = prepared.projection_rows;
    plan.destination_level = level;
    plan.output_generation = generation;
    plan.output_begin = destination;
    plan.output_capacity = capacity;
    plan.status = gpulsmopt2_detail::kPublicationSuccess;
    CUDA_CHECK(cudaMemcpyAsync(
        resident_plan_.data(), &plan, sizeof(plan),
        cudaMemcpyHostToDevice, stream));
    gpulsmopt2_detail::build_canonical_rank_from_run_kernel<<<
        gpulsmopt2_detail::kQuotients,
        gpulsmopt2_detail::kFoundationCells, 0, stream>>>(
        resident_rows(), destination, workspace.section_offsets(), level,
        canonical_cell_ranks_.data());
    gpulsmopt2_detail::finalize_canonical_level_metadata_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
        workspace.section_offsets(), level_storage_spans_.data(),
        resident_plan_.data(), descriptors_.data(),
        static_cast<std::uint32_t>(route_stride_), route_headers_.data(),
        route_slices_.data(), route_logical_begins_.data(),
        route_quotients_.data(), level_q_logical_offsets_.data());
    CUDA_CHECK(cudaGetLastError());
    return prepared.projection_rows;
  }

  static std::uint32_t next_sealed_output(
      const gpulsm_sparse::ForestPlan &plan,
      const std::array<bool, gpulsmopt2_detail::kMaximumLevels> &built) {
    for (std::uint32_t candidate = 0u;
         candidate < plan.output_count; ++candidate) {
      if (built[candidate]) continue;
      const std::uint64_t destination_bit =
          std::uint64_t{1u} << plan.outputs[candidate].destination;
      bool has_unbuilt_reader = false;
      for (std::uint32_t reader = 0u; reader < plan.output_count;
           ++reader) {
        if (reader == candidate || built[reader]) continue;
        has_unbuilt_reader |=
            (plan.outputs[reader].resident_sources & destination_bit) != 0u;
      }
      if (!has_unbuilt_reader) return candidate;
    }
    throw std::logic_error("sealed slot-lifetime dependency cycle");
  }

  std::uint32_t sealed_output_generation(
      const gpulsm_sparse::ForestPlan &plan,
      const gpulsm_sparse::PlannedRoot &root,
      const gpulsmopt2_detail::DeviceManifest &manifest,
      gpulsm_sparse::SealedForestCommand &command,
      cudaStream_t stream) {
    if (root.destination >= canonical_level_count_)
      throw std::overflow_error("sealed destination exceeds forest");
    const std::uint64_t bit = std::uint64_t{1u} << root.destination;
    std::uint32_t generation = 0u;
    if ((plan.old_mask & bit) && (root.resident_sources & bit)) {
      if (root.destination + 1u != canonical_level_count_)
        throw std::logic_error(
            "sealed root overwrites its own nonterminal source");
      ensure_canonical_top_rollover_bank(stream);
      generation =
          (manifest.levels[root.destination].storage_generation ^ 1u) & 1u;
    }
    // A consumed nonterminal destination is reused only after
    // next_sealed_output has scheduled every old reader of that slot.
    if (generation) command.output_generation_bits |= bit;
    return generation;
  }

  void commit_sealed_forest(
      const gpulsm_sparse::ForestPlan &plan,
      const gpulsm_sparse::SealedForestCommand &command,
      const std::array<std::uint32_t,
                       gpulsmopt2_detail::kMaximumLevels> &output_counts,
      std::array<std::shared_ptr<gpulsm_sparse::StagedRootOverlay>,
                 gpulsmopt2_detail::kMaximumLevels> &staged_sparse_outputs,
      std::uint32_t active, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(
        sealed_device_command_.data(), &command, sizeof(command),
        cudaMemcpyHostToDevice, stream));
    gpulsm_sparse::publish_sealed_forest<<<
        1, gpulsmopt2_detail::kMaximumLevels, 0, stream>>>(
        sealed_device_command_.data(), device_manifests_.data(),
        device_sparse_manifests_.data(), active_device_manifest_.data(),
        query_occupied_level_mask_.data(), sealed_device_receipt_.data());
    // Preserve the sparse table's shared-index invariant for the next
    // untouched ordinary publication graph.
    CUDA_CHECK(cudaMemcpyAsync(
        device_sparse_manifests_.data() + active,
        device_sparse_manifests_.data() + (active ^ 1u),
        sizeof(gpulsm_sparse::DeviceSparseManifest),
        cudaMemcpyDeviceToDevice, stream));
    gpulsm_sparse::SealedForestReceipt receipt{};
    CUDA_CHECK(cudaMemcpyAsync(
        &receipt, sealed_device_receipt_.data(), sizeof(receipt),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (receipt.status || receipt.occupied_mask != plan.final_mask)
      throw std::runtime_error("sealed forest publication rejected");

    for (std::uint32_t level = 0u;
         level < gpulsmopt2_detail::kMaximumLevels; ++level) {
      auto &destination = staged_sparse_outputs[level];
      if (!destination) continue;
      destination->segments.reserve(
          destination->segments.size() + destination->transfers.size());
      for (gpulsm_sparse::CapsuleTransferIntent &intent :
           destination->transfers) {
        auto *source = intent.source;
        if (!source || !source->segment)
          throw std::logic_error(
              "sealed capsule transfer source disappeared");
        source->segment->commit_usage(
            intent.live_bytes, intent.garbage_bytes);
        gpulsm_sparse::CapsuleSegmentOwnership moved{
            std::move(source->segment), intent.ordinal, 0u,
            intent.live_bytes, intent.garbage_bytes};
        source->ordinal = 0u;
        source->live_bytes = 0u;
        source->garbage_bytes = 0u;
        destination->segments.push_back(std::move(moved));
      }
      destination->transfers.clear();
    }

    const std::uint64_t replaced = plan.consumed_mask | plan.output_mask;
    std::uint64_t levels = replaced;
    while (levels) {
      const std::uint32_t level = static_cast<std::uint32_t>(
          __builtin_ctzll(levels));
      levels &= levels - 1u;
      sparse_roots_[level].reset();
    }
    sparse_exact_level_mask_ &= ~replaced;
    sparse_capsule_level_mask_ &= ~replaced;
    for (std::uint32_t level = 0u;
         level < gpulsmopt2_detail::kMaximumLevels; ++level) {
      if (!staged_sparse_outputs[level]) continue;
      const std::uint64_t bit = std::uint64_t{1u} << level;
      sparse_roots_[level] = std::move(staged_sparse_outputs[level]);
      if (command.sparse_flags[level] &
          gpulsm_sparse::kSparseHasExactHeads)
        sparse_exact_level_mask_ |= bit;
      if (command.sparse_flags[level] &
          gpulsm_sparse::kSparseHasCapsules)
        sparse_capsule_level_mask_ |= bit;
    }
    for (std::uint32_t level = 0u;
         level < gpulsmopt2_detail::kMaximumLevels; ++level) {
      const std::uint64_t bit = std::uint64_t{1u} << level;
      if (plan.consumed_mask & bit) level_counts_[level] = 0u;
      if (plan.output_mask & bit) level_counts_[level] = output_counts[level];
    }
    host_occupied_level_mask_ = plan.final_mask;
    refresh_active_levels();
  }

  std::size_t try_admit_sealed_inline(
      const std::uint32_t *keys, const std::uint32_t *values,
      std::size_t count, cudaStream_t stream) {
    if (pending_batches_ || !count) return 0u;
    const std::uint64_t epoch_rows =
        std::uint64_t{batch_capacity_} *
        gpulsmopt2_detail::kBatchesPerEpoch;
    const std::uint64_t complete_epochs = count / epoch_rows;
    if (!complete_epochs) return 0u;
    const gpulsm_sparse::ForestPlan plan = gpulsm_sparse::plan_forest(
        host_occupied_level_mask_, complete_epochs, sealed_tier_slots());
    bool fusion = false;
    std::uint64_t maximum_interval = 0u;
    for (std::uint32_t output = 0u; output < plan.output_count; ++output) {
      const auto &root = plan.outputs[output];
      fusion |= root.raw_epochs > 1u || root.resident_sources != 0u;
      maximum_interval = std::max(
          maximum_interval, root.raw_epochs * epoch_rows);
    }
    if (!fusion) return 0u;
    if (!maximum_interval ||
        maximum_interval > std::numeric_limits<std::uint32_t>::max())
      throw std::length_error("sealed interval exceeds 32 bits");
    if (!sealed_workspace_ ||
        maximum_interval > sealed_workspace_->capacity())
      return 0u;

    CUDA_CHECK(cudaStreamSynchronize(stream));
    gpulsmopt2_detail::DeviceManifest manifests[2]{};
    std::uint32_t active = 0u;
    CUDA_CHECK(cudaMemcpy(&active, active_device_manifest_.data(),
                          sizeof(active), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(manifests, device_manifests_.data(),
                          sizeof(manifests), cudaMemcpyDeviceToHost));
    active &= 1u;
    gpulsm_sparse::SealedForestCommand command{};
    command.expected_mask = plan.old_mask;
    command.consumed_mask = plan.consumed_mask;
    command.output_mask = plan.output_mask;
    command.final_mask = plan.final_mask;
    std::array<std::uint32_t, gpulsmopt2_detail::kMaximumLevels>
        output_counts{};
    std::array<std::shared_ptr<gpulsm_sparse::StagedRootOverlay>,
               gpulsmopt2_detail::kMaximumLevels>
        staged_sparse_outputs{};
    gpulsm_sparse::SealedWorkspace &workspace = *sealed_workspace_;
    std::array<bool, gpulsmopt2_detail::kMaximumLevels> built{};
    for (std::uint32_t ordinal = 0u; ordinal < plan.output_count;
         ++ordinal) {
      const std::uint32_t output = next_sealed_output(plan, built);
      built[output] = true;
      const auto &root = plan.outputs[output];
      if (root.destination >= canonical_level_count_)
        throw std::overflow_error("sealed destination exceeds forest");
      const std::uint64_t raw_begin = root.raw_begin * epoch_rows;
      const std::uint64_t raw_count = root.raw_epochs * epoch_rows;
      if (raw_begin > count || raw_count > count - raw_begin ||
          raw_count > level_capacity(root.destination))
        throw std::length_error("invalid sealed root interval");
      const std::uint32_t generation = sealed_output_generation(
          plan, root, manifests[active], command, stream);
      const auto *root_keys = keys + raw_begin;
      const auto *root_values = values + raw_begin;
      output_counts[root.destination] = root.resident_sources
          ? build_sealed_inline_carry_root(
                workspace, root_keys, root_values,
                static_cast<std::uint32_t>(raw_count),
                root.resident_sources, root.destination, generation,
                stream)
          : build_sealed_inline_root(
                workspace, root_keys, root_values,
                static_cast<std::uint32_t>(raw_count), root.destination,
                generation, stream);
      const std::uint64_t sparse_sources = root.resident_sources &
          (sparse_exact_level_mask_ | sparse_capsule_level_mask_);
      if (sparse_sources) {
        if (!sparse_refinement_workspace_)
          sparse_refinement_workspace_ = std::make_unique<
              gpulsm_sparse::SparseRefinementWorkspace>();
        const auto roster = sparse_refinement_workspace_->prepare_roster(
            sparse_pending_slots_, 0u, sparse_roots_,
            root.resident_sources, stream);
        if (!roster.count)
          throw std::logic_error(
              "sealed sparse carry has no exceptional roster");
        gpulsm_sparse::CompletionSourceView completion{};
        completion.incoming = gpulsm_sparse::inline_u32_batch(
            workspace.sorted_keys(), workspace.sorted_values(), raw_count);
        completion.incoming_sorted_heads = workspace.sorted_keys();
        completion.incoming_sorted_refs = nullptr;
        completion.incoming_records = static_cast<std::uint32_t>(raw_count);
        auto staged =
            std::make_shared<gpulsm_sparse::StagedRootOverlay>();
        const std::uint64_t capacity = level_capacity(root.destination);
        const std::uint64_t destination = level_begin(root.destination) +
            std::uint64_t{generation} * capacity;
        const gpulsm_sparse::SparseRefinementResult refined =
            sparse_refinement_workspace_->refine(
                completion, 0u, resident_rows(), descriptors_.data(),
                root.destination, destination,
                output_counts[root.destination], true, generation,
                next_capsule_segment_ordinal_, *staged, stream);
        if (refined.allocated_capsule_bytes) {
          if (next_capsule_segment_ordinal_ ==
              std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error(
                "GPULSMOpt capsule ordinal overflow");
          ++next_capsule_segment_ordinal_;
        }
        std::uint32_t flags = 0u;
        if (refined.exact_heads)
          flags |= gpulsm_sparse::kSparseHasExactHeads;
        if (refined.capsules)
          flags |= gpulsm_sparse::kSparseHasCapsules;
        if (flags) {
          command.sparse_states[root.destination] =
              reinterpret_cast<std::uint64_t>(staged->device_state.data());
          command.sparse_flags[root.destination] = flags;
          staged_sparse_outputs[root.destination] = std::move(staged);
        }
      }
    }

    note_sparse_workspace_high_water(workspace.bytes());
    commit_sealed_forest(
        plan, command, output_counts, staged_sparse_outputs, active, stream);
    return static_cast<std::size_t>(complete_epochs * epoch_rows);
  }

  std::uint64_t try_admit_sealed_mixed(
      gpulsm_sparse::RecordBatchView source, cudaStream_t stream) {
    if (pending_batches_ || !source.count) return 0u;
    const std::uint64_t epoch_rows =
        std::uint64_t{batch_capacity_} *
        gpulsmopt2_detail::kBatchesPerEpoch;
    const std::uint64_t complete_epochs = source.count / epoch_rows;
    if (!complete_epochs) return 0u;
    const gpulsm_sparse::ForestPlan plan = gpulsm_sparse::plan_forest(
        host_occupied_level_mask_, complete_epochs, sealed_tier_slots());
    bool fusion = false;
    std::uint64_t maximum_interval = 0u;
    for (std::uint32_t output = 0u; output < plan.output_count; ++output) {
      const auto &root = plan.outputs[output];
      fusion |= root.raw_epochs > 1u || root.resident_sources != 0u;
      maximum_interval = std::max(
          maximum_interval, root.raw_epochs * epoch_rows);
    }
    if (!fusion) return 0u;
    if (!maximum_interval ||
        maximum_interval >= gpulsm_sparse::kCompletionIncoming)
      throw std::length_error(
          "sealed mixed interval exceeds exact locator capacity");

    CUDA_CHECK(cudaStreamSynchronize(stream));
    gpulsmopt2_detail::DeviceManifest manifests[2]{};
    std::uint32_t active = 0u;
    CUDA_CHECK(cudaMemcpy(&active, active_device_manifest_.data(),
                          sizeof(active), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(manifests, device_manifests_.data(),
                          sizeof(manifests), cudaMemcpyDeviceToHost));
    active &= 1u;
    gpulsm_sparse::SealedForestCommand command{};
    command.expected_mask = plan.old_mask;
    command.consumed_mask = plan.consumed_mask;
    command.output_mask = plan.output_mask;
    command.final_mask = plan.final_mask;
    std::array<std::uint32_t, gpulsmopt2_detail::kMaximumLevels>
        output_counts{};
    std::array<std::shared_ptr<gpulsm_sparse::StagedRootOverlay>,
               gpulsmopt2_detail::kMaximumLevels>
        staged_sparse_outputs{};
    gpulsm_sparse::DirectRootWorkspace workspace(
        static_cast<std::uint32_t>(maximum_interval));
    std::array<bool, gpulsmopt2_detail::kMaximumLevels> built{};
    for (std::uint32_t ordinal = 0u; ordinal < plan.output_count;
         ++ordinal) {
      const std::uint32_t output = next_sealed_output(plan, built);
      built[output] = true;
      const auto &root = plan.outputs[output];
      if (root.destination >= canonical_level_count_)
        throw std::overflow_error("sealed destination exceeds forest");
      const std::uint64_t raw_begin = root.raw_begin * epoch_rows;
      const std::uint64_t raw_count = root.raw_epochs * epoch_rows;
      if (raw_begin > source.count ||
          raw_count > source.count - raw_begin ||
          raw_count > level_capacity(root.destination) ||
          raw_count >= gpulsm_sparse::kCompletionIncoming)
        throw std::length_error("invalid sealed mixed root interval");
      const std::uint32_t generation = sealed_output_generation(
          plan, root, manifests[active], command, stream);
      const gpulsm_sparse::RecordBatchView root_source =
          slice_record_batch(source, raw_begin, raw_count);
      const auto prepared = workspace.prepare(
          root_source, static_cast<std::uint32_t>(raw_count), stream);
      note_sparse_workspace_high_water(workspace.bytes());
      workspace.materialize_epoch(
          root_source, prepared.projection_rows, stream);
      output_counts[root.destination] = root.resident_sources
          ? build_sealed_projection_carry_root(
                workspace.projection_rows(), workspace.section_offsets(),
                workspace.section_counts(), workspace.epoch_ranks(),
                prepared.projection_rows, root.resident_sources,
                root.destination, generation, stream)
          : build_sealed_direct_root(
                workspace, prepared, root.destination, generation, stream);

      const std::uint64_t sparse_sources = root.resident_sources &
          (sparse_exact_level_mask_ | sparse_capsule_level_mask_);
      if (!prepared.sparse_heads && !sparse_sources) continue;
      if (!sparse_refinement_workspace_)
        sparse_refinement_workspace_ = std::make_unique<
            gpulsm_sparse::SparseRefinementWorkspace>();
      const auto roster = sparse_refinement_workspace_->prepare_roster(
          sparse_pending_slots_, 0u, sparse_roots_,
          root.resident_sources, stream, workspace.sparse_heads(),
          prepared.sparse_heads);
      if (!roster.count)
        throw std::logic_error(
            "sealed mixed carry has no exceptional roster");
      gpulsm_sparse::CompletionSourceView completion{};
      completion.incoming = root_source;
      completion.incoming_sorted_heads = workspace.sorted_heads();
      completion.incoming_sorted_refs = workspace.sorted_refs();
      completion.incoming_records = static_cast<std::uint32_t>(raw_count);
      auto staged =
          std::make_shared<gpulsm_sparse::StagedRootOverlay>();
      const std::uint64_t capacity = level_capacity(root.destination);
      const std::uint64_t destination = level_begin(root.destination) +
          std::uint64_t{generation} * capacity;
      const gpulsm_sparse::SparseRefinementResult refined =
          sparse_refinement_workspace_->refine(
              completion, 0u, resident_rows(), descriptors_.data(),
              root.destination, destination,
              output_counts[root.destination], true, generation,
              next_capsule_segment_ordinal_, *staged, stream);
      if (!root.resident_sources &&
          staged->state.logical_count != prepared.logical_rows)
        throw std::logic_error(
            "sealed mixed logical/projection count mismatch");
      if (refined.allocated_capsule_bytes) {
        if (next_capsule_segment_ordinal_ ==
            std::numeric_limits<std::uint32_t>::max())
          throw std::overflow_error("GPULSMOpt capsule ordinal overflow");
        ++next_capsule_segment_ordinal_;
      }
      std::uint32_t flags = 0u;
      if (refined.exact_heads)
        flags |= gpulsm_sparse::kSparseHasExactHeads;
      if (refined.capsules)
        flags |= gpulsm_sparse::kSparseHasCapsules;
      if (flags) {
        command.sparse_states[root.destination] =
            reinterpret_cast<std::uint64_t>(staged->device_state.data());
        command.sparse_flags[root.destination] = flags;
        staged_sparse_outputs[root.destination] = std::move(staged);
      }
    }

    note_sparse_workspace_high_water(workspace.bytes());
    commit_sealed_forest(
        plan, command, output_counts, staged_sparse_outputs, active, stream);
    return complete_epochs * epoch_rows;
  }

  void admit(const std::uint32_t *keys, const std::uint32_t *values,
             std::size_t count, bool tombstone, cudaStream_t stream) {
    if (!count) return;
    if (!keys || (!tombstone && !values))
      throw std::invalid_argument("invalid GPULSMOpt update batch");
    begin_operation(stream);
    std::size_t consumed = 0u;
    bool incomplete = false;
    if (!tombstone && pending_batches_) {
      const std::size_t slots =
          gpulsmopt2_detail::kBatchesPerEpoch - pending_batches_;
      const std::size_t prefix_rows = slots * batch_capacity_;
      if (count >= prefix_rows) {
        while (pending_batches_ < gpulsmopt2_detail::kBatchesPerEpoch) {
          admit_tile(keys + consumed, values + consumed,
                     static_cast<std::uint32_t>(batch_capacity_), false,
                     stream);
          consumed += batch_capacity_;
        }
        if (publication_receipt_pending_)
          resolve_publication_receipt_on_stream(stream);
        if (publication_failed_) incomplete = true;
      }
    }
    if (!incomplete && !tombstone && consumed < count)
      consumed += try_admit_sealed_inline(
          keys + consumed, values + consumed, count - consumed, stream);
    while (!incomplete && consumed < count) {
      const std::size_t remaining = count - consumed;
      const std::uint32_t tile_count = static_cast<std::uint32_t>(
          std::min(remaining, batch_capacity_));
      admit_tile(keys + consumed,
                 tombstone ? nullptr : values + consumed,
                 tile_count, tombstone, stream);
      consumed += tile_count;
      if (consumed < count && publication_receipt_pending_)
        resolve_publication_receipt_on_stream(stream);
      if (consumed < count && publication_failed_) {
        incomplete = true;
        break;
      }
    }
    end_operation(stream);
    if (incomplete)
      throw std::runtime_error(
          "GPULSMOpt publication failed while tiling an update; "
          "accepted pending records were preserved");
  }

  void admit_tile(const std::uint32_t *keys, const std::uint32_t *values,
                  std::uint32_t n, bool tombstone,
                  cudaStream_t stream) {
    const std::uint32_t slot = pending_batches_;
    pending_records_ += n;
    pending_has_tombstones_ |= tombstone;
    std::uint64_t *batch_signatures = raw_signatures_.data() +
        std::size_t{slot} * gpulsmopt2_detail::kQuotients;
    CUDA_CHECK(cudaMemsetAsync(
        batch_signatures, 0,
        gpulsmopt2_detail::kQuotients * sizeof(std::uint64_t), stream));
    std::uint32_t *batch_offsets = raw_offsets_.data() +
        std::size_t{slot} * (gpulsmopt2_detail::kQuotients + 1u);
    gpulsmopt2_detail::count_admission_quotients_kernel<<<
        blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
            keys, n, admission_counts_.data(), radix_ids_out_.data());
    std::size_t scan_bytes = admission_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        admission_temp_.data(), scan_bytes, admission_counts_.data(),
        batch_offsets, gpulsmopt2_detail::kQuotients + 1u, stream));
    std::uint32_t *destination_keys = raw_keys_.data() +
        std::size_t{slot} * batch_capacity_;
    gpulsmopt2_detail::RawPayload *destination_payloads =
        raw_payloads_.data() + std::size_t{slot} * batch_capacity_;
    if (tombstone) {
      gpulsmopt2_detail::scatter_admission_records_kernel<true><<<
          blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
              keys, values, n, slot, batch_offsets, radix_ids_out_.data(),
              destination_keys, destination_payloads);
    } else {
      gpulsmopt2_detail::scatter_admission_records_kernel<false><<<
          blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
              keys, values, n, slot, batch_offsets, radix_ids_out_.data(),
              destination_keys, destination_payloads);
    }
    gpulsmopt2_detail::build_admission_signatures_kernel<<<
        blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
            destination_keys, n, batch_signatures);
    gpulsmopt2_detail::commit_admission_metadata_kernel<<<
        gpulsmopt2_detail::kQuotients / gpulsmopt2_detail::kThreads,
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            admission_counts_.data(), batch_signatures,
            raw_epoch_signatures_.data());
    CUDA_CHECK(cudaGetLastError());
    raw_batch_counts_[slot] = n;
    ++pending_batches_;
    if (pending_batches_ == gpulsmopt2_detail::kBatchesPerEpoch)
      publish_epoch(stream);
  }

  void launch_canonical_epoch_resolution(
      cudaStream_t stream, bool materialize_resident,
      std::uint32_t destination_level,
      std::uint16_t *rank_output = nullptr) {
    CUDA_CHECK(cudaMemsetAsync(
        publication_selected_count_.data(), 0, sizeof(std::uint32_t),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        local_epoch_overflow_flag_.data(), 0, sizeof(std::uint32_t),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        canonical_next_job_.data(), 0, sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        canonical_cell_counts_.data(), 0,
        std::size_t{canonical_epoch_workspace_slots_ + 1u} *
            sizeof(std::uint32_t),
        stream));
    gpulsmopt2_detail::count_canonical_epoch_jobs_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            raw_offsets_.data(), gpulsmopt2_detail::kBatchesPerEpoch,
            foundation_section_output_counts_.data(),
            balanced_merge_raw_counts_.data(),
            publication_selected_count_.data(),
            local_epoch_overflow_flag_.data());
    std::size_t scan_bytes = resident_scan_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        resident_scan_temp_.data(), scan_bytes,
        foundation_section_output_counts_.data(),
        foundation_source_offsets_.data(),
        gpulsmopt2_detail::kQuotients + 1u, stream));
    std::uint16_t *epoch_ranks = rank_output
        ? rank_output
        : canonical_cell_ranks_.data() +
              std::size_t{destination_level} *
                  gpulsmopt2_detail::kLocalRankEntries;
    const std::uint64_t resident_destination =
        level_begin(destination_level);

    if (materialize_resident) {
      gpulsmopt2_detail::resolve_canonical_epoch_active_jobs_kernel<true><<<
          canonical_epoch_resolver_blocks_,
          gpulsmopt2_detail::kFoundationCompactionThreads, 0, stream>>>(
              balanced_merge_raw_counts_.data(),
              publication_selected_count_.data(),
              canonical_next_job_.data(), raw_keys_.data(),
              raw_payloads_.data(), raw_offsets_.data(),
              static_cast<std::uint32_t>(batch_capacity_),
              gpulsmopt2_detail::kBatchesPerEpoch,
              foundation_source_offsets_.data(),
              publication_rows_a_.data(), resident_rows(),
              resident_destination, foundation_section_output_counts_.data(),
              epoch_ranks);
      gpulsmopt2_detail::resolve_canonical_epoch_oversized_kernel<true><<<
          canonical_epoch_resolver_blocks_,
          gpulsmopt2_detail::kFoundationCompactionThreads, 0, stream>>>(
              balanced_merge_raw_counts_.data(),
              local_epoch_overflow_flag_.data(),
              canonical_cell_counts_.data() +
                  canonical_epoch_workspace_slots_,
              raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
              static_cast<std::uint32_t>(batch_capacity_),
              gpulsmopt2_detail::kBatchesPerEpoch,
              foundation_source_offsets_.data(),
              publication_rows_a_.data(), resident_rows(),
              resident_destination, foundation_section_output_counts_.data(),
              epoch_ranks,
              reinterpret_cast<unsigned long long *>(
                  canonical_epoch_workspace_.data()),
              canonical_cell_counts_.data(),
              canonical_epoch_workspace_slots_);
    } else {
      gpulsmopt2_detail::resolve_canonical_epoch_active_jobs_kernel<false><<<
          canonical_epoch_resolver_blocks_,
          gpulsmopt2_detail::kFoundationCompactionThreads, 0, stream>>>(
              balanced_merge_raw_counts_.data(),
              publication_selected_count_.data(),
              canonical_next_job_.data(), raw_keys_.data(),
              raw_payloads_.data(), raw_offsets_.data(),
              static_cast<std::uint32_t>(batch_capacity_),
              gpulsmopt2_detail::kBatchesPerEpoch,
              foundation_source_offsets_.data(),
              publication_rows_a_.data(), resident_rows(),
              resident_destination, foundation_section_output_counts_.data(),
              epoch_ranks);
      gpulsmopt2_detail::resolve_canonical_epoch_oversized_kernel<false><<<
          canonical_epoch_resolver_blocks_,
          gpulsmopt2_detail::kFoundationCompactionThreads, 0, stream>>>(
              balanced_merge_raw_counts_.data(),
              local_epoch_overflow_flag_.data(),
              canonical_cell_counts_.data() +
                  canonical_epoch_workspace_slots_,
              raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
              static_cast<std::uint32_t>(batch_capacity_),
              gpulsmopt2_detail::kBatchesPerEpoch,
              foundation_source_offsets_.data(),
              publication_rows_a_.data(), resident_rows(),
              resident_destination, foundation_section_output_counts_.data(),
              epoch_ranks,
              reinterpret_cast<unsigned long long *>(
                  canonical_epoch_workspace_.data()),
              canonical_cell_counts_.data(),
              canonical_epoch_workspace_slots_);
    }
    gpulsmopt2_detail::sum_canonical_section_counts_kernel<<<
        1, gpulsmopt2_detail::kThreads, 0, stream>>>(
            foundation_section_output_counts_.data(),
            publication_selected_count_.data());
  }

  void launch_canonical_publication_commands(
      cudaStream_t stream, std::uint32_t destination,
      std::uint32_t source_count, bool direct_epoch,
      bool include_receipt, bool top_level_rollover,
      bool publish_manifest = true,
      const std::uint32_t *sparse_roster = nullptr,
      std::uint32_t sparse_roster_count = 0u) {
    // This check must precede epoch resolution: that stage indexes the rank
    // directory with destination and therefore cannot safely discover the
    // capacity error itself.
    if (destination >= canonical_level_count_) {
      auto &failure = publication_receipt_.data()[0];
      failure = {};
      failure.selected_count = pending_records_;
      failure.destination_level = destination;
      failure.source_count = source_count;
      failure.output_capacity = publication_capacity_;
      failure.status = gpulsmopt2_detail::kPublicationLevelOverflow;
      CUDA_CHECK(cudaMemcpyAsync(
          resident_plan_.data(), &failure, sizeof(failure),
          cudaMemcpyHostToDevice, stream));
      if (include_receipt) {
        CUDA_CHECK(cudaMemcpyAsync(
            publication_receipt_.data(), resident_plan_.data(),
            sizeof(gpulsmopt2_detail::ResidentPublicationPlan),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemsetAsync(
            raw_epoch_signatures_.data(), 0,
            raw_epoch_signatures_.size() * sizeof(std::uint64_t), stream));
      }
      return;
    }
    std::uint16_t *epoch_ranks = top_level_rollover
        ? canonical_rollover_epoch_ranks_->data()
        : canonical_cell_ranks_.data() +
              std::size_t{destination} *
                  gpulsmopt2_detail::kLocalRankEntries;
    launch_canonical_epoch_resolution(
        stream, direct_epoch, destination, epoch_ranks);
    if (sparse_roster_count) {
      gpulsm_sparse::patch_epoch_sparse_placeholders<<<
          blocks(sparse_roster_count), gpulsmopt2_detail::kThreads, 0,
          stream>>>(
          sparse_roster, sparse_roster_count,
          foundation_source_offsets_.data(),
          foundation_section_output_counts_.data(),
          publication_rows_a_.data(), resident_rows(),
          level_begin(destination), direct_epoch);
    }
    const std::uint32_t job_capacity =
        source_count < canonical_job_capacities_.size() &&
                canonical_job_capacities_[source_count]
            ? canonical_job_capacities_[source_count]
            : resident_merge_capacity_;
    const std::uint32_t tournament_workspace_bytes =
        source_count < canonical_tournament_shared_bytes_.size()
            ? static_cast<std::uint32_t>(
                  canonical_tournament_shared_bytes_[source_count])
            : 0u;
    gpulsmopt2_detail::choose_canonical_publication_path_kernel<<<
        1, 1, 0, stream>>>(
            publication_selected_count_.data(), device_manifests_.data(),
            active_device_manifest_.data(), level_storage_spans_.data(),
            canonical_regular_level_count_, canonical_level_count_,
            job_capacity,
            tournament_workspace_bytes, top_level_rollover,
            resident_plan_.data());

    if (direct_epoch) {
      gpulsmopt2_detail::finalize_canonical_section_metadata_kernel<<<
          blocks(gpulsmopt2_detail::kQuotients + 1u),
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              foundation_source_offsets_.data(),
              foundation_section_output_counts_.data(),
              publication_selected_count_.data(),
              level_storage_spans_.data(), resident_plan_.data(),
              descriptors_.data(),
              static_cast<std::uint32_t>(route_stride_),
              route_headers_.data(), route_slices_.data(),
              route_logical_begins_.data(), route_quotients_.data(),
              level_q_logical_offsets_.data());
    } else {
      CUDA_CHECK(cudaMemsetAsync(
          canonical_cell_counts_.data(), 0,
          canonical_cell_counts_.size() * sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(
          canonical_job_prefixes_.data(), 0,
          canonical_job_prefixes_.size() *
              sizeof(gpulsmopt2_detail::CanonicalJobPrefix), stream));
      CUDA_CHECK(cudaMemsetAsync(
          canonical_next_job_.data(), 0, sizeof(std::uint32_t), stream));
      gpulsmopt2_detail::count_canonical_merge_work_kernel<<<
          blocks(gpulsmopt2_detail::kQuotients),
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              foundation_section_output_counts_.data(), descriptors_.data(),
              device_manifests_.data(), active_device_manifest_.data(),
              resident_plan_.data(), balanced_merge_raw_counts_.data());
      gpulsmopt2_detail::count_canonical_planning_jobs_kernel<<<
          gpulsmopt2_detail::kPlanningTiles,
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              balanced_merge_raw_counts_.data(), resident_plan_.data(),
              resident_tile_job_counts_.data());
      std::size_t scan_bytes = resident_scan_temp_.size();
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          resident_scan_temp_.data(), scan_bytes,
          resident_tile_job_counts_.data(),
          resident_tile_job_offsets_.data(),
          gpulsmopt2_detail::kPlanningTiles + 1u, stream));
      gpulsmopt2_detail::emit_canonical_planning_jobs_kernel<<<
          gpulsmopt2_detail::kPlanningTiles,
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              balanced_merge_raw_counts_.data(),
              resident_tile_job_offsets_.data(), resident_plan_.data(),
              static_cast<std::uint32_t>(maximum_resident_jobs_),
              balanced_merge_jobs_.data(),
              resident_job_raw_reservations_.data());
      gpulsmopt2_detail::validate_canonical_plan_kernel<<<1, 1, 0, stream>>>(
          resident_plan_.data(), resident_tile_job_offsets_.data(),
          static_cast<std::uint32_t>(maximum_resident_jobs_));
      gpulsmopt2_detail::resolve_canonical_job_boundaries_kernel<<<
          resident_planner_blocks_, 32u, 0, stream>>>(
              balanced_merge_raw_counts_.data(), balanced_merge_jobs_.data(),
              resident_job_raw_reservations_.data(), resident_plan_.data(),
              publication_rows_a_.data(), foundation_source_offsets_.data(),
              foundation_section_output_counts_.data(),
              epoch_ranks,
              resident_rows(), descriptors_.data(),
              canonical_cell_ranks_.data(), device_manifests_.data(),
              active_device_manifest_.data());
      if (source_count < canonical_tournament_blocks_.size() &&
          canonical_tournament_blocks_[source_count]) {
        const std::size_t tournament_shared_bytes =
            canonical_tournament_shared_bytes_[source_count];
        gpulsmopt2_detail::canonical_tournament_carry_jobs_kernel<<<
            canonical_tournament_blocks_[source_count],
            gpulsmopt2_detail::kFoundationCompactionThreads,
            tournament_shared_bytes, stream>>>(
                balanced_merge_jobs_.data(),
                resident_job_raw_reservations_.data(),
                resident_plan_.data(), publication_rows_a_.data(),
                foundation_source_offsets_.data(),
                foundation_section_output_counts_.data(),
                epoch_ranks,
                resident_rows(), descriptors_.data(),
                level_storage_spans_.data(), canonical_cell_ranks_.data(),
                device_manifests_.data(), active_device_manifest_.data(),
                canonical_job_prefixes_.data(), canonical_next_job_.data(),
                canonical_cell_counts_.data());
      } else {
        gpulsmopt2_detail::canonical_fallback_carry_jobs_kernel<<<
            canonical_fallback_blocks_,
            gpulsmopt2_detail::kFoundationCompactionThreads,
            canonical_merge_workspace_bytes_, stream>>>(
                balanced_merge_jobs_.data(), resident_plan_.data(),
                publication_rows_a_.data(),
                foundation_source_offsets_.data(),
                foundation_section_output_counts_.data(), resident_rows(),
                descriptors_.data(), level_storage_spans_.data(),
                level_q_logical_offsets_.data(), device_manifests_.data(),
                active_device_manifest_.data(),
                canonical_job_prefixes_.data(), canonical_next_job_.data(),
                canonical_cell_counts_.data());
      }
      gpulsmopt2_detail::build_canonical_rank_from_counts_kernel<<<
          gpulsmopt2_detail::kQuotients,
          gpulsmopt2_detail::kFoundationCells, 0, stream>>>(
              canonical_cell_counts_.data(), resident_plan_.data(),
              canonical_level_count_,
              canonical_cell_ranks_.data(),
              foundation_section_output_counts_.data());
      scan_bytes = resident_scan_temp_.size();
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          resident_scan_temp_.data(), scan_bytes,
          foundation_section_output_counts_.data(),
          foundation_source_offsets_.data(),
          gpulsmopt2_detail::kQuotients + 1u, stream));
      gpulsmopt2_detail::finalize_canonical_level_metadata_kernel<<<
          blocks(gpulsmopt2_detail::kQuotients + 1u),
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              foundation_source_offsets_.data(), level_storage_spans_.data(),
              resident_plan_.data(), descriptors_.data(),
              static_cast<std::uint32_t>(route_stride_),
              route_headers_.data(), route_slices_.data(),
              route_logical_begins_.data(), route_quotients_.data(),
              level_q_logical_offsets_.data());
    }
    if (publish_manifest)
      gpulsmopt2_detail::publish_resident_manifest_kernel<<<
          1, gpulsmopt2_detail::kMaximumLevels, 0, stream>>>(
              resident_plan_.data(), device_manifests_.data(),
              active_device_manifest_.data(),
              query_occupied_level_mask_.data());
    if (include_receipt) {
      CUDA_CHECK(cudaMemcpyAsync(
          publication_receipt_.data(), resident_plan_.data(),
          sizeof(gpulsmopt2_detail::ResidentPublicationPlan),
          cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemsetAsync(
          raw_epoch_signatures_.data(), 0,
          raw_epoch_signatures_.size() * sizeof(std::uint64_t), stream));
    }
    CUDA_CHECK(cudaGetLastError());
  }

  void canonical_publication_parameters(
      std::uint32_t &destination, std::uint32_t &source_count,
      bool &direct_epoch) const {
    destination = gpulsmopt2_detail::kMaximumLevels;
    std::uint32_t tier_begin = 0u;
    for (std::uint32_t begin = 0u;
         begin < canonical_regular_level_count_; begin += 3u) {
      const std::uint32_t slots = std::min(
          3u, canonical_regular_level_count_ - begin);
      const std::uint64_t tier_mask =
          ((std::uint64_t{1} << slots) - 1u) << begin;
      const std::uint32_t filled = static_cast<std::uint32_t>(
          __builtin_popcountll(host_occupied_level_mask_ & tier_mask));
      if (filled < slots) {
        destination = begin + slots - 1u - filled;
        tier_begin = begin;
        break;
      }
    }
    if (destination == gpulsmopt2_detail::kMaximumLevels &&
        canonical_regular_level_count_ < canonical_level_count_ &&
        !(host_occupied_level_mask_ &
          (std::uint64_t{1} << canonical_regular_level_count_))) {
      destination = canonical_regular_level_count_;
      tier_begin = canonical_regular_level_count_;
    }
    const std::uint64_t carried = tier_begin
        ? (std::uint64_t{1} << tier_begin) - 1u : 0u;
    source_count = 1u + static_cast<std::uint32_t>(
        __builtin_popcountll(host_occupied_level_mask_ & carried));
    direct_epoch = tier_begin == 0u &&
        (host_occupied_level_mask_ != 0u || !pending_has_tombstones_) &&
        pending_records_ <= level_zero_capacity_;
  }

  bool launch_canonical_publication(cudaStream_t stream) {
    std::uint32_t destination = 0u, source_count = 0u;
    bool direct_epoch = false;
    canonical_publication_parameters(
        destination, source_count, direct_epoch);
    const bool top_level_rollover =
        destination >= canonical_level_count_;
    if (top_level_rollover) {
      // Recycle the full hierarchy into the alternate bank of its top level.
      // This is what lets arbitrarily many partially filled epochs proceed
      // while the number of live rows still fits the configured capacity.
      ensure_canonical_top_rollover_bank(stream);
      destination = canonical_level_count_ - 1u;
      const std::uint64_t source_mask = destination ==
              gpulsmopt2_detail::kMaximumLevels - 1u
          ? ~std::uint64_t{0}
          : (std::uint64_t{1} << (destination + 1u)) - 1u;
      source_count = 1u + static_cast<std::uint32_t>(
          __builtin_popcountll(host_occupied_level_mask_ & source_mask));
      direct_epoch = false;
    } else {
      ensure_level_storage_mapped(destination, stream);
    }
    const std::uint32_t tier_begin = destination <
            canonical_regular_level_count_
        ? (destination / 3u) * 3u : canonical_regular_level_count_;
    const bool graph_compatible = tier_begin != 0u || direct_epoch;
    cudaGraphExec_t graph_exec = !top_level_rollover && graph_compatible &&
            destination < canonical_publication_graph_execs_.size()
        ? canonical_publication_graph_execs_[destination] : nullptr;
    if (graph_exec) {
      CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
      return true;
    }
    launch_canonical_publication_commands(
        stream, destination, source_count, direct_epoch, false,
        top_level_rollover);
    return false;
  }

  std::uint64_t canonical_sparse_source_mask(
      std::uint32_t destination, bool top_level_rollover) const {
    if (top_level_rollover) {
      const std::uint64_t mask = destination >=
              gpulsmopt2_detail::kMaximumLevels - 1u
          ? ~std::uint64_t{0}
          : (std::uint64_t{1u} << (destination + 1u)) - 1u;
      return host_occupied_level_mask_ & mask;
    }
    const std::uint32_t tier_begin = destination <
            canonical_regular_level_count_
        ? (destination / 3u) * 3u
        : canonical_regular_level_count_;
    const std::uint64_t mask = tier_begin
        ? (std::uint64_t{1u} << tier_begin) - 1u : 0u;
    return host_occupied_level_mask_ & mask;
  }

  bool sparse_publication_required() const {
    if (pending_sparse_capsules_ || pending_sparse_exact_) return true;
    std::uint32_t destination = 0u, source_count = 0u;
    bool direct_epoch = false;
    canonical_publication_parameters(
        destination, source_count, direct_epoch);
    (void)source_count;
    (void)direct_epoch;
    const bool top_level_rollover =
        destination >= canonical_level_count_;
    if (top_level_rollover) destination = canonical_level_count_ - 1u;
    const std::uint64_t source_mask = canonical_sparse_source_mask(
        destination, top_level_rollover);
    return (source_mask &
            (sparse_exact_level_mask_ | sparse_capsule_level_mask_)) != 0u;
  }

  bool launch_sparse_publication(cudaStream_t stream) {
    std::uint32_t destination = 0u, source_count = 0u;
    bool direct_epoch = false;
    canonical_publication_parameters(
        destination, source_count, direct_epoch);
    const bool top_level_rollover =
        destination >= canonical_level_count_;
    if (top_level_rollover) {
      ensure_canonical_top_rollover_bank(stream);
      destination = canonical_level_count_ - 1u;
      const std::uint64_t source_mask = destination ==
              gpulsmopt2_detail::kMaximumLevels - 1u
          ? ~std::uint64_t{0}
          : (std::uint64_t{1u} << (destination + 1u)) - 1u;
      source_count = 1u + static_cast<std::uint32_t>(
          __builtin_popcountll(host_occupied_level_mask_ & source_mask));
      direct_epoch = false;
    } else {
      ensure_level_storage_mapped(destination, stream);
    }
    const std::uint64_t source_mask = canonical_sparse_source_mask(
        destination, top_level_rollover);
    if (!sparse_refinement_workspace_)
      sparse_refinement_workspace_ =
          std::make_unique<gpulsm_sparse::SparseRefinementWorkspace>();
    const auto roster = sparse_refinement_workspace_->prepare_roster(
        sparse_pending_slots_, pending_batches_, sparse_roots_, source_mask,
        stream);
    if (!roster.count)
      throw std::logic_error(
          "sparse publication was selected without exceptional heads");

    launch_canonical_publication_commands(
        stream, destination, source_count, direct_epoch, false,
        top_level_rollover, false, roster.heads, roster.count);
    gpulsmopt2_detail::ResidentPublicationPlan plan{};
    CUDA_CHECK(cudaMemcpyAsync(
        &plan, resident_plan_.data(), sizeof(plan), cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (plan.status) return false;
    if (plan.survivor_count > std::numeric_limits<std::uint32_t>::max())
      throw std::length_error("sparse root projection exceeds 32 bits");

    auto staged = std::make_shared<gpulsm_sparse::StagedRootOverlay>();
    const std::uint32_t segment_ordinal = next_capsule_segment_ordinal_;
    gpulsm_sparse::CompletionSourceView source{};
    source.pending_keys = raw_keys_.data();
    source.pending_payloads = raw_payloads_.data();
    source.pending_offsets = raw_offsets_.data();
    source.pending_slots = device_pending_sparse_states_.data();
    source.batch_capacity = static_cast<std::uint32_t>(batch_capacity_);
    source.pending_records = pending_records_;
    const gpulsm_sparse::SparseRefinementResult refined =
        sparse_refinement_workspace_->refine(
            source, pending_batches_, resident_rows(), descriptors_.data(),
            plan.destination_level, plan.output_begin,
            static_cast<std::uint32_t>(plan.survivor_count),
            plan.keep_tombstones != 0u, plan.output_generation,
            segment_ordinal, *staged, stream);
    note_sparse_workspace_high_water();
    if (refined.allocated_capsule_bytes) {
      if (next_capsule_segment_ordinal_ ==
          std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("GPULSMOpt capsule ordinal overflow");
      ++next_capsule_segment_ordinal_;
    }
    std::uint32_t sparse_flags = 0u;
    if (refined.exact_heads)
      sparse_flags |= gpulsm_sparse::kSparseHasExactHeads;
    if (refined.capsules)
      sparse_flags |= gpulsm_sparse::kSparseHasCapsules;
    gpulsm_sparse::publish_refined_manifest_kernel<<<
        1, gpulsmopt2_detail::kMaximumLevels, 0, stream>>>(
        resident_plan_.data(), device_manifests_.data(),
        device_sparse_manifests_.data(), active_device_manifest_.data(),
        query_occupied_level_mask_.data(),
        sparse_flags ? staged->device_state.data() : nullptr,
        sparse_flags);
    // The active ordinary index is shared with sparse metadata.  Keep the
    // tiny sparse table identical in both slots after a mixed commit so a
    // later all-ordinary graph can flip its original manifest unchanged.
    CUDA_CHECK(cudaMemcpyAsync(
        device_sparse_manifests_.data() + plan.active_manifest,
        device_sparse_manifests_.data() + plan.inactive_manifest,
        sizeof(gpulsm_sparse::DeviceSparseManifest),
        cudaMemcpyDeviceToDevice, stream));
    pending_sparse_publication_root_ = std::move(staged);
    pending_sparse_publication_consumed_mask_ = source_mask;
    pending_sparse_publication_destination_ = plan.destination_level;
    pending_sparse_publication_flags_ = sparse_flags;
    pending_sparse_publication_ = true;
    CUDA_CHECK(cudaGetLastError());
    return false;
  }

  void publish_epoch(cudaStream_t stream) {
    if (pending_records_ > publication_capacity_) {
      // Preserve an epoch that cannot reserve output.
      publication_failed_ = true;
      publication_failure_status_ =
          gpulsmopt2_detail::kPublicationOutputOverflow;
      publication_failure_receipt_ = {};
      publication_failure_receipt_.selected_count =
          static_cast<std::uint32_t>(pending_records_);
      publication_failure_receipt_.output_capacity = publication_capacity_;
      publication_failure_receipt_.status = publication_failure_status_;
      failed_epoch_signatures_ready_ = true;
      return;
    }
    const bool receipt_in_graph = sparse_publication_required()
        ? launch_sparse_publication(stream)
        : launch_canonical_publication(stream);
    CUDA_CHECK(cudaGetLastError());

    if (!receipt_in_graph) {
      // Copy the receipt to pinned host memory.
      CUDA_CHECK(cudaMemcpyAsync(
          publication_receipt_.data(), resident_plan_.data(),
          sizeof(gpulsmopt2_detail::ResidentPublicationPlan),
          cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemsetAsync(
          raw_epoch_signatures_.data(), 0,
          raw_epoch_signatures_.size() * sizeof(std::uint64_t), stream));
    }
    publication_receipt_pending_ = true;
    failed_epoch_signatures_ready_ = false;
    return;
  }

  static std::size_t aligned_id_bytes(std::size_t count) {
    return (count * sizeof(std::uint32_t) + 255u) & ~std::size_t{255u};
  }
  static std::size_t aligned_operation_bytes(std::size_t bytes) {
    return (bytes + 255u) & ~std::size_t{255u};
  }
  static std::size_t operation_cell_counts_bytes() {
    return std::size_t{gpulsmopt2_detail::kLocalRankEntries} *
        sizeof(std::uint32_t);
  }
  static std::size_t operation_epoch_workspace_rows(
      std::size_t batch_capacity) {
    return std::max<std::size_t>(
        batch_capacity * gpulsmopt2_detail::kBatchesPerEpoch,
        gpulsmopt2_detail::kCanonicalResolverSuffixes);
  }
  static std::size_t operation_epoch_workspace_offset() {
    return aligned_operation_bytes(operation_cell_counts_bytes());
  }
  static std::size_t operation_radix_storage_offset(
      std::size_t batch_capacity) {
    return aligned_operation_bytes(
        operation_epoch_workspace_offset() +
        operation_epoch_workspace_rows(batch_capacity) *
            sizeof(gpulsmopt2_detail::RawPayload));
  }
  static std::size_t radix_workspace_bytes(std::size_t count) {
    return aligned_id_bytes(count) * 3u +
        aligned_id_bytes(gpulsmopt2_detail::kQuotients + 1u);
  }
  static std::uint32_t tqrj_maximum_queries(std::size_t batch_capacity) {
    return static_cast<std::uint32_t>(
        batch_capacity * gpulsmopt2_detail::kBatchesPerEpoch);
  }
  static std::uint32_t tqrj_hash_maximum_entries(
      std::size_t batch_capacity) {
    return gpulsmopt2_detail::tqrj_hash_capacity(
        tqrj_maximum_queries(batch_capacity));
  }
  static std::size_t tqrj_hash_table_offset(std::size_t batch_capacity) {
    return aligned_operation_bytes(
        std::size_t{tqrj_maximum_queries(batch_capacity)} *
        sizeof(std::uint32_t));
  }
  static std::size_t tqrj_hash_task_offset(std::size_t batch_capacity) {
    return aligned_operation_bytes(
        tqrj_hash_table_offset(batch_capacity) +
        std::size_t{tqrj_hash_maximum_entries(batch_capacity)} *
            sizeof(std::uint32_t));
  }
  static std::size_t tqrj_hash_counter_offset(
      std::size_t batch_capacity) {
    return tqrj_hash_task_offset(batch_capacity) +
        gpulsmopt2_detail::kTqrjHashTaskBytes;
  }
  static std::size_t tqrj_hash_tile_offset(std::size_t batch_capacity) {
    return aligned_operation_bytes(
        tqrj_hash_counter_offset(batch_capacity) +
        gpulsmopt2_detail::kTqrjHashCounterBytes);
  }
  static std::uint32_t tqrj_maximum_tiles(
      std::size_t batch_capacity) {
    const std::uint32_t queries = tqrj_maximum_queries(batch_capacity);
    const std::uint32_t active =
        std::min(queries, gpulsmopt2_detail::kQuotients);
    return gpulsmopt2_detail::tqrj_hash_tile_count(queries) + active;
  }
  static std::size_t tqrj_hash_query_bases_offset(
      std::size_t batch_capacity) {
    return aligned_operation_bytes(
        tqrj_hash_tile_offset(batch_capacity) +
        std::size_t{2u * tqrj_maximum_tiles(batch_capacity)} *
            sizeof(gpulsmopt2_detail::TqrjHashTile));
  }
  static std::size_t tqrj_hash_workspace_bytes(
      std::size_t batch_capacity) {
    return tqrj_hash_query_bases_offset(batch_capacity) +
        aligned_id_bytes(gpulsmopt2_detail::kQuotients + 1u);
  }
  static std::size_t operation_workspace_initial_bytes(
      std::size_t batch_capacity) {
    return std::max(
        operation_radix_storage_offset(batch_capacity) +
            radix_workspace_bytes(batch_capacity),
        tqrj_hash_workspace_bytes(batch_capacity));
  }
  static std::size_t operation_workspace_maximum_bytes(
      std::size_t batch_capacity) {
    return std::max(
        operation_radix_storage_offset(batch_capacity) +
            radix_workspace_bytes(gpulsmopt2_detail::kMaximumOperationTile),
        tqrj_hash_workspace_bytes(batch_capacity));
  }
  void initialize_operation_workspace_views() {
    std::uint8_t *storage = operation_workspace_.data();
    canonical_cell_counts_.attach(
        reinterpret_cast<std::uint32_t *>(storage),
        gpulsmopt2_detail::kLocalRankEntries);
    canonical_epoch_workspace_.attach(
        reinterpret_cast<gpulsmopt2_detail::RawPayload *>(
            storage + operation_epoch_workspace_offset()),
        operation_epoch_workspace_rows(batch_capacity_));
  }
  void ensure_radix_workspace(std::size_t count) {
    const std::size_t capacity = std::max(radix_id_capacity_, count);
    const std::size_t ids_bytes = aligned_id_bytes(capacity);
    const std::size_t query_offset_bytes =
        aligned_id_bytes(gpulsmopt2_detail::kQuotients + 1u);
    const std::size_t required = ids_bytes * 3u + query_offset_bytes;
    const std::size_t radix_offset =
        operation_radix_storage_offset(batch_capacity_);
    operation_workspace_.grow(radix_offset + required);
    std::uint8_t *storage = operation_workspace_.data() + radix_offset;
    radix_keys_.attach(reinterpret_cast<std::uint32_t *>(storage), capacity);
    radix_ids_out_.attach(
        reinterpret_cast<std::uint32_t *>(storage + ids_bytes), capacity);
    radix_input_ids_ =
        reinterpret_cast<std::uint32_t *>(storage + ids_bytes * 2u);
    radix_workspace_ = storage + ids_bytes * 3u;
    radix_id_capacity_ = capacity;
  }
  std::uint32_t *radix_input_ids() { return radix_input_ids_; }
  std::uint32_t *query_quotient_offsets() {
    return reinterpret_cast<std::uint32_t *>(radix_workspace_);
  }
  void materialize_range_hot_sections(std::uint64_t total,
                                      cudaStream_t stream) {
    if (!total) return;
    if (total > gpulsmopt2_detail::kMaximumPublicationRows)
      throw std::length_error("GPULSMOpt crowded range input is too large");

    struct Window {
      std::uint32_t quotient_begin;
      std::uint32_t quotient_end;
      std::uint64_t input_begin;
      std::uint32_t input_count;
    };
    std::vector<Window> windows;
    if (total <= gpulsmopt2_detail::kRangeHotWindowRows) {
      windows.push_back({0u, gpulsmopt2_detail::kQuotients, 0u,
                         static_cast<std::uint32_t>(total)});
    } else {
      CUDA_CHECK(cudaMemcpyAsync(
          range_hot_offsets_receipt_.data(), range_hot_offsets_.data(),
          (gpulsmopt2_detail::kQuotients + 1u) * sizeof(std::uint64_t),
          cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const std::uint64_t *offsets = range_hot_offsets_receipt_.data();
      std::uint32_t begin = 0u;
      while (begin < gpulsmopt2_detail::kQuotients &&
             offsets[begin] < total) {
        const std::uint64_t base = offsets[begin];
        std::uint32_t end = begin;
        while (end < gpulsmopt2_detail::kQuotients &&
               offsets[end + 1u] - base <=
                   gpulsmopt2_detail::kRangeHotWindowRows)
          ++end;
        if (end == begin) ++end;
        const std::uint64_t count = offsets[end] - base;
        if (count > std::numeric_limits<int>::max())
          throw std::length_error(
              "one GPULSMOpt crowded section exceeds the GPU sort limit");
        windows.push_back({begin, end, base,
                           static_cast<std::uint32_t>(count)});
        begin = end;
      }
    }

    std::size_t maximum_window = 0u;
    for (const Window &window : windows)
      maximum_window = std::max<std::size_t>(
          maximum_window, window.input_count);
    if (range_hot_token_capacity_ < maximum_window) {
      range_hot_tokens_a_.resize(maximum_window);
      range_hot_tokens_b_.resize(maximum_window);
      range_hot_token_capacity_ = maximum_window;
    }
    ensure_publication_capacity(static_cast<std::size_t>(total), stream);

    std::size_t required_temp = 0u;
    for (const Window &window : windows) {
      std::size_t sort_bytes{};
      CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
          nullptr, sort_bytes, range_hot_tokens_a_.data(),
          range_hot_tokens_b_.data(),
          static_cast<int>(window.input_count),
          static_cast<int>(window.quotient_end - window.quotient_begin),
          range_hot_window_offsets_.data(),
          range_hot_window_offsets_.data() + 1u, 32, 48, stream));
      using KeyIterator = cub::TransformInputIterator<
          std::uint32_t, gpulsmopt2_detail::RangeHotTokenKey,
          const std::uint64_t *>;
      const KeyIterator keys(range_hot_tokens_b_.data(),
                             gpulsmopt2_detail::RangeHotTokenKey{});
      auto *winner_tokens = reinterpret_cast<std::uint64_t *>(
          publication_rows_a_.data() + window.input_begin);
      std::size_t reduce_bytes{};
      CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
          nullptr, reduce_bytes, keys,
          publication_keys_a_.data() + window.input_begin,
          range_hot_tokens_b_.data(), winner_tokens,
          range_hot_selected_count_.data(),
          gpulsmopt2_detail::RangeHotNewestToken{
              raw_payloads_.data(), static_cast<std::uint32_t>(
                  batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch)},
          window.input_count, stream));
      required_temp = std::max(required_temp,
                               std::max(sort_bytes, reduce_bytes));
    }
    if (range_hot_temp_.size() < required_temp)
      range_hot_temp_.resize(required_temp);

    for (const Window &window : windows) {
      const std::uint32_t quotient_count =
          window.quotient_end - window.quotient_begin;
      gpulsmopt2_detail::make_range_hot_window_offsets_kernel<<<
          blocks(std::size_t{quotient_count} + 1u),
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              range_hot_offsets_.data(), window.quotient_begin,
              quotient_count, window.input_begin,
              range_hot_window_offsets_.data());
      gpulsmopt2_detail::emit_range_hot_tokens_kernel<<<
          quotient_count, gpulsmopt2_detail::kThreads, 0, stream>>>(
              raw_keys_.data(), raw_offsets_.data(),
              static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
              resident_rows(), descriptors_.data(), route_headers_.data(),
              route_slices_.data(), route_logical_begins_.data(),
              level_q_logical_offsets_.data(),
              query_occupied_level_mask_.data(), range_hot_counts_.data(),
              range_hot_offsets_.data(), window.quotient_begin,
              window.quotient_end, window.input_begin,
              range_hot_tokens_a_.data());
      std::size_t workspace_bytes = range_hot_temp_.size();
      CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
          range_hot_temp_.data(), workspace_bytes,
          range_hot_tokens_a_.data(), range_hot_tokens_b_.data(),
          static_cast<int>(window.input_count),
          static_cast<int>(quotient_count),
          range_hot_window_offsets_.data(),
          range_hot_window_offsets_.data() + 1u, 32, 48, stream));
      using KeyIterator = cub::TransformInputIterator<
          std::uint32_t, gpulsmopt2_detail::RangeHotTokenKey,
          const std::uint64_t *>;
      const KeyIterator keys(range_hot_tokens_b_.data(),
                             gpulsmopt2_detail::RangeHotTokenKey{});
      auto *winner_tokens = reinterpret_cast<std::uint64_t *>(
          publication_rows_a_.data() + window.input_begin);
      workspace_bytes = range_hot_temp_.size();
      CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
          range_hot_temp_.data(), workspace_bytes, keys,
          publication_keys_a_.data() + window.input_begin,
          range_hot_tokens_b_.data(), winner_tokens,
          range_hot_selected_count_.data(),
          gpulsmopt2_detail::RangeHotNewestToken{
              raw_payloads_.data(), static_cast<std::uint32_t>(
                  batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch)},
          window.input_count, stream));
      gpulsmopt2_detail::materialize_range_hot_winners_kernel<<<
          blocks(window.input_count), gpulsmopt2_detail::kThreads, 0,
          stream>>>(
              publication_rows_a_.data() + window.input_begin,
              range_hot_selected_count_.data(), raw_payloads_.data(),
              resident_rows(), route_headers_.data(), route_slices_.data(),
              route_logical_begins_.data(),
              level_q_logical_offsets_.data());
      gpulsmopt2_detail::build_range_hot_descriptors_kernel<<<
          blocks(quotient_count), gpulsmopt2_detail::kThreads, 0, stream>>>(
              publication_keys_a_.data() + window.input_begin,
              range_hot_selected_count_.data(), window.quotient_begin,
              window.quotient_end, window.input_begin,
              range_hot_descriptors_.data());
    }
    CUDA_CHECK(cudaGetLastError());
  }
  void launch_section_ranges(cudaStream_t stream, bool hot_ready) {
    const std::uint16_t *foundation_ranks = canonical_cell_ranks_.data();
    if (active_levels_)
      foundation_ranks = canonical_cell_ranks_.data() +
          std::size_t{foundation_level()} *
              gpulsmopt2_detail::kLocalRankEntries;
    gpulsmopt2_detail::cooperative_section_owned_range_kernel<
        gpulsmopt2_detail::SumRowsAggregate>
        <<<range_section_blocks_, gpulsmopt2_detail::kSectionRangeThreads,
           0, stream>>>(
            range_section_fragments_out_.data(),
            range_section_tasks_.data(),
            range_section_task_offsets_.data() +
                gpulsmopt2_detail::kQuotients,
            resident_rows(), descriptors_.data(), route_headers_.data(),
            route_slices_.data(), foundation_ranks,
            publication_rows_a_.data(),
            range_hot_descriptors_.data(), hot_ready,
            raw_keys_.data(),
            raw_payloads_.data(),
            raw_offsets_.data(), static_cast<std::uint32_t>(batch_capacity_),
            pending_batches_, range_fragment_partials_.data(),
            query_occupied_level_mask_.data());
  }
  void launch_fragment_ranges(std::uint32_t fragment_count,
                              std::uint32_t query_count,
                              const DeviceRangeOutputBatch &batch,
                              cudaStream_t stream, bool hot_ready) {
    gpulsmopt2_detail::warp_range_fragment_kernel<
        gpulsmopt2_detail::SumRowsAggregate>
        <<<(fragment_count + 3u) / 4u, 128, 0, stream>>>(
            range_fragments_.data(), fragment_count,
            range_fragment_offsets_.data() + query_count,
            batch.lo, batch.hi, resident_rows(), descriptors_.data(),
            route_headers_.data(), route_slices_.data(),
            publication_rows_a_.data(),
            range_hot_descriptors_.data(), hot_ready,
            raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
            static_cast<std::uint32_t>(batch_capacity_),
            pending_batches_, range_fragment_partials_.data(),
            query_occupied_level_mask_.data());
  }
  static std::size_t aligned_range_bytes(std::size_t bytes) {
    return (bytes + 255u) & ~std::size_t{255u};
  }
  template <class T>
  static void attach_range_view(gpulsmopt2_detail::Buffer<T> &view,
                                std::uint8_t *storage,
                                std::size_t &offset,
                                std::size_t count) {
    offset = aligned_range_bytes(offset);
    view.attach(reinterpret_cast<T *>(storage + offset), count);
    offset += count * sizeof(T);
  }
  void ensure_range_fragment_query_capacity(std::size_t count,
                                            std::size_t temp_bytes) {
    const std::size_t entries = count + 1u;
    const std::size_t view_bytes = aligned_range_bytes(
        entries * sizeof(std::uint32_t));
    const std::size_t bytes = view_bytes * 2u +
        aligned_range_bytes(temp_bytes);
    if (range_query_storage_.size() < bytes)
      range_query_storage_.resize(bytes);
    std::uint8_t *storage = range_query_storage_.data();
    std::size_t offset = 0u;
    attach_range_view(range_fragment_counts_, storage, offset, entries);
    attach_range_view(range_fragment_offsets_, storage, offset, entries);
    offset = aligned_range_bytes(offset);
    range_query_temp_ = storage + offset;
  }
  void ensure_range_fragment_capacity(std::size_t count) {
    if (range_fragments_.size() >= count &&
        range_fragment_partials_.size() >= count) return;
    std::size_t bytes = 0u;
    bytes += aligned_range_bytes(
        count * sizeof(gpulsmopt2_detail::RangeFragment));
    bytes += aligned_range_bytes(count * sizeof(unsigned long long));
    range_fragment_storage_.resize(bytes);
    std::uint8_t *storage = range_fragment_storage_.data();
    std::size_t offset = 0u;
    attach_range_view(range_fragments_, storage, offset, count);
    attach_range_view(range_fragment_partials_, storage, offset, count);
  }
  void ensure_range_section_capacity(std::size_t count,
                                     std::size_t temp_bytes) {
    if (range_section_keys_in_.size() >= count &&
        range_fragment_partials_.size() >= count &&
        range_section_temp_bytes_ >= temp_bytes) return;
    constexpr std::size_t sections =
        gpulsmopt2_detail::kQuotients + 1u;
    const std::size_t maximum_tasks = gpulsmopt2_detail::kQuotients +
        (count + gpulsmopt2_detail::kSectionTaskFragments - 1u) /
            gpulsmopt2_detail::kSectionTaskFragments;
    std::size_t bytes = 0u;
    bytes += aligned_range_bytes(count * sizeof(std::uint32_t)) * 2u;
    bytes += aligned_range_bytes(
        count * sizeof(gpulsmopt2_detail::SectionRangeFragment)) * 2u;
    bytes += aligned_range_bytes(count * sizeof(unsigned long long));
    bytes += aligned_range_bytes(sections * sizeof(std::uint32_t)) * 3u;
    bytes += aligned_range_bytes(
        maximum_tasks * sizeof(gpulsmopt2_detail::SectionRangeTask));
    bytes += aligned_range_bytes(temp_bytes);
    range_section_storage_.resize(bytes);
    std::uint8_t *storage = range_section_storage_.data();
    std::size_t offset = 0u;
    attach_range_view(range_section_keys_in_, storage, offset, count);
    attach_range_view(range_section_keys_out_, storage, offset, count);
    attach_range_view(range_section_fragments_in_, storage, offset, count);
    attach_range_view(range_section_fragments_out_, storage, offset, count);
    attach_range_view(range_fragment_partials_, storage, offset, count);
    attach_range_view(range_section_offsets_, storage, offset, sections);
    attach_range_view(range_section_task_offsets_, storage, offset, sections);
    attach_range_view(range_section_task_counts_, storage, offset, sections);
    attach_range_view(range_section_tasks_, storage, offset, maximum_tasks);
    offset = aligned_range_bytes(offset);
    range_section_temp_ = storage + offset;
    range_section_temp_bytes_ = temp_bytes;
  }
  std::size_t batch_capacity_{};
  std::size_t publication_capacity_{};
  std::size_t level_zero_capacity_{};
  std::size_t level_pool_capacity_{};
  std::uint32_t canonical_regular_level_count_{};
  std::uint32_t canonical_level_count_{};
  std::array<gpulsmopt2_detail::LevelStorageSpan,
             gpulsmopt2_detail::kMaximumLevels>
      level_storage_spans_host_{};
  std::uint32_t resident_merge_capacity_{};
  std::size_t canonical_merge_workspace_bytes_{};
  std::size_t maximum_resident_jobs_{};
  std::size_t route_stride_{};
  mutable std::mutex operation_mutex_;
  std::uint64_t host_occupied_level_mask_{};
  std::uint32_t pending_batches_{};
  std::uint32_t pending_records_{};
  std::uint32_t active_levels_{};
  bool pending_has_tombstones_{};
  bool publication_receipt_pending_{};
  bool publication_failed_{};
  bool failed_epoch_signatures_ready_{};
  std::uint32_t publication_failure_status_{};
  gpulsmopt2_detail::ResidentPublicationPlan
      publication_failure_receipt_{};
  std::size_t radix_id_capacity_{};
  std::uint32_t *radix_input_ids_{};
  void *radix_workspace_{};
  std::uint8_t *range_query_temp_{};
  std::uint8_t *range_section_temp_{};
  std::size_t range_section_temp_bytes_{};
  cudaEvent_t operation_done_{};
  std::array<cudaGraphExec_t,
             gpulsmopt2_detail::kMaximumLevels>
      canonical_publication_graph_execs_{};
  std::uint32_t canonical_fallback_blocks_{};
  std::array<std::uint32_t,
             gpulsmopt2_detail::kMaximumMergeSources + 1u>
      canonical_tournament_blocks_{};
  std::array<std::uint32_t,
             gpulsmopt2_detail::kMaximumMergeSources + 1u>
      canonical_job_capacities_{};
  std::uint32_t canonical_epoch_resolver_blocks_{};
  std::uint32_t canonical_epoch_workspace_slots_{};
  std::uint32_t tqrj_hash_worker_blocks_{};
  std::array<std::size_t,
             gpulsmopt2_detail::kMaximumMergeSources + 1u>
      canonical_tournament_shared_bytes_{};
  std::uint32_t resident_planner_blocks_{};
  std::uint32_t range_section_blocks_{};

  gpulsmopt2_detail::VirtualBuffer<std::uint32_t> arena_key_flags_;
  gpulsmopt2_detail::VirtualBuffer<std::uint32_t> arena_values_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::Descriptor> descriptors_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RouteHeader> route_headers_;
  gpulsmopt2_detail::VirtualBuffer<gpulsmopt2_detail::RouteSlice>
      route_slices_;
  gpulsmopt2_detail::Buffer<std::uint32_t> route_logical_begins_;
  gpulsmopt2_detail::Buffer<std::uint16_t> route_quotients_;
  gpulsmopt2_detail::Buffer<std::uint32_t> level_q_logical_offsets_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::DeviceManifest>
      device_manifests_;
  gpulsmopt2_detail::Buffer<std::uint32_t> active_device_manifest_;
  gpulsmopt2_detail::Buffer<std::uint64_t> query_occupied_level_mask_;
  gpulsm_sparse::Buffer<gpulsm_sparse::DeviceSparseManifest>
      device_sparse_manifests_;
  gpulsm_sparse::Buffer<gpulsm_sparse::PendingSlotBuildState>
      device_pending_sparse_states_;
  gpulsm_sparse::Buffer<gpulsm_sparse::SealedForestCommand>
      sealed_device_command_;
  gpulsm_sparse::Buffer<gpulsm_sparse::SealedForestReceipt>
      sealed_device_receipt_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::ResidentPublicationPlan>
      resident_plan_;
  gpulsmopt2_detail::PinnedBuffer<
      gpulsmopt2_detail::ResidentPublicationPlan> publication_receipt_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::LevelStorageSpan>
      level_storage_spans_;
  gpulsmopt2_detail::Buffer<std::uint16_t> canonical_cell_ranks_;
  std::unique_ptr<gpulsmopt2_detail::Buffer<std::uint16_t>>
      canonical_rollover_epoch_ranks_;
  gpulsmopt2_detail::VirtualBuffer<std::uint8_t> operation_workspace_;
  std::unique_ptr<gpulsm_sparse::SealedWorkspace> sealed_workspace_;
  gpulsmopt2_detail::Buffer<std::uint32_t> canonical_cell_counts_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::CanonicalJobPrefix>
      canonical_job_prefixes_;
  gpulsmopt2_detail::Buffer<std::uint32_t> canonical_next_job_;
  gpulsmopt2_detail::Buffer<std::uint32_t> raw_keys_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RawPayload> raw_payloads_;
  gpulsmopt2_detail::Buffer<std::uint32_t> raw_offsets_;
  gpulsmopt2_detail::Buffer<std::uint64_t> raw_signatures_;
  gpulsmopt2_detail::Buffer<std::uint64_t> raw_epoch_signatures_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RawPayload>
      canonical_epoch_workspace_;
  gpulsmopt2_detail::VirtualBuffer<std::uint32_t> publication_keys_a_;
  gpulsmopt2_detail::Buffer<std::uint32_t> publication_selected_count_;
  gpulsmopt2_detail::Buffer<std::uint32_t> foundation_source_offsets_,
      foundation_section_output_counts_;
  gpulsmopt2_detail::Buffer<std::uint64_t> balanced_merge_raw_counts_,
      resident_job_raw_reservations_;
  gpulsmopt2_detail::Buffer<std::uint32_t>
      resident_tile_job_counts_, resident_tile_job_offsets_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::BalancedMergeJob>
      balanced_merge_jobs_;
  gpulsmopt2_detail::Buffer<std::uint8_t> resident_scan_temp_;
  gpulsmopt2_detail::Buffer<std::uint32_t> local_epoch_overflow_flag_;
  gpulsmopt2_detail::VirtualBuffer<gpulsmopt2_detail::Row>
      publication_rows_a_;
  gpulsmopt2_detail::Buffer<std::uint32_t> admission_counts_;
  gpulsmopt2_detail::Buffer<std::uint8_t> admission_temp_;
  std::uint32_t level_counts_[gpulsmopt2_detail::kMaximumLevels]{};
  std::uint32_t raw_batch_counts_[gpulsmopt2_detail::kBatchesPerEpoch]{};
  gpulsmopt2_detail::Buffer<std::uint32_t> radix_keys_, radix_ids_out_;

  std::array<std::shared_ptr<gpulsm_sparse::StagedRootOverlay>,
             gpulsmopt2_detail::kMaximumLevels>
      sparse_roots_{};
  std::array<std::shared_ptr<gpulsm_sparse::StagedPendingSlotOverlay>,
             gpulsmopt2_detail::kBatchesPerEpoch>
      sparse_pending_slots_{};
  std::unique_ptr<gpulsm_sparse::PendingWorkspace>
      sparse_pending_workspace_;
  std::unique_ptr<gpulsm_sparse::SparseRefinementWorkspace>
      sparse_refinement_workspace_;
  std::unique_ptr<gpulsm_sparse::SparseReadWorkspace>
      sparse_read_workspace_;
  std::shared_ptr<gpulsm_sparse::StagedRootOverlay>
      pending_sparse_publication_root_;
  std::uint64_t pending_sparse_publication_consumed_mask_{};
  std::uint32_t pending_sparse_publication_destination_{};
  std::uint32_t pending_sparse_publication_flags_{};
  std::uint32_t pending_sparse_generation_{1u};
  std::uint32_t pending_sparse_device_generation_{};
  std::uint32_t next_capsule_segment_ordinal_{1u};
  std::uint64_t sparse_exact_level_mask_{};
  std::uint64_t sparse_capsule_level_mask_{};
  std::uint64_t sparse_workspace_high_water_bytes_{};
  bool pending_sparse_exact_{};
  bool pending_sparse_capsules_{};
  bool pending_sparse_publication_{};

  gpulsmopt2_detail::Buffer<unsigned long long> range_partials_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_reduction_completion_;
  gpulsmopt2_detail::Buffer<std::uint64_t> range_fragment_total_;
  gpulsmopt2_detail::PinnedBuffer<std::uint64_t> range_total_receipt_;
  gpulsmopt2_detail::Buffer<std::uint64_t> range_hot_counts_,
      range_hot_offsets_, range_hot_tokens_a_, range_hot_tokens_b_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_hot_window_offsets_,
      range_hot_selected_count_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::Descriptor>
      range_hot_descriptors_;
  gpulsmopt2_detail::Buffer<std::uint8_t> range_hot_temp_;
  gpulsmopt2_detail::PinnedBuffer<std::uint64_t> range_hot_total_receipt_,
      range_hot_offsets_receipt_;
  std::size_t range_hot_token_capacity_{};
  gpulsmopt2_detail::Buffer<std::uint8_t> range_query_storage_,
      range_fragment_storage_, range_section_storage_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_fragment_counts_,
      range_fragment_offsets_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RangeFragment> range_fragments_;
  gpulsmopt2_detail::Buffer<unsigned long long> range_fragment_partials_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_section_keys_in_,
      range_section_keys_out_, range_section_offsets_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::SectionRangeFragment>
      range_section_fragments_in_, range_section_fragments_out_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::SectionRangeTask>
      range_section_tasks_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_section_task_counts_,
      range_section_task_offsets_;
};
