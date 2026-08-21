#pragma once

#include "gpu_dictionary_adapter.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
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
constexpr std::uint32_t kLookupRouterSlots = 2048u;
constexpr std::uint32_t kLookupRouterMask = kLookupRouterSlots - 1u;
constexpr std::uint32_t kLookupRouterAttempts = 4u;
constexpr std::uint32_t kLookupRouterProbeTarget = 8u;
constexpr std::uint32_t kLookupRouterDenseRowsPerSection = 8u;
constexpr std::uint32_t kLookupRouterMinimumBatches = 5u;
static_assert((kLookupRouterSlots & (kLookupRouterSlots - 1u)) == 0u);
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

__global__ void mark_last_key_kernel(
    const std::uint32_t *keys, std::uint32_t count, std::uint8_t *keep) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  keep[i] = i + 1u == count || keys[i] != keys[i + 1u];
}

__global__ void gather_initial_level_kernel(
    const std::uint32_t *sorted_keys,
    const std::uint32_t *sorted_values,
    const std::uint32_t *selected,
    std::uint32_t count,
    std::uint32_t *level_keys,
    Row *level_rows) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t source = selected[i];
  const std::uint32_t key = sorted_keys[source];
  level_keys[i] = key;
  level_rows[i] = make_row(key, sorted_values[source], 0u);
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
    const std::uint32_t destination =
        query_ids ? query_ids[query_index] : query_index;
    out_values[destination] = live ? winner.value : out_found ? 0u : kInvalid;
    if (out_found) out_found[destination] = live;
    return;
  }
  const std::uint32_t destination =
      query_ids ? query_ids[query_index] : query_index;
  out_values[destination] = out_found ? 0u : kInvalid;
  if (out_found) out_found[destination] = 0u;
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
        const std::uint32_t destination = query_ids ? query_ids[i] : i;
        out_values[destination] =
            live ? winner.value : out_found ? 0u : kInvalid;
        if (out_found) out_found[destination] = live;
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

__device__ __forceinline__ std::uint32_t lookup_router_mix(
    std::uint32_t key, std::uint32_t seed) {
  std::uint32_t value = key ^ (0x9e3779b9u * (seed + 1u));
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return value;
}


__device__ __forceinline__ void lookup_router_location(
    std::uint32_t key, std::uint32_t seed,
    std::uint32_t &slot, std::uint32_t &step) {
  const std::uint32_t first = lookup_router_mix(key, seed);
  const std::uint32_t second = lookup_router_mix(
      key ^ 0xa511e9b3u, seed + 0x632be59bu);
  slot = first & kLookupRouterMask;
  step = (second | 1u) & kLookupRouterMask;
}


__device__ __forceinline__ std::uint32_t lookup_router_find(
    std::uint32_t key, std::uint32_t seed,
    const std::uint32_t *owners, const std::uint32_t *query_keys) {
  std::uint32_t slot{}, step{};
  lookup_router_location(key, seed, slot, step);
#pragma unroll 1
  for (std::uint32_t probe = 0u; probe < kLookupRouterSlots; ++probe) {
    const std::uint32_t owner = owners[slot];
    if (owner == kInvalid) return kInvalid;
    if (query_keys[owner] == key) return owner;
    slot = (slot + step) & kLookupRouterMask;
  }
  return kInvalid;
}


__global__ void lookup_with_dense_router_kernel(
    const std::uint32_t *queries, std::uint32_t *out_values,
    std::uint8_t *out_found, std::uint32_t count,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets,
    std::uint32_t batch_stride, std::uint32_t pending_batches,
    const std::uint64_t *epoch_signatures,
    ResidentRows arena, const Descriptor *descriptors,
    const std::uint32_t *query_ids,
    const std::uint16_t *canonical_cell_ranks,
    const std::uint64_t *query_occupied_level_mask) {
  __shared__ std::uint32_t router_owners[kLookupRouterSlots];
  __shared__ std::uint32_t router_query_keys[kThreads];
  __shared__ unsigned long long router_winners[kThreads];
  __shared__ std::uint16_t router_sections[kThreads];
  __shared__ std::uint32_t router_section_count;
  __shared__ std::uint32_t router_max_probe;
  __shared__ std::uint32_t router_seed;

  const std::uint32_t local = threadIdx.x;
  const std::uint32_t i = blockIdx.x * blockDim.x + local;
  const bool valid = i < count;
  const std::uint32_t key = valid ? queries[i] : 0u;
  const std::uint32_t q = key >> 16u;
  const DeviceManifestSnapshot manifest =
      load_query_manifest(query_occupied_level_mask);
  const std::uint64_t occupied_levels = manifest.occupied_level_mask;

  bool possible = false;
  if (valid) {
    const std::uint64_t bits = pending_signature_bits(key);
    possible = (epoch_signatures[q] & bits) == bits;
  }
  if (!__syncthreads_or(possible)) {
    if (valid)
      canonical_lookup_resident_only(
          key, i, out_values, out_found, arena, descriptors,
          canonical_cell_ranks, occupied_levels, query_ids);
    return;
  }

  router_query_keys[local] = key;
  router_winners[local] = 0ull;
  if (local == 0u) router_section_count = 0u;
  __syncthreads();

  if (valid && (local == 0u ||
                (router_query_keys[local - 1u] >> 16u) != q)) {
    const std::uint32_t section = atomicAdd(&router_section_count, 1u);
    router_sections[section] = static_cast<std::uint16_t>(q);
  }
  __syncthreads();

  for (std::uint32_t attempt = 0u;
       attempt < kLookupRouterAttempts; ++attempt) {
    for (std::uint32_t slot = local; slot < kLookupRouterSlots;
         slot += kThreads)
      router_owners[slot] = kInvalid;
    if (local == 0u) router_max_probe = 0u;
    __syncthreads();

    std::uint32_t probes = 0u;
    if (valid) {
      std::uint32_t slot{}, step{};
      lookup_router_location(key, attempt, slot, step);
#pragma unroll 1
      for (; probes < kLookupRouterSlots; ++probes) {
        const std::uint32_t owner = atomicCAS(
            router_owners + slot, kInvalid, local);
        if (owner == kInvalid || router_query_keys[owner] == key) {
          ++probes;
          break;
        }
        slot = (slot + step) & kLookupRouterMask;
      }
      atomicMax(&router_max_probe, probes);
    }
    __syncthreads();
    if (router_max_probe <= kLookupRouterProbeTarget ||
        attempt + 1u == kLookupRouterAttempts) {
      if (local == 0u) router_seed = attempt;
      break;
    }
  }
  __syncthreads();

  const std::uint32_t section_count = router_section_count;
  const std::uint32_t task_count = section_count * pending_batches;
  std::uint32_t worker_width = 16u;
  if (task_count < 16u) {
    worker_width = task_count >= 8u ? 32u
        : task_count >= 4u ? 64u
        : task_count >= 2u ? 128u : 256u;
  }
  const std::uint32_t worker_groups = kThreads / worker_width;
  const std::uint32_t worker_group = local / worker_width;
  const std::uint32_t worker_local = local & (worker_width - 1u);

  for (std::uint32_t task = worker_group;
       task < task_count; task += worker_groups) {
    const std::uint32_t batch = task / section_count;
    const std::uint32_t section =
        router_sections[task - batch * section_count];
    const std::size_t oi =
        std::size_t{batch} * (kQuotients + 1u) + section;
    const std::uint32_t begin = raw_offsets[oi];
    const std::uint32_t end = raw_offsets[oi + 1u];
    for (std::uint32_t position = begin + worker_local;
         position < end; position += worker_width) {
      const std::uint32_t record = batch * batch_stride + position;
      const std::uint32_t pending_key = raw_keys[record];
      const std::uint32_t owner = lookup_router_find(
          pending_key, router_seed, router_owners, router_query_keys);
      const bool matched = owner != kInvalid;
      if (!matched) continue;
      const RawPayload payload = raw_payloads[record];
      const std::uint64_t order =
          static_cast<std::uint64_t>(raw_position(payload)) + 1u;
      const unsigned long long token =
          static_cast<unsigned long long>((order << 24u) | record);
      atomicMax(router_winners + owner, token);
    }
  }
  __syncthreads();

  if (!valid) return;
  const std::uint32_t owner = lookup_router_find(
      key, router_seed, router_owners, router_query_keys);
  const unsigned long long winner = owner == kInvalid
      ? 0ull : router_winners[owner];
  if (winner) {
    const std::uint32_t record =
        static_cast<std::uint32_t>(winner & ((1ull << 24u) - 1u));
    const RawPayload payload = raw_payloads[record];
    const bool live = (payload.metadata & kRawTombstone) == 0u;
    const std::uint32_t destination = query_ids ? query_ids[i] : i;
    out_values[destination] = live ? payload.value : out_found ? 0u : kInvalid;
    if (out_found) out_found[destination] = live;
    return;
  }
  canonical_lookup_resident_only(
      key, i, out_values, out_found, arena, descriptors,
      canonical_cell_ranks, occupied_levels, query_ids);
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

class GPULSMOpt {
public:
  struct DeviceKeyBatch {
    const std::uint32_t *keys = nullptr;
    std::size_t count = 0u;
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
        resident_plan_(1u),
        publication_receipt_(1u),
        level_storage_spans_(gpulsmopt2_detail::kMaximumLevels),
        canonical_cell_ranks_(
            std::size_t{canonical_level_count_} *
                gpulsmopt2_detail::kLocalRankEntries),
        canonical_cell_counts_(gpulsmopt2_detail::kLocalRankEntries),
        canonical_job_prefixes_(maximum_resident_jobs_),
        canonical_next_job_(1u),
        raw_keys_(gpulsmopt2_detail::kBatchesPerEpoch * batch_capacity_),
        raw_payloads_(gpulsmopt2_detail::kBatchesPerEpoch * batch_capacity_),
        raw_offsets_(std::size_t{gpulsmopt2_detail::kBatchesPerEpoch} *
                     (gpulsmopt2_detail::kQuotients + 1u)),
        raw_signatures_(std::size_t{gpulsmopt2_detail::kBatchesPerEpoch} *
                        gpulsmopt2_detail::kQuotients),
        raw_epoch_signatures_(gpulsmopt2_detail::kQuotients),
        canonical_epoch_workspace_(
            std::max<std::size_t>(
                batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch,
                gpulsmopt2_detail::kCanonicalResolverSuffixes)),
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
    // Preload the optional dense lookup kernel.
    cudaFuncAttributes dense_router_attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(
        &dense_router_attributes,
        gpulsmopt2_detail::lookup_with_dense_router_kernel));
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
    gpulsmopt2_detail::Buffer<std::uint8_t> keep(n);
    gpulsmopt2_detail::Buffer<std::uint32_t> selected_ids(n);
    gpulsmopt2_detail::Buffer<std::uint32_t> selected_count(1u);
    std::size_t sort_bytes{};
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, keys, sorted_keys.data(), values,
        sorted_values.data(), n, 0, 32, stream));
    gpulsmopt2_detail::Buffer<std::uint8_t> sort_temp(sort_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sort_temp.data(), sort_bytes, keys, sorted_keys.data(), values,
        sorted_values.data(), n, 0, 32, stream));
    gpulsmopt2_detail::mark_last_key_kernel<<<
        blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
        sorted_keys.data(), n, keep.data());
    CUDA_CHECK(cudaGetLastError());
    cub::CountingInputIterator<std::uint32_t> input_ids(0u);
    std::size_t select_bytes{};
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, select_bytes, input_ids, keep.data(), selected_ids.data(),
        selected_count.data(), n, stream));
    gpulsmopt2_detail::Buffer<std::uint8_t> select_temp(select_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        select_temp.data(), select_bytes, input_ids, keep.data(),
        selected_ids.data(), selected_count.data(), n, stream));
    std::uint32_t base_count{};
    CUDA_CHECK(cudaMemcpyAsync(&base_count, selected_count.data(),
                               sizeof(base_count), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const std::uint32_t level = initial_level_for_records(base_count);
    ensure_level_storage_mapped(level, stream);
    ensure_publication_capacity(base_count, stream);
    gpulsmopt2_detail::gather_initial_level_kernel<<<
        blocks(base_count), gpulsmopt2_detail::kThreads, 0, stream>>>(
            sorted_keys.data(), sorted_values.data(), selected_ids.data(),
            base_count, publication_keys_a_.data(),
            publication_rows_a_.data());
    CUDA_CHECK(cudaGetLastError());
    gpulsmopt2_detail::build_query_quotient_offsets_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            publication_keys_a_.data(), base_count,
            foundation_source_offsets_.data());
    const std::uint64_t destination = level_begin(level);
    const std::uint64_t capacity = level_capacity(level);
    if (base_count > capacity)
      throw std::bad_alloc();
    gpulsmopt2_detail::copy_canonical_epoch_kernel<<<
        blocks(base_count), gpulsmopt2_detail::kThreads, 0, stream>>>(
            publication_rows_a_.data(), selected_count.data(),
            resident_rows(), destination);
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

  void insert(const DeviceKeyValueBatch &batch, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    reject_updates_after_publication_failure();
    admit(batch.keys, batch.values, batch.count, false, stream);
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
    lookup_locked(batch, stream, quotients_grouped);
  }

  void range(const DeviceRangeOutputBatch &batch, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    prepare_failed_epoch_for_reads(stream);
    range_locked(batch, stream);
  }

  void successor(const DeviceSuccessorBatch &batch, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    prepare_failed_epoch_for_reads(stream);
    successor_locked(batch, stream);
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

  void lookup_locked(const DeviceLookupBatch &batch, cudaStream_t stream,
                     bool quotients_grouped) {
    if (!batch.count) return;
    if (!batch.queries || !batch.out_values)
      throw std::invalid_argument("invalid GPULSMOpt lookup");
    if (batch.count > gpulsmopt2_detail::kMaximumOperationTile) {
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
    const std::uint64_t dense_router_threshold =
        std::uint64_t{pending_batches_} * gpulsmopt2_detail::kQuotients *
        gpulsmopt2_detail::kLookupRouterDenseRowsPerSection;
    const bool use_dense_router = grouped &&
        pending_batches_ >= gpulsmopt2_detail::kLookupRouterMinimumBatches &&
        std::uint64_t{pending_records_} >= dense_router_threshold;
    if (use_dense_router) {
      gpulsmopt2_detail::lookup_with_dense_router_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
              queries, batch.out_values, batch.out_found, count,
              raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
              static_cast<std::uint32_t>(batch_capacity_),
              pending_batches_, raw_epoch_signatures_.data(),
              resident_rows(), descriptors_.data(), query_ids,
              canonical_cell_ranks_.data(),
              query_occupied_level_mask_.data());
    } else {
      gpulsmopt2_detail::canonical_lookup_with_pending_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
              queries, batch.out_values, batch.out_found, count,
              raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
              static_cast<std::uint32_t>(batch_capacity_),
              pending_batches_, raw_signatures_.data(),
              raw_epoch_signatures_.data(), resident_rows(),
              descriptors_.data(), canonical_cell_ranks_.data(),
              query_ids, query_occupied_level_mask_.data());
    }
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
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

public:
  std::size_t gpu_resident_bytes() const {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    const_cast<GPULSMOpt *>(this)->resolve_publication_receipt();
    const std::size_t rollover_rank_bytes = canonical_rollover_epoch_ranks_
        ? canonical_rollover_epoch_ranks_->size() * sizeof(std::uint16_t)
        : 0u;
    return rollover_rank_bytes +
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
        resident_plan_.size() *
            sizeof(gpulsmopt2_detail::ResidentPublicationPlan) +
        level_storage_spans_.size() *
            sizeof(gpulsmopt2_detail::LevelStorageSpan) +
        canonical_cell_ranks_.size() * sizeof(std::uint16_t) +
        canonical_cell_counts_.size() * sizeof(std::uint32_t) +
        canonical_job_prefixes_.size() *
            sizeof(gpulsmopt2_detail::CanonicalJobPrefix) +
        canonical_next_job_.size() * sizeof(std::uint32_t) +
        raw_keys_.size() * sizeof(std::uint32_t) +
        raw_payloads_.size() * sizeof(gpulsmopt2_detail::RawPayload) +
        raw_offsets_.size() * sizeof(std::uint32_t) +
        raw_signatures_.size() * sizeof(std::uint64_t) +
        raw_epoch_signatures_.size() * sizeof(std::uint64_t) +
        canonical_epoch_workspace_.size() *
            sizeof(gpulsmopt2_detail::RawPayload) +
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
        radix_storage_.size() * sizeof(std::uint8_t) +
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
  }

private:
  gpulsmopt2_detail::ResidentRows resident_rows() {
    return {arena_key_flags_.data(), arena_values_.data()};
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
    int blocks_per_sm = 0;
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

  void admit(const std::uint32_t *keys, const std::uint32_t *values,
             std::size_t count, bool tombstone, cudaStream_t stream) {
    if (!count) return;
    if (!keys || (!tombstone && !values))
      throw std::invalid_argument("invalid GPULSMOpt update batch");
    begin_operation(stream);
    std::size_t consumed = 0u;
    bool incomplete = false;
    while (consumed < count) {
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
      bool include_receipt, bool top_level_rollover) {
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
    const bool receipt_in_graph = launch_canonical_publication(stream);
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
  void ensure_radix_workspace(std::size_t count) {
    const std::size_t capacity = std::max(radix_id_capacity_, count);
    const std::size_t ids_bytes = aligned_id_bytes(capacity);
    const std::size_t query_offset_bytes =
        aligned_id_bytes(gpulsmopt2_detail::kQuotients + 1u);
    const std::size_t required = ids_bytes * 3u + query_offset_bytes;
    if (radix_storage_.size() < required) radix_storage_.resize(required);
    std::uint8_t *storage = radix_storage_.data();
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
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::ResidentPublicationPlan>
      resident_plan_;
  gpulsmopt2_detail::PinnedBuffer<
      gpulsmopt2_detail::ResidentPublicationPlan> publication_receipt_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::LevelStorageSpan>
      level_storage_spans_;
  gpulsmopt2_detail::Buffer<std::uint16_t> canonical_cell_ranks_;
  std::unique_ptr<gpulsmopt2_detail::Buffer<std::uint16_t>>
      canonical_rollover_epoch_ranks_;
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
  gpulsmopt2_detail::Buffer<std::uint8_t> radix_storage_;
  gpulsmopt2_detail::Buffer<std::uint32_t> radix_keys_, radix_ids_out_;

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
