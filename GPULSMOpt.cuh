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
#include <thrust/iterator/transform_output_iterator.h>

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
// Eight pivots form nine search regions.
constexpr std::uint32_t kGuideRegions = 9u;
constexpr std::uint32_t kGuideSamples = kGuideRegions - 1u;
constexpr std::size_t kGuideEntriesPerLevel =
    std::size_t{kQuotients} * kGuideSamples;
constexpr std::uint32_t kThreads = 256u;
constexpr std::uint32_t kRangeSchedulerBlocks = 256u;
constexpr std::uint32_t kSectionRangeThreads = 128u;
constexpr std::uint32_t kSectionRangeWarps =
    kSectionRangeThreads / 32u;
constexpr std::uint32_t kInvalid = 0xffffffffu;
constexpr std::uint32_t kInvalidAge = 0xffffffffu;
constexpr std::uint32_t kTombstone = 1u;
constexpr std::size_t kMaximumOperationTile = std::size_t{1} << 20u;
static_assert(kMaximumOperationTile <=
              (std::size_t{1} << kBatchPositionBits));
constexpr std::size_t kMaximumPublicationRows =
    std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kDescriptorOffsetBits = 47u;
constexpr std::uint64_t kDescriptorSplitFlag =
    std::uint64_t{1} << (kDescriptorOffsetBits - 1u);
constexpr std::uint64_t kDescriptorOffsetMask =
    kDescriptorSplitFlag - 1u;
static_assert(kMaximumPublicationRows < kDescriptorSplitFlag);
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
constexpr std::uint32_t kDenseCellRankMinimumRows = kFoundationCells;
// A full section ends at the descriptor count.
__host__ __device__ constexpr bool cell_rank_supported(
    std::uint64_t count) {
  return count <= kQuotients;
}
static_assert((kFoundationCells - 1u) * kFoundationCellKeys <=
              std::numeric_limits<std::uint16_t>::max());
constexpr std::uint32_t kCellOwnedQuotients = 2u;
constexpr std::uint32_t kCellOwnedCells =
    kCellOwnedQuotients * kFoundationCells;
constexpr std::uint32_t kCellOwnedWarpMaximum = 32u;
constexpr std::uint32_t kCellOwnedCostBuckets = 8u;
constexpr std::uint32_t kRankCellThreads = kFoundationCells;
constexpr std::uint32_t kRankCellMaximumSources = 4u;
constexpr std::uint32_t kRankCellTapeEntries = 9216u;
constexpr std::uint32_t kRankCellPaddingPerCell = 2u;
constexpr std::uint32_t kRankCellMinimumSectionRows = 3072u;
constexpr std::uint32_t kRankCellReferenceBits = 14u;
constexpr std::uint32_t kRankCellReferenceMask =
    (1u << kRankCellReferenceBits) - 1u;
constexpr std::size_t kRankCellDynamicSharedBytes =
    std::size_t{kRankCellTapeEntries} * sizeof(std::uint16_t);
constexpr std::uint32_t kPlanningTiles = 128u;
constexpr std::uint32_t kPlanningTileQuotients =
    kQuotients / kPlanningTiles;
constexpr std::uint32_t kMaximumMergeSources = kMaximumLevels + 1u;
constexpr std::uint32_t kBalancedMergeCapacityCeiling =
    kFoundationCompactionThreads * 32u;
#ifndef GPULSMOPT_CANONICAL_CARRY
#define GPULSMOPT_CANONICAL_CARRY 1
#endif
constexpr bool kCanonicalCarry = GPULSMOPT_CANONICAL_CARRY != 0;
#ifndef GPULSMOPT_CANONICAL_LOCAL_EPOCH
#define GPULSMOPT_CANONICAL_LOCAL_EPOCH 1
#endif
constexpr bool kCanonicalLocalEpoch =
    GPULSMOPT_CANONICAL_LOCAL_EPOCH != 0;
#ifndef GPULSMOPT_CANONICAL_TOURNAMENT_MERGE
#define GPULSMOPT_CANONICAL_TOURNAMENT_MERGE 1
#endif
constexpr bool kCanonicalTournamentMerge =
    GPULSMOPT_CANONICAL_TOURNAMENT_MERGE != 0;
#ifndef GPULSMOPT_CANONICAL_TOURNAMENT_MINIMUM_SOURCES
#define GPULSMOPT_CANONICAL_TOURNAMENT_MINIMUM_SOURCES 3
#endif
constexpr std::uint32_t kCanonicalTournamentMinimumSources =
    GPULSMOPT_CANONICAL_TOURNAMENT_MINIMUM_SOURCES;
static_assert(kCanonicalTournamentMinimumSources >= 3u &&
              kCanonicalTournamentMinimumSources <= kMaximumMergeSources);
#ifndef GPULSMOPT_CANONICAL_PUBLICATION_GRAPH
#define GPULSMOPT_CANONICAL_PUBLICATION_GRAPH 1
#endif
constexpr bool kCanonicalPublicationGraph =
    GPULSMOPT_CANONICAL_PUBLICATION_GRAPH != 0;
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
#ifndef GPULSMOPT_CANONICAL_COMPACT_MULTIWAY
#define GPULSMOPT_CANONICAL_COMPACT_MULTIWAY 1
#endif
constexpr bool kCanonicalCompactMultiway =
    GPULSMOPT_CANONICAL_COMPACT_MULTIWAY != 0;
using CanonicalTournamentReference = std::conditional_t<
    kCanonicalCompactMultiway, std::uint16_t, std::uint32_t>;
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

inline std::size_t canonical_tournament_layout_bytes(
    std::uint32_t capacity, std::uint32_t source_count,
    bool cached_sources) {
  const std::uint32_t leaves = canonical_next_power_of_two(source_count);
  std::size_t bytes = 0u;
  const auto reserve = [&bytes](std::size_t count, std::size_t item_bytes,
                                std::size_t alignment) {
    bytes = canonical_align_bytes(bytes, alignment);
    bytes += count * item_bytes;
  };
  const std::size_t states =
      std::size_t{kCanonicalTournamentChains} * source_count;
  if (cached_sources) {
    // One packed (remaining, position) cursor replaces the two 16-bit
    // arrays.  Cell-local positions live in unused head bits, so the compact
    // reference path no longer needs a per-chain slice-base array.
    reserve(states, sizeof(std::uint32_t), alignof(std::uint32_t));
  } else {
    reserve(states, sizeof(std::uint16_t), alignof(std::uint16_t));
    reserve(states, sizeof(std::uint16_t), alignof(std::uint16_t));
    if (kCanonicalCompactMultiway)
      reserve(states, sizeof(std::uint16_t), alignof(std::uint16_t));
  }
  reserve(states, sizeof(std::uint32_t), alignof(std::uint32_t));
  if (cached_sources) {
    const std::size_t source_quotients =
        std::size_t{kCanonicalJobQuotients} * source_count;
    reserve(source_quotients, sizeof(std::uint64_t),
            alignof(std::uint64_t));
    reserve(source_quotients, sizeof(std::uint32_t),
            alignof(std::uint32_t));
  }
  reserve(std::size_t{kCanonicalTournamentChains} * leaves,
          sizeof(std::uint8_t), alignof(std::uint8_t));
  reserve(capacity, sizeof(CanonicalTournamentReference),
          alignof(CanonicalTournamentReference));
  reserve(kCanonicalTournamentTasks, sizeof(std::uint16_t),
          alignof(std::uint16_t));
  reserve(kCanonicalTournamentTasks + 1u, sizeof(std::uint16_t),
          alignof(std::uint16_t));
  return canonical_align_bytes(bytes, 16u);
}

inline std::size_t canonical_tournament_dynamic_shared_bytes(
    std::uint32_t capacity, std::uint32_t source_count) {
  // Keep the old layout as a sizing floor.  The cache overlay therefore does
  // not silently enlarge jobs just because it repurposes shared memory.  A
  // larger occupancy-selected grid can still result from lower register use.
  return std::max(
      canonical_tournament_layout_bytes(capacity, source_count, false),
      canonical_tournament_layout_bytes(capacity, source_count, true));
}
#if defined(GPULSMOPT_FORCE_UNIFIED_MERGE)
constexpr bool kForceUnifiedMergeExperiment = true;
#else
constexpr bool kForceUnifiedMergeExperiment = false;
#endif
#if defined(GPULSMOPT_FORCE_UNIFIED_COMPILE_ELIDE)
static_assert(kForceUnifiedMergeExperiment,
              "compile-elided mode requires forced-unified mode");
constexpr bool kForceUnifiedCompileElision = true;
#else
constexpr bool kForceUnifiedCompileElision = false;
#endif
// Bound for splitting a maximally dense quotient.
constexpr std::uint64_t kResidentWorkFlag = std::uint64_t{1} << 63u;
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

__host__ __device__ __forceinline__ std::uint64_t resident_work_count(
    std::uint64_t encoded) {
  return encoded & ~kResidentWorkFlag;
}

__host__ __device__ __forceinline__ bool resident_work_present(
    std::uint64_t encoded) {
  return (encoded & kResidentWorkFlag) != 0u;
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

inline std::size_t foundation_pool_capacity(std::size_t requested) {
  constexpr std::size_t even_maximum = kMaximumPublicationRows - 1u;
  // Two banks reserve worst-case raw rows.
  const std::size_t requested_banks = requested > even_maximum / 4u
      ? even_maximum : requested * 4u;
  const std::size_t capacity = std::max<std::size_t>(
      requested_banks, std::size_t{kQuotients} * 1024u);
  return std::min(even_maximum, (capacity + 1u) & ~std::size_t{1u});
}

inline std::size_t maximum_resident_merge_jobs(
    std::size_t maximum_raw_rows, std::uint32_t merge_capacity) {
  // Reserve one job per section plus hot pieces.
  const std::size_t safe =
      merge_capacity - (kMaximumMergeSources - 1u);
  return std::size_t{kQuotients} +
      (maximum_raw_rows + safe - 1u) / safe + 1u;
}

inline std::size_t adaptive_route_stride(
    std::size_t maximum_raw_rows, std::uint32_t merge_capacity) {
  // Routes cover normal and oversized sections.
  return std::size_t{kQuotients} +
      maximum_resident_merge_jobs(maximum_raw_rows, merge_capacity);
}

inline std::size_t preassigned_level_pool_capacity(
    std::size_t maximum_raw_rows, std::size_t epoch_capacity) {
  std::size_t result = 0u;
  std::size_t capacity = std::min(maximum_raw_rows, epoch_capacity);
  for (std::uint32_t level = 0u; level < kMaximumLevels; ++level) {
    if (result > std::numeric_limits<std::size_t>::max() - capacity)
      throw std::bad_alloc();
    result += capacity;
    if (capacity == maximum_raw_rows) break;
    capacity = capacity > maximum_raw_rows / 2u
        ? maximum_raw_rows : capacity * 2u;
  }
  return result;
}

inline std::size_t preassigned_level_rank_blocks(
    std::size_t maximum_raw_rows, std::size_t epoch_capacity) {
  std::size_t result = 0u;
  std::size_t capacity = std::min(maximum_raw_rows, epoch_capacity);
  for (std::uint32_t level = 0u; level < kMaximumLevels; ++level) {
    const std::size_t blocks = std::min<std::size_t>(
        kQuotients,
        (capacity + kDenseCellRankMinimumRows - 1u) /
            kDenseCellRankMinimumRows);
    if (result > std::numeric_limits<std::size_t>::max() - blocks)
      throw std::bad_alloc();
    result += blocks;
    if (capacity == maximum_raw_rows) break;
    capacity = capacity > maximum_raw_rows / 2u
        ? maximum_raw_rows : capacity * 2u;
  }
  return result;
}

inline std::uint32_t canonical_level_count(
    std::size_t maximum_raw_rows, std::size_t epoch_capacity) {
  std::uint32_t levels = 1u;
  std::size_t capacity = std::min(maximum_raw_rows, epoch_capacity);
  while (capacity < maximum_raw_rows && levels < kMaximumLevels) {
    capacity = capacity > maximum_raw_rows / 2u
        ? maximum_raw_rows : capacity * 2u;
    ++levels;
  }
  return levels;
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

// Candidate tokens exclude values and tombstones.
using CandidateToken = std::uint32_t;

__host__ __device__ __forceinline__ CandidateToken make_candidate_token(
    std::uint32_t local_quotient, std::uint32_t suffix,
    std::uint32_t source_age) {
  return (local_quotient << (16u + kMergeSourceBits)) |
      ((suffix & 0xffffu) << kMergeSourceBits) |
      (source_age & ((1u << kMergeSourceBits) - 1u));
}

__host__ __device__ __forceinline__ std::uint16_t candidate_token_key(
    CandidateToken token) {
  return static_cast<std::uint16_t>(token >> kMergeSourceBits);
}

__host__ __device__ __forceinline__ std::uint32_t
candidate_token_local_quotient(CandidateToken token) {
  return token >> (16u + kMergeSourceBits);
}

__host__ __device__ __forceinline__ std::uint32_t candidate_token_source(
    CandidateToken token) {
  return token & ((1u << kMergeSourceBits) - 1u);
}

__host__ __device__ __forceinline__ std::uint32_t
candidate_token_logical_key(CandidateToken token) {
  return token >> kMergeSourceBits;
}

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

struct RawPayload {
  std::uint32_t value;
  std::uint32_t metadata;
};

static_assert(sizeof(Row) == 8u);
static_assert(sizeof(RawAssignment) == 12u);
static_assert(sizeof(RawPayload) == 8u);

inline std::size_t balanced_merge_dynamic_shared_bytes(
    std::uint32_t capacity) {
  // Cell planes share the merge workspace.
  constexpr std::size_t cell_words =
      (kCellOwnedCells + 2u) + kCellOwnedCells +
      kCellOwnedCells + kCellOwnedCells + (kCellOwnedCells + 1u);
  const std::size_t tombstone_bytes =
      std::size_t{(capacity + 31u) / 32u} * sizeof(std::uint32_t);
  const std::size_t cell_bytes = cell_words * sizeof(std::uint16_t);
  return std::size_t{capacity} * sizeof(CandidateToken) +
      std::size_t{capacity + 1u} * sizeof(std::uint16_t) * 2u +
      tombstone_bytes + cell_bytes;
}

constexpr std::uint32_t kRawTombstone = 0x80000000u;


__host__ __device__ __forceinline__ RawPayload make_raw_payload(
    std::uint32_t value, std::uint32_t logical_position, bool tombstone) {
  return {tombstone ? 0u : value,
          logical_position | (tombstone ? kRawTombstone : 0u)};
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
  __host__ __device__ static Descriptor make_split(std::uint32_t count) {
    return {kDescriptorSplitFlag |
            (std::uint64_t{count} << kDescriptorOffsetBits)};
  }
  __host__ __device__ std::uint64_t offset() const {
    return bits & kDescriptorOffsetMask;
  }
  __host__ __device__ std::uint32_t count() const {
    return static_cast<std::uint32_t>(bits >> kDescriptorOffsetBits);
  }
  __host__ __device__ bool split() const {
    return (bits & kDescriptorSplitFlag) != 0u;
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

// Dense sections store exact cell boundaries.
struct LevelRankSpan {
  std::uint64_t begin_block{};
  std::uint32_t capacity_blocks{};
};

enum ResidentPublicationStatus : std::uint32_t {
  kPublicationSuccess = 0u,
  kPublicationJobOverflow = 1u << 0u,
  kPublicationRouteOverflow = 1u << 1u,
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
  std::uint32_t destination_is_foundation{};
  std::uint32_t keep_tombstones{};
  std::uint32_t output_generation{};
  std::uint32_t job_count{};
  std::uint32_t route_count{};
  std::uint32_t rank_block_count{};
  std::uint32_t status{};
  std::uint32_t job_capacity{};
  std::uint64_t output_begin{};
  std::uint64_t output_capacity{};
  std::uint64_t raw_reservation{};
  std::uint64_t survivor_count{};
};

static_assert(sizeof(ResidentPublicationPlan) == 88u);

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

__device__ __forceinline__ Descriptor routed_descriptor_for_suffix(
    std::uint32_t q, std::uint32_t level, std::uint32_t suffix,
    const RouteHeader *route_headers, const RouteSlice *route_slices) {
  const RouteHeader header = route_headers[descriptor_index(q, level)];
  std::uint32_t low = 0u, high = header.count;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (route_slices[header.begin + middle].suffix_end <= suffix)
      low = middle + 1u;
    else
      high = middle;
  }
  if (low == header.count) return {};
  const RouteSlice route = route_slices[header.begin + low];
  return suffix >= route.suffix_begin && suffix < route.suffix_end
      ? route.rows : Descriptor{};
}

struct RoutedSliceSelection {
  Descriptor rows{};
  std::uint32_t route{};
  bool valid{};
};

__device__ __forceinline__ RoutedSliceSelection routed_slice_for_suffix(
    std::uint32_t q, std::uint32_t level, std::uint32_t suffix,
    const RouteHeader *route_headers, const RouteSlice *route_slices) {
  const RouteHeader header = route_headers[descriptor_index(q, level)];
  std::uint32_t low = 0u, high = header.count;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (route_slices[header.begin + middle].suffix_end <= suffix)
      low = middle + 1u;
    else
      high = middle;
  }
  if (low == header.count) return {};
  const std::uint32_t route = header.begin + low;
  const RouteSlice slice = route_slices[route];
  return suffix >= slice.suffix_begin && suffix < slice.suffix_end
      ? RoutedSliceSelection{slice.rows, route, true}
      : RoutedSliceSelection{};
}

__host__ __device__ __forceinline__ std::size_t guide_index(
    std::uint32_t q, std::uint32_t level) {
  return std::size_t{level} * kGuideEntriesPerLevel +
      std::size_t{q} * kGuideSamples;
}

struct CellInputSlice {
  std::uint32_t begin{};
  std::uint32_t count{};
};

__device__ __forceinline__ bool exact_cell_input_slice(
    std::uint32_t q, std::uint32_t level, std::uint32_t cell,
    std::uint32_t foundation_level, std::uint32_t section_count,
    const std::uint16_t *local_rank,
    const std::uint32_t *level_cell_rank_blocks,
    const std::uint16_t *level_cell_ranks, CellInputSlice &slice) {
  if (!cell_rank_supported(section_count)) return false;
  const std::uint16_t *ranks = nullptr;
  if (level == foundation_level) {
    ranks = local_rank + std::size_t{q} * kFoundationCells;
  } else {
    const std::uint32_t block =
        level_cell_rank_blocks[descriptor_index(q, level)];
    if (block == kInvalid) return false;
    ranks = level_cell_ranks +
        std::size_t{block} * kFoundationCells;
  }
  const std::uint32_t begin = ranks[cell];
  const std::uint32_t end = cell + 1u < kFoundationCells
      ? ranks[cell + 1u] : section_count;
  slice = {begin, end - begin};
  return true;
}

__device__ __forceinline__ CellInputSlice resident_candidate_cell_slice(
    std::uint32_t q, std::uint32_t level, std::uint32_t cell,
    std::uint32_t foundation_level, std::uint32_t section_count,
    std::uint32_t candidate_begin, const CandidateToken *candidate_tokens,
    const std::uint16_t *local_rank,
    const std::uint32_t *level_cell_rank_blocks,
    const std::uint16_t *level_cell_ranks) {
  CellInputSlice slice{};
  if (!exact_cell_input_slice(
          q, level, cell, foundation_level, section_count, local_rank,
          level_cell_rank_blocks, level_cell_ranks, slice)) {
    const std::uint32_t low_suffix = cell * kFoundationCellKeys;
    const std::uint32_t high_suffix = low_suffix + kFoundationCellKeys;
    std::uint32_t low = 0u, high = section_count;
    while (low < high) {
      const std::uint32_t middle = (low + high) >> 1u;
      if (candidate_token_key(candidate_tokens[candidate_begin + middle]) <
          low_suffix)
        low = middle + 1u;
      else
        high = middle;
    }
    const std::uint32_t begin = low;
    high = section_count;
    while (low < high) {
      const std::uint32_t middle = (low + high) >> 1u;
      if (candidate_token_key(candidate_tokens[candidate_begin + middle]) <
          high_suffix)
        low = middle + 1u;
      else
        high = middle;
    }
    slice = {begin, low - begin};
  }
  slice.begin += candidate_begin;
  return slice;
}

struct TaggedRow {
  Row row;
  std::uint32_t age;
};

static_assert(sizeof(TaggedRow) == 12u);

struct NewestAssignment {
  __host__ __device__ RawAssignment operator()(
      const RawAssignment &first, const RawAssignment &second) const {
    return raw_position(second) > raw_position(first)
        ? second : first;
  }
};

struct NewestPayload {
  __host__ __device__ RawPayload operator()(
      const RawPayload &first, const RawPayload &second) const {
    return raw_position(second) > raw_position(first) ? second : first;
  }
};

struct AssignmentRow {
  __host__ __device__ Row operator()(const RawAssignment &assignment) const {
    return raw_row(assignment);
  }
};

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
  std::uint64_t existing_offset{};
  std::uint32_t quotient_begin;
  std::uint32_t quotient_end;
  std::uint32_t existing_capacity{};
  std::uint32_t output_count{};
  std::uint32_t route_ordinal{};
  std::uint16_t hot_piece{};
  std::uint16_t hot_pieces{};
};

static_assert(sizeof(BalancedMergeJob) == 48u);

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

// Split routes use leftmost-match semantics.
__device__ __forceinline__ bool find_leftmost_point_row(
    const Row *rows, std::uint32_t count, std::uint32_t key, Row &result) {
  std::uint32_t lo = 0u, hi = count;
  bool matched = false;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    const Row candidate = rows[mid];
    if (candidate.key < key) lo = mid + 1u;
    else {
      hi = mid;
      if (candidate.key == key) {
        result = candidate;
        matched = true;
      }
    }
  }
  return matched;
}

__device__ __forceinline__ bool find_leftmost_point_row(
    ResidentRows rows, std::uint32_t count, std::uint32_t key, Row &result) {
  std::uint32_t lo = 0u, hi = count;
  bool matched = false;
  std::uint32_t matched_position = 0u;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    const std::uint16_t candidate = rows.key_at(mid);
    if (candidate < key) lo = mid + 1u;
    else {
      hi = mid;
      if (candidate == key) {
        matched_position = mid;
        matched = true;
      }
    }
  }
  if (matched) result = rows[matched_position];
  return matched;
}

__device__ __forceinline__ void guide_search_bounds(
    const std::uint16_t *guides, std::uint32_t q,
    std::uint32_t level, std::uint32_t count, std::uint32_t key,
    std::uint32_t &begin, std::uint32_t &end) {
  if (count < kGuideRegions) {
    begin = 0u;
    end = count;
    return;
  }
  const std::uint16_t *samples = guides + guide_index(q, level);
  std::uint32_t lo = 0u, hi = kGuideSamples;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    if (samples[mid] <= key) lo = mid + 1u;
    else hi = mid;
  }
  begin = lo * count / kGuideRegions;
  end = (lo + 1u) * count / kGuideRegions;
}

__device__ __forceinline__ void resident_point_search_bounds(
    std::uint32_t q, std::uint32_t level, std::uint32_t suffix,
    std::uint32_t foundation_level, const Descriptor &selected_rows,
    std::uint32_t selected_route, std::uint32_t route_count,
    std::uint32_t logical_count,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    const std::uint16_t *local_rank, const std::uint16_t *level_guides,
    const std::uint32_t *level_cell_rank_blocks,
    const std::uint16_t *level_cell_ranks,
    std::uint32_t &begin, std::uint32_t &end) {
  const std::uint32_t cell = suffix / kFoundationCellKeys;
  CellInputSlice logical{};
  const bool ranked = exact_cell_input_slice(
      q, level, cell, foundation_level, logical_count, local_rank,
      level_cell_rank_blocks, level_cell_ranks, logical);
  if (!ranked) {
    guide_search_bounds(level_guides, q, level, logical_count, suffix,
                        logical.begin, logical.count);
    logical.count -= logical.begin;
  }
  const std::uint32_t logical_begin = logical.begin;
  const std::uint32_t logical_end = logical.begin + logical.count;
  if (route_count == 1u) {
    begin = logical_begin;
    end = min(selected_rows.count(), logical_end);
    return;
  }
  const std::uint32_t section_begin = level_q_logical_offsets[
      std::size_t{level} * (kQuotients + 1u) + q];
  const std::uint32_t route_begin =
      route_logical_begins[selected_route] - section_begin;
  begin = logical_begin > route_begin ? logical_begin - route_begin : 0u;
  end = min(selected_rows.count(), logical_end > route_begin
      ? logical_end - route_begin : 0u);
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
__device__ __forceinline__ std::uint32_t logical_section_bound(
    std::uint32_t q, std::uint32_t level, std::uint32_t key, bool upper,
    ResidentRows arena, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets) {
  const RouteHeader header = route_headers[descriptor_index(q, level)];
  if (header.count == 1u) {
    const Descriptor rows = route_slices[header.begin].rows;
    return upper
        ? upper_bound_rows(arena + rows.offset(), rows.count(), key)
        : lower_bound_rows(arena + rows.offset(), rows.count(), key);
  }
  const std::size_t q_index =
      std::size_t{level} * (kQuotients + 1u) + q;
  const std::uint32_t section_begin = level_q_logical_offsets[q_index];
  if (!header.count) return 0u;

  std::uint32_t low = 0u, high = header.count;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (route_slices[header.begin + middle].suffix_end <= key)
      low = middle + 1u;
    else
      high = middle;
  }
  if (low == header.count)
    return level_q_logical_offsets[q_index + 1u] - section_begin;

  const std::uint32_t route = header.begin + low;
  const RouteSlice slice = route_slices[route];
  const std::uint32_t preceding =
      route_logical_begins[route] - section_begin;
  if (key < slice.suffix_begin) return preceding;
  const ResidentRows rows = arena + slice.rows.offset();
  return preceding + (upper
      ? upper_bound_rows(rows, slice.rows.count(), key)
      : lower_bound_rows(rows, slice.rows.count(), key));
}

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

__global__ void scatter_admission_records_kernel(
    const std::uint32_t *keys, const std::uint32_t *values,
    std::uint32_t count, std::uint32_t batch_slot, bool tombstone,
    const std::uint32_t *offsets, const std::uint32_t *reservation_ranks,
    std::uint32_t *destination_keys, RawPayload *destination_payloads) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t key = keys[i];
  const std::uint32_t quotient = key >> 16u;
  const std::uint32_t output =
      offsets[quotient] + reservation_ranks[i];
  destination_keys[output] = key;
  destination_payloads[output] = make_raw_payload(
      tombstone ? 0u : values[i],
      (batch_slot << kBatchPositionBits) | i, tombstone);
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

__global__ void pack_publication_epoch_kernel(
    const std::uint32_t *pending_keys, const RawPayload *pending_payloads,
    std::uint32_t batch_stride,
    const std::uint32_t *batch_offsets,
    std::uint32_t *keys, RawAssignment *output_assignments) {
  const std::uint32_t batch = blockIdx.y;
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count =
      batch_offsets[batch + 1u] - batch_offsets[batch];
  if (position >= count) return;
  const std::uint32_t source = batch * batch_stride + position;
  const std::uint32_t output = batch_offsets[batch] + position;
  const RawPayload payload = pending_payloads[source];
  const std::uint32_t key = pending_keys[source];
  keys[output] = key;
  output_assignments[output] =
      {key, payload.value, payload.metadata};
}

__global__ void pack_canonical_epoch_kernel(
    const std::uint32_t *pending_keys, const RawPayload *pending_payloads,
    std::uint32_t batch_stride,
    const std::uint32_t *batch_offsets,
    std::uint32_t *keys, RawPayload *payloads) {
  const std::uint32_t batch = blockIdx.y;
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count =
      batch_offsets[batch + 1u] - batch_offsets[batch];
  if (position >= count) return;
  const std::uint32_t source = batch * batch_stride + position;
  const std::uint32_t output = batch_offsets[batch] + position;
  keys[output] = pending_keys[source];
  payloads[output] = pending_payloads[source];
}

// Pad captured sorts with the first assignment.
__global__ void pad_publication_epoch_kernel(
    std::uint32_t epoch_capacity, const std::uint32_t *batch_offsets,
    std::uint32_t *keys, RawAssignment *assignments) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count = batch_offsets[kBatchesPerEpoch];
  if (i < count || i >= epoch_capacity || !count) return;
  keys[i] = keys[0];
  assignments[i] = assignments[0];
}

__global__ void pad_canonical_epoch_kernel(
    std::uint32_t epoch_capacity, const std::uint32_t *batch_offsets,
    std::uint32_t *keys, RawPayload *payloads) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count = batch_offsets[kBatchesPerEpoch];
  if (i < count || i >= epoch_capacity || !count) return;
  keys[i] = keys[0];
  payloads[i] = payloads[0];
}

__global__ void materialize_canonical_epoch_rows_kernel(
    const std::uint32_t *keys, const RawPayload *payloads,
    const std::uint32_t *count, Row *rows) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < *count) rows[i] = raw_row(keys[i], payloads[i]);
}

__global__ void materialize_canonical_epoch_resident_kernel(
    const std::uint32_t *keys, const RawPayload *payloads,
    const std::uint32_t *count, ResidentRows rows,
    std::uint64_t destination) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < *count)
    rows.store(destination + i, raw_row(keys[i], payloads[i]));
}

__device__ __forceinline__ std::uint32_t lower_bound_full_keys(
    const std::uint32_t *keys, std::uint32_t count, std::uint32_t target) {
  std::uint32_t low = 0u, high = count;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (keys[middle] < target) low = middle + 1u;
    else high = middle;
  }
  return low;
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

__global__ void store_resident_rows_kernel(
    const Row *source, std::uint32_t count, ResidentRows destination,
    std::uint64_t destination_begin = 0u) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) destination.store(destination_begin + i, source[i]);
}

__global__ void publish_foundation_build_kernel(
    const std::uint32_t *section_offsets, std::uint32_t level,
    Descriptor *descriptors) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  const std::uint32_t count = section_offsets[q + 1u] - section_offsets[q];
  descriptors[descriptor_index(q, level)] = count
      ? Descriptor::make(section_offsets[q], count) : Descriptor{};
}


__global__ void count_resident_merge_work_kernel(
    const std::uint32_t *current_offsets, const Descriptor *descriptors,
    const DeviceManifest *manifests, const std::uint32_t *active_manifest,
    const ResidentPublicationPlan *plan, std::uint64_t *raw_counts) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients || plan->status) return;
  const DeviceManifestSnapshot manifest = load_active_manifest(
      manifests, active_manifest);
  std::uint64_t count =
      current_offsets[q + 1u] - current_offsets[q];
  bool has_resident = false;
  for (std::uint32_t level = 0u;
       level <= plan->source_level_limit; ++level)
    if (level_is_occupied(manifest.occupied_level_mask, level)) {
      has_resident |= descriptors[descriptor_index(q, level)].count() != 0u;
      count += descriptors[descriptor_index(q, level)].count();
    }
  raw_counts[q] = count | (has_resident ? kResidentWorkFlag : 0u);
}

__global__ void build_staged_rank_directory_kernel(
    const Row *rows, const std::uint32_t *offsets,
    std::uint16_t *ranks) {
  const std::uint32_t q = blockIdx.x;
  const std::uint32_t cell = threadIdx.x;
  if (q >= kQuotients || cell >= kFoundationCells) return;
  const std::uint32_t begin = offsets[q];
  const std::uint32_t count = offsets[q + 1u] - begin;
  const std::uint32_t position = cell_rank_supported(count)
      ? lower_bound_rows(rows + begin, count, cell * kFoundationCellKeys)
      : 0u;
  ranks[std::size_t{q} * kFoundationCells + cell] =
      static_cast<std::uint16_t>(position);
}

__global__ void resolve_epoch_sections_kernel(
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches,
    const std::uint32_t *raw_section_offsets,
    std::uint32_t *scratch_keys, Row *scratch_rows,
    std::uint32_t *resolved_counts, std::uint16_t *resolved_ranks) {
  constexpr std::uint32_t kThreads = kFoundationCompactionThreads;
  constexpr std::uint32_t kItems = kLocalEpochItemsPerThread;
  using BlockSort = cub::BlockRadixSort<
      std::uint32_t, kThreads, kItems, std::uint32_t>;
  using BlockScan = cub::BlockScan<std::uint32_t, kThreads>;
  union ResolutionStorage {
    typename BlockSort::TempStorage sort;
    unsigned long long winners[kLocalEpochCapacity];
  };
  __shared__ ResolutionStorage resolution_storage;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::uint32_t batch_prefix[kBatchesPerEpoch + 1u];
  __shared__ std::uint16_t sorted_suffixes[kLocalEpochCapacity];
  __shared__ std::uint32_t cell_counts[kFoundationCells];
  const std::uint32_t q = blockIdx.x;
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
  }
  __syncthreads();
  const std::uint32_t raw_count = batch_prefix[pending_batches];
  if (!raw_count) {
    if (threadIdx.x == 0u) resolved_counts[q] = 0u;
    if (threadIdx.x < kFoundationCells)
      resolved_ranks[std::size_t{q} * kFoundationCells + threadIdx.x] = 0u;
    return;
  }

  std::uint32_t sort_keys[kItems];
  std::uint32_t sort_sources[kItems];
  constexpr std::uint32_t kInvalidSortKey = 1u << 16u;
  for (std::uint32_t item = 0u; item < kItems; ++item) {
    const std::uint32_t local = threadIdx.x * kItems + item;
    sort_keys[item] = kInvalidSortKey;
    sort_sources[item] = 0u;
    if (local >= raw_count) continue;
    std::uint32_t batch = 0u;
    while (batch + 1u < pending_batches &&
           local >= batch_prefix[batch + 1u])
      ++batch;
    const std::size_t base =
        std::size_t{batch} * (kQuotients + 1u) + q;
    const std::uint32_t source = static_cast<std::uint32_t>(
        std::size_t{batch} * batch_stride + raw_offsets[base] +
        local - batch_prefix[batch]);
    sort_keys[item] = key_suffix(raw_keys[source]);
    sort_sources[item] = source;
  }
  BlockSort(resolution_storage.sort).Sort(
      sort_keys, sort_sources, 0, 17);
  for (std::uint32_t item = 0u; item < kItems; ++item) {
    const std::uint32_t local = threadIdx.x * kItems + item;
    sorted_suffixes[local] = static_cast<std::uint16_t>(sort_keys[item]);
  }
  if (threadIdx.x < kFoundationCells)
    cell_counts[threadIdx.x] = 0u;
  __syncthreads();

  bool leader[kItems];
  std::uint32_t leader_count = 0u;
  for (std::uint32_t item = 0u; item < kItems; ++item) {
    const std::uint32_t local = threadIdx.x * kItems + item;
    const bool valid = sort_keys[item] != kInvalidSortKey;
    leader[item] = valid &&
        (!local || sorted_suffixes[local - 1u] != sorted_suffixes[local]);
    leader_count += leader[item];
  }
  std::uint32_t output_prefix = 0u, output_count = 0u;
  BlockScan(scan_storage).ExclusiveSum(
      leader_count, output_prefix, output_count);
  for (std::uint32_t group = threadIdx.x; group < output_count;
       group += blockDim.x)
    resolution_storage.winners[group] = 0u;
  __syncthreads();

  std::uint32_t leaders_seen = 0u;
  for (std::uint32_t item = 0u; item < kItems; ++item) {
    if (sort_keys[item] == kInvalidSortKey) continue;
    const std::uint32_t group = leader[item]
        ? output_prefix + leaders_seen
        : output_prefix + leaders_seen - 1u;
    if (leader[item]) {
      sorted_suffixes[group] = static_cast<std::uint16_t>(sort_keys[item]);
      ++leaders_seen;
    }
    const std::uint32_t source = sort_sources[item];
    const std::uint32_t age = raw_position(raw_payloads[source]);
    const unsigned long long token =
        (static_cast<unsigned long long>(age + 1u) << 32u) | source;
    atomicMax(resolution_storage.winners + group, token);
  }
  if (threadIdx.x == 0u) resolved_counts[q] = output_count;
  __syncthreads();

  const std::uint32_t scratch_begin = raw_section_offsets[q];
  for (std::uint32_t group = threadIdx.x; group < output_count;
       group += blockDim.x) {
    const std::uint32_t source = static_cast<std::uint32_t>(
        resolution_storage.winners[group]);
    const std::uint32_t suffix = sorted_suffixes[group];
    scratch_keys[scratch_begin + group] = full_key(q, suffix);
    scratch_rows[scratch_begin + group] =
        raw_row(full_key(q, suffix), raw_payloads[source]);
    atomicAdd(cell_counts + suffix / kFoundationCellKeys, 1u);
  }
  __syncthreads();

  const std::uint32_t cell_count = threadIdx.x < kFoundationCells
      ? cell_counts[threadIdx.x] : 0u;
  std::uint32_t cell_prefix = 0u, ignored = 0u;
  BlockScan(scan_storage).ExclusiveSum(
      cell_count, cell_prefix, ignored);
  if (threadIdx.x < kFoundationCells)
    resolved_ranks[std::size_t{q} * kFoundationCells + threadIdx.x] =
        static_cast<std::uint16_t>(cell_prefix);
}

__global__ void compact_resolved_epoch_sections_kernel(
    const std::uint32_t *scratch_keys, const Row *scratch_rows,
    const std::uint32_t *raw_section_offsets,
    const std::uint32_t *resolved_offsets,
    std::uint32_t *keys, Row *rows) {
  const std::uint32_t q = blockIdx.x;
  const std::uint32_t count =
      resolved_offsets[q + 1u] - resolved_offsets[q];
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    keys[resolved_offsets[q] + i] = scratch_keys[raw_section_offsets[q] + i];
    rows[resolved_offsets[q] + i] = scratch_rows[raw_section_offsets[q] + i];
  }
}

__global__ void count_canonical_epoch_sections_kernel(
    const std::uint32_t *raw_offsets, std::uint32_t pending_batches,
    std::uint32_t *section_counts, std::uint32_t *overflow_quotients,
    std::uint32_t *overflow_count) {
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
  if (count > kLocalEpochCapacity) {
    const std::uint32_t position = atomicAdd(overflow_count, 1u);
    overflow_quotients[position] = q;
  }
}

__global__ void canonical_section_counts_from_offsets_kernel(
    const std::uint32_t *section_offsets, std::uint32_t *section_counts) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  section_counts[q] = q < kQuotients
      ? section_offsets[q + 1u] - section_offsets[q] : 0u;
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

template <bool ResidentOutput>
__global__ void resolve_canonical_epoch_local_kernel(
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const std::uint32_t *raw_section_offsets,
    Row *output_rows, ResidentRows resident_rows,
    std::uint64_t resident_begin, std::uint32_t *resolved_counts,
    std::uint16_t *cell_ranks) {
  constexpr std::uint32_t kThreads = kFoundationCompactionThreads;
  constexpr std::uint32_t kItems = kLocalEpochItemsPerThread;
  using BlockSort = cub::BlockRadixSort<
      std::uint32_t, kThreads, kItems, std::uint32_t>;
  using BlockScan = cub::BlockScan<std::uint32_t, kThreads>;
  union ResolutionStorage {
    typename BlockSort::TempStorage sort;
    unsigned long long winners[kLocalEpochCapacity];
  };
  __shared__ ResolutionStorage resolution_storage;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::uint32_t batch_prefix[kBatchesPerEpoch + 1u];
  __shared__ std::uint16_t sorted_suffixes[kLocalEpochCapacity];
  __shared__ std::uint32_t cell_counts[kFoundationCells];
  __shared__ std::uint32_t output_count_shared;
  const std::uint32_t q = blockIdx.x;
  if (q >= kQuotients) return;

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
  const std::uint32_t raw_count = batch_prefix[pending_batches];
  if (!raw_count) {
    if (threadIdx.x == 0u) resolved_counts[q] = 0u;
    if (threadIdx.x < kFoundationCells)
      cell_ranks[std::size_t{q} * kFoundationCells + threadIdx.x] = 0u;
    return;
  }
  if (raw_count > kLocalEpochCapacity) {
    if (threadIdx.x == 0u) resolved_counts[q] = kInvalid;
    if (threadIdx.x < kFoundationCells)
      cell_ranks[std::size_t{q} * kFoundationCells + threadIdx.x] = 0u;
    return;
  }

  std::uint32_t sort_keys[kItems];
  std::uint32_t sort_sources[kItems];
  constexpr std::uint32_t kInvalidSortKey = 1u << 16u;
  for (std::uint32_t item = 0u; item < kItems; ++item) {
    const std::uint32_t local = threadIdx.x * kItems + item;
    sort_keys[item] = kInvalidSortKey;
    sort_sources[item] = 0u;
    if (local >= raw_count) continue;
    std::uint32_t batch = 0u;
    while (batch + 1u < pending_batches &&
           local >= batch_prefix[batch + 1u])
      ++batch;
    const std::size_t base =
        std::size_t{batch} * (kQuotients + 1u) + q;
    const std::uint32_t source = static_cast<std::uint32_t>(
        std::size_t{batch} * batch_stride + raw_offsets[base] +
        local - batch_prefix[batch]);
    sort_keys[item] = key_suffix(raw_keys[source]);
    sort_sources[item] = source;
  }
  BlockSort(resolution_storage.sort).Sort(
      sort_keys, sort_sources, 0, 17);
  for (std::uint32_t item = 0u; item < kItems; ++item) {
    const std::uint32_t local = threadIdx.x * kItems + item;
    sorted_suffixes[local] = static_cast<std::uint16_t>(sort_keys[item]);
  }
  __syncthreads();

  bool leaders[kItems];
  std::uint32_t leader_count = 0u;
  for (std::uint32_t item = 0u; item < kItems; ++item) {
    const std::uint32_t local = threadIdx.x * kItems + item;
    const bool valid = sort_keys[item] != kInvalidSortKey;
    leaders[item] = valid &&
        (!local || sorted_suffixes[local - 1u] != sorted_suffixes[local]);
    leader_count += leaders[item];
  }
  std::uint32_t output_prefix = 0u, output_count = 0u;
  BlockScan(scan_storage).ExclusiveSum(
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
  for (std::uint32_t item = 0u; item < kItems; ++item) {
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
  BlockScan(scan_storage).ExclusiveSum(
      cell_count, cell_prefix, ignored);
  if (threadIdx.x < kFoundationCells)
    cell_ranks[std::size_t{q} * kFoundationCells + threadIdx.x] =
        static_cast<std::uint16_t>(cell_prefix);
}

template <bool ResidentOutput>
__global__ void resolve_canonical_epoch_oversized_kernel(
    const std::uint32_t *overflow_quotients,
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
          ? overflow_quotients[ticket] : kInvalid;
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

__global__ void set_resolved_epoch_count_kernel(
    const std::uint32_t *resolved_offsets, std::uint32_t *selected_count) {
  if (!blockIdx.x && !threadIdx.x)
    *selected_count = resolved_offsets[kQuotients];
}

// Count grouped raw intervals without suffix sorting.
__global__ void count_direct_epoch_merge_work_kernel(
    const std::uint32_t *raw_offsets, std::uint32_t pending_batches,
    const Descriptor *descriptors, const DeviceManifest *manifests,
    const std::uint32_t *active_manifest, ResidentPublicationPlan *plan,
    std::uint64_t *raw_counts, std::uint32_t *raw_section_counts,
    std::uint32_t *crowded_flag,
    std::uint32_t *local_epoch_overflow_flag) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients || plan->status) return;
  const DeviceManifestSnapshot manifest = load_active_manifest(
      manifests, active_manifest);
  std::uint64_t count = 0u;
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t offset =
        std::size_t{batch} * (kQuotients + 1u) + q;
    count += raw_offsets[offset + 1u] - raw_offsets[offset];
  }
  const std::uint32_t raw_count = static_cast<std::uint32_t>(count);
  raw_section_counts[q] = raw_count;
  if (raw_count > kLocalEpochCapacity)
    atomicExch(local_epoch_overflow_flag, 1u);
  bool has_resident = false;
  for (std::uint32_t level = 0u;
       level <= plan->source_level_limit; ++level)
    if (level_is_occupied(manifest.occupied_level_mask, level)) {
      has_resident |= descriptors[descriptor_index(q, level)].count() != 0u;
      count += descriptors[descriptor_index(q, level)].count();
    }

  raw_counts[q] = count | (has_resident ? kResidentWorkFlag : 0u);
  if (count > plan->job_capacity) atomicExch(crowded_flag, 1u);
}

__global__ void count_direct_epoch_records_kernel(
    const std::uint32_t *raw_offsets, std::uint32_t pending_batches,
    std::uint32_t *count) {
  if (blockIdx.x || threadIdx.x) return;
  std::uint32_t total = 0u;
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch)
    total += raw_offsets[
        std::size_t{batch} * (kQuotients + 1u) + kQuotients];
  *count = total;
}

__global__ void build_publication_batch_offsets_kernel(
    const std::uint32_t *raw_offsets, std::uint32_t *batch_offsets) {
  if (blockIdx.x || threadIdx.x) return;
  std::uint32_t total = 0u;
  batch_offsets[0] = 0u;
  for (std::uint32_t batch = 0u; batch < kBatchesPerEpoch; ++batch) {
    total += raw_offsets[
        std::size_t{batch} * (kQuotients + 1u) + kQuotients];
    batch_offsets[batch + 1u] = total;
  }
}

__global__ void choose_crowded_epoch_path_kernel(
    cudaGraphConditionalHandle conditional,
    const std::uint32_t *crowded_flag) {
  if (blockIdx.x || threadIdx.x) return;
  const std::uint32_t crowded =
      atomicAdd(const_cast<std::uint32_t *>(crowded_flag), 0u);
  cudaGraphSetConditional(conditional, crowded ? 1u : 0u);
}

__global__ void choose_local_epoch_path_kernel(
    cudaGraphConditionalHandle conditional,
    const std::uint32_t *local_epoch_overflow_flag) {
  if (blockIdx.x || threadIdx.x) return;
  const std::uint32_t local_overflow = atomicAdd(
      const_cast<std::uint32_t *>(local_epoch_overflow_flag), 0u);
  cudaGraphSetConditional(
      conditional, local_overflow ? 1u : 0u);
}

__global__ void set_staged_epoch_mode_kernel(std::uint32_t *mode) {
  if (!blockIdx.x && !threadIdx.x) *mode = 1u;
}

__global__ void initialize_rank_cell_mode_kernel(
    const ResidentPublicationPlan *plan,
    const std::uint32_t *staged_epoch_mode,
    std::uint32_t *rank_cell_mode) {
  if (blockIdx.x || threadIdx.x) return;
  const bool eligible = !kForceUnifiedMergeExperiment &&
      !plan->status && *staged_epoch_mode &&
      plan->source_count <= kRankCellMaximumSources;
  *rank_cell_mode = eligible ? 1u : 0u;
}

__global__ void validate_rank_cell_mode_kernel(
    const std::uint64_t *raw_counts,
    const std::uint32_t *staged_offsets,
    const Descriptor *descriptors,
    const DeviceManifest *manifests,
    const std::uint32_t *active_manifest,
    const ResidentPublicationPlan *plan,
    const RouteHeader *route_headers,
    const std::uint32_t *level_cell_rank_blocks,
    std::uint32_t *rank_cell_mode) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients || !atomicAdd(rank_cell_mode, 0u)) return;
  const std::uint64_t raw = resident_work_count(raw_counts[q]);
  bool valid = raw <= std::numeric_limits<std::uint16_t>::max() &&
      raw + kRankCellPaddingPerCell * kFoundationCells <=
          kRankCellTapeEntries;
  if (raw) valid &= raw >= kRankCellMinimumSectionRows;
  const std::uint32_t staged_count =
      staged_offsets[q + 1u] - staged_offsets[q];
  valid &= cell_rank_supported(staged_count);
  const DeviceManifestSnapshot manifest = load_active_manifest(
      manifests, active_manifest);
  for (std::uint32_t level = 0u;
       level <= plan->source_level_limit && valid; ++level) {
    if (!level_is_occupied(manifest.occupied_level_mask, level)) continue;
    const std::uint32_t count =
        descriptors[descriptor_index(q, level)].count();
    valid &= cell_rank_supported(count);
    if (count)
      valid &= route_headers[descriptor_index(q, level)].count == 1u;
    if (count && level != manifest.foundation_level)
      valid &= level_cell_rank_blocks[
          descriptor_index(q, level)] != kInvalid;
  }
  if (!valid) atomicExch(rank_cell_mode, 0u);
}

__global__ void prepare_rank_cell_reservations_kernel(
    const std::uint64_t *raw_counts,
    const std::uint32_t *rank_cell_mode,
    std::uint64_t *reservations) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  reservations[q] = q < kQuotients && *rank_cell_mode
      ? resident_work_count(raw_counts[q]) : 0u;
}

__global__ void override_rank_cell_route_headers_kernel(
    const std::uint64_t *raw_counts,
    const ResidentPublicationPlan *plan,
    const std::uint32_t *rank_cell_mode,
    std::uint32_t route_stride,
    RouteHeader *next_headers) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients || !*rank_cell_mode || plan->status) return;
  next_headers[q] = {
      plan->destination_level * route_stride + q,
      resident_work_count(raw_counts[q]) ? 1u : 0u};
}

__global__ void finalize_rank_cell_plan_kernel(
    ResidentPublicationPlan *plan,
    const std::uint64_t *section_raw_offsets,
    const std::uint32_t *route_offsets,
    const std::uint32_t *rank_cell_mode) {
  if (blockIdx.x || threadIdx.x || !*rank_cell_mode || plan->status) return;
  plan->raw_reservation = section_raw_offsets[kQuotients];
  plan->route_count = route_offsets[kQuotients];
  plan->job_count = plan->route_count;
  if (plan->raw_reservation > plan->output_capacity)
    plan->status |= kPublicationOutputOverflow;
}

__global__ void validate_direct_epoch_plan_kernel(
    ResidentPublicationPlan *plan,
    const std::uint32_t *tile_job_offsets,
    const std::uint64_t *job_output_offsets,
    const std::uint32_t *route_offsets,
    std::uint32_t maximum_jobs, std::uint32_t route_capacity) {
  if (blockIdx.x || threadIdx.x || plan->status) return;
  plan->job_count = tile_job_offsets[kPlanningTiles];
  plan->raw_reservation = job_output_offsets[maximum_jobs];
  plan->route_count = route_offsets[kQuotients];
  if (plan->job_count > maximum_jobs)
    plan->status |= kPublicationJobOverflow;
  if (plan->route_count > route_capacity)
    plan->status |= kPublicationRouteOverflow;
  if (plan->raw_reservation > plan->output_capacity)
    plan->status |= kPublicationOutputOverflow;
}

// GPU-resident publication.

__global__ void choose_resident_publication_path_kernel(
    const std::uint32_t *selected_count,
    const DeviceManifest *manifests, const std::uint32_t *active_manifest,
    const LevelStorageSpan *level_spans,
    std::uint64_t foundation_bank_capacity, std::uint32_t job_capacity,
    ResidentPublicationPlan *plan) {
  if (blockIdx.x || threadIdx.x) return;
  const std::uint32_t active = atomicAdd(
      const_cast<std::uint32_t *>(active_manifest), 0u) & 1u;
  const DeviceManifest *manifest = manifests + active;
  const std::uint64_t occupied = manifest->occupied_level_mask;
  const std::uint64_t empty = ~occupied;
  std::uint32_t destination = empty
      ? static_cast<std::uint32_t>(__ffsll(empty) - 1)
      : kMaximumLevels;

  ResidentPublicationPlan next{};
  next.selected_count = *selected_count;
  next.active_manifest = active;
  next.inactive_manifest = active ^ 1u;
  next.destination_level = destination;
  // Use one quotient-owned planner and merger.
  next.source_level_limit = destination ? destination - 1u : 0u;
  next.source_count = destination ? destination + 1u : 1u;
  next.destination_is_foundation =
      destination < kMaximumLevels &&
      (manifest->foundation_level == kMaximumLevels ||
       destination > manifest->foundation_level);
  next.keep_tombstones = !next.destination_is_foundation;
  next.job_capacity = job_capacity;
  next.status = destination < kMaximumLevels
      ? kPublicationSuccess : kPublicationJobOverflow;

  if (next.destination_is_foundation) {
    const std::uint32_t old_generation =
        manifest->foundation_level < kMaximumLevels
            ? manifest->levels[manifest->foundation_level].storage_generation
            : 1u;
    next.output_generation = old_generation ^ 1u;
    next.output_begin =
        std::uint64_t{next.output_generation} * foundation_bank_capacity;
    next.output_capacity = foundation_bank_capacity;
  } else if (destination < kMaximumLevels) {
    const LevelStorageSpan span = level_spans[destination];
    next.output_generation = manifest->generation + 1u;
    next.output_begin = span.begin;
    next.output_capacity = span.capacity;
  }
  if (next.selected_count > next.output_capacity)
    next.status |= kPublicationOutputOverflow;
  *plan = next;
}

__global__ void build_query_quotient_offsets_device_count_kernel(
    const std::uint32_t *keys, const std::uint32_t *device_count,
    std::uint32_t *offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  const std::uint32_t count = *device_count;
  if (q == kQuotients) {
    offsets[q] = count;
    return;
  }
  offsets[q] = lower_bound_full_keys(keys, count, q << 16u);
}


__device__ __forceinline__ std::uint32_t planning_tile_jobs(
    const std::uint64_t *raw_counts, std::uint32_t tile,
    std::uint32_t target, std::uint32_t safe_target) {
  const std::uint32_t first = tile * kPlanningTileQuotients;
  const std::uint32_t last = first + kPlanningTileQuotients;
  std::uint32_t jobs = 0u;
  std::uint64_t raw_work = 0u;
  for (std::uint32_t q = first; q < last; ++q) {
    const std::uint64_t encoded = raw_counts[q];
    const std::uint64_t count = resident_work_count(encoded);
    const bool resident = resident_work_present(encoded);
    if (resident || count > target) {
      if (raw_work) {
        ++jobs;
        raw_work = 0u;
      }
      if (count > target) {
      jobs += static_cast<std::uint32_t>(
          (count + safe_target - 1u) / safe_target);
      } else {
        ++jobs;
        if (q + 1u < last) {
          const std::uint64_t next_encoded = raw_counts[q + 1u];
          const std::uint64_t next = resident_work_count(next_encoded);
          if (resident_work_present(next_encoded) && next <= target &&
              count + next <= target)
            ++q;
        }
      }
    } else if (count) {
      if (raw_work && raw_work + count > target) {
        ++jobs;
        raw_work = count;
      } else {
        raw_work += count;
      }
    }
  }
  if (raw_work) ++jobs;
  return jobs;
}

__global__ void count_resident_planning_jobs_kernel(
    const std::uint64_t *raw_counts,
    const ResidentPublicationPlan *plan,
    const std::uint32_t *rank_cell_mode,
    std::uint32_t *tile_job_counts) {
  __shared__ std::uint64_t counts[kPlanningTileQuotients];
  const std::uint32_t tile = blockIdx.x;
  if (tile >= kPlanningTiles) return;
  const std::uint32_t first = tile * kPlanningTileQuotients;
  counts[threadIdx.x] = raw_counts[first + threadIdx.x];
  counts[threadIdx.x + blockDim.x] =
      raw_counts[first + threadIdx.x + blockDim.x];
  __syncthreads();
  if (threadIdx.x == 0u) {
    if (plan->status || *rank_cell_mode) {
      tile_job_counts[tile] = 0u;
    } else {
      const std::uint32_t safe = plan->job_capacity -
          (plan->source_count - 1u);
      tile_job_counts[tile] = planning_tile_jobs(
          counts, 0u, plan->job_capacity, safe);
    }
    if (tile + 1u == kPlanningTiles)
      tile_job_counts[kPlanningTiles] = 0u;
  }
}

__device__ __forceinline__ void emit_resident_job(
    BalancedMergeJob *jobs, std::uint64_t *job_raw_reservations,
    std::uint32_t global_index,
    std::uint64_t key_begin, std::uint64_t key_end,
    std::uint32_t q_begin, std::uint32_t q_end,
    std::uint64_t raw_count, std::uint16_t hot_piece,
    std::uint16_t hot_pieces) {
  BalancedMergeJob job{};
  job.key_begin = key_begin;
  job.key_end = key_end;
  job.quotient_begin = q_begin;
  job.quotient_end = q_end;
  job.route_ordinal = hot_pieces ? hot_piece : 0u;
  job.hot_piece = hot_piece;
  job.hot_pieces = hot_pieces;
  jobs[global_index] = job;
  job_raw_reservations[global_index] = raw_count;
}

__global__ void emit_resident_planning_jobs_kernel(
    const std::uint64_t *raw_counts,
    const std::uint32_t *tile_job_offsets,
    ResidentPublicationPlan *plan, std::uint32_t maximum_jobs,
    const std::uint32_t *rank_cell_mode,
    BalancedMergeJob *jobs, std::uint64_t *job_raw_reservations) {
  __shared__ std::uint64_t counts[kPlanningTileQuotients];
  const std::uint32_t tile = blockIdx.x;
  if (tile >= kPlanningTiles) return;
  const std::uint32_t first = tile * kPlanningTileQuotients;
  counts[threadIdx.x] = raw_counts[first + threadIdx.x];
  counts[threadIdx.x + blockDim.x] =
      raw_counts[first + threadIdx.x + blockDim.x];
  __syncthreads();
  if (threadIdx.x != 0u || plan->status || *rank_cell_mode) return;

  const std::uint32_t total_jobs = tile_job_offsets[kPlanningTiles];
  if (tile == 0u) {
    plan->job_count = total_jobs;
    if (total_jobs > maximum_jobs)
      atomicOr(&plan->status, kPublicationJobOverflow);
  }
  if (total_jobs > maximum_jobs) return;

  const std::uint32_t safe = plan->job_capacity -
      (plan->source_count - 1u);
  std::uint32_t global = tile_job_offsets[tile];
  std::uint32_t raw_begin = first;
  std::uint64_t raw_work = 0u;
  for (std::uint32_t q = first; q < first + kPlanningTileQuotients; ++q) {
    const std::uint64_t encoded = counts[q - first];
    const std::uint64_t count = resident_work_count(encoded);
    const bool resident = resident_work_present(encoded);
    if (resident || count > plan->job_capacity) {
      if (raw_work) {
        emit_resident_job(
            jobs, job_raw_reservations, global++,
            std::uint64_t{raw_begin} << 16u, std::uint64_t{q} << 16u,
            raw_begin, q, raw_work, 0u, 0u);
        raw_work = 0u;
      }
      if (count > plan->job_capacity) {
        const std::uint32_t pieces = static_cast<std::uint32_t>(
            (count + safe - 1u) / safe);
        for (std::uint32_t piece = 0u; piece < pieces; ++piece) {
          emit_resident_job(
              jobs, job_raw_reservations, global++,
              std::uint64_t{q} << 16u, std::uint64_t{q + 1u} << 16u,
              q, q + 1u, 0u, static_cast<std::uint16_t>(piece),
              static_cast<std::uint16_t>(pieces));
        }
      } else {
        const std::uint32_t begin = q;
        std::uint32_t end = q + 1u;
        std::uint64_t work = count;
        if (end < first + kPlanningTileQuotients) {
          const std::uint64_t next_encoded = counts[end - first];
          const std::uint64_t next = resident_work_count(next_encoded);
          if (resident_work_present(next_encoded) &&
              next <= plan->job_capacity &&
              work + next <= plan->job_capacity) {
            work += next;
            ++end;
            ++q;
          }
        }
        emit_resident_job(
            jobs, job_raw_reservations, global++,
            std::uint64_t{begin} << 16u, std::uint64_t{end} << 16u,
            begin, end, work, 0u, 0u);
      }
      raw_begin = q + 1u;
    } else if (count) {
      if (!raw_work) raw_begin = q;
      if (raw_work && raw_work + count > plan->job_capacity) {
        emit_resident_job(
            jobs, job_raw_reservations, global++,
            std::uint64_t{raw_begin} << 16u, std::uint64_t{q} << 16u,
            raw_begin, q, raw_work, 0u, 0u);
        raw_begin = q;
        raw_work = count;
      } else {
        raw_work += count;
      }
    }
  }
  if (raw_work) {
    std::uint32_t raw_end = first + kPlanningTileQuotients;
    while (raw_end > raw_begin &&
           !resident_work_count(counts[raw_end - first - 1u]))
      --raw_end;
    // Stop after the last nonempty quotient.
    if (raw_end == raw_begin) raw_end = first + kPlanningTileQuotients;
    emit_resident_job(
        jobs, job_raw_reservations, global,
        std::uint64_t{raw_begin} << 16u, std::uint64_t{raw_end} << 16u,
        raw_begin, raw_end, raw_work, 0u, 0u);
  }
}

__device__ __forceinline__ std::uint32_t
balanced_merge_prefix_count_warp(
    std::uint32_t q, std::uint32_t suffix,
    const Row *current_rows, const std::uint32_t *current_offsets,
    ResidentRows arena, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    std::uint32_t source_level_limit, std::uint64_t occupied_levels) {
  constexpr unsigned mask = 0xffffffffu;
  const std::uint32_t lane = threadIdx.x & 31u;
  std::uint32_t result = 0u;
  if (lane == 0u) {
    const std::uint32_t current_begin = current_offsets[q];
    result = lower_bound_rows(
        current_rows + current_begin,
        current_offsets[q + 1u] - current_begin, suffix);
  }
  for (std::uint32_t level = lane; level <= source_level_limit;
       level += 32u) {
    if (level_is_occupied(occupied_levels, level))
      result += logical_section_bound(
          q, level, suffix, false, arena, route_headers, route_slices,
          route_logical_begins, level_q_logical_offsets);
  }
  for (std::uint32_t offset = 16u; offset; offset >>= 1u)
    result += __shfl_down_sync(mask, result, offset);
  return __shfl_sync(mask, result, 0u);
}

__device__ __forceinline__ std::uint32_t resident_hot_boundary_warp(
    std::uint32_t q, std::uint32_t target,
    const Row *current_rows, const std::uint32_t *current_offsets,
    ResidentRows arena, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    std::uint32_t source_level_limit, std::uint64_t occupied_levels) {
  if (!target) return 0u;
  std::uint32_t low = 1u, high = 1u << 16u;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (balanced_merge_prefix_count_warp(
            q, middle, current_rows, current_offsets, arena,
            route_headers, route_slices, route_logical_begins,
            level_q_logical_offsets, source_level_limit,
            occupied_levels) < target)
      low = middle + 1u;
    else
      high = middle;
  }
  return low;
}

__device__ __forceinline__ std::uint32_t ranked_merge_prefix_warp(
    std::uint32_t q, std::uint32_t cell,
    const Row *current_rows, const std::uint32_t *current_offsets,
    const std::uint16_t *current_ranks,
    const Descriptor *descriptors,
    const std::uint32_t *level_cell_rank_blocks,
    const std::uint16_t *level_cell_ranks,
    const std::uint16_t *local_rank,
    std::uint32_t source_level_limit, std::uint32_t foundation_level,
    std::uint64_t occupied_levels, bool &supported) {
  constexpr unsigned mask = 0xffffffffu;
  const std::uint32_t lane = threadIdx.x & 31u;
  std::uint32_t result = 0u;
  bool valid = true;
  if (lane == 0u) {
    const std::uint32_t raw_count =
        current_offsets[q + 1u] - current_offsets[q];
    valid = cell_rank_supported(raw_count);
    if (valid)
      result = cell == kFoundationCells ? raw_count
          : current_ranks[std::size_t{q} * kFoundationCells + cell];
  }
  for (std::uint32_t level = lane; level <= source_level_limit;
       level += 32u) {
    if (!level_is_occupied(occupied_levels, level)) continue;
    const std::uint32_t count =
        descriptors[descriptor_index(q, level)].count();
    if (!cell_rank_supported(count)) {
      valid = false;
      continue;
    }
    if (cell == kFoundationCells) {
      result += count;
      continue;
    }
    const std::uint16_t *ranks = nullptr;
    if (level == foundation_level) {
      ranks = local_rank + std::size_t{q} * kFoundationCells;
    } else {
      const std::uint32_t block =
          level_cell_rank_blocks[descriptor_index(q, level)];
      if (block == kInvalid) {
        valid = false;
        continue;
      }
      ranks = level_cell_ranks +
          std::size_t{block} * kFoundationCells;
    }
    result += ranks[cell];
  }
  for (std::uint32_t offset = 16u; offset; offset >>= 1u) {
    result += __shfl_down_sync(mask, result, offset);
    valid &= __shfl_down_sync(mask, valid, offset);
  }
  supported = __shfl_sync(mask, valid, 0u);
  return __shfl_sync(mask, result, 0u);
}

__device__ __forceinline__ std::uint32_t ranked_merge_boundary_warp(
    std::uint32_t q, std::uint32_t target,
    const Row *current_rows, const std::uint32_t *current_offsets,
    const std::uint16_t *current_ranks,
    const Descriptor *descriptors,
    const std::uint32_t *level_cell_rank_blocks,
    const std::uint16_t *level_cell_ranks,
    const std::uint16_t *local_rank,
    std::uint32_t source_level_limit, std::uint32_t foundation_level,
    std::uint64_t occupied_levels, bool &supported) {
  if (!target) {
    supported = true;
    return 0u;
  }
  std::uint32_t low = 1u, high = kFoundationCells;
  supported = true;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    bool valid = true;
    const std::uint32_t count = ranked_merge_prefix_warp(
        q, middle, current_rows, current_offsets, current_ranks,
        descriptors, level_cell_rank_blocks, level_cell_ranks, local_rank,
        source_level_limit, foundation_level, occupied_levels, valid);
    supported &= valid;
    if (count < target) low = middle + 1u;
    else high = middle;
  }
  return low;
}

__global__ void resolve_resident_job_boundaries_kernel(
    BalancedMergeJob *jobs, std::uint64_t *job_raw_reservations,
    const ResidentPublicationPlan *plan,
    const Row *current_rows, const std::uint32_t *current_offsets,
    ResidentRows arena, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    const DeviceManifest *manifests,
    const std::uint32_t *active_manifest,
    const Descriptor *descriptors,
    const std::uint32_t *rank_cell_mode,
    const std::uint16_t *current_ranks,
    const std::uint32_t *level_cell_rank_blocks,
    const std::uint16_t *level_cell_ranks,
    const std::uint16_t *local_rank) {
  if (*rank_cell_mode) return;
  const std::uint32_t lane = threadIdx.x & 31u;
  constexpr unsigned mask = 0xffffffffu;
  const DeviceManifestSnapshot manifest = load_active_manifest(
      manifests, active_manifest);
  for (std::uint32_t job_index = blockIdx.x;
       job_index < plan->job_count && !plan->status;
       job_index += gridDim.x) {
  std::uint32_t hot_pieces = lane == 0u
      ? jobs[job_index].hot_pieces : 0u;
  hot_pieces = __shfl_sync(mask, hot_pieces, 0u);
  if (!hot_pieces) {
    if (lane == 0u &&
        job_raw_reservations[job_index] > plan->job_capacity)
      atomicOr(const_cast<std::uint32_t *>(&plan->status),
               kPublicationJobTooLarge);
    continue;
  }
  BalancedMergeJob job = jobs[job_index];
  const std::uint32_t q = job.quotient_begin;
  bool ranked_total = true;
  std::uint32_t raw = ranked_merge_prefix_warp(
      q, kFoundationCells, current_rows, current_offsets, current_ranks,
      descriptors, level_cell_rank_blocks, level_cell_ranks, local_rank,
      plan->source_level_limit, manifest.foundation_level,
      manifest.occupied_level_mask, ranked_total);
  if (!ranked_total)
    raw = balanced_merge_prefix_count_warp(
        q, 1u << 16u, current_rows, current_offsets, arena,
        route_headers, route_slices, route_logical_begins,
        level_q_logical_offsets, plan->source_level_limit,
        manifest.occupied_level_mask);
  std::uint32_t low_cell = 0u, high_cell = kFoundationCells;
  std::uint32_t ranked_begin = 0u, ranked_end = raw;
  bool use_ranked = ranked_total;
  std::uint32_t previous_cell = 0u;
  std::uint32_t previous_count = 0u;
  for (std::uint32_t piece = 0u;
       piece < job.hot_pieces && use_ranked; ++piece) {
    const std::uint32_t target = static_cast<std::uint32_t>(
        (std::uint64_t{raw} * (piece + 1u) + job.hot_pieces - 1u) /
        job.hot_pieces);
    bool boundary_valid = true;
    const std::uint32_t next_cell = piece + 1u == job.hot_pieces
        ? kFoundationCells
        : ranked_merge_boundary_warp(
              q, target, current_rows, current_offsets, current_ranks,
              descriptors, level_cell_rank_blocks, level_cell_ranks,
              local_rank, plan->source_level_limit,
              manifest.foundation_level, manifest.occupied_level_mask,
              boundary_valid);
    bool prefix_valid = true;
    const std::uint32_t next_count = ranked_merge_prefix_warp(
        q, next_cell, current_rows, current_offsets, current_ranks,
        descriptors, level_cell_rank_blocks, level_cell_ranks, local_rank,
        plan->source_level_limit, manifest.foundation_level,
        manifest.occupied_level_mask, prefix_valid);
    const std::uint32_t piece_count = next_count - previous_count;
    use_ranked = boundary_valid && prefix_valid &&
        next_cell > previous_cell && piece_count <= plan->job_capacity;
    if (piece == job.hot_piece) {
      low_cell = previous_cell;
      high_cell = next_cell;
      ranked_begin = previous_count;
      ranked_end = next_count;
    }
    previous_cell = next_cell;
    previous_count = next_count;
  }
  const std::uint32_t low_target = static_cast<std::uint32_t>(
      (std::uint64_t{raw} * job.hot_piece + job.hot_pieces - 1u) /
      job.hot_pieces);
  const std::uint32_t high_target = static_cast<std::uint32_t>(
      (std::uint64_t{raw} * (job.hot_piece + 1u) +
       job.hot_pieces - 1u) / job.hot_pieces);
  std::uint32_t low = low_cell * kFoundationCellKeys;
  std::uint32_t high = high_cell * kFoundationCellKeys;
  std::uint32_t exact = ranked_end - ranked_begin;
  if (!use_ranked) {
    low = resident_hot_boundary_warp(
        q, low_target, current_rows, current_offsets, arena,
        route_headers, route_slices, route_logical_begins,
        level_q_logical_offsets, plan->source_level_limit,
        manifest.occupied_level_mask);
    high = job.hot_piece + 1u == job.hot_pieces
        ? (1u << 16u)
        : resident_hot_boundary_warp(
              q, high_target, current_rows, current_offsets, arena,
              route_headers, route_slices, route_logical_begins,
              level_q_logical_offsets, plan->source_level_limit,
              manifest.occupied_level_mask);
    exact = balanced_merge_prefix_count_warp(
        q, high, current_rows, current_offsets, arena, route_headers,
        route_slices, route_logical_begins, level_q_logical_offsets,
        plan->source_level_limit, manifest.occupied_level_mask) -
        balanced_merge_prefix_count_warp(
            q, low, current_rows, current_offsets, arena, route_headers,
            route_slices, route_logical_begins, level_q_logical_offsets,
            plan->source_level_limit, manifest.occupied_level_mask);
  }
  if (lane == 0u) {
    job.key_begin = (std::uint64_t{q} << 16u) + low;
    job.key_end = (std::uint64_t{q} << 16u) + high;
    job_raw_reservations[job_index] = exact;
    jobs[job_index] = job;
    if (exact > plan->job_capacity)
      atomicOr(const_cast<std::uint32_t *>(&plan->status),
               kPublicationJobTooLarge);
  }
  }
}

__global__ void count_resident_route_slots_kernel(
    const std::uint64_t *raw_counts,
    const LevelRankSpan *rank_spans, ResidentPublicationPlan *plan,
    const std::uint32_t *rank_cell_mode,
    std::uint32_t *route_counts,
    std::uint32_t *level_cell_rank_blocks) {
  using BlockScan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::uint32_t block_base;
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t destination = plan->destination_level;
  const std::uint64_t raw = q < kQuotients
      ? resident_work_count(raw_counts[q]) : 0u;
  const std::uint32_t requested =
      !plan->status && destination < kMaximumLevels &&
      !plan->destination_is_foundation && q < kQuotients &&
      raw >= kDenseCellRankMinimumRows &&
      cell_rank_supported(raw);
  std::uint32_t local{}, block_count{};
  BlockScan(scan_storage).ExclusiveSum(requested, local, block_count);
  if (threadIdx.x == 0u)
    block_base = atomicAdd(&plan->rank_block_count, block_count);
  __syncthreads();

  if (q < kQuotients && destination < kMaximumLevels) {
    const std::size_t mapping = descriptor_index(q, destination);
    if (!requested) {
      level_cell_rank_blocks[mapping] = kInvalid;
    } else {
      const LevelRankSpan span = rank_spans[destination];
      const std::uint32_t relative = block_base + local;
      if (relative < span.capacity_blocks) {
        level_cell_rank_blocks[mapping] =
            static_cast<std::uint32_t>(span.begin_block + relative);
      } else {
        level_cell_rank_blocks[mapping] = kInvalid;
        atomicOr(&plan->status, kPublicationOutputOverflow);
      }
    }
  }
  if (q > kQuotients) return;
  if (q == kQuotients || plan->status) {
    route_counts[q] = 0u;
    return;
  }
  if (!raw) {
    route_counts[q] = 0u;
  } else if (*rank_cell_mode) {
    route_counts[q] = 1u;
  } else if (raw <= plan->job_capacity) {
    route_counts[q] = 1u;
  } else {
    const std::uint32_t safe = plan->job_capacity -
        (plan->source_count - 1u);
    route_counts[q] = static_cast<std::uint32_t>(
        (raw + safe - 1u) / safe);
  }
}

__global__ void prepare_resident_route_headers_kernel(
    const std::uint32_t *route_counts,
    const std::uint32_t *route_offsets,
    const ResidentPublicationPlan *plan, std::uint32_t route_stride,
    RouteHeader *next_headers) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients || plan->status) return;
  next_headers[q] = {
      plan->destination_level * route_stride + route_offsets[q],
      route_counts[q]};
}

__global__ void assign_resident_output_offsets_kernel(
    BalancedMergeJob *jobs, const std::uint64_t *job_output_offsets,
    const std::uint64_t *job_raw_reservations,
    const ResidentPublicationPlan *plan) {
  if (plan->status) return;
  for (std::uint32_t job = blockIdx.x * blockDim.x + threadIdx.x;
       job < plan->job_count; job += gridDim.x * blockDim.x) {
    jobs[job].existing_offset = plan->output_begin + job_output_offsets[job];
    jobs[job].existing_capacity =
        static_cast<std::uint32_t>(job_raw_reservations[job]);
  }
}

__global__ void finalize_resident_route_metadata_kernel(
    const ResidentPublicationPlan *plan,
    const RouteHeader *next_headers,
    const std::uint32_t *section_logical_offsets,
    std::uint32_t route_stride, RouteSlice *route_slices,
    RouteHeader *route_headers, Descriptor *descriptors,
    std::uint32_t *route_logical_begins,
    std::uint16_t *route_quotients,
    std::uint32_t *level_q_logical_offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients || plan->status) return;
  const std::uint32_t level = plan->destination_level;
  level_q_logical_offsets[
      std::size_t{level} * (kQuotients + 1u) + q] =
      section_logical_offsets[q];
  if (q == kQuotients) return;
  RouteHeader header = next_headers[q];
  std::uint32_t logical = section_logical_offsets[q];
  std::uint32_t total = 0u;
  for (std::uint32_t local = 0u; local < header.count; ++local) {
    const std::uint32_t route = header.begin + local;
    route_logical_begins[route] = logical + total;
    route_quotients[route] = static_cast<std::uint16_t>(q);
    total += route_slices[route].rows.count();
  }
  if (!total) header.count = 0u;
  route_headers[descriptor_index(q, level)] = header;
  descriptors[descriptor_index(q, level)] =
      header.count == 1u ? route_slices[header.begin].rows
      : header.count > 1u ? Descriptor::make_split(total)
                          : Descriptor{};
}

__global__ void build_split_resident_query_metadata_kernel(
    const ResidentPublicationPlan *plan, ResidentRows arena,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    const std::uint32_t *level_cell_rank_blocks,
    std::uint16_t *level_cell_ranks, std::uint16_t *local_rank,
    std::uint16_t *level_guides) {
  if (plan->status) return;
  const std::uint32_t level = plan->destination_level;
  for (std::uint32_t q = blockIdx.x; q < kQuotients; q += gridDim.x) {
  const RouteHeader header = route_headers[descriptor_index(q, level)];
  if (header.count <= 1u) continue;
  const Descriptor descriptor = descriptors[descriptor_index(q, level)];
  if (plan->destination_is_foundation) {
    for (std::uint32_t cell = threadIdx.x; cell < 128u;
         cell += blockDim.x) {
      const std::uint32_t target = cell << 9u;
      const std::uint32_t position = cell_rank_supported(descriptor.count())
          ? logical_section_bound(
                q, level, target, false, arena, route_headers,
                route_slices, route_logical_begins,
                level_q_logical_offsets)
          : 0u;
      local_rank[std::size_t{q} * 128u + cell] =
          static_cast<std::uint16_t>(position);
    }
  } else {
    const std::uint32_t rank_block =
        level_cell_rank_blocks[descriptor_index(q, level)];
    if (rank_block != kInvalid) {
      for (std::uint32_t cell = threadIdx.x; cell < kFoundationCells;
           cell += blockDim.x) {
        const std::uint32_t position = logical_section_bound(
            q, level, cell * kFoundationCellKeys, false, arena,
            route_headers, route_slices, route_logical_begins,
            level_q_logical_offsets);
        level_cell_ranks[
            std::size_t{rank_block} * kFoundationCells + cell] =
            static_cast<std::uint16_t>(position);
      }
    }
    if (descriptor.count() >= kGuideRegions) {
    for (std::uint32_t sample = threadIdx.x; sample < kGuideSamples;
         sample += blockDim.x) {
      const std::uint32_t position =
          (sample + 1u) * descriptor.count() / kGuideRegions;
      level_guides[guide_index(q, level) + sample] =
          logical_section_row(
              q, level, position, arena, route_headers, route_slices,
              route_logical_begins, level_q_logical_offsets).key;
    }
    }
  }
  }
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

__global__ void set_merged_survivor_count_kernel(
    ResidentPublicationPlan *plan,
    const std::uint32_t *section_logical_offsets) {
  if (blockIdx.x || threadIdx.x || plan->status)
    return;
  plan->survivor_count = section_logical_offsets[kQuotients];
}

__global__ void fold_resident_merge_status_kernel(
    ResidentPublicationPlan *plan, const std::uint32_t *overflow_flag) {
  if (blockIdx.x || threadIdx.x) return;
  if (*overflow_flag) plan->status |= kPublicationOutputOverflow;
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

// Canonical quotient-run carry.  Every occupied level is one immutable,
// quotient-major run.  The persistent directory is a quotient prefix plus
// 128 exact cell starts per quotient; routes and guides are not consulted by
// the canonical point path.

__global__ void choose_canonical_publication_path_kernel(
    const std::uint32_t *selected_count,
    const DeviceManifest *manifests, const std::uint32_t *active_manifest,
    const LevelStorageSpan *level_spans, std::uint32_t level_count,
    std::uint32_t job_capacity, bool top_level_rollover,
    ResidentPublicationPlan *plan) {
  if (blockIdx.x || threadIdx.x) return;
  const std::uint32_t active = atomicAdd(
      const_cast<std::uint32_t *>(active_manifest), 0u) & 1u;
  const DeviceManifest *manifest = manifests + active;
  const std::uint64_t occupied = manifest->occupied_level_mask;
  const std::uint64_t empty = ~occupied;
  const std::uint32_t natural_destination = empty
      ? static_cast<std::uint32_t>(__ffsll(empty) - 1)
      : kMaximumLevels;
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
      ? destination : destination ? destination - 1u : 0u;
  const std::uint64_t consumed = valid_rollover
      ? destination == kMaximumLevels - 1u
          ? ~std::uint64_t{0}
          : (std::uint64_t{1} << (destination + 1u)) - 1u
      : destination == kMaximumLevels
          ? ~std::uint64_t{0}
          : destination ? (std::uint64_t{1} << destination) - 1u : 0u;
  next.source_count = 1u + static_cast<std::uint32_t>(
      __popcll(occupied & consumed));
  next.destination_is_foundation = valid_rollover ||
      (destination < kMaximumLevels &&
      (manifest->foundation_level == kMaximumLevels ||
       destination > manifest->foundation_level));
  next.keep_tombstones = !next.destination_is_foundation;
  next.output_generation = valid_rollover
      ? (manifest->levels[destination].storage_generation ^ 1u) & 1u
      : 0u;
  next.job_capacity = job_capacity;
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
    const std::uint64_t *counts, std::uint32_t capacity,
    std::uint32_t safe_capacity) {
  std::uint32_t jobs = 0u;
  std::uint64_t run_rows = 0u;
  std::uint32_t run_begin = 0u;
  for (std::uint32_t local_q = 0u;
       local_q < kPlanningTileQuotients; ++local_q) {
    const std::uint64_t count = counts[local_q];
    if (count > capacity) {
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
    if (run_rows &&
        (run_rows + count > capacity ||
         local_q - run_begin >= kCanonicalJobQuotients)) {
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
  const std::uint32_t tile = blockIdx.x;
  if (tile >= kPlanningTiles) return;
  const std::uint32_t first = tile * kPlanningTileQuotients;
  tile_counts[threadIdx.x] = counts[first + threadIdx.x];
  tile_counts[threadIdx.x + blockDim.x] =
      counts[first + threadIdx.x + blockDim.x];
  __syncthreads();
  if (threadIdx.x == 0u) {
    const std::uint32_t safe = plan->job_capacity > plan->source_count
        ? plan->job_capacity - (plan->source_count - 1u) : 1u;
    tile_job_counts[tile] = plan->status ? 0u :
        canonical_tile_job_count(tile_counts, plan->job_capacity, safe);
    if (tile + 1u == kPlanningTiles)
      tile_job_counts[kPlanningTiles] = 0u;
  }
}

__global__ void emit_canonical_planning_jobs_kernel(
    std::uint64_t *counts, const std::uint32_t *tile_job_offsets,
    ResidentPublicationPlan *plan, std::uint32_t maximum_jobs,
    BalancedMergeJob *jobs, std::uint64_t *reservations) {
  __shared__ std::uint64_t tile_counts[kPlanningTileQuotients];
  const std::uint32_t tile = blockIdx.x;
  if (tile >= kPlanningTiles) return;
  const std::uint32_t first = tile * kPlanningTileQuotients;
  tile_counts[threadIdx.x] = counts[first + threadIdx.x];
  tile_counts[threadIdx.x + blockDim.x] =
      counts[first + threadIdx.x + blockDim.x];
  __syncthreads();
  if (threadIdx.x != 0u || plan->status) return;
  const std::uint32_t total_jobs = tile_job_offsets[kPlanningTiles];
  if (tile == 0u) {
    plan->job_count = total_jobs;
    if (total_jobs > maximum_jobs)
      atomicOr(&plan->status, kPublicationJobOverflow);
  }
  if (total_jobs > maximum_jobs) return;
  const std::uint32_t safe = plan->job_capacity > plan->source_count
      ? plan->job_capacity - (plan->source_count - 1u) : 1u;
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
        run_begin, run_end, run_rows, 0u, 0u);
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
    if (count > plan->job_capacity) {
      flush();
      const std::uint32_t pieces = static_cast<std::uint32_t>(
          (count + safe - 1u) / safe);
      const std::uint32_t first_job = global;
      for (std::uint32_t piece = 0u; piece < pieces; ++piece)
        emit_resident_job(
            jobs, reservations, global++,
            std::uint64_t{q} << 16u,
            std::uint64_t{q + 1u} << 16u,
            q, q + 1u, 0u, static_cast<std::uint16_t>(piece),
            static_cast<std::uint16_t>(pieces));
      counts[q] = canonical_hot_job(first_job, pieces);
      continue;
    }
    if (!count) continue;
    if (!run_rows) {
      run_begin = q;
      run_end = q + 1u;
    }
    if (run_rows &&
        (run_rows + count > plan->job_capacity ||
         q - run_begin >= kCanonicalJobQuotients)) {
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
  if (lane == 0u)
    total = cell < kFoundationCells
        ? epoch_cell_ranks[
              std::size_t{q} * kFoundationCells + cell]
        : epoch_counts[q];
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
    const std::uint32_t section_begin = epoch_offsets[q];
    const std::uint16_t *ranks = epoch_cell_ranks +
        std::size_t{q} * kFoundationCells;
    const std::uint32_t begin = ranks[cell];
    const std::uint32_t end = cell + 1u < kFoundationCells
        ? ranks[cell + 1u] : epoch_counts[q];
    total = begin + lower_bound_rows(
        epoch_rows + section_begin + begin, end - begin, suffix);
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
            exact > plan->job_capacity)
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

__device__ __forceinline__ std::uint64_t canonical_source_bound(
    std::uint32_t source, std::uint32_t level,
    std::uint64_t key, const Row *epoch_rows,
    const std::uint32_t *epoch_offsets, ResidentRows arena,
    const LevelStorageSpan *level_spans,
    const std::uint32_t *level_q_offsets) {
  const std::uint32_t q = static_cast<std::uint32_t>(key >> 16u);
  if (source == 0u) {
    if (q >= kQuotients) return epoch_offsets[kQuotients];
    const std::uint32_t begin = epoch_offsets[q];
    const std::uint32_t count = epoch_offsets[q + 1u] - begin;
    return begin + lower_bound_rows(
        epoch_rows + begin, count,
        static_cast<std::uint32_t>(key & 0xffffu));
  }
  const LevelStorageSpan span = level_spans[level];
  const std::size_t base =
      std::size_t{level} * (kQuotients + 1u);
  if (q >= kQuotients)
    return span.begin + level_q_offsets[base + kQuotients];
  const std::uint32_t begin = level_q_offsets[base + q];
  const std::uint32_t count = level_q_offsets[base + q + 1u] - begin;
  return span.begin + begin + lower_bound_rows(
      arena + span.begin + begin, count,
      static_cast<std::uint32_t>(key & 0xffffu));
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

template <std::uint32_t StaticSources>
__global__ void canonical_carry_jobs_kernel(
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
  constexpr std::uint32_t kSourceSlots =
      StaticSources ? StaticSources : kMaximumMergeSources;
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
      const std::uint32_t expected_sources = StaticSources
          ? StaticSources : plan->source_count;
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
        if constexpr (StaticSources != 2u) {
          if (physical + 1u == physical_run_count_shared)
            plane_b[destination + position] = record;
        }
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
      jobs[job_index].existing_offset = plan->output_begin + prefix;
      jobs[job_index].output_count = job_output_count;
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

    if (threadIdx.x == 0u) {
      task_count_shared =
          (job.quotient_end - job.quotient_begin) * kFoundationCells;
      next_task_shared = 0u;
      tape_cursor_shared = 0u;
      job_valid_shared = source_count_shared == plan->source_count &&
          source_count_shared <= kMaximumMergeSources && task_count_shared &&
          task_count_shared <= kCanonicalTournamentTasks;
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
        std::size_t{kCanonicalJobQuotients} * source_count;
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
    offset += std::size_t{plan->job_capacity} *
        sizeof(CanonicalTournamentReference);
    offset = canonical_align_bytes(offset, alignof(std::uint16_t));
    std::uint16_t *task_tape_bases =
        reinterpret_cast<std::uint16_t *>(workspace + offset);
    offset += std::size_t{kCanonicalTournamentTasks} *
        sizeof(std::uint16_t);
    std::uint16_t *task_output_offsets =
        reinterpret_cast<std::uint16_t *>(workspace + offset);

    if (!job_valid_shared) {
      if (threadIdx.x == 0u) {
        atomicOr(&plan->status, kPublicationJobTooLarge);
        canonical_job_prefix(job_index, 0u, prefixes);
      }
      __syncthreads();
      continue;
    }

    const std::uint32_t quotient_count =
        job.quotient_end - job.quotient_begin;
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
        if (tape_begin + raw_count > plan->job_capacity) {
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
              if constexpr (kCanonicalCompactMultiway) {
                // Canonical epoch and level runs contain at most one row for
                // each logical key.  Therefore a source contributes at most
                // the 512 key positions in one cell.  Duplicate user writes
                // are resolved before this merge; they are not an external
                // unique-key requirement.
                survivor_tape[tape_begin + survivor_count++] =
                    static_cast<std::uint16_t>(
                        (source << kCanonicalTournamentReferenceBits) |
                        local_position);
              } else {
                survivor_tape[tape_begin + survivor_count++] =
                    (source << 16u) | position;
              }
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
      jobs[job_index].existing_offset = plan->output_begin + prefix;
      jobs[job_index].output_count = job_output_count;
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
        if constexpr (kCanonicalCompactMultiway) {
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
        } else {
          source = reference >> 16u;
          position = reference & 0xffffu;
          const std::uint32_t local_q = q - job.quotient_begin;
          source_base = source_bases[
              std::size_t{local_q} * source_count + source];
        }
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

__global__ void initialize_single_route_auxiliary_kernel(
    const std::uint32_t *section_offsets, std::uint32_t level,
    std::uint32_t route_stride, std::uint32_t *route_logical_begins,
    std::uint16_t *route_quotients,
    std::uint32_t *level_q_logical_offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  level_q_logical_offsets[
      std::size_t{level} * (kQuotients + 1u) + q] = section_offsets[q];
  if (q == kQuotients) return;
  const std::uint32_t route = level * route_stride + q;
  route_logical_begins[route] = section_offsets[q];
  route_quotients[route] = static_cast<std::uint16_t>(q);
}






__device__ __forceinline__ std::uint32_t direct_raw_batch_for_candidate(
    std::uint16_t candidate,
    const std::uint32_t *raw_candidate_offsets,
    std::uint32_t pending_batches) {
  std::uint32_t batch = 0u;
#pragma unroll
  for (std::uint32_t b = 1u; b < kBatchesPerEpoch; ++b)
    if (b < pending_batches && candidate >= raw_candidate_offsets[b])
      batch = b;
  return batch;
}

__device__ __forceinline__ std::uint32_t direct_raw_metadata(
    std::uint16_t candidate, std::uint32_t q,
    const std::uint32_t *raw_candidate_offsets,
    const RawPayload *raw_payloads, const std::uint32_t *raw_offsets,
    std::uint32_t batch_stride, std::uint32_t pending_batches) {
  const std::uint32_t batch = direct_raw_batch_for_candidate(
      candidate, raw_candidate_offsets, pending_batches);
  const std::uint32_t local =
      candidate - raw_candidate_offsets[batch];
  const std::size_t offset =
      std::size_t{batch} * (kQuotients + 1u) + q;
  return raw_payloads[std::size_t{batch} * batch_stride +
                      raw_offsets[offset] + local].metadata;
}

__device__ __forceinline__ std::uint32_t direct_raw_metadata_known_batch(
    std::uint16_t candidate, std::uint32_t batch, std::uint32_t q,
    const std::uint32_t *raw_candidate_offsets,
    const RawPayload *raw_payloads, const std::uint32_t *raw_offsets,
    std::uint32_t batch_stride) {
  const std::uint32_t local = candidate - raw_candidate_offsets[batch];
  const std::size_t offset =
      std::size_t{batch} * (kQuotients + 1u) + q;
  return raw_payloads[std::size_t{batch} * batch_stride +
                      raw_offsets[offset] + local].metadata;
}

__device__ __forceinline__ bool direct_epoch_candidate_less(
    std::uint16_t left, std::uint16_t right,
    const CandidateToken *candidate_tokens) {
  const std::uint32_t a_key =
      candidate_token_logical_key(candidate_tokens[left]);
  const std::uint32_t b_key =
      candidate_token_logical_key(candidate_tokens[right]);
  // Candidate IDs break equal-key partition ties.
  return a_key != b_key ? a_key < b_key : left < right;
}

__device__ __forceinline__ std::uint32_t direct_epoch_merge_partition(
    const std::uint16_t *left, std::uint32_t left_count,
    const std::uint16_t *right, std::uint32_t right_count,
    std::uint32_t diagonal, const CandidateToken *candidate_tokens) {
  std::uint32_t low = diagonal > right_count
      ? diagonal - right_count : 0u;
  std::uint32_t high = min(diagonal, left_count);
  while (low <= high) {
    const std::uint32_t li = (low + high) >> 1u;
    const std::uint32_t ri = diagonal - li;
    if (li && ri < right_count && direct_epoch_candidate_less(
            right[ri], left[li - 1u], candidate_tokens)) {
      high = li - 1u;
    } else if (ri && li < left_count && direct_epoch_candidate_less(
                   left[li], right[ri - 1u], candidate_tokens)) {
      low = li + 1u;
    } else {
      return li;
    }
  }
  return low;
}

__device__ __forceinline__ bool cell_owned_candidate_less(
    std::uint16_t left, std::uint16_t right,
    const CandidateToken *candidate_tokens) {
  const std::uint16_t left_key = candidate_token_key(candidate_tokens[left]);
  const std::uint16_t right_key =
      candidate_token_key(candidate_tokens[right]);
  return left_key != right_key ? left_key < right_key : left < right;
}

__device__ __forceinline__ std::uint32_t cell_owned_merge_partition(
    const std::uint16_t *left, std::uint32_t left_count,
    const std::uint16_t *right, std::uint32_t right_count,
    std::uint32_t diagonal, const CandidateToken *candidate_tokens) {
  std::uint32_t low = diagonal > right_count
      ? diagonal - right_count : 0u;
  std::uint32_t high = min(diagonal, left_count);
  while (low <= high) {
    const std::uint32_t li = (low + high) >> 1u;
    const std::uint32_t ri = diagonal - li;
    if (li && ri < right_count && cell_owned_candidate_less(
            right[ri], left[li - 1u], candidate_tokens)) {
      high = li - 1u;
    } else if (ri && li < left_count && cell_owned_candidate_less(
                   left[li], right[ri - 1u], candidate_tokens)) {
      low = li + 1u;
    } else {
      return li;
    }
  }
  return low;
}

__device__ __forceinline__ bool cell_owned_candidate_is_newer(
    std::uint16_t candidate, std::uint16_t current,
    const CandidateToken *candidate_tokens, std::uint32_t raw_count,
    bool updates_are_resolved, std::uint32_t q_begin,
    const std::uint32_t *raw_candidate_offsets,
    const RawPayload *raw_payloads, const std::uint32_t *raw_offsets,
    std::uint32_t batch_stride, std::uint32_t pending_batches) {
  const bool candidate_raw = candidate < raw_count;
  const bool current_raw = current < raw_count;
  if (candidate_raw != current_raw) return candidate_raw;
  if (candidate_raw) {
    if (updates_are_resolved) return candidate < current;
    const std::uint32_t candidate_batch =
        candidate_token_source(candidate_tokens[candidate]);
    const std::uint32_t current_batch =
        candidate_token_source(candidate_tokens[current]);
    if (candidate_batch != current_batch)
      return candidate_batch > current_batch;
    const std::uint32_t candidate_age = direct_raw_metadata_known_batch(
        candidate, candidate_batch, q_begin, raw_candidate_offsets,
        raw_payloads, raw_offsets, batch_stride) & ~kRawTombstone;
    const std::uint32_t current_age = direct_raw_metadata_known_batch(
        current, current_batch, q_begin, raw_candidate_offsets,
        raw_payloads, raw_offsets, batch_stride) & ~kRawTombstone;
    return candidate_age > current_age;
  }
  const std::uint32_t candidate_age =
      candidate_token_source(candidate_tokens[candidate]);
  const std::uint32_t current_age =
      candidate_token_source(candidate_tokens[current]);
  return candidate_age < current_age;
}

__device__ __forceinline__ bool cell_owned_candidate_is_tombstone(
    std::uint16_t candidate, const std::uint32_t *tombstone_words) {
  return (tombstone_words[candidate >> 5u] &
          (1u << (candidate & 31u))) != 0u;
}

// Bucket cells by estimated serial work.
__device__ __forceinline__ std::uint32_t cell_owned_cost_bucket(
    std::uint32_t total_count, std::uint32_t raw_count) {
  const std::uint32_t raw_sort_work =
      raw_count * (raw_count ? raw_count - 1u : 0u) / 2u;
  const std::uint32_t score = total_count + raw_sort_work;
  const std::uint32_t bucket = score >> 1u;
  return bucket < kCellOwnedCostBuckets
      ? bucket : kCellOwnedCostBuckets - 1u;
}

__device__ __forceinline__ void record_candidate_tombstone(
    std::uint32_t candidate, bool tombstone,
    std::uint32_t *tombstone_words) {
  const unsigned active = __activemask();
  const std::uint32_t word = candidate >> 5u;
  const unsigned peers = __match_any_sync(active, word);
  const std::uint32_t bit = tombstone
      ? 1u << (candidate & 31u) : 0u;
  const std::uint32_t bits = __reduce_or_sync(peers, bit);
  const std::uint32_t leader = __ffs(peers) - 1u;
  if ((threadIdx.x & 31u) == leader && bits)
    atomicOr(tombstone_words + word, bits);
}

__device__ __forceinline__ bool candidate_token_index_less(
    std::uint16_t left, std::uint16_t right,
    const CandidateToken *candidate_tokens) {
  const CandidateToken a = candidate_tokens[left];
  const CandidateToken b = candidate_tokens[right];
  return a != b ? a < b : left < right;
}

__device__ __forceinline__ std::uint32_t candidate_token_merge_partition(
    const std::uint16_t *left, std::uint32_t left_count,
    const std::uint16_t *right, std::uint32_t right_count,
    std::uint32_t diagonal, const CandidateToken *candidate_tokens) {
  std::uint32_t low = diagonal > right_count
      ? diagonal - right_count : 0u;
  std::uint32_t high = min(diagonal, left_count);
  while (low <= high) {
    const std::uint32_t li = (low + high) >> 1u;
    const std::uint32_t ri = diagonal - li;
    if (li && ri < right_count && candidate_token_index_less(
            right[ri], left[li - 1u], candidate_tokens)) {
      high = li - 1u;
    } else if (ri && li < left_count && candidate_token_index_less(
                   left[li], right[ri - 1u], candidate_tokens)) {
      low = li + 1u;
    } else {
      return li;
    }
  }
  return low;
}

__device__ __forceinline__ std::uint32_t load_candidate_value(
    std::uint16_t candidate, CandidateToken token,
    std::uint32_t raw_count, bool raw_source_is_encoded,
    bool staged_epoch, bool crowded_piece,
    std::uint32_t raw_storage_begin, std::uint32_t q_begin,
    const std::uint32_t *source_offsets,
    const RawPayload *raw_payloads, const std::uint32_t *raw_offsets,
    std::uint32_t batch_stride, std::uint32_t pending_batches,
    const Row *staged_rows, ResidentRows arena,
    const RouteHeader *route_headers, const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    const std::uint16_t *crowded_level_begins) {
  if (candidate < raw_count) {
    if (staged_epoch || crowded_piece)
      return staged_rows[raw_storage_begin + candidate].value;
    const std::uint32_t batch = raw_source_is_encoded
        ? candidate_token_source(token)
        : direct_raw_batch_for_candidate(
              candidate, source_offsets, pending_batches);
    const std::uint32_t local = candidate - source_offsets[batch];
    const std::size_t offset =
        std::size_t{batch} * (kQuotients + 1u) + q_begin;
    return raw_payloads[
        std::size_t{batch} * batch_stride + raw_offsets[offset] + local]
        .value;
  }

  const std::uint32_t level = candidate_token_source(token) - 1u;
  const std::uint32_t q =
      q_begin + candidate_token_local_quotient(token);
  std::uint32_t position =
      candidate - source_offsets[kBatchesPerEpoch + level];
  if (crowded_piece) {
    position += crowded_level_begins[level];
  } else {
    const std::size_t level_base =
        std::size_t{level} * (kQuotients + 1u);
    position -= level_q_logical_offsets[level_base + q] -
        level_q_logical_offsets[level_base + q_begin];
  }
  return logical_section_row(
      q, level, position, arena, route_headers, route_slices,
      route_logical_begins, level_q_logical_offsets).value;
}

__device__ __forceinline__ std::uint32_t rank_cell_choose_source(
    std::uint32_t key0, std::uint32_t key1,
    std::uint32_t key2, std::uint32_t key3) {
  const std::uint32_t left = key1 < key0 ? 1u : 0u;
  const std::uint32_t left_key = left ? key1 : key0;
  const std::uint32_t right = key3 < key2 ? 3u : 2u;
  const std::uint32_t right_key = right == 3u ? key3 : key2;
  return right_key < left_key ? right : left;
}

__device__ __forceinline__ CellInputSlice rank_cell_level_slice(
    std::uint32_t q, std::uint32_t level, std::uint32_t cell,
    const DeviceManifestSnapshot &manifest,
    const Descriptor *descriptors,
    const std::uint32_t *level_cell_rank_blocks,
    const std::uint16_t *level_cell_ranks,
    const std::uint16_t *local_rank) {
  if (!level_is_occupied(manifest.occupied_level_mask, level)) return {};
  const std::uint32_t count =
      descriptors[descriptor_index(q, level)].count();
  CellInputSlice slice{};
  exact_cell_input_slice(
      q, level, cell, manifest.foundation_level, count, local_rank,
      level_cell_rank_blocks, level_cell_ranks, slice);
  return slice;
}

__global__ __launch_bounds__(kRankCellThreads)
void compact_rank_cell_sections_kernel(
    const ResidentPublicationPlan *plan,
    const std::uint64_t *raw_counts,
    const std::uint64_t *section_raw_offsets,
    const std::uint32_t *rank_cell_mode,
    const Row *staged_rows,
    const std::uint32_t *staged_offsets,
    const std::uint16_t *staged_ranks,
    ResidentRows arena,
    const DeviceManifest *manifests,
    const std::uint32_t *active_manifest,
    const Descriptor *descriptors,
    const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    std::uint32_t route_stride,
    RouteSlice *next_route_slices,
    std::uint32_t *section_output_counts,
    std::uint32_t *overflow_flag,
    const std::uint32_t *level_cell_rank_blocks,
    std::uint16_t *level_cell_ranks,
    std::uint16_t *local_rank,
    std::uint16_t *level_guides) {
  using BlockScan = cub::BlockScan<std::uint32_t, kRankCellThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::uint32_t source_counts[kRankCellMaximumSources];
  __shared__ std::uint64_t source_section_bases[
      kRankCellMaximumSources - 1u];
  __shared__ std::uint64_t source_cell_bases[
      (kRankCellMaximumSources - 1u) * kFoundationCells];
  __shared__ std::uint32_t survivor_prefix[kFoundationCells + 1u];
  __shared__ std::uint16_t tape_bases[kFoundationCells];
  __shared__ std::uint32_t section_raw_count;
  __shared__ std::uint32_t section_survivor_count;
  __shared__ std::uint32_t output_valid;
  extern __shared__ std::uint16_t survivor_tape[];
  if (plan->status || !*rank_cell_mode) return;
  const DeviceManifestSnapshot manifest = load_active_manifest(
      manifests, active_manifest);
  const std::uint32_t cell = threadIdx.x;

  for (std::uint32_t q = blockIdx.x; q < kQuotients; q += gridDim.x) {
    const std::uint32_t expected_raw = static_cast<std::uint32_t>(
        resident_work_count(raw_counts[q]));
    if (!expected_raw) continue;
    if (threadIdx.x == 0u) {
      source_counts[0] = staged_offsets[q + 1u] - staged_offsets[q];
      for (std::uint32_t source = 1u;
           source < kRankCellMaximumSources; ++source) {
        const std::uint32_t level = source - 1u;
        source_counts[source] =
            level <= plan->source_level_limit &&
            level_is_occupied(manifest.occupied_level_mask, level)
                ? descriptors[descriptor_index(q, level)].count() : 0u;
        if (source_counts[source]) {
          const RouteHeader header =
              route_headers[descriptor_index(q, level)];
          source_section_bases[level] =
              route_slices[header.begin].rows.offset();
        } else {
          source_section_bases[level] = 0u;
        }
      }
      section_raw_count = source_counts[0] + source_counts[1] +
          source_counts[2] + source_counts[3];
    }
    __syncthreads();

    const std::uint32_t staged_rank_begin =
        staged_ranks[std::size_t{q} * kFoundationCells + cell];
    const std::uint32_t staged_rank_end = cell + 1u < kFoundationCells
        ? staged_ranks[std::size_t{q} * kFoundationCells + cell + 1u]
        : source_counts[0];
    const std::uint32_t staged_q_begin = staged_offsets[q];
    const std::uint32_t begin0 = staged_q_begin + staged_rank_begin;
    const std::uint32_t end0 = staged_q_begin + staged_rank_end;
    const CellInputSlice slice1 = rank_cell_level_slice(
        q, 0u, cell, manifest, descriptors,
        level_cell_rank_blocks, level_cell_ranks, local_rank);
    const CellInputSlice slice2 = plan->source_level_limit >= 1u
        ? rank_cell_level_slice(
              q, 1u, cell, manifest, descriptors,
              level_cell_rank_blocks, level_cell_ranks, local_rank)
        : CellInputSlice{};
    const CellInputSlice slice3 = plan->source_level_limit >= 2u
        ? rank_cell_level_slice(
              q, 2u, cell, manifest, descriptors,
              level_cell_rank_blocks, level_cell_ranks, local_rank)
        : CellInputSlice{};
    const std::uint32_t begin1 = slice1.begin;
    const std::uint32_t begin2 = slice2.begin;
    const std::uint32_t begin3 = slice3.begin;
    const std::uint32_t end1 = begin1 + slice1.count;
    const std::uint32_t end2 = begin2 + slice2.count;
    const std::uint32_t end3 = begin3 + slice3.count;
    const std::uint32_t raw_count = end0 - begin0 + slice1.count +
        slice2.count + slice3.count;
    source_cell_bases[cell] = source_section_bases[0] + begin1;
    source_cell_bases[kFoundationCells + cell] =
        source_section_bases[1] + begin2;
    source_cell_bases[2u * kFoundationCells + cell] =
        source_section_bases[2] + begin3;

    std::uint32_t raw_prefix = 0u, raw_total = 0u;
    BlockScan(scan_storage).ExclusiveSum(raw_count, raw_prefix, raw_total);
    if (threadIdx.x == 0u) {
      output_valid = raw_total == expected_raw &&
          raw_total == section_raw_count &&
          raw_total + kRankCellPaddingPerCell * kFoundationCells <=
              kRankCellTapeEntries;
      if (!output_valid) atomicExch(overflow_flag, 1u);
    }
    __syncthreads();

    const std::uint32_t tape_begin = raw_prefix +
        kRankCellPaddingPerCell * cell;
    tape_bases[cell] = static_cast<std::uint16_t>(tape_begin);
    std::uint32_t position0 = begin0;
    std::uint32_t position1 = begin1;
    std::uint32_t position2 = begin2;
    std::uint32_t position3 = begin3;
    Row staged_head{};
    if (position0 < end0) staged_head = staged_rows[position0];
    std::uint32_t head_flags0 = position0 < end0
        ? std::uint32_t{staged_head.key} |
              (std::uint32_t{staged_head.flags} << 16u) : 0u;
    std::uint32_t head_flags1 = position1 < end1
        ? arena.key_flags[source_section_bases[0] + position1] : 0u;
    std::uint32_t head_flags2 = position2 < end2
        ? arena.key_flags[source_section_bases[1] + position2] : 0u;
    std::uint32_t head_flags3 = position3 < end3
        ? arena.key_flags[source_section_bases[2] + position3] : 0u;
    std::uint32_t key0 = position0 < end0
        ? head_flags0 & 0xffffu : 1u << 16u;
    std::uint32_t key1 = position1 < end1
        ? head_flags1 & 0xffffu : 1u << 16u;
    std::uint32_t key2 = position2 < end2
        ? head_flags2 & 0xffffu : 1u << 16u;
    std::uint32_t key3 = position3 < end3
        ? head_flags3 & 0xffffu : 1u << 16u;
    std::uint32_t survivor_count = 0u;
    std::uint32_t previous_key = 1u << 16u;

    for (std::uint32_t remaining = raw_count; remaining; --remaining) {
      const std::uint32_t source =
          rank_cell_choose_source(key0, key1, key2, key3);
      std::uint32_t source_position = 0u;
      std::uint32_t key_flags = 0u;
      if (source == 0u) {
        source_position = position0;
        key_flags = head_flags0;
        ++position0;
        if (position0 < end0) {
          staged_head = staged_rows[position0];
          head_flags0 = std::uint32_t{staged_head.key} |
              (std::uint32_t{staged_head.flags} << 16u);
          key0 = head_flags0 & 0xffffu;
        } else {
          key0 = 1u << 16u;
        }
      } else if (source == 1u) {
        source_position = position1;
        key_flags = head_flags1;
        ++position1;
        if (position1 < end1) {
          head_flags1 =
              arena.key_flags[source_section_bases[0] + position1];
          key1 = head_flags1 & 0xffffu;
        } else {
          key1 = 1u << 16u;
        }
      } else if (source == 2u) {
        source_position = position2;
        key_flags = head_flags2;
        ++position2;
        if (position2 < end2) {
          head_flags2 =
              arena.key_flags[source_section_bases[1] + position2];
          key2 = head_flags2 & 0xffffu;
        } else {
          key2 = 1u << 16u;
        }
      } else {
        source_position = position3;
        key_flags = head_flags3;
        ++position3;
        if (position3 < end3) {
          head_flags3 =
              arena.key_flags[source_section_bases[2] + position3];
          key3 = head_flags3 & 0xffffu;
        } else {
          key3 = 1u << 16u;
        }
      }
      const std::uint32_t key = key_flags & 0xffffu;
      if (key != previous_key) {
        const bool tombstone = (key_flags >> 16u) & kTombstone;
        if (plan->keep_tombstones || !tombstone) {
          const std::uint32_t source_begin = source == 0u ? begin0 :
              source == 1u ? begin1 : source == 2u ? begin2 : begin3;
          const std::uint32_t local_position =
              source_position - source_begin;
          const std::uint32_t reference =
              (source << kRankCellReferenceBits) | local_position;
          if (local_position > kRankCellReferenceMask ||
              tape_begin + survivor_count >= kRankCellTapeEntries) {
            atomicExch(overflow_flag, 1u);
            atomicExch(&output_valid, 0u);
          } else {
            survivor_tape[tape_begin + survivor_count++] =
                static_cast<std::uint16_t>(reference);
          }
        }
        previous_key = key;
      }
    }
    __syncthreads();

    std::uint32_t output_prefix = 0u, output_total = 0u;
    BlockScan(scan_storage).ExclusiveSum(
        survivor_count, output_prefix, output_total);
    survivor_prefix[cell] = output_prefix;
    if (threadIdx.x == 0u) {
      survivor_prefix[kFoundationCells] = output_total;
      section_survivor_count = output_total;
      const std::uint64_t reservation_begin = section_raw_offsets[q];
      const std::uint64_t reservation_end = section_raw_offsets[q + 1u];
      output_valid &= output_total <= reservation_end - reservation_begin &&
          plan->output_begin + reservation_end <=
              plan->output_begin + plan->output_capacity;
      if (!output_valid) atomicExch(overflow_flag, 1u);
    }
    __syncthreads();

    const std::uint64_t output_begin =
        plan->output_begin + section_raw_offsets[q];
    for (std::uint32_t logical = threadIdx.x;
         logical < section_survivor_count; logical += blockDim.x) {
      std::uint32_t low = 0u, high = kFoundationCells;
      while (low + 1u < high) {
        const std::uint32_t middle = (low + high) >> 1u;
        if (survivor_prefix[middle] <= logical) low = middle;
        else high = middle;
      }
      const std::uint16_t reference = survivor_tape[
          std::uint32_t{tape_bases[low]} + logical - survivor_prefix[low]];
      const std::uint32_t source =
          reference >> kRankCellReferenceBits;
      const std::uint32_t local_position =
          reference & kRankCellReferenceMask;
      Row row{};
      if (source == 0u) {
        const std::uint32_t cell_begin =
            staged_ranks[std::size_t{q} * kFoundationCells + low];
        row = staged_rows[staged_q_begin + cell_begin + local_position];
      } else {
        row = arena[source_cell_bases[
            (source - 1u) * kFoundationCells + low] + local_position];
      }
      if (output_valid) arena.store(output_begin + logical, row);
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
      section_output_counts[q] = section_survivor_count;
      const std::uint32_t route =
          plan->destination_level * route_stride + q;
      next_route_slices[route] = {
          output_valid
              ? Descriptor::make(output_begin, section_survivor_count)
              : Descriptor{}, 0u, 1u << 16u};
    }
    if (plan->destination_is_foundation) {
      local_rank[std::size_t{q} * kFoundationCells + cell] =
          static_cast<std::uint16_t>(output_prefix);
    } else {
      const std::uint32_t rank_block = level_cell_rank_blocks[
          descriptor_index(q, plan->destination_level)];
      if (rank_block != kInvalid)
        level_cell_ranks[
            std::size_t{rank_block} * kFoundationCells + cell] =
            static_cast<std::uint16_t>(output_prefix);
    }
    __syncthreads();
    if (!plan->destination_is_foundation &&
        threadIdx.x < kGuideSamples &&
        section_survivor_count >= kGuideRegions) {
      const std::uint32_t position =
          (threadIdx.x + 1u) * section_survivor_count / kGuideRegions;
      level_guides[guide_index(
          q, plan->destination_level) + threadIdx.x] =
          arena.key_at(output_begin + position);
    }
    __syncthreads();
  }
}

// Merge one adaptive quotient range per CTA.
__global__ void compact_direct_epoch_jobs_kernel(
    BalancedMergeJob *jobs, const ResidentPublicationPlan *plan,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const std::uint32_t *staged_keys,
    const Row *staged_rows, const std::uint32_t *staged_offsets,
    const std::uint32_t *staged_epoch_mode,
    const std::uint32_t *rank_cell_mode, ResidentRows arena,
    const DeviceManifest *manifests, const std::uint32_t *active_manifest,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    const RouteHeader *next_route_headers, RouteSlice *next_route_slices,
    std::uint32_t *section_output_counts, std::uint32_t *overflow_flag,
    std::uint32_t *level_cell_rank_blocks,
    std::uint16_t *level_cell_ranks, std::uint16_t *local_rank,
    std::uint16_t *level_guides
#if defined(GPULSMOPT_FORCE_UNIFIED_MERGE) && \
    !defined(GPULSMOPT_FORCE_UNIFIED_COMPILE_ELIDE)
    , bool force_unified_merge
#endif
    ) {
  constexpr std::uint32_t kThreads = kFoundationCompactionThreads;
  using BlockScan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage block_scan_storage;
  extern __shared__ __align__(16) unsigned char merge_workspace[];
  const std::uint32_t capacity = plan->job_capacity;
  CandidateToken *candidate_tokens =
      reinterpret_cast<CandidateToken *>(merge_workspace);
  std::uint16_t *indices_a = reinterpret_cast<std::uint16_t *>(
      candidate_tokens + capacity);
  std::uint16_t *indices_b = indices_a + capacity + 1u;
  std::uint32_t *tombstone_words = reinterpret_cast<std::uint32_t *>(
      indices_b + capacity + 1u);
  const std::uint32_t tombstone_word_capacity = (capacity + 31u) / 32u;
  std::uint16_t *cell_input_offsets =
      reinterpret_cast<std::uint16_t *>(
          tombstone_words + tombstone_word_capacity);
  std::uint16_t *cell_raw_counts =
      cell_input_offsets + kCellOwnedCells + 2u;
  std::uint16_t *cell_output_counts =
      cell_raw_counts + kCellOwnedCells;
  std::uint16_t *cell_queue = cell_output_counts + kCellOwnedCells;
  std::uint16_t *cell_output_offsets = cell_queue + kCellOwnedCells;
  __shared__ std::uint32_t source_offsets[kBatchesPerEpoch +
                                           kMaximumLevels + 2u];
  // Preserve crowded per-level start positions.
  __shared__ std::uint16_t crowded_level_begins[kMaximumLevels + 2u];
  __shared__ std::uint16_t run_offsets[kMaximumLevels + 2u];
  __shared__ std::uint16_t run_lengths[kMaximumLevels + 2u];
  __shared__ std::uint16_t run_sources[kMaximumLevels + 2u];
  __shared__ std::uint32_t raw_count_shared;
  __shared__ std::uint32_t task_rows_shared;
  __shared__ std::uint32_t resolved_raw_count_shared;
  __shared__ std::uint32_t run_count_shared;
  __shared__ std::uint32_t small_count_shared;
  __shared__ std::uint32_t largest_source_offset_shared;
  __shared__ std::uint32_t largest_count_shared;
  __shared__ std::uint32_t reorder_sources_shared;
  __shared__ std::uint32_t task_output_count_shared;
  __shared__ std::uint32_t output_valid_shared;
  __shared__ std::uint32_t metadata_begin_shared;
  __shared__ std::uint32_t metadata_count_shared;
  __shared__ std::uint32_t large_cell_count_shared;
  __shared__ std::uint32_t raw_storage_begin_shared;
  if (plan->status || *rank_cell_mode) return;
  const DeviceManifestSnapshot manifest = load_active_manifest(
      manifests, active_manifest);
  const bool staged_epoch = __ldg(staged_epoch_mode) != 0u;

  for (std::uint32_t job_index = blockIdx.x;
       job_index < plan->job_count; job_index += gridDim.x) {
    const BalancedMergeJob job = jobs[job_index];
    const std::uint32_t q_begin = job.quotient_begin;
    const std::uint32_t q_end = job.quotient_end;
    const std::uint32_t quotient_count = q_end - q_begin;
    const bool crowded_piece = job.hot_pieces != 0u;
#if defined(GPULSMOPT_FORCE_UNIFIED_MERGE) && \
    !defined(GPULSMOPT_FORCE_UNIFIED_COMPILE_ELIDE)
    // A kernel argument keeps the production cell-owned implementation in
    // the compiled resource shape while the experiment forces it off.
    const bool cell_owned_shape =
        !force_unified_merge && !crowded_piece &&
        quotient_count <= kCellOwnedQuotients;
#else
    const bool cell_owned_shape =
        !kForceUnifiedMergeExperiment && !crowded_piece &&
        quotient_count <= kCellOwnedQuotients;
#endif
    const bool updates_are_resolved = staged_epoch || crowded_piece;
    if (threadIdx.x == 0u) {
      std::uint32_t total = 0u;
      if (crowded_piece) {
        const std::uint32_t suffix_begin =
            static_cast<std::uint32_t>(job.key_begin & 0xffffu);
        const std::uint32_t suffix_end = static_cast<std::uint32_t>(
            job.key_end - (std::uint64_t{q_begin} << 16u));
        const std::uint32_t hot_begin = staged_offsets[q_begin];
        const std::uint32_t hot_count =
            staged_offsets[q_begin + 1u] - hot_begin;
        const std::uint32_t begin = lower_bound_rows(
            staged_rows + hot_begin, hot_count, suffix_begin);
        const std::uint32_t end = lower_bound_rows(
            staged_rows + hot_begin, hot_count, suffix_end);
        source_offsets[0] = 0u;
        raw_storage_begin_shared = hot_begin + begin;
        total = end - begin;
        for (std::uint32_t batch = 1u; batch <= kBatchesPerEpoch; ++batch)
          source_offsets[batch] = total;
      } else if (staged_epoch) {
        source_offsets[0] = 0u;
        raw_storage_begin_shared = staged_offsets[q_begin];
        total = staged_offsets[q_end] - staged_offsets[q_begin];
        for (std::uint32_t batch = 1u; batch <= kBatchesPerEpoch; ++batch)
          source_offsets[batch] = total;
      } else {
        raw_storage_begin_shared = 0u;
        for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
          source_offsets[batch] = total;
          const std::size_t base =
              std::size_t{batch} * (kQuotients + 1u);
          total += raw_offsets[base + q_end] - raw_offsets[base + q_begin];
        }
        for (std::uint32_t batch = pending_batches;
             batch <= kBatchesPerEpoch; ++batch)
          source_offsets[batch] = total;
      }
      raw_count_shared = total;
      for (std::uint32_t level = 0u;
           level <= plan->source_level_limit; ++level) {
        crowded_level_begins[level] = 0u;
        source_offsets[kBatchesPerEpoch + level] = total;
        if (level_is_occupied(manifest.occupied_level_mask, level)) {
          if (crowded_piece) {
            const std::uint32_t low = static_cast<std::uint32_t>(
                job.key_begin & 0xffffu);
            const std::uint32_t high = static_cast<std::uint32_t>(
                job.key_end - (std::uint64_t{q_begin} << 16u));
            const std::uint32_t begin = logical_section_bound(
                q_begin, level, low, false, arena, route_headers,
                route_slices, route_logical_begins,
                level_q_logical_offsets);
            const std::uint32_t end = logical_section_bound(
                q_begin, level, high, false, arena, route_headers,
                route_slices, route_logical_begins,
                level_q_logical_offsets);
            // Suffix range starts fit in 16 bits.
            crowded_level_begins[level] =
                static_cast<std::uint16_t>(begin);
            total += end - begin;
          } else {
            for (std::uint32_t q = q_begin; q < q_end; ++q)
              total += descriptors[descriptor_index(q, level)].count();
          }
        }
        source_offsets[kBatchesPerEpoch + level + 1u] = total;
      }
      task_rows_shared = total;
    }
    __syncthreads();
    const std::uint32_t task_rows = task_rows_shared;
    const std::uint32_t raw_count = raw_count_shared;
    const bool cell_owned =
        cell_owned_shape && task_rows > raw_count;
    if (!task_rows) {
      __syncthreads();
      continue;
    }
    if (task_rows > capacity) {
      if (threadIdx.x == 0u) atomicExch(overflow_flag, 1u);
      __syncthreads();
      continue;
    }

    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t warp = threadIdx.x >> 5u;
    if (cell_owned && !staged_epoch && !crowded_piece) {
      const std::uint32_t words = (task_rows + 31u) / 32u;
      for (std::uint32_t word = threadIdx.x; word < words;
           word += blockDim.x)
        tombstone_words[word] = 0u;
      __syncthreads();

      // Assign complete raw batches to warps.
      for (std::uint32_t batch = warp; batch < pending_batches;
           batch += blockDim.x / 32u) {
        const std::uint32_t batch_begin = source_offsets[batch];
        const std::uint32_t batch_end = source_offsets[batch + 1u];
        const std::size_t base =
            std::size_t{batch} * (kQuotients + 1u);
        const std::size_t source_begin =
            std::size_t{batch} * batch_stride +
            raw_offsets[base + q_begin];
        for (std::uint32_t candidate = batch_begin + lane;
             candidate < batch_end; candidate += 32u) {
          const std::size_t source =
              source_begin + candidate - batch_begin;
          const std::uint32_t key = raw_keys[source];
          const RawPayload payload = raw_payloads[source];
          candidate_tokens[candidate] = make_candidate_token(
              (key >> 16u) - q_begin, key, batch);
          indices_a[candidate] = static_cast<std::uint16_t>(candidate);
          record_candidate_tombstone(
              candidate, (payload.metadata & kRawTombstone) != 0u,
              tombstone_words);
        }
      }

      // Load resident rows in one CTA-wide pass.
      for (std::uint32_t candidate = raw_count + threadIdx.x;
           candidate < task_rows; candidate += blockDim.x) {
        std::uint32_t level = 0u;
        while (level <= plan->source_level_limit &&
               candidate >=
                   source_offsets[kBatchesPerEpoch + level + 1u])
          ++level;
        std::uint32_t position = candidate -
            source_offsets[kBatchesPerEpoch + level];
        std::uint32_t q = q_begin;
        while (q < q_end) {
          const std::uint32_t count =
              descriptors[descriptor_index(q, level)].count();
          if (position < count) break;
          position -= count;
          ++q;
        }
        Row row = logical_section_row(
            q, level, position, arena, route_headers, route_slices,
            route_logical_begins, level_q_logical_offsets);
        const bool tombstone = (row.flags & kTombstone) != 0u;
        candidate_tokens[candidate] = make_candidate_token(
            q - q_begin, row.key, level + 1u);
        indices_a[candidate] = static_cast<std::uint16_t>(candidate);
        record_candidate_tombstone(
            candidate, tombstone, tombstone_words);
      }
    } else {
      const std::uint32_t words = (task_rows + 31u) / 32u;
      for (std::uint32_t word = warp; word < words;
           word += blockDim.x / 32u) {
        const std::uint32_t candidate = word * 32u + lane;
        bool tombstone = false;
        if (candidate < task_rows) {
          CandidateToken token{};
          if (candidate < raw_count) {
            if (crowded_piece) {
              const std::uint32_t suffix_begin =
                  static_cast<std::uint32_t>(job.key_begin & 0xffffu);
              const std::uint32_t hot_begin = staged_offsets[q_begin];
              const std::uint32_t hot_count =
                  staged_offsets[q_begin + 1u] - hot_begin;
              const std::uint32_t begin = lower_bound_rows(
                  staged_rows + hot_begin, hot_count, suffix_begin);
              const Row row = staged_rows[hot_begin + begin + candidate];
              tombstone = (row.flags & kTombstone) != 0u;
              token = make_candidate_token(0u, row.key, 0u);
            } else if (staged_epoch) {
              const std::uint32_t source =
                  staged_offsets[q_begin] + candidate;
              const std::uint32_t key = staged_keys[source];
              const Row row = staged_rows[source];
              tombstone = (row.flags & kTombstone) != 0u;
              token = make_candidate_token(
                  (key >> 16u) - q_begin, key, 0u);
            } else {
              const std::uint32_t batch = direct_raw_batch_for_candidate(
                  static_cast<std::uint16_t>(candidate),
                  source_offsets, pending_batches);
              const std::uint32_t local =
                  candidate - source_offsets[batch];
              const std::size_t base =
                  std::size_t{batch} * (kQuotients + 1u);
              const std::size_t source =
                  std::size_t{batch} * batch_stride +
                  raw_offsets[base + q_begin] + local;
              const RawPayload payload = raw_payloads[source];
              const std::uint32_t key = raw_keys[source];
              tombstone = (payload.metadata & kRawTombstone) != 0u;
              token = make_candidate_token(
                  (key >> 16u) - q_begin, key, batch);
            }
          } else {
            std::uint32_t level = 0u;
            while (level <= plan->source_level_limit &&
                   candidate >=
                       source_offsets[kBatchesPerEpoch + level + 1u])
              ++level;
            std::uint32_t position = candidate -
                source_offsets[kBatchesPerEpoch + level];
            std::uint32_t q = q_begin;
            if (crowded_piece) {
              position += crowded_level_begins[level];
            } else while (q < q_end) {
              const std::uint32_t count =
                  descriptors[descriptor_index(q, level)].count();
              if (position < count) break;
              position -= count;
              ++q;
            }
            const Row row = logical_section_row(
                q, level, position, arena, route_headers, route_slices,
                route_logical_begins, level_q_logical_offsets);
            tombstone = (row.flags & kTombstone) != 0u;
            token = make_candidate_token(
                q - q_begin, row.key, level + 1u);
          }
          candidate_tokens[candidate] = token;
          indices_a[candidate] = static_cast<std::uint16_t>(candidate);
        }
        const std::uint32_t mask = __ballot_sync(0xffffffffu, tombstone);
        if (lane == 0u) tombstone_words[word] = mask;
      }
    }
    __syncthreads();

    if (cell_owned) {
      const std::uint32_t total_cells =
          quotient_count * kFoundationCells;
      std::uint32_t *cell_cursors =
          reinterpret_cast<std::uint32_t *>(indices_b);

      // Let each cell pull sorted resident slices.
      for (std::uint32_t cell = threadIdx.x; cell < total_cells;
           cell += blockDim.x) {
        const std::uint32_t local_q = cell / kFoundationCells;
        const std::uint32_t q = q_begin + local_q;
        const std::uint32_t local_cell =
            cell & (kFoundationCells - 1u);
        std::uint32_t resident_count = 0u;
        for (std::uint32_t level = 0u;
             level <= plan->source_level_limit; ++level) {
          const std::uint32_t level_begin =
              source_offsets[kBatchesPerEpoch + level];
          const std::uint32_t level_end =
              source_offsets[kBatchesPerEpoch + level + 1u];
          if (level_begin == level_end) continue;
          std::uint32_t q_begin_in_level = level_begin;
          for (std::uint32_t previous = q_begin; previous < q; ++previous)
            q_begin_in_level +=
                descriptors[descriptor_index(previous, level)].count();
          const std::uint32_t q_count =
              descriptors[descriptor_index(q, level)].count();
          resident_count += resident_candidate_cell_slice(
              q, level, local_cell, manifest.foundation_level, q_count,
              q_begin_in_level, candidate_tokens, local_rank,
              level_cell_rank_blocks, level_cell_ranks).count;
        }
        cell_cursors[cell] = resident_count;
      }
      __syncthreads();

      // Histogram only unsorted raw updates.
      for (std::uint32_t candidate = threadIdx.x; candidate < raw_count;
           candidate += blockDim.x) {
        const CandidateToken token = candidate_tokens[candidate];
        const std::uint32_t local_q =
            candidate_token_local_quotient(token);
        const std::uint32_t cell =
            local_q * kFoundationCells +
            candidate_token_key(token) / kFoundationCellKeys;
        const unsigned active = __activemask();
        const unsigned peers = __match_any_sync(active, cell);
        const std::uint32_t leader = __ffs(peers) - 1u;
        if (lane == leader)
          atomicAdd(cell_cursors + cell,
                    __popc(peers) | (__popc(peers) << 16u));
      }
      __syncthreads();

      if (threadIdx.x == 0u) {
        std::uint32_t prefix = 0u;
        for (std::uint32_t cell = 0u; cell < total_cells; ++cell) {
          const std::uint32_t packed_count = cell_cursors[cell];
          cell_input_offsets[cell] = static_cast<std::uint16_t>(prefix);
          prefix += packed_count & 0xffffu;
          cell_raw_counts[cell] = static_cast<std::uint16_t>(
              packed_count >> 16u);
          cell_cursors[cell] = 0u;
        }
        cell_input_offsets[total_cells] =
            static_cast<std::uint16_t>(prefix);
      }
      __syncthreads();

      for (std::uint32_t candidate = threadIdx.x; candidate < raw_count;
           candidate += blockDim.x) {
        const CandidateToken token = candidate_tokens[candidate];
        const std::uint32_t local_q =
            candidate_token_local_quotient(token);
        const std::uint32_t cell =
            local_q * kFoundationCells +
            candidate_token_key(token) / kFoundationCellKeys;
        const unsigned active = __activemask();
        const unsigned peers = __match_any_sync(active, cell);
        const std::uint32_t leader = __ffs(peers) - 1u;
        std::uint32_t base = 0u;
        if (lane == leader)
          base = atomicAdd(cell_cursors + cell, __popc(peers));
        base = __shfl_sync(peers, base, leader);
        const std::uint32_t rank =
            base + __popc(peers & ((1u << lane) - 1u));
        indices_a[cell_input_offsets[cell] + rank] =
            static_cast<std::uint16_t>(candidate);
      }
      __syncthreads();

      // Pull each cell's resident slices directly.
      if (threadIdx.x < total_cells) {
        const std::uint32_t cell = threadIdx.x;
        const std::uint32_t total_cell_count =
            cell_input_offsets[cell + 1u] - cell_input_offsets[cell];
        if (total_cell_count > kCellOwnedWarpMaximum) {
        const std::uint32_t local_q = cell / kFoundationCells;
        const std::uint32_t q = q_begin + local_q;
        const std::uint32_t local_cell =
            cell & (kFoundationCells - 1u);
        std::uint32_t destination =
            cell_input_offsets[cell] + cell_raw_counts[cell];
        for (std::uint32_t level = 0u;
             level <= plan->source_level_limit; ++level) {
          const std::uint32_t level_begin =
              source_offsets[kBatchesPerEpoch + level];
          const std::uint32_t level_end =
              source_offsets[kBatchesPerEpoch + level + 1u];
          if (level_begin == level_end) continue;
          std::uint32_t q_begin_in_level = level_begin;
          for (std::uint32_t previous = q_begin; previous < q; ++previous)
            q_begin_in_level +=
                descriptors[descriptor_index(previous, level)].count();
          const std::uint32_t q_count =
              descriptors[descriptor_index(q, level)].count();
          const CellInputSlice slice = resident_candidate_cell_slice(
              q, level, local_cell, manifest.foundation_level, q_count,
              q_begin_in_level, candidate_tokens, local_rank,
              level_cell_rank_blocks, level_cell_ranks);
          for (std::uint32_t position = 0u; position < slice.count;
               ++position)
            indices_a[destination++] = static_cast<std::uint16_t>(
                slice.begin + position);
        }
        }
      }
      __syncthreads();

      const std::uint32_t bucket_slots =
          quotient_count * kCellOwnedCostBuckets;
      // Reuse the second index plane for queue scratch.
      for (std::uint32_t slot = threadIdx.x; slot <= bucket_slots;
           slot += blockDim.x)
        cell_cursors[slot] = 0u;
      for (std::uint32_t cell = threadIdx.x; cell < total_cells;
           cell += blockDim.x)
        cell_output_counts[cell] = 0u;
      __syncthreads();

      const bool owns_queue_cell = threadIdx.x < total_cells;
      std::uint32_t queue_slot = bucket_slots;
      if (owns_queue_cell) {
        const std::uint32_t cell = threadIdx.x;
        const std::uint32_t total_count =
            cell_input_offsets[cell + 1u] - cell_input_offsets[cell];
        if (total_count <= kCellOwnedWarpMaximum) {
          const std::uint32_t local_q = cell / kFoundationCells;
          queue_slot = local_q * kCellOwnedCostBuckets +
              cell_owned_cost_bucket(total_count, cell_raw_counts[cell]);
        }
      }
      const unsigned queue_active = __ballot_sync(
          0xffffffffu, owns_queue_cell);
      if (owns_queue_cell) {
        const unsigned peers = __match_any_sync(queue_active, queue_slot);
        const std::uint32_t leader = __ffs(peers) - 1u;
        if (lane == leader)
          atomicAdd(cell_cursors + queue_slot, __popc(peers));
      }
      __syncthreads();

      if (threadIdx.x == 0u) {
        std::uint32_t prefix = 0u;
        for (std::uint32_t slot = 0u; slot <= bucket_slots; ++slot) {
          const std::uint32_t count = cell_cursors[slot];
          cell_cursors[slot] = prefix;
          prefix += count;
          if (slot == bucket_slots) large_cell_count_shared = count;
        }
      }
      __syncthreads();

      if (owns_queue_cell) {
        const unsigned peers = __match_any_sync(queue_active, queue_slot);
        const std::uint32_t leader = __ffs(peers) - 1u;
        std::uint32_t base = 0u;
        if (lane == leader)
          base = atomicAdd(cell_cursors + queue_slot, __popc(peers));
        base = __shfl_sync(peers, base, leader);
        const std::uint32_t rank =
            base + __popc(peers & ((1u << lane) - 1u));
        cell_queue[rank] = static_cast<std::uint16_t>(threadIdx.x);
      }
      __syncthreads();

      // Give rare oversized cells the full CTA.
      for (std::uint32_t large_index = 0u;
           large_index < large_cell_count_shared; ++large_index) {
        const std::uint32_t cell = cell_queue[
            total_cells - large_cell_count_shared + large_index];
        const std::uint32_t cell_begin = cell_input_offsets[cell];
        const std::uint32_t cell_count =
            cell_input_offsets[cell + 1u] - cell_begin;
        bool cell_input_is_a = true;
        for (std::uint32_t width = 1u; width < cell_count; width <<= 1u) {
          const std::uint16_t *input =
              cell_input_is_a ? indices_a : indices_b;
          std::uint16_t *output =
              cell_input_is_a ? indices_b : indices_a;
          const std::uint32_t items =
              (cell_count + kThreads - 1u) / kThreads;
          std::uint32_t position = threadIdx.x * items;
          const std::uint32_t thread_end = min(position + items, cell_count);
          while (position < thread_end) {
            const std::uint32_t pair_begin =
                (position / (width * 2u)) * width * 2u;
            const std::uint32_t left_count =
                min(width, cell_count - pair_begin);
            const std::uint32_t right_begin = pair_begin + left_count;
            const std::uint32_t right_count = right_begin < cell_count
                ? min(width, cell_count - right_begin) : 0u;
            const std::uint32_t pair_end = right_begin + right_count;
            const std::uint32_t output_end = min(thread_end, pair_end);
            if (!right_count) {
              while (position < output_end) {
                output[cell_begin + position] =
                    input[cell_begin + position];
                ++position;
              }
              continue;
            }
            const std::uint16_t *left = input + cell_begin + pair_begin;
            const std::uint16_t *right = input + cell_begin + right_begin;
            const std::uint32_t diagonal = position - pair_begin;
            std::uint32_t li = cell_owned_merge_partition(
                left, left_count, right, right_count, diagonal,
                candidate_tokens);
            std::uint32_t ri = diagonal - li;
            while (position < output_end) {
              const bool take_left = ri >= right_count ||
                  (li < left_count && cell_owned_candidate_less(
                       left[li], right[ri], candidate_tokens));
              output[cell_begin + position++] =
                  take_left ? left[li++] : right[ri++];
            }
          }
          __syncthreads();
          cell_input_is_a = !cell_input_is_a;
        }

        std::uint16_t *sorted =
            cell_input_is_a ? indices_a : indices_b;
        std::uint16_t *resolved =
            cell_input_is_a ? indices_b : indices_a;
        const std::uint32_t items =
            (cell_count + kThreads - 1u) / kThreads;
        std::uint32_t winner_mask = 0u, local_winners = 0u;
        for (std::uint32_t item = 0u; item < items; ++item) {
          const std::uint32_t index = threadIdx.x * items + item;
          if (index >= cell_count) continue;
          const std::uint16_t candidate = sorted[cell_begin + index];
          const bool first = index == 0u ||
              candidate_token_key(candidate_tokens[
                  sorted[cell_begin + index - 1u]]) !=
                  candidate_token_key(candidate_tokens[candidate]);
          if (!first) continue;
          std::uint16_t winner = candidate;
          std::uint32_t next = index + 1u;
          while (next < cell_count &&
                 candidate_token_key(candidate_tokens[
                     sorted[cell_begin + next]]) ==
                     candidate_token_key(candidate_tokens[candidate])) {
            const std::uint16_t other = sorted[cell_begin + next++];
            if (cell_owned_candidate_is_newer(
                    other, winner, candidate_tokens, raw_count,
                    updates_are_resolved, q_begin, source_offsets,
                    raw_payloads, raw_offsets, batch_stride,
                    pending_batches)) {
              winner = other;
            }
          }
          if (!plan->keep_tombstones &&
              cell_owned_candidate_is_tombstone(
                  winner, tombstone_words))
            continue;
          resolved[cell_begin + index] = winner;
          winner_mask |= 1u << item;
          ++local_winners;
        }
        std::uint32_t cell_thread_base{}, cell_winners{};
        BlockScan(block_scan_storage).ExclusiveSum(
            local_winners, cell_thread_base, cell_winners);
        std::uint32_t local_rank = 0u;
        for (std::uint32_t item = 0u; item < items; ++item) {
          if (!(winner_mask & (1u << item))) continue;
          const std::uint32_t index = threadIdx.x * items + item;
          sorted[cell_begin + cell_thread_base + local_rank++] =
              resolved[cell_begin + index];
        }
        __syncthreads();
        if (sorted != indices_a) {
          for (std::uint32_t i = threadIdx.x; i < cell_winners;
               i += blockDim.x)
            indices_a[cell_begin + i] = sorted[cell_begin + i];
        }
        if (threadIdx.x == 0u)
          cell_output_counts[cell] =
              static_cast<std::uint16_t>(cell_winners);
        __syncthreads();
      }

      // Sort raw updates, then merge resident slices.
      const std::uint32_t processing_cell = threadIdx.x < total_cells
          ? cell_queue[threadIdx.x] : total_cells;
      if (processing_cell < total_cells) {
        const std::uint32_t cell_begin =
            cell_input_offsets[processing_cell];
        const std::uint32_t cell_count =
            cell_input_offsets[processing_cell + 1u] - cell_begin;
        if (cell_count && cell_count <= kCellOwnedWarpMaximum) {
          const std::uint32_t raw_cell_count =
              cell_raw_counts[processing_cell];
          for (std::uint32_t i = 1u; i < raw_cell_count; ++i) {
            const std::uint16_t value = indices_a[cell_begin + i];
            std::uint32_t j = i;
            while (j && cell_owned_candidate_less(
                            value, indices_a[cell_begin + j - 1u],
                            candidate_tokens)) {
              indices_a[cell_begin + j] = indices_a[cell_begin + j - 1u];
              --j;
            }
            indices_a[cell_begin + j] = value;
          }
          std::uint32_t current_count = 0u;
          for (std::uint32_t i = 0u; i < raw_cell_count;) {
            std::uint16_t winner = indices_a[cell_begin + i];
            std::uint32_t next = i + 1u;
            while (next < raw_cell_count &&
                   candidate_token_key(candidate_tokens[
                       indices_a[cell_begin + next]]) ==
                       candidate_token_key(candidate_tokens[winner])) {
              const std::uint16_t other = indices_a[cell_begin + next++];
              if (cell_owned_candidate_is_newer(
                      other, winner, candidate_tokens, raw_count,
                      updates_are_resolved, q_begin, source_offsets,
                      raw_payloads, raw_offsets, batch_stride,
                      pending_batches)) {
                winner = other;
              }
            }
            // Keep tombstones until older levels merge.
            indices_a[cell_begin + current_count++] = winner;
            i = next;
          }
          bool current_is_a = true;
          const std::uint32_t local_q =
              processing_cell / kFoundationCells;
          const std::uint32_t q = q_begin + local_q;
          const std::uint32_t local_cell =
              processing_cell & (kFoundationCells - 1u);
          for (std::uint32_t level = 0u;
               level <= plan->source_level_limit; ++level) {
            const std::uint32_t level_begin =
                source_offsets[kBatchesPerEpoch + level];
            const std::uint32_t level_end =
                source_offsets[kBatchesPerEpoch + level + 1u];
            if (level_begin == level_end) continue;
            std::uint32_t q_begin_in_level = level_begin;
            for (std::uint32_t previous = q_begin;
                 previous < q; ++previous)
              q_begin_in_level +=
                  descriptors[descriptor_index(previous, level)].count();
            const std::uint32_t q_count =
                descriptors[descriptor_index(q, level)].count();
            const CellInputSlice resident = resident_candidate_cell_slice(
                q, level, local_cell, manifest.foundation_level, q_count,
                q_begin_in_level, candidate_tokens, local_rank,
                level_cell_rank_blocks, level_cell_ranks);
            const std::uint32_t resident_count = resident.count;
            if (!resident_count) continue;
            const std::uint16_t *current =
                current_is_a ? indices_a : indices_b;
            std::uint16_t *next_run = current_is_a ? indices_b : indices_a;
            std::uint32_t current_position = 0u;
            std::uint32_t resident_position = 0u;
            std::uint32_t merged = 0u;
            while (current_position < current_count ||
                   resident_position < resident_count) {
              if (resident_position == resident_count) {
                next_run[cell_begin + merged++] =
                    current[cell_begin + current_position++];
                continue;
              }
              const std::uint16_t resident_candidate =
                  static_cast<std::uint16_t>(
                      resident.begin + resident_position);
              if (current_position == current_count) {
                next_run[cell_begin + merged++] = resident_candidate;
                ++resident_position;
                continue;
              }
              const std::uint16_t current_candidate =
                  current[cell_begin + current_position];
              const std::uint16_t current_key =
                  candidate_token_key(candidate_tokens[current_candidate]);
              const std::uint16_t resident_key =
                  candidate_token_key(candidate_tokens[resident_candidate]);
              if (current_key <= resident_key) {
                next_run[cell_begin + merged++] = current_candidate;
                ++current_position;
                if (current_key == resident_key) ++resident_position;
              } else {
                next_run[cell_begin + merged++] = resident_candidate;
                ++resident_position;
              }
            }
            current_count = merged;
            current_is_a = !current_is_a;
          }
          const std::uint16_t *final_run =
              current_is_a ? indices_a : indices_b;
          std::uint32_t output = 0u;
          for (std::uint32_t i = 0u; i < current_count; ++i) {
            const std::uint16_t candidate = final_run[cell_begin + i];
            if (plan->keep_tombstones ||
                !cell_owned_candidate_is_tombstone(
                    candidate, tombstone_words))
              indices_a[cell_begin + output++] = candidate;
          }
          cell_output_counts[processing_cell] =
              static_cast<std::uint16_t>(output);
        }
      }
      __syncthreads();

      const std::uint32_t owned_count = threadIdx.x < total_cells
          ? cell_output_counts[threadIdx.x] : 0u;
      std::uint32_t owned_output_begin{}, cell_total_output{};
      BlockScan(block_scan_storage).ExclusiveSum(
          owned_count, owned_output_begin, cell_total_output);
      if (threadIdx.x < total_cells)
        cell_output_offsets[threadIdx.x] =
            static_cast<std::uint16_t>(owned_output_begin);
      if (threadIdx.x == 0u) {
        cell_output_offsets[total_cells] =
            static_cast<std::uint16_t>(cell_total_output);
        const bool valid = cell_total_output <= job.existing_capacity &&
            job.existing_offset + job.existing_capacity <=
                plan->output_begin + plan->output_capacity;
        if (!valid) atomicExch(overflow_flag, 1u);
        task_output_count_shared = cell_total_output;
        output_valid_shared = valid;
        jobs[job_index].output_count = cell_total_output;
      }
      __syncthreads();

      // Restore key order before output.
      const std::uint32_t output_cell = threadIdx.x;
      if (output_cell < total_cells) {
        const std::uint32_t input_begin = cell_input_offsets[output_cell];
        const std::uint32_t input_count =
            cell_input_offsets[output_cell + 1u] - input_begin;
        if (input_count <= kCellOwnedWarpMaximum) {
          const std::uint32_t output_begin =
              cell_output_offsets[output_cell];
          const std::uint32_t output_count =
              cell_output_counts[output_cell];
          for (std::uint32_t i = 0u; i < output_count; ++i) {
            const std::uint16_t candidate = indices_a[input_begin + i];
            indices_b[output_begin + i] = candidate;
            const bool tombstone = cell_owned_candidate_is_tombstone(
                candidate, tombstone_words);
            const CandidateToken token = candidate_tokens[candidate];
            const std::uint32_t value = tombstone ? 0u :
                load_candidate_value(
                    candidate, token, raw_count, true, staged_epoch,
                    false, raw_storage_begin_shared, q_begin,
                    source_offsets, raw_payloads, raw_offsets, batch_stride,
                    pending_batches, staged_rows, arena, route_headers,
                    route_slices, route_logical_begins,
                    level_q_logical_offsets, crowded_level_begins);
            const Row row{value, candidate_token_key(token),
                          static_cast<std::uint16_t>(
                              tombstone ? kTombstone : 0u)};
            if (output_valid_shared)
              arena.store(job.existing_offset + output_begin + i, row);
          }
        }
      }
      __syncthreads();
      for (std::uint32_t large_index = 0u;
           large_index < large_cell_count_shared; ++large_index) {
        const std::uint32_t cell = cell_queue[
            total_cells - large_cell_count_shared + large_index];
        const std::uint32_t input_begin = cell_input_offsets[cell];
        const std::uint32_t output_begin = cell_output_offsets[cell];
        const std::uint32_t output_count = cell_output_counts[cell];
        for (std::uint32_t i = threadIdx.x; i < output_count;
             i += blockDim.x) {
          const std::uint16_t candidate = indices_a[input_begin + i];
          indices_b[output_begin + i] = candidate;
          const bool tombstone = cell_owned_candidate_is_tombstone(
              candidate, tombstone_words);
          const CandidateToken token = candidate_tokens[candidate];
          const std::uint32_t value = tombstone ? 0u :
              load_candidate_value(
                  candidate, token, raw_count, true, staged_epoch,
                  false, raw_storage_begin_shared, q_begin,
                  source_offsets, raw_payloads, raw_offsets, batch_stride,
                  pending_batches, staged_rows, arena, route_headers,
                  route_slices, route_logical_begins,
                  level_q_logical_offsets, crowded_level_begins);
          const Row row{value, candidate_token_key(token),
                        static_cast<std::uint16_t>(
                            tombstone ? kTombstone : 0u)};
          if (output_valid_shared)
            arena.store(job.existing_offset + output_begin + i, row);
        }
        __syncthreads();
      }

      if (threadIdx.x == 0u) {
        for (std::uint32_t local_q = 0u;
             local_q < quotient_count; ++local_q) {
          const std::uint32_t first_cell = local_q * kFoundationCells;
          const std::uint32_t q_output_begin =
              cell_output_offsets[first_cell];
          const std::uint32_t q_output_count =
              cell_output_offsets[first_cell + kFoundationCells] -
              q_output_begin;
          const std::uint32_t q = q_begin + local_q;
          section_output_counts[q] = q_output_count;
          const RouteHeader route = next_route_headers[q];
          if (route.count) {
            next_route_slices[route.begin] = {
                output_valid_shared
                    ? Descriptor::make(
                          job.existing_offset + q_output_begin,
                          q_output_count)
                    : Descriptor{}, 0u, 1u << 16u};
          }
        }
      }
      __syncthreads();

      if (threadIdx.x < total_cells) {
        const std::uint32_t local_q = threadIdx.x / kFoundationCells;
        const std::uint32_t q = q_begin + local_q;
        if (plan->destination_is_foundation) {
          const std::uint32_t q_output_begin =
              cell_output_offsets[local_q * kFoundationCells];
          local_rank[std::size_t{q} * kFoundationCells +
                     (threadIdx.x & (kFoundationCells - 1u))] =
              static_cast<std::uint16_t>(
                  cell_output_offsets[threadIdx.x] - q_output_begin);
        } else {
          const std::uint32_t rank_block = level_cell_rank_blocks[
              descriptor_index(q, plan->destination_level)];
          if (rank_block != kInvalid) {
            const std::uint32_t q_output_begin =
                cell_output_offsets[local_q * kFoundationCells];
            level_cell_ranks[
                std::size_t{rank_block} * kFoundationCells +
                (threadIdx.x & (kFoundationCells - 1u))] =
                static_cast<std::uint16_t>(
                    cell_output_offsets[threadIdx.x] - q_output_begin);
          }
        }
      }
      if (!plan->destination_is_foundation) {
        for (std::uint32_t sample_index = threadIdx.x;
             sample_index < quotient_count * kGuideSamples;
             sample_index += blockDim.x) {
          const std::uint32_t local_q = sample_index / kGuideSamples;
          const std::uint32_t sample = sample_index % kGuideSamples;
          const std::uint32_t first_cell = local_q * kFoundationCells;
          const std::uint32_t q_output_begin =
              cell_output_offsets[first_cell];
          const std::uint32_t q_output_count =
              cell_output_offsets[first_cell + kFoundationCells] -
              q_output_begin;
          if (q_output_count >= kGuideRegions) {
            const std::uint32_t position = q_output_begin +
                (sample + 1u) * q_output_count / kGuideRegions;
            level_guides[guide_index(
                q_begin + local_q, plan->destination_level) + sample] =
                candidate_token_key(candidate_tokens[indices_b[position]]);
          }
        }
      }
      __syncthreads();
      continue;
    }

    bool input_is_a = true;
    for (std::uint32_t width = 1u;
         !updates_are_resolved && width < raw_count; width <<= 1u) {
      const std::uint16_t *input = input_is_a ? indices_a : indices_b;
      std::uint16_t *output = input_is_a ? indices_b : indices_a;
      // Merge one consecutive interval per worker.
      const std::uint32_t items_per_thread =
          (raw_count + kThreads - 1u) / kThreads;
      std::uint32_t position = threadIdx.x * items_per_thread;
      const std::uint32_t thread_end = min(
          position + items_per_thread, raw_count);
      while (position < thread_end) {
        const std::uint32_t pair_begin =
            (position / (width * 2u)) * width * 2u;
        const std::uint32_t left_count = min(
            width, raw_count - pair_begin);
        const std::uint32_t right_begin = pair_begin + left_count;
        const std::uint32_t right_count = right_begin < raw_count
            ? min(width, raw_count - right_begin) : 0u;
        const std::uint32_t pair_end = right_begin + right_count;
        const std::uint32_t output_end = min(thread_end, pair_end);
        if (!right_count) {
          while (position < output_end) {
            output[position] = input[position];
            ++position;
          }
          continue;
        }
        const std::uint16_t *left = input + pair_begin;
        const std::uint16_t *right = input + right_begin;
        const std::uint32_t diagonal = position - pair_begin;
        std::uint32_t left_index = direct_epoch_merge_partition(
            left, left_count, right, right_count, diagonal,
            candidate_tokens);
        std::uint32_t right_index = diagonal - left_index;
        while (position < output_end) {
          const bool take_left = right_index >= right_count ||
              (left_index < left_count && direct_epoch_candidate_less(
                  left[left_index], right[right_index], candidate_tokens));
          output[position++] = take_left
              ? left[left_index++] : right[right_index++];
        }
      }
      __syncthreads();
      input_is_a = !input_is_a;
    }

    // Resolve duplicate updates after sorting.
    const std::uint16_t *raw_sorted = input_is_a ? indices_a : indices_b;
    std::uint16_t *raw_resolved = input_is_a ? indices_b : indices_a;
    const std::uint32_t items_per_thread =
        (raw_count + kThreads - 1u) / kThreads;
    std::uint32_t winner_mask = 0u, local_winners = 0u;
    for (std::uint32_t item = 0u; item < items_per_thread; ++item) {
      const std::uint32_t index = threadIdx.x * items_per_thread + item;
      if (index >= raw_count) continue;
      const std::uint16_t candidate = raw_sorted[index];
      if (updates_are_resolved) {
        raw_resolved[index] = candidate;
        winner_mask |= 1u << item;
        ++local_winners;
        continue;
      }
      const bool first = index == 0u ||
          candidate_token_logical_key(
              candidate_tokens[raw_sorted[index - 1u]]) !=
              candidate_token_logical_key(candidate_tokens[candidate]);
      if (!first) continue;
      std::uint16_t winner = candidate;
      std::uint32_t newest = direct_raw_metadata(
          candidate, q_begin, source_offsets, raw_payloads, raw_offsets,
          batch_stride, pending_batches) & ~kRawTombstone;
      std::uint32_t next = index + 1u;
      while (next < raw_count &&
             candidate_token_logical_key(
                 candidate_tokens[raw_sorted[next]]) ==
                 candidate_token_logical_key(
                     candidate_tokens[candidate])) {
        const std::uint16_t other = raw_sorted[next++];
        const std::uint32_t age = direct_raw_metadata(
            other, q_begin, source_offsets, raw_payloads, raw_offsets,
            batch_stride, pending_batches) & ~kRawTombstone;
        if (age > newest) {
          newest = age;
          winner = other;
        }
      }
      raw_resolved[index] = winner;
      winner_mask |= 1u << item;
      ++local_winners;
    }
    std::uint32_t thread_base{}, resolved_raw_count{};
    BlockScan(block_scan_storage).ExclusiveSum(
        local_winners, thread_base, resolved_raw_count);
    std::uint32_t local_rank_in_thread = 0u;
    for (std::uint32_t item = 0u; item < items_per_thread; ++item) {
      if (!(winner_mask & (1u << item))) continue;
      const std::uint32_t index = threadIdx.x * items_per_thread + item;
      const std::uint16_t winner = raw_resolved[index];
      if (task_rows != raw_count)
        candidate_tokens[winner] &= ~((1u << kMergeSourceBits) - 1u);
      const_cast<std::uint16_t *>(raw_sorted)[
          thread_base + local_rank_in_thread++] = winner;
    }
    __syncthreads();

    if (raw_sorted != indices_a) {
      for (std::uint32_t i = threadIdx.x; i < resolved_raw_count;
           i += blockDim.x)
        indices_a[i] = raw_sorted[i];
      __syncthreads();
    }

    if (threadIdx.x == 0u)
      resolved_raw_count_shared = resolved_raw_count;
    __syncthreads();

    // Append resident IDs after resolved updates.
    std::uint32_t packed_begin = resolved_raw_count_shared;
    for (std::uint32_t level = 0u;
         level <= plan->source_level_limit; ++level) {
      const std::uint32_t begin =
          source_offsets[kBatchesPerEpoch + level];
      const std::uint32_t count =
          source_offsets[kBatchesPerEpoch + level + 1u] - begin;
      for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x)
        indices_a[packed_begin + i] = static_cast<std::uint16_t>(begin + i);
      packed_begin += count;
    }
    __syncthreads();

    // Merge smaller runs before the largest run.
    if (threadIdx.x == 0u) {
      resolved_raw_count_shared = resolved_raw_count;
      std::uint32_t input_run_count = 0u;
      std::uint32_t packed = 0u;
      if (resolved_raw_count) {
        run_offsets[input_run_count] = 0u;
        run_lengths[input_run_count++] =
            static_cast<std::uint16_t>(resolved_raw_count);
        packed = resolved_raw_count;
      }
      for (std::uint32_t level = 0u;
           level <= plan->source_level_limit; ++level) {
        const std::uint32_t begin =
            source_offsets[kBatchesPerEpoch + level];
        const std::uint32_t count =
            source_offsets[kBatchesPerEpoch + level + 1u] - begin;
        if (!count) continue;
        run_offsets[input_run_count] = static_cast<std::uint16_t>(packed);
        run_lengths[input_run_count++] = static_cast<std::uint16_t>(count);
        packed += count;
      }

      std::uint32_t largest_run = 0u;
      std::uint32_t largest_count = 0u;
      for (std::uint32_t run = 0u; run < input_run_count; ++run) {
        if (run_lengths[run] > largest_count) {
          largest_count = run_lengths[run];
          largest_run = run;
        }
      }
      largest_source_offset_shared = run_offsets[largest_run];
      largest_count_shared = largest_count;
      small_count_shared = packed - largest_count;
      reorder_sources_shared =
          largest_source_offset_shared != small_count_shared;

      std::uint32_t small_run_count = 0u;
      std::uint32_t small_offset = 0u;
      for (std::uint32_t run = 0u; run < input_run_count; ++run) {
        if (run == largest_run) continue;
        const std::uint16_t count = run_lengths[run];
        run_sources[small_run_count] = run_offsets[run];
        run_offsets[small_run_count] =
            static_cast<std::uint16_t>(small_offset);
        run_lengths[small_run_count] = count;
        small_offset += count;
        ++small_run_count;
      }
      run_count_shared = small_run_count;
    }
    __syncthreads();

    if (reorder_sources_shared) {
      for (std::uint32_t run = 0u; run < run_count_shared; ++run) {
        const std::uint32_t count = run_lengths[run];
        for (std::uint32_t i = threadIdx.x; i < count;
             i += blockDim.x)
          indices_b[run_offsets[run] + i] =
              indices_a[run_sources[run] + i];
      }
    }
    for (std::uint32_t i = threadIdx.x; i < largest_count_shared;
         i += blockDim.x)
      indices_b[small_count_shared + i] =
          indices_a[largest_source_offset_shared + i];
    __syncthreads();
    if (reorder_sources_shared) {
      const std::uint32_t merged_count =
          small_count_shared + largest_count_shared;
      for (std::uint32_t i = threadIdx.x; i < merged_count;
           i += blockDim.x)
        indices_a[i] = indices_b[i];
    }
    __syncthreads();

    input_is_a = true;
    while (run_count_shared > 1u) {
      const std::uint16_t *input = input_is_a ? indices_a : indices_b;
      std::uint16_t *output = input_is_a ? indices_b : indices_a;
      const std::uint32_t items =
          (small_count_shared + kThreads - 1u) / kThreads;
      std::uint32_t position = threadIdx.x * items;
      const std::uint32_t end_position = min(
          position + items, small_count_shared);
      while (position < end_position) {
        std::uint32_t pair = 0u;
        while (pair * 2u < run_count_shared) {
          const std::uint32_t first_run = pair * 2u;
          const std::uint32_t pair_count = run_lengths[first_run] +
              (first_run + 1u < run_count_shared
                   ? run_lengths[first_run + 1u] : 0u);
          if (position < run_offsets[first_run] + pair_count) break;
          ++pair;
        }
        const std::uint32_t first_run = pair * 2u;
        const std::uint32_t begin = run_offsets[first_run];
        const std::uint32_t left_count = run_lengths[first_run];
        const std::uint32_t right_count =
            first_run + 1u < run_count_shared
                ? run_lengths[first_run + 1u] : 0u;
        const std::uint32_t pair_end = begin + left_count + right_count;
        const std::uint32_t output_end = min(end_position, pair_end);
        if (!right_count) {
          while (position < output_end) {
            output[position] = input[position];
            ++position;
          }
          continue;
        }
        const std::uint16_t *left = input + begin;
        const std::uint16_t *right = left + left_count;
        const std::uint32_t diagonal = position - begin;
        std::uint32_t li = candidate_token_merge_partition(
            left, left_count, right, right_count, diagonal,
            candidate_tokens);
        std::uint32_t ri = diagonal - li;
        while (position < output_end) {
          const bool take_left = ri >= right_count ||
              (li < left_count &&
               candidate_token_index_less(
                   left[li], right[ri], candidate_tokens));
          output[position++] = take_left ? left[li++] : right[ri++];
        }
      }
      __syncthreads();
      if (threadIdx.x == 0u) {
        const std::uint32_t old_count = run_count_shared;
        const std::uint32_t next_count = (old_count + 1u) >> 1u;
        for (std::uint32_t next = 0u; next < next_count; ++next) {
          const std::uint32_t first_run = next * 2u;
          run_offsets[next] = run_offsets[first_run];
          run_lengths[next] = static_cast<std::uint16_t>(
              run_lengths[first_run] +
              (first_run + 1u < old_count
                   ? run_lengths[first_run + 1u] : 0u));
        }
        run_count_shared = next_count;
      }
      input_is_a = !input_is_a;
      __syncthreads();
    }

    const std::uint32_t merged_count =
        task_rows - raw_count + resolved_raw_count_shared;
    // Merge and compact survivors directly to output.
    std::uint16_t *merge_input = input_is_a ? indices_a : indices_b;
    std::uint16_t *live_scratch = input_is_a ? indices_b : indices_a;
    std::uint16_t *survivors = merge_input;
    const std::uint32_t merged_items_per_thread =
        (merged_count + kThreads - 1u) / kThreads;
    std::uint32_t live_mask = 0u, local_live = 0u;
    const std::uint32_t thread_begin =
        threadIdx.x * merged_items_per_thread;
    const std::uint32_t thread_end = min(
        thread_begin + merged_items_per_thread, merged_count);
    if (thread_begin < thread_end) {
      std::uint32_t position = thread_begin;
      std::uint32_t previous_key = 0u;
      if (small_count_shared && largest_count_shared) {
        const std::uint16_t *left = merge_input;
        const std::uint16_t *right = merge_input + small_count_shared;
        std::uint32_t left_index = candidate_token_merge_partition(
            left, small_count_shared, right, largest_count_shared,
            position, candidate_tokens);
        std::uint32_t right_index = position - left_index;
        if (position) {
          std::uint16_t previous;
          if (!left_index) {
            previous = right[right_index - 1u];
          } else if (!right_index) {
            previous = left[left_index - 1u];
          } else {
            const std::uint16_t left_previous = left[left_index - 1u];
            const std::uint16_t right_previous = right[right_index - 1u];
            previous = candidate_token_index_less(
                left_previous, right_previous, candidate_tokens)
                ? right_previous : left_previous;
          }
          previous_key = candidate_token_logical_key(
              candidate_tokens[previous]);
        }
        while (position < thread_end) {
          const bool take_left = right_index >= largest_count_shared ||
              (left_index < small_count_shared &&
               candidate_token_index_less(
                   left[left_index], right[right_index],
                   candidate_tokens));
          const std::uint16_t candidate = take_left
              ? left[left_index++] : right[right_index++];
          const std::uint32_t key = candidate_token_logical_key(
              candidate_tokens[candidate]);
          const bool first = position == 0u || key != previous_key;
          const bool tombstone =
              (tombstone_words[candidate >> 5u] &
               (1u << (candidate & 31u))) != 0u;
          if (first && (plan->keep_tombstones || !tombstone)) {
            live_scratch[position] = candidate;
            live_mask |= 1u << (position - thread_begin);
            ++local_live;
          }
          previous_key = key;
          ++position;
        }
      } else {
        if (position)
          previous_key = candidate_token_logical_key(
              candidate_tokens[merge_input[position - 1u]]);
        while (position < thread_end) {
          const std::uint16_t candidate = merge_input[position];
          const std::uint32_t key = candidate_token_logical_key(
              candidate_tokens[candidate]);
          const bool first = position == 0u || key != previous_key;
          const bool tombstone =
              (tombstone_words[candidate >> 5u] &
               (1u << (candidate & 31u))) != 0u;
          if (first && (plan->keep_tombstones || !tombstone)) {
            live_scratch[position] = candidate;
            live_mask |= 1u << (position - thread_begin);
            ++local_live;
          }
          previous_key = key;
          ++position;
        }
      }
    }
    std::uint32_t output_count{};
    BlockScan(block_scan_storage).ExclusiveSum(
        local_live, thread_base, output_count);
    if (threadIdx.x == 0u) {
      const bool valid = output_count <= job.existing_capacity &&
          job.existing_offset + job.existing_capacity <=
              plan->output_begin + plan->output_capacity;
      if (!valid) atomicExch(overflow_flag, 1u);
      task_output_count_shared = output_count;
      output_valid_shared = valid;
      jobs[job_index].output_count = output_count;
    }
    __syncthreads();

    const std::uint32_t published_count = task_output_count_shared;
    local_rank_in_thread = 0u;
    for (std::uint32_t item = 0u; item < merged_items_per_thread; ++item) {
      if (!(live_mask & (1u << item))) continue;
      const std::uint32_t index = thread_begin + item;
      const std::uint16_t candidate = live_scratch[index];
      const std::uint32_t rank = thread_base + local_rank_in_thread++;
      survivors[rank] = candidate;
      const bool tombstone =
          (tombstone_words[candidate >> 5u] &
           (1u << (candidate & 31u))) != 0u;
      const CandidateToken token = candidate_tokens[candidate];
      const std::uint32_t value = tombstone ? 0u : load_candidate_value(
          candidate, token, raw_count, task_rows == raw_count,
          staged_epoch, crowded_piece, raw_storage_begin_shared, q_begin,
          source_offsets, raw_payloads, raw_offsets, batch_stride,
          pending_batches, staged_rows, arena, route_headers, route_slices,
          route_logical_begins, level_q_logical_offsets,
          crowded_level_begins);
      const Row row{value, candidate_token_key(token),
                    static_cast<std::uint16_t>(
                        tombstone ? kTombstone : 0u)};
      if (output_valid_shared)
        arena.store(job.existing_offset + rank, row);
    }
    __syncthreads();

    if (crowded_piece && threadIdx.x == 0u) {
      const std::uint32_t q = q_begin;
      atomicAdd(section_output_counts + q, published_count);
      const RouteHeader route = next_route_headers[q];
      if (job.route_ordinal < route.count) {
        const std::uint32_t suffix_begin = static_cast<std::uint32_t>(
            job.key_begin & 0xffffu);
        const std::uint32_t suffix_end = static_cast<std::uint32_t>(
            job.key_end - (std::uint64_t{q} << 16u));
        next_route_slices[route.begin + job.route_ordinal] = {
            output_valid_shared
                ? Descriptor::make(job.existing_offset, published_count)
                : Descriptor{}, suffix_begin, suffix_end};
      }
    }

    // Emit metadata while the quotient is resident.
    for (std::uint32_t local_q = 0u;
         !crowded_piece && local_q < quotient_count; ++local_q) {
      if (threadIdx.x == 0u) {
      std::uint32_t begin = 0u, end = published_count;
      while (begin < end) {
        const std::uint32_t middle = (begin + end) >> 1u;
        if (candidate_token_local_quotient(
                candidate_tokens[survivors[middle]]) < local_q)
          begin = middle + 1u;
        else
          end = middle;
      }
      const std::uint32_t q_output_begin = begin;
      end = published_count;
      while (begin < end) {
        const std::uint32_t middle = (begin + end) >> 1u;
        if (candidate_token_local_quotient(
                candidate_tokens[survivors[middle]]) <= local_q)
          begin = middle + 1u;
        else
          end = middle;
      }
      metadata_begin_shared = q_output_begin;
      metadata_count_shared = begin - q_output_begin;
      const std::uint32_t q = q_begin + local_q;
      section_output_counts[q] = metadata_count_shared;
      const RouteHeader route = next_route_headers[q];
      if (job.route_ordinal < route.count) {
        const std::uint32_t suffix_begin = q == q_begin
            ? static_cast<std::uint32_t>(job.key_begin & 0xffffu) : 0u;
        const std::uint32_t suffix_end = q + 1u == q_end
            ? static_cast<std::uint32_t>(
                  job.key_end - (std::uint64_t{q} << 16u))
            : 1u << 16u;
        next_route_slices[route.begin + job.route_ordinal] = {
            output_valid_shared
                ? Descriptor::make(
                      job.existing_offset + q_output_begin,
                      metadata_count_shared)
                : Descriptor{}, suffix_begin, suffix_end};
      }
      }
      __syncthreads();
      const std::uint32_t q_output_begin = metadata_begin_shared;
      const std::uint32_t q_output_count = metadata_count_shared;
      const std::uint32_t q = q_begin + local_q;
      if (plan->destination_is_foundation) {
        for (std::uint32_t cell = threadIdx.x; cell < 128u;
             cell += blockDim.x) {
        const std::uint32_t target = cell << 9u;
        std::uint32_t low = q_output_begin;
        std::uint32_t high = q_output_begin + q_output_count;
        while (low < high) {
          const std::uint32_t middle = (low + high) >> 1u;
          if (candidate_token_key(
                  candidate_tokens[survivors[middle]]) < target)
            low = middle + 1u;
          else
            high = middle;
        }
        local_rank[std::size_t{q} * 128u + cell] = static_cast<std::uint16_t>(
            low - q_output_begin);
        }
      } else {
        const std::uint32_t rank_block = level_cell_rank_blocks[
            descriptor_index(q, plan->destination_level)];
        if (rank_block != kInvalid) {
          for (std::uint32_t cell = threadIdx.x;
               cell < kFoundationCells; cell += blockDim.x) {
            const std::uint32_t target = cell * kFoundationCellKeys;
            std::uint32_t low = q_output_begin;
            std::uint32_t high = q_output_begin + q_output_count;
            while (low < high) {
              const std::uint32_t middle = (low + high) >> 1u;
              if (candidate_token_key(
                      candidate_tokens[survivors[middle]]) < target)
                low = middle + 1u;
              else
                high = middle;
            }
            level_cell_ranks[
                std::size_t{rank_block} * kFoundationCells + cell] =
                static_cast<std::uint16_t>(low - q_output_begin);
          }
        }
        if (q_output_count >= kGuideRegions) {
          for (std::uint32_t sample = threadIdx.x;
               sample < kGuideSamples; sample += blockDim.x) {
            const std::uint32_t position = q_output_begin +
                (sample + 1u) * q_output_count / kGuideRegions;
            level_guides[guide_index(q, plan->destination_level) + sample] =
                candidate_token_key(candidate_tokens[survivors[position]]);
          }
        }
      }
      __syncthreads();
    }
    __syncthreads();
  }
}

constexpr std::uint32_t kPartitionAuditWarps = 8u;
constexpr std::uint32_t kDirectAuditPieceBuckets = 16u;

struct DirectOutputAuditDevice {
  unsigned long long row_count{};
  unsigned long long digest_sum{};
  unsigned long long digest_xor{};
  unsigned long long unsupported_jobs{};
  unsigned long long count_mismatch_jobs{};
  unsigned long long unsorted_rows{};
  unsigned long long duplicate_rows{};
  unsigned long long value_errors{};
  unsigned long long flag_errors{};
  unsigned long long range_errors{};
  unsigned long long cell_errors[kFoundationCells]{};
  unsigned long long piece_errors[kDirectAuditPieceBuckets]{};
};

struct PartitionAuditDeviceJob {
  std::uint32_t task_rows{};
  std::uint32_t max_source_rows{};
  std::uint32_t imbalance_ppm{};
  std::uint16_t source_count{};
  std::uint16_t nonempty_sources{};
  std::uint16_t rank_supported_sources{};
  std::uint16_t quotient_width{};
  std::uint16_t packed_bits{};
  std::uint16_t duplicate_boundaries{};
  std::uint16_t maximum_boundary_equal_rows{};
  std::uint16_t partition_rows[kPartitionAuditWarps]{};
  std::uint8_t resolved{};
  std::uint8_t rank_supported{};
  std::uint8_t direct_path{};
  std::uint8_t eligible{};
};

__host__ __device__ __forceinline__ std::uint64_t direct_audit_mix(
    std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

__device__ __forceinline__ unsigned long long warp_sum_u64(
    unsigned long long value) {
  for (std::uint32_t offset = 16u; offset; offset >>= 1u)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

__device__ __forceinline__ unsigned long long warp_xor_u64(
    unsigned long long value) {
  for (std::uint32_t offset = 16u; offset; offset >>= 1u)
    value ^= __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

// Audits the physical output immediately after the production direct merge.
// The paper experiment uses unique live values, so strict ordering, value=1,
// and flags=0 are intentional workload-specific reference checks.
__global__ void audit_direct_output_kernel(
    const BalancedMergeJob *jobs, const ResidentPublicationPlan *plan,
    const std::uint64_t *raw_rows, ResidentRows arena,
    DirectOutputAuditDevice *audit) {
  const std::uint32_t lane = threadIdx.x & 31u;
  for (std::uint32_t job_index = blockIdx.x;
       job_index < plan->job_count && !plan->status;
       job_index += gridDim.x) {
    const BalancedMergeJob job = jobs[job_index];
    const bool supported = job.hot_pieces != 0u &&
        job.quotient_end == job.quotient_begin + 1u;
    if (!supported) {
      if (threadIdx.x == 0u)
        atomicAdd(&audit->unsupported_jobs, 1ull);
      __syncthreads();
      continue;
    }
    if (threadIdx.x == 0u) {
      atomicAdd(&audit->row_count,
                static_cast<unsigned long long>(job.output_count));
      if (raw_rows && raw_rows[job_index] != job.output_count)
        atomicAdd(&audit->count_mismatch_jobs, 1ull);
    }

    unsigned long long local_sum = 0ull, local_xor = 0ull;
    unsigned long long local_unsorted = 0ull, local_duplicate = 0ull;
    unsigned long long local_value = 0ull, local_flags = 0ull;
    unsigned long long local_range = 0ull;
    for (std::uint32_t position = threadIdx.x;
         position < job.output_count; position += blockDim.x) {
      const Row row = arena[job.existing_offset + position];
      const std::uint64_t key =
          (std::uint64_t{job.quotient_begin} << 16u) | row.key;
      const bool value_error = row.value != 1u;
      const bool flag_error = row.flags != 0u;
      const bool range_error = key < job.key_begin || key >= job.key_end;
      bool unsorted = false, duplicate = false;
      if (position) {
        const std::uint16_t previous =
            arena.key_at(job.existing_offset + position - 1u);
        unsorted = previous > row.key;
        duplicate = previous == row.key;
      }
      local_unsorted += unsorted;
      local_duplicate += duplicate;
      local_value += value_error;
      local_flags += flag_error;
      local_range += range_error;
      const std::uint64_t mixed = direct_audit_mix(
          (key << 32u) ^ (std::uint64_t{row.value} << 1u) ^ row.flags);
      local_sum += mixed;
      local_xor ^= mixed;
      if (value_error || flag_error || range_error || unsorted || duplicate) {
        atomicAdd(&audit->cell_errors[row.key / kFoundationCellKeys], 1ull);
        const std::uint32_t piece = min(
            static_cast<std::uint32_t>(job.hot_piece),
            kDirectAuditPieceBuckets - 1u);
        atomicAdd(&audit->piece_errors[piece], 1ull);
      }
    }
    local_sum = warp_sum_u64(local_sum);
    local_xor = warp_xor_u64(local_xor);
    local_unsorted = warp_sum_u64(local_unsorted);
    local_duplicate = warp_sum_u64(local_duplicate);
    local_value = warp_sum_u64(local_value);
    local_flags = warp_sum_u64(local_flags);
    local_range = warp_sum_u64(local_range);
    if (lane == 0u) {
      atomicAdd(&audit->digest_sum, local_sum);
      atomicXor(&audit->digest_xor, local_xor);
      atomicAdd(&audit->unsorted_rows, local_unsorted);
      atomicAdd(&audit->duplicate_rows, local_duplicate);
      atomicAdd(&audit->value_errors, local_value);
      atomicAdd(&audit->flag_errors, local_flags);
      atomicAdd(&audit->range_errors, local_range);
    }
    __syncthreads();
  }
}

__device__ __forceinline__ std::uint32_t audit_bit_width(
    std::uint32_t value) {
  return value <= 1u ? 0u : 32u - __clz(value - 1u);
}

// Simulates eight half-open key partitions without changing the merge.  One
// warp owns one job and uses the production source/routing representation.
__global__ void audit_partition_feasibility_kernel(
    const BalancedMergeJob *jobs, const ResidentPublicationPlan *plan,
    const Row *current_rows, const std::uint32_t *current_offsets,
    const std::uint32_t *raw_offsets, std::uint32_t raw_batches,
    const std::uint16_t *current_ranks, ResidentRows arena,
    const DeviceManifest *manifests, const std::uint32_t *active_manifest,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    const std::uint32_t *level_cell_rank_blocks,
    const std::uint16_t *level_cell_ranks,
    const std::uint16_t *local_rank, const std::uint32_t *staged_epoch_mode,
    const std::uint32_t *rank_cell_mode,
    PartitionAuditDeviceJob *output, std::uint16_t *source_rows) {
  const std::uint32_t lane = threadIdx.x & 31u;
  constexpr unsigned mask = 0xffffffffu;
  const DeviceManifestSnapshot manifest = load_active_manifest(
      manifests, active_manifest);
  // These flags are written by kernels immediately preceding this audit.
  // Use coherent atomic reads rather than the read-only cache so every
  // publication is classified using its current mode.
  const bool staged = atomicAdd(
      const_cast<std::uint32_t *>(staged_epoch_mode), 0u) != 0u;
  const bool direct_path = atomicAdd(
      const_cast<std::uint32_t *>(rank_cell_mode), 0u) == 0u;
  for (std::uint32_t job_index = blockIdx.x;
       job_index < plan->job_count && !plan->status;
       job_index += gridDim.x) {
    const BalancedMergeJob job = jobs[job_index];
    std::uint16_t *job_sources = source_rows +
        std::size_t{job_index} * kMaximumMergeSources;
    for (std::uint32_t source = lane; source < kMaximumMergeSources;
         source += 32u)
      job_sources[source] = 0u;
    __syncwarp();

    const std::uint32_t q = job.quotient_begin;
    const std::uint32_t quotient_width =
        job.quotient_end - job.quotient_begin;
    const bool crowded = job.hot_pieces != 0u;
    const bool resolved = staged || crowded;
    std::uint32_t local_total = 0u, local_max = 0u;
    std::uint32_t local_nonempty = 0u, local_ranked = 0u;
    bool local_supported = true;
    if (lane == 0u) {
      std::uint32_t raw = 0u;
      if (resolved && crowded) {
        const std::uint32_t low =
            static_cast<std::uint32_t>(job.key_begin & 0xffffu);
        const std::uint32_t high = static_cast<std::uint32_t>(
            job.key_end - (std::uint64_t{q} << 16u));
        const std::uint32_t begin = current_offsets[q];
        const std::uint32_t count = current_offsets[q + 1u] - begin;
        raw = lower_bound_rows(current_rows + begin, count, high) -
            lower_bound_rows(current_rows + begin, count, low);
      } else if (resolved) {
        raw = current_offsets[job.quotient_end] - current_offsets[q];
      } else {
        for (std::uint32_t batch = 0u; batch < raw_batches; ++batch) {
          const std::size_t base =
              std::size_t{batch} * (kQuotients + 1u);
          raw += raw_offsets[base + job.quotient_end] -
              raw_offsets[base + q];
        }
      }
      job_sources[0] = static_cast<std::uint16_t>(raw);
      local_total = raw;
      local_max = raw;
      local_nonempty = raw != 0u;
      const std::uint32_t raw_section = resolved && quotient_width == 1u
          ? current_offsets[q + 1u] - current_offsets[q] : 0u;
      const bool ranked = resolved && quotient_width == 1u &&
          cell_rank_supported(raw_section);
      local_ranked = raw != 0u && ranked;
      local_supported = raw == 0u || ranked;
    }
    for (std::uint32_t level = lane;
         level <= plan->source_level_limit; level += 32u) {
      if (!level_is_occupied(manifest.occupied_level_mask, level)) continue;
      std::uint32_t count = 0u;
      if (crowded) {
        const std::uint32_t low =
            static_cast<std::uint32_t>(job.key_begin & 0xffffu);
        const std::uint32_t high = static_cast<std::uint32_t>(
            job.key_end - (std::uint64_t{q} << 16u));
        count = logical_section_bound(
            q, level, high, false, arena, route_headers, route_slices,
            route_logical_begins, level_q_logical_offsets) -
            logical_section_bound(
                q, level, low, false, arena, route_headers, route_slices,
                route_logical_begins, level_q_logical_offsets);
      } else {
        for (std::uint32_t section = q;
             section < job.quotient_end; ++section)
          count += descriptors[descriptor_index(section, level)].count();
      }
      job_sources[level + 1u] = static_cast<std::uint16_t>(count);
      local_total += count;
      local_max = max(local_max, count);
      local_nonempty += count != 0u;
      bool ranked = quotient_width == 1u;
      if (ranked) {
        const std::uint32_t section_count =
            descriptors[descriptor_index(q, level)].count();
        ranked = cell_rank_supported(section_count);
        if (ranked && level != manifest.foundation_level)
          ranked = level_cell_rank_blocks[descriptor_index(q, level)] !=
              kInvalid;
      }
      local_ranked += count != 0u && ranked;
      local_supported &= count == 0u || ranked;
    }
    for (std::uint32_t offset = 16u; offset; offset >>= 1u) {
      local_total += __shfl_down_sync(mask, local_total, offset);
      local_max = max(local_max,
                      __shfl_down_sync(mask, local_max, offset));
      local_nonempty += __shfl_down_sync(mask, local_nonempty, offset);
      local_ranked += __shfl_down_sync(mask, local_ranked, offset);
      local_supported &= __shfl_down_sync(mask, local_supported, offset);
    }
    const std::uint32_t total = __shfl_sync(mask, local_total, 0u);
    const std::uint32_t maximum_source = __shfl_sync(mask, local_max, 0u);
    const std::uint32_t nonempty = __shfl_sync(mask, local_nonempty, 0u);
    const std::uint32_t ranked_sources =
        __shfl_sync(mask, local_ranked, 0u);
    const bool ranks_supported =
        __shfl_sync(mask, local_supported, 0u);

    std::uint32_t partitions[kPartitionAuditWarps]{};
    std::uint32_t duplicate_boundaries = 0u;
    std::uint32_t maximum_equal = 0u;
    if (resolved && quotient_width == 1u && total) {
      const std::uint32_t low_suffix =
          static_cast<std::uint32_t>(job.key_begin & 0xffffu);
      const std::uint32_t high_suffix = static_cast<std::uint32_t>(
          job.key_end - (std::uint64_t{q} << 16u));
      const std::uint32_t base = balanced_merge_prefix_count_warp(
          q, low_suffix, current_rows, current_offsets, arena,
          route_headers, route_slices, route_logical_begins,
          level_q_logical_offsets, plan->source_level_limit,
          manifest.occupied_level_mask);
      std::uint32_t previous = base;
      for (std::uint32_t partition = 0u;
           partition + 1u < kPartitionAuditWarps; ++partition) {
        const std::uint32_t target = base + static_cast<std::uint32_t>(
            (std::uint64_t{total} * (partition + 1u) +
             kPartitionAuditWarps - 1u) / kPartitionAuditWarps);
        std::uint32_t boundary = resident_hot_boundary_warp(
            q, target, current_rows, current_offsets, arena,
            route_headers, route_slices, route_logical_begins,
            level_q_logical_offsets, plan->source_level_limit,
            manifest.occupied_level_mask);
        boundary = min(max(boundary, low_suffix), high_suffix);
        const std::uint32_t prefix = balanced_merge_prefix_count_warp(
            q, boundary, current_rows, current_offsets, arena,
            route_headers, route_slices, route_logical_begins,
            level_q_logical_offsets, plan->source_level_limit,
            manifest.occupied_level_mask);
        if (lane == 0u) partitions[partition] = prefix - previous;
        previous = prefix;
        if (boundary > low_suffix) {
          const std::uint32_t before = balanced_merge_prefix_count_warp(
              q, boundary - 1u, current_rows, current_offsets, arena,
              route_headers, route_slices, route_logical_begins,
              level_q_logical_offsets, plan->source_level_limit,
              manifest.occupied_level_mask);
          const std::uint32_t equal = prefix - before;
          if (lane == 0u) {
            duplicate_boundaries += equal > 1u;
            maximum_equal = max(maximum_equal, equal);
          }
        }
      }
      if (lane == 0u)
        partitions[kPartitionAuditWarps - 1u] = base + total - previous;
    }
    if (lane == 0u) {
      PartitionAuditDeviceJob item{};
      item.task_rows = total;
      item.max_source_rows = maximum_source;
      item.source_count = static_cast<std::uint16_t>(plan->source_count);
      item.nonempty_sources = static_cast<std::uint16_t>(nonempty);
      item.rank_supported_sources =
          static_cast<std::uint16_t>(ranked_sources);
      item.quotient_width = static_cast<std::uint16_t>(quotient_width);
      item.packed_bits = static_cast<std::uint16_t>(
          16u + audit_bit_width(quotient_width) +
          audit_bit_width(plan->job_capacity));
      item.duplicate_boundaries =
          static_cast<std::uint16_t>(duplicate_boundaries);
      item.maximum_boundary_equal_rows =
          static_cast<std::uint16_t>(maximum_equal);
      std::uint32_t maximum_partition = 0u;
      for (std::uint32_t partition = 0u;
           partition < kPartitionAuditWarps; ++partition) {
        item.partition_rows[partition] =
            static_cast<std::uint16_t>(partitions[partition]);
        maximum_partition = max(maximum_partition, partitions[partition]);
      }
      item.imbalance_ppm = total
          ? static_cast<std::uint32_t>(
                std::uint64_t{maximum_partition} *
                kPartitionAuditWarps * 1000000ull / total)
          : 0u;
      item.resolved = resolved;
      item.rank_supported = ranks_supported;
      item.direct_path = direct_path;
      item.eligible = direct_path && resolved && ranks_supported &&
          quotient_width <= 16u && item.packed_bits <= 32u && total >= 1024u;
      output[job_index] = item;
    }
  }
}

__global__ void audit_lookup_jobs_kernel(
    const std::uint32_t *queries, std::uint32_t count,
    const BalancedMergeJob *jobs, const ResidentPublicationPlan *plan,
    ResidentRows arena, std::uint32_t *results) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint64_t key = queries[i];
  std::uint32_t low = 0u, high = plan->job_count;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (jobs[middle].key_end <= key) low = middle + 1u;
    else high = middle;
  }
  if (low == plan->job_count || jobs[low].key_begin > key ||
      key >= jobs[low].key_end) {
    results[i] = kInvalid;
    return;
  }
  const BalancedMergeJob job = jobs[low];
  Row row{};
  const bool matched = find_leftmost_point_row(
      arena + job.existing_offset, job.output_count,
      key_suffix(static_cast<std::uint32_t>(key)), row);
  results[i] = matched && !(row.flags & kTombstone) ? row.value : kInvalid;
}

__global__ void audit_lookup_routes_kernel(
    const std::uint32_t *queries, std::uint32_t count,
    const ResidentPublicationPlan *plan, ResidentRows arena,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices, std::uint32_t *results) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t key = queries[i];
  const std::uint32_t q = key >> 16u;
  const std::uint32_t suffix = key_suffix(key);
  const std::size_t mapping = descriptor_index(q, plan->destination_level);
  const Descriptor logical = descriptors[mapping];
  const RoutedSliceSelection selection = logical.split()
      ? routed_slice_for_suffix(
            q, plan->destination_level, suffix,
            route_headers, route_slices)
      : RoutedSliceSelection{logical, 0u, logical.count() != 0u};
  Row row{};
  const bool matched = selection.valid && find_leftmost_point_row(
      arena + selection.rows.offset(), selection.rows.count(), suffix, row);
  results[i] = matched && !(row.flags & kTombstone) ? row.value : kInvalid;
}

inline std::uint32_t select_balanced_merge_capacity() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

  int maximum_blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maximum_blocks, compact_direct_epoch_jobs_kernel,
      kFoundationCompactionThreads, 0u));
  // Preserve merge occupancy over larger jobs.
  const int desired_blocks = std::max(1, std::min(4, maximum_blocks));
  // Overlay cell cursors on the second index plane.
  const std::uint32_t minimum_capacity = std::max(
      kMaximumMergeSources, kCellOwnedCells * 2u + 1u);
  std::uint32_t block_low = minimum_capacity;
  std::uint32_t block_high = kBalancedMergeCapacityCeiling;
  while (block_low < block_high) {
    const std::uint32_t middle =
        block_low + (block_high - block_low + 1u) / 2u;
    if (balanced_merge_dynamic_shared_bytes(middle) <=
        properties.sharedMemPerBlock)
      block_low = middle;
    else
      block_high = middle - 1u;
  }
  std::uint32_t low = minimum_capacity;
  std::uint32_t high = block_low;
  if (high < low)
    throw std::runtime_error("insufficient shared memory for GPULSMOpt merge");

  while (low < high) {
    const std::uint32_t middle = low + (high - low + 1u) / 2u;
    int blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, compact_direct_epoch_jobs_kernel,
        kFoundationCompactionThreads,
        balanced_merge_dynamic_shared_bytes(middle)));
    if (blocks >= desired_blocks) low = middle;
    else high = middle - 1u;
  }
  // Keep both index planes four-byte aligned.
  return (low & 1u) ? low : low - 1u;
}

__global__ void publish_single_route_directory_kernel(
    const Descriptor *descriptors, std::uint32_t destination_level,
    std::uint32_t route_stride, RouteHeader *route_headers,
    RouteSlice *route_slices) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  for (std::uint32_t level = 0u; level < destination_level; ++level)
    route_headers[descriptor_index(q, level)] = {};
  const Descriptor descriptor =
      descriptors[descriptor_index(q, destination_level)];
  const std::uint32_t route = destination_level * route_stride + q;
  route_headers[descriptor_index(q, destination_level)] =
      RouteHeader{route, descriptor.count() ? 1u : 0u};
  route_slices[route] = {descriptor, 0u, 1u << 16u};
}


__global__ void build_foundation_rank_directory_kernel(
    ResidentRows arena, const Descriptor *descriptors,
    std::uint32_t foundation_level, std::uint16_t *local_rank) {
  const std::uint32_t q = blockIdx.x;
  const std::uint32_t cell = threadIdx.x;
  const Descriptor descriptor =
      descriptors[descriptor_index(q, foundation_level)];
  const std::uint32_t target = cell << 9u;
  const std::uint32_t position = cell_rank_supported(descriptor.count())
      ? lower_bound_rows(arena + descriptor.offset(), descriptor.count(),
                         target)
      : 0u;
  local_rank[std::size_t{q} * 128u + cell] =
      static_cast<std::uint16_t>(position);
}


__global__ void lookup_with_pending_kernel(
    const std::uint32_t *queries, std::uint32_t *out_values,
    std::uint8_t *out_found, std::uint32_t count,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets,
    std::uint32_t batch_stride, std::uint32_t pending_batches,
    const std::uint64_t *batch_signatures,
    const std::uint64_t *epoch_signatures,
    ResidentRows arena, const Descriptor *descriptors,
    const RouteHeader *route_headers, const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    const std::uint16_t *local_rank,
    const std::uint16_t *level_guides,
    const std::uint32_t *level_cell_rank_blocks,
    const std::uint16_t *level_cell_ranks,
    const std::uint32_t *query_ids,
    const std::uint64_t *query_occupied_level_mask) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const DeviceManifestSnapshot manifest =
      load_query_manifest(query_occupied_level_mask);
  const std::uint32_t active_levels = manifest.active_levels;
  const std::uint32_t foundation_level = manifest.foundation_level;
  std::uint64_t occupied_levels = manifest.occupied_level_mask;
  const std::uint32_t key = queries[i], q = key >> 16u;
  const std::uint32_t suffix = key_suffix(key);
  const std::uint64_t signature_bits = pending_signature_bits(key);
  if (pending_batches &&
      (epoch_signatures[q] & signature_bits) == signature_bits) {
    for (int batch = int(pending_batches) - 1; batch >= 0; --batch) {
      const std::uint32_t batch_index = static_cast<std::uint32_t>(batch);
      const std::uint64_t signature =
          batch_signatures[std::size_t{batch_index} * kQuotients + q];
      if ((signature & signature_bits) != signature_bits) continue;
      const std::size_t oi =
          std::size_t{batch_index} * (kQuotients + 1u) + q;
      const std::uint32_t begin = raw_offsets[oi], end = raw_offsets[oi + 1u];
      Row row{};
      std::uint32_t newest_position = 0u;
      bool matched = false;
      for (std::uint32_t position = begin; position < end; ++position) {
        const std::uint32_t record =
            batch_index * batch_stride + position;
        if (key_suffix(raw_keys[record]) == suffix) {
          const RawPayload payload = raw_payloads[record];
          const std::uint32_t candidate_position = raw_position(payload);
          if (!matched || candidate_position > newest_position) {
            row = raw_row(key, payload);
            newest_position = candidate_position;
            matched = true;
          }
        }
      }
      if (matched) {
        const bool live = (row.flags & kTombstone) == 0u;
        const std::uint32_t destination = query_ids ? query_ids[i] : i;
        out_values[destination] = live ? row.value : out_found ? 0u : kInvalid;
        if (out_found) out_found[destination] = live;
        return;
      }
    }
  }
  if (foundation_level < kMaximumLevels)
    occupied_levels &= ~(std::uint64_t{1} << foundation_level);
  while (occupied_levels) {
    const std::uint32_t level =
        static_cast<std::uint32_t>(__ffsll(occupied_levels) - 1);
    occupied_levels &= occupied_levels - 1u;
    const std::size_t mapping = descriptor_index(q, level);
    const Descriptor logical_descriptor = descriptors[mapping];
    const RouteHeader route_header = logical_descriptor.split()
        ? route_headers[mapping] : RouteHeader{};
    const std::uint32_t route_count = logical_descriptor.split()
        ? route_header.count : logical_descriptor.count() ? 1u : 0u;
    const RoutedSliceSelection selected = !logical_descriptor.split()
        ? RoutedSliceSelection{logical_descriptor, 0u,
                               logical_descriptor.count() != 0u}
        : routed_slice_for_suffix(
              q, level, suffix, route_headers, route_slices);
    const Descriptor descriptor = selected.rows;
    if (!descriptor.count()) continue;
    const ResidentRows rows = arena + descriptor.offset();
    std::uint32_t begin{}, end{};
    resident_point_search_bounds(
        q, level, suffix, foundation_level, descriptor, selected.route,
        route_count, logical_descriptor.count(), route_logical_begins,
        level_q_logical_offsets, local_rank, level_guides,
        level_cell_rank_blocks, level_cell_ranks, begin, end);
    if (begin >= end) continue;
    Row matched_row{};
    const bool matched = logical_descriptor.split()
        ? find_leftmost_point_row(
              rows + begin, end - begin, suffix, matched_row)
        : find_unique_point_row(
              rows + begin, end - begin, suffix, matched_row);
    if (matched) {
      const bool live = (matched_row.flags & kTombstone) == 0u;
      const std::uint32_t destination = query_ids ? query_ids[i] : i;
      out_values[destination] =
          live ? matched_row.value : out_found ? 0u : kInvalid;
      if (out_found) out_found[destination] = live;
      return;
    }
  }
  const std::size_t foundation_mapping =
      descriptor_index(q, foundation_level);
  const Descriptor foundation_logical = foundation_level < active_levels
      ? descriptors[foundation_mapping] : Descriptor{};
  const RouteHeader foundation_header = foundation_logical.split()
      ? route_headers[foundation_mapping] : RouteHeader{};
  const std::uint32_t foundation_route_count = foundation_logical.split()
      ? foundation_header.count : foundation_logical.count() ? 1u : 0u;
  const RoutedSliceSelection foundation_selection =
      !foundation_logical.split()
          ? RoutedSliceSelection{foundation_logical, 0u,
                                 foundation_logical.count() != 0u}
          : foundation_level < active_levels
              ? routed_slice_for_suffix(
                    q, foundation_level, suffix,
                    route_headers, route_slices)
              : RoutedSliceSelection{};
  const Descriptor foundation = foundation_selection.rows;
  const ResidentRows foundation_rows = arena + foundation.offset();
  const std::uint32_t cell = (key >> 9u) & 127u;
  const std::size_t local_index = std::size_t{q} * 128u + cell;
  const std::uint32_t foundation_logical_count = foundation_logical.count();
  const bool ranked = foundation_route_count &&
      cell_rank_supported(foundation_logical_count);
  const std::uint32_t logical_begin =
      ranked ? local_rank[local_index] : 0u;
  const std::uint32_t logical_end = ranked
      ? (cell == 127u ? foundation_logical_count
                      : local_rank[local_index + 1u])
      : foundation_logical_count;
  std::uint32_t begin = logical_begin;
  std::uint32_t end = logical_end;
  if (foundation_route_count != 1u) {
    const std::uint32_t foundation_section_begin =
        foundation_level < active_levels
            ? level_q_logical_offsets[
                  std::size_t{foundation_level} * (kQuotients + 1u) + q]
            : 0u;
    const std::uint32_t foundation_route_begin = foundation_selection.valid
        ? route_logical_begins[foundation_selection.route] -
              foundation_section_begin
        : 0u;
    begin = foundation_selection.valid &&
            logical_begin > foundation_route_begin
        ? logical_begin - foundation_route_begin : 0u;
    end = foundation_selection.valid
        ? min(foundation.count(), logical_end > foundation_route_begin
              ? logical_end - foundation_route_begin : 0u)
        : 0u;
  }
  Row matched_row{};
  const bool matched = foundation_logical.split()
      ? find_leftmost_point_row(
            foundation_rows + begin, end - begin, suffix, matched_row)
      : find_unique_point_row(
            foundation_rows + begin, end - begin, suffix, matched_row);
  const bool live = matched && (matched_row.flags & kTombstone) == 0u;
  const std::uint32_t destination = query_ids ? query_ids[i] : i;
  out_values[destination] =
      live ? matched_row.value : out_found ? 0u : kInvalid;
  if (out_found) out_found[destination] = live;
}


__device__ __forceinline__ void lookup_resident_only(
    std::uint32_t key, std::uint32_t query_index,
    std::uint32_t *out_values, std::uint8_t *out_found,
    ResidentRows arena, const Descriptor *descriptors,
    const RouteHeader *route_headers, const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    const std::uint16_t *local_rank, const std::uint16_t *level_guides,
    const std::uint32_t *level_cell_rank_blocks,
    const std::uint16_t *level_cell_ranks,
    std::uint32_t active_levels, std::uint32_t foundation_level,
    std::uint64_t occupied_levels, const std::uint32_t *query_ids) {
  const std::uint32_t q = key >> 16u;
  const std::uint32_t suffix = key_suffix(key);
  if (foundation_level < kMaximumLevels)
    occupied_levels &= ~(std::uint64_t{1} << foundation_level);
  while (occupied_levels) {
    const std::uint32_t level =
        static_cast<std::uint32_t>(__ffsll(occupied_levels) - 1);
    occupied_levels &= occupied_levels - 1u;
    const std::size_t mapping = descriptor_index(q, level);
    const Descriptor logical_descriptor = descriptors[mapping];
    const RouteHeader route_header = logical_descriptor.split()
        ? route_headers[mapping] : RouteHeader{};
    const std::uint32_t route_count = logical_descriptor.split()
        ? route_header.count : logical_descriptor.count() ? 1u : 0u;
    const RoutedSliceSelection selected = !logical_descriptor.split()
        ? RoutedSliceSelection{logical_descriptor, 0u,
                               logical_descriptor.count() != 0u}
        : routed_slice_for_suffix(
              q, level, suffix, route_headers, route_slices);
    const Descriptor descriptor = selected.rows;
    if (!descriptor.count()) continue;
    const ResidentRows rows = arena + descriptor.offset();
    std::uint32_t begin{}, end{};
    resident_point_search_bounds(
        q, level, suffix, foundation_level, descriptor, selected.route,
        route_count, logical_descriptor.count(), route_logical_begins,
        level_q_logical_offsets, local_rank, level_guides,
        level_cell_rank_blocks, level_cell_ranks, begin, end);
    if (begin >= end) continue;
    Row matched_row{};
    const bool matched = logical_descriptor.split()
        ? find_leftmost_point_row(
              rows + begin, end - begin, suffix, matched_row)
        : find_unique_point_row(
              rows + begin, end - begin, suffix, matched_row);
    if (matched) {
      const bool live = (matched_row.flags & kTombstone) == 0u;
      const std::uint32_t destination =
          query_ids ? query_ids[query_index] : query_index;
      out_values[destination] =
          live ? matched_row.value : out_found ? 0u : kInvalid;
      if (out_found) out_found[destination] = live;
      return;
    }
  }
  const std::size_t foundation_mapping =
      descriptor_index(q, foundation_level);
  const Descriptor foundation_logical = foundation_level < active_levels
      ? descriptors[foundation_mapping] : Descriptor{};
  const RouteHeader foundation_header = foundation_logical.split()
      ? route_headers[foundation_mapping] : RouteHeader{};
  const std::uint32_t foundation_route_count = foundation_logical.split()
      ? foundation_header.count : foundation_logical.count() ? 1u : 0u;
  const RoutedSliceSelection foundation_selection =
      !foundation_logical.split()
          ? RoutedSliceSelection{foundation_logical, 0u,
                                 foundation_logical.count() != 0u}
          : foundation_level < active_levels
              ? routed_slice_for_suffix(
                    q, foundation_level, suffix,
                    route_headers, route_slices)
              : RoutedSliceSelection{};
  const Descriptor foundation = foundation_selection.rows;
  const ResidentRows foundation_rows = arena + foundation.offset();
  const std::uint32_t cell = (key >> 9u) & 127u;
  const std::size_t local_index = std::size_t{q} * 128u + cell;
  const std::uint32_t foundation_logical_count = foundation_logical.count();
  const bool ranked = foundation_route_count &&
      cell_rank_supported(foundation_logical_count);
  const std::uint32_t logical_begin =
      ranked ? local_rank[local_index] : 0u;
  const std::uint32_t logical_end = ranked
      ? (cell == 127u ? foundation_logical_count
                      : local_rank[local_index + 1u])
      : foundation_logical_count;
  std::uint32_t begin = logical_begin;
  std::uint32_t end = logical_end;
  if (foundation_route_count != 1u) {
    const std::uint32_t foundation_section_begin =
        foundation_level < active_levels
            ? level_q_logical_offsets[
                  std::size_t{foundation_level} * (kQuotients + 1u) + q]
            : 0u;
    const std::uint32_t foundation_route_begin = foundation_selection.valid
        ? route_logical_begins[foundation_selection.route] -
              foundation_section_begin
        : 0u;
    begin = foundation_selection.valid &&
            logical_begin > foundation_route_begin
        ? logical_begin - foundation_route_begin : 0u;
    end = foundation_selection.valid
        ? min(foundation.count(), logical_end > foundation_route_begin
              ? logical_end - foundation_route_begin : 0u)
        : 0u;
  }
  Row matched_row{};
  const bool matched = foundation_logical.split()
      ? find_leftmost_point_row(
            foundation_rows + begin, end - begin, suffix, matched_row)
      : find_unique_point_row(
            foundation_rows + begin, end - begin, suffix, matched_row);
  const bool live = matched && (matched_row.flags & kTombstone) == 0u;
  const std::uint32_t destination =
      query_ids ? query_ids[query_index] : query_index;
  out_values[destination] =
      live ? matched_row.value : out_found ? 0u : kInvalid;
  if (out_found) out_found[destination] = live;
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
    const RouteHeader *route_headers, const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    const std::uint16_t *local_rank,
    const std::uint16_t *level_guides,
    const std::uint32_t *level_cell_rank_blocks,
    const std::uint16_t *level_cell_ranks,
    const std::uint32_t *query_ids,
    const std::uint16_t *canonical_cell_ranks,
    bool canonical,
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
  const std::uint32_t active_levels = manifest.active_levels;
  const std::uint32_t foundation_level = manifest.foundation_level;
  const std::uint64_t occupied_levels = manifest.occupied_level_mask;

  bool possible = false;
  if (valid) {
    const std::uint64_t bits = pending_signature_bits(key);
    possible = (epoch_signatures[q] & bits) == bits;
  }
  if (!__syncthreads_or(possible)) {
    if (valid) {
      if (canonical)
        canonical_lookup_resident_only(
            key, i, out_values, out_found, arena, descriptors,
            canonical_cell_ranks, occupied_levels, query_ids);
      else
        lookup_resident_only(
            key, i, out_values, out_found, arena, descriptors,
            route_headers, route_slices, route_logical_begins,
            level_q_logical_offsets, local_rank, level_guides,
            level_cell_rank_blocks, level_cell_ranks,
            active_levels, foundation_level, occupied_levels, query_ids);
    }
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
      unsigned matched_mask = __ballot_sync(__activemask(), matched);
      if (!matched) continue;
      const RawPayload payload = raw_payloads[record];
      const std::uint64_t order =
          static_cast<std::uint64_t>(raw_position(payload)) + 1u;
      const unsigned long long token =
          static_cast<unsigned long long>((order << 24u) | record);
      const unsigned peers = __match_any_sync(matched_mask, owner);
      unsigned long long newest = token;
      unsigned remaining = peers;
      while (remaining) {
        const std::uint32_t source =
            static_cast<std::uint32_t>(__ffs(remaining) - 1);
        newest = max(newest, __shfl_sync(peers, token, source));
        remaining &= remaining - 1u;
      }
      const std::uint32_t lane = threadIdx.x & 31u;
      if (lane == static_cast<std::uint32_t>(__ffs(peers) - 1))
        atomicMax(router_winners + owner, newest);
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
  if (canonical)
    canonical_lookup_resident_only(
        key, i, out_values, out_found, arena, descriptors,
        canonical_cell_ranks, occupied_levels, query_ids);
  else
    lookup_resident_only(
        key, i, out_values, out_found, arena, descriptors,
        route_headers, route_slices, route_logical_begins,
        level_q_logical_offsets, local_rank, level_guides,
        level_cell_rank_blocks, level_cell_ranks,
        active_levels, foundation_level, occupied_levels, query_ids);
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

  struct MaintenanceStats {
    std::uint64_t epochs_published = 0u;
    std::uint64_t admitted_batches = 0u;
    std::uint64_t admitted_records = 0u;
    std::uint32_t pending_batches = 0u;
    std::uint32_t pending_records = 0u;
    std::uint32_t active_levels = 0u;
  };

  struct CanonicalCarryStats {
    std::uint32_t destination_level = 0u;
    std::uint32_t source_count = 0u;
    std::uint32_t status = 0u;
    std::uint32_t job_count = 0u;
    std::uint32_t claimed_jobs = 0u;
    std::uint64_t selected_count = 0u;
    std::uint64_t input_count = 0u;
    std::uint64_t survivor_count = 0u;
    std::uint64_t quotient_total = 0u;
  };

  struct DirectMergeJobProfile {
    std::uint32_t job_index = 0u;
    std::uint64_t key_begin = 0u;
    std::uint64_t key_end = 0u;
    std::uint64_t existing_offset = 0u;
    std::uint32_t quotient_begin = 0u;
    std::uint32_t quotient_end = 0u;
    std::uint32_t quotient_count = 0u;
    std::uint64_t raw_rows = 0u;
    std::uint32_t output_rows = 0u;
    std::uint32_t existing_capacity = 0u;
    std::uint32_t route_ordinal = 0u;
    std::uint32_t hot_pieces = 0u;
    std::uint32_t hot_piece = 0u;
    bool crowded = false;
    bool cell_owned_shape = false;
  };

  struct DirectMergeProfileStats {
    bool standalone = false;
    bool forced_unified_merge = false;
    bool forced_unified_compile_elision = false;
    std::uint32_t direct_launch_count = 0u;
    std::uint32_t rank_cell_mode = 0u;
    std::uint32_t grid_x = 0u;
    std::uint32_t block_x = 0u;
    std::size_t dynamic_shared_bytes = 0u;
    std::uint32_t selected_count = 0u;
    std::uint32_t source_count = 0u;
    std::uint32_t destination_level = 0u;
    std::uint32_t publication_status = 0u;
    std::uint32_t job_count = 0u;
    std::uint64_t raw_reservation = 0u;
    std::uint64_t survivor_count = 0u;
    std::uint64_t occupied_level_mask = 0u;
    std::vector<DirectMergeJobProfile> jobs;
  };

  struct DirectCorrectnessAuditStats {
    bool enabled = false;
    std::uint64_t row_count = 0u;
    std::uint64_t digest_sum = 0u;
    std::uint64_t digest_xor = 0u;
    std::uint64_t unsupported_jobs = 0u;
    std::uint64_t count_mismatch_jobs = 0u;
    std::uint64_t unsorted_rows = 0u;
    std::uint64_t duplicate_rows = 0u;
    std::uint64_t value_errors = 0u;
    std::uint64_t flag_errors = 0u;
    std::uint64_t range_errors = 0u;
    std::uint64_t route_header_errors = 0u;
    std::uint64_t route_slice_errors = 0u;
    std::uint64_t route_logical_errors = 0u;
    std::uint64_t descriptor_errors = 0u;
    std::array<std::uint64_t,
               gpulsmopt2_detail::kFoundationCells> cell_errors{};
    std::array<std::uint64_t,
               gpulsmopt2_detail::kDirectAuditPieceBuckets> piece_errors{};
  };

  struct PartitionAuditStats {
    bool enabled = false;
    std::uint32_t selected_count = 0u;
    std::uint32_t source_count = 0u;
    std::uint32_t destination_level = 0u;
    std::uint32_t rank_cell_mode = 0u;
    std::uint32_t job_count = 0u;
    std::uint64_t total_rows = 0u;
    std::uint64_t direct_rows = 0u;
    std::uint64_t resolved_rows = 0u;
    std::uint64_t rank_supported_rows = 0u;
    std::uint64_t eligible_rows = 0u;
    std::uint64_t direct_jobs = 0u;
    std::uint64_t resolved_jobs = 0u;
    std::uint64_t rank_supported_jobs = 0u;
    std::uint64_t eligible_jobs = 0u;
    std::uint64_t packed_word_failure_jobs = 0u;
    std::uint64_t ordering_equivalence_violations = 0u;
    std::uint64_t duplicate_boundary_count = 0u;
    std::uint32_t maximum_boundary_equal_rows = 0u;
    std::uint32_t maximum_source_rows = 0u;
    std::uint32_t maximum_packed_bits = 0u;
    std::uint32_t imbalance_p50_ppm = 0u;
    std::uint32_t imbalance_p95_ppm = 0u;
    std::uint32_t imbalance_p99_ppm = 0u;
    std::uint32_t imbalance_maximum_ppm = 0u;
    std::array<std::uint64_t, 17u> source_run_log2_histogram{};
    std::array<std::uint64_t,
               gpulsmopt2_detail::kMaximumMergeSources + 1u>
        nonempty_source_histogram{};
  };

  // Retained for the earlier rank-cell standalone evidence.
  struct RankCellProfileStats {
    bool standalone = false;
    std::uint32_t rank_cell_mode = 0u;
    std::uint32_t launch_count = 0u;
    std::uint32_t grid_x = 0u;
    std::uint32_t block_x = 0u;
    std::size_t dynamic_shared_bytes = 0u;
    std::uint32_t selected_count = 0u;
    std::uint32_t destination_level = 0u;
    std::uint32_t publication_status = 0u;
    std::uint64_t raw_reservation = 0u;
    std::uint64_t survivor_count = 0u;
    std::uint64_t occupied_level_mask = 0u;
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
        foundation_pool_capacity_(
            gpulsmopt2_detail::kCanonicalCarry ? 0u :
            gpulsmopt2_detail::foundation_pool_capacity(
                publication_capacity_)),
        level_pool_capacity_(
            gpulsmopt2_detail::preassigned_level_pool_capacity(
                publication_capacity_, level_zero_capacity_)),
        level_rank_block_capacity_(
            gpulsmopt2_detail::preassigned_level_rank_blocks(
                publication_capacity_, level_zero_capacity_)),
        canonical_level_count_(gpulsmopt2_detail::canonical_level_count(
            publication_capacity_, level_zero_capacity_)),
        resident_merge_capacity_(
            gpulsmopt2_detail::select_balanced_merge_capacity() -
            (gpulsmopt2_detail::kCanonicalCarry
                 ? gpulsmopt2_detail::kCanonicalCapacityAdjustment : 0u)),
        resident_merge_workspace_bytes_(
            gpulsmopt2_detail::balanced_merge_dynamic_shared_bytes(
                resident_merge_capacity_)),
        canonical_merge_workspace_bytes_(
            std::size_t{resident_merge_capacity_} *
            sizeof(std::uint32_t) * 2u),
        maximum_resident_jobs_(
            gpulsmopt2_detail::maximum_resident_merge_jobs(
                publication_capacity_, resident_merge_capacity_)),
        route_stride_(gpulsmopt2_detail::kCanonicalCarry
            ? gpulsmopt2_detail::kQuotients
            : gpulsmopt2_detail::adaptive_route_stride(
                  publication_capacity_, resident_merge_capacity_)),
        local_rank_(gpulsmopt2_detail::kCanonicalCarry
            ? 1u : gpulsmopt2_detail::kLocalRankEntries),
        level_guides_(
            gpulsmopt2_detail::kMaximumLevels *
                gpulsmopt2_detail::kGuideEntriesPerLevel,
            gpulsmopt2_detail::kCanonicalCarry ? 1u :
                gpulsmopt2_detail::kMaximumLevels *
                    gpulsmopt2_detail::kGuideEntriesPerLevel),
        arena_key_flags_(gpulsmopt2_detail::maximum_resident_elements<
                             gpulsmopt2_detail::Row>(),
                         foundation_pool_capacity_ + level_pool_capacity_),
        arena_values_(gpulsmopt2_detail::maximum_resident_elements<
                          gpulsmopt2_detail::Row>(),
                      foundation_pool_capacity_ + level_pool_capacity_),
        descriptors_(std::size_t{gpulsmopt2_detail::kQuotients} *
                     gpulsmopt2_detail::kMaximumLevels),
        route_headers_(std::size_t{gpulsmopt2_detail::kQuotients} *
                       gpulsmopt2_detail::kMaximumLevels),
        route_slices_(
            route_stride_ * (gpulsmopt2_detail::kCanonicalCarry
                ? canonical_level_count_
                : gpulsmopt2_detail::kMaximumLevels),
            route_stride_ * (gpulsmopt2_detail::kCanonicalCarry
                ? canonical_level_count_
                : gpulsmopt2_detail::kMaximumLevels)),
        route_logical_begins_(
            route_stride_ * (gpulsmopt2_detail::kCanonicalCarry
                ? canonical_level_count_
                : gpulsmopt2_detail::kMaximumLevels)),
        route_quotients_(
            route_stride_ * (gpulsmopt2_detail::kCanonicalCarry
                ? canonical_level_count_
                : gpulsmopt2_detail::kMaximumLevels)),
        level_q_logical_offsets_(
            std::size_t{gpulsmopt2_detail::kCanonicalCarry
                ? canonical_level_count_
                : gpulsmopt2_detail::kMaximumLevels} *
            (gpulsmopt2_detail::kQuotients + 1u)),
        device_manifests_(2u),
        active_device_manifest_(1u),
        query_occupied_level_mask_(1u),
        staged_epoch_mode_(1u),
        rank_cell_mode_(1u),
        resident_plan_(1u),
        publication_receipt_(1u),
        level_storage_spans_(gpulsmopt2_detail::kMaximumLevels),
        level_rank_spans_(gpulsmopt2_detail::kMaximumLevels),
        level_cell_rank_blocks_(
            gpulsmopt2_detail::kCanonicalCarry ? 1u :
                std::size_t{gpulsmopt2_detail::kQuotients} *
                    gpulsmopt2_detail::kMaximumLevels),
        level_cell_ranks_(
            gpulsmopt2_detail::kCanonicalCarry ? 1u :
                level_rank_block_capacity_ *
                    gpulsmopt2_detail::kFoundationCells),
        publication_cell_ranks_(
            gpulsmopt2_detail::kCanonicalCarry ? 1u :
                std::size_t{gpulsmopt2_detail::kQuotients} *
                    gpulsmopt2_detail::kFoundationCells),
        canonical_cell_ranks_(
            gpulsmopt2_detail::kCanonicalCarry
                ? std::size_t{canonical_level_count_} *
                      gpulsmopt2_detail::kLocalRankEntries
                : 1u),
        canonical_cell_counts_(gpulsmopt2_detail::kCanonicalCarry
            ? gpulsmopt2_detail::kLocalRankEntries : 1u),
        canonical_job_prefixes_(gpulsmopt2_detail::kCanonicalCarry
            ? maximum_resident_jobs_ : 1u),
        canonical_next_job_(1u),
        raw_keys_(gpulsmopt2_detail::kBatchesPerEpoch * batch_capacity_),
        raw_payloads_(gpulsmopt2_detail::kBatchesPerEpoch * batch_capacity_),
        raw_offsets_(std::size_t{gpulsmopt2_detail::kBatchesPerEpoch} *
                     (gpulsmopt2_detail::kQuotients + 1u)),
        raw_signatures_(std::size_t{gpulsmopt2_detail::kBatchesPerEpoch} *
                        gpulsmopt2_detail::kQuotients),
        raw_epoch_signatures_(gpulsmopt2_detail::kQuotients),
        publication_epoch_keys_a_(
            batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch),
        publication_epoch_keys_b_(
            batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch),
        publication_epoch_assignments_a_(
            gpulsmopt2_detail::kCanonicalCarry ? 1u :
                batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch),
        publication_epoch_assignments_b_(
            gpulsmopt2_detail::kCanonicalCarry ? 1u :
                batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch),
        publication_epoch_payloads_a_(
            gpulsmopt2_detail::kCanonicalCarry
                ? batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch
                : 1u),
        publication_epoch_payloads_b_(
            gpulsmopt2_detail::kCanonicalCarry
                ? std::max<std::size_t>(
                      batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch,
                      gpulsmopt2_detail::kCanonicalResolverSuffixes)
                : 1u),
        publication_keys_a_(gpulsmopt2_detail::kMaximumPublicationRows,
            gpulsmopt2_detail::kCanonicalCarry
                ? std::min(publication_capacity_,
                      batch_capacity_ *
                          gpulsmopt2_detail::kBatchesPerEpoch)
                : publication_capacity_),
        publication_rows_a_(gpulsmopt2_detail::kMaximumPublicationRows,
            gpulsmopt2_detail::kCanonicalCarry
                ? std::min(publication_capacity_,
                      batch_capacity_ *
                          gpulsmopt2_detail::kBatchesPerEpoch)
                : publication_capacity_),
        publication_selected_count_(1u),
        publication_batch_offsets_(gpulsmopt2_detail::kBatchesPerEpoch + 1u),
        foundation_source_offsets_(gpulsmopt2_detail::kQuotients + 1u),
        foundation_next_route_headers_(gpulsmopt2_detail::kCanonicalCarry
            ? 1u : gpulsmopt2_detail::kQuotients),
        foundation_section_output_counts_(gpulsmopt2_detail::kQuotients + 1u),
        balanced_merge_raw_counts_(gpulsmopt2_detail::kQuotients),
        resident_tile_job_counts_(gpulsmopt2_detail::kPlanningTiles + 1u),
        resident_tile_job_offsets_(gpulsmopt2_detail::kPlanningTiles + 1u),
        resident_job_raw_reservations_(maximum_resident_jobs_ + 1u),
        resident_job_output_offsets_(gpulsmopt2_detail::kCanonicalCarry
            ? 1u : maximum_resident_jobs_ + 1u),
        resident_route_counts_(gpulsmopt2_detail::kCanonicalCarry
            ? 1u : gpulsmopt2_detail::kQuotients + 1u),
        resident_route_offsets_(gpulsmopt2_detail::kCanonicalCarry
            ? 1u : gpulsmopt2_detail::kQuotients + 1u),
        resident_section_logical_offsets_(
            gpulsmopt2_detail::kCanonicalCarry
                ? 1u : gpulsmopt2_detail::kQuotients + 1u),
        balanced_merge_jobs_(maximum_resident_jobs_),
        foundation_overflow_flag_(1u),
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
    initialize_publication_workspace();
    initialize_resident_workspace();
    if (gpulsmopt2_detail::kCanonicalCarry) {
      initialize_canonical_workspace();
      if (gpulsmopt2_detail::kCanonicalPublicationGraph)
        initialize_canonical_publication_graphs();
    } else
      initialize_resident_publication_graph();
    CUDA_CHECK(cudaEventRecord(operation_done_, 0));
    reset_updates(0);
    CUDA_CHECK(cudaMemset(local_rank_.data(), 0,
                          local_rank_.size() * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaEventRecord(operation_done_, 0));
  }

  GPULSMOpt(const GPULSMOpt &) = delete;
  GPULSMOpt &operator=(const GPULSMOpt &) = delete;

  ~GPULSMOpt() {
    if (operation_done_) {
      cudaEventSynchronize(operation_done_);
      cudaEventDestroy(operation_done_);
    }
    if (resident_publication_graph_exec_)
      cudaGraphExecDestroy(resident_publication_graph_exec_);
    if (resident_publication_graph_)
      cudaGraphDestroy(resident_publication_graph_);
    if (canonical_direct_publication_graph_exec_)
      cudaGraphExecDestroy(canonical_direct_publication_graph_exec_);
    for (cudaGraphExec_t graph_exec : canonical_publication_graph_execs_)
      if (graph_exec) cudaGraphExecDestroy(graph_exec);
  }

  void clear(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    begin_operation(stream);
    CUDA_CHECK(cudaMemsetAsync(local_rank_.data(), 0,
                               local_rank_.size() * sizeof(std::uint16_t),
                               stream));
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
    CUDA_CHECK(cudaMemsetAsync(local_rank_.data(), 0,
                               local_rank_.size() * sizeof(std::uint16_t),
                               stream));
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

    const std::uint32_t level = initial_level_for_records(count);
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
    if (gpulsmopt2_detail::kCanonicalCarry) {
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
      return;
    }
    if (base_count > foundation_pool_capacity_ / 2u)
      throw std::bad_alloc();
    gpulsmopt2_detail::store_resident_rows_kernel<<<
        blocks(base_count), gpulsmopt2_detail::kThreads, 0, stream>>>(
            publication_rows_a_.data(), base_count, resident_rows());
    CUDA_CHECK(cudaGetLastError());
    gpulsmopt2_detail::publish_foundation_build_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            foundation_source_offsets_.data(), level, descriptors_.data());
    CUDA_CHECK(cudaGetLastError());
    level_counts_[level] = base_count;
    host_occupied_level_mask_ = std::uint64_t{1} << level;
    gpulsmopt2_detail::publish_single_route_directory_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            descriptors_.data(), level,
            static_cast<std::uint32_t>(route_stride_),
            route_headers_.data(), route_slices_.data());
    gpulsmopt2_detail::initialize_single_route_auxiliary_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            foundation_source_offsets_.data(), level,
            static_cast<std::uint32_t>(route_stride_),
            route_logical_begins_.data(), route_quotients_.data(),
            level_q_logical_offsets_.data());
    gpulsmopt2_detail::initialize_device_manifest_kernel<<<1, 1, 0, stream>>>(
        device_manifests_.data(), active_device_manifest_.data(),
        query_occupied_level_mask_.data(), level, base_count, 0u);
    refresh_active_levels();
    rebuild_foundation_rank(stream);
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

  void set_rank_cell_standalone_profile(bool enabled) {
    rank_cell_standalone_profile_ = enabled;
  }

  RankCellProfileStats rank_cell_profile_stats() const {
    return rank_cell_profile_stats_;
  }

  void set_direct_standalone_profile(bool enabled) {
    direct_standalone_profile_ = enabled;
  }

  DirectMergeProfileStats direct_profile_stats() const {
    return direct_profile_stats_;
  }

  void set_direct_correctness_audit(bool enabled) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    direct_correctness_audit_enabled_ = enabled;
    if (enabled && !direct_output_audit_device_.size()) {
      direct_output_audit_device_.resize(1u);
      direct_audit_raw_rows_.resize(maximum_resident_jobs_);
    }
  }

  DirectCorrectnessAuditStats direct_correctness_audit_stats() const {
    return direct_correctness_audit_stats_;
  }

  void set_partition_audit(bool enabled) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    partition_audit_enabled_ = enabled;
    if (enabled && !partition_audit_device_jobs_.size()) {
      partition_audit_device_jobs_.resize(maximum_resident_jobs_);
      partition_audit_source_rows_.resize(
          maximum_resident_jobs_ * gpulsmopt2_detail::kMaximumMergeSources);
    }
  }

  PartitionAuditStats partition_audit_stats() const {
    return partition_audit_stats_;
  }

  CanonicalCarryStats canonical_carry_stats() const {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    auto *self = const_cast<GPULSMOpt *>(this);
    self->resolve_publication_receipt();
    gpulsmopt2_detail::ResidentPublicationPlan plan{};
    std::uint32_t claimed = 0u;
    std::uint32_t total = 0u;
    CUDA_CHECK(cudaMemcpy(
        &plan, self->resident_plan_.data(), sizeof(plan),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &claimed, self->canonical_next_job_.data(), sizeof(claimed),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &total,
        self->foundation_source_offsets_.data() +
            gpulsmopt2_detail::kQuotients,
        sizeof(total), cudaMemcpyDeviceToHost));
    return {plan.destination_level, plan.source_count, plan.status,
            plan.job_count, claimed, plan.selected_count,
            plan.raw_reservation, plan.survivor_count, total};
  }

  void audit_direct_lookup(
      const std::uint32_t *queries, std::size_t count,
      std::uint32_t *job_results, std::uint32_t *route_results,
      cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    if (!count) return;
    if (!direct_correctness_audit_enabled_ || !queries || !job_results ||
        !route_results)
      throw std::invalid_argument("invalid GPULSMOpt direct audit lookup");
    begin_operation(stream);
    const std::uint32_t n = static_cast<std::uint32_t>(count);
    gpulsmopt2_detail::audit_lookup_jobs_kernel<<<
        blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
            queries, n, balanced_merge_jobs_.data(), resident_plan_.data(),
            resident_rows(), job_results);
    gpulsmopt2_detail::audit_lookup_routes_kernel<<<
        blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
            queries, n, resident_plan_.data(), resident_rows(),
            descriptors_.data(), route_headers_.data(), route_slices_.data(),
            route_results);
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
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
    if (gpulsmopt2_detail::kCanonicalCarry) {
      if (use_dense_router) {
        gpulsmopt2_detail::lookup_with_dense_router_kernel<<<
            blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
                queries, batch.out_values, batch.out_found, count,
                raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
                static_cast<std::uint32_t>(batch_capacity_),
                pending_batches_, raw_epoch_signatures_.data(),
                resident_rows(), descriptors_.data(), route_headers_.data(),
                route_slices_.data(), route_logical_begins_.data(),
                level_q_logical_offsets_.data(), local_rank_.data(),
                level_guides_.data(), level_cell_rank_blocks_.data(),
                level_cell_ranks_.data(), query_ids,
                canonical_cell_ranks_.data(), true,
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
      return;
    }
    if (grouped) {
      if (use_dense_router) {
        gpulsmopt2_detail::lookup_with_dense_router_kernel<<<
            blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
            queries, batch.out_values, batch.out_found, count,
            raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
            static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
            raw_epoch_signatures_.data(), resident_rows(), descriptors_.data(),
            route_headers_.data(), route_slices_.data(),
            route_logical_begins_.data(), level_q_logical_offsets_.data(),
            local_rank_.data(), level_guides_.data(),
            level_cell_rank_blocks_.data(), level_cell_ranks_.data(),
            query_ids, nullptr, false,
            query_occupied_level_mask_.data());
      } else {
        gpulsmopt2_detail::lookup_with_pending_kernel<<<
            blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
            queries, batch.out_values, batch.out_found, count,
            raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
            static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
            raw_signatures_.data(), raw_epoch_signatures_.data(),
            resident_rows(), descriptors_.data(), route_headers_.data(),
            route_slices_.data(), route_logical_begins_.data(),
            level_q_logical_offsets_.data(), local_rank_.data(),
            level_guides_.data(), level_cell_rank_blocks_.data(),
            level_cell_ranks_.data(), query_ids,
            query_occupied_level_mask_.data());
      }
      CUDA_CHECK(cudaGetLastError());
    } else {
      gpulsmopt2_detail::lookup_with_pending_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          batch.queries, batch.out_values, batch.out_found, count,
          raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
          static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
          raw_signatures_.data(),
          raw_epoch_signatures_.data(), resident_rows(), descriptors_.data(),
          route_headers_.data(), route_slices_.data(),
          route_logical_begins_.data(), level_q_logical_offsets_.data(),
          local_rank_.data(), level_guides_.data(),
          level_cell_rank_blocks_.data(), level_cell_ranks_.data(),
          nullptr, query_occupied_level_mask_.data());
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
  MaintenanceStats maintenance_stats() const {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    const_cast<GPULSMOpt *>(this)->resolve_publication_receipt();
    MaintenanceStats result = stats_;
    result.pending_batches = pending_batches_;
    result.pending_records = pending_records_;
    result.active_levels = active_levels_;
    return result;
  }

  std::size_t gpu_resident_bytes() const {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    const_cast<GPULSMOpt *>(this)->resolve_publication_receipt();
    const std::size_t rollover_rank_bytes = canonical_rollover_epoch_ranks_
        ? canonical_rollover_epoch_ranks_->size() * sizeof(std::uint16_t)
        : 0u;
    return rollover_rank_bytes +
        local_rank_.size() * sizeof(std::uint16_t) +
        level_guides_.size() * sizeof(std::uint16_t) +
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
        (staged_epoch_mode_.size() + rank_cell_mode_.size()) *
            sizeof(std::uint32_t) +
        resident_plan_.size() *
            sizeof(gpulsmopt2_detail::ResidentPublicationPlan) +
        level_storage_spans_.size() *
            sizeof(gpulsmopt2_detail::LevelStorageSpan) +
        level_rank_spans_.size() *
            sizeof(gpulsmopt2_detail::LevelRankSpan) +
        level_cell_rank_blocks_.size() * sizeof(std::uint32_t) +
        level_cell_ranks_.size() * sizeof(std::uint16_t) +
        publication_cell_ranks_.size() * sizeof(std::uint16_t) +
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
        (publication_epoch_keys_a_.size() +
         publication_epoch_keys_b_.size()) * sizeof(std::uint32_t) +
        (publication_epoch_assignments_a_.size() +
         publication_epoch_assignments_b_.size()) *
            sizeof(gpulsmopt2_detail::RawAssignment) +
        (publication_epoch_payloads_a_.size() +
         publication_epoch_payloads_b_.size()) *
            sizeof(gpulsmopt2_detail::RawPayload) +
        publication_rows_a_.size() * sizeof(gpulsmopt2_detail::Row) +
        (publication_keys_a_.size() +
         publication_selected_count_.size() +
         publication_batch_offsets_.size()) * sizeof(std::uint32_t) +
        (foundation_source_offsets_.size() +
         foundation_section_output_counts_.size() +
         foundation_overflow_flag_.size() +
         local_epoch_overflow_flag_.size()) * sizeof(std::uint32_t) +
        balanced_merge_raw_counts_.size() * sizeof(std::uint64_t) +
        (resident_tile_job_counts_.size() +
         resident_tile_job_offsets_.size() +
         resident_route_counts_.size() +
         resident_route_offsets_.size() +
         resident_section_logical_offsets_.size()) * sizeof(std::uint32_t) +
        (resident_job_raw_reservations_.size() +
         resident_job_output_offsets_.size()) * sizeof(std::uint64_t) +
        balanced_merge_jobs_.size() *
            sizeof(gpulsmopt2_detail::BalancedMergeJob) +
        foundation_next_route_headers_.size() *
            sizeof(gpulsmopt2_detail::RouteHeader) +
        resident_scan_temp_.size() * sizeof(std::uint8_t) +
        publication_temp_.size() * sizeof(std::uint8_t) +
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
    std::size_t capacity = level_zero_capacity_;
    std::uint32_t level = 0u;
    while (level + 1u < gpulsmopt2_detail::kMaximumLevels &&
           capacity <= count / 2u) {
      capacity *= 2u;
      ++level;
    }
    return level;
  }

  std::uint64_t level_begin(std::uint32_t target) const {
    std::uint64_t begin = foundation_pool_capacity_;
    std::size_t capacity = level_zero_capacity_;
    for (std::uint32_t level = 0u; level < target; ++level) {
      begin += capacity;
      capacity = capacity > publication_capacity_ / 2u
          ? publication_capacity_ : capacity * 2u;
    }
    return begin;
  }

  std::uint64_t level_capacity(std::uint32_t target) const {
    std::size_t capacity = level_zero_capacity_;
    for (std::uint32_t level = 0u; level < target; ++level)
      capacity = capacity > publication_capacity_ / 2u
          ? publication_capacity_ : capacity * 2u;
    return capacity;
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

  void rebuild_foundation_rank(cudaStream_t stream) {
    if (!active_levels_) {
      CUDA_CHECK(cudaMemsetAsync(local_rank_.data(), 0,
                                 local_rank_.size() * sizeof(std::uint16_t),
                                 stream));
      return;
    }
    gpulsmopt2_detail::build_foundation_rank_directory_kernel<<<
        gpulsmopt2_detail::kQuotients, 128u, 0, stream>>>(
            resident_rows(), descriptors_.data(), foundation_level(),
            local_rank_.data());
    CUDA_CHECK(cudaGetLastError());
  }


  void initialize_publication_workspace() {
    const std::uint32_t epoch_capacity = static_cast<std::uint32_t>(
        batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch);
    std::size_t sort_bytes{}, raw_reduce_bytes{};
    if (gpulsmopt2_detail::kCanonicalCarry) {
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, sort_bytes, publication_epoch_keys_a_.data(),
          publication_epoch_keys_b_.data(),
          publication_epoch_payloads_a_.data(),
          publication_epoch_payloads_b_.data(), epoch_capacity, 0, 32, 0));
      CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
          nullptr, raw_reduce_bytes, publication_epoch_keys_b_.data(),
          publication_keys_a_.data(), publication_epoch_payloads_b_.data(),
          publication_epoch_payloads_a_.data(),
          publication_selected_count_.data(),
          gpulsmopt2_detail::NewestPayload{}, epoch_capacity, 0));
    } else {
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, sort_bytes, publication_epoch_keys_a_.data(),
          publication_epoch_keys_b_.data(),
          publication_epoch_assignments_a_.data(),
          publication_epoch_assignments_b_.data(), epoch_capacity, 0, 32,
          0));
      auto row_output = thrust::make_transform_output_iterator(
          publication_rows_a_.data(), gpulsmopt2_detail::AssignmentRow{});
      CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
          nullptr, raw_reduce_bytes, publication_epoch_keys_b_.data(),
          publication_keys_a_.data(),
          publication_epoch_assignments_b_.data(), row_output,
          publication_selected_count_.data(),
          gpulsmopt2_detail::NewestAssignment{}, epoch_capacity, 0));
    }
    publication_temp_.resize(std::max(sort_bytes, raw_reduce_bytes));
  }

  void initialize_resident_workspace() {
    std::array<gpulsmopt2_detail::LevelStorageSpan,
               gpulsmopt2_detail::kMaximumLevels> spans{};
    std::array<gpulsmopt2_detail::LevelRankSpan,
               gpulsmopt2_detail::kMaximumLevels> rank_spans{};
    std::uint64_t cursor = foundation_pool_capacity_;
    std::uint64_t rank_cursor = 0u;
    std::size_t capacity = level_zero_capacity_;
    for (std::uint32_t level = 0u;
         level < gpulsmopt2_detail::kMaximumLevels; ++level) {
      spans[level] = {cursor, capacity};
      const std::uint32_t rank_blocks = static_cast<std::uint32_t>(
          std::min<std::size_t>(
              gpulsmopt2_detail::kQuotients,
              (capacity +
               gpulsmopt2_detail::kDenseCellRankMinimumRows - 1u) /
                  gpulsmopt2_detail::kDenseCellRankMinimumRows));
      rank_spans[level] = {rank_cursor, rank_blocks};
      rank_cursor += rank_blocks;
      cursor += capacity;
      if (capacity == publication_capacity_) break;
      capacity = capacity > publication_capacity_ / 2u
          ? publication_capacity_ : capacity * 2u;
    }
    if (cursor > foundation_pool_capacity_ + level_pool_capacity_)
      throw std::logic_error("GPULSMOpt preassigned level spans overflow");
    if (rank_cursor > level_rank_block_capacity_)
      throw std::logic_error("GPULSMOpt preassigned rank spans overflow");
    CUDA_CHECK(cudaMemcpy(
        level_storage_spans_.data(), spans.data(), sizeof(spans),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        level_rank_spans_.data(), rank_spans.data(), sizeof(rank_spans),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(
        level_cell_rank_blocks_.data(), 0xff,
        level_cell_rank_blocks_.size() * sizeof(std::uint32_t)));
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
    if (gpulsmopt2_detail::kCanonicalCarry) {
      bytes = 0u;
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          nullptr, bytes, foundation_section_output_counts_.data(),
          foundation_source_offsets_.data(),
          gpulsmopt2_detail::kQuotients + 1u, 0));
      maximum_scan_bytes = std::max(maximum_scan_bytes, bytes);
    } else {
      bytes = 0u;
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          nullptr, bytes, resident_job_raw_reservations_.data(),
          resident_job_output_offsets_.data(), maximum_resident_jobs_ + 1u,
          0));
      maximum_scan_bytes = std::max(maximum_scan_bytes, bytes);
      bytes = 0u;
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          nullptr, bytes, resident_route_counts_.data(),
          resident_route_offsets_.data(),
          gpulsmopt2_detail::kQuotients + 1u, 0));
      maximum_scan_bytes = std::max(maximum_scan_bytes, bytes);
      bytes = 0u;
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          nullptr, bytes, foundation_section_output_counts_.data(),
          resident_section_logical_offsets_.data(),
          gpulsmopt2_detail::kQuotients + 1u, 0));
      maximum_scan_bytes = std::max(maximum_scan_bytes, bytes);
    }
    resident_scan_temp_.resize(maximum_scan_bytes);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    int blocks_per_sm = 0;
    if (!gpulsmopt2_detail::kCanonicalCarry) {
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm,
          gpulsmopt2_detail::compact_direct_epoch_jobs_kernel,
          gpulsmopt2_detail::kFoundationCompactionThreads,
          resident_merge_workspace_bytes_));
      resident_merge_blocks_ = static_cast<std::uint32_t>(
          std::max(1, blocks_per_sm) * properties.multiProcessorCount);
      blocks_per_sm = 0;
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm,
          gpulsmopt2_detail::compact_rank_cell_sections_kernel,
          gpulsmopt2_detail::kRankCellThreads,
          gpulsmopt2_detail::kRankCellDynamicSharedBytes));
      resident_rank_cell_blocks_ = static_cast<std::uint32_t>(
          std::max(1, std::min(4, blocks_per_sm)) *
          properties.multiProcessorCount);
    }
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
        gpulsmopt2_detail::canonical_carry_jobs_kernel<2u>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(canonical_merge_workspace_bytes_)));
    CUDA_CHECK(cudaFuncSetAttribute(
        gpulsmopt2_detail::canonical_carry_jobs_kernel<0u>,
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
        publication_epoch_payloads_b_.size() /
        gpulsmopt2_detail::kCanonicalResolverSuffixes);
    canonical_epoch_workspace_slots_ =
        std::max(1u, canonical_epoch_workspace_slots_);

    const std::uint32_t maximum_sources = std::min(
        gpulsmopt2_detail::kMaximumMergeSources,
        std::max(1u, canonical_level_count_));
    canonical_local_epoch_enabled_ =
        gpulsmopt2_detail::kCanonicalLocalEpoch;
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
    if (gpulsmopt2_detail::kCanonicalTournamentMerge) {
      for (std::uint32_t source_count =
               gpulsmopt2_detail::kCanonicalTournamentMinimumSources;
           source_count <= maximum_sources; ++source_count) {
        const auto active_blocks = [&](std::uint32_t capacity) {
          const std::size_t shared_bytes =
              gpulsmopt2_detail::canonical_tournament_dynamic_shared_bytes(
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
        // Grow jobs only while preserving the occupancy supported by the
        // original capacity on this device.
        const int desired_blocks = baseline_blocks;
        std::uint32_t low = resident_merge_capacity_;
        std::uint32_t high = gpulsmopt2_detail::kCanonicalCompactMultiway
            ? std::max(low,
                  gpulsmopt2_detail::kCanonicalTournamentCapacityCeiling)
            : low;
        while (low < high) {
          const std::uint32_t middle =
              low + (high - low + 1u) / 2u;
          if (active_blocks(middle) >= desired_blocks)
            low = middle;
          else
            high = middle - 1u;
        }
        const std::uint32_t capacity = low;
        const std::size_t shared_bytes =
            gpulsmopt2_detail::canonical_tournament_dynamic_shared_bytes(
                capacity, source_count);
        canonical_job_capacities_[source_count] = capacity;
        canonical_tournament_shared_bytes_[source_count] = shared_bytes;
        tournament_attribute_bytes =
            std::max(tournament_attribute_bytes, shared_bytes);
      }
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
      if (!blocks_per_sm) continue;
      canonical_tournament_blocks_[source_count] =
          static_cast<std::uint32_t>(
              blocks_per_sm * properties.multiProcessorCount);
    }

    blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        gpulsmopt2_detail::canonical_carry_jobs_kernel<2u>,
        gpulsmopt2_detail::kFoundationCompactionThreads,
        canonical_merge_workspace_bytes_));
    canonical_two_way_blocks_ = static_cast<std::uint32_t>(
        std::max(1, blocks_per_sm) * properties.multiProcessorCount);

    blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        gpulsmopt2_detail::canonical_carry_jobs_kernel<0u>,
        gpulsmopt2_detail::kFoundationCompactionThreads,
        canonical_merge_workspace_bytes_));
    canonical_merge_blocks_ = static_cast<std::uint32_t>(
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
    canonical_direct_publication_graph_exec_ =
        capture_canonical_publication_graph(
            capture_stream, 0u, 1u, true);
    const std::uint32_t maximum_sources = std::min<std::uint32_t>(
        canonical_level_count_,
        static_cast<std::uint32_t>(
            canonical_publication_graph_execs_.size() - 1u));
    for (std::uint32_t source_count = 1u;
         source_count <= maximum_sources; ++source_count)
      canonical_publication_graph_execs_[source_count] =
          capture_canonical_publication_graph(
              capture_stream, source_count - 1u, source_count, false);
    CUDA_CHECK(cudaStreamDestroy(capture_stream));
  }

  void launch_resident_merge_pre(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(
        foundation_overflow_flag_.data(), 0, sizeof(std::uint32_t),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        local_epoch_overflow_flag_.data(), 0, sizeof(std::uint32_t),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        foundation_section_output_counts_.data(), 0,
        foundation_section_output_counts_.size() * sizeof(std::uint32_t),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        staged_epoch_mode_.data(), 0, sizeof(std::uint32_t),
        stream));
    gpulsmopt2_detail::count_direct_epoch_merge_work_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            raw_offsets_.data(), gpulsmopt2_detail::kBatchesPerEpoch,
            descriptors_.data(), device_manifests_.data(),
            active_device_manifest_.data(), resident_plan_.data(),
            balanced_merge_raw_counts_.data(),
            foundation_section_output_counts_.data(),
            foundation_overflow_flag_.data(),
            local_epoch_overflow_flag_.data());
  }

  cudaGraph_t capture_resident_merge_pre_graph(
      cudaStream_t capture_stream) {
    cudaGraph_t graph{};
    CUDA_CHECK(cudaStreamBeginCapture(
        capture_stream, cudaStreamCaptureModeThreadLocal));
    launch_resident_merge_pre(capture_stream);
    CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
    return graph;
  }

  void launch_crowded_epoch_stage(cudaStream_t stream) {
    gpulsmopt2_detail::build_publication_batch_offsets_kernel<<<
        1, 1, 0, stream>>>(
            raw_offsets_.data(), publication_batch_offsets_.data());
    const dim3 publication_grid(
        blocks(batch_capacity_), gpulsmopt2_detail::kBatchesPerEpoch);
    gpulsmopt2_detail::pack_publication_epoch_kernel<<<
        publication_grid, gpulsmopt2_detail::kThreads, 0, stream>>>(
            raw_keys_.data(), raw_payloads_.data(),
            static_cast<std::uint32_t>(batch_capacity_),
            publication_batch_offsets_.data(),
            publication_epoch_keys_a_.data(),
            publication_epoch_assignments_a_.data());
    const std::uint32_t epoch_capacity = static_cast<std::uint32_t>(
        batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch);
    gpulsmopt2_detail::pad_publication_epoch_kernel<<<
        blocks(epoch_capacity), gpulsmopt2_detail::kThreads, 0,
        stream>>>(
            epoch_capacity, publication_batch_offsets_.data(),
            publication_epoch_keys_a_.data(),
            publication_epoch_assignments_a_.data());
    std::size_t workspace_bytes = publication_temp_.size();
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        publication_temp_.data(), workspace_bytes,
        publication_epoch_keys_a_.data(), publication_epoch_keys_b_.data(),
        publication_epoch_assignments_a_.data(),
        publication_epoch_assignments_b_.data(), epoch_capacity, 0, 32,
        stream));
    auto row_output = thrust::make_transform_output_iterator(
        publication_rows_a_.data(), gpulsmopt2_detail::AssignmentRow{});
    workspace_bytes = publication_temp_.size();
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        publication_temp_.data(), workspace_bytes,
        publication_epoch_keys_b_.data(), publication_keys_a_.data(),
        publication_epoch_assignments_b_.data(), row_output,
        publication_selected_count_.data(),
        gpulsmopt2_detail::NewestAssignment{}, epoch_capacity,
        stream));
    gpulsmopt2_detail::set_staged_epoch_mode_kernel<<<
        1, 1, 0, stream>>>(staged_epoch_mode_.data());
    gpulsmopt2_detail::build_query_quotient_offsets_device_count_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            publication_keys_a_.data(), publication_selected_count_.data(),
            foundation_source_offsets_.data());
    gpulsmopt2_detail::build_staged_rank_directory_kernel<<<
        gpulsmopt2_detail::kQuotients,
        gpulsmopt2_detail::kFoundationCells, 0, stream>>>(
            publication_rows_a_.data(), foundation_source_offsets_.data(),
            publication_cell_ranks_.data());
    gpulsmopt2_detail::count_resident_merge_work_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            foundation_source_offsets_.data(), descriptors_.data(),
            device_manifests_.data(), active_device_manifest_.data(),
            resident_plan_.data(), balanced_merge_raw_counts_.data());
  }

  cudaGraph_t capture_crowded_epoch_stage_graph(
      cudaStream_t capture_stream) {
    cudaGraph_t graph{};
    CUDA_CHECK(cudaStreamBeginCapture(
        capture_stream, cudaStreamCaptureModeThreadLocal));
    launch_crowded_epoch_stage(capture_stream);
    CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
    return graph;
  }

  void launch_local_epoch_stage(cudaStream_t stream) {
    std::size_t scan_bytes = resident_scan_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        resident_scan_temp_.data(), scan_bytes,
        foundation_section_output_counts_.data(),
        resident_section_logical_offsets_.data(),
        gpulsmopt2_detail::kQuotients + 1u, stream));
    auto *scratch_rows = reinterpret_cast<gpulsmopt2_detail::Row *>(
        publication_epoch_assignments_a_.data());
    gpulsmopt2_detail::resolve_epoch_sections_kernel<<<
        gpulsmopt2_detail::kQuotients,
        gpulsmopt2_detail::kFoundationCompactionThreads,
        0, stream>>>(
            raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
            static_cast<std::uint32_t>(batch_capacity_),
            gpulsmopt2_detail::kBatchesPerEpoch,
            resident_section_logical_offsets_.data(),
            publication_epoch_keys_a_.data(), scratch_rows,
            foundation_section_output_counts_.data(),
            publication_cell_ranks_.data());
    scan_bytes = resident_scan_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        resident_scan_temp_.data(), scan_bytes,
        foundation_section_output_counts_.data(),
        foundation_source_offsets_.data(),
        gpulsmopt2_detail::kQuotients + 1u, stream));
    gpulsmopt2_detail::set_resolved_epoch_count_kernel<<<
        1, 1, 0, stream>>>(
            foundation_source_offsets_.data(),
            publication_selected_count_.data());
    gpulsmopt2_detail::compact_resolved_epoch_sections_kernel<<<
        gpulsmopt2_detail::kQuotients,
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            publication_epoch_keys_a_.data(), scratch_rows,
            resident_section_logical_offsets_.data(),
            foundation_source_offsets_.data(), publication_keys_a_.data(),
            publication_rows_a_.data());
    gpulsmopt2_detail::set_staged_epoch_mode_kernel<<<
        1, 1, 0, stream>>>(staged_epoch_mode_.data());
    gpulsmopt2_detail::count_resident_merge_work_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            foundation_source_offsets_.data(), descriptors_.data(),
            device_manifests_.data(), active_device_manifest_.data(),
            resident_plan_.data(), balanced_merge_raw_counts_.data());
  }

  cudaGraph_t capture_local_epoch_stage_graph(
      cudaStream_t capture_stream) {
    cudaGraph_t graph{};
    CUDA_CHECK(cudaStreamBeginCapture(
        capture_stream, cudaStreamCaptureModeThreadLocal));
    launch_local_epoch_stage(capture_stream);
    CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
    return graph;
  }

  void launch_resident_merge_finish(cudaStream_t stream) {
    cudaStream_t capture_stream = stream;
    CUDA_CHECK(cudaMemsetAsync(
        resident_job_raw_reservations_.data(), 0,
        resident_job_raw_reservations_.size() * sizeof(std::uint64_t),
        capture_stream));
    CUDA_CHECK(cudaMemsetAsync(
        foundation_section_output_counts_.data(), 0,
        foundation_section_output_counts_.size() * sizeof(std::uint32_t),
        capture_stream));
    CUDA_CHECK(cudaMemsetAsync(
        foundation_overflow_flag_.data(), 0, sizeof(std::uint32_t),
        capture_stream));

    gpulsmopt2_detail::initialize_rank_cell_mode_kernel<<<
        1, 1, 0, capture_stream>>>(
            resident_plan_.data(), staged_epoch_mode_.data(),
            rank_cell_mode_.data());
    gpulsmopt2_detail::validate_rank_cell_mode_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, capture_stream>>>(
            balanced_merge_raw_counts_.data(),
            foundation_source_offsets_.data(), descriptors_.data(),
            device_manifests_.data(), active_device_manifest_.data(),
            resident_plan_.data(), route_headers_.data(),
            level_cell_rank_blocks_.data(),
            rank_cell_mode_.data());
    std::size_t scan_bytes = resident_scan_temp_.size();
    gpulsmopt2_detail::count_resident_planning_jobs_kernel<<<
        gpulsmopt2_detail::kPlanningTiles, gpulsmopt2_detail::kThreads, 0,
        capture_stream>>>(
            balanced_merge_raw_counts_.data(), resident_plan_.data(),
            rank_cell_mode_.data(), resident_tile_job_counts_.data());
    scan_bytes = resident_scan_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        resident_scan_temp_.data(), scan_bytes,
        resident_tile_job_counts_.data(), resident_tile_job_offsets_.data(),
        gpulsmopt2_detail::kPlanningTiles + 1u, capture_stream));
    gpulsmopt2_detail::emit_resident_planning_jobs_kernel<<<
        gpulsmopt2_detail::kPlanningTiles, gpulsmopt2_detail::kThreads, 0,
        capture_stream>>>(
            balanced_merge_raw_counts_.data(),
            resident_tile_job_offsets_.data(), resident_plan_.data(),
            static_cast<std::uint32_t>(maximum_resident_jobs_),
            rank_cell_mode_.data(),
            balanced_merge_jobs_.data(), resident_job_raw_reservations_.data());
    gpulsmopt2_detail::resolve_resident_job_boundaries_kernel<<<
        resident_planner_blocks_, 32u, 0,
        capture_stream>>>(
            balanced_merge_jobs_.data(), resident_job_raw_reservations_.data(),
            resident_plan_.data(),
            publication_rows_a_.data(), foundation_source_offsets_.data(),
            resident_rows(), route_headers_.data(), route_slices_.data(),
            route_logical_begins_.data(), level_q_logical_offsets_.data(),
            device_manifests_.data(), active_device_manifest_.data(),
            descriptors_.data(), rank_cell_mode_.data(),
            publication_cell_ranks_.data(),
            level_cell_rank_blocks_.data(), level_cell_ranks_.data(),
            local_rank_.data());
    scan_bytes = resident_scan_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        resident_scan_temp_.data(), scan_bytes,
        resident_job_raw_reservations_.data(),
        resident_job_output_offsets_.data(), maximum_resident_jobs_ + 1u,
        capture_stream));
    gpulsmopt2_detail::assign_resident_output_offsets_kernel<<<
        resident_planner_blocks_, gpulsmopt2_detail::kThreads, 0,
        capture_stream>>>(
            balanced_merge_jobs_.data(),
            resident_job_output_offsets_.data(),
            resident_job_raw_reservations_.data(), resident_plan_.data());

    if (partition_audit_enabled_) {
      gpulsmopt2_detail::audit_partition_feasibility_kernel<<<
          resident_planner_blocks_, 32u, 0, capture_stream>>>(
              balanced_merge_jobs_.data(), resident_plan_.data(),
              publication_rows_a_.data(), foundation_source_offsets_.data(),
              raw_offsets_.data(), gpulsmopt2_detail::kBatchesPerEpoch,
              publication_cell_ranks_.data(), resident_rows(),
              device_manifests_.data(), active_device_manifest_.data(),
              descriptors_.data(), route_headers_.data(),
              route_slices_.data(), route_logical_begins_.data(),
              level_q_logical_offsets_.data(),
              level_cell_rank_blocks_.data(), level_cell_ranks_.data(),
              local_rank_.data(), staged_epoch_mode_.data(),
              rank_cell_mode_.data(), partition_audit_device_jobs_.data(),
              partition_audit_source_rows_.data());
    }

    // The rank-cell reservation preparation below intentionally reuses the
    // planner reservation buffer.  Preserve the direct planner's per-job
    // raw rows for the profiling-only report before that reuse.
    direct_profile_raw_rows_.clear();
    if (direct_standalone_profile_ || direct_correctness_audit_enabled_) {
      CUDA_CHECK(cudaStreamSynchronize(capture_stream));
      gpulsmopt2_detail::ResidentPublicationPlan planned{};
      CUDA_CHECK(cudaMemcpy(
          &planned, resident_plan_.data(), sizeof(planned),
          cudaMemcpyDeviceToHost));
      direct_profile_raw_rows_.resize(planned.job_count);
      if (!direct_profile_raw_rows_.empty())
        CUDA_CHECK(cudaMemcpy(
            direct_profile_raw_rows_.data(), resident_job_raw_reservations_.data(),
            direct_profile_raw_rows_.size() * sizeof(std::uint64_t),
            cudaMemcpyDeviceToHost));
      if (direct_correctness_audit_enabled_ &&
          !direct_profile_raw_rows_.empty())
        CUDA_CHECK(cudaMemcpyAsync(
            direct_audit_raw_rows_.data(), direct_profile_raw_rows_.data(),
            direct_profile_raw_rows_.size() * sizeof(std::uint64_t),
            cudaMemcpyHostToDevice, capture_stream));
    }

    if (direct_correctness_audit_enabled_)
      CUDA_CHECK(cudaMemsetAsync(
          direct_output_audit_device_.data(), 0,
          sizeof(gpulsmopt2_detail::DirectOutputAuditDevice),
          capture_stream));

    gpulsmopt2_detail::count_resident_route_slots_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, capture_stream>>>(
            balanced_merge_raw_counts_.data(), level_rank_spans_.data(),
            resident_plan_.data(), rank_cell_mode_.data(),
            resident_route_counts_.data(),
            level_cell_rank_blocks_.data());
    scan_bytes = resident_scan_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        resident_scan_temp_.data(), scan_bytes,
        resident_route_counts_.data(), resident_route_offsets_.data(),
        gpulsmopt2_detail::kQuotients + 1u, capture_stream));
    gpulsmopt2_detail::prepare_resident_route_headers_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, capture_stream>>>(
            resident_route_counts_.data(), resident_route_offsets_.data(),
            resident_plan_.data(), static_cast<std::uint32_t>(route_stride_),
            foundation_next_route_headers_.data());
    gpulsmopt2_detail::override_rank_cell_route_headers_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, capture_stream>>>(
            balanced_merge_raw_counts_.data(), resident_plan_.data(),
            rank_cell_mode_.data(), static_cast<std::uint32_t>(route_stride_),
            foundation_next_route_headers_.data());

    gpulsmopt2_detail::validate_direct_epoch_plan_kernel<<<
        1, 1, 0, capture_stream>>>(
            resident_plan_.data(), resident_tile_job_offsets_.data(),
            resident_job_output_offsets_.data(),
            resident_route_offsets_.data(),
            static_cast<std::uint32_t>(maximum_resident_jobs_),
            static_cast<std::uint32_t>(route_stride_));
    gpulsmopt2_detail::prepare_rank_cell_reservations_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, capture_stream>>>(
            balanced_merge_raw_counts_.data(), rank_cell_mode_.data(),
            resident_job_raw_reservations_.data());
    scan_bytes = resident_scan_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        resident_scan_temp_.data(), scan_bytes,
        resident_job_raw_reservations_.data(),
        resident_job_output_offsets_.data(),
        gpulsmopt2_detail::kQuotients + 1u, capture_stream));
    gpulsmopt2_detail::finalize_rank_cell_plan_kernel<<<
        1, 1, 0, capture_stream>>>(
            resident_plan_.data(), resident_job_output_offsets_.data(),
            resident_route_offsets_.data(), rank_cell_mode_.data());
    gpulsmopt2_detail::compact_rank_cell_sections_kernel<<<
        resident_rank_cell_blocks_, gpulsmopt2_detail::kRankCellThreads,
        gpulsmopt2_detail::kRankCellDynamicSharedBytes, capture_stream>>>(
            resident_plan_.data(), balanced_merge_raw_counts_.data(),
            resident_job_output_offsets_.data(), rank_cell_mode_.data(),
            publication_rows_a_.data(), foundation_source_offsets_.data(),
            publication_cell_ranks_.data(), resident_rows(),
            device_manifests_.data(), active_device_manifest_.data(),
            descriptors_.data(), route_headers_.data(), route_slices_.data(),
            static_cast<std::uint32_t>(route_stride_),
            route_slices_.data(),
            foundation_section_output_counts_.data(),
            foundation_overflow_flag_.data(),
            level_cell_rank_blocks_.data(), level_cell_ranks_.data(),
            local_rank_.data(), level_guides_.data());
    gpulsmopt2_detail::compact_direct_epoch_jobs_kernel<<<
        resident_merge_blocks_,
        gpulsmopt2_detail::kFoundationCompactionThreads,
        resident_merge_workspace_bytes_,
        capture_stream>>>(
            balanced_merge_jobs_.data(), resident_plan_.data(),
            raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
            static_cast<std::uint32_t>(batch_capacity_),
            gpulsmopt2_detail::kBatchesPerEpoch,
            publication_keys_a_.data(), publication_rows_a_.data(),
            foundation_source_offsets_.data(),
            staged_epoch_mode_.data(), rank_cell_mode_.data(),
            resident_rows(),
            device_manifests_.data(), active_device_manifest_.data(),
            descriptors_.data(), route_headers_.data(), route_slices_.data(),
            route_logical_begins_.data(), level_q_logical_offsets_.data(),
            foundation_next_route_headers_.data(), route_slices_.data(),
            foundation_section_output_counts_.data(),
            foundation_overflow_flag_.data(),
            level_cell_rank_blocks_.data(), level_cell_ranks_.data(),
            local_rank_.data(), level_guides_.data()
#if defined(GPULSMOPT_FORCE_UNIFIED_MERGE) && \
    !defined(GPULSMOPT_FORCE_UNIFIED_COMPILE_ELIDE)
            , true
#endif
            );
    if (direct_correctness_audit_enabled_)
      gpulsmopt2_detail::audit_direct_output_kernel<<<
          resident_merge_blocks_,
          gpulsmopt2_detail::kFoundationCompactionThreads, 0,
          capture_stream>>>(
              balanced_merge_jobs_.data(), resident_plan_.data(),
              direct_audit_raw_rows_.data(), resident_rows(),
              direct_output_audit_device_.data());
    gpulsmopt2_detail::fold_resident_merge_status_kernel<<<
        1, 1, 0, capture_stream>>>(
            resident_plan_.data(), foundation_overflow_flag_.data());
    scan_bytes = resident_scan_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        resident_scan_temp_.data(), scan_bytes,
        foundation_section_output_counts_.data(),
        resident_section_logical_offsets_.data(),
        gpulsmopt2_detail::kQuotients + 1u, capture_stream));
    gpulsmopt2_detail::set_merged_survivor_count_kernel<<<
        1, 1, 0, capture_stream>>>(
            resident_plan_.data(),
            resident_section_logical_offsets_.data());
    gpulsmopt2_detail::finalize_resident_route_metadata_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, capture_stream>>>(
            resident_plan_.data(),
            foundation_next_route_headers_.data(),
            resident_section_logical_offsets_.data(),
            static_cast<std::uint32_t>(route_stride_),
            route_slices_.data(), route_headers_.data(), descriptors_.data(),
            route_logical_begins_.data(), route_quotients_.data(),
            level_q_logical_offsets_.data());
    gpulsmopt2_detail::build_split_resident_query_metadata_kernel<<<
        resident_planner_blocks_, 128u, 0, capture_stream>>>(
            resident_plan_.data(), resident_rows(), descriptors_.data(),
            route_headers_.data(), route_slices_.data(),
            route_logical_begins_.data(), level_q_logical_offsets_.data(),
            level_cell_rank_blocks_.data(), level_cell_ranks_.data(),
            local_rank_.data(), level_guides_.data());
    gpulsmopt2_detail::publish_resident_manifest_kernel<<<
        1, gpulsmopt2_detail::kMaximumLevels, 0, capture_stream>>>(
            resident_plan_.data(), device_manifests_.data(),
            active_device_manifest_.data(),
            query_occupied_level_mask_.data());
  }

  cudaGraph_t capture_resident_merge_finish_graph(
      cudaStream_t capture_stream) {
    cudaGraph_t graph{};
    CUDA_CHECK(cudaStreamBeginCapture(
        capture_stream, cudaStreamCaptureModeThreadLocal));
    launch_resident_merge_finish(capture_stream);
    CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
    return graph;
  }

  void launch_normal_epoch_stage(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(
        staged_epoch_mode_.data(), 0, sizeof(std::uint32_t), stream));
  }

  void collect_partition_audit(
      const gpulsmopt2_detail::ResidentPublicationPlan &plan,
      std::uint32_t rank_cell_mode) {
    partition_audit_stats_ = {};
    auto &stats = partition_audit_stats_;
    stats.enabled = partition_audit_enabled_;
    stats.selected_count = plan.selected_count;
    stats.source_count = plan.source_count;
    stats.destination_level = plan.destination_level;
    stats.rank_cell_mode = rank_cell_mode;
    // In rank-cell mode finalize_rank_cell_plan_kernel repurposes job_count as
    // the route-cell count.  No BalancedMergeJob entries execute in that mode,
    // so do not mistake stale planner slots for direct/general-merge jobs.
    stats.job_count = rank_cell_mode ? 0u : plan.job_count;
    if (!partition_audit_enabled_ || rank_cell_mode || !plan.job_count ||
        plan.status)
      return;

    std::vector<gpulsmopt2_detail::PartitionAuditDeviceJob> jobs(
        plan.job_count);
    std::vector<std::uint16_t> source_rows(
        std::size_t{plan.job_count} *
        gpulsmopt2_detail::kMaximumMergeSources);
    CUDA_CHECK(cudaMemcpy(
        jobs.data(), partition_audit_device_jobs_.data(),
        jobs.size() * sizeof(jobs[0]), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        source_rows.data(), partition_audit_source_rows_.data(),
        source_rows.size() * sizeof(source_rows[0]),
        cudaMemcpyDeviceToHost));

    std::vector<std::uint32_t> imbalances;
    imbalances.reserve(jobs.size());
    const std::uint32_t locator_bits = plan.job_capacity <= 1u
        ? 0u : 32u - static_cast<std::uint32_t>(
              __builtin_clz(plan.job_capacity - 1u));
    const std::uint32_t locator_limit =
        locator_bits >= 32u ? ~0u : std::uint32_t{1} << locator_bits;
    for (std::size_t job_index = 0u; job_index < jobs.size(); ++job_index) {
      const auto &job = jobs[job_index];
      stats.total_rows += job.task_rows;
      stats.direct_rows += job.direct_path ? job.task_rows : 0u;
      stats.resolved_rows += job.resolved ? job.task_rows : 0u;
      stats.rank_supported_rows += job.rank_supported ? job.task_rows : 0u;
      stats.eligible_rows += job.eligible ? job.task_rows : 0u;
      stats.direct_jobs += job.direct_path;
      stats.resolved_jobs += job.resolved;
      stats.rank_supported_jobs += job.rank_supported;
      stats.eligible_jobs += job.eligible;
      stats.packed_word_failure_jobs += job.packed_bits > 32u;
      stats.duplicate_boundary_count += job.duplicate_boundaries;
      stats.maximum_boundary_equal_rows = std::max<std::uint32_t>(
          stats.maximum_boundary_equal_rows,
          job.maximum_boundary_equal_rows);
      stats.maximum_source_rows = std::max(
          stats.maximum_source_rows, job.max_source_rows);
      stats.maximum_packed_bits = std::max<std::uint32_t>(
          stats.maximum_packed_bits, job.packed_bits);
      if (job.imbalance_ppm) imbalances.push_back(job.imbalance_ppm);
      const std::uint32_t nonempty_bucket = std::min<std::uint32_t>(
          job.nonempty_sources,
          gpulsmopt2_detail::kMaximumMergeSources);
      ++stats.nonempty_source_histogram[nonempty_bucket];

      const std::uint16_t *runs = source_rows.data() +
          job_index * gpulsmopt2_detail::kMaximumMergeSources;
      std::vector<std::pair<std::uint32_t, std::uint32_t>> sources;
      std::uint32_t candidate = 0u;
      for (std::uint32_t source = 0u;
           source < gpulsmopt2_detail::kMaximumMergeSources; ++source) {
        const std::uint32_t count = runs[source];
        if (!count) continue;
        const std::uint32_t bucket = std::min<std::uint32_t>(
            16u, 31u - static_cast<std::uint32_t>(__builtin_clz(count)));
        ++stats.source_run_log2_histogram[bucket];
        sources.emplace_back(source, candidate);
        candidate += count;
      }
      if (candidate > locator_limit)
        ++stats.ordering_equivalence_violations;
      constexpr std::uint32_t key = 0x5a5au;
      for (std::size_t left = 0u; left < sources.size(); ++left) {
        for (std::size_t right = left + 1u;
             right < sources.size(); ++right) {
          const std::uint32_t left_source = sources[left].first;
          const std::uint32_t right_source = sources[right].first;
          const std::uint32_t left_id = sources[left].second;
          const std::uint32_t right_id = sources[right].second;
          const auto left_token = gpulsmopt2_detail::make_candidate_token(
              0u, key, left_source);
          const auto right_token = gpulsmopt2_detail::make_candidate_token(
              0u, key, right_source);
          const bool current_less = left_token != right_token
              ? left_token < right_token : left_id < right_id;
          const std::uint32_t left_packed =
              (key << locator_bits) | left_id;
          const std::uint32_t right_packed =
              (key << locator_bits) | right_id;
          if (current_less != (left_packed < right_packed))
            ++stats.ordering_equivalence_violations;
        }
      }
    }
    if (!imbalances.empty()) {
      std::sort(imbalances.begin(), imbalances.end());
      const auto percentile = [&](std::uint32_t percent) {
        const std::size_t rank =
            (imbalances.size() * percent + 99u) / 100u;
        return imbalances[std::min(imbalances.size() - 1u, rank - 1u)];
      };
      stats.imbalance_p50_ppm = percentile(50u);
      stats.imbalance_p95_ppm = percentile(95u);
      stats.imbalance_p99_ppm = percentile(99u);
      stats.imbalance_maximum_ppm = imbalances.back();
    }
  }

  void collect_direct_correctness_audit(
      const gpulsmopt2_detail::ResidentPublicationPlan &plan) {
    direct_correctness_audit_stats_ = {};
    auto &stats = direct_correctness_audit_stats_;
    stats.enabled = direct_correctness_audit_enabled_;
    if (!direct_correctness_audit_enabled_ || plan.status) return;
    gpulsmopt2_detail::DirectOutputAuditDevice device{};
    CUDA_CHECK(cudaMemcpy(
        &device, direct_output_audit_device_.data(), sizeof(device),
        cudaMemcpyDeviceToHost));
    stats.row_count = device.row_count;
    stats.digest_sum = device.digest_sum;
    stats.digest_xor = device.digest_xor;
    stats.unsupported_jobs = device.unsupported_jobs;
    stats.count_mismatch_jobs = device.count_mismatch_jobs;
    stats.unsorted_rows = device.unsorted_rows;
    stats.duplicate_rows = device.duplicate_rows;
    stats.value_errors = device.value_errors;
    stats.flag_errors = device.flag_errors;
    stats.range_errors = device.range_errors;
    std::copy(std::begin(device.cell_errors), std::end(device.cell_errors),
              stats.cell_errors.begin());
    std::copy(std::begin(device.piece_errors), std::end(device.piece_errors),
              stats.piece_errors.begin());
    if (direct_profile_stats_.jobs.empty()) return;

    std::vector<gpulsmopt2_detail::Descriptor> descriptors(
        gpulsmopt2_detail::kQuotients);
    std::vector<gpulsmopt2_detail::RouteHeader> headers(
        gpulsmopt2_detail::kQuotients);
    std::vector<std::uint32_t> section_offsets(
        gpulsmopt2_detail::kQuotients + 1u);
    CUDA_CHECK(cudaMemcpy2D(
        descriptors.data(), sizeof(descriptors[0]),
        descriptors_.data() + plan.destination_level,
        gpulsmopt2_detail::kMaximumLevels * sizeof(descriptors[0]),
        sizeof(descriptors[0]), descriptors.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(
        headers.data(), sizeof(headers[0]),
        route_headers_.data() + plan.destination_level,
        gpulsmopt2_detail::kMaximumLevels * sizeof(headers[0]),
        sizeof(headers[0]), headers.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        section_offsets.data(), level_q_logical_offsets_.data() +
            std::size_t{plan.destination_level} *
                (gpulsmopt2_detail::kQuotients + 1u),
        section_offsets.size() * sizeof(section_offsets[0]),
        cudaMemcpyDeviceToHost));
    const std::size_t route_base =
        std::size_t{plan.destination_level} * route_stride_;
    std::vector<gpulsmopt2_detail::RouteSlice> slices(route_stride_);
    std::vector<std::uint32_t> logical_begins(route_stride_);
    CUDA_CHECK(cudaMemcpy(
        slices.data(), route_slices_.data() + route_base,
        slices.size() * sizeof(slices[0]), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        logical_begins.data(), route_logical_begins_.data() + route_base,
        logical_begins.size() * sizeof(logical_begins[0]),
        cudaMemcpyDeviceToHost));

    std::size_t begin = 0u;
    while (begin < direct_profile_stats_.jobs.size()) {
      const std::uint32_t q =
          direct_profile_stats_.jobs[begin].quotient_begin;
      std::size_t end = begin;
      std::uint32_t total = 0u;
      while (end < direct_profile_stats_.jobs.size() &&
             direct_profile_stats_.jobs[end].quotient_begin == q) {
        total += direct_profile_stats_.jobs[end].output_rows;
        ++end;
      }
      const auto header = headers[q];
      if (header.count != end - begin || header.begin < route_base ||
          header.begin + header.count > route_base + route_stride_)
        ++stats.route_header_errors;
      const auto descriptor = descriptors[q];
      if (descriptor.count() != total ||
          descriptor.split() != ((end - begin) > 1u))
        ++stats.descriptor_errors;
      std::uint32_t prefix = 0u;
      for (std::size_t index = begin; index < end; ++index) {
        const auto &job = direct_profile_stats_.jobs[index];
        const std::size_t route = header.begin + job.route_ordinal;
        if (route < route_base || route >= route_base + route_stride_) {
          ++stats.route_slice_errors;
          continue;
        }
        const std::size_t local = route - route_base;
        const auto slice = slices[local];
        const std::uint32_t suffix_begin = static_cast<std::uint32_t>(
            job.key_begin - (std::uint64_t{q} << 16u));
        const std::uint32_t suffix_end = static_cast<std::uint32_t>(
            job.key_end - (std::uint64_t{q} << 16u));
        if (slice.rows.offset() != job.existing_offset ||
            slice.rows.count() != job.output_rows ||
            slice.suffix_begin != suffix_begin ||
            slice.suffix_end != suffix_end)
          ++stats.route_slice_errors;
        if (logical_begins[local] != section_offsets[q] + prefix)
          ++stats.route_logical_errors;
        prefix += job.output_rows;
      }
      begin = end;
    }
  }

  void launch_resident_merge_standalone(cudaStream_t stream) {
    const std::uint64_t bank_capacity = foundation_pool_capacity_ / 2u;
    const std::uint32_t job_capacity = resident_merge_capacity_;
    gpulsmopt2_detail::choose_resident_publication_path_kernel<<<
        1, 1, 0, stream>>>(
            publication_selected_count_.data(), device_manifests_.data(),
            active_device_manifest_.data(), level_storage_spans_.data(),
            bank_capacity, job_capacity, resident_plan_.data());
    launch_resident_merge_pre(stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::uint32_t crowded = 0u;
    std::uint32_t local_overflow = 0u;
    CUDA_CHECK(cudaMemcpy(
        &crowded, foundation_overflow_flag_.data(), sizeof(crowded),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &local_overflow, local_epoch_overflow_flag_.data(),
        sizeof(local_overflow), cudaMemcpyDeviceToHost));
    if (crowded) {
      if (local_overflow)
        launch_crowded_epoch_stage(stream);
      else
        launch_local_epoch_stage(stream);
    } else {
      launch_normal_epoch_stage(stream);
    }
    launch_resident_merge_finish(stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    gpulsmopt2_detail::ResidentPublicationPlan plan{};
    std::uint32_t rank_cell_mode = 0u;
    std::uint64_t occupied_level_mask = 0u;
    CUDA_CHECK(cudaMemcpy(
        &plan, resident_plan_.data(), sizeof(plan), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &rank_cell_mode, rank_cell_mode_.data(), sizeof(rank_cell_mode),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &occupied_level_mask, query_occupied_level_mask_.data(),
        sizeof(occupied_level_mask), cudaMemcpyDeviceToHost));

    rank_cell_profile_stats_ = {};
    rank_cell_profile_stats_.standalone = rank_cell_standalone_profile_;
    rank_cell_profile_stats_.rank_cell_mode = rank_cell_mode;
    rank_cell_profile_stats_.launch_count = 1u;
    rank_cell_profile_stats_.grid_x = resident_rank_cell_blocks_;
    rank_cell_profile_stats_.block_x = gpulsmopt2_detail::kRankCellThreads;
    rank_cell_profile_stats_.dynamic_shared_bytes =
        gpulsmopt2_detail::kRankCellDynamicSharedBytes;
    rank_cell_profile_stats_.selected_count = plan.selected_count;
    rank_cell_profile_stats_.destination_level = plan.destination_level;
    rank_cell_profile_stats_.publication_status = plan.status;
    rank_cell_profile_stats_.raw_reservation = plan.raw_reservation;
    rank_cell_profile_stats_.survivor_count = plan.survivor_count;
    rank_cell_profile_stats_.occupied_level_mask = occupied_level_mask;

    direct_profile_stats_ = {};
    direct_profile_stats_.standalone = direct_standalone_profile_;
    direct_profile_stats_.forced_unified_merge =
        gpulsmopt2_detail::kForceUnifiedMergeExperiment;
    direct_profile_stats_.forced_unified_compile_elision =
        gpulsmopt2_detail::kForceUnifiedCompileElision;
    direct_profile_stats_.direct_launch_count = 1u;
    direct_profile_stats_.rank_cell_mode = rank_cell_mode;
    direct_profile_stats_.grid_x = resident_merge_blocks_;
    direct_profile_stats_.block_x = gpulsmopt2_detail::kFoundationCompactionThreads;
    direct_profile_stats_.dynamic_shared_bytes = resident_merge_workspace_bytes_;
    direct_profile_stats_.selected_count = plan.selected_count;
    direct_profile_stats_.source_count = plan.source_count;
    direct_profile_stats_.destination_level = plan.destination_level;
    direct_profile_stats_.publication_status = plan.status;
    direct_profile_stats_.job_count = plan.job_count;
    direct_profile_stats_.raw_reservation = plan.raw_reservation;
    direct_profile_stats_.survivor_count = plan.survivor_count;
    direct_profile_stats_.occupied_level_mask = occupied_level_mask;
    if ((direct_standalone_profile_ || direct_correctness_audit_enabled_) &&
        plan.job_count) {
      std::vector<gpulsmopt2_detail::BalancedMergeJob> jobs(plan.job_count);
      std::vector<std::uint64_t> raw(plan.job_count);
      CUDA_CHECK(cudaMemcpy(
          jobs.data(), balanced_merge_jobs_.data(),
          jobs.size() * sizeof(jobs[0]), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(
          raw.data(), resident_job_raw_reservations_.data(),
          raw.size() * sizeof(raw[0]), cudaMemcpyDeviceToHost));
      direct_profile_stats_.jobs.reserve(plan.job_count);
      for (std::uint32_t i = 0u; i < plan.job_count; ++i) {
        const auto &job = jobs[i];
        DirectMergeJobProfile item;
        item.job_index = i;
        item.key_begin = job.key_begin;
        item.key_end = job.key_end;
        item.existing_offset = job.existing_offset;
        item.quotient_begin = job.quotient_begin;
        item.quotient_end = job.quotient_end;
        item.quotient_count = job.quotient_end - job.quotient_begin;
        item.raw_rows = i < direct_profile_raw_rows_.size()
            ? direct_profile_raw_rows_[i] : raw[i];
        item.output_rows = job.output_count;
        item.existing_capacity = job.existing_capacity;
        item.route_ordinal = job.route_ordinal;
        item.hot_pieces = job.hot_pieces;
        item.hot_piece = job.hot_piece;
        item.crowded = job.hot_pieces != 0u;
        item.cell_owned_shape = !item.crowded &&
            item.quotient_count <= gpulsmopt2_detail::kCellOwnedQuotients;
        direct_profile_stats_.jobs.push_back(item);
      }
    }
    collect_partition_audit(plan, rank_cell_mode);
    collect_direct_correctness_audit(plan);
  }

  cudaGraph_t capture_resident_merge_graph(cudaStream_t capture_stream) {
    cudaGraph_t graph{};
    CUDA_CHECK(cudaGraphCreate(&graph, 0u));
    cudaGraphConditionalHandle crowded_conditional{};
    CUDA_CHECK(cudaGraphConditionalHandleCreate(
        &crowded_conditional, graph, 0u, cudaGraphCondAssignDefault));

    cudaGraph_t pre_graph{}, local_graph{}, crowded_graph{}, normal_graph{},
        finish_graph{};
    try {
      pre_graph = capture_resident_merge_pre_graph(capture_stream);
      local_graph = capture_local_epoch_stage_graph(capture_stream);
      crowded_graph = capture_crowded_epoch_stage_graph(capture_stream);
      CUDA_CHECK(cudaStreamBeginCapture(
          capture_stream, cudaStreamCaptureModeThreadLocal));
      launch_normal_epoch_stage(capture_stream);
      CUDA_CHECK(cudaStreamEndCapture(capture_stream, &normal_graph));
      finish_graph = capture_resident_merge_finish_graph(capture_stream);

      auto *selected_count = publication_selected_count_.data();
      auto *manifests = device_manifests_.data();
      auto *active_manifest = active_device_manifest_.data();
      auto *level_spans = level_storage_spans_.data();
      const std::uint64_t bank_capacity = foundation_pool_capacity_ / 2u;
      const std::uint32_t job_capacity = resident_merge_capacity_;
      auto *plan = resident_plan_.data();
      void *plan_arguments[] = {
          &selected_count, &manifests, &active_manifest, &level_spans,
          const_cast<std::uint64_t *>(&bank_capacity),
          const_cast<std::uint32_t *>(&job_capacity), &plan};
      cudaKernelNodeParams plan_params{};
      plan_params.func = reinterpret_cast<void *>(
          gpulsmopt2_detail::choose_resident_publication_path_kernel);
      plan_params.gridDim = dim3(1u);
      plan_params.blockDim = dim3(1u);
      plan_params.kernelParams = plan_arguments;
      cudaGraphNode_t plan_node{};
      CUDA_CHECK(cudaGraphAddKernelNode(
          &plan_node, graph, nullptr, 0u, &plan_params));

      cudaGraphNode_t pre_node{};
      CUDA_CHECK(cudaGraphAddChildGraphNode(
          &pre_node, graph, &plan_node, 1u, pre_graph));

      auto *crowded_flag = foundation_overflow_flag_.data();
      void *controller_arguments[] = {
          &crowded_conditional, &crowded_flag};
      cudaKernelNodeParams controller_params{};
      controller_params.func = reinterpret_cast<void *>(
          gpulsmopt2_detail::choose_crowded_epoch_path_kernel);
      controller_params.gridDim = dim3(1u);
      controller_params.blockDim = dim3(1u);
      controller_params.kernelParams = controller_arguments;
      cudaGraphNode_t controller{};
      CUDA_CHECK(cudaGraphAddKernelNode(
          &controller, graph, &pre_node, 1u, &controller_params));

      cudaGraphNodeParams conditional_params{};
      conditional_params.type = cudaGraphNodeTypeConditional;
      conditional_params.conditional.handle = crowded_conditional;
      conditional_params.conditional.type = cudaGraphCondTypeIf;
      conditional_params.conditional.size = 2u;
      cudaGraphNode_t conditional{};
      CUDA_CHECK(cudaGraphAddNode(
          &conditional, graph, &controller, 1u, &conditional_params));
      cudaGraph_t *bodies = conditional_params.conditional.phGraph_out;
      if (!bodies)
        throw std::runtime_error(
            "CUDA did not create crowded publication bodies");

      cudaGraphConditionalHandle local_overflow_conditional{};
      CUDA_CHECK(cudaGraphConditionalHandleCreate(
          &local_overflow_conditional, bodies[0], 0u,
          cudaGraphCondAssignDefault));
      auto *local_overflow_flag = local_epoch_overflow_flag_.data();
      void *local_controller_arguments[] = {
          &local_overflow_conditional, &local_overflow_flag};
      cudaKernelNodeParams local_controller_params{};
      local_controller_params.func = reinterpret_cast<void *>(
          gpulsmopt2_detail::choose_local_epoch_path_kernel);
      local_controller_params.gridDim = dim3(1u);
      local_controller_params.blockDim = dim3(1u);
      local_controller_params.kernelParams = local_controller_arguments;
      cudaGraphNode_t local_controller{};
      CUDA_CHECK(cudaGraphAddKernelNode(
          &local_controller, bodies[0], nullptr, 0u,
          &local_controller_params));
      cudaGraphNodeParams local_conditional_params{};
      local_conditional_params.type = cudaGraphNodeTypeConditional;
      local_conditional_params.conditional.handle =
          local_overflow_conditional;
      local_conditional_params.conditional.type = cudaGraphCondTypeIf;
      local_conditional_params.conditional.size = 2u;
      cudaGraphNode_t local_conditional{};
      CUDA_CHECK(cudaGraphAddNode(
          &local_conditional, bodies[0], &local_controller, 1u,
          &local_conditional_params));
      cudaGraph_t *local_bodies =
          local_conditional_params.conditional.phGraph_out;
      if (!local_bodies)
        throw std::runtime_error(
            "CUDA did not create local publication bodies");
      cudaGraphNode_t child{};
      CUDA_CHECK(cudaGraphAddChildGraphNode(
          &child, local_bodies[0], nullptr, 0u, crowded_graph));
      CUDA_CHECK(cudaGraphAddChildGraphNode(
          &child, local_bodies[1], nullptr, 0u, local_graph));
      CUDA_CHECK(cudaGraphAddChildGraphNode(
          &child, bodies[1], nullptr, 0u, normal_graph));

      cudaGraphNode_t finish_node{};
      CUDA_CHECK(cudaGraphAddChildGraphNode(
          &finish_node, graph, &conditional, 1u, finish_graph));
    } catch (...) {
      if (pre_graph) cudaGraphDestroy(pre_graph);
      if (local_graph) cudaGraphDestroy(local_graph);
      if (crowded_graph) cudaGraphDestroy(crowded_graph);
      if (normal_graph) cudaGraphDestroy(normal_graph);
      if (finish_graph) cudaGraphDestroy(finish_graph);
      cudaGraphDestroy(graph);
      throw;
    }
    CUDA_CHECK(cudaGraphDestroy(pre_graph));
    CUDA_CHECK(cudaGraphDestroy(local_graph));
    CUDA_CHECK(cudaGraphDestroy(crowded_graph));
    CUDA_CHECK(cudaGraphDestroy(normal_graph));
    CUDA_CHECK(cudaGraphDestroy(finish_graph));
    return graph;
  }

  void initialize_resident_publication_graph() {
    cudaStream_t capture_stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &capture_stream, cudaStreamNonBlocking));
    try {
      resident_publication_graph_ =
          capture_resident_merge_graph(capture_stream);
      CUDA_CHECK(cudaGraphInstantiate(
          &resident_publication_graph_exec_,
          resident_publication_graph_, 0ull));
    } catch (...) {
      if (resident_publication_graph_)
        cudaGraphDestroy(resident_publication_graph_);
      resident_publication_graph_ = nullptr;
      cudaStreamDestroy(capture_stream);
      throw;
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
    initialize_publication_workspace();
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
    if (destination >= gpulsmopt2_detail::kMaximumLevels ||
        (gpulsmopt2_detail::kCanonicalCarry &&
         destination >= canonical_level_count_)) {
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
    ++stats_.epochs_published;
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
        ", routes=" + std::to_string(receipt.route_count) +
        ", ranks=" + std::to_string(receipt.rank_block_count) +
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
    stats_ = {};
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
    ++stats_.admitted_batches;
    stats_.admitted_records += consumed;
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
    gpulsmopt2_detail::scatter_admission_records_kernel<<<
        blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
            keys, values, n, slot, tombstone,
            batch_offsets, radix_ids_out_.data(),
            destination_keys, destination_payloads);
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
    if (canonical_local_epoch_enabled_) {
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
          std::size_t{canonical_epoch_workspace_slots_} *
              sizeof(std::uint32_t),
          stream));
      gpulsmopt2_detail::count_canonical_epoch_sections_kernel<<<
          blocks(gpulsmopt2_detail::kQuotients + 1u),
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              raw_offsets_.data(), gpulsmopt2_detail::kBatchesPerEpoch,
              foundation_section_output_counts_.data(),
              reinterpret_cast<std::uint32_t *>(
                  balanced_merge_raw_counts_.data()),
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
      if (materialize_resident) {
        gpulsmopt2_detail::resolve_canonical_epoch_local_kernel<true><<<
            gpulsmopt2_detail::kQuotients,
            gpulsmopt2_detail::kFoundationCompactionThreads, 0, stream>>>(
                raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
                static_cast<std::uint32_t>(batch_capacity_),
                gpulsmopt2_detail::kBatchesPerEpoch,
                foundation_source_offsets_.data(),
                publication_rows_a_.data(), resident_rows(),
                level_zero_begin(), foundation_section_output_counts_.data(),
                epoch_ranks);
        gpulsmopt2_detail::resolve_canonical_epoch_oversized_kernel<true><<<
            canonical_epoch_resolver_blocks_,
            gpulsmopt2_detail::kFoundationCompactionThreads, 0, stream>>>(
                reinterpret_cast<std::uint32_t *>(
                    balanced_merge_raw_counts_.data()),
                local_epoch_overflow_flag_.data(),
                canonical_next_job_.data(), raw_keys_.data(),
                raw_payloads_.data(), raw_offsets_.data(),
                static_cast<std::uint32_t>(batch_capacity_),
                gpulsmopt2_detail::kBatchesPerEpoch,
                foundation_source_offsets_.data(),
                publication_rows_a_.data(), resident_rows(),
                level_zero_begin(), foundation_section_output_counts_.data(),
                epoch_ranks,
                reinterpret_cast<unsigned long long *>(
                    publication_epoch_payloads_b_.data()),
                canonical_cell_counts_.data(),
                canonical_epoch_workspace_slots_);
      } else {
        gpulsmopt2_detail::resolve_canonical_epoch_local_kernel<false><<<
            gpulsmopt2_detail::kQuotients,
            gpulsmopt2_detail::kFoundationCompactionThreads, 0, stream>>>(
                raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
                static_cast<std::uint32_t>(batch_capacity_),
                gpulsmopt2_detail::kBatchesPerEpoch,
                foundation_source_offsets_.data(),
                publication_rows_a_.data(), resident_rows(),
                level_zero_begin(), foundation_section_output_counts_.data(),
                epoch_ranks);
        gpulsmopt2_detail::resolve_canonical_epoch_oversized_kernel<false><<<
            canonical_epoch_resolver_blocks_,
            gpulsmopt2_detail::kFoundationCompactionThreads, 0, stream>>>(
                reinterpret_cast<std::uint32_t *>(
                    balanced_merge_raw_counts_.data()),
                local_epoch_overflow_flag_.data(),
                canonical_next_job_.data(), raw_keys_.data(),
                raw_payloads_.data(), raw_offsets_.data(),
                static_cast<std::uint32_t>(batch_capacity_),
                gpulsmopt2_detail::kBatchesPerEpoch,
                foundation_source_offsets_.data(),
                publication_rows_a_.data(), resident_rows(),
                level_zero_begin(), foundation_section_output_counts_.data(),
                epoch_ranks,
                reinterpret_cast<unsigned long long *>(
                    publication_epoch_payloads_b_.data()),
                canonical_cell_counts_.data(),
                canonical_epoch_workspace_slots_);
      }
      gpulsmopt2_detail::sum_canonical_section_counts_kernel<<<
          1, gpulsmopt2_detail::kThreads, 0, stream>>>(
              foundation_section_output_counts_.data(),
              publication_selected_count_.data());
      return;
    }
    gpulsmopt2_detail::build_publication_batch_offsets_kernel<<<
        1, 1, 0, stream>>>(
            raw_offsets_.data(), publication_batch_offsets_.data());
    const dim3 publication_grid(
        blocks(batch_capacity_), gpulsmopt2_detail::kBatchesPerEpoch);
    gpulsmopt2_detail::pack_canonical_epoch_kernel<<<
        publication_grid, gpulsmopt2_detail::kThreads, 0, stream>>>(
            raw_keys_.data(), raw_payloads_.data(),
            static_cast<std::uint32_t>(batch_capacity_),
            publication_batch_offsets_.data(),
            publication_epoch_keys_a_.data(),
            publication_epoch_payloads_a_.data());
    const std::uint32_t epoch_capacity = static_cast<std::uint32_t>(
        batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch);
    gpulsmopt2_detail::pad_canonical_epoch_kernel<<<
        blocks(epoch_capacity), gpulsmopt2_detail::kThreads, 0, stream>>>(
            epoch_capacity, publication_batch_offsets_.data(),
            publication_epoch_keys_a_.data(),
            publication_epoch_payloads_a_.data());
    std::size_t workspace_bytes = publication_temp_.size();
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        publication_temp_.data(), workspace_bytes,
        publication_epoch_keys_a_.data(), publication_epoch_keys_b_.data(),
        publication_epoch_payloads_a_.data(),
        publication_epoch_payloads_b_.data(), epoch_capacity, 0, 32,
        stream));
    workspace_bytes = publication_temp_.size();
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        publication_temp_.data(), workspace_bytes,
        publication_epoch_keys_b_.data(), publication_keys_a_.data(),
        publication_epoch_payloads_b_.data(),
        publication_epoch_payloads_a_.data(),
        publication_selected_count_.data(),
        gpulsmopt2_detail::NewestPayload{}, epoch_capacity, stream));
    if (materialize_resident) {
      gpulsmopt2_detail::materialize_canonical_epoch_resident_kernel<<<
          blocks(epoch_capacity), gpulsmopt2_detail::kThreads, 0, stream>>>(
              publication_keys_a_.data(),
              publication_epoch_payloads_a_.data(),
              publication_selected_count_.data(), resident_rows(),
              level_zero_begin());
    } else {
      gpulsmopt2_detail::materialize_canonical_epoch_rows_kernel<<<
          blocks(epoch_capacity), gpulsmopt2_detail::kThreads, 0, stream>>>(
              publication_keys_a_.data(),
              publication_epoch_payloads_a_.data(),
              publication_selected_count_.data(),
              publication_rows_a_.data());
    }
    gpulsmopt2_detail::build_query_quotient_offsets_device_count_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            publication_keys_a_.data(), publication_selected_count_.data(),
            foundation_source_offsets_.data());
    gpulsmopt2_detail::canonical_section_counts_from_offsets_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            foundation_source_offsets_.data(),
            foundation_section_output_counts_.data());
    if (!materialize_resident)
      gpulsmopt2_detail::build_staged_rank_directory_kernel<<<
          gpulsmopt2_detail::kQuotients,
          gpulsmopt2_detail::kFoundationCells, 0, stream>>>(
              publication_rows_a_.data(),
              foundation_source_offsets_.data(),
              rank_output
                  ? rank_output
                  : canonical_cell_ranks_.data() +
                        std::size_t{destination_level} *
                            gpulsmopt2_detail::kLocalRankEntries);
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
    gpulsmopt2_detail::choose_canonical_publication_path_kernel<<<
        1, 1, 0, stream>>>(
            publication_selected_count_.data(), device_manifests_.data(),
            active_device_manifest_.data(), level_storage_spans_.data(),
            canonical_level_count_, job_capacity, top_level_rollover,
            resident_plan_.data());

    if (direct_epoch) {
      if (!canonical_local_epoch_enabled_)
        gpulsmopt2_detail::build_canonical_rank_from_run_kernel<<<
            gpulsmopt2_detail::kQuotients,
            gpulsmopt2_detail::kFoundationCells, 0, stream>>>(
                resident_rows(), level_zero_begin(),
                foundation_source_offsets_.data(), destination,
                canonical_cell_ranks_.data());
      if (canonical_local_epoch_enabled_)
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
      else
        gpulsmopt2_detail::finalize_canonical_level_metadata_kernel<<<
            blocks(gpulsmopt2_detail::kQuotients + 1u),
            gpulsmopt2_detail::kThreads, 0, stream>>>(
                foundation_source_offsets_.data(),
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
      } else if (source_count == 2u) {
        gpulsmopt2_detail::canonical_carry_jobs_kernel<2u><<<
            canonical_two_way_blocks_,
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
      } else {
        gpulsmopt2_detail::canonical_carry_jobs_kernel<0u><<<
            canonical_merge_blocks_,
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
    const std::uint64_t empty = ~host_occupied_level_mask_;
    destination = empty
        ? static_cast<std::uint32_t>(__builtin_ctzll(empty))
        : gpulsmopt2_detail::kMaximumLevels;
    const std::uint64_t carried = destination >=
            gpulsmopt2_detail::kMaximumLevels
        ? ~std::uint64_t{0}
        : destination ? (std::uint64_t{1} << destination) - 1u : 0u;
    source_count = 1u + static_cast<std::uint32_t>(
        __builtin_popcountll(host_occupied_level_mask_ & carried));
    direct_epoch = destination == 0u &&
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
    }
    cudaGraphExec_t graph_exec = direct_epoch
        ? canonical_direct_publication_graph_exec_
        : !top_level_rollover &&
              source_count < canonical_publication_graph_execs_.size()
            ? canonical_publication_graph_execs_[source_count] : nullptr;
    if (gpulsmopt2_detail::kCanonicalPublicationGraph && graph_exec) {
      CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
      return true;
    }
    launch_canonical_publication_commands(
        stream, destination, source_count, direct_epoch, false,
        top_level_rollover);
    return false;
  }

  std::uint64_t level_zero_begin() const {
    return foundation_pool_capacity_;
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
    bool receipt_in_graph = false;
    if (gpulsmopt2_detail::kCanonicalCarry) {
      receipt_in_graph = launch_canonical_publication(stream);
    } else {
      gpulsmopt2_detail::count_direct_epoch_records_kernel<<<1, 1, 0, stream>>>(
          raw_offsets_.data(), gpulsmopt2_detail::kBatchesPerEpoch,
          publication_selected_count_.data());
      if (rank_cell_standalone_profile_ || direct_standalone_profile_ ||
          direct_correctness_audit_enabled_ || partition_audit_enabled_)
        launch_resident_merge_standalone(stream);
      else
        CUDA_CHECK(cudaGraphLaunch(resident_publication_graph_exec_, stream));
    }
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
    const std::uint16_t *foundation_ranks = local_rank_.data();
    if (gpulsmopt2_detail::kCanonicalCarry && active_levels_)
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
    std::uint8_t *storage = nullptr;
    if (range_query_storage_.size() >= bytes) {
      storage = range_query_storage_.data();
    } else {
      // Borrow idle publication scratch for ranges.
      const std::size_t borrowed_bytes =
          publication_epoch_keys_a_.size() * sizeof(std::uint32_t);
      if (borrowed_bytes >= bytes) {
        storage = reinterpret_cast<std::uint8_t *>(
            publication_epoch_keys_a_.data());
      } else {
        range_query_storage_.resize(bytes);
        storage = range_query_storage_.data();
      }
    }
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
    std::uint8_t *storage = nullptr;
    const std::size_t borrowed_bytes =
        publication_epoch_assignments_b_.size() *
        sizeof(gpulsmopt2_detail::RawAssignment);
    if (borrowed_bytes >= bytes) {
      storage = reinterpret_cast<std::uint8_t *>(
          publication_epoch_assignments_b_.data());
    } else {
      range_fragment_storage_.resize(bytes);
      storage = range_fragment_storage_.data();
    }
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
    std::uint8_t *storage = nullptr;
    const std::size_t borrowed_bytes =
        publication_epoch_assignments_a_.size() *
        sizeof(gpulsmopt2_detail::RawAssignment);
    if (borrowed_bytes >= bytes) {
      storage = reinterpret_cast<std::uint8_t *>(
          publication_epoch_assignments_a_.data());
    } else {
      range_section_storage_.resize(bytes);
      storage = range_section_storage_.data();
    }
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
  std::size_t foundation_pool_capacity_{};
  std::size_t level_pool_capacity_{};
  std::size_t level_rank_block_capacity_{};
  std::uint32_t canonical_level_count_{};
  std::uint32_t resident_merge_capacity_{};
  std::size_t resident_merge_workspace_bytes_{};
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
  MaintenanceStats stats_{};
  cudaEvent_t operation_done_{};
  cudaGraph_t resident_publication_graph_{};
  cudaGraphExec_t resident_publication_graph_exec_{};
  cudaGraphExec_t canonical_direct_publication_graph_exec_{};
  std::array<cudaGraphExec_t,
             gpulsmopt2_detail::kMaximumMergeSources + 1u>
      canonical_publication_graph_execs_{};
  std::uint32_t resident_merge_blocks_{};
  std::uint32_t canonical_merge_blocks_{};
  std::uint32_t canonical_two_way_blocks_{};
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
  bool canonical_local_epoch_enabled_{};
  std::uint32_t resident_rank_cell_blocks_{};
  std::uint32_t resident_planner_blocks_{};
  std::uint32_t range_section_blocks_{};
  bool rank_cell_standalone_profile_ = false;
  bool direct_standalone_profile_ = false;
  bool direct_correctness_audit_enabled_ = false;
  bool partition_audit_enabled_ = false;
  RankCellProfileStats rank_cell_profile_stats_{};
  DirectMergeProfileStats direct_profile_stats_{};
  DirectCorrectnessAuditStats direct_correctness_audit_stats_{};
  PartitionAuditStats partition_audit_stats_{};
  std::vector<std::uint64_t> direct_profile_raw_rows_;

  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::DirectOutputAuditDevice>
      direct_output_audit_device_;
  gpulsmopt2_detail::Buffer<std::uint64_t> direct_audit_raw_rows_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::PartitionAuditDeviceJob>
      partition_audit_device_jobs_;
  gpulsmopt2_detail::Buffer<std::uint16_t> partition_audit_source_rows_;

  gpulsmopt2_detail::Buffer<std::uint16_t> local_rank_;
  gpulsmopt2_detail::VirtualBuffer<std::uint16_t> level_guides_;
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
  gpulsmopt2_detail::Buffer<std::uint32_t> staged_epoch_mode_;
  gpulsmopt2_detail::Buffer<std::uint32_t> rank_cell_mode_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::ResidentPublicationPlan>
      resident_plan_;
  gpulsmopt2_detail::PinnedBuffer<
      gpulsmopt2_detail::ResidentPublicationPlan> publication_receipt_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::LevelStorageSpan>
      level_storage_spans_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::LevelRankSpan>
      level_rank_spans_;
  gpulsmopt2_detail::Buffer<std::uint32_t> level_cell_rank_blocks_;
  gpulsmopt2_detail::Buffer<std::uint16_t> level_cell_ranks_;
  gpulsmopt2_detail::Buffer<std::uint16_t> publication_cell_ranks_;
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
  gpulsmopt2_detail::Buffer<std::uint32_t> publication_epoch_keys_a_,
      publication_epoch_keys_b_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RawAssignment>
      publication_epoch_assignments_a_, publication_epoch_assignments_b_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RawPayload>
      publication_epoch_payloads_a_, publication_epoch_payloads_b_;
  gpulsmopt2_detail::VirtualBuffer<std::uint32_t> publication_keys_a_;
  gpulsmopt2_detail::Buffer<std::uint32_t> publication_selected_count_,
      publication_batch_offsets_;
  gpulsmopt2_detail::Buffer<std::uint32_t> foundation_source_offsets_,
      foundation_section_output_counts_;
  gpulsmopt2_detail::Buffer<std::uint64_t> balanced_merge_raw_counts_,
      resident_job_raw_reservations_, resident_job_output_offsets_;
  gpulsmopt2_detail::Buffer<std::uint32_t>
      resident_tile_job_counts_, resident_tile_job_offsets_,
      resident_route_counts_, resident_route_offsets_,
      resident_section_logical_offsets_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::BalancedMergeJob>
      balanced_merge_jobs_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RouteHeader>
      foundation_next_route_headers_;
  gpulsmopt2_detail::Buffer<std::uint8_t> resident_scan_temp_;
  gpulsmopt2_detail::Buffer<std::uint32_t> foundation_overflow_flag_,
      local_epoch_overflow_flag_;
  gpulsmopt2_detail::VirtualBuffer<gpulsmopt2_detail::Row>
      publication_rows_a_;
  gpulsmopt2_detail::Buffer<std::uint8_t> publication_temp_;
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
