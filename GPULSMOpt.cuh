#pragma once
#include "gpu_dictionary_adapter.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <thrust/iterator/transform_output_iterator.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
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
constexpr std::uint32_t kPlanningTiles = 128u;
constexpr std::uint32_t kPlanningTileQuotients =
    kQuotients / kPlanningTiles;
constexpr std::uint32_t kMaximumMergeSources = kMaximumLevels + 1u;
constexpr std::uint32_t kBalancedMergeCapacityCeiling =
    kFoundationCompactionThreads * 32u;
// Bound for splitting a maximally dense quotient.
constexpr std::uint32_t kMergeSourceBits = 7u;
constexpr std::uint64_t kResidentWorkFlag = std::uint64_t{1} << 63u;

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

__device__ __forceinline__ Row logical_section_row(
    std::uint32_t q, std::uint32_t level, std::uint32_t position,
    ResidentRows arena, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets) {
  const RouteHeader header = route_headers[descriptor_index(q, level)];
  if (header.count == 1u)
    return arena[route_slices[header.begin].rows.offset() + position];
  const std::uint32_t section_begin = level_q_logical_offsets[
      std::size_t{level} * (kQuotients + 1u) + q];
  const std::uint32_t logical = section_begin + position;
  for (std::uint32_t local = 0u; local < header.count; ++local) {
    const std::uint32_t route = header.begin + local;
    const RouteSlice slice = route_slices[route];
    const std::uint32_t begin = route_logical_begins[route];
    if (logical >= begin && logical < begin + slice.rows.count())
      return arena[slice.rows.offset() + logical - begin];
  }
  return {};
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

__device__ __noinline__ unsigned long long warp_sum_visible_by_verification(
    std::uint32_t q, std::uint32_t low, std::uint32_t high,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets,
    std::uint32_t batch_stride, std::uint32_t pending_batches,
    ResidentRows arena, const Descriptor *descriptors,
    const RouteHeader *route_headers, const RouteSlice *route_slices,
    std::uint32_t active_levels, std::uint64_t occupied_levels) {
  constexpr unsigned mask = 0xffffffffu;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t low_suffix = key_suffix(low);
  const std::uint32_t high_suffix = key_suffix(high);
  unsigned long long local = 0ull;

  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t oi = std::size_t{batch} * (kQuotients + 1u) + q;
    const std::uint32_t section_begin = raw_offsets[oi];
    const std::uint32_t section_end = raw_offsets[oi + 1u];
    for (std::uint32_t index = section_begin + lane; index < section_end;
         index += 32u) {
      const std::uint32_t record = batch * batch_stride + index;
      const RawAssignment item =
          load_raw_assignment(raw_keys, raw_payloads, record);
      const Row item_row = raw_row(item);
      if (item_row.key < low_suffix || item_row.key > high_suffix) continue;
      bool covered = false;
      for (std::uint32_t other_batch = 0u;
           other_batch < pending_batches && !covered; ++other_batch) {
        const std::size_t noi =
            std::size_t{other_batch} * (kQuotients + 1u) + q;
        const std::uint32_t nb = raw_offsets[noi], ne = raw_offsets[noi + 1u];
        for (std::uint32_t other = nb; other < ne; ++other) {
          const std::uint32_t other_record =
              other_batch * batch_stride + other;
          const RawAssignment candidate =
              load_raw_assignment(raw_keys, raw_payloads, other_record);
          if (key_suffix(candidate.key) == item_row.key &&
              raw_position(candidate) > raw_position(item)) {
            covered = true;
            break;
          }
        }
      }
      if (!covered && (item_row.flags & kTombstone) == 0u)
        local += item_row.value;
    }
  }

  for (std::uint32_t level = 0u; level < active_levels; ++level) {
    if (!level_is_occupied(occupied_levels, level)) continue;
    const RouteHeader header = route_headers[descriptor_index(q, level)];
    for (std::uint32_t route_index = 0u;
         route_index < header.count; ++route_index) {
      const RouteSlice route = route_slices[header.begin + route_index];
      if (route.suffix_end <= low_suffix ||
          route.suffix_begin > high_suffix) continue;
      const ResidentRows rows = arena + route.rows.offset();
      const std::uint32_t begin = lower_bound_rows(
          rows, route.rows.count(), low_suffix);
      const std::uint32_t end = high == kInvalid
          ? route.rows.count()
          : upper_bound_rows(rows, route.rows.count(), high_suffix);
      for (std::uint32_t index = begin + lane; index < end; index += 32u) {
        const Row item = rows[index];
      bool covered = false;
      for (std::uint32_t batch = 0u;
           batch < pending_batches && !covered; ++batch) {
        const std::size_t oi =
            std::size_t{batch} * (kQuotients + 1u) + q;
        const std::uint32_t rb = raw_offsets[oi], re = raw_offsets[oi + 1u];
        for (std::uint32_t position = rb; position < re; ++position)
          if (key_suffix(raw_keys[batch * batch_stride + position]) ==
              item.key) {
            covered = true;
            break;
          }
      }
      for (std::uint32_t newer = 0u; newer < level && !covered; ++newer) {
        if (!level_is_occupied(occupied_levels, newer)) continue;
        const RouteHeader newer_header =
            route_headers[descriptor_index(q, newer)];
        const Descriptor newer_descriptor = newer_header.count == 1u
            ? descriptors[descriptor_index(q, newer)]
            : routed_descriptor_for_suffix(
                  q, newer, item.key, route_headers, route_slices);
        const ResidentRows newer_rows = arena + newer_descriptor.offset();
        const std::uint32_t position =
            lower_bound_rows(newer_rows, newer_descriptor.count(), item.key);
        covered = position < newer_descriptor.count() &&
            newer_rows[position].key == item.key;
      }
      if (!covered && (item.flags & kTombstone) == 0u) local += item.value;
      }
    }
  }

  for (std::uint32_t offset = 16u; offset; offset >>= 1u)
    local += __shfl_down_sync(mask, local, offset);
  return __shfl_sync(mask, local, 0u);
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
    const std::uint32_t *raw_keys,
    const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches,
    typename Aggregate::State *aggregate_partials,
    const std::uint64_t *query_occupied_level_mask) {
  constexpr std::uint32_t kCapacity = 512u;
  using BlockScan =
      cub::BlockScan<std::uint32_t, kSectionRangeThreads>;
  union Workspace {
    Row merged[kCapacity];
    TaggedRow tagged[kCapacity];
    Row base_rows[kSectionRangeWarps][128u];
  };
  __shared__ Row current[kCapacity];
  __shared__ Workspace workspace;
  __shared__ typename BlockScan::TempStorage scan_storage;
  constexpr std::uint32_t kFragmentBaseMaskRows = 2048u;
  constexpr std::uint32_t kFragmentBaseMaskWords =
      kFragmentBaseMaskRows / 32u;
  constexpr std::uint32_t kSectionBaseMaskRows = 16384u;
  constexpr std::uint32_t kSectionBaseMaskWords =
      kSectionBaseMaskRows / 32u;
  union BaseMaskWorkspace {
    std::uint32_t section[kSectionBaseMaskWords];
    std::uint32_t fragments[kSectionRangeWarps][kFragmentBaseMaskWords];
    // Reuse row tiles across overlapping ranges.
    unsigned long long
        warp_tile_prefix[kSectionRangeWarps][128u + 1u];
  };
  __shared__ BaseMaskWorkspace base_mask_workspace;
  __shared__ std::uint32_t current_count_shared;
  __shared__ std::uint32_t pending_count_shared;
  __shared__ std::uint32_t overflow_shared;
  __shared__ std::uint32_t next_fragment_shared;
  __shared__ std::uint32_t dynamic_queue_shared;
  __shared__ std::uint32_t worker_width_shared;
  __shared__ std::uint32_t minimum_work_shared;
  __shared__ std::uint32_t maximum_work_shared;
  __shared__ std::uint32_t use_union_sweep_shared;
  __shared__ unsigned long long
      union_direct_warp_shared[kSectionRangeWarps];
  __shared__ unsigned long long
      union_sweep_warp_shared[kSectionRangeWarps];
  __shared__ std::uint32_t union_count_warp_shared[kSectionRangeWarps];
  __shared__ std::uint32_t fragment_work_shared[kSectionTaskFragments];
  __shared__ RangeFragmentBounds
      fragment_bounds_shared[kSectionTaskFragments];
  // Extend cell starts with the section endpoint.
  __shared__ std::uint32_t foundation_cell_ranks[kFoundationCells + 1u];
  __shared__ std::uint32_t foundation_ranks_valid_shared;
  __shared__ std::uint32_t section_base_mask_valid_shared;
  __shared__ std::uint32_t quotient_shared;
  __shared__ std::uint32_t fragment_begin_shared;
  __shared__ std::uint32_t fragment_end_shared;
  __shared__ std::uint32_t task_valid_shared;
  __shared__ std::uint32_t base_section_count_shared;
  __shared__ Descriptor foundation_descriptor_shared;
  __shared__ Descriptor section_descriptors[kMaximumLevels];

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
    do {
  const std::uint32_t q = quotient_shared;
  const std::uint32_t fragment_begin = fragment_begin_shared;
  const std::uint32_t fragment_end = fragment_end_shared;
  if (threadIdx.x < active_levels &&
      level_is_occupied(occupied_levels, threadIdx.x))
    section_descriptors[threadIdx.x] =
        descriptors[descriptor_index(q, threadIdx.x)];
  else if (threadIdx.x < active_levels)
    section_descriptors[threadIdx.x] = {};
  // Warp zero loads the rank block during setup.
  if (threadIdx.x < 32u) {
    bool ranked = false;
    Descriptor rank_descriptor{};
    if (local_rank && foundation_level < active_levels) {
      rank_descriptor =
          descriptors[descriptor_index(q, foundation_level)];
      const RouteHeader rank_header =
          route_headers[descriptor_index(q, foundation_level)];
      ranked = rank_header.count == 1u &&
          cell_rank_supported(rank_descriptor.count());
    }
    if (threadIdx.x == 0u) {
      foundation_ranks_valid_shared = ranked;
      foundation_cell_ranks[kFoundationCells] = rank_descriptor.count();
    }
    if (ranked) {
      for (std::uint32_t cell = threadIdx.x; cell < kFoundationCells;
           cell += 32u)
        foundation_cell_ranks[cell] =
            local_rank[std::size_t{q} * kFoundationCells + cell];
    }
  }
  if (threadIdx.x == 0u) {
    foundation_descriptor_shared = foundation_level < active_levels
        ? section_descriptors[foundation_level] : Descriptor{};
    base_section_count_shared = foundation_descriptor_shared.count();
    std::uint32_t physical = 0u;
    pending_count_shared = 0u;
    if (pending_batches) {
      for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
        const std::size_t oi =
            std::size_t{batch} * (kQuotients + 1u) + q;
        pending_count_shared += raw_offsets[oi + 1u] - raw_offsets[oi];
      }
      physical += pending_count_shared;
    }
    for (std::uint32_t level = 0u; level < active_levels; ++level)
      if (level != foundation_level &&
          level_is_occupied(occupied_levels, level))
        physical += section_descriptors[level].count();
    overflow_shared = physical > kCapacity;
    current_count_shared = 0u;
  }
  __syncthreads();

  if (!overflow_shared && pending_count_shared) {
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
        constexpr unsigned full_mask = 0xffffffffu;
        for (std::uint32_t width = 2u; width <= 32u; width <<= 1u) {
          for (std::uint32_t stride = width >> 1u; stride; stride >>= 1u) {
            TaggedRow other{};
            other.row.key = __shfl_xor_sync(full_mask, item.row.key, stride);
            other.row.value =
                __shfl_xor_sync(full_mask, item.row.value, stride);
            other.row.flags =
                __shfl_xor_sync(full_mask, item.row.flags, stride);
            other.age = __shfl_xor_sync(full_mask, item.age, stride);
            const bool ascending = (lane & width) == 0u;
            const bool take_min = ((lane & stride) == 0u) == ascending;
            if ((take_min && tagged_less(other, item)) ||
                (!take_min && tagged_less(item, other)))
              item = other;
          }
        }
        const std::uint32_t next_key =
            __shfl_down_sync(full_mask, item.row.key, 1u);
        const std::uint32_t next_age =
            __shfl_down_sync(full_mask, item.age, 1u);
        const bool winner = item.age != kInvalidAge &&
            (lane == 31u || next_age == kInvalid ||
             item.row.key != next_key);
        const unsigned winners = __ballot_sync(full_mask, winner);
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
              {raw_row(loaded),
               raw_age(raw_position(loaded), batch_stride)};
        } else {
          workspace.tagged[ordinal] = {{0u, 0u, 0u}, kInvalidAge};
        }
      }
      __syncthreads();
      for (std::uint32_t width = 2u; width <= sort_size; width <<= 1u) {
        for (std::uint32_t stride = width >> 1u; stride; stride >>= 1u) {
          for (std::uint32_t index = threadIdx.x; index < sort_size;
               index += blockDim.x) {
            const std::uint32_t other_index = index ^ stride;
            if (other_index > index) {
              const TaggedRow x = workspace.tagged[index];
              const TaggedRow y = workspace.tagged[other_index];
              const bool ascending = (index & width) == 0u;
              const bool swap = ascending ? tagged_less(y, x)
                                          : tagged_less(x, y);
              if (swap) {
                workspace.tagged[index] = y;
                workspace.tagged[other_index] = x;
              }
            }
          }
          __syncthreads();
        }
      }
      const std::uint32_t chunk_begin = threadIdx.x * 2u;
      std::uint32_t local_winners = 0u;
      bool winner[2]{};
#pragma unroll
      for (std::uint32_t item = 0u; item < 2u; ++item) {
        const std::uint32_t index = chunk_begin + item;
        winner[item] = index < sort_size &&
            workspace.tagged[index].age != kInvalidAge &&
            (index + 1u == sort_size ||
             workspace.tagged[index + 1u].age == kInvalidAge ||
             workspace.tagged[index].row.key !=
                 workspace.tagged[index + 1u].row.key);
        local_winners += winner[item];
      }
      std::uint32_t thread_base{}, winner_count{};
      BlockScan(scan_storage).ExclusiveSum(local_winners, thread_base,
                                            winner_count);
      std::uint32_t local_rank = 0u;
#pragma unroll
      for (std::uint32_t item = 0u; item < 2u; ++item)
        if (winner[item])
          current[thread_base + local_rank++] =
              workspace.tagged[chunk_begin + item].row;
      __syncthreads();
      if (threadIdx.x == 0u) current_count_shared = winner_count;
      __syncthreads();
    }
  }

  if (!overflow_shared) {
    for (std::uint32_t level = 0u; level < active_levels; ++level) {
      if (level == foundation_level ||
          !level_is_occupied(occupied_levels, level)) continue;
      const RouteHeader source_header =
          route_headers[descriptor_index(q, level)];
      for (std::uint32_t source_route = 0u;
           source_route < source_header.count; ++source_route) {
      const Descriptor descriptor =
          route_slices[source_header.begin + source_route].rows;
      const std::uint32_t source_count = descriptor.count();
      if (!source_count) continue;
      const ResidentRows source = arena + descriptor.offset();
      const std::uint32_t current_count = current_count_shared;
      if (!current_count) {
        for (std::uint32_t index = threadIdx.x; index < source_count;
             index += blockDim.x)
          current[index] = source[index];
        __syncthreads();
        if (threadIdx.x == 0u) current_count_shared = source_count;
        __syncthreads();
        continue;
      }

      const std::uint32_t merged_count = current_count + source_count;
      const std::uint32_t tile =
          (merged_count + blockDim.x - 1u) / blockDim.x;
      const std::uint32_t diagonal =
          min(threadIdx.x * tile, merged_count);
      std::uint32_t partition_low =
          diagonal > current_count ? diagonal - current_count : 0u;
      std::uint32_t partition_high = min(diagonal, source_count);
      while (partition_low < partition_high) {
        const std::uint32_t source_index =
            (partition_low + partition_high) >> 1u;
        const std::uint32_t current_index = diagonal - source_index;
        if (source_index < source_count && current_index > 0u &&
            current[current_index - 1u].key >= source[source_index].key)
          partition_low = source_index + 1u;
        else
          partition_high = source_index;
      }
      std::uint32_t source_index = partition_low;
      std::uint32_t current_index = diagonal - source_index;
#pragma unroll
      for (std::uint32_t item = 0u; item < 2u; ++item) {
        const std::uint32_t output_index = diagonal + item;
        if (item >= tile || output_index >= merged_count) break;
        const bool choose_source = source_index < source_count &&
            (current_index >= current_count ||
             source[source_index].key <= current[current_index].key);
        workspace.merged[output_index] = choose_source
            ? source[source_index++] : current[current_index++];
      }
      __syncthreads();

      const std::uint32_t chunk_begin = threadIdx.x * 2u;
      std::uint32_t local_winners = 0u;
      bool winner[2]{};
#pragma unroll
      for (std::uint32_t item = 0u; item < 2u; ++item) {
        const std::uint32_t index = chunk_begin + item;
        winner[item] = index < merged_count &&
            (index + 1u == merged_count ||
             workspace.merged[index].key !=
                 workspace.merged[index + 1u].key);
        local_winners += winner[item];
      }
      std::uint32_t thread_base{}, winner_count{};
      BlockScan(scan_storage).ExclusiveSum(local_winners, thread_base,
                                            winner_count);
      std::uint32_t local_rank = 0u;
#pragma unroll
      for (std::uint32_t item = 0u; item < 2u; ++item)
        if (winner[item])
          current[thread_base + local_rank++] =
              workspace.merged[chunk_begin + item];
      __syncthreads();
      if (threadIdx.x == 0u) current_count_shared = winner_count;
      __syncthreads();
      }
    }
  }

  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  constexpr unsigned full_mask = 0xffffffffu;
  const std::uint32_t current_count = current_count_shared;
  const std::uint32_t base_section_count = base_section_count_shared;
  const RouteHeader foundation_header = foundation_level < active_levels
      ? route_headers[descriptor_index(q, foundation_level)]
      : RouteHeader{};

  // Schedule fragments by input-row count.
  if (threadIdx.x == 0u) {
    minimum_work_shared = 0xffffffffu;
    maximum_work_shared = 0u;
  }
  __syncthreads();
  for (std::uint32_t fragment_index = fragment_begin + threadIdx.x;
       fragment_index < fragment_end; fragment_index += blockDim.x) {
    const SectionRangeFragment fragment = fragments[fragment_index];
    const std::uint32_t update_begin = lower_bound_rows(
        current, current_count, fragment.low_suffix);
    const std::uint32_t update_end = upper_bound_rows(
        current, current_count, fragment.high_suffix);
    RangeFragmentBounds bounds{update_begin, update_end, 0u, 0u};
    std::uint32_t base_work = 0u;
    if (foundation_header.count == 1u) {
      const Descriptor descriptor =
          route_slices[foundation_header.begin].rows;
      const ResidentRows rows = arena + descriptor.offset();
      if (foundation_ranks_valid_shared) {
        const std::uint32_t low_cell =
            std::uint32_t{fragment.low_suffix} / kFoundationCellKeys;
        const std::uint32_t low_begin =
            foundation_cell_ranks[low_cell];
        const std::uint32_t low_end =
            foundation_cell_ranks[low_cell + 1u];
        bounds.base_begin = low_begin + lower_bound_rows(
            rows + low_begin, low_end - low_begin,
            fragment.low_suffix);

        const std::uint32_t high_cell =
            std::uint32_t{fragment.high_suffix} / kFoundationCellKeys;
        const std::uint32_t high_begin =
            foundation_cell_ranks[high_cell];
        const std::uint32_t high_end =
            foundation_cell_ranks[high_cell + 1u];
        bounds.base_end = high_begin + upper_bound_rows(
            rows + high_begin, high_end - high_begin,
            fragment.high_suffix);
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
    const std::uint32_t work =
        update_end - update_begin + base_work;
    fragment_bounds_shared[fragment_index - fragment_begin] = bounds;
    fragment_work_shared[fragment_index - fragment_begin] = work;
    atomicMin(&minimum_work_shared, work);
    atomicMax(&maximum_work_shared, work);
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    const std::uint32_t minimum_width =
        minimum_work_shared <= kRangeThreadWork ? 1u
        : minimum_work_shared <= kRangeSubgroupWork ? 8u : 32u;
    const std::uint32_t maximum_width =
        maximum_work_shared <= kRangeThreadWork ? 1u
        : maximum_work_shared <= kRangeSubgroupWork ? 8u : 32u;
    worker_width_shared = maximum_width;
    dynamic_queue_shared = minimum_width != maximum_width;
    next_fragment_shared = fragment_begin;
  }
  __syncthreads();

  // Warp scans form disjoint interval unions.
  const std::uint32_t planning_warp = threadIdx.x >> 5u;
  const std::uint32_t planning_lane = threadIdx.x & 31u;
  constexpr unsigned planning_mask = 0xffffffffu;
  const std::uint32_t planning_index = fragment_begin + threadIdx.x;
  const bool planning_valid = planning_index < fragment_end;
  SectionRangeFragment planning_fragment{};
  RangeFragmentBounds planning_bounds{};
  unsigned long long planning_direct = 0ull;
  if (planning_valid) {
    planning_fragment = fragments[planning_index];
    planning_bounds =
        fragment_bounds_shared[planning_index - fragment_begin];
    planning_direct =
        fragment_work_shared[planning_index - fragment_begin];
  }

  // Split after the prior maximum endpoint.
  std::uint32_t prefix_high = planning_valid
      ? planning_fragment.high_suffix : 0u;
  for (std::uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
    const std::uint32_t preceding =
        __shfl_up_sync(planning_mask, prefix_high, offset);
    if (planning_lane >= offset)
      prefix_high = max(prefix_high, preceding);
  }
  const std::uint32_t preceding_high =
      __shfl_up_sync(planning_mask, prefix_high, 1u);
  const bool union_leader = planning_valid &&
      (planning_lane == 0u ||
       std::uint32_t{planning_fragment.low_suffix} > preceding_high);
  const unsigned valid_lanes =
      __ballot_sync(planning_mask, planning_valid);
  const unsigned union_leaders =
      __ballot_sync(planning_mask, union_leader);

  std::uint32_t prefix_update_end = planning_bounds.update_end;
  std::uint32_t prefix_base_end = planning_bounds.base_end;
  for (std::uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
    const std::uint32_t preceding_update =
        __shfl_up_sync(planning_mask, prefix_update_end, offset);
    const std::uint32_t preceding_base =
        __shfl_up_sync(planning_mask, prefix_base_end, offset);
    if (planning_lane >= offset) {
      prefix_update_end = max(prefix_update_end, preceding_update);
      prefix_base_end = max(prefix_base_end, preceding_base);
    }
  }
  const unsigned through_lane = planning_lane == 31u
      ? 0xffffffffu : ((1u << (planning_lane + 1u)) - 1u);
  const unsigned preceding_leaders = union_leaders & through_lane;
  const std::uint32_t leader_lane = preceding_leaders
      ? 31u - static_cast<std::uint32_t>(__clz(preceding_leaders)) : 0u;
  const std::uint32_t union_update_begin = __shfl_sync(
      planning_mask, planning_bounds.update_begin, leader_lane);
  const std::uint32_t union_base_begin = __shfl_sync(
      planning_mask, planning_bounds.base_begin, leader_lane);
  const bool union_last = planning_valid &&
      (planning_lane == 31u ||
       (valid_lanes & (1u << (planning_lane + 1u))) == 0u ||
       (union_leaders & (1u << (planning_lane + 1u))) != 0u);
  unsigned long long planning_union = union_last
      ? std::uint64_t{prefix_update_end - union_update_begin} +
          (foundation_header.count <= 1u
              ? prefix_base_end - union_base_begin : 0u)
      : 0ull;
  for (std::uint32_t offset = 16u; offset; offset >>= 1u) {
    planning_direct += __shfl_down_sync(
        planning_mask, planning_direct, offset);
    planning_union += __shfl_down_sync(
        planning_mask, planning_union, offset);
  }
  if (planning_lane == 0u) {
    union_direct_warp_shared[planning_warp] = planning_direct;
    union_sweep_warp_shared[planning_warp] = planning_union;
    union_count_warp_shared[planning_warp] = __popc(union_leaders);
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    unsigned long long direct_work = 0ull;
    unsigned long long union_work = 0ull;
    std::uint32_t union_count = 0u;
    for (std::uint32_t warp = 0u; warp < kSectionRangeWarps; ++warp) {
      direct_work += union_direct_warp_shared[warp];
      union_work += union_sweep_warp_shared[warp];
      union_count += union_count_warp_shared[warp];
    }
    const std::uint32_t fragment_count = fragment_end - fragment_begin;

    // Count split routes only for dense overlap.
    if (foundation_header.count > 1u && fragment_count >= 8u &&
        union_count * 4u <= fragment_count) {
      for (std::uint32_t warp_begin = fragment_begin;
           warp_begin < fragment_end; warp_begin += 32u) {
        const std::uint32_t warp_end = min(warp_begin + 32u, fragment_end);
        std::uint32_t cursor = warp_begin;
        while (cursor < warp_end) {
          const SectionRangeFragment first = fragments[cursor];
          const std::uint32_t low = first.low_suffix;
          std::uint32_t high = first.high_suffix;
          ++cursor;
          while (cursor < warp_end) {
            const SectionRangeFragment next = fragments[cursor];
            if (std::uint32_t{next.low_suffix} > high) break;
            high = max(high, std::uint32_t{next.high_suffix});
            ++cursor;
          }
          union_work += route_range_row_count(
              low, high, arena, foundation_header, route_slices);
        }
      }
    } else if (foundation_header.count > 1u) {
      union_work = ~0ull;
    }
    use_union_sweep_shared = !overflow_shared && fragment_count >= 8u &&
        union_work != 0ull && direct_work >= union_work * 2ull;
  }
  __syncthreads();

  if (use_union_sweep_shared) {
    // Four warps sweep independent interval groups.
    const std::uint32_t union_warp = threadIdx.x >> 5u;
    const std::uint32_t union_lane = threadIdx.x & 31u;
    constexpr unsigned union_mask = 0xffffffffu;
    std::uint32_t cursor = min(
        fragment_begin + union_warp * 32u, fragment_end);
    const std::uint32_t warp_fragment_end = min(cursor + 32u, fragment_end);
    while (cursor < warp_fragment_end) {
      std::uint32_t union_end = cursor;
      std::uint32_t union_low = 0u;
      std::uint32_t union_high = 0u;
      if (union_lane == 0u) {
        const SectionRangeFragment first = fragments[cursor];
        union_low = first.low_suffix;
        union_high = first.high_suffix;
        union_end = cursor + 1u;
        while (union_end < warp_fragment_end) {
          const SectionRangeFragment next = fragments[union_end];
          if (std::uint32_t{next.low_suffix} > union_high) break;
          union_high = max(union_high,
                           std::uint32_t{next.high_suffix});
          ++union_end;
        }
      }
      union_end = __shfl_sync(union_mask, union_end, 0u);
      union_low = __shfl_sync(union_mask, union_low, 0u);
      union_high = __shfl_sync(union_mask, union_high, 0u);

      const std::uint32_t fragment_index = cursor + union_lane;
      const bool fragment_active = fragment_index < union_end;
      SectionRangeFragment fragment{};
      RangeFragmentBounds fragment_bounds{};
      unsigned long long fragment_sum = 0ull;
      if (fragment_active) {
        fragment = fragments[fragment_index];
        fragment_bounds =
            fragment_bounds_shared[fragment_index - fragment_begin];
        for (std::uint32_t index = fragment_bounds.update_begin;
             index < fragment_bounds.update_end; ++index) {
          const Row row = current[index];
          if ((row.flags & kTombstone) == 0u)
            fragment_sum = Aggregate::consume(fragment_sum, row);
        }
      }

      // Reuse saved bounds for single-route sections.
      std::uint32_t saved_union_begin = fragment_bounds.base_begin;
      std::uint32_t saved_union_end = fragment_active
          ? fragment_bounds.base_end : 0u;
      if (foundation_header.count == 1u) {
        saved_union_begin = __shfl_sync(
            union_mask, saved_union_begin, 0u);
        for (std::uint32_t offset = 16u; offset; offset >>= 1u)
          saved_union_end = max(saved_union_end, __shfl_down_sync(
              union_mask, saved_union_end, offset));
        saved_union_end = __shfl_sync(
            union_mask, saved_union_end, 0u);
      }

      // Scan every route in bounded row tiles.
      for (std::uint32_t local_route = 0u;
           local_route < foundation_header.count; ++local_route) {
        const RouteSlice route =
            route_slices[foundation_header.begin + local_route];
        const ResidentRows route_rows = arena + route.rows.offset();
        std::uint32_t route_scan_begin = 0u;
        std::uint32_t route_scan_end = 0u;
        if (foundation_header.count == 1u) {
          route_scan_begin = saved_union_begin;
          route_scan_end = saved_union_end;
        } else if (union_lane == 0u && route.suffix_end > union_low &&
                   route.suffix_begin <= union_high) {
          route_scan_begin = lower_bound_rows(
              route_rows, route.rows.count(), union_low);
          route_scan_end = upper_bound_rows(
              route_rows, route.rows.count(), union_high);
        }
        route_scan_begin =
            __shfl_sync(union_mask, route_scan_begin, 0u);
        route_scan_end = __shfl_sync(union_mask, route_scan_end, 0u);

        std::uint32_t fragment_route_begin = 0u;
        std::uint32_t fragment_route_end = 0u;
        if (fragment_active && route_scan_begin != route_scan_end) {
          if (foundation_header.count == 1u) {
            fragment_route_begin = fragment_bounds.base_begin;
            fragment_route_end = fragment_bounds.base_end;
          } else {
            fragment_route_begin = lower_bound_rows(
                route_rows, route.rows.count(), fragment.low_suffix);
            fragment_route_end = upper_bound_rows(
                route_rows, route.rows.count(), fragment.high_suffix);
          }
        }

        constexpr std::uint32_t kUnionTileRows = 128u;
        for (std::uint32_t tile_begin = route_scan_begin;
             tile_begin < route_scan_end;
             tile_begin += kUnionTileRows) {
          const std::uint32_t tile_end =
              min(tile_begin + kUnionTileRows, route_scan_end);
          const std::uint32_t tile_count = tile_end - tile_begin;
          for (std::uint32_t index = union_lane; index < tile_count;
               index += 32u)
            workspace.base_rows[union_warp][index] =
                route_rows[tile_begin + index];
          __syncwarp(union_mask);

          const std::uint32_t chunk_begin = union_lane * 4u;
          unsigned long long item_prefix[4]{};
          unsigned long long chunk_sum = 0ull;
#pragma unroll
          for (std::uint32_t item = 0u; item < 4u; ++item) {
            const std::uint32_t index = chunk_begin + item;
            if (index < tile_count) {
              const Row row = workspace.base_rows[union_warp][index];
              bool covered = false;
              if (current_count) {
                const std::uint32_t update =
                    lower_bound_rows(current, current_count, row.key);
                covered = update < current_count &&
                    current[update].key == row.key;
              }
              if (!covered && (row.flags & kTombstone) == 0u)
                chunk_sum = Aggregate::consume(chunk_sum, row);
            }
            item_prefix[item] = chunk_sum;
          }
          unsigned long long chunk_prefix = chunk_sum;
          for (std::uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
            const unsigned long long preceding =
                __shfl_up_sync(union_mask, chunk_prefix, offset);
            if (union_lane >= offset) chunk_prefix += preceding;
          }
          const unsigned long long chunk_base = chunk_prefix - chunk_sum;
          if (union_lane == 0u)
            base_mask_workspace
                .warp_tile_prefix[union_warp][0] = 0ull;
#pragma unroll
          for (std::uint32_t item = 0u; item < 4u; ++item) {
            const std::uint32_t index = chunk_begin + item;
            if (index < tile_count)
              base_mask_workspace
                  .warp_tile_prefix[union_warp][index + 1u] =
                      chunk_base + item_prefix[item];
          }
          __syncwarp(union_mask);

          if (fragment_active) {
            const std::uint32_t consume_begin =
                max(fragment_route_begin, tile_begin);
            const std::uint32_t consume_end =
                min(fragment_route_end, tile_end);
            if (consume_begin < consume_end)
              fragment_sum += base_mask_workspace
                  .warp_tile_prefix[union_warp]
                      [consume_end - tile_begin] -
                  base_mask_workspace
                      .warp_tile_prefix[union_warp]
                          [consume_begin - tile_begin];
          }
          __syncwarp(union_mask);
        }
      }
      if (fragment_active)
        aggregate_partials[fragment.original_index] = fragment_sum;
      cursor = union_end;
    }
    break;
  }

  // Choose worker width from actual rows.
  if (!overflow_shared &&
      (foundation_header.count > 1u || dynamic_queue_shared)) {
    for (std::uint32_t fragment_index = fragment_begin + threadIdx.x;
         fragment_index < fragment_end; fragment_index += blockDim.x) {
      const SectionRangeFragment fragment = fragments[fragment_index];
      const std::uint32_t work =
          fragment_work_shared[fragment_index - fragment_begin];
      if (work <= kRangeThreadWork) {
        aggregate_partials[fragment.original_index] =
            cooperative_sum_visible_route_runs<Aggregate>(
                fragment.low_suffix, fragment.high_suffix,
                current, current_count, arena, foundation_header,
                route_slices, 0u, 1u);
      }
    }
    __syncthreads();

    constexpr std::uint32_t kSubgroup = 8u;
    const std::uint32_t subgroup = threadIdx.x / kSubgroup;
    const std::uint32_t subgroup_lane = threadIdx.x & (kSubgroup - 1u);
    const std::uint32_t subgroup_leader = subgroup * kSubgroup;
    const unsigned subgroup_mask =
        ((1u << kSubgroup) - 1u) << subgroup_leader;
    for (std::uint32_t fragment_index = fragment_begin + subgroup;
         fragment_index < fragment_end;
         fragment_index += blockDim.x / kSubgroup) {
      SectionRangeFragment fragment{};
      std::uint32_t work = 0u;
      if (subgroup_lane == 0u) {
        fragment = fragments[fragment_index];
        work = fragment_work_shared[fragment_index - fragment_begin];
      }
      fragment.original_index = __shfl_sync(
          subgroup_mask, fragment.original_index, subgroup_leader);
      fragment.low_suffix = static_cast<std::uint16_t>(__shfl_sync(
          subgroup_mask, std::uint32_t{fragment.low_suffix}, subgroup_leader));
      fragment.high_suffix = static_cast<std::uint16_t>(__shfl_sync(
          subgroup_mask, std::uint32_t{fragment.high_suffix}, subgroup_leader));
      work = __shfl_sync(subgroup_mask, work, subgroup_leader);
      if (work > kRangeThreadWork && work <= kRangeSubgroupWork) {
        unsigned long long value =
            cooperative_sum_visible_route_runs<Aggregate>(
                fragment.low_suffix, fragment.high_suffix,
                current, current_count, arena, foundation_header,
                route_slices, subgroup_lane, kSubgroup);
        for (std::uint32_t offset = kSubgroup / 2u; offset; offset >>= 1u)
          value += __shfl_down_sync(
              subgroup_mask, value, offset, kSubgroup);
        if (subgroup_lane == 0u)
          aggregate_partials[fragment.original_index] = value;
      }
    }
    __syncthreads();

    for (std::uint32_t fragment_index = fragment_begin + warp;
         fragment_index < fragment_end;
         fragment_index += kSectionRangeWarps) {
      SectionRangeFragment fragment{};
      std::uint32_t work = 0u;
      if (lane == 0u) {
        fragment = fragments[fragment_index];
        work = fragment_work_shared[fragment_index - fragment_begin];
      }
      fragment.original_index =
          __shfl_sync(full_mask, fragment.original_index, 0u);
      fragment.low_suffix = static_cast<std::uint16_t>(__shfl_sync(
          full_mask, std::uint32_t{fragment.low_suffix}, 0u));
      fragment.high_suffix = static_cast<std::uint16_t>(__shfl_sync(
          full_mask, std::uint32_t{fragment.high_suffix}, 0u));
      work = __shfl_sync(full_mask, work, 0u);
      if (work > kRangeSubgroupWork) {
        unsigned long long value =
            cooperative_sum_visible_route_runs<Aggregate>(
                fragment.low_suffix, fragment.high_suffix,
                current, current_count, arena, foundation_header,
                route_slices, lane, 32u);
        for (std::uint32_t offset = 16u; offset; offset >>= 1u)
          value += __shfl_down_sync(full_mask, value, offset);
        if (lane == 0u)
          aggregate_partials[fragment.original_index] = value;
      }
    }
    break;
  }
  const ResidentRows foundation_rows =
      arena + foundation_descriptor_shared.offset();

  if (!overflow_shared && current_count &&
      base_section_count <= kSectionBaseMaskRows) {
    const std::uint32_t words = (base_section_count + 31u) >> 5u;
    for (std::uint32_t word = threadIdx.x; word < words;
         word += blockDim.x)
      base_mask_workspace.section[word] = 0u;
    __syncthreads();
    for (std::uint32_t update = threadIdx.x; update < current_count;
         update += blockDim.x) {
      const std::uint32_t key = current[update].key;
      const std::uint32_t position = lower_bound_rows(
          foundation_rows, base_section_count, key);
      if (position < base_section_count &&
          foundation_rows[position].key == key)
        atomicOr(base_mask_workspace.section + (position >> 5u),
                 1u << (position & 31u));
    }
    __syncthreads();
    if (threadIdx.x == 0u) section_base_mask_valid_shared = 1u;
  } else {
    if (threadIdx.x == 0u) section_base_mask_valid_shared = 0u;
  }
  __syncthreads();
  if (!overflow_shared && worker_width_shared == 1u) {
      for (std::uint32_t fragment_index = fragment_begin + threadIdx.x;
           fragment_index < fragment_end; fragment_index += blockDim.x) {
        const SectionRangeFragment fragment = fragments[fragment_index];
        aggregate_partials[fragment.original_index] =
            cooperative_sum_visible_route_runs<Aggregate>(
                fragment.low_suffix, fragment.high_suffix,
                current, current_count, arena, foundation_header,
                route_slices, 0u, 1u);
      }
      break;
  }
  if (!overflow_shared && !dynamic_queue_shared &&
      worker_width_shared == 8u &&
      (section_base_mask_valid_shared || current_count == 0u)) {
      constexpr std::uint32_t kGroup = 4u;
      constexpr std::uint32_t kSubgroup = 32u / kGroup;
      constexpr std::uint32_t kTileRows = 128u;
      const std::uint32_t subgroup = lane / kSubgroup;
      const std::uint32_t subgroup_lane = lane & (kSubgroup - 1u);
      const unsigned subgroup_mask =
          ((1u << kSubgroup) - 1u) << (subgroup * kSubgroup);
      std::uint32_t group_begin =
          fragment_begin + warp * kGroup;
      while (true) {
        if (dynamic_queue_shared) {
          if (lane == 0u)
            group_begin = atomicAdd(&next_fragment_shared, kGroup);
          group_begin = __shfl_sync(full_mask, group_begin, 0u);
        }
        if (group_begin >= fragment_end) break;
        const std::uint32_t fragment_index = group_begin + subgroup;
        const bool active = fragment_index < fragment_end;
        SectionRangeFragment fragment{};
        if (active && subgroup_lane == 0u)
          fragment = fragments[fragment_index];
        std::uint32_t update_begin = 0u, update_end = 0u;
        std::uint32_t base_begin = 0u, base_end = 0u;
        if (active && subgroup_lane == 0u) {
          const RangeFragmentBounds bounds =
              fragment_bounds_shared[fragment_index - fragment_begin];
          update_begin = bounds.update_begin;
          update_end = bounds.update_end;
          base_begin = bounds.base_begin;
          base_end = bounds.base_end;
        }
        const std::uint32_t leader = subgroup * kSubgroup;
        update_begin = __shfl_sync(full_mask, update_begin, leader);
        update_end = __shfl_sync(full_mask, update_end, leader);
        base_begin = __shfl_sync(full_mask, base_begin, leader);
        base_end = __shfl_sync(full_mask, base_end, leader);

        unsigned long long local = 0ull;
        if (active) {
          for (std::uint32_t index = update_begin + subgroup_lane;
               index < update_end; index += kSubgroup) {
            const Row row = current[index];
            if ((row.flags & kTombstone) == 0u)
              local = Aggregate::consume(local, row);
          }
        }

        std::uint32_t union_begin = base_section_count;
        std::uint32_t union_end = 0u;
        std::uint32_t direct_rows = 0u;
#pragma unroll
        for (std::uint32_t member = 0u; member < kGroup; ++member) {
          const std::uint32_t member_lane = member * kSubgroup;
          const bool member_active =
              __shfl_sync(full_mask, active, member_lane);
          const std::uint32_t member_begin =
              __shfl_sync(full_mask, base_begin, member_lane);
          const std::uint32_t member_end =
              __shfl_sync(full_mask, base_end, member_lane);
          if (member_active) {
            union_begin = min(union_begin, member_begin);
            union_end = max(union_end, member_end);
            direct_rows += member_end - member_begin;
          }
        }
        const std::uint32_t union_rows = union_end - union_begin;
        const bool share_rows = union_rows != 0u &&
            direct_rows >= union_rows + (union_rows >> 1u);

        if (share_rows) {
          for (std::uint32_t tile_begin = union_begin;
               tile_begin < union_end; tile_begin += kTileRows) {
            const std::uint32_t tile_end =
                min(tile_begin + kTileRows, union_end);
            for (std::uint32_t index = lane;
                 index < tile_end - tile_begin; index += 32u) {
              const std::uint32_t position =
                  tile_begin + index;
              workspace.base_rows[warp][index] =
                  foundation_rows[position];
            }
            __syncwarp();
            if (active) {
              const std::uint32_t consume_begin =
                  max(base_begin, tile_begin);
              const std::uint32_t consume_end =
                  min(base_end, tile_end);
              for (std::uint32_t index =
                       consume_begin + subgroup_lane;
                   index < consume_end; index += kSubgroup) {
                const Row row =
                    workspace.base_rows[warp][index - tile_begin];
                bool covered = false;
                if (section_base_mask_valid_shared) {
                  covered =
                      (base_mask_workspace.section[index >> 5u] &
                       (1u << (index & 31u))) != 0u;
                } else if (current_count) {
                  const std::uint32_t update_position =
                      lower_bound_rows(current, current_count,
                                       row.key);
                  covered = update_position < current_count &&
                      current[update_position].key == row.key;
                }
                if (!covered && (row.flags & kTombstone) == 0u)
                  local = Aggregate::consume(local, row);
              }
            }
            __syncwarp();
          }
        } else if (active) {
          for (std::uint32_t index = base_begin + subgroup_lane;
               index < base_end; index += kSubgroup) {
            const std::uint32_t position =
                index;
            const Row row = foundation_rows[position];
            bool covered = false;
            if (section_base_mask_valid_shared) {
              covered =
                  (base_mask_workspace.section[index >> 5u] &
                   (1u << (index & 31u))) != 0u;
            } else if (current_count) {
              const std::uint32_t update_position =
                  lower_bound_rows(current, current_count, row.key);
              covered = update_position < current_count &&
                  current[update_position].key == row.key;
            }
            if (!covered && (row.flags & kTombstone) == 0u)
              local = Aggregate::consume(local, row);
          }
        }

        for (std::uint32_t offset = kSubgroup / 2u; offset;
             offset >>= 1u) {
          const unsigned long long other = __shfl_down_sync(
              subgroup_mask, local, offset, kSubgroup);
          local = Aggregate{}(local, other);
        }
        if (active && subgroup_lane == 0u)
          aggregate_partials[fragment.original_index] = local;
        if (!dynamic_queue_shared)
          group_begin += kSectionRangeWarps * kGroup;
      }
      break;
  }
  std::uint32_t fragment_index = fragment_begin + warp;
  while (true) {
    if (dynamic_queue_shared) {
      if (lane == 0u)
        fragment_index = atomicAdd(&next_fragment_shared, 1u);
      fragment_index = __shfl_sync(full_mask, fragment_index, 0u);
    }
    if (fragment_index >= fragment_end) break;
    const SectionRangeFragment fragment = fragments[fragment_index];
    const std::uint32_t q_low = q << 16u;
    const std::uint32_t low = q_low | fragment.low_suffix;
    const std::uint32_t high = q_low | fragment.high_suffix;
    unsigned long long local = 0ull;
    if (overflow_shared) {
      local = warp_sum_visible_by_verification(
          q, low, high, raw_keys, raw_payloads, raw_offsets, batch_stride,
          pending_batches, arena, descriptors,
          route_headers, route_slices, active_levels, occupied_levels);
    } else {
      std::uint32_t update_begin = 0u, update_end = 0u;
      std::uint32_t begin = 0u, end = 0u;
      if (lane == 0u) {
        const RangeFragmentBounds bounds =
            fragment_bounds_shared[fragment_index - fragment_begin];
        update_begin = bounds.update_begin;
        update_end = bounds.update_end;
        begin = bounds.base_begin;
        end = bounds.base_end;
      }
      update_begin = __shfl_sync(full_mask, update_begin, 0u);
      update_end = __shfl_sync(full_mask, update_end, 0u);
      begin = __shfl_sync(full_mask, begin, 0u);
      end = __shfl_sync(full_mask, end, 0u);
      const std::uint32_t update_count = update_end - update_begin;
      for (std::uint32_t index = update_begin + lane; index < update_end;
           index += 32u) {
        const Row row = current[index];
        if ((row.flags & kTombstone) == 0u)
          local = Aggregate::consume(local, row);
      }

      if (!update_count) {
        for (std::uint32_t index = begin + lane; index < end; index += 32u)
          if ((foundation_rows[index].flags & kTombstone) == 0u)
            local = Aggregate::consume(local, foundation_rows[index]);
      } else {
        const Row *updates = current + update_begin;
        const std::uint32_t base_count = end - begin;
        if (section_base_mask_valid_shared) {
          for (std::uint32_t index = begin + lane; index < end;
               index += 32u)
            if ((base_mask_workspace.section[index >> 5u] &
                 (1u << (index & 31u))) == 0u)
              if ((foundation_rows[index].flags & kTombstone) == 0u)
                local = Aggregate::consume(local, foundation_rows[index]);
        } else if (base_count <= kFragmentBaseMaskRows) {
          const std::uint32_t mask_words = (base_count + 31u) >> 5u;
          for (std::uint32_t word = lane; word < mask_words; word += 32u)
            base_mask_workspace.fragments[warp][word] = 0u;
          __syncwarp();

          for (std::uint32_t update = lane; update < update_count;
               update += 32u) {
            const std::uint32_t key = updates[update].key;
            const std::uint32_t position =
                lower_bound_rows(foundation_rows + begin, base_count, key);
            if (position < base_count &&
                foundation_rows[begin + position].key == key)
              atomicOr(base_mask_workspace.fragments[warp] +
                           (position >> 5u),
                       1u << (position & 31u));
          }
          __syncwarp();
          for (std::uint32_t index = lane; index < base_count; index += 32u)
            if ((base_mask_workspace.fragments[warp][index >> 5u] &
                 (1u << (index & 31u))) == 0u)
              if ((foundation_rows[begin + index].flags & kTombstone) == 0u)
                local = Aggregate::consume(
                    local, foundation_rows[begin + index]);
        } else {
          for (std::uint32_t index = begin + lane; index < end;
               index += 32u) {
            const Row row = foundation_rows[index];
            const std::uint32_t update_position =
                lower_bound_rows(updates, update_count, row.key);
            if (update_position == update_count ||
                updates[update_position].key != row.key)
              if ((row.flags & kTombstone) == 0u)
                local = Aggregate::consume(local, row);
          }
        }
      }
      for (std::uint32_t offset = 16u; offset; offset >>= 1u)
        local += __shfl_down_sync(full_mask, local, offset);
    }
    if (lane == 0u)
      aggregate_partials[fragment.original_index] = local;
    if (!dynamic_queue_shared)
      fragment_index += kSectionRangeWarps;
  }
    } while (false);
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
    const RouteSlice *route_slices, const std::uint32_t *raw_keys,
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
      const unsigned long long value = warp_sum_visible_by_verification(
          q, low, high, raw_keys, raw_payloads, raw_offsets, batch_stride,
          pending_batches, arena, descriptors, route_headers, route_slices,
          active_levels, occupied_levels);
      if (lane == 0u) aggregate_partials[fragment_index] = value;
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
    const unsigned long long value = warp_sum_visible_by_verification(
        q, low, high, raw_keys, raw_payloads, raw_offsets, batch_stride,
        pending_batches, arena, descriptors, route_headers, route_slices,
        active_levels, occupied_levels);
    if (lane == 0u) aggregate_partials[fragment_index] = value;
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

// Count grouped raw intervals without suffix sorting.
__global__ void count_direct_epoch_merge_work_kernel(
    const std::uint32_t *raw_offsets, std::uint32_t pending_batches,
    const Descriptor *descriptors, const DeviceManifest *manifests,
    const std::uint32_t *active_manifest, ResidentPublicationPlan *plan,
    std::uint64_t *raw_counts, std::uint32_t *crowded_flag) {
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
  cudaGraphSetConditional(
      conditional, atomicAdd(const_cast<std::uint32_t *>(crowded_flag), 0u)
                           ? 1u : 0u);
}

__global__ void set_staged_epoch_mode_kernel(std::uint32_t *mode) {
  if (!blockIdx.x && !threadIdx.x) *mode = 1u;
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
    if (plan->status) {
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
    BalancedMergeJob *jobs, std::uint64_t *job_raw_reservations) {
  __shared__ std::uint64_t counts[kPlanningTileQuotients];
  const std::uint32_t tile = blockIdx.x;
  if (tile >= kPlanningTiles) return;
  const std::uint32_t first = tile * kPlanningTileQuotients;
  counts[threadIdx.x] = raw_counts[first + threadIdx.x];
  counts[threadIdx.x + blockDim.x] =
      raw_counts[first + threadIdx.x + blockDim.x];
  __syncthreads();
  if (threadIdx.x != 0u || plan->status) return;

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

__global__ void resolve_resident_job_boundaries_kernel(
    BalancedMergeJob *jobs, std::uint64_t *job_raw_reservations,
    const ResidentPublicationPlan *plan,
    const Row *current_rows, const std::uint32_t *current_offsets,
    ResidentRows arena, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    const DeviceManifest *manifests,
    const std::uint32_t *active_manifest) {
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
  const std::uint32_t raw = static_cast<std::uint32_t>(
      balanced_merge_prefix_count_warp(
          q, 1u << 16u, current_rows, current_offsets, arena,
          route_headers, route_slices, route_logical_begins,
          level_q_logical_offsets, plan->source_level_limit,
          manifest.occupied_level_mask));
  const std::uint32_t low_target = static_cast<std::uint32_t>(
      (std::uint64_t{raw} * job.hot_piece + job.hot_pieces - 1u) /
      job.hot_pieces);
  const std::uint32_t high_target = static_cast<std::uint32_t>(
      (std::uint64_t{raw} * (job.hot_piece + 1u) +
       job.hot_pieces - 1u) / job.hot_pieces);
  const std::uint32_t low = resident_hot_boundary_warp(
      q, low_target, current_rows, current_offsets, arena,
      route_headers, route_slices, route_logical_begins,
      level_q_logical_offsets, plan->source_level_limit,
      manifest.occupied_level_mask);
  const std::uint32_t high = job.hot_piece + 1u == job.hot_pieces
      ? (1u << 16u)
      : resident_hot_boundary_warp(
            q, high_target, current_rows, current_offsets, arena,
            route_headers, route_slices, route_logical_begins,
            level_q_logical_offsets, plan->source_level_limit,
            manifest.occupied_level_mask);
  const std::uint32_t exact = balanced_merge_prefix_count_warp(
      q, high, current_rows, current_offsets, arena, route_headers,
      route_slices, route_logical_begins, level_q_logical_offsets,
      plan->source_level_limit, manifest.occupied_level_mask) -
      balanced_merge_prefix_count_warp(
          q, low, current_rows, current_offsets, arena, route_headers,
          route_slices, route_logical_begins, level_q_logical_offsets,
          plan->source_level_limit, manifest.occupied_level_mask);
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

// Merge one adaptive quotient range per CTA.
__global__ void compact_direct_epoch_jobs_kernel(
    BalancedMergeJob *jobs, const ResidentPublicationPlan *plan,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const std::uint32_t *staged_keys,
    const Row *staged_rows, const std::uint32_t *staged_offsets,
    const std::uint32_t *staged_epoch_mode, ResidentRows arena,
    const DeviceManifest *manifests, const std::uint32_t *active_manifest,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    const RouteHeader *next_route_headers, RouteSlice *next_route_slices,
    std::uint32_t *section_output_counts, std::uint32_t *overflow_flag,
    std::uint32_t *level_cell_rank_blocks,
    std::uint16_t *level_cell_ranks, std::uint16_t *local_rank,
    std::uint16_t *level_guides) {
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
  if (plan->status) return;
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
    const bool cell_owned_shape =
        !crowded_piece && quotient_count <= kCellOwnedQuotients;
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
    if (valid)
      lookup_resident_only(
          key, i, out_values, out_found, arena, descriptors,
          route_headers, route_slices, route_logical_begins,
          level_q_logical_offsets, local_rank, level_guides,
          level_cell_rank_blocks, level_cell_ranks,
          active_levels, foundation_level, occupied_levels, query_ids);
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

  explicit GPULSMOpt(const DictionaryConfig &config)
      : batch_capacity_(std::min(
            gpulsmopt2_detail::kMaximumOperationTile,
            std::max<std::size_t>(1u, config.batch_capacity))),
        publication_capacity_(gpulsmopt2_detail::initial_storage_capacity(
            config.max_elements, batch_capacity_)),
        foundation_pool_capacity_(
            gpulsmopt2_detail::foundation_pool_capacity(
                publication_capacity_)),
        level_pool_capacity_(
            gpulsmopt2_detail::preassigned_level_pool_capacity(
                publication_capacity_,
                batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch)),
        level_rank_block_capacity_(
            gpulsmopt2_detail::preassigned_level_rank_blocks(
                publication_capacity_,
                batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch)),
        resident_merge_capacity_(
            gpulsmopt2_detail::select_balanced_merge_capacity()),
        resident_merge_workspace_bytes_(
            gpulsmopt2_detail::balanced_merge_dynamic_shared_bytes(
                resident_merge_capacity_)),
        maximum_resident_jobs_(
            gpulsmopt2_detail::maximum_resident_merge_jobs(
                publication_capacity_, resident_merge_capacity_)),
        route_stride_(gpulsmopt2_detail::adaptive_route_stride(
            publication_capacity_, resident_merge_capacity_)),
        local_rank_(gpulsmopt2_detail::kLocalRankEntries),
        level_guides_(
            gpulsmopt2_detail::kMaximumLevels *
                gpulsmopt2_detail::kGuideEntriesPerLevel,
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
            route_stride_ * gpulsmopt2_detail::kMaximumLevels,
            route_stride_ * gpulsmopt2_detail::kMaximumLevels),
        route_logical_begins_(
            route_stride_ * gpulsmopt2_detail::kMaximumLevels),
        route_quotients_(
            route_stride_ * gpulsmopt2_detail::kMaximumLevels),
        level_q_logical_offsets_(
            std::size_t{gpulsmopt2_detail::kMaximumLevels} *
            (gpulsmopt2_detail::kQuotients + 1u)),
        device_manifests_(2u),
        active_device_manifest_(1u),
        query_occupied_level_mask_(1u),
        staged_epoch_mode_(1u),
        resident_plan_(1u),
        publication_receipt_(1u),
        level_storage_spans_(gpulsmopt2_detail::kMaximumLevels),
        level_rank_spans_(gpulsmopt2_detail::kMaximumLevels),
        level_cell_rank_blocks_(
            std::size_t{gpulsmopt2_detail::kQuotients} *
            gpulsmopt2_detail::kMaximumLevels),
        level_cell_ranks_(
            level_rank_block_capacity_ *
            gpulsmopt2_detail::kFoundationCells),
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
            batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch),
        publication_epoch_assignments_b_(
            batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch),
        publication_keys_a_(gpulsmopt2_detail::kMaximumPublicationRows,
                            publication_capacity_),
        publication_rows_a_(gpulsmopt2_detail::kMaximumPublicationRows,
                            publication_capacity_),
        publication_selected_count_(1u),
        publication_batch_offsets_(gpulsmopt2_detail::kBatchesPerEpoch + 1u),
        foundation_source_offsets_(gpulsmopt2_detail::kQuotients + 1u),
        foundation_next_route_headers_(gpulsmopt2_detail::kQuotients),
        foundation_section_output_counts_(gpulsmopt2_detail::kQuotients + 1u),
        balanced_merge_raw_counts_(gpulsmopt2_detail::kQuotients),
        resident_tile_job_counts_(gpulsmopt2_detail::kPlanningTiles + 1u),
        resident_tile_job_offsets_(gpulsmopt2_detail::kPlanningTiles + 1u),
        resident_job_raw_reservations_(maximum_resident_jobs_ + 1u),
        resident_job_output_offsets_(maximum_resident_jobs_ + 1u),
        resident_route_counts_(gpulsmopt2_detail::kQuotients + 1u),
        resident_route_offsets_(gpulsmopt2_detail::kQuotients + 1u),
        resident_section_logical_offsets_(gpulsmopt2_detail::kQuotients + 1u),
        balanced_merge_jobs_(maximum_resident_jobs_),
        foundation_overflow_flag_(1u),
        admission_counts_(gpulsmopt2_detail::kQuotients + 1u),
        range_partials_(gpulsmopt2_detail::kRangeSchedulerBlocks),
        range_reduction_completion_(1u),
        range_fragment_total_(1u),
        range_total_receipt_(1u) {
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
            query_ids, query_occupied_level_mask_.data());
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
      launch_section_ranges(stream);
    } else {
      launch_fragment_ranges(fragment_count, query_count, batch, stream);
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
    return local_rank_.size() * sizeof(std::uint16_t) +
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
        staged_epoch_mode_.size() * sizeof(std::uint32_t) +
        resident_plan_.size() *
            sizeof(gpulsmopt2_detail::ResidentPublicationPlan) +
        level_storage_spans_.size() *
            sizeof(gpulsmopt2_detail::LevelStorageSpan) +
        level_rank_spans_.size() *
            sizeof(gpulsmopt2_detail::LevelRankSpan) +
        level_cell_rank_blocks_.size() * sizeof(std::uint32_t) +
        level_cell_ranks_.size() * sizeof(std::uint16_t) +
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
        publication_rows_a_.size() * sizeof(gpulsmopt2_detail::Row) +
        (publication_keys_a_.size() +
         publication_selected_count_.size() +
         publication_batch_offsets_.size()) * sizeof(std::uint32_t) +
        (foundation_source_offsets_.size() +
         foundation_section_output_counts_.size() +
         foundation_overflow_flag_.size()) * sizeof(std::uint32_t) +
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
    std::size_t capacity =
        batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch;
    std::uint32_t level = 0u;
    while (level + 1u < gpulsmopt2_detail::kMaximumLevels &&
           capacity <= count / 2u) {
      capacity *= 2u;
      ++level;
    }
    return level;
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
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, publication_epoch_keys_a_.data(),
        publication_epoch_keys_b_.data(),
        publication_epoch_assignments_a_.data(),
        publication_epoch_assignments_b_.data(), epoch_capacity, 0, 32, 0));
    auto row_output = thrust::make_transform_output_iterator(
        publication_rows_a_.data(), gpulsmopt2_detail::AssignmentRow{});
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        nullptr, raw_reduce_bytes, publication_epoch_keys_b_.data(),
        publication_keys_a_.data(), publication_epoch_assignments_b_.data(),
        row_output, publication_selected_count_.data(),
        gpulsmopt2_detail::NewestAssignment{}, epoch_capacity, 0));
    publication_temp_.resize(std::max(sort_bytes, raw_reduce_bytes));
  }

  void initialize_resident_workspace() {
    std::array<gpulsmopt2_detail::LevelStorageSpan,
               gpulsmopt2_detail::kMaximumLevels> spans{};
    std::array<gpulsmopt2_detail::LevelRankSpan,
               gpulsmopt2_detail::kMaximumLevels> rank_spans{};
    std::uint64_t cursor = foundation_pool_capacity_;
    std::uint64_t rank_cursor = 0u;
    std::size_t capacity = std::min(
        publication_capacity_,
        batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch);
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
    bytes = 0u;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, bytes, resident_job_raw_reservations_.data(),
        resident_job_output_offsets_.data(), maximum_resident_jobs_ + 1u,
        0));
    maximum_scan_bytes = std::max(maximum_scan_bytes, bytes);
    bytes = 0u;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, bytes, resident_route_counts_.data(),
        resident_route_offsets_.data(), gpulsmopt2_detail::kQuotients + 1u,
        0));
    maximum_scan_bytes = std::max(maximum_scan_bytes, bytes);
    bytes = 0u;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, bytes, foundation_section_output_counts_.data(),
        resident_section_logical_offsets_.data(),
        gpulsmopt2_detail::kQuotients + 1u, 0));
    maximum_scan_bytes = std::max(maximum_scan_bytes, bytes);
    resident_scan_temp_.resize(maximum_scan_bytes);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        gpulsmopt2_detail::compact_direct_epoch_jobs_kernel,
        gpulsmopt2_detail::kFoundationCompactionThreads,
        resident_merge_workspace_bytes_));
    resident_merge_blocks_ = static_cast<std::uint32_t>(
        std::max(1, blocks_per_sm) * properties.multiProcessorCount);
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

  cudaGraph_t capture_resident_merge_pre_graph(
      cudaStream_t capture_stream) {
    cudaGraph_t graph{};
    CUDA_CHECK(cudaStreamBeginCapture(
        capture_stream, cudaStreamCaptureModeThreadLocal));
    CUDA_CHECK(cudaMemsetAsync(
        foundation_overflow_flag_.data(), 0, sizeof(std::uint32_t),
        capture_stream));
    CUDA_CHECK(cudaMemsetAsync(
        staged_epoch_mode_.data(), 0, sizeof(std::uint32_t),
        capture_stream));
    gpulsmopt2_detail::count_direct_epoch_merge_work_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, capture_stream>>>(
            raw_offsets_.data(), gpulsmopt2_detail::kBatchesPerEpoch,
            descriptors_.data(), device_manifests_.data(),
            active_device_manifest_.data(), resident_plan_.data(),
            balanced_merge_raw_counts_.data(),
            foundation_overflow_flag_.data());
    CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
    return graph;
  }

  cudaGraph_t capture_crowded_epoch_stage_graph(
      cudaStream_t capture_stream) {
    cudaGraph_t graph{};
    CUDA_CHECK(cudaStreamBeginCapture(
        capture_stream, cudaStreamCaptureModeThreadLocal));
    gpulsmopt2_detail::build_publication_batch_offsets_kernel<<<
        1, 1, 0, capture_stream>>>(
            raw_offsets_.data(), publication_batch_offsets_.data());
    const dim3 publication_grid(
        blocks(batch_capacity_), gpulsmopt2_detail::kBatchesPerEpoch);
    gpulsmopt2_detail::pack_publication_epoch_kernel<<<
        publication_grid, gpulsmopt2_detail::kThreads, 0, capture_stream>>>(
            raw_keys_.data(), raw_payloads_.data(),
            static_cast<std::uint32_t>(batch_capacity_),
            publication_batch_offsets_.data(),
            publication_epoch_keys_a_.data(),
            publication_epoch_assignments_a_.data());
    const std::uint32_t epoch_capacity = static_cast<std::uint32_t>(
        batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch);
    gpulsmopt2_detail::pad_publication_epoch_kernel<<<
        blocks(epoch_capacity), gpulsmopt2_detail::kThreads, 0,
        capture_stream>>>(
            epoch_capacity, publication_batch_offsets_.data(),
            publication_epoch_keys_a_.data(),
            publication_epoch_assignments_a_.data());
    std::size_t workspace_bytes = publication_temp_.size();
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        publication_temp_.data(), workspace_bytes,
        publication_epoch_keys_a_.data(), publication_epoch_keys_b_.data(),
        publication_epoch_assignments_a_.data(),
        publication_epoch_assignments_b_.data(), epoch_capacity, 0, 32,
        capture_stream));
    auto row_output = thrust::make_transform_output_iterator(
        publication_rows_a_.data(), gpulsmopt2_detail::AssignmentRow{});
    workspace_bytes = publication_temp_.size();
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        publication_temp_.data(), workspace_bytes,
        publication_epoch_keys_b_.data(), publication_keys_a_.data(),
        publication_epoch_assignments_b_.data(), row_output,
        publication_selected_count_.data(),
        gpulsmopt2_detail::NewestAssignment{}, epoch_capacity,
        capture_stream));
    gpulsmopt2_detail::set_staged_epoch_mode_kernel<<<
        1, 1, 0, capture_stream>>>(staged_epoch_mode_.data());
    gpulsmopt2_detail::build_query_quotient_offsets_device_count_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, capture_stream>>>(
            publication_keys_a_.data(), publication_selected_count_.data(),
            foundation_source_offsets_.data());
    gpulsmopt2_detail::count_resident_merge_work_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, capture_stream>>>(
            foundation_source_offsets_.data(), descriptors_.data(),
            device_manifests_.data(), active_device_manifest_.data(),
            resident_plan_.data(), balanced_merge_raw_counts_.data());
    CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
    return graph;
  }

  cudaGraph_t capture_resident_merge_finish_graph(
      cudaStream_t capture_stream) {
    cudaGraph_t graph{};
    CUDA_CHECK(cudaStreamBeginCapture(
        capture_stream, cudaStreamCaptureModeThreadLocal));
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

    gpulsmopt2_detail::count_resident_planning_jobs_kernel<<<
        gpulsmopt2_detail::kPlanningTiles, gpulsmopt2_detail::kThreads, 0,
        capture_stream>>>(
            balanced_merge_raw_counts_.data(), resident_plan_.data(),
            resident_tile_job_counts_.data());
    std::size_t scan_bytes = resident_scan_temp_.size();
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
            balanced_merge_jobs_.data(), resident_job_raw_reservations_.data());
    gpulsmopt2_detail::resolve_resident_job_boundaries_kernel<<<
        resident_planner_blocks_, 32u, 0,
        capture_stream>>>(
            balanced_merge_jobs_.data(), resident_job_raw_reservations_.data(),
            resident_plan_.data(),
            publication_rows_a_.data(), foundation_source_offsets_.data(),
            resident_rows(), route_headers_.data(), route_slices_.data(),
            route_logical_begins_.data(), level_q_logical_offsets_.data(),
            device_manifests_.data(), active_device_manifest_.data());
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

    gpulsmopt2_detail::count_resident_route_slots_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, capture_stream>>>(
            balanced_merge_raw_counts_.data(), level_rank_spans_.data(),
            resident_plan_.data(), resident_route_counts_.data(),
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

    gpulsmopt2_detail::validate_direct_epoch_plan_kernel<<<
        1, 1, 0, capture_stream>>>(
            resident_plan_.data(), resident_tile_job_offsets_.data(),
            resident_job_output_offsets_.data(),
            resident_route_offsets_.data(),
            static_cast<std::uint32_t>(maximum_resident_jobs_),
            static_cast<std::uint32_t>(route_stride_));
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
            staged_epoch_mode_.data(),
            resident_rows(),
            device_manifests_.data(), active_device_manifest_.data(),
            descriptors_.data(), route_headers_.data(), route_slices_.data(),
            route_logical_begins_.data(), level_q_logical_offsets_.data(),
            foundation_next_route_headers_.data(), route_slices_.data(),
            foundation_section_output_counts_.data(),
            foundation_overflow_flag_.data(),
            level_cell_rank_blocks_.data(), level_cell_ranks_.data(),
            local_rank_.data(), level_guides_.data());
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
    CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
    return graph;
  }

  cudaGraph_t capture_resident_merge_graph(cudaStream_t capture_stream) {
    cudaGraph_t graph{};
    CUDA_CHECK(cudaGraphCreate(&graph, 0u));
    cudaGraphConditionalHandle crowded_conditional{};
    CUDA_CHECK(cudaGraphConditionalHandleCreate(
        &crowded_conditional, graph, 0u, cudaGraphCondAssignDefault));

    cudaGraph_t pre_graph{}, crowded_graph{}, normal_graph{}, finish_graph{};
    try {
      pre_graph = capture_resident_merge_pre_graph(capture_stream);
      crowded_graph = capture_crowded_epoch_stage_graph(capture_stream);
      CUDA_CHECK(cudaStreamBeginCapture(
          capture_stream, cudaStreamCaptureModeThreadLocal));
      CUDA_CHECK(cudaMemsetAsync(
          staged_epoch_mode_.data(), 0, sizeof(std::uint32_t),
          capture_stream));
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
      cudaGraphNode_t child{};
      CUDA_CHECK(cudaGraphAddChildGraphNode(
          &child, bodies[0], nullptr, 0u, crowded_graph));
      CUDA_CHECK(cudaGraphAddChildGraphNode(
          &child, bodies[1], nullptr, 0u, normal_graph));

      cudaGraphNode_t finish_node{};
      CUDA_CHECK(cudaGraphAddChildGraphNode(
          &finish_node, graph, &conditional, 1u, finish_graph));
    } catch (...) {
      if (pre_graph) cudaGraphDestroy(pre_graph);
      if (crowded_graph) cudaGraphDestroy(crowded_graph);
      if (normal_graph) cudaGraphDestroy(normal_graph);
      if (finish_graph) cudaGraphDestroy(finish_graph);
      cudaGraphDestroy(graph);
      throw;
    }
    CUDA_CHECK(cudaGraphDestroy(pre_graph));
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
    if (count <= publication_capacity_) return;
    if (count > gpulsmopt2_detail::kMaximumPublicationRows)
      throw std::bad_alloc();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    publication_keys_a_.grow(count);
    publication_rows_a_.grow(count);
    publication_capacity_ =
        std::min(publication_keys_a_.size(), publication_rows_a_.size());
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
      publication_failed_ = true;
      failed_epoch_signatures_ready_ = false;
      return;
    }

    const std::uint32_t destination = receipt.destination_level;
    if (destination >= gpulsmopt2_detail::kMaximumLevels) {
      publication_failed_ = true;
      publication_failure_status_ =
          gpulsmopt2_detail::kPublicationJobOverflow;
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
    std::fill_n(raw_batch_counts_, gpulsmopt2_detail::kBatchesPerEpoch, 0u);
    publication_failed_ = false;
    publication_failure_status_ = gpulsmopt2_detail::kPublicationSuccess;
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
    throw std::runtime_error(
        "GPULSMOpt publication failed with status " +
        std::to_string(publication_failure_status_) +
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
    publication_receipt_pending_ = false;
    publication_failed_ = false;
    publication_failure_status_ = gpulsmopt2_detail::kPublicationSuccess;
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

  void publish_epoch(cudaStream_t stream) {
    if (pending_records_ > publication_capacity_) {
      // Preserve an epoch that cannot reserve output.
      publication_failed_ = true;
      publication_failure_status_ =
          gpulsmopt2_detail::kPublicationOutputOverflow;
      failed_epoch_signatures_ready_ = true;
      return;
    }
    gpulsmopt2_detail::count_direct_epoch_records_kernel<<<1, 1, 0, stream>>>(
        raw_offsets_.data(), gpulsmopt2_detail::kBatchesPerEpoch,
        publication_selected_count_.data());
    CUDA_CHECK(cudaGraphLaunch(resident_publication_graph_exec_, stream));
    CUDA_CHECK(cudaGetLastError());

    // Copy the receipt to pinned host memory.
    CUDA_CHECK(cudaMemcpyAsync(
        publication_receipt_.data(), resident_plan_.data(),
        sizeof(gpulsmopt2_detail::ResidentPublicationPlan),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemsetAsync(
        raw_epoch_signatures_.data(), 0,
        raw_epoch_signatures_.size() * sizeof(std::uint64_t), stream));
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
  void launch_section_ranges(cudaStream_t stream) {
    gpulsmopt2_detail::cooperative_section_owned_range_kernel<
        gpulsmopt2_detail::SumRowsAggregate>
        <<<range_section_blocks_, gpulsmopt2_detail::kSectionRangeThreads,
           0, stream>>>(
            range_section_fragments_out_.data(),
            range_section_tasks_.data(),
            range_section_task_offsets_.data() +
                gpulsmopt2_detail::kQuotients,
            resident_rows(), descriptors_.data(), route_headers_.data(),
            route_slices_.data(), local_rank_.data(), raw_keys_.data(),
            raw_payloads_.data(),
            raw_offsets_.data(), static_cast<std::uint32_t>(batch_capacity_),
            pending_batches_, range_fragment_partials_.data(),
            query_occupied_level_mask_.data());
  }
  void launch_fragment_ranges(std::uint32_t fragment_count,
                              std::uint32_t query_count,
                              const DeviceRangeOutputBatch &batch,
                              cudaStream_t stream) {
    gpulsmopt2_detail::warp_range_fragment_kernel<
        gpulsmopt2_detail::SumRowsAggregate>
        <<<(fragment_count + 3u) / 4u, 128, 0, stream>>>(
            range_fragments_.data(), fragment_count,
            range_fragment_offsets_.data() + query_count,
            batch.lo, batch.hi, resident_rows(), descriptors_.data(),
            route_headers_.data(), route_slices_.data(),
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
  std::size_t foundation_pool_capacity_{};
  std::size_t level_pool_capacity_{};
  std::size_t level_rank_block_capacity_{};
  std::uint32_t resident_merge_capacity_{};
  std::size_t resident_merge_workspace_bytes_{};
  std::size_t maximum_resident_jobs_{};
  std::size_t route_stride_{};
  mutable std::mutex operation_mutex_;
  std::uint64_t host_occupied_level_mask_{};
  std::uint32_t pending_batches_{};
  std::uint32_t pending_records_{};
  std::uint32_t active_levels_{};
  bool publication_receipt_pending_{};
  bool publication_failed_{};
  bool failed_epoch_signatures_ready_{};
  std::uint32_t publication_failure_status_{};
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
  std::uint32_t resident_merge_blocks_{};
  std::uint32_t resident_planner_blocks_{};
  std::uint32_t range_section_blocks_{};

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
  gpulsmopt2_detail::Buffer<std::uint32_t> raw_keys_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RawPayload> raw_payloads_;
  gpulsmopt2_detail::Buffer<std::uint32_t> raw_offsets_;
  gpulsmopt2_detail::Buffer<std::uint64_t> raw_signatures_;
  gpulsmopt2_detail::Buffer<std::uint64_t> raw_epoch_signatures_;
  gpulsmopt2_detail::Buffer<std::uint32_t> publication_epoch_keys_a_,
      publication_epoch_keys_b_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RawAssignment>
      publication_epoch_assignments_a_, publication_epoch_assignments_b_;
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
  gpulsmopt2_detail::Buffer<std::uint32_t> foundation_overflow_flag_;
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
