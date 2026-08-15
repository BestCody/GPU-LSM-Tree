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
#include <thrust/iterator/transform_output_iterator.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
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
// Eight pivots divide a logical section into nine search regions.  Keeping the
// guide small matters because one guide is reserved for every section/level.
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
constexpr std::uint64_t kDescriptorOffsetMask =
    (std::uint64_t{1} << kDescriptorOffsetBits) - 1u;
constexpr std::uint64_t kInvalidOffset =
    std::numeric_limits<std::uint64_t>::max();
constexpr std::uint32_t kSectionOwnerMinimumReuse = 4u;
constexpr std::uint32_t kRangeThreadWork = 8u;
constexpr std::uint32_t kRangeSubgroupWork = 512u;
constexpr std::uint32_t kLookupWarpsPerBlock = 8u;
constexpr std::uint32_t kLookupHashSlots = 64u;
constexpr std::uint32_t kEmptyLookupKey = 1u << 16u;
constexpr std::uint32_t kAdmissionCtaGroupMaximum = 64u;
constexpr std::uint32_t kAdmissionCtaHashSlots = 128u;
static_assert((kAdmissionCtaHashSlots &
               (kAdmissionCtaHashSlots - 1u)) == 0u);
constexpr std::uint32_t kFoundationCompactionThreads = 256u;
constexpr std::uint32_t kFoundationItemsPerThread = 9u;
constexpr std::uint32_t kFoundationSectionCapacity =
    kFoundationCompactionThreads * kFoundationItemsPerThread;
constexpr std::uint32_t kFoundationAgeBits = 6u;
constexpr std::uint32_t kFoundationCells = 128u;
constexpr std::uint32_t kFoundationCellKeys = 512u;
constexpr std::uint32_t kBalancedMergeTarget =
    kFoundationSectionCapacity;
constexpr std::uint32_t kBalancedMergeMaximumQuotients = 512u;
constexpr std::uint32_t kPlanningTiles = 128u;
constexpr std::uint32_t kPlanningTileQuotients =
    kQuotients / kPlanningTiles;
constexpr std::uint32_t kMaximumMergeSources = kMaximumLevels + 1u;
// A quotient has at most 65 sorted sources and 65,536 distinct suffixes.
// This upper bound is sufficient to split even that theoretical maximum into
// on-chip jobs without imposing a fixed number of pieces on ordinary data.
constexpr std::uint32_t kBalancedHotRanges = 2048u;
constexpr std::uint32_t kBalancedMergeWorkspace =
    kBalancedMergeTarget + kMaximumLevels;
constexpr std::uint32_t kMergeSourceBits = 7u;

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
  // Two alternating banks.  A merge reserves one output slot per raw input
  // before duplicate removal, so each bank keeps a 2x raw-row bound.
  const std::size_t requested_banks = requested > even_maximum / 4u
      ? even_maximum : requested * 4u;
  const std::size_t capacity = std::max<std::size_t>(
      requested_banks, std::size_t{kQuotients} * 1024u);
  return std::min(even_maximum, (capacity + 1u) & ~std::size_t{1u});
}

inline std::size_t maximum_resident_merge_jobs(std::size_t maximum_raw_rows) {
  // The smallest possible safe target occurs when all 64 levels and the new
  // epoch participate.  Every whole section can require its own job even when
  // the total raw-row count divided by the target is smaller (for example,
  // 65,536 sections with 1,153 rows each cannot be paired).  One slot per
  // section plus the hot-section pieces is therefore the distribution-
  // independent section-preserving bound.
  constexpr std::size_t safe =
      kBalancedMergeTarget - (kMaximumMergeSources - 1u);
  return std::size_t{kQuotients} +
      (maximum_raw_rows + safe - 1u) / safe + 1u;
}

inline std::size_t adaptive_route_stride(std::size_t maximum_raw_rows) {
  // A normal section needs one route and only oversized sections add routes.
  // Size this from the maximum raw carry, rather than expected live rows.
  return std::size_t{kQuotients} +
      maximum_resident_merge_jobs(maximum_raw_rows);
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


struct Row {
  std::uint32_t value;
  std::uint16_t key;
  std::uint16_t flags;
};

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
  __host__ __device__ std::uint64_t offset() const {
    return bits & kDescriptorOffsetMask;
  }
  __host__ __device__ std::uint32_t count() const {
    return static_cast<std::uint32_t>(bits >> kDescriptorOffsetBits);
  }
};

static_assert(sizeof(Descriptor) == 8u);

// A quotient can intersect more than one adaptive location.  The first-level
// directory is indexed exactly like the old descriptor table; it names a
// short, key-ordered run of route slices.  Each slice is a packed, contiguous
// part of one adaptive location.
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

enum DeviceLevelFlags : std::uint32_t {
  kLevelKeepsTombstones = 1u << 0u,
  kLevelHasGuide = 1u << 1u,
  kLevelHasRank = 1u << 2u,
  kLevelIsFoundation = 1u << 3u,
};

struct DeviceLevelState {
  std::uint64_t logical_rows{};
  std::uint64_t data_begin{};
  std::uint64_t data_capacity{};
  std::uint32_t route_begin{};
  std::uint32_t route_count{};
  std::uint32_t storage_generation{};
  std::uint32_t guide_generation{};
  std::uint32_t flags{};
  std::uint32_t reserved{};
};

struct DeviceManifest {
  std::uint64_t occupied_level_mask{};
  std::uint32_t active_levels{};
  std::uint32_t foundation_level{kMaximumLevels};
  std::uint32_t generation{};
  std::uint32_t reserved{};
  DeviceLevelState levels[kMaximumLevels]{};
};

struct LevelStorageSpan {
  std::uint64_t begin{};
  std::uint64_t capacity{};
};

enum ResidentPublicationStatus : std::uint32_t {
  kPublicationSuccess = 0u,
  kPublicationJobOverflow = 1u << 0u,
  kPublicationRouteOverflow = 1u << 1u,
  kPublicationSliceOverflow = 1u << 2u,
  kPublicationOutputOverflow = 1u << 3u,
  kPublicationJobTooLarge = 1u << 4u,
  kPublicationCursorMismatch = 1u << 5u,
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
  std::uint32_t boundary_count{};
  std::uint32_t route_count{};
  std::uint32_t slice_count{};
  std::uint32_t status{};
  std::uint32_t direct_store{};
  std::uint32_t reserved0{};
  std::uint64_t output_begin{};
  std::uint64_t output_capacity{};
  std::uint64_t raw_reservation{};
  std::uint64_t survivor_count{};
};

struct BoundaryCursor {
  std::uint32_t logical_rank{};
  std::uint32_t route_ordinal{};
};

static_assert(sizeof(BoundaryCursor) == 8u);

struct DeviceManifestSnapshot {
  std::uint64_t occupied_level_mask{};
  std::uint32_t active_levels{};
  std::uint32_t foundation_level{kMaximumLevels};
  std::uint32_t generation{};
};

__device__ __forceinline__ DeviceManifestSnapshot load_active_manifest(
    const DeviceManifest *manifests,
    const std::uint32_t *active_manifest) {
  // Publication is an atomic release.  Readers only need one cached load;
  // contending on an atomic read per query would destroy the fast path.
  const std::uint32_t index = __ldg(active_manifest) & 1u;
  const DeviceManifest *manifest = manifests + index;
  return {manifest->occupied_level_mask, manifest->active_levels,
          manifest->foundation_level, manifest->generation};
}

__device__ __forceinline__ DeviceManifestSnapshot load_query_manifest(
    const std::uint64_t *query_occupied_level_mask) {
  const std::uint64_t occupied = __ldg(query_occupied_level_mask);
  const std::uint32_t active_levels = occupied
      ? 64u - static_cast<std::uint32_t>(__clzll(occupied)) : 0u;
  return {occupied, active_levels,
          active_levels ? active_levels - 1u : kMaximumLevels, 0u};
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

template <class T> struct ConditionalOutputReference {
  T *staging{};
  T *direct{};
  const std::uint32_t *direct_mode{};
  std::ptrdiff_t index{};

  template <class U>
  __device__ __forceinline__ const ConditionalOutputReference &operator=(
      U &&value) const {
    T *destination = __ldg(direct_mode) ? direct : staging;
    destination[index] = static_cast<T>(value);
    return *this;
  }
};

template <class T> class ConditionalOutputIterator {
 public:
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = ConditionalOutputReference<T>;
  using iterator_category = std::random_access_iterator_tag;

  __host__ __device__ ConditionalOutputIterator() {}
  __host__ __device__ ConditionalOutputIterator(
      T *staging, T *direct, const std::uint32_t *direct_mode,
      difference_type index = 0)
      : staging_(staging), direct_(direct), direct_mode_(direct_mode),
        index_(index) {}

  __host__ __device__ reference operator*() const {
    return {staging_, direct_, direct_mode_, index_};
  }
  __host__ __device__ reference operator[](difference_type offset) const {
    return {staging_, direct_, direct_mode_, index_ + offset};
  }
  __host__ __device__ ConditionalOutputIterator operator+(
      difference_type offset) const {
    return {staging_, direct_, direct_mode_, index_ + offset};
  }
  __host__ __device__ ConditionalOutputIterator &operator+=(
      difference_type offset) {
    index_ += offset;
    return *this;
  }
  __host__ __device__ ConditionalOutputIterator &operator++() {
    ++index_;
    return *this;
  }
  __host__ __device__ ConditionalOutputIterator operator++(int) {
    ConditionalOutputIterator copy = *this;
    ++index_;
    return copy;
  }
  __host__ __device__ difference_type operator-(
      const ConditionalOutputIterator &other) const {
    return index_ - other.index_;
  }
  __host__ __device__ bool operator==(
      const ConditionalOutputIterator &other) const {
    return index_ == other.index_;
  }
  __host__ __device__ bool operator!=(
      const ConditionalOutputIterator &other) const {
    return !(*this == other);
  }

 private:
  T *staging_{};
  T *direct_{};
  const std::uint32_t *direct_mode_{};
  difference_type index_{};
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
  std::uint32_t boundary_begin{};
  std::uint32_t existing_capacity{};
  std::uint32_t output_count{};
  std::uint32_t slice_begin{};
  std::uint32_t slice_count{};
  std::uint32_t route_ordinal{};
  std::uint16_t hot_piece{};
  std::uint16_t hot_pieces{};
};

static_assert(sizeof(BalancedMergeJob) == 64u);

// One exact, contiguous input interval owned by a destination job.  Slots are
// source-major and quotient-minor.  Empty slots remain present so the source
// and quotient are recoverable without storing expanded per-record metadata.
struct PullSlice {
  std::uint64_t offset;
  std::uint32_t count;
  std::uint16_t candidate_begin;
  std::uint16_t tag;
};

static_assert(sizeof(PullSlice) == 16u);

template <class T> class Buffer {
public:
  Buffer() = default;
  explicit Buffer(std::size_t count) { resize(count); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer &operator=(Buffer &&other) noexcept {
    if (this != &other) {
      if (pointer_ && owns_) cudaFree(pointer_);
      pointer_ = other.pointer_;
      count_ = other.count_;
      owns_ = other.owns_;
      other.pointer_ = nullptr;
      other.count_ = 0u;
      other.owns_ = true;
    }
    return *this;
  }
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
  VirtualBuffer() = default;
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
  const T *data() const {
    return reinterpret_cast<const T *>(
        static_cast<std::uintptr_t>(address_));
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

__global__ void iota_kernel(std::uint32_t *ids, std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) ids[i] = i;
}

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

// Search a section as one logical sorted list even when its rows are stored in
// several route extents.  The returned position is relative to the section,
// not to any physical extent.
__device__ __forceinline__ std::uint32_t logical_section_bound(
    std::uint32_t q, std::uint32_t level, std::uint32_t key, bool upper,
    const Row *arena, const RouteHeader *route_headers,
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
  const Row *rows = arena + slice.rows.offset();
  return preceding + (upper
      ? upper_bound_rows(rows, slice.rows.count(), key)
      : lower_bound_rows(rows, slice.rows.count(), key));
}

__device__ __forceinline__ Row logical_section_row(
    std::uint32_t q, std::uint32_t level, std::uint32_t position,
    const Row *arena, const RouteHeader *route_headers,
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
    std::uint32_t low, std::uint32_t high, const Row *arena,
    RouteHeader header, const RouteSlice *route_slices) {
  std::uint32_t count = 0u;
  for (std::uint32_t local = 0u; local < header.count; ++local) {
    const RouteSlice route = route_slices[header.begin + local];
    if (route.suffix_end <= low || route.suffix_begin > high) continue;
    const Row *rows = arena + route.rows.offset();
    const std::uint32_t begin = lower_bound_rows(rows, route.rows.count(), low);
    const std::uint32_t end =
        upper_bound_rows(rows, route.rows.count(), high);
    count += end - begin;
  }
  return count;
}

// The newer rows have already been resolved into one sorted list.  A worker
// group can therefore scan every physical base extent once and only test that
// compact newest-visible list, instead of re-searching every newer level for
// every candidate row.
template <class Aggregate>
__device__ __forceinline__ typename Aggregate::State
cooperative_sum_visible_route_runs(
    std::uint32_t low, std::uint32_t high,
    const Row *current, std::uint32_t current_count,
    const Row *arena, RouteHeader base_header,
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
    const Row *rows = arena + route.rows.offset();
    const std::uint32_t begin = lower_bound_rows(rows, route.rows.count(), low);
    const std::uint32_t end =
        upper_bound_rows(rows, route.rows.count(), high);
    const std::uint32_t count = end - begin;
    const std::uint32_t lane_begin =
        begin + (std::uint64_t{count} * group_lane) / group_size;
    const std::uint32_t lane_end =
        begin + (std::uint64_t{count} * (group_lane + 1u)) / group_size;
    if (lane_begin == lane_end) continue;

    // Give every worker one consecutive part of the stored run.  It locates
    // its first newer row once, then advances through both sorted sequences.
    // Thus a stored row is never followed by another search through levels or
    // through the already-resolved newest-visible view.
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
    const Row *arena, const Descriptor *descriptors,
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
      const Row *rows = arena + route.rows.offset();
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
        const Row *newer_rows = arena + newer_descriptor.offset();
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
    const std::uint32_t *task_count, const Row *arena,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices, const std::uint32_t *raw_keys,
    const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, std::uint32_t active_levels,
    std::uint32_t foundation_level,
    typename Aggregate::State *aggregate_partials,
    std::uint64_t occupied_levels = ~std::uint64_t{0},
    const std::uint64_t *query_occupied_level_mask = nullptr) {
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
  __shared__ std::uint32_t fragment_work_shared[kSectionTaskFragments];
  __shared__ RangeFragmentBounds
      fragment_bounds_shared[kSectionTaskFragments];
  __shared__ std::uint32_t section_base_mask_valid_shared;
  __shared__ std::uint32_t quotient_shared;
  __shared__ std::uint32_t fragment_begin_shared;
  __shared__ std::uint32_t fragment_end_shared;
  __shared__ std::uint32_t task_valid_shared;
  __shared__ std::uint32_t base_section_count_shared;
  __shared__ Descriptor foundation_descriptor_shared;
  __shared__ Descriptor section_descriptors[kMaximumLevels];

  if (query_occupied_level_mask) {
    const DeviceManifestSnapshot manifest =
        load_query_manifest(query_occupied_level_mask);
    active_levels = manifest.active_levels;
    foundation_level = manifest.foundation_level;
    occupied_levels = manifest.occupied_level_mask;
  }

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
      const Row *source = arena + descriptor.offset();
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

  // Count the rows of every fragment in parallel.  Scheduling is based on
  // actual input rows, never on key-space width.  A task can use one fast
  // uniform schedule when all fragments fall in the same worker class; mixed
  // tasks use the per-fragment adaptive schedule below.
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
      const Row *rows = arena + descriptor.offset();
      bounds.base_begin = lower_bound_rows(
          rows, descriptor.count(), fragment.low_suffix);
      bounds.base_end = upper_bound_rows(
          rows, descriptor.count(), fragment.high_suffix);
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

  // Split storage is scanned as one logical run.  Each fragment chooses its
  // cooperation width from the number of rows it will actually examine.  The
  // same adaptive schedule handles an uneven task in an unsplit section.
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
  const Row *foundation_rows =
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
    const Row *arena,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices, const std::uint32_t *raw_keys,
    const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, std::uint32_t active_levels,
    std::uint32_t foundation_level,
    typename Aggregate::State *aggregate_partials,
    std::uint64_t occupied_levels = ~std::uint64_t{0},
    const std::uint64_t *query_occupied_level_mask = nullptr) {
  constexpr std::uint32_t kWarps = 4u;
  constexpr std::uint32_t kUpdateCapacity = 128u;
  union WarpScratch {
    Row merged[kUpdateCapacity];
    TaggedRow tagged[kUpdateCapacity];
  };
  __shared__ Row current_shared[kWarps][kUpdateCapacity];
  __shared__ WarpScratch scratch[kWarps];
  if (query_occupied_level_mask) {
    const DeviceManifestSnapshot manifest =
        load_query_manifest(query_occupied_level_mask);
    active_levels = manifest.active_levels;
    foundation_level = manifest.foundation_level;
    occupied_levels = manifest.occupied_level_mask;
  }
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
    const Row *rows = arena + descriptor.offset();
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
    const Row *older = rows + older_begin;
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

  // Random input usually has almost no repeated quotient inside a CTA.  In
  // that case a CTA hash only adds synchronization and probing, so preserve
  // the same histogram/scatter algorithm but stop aggregation at the warp.
  // Grouped or skewed input uses the CTA hash and emits one global update per
  // quotient for the whole CTA.
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
  if (q >= kQuotients || plan->status || plan->direct_store) return;
  const DeviceManifestSnapshot manifest = load_active_manifest(
      manifests, active_manifest);
  std::uint64_t count =
      current_offsets[q + 1u] - current_offsets[q];
  for (std::uint32_t level = 0u;
       level <= plan->source_level_limit; ++level)
    if (level_is_occupied(manifest.occupied_level_mask, level))
      count += descriptors[descriptor_index(q, level)].count();
  raw_counts[q] = count;
}




// --------------------------------------------------------------------------
// GPU-resident plan-and-publish pipeline.
// --------------------------------------------------------------------------

__global__ void choose_resident_reduce_output_kernel(
    const DeviceManifest *manifests, const std::uint32_t *active_manifest,
    std::uint32_t *direct_mode) {
  if (blockIdx.x || threadIdx.x) return;
  const std::uint32_t active = __ldg(active_manifest) & 1u;
  *direct_mode = (manifests[active].occupied_level_mask & 1u) == 0u;
}

__global__ void choose_resident_publication_path_kernel(
    cudaGraphConditionalHandle conditional,
    const std::uint32_t *selected_count,
    const DeviceManifest *manifests, const std::uint32_t *active_manifest,
    const LevelStorageSpan *level_spans,
    std::uint64_t foundation_bank_capacity,
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
  next.direct_store = destination == 0u;
  next.source_level_limit = destination ? destination - 1u : 0u;
  next.source_count = destination ? destination + 1u : 1u;
  next.destination_is_foundation =
      destination < kMaximumLevels &&
      (manifest->foundation_level == kMaximumLevels ||
       destination > manifest->foundation_level);
  next.keep_tombstones = !next.destination_is_foundation;
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
  cudaGraphSetConditional(conditional,
                          next.direct_store ? 0u : 1u);
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


__global__ void build_direct_level_directory_kernel(
    const std::uint32_t *section_offsets,
    const ResidentPublicationPlan *plan, std::uint32_t route_stride,
    Descriptor *descriptors, RouteHeader *route_headers,
    RouteSlice *route_slices, std::uint32_t *route_logical_begins,
    std::uint16_t *route_quotients,
    std::uint32_t *level_q_logical_offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients || plan->status || !plan->direct_store) return;
  const std::uint32_t level = plan->destination_level;
  level_q_logical_offsets[
      std::size_t{level} * (kQuotients + 1u) + q] = section_offsets[q];
  if (q == kQuotients) return;
  const std::uint32_t begin = section_offsets[q];
  const std::uint32_t end = section_offsets[q + 1u];
  const Descriptor descriptor = begin == end
      ? Descriptor{}
      : Descriptor::make(plan->output_begin + begin, end - begin);
  descriptors[descriptor_index(q, level)] = descriptor;
  const std::uint32_t route = level * route_stride + q;
  route_headers[descriptor_index(q, level)] =
      {route, descriptor.count() ? 1u : 0u};
  route_slices[route] = {descriptor, 0u, 1u << 16u};
  route_logical_begins[route] = begin;
  route_quotients[route] = static_cast<std::uint16_t>(q);
}

__device__ __forceinline__ std::uint32_t planning_tile_jobs(
    const std::uint64_t *raw_counts, std::uint32_t tile,
    std::uint32_t safe_target) {
  const std::uint32_t first = tile * kPlanningTileQuotients;
  const std::uint32_t last = first + kPlanningTileQuotients;
  std::uint32_t jobs = 0u, begin = first;
  std::uint64_t work = 0u;
  for (std::uint32_t q = first; q < last; ++q) {
    const std::uint64_t count = raw_counts[q];
    if (count > kBalancedMergeTarget) {
      if (q != begin) ++jobs;
      jobs += static_cast<std::uint32_t>(
          (count + safe_target - 1u) / safe_target);
      begin = q + 1u;
      work = 0u;
    } else if (q != begin && work + count > kBalancedMergeTarget) {
      ++jobs;
      begin = q;
      work = count;
    } else {
      work += count;
    }
  }
  if (begin != last) ++jobs;
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
    if (plan->status || plan->direct_store) {
      tile_job_counts[tile] = 0u;
    } else {
      const std::uint32_t safe = kBalancedMergeTarget -
          (plan->source_count - 1u);
      tile_job_counts[tile] = planning_tile_jobs(counts, 0u, safe);
    }
    if (tile + 1u == kPlanningTiles)
      tile_job_counts[kPlanningTiles] = 0u;
  }
}

__device__ __forceinline__ void emit_resident_job(
    BalancedMergeJob *jobs, std::uint64_t *boundary_keys,
    std::uint64_t *job_raw_reservations,
    std::uint32_t tile, std::uint32_t global_index,
    std::uint64_t key_begin, std::uint64_t key_end,
    std::uint32_t q_begin, std::uint32_t q_end,
    std::uint64_t raw_count, std::uint16_t hot_piece,
    std::uint16_t hot_pieces) {
  BalancedMergeJob job{};
  job.key_begin = key_begin;
  job.key_end = key_end;
  job.quotient_begin = q_begin;
  job.quotient_end = q_end;
  job.boundary_begin = global_index + tile;
  job.route_ordinal = hot_pieces ? hot_piece : 0u;
  job.hot_piece = hot_piece;
  job.hot_pieces = hot_pieces;
  jobs[global_index] = job;
  boundary_keys[job.boundary_begin] = key_begin;
  boundary_keys[job.boundary_begin + 1u] = key_end;
  job_raw_reservations[global_index] = raw_count;
}

__global__ void emit_resident_planning_jobs_kernel(
    const std::uint64_t *raw_counts,
    const std::uint32_t *tile_job_offsets,
    ResidentPublicationPlan *plan, std::uint32_t maximum_jobs,
    BalancedMergeJob *jobs, std::uint64_t *boundary_keys,
    std::uint64_t *job_raw_reservations) {
  __shared__ std::uint64_t counts[kPlanningTileQuotients];
  const std::uint32_t tile = blockIdx.x;
  if (tile >= kPlanningTiles) return;
  const std::uint32_t first = tile * kPlanningTileQuotients;
  counts[threadIdx.x] = raw_counts[first + threadIdx.x];
  counts[threadIdx.x + blockDim.x] =
      raw_counts[first + threadIdx.x + blockDim.x];
  __syncthreads();
  if (threadIdx.x != 0u || plan->status || plan->direct_store) return;

  const std::uint32_t total_jobs = tile_job_offsets[kPlanningTiles];
  if (tile == 0u) {
    plan->job_count = total_jobs;
    plan->boundary_count = total_jobs + kPlanningTiles;
    if (total_jobs > maximum_jobs)
      atomicOr(&plan->status, kPublicationJobOverflow);
  }
  if (total_jobs > maximum_jobs) return;

  const std::uint32_t safe = kBalancedMergeTarget -
      (plan->source_count - 1u);
  std::uint32_t global = tile_job_offsets[tile];
  std::uint32_t begin = first;
  std::uint64_t work = 0u;
  for (std::uint32_t q = first; q < first + kPlanningTileQuotients; ++q) {
    const std::uint64_t count = counts[q - first];
    if (count > kBalancedMergeTarget) {
      if (q != begin) {
        emit_resident_job(
            jobs, boundary_keys, job_raw_reservations, tile, global++,
            std::uint64_t{begin} << 16u, std::uint64_t{q} << 16u,
            begin, q, work, 0u, 0u);
      }
      const std::uint32_t pieces = static_cast<std::uint32_t>(
          (count + safe - 1u) / safe);
      for (std::uint32_t piece = 0u; piece < pieces; ++piece) {
        emit_resident_job(
            jobs, boundary_keys, job_raw_reservations, tile, global++,
            std::uint64_t{q} << 16u, std::uint64_t{q + 1u} << 16u,
            q, q + 1u, 0u, static_cast<std::uint16_t>(piece),
            static_cast<std::uint16_t>(pieces));
      }
      begin = q + 1u;
      work = 0u;
    } else if (q != begin && work + count > kBalancedMergeTarget) {
      emit_resident_job(
          jobs, boundary_keys, job_raw_reservations, tile, global++,
          std::uint64_t{begin} << 16u, std::uint64_t{q} << 16u,
          begin, q, work, 0u, 0u);
      begin = q;
      work = count;
    } else {
      work += count;
    }
  }
  const std::uint32_t last = first + kPlanningTileQuotients;
  if (begin != last)
    emit_resident_job(
        jobs, boundary_keys, job_raw_reservations, tile, global,
        std::uint64_t{begin} << 16u, std::uint64_t{last} << 16u,
        begin, last, work, 0u, 0u);
}

__device__ __forceinline__ std::uint32_t
balanced_merge_prefix_count_warp(
    std::uint32_t q, std::uint32_t suffix,
    const Row *current_rows, const std::uint32_t *current_offsets,
    const Row *arena, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    std::uint32_t source_level_limit) {
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
    const Row *arena, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    std::uint32_t source_level_limit) {
  if (!target) return 0u;
  std::uint32_t low = 1u, high = 1u << 16u;
  while (low < high) {
    const std::uint32_t middle = (low + high) >> 1u;
    if (balanced_merge_prefix_count_warp(
            q, middle, current_rows, current_offsets, arena,
            route_headers, route_slices, route_logical_begins,
            level_q_logical_offsets, source_level_limit) < target)
      low = middle + 1u;
    else
      high = middle;
  }
  return low;
}

__global__ void resolve_resident_job_boundaries_kernel(
    BalancedMergeJob *jobs, std::uint64_t *boundary_keys,
    std::uint64_t *job_raw_reservations,
    const ResidentPublicationPlan *plan,
    const Row *current_rows, const std::uint32_t *current_offsets,
    const Row *arena, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets) {
  const std::uint32_t lane = threadIdx.x & 31u;
  constexpr unsigned mask = 0xffffffffu;
  for (std::uint32_t job_index = blockIdx.x;
       job_index < plan->job_count && !plan->status;
       job_index += gridDim.x) {
  std::uint32_t hot_pieces = lane == 0u
      ? jobs[job_index].hot_pieces : 0u;
  hot_pieces = __shfl_sync(mask, hot_pieces, 0u);
  if (!hot_pieces) {
    if (lane == 0u &&
        job_raw_reservations[job_index] > kBalancedMergeTarget)
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
          level_q_logical_offsets, plan->source_level_limit));
  const std::uint32_t low_target = static_cast<std::uint32_t>(
      (std::uint64_t{raw} * job.hot_piece + job.hot_pieces - 1u) /
      job.hot_pieces);
  const std::uint32_t high_target = static_cast<std::uint32_t>(
      (std::uint64_t{raw} * (job.hot_piece + 1u) +
       job.hot_pieces - 1u) / job.hot_pieces);
  const std::uint32_t low = resident_hot_boundary_warp(
      q, low_target, current_rows, current_offsets, arena,
      route_headers, route_slices, route_logical_begins,
      level_q_logical_offsets, plan->source_level_limit);
  const std::uint32_t high = job.hot_piece + 1u == job.hot_pieces
      ? (1u << 16u)
      : resident_hot_boundary_warp(
            q, high_target, current_rows, current_offsets, arena,
            route_headers, route_slices, route_logical_begins,
            level_q_logical_offsets, plan->source_level_limit);
  const std::uint32_t exact = balanced_merge_prefix_count_warp(
      q, high, current_rows, current_offsets, arena, route_headers,
      route_slices, route_logical_begins, level_q_logical_offsets,
      plan->source_level_limit) -
      balanced_merge_prefix_count_warp(
          q, low, current_rows, current_offsets, arena, route_headers,
          route_slices, route_logical_begins, level_q_logical_offsets,
          plan->source_level_limit);
  if (lane == 0u) {
    job.key_begin = (std::uint64_t{q} << 16u) + low;
    job.key_end = (std::uint64_t{q} << 16u) + high;
    job_raw_reservations[job_index] = exact;
    jobs[job_index] = job;
    boundary_keys[job.boundary_begin] = job.key_begin;
    boundary_keys[job.boundary_begin + 1u] = job.key_end;
    if (exact > kBalancedMergeTarget)
      atomicOr(const_cast<std::uint32_t *>(&plan->status),
               kPublicationJobTooLarge);
  }
  }
}

__global__ void count_resident_route_slots_kernel(
    const std::uint64_t *raw_counts,
    const ResidentPublicationPlan *plan,
    std::uint32_t *route_counts) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  if (q == kQuotients || plan->status || plan->direct_store) {
    route_counts[q] = 0u;
    return;
  }
  const std::uint64_t raw = raw_counts[q];
  if (!raw) {
    route_counts[q] = 0u;
  } else if (raw <= kBalancedMergeTarget) {
    route_counts[q] = 1u;
  } else {
    const std::uint32_t safe = kBalancedMergeTarget -
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
  if (q >= kQuotients || plan->status || plan->direct_store) return;
  next_headers[q] = {
      plan->destination_level * route_stride + route_offsets[q],
      route_counts[q]};
}

__global__ void construct_boundary_cursors_kernel(
    const std::uint64_t *boundary_keys,
    const ResidentPublicationPlan *plan,
    const std::uint32_t *current_keys,
    const std::uint32_t *current_offsets,
    const Row *arena, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    std::uint32_t route_stride, BoundaryCursor *cursors) {
  const std::uint64_t cursor_count =
      std::uint64_t{plan->boundary_count} * plan->source_count;
  if (plan->status || plan->direct_store) return;
  for (std::uint64_t cursor_index =
           std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
       cursor_index < cursor_count;
       cursor_index += std::uint64_t{gridDim.x} * blockDim.x) {
  const std::uint32_t boundary = static_cast<std::uint32_t>(
      cursor_index / plan->source_count);
  const std::uint32_t source = static_cast<std::uint32_t>(
      cursor_index - std::uint64_t{boundary} * plan->source_count);
  const std::uint64_t key = boundary_keys[boundary];
  BoundaryCursor cursor{};
  if (key == (std::uint64_t{1} << 32u)) {
    if (source == 0u) {
      cursor.logical_rank = current_offsets[kQuotients];
      cursor.route_ordinal = kQuotients;
    } else {
      const std::uint32_t level = source - 1u;
      cursor.logical_rank = level_q_logical_offsets[
          std::size_t{level} * (kQuotients + 1u) + kQuotients];
      cursor.route_ordinal = level * route_stride + route_stride;
    }
  } else {
    const std::uint32_t q = static_cast<std::uint32_t>(key >> 16u);
    const std::uint32_t suffix = static_cast<std::uint32_t>(key & 0xffffu);
    if (source == 0u) {
      const std::uint32_t begin = current_offsets[q];
      const std::uint32_t count = current_offsets[q + 1u] - begin;
      cursor.logical_rank = begin + lower_bound_full_keys(
          current_keys + begin, count, static_cast<std::uint32_t>(key));
      cursor.route_ordinal = q;
    } else {
      const std::uint32_t level = source - 1u;
      const RouteHeader header =
          route_headers[descriptor_index(q, level)];
      std::uint32_t low = 0u, high = header.count;
      while (low < high) {
        const std::uint32_t middle = (low + high) >> 1u;
        if (route_slices[header.begin + middle].suffix_end <= suffix)
          low = middle + 1u;
        else
          high = middle;
      }
      cursor.route_ordinal = header.begin + low;
      if (low == header.count) {
        cursor.logical_rank = level_q_logical_offsets[
            std::size_t{level} * (kQuotients + 1u) + q + 1u];
      } else {
        const std::uint32_t route = header.begin + low;
        const RouteSlice slice = route_slices[route];
        cursor.logical_rank = route_logical_begins[route] +
            lower_bound_rows(arena + slice.rows.offset(),
                             slice.rows.count(), suffix);
      }
    }
  }
  cursors[std::size_t{boundary} * kMaximumMergeSources + source] = cursor;
  }
}

__global__ void count_cursor_pull_slices_kernel(
    BalancedMergeJob *jobs, const ResidentPublicationPlan *plan,
    const BoundaryCursor *cursors,
    const std::uint32_t *current_offsets,
    const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    std::uint32_t route_stride,
    std::uint32_t *source_slice_offsets,
    std::uint16_t *source_candidate_offsets,
    std::uint32_t *slice_reservations) {
  __shared__ std::uint32_t slice_counts[kMaximumMergeSources];
  __shared__ std::uint32_t row_counts[kMaximumMergeSources];
  if (plan->status || plan->direct_store) return;
  for (std::uint32_t job_index = blockIdx.x;
       job_index < plan->job_count; job_index += gridDim.x) {
  const BalancedMergeJob job = jobs[job_index];
  const std::uint32_t source = threadIdx.x;
  if (source < plan->source_count) {
    const BoundaryCursor left = cursors[
        std::size_t{job.boundary_begin} * kMaximumMergeSources + source];
    const BoundaryCursor right = cursors[
        std::size_t{job.boundary_begin + 1u} * kMaximumMergeSources + source];
    row_counts[source] = right.logical_rank - left.logical_rank;
    std::uint32_t slices = 0u;
    if (source == 0u) {
      for (std::uint32_t q = job.quotient_begin;
           q < job.quotient_end; ++q) {
        const std::uint32_t begin = max(left.logical_rank,
                                        current_offsets[q]);
        const std::uint32_t end = min(right.logical_rank,
                                      current_offsets[q + 1u]);
        slices += begin < end;
      }
    } else {
      const std::uint32_t level = source - 1u;
      const RouteHeader last_header =
          route_headers[descriptor_index(kQuotients - 1u, level)];
      const std::uint32_t route_limit =
          last_header.begin + last_header.count;
      for (std::uint32_t route = left.route_ordinal;
           route <= right.route_ordinal && route < route_limit; ++route) {
        const RouteSlice rows = route_slices[route];
        const std::uint32_t logical = route_logical_begins[route];
        const std::uint32_t begin = max(left.logical_rank, logical);
        const std::uint32_t end = min(
            right.logical_rank, logical + rows.rows.count());
        slices += begin < end;
      }
    }
    slice_counts[source] = slices;
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    const std::size_t base =
        std::size_t{job_index} * (kMaximumMergeSources + 1u);
    std::uint32_t slices = 0u, rows = 0u;
    for (std::uint32_t source_index = 0u;
         source_index < plan->source_count; ++source_index) {
      source_slice_offsets[base + source_index] = slices;
      source_candidate_offsets[base + source_index] =
          static_cast<std::uint16_t>(rows);
      slices += slice_counts[source_index];
      rows += row_counts[source_index];
    }
    source_slice_offsets[base + plan->source_count] = slices;
    source_candidate_offsets[base + plan->source_count] =
        static_cast<std::uint16_t>(rows);
    jobs[job_index].slice_count = slices;
    slice_reservations[job_index] = slices + 1u;
    if (rows > kBalancedMergeTarget)
      atomicOr(const_cast<std::uint32_t *>(&plan->status),
               kPublicationJobTooLarge);
  }
  __syncthreads();
  }
}

__global__ void materialize_cursor_pull_slices_kernel(
    BalancedMergeJob *jobs, const ResidentPublicationPlan *plan,
    const BoundaryCursor *cursors,
    const std::uint32_t *current_offsets,
    const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint16_t *route_quotients,
    std::uint32_t route_stride,
    const std::uint32_t *source_slice_offsets,
    const std::uint16_t *source_candidate_offsets,
    const std::uint32_t *slice_offsets, PullSlice *slices) {
  if (plan->status || plan->direct_store) return;
  for (std::uint32_t job_index = blockIdx.x;
       job_index < plan->job_count; job_index += gridDim.x) {
  BalancedMergeJob job = jobs[job_index];
  const std::size_t source_base =
      std::size_t{job_index} * (kMaximumMergeSources + 1u);
  if (threadIdx.x == 0u) {
    job.slice_begin = slice_offsets[job_index];
    jobs[job_index].slice_begin = job.slice_begin;
  }
  __syncthreads();
  job.slice_begin = slice_offsets[job_index];
  const std::uint32_t source = threadIdx.x;
  if (source < plan->source_count) {
    const BoundaryCursor left = cursors[
        std::size_t{job.boundary_begin} * kMaximumMergeSources + source];
    const BoundaryCursor right = cursors[
        std::size_t{job.boundary_begin + 1u} * kMaximumMergeSources + source];
    std::uint32_t slot = job.slice_begin +
        source_slice_offsets[source_base + source];
    std::uint32_t candidate =
        source_candidate_offsets[source_base + source];
    if (source == 0u) {
      for (std::uint32_t q = job.quotient_begin;
           q < job.quotient_end; ++q) {
        const std::uint32_t begin = max(left.logical_rank,
                                        current_offsets[q]);
        const std::uint32_t end = min(right.logical_rank,
                                      current_offsets[q + 1u]);
        if (begin == end) continue;
        slices[slot++] = {
            begin, end - begin, static_cast<std::uint16_t>(candidate),
            static_cast<std::uint16_t>(q - job.quotient_begin)};
        candidate += end - begin;
      }
    } else {
      const std::uint32_t level = source - 1u;
      const RouteHeader last_header =
          route_headers[descriptor_index(kQuotients - 1u, level)];
      const std::uint32_t route_limit =
          last_header.begin + last_header.count;
      for (std::uint32_t route = left.route_ordinal;
           route <= right.route_ordinal && route < route_limit; ++route) {
        const RouteSlice input = route_slices[route];
        const std::uint32_t logical = route_logical_begins[route];
        const std::uint32_t begin = max(left.logical_rank, logical);
        const std::uint32_t end = min(
            right.logical_rank, logical + input.rows.count());
        if (begin == end) continue;
        const std::uint32_t q = route_quotients[route];
        const std::uint16_t tag = static_cast<std::uint16_t>(
            (source << 9u) | (q - job.quotient_begin));
        slices[slot++] = {
            input.rows.offset() + begin - logical, end - begin,
            static_cast<std::uint16_t>(candidate), tag};
        candidate += end - begin;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    const std::uint32_t total_rows =
        source_candidate_offsets[source_base + plan->source_count];
    slices[job.slice_begin + job.slice_count] = {
        0u, 0u, static_cast<std::uint16_t>(total_rows), 0u};
    if (total_rows != job.existing_capacity)
      atomicOr(const_cast<std::uint32_t *>(&plan->status),
               kPublicationCursorMismatch);
  }
  __syncthreads();
  }
}

__global__ void validate_resident_plan_kernel(
    ResidentPublicationPlan *plan,
    const std::uint32_t *tile_job_offsets,
    const std::uint64_t *job_output_offsets,
    const std::uint32_t *route_offsets,
    const std::uint32_t *slice_offsets,
    std::uint32_t maximum_jobs, std::uint32_t maximum_boundaries,
    std::uint32_t route_capacity, std::uint32_t slice_capacity) {
  if (blockIdx.x || threadIdx.x || plan->direct_store) return;
  plan->job_count = tile_job_offsets[kPlanningTiles];
  plan->boundary_count = plan->job_count + kPlanningTiles;
  plan->raw_reservation = job_output_offsets[maximum_jobs];
  plan->route_count = route_offsets[kQuotients];
  plan->slice_count = slice_offsets[maximum_jobs];
  if (plan->job_count > maximum_jobs ||
      plan->boundary_count > maximum_boundaries)
    plan->status |= kPublicationJobOverflow;
  if (plan->route_count > route_capacity)
    plan->status |= kPublicationRouteOverflow;
  if (plan->slice_count > slice_capacity)
    plan->status |= kPublicationSliceOverflow;
  if (plan->raw_reservation > plan->output_capacity)
    plan->status |= kPublicationOutputOverflow;
}

__global__ void assign_resident_output_offsets_kernel(
    BalancedMergeJob *jobs, const std::uint64_t *job_output_offsets,
    const std::uint64_t *job_raw_reservations,
    const ResidentPublicationPlan *plan) {
  if (plan->status || plan->direct_store) return;
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
  if (q > kQuotients || plan->status || plan->direct_store) return;
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
                         : Descriptor::make(0u, total);
}

__global__ void build_resident_query_metadata_kernel(
    const ResidentPublicationPlan *plan, const Row *arena,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    std::uint16_t *local_rank, std::uint16_t *level_guides) {
  const std::uint32_t q = blockIdx.x;
  if (q >= kQuotients || plan->status) return;
  const std::uint32_t level = plan->destination_level;
  const RouteHeader header = route_headers[descriptor_index(q, level)];
  const Descriptor descriptor = descriptors[descriptor_index(q, level)];
  if (plan->destination_is_foundation) {
    for (std::uint32_t cell = threadIdx.x; cell < 128u;
         cell += blockDim.x) {
      const std::uint32_t target = cell << 9u;
      const std::uint32_t position =
          header.count && descriptor.count() <= (1u << 16u)
              ? logical_section_bound(
                    q, level, target, false, arena, route_headers,
                    route_slices, route_logical_begins,
                    level_q_logical_offsets)
              : 0u;
      local_rank[std::size_t{q} * 128u + cell] =
          static_cast<std::uint16_t>(position);
    }
  } else if (header.count && descriptor.count() >= kGuideRegions) {
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

__global__ void publish_resident_manifest_kernel(
    ResidentPublicationPlan *plan, DeviceManifest *manifests,
    std::uint32_t *active_manifest,
    std::uint64_t *query_occupied_level_mask,
    std::uint32_t route_stride) {
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
    next->reserved = current->reserved;
    DeviceLevelState state{};
    state.logical_rows = plan->survivor_count;
    state.data_begin = plan->output_begin;
    state.data_capacity = plan->output_capacity;
    state.route_begin = destination * route_stride;
    state.route_count = plan->direct_store ? kQuotients : plan->route_count;
    state.storage_generation = plan->output_generation;
    state.guide_generation = current->generation + 1u;
    state.flags = plan->keep_tombstones ? kLevelKeepsTombstones : 0u;
    state.flags |= plan->destination_is_foundation
        ? (kLevelHasRank | kLevelIsFoundation) : kLevelHasGuide;
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

__global__ void set_direct_survivor_count_kernel(
    ResidentPublicationPlan *plan) {
  if (blockIdx.x || threadIdx.x || plan->status || !plan->direct_store)
    return;
  plan->survivor_count = plan->selected_count;
}

__global__ void set_merged_survivor_count_kernel(
    ResidentPublicationPlan *plan,
    const std::uint32_t *section_logical_offsets) {
  if (blockIdx.x || threadIdx.x || plan->status || plan->direct_store)
    return;
  plan->survivor_count = section_logical_offsets[kQuotients];
}

__global__ void fold_resident_merge_status_kernel(
    ResidentPublicationPlan *plan, const std::uint32_t *overflow_flag) {
  if (blockIdx.x || threadIdx.x || plan->direct_store) return;
  if (*overflow_flag) plan->status |= kPublicationOutputOverflow;
}

__global__ void initialize_device_manifest_kernel(
    DeviceManifest *manifests, std::uint32_t *active_manifest,
    std::uint64_t *query_occupied_level_mask,
    std::uint32_t level, std::uint32_t count,
    std::uint64_t data_begin, std::uint64_t data_capacity,
    std::uint32_t route_stride, std::uint32_t storage_generation) {
  if (blockIdx.x || threadIdx.x) return;
  DeviceManifest manifest{};
  if (count) {
    manifest.occupied_level_mask = std::uint64_t{1} << level;
    manifest.active_levels = level + 1u;
    manifest.foundation_level = level;
    manifest.generation = 1u;
    DeviceLevelState state{};
    state.logical_rows = count;
    state.data_begin = data_begin;
    state.data_capacity = data_capacity;
    state.route_begin = level * route_stride;
    state.route_count = kQuotients;
    state.storage_generation = storage_generation;
    state.guide_generation = 1u;
    state.flags = kLevelHasRank | kLevelIsFoundation;
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





__device__ __forceinline__ std::uint32_t pull_candidate_order_key(
    const Row &row) {
  const std::uint32_t local_q = row.flags >> kMergeSourceBits;
  const std::uint32_t age =
      row.flags & ((1u << kMergeSourceBits) - 1u);
  return (local_q << (16u + kMergeSourceBits)) |
      (std::uint32_t{row.key} << kMergeSourceBits) | age;
}

__device__ __forceinline__ bool pull_index_less(
    std::uint16_t left, std::uint16_t right, const Row *candidates) {
  return pull_candidate_order_key(candidates[left]) <
      pull_candidate_order_key(candidates[right]);
}

__device__ __forceinline__ std::uint32_t pull_merge_partition(
    const std::uint16_t *left, std::uint32_t left_count,
    const std::uint16_t *right, std::uint32_t right_count,
    std::uint32_t diagonal, const Row *candidates) {
  std::uint32_t low = diagonal > right_count
      ? diagonal - right_count : 0u;
  std::uint32_t high = min(diagonal, left_count);
  while (low <= high) {
    const std::uint32_t left_index = (low + high) >> 1u;
    const std::uint32_t right_index = diagonal - left_index;
    if (left_index && right_index < right_count &&
        pull_index_less(right[right_index], left[left_index - 1u],
                        candidates)) {
      high = left_index - 1u;
    } else if (right_index && left_index < left_count &&
               pull_index_less(left[left_index], right[right_index - 1u],
                               candidates)) {
      low = left_index + 1u;
    } else {
      return left_index;
    }
  }
  return low;
}


__global__ void compact_balanced_merge_jobs_kernel(
    BalancedMergeJob *jobs, const ResidentPublicationPlan *resident_plan,
    const PullSlice *pull_slices,
    const Row *current_rows, Row *arena,
    const RouteHeader *next_route_headers, RouteSlice *route_slices,
    std::uint32_t *section_output_counts, std::uint32_t *overflow_flag) {
  constexpr std::uint32_t kThreads = kFoundationCompactionThreads;
  constexpr std::uint32_t kItemsPerThread =
      (kBalancedMergeTarget + kThreads - 1u) / kThreads;
  using BlockScan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage block_scan_storage;
  __shared__ Row candidates[kBalancedMergeTarget];
  // The input intervals are prepared once in global job metadata.  These are
  // therefore used only as the two ping-pong merge-index planes.
  __shared__ std::uint16_t merge_indices_a[kBalancedMergeTarget + 1u];
  __shared__ std::uint16_t merge_indices_b[kBalancedMergeTarget + 1u];
  __shared__ std::uint32_t tombstone_words[
      (kBalancedMergeTarget + 31u) / 32u];
  __shared__ std::uint16_t quotient_offsets[
      kBalancedMergeMaximumQuotients + 1u];
  __shared__ std::uint32_t quotient_output_counts[
      kBalancedMergeMaximumQuotients];
  __shared__ std::uint32_t quotient_output_offsets[
      kBalancedMergeMaximumQuotients];
  __shared__ std::uint64_t quotient_output_bits[
      kBalancedMergeMaximumQuotients];
  __shared__ std::uint32_t source_offsets[kMaximumLevels + 2u];
  __shared__ std::uint16_t run_offsets[kMaximumLevels + 1u];
  __shared__ std::uint16_t run_lengths[kMaximumLevels + 1u];
  __shared__ std::uint16_t run_sources[kMaximumLevels + 1u];
  __shared__ std::uint32_t slice_count_shared;
  __shared__ std::uint32_t run_count_shared;
  __shared__ std::uint32_t small_count_shared;
  __shared__ std::uint32_t largest_source_shared;
  __shared__ std::uint32_t largest_count_shared;

  if (resident_plan->status || resident_plan->direct_store) return;
  for (std::uint32_t job_index = blockIdx.x;
       job_index < resident_plan->job_count; job_index += gridDim.x) {
  const std::uint32_t source_level_limit =
      resident_plan->source_level_limit;
  const bool keep_tombstones = resident_plan->keep_tombstones != 0u;
  const unsigned long long output_limit =
      resident_plan->output_begin + resident_plan->output_capacity;
  const BalancedMergeJob job = jobs[job_index];
  const std::uint32_t quotient_count =
      job.quotient_end - job.quotient_begin;
  const std::uint32_t sources = source_level_limit + 2u;
  if (threadIdx.x == 0u) {
    slice_count_shared = job.slice_count;
    std::uint32_t slice = 0u;
    for (std::uint32_t source = 0u; source < sources; ++source) {
      source_offsets[source] = pull_slices[
          job.slice_begin + slice].candidate_begin;
      while (slice < job.slice_count &&
             (pull_slices[job.slice_begin + slice].tag >> 9u) == source)
        ++slice;
    }
    source_offsets[sources] = pull_slices[
        job.slice_begin + job.slice_count].candidate_begin;
    quotient_offsets[0u] = 0u;
    for (std::uint32_t local = 0u; local < quotient_count; ++local)
      quotient_offsets[local + 1u] = 0u;
    for (std::uint32_t index = 0u; index < job.slice_count; ++index) {
      const PullSlice slice_row = pull_slices[job.slice_begin + index];
      quotient_offsets[(slice_row.tag & 0x1ffu) + 1u] +=
          static_cast<std::uint16_t>(slice_row.count);
    }
    for (std::uint32_t local = 0u; local < quotient_count; ++local)
      quotient_offsets[local + 1u] += quotient_offsets[local];

    std::uint32_t largest_source = 0u;
    std::uint32_t largest_count = 0u;
    for (std::uint32_t source = 0u; source < sources; ++source) {
      const std::uint32_t count =
          source_offsets[source + 1u] - source_offsets[source];
      if (count > largest_count) {
        largest_count = count;
        largest_source = source;
      }
    }
    largest_source_shared = largest_source;
    largest_count_shared = largest_count;
  }
  for (std::uint32_t local = threadIdx.x; local < quotient_count;
       local += blockDim.x)
    quotient_output_counts[local] = 0u;
  for (std::uint32_t word = threadIdx.x;
       word < (kBalancedMergeTarget + 31u) / 32u;
       word += blockDim.x)
    tombstone_words[word] = 0u;
  __syncthreads();

  const std::uint32_t task_rows = quotient_offsets[quotient_count];
  for (std::uint32_t index = threadIdx.x; index < task_rows;
       index += blockDim.x) {
    std::uint32_t low = 0u, high = slice_count_shared;
    while (low + 1u < high) {
      const std::uint32_t middle = (low + high) >> 1u;
      if (pull_slices[job.slice_begin + middle].candidate_begin <= index)
        low = middle;
      else
        high = middle;
    }
    const PullSlice slice = pull_slices[job.slice_begin + low];
    const std::uint32_t local = slice.tag & 0x1ffu;
    const std::uint32_t source = slice.tag >> 9u;
    Row row = source == 0u
        ? current_rows[slice.offset + index - slice.candidate_begin]
        : arena[slice.offset + index - slice.candidate_begin];
    if (row.flags & kTombstone)
      atomicOr(tombstone_words + (index >> 5u),
               1u << (index & 31u));
    row.flags = static_cast<std::uint16_t>(
        (local << kMergeSourceBits) | source);
    candidates[index] = row;
  }
  __syncthreads();

  // Pack all smaller source runs first and reserve the largest source at the
  // end of both index planes.  The smaller runs merge into one aggregate;
  // the largest source participates only in the final round.
  if (threadIdx.x == 0u) {
    std::uint32_t run_count = 0u;
    std::uint32_t packed = 0u;
    for (std::uint32_t source = 0u; source < sources; ++source) {
      const std::uint32_t count =
          source_offsets[source + 1u] - source_offsets[source];
      if (!count || source == largest_source_shared) continue;
      run_offsets[run_count] = static_cast<std::uint16_t>(packed);
      run_lengths[run_count] = static_cast<std::uint16_t>(count);
      run_sources[run_count] = static_cast<std::uint16_t>(source);
      packed += count;
      ++run_count;
    }
    run_count_shared = run_count;
    small_count_shared = packed;
  }
  __syncthreads();

  for (std::uint32_t run = 0u; run < run_count_shared; ++run) {
    const std::uint32_t count = run_lengths[run];
    const std::uint32_t destination = run_offsets[run];
    const std::uint32_t source = run_sources[run];
    const std::uint32_t candidate_begin = source_offsets[source];
    for (std::uint32_t index = threadIdx.x; index < count;
         index += blockDim.x)
      merge_indices_a[destination + index] =
          static_cast<std::uint16_t>(candidate_begin + index);
  }
  for (std::uint32_t index = threadIdx.x;
       index < largest_count_shared; index += blockDim.x) {
    const std::uint16_t candidate = static_cast<std::uint16_t>(
        source_offsets[largest_source_shared] + index);
    merge_indices_a[small_count_shared + index] = candidate;
    merge_indices_b[small_count_shared + index] = candidate;
  }
  __syncthreads();

  bool input_is_a = true;
  while (run_count_shared > 1u) {
    const std::uint16_t *input =
        input_is_a ? merge_indices_a : merge_indices_b;
    std::uint16_t *output =
        input_is_a ? merge_indices_b : merge_indices_a;
    std::uint32_t position = threadIdx.x * kItemsPerThread;
    const std::uint32_t thread_end = min(
        position + kItemsPerThread, small_count_shared);
    while (position < thread_end) {
      std::uint32_t pair = 0u;
      while (pair * 2u < run_count_shared) {
        const std::uint32_t first = pair * 2u;
        const std::uint32_t pair_count = run_lengths[first] +
            (first + 1u < run_count_shared
                 ? run_lengths[first + 1u] : 0u);
        if (position < run_offsets[first] + pair_count) break;
        ++pair;
      }
      const std::uint32_t first = pair * 2u;
      const std::uint32_t pair_begin = run_offsets[first];
      const std::uint32_t left_count = run_lengths[first];
      const std::uint32_t right_count = first + 1u < run_count_shared
          ? run_lengths[first + 1u] : 0u;
      const std::uint32_t pair_end =
          pair_begin + left_count + right_count;
      const std::uint32_t output_end = min(thread_end, pair_end);
      if (!right_count) {
        while (position < output_end) {
          output[position] = input[position];
          ++position;
        }
        continue;
      }
      const std::uint16_t *left = input + pair_begin;
      const std::uint16_t *right = left + left_count;
      const std::uint32_t diagonal = position - pair_begin;
      std::uint32_t left_index = pull_merge_partition(
          left, left_count, right, right_count, diagonal, candidates);
      std::uint32_t right_index = diagonal - left_index;
      while (position < output_end) {
        const bool choose_left = right_index >= right_count ||
            (left_index < left_count &&
             pull_index_less(left[left_index], right[right_index],
                             candidates));
        output[position++] = choose_left
            ? left[left_index++] : right[right_index++];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
      const std::uint32_t old_run_count = run_count_shared;
      const std::uint32_t next_run_count =
          (old_run_count + 1u) >> 1u;
      for (std::uint32_t next = 0u; next < next_run_count; ++next) {
        const std::uint32_t first = next * 2u;
        run_offsets[next] = run_offsets[first];
        run_lengths[next] = static_cast<std::uint16_t>(
            run_lengths[first] +
            (first + 1u < old_run_count ? run_lengths[first + 1u] : 0u));
      }
      run_count_shared = next_run_count;
    }
    input_is_a = !input_is_a;
    __syncthreads();
  }

  // Merge the aggregate of every smaller source with the largest source.
  // This is also the only merge round in the common two-source carry.
  if (small_count_shared && largest_count_shared) {
    const std::uint16_t *input =
        input_is_a ? merge_indices_a : merge_indices_b;
    std::uint16_t *output =
        input_is_a ? merge_indices_b : merge_indices_a;
    const std::uint16_t *left = input;
    const std::uint16_t *right = input + small_count_shared;
    std::uint32_t position = threadIdx.x * kItemsPerThread;
    const std::uint32_t output_end = min(
        position + kItemsPerThread, task_rows);
    if (position < output_end) {
      std::uint32_t left_index = pull_merge_partition(
          left, small_count_shared, right, largest_count_shared,
          position, candidates);
      std::uint32_t right_index = position - left_index;
      while (position < output_end) {
        const bool choose_left = right_index >= largest_count_shared ||
            (left_index < small_count_shared &&
             pull_index_less(left[left_index], right[right_index],
                             candidates));
        output[position++] = choose_left
            ? left[left_index++] : right[right_index++];
      }
    }
    input_is_a = !input_is_a;
    __syncthreads();
  }

  const std::uint16_t *sorted_indices =
      input_is_a ? merge_indices_a : merge_indices_b;
  bool live[kItemsPerThread]{};
  std::uint32_t local_live = 0u;
#pragma unroll
  for (std::uint32_t item = 0u; item < kItemsPerThread; ++item) {
    const std::uint32_t index = threadIdx.x * kItemsPerThread + item;
    if (index >= task_rows) continue;
    const std::uint16_t candidate_index = sorted_indices[index];
    const Row row = candidates[candidate_index];
    const std::uint32_t full_local_key =
        pull_candidate_order_key(row) >> kMergeSourceBits;
    const bool first = index == 0u || full_local_key !=
        (pull_candidate_order_key(candidates[sorted_indices[index - 1u]]) >>
         kMergeSourceBits);
    const bool tombstone =
        (tombstone_words[candidate_index >> 5u] &
         (1u << (candidate_index & 31u))) != 0u;
    live[item] = first && (keep_tombstones || !tombstone);
    if (live[item]) {
      ++local_live;
      const std::uint32_t local_q = row.flags >> kMergeSourceBits;
      atomicAdd(quotient_output_counts + local_q, 1u);
    }
  }
  __syncthreads();

  std::uint32_t thread_output_base{}, task_output_count{};
  BlockScan(block_scan_storage).ExclusiveSum(
      local_live, thread_output_base, task_output_count);
  if (threadIdx.x == 0u) {
    std::uint32_t task_output_offset = 0u;
    for (std::uint32_t local = 0u; local < quotient_count; ++local) {
      const std::uint32_t output_count = quotient_output_counts[local];
      quotient_output_offsets[local] = task_output_offset;
      task_output_offset += output_count;
    }

    // The GPU scan assigned this key-ordered raw-input slab before the merge.
    // Its unused tail is intentionally left invisible through route counts.
    const std::uint32_t page_capacity = job.existing_capacity;
    const unsigned long long page_offset = job.existing_offset;
    const bool output_valid = task_output_count <= page_capacity &&
        page_offset + page_capacity <= output_limit;
    if (!output_valid) atomicExch(overflow_flag, 1u);

    for (std::uint32_t local = 0u; local < quotient_count; ++local) {
      const std::uint32_t output_q = job.quotient_begin + local;
      const std::uint32_t output_count = quotient_output_counts[local];
      Descriptor output_descriptor{};
      if (output_count && output_valid)
        output_descriptor = Descriptor::make(
            page_offset + quotient_output_offsets[local], output_count);
      quotient_output_bits[local] = output_descriptor.bits;
      atomicAdd(section_output_counts + output_q, output_count);
      const RouteHeader route = next_route_headers[output_q];
      if (job.route_ordinal < route.count) {
        const std::uint32_t suffix_begin = output_q == job.quotient_begin
            ? static_cast<std::uint32_t>(job.key_begin & 0xffffu) : 0u;
        const std::uint32_t suffix_end =
            output_q + 1u == job.quotient_end
            ? static_cast<std::uint32_t>(
                  job.key_end - (std::uint64_t{output_q} << 16u))
            : 1u << 16u;
        route_slices[route.begin + job.route_ordinal] = {
            output_descriptor, suffix_begin, suffix_end};
      }
    }
    jobs[job_index].existing_offset = page_offset;
    jobs[job_index].existing_capacity = output_valid ? page_capacity : 0u;
    jobs[job_index].output_count = task_output_count;
  }
  __syncthreads();

  std::uint32_t local_rank = 0u;
#pragma unroll
  for (std::uint32_t item = 0u; item < kItemsPerThread; ++item) {
    if (!live[item]) continue;
    const std::uint32_t index = threadIdx.x * kItemsPerThread + item;
    const std::uint16_t candidate_index = sorted_indices[index];
    Row row = candidates[candidate_index];
    const std::uint32_t local_q = row.flags >> kMergeSourceBits;
    const bool tombstone =
        (tombstone_words[candidate_index >> 5u] &
         (1u << (candidate_index & 31u))) != 0u;
    row.flags = tombstone ? kTombstone : 0u;
    const Descriptor output_descriptor{quotient_output_bits[local_q]};
    if (output_descriptor.count()) {
      const std::uint32_t task_rank = thread_output_base + local_rank;
      arena[output_descriptor.offset() + task_rank -
            quotient_output_offsets[local_q]] = row;
    }
    ++local_rank;
  }
  __syncthreads();
  }
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
    const Row *arena, const Descriptor *descriptors,
    std::uint32_t foundation_level, std::uint16_t *local_rank) {
  const std::uint32_t q = blockIdx.x;
  const std::uint32_t cell = threadIdx.x;
  const Descriptor descriptor =
      descriptors[descriptor_index(q, foundation_level)];
  const std::uint32_t target = cell << 9u;
  const std::uint32_t position = descriptor.count() <= 0xffffu
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
    const Row *arena, const Descriptor *descriptors,
    const RouteHeader *route_headers, const RouteSlice *route_slices,
    const std::uint32_t *route_logical_begins,
    const std::uint32_t *level_q_logical_offsets,
    const std::uint16_t *local_rank,
    const std::uint16_t *level_guides,
    std::uint32_t active_levels,
    std::uint32_t foundation_level,
    std::uint64_t occupied_levels,
    const std::uint32_t *query_ids,
    std::uint32_t *final_values, std::uint8_t *final_found,
    const std::uint64_t *query_occupied_level_mask = nullptr) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  if (query_occupied_level_mask) {
    const DeviceManifestSnapshot manifest =
        load_query_manifest(query_occupied_level_mask);
    active_levels = manifest.active_levels;
    foundation_level = manifest.foundation_level;
    occupied_levels = manifest.occupied_level_mask;
  }
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
        std::uint32_t *values = final_values ? final_values : out_values;
        std::uint8_t *found = final_found ? final_found : out_found;
        values[destination] = live ? row.value : found ? 0u : kInvalid;
        if (found) found[destination] = live;
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
    const RouteHeader route_header =
        route_headers[descriptor_index(q, level)];
    const RoutedSliceSelection selected = route_header.count == 1u
        ? RoutedSliceSelection{
              route_slices[route_header.begin].rows, route_header.begin, true}
        : routed_slice_for_suffix(
              q, level, suffix, route_headers, route_slices);
    const Descriptor descriptor = selected.rows;
    if (!descriptor.count()) continue;
    const Row *rows = arena + descriptor.offset();
    std::uint32_t begin = 0u, end = descriptor.count();
    if (route_header.count == 1u) {
      guide_search_bounds(level_guides, q, level, descriptor.count(), suffix,
                          begin, end);
    } else {
      const std::uint32_t logical_count =
          descriptors[descriptor_index(q, level)].count();
      std::uint32_t logical_begin = 0u, logical_end = logical_count;
      guide_search_bounds(level_guides, q, level, logical_count, suffix,
                          logical_begin, logical_end);
      const std::uint32_t section_begin = level_q_logical_offsets[
          std::size_t{level} * (kQuotients + 1u) + q];
      const std::uint32_t route_begin =
          route_logical_begins[selected.route] - section_begin;
      begin = logical_begin > route_begin
          ? logical_begin - route_begin : 0u;
      end = min(descriptor.count(), logical_end > route_begin
          ? logical_end - route_begin : 0u);
    }
    if (begin >= end) continue;
    const std::uint32_t position =
        lower_bound_rows(rows + begin, end - begin, suffix);
    if (position < end - begin && rows[begin + position].key == suffix) {
      const bool live =
          (rows[begin + position].flags & kTombstone) == 0u;
      const std::uint32_t destination = query_ids ? query_ids[i] : i;
      std::uint32_t *values = final_values ? final_values : out_values;
      std::uint8_t *found = final_found ? final_found : out_found;
      values[destination] =
          live ? rows[begin + position].value : found ? 0u : kInvalid;
      if (found) found[destination] = live;
      return;
    }
  }
  const RouteHeader foundation_header = foundation_level < active_levels
      ? route_headers[descriptor_index(q, foundation_level)]
      : RouteHeader{};
  const RoutedSliceSelection foundation_selection =
      foundation_header.count == 1u
          ? RoutedSliceSelection{
                route_slices[foundation_header.begin].rows,
                foundation_header.begin, true}
          : foundation_level < active_levels
              ? routed_slice_for_suffix(
                    q, foundation_level, suffix,
                    route_headers, route_slices)
              : RoutedSliceSelection{};
  const Descriptor foundation = foundation_selection.rows;
  const Row *foundation_rows = arena + foundation.offset();
  const std::uint32_t cell = (key >> 9u) & 127u;
  const std::size_t local_index = std::size_t{q} * 128u + cell;
  const std::uint32_t foundation_logical_count =
      foundation_level < active_levels
          ? descriptors[descriptor_index(q, foundation_level)].count() : 0u;
  const bool ranked = foundation_header.count &&
      foundation_logical_count <= (1u << 16u);
  const std::uint32_t logical_begin =
      ranked ? local_rank[local_index] : 0u;
  const std::uint32_t logical_end = ranked
      ? (cell == 127u ? foundation_logical_count
                      : local_rank[local_index + 1u])
      : foundation_logical_count;
  std::uint32_t begin = logical_begin;
  std::uint32_t end = logical_end;
  if (foundation_header.count != 1u) {
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
  const std::uint32_t position =
      lower_bound_rows(foundation_rows + begin, end - begin, suffix);
  const bool matched = position < end - begin &&
      foundation_rows[begin + position].key == suffix;
  const bool live = matched &&
      (foundation_rows[begin + position].flags & kTombstone) == 0u;
  const std::uint32_t destination = query_ids ? query_ids[i] : i;
  std::uint32_t *values = final_values ? final_values : out_values;
  std::uint8_t *found = final_found ? final_found : out_found;
  values[destination] =
      live ? foundation_rows[begin + position].value
           : found ? 0u : kInvalid;
  if (found) found[destination] = live;
}



__device__ bool first_visible_in_quotient(
    std::uint32_t q, std::uint32_t lower,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const Row *arena,
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
    std::uint32_t pending_batches, const Row *arena,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices, std::uint32_t active_levels,
    const std::uint64_t *query_occupied_level_mask = nullptr) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  std::uint64_t occupied_levels = active_levels == 64u
      ? ~std::uint64_t{0}
      : ((std::uint64_t{1} << active_levels) - 1u);
  if (query_occupied_level_mask) {
    const DeviceManifestSnapshot manifest =
        load_query_manifest(query_occupied_level_mask);
    active_levels = manifest.active_levels;
    occupied_levels = manifest.occupied_level_mask;
  }
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
    bool sorted = false;
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
        maximum_resident_jobs_(
            gpulsmopt2_detail::maximum_resident_merge_jobs(
                publication_capacity_)),
        maximum_resident_boundaries_(
            maximum_resident_jobs_ + gpulsmopt2_detail::kPlanningTiles),
        maximum_pull_slices_(publication_capacity_ + maximum_resident_jobs_),
        route_stride_(gpulsmopt2_detail::adaptive_route_stride(
            publication_capacity_)),
        local_rank_(gpulsmopt2_detail::kLocalRankEntries),
        level_guides_(
            gpulsmopt2_detail::kMaximumLevels *
                gpulsmopt2_detail::kGuideEntriesPerLevel,
            gpulsmopt2_detail::kMaximumLevels *
                gpulsmopt2_detail::kGuideEntriesPerLevel),
        arena_(gpulsmopt2_detail::maximum_resident_elements<
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
        resident_reduce_direct_mode_(1u),
        resident_plan_(1u),
        level_storage_spans_(gpulsmopt2_detail::kMaximumLevels),
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
        resident_boundary_keys_(maximum_resident_boundaries_),
        resident_boundary_cursors_(
            maximum_resident_boundaries_ *
            gpulsmopt2_detail::kMaximumMergeSources),
        resident_route_counts_(gpulsmopt2_detail::kQuotients + 1u),
        resident_route_offsets_(gpulsmopt2_detail::kQuotients + 1u),
        resident_source_slice_offsets_(
            maximum_resident_jobs_ *
            (gpulsmopt2_detail::kMaximumMergeSources + 1u)),
        resident_source_candidate_offsets_(
            maximum_resident_jobs_ *
            (gpulsmopt2_detail::kMaximumMergeSources + 1u)),
        resident_slice_reservations_(maximum_resident_jobs_ + 1u),
        resident_slice_offsets_(maximum_resident_jobs_ + 1u),
        resident_section_logical_offsets_(gpulsmopt2_detail::kQuotients + 1u),
        balanced_merge_jobs_(maximum_resident_jobs_),
        balanced_merge_pull_slices_(maximum_pull_slices_),
        foundation_overflow_flag_(1u),
        admission_counts_(gpulsmopt2_detail::kQuotients + 1u),
        range_partials_(gpulsmopt2_detail::kRangeSchedulerBlocks),
        range_reduction_completion_(1u) {
    CUDA_CHECK(cudaEventCreateWithFlags(&operation_done_,
                                         cudaEventDisableTiming));
    std::size_t initial_sort_bytes{};
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, initial_sort_bytes,
        raw_keys_.data(),
        radix_keys_.data(), radix_ids_out_.data(), radix_ids_out_.data(),
        static_cast<std::uint32_t>(batch_capacity_), 16, 32, 0));
    ensure_radix_workspace(initial_sort_bytes, batch_capacity_, 0);
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
    begin_operation(stream);
    CUDA_CHECK(cudaMemsetAsync(local_rank_.data(), 0,
                               local_rank_.size() * sizeof(std::uint16_t),
                               stream));
    reset_updates(stream);
    end_operation(stream);
  }

  void bulk_build(const std::uint32_t *keys, const std::uint32_t *values,
                  std::size_t count, cudaStream_t stream) {
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
    CUDA_CHECK(cudaMemcpyAsync(
        arena_.data(), publication_rows_a_.data(),
        std::size_t{base_count} * sizeof(gpulsmopt2_detail::Row),
        cudaMemcpyDeviceToDevice, stream));
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
        query_occupied_level_mask_.data(), level, base_count, 0u,
        foundation_pool_capacity_ / 2u,
        static_cast<std::uint32_t>(route_stride_), 0u);
    refresh_active_levels();
    rebuild_foundation_rank(stream);
    end_operation(stream);
  }

  void insert(const DeviceKeyValueBatch &batch, cudaStream_t stream) {
    admit(batch.keys, batch.values, batch.count, false, stream);
  }

  void erase(const DeviceKeyBatch &batch, cudaStream_t stream) {
    admit(batch.keys, nullptr, batch.count, true, stream);
  }

  void lookup(const DeviceLookupBatch &batch, cudaStream_t stream,
              bool quotients_grouped = false) {
    if (!batch.count) return;
    if (!batch.queries || !batch.out_values)
      throw std::invalid_argument("invalid GPULSMOpt lookup");
    if (batch.count > gpulsmopt2_detail::kMaximumOperationTile) {
      for (std::size_t begin = 0u; begin < batch.count;
           begin += gpulsmopt2_detail::kMaximumOperationTile) {
        const std::size_t count = std::min(
            batch.count - begin,
            gpulsmopt2_detail::kMaximumOperationTile);
        lookup(DeviceLookupBatch{
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
      std::size_t bytes{};
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, bytes, batch.queries, radix_keys_.data(),
          radix_ids_out_.data(), radix_ids_out_.data(), count, 16, 32,
          stream));
      ensure_radix_workspace(bytes, count, stream);
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          radix_workspace(), bytes, batch.queries, radix_keys_.data(),
          radix_input_ids(), radix_ids_out_.data(), count, 16, 32, stream));
      queries = radix_keys_.data();
      query_ids = radix_ids_out_.data();
    }
    if (grouped) {
      gpulsmopt2_detail::lookup_with_pending_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          queries, batch.out_values, batch.out_found, count,
          raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
          static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
          raw_signatures_.data(), raw_epoch_signatures_.data(),
          arena_.data(), descriptors_.data(), route_headers_.data(),
          route_slices_.data(), route_logical_begins_.data(),
          level_q_logical_offsets_.data(), local_rank_.data(),
          level_guides_.data(),
          active_levels_, foundation_level(), occupied_level_mask(),
          query_ids, nullptr, nullptr, query_occupied_level_mask_.data());
      CUDA_CHECK(cudaGetLastError());
    } else {
      gpulsmopt2_detail::lookup_with_pending_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          batch.queries, batch.out_values, batch.out_found, count,
          raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
          static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
          raw_signatures_.data(),
          raw_epoch_signatures_.data(), arena_.data(), descriptors_.data(),
          route_headers_.data(), route_slices_.data(),
          route_logical_begins_.data(), level_q_logical_offsets_.data(),
          local_rank_.data(), level_guides_.data(), active_levels_,
          foundation_level(),
          occupied_level_mask(),
          nullptr, nullptr, nullptr, query_occupied_level_mask_.data());
    }
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

  void range(const DeviceRangeOutputBatch &batch, cudaStream_t stream) {
    if (!batch.query_count) return;
    if (!batch.lo || !batch.hi || !batch.out_sums)
      throw std::invalid_argument("invalid GPULSMOpt range input");
    if (batch.query_count > gpulsmopt2_detail::kMaximumOperationTile) {
      for (std::size_t begin = 0u; begin < batch.query_count;
           begin += gpulsmopt2_detail::kMaximumOperationTile) {
        const std::size_t count = std::min(
            batch.query_count - begin,
            gpulsmopt2_detail::kMaximumOperationTile);
        range(DeviceRangeOutputBatch{
            batch.lo + begin, batch.hi + begin, count,
            batch.out_sums + begin}, stream);
      }
      return;
    }
    begin_operation(stream);
    const std::uint32_t query_count =
        static_cast<std::uint32_t>(batch.query_count);
    std::size_t scan_bytes{};
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, range_fragment_counts_.data(),
        range_fragment_offsets_.data(), query_count + 1u, stream));
    ensure_range_fragment_query_capacity(query_count, scan_bytes);
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
    } else {
      CUDA_CHECK(cudaMemcpyAsync(
          &fragment_count, range_fragment_offsets_.data() + query_count,
          sizeof(fragment_count), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      if (!fragment_count) {
        CUDA_CHECK(cudaMemsetAsync(batch.out_sums, 0,
                                   std::size_t{query_count} *
                                       sizeof(std::uint32_t),
                                   stream));
        end_operation(stream);
        return;
      }
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

  void successor(const DeviceSuccessorBatch &batch, cudaStream_t stream) {
    if (!batch.count) return;
    if (!batch.queries || !batch.out_keys)
      throw std::invalid_argument("invalid GPULSMOpt successor input");
    if (batch.count > gpulsmopt2_detail::kMaximumOperationTile) {
      for (std::size_t begin = 0u; begin < batch.count;
           begin += gpulsmopt2_detail::kMaximumOperationTile) {
        const std::size_t count = std::min(
            batch.count - begin,
            gpulsmopt2_detail::kMaximumOperationTile);
        successor(DeviceSuccessorBatch{
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
        pending_batches_, arena_.data(), descriptors_.data(),
        route_headers_.data(), route_slices_.data(), active_levels_,
        query_occupied_level_mask_.data());
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

  MaintenanceStats maintenance_stats() const {
    MaintenanceStats result = stats_;
    result.pending_batches = pending_batches_;
    result.pending_records = pending_records_;
    result.active_levels = active_levels_;
    return result;
  }

  std::size_t gpu_resident_bytes() const {
    return local_rank_.size() * sizeof(std::uint16_t) +
        level_guides_.size() * sizeof(std::uint16_t) +
        arena_.size() * sizeof(gpulsmopt2_detail::Row) +
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
        resident_reduce_direct_mode_.size() * sizeof(std::uint32_t) +
        resident_plan_.size() *
            sizeof(gpulsmopt2_detail::ResidentPublicationPlan) +
        level_storage_spans_.size() *
            sizeof(gpulsmopt2_detail::LevelStorageSpan) +
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
         resident_source_slice_offsets_.size() +
         resident_slice_reservations_.size() +
         resident_slice_offsets_.size() +
         resident_section_logical_offsets_.size()) * sizeof(std::uint32_t) +
        resident_source_candidate_offsets_.size() * sizeof(std::uint16_t) +
        (resident_job_raw_reservations_.size() +
         resident_job_output_offsets_.size() +
         resident_boundary_keys_.size()) * sizeof(std::uint64_t) +
        resident_boundary_cursors_.size() *
            sizeof(gpulsmopt2_detail::BoundaryCursor) +
        balanced_merge_jobs_.size() *
            sizeof(gpulsmopt2_detail::BalancedMergeJob) +
        balanced_merge_pull_slices_.size() *
            sizeof(gpulsmopt2_detail::PullSlice) +
        foundation_next_route_headers_.size() *
            sizeof(gpulsmopt2_detail::RouteHeader) +
        resident_scan_temp_.size() * sizeof(std::uint8_t) +
        publication_temp_.size() * sizeof(std::uint8_t) +
        admission_counts_.size() * sizeof(std::uint32_t) +
        admission_temp_.size() * sizeof(std::uint8_t) +
        radix_storage_.size() * sizeof(std::uint8_t) +
        range_partials_.size() * sizeof(unsigned long long) +
        range_reduction_completion_.size() * sizeof(std::uint32_t) +
        range_query_storage_.size() + range_fragment_storage_.size() +
        range_section_storage_.size();
  }

private:
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

  std::uint64_t occupied_level_mask() const {
    return host_occupied_level_mask_;
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
            arena_.data(), descriptors_.data(), foundation_level(),
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
    gpulsmopt2_detail::ConditionalOutputIterator<gpulsmopt2_detail::Row>
        conditional_row_output(
            publication_rows_a_.data(),
            arena_.data() + foundation_pool_capacity_,
            resident_reduce_direct_mode_.data());
    auto row_output = thrust::make_transform_output_iterator(
        conditional_row_output, gpulsmopt2_detail::AssignmentRow{});
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
    std::uint64_t cursor = foundation_pool_capacity_;
    std::size_t capacity = std::min(
        publication_capacity_,
        batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch);
    for (std::uint32_t level = 0u;
         level < gpulsmopt2_detail::kMaximumLevels; ++level) {
      spans[level] = {cursor, capacity};
      cursor += capacity;
      if (capacity == publication_capacity_) break;
      capacity = capacity > publication_capacity_ / 2u
          ? publication_capacity_ : capacity * 2u;
    }
    if (cursor > foundation_pool_capacity_ + level_pool_capacity_)
      throw std::logic_error("GPULSMOpt preassigned level spans overflow");
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
        nullptr, bytes, resident_slice_reservations_.data(),
        resident_slice_offsets_.data(), maximum_resident_jobs_ + 1u, 0));
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
        gpulsmopt2_detail::compact_balanced_merge_jobs_kernel,
        gpulsmopt2_detail::kFoundationCompactionThreads, 0u));
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

  cudaGraph_t capture_resident_direct_graph(cudaStream_t capture_stream) {
    cudaGraph_t graph{};
    CUDA_CHECK(cudaStreamBeginCapture(
        capture_stream, cudaStreamCaptureModeThreadLocal));
    gpulsmopt2_detail::build_query_quotient_offsets_device_count_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, capture_stream>>>(
            publication_keys_a_.data(),
            publication_selected_count_.data(),
            foundation_source_offsets_.data());
    gpulsmopt2_detail::build_direct_level_directory_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, capture_stream>>>(
            foundation_source_offsets_.data(), resident_plan_.data(),
            static_cast<std::uint32_t>(route_stride_), descriptors_.data(),
            route_headers_.data(), route_slices_.data(),
            route_logical_begins_.data(), route_quotients_.data(),
            level_q_logical_offsets_.data());
    gpulsmopt2_detail::set_direct_survivor_count_kernel<<<
        1, 1, 0, capture_stream>>>(resident_plan_.data());
    gpulsmopt2_detail::build_resident_query_metadata_kernel<<<
        gpulsmopt2_detail::kQuotients, 128u, 0, capture_stream>>>(
            resident_plan_.data(), arena_.data(), descriptors_.data(),
            route_headers_.data(), route_slices_.data(),
            route_logical_begins_.data(), level_q_logical_offsets_.data(),
            local_rank_.data(), level_guides_.data());
    gpulsmopt2_detail::publish_resident_manifest_kernel<<<
        1, gpulsmopt2_detail::kMaximumLevels, 0, capture_stream>>>(
            resident_plan_.data(), device_manifests_.data(),
            active_device_manifest_.data(),
            query_occupied_level_mask_.data(),
            static_cast<std::uint32_t>(route_stride_));
    CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
    return graph;
  }

  cudaGraph_t capture_resident_merge_graph(cudaStream_t capture_stream) {
    cudaGraph_t graph{};
    CUDA_CHECK(cudaStreamBeginCapture(
        capture_stream, cudaStreamCaptureModeThreadLocal));
    CUDA_CHECK(cudaMemsetAsync(
        resident_job_raw_reservations_.data(), 0,
        resident_job_raw_reservations_.size() * sizeof(std::uint64_t),
        capture_stream));
    CUDA_CHECK(cudaMemsetAsync(
        resident_slice_reservations_.data(), 0,
        resident_slice_reservations_.size() * sizeof(std::uint32_t),
        capture_stream));
    CUDA_CHECK(cudaMemsetAsync(
        foundation_section_output_counts_.data(), 0,
        foundation_section_output_counts_.size() * sizeof(std::uint32_t),
        capture_stream));
    CUDA_CHECK(cudaMemsetAsync(
        foundation_overflow_flag_.data(), 0, sizeof(std::uint32_t),
        capture_stream));

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
            balanced_merge_jobs_.data(), resident_boundary_keys_.data(),
            resident_job_raw_reservations_.data());
    gpulsmopt2_detail::resolve_resident_job_boundaries_kernel<<<
        resident_planner_blocks_, 32u, 0,
        capture_stream>>>(
            balanced_merge_jobs_.data(), resident_boundary_keys_.data(),
            resident_job_raw_reservations_.data(), resident_plan_.data(),
            publication_rows_a_.data(), foundation_source_offsets_.data(),
            arena_.data(), route_headers_.data(), route_slices_.data(),
            route_logical_begins_.data(), level_q_logical_offsets_.data());
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
            balanced_merge_raw_counts_.data(), resident_plan_.data(),
            resident_route_counts_.data());
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

    gpulsmopt2_detail::construct_boundary_cursors_kernel<<<
        resident_planner_blocks_, gpulsmopt2_detail::kThreads, 0,
        capture_stream>>>(
            resident_boundary_keys_.data(), resident_plan_.data(),
            publication_keys_a_.data(), foundation_source_offsets_.data(),
            arena_.data(), route_headers_.data(), route_slices_.data(),
            route_logical_begins_.data(), level_q_logical_offsets_.data(),
            static_cast<std::uint32_t>(route_stride_),
            resident_boundary_cursors_.data());
    gpulsmopt2_detail::count_cursor_pull_slices_kernel<<<
        resident_planner_blocks_,
        gpulsmopt2_detail::kThreads, 0, capture_stream>>>(
            balanced_merge_jobs_.data(), resident_plan_.data(),
            resident_boundary_cursors_.data(),
            foundation_source_offsets_.data(), route_headers_.data(),
            route_slices_.data(),
            route_logical_begins_.data(),
            static_cast<std::uint32_t>(route_stride_),
            resident_source_slice_offsets_.data(),
            resident_source_candidate_offsets_.data(),
            resident_slice_reservations_.data());
    scan_bytes = resident_scan_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        resident_scan_temp_.data(), scan_bytes,
        resident_slice_reservations_.data(), resident_slice_offsets_.data(),
        maximum_resident_jobs_ + 1u, capture_stream));
    gpulsmopt2_detail::validate_resident_plan_kernel<<<
        1, 1, 0, capture_stream>>>(
            resident_plan_.data(), resident_tile_job_offsets_.data(),
            resident_job_output_offsets_.data(),
            resident_route_offsets_.data(), resident_slice_offsets_.data(),
            static_cast<std::uint32_t>(maximum_resident_jobs_),
            static_cast<std::uint32_t>(maximum_resident_boundaries_),
            static_cast<std::uint32_t>(route_stride_),
            static_cast<std::uint32_t>(maximum_pull_slices_));
    gpulsmopt2_detail::materialize_cursor_pull_slices_kernel<<<
        resident_planner_blocks_,
        gpulsmopt2_detail::kThreads, 0, capture_stream>>>(
            balanced_merge_jobs_.data(), resident_plan_.data(),
            resident_boundary_cursors_.data(),
            foundation_source_offsets_.data(), route_headers_.data(),
            route_slices_.data(),
            route_logical_begins_.data(), route_quotients_.data(),
            static_cast<std::uint32_t>(route_stride_),
            resident_source_slice_offsets_.data(),
            resident_source_candidate_offsets_.data(),
            resident_slice_offsets_.data(),
            balanced_merge_pull_slices_.data());

    gpulsmopt2_detail::compact_balanced_merge_jobs_kernel<<<
        resident_merge_blocks_,
        gpulsmopt2_detail::kFoundationCompactionThreads, 0,
        capture_stream>>>(
            balanced_merge_jobs_.data(), resident_plan_.data(),
            balanced_merge_pull_slices_.data(),
            publication_rows_a_.data(), arena_.data(),
            foundation_next_route_headers_.data(), route_slices_.data(),
            foundation_section_output_counts_.data(),
            foundation_overflow_flag_.data());
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
    gpulsmopt2_detail::build_resident_query_metadata_kernel<<<
        gpulsmopt2_detail::kQuotients, 128u, 0, capture_stream>>>(
            resident_plan_.data(), arena_.data(), descriptors_.data(),
            route_headers_.data(), route_slices_.data(),
            route_logical_begins_.data(), level_q_logical_offsets_.data(),
            local_rank_.data(), level_guides_.data());
    gpulsmopt2_detail::publish_resident_manifest_kernel<<<
        1, gpulsmopt2_detail::kMaximumLevels, 0, capture_stream>>>(
            resident_plan_.data(), device_manifests_.data(),
            active_device_manifest_.data(),
            query_occupied_level_mask_.data(),
            static_cast<std::uint32_t>(route_stride_));
    CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
    return graph;
  }

  void initialize_resident_publication_graph() {
    CUDA_CHECK(cudaGraphCreate(&resident_publication_graph_, 0u));
    CUDA_CHECK(cudaGraphConditionalHandleCreate(
        &resident_publication_conditional_, resident_publication_graph_,
        0u, cudaGraphCondAssignDefault));

    auto *selected_count = publication_selected_count_.data();
    auto *manifests = device_manifests_.data();
    auto *active_manifest = active_device_manifest_.data();
    auto *level_spans = level_storage_spans_.data();
    const std::uint64_t bank_capacity = foundation_pool_capacity_ / 2u;
    auto *plan = resident_plan_.data();
    void *controller_arguments[] = {
        &resident_publication_conditional_, &selected_count, &manifests,
        &active_manifest, &level_spans,
        const_cast<std::uint64_t *>(&bank_capacity), &plan};
    cudaKernelNodeParams controller_params{};
    controller_params.func = reinterpret_cast<void *>(
        gpulsmopt2_detail::choose_resident_publication_path_kernel);
    controller_params.gridDim = dim3(1u);
    controller_params.blockDim = dim3(1u);
    controller_params.kernelParams = controller_arguments;
    cudaGraphNode_t controller{};
    CUDA_CHECK(cudaGraphAddKernelNode(
        &controller, resident_publication_graph_, nullptr, 0u,
        &controller_params));

    cudaGraphNodeParams conditional_params{};
    conditional_params.type = cudaGraphNodeTypeConditional;
    conditional_params.conditional.handle =
        resident_publication_conditional_;
    conditional_params.conditional.type = cudaGraphCondTypeIf;
    conditional_params.conditional.size = 2u;
    cudaGraphNode_t conditional{};
    CUDA_CHECK(cudaGraphAddNode(
        &conditional, resident_publication_graph_, &controller, 1u,
        &conditional_params));
    cudaGraph_t *bodies = conditional_params.conditional.phGraph_out;
    if (!bodies)
      throw std::runtime_error(
          "CUDA did not create conditional publication bodies");

    cudaStream_t capture_stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &capture_stream, cudaStreamNonBlocking));
    cudaGraph_t merge_graph{}, direct_graph{};
    try {
      merge_graph = capture_resident_merge_graph(capture_stream);
      direct_graph = capture_resident_direct_graph(capture_stream);
      cudaGraphNode_t child{};
      CUDA_CHECK(cudaGraphAddChildGraphNode(
          &child, bodies[0], nullptr, 0u, merge_graph));
      CUDA_CHECK(cudaGraphAddChildGraphNode(
          &child, bodies[1], nullptr, 0u, direct_graph));
      CUDA_CHECK(cudaGraphInstantiate(
          &resident_publication_graph_exec_,
          resident_publication_graph_, 0ull));
    } catch (...) {
      if (merge_graph) cudaGraphDestroy(merge_graph);
      if (direct_graph) cudaGraphDestroy(direct_graph);
      cudaStreamDestroy(capture_stream);
      throw;
    }
    CUDA_CHECK(cudaGraphDestroy(merge_graph));
    CUDA_CHECK(cudaGraphDestroy(direct_graph));
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
    active_levels_ = 0u;
    host_occupied_level_mask_ = 0u;
    std::fill_n(level_counts_, gpulsmopt2_detail::kMaximumLevels, 0u);
    gpulsmopt2_detail::initialize_device_manifest_kernel<<<1, 1, 0, stream>>>(
        device_manifests_.data(), active_device_manifest_.data(),
        query_occupied_level_mask_.data(), 0u, 0u, 0u, 0u,
        static_cast<std::uint32_t>(route_stride_), 0u);
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
    while (consumed < count) {
      const std::size_t remaining = count - consumed;
      const std::uint32_t tile_count = static_cast<std::uint32_t>(
          std::min(remaining, batch_capacity_));
      admit_tile(keys + consumed,
                 tombstone ? nullptr : values + consumed,
                 tile_count, tombstone, stream);
      consumed += tile_count;
    }
    ++stats_.admitted_batches;
    stats_.admitted_records += count;
    end_operation(stream);
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
    if (pending_records_ > publication_capacity_)
      throw std::bad_alloc();
    std::uint32_t batch_offsets[gpulsmopt2_detail::kBatchesPerEpoch + 1u]{};
    for (std::uint32_t batch = 0u;
         batch < gpulsmopt2_detail::kBatchesPerEpoch; ++batch)
      batch_offsets[batch + 1u] =
          batch_offsets[batch] + raw_batch_counts_[batch];
    CUDA_CHECK(cudaMemcpyAsync(
        publication_batch_offsets_.data(), batch_offsets,
        sizeof(batch_offsets), cudaMemcpyHostToDevice, stream));
    const dim3 publication_grid(
        blocks(batch_capacity_), gpulsmopt2_detail::kBatchesPerEpoch);
    gpulsmopt2_detail::pack_publication_epoch_kernel<<<
        publication_grid, gpulsmopt2_detail::kThreads, 0, stream>>>(
            raw_keys_.data(), raw_payloads_.data(),
            static_cast<std::uint32_t>(batch_capacity_),
            publication_batch_offsets_.data(),
            publication_epoch_keys_a_.data(),
            publication_epoch_assignments_a_.data());
    std::size_t workspace_bytes = publication_temp_.size();
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        publication_temp_.data(), workspace_bytes,
        publication_epoch_keys_a_.data(), publication_epoch_keys_b_.data(),
        publication_epoch_assignments_a_.data(),
        publication_epoch_assignments_b_.data(), pending_records_, 0, 32,
        stream));
    gpulsmopt2_detail::choose_resident_reduce_output_kernel<<<1, 1, 0, stream>>>(
        device_manifests_.data(), active_device_manifest_.data(),
        resident_reduce_direct_mode_.data());
    gpulsmopt2_detail::ConditionalOutputIterator<gpulsmopt2_detail::Row>
        conditional_row_output(
            publication_rows_a_.data(),
            arena_.data() + foundation_pool_capacity_,
            resident_reduce_direct_mode_.data());
    auto row_output = thrust::make_transform_output_iterator(
        conditional_row_output, gpulsmopt2_detail::AssignmentRow{});
    workspace_bytes = publication_temp_.size();
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        publication_temp_.data(), workspace_bytes,
        publication_epoch_keys_b_.data(), publication_keys_a_.data(),
        publication_epoch_assignments_b_.data(), row_output,
        publication_selected_count_.data(),
        gpulsmopt2_detail::NewestAssignment{}, pending_records_, stream));
    CUDA_CHECK(cudaGraphLaunch(resident_publication_graph_exec_, stream));
    CUDA_CHECK(cudaGetLastError());

    // This mirror is only for human-facing statistics.  Planning, allocation,
    // query visibility, and the next carry all use the device manifest.
    std::uint32_t destination = 0u;
    while (destination < gpulsmopt2_detail::kMaximumLevels &&
           (host_occupied_level_mask_ &
            (std::uint64_t{1} << destination)))
      ++destination;
    if (destination < gpulsmopt2_detail::kMaximumLevels) {
      const std::uint64_t consumed = destination
          ? ((std::uint64_t{1} << destination) - 1u) : 0u;
      host_occupied_level_mask_ &= ~consumed;
      host_occupied_level_mask_ |= std::uint64_t{1} << destination;
      active_levels_ = host_occupied_level_mask_
          ? 64u - static_cast<std::uint32_t>(
                        __builtin_clzll(host_occupied_level_mask_))
          : 0u;
    }
    CUDA_CHECK(cudaMemsetAsync(
        raw_epoch_signatures_.data(), 0,
        raw_epoch_signatures_.size() * sizeof(std::uint64_t), stream));
    pending_batches_ = 0u;
    pending_records_ = 0u;
    ++stats_.epochs_published;
    return;
  }

  static std::size_t aligned_id_bytes(std::size_t count) {
    return (count * sizeof(std::uint32_t) + 255u) & ~std::size_t{255u};
  }
  void ensure_radix_workspace(std::size_t bytes, std::size_t count,
                              cudaStream_t stream) {
    const std::size_t capacity = std::max(radix_id_capacity_, count);
    const std::size_t ids_bytes = aligned_id_bytes(capacity);
    const std::size_t required = ids_bytes * 3u + bytes;
    const bool resized = radix_storage_.size() < required;
    if (resized) radix_storage_.resize(required);
    std::uint8_t *storage = radix_storage_.data();
    radix_keys_.attach(reinterpret_cast<std::uint32_t *>(storage), capacity);
    radix_ids_out_.attach(
        reinterpret_cast<std::uint32_t *>(storage + ids_bytes), capacity);
    radix_input_ids_ =
        reinterpret_cast<std::uint32_t *>(storage + ids_bytes * 2u);
    radix_workspace_ = storage + ids_bytes * 3u;
    if (resized || count > radix_id_capacity_) {
      gpulsmopt2_detail::iota_kernel<<<
          blocks(capacity), gpulsmopt2_detail::kThreads, 0, stream>>>(
              radix_input_ids_, static_cast<std::uint32_t>(capacity));
      CUDA_CHECK(cudaGetLastError());
    }
    radix_id_capacity_ = capacity;
  }
  std::uint32_t *radix_input_ids() { return radix_input_ids_; }
  void *radix_workspace() { return radix_workspace_; }
  void launch_section_ranges(cudaStream_t stream) {
    gpulsmopt2_detail::cooperative_section_owned_range_kernel<
        gpulsmopt2_detail::SumRowsAggregate>
        <<<range_section_blocks_, gpulsmopt2_detail::kSectionRangeThreads,
           0, stream>>>(
            range_section_fragments_out_.data(),
            range_section_tasks_.data(),
            range_section_task_offsets_.data() +
                gpulsmopt2_detail::kQuotients,
            arena_.data(), descriptors_.data(), route_headers_.data(),
            route_slices_.data(), raw_keys_.data(),
            raw_payloads_.data(),
            raw_offsets_.data(), static_cast<std::uint32_t>(batch_capacity_),
            pending_batches_,
            active_levels_,
            active_levels_ ? active_levels_ - 1u
                           : gpulsmopt2_detail::kInvalid,
            range_fragment_partials_.data(), occupied_level_mask(),
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
            batch.lo, batch.hi, arena_.data(), descriptors_.data(),
            route_headers_.data(), route_slices_.data(),
            raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
            static_cast<std::uint32_t>(batch_capacity_),
            pending_batches_,
            active_levels_,
            active_levels_ ? active_levels_ - 1u
                           : gpulsmopt2_detail::kInvalid,
            range_fragment_partials_.data(),
            occupied_level_mask(), query_occupied_level_mask_.data());
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
    std::size_t offset = 0u;
    attach_range_view(range_fragment_counts_, range_query_storage_.data(),
                      offset, entries);
    attach_range_view(range_fragment_offsets_, range_query_storage_.data(),
                      offset, entries);
    offset = aligned_range_bytes(offset);
    range_query_temp_ = range_query_storage_.data() + offset;
  }
  void ensure_range_fragment_capacity(std::size_t count) {
    if (range_fragments_.size() >= count &&
        range_fragment_partials_.size() >= count) return;
    std::size_t bytes = 0u;
    bytes += aligned_range_bytes(
        count * sizeof(gpulsmopt2_detail::RangeFragment));
    bytes += aligned_range_bytes(count * sizeof(unsigned long long));
    range_fragment_storage_.resize(bytes);
    std::size_t offset = 0u;
    attach_range_view(range_fragments_, range_fragment_storage_.data(),
                      offset, count);
    attach_range_view(range_fragment_partials_,
                      range_fragment_storage_.data(), offset, count);
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
    std::size_t offset = 0u;
    attach_range_view(range_section_keys_in_, range_section_storage_.data(),
                      offset, count);
    attach_range_view(range_section_keys_out_, range_section_storage_.data(),
                      offset, count);
    attach_range_view(range_section_fragments_in_,
                      range_section_storage_.data(), offset, count);
    attach_range_view(range_section_fragments_out_,
                      range_section_storage_.data(), offset, count);
    attach_range_view(range_fragment_partials_, range_section_storage_.data(),
                      offset, count);
    attach_range_view(range_section_offsets_, range_section_storage_.data(),
                      offset, sections);
    attach_range_view(range_section_task_offsets_,
                      range_section_storage_.data(), offset, sections);
    attach_range_view(range_section_task_counts_,
                      range_section_storage_.data(), offset, sections);
    attach_range_view(range_section_tasks_, range_section_storage_.data(),
                      offset, maximum_tasks);
    offset = aligned_range_bytes(offset);
    range_section_temp_ = range_section_storage_.data() + offset;
    range_section_temp_bytes_ = temp_bytes;
  }
  std::size_t batch_capacity_{};
  std::size_t publication_capacity_{};
  std::size_t foundation_pool_capacity_{};
  std::size_t level_pool_capacity_{};
  std::size_t maximum_resident_jobs_{};
  std::size_t maximum_resident_boundaries_{};
  std::size_t maximum_pull_slices_{};
  std::size_t route_stride_{};
  std::uint64_t host_occupied_level_mask_{};
  std::uint32_t pending_batches_{};
  std::uint32_t pending_records_{};
  std::uint32_t active_levels_{};
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
  cudaGraphConditionalHandle resident_publication_conditional_{};
  std::uint32_t resident_merge_blocks_{};
  std::uint32_t resident_planner_blocks_{};
  std::uint32_t range_section_blocks_{};

  gpulsmopt2_detail::Buffer<std::uint16_t> local_rank_;
  gpulsmopt2_detail::VirtualBuffer<std::uint16_t> level_guides_;
  gpulsmopt2_detail::VirtualBuffer<gpulsmopt2_detail::Row> arena_;
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
  gpulsmopt2_detail::Buffer<std::uint32_t> resident_reduce_direct_mode_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::ResidentPublicationPlan>
      resident_plan_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::LevelStorageSpan>
      level_storage_spans_;
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
      resident_job_raw_reservations_, resident_job_output_offsets_,
      resident_boundary_keys_;
  gpulsmopt2_detail::Buffer<std::uint32_t>
      resident_tile_job_counts_, resident_tile_job_offsets_,
      resident_route_counts_, resident_route_offsets_,
      resident_source_slice_offsets_, resident_slice_reservations_,
      resident_slice_offsets_, resident_section_logical_offsets_;
  gpulsmopt2_detail::Buffer<std::uint16_t>
      resident_source_candidate_offsets_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::BoundaryCursor>
      resident_boundary_cursors_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::BalancedMergeJob>
      balanced_merge_jobs_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::PullSlice>
      balanced_merge_pull_slices_;
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
