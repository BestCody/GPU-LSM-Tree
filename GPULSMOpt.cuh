#pragma once
#include "gpu_dictionary_adapter.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_merge.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <thrust/iterator/transform_output_iterator.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
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
constexpr std::uint32_t kGuideRegions = 16u;
constexpr std::uint32_t kGuideSamples = kGuideRegions - 1u;
constexpr std::size_t kGuideEntriesPerLevel =
    std::size_t{kQuotients} * kGuideSamples;
constexpr std::uint32_t kThreads = 256u;
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
constexpr std::uint32_t kLookupWarpsPerBlock = 8u;
constexpr std::uint32_t kLookupHashSlots = 64u;
constexpr std::uint32_t kEmptyLookupKey = 1u << 16u;
constexpr std::uint32_t kDirectAdmissionMinimum = kQuotients / 16u;
constexpr std::uint32_t kDirectAdmissionMaximum =
    kQuotients * 8u;
constexpr std::uint32_t kFoundationCompactionThreads = 256u;
constexpr std::uint32_t kFoundationItemsPerThread = 9u;
constexpr std::uint32_t kFoundationSectionCapacity =
    kFoundationCompactionThreads * kFoundationItemsPerThread;
constexpr std::uint32_t kFoundationAgeBits = 6u;
constexpr std::uint32_t kFoundationCells = 128u;
constexpr std::uint32_t kFoundationCellKeys = 512u;
constexpr std::uint32_t kFoundationCapacityGranularity = 128u;
constexpr std::uint32_t kBalancedMergeTarget =
    kFoundationSectionCapacity;
constexpr std::uint32_t kBalancedMergeMaximumQuotients = 512u;
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
  // Two alternating banks.  Each bank needs up to 1.5x live rows for section
  // slack plus at most one capacity-granularity rounding unit per quotient;
  // 2x per bank is the simple distribution-independent bound.
  const std::size_t requested_banks = requested > even_maximum / 4u
      ? even_maximum : requested * 4u;
  const std::size_t capacity = std::max<std::size_t>(
      requested_banks, std::size_t{kQuotients} * 1024u);
  return std::min(even_maximum, (capacity + 1u) & ~std::size_t{1u});
}

inline std::size_t adaptive_route_stride(std::size_t requested) {
  return std::size_t{kQuotients} +
      (requested + kBalancedMergeTarget - 1u) / kBalancedMergeTarget + 1u;
}

inline std::size_t initial_crowded_merge_sections(std::size_t capacity) {
  const std::size_t maximum_rows = capacity > kMaximumPublicationRows / 2u
      ? kMaximumPublicationRows : capacity * 2u;
  return std::max<std::size_t>(
      1u, std::min<std::size_t>(
              kQuotients,
              (maximum_rows + kBalancedMergeTarget) /
                  (kBalancedMergeTarget + 1u)));
}

struct Row {
  std::uint32_t value;
  std::uint16_t key;
  std::uint16_t flags;
};

__host__ __device__ __forceinline__ std::uint32_t
foundation_section_capacity(std::uint32_t count) {
  if (!count) return 0u;
  const std::uint32_t slack = max((count + 1u) / 2u,
                                  kFoundationCapacityGranularity);
  const std::uint32_t requested = min(0x10000u, count + slack);
  return min(0x10000u,
             (requested + kFoundationCapacityGranularity - 1u) &
                 ~(kFoundationCapacityGranularity - 1u));
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

__host__ __device__ __forceinline__ std::uint32_t mix_lookup_key(
    std::uint32_t key) {
  key ^= key >> 16u;
  key *= 0x7feb352du;
  key ^= key >> 15u;
  key *= 0x846ca68bu;
  return key ^ (key >> 16u);
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

__host__ __device__ __forceinline__ RawAssignment make_raw_assignment(
    std::uint32_t key, std::uint32_t value,
    std::uint32_t logical_position, bool tombstone) {
  return {key, tombstone ? 0u : value,
          logical_position | (tombstone ? kRawTombstone : 0u)};
}

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

struct BalancedMergeJob {
  std::uint64_t key_begin{};
  std::uint64_t key_end{};
  std::uint64_t existing_offset{};
  std::uint32_t quotient_begin;
  std::uint32_t quotient_end;
  std::uint32_t reuse_existing_page{};
  std::uint32_t existing_capacity{};
  std::uint32_t output_count{};
  std::uint32_t slice_begin{};
  std::uint32_t slice_count{};
  std::uint32_t route_ordinal{};
};

static_assert(sizeof(BalancedMergeJob) == 56u);

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

struct CrowdedMergeTask {
  std::uint32_t hot_index;
  std::uint32_t range;
};

static_assert(sizeof(CrowdedMergeTask) == 8u);

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
  std::size_t resident_bytes() const { return mapped_bytes_; }

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

struct SumRowsAggregate {
  using State = unsigned long long;
  __device__ static State identity() { return 0ull; }
  __device__ static State consume(State state, const Row &row) {
    return state + row.value;
  }
  __device__ State operator()(State a, State b) const { return a + b; }
};

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

template <bool SectionOwned>
__device__ __forceinline__ void emit_range_fragment(
    std::uint32_t index, std::uint32_t query, std::uint32_t quotient,
    std::uint32_t low, std::uint32_t high, RangeFragment *fragments,
    std::uint32_t *section_keys, SectionRangeFragment *section_fragments) {
  if constexpr (SectionOwned) {
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

template <bool SectionOwned>
__global__ void emit_range_fragments_kernel(
    const std::uint32_t *low, const std::uint32_t *high,
    const std::uint32_t *offsets, std::uint32_t query_count,
    RangeFragment *fragments, std::uint32_t *section_keys,
    SectionRangeFragment *section_fragments) {
  const std::uint32_t warp =
      (blockIdx.x * blockDim.x + threadIdx.x) >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  if (warp >= query_count || low[warp] > high[warp]) return;
  const std::uint32_t first = low[warp] >> 16u;
  const std::uint32_t count = offsets[warp + 1u] - offsets[warp];
  for (std::uint32_t local = lane; local < count; local += 32u)
    emit_range_fragment<SectionOwned>(
        offsets[warp] + local, warp, first + local, low[warp], high[warp],
        fragments, section_keys, section_fragments);
}

template <bool SectionOwned>
__global__ void emit_range_fragments_thread_kernel(
    const std::uint32_t *low, const std::uint32_t *high,
    const std::uint32_t *offsets, std::uint32_t query_count,
    RangeFragment *fragments, std::uint32_t *section_keys,
    SectionRangeFragment *section_fragments) {
  const std::uint32_t query =
      blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= query_count || low[query] > high[query]) return;
  const std::uint32_t first = low[query] >> 16u;
  const std::uint32_t count = offsets[query + 1u] - offsets[query];
  for (std::uint32_t local = 0u; local < count; ++local)
    emit_range_fragment<SectionOwned>(
        offsets[query] + local, query, first + local, low[query], high[query],
        fragments, section_keys, section_fragments);
}
template <bool SectionOwned>
__global__ void emit_single_range_fragments_kernel(
    const std::uint32_t *low, const std::uint32_t *high,
    const std::uint32_t *offsets, RangeFragment *fragments,
    std::uint32_t *section_keys,
    SectionRangeFragment *section_fragments) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count = offsets[1u];
  if (index < count) {
    const std::uint32_t quotient = (low[0u] >> 16u) + index;
    emit_range_fragment<SectionOwned>(
        index, 0u, quotient, low[0u], high[0u], fragments, section_keys,
        section_fragments);
  }
}

__global__ void reduce_range_fragment_partials_kernel(
    const std::uint32_t *offsets, std::uint32_t query_count,
    const unsigned long long *partials, std::uint32_t *out_sums) {
  const std::uint32_t warp =
      (blockIdx.x * blockDim.x + threadIdx.x) >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  if (warp >= query_count) return;
  unsigned long long local = 0ull;
  for (std::uint32_t index = offsets[warp] + lane;
       index < offsets[warp + 1u]; index += 32u)
    local += partials[index];
  for (std::uint32_t offset = 16u; offset; offset >>= 1u)
    local += __shfl_down_sync(0xffffffffu, local, offset);
  if (lane == 0u) out_sums[warp] = static_cast<std::uint32_t>(local);
}

__global__ void reduce_range_fragment_partials_thread_kernel(
    const std::uint32_t *offsets, std::uint32_t query_count,
    const unsigned long long *partials, std::uint32_t *out_sums) {
  const std::uint32_t query =
      blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= query_count) return;
  unsigned long long local = 0ull;
  for (std::uint32_t index = offsets[query];
       index < offsets[query + 1u]; ++index)
    local += partials[index];
  out_sums[query] = static_cast<std::uint32_t>(local);
}
__global__ void reduce_single_range_partials_stage1_kernel(
    const unsigned long long *partials,
    const std::uint32_t *device_fragment_count,
    unsigned long long *block_partials) {
  __shared__ unsigned long long values[256];
  unsigned long long local = 0ull;
  const std::uint32_t count = *device_fragment_count;
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += gridDim.x * blockDim.x)
    local += partials[index];
  values[threadIdx.x] = local;
  __syncthreads();
  for (std::uint32_t stride = 128u; stride; stride >>= 1u) {
    if (threadIdx.x < stride)
      values[threadIdx.x] += values[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0u) block_partials[blockIdx.x] = values[0u];
}

__global__ void reduce_single_range_partials_stage2_kernel(
    const unsigned long long *block_partials, std::uint32_t *out_sum) {
  __shared__ unsigned long long values[256];
  values[threadIdx.x] = block_partials[threadIdx.x];
  __syncthreads();
  for (std::uint32_t stride = 128u; stride; stride >>= 1u) {
    if (threadIdx.x < stride)
      values[threadIdx.x] += values[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0u)
    out_sum[0u] = static_cast<std::uint32_t>(values[0u]);
}

__device__ __noinline__ unsigned long long warp_sum_visible_by_verification(
    std::uint32_t q, std::uint32_t low, std::uint32_t high,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets,
    std::uint32_t batch_stride, std::uint32_t pending_batches,
    const Row *arena, const Descriptor *descriptors,
    const RouteHeader *route_headers, const RouteSlice *route_slices,
    std::uint32_t active_levels) {
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
        const Descriptor newer_descriptor = routed_descriptor_for_suffix(
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

__global__ void summarize_section_fragment_density_kernel(
    const std::uint32_t *section_offsets,
    std::uint32_t *maximum_fragments) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  atomicMax(maximum_fragments,
            section_offsets[q + 1u] - section_offsets[q]);
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

template <class Aggregate, bool HasPending, bool Tiled>
__global__ void cooperative_section_owned_range_kernel(
    const SectionRangeFragment *fragments,
    const std::uint32_t *section_offsets, const SectionRangeTask *tasks,
    const std::uint32_t *task_count, const Row *arena,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices, const std::uint32_t *raw_keys,
    const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, std::uint32_t active_levels,
    std::uint32_t foundation_level,
    typename Aggregate::State *aggregate_partials) {
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
  __shared__ std::uint32_t section_base_mask_valid_shared;
  __shared__ std::uint32_t quotient_shared;
  __shared__ std::uint32_t fragment_begin_shared;
  __shared__ std::uint32_t fragment_end_shared;
  __shared__ std::uint32_t task_valid_shared;
  __shared__ std::uint32_t base_section_count_shared;
  __shared__ Descriptor foundation_descriptor_shared;
  __shared__ Descriptor section_descriptors[kMaximumLevels];

  if (threadIdx.x == 0u) {
    task_valid_shared = 1u;
    if constexpr (Tiled) {
      const std::uint32_t task_index = blockIdx.x;
      if (task_index >= *task_count) {
        task_valid_shared = 0u;
      } else {
        const SectionRangeTask task = tasks[task_index];
        quotient_shared = task.quotient;
        fragment_begin_shared = task.begin;
        fragment_end_shared = task.end;
      }
    } else {
      quotient_shared = blockIdx.x;
      fragment_begin_shared = section_offsets[blockIdx.x];
      fragment_end_shared = section_offsets[blockIdx.x + 1u];
      task_valid_shared =
          fragment_begin_shared != fragment_end_shared;
    }
  }
  __syncthreads();
  if (!task_valid_shared) return;
  const std::uint32_t q = quotient_shared;
  const std::uint32_t fragment_begin = fragment_begin_shared;
  const std::uint32_t fragment_end = fragment_end_shared;
  if (threadIdx.x < active_levels)
    section_descriptors[threadIdx.x] =
        descriptors[descriptor_index(q, threadIdx.x)];
  if (threadIdx.x == 0u) {
    foundation_descriptor_shared = foundation_level < active_levels
        ? section_descriptors[foundation_level] : Descriptor{};
    base_section_count_shared = foundation_descriptor_shared.count();
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    std::uint32_t physical = 0u;
    pending_count_shared = 0u;
    if constexpr (HasPending) {
      for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
        const std::size_t oi =
            std::size_t{batch} * (kQuotients + 1u) + q;
        pending_count_shared += raw_offsets[oi + 1u] - raw_offsets[oi];
      }
      physical += pending_count_shared;
    }
    for (std::uint32_t level = 0u; level < active_levels; ++level)
      if (level != foundation_level)
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
      if (level == foundation_level) continue;
      const Descriptor descriptor = section_descriptors[level];
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

  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  constexpr unsigned full_mask = 0xffffffffu;
  const std::uint32_t current_count = current_count_shared;
  const std::uint32_t base_section_count = base_section_count_shared;
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
  if constexpr (Tiled) {
    if (threadIdx.x == 0u) {
      const std::uint32_t count = fragment_end - fragment_begin;
      const std::uint32_t samples = min(count, 8u);
      std::uint32_t widths[8]{};
      for (std::uint32_t sample = 0u; sample < samples; ++sample) {
        const std::uint32_t local =
            (std::uint64_t{sample} * count) / samples;
        const SectionRangeFragment item = fragments[fragment_begin + local];
        widths[sample] =
            std::uint32_t{item.high_suffix} - item.low_suffix + 1u;
        for (std::uint32_t position = sample; position > 0u &&
             widths[position] < widths[position - 1u]; --position) {
          const std::uint32_t temporary = widths[position];
          widths[position] = widths[position - 1u];
          widths[position - 1u] = temporary;
        }
      }
      const std::uint32_t lower = widths[samples / 4u];
      const std::uint32_t upper = widths[(3u * samples) / 4u];
      const bool variable = 10u * upper > 11u * lower;
      dynamic_queue_shared = variable;
      next_fragment_shared = fragment_begin;
    }
    __syncthreads();
  }
  if (!overflow_shared && (!Tiled || !dynamic_queue_shared) &&
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
        if constexpr (Tiled) {
          if (dynamic_queue_shared) {
            if (lane == 0u)
              group_begin = atomicAdd(&next_fragment_shared, kGroup);
            group_begin = __shfl_sync(full_mask, group_begin, 0u);
          }
        }
        if (group_begin >= fragment_end) break;
        const std::uint32_t fragment_index = group_begin + subgroup;
        const bool active = fragment_index < fragment_end;
        SectionRangeFragment fragment{};
        if (active && subgroup_lane == 0u)
          fragment = fragments[fragment_index];
        std::uint32_t low_suffix = fragment.low_suffix;
        std::uint32_t high_suffix = fragment.high_suffix;
        low_suffix = __shfl_sync(full_mask, low_suffix,
                                 subgroup * kSubgroup);
        high_suffix = __shfl_sync(full_mask, high_suffix,
                                  subgroup * kSubgroup);
        std::uint32_t update_begin = 0u, update_end = 0u;
        std::uint32_t base_begin = 0u, base_end = 0u;
        if (active && subgroup_lane == 0u) {
          if (current_count) {
            update_begin = lower_bound_rows(
                current, current_count, low_suffix);
            update_end = upper_bound_rows(
                current, current_count, high_suffix);
          }
          if (base_section_count) {
            base_begin = lower_bound_rows(
                foundation_rows, base_section_count, low_suffix);
            base_end = upper_bound_rows(
                foundation_rows, base_section_count, high_suffix);
          }
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
        if constexpr (Tiled) {
          if (!dynamic_queue_shared)
            group_begin += kSectionRangeWarps * kGroup;
        } else {
          group_begin += kSectionRangeWarps * kGroup;
        }
      }
      return;
  }
  std::uint32_t fragment_index = fragment_begin + warp;
  while (true) {
    if constexpr (Tiled) {
      if (dynamic_queue_shared) {
        if (lane == 0u)
          fragment_index = atomicAdd(&next_fragment_shared, 1u);
        fragment_index = __shfl_sync(full_mask, fragment_index, 0u);
      }
    }
    if (fragment_index >= fragment_end) break;
    const SectionRangeFragment fragment = fragments[fragment_index];
    const std::uint32_t q_low = q << 16u;
    const std::uint32_t low = q_low | fragment.low_suffix;
    const std::uint32_t high = q_low | fragment.high_suffix;
    const std::uint32_t low_suffix = fragment.low_suffix;
    const std::uint32_t high_suffix = fragment.high_suffix;
    unsigned long long local = 0ull;
    if (overflow_shared) {
      local = warp_sum_visible_by_verification(
          q, low, high, HasPending ? raw_keys : nullptr,
          HasPending ? raw_payloads : nullptr,
          HasPending ? raw_offsets : nullptr,
          batch_stride,
          HasPending ? pending_batches : 0u, arena, descriptors,
          route_headers, route_slices, active_levels);
    } else {
      std::uint32_t update_begin = 0u, update_end = 0u;
      if (lane == 0u && current_count) {
        update_begin = lower_bound_rows(
            current, current_count, low_suffix);
        update_end = upper_bound_rows(
            current, current_count, high_suffix);
      }
      update_begin = __shfl_sync(full_mask, update_begin, 0u);
      update_end = __shfl_sync(full_mask, update_end, 0u);
      const std::uint32_t update_count = update_end - update_begin;
      for (std::uint32_t index = update_begin + lane; index < update_end;
           index += 32u) {
        const Row row = current[index];
        if ((row.flags & kTombstone) == 0u)
          local = Aggregate::consume(local, row);
      }

      std::uint32_t begin = 0u, end = 0u;
      if (lane == 0u && base_section_count) {
        begin = lower_bound_rows(
            foundation_rows, base_section_count, low_suffix);
        end = upper_bound_rows(
            foundation_rows, base_section_count, high_suffix);
      }
      begin = __shfl_sync(full_mask, begin, 0u);
      end = __shfl_sync(full_mask, end, 0u);
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
    if constexpr (Tiled) {
      if (!dynamic_queue_shared) fragment_index += kSectionRangeWarps;
    } else {
      fragment_index += kSectionRangeWarps;
    }
  }
}

template <class Aggregate, bool HasPending>
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
    std::uint32_t *overflow_fragments, std::uint32_t *overflow_count) {
  constexpr std::uint32_t kWarps = 4u;
  constexpr std::uint32_t kUpdateCapacity = 128u;
  union WarpScratch {
    Row merged[kUpdateCapacity];
    TaggedRow tagged[HasPending ? kUpdateCapacity : 1u];
  };
  __shared__ Row current_shared[kWarps][kUpdateCapacity];
  __shared__ WarpScratch scratch[kWarps];
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

  bool adaptive = false;
  if (lane == 0u)
    for (std::uint32_t level = 0u; level < active_levels; ++level)
      adaptive |= route_headers[descriptor_index(q, level)].count > 1u;
  adaptive = __shfl_sync(full_mask, adaptive, 0u);
  if (adaptive) {
    if (lane == 0u) {
      aggregate_partials[fragment_index] = 0ull;
      overflow_fragments[atomicAdd(overflow_count, 1u)] = fragment_index;
    }
    return;
  }

  std::uint32_t current_count = 0u;
  if constexpr (HasPending) {
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
      if (lane == 0u) {
        aggregate_partials[fragment_index] = 0ull;
        overflow_fragments[atomicAdd(overflow_count, 1u)] = fragment_index;
      }
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
    if (level == foundation_level) continue;
    unsigned long long descriptor_bits = 0ull;
    if (lane == 0u)
      descriptor_bits = descriptors[descriptor_index(q, level)].bits;
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
  class_overflow = __shfl_sync(full_mask, class_overflow, 0u);
  if (class_overflow) {
    if (lane == 0u) {
      aggregate_partials[fragment_index] = 0ull;
      overflow_fragments[atomicAdd(overflow_count, 1u)] = fragment_index;
    }
    return;
  }

  typename Aggregate::State local = Aggregate::identity();
  for (std::uint32_t index = lane; index < current_count; index += 32u) {
    const Row row = current[index];
    if ((row.flags & kTombstone) == 0u)
      local = Aggregate::consume(local, row);
  }
  unsigned long long foundation_bits = 0ull;
  if (lane == 0u && foundation_level < active_levels)
    foundation_bits =
        descriptors[descriptor_index(q, foundation_level)].bits;
  foundation_bits = __shfl_sync(full_mask, foundation_bits, 0u);
  const Descriptor foundation{foundation_bits};
  const Row *foundation_rows = arena + foundation.offset();
  const std::uint32_t base_section_count = foundation.count();
  std::uint32_t base_begin = 0u, base_end = 0u;
  if (lane == 0u && base_section_count) {
    base_begin = low == q_low
        ? 0u
        : lower_bound_rows(foundation_rows,
                           base_section_count, low_suffix);
    base_end = high == q_high
        ? base_section_count
        : upper_bound_rows(foundation_rows,
                           base_section_count, high_suffix);
  }
  base_begin = __shfl_sync(full_mask, base_begin, 0u);
  base_end = __shfl_sync(full_mask, base_end, 0u);
  if (!current_count) {
    for (std::uint32_t index = base_begin + lane; index < base_end;
         index += 32u)
      if ((foundation_rows[index].flags & kTombstone) == 0u)
        local = Aggregate::consume(local, foundation_rows[index]);
  } else {
    for (std::uint32_t index = base_begin + lane; index < base_end;
         index += 32u) {
      const Row row = foundation_rows[index];
      const std::uint32_t position =
          lower_bound_rows(current, current_count, row.key);
      if ((position == current_count || current[position].key != row.key) &&
          (row.flags & kTombstone) == 0u)
        local = Aggregate::consume(local, row);
    }
  }
  for (std::uint32_t offset = 16u; offset; offset >>= 1u)
    local += __shfl_down_sync(full_mask, local, offset);
  if (lane == 0u) {
    aggregate_partials[fragment_index] = local;
  }
}

template <bool HasPending>
__global__ void overflow_range_fragment_kernel(
    const std::uint32_t *overflow_fragments,
    const std::uint32_t *overflow_count,
    const RangeFragment *fragments,
    const std::uint32_t *query_low, const std::uint32_t *query_high,
    const Row *arena,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices, const std::uint32_t *raw_keys,
    const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, std::uint32_t active_levels,
    unsigned long long *aggregate_partials) {
  constexpr std::uint32_t kWarps = 4u;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp_in_block = threadIdx.x >> 5u;
  const std::uint32_t worker = blockIdx.x * kWarps + warp_in_block;
  const std::uint32_t workers = gridDim.x * kWarps;
  const std::uint32_t count = *overflow_count;
  for (std::uint32_t overflow_index = worker; overflow_index < count;
       overflow_index += workers) {
    std::uint32_t fragment_index = 0u, query = 0u, q = 0u;
    if (lane == 0u) {
      fragment_index = overflow_fragments[overflow_index];
      const RangeFragment fragment = fragments[fragment_index];
      query = fragment.query;
      q = fragment.quotient;
    }
    constexpr unsigned full_mask = 0xffffffffu;
    fragment_index = __shfl_sync(full_mask, fragment_index, 0u);
    query = __shfl_sync(full_mask, query, 0u);
    q = __shfl_sync(full_mask, q, 0u);
    const std::uint32_t q_low = q << 16u;
    const std::uint32_t q_high = q_low | 0xffffu;
    const std::uint32_t low = max(query_low[query], q_low);
    const std::uint32_t high = min(query_high[query], q_high);
    const unsigned long long value = warp_sum_visible_by_verification(
        q, low, high, raw_keys, raw_payloads, raw_offsets, batch_stride,
        HasPending ? pending_batches : 0u, arena, descriptors,
        route_headers, route_slices, active_levels);
    if (lane == 0u) aggregate_partials[fragment_index] = value;
  }
}

__global__ void gather_raw_batch_kernel(
    const std::uint32_t *sorted_keys, const std::uint32_t *sorted_ids,
    const std::uint32_t *values, std::uint32_t count, std::uint32_t batch_slot,
    bool tombstone, std::uint32_t *destination_keys,
    RawPayload *destination_payloads,
    std::uint64_t *batch_signatures,
    std::uint64_t *epoch_signatures) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t original = sorted_ids[i];
  destination_keys[i] = sorted_keys[i];
  destination_payloads[i] = make_raw_payload(
      tombstone ? 0u : values[original],
      (batch_slot << kBatchPositionBits) | original, tombstone);
  const std::uint32_t key = sorted_keys[i];
  const std::uint32_t first = key * 0x9e3779b1u;
  const std::uint32_t second = (key ^ (key >> 16u)) * 0x85ebca6bu;
  const std::uint64_t bits =
      (1ull << (first >> 26u)) | (1ull << (second >> 26u));
  const unsigned active = __activemask();
  const std::uint32_t quotient = key >> 16u;
  const unsigned peers = __match_any_sync(active, quotient);
  const std::uint32_t low = __reduce_or_sync(
      peers, static_cast<std::uint32_t>(bits));
  const std::uint32_t high = __reduce_or_sync(
      peers, static_cast<std::uint32_t>(bits >> 32u));
  if ((threadIdx.x & 31u) == static_cast<std::uint32_t>(__ffs(peers) - 1)) {
    const std::uint64_t aggregate =
        std::uint64_t{low} | (std::uint64_t{high} << 32u);
    atomicOr(reinterpret_cast<unsigned long long *>(
                 batch_signatures + quotient),
             static_cast<unsigned long long>(aggregate));
    atomicOr(reinterpret_cast<unsigned long long *>(
                 epoch_signatures + quotient),
             static_cast<unsigned long long>(aggregate));
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
    std::uint32_t *reservation_ranks,
    std::uint64_t *batch_signatures) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t key = keys[i];
  const std::uint32_t quotient = key >> 16u;
  const unsigned active = __activemask();
  const unsigned peers = __match_any_sync(active, quotient);
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t leader = __ffs(peers) - 1u;
  std::uint32_t base = 0u;
  if (lane == leader)
    base = atomicAdd(quotient_counts + quotient, __popc(peers));
  base = __shfl_sync(peers, base, leader);
  const unsigned before = lane ? ((1u << lane) - 1u) : 0u;
  reservation_ranks[i] = base + __popc(peers & before);
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

__global__ void finalize_quotient_metadata_kernel(
    const std::uint32_t *sorted_keys, std::uint32_t count,
    std::uint32_t *offsets) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  const std::uint32_t key = q << 16u;
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = (lo + hi) >> 1u;
    if (sorted_keys[mid] < key) lo = mid + 1u;
    else hi = mid;
  }
  offsets[q] = lo;
  if (q + 1u == kQuotients) offsets[kQuotients] = count;
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

__global__ void publish_global_level_descriptors_kernel(
    const std::uint32_t *keys, std::uint32_t count,
    std::uint64_t arena_offset, std::uint32_t level,
    Descriptor *descriptors) {
  __shared__ std::uint32_t boundaries[kThreads + 1u];
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  boundaries[threadIdx.x] = q < kQuotients
      ? lower_bound_full_keys(keys, count, q << 16u) : count;
  if (threadIdx.x + 1u == blockDim.x) {
    const std::uint32_t next = min(q + 1u, kQuotients);
    boundaries[blockDim.x] = next < kQuotients
        ? lower_bound_full_keys(keys, count, next << 16u) : count;
  }
  __syncthreads();
  if (q >= kQuotients) return;
  for (std::uint32_t old = 0u; old < level; ++old)
    descriptors[descriptor_index(q, old)] = {};
  const std::uint32_t begin = boundaries[threadIdx.x];
  const std::uint32_t end = boundaries[threadIdx.x + 1u];
  descriptors[descriptor_index(q, level)] = begin == end
      ? Descriptor{} : Descriptor::make(arena_offset + begin, end - begin);
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

__global__ void prepare_foundation_build_kernel(
    const std::uint32_t *section_offsets, std::uint32_t *capacities) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > kQuotients) return;
  capacities[q] = q == kQuotients
      ? 0u
      : foundation_section_capacity(
            section_offsets[q + 1u] - section_offsets[q]);
}

__global__ void scatter_foundation_build_kernel(
    const std::uint32_t *keys, const Row *rows, std::uint32_t count,
    const std::uint32_t *section_offsets,
    const std::uint32_t *physical_offsets, Row *arena) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const std::uint32_t q = keys[index] >> 16u;
  arena[physical_offsets[q] + index - section_offsets[q]] = rows[index];
}

__global__ void publish_foundation_build_kernel(
    const std::uint32_t *section_offsets,
    const std::uint32_t *physical_offsets,
    const std::uint32_t *capacities, std::uint32_t level,
    Descriptor *descriptors, std::uint32_t *foundation_capacities) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  const std::uint32_t count = section_offsets[q + 1u] - section_offsets[q];
  descriptors[descriptor_index(q, level)] = count
      ? Descriptor::make(physical_offsets[q], count) : Descriptor{};
  foundation_capacities[q] = capacities[q];
}

__global__ void count_balanced_merge_work_kernel(
    const std::uint32_t *current_offsets, const Descriptor *descriptors,
    std::uint32_t source_level_limit, std::uint32_t *raw_counts) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  std::uint32_t count = current_offsets[q + 1u] - current_offsets[q];
  for (std::uint32_t level = 0u; level <= source_level_limit; ++level)
    count += descriptors[descriptor_index(q, level)].count();
  raw_counts[q] = count;
}

__device__ __forceinline__ std::uint32_t balanced_merge_prefix_count(
    std::uint32_t q, std::uint32_t suffix,
    const Row *current_rows, const std::uint32_t *current_offsets,
    const Row *arena, const RouteHeader *route_headers,
    const RouteSlice *route_slices,
    std::uint32_t source_level_limit) {
  const std::uint32_t current_begin = current_offsets[q];
  std::uint32_t result = lower_bound_rows(
      current_rows + current_begin,
      current_offsets[q + 1u] - current_begin, suffix);
  for (std::uint32_t level = 0u; level <= source_level_limit; ++level) {
    const RouteHeader header =
        route_headers[descriptor_index(q, level)];
    for (std::uint32_t route_index = 0u;
         route_index < header.count; ++route_index) {
      const RouteSlice route = route_slices[header.begin + route_index];
      if (suffix <= route.suffix_begin) continue;
      if (suffix >= route.suffix_end) {
        result += route.rows.count();
        continue;
      }
      result += lower_bound_rows(arena + route.rows.offset(),
                                 route.rows.count(), suffix);
    }
  }
  return result;
}

__global__ void count_adaptive_location_work_kernel(
    const BalancedMergeJob *locations, std::uint32_t location_count,
    const Row *current_rows, const std::uint32_t *current_offsets,
    const Row *arena, const RouteHeader *route_headers,
    const RouteSlice *route_slices, std::uint32_t source_level_limit,
    std::uint32_t *work_counts) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= location_count) return;
  const BalancedMergeJob location = locations[index];
  std::uint32_t count = 0u;
  for (std::uint32_t q = location.quotient_begin;
       q < location.quotient_end; ++q) {
    const std::uint64_t q_begin = std::uint64_t{q} << 16u;
    const std::uint32_t low = static_cast<std::uint32_t>(
        location.key_begin > q_begin ? location.key_begin - q_begin : 0u);
    const std::uint32_t high = static_cast<std::uint32_t>(
        min(location.key_end - q_begin, std::uint64_t{1} << 16u));
    count += balanced_merge_prefix_count(
        q, high, current_rows, current_offsets, arena, route_headers,
        route_slices, source_level_limit) -
        balanced_merge_prefix_count(
            q, low, current_rows, current_offsets, arena, route_headers,
            route_slices, source_level_limit);
  }
  work_counts[index] = count;
}

__global__ void plan_crowded_merge_ranges_kernel(
    const std::uint32_t *hot_sections, std::uint32_t hot_count,
    const std::uint32_t *hot_lows, const std::uint32_t *hot_highs,
    const std::uint32_t *hot_raw_counts, const Row *current_rows,
    const std::uint32_t *current_offsets, const Row *arena,
    const RouteHeader *route_headers, const RouteSlice *route_slices,
    std::uint32_t source_level_limit,
    const std::uint32_t *range_counts,
    const std::uint32_t *range_bases,
    std::uint32_t *range_boundaries) {
  const std::uint32_t hot_index = blockIdx.x;
  if (hot_index >= hot_count) return;
  const std::uint32_t q = hot_sections[hot_index];
  const std::uint32_t raw = hot_raw_counts[hot_index];
  const std::uint32_t low_limit = hot_lows[hot_index];
  const std::uint32_t high_limit = hot_highs[hot_index];
  const std::uint32_t ranges = range_counts[hot_index];
  const std::size_t base = range_bases[hot_index];
  if (threadIdx.x == 0u) {
    range_boundaries[base] = low_limit;
    range_boundaries[base + ranges] = high_limit;
  }
  const std::uint32_t prefix_before = balanced_merge_prefix_count(
      q, low_limit, current_rows, current_offsets, arena, route_headers,
      route_slices, source_level_limit);
  for (std::uint32_t range = threadIdx.x + 1u;
       range < ranges; range += blockDim.x) {
    const std::uint32_t target = prefix_before + static_cast<std::uint32_t>(
        (std::uint64_t{raw} * range + ranges - 1u) / ranges);
    std::uint32_t low = low_limit + 1u;
    std::uint32_t high = high_limit;
    while (low < high) {
      const std::uint32_t middle = (low + high) >> 1u;
      if (balanced_merge_prefix_count(
              q, middle, current_rows, current_offsets, arena,
              route_headers, route_slices, source_level_limit) < target)
        low = middle + 1u;
      else
        high = middle;
    }
    range_boundaries[base + range] = low;
  }
}

__device__ __forceinline__ Row load_foundation_candidate(
    std::uint32_t q, std::uint32_t ordinal,
    const Row *current_rows, const std::uint32_t *current_offsets,
    const Row *arena, const Descriptor *descriptors,
    std::uint32_t foundation_level, std::uint32_t &age) {
  const std::uint32_t current_begin = current_offsets[q];
  const std::uint32_t current_count =
      current_offsets[q + 1u] - current_begin;
  if (ordinal < current_count) {
    age = 0u;
    return current_rows[current_begin + ordinal];
  }
  ordinal -= current_count;
  for (std::uint32_t level = 0u; level <= foundation_level; ++level) {
    const Descriptor descriptor = descriptors[descriptor_index(q, level)];
    if (ordinal < descriptor.count()) {
      age = level + 1u;
      return arena[descriptor.offset() + ordinal];
    }
    ordinal -= descriptor.count();
  }
  age = 0u;
  return {};
}

__device__ __forceinline__ std::uint32_t foundation_raw_count(
    std::uint32_t q, const std::uint32_t *current_offsets,
    const Descriptor *descriptors, std::uint32_t foundation_level) {
  std::uint32_t count = current_offsets[q + 1u] - current_offsets[q];
  for (std::uint32_t level = 0u; level <= foundation_level; ++level)
    count += descriptors[descriptor_index(q, level)].count();
  return count;
}

__device__ __forceinline__ std::uint32_t warp_merge_foundation_sources(
    const Row *candidates, const std::uint32_t *source_offsets,
    std::uint32_t sources, std::uint32_t low, std::uint32_t high,
    bool keep_tombstones, std::uint16_t *resolved_indices,
    std::uint32_t output_base) {
  constexpr unsigned mask = 0xffffffffu;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t source_a = lane;
  const std::uint32_t source_b = lane + 32u;
  std::uint32_t cursor_a{}, end_a{}, cursor_b{}, end_b{};
  if (source_a < sources) {
    const std::uint32_t base = source_offsets[source_a];
    const std::uint32_t count = source_offsets[source_a + 1u] - base;
    cursor_a = base + lower_bound_rows(candidates + base, count, low);
    end_a = base + lower_bound_rows(candidates + base, count, high);
  }
  if (source_b < sources) {
    const std::uint32_t base = source_offsets[source_b];
    const std::uint32_t count = source_offsets[source_b + 1u] - base;
    cursor_b = base + lower_bound_rows(candidates + base, count, low);
    end_b = base + lower_bound_rows(candidates + base, count, high);
  }

  std::uint32_t output_count = 0u;
  while (true) {
    const std::uint32_t key_a = cursor_a < end_a
        ? candidates[cursor_a].key : 0x10000u;
    const std::uint32_t key_b = cursor_b < end_b
        ? candidates[cursor_b].key : 0x10000u;
    const std::uint32_t tagged_a = key_a < 0x10000u
        ? (key_a << kMergeSourceBits) | source_a : kInvalid;
    const std::uint32_t tagged_b = key_b < 0x10000u
        ? (key_b << kMergeSourceBits) | source_b : kInvalid;
    std::uint32_t winner = min(tagged_a, tagged_b);
    for (std::uint32_t offset = 16u; offset; offset >>= 1u)
      winner = min(winner, __shfl_down_sync(mask, winner, offset));
    winner = __shfl_sync(mask, winner, 0u);
    if (winner == kInvalid) break;

    const std::uint32_t winner_key = winner >> kMergeSourceBits;
    const std::uint32_t winner_source =
        winner & ((1u << kMergeSourceBits) - 1u);
    std::uint32_t winner_index = kInvalid;
    if (source_a == winner_source) winner_index = cursor_a;
    if (source_b == winner_source) winner_index = cursor_b;
    winner_index =
        __shfl_sync(mask, winner_index, winner_source & 31u);
    const Row row = candidates[winner_index];
    if (lane == 0u &&
        (keep_tombstones || (row.flags & kTombstone) == 0u)) {
      resolved_indices[output_base + output_count] =
          static_cast<std::uint16_t>(winner_index);
      ++output_count;
    }
    if (cursor_a < end_a && candidates[cursor_a].key == winner_key)
      ++cursor_a;
    if (cursor_b < end_b && candidates[cursor_b].key == winner_key)
      ++cursor_b;
  }
  return __shfl_sync(mask, output_count, 0u);
}

__global__ void compact_normal_foundation_sections_kernel(
    const BalancedMergeJob *jobs, std::uint32_t job_count,
    const std::uint32_t *raw_counts,
    const Row *current_rows, const std::uint32_t *current_offsets,
    Row *arena, const Descriptor *descriptors,
    std::uint32_t source_level_limit,
    bool keep_tombstones, bool reserve_output_capacity,
    bool reuse_existing_sections,
    const std::uint32_t *foundation_capacities,
    unsigned long long *next_output_offset,
    unsigned long long output_limit,
    Descriptor *next_descriptors, std::uint32_t *next_capacities,
    const RouteHeader *next_route_headers, RouteSlice *route_slices,
    std::uint32_t *section_output_counts,
    std::uint32_t *total_output_count, std::uint32_t *overflow_flag) {
  constexpr std::uint32_t kWarps = kFoundationCompactionThreads / 32u;
  constexpr unsigned mask = 0xffffffffu;
  __shared__ Row candidates[kFoundationSectionCapacity];
  __shared__ std::uint16_t resolved_indices[kFoundationSectionCapacity];
  __shared__ std::uint32_t quotient_offsets[
      kBalancedMergeMaximumQuotients + 1u];
  __shared__ std::uint32_t quotient_warp_counts[kWarps];
  __shared__ std::uint32_t quotient_first_warp[kWarps];
  __shared__ std::uint32_t warp_quotients[kWarps];
  __shared__ std::uint32_t warp_bands[kWarps];
  __shared__ std::uint32_t warp_raw_counts[kWarps];
  __shared__ std::uint32_t warp_raw_bases[kWarps];
  __shared__ std::uint32_t warp_output_counts[kWarps];
  __shared__ std::uint32_t warp_output_bases[kWarps];
  __shared__ std::uint64_t quotient_output_bits[kWarps];

  const std::uint32_t job_index = blockIdx.x;
  if (job_index >= job_count) return;
  const BalancedMergeJob job = jobs[job_index];
  const std::uint32_t quotient_count =
      job.quotient_end - job.quotient_begin;
  if (threadIdx.x == 0u) {
    quotient_offsets[0u] = 0u;
    for (std::uint32_t local = 0u; local < quotient_count; ++local)
      quotient_offsets[local + 1u] = quotient_offsets[local] +
          raw_counts[job.quotient_begin + local];
  }
  __syncthreads();
  const std::uint32_t task_rows = quotient_offsets[quotient_count];
  if (task_rows > kBalancedMergeTarget) {
    if (threadIdx.x == 0u) atomicExch(overflow_flag, 1u);
    return;
  }
  for (std::uint32_t index = threadIdx.x; index < task_rows;
       index += blockDim.x) {
    std::uint32_t low = 0u, high = quotient_count;
    while (low + 1u < high) {
      const std::uint32_t middle = (low + high) >> 1u;
      if (quotient_offsets[middle] <= index)
        low = middle;
      else
        high = middle;
    }
    const std::uint32_t q = job.quotient_begin + low;
    std::uint32_t age{};
    candidates[index] = load_foundation_candidate(
        q, index - quotient_offsets[low], current_rows, current_offsets,
        arena, descriptors, source_level_limit, age);
  }
  __syncthreads();

  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t sources = source_level_limit + 2u;
  if (quotient_count <= kWarps) {
    if (threadIdx.x == 0u) {
      for (std::uint32_t local = 0u; local < quotient_count; ++local)
        quotient_warp_counts[local] = 1u;
      for (std::uint32_t remaining = kWarps - quotient_count;
           remaining; --remaining) {
        std::uint32_t best = 0u;
        for (std::uint32_t local = 1u; local < quotient_count; ++local) {
          const std::uint32_t best_rows =
              quotient_offsets[best + 1u] - quotient_offsets[best];
          const std::uint32_t local_rows =
              quotient_offsets[local + 1u] - quotient_offsets[local];
          if (std::uint64_t{local_rows} * quotient_warp_counts[best] >
              std::uint64_t{best_rows} * quotient_warp_counts[local])
            best = local;
        }
        ++quotient_warp_counts[best];
      }
      std::uint32_t next_warp = 0u;
      for (std::uint32_t local = 0u; local < quotient_count; ++local) {
        quotient_first_warp[local] = next_warp;
        for (std::uint32_t band = 0u;
             band < quotient_warp_counts[local]; ++band) {
          warp_quotients[next_warp] = local;
          warp_bands[next_warp] = band;
          ++next_warp;
        }
      }
    }
    __syncthreads();
    const std::uint32_t local = warp_quotients[warp];
    const std::uint32_t q = job.quotient_begin + local;
    const std::uint32_t q_base = quotient_offsets[local];
    const std::uint32_t bands = quotient_warp_counts[local];
    const std::uint32_t band = warp_bands[warp];
    const std::uint32_t suffix_low = static_cast<std::uint32_t>(
        (std::uint64_t{1u << 16u} * band) / bands);
    const std::uint32_t suffix_high = static_cast<std::uint32_t>(
        (std::uint64_t{1u << 16u} * (band + 1u)) / bands);
    if (lane == 0u) {
      std::uint32_t band_rows = 0u;
      for (std::uint32_t source = 0u; source < sources; ++source) {
        std::uint32_t source_base = q_base;
        if (source) {
          source_base += current_offsets[q + 1u] - current_offsets[q];
          for (std::uint32_t level = 0u; level + 1u < source; ++level)
            source_base +=
                descriptors[descriptor_index(q, level)].count();
        }
        const std::uint32_t count = source == 0u
            ? current_offsets[q + 1u] - current_offsets[q]
            : descriptors[descriptor_index(q, source - 1u)].count();
        band_rows += lower_bound_rows(
            candidates + source_base, count, suffix_high) -
            lower_bound_rows(candidates + source_base, count, suffix_low);
      }
      warp_raw_counts[warp] = band_rows;
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
      for (std::uint32_t local_q = 0u; local_q < quotient_count;
           ++local_q) {
        std::uint32_t raw_base = 0u;
        const std::uint32_t first = quotient_first_warp[local_q];
        for (std::uint32_t index = 0u;
             index < quotient_warp_counts[local_q]; ++index) {
          warp_raw_bases[first + index] = raw_base;
          raw_base += warp_raw_counts[first + index];
        }
      }
    }
    __syncthreads();

    const std::uint32_t source_a = lane;
    const std::uint32_t source_b = lane + 32u;
    std::uint32_t cursor_a = q_base, end_a = q_base;
    std::uint32_t cursor_b = q_base, end_b = q_base;
    if (source_a < sources) {
      std::uint32_t source_base = q_base;
      if (source_a) {
        source_base += current_offsets[q + 1u] - current_offsets[q];
        for (std::uint32_t level = 0u; level + 1u < source_a; ++level)
          source_base += descriptors[descriptor_index(q, level)].count();
      }
      const std::uint32_t count = source_a == 0u
          ? current_offsets[q + 1u] - current_offsets[q]
          : descriptors[descriptor_index(q, source_a - 1u)].count();
      cursor_a = source_base + lower_bound_rows(
          candidates + source_base, count, suffix_low);
      end_a = source_base + lower_bound_rows(
          candidates + source_base, count, suffix_high);
    }
    if (source_b < sources) {
      std::uint32_t source_base = q_base +
          current_offsets[q + 1u] - current_offsets[q];
      for (std::uint32_t level = 0u; level + 1u < source_b; ++level)
        source_base += descriptors[descriptor_index(q, level)].count();
      const std::uint32_t count =
          descriptors[descriptor_index(q, source_b - 1u)].count();
      cursor_b = source_base + lower_bound_rows(
          candidates + source_base, count, suffix_low);
      end_b = source_base + lower_bound_rows(
          candidates + source_base, count, suffix_high);
    }
    std::uint32_t output_count = 0u;
    while (true) {
      const std::uint32_t key_a = cursor_a < end_a
          ? candidates[cursor_a].key : 0x10000u;
      const std::uint32_t key_b = cursor_b < end_b
          ? candidates[cursor_b].key : 0x10000u;
      const std::uint32_t tagged_a = key_a < 0x10000u
          ? (key_a << kMergeSourceBits) | source_a : kInvalid;
      const std::uint32_t tagged_b = key_b < 0x10000u
          ? (key_b << kMergeSourceBits) | source_b : kInvalid;
      std::uint32_t winner = min(tagged_a, tagged_b);
      for (std::uint32_t offset = 16u; offset; offset >>= 1u)
        winner = min(winner, __shfl_down_sync(mask, winner, offset));
      winner = __shfl_sync(mask, winner, 0u);
      if (winner == kInvalid) break;
      const std::uint32_t winner_key = winner >> kMergeSourceBits;
      const std::uint32_t winner_source =
          winner & ((1u << kMergeSourceBits) - 1u);
      std::uint32_t winner_index = kInvalid;
      if (source_a == winner_source) winner_index = cursor_a;
      if (source_b == winner_source) winner_index = cursor_b;
      winner_index =
          __shfl_sync(mask, winner_index, winner_source & 31u);
      const Row row = candidates[winner_index];
      if (lane == 0u &&
          (keep_tombstones || (row.flags & kTombstone) == 0u))
        resolved_indices[q_base + warp_raw_bases[warp] + output_count++] =
            static_cast<std::uint16_t>(winner_index);
      if (cursor_a < end_a && key_a == winner_key) ++cursor_a;
      if (cursor_b < end_b && key_b == winner_key) ++cursor_b;
    }
    if (lane == 0u) warp_output_counts[warp] = output_count;
    __syncthreads();
    if (threadIdx.x == 0u) {
      for (std::uint32_t local_q = 0u; local_q < quotient_count;
           ++local_q) {
        const std::uint32_t first = quotient_first_warp[local_q];
        std::uint32_t total = 0u;
        for (std::uint32_t index = 0u;
             index < quotient_warp_counts[local_q]; ++index) {
          warp_output_bases[first + index] = total;
          total += warp_output_counts[first + index];
        }
        const std::uint32_t output_q = job.quotient_begin + local_q;
        Descriptor output_descriptor{};
        std::uint32_t output_capacity = 0u;
        const std::uint32_t old_capacity = reuse_existing_sections
            ? foundation_capacities[output_q] : 0u;
        if (total && reuse_existing_sections && total <= old_capacity) {
          const Descriptor old_descriptor = descriptors[
              descriptor_index(output_q, source_level_limit)];
          output_descriptor =
              Descriptor::make(old_descriptor.offset(), total);
          output_capacity = old_capacity;
        } else if (total) {
          output_capacity = reuse_existing_sections
              ? foundation_section_capacity(total) : total;
          const unsigned long long offset = atomicAdd(
              next_output_offset,
              static_cast<unsigned long long>(output_capacity));
          if (offset + output_capacity > output_limit)
            atomicExch(overflow_flag, 1u);
          else
            output_descriptor = Descriptor::make(offset, total);
        }
        quotient_output_bits[local_q] = output_descriptor.bits;
        next_descriptors[output_q] = output_descriptor;
        next_capacities[output_q] = output_capacity;
        section_output_counts[output_q] = total;
        atomicAdd(total_output_count, total);
      }
    }
    __syncthreads();
    const Descriptor output_descriptor{quotient_output_bits[local]};
    if (output_descriptor.count()) {
      const std::uint16_t *indices = resolved_indices + q_base +
          warp_raw_bases[warp];
      Row *output = arena + output_descriptor.offset() +
          warp_output_bases[warp];
      for (std::uint32_t index = lane;
           index < warp_output_counts[warp]; index += 32u)
        output[index] = candidates[indices[index]];
    }
    return;
  }
  for (std::uint32_t local = warp; local < quotient_count;
       local += kWarps) {
    const std::uint32_t q = job.quotient_begin + local;
    const std::uint32_t q_base = quotient_offsets[local];
    const std::uint32_t q_end = quotient_offsets[local + 1u];
    if (q_base == q_end) continue;

    const std::uint32_t source_a = lane;
    const std::uint32_t source_b = lane + 32u;
    std::uint32_t cursor_a = q_base, end_a = q_base;
    std::uint32_t cursor_b = q_base, end_b = q_base;
    if (source_a < sources) {
      std::uint32_t base = q_base;
      if (source_a) {
        base += current_offsets[q + 1u] - current_offsets[q];
        for (std::uint32_t level = 0u; level + 1u < source_a; ++level)
          base += descriptors[descriptor_index(q, level)].count();
      }
      const std::uint32_t count = source_a == 0u
          ? current_offsets[q + 1u] - current_offsets[q]
          : descriptors[descriptor_index(q, source_a - 1u)].count();
      cursor_a = base;
      end_a = base + count;
    }
    if (source_b < sources) {
      std::uint32_t base = q_base +
          current_offsets[q + 1u] - current_offsets[q];
      for (std::uint32_t level = 0u; level + 1u < source_b; ++level)
        base += descriptors[descriptor_index(q, level)].count();
      const std::uint32_t count =
          descriptors[descriptor_index(q, source_b - 1u)].count();
      cursor_b = base;
      end_b = base + count;
    }

    std::uint32_t output_count = 0u;
    while (true) {
      const std::uint32_t key_a = cursor_a < end_a
          ? candidates[cursor_a].key : 0x10000u;
      const std::uint32_t key_b = cursor_b < end_b
          ? candidates[cursor_b].key : 0x10000u;
      const std::uint32_t tagged_a = key_a < 0x10000u
          ? (key_a << kMergeSourceBits) | source_a : kInvalid;
      const std::uint32_t tagged_b = key_b < 0x10000u
          ? (key_b << kMergeSourceBits) | source_b : kInvalid;
      std::uint32_t winner = min(tagged_a, tagged_b);
      for (std::uint32_t offset = 16u; offset; offset >>= 1u)
        winner = min(winner, __shfl_down_sync(mask, winner, offset));
      winner = __shfl_sync(mask, winner, 0u);
      if (winner == kInvalid) break;
      const std::uint32_t winner_key = winner >> kMergeSourceBits;
      const std::uint32_t winner_source =
          winner & ((1u << kMergeSourceBits) - 1u);
      std::uint32_t winner_index = kInvalid;
      if (source_a == winner_source) winner_index = cursor_a;
      if (source_b == winner_source) winner_index = cursor_b;
      winner_index =
          __shfl_sync(mask, winner_index, winner_source & 31u);
      const Row row = candidates[winner_index];
      if (lane == 0u &&
          (keep_tombstones || (row.flags & kTombstone) == 0u))
        resolved_indices[q_base + output_count++] =
            static_cast<std::uint16_t>(winner_index);
      if (cursor_a < end_a && key_a == winner_key) ++cursor_a;
      if (cursor_b < end_b && key_b == winner_key) ++cursor_b;
    }
    output_count = __shfl_sync(mask, output_count, 0u);

    std::uint64_t output_bits = 0u;
    if (lane == 0u) {
      std::uint32_t output_capacity = 0u;
      Descriptor output_descriptor{};
      const Descriptor old_descriptor = reuse_existing_sections
          ? descriptors[descriptor_index(q, source_level_limit)]
          : Descriptor{};
      const std::uint32_t old_capacity = reuse_existing_sections
          ? foundation_capacities[q] : 0u;
      if (output_count && reuse_existing_sections &&
          output_count <= old_capacity) {
        output_descriptor =
            Descriptor::make(old_descriptor.offset(), output_count);
        output_capacity = old_capacity;
      } else if (output_count) {
        output_capacity = reuse_existing_sections
            ? foundation_section_capacity(output_count) : output_count;
        const unsigned long long offset = atomicAdd(
            next_output_offset,
            static_cast<unsigned long long>(output_capacity));
        if (offset + output_capacity > output_limit) {
          atomicExch(overflow_flag, 1u);
        } else {
          output_descriptor = Descriptor::make(offset, output_count);
        }
      }
      next_descriptors[q] = output_descriptor;
      next_capacities[q] = output_capacity;
      section_output_counts[q] = output_count;
      atomicAdd(total_output_count, output_count);
      output_bits = output_descriptor.bits;
    }
    const std::uint32_t output_low = __shfl_sync(
        mask, static_cast<std::uint32_t>(output_bits), 0u);
    const std::uint32_t output_high = __shfl_sync(
        mask, static_cast<std::uint32_t>(output_bits >> 32u), 0u);
    output_bits = std::uint64_t{output_low} |
        (std::uint64_t{output_high} << 32u);
    const Descriptor output_descriptor{output_bits};
    if (output_descriptor.count()) {
      Row *output = arena + output_descriptor.offset();
      for (std::uint32_t index = lane; index < output_count; index += 32u)
        output[index] = candidates[resolved_indices[q_base + index]];
    }
  }
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

__global__ void prepare_balanced_pull_slices_kernel(
    const BalancedMergeJob *jobs, std::uint32_t job_count,
    const Row *current_rows,
    const std::uint32_t *current_offsets,
    const Row *arena, const RouteHeader *route_headers,
    const RouteSlice *route_slices, std::uint32_t source_level_limit,
    PullSlice *slices) {
  const std::uint32_t job_index = blockIdx.x;
  if (job_index >= job_count || threadIdx.x != 0u) return;
  const BalancedMergeJob job = jobs[job_index];
  const std::uint32_t sources = source_level_limit + 2u;
  std::uint32_t slot = 0u;
  std::uint32_t candidate = 0u;
  for (std::uint32_t source = 0u; source < sources; ++source) {
    for (std::uint32_t q = job.quotient_begin;
         q < job.quotient_end; ++q) {
      const std::uint64_t q_begin = std::uint64_t{q} << 16u;
      const std::uint32_t low = static_cast<std::uint32_t>(
          job.key_begin > q_begin ? job.key_begin - q_begin : 0u);
      const std::uint32_t high = static_cast<std::uint32_t>(
          min(job.key_end - q_begin, std::uint64_t{1} << 16u));
      const std::uint16_t tag = static_cast<std::uint16_t>(
          (source << 9u) | (q - job.quotient_begin));
      if (source == 0u) {
        const std::uint32_t q_offset = current_offsets[q];
        const std::uint32_t q_count =
            current_offsets[q + 1u] - q_offset;
        const std::uint32_t begin = lower_bound_rows(
            current_rows + q_offset, q_count, low);
        const std::uint32_t end = lower_bound_rows(
            current_rows + q_offset, q_count, high);
        slices[job.slice_begin + slot++] = {
            q_offset + begin, end - begin,
            static_cast<std::uint16_t>(candidate), tag};
        candidate += end - begin;
        continue;
      }
      const RouteHeader header = route_headers[
          descriptor_index(q, source - 1u)];
      if (!header.count) {
        slices[job.slice_begin + slot++] = {
            0u, 0u, static_cast<std::uint16_t>(candidate), tag};
        continue;
      }
      for (std::uint32_t route_index = 0u;
           route_index < header.count; ++route_index) {
        const RouteSlice route = route_slices[header.begin + route_index];
        const std::uint32_t slice_low = max(low, route.suffix_begin);
        const std::uint32_t slice_high = min(high, route.suffix_end);
        std::uint32_t begin = 0u, end = 0u;
        if (slice_low < slice_high) {
          begin = lower_bound_rows(
              arena + route.rows.offset(), route.rows.count(), slice_low);
          end = lower_bound_rows(
              arena + route.rows.offset(), route.rows.count(), slice_high);
        }
        slices[job.slice_begin + slot++] = {
            route.rows.offset() + begin, end - begin,
            static_cast<std::uint16_t>(candidate), tag};
        candidate += end - begin;
      }
    }
  }
  slices[job.slice_begin + job.slice_count] = {
      0u, 0u, static_cast<std::uint16_t>(candidate), 0u};
}

__global__ void compact_balanced_merge_jobs_kernel(
    BalancedMergeJob *jobs, std::uint32_t job_count,
    const std::uint32_t *raw_counts,
    const PullSlice *pull_slices,
    const Row *current_rows, const std::uint32_t *current_offsets,
    Row *arena, const Descriptor *descriptors,
    std::uint32_t source_level_limit,
    bool keep_tombstones, bool reserve_output_capacity,
    bool reuse_existing_sections,
    const std::uint32_t *foundation_capacities,
    unsigned long long *next_output_offset,
    unsigned long long output_limit,
    Descriptor *next_descriptors, std::uint32_t *next_capacities,
    const RouteHeader *next_route_headers, RouteSlice *route_slices,
    std::uint32_t *section_output_counts,
    std::uint32_t *total_output_count, std::uint32_t *overflow_flag) {
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

  const std::uint32_t job_index = blockIdx.x;
  if (job_index >= job_count) return;
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

    bool reuse_page = reuse_existing_sections && job.reuse_existing_page;
    std::uint64_t old_page_offset = job.existing_offset;
    std::uint32_t old_page_capacity = job.existing_capacity;
    if (reuse_page && !old_page_capacity && quotient_count == 1u) {
      const std::uint32_t q = job.quotient_begin;
      old_page_offset = descriptors[
          descriptor_index(q, source_level_limit)].offset();
      old_page_capacity = foundation_capacities[q];
    }
    if (!old_page_capacity || task_output_count > old_page_capacity)
      reuse_page = false;
    const std::uint32_t page_capacity = task_output_count
        ? (reuse_page ? old_page_capacity
                      : (reserve_output_capacity
                             ? foundation_section_capacity(task_output_count)
                             : task_output_count))
        : 0u;
    bool output_valid = true;
    unsigned long long page_offset = reuse_page ? old_page_offset : 0u;
    if (!reuse_page && page_capacity) {
      page_offset = atomicAdd(next_output_offset,
                              static_cast<unsigned long long>(page_capacity));
      if (page_offset + page_capacity > output_limit) {
        atomicExch(overflow_flag, 1u);
        output_valid = false;
      }
    }

    for (std::uint32_t local = 0u; local < quotient_count; ++local) {
      const std::uint32_t output_q = job.quotient_begin + local;
      const std::uint32_t output_count = quotient_output_counts[local];
      Descriptor output_descriptor{};
      if (output_count && output_valid)
        output_descriptor = Descriptor::make(
            page_offset + quotient_output_offsets[local], output_count);
      quotient_output_bits[local] = output_descriptor.bits;
      next_descriptors[output_q] = output_descriptor;
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
    jobs[job_index].reuse_existing_page =
        output_valid && page_capacity != 0u;
    atomicAdd(total_output_count, task_output_count);
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
}

template <bool Emit>
__global__ void resolve_hot_foundation_cells_kernel(
    const std::uint32_t *hot_sections, std::uint32_t hot_count,
    const Row *current_rows, const std::uint32_t *current_offsets,
    const Row *arena, const Descriptor *descriptors,
    std::uint32_t foundation_level,
    const std::uint32_t *staging_offsets,
    std::uint32_t staging_capacity,
    const std::uint32_t *cell_offsets, std::uint32_t *cell_counts,
    Row *staging_rows) {
  using BlockScan = cub::BlockScan<std::uint32_t, 256u>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::uint32_t seen[kFoundationCellKeys / 32u];
  __shared__ Row winners[kFoundationCellKeys];
  __shared__ std::uint32_t source_begin, source_end;
  const std::uint32_t task = blockIdx.x;
  const std::uint32_t hot_index = task / kFoundationCells;
  const std::uint32_t cell = task % kFoundationCells;
  if (hot_index >= hot_count) return;
  const std::uint32_t q = hot_sections[hot_index];
  const std::uint32_t low = cell * kFoundationCellKeys;
  const std::uint32_t high = low + kFoundationCellKeys;
  for (std::uint32_t word = threadIdx.x;
       word < kFoundationCellKeys / 32u; word += blockDim.x)
    seen[word] = 0u;
  __syncthreads();

  const std::uint32_t sources = foundation_level + 2u;
  for (std::uint32_t source = 0u; source < sources; ++source) {
    const Row *rows = nullptr;
    std::uint32_t count = 0u;
    if (source == 0u) {
      rows = current_rows + current_offsets[q];
      count = current_offsets[q + 1u] - current_offsets[q];
    } else {
      const Descriptor descriptor =
          descriptors[descriptor_index(q, source - 1u)];
      rows = arena + descriptor.offset();
      count = descriptor.count();
    }
    if (threadIdx.x == 0u) {
      source_begin = lower_bound_rows(rows, count, low);
      source_end = lower_bound_rows(rows, count, high);
    }
    __syncthreads();
    for (std::uint32_t index = source_begin + threadIdx.x;
         index < source_end; index += blockDim.x) {
      const Row row = rows[index];
      const std::uint32_t local = row.key - low;
      const std::uint32_t bit = 1u << (local & 31u);
      const std::uint32_t prior = atomicOr(seen + (local >> 5u), bit);
      if ((prior & bit) == 0u) winners[local] = row;
    }
    __syncthreads();
  }

  const std::uint32_t first = threadIdx.x * 2u;
  std::uint32_t local_live = 0u;
  bool live[2]{};
#pragma unroll
  for (std::uint32_t item = 0u; item < 2u; ++item) {
    const std::uint32_t local = first + item;
    const bool present = seen[local >> 5u] & (1u << (local & 31u));
    live[item] = present && (winners[local].flags & kTombstone) == 0u;
    local_live += live[item];
  }
  std::uint32_t thread_base{}, total_live{};
  BlockScan(scan_storage).ExclusiveSum(local_live, thread_base, total_live);
  const std::size_t cell_index =
      std::size_t{hot_index} * (kFoundationCells + 1u) + cell;
  if constexpr (!Emit) {
    if (threadIdx.x == 0u) cell_counts[cell_index] = total_live;
  } else {
    const std::uint32_t output_begin = cell_offsets[cell_index];
    const std::uint32_t staging_begin = staging_offsets[hot_index];
    std::uint32_t local_rank = 0u;
#pragma unroll
    for (std::uint32_t item = 0u; item < 2u; ++item)
      if (live[item]) {
        const std::uint32_t output = staging_begin + output_begin +
            thread_base + local_rank++;
        if (output < staging_capacity)
          staging_rows[output] = winners[first + item];
      }
  }
}

template <bool Emit>
__global__ void resolve_crowded_onchip_ranges_kernel(
    const std::uint32_t *hot_sections, std::uint32_t hot_count,
    const CrowdedMergeTask *tasks, std::uint32_t task_count,
    const std::uint32_t *range_counts,
    const std::uint32_t *range_bases,
    const std::uint32_t *range_boundaries,
    const Row *current_rows, const std::uint32_t *current_offsets,
    const Row *arena, const Descriptor *descriptors,
    std::uint32_t source_level_limit, bool keep_tombstones,
    const std::uint32_t *staging_offsets,
    std::uint32_t staging_capacity,
    const std::uint32_t *range_offsets,
    std::uint32_t *survivor_counts, Row *staging_rows,
    std::uint8_t *streaming_flags) {
  constexpr std::uint32_t kItemsPerThread =
      (kBalancedMergeWorkspace + kFoundationCompactionThreads - 1u) /
      kFoundationCompactionThreads;
  using BlockScan = cub::BlockScan<std::uint32_t,
                                   kFoundationCompactionThreads>;
  __shared__ typename BlockScan::TempStorage block_scan_storage;
  __shared__ Row candidates[kBalancedMergeWorkspace];
  __shared__ std::uint16_t merge_indices_a[kBalancedMergeWorkspace + 1u];
  __shared__ std::uint16_t merge_indices_b[kBalancedMergeWorkspace + 1u];
  __shared__ std::uint32_t tombstone_words[
      (kBalancedMergeWorkspace + 31u) / 32u];
  __shared__ std::uint32_t source_offsets[kMaximumLevels + 2u];
  __shared__ std::uint32_t source_begins[kMaximumLevels + 1u];
  __shared__ std::uint16_t run_offsets[kMaximumLevels + 1u];
  __shared__ std::uint16_t run_lengths[kMaximumLevels + 1u];
  __shared__ std::uint16_t run_sources[kMaximumLevels + 1u];
  __shared__ std::uint32_t raw_count_shared;
  __shared__ std::uint32_t run_count_shared;
  __shared__ std::uint32_t small_count_shared;
  __shared__ std::uint32_t largest_source_shared;
  __shared__ std::uint32_t largest_count_shared;

  const std::uint32_t slot = blockIdx.x;
  if (slot >= task_count) return;
  const CrowdedMergeTask task = tasks[slot];
  const std::uint32_t hot_index = task.hot_index;
  const std::uint32_t range = task.range;
  if (hot_index >= hot_count || range >= range_counts[hot_index]) return;
  const std::uint32_t q = hot_sections[hot_index];
  const std::size_t base = range_bases[hot_index];
  const std::uint32_t low = range_boundaries[base + range];
  const std::uint32_t high = range_boundaries[base + range + 1u];
  const std::uint32_t sources = source_level_limit + 2u;

  if (threadIdx.x == 0u) {
    source_offsets[0u] = 0u;
    for (std::uint32_t source = 0u; source < sources; ++source) {
      const Row *rows = nullptr;
      std::uint32_t count = 0u;
      if (source == 0u) {
        const std::uint32_t begin = current_offsets[q];
        rows = current_rows + begin;
        count = current_offsets[q + 1u] - begin;
      } else {
        const Descriptor descriptor =
            descriptors[descriptor_index(q, source - 1u)];
        rows = arena + descriptor.offset();
        count = descriptor.count();
      }
      const std::uint32_t begin = lower_bound_rows(rows, count, low);
      const std::uint32_t end = lower_bound_rows(rows, count, high);
      source_begins[source] = begin;
      source_offsets[source + 1u] =
          source_offsets[source] + end - begin;
    }
    raw_count_shared = source_offsets[sources];
    streaming_flags[slot] =
        raw_count_shared > kBalancedMergeWorkspace;
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
  for (std::uint32_t word = threadIdx.x;
       word < (kBalancedMergeWorkspace + 31u) / 32u;
       word += blockDim.x)
    tombstone_words[word] = 0u;
  __syncthreads();
  const std::uint32_t raw_count = raw_count_shared;
  if (raw_count > kBalancedMergeWorkspace) return;

  for (std::uint32_t ordinal = threadIdx.x; ordinal < raw_count;
       ordinal += blockDim.x) {
    std::uint32_t source = 0u;
    while (source_offsets[source + 1u] <= ordinal) ++source;
    const std::uint32_t source_index =
        source_begins[source] + ordinal - source_offsets[source];
    if (source == 0u) {
      candidates[ordinal] = current_rows[
          current_offsets[q] + source_index];
    } else {
      const Descriptor descriptor =
          descriptors[descriptor_index(q, source - 1u)];
      candidates[ordinal] = arena[descriptor.offset() + source_index];
    }
    if (candidates[ordinal].flags & kTombstone)
      atomicOr(tombstone_words + (ordinal >> 5u),
               1u << (ordinal & 31u));
    // This range belongs to one quotient, so the temporary ordering tag only
    // needs the source age.  The persistent tombstone bit is restored later.
    candidates[ordinal].flags = static_cast<std::uint16_t>(source);
  }
  __syncthreads();

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

  if (small_count_shared && largest_count_shared) {
    const std::uint16_t *input =
        input_is_a ? merge_indices_a : merge_indices_b;
    std::uint16_t *output =
        input_is_a ? merge_indices_b : merge_indices_a;
    const std::uint16_t *left = input;
    const std::uint16_t *right = input + small_count_shared;
    std::uint32_t position = threadIdx.x * kItemsPerThread;
    const std::uint32_t output_end = min(
        position + kItemsPerThread, raw_count);
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
    if (index >= raw_count) continue;
    const std::uint16_t candidate_index = sorted_indices[index];
    const Row row = candidates[candidate_index];
    const bool first = index == 0u || row.key !=
        candidates[sorted_indices[index - 1u]].key;
    const bool tombstone =
        (tombstone_words[candidate_index >> 5u] &
         (1u << (candidate_index & 31u))) != 0u;
    live[item] = first && (keep_tombstones || !tombstone);
    local_live += live[item];
  }
  std::uint32_t thread_base{}, total{};
  BlockScan(block_scan_storage).ExclusiveSum(
      local_live, thread_base, total);
  if (threadIdx.x == 0u && !Emit)
    survivor_counts[base + range] = total;
  if constexpr (Emit) {
    const std::uint32_t output_base = staging_offsets[hot_index] +
        range_offsets[base + range] + thread_base;
    std::uint32_t local_rank = 0u;
#pragma unroll
    for (std::uint32_t item = 0u; item < kItemsPerThread; ++item) {
      if (!live[item]) continue;
      const std::uint32_t index = threadIdx.x * kItemsPerThread + item;
      const std::uint16_t candidate_index = sorted_indices[index];
      Row row = candidates[candidate_index];
      const bool tombstone =
          (tombstone_words[candidate_index >> 5u] &
           (1u << (candidate_index & 31u))) != 0u;
      row.flags = tombstone ? kTombstone : 0u;
      const std::uint32_t output = output_base + local_rank++;
      if (output < staging_capacity) staging_rows[output] = row;
    }
  }
}

template <bool Emit>
__global__ void resolve_crowded_merge_ranges_kernel(
    const std::uint32_t *hot_sections, std::uint32_t hot_count,
    const CrowdedMergeTask *tasks, std::uint32_t task_count,
    const std::uint32_t *range_counts,
    const std::uint32_t *range_bases,
    const std::uint32_t *range_boundaries,
    const Row *current_rows, const std::uint32_t *current_offsets,
    const Row *arena, const Descriptor *descriptors,
    std::uint32_t source_level_limit, bool keep_tombstones,
    const std::uint32_t *staging_offsets,
    std::uint32_t staging_capacity,
    const std::uint32_t *range_offsets,
    std::uint32_t *survivor_counts, Row *staging_rows,
    const std::uint8_t *streaming_flags) {
  constexpr std::uint32_t kWarpsPerBlock = 4u;
  constexpr unsigned mask = 0xffffffffu;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t slot = blockIdx.x * kWarpsPerBlock + warp;
  if (slot >= task_count) return;
  const CrowdedMergeTask task = tasks[slot];
  const std::uint32_t hot_index = task.hot_index;
  const std::uint32_t range = task.range;
  if (hot_index >= hot_count || range >= range_counts[hot_index] ||
      !streaming_flags[slot]) return;
  const std::uint32_t q = hot_sections[hot_index];
  const std::size_t base = range_bases[hot_index];
  const std::uint32_t low = range_boundaries[base + range];
  const std::uint32_t high = range_boundaries[base + range + 1u];
  const std::uint32_t sources = source_level_limit + 2u;
  const std::uint32_t source_a = lane;
  const std::uint32_t source_b = lane + 32u;

  const Row *rows_a = nullptr, *rows_b = nullptr;
  std::uint32_t cursor_a = 0u, end_a = 0u;
  std::uint32_t cursor_b = 0u, end_b = 0u;
  if (source_a < sources) {
    std::uint32_t count = 0u;
    if (source_a == 0u) {
      cursor_a = current_offsets[q];
      count = current_offsets[q + 1u] - cursor_a;
      rows_a = current_rows + cursor_a;
      cursor_a = 0u;
    } else {
      const Descriptor descriptor =
          descriptors[descriptor_index(q, source_a - 1u)];
      rows_a = arena + descriptor.offset();
      count = descriptor.count();
    }
    cursor_a = lower_bound_rows(rows_a, count, low);
    end_a = lower_bound_rows(rows_a, count, high);
  }
  if (source_b < sources) {
    std::uint32_t count = 0u;
    if (source_b == 0u) {
      cursor_b = current_offsets[q];
      count = current_offsets[q + 1u] - cursor_b;
      rows_b = current_rows + cursor_b;
      cursor_b = 0u;
    } else {
      const Descriptor descriptor =
          descriptors[descriptor_index(q, source_b - 1u)];
      rows_b = arena + descriptor.offset();
      count = descriptor.count();
    }
    cursor_b = lower_bound_rows(rows_b, count, low);
    end_b = lower_bound_rows(rows_b, count, high);
  }

  std::uint32_t output_count = 0u;
  while (true) {
    const std::uint32_t key_a = cursor_a < end_a
        ? rows_a[cursor_a].key : 0x10000u;
    const std::uint32_t key_b = cursor_b < end_b
        ? rows_b[cursor_b].key : 0x10000u;
    const std::uint32_t tagged_a = key_a < 0x10000u
        ? (key_a << kMergeSourceBits) | source_a : kInvalid;
    const std::uint32_t tagged_b = key_b < 0x10000u
        ? (key_b << kMergeSourceBits) | source_b : kInvalid;
    std::uint32_t winner = min(tagged_a, tagged_b);
    for (std::uint32_t offset = 16u; offset; offset >>= 1u)
      winner = min(winner, __shfl_down_sync(mask, winner, offset));
    winner = __shfl_sync(mask, winner, 0u);
    if (winner == kInvalid) break;

    const std::uint32_t winner_key = winner >> kMergeSourceBits;
    const std::uint32_t winner_source =
        winner & ((1u << kMergeSourceBits) - 1u);
    std::uint32_t winner_index = kInvalid;
    if (source_a == winner_source) winner_index = cursor_a;
    if (source_b == winner_source) winner_index = cursor_b;
    winner_index =
        __shfl_sync(mask, winner_index, winner_source & 31u);
    if (lane == 0u) {
      Row row{};
      if (winner_source == 0u) {
        const std::uint32_t begin = current_offsets[q];
        row = current_rows[begin + winner_index];
      } else {
        const Descriptor descriptor =
            descriptors[descriptor_index(q, winner_source - 1u)];
        row = arena[descriptor.offset() + winner_index];
      }
      if (keep_tombstones || (row.flags & kTombstone) == 0u) {
        if constexpr (Emit) {
          const std::uint32_t output = staging_offsets[hot_index] +
              range_offsets[base + range] + output_count;
          if (output < staging_capacity) staging_rows[output] = row;
        }
        ++output_count;
      }
    }
    if (cursor_a < end_a && key_a == winner_key) ++cursor_a;
    if (cursor_b < end_b && key_b == winner_key) ++cursor_b;
  }
  if constexpr (!Emit) {
    if (lane == 0u) survivor_counts[base + range] = output_count;
  }
}

__global__ void scan_and_allocate_hot_foundation_kernel(
    const std::uint32_t *hot_sections, std::uint32_t hot_count,
    const std::uint32_t *range_counts,
    const std::uint32_t *range_bases,
    std::uint32_t *cell_counts, std::uint32_t *cell_offsets,
    const Descriptor *descriptors, std::uint32_t source_level_limit,
    bool reuse_existing_sections,
    const std::uint32_t *foundation_capacities,
    unsigned long long *next_output_offset,
    unsigned long long output_limit,
    std::uint32_t *hot_staging_next,
    std::uint32_t hot_staging_capacity,
    std::uint32_t *hot_staging_offsets,
    Descriptor *next_descriptors, std::uint32_t *next_capacities,
    std::uint32_t *section_output_counts,
    std::uint32_t *total_output_count, std::uint32_t *overflow_flag) {
  const std::uint32_t hot_index = blockIdx.x;
  if (hot_index >= hot_count || threadIdx.x) return;
  const std::uint32_t q = hot_sections[hot_index];
  const std::size_t base = range_bases[hot_index];
  const std::uint32_t ranges = range_counts[hot_index];
  std::uint32_t total = 0u;
  for (std::uint32_t range = 0u; range < ranges; ++range) {
    cell_offsets[base + range] = total;
    total += cell_counts[base + range];
  }
  cell_offsets[base + ranges] = total;
    section_output_counts[q] = total;
    atomicAdd(total_output_count, total);
    if (!total) {
      hot_staging_offsets[hot_index] = 0u;
      next_descriptors[q] = {};
      next_capacities[q] = 0u;
      return;
    }
    const std::uint32_t staging = atomicAdd(hot_staging_next, total);
    hot_staging_offsets[hot_index] = staging;
    if (std::uint64_t{staging} + total > hot_staging_capacity) {
      atomicExch(overflow_flag, 1u);
      next_descriptors[q] = {};
      next_capacities[q] = 0u;
      return;
    }

    const Descriptor old_descriptor = reuse_existing_sections
        ? descriptors[descriptor_index(q, source_level_limit)]
        : Descriptor{};
    const std::uint32_t old_capacity = reuse_existing_sections
        ? foundation_capacities[q] : 0u;
    if (reuse_existing_sections && total <= old_capacity) {
      next_descriptors[q] =
          Descriptor::make(old_descriptor.offset(), total);
      next_capacities[q] = old_capacity;
    } else {
      const std::uint32_t capacity = reuse_existing_sections
          ? foundation_section_capacity(total) : total;
      const unsigned long long physical = atomicAdd(
          next_output_offset,
          static_cast<unsigned long long>(capacity));
      if (physical + capacity > output_limit) {
        atomicExch(overflow_flag, 1u);
        next_descriptors[q] = {};
      } else {
        next_descriptors[q] = Descriptor::make(physical, total);
      }
      next_capacities[q] = capacity;
    }
}

__global__ void copy_hot_foundation_cells_kernel(
    const std::uint32_t *hot_sections, std::uint32_t hot_count,
    const CrowdedMergeTask *tasks, std::uint32_t task_count,
    const std::uint32_t *range_counts,
    const std::uint32_t *range_bases,
    const std::uint32_t *hot_staging_offsets,
    const std::uint32_t *cell_offsets,
    const Descriptor *next_descriptors, const Row *staging_rows,
    Row *arena) {
  const std::uint32_t task_index = blockIdx.x;
  if (task_index >= task_count) return;
  const CrowdedMergeTask task = tasks[task_index];
  const std::uint32_t hot_index = task.hot_index;
  const std::uint32_t cell = task.range;
  if (hot_index >= hot_count || cell >= range_counts[hot_index]) return;
  const std::uint32_t q = hot_sections[hot_index];
  const std::size_t base = range_bases[hot_index];
  const std::uint32_t begin = cell_offsets[base + cell];
  const std::uint32_t end = cell_offsets[base + cell + 1u];
  const std::uint32_t staging = hot_staging_offsets[hot_index];
  const Descriptor output = next_descriptors[q];
  if (!output.count()) return;
  for (std::uint32_t index = begin + threadIdx.x; index < end;
       index += blockDim.x)
    arena[output.offset() + index] = staging_rows[staging + index];
}

__global__ void finalize_foundation_descriptors_kernel(
    const Descriptor *next_descriptors,
    const std::uint32_t *next_capacities,
    std::uint32_t destination_level, Descriptor *descriptors,
    std::uint32_t *foundation_capacities,
    bool destination_is_foundation) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  for (std::uint32_t level = 0u; level < destination_level; ++level)
    descriptors[descriptor_index(q, level)] = {};
  descriptors[descriptor_index(q, destination_level)] = next_descriptors[q];
  if (destination_is_foundation)
    foundation_capacities[q] = next_capacities[q];
}

__global__ void summarize_adaptive_routes_kernel(
    const RouteHeader *next_headers, const RouteSlice *route_slices,
    bool reserve_slack, Descriptor *next_descriptors,
    std::uint32_t *next_capacities) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  const RouteHeader header = next_headers[q];
  std::uint32_t count = 0u;
  for (std::uint32_t route = 0u; route < header.count; ++route)
    count += route_slices[header.begin + route].rows.count();
  next_descriptors[q] = header.count == 1u
      ? route_slices[header.begin].rows
      : Descriptor::make(0u, count);
  next_capacities[q] = reserve_slack
      ? foundation_section_capacity(count) : count;
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
      descriptor.count() ? RouteHeader{route, 1u} : RouteHeader{};
  if (descriptor.count())
    route_slices[route] = {descriptor, 0u, 1u << 16u};
}

__global__ void publish_adaptive_route_directory_kernel(
    const RouteHeader *next_headers, std::uint32_t destination_level,
    RouteHeader *route_headers) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kQuotients) return;
  for (std::uint32_t level = 0u; level < destination_level; ++level)
    route_headers[descriptor_index(q, level)] = {};
  route_headers[descriptor_index(q, destination_level)] = next_headers[q];
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

__global__ void build_level_guide_kernel(
    const Row *arena, const Descriptor *descriptors,
    std::uint32_t level, std::uint16_t *guides) {
  const std::uint32_t index =
      blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kGuideEntriesPerLevel) return;
  const std::uint32_t q = index / kGuideSamples;
  const std::uint32_t sample = index % kGuideSamples;
  const Descriptor descriptor = descriptors[descriptor_index(q, level)];
  if (descriptor.count() < kGuideRegions) return;
  const std::uint32_t position =
      (sample + 1u) * descriptor.count() / kGuideRegions;
  guides[std::size_t{level} * kGuideEntriesPerLevel + index] =
      arena[descriptor.offset() + position].key;
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
    const std::uint16_t *local_rank,
    const std::uint16_t *level_guides,
    std::uint32_t active_levels,
    std::uint32_t foundation_level,
    std::uint64_t occupied_levels,
    const std::uint32_t *query_ids,
    std::uint32_t *final_values, std::uint8_t *final_found) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
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
    const Descriptor descriptor = routed_descriptor_for_suffix(
        q, level, suffix, route_headers, route_slices);
    if (!descriptor.count()) continue;
    const Row *rows = arena + descriptor.offset();
    std::uint32_t begin = 0u, end = descriptor.count();
    if (route_header.count == 1u)
      guide_search_bounds(level_guides, q, level, descriptor.count(), suffix,
                          begin, end);
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
  const Descriptor foundation = foundation_level < active_levels
      ? routed_descriptor_for_suffix(
            q, foundation_level, suffix, route_headers, route_slices)
      : Descriptor{};
  const Row *foundation_rows = arena + foundation.offset();
  const std::uint32_t cell = (key >> 9u) & 127u;
  const std::size_t local_index = std::size_t{q} * 128u + cell;
  const bool ranked = foundation_level < active_levels &&
      route_headers[descriptor_index(q, foundation_level)].count == 1u &&
      foundation.count() <= 0xffffu;
  const std::uint32_t begin = ranked ? local_rank[local_index] : 0u;
  const std::uint32_t end = ranked
      ? (cell == 127u ? foundation.count()
                      : local_rank[local_index + 1u])
      : foundation.count();
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

__device__ __forceinline__ std::uint32_t warp_lower_bound_rows(
    const Row *rows, std::uint32_t begin, std::uint32_t end,
    std::uint32_t key, bool enabled) {
  constexpr unsigned full = 0xffffffffu;
  std::uint32_t low = begin, high = end;
  while (true) {
    const bool searching = enabled && low < high;
    const unsigned active = __ballot_sync(full, searching);
    if (!active) break;
    if (searching) {
      const std::uint32_t middle = (low + high) >> 1u;
      const unsigned peers = __match_any_sync(active, middle);
      const std::uint32_t leader = __ffs(peers) - 1u;
      std::uint32_t pivot = 0u;
      if ((threadIdx.x & 31u) == leader)
        pivot = rows[middle].key;
      pivot = __shfl_sync(peers, pivot, leader);
      if (pivot < key) low = middle + 1u;
      else high = middle;
    }
  }
  return low;
}

__global__ void section_owned_lookup_kernel(
    const std::uint32_t *queries, const std::uint32_t *query_offsets,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets,
    std::uint32_t batch_stride, std::uint32_t pending_batches, const Row *arena,
    const Descriptor *descriptors, const std::uint16_t *local_rank,
    const std::uint16_t *level_guides,
    std::uint32_t active_levels, std::uint32_t foundation_level,
    std::uint64_t occupied_levels, const std::uint32_t *query_ids,
    std::uint32_t *final_values, std::uint8_t *final_found) {
  __shared__ std::uint32_t hash_keys[kLookupWarpsPerBlock][kLookupHashSlots];
  __shared__ unsigned long long hash_best[kLookupWarpsPerBlock]
                                         [kLookupHashSlots];
  __shared__ std::uint32_t unique_queries[kLookupWarpsPerBlock];
  __shared__ std::uint32_t matched_queries[kLookupWarpsPerBlock];

  constexpr unsigned full = 0xffffffffu;
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t q = blockIdx.x * kLookupWarpsPerBlock + warp;
  if (q >= kQuotients)
    return;
  const std::uint32_t section_begin = query_offsets[q];
  const std::uint32_t section_end = query_offsets[q + 1u];
  for (std::uint32_t query_begin = section_begin; query_begin < section_end;
       query_begin += kLookupHashSlots) {
    const std::uint32_t query_end =
        min(query_begin + kLookupHashSlots, section_end);

    for (std::uint32_t slot = lane; slot < kLookupHashSlots; slot += 32u) {
      hash_keys[warp][slot] = kEmptyLookupKey;
      hash_best[warp][slot] = 0ull;
    }
    if (lane == 0u) {
      unique_queries[warp] = 0u;
      matched_queries[warp] = 0u;
    }
    __syncwarp();

    for (std::uint32_t i = query_begin + lane; i < query_end; i += 32u) {
      const std::uint32_t suffix = key_suffix(queries[i]);
      std::uint32_t slot = mix_lookup_key(suffix) & (kLookupHashSlots - 1u);
      for (std::uint32_t probe = 0u; probe < kLookupHashSlots; ++probe) {
        const std::uint32_t prior =
            atomicCAS(&hash_keys[warp][slot], kEmptyLookupKey, suffix);
        if (prior == kEmptyLookupKey) {
          atomicAdd(&unique_queries[warp], 1u);
          break;
        }
        if (prior == suffix)
          break;
        slot = (slot + 1u) & (kLookupHashSlots - 1u);
      }
    }
    __syncwarp();

    for (int batch = int(pending_batches) - 1; batch >= 0; --batch) {
      const std::size_t oi =
          std::size_t{std::uint32_t(batch)} * (kQuotients + 1u) + q;
      const std::uint32_t begin = raw_offsets[oi];
      const std::uint32_t end = raw_offsets[oi + 1u];
      for (std::uint32_t position = begin + lane; position < end;
           position += 32u) {
        const std::uint32_t record =
            std::uint32_t(batch) * batch_stride + position;
        const std::uint32_t suffix = key_suffix(raw_keys[record]);
        std::uint32_t slot = mix_lookup_key(suffix) & (kLookupHashSlots - 1u);
        for (std::uint32_t probe = 0u; probe < kLookupHashSlots; ++probe) {
          const std::uint32_t observed = hash_keys[warp][slot];
          if (observed == kEmptyLookupKey)
            break;
          if (observed == suffix) {
            const RawPayload payload = raw_payloads[record];
            const unsigned long long token =
                (static_cast<unsigned long long>(raw_position(payload) + 1u)
                 << 32u) |
                static_cast<unsigned long long>(record + 1u);
            const unsigned long long prior =
                atomicMax(&hash_best[warp][slot], token);
            if (prior == 0ull)
              atomicAdd(&matched_queries[warp], 1u);
            break;
          }
          slot = (slot + 1u) & (kLookupHashSlots - 1u);
        }
      }
      __syncwarp();
      if (matched_queries[warp] == unique_queries[warp])
        break;
    }

    for (std::uint32_t round = 0u; query_begin + round * 32u < query_end;
         ++round) {
      const std::uint32_t i = query_begin + round * 32u + lane;
      const bool valid = i < query_end;
      const std::uint32_t key = valid ? queries[i] : 0u;
      const std::uint32_t suffix = key_suffix(key);
      unsigned long long token = 0ull;
      if (valid) {
        std::uint32_t slot = mix_lookup_key(suffix) & (kLookupHashSlots - 1u);
        for (std::uint32_t probe = 0u; probe < kLookupHashSlots; ++probe) {
          const std::uint32_t observed = hash_keys[warp][slot];
          if (observed == suffix) {
            token = hash_best[warp][slot];
            break;
          }
          if (observed == kEmptyLookupKey)
            break;
          slot = (slot + 1u) & (kLookupHashSlots - 1u);
        }
      }

      Row result{};
      bool resolved = valid && token != 0ull;
      if (resolved) {
        const std::uint32_t record = static_cast<std::uint32_t>(token) - 1u;
        const RawPayload payload = raw_payloads[record];
        result = payload.metadata & kRawTombstone
                     ? make_row(key, 0u, kTombstone)
                     : make_row(key, payload.value, 0u);
      }

      std::uint64_t levels = occupied_levels;
      if (foundation_level < kMaximumLevels)
        levels &= ~(std::uint64_t{1} << foundation_level);
      while (levels) {
        const std::uint32_t level =
            static_cast<std::uint32_t>(__ffsll(levels) - 1);
        levels &= levels - 1u;
        unsigned long long descriptor_bits = 0ull;
        if (lane == 0u)
          descriptor_bits = descriptors[descriptor_index(q, level)].bits;
        descriptor_bits = __shfl_sync(full, descriptor_bits, 0u);
        const Descriptor descriptor{descriptor_bits};
        const Row *rows = arena + descriptor.offset();
        const bool search = valid && !resolved && descriptor.count();
        std::uint32_t begin = 0u, end = descriptor.count();
        if (search)
          guide_search_bounds(level_guides, q, level, descriptor.count(),
                              suffix, begin, end);
        const std::uint32_t position =
            warp_lower_bound_rows(rows, begin, end, suffix, search);
        if (search && position < end) {
          const Row candidate = rows[position];
          if (candidate.key == suffix) {
            result = candidate;
            resolved = true;
          }
        }
      }

      if (valid && !resolved) {
        const Descriptor foundation =
            foundation_level < active_levels
                ? descriptors[descriptor_index(q, foundation_level)]
                : Descriptor{};
        const Row *rows = arena + foundation.offset();
        const std::uint32_t cell = (key >> 9u) & 127u;
        const std::size_t rank = std::size_t{q} * 128u + cell;
        const bool ranked = foundation.count() <= 0xffffu;
        const std::uint32_t begin = ranked ? local_rank[rank] : 0u;
        const std::uint32_t end =
            ranked ? (cell == 127u ? foundation.count() : local_rank[rank + 1u])
                   : foundation.count();
        const std::uint32_t position =
            lower_bound_rows(rows + begin, end - begin, suffix);
        if (position < end - begin && rows[begin + position].key == suffix) {
          result = rows[begin + position];
          resolved = true;
        }
      }

      if (valid) {
        const bool live = resolved && (result.flags & kTombstone) == 0u;
        const std::uint32_t destination = query_ids ? query_ids[i] : i;
        final_values[destination] = live          ? result.value
                                    : final_found ? 0u
                                                  : kInvalid;
      if (final_found)
        final_found[destination] = live;
    }
    __syncwarp();
  }
}
}

__device__ bool first_visible_in_quotient(
    std::uint32_t q, std::uint32_t lower,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const Row *arena,
    const Descriptor *descriptors, const RouteHeader *route_headers,
    const RouteSlice *route_slices, std::uint32_t active_levels,
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
    const RouteSlice *route_slices, std::uint32_t active_levels) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t query = queries[i];
  for (std::uint32_t q = query >> 16u; q < kQuotients; ++q) {
    const std::uint32_t lower = q == (query >> 16u) ? query : q << 16u;
    std::uint32_t result{};
    if (first_visible_in_quotient(
            q, lower, raw_keys, raw_payloads, raw_offsets, batch_stride,
            pending_batches, arena, descriptors, route_headers,
            route_slices, active_levels, result)) {
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
        route_stride_(gpulsmopt2_detail::adaptive_route_stride(
            publication_capacity_)),
        local_rank_(gpulsmopt2_detail::kLocalRankEntries),
        level_guides_(
            gpulsmopt2_detail::kMaximumLevels *
                gpulsmopt2_detail::kGuideEntriesPerLevel,
            2u * gpulsmopt2_detail::kGuideEntriesPerLevel),
        arena_(gpulsmopt2_detail::maximum_resident_elements<
                   gpulsmopt2_detail::Row>(),
               foundation_pool_capacity_ + publication_capacity_),
        descriptors_(std::size_t{gpulsmopt2_detail::kQuotients} *
                     gpulsmopt2_detail::kMaximumLevels),
        route_headers_(std::size_t{gpulsmopt2_detail::kQuotients} *
                       gpulsmopt2_detail::kMaximumLevels),
        route_slices_(
            route_stride_ * gpulsmopt2_detail::kMaximumLevels,
            route_stride_ * 2u),
        raw_keys_(gpulsmopt2_detail::kBatchesPerEpoch * batch_capacity_),
        raw_payloads_(gpulsmopt2_detail::kBatchesPerEpoch * batch_capacity_),
        raw_offsets_(std::size_t{gpulsmopt2_detail::kBatchesPerEpoch} *
                     (gpulsmopt2_detail::kQuotients + 1u)),
        raw_signatures_(std::size_t{gpulsmopt2_detail::kBatchesPerEpoch} *
                        gpulsmopt2_detail::kQuotients),
        raw_epoch_signatures_(gpulsmopt2_detail::kQuotients),
        level_keys_(gpulsmopt2_detail::maximum_resident_elements<
                        std::uint32_t>(),
                    foundation_pool_capacity_ + publication_capacity_),
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
        publication_keys_b_(gpulsmopt2_detail::kMaximumPublicationRows,
                            publication_capacity_),
        publication_rows_a_(gpulsmopt2_detail::kMaximumPublicationRows,
                            publication_capacity_),
        publication_rows_b_(gpulsmopt2_detail::kMaximumPublicationRows,
                            publication_capacity_),
        publication_selected_count_(1u),
        publication_batch_offsets_(gpulsmopt2_detail::kBatchesPerEpoch + 1u),
        foundation_source_offsets_(gpulsmopt2_detail::kQuotients + 1u),
        foundation_build_capacities_(gpulsmopt2_detail::kQuotients + 1u),
        foundation_physical_offsets_(gpulsmopt2_detail::kQuotients + 1u),
        foundation_capacities_(gpulsmopt2_detail::kQuotients),
        foundation_next_capacities_(gpulsmopt2_detail::kQuotients),
        foundation_next_descriptors_(gpulsmopt2_detail::kQuotients),
        foundation_next_route_headers_(gpulsmopt2_detail::kQuotients),
        foundation_section_output_counts_(gpulsmopt2_detail::kQuotients),
        balanced_merge_raw_counts_(gpulsmopt2_detail::kQuotients),
        balanced_merge_location_counts_(1u),
        balanced_merge_jobs_(gpulsmopt2_detail::kQuotients),
        balanced_merge_pull_slices_(1u),
        balanced_merge_job_count_(1u),
        foundation_hot_flags_(gpulsmopt2_detail::kQuotients),
        foundation_hot_sections_(gpulsmopt2_detail::kQuotients),
        foundation_hot_selected_count_(1u),
        foundation_hot_range_counts_(gpulsmopt2_detail::kQuotients),
        foundation_hot_range_bases_(gpulsmopt2_detail::kQuotients + 1u),
        foundation_hot_lows_(1u),
        foundation_hot_highs_(1u),
        foundation_hot_raw_counts_(1u),
        foundation_hot_tasks_(1u),
        foundation_hot_staging_offsets_(gpulsmopt2_detail::kQuotients),
        foundation_hot_staging_next_(1u),
        foundation_hot_cell_counts_(1u),
        foundation_hot_cell_offsets_(1u),
        foundation_hot_range_boundaries_(1u),
        foundation_hot_streaming_flags_(1u),
        foundation_next_offset_(1u),
        balanced_merge_next_offset_(1u),
        foundation_total_output_count_(1u),
        foundation_overflow_flag_(1u),
        admission_counts_(gpulsmopt2_detail::kQuotients + 1u),
        lookup_quotient_offsets_(gpulsmopt2_detail::kQuotients + 1u),
        range_partials_(gpulsmopt2_detail::kThreads),
        range_overflow_count_(1u) {
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
    initialize_foundation_workspace();
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
    gpulsmopt2_detail::prepare_foundation_build_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            foundation_source_offsets_.data(),
            foundation_build_capacities_.data());
    std::size_t foundation_scan_bytes = foundation_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        foundation_temp_.data(), foundation_scan_bytes,
        foundation_build_capacities_.data(),
        foundation_physical_offsets_.data(),
        gpulsmopt2_detail::kQuotients + 1u, stream));
    std::uint32_t foundation_physical_count{};
    CUDA_CHECK(cudaMemcpyAsync(
        &foundation_physical_count,
        foundation_physical_offsets_.data() + gpulsmopt2_detail::kQuotients,
        sizeof(foundation_physical_count), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        foundation_capacities_host_.data(),
        foundation_build_capacities_.data(),
        gpulsmopt2_detail::kQuotients * sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        foundation_physical_offsets_host_.data(),
        foundation_physical_offsets_.data(),
        std::size_t{gpulsmopt2_detail::kQuotients + 1u} *
            sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (foundation_physical_count > foundation_pool_capacity_ / 2u)
      throw std::bad_alloc();
    gpulsmopt2_detail::scatter_foundation_build_kernel<<<
        blocks(base_count), gpulsmopt2_detail::kThreads, 0, stream>>>(
            publication_keys_a_.data(), publication_rows_a_.data(),
            base_count, foundation_source_offsets_.data(),
            foundation_physical_offsets_.data(), arena_.data());
    gpulsmopt2_detail::publish_foundation_build_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            foundation_source_offsets_.data(),
            foundation_physical_offsets_.data(),
            foundation_build_capacities_.data(), level,
            descriptors_.data(), foundation_capacities_.data());
    const std::uint64_t next_foundation_offset = foundation_physical_count;
    foundation_tail_ = next_foundation_offset;
    CUDA_CHECK(cudaMemcpyAsync(
        foundation_next_offset_.data(), &next_foundation_offset,
        sizeof(next_foundation_offset), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaGetLastError());
    level_offsets_[level] = 0u;
    level_span_sizes_[level] = 0u;
    level_counts_[level] = base_count;
    auto &initial_locations = level_locations_[level];
    initial_locations.clear();
    initial_locations.reserve(gpulsmopt2_detail::kQuotients);
    for (std::uint32_t q = 0u; q < gpulsmopt2_detail::kQuotients; ++q)
      initial_locations.push_back({
          std::uint64_t{q} << 16u, std::uint64_t{q + 1u} << 16u,
          foundation_physical_offsets_host_[q], q, q + 1u, 1u,
          foundation_capacities_host_[q]});
    level_route_counts_[level].assign(
        gpulsmopt2_detail::kQuotients, 1u);
    has_adaptive_routes_ = false;
    ensure_route_level(level, stream);
    gpulsmopt2_detail::publish_single_route_directory_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            descriptors_.data(), level,
            static_cast<std::uint32_t>(route_stride_),
            route_headers_.data(), route_slices_.data());
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
          route_slices_.data(), local_rank_.data(), level_guides_.data(),
          active_levels_, foundation_level(), occupied_level_mask(),
          query_ids, nullptr, nullptr);
      CUDA_CHECK(cudaGetLastError());
    } else {
      gpulsmopt2_detail::lookup_with_pending_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          batch.queries, batch.out_values, batch.out_found, count,
          raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
          static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
          raw_signatures_.data(),
          raw_epoch_signatures_.data(), arena_.data(), descriptors_.data(),
          route_headers_.data(), route_slices_.data(), local_rank_.data(),
          level_guides_.data(), active_levels_,
          foundation_level(),
          occupied_level_mask(),
          nullptr, nullptr, nullptr);
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
    const bool use_section_owners = !has_adaptive_routes_ &&
        query_count > 1u &&
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
    if (query_count == 1u) {
      gpulsmopt2_detail::emit_single_range_fragments_kernel<false><<<
          gpulsmopt2_detail::kQuotients / gpulsmopt2_detail::kThreads,
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              batch.lo, batch.hi, range_fragment_offsets_.data(),
              range_fragments_.data(), nullptr, nullptr);
    } else if (std::uint64_t{fragment_count} <=
               std::uint64_t{query_count} * 4u) {
      if (use_section_owners)
        gpulsmopt2_detail::emit_range_fragments_thread_kernel<true><<<
            blocks(query_count), gpulsmopt2_detail::kThreads, 0, stream>>>(
                batch.lo, batch.hi, range_fragment_offsets_.data(), query_count,
                nullptr, range_section_keys_in_.data(),
                range_section_fragments_in_.data());
      else
        gpulsmopt2_detail::emit_range_fragments_thread_kernel<false><<<
            blocks(query_count), gpulsmopt2_detail::kThreads, 0, stream>>>(
                batch.lo, batch.hi, range_fragment_offsets_.data(), query_count,
                range_fragments_.data(), nullptr, nullptr);
    } else {
      if (use_section_owners)
        gpulsmopt2_detail::emit_range_fragments_kernel<true><<<
            (query_count + 7u) / 8u, gpulsmopt2_detail::kThreads, 0, stream>>>(
                batch.lo, batch.hi, range_fragment_offsets_.data(), query_count,
                nullptr, range_section_keys_in_.data(),
                range_section_fragments_in_.data());
      else
        gpulsmopt2_detail::emit_range_fragments_kernel<false><<<
            (query_count + 7u) / 8u, gpulsmopt2_detail::kThreads, 0, stream>>>(
                batch.lo, batch.hi, range_fragment_offsets_.data(), query_count,
                range_fragments_.data(), nullptr, nullptr);
    }
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
      CUDA_CHECK(cudaMemsetAsync(range_section_max_fragments_.data(), 0,
                                 sizeof(std::uint32_t), stream));
      gpulsmopt2_detail::summarize_section_fragment_density_kernel<<<
          blocks(gpulsmopt2_detail::kQuotients),
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              range_section_offsets_.data(),
              range_section_max_fragments_.data());
      std::uint32_t maximum_section_fragments{};
      CUDA_CHECK(cudaMemcpyAsync(
          &maximum_section_fragments, range_section_max_fragments_.data(),
          sizeof(maximum_section_fragments), cudaMemcpyDeviceToHost,
          stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const bool tile_sections = maximum_section_fragments >
          gpulsmopt2_detail::kSectionTaskFragments;
      const std::uint32_t maximum_section_tasks =
          gpulsmopt2_detail::kQuotients +
          (fragment_count + gpulsmopt2_detail::kSectionTaskFragments - 1u) /
              gpulsmopt2_detail::kSectionTaskFragments;
      if (tile_sections) {
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
      }
      if (pending_batches_)
        tile_sections
            ? launch_section_ranges<true, true>(maximum_section_tasks, stream)
            : launch_section_ranges<true, false>(maximum_section_tasks,
                                                 stream);
      else
        tile_sections
            ? launch_section_ranges<false, true>(maximum_section_tasks,
                                                 stream)
            : launch_section_ranges<false, false>(maximum_section_tasks,
                                                  stream);
    } else if (pending_batches_) {
      launch_fragment_ranges<true>(fragment_count, query_count, batch, stream);
    } else {
      launch_fragment_ranges<false>(fragment_count, query_count, batch,
                                    stream);
    }
    if (query_count == 1u) {
      gpulsmopt2_detail::reduce_single_range_partials_stage1_kernel<<<
          256, gpulsmopt2_detail::kThreads, 0, stream>>>(
              range_fragment_partials_.data(),
              range_fragment_offsets_.data() + 1u,
              range_partials_.data());
      gpulsmopt2_detail::reduce_single_range_partials_stage2_kernel<<<
          1, gpulsmopt2_detail::kThreads, 0, stream>>>(
              range_partials_.data(), batch.out_sums);
    } else if (std::uint64_t{fragment_count} <=
               std::uint64_t{query_count} * 4u) {
      gpulsmopt2_detail::reduce_range_fragment_partials_thread_kernel<<<
          blocks(query_count), gpulsmopt2_detail::kThreads, 0, stream>>>(
              range_fragment_offsets_.data(), query_count,
              range_fragment_partials_.data(), batch.out_sums);
    } else {
      gpulsmopt2_detail::reduce_range_fragment_partials_kernel<<<
          (query_count + 7u) / 8u, gpulsmopt2_detail::kThreads, 0, stream>>>(
              range_fragment_offsets_.data(), query_count,
              range_fragment_partials_.data(), batch.out_sums);
    }
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
        route_headers_.data(), route_slices_.data(), active_levels_);
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
        raw_keys_.size() * sizeof(std::uint32_t) +
        raw_payloads_.size() * sizeof(gpulsmopt2_detail::RawPayload) +
        raw_offsets_.size() * sizeof(std::uint32_t) +
        raw_signatures_.size() * sizeof(std::uint64_t) +
        raw_epoch_signatures_.size() * sizeof(std::uint64_t) +
        level_keys_.size() * sizeof(std::uint32_t) +
        (publication_epoch_keys_a_.size() +
         publication_epoch_keys_b_.size()) * sizeof(std::uint32_t) +
        (publication_epoch_assignments_a_.size() +
         publication_epoch_assignments_b_.size()) *
            sizeof(gpulsmopt2_detail::RawAssignment) +
        (publication_rows_a_.size() + publication_rows_b_.size()) *
            sizeof(gpulsmopt2_detail::Row) +
        (publication_keys_a_.size() + publication_keys_b_.size() +
         publication_selected_count_.size() +
         publication_batch_offsets_.size()) * sizeof(std::uint32_t) +
        (foundation_source_offsets_.size() +
         foundation_build_capacities_.size() +
         foundation_physical_offsets_.size() +
         foundation_capacities_.size() +
         foundation_next_capacities_.size() +
         foundation_section_output_counts_.size() +
         balanced_merge_raw_counts_.size() +
         balanced_merge_location_counts_.size() +
         balanced_merge_job_count_.size() +
         foundation_hot_sections_.size() +
         foundation_hot_selected_count_.size() +
         foundation_hot_range_counts_.size() +
         foundation_hot_range_bases_.size() +
         foundation_hot_lows_.size() +
         foundation_hot_highs_.size() +
         foundation_hot_raw_counts_.size() +
         foundation_hot_staging_offsets_.size() +
         foundation_hot_staging_next_.size() +
         foundation_hot_cell_counts_.size() +
         foundation_hot_cell_offsets_.size() +
         foundation_hot_range_boundaries_.size() +
         foundation_total_output_count_.size() +
         foundation_overflow_flag_.size()) * sizeof(std::uint32_t) +
        balanced_merge_jobs_.size() *
            sizeof(gpulsmopt2_detail::BalancedMergeJob) +
        balanced_merge_pull_slices_.size() *
            sizeof(gpulsmopt2_detail::PullSlice) +
        foundation_hot_tasks_.size() *
            sizeof(gpulsmopt2_detail::CrowdedMergeTask) +
        foundation_next_descriptors_.size() *
            sizeof(gpulsmopt2_detail::Descriptor) +
        foundation_next_route_headers_.size() *
            sizeof(gpulsmopt2_detail::RouteHeader) +
        foundation_hot_flags_.size() * sizeof(std::uint8_t) +
        foundation_hot_streaming_flags_.size() * sizeof(std::uint8_t) +
        (foundation_next_offset_.size() +
         balanced_merge_next_offset_.size()) * sizeof(std::uint64_t) +
        foundation_temp_.size() * sizeof(std::uint8_t) +
        publication_temp_.size() * sizeof(std::uint8_t) +
        admission_counts_.size() * sizeof(std::uint32_t) +
        admission_temp_.size() * sizeof(std::uint8_t) +
        lookup_quotient_offsets_.size() * sizeof(std::uint32_t) +
        radix_storage_.size() * sizeof(std::uint8_t) +
        range_partials_.size() * sizeof(unsigned long long) +
        range_overflow_count_.size() * sizeof(std::uint32_t) +
        range_query_storage_.size() + range_fragment_storage_.size() +
        range_section_storage_.size();
  }

private:
  static int blocks(std::size_t count) {
    return static_cast<int>((count + gpulsmopt2_detail::kThreads - 1u) /
                            gpulsmopt2_detail::kThreads);
  }

  static std::uint32_t levels_for_epochs(std::uint64_t epochs) {
    std::uint32_t levels = 0u;
    while (epochs) {
      ++levels;
      epochs >>= 1u;
    }
    return levels;
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
    std::uint64_t mask = 0u;
    for (std::uint32_t level = 0u;
         level < gpulsmopt2_detail::kMaximumLevels; ++level)
      if (level_counts_[level]) mask |= std::uint64_t{1} << level;
    return mask;
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

  void rebuild_level_guide(std::uint32_t level, cudaStream_t stream) {
    const std::size_t required =
        (std::size_t{level} + 1u) *
        gpulsmopt2_detail::kGuideEntriesPerLevel;
    if (required > level_guides_.size()) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
      level_guides_.grow(required);
    }
    gpulsmopt2_detail::build_level_guide_kernel<<<
        blocks(gpulsmopt2_detail::kGuideEntriesPerLevel),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            arena_.data(), descriptors_.data(), level,
            level_guides_.data());
    CUDA_CHECK(cudaGetLastError());
  }

  void initialize_publication_workspace() {
    const std::uint32_t epoch_capacity = static_cast<std::uint32_t>(
        batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch);
    const std::uint32_t capacity =
        static_cast<std::uint32_t>(publication_capacity_);
    std::size_t sort_bytes{}, raw_reduce_bytes{}, merge_bytes{};
    std::size_t merge_unique_bytes{};
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
    const std::uint32_t first = capacity / 2u;
    const std::uint32_t second = capacity - first;
    CUDA_CHECK(cub::DeviceMerge::MergePairs(
        nullptr, merge_bytes, publication_keys_a_.data(),
        publication_rows_a_.data(), first, level_keys_.data(),
        arena_.data(), second, publication_keys_b_.data(),
        publication_rows_b_.data(), cuda::std::less<>{}, 0));
    CUDA_CHECK(cub::DeviceSelect::UniqueByKey(
        nullptr, merge_unique_bytes, publication_keys_b_.data(),
        publication_rows_b_.data(), publication_keys_a_.data(),
        publication_rows_a_.data(), publication_selected_count_.data(),
        capacity, cuda::std::equal_to<>{}, 0));
    publication_temp_.resize(std::max({sort_bytes, raw_reduce_bytes,
                                       merge_bytes, merge_unique_bytes}));
  }

  void initialize_foundation_workspace() {
    std::size_t scan_bytes{}, select_bytes{};
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, foundation_build_capacities_.data(),
        foundation_physical_offsets_.data(),
        gpulsmopt2_detail::kQuotients + 1u, 0));
    cub::CountingInputIterator<std::uint32_t> sections(0u);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, select_bytes, sections, foundation_hot_flags_.data(),
        foundation_hot_sections_.data(),
        foundation_hot_selected_count_.data(),
        gpulsmopt2_detail::kQuotients, 0));
    foundation_temp_.resize(std::max(scan_bytes, select_bytes));
  }

  void ensure_crowded_merge_workspace(std::uint32_t task_count,
                                      std::uint32_t boundary_slots,
                                      cudaStream_t stream) {
    if (boundary_slots <= foundation_hot_cell_counts_.size() &&
        boundary_slots <= foundation_hot_cell_offsets_.size() &&
        boundary_slots <= foundation_hot_range_boundaries_.size() &&
        task_count <= foundation_hot_streaming_flags_.size() &&
        task_count <= foundation_hot_tasks_.size() &&
        task_count <= foundation_hot_lows_.size() &&
        task_count <= foundation_hot_highs_.size() &&
        task_count <= foundation_hot_raw_counts_.size() &&
        task_count <= foundation_hot_sections_.size() &&
        task_count <= foundation_hot_range_counts_.size() &&
        task_count + 1u <= foundation_hot_range_bases_.size())
      return;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    foundation_hot_cell_counts_.resize(std::max(1u, boundary_slots));
    foundation_hot_cell_offsets_.resize(std::max(1u, boundary_slots));
    foundation_hot_range_boundaries_.resize(std::max(1u, boundary_slots));
    foundation_hot_streaming_flags_.resize(std::max(1u, task_count));
    foundation_hot_tasks_.resize(std::max(1u, task_count));
    foundation_hot_lows_.resize(std::max(1u, task_count));
    foundation_hot_highs_.resize(std::max(1u, task_count));
    foundation_hot_raw_counts_.resize(std::max(1u, task_count));
    foundation_hot_sections_.resize(std::max(1u, task_count));
    foundation_hot_range_counts_.resize(std::max(1u, task_count));
    foundation_hot_range_bases_.resize(std::max(1u, task_count + 1u));
  }

  void ensure_pull_slice_workspace(std::uint32_t slice_slots,
                                   cudaStream_t stream) {
    if (slice_slots <= balanced_merge_pull_slices_.size()) return;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    balanced_merge_pull_slices_.resize(std::max(1u, slice_slots));
  }

  void ensure_balanced_job_workspace(std::uint32_t job_count,
                                     cudaStream_t stream) {
    if (job_count <= balanced_merge_jobs_.size() &&
        job_count <= balanced_merge_location_counts_.size()) return;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    balanced_merge_jobs_.resize(std::max(1u, job_count));
    balanced_merge_location_counts_.resize(std::max(1u, job_count));
  }

  void ensure_publication_capacity(std::size_t count,
                                   cudaStream_t stream) {
    if (count <= publication_capacity_) return;
    if (count > gpulsmopt2_detail::kMaximumPublicationRows)
      throw std::bad_alloc();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    publication_keys_a_.grow(count);
    publication_keys_b_.grow(count);
    publication_rows_a_.grow(count);
    publication_rows_b_.grow(count);
    publication_capacity_ = std::min(
        {publication_keys_a_.size(), publication_keys_b_.size(),
         publication_rows_a_.size(), publication_rows_b_.size()});
    initialize_publication_workspace();
  }

  void ensure_arena_capacity(std::size_t count, cudaStream_t stream) {
    if (count <= arena_.size() && count <= level_keys_.size()) return;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    arena_.grow(count);
    level_keys_.grow(count);
  }

  void ensure_route_level(std::uint32_t level, cudaStream_t stream) {
    const std::size_t required =
        (std::size_t{level} + 1u) * route_stride_;
    if (required <= route_slices_.size()) return;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    route_slices_.grow(required);
  }

  std::uint64_t allocate_level_span(std::uint32_t count,
                                    cudaStream_t stream) {
    struct Interval { std::uint64_t begin, end; };
    std::vector<Interval> occupied;
    for (std::uint32_t level = 0u;
         level < gpulsmopt2_detail::kMaximumLevels; ++level) {
      if (level_counts_[level])
        if (level_offsets_[level] >= foundation_pool_capacity_)
          occupied.push_back({level_offsets_[level],
                              level_offsets_[level] +
                                  level_span_sizes_[level]});
    }
    std::sort(occupied.begin(), occupied.end(),
              [](const Interval &a, const Interval &b) {
                return a.begin < b.begin;
              });
    std::uint64_t cursor = foundation_pool_capacity_;
    for (const Interval &interval : occupied) {
      if (interval.begin >= cursor && interval.begin - cursor >= count)
        return cursor;
      cursor = std::max(cursor, interval.end);
    }
    const std::size_t required = static_cast<std::size_t>(cursor) + count;
    ensure_arena_capacity(required, stream);
    return cursor;
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
    foundation_bank_ = 0u;
    foundation_tail_ = 0u;
    has_adaptive_routes_ = false;
    std::fill_n(level_counts_, gpulsmopt2_detail::kMaximumLevels, 0u);
    std::fill_n(level_offsets_, gpulsmopt2_detail::kMaximumLevels, 0u);
    std::fill_n(level_span_sizes_, gpulsmopt2_detail::kMaximumLevels, 0u);
    for (auto &locations : level_locations_) locations.clear();
    for (auto &counts : level_route_counts_) counts.clear();
    std::fill(foundation_capacities_host_.begin(),
              foundation_capacities_host_.end(), 0u);
    CUDA_CHECK(cudaMemsetAsync(
        foundation_capacities_.data(), 0,
        foundation_capacities_.size() * sizeof(std::uint32_t), stream));
    const std::uint64_t empty_foundation_offset = 0u;
    CUDA_CHECK(cudaMemcpyAsync(
        foundation_next_offset_.data(), &empty_foundation_offset,
        sizeof(empty_foundation_offset), cudaMemcpyHostToDevice, stream));
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
    const bool direct =
        n >= gpulsmopt2_detail::kDirectAdmissionMinimum &&
        n <= gpulsmopt2_detail::kDirectAdmissionMaximum;
    if (direct) {
      gpulsmopt2_detail::count_admission_quotients_kernel<<<
          blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
              keys, n, admission_counts_.data(), radix_ids_out_.data(),
              batch_signatures);
      std::size_t scan_bytes = admission_temp_.size();
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          admission_temp_.data(), scan_bytes, admission_counts_.data(),
          batch_offsets, gpulsmopt2_detail::kQuotients + 1u, stream));
      gpulsmopt2_detail::commit_admission_metadata_kernel<<<
          gpulsmopt2_detail::kQuotients / gpulsmopt2_detail::kThreads,
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              admission_counts_.data(), batch_signatures,
              raw_epoch_signatures_.data());
      gpulsmopt2_detail::scatter_admission_records_kernel<<<
          blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
              keys, values, n, slot, tombstone,
              batch_offsets, radix_ids_out_.data(),
              raw_keys_.data() + std::size_t{slot} * batch_capacity_,
              raw_payloads_.data() + std::size_t{slot} * batch_capacity_);
    } else {
      std::size_t sort_bytes{};
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, sort_bytes, keys, radix_keys_.data(),
          radix_ids_out_.data(), radix_ids_out_.data(), n, 16, 32, stream));
      ensure_radix_workspace(sort_bytes, n, stream);
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          radix_workspace(), sort_bytes, keys, radix_keys_.data(),
          radix_input_ids(), radix_ids_out_.data(), n, 16, 32, stream));
      gpulsmopt2_detail::gather_raw_batch_kernel<<<
          blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
              radix_keys_.data(), radix_ids_out_.data(), values, n, slot,
              tombstone,
              raw_keys_.data() + std::size_t{slot} * batch_capacity_,
              raw_payloads_.data() + std::size_t{slot} * batch_capacity_,
              batch_signatures, raw_epoch_signatures_.data());
      gpulsmopt2_detail::finalize_quotient_metadata_kernel<<<
          gpulsmopt2_detail::kQuotients / gpulsmopt2_detail::kThreads,
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              radix_keys_.data(), n, batch_offsets);
    }
    CUDA_CHECK(cudaGetLastError());
    raw_batch_counts_[slot] = n;
    ++pending_batches_;
    if (pending_batches_ == gpulsmopt2_detail::kBatchesPerEpoch)
      publish_epoch(stream);
  }

  std::uint32_t publish_balanced_merge(
      const std::uint32_t *current_keys,
      const gpulsmopt2_detail::Row *current_rows,
      std::uint32_t current_count, std::uint32_t source_level_limit,
      std::uint32_t destination_level, bool destination_is_foundation,
      std::uint64_t destination_span, std::uint32_t destination_capacity,
      cudaStream_t stream) {
    ensure_route_level(destination_level, stream);
    gpulsmopt2_detail::build_query_quotient_offsets_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            current_keys, current_count,
            foundation_source_offsets_.data());
    gpulsmopt2_detail::count_balanced_merge_work_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            foundation_source_offsets_.data(), descriptors_.data(),
            source_level_limit, balanced_merge_raw_counts_.data());
    CUDA_CHECK(cudaMemcpyAsync(
        balanced_merge_raw_counts_host_.data(),
        balanced_merge_raw_counts_.data(),
        gpulsmopt2_detail::kQuotients * sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemsetAsync(
        foundation_next_descriptors_.data(), 0,
        foundation_next_descriptors_.size() *
            sizeof(gpulsmopt2_detail::Descriptor), stream));
    CUDA_CHECK(cudaMemsetAsync(
        foundation_next_capacities_.data(), 0,
        foundation_next_capacities_.size() * sizeof(std::uint32_t),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        foundation_section_output_counts_.data(), 0,
        foundation_section_output_counts_.size() * sizeof(std::uint32_t),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        foundation_total_output_count_.data(), 0, sizeof(std::uint32_t),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        foundation_overflow_flag_.data(), 0, sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        foundation_hot_staging_next_.data(), 0, sizeof(std::uint32_t),
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::uint32_t normal_job_count = 0u, hot_count = 0u;
    std::uint32_t job_begin = 0u, job_end = 0u, job_work = 0u;
    std::uint64_t job_key_begin = 0u, job_key_end = 0u;
    std::uint64_t job_existing_offset = 0u;
    std::uint32_t job_existing_capacity = 0u;
    const auto &starting_locations = level_locations_[source_level_limit];
    balanced_merge_location_counts_host_.assign(
        starting_locations.size(), 0u);
    if (!starting_locations.empty()) {
      ensure_balanced_job_workspace(
          static_cast<std::uint32_t>(starting_locations.size()), stream);
      CUDA_CHECK(cudaMemcpyAsync(
          balanced_merge_jobs_.data(), starting_locations.data(),
          starting_locations.size() *
              sizeof(gpulsmopt2_detail::BalancedMergeJob),
          cudaMemcpyHostToDevice, stream));
      gpulsmopt2_detail::count_adaptive_location_work_kernel<<<
          blocks(starting_locations.size()), gpulsmopt2_detail::kThreads,
          0, stream>>>(
              balanced_merge_jobs_.data(),
              static_cast<std::uint32_t>(starting_locations.size()),
              current_rows, foundation_source_offsets_.data(),
              arena_.data(), route_headers_.data(), route_slices_.data(),
              source_level_limit, balanced_merge_location_counts_.data());
      CUDA_CHECK(cudaMemcpyAsync(
          balanced_merge_location_counts_host_.data(),
          balanced_merge_location_counts_.data(),
          starting_locations.size() * sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    bool open_job = false;
    bool job_reuses_page = false;
    auto flush_job = [&]() {
      if (!open_job) return;
      balanced_merge_jobs_host_[normal_job_count++] =
          {job_key_begin, job_key_end,
           job_existing_offset, job_begin, job_end,
           job_reuses_page ? 1u : 0u, job_existing_capacity};
      open_job = false;
      job_work = 0u;
      job_reuses_page = false;
      job_existing_offset = 0u;
      job_existing_capacity = 0u;
      job_key_begin = 0u;
      job_key_end = 0u;
    };
    auto append_quotient = [&](std::uint32_t q) {
      const std::uint32_t work = balanced_merge_raw_counts_host_[q];
      if (work > gpulsmopt2_detail::kBalancedMergeTarget) {
        flush_job();
        if (hot_count == foundation_hot_sections_host_.size())
          foundation_hot_sections_host_.resize(
              foundation_hot_sections_host_.size() * 2u);
        foundation_hot_sections_host_[hot_count] = q;
        foundation_hot_lows_host_.push_back(0u);
        foundation_hot_highs_host_.push_back(1u << 16u);
        foundation_hot_raw_counts_host_.push_back(work);
        ++hot_count;
      } else if (work) {
        if (!open_job) {
          job_begin = q;
          job_end = q + 1u;
          job_key_begin = std::uint64_t{q} << 16u;
          job_key_end = std::uint64_t{q + 1u} << 16u;
          job_work = work;
          open_job = true;
          job_reuses_page = false;
        } else if (job_work + work <=
                       gpulsmopt2_detail::kBalancedMergeTarget &&
                   q - job_begin <
                       gpulsmopt2_detail::kBalancedMergeMaximumQuotients) {
          job_end = q + 1u;
          job_key_end = std::uint64_t{q + 1u} << 16u;
          job_work += work;
          job_existing_offset = 0u;
          job_existing_capacity = 0u;
        } else {
          flush_job();
          job_begin = q;
          job_end = q + 1u;
          job_key_begin = std::uint64_t{q} << 16u;
          job_key_end = std::uint64_t{q + 1u} << 16u;
          job_work = work;
          open_job = true;
          job_reuses_page = false;
          job_existing_offset = 0u;
          job_existing_capacity = 0u;
        }
      }
    };
    auto process_starting_location =
        [&](const gpulsmopt2_detail::BalancedMergeJob &location,
            std::uint32_t work) {
      if (!work) return;
      const std::uint32_t span =
          location.quotient_end - location.quotient_begin;
      if (work <= gpulsmopt2_detail::kBalancedMergeTarget &&
          span <= gpulsmopt2_detail::kBalancedMergeMaximumQuotients) {
        if (open_job &&
            job_work + work <= gpulsmopt2_detail::kBalancedMergeTarget &&
            location.quotient_end - job_begin <=
                gpulsmopt2_detail::kBalancedMergeMaximumQuotients) {
          job_end = location.quotient_end;
          job_key_end = location.key_end;
          job_work += work;
          job_reuses_page = false;
          job_existing_offset = 0u;
          job_existing_capacity = 0u;
        } else {
          flush_job();
          job_begin = location.quotient_begin;
          job_end = location.quotient_end;
          job_key_begin = location.key_begin;
          job_key_end = location.key_end;
          job_work = work;
          open_job = true;
          job_reuses_page = location.reuse_existing_page != 0u;
          job_existing_offset = location.existing_offset;
          job_existing_capacity = location.existing_capacity;
        }
        return;
      }
      flush_job();
      if (span == 1u) {
        const std::uint32_t q = location.quotient_begin;
        if (hot_count == foundation_hot_sections_host_.size())
          foundation_hot_sections_host_.resize(
              foundation_hot_sections_host_.size() * 2u);
        foundation_hot_sections_host_[hot_count] = q;
        foundation_hot_lows_host_.push_back(static_cast<std::uint32_t>(
            location.key_begin - (std::uint64_t{q} << 16u)));
        foundation_hot_highs_host_.push_back(static_cast<std::uint32_t>(
            location.key_end - (std::uint64_t{q} << 16u)));
        foundation_hot_raw_counts_host_.push_back(work);
        ++hot_count;
        return;
      }
      for (std::uint32_t q = location.quotient_begin;
           q < location.quotient_end; ++q)
        append_quotient(q);
      flush_job();
    };

    foundation_hot_lows_host_.clear();
    foundation_hot_highs_host_.clear();
    foundation_hot_raw_counts_host_.clear();
    if (starting_locations.empty())
      process_starting_location(
          {0u, std::uint64_t{1} << 32u,
           0u, 0u, gpulsmopt2_detail::kQuotients},
          std::accumulate(balanced_merge_raw_counts_host_.begin(),
                          balanced_merge_raw_counts_host_.end(), 0u));
    else
      for (std::size_t index = 0u; index < starting_locations.size(); ++index)
        process_starting_location(
            starting_locations[index],
            balanced_merge_location_counts_host_[index]);
    flush_job();
    std::uint32_t hot_task_count = 0u;
    std::uint32_t hot_boundary_slots = 0u;
    if (hot_count) {
      if (foundation_hot_range_counts_host_.size() < hot_count)
        foundation_hot_range_counts_host_.resize(hot_count);
      if (foundation_hot_range_bases_host_.size() < hot_count + 1u)
        foundation_hot_range_bases_host_.resize(hot_count + 1u);
      foundation_hot_tasks_host_.clear();
      for (std::uint32_t hot_index = 0u; hot_index < hot_count;
           ++hot_index) {
        const std::uint32_t q = foundation_hot_sections_host_[hot_index];
        const std::uint32_t raw =
            foundation_hot_raw_counts_host_[hot_index];
        const std::uint32_t safe_target =
            gpulsmopt2_detail::kBalancedMergeTarget -
            (source_level_limit + 2u);
        const std::uint32_t ranges = std::max(
            1u, (raw + safe_target - 1u) / safe_target);
        if (ranges > gpulsmopt2_detail::kBalancedHotRanges)
          throw std::overflow_error(
              "GPULSMOpt crowded range exceeds key-space bound");
        foundation_hot_range_counts_host_[hot_index] = ranges;
        foundation_hot_range_bases_host_[hot_index] = hot_boundary_slots;
        hot_boundary_slots += ranges + 1u;
        for (std::uint32_t range = 0u; range < ranges; ++range)
          foundation_hot_tasks_host_.push_back({hot_index, range});
      }
      foundation_hot_range_bases_host_[hot_count] = hot_boundary_slots;
      hot_task_count = static_cast<std::uint32_t>(
          foundation_hot_tasks_host_.size());
      ensure_crowded_merge_workspace(
          hot_task_count, hot_boundary_slots, stream);
      CUDA_CHECK(cudaMemcpyAsync(
          foundation_hot_sections_.data(),
          foundation_hot_sections_host_.data(),
          std::size_t{hot_count} * sizeof(std::uint32_t),
          cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          foundation_hot_range_counts_.data(),
          foundation_hot_range_counts_host_.data(),
          std::size_t{hot_count} * sizeof(std::uint32_t),
          cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          foundation_hot_range_bases_.data(),
          foundation_hot_range_bases_host_.data(),
          std::size_t{hot_count + 1u} * sizeof(std::uint32_t),
          cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          foundation_hot_lows_.data(), foundation_hot_lows_host_.data(),
          std::size_t{hot_count} * sizeof(std::uint32_t),
          cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          foundation_hot_highs_.data(), foundation_hot_highs_host_.data(),
          std::size_t{hot_count} * sizeof(std::uint32_t),
          cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          foundation_hot_raw_counts_.data(),
          foundation_hot_raw_counts_host_.data(),
          std::size_t{hot_count} * sizeof(std::uint32_t),
          cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          foundation_hot_tasks_.data(), foundation_hot_tasks_host_.data(),
          std::size_t{hot_task_count} *
              sizeof(gpulsmopt2_detail::CrowdedMergeTask),
          cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemsetAsync(
          foundation_hot_cell_counts_.data(), 0,
          std::size_t{hot_boundary_slots} * sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(
          foundation_hot_cell_offsets_.data(), 0,
          std::size_t{hot_boundary_slots} * sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(
          foundation_hot_streaming_flags_.data(), 0,
          std::size_t{hot_task_count} * sizeof(std::uint8_t), stream));
      gpulsmopt2_detail::plan_crowded_merge_ranges_kernel<<<
          hot_count, gpulsmopt2_detail::kFoundationCompactionThreads,
          0, stream>>>(
              foundation_hot_sections_.data(), hot_count,
              foundation_hot_lows_.data(), foundation_hot_highs_.data(),
              foundation_hot_raw_counts_.data(), current_rows,
              foundation_source_offsets_.data(), arena_.data(),
              route_headers_.data(), route_slices_.data(),
              source_level_limit,
              foundation_hot_range_counts_.data(),
              foundation_hot_range_bases_.data(),
              foundation_hot_range_boundaries_.data());
      foundation_hot_range_boundaries_host_.resize(hot_boundary_slots);
      CUDA_CHECK(cudaMemcpyAsync(
          foundation_hot_range_boundaries_host_.data(),
          foundation_hot_range_boundaries_.data(),
          std::size_t{hot_boundary_slots} * sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::uint32_t total_job_count = normal_job_count;
    std::vector<std::uint32_t> adaptive_route_counts(
        gpulsmopt2_detail::kQuotients, 0u);
    for (std::uint32_t hot_index = 0u; hot_index < hot_count;
         ++hot_index) {
      const std::uint32_t q = foundation_hot_sections_host_[hot_index];
      const std::uint32_t base =
          foundation_hot_range_bases_host_[hot_index];
      const std::uint32_t proposed =
          foundation_hot_range_counts_host_[hot_index];
      std::uint32_t emitted = 0u;
      for (std::uint32_t range = 0u; range < proposed; ++range) {
        const std::uint32_t low =
            foundation_hot_range_boundaries_host_[base + range];
        const std::uint32_t high =
            foundation_hot_range_boundaries_host_[base + range + 1u];
        if (low == high) continue;
        if (total_job_count == balanced_merge_jobs_host_.size())
          balanced_merge_jobs_host_.resize(
              balanced_merge_jobs_host_.size() * 2u);
        balanced_merge_jobs_host_[total_job_count++] = {
            (std::uint64_t{q} << 16u) + low,
            (std::uint64_t{q} << 16u) + high,
            0u, q, q + 1u, 0u, 0u, 0u, 0u, 0u,
            adaptive_route_counts[q]++};
        ++emitted;
      }
      foundation_hot_range_counts_host_[hot_index] = emitted;
    }

    // Crowded locations are appended after normal locations while they are
    // planned.  Restore global key order before assigning route ordinals so a
    // quotient containing several differently sized locations always has its
    // route slices in increasing suffix order.
    std::sort(balanced_merge_jobs_host_.begin(),
              balanced_merge_jobs_host_.begin() + total_job_count,
              [](const gpulsmopt2_detail::BalancedMergeJob &left,
                 const gpulsmopt2_detail::BalancedMergeJob &right) {
                if (left.key_begin != right.key_begin)
                  return left.key_begin < right.key_begin;
                return left.key_end < right.key_end;
              });

    std::fill(foundation_route_counts_host_.begin(),
              foundation_route_counts_host_.end(), 0u);
    for (std::uint32_t index = 0u; index < total_job_count; ++index) {
      auto &job = balanced_merge_jobs_host_[index];
      if (job.quotient_end - job.quotient_begin == 1u) {
        job.route_ordinal =
            foundation_route_counts_host_[job.quotient_begin]++;
      } else {
        job.route_ordinal = 0u;
        for (std::uint32_t q = job.quotient_begin;
             q < job.quotient_end; ++q) {
          if (foundation_route_counts_host_[q])
            throw std::logic_error(
                "GPULSMOpt planner produced overlapping locations");
          foundation_route_counts_host_[q] = 1u;
        }
      }
    }
    std::uint64_t route_cursor =
        std::uint64_t{destination_level} * route_stride_;
    const std::uint64_t route_limit = route_cursor + route_stride_;
    for (std::uint32_t q = 0u;
         q < gpulsmopt2_detail::kQuotients; ++q) {
      const std::uint32_t count = foundation_route_counts_host_[q];
      if (route_cursor + count > route_limit)
        throw std::overflow_error("GPULSMOpt adaptive route table exhausted");
      foundation_next_route_headers_host_[q] = {
          static_cast<std::uint32_t>(route_cursor), count};
      route_cursor += count;
    }
    CUDA_CHECK(cudaMemcpyAsync(
        foundation_next_route_headers_.data(),
        foundation_next_route_headers_host_.data(),
        gpulsmopt2_detail::kQuotients *
            sizeof(gpulsmopt2_detail::RouteHeader),
        cudaMemcpyHostToDevice, stream));

    std::uint64_t pull_slice_slots = 0u;
    for (std::uint32_t index = 0u; index < total_job_count; ++index) {
      auto &job = balanced_merge_jobs_host_[index];
      std::uint64_t slots =
          job.quotient_end - job.quotient_begin;
      for (std::uint32_t level = 0u; level <= source_level_limit;
           ++level) {
        const auto &counts = level_route_counts_[level];
        for (std::uint32_t q = job.quotient_begin;
             q < job.quotient_end; ++q)
          slots += counts.empty() ? 1u : std::max<std::uint32_t>(
              1u, counts[q]);
      }
      if (pull_slice_slots + slots + 1u >
          std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("GPULSMOpt pull-slice metadata exhausted");
      job.slice_begin = static_cast<std::uint32_t>(pull_slice_slots);
      job.slice_count = static_cast<std::uint32_t>(slots);
      pull_slice_slots += slots + 1u;
    }
    if (total_job_count) {
      ensure_balanced_job_workspace(total_job_count, stream);
      ensure_pull_slice_workspace(
          static_cast<std::uint32_t>(pull_slice_slots), stream);
      CUDA_CHECK(cudaMemcpyAsync(
          balanced_merge_jobs_.data(), balanced_merge_jobs_host_.data(),
          std::size_t{total_job_count} *
              sizeof(gpulsmopt2_detail::BalancedMergeJob),
          cudaMemcpyHostToDevice, stream));
      gpulsmopt2_detail::prepare_balanced_pull_slices_kernel<<<
          total_job_count, 1u, 0, stream>>>(
              balanced_merge_jobs_.data(), total_job_count, current_rows,
              foundation_source_offsets_.data(), arena_.data(),
              route_headers_.data(), route_slices_.data(),
              source_level_limit, balanced_merge_pull_slices_.data());
      CUDA_CHECK(cudaGetLastError());
    }

    unsigned long long *next_output_offset =
        reinterpret_cast<unsigned long long *>(
            destination_is_foundation
                ? foundation_next_offset_.data()
                : balanced_merge_next_offset_.data());
    const std::uint64_t foundation_bank_capacity =
        foundation_pool_capacity_ / 2u;
    bool foundation_in_place = destination_is_foundation;
    std::uint64_t growth_bound = 0u;
    if (foundation_in_place) {
      const std::uint64_t bank_end =
          std::uint64_t{foundation_bank_ + 1u} * foundation_bank_capacity;
      for (std::uint32_t index = 0u; index < total_job_count; ++index) {
        const auto &job = balanced_merge_jobs_host_[index];
        std::uint32_t raw_bound = 0u;
        if (job.quotient_end - job.quotient_begin == 1u &&
            job.key_begin != (std::uint64_t{job.quotient_begin} << 16u))
          raw_bound = gpulsmopt2_detail::kBalancedMergeTarget;
        else
          for (std::uint32_t q = job.quotient_begin;
               q < job.quotient_end; ++q)
            raw_bound += balanced_merge_raw_counts_host_[q];
        if (!job.reuse_existing_page || raw_bound > job.existing_capacity)
          growth_bound += gpulsmopt2_detail::foundation_section_capacity(
              raw_bound);
      }
      foundation_in_place = foundation_tail_ + growth_bound <= bank_end;
    }
    const std::uint32_t next_foundation_bank = destination_is_foundation
        ? (foundation_in_place ? foundation_bank_ : 1u - foundation_bank_)
        : foundation_bank_;
    const std::uint64_t initial_output_offset = destination_is_foundation
        ? (foundation_in_place
               ? foundation_tail_
               : std::uint64_t{next_foundation_bank} *
                     foundation_bank_capacity)
        : destination_span;
    const unsigned long long output_limit = destination_is_foundation
        ? initial_output_offset + foundation_bank_capacity
        : static_cast<unsigned long long>(destination_span) +
              destination_capacity;
    CUDA_CHECK(cudaMemcpyAsync(
        destination_is_foundation
            ? foundation_next_offset_.data()
            : balanced_merge_next_offset_.data(),
        &initial_output_offset, sizeof(initial_output_offset),
        cudaMemcpyHostToDevice, stream));

    if (total_job_count)
      gpulsmopt2_detail::compact_balanced_merge_jobs_kernel<<<
          total_job_count,
          gpulsmopt2_detail::kFoundationCompactionThreads, 0, stream>>>(
            balanced_merge_jobs_.data(), total_job_count,
            balanced_merge_raw_counts_.data(),
            balanced_merge_pull_slices_.data(),
            current_rows, foundation_source_offsets_.data(), arena_.data(),
            descriptors_.data(), source_level_limit,
            !destination_is_foundation, destination_is_foundation,
            foundation_in_place,
            foundation_capacities_.data(),
            next_output_offset, output_limit,
            foundation_next_descriptors_.data(),
            foundation_next_capacities_.data(),
            foundation_next_route_headers_.data(), route_slices_.data(),
            foundation_section_output_counts_.data(),
            foundation_total_output_count_.data(),
            foundation_overflow_flag_.data());
    gpulsmopt2_detail::summarize_adaptive_routes_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            foundation_next_route_headers_.data(), route_slices_.data(),
            destination_is_foundation,
            foundation_next_descriptors_.data(),
            foundation_next_capacities_.data());
    gpulsmopt2_detail::finalize_foundation_descriptors_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            foundation_next_descriptors_.data(),
            foundation_next_capacities_.data(), destination_level,
            descriptors_.data(), foundation_capacities_.data(),
            destination_is_foundation);
    gpulsmopt2_detail::publish_adaptive_route_directory_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            foundation_next_route_headers_.data(), destination_level,
            route_headers_.data());
    std::uint32_t total_count{}, overflow{};
    std::uint64_t final_output_offset{};
    CUDA_CHECK(cudaMemcpyAsync(
        &total_count, foundation_total_output_count_.data(),
        sizeof(total_count), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &overflow, foundation_overflow_flag_.data(), sizeof(overflow),
        cudaMemcpyDeviceToHost, stream));
    if (total_job_count)
      CUDA_CHECK(cudaMemcpyAsync(
          balanced_merge_jobs_host_.data(), balanced_merge_jobs_.data(),
          std::size_t{total_job_count} *
              sizeof(gpulsmopt2_detail::BalancedMergeJob),
          cudaMemcpyDeviceToHost, stream));
    if (destination_is_foundation)
      CUDA_CHECK(cudaMemcpyAsync(
          foundation_capacities_host_.data(),
          foundation_next_capacities_.data(),
          gpulsmopt2_detail::kQuotients * sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost, stream));
    if (destination_is_foundation)
      CUDA_CHECK(cudaMemcpyAsync(
          &final_output_offset, foundation_next_offset_.data(),
          sizeof(final_output_offset), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (overflow)
      throw std::overflow_error("GPULSMOpt balanced merge output exhausted");
    if (destination_is_foundation) {
      foundation_bank_ = next_foundation_bank;
      foundation_tail_ = final_output_offset;
    }
    std::vector<gpulsmopt2_detail::BalancedMergeJob> next_locations;
    next_locations.reserve(total_job_count);
    for (std::uint32_t index = 0u; index < total_job_count; ++index) {
      auto location = balanced_merge_jobs_host_[index];
      location.slice_begin = 0u;
      location.slice_count = 0u;
      next_locations.push_back(location);
    }
    std::sort(next_locations.begin(), next_locations.end(),
              [](const auto &left, const auto &right) {
                return left.key_begin < right.key_begin;
              });
    std::vector<gpulsmopt2_detail::BalancedMergeJob> complete_locations;
    complete_locations.reserve(
        next_locations.size() + gpulsmopt2_detail::kQuotients);
    std::uint64_t covered_key = 0u;
    for (const auto &location : next_locations) {
      while (covered_key < location.key_begin) {
        const std::uint32_t q =
            static_cast<std::uint32_t>(covered_key >> 16u);
        const std::uint64_t end = min(
            location.key_begin, std::uint64_t{q + 1u} << 16u);
        complete_locations.push_back({
            covered_key, end, 0u, q, q + 1u});
        covered_key = end;
      }
      complete_locations.push_back(location);
      covered_key = max(covered_key, location.key_end);
    }
    while (covered_key < (std::uint64_t{1} << 32u)) {
      const std::uint32_t q =
          static_cast<std::uint32_t>(covered_key >> 16u);
      const std::uint64_t end = std::uint64_t{q + 1u} << 16u;
      complete_locations.push_back({
          covered_key, end, 0u, q, q + 1u});
      covered_key = end;
    }
    for (std::uint32_t level = 0u; level < destination_level; ++level) {
      level_counts_[level] = 0u;
      level_offsets_[level] = 0u;
      level_span_sizes_[level] = 0u;
      level_locations_[level].clear();
      level_route_counts_[level].clear();
    }
    level_locations_[destination_level] = std::move(complete_locations);
    auto &destination_route_counts = level_route_counts_[destination_level];
    destination_route_counts.assign(
        gpulsmopt2_detail::kQuotients, 0u);
    for (std::uint32_t q = 0u; q < gpulsmopt2_detail::kQuotients; ++q)
      destination_route_counts[q] = static_cast<std::uint16_t>(
          foundation_route_counts_host_[q]);
    has_adaptive_routes_ = std::any_of(
        foundation_route_counts_host_.begin(),
        foundation_route_counts_host_.end(),
        [](std::uint32_t count) { return count > 1u; });
    level_counts_[destination_level] = total_count;
    level_offsets_[destination_level] =
        destination_is_foundation ? 0u : destination_span;
    level_span_sizes_[destination_level] =
        destination_is_foundation ? 0u : destination_capacity;
    return total_count;
  }

  void publish_epoch(cudaStream_t stream) {
    const std::uint32_t old_foundation = foundation_level();
    const std::uint32_t publication_levels =
        levels_for_epochs(stats_.epochs_published + 1u);
    if (publication_levels > gpulsmopt2_detail::kMaximumLevels)
      throw std::overflow_error("GPULSMOpt epoch counter exhausted");
    ensure_publication_capacity(pending_records_, stream);
    std::uint32_t carry_levels = 0u;
    while (carry_levels < gpulsmopt2_detail::kMaximumLevels &&
           level_counts_[carry_levels])
      ++carry_levels;
    if (carry_levels == gpulsmopt2_detail::kMaximumLevels)
      throw std::overflow_error("GPULSMOpt epoch counter exhausted");
    std::uint64_t destination = gpulsmopt2_detail::kInvalidOffset;
    if (carry_levels == 0u)
      destination = allocate_level_span(pending_records_, stream);
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
    auto row_output = thrust::make_transform_output_iterator(
        carry_levels ? publication_rows_a_.data()
                     : arena_.data() + destination,
        gpulsmopt2_detail::AssignmentRow{});
    workspace_bytes = publication_temp_.size();
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        publication_temp_.data(), workspace_bytes,
        publication_epoch_keys_b_.data(),
        carry_levels ? publication_keys_a_.data()
                     : level_keys_.data() + destination,
        publication_epoch_assignments_b_.data(), row_output,
        publication_selected_count_.data(),
        gpulsmopt2_detail::NewestAssignment{}, pending_records_, stream));
    std::uint32_t current_count{};
    CUDA_CHECK(cudaMemcpyAsync(
        &current_count, publication_selected_count_.data(),
        sizeof(current_count), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::uint32_t destination_level = carry_levels;
    const bool foundation_carry =
        old_foundation < gpulsmopt2_detail::kMaximumLevels &&
        old_foundation < carry_levels;
    if (carry_levels) {
      std::size_t merged_size = current_count;
      for (std::uint32_t level = 0u; level < carry_levels; ++level) {
        merged_size += level_counts_[level];
      }
      if (merged_size > gpulsmopt2_detail::kMaximumPublicationRows)
        throw std::bad_alloc();
      ensure_publication_capacity(merged_size, stream);
      const std::uint32_t merged_count =
          static_cast<std::uint32_t>(merged_size);
      if (!foundation_carry)
        destination = allocate_level_span(merged_count, stream);
      current_count = publish_balanced_merge(
          publication_keys_a_.data(), publication_rows_a_.data(),
          current_count, carry_levels - 1u, destination_level,
          foundation_carry,
          foundation_carry ? 0u : destination, merged_count, stream);
      refresh_active_levels();
      if (foundation_carry)
        rebuild_foundation_rank(stream);
      else
        rebuild_level_guide(destination_level, stream);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemsetAsync(
          raw_epoch_signatures_.data(), 0,
          raw_epoch_signatures_.size() * sizeof(std::uint64_t), stream));
      pending_batches_ = 0u;
      pending_records_ = 0u;
      ++stats_.epochs_published;
      return;
    }
    gpulsmopt2_detail::publish_global_level_descriptors_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            level_keys_.data() + destination, current_count, destination,
            destination_level, descriptors_.data());
    ensure_route_level(destination_level, stream);
    gpulsmopt2_detail::publish_single_route_directory_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            descriptors_.data(), destination_level,
            static_cast<std::uint32_t>(route_stride_),
            route_headers_.data(), route_slices_.data());
    level_offsets_[destination_level] = destination;
    level_span_sizes_[destination_level] = current_count;
    level_counts_[destination_level] = current_count;
    level_locations_[destination_level] = {
        {0u, std::uint64_t{1} << 32u,
         0u, 0u, gpulsmopt2_detail::kQuotients, 1u}};
    level_route_counts_[destination_level].assign(
        gpulsmopt2_detail::kQuotients, 1u);
    has_adaptive_routes_ = false;
    refresh_active_levels();
    if (destination_level != foundation_level())
      rebuild_level_guide(destination_level, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(
        raw_epoch_signatures_.data(), 0,
        raw_epoch_signatures_.size() * sizeof(std::uint64_t), stream));
    pending_batches_ = 0u;
    pending_records_ = 0u;
    if (foundation_level() != old_foundation)
      rebuild_foundation_rank(stream);
    ++stats_.epochs_published;
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
  template <bool HasPending, bool Tiled>
  void launch_section_ranges(std::uint32_t maximum_tasks,
                             cudaStream_t stream) {
    const std::uint32_t grid =
        Tiled ? maximum_tasks : gpulsmopt2_detail::kQuotients;
    const auto *tasks = Tiled ? range_section_tasks_.data() : nullptr;
    const auto *task_count = Tiled
        ? range_section_task_offsets_.data() + gpulsmopt2_detail::kQuotients
        : nullptr;
    gpulsmopt2_detail::cooperative_section_owned_range_kernel<
        gpulsmopt2_detail::SumRowsAggregate, HasPending, Tiled>
        <<<grid, gpulsmopt2_detail::kSectionRangeThreads, 0, stream>>>(
            range_section_fragments_out_.data(),
            range_section_offsets_.data(), tasks, task_count,
            arena_.data(), descriptors_.data(), route_headers_.data(),
            route_slices_.data(), raw_keys_.data(),
            raw_payloads_.data(),
            raw_offsets_.data(), static_cast<std::uint32_t>(batch_capacity_),
            HasPending ? pending_batches_ : 0u,
            active_levels_,
            active_levels_ ? active_levels_ - 1u
                           : gpulsmopt2_detail::kInvalid,
            range_fragment_partials_.data());
  }
  template <bool HasPending>
  void launch_fragment_ranges(std::uint32_t fragment_count,
                              std::uint32_t query_count,
                              const DeviceRangeOutputBatch &batch,
                              cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(range_overflow_count_.data(), 0,
                               sizeof(std::uint32_t), stream));
    gpulsmopt2_detail::warp_range_fragment_kernel<
        gpulsmopt2_detail::SumRowsAggregate, HasPending>
        <<<(fragment_count + 3u) / 4u, 128, 0, stream>>>(
            range_fragments_.data(), fragment_count,
            range_fragment_offsets_.data() + query_count,
            batch.lo, batch.hi, arena_.data(), descriptors_.data(),
            route_headers_.data(), route_slices_.data(),
            raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
            static_cast<std::uint32_t>(batch_capacity_),
            HasPending ? pending_batches_ : 0u,
            active_levels_,
            active_levels_ ? active_levels_ - 1u
                           : gpulsmopt2_detail::kInvalid,
            range_fragment_partials_.data(),
            range_overflow_fragments_.data(), range_overflow_count_.data());
    gpulsmopt2_detail::overflow_range_fragment_kernel<HasPending>
        <<<256, 128, 0, stream>>>(
            range_overflow_fragments_.data(), range_overflow_count_.data(),
            range_fragments_.data(), batch.lo, batch.hi,
            arena_.data(), descriptors_.data(), route_headers_.data(),
            route_slices_.data(), raw_keys_.data(),
            raw_payloads_.data(),
            raw_offsets_.data(), static_cast<std::uint32_t>(batch_capacity_),
            HasPending ? pending_batches_ : 0u,
            active_levels_,
            range_fragment_partials_.data());
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
    bytes += aligned_range_bytes(count * sizeof(std::uint32_t));
    bytes += aligned_range_bytes(count * sizeof(unsigned long long));
    range_fragment_storage_.resize(bytes);
    std::size_t offset = 0u;
    attach_range_view(range_fragments_, range_fragment_storage_.data(),
                      offset, count);
    attach_range_view(range_overflow_fragments_,
                      range_fragment_storage_.data(), offset, count);
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
    bytes += aligned_range_bytes(sizeof(std::uint32_t));
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
    attach_range_view(range_section_max_fragments_,
                      range_section_storage_.data(), offset, 1u);
    offset = aligned_range_bytes(offset);
    range_section_temp_ = range_section_storage_.data() + offset;
    range_section_temp_bytes_ = temp_bytes;
  }
  std::size_t batch_capacity_{};
  std::size_t publication_capacity_{};
  std::size_t foundation_pool_capacity_{};
  std::size_t route_stride_{};
  std::uint32_t foundation_bank_{};
  std::uint64_t foundation_tail_{};
  bool has_adaptive_routes_{};
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

  gpulsmopt2_detail::Buffer<std::uint16_t> local_rank_;
  gpulsmopt2_detail::VirtualBuffer<std::uint16_t> level_guides_;
  gpulsmopt2_detail::VirtualBuffer<gpulsmopt2_detail::Row> arena_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::Descriptor> descriptors_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RouteHeader> route_headers_;
  gpulsmopt2_detail::VirtualBuffer<gpulsmopt2_detail::RouteSlice>
      route_slices_;
  gpulsmopt2_detail::Buffer<std::uint32_t> raw_keys_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RawPayload> raw_payloads_;
  gpulsmopt2_detail::Buffer<std::uint32_t> raw_offsets_;
  gpulsmopt2_detail::Buffer<std::uint64_t> raw_signatures_;
  gpulsmopt2_detail::Buffer<std::uint64_t> raw_epoch_signatures_;
  gpulsmopt2_detail::VirtualBuffer<std::uint32_t> level_keys_;
  gpulsmopt2_detail::Buffer<std::uint32_t> publication_epoch_keys_a_,
      publication_epoch_keys_b_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RawAssignment>
      publication_epoch_assignments_a_, publication_epoch_assignments_b_;
  gpulsmopt2_detail::VirtualBuffer<std::uint32_t> publication_keys_a_,
      publication_keys_b_;
  gpulsmopt2_detail::Buffer<std::uint32_t> publication_selected_count_,
      publication_batch_offsets_;
  gpulsmopt2_detail::Buffer<std::uint32_t> foundation_source_offsets_,
      foundation_build_capacities_, foundation_physical_offsets_,
      foundation_capacities_, foundation_next_capacities_,
      foundation_section_output_counts_, balanced_merge_raw_counts_,
      balanced_merge_location_counts_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::BalancedMergeJob>
      balanced_merge_jobs_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::PullSlice>
      balanced_merge_pull_slices_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::CrowdedMergeTask>
      foundation_hot_tasks_;
  gpulsmopt2_detail::Buffer<std::uint32_t>
      balanced_merge_job_count_, foundation_hot_sections_,
      foundation_hot_selected_count_, foundation_hot_range_counts_,
      foundation_hot_range_bases_,
      foundation_hot_lows_, foundation_hot_highs_,
      foundation_hot_raw_counts_,
      foundation_hot_staging_offsets_, foundation_hot_staging_next_,
      foundation_hot_cell_counts_, foundation_hot_cell_offsets_,
      foundation_hot_range_boundaries_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::Descriptor>
      foundation_next_descriptors_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RouteHeader>
      foundation_next_route_headers_;
  gpulsmopt2_detail::Buffer<std::uint8_t> foundation_hot_flags_,
      foundation_hot_streaming_flags_, foundation_temp_;
  gpulsmopt2_detail::Buffer<std::uint64_t> foundation_next_offset_,
      balanced_merge_next_offset_;
  gpulsmopt2_detail::Buffer<std::uint32_t> foundation_total_output_count_,
      foundation_overflow_flag_;
  gpulsmopt2_detail::VirtualBuffer<gpulsmopt2_detail::Row>
      publication_rows_a_, publication_rows_b_;
  gpulsmopt2_detail::Buffer<std::uint8_t> publication_temp_;
  gpulsmopt2_detail::Buffer<std::uint32_t> admission_counts_;
  gpulsmopt2_detail::Buffer<std::uint8_t> admission_temp_;
  std::uint32_t level_counts_[gpulsmopt2_detail::kMaximumLevels]{};
  std::uint64_t level_offsets_[gpulsmopt2_detail::kMaximumLevels]{};
  std::uint64_t level_span_sizes_[gpulsmopt2_detail::kMaximumLevels]{};
  std::uint32_t raw_batch_counts_[gpulsmopt2_detail::kBatchesPerEpoch]{};
  std::array<std::vector<gpulsmopt2_detail::BalancedMergeJob>,
             gpulsmopt2_detail::kMaximumLevels> level_locations_{};
  std::array<std::vector<std::uint16_t>,
             gpulsmopt2_detail::kMaximumLevels> level_route_counts_{};
  std::vector<std::uint32_t> balanced_merge_raw_counts_host_ =
      std::vector<std::uint32_t>(gpulsmopt2_detail::kQuotients);
  std::vector<std::uint32_t> foundation_capacities_host_ =
      std::vector<std::uint32_t>(gpulsmopt2_detail::kQuotients);
  std::vector<std::uint32_t> foundation_physical_offsets_host_ =
      std::vector<std::uint32_t>(gpulsmopt2_detail::kQuotients + 1u);
  std::vector<std::uint32_t> foundation_route_counts_host_ =
      std::vector<std::uint32_t>(gpulsmopt2_detail::kQuotients);
  std::vector<gpulsmopt2_detail::RouteHeader>
      foundation_next_route_headers_host_ =
          std::vector<gpulsmopt2_detail::RouteHeader>(
              gpulsmopt2_detail::kQuotients);
  std::vector<gpulsmopt2_detail::BalancedMergeJob>
      balanced_merge_jobs_host_ =
          std::vector<gpulsmopt2_detail::BalancedMergeJob>(
              gpulsmopt2_detail::kQuotients);
  std::vector<std::uint32_t> foundation_hot_sections_host_ =
      std::vector<std::uint32_t>(gpulsmopt2_detail::kQuotients);
  std::vector<std::uint32_t> foundation_hot_range_counts_host_ =
      std::vector<std::uint32_t>(gpulsmopt2_detail::kQuotients);
  std::vector<std::uint32_t> foundation_hot_range_bases_host_ =
      std::vector<std::uint32_t>(gpulsmopt2_detail::kQuotients + 1u);
  std::vector<std::uint32_t> balanced_merge_location_counts_host_;
  std::vector<std::uint32_t> foundation_hot_lows_host_;
  std::vector<std::uint32_t> foundation_hot_highs_host_;
  std::vector<std::uint32_t> foundation_hot_raw_counts_host_;
  std::vector<gpulsmopt2_detail::CrowdedMergeTask>
      foundation_hot_tasks_host_;
  std::vector<std::uint32_t> foundation_hot_range_boundaries_host_;

  gpulsmopt2_detail::Buffer<std::uint8_t> radix_storage_;
  gpulsmopt2_detail::Buffer<std::uint32_t> radix_keys_, radix_ids_out_;
  gpulsmopt2_detail::Buffer<std::uint32_t> lookup_quotient_offsets_;

  gpulsmopt2_detail::Buffer<unsigned long long> range_partials_;
  gpulsmopt2_detail::Buffer<std::uint8_t> range_query_storage_,
      range_fragment_storage_, range_section_storage_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_fragment_counts_,
      range_fragment_offsets_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RangeFragment> range_fragments_;
  gpulsmopt2_detail::Buffer<unsigned long long> range_fragment_partials_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_overflow_fragments_,
      range_overflow_count_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_section_keys_in_,
      range_section_keys_out_, range_section_offsets_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::SectionRangeFragment>
      range_section_fragments_in_, range_section_fragments_out_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::SectionRangeTask>
      range_section_tasks_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_section_task_counts_,
      range_section_task_offsets_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_section_max_fragments_;
};
