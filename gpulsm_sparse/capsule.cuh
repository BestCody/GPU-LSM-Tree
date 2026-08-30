#pragma once

#include "common.cuh"

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

struct CapsuleLifecycleSegmentView {
  std::uint32_t ordinal{};
  std::uint32_t reserved{};
  std::uint64_t live_bytes{};
  std::uint64_t garbage_bytes{};
};

struct CapsuleLifecycleInput {
  const CapsuleLifecycleSegmentView *segments{};
  std::uint32_t count{};
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

struct DeviceRootOverlay {
  const ExactHeadDescriptor *exact_heads{};
  const gpulsmopt2_detail::Row *exact_rows{};
  std::uint32_t exact_head_count{};
  std::uint32_t segment_ordinal{};
  DeviceCapsulePlane capsules{};
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

struct StagedPendingOverlay {
  std::array<std::shared_ptr<StagedPendingSlotOverlay>,
             gpulsmopt2_detail::kBatchesPerEpoch> slots;
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
