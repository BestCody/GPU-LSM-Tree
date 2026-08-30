#pragma once

#include "capsule_lifecycle.cuh"

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
