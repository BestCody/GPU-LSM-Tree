#pragma once

#include "exact.cuh"

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
