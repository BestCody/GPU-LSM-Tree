#pragma once

// Pending batches: unsorted, grouped by region, with per-region filters.

#include "layout.cuh"

namespace celllsm {

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

static_assert(sizeof(RawAssignment) == 12u);
static_assert(sizeof(RawPayload) == 8u);
static_assert(alignof(RawPayload) == 8u);

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

__device__ RawAssignment load_pending_raw_ordinal(
    const std::uint32_t *keys, const RawPayload *payloads,
    const std::uint32_t *offsets,
    std::uint32_t batch_stride, std::uint32_t pending_batches,
    std::uint32_t q, std::uint32_t ordinal) {
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t oi = std::size_t{batch} * (kRegions + 1u) + q;
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

__device__ __forceinline__ std::uint64_t pending_signature_bits(
    std::uint32_t key) {
  const std::uint32_t first = key * 0x9e3779b1u;
  const std::uint32_t second = (key ^ (key >> 16u)) * 0x85ebca6bu;
  return (1ull << (first >> 26u)) | (1ull << (second >> 26u));
}

// Share reservations; track active regions only for shared lookup.
template <bool TrackActive = false>
__global__ void count_regions_kernel(
    const std::uint32_t *keys, std::uint32_t count,
    std::uint32_t *region_counts, std::uint32_t *reservation_ranks,
    std::uint32_t *active_regions = nullptr,
    std::uint32_t *active_count = nullptr) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned active = __ballot_sync(0xffffffffu, i < count);
  if (i >= count) return;
  const std::uint32_t region = keys[i] >> 16u;
  const unsigned peers = __match_any_sync(active, region);
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t leader = __ffs(peers) - 1u;
  std::uint32_t base = 0u;
  if (lane == leader) {
    base = atomicAdd(region_counts + region, __popc(peers));
    if constexpr (TrackActive) {
      if (!base) {
        const std::uint32_t ticket = atomicAdd(active_count, 1u);
        active_regions[ticket] = region;
      }
    }
  }
  base = __shfl_sync(peers, base, leader);
  const unsigned before = lane ? ((1u << lane) - 1u) : 0u;
  reservation_ranks[i] = base + __popc(peers & before);
}

__global__ void build_admission_signatures_kernel(
    const std::uint32_t *region_grouped_keys, std::uint32_t count,
    std::uint64_t *batch_signatures) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t key = region_grouped_keys[i];
  const std::uint32_t region = key >> 16u;
  const unsigned active = __activemask();
  const unsigned peers = __match_any_sync(active, region);
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
                 batch_signatures + region), aggregate);
  }
}

__global__ void commit_admission_metadata_kernel(
    std::uint32_t *counts, const std::uint64_t *batch_signatures,
    std::uint64_t *epoch_signatures) {
  const std::uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= kRegions) return;
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
  if (q >= kRegions) return;
  std::uint64_t aggregate = 0u;
  for (std::uint32_t batch = 0u; batch < kBatchesPerEpoch; ++batch)
    aggregate |= batch_signatures[
        std::size_t{batch} * kRegions + q];
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

  // Publish loads before dependent lookup.
  const std::uint32_t key = keys[i];
  const std::uint32_t rank = reservation_ranks[i];
  std::uint32_t value = 0u;
  if constexpr (!Tombstone) value = values[i];
  const std::uint32_t region = key >> 16u;
  const std::uint32_t output = offsets[region] + rank;
  destination_keys[output] = key;
  const std::uint32_t metadata =
      (batch_slot << kBatchPositionBits) | i |
      (Tombstone ? kRawTombstone : 0u);
  store_raw_payload(destination_payloads, output, value, metadata);
}

}  // namespace celllsm
