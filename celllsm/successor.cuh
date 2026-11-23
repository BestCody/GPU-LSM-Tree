#pragma once

// Successor: next non-empty section index, then the smallest visible key.

#include "pending.cuh"

namespace gpulsmopt2_detail {

constexpr std::uint32_t kSuccessorRegionChunk = kThreads;
constexpr std::uint32_t kSuccessorRegionChunks =
    kQuotients / kSuccessorRegionChunk;
constexpr std::uint32_t kSuccessorSharedPendingQueries =
    kQuotients * 4u;
static_assert(kQuotients % kSuccessorRegionChunk == 0u);

__device__ __forceinline__ bool first_resident_at_or_after(
    std::uint32_t q, std::uint32_t lower_suffix, std::uint32_t level,
    ResidentRows arena, const Descriptor *descriptors,
    const std::uint16_t *cell_ranks, Row &candidate) {
  const Descriptor rows = descriptors[descriptor_index(q, level)];
  if (!rows.count()) return false;
  const std::uint16_t *ranks = cell_ranks +
      std::size_t{level} * kLocalRankEntries +
      std::size_t{q} * kFoundationCells;
  const std::uint32_t first_cell = lower_suffix / kFoundationCellKeys;
  std::uint32_t begin = ranks[first_cell];
  std::uint32_t end = first_cell + 1u < kFoundationCells
      ? ranks[first_cell + 1u] : rows.count();
  const std::uint32_t local = lower_bound_rows(
      arena + rows.offset() + begin, end - begin, lower_suffix);
  if (local < end - begin) {
    candidate = arena[rows.offset() + begin + local];
    return true;
  }
  if (end < rows.count()) {
    candidate = arena[rows.offset() + end];
    return true;
  }
  return false;
}

__device__ __forceinline__ bool find_resident_successor_version(
    std::uint32_t q, std::uint32_t suffix, std::uint64_t occupied_levels,
    ResidentRows arena, const Descriptor *descriptors,
    const std::uint16_t *cell_ranks, Row &winner) {
  while (occupied_levels) {
    const std::uint32_t level =
        static_cast<std::uint32_t>(__ffsll(occupied_levels) - 1);
    occupied_levels &= occupied_levels - 1u;
    const Descriptor rows = descriptors[descriptor_index(q, level)];
    if (!rows.count()) continue;
    const std::uint32_t cell = suffix / kFoundationCellKeys;
    const std::uint16_t *ranks = cell_ranks +
        std::size_t{level} * kLocalRankEntries +
        std::size_t{q} * kFoundationCells;
    const std::uint32_t begin = ranks[cell];
    const std::uint32_t end = cell + 1u < kFoundationCells
        ? ranks[cell + 1u] : rows.count();
    if (find_unique_point_row(
            arena + rows.offset() + begin, end - begin, suffix, winner))
      return true;
  }
  return false;
}

__device__ __forceinline__ bool find_raw_successor_version(
    std::uint32_t q, std::uint32_t suffix,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, Row &winner) {
  for (int batch = static_cast<int>(pending_batches) - 1;
       batch >= 0; --batch) {
    const std::uint32_t slot = static_cast<std::uint32_t>(batch);
    const std::size_t offset =
        std::size_t{slot} * (kQuotients + 1u) + q;
    const std::uint32_t begin = raw_offsets[offset];
    const std::uint32_t end = raw_offsets[offset + 1u];
    bool found = false;
    std::uint32_t newest = 0u;
    for (std::uint32_t local = begin; local < end; ++local) {
      const std::uint32_t record = slot * batch_stride + local;
      if (key_suffix(raw_keys[record]) != suffix) continue;
      const RawPayload payload = raw_payloads[record];
      const std::uint32_t age = raw_position(payload);
      if (!found || age > newest) {
        winner = raw_row(raw_keys[record], payload);
        newest = age;
        found = true;
      }
    }
    if (found) return true;
  }
  return false;
}

__device__ __forceinline__ bool first_visible_in_quotient(
    std::uint32_t q, std::uint32_t lower,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const Row *prepared_pending_rows,
    const std::uint32_t *prepared_pending_offsets,
    const std::uint32_t *prepared_pending_counts,
    ResidentRows arena, const Descriptor *descriptors,
    const std::uint16_t *cell_ranks, std::uint64_t occupied_levels,
    std::uint32_t &result) {
  std::uint32_t lower_suffix = key_suffix(lower);
  for (;;) {
    std::uint32_t minimum = kInvalid;
    bool found = false;

    if (prepared_pending_rows) {
      const std::uint32_t begin = prepared_pending_offsets[q];
      const std::uint32_t count = prepared_pending_counts[q];
      const std::uint32_t local = lower_bound_rows(
          prepared_pending_rows + begin, count, lower_suffix);
      if (local < count) {
        minimum = prepared_pending_rows[begin + local].key;
        found = true;
      }
    } else {
      for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
        const std::size_t offset =
            std::size_t{batch} * (kQuotients + 1u) + q;
        const std::uint32_t begin = raw_offsets[offset];
        const std::uint32_t end = raw_offsets[offset + 1u];
        for (std::uint32_t local = begin; local < end; ++local) {
          const std::uint32_t suffix = key_suffix(
              raw_keys[batch * batch_stride + local]);
          if (suffix >= lower_suffix && (!found || suffix < minimum)) {
            minimum = suffix;
            found = true;
          }
        }
      }
    }

    std::uint64_t levels = occupied_levels;
    while (levels) {
      const std::uint32_t level =
          static_cast<std::uint32_t>(__ffsll(levels) - 1);
      levels &= levels - 1u;
      Row candidate{};
      if (first_resident_at_or_after(
              q, lower_suffix, level, arena, descriptors,
              cell_ranks, candidate) &&
          (!found || candidate.key < minimum)) {
        minimum = candidate.key;
        found = true;
      }
    }
    if (!found) return false;

    Row winner{};
    bool have_winner = false;
    if (prepared_pending_rows) {
      const std::uint32_t begin = prepared_pending_offsets[q];
      const std::uint32_t count = prepared_pending_counts[q];
      const std::uint32_t local = lower_bound_rows(
          prepared_pending_rows + begin, count, minimum);
      if (local < count && prepared_pending_rows[begin + local].key == minimum) {
        winner = prepared_pending_rows[begin + local];
        have_winner = true;
      }
    } else {
      have_winner = find_raw_successor_version(
          q, minimum, raw_keys, raw_payloads, raw_offsets,
          batch_stride, pending_batches, winner);
    }
    if (!have_winner)
      have_winner = find_resident_successor_version(
          q, minimum, occupied_levels, arena, descriptors,
          cell_ranks, winner);
    if (have_winner && (winner.flags & kTombstone) == 0u) {
      result = full_key(q, minimum);
      return true;
    }
    if (minimum == std::numeric_limits<std::uint16_t>::max()) return false;
    lower_suffix = minimum + 1u;
  }
}

__global__ void build_successor_region_index_kernel(
    const std::uint32_t *raw_offsets, std::uint32_t pending_batches,
    const Descriptor *descriptors, const std::uint64_t *occupied_mask,
    std::uint32_t *next_regions, std::uint32_t *chunk_first) {
  __shared__ std::uint32_t occupied[kSuccessorRegionChunk];
  const std::uint32_t q =
      blockIdx.x * kSuccessorRegionChunk + threadIdx.x;
  bool present = false;
  for (std::uint32_t batch = 0u; batch < pending_batches && !present; ++batch) {
    const std::size_t offset =
        std::size_t{batch} * (kQuotients + 1u) + q;
    present = raw_offsets[offset] != raw_offsets[offset + 1u];
  }
  std::uint64_t levels = __ldg(occupied_mask);
  while (levels && !present) {
    const std::uint32_t level =
        static_cast<std::uint32_t>(__ffsll(levels) - 1);
    levels &= levels - 1u;
    present = descriptors[descriptor_index(q, level)].count() != 0u;
  }
  occupied[threadIdx.x] = present ? q : kInvalid;
  __syncthreads();
  if (threadIdx.x == 0u) {
    std::uint32_t next = kInvalid;
    for (std::uint32_t local = kSuccessorRegionChunk; local-- > 0u;) {
      if (occupied[local] != kInvalid) next = occupied[local];
      next_regions[blockIdx.x * kSuccessorRegionChunk + local] = next;
    }
    chunk_first[blockIdx.x] = next;
  }
}

__global__ void build_successor_chunk_carry_kernel(
    const std::uint32_t *chunk_first, std::uint32_t *chunk_carry,
    std::uint32_t *next_regions) {
  if (blockIdx.x || threadIdx.x) return;
  std::uint32_t next = kInvalid;
  for (std::uint32_t chunk = kSuccessorRegionChunks; chunk-- > 0u;) {
    chunk_carry[chunk] = next;
    if (chunk_first[chunk] != kInvalid) next = chunk_first[chunk];
  }
  next_regions[kQuotients] = kInvalid;
}

__global__ void finish_successor_region_index_kernel(
    std::uint32_t *next_regions, const std::uint32_t *chunk_carry) {
  const std::uint32_t q =
      blockIdx.x * kSuccessorRegionChunk + threadIdx.x;
  if (next_regions[q] == kInvalid)
    next_regions[q] = chunk_carry[blockIdx.x];
}

__global__ void successor_with_pending_kernel(
    const std::uint32_t *queries, std::uint32_t count,
    std::uint32_t *out_keys, const std::uint32_t *raw_keys,
    const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const Row *prepared_pending_rows,
    const std::uint32_t *prepared_pending_offsets,
    const std::uint32_t *prepared_pending_counts,
    ResidentRows arena, const Descriptor *descriptors,
    const std::uint16_t *cell_ranks,
    const std::uint32_t *next_regions,
    const std::uint64_t *query_occupied_level_mask) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint64_t occupied_levels =
      __ldg(query_occupied_level_mask);
  const std::uint32_t query = queries[i];
  const std::uint32_t first_q = query >> 16u;
  std::uint32_t q = next_regions[first_q];
  while (q != kInvalid) {
    const std::uint32_t lower = q == first_q ? query : q << 16u;
    std::uint32_t result{};
    if (first_visible_in_quotient(
            q, lower, raw_keys, raw_payloads, raw_offsets, batch_stride,
            pending_batches, prepared_pending_rows, prepared_pending_offsets,
            prepared_pending_counts, arena, descriptors, cell_ranks,
            occupied_levels, result)) {
      out_keys[i] = result;
      return;
    }
    q = q + 1u < kQuotients ? next_regions[q + 1u] : kInvalid;
  }
  out_keys[i] = kInvalid;
}

}  // namespace gpulsmopt2_detail
