#pragma once

// Point lookup: one thread per query, the tiled path for huge batches, and
// the shared pending lookup for dense pending batches.

#include "pending.cuh"

namespace gpulsmopt2_detail {

constexpr std::uint32_t kTqrjHashThreads = 1024u;
static_assert(kTqrjHashThreads <= 1024u &&
              (kTqrjHashThreads & 31u) == 0u);

// Bound scratch with fixed tiles and 8-bit key bins.
constexpr std::uint32_t kLookupTileItems = 16u;
constexpr std::uint32_t kLookupTileRows = kThreads * kLookupTileItems;
constexpr std::uint32_t kLookupTileBins = 256u;
constexpr std::uint32_t kLookupTileLanes = 8u;
constexpr std::uint32_t kMaximumLookupWindowTiles = 4096u;

// Tile large batches over one run without pending rows.
constexpr std::size_t kMinimumTileLookupQueries = std::size_t{1} << 24u;
constexpr std::size_t kMinimumTileLookupResidentRows = std::size_t{1} << 24u;
static_assert(kLookupTileRows < std::numeric_limits<std::uint16_t>::max());

// TQRJ scans each owned pending section once.
constexpr std::uint32_t kTqrjDenseRowsPerSection = 8u;
constexpr std::uint32_t kTqrjMinimumBatches = 5u;
constexpr std::uint32_t kTqrjDirectCapacity = 1280u;
constexpr std::uint32_t kTqrjDirectoryBins = 256u;
constexpr std::uint32_t kTqrjDirectIntervalLimit = 64u;
constexpr std::uint32_t kTqrjDirectPendingRows = 1u << 16u;
constexpr std::uint32_t kTqrjHashTileRows = 2048u;
// Fingerprints reject; complete keys confirm matches.
constexpr std::uint32_t kTqrjHashOwnerBits = 25u;
constexpr std::uint32_t kTqrjHashOwnerMask =
    (1u << kTqrjHashOwnerBits) - 1u;
constexpr std::uint32_t kTqrjHashFingerprintBits =
    32u - kTqrjHashOwnerBits;
constexpr std::uint32_t kTqrjHashAttempts = 3u;
constexpr std::uint32_t kTqrjHashProbeLimit = 64u;
constexpr std::uint32_t kTqrjHashCapacityAlignment = 256u;
static_assert(kTqrjDirectCapacity == 1280u);
static_assert(kTqrjDirectCapacity <=
              std::numeric_limits<std::uint16_t>::max());
static_assert(kTqrjDirectoryBins == kThreads);
static_assert(kTqrjDirectPendingRows > kTqrjDirectCapacity);
static_assert(kTqrjHashTileRows >= kTqrjHashThreads &&
              (kTqrjHashTileRows & (kTqrjHashTileRows - 1u)) == 0u);
static_assert(kMaximumOperationTile * kBatchesPerEpoch <=
              kTqrjHashOwnerMask);

__global__ void materialize_lookup_active_counts_kernel(
    const std::uint32_t *active_quotients,
    const std::uint32_t *active_count,
    const std::uint32_t *quotient_counts,
    std::uint32_t active_capacity,
    std::uint32_t *active_counts) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > active_capacity) return;
  const std::uint32_t count = *active_count;
  active_counts[i] = i < count
      ? quotient_counts[active_quotients[i]] : 0u;
}

__global__ void publish_lookup_active_bases_kernel(
    const std::uint32_t *active_quotients,
    const std::uint32_t *active_count,
    const std::uint32_t *active_offsets,
    std::uint32_t active_capacity,
    std::uint32_t *query_bases) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count = *active_count;
  if (i >= active_capacity || i >= count) return;
  const std::uint32_t q = active_quotients[i];
  query_bases[q] = active_offsets[i];
}

__global__ void reset_lookup_quotient_counts_kernel(
    const std::uint32_t *active_quotients,
    const std::uint32_t *active_count,
    std::uint32_t active_capacity,
    std::uint32_t *quotient_counts) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count = *active_count;
  if (i >= active_capacity || i >= count) return;
  quotient_counts[active_quotients[i]] = 0u;
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

__device__ __forceinline__ void emit_lookup_result(
    std::uint32_t query_index, bool found, std::uint32_t value,
    std::uint32_t *out_values, std::uint8_t *out_found,
    const std::uint32_t *query_ids) {
  const std::uint32_t destination =
      query_ids ? query_ids[query_index] : query_index;
  out_values[destination] = found ? value : out_found ? 0u : kInvalid;
  if (out_found) out_found[destination] = found;
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
    emit_lookup_result(
        query_index, live, winner.value, out_values, out_found, query_ids);
    return;
  }
  emit_lookup_result(
      query_index, false, 0u, out_values, out_found, query_ids);
}

// Match the newest batch, then its newest matching row.
template <bool FilterSignatures, class Match>
__device__ __forceinline__ bool find_pending_lookup_winner(
    std::uint32_t head, const std::uint32_t *offsets,
    const RawPayload *payloads, std::uint32_t batch_stride,
    std::uint32_t batch_count, Match match, RawPayload &winner,
    const std::uint64_t *batch_signatures = nullptr,
    const std::uint64_t *epoch_signatures = nullptr) {
  const std::uint32_t section = head >> 16u;
  const std::uint64_t signature_bits = pending_signature_bits(head);
  if constexpr (FilterSignatures) {
    if (!batch_count ||
        (epoch_signatures[section] & signature_bits) != signature_bits)
      return false;
  }
  for (int batch = static_cast<int>(batch_count) - 1; batch >= 0; --batch) {
    const std::uint32_t slot = static_cast<std::uint32_t>(batch);
    if constexpr (FilterSignatures) {
      const std::uint64_t signature =
          batch_signatures[std::size_t{slot} * kQuotients + section];
      if ((signature & signature_bits) != signature_bits) continue;
    }
    const std::size_t offset = std::size_t{slot} * (kQuotients + 1u) + section;
    const std::uint32_t begin = offsets[offset], end = offsets[offset + 1u];
    bool found = false;
    std::uint32_t newest = 0u;
    for (std::uint32_t local = begin; local < end; ++local) {
      const std::uint32_t record = slot * batch_stride + local;
      if (!match(slot, local, record)) continue;
      const RawPayload payload = payloads[record];
      const std::uint32_t age = payload.metadata & Match::kAgeMask;
      if (!found || age > newest) {
        winner = payload;
        newest = age;
        found = true;
      }
    }
    if (found) return true;
  }
  return false;
}

struct NativePendingKeyMatch {
  static constexpr std::uint32_t kAgeMask = ~kRawTombstone;
  const std::uint32_t *keys;
  std::uint32_t suffix;
  __device__ __forceinline__ bool operator()(
      std::uint32_t, std::uint32_t, std::uint32_t record) const {
    return key_suffix(keys[record]) == suffix;
  }
};

__device__ __forceinline__ void canonical_lookup_point_with_pending(
    std::uint32_t key, std::uint32_t i, std::uint32_t *out_values,
    std::uint8_t *out_found,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const std::uint64_t *batch_signatures,
    const std::uint64_t *epoch_signatures, ResidentRows arena,
    const Descriptor *descriptors, const std::uint16_t *cell_ranks,
    const std::uint32_t *query_ids,
    const std::uint64_t *occupied_mask) {
  RawPayload winner{};
  if (find_pending_lookup_winner<true>(
          key, raw_offsets, raw_payloads, batch_stride, pending_batches,
          NativePendingKeyMatch{raw_keys, key_suffix(key)}, winner,
          batch_signatures, epoch_signatures)) {
    const bool live = (winner.metadata & kRawTombstone) == 0u;
    emit_lookup_result(i, live, winner.value, out_values, out_found, query_ids);
    return;
  }
  canonical_lookup_resident_only(
      key, i, out_values, out_found, arena, descriptors, cell_ranks,
      __ldg(occupied_mask), query_ids);
}

__global__ void canonical_lookup_with_pending_kernel(
    const std::uint32_t *queries, std::uint32_t *out_values,
    std::uint8_t *out_found, std::uint32_t count,
    const std::uint32_t *raw_keys, const RawPayload *raw_payloads,
    const std::uint32_t *raw_offsets, std::uint32_t batch_stride,
    std::uint32_t pending_batches, const std::uint64_t *batch_signatures,
    const std::uint64_t *epoch_signatures, ResidentRows arena,
    const Descriptor *descriptors, const std::uint16_t *cell_ranks,
    const std::uint32_t *query_ids, const std::uint64_t *occupied_mask) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  canonical_lookup_point_with_pending(
      queries[i], i, out_values, out_found, raw_keys, raw_payloads,
      raw_offsets, batch_stride, pending_batches, batch_signatures,
      epoch_signatures, arena, descriptors, cell_ranks, query_ids,
      occupied_mask);
}

struct TileLookupView {
  const std::uint32_t *raw_keys;
  const RawPayload *raw_payloads;
  const std::uint32_t *raw_offsets;
  std::uint32_t batch_stride;
  std::uint32_t pending_batches;
  const std::uint64_t *batch_signatures;
  const std::uint64_t *epoch_signatures;
  ResidentRows arena;
  const Descriptor *descriptors;
  const std::uint16_t *cell_ranks;
  const std::uint64_t *occupied_mask;
};

__global__ void pack_lookup_tiles_kernel(
    const std::uint32_t *input, std::uint32_t *local,
    std::uint16_t *inverse, std::uint16_t *offsets,
    std::uint32_t count, std::uint32_t tiles) {
  using Sort = cub::BlockRadixSort<
      std::uint32_t, kThreads, kLookupTileItems, std::uint16_t>;
  using Scan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ union {
    typename Sort::TempStorage sort;
    typename Scan::TempStorage scan;
  } temp;
  __shared__ std::uint32_t counts[kLookupTileBins];
  __shared__ std::uint16_t positions[kLookupTileRows];
  std::uint32_t keys[kLookupTileItems];
  std::uint16_t original[kLookupTileItems];
  const std::uint32_t base = blockIdx.x * kLookupTileRows;
  counts[threadIdx.x] = 0u;
  __syncthreads();
  for (std::uint32_t j = 0u; j < kLookupTileItems; ++j) {
    const std::uint32_t p = threadIdx.x * kLookupTileItems + j;
    keys[j] = base + p < count ? input[base + p] : kInvalid;
    original[j] = static_cast<std::uint16_t>(p);
    if (base + p < count) atomicAdd(counts + (keys[j] >> 24u), 1u);
  }
  __syncthreads();
  // Stable high-byte sorting keeps padding last.
  Sort(temp.sort).SortBlockedToStriped(keys, original, 24, 32);
  __syncthreads();
  for (std::uint32_t j = 0u; j < kLookupTileItems; ++j) {
    const std::uint32_t p = j * kThreads + threadIdx.x;
    if (base + original[j] < count) {
      local[base + p] = keys[j];
      positions[original[j]] = static_cast<std::uint16_t>(p);
    }
  }
  __syncthreads();
  for (std::uint32_t j = 0u; j < kLookupTileItems; ++j) {
    const std::uint32_t p = j * kThreads + threadIdx.x;
    if (base + p < count) inverse[base + p] = positions[p];
  }
  std::uint32_t begin = 0u;
  Scan(temp.scan).ExclusiveSum(counts[threadIdx.x], begin);
  offsets[std::size_t{threadIdx.x} * tiles + blockIdx.x] =
      static_cast<std::uint16_t>(begin);
  if (threadIdx.x == 0u)
    offsets[std::size_t{kLookupTileBins} * tiles + blockIdx.x] =
        static_cast<std::uint16_t>(min(kLookupTileRows, count - base));
}

template <bool WithFound>
__global__ void execute_lookup_tiles_kernel(
    std::uint32_t *local, std::uint8_t *found,
    const std::uint16_t *offsets, std::uint32_t tiles, TileLookupView view) {
  constexpr std::uint32_t tiles_per_block = kThreads / kLookupTileLanes;
  const std::uint32_t tile =
      blockIdx.x * tiles_per_block + threadIdx.x / kLookupTileLanes;
  if (tile >= tiles) return;
  const std::uint32_t bin = blockIdx.y;
  const std::uint32_t begin = offsets[std::size_t{bin} * tiles + tile];
  const std::uint32_t end = offsets[std::size_t{bin + 1u} * tiles + tile];
  for (std::uint32_t p = begin + threadIdx.x % kLookupTileLanes;
       p < end; p += kLookupTileLanes) {
    const std::uint32_t position = tile * kLookupTileRows + p;
    const std::uint32_t key = local[position];
    // Each query owns its scratch slot exclusively.
    canonical_lookup_point_with_pending(
        key, position, local, WithFound ? found : nullptr,
        view.raw_keys, view.raw_payloads, view.raw_offsets,
        view.batch_stride, view.pending_batches, view.batch_signatures,
        view.epoch_signatures, view.arena, view.descriptors, view.cell_ranks,
        nullptr, view.occupied_mask);
  }
}

template <bool WithFound>
__global__ void unpack_lookup_tiles_kernel(
    const std::uint32_t *local, const std::uint8_t *found,
    const std::uint16_t *inverse, std::uint32_t *output,
    std::uint8_t *out_found, std::uint32_t count) {
  __shared__ std::uint32_t values[kLookupTileRows];
  __shared__ std::uint8_t flags[WithFound ? kLookupTileRows : 1u];
  const std::uint32_t base = blockIdx.x * kLookupTileRows;
  for (std::uint32_t j = 0u; j < kLookupTileItems; ++j) {
    const std::uint32_t p = j * kThreads + threadIdx.x;
    if (base + p < count) {
      values[p] = local[base + p];
      if constexpr (WithFound) flags[p] = found[base + p];
    }
  }
  __syncthreads();
  for (std::uint32_t j = 0u; j < kLookupTileItems; ++j) {
    const std::uint32_t p = j * kThreads + threadIdx.x;
    if (base + p < count) {
      const std::uint32_t position = inverse[base + p];
      output[base + p] = values[position];
      if constexpr (WithFound) out_found[base + p] = flags[position];
    }
  }
}

__device__ __forceinline__ std::uint32_t tqrj_directory_find(
    const std::uint16_t *directory_offsets,
    const std::uint16_t *directory_suffixes,
    std::uint32_t target) {
  const std::uint32_t interval = target >> 8u;
  for (std::uint32_t p = directory_offsets[interval];
       p < directory_offsets[interval + 1u]; ++p)
    if (directory_suffixes[p] == target) return p;
  return kInvalid;
}

__device__ __forceinline__ unsigned long long tqrj_pending_token(
    const RawPayload &payload) {
  const std::uint64_t order =
      static_cast<std::uint64_t>(raw_position(payload)) + 1u;
  const std::uint64_t live =
      (payload.metadata & kRawTombstone) == 0u ? 1u : 0u;
  return static_cast<unsigned long long>(
      (order << 33u) | (live << 32u) | payload.value);
}

// Aggregate matching lanes by exact owner.
__device__ __forceinline__ void tqrj_warp_atomic_max(
    bool matched, std::uint32_t owner, unsigned long long token,
    unsigned long long *winners) {
  const unsigned matched_mask = __ballot_sync(0xffffffffu, matched);
  if (!matched) return;
  const unsigned peers = __match_any_sync(matched_mask, owner);
  unsigned long long aggregate = token;
  for (unsigned remaining = peers; remaining;
       remaining &= remaining - 1u) {
    const int source_lane = __ffs(remaining) - 1;
    const unsigned long long candidate =
        __shfl_sync(peers, token, source_lane);
    aggregate = candidate > aggregate ? candidate : aggregate;
  }
  const std::uint32_t lane = threadIdx.x & 31u;
  if (lane == static_cast<std::uint32_t>(__ffs(peers) - 1))
    atomicMax(winners + owner, aggregate);
}

struct TqrjHashTask {
  std::uint32_t quotient;
  std::uint32_t pending_rows;
  std::uint32_t query_tile_base;
  std::uint32_t pending_tile_base;
};

struct TqrjHashTile {
  std::uint32_t task;
  std::uint32_t begin;
};

// TQRJ aliases reuse operation-locked storage.
struct TqrjLookupWorkspace {
  std::uint32_t *grouped_queries;
  std::uint32_t *query_ids;
  std::uint32_t *reservation_ranks;
  std::uint32_t *active_quotients;
  std::uint32_t *active_query_counts;
  std::uint32_t *active_query_offsets;
  std::uint32_t *active_quotient_count;
  std::uint32_t *query_bases;

  TqrjHashTask *hash_tasks;
  std::uint32_t *hash_counters;
  std::uint32_t *hash_task_count;
  unsigned long long *hash_winners;
  TqrjHashTile *hash_query_tiles;
  TqrjHashTile *hash_pending_tiles;
  std::uint32_t *hash_table;
  std::uint32_t hash_table_capacity;
};

constexpr std::size_t kTqrjHashTaskBytes =
    std::size_t{kQuotients} * sizeof(TqrjHashTask);
constexpr std::size_t kTqrjHashCounterBytes =
    6u * sizeof(std::uint32_t);
static_assert(kTqrjHashTaskBytes + kTqrjHashCounterBytes <=
              std::size_t{kLocalRankEntries} * sizeof(std::uint32_t));

__device__ __forceinline__ void tqrj_enqueue_hash_task(
    std::uint32_t quotient, std::uint32_t pending_rows,
    TqrjHashTask *tasks, std::uint32_t *task_count) {
  const std::uint32_t ticket = atomicAdd(task_count, 1u);
  tasks[ticket] = {quotient, pending_rows, 0u, 0u};
}

__device__ __forceinline__ std::uint32_t tqrj_pending_rows(
    std::uint32_t q, const std::uint32_t *raw_offsets,
    std::uint32_t pending_batches) {
  std::uint32_t total = 0u;
  for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
    const std::size_t oi =
        std::size_t{batch} * (kQuotients + 1u) + q;
    total += raw_offsets[oi + 1u] - raw_offsets[oi];
  }
  return total;
}

__device__ __forceinline__ void tqrj_write_result(
    std::uint32_t key, std::uint32_t query_index,
    unsigned long long winner, std::uint32_t *out_values,
    std::uint8_t *out_found, ResidentRows arena,
    const Descriptor *descriptors,
    const std::uint16_t *canonical_cell_ranks,
    std::uint64_t occupied_levels, const std::uint32_t *query_ids) {
  if (winner) {
    const bool live = ((winner >> 32u) & 1u) != 0u;
    emit_lookup_result(
        query_index, live, static_cast<std::uint32_t>(winner), out_values,
        out_found, query_ids);
    return;
  }
  canonical_lookup_resident_only(
      key, query_index, out_values, out_found, arena, descriptors,
      canonical_cell_ranks, occupied_levels, query_ids);
}

__host__ __device__ constexpr std::uint32_t tqrj_hash_tile_count(
    std::uint32_t rows) {
  return (rows + kTqrjHashTileRows - 1u) / kTqrjHashTileRows;
}

// Size overflow storage from actual queries.
__host__ __device__ constexpr std::uint64_t tqrj_hash_seed(
    std::uint32_t attempt) {
  return attempt == 0u ? 0x6a09e667f3bcc909ull
      : attempt == 1u ? 0x243f6a8885a308d3ull
      : 0x13198a2e03707344ull;
}

__host__ __device__ __forceinline__ std::uint64_t tqrj_hash_mix(
    std::uint64_t value) {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

__host__ __device__ __forceinline__ std::uint64_t tqrj_hash_key(
    std::uint32_t key, std::uint64_t seed) {
  return tqrj_hash_mix(static_cast<std::uint64_t>(key) ^ seed);
}

__host__ __device__ constexpr std::uint32_t tqrj_hash_capacity(
    std::uint32_t rows) {
  if (!rows) return 0u;
  const std::uint64_t proportional =
      (std::uint64_t{13u} * rows + 7u) / 8u;
  return static_cast<std::uint32_t>(
      (proportional + kTqrjHashCapacityAlignment - 1u) &
      ~std::uint64_t{kTqrjHashCapacityAlignment - 1u});
}

__host__ __device__ __forceinline__ std::uint32_t tqrj_hash_slot(
    std::uint64_t hash, std::uint32_t capacity) {
  return static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(hash)) *
       capacity) >> 32u);
}

__host__ __device__ __forceinline__ std::uint32_t tqrj_hash_entry(
    std::uint32_t owner, std::uint64_t hash) {
  const std::uint32_t fingerprint = static_cast<std::uint32_t>(
      hash >> (64u - kTqrjHashFingerprintBits));
  return (fingerprint << kTqrjHashOwnerBits) | (owner + 1u);
}

__host__ __device__ __forceinline__ std::uint32_t
tqrj_hash_entry_owner(std::uint32_t entry) {
  return (entry & kTqrjHashOwnerMask) - 1u;
}

__host__ __device__ __forceinline__ std::uint32_t
tqrj_hash_entry_fingerprint(std::uint32_t entry) {
  return entry >> kTqrjHashOwnerBits;
}

// Confirm fingerprint matches with complete keys.
template <class EqualOwner>
__device__ __forceinline__ std::uint32_t tqrj_hash_insert_owner(
    std::uint32_t query, std::uint64_t hash, std::uint32_t *table,
    std::uint32_t capacity, std::uint32_t maximum_probes,
    std::uint32_t *failure, EqualOwner equal_owner) {
  const std::uint32_t desired = tqrj_hash_entry(query, hash);
  const std::uint32_t fingerprint =
      tqrj_hash_entry_fingerprint(desired);
  std::uint32_t slot = tqrj_hash_slot(hash, capacity);
  for (std::uint32_t probe = 0u; probe < maximum_probes; ++probe) {
    const std::uint32_t previous = atomicCAS(table + slot, 0u, desired);
    if (!previous) return query;
    if (tqrj_hash_entry_fingerprint(previous) == fingerprint) {
      const std::uint32_t owner = tqrj_hash_entry_owner(previous);
      if (equal_owner(owner)) return owner;
    }
    if (++slot == capacity) slot = 0u;
  }
  atomicExch(failure, 1u);
  return kInvalid;
}

template <class EqualOwner>
__device__ __forceinline__ std::uint32_t tqrj_hash_find_owner(
    std::uint64_t hash, const std::uint32_t *table,
    std::uint32_t capacity, EqualOwner equal_owner) {
  const std::uint32_t fingerprint = tqrj_hash_entry_fingerprint(
      tqrj_hash_entry(0u, hash));
  std::uint32_t slot = tqrj_hash_slot(hash, capacity);
  for (std::uint32_t probe = 0u; probe < capacity; ++probe) {
    const std::uint32_t entry = table[slot];
    if (!entry) return kInvalid;
    if (tqrj_hash_entry_fingerprint(entry) == fingerprint) {
      const std::uint32_t owner = tqrj_hash_entry_owner(entry);
      if (equal_owner(owner)) return owner;
    }
    if (++slot == capacity) slot = 0u;
  }
  return kInvalid;
}

__device__ __forceinline__ std::uint32_t tqrj_hash_insert(
    const std::uint32_t *grouped_queries, std::uint32_t query,
    std::uint32_t *table, std::uint32_t capacity,
    std::uint64_t seed, std::uint32_t maximum_probes,
    std::uint32_t *failure) {
  const std::uint32_t key = grouped_queries[query];
  return tqrj_hash_insert_owner(query, tqrj_hash_key(key, seed), table,
      capacity, maximum_probes, failure, [=] __device__ (std::uint32_t owner) {
        return grouped_queries[owner] == key;
      });
}

__device__ __forceinline__ std::uint32_t tqrj_hash_find(
    std::uint32_t key, const std::uint32_t *grouped_queries,
    const std::uint32_t *table, std::uint32_t capacity,
    std::uint64_t seed) {
  return tqrj_hash_find_owner(tqrj_hash_key(key, seed), table, capacity,
      [=] __device__ (std::uint32_t owner) {
        return grouped_queries[owner] == key;
      });
}

// Share block matching; adapters define record semantics.
template <class Access>
__device__ __forceinline__ void tqrj_direct_lookup(
    Access access, const std::uint32_t *active_quotients,
    const std::uint32_t *active_counts, const std::uint32_t *active_count,
    const std::uint32_t *query_bases, const std::uint32_t *raw_offsets,
    std::uint32_t pending_batches, TqrjHashTask *hash_tasks,
    std::uint32_t *hash_task_count) {
  using BlockScan = cub::BlockScan<std::uint32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::uint32_t directory_counts[kTqrjDirectoryBins];
  __shared__ std::uint16_t directory_offsets[kTqrjDirectoryBins + 1u];
  __shared__ __align__(4)
      std::uint16_t directory_suffixes[kTqrjDirectCapacity];
  __shared__ unsigned long long winners[kTqrjDirectCapacity];
  __shared__ typename Access::ReadState read_state;
  __shared__ typename Access::DirectoryOwners directory_owners;
  __shared__ std::uint32_t pending_rows;

  const std::uint32_t task = blockIdx.x;
  if (task >= *active_count) return;
  const std::uint32_t q = active_quotients[task];
  const std::uint32_t query_begin = query_bases[q];
  const std::uint32_t query_count = active_counts[task];
  if (threadIdx.x == 0u) {
    pending_rows = tqrj_pending_rows(q, raw_offsets, pending_batches);
    access.load_read_state(read_state);
  }
  __syncthreads();

  if (query_count > kTqrjDirectCapacity ||
      pending_rows > kTqrjDirectPendingRows) {
    if (threadIdx.x == 0u)
      tqrj_enqueue_hash_task(
          q, pending_rows, hash_tasks, hash_task_count);
    return;
  }
  directory_counts[threadIdx.x] = 0u;
  __syncthreads();

  // Direct tasks index compact suffix lists.
  for (std::uint32_t local = threadIdx.x; local < query_count;
       local += blockDim.x) {
    const std::uint32_t suffix =
        key_suffix(access.heads[query_begin + local]);
    atomicAdd(directory_counts + (suffix >> 8u), 1u);
  }
  __syncthreads();
  const std::uint32_t interval_count = directory_counts[threadIdx.x];
  std::uint32_t interval_begin = 0u;
  BlockScan(scan_storage).ExclusiveSum(interval_count, interval_begin);
  directory_offsets[threadIdx.x] =
      static_cast<std::uint16_t>(interval_begin);
  if (threadIdx.x + 1u == kTqrjDirectoryBins)
    directory_offsets[kTqrjDirectoryBins] =
        static_cast<std::uint16_t>(interval_begin + interval_count);
  directory_counts[threadIdx.x] = 0u;
  const bool crowded = __syncthreads_or(
      interval_count > kTqrjDirectIntervalLimit);
  if (crowded) {
    if (threadIdx.x == 0u)
      tqrj_enqueue_hash_task(
          q, pending_rows, hash_tasks, hash_task_count);
    return;
  }

  for (std::uint32_t local = threadIdx.x; local < query_count;
       local += blockDim.x) {
    const std::uint32_t suffix =
        key_suffix(access.heads[query_begin + local]);
    const std::uint32_t interval = suffix >> 8u;
    const std::uint32_t rank = atomicAdd(
        directory_counts + interval, 1u);
    const std::uint32_t position = directory_offsets[interval] + rank;
    directory_suffixes[position] = static_cast<std::uint16_t>(suffix);
    directory_owners.set(position, local);
    winners[local] = 0ull;
  }
  __syncthreads();

  // Assign contiguous pending batches to warps.
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  constexpr std::uint32_t kWarps = kThreads / 32u;
  for (std::uint32_t batch = warp; batch < pending_batches;
       batch += kWarps) {
    const std::size_t oi =
        std::size_t{batch} * (kQuotients + 1u) + q;
    const std::uint32_t begin = raw_offsets[oi];
    const std::uint32_t end = raw_offsets[oi + 1u];
    for (std::uint32_t base = begin; base < end; base += 32u) {
      const std::uint32_t position = base + lane;
      bool matched = false;
      std::uint32_t owner = kInvalid;
      unsigned long long token = 0ull;
      if (position < end) {
        const auto candidate = access.pending_record(batch, position);
        owner = access.direct_owner(candidate, directory_offsets,
            directory_suffixes, directory_owners, query_begin);
        matched = owner != kInvalid;
        if (matched) token = access.pending_token(candidate);
      }
      tqrj_warp_atomic_max(matched, owner, token, winners);
    }
  }
  __syncthreads();

  for (std::uint32_t local = threadIdx.x; local < query_count;
       local += blockDim.x) {
    const std::uint32_t query_index = query_begin + local;
    const auto query = access.query(query_index);
    const std::uint32_t owner = access.direct_owner(query, directory_offsets,
        directory_suffixes, directory_owners, query_begin);
    const unsigned long long winner = owner == kInvalid
        ? 0ull : winners[owner];
    access.write_result(query, query_index, winner, read_state);
  }
}

// Share global matching with format-specific owner storage.
template <class Access>
__device__ __forceinline__ void tqrj_hash_lookup(
    Access access, TqrjHashTask *tasks, const std::uint32_t *task_count,
    const std::uint32_t *query_bases, const std::uint32_t *query_counts,
    TqrjHashTile *query_tiles, TqrjHashTile *pending_tiles,
    std::uint32_t *counters, std::uint32_t *hash_table,
    std::uint32_t maximum_hash_entries, unsigned long long *winners,
    const std::uint32_t *raw_offsets, std::uint32_t pending_batches,
    std::uint32_t *query_owners) {
  const cooperative_groups::grid_group grid =
      cooperative_groups::this_grid();
  __shared__ std::uint32_t section_prefixes[kBatchesPerEpoch + 1u];
  __shared__ TqrjHashTask shared_task;
  __shared__ std::uint32_t shared_a, shared_b;
  __shared__ typename Access::ReadState read_state;

  const std::uint32_t tasks_in_queue = *task_count;
  if (!tasks_in_queue) return;
  const std::uint32_t global_thread =
      blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t global_stride = blockDim.x * gridDim.x;

  // Materialize direct-match overflow only.
  for (std::uint32_t task_index = blockIdx.x;
       task_index < tasks_in_queue; task_index += gridDim.x) {
    if (threadIdx.x == 0u) {
      const TqrjHashTask task = tasks[task_index];
      const std::uint32_t query_count = query_counts[task.quotient];
      shared_a = tqrj_hash_tile_count(query_count);
      shared_b = tqrj_hash_tile_count(task.pending_rows);
      tasks[task_index].query_tile_base = atomicAdd(counters, shared_a);
      tasks[task_index].pending_tile_base = atomicAdd(counters + 1u, shared_b);
      atomicAdd(counters + 2u, query_count);
    }
    __syncthreads();
    const TqrjHashTask task = tasks[task_index];
    for (std::uint32_t tile = threadIdx.x; tile < shared_a;
         tile += blockDim.x)
      query_tiles[task.query_tile_base + tile] =
          {task_index, tile * kTqrjHashTileRows};
    for (std::uint32_t tile = threadIdx.x; tile < shared_b;
         tile += blockDim.x)
      pending_tiles[task.pending_tile_base + tile] =
          {task_index, tile * kTqrjHashTileRows};
    __syncthreads();
  }
  grid.sync();

  const std::uint32_t query_tile_count = counters[0];
  const std::uint32_t pending_tile_count = counters[1];
  const std::uint32_t hash_capacity = tqrj_hash_capacity(counters[2]);
  if (global_thread == 0u) {
    counters[4] = 0u;
    counters[5] = hash_capacity > maximum_hash_entries ? 1u : 0u;
  }
  grid.sync();
  if (counters[5]) return;

  // The last retry may probe the complete table.
  for (std::uint32_t attempt = 0u; attempt < kTqrjHashAttempts; ++attempt) {
    if (global_thread == 0u) counters[3] = 0u;
    for (std::uint32_t slot = global_thread; slot < hash_capacity;
         slot += global_stride)
      hash_table[slot] = 0u;
    grid.sync();
    const std::uint64_t seed = tqrj_hash_seed(attempt);
    const std::uint32_t probe_limit =
        attempt + 1u == kTqrjHashAttempts
            ? hash_capacity : kTqrjHashProbeLimit;
    for (std::uint32_t tile_index = blockIdx.x;
         tile_index < query_tile_count; tile_index += gridDim.x) {
      const TqrjHashTile tile = query_tiles[tile_index];
      const TqrjHashTask task = tasks[tile.task];
      const std::uint32_t query_begin = query_bases[task.quotient];
      const std::uint32_t query_count = query_counts[task.quotient];
      for (std::uint32_t local = threadIdx.x; local < kTqrjHashTileRows;
           local += blockDim.x) {
        const std::uint32_t row = tile.begin + local;
        const std::uint32_t query_index = query_begin + row;
        // All lanes must join native duplicate aggregation.
        const std::uint32_t owner = access.insert_hash(
            row < query_count, query_index, hash_table, hash_capacity, seed,
            probe_limit, counters + 3u);
        if (row < query_count)
          query_owners[access.query_ids[query_index]] = owner;
      }
    }
    grid.sync();
    if (!counters[3]) {
      if (global_thread == 0u) counters[4] = attempt;
      grid.sync();
      break;
    }
    grid.sync();
  }

  // Initialize only query slots owned by hash tasks.
  for (std::uint32_t tile_index = blockIdx.x;
       tile_index < query_tile_count; tile_index += gridDim.x) {
    const TqrjHashTile tile = query_tiles[tile_index];
    const TqrjHashTask task = tasks[tile.task];
    const std::uint32_t query_begin = query_bases[task.quotient];
    const std::uint32_t query_count = query_counts[task.quotient];
    for (std::uint32_t local = threadIdx.x; local < kTqrjHashTileRows;
         local += blockDim.x) {
      const std::uint32_t row = tile.begin + local;
      if (row >= query_count) continue;
      const std::uint32_t query_index = query_begin + row;
      winners[query_index] = 0ull;
    }
  }
  grid.sync();

  const std::uint64_t selected_seed = tqrj_hash_seed(counters[4]);
  for (std::uint32_t tile_index = blockIdx.x;
       tile_index < pending_tile_count; tile_index += gridDim.x) {
    const TqrjHashTile tile = pending_tiles[tile_index];
    if (threadIdx.x == 0u) {
      shared_task = tasks[tile.task];
      std::uint32_t prefix = 0u;
      section_prefixes[0] = 0u;
      for (std::uint32_t batch = 0u; batch < pending_batches; ++batch) {
        const std::size_t oi = std::size_t{batch} *
            (kQuotients + 1u) + shared_task.quotient;
        prefix += raw_offsets[oi + 1u] - raw_offsets[oi];
        section_prefixes[batch + 1u] = prefix;
      }
    }
    __syncthreads();
    for (std::uint32_t local = threadIdx.x; local < kTqrjHashTileRows;
         local += blockDim.x) {
      const std::uint32_t row = tile.begin + local;
      bool matched = false;
      std::uint32_t owner = kInvalid;
      unsigned long long token = 0ull;
      if (row < shared_task.pending_rows) {
        std::uint32_t low = 0u, high = pending_batches;
        while (low + 1u < high) {
          const std::uint32_t middle = (low + high) >> 1u;
          if (section_prefixes[middle] <= row)
            low = middle;
          else
            high = middle;
        }
        const std::uint32_t batch = low;
        const std::size_t oi = std::size_t{batch} *
            (kQuotients + 1u) + shared_task.quotient;
        const std::uint32_t position =
            raw_offsets[oi] + row - section_prefixes[batch];
        const auto candidate = access.pending_record(batch, position);
        owner = access.find_hash(candidate, hash_table, hash_capacity,
                                 selected_seed);
        matched = owner != kInvalid;
        if (matched) token = access.pending_token(candidate);
      }
      tqrj_warp_atomic_max(matched, owner, token, winners);
    }
    __syncthreads();
  }
  grid.sync();

  if (threadIdx.x == 0u)
    access.load_read_state(read_state);
  __syncthreads();
  for (std::uint32_t tile_index = blockIdx.x;
       tile_index < query_tile_count; tile_index += gridDim.x) {
    const TqrjHashTile tile = query_tiles[tile_index];
    const TqrjHashTask task = tasks[tile.task];
    const std::uint32_t query_begin = query_bases[task.quotient];
    const std::uint32_t query_count = query_counts[task.quotient];
    for (std::uint32_t local = threadIdx.x; local < kTqrjHashTileRows;
         local += blockDim.x) {
      const std::uint32_t row = tile.begin + local;
      if (row >= query_count) continue;
      const std::uint32_t query_index = query_begin + row;
      const std::uint32_t owner = query_owners[access.query_ids[query_index]];
      const unsigned long long winner = winners[owner];
      access.write_result(access.query(query_index), query_index, winner,
                          read_state);
    }
    __syncthreads();
  }
}

struct NativeTqrjAccess {
  const std::uint32_t *heads;
  std::uint32_t *out_values;
  std::uint8_t *out_found;
  const std::uint32_t *raw_keys;
  const RawPayload *raw_payloads;
  std::uint32_t batch_stride;
  ResidentRows arena;
  const Descriptor *descriptors;
  const std::uint32_t *query_ids;
  const std::uint16_t *cell_ranks;
  const std::uint64_t *occupied_mask;

  struct ReadState { std::uint64_t occupied_levels; };
  struct DirectoryOwners {
    // Use suffix positions as native owner IDs.
    __device__ __forceinline__ void set(std::uint32_t, std::uint32_t) {}
  };
  struct Query {
    std::uint32_t head;
    __device__ __forceinline__ std::uint32_t head4() const { return head; }
  };
  struct Pending {
    const std::uint32_t *keys;
    std::uint32_t record;
    __device__ __forceinline__ std::uint32_t head4() const { return keys[record]; }
  };
  __device__ __forceinline__ void load_read_state(ReadState &state) const {
    state.occupied_levels = load_query_manifest(occupied_mask).occupied_level_mask;
  }
  __device__ __forceinline__ Query query(std::uint32_t grouped) const { return {heads[grouped]}; }
  __device__ __forceinline__ Pending pending_record(std::uint32_t batch, std::uint32_t row) const {
    return {raw_keys, batch * batch_stride + row};
  }
  __device__ __forceinline__ unsigned long long pending_token(Pending candidate) const {
    return tqrj_pending_token(raw_payloads[candidate.record]);
  }
  template <class Candidate>
  __device__ __forceinline__ std::uint32_t direct_owner(const Candidate &candidate,
      const std::uint16_t *offsets, const std::uint16_t *suffixes,
      const DirectoryOwners &, std::uint32_t) const {
    return tqrj_directory_find(offsets, suffixes, key_suffix(candidate.head4()));
  }
  __device__ __forceinline__ std::uint32_t insert_hash(bool valid, std::uint32_t grouped,
      std::uint32_t *table, std::uint32_t capacity, std::uint64_t seed,
      std::uint32_t probes, std::uint32_t *failure) const {
    const std::uint32_t key = valid ? heads[grouped] : 0u;
    const unsigned active = __ballot_sync(0xffffffffu, valid);
    if (!valid) return kInvalid;
    const unsigned peers = __match_any_sync(active, key);
    const std::uint32_t leader = static_cast<std::uint32_t>(__ffs(peers) - 1);
    std::uint32_t owner = kInvalid;
    if ((threadIdx.x & 31u) == leader)
      owner = tqrj_hash_insert(heads, grouped, table, capacity, seed, probes, failure);
    return __shfl_sync(peers, owner, leader);
  }
  __device__ __forceinline__ std::uint32_t find_hash(Pending candidate,
      const std::uint32_t *table, std::uint32_t capacity, std::uint64_t seed) const {
    return tqrj_hash_find(candidate.head4(), heads, table, capacity, seed);
  }
  __device__ __forceinline__ void write_result(Query query, std::uint32_t grouped,
      unsigned long long winner, const ReadState &state) const {
    tqrj_write_result(query.head4(), grouped, winner, out_values, out_found,
        arena, descriptors, cell_ranks, state.occupied_levels, query_ids);
  }
};

struct TqrjDirectLaunch {
  const std::uint32_t *active_quotients, *active_counts, *active_count;
  const std::uint32_t *query_bases, *raw_offsets;
  std::uint32_t pending_batches;
  TqrjHashTask *hash_tasks;
  std::uint32_t *hash_task_count;
};

template <class Access>
__global__ void tqrj_direct_lookup_kernel(Access access, TqrjDirectLaunch work) {
  tqrj_direct_lookup(access, work.active_quotients, work.active_counts,
      work.active_count, work.query_bases, work.raw_offsets,
      work.pending_batches, work.hash_tasks, work.hash_task_count);
}

struct TqrjHashLaunch {
  TqrjHashTask *tasks;
  const std::uint32_t *task_count, *query_bases, *query_counts;
  TqrjHashTile *query_tiles, *pending_tiles;
  std::uint32_t *counters, *hash_table;
  std::uint32_t maximum_hash_entries;
  unsigned long long *winners;
  const std::uint32_t *raw_offsets;
  std::uint32_t pending_batches;
  std::uint32_t *query_owners;

  template <class Access>
  __device__ __forceinline__ void run(Access access) const {
    tqrj_hash_lookup(access, tasks, task_count, query_bases, query_counts,
        query_tiles, pending_tiles, counters, hash_table, maximum_hash_entries,
        winners, raw_offsets, pending_batches, query_owners);
  }
};

// Preserve the distinct native and sparse launch bounds.
__global__ void tqrj_hash_lookup_kernel(
    NativeTqrjAccess access, TqrjHashLaunch work) {
  work.run(access);
}

}  // namespace gpulsmopt2_detail
