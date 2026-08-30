#pragma once

#include "capsule.cuh"

namespace gpulsm_sparse {

constexpr std::uint32_t kContinuationTokenBits = 36u;
constexpr std::uint64_t kContinuationTokenMask =
    (std::uint64_t{1u} << kContinuationTokenBits) - 1u;

template <class Source>
__global__ void initialize_universal_records(
    Source source, const std::uint32_t *prepared_heads,
    const std::uint32_t *prepared_refs, std::uint32_t *heads,
    std::uint32_t *sort_heads, std::uint32_t *refs,
    std::uint32_t rows) {
  const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const std::uint32_t ref = prepared_refs ? prepared_refs[row] : row;
  const std::uint32_t head = prepared_heads
      ? prepared_heads[row] : source_key(source, ref).head4();
  heads[row] = head;
  sort_heads[row] = head;
  refs[row] = ref;
}

template <class Source>
__global__ void classify_primary_groups(
    Source source, const std::uint32_t *sorted_refs,
    const std::uint32_t *group_counts,
    const std::uint32_t *group_starts,
    const std::uint32_t *group_count,
    std::uint32_t *active_rows, std::uint32_t *active_groups,
    std::uint32_t *group_depths) {
  const std::uint32_t group = blockIdx.x;
  if (group >= *group_count) return;
  const std::uint32_t begin = group_starts[group];
  const std::uint32_t count = group_counts[group];
  __shared__ std::uint32_t ambiguous;
  __shared__ std::uint32_t short_key;
  if (threadIdx.x == 0u) {
    ambiguous = 0u;
    short_key = 0u;
  }
  __syncthreads();
  for (std::uint32_t local = threadIdx.x; local < count;
       local += blockDim.x) {
    const std::uint32_t ref = sorted_refs[begin + local];
    const std::uint64_t length = source_key(source, ref).length();
    if (length != 4u || source_tombstone(source, ref))
      atomicExch(&ambiguous, 1u);
    if (length < 4u) atomicExch(&short_key, 1u);
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    active_rows[group] = ambiguous ? count : 0u;
    active_groups[group] = ambiguous;
    group_depths[group] = short_key ? 0u : 4u;
  }
}

__global__ void finish_scan_total(
    const std::uint32_t *values, const std::uint32_t *offsets,
    const std::uint32_t *count, std::uint32_t *total) {
  if (blockIdx.x || threadIdx.x) return;
  const std::uint32_t size = *count;
  *total = size ? offsets[size - 1u] + values[size - 1u] : 0u;
}

__global__ void initialize_exact_tasks(
    const std::uint32_t *sorted_refs,
    const std::uint32_t *group_counts,
    const std::uint32_t *group_starts,
    const std::uint32_t *group_count,
    const std::uint32_t *active_offsets,
    const std::uint32_t *active_group_ids,
    const std::uint32_t *active_groups,
    const std::uint32_t *group_depths,
    std::uint32_t *active_refs, std::uint32_t *active_task_ids,
    std::uint32_t *ordered_refs, std::uint32_t *task_begins,
    std::uint32_t *task_bases, std::uint32_t *task_depths) {
  const std::uint32_t group = blockIdx.x;
  if (group >= *group_count) return;
  const std::uint32_t begin = group_starts[group];
  const std::uint32_t count = group_counts[group];
  if (!active_groups[group]) {
    for (std::uint32_t local = threadIdx.x; local < count;
         local += blockDim.x)
      ordered_refs[begin + local] = sorted_refs[begin + local];
    return;
  }
  const std::uint32_t output = active_offsets[group];
  const std::uint32_t task = active_group_ids[group];
  for (std::uint32_t local = threadIdx.x; local < count;
       local += blockDim.x) {
    active_refs[output + local] = sorted_refs[begin + local];
    active_task_ids[output + local] = task;
  }
  if (threadIdx.x == 0u) {
    task_begins[task] = output;
    task_bases[task] = begin;
    task_depths[task] = group_depths[group];
  }
}

template <class Source>
__global__ void make_continuation_composites(
    Source source, const std::uint32_t *refs,
    const std::uint32_t *task_ids,
    const std::uint32_t *task_depths,
    std::uint64_t *composites, std::uint32_t active_rows) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= active_rows) return;
  const std::uint32_t ref = refs[position];
  const std::uint32_t task = task_ids[position];
  const std::uint32_t depth = task_depths[task];
  const auto key = source_key(source, ref);
  std::uint64_t token = 0u;
  for (std::uint32_t symbol = 0u; symbol < 4u; ++symbol) {
    token <<= 9u;
    const std::uint64_t byte = std::uint64_t{depth} + symbol;
    if (byte < key.length()) token |= std::uint32_t{key.byte(byte)} + 1u;
  }
  composites[position] =
      (std::uint64_t{task} << kContinuationTokenBits) | token;
}

__device__ __forceinline__ bool continuation_token_has_end(
    std::uint64_t token) {
  for (std::uint32_t symbol = 0u; symbol < 4u; ++symbol) {
    if (!(token & 0x1ffu)) return true;
    token >>= 9u;
  }
  return false;
}

__global__ void classify_continuation_groups(
    const std::uint64_t *unique_composites,
    const std::uint32_t *group_counts,
    const std::uint32_t *group_count,
    std::uint32_t *continuation_rows,
    std::uint32_t *continuation_groups) {
  const std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= *group_count) return;
  const std::uint64_t token =
      unique_composites[group] & kContinuationTokenMask;
  const bool continuation =
      group_counts[group] > 1u && !continuation_token_has_end(token);
  continuation_rows[group] = continuation ? group_counts[group] : 0u;
  continuation_groups[group] = continuation;
}

__global__ void scatter_continuation_groups(
    const std::uint64_t *unique_composites,
    const std::uint32_t *sorted_refs,
    const std::uint32_t *group_counts,
    const std::uint32_t *group_starts,
    const std::uint32_t *group_count,
    const std::uint32_t *continuation_offsets,
    const std::uint32_t *continuation_group_ids,
    const std::uint32_t *continuation_groups,
    const std::uint32_t *task_begins,
    const std::uint32_t *task_bases,
    const std::uint32_t *task_depths,
    std::uint32_t *next_refs, std::uint32_t *next_task_ids,
    std::uint32_t *ordered_refs,
    std::uint32_t *next_task_begins,
    std::uint32_t *next_task_bases,
    std::uint32_t *next_task_depths) {
  const std::uint32_t group = blockIdx.x;
  if (group >= *group_count) return;
  const std::uint32_t count = group_counts[group];
  const std::uint32_t begin = group_starts[group];
  const std::uint32_t parent = static_cast<std::uint32_t>(
      unique_composites[group] >> kContinuationTokenBits);
  const std::uint32_t base =
      task_bases[parent] + begin - task_begins[parent];
  if (!continuation_groups[group]) {
    for (std::uint32_t local = threadIdx.x; local < count;
         local += blockDim.x)
      ordered_refs[base + local] = sorted_refs[begin + local];
    return;
  }
  const std::uint32_t output = continuation_offsets[group];
  const std::uint32_t child = continuation_group_ids[group];
  for (std::uint32_t local = threadIdx.x; local < count;
       local += blockDim.x) {
    next_refs[output + local] = sorted_refs[begin + local];
    next_task_ids[output + local] = child;
  }
  if (threadIdx.x == 0u) {
    next_task_begins[child] = output;
    next_task_bases[child] = base;
    next_task_depths[child] = task_depths[parent] + 4u;
  }
}

template <class Source>
__global__ void mark_last_universal_key(
    Source source, const std::uint32_t *ordered,
    std::uint8_t *last, std::uint32_t rows) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= rows) return;
  bool keep = position + 1u == rows;
  if (!keep) {
    const auto left = source_key(source, ordered[position]);
    const auto right = source_key(source, ordered[position + 1u]);
    keep = compare_keys(left, right) != 0;
  }
  last[position] = static_cast<std::uint8_t>(keep);
}

template <class Source>
__global__ void classify_winners(
    Source source, const std::uint32_t *winners,
    const std::uint32_t *winner_count,
    const std::uint32_t *unique_heads,
    const std::uint32_t *primary_group_flags,
    const std::uint32_t *primary_group_count,
    std::uint8_t *exact_flags, std::uint8_t *capsule_flags) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t count = *winner_count;
  if (position >= count) return;
  const std::uint32_t ref = winners[position];
  const auto key = source_key(source, ref);
  const auto value = source_value(source, ref);
  const std::uint32_t head = key.head4();
  std::uint32_t low = 0u;
  std::uint32_t high = *primary_group_count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (unique_heads[middle] < head) low = middle + 1u;
    else high = middle;
  }
  exact_flags[position] =
      low < *primary_group_count && unique_heads[low] == head
          ? static_cast<std::uint8_t>(primary_group_flags[low] != 0u)
          : 0u;
  const bool tombstone = source_tombstone(source, ref);
  const std::uint32_t inline_value = value.inline_word();
  const std::uint32_t summary = source_summary(source, ref);
  capsule_flags[position] = static_cast<std::uint8_t>(
      key.length() != 4u || (!tombstone && value.length() != 4u) ||
      (!tombstone && summary != inline_value));
}

struct ExactOrderResult {
  std::uint32_t winner_count{};
  std::uint32_t primary_groups{};
  std::uint32_t active_rows{};
  std::uint32_t rounds{};
};

class ExactOrderWorkspace {
 public:
  explicit ExactOrderWorkspace(std::uint32_t capacity)
      : capacity_(capacity), heads_(capacity), sort_heads_a_(capacity),
        sort_heads_b_(capacity), refs_a_(capacity), refs_b_(capacity),
        task_ids_a_(capacity), task_ids_b_(capacity),
        unique_heads_(capacity), primary_counts_(capacity),
        primary_starts_(capacity), primary_count_(1u),
        primary_rows_(capacity), primary_flags_(capacity),
        primary_depths_(capacity), primary_offsets_(capacity),
        primary_ids_(capacity), composites_a_(capacity),
        composites_b_(capacity), unique_composites_(capacity),
        round_counts_(capacity), round_starts_(capacity), round_count_(1u),
        continuation_rows_(capacity), continuation_flags_(capacity),
        continuation_offsets_(capacity), continuation_ids_(capacity),
        task_begins_a_(capacity), task_begins_b_(capacity),
        task_bases_a_(capacity), task_bases_b_(capacity),
        task_depths_a_(capacity), task_depths_b_(capacity),
        ordered_(capacity), last_(capacity), winners_(capacity),
        winner_count_(1u), totals_(2u), exact_flags_(capacity),
        capsule_flags_(capacity) {
    size_temporary();
  }

  template <class Source>
  ExactOrderResult order(
      Source source, std::uint32_t rows, cudaStream_t stream,
      const std::uint32_t *prepared_heads = nullptr,
      const std::uint32_t *prepared_refs = nullptr) {
    if (rows > capacity_) throw std::length_error("exact order capacity");
    check(cudaMemsetAsync(
              ordered_.data(), 0xff, rows * sizeof(std::uint32_t), stream),
          "clear exact order");
    initialize_universal_records<<<blocks(rows), kThreads, 0, stream>>>(
        source, prepared_heads, prepared_refs, heads_.data(),
        sort_heads_a_.data(), refs_a_.data(), rows);
    std::size_t bytes = sort32_bytes_;
    check(cub::DeviceRadixSort::SortPairs(
              temporary_.data(), bytes, sort_heads_a_.data(),
              sort_heads_b_.data(), refs_a_.data(), refs_b_.data(), rows,
              0, 32, stream),
          "sort universal heads");
    bytes = rle32_bytes_;
    check(cub::DeviceRunLengthEncode::Encode(
              temporary_.data(), bytes, sort_heads_b_.data(),
              unique_heads_.data(), primary_counts_.data(),
              primary_count_.data(), rows, stream),
          "encode universal heads");
    std::uint32_t groups = 0u;
    check(cudaMemcpyAsync(&groups, primary_count_.data(), sizeof(groups),
                          cudaMemcpyDeviceToHost, stream),
          "copy head group count");
    check(cudaStreamSynchronize(stream), "wait head group count");
    bytes = scan32_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              temporary_.data(), bytes, primary_counts_.data(),
              primary_starts_.data(), groups, stream),
          "scan head groups");
    classify_primary_groups<<<groups, kThreads, 0, stream>>>(
        source, refs_b_.data(), primary_counts_.data(),
        primary_starts_.data(), primary_count_.data(), primary_rows_.data(),
        primary_flags_.data(), primary_depths_.data());
    bytes = scan32_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              temporary_.data(), bytes, primary_rows_.data(),
              primary_offsets_.data(), groups, stream),
          "scan active rows");
    bytes = scan32_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              temporary_.data(), bytes, primary_flags_.data(),
              primary_ids_.data(), groups, stream),
          "scan active groups");
    finish_scan_total<<<1, 1, 0, stream>>>(
        primary_rows_.data(), primary_offsets_.data(),
        primary_count_.data(), totals_.data());
    finish_scan_total<<<1, 1, 0, stream>>>(
        primary_flags_.data(), primary_ids_.data(),
        primary_count_.data(), totals_.data() + 1u);
    std::uint32_t initial[2]{};
    check(cudaMemcpyAsync(initial, totals_.data(), sizeof(initial),
                          cudaMemcpyDeviceToHost, stream),
          "copy initial exact totals");
    check(cudaStreamSynchronize(stream), "wait exact totals");
    initialize_exact_tasks<<<groups, kThreads, 0, stream>>>(
        refs_b_.data(), primary_counts_.data(), primary_starts_.data(),
        primary_count_.data(), primary_offsets_.data(), primary_ids_.data(),
        primary_flags_.data(), primary_depths_.data(), refs_a_.data(),
        task_ids_a_.data(), ordered_.data(), task_begins_a_.data(),
        task_bases_a_.data(), task_depths_a_.data());

    std::uint32_t active_rows = initial[0];
    std::uint32_t rounds = 0u;
    bool task_ping = false;
    while (active_rows) {
      if (++rounds > 64u)
        throw std::runtime_error("exact continuation did not converge");
      std::uint32_t *task_begins = task_ping
          ? task_begins_b_.data() : task_begins_a_.data();
      std::uint32_t *task_bases = task_ping
          ? task_bases_b_.data() : task_bases_a_.data();
      std::uint32_t *task_depths = task_ping
          ? task_depths_b_.data() : task_depths_a_.data();
      std::uint32_t *next_begins = task_ping
          ? task_begins_a_.data() : task_begins_b_.data();
      std::uint32_t *next_bases = task_ping
          ? task_bases_a_.data() : task_bases_b_.data();
      std::uint32_t *next_depths = task_ping
          ? task_depths_a_.data() : task_depths_b_.data();
      make_continuation_composites<<<
          blocks(active_rows), kThreads, 0, stream>>>(
          source, refs_a_.data(), task_ids_a_.data(), task_depths,
          composites_a_.data(), active_rows);
      bytes = sort64_bytes_;
      check(cub::DeviceRadixSort::SortPairs(
                temporary_.data(), bytes, composites_a_.data(),
                composites_b_.data(), refs_a_.data(), refs_b_.data(),
                active_rows, 0, 64, stream),
            "sort exact continuation");
      bytes = rle64_bytes_;
      check(cub::DeviceRunLengthEncode::Encode(
                temporary_.data(), bytes, composites_b_.data(),
                unique_composites_.data(), round_counts_.data(),
                round_count_.data(), active_rows, stream),
            "encode exact continuation");
      std::uint32_t round_groups = 0u;
      check(cudaMemcpyAsync(&round_groups, round_count_.data(),
                            sizeof(round_groups), cudaMemcpyDeviceToHost,
                            stream),
            "copy exact round groups");
      check(cudaStreamSynchronize(stream), "wait exact round groups");
      bytes = scan32_bytes_;
      check(cub::DeviceScan::ExclusiveSum(
                temporary_.data(), bytes, round_counts_.data(),
                round_starts_.data(), round_groups, stream),
            "scan exact round groups");
      classify_continuation_groups<<<
          blocks(round_groups), kThreads, 0, stream>>>(
          unique_composites_.data(), round_counts_.data(),
          round_count_.data(), continuation_rows_.data(),
          continuation_flags_.data());
      bytes = scan32_bytes_;
      check(cub::DeviceScan::ExclusiveSum(
                temporary_.data(), bytes, continuation_rows_.data(),
                continuation_offsets_.data(), round_groups, stream),
            "scan continuation rows");
      bytes = scan32_bytes_;
      check(cub::DeviceScan::ExclusiveSum(
                temporary_.data(), bytes, continuation_flags_.data(),
                continuation_ids_.data(), round_groups, stream),
            "scan continuation tasks");
      finish_scan_total<<<1, 1, 0, stream>>>(
          continuation_rows_.data(), continuation_offsets_.data(),
          round_count_.data(), totals_.data());
      finish_scan_total<<<1, 1, 0, stream>>>(
          continuation_flags_.data(), continuation_ids_.data(),
          round_count_.data(), totals_.data() + 1u);
      scatter_continuation_groups<<<round_groups, kThreads, 0, stream>>>(
          unique_composites_.data(), refs_b_.data(), round_counts_.data(),
          round_starts_.data(), round_count_.data(),
          continuation_offsets_.data(), continuation_ids_.data(),
          continuation_flags_.data(), task_begins, task_bases, task_depths,
          refs_a_.data(), task_ids_a_.data(), ordered_.data(), next_begins,
          next_bases, next_depths);
      std::uint32_t next[2]{};
      check(cudaMemcpyAsync(next, totals_.data(), sizeof(next),
                            cudaMemcpyDeviceToHost, stream),
            "copy continuation totals");
      check(cudaStreamSynchronize(stream), "wait continuation totals");
      active_rows = next[0];
      task_ping = !task_ping;
    }

    mark_last_universal_key<<<blocks(rows), kThreads, 0, stream>>>(
        source, ordered_.data(), last_.data(), rows);
    bytes = select_bytes_;
    check(cub::DeviceSelect::Flagged(
              temporary_.data(), bytes, ordered_.data(), last_.data(),
              winners_.data(), winner_count_.data(), rows, stream),
          "select exact winners");
    std::uint32_t winners = 0u;
    check(cudaMemcpyAsync(&winners, winner_count_.data(), sizeof(winners),
                          cudaMemcpyDeviceToHost, stream),
          "copy exact winner count");
    check(cudaStreamSynchronize(stream), "wait exact winner count");
    classify_winners<<<blocks(winners), kThreads, 0, stream>>>(
        source, winners_.data(), winner_count_.data(), unique_heads_.data(),
        primary_flags_.data(), primary_count_.data(), exact_flags_.data(),
        capsule_flags_.data());
    check(cudaGetLastError(), "classify exact winners");
    return {winners, groups, initial[0], rounds};
  }

  std::uint32_t *heads() const { return heads_.data(); }
  std::uint32_t *ordered() const { return ordered_.data(); }
  std::uint32_t *winners() const { return winners_.data(); }
  std::uint32_t *winner_count() const { return winner_count_.data(); }
  std::uint8_t *exact_flags() const { return exact_flags_.data(); }
  std::uint8_t *capsule_flags() const { return capsule_flags_.data(); }
  std::size_t bytes() const {
    return temporary_.bytes() +
        26u * std::size_t{capacity_} * sizeof(std::uint32_t) +
        3u * std::size_t{capacity_} * sizeof(std::uint64_t) +
        3u * std::size_t{capacity_} * sizeof(std::uint8_t);
  }

 private:
  void size_temporary() {
    std::size_t sizes[7]{};
    check(cub::DeviceRadixSort::SortPairs(
              nullptr, sizes[0], sort_heads_a_.data(), sort_heads_b_.data(),
              refs_a_.data(), refs_b_.data(), capacity_),
          "size head sort");
    check(cub::DeviceRadixSort::SortPairs(
              nullptr, sizes[1], composites_a_.data(), composites_b_.data(),
              refs_a_.data(), refs_b_.data(), capacity_),
          "size continuation sort");
    check(cub::DeviceRunLengthEncode::Encode(
              nullptr, sizes[2], sort_heads_b_.data(), unique_heads_.data(),
              primary_counts_.data(), primary_count_.data(), capacity_),
          "size head encoding");
    check(cub::DeviceRunLengthEncode::Encode(
              nullptr, sizes[3], composites_b_.data(),
              unique_composites_.data(), round_counts_.data(),
              round_count_.data(), capacity_),
          "size continuation encoding");
    check(cub::DeviceScan::ExclusiveSum(
              nullptr, sizes[4], primary_counts_.data(),
              primary_starts_.data(), capacity_),
          "size exact scan");
    check(cub::DeviceSelect::Flagged(
              nullptr, sizes[5], ordered_.data(), last_.data(),
              winners_.data(), winner_count_.data(), capacity_),
          "size winner selection");
    check(cub::DeviceSelect::Flagged(
              nullptr, sizes[6], winners_.data(), exact_flags_.data(),
              refs_b_.data(), totals_.data(), capacity_),
          "size exact selection");
    sort32_bytes_ = sizes[0];
    sort64_bytes_ = sizes[1];
    rle32_bytes_ = sizes[2];
    rle64_bytes_ = sizes[3];
    scan32_bytes_ = sizes[4];
    select_bytes_ = std::max(sizes[5], sizes[6]);
    temporary_.reset(*std::max_element(std::begin(sizes), std::end(sizes)));
  }

  std::uint32_t capacity_{};
  Buffer<std::uint32_t> heads_, sort_heads_a_, sort_heads_b_, refs_a_, refs_b_;
  Buffer<std::uint32_t> task_ids_a_, task_ids_b_;
  Buffer<std::uint32_t> unique_heads_, primary_counts_, primary_starts_;
  Buffer<std::uint32_t> primary_count_, primary_rows_, primary_flags_;
  Buffer<std::uint32_t> primary_depths_, primary_offsets_, primary_ids_;
  Buffer<std::uint64_t> composites_a_, composites_b_, unique_composites_;
  Buffer<std::uint32_t> round_counts_, round_starts_, round_count_;
  Buffer<std::uint32_t> continuation_rows_, continuation_flags_;
  Buffer<std::uint32_t> continuation_offsets_, continuation_ids_;
  Buffer<std::uint32_t> task_begins_a_, task_begins_b_;
  Buffer<std::uint32_t> task_bases_a_, task_bases_b_;
  Buffer<std::uint32_t> task_depths_a_, task_depths_b_;
  Buffer<std::uint32_t> ordered_, winners_, winner_count_, totals_;
  Buffer<std::uint8_t> last_, exact_flags_, capsule_flags_;
  Buffer<std::uint8_t> temporary_;
  std::size_t sort32_bytes_{}, sort64_bytes_{}, rle32_bytes_{};
  std::size_t rle64_bytes_{}, scan32_bytes_{}, select_bytes_{};
};

}  // namespace gpulsm_sparse
