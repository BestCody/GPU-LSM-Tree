#pragma once

#include "refine.cuh"

namespace gpulsm_sparse {

// Direct extraction of the observed-head front of the closure prototype's
// UniversalRootWorkspace.  It produces exactly one restored projection row
// per head and compacts only exceptional head groups for exact refinement.
template <class Source>
__device__ __forceinline__ bool direct_source_requires_sparse(
    Source source, std::uint32_t ref) {
  const auto key = source_key(source, ref);
  const auto value = source_value(source, ref);
  const bool tombstone = source_tombstone(source, ref);
  const std::uint32_t inline_value = value.inline_word();
  return source_owned_capsule(source, ref) || key.length() != 4u ||
      (!tombstone && value.length() != 4u) ||
      (!tombstone && source_summary(source, ref) != inline_value);
}

template <class Source>
__global__ void classify_direct_head_groups(
    Source source, const std::uint32_t *sorted_refs,
    const std::uint32_t *group_counts, const std::uint32_t *group_starts,
    std::uint32_t group_count, std::uint32_t *sparse_rows,
    std::uint8_t *sparse_flags) {
  const std::uint32_t group = blockIdx.x;
  if (group >= group_count) return;
  const std::uint32_t begin = group_starts[group];
  const std::uint32_t count = group_counts[group];
  __shared__ std::uint32_t sparse;
  if (!threadIdx.x) sparse = 0u;
  __syncthreads();
  for (std::uint32_t local = threadIdx.x; local < count;
       local += blockDim.x)
    if (direct_source_requires_sparse(source, sorted_refs[begin + local]))
      atomicExch(&sparse, 1u);
  __syncthreads();
  if (!threadIdx.x) {
    sparse_rows[group] = sparse ? count : 0u;
    sparse_flags[group] = static_cast<std::uint8_t>(sparse);
  }
}

__global__ void scatter_direct_sparse_refs(
    const std::uint32_t *sorted_refs, const std::uint32_t *group_counts,
    const std::uint32_t *group_starts,
    const std::uint32_t *sparse_offsets,
    const std::uint8_t *sparse_flags, std::uint32_t group_count,
    std::uint32_t *sparse_refs) {
  const std::uint32_t group = blockIdx.x;
  if (group >= group_count || !sparse_flags[group]) return;
  const std::uint32_t count = group_counts[group];
  for (std::uint32_t local = threadIdx.x; local < count;
       local += blockDim.x)
    sparse_refs[sparse_offsets[group] + local] =
        sorted_refs[group_starts[group] + local];
}

template <class Source>
__global__ void emit_direct_projection(
    Source source, const std::uint32_t *heads,
    const std::uint32_t *sorted_refs,
    const std::uint32_t *group_counts, const std::uint32_t *group_starts,
    const std::uint8_t *sparse_flags, std::uint32_t group_count,
    std::uint32_t *projection_keys,
    gpulsmopt2_detail::Row *projection_rows) {
  const std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= group_count) return;
  const std::uint32_t head = heads[group];
  projection_keys[group] = head;
  if (sparse_flags[group]) {
    projection_rows[group] = exact_group_row(head, 0u);
    return;
  }
  // DeviceRadixSort is stable, so the last member of an ordinary head group
  // is the newest input update, matching restored bulk/admission chronology.
  const std::uint32_t ref = sorted_refs[
      group_starts[group] + group_counts[group] - 1u];
  const bool tombstone = source_tombstone(source, ref);
  projection_rows[group] = {
      tombstone ? gpulsmopt2_detail::kInvalid : source_summary(source, ref),
      static_cast<std::uint16_t>(head),
      static_cast<std::uint16_t>(
          tombstone ? gpulsmopt2_detail::kTombstone : 0u)};
}

__global__ void build_direct_section_offsets(
    const std::uint32_t *heads, std::uint32_t count,
    std::uint32_t *offsets) {
  const std::uint32_t section = blockIdx.x * blockDim.x + threadIdx.x;
  if (section > kSections) return;
  if (section == kSections) {
    offsets[section] = count;
    return;
  }
  const std::uint32_t target = section << 16u;
  std::uint32_t low = 0u;
  std::uint32_t high = count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) >> 1u);
    if (heads[middle] < target) low = middle + 1u;
    else high = middle;
  }
  offsets[section] = low;
}

__global__ void build_direct_epoch_counts_and_ranks(
    const gpulsmopt2_detail::Row *rows,
    const std::uint32_t *section_offsets, std::uint32_t *section_counts,
    std::uint16_t *cell_ranks) {
  const std::uint32_t section = blockIdx.x;
  const std::uint32_t cell = threadIdx.x;
  const std::uint32_t begin = section_offsets[section];
  const std::uint32_t count =
      section_offsets[section + 1u] - begin;
  if (!cell) {
    section_counts[section] = count;
    if (!section) section_counts[kSections] = 0u;
  }
  cell_ranks[std::size_t{section} *
                 gpulsmopt2_detail::kFoundationCells + cell] =
      static_cast<std::uint16_t>(gpulsmopt2_detail::lower_bound_rows(
          rows + begin, count,
          cell * gpulsmopt2_detail::kFoundationCellKeys));
}

struct DirectRootPreparation {
  std::uint32_t projection_rows{};
  std::uint32_t logical_rows{};
  std::uint32_t sparse_heads{};
  std::uint32_t sparse_input_rows{};
};

class DirectRootWorkspace {
 public:
  explicit DirectRootWorkspace(std::uint32_t capacity)
      : capacity_(capacity), heads_a_(capacity), heads_b_(capacity),
        refs_a_(capacity), refs_b_(capacity), group_heads_(capacity),
        group_counts_(capacity), group_starts_(capacity), group_count_(1u),
        sparse_rows_(capacity), sparse_offsets_(capacity),
        sparse_flags_(capacity), sparse_heads_(capacity),
        sparse_head_count_(1u), sparse_refs_(capacity), totals_(1u) {
    projection_keys_.reset(capacity);
    projection_rows_.reset(capacity);
    section_offsets_.reset(kSections + 1u);
    section_counts_.reset(kSections + 1u);
    epoch_ranks_.reset(gpulsmopt2_detail::kLocalRankEntries);
    std::size_t sizes[4]{};
    check(cub::DeviceRadixSort::SortPairs(
              nullptr, sizes[0], heads_a_.data(), heads_b_.data(),
              refs_a_.data(), refs_b_.data(), capacity),
          "size direct head sort");
    check(cub::DeviceRunLengthEncode::Encode(
              nullptr, sizes[1], heads_b_.data(), group_heads_.data(),
              group_counts_.data(), group_count_.data(), capacity),
          "size direct head encoding");
    check(cub::DeviceScan::ExclusiveSum(
              nullptr, sizes[2], sparse_rows_.data(),
              sparse_offsets_.data(), capacity),
          "size direct sparse scan");
    check(cub::DeviceSelect::Flagged(
              nullptr, sizes[3], group_heads_.data(), sparse_flags_.data(),
              sparse_heads_.data(), sparse_head_count_.data(), capacity),
          "size direct sparse-head selection");
    sort_bytes_ = sizes[0];
    rle_bytes_ = sizes[1];
    scan_bytes_ = sizes[2];
    select_bytes_ = sizes[3];
    temporary_.reset(*std::max_element(
        std::begin(sizes), std::end(sizes)));
  }

  DirectRootPreparation prepare(
      RecordBatchView source, std::uint32_t rows, cudaStream_t stream) {
    if (!rows || rows > capacity_)
      throw std::length_error("direct root capacity");
    initialize_universal_records<<<blocks(rows), kThreads, 0, stream>>>(
        source, nullptr, nullptr, heads_a_.data(), heads_a_.data(),
        refs_a_.data(), rows);
    std::size_t bytes = sort_bytes_;
    check(cub::DeviceRadixSort::SortPairs(
              temporary_.data(), bytes, heads_a_.data(), heads_b_.data(),
              refs_a_.data(), refs_b_.data(), rows, 0, 32, stream),
          "sort direct root heads");
    bytes = rle_bytes_;
    check(cub::DeviceRunLengthEncode::Encode(
              temporary_.data(), bytes, heads_b_.data(),
              group_heads_.data(), group_counts_.data(),
              group_count_.data(), rows, stream),
          "encode direct root heads");
    std::uint32_t groups = 0u;
    check(cudaMemcpyAsync(&groups, group_count_.data(), sizeof(groups),
                          cudaMemcpyDeviceToHost, stream),
          "copy direct projection count");
    check(cudaStreamSynchronize(stream),
          "wait direct projection count");
    bytes = scan_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              temporary_.data(), bytes, group_counts_.data(),
              group_starts_.data(), groups, stream),
          "scan direct head groups");
    classify_direct_head_groups<<<groups, kThreads, 0, stream>>>(
        source, refs_b_.data(), group_counts_.data(), group_starts_.data(),
        groups, sparse_rows_.data(), sparse_flags_.data());
    bytes = scan_bytes_;
    check(cub::DeviceScan::ExclusiveSum(
              temporary_.data(), bytes, sparse_rows_.data(),
              sparse_offsets_.data(), groups, stream),
          "scan direct sparse rows");
    finish_scan_total<<<1, 1, 0, stream>>>(
        sparse_rows_.data(), sparse_offsets_.data(), group_count_.data(),
        totals_.data());
    bytes = select_bytes_;
    check(cub::DeviceSelect::Flagged(
              temporary_.data(), bytes, group_heads_.data(),
              sparse_flags_.data(), sparse_heads_.data(),
              sparse_head_count_.data(), groups, stream),
          "select direct sparse heads");
    std::uint32_t sparse[2]{};
    check(cudaMemcpyAsync(sparse, sparse_head_count_.data(),
                          sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                          stream),
          "copy direct sparse head count");
    check(cudaMemcpyAsync(sparse + 1u, totals_.data(),
                          sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                          stream),
          "copy direct sparse row count");
    check(cudaStreamSynchronize(stream), "wait direct sparse counts");

    std::uint32_t sparse_winners = 0u;
    if (sparse[1]) {
      scatter_direct_sparse_refs<<<groups, kThreads, 0, stream>>>(
          refs_b_.data(), group_counts_.data(), group_starts_.data(),
          sparse_offsets_.data(), sparse_flags_.data(), groups,
          sparse_refs_.data());
      exact_order_ = std::make_unique<ExactOrderWorkspace>(sparse[1]);
      sparse_winners = exact_order_->order(
          source, sparse[1], stream, nullptr, sparse_refs_.data())
          .winner_count;
    }
    const std::uint64_t logical =
        std::uint64_t{groups} - sparse[0] + sparse_winners;
    if (logical > std::numeric_limits<std::uint32_t>::max())
      throw std::length_error("direct logical root exceeds 32 bits");
    return {groups, static_cast<std::uint32_t>(logical), sparse[0],
            sparse[1]};
  }

  void materialize_projection(
      RecordBatchView source, std::uint32_t groups,
      std::uint32_t *projection_keys,
      gpulsmopt2_detail::Row *projection_rows,
      cudaStream_t stream) const {
    emit_direct_projection<<<blocks(groups), kThreads, 0, stream>>>(
        source, group_heads_.data(), refs_b_.data(), group_counts_.data(),
        group_starts_.data(), sparse_flags_.data(), groups,
        projection_keys, projection_rows);
    check(cudaGetLastError(), "materialize direct root projection");
  }

  void materialize_epoch(
      RecordBatchView source, std::uint32_t groups,
      cudaStream_t stream) {
    materialize_projection(source, groups, projection_keys_.data(),
                           projection_rows_.data(), stream);
    build_direct_section_offsets<<<blocks(kSections + 1u), kThreads, 0,
                                   stream>>>(
        projection_keys_.data(), groups, section_offsets_.data());
    build_direct_epoch_counts_and_ranks<<<
        kSections, gpulsmopt2_detail::kFoundationCells, 0, stream>>>(
        projection_rows_.data(), section_offsets_.data(),
        section_counts_.data(), epoch_ranks_.data());
    check(cudaGetLastError(), "materialize direct projection epoch");
  }

  const std::uint32_t *sorted_heads() const { return heads_b_.data(); }
  const std::uint32_t *sorted_refs() const { return refs_b_.data(); }
  const std::uint32_t *sparse_heads() const {
    return sparse_heads_.data();
  }
  const std::uint32_t *projection_keys() const {
    return projection_keys_.data();
  }
  const gpulsmopt2_detail::Row *projection_rows() const {
    return projection_rows_.data();
  }
  const std::uint32_t *section_offsets() const {
    return section_offsets_.data();
  }
  const std::uint32_t *section_counts() const {
    return section_counts_.data();
  }
  const std::uint16_t *epoch_ranks() const { return epoch_ranks_.data(); }
  const std::uint32_t *projection_count_device() const {
    return group_count_.data();
  }

  std::size_t bytes() const {
    return heads_a_.bytes() + heads_b_.bytes() + refs_a_.bytes() +
        refs_b_.bytes() + group_heads_.bytes() + group_counts_.bytes() +
        group_starts_.bytes() + group_count_.bytes() +
        sparse_rows_.bytes() + sparse_offsets_.bytes() +
        sparse_flags_.bytes() + sparse_heads_.bytes() +
        sparse_head_count_.bytes() + sparse_refs_.bytes() + totals_.bytes() +
        projection_keys_.bytes() + projection_rows_.bytes() +
        section_offsets_.bytes() + section_counts_.bytes() +
        epoch_ranks_.bytes() + temporary_.bytes() +
        (exact_order_ ? exact_order_->bytes() : 0u);
  }

 private:
  std::uint32_t capacity_{};
  Buffer<std::uint32_t> heads_a_, heads_b_, refs_a_, refs_b_;
  Buffer<std::uint32_t> group_heads_, group_counts_, group_starts_;
  Buffer<std::uint32_t> group_count_, sparse_rows_, sparse_offsets_;
  Buffer<std::uint8_t> sparse_flags_;
  Buffer<std::uint32_t> sparse_heads_, sparse_head_count_, sparse_refs_;
  Buffer<std::uint32_t> totals_;
  Buffer<std::uint32_t> projection_keys_, section_offsets_,
      section_counts_;
  Buffer<gpulsmopt2_detail::Row> projection_rows_;
  Buffer<std::uint16_t> epoch_ranks_;
  Buffer<std::uint8_t> temporary_;
  std::unique_ptr<ExactOrderWorkspace> exact_order_;
  std::size_t sort_bytes_{}, rle_bytes_{}, scan_bytes_{}, select_bytes_{};
};

__global__ void initialize_direct_sparse_manifest(
    DeviceSparseManifest *manifests, const RootBuildState *root,
    std::uint32_t level, std::uint32_t flags) {
  if (blockIdx.x || threadIdx.x) return;
  DeviceSparseManifest manifest{};
  manifest.generation = 1u;
  if (root && flags) {
    manifest.levels[level] = {
        reinterpret_cast<std::uint64_t>(root), root->generation, flags};
    if (flags & kSparseHasExactHeads)
      manifest.exact_level_mask = std::uint64_t{1u} << level;
    if (flags & kSparseHasCapsules)
      manifest.capsule_level_mask = std::uint64_t{1u} << level;
  }
  manifests[0] = manifest;
  manifests[1] = manifest;
}

}  // namespace gpulsm_sparse
