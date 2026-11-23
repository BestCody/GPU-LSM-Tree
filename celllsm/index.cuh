#pragma once

// The GPULSMOpt class: owns all storage and runs each operation.

#include "../gpu_dictionary_adapter.cuh"
#include "device_memory.cuh"
#include "publish.cuh"
#include "merge.cuh"
#include "large_batch.cuh"
#include "lookup.cuh"
#include "range.cuh"
#include "successor.cuh"

namespace gpulsmopt2_detail {

struct PinnedReceipts {
  ResidentPublicationPlan publication;
  std::uint64_t range_total;
  std::uint64_t range_hot_total;
  std::uint64_t range_hot_offsets[kQuotients + 1u];
};

static_assert(sizeof(PinnedReceipts) ==
              sizeof(ResidentPublicationPlan) +
                  (kQuotients + 3u) * sizeof(std::uint64_t));

}  // namespace gpulsmopt2_detail

class GPULSMOpt {
public:
  struct BulkBootstrapTag {};

  struct DeviceKeyBatch {
    const std::uint32_t *keys = nullptr;
    std::size_t count = 0u;
  };

  explicit GPULSMOpt(const DictionaryConfig &config)
      : batch_capacity_(std::min(
            gpulsmopt2_detail::kMaximumOperationTile,
            std::max<std::size_t>(1u, config.batch_capacity))),
        publication_capacity_(gpulsmopt2_detail::initial_storage_capacity(
            config.max_elements, batch_capacity_)),
        level_zero_capacity_(gpulsmopt2_detail::initial_level_capacity(
            config.level_zero_capacity,
            batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch,
            publication_capacity_)),
        level_pool_capacity_(
            gpulsmopt2_detail::preassigned_level_pool_capacity(
                publication_capacity_, level_zero_capacity_)),
        canonical_regular_level_count_(
            gpulsmopt2_detail::canonical_regular_level_count(
                publication_capacity_, level_zero_capacity_)),
        canonical_level_count_(gpulsmopt2_detail::canonical_level_count(
            publication_capacity_, level_zero_capacity_)),
        resident_merge_capacity_(
            gpulsmopt2_detail::select_canonical_merge_capacity(
                std::min(
                    gpulsmopt2_detail::kMaximumMergeSources,
                    std::max(2u, canonical_level_count_ + 1u)))),
        maximum_resident_jobs_(
            gpulsmopt2_detail::maximum_resident_merge_jobs(
                publication_capacity_, resident_merge_capacity_)),
        arena_key_flags_(gpulsmopt2_detail::maximum_resident_elements<
                             gpulsmopt2_detail::Row>(),
                         level_pool_capacity_),
        arena_values_(gpulsmopt2_detail::maximum_resident_elements<
                          gpulsmopt2_detail::Row>(),
                      level_pool_capacity_),
        pinned_receipts_(1u),
        phase_workspace_(phase_workspace_maximum_bytes(
                             publication_capacity_, batch_capacity_),
                         phase_workspace_initial_bytes(batch_capacity_)),
        operation_workspace_(),
        sealed_workspace_(),
        raw_keys_(),
        publication_keys_a_(gpulsmopt2_detail::kMaximumPublicationRows,
            std::min(publication_capacity_,
                batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch)),
        publication_rows_a_(gpulsmopt2_detail::kMaximumPublicationRows,
            std::min(publication_capacity_,
                batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch)) {
    initialize_static_workspace_views();
    CUDA_CHECK(cudaEventCreateWithFlags(&operation_done_,
                                         cudaEventDisableTiming));
    initialize_phase_workspace_views();
    initialize_operation_workspace_views();
    ensure_radix_workspace(batch_capacity_);
    std::size_t admission_scan_bytes{};
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, admission_scan_bytes, admission_counts_.data(),
        raw_offsets_.data(), gpulsmopt2_detail::kQuotients + 1u, 0));
    admission_temp_.resize(admission_scan_bytes);
    const DeviceCapabilities capabilities = query_device_capabilities();
    initialize_resident_workspace(capabilities);
    initialize_canonical_workspace(capabilities);
    CUDA_CHECK(cudaEventRecord(operation_done_, 0));
    reset_updates(0);
    CUDA_CHECK(cudaEventRecord(operation_done_, 0));
  }

  // Preserve the bulk-bootstrap constructor API.
  GPULSMOpt(const DictionaryConfig &config, BulkBootstrapTag)
      : GPULSMOpt(config) {}

  GPULSMOpt(const GPULSMOpt &) = delete;
  GPULSMOpt &operator=(const GPULSMOpt &) = delete;

  ~GPULSMOpt() {
    if (operation_done_) {
      cudaEventSynchronize(operation_done_);
      cudaEventDestroy(operation_done_);
    }
  }

  void clear(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    begin_operation(stream);
    reset_updates(stream);
    end_operation(stream);
  }

  void bulk_build(const std::uint32_t *keys, const std::uint32_t *values,
                  std::size_t count, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    if ((count && (!keys || !values)) || count > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("invalid GPULSMOpt initial input");
    begin_operation(stream);
    reset_updates(stream);
    if (!count) {
      end_operation(stream);
      return;
    }
    const std::uint32_t n = static_cast<std::uint32_t>(count);
    const std::size_t phase_restore_bytes = phase_workspace_.size();
    const std::size_t bulk_bytes = gpulsm_sparse::NativeRootWorkspace<false>::required_bytes(count);
    phase_workspace_.grow(bulk_bytes);
    gpulsm_sparse::NativeRootWorkspace<false> workspace(
        n, phase_workspace_.data(), phase_workspace_.size());
    const std::uint32_t base_count = workspace.prepare(keys, values, n, false, false, stream);

    const std::uint32_t level = initial_level_for_records(base_count);
    ensure_level_storage_mapped(level, stream);
    const std::uint64_t destination = level_begin(level);
    const std::uint64_t capacity = level_capacity(level);
    if (base_count > capacity)
      throw std::bad_alloc();
    workspace.deposit(resident_rows(), destination,
                      foundation_source_offsets_.data(), stream);
    gpulsmopt2_detail::ResidentPublicationPlan build_plan{};
    build_plan.destination_level = level;
    build_plan.output_begin = destination;
    build_plan.output_capacity = capacity;
    build_plan.status = gpulsmopt2_detail::kPublicationSuccess;
    finalize_direct_root(build_plan, foundation_source_offsets_.data(), stream);
    level_counts_[level] = base_count;
    host_occupied_level_mask_ = std::uint64_t{1} << level;
    gpulsmopt2_detail::initialize_device_manifest_kernel<<<1, 1, 0, stream>>>(
        device_manifests_.data(), active_device_manifest_.data(),
        query_occupied_level_mask_.data(), level, base_count, 0u);
    CUDA_CHECK(cudaGetLastError());
    // The large-deletion shortcut sorts keys without values. Initialize its
    // separate sorting kernels during bulk build, after the initial run has
    // been copied. Reuse the build workspace; this does not change the index.
    if (count >= epoch_row_capacity() &&
        sealed_workspace_capacity(publication_capacity_, level_zero_capacity_) >
            level_zero_capacity_) {
      const auto warm_rows = static_cast<std::uint32_t>(std::min<std::size_t>(
          {count, epoch_row_capacity(), std::size_t{1u} << 20u}));
      workspace.prepare(keys, nullptr, warm_rows, true, false, stream);
    }
    end_operation(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    phase_workspace_.shrink(phase_restore_bytes);
  }

  void insert(const DeviceKeyValueBatch &batch, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    reject_updates_after_publication_failure();
    admit(batch.keys, batch.values, batch.count, false, stream);
  }

  void erase(const DeviceKeyBatch &batch, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    reject_updates_after_publication_failure();
    admit(batch.keys, nullptr, batch.count, true, stream);
  }


  void lookup(const DeviceLookupBatch &batch, cudaStream_t stream,
              bool quotients_grouped = false) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    prepare_failed_epoch_for_reads(stream);
    lookup_locked(batch, stream, quotients_grouped);
  }


  void range(const DeviceRangeOutputBatch &batch, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    prepare_failed_epoch_for_reads(stream);
    range_locked(batch, stream);
  }


  void successor(const DeviceSuccessorBatch &batch, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    resolve_publication_receipt();
    prepare_failed_epoch_for_reads(stream);
    successor_locked(batch, stream);
  }

  std::uint32_t canonical_carry_status() const {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    auto *self = const_cast<GPULSMOpt *>(this);
    self->resolve_publication_receipt();
    gpulsmopt2_detail::ResidentPublicationPlan plan{};
    CUDA_CHECK(cudaMemcpy(
        &plan, self->resident_plan_.data(), sizeof(plan),
        cudaMemcpyDeviceToHost));
    return plan.status;
  }

private:

  friend struct GPULSMOptDeletionTestAccess;

  class PublicationLifecycle {
   public:
    bool receipt_pending() const { return state_ == State::awaiting_receipt; }
    bool failed() const {
      return state_ == State::failed_needs_signatures ||
          state_ == State::failed_signatures_queued;
    }
    bool needs_signature_rebuild() const {
      return state_ == State::failed_needs_signatures;
    }
    const gpulsmopt2_detail::ResidentPublicationPlan &failure() const {
      return failure_;
    }
    void await_receipt() { state_ = State::awaiting_receipt; }
    bool consume_receipt() {
      if (!receipt_pending()) return false;
      state_ = State::accepting;
      return true;
    }
    void fail(const gpulsmopt2_detail::ResidentPublicationPlan &receipt,
              bool signatures_intact = false) {
      failure_ = receipt;
      state_ = signatures_intact ? State::failed_signatures_queued
                                 : State::failed_needs_signatures;
    }
    // The completion event orders later readers.
    void signatures_queued() { state_ = State::failed_signatures_queued; }
    void reset() { *this = {}; }

   private:
    enum class State : std::uint8_t {
      accepting, awaiting_receipt,
      failed_needs_signatures, failed_signatures_queued
    };
    State state_ = State::accepting;
    gpulsmopt2_detail::ResidentPublicationPlan failure_{};
  };

  std::size_t epoch_row_capacity() const {
    return std::min(level_zero_capacity_,
        batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch);
  }

  template <class AdmitTile, class AdmitSealed>
  void admit_batches(std::uint64_t count, cudaStream_t stream,
                     AdmitTile admit_one, AdmitSealed admit_sealed) {
    begin_operation(stream);
    std::uint64_t consumed = 0u;
    while (consumed < count) {
      if (!pending_batches_) {
        consumed += admit_sealed(consumed, count - consumed);
        if (consumed == count) break;
      }
      // Fit each epoch in its smallest destination.
      const std::uint32_t tile_count = static_cast<std::uint32_t>(
          std::min<std::uint64_t>(count - consumed,
              std::min(batch_capacity_, epoch_row_capacity() - pending_records_)));
      admit_one(consumed, tile_count);
      consumed += tile_count;
      if (consumed < count && publication_.receipt_pending())
        resolve_publication_receipt_on_stream(stream);
      if (consumed < count && publication_.failed()) {
        end_operation(stream);
        throw std::runtime_error(
            "GPULSMOpt publication failed while tiling an update; "
            "accepted pending records were preserved");
      }
    }
    end_operation(stream);
  }

  struct AdmissionTile {
    std::uint32_t slot, count;
    std::uint32_t *keys;
    gpulsmopt2_detail::RawPayload *payloads;
    std::uint32_t *offsets;
    std::uint64_t *signatures;
  };


  AdmissionTile begin_admission_tile(std::uint32_t count, bool tombstones) {
    const std::uint32_t slot = pending_batches_;
    if (slot >= gpulsmopt2_detail::kBatchesPerEpoch)
      throw std::logic_error("GPULSMOpt pending slot overflow");
    pending_records_ += count;
    pending_has_tombstones_ |= tombstones;
    return {slot, count,
            raw_keys_.data() + std::size_t{slot} * batch_capacity_,
            raw_payloads_.data() + std::size_t{slot} * batch_capacity_,
            raw_offsets_.data() +
                std::size_t{slot} * (gpulsmopt2_detail::kQuotients + 1u),
            raw_signatures_.data() +
                std::size_t{slot} * gpulsmopt2_detail::kQuotients};
  }

  void commit_admission_signatures(const AdmissionTile &tile,
                                   cudaStream_t stream) {
    gpulsmopt2_detail::commit_admission_metadata_kernel<<<
        gpulsmopt2_detail::kQuotients / gpulsmopt2_detail::kThreads,
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            admission_counts_.data(), tile.signatures,
            raw_epoch_signatures_.data());
  }

  void finish_admission_tile(const AdmissionTile &tile, cudaStream_t stream) {
    CUDA_CHECK(cudaGetLastError());
    raw_batch_counts_[tile.slot] = tile.count;
    ++pending_batches_;
    if (pending_batches_ == gpulsmopt2_detail::kBatchesPerEpoch ||
        pending_records_ == epoch_row_capacity())
      publish_epoch(stream);
  }

  gpulsmopt2_detail::TqrjLookupWorkspace borrow_tqrj_lookup_workspace(
      std::uint32_t maximum_query_tiles) {
    std::uint8_t *operation_storage = operation_workspace_.data();
    std::uint32_t *query_ids = reinterpret_cast<std::uint32_t *>(
        operation_storage);
    // Reuse reservation ranks after scatter.
    std::uint32_t *reservation_ranks =
        reinterpret_cast<std::uint32_t *>(
            operation_storage + tqrj_hash_table_offset(batch_capacity_));
    std::uint8_t *hash_task_storage =
        operation_storage + tqrj_hash_task_offset(batch_capacity_);
    gpulsmopt2_detail::TqrjHashTile *query_tiles =
        reinterpret_cast<gpulsmopt2_detail::TqrjHashTile *>(
            operation_storage + tqrj_hash_tile_offset(batch_capacity_));

    gpulsmopt2_detail::TqrjLookupWorkspace workspace{};
    workspace.grouped_queries = publication_keys_a_.data();
    workspace.query_ids = query_ids;
    workspace.reservation_ranks = reservation_ranks;
    workspace.active_quotients = range_hot_window_offsets_.data();
    workspace.active_query_counts = foundation_section_output_counts_.data();
    workspace.active_query_offsets = foundation_source_offsets_.data();
    workspace.active_quotient_count = range_hot_selected_count_.data();
    workspace.query_bases = reinterpret_cast<std::uint32_t *>(
        operation_storage + tqrj_hash_query_bases_offset(batch_capacity_));
    workspace.hash_tasks =
        reinterpret_cast<gpulsmopt2_detail::TqrjHashTask *>(
            hash_task_storage);
    workspace.hash_counters = reinterpret_cast<std::uint32_t *>(
        hash_task_storage + gpulsmopt2_detail::kTqrjHashTaskBytes);
    workspace.hash_task_count = publication_selected_count_.data();
    workspace.hash_winners = reinterpret_cast<unsigned long long *>(
        publication_rows_a_.data());
    workspace.hash_query_tiles = query_tiles;
    workspace.hash_pending_tiles = query_tiles + maximum_query_tiles;
    workspace.hash_table = reinterpret_cast<std::uint32_t *>(
        operation_storage + tqrj_hash_table_offset(batch_capacity_));
    workspace.hash_table_capacity =
        tqrj_hash_maximum_entries(batch_capacity_);
    return workspace;
  }

  template <bool WithFound>
  void lookup_tiles_locked(const DeviceLookupBatch &batch,
                           cudaStream_t stream) {
    using namespace gpulsmopt2_detail;
    constexpr std::size_t bytes_per_tile =
        kLookupTileRows * (sizeof(std::uint32_t) + sizeof(std::uint16_t) +
                           (WithFound ? sizeof(std::uint8_t) : 0u)) +
        (kLookupTileBins + 1u) * sizeof(std::uint16_t);
    // Tiles borrow mapped scratch without allocation.
    const std::size_t window_tiles = std::min<std::size_t>(
        kMaximumLookupWindowTiles, operation_workspace_.size() / bytes_per_tile);
    if (!window_tiles)
      throw std::length_error("insufficient GPULSMOpt tile lookup workspace");
    const std::size_t capacity = window_tiles * kLookupTileRows;
    std::uint8_t *storage = operation_workspace_.data();
    auto *local = reinterpret_cast<std::uint32_t *>(storage);
    auto *inverse = reinterpret_cast<std::uint16_t *>(
        storage + capacity * sizeof(std::uint32_t));
    auto *offsets = inverse + capacity;
    auto *found = reinterpret_cast<std::uint8_t *>(
        offsets + (kLookupTileBins + 1u) * window_tiles);
    const TileLookupView view{
        raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
        static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
        raw_signatures_.data(), raw_epoch_signatures_.data(), resident_rows(),
        descriptors_.data(), canonical_cell_ranks_.data(),
        query_occupied_level_mask_.data()};
    constexpr std::uint32_t tiles_per_block = kThreads / kLookupTileLanes;
    begin_operation(stream);
    for (std::size_t first = 0u; first < batch.count; first += capacity) {
      const std::uint32_t count = static_cast<std::uint32_t>(
          std::min(capacity, batch.count - first));
      const std::uint32_t tiles =
          (count + kLookupTileRows - 1u) / kLookupTileRows;
      pack_lookup_tiles_kernel<<<tiles, kThreads, 0, stream>>>(
          batch.queries + first, local, inverse, offsets, count, tiles);
      execute_lookup_tiles_kernel<WithFound><<<dim3(
          (tiles + tiles_per_block - 1u) / tiles_per_block, kLookupTileBins),
          kThreads, 0, stream>>>(local, found, offsets, tiles, view);
      unpack_lookup_tiles_kernel<WithFound><<<tiles, kThreads, 0, stream>>>(
          local, found, inverse, batch.out_values + first,
          WithFound ? batch.out_found + first : nullptr, count);
    }
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

  struct LookupPlan {
    bool shared_pending;
    bool chunked;
  };

  LookupPlan plan_lookup(std::uint64_t count) const {
    const std::uint64_t dense_threshold =
        std::uint64_t{pending_batches_} * gpulsmopt2_detail::kQuotients *
        gpulsmopt2_detail::kTqrjDenseRowsPerSection;
    const bool dense =
        pending_batches_ >= gpulsmopt2_detail::kTqrjMinimumBatches &&
        std::uint64_t{pending_records_} >= dense_threshold;
    const bool shared = dense &&
        count >= gpulsmopt2_detail::kQuotients * 4u &&
        count <= tqrj_maximum_queries(batch_capacity_);
    return {shared,
            count > gpulsmopt2_detail::kMaximumOperationTile && !shared};
  }

  template <class LookupChunk>
  static void for_lookup_chunks(std::uint64_t count, LookupChunk lookup_chunk) {
    for (std::uint64_t first = 0u; first < count;
         first += gpulsmopt2_detail::kMaximumOperationTile) {
      lookup_chunk(first, std::min<std::uint64_t>(
          count - first, gpulsmopt2_detail::kMaximumOperationTile));
    }
  }

  struct PointLookupQueries {
    const std::uint32_t *keys;
    const std::uint32_t *ids;
  };

  PointLookupQueries prepare_point_lookup(
      const std::uint32_t *keys, std::uint32_t count,
      bool quotients_grouped, cudaStream_t stream) {
    if (count < gpulsmopt2_detail::kQuotients * 4u || quotients_grouped)
      return {keys, nullptr};
    const auto workspace = ensure_radix_workspace(count);
    gpulsmopt2_detail::count_quotients_kernel<false><<<
        blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
        keys, count, admission_counts_.data(), workspace.input_ids);
    std::size_t scan_bytes = admission_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        admission_temp_.data(), scan_bytes, admission_counts_.data(),
        workspace.quotient_offsets, gpulsmopt2_detail::kQuotients + 1u, stream));
    gpulsmopt2_detail::scatter_query_records_kernel<<<
        blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
        keys, count, workspace.quotient_offsets, workspace.input_ids,
        radix_keys_.data(), radix_ids_out_.data());
    CUDA_CHECK(cudaMemsetAsync(
        admission_counts_.data(), 0,
        admission_counts_.size() * sizeof(std::uint32_t), stream));
    return {radix_keys_.data(), radix_ids_out_.data()};
  }

  gpulsmopt2_detail::TqrjLookupWorkspace prepare_shared_lookup(
      const std::uint32_t *keys, std::uint32_t count, cudaStream_t stream) {
    // Order scratch growth with begin_operation first.
    ensure_publication_capacity(count, stream);
    if (operation_workspace_.size() < tqrj_hash_workspace_bytes(batch_capacity_) ||
        count > tqrj_maximum_queries(batch_capacity_))
      throw std::length_error("insufficient idle publication workspace for TQRJ");
    const std::uint32_t active_capacity = std::min(
        count, gpulsmopt2_detail::kQuotients);
    const std::uint32_t maximum_query_tiles =
        gpulsmopt2_detail::tqrj_hash_tile_count(count) + active_capacity;
    const std::uint32_t maximum_pending_tiles =
        gpulsmopt2_detail::tqrj_hash_tile_count(pending_records_) + active_capacity;
    if (maximum_query_tiles > tqrj_maximum_tiles(batch_capacity_) ||
        maximum_pending_tiles > tqrj_maximum_tiles(batch_capacity_))
      throw std::length_error("insufficient recycled tile workspace for TQRJ");
    auto workspace = borrow_tqrj_lookup_workspace(maximum_query_tiles);
    CUDA_CHECK(cudaMemsetAsync(
        workspace.active_quotient_count, 0, sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        workspace.hash_task_count, 0, sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        workspace.hash_counters, 0,
        gpulsmopt2_detail::kTqrjHashCounterBytes, stream));
    gpulsmopt2_detail::count_quotients_kernel<true><<<
        blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
        keys, count, admission_counts_.data(), workspace.reservation_ranks,
        workspace.active_quotients, workspace.active_quotient_count);
    gpulsmopt2_detail::materialize_lookup_active_counts_kernel<<<
        blocks(active_capacity + 1u), gpulsmopt2_detail::kThreads, 0, stream>>>(
        workspace.active_quotients, workspace.active_quotient_count,
        admission_counts_.data(), active_capacity, workspace.active_query_counts);
    std::size_t scan_bytes = admission_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        admission_temp_.data(), scan_bytes, workspace.active_query_counts,
        workspace.active_query_offsets, active_capacity + 1u, stream));
    gpulsmopt2_detail::publish_lookup_active_bases_kernel<<<
        blocks(active_capacity), gpulsmopt2_detail::kThreads, 0, stream>>>(
        workspace.active_quotients, workspace.active_quotient_count,
        workspace.active_query_offsets, active_capacity, workspace.query_bases);
    gpulsmopt2_detail::scatter_query_records_kernel<<<
        blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
        keys, count, workspace.query_bases, workspace.reservation_ranks,
        workspace.grouped_queries, workspace.query_ids);
    return workspace;
  }

  template <class T> struct LookupArgument { using type = T; };

  template <class... Arguments>
  static void launch_lookup_hash(void (*kernel)(Arguments...),
      std::uint32_t workers, cudaStream_t stream,
      typename LookupArgument<Arguments>::type... arguments) {
    // Deduce kernel argument types for compile-time checks.
    void *parameters[] = {static_cast<void *>(&arguments)...};
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void *>(kernel), dim3(workers),
        dim3(gpulsmopt2_detail::kTqrjHashThreads), parameters, 0u, stream));
  }

  void finish_shared_lookup(const gpulsmopt2_detail::TqrjLookupWorkspace &workspace,
                            std::uint32_t count, cudaStream_t stream) {
    const std::uint32_t active_capacity = std::min(
        count, gpulsmopt2_detail::kQuotients);
    gpulsmopt2_detail::reset_lookup_quotient_counts_kernel<<<
        blocks(active_capacity), gpulsmopt2_detail::kThreads, 0, stream>>>(
        workspace.active_quotients, workspace.active_quotient_count,
        active_capacity, admission_counts_.data());
  }

  bool use_tile_lookup(std::size_t count, bool quotients_grouped) const {
    if (quotients_grouped || pending_batches_ ||
        count < gpulsmopt2_detail::kMinimumTileLookupQueries ||
        !host_occupied_level_mask_ ||
        (host_occupied_level_mask_ & (host_occupied_level_mask_ - 1u)))
      return false;
    const auto level = static_cast<std::uint32_t>(
        __builtin_ctzll(host_occupied_level_mask_));
    return level_counts_[level] >=
        gpulsmopt2_detail::kMinimumTileLookupResidentRows;
  }


  void lookup_locked(const DeviceLookupBatch &batch, cudaStream_t stream,
                     bool quotients_grouped) {
    if (!batch.count) return;
    if (!batch.queries || !batch.out_values)
      throw std::invalid_argument("invalid GPULSMOpt lookup");
    // Choose tiles before dividing large lookup windows.
    if (use_tile_lookup(batch.count, quotients_grouped)) {
      if (batch.out_found) lookup_tiles_locked<true>(batch, stream);
      else lookup_tiles_locked<false>(batch, stream);
      return;
    }
    const LookupPlan plan = plan_lookup(batch.count);
    if (plan.chunked) {
      for_lookup_chunks(batch.count, [&](std::uint64_t first, std::uint64_t count) {
        lookup_locked(DeviceLookupBatch{
            batch.queries + first, static_cast<std::size_t>(count),
            batch.out_values + first,
            batch.out_found ? batch.out_found + first : nullptr},
            stream, quotients_grouped);
      });
      return;
    }
    const std::uint32_t count = static_cast<std::uint32_t>(batch.count);
    begin_operation(stream);
    if (plan.shared_pending) {
      using namespace gpulsmopt2_detail;
      const std::uint32_t active_capacity = std::min(count, kQuotients);
      auto workspace = prepare_shared_lookup(batch.queries, count, stream);
      const NativeTqrjAccess access{
          workspace.grouped_queries, batch.out_values, batch.out_found,
          raw_keys_.data(), raw_payloads_.data(),
          static_cast<std::uint32_t>(batch_capacity_), resident_rows(),
          descriptors_.data(), workspace.query_ids,
          canonical_cell_ranks_.data(), query_occupied_level_mask_.data()};
      const TqrjDirectLaunch direct{
          workspace.active_quotients, workspace.active_query_counts,
          workspace.active_quotient_count, workspace.query_bases,
          raw_offsets_.data(), pending_batches_, workspace.hash_tasks,
          workspace.hash_task_count};
      tqrj_direct_lookup_kernel<<<active_capacity, kThreads, 0, stream>>>(access, direct);
      const TqrjHashLaunch hash{
          workspace.hash_tasks, workspace.hash_task_count, workspace.query_bases,
          admission_counts_.data(), workspace.hash_query_tiles,
          workspace.hash_pending_tiles, workspace.hash_counters,
          workspace.hash_table, workspace.hash_table_capacity,
          workspace.hash_winners, raw_offsets_.data(), pending_batches_,
          batch.out_values};
      launch_lookup_hash(tqrj_hash_lookup_kernel, tqrj_hash_worker_blocks_,
                         stream, access, hash);
      finish_shared_lookup(workspace, count, stream);
    } else {
      const auto prepared = prepare_point_lookup(
          batch.queries, count, quotients_grouped, stream);
      gpulsmopt2_detail::canonical_lookup_with_pending_kernel<<<
          blocks(count), gpulsmopt2_detail::kThreads, 0, stream>>>(
          prepared.keys, batch.out_values, batch.out_found, count,
          raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
          static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
          raw_signatures_.data(), raw_epoch_signatures_.data(), resident_rows(),
          descriptors_.data(), canonical_cell_ranks_.data(), prepared.ids,
          query_occupied_level_mask_.data());
    }
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

  void range_locked(const DeviceRangeOutputBatch &batch,
                    cudaStream_t stream) {
    if (!batch.query_count) return;
    if (!batch.lo || !batch.hi || !batch.out_sums)
      throw std::invalid_argument("invalid GPULSMOpt range input");
    if (batch.query_count > gpulsmopt2_detail::kMaximumOperationTile) {
      for (std::size_t begin = 0u; begin < batch.query_count;
           begin += gpulsmopt2_detail::kMaximumOperationTile) {
        const std::size_t count = std::min(
            batch.query_count - begin,
            gpulsmopt2_detail::kMaximumOperationTile);
        range_locked(DeviceRangeOutputBatch{
            batch.lo + begin, batch.hi + begin, count,
            batch.out_sums + begin}, stream);
      }
      return;
    }
    begin_operation(stream);
    const std::uint32_t query_count =
        static_cast<std::uint32_t>(batch.query_count);
    const bool may_have_crowded_newer = pending_batches_ != 0u ||
        (host_occupied_level_mask_ & (host_occupied_level_mask_ - 1u)) != 0u;
    if (may_have_crowded_newer) {
      std::size_t hot_scan_bytes{};
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          nullptr, hot_scan_bytes, range_hot_counts_.data(),
          range_hot_offsets_.data(),
          gpulsmopt2_detail::kQuotients + 1u, stream));
      if (range_hot_temp_.size() < hot_scan_bytes)
        range_hot_temp_.resize(hot_scan_bytes);
      CUDA_CHECK(cudaMemsetAsync(
          range_hot_descriptors_.data(), 0,
          range_hot_descriptors_.size() *
              sizeof(gpulsmopt2_detail::Descriptor), stream));
      gpulsmopt2_detail::count_range_hot_newer_rows_kernel<<<
          blocks(gpulsmopt2_detail::kQuotients + 1u),
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              raw_offsets_.data(), pending_batches_, descriptors_.data(),
              query_occupied_level_mask_.data(), range_hot_counts_.data());
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          range_hot_temp_.data(), hot_scan_bytes, range_hot_counts_.data(),
          range_hot_offsets_.data(),
          gpulsmopt2_detail::kQuotients + 1u, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          range_hot_total_receipt(),
          range_hot_offsets_.data() + gpulsmopt2_detail::kQuotients,
          sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream));
    }
    const bool needs_wide_total = query_count >
        std::numeric_limits<std::uint32_t>::max() /
            gpulsmopt2_detail::kQuotients;
    std::size_t scan_bytes{};
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, range_fragment_counts_.data(),
        range_fragment_offsets_.data(), query_count + 1u, stream));
    std::size_t reduce_bytes{};
    if (needs_wide_total) {
      using WideCountIterator = cub::TransformInputIterator<
          std::uint64_t, gpulsmopt2_detail::WidenFragmentCount,
          const std::uint32_t *>;
      const WideCountIterator counts(
          range_fragment_counts_.data(),
          gpulsmopt2_detail::WidenFragmentCount{});
      CUDA_CHECK(cub::DeviceReduce::Sum(
          nullptr, reduce_bytes, counts, range_fragment_total_.data(),
          query_count, stream));
    }
    ensure_range_fragment_query_capacity(
        query_count, std::max(scan_bytes, reduce_bytes));
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
      if (may_have_crowded_newer) CUDA_CHECK(cudaStreamSynchronize(stream));
    } else if (needs_wide_total) {
      using WideCountIterator = cub::TransformInputIterator<
          std::uint64_t, gpulsmopt2_detail::WidenFragmentCount,
          const std::uint32_t *>;
      const WideCountIterator counts(
          range_fragment_counts_.data(),
          gpulsmopt2_detail::WidenFragmentCount{});
      CUDA_CHECK(cub::DeviceReduce::Sum(
          range_query_temp_, reduce_bytes, counts,
          range_fragment_total_.data(), query_count, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          range_total_receipt(), range_fragment_total_.data(),
          sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const std::uint64_t total = range_total_receipt()[0];
      if (total > std::numeric_limits<std::uint32_t>::max()) {
        // Reject fragment totals above 32 bits.
        end_operation(stream);
        throw std::length_error(
            "GPULSMOpt range produces more than 2^32-1 fragments");
      }
      fragment_count = static_cast<std::uint32_t>(total);
    } else {
      CUDA_CHECK(cudaMemcpyAsync(
          &fragment_count, range_fragment_offsets_.data() + query_count,
          sizeof(fragment_count), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const std::uint64_t hot_total = may_have_crowded_newer
        ? range_hot_total_receipt()[0] : 0u;
    if (hot_total) materialize_range_hot_sections(hot_total, stream);
    const bool hot_ready = hot_total != 0u;
    if (!fragment_count) {
      CUDA_CHECK(cudaMemsetAsync(batch.out_sums, 0,
                                 std::size_t{query_count} *
                                     sizeof(std::uint32_t),
                                 stream));
      end_operation(stream);
      return;
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
      launch_section_ranges(stream, hot_ready);
    } else {
      launch_fragment_ranges(
          fragment_count, query_count, batch, stream, hot_ready);
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

  void successor_locked(const DeviceSuccessorBatch &batch,
                        cudaStream_t stream) {
    if (!batch.count) return;
    if (!batch.queries || !batch.out_keys)
      throw std::invalid_argument("invalid GPULSMOpt successor input");
    if (batch.count > std::numeric_limits<std::uint32_t>::max())
      throw std::length_error("GPULSMOpt successor batch is too large");
    begin_operation(stream);

    const bool prepare_pending = pending_records_ &&
        batch.count >= gpulsmopt2_detail::kSuccessorSharedPendingQueries;
    const gpulsmopt2_detail::Row *prepared_pending_rows = nullptr;
    const std::uint32_t *prepared_pending_offsets = nullptr;
    const std::uint32_t *prepared_pending_counts = nullptr;
    if (prepare_pending) {
      ensure_publication_capacity(pending_records_, stream);
      launch_canonical_epoch_resolution(
          stream, false, 0u, nullptr, false);
      prepared_pending_rows = publication_rows_a_.data();
      prepared_pending_offsets = foundation_source_offsets_.data();
      prepared_pending_counts = foundation_section_output_counts_.data();
    }

    constexpr std::size_t region_words =
        gpulsmopt2_detail::kQuotients + 1u;
    constexpr std::size_t chunk_words =
        gpulsmopt2_detail::kSuccessorRegionChunks;
    constexpr std::size_t required_words = region_words + 2u * chunk_words;
    if (operation_workspace_.size() < required_words * sizeof(std::uint32_t))
      throw std::length_error("insufficient GPULSMOpt successor workspace");
    auto *next_regions = reinterpret_cast<std::uint32_t *>(
        operation_workspace_.data());
    auto *chunk_first = next_regions + region_words;
    auto *chunk_carry = chunk_first + chunk_words;
    gpulsmopt2_detail::build_successor_region_index_kernel<<<
        gpulsmopt2_detail::kSuccessorRegionChunks,
        gpulsmopt2_detail::kSuccessorRegionChunk, 0, stream>>>(
            raw_offsets_.data(), pending_batches_, descriptors_.data(),
            query_occupied_level_mask_.data(), next_regions, chunk_first);
    gpulsmopt2_detail::build_successor_chunk_carry_kernel<<<1, 1, 0, stream>>>(
        chunk_first, chunk_carry, next_regions);
    gpulsmopt2_detail::finish_successor_region_index_kernel<<<
        gpulsmopt2_detail::kSuccessorRegionChunks,
        gpulsmopt2_detail::kSuccessorRegionChunk, 0, stream>>>(
            next_regions, chunk_carry);
    gpulsmopt2_detail::successor_with_pending_kernel<<<
        blocks(batch.count), gpulsmopt2_detail::kThreads, 0, stream>>>(
        batch.queries, static_cast<std::uint32_t>(batch.count), batch.out_keys,
        raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
        static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
        prepared_pending_rows, prepared_pending_offsets,
        prepared_pending_counts, resident_rows(), descriptors_.data(),
        canonical_cell_ranks_.data(), next_regions,
        query_occupied_level_mask_.data());
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
  }

public:

  std::size_t gpu_resident_bytes() const {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    const_cast<GPULSMOpt *>(this)->resolve_publication_receipt();
    const std::size_t rollover_rank_bytes = canonical_rollover_epoch_ranks_
        ? canonical_rollover_epoch_ranks_->size() * sizeof(std::uint16_t)
        : 0u;
    const std::uint64_t retained = rollover_rank_bytes +
        arena_key_flags_.size() * sizeof(std::uint32_t) +
        arena_values_.size() * sizeof(std::uint32_t) +
        static_workspace_.size() + phase_workspace_.size() +
        (sealed_workspace_ ? sealed_workspace_->bytes() : 0u) +
        publication_rows_a_.size() * sizeof(gpulsmopt2_detail::Row) +
        publication_keys_a_.size() * sizeof(std::uint32_t) +
        resident_scan_temp_.size() * sizeof(std::uint8_t) +
        admission_temp_.size() * sizeof(std::uint8_t) +
        (range_hot_tokens_a_.size() + range_hot_tokens_b_.size()) *
            sizeof(std::uint64_t) +
        range_hot_temp_.size() * sizeof(std::uint8_t) +
        range_query_storage_.size() + range_fragment_storage_.size() +
        range_section_storage_.size();
    if (retained > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("GPULSMOpt memory accounting overflow");
    return static_cast<std::size_t>(retained);
  }

private:
  struct DeviceCapabilities {
    int multiprocessors{};
    bool cooperative_launch{};
    std::size_t shared_memory{};
    std::size_t optin_shared_memory{};
  };

  static DeviceCapabilities query_device_capabilities() {
    int device = 0;
    int multiprocessors = 0;
    int cooperative_launch = 0;
    int shared_memory = 0;
    int optin_shared_memory = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &multiprocessors, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &cooperative_launch, cudaDevAttrCooperativeLaunch, device));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &shared_memory, cudaDevAttrMaxSharedMemoryPerBlock, device));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &optin_shared_memory,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (multiprocessors <= 0 || shared_memory <= 0)
      throw std::runtime_error("invalid CUDA device capabilities");
    return {multiprocessors, cooperative_launch != 0,
            static_cast<std::size_t>(shared_memory),
            static_cast<std::size_t>(optin_shared_memory)};
  }

  gpulsmopt2_detail::ResidentRows resident_rows() {
    return {arena_key_flags_.data(), arena_values_.data()};
  }

  static int blocks(std::size_t count) {
    return static_cast<int>((count + gpulsmopt2_detail::kThreads - 1u) /
                            gpulsmopt2_detail::kThreads);
  }


  std::uint32_t initial_level_for_records(std::size_t count) const {
    for (std::uint32_t tier_begin = 0u;
         tier_begin < canonical_regular_level_count_; tier_begin += 3u) {
      const std::uint32_t slots = std::min(
          3u, canonical_regular_level_count_ - tier_begin);
      if (count <= level_storage_spans_host_[tier_begin].capacity)
        return tier_begin + slots - 1u;
    }
    if (canonical_regular_level_count_ < canonical_level_count_ &&
        count <= level_storage_spans_host_[canonical_regular_level_count_]
                     .capacity)
      return canonical_regular_level_count_;
    throw std::bad_alloc();
  }

  std::uint64_t level_begin(std::uint32_t target) const {
    if (target >= canonical_level_count_) throw std::out_of_range(
        "GPULSMOpt physical run slot is out of range");
    return level_storage_spans_host_[target].begin;
  }

  std::uint64_t level_capacity(std::uint32_t target) const {
    if (target >= canonical_level_count_) throw std::out_of_range(
        "GPULSMOpt physical run slot is out of range");
    return level_storage_spans_host_[target].capacity;
  }

  void ensure_level_storage_mapped(
      std::uint32_t level, cudaStream_t stream) {
    const std::uint64_t required = level_begin(level) + level_capacity(level);
    if (required <= arena_key_flags_.size() &&
        required <= arena_values_.size())
      return;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    arena_key_flags_.grow(required);
    arena_values_.grow(required);
  }

  void ensure_canonical_top_rollover_bank(cudaStream_t stream) {
    if (!canonical_level_count_)
      throw std::logic_error("GPULSMOpt has no canonical levels");
    const std::uint32_t top = canonical_level_count_ - 1u;
    const std::uint64_t required = level_begin(top) +
        2u * level_capacity(top);
    const bool rows_ready = required <= arena_key_flags_.size() &&
        required <= arena_values_.size();
    if (rows_ready && canonical_rollover_epoch_ranks_)
      return;

    // Map the rollover bank on first use.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!rows_ready) {
      arena_key_flags_.grow(required);
      arena_values_.grow(required);
    }
    if (!canonical_rollover_epoch_ranks_)
      canonical_rollover_epoch_ranks_ = std::make_unique<
          gpulsmopt2_detail::Buffer<std::uint16_t>>(
              gpulsmopt2_detail::kLocalRankEntries);
  }

  std::uint32_t foundation_level() const {
    return host_occupied_level_mask_
        ? 63u - static_cast<std::uint32_t>(
                    __builtin_clzll(host_occupied_level_mask_))
        : gpulsmopt2_detail::kMaximumLevels;
  }

  void initialize_resident_workspace(
      const DeviceCapabilities &capabilities) {
    std::array<gpulsmopt2_detail::LevelStorageSpan,
               gpulsmopt2_detail::kMaximumLevels> spans{};
    std::uint64_t cursor = 0u;
    std::size_t capacity = level_zero_capacity_;
    for (std::uint32_t level = 0u;
         level < canonical_regular_level_count_; ++level) {
      spans[level] = {cursor, capacity};
      cursor += capacity;
      if ((level + 1u) % 3u == 0u)
        capacity = capacity > publication_capacity_ / 4u
            ? publication_capacity_ : capacity * 4u;
    }
    if (cursor != level_pool_capacity_)
      throw std::logic_error("GPULSMOpt preassigned level spans overflow");
    if (canonical_regular_level_count_ < canonical_level_count_)
      spans[canonical_regular_level_count_] = {
          cursor, publication_capacity_};
    level_storage_spans_host_ = spans;
    CUDA_CHECK(cudaMemcpy(
        level_storage_spans_.data(), spans.data(), sizeof(spans),
        cudaMemcpyHostToDevice));

    std::size_t maximum_scan_bytes = 0u, bytes = 0u;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, bytes, resident_tile_job_counts_.data(),
        resident_tile_job_offsets_.data(),
        gpulsmopt2_detail::kPlanningTiles + 1u, 0));
    maximum_scan_bytes = std::max(maximum_scan_bytes, bytes);
    bytes = 0u;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, bytes, foundation_section_output_counts_.data(),
        foundation_source_offsets_.data(),
        gpulsmopt2_detail::kQuotients + 1u, 0));
    maximum_scan_bytes = std::max(maximum_scan_bytes, bytes);
    resident_scan_temp_.resize(maximum_scan_bytes);

    if (!capabilities.cooperative_launch)
      throw std::runtime_error(
          "GPULSMOpt TQRJ requires cooperative kernel launch support");
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, gpulsmopt2_detail::tqrj_hash_lookup_kernel,
        gpulsmopt2_detail::kTqrjHashThreads, 0u));
    // Size the hash grid from device occupancy.
    tqrj_hash_worker_blocks_ = static_cast<std::uint32_t>(
        std::max(1, std::min(4, blocks_per_sm)) *
        capabilities.multiprocessors);
    blocks_per_sm = 0;
    resident_planner_blocks_ = static_cast<std::uint32_t>(std::max<std::size_t>(
        1u, std::min<std::size_t>(maximum_resident_jobs_,
            static_cast<std::size_t>(capabilities.multiprocessors) * 4u)));
    blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        gpulsmopt2_detail::cooperative_section_owned_range_kernel<
            gpulsmopt2_detail::RangeSumSink>,
        gpulsmopt2_detail::kSectionRangeThreads, 0u));
    range_section_blocks_ = static_cast<std::uint32_t>(
        std::max(1, blocks_per_sm) * capabilities.multiprocessors);
  }

  void initialize_canonical_workspace(
      const DeviceCapabilities &capabilities) {
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        gpulsmopt2_detail::resolve_canonical_epoch_oversized_kernel<false>,
        gpulsmopt2_detail::kFoundationCompactionThreads, 0u));
    canonical_epoch_resolver_blocks_ = static_cast<std::uint32_t>(
        std::max(1, blocks_per_sm) * capabilities.multiprocessors);
    canonical_epoch_workspace_slots_ = static_cast<std::uint32_t>(
        canonical_epoch_workspace_.size() /
        gpulsmopt2_detail::kCanonicalResolverSuffixes);
    canonical_epoch_workspace_slots_ =
        std::max(1u, canonical_epoch_workspace_slots_);

    const std::uint32_t maximum_sources = std::min(
        gpulsmopt2_detail::kMaximumMergeSources,
        std::max(2u, canonical_level_count_ + 1u));
    cudaFuncAttributes tournament_attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(
        &tournament_attributes,
        gpulsmopt2_detail::canonical_tournament_carry_jobs_kernel));
    const std::size_t optin_shared_bytes = std::max<std::size_t>(
        capabilities.shared_memory,
        capabilities.optin_shared_memory);
    const std::size_t maximum_dynamic_shared_bytes =
        optin_shared_bytes > tournament_attributes.sharedSizeBytes
            ? optin_shared_bytes - tournament_attributes.sharedSizeBytes
            : 0u;
    canonical_job_capacities_.fill(0u);
    canonical_tournament_shared_bytes_.fill(0u);
    canonical_tournament_blocks_.fill(0u);
    std::size_t tournament_attribute_bytes = 0u;
    for (std::uint32_t source_count =
             gpulsmopt2_detail::kCanonicalTournamentMinimumSources;
         source_count <= maximum_sources; ++source_count) {
      const auto active_blocks = [&](std::uint32_t capacity) {
        const std::size_t shared_bytes =
            gpulsmopt2_detail::canonical_tournament_workspace_bytes(
                capacity, source_count);
        if (shared_bytes > maximum_dynamic_shared_bytes) return 0;
        int blocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks,
            gpulsmopt2_detail::canonical_tournament_carry_jobs_kernel,
            gpulsmopt2_detail::kFoundationCompactionThreads,
            shared_bytes));
        return blocks;
      };
      const int desired_blocks =
          active_blocks(resident_merge_capacity_);
      if (!desired_blocks)
        throw std::runtime_error(
            "unsupported GPULSMOpt tournament source shape");
      std::uint32_t low = resident_merge_capacity_;
      std::uint32_t high =
          gpulsmopt2_detail::kCanonicalTournamentCapacityCeiling;
      while (low < high) {
        const std::uint32_t middle =
            low + (high - low + 1u) / 2u;
        if (active_blocks(middle) >= desired_blocks)
          low = middle;
        else
          high = middle - 1u;
      }
      const std::size_t shared_bytes =
          gpulsmopt2_detail::canonical_tournament_workspace_bytes(
              low, source_count);
      const std::uint32_t capacity =
          gpulsmopt2_detail::canonical_tournament_capacity(
              shared_bytes, source_count, 1u);
      if (!capacity)
        throw std::logic_error(
            "GPULSMOpt tournament has no job capacity");
      canonical_job_capacities_[source_count] = capacity;
      canonical_tournament_shared_bytes_[source_count] =
          shared_bytes;
      tournament_attribute_bytes =
          std::max(tournament_attribute_bytes, shared_bytes);
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        gpulsmopt2_detail::canonical_tournament_carry_jobs_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(tournament_attribute_bytes)));
    for (std::uint32_t source_count =
             gpulsmopt2_detail::kCanonicalTournamentMinimumSources;
         source_count <= maximum_sources; ++source_count) {
      const std::size_t shared_bytes =
          canonical_tournament_shared_bytes_[source_count];
      blocks_per_sm = 0;
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm,
          gpulsmopt2_detail::canonical_tournament_carry_jobs_kernel,
          gpulsmopt2_detail::kFoundationCompactionThreads,
          shared_bytes));
      if (!blocks_per_sm)
        throw std::runtime_error(
            "GPULSMOpt tournament cannot become resident");
      canonical_tournament_blocks_[source_count] =
          static_cast<std::uint32_t>(
              blocks_per_sm * capabilities.multiprocessors);
    }
    CUDA_CHECK(cudaMemset(
        canonical_cell_counts_.data(), 0,
        canonical_cell_counts_.size() * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(
        canonical_job_prefixes_.data(), 0,
        canonical_job_prefixes_.size() *
            sizeof(gpulsmopt2_detail::CanonicalJobPrefix)));
    CUDA_CHECK(cudaMemset(
        canonical_next_job_.data(), 0, sizeof(std::uint32_t)));
  }
  gpulsmopt2_detail::PinnedReceipts &pinned_receipts() {
    return *pinned_receipts_.data();
  }

  gpulsmopt2_detail::ResidentPublicationPlan *publication_receipt() {
    return &pinned_receipts().publication;
  }

  std::uint64_t *range_total_receipt() {
    return &pinned_receipts().range_total;
  }

  std::uint64_t *range_hot_total_receipt() {
    return &pinned_receipts().range_hot_total;
  }

  std::uint64_t *range_hot_offsets_receipt() {
    return pinned_receipts().range_hot_offsets;
  }

  void ensure_publication_capacity(std::size_t count,
                                   cudaStream_t stream) {
    const std::size_t current =
        std::min(publication_keys_a_.size(), publication_rows_a_.size());
    if (count <= current) return;
    if (count > gpulsmopt2_detail::kMaximumPublicationRows)
      throw std::bad_alloc();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    publication_keys_a_.grow(count);
    publication_rows_a_.grow(count);
  }

  void reset_pending_batch_counts() {
    pending_batches_ = 0u;
    pending_records_ = 0u;
    pending_has_tombstones_ = false;
    std::fill_n(raw_batch_counts_, gpulsmopt2_detail::kBatchesPerEpoch, 0u);
  }

  void apply_publication_receipt() {
    if (!publication_.consume_receipt()) return;
    const gpulsmopt2_detail::ResidentPublicationPlan &receipt =
        publication_receipt()[0];
    if (receipt.status != gpulsmopt2_detail::kPublicationSuccess) {
      // Keep raw batches when publication fails.
      publication_.fail(receipt);
      return;
    }

    const std::uint32_t destination = receipt.destination_level;
    if (destination >= canonical_level_count_) {
      auto failure = receipt;
      failure.status |= gpulsmopt2_detail::kPublicationLevelOverflow;
      publication_.fail(failure);
      return;
    }
    const std::uint64_t consumed = destination
        ? ((std::uint64_t{1} << destination) - 1u) : 0u;
    host_occupied_level_mask_ &= ~consumed;
    for (std::uint32_t level = 0u; level < destination; ++level)
      level_counts_[level] = 0u;
    level_counts_[destination] =
        static_cast<std::uint32_t>(receipt.survivor_count);
    if (receipt.survivor_count)
      host_occupied_level_mask_ |= std::uint64_t{1} << destination;
    else
      host_occupied_level_mask_ &= ~(std::uint64_t{1} << destination);
    reset_pending_batch_counts();
    publication_.reset();
  }

  void resolve_publication_receipt() {
    if (!publication_.receipt_pending()) return;
    // Resolve the asynchronous publication receipt.
    const cudaError_t ready = cudaEventQuery(operation_done_);
    if (ready == cudaErrorNotReady)
      CUDA_CHECK(cudaEventSynchronize(operation_done_));
    else
      CUDA_CHECK(ready);
    apply_publication_receipt();
  }

  void resolve_publication_receipt_on_stream(cudaStream_t stream) {
    if (!publication_.receipt_pending()) return;
    // Synchronize between tiles of one large update.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    apply_publication_receipt();
  }

  void reject_updates_after_publication_failure() const {
    if (!publication_.failed()) return;
    const auto &receipt = publication_.failure();
    const std::string reason =
        receipt.status &
                gpulsmopt2_detail::kPublicationLevelOverflow
            ? "canonical carry capacity exhausted"
            : "publication failed";
    throw std::runtime_error(
        "GPULSMOpt " + reason + " with status " +
        std::to_string(receipt.status) +
        "; destination=" + std::to_string(receipt.destination_level) +
        ", selected=" + std::to_string(receipt.selected_count) +
        ", reservation=" + std::to_string(receipt.raw_reservation) +
        ", capacity=" + std::to_string(receipt.output_capacity) +
        ", survivors=" + std::to_string(receipt.survivor_count) +
        ", jobs=" + std::to_string(receipt.job_count) +
        "; pending updates were preserved");
  }

  void prepare_failed_epoch_for_reads(cudaStream_t stream) {
    if (!publication_.needs_signature_rebuild()) return;
    begin_operation(stream);
    gpulsmopt2_detail::rebuild_epoch_signatures_kernel<<<
        gpulsmopt2_detail::kQuotients / gpulsmopt2_detail::kThreads,
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            raw_signatures_.data(), raw_epoch_signatures_.data());
    CUDA_CHECK(cudaGetLastError());
    end_operation(stream);
    publication_.signatures_queued();
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
    CUDA_CHECK(cudaMemsetAsync(raw_offsets_.data(), 0,
                               raw_offsets_.size() * sizeof(std::uint32_t),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(
        raw_epoch_signatures_.data(), 0,
        raw_epoch_signatures_.size() * sizeof(std::uint64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        admission_counts_.data(), 0,
        admission_counts_.size() * sizeof(std::uint32_t), stream));
    reset_pending_batch_counts();
    publication_.reset();
    host_occupied_level_mask_ = 0u;
    std::fill_n(level_counts_, gpulsmopt2_detail::kMaximumLevels, 0u);
    gpulsmopt2_detail::initialize_device_manifest_kernel<<<1, 1, 0, stream>>>(
        device_manifests_.data(), active_device_manifest_.data(),
        query_occupied_level_mask_.data(), 0u, 0u, 0u);
  }

  std::vector<std::uint32_t> sealed_tier_slots() const {
    std::vector<std::uint32_t> result;
    std::uint32_t remaining = canonical_regular_level_count_;
    while (remaining) {
      const std::uint32_t slots = std::min(3u, remaining);
      result.push_back(slots);
      remaining -= slots;
    }
    if (canonical_regular_level_count_ < canonical_level_count_)
      result.push_back(1u);
    return result;
  }

  static std::size_t sealed_workspace_capacity(
      std::size_t publication_capacity,
      std::size_t level_zero_capacity) {
    return gpulsmopt2_detail::canonical_level_layout(
        publication_capacity, level_zero_capacity)
        .highest_regular_capacity;
  }

  static std::size_t align_phase_workspace(std::size_t bytes,
                                           std::size_t alignment = 256u) {
    return (bytes + alignment - 1u) / alignment * alignment;
  }

  class StaticWorkspaceBuilder {
   public:
    explicit StaticWorkspaceBuilder(std::uint8_t *storage = nullptr)
        : storage_(storage) {}

    template <class T>
    void append(gpulsmopt2_detail::Buffer<T> &view, std::size_t count) {
      append_view<T>(view, count);
    }

    template <class T>
    void append(gpulsm_sparse::Buffer<T> &view, std::size_t count) {
      append_view<T>(view, count);
    }

    std::size_t bytes() const {
      return align_phase_workspace(cursor_);
    }

   private:
    template <class T, class BufferType>
    void append_view(BufferType &view, std::size_t count) {
      cursor_ = align_phase_workspace(cursor_);
      if (count > (std::numeric_limits<std::size_t>::max() - cursor_) /
                      sizeof(T))
        throw std::length_error("static workspace capacity");
      if (storage_)
        view.attach(reinterpret_cast<T *>(storage_ + cursor_), count);
      cursor_ += count * sizeof(T);
    }

    std::uint8_t *storage_{};
    std::size_t cursor_{};
  };

  void initialize_static_workspace_views() {
    const auto append_views = [&](StaticWorkspaceBuilder &layout) {
      layout.append(
          descriptors_, std::size_t{gpulsmopt2_detail::kQuotients} *
              gpulsmopt2_detail::kMaximumLevels);
      layout.append(device_manifests_, 2u);
      layout.append(active_device_manifest_, 1u);
      layout.append(query_occupied_level_mask_, 1u);
      layout.append(sealed_device_command_, 1u);
      layout.append(sealed_device_receipt_, 1u);
      layout.append(resident_plan_, 1u);
      layout.append(
          level_storage_spans_, gpulsmopt2_detail::kMaximumLevels);
      layout.append(
          canonical_cell_ranks_,
          std::size_t{canonical_level_count_} *
              gpulsmopt2_detail::kLocalRankEntries);
      layout.append(canonical_job_prefixes_, maximum_resident_jobs_);
      layout.append(canonical_next_job_, 1u);
      layout.append(
          raw_payloads_,
          std::size_t{gpulsmopt2_detail::kBatchesPerEpoch} *
              batch_capacity_);
      layout.append(
          raw_offsets_,
          std::size_t{gpulsmopt2_detail::kBatchesPerEpoch} *
              (gpulsmopt2_detail::kQuotients + 1u));
      layout.append(
          raw_signatures_,
          std::size_t{gpulsmopt2_detail::kBatchesPerEpoch} *
              gpulsmopt2_detail::kQuotients);
      layout.append(
          raw_epoch_signatures_, gpulsmopt2_detail::kQuotients);
      layout.append(publication_selected_count_, 1u);
      layout.append(
          foundation_source_offsets_,
          gpulsmopt2_detail::kQuotients + 1u);
      layout.append(
          foundation_section_output_counts_,
          gpulsmopt2_detail::kQuotients + 1u);
      layout.append(
          balanced_merge_raw_counts_, gpulsmopt2_detail::kQuotients);
      layout.append(
          resident_tile_job_counts_,
          gpulsmopt2_detail::kPlanningTiles + 1u);
      layout.append(
          resident_tile_job_offsets_,
          gpulsmopt2_detail::kPlanningTiles + 1u);
      layout.append(
          resident_job_raw_reservations_, maximum_resident_jobs_ + 1u);
      layout.append(balanced_merge_jobs_, maximum_resident_jobs_);
      layout.append(local_epoch_overflow_flag_, 1u);
      layout.append(
          admission_counts_, gpulsmopt2_detail::kQuotients + 1u);
      layout.append(
          range_partials_, gpulsmopt2_detail::kRangeSchedulerBlocks);
      layout.append(range_reduction_completion_, 1u);
      layout.append(range_fragment_total_, 1u);
      layout.append(
          range_hot_counts_, gpulsmopt2_detail::kQuotients + 1u);
      layout.append(
          range_hot_offsets_, gpulsmopt2_detail::kQuotients + 1u);
      layout.append(
          range_hot_window_offsets_,
          gpulsmopt2_detail::kQuotients + 1u);
      layout.append(
          range_hot_descriptors_, gpulsmopt2_detail::kQuotients);
      layout.append(range_hot_selected_count_, 1u);
    };

    StaticWorkspaceBuilder measurement;
    append_views(measurement);
    static_workspace_.resize(measurement.bytes());
    StaticWorkspaceBuilder attachment(static_workspace_.data());
    append_views(attachment);
    if (attachment.bytes() != static_workspace_.size())
      throw std::logic_error("static workspace layout mismatch");
  }

  static std::size_t phase_operation_workspace_offset(std::size_t batch_capacity) {
    return align_phase_workspace(
        gpulsmopt2_detail::kBatchesPerEpoch * batch_capacity * sizeof(std::uint32_t));
  }

  static std::size_t phase_workspace_initial_bytes(std::size_t batch_capacity) {
    return phase_operation_workspace_offset(batch_capacity) +
        operation_workspace_initial_bytes(batch_capacity);
  }

  static std::size_t phase_workspace_maximum_bytes(
      std::size_t publication_capacity, std::size_t batch_capacity) {
    return std::max(
        phase_operation_workspace_offset(batch_capacity) +
            operation_workspace_maximum_bytes(batch_capacity),
        gpulsm_sparse::NativeRootWorkspace<false>::required_bytes(publication_capacity));
  }

  void initialize_phase_workspace_views() {
    std::uint8_t *storage = phase_workspace_.data();
    raw_keys_.attach(reinterpret_cast<std::uint32_t *>(storage),
        gpulsmopt2_detail::kBatchesPerEpoch * batch_capacity_);
    operation_workspace_.attach(
        storage + phase_operation_workspace_offset(batch_capacity_),
        operation_workspace_maximum_bytes(batch_capacity_),
        operation_workspace_initial_bytes(batch_capacity_));
  }

  // Share resolved input for epoch and sealed publication.
  struct PublicationInput {
    const gpulsmopt2_detail::Row *rows;
    const std::uint32_t *offsets;
    const std::uint32_t *counts;
    const std::uint16_t *ranks;
  };

  template <bool SeparateCounts = false>
  void finalize_publication_metadata(
      const std::uint32_t *section_offsets, cudaStream_t stream,
      const std::uint32_t *section_counts = nullptr,
      const std::uint32_t *selected_count = nullptr) {
    gpulsmopt2_detail::finalize_canonical_metadata_kernel<SeparateCounts><<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
        section_offsets, section_counts, selected_count,
        resident_plan_.data(), descriptors_.data());
  }

  void finalize_direct_root(
      const gpulsmopt2_detail::ResidentPublicationPlan &plan,
      const std::uint32_t *section_offsets, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(
        resident_plan_.data(), &plan, sizeof(plan),
        cudaMemcpyHostToDevice, stream));
    gpulsmopt2_detail::build_canonical_rank_from_run_kernel<<<
        gpulsmopt2_detail::kQuotients,
        gpulsmopt2_detail::kFoundationCells, 0, stream>>>(
        resident_rows(), plan.output_begin, section_offsets,
        plan.destination_level, canonical_cell_ranks_.data());
    finalize_publication_metadata(section_offsets, stream);
  }

  void validate_publication_sources(std::uint32_t source_count) const {
    if (source_count >= canonical_job_capacities_.size() ||
        !canonical_job_capacities_[source_count] ||
        !canonical_tournament_shared_bytes_[source_count] ||
        !canonical_tournament_blocks_[source_count])
      throw std::logic_error(
          "unsupported publication tournament source shape");
  }

  void launch_publication_merge(PublicationInput input,
      std::uint32_t source_count, cudaStream_t stream) {
    // Consume input before reusing its scratch for output.
    CUDA_CHECK(cudaMemsetAsync(
        canonical_cell_counts_.data(), 0,
        canonical_cell_counts_.size() * sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        canonical_job_prefixes_.data(), 0,
        canonical_job_prefixes_.size() *
            sizeof(gpulsmopt2_detail::CanonicalJobPrefix),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        canonical_next_job_.data(), 0, sizeof(std::uint32_t), stream));
    gpulsmopt2_detail::count_canonical_merge_work_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
        input.counts, descriptors_.data(),
        device_manifests_.data(), active_device_manifest_.data(),
        resident_plan_.data(), balanced_merge_raw_counts_.data());
    gpulsmopt2_detail::count_canonical_planning_jobs_kernel<<<
        gpulsmopt2_detail::kPlanningTiles,
        gpulsmopt2_detail::kThreads, 0, stream>>>(
        balanced_merge_raw_counts_.data(), resident_plan_.data(),
        resident_tile_job_counts_.data());
    std::size_t scan_bytes = resident_scan_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        resident_scan_temp_.data(), scan_bytes,
        resident_tile_job_counts_.data(),
        resident_tile_job_offsets_.data(),
        gpulsmopt2_detail::kPlanningTiles + 1u, stream));
    gpulsmopt2_detail::emit_canonical_planning_jobs_kernel<<<
        gpulsmopt2_detail::kPlanningTiles,
        gpulsmopt2_detail::kThreads, 0, stream>>>(
        balanced_merge_raw_counts_.data(),
        resident_tile_job_offsets_.data(), resident_plan_.data(),
        static_cast<std::uint32_t>(maximum_resident_jobs_),
        balanced_merge_jobs_.data(), resident_job_raw_reservations_.data());
    gpulsmopt2_detail::validate_canonical_plan_kernel<<<1, 1, 0, stream>>>(
        resident_plan_.data(), resident_tile_job_offsets_.data(),
        static_cast<std::uint32_t>(maximum_resident_jobs_));
    gpulsmopt2_detail::resolve_canonical_job_boundaries_kernel<<<
        resident_planner_blocks_, 32u, 0, stream>>>(
        balanced_merge_raw_counts_.data(), balanced_merge_jobs_.data(),
        resident_job_raw_reservations_.data(), resident_plan_.data(),
        input.rows, input.offsets, input.counts, input.ranks,
        resident_rows(), descriptors_.data(), canonical_cell_ranks_.data(),
        device_manifests_.data(), active_device_manifest_.data());
    gpulsmopt2_detail::canonical_tournament_carry_jobs_kernel<<<
        canonical_tournament_blocks_[source_count],
        gpulsmopt2_detail::kFoundationCompactionThreads,
        canonical_tournament_shared_bytes_[source_count], stream>>>(
        balanced_merge_jobs_.data(),
        resident_job_raw_reservations_.data(), resident_plan_.data(),
        input.rows, input.offsets, input.counts, input.ranks,
        resident_rows(), descriptors_.data(),
        canonical_cell_ranks_.data(), device_manifests_.data(),
        active_device_manifest_.data(), canonical_job_prefixes_.data(),
        canonical_next_job_.data(), canonical_cell_counts_.data());
    gpulsmopt2_detail::build_canonical_rank_from_counts_kernel<<<
        gpulsmopt2_detail::kQuotients,
        gpulsmopt2_detail::kFoundationCells, 0, stream>>>(
        canonical_cell_counts_.data(), resident_plan_.data(),
        canonical_level_count_, canonical_cell_ranks_.data(),
        foundation_section_output_counts_.data());
    scan_bytes = resident_scan_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        resident_scan_temp_.data(), scan_bytes,
        foundation_section_output_counts_.data(),
        foundation_source_offsets_.data(),
        gpulsmopt2_detail::kQuotients + 1u, stream));
    finalize_publication_metadata(foundation_source_offsets_.data(), stream);
  }


  std::uint32_t build_sealed_root(
      gpulsm_sparse::NativeRootWorkspace<true> &workspace,
      std::uint32_t selected, std::uint64_t resident_sources,
      std::uint32_t level, std::uint32_t generation,
      bool merge, bool keep_tombstones,
      cudaStream_t stream) {
    gpulsmopt2_detail::ResidentPublicationPlan plan{};
    plan.selected_count = selected;
    plan.destination_level = level;
    plan.output_generation = generation;
    plan.keep_tombstones = keep_tombstones;
    if (merge) {
      plan.source_count = 1u + static_cast<std::uint32_t>(
          __builtin_popcountll(resident_sources));
      if (plan.source_count > gpulsmopt2_detail::kMaximumMergeSources)
        throw std::invalid_argument("invalid sealed carry sources");
      validate_publication_sources(plan.source_count);
      if (resident_sources) {
        plan.source_level_limit = 63u -
            static_cast<std::uint32_t>(__builtin_clzll(resident_sources));
        const std::uint64_t source_prefix = plan.source_level_limit == 63u
            ? ~std::uint64_t{0}
            : (std::uint64_t{1u} << (plan.source_level_limit + 1u)) - 1u;
        if ((host_occupied_level_mask_ & source_prefix) != resident_sources)
          throw std::invalid_argument("non-prefix sealed carry sources");
      }
      plan.job_capacity = canonical_job_capacities_[plan.source_count];
      plan.tournament_workspace_bytes = static_cast<std::uint32_t>(
          canonical_tournament_shared_bytes_[plan.source_count]);
    }
    ensure_level_storage_mapped(level, stream);
    plan.output_capacity = level_capacity(level);
    plan.output_begin = level_begin(level) +
        std::uint64_t{generation} * plan.output_capacity;
    plan.status = selected > plan.output_capacity
        ? gpulsmopt2_detail::kPublicationOutputOverflow
        : gpulsmopt2_detail::kPublicationSuccess;

    if (!merge) {
      if (plan.status) throw std::length_error("sealed direct root capacity");
      workspace.deposit(resident_rows(), plan.output_begin,
                        workspace.section_offsets(), stream);
      plan.survivor_count = selected;
      finalize_direct_root(plan, workspace.section_offsets(), stream);
      CUDA_CHECK(cudaGetLastError());
      return selected;
    }

    CUDA_CHECK(cudaMemcpyAsync(
        resident_plan_.data(), &plan, sizeof(plan),
        cudaMemcpyHostToDevice, stream));
    launch_publication_merge(
        {workspace.epoch_rows(), workspace.section_offsets(),
         workspace.section_counts(), workspace.epoch_ranks()},
        plan.source_count, stream);
    CUDA_CHECK(cudaMemcpyAsync(
        &plan, resident_plan_.data(), sizeof(plan),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (plan.status)
      throw std::runtime_error("sealed canonical carry failed");
    return static_cast<std::uint32_t>(plan.survivor_count);
  }

  static std::uint32_t next_sealed_output(
      const gpulsm_sparse::ForestPlan &plan,
      const std::array<bool, gpulsmopt2_detail::kMaximumLevels> &built) {
    for (std::uint32_t candidate = 0u;
         candidate < plan.output_count; ++candidate) {
      if (built[candidate]) continue;
      const std::uint64_t destination_bit =
          std::uint64_t{1u} << plan.outputs[candidate].destination;
      bool has_unbuilt_reader = false;
      for (std::uint32_t reader = 0u; reader < plan.output_count;
           ++reader) {
        if (reader == candidate || built[reader]) continue;
        has_unbuilt_reader |=
            (plan.outputs[reader].resident_sources & destination_bit) != 0u;
      }
      if (!has_unbuilt_reader) return candidate;
    }
    throw std::logic_error("sealed slot-lifetime dependency cycle");
  }

  std::uint32_t sealed_output_generation(
      const gpulsm_sparse::ForestPlan &plan,
      const gpulsm_sparse::PlannedRoot &root,
      const gpulsmopt2_detail::DeviceManifest &manifest,
      gpulsm_sparse::SealedForestCommand &command,
      cudaStream_t stream) {
    if (root.destination >= canonical_level_count_)
      throw std::overflow_error("sealed destination exceeds forest");
    const std::uint64_t bit = std::uint64_t{1u} << root.destination;
    std::uint32_t generation = 0u;
    if ((plan.old_mask & bit) && (root.resident_sources & bit)) {
      if (root.destination + 1u != canonical_level_count_)
        throw std::logic_error(
            "sealed root overwrites its own nonterminal source");
      ensure_canonical_top_rollover_bank(stream);
      generation =
          (manifest.levels[root.destination].storage_generation ^ 1u) & 1u;
    }
    // Reuse a slot after its readers are queued.
    if (generation) command.output_generation_bits |= bit;
    return generation;
  }


  void commit_sealed_forest(
      const gpulsm_sparse::ForestPlan &plan,
      const gpulsm_sparse::SealedForestCommand &command,
      const std::array<std::uint32_t,
                       gpulsmopt2_detail::kMaximumLevels> &output_counts,
      cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(
        sealed_device_command_.data(), &command, sizeof(command),
        cudaMemcpyHostToDevice, stream));
    gpulsm_sparse::publish_sealed_forest<<<
        1, gpulsmopt2_detail::kMaximumLevels, 0, stream>>>(
        sealed_device_command_.data(), device_manifests_.data(),
        active_device_manifest_.data(), query_occupied_level_mask_.data(),
        sealed_device_receipt_.data());
    gpulsm_sparse::SealedForestReceipt receipt{};
    CUDA_CHECK(cudaMemcpyAsync(
        &receipt, sealed_device_receipt_.data(), sizeof(receipt),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (receipt.status || receipt.occupied_mask != plan.final_mask)
      throw std::runtime_error("sealed forest publication rejected");
    for (std::uint32_t level = 0u;
         level < gpulsmopt2_detail::kMaximumLevels; ++level) {
      const std::uint64_t bit = std::uint64_t{1u} << level;
      if (plan.consumed_mask & bit) level_counts_[level] = 0u;
      if (plan.output_mask & bit) level_counts_[level] = output_counts[level];
    }
    host_occupied_level_mask_ = plan.final_mask;
  }

  struct SealedAdmission {
    gpulsm_sparse::ForestPlan plan;
    std::uint64_t epoch_rows;
    std::uint64_t accepted_rows;
    std::uint64_t maximum_interval;
  };

  // Empty admissions skip planning and allocation.
  std::optional<SealedAdmission> plan_sealed_admission(std::uint64_t count) {
    if (pending_batches_ || !count) return std::nullopt;
    const std::uint64_t epoch_rows = epoch_row_capacity();
    const std::uint64_t complete_epochs = count / epoch_rows;
    if (!complete_epochs) return std::nullopt;
    SealedAdmission admission{
        gpulsm_sparse::plan_forest(
            host_occupied_level_mask_, complete_epochs, sealed_tier_slots()),
        epoch_rows, complete_epochs * epoch_rows, 0u};
    bool fusion = false;
    for (std::uint32_t output = 0u; output < admission.plan.output_count; ++output) {
      const auto &root = admission.plan.outputs[output];
      fusion |= root.raw_epochs > 1u || root.resident_sources != 0u;
      admission.maximum_interval = std::max(
          admission.maximum_interval, root.raw_epochs * epoch_rows);
    }
    if (!fusion) return std::nullopt;
    return admission;
  }

  struct SealedManifestSnapshot {
    gpulsmopt2_detail::DeviceManifest manifests[2]{};
    std::uint32_t active{};
  };

  SealedManifestSnapshot read_sealed_manifest() {
    SealedManifestSnapshot snapshot;
    CUDA_CHECK(cudaMemcpy(&snapshot.active, active_device_manifest_.data(),
                          sizeof(snapshot.active), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(snapshot.manifests, device_manifests_.data(),
                          sizeof(snapshot.manifests), cudaMemcpyDeviceToHost));
    snapshot.active &= 1u;
    return snapshot;
  }


  // Build every planned root from its slice of the input, then publish them
  // together.
  void execute_sealed_admission(
      const std::uint32_t *keys, const std::uint32_t *values,
      std::uint64_t count, bool tombstone,
      const SealedAdmission &admission, const SealedManifestSnapshot &snapshot,
      gpulsm_sparse::NativeRootWorkspace<true> &workspace,
      cudaStream_t stream) {
    const auto &plan = admission.plan;
    const auto &manifests = snapshot.manifests;
    const std::uint32_t active = snapshot.active;
    const std::uint64_t epoch_rows = admission.epoch_rows;
    gpulsm_sparse::SealedForestCommand command{};
    command.expected_mask = plan.old_mask;
    command.consumed_mask = plan.consumed_mask;
    command.output_mask = plan.output_mask;
    command.final_mask = plan.final_mask;
    std::array<std::uint32_t, gpulsmopt2_detail::kMaximumLevels>
        output_counts{};
    std::array<bool, gpulsmopt2_detail::kMaximumLevels> built{};
    for (std::uint32_t ordinal = 0u; ordinal < plan.output_count;
         ++ordinal) {
      const std::uint32_t output = next_sealed_output(plan, built);
      built[output] = true;
      const auto &root = plan.outputs[output];
      if (root.destination >= canonical_level_count_)
        throw std::overflow_error("sealed destination exceeds forest");
      const std::uint64_t raw_begin = root.raw_begin * epoch_rows;
      const std::uint64_t raw_count = root.raw_epochs * epoch_rows;
      if (raw_begin > count || raw_count > count - raw_begin ||
          raw_count > level_capacity(root.destination))
        throw std::length_error("invalid sealed root interval");
      const std::uint32_t generation = sealed_output_generation(
          plan, root, manifests[active], command, stream);
      // Higher slots are older. Include other outputs of this operation,
      // not only runs that were present before it, in the retention decision.
      const bool keep_tombstones = root.destination < 63u &&
          (plan.final_mask >> (root.destination + 1u)) != 0u;
      const bool merge = root.resident_sources ||
          (!keep_tombstones && tombstone);
      const std::uint32_t selected = workspace.prepare(
          keys + raw_begin, tombstone ? nullptr : values + raw_begin,
          static_cast<std::uint32_t>(raw_count), tombstone, merge, stream);
      // Empty outputs keep their planned slot to preserve the radix-4 slot
      // order. Their region counts are zero, so searches skip them.
      output_counts[root.destination] = build_sealed_root(
          workspace, selected, root.resident_sources,
          root.destination, generation, merge, keep_tombstones, stream);
    }
    commit_sealed_forest(plan, command, output_counts, stream);
  }

  std::size_t try_admit_sealed_inline(
      const std::uint32_t *keys, const std::uint32_t *values,
      std::size_t count, bool tombstone, cudaStream_t stream) {
    const auto admission = plan_sealed_admission(count);
    if (!admission) return 0u;
    const std::uint64_t maximum_interval = admission->maximum_interval;
    if (!maximum_interval || maximum_interval > std::numeric_limits<std::uint32_t>::max())
      throw std::length_error("sealed interval exceeds 32 bits");
    const std::size_t maximum_capacity = sealed_workspace_capacity(
        publication_capacity_, level_zero_capacity_);
    if (maximum_capacity <= level_zero_capacity_ || maximum_interval > maximum_capacity)
      return 0u;

    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!sealed_workspace_ || maximum_interval > sealed_workspace_->capacity()) {
      // Release old scratch before allocating more.
      sealed_workspace_.reset();
      sealed_workspace_ = std::make_unique<gpulsm_sparse::NativeRootWorkspace<true>>(
          static_cast<std::uint32_t>(maximum_interval));
    }
    const auto snapshot = read_sealed_manifest();
    execute_sealed_admission(keys, values, count, tombstone, *admission,
                             snapshot, *sealed_workspace_, stream);
    return static_cast<std::size_t>(admission->accepted_rows);
  }

  void admit(const std::uint32_t *keys, const std::uint32_t *values,
             std::size_t count, bool tombstone, cudaStream_t stream) {
    if (!count) return;
    if (!keys || (!tombstone && !values))
      throw std::invalid_argument("invalid GPULSMOpt update batch");
    admit_batches(count, stream,
        [&](std::uint64_t first, std::uint32_t count) {
          admit_tile(keys + first, tombstone ? nullptr : values + first,
                     count, tombstone, stream);
        },
        [&](std::uint64_t first, std::uint64_t count) -> std::uint64_t {
          return try_admit_sealed_inline(
              keys + first, tombstone ? nullptr : values + first,
              count, tombstone, stream);
        });
  }

  void admit_tile(const std::uint32_t *keys, const std::uint32_t *values,
                  std::uint32_t n, bool tombstone,
                  cudaStream_t stream) {
    const auto tile = begin_admission_tile(n, tombstone);
    CUDA_CHECK(cudaMemsetAsync(
        tile.signatures, 0,
        gpulsmopt2_detail::kQuotients * sizeof(std::uint64_t), stream));
    gpulsmopt2_detail::count_quotients_kernel<false><<<
        blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
            keys, n, admission_counts_.data(), radix_ids_out_.data());
    std::size_t scan_bytes = admission_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        admission_temp_.data(), scan_bytes, admission_counts_.data(),
        tile.offsets, gpulsmopt2_detail::kQuotients + 1u, stream));
    if (tombstone) {
      gpulsmopt2_detail::scatter_admission_records_kernel<true><<<
          blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
              keys, values, n, tile.slot, tile.offsets, radix_ids_out_.data(),
              tile.keys, tile.payloads);
    } else {
      gpulsmopt2_detail::scatter_admission_records_kernel<false><<<
          blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
              keys, values, n, tile.slot, tile.offsets, radix_ids_out_.data(),
              tile.keys, tile.payloads);
    }
    gpulsmopt2_detail::build_admission_signatures_kernel<<<
        blocks(n), gpulsmopt2_detail::kThreads, 0, stream>>>(
            tile.keys, n, tile.signatures);
    commit_admission_signatures(tile, stream);
    finish_admission_tile(tile, stream);
  }

  template <bool ResidentOutput>
  void launch_epoch_resolvers(cudaStream_t stream,
                               std::uint64_t resident_destination,
                               std::uint16_t *epoch_ranks) {
    gpulsmopt2_detail::resolve_canonical_epoch_active_jobs_kernel<ResidentOutput><<<
        canonical_epoch_resolver_blocks_,
        gpulsmopt2_detail::kFoundationCompactionThreads, 0, stream>>>(
            balanced_merge_raw_counts_.data(),
            publication_selected_count_.data(),
            canonical_next_job_.data(), raw_keys_.data(),
            raw_payloads_.data(), raw_offsets_.data(),
            static_cast<std::uint32_t>(batch_capacity_),
            pending_batches_,
            foundation_source_offsets_.data(),
            publication_rows_a_.data(), resident_rows(),
            resident_destination, foundation_section_output_counts_.data(),
            epoch_ranks);
    gpulsmopt2_detail::resolve_canonical_epoch_oversized_kernel<ResidentOutput><<<
        canonical_epoch_resolver_blocks_,
        gpulsmopt2_detail::kFoundationCompactionThreads, 0, stream>>>(
            balanced_merge_raw_counts_.data(),
            local_epoch_overflow_flag_.data(),
            canonical_cell_counts_.data() +
                canonical_epoch_workspace_slots_,
            raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
            static_cast<std::uint32_t>(batch_capacity_),
            pending_batches_,
            foundation_source_offsets_.data(),
            publication_rows_a_.data(), resident_rows(),
            resident_destination, foundation_section_output_counts_.data(),
            epoch_ranks,
            reinterpret_cast<unsigned long long *>(
                canonical_epoch_workspace_.data()),
            canonical_cell_counts_.data(),
            canonical_epoch_workspace_slots_);
  }

  void launch_canonical_epoch_resolution(
      cudaStream_t stream, bool materialize_resident,
      std::uint32_t destination_level,
      std::uint16_t *rank_output = nullptr,
      bool build_cell_ranks = true) {
    CUDA_CHECK(cudaMemsetAsync(
        publication_selected_count_.data(), 0, sizeof(std::uint32_t),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        local_epoch_overflow_flag_.data(), 0, sizeof(std::uint32_t),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        canonical_next_job_.data(), 0, sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        canonical_cell_counts_.data(), 0,
        std::size_t{canonical_epoch_workspace_slots_ + 1u} *
            sizeof(std::uint32_t),
        stream));
    gpulsmopt2_detail::count_canonical_epoch_jobs_kernel<<<
        blocks(gpulsmopt2_detail::kQuotients + 1u),
        gpulsmopt2_detail::kThreads, 0, stream>>>(
            raw_offsets_.data(), pending_batches_,
            foundation_section_output_counts_.data(),
            balanced_merge_raw_counts_.data(),
            publication_selected_count_.data(),
            local_epoch_overflow_flag_.data());
    std::size_t scan_bytes = resident_scan_temp_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        resident_scan_temp_.data(), scan_bytes,
        foundation_section_output_counts_.data(),
        foundation_source_offsets_.data(),
        gpulsmopt2_detail::kQuotients + 1u, stream));
    std::uint16_t *epoch_ranks = build_cell_ranks
        ? rank_output
            ? rank_output
            : canonical_cell_ranks_.data() +
                  std::size_t{destination_level} *
                      gpulsmopt2_detail::kLocalRankEntries
        : nullptr;
    const std::uint64_t resident_destination =
        level_begin(destination_level);

    if (materialize_resident)
      launch_epoch_resolvers<true>(stream, resident_destination, epoch_ranks);
    else
      launch_epoch_resolvers<false>(stream, resident_destination, epoch_ranks);
    gpulsmopt2_detail::sum_canonical_section_counts_kernel<<<
        1, gpulsmopt2_detail::kThreads, 0, stream>>>(
            foundation_section_output_counts_.data(),
            publication_selected_count_.data());
  }

  void launch_canonical_publication_commands(
      cudaStream_t stream, std::uint32_t destination,
      std::uint32_t source_count, bool direct_epoch,
      bool top_level_rollover) {
    validate_publication_sources(source_count);
    // Validate destination before rank indexing.
    if (destination >= canonical_level_count_) {
      auto &failure = publication_receipt()[0];
      failure = {};
      failure.selected_count = pending_records_;
      failure.destination_level = destination;
      failure.source_count = source_count;
      failure.output_capacity = publication_capacity_;
      failure.status = gpulsmopt2_detail::kPublicationLevelOverflow;
      CUDA_CHECK(cudaMemcpyAsync(
          resident_plan_.data(), &failure, sizeof(failure),
          cudaMemcpyHostToDevice, stream));
      return;
    }
    const std::uint32_t tier_begin = destination <
            canonical_regular_level_count_
        ? (destination / 3u) * 3u : canonical_regular_level_count_;
    const std::uint32_t epoch_rank_level = direct_epoch
        ? destination : tier_begin;
    std::uint16_t *epoch_ranks = top_level_rollover
        ? canonical_rollover_epoch_ranks_->data()
        : canonical_cell_ranks_.data() +
              std::size_t{epoch_rank_level} *
                  gpulsmopt2_detail::kLocalRankEntries;
    launch_canonical_epoch_resolution(
        stream, direct_epoch, destination, epoch_ranks);
    const std::uint32_t job_capacity =
        canonical_job_capacities_[source_count];
    const std::uint32_t tournament_workspace_bytes =
        static_cast<std::uint32_t>(
            canonical_tournament_shared_bytes_[source_count]);
    gpulsmopt2_detail::choose_canonical_publication_path_kernel<<<
        1, 1, 0, stream>>>(
            publication_selected_count_.data(), device_manifests_.data(),
            active_device_manifest_.data(), level_storage_spans_.data(),
            canonical_regular_level_count_, canonical_level_count_,
            job_capacity,
            tournament_workspace_bytes, top_level_rollover,
            resident_plan_.data());

    if (direct_epoch) {
      finalize_publication_metadata<true>(
          foundation_source_offsets_.data(), stream,
          foundation_section_output_counts_.data(),
          publication_selected_count_.data());
    } else {
      launch_publication_merge(
          {publication_rows_a_.data(), foundation_source_offsets_.data(),
           foundation_section_output_counts_.data(), epoch_ranks},
          source_count, stream);
    }
    gpulsm_sparse::publish_manifest_kernel<<<
        1, gpulsmopt2_detail::kMaximumLevels, 0, stream>>>(
        resident_plan_.data(), device_manifests_.data(),
        active_device_manifest_.data(), query_occupied_level_mask_.data());
    CUDA_CHECK(cudaGetLastError());
  }

  void canonical_publication_parameters(
      std::uint32_t &destination, bool &direct_epoch) const {
    destination = gpulsmopt2_detail::kMaximumLevels;
    std::uint32_t tier_begin = 0u;
    for (std::uint32_t begin = 0u;
         begin < canonical_regular_level_count_; begin += 3u) {
      const std::uint32_t slots = std::min(
          3u, canonical_regular_level_count_ - begin);
      const std::uint64_t tier_mask =
          ((std::uint64_t{1} << slots) - 1u) << begin;
      const std::uint32_t filled = static_cast<std::uint32_t>(
          __builtin_popcountll(host_occupied_level_mask_ & tier_mask));
      if (filled < slots) {
        destination = begin + slots - 1u - filled;
        tier_begin = begin;
        break;
      }
    }
    if (destination == gpulsmopt2_detail::kMaximumLevels &&
        canonical_regular_level_count_ < canonical_level_count_ &&
        !(host_occupied_level_mask_ &
          (std::uint64_t{1} << canonical_regular_level_count_))) {
      destination = canonical_regular_level_count_;
      tier_begin = canonical_regular_level_count_;
    }
    direct_epoch = tier_begin == 0u &&
        (host_occupied_level_mask_ != 0u || !pending_has_tombstones_) &&
        pending_records_ <= level_zero_capacity_;
  }

  std::uint64_t publication_source_mask(
      std::uint32_t destination, bool top_level_rollover) const {
    if (top_level_rollover) {
      const std::uint64_t mask = destination >=
              gpulsmopt2_detail::kMaximumLevels - 1u
          ? ~std::uint64_t{0}
          : (std::uint64_t{1u} << (destination + 1u)) - 1u;
      return host_occupied_level_mask_ & mask;
    }
    const std::uint32_t tier_begin = destination <
            canonical_regular_level_count_
        ? (destination / 3u) * 3u
        : canonical_regular_level_count_;
    const std::uint64_t mask = tier_begin
        ? (std::uint64_t{1u} << tier_begin) - 1u : 0u;
    return host_occupied_level_mask_ & mask;
  }

  void launch_epoch_publication(cudaStream_t stream) {
    std::uint32_t destination = 0u;
    bool direct_epoch = false;
    canonical_publication_parameters(destination, direct_epoch);
    const bool top_level_rollover = destination >= canonical_level_count_;
    if (top_level_rollover) {
      ensure_canonical_top_rollover_bank(stream);
      destination = canonical_level_count_ - 1u;
      direct_epoch = false;
    } else {
      ensure_level_storage_mapped(destination, stream);
    }
    const std::uint64_t source_mask = publication_source_mask(
        destination, top_level_rollover);
    const std::uint32_t source_count = 1u + static_cast<std::uint32_t>(
        __builtin_popcountll(source_mask));
    launch_canonical_publication_commands(
        stream, destination, source_count, direct_epoch,
        top_level_rollover);
  }

  void publish_epoch(cudaStream_t stream) {
    if (pending_records_ > publication_capacity_) {
      // Preserve an epoch that cannot reserve output.
      gpulsmopt2_detail::ResidentPublicationPlan failure{};
      failure.selected_count =
          static_cast<std::uint32_t>(pending_records_);
      failure.output_capacity = publication_capacity_;
      failure.status =
          gpulsmopt2_detail::kPublicationOutputOverflow;
      publication_.fail(failure, true);
      return;
    }
    launch_epoch_publication(stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(
        publication_receipt(), resident_plan_.data(),
        sizeof(gpulsmopt2_detail::ResidentPublicationPlan),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemsetAsync(
        raw_epoch_signatures_.data(), 0,
        raw_epoch_signatures_.size() * sizeof(std::uint64_t), stream));
    publication_.await_receipt();
    return;
  }

  static std::size_t aligned_id_bytes(std::size_t count) {
    return (count * sizeof(std::uint32_t) + 255u) & ~std::size_t{255u};
  }
  static std::size_t aligned_operation_bytes(std::size_t bytes) {
    return (bytes + 255u) & ~std::size_t{255u};
  }
  static std::size_t operation_cell_counts_bytes() {
    return std::size_t{gpulsmopt2_detail::kLocalRankEntries} *
        sizeof(std::uint32_t);
  }
  static std::size_t operation_epoch_workspace_rows(
      std::size_t batch_capacity) {
    return std::max<std::size_t>(
        batch_capacity * gpulsmopt2_detail::kBatchesPerEpoch,
        gpulsmopt2_detail::kCanonicalResolverSuffixes);
  }
  static std::size_t operation_epoch_workspace_offset() {
    return aligned_operation_bytes(operation_cell_counts_bytes());
  }
  static std::size_t operation_radix_storage_offset(
      std::size_t batch_capacity) {
    return aligned_operation_bytes(
        operation_epoch_workspace_offset() +
        operation_epoch_workspace_rows(batch_capacity) *
            sizeof(gpulsmopt2_detail::RawPayload));
  }
  static std::size_t radix_workspace_bytes(std::size_t count) {
    return aligned_id_bytes(count) * 3u +
        aligned_id_bytes(gpulsmopt2_detail::kQuotients + 1u);
  }
  static std::uint32_t tqrj_maximum_queries(std::size_t batch_capacity) {
    return static_cast<std::uint32_t>(
        batch_capacity * gpulsmopt2_detail::kBatchesPerEpoch);
  }
  static std::uint32_t tqrj_hash_maximum_entries(
      std::size_t batch_capacity) {
    return gpulsmopt2_detail::tqrj_hash_capacity(
        tqrj_maximum_queries(batch_capacity));
  }
  static std::size_t tqrj_hash_table_offset(std::size_t batch_capacity) {
    return aligned_operation_bytes(
        std::size_t{tqrj_maximum_queries(batch_capacity)} *
        sizeof(std::uint32_t));
  }
  static std::size_t tqrj_hash_task_offset(std::size_t batch_capacity) {
    return aligned_operation_bytes(
        tqrj_hash_table_offset(batch_capacity) +
        std::size_t{tqrj_hash_maximum_entries(batch_capacity)} *
            sizeof(std::uint32_t));
  }
  static std::size_t tqrj_hash_counter_offset(
      std::size_t batch_capacity) {
    return tqrj_hash_task_offset(batch_capacity) +
        gpulsmopt2_detail::kTqrjHashTaskBytes;
  }
  static std::size_t tqrj_hash_tile_offset(std::size_t batch_capacity) {
    return aligned_operation_bytes(
        tqrj_hash_counter_offset(batch_capacity) +
        gpulsmopt2_detail::kTqrjHashCounterBytes);
  }
  static std::uint32_t tqrj_maximum_tiles(
      std::size_t batch_capacity) {
    const std::uint32_t queries = tqrj_maximum_queries(batch_capacity);
    const std::uint32_t active =
        std::min(queries, gpulsmopt2_detail::kQuotients);
    return gpulsmopt2_detail::tqrj_hash_tile_count(queries) + active;
  }
  static std::size_t tqrj_hash_query_bases_offset(
      std::size_t batch_capacity) {
    return aligned_operation_bytes(
        tqrj_hash_tile_offset(batch_capacity) +
        std::size_t{2u * tqrj_maximum_tiles(batch_capacity)} *
            sizeof(gpulsmopt2_detail::TqrjHashTile));
  }
  static std::size_t tqrj_hash_workspace_bytes(
      std::size_t batch_capacity) {
    return tqrj_hash_query_bases_offset(batch_capacity) +
        aligned_id_bytes(gpulsmopt2_detail::kQuotients + 1u);
  }
  static std::size_t operation_workspace_initial_bytes(
      std::size_t batch_capacity) {
    return std::max(
        operation_radix_storage_offset(batch_capacity) +
            radix_workspace_bytes(batch_capacity),
        tqrj_hash_workspace_bytes(batch_capacity));
  }
  static std::size_t operation_workspace_maximum_bytes(
      std::size_t batch_capacity) {
    return std::max(
        operation_radix_storage_offset(batch_capacity) +
            radix_workspace_bytes(gpulsmopt2_detail::kMaximumOperationTile),
        tqrj_hash_workspace_bytes(batch_capacity));
  }
  void initialize_operation_workspace_views() {
    std::uint8_t *storage = operation_workspace_.data();
    canonical_cell_counts_.attach(
        reinterpret_cast<std::uint32_t *>(storage),
        gpulsmopt2_detail::kLocalRankEntries);
    canonical_epoch_workspace_.attach(
        reinterpret_cast<gpulsmopt2_detail::RawPayload *>(
            storage + operation_epoch_workspace_offset()),
        operation_epoch_workspace_rows(batch_capacity_));
  }
  struct RadixWorkspaceView {
    std::uint32_t *input_ids;
    std::uint32_t *quotient_offsets;
  };

  RadixWorkspaceView ensure_radix_workspace(std::size_t count) {
    const std::size_t capacity = std::max(radix_keys_.size(), count);
    const std::size_t ids_bytes = aligned_id_bytes(capacity);
    const std::size_t query_offset_bytes =
        aligned_id_bytes(gpulsmopt2_detail::kQuotients + 1u);
    const std::size_t required = ids_bytes * 3u + query_offset_bytes;
    const std::size_t radix_offset =
        operation_radix_storage_offset(batch_capacity_);
    const std::size_t operation_required = radix_offset + required;
    const std::size_t operation_offset = phase_operation_workspace_offset(batch_capacity_);
    phase_workspace_.grow(operation_offset + operation_required);
    operation_workspace_.grow(operation_required);
    std::uint8_t *storage = operation_workspace_.data() + radix_offset;
    radix_keys_.attach(reinterpret_cast<std::uint32_t *>(storage), capacity);
    radix_ids_out_.attach(
        reinterpret_cast<std::uint32_t *>(storage + ids_bytes), capacity);
    return {reinterpret_cast<std::uint32_t *>(storage + ids_bytes * 2u),
            reinterpret_cast<std::uint32_t *>(storage + ids_bytes * 3u)};
  }
  void materialize_range_hot_sections(std::uint64_t total,
                                      cudaStream_t stream) {
    if (!total) return;
    if (total > gpulsmopt2_detail::kMaximumPublicationRows)
      throw std::length_error("GPULSMOpt crowded range input is too large");

    struct Window {
      std::uint32_t quotient_begin;
      std::uint32_t quotient_end;
      std::uint64_t input_begin;
      std::uint32_t input_count;
    };
    std::vector<Window> windows;
    if (total <= gpulsmopt2_detail::kRangeHotWindowRows) {
      windows.push_back({0u, gpulsmopt2_detail::kQuotients, 0u,
                         static_cast<std::uint32_t>(total)});
    } else {
      CUDA_CHECK(cudaMemcpyAsync(
          range_hot_offsets_receipt(), range_hot_offsets_.data(),
          (gpulsmopt2_detail::kQuotients + 1u) * sizeof(std::uint64_t),
          cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const std::uint64_t *offsets = range_hot_offsets_receipt();
      std::uint32_t begin = 0u;
      while (begin < gpulsmopt2_detail::kQuotients &&
             offsets[begin] < total) {
        const std::uint64_t base = offsets[begin];
        std::uint32_t end = begin;
        while (end < gpulsmopt2_detail::kQuotients &&
               offsets[end + 1u] - base <=
                   gpulsmopt2_detail::kRangeHotWindowRows)
          ++end;
        if (end == begin) ++end;
        const std::uint64_t count = offsets[end] - base;
        if (count > std::numeric_limits<int>::max())
          throw std::length_error(
              "one GPULSMOpt crowded section exceeds the GPU sort limit");
        windows.push_back({begin, end, base,
                           static_cast<std::uint32_t>(count)});
        begin = end;
      }
    }

    std::size_t maximum_window = 0u;
    for (const Window &window : windows)
      maximum_window = std::max<std::size_t>(
          maximum_window, window.input_count);
    if (range_hot_token_capacity_ < maximum_window) {
      range_hot_tokens_a_.resize(maximum_window);
      range_hot_tokens_b_.resize(maximum_window);
      range_hot_token_capacity_ = maximum_window;
    }
    ensure_publication_capacity(static_cast<std::size_t>(total), stream);

    std::size_t required_temp = 0u;
    for (const Window &window : windows) {
      std::size_t sort_bytes{};
      CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
          nullptr, sort_bytes, range_hot_tokens_a_.data(),
          range_hot_tokens_b_.data(),
          static_cast<int>(window.input_count),
          static_cast<int>(window.quotient_end - window.quotient_begin),
          range_hot_window_offsets_.data(),
          range_hot_window_offsets_.data() + 1u, 32, 48, stream));
      using KeyIterator = cub::TransformInputIterator<
          std::uint32_t, gpulsmopt2_detail::RangeHotTokenKey,
          const std::uint64_t *>;
      const KeyIterator keys(range_hot_tokens_b_.data(),
                             gpulsmopt2_detail::RangeHotTokenKey{});
      auto *winner_tokens = reinterpret_cast<std::uint64_t *>(
          publication_rows_a_.data() + window.input_begin);
      std::size_t reduce_bytes{};
      CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
          nullptr, reduce_bytes, keys,
          publication_keys_a_.data() + window.input_begin,
          range_hot_tokens_b_.data(), winner_tokens,
          range_hot_selected_count_.data(),
          gpulsmopt2_detail::RangeHotNewestToken{
              raw_payloads_.data(), static_cast<std::uint32_t>(
                  batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch)},
          window.input_count, stream));
      required_temp = std::max(required_temp,
                               std::max(sort_bytes, reduce_bytes));
    }
    if (range_hot_temp_.size() < required_temp)
      range_hot_temp_.resize(required_temp);

    for (const Window &window : windows) {
      const std::uint32_t quotient_count =
          window.quotient_end - window.quotient_begin;
      gpulsmopt2_detail::make_range_hot_window_offsets_kernel<<<
          blocks(std::size_t{quotient_count} + 1u),
          gpulsmopt2_detail::kThreads, 0, stream>>>(
              range_hot_offsets_.data(), window.quotient_begin,
              quotient_count, window.input_begin,
              range_hot_window_offsets_.data());
      gpulsmopt2_detail::emit_range_hot_tokens_kernel<<<
          quotient_count, gpulsmopt2_detail::kThreads, 0, stream>>>(
              raw_keys_.data(), raw_offsets_.data(),
              static_cast<std::uint32_t>(batch_capacity_), pending_batches_,
              resident_rows(), descriptors_.data(),
              query_occupied_level_mask_.data(), range_hot_counts_.data(),
              range_hot_offsets_.data(), window.quotient_begin,
              window.quotient_end, window.input_begin,
              range_hot_tokens_a_.data());
      std::size_t workspace_bytes = range_hot_temp_.size();
      CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
          range_hot_temp_.data(), workspace_bytes,
          range_hot_tokens_a_.data(), range_hot_tokens_b_.data(),
          static_cast<int>(window.input_count),
          static_cast<int>(quotient_count),
          range_hot_window_offsets_.data(),
          range_hot_window_offsets_.data() + 1u, 32, 48, stream));
      using KeyIterator = cub::TransformInputIterator<
          std::uint32_t, gpulsmopt2_detail::RangeHotTokenKey,
          const std::uint64_t *>;
      const KeyIterator keys(range_hot_tokens_b_.data(),
                             gpulsmopt2_detail::RangeHotTokenKey{});
      auto *winner_tokens = reinterpret_cast<std::uint64_t *>(
          publication_rows_a_.data() + window.input_begin);
      workspace_bytes = range_hot_temp_.size();
      CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
          range_hot_temp_.data(), workspace_bytes, keys,
          publication_keys_a_.data() + window.input_begin,
          range_hot_tokens_b_.data(), winner_tokens,
          range_hot_selected_count_.data(),
          gpulsmopt2_detail::RangeHotNewestToken{
              raw_payloads_.data(), static_cast<std::uint32_t>(
                  batch_capacity_ * gpulsmopt2_detail::kBatchesPerEpoch)},
          window.input_count, stream));
      gpulsmopt2_detail::materialize_range_hot_winners_kernel<<<
          blocks(window.input_count), gpulsmopt2_detail::kThreads, 0,
          stream>>>(
              publication_rows_a_.data() + window.input_begin,
              range_hot_selected_count_.data(), raw_payloads_.data(),
              resident_rows(), descriptors_.data());
      gpulsmopt2_detail::build_range_hot_descriptors_kernel<<<
          blocks(quotient_count), gpulsmopt2_detail::kThreads, 0, stream>>>(
              publication_keys_a_.data() + window.input_begin,
              range_hot_selected_count_.data(), window.quotient_begin,
              window.quotient_end, window.input_begin,
              range_hot_descriptors_.data());
    }
    CUDA_CHECK(cudaGetLastError());
  }
  void launch_section_ranges(cudaStream_t stream, bool hot_ready) {
    const std::uint16_t *foundation_ranks = canonical_cell_ranks_.data();
    const std::uint32_t foundation = foundation_level();
    if (foundation != gpulsmopt2_detail::kMaximumLevels)
      foundation_ranks = canonical_cell_ranks_.data() +
          std::size_t{foundation} *
              gpulsmopt2_detail::kLocalRankEntries;
    gpulsmopt2_detail::cooperative_section_owned_range_kernel<
        gpulsmopt2_detail::RangeSumSink>
        <<<range_section_blocks_, gpulsmopt2_detail::kSectionRangeThreads,
           0, stream>>>(
            range_section_fragments_out_.data(),
            range_section_tasks_.data(),
            range_section_task_offsets_.data() +
                gpulsmopt2_detail::kQuotients,
            resident_rows(), descriptors_.data(), foundation_ranks,
            publication_rows_a_.data(),
            range_hot_descriptors_.data(), hot_ready,
            raw_keys_.data(),
            raw_payloads_.data(),
            raw_offsets_.data(), static_cast<std::uint32_t>(batch_capacity_),
            pending_batches_, range_fragment_partials_.data(),
            query_occupied_level_mask_.data());
  }
  void launch_fragment_ranges(std::uint32_t fragment_count,
                              std::uint32_t query_count,
                              const DeviceRangeOutputBatch &batch,
                              cudaStream_t stream, bool hot_ready) {
    gpulsmopt2_detail::warp_range_fragment_kernel<
        gpulsmopt2_detail::RangeSumSink>
        <<<(fragment_count + 3u) / 4u, 128, 0, stream>>>(
            range_fragments_.data(), fragment_count,
            range_fragment_offsets_.data() + query_count,
            batch.lo, batch.hi, resident_rows(), descriptors_.data(),
            publication_rows_a_.data(),
            range_hot_descriptors_.data(), hot_ready,
            raw_keys_.data(), raw_payloads_.data(), raw_offsets_.data(),
            static_cast<std::uint32_t>(batch_capacity_),
            pending_batches_, range_fragment_partials_.data(),
            query_occupied_level_mask_.data());
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
    std::uint8_t *storage = range_query_storage_.data();
    std::size_t offset = 0u;
    attach_range_view(range_fragment_counts_, storage, offset, entries);
    attach_range_view(range_fragment_offsets_, storage, offset, entries);
    offset = aligned_range_bytes(offset);
    range_query_temp_ = storage + offset;
  }
  void ensure_range_fragment_capacity(std::size_t count) {
    if (range_fragments_.size() >= count &&
        range_fragment_partials_.size() >= count) return;
    std::size_t bytes = 0u;
    bytes += aligned_range_bytes(
        count * sizeof(gpulsmopt2_detail::RangeFragment));
    bytes += aligned_range_bytes(count * sizeof(unsigned long long));
    range_fragment_storage_.resize(bytes);
    std::uint8_t *storage = range_fragment_storage_.data();
    std::size_t offset = 0u;
    attach_range_view(range_fragments_, storage, offset, count);
    attach_range_view(range_fragment_partials_, storage, offset, count);
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
    std::uint8_t *storage = range_section_storage_.data();
    std::size_t offset = 0u;
    attach_range_view(range_section_keys_in_, storage, offset, count);
    attach_range_view(range_section_keys_out_, storage, offset, count);
    attach_range_view(range_section_fragments_in_, storage, offset, count);
    attach_range_view(range_section_fragments_out_, storage, offset, count);
    attach_range_view(range_fragment_partials_, storage, offset, count);
    attach_range_view(range_section_offsets_, storage, offset, sections);
    attach_range_view(range_section_task_offsets_, storage, offset, sections);
    attach_range_view(range_section_task_counts_, storage, offset, sections);
    attach_range_view(range_section_tasks_, storage, offset, maximum_tasks);
    offset = aligned_range_bytes(offset);
    range_section_temp_ = storage + offset;
    range_section_temp_bytes_ = temp_bytes;
  }
  std::size_t batch_capacity_{};
  std::size_t publication_capacity_{};
  std::size_t level_zero_capacity_{};
  std::size_t level_pool_capacity_{};
  std::uint32_t canonical_regular_level_count_{};
  std::uint32_t canonical_level_count_{};
  std::array<gpulsmopt2_detail::LevelStorageSpan,
             gpulsmopt2_detail::kMaximumLevels>
      level_storage_spans_host_{};
  std::uint32_t resident_merge_capacity_{};
  std::size_t maximum_resident_jobs_{};
  mutable std::mutex operation_mutex_;
  std::uint64_t host_occupied_level_mask_{};
  std::uint32_t pending_batches_{};
  std::uint32_t pending_records_{};
  bool pending_has_tombstones_{};
  PublicationLifecycle publication_;
  std::uint8_t *range_query_temp_{};
  std::uint8_t *range_section_temp_{};
  std::size_t range_section_temp_bytes_{};
  cudaEvent_t operation_done_{};
  std::array<std::uint32_t,
             gpulsmopt2_detail::kMaximumMergeSources + 1u>
      canonical_tournament_blocks_{};
  std::array<std::uint32_t,
             gpulsmopt2_detail::kMaximumMergeSources + 1u>
      canonical_job_capacities_{};
  std::uint32_t canonical_epoch_resolver_blocks_{};
  std::uint32_t canonical_epoch_workspace_slots_{};
  std::uint32_t tqrj_hash_worker_blocks_{};
  std::array<std::size_t,
             gpulsmopt2_detail::kMaximumMergeSources + 1u>
      canonical_tournament_shared_bytes_{};
  std::uint32_t resident_planner_blocks_{};
  std::uint32_t range_section_blocks_{};

  gpulsmopt2_detail::VirtualBuffer<std::uint32_t> arena_key_flags_;
  gpulsmopt2_detail::VirtualBuffer<std::uint32_t> arena_values_;
  gpulsmopt2_detail::Buffer<std::uint8_t> static_workspace_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::Descriptor> descriptors_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::DeviceManifest>
      device_manifests_;
  gpulsmopt2_detail::Buffer<std::uint32_t> active_device_manifest_;
  gpulsmopt2_detail::Buffer<std::uint64_t> query_occupied_level_mask_;
  gpulsm_sparse::Buffer<gpulsm_sparse::SealedForestCommand>
      sealed_device_command_;
  gpulsm_sparse::Buffer<gpulsm_sparse::SealedForestReceipt>
      sealed_device_receipt_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::ResidentPublicationPlan>
      resident_plan_;
  gpulsmopt2_detail::PinnedBuffer<gpulsmopt2_detail::PinnedReceipts>
      pinned_receipts_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::LevelStorageSpan>
      level_storage_spans_;
  gpulsmopt2_detail::Buffer<std::uint16_t> canonical_cell_ranks_;
  std::unique_ptr<gpulsmopt2_detail::Buffer<std::uint16_t>>
      canonical_rollover_epoch_ranks_;
  gpulsmopt2_detail::VirtualBuffer<std::uint8_t> phase_workspace_;
  gpulsmopt2_detail::VirtualBuffer<std::uint8_t> operation_workspace_;
  std::unique_ptr<gpulsm_sparse::NativeRootWorkspace<true>> sealed_workspace_;
  gpulsmopt2_detail::Buffer<std::uint32_t> canonical_cell_counts_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::CanonicalJobPrefix>
      canonical_job_prefixes_;
  gpulsmopt2_detail::Buffer<std::uint32_t> canonical_next_job_;
  gpulsmopt2_detail::Buffer<std::uint32_t> raw_keys_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RawPayload> raw_payloads_;
  gpulsmopt2_detail::Buffer<std::uint32_t> raw_offsets_;
  gpulsmopt2_detail::Buffer<std::uint64_t> raw_signatures_;
  gpulsmopt2_detail::Buffer<std::uint64_t> raw_epoch_signatures_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::RawPayload>
      canonical_epoch_workspace_;
  gpulsmopt2_detail::VirtualBuffer<std::uint32_t> publication_keys_a_;
  gpulsmopt2_detail::Buffer<std::uint32_t> publication_selected_count_;
  gpulsmopt2_detail::Buffer<std::uint32_t> foundation_source_offsets_,
      foundation_section_output_counts_;
  gpulsmopt2_detail::Buffer<std::uint64_t> balanced_merge_raw_counts_,
      resident_job_raw_reservations_;
  gpulsmopt2_detail::Buffer<std::uint32_t>
      resident_tile_job_counts_, resident_tile_job_offsets_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::BalancedMergeJob>
      balanced_merge_jobs_;
  gpulsmopt2_detail::Buffer<std::uint8_t> resident_scan_temp_;
  gpulsmopt2_detail::Buffer<std::uint32_t> local_epoch_overflow_flag_;
  gpulsmopt2_detail::VirtualBuffer<gpulsmopt2_detail::Row>
      publication_rows_a_;
  gpulsmopt2_detail::Buffer<std::uint32_t> admission_counts_;
  gpulsmopt2_detail::Buffer<std::uint8_t> admission_temp_;
  std::uint32_t level_counts_[gpulsmopt2_detail::kMaximumLevels]{};
  std::uint32_t raw_batch_counts_[gpulsmopt2_detail::kBatchesPerEpoch]{};
  gpulsmopt2_detail::Buffer<std::uint32_t> radix_keys_, radix_ids_out_;

  gpulsmopt2_detail::Buffer<unsigned long long> range_partials_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_reduction_completion_;
  gpulsmopt2_detail::Buffer<std::uint64_t> range_fragment_total_;
  gpulsmopt2_detail::Buffer<std::uint64_t> range_hot_counts_,
      range_hot_offsets_, range_hot_tokens_a_, range_hot_tokens_b_;
  gpulsmopt2_detail::Buffer<std::uint32_t> range_hot_window_offsets_,
      range_hot_selected_count_;
  gpulsmopt2_detail::Buffer<gpulsmopt2_detail::Descriptor>
      range_hot_descriptors_;
  gpulsmopt2_detail::Buffer<std::uint8_t> range_hot_temp_;
  std::size_t range_hot_token_capacity_{};
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
