// =============================================================================
// File: impl_gpulsmopt.cuh
// Description: FliX benchmark adapter for the root-level GPULSMOpt
// implementation.
// =============================================================================

#ifndef IMPL_GPULSMOPT_CUH
#define IMPL_GPULSMOPT_CUH

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../gpu_dictionary_adapter.cuh"
#include "cuda_buffer.cuh"
#include "definitions.cuh"

#ifdef CUDA_CHECK
#define GPULSMOPT_RESTORE_FLIX_CUDA_CHECK
#undef CUDA_CHECK
#endif

#include "../GPULSMOpt.cuh"

#ifdef GPULSMOPT_RESTORE_FLIX_CUDA_CHECK
#undef CUDA_CHECK
#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(err));                                         \
      asm("trap;");                                                            \
    }                                                                          \
  } while (0)
#undef GPULSMOPT_RESTORE_FLIX_CUDA_CHECK
#endif

#ifndef GPULSMOPT_BATCH_SIZE
#define GPULSMOPT_BATCH_SIZE 65536
#endif

namespace gpulsmopt_adapter_detail {

inline void check_cuda(cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

class scoped_cuda_event_timer final {
public:
  scoped_cuda_event_timer(cudaStream_t stream, double *output)
      : stream_(stream), output_(output) {
    if (output_) {
      check_cuda(cudaEventCreate(&start_));
      check_cuda(cudaEventCreate(&stop_));
      check_cuda(cudaEventRecord(start_, stream_));
    }
  }

  scoped_cuda_event_timer(const scoped_cuda_event_timer &) = delete;
  scoped_cuda_event_timer &operator=(const scoped_cuda_event_timer &) = delete;

  ~scoped_cuda_event_timer() {
    if (start_)
      cudaEventDestroy(start_);
    if (stop_)
      cudaEventDestroy(stop_);
  }

  void stop() {
    if (!output_ || stopped_)
      return;
    check_cuda(cudaEventRecord(stop_, stream_));
    check_cuda(cudaEventSynchronize(stop_));
    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
    *output_ += elapsed_ms;
    stopped_ = true;
  }

private:
  cudaStream_t stream_ = 0;
  double *output_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
  bool stopped_ = false;
};

__global__ void fill_sequence_kernel(std::uint32_t *values, size_t size) {
  const size_t tid = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (tid < size)
    values[tid] = static_cast<std::uint32_t>(tid);
}

__global__ void finalize_lookup_kernel(const std::uint8_t *found,
                                       smallsize *values, size_t size) {
  const size_t tid = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (tid < size && found[tid] == 0)
    values[tid] = not_found;
}

inline void fill_sequence(std::uint32_t *values, size_t size,
                          cudaStream_t stream) {
  if (size == 0)
    return;
  constexpr int threads_per_block = 256;
  const int blocks =
      static_cast<int>((size + threads_per_block - 1) / threads_per_block);
  fill_sequence_kernel<<<blocks, threads_per_block, 0, stream>>>(values, size);
  check_cuda(cudaGetLastError());
}

inline void finalize_lookup(const std::uint8_t *found, smallsize *values,
                            size_t size, cudaStream_t stream) {
  if (size == 0)
    return;
  constexpr int threads_per_block = 256;
  const int blocks =
      static_cast<int>((size + threads_per_block - 1) / threads_per_block);
  finalize_lookup_kernel<<<blocks, threads_per_block, 0, stream>>>(
      found, values, size);
  check_cuda(cudaGetLastError());
}

} // namespace gpulsmopt_adapter_detail

template <typename key_type_> class gpulsmopt final {
public:
  using key_type = key_type_;
  using value_type = smallsize;

  static_assert(std::is_same_v<key_type, key32>,
                "GPULSMOpt's current FliX adapter supports key32 only");
  static_assert(std::is_same_v<smallsize, std::uint32_t>,
                "GPULSMOpt values must match FliX smallsize values");

  static constexpr const char *name = "gpulsmopt";
  static constexpr operation_support can_lookup = operation_support::async;
  static constexpr operation_support can_lower_bound_rank =
      operation_support::none;
  static constexpr operation_support can_multi_lookup = operation_support::none;
  static constexpr operation_support can_range_lookup =
      operation_support::async;
  static constexpr operation_support can_insert = operation_support::async;
  static constexpr operation_support can_delete = operation_support::async;
  static constexpr operation_support can_update = operation_support::async;
  static constexpr operation_support can_successor = operation_support::async;

  static std::string short_description() {
    return "gpulsmopt";
  }

  static parameters_type parameters() {
    return {
        {"batch_size",
         std::to_string(static_cast<size_t>(GPULSMOPT_BATCH_SIZE))},
        {"segment_buckets",
         std::to_string(static_cast<size_t>(GPULSMOPT_SEGMENT_BUCKETS))},
        {"target_fill",
         std::to_string(static_cast<size_t>(GPULSMOPT_TARGET_FILL))},
        {"inplace_max_incoming",
         std::to_string(static_cast<size_t>(GPULSMOPT_INPLACE_MAX_INCOMING))},
        {"enable_tier2",
         std::to_string(static_cast<size_t>(GPULSMOPT_ENABLE_TIER2))},
        {"tier2_fanout",
         std::to_string(static_cast<size_t>(GPULSMOPT_TIER2_FANOUT))},
        {"tier2_cdg_tables", "1"},
        {"mutable_flush_budget",
         std::to_string(static_cast<size_t>(GPULSMOPT_MUTABLE_FLUSH_BUDGET))},
        {"distinct_keys",
         std::to_string(static_cast<size_t>(GPULSMOPT_DISTINCT_KEYS))},
        {"compact_dg_c0",
         std::to_string(static_cast<size_t>(GPULSMOPT_COMPACT_DG_C0))},
        {"dg_c0_slots",
         std::to_string(static_cast<size_t>(GPULSMOPT_DG_C0_SLOTS))},
    };
  }

  static size_t estimate_build_bytes(size_t size) {
    // segmented sheet slot (key + value + validity byte) over-provisioned for
    // the target fill, plus the adapter's value buffer
    return (2 * sizeof(std::uint32_t) + 1) * size * 2 +
           sizeof(smallsize) * size;
  }

  size_t gpu_resident_bytes() {
    size_t total = dictionary_ ? dictionary_->gpu_resident_bytes() : 0;
    total += build_values_buffer_.num_elements * sizeof(smallsize);
    total += lookup_found_buffer_.num_elements * sizeof(std::uint8_t);
    return total;
  }

  void build(const key_type *keys, size_t size, size_t max_size,
             size_t available_memory_bytes, double *build_time_ms,
             size_t *build_bytes) {
    (void)available_memory_bytes;
    const size_t configured_max_size = std::max(max_size, size);
    if (configured_max_size >
        static_cast<size_t>(std::numeric_limits<std::uint32_t>::max() / 2)) {
      throw std::runtime_error("GPULSMOpt supports at most 2^31-1 records");
    }

    build_values_buffer_.resize(size);

    const size_t configured_batch_size =
        static_cast<size_t>(GPULSMOPT_BATCH_SIZE);
    const size_t config_batch_size = std::max<size_t>(
        1, std::min(configured_max_size == 0 ? size : configured_max_size,
                    configured_batch_size));
    DictionaryConfig config;
    config.max_elements = configured_max_size;
    config.batch_size = config_batch_size;

    gpulsmopt_adapter_detail::scoped_cuda_event_timer timer(0, build_time_ms);
    dictionary_ = std::make_unique<GPULSMOpt>(config);
    gpulsmopt_adapter_detail::fill_sequence(
        reinterpret_cast<std::uint32_t *>(build_values_buffer_.ptr()), size, 0);

    DeviceKeyValueBatch batch;
    batch.keys = reinterpret_cast<const std::uint32_t *>(keys);
    batch.values =
        reinterpret_cast<const std::uint32_t *>(build_values_buffer_.ptr());
    batch.count = size;
    dictionary_->insert(batch, 0);
    dictionary_->drain_to_sheet(0);
    timer.stop();

    if (build_bytes)
      *build_bytes += gpu_resident_bytes();
  }

  // overload used by the point/range benchmarks, which pass no separate
  // max_size / memory budget
  void build(const key_type *keys, size_t size, double *build_time_ms,
             size_t *build_bytes) {
    build(keys, size, size, std::numeric_limits<size_t>::max(), build_time_ms,
          build_bytes);
  }

  void destroy() {
    dictionary_.reset();
    build_values_buffer_.free();
    lookup_found_buffer_.free();
  }

  void lookup(const key_type *keys, value_type *result, size_t size,
              cudaStream_t stream) {
    ensure_built();
    if (size == 0)
      return;
    if (lookup_found_buffer_.num_elements < size) {
      lookup_found_buffer_.resize(size);
    }

    DeviceLookupBatch batch;
    batch.queries = reinterpret_cast<const std::uint32_t *>(keys);
    batch.count = size;
    batch.out_values = reinterpret_cast<std::uint32_t *>(result);
    batch.out_found = lookup_found_buffer_.ptr();
    dictionary_->lookup(batch, stream);
    gpulsmopt_adapter_detail::finalize_lookup(lookup_found_buffer_.ptr(),
                                              result, size, stream);
  }

  void multi_lookup_sum(const key_type *keys, value_type *result, size_t size,
                        cudaStream_t stream) {
    lookup(keys, result, size, stream);
  }

  void range_lookup_sum(const key_type *lower, const key_type *upper,
                        value_type *result, size_t size, cudaStream_t stream) {
    ensure_built();
    if (size == 0)
      return;
    DeviceRangeOutputBatch batch;
    batch.lo = reinterpret_cast<const std::uint32_t *>(lower);
    batch.hi = reinterpret_cast<const std::uint32_t *>(upper);
    batch.query_count = size;
    batch.out_sums = reinterpret_cast<std::uint32_t *>(result);
    batch.out_counts = nullptr;
    dictionary_->range(batch, stream);
  }

  void insert(const key_type *update_list, const smallsize *offsets,
              size_t size, cudaStream_t stream) {
    ensure_built();
    DeviceKeyValueBatch batch;
    batch.keys = reinterpret_cast<const std::uint32_t *>(update_list);
    batch.values = reinterpret_cast<const std::uint32_t *>(offsets);
    batch.count = size;
    dictionary_->insert(batch, stream);
  }

  void remove(const key_type *update_list, size_t size, cudaStream_t stream) {
    ensure_built();
    GPULSMOpt::DeviceKeyBatch batch;
    batch.keys = reinterpret_cast<const std::uint32_t *>(update_list);
    batch.count = size;
    dictionary_->erase(batch, stream);
  }

  void lookups_successor(const key_type *keys, key_type *result, size_t size,
                         cudaStream_t stream) {
    ensure_built();
    if (size == 0)
      return;
    DeviceSuccessorBatch batch;
    batch.queries = reinterpret_cast<const std::uint32_t *>(keys);
    batch.count = size;
    batch.out_keys = reinterpret_cast<std::uint32_t *>(result);
    dictionary_->successor(batch, stream);
  }

private:
  void ensure_built() const {
    if (!dictionary_) {
      throw std::runtime_error("GPULSMOpt index used before build");
    }
  }

  std::unique_ptr<GPULSMOpt> dictionary_;
  cuda_buffer<smallsize> build_values_buffer_;
  cuda_buffer<std::uint8_t> lookup_found_buffer_;
};

#endif
