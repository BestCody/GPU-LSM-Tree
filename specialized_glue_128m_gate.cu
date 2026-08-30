#ifdef GPULSMOPT_HEADER
#include GPULSMOPT_HEADER
#else
#include "GPULSMOpt.cuh"
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_) CUDA_CHECK(cudaMalloc(&pointer_, count_ * sizeof(T)));
  }
  ~DeviceBuffer() { cudaFree(pointer_); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  T *data() const { return pointer_; }

 private:
  T *pointer_{};
  std::size_t count_{};
};

__global__ void generate_paper_rows(
    std::uint32_t *keys, std::uint32_t *values, std::uint32_t count) {
  for (std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
       row < count; row += blockDim.x * gridDim.x) {
    keys[row] = row * 747796405u + 289133645u;
    values[row] = row + 1u;
  }
}

__global__ void make_sample_queries(
    std::uint32_t *queries, std::uint32_t *expected,
    std::uint32_t count, std::uint32_t source_count) {
  const std::uint32_t query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= count) return;
  const std::uint32_t source = static_cast<std::uint32_t>(
      (static_cast<unsigned long long>(query) * source_count) / count);
  queries[query] = source * 747796405u + 289133645u;
  expected[query] = source + 1u;
}

__global__ void validate_samples(
    const std::uint32_t *output, const std::uint32_t *expected,
    const std::uint8_t *found, std::uint32_t count,
    unsigned long long *errors) {
  const std::uint32_t query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query < count && (!found[query] || output[query] != expected[query]))
    atomicAdd(errors, 1ull);
}

int main() {
  try {
    constexpr std::uint32_t rows = 1u << 27u;
    constexpr std::uint32_t epoch_rows = 1u << 24u;
    constexpr std::uint32_t query_count = 1u << 15u;
    constexpr std::uint32_t threads = 256u;
    DeviceBuffer<std::uint32_t> keys(rows), values(rows);
    DeviceBuffer<std::uint32_t> queries(query_count), expected(query_count),
        output(query_count);
    DeviceBuffer<std::uint8_t> found(query_count);
    DeviceBuffer<unsigned long long> errors(1u);
    generate_paper_rows<<<4096, threads>>>(keys.data(), values.data(), rows);
    make_sample_queries<<<(query_count + threads - 1u) / threads, threads>>>(
        queries.data(), expected.data(), query_count, rows);
    CUDA_CHECK(cudaDeviceSynchronize());

    DictionaryConfig config{};
    config.max_elements = rows;
    config.batch_capacity = 1u << 20u;
    config.level_zero_capacity = epoch_rows;
    GPULSMOpt dictionary(config);
    const std::size_t fresh_bytes = dictionary.gpu_resident_bytes();

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const auto wall_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start));
    dictionary.insert({keys.data(), values.data(), rows}, 0);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const auto wall_stop = std::chrono::steady_clock::now();
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    dictionary.lookup(
        {queries.data(), query_count, output.data(), found.data()}, 0);
    CUDA_CHECK(cudaMemset(errors.data(), 0, sizeof(unsigned long long)));
    validate_samples<<<(query_count + threads - 1u) / threads, threads>>>(
        output.data(), expected.data(), found.data(), query_count,
        errors.data());
    unsigned long long host_errors = 0u;
    CUDA_CHECK(cudaMemcpy(&host_errors, errors.data(), sizeof(host_errors),
                          cudaMemcpyDeviceToHost));
    if (host_errors)
      throw std::runtime_error("128M sampled lookup validation failed");

    const std::size_t resident_bytes = dictionary.gpu_resident_bytes();
#ifndef RESTORED_BASELINE
    const auto sparse = dictionary.sparse_memory_accounting();
    if (sparse.overlay_metadata_bytes || sparse.workspace_bytes ||
        sparse.capsule_mapped_bytes || sparse.capsule_segments)
      throw std::runtime_error("all-inline call retained sparse state");
#endif
    const double wall_ms = std::chrono::duration<double, std::milli>(
        wall_stop - wall_start).count();
#ifndef RESTORED_BASELINE
    std::printf(
        "gate=integrated_sealed_128m rows=%u gpu_ms=%.6f wall_ms=%.6f "
        "fresh_bytes=%zu resident_bytes=%zu sparse_manifest_bytes=%llu "
        "workspace_high_water=%llu errors=%llu\n",
        rows, gpu_ms, wall_ms, fresh_bytes, resident_bytes,
        static_cast<unsigned long long>(sparse.manifest_bytes),
        static_cast<unsigned long long>(
            sparse.workspace_high_water_bytes),
        host_errors);
#else
    std::printf(
        "gate=restored_sealed_128m rows=%u gpu_ms=%.6f wall_ms=%.6f "
        "fresh_bytes=%zu resident_bytes=%zu errors=%llu\n",
        rows, gpu_ms, wall_ms, fresh_bytes, resident_bytes, host_errors);
#endif
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
