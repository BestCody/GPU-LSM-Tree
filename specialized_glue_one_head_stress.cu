#include "GPULSMOpt.cuh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { reset(count); }
  ~DeviceBuffer() { cudaFree(pointer_); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  void reset(std::size_t count) {
    if (pointer_) CUDA_CHECK(cudaFree(pointer_));
    pointer_ = nullptr;
    count_ = count;
    if (count_) CUDA_CHECK(cudaMalloc(&pointer_, count_ * sizeof(T)));
  }

  void copy_from(const std::vector<T> &values) {
    reset(values.size());
    if (!values.empty())
      CUDA_CHECK(cudaMemcpy(pointer_, values.data(),
                            values.size() * sizeof(T),
                            cudaMemcpyHostToDevice));
  }

  T *data() const { return pointer_; }

 private:
  T *pointer_{};
  std::size_t count_{};
};

static std::uint32_t prototype_mix(std::uint32_t value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  return value ^ (value >> 16u);
}

// This is the production-closure prototype's one_head_entropy key pattern.
// The final word is a permutation of logical, so every generated key is
// distinct while all rows share the same four-byte projection head.
static void make_one_head_key(std::uint32_t logical, std::uint8_t *key) {
  const std::uint32_t code = prototype_mix(logical ^ 0x63d83595u);
  for (std::uint32_t position = 0u; position < 12u; ++position)
    key[position] = static_cast<std::uint8_t>(
        prototype_mix(logical ^ (position * 0x9e3779b9u)) >> 24u);
  constexpr std::uint8_t fixed[4]{0x12u, 0x34u, 0x56u, 0x78u};
  std::memcpy(key, fixed, sizeof(fixed));
  key[8] = static_cast<std::uint8_t>(code >> 24u);
  key[9] = static_cast<std::uint8_t>(code >> 16u);
  key[10] = static_cast<std::uint8_t>(code >> 8u);
  key[11] = static_cast<std::uint8_t>(code);
}

int main() {
  try {
    constexpr std::uint32_t rows = 1000003u;
    constexpr std::uint32_t query_count = 4096u;
    std::vector<std::uint8_t> key_bytes(std::size_t{rows} * 12u);
    std::vector<std::uint64_t> key_offsets(std::size_t{rows} + 1u);
    std::vector<std::uint32_t> values(rows);
    std::uint32_t expected_range_sum = 0u;
    for (std::uint32_t row = 0u; row < rows; ++row) {
      key_offsets[row] = std::uint64_t{row} * 12u;
      make_one_head_key(row, key_bytes.data() + std::size_t{row} * 12u);
      values[row] = prototype_mix(row ^ 0x9e3779b9u);
      expected_range_sum += values[row];
    }
    key_offsets[rows] = key_bytes.size();

    std::uint32_t minimum_row = 0u;
    std::uint32_t maximum_row = 0u;
    for (std::uint32_t row = 1u; row < rows; ++row) {
      const auto *key = key_bytes.data() + std::size_t{row} * 12u;
      const auto *minimum =
          key_bytes.data() + std::size_t{minimum_row} * 12u;
      const auto *maximum =
          key_bytes.data() + std::size_t{maximum_row} * 12u;
      if (std::lexicographical_compare(key, key + 12u,
                                       minimum, minimum + 12u))
        minimum_row = row;
      if (std::lexicographical_compare(maximum, maximum + 12u,
                                       key, key + 12u))
        maximum_row = row;
    }

    DeviceBuffer<std::uint8_t> device_keys;
    DeviceBuffer<std::uint64_t> device_offsets;
    DeviceBuffer<std::uint32_t> device_values;
    device_keys.copy_from(key_bytes);
    device_offsets.copy_from(key_offsets);
    device_values.copy_from(values);

    DictionaryConfig config{};
    config.max_elements = 1u << 22u;
    config.batch_capacity = 1u << 20u;
    config.level_zero_capacity = 1u << 24u;
    GPULSMOpt dictionary(config);
    DeviceRecordBatchView input{};
    input.keys = {device_keys.data(), device_offsets.data(), 0u};
    input.values = device_u32_source(device_values.data());
    input.count = rows;
    input.key_encoding = DeviceKeyEncoding::ordered_bytes;

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const auto wall_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start));
    dictionary.bulk_build(input, 0);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const auto wall_stop = std::chrono::steady_clock::now();
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const auto memory = dictionary.sparse_memory_accounting();
    std::uint64_t object_bytes = 0u;
    if (!gpulsm_sparse::capsule_object_bytes(12u, 4u, object_bytes) ||
        memory.capsule_segments != 1u ||
        memory.capsule_live_bytes != std::uint64_t{rows} * object_bytes ||
        memory.capsule_garbage_bytes || !memory.overlay_metadata_bytes ||
        memory.workspace_high_water_bytes < memory.workspace_bytes)
      throw std::runtime_error("one-head sparse accounting mismatch");

    std::vector<std::uint8_t> query_bytes(
        std::size_t{query_count + 1u} * 12u);
    std::vector<std::uint64_t> query_offsets(query_count + 2u);
    std::vector<std::uint32_t> expected(query_count);
    for (std::uint32_t query = 0u; query < query_count; ++query) {
      const std::uint32_t row = query + 1u == query_count
          ? rows - 1u
          : static_cast<std::uint32_t>(
                (std::uint64_t{query} * 2654435761u) % rows);
      query_offsets[query] = std::uint64_t{query} * 12u;
      make_one_head_key(
          row, query_bytes.data() + std::size_t{query} * 12u);
      expected[query] = values[row];
    }
    query_offsets[query_count] = std::uint64_t{query_count} * 12u;
    make_one_head_key(0xf0000000u,
                      query_bytes.data() +
                          std::size_t{query_count} * 12u);
    query_offsets[query_count + 1u] = query_bytes.size();

    DeviceBuffer<std::uint8_t> device_queries;
    DeviceBuffer<std::uint64_t> device_query_offsets;
    DeviceBuffer<std::uint32_t> device_output(query_count + 1u);
    DeviceBuffer<std::uint64_t> device_lengths(query_count + 1u);
    DeviceBuffer<std::uint8_t> device_found(query_count + 1u),
        device_overflow(query_count + 1u);
    device_queries.copy_from(query_bytes);
    device_query_offsets.copy_from(query_offsets);
    DeviceLookupBatchView lookup{};
    lookup.queries.keys = {
        device_queries.data(), device_query_offsets.data(), 0u};
    lookup.queries.count = query_count + 1u;
    lookup.output.values =
        device_u32_sink(device_output.data(), query_count + 1u);
    lookup.output.required_value_lengths = device_lengths.data();
    lookup.output.found = device_found.data();
    lookup.output.overflow = device_overflow.data();
    dictionary.lookup(lookup, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<std::uint32_t> output(query_count + 1u);
    std::vector<std::uint64_t> lengths(query_count + 1u);
    std::vector<std::uint8_t> found(query_count + 1u),
        overflow(query_count + 1u);
    CUDA_CHECK(cudaMemcpy(output.data(), device_output.data(),
                          output.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lengths.data(), device_lengths.data(),
                          lengths.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(found.data(), device_found.data(), found.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(overflow.data(), device_overflow.data(),
                          overflow.size(), cudaMemcpyDeviceToHost));
    for (std::uint32_t query = 0u; query < query_count; ++query)
      if (!found[query] || overflow[query] || lengths[query] != 4u ||
          output[query] != expected[query])
        throw std::runtime_error("one-head exact lookup mismatch");
    if (found[query_count] || overflow[query_count] ||
        lengths[query_count])
      throw std::runtime_error("one-head exact miss mismatch");

    std::vector<std::uint8_t> range_lower(12u), range_upper(12u);
    std::memcpy(range_lower.data(),
                key_bytes.data() + std::size_t{minimum_row} * 12u, 12u);
    std::memcpy(range_upper.data(),
                key_bytes.data() + std::size_t{maximum_row} * 12u, 12u);
    const std::vector<std::uint64_t> range_offsets{0u, 12u};
    DeviceBuffer<std::uint8_t> device_range_lower, device_range_upper;
    DeviceBuffer<std::uint64_t> device_range_lower_offsets,
        device_range_upper_offsets;
    DeviceBuffer<std::uint32_t> device_range_sum(1u);
    DeviceBuffer<std::uint8_t> device_range_valid(1u);
    device_range_lower.copy_from(range_lower);
    device_range_upper.copy_from(range_upper);
    device_range_lower_offsets.copy_from(range_offsets);
    device_range_upper_offsets.copy_from(range_offsets);
    DeviceRangeSumBatchView range{};
    range.queries.lower.keys = {
        device_range_lower.data(), device_range_lower_offsets.data(), 0u};
    range.queries.lower.count = 1u;
    range.queries.upper.keys = {
        device_range_upper.data(), device_range_upper_offsets.data(), 0u};
    range.queries.upper.count = 1u;
    range.out_sums = device_range_sum.data();
    range.valid = device_range_valid.data();
    cudaEvent_t range_start{}, range_stop{};
    CUDA_CHECK(cudaEventCreate(&range_start));
    CUDA_CHECK(cudaEventCreate(&range_stop));
    CUDA_CHECK(cudaEventRecord(range_start));
    dictionary.range(range, 0);
    CUDA_CHECK(cudaEventRecord(range_stop));
    CUDA_CHECK(cudaEventSynchronize(range_stop));
    float range_gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&range_gpu_ms, range_start, range_stop));
    CUDA_CHECK(cudaEventDestroy(range_start));
    CUDA_CHECK(cudaEventDestroy(range_stop));
    std::uint32_t host_range_sum = 0u;
    std::uint8_t host_range_valid = 0u;
    CUDA_CHECK(cudaMemcpy(&host_range_sum, device_range_sum.data(),
                          sizeof(host_range_sum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_range_valid, device_range_valid.data(),
                          sizeof(host_range_valid), cudaMemcpyDeviceToHost));
    if (!host_range_valid || host_range_sum != expected_range_sum)
      throw std::runtime_error("one-head full-range sum mismatch");

    const double wall_ms = std::chrono::duration<double, std::milli>(
        wall_stop - wall_start).count();
    std::printf(
        "specialized glue one-head stress passed rows=%u gpu_ms=%.6f "
        "wall_ms=%.6f range_gpu_ms=%.6f capsule_live_bytes=%llu "
        "workspace_high_water=%llu\n",
        rows, gpu_ms, wall_ms, range_gpu_ms,
        static_cast<unsigned long long>(memory.capsule_live_bytes),
        static_cast<unsigned long long>(
            memory.workspace_high_water_bytes));
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
