#include "GPULSMOpt.cuh"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

template <class T>
struct DeviceArray {
  T *pointer{};
  explicit DeviceArray(std::size_t count) {
    CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
  }
  ~DeviceArray() { cudaFree(pointer); }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
};

int main() {
  try {
    constexpr std::uint32_t batch_capacity = 1u << 19u;
    constexpr std::uint32_t query_count = 1u << 18u;

    const std::vector<std::uint8_t> root_key_bytes{
        0x12u, 0x34u, 0x56u, 0x78u,
        0x12u, 0x34u, 0x56u, 0x78u, 0x42u,
        0x20u, 0x00u, 0x00u, 0x02u,
        0x20u, 0x00u, 0x00u, 0x03u,
        0x20u, 0x00u, 0x00u, 0x04u,
        0x20u, 0x00u, 0x00u, 0x05u,
        0x20u, 0x00u, 0x00u, 0x06u,
        0x20u, 0x00u, 0x00u, 0x07u,
        0x20u, 0x00u, 0x00u, 0x08u,
        0x20u, 0x00u, 0x00u, 0x09u,
        0x20u, 0x00u, 0x00u, 0x0au,
        0x20u, 0x00u, 0x00u, 0x0bu,
        0x20u, 0x00u, 0x00u, 0x0cu,
        0x20u, 0x00u, 0x00u, 0x0du,
        0x20u, 0x00u, 0x00u, 0x0eu,
        0x20u, 0x00u, 0x00u, 0x0fu};
    std::vector<std::uint64_t> root_offsets(17u);
    root_offsets[0] = 0u;
    root_offsets[1] = 4u;
    root_offsets[2] = 9u;
    for (std::uint32_t row = 3u; row <= 16u; ++row)
      root_offsets[row] = 9u + std::uint64_t{row - 2u} * 4u;
    std::vector<std::uint32_t> root_values(16u);
    for (std::uint32_t row = 0u; row < 16u; ++row)
      root_values[row] = 1000u + row;
    DeviceArray<std::uint8_t> device_root_keys(root_key_bytes.size());
    DeviceArray<std::uint64_t> device_root_offsets(root_offsets.size());
    DeviceArray<std::uint32_t> device_root_values(root_values.size());
    CUDA_CHECK(cudaMemcpy(device_root_keys.pointer, root_key_bytes.data(),
                          root_key_bytes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_root_offsets.pointer, root_offsets.data(),
                          root_offsets.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_root_values.pointer, root_values.data(),
                          root_values.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));

    DictionaryConfig config{};
    config.max_elements = 4u * 1024u * 1024u;
    config.batch_capacity = batch_capacity;
    config.level_zero_capacity = 4u * 1024u * 1024u;
    GPULSMOpt dictionary(config);
    for (std::uint32_t row = 0u; row < 16u; ++row) {
      DeviceRecordBatchView input{};
      input.keys = {
          device_root_keys.pointer, device_root_offsets.pointer + row, 0u};
      input.values = {
          reinterpret_cast<const std::uint8_t *>(
              device_root_values.pointer + row),
          nullptr, sizeof(std::uint32_t)};
      input.count = 1u;
      dictionary.insert(input, 0);
    }
    if (dictionary.canonical_carry_status())
      throw std::runtime_error("TQRJ root publication failed");

    std::vector<std::uint32_t> pending_keys(batch_capacity);
    std::vector<std::uint32_t> pending_values(batch_capacity);
    for (std::uint32_t row = 0u; row < batch_capacity; ++row) {
      pending_keys[row] = 0x80000000u + row;
      pending_values[row] = 0x50000000u + row;
    }
    DeviceArray<std::uint32_t> device_pending_keys(batch_capacity);
    DeviceArray<std::uint32_t> device_pending_values(batch_capacity);
    CUDA_CHECK(cudaMemcpy(device_pending_keys.pointer, pending_keys.data(),
                          pending_keys.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_pending_values.pointer,
                          pending_values.data(),
                          pending_values.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    for (std::uint32_t batch = 0u; batch < 5u; ++batch)
      dictionary.insert(
          {device_pending_keys.pointer, device_pending_values.pointer,
           batch_capacity},
          0);

    std::vector<std::uint32_t> direct_queries(query_count);
    for (std::uint32_t row = 0u; row < query_count; ++row)
      direct_queries[row] = ((row & 0xffffu) << 16u) | (row >> 16u);
    direct_queries[0] = 0x12345678u;
    DeviceArray<std::uint32_t> device_direct_queries(query_count);
    DeviceArray<std::uint32_t> device_direct_values(query_count);
    DeviceArray<std::uint8_t> device_direct_found(query_count);
    CUDA_CHECK(cudaMemcpy(device_direct_queries.pointer,
                          direct_queries.data(),
                          direct_queries.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    dictionary.lookup(
        {device_direct_queries.pointer, query_count,
         device_direct_values.pointer, device_direct_found.pointer},
        0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::uint32_t direct_value = 0u;
    std::uint8_t direct_found = 0u;
    CUDA_CHECK(cudaMemcpy(&direct_value, device_direct_values.pointer,
                          sizeof(direct_value), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&direct_found, device_direct_found.pointer,
                          sizeof(direct_found), cudaMemcpyDeviceToHost));
    if (!direct_found || direct_value != 1000u)
      throw std::runtime_error("sparse TQRJ direct exact lookup failed");

    std::vector<std::uint8_t> hash_query_bytes(
        std::size_t{query_count} * 5u);
    std::vector<std::uint64_t> hash_query_offsets(query_count + 1u);
    for (std::uint32_t row = 0u; row < query_count; ++row) {
      const std::size_t begin = std::size_t{row} * 5u;
      hash_query_offsets[row] = begin;
      hash_query_bytes[begin] = 0x12u;
      hash_query_bytes[begin + 1u] = 0x34u;
      hash_query_bytes[begin + 2u] = 0x56u;
      hash_query_bytes[begin + 3u] = 0x78u;
      hash_query_bytes[begin + 4u] = 0x42u;
    }
    hash_query_offsets[query_count] = hash_query_bytes.size();
    DeviceArray<std::uint8_t> device_hash_query_bytes(
        hash_query_bytes.size());
    DeviceArray<std::uint64_t> device_hash_query_offsets(
        hash_query_offsets.size());
    DeviceArray<std::uint32_t> device_hash_values(query_count);
    DeviceArray<std::uint8_t> device_hash_found(query_count);
    CUDA_CHECK(cudaMemcpy(device_hash_query_bytes.pointer,
                          hash_query_bytes.data(), hash_query_bytes.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_hash_query_offsets.pointer,
                          hash_query_offsets.data(),
                          hash_query_offsets.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    DeviceLookupBatchView hash_lookup{};
    hash_lookup.queries.keys = {
        device_hash_query_bytes.pointer,
        device_hash_query_offsets.pointer, 0u};
    hash_lookup.queries.count = query_count;
    hash_lookup.output.values =
        device_u32_sink(device_hash_values.pointer, query_count);
    hash_lookup.output.found = device_hash_found.pointer;

    // Run the identical complete key through canonical and TQRJ-direct
    // dispatch before forcing the same quotient into TQRJ overflow/hash.
    DeviceLookupBatchView single_exact = hash_lookup;
    single_exact.queries.count = 1u;
    single_exact.output.values =
        device_u32_sink(device_hash_values.pointer, 1u);
    dictionary.lookup(single_exact, 0, false);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::uint32_t canonical_value = 0u;
    std::uint8_t canonical_found = 0u;
    CUDA_CHECK(cudaMemcpy(&canonical_value, device_hash_values.pointer,
                          sizeof(canonical_value), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&canonical_found, device_hash_found.pointer,
                          sizeof(canonical_found), cudaMemcpyDeviceToHost));
    dictionary.lookup(single_exact, 0, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::uint32_t direct_exact_value = 0u;
    std::uint8_t direct_exact_found = 0u;
    CUDA_CHECK(cudaMemcpy(&direct_exact_value, device_hash_values.pointer,
                          sizeof(direct_exact_value),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&direct_exact_found, device_hash_found.pointer,
                          sizeof(direct_exact_found),
                          cudaMemcpyDeviceToHost));
    if (!canonical_found || !direct_exact_found ||
        canonical_value != 1001u || direct_exact_value != 1001u)
      throw std::runtime_error(
          "canonical/TQRJ-direct exact lookup disagreement");

    dictionary.lookup(hash_lookup, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::uint32_t hash_value = 0u;
    std::uint8_t hash_found = 0u;
    CUDA_CHECK(cudaMemcpy(&hash_value, device_hash_values.pointer,
                          sizeof(hash_value), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hash_found, device_hash_found.pointer,
                          sizeof(hash_found), cudaMemcpyDeviceToHost));
    if (!hash_found || hash_value != canonical_value ||
        hash_value != direct_exact_value)
      throw std::runtime_error(
          "canonical/direct/hash exact lookup disagreement");

    std::puts("specialized glue sparse TQRJ smoke passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
