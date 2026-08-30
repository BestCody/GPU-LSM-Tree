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

static std::uint32_t mix32(std::uint32_t value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  return value ^ (value >> 16u);
}

static void verify(GPULSMOpt &dictionary,
                   const std::vector<std::uint32_t> &keys,
                   const std::vector<std::uint32_t> &values) {
  DeviceArray<std::uint32_t> queries(keys.size()), output(keys.size());
  DeviceArray<std::uint8_t> found(keys.size());
  CUDA_CHECK(cudaMemcpy(queries.pointer, keys.data(),
                        keys.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice));
  dictionary.lookup(
      {queries.pointer, keys.size(), output.pointer, found.pointer}, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::uint32_t> actual(keys.size());
  std::vector<std::uint8_t> hits(keys.size());
  CUDA_CHECK(cudaMemcpy(actual.data(), output.pointer,
                        actual.size() * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hits.data(), found.pointer, hits.size(),
                        cudaMemcpyDeviceToHost));
  for (std::size_t row = 0u; row < keys.size(); ++row)
    if (!hits[row] || actual[row] != values[row])
      throw std::runtime_error("sealed lookup mismatch");
}

int main() {
  try {
    constexpr std::uint32_t batch_rows = 16u;
    constexpr std::uint32_t epoch_rows = 16u * batch_rows;
    DictionaryConfig config{};
    config.max_elements = 4096u;
    config.batch_capacity = batch_rows;
    config.level_zero_capacity = epoch_rows;

    // Eight complete epochs produce two final radix-4 roots directly from
    // symbolic raw intervals.
    constexpr std::uint32_t direct_rows = 8u * epoch_rows;
    std::vector<std::uint32_t> direct_keys(direct_rows);
    std::vector<std::uint32_t> direct_values(direct_rows);
    for (std::uint32_t row = 0u; row < direct_rows; ++row) {
      direct_keys[row] = mix32(row + 1u);
      direct_values[row] = row + 100u;
    }
    DeviceArray<std::uint32_t> device_direct_keys(direct_rows);
    DeviceArray<std::uint32_t> device_direct_values(direct_rows);
    CUDA_CHECK(cudaMemcpy(device_direct_keys.pointer, direct_keys.data(),
                          direct_rows * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_direct_values.pointer,
                          direct_values.data(),
                          direct_rows * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    GPULSMOpt direct(config);
    direct.insert({device_direct_keys.pointer, device_direct_values.pointer,
                   direct_rows}, 0);
    verify(direct, direct_keys, direct_values);

    // Three ordinary epochs fill the first tier.  The next epoch must use
    // the planner's resident-source root and the original canonical carry.
    constexpr std::uint32_t seed_rows = 3u * epoch_rows;
    std::vector<std::uint32_t> seed_keys(seed_rows);
    std::vector<std::uint32_t> seed_values(seed_rows);
    for (std::uint32_t row = 0u; row < seed_rows; ++row) {
      seed_keys[row] = row;
      seed_values[row] = row + 1u;
    }
    DeviceArray<std::uint32_t> device_seed_keys(seed_rows);
    DeviceArray<std::uint32_t> device_seed_values(seed_rows);
    CUDA_CHECK(cudaMemcpy(device_seed_keys.pointer, seed_keys.data(),
                          seed_rows * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_seed_values.pointer, seed_values.data(),
                          seed_rows * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    GPULSMOpt carry(config);
    carry.insert({device_seed_keys.pointer, device_seed_values.pointer,
                  seed_rows}, 0);

    std::vector<std::uint32_t> incoming_keys(epoch_rows);
    std::vector<std::uint32_t> incoming_values(epoch_rows);
    for (std::uint32_t row = 0u; row < epoch_rows; ++row) {
      incoming_keys[row] = row < epoch_rows / 2u
          ? row : 1000u + row;
      incoming_values[row] = 10000u + row;
    }
    DeviceArray<std::uint32_t> device_incoming_keys(epoch_rows);
    DeviceArray<std::uint32_t> device_incoming_values(epoch_rows);
    CUDA_CHECK(cudaMemcpy(device_incoming_keys.pointer,
                          incoming_keys.data(),
                          epoch_rows * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_incoming_values.pointer,
                          incoming_values.data(),
                          epoch_rows * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    carry.insert({device_incoming_keys.pointer,
                  device_incoming_values.pointer, epoch_rows}, 0);

    std::vector<std::uint32_t> expected_keys;
    std::vector<std::uint32_t> expected_values;
    for (std::uint32_t row = 0u; row < seed_rows; ++row) {
      expected_keys.push_back(row);
      expected_values.push_back(row < epoch_rows / 2u
          ? 10000u + row : row + 1u);
    }
    for (std::uint32_t row = epoch_rows / 2u; row < epoch_rows; ++row) {
      expected_keys.push_back(1000u + row);
      expected_values.push_back(10000u + row);
    }
    verify(carry, expected_keys, expected_values);

    // A partial pending prefix is completed by the restored epoch path.  The
    // remaining four epochs force a plan where a carry must read level 2
    // before a newer final root reuses that physical slot.
    GPULSMOpt partial(config);
    std::vector<std::uint32_t> prefix_keys(7u), prefix_values(7u);
    for (std::uint32_t row = 0u; row < 7u; ++row) {
      prefix_keys[row] = 20000u + row;
      prefix_values[row] = 30000u + row;
    }
    DeviceArray<std::uint32_t> device_prefix_keys(prefix_keys.size());
    DeviceArray<std::uint32_t> device_prefix_values(prefix_values.size());
    CUDA_CHECK(cudaMemcpy(device_prefix_keys.pointer, prefix_keys.data(),
                          prefix_keys.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_prefix_values.pointer,
                          prefix_values.data(),
                          prefix_values.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    partial.insert({device_prefix_keys.pointer, device_prefix_values.pointer,
                    prefix_keys.size()}, 0);
    constexpr std::uint32_t completion_rows = 15u * batch_rows;
    constexpr std::uint32_t fused_rows = 4u * epoch_rows;
    std::vector<std::uint32_t> suffix_keys(completion_rows + fused_rows);
    std::vector<std::uint32_t> suffix_values(suffix_keys.size());
    for (std::uint32_t row = 0u; row < suffix_keys.size(); ++row) {
      suffix_keys[row] = 21000u + row;
      suffix_values[row] = 31000u + row;
    }
    DeviceArray<std::uint32_t> device_suffix_keys(suffix_keys.size());
    DeviceArray<std::uint32_t> device_suffix_values(suffix_values.size());
    CUDA_CHECK(cudaMemcpy(device_suffix_keys.pointer, suffix_keys.data(),
                          suffix_keys.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_suffix_values.pointer,
                          suffix_values.data(),
                          suffix_values.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    partial.insert({device_suffix_keys.pointer, device_suffix_values.pointer,
                    suffix_keys.size()}, 0);
    prefix_keys.insert(prefix_keys.end(), suffix_keys.begin(),
                       suffix_keys.end());
    prefix_values.insert(prefix_values.end(), suffix_values.begin(),
                         suffix_values.end());
    verify(partial, prefix_keys, prefix_values);

    // Seed an exact sidecar at level 2, then consume it with three ordinary
    // epochs.  The original carry moves one placeholder while sparse fan-in
    // preserves the old long key and lets the newer four-byte update win.
    GPULSMOpt sparse_carry(config);
    const std::vector<std::uint8_t> mixed_key_bytes{
        0x12u, 0x34u, 0x56u, 0x78u,
        0x12u, 0x34u, 0x56u, 0x78u, 0x42u};
    const std::uint64_t mixed_key_offsets[]{0u, 4u, 9u};
    const std::uint32_t mixed_values[]{10u, 11u};
    DeviceArray<std::uint8_t> device_mixed_keys(mixed_key_bytes.size());
    DeviceArray<std::uint64_t> device_mixed_offsets(3u);
    DeviceArray<std::uint32_t> device_mixed_values(2u);
    CUDA_CHECK(cudaMemcpy(device_mixed_keys.pointer,
                          mixed_key_bytes.data(), mixed_key_bytes.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_mixed_offsets.pointer, mixed_key_offsets,
                          sizeof(mixed_key_offsets),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_mixed_values.pointer, mixed_values,
                          sizeof(mixed_values), cudaMemcpyHostToDevice));
    DeviceRecordBatchView mixed_seed{};
    mixed_seed.keys = {
        device_mixed_keys.pointer, device_mixed_offsets.pointer, 0u};
    mixed_seed.values = device_u32_source(device_mixed_values.pointer);
    mixed_seed.count = 2u;
    sparse_carry.bulk_build(mixed_seed, 0);

    constexpr std::uint32_t sparse_suffix_rows = 3u * epoch_rows;
    std::vector<std::uint32_t> sparse_suffix_keys(sparse_suffix_rows);
    std::vector<std::uint32_t> sparse_suffix_values(sparse_suffix_rows);
    for (std::uint32_t row = 0u; row < sparse_suffix_rows; ++row) {
      sparse_suffix_keys[row] = 0x70000000u + row;
      sparse_suffix_values[row] = 40000u + row;
    }
    sparse_suffix_keys[100u] = 0x12345678u;
    sparse_suffix_values[100u] = 9999u;
    DeviceArray<std::uint32_t> device_sparse_suffix_keys(
        sparse_suffix_rows);
    DeviceArray<std::uint32_t> device_sparse_suffix_values(
        sparse_suffix_rows);
    CUDA_CHECK(cudaMemcpy(device_sparse_suffix_keys.pointer,
                          sparse_suffix_keys.data(),
                          sparse_suffix_rows * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_sparse_suffix_values.pointer,
                          sparse_suffix_values.data(),
                          sparse_suffix_rows * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    sparse_carry.insert({device_sparse_suffix_keys.pointer,
                         device_sparse_suffix_values.pointer,
                         sparse_suffix_rows}, 0);

    const std::vector<std::uint8_t> exact_queries{
        0x12u, 0x34u, 0x56u, 0x78u,
        0x12u, 0x34u, 0x56u, 0x78u, 0x42u};
    const std::uint64_t exact_offsets[]{0u, 4u, 9u};
    DeviceArray<std::uint8_t> device_exact_queries(exact_queries.size());
    DeviceArray<std::uint64_t> device_exact_offsets(3u);
    DeviceArray<std::uint32_t> exact_output(2u);
    DeviceArray<std::uint64_t> exact_lengths(2u);
    DeviceArray<std::uint8_t> exact_found(2u), exact_overflow(2u);
    CUDA_CHECK(cudaMemcpy(device_exact_queries.pointer,
                          exact_queries.data(), exact_queries.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_exact_offsets.pointer, exact_offsets,
                          sizeof(exact_offsets), cudaMemcpyHostToDevice));
    DeviceLookupBatchView exact_lookup{};
    exact_lookup.queries.keys = {
        device_exact_queries.pointer, device_exact_offsets.pointer, 0u};
    exact_lookup.queries.count = 2u;
    exact_lookup.output.values = device_u32_sink(exact_output.pointer, 2u);
    exact_lookup.output.required_value_lengths = exact_lengths.pointer;
    exact_lookup.output.found = exact_found.pointer;
    exact_lookup.output.overflow = exact_overflow.pointer;
    sparse_carry.lookup(exact_lookup, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::uint32_t host_exact_output[2]{};
    std::uint8_t host_exact_found[2]{};
    CUDA_CHECK(cudaMemcpy(host_exact_output, exact_output.pointer,
                          sizeof(host_exact_output),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_exact_found, exact_found.pointer,
                          sizeof(host_exact_found),
                          cudaMemcpyDeviceToHost));
    if (!host_exact_found[0] || !host_exact_found[1] ||
        host_exact_output[0] != 9999u || host_exact_output[1] != 11u)
      throw std::runtime_error("sealed sparse fan-in mismatch");

    // The inverse carry uses a mixed symbolic interval as the newer source.
    // Its observed-head projection feeds the restored carry, while the
    // direct sparse roster and the resident sidecar feed exact fan-in.
    GPULSMOpt mixed_sealed_carry(config);
    mixed_sealed_carry.bulk_build(mixed_seed, 0);
    std::vector<std::uint8_t> mixed_sealed_keys;
    std::vector<std::uint64_t> mixed_sealed_offsets(
        sparse_suffix_rows + 1u);
    std::vector<std::uint32_t> mixed_sealed_values(sparse_suffix_rows);
    mixed_sealed_keys.reserve(
        std::size_t{sparse_suffix_rows} * sizeof(std::uint32_t) + 1u);
    for (std::uint32_t row = 0u; row < sparse_suffix_rows; ++row) {
      mixed_sealed_offsets[row] = mixed_sealed_keys.size();
      std::uint32_t head = 0x71000000u + row;
      std::uint8_t suffix = 0u;
      if (row == 100u) head = 0x12345678u;
      if (row == 200u) {
        head = 0x12345678u;
        suffix = 0x43u;
      }
      mixed_sealed_keys.push_back(static_cast<std::uint8_t>(head >> 24u));
      mixed_sealed_keys.push_back(static_cast<std::uint8_t>(head >> 16u));
      mixed_sealed_keys.push_back(static_cast<std::uint8_t>(head >> 8u));
      mixed_sealed_keys.push_back(static_cast<std::uint8_t>(head));
      if (suffix) mixed_sealed_keys.push_back(suffix);
      mixed_sealed_values[row] = row == 100u ? 9999u
          : row == 200u ? 8888u : 50000u + row;
    }
    mixed_sealed_offsets[sparse_suffix_rows] = mixed_sealed_keys.size();
    DeviceArray<std::uint8_t> device_mixed_sealed_keys(
        mixed_sealed_keys.size());
    DeviceArray<std::uint64_t> device_mixed_sealed_offsets(
        mixed_sealed_offsets.size());
    DeviceArray<std::uint32_t> device_mixed_sealed_values(
        mixed_sealed_values.size());
    CUDA_CHECK(cudaMemcpy(device_mixed_sealed_keys.pointer,
                          mixed_sealed_keys.data(),
                          mixed_sealed_keys.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_mixed_sealed_offsets.pointer,
                          mixed_sealed_offsets.data(),
                          mixed_sealed_offsets.size() *
                              sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_mixed_sealed_values.pointer,
                          mixed_sealed_values.data(),
                          mixed_sealed_values.size() *
                              sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    DeviceRecordBatchView mixed_sealed_batch{};
    mixed_sealed_batch.keys = {
        device_mixed_sealed_keys.pointer,
        device_mixed_sealed_offsets.pointer, 0u};
    mixed_sealed_batch.values =
        device_u32_source(device_mixed_sealed_values.pointer);
    mixed_sealed_batch.count = sparse_suffix_rows;
    mixed_sealed_carry.insert(mixed_sealed_batch, 0);

    const std::vector<std::uint8_t> mixed_sealed_queries{
        0x12u, 0x34u, 0x56u, 0x78u,
        0x12u, 0x34u, 0x56u, 0x78u, 0x42u,
        0x12u, 0x34u, 0x56u, 0x78u, 0x43u};
    const std::uint64_t mixed_sealed_query_offsets[]{0u, 4u, 9u, 14u};
    DeviceArray<std::uint8_t> device_mixed_sealed_queries(
        mixed_sealed_queries.size());
    DeviceArray<std::uint64_t> device_mixed_sealed_query_offsets(4u);
    DeviceArray<std::uint32_t> mixed_sealed_output(3u);
    DeviceArray<std::uint8_t> mixed_sealed_found(3u);
    CUDA_CHECK(cudaMemcpy(device_mixed_sealed_queries.pointer,
                          mixed_sealed_queries.data(),
                          mixed_sealed_queries.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_mixed_sealed_query_offsets.pointer,
                          mixed_sealed_query_offsets,
                          sizeof(mixed_sealed_query_offsets),
                          cudaMemcpyHostToDevice));
    DeviceLookupBatchView mixed_sealed_lookup{};
    mixed_sealed_lookup.queries.keys = {
        device_mixed_sealed_queries.pointer,
        device_mixed_sealed_query_offsets.pointer, 0u};
    mixed_sealed_lookup.queries.count = 3u;
    mixed_sealed_lookup.output.values =
        device_u32_sink(mixed_sealed_output.pointer, 3u);
    mixed_sealed_lookup.output.found = mixed_sealed_found.pointer;
    mixed_sealed_carry.lookup(mixed_sealed_lookup, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::uint32_t host_mixed_sealed_output[3]{};
    std::uint8_t host_mixed_sealed_found[3]{};
    CUDA_CHECK(cudaMemcpy(host_mixed_sealed_output,
                          mixed_sealed_output.pointer,
                          sizeof(host_mixed_sealed_output),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_mixed_sealed_found,
                          mixed_sealed_found.pointer,
                          sizeof(host_mixed_sealed_found),
                          cudaMemcpyDeviceToHost));
    if (!host_mixed_sealed_found[0] ||
        !host_mixed_sealed_found[1] ||
        !host_mixed_sealed_found[2] ||
        host_mixed_sealed_output[0] != 9999u ||
        host_mixed_sealed_output[1] != 11u ||
        host_mixed_sealed_output[2] != 8888u)
      throw std::runtime_error("mixed sealed sparse fan-in mismatch");

    std::puts("specialized glue sealed-forest smoke passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
