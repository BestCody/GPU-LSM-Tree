#include "GPULSMOpt.cuh"

#include <cstdint>
#include <cstring>
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
    constexpr std::uint32_t count = 64u;
    std::vector<std::uint8_t> key_bytes;
    std::vector<std::uint64_t> key_offsets(count + 1u);
    std::vector<std::uint32_t> values(count);
    for (std::uint32_t row = 0u; row < count; ++row) {
      key_offsets[row] = key_bytes.size();
      std::uint32_t key = 0x80000000u + row;
      if (row == 0u || row == 1u) key = 0x12345678u;
      key_bytes.push_back(static_cast<std::uint8_t>(key >> 24u));
      key_bytes.push_back(static_cast<std::uint8_t>(key >> 16u));
      key_bytes.push_back(static_cast<std::uint8_t>(key >> 8u));
      key_bytes.push_back(static_cast<std::uint8_t>(key));
      if (row == 1u) key_bytes.push_back(0x42u);
      values[row] = 1000u + row;
    }
    key_offsets[count] = key_bytes.size();

    DeviceArray<std::uint8_t> device_keys(key_bytes.size());
    DeviceArray<std::uint64_t> device_offsets(key_offsets.size());
    DeviceArray<std::uint32_t> device_values(values.size());
    CUDA_CHECK(cudaMemcpy(device_keys.pointer, key_bytes.data(),
                          key_bytes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_offsets.pointer, key_offsets.data(),
                          key_offsets.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_values.pointer, values.data(),
                          values.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));

    DictionaryConfig config{};
    config.max_elements = 1024u;
    config.batch_capacity = 4u;
    config.level_zero_capacity = 64u;
    GPULSMOpt dictionary(config);
    DeviceRecordBatchView batch{};
    batch.keys = {device_keys.pointer, device_offsets.pointer, 0u};
    batch.values = {
        reinterpret_cast<const std::uint8_t *>(device_values.pointer),
        nullptr, sizeof(std::uint32_t)};
    batch.count = count;
    batch.key_encoding = DeviceKeyEncoding::ordered_bytes;
    dictionary.insert(batch, 0);
    if (dictionary.canonical_carry_status())
      throw std::runtime_error("mixed publication reported failure");

    const std::uint32_t query = 0x80000020u;
    DeviceArray<std::uint32_t> device_query(1u);
    DeviceArray<std::uint32_t> device_output(1u);
    DeviceArray<std::uint8_t> device_found(1u);
    CUDA_CHECK(cudaMemcpy(device_query.pointer, &query, sizeof(query),
                          cudaMemcpyHostToDevice));
    dictionary.lookup(
        {device_query.pointer, 1u, device_output.pointer,
         device_found.pointer},
        0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::uint32_t output = 0u;
    std::uint8_t found = 0u;
    CUDA_CHECK(cudaMemcpy(&output, device_output.pointer, sizeof(output),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&found, device_found.pointer, sizeof(found),
                          cudaMemcpyDeviceToHost));
    if (!found || output != 1032u)
      throw std::runtime_error("ordinary lookup changed by sparse carry");

    const std::uint32_t exact_four = 0x12345678u;
    CUDA_CHECK(cudaMemcpy(device_query.pointer, &exact_four,
                          sizeof(exact_four), cudaMemcpyHostToDevice));
    dictionary.lookup(
        {device_query.pointer, 1u, device_output.pointer,
         device_found.pointer},
        0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&output, device_output.pointer, sizeof(output),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&found, device_found.pointer, sizeof(found),
                          cudaMemcpyDeviceToHost));
    if (!found || output != 1000u)
      throw std::runtime_error(
          "newer long key hid the exact four-byte key");

    const std::vector<std::uint8_t> query_bytes{
        0x12u, 0x34u, 0x56u, 0x78u,
        0x12u, 0x34u, 0x56u, 0x78u, 0x42u,
        0x12u, 0x34u, 0x56u, 0x78u, 0x43u};
    const std::vector<std::uint64_t> query_offsets{0u, 4u, 9u, 14u};
    DeviceArray<std::uint8_t> exact_query_bytes(query_bytes.size());
    DeviceArray<std::uint64_t> exact_query_offsets(query_offsets.size());
    DeviceArray<std::uint32_t> exact_values(3u);
    DeviceArray<std::uint64_t> exact_lengths(3u);
    DeviceArray<std::uint8_t> exact_found(3u);
    DeviceArray<std::uint8_t> exact_overflow(3u);
    CUDA_CHECK(cudaMemcpy(exact_query_bytes.pointer, query_bytes.data(),
                          query_bytes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        exact_query_offsets.pointer, query_offsets.data(),
        query_offsets.size() * sizeof(std::uint64_t),
        cudaMemcpyHostToDevice));
    DeviceLookupBatchView exact_lookup{};
    exact_lookup.queries.keys = {
        exact_query_bytes.pointer, exact_query_offsets.pointer, 0u};
    exact_lookup.queries.count = 3u;
    exact_lookup.output.values =
        device_u32_sink(exact_values.pointer, 3u);
    exact_lookup.output.required_value_lengths = exact_lengths.pointer;
    exact_lookup.output.found = exact_found.pointer;
    exact_lookup.output.overflow = exact_overflow.pointer;
    dictionary.lookup(exact_lookup, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::uint32_t host_exact_values[3]{};
    std::uint64_t host_exact_lengths[3]{};
    std::uint8_t host_exact_found[3]{};
    std::uint8_t host_exact_overflow[3]{};
    CUDA_CHECK(cudaMemcpy(host_exact_values, exact_values.pointer,
                          sizeof(host_exact_values), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_exact_lengths, exact_lengths.pointer,
                          sizeof(host_exact_lengths),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_exact_found, exact_found.pointer,
                          sizeof(host_exact_found), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_exact_overflow, exact_overflow.pointer,
                          sizeof(host_exact_overflow),
                          cudaMemcpyDeviceToHost));
    if (!host_exact_found[0] || !host_exact_found[1] ||
        host_exact_found[2] || host_exact_values[0] != 1000u ||
        host_exact_values[1] != 1001u || host_exact_lengths[0] != 4u ||
        host_exact_lengths[1] != 4u || host_exact_lengths[2] != 0u ||
        host_exact_overflow[0] || host_exact_overflow[1] ||
        host_exact_overflow[2])
      throw std::runtime_error("exact complete-key lookup mismatch");

    const std::vector<std::uint8_t> successor_query_bytes{
        0x12u, 0x34u, 0x56u, 0x78u, 0x40u,
        0x12u, 0x34u, 0x56u, 0x78u, 0x42u,
        0x12u, 0x34u, 0x56u, 0x78u, 0x43u};
    const std::vector<std::uint64_t> successor_query_offsets{
        0u, 5u, 10u, 15u};
    DeviceArray<std::uint8_t> successor_queries(
        successor_query_bytes.size());
    DeviceArray<std::uint64_t> successor_offsets(
        successor_query_offsets.size());
    DeviceArray<std::uint8_t> successor_keys(3u * 8u);
    DeviceArray<std::uint64_t> successor_lengths(3u);
    DeviceArray<std::uint8_t> successor_found(3u);
    DeviceArray<std::uint8_t> successor_overflow(3u);
    CUDA_CHECK(cudaMemcpy(successor_queries.pointer,
                          successor_query_bytes.data(),
                          successor_query_bytes.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(successor_offsets.pointer,
                          successor_query_offsets.data(),
                          successor_query_offsets.size() *
                              sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    DeviceSuccessorBatchView successor_batch{};
    successor_batch.queries.keys = {
        successor_queries.pointer, successor_offsets.pointer, 0u};
    successor_batch.queries.count = 3u;
    successor_batch.output.keys = {
        successor_keys.pointer, nullptr, 3u * 8u, 8u,
        DeviceSinkLayout::fixed_stride};
    successor_batch.output.required_key_lengths =
        successor_lengths.pointer;
    successor_batch.output.found = successor_found.pointer;
    successor_batch.output.overflow = successor_overflow.pointer;
    dictionary.successor(successor_batch, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::uint8_t host_successor_keys[3u * 8u]{};
    std::uint64_t host_successor_lengths[3]{};
    std::uint8_t host_successor_found[3]{};
    std::uint8_t host_successor_overflow[3]{};
    CUDA_CHECK(cudaMemcpy(host_successor_keys, successor_keys.pointer,
                          sizeof(host_successor_keys),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_successor_lengths, successor_lengths.pointer,
                          sizeof(host_successor_lengths),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_successor_found, successor_found.pointer,
                          sizeof(host_successor_found),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_successor_overflow,
                          successor_overflow.pointer,
                          sizeof(host_successor_overflow),
                          cudaMemcpyDeviceToHost));
    const std::uint8_t expected_long[5]{
        0x12u, 0x34u, 0x56u, 0x78u, 0x42u};
    const std::uint8_t expected_next[4]{
        0x80u, 0x00u, 0x00u, 0x02u};
    if (!host_successor_found[0] || !host_successor_found[1] ||
        !host_successor_found[2] || host_successor_overflow[0] ||
        host_successor_overflow[1] || host_successor_overflow[2] ||
        host_successor_lengths[0] != 5u ||
        host_successor_lengths[1] != 5u ||
        host_successor_lengths[2] != 4u ||
        std::memcmp(host_successor_keys, expected_long, 5u) ||
        std::memcmp(host_successor_keys + 8u, expected_long, 5u) ||
        std::memcmp(host_successor_keys + 16u, expected_next, 4u))
      throw std::runtime_error("exact successor refinement mismatch");

    const std::vector<std::uint8_t> range_lower_bytes{
        0x12u, 0x34u, 0x56u, 0x78u,
        0x12u, 0x34u, 0x56u, 0x78u, 0x42u,
        0x12u, 0x34u, 0x56u, 0x78u};
    const std::vector<std::uint64_t> range_lower_offsets{
        0u, 4u, 9u, 13u};
    const std::vector<std::uint8_t> range_upper_bytes{
        0x12u, 0x34u, 0x56u, 0x78u, 0x42u,
        0x12u, 0x34u, 0x56u, 0x78u, 0x42u,
        0x80u, 0x00u, 0x00u, 0x02u};
    const std::vector<std::uint64_t> range_upper_offsets{
        0u, 5u, 10u, 14u};
    DeviceArray<std::uint8_t> range_lowers(range_lower_bytes.size());
    DeviceArray<std::uint64_t> range_lower_index(
        range_lower_offsets.size());
    DeviceArray<std::uint8_t> range_uppers(range_upper_bytes.size());
    DeviceArray<std::uint64_t> range_upper_index(
        range_upper_offsets.size());
    DeviceArray<std::uint32_t> range_sums(3u);
    DeviceArray<std::uint8_t> range_valid(3u);
    CUDA_CHECK(cudaMemcpy(range_lowers.pointer, range_lower_bytes.data(),
                          range_lower_bytes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(range_lower_index.pointer,
                          range_lower_offsets.data(),
                          range_lower_offsets.size() *
                              sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(range_uppers.pointer, range_upper_bytes.data(),
                          range_upper_bytes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(range_upper_index.pointer,
                          range_upper_offsets.data(),
                          range_upper_offsets.size() *
                              sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    DeviceRangeSumBatchView range_batch{};
    range_batch.queries.lower.keys = {
        range_lowers.pointer, range_lower_index.pointer, 0u};
    range_batch.queries.lower.count = 3u;
    range_batch.queries.upper.keys = {
        range_uppers.pointer, range_upper_index.pointer, 0u};
    range_batch.queries.upper.count = 3u;
    range_batch.out_sums = range_sums.pointer;
    range_batch.valid = range_valid.pointer;
    dictionary.range(range_batch, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::uint32_t host_range_sums[3]{};
    std::uint8_t host_range_valid[3]{};
    CUDA_CHECK(cudaMemcpy(host_range_sums, range_sums.pointer,
                          sizeof(host_range_sums), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_range_valid, range_valid.pointer,
                          sizeof(host_range_valid), cudaMemcpyDeviceToHost));
    if (!host_range_valid[0] || !host_range_valid[1] ||
        !host_range_valid[2] || host_range_sums[0] != 2001u ||
        host_range_sums[1] != 1001u || host_range_sums[2] != 3003u)
      throw std::runtime_error("rank-roster exact range mismatch");
    std::puts("specialized glue mixed-publication smoke passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
