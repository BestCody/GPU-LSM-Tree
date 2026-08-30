#include "GPULSMOpt.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>
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

static void append_u32(std::vector<std::uint8_t> &bytes,
                       std::uint32_t value) {
  for (std::uint32_t position = 0u; position < 4u; ++position)
    bytes.push_back(static_cast<std::uint8_t>(value >> (8u * position)));
}

int main() {
  try {
    const std::vector<std::vector<std::uint8_t>> keys{
        {0x12u, 0x34u, 0x56u, 0x78u},
        {0x12u, 0x34u, 0x56u, 0x78u, 0x42u},
        {0x12u, 0x34u, 0x56u, 0x78u},
        {0x22u, 0x33u, 0x44u, 0x55u},
        {0x33u, 0x44u, 0x55u, 0x66u},
        {0x33u, 0x44u, 0x55u, 0x66u},
        {0x33u, 0x44u, 0x55u, 0x66u},
        {0x55u, 0x66u, 0x77u, 0x88u},
        {0x55u, 0x66u, 0x77u, 0x88u},
        {0xaau, 0xbbu, 0xccu},
        {0xaau, 0xbbu, 0xccu, 0x00u},
        {0x44u, 0x55u, 0x66u, 0x77u}};
    const std::uint32_t words[]{
        10u, 11u, 12u, 0u, 30u, 0u, 31u, 60u, 0u, 50u, 51u, 40u};
    const std::uint8_t operations[]{
        0u, 0u, 0u, 0u, 0u, 1u, 0u, 0u, 1u, 0u, 0u, 0u};

    std::vector<std::uint8_t> key_bytes, value_bytes;
    std::vector<std::uint64_t> key_offsets(keys.size() + 1u);
    std::vector<std::uint64_t> value_offsets(keys.size() + 1u);
    for (std::size_t row = 0u; row < keys.size(); ++row) {
      key_offsets[row] = key_bytes.size();
      key_bytes.insert(key_bytes.end(), keys[row].begin(), keys[row].end());
      value_offsets[row] = value_bytes.size();
      if (operations[row]) continue;
      if (row == 3u) {
        const std::uint8_t long_value[]{1u, 2u, 3u, 4u, 5u, 6u};
        value_bytes.insert(value_bytes.end(), std::begin(long_value),
                           std::end(long_value));
      } else {
        append_u32(value_bytes, words[row]);
      }
    }
    key_offsets.back() = key_bytes.size();
    value_offsets.back() = value_bytes.size();

    DeviceArray<std::uint8_t> device_keys(key_bytes.size());
    DeviceArray<std::uint64_t> device_key_offsets(key_offsets.size());
    DeviceArray<std::uint8_t> device_values(value_bytes.size());
    DeviceArray<std::uint64_t> device_value_offsets(value_offsets.size());
    DeviceArray<std::uint8_t> device_operations(keys.size());
    CUDA_CHECK(cudaMemcpy(device_keys.pointer, key_bytes.data(),
                          key_bytes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_key_offsets.pointer, key_offsets.data(),
                          key_offsets.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_values.pointer, value_bytes.data(),
                          value_bytes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_value_offsets.pointer,
                          value_offsets.data(),
                          value_offsets.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_operations.pointer, operations,
                          sizeof(operations), cudaMemcpyHostToDevice));

    DictionaryConfig config{};
    config.max_elements = 1024u;
    config.batch_capacity = 16u;
    config.level_zero_capacity = 64u;
    GPULSMOpt dictionary(config);
    const auto fresh_memory = dictionary.sparse_memory_accounting();
    if (!fresh_memory.manifest_bytes ||
        fresh_memory.overlay_metadata_bytes || fresh_memory.workspace_bytes ||
        fresh_memory.workspace_high_water_bytes ||
        fresh_memory.capsule_reserved_bytes ||
        fresh_memory.capsule_mapped_bytes || fresh_memory.capsule_live_bytes ||
        fresh_memory.capsule_garbage_bytes || fresh_memory.capsule_segments)
      throw std::runtime_error("fresh sparse memory accounting mismatch");
    DeviceRecordBatchView batch{};
    batch.keys = {device_keys.pointer, device_key_offsets.pointer, 0u};
    batch.values = {
        device_values.pointer, device_value_offsets.pointer, 0u};
    batch.operations = device_operations.pointer;
    batch.count = keys.size();
    batch.key_encoding = DeviceKeyEncoding::ordered_bytes;
    dictionary.bulk_build(batch, 0);
    const auto mixed_memory = dictionary.sparse_memory_accounting();
    if (!mixed_memory.overlay_metadata_bytes ||
        !mixed_memory.workspace_bytes ||
        mixed_memory.workspace_high_water_bytes <
            mixed_memory.workspace_bytes ||
        !mixed_memory.capsule_reserved_bytes ||
        !mixed_memory.capsule_mapped_bytes ||
        !mixed_memory.capsule_live_bytes || !mixed_memory.capsule_segments ||
        mixed_memory.capsule_live_bytes > mixed_memory.capsule_mapped_bytes ||
        mixed_memory.capsule_garbage_bytes >
            mixed_memory.capsule_mapped_bytes -
                mixed_memory.capsule_live_bytes)
      throw std::runtime_error("mixed sparse memory accounting mismatch");

    const std::vector<std::vector<std::uint8_t>> queries{
        keys[0], keys[1], keys[3], keys[4], keys[7], keys[9], keys[10],
        keys[11]};
    std::vector<std::uint8_t> query_bytes;
    std::vector<std::uint64_t> query_offsets(queries.size() + 1u);
    for (std::size_t row = 0u; row < queries.size(); ++row) {
      query_offsets[row] = query_bytes.size();
      query_bytes.insert(query_bytes.end(), queries[row].begin(),
                         queries[row].end());
    }
    query_offsets.back() = query_bytes.size();
    DeviceArray<std::uint8_t> device_queries(query_bytes.size());
    DeviceArray<std::uint64_t> device_query_offsets(query_offsets.size());
    DeviceArray<std::uint8_t> output(queries.size() * 8u);
    DeviceArray<std::uint64_t> lengths(queries.size());
    DeviceArray<std::uint8_t> found(queries.size());
    DeviceArray<std::uint8_t> overflow(queries.size());
    CUDA_CHECK(cudaMemcpy(device_queries.pointer, query_bytes.data(),
                          query_bytes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_query_offsets.pointer,
                          query_offsets.data(),
                          query_offsets.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    DeviceLookupBatchView lookup{};
    lookup.queries.keys = {
        device_queries.pointer, device_query_offsets.pointer, 0u};
    lookup.queries.count = queries.size();
    lookup.output.values = {
        output.pointer, nullptr, queries.size() * 8u, 8u,
        DeviceSinkLayout::fixed_stride};
    lookup.output.required_value_lengths = lengths.pointer;
    lookup.output.found = found.pointer;
    lookup.output.overflow = overflow.pointer;
    dictionary.lookup(lookup, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<std::uint8_t> host_output(queries.size() * 8u);
    std::vector<std::uint64_t> host_lengths(queries.size());
    std::vector<std::uint8_t> host_found(queries.size());
    std::vector<std::uint8_t> host_overflow(queries.size());
    CUDA_CHECK(cudaMemcpy(host_output.data(), output.pointer,
                          host_output.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_lengths.data(), lengths.pointer,
                          host_lengths.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_found.data(), found.pointer,
                          host_found.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_overflow.data(), overflow.pointer,
                          host_overflow.size(), cudaMemcpyDeviceToHost));
    const std::uint32_t expected_words[]{12u, 11u, 0u, 31u, 0u, 50u, 51u,
                                         40u};
    for (std::size_t row = 0u; row < queries.size(); ++row) {
      const bool expected_found = row != 4u;
      if (host_found[row] != expected_found || host_overflow[row])
        throw std::runtime_error("mixed bulk found/overflow mismatch");
      if (!expected_found) {
        if (host_lengths[row])
          throw std::runtime_error("mixed bulk tombstone length mismatch");
        continue;
      }
      if (row == 2u) {
        const std::uint8_t expected[]{1u, 2u, 3u, 4u, 5u, 6u};
        if (host_lengths[row] != sizeof(expected) ||
            std::memcmp(host_output.data() + row * 8u, expected,
                        sizeof(expected)))
          throw std::runtime_error("mixed bulk long value mismatch");
      } else {
        std::uint32_t actual = 0u;
        std::memcpy(&actual, host_output.data() + row * 8u,
                    sizeof(actual));
        if (host_lengths[row] != sizeof(actual) ||
            actual != expected_words[row])
          throw std::runtime_error("mixed bulk inline value mismatch");
      }
    }

    // Representation-compatible input must enter the restored native bulk
    // overload directly and remain readable through its protected path.
    const std::uint32_t native_keys[]{7u, 3u, 7u, 9u};
    const std::uint32_t native_values[]{70u, 30u, 71u, 90u};
    DeviceArray<std::uint32_t> device_native_keys(4u);
    DeviceArray<std::uint32_t> device_native_values(4u);
    CUDA_CHECK(cudaMemcpy(device_native_keys.pointer, native_keys,
                          sizeof(native_keys), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_native_values.pointer, native_values,
                          sizeof(native_values), cudaMemcpyHostToDevice));
    GPULSMOpt native(config);
    DeviceRecordBatchView native_batch{};
    native_batch.keys = device_head4_source(device_native_keys.pointer);
    native_batch.values = device_u32_source(device_native_values.pointer);
    native_batch.count = 4u;
    native_batch.key_encoding = DeviceKeyEncoding::head4_words;
    native.bulk_build(native_batch, 0);
    DeviceArray<std::uint32_t> native_query(1u), native_output(1u);
    DeviceArray<std::uint8_t> native_found(1u);
    const std::uint32_t seven = 7u;
    CUDA_CHECK(cudaMemcpy(native_query.pointer, &seven, sizeof(seven),
                          cudaMemcpyHostToDevice));
    native.lookup({native_query.pointer, 1u, native_output.pointer,
                   native_found.pointer}, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::uint32_t native_result = 0u;
    std::uint8_t native_hit = 0u;
    CUDA_CHECK(cudaMemcpy(&native_result, native_output.pointer,
                          sizeof(native_result), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&native_hit, native_found.pointer,
                          sizeof(native_hit), cudaMemcpyDeviceToHost));
    if (!native_hit || native_result != 71u)
      throw std::runtime_error("restored native bulk dispatch mismatch");

    std::puts("specialized glue direct-bulk smoke passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
