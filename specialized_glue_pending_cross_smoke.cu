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

struct PackedRecords {
  std::vector<std::uint8_t> keys;
  std::vector<std::uint64_t> offsets{0u};
  std::vector<std::uint32_t> values;

  void push(std::uint32_t head, std::uint32_t value,
            int suffix = -1) {
    keys.push_back(static_cast<std::uint8_t>(head >> 24u));
    keys.push_back(static_cast<std::uint8_t>(head >> 16u));
    keys.push_back(static_cast<std::uint8_t>(head >> 8u));
    keys.push_back(static_cast<std::uint8_t>(head));
    if (suffix >= 0)
      keys.push_back(static_cast<std::uint8_t>(suffix));
    offsets.push_back(keys.size());
    values.push_back(value);
  }
};

static void insert_packed(GPULSMOpt &dictionary,
                          const PackedRecords &records) {
  DeviceArray<std::uint8_t> keys(records.keys.size());
  DeviceArray<std::uint64_t> offsets(records.offsets.size());
  DeviceArray<std::uint32_t> values(records.values.size());
  CUDA_CHECK(cudaMemcpy(keys.pointer, records.keys.data(),
                        records.keys.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(offsets.pointer, records.offsets.data(),
                        records.offsets.size() * sizeof(std::uint64_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(values.pointer, records.values.data(),
                        records.values.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice));
  DeviceRecordBatchView batch{};
  batch.keys = {keys.pointer, offsets.pointer, 0u};
  batch.values = device_u32_source(values.pointer);
  batch.count = records.values.size();
  dictionary.insert(batch, 0);
}

static void verify_packed(GPULSMOpt &dictionary,
                          const PackedRecords &records) {
  DeviceArray<std::uint8_t> keys(records.keys.size());
  DeviceArray<std::uint64_t> offsets(records.offsets.size());
  DeviceArray<std::uint32_t> output(records.values.size());
  DeviceArray<std::uint8_t> found(records.values.size());
  CUDA_CHECK(cudaMemcpy(keys.pointer, records.keys.data(),
                        records.keys.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(offsets.pointer, records.offsets.data(),
                        records.offsets.size() * sizeof(std::uint64_t),
                        cudaMemcpyHostToDevice));
  DeviceLookupBatchView lookup{};
  lookup.queries.keys = {keys.pointer, offsets.pointer, 0u};
  lookup.queries.count = records.values.size();
  lookup.output.values =
      device_u32_sink(output.pointer, records.values.size());
  lookup.output.found = found.pointer;
  dictionary.lookup(lookup, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::uint32_t> actual(records.values.size());
  std::vector<std::uint8_t> hits(records.values.size());
  CUDA_CHECK(cudaMemcpy(actual.data(), output.pointer,
                        actual.size() * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hits.data(), found.pointer, hits.size(),
                        cudaMemcpyDeviceToHost));
  for (std::size_t row = 0u; row < records.values.size(); ++row)
    if (!hits[row] || actual[row] != records.values[row])
      throw std::runtime_error("packed pending-cross lookup mismatch");
}

static void insert_inline(GPULSMOpt &dictionary,
                          const std::vector<std::uint32_t> &keys,
                          const std::vector<std::uint32_t> &values) {
  DeviceArray<std::uint32_t> device_keys(keys.size());
  DeviceArray<std::uint32_t> device_values(values.size());
  CUDA_CHECK(cudaMemcpy(device_keys.pointer, keys.data(),
                        keys.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_values.pointer, values.data(),
                        values.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice));
  dictionary.insert(
      {device_keys.pointer, device_values.pointer, keys.size()}, 0);
}

static void verify_inline(GPULSMOpt &dictionary,
                          const std::vector<std::uint32_t> &keys,
                          const std::vector<std::uint32_t> &values) {
  DeviceArray<std::uint32_t> device_keys(keys.size());
  DeviceArray<std::uint32_t> output(keys.size());
  DeviceArray<std::uint8_t> found(keys.size());
  CUDA_CHECK(cudaMemcpy(device_keys.pointer, keys.data(),
                        keys.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice));
  dictionary.lookup(
      {device_keys.pointer, keys.size(), output.pointer, found.pointer}, 0);
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
      throw std::runtime_error("inline pending-cross lookup mismatch");
}

int main() {
  try {
    constexpr std::uint32_t batch_rows = 16u;
    constexpr std::uint32_t epoch_rows = 16u * batch_rows;
    constexpr std::uint32_t completion_rows = 15u * batch_rows;
    constexpr std::uint32_t sealed_rows = 4u * epoch_rows;
    DictionaryConfig config{};
    config.max_elements = 4096u;
    config.batch_capacity = batch_rows;
    config.level_zero_capacity = epoch_rows;

    // A sparse mixed pending slot is completed by the restored ordinary
    // admission suffix.  The remaining complete epochs are then planned
    // structurally and must carry that sidecar forward.
    GPULSMOpt mixed_then_inline(config);
    PackedRecords mixed_prefix;
    for (std::uint32_t row = 0u; row < 7u; ++row)
      mixed_prefix.push(0x22000000u + row, 1000u + row,
                        row == 0u ? 0x41 : -1);
    insert_packed(mixed_then_inline, mixed_prefix);
    std::vector<std::uint32_t> inline_suffix_keys(
        completion_rows + sealed_rows);
    std::vector<std::uint32_t> inline_suffix_values(
        inline_suffix_keys.size());
    for (std::uint32_t row = 0u; row < inline_suffix_keys.size(); ++row) {
      inline_suffix_keys[row] = 0x30000000u + row;
      inline_suffix_values[row] = 2000u + row;
    }
    insert_inline(mixed_then_inline, inline_suffix_keys,
                  inline_suffix_values);
    verify_packed(mixed_then_inline, mixed_prefix);
    verify_inline(mixed_then_inline, inline_suffix_keys,
                  inline_suffix_values);

    // The opposite direction starts with an ordinary pending slot.  A mixed
    // suffix completes that epoch, including one long key, and leaves four
    // complete mixed epochs for direct-root sealed planning.
    GPULSMOpt inline_then_mixed(config);
    std::vector<std::uint32_t> inline_prefix_keys(7u);
    std::vector<std::uint32_t> inline_prefix_values(7u);
    for (std::uint32_t row = 0u; row < 7u; ++row) {
      inline_prefix_keys[row] = 0x40000000u + row;
      inline_prefix_values[row] = 3000u + row;
    }
    insert_inline(inline_then_mixed, inline_prefix_keys,
                  inline_prefix_values);
    PackedRecords mixed_suffix;
    for (std::uint32_t row = 0u;
         row < completion_rows + sealed_rows; ++row) {
      const int suffix = row == 3u ? 0x51
          : row == completion_rows + 100u ? 0x52 : -1;
      mixed_suffix.push(0x50000000u + row, 4000u + row, suffix);
    }
    insert_packed(inline_then_mixed, mixed_suffix);
    verify_inline(inline_then_mixed, inline_prefix_keys,
                  inline_prefix_values);
    verify_packed(inline_then_mixed, mixed_suffix);

    std::puts("specialized glue cross-mode pending smoke passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
