#include "GPULSMOpt.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t error, const char *where) {
  if (error != cudaSuccess)
    throw std::runtime_error(
        std::string(where) + ": " + cudaGetErrorString(error));
}

template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { reset(count); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() { cudaFree(data_); }

  void reset(std::size_t count) {
    if (data_) check_cuda(cudaFree(data_), "cudaFree");
    data_ = nullptr;
    count_ = count;
    if (count) check_cuda(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
  }

  void upload(const std::vector<T> &source) {
    reset(source.size());
    if (!source.empty())
      check_cuda(cudaMemcpy(data_, source.data(), source.size() * sizeof(T),
                            cudaMemcpyHostToDevice),
                 "copy to GPU");
  }

  std::vector<T> download() const {
    std::vector<T> result(count_);
    if (count_)
      check_cuda(cudaMemcpy(result.data(), data_, count_ * sizeof(T),
                            cudaMemcpyDeviceToHost),
                 "copy from GPU");
    return result;
  }

  T *data() const { return data_; }

 private:
  T *data_{};
  std::size_t count_{};
};

std::uint32_t expected_successor(
    const std::map<std::uint32_t, std::uint32_t> &visible,
    std::uint32_t query) {
  const auto answer = visible.lower_bound(query);
  return answer == visible.end()
      ? std::numeric_limits<std::uint32_t>::max()
      : answer->first;
}

void insert(GPULSMOpt &index,
            std::map<std::uint32_t, std::uint32_t> &visible,
            const std::vector<std::uint32_t> &keys,
            std::uint32_t value_base) {
  std::vector<std::uint32_t> values(keys.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = value_base + static_cast<std::uint32_t>(i);
  DeviceBuffer<std::uint32_t> device_keys;
  DeviceBuffer<std::uint32_t> device_values;
  device_keys.upload(keys);
  device_values.upload(values);
  index.insert(
      {device_keys.data(), device_values.data(), keys.size()}, 0);
  check_cuda(cudaDeviceSynchronize(), "finish insertion");
  for (std::size_t i = 0; i < keys.size(); ++i) visible[keys[i]] = values[i];
}

void erase(GPULSMOpt &index,
           std::map<std::uint32_t, std::uint32_t> &visible,
           const std::vector<std::uint32_t> &keys) {
  DeviceBuffer<std::uint32_t> device_keys;
  device_keys.upload(keys);
  index.erase({device_keys.data(), keys.size()}, 0);
  check_cuda(cudaDeviceSynchronize(), "finish deletion");
  for (const std::uint32_t key : keys) visible.erase(key);
}

void check_successors(
    GPULSMOpt &index,
    const std::map<std::uint32_t, std::uint32_t> &visible,
    const std::vector<std::uint32_t> &queries,
    const char *phase) {
  DeviceBuffer<std::uint32_t> device_queries;
  DeviceBuffer<std::uint32_t> device_results(queries.size());
  device_queries.upload(queries);
  index.successor(
      {device_queries.data(), queries.size(), device_results.data()}, 0);
  check_cuda(cudaDeviceSynchronize(), "finish successor");
  const auto results = device_results.download();
  for (std::size_t i = 0; i < queries.size(); ++i) {
    const std::uint32_t expected = expected_successor(visible, queries[i]);
    if (results[i] != expected)
      throw std::runtime_error(
          std::string("successor mismatch in ") + phase +
          " at query " + std::to_string(queries[i]) +
          ": expected " + std::to_string(expected) +
          ", received " + std::to_string(results[i]));
  }
}

}  // namespace

int main() {
  try {
    check_cuda(cudaSetDevice(0), "select GPU");
    DictionaryConfig config;
    config.max_elements = 1u << 18u;
    config.batch_capacity = 4096u;
    config.level_zero_capacity = 1u << 16u;
    GPULSMOpt index(config);

    std::map<std::uint32_t, std::uint32_t> visible;
    const std::vector<std::uint32_t> initial{
        0x00000010u, 0x00000210u, 0x00000420u, 0x0000ffffu,
        0x00010020u,
        0x0001f010u, 0x01000030u, 0x02010040u, 0x7fff0001u,
        0xfffffffeu};
    std::vector<std::uint32_t> values(initial.size());
    for (std::size_t i = 0; i < initial.size(); ++i) {
      values[i] = static_cast<std::uint32_t>(i + 1u);
      visible[initial[i]] = values[i];
    }
    DeviceBuffer<std::uint32_t> initial_keys;
    DeviceBuffer<std::uint32_t> initial_values;
    initial_keys.upload(initial);
    initial_values.upload(values);
    index.bulk_build(initial_keys.data(), initial_values.data(),
                     initial.size(), 0);
    check_cuda(cudaDeviceSynchronize(), "finish build");

    const std::vector<std::uint32_t> probes{
        0u, 0x00000010u, 0x00000011u, 0x000001ffu,
        0x00000210u, 0x0000ffffu, 0x00010021u,
        0x01000030u, 0x01000031u, 0x7fff0001u, 0x7fff0002u,
        0xffff0000u, 0xfffffffeu, 0xffffffffu};
    check_successors(index, visible, probes, "published run");

    insert(index, visible, {0x00000100u}, 100u);
    erase(index, visible, {0x00000210u, 0x0000ffffu});
    insert(index, visible, {0x00000220u}, 200u);
    erase(index, visible, {0x00010020u});
    insert(index, visible, {0x00010020u}, 300u);
    check_successors(index, visible, probes, "pending updates");

    for (std::uint32_t batch = 0u; batch < 11u; ++batch)
      insert(index, visible, {0x03000000u + batch * 0x401u}, 400u + batch);
    check_successors(index, visible, probes, "after publication");

    erase(index, visible, {0x00000100u, 0x00000420u, 0x01000030u});
    insert(index, visible, {0x00000300u, 0x01000040u}, 600u);
    check_successors(index, visible, probes, "pending tombstones");

    const std::size_t large_count = (std::size_t{1} << 20u) + 17u;
    std::vector<std::uint32_t> large_queries(large_count);
    for (std::size_t i = 0; i < large_count; ++i)
      large_queries[i] = probes[i % probes.size()];
    check_successors(index, visible, large_queries,
                     "shared pending full batch");

    std::vector<std::uint32_t> crowded_region(2048u);
    for (std::uint32_t i = 0u; i < crowded_region.size(); ++i)
      crowded_region[i] = 0x04000000u + 2u * i;
    insert(index, visible, crowded_region, 1000u);
    const std::vector<std::uint32_t> crowded_probes{
        0x03ffffffu, 0x04000000u, 0x04000001u,
        0x04000ffeu, 0x04000fffu, 0x04001000u};
    check_successors(index, visible, crowded_probes,
                     "crowded pending region");
    for (std::size_t i = 0; i < large_count; ++i)
      large_queries[i] = crowded_probes[i % crowded_probes.size()];
    check_successors(index, visible, large_queries,
                     "shared crowded pending region");

    std::cout << "PASS CellLSM successor across cells, empty regions, "
                 "pending updates, publication, and tombstones\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
