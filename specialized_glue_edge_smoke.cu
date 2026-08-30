#include "GPULSMOpt.cuh"

#include <algorithm>
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
  std::size_t size() const { return count_; }

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

static std::vector<std::uint8_t> u32_bytes(std::uint32_t value) {
  std::vector<std::uint8_t> bytes(sizeof(value));
  std::memcpy(bytes.data(), &value, sizeof(value));
  return bytes;
}

static std::vector<std::uint8_t> ordered_head(std::uint32_t head) {
  return {
      static_cast<std::uint8_t>(head >> 24u),
      static_cast<std::uint8_t>(head >> 16u),
      static_cast<std::uint8_t>(head >> 8u),
      static_cast<std::uint8_t>(head)};
}

class UploadedRecords {
 public:
  UploadedRecords(const std::vector<std::vector<std::uint8_t>> &keys,
                  const std::vector<std::vector<std::uint8_t>> &values,
                  const std::vector<std::uint8_t> &operations) {
    if (keys.size() != values.size() || keys.size() != operations.size())
      throw std::invalid_argument("uploaded record shape mismatch");
    count_ = keys.size();
    std::vector<std::uint8_t> key_bytes, value_bytes;
    std::vector<std::uint64_t> key_offsets(count_ + 1u);
    std::vector<std::uint64_t> value_offsets(count_ + 1u);
    for (std::size_t row = 0u; row < count_; ++row) {
      key_offsets[row] = key_bytes.size();
      key_bytes.insert(key_bytes.end(), keys[row].begin(), keys[row].end());
      value_offsets[row] = value_bytes.size();
      value_bytes.insert(value_bytes.end(), values[row].begin(),
                         values[row].end());
    }
    key_offsets[count_] = key_bytes.size();
    value_offsets[count_] = value_bytes.size();
    key_bytes_.copy_from(key_bytes);
    value_bytes_.copy_from(value_bytes);
    key_offsets_.copy_from(key_offsets);
    value_offsets_.copy_from(value_offsets);
    operations_.copy_from(operations);
  }

  DeviceRecordBatchView view() const {
    DeviceRecordBatchView result{};
    result.keys = {key_bytes_.data(), key_offsets_.data(), 0u};
    result.values = {value_bytes_.data(), value_offsets_.data(), 0u};
    result.operations = operations_.data();
    result.count = count_;
    result.key_encoding = DeviceKeyEncoding::ordered_bytes;
    return result;
  }

 private:
  std::size_t count_{};
  DeviceBuffer<std::uint8_t> key_bytes_, value_bytes_, operations_;
  DeviceBuffer<std::uint64_t> key_offsets_, value_offsets_;
};

class UploadedKeys {
 public:
  explicit UploadedKeys(
      const std::vector<std::vector<std::uint8_t>> &keys) {
    count_ = keys.size();
    std::vector<std::uint8_t> bytes;
    std::vector<std::uint64_t> offsets(count_ + 1u);
    for (std::size_t row = 0u; row < count_; ++row) {
      offsets[row] = bytes.size();
      bytes.insert(bytes.end(), keys[row].begin(), keys[row].end());
    }
    offsets[count_] = bytes.size();
    bytes_.copy_from(bytes);
    offsets_.copy_from(offsets);
  }

  DeviceKeyBatchView view() const {
    DeviceKeyBatchView result{};
    result.keys = {bytes_.data(), offsets_.data(), 0u};
    result.count = count_;
    result.key_encoding = DeviceKeyEncoding::ordered_bytes;
    return result;
  }

 private:
  std::size_t count_{};
  DeviceBuffer<std::uint8_t> bytes_;
  DeviceBuffer<std::uint64_t> offsets_;
};

struct ExpectedLookup {
  bool found{};
  std::vector<std::uint8_t> value;
};

static void verify_lookup(
    GPULSMOpt &dictionary,
    const std::vector<std::vector<std::uint8_t>> &queries,
    const std::vector<ExpectedLookup> &expected) {
  if (queries.size() != expected.size())
    throw std::invalid_argument("lookup expectation shape mismatch");
  UploadedKeys uploaded(queries);
  std::uint64_t stride = 1u;
  for (const auto &item : expected)
    stride = std::max<std::uint64_t>(stride, item.value.size());
  DeviceBuffer<std::uint8_t> output(queries.size() * stride);
  DeviceBuffer<std::uint64_t> lengths(queries.size());
  DeviceBuffer<std::uint8_t> found(queries.size()), overflow(queries.size());
  DeviceLookupBatchView batch{};
  batch.queries = uploaded.view();
  batch.output.values = {output.data(), nullptr,
                         queries.size() * stride, stride,
                         DeviceSinkLayout::fixed_stride};
  batch.output.required_value_lengths = lengths.data();
  batch.output.found = found.data();
  batch.output.overflow = overflow.data();
  dictionary.lookup(batch, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::uint8_t> host_output(queries.size() * stride);
  std::vector<std::uint64_t> host_lengths(queries.size());
  std::vector<std::uint8_t> host_found(queries.size());
  std::vector<std::uint8_t> host_overflow(queries.size());
  if (!host_output.empty())
    CUDA_CHECK(cudaMemcpy(host_output.data(), output.data(),
                          host_output.size(), cudaMemcpyDeviceToHost));
  if (!host_lengths.empty()) {
    CUDA_CHECK(cudaMemcpy(host_lengths.data(), lengths.data(),
                          host_lengths.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_found.data(), found.data(),
                          host_found.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_overflow.data(), overflow.data(),
                          host_overflow.size(), cudaMemcpyDeviceToHost));
  }
  for (std::size_t row = 0u; row < expected.size(); ++row) {
    if (host_found[row] != expected[row].found || host_overflow[row])
      throw std::runtime_error("edge lookup found/overflow mismatch");
    if (!expected[row].found) {
      if (host_lengths[row])
        throw std::runtime_error("missing edge lookup has a value length");
      continue;
    }
    if (host_lengths[row] != expected[row].value.size() ||
        std::memcmp(host_output.data() + row * stride,
                    expected[row].value.data(), expected[row].value.size()))
      throw std::runtime_error("edge lookup value mismatch");
  }
}

static void run_empty_and_embedded_gate() {
  const std::vector<std::uint8_t> empty_key{};
  const std::vector<std::uint8_t> one_key{0x00u};
  const std::vector<std::uint8_t> two_key{0x00u, 0x01u};
  const std::vector<std::uint8_t> three_key{0x12u, 0x00u, 0x34u};
  const std::vector<std::uint8_t> four_key{0x12u, 0x34u, 0x00u, 0x78u};
  std::vector<std::uint8_t> long_key(102u);
  for (std::size_t index = 0u; index < long_key.size(); ++index)
    long_key[index] = static_cast<std::uint8_t>(index * 37u);
  long_key[0] = 0x12u;
  long_key[1] = 0x34u;
  long_key[2] = 0x00u;
  long_key[3] = 0x78u;
  long_key[17] = 0u;
  const std::vector<std::uint8_t> prefix_three{0x55u, 0x66u, 0x77u};
  const std::vector<std::uint8_t> prefix_four{
      0x55u, 0x66u, 0x77u, 0x00u};
  const std::vector<std::uint8_t> prefix_long{
      0x55u, 0x66u, 0x77u, 0x00u, 0x42u};
  const std::vector<std::uint8_t> tombstoned{
      0xaau, 0xbbu, 0x00u, 0xddu, 0x42u};

  std::vector<std::uint8_t> value_129(129u);
  for (std::size_t index = 0u; index < value_129.size(); ++index)
    value_129[index] = static_cast<std::uint8_t>(index * 19u);
  value_129[0] = value_129[64] = value_129[128] = 0u;
  const std::vector<std::vector<std::uint8_t>> keys{
      empty_key, one_key, two_key, three_key, four_key, long_key,
      prefix_three, prefix_four, prefix_long, prefix_long, prefix_long,
      prefix_long, tombstoned, tombstoned, long_key, long_key, empty_key};
  const std::vector<std::vector<std::uint8_t>> values{
      {}, {0u}, {0u, 0xffu, 0u}, {1u, 0u, 2u, 0u},
      {3u, 0u, 4u, 0u, 5u}, value_129, {}, {7u, 0u, 8u},
      {1u, 2u, 3u}, {4u, 5u, 6u}, {}, {7u, 8u, 9u},
      {9u, 8u, 7u}, {}, {}, {6u, 0u, 5u}, {0xabu}};
  const std::vector<std::uint8_t> operations{
      0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 0u,
      0u, 1u, 1u, 0u, 0u};
  UploadedRecords uploaded(keys, values, operations);
  const std::vector<std::vector<std::uint8_t>> queries{
      empty_key, one_key, two_key, three_key, four_key, long_key,
      prefix_three, prefix_four, prefix_long, tombstoned,
      {0x12u, 0x34u, 0x00u, 0x78u, 0xffu}};
  const std::vector<ExpectedLookup> expected{
      {true, {0xabu}}, {true, {0u}}, {true, {0u, 0xffu, 0u}},
      {true, {1u, 0u, 2u, 0u}}, {true, {3u, 0u, 4u, 0u, 5u}},
      {true, {6u, 0u, 5u}}, {true, {}}, {true, {7u, 0u, 8u}},
      {true, {7u, 8u, 9u}}, {false, {}}, {false, {}}};

  DictionaryConfig config{};
  config.max_elements = 4096u;
  config.batch_capacity = 4u;
  config.level_zero_capacity = 64u;
  GPULSMOpt direct(config);
  direct.bulk_build(uploaded.view(), 0);
  verify_lookup(direct, queries, expected);

  // The same prototype edge rows first remain visible in sparse pending
  // state, then cross into the restored completed-epoch publication.
  GPULSMOpt pending(config);
  pending.insert(uploaded.view(), 0);
  verify_lookup(pending, queries, expected);
  constexpr std::uint32_t completion_rows = 12u * 4u;
  std::vector<std::uint32_t> completion_keys(completion_rows);
  std::vector<std::uint32_t> completion_values(completion_rows);
  for (std::uint32_t row = 0u; row < completion_rows; ++row) {
    completion_keys[row] = 0xe0000000u + row;
    completion_values[row] = 0xf0000000u + row;
  }
  DeviceBuffer<std::uint32_t> device_completion_keys,
      device_completion_values;
  device_completion_keys.copy_from(completion_keys);
  device_completion_values.copy_from(completion_values);
  pending.insert({device_completion_keys.data(),
                  device_completion_values.data(), completion_rows}, 0);
  verify_lookup(pending, queries, expected);
}

static std::vector<std::uint8_t> prototype_collision_key(
    std::uint32_t logical) {
  std::vector<std::uint8_t> key(12u);
  constexpr std::uint8_t fixed[4]{0x12u, 0x34u, 0x56u, 0x78u};
  std::copy(std::begin(fixed), std::end(fixed), key.begin());
  const std::uint32_t tail = prototype_mix(logical ^ 0xa511e9b3u);
  for (std::uint32_t position = 0u; position < 4u; ++position) {
    key[4u + position] = static_cast<std::uint8_t>(
        logical >> (24u - 8u * position));
    key[8u + position] = static_cast<std::uint8_t>(
        tail >> (24u - 8u * position));
  }
  return key;
}

static std::uint32_t prototype_rejection_hash(
    const std::vector<std::uint8_t> &key) {
  std::uint32_t hash = 2166136261u;
  for (std::uint8_t byte : key) hash = (hash ^ byte) * 16777619u;
  return (hash ^ static_cast<std::uint32_t>(key.size())) * 16777619u;
}

static void run_forced_collision_gate() {
  constexpr std::uint32_t rows = 1u << 16u;
  const auto first = prototype_collision_key(60732u);
  const auto second = prototype_collision_key(67711u);
  if (first == second || prototype_rejection_hash(first) != 0xb8bed1a7u ||
      prototype_rejection_hash(second) != 0xb8bed1a7u)
    throw std::runtime_error("prototype collision pair changed");
  std::vector<std::vector<std::uint8_t>> keys(rows), values(rows);
  std::vector<std::uint8_t> operations(rows);
  for (std::uint32_t row = 0u; row < rows; ++row) {
    keys[row] = (row & 1u) ? second : first;
    values[row] = u32_bytes(1000u + row);
  }
  UploadedRecords uploaded(keys, values, operations);
  DictionaryConfig config{};
  config.max_elements = 1u << 20u;
  config.batch_capacity = 1u << 16u;
  config.level_zero_capacity = 1u << 18u;
  GPULSMOpt dictionary(config);
  dictionary.bulk_build(uploaded.view(), 0);
  auto absent = first;
  absent[7] ^= 0x80u;
  verify_lookup(dictionary, {first, second, absent},
                {{true, u32_bytes(1000u + rows - 2u)},
                 {true, u32_bytes(1000u + rows - 1u)},
                 {false, {}}});
}

static void run_capsule_lifecycle_gate(std::uint32_t overwrite_count,
                                       bool expect_compaction) {
  constexpr std::uint32_t seed_rows = 64u;
  constexpr std::uint32_t batch_rows = 16u;
  constexpr std::uint32_t epoch_rows = 16u * batch_rows;
  constexpr std::uint32_t incoming_rows = 3u * epoch_rows;
  std::vector<std::uint32_t> seed_keys(seed_rows);
  std::vector<std::uint8_t> seed_values(seed_rows * 17u);
  std::vector<std::vector<std::uint8_t>> query_keys(seed_rows);
  std::vector<ExpectedLookup> expected(seed_rows);
  for (std::uint32_t row = 0u; row < seed_rows; ++row) {
    seed_keys[row] = 0x30000000u + row;
    query_keys[row] = ordered_head(seed_keys[row]);
    expected[row].found = true;
    expected[row].value.resize(17u);
    for (std::uint32_t position = 0u; position < 17u; ++position) {
      const std::uint8_t value = static_cast<std::uint8_t>(
          prototype_mix(row ^ (position * 0x85ebca6bu)) >> 24u);
      seed_values[std::size_t{row} * 17u + position] = value;
      expected[row].value[position] = value;
    }
  }
  DeviceBuffer<std::uint32_t> device_seed_keys;
  DeviceBuffer<std::uint8_t> device_seed_values;
  device_seed_keys.copy_from(seed_keys);
  device_seed_values.copy_from(seed_values);

  DictionaryConfig config{};
  config.max_elements = 8192u;
  config.batch_capacity = batch_rows;
  config.level_zero_capacity = epoch_rows;
  GPULSMOpt dictionary(config);
  DeviceRecordBatchView seed{};
  seed.keys = device_head4_source(device_seed_keys.data());
  seed.values = {device_seed_values.data(), nullptr, 17u};
  seed.count = seed_rows;
  seed.key_encoding = DeviceKeyEncoding::head4_words;
  dictionary.bulk_build(seed, 0);

  std::uint64_t object_bytes = 0u;
  if (!gpulsm_sparse::capsule_object_bytes(4u, 17u, object_bytes))
    throw std::runtime_error("capsule object size overflow");
  const auto before = dictionary.sparse_memory_accounting();
  if (before.capsule_segments != 1u ||
      before.capsule_live_bytes != seed_rows * object_bytes ||
      before.capsule_garbage_bytes)
    throw std::runtime_error("seed capsule accounting mismatch");

  std::vector<std::uint32_t> incoming_keys(incoming_rows);
  std::vector<std::uint32_t> incoming_values(incoming_rows);
  for (std::uint32_t row = 0u; row < incoming_rows; ++row) {
    if (row < overwrite_count) {
      incoming_keys[row] = seed_keys[row];
      incoming_values[row] = 0x90000000u + row;
      expected[row].value = u32_bytes(incoming_values[row]);
    } else {
      incoming_keys[row] = 0x50000000u + row;
      incoming_values[row] = 0xa0000000u + row;
    }
  }
  DeviceBuffer<std::uint32_t> device_incoming_keys,
      device_incoming_values;
  device_incoming_keys.copy_from(incoming_keys);
  device_incoming_values.copy_from(incoming_values);
  dictionary.insert({device_incoming_keys.data(),
                     device_incoming_values.data(), incoming_rows}, 0);
  verify_lookup(dictionary, query_keys, expected);

  const auto after = dictionary.sparse_memory_accounting();
  const std::uint64_t survivors = seed_rows - overwrite_count;
  const std::uint64_t expected_garbage = expect_compaction
      ? 0u : overwrite_count * object_bytes;
  if (after.capsule_segments != 1u ||
      after.capsule_live_bytes != survivors * object_bytes ||
      after.capsule_garbage_bytes != expected_garbage ||
      after.capsule_live_bytes > after.capsule_mapped_bytes ||
      after.capsule_garbage_bytes >
          after.capsule_mapped_bytes - after.capsule_live_bytes)
    throw std::runtime_error("capsule lifecycle accounting mismatch");
  if (!expect_compaction &&
      (after.capsule_mapped_bytes != before.capsule_mapped_bytes ||
       !after.capsule_garbage_bytes))
    throw std::runtime_error("capsule transfer was not retained");
  if (expect_compaction && after.capsule_garbage_bytes)
    throw std::runtime_error("dead-at-least-live segment was not reclaimed");
}

int main() {
  try {
    run_empty_and_embedded_gate();
    run_forced_collision_gate();
    run_capsule_lifecycle_gate(16u, false);
    run_capsule_lifecycle_gate(40u, true);
    std::puts("specialized glue edge/lifecycle smoke passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
