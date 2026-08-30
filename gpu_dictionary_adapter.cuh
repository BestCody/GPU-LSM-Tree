#pragma once

#include <cstddef>
#include <cstdint>

enum class DeleteOrderPolicy : std::uint8_t {
  adaptive,
  uniform_live,
};

enum class BaseDeleteValuePolicy : std::uint8_t {
  eager,
  mark_only,
};

struct DictionaryConfig {
  std::size_t max_elements = 0;
  std::size_t batch_capacity = 0;
  std::size_t level_zero_capacity = 0;
  DeleteOrderPolicy delete_order = DeleteOrderPolicy::adaptive;
  BaseDeleteValuePolicy base_delete_values =
      BaseDeleteValuePolicy::eager;
};

struct DeviceKeyValueBatch {
  const std::uint32_t* keys = nullptr;
  const std::uint32_t* values = nullptr;
  std::size_t count = 0;
};

struct DeviceLookupBatch {
  const std::uint32_t* queries = nullptr;
  std::size_t count = 0;
  std::uint32_t* out_values = nullptr;
  std::uint8_t* out_found = nullptr;
};

struct DeviceSuccessorBatch {
  const std::uint32_t* queries = nullptr;
  std::size_t count = 0;
  std::uint32_t* out_keys = nullptr;
};

struct DeviceRangeOutputBatch {
  const std::uint32_t* lo = nullptr;
  const std::uint32_t* hi = nullptr;
  std::size_t query_count = 0;
  std::uint32_t* out_sums = nullptr;
};

// Universal source/sink capabilities.  The existing four-byte adapter above
// remains the native FliX contract; these views add mixed-length records
// without selecting a different dictionary implementation.
enum class DeviceKeyEncoding : std::uint8_t {
  ordered_bytes,
  head4_words,
};

enum class DeviceMutation : std::uint8_t {
  put,
  erase,
};

enum class DeviceSourceLifetime : std::uint8_t {
  operation,
};

enum class DeviceSinkLayout : std::uint8_t {
  fixed_stride,
  packed,
};

struct DeviceByteSource {
  const std::uint8_t* bytes = nullptr;
  const std::uint64_t* offsets = nullptr;
  std::uint64_t stride = 0;
};

struct DeviceRecordBatchView {
  DeviceByteSource keys{};
  DeviceByteSource values{};
  const std::uint8_t* operations = nullptr;
  const std::uint32_t* range_contributions = nullptr;
  std::uint64_t count = 0;
  DeviceKeyEncoding key_encoding = DeviceKeyEncoding::ordered_bytes;
  DeviceMutation uniform_operation = DeviceMutation::put;
  DeviceSourceLifetime lifetime = DeviceSourceLifetime::operation;
};

struct DeviceKeyBatchView {
  DeviceByteSource keys{};
  std::uint64_t count = 0;
  DeviceKeyEncoding key_encoding = DeviceKeyEncoding::ordered_bytes;
  DeviceSourceLifetime lifetime = DeviceSourceLifetime::operation;
};

struct DeviceByteSink {
  std::uint8_t* bytes = nullptr;
  std::uint64_t* offsets = nullptr;
  std::uint64_t capacity_bytes = 0;
  std::uint64_t stride = 0;
  DeviceSinkLayout layout = DeviceSinkLayout::fixed_stride;
};

struct DeviceLookupResultView {
  DeviceByteSink values{};
  std::uint64_t* required_value_lengths = nullptr;
  std::uint8_t* found = nullptr;
  std::uint8_t* overflow = nullptr;
};

struct DeviceLookupBatchView {
  DeviceKeyBatchView queries{};
  DeviceLookupResultView output{};
};

struct DeviceSuccessorResultView {
  DeviceByteSink keys{};
  std::uint64_t* required_key_lengths = nullptr;
  std::uint8_t* found = nullptr;
  std::uint8_t* overflow = nullptr;
};

struct DeviceSuccessorBatchView {
  DeviceKeyBatchView queries{};
  DeviceSuccessorResultView output{};
};

struct DeviceRangeQueryView {
  DeviceKeyBatchView lower{};
  DeviceKeyBatchView upper{};
};

struct DeviceRangeSumBatchView {
  DeviceRangeQueryView queries{};
  std::uint32_t* out_sums = nullptr;
  std::uint8_t* valid = nullptr;
};

inline DeviceByteSource device_u32_source(const std::uint32_t* values) {
  return {reinterpret_cast<const std::uint8_t*>(values), nullptr,
          sizeof(std::uint32_t)};
}

inline DeviceByteSource device_head4_source(const std::uint32_t* heads) {
  return {reinterpret_cast<const std::uint8_t*>(heads), nullptr,
          sizeof(std::uint32_t)};
}

inline DeviceByteSink device_u32_sink(std::uint32_t* values,
                                      std::uint64_t count) {
  return {reinterpret_cast<std::uint8_t*>(values), nullptr,
          count * sizeof(std::uint32_t), sizeof(std::uint32_t),
          DeviceSinkLayout::fixed_stride};
}
