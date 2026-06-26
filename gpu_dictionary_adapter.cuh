#pragma once

#include <cstddef>
#include <cstdint>

struct DictionaryConfig {
  std::size_t max_elements = 0;
  std::size_t batch_size = 0;
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

struct DeviceRangeBatch {
  const std::uint32_t* lo = nullptr;
  const std::uint32_t* hi = nullptr;
  std::uint32_t* out_counts = nullptr;
  std::size_t count = 0;
};

struct DeviceRangeOutputBatch {
  const std::uint32_t* lo = nullptr;
  const std::uint32_t* hi = nullptr;
  std::size_t query_count = 0;
  std::size_t output_capacity = 0;
  std::uint32_t* out_counts = nullptr;
  std::uint32_t* out_offsets = nullptr;
  std::uint32_t* out_keys = nullptr;
  std::uint32_t* out_values = nullptr;
};
