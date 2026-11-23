#pragma once

// Constants shared by every module and the tier/slot capacity math.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t e__ = (call);                                             \
    if (e__ != cudaSuccess)                                                     \
      throw std::runtime_error(std::string(cudaGetErrorString(e__)) +           \
                               " at " + __FILE__ + ":" +                     \
                               std::to_string(__LINE__));                       \
  } while (false)
#endif

namespace gpulsmopt2_detail {

constexpr std::uint32_t kQuotients = 1u << 16u;
constexpr std::uint32_t kMaximumLevels = 64u;
constexpr std::uint32_t kBatchesPerEpoch = 16u;
constexpr std::uint32_t kBatchPositionBits = 20u;
constexpr std::uint32_t kLocalRankBits = 7u;
constexpr std::uint32_t kLocalRankEntries =
    kQuotients * (1u << kLocalRankBits);
constexpr std::uint32_t kThreads = 256u;

constexpr std::uint32_t kInvalid = 0xffffffffu;
constexpr std::uint32_t kInvalidAge = 0xffffffffu;
constexpr std::uint32_t kTombstone = 1u;
constexpr std::size_t kMaximumOperationTile = std::size_t{1} << 20u;

static_assert(kMaximumOperationTile <=
              (std::size_t{1} << kBatchPositionBits));
constexpr std::size_t kMaximumPublicationRows =
    std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kDescriptorOffsetBits = 47u;
constexpr std::uint64_t kDescriptorOffsetMask =
    (std::uint64_t{1} << kDescriptorOffsetBits) - 1u;
static_assert(kMaximumPublicationRows <= kDescriptorOffsetMask);

constexpr std::uint32_t kFoundationCompactionThreads = 256u;

constexpr std::uint32_t kFoundationCells = 128u;
constexpr std::uint32_t kFoundationCellKeys = 512u;
// A full section ends at the descriptor count.
__host__ __device__ constexpr bool cell_rank_supported(
    std::uint64_t count) {
  return count <= kQuotients;
}
static_assert((kFoundationCells - 1u) * kFoundationCellKeys <=
              std::numeric_limits<std::uint16_t>::max());

inline std::size_t initial_storage_capacity(
    std::size_t requested, std::size_t tile_capacity) {
  const std::size_t capacity = std::max(
      requested, tile_capacity * kBatchesPerEpoch);
  if (capacity > kMaximumPublicationRows)
    throw std::invalid_argument(
        "GPULSMOpt capacity exceeds 32-bit key space");
  return capacity;
}

inline std::size_t initial_level_capacity(
    std::size_t requested, std::size_t default_capacity,
    std::size_t maximum) {
  const std::size_t capacity = requested
      ? requested : default_capacity;
  return std::min(maximum, std::max<std::size_t>(1u, capacity));
}

struct CanonicalLevelLayout {
  std::size_t initial_pool_capacity{};
  std::size_t highest_regular_capacity{};
  std::uint32_t regular_level_count{};
  std::uint32_t level_count{};
};

// Each radix-4 tier holds three immutable runs.
inline CanonicalLevelLayout canonical_level_layout(
    std::size_t maximum_raw_rows, std::size_t epoch_capacity) {
  CanonicalLevelLayout layout{};
  std::size_t remaining = maximum_raw_rows;
  std::size_t capacity = std::min(maximum_raw_rows, epoch_capacity);
  while (remaining) {
    const std::size_t slots = std::min<std::size_t>(
        3u, (remaining + capacity - 1u) / capacity);
    if (slots > (std::numeric_limits<std::size_t>::max() -
                 layout.initial_pool_capacity) / capacity)
      throw std::bad_alloc();
    layout.initial_pool_capacity += slots * capacity;
    if (layout.regular_level_count > kMaximumLevels - slots)
      throw std::invalid_argument("GPULSMOpt radix-4 slot count overflow");
    layout.regular_level_count += static_cast<std::uint32_t>(slots);
    layout.highest_regular_capacity = capacity;
    const std::size_t covered = std::min(remaining, slots * capacity);
    remaining -= covered;
    if (!remaining) break;
    capacity = capacity > maximum_raw_rows / 4u
        ? maximum_raw_rows : capacity * 4u;
  }
  layout.level_count = layout.regular_level_count;
  if (layout.highest_regular_capacity < maximum_raw_rows) {
    if (layout.level_count == kMaximumLevels)
      throw std::invalid_argument("GPULSMOpt radix-4 slot count overflow");
    ++layout.level_count;
  }
  return layout;
}

inline std::size_t preassigned_level_pool_capacity(
    std::size_t maximum_raw_rows, std::size_t epoch_capacity) {
  return canonical_level_layout(
      maximum_raw_rows, epoch_capacity).initial_pool_capacity;
}

inline std::uint32_t canonical_regular_level_count(
    std::size_t maximum_raw_rows, std::size_t epoch_capacity) {
  return canonical_level_layout(
      maximum_raw_rows, epoch_capacity).regular_level_count;
}

inline std::uint32_t canonical_level_count(
    std::size_t maximum_raw_rows, std::size_t epoch_capacity) {
  return canonical_level_layout(maximum_raw_rows, epoch_capacity).level_count;
}

}  // namespace gpulsmopt2_detail
