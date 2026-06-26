// =============================================================================
// File: definitions_coarse_granular.cuh
// Author: Justus Henneberg
// Description: Implements definitions_coarse_granular     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef DEFINITIONS_COARSE_GRANULAR_H
#define DEFINITIONS_COARSE_GRANULAR_H

#include "definitions.cuh"

#include "optix_wrapper.cuh"


namespace cg {

// compute base^exp with integer exp at compile time
constexpr float expi(float base, uint32_t exp) {
    return exp == 0 ? 1 : exp & 1 ? base * expi(base * base, exp >> 1u) : expi(base * base, exp >> 1u);
}

constexpr float y_scale = expi(2.0, 15);
constexpr float z_scale = expi(2.0, 25);

constexpr float x_eps = 0.5;
constexpr float y_eps = 0.5 * y_scale;
constexpr float z_eps = 0.5 * z_scale;

template <typename key_type>
HOSTQUALIFIER DEVICEQUALIFIER INLINEQUALIFIER
constexpr float key_to_float(key_type i) {
    return static_cast<float>(i) + 0.5;
}

template <typename key_type> constexpr uint32_t x_bits = sizeof(key_type) == 8 ? 23 : 19;
template <typename key_type> constexpr uint32_t y_bits = sizeof(key_type) == 8 ? 23 : 13;
template <typename key_type> constexpr uint32_t z_bits = sizeof(key_type) == 8 ? 18 : 0;

template <typename key_type> constexpr uint32_t x_mask = (uint32_t{1} << x_bits<key_type>) - 1;
template <typename key_type> constexpr uint32_t y_mask = (uint32_t{1} << y_bits<key_type>) - 1;
template <typename key_type> constexpr uint32_t z_mask = (uint32_t{1} << z_bits<key_type>) - 1;


template <typename key_type>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
constexpr float key_to_x(key_type key) {
    return key_to_float(key & x_mask<key_type>);
}

template <typename key_type>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
constexpr float key_to_y(key_type key) {
    return key_to_float((key >> x_bits<key_type>) & y_mask<key_type>) * y_scale;
}

template <typename key_type>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
constexpr float key_to_z(key_type key) {
    return key_to_float(0) * z_scale;
}

template <>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
constexpr float key_to_z<uint64_t>(uint64_t key) {
    return key_to_float(key >> (x_bits<uint64_t> + y_bits<uint64_t>)) * z_scale;
}

template <typename key_type> constexpr float row_marker_x = key_to_x(~key_type(0));   // max x
template <typename key_type> constexpr float plane_marker_y = key_to_y(~key_type(0)); // max y


template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER
key_type extract(const void* buffer, smallsize byte_offset);

template <>
DEVICEQUALIFIER INLINEQUALIFIER
uint32_t extract<uint32_t>(const void* buffer, smallsize byte_offset) {
    auto full_offset = reinterpret_cast<const uint8_t*>(buffer) + byte_offset;
    return (uint32_t)*reinterpret_cast<const uint32_t*>(full_offset);
}

template <>
DEVICEQUALIFIER INLINEQUALIFIER
uint64_t extract<uint64_t>(const void* buffer, smallsize byte_offset) {
    auto full_offset = reinterpret_cast<const uint8_t*>(buffer) + byte_offset;
    auto lower = reinterpret_cast<const uint32_t*>(full_offset + 0);
    auto upper = reinterpret_cast<const uint32_t*>(full_offset + 4);
    return ((uint64_t)*upper << 32u | (uint64_t)*lower);
}

template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER
void set(void* buffer, smallsize byte_offset, key_type key);

template <>
DEVICEQUALIFIER INLINEQUALIFIER
void set<uint32_t>(void* buffer, smallsize byte_offset, uint32_t key) {
    auto full_offset = reinterpret_cast<uint8_t*>(buffer) + byte_offset;
    *reinterpret_cast<uint32_t*>(full_offset) = key;
}

template <>
DEVICEQUALIFIER INLINEQUALIFIER
void set<uint64_t>(void* buffer, smallsize byte_offset, uint64_t key) {
    auto full_offset = reinterpret_cast<uint8_t*>(buffer) + byte_offset;
    auto lower = reinterpret_cast<uint32_t*>(full_offset + 0);
    auto upper = reinterpret_cast<uint32_t*>(full_offset + 4);
    *lower = key;
    *upper = key >> 32u;
}


DEVICEQUALIFIER INLINEQUALIFIER
void trace(OptixTraversableHandle bvh, float3 origin, float3 direction) {
    constexpr uint32_t sbt_offset = 0;
    constexpr uint32_t sbt_stride = 0;
    constexpr uint32_t sbt_miss_index = 0;
    constexpr float ray_time = 0;
    constexpr float tmin = 0;
    constexpr float tmax = float(1 << 24u) * z_scale;
    constexpr uint32_t ray_flags = OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT;

    optixTraverse(
        bvh, origin, direction, tmin, tmax,
        ray_time, OptixVisibilityMask(255), ray_flags, sbt_offset, sbt_stride, sbt_miss_index);
}

DEVICEQUALIFIER INLINEQUALIFIER
void trace_x(OptixTraversableHandle bvh, float origin_x, float origin_y, float origin_z) {
    trace(bvh, make_float3(origin_x, origin_y, origin_z), make_float3(1, 0, 0));
}

DEVICEQUALIFIER INLINEQUALIFIER
void trace_y(OptixTraversableHandle bvh, float origin_x, float origin_y, float origin_z) {
    trace(bvh, make_float3(origin_x, origin_y, origin_z), make_float3(0, 1, 0));
}

DEVICEQUALIFIER INLINEQUALIFIER
void trace_z(OptixTraversableHandle bvh, float origin_x, float origin_y, float origin_z) {
    trace(bvh, make_float3(origin_x, origin_y, origin_z), make_float3(0, 0, 1));
}


template <typename key_type>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
bool is_same_row(key_type lhs, key_type rhs) {
    return (lhs >> x_bits<key_type>) == (rhs >> x_bits<key_type>);
}

template <typename key_type>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
bool is_same_plane(key_type lhs, key_type rhs) {
    return true;
}

template <>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
bool is_same_plane<uint64_t>(uint64_t lhs, uint64_t rhs) {
    return (lhs >> (x_bits<uint64_t> + y_bits<uint64_t>)) == (rhs >> (x_bits<uint64_t> + y_bits<uint64_t>));
}

template <typename key_type>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
bool is_last_slot_in_row(key_type key) {
    return (key & x_mask<key_type>) == x_mask<key_type>;
}

template <typename key_type>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
bool is_last_row_in_plane(key_type key) {
    return ((key >> x_bits<key_type>) & y_mask<key_type>) == y_mask<key_type>;
}

template <typename key_type>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
bool is_last_slot_in_plane(key_type key) {
    return is_last_slot_in_row(key) && is_last_row_in_plane(key);
}

template <typename key_type>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
bool is_safe_to_move(key_type representative, key_type next_key) {
    return !is_same_row(representative, next_key);
}


template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER
smallsize find_partition_offset_with_rays(OptixTraversableHandle traversable, smallsize partition_count, key_type lower_bound) {
    // we assume lower_bound <= MAX_KEY, so misses are impossible
    const float x = key_to_x(lower_bound);
    const float y = key_to_y(lower_bound);
    const float z = key_to_z(lower_bound);

    // find a representative on the same row
    trace_x(traversable, x - x_eps, y, z);
    if (optixHitObjectIsHit()) {
        smallsize partition_offset = optixHitObjectGetPrimitiveIndex();
        if (partition_offset >= 2 * partition_count) {
            // hit_eop_1: we hit the end-of-plane marker
            return partition_offset - 2 * partition_count + 1;
        }
        if (partition_offset >= partition_count) {
            // hit_eor: we hit the end-of-row marker
            return partition_offset - partition_count + 1;
        }
        // hit_regular_1 or hit_moved: we hit a regular representative (same row)
        return partition_offset;
    }

    // no representative on the same row, find a row marker
    trace_y(traversable, row_marker_x<key_type>, y + y_eps, z);
    if (optixHitObjectIsHit()) {
        smallsize partition_offset = optixHitObjectGetPrimitiveIndex();
        if (partition_offset >= 2 * partition_count) {
            // hit_eop_2: we hit the end-of-plane marker
            return partition_offset - 2 * partition_count + 1;
        }
        if (optixIsFrontFaceHit(optixHitObjectGetHitKind())) {
            // hit_flipped_1: we hit the only representative in this row, no need to search the row
            return optixHitObjectGetPrimitiveIndex();
        }
        float populated_y = y + y_eps + optixHitObjectGetRayTmax();
        trace_x(traversable, 0, populated_y, z);
        // hit_regular_2: we hit a regular representative (same plane, different row)
        return optixHitObjectGetPrimitiveIndex();
    }

    // not a single triangle on this plane, find a plane marker
    trace_z(traversable, row_marker_x<key_type>, plane_marker_y<key_type>, z + z_eps);
    float populated_z = z + z_eps + optixHitObjectGetRayTmax();
    trace_y(traversable, row_marker_x<key_type>, 0, populated_z);
    if (optixIsFrontFaceHit(optixHitObjectGetHitKind())) {
        // hit_flipped_2: we hit the only representative in this row, no need to search the row
        return optixHitObjectGetPrimitiveIndex();
    }
    float populated_y = optixHitObjectGetRayTmax();
    trace_x(traversable, 0, populated_y, populated_z);
    // hit_regular_3: we hit a regular representative (different plane)
    return optixHitObjectGetPrimitiveIndex();
}

}

#endif
