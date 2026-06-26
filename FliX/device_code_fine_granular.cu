// =============================================================================
// File: device_code_fine_granular.cu
// Author: Justus Henneberg
// Description: Launch experiments     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#include "launch_parameters.cuh"
#include "definitions_fine_granular.cuh"


extern "C" __constant__ fg_params params;


template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
packed_type get_payload_32() {
    static_assert(sizeof(packed_type) <= 4);
    return (packed_type) optixGetPayload_0();
}


template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
void set_payload_32(packed_type i) {
    static_assert(sizeof(packed_type) <= 4);
    optixSetPayload_0((uint32_t) i);
}


template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
packed_type get_secondary_payload_32() {
    static_assert(sizeof(packed_type) <= 4);
    return (packed_type) optixGetPayload_1();
}


template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
void set_secondary_payload_32(packed_type i) {
    static_assert(sizeof(packed_type) <= 4);
    optixSetPayload_1((uint32_t) i);
}


extern "C" GLOBALQUALIFIER void __anyhit__lookup() {
    const unsigned int primitive_id = optixGetPrimitiveIndex();

    smallsize position = primitive_id;

    bool multiple_hits_possible = get_secondary_payload_32<bool>();
    if (multiple_hits_possible) {
        set_payload_32(get_payload_32<smallsize>() + position);
        // reject the hit, this prevents tmax from being reduced
        optixIgnoreIntersection();
    } else {
        set_payload_32(position);
    }
}


extern "C" GLOBALQUALIFIER void __raygen__lookup() {
    const unsigned int ix = optixGetLaunchIndex().x;

    key64 key = params.long_keys ? ((key64*)params.query_lower)[ix] : ((key32*)params.query_lower)[ix];
    key64 lower_bound, upper_bound;

    bool is_range_query = params.has_range_queries;
    if (is_range_query) {
        lower_bound = key;
        upper_bound = params.long_keys ? ((key64*)params.query_upper)[ix] : ((key32*)params.query_upper)[ix];
    }

    // a ray can hit multiple triangles if either
    // - the query is a range query
    // - the query is a point query, but some triangles exist multiple times
    bool multiple_hits_possible = is_range_query || !params.keys_are_unique;

    // aggregation register
    smallsize i0 = multiple_hits_possible ? 0 : not_found;
    uint32_t i1 = multiple_hits_possible;

    // if lower_bound == upper_bound, we can cast a perpendicular ray!
    if (is_range_query && lower_bound != upper_bound) {
        // decompose the key into x and yz
        key64 smallest_yz = lower_bound >> fg::x_bits;
        key64 largest_yz = upper_bound >> fg::x_bits;
        float smallest_x = fg::uint32_as_float(lower_bound & fg::x_mask);
        float largest_x = fg::uint32_as_float(upper_bound & fg::x_mask);
        float smallest_possible_x = fg::uint32_as_float(0);
        float largest_possible_x = fg::uint32_as_float(fg::x_mask);

        // cast one ray per yz offset
        for (uint64_t yz = smallest_yz; yz <= largest_yz; ++yz) {
            float y = fg::uint32_as_float(yz & fg::y_mask);
            float z = fg::uint32_as_float(yz >> fg::y_bits);

            float from = (yz == smallest_yz ? smallest_x : smallest_possible_x) - fg::eps;
            float to = (yz == largest_yz ? largest_x : largest_possible_x) + fg::eps;

            float3 origin = make_float3(from, y, z);
            float3 direction = make_float3(1, 0, 0);
            float tmin = 0;
            float tmax = to - from;

            optixTrace(
                    params.traversable,
                    origin,
                    direction,
                    tmin,
                    tmax,
                    0.0f,
                    OptixVisibilityMask(255),
                    // we can use TERMINATE_ON_FIRST_HIT for a range
                    // query since all hits will be rejected anyway
                    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                    0,
                    0,
                    0,
                    i0,
                    i1);
        }
        params.result[ix] = i0;
    } else {
        float x = fg::uint32_as_float(key & fg::x_mask);
        float y = fg::uint32_as_float((key >> fg::x_bits) & fg::y_mask);
        float z = fg::uint32_as_float(key >> (fg::x_bits + fg::y_bits));

        // perpendicular ray
        float3 origin = make_float3(x, y, z - fg::eps);
        float3 direction = make_float3(0, 0, 1);
        float tmin = 0;
        float tmax = 1;

        optixTrace(
                params.traversable,
                origin,
                direction,
                tmin,
                tmax,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                0,
                0,
                0,
                i0,
                i1);
        params.result[ix] = i0;
    }
}
