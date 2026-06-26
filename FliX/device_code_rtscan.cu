// =============================================================================
// File: device_code_rtscan.cu
// Author: Justus Henneberg
// Description: Launch experiments     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#include "launch_parameters_rtscan.h"


extern "C" __constant__ rtscan_params params;


static __forceinline__ __device__ bool set_result(uint32_t *bitmap, smallsize idx) {
    smallsize size = sizeof(uint32_t) << 3;
    smallsize pos = idx / size;
    smallsize pos_in_size = idx & (size - 1);

    uint32_t old = atomicOr(&bitmap[pos], uint32_t(1) << pos_in_size);
    // return true if this is the first time this bit was set
    return !(old & (uint32_t(1) << pos_in_size));
}


extern "C" __global__ void __raygen__rg() {
    const smallsize ix = optixGetLaunchIndex().x;
    const smallsize launch_size = optixGetLaunchDimensions().x;

    smallsize item_to_process = ix;

    for (smallsize current_lookup = 0; current_lookup < params.batch_size; ++current_lookup) {
        key32 lower_bound = params.query_lower[current_lookup];
        key32 upper_bound = params.query_upper[current_lookup];

        // taken from "refineWithOptixRTc1"
        smallsize rays_required = (smallsize) ((upper_bound - lower_bound) / params.ray_spacing) + 2;
        // given the original definition of ray_spacing, the following holds
        // rays_required ~ (query_upper - query_lower) / (data_max - data_min) * rays_for_full_range

        if (item_to_process >= rays_required) {
            item_to_process -= rays_required;
            continue;
        }

        // actually computed in impl_rtscan.cuh
        // but since we are in 1d, no bounds are necessary
        float ray_length = 1e20;

        // inlined from "computeRay" in "optixScan.cu" with "params.direction == 1"
        float3 ray_origin = {float(lower_bound + (item_to_process + 1) * params.ray_spacing), 0.0f, 0.0f};
        float3 ray_direction = {0.0f, 1.0f, 0.0f};

        smallsize agg = 0;
        optixTrace(
                params.traversable,
                ray_origin,
                ray_direction,
                0.0f,                           // Min intersection distance
                ray_length,                     // Max intersection distance
                0.0f,                           // rayTime -- used for motion blur
                OptixVisibilityMask(255),       // Specify always visible
                OPTIX_RAY_FLAG_NONE,
                0,                   // SBT offset   -- See SBT discussion
                1,                   // SBT stride   -- See SBT discussion
                0,                   // missSBTIndex -- See SBT discussion
                agg,
                current_lookup
        );
        atomicAdd(&params.result[current_lookup], agg);

        item_to_process = item_to_process + launch_size - rays_required;
    }
}

extern "C" __global__ void __anyhit__get_prim_id() {
    smallsize primitive_index = optixGetPrimitiveIndex();
    smallsize current_lookup = optixGetPayload_1();
    key32 point = params.stored_keys[primitive_index];
    uint32_t* target_bitmap = params.bitmaps + size_t(params.bitmap_entries) * current_lookup;
    if (point >= params.query_lower[current_lookup] && point <= params.query_upper[current_lookup]) {
        if (set_result(target_bitmap, primitive_index)) {
            optixSetPayload_0(optixGetPayload_0() + primitive_index);
        }
    }
    optixIgnoreIntersection();
}
