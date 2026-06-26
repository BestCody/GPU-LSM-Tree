// =============================================================================
// File: impl_rtx_index.cuh
// Author: Justus Henneberg
// Description: Implements impl_rtx_index     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef RTX_INDEX_H
#define RTX_INDEX_H

#include "definitions.cuh"
#include "cuda_buffer.cuh"
#include "utilities.cuh"
#include "optix_wrapper.cuh"
#include "optix_pipeline.cuh"
#include "launch_parameters.cuh"
#include "definitions_fine_granular.cuh"
#include "build_fine_granular.h"

#include <nvtx3/nvtx3.hpp>

#include <chrono>


extern "C" char fine_granular_embedded_ptx_code[];

// this will be initialized in main.cu
extern optix_wrapper optix;

// for nvtx
struct nvtx_rtx_domain{ static constexpr char const* name{"rtx"}; };


template <typename key_type>
GLOBALQUALIFIER
void convert_keys_to_primitives_kernel(const key_type* keys_d, size_t key_count, float3* primitives_d) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= key_count) return;

    key64 key = (key64) keys_d[tid];
    float x = fg::uint32_as_float(key & fg::x_mask);
    float y = fg::uint32_as_float((key >> fg::x_bits) & fg::y_mask);
    float z = fg::uint32_as_float(key >> (fg::x_bits + fg::y_bits));

    float just_below_x = x - fg::eps;
    float just_above_x = x + fg::eps;
    float just_below_y = y - fg::eps;
    float just_above_y = y + fg::eps;
    float just_below_z = z - fg::eps;
    float just_above_z = z + fg::eps;

    primitives_d[3 * tid + 0] = make_float3(just_below_x, just_above_y, just_below_z);
    primitives_d[3 * tid + 1] = make_float3(just_below_x, just_below_y, just_below_z);
    primitives_d[3 * tid + 2] = make_float3(just_above_x,            y, just_above_z);
}


template <typename key_type>
void convert_keys_to_primitives(
    const key_type* keys_device_pointer,
    size_t key_count,
    cuda_buffer<float3>& primitive_buffer,
    double* build_time_ms
) {
    primitive_buffer.alloc(3 * key_count);

    cuda_timer timer(0);
    timer.start();
    convert_keys_to_primitives_kernel<<<SDIV(key_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(
            keys_device_pointer, key_count, primitive_buffer.ptr());
    timer.stop();
    if (build_time_ms) *build_time_ms += timer.time_ms();
}


GLOBALQUALIFIER
void setup_build_data(
    fg_params* launch_params,
    OptixTraversableHandle traversable
) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1) return;

    launch_params->traversable = traversable;
}


template <typename key_type>
GLOBALQUALIFIER
void setup_probe_data(
    fg_params* launch_params,
    bool long_keys,
    bool has_range_queries,
    bool keys_are_unique,
    const key_type* query_lower,
    const key_type* query_upper,
    smallsize* result
) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1) return;

    launch_params->long_keys = long_keys;
    launch_params->has_range_queries = has_range_queries;
    launch_params->keys_are_unique = keys_are_unique;
    launch_params->query_lower = query_lower;
    launch_params->query_upper = query_upper;
    launch_params->result = result;
}


template <typename key_type_>
class rtx_index {
public:
    using key_type = key_type_;

private:
    cuda_buffer<fg_params> launch_params_buffer;
    cuda_buffer<uint8_t> as_buffer;
    std::optional<optix_pipeline> pipeline;

public:
    static constexpr bool can_lookup = true;
    static constexpr bool can_multi_lookup = true;
    static constexpr bool can_range_lookup = true;
    static constexpr bool can_update = false;

    static std::string short_description() {
        return std::string("rtx_index");
    }

    static size_t estimate_build_bytes(size_t size) {
        size_t triangle_buffer_bytes = size * 9 * sizeof(float);
        size_t bvh_bytes = triangle_buffer_bytes * 13 / 10;
        size_t aux_bytes = triangle_buffer_bytes * 2 / 10;
        return std::max(triangle_buffer_bytes + bvh_bytes, 2 * bvh_bytes) + aux_bytes;
    }

    size_t gpu_resident_bytes() {
        return as_buffer.size_in_bytes() + launch_params_buffer.size_in_bytes();
    }

    void build(const key_type* keys, size_t size, double* build_time_ms, size_t* build_bytes) {

        if (!pipeline) {
            pipeline.emplace(
                &optix,
                fine_granular_embedded_ptx_code,
                "params",
                "__raygen__lookup",
                "",
                "__anyhit__lookup",
                "",
                2
            );
        }

        launch_params_buffer.alloc(1);
        if (build_bytes) *build_bytes += launch_params_buffer.size_in_bytes();

        cuda_buffer<float3> primitive_buffer;
        convert_keys_to_primitives(keys, size, primitive_buffer, build_time_ms);
        auto traversable = consume_primitives_build_traversable(
                optix, size, primitive_buffer, as_buffer, build_time_ms, build_bytes); CUERR

        setup_build_data<<<1, 1>>>(launch_params_buffer.ptr(), traversable); CUERR

        cudaDeviceSynchronize(); CUERR
    }

    void lookup(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {

        {
            nvtx3::scoped_range_in<nvtx_rtx_domain> upload{"upload-params"};
            setup_probe_data<<<1, 1, 0, stream>>>(
                launch_params_buffer.ptr(),
                sizeof(key_type) == 8,
                false,
                true,
                keys,
                keys,
                result);
        }

        {
            nvtx3::scoped_range_in<nvtx_rtx_domain> launch{"launch"};
            OPTIX_CHECK(optixLaunch(
                pipeline->pipeline,
                stream,
                launch_params_buffer.cu_ptr(),
                launch_params_buffer.size_in_bytes(),
                &pipeline.value().sbt,
                size,
                1,
                1))
        }
    }

    void multi_lookup_sum(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {

        {
            nvtx3::scoped_range_in<nvtx_rtx_domain> upload{"upload-params"};
            setup_probe_data<<<1, 1, 0, stream>>>(
                launch_params_buffer.ptr(),
                sizeof(key_type) == 8,
                false,
                false,
                keys,
                keys,
                result);
        }

        {
            nvtx3::scoped_range_in<nvtx_rtx_domain> launch{"launch"};
            OPTIX_CHECK(optixLaunch(
                pipeline->pipeline,
                stream,
                launch_params_buffer.cu_ptr(),
                launch_params_buffer.size_in_bytes(),
                &pipeline.value().sbt,
                size,
                1,
                1))
        }
    }

    void range_lookup_sum(const key_type* lower, const key_type* upper, smallsize* result, size_t size, cudaStream_t stream) {

        {
            nvtx3::scoped_range_in<nvtx_rtx_domain> upload{"upload-params"};
            setup_probe_data<<<1, 1, 0, stream>>>(
                launch_params_buffer.ptr(),
                sizeof(key_type) == 8,
                true,
                false,
                lower,
                upper,
                result);
        }

        {
            nvtx3::scoped_range_in<nvtx_rtx_domain> launch{"launch"};
            OPTIX_CHECK(optixLaunch(
                pipeline->pipeline,
                stream,
                launch_params_buffer.cu_ptr(),
                launch_params_buffer.size_in_bytes(),
                &pipeline.value().sbt,
                size,
                1,
                1))
        }
    }

    void destroy() {
        as_buffer.free();
        launch_params_buffer.free();
    }

    void insert(const key_type* update_list, const smallsize* offsets, size_t size, cudaStream_t stream) {}
    void remove(const key_type* update_list, size_t size, cudaStream_t stream) {}
};

#endif
