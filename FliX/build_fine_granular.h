// =============================================================================
// File: build_fine_granular.h
// Author: Justus Henneberg
// Description: Defines expressions and constants used in FliX     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INDEX_PROTOTYPE_BUILD_FINE_GRANULAR_H
#define INDEX_PROTOTYPE_BUILD_FINE_GRANULAR_H

inline
OptixTraversableHandle consume_primitives_build_traversable(
    const optix_wrapper& optix,
    size_t key_count,
    cuda_buffer<float3>& prefilled_primitive_buffer,
    cuda_buffer<uint8_t>& as_buffer,
    double* build_time_ms,
    size_t* build_bytes
) {
    cuda_buffer<uint64_t> compacted_size_buffer;
    cuda_buffer<uint8_t> temp_buffer, uncompacted_structure_buffer;

    OptixTraversableHandle structure_handle{0};

    uint32_t build_input_flags[1] = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };
    CUdeviceptr vertices = prefilled_primitive_buffer.cu_ptr();

    OptixBuildInput structure_input = {};
    structure_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    structure_input.triangleArray.numVertices         = 3 * (unsigned) key_count;
    structure_input.triangleArray.vertexBuffers       = &vertices;
    structure_input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    structure_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    structure_input.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_NONE;
    structure_input.triangleArray.flags               = build_input_flags;
    structure_input.triangleArray.numSbtRecords       = 1;

    OptixAccelBuildOptions structure_options = {};
    structure_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    structure_options.motionOptions.numKeys = 1;
    structure_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes structure_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix.optix_context,
            &structure_options,
            &structure_input,
            1,  // num_build_inputs
            &structure_buffer_sizes
    ))

    compacted_size_buffer.alloc(1);

    OptixAccelEmitDesc emit_desc;
    emit_desc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = compacted_size_buffer.cu_ptr();

    uncompacted_structure_buffer.alloc(structure_buffer_sizes.outputSizeInBytes);
    temp_buffer.alloc(structure_buffer_sizes.tempSizeInBytes);

    cudaDeviceSynchronize();

    {
        cuda_timer timer(0);
        timer.start();
        OPTIX_CHECK(optixAccelBuild(
                optix.optix_context,
                0,
                &structure_options,
                &structure_input,
                1,
                temp_buffer.cu_ptr(),
                temp_buffer.size_in_bytes(),
                uncompacted_structure_buffer.cu_ptr(),
                uncompacted_structure_buffer.size_in_bytes(),
                &structure_handle,
                &emit_desc, 1
        ))
        timer.stop();
        if (build_time_ms) *build_time_ms += timer.time_ms();
    }

    uint64_t compacted_size;
    compacted_size_buffer.download(&compacted_size, 1);

    size_t temp_bytes1 = as_buffer.size_in_bytes() + prefilled_primitive_buffer.size_in_bytes() +
                         compacted_size_buffer.size_in_bytes() + temp_buffer.size_in_bytes() + uncompacted_structure_buffer.size_in_bytes();

    prefilled_primitive_buffer.free();
    compacted_size_buffer.free();

    as_buffer.alloc(compacted_size);

    {
        cuda_timer timer(0);
        timer.start();
        OPTIX_CHECK(optixAccelCompact(
                optix.optix_context,
                0,
                structure_handle,
                as_buffer.cu_ptr(),
                as_buffer.size_in_bytes(),
                &structure_handle));
        timer.stop();
        if (build_time_ms) *build_time_ms += timer.time_ms();
    }

    size_t temp_bytes2 = as_buffer.size_in_bytes() + prefilled_primitive_buffer.size_in_bytes() +
                         compacted_size_buffer.size_in_bytes() + temp_buffer.size_in_bytes() + uncompacted_structure_buffer.size_in_bytes();

    // this computation is more complicated since buffers can be partially reused during build
    if (build_bytes) *build_bytes += std::max(temp_bytes1, temp_bytes2);

    return structure_handle;
}

#endif
