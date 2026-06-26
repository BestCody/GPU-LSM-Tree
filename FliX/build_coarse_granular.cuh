// =============================================================================
// File: build_coarse_granular.cuh
// Author: Justus Henneberg
// Description: Implements build_coarse_granular : For a Ray Tracing Index Layer
//              Ray Tracing is removed in final version of FliX. 
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef BUILD_COARSE_GRANULAR_CUH
#define BUILD_COARSE_GRANULAR_CUH

#include "definitions_coarse_granular.cuh"

namespace cg {

template <typename key_type>
GLOBALQUALIFIER
void make_triangles_from_sorted_keys_kernel(
    const void* buffer,
    smallsize partition_count,
    smallsize representative_offset,
    smallsize representative_stride,
    smallsize last_representative_offset,
    smallsize representative_successor_offset,
    __restrict__ float3* triangle_buffer
) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= partition_count) return;

    key_type min_representative = extract<key_type>(buffer, min(representative_offset, last_representative_offset));
    key_type max_representative = extract<key_type>(buffer, last_representative_offset);
    bool single_row = is_same_row(min_representative, max_representative);
    bool single_plane = is_same_plane(min_representative, max_representative);

    bool is_first_group = tid == 0;
    bool is_last_group = tid == partition_count - 1;
    bool moving_forbidden = representative_successor_offset == 0;

    smallsize partition_byte_offset = is_last_group ? last_representative_offset : representative_offset + tid * representative_stride;

    key_type partition_representative = extract<key_type>(buffer, partition_byte_offset);
    key_type previous_partition_representative = is_first_group ? 0 : extract<key_type>(buffer, partition_byte_offset - representative_stride);
    key_type next_partition_representative = is_last_group ? 0 : extract<key_type>(buffer, partition_byte_offset + representative_stride);
    key_type representative_successor = moving_forbidden || is_last_group ? 0 : extract<key_type>(buffer, partition_byte_offset + representative_successor_offset);

    float x = key_to_x(partition_representative);
    float y = key_to_y(partition_representative);
    float z = key_to_z(partition_representative);

    bool is_last_group_marker_in_row = is_last_group || !is_same_row(partition_representative, next_partition_representative);
    bool is_last_group_marker_in_plane = is_last_group || !is_same_plane(partition_representative, next_partition_representative);

    bool move_triangle_to_end_of_row = !moving_forbidden && is_safe_to_move(partition_representative, representative_successor);
    bool insert_triangle_at_end_of_row = !single_row && is_last_group_marker_in_row && !is_last_slot_in_row(partition_representative) && !move_triangle_to_end_of_row;
    bool insert_triangle_at_end_of_plane = !single_plane && is_last_group_marker_in_plane && !is_last_row_in_plane(partition_representative);

    // flip the marker triangle if
    // - the successor of the representative is in the next row
    // - there is no representative in the same row
    // this implies you can skip the ray in x direction
    bool previous_representative_same_row = !is_first_group && is_same_row(previous_partition_representative, partition_representative);
    bool flip_representative = move_triangle_to_end_of_row && !previous_representative_same_row;

    bool generate_representative = tid == 0 || previous_partition_representative != partition_representative || (move_triangle_to_end_of_row && !is_last_slot_in_row(partition_representative));

    if (generate_representative) {
        auto local_x = move_triangle_to_end_of_row ? row_marker_x<key_type> : x;
        bool f = flip_representative;
        triangle_buffer[3 * tid + f]  = make_float3(local_x + x_eps, y - y_eps, z        );
        triangle_buffer[3 * tid + !f] = make_float3(local_x        , y + y_eps, z - z_eps);
        triangle_buffer[3 * tid + 2]  = make_float3(local_x - x_eps, y        , z + z_eps);
    }
    if (insert_triangle_at_end_of_row) {
        // generate slanted triangles for this method
        triangle_buffer[3 * partition_count + 3 * tid + 0] = make_float3(row_marker_x<key_type> + x_eps, y - y_eps, z        );
        triangle_buffer[3 * partition_count + 3 * tid + 1] = make_float3(row_marker_x<key_type>        , y + y_eps, z - z_eps);
        triangle_buffer[3 * partition_count + 3 * tid + 2] = make_float3(row_marker_x<key_type> - x_eps, y        , z + z_eps);
    }
    if (insert_triangle_at_end_of_plane) {
        // generate slanted triangles for this method
        triangle_buffer[6 * partition_count + 3 * tid + 0] = make_float3(row_marker_x<key_type> + x_eps, plane_marker_y<key_type> - y_eps, z        );
        triangle_buffer[6 * partition_count + 3 * tid + 1] = make_float3(row_marker_x<key_type>        , plane_marker_y<key_type> + y_eps, z - z_eps);
        triangle_buffer[6 * partition_count + 3 * tid + 2] = make_float3(row_marker_x<key_type> - x_eps, plane_marker_y<key_type>        , z + z_eps);
    }
}

template <typename key_type>
void make_triangles_from_sorted_keys(
    const void* buffer,
    size_t partition_count,
    size_t representative_offset,
    size_t representative_stride,
    size_t last_representative_offset,
    // byte offset from current representative, set to zero for no successor
    size_t representative_successor_offset,
    float3* triangle_buffer,
    double* time_ms
) {
    {
        scoped_cuda_timer timer(0, time_ms);
        make_triangles_from_sorted_keys_kernel<key_type><<<SDIV(partition_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(
                buffer,
                partition_count,
                representative_offset,
                representative_stride,
                last_representative_offset,
                representative_successor_offset,
                triangle_buffer);
    }
    cudaStreamSynchronize(0); CUERR
}

void setup_build_input(OptixAccelBuildOptions& bi) {
    bi.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    bi.motionOptions.numKeys = 1;
    bi.operation = OPTIX_BUILD_OPERATION_BUILD;
}

void setup_structure_input(OptixBuildInput& bi, void** buffer, size_t primitive_count) {
    static uint32_t build_input_flags[1] = { 0 };

    bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.vertexBuffers       = (CUdeviceptr*) buffer;
    bi.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    bi.triangleArray.vertexStrideInBytes = sizeof(float3);
    bi.triangleArray.numIndexTriplets    = 0;
    bi.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_NONE;
    bi.triangleArray.preTransform        = 0;
    bi.triangleArray.flags               = build_input_flags;
    bi.triangleArray.numVertices         = 3 * (unsigned) primitive_count;
    bi.triangleArray.numSbtRecords       = 1;
}

void alloc_as_buffers(
    const optix_wrapper& optix,
    const OptixAccelBuildOptions& structure_options,
    const OptixBuildInput& structure_input,
    cuda_buffer<uint8_t>& structure_buffer,
    cuda_buffer<uint8_t>& temp_buffer
) {
    OptixAccelBufferSizes structure_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix.optix_context,
            &structure_options,
            &structure_input,
            1,  // num_build_inputs
            &structure_buffer_sizes))
    structure_buffer.alloc(structure_buffer_sizes.outputSizeInBytes); CUERR
    temp_buffer.alloc(structure_buffer_sizes.tempSizeInBytes); CUERR
}

OptixTraversableHandle build_as(
    const optix_wrapper& optix,
    const OptixBuildInput& structure_input,
    const OptixAccelBuildOptions& structure_options,
    cuda_buffer<uint8_t>& structure_buffer,
    cuda_buffer<uint8_t>& temp_buffer,
    double* time_ms
) {
    OptixTraversableHandle as{0};
    {
        scoped_cuda_timer timer(0, time_ms);
        OPTIX_CHECK(optixAccelBuild(
                optix.optix_context,
                0,
                &structure_options,
                &structure_input,
                1,
                temp_buffer.cu_ptr(),
                temp_buffer.size_in_bytes(),
                structure_buffer.cu_ptr(),
                structure_buffer.size_in_bytes(),
                &as,
                nullptr, 0))
    }
    return as;
}

void alloc_compacted_as_buffer(
    const optix_wrapper& optix,
    OptixTraversableHandle as,
    cuda_buffer<uint8_t>& as_buffer
) {
    cuda_buffer<uint64_t> compacted_size_buffer;
    compacted_size_buffer.alloc(1); CUERR

    OptixAccelEmitDesc emit_desc;
    emit_desc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = compacted_size_buffer.cu_ptr();

    OPTIX_CHECK(optixAccelEmitProperty(optix.optix_context, 0, as, &emit_desc))
    size_t compacted_size = compacted_size_buffer.download_first_item();
    as_buffer.alloc(compacted_size); CUERR
}

OptixTraversableHandle compact_as(
    const optix_wrapper& optix,
    OptixTraversableHandle uncompacted_as,
    cuda_buffer<uint8_t>& as_buffer,
    double* time_ms
) {
    OptixTraversableHandle as{0};
    {
        scoped_cuda_timer timer(0, time_ms);
        OPTIX_CHECK(optixAccelCompact(
                optix.optix_context,
                0,
                uncompacted_as,
                as_buffer.cu_ptr(),
                as_buffer.size_in_bytes(),
                &as))
    }
    return as;
}

template <typename key_type>
void find_minmax_representative(
    const void* buffer,
    size_t representative_offset,
    size_t last_representative_offset,
    key_type& min_r,
    key_type& max_r
) {
    auto byte_buffer = reinterpret_cast<const uint8_t*>(buffer);
    cudaMemcpy(&min_r, byte_buffer + std::min(representative_offset, last_representative_offset), sizeof(key_type), D2H); CUERR
    cudaMemcpy(&max_r, byte_buffer + last_representative_offset, sizeof(key_type), D2H); CUERR
}

template <typename key_type>
OptixTraversableHandle build_compacted_as_from_representatives(
    const optix_wrapper& optix,
    const void* buffer,
    size_t partition_count,
    size_t representative_offset,
    size_t representative_stride,
    size_t last_representative_offset,
    size_t representative_successor_offset, // set to zero for no successor
    cuda_buffer<uint8_t>& as_buffer,
    double* build_time_ms,
    size_t* max_bytes_required_during_build
) {
    cuda_buffer<float3> triangle_buffer;
    cuda_buffer<uint8_t> uncompacted_as_buffer;
    cuda_buffer<uint8_t> build_temp_buffer;

    // this is required to find the maximum space needed for the build
    auto size_snapshot = [&]{
        return triangle_buffer.size_in_bytes()
             + uncompacted_as_buffer.size_in_bytes()
             + build_temp_buffer.size_in_bytes()
             + as_buffer.size_in_bytes();
    };

    // find min and max representative
    key_type min_r, max_r;
    find_minmax_representative(buffer, representative_offset, last_representative_offset, min_r, max_r);
    bool single_row = is_same_row(min_r, max_r);
    bool single_plane = is_same_plane(min_r, max_r);
    size_t marker_overhead = single_row ? 0 : single_plane ? 1 : 2;

    // create triangles from sorted keys
    triangle_buffer.alloc((1 + marker_overhead) * 3 * partition_count); CUERR
    triangle_buffer.zero(); CUERR
    make_triangles_from_sorted_keys<key_type>(
            buffer,
            partition_count,
            representative_offset,
            representative_stride,
            last_representative_offset,
            representative_successor_offset,
            triangle_buffer,
            build_time_ms);

    // prepare acceleration structure build
    OptixAccelBuildOptions structure_options = {};
    setup_build_input(structure_options);
    OptixBuildInput structure_input = {};
    setup_structure_input(structure_input, &triangle_buffer.raw_ptr, (1 + marker_overhead) * partition_count);

    // build acceleration structure
    alloc_as_buffers(optix, structure_options, structure_input, uncompacted_as_buffer, build_temp_buffer);
    OptixTraversableHandle uncompacted_as = build_as(
            optix,
            structure_input,
            structure_options,
            uncompacted_as_buffer,
            build_temp_buffer,
            build_time_ms);
    size_t max_bytes_during_build = size_snapshot();
    triangle_buffer.free();

    // compact acceleration structure
    alloc_compacted_as_buffer(optix, uncompacted_as, as_buffer);
    OptixTraversableHandle compacted_as = compact_as(
            optix,
            uncompacted_as,
            as_buffer,
            build_time_ms
            );
    size_t max_bytes_during_compact = size_snapshot();
    uncompacted_as_buffer.free();

    if (max_bytes_required_during_build) *max_bytes_required_during_build = std::max(max_bytes_during_build, max_bytes_during_compact);
    return compacted_as;
}

}

#endif
