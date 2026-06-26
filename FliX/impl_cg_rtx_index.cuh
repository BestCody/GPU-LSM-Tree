// =============================================================================
// File: impl_cg_rtx_index.cuh
// Author: Justus Henneberg
// Description: Implements impl_cg_rtx_index     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef IMPL_COARSE_RTX_INDEX_CUH
#define IMPL_COARSE_RTX_INDEX_CUH

#include "definitions.cuh"
#include "cuda_buffer.cuh"
#include "utilities.cuh"
#include "optix_wrapper.cuh"
#include "optix_pipeline.cuh"
#include "launch_parameters.cuh"
#include "definitions_coarse_granular.cuh"
#include "build_coarse_granular.cuh"

#include <nvtx3/nvtx3.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace coop = cooperative_groups;


extern "C" char coarse_granular_embedded_ptx_code[];

// this will be initialized in main.cu
extern optix_wrapper optix;

// for nvtx
struct nvtx_cg_rtx_domain{ static constexpr char const* name{"cg_rtx"}; };

/*
template <typename key_type>
GLOBALQUALIFIER
void transform_into_row_layout_kernel(const key_type* keys, const smallsize* offsets, void* row_buffer, smallsize key_count) {
    static_assert(sizeof(key_type) == 4 || sizeof(key_type) == 8, "key_type must be 4 or 8 bytes");

    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= key_count) return;

    smallsize byte_offset = tid * (sizeof(key_type) + 4);
    cg::set(row_buffer, byte_offset, keys[tid]);
    cg::set(row_buffer, byte_offset + sizeof(key_type), offsets[tid]);
}


template <typename key_type>
void transform_into_row_layout(
    const key_type* keys,
    const smallsize* offsets,
    void* row_buffer,
    size_t key_count,
    double* time_ms
) {
    scoped_cuda_timer timer(0, time_ms);
    transform_into_row_layout_kernel<<<SDIV(key_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(keys, offsets, row_buffer, key_count);
    cudaStreamSynchronize(0); CUERR
}
*/

template <typename key_type>
OptixTraversableHandle build_structures(
    const optix_wrapper& optix,
    const key_type* keys_device_pointer,
    size_t key_count,
    size_t partition_size,
    size_t* out_partition_count,
    cuda_buffer<uint8_t>& as_buffer,
    cuda_buffer<uint8_t>& ordered_key_offset_pairs_buffer,
    double* build_time_ms,
    size_t* build_bytes
) {
    cuda_buffer<key_type> ordered_key_buffer;
    cuda_buffer<smallsize> offset_buffer;
    cuda_buffer<smallsize> ordered_offset_buffer;
    cuda_buffer<uint8_t> sort_temp_buffer;

    // this is required to find the maximum space needed for the build
    auto size_snapshot = [&]{
        return ordered_key_offset_pairs_buffer.size_in_bytes()
             + ordered_key_buffer.size_in_bytes()
             + offset_buffer.size_in_bytes()
             + ordered_offset_buffer.size_in_bytes()
             + sort_temp_buffer.size_in_bytes();
    };

    // fill offset buffer with 0, 1, 2, 3, ...
    offset_buffer.alloc(key_count); CUERR
    init_offsets(offset_buffer, key_count, build_time_ms);

    // sort keys and offsets
    size_t sort_temp_buffer_bytes = find_pair_sort_buffer_size<key_type, smallsize>(key_count);
    sort_temp_buffer.alloc(sort_temp_buffer_bytes); CUERR
    ordered_key_buffer.alloc(key_count); CUERR
    ordered_offset_buffer.alloc(key_count); CUERR
    timed_pair_sort(
            sort_temp_buffer.raw_ptr,
            sort_temp_buffer_bytes,
            keys_device_pointer,
            ordered_key_buffer.ptr(),
            offset_buffer.ptr(),
            ordered_offset_buffer.ptr(),
            key_count,
            build_time_ms);
    size_t max_bytes_during_sort = size_snapshot();
    sort_temp_buffer.free();
    offset_buffer.free();

    size_t row_stride = sizeof(key_type) + 4;

    // pack offsets and keys into row layout
    ordered_key_offset_pairs_buffer.alloc(row_stride * key_count); CUERR
    transform_into_row_layout(
            ordered_key_buffer.ptr(),
            ordered_offset_buffer.ptr(),
            ordered_key_offset_pairs_buffer.ptr(),
            key_count,
            build_time_ms);
    size_t max_bytes_during_transform = size_snapshot();
    ordered_offset_buffer.free();
    ordered_key_buffer.free();

    // find out how many partitions are required
    size_t partition_count = SDIV(key_count, partition_size);

    size_t max_bytes_during_build;
    // feed partition structure into the bvh builder
    auto compacted_as = cg::build_compacted_as_from_representatives<key_type>(
            optix,
            ordered_key_offset_pairs_buffer.ptr(),
            partition_count,
            // byte offset first entry
            (partition_size - 1) * row_stride,
            // stride in bytes
            partition_size * row_stride,
            // byte offset last entry
            (key_count - 1) * row_stride,
            // offset to next key
            row_stride,
            as_buffer,
            build_time_ms,
            &max_bytes_during_build);
    max_bytes_during_build += size_snapshot();

    if (build_bytes) *build_bytes = std::max(max_bytes_during_sort, std::max(max_bytes_during_build, max_bytes_during_transform));
    if (out_partition_count) *out_partition_count = partition_count;
    return compacted_as;
}


template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER
key_type extract_key(const void* buf, smallsize i) {
    return cg::extract<key_type>(buf, i * (sizeof(key_type) + sizeof(smallsize)));
}


template <typename key_type>
DEVICEQUALIFIER INLINEQUALIFIER
smallsize extract_offset(const void* buf, smallsize i) {
    return cg::extract<smallsize>(buf, i * (sizeof(key_type) + sizeof(smallsize)) + sizeof(key_type));
}


template <uint32_t cg_size, typename key_type, typename params_type>
GLOBALQUALIFIER
void collaborative_range_scan_kernel(const params_type* params_ptr, size_t probe_size) {
    const auto &params = *params_ptr;

    const smallsize ix = (uint32_t) blockDim.x * blockIdx.x + threadIdx.x;
    auto group = coop::tiled_partition<cg_size>(coop::this_thread_block());

    if ((ix - group.thread_rank()) >= probe_size) return;

    key_type lower_bound;
    key_type upper_bound;
    smallsize partition_id;
    smallsize result = 0;
    bool to_find = false;
    if (ix < probe_size) {
        lower_bound = reinterpret_cast<const key_type*>(params.query_lower)[ix];
        upper_bound = reinterpret_cast<const key_type*>(params.query_upper)[ix];
        partition_id = params.result[ix];
        to_find = partition_id != not_found;
    }

    auto work_queue = group.ballot(to_find);
    while (work_queue) {
        auto cur_rank = __ffs(work_queue) - 1;
        key_type cur_lower_bound = group.shfl(lower_bound, cur_rank);
        key_type cur_upper_bound = group.shfl(upper_bound, cur_rank);
        uint32_t cur_partition_id = group.shfl(partition_id, cur_rank);

        smallsize cur_result = 0;
        {
            uint32_t offset = params.partition_size * cur_partition_id;
            for (;; offset += cg_size) {
                bool is_valid = offset + group.thread_rank() < params.stored_size;

                key_type key = !is_valid ? 0 : extract_key<key_type>(params.ordered_key_offset_pairs, offset + group.thread_rank());
                bool overstepped = !is_valid || key > cur_upper_bound;
                bool is_in_range = !overstepped && key >= cur_lower_bound;

                cur_result += !is_in_range ? 0 : extract_offset<key_type>(params.ordered_key_offset_pairs, offset + group.thread_rank());
                if (group.any(overstepped)) break;
            }
            cur_result = coop::reduce(group, cur_result, coop::plus<smallsize>());
        }

        if (cur_rank == group.thread_rank()) {
            result = cur_result;
            to_find = false;
        }
        work_queue = group.ballot(to_find);
    }

    if (ix < probe_size) {
        params.result[ix] = result;
    }
}


GLOBALQUALIFIER
void setup_build_data(
    cg_params* launch_params,
    OptixTraversableHandle traversable,
    const void* ordered_key_offset_pairs,
    smallsize stored_size,
    smallsize partition_size,
    smallsize partition_count
) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1) return;

    launch_params->traversable = traversable;
    launch_params->ordered_key_offset_pairs = ordered_key_offset_pairs;
    launch_params->stored_size = stored_size;
    launch_params->partition_size = partition_size;
    launch_params->partition_count = partition_count;
}


template <typename key_type>
GLOBALQUALIFIER
void setup_probe_data(
    cg_params* launch_params,
    bool long_keys,
    bool aggregate_results,
    bool find_partition_only,
    const key_type* query_lower,
    const key_type* query_upper,
    smallsize* result
) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1) return;

    launch_params->long_keys = long_keys;
    launch_params->aggregate_results = aggregate_results;
    launch_params->find_partition_only = find_partition_only;
    launch_params->query_lower = query_lower;
    launch_params->query_upper = query_upper;
    launch_params->result = result;
}


template <typename key_type_, uint8_t partition_size_log = 4, uint8_t scan_cg_size_log = 4>
class cg_rtx_index {
public:
    using key_type = key_type_;

private:
    cuda_buffer<uint8_t> as_buffer;
    cuda_buffer<uint8_t> ordered_key_offset_pairs_buffer;
    cuda_buffer<cg_params> launch_params_buffer;
    std::optional<optix_pipeline> pipeline;

public:
    static constexpr bool can_lookup = true;
    static constexpr bool can_multi_lookup = true;
    static constexpr bool can_range_lookup = true;
    static constexpr bool can_update = false;

    static std::string short_description() {
        std::string desc = "cg_rtx_index";
        desc += "_" + std::to_string(partition_size_log);
        desc += "_" + std::to_string(scan_cg_size_log);
        return desc;
    }

    static size_t estimate_build_bytes(size_t size) {
        size_t partition_size = size_t(1) << partition_size_log;
        size_t permutation_bytes = (sizeof(key_type) + 4) * size;
        size_t sort_aux_bytes = sizeof(smallsize) * size + find_pair_sort_buffer_size<key_type, smallsize>(size);
        size_t triangle_bytes = 3 * 9 * SDIV(size, partition_size) * sizeof(float);
        size_t bvh_bytes = triangle_bytes * 13 / 10;
        size_t aux_bytes = triangle_bytes * 2 / 10;

        size_t phase_1_bytes = permutation_bytes + sort_aux_bytes;
        size_t phase_2_bytes = permutation_bytes + triangle_bytes + bvh_bytes + aux_bytes;
        size_t phase_3_bytes = permutation_bytes + 2 * bvh_bytes;
        return std::max(phase_1_bytes, std::max(phase_2_bytes, phase_3_bytes));
    }

    size_t gpu_resident_bytes() {
        return as_buffer.size_in_bytes() + ordered_key_offset_pairs_buffer.size_in_bytes() + launch_params_buffer.size_in_bytes();
    }

    void build(const key_type* keys, size_t size, double* build_time_ms, size_t* build_bytes) {

        if (!pipeline) {
            pipeline.emplace(
                &optix,
                coarse_granular_embedded_ptx_code,
                "params",
                "__raygen__lookup"
            );
        }

        launch_params_buffer.alloc(1); CUERR
        if (build_bytes) *build_bytes += launch_params_buffer.size_in_bytes();

        size_t partition_size = size_t(1) << partition_size_log;
        size_t partition_count;
        OptixTraversableHandle traversable = build_structures(
                *pipeline->optix,
                keys,
                size,
                partition_size,
                &partition_count,
                as_buffer,
                ordered_key_offset_pairs_buffer,
                build_time_ms,
                build_bytes);

        setup_build_data<<<1, 1>>>(
                launch_params_buffer.ptr(),
                traversable,
                ordered_key_offset_pairs_buffer.ptr(),
                size,
                partition_size,
                partition_count);

        cudaDeviceSynchronize(); CUERR
    }

    void lookup(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {

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
                &pipeline->sbt,
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
                scan_cg_size_log > 0,
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
                &pipeline->sbt,
                size,
                1,
                1))
            // todo test this
            if constexpr (scan_cg_size_log > 0) {
                constexpr uint8_t cg_size = 1u << scan_cg_size_log;
                collaborative_range_scan_kernel<cg_size, key_type><<<SDIV(size, 1024), 1024, 0, stream>>>(
                    launch_params_buffer.ptr(), size);
            }
        }
    }

    void multi_lookup_sum(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        range_lookup_sum(keys, keys, result, size, stream);
    }

    void destroy() {
        as_buffer.free();
        ordered_key_offset_pairs_buffer.free();
        launch_params_buffer.free();
    }

    void insert(const key_type* update_list, const smallsize* offsets, size_t size, cudaStream_t stream) {}
    void remove(const key_type* update_list, size_t size, cudaStream_t stream) {}
};

#endif
