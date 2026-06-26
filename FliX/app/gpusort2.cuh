// =============================================================================
// File: gpusort2.cuh
// Author: Justus Henneberg
// Description: Implements gpusort2     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

// file: app/main_micro_bench_168.cu
#include <cstdint>
#include <vector>
#include <cstdio>
#include "../cuda_buffer.cuh"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
// Assumes your environment provides:
// - cuda_buffer<T>, scoped_cuda_timer, rc::result_collector
// - find_pair_sort_buffer_size<>, untimed_pair_sort(...)
// - lambda_kernel, SDIV, DEVICEQUALIFIER
// If names differ, adjust includes accordingly.

template <typename element_type>
void micro_benchmark_sort_and_coalesced_onesize(std::size_t build_size_log,
                                                std::size_t lookup_size_log = 16,
                                                std::size_t runs = 20)
{
    rc::result_collector rc_sort;
    rc::result_collector rc_coalesced;

    const std::size_t build_size  = std::size_t(1) << build_size_log;
    const std::size_t lookup_size = std::size_t(1) << lookup_size_log;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Fill keys [0..build_size)
    std::vector<element_type> elements(build_size);
    for (std::size_t i = 0; i < build_size; ++i) {
        elements[i] = static_cast<element_type>(i);
    }

    // Buffers
    cuda_buffer<element_type> elements_buffer, dummy_buffer, output_elements_buffer, output_dummy_buffer;
    elements_buffer.alloc_and_upload(elements);
    dummy_buffer.alloc_and_upload(elements);          // dummy companion
    output_elements_buffer.alloc_for_size(elements);  // output slots
    output_dummy_buffer.alloc_for_size(elements);

    // Sort aux
    cuda_buffer<uint8_t> aux_buffer;
    const std::size_t aux_size = find_pair_sort_buffer_size<element_type, element_type>(build_size);
    aux_buffer.alloc(aux_size);

    auto* elements_ptr        = elements_buffer.ptr();
    auto* dummy_ptr           = dummy_buffer.ptr();
    auto* output_elements_ptr = output_elements_buffer.ptr();
    auto* output_dummy_ptr    = output_dummy_buffer.ptr();
    auto* aux_ptr             = aux_buffer.ptr();

    // ---- SORT TIMING at exactly 2^build_size_log keys ----
    {
        const std::size_t sort_size_log = build_size_log;
        const std::size_t sort_size     = std::size_t(1) << sort_size_log;

        double time_ms = 0.0;
        {
            scoped_cuda_timer timer(stream, &time_ms);
            for (std::size_t i = 0; i < runs; ++i) {
                untimed_pair_sort(aux_ptr, aux_size,
                                  elements_ptr,        output_elements_ptr,
                                  dummy_ptr,           output_dummy_ptr,
                                  sort_size,           stream);
            }
        }
        rc::auto_commit_result(rc_sort)
            .add_parameter("key_size", sizeof(element_type))
            .add_parameter("sort_size_log", sort_size_log)
            .add_parameter("sort_size", sort_size)
            .add_measurement("time_ms_avg", time_ms / runs)
            .add_measurement("time_ms_total", time_ms);
    }

    // ---- COALESCED ACCESS MICRO ----
    for (std::size_t threads_per_block_log = 5; threads_per_block_log <= 10; ++threads_per_block_log) {
        const std::size_t threads_per_block = std::size_t(1) << threads_per_block_log;

        for (std::size_t group_size_log = 0; group_size_log <= 10; ++group_size_log) {
            const std::size_t group_size = std::size_t(1) << group_size_log;

            double time_ms = 0.0;
            {
                scoped_cuda_timer timer(stream, &time_ms);
                for (std::size_t i = 0; i < runs; ++i) {
                    lambda_kernel<<<SDIV(lookup_size, threads_per_block), threads_per_block, 0, stream>>>([=] DEVICEQUALIFIER () {
                        const std::size_t tid       = threadIdx.x + blockIdx.x * blockDim.x;
                        const std::size_t gid       = tid >> group_size_log;
                        const std::size_t local_tid = tid & (group_size - 1);

                        std::uint32_t offset = static_cast<std::uint32_t>(gid);
                        element_type agg = 0;
                        #pragma unroll 1
                        for (std::uint32_t it = 0; it < 1024u; ++it) {
                            offset = 1919191919u * offset + 555555555u; // LCG
                            const std::size_t actual_offset = (static_cast<std::size_t>(offset) + local_tid) & (build_size - 1);
                            agg += elements_ptr[actual_offset];
                        }
                        output_elements_ptr[tid] = agg;
                    });
                }
            }

            rc::auto_commit_result(rc_coalesced)
                .add_parameter("key_size", sizeof(element_type))
                .add_parameter("threads_per_block_log", threads_per_block_log)
                .add_parameter("group_size_log", group_size_log)
                .add_measurement("time_ms_avg", time_ms / runs)
                .add_measurement("time_ms_total", time_ms);
        }
    }

    rc_sort.write_csv(std::cout, rc::first_line_header, rc::wide_form, rc::pad_columns);
    rc_coalesced.write_csv(std::cout, rc::first_line_header, rc::wide_form, rc::pad_columns);

    cudaStreamDestroy(stream);
}

// -------------------- main --------------------
int main()
{
    // 2^24 = 16,777,216 ≈ 16.8M keys
    constexpr std::size_t build_size_log = 24;
    constexpr std::size_t lookup_size_log = 16; // keep as in your code
    constexpr std::size_t runs = 20;

    std::printf("Running micro-benchmark with 2^%zu = %zu keys\n",
                build_size_log, (std::size_t{1} << build_size_log));

    // Choose key type; change to uint64_t if needed.
    micro_benchmark_sort_and_coalesced_onesize<std::uint32_t>(build_size_log, lookup_size_log, runs);

    return 0;
}
