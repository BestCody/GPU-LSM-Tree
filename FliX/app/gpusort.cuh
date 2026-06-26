// =============================================================================
// File: gpusort.cuh
// Author: Justus Henneberg
// Description: Implements gpusort     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

template <typename element_type>
void mirco_benchmark_search_and_coalesced() {
    rc::result_collector rc_sort;
    rc::result_collector rc_coalesced;

    constexpr size_t runs = 20;

    size_t build_size_log = 28u;
    size_t lookup_size_log = 16u;
    size_t build_size = size_t(1) << build_size_log;
    size_t lookup_size = size_t(1) << lookup_size_log;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<element_type> elements(build_size);
    for (size_t i = 0; i < build_size; i++) {
        elements.push_back(static_cast<element_type>(i));
    }

    cuda_buffer<element_type> elements_buffer, dummy_buffer, output_elements_buffer, output_dummy_buffer;
    elements_buffer.alloc_and_upload(elements);
    dummy_buffer.alloc_and_upload(elements);
    output_elements_buffer.alloc_for_size(elements);
    output_dummy_buffer.alloc_for_size(elements);

    cuda_buffer<uint8_t> aux_buffer;
    size_t aux_size = find_pair_sort_buffer_size<element_type, element_type>(build_size);
    aux_buffer.alloc(aux_size);

    auto elements_ptr = elements_buffer.ptr();
    auto dummy_ptr = dummy_buffer.ptr();
    auto output_elements_ptr = output_elements_buffer.ptr();
    auto output_dummy_ptr = output_dummy_buffer.ptr();
    auto aux_ptr = aux_buffer.ptr();

    for (size_t sort_size_log = 8; sort_size_log <= build_size_log; sort_size_log++) {
        size_t sort_size = size_t(1) << sort_size_log;
        double time_ms = 0;
        {
            scoped_cuda_timer timer(stream, &time_ms);
            for (size_t i = 0; i < runs; i++) {
                untimed_pair_sort(aux_ptr, aux_size, elements_ptr, output_elements_ptr, dummy_ptr, output_dummy_ptr, sort_size, stream);
            }
        }
        rc::auto_commit_result(rc_sort)
            .add_parameter("key_size", sizeof(element_type))
            .add_parameter("sort_size_log", sort_size_log)
            .add_measurement("time_ms", time_ms / runs);
    }

    for (size_t threads_per_block_log = 5; threads_per_block_log <= 10; threads_per_block_log++) {
        size_t threads_per_block = size_t(1) << threads_per_block_log;

        for (size_t group_size_log = 0; group_size_log <= 10; group_size_log++) {
            size_t group_size = size_t(1) << group_size_log;
            double time_ms = 0;
            {
                scoped_cuda_timer timer(stream, &time_ms);
                for (size_t i = 0; i < runs; i++) {
                    lambda_kernel<<<SDIV(lookup_size, threads_per_block), threads_per_block, 0, stream>>>([=] DEVICEQUALIFIER () {
                        smallsize tid = threadIdx.x + blockIdx.x * blockDim.x;
                        smallsize gid = tid >> group_size_log;
                        smallsize local_tid = tid & (group_size - 1);

                        smallsize offset = gid;
                        smallsize agg = 0;
                        for (smallsize it = 0; it < 1024; ++it) {
                            offset = 1919191919u * offset + 555555555u;
                            auto actual_offset = (offset + local_tid) & (build_size - 1);
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
                .add_measurement("time_ms", time_ms / runs);
        }
    }
    rc_sort.write_csv(std::cout, rc::first_line_header, rc::wide_form, rc::pad_columns);
    rc_coalesced.write_csv(std::cout, rc::first_line_header, rc::wide_form, rc::pad_columns);
    cudaStreamDestroy(stream);
}