// =============================================================================
// File: benchmarks.cuh
// Author: Justus Henneberg
// Description: Implements benchmarks     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef BENCHMARKS_CUH
#define BENCHMARKS_CUH

#include "input_generation.h"
#include "definitions.cuh"
#include "result_collector.h"
#include "utilities.cuh"

#if !defined(GPULSMOPT)
#include <nvtx3/nvtx3.hpp>
#endif

#include <algorithm>
#include <iterator>
#include <thread>


extern const char* cache_directory;


// for nvtx
#if !defined(GPULSMOPT)
struct nvtx_benchmark_domain{ static constexpr char const* name{"benchmark"}; };
#endif


template <typename key_type>
size_t estimate_point_query_setup_bytes(size_t build_size, size_t probe_size) {
    return build_size * sizeof(key_type) + probe_size * (sizeof(key_type) + sizeof(smallsize));
}


template <typename key_type>
size_t estimate_range_query_setup_bytes(size_t build_size, size_t probe_size) {
    return probe_size * sizeof(key_type) + estimate_point_query_setup_bytes<key_type>(build_size, probe_size);
}


struct test_configuration {
    std::string description;
    std::vector<size_t> log_build_size_options;
    std::vector<size_t> log_probe_size_options;
    std::vector<std::pair<size_t, size_t>> misses_percent_options;
    std::vector<size_t> build_key_uniformity_percent_options;
    std::vector<double> probe_zipf_coefficient_options;
    std::vector<size_t> log_key_multiplicity_options;
    std::vector<bool> sort_insert_options;
    std::vector<bool> sort_probe_options;
};


template <typename index_type>
void benchmark_point_query(
    rc::result_collector& rc,
    size_t runs,
    bool single_plane,
    bool run_advanced_tests,
    size_t max_log_build_size,
    size_t max_log_probe_size
) {

#if !defined(GPULSMOPT)
    nvtx3::scoped_range_in<nvtx_benchmark_domain> index_experiment{index_type::short_description()};
#endif

    using key_type = typename index_type::key_type;

    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);

#if 1
#warning "a test setup is configured, pre-configured benchmarks will be skipped"
    std::vector<test_configuration> test_configuration_options {
        {
            "test",
            {26}, // log_build_size_options
            {27}, // log_probe_size_options
            //{{0, 0}}, // misses_percent_options
            {{0, 0}, {1, 0}, {10, 0}, {20, 0}, {50, 0}, {100, 0}}, // misses_percent_options
            {100}, // build_key_uniformity_percent_options
            {0.0}, // probe_zipf_coefficient_options
            {0}, // log_key_multiplicity_options
            {false}, // sort_insert_options
            {false} // sort_probe_options
        }
    };
#else
    std::vector<test_configuration> test_configuration_options {
        {
            "basic",
            {26}, // log_build_size_options
            {27}, // log_probe_size_options
            {{0, 0}}, // misses_percent_options
            {100}, // build_key_uniformity_percent_options
            {0.0}, // probe_zipf_coefficient_options
            {0}, // log_key_multiplicity_options
            {false, true}, // sort_insert_options
            {false, true} // sort_probe_options
        }
    };
    std::vector<test_configuration> advanced_test_configuration_options {
        {
            "probe_size",
            {26}, // log_build_size_options
            {9, 12, 15, 18, 21, 24, 27, 28, 29}, // log_probe_size_options
            {{0, 0}}, // misses_percent_options
            {100}, // build_key_uniformity_percent_options
            {0.0}, // probe_zipf_coefficient_options
            {0}, // log_key_multiplicity_options
            {false}, // sort_insert_options
            {false, true} // sort_probe_options
        },
        {
            "build_size",
            {20, 21, 22, 23, 24, 25, 26, 27, 28, /*29*/}, // log_build_size_options
            {27}, // log_probe_size_options
            {{0, 0}}, // misses_percent_options
            {100, 20, 0}, // build_key_uniformity_percent_options
            {0.0}, // probe_zipf_coefficient_options
            {0}, // log_key_multiplicity_options
            {false, true}, // sort_insert_options
            {false} // sort_probe_options
        },
        {
            "build_skew",
            {26}, // log_build_size_options
            {27}, // log_probe_size_options
            {{0, 0}}, // misses_percent_options
            {100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1, 0}, // build_key_uniformity_percent_options
            {0.0}, // probe_zipf_coefficient_options
            {0}, // log_key_multiplicity_options
            {false}, // sort_insert_options
            {false, true} // sort_probe_options
        },
        {
            "misses",
            {26}, // log_build_size_options
            {27}, // log_probe_size_options
            {{0, 0}, {1, 0}, {10, 0}, {30, 0}, {50, 0}, {70, 0}, {90, 0}, {99, 0}, {100, 0}, {50, 50}, {0, 100}}, // misses_percent_options
            {100}, // build_key_uniformity_percent_options
            {0.0}, // probe_zipf_coefficient_options
            {0}, // log_key_multiplicity_options
            {false}, // sort_insert_options
            {false, true} // sort_probe_options
        },
        {
            "probe_skew",
            {26}, // log_build_size_options
            {27}, // log_probe_size_options
            {{0, 0}}, // misses_percent_options
            {100}, // build_key_uniformity_percent_options
            {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0}, // probe_zipf_coefficient_options
            {0}, // log_key_multiplicity_options
            {false}, // sort_insert_options
            {false, true} // sort_probe_options
        },
        {
            "key_multiplicity",
            {26}, // log_build_size_options
            {27}, // log_probe_size_options
            {{0, 0}}, // misses_percent_options
            {100}, // build_key_uniformity_percent_options
            {0.0}, // probe_zipf_coefficient_options
            {0, 1, 2, 3, 4, 5, 6, 7, 8}, // log_key_multiplicity_options
            {false}, // sort_insert_options
            {false, true} // sort_probe_options
        }
    };
    if (run_advanced_tests) {
        std::copy(
            advanced_test_configuration_options.begin(),
            advanced_test_configuration_options.end(),
            std::back_inserter(test_configuration_options));
    }
#endif

    for (size_t tci = 0; tci < test_configuration_options.size(); ++tci) {
        const auto& tc = test_configuration_options[tci];
    for (auto log_build_size : tc.log_build_size_options) {
    for (auto log_probe_size : tc.log_probe_size_options) {
    for (auto [misses_percent, outlier_misses_percent] : tc.misses_percent_options) {
    for (auto build_key_uniformity_percent : tc.build_key_uniformity_percent_options) {
    for (auto probe_zipf_coefficient : tc.probe_zipf_coefficient_options) {
    for (auto log_key_multiplicity : tc.log_key_multiplicity_options) {
    for (auto sort_insert : tc.sort_insert_options) {
    for (auto sort_probe : tc.sort_probe_options) {

        auto key_bits_used = single_plane && sizeof(key_type) > 4 ? 46 : sizeof(key_type) * 8;

        std::cerr << "experiment " << tc.description << " for " << index_type::short_description() << std::endl;
        std::cerr << "  bits: " << sizeof(key_type) * 8 << std::endl;
        std::cerr << "  bits used: " << key_bits_used << std::endl;
        std::cerr << "  log_build_size: " << log_build_size << std::endl;
        std::cerr << "  log_probe_size: " << log_probe_size << std::endl;
        std::cerr << "  misses_percent: " << misses_percent << std::endl;
        std::cerr << "  outlier_misses_percent: " << outlier_misses_percent << std::endl;
        std::cerr << "  build_key_uniformity_percent: " << build_key_uniformity_percent << std::endl;
        std::cerr << "  probe_zipf_coefficient: " << probe_zipf_coefficient << std::endl;
        std::cerr << "  log_key_multiplicity: " << log_key_multiplicity << std::endl;
        std::cerr << "  sort_insert: " << sort_insert << std::endl;
        std::cerr << "  sort_probe: " << sort_probe << std::endl;
        std::cerr << "  min_usable_key: " << min_usable_key<key_type>(key_bits_used) << std::endl;
        std::cerr << "  max_usable_key: " << max_usable_key<key_type>(key_bits_used) << std::endl;

        if (log_build_size > max_log_build_size)
            continue;
        if (log_probe_size > max_log_probe_size)
            continue;

#if !defined(GPULSMOPT)
        nvtx3::scoped_range_in<nvtx_benchmark_domain> experiment{tc.description};
#endif

        size_t build_size = size_t{1} << log_build_size;
        size_t probe_size = size_t{1} << log_probe_size;
        size_t key_multiplicity = size_t{1} << log_key_multiplicity;

        // index does not support operation
        bool skip = key_multiplicity > 1 ? !index_type::can_multi_lookup : !index_type::can_lookup;
        if (skip) continue;

        // not enough memory
        if (free_memory < (estimate_point_query_setup_bytes<key_type>(build_size, probe_size) + index_type::estimate_build_bytes(build_size)) * 11 / 10)
            continue;

        rti_assert(build_key_uniformity_percent <= 100);
        rti_assert(misses_percent <= 100);
        rti_assert(outlier_misses_percent <= 100);
        rti_assert(misses_percent + outlier_misses_percent <= 100);

        std::optional<point_query_dataset<key_type, smallsize>> input_data;
        {
#if !defined(GPULSMOPT)
            nvtx3::scoped_range_in<nvtx_benchmark_domain> gen{"input-gen"};
#endif
            input_data.emplace(
                seed,
                cache_directory,
                build_size,
                probe_size,
                1,
                0,
                sort_insert,
                sort_probe,
                build_key_uniformity_percent,
                probe_zipf_coefficient,
                misses_percent,
                outlier_misses_percent,
                key_multiplicity,
                false,
                min_usable_key<key_type>(key_bits_used),
                max_usable_key<key_type>(key_bits_used));
        }

        cuda_buffer<key_type> build_keys_buffer, probe_keys_buffer;
        cuda_buffer<smallsize> result_buffer;
        build_keys_buffer.alloc_and_upload(input_data->build_keys);
        probe_keys_buffer.alloc_and_upload(input_data->probe_keys);
        result_buffer.alloc(probe_size);
        result_buffer.zero();

        std::cerr << " -> setup complete" << std::endl;

        for (size_t run = 0; run < runs; ++run) {
            double build_time_ms = 0, sort_time_ms = 0, probe_time_ms = 0;
            size_t build_bytes = 0, gpu_resident_bytes = 0;

#if !defined(GPULSMOPT)
            nvtx3::scoped_range_in<nvtx_benchmark_domain> gen{"run-" + std::to_string(run)};
#endif

            index_type index;
            {
#if !defined(GPULSMOPT)
                nvtx3::scoped_range_in<nvtx_benchmark_domain> build{"build-phase"};
#endif
                index.build(build_keys_buffer, build_size, &build_time_ms, &build_bytes);
            }

            gpu_resident_bytes = index.gpu_resident_bytes();

            std::cerr << " -> " << gpu_resident_bytes << " bytes" << std::endl;

            // alloc sort buffers
            cuda_buffer<uint8_t> sort_temp_buffer;
            cuda_buffer<key_type> sorted_probe_keys_buffer;
            size_t sort_temp_bytes = 0;
            if (sort_probe) {
                sort_temp_bytes = find_sort_buffer_size<key_type>(probe_size);
                sort_temp_buffer.alloc(sort_temp_bytes);
                sorted_probe_keys_buffer.alloc(probe_size);
            }

            cuda_timer timer(0);
            {
#if !defined(GPULSMOPT)
                nvtx3::scoped_range_in<nvtx_benchmark_domain> lookup{"lookup-phase"};
#endif

                key_type* probe_keys_pointer = probe_keys_buffer;

                if (sort_probe) {
#if !defined(GPULSMOPT)
                    nvtx3::scoped_range_in<nvtx_benchmark_domain> sort_batch{"sort"};
#endif
                    timer.start();
                    untimed_sort(sort_temp_buffer.raw_ptr, sort_temp_bytes, probe_keys_buffer.ptr(), sorted_probe_keys_buffer.ptr(), probe_size);
                    timer.stop();
                    probe_keys_pointer = sorted_probe_keys_buffer;
                    sort_time_ms = timer.time_ms();
                }

                {
#if !defined(GPULSMOPT)
                    nvtx3::scoped_range_in<nvtx_benchmark_domain> lookup_batch{"lookup"};
#endif
                    timer.start();
                    if (key_multiplicity > 1) {
                        index.multi_lookup_sum(probe_keys_pointer, result_buffer.ptr(), probe_size, 0);
                    } else {
                        index.lookup(probe_keys_pointer, result_buffer.ptr(), probe_size, 0);
                    }
                    timer.stop();
                    probe_time_ms = timer.time_ms();
                }
            }
            cudaDeviceSynchronize(); CUERR

            std::cerr << " -> " << probe_time_ms << " ms" << std::endl;

            check_result(input_data->probe_keys, input_data->probe_keys, input_data->expected_result, result_buffer);

            rc::auto_commit_result(rc)
                .add_parameter("EXPERIMENT", tc.description)
                .add_parameter("index_type", index_type::short_description())
                .add_parameter("run", run)
                .add_parameter("key_bits", sizeof(key_type) * 8)
                .add_parameter("key_bits_used", key_bits_used)
                .add_parameter("log_build_size", log_build_size)
                .add_parameter("log_probe_size", log_probe_size)
                .add_parameter("misses_percent", misses_percent)
                .add_parameter("outlier_misses_percent", outlier_misses_percent)
                .add_parameter("build_key_uniformity_percent", build_key_uniformity_percent)
                .add_parameter("probe_zipf_coefficient", probe_zipf_coefficient)
                .add_parameter("log_key_multiplicity", log_key_multiplicity)
                .add_parameter("sort_insert", sort_insert)
                .add_parameter("sort_probe", sort_probe)
                .add_parameter("min_usable_key", min_usable_key<key_type>(key_bits_used))
                .add_parameter("max_usable_key", max_usable_key<key_type>(key_bits_used))
                .add_measurement("build_time_ms", build_time_ms)
                .add_measurement("sort_time_ms", sort_time_ms)
                .add_measurement("probe_time_ms", probe_time_ms)
                .add_measurement("build_bytes", build_bytes)
                .add_measurement("gpu_resident_bytes", gpu_resident_bytes)
                ;
        }
    }}}}}}}}}
}


template <typename index_type>
void benchmark_point_query(rc::result_collector& rc, size_t runs, bool single_plane, bool run_advanced_tests) {
    benchmark_point_query<index_type>(rc, runs, single_plane, run_advanced_tests, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max());
}


struct range_query_configuration {
    size_t log_build_size;
    size_t log_probe_size;
    size_t log_key_range_multiplier;
    size_t log_expected_hits;
};


template <typename index_type>
void benchmark_range_query(
        rc::result_collector& rc,
        size_t runs,
        size_t max_log_build_size,
        size_t max_log_probe_size
) {

#if !defined(GPULSMOPT)
    nvtx3::scoped_range_in<nvtx_benchmark_domain> index_experiment{index_type::short_description()};
#endif

    using key_type = typename index_type::key_type;

    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);

#if defined(RANGE_BUILD_SIZE_LOG) || defined(RANGE_PROBE_SIZE_LOG) || \
    defined(RANGE_KEY_RANGE_MULTIPLIER_LOG) || defined(RANGE_EXPECTED_HITS_LOG) || \
    defined(RANGE_SORT_PROBE)
#ifndef RANGE_BUILD_SIZE_LOG
#define RANGE_BUILD_SIZE_LOG 20
#endif
#ifndef RANGE_PROBE_SIZE_LOG
#define RANGE_PROBE_SIZE_LOG 14
#endif
#ifndef RANGE_KEY_RANGE_MULTIPLIER_LOG
#define RANGE_KEY_RANGE_MULTIPLIER_LOG 2
#endif
#ifndef RANGE_EXPECTED_HITS_LOG
#define RANGE_EXPECTED_HITS_LOG 8
#endif
#ifndef RANGE_SORT_PROBE
#define RANGE_SORT_PROBE 0
#endif
#warning "custom range query setup is configured"
    std::vector<range_query_configuration> configuration_options {
        {RANGE_BUILD_SIZE_LOG, RANGE_PROBE_SIZE_LOG, RANGE_KEY_RANGE_MULTIPLIER_LOG, RANGE_EXPECTED_HITS_LOG}
    };
    std::vector<bool> sort_insert_options {false};
    std::vector<bool> sort_probe_options {static_cast<bool>(RANGE_SORT_PROBE)};
#elif 0
#warning "a test setup is configured, pre-configured benchmarks will be skipped"
    std::vector<range_query_configuration> configuration_options { {26, 18, 3, 15}, {26, 18, 3, 10} };
    std::vector<bool> sort_insert_options {false};
    std::vector<bool> sort_probe_options {false};
#else
    std::vector<range_query_configuration> configuration_options {
        // mixed workload
        {26, 16, 0, 0},
        {26, 16, 0, 4},
        {26, 16, 0, 8},
        {26, 16, 0, 12},
        {26, 16, 0, 16},
        {26, 16, 0, 20},
        {26, 16, 0, 24},
        // dense keys
        //{26, 27, 0,  0},
        //{26, 27, 0,  2},
        //{26, 27, 0,  4},
        //{26, 27, 0,  6},
        //{26, 27, 0,  8},
        //{26, 27, 0, 10},
        // sparse keys
        //{26, 27,  2, 10},
        //{26, 27,  4, 10},
        //{26, 27,  6, 10},
        //{26, 27,  8, 10},
        //{26, 27, 10, 10},
        // smaller build size
        //{25, 27, 6, 10},
        //{24, 27, 6, 10},
        //{23, 27, 6, 10},
        //{22, 27, 6, 10},
        //{21, 27, 6, 10},
        //{20, 27, 6, 10},
        //{19, 27, 6, 10}
    };
    std::vector<bool> sort_insert_options {false};
    std::vector<bool> sort_probe_options {false, true};
#endif

    for (auto configuration : configuration_options) {
    for (bool sort_insert : sort_insert_options) {
    for (bool sort_probe : sort_probe_options) {

        size_t log_build_size = configuration.log_build_size;
        size_t log_probe_size = configuration.log_probe_size;
        size_t log_key_range_multiplier = configuration.log_key_range_multiplier;
        size_t log_expected_hits = configuration.log_expected_hits;

        std::cerr << "experiment range_query for " << index_type::short_description() << std::endl;
        std::cerr << "  bits: " << sizeof(key_type) * 8 << std::endl;
        std::cerr << "  log_build_size: " << log_build_size << std::endl;
        std::cerr << "  log_probe_size: " << log_probe_size << std::endl;
        std::cerr << "  log_key_range_multiplier: " << log_key_range_multiplier << std::endl;
        std::cerr << "  log_expected_hits: " << log_expected_hits << std::endl;
        std::cerr << "  sort_insert: " << sort_insert << std::endl;
        std::cerr << "  sort_probe: " << sort_probe << std::endl;

        if (log_build_size > max_log_build_size)
            continue;
        if (log_probe_size > max_log_probe_size)
            continue;

        size_t build_size = size_t{1} << log_build_size;
        size_t probe_size = size_t{1} << log_probe_size;
        size_t expected_hits = key_type{1} << log_expected_hits;
        size_t key_range_multiplier = size_t{1} << log_key_range_multiplier;
        size_t key_range = build_size * key_range_multiplier;

        // make sure all keys can be represented
        if (log_key_range_multiplier + log_build_size >= 8 * sizeof(key_type))
            continue;
        // make sure there are enough keys to allow a range query of the specified size (with added margin)
        if (expected_hits * 2 > key_range)
            continue;
        if (free_memory < (estimate_range_query_setup_bytes<key_type>(build_size, probe_size) + index_type::estimate_build_bytes(build_size)) * 12 / 10)
            continue;

        std::optional<range_query_dataset<key_type, smallsize>> input_data;
        {
#if !defined(GPULSMOPT)
            nvtx3::scoped_range_in<nvtx_benchmark_domain> gen{"input-gen"};
#endif
            input_data.emplace(
                    seed,
                    cache_directory,
                    build_size,
                    probe_size,
                    expected_hits,
                    key_range_multiplier,
                    sort_insert,
                    sort_probe,
                    true,
                    false,
                    min_usable_key<key_type>(sizeof(key_type) * 8),
                    max_usable_key<key_type>(sizeof(key_type) * 8));
        }

        cuda_buffer<key_type> build_keys_buffer, lower_keys_buffer, upper_keys_buffer;
        cuda_buffer<smallsize> result_buffer;
        build_keys_buffer.alloc_and_upload(input_data->build_keys);
        lower_keys_buffer.alloc_and_upload(input_data->lower_probe_keys);
        upper_keys_buffer.alloc_and_upload(input_data->upper_probe_keys);
        result_buffer.alloc(probe_size);
        result_buffer.zero();

        std::cerr << " -> setup complete" << std::endl;

        for (size_t run = 0; run < runs; ++run) {
            double build_time_ms = 0, sort_time_ms = 0, probe_time_ms = 0;
            size_t build_bytes = 0, gpu_resident_bytes = 0;

#if !defined(GPULSMOPT)
            nvtx3::scoped_range_in<nvtx_benchmark_domain> gen{"run-" + std::to_string(run)};
#endif

            index_type index;
            {
#if !defined(GPULSMOPT)
                nvtx3::scoped_range_in<nvtx_benchmark_domain> build{"build-phase"};
#endif
                index.build(build_keys_buffer, build_size, &build_time_ms, &build_bytes);
            }

            gpu_resident_bytes = index.gpu_resident_bytes();

            // alloc sort buffers
            cuda_buffer<uint8_t> sort_temp_buffer;
            cuda_buffer<key_type> sorted_lower_keys_buffer, sorted_upper_keys_buffer;
            size_t sort_temp_bytes = 0;
            if (sort_probe) {
                sort_temp_bytes = find_pair_sort_buffer_size<key_type, key_type>(probe_size);
                sort_temp_buffer.alloc(sort_temp_bytes);
                sorted_lower_keys_buffer.alloc(probe_size);
                sorted_upper_keys_buffer.alloc(probe_size);
            }

            cuda_timer timer(0);
            {
#if !defined(GPULSMOPT)
                nvtx3::scoped_range_in<nvtx_benchmark_domain> lookup{"lookup-phase"};
#endif

                key_type* lower_keys_pointer = lower_keys_buffer;
                key_type* upper_keys_pointer = upper_keys_buffer;

                if (sort_probe) {
#if !defined(GPULSMOPT)
                    nvtx3::scoped_range_in<nvtx_benchmark_domain> sort_batch{"sort"};
#endif
                    timer.start();
                    untimed_pair_sort(sort_temp_buffer.raw_ptr, sort_temp_bytes,
                                      lower_keys_buffer.ptr(), sorted_lower_keys_buffer.ptr(),
                                      upper_keys_buffer.ptr(), sorted_upper_keys_buffer.ptr(), probe_size, 0);
                    timer.stop();
                    lower_keys_pointer = sorted_lower_keys_buffer;
                    upper_keys_pointer = sorted_upper_keys_buffer;
                    sort_time_ms = timer.time_ms();
                }
                {
#if !defined(GPULSMOPT)
                    nvtx3::scoped_range_in<nvtx_benchmark_domain> lookup_batch{"lookup"};
#endif
                    timer.start();
                    index.range_lookup_sum(lower_keys_pointer, upper_keys_pointer, result_buffer.ptr(), probe_size, 0);
                    timer.stop();
                    probe_time_ms = timer.time_ms();
                }
            }
            cudaDeviceSynchronize(); CUERR

            std::cerr << " -> " << probe_time_ms << " ms" << std::endl;

            check_result(input_data->lower_probe_keys, input_data->upper_probe_keys, input_data->expected_result, result_buffer);

            rc::auto_commit_result(rc)
                .add_parameter("EXPERIMENT", "range_query")
                .add_parameter("index_type", index_type::short_description())
                .add_parameter("run", run)
                .add_parameter("key_bits", sizeof(key_type) * 8)
                .add_parameter("log_build_size", log_build_size)
                .add_parameter("log_probe_size", log_probe_size)
                .add_parameter("log_key_range_multiplier", log_key_range_multiplier)
                .add_parameter("log_expected_hits", log_expected_hits)
                .add_parameter("sort_insert", sort_insert)
                .add_parameter("sort_probe", sort_probe)
                .add_parameter("min_usable_key", min_usable_key<key_type>(sizeof(key_type) * 8))
                .add_parameter("max_usable_key", max_usable_key<key_type>(sizeof(key_type) * 8))
                .add_measurement("build_time_ms", build_time_ms)
                .add_measurement("sort_time_ms", sort_time_ms)
                .add_measurement("probe_time_ms", probe_time_ms)
                .add_measurement("build_bytes", build_bytes)
                .add_measurement("gpu_resident_bytes", gpu_resident_bytes)
                ;
        }
    }}}
}


template <typename index_type>
void benchmark_range_query(rc::result_collector& rc, size_t runs) {
    benchmark_range_query<index_type>(rc, runs, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max());
}

#endif
