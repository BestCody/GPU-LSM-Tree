// =============================================================================
// File: benchmarks_updates_saveMarch15.cuh
// Author: Justus Henneberg
// Description: Implements benchmarks_updates_saveMarch15     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef BENCHMARKS_UPDATES_CUH
#define BENCHMARKS_UPDATES_CUH

#include <iostream>
#include <fstream>
#include <algorithm>
#include <utility>
#include <vector>
#include <random>
#include <unordered_set>
#include <cmath>
#include <random>
#include <fstream>
#include <filesystem>  // To check file existence

#include "../input_generation.h"

#define DEBUG_BENCHMARK_OUTPUT
#define REBUILD_ON

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <random>
#include <string>

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <random>

template <typename key_type>
void generate_keys_hybrid_skip_readin(
    size_t size,
    size_t build_size,
    size_t insert_list_size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    const std::string& filename = "keys_cache.txt"
) {
    std::ifstream infile(filename);
    if (infile) {
        key_type key;
        while (infile >> key) {
            generated_keys.push_back(key);
        }
        infile.close();
        std::cerr << "Loaded " << generated_keys.size() << " keys from cache file: " << filename << std::endl;
        return; // Use existing keys if found
    }

    std::mt19937_64 gen(42); // Fixed seed for reproducibility
    std::unordered_set<key_type> unique_keys;
    std::cerr << "Generating Hybrid keys..." << std::endl;

    // -------------------- Step 1: Generate Uniform Keys --------------------
    std::cerr << "Generating Uniformly-distributed keys..." << std::endl;
    std::uniform_int_distribution<key_type> uniform_dist(min_usable_key, max_usable_key);
    std::vector<key_type> uniform_vector;

    while (uniform_vector.size() < build_size) {
        key_type key = uniform_dist(gen);
        if (unique_keys.insert(key).second) {
            uniform_vector.push_back(key);
        }
    }
    std::cerr << "Uniform keys generated: " << uniform_vector.size() << std::endl;

    // -------------------- Step 2: Generate Skip Pattern Keys --------------------
    std::cerr << "Generating Skip-Pattern keys..." << std::endl;
    std::vector<key_type> skip_pattern_vector;
    key_type start_value = 11;
    key_type skip_value = 111;
    key_type skip_jump = 100000;
    key_type new_key = 0;

    while (unique_keys.size() < size) {
        key_type current_key = start_value;
        new_key = current_key;

        for (size_t i = 0; i < insert_list_size; ++i) {
            if (unique_keys.size() >= size) break;
            if (unique_keys.insert(new_key).second) {
                skip_pattern_vector.push_back(new_key);
            }
            new_key += skip_value;
        }
        std::cerr << "Skip-Pattern Next Insert Batch " << skip_pattern_vector.size() << std::endl;
        start_value = new_key + skip_jump;
    }
    std::cerr << "Skip-Pattern keys generated: " << skip_pattern_vector.size() << std::endl;

    // -------------------- Step 3: Merge Uniform and Skip Keys --------------------
    generated_keys.clear();
    generated_keys.reserve(size);
    generated_keys.insert(generated_keys.end(), uniform_vector.begin(), uniform_vector.end());
    generated_keys.insert(generated_keys.end(), skip_pattern_vector.begin(), skip_pattern_vector.end());
    std::cerr << "Hybrid key generation completed." << std::endl;

    // -------------------- Step 4: Save to File --------------------
    std::ofstream outfile(filename, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        outfile.close();
        std::cerr << "Saved generated keys to cache: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save keys to file!" << std::endl;
    }
}

// --------------------------------------------------------------

template <typename key_type>
void generate_keys_hybrid_skip(
    size_t size,
    size_t build_size,
    size_t insert_list_size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    const std::string& filename = "keys_cache.txt"
) {
    std::mt19937_64 gen(42); // Fixed seed for reproducibility
    std::unordered_set<key_type> unique_keys; // Ensure uniqueness

    std::cerr << "Generating Hybrid keys..." << std::endl;

    // -------------------- Step 1: Generate Uniform Keys --------------------
    std::cerr << "Generating Uniformly-distributed keys..." << std::endl;

    std::uniform_int_distribution<key_type> uniform_dist(min_usable_key, max_usable_key);
    std::vector<key_type> uniform_vector;

    while (uniform_vector.size() < build_size) {
        key_type key = uniform_dist(gen);
        if (unique_keys.insert(key).second) {
            uniform_vector.push_back(key);
        }
    }

    std::cerr << "Uniform keys generated: " << uniform_vector.size() << std::endl;

// -------------------- Step 2: Generate Skip Pattern Keys --------------------
    std::cerr << "Generating Skip-Pattern keys..." << std::endl;

    std::vector<key_type> skip_pattern_vector;
    key_type start_value = 11; // First key
    key_type skip_value = 111; // Jump after each batch
    key_type skip_jump = 100000; // Jump after each batch
    key_type new_key =0;

    while (unique_keys.size() < size) {
        key_type current_key = start_value;
        new_key = current_key;

        // Generate batch of `insert_list_size` keys
        for (size_t i = 0; i < insert_list_size; ++i) {
            if (unique_keys.size() >= size) break; // Stop if we reach required size
            //new_key = current_key; // Skip pattern
            if (unique_keys.insert(new_key).second) {
                skip_pattern_vector.push_back(new_key);
            }
            new_key = new_key+ skip_value; // + i; // Skip pattern

        }
        std::cerr << "Skip-Pattern Next Insert Batch " << skip_pattern_vector.size() << std::endl;
        // Move to next batch
        start_value = new_key+ skip_jump;
    }

    std::cerr << "Skip-Pattern keys generated: " << skip_pattern_vector.size() << std::endl;

    // -------------------- Step 3: Merge Uniform and Skip Keys --------------------
    generated_keys.clear();
    generated_keys.reserve(size);

    generated_keys.insert(generated_keys.end(), uniform_vector.begin(), uniform_vector.end());
    generated_keys.insert(generated_keys.end(), skip_pattern_vector.begin(), skip_pattern_vector.end());

    std::cerr << "Hybrid key generation completed." << std::endl;

    // -------------------- Step 4: Save to File --------------------
    std::ofstream outfile(filename, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        outfile.close();
        std::cerr << "Saved generated keys to cache: " << filename << std::endl;
    } else {
            std::cerr << "Failed to save keys to file!" << std::endl;
    }
}

//--------------------------

template <typename key_type>
void generate_keys(
    size_t size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys
) {
    std::mt19937_64 gen(42); //ensuring deterministic behaviour
    std::uniform_int_distribution<key_type> uniform_key_dist(min_usable_key, max_usable_key);

    std::unordered_set<key_type> generated_keys_set; //unordered set so keys are unique, not repeating
    while (generated_keys_set.size() < size) {
        generated_keys_set.insert(uniform_key_dist(gen));
    }

    generated_keys.resize(size);
    std::copy(generated_keys_set.begin(), generated_keys_set.end(), generated_keys.begin());

    //randomness and reproducability due to Fixed Keys
}

template <typename key_type>
void draw_probes(
    size_t size,
    const std::vector<key_type>& build_keys,
    size_t active_range_start,
    size_t active_range_end,
    bool start_expected_offset_at_zero,
    std::vector<key_type>& probe_keys,
    std::vector<smallsize>& expected_result
) {
    std::mt19937_64 gen(42);
    std::uniform_int_distribution<size_t> index_dist(active_range_start, active_range_end - 1);

    //A uniform distribution (std::uniform_int_distribution<size_t> index_dist(active_range_start, active_range_end - 1)) 
    //selects random indices within the active range.

    probe_keys.resize(size);
    expected_result.resize(size);

    for (size_t i = 0; i < size; ++i) {
        size_t index = index_dist(gen);
        probe_keys[i] = build_keys[index];
        expected_result[i] = start_expected_offset_at_zero ? index - active_range_start : index;
    }
    // expected result is the just (offset)index of the key in the active range
}


void check_all_misses(const std::vector<smallsize>& output) {
    for (size_t i = 0; i < output.size(); ++i) {
        if (not_found != output[i]) {
            std::cerr << "data mismatch at index " << i << ": expected MISS, but received " << output[i] << std::endl;
            throw std::logic_error("stop");
        }
    }
}
template <typename key_type>
void check_all_misses_results(const std::vector<smallsize>& output, const std::vector<key_type>& probe_keys) {
    for (size_t i = 0; i < output.size(); ++i) {
        if (not_found != output[i]) {
            std::cerr << "data mismatch at index " << i << ": expected MISS for key " << probe_keys[i] << ", but received " << output[i] << std::endl;
            throw std::logic_error("stop");
        }
    }
}

struct update_configuration {
    std::string description;
    size_t build_size_log;
    size_t probe_size_log;
    size_t batch_count;
    size_t total_inserts_percentage_of_build_size;
    size_t total_deletes_percentage_of_build_size;
   // size_t cache_line;
   // size_t node_size_log;
    bool alternating_insert_delete;
    bool insert_before_delete;
};


template <typename index_type>
void benchmark_updates(
    rc::result_collector& rc,
    size_t runs
) {
    using key_type = typename index_type::key_type;

    constexpr bool supports_updates = index_type::can_update;

    std::vector<update_configuration> test_configuration_options {
        //{"delete-insert", 26, 27, 1, 10, 10, false, false},
        //{"insert-delete", 26, 27, 1, 10, 10, false, tsrue},
        //{"batches", 22, 23, 4, 150, 150, false, true},
        //{"batches", 22, 23, 2, 200, 200, false, true},
         {"batches", 25, 26, 8, 150, 150, false, true},
    };

    for (size_t tci = 0; tci < test_configuration_options.size(); ++tci) {
        const auto& tc = test_configuration_options[tci];

        rti_assert(tc.probe_size_log >= tc.build_size_log);

        size_t build_size = size_t{1} << tc.build_size_log;
        size_t key_generation_size = build_size * (100 + tc.total_inserts_percentage_of_build_size) / 100;
        size_t probe_size = size_t{1} << tc.probe_size_log;
        size_t insert_batch_size = build_size * tc.total_inserts_percentage_of_build_size / (100 * tc.batch_count);
        size_t delete_batch_size = build_size * tc.total_deletes_percentage_of_build_size / (100 * tc.batch_count);
        //size_t cache_line = tc.cache_line;
       // size_t node_size_log = size_t{1} << tc.node_size_log;

        rti_assert(key_generation_size * 2 < std::numeric_limits<smallsize>::max());

        std::cerr << "update experiment " << tc.description << " for " << index_type::short_description() << std::endl;
        std::cerr << "  bits: " << sizeof(key_type) * 8 << std::endl;
        std::cerr << "  build_size_log: " << tc.build_size_log << std::endl;
        std::cerr << "  probe_size_log: " << tc.probe_size_log << std::endl;
        std::cerr << "  batch_count: " << tc.batch_count << std::endl;
        std::cerr << "  alternating_insert_delete: " << tc.alternating_insert_delete << std::endl;
        std::cerr << "  insert_before_delete: " << tc.insert_before_delete << std::endl;

        std::cerr << "  total_inserts_percentage_of_build_size: " << tc.total_inserts_percentage_of_build_size << std::endl;
        std::cerr << "  total_deletes_percentage_of_build_size: " << tc.total_deletes_percentage_of_build_size << std::endl;
        //std::cerr << " cache_line: " << tc.cache_line << std::endl;
       // std::cerr << "node_size_log: " << tc.node_size_log << std::endl;

        std::cerr << "  insert_batch_size: " << insert_batch_size << std::endl;
        std::cerr << "  delete_batch_size: " << delete_batch_size << std::endl;

        /*
        // gen input
        std::vector<key_type> generated_keys;
        generate_keys(key_generation_size, min_usable_key<key_type>(), max_usable_key<key_type>(), generated_keys);
        //generate_keys_zipf(key_generation_size, min_usable_key<key_type>(), max_usable_key<key_type>(), generated_keys);
        */




        // -------------------- NEW GENERATED KEYS
        // Generate input keys
        std::vector<key_type> generated_keys;

        // Call generate_keys_hybrid
        generate_keys_hybrid_skip(
            key_generation_size,  // Total number of keys
            build_size,           // Number of uniform keys
            insert_batch_size,    // Number of insert keys  
            min_usable_key<key_type>(),
            max_usable_key<key_type>(),
            generated_keys
        );

            // Print build_size
        std::cout << "FINISHED HYBRID GEN: Build Size: " << build_size << std::endl;
#ifdef PRINT_GENERATED_HYBRID_KEYS
        // Print first `build_size` uniform keys
        std::cout << "Uniform Keys: " << std::endl;
        for (size_t i = 0; i < build_size; ++i) {
            std::cout << generated_keys[i] << " ";
            if (i % 10 == 9) std::cout << std::endl; // Print 10 keys per line
        }
        std::cout << "\n\n"; // Add gap

        // Print Zipf keys
        std::cout << "REST KEYS:" << std::endl;
        for (size_t i = build_size; i < generated_keys.size(); ++i) {
            std::cout << generated_keys[i] << std::endl; // One Zipf key per line
        }

        std::cout << std::endl;
#endif
        
        // ---------------------------------------------
        std::cerr << "generated key set of size " << key_generation_size << std::endl;

        cuda_buffer<key_type> insert_delete_keys_buffer;
        cuda_buffer<smallsize> insert_offsets_buffer;
        cuda_buffer<key_type> probe_keys_buffer;
        cuda_buffer<smallsize> result_buffer;

        insert_delete_keys_buffer.alloc(key_generation_size);
        insert_offsets_buffer.alloc(insert_batch_size);
        probe_keys_buffer.alloc(probe_size);
        result_buffer.alloc(probe_size);

        for (size_t run = 0; run < runs; ++run) {
            cuda_timer timer(0);

            std::cerr << "  run " << run << std::endl;

            size_t active_range_start = 0;
            size_t active_range_end = build_size;

            // upload initial build keys
            insert_delete_keys_buffer.upload(generated_keys.data(), active_range_end - active_range_start);

            // initial build (untimed)
            index_type index;
            index.build(insert_delete_keys_buffer.ptr(), active_range_end - active_range_start, nullptr, nullptr);
            #ifdef DEBUG_BENCHMARK_OUTPUT
                smallsize next_free= index.allocation_buffer_next_free();
                smallsize total_alloc_nodes= index.allocation_buffer_total_nodes();
                std::cerr << "After Build  Next Free: " << next_free<<  " of total available nodes: " << total_alloc_nodes << std::endl;
            #endif
            for (size_t step = 0; step < 2 * tc.batch_count + 1; ++step) {

                double insert_or_delete_time_ms = 0;
                double probe_time_ms = 0;
                double rebuild_time_ms = 0;
                size_t before_update_bytes = 0;
                size_t after_update_bytes = 0;

                bool do_insert =
                    !tc.alternating_insert_delete && tc.insert_before_delete  && (step < tc.batch_count + 1) ||
                    tc.alternating_insert_delete  && tc.insert_before_delete  && ((step & 1) == 1) ||
                    !tc.alternating_insert_delete && !tc.insert_before_delete && !(step < tc.batch_count + 1) ||
                    tc.alternating_insert_delete  && !tc.insert_before_delete && !((step & 1) == 1);
                bool do_delete = !do_insert;

                size_t new_active_range_end = do_insert ? active_range_end + insert_batch_size : active_range_end;
                size_t new_active_range_start = do_delete ? active_range_start + delete_batch_size : active_range_start;

                size_t before_update_size = active_range_end - active_range_start;
                size_t after_update_size = new_active_range_end - new_active_range_start;

                if (step == 0) {
                    std::cerr << "    step 0: probe only" << std::endl;
                    goto probe;
                }

                std::cerr << "    step " << step << ": ";
                std::cerr << (do_insert ? "insert" : do_delete ? "delete" : "nothing") << std::endl;

                // active range has to be non-empty
                if (new_active_range_start >= new_active_range_end) {
                    goto end_run;
                }

                before_update_bytes = index.gpu_resident_bytes();

                if constexpr (supports_updates) {

                    // a Dummy round of Insertions:
                    if (step == 1) {
                        size_t dummy_insert_size = 20; //2000;
                        if (dummy_insert_size > active_range_end - active_range_start) {
                            dummy_insert_size = active_range_end - active_range_start; // Ensure we don't exceed available keys
                        }
                        std::vector<key_type> dummy_inserts(
                            generated_keys.begin(),
                            generated_keys.begin() + dummy_insert_size
                        );

                        std::vector<smallsize> dummy_offsets(dummy_insert_size);
                        std::iota(dummy_offsets.begin(), dummy_offsets.end(), 0);

                        auto sort_perm_dum = sort_permutation(dummy_inserts, std::less<key_type>());
                        apply_permutation(dummy_inserts, sort_perm_dum);
                        apply_permutation(dummy_offsets, sort_perm_dum);

                        insert_delete_keys_buffer.upload(dummy_inserts.data(), dummy_insert_size);
                        insert_offsets_buffer.upload(dummy_offsets.data(), dummy_insert_size);

                        std::cerr << "Dummy insert size: " << dummy_insert_size << std::endl;
                        //NOTE Should be SORTING the DUMMY LISTS but it does not matter All keys already exist
                        // timer.start();
                        index.insert(insert_delete_keys_buffer.ptr(), insert_offsets_buffer.ptr(), dummy_insert_size, 0);
                        //timer.stop();
                        cudaDeviceSynchronize(); CUERR
                    }

                    // insert the next batch
                    if (do_insert) {
                        std::vector<key_type> relevant_inserts(generated_keys.begin() + active_range_end, generated_keys.begin() + active_range_end + insert_batch_size);
                        std::vector<smallsize> associated_offsets(insert_batch_size);
                        std::iota(associated_offsets.begin(), associated_offsets.end(), active_range_end);
                        auto sort_perm = sort_permutation(relevant_inserts, std::less<key_type>());
                        apply_permutation(relevant_inserts, sort_perm);
                        apply_permutation(associated_offsets, sort_perm);
                        insert_delete_keys_buffer.upload(relevant_inserts.data(), insert_batch_size);
                        insert_offsets_buffer.upload(associated_offsets.data(), insert_batch_size);
                        CUERR
                        std::cerr << "insert size: " << insert_batch_size << std::endl;
                        timer.start();
                        index.insert(insert_delete_keys_buffer.ptr(), insert_offsets_buffer.ptr(), insert_batch_size, 0);
                        timer.stop();
                        insert_or_delete_time_ms = timer.time_ms();
                        cudaDeviceSynchronize(); CUERR
                         #ifdef DEBUG_BENCHMARK_OUTPUT
                            next_free= index.allocation_buffer_next_free();
                            //smallsize total_alloc_nodes= index.allocation_buffer_total_nodes();
                             std::cerr << "After Insert  Next Free: " << next_free<<  " of total available nodes: " << total_alloc_nodes << std::endl;
                        #endif
                    }

                    // or delete the next batch
                    if (do_delete) {
                        std::vector<key_type> relevant_deletes(generated_keys.begin() + active_range_start, generated_keys.begin() + active_range_start + delete_batch_size);
                        std::sort(relevant_deletes.begin(), relevant_deletes.end());
                        insert_delete_keys_buffer.upload(relevant_deletes.data(), delete_batch_size);
                        CUERR
                        std::cerr << "delete size: " << delete_batch_size << std::endl;
                        timer.start();
                        index.remove(insert_delete_keys_buffer.ptr(), delete_batch_size, 0);
                        timer.stop();
                        insert_or_delete_time_ms = timer.time_ms();
                        cudaDeviceSynchronize(); CUERR
                        #ifdef DEBUG_BENCHMARK_OUTPUT
                            next_free= index.allocation_buffer_next_free();
                            //smallsize total_alloc_nodes= index.allocation_buffer_total_nodes();
                             std::cerr << "After Delete  Next Free: " << next_free<<  " of total available nodes: " << total_alloc_nodes << std::endl;
                        #endif
                    }
                }

                // adjust active range
                active_range_end = new_active_range_end;
                active_range_start = new_active_range_start;

                if constexpr (!supports_updates) {
                    index.destroy();
                    insert_delete_keys_buffer.upload(generated_keys.data() + active_range_start, active_range_end - active_range_start);
                    index.build(insert_delete_keys_buffer.ptr(), active_range_end - active_range_start, &insert_or_delete_time_ms, nullptr);
                }

                std::cerr << "     -> " << insert_or_delete_time_ms << " ms" << std::endl;

probe:;
                // probe all inserted keys
                if (active_range_end > active_range_start) {
                    std::vector<key_type> probe_keys;
                    std::vector<smallsize> expected_result;
                    draw_probes(probe_size, generated_keys, active_range_start, active_range_end, !supports_updates, probe_keys, expected_result);
                    probe_keys_buffer.upload(probe_keys.data(), probe_size);
                    result_buffer.zero();

                    timer.start();
                    index.lookup(probe_keys_buffer.ptr(), result_buffer.ptr(), probe_size, 0);
                    timer.stop();
                    probe_time_ms = timer.time_ms();
                    cudaDeviceSynchronize(); CUERR

                    check_result(probe_keys, probe_keys, expected_result, result_buffer);
                }

                // check if keys before the inserted range are all misses
               /*  if (active_range_start > 0) {
                    probe_keys_buffer.upload(generated_keys.data(), active_range_start);
                    result_buffer.zero();

                    index.lookup(probe_keys_buffer.ptr(), result_buffer.ptr(), active_range_start - 0, 0);
                    cudaDeviceSynchronize();

                    auto result = result_buffer.download(active_range_start - 0);
                    check_all_misses(result);
                }
                */

                /*
                // check if keys beyond the inserted range are all misses
                if (active_range_end < key_generation_size) {
                    probe_keys_buffer.upload(generated_keys.data() + active_range_end, key_generation_size - active_range_end);
                    result_buffer.zero();

                    index.lookup(probe_keys_buffer.ptr(), result_buffer.ptr(), key_generation_size - active_range_end, 0);
                    cudaDeviceSynchronize();

                    auto result = result_buffer.download(key_generation_size - active_range_end);
                    check_all_misses(result);
                }
                */
               

                // check if keys before the inserted range are all misses
                if (active_range_start > 0) {
                    probe_keys_buffer.upload(generated_keys.data(), active_range_start);
                    result_buffer.zero();

                    index.lookup(probe_keys_buffer.ptr(), result_buffer.ptr(), active_range_start, 0);
                    cudaDeviceSynchronize();

                    auto result = result_buffer.download(active_range_start);
                    std::vector<key_type> probe_keys(generated_keys.begin(), generated_keys.begin() + active_range_start);
                    check_all_misses_results<key_type>(result, probe_keys);
                }

                // check if keys beyond the inserted range are all misses
                if (active_range_end < key_generation_size) {
                    probe_keys_buffer.upload(generated_keys.data() + active_range_end, key_generation_size - active_range_end);
                    result_buffer.zero();

                    index.lookup(probe_keys_buffer.ptr(), result_buffer.ptr(), key_generation_size - active_range_end, 0);
                    cudaDeviceSynchronize();

                    auto result = result_buffer.download(key_generation_size - active_range_end);
                    std::vector<key_type> probe_keys(generated_keys.begin() + active_range_end, generated_keys.end());
                    check_all_misses_results<key_type>(result, probe_keys);
                }
                                                std::cerr << "     -> " << probe_time_ms << " ms" << std::endl;

                after_update_bytes = index.gpu_resident_bytes();


 
#ifdef REBUILD_ON
                // rebuild 
            rebuild_time_ms = 0;
            if (step%4 ==0) {
              timer.start();
              index.rebuild( nullptr, nullptr);
              timer.stop();
              rebuild_time_ms = timer.time_ms(); 
            }
              
#endif 
            #ifdef DEBUG_BENCHMARK_OUTPUT
                next_free= index.allocation_buffer_next_free();
                total_alloc_nodes= index.allocation_buffer_total_nodes();
#ifdef REBUILD_ON
                std::cerr << "After- Rebuild ON  Next Free: " << next_free<<  " of total available nodes: " << total_alloc_nodes << " Rebuild time: " << rebuild_time_ms << std::endl;
#else
                std::cerr << "After- NO Rebuild  Next Free: " << next_free<<  " of total available nodes: " << total_alloc_nodes << std::endl;   
#endif
            #endif  

                rc::auto_commit_result(rc)
                    .add_parameter("EXPERIMENT", tc.description)
                    .add_parameter("index_type", index_type::short_description())
                    .add_parameter("run", run)
                    .add_parameter("key_bits", sizeof(key_type) * 8)
                    .add_parameter("build_size_log", tc.build_size_log)
                    .add_parameter("probe_size_log", tc.probe_size_log)
                    .add_parameter("batch_count", tc.batch_count)
                    .add_parameter("total_inserts_percentage_of_build_size", tc.total_inserts_percentage_of_build_size)
                    .add_parameter("total_deletes_percentage_of_build_size", tc.total_deletes_percentage_of_build_size)
                    //.add_parameter("cache_line_size", tc.cache_line)
                   // .add_parameter("node_size_log", tc.node_size_log)
                    .add_parameter("alternating_insert_delete", tc.alternating_insert_delete)
                    .add_parameter("insert_before_delete", tc.insert_before_delete)
                    .add_parameter("step", step)
                    .add_parameter("do_insert", do_insert)
                    .add_parameter("do_delete", do_delete)
                    .add_parameter("current_batch_size", do_insert ? insert_batch_size : do_delete ? delete_batch_size : 0)
                    .add_parameter("before_update_size", before_update_size)
                    .add_parameter("after_update_size", after_update_size)
                    .add_measurement("insert_or_delete_time_ms", insert_or_delete_time_ms)
                    .add_measurement("probe_time_ms", probe_time_ms)
                    .add_measurement("before_update_bytes", before_update_bytes)
                    .add_measurement("after_update_bytes", after_update_bytes)
                    ;
end_run:;
            }
        }
    }
}

#endif



/*




template <typename key_type>
void generate_keys_hybrid_zifp(
    size_t size,
    size_t build_size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    const std::string& filename = "zipf_keys_cache.txt"
) {
    std::mt19937_64 gen(42); // Fixed seed for reproducibility

    size_t range = max_usable_key - min_usable_key + 1;

    // -------------------- Step 1: Generate Zipfian Keys Using `zipf_index_distribution` --------------------
    std::unordered_set<key_type> zipf_keys;
    std::vector<key_type> zipf_vector;

    std::cerr << "Generating Zipfian-distributed keys (Fast Version)..." << std::endl;

    double zipfian_s = 0.5;
    zipf_index_distribution zipf_dist(size - build_size, zipfian_s); // Using the fast Zipf distribution

    while (zipf_keys.size() < size - build_size) {
        key_type key = min_usable_key + zipf_dist(gen);  // Faster Zipfian sampling
        if (zipf_keys.insert(key).second) {
            zipf_vector.push_back(key);
        }
    }

    std::cerr << "Zipfian keys generated: " << zipf_keys.size() << std::endl;

    // -------------------- Step 2: Generate Uniform Keys --------------------
    std::unordered_set<key_type> uniform_keys;
    std::vector<key_type> uniform_vector;

    std::cerr << "Generating Uniformly-distributed keys..." << std::endl;

    std::uniform_int_distribution<key_type> uniform_dist(min_usable_key, max_usable_key);

    while (uniform_keys.size() < build_size) {
        key_type key = uniform_dist(gen);
        if (zipf_keys.find(key) == zipf_keys.end() && uniform_keys.insert(key).second) {
            uniform_vector.push_back(key);
        }
    }

    std::cerr << "Uniform keys generated: " << uniform_keys.size() << std::endl;

    // -------------------- Step 3: Merge Uniform and Zipfian Keys --------------------
    generated_keys.clear();
    generated_keys.reserve(size);

    generated_keys.insert(generated_keys.end(), uniform_vector.begin(), uniform_vector.end());
    generated_keys.insert(generated_keys.end(), zipf_vector.begin(), zipf_vector.end());

    std::cerr << "Hybrid key generation completed." << std::endl;

    // -------------------- Step 4: Save to File --------------------
    std::ofstream outfile(filename, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        outfile.close();
        std::cerr << "Saved generated keys to cache: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save keys to file!" << std::endl;
    }
}


template <typename key_type>
void generate_keys_zipf(
    size_t size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    const std::string& filename = "zipf_keys_cache.txt"
) {
    std::ifstream infile(filename);
    if (infile.good()) {
        key_type key;
        while (infile >> key) {
            generated_keys.push_back(key);
            if (generated_keys.size() >= size) break;
        }
        infile.close();
        if (generated_keys.size() == size) {
            std::cerr << "Loaded keys from cache: " << filename << std::endl;
            return;
        }
    }

    std::cerr << "Generating new Zipfian keys..." << std::endl;
    std::mt19937_64 gen(42); // Ensuring deterministic behavior
    size_t range = max_usable_key - min_usable_key + 1;

    // Compute Zipfian probabilities
    double zipfian_s = 0.5;
    double sum = 0.0;
    std::vector<double> probabilities(range);
    for (size_t i = 1; i <= range; ++i) {
        sum += 1.0 / std::pow(i, zipfian_s);
    }
    for (size_t i = 1; i <= range; ++i) {
        probabilities[i - 1] = (1.0 / std::pow(i, zipfian_s)) / sum;
    }

    std::discrete_distribution<size_t> zipf_dist(probabilities.begin(), probabilities.end());

    std::unordered_set<key_type> unique_keys;
    while (unique_keys.size() < size) {
        key_type key = min_usable_key + zipf_dist(gen);
        unique_keys.insert(key);
    }

    // Move unique keys to the output vector
    generated_keys.assign(unique_keys.begin(), unique_keys.end());

    // Write keys to file for future use
    std::ofstream outfile(filename, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        outfile.close();
        std::cerr << "Saved generated keys to cache: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save keys to file!" << std::endl;
    }




template <typename key_type>
void generate_keys_hybrid_skip_readin(
    size_t size,
    size_t build_size,
    size_t insert_list_size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    const std::string& filename = "keys_cache.txt"
) {
    std::ifstream infile(filename);
    generated_keys.clear();
    key_type key;

    if (infile) {
        std::cerr << "Reading keys from cache..." << std::endl;
        while (infile >> key) {
            generated_keys.push_back(key);
            if (generated_keys.size() >= size) {
                std::cerr << "Sufficient keys loaded from cache." << std::endl;
                return;
            }
        }
        infile.close();
    }
    
    std::cerr << "Generating Hybrid keys..." << std::endl;
    std::mt19937_64 gen(42); // Fixed seed for reproducibility
    std::unordered_set<key_type> unique_keys(generated_keys.begin(), generated_keys.end());

    // -------------------- Step 1: Generate Uniform Keys --------------------
    std::cerr << "Generating Uniformly-distributed keys..." << std::endl;
    std::uniform_int_distribution<key_type> uniform_dist(min_usable_key, max_usable_key);
    std::vector<key_type> uniform_vector;
    while (uniform_vector.size() < build_size) {
        key_type key = uniform_dist(gen);
        if (unique_keys.insert(key).second) {
            uniform_vector.push_back(key);
        }
    }
    std::cerr << "Uniform keys generated: " << uniform_vector.size() << std::endl;

    // -------------------- Step 2: Generate Skip Pattern Keys --------------------
    std::cerr << "Generating Skip-Pattern keys..." << std::endl;
    std::vector<key_type> skip_pattern_vector;
    key_type start_value = 111;
    key_type skip_jump = 100000;
    while (unique_keys.size() < size) {
        key_type current_key = start_value;
        for (size_t i = 0; i < insert_list_size; ++i) {
            if (unique_keys.size() >= size) break;
            key_type new_key = current_key + i;
            if (unique_keys.insert(new_key).second) {
                skip_pattern_vector.push_back(new_key);
            }
        }
        start_value += skip_jump;
    }
    std::cerr << "Skip-Pattern keys generated: " << skip_pattern_vector.size() << std::endl;

    // -------------------- Step 3: Merge and Save Keys --------------------
    generated_keys.insert(generated_keys.end(), uniform_vector.begin(), uniform_vector.end());
    generated_keys.insert(generated_keys.end(), skip_pattern_vector.begin(), skip_pattern_vector.end());
    std::cerr << "Hybrid key generation completed." << std::endl;
    
    std::ofstream outfile(filename, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        outfile.close();
        std::cerr << "Saved generated keys to cache: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save keys to file!" << std::endl;
    }
}


template <typename key_type>
void generate_keys_hybrid_skip_file(
    size_t size,
    size_t build_size,
    size_t insert_list_size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys
) {
    std::mt19937_64 gen(42); // Fixed seed for reproducibility

    // Define the cache file path
    std::string cache_filename = "data_cache/keys_cache_" + std::to_string(build_size) + ".txt";

    // -------------------- Step 1: Check for Existing Cache --------------------
    if (std::filesystem::exists(cache_filename)) {
        std::ifstream infile(cache_filename);
        if (infile.good()) {
            key_type key;
            while (infile >> key) {
                generated_keys.push_back(key);
                if (generated_keys.size() >= size) break;
            }
            infile.close();

            if (generated_keys.size() == size) {
                std::cerr << "Loaded keys from cache: " << cache_filename << std::endl;
                return;
            }
        }
    }

    // -------------------- Step 2: Generate Uniform Keys --------------------
    std::cerr << "Generating Uniformly-distributed keys..." << std::endl;

    std::unordered_set<key_type> unique_keys;
    std::vector<key_type> uniform_vector;
    std::uniform_int_distribution<key_type> uniform_dist(min_usable_key, max_usable_key);

    while (uniform_vector.size() < build_size) {
        key_type key = uniform_dist(gen);
        if (unique_keys.insert(key).second) {
            uniform_vector.push_back(key);
        }
    }

    std::cerr << "Uniform keys generated: " << uniform_vector.size() << std::endl;

    // -------------------- Step 3: Generate Skip Pattern Keys --------------------
    std::cerr << "Generating Skip-Pattern keys..." << std::endl;

    std::vector<key_type> skip_pattern_vector;
    key_type start_value = 111; // First key
    key_type skip_jump = 100000; // Jump after each batch

    while (unique_keys.size() < size) {
        key_type current_key = start_value;

        for (size_t i = 0; i < insert_list_size; ++i) {
            if (unique_keys.size() >= size) break;
            key_type new_key = current_key + i;
            if (unique_keys.insert(new_key).second) {
                skip_pattern_vector.push_back(new_key);
            }
        }

        start_value += skip_jump; // Move to next batch
    }

    std::cerr << "Skip-Pattern keys generated: " << skip_pattern_vector.size() << std::endl;

    // -------------------- Step 4: Merge Uniform and Skip Keys --------------------
    generated_keys.clear();
    generated_keys.reserve(size);

    generated_keys.insert(generated_keys.end(), uniform_vector.begin(), uniform_vector.end());
    generated_keys.insert(generated_keys.end(), skip_pattern_vector.begin(), skip_pattern_vector.end());

    std::cerr << "Hybrid key generation completed." << std::endl;

    // -------------------- Step 5: Save to File --------------------
    std::ofstream outfile(cache_filename, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        outfile.close();
        std::cerr << "Saved generated keys to cache: " << cache_filename << std::endl;
    } else {
        std::cerr << "Failed to save keys to file!" << std::endl;
    }
}

}*/