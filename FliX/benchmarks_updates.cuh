// =============================================================================
// File: benchmarks_updates.cuh
// Author: Justus Henneberg, Rosina Kharal
// Description: Implements benchmarks_updates     
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
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <random>
#include <fstream>

#include <limits>
#include <random>
#include <type_traits>
#include <thrust/system/cuda/execution_policy.h>
#include <iostream>
#include <stdexcept>

#include <filesystem> // To check file existence
// #include "utilities.cuh"
#include "benchmark_inputs.cuh"
#include "benchmark_lookup.cuh"
#include "input_generation.h"
#include "benchmarks.cuh" // for benchmark_range_query (merged range-sum benchmark)

// #define DEBUG_BENCHMARK_OUTPUT//
//#define REBUILD_GROUPS
//#define REBUILD_ON
// #define COMBINE_INSERT_DELETE
// #define COMPUTE_TOTALKEYS

#ifdef REBUILD_INTERVAL
constexpr smallsize rebuild_frequency = REBUILD_INTERVAL; // every round;
#else
constexpr smallsize rebuild_frequency = 1;
#endif

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
#include "input_generation.h"
#include "debug_definitions_updates.cuh"
#include "utilities.cuh"

#ifdef SHIFTED
constexpr bool shift_insert_range = true;
#else
constexpr bool shift_insert_range = false;
#endif

#ifndef UPDATE_INSERT_PERCENT
#define UPDATE_INSERT_PERCENT 25
#endif

#ifndef UPDATE_DELETE_PERCENT
#define UPDATE_DELETE_PERCENT UPDATE_INSERT_PERCENT
#endif

#ifdef UPDATE_INSERT_BATCH_LOG
constexpr size_t fixed_insert_batch_size =
    size_t{1} << UPDATE_INSERT_BATCH_LOG;
#endif

#ifdef UPDATE_DELETE_BATCH_LOG
constexpr size_t fixed_delete_batch_size = size_t{1} << UPDATE_DELETE_BATCH_LOG;
#endif

#ifndef UPDATE_DELETE_SEED
#define UPDATE_DELETE_SEED 42
#endif

constexpr uint64_t uniform_delete_seed = UPDATE_DELETE_SEED;

// probe_key_sort_timed.cuh

#ifndef GLOBALQUALIFIER
#define GLOBALQUALIFIER __global__
#endif

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

inline void cuda_check_impl(cudaError_t e, const char* expr, const char* file, int line) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", expr, file, line, cudaGetErrorString(e));
    std::fflush(stderr);
    std::abort();
  }
}

#define CUCHECK(expr) cuda_check_impl((expr), #expr, __FILE__, __LINE__)

#define CUDA_SYNC_POINT(tag) do {                                      \
  CUCHECK(cudaPeekAtLastError());                                      \
  CUCHECK(cudaDeviceSynchronize());                                    \
  std::fprintf(stderr, "[CUDA_SYNC_POINT] %s OK (%s:%d)\n",            \
               tag, __FILE__, __LINE__);                               \
} while(0)


template <typename index_type, typename = void>
struct benchmark_supports_successor : std::false_type {};

template <typename index_type>
struct benchmark_supports_successor<index_type, std::void_t<decltype(index_type::can_successor)>>
    : std::bool_constant<index_type::can_successor == operation_support::async> {};

template <typename index_type, typename = void>
struct benchmark_index_stores_tombstones : std::false_type {};

template <typename index_type>
struct benchmark_index_stores_tombstones<
    index_type, std::void_t<decltype(index_type::stores_tombstones)>>
    : std::bool_constant<index_type::stores_tombstones> {};

template <typename index_type, typename = void>
struct benchmark_update_storage_granularity
    : std::integral_constant<size_t, 1> {};

template <typename index_type>
struct benchmark_update_storage_granularity<
    index_type, std::void_t<decltype(index_type::update_storage_granularity)>>
    : std::integral_constant<size_t, index_type::update_storage_granularity> {};

template <typename index_type, typename = void>
struct benchmark_requires_sorted_updates : std::false_type {};

template <typename index_type>
struct benchmark_requires_sorted_updates<
    index_type, std::void_t<decltype(index_type::requires_sorted_updates)>>
    : std::bool_constant<index_type::requires_sorted_updates> {};

template <typename index_type>
void prepare_update_inputs(typename index_type::key_type *keys,
                           smallsize *values, size_t size, cudaStream_t stream)
{
    if constexpr (benchmark_requires_sorted_updates<index_type>::value)
    {
        if (size < 2) return;
        if (values)
            thrust::sort_by_key(thrust::cuda::par.on(stream), keys, keys + size, values);
        else
            thrust::sort(thrust::cuda::par.on(stream), keys, keys + size);
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
    std::vector<key_type> &generated_keys,
    const std::string &filename = "keys_cache.txt")
{
    std::mt19937_64 gen(42);                  // Fixed seed for reproducibility
    std::unordered_set<key_type> unique_keys; // Ensure uniqueness

    std::cerr << "Generating Hybrid keys..." << std::endl;

    // -------------------- Step 1: Generate Uniform Keys --------------------
    std::cerr << "Generating Uniformly-distributed keys..." << std::endl;

    std::uniform_int_distribution<key_type> uniform_dist(min_usable_key, max_usable_key);
    std::vector<key_type> uniform_vector;

    while (uniform_vector.size() < build_size)
    {
        key_type key = uniform_dist(gen);
        if (unique_keys.insert(key).second)
        {
            uniform_vector.push_back(key);
        }
    }

    std::cerr << "Uniform keys generated: " << uniform_vector.size() << std::endl;

    // -------------------- Step 2: Generate Skip Pattern Keys --------------------
    std::cerr << "Generating Skip-Pattern keys..." << std::endl;

    std::vector<key_type> skip_pattern_vector;
    key_type start_value = 11;   // First key
    key_type skip_value = 111;   // Jump after each batch
    key_type skip_jump = 100000; // Jump after each batch
    key_type new_key = 0;

    while (unique_keys.size() < size)
    {
        key_type current_key = start_value;
        new_key = current_key;

        // Generate batch of `insert_list_size` keys
        for (size_t i = 0; i < insert_list_size; ++i)
        {
            if (unique_keys.size() >= size)
                break; // Stop if we reach required size
            // new_key = current_key; // Skip pattern
            if (unique_keys.insert(new_key).second)
            { // make sure not a duplicate from uniform_vector
                skip_pattern_vector.push_back(new_key);
            }
            new_key = new_key + skip_value; // + i; // Skip pattern
        }
        std::cerr << "Skip-Pattern Next Insert Batch " << skip_pattern_vector.size() << std::endl;
        // Move to next batch
        start_value = new_key + skip_jump;
    }

    std::cerr << "Skip-Pattern keys generated: " << skip_pattern_vector.size() << std::endl;

    // -------------------- Step 3: Merge Uniform and Skip Keys --------------------
    generated_keys.clear();
    generated_keys.reserve(size);

    generated_keys.insert(generated_keys.end(), uniform_vector.begin(), uniform_vector.end());
    generated_keys.insert(generated_keys.end(), skip_pattern_vector.begin(), skip_pattern_vector.end());

    std::cerr << "Hybrid key generation completed." << std::endl;

#if FLIX_USE_KEY_CACHE
    // -------------------- Step 4: Save to File --------------------
    std::ofstream outfile(filename, std::ios::trunc);
    if (outfile)
    {
        for (const auto &key : generated_keys)
        {
            outfile << key << "\n";
        }
        outfile.close();
        std::cerr << "Saved generated keys to cache: " << filename << std::endl;
    }
    else
    {
        std::cerr << "Failed to save keys to file!" << std::endl;
    }
#endif
}

// Generate hybrid-dense keys using uniform + dense pattern strategy and save to text file

template <typename key_type>
void generate_keys_file(
    size_t size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type> &generated_keys,
    const std::string &base_filename)
{
    namespace fs = std::filesystem;

    std::string filename = "data_cache/" + base_filename;
#if FLIX_USE_KEY_CACHE
    if (fs::exists(filename))
    {
        std::ifstream infile(filename, std::ios::binary);
        if (infile)
        {
            generated_keys.resize(size);
            infile.read(reinterpret_cast<char *>(generated_keys.data()), sizeof(key_type) * size);
            return;
        }
    }
#endif

    std::mt19937_64 gen(42);
    std::uniform_int_distribution<key_type> uniform_key_dist(min_usable_key + 2, max_usable_key - 3);

    std::unordered_set<key_type> generated_keys_set;
    while (generated_keys_set.size() < size)
    {
        generated_keys_set.insert(uniform_key_dist(gen));
    }

    generated_keys.resize(size);
    std::copy(generated_keys_set.begin(), generated_keys_set.end(), generated_keys.begin());

#if FLIX_USE_KEY_CACHE
    fs::create_directories("data_cache");
    std::ofstream outfile(filename, std::ios::binary);
    if (outfile)
    {
        outfile.write(reinterpret_cast<const char *>(generated_keys.data()), sizeof(key_type) * size);
    }
#endif
}
//--------------------------

template <typename key_type>
std::vector<key_type> draw_uniform_live_deletes(
    size_t size,
    std::vector<key_type> &live_keys,
    std::mt19937_64 &gen)
{
    if (size > live_keys.size())
        throw std::out_of_range("delete batch exceeds live set");

    std::vector<key_type> delete_keys;
    delete_keys.reserve(size);

    for (size_t i = 0; i < size; ++i)
    {
        std::uniform_int_distribution<size_t> index_dist(
            0, live_keys.size() - 1);
        const size_t index = index_dist(gen);
        delete_keys.push_back(live_keys[index]);
        live_keys[index] = live_keys.back();
        live_keys.pop_back();
    }

    return delete_keys;
}

template <typename key_type>
void draw_live_probes(
    size_t size,
    const std::vector<key_type> &live_keys,
    const std::unordered_map<key_type, smallsize> &live_map,
    std::vector<key_type> &probe_keys,
    std::vector<smallsize> &expected_result)
{
    if (live_keys.empty())
        throw std::out_of_range("cannot probe an empty live set");

    std::mt19937_64 gen(42);
    std::uniform_int_distribution<size_t> index_dist(
        0, live_keys.size() - 1);
    probe_keys.resize(size);
    expected_result.resize(size);

    for (size_t i = 0; i < size; ++i)
    {
        const key_type key = live_keys[index_dist(gen)];
        const auto it = live_map.find(key);
        if (it == live_map.end())
            throw std::logic_error("live-key oracle is inconsistent");
        probe_keys[i] = key;
        expected_result[i] = it->second;
    }
}

template <typename key_type>
void draw_probes(
    size_t size,
    const std::vector<key_type> &build_keys,
    size_t active_range_start,
    size_t active_range_end,
    bool start_expected_offset_at_zero,
    std::vector<key_type> &probe_keys,
    std::vector<smallsize> &expected_result)
{
   // std::cerr << "TOP Drawing: " << size
   //         << " probes from active range ["
   //        << active_range_start << ", "
   //        << active_range_end << ")\n";

    std::mt19937_64 gen(42);
    std::uniform_int_distribution<size_t> index_dist(
        active_range_start, active_range_end - 1);

    probe_keys.resize(size);
    expected_result.resize(size);

    // std::cerr << " Next Drawing " << size
    //        << " probes from active range size ["
    //        << active_range_start << ", "
    //        << size << ")\n";

    for (size_t i = 0; i < size; ++i)
    {
        const size_t index = index_dist(gen);
        probe_keys[i] = build_keys[index];
        expected_result[i] = start_expected_offset_at_zero
                                 ? static_cast<smallsize>(index - active_range_start)
                                 : static_cast<smallsize>(index);
    }

    /* if (size > 0)
     {
         std::cerr << "Probes drawn. Sample probe key: "
                   << probe_keys[0]
                   << ", expected result: "
                   << expected_result[0]
                   << '\n';
     } */
}

bool is_all_misses(const std::vector<smallsize> &output, std::string &error_message)
{
    for (size_t i = 0; i < output.size(); ++i)
    {
        if (not_found != output[i])
        {
            error_message = "data mismatch at index " + std::to_string(i) + ": expected MISS, but received " + std::to_string(output[i]);
            return false;
        }
    }
    return true;
}

template <typename key_type>
void check_all_misses_results(const std::vector<smallsize> &output, const std::vector<key_type> &probe_keys)
{
    for (size_t i = 0; i < output.size(); ++i)
    {
        if (not_found != output[i])
        {
            std::cerr << "data mismatch at index " << i << ": expected MISS for key " << probe_keys[i] << ", but received " << output[i] << std::endl;
            throw std::logic_error("stop");
        }
    }
}

struct update_configuration
{
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
    rc::result_collector &rc,
    size_t runs)
{

    cudaSetDevice(0);
    cudaFree(0); // forces CUDA context init on this thread

    using key_type = typename index_type::key_type;

    size_t free_memory_bytes, total_memory_bytes;
    cudaMemGetInfo(&free_memory_bytes, &total_memory_bytes);

    // todo sync???
    constexpr bool supports_updates = index_type::can_update == operation_support::async;
    constexpr bool supports_successor = benchmark_supports_successor<index_type>::value;

    std::vector<update_configuration> test_configuration_options{
        //{"delete-insert", 26, 27, 1, 10, 10, false, false},
        //{"insert-delete", 26, 27, 1, 10, 10, false, true},
        //{"batches", 22, 23, 4, 150, 150, false, true},
        //{"batches", 22, 23, 2, 200, 200, false, true},
        //{"batches", 15, 16, 4, 125, 125, false, true},
        //{"batches", 16, 17, 8, 100, 100, false, true},
        // {"batches", 25, 26, 2, 100, 100, false, true},
        //{"batches", 25, 26, 8, 100, 100, false, true},
        {"batches", init_build_size_log, init_probe_size_log, rounds,
         UPDATE_INSERT_PERCENT, UPDATE_DELETE_PERCENT, false, true},
    };

    // *** MAKE TRUE TRUE

    for (size_t tci = 0; tci < test_configuration_options.size(); ++tci)
    {
        const auto &tc = test_configuration_options[tci];

        /* 
        // -------------- TEMPORARAY MANUAL OVERRIDE FOR TESTING: BUILD_SIZE and PROBE_SIZE
        // --> Adjust
        size_t build_size = ( size_t{1} << tc.build_size_log ); //------->  + 333; // + 100;
        //size_t build_size = 99999999 + 6; //100000005;
        std::cerr << " Build Size at top of File " << build_size << " free memory: " <<    free_memory_bytes << " total mem: " << total_memory_bytes << std::endl;
        // size_t build_size = (size_t{1} << tc.build_size_log); // + 100;
        // size_t build_size = (size_t{1} << tc.build_size_log) ; // + 100;
        // cudaSetDevice(0);
        // cudaFree(0); // forces CUDA context init on this thread
        //------------- size_t build_size = 99999999 + 3; //100000005;
        // tc.build_size_log = static_cast<size_t>(std::log2(build_size));
    
        size_t key_generation_size = build_size * (100 + tc.total_inserts_percentage_of_build_size) / 100;
        // --- 100M size_t probe_size = 99999999 +2; 
        size_t probe_size = (size_t{1} << tc.probe_size_log);
        
        // ----------------- END TEMPORARY OVERRIDE -----------------
        */


        #ifdef OVERRIDE_TO_100MILLION
        #pragma message "OVERRIDE_TO_100MILLION is ON: Using build_size=100M and probe_size =100M for all tests, ignoring config logs"

            const size_t NinetyNineMillion = 99999999;
            const size_t constant_above_99M = 6; // to ensure we are above 99M, which seems to be a critical point for some implementations 
            size_t build_size = NinetyNineMillion + constant_above_99M  ;   
            std::cerr << " --> Build Size at top of File " << build_size << " free memory: " <<    free_memory_bytes << " total mem: " << total_memory_bytes << std::endl;
            size_t probe_size = NinetyNineMillion +2; 

        #else
        #pragma message "OVERRIDE_TO_100MILLION is OFF: Using build_size_log and probe_size_log from config"

            size_t build_size = ( size_t{1} << tc.build_size_log ); 
            std::cerr << " --> Build Size defined at Top of File " << build_size << " Free memory: " <<    free_memory_bytes << " Total mem: " << total_memory_bytes << std::endl;
            size_t probe_size = (size_t{1} << tc.probe_size_log);


        #endif


        size_t insert_batch_size = build_size * tc.total_inserts_percentage_of_build_size / (100 * tc.batch_count);
        size_t delete_batch_size = build_size * tc.total_deletes_percentage_of_build_size / (100 * tc.batch_count);
#ifdef UPDATE_INSERT_BATCH_LOG
        insert_batch_size = fixed_insert_batch_size;
#endif
#ifdef UPDATE_DELETE_BATCH_LOG
        delete_batch_size = fixed_delete_batch_size;
#endif
        size_t key_generation_size = build_size + insert_batch_size * tc.batch_count;
        // size_t cache_line = tc.cache_line;
        // size_t node_size_log = size_t{1} << tc.node_size_log;

        rti_assert(key_generation_size * 2 < std::numeric_limits<smallsize>::max());

        std::cerr << "  update experiment " << tc.description << " for " << index_type::short_description() << std::endl;
        std::cerr << "  bits: " << sizeof(key_type) * 8 << std::endl;
        std::cerr << "  build_size_log: " << tc.build_size_log << std::endl;
        std::cerr << "  probe_size_log: " << tc.probe_size_log << std::endl;
        std::cerr << "  batch_count: " << tc.batch_count << std::endl;
        std::cerr << "  alternating_insert_delete: " << tc.alternating_insert_delete << std::endl;
        std::cerr << "  insert_before_delete: " << tc.insert_before_delete << std::endl;

        std::cerr << "  total_inserts_percentage_of_build_size: " << tc.total_inserts_percentage_of_build_size << std::endl;
        std::cerr << "  total_deletes_percentage_of_build_size: " << tc.total_deletes_percentage_of_build_size << std::endl;
        // std::cerr << " cache_line: " << tc.cache_line << std::endl;
        // std::cerr << " node_size_log: " << tc.node_size_log << std::endl;

        std::cerr << "  insert_batch_size: " << insert_batch_size << std::endl;
        std::cerr << "  delete_batch_size: " << delete_batch_size << std::endl;
        std::cerr << "  delete_selection: uniform_live" << std::endl;
        std::cerr << "  delete_seed: " << uniform_delete_seed << std::endl;
        std::cerr << "  parameters: " << std::endl;
        for (auto &[key, value] : index_type::parameters())
        {
            std::cerr << "    " << key << ": " << value << std::endl;
        }


#ifdef REGKEYGEN
#pragma message "Regular key generation"
        // -------------------- REGULAR GENERATED KEYS

        std::vector<key_type> generated_keys;
        generate_keys_file(
            key_generation_size, min_usable_key<key_type>(),
            flix_benchmark::common_max_key<index_type>(),
            generated_keys, "keys_cache.txt");
        // generate_keys(key_generation_size, min_usable_key<key_type>(), max_usable_key<key_type>(), generated_keys);

        std::cerr << "Regular generated key set of size " << key_generation_size << std::endl;

#elif defined(HYBRIDKEYGEN)
#pragma message "Skewed Hybrid Dist key generation"

        // -------------------- NEW GENERATED KEYS
        // Generate input keys
        std::vector<key_type> generated_keys;

        // Call generate_keys_hybrid
        generate_keys_hybrid_skip(
            key_generation_size, // Total number of keys
            build_size,          // Number of uniform keys
            insert_batch_size,   // Number of insert keys
            min_usable_key<key_type>(),
            flix_benchmark::common_max_key<index_type>(),
            generated_keys);

        // Print build_size
        std::cout << "FINISHED HYBRID GEN: Build Size: " << build_size << std::endl;

#elif defined(DENSEKEYGEN)
#pragma message "Hybrid Dense Key Generation"
        // -------------------- HYBRID DENSE KEYS

        // --------------------------------------------- Prior RRun SORT TEST
        // micro_benchmark_search_and_coalesced<uint32_t>();
        // Generate input keys
        std::vector<key_type> generated_keys;
        bool rotate_dense_key_range = false; // Set to true to generate dense keys
                                             // size_t percentage_distribution_dense_keys = 25; // defaults to 50%
                                             // size_t percentage_new_keys_from_dense_region = 90; // defaults to 90%
        generate_keys_hybrid_dense_file2_shifted<key_type>(
            key_generation_size, // Total number of keys
            build_size,          // 50% uniform keys
            insert_batch_size,   // Number of insert keys
            tc.batch_count,      // Number of batches
            min_usable_key<key_type>(),
            flix_benchmark::common_max_key<index_type>(),
            generated_keys,
            shift_insert_range,
            percentage_distribution_dense_keys, // Percentage of keys in the dense pattern (25% of the total key range)
            percentage_new_keys_from_dense_region);

        // Print build_size
        std::cout << "FINISHED HYBRID DENSE GEN: Build Size: " << build_size << std::endl;
        std::cout << "Shifted Dense Key Range: " << (rotate_dense_key_range ? "YES" : "NO") << std::endl;
        std::cout << "Percentage Distribution Dense Keys: " << percentage_distribution_dense_keys << "%" << std::endl;
        std::cout << "Percentage New Keys from Dense Region: " << percentage_new_keys_from_dense_region << "%" << std::endl;
        std::cout << "Total Generated Keys: " << generated_keys.size() << std::endl;
        std::cout << " input key_generation_size: " << key_generation_size << std::endl;
        std::cout << " input insert_batch_size:    " << insert_batch_size << std::endl;
        std::cout << " ------------------------------------------ " << std::endl;
#endif

        key_generation_size = generated_keys.size();
        flix_benchmark::input_checksum workload_keys_checksum;
        workload_keys_checksum.add(build_size);
        workload_keys_checksum.add(generated_keys);
        constexpr size_t storage_granularity =
            benchmark_update_storage_granularity<index_type>::value;
        static_assert(storage_granularity > 0, "Storage granularity must be positive");
        const auto padded_size = [](size_t size) {
            return size == 0 ? size_t{0} :
                (1 + (size - 1) / storage_granularity) * storage_granularity;
        };
        // Fixed-size LSM batches consume padding on every call.
        const size_t index_capacity_size = std::max(
            key_generation_size,
            padded_size(build_size) + tc.batch_count *
                (padded_size(insert_batch_size) +
                 (benchmark_index_stores_tombstones<index_type>::value
                      ? padded_size(delete_batch_size) : 0)));

#ifdef PRINT_GENERATED_KEYS
#pragma message "PRINT_GENERATED_HYBRID_KEYS=YES"
        // Print first `build_size` uniform keys
        std::cout << "Uniform Keys: " << std::endl;
        for (size_t i = 0; i < build_size; ++i)
        {
            std::cout << generated_keys[i] << " ";
            if (i % 10 == 9)
                std::cout << std::endl; // Print 10 keys per line
        }
        std::cout << "\n\n"; // Add gap

        // Print Dense keys
        std::cout << "REST KEYS:" << std::endl;
        for (size_t i = build_size; i < generated_keys.size(); ++i)
        {
            std::cout << generated_keys[i] << std::endl; // One key per line
        }

        std::cout << std::endl;
#endif

        // ---------------------------------------------
        std::cerr << "generated key set of size " << key_generation_size << std::endl;

        cuda_buffer<key_type> insert_delete_keys_buffer;

#ifdef COMBINE_INSERT_DELETE
        cuda_buffer<key_type> insert_keys_buffer;
        cuda_buffer<key_type> delete_keys_buffer;
        insert_keys_buffer.alloc(insert_batch_size);
        delete_keys_buffer.alloc(delete_batch_size);
#endif

        cuda_buffer<smallsize> insert_offsets_buffer;
        cuda_buffer<key_type> probe_keys_buffer;
        cuda_buffer<smallsize> result_buffer;
        cuda_buffer<smallsize> result_keys_buffer;
        // not enough memory for setup (with 10% margin)
        size_t setup_bytes = (key_generation_size * sizeof(key_type) + insert_batch_size * sizeof(smallsize) + probe_size * (sizeof(key_type) + sizeof(smallsize))) * 11 / 10;
        if (free_memory_bytes < setup_bytes)
        {
            std::cerr << " -> skipping, not enough memory for setup: " << setup_bytes << " bytes > " << free_memory_bytes << " bytes" << std::endl;
            continue;
        }
        size_t available_bytes_for_index = free_memory_bytes - setup_bytes;
        insert_delete_keys_buffer.alloc(key_generation_size);
        insert_offsets_buffer.alloc(insert_batch_size);
        probe_keys_buffer.alloc(probe_size);
        result_buffer.alloc(probe_size);
        result_keys_buffer.alloc(probe_size);

        smallsize next_free = 0;

        for (size_t run = 0; run < runs; ++run)
        {
            cuda_timer timer(0);

            std::cerr << " -> run " << run << std::endl;

            size_t inserted_range_end = build_size;

            // upload initial build keys
            insert_delete_keys_buffer.upload(generated_keys.data(), build_size);

            // initial build (untimed)
            index_type index;
            flix_benchmark::lookup_workspace<key_type> lookup_workspace;
            flix_benchmark::successor_workspace<key_type> successor_workspace;
            // index.build(insert_delete_keys_buffer.ptr(), active_range_end - active_range_start, nullptr, nullptr);
#ifdef DEBUG_BENCHMARK_OUTPUT
            next_free = index.allocation_buffer_next_free();
            smallsize total_alloc_nodes = index.allocation_buffer_total_nodes();
            std::cerr << "After Build  # of Assigned Free Nodes : " << next_free << " of total available nodes: " << total_alloc_nodes << std::endl;
#endif
            try
            {
                // todo final size does not have to be equal to key_generation_size, make sure to compute that properly
                // Note:  max size will not always be key_generation_size, it can be smaller if the index does not need all keys
                // for Baselines
#ifdef BASELINES
#pragma message "BASELINE BUILD"
                index.build(insert_delete_keys_buffer.ptr(), build_size,
                            index_capacity_size, available_bytes_for_index,
                            nullptr, nullptr);
                // index.build_static_tree(insert_delete_keys_buffer.ptr(), build_size, key_generation_size, available_bytes_for_index, nullptr, nullptr);
#else
#pragma message "REGULAR CGRXU BUILD-  BUCKET LAYER ONLY"
                index.build_bucket_layer_only(insert_delete_keys_buffer.ptr(), build_size, key_generation_size, available_bytes_for_index, nullptr, nullptr);
#endif
            }
            catch (const std::exception &e)
            {
                std::cerr << "   -> SKIP " << e.what() << std::endl;
                break;
            }

            // host oracle of the index state (key -> newest value) so checks expect newest-wins under duplicates
            std::unordered_map<key_type, smallsize> live_map;
            std::vector<key_type> live_keys;
            live_map.reserve(key_generation_size);
            live_keys.reserve(key_generation_size);
            for (size_t p = 0; p < build_size; ++p)
            {
                const key_type key = generated_keys[p];
                const auto [it, inserted] = live_map.emplace(
                    key, static_cast<smallsize>(p));
                if (inserted)
                    live_keys.push_back(key);
                else
                    it->second = static_cast<smallsize>(p);
            }
            std::mt19937_64 delete_gen(uniform_delete_seed + run);

#ifdef COMBINE_INSERT_DELETE
#pragma message "COMBINE_INSERT_DELETE=YES"
            // run updates
            for (size_t step = 0; step < tc.batch_count + 1; ++step)
            {
#else
#pragma message "COMBINE_INSERT_DELETE=NO"

            for (size_t step = 0; step < 2 * tc.batch_count + 1; ++step)
            // for (size_t step = 0; step < tc.batch_count + 1; ++step)

            {
#endif
                std::vector<key_type> last_deleted_keys;
                bool did_delete_this_step = false;
                flix_benchmark::lookup_times hit_times, miss_times, deleted_times;
                flix_benchmark::input_checksum request_checksum;
                request_checksum.add(step);
                size_t hit_count = 0, miss_count = 0, deleted_count = 0;

                double insert_or_delete_time_ms = 0;
                double probe_time_ms = 0;
                double successor_hits_probe_time_ms = 0;
                double successor_misses_probe_time_ms = 0;
                double successor_deleted_probe_time_ms = 0;

                double rebuild_time_ms = 0;
                double probe_miss_time_ms = 0;

                double index_layer_time_ms = 0;
                double bucket_layer_time_ms = 0;
                size_t before_update_bytes = 0;
                size_t after_update_bytes = 0;

                bool correct = true;
                std::string error_message;
                size_t totalkeys = 0;

                bool do_insert =
                    !tc.alternating_insert_delete && tc.insert_before_delete && (step < tc.batch_count + 1) ||
                    tc.alternating_insert_delete && tc.insert_before_delete && ((step & 1) == 1) ||
                    !tc.alternating_insert_delete && !tc.insert_before_delete && !(step < tc.batch_count + 1) ||
                    tc.alternating_insert_delete && !tc.insert_before_delete && !((step & 1) == 1);
                bool do_delete = !do_insert;

                const size_t new_inserted_range_end = do_insert
                    ? inserted_range_end + insert_batch_size
                    : inserted_range_end;
                size_t before_update_size = live_keys.size();
                size_t after_update_size = before_update_size;

                std::cerr << "----> Update step " << step
                          << ": introduced range [0, " << inserted_range_end
                          << "), live keys " << live_keys.size() << std::endl;

                // print key generartion size
                // std::cerr << "    key_generation_size: " << key_generation_size << std::endl;

                if (step == 0)
                {
                    std::cerr << "----> step 0: No Updates Probe only" << std::endl;
                    goto probe;
                }

                std::cerr << "----> step " << step << ": ";
                std::cerr << (do_insert ? "insert" : do_delete ? "delete"
                                                               : "nothing")
                          << std::endl;

                if (do_insert && new_inserted_range_end > generated_keys.size())
                {
                    std::cerr << "WARNING: insert batch exceeds generated key set\n";
                    goto end_run;
                }
                if (do_delete && delete_batch_size > live_keys.size())
                {
                    std::cerr << "WARNING: delete batch exceeds live key set\n";
                    goto end_run;
                }

                before_update_bytes = index.gpu_resident_bytes() +
                                      lookup_workspace.gpu_resident_bytes();

                if constexpr (supports_updates)
                {
#ifdef COMBINE_INSERT_DELETE
                    // insert the next batch
                    if (do_insert)
                    {
                        std::vector<key_type> relevant_inserts(
                            generated_keys.begin() + inserted_range_end,
                            generated_keys.begin() + inserted_range_end + insert_batch_size);
                        std::vector<smallsize> associated_offsets(insert_batch_size);
                        std::iota(associated_offsets.begin(), associated_offsets.end(), inserted_range_end);
                        request_checksum.add(1);
                        request_checksum.add(relevant_inserts);
                        request_checksum.add(associated_offsets);
                        insert_keys_buffer.upload(relevant_inserts.data(), insert_batch_size);
                        insert_offsets_buffer.upload(associated_offsets.data(), insert_batch_size);
                        C2EX
                                std::cerr
                            << "insert size: " << insert_batch_size << std::endl;
                        /*
                        timer.start();
                        index.insert(insert_delete_keys_buffer.ptr(), insert_offsets_buffer.ptr(), insert_batch_size, 0);
                        timer.stop();
                        insert_or_delete_time_ms = timer.time_ms();
                        cudaDeviceSynchronize();  CUERR
                         #ifdef DEBUG_BENCHMARK_OUTPUT
                            next_free= index.allocation_buffer_next_free();
                            //smallsize total_alloc_nodes= index.allocation_buffer_total_nodes();
                             std::cerr << "After Insert  #of Used New Nodes: " << next_free<<  " of total available nodes: " << total_alloc_nodes << std::endl;
                        #endif
                        */
                    }
                    // or delete the next batch
                    if (do_delete)
                    {
                        std::vector<key_type> relevant_deletes =
                            draw_uniform_live_deletes<key_type>(
                                delete_batch_size, live_keys, delete_gen);
                        request_checksum.add(2);
                        request_checksum.add(relevant_deletes);
                        delete_keys_buffer.upload(relevant_deletes.data(), delete_batch_size);
                        C2EX
                                std::cerr
                            << "delete size: " << delete_batch_size << std::endl;
                        /*
                        timer.start();
                        index.remove(insert_delete_keys_buffer.ptr(), delete_batch_size, 0);
                        timer.stop();
                        insert_or_delete_time_ms = timer.time_ms();
                        cudaDeviceSynchronize();  CUERR
                        #ifdef DEBUG_BENCHMARK_OUTPUT
                            next_free= index.allocation_buffer_next_free();
                            //smallsize total_alloc_nodes= index.allocation_buffer_total_nodes();
                             std::cerr << "After Delete  #of Used New Nodes: " << next_free<<  " of total available nodes: " << total_alloc_nodes << std::endl;
                        #endif
                        */
                    }
                    timer.start();
                    prepare_update_inputs<index_type>(insert_keys_buffer.ptr(),
                        insert_offsets_buffer.ptr(), insert_batch_size, 0);
                    prepare_update_inputs<index_type>(delete_keys_buffer.ptr(),
                        nullptr, delete_batch_size, 0);
                    index.insert_and_remove(insert_keys_buffer.ptr(), insert_offsets_buffer.ptr(), insert_batch_size, delete_keys_buffer.ptr(), 0);
                    timer.stop();
                    insert_or_delete_time_ms = timer.time_ms();
                    cudaDeviceSynchronize();
                    CUERR
#ifdef DEBUG_BENCHMARK_OUTPUT
                    next_free = index.allocation_buffer_next_free();
                    // smallsize total_alloc_nodes= index.allocation_buffer_total_nodes();
                    std::cerr << "After Combined  #of Used New Nodes: " << next_free << " of total available nodes: " << total_alloc_nodes << std::endl;
#endif

#else
                    // insert the next batch
                    if (do_insert)
                    {
                        const size_t ins_begin = inserted_range_end;
                        const size_t ins_end = ins_begin + insert_batch_size;

                        std::vector<key_type> relevant_inserts(
                            generated_keys.begin() + ins_begin,
                            generated_keys.begin() + ins_end);
                        std::vector<smallsize> associated_offsets(insert_batch_size);
                        std::iota(associated_offsets.begin(), associated_offsets.end(), ins_begin);
                        request_checksum.add(1);
                        request_checksum.add(relevant_inserts);
                        request_checksum.add(associated_offsets);
                        insert_delete_keys_buffer.upload(relevant_inserts.data(), insert_batch_size);
                        insert_offsets_buffer.upload(associated_offsets.data(), insert_batch_size);
                        C2EX
                                std::cerr
                            << "      insert size: " << insert_batch_size << std::endl;
                        timer.start();
                        // Charge required ordering to the update measurement.
                        prepare_update_inputs<index_type>(insert_delete_keys_buffer.ptr(),
                            insert_offsets_buffer.ptr(), insert_batch_size, 0);
                        index.insert(insert_delete_keys_buffer.ptr(), insert_offsets_buffer.ptr(), insert_batch_size, 0);
                        timer.stop();
                        insert_or_delete_time_ms = timer.time_ms();
                        cudaDeviceSynchronize();
                        CUERR

#ifdef DEBUG_BENCHMARK_OUTPUT
                        next_free = index.allocation_buffer_next_free();
                        // smallsize total_alloc_nodes= index.allocation_buffer_total_nodes();
                        std::cerr << "After Insert  #of Used New Nodes: " << next_free << " of total available nodes: " << total_alloc_nodes << std::endl;
#endif

                        for (size_t p = ins_begin; p < ins_end; ++p)
                        {
                            const key_type key = generated_keys[p];
                            const auto [it, inserted] = live_map.emplace(
                                key, static_cast<smallsize>(p));
                            if (inserted)
                                live_keys.push_back(key);
                            else
                                it->second = static_cast<smallsize>(p);
                        }
                    }

                    if (do_delete)
                    {
                        std::vector<key_type> relevant_deletes =
                            draw_uniform_live_deletes<key_type>(
                                delete_batch_size, live_keys, delete_gen);
                        request_checksum.add(2);
                        request_checksum.add(relevant_deletes);
                        insert_delete_keys_buffer.upload(relevant_deletes.data(), delete_batch_size);
                        C2EX
                                std::cerr
                            << "delete size: " << delete_batch_size << std::endl;
                        timer.start();
                        prepare_update_inputs<index_type>(insert_delete_keys_buffer.ptr(),
                            nullptr, delete_batch_size, 0);
                        index.remove(insert_delete_keys_buffer.ptr(), delete_batch_size, 0);
                        timer.stop();
                        insert_or_delete_time_ms = timer.time_ms();
                        cudaDeviceSynchronize();
                        CUERR
#ifdef DEBUG_BENCHMARK_OUTPUT
                        next_free = index.allocation_buffer_next_free();
                        // smallsize total_alloc_nodes= index.allocation_buffer_total_nodes();
                        std::cerr << "After Delete  #of Used New Nodes: " << next_free << " of total available nodes: " << total_alloc_nodes << std::endl;
#endif
                        for (const key_type key : relevant_deletes)
                            live_map.erase(key);

                        last_deleted_keys = std::move(relevant_deletes);
                        did_delete_this_step = true;
                    }

#endif
                }

                if constexpr (supports_updates)
                {
                    if (did_delete_this_step && !last_deleted_keys.empty())
                    {
                        const smallsize del_probe_size = static_cast<smallsize>(last_deleted_keys.size());

                        cuda_buffer<key_type> del_probe_keys_buffer;
                        cuda_buffer<smallsize> del_result_buffer;
                        del_probe_keys_buffer.alloc(del_probe_size);
                        del_result_buffer.alloc(del_probe_size);
                        del_probe_keys_buffer.upload(last_deleted_keys.data(), del_probe_size);

                        request_checksum.add(3);
                        request_checksum.add(last_deleted_keys);
                        deleted_count = del_probe_size;
                        deleted_times = lookup_workspace.lookup(index,
                            del_probe_keys_buffer.ptr(), del_result_buffer.ptr(),
                            del_probe_size);
                        auto del_probe_result = del_result_buffer.download(del_probe_size);

                            // Validate: every just-deleted key MUST be a miss
                            check_all_misses_results<key_type>(del_probe_result, last_deleted_keys);
                        correct = is_all_misses(del_probe_result, error_message);
                        if (!correct)
                        {
                            std::cerr << " -> SKIP post-delete miss-probe failed: " << error_message << std::endl;
                            break;
                        }

                        if constexpr (supports_successor && PERFORM_SUCCESSOR_PROBES)
                        {
                            cuda_buffer<key_type> successor_results;
                            successor_results.alloc(del_probe_size);
                            const auto successor_times = successor_workspace.lookup(
                                index, del_probe_keys_buffer.ptr(),
                                successor_results.ptr(), del_probe_size);
                            successor_deleted_probe_time_ms =
                                successor_times.total_ms;
                        }

                    }
                }

                if constexpr (!supports_updates)
                {
                    if (do_insert)
                    {
                        for (size_t p = inserted_range_end;
                             p < new_inserted_range_end; ++p)
                        {
                            const key_type key = generated_keys[p];
                            const auto [it, inserted] = live_map.emplace(
                                key, static_cast<smallsize>(p));
                            if (inserted)
                                live_keys.push_back(key);
                            else
                                it->second = static_cast<smallsize>(p);
                        }
                    }
                    else
                    {
                        const std::vector<key_type> relevant_deletes =
                            draw_uniform_live_deletes<key_type>(
                                delete_batch_size, live_keys, delete_gen);
                        for (const key_type key : relevant_deletes)
                            live_map.erase(key);
                    }

                    index.destroy();
                    const size_t rebuild_size = live_keys.size();
                    insert_delete_keys_buffer.upload(
                        live_keys.data(), rebuild_size);
                    try
                    {
                        index.build(insert_delete_keys_buffer.ptr(), rebuild_size, rebuild_size, available_bytes_for_index, &insert_or_delete_time_ms, nullptr);
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << " -> SKIP " << e.what() << std::endl;
                        break;
                    }
                    for (size_t p = 0; p < rebuild_size; ++p)
                        live_map[live_keys[p]] = static_cast<smallsize>(p);
                }

                inserted_range_end = new_inserted_range_end;
                after_update_size = live_keys.size();

                std::cerr << "     -> " << insert_or_delete_time_ms << " ms" << std::endl;

                //***************************** REBUILD AFTER INSERTIONS OR DELETIONS */

               // #define REBUILD_ON
#ifdef REBUILD_ON
#pragma message "REBUILD_ON=YES"

                // rebuild
                rebuild_time_ms = 0; // REBUILD FOR DELETES ONLY
                // ----if ( (step % rebuild_frequency == 0) && do_delete)
                if ( (step % rebuild_frequency == 0) )
                {
                    next_free = index.allocation_buffer_next_free();
                    timer.start();
                    index.rebuild_bucket_layer_only(nullptr, nullptr);
                    timer.stop();
                    rebuild_time_ms = timer.time_ms();
                    // next_free = index.allocation_buffer_next_free();
                    std::cerr << "Rebuild ON:  # Used new nodes prior to Rebuild: " << next_free << " Rebuild time: " << rebuild_time_ms << std::endl;
                }
#endif
#ifdef DEBUG_BENCHMARK_OUTPUT
                next_free = index.allocation_buffer_next_free();
                total_alloc_nodes = index.allocation_buffer_total_nodes();
#ifdef REBUILD_ON
                std::cerr << "After- Rebuild ON  #of Used New Nodes: " << next_free << " of total available nodes: " << total_alloc_nodes << " Rebuild time: " << rebuild_time_ms << std::endl;
#else
                std::cerr << "After- NO Rebuild  #of Used New Nodes: " << next_free << " of total available nodes: " << total_alloc_nodes << std::endl;
#endif
#endif
                //*************************************************************************** */
            probe:;
                if (!live_keys.empty())
                {
                    std::vector<key_type> probe_keys;
                    std::vector<smallsize> expected_result;
                    draw_live_probes(probe_size, live_keys, live_map,
                                     probe_keys, expected_result);
                    request_checksum.add(4);
                    request_checksum.add(probe_keys);
                    request_checksum.add(expected_result);
                    hit_count = probe_keys.size();
                    probe_keys_buffer.upload(probe_keys.data(), hit_count);
                    hit_times = lookup_workspace.lookup(index,
                        probe_keys_buffer.ptr(), result_buffer.ptr(), hit_count);
                    probe_time_ms = hit_times.total_ms;
                    correct = is_matching_result(probe_keys, probe_keys,
                        expected_result, result_buffer, error_message);
                    if (!correct)
                    {
                        std::cerr << " -> SKIP " << error_message << std::endl;
                        break;
                    }

                    if constexpr (supports_successor && PERFORM_SUCCESSOR_PROBES)
                    {
                        cuda_buffer<key_type> successor_results;
                        successor_results.alloc(hit_count);
                        const auto successor_times = successor_workspace.lookup(
                            index, probe_keys_buffer.ptr(),
                            successor_results.ptr(), hit_count);
                        successor_hits_probe_time_ms =
                            successor_times.total_ms;
                    }
                }

                // Future insertions provide the same miss queries for every backend.
                if (inserted_range_end < key_generation_size)
                {
                    std::vector<key_type> probe_keys_misses;
                    std::vector<smallsize> expected_result_misses;
                    draw_probes(probe_size, generated_keys, inserted_range_end,
                        key_generation_size, supports_updates,
                        probe_keys_misses, expected_result_misses);
                    probe_keys_misses.erase(std::remove_if(probe_keys_misses.begin(),
                        probe_keys_misses.end(), [&](key_type key) {
                            return live_map.find(key) != live_map.end();
                        }), probe_keys_misses.end());
                    miss_count = probe_keys_misses.size();
                    request_checksum.add(5);
                    request_checksum.add(probe_keys_misses);
                    if (miss_count > 0)
                    {
                        probe_keys_buffer.upload(probe_keys_misses.data(), miss_count);
                        miss_times = lookup_workspace.lookup(index,
                            probe_keys_buffer.ptr(), result_buffer.ptr(), miss_count);
                        probe_miss_time_ms = miss_times.total_ms;
                        auto results = result_buffer.download(miss_count);
                        correct = is_all_misses(results, error_message);
                        if (!correct)
                        {
                            std::cerr << " -> SKIP " << error_message << std::endl;
                            break;
                        }

                        if constexpr (supports_successor && PERFORM_SUCCESSOR_PROBES)
                        {
                            cuda_buffer<key_type> successor_results;
                            successor_results.alloc(miss_count);
                            const auto successor_times = successor_workspace.lookup(
                                index, probe_keys_buffer.ptr(),
                                successor_results.ptr(), miss_count);
                            successor_misses_probe_time_ms =
                                successor_times.total_ms;
                        }
                    }
                }

                // PRINT PROBE TIME HERE
                std::cerr << "    --Probe Time: -> " << probe_time_ms << " ms" << std::endl;
                std::cerr << "    --Probe Miss Time: -> " << probe_miss_time_ms << " ms" << std::endl;
                std::cerr << "    ---------------------- " << std::endl;

                after_update_bytes = index.gpu_resident_bytes() +
                                     lookup_workspace.gpu_resident_bytes();

#ifdef COMPUTE_TOTALKEYS
#pragma message "COMPUTE_TOTALKEYS=YES"
                totalkeys = index.compute_total_size(0);
                std::cerr << "         Total Keys in DS: " << totalkeys << std::endl;

#else
#pragma message "COMPUTE_TOTALKEYS=NO"
                // std::cerr << "Total Keys in DS: " << after_update_size <<  std::endl;
#endif

                {
                    auto index_id = index_type::short_description() + "__" + parameters_to_name(index_type::parameters());
                    rc::auto_commit_result(rc)
                        .add_parameter("EXPERIMENT", tc.description)
                        .add_parameter("index_type", index_type::short_description())
                        .add_parameter("index_parameters", parameters_to_string(index_type::parameters()))
                        .add_parameter("index_id", index_id)
                        .add_parameter("run", run)
                        .add_parameter("key_bits", sizeof(key_type) * 8)
                        .add_parameter("workload_key_bits", FLIX_BENCHMARK_KEY_BITS)
                        .add_parameter("workload_min_key", min_usable_key<key_type>())
                        .add_parameter("workload_max_key", flix_benchmark::common_max_key<index_type>())
                        .add_parameter("workload_keys_checksum", workload_keys_checksum.str())
                        .add_parameter("request_checksum", request_checksum.str())
                        .add_parameter("checksum_format", "fnv1a64_u64le_v1")
                        .add_parameter("lookup_timing", "complete_unsorted_v1")
                        .add_parameter("successor_timing", "complete_unsorted_v1")
                        .add_parameter("probe_input_order", "unsorted")
                        .add_parameter("external_lookup_sort", flix_benchmark::requires_ordered_lookup<index_type>::value)
                        .add_parameter("hit_query_count", hit_count)
                        .add_parameter("miss_query_count", miss_count)
                        .add_parameter("deleted_query_count", deleted_count)
                        .add_parameter("build_size_log", tc.build_size_log)
                        .add_parameter("probe_size_log", tc.probe_size_log)
                        .add_parameter("batch_count", tc.batch_count)
                        .add_parameter("total_inserts_percentage_of_build_size", tc.total_inserts_percentage_of_build_size)
                        .add_parameter("total_deletes_percentage_of_build_size", tc.total_deletes_percentage_of_build_size)
                        .add_parameter("delete_selection", "uniform_live")
                        .add_parameter("delete_seed", uniform_delete_seed + run)
                        // -----> .add_parameter("cache_line_size", tc.cache_line)
                        // -----> .add_parameter("node_size_log", tc.node_size_log)
                        .add_parameter("alternating_insert_delete", tc.alternating_insert_delete)
                        .add_parameter("insert_before_delete", tc.insert_before_delete)
                        .add_parameter("step", step)
                        .add_parameter("do_insert", do_insert)
                        .add_parameter("do_delete", do_delete)
                        .add_parameter("current_batch_size", do_insert ? insert_batch_size : do_delete ? delete_batch_size
                                                                                                       : 0)
                        .add_parameter("before_update_size", before_update_size)
                        .add_parameter("after_update_size", after_update_size)
                        .add_measurement("insert_or_delete_time_ms", insert_or_delete_time_ms)
                        .add_measurement("probe_time_ms", probe_time_ms)
                        .add_measurement("probe_wall_time_ms", hit_times.wall_ms)
                        .add_measurement("probe_prepare_time_ms", hit_times.prepare_ms)
                        .add_measurement("probe_search_time_ms", hit_times.search_ms)
                        .add_measurement("probe_restore_time_ms", hit_times.restore_ms)
                        .add_measurement("probe_miss_time_ms", probe_miss_time_ms)
                        .add_measurement("probe_miss_wall_time_ms", miss_times.wall_ms)
                        .add_measurement("probe_miss_prepare_time_ms", miss_times.prepare_ms)
                        .add_measurement("probe_miss_search_time_ms", miss_times.search_ms)
                        .add_measurement("probe_miss_restore_time_ms", miss_times.restore_ms)
                        .add_measurement("deleted_keys_probe_time_ms", deleted_times.total_ms)
                        .add_measurement("deleted_keys_probe_wall_time_ms", deleted_times.wall_ms)
                        .add_measurement("deleted_keys_probe_prepare_time_ms", deleted_times.prepare_ms)
                        .add_measurement("deleted_keys_probe_search_time_ms", deleted_times.search_ms)
                        .add_measurement("deleted_keys_probe_restore_time_ms", deleted_times.restore_ms)
                        .add_measurement("successor_hits_probe_time_ms", successor_hits_probe_time_ms)
                        .add_measurement("successor_misses_probe_time_ms", successor_misses_probe_time_ms)
                        .add_measurement("successor_deleted_probe_time_ms", successor_deleted_probe_time_ms)
                        .add_measurement("index_layer_time_ms", index_layer_time_ms)
                        .add_measurement("bucket_layer_time_ms", bucket_layer_time_ms)
                        .add_measurement("rebuild_time_ms", rebuild_time_ms)
                        .add_measurement("before_update_bytes", before_update_bytes)
                        .add_measurement("after_update_bytes", after_update_bytes);
                }
            end_run:;
            }
        }
    }

    // also benchmark range sums for indexes that support it
    if constexpr (index_type::can_range_lookup == operation_support::async)
    {
        benchmark_range_query<index_type>(rc, runs);
    }
}


#endif
