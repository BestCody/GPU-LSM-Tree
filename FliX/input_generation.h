// =============================================================================
// File: input_generation.h
// Author: Justus Henneberg
// Description: Defines expressions and constants used in FliX     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INPUT_GENERATION_H
#define INPUT_GENERATION_H

#include <thrust/sort.h>

#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <string_view>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>
#include <fstream>
#include <filesystem>  // To check file existence
#include "definitions.cuh"
#include "definitions_updates.cuh"
#include "debug_definitions_updates.cuh"

#define PRINT_BATCHES_GENERATED_KEYS
//NOTE : REMAINING--- add in shifting window to Dense Keys Generation Function

template <typename key_type>
void generate_keys_hybrid_dense_file2(
    size_t size,
    size_t build_size,
    size_t insert_list_size,
    size_t num_batches,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    size_t percentage_distribution_dense_keys = 25,
    size_t percentage_new_keys_from_dense_region =75, // defaults to 75
    const std::string& filename = "keys_cache.txt"
   
) {

    size_t x = percentage_distribution_dense_keys;
    size_t y = percentage_new_keys_from_dense_region;

     //printf("Calculated Range for x: %u Percent of distribution Length of Range per batch: %u, MINkey %u and MAXkey %u \n", x, range, min_usable_key, max_usable_key);
    printf("Non Shifting Dense Range (Same Dense Range Always) X: %u Percent of distribution receives Y: %u of the insert keys per batch \n", x, y);
   

    // ---------------------------CHECK FOR EXISTING CACHE---------------------------
    namespace fs = std::filesystem;
    std::string full_path = "data_cache/" + filename;

    if (fs::exists(full_path)) {
        std::ifstream infile(full_path);
        if (infile) {
            generated_keys.clear();
            key_type key;
            while (generated_keys.size() < size && infile >> key) {
                generated_keys.push_back(key);
            }
            if (generated_keys.size() == size) {
                std::cerr << "Loaded keys from cache: " << full_path << std::endl;
                return;
            } else {
                std::cerr << "Cache incomplete, regenerating keys..." << std::endl;
            }
        }
    }

    //--------------------- Step 1: Generate Uniform Keys --------------------
    std::mt19937_64 gen(42);
    std::unordered_set<key_type> unique_keys;

    std::cerr << "Generating Hybrid-Dense keys with " << build_size << " uniform keys and "
              << (size - build_size) << " dense keys..." << std::endl;

    std::uniform_int_distribution<key_type> uniform_dist(min_usable_key, max_usable_key);
    std::vector<key_type> uniform_vector;

    while (uniform_vector.size() < build_size) {
        key_type key = uniform_dist(gen);
        if (unique_keys.insert(key).second) {
            uniform_vector.push_back(key);
        }
    }
     //--------------------- Step 2: Generate Dense Keys --------------------
    std::cerr << "Generating Dense-Pattern + Uniform keys per batch..." << std::endl;
       std::vector<key_type> dense_pattern_vector;

   // key_type range = static_cast<key_type>((max_usable_key - min_usable_key) * (x / 100.0));

    key_type range = SDIV((max_usable_key - min_usable_key) * x, 100);
   
    key_type lower_bound_dense_region = min_usable_key;
    key_type top_range_dense_region = min_usable_key + range;

     printf("Calculated Range for x: %u Percent of distribution Length of Range per batch: %u, Minkey %u and Maxkey %u \n", x, range, min_usable_key, max_usable_key);
    printf("x: %u Percent of distributio receives y: %u of the insert keys per batch \n", x, y);
   

    std::uniform_int_distribution<key_type> dense_dist(lower_bound_dense_region, top_range_dense_region);

    for (size_t batch = 0; batch < num_batches && dense_pattern_vector.size() < size; ++batch) {
        std::vector<key_type> batch_vector;
        size_t dense_keys_in_batch = insert_list_size * y / 100;
        size_t uniform_keys_in_batch = insert_list_size - dense_keys_in_batch;

        size_t dense_count = 0, uniform_count = 0;

        while (dense_keys_in_batch > 0 && dense_pattern_vector.size() < size) {
            key_type key = dense_dist(gen);
            if (unique_keys.insert(key).second) {
                batch_vector.push_back(key);
                --dense_keys_in_batch;
                ++dense_count;
            }
        }

        while (uniform_keys_in_batch > 0 && dense_pattern_vector.size() < size) {
            key_type key = uniform_dist(gen);
            if (key >= lower_bound_dense_region && key <= top_range_dense_region) {
                continue;
            }
            if (unique_keys.insert(key).second) {
                batch_vector.push_back(key);
                --uniform_keys_in_batch;
                ++uniform_count;
            }
        }

        dense_pattern_vector.insert(
            dense_pattern_vector.end(),
            batch_vector.begin(),
            batch_vector.end()
        );


    #ifdef PRINT_BATCHES_GENERATED_KEYS
        // Debug prints
        std::cerr << "*************PRINT BATCH HERE **********" << std::endl;
        std::cerr << "Batch " << batch + 1 << ": Dense Region ["
                  << lower_bound_dense_region << ", " << top_range_dense_region << "]" << std::endl;

        #ifdef PRINT_GENERATED_KEYS   
        std::cerr << "Dense set: ";
        for (size_t i = 0; i < dense_count; ++i) {
            std::cerr << batch_vector[i] << " ";
        }
        std::cerr << std::endl;

        std::cerr << "Uniform set: ";
        for (size_t i = dense_count; i < batch_vector.size(); ++i) {
            std::cerr << batch_vector[i] << " ";
        }
        std::cerr << std::endl;
        #endif
    #endif
    }

    //--------------------- Step 3: MERGE --------------------
    generated_keys.clear();
    generated_keys.reserve(size);
    generated_keys.insert(generated_keys.end(), uniform_vector.begin(), uniform_vector.end());
    generated_keys.insert(generated_keys.end(), dense_pattern_vector.begin(), dense_pattern_vector.end());

    fs::create_directories("data_cache");
    std::ofstream outfile(full_path, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        std::cerr << "Saved generated keys to cache: " << full_path << std::endl;
    } else {
        std::cerr << "Failed to save keys to file: " << full_path << std::endl;
    }
}



//%%------------------------------------------------------------------ DENSE KEYS GENERATION FUNCTION ------------------------------------------------------------------%%

// Modified Dense Key Generation with Shifting Dense Region
// File: hybrid_dense_key_generator_shifted.cpp

template <typename key_type>
void generate_keys_hybrid_dense_file2_shifted(
    size_t size,
    size_t build_size,
    size_t insert_list_size,
    size_t num_batches,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    bool shift_dense_keys = false,
    size_t percentage_distribution_dense_keys = 25,
    size_t percentage_new_keys_from_dense_region = 75,
    const std::string& filename = "keys_cache_shifted.txt"
) {
    size_t x = percentage_distribution_dense_keys;
    size_t y = percentage_new_keys_from_dense_region;

    if (!shift_dense_keys) {
        generate_keys_hybrid_dense_file2<key_type>(size, build_size, insert_list_size , num_batches , min_usable_key, max_usable_key, generated_keys, x, y);
        return;
    }

    namespace fs = std::filesystem;
    std::string full_path = "data_cache/" + filename;

    if (fs::exists(full_path)) {
        std::ifstream infile(full_path);
        if (infile) {
            generated_keys.clear();
            key_type key;
            while (generated_keys.size() < size && infile >> key) {
                generated_keys.push_back(key);
            }
            if (generated_keys.size() == size) {
                std::cerr << "Loaded keys from cache: " << full_path << std::endl;
                return;
            } else {
                std::cerr << "Cache incomplete, regenerating keys..." << std::endl;
            }
        }
    }

     //--------------------- Step 1: Uniform Keys for BUILD --------------------
    std::mt19937_64 gen(42);
    std::unordered_set<key_type> unique_keys;

    std::cerr << "Generating Hybrid-Dense keys with shifting dense region..." << std::endl;

    std::uniform_int_distribution<key_type> uniform_dist(min_usable_key, max_usable_key);
    std::vector<key_type> uniform_vector;

    while (uniform_vector.size() < build_size) {
        key_type key = uniform_dist(gen);
        if (unique_keys.insert(key).second) {
            uniform_vector.push_back(key);
        }
    }

     //--------------------- Step 2: Generate Dense Keys SHIFTING --------------------

     std::cerr << "Generating Dense Region with Shifting throughout range..." << std::endl;

std::vector<key_type> dense_pattern_vector;

    //key_type range = static_cast<key_type>((max_usable_key - min_usable_key) * (x / 100.0));
    key_type range = SDIV((max_usable_key - min_usable_key) * x, 100) -1 ;

    printf("Calculated Range for x: %u Percent of distribution Length of Range per batch: %u, MINkey %u and MAXkey %u \n", x, range, min_usable_key, max_usable_key);
    printf("x: %u Percent of distributio receives y: %u of the insert keys per batch \n", x, y);
   
   
    key_type lower_bound_dense_region = min_usable_key;
    key_type top_range_dense_region = min_usable_key + range;

    
    for (size_t batch = 0; batch < num_batches && dense_pattern_vector.size() < size; ++batch) {

        std::uniform_int_distribution<key_type> dense_dist(lower_bound_dense_region, top_range_dense_region);
        std::vector<key_type> dense_keys;
        std::vector<key_type> uniform_keys;

        //size_t dense_keys_in_batch = insert_list_size * y / 100;
        size_t dense_keys_in_batch = SDIV(insert_list_size * y, 100);
        size_t uniform_keys_in_batch = insert_list_size - dense_keys_in_batch;


        while (dense_keys_in_batch > 0 && dense_pattern_vector.size() < size) {
            key_type key = dense_dist(gen);
            if (unique_keys.insert(key).second) {
                dense_keys.push_back(key);
                --dense_keys_in_batch;
            }
        }

        while (uniform_keys_in_batch > 0 && dense_pattern_vector.size() < size) {
            key_type key = uniform_dist(gen);
            if (key >= lower_bound_dense_region && key <= top_range_dense_region) {
                continue;
            }
            if (unique_keys.insert(key).second) {
                uniform_keys.push_back(key);
                --uniform_keys_in_batch;
            }
        }

        dense_pattern_vector.insert(
            dense_pattern_vector.end(),
            dense_keys.begin(),
            dense_keys.end()
        );
        dense_pattern_vector.insert(
            dense_pattern_vector.end(),
            uniform_keys.begin(),
            uniform_keys.end()
        );

        #ifdef PRINT_GENERATED_KEYS
        // Debug prints
        std::cerr << "***********************" << std::endl;
        std::cerr << "Batch " << batch + 1 << ": Dense Region ["
                  << lower_bound_dense_region << ", " << top_range_dense_region << "]" << std::endl;

        std::cerr << "Dense set: ";
        for (const auto& key : dense_keys) {
            std::cerr << key << " ";
        }
        std::cerr << std::endl;

        std::cerr << "Uniform set: ";
        for (const auto& key : uniform_keys) {
            std::cerr << key << " ";
        }
        std::cerr << std::endl;
        #endif

        // Shift dense region
        lower_bound_dense_region = top_range_dense_region + 1;
        top_range_dense_region += range;
        
        //reset
        if (top_range_dense_region >= (max_usable_key) || lower_bound_dense_region >= (max_usable_key-2)) {
            lower_bound_dense_region = min_usable_key;
            top_range_dense_region = min_usable_key + range;
        }
    }



    //--------------------- Step 3: MERGE --------------------
    generated_keys.clear();
    generated_keys.clear();
    generated_keys.reserve(size);
    generated_keys.insert(generated_keys.end(), uniform_vector.begin(), uniform_vector.end());
    generated_keys.insert(generated_keys.end(), dense_pattern_vector.begin(), dense_pattern_vector.end());

    fs::create_directories("data_cache");
    std::ofstream outfile(full_path, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        std::cerr << "Saved generated keys to cache: " << full_path << std::endl;
    } else {
        std::cerr << "Failed to save keys to file: " << full_path << std::endl;
    }
}

//%%------------------------------------------------------------------ DENSE KEYS GENERATION ------------------------------------------------------------------%%

class zipf_index_distribution_raw final {
    std::vector<double> cdf;
    double normalization;
    std::uniform_real_distribution<double> dis;

public:
    zipf_index_distribution_raw(size_t size, double exp) : dis(0.0, 1.0) {
        double sum = 0;
        for (size_t i = 0; i < size; ++i) {
            sum += 1.0 / std::pow(i + 1, exp);
        }
        normalization = 1.0 / sum;

        cdf.resize(size);
        double cumsum = 0;
        for (size_t i = 0; i < size; ++i) {
            cdf[i] = cumsum = cumsum + normalization / std::pow(i + 1, exp);
        }
    }

    template <typename gen_type>
    size_t operator()(gen_type& gen) {
        double draw = dis(gen);

        size_t offset = 0;
        for (size_t skip = size_t(1) << 63u; skip > 0; skip >>= 1) {
            if (offset + skip >= cdf.size())
                continue;
            if (draw <= cdf[offset + skip])
                continue;
            offset += skip;
        }
        return offset;
    }
};


class zipf_index_distribution final {
    std::optional<zipf_index_distribution_raw> zipf;
    std::optional<std::uniform_int_distribution<size_t>> uniform;

public:
    zipf_index_distribution(size_t size, double exp) {
        if (exp == 0) {
            uniform.emplace(0, size - 1);
        } else {
            zipf.emplace(size, exp);
        }
    }

    template <typename gen_type>
    size_t operator()(gen_type& gen) {
        if (zipf) {
            return (*zipf)(gen);
        } else {
            return (*uniform)(gen);
        }
    }
};


template <typename elem_type>
void sort_vector(std::vector<elem_type>& vec) {
    thrust::sort(thrust::host, vec.begin(), vec.end());
    //std::sort(vec.begin(), vec.end());
}


template <typename elem_type, typename random_type>
void shuffle_vector(std::vector<elem_type>& vec, random_type& gen) {
    std::shuffle(vec.begin(), vec.end(), gen);
}


std::vector<std::size_t> identity_permutation(size_t size) {
    std::vector<std::size_t> p(size);
    std::iota(p.begin(), p.end(), 0);
    return p;
}


template <typename random_type>
std::vector<size_t> shuffle_permutation(size_t size, random_type& gen) {
    std::vector<std::size_t> p(size);
    std::iota(p.begin(), p.end(), 0);
    shuffle_vector(p, gen);
    return p;
}


template <typename key_type, typename compare_type = std::less<key_type>>
std::vector<std::size_t> sort_permutation(const std::vector<key_type>& vec, compare_type compare = {}) {
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    thrust::sort(thrust::host, p.begin(), p.end(), [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
    //std::sort(p.begin(), p.end(), [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
    return p;
}


std::vector<std::size_t> invert_permutation(const std::vector<std::size_t>& permutation) {
    std::vector<std::size_t> inverse(permutation.size());
    for (std::size_t i = 0; i < permutation.size(); ++i) {
        inverse[permutation[i]] = i;
    }
    return inverse;
}


template <typename key_type>
void apply_permutation(std::vector<key_type>& vec, const std::vector<std::size_t>& permutation) {
    std::vector<key_type> sorted_vec(vec.size());
    thrust::transform(thrust::host, permutation.begin(), permutation.end(), sorted_vec.begin(), [&](std::size_t i){ return vec[i]; });
    //std::transform(permutation.begin(), permutation.end(), sorted_vec.begin(), [&](std::size_t i){ return vec[i]; });
    std::swap(vec, sorted_vec);
}


// load/store binary dumps of POD vectors
constexpr std::size_t binary_dump_magic_number() noexcept {
    return 0xAAAAAAAA55555555;
}


template<class T>
void dump_binary(const std::string& filename, const std::vector<T>& data) noexcept {
    std::ofstream ofile(filename, std::ios::binary);

    if (ofile.good()) {
        const size_t magic_number = binary_dump_magic_number();
        const size_t t_bytes = sizeof(T);
        const size_t size = data.size();

        ofile.write((char *) &magic_number, sizeof(size_t));
        ofile.write((char *) &t_bytes, sizeof(size_t));
        ofile.write((char *) &size, sizeof(size_t));

        ofile.write((char *) data.data(), sizeof(T) * size);
    } else {
        std::cerr << "unable to dump binary file" << std::endl;
    }
}


template<class T>
bool load_binary(const std::string& filename, std::vector<T>& data) noexcept {
    std::ifstream ifile(filename, std::ios::binary);

    if (!ifile.is_open()) {
        std::cerr << "unable to open dump file" << std::endl;
        return false;
    }

    std::size_t magic_number;
    ifile.read((char *) &magic_number, sizeof(std::size_t));

    if (magic_number != binary_dump_magic_number()) {
        std::cerr << "invalid dump file format" << std::endl;
        return false;
    }

    std::size_t t_bytes;
    ifile.read((char* ) &t_bytes, sizeof(std::size_t));

    if (t_bytes != sizeof(T)) {
        std::cerr << "dump file data type mismatch" << std::endl;
        return false;
    }

    std::size_t size;
    ifile.read((char* ) &size, sizeof(std::size_t));
    data.resize(size);
    ifile.read((char *) data.data(), sizeof(T) * size);

    return true;
}


template <typename... types>
std::string make_identifier(const types&... args) {
    std::stringstream ss;
    std::size_t n{0};
    ((ss << (++n != 1 ? "-" : "") << args), ...);
    return ss.str();
}


template <typename key_type, typename value_type>
struct point_query_dataset {

    constexpr static size_t num_reserved_keys = 10'000'000;

    std::string dataset_identifier;
    std::vector<key_type> build_keys;
    std::vector<value_type> build_values;
    std::vector<key_type> probe_keys;
    std::vector<value_type> expected_result;

    point_query_dataset(
        size_t seed,
        const char* cache_directory,
        size_t num_build_keys,
        size_t num_probe_keys,
        size_t key_stride,
        size_t first_key_offset,
        bool sort_insert,
        bool sort_probe_results,
        size_t build_key_uniformity_percent,
        double probe_zipf_coefficient,
        size_t misses_percent,
        size_t outlier_misses_percent,
        size_t key_multiplicity,
        bool with_values,
        key_type min_usable_key,
        key_type max_usable_key
    ) {
        dataset_identifier = make_identifier(
            seed,
            sizeof(key_type) * 8,
            sizeof(value_type) * 8,
            with_values,
            num_build_keys,
            num_probe_keys,
            key_stride,
            first_key_offset,
            sort_insert,
            sort_probe_results,
            build_key_uniformity_percent,
            probe_zipf_coefficient,
            misses_percent,
            outlier_misses_percent,
            key_multiplicity,
            min_usable_key,
            max_usable_key);

        std::string build_keys_file;
        std::string build_values_file;
        std::string probe_keys_file;
        std::string expected_result_file;

        if (cache_directory != nullptr) {
            build_keys_file = std::string(cache_directory) + "point-build-keys-" + dataset_identifier + ".bin";
            build_values_file = std::string(cache_directory) + "point-build-values-" + dataset_identifier + ".bin";
            probe_keys_file = std::string(cache_directory) + "point-probe-keys-" + dataset_identifier + ".bin";
            expected_result_file = std::string(cache_directory) + "point-expected-result-" + dataset_identifier + ".bin";

            bool keys_existed = load_binary(build_keys_file, build_keys) &&
                                load_binary(probe_keys_file, probe_keys) &&
                                load_binary(expected_result_file, expected_result) &&
                                (!with_values || load_binary(build_values_file, build_values));
            if (keys_existed) return;
        }

        std::mt19937_64 gen(seed);

        key_type min_out_of_range = min_usable_key - 1;
        key_type max_out_of_range = max_usable_key + 1;
        value_type not_found = std::numeric_limits<value_type>::max();

        size_t num_miss_keys_to_probe = num_probe_keys * misses_percent / 100;
        size_t num_out_of_range_keys_to_probe = num_probe_keys * outlier_misses_percent / 100;
        size_t num_unique_keys_for_insertion = num_build_keys / key_multiplicity;

        size_t num_keys_to_generate = num_unique_keys_for_insertion + num_reserved_keys;
        size_t num_uniform_keys_to_generate = num_keys_to_generate * build_key_uniformity_percent / 100;
        size_t num_dense_keys_to_generate = num_keys_to_generate - num_uniform_keys_to_generate;

        size_t num_keys_with_multiplicity = num_unique_keys_for_insertion * key_multiplicity;

        if (first_key_offset == 0)
            first_key_offset = min_usable_key;

        if (num_keys_to_generate + min_usable_key > max_usable_key) throw std::logic_error("impossible");
        if (first_key_offset < min_usable_key) throw std::logic_error("impossible");
        if (key_stride < 1) throw std::logic_error("impossible");
        if (key_multiplicity < 1) throw std::logic_error("impossible");

        // generate dense keys
        std::vector<key_type> generated_dense_keys(num_dense_keys_to_generate);
        for (size_t i = 0; i < num_dense_keys_to_generate; ++i) {
            generated_dense_keys[i] = key_type(i) * key_stride + first_key_offset;
        }

        // generate uniform keys
        size_t next_possible_offset = num_dense_keys_to_generate * key_stride + first_key_offset;
        size_t base_max_key = (max_usable_key - next_possible_offset) / key_stride;
        std::uniform_int_distribution<size_t> uniform_key_dist(0, base_max_key);
        std::unordered_set<key_type> generated_keys_set;
        while (generated_keys_set.size() < num_uniform_keys_to_generate) {
            key_type new_key = uniform_key_dist(gen) * key_stride + next_possible_offset;
            generated_keys_set.insert(new_key);
        }
        std::vector<key_type> generated_uniform_keys(generated_keys_set.begin(), generated_keys_set.end());

        // join key sets together
        std::vector<key_type> generated_keys(num_keys_to_generate);
        std::copy(generated_dense_keys.begin(), generated_dense_keys.end(), generated_keys.begin());
        std::copy(generated_uniform_keys.begin(), generated_uniform_keys.end(), generated_keys.begin() + num_dense_keys_to_generate);
        shuffle_vector(generated_keys, gen);

        // the first few keys will be used to build the index
        build_keys.resize(num_keys_with_multiplicity);
        for (size_t repl = 0; repl < key_multiplicity; ++repl) {
            std::copy(
                generated_keys.begin(),
                generated_keys.begin() + num_unique_keys_for_insertion,
                build_keys.begin() + num_unique_keys_for_insertion * repl
            );
        }

        // reorder the build set
        if (sort_insert) {
            sort_vector(build_keys);
        } else {
            shuffle_vector(build_keys, gen);
        }

        // if we need values, generate them here
        if (with_values) {
            build_values.resize(num_keys_with_multiplicity);
            for (size_t i = 0; i < num_keys_with_multiplicity; ++i) {
                build_values[i] = i;
            }
        }

        // compute expected values for keys (only required if a key's position is not unique)
        std::unordered_map<key_type, size_t> value_sum_for_key;
        if (key_multiplicity > 1) {
            for (size_t i = 0; i < build_keys.size(); ++i) {
                auto key = build_keys[i];
                auto value = with_values ? build_values[i] : i;
                if (value_sum_for_key.find(key) != value_sum_for_key.end()) {
                    value_sum_for_key[key] += value;
                } else {
                    value_sum_for_key[key] = value;
                }
            }
        }

        // reserve the remaining keys to simulate misses
        std::vector<key_type> reserved_keys(
                generated_keys.begin() + num_unique_keys_for_insertion,
                generated_keys.begin() + num_unique_keys_for_insertion + num_reserved_keys);

        zipf_index_distribution miss_index_distribution(num_reserved_keys, probe_zipf_coefficient);
        zipf_index_distribution hit_index_distribution(num_build_keys, probe_zipf_coefficient);

        probe_keys.resize(num_probe_keys);
        expected_result.resize(num_probe_keys);
        // fill first part with missed keys, second part with out-of-range keys, last part with existing keys
        #pragma omp parallel for
        for (size_t thread = 0; thread < 128; ++thread) {
            size_t start_index = num_probe_keys * thread / 128;
            size_t end_index = num_probe_keys * (thread + 1) / 128;
            std::mt19937_64 local_gen(seed + thread);

            for (size_t i = start_index; i < end_index; ++i) {
                if (i < num_miss_keys_to_probe) {
                    size_t random_index = miss_index_distribution(local_gen);
                    probe_keys[i] = reserved_keys[random_index];
                    expected_result[i] = key_multiplicity > 1 ? 0 : not_found;
                } else if (i < num_miss_keys_to_probe + num_out_of_range_keys_to_probe) {
                    probe_keys[i] = i & 1 ? min_out_of_range : max_out_of_range;
                    expected_result[i] = key_multiplicity > 1 ? 0 : not_found;
                } else {
                    size_t random_index = hit_index_distribution(local_gen);
                    probe_keys[i] = build_keys[random_index];
                    // use aggregated value or compute value directly
                    expected_result[i] =
                        key_multiplicity > 1 ? value_sum_for_key[build_keys[random_index]] : (with_values ? build_values[random_index] : random_index);
                }
            }
        }

        auto shuffle_perm = shuffle_permutation(num_probe_keys, gen);
        if (sort_probe_results) {
            auto sort_perm = sort_permutation(probe_keys, std::less<key_type>());
            apply_permutation(expected_result, sort_perm);
        } else {
            apply_permutation(expected_result, shuffle_perm);
        }
        apply_permutation(probe_keys, shuffle_perm);

        if (cache_directory != nullptr) {
            dump_binary(build_keys_file, build_keys);
            dump_binary(probe_keys_file, probe_keys);
            dump_binary(expected_result_file, expected_result);
            if (with_values) {
                dump_binary(build_values_file, build_values);
            }
        }
    }
};


template <typename key_type, typename value_type>
struct range_query_dataset {

    std::string dataset_identifier;
    std::vector<key_type> build_keys;
    std::vector<value_type> build_values;
    std::vector<key_type> lower_probe_keys;
    std::vector<key_type> upper_probe_keys;
    std::vector<value_type> expected_result;

    range_query_dataset(
        size_t seed,
        const char* cache_directory,
        size_t num_build_keys,
        size_t num_probe_keys,
        size_t target_expected_hits,
        // generate keys in range [0, key_range_multiplier * num_build_keys)
        size_t key_range_multiplier,
        bool sort_insert,
        bool sort_probe_results,
        bool draw_keys_uniformly,
        bool with_values,
        key_type min_usable_key,
        key_type max_usable_key
    ) {
        dataset_identifier = make_identifier(
            seed,
            sizeof(key_type) * 8,
            sizeof(value_type) * 8,
            with_values,
            num_build_keys,
            num_probe_keys,
            target_expected_hits,
            key_range_multiplier,
            sort_insert,
            sort_probe_results,
            draw_keys_uniformly,
            min_usable_key,
            max_usable_key);

        std::string build_keys_file;
        std::string build_values_file;
        std::string lower_keys_file;
        std::string upper_keys_file;
        std::string expected_result_file;
        if (cache_directory != nullptr) {
            build_keys_file = std::string(cache_directory) + "range-build-keys-" + dataset_identifier + ".bin";
            build_values_file = std::string(cache_directory) + "range-build-values-" + dataset_identifier + ".bin";
            lower_keys_file = std::string(cache_directory) + "range-lower-keys-" + dataset_identifier + ".bin";
            upper_keys_file = std::string(cache_directory) + "range-upper-keys-" + dataset_identifier + ".bin";
            expected_result_file = std::string(cache_directory) + "range-expected-result-" + dataset_identifier + ".bin";

            bool keys_existed = load_binary(build_keys_file, build_keys) &&
                                load_binary(lower_keys_file, lower_probe_keys) &&
                                load_binary(upper_keys_file, upper_probe_keys) &&
                                load_binary(expected_result_file, expected_result) &&
                                (!with_values || load_binary(build_values_file, build_values));
            if (keys_existed) return;
        }

        std::mt19937_64 gen(seed);

        if (num_build_keys + min_usable_key > max_usable_key) throw std::logic_error("impossible");
        if (key_range_multiplier < 1) throw std::logic_error("impossible");

        size_t key_range = num_build_keys * key_range_multiplier;
        size_t query_range_size = target_expected_hits * key_range_multiplier;
        key_type min_generated_key = min_usable_key;
        key_type max_generated_key = min_generated_key + key_range - 1;

        if (query_range_size > max_generated_key - min_generated_key + 1) throw std::logic_error("range query too large");

        build_keys.resize(num_build_keys);
        lower_probe_keys.resize(num_probe_keys);
        upper_probe_keys.resize(num_probe_keys);
        expected_result.resize(num_probe_keys);

        if (draw_keys_uniformly && key_range_multiplier > 1) {
            std::uniform_int_distribution<size_t> uniform_key_dist(min_generated_key, max_generated_key);
            std::unordered_set<key_type> generated_keys_set;
            while (generated_keys_set.size() < num_build_keys) {
                generated_keys_set.insert(key_type(uniform_key_dist(gen)));
            }
            std::copy(generated_keys_set.begin(), generated_keys_set.end(), build_keys.begin());
            if (sort_insert) {
                sort_vector(build_keys);
            }
        } else {
            for (size_t i = 0; i < num_build_keys; ++i) {
                build_keys[i] = key_type(i) * key_range_multiplier + min_generated_key;
            }
            if (!sort_insert) {
                shuffle_vector(build_keys, gen);
            }
        }

        // if we need values, generate them here
        if (with_values) {
            build_values.resize(num_build_keys);
            for (size_t i = 0; i < num_build_keys; ++i) {
                build_values[i] = i;
            }
        }

        std::uniform_int_distribution<key_type> range_start_distribution(min_generated_key, max_generated_key - query_range_size);
        #pragma omp parallel for
        for (size_t thread = 0; thread < 128; ++thread) {
            size_t start_index = num_probe_keys * thread / 128;
            size_t end_index = num_probe_keys * (thread + 1) / 128;
            std::mt19937_64 local_gen(seed + thread);

            for (size_t i = start_index; i < end_index; ++i) {
                lower_probe_keys[i] = range_start_distribution(local_gen);
                upper_probe_keys[i] = lower_probe_keys[i] + query_range_size - 1;
                if (upper_probe_keys[i] < lower_probe_keys[i])
                    throw std::logic_error("upper key is smaller than lower key for some reason");
            }
        }

        auto build_p = sort_permutation(build_keys, std::less<key_type>());
        std::vector<key_type> sorted_build_keys = build_keys;
        apply_permutation(sorted_build_keys, build_p);

        auto probe_p = sort_probe_results ? sort_permutation(lower_probe_keys, std::less<key_type>()) : identity_permutation(num_probe_keys);
        #pragma omp parallel for
        for (size_t i = 0; i < num_probe_keys; ++i) {
            auto lower = std::lower_bound(sorted_build_keys.begin(), sorted_build_keys.end(), lower_probe_keys[probe_p[i]]);
            auto upper = std::upper_bound(sorted_build_keys.begin(), sorted_build_keys.end(), upper_probe_keys[probe_p[i]]);
            value_type agg = 0;
            for (auto it = lower; it != upper; ++it) {
                size_t value_index = build_p[it - sorted_build_keys.begin()];
                agg += with_values ? build_values[value_index] : value_index;
            }
            expected_result[i] = agg;
        }

        if (cache_directory != nullptr) {
            dump_binary(build_keys_file, build_keys);
            dump_binary(lower_keys_file, lower_probe_keys);
            dump_binary(upper_keys_file, upper_probe_keys);
            dump_binary(expected_result_file, expected_result);
            if (with_values) {
                dump_binary(build_values_file, build_values);
            }
        }
    }
};

#endif
