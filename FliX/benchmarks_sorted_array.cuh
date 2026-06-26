// =============================================================================
// File: benchmarks_sorted_array.cuh
// Author: Justus Henneberg
// Description: Implements benchmarks_sorted_array     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef BENCHMARKS_BINARY_SEARCH_CUH
#define BENCHMARKS_BINARY_SEARCH_CUH

// todo move design experiments for improved sorted array here

/*
template <typename key_type, uint16_t threads_per_block, uint16_t items_per_thread>
void test_sorted_array_variants_helper(size_t runs, rc::result_collector& rc) {
    benchmark_point_query<opt_sorted_array<key_type, sa::opt::none, threads_per_block, items_per_thread>>(rc, runs, false, false);
    benchmark_point_query<opt_sorted_array<key_type, sa::opt::share_levels, threads_per_block, items_per_thread>>(rc, runs, false, false);
    benchmark_point_query<opt_sorted_array<key_type, sa::opt::share_levels_partially, threads_per_block, items_per_thread>>(rc, runs, false, false);
    benchmark_point_query<opt_sorted_array<key_type, sa::opt::sort_lookups_locally, threads_per_block, items_per_thread>>(rc, runs, false, false);
    benchmark_point_query<opt_sorted_array<key_type, sa::opt::sort_unsort_lookups_locally, threads_per_block, items_per_thread>>(rc, runs, false, false);
    benchmark_point_query<opt_sorted_array<key_type, sa::opt::sort_unsort_lookups_locally | sa::opt::share_levels_partially, threads_per_block, items_per_thread>>(rc, runs, false, false);
}
*/


// todo rename
/*
template <typename key_type>
void test_sorted_array_variants(size_t runs, rc::result_collector& rc) {
    benchmark_point_query<sorted_array<key_type>>(rc, runs, false, false);
    benchmark_point_query<hashtable<key_type>>(rc, runs, false, false);
    if constexpr (sizeof(key_type) < 8) {
        benchmark_point_query<tree<key_type>>(rc, runs, false, false);
    }
    test_sorted_array_variants_helper<key_type, 1024, 1>(runs, rc);
    test_sorted_array_variants_helper<key_type, 512, 2>(runs, rc);
    test_sorted_array_variants_helper<key_type, 256, 4>(runs, rc);
    test_sorted_array_variants_helper<key_type, 128, 8>(runs, rc);
    test_sorted_array_variants_helper<key_type, 64, 16>(runs, rc);

    test_sorted_array_variants_helper<key_type, 1024, 3>(runs, rc);
    test_sorted_array_variants_helper<key_type, 512, 6>(runs, rc);
    test_sorted_array_variants_helper<key_type, 256, 12>(runs, rc);
    test_sorted_array_variants_helper<key_type, 128, 24>(runs, rc);
    test_sorted_array_variants_helper<key_type, 64, 48>(runs, rc);

    test_sorted_array_variants_helper<key_type, 128, 10>(runs, rc); // opt
    test_sorted_array_variants_helper<key_type, 64, 46>(runs, rc);  // opt
}

*/

#endif
