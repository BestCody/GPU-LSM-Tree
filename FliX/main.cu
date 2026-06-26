// =============================================================================
// File: main.cu
// Author: Justus Henneberg, Rosina Kharal
// Description: Launch experiments     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#include "result_collector.h"
#if !defined(GPULSMOPT)
#include "benchmarks.cuh"
#include "benchmarks_lookups.cuh"
#endif
#include "benchmarks_updates.cuh"
//#include "optix_wrapper.cuh"
#if !defined(GPULSMOPT)
#include "impl_binsearch.cuh"
#include "impl_hashtable_warpcore.cuh"
#include "impl_hashtable_slab.cuh"
#include "impl_rtx_index.cuh"
#include "impl_cg_rtx_index.cuh"
#include "impl_cg_rtx_index_updates.cuh"
#include "impl_scan.cuh"
#include "impl_tree_awad.cuh"
#include "impl_rtscan.cuh"
#include "impl_opt_sorted_array.cuh"
#include "impl_lsm_tree.cuh"
#endif
#include "impl_gpulsmopt.cuh"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "definitions_updates.cuh"

//optix_wrapper optix;
const char* cache_directory = nullptr;


#if !defined(GPULSMOPT)
template <typename key_type>
void test_point_query(size_t runs, bool single_plane, bool run_advanced_tests, rc::result_collector& rc) {
    benchmark_point_query<cg_rtx_index_updates<key_type, 6>>(rc, runs, single_plane, run_advanced_tests, 26, std::numeric_limits<size_t>::max());
    benchmark_point_query<cg_rtx_index_updates<key_type, 14>>(rc, runs, single_plane, run_advanced_tests, 26, std::numeric_limits<size_t>::max());
    benchmark_point_query<cg_rtx_index<key_type, 5>>(rc, runs, single_plane, run_advanced_tests);
    benchmark_point_query<cg_rtx_index<key_type, 8>>(rc, runs, single_plane, run_advanced_tests);
    benchmark_point_query<rtx_index<key_type>>(rc, runs, single_plane, run_advanced_tests);
    benchmark_point_query<sorted_array<key_type>>(rc, runs, single_plane, run_advanced_tests);
    benchmark_point_query<hashtable_warpcore<key_type>>(rc, runs, single_plane, run_advanced_tests);
    if constexpr (sizeof(key_type) < 8) {
        benchmark_point_query<tree_awad<key_type>>(rc, runs, single_plane, run_advanced_tests);
    }
}


template <typename key_type>
void test_range_query(size_t runs, rc::result_collector& rc) {
    benchmark_range_query<rtx_index<key_type>>(rc, runs);
    benchmark_range_query<cg_rtx_index<key_type, 5>>(rc, runs);
    benchmark_range_query<cg_rtx_index<key_type, 8>>(rc, runs);
    benchmark_range_query<sorted_array<key_type>>(rc, runs);
    benchmark_range_query<scan<key_type>>(rc, runs, 26, 16);
    if constexpr (sizeof(key_type) < 8) {
        benchmark_range_query<rtscan<key_type>>(rc, runs, std::numeric_limits<size_t>::max(), 20);
        benchmark_range_query<tree_awad<key_type>>(rc, runs);
    }
}


template <typename key_type, typename value_type>
void point_query_dataset_to_csv(const point_query_dataset<key_type, value_type>& dataset) {
    {
        std::ofstream file("point-build-keys-" + dataset.dataset_identifier + ".csv");
        file << "index,VALUE\n";
        for (size_t i = 0; i < dataset.build_keys.size(); i++) {
            file << i << "," << dataset.build_keys[i] << "\n";
        }
    }
    {
        std::ofstream file("point-probe-keys-" + dataset.dataset_identifier + ".csv");
        file << "index,VALUE\n";
        for (size_t i = 0; i < dataset.probe_keys.size(); i++) {
            file << i << "," << dataset.probe_keys[i] << "\n";
        }
    }
}


template <typename index_type>
void test_interface() {
    //static_assert(index_type::can_lookup, "index must support lookup");
    static_assert(index_type::can_lookup == operation_support::async, "index must support async lookup");
    
    using key_type = typename index_type::key_type;

    std::vector<key_type> build_keys =
            {100, 90, 80, 70, 60, 50, 40, 30, 20, 10};
    std::vector<key_type> insert_keys =
            {5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105};
    std::vector<smallsize> insert_offsets =
            {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<key_type> delete_keys =
            {30, 35, 40, 45, 75,80};
    std::vector<key_type> probe_keys =
            {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105};
    cuda_buffer<key_type> build_keys_buffer, insert_keys_buffer, delete_keys_buffer, probe_keys_buffer;
    cuda_buffer<smallsize> result_buffer;
    cuda_buffer<smallsize> insert_offsets_buffer;
    build_keys_buffer.alloc_and_upload(build_keys);
    insert_keys_buffer.alloc_and_upload(insert_keys);
    insert_offsets_buffer.alloc_and_upload(insert_offsets);
    delete_keys_buffer.alloc_and_upload(delete_keys);
    probe_keys_buffer.alloc_and_upload(probe_keys);
    result_buffer.alloc_for_size(probe_keys);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    index_type index;
    auto show = [&](){
        index.lookup(probe_keys_buffer.ptr(), result_buffer.ptr(), probe_keys.size(), stream);
        cudaDeviceSynchronize();
        auto results = result_buffer.download(probe_keys.size());
        for (size_t i = 0; i < probe_keys.size(); i++) {
            std::cout << "lookup(" << probe_keys[i] << ") = " << results[i] << std::endl;
        }
    };

   
    double time_ms = 0;
    size_t build_bytes = 0;

      index.build(build_keys_buffer.ptr(), build_keys.size(), build_keys.size() + insert_keys.size(), 1000000000, &time_ms, &build_bytes);
  //  index.build(build_keys_buffer.ptr(), build_keys.size(), &time_ms, &build_bytes);
    std::cout << "build time: " << time_ms << " ms, build bytes: " << build_bytes << std::endl;
    show();
    //if constexpr (index_type::can_update) {
    if constexpr (index_type::can_update == operation_support::async) {

        index.insert(insert_keys_buffer.ptr(), insert_offsets_buffer.ptr(), insert_keys.size(), stream);
        show();
        index.remove(delete_keys_buffer.ptr(), delete_keys.size(), stream);
        show();
    }

    cudaStreamDestroy(stream);
}

#endif

int main(int argc, char* argv[]) {
    constexpr size_t runs = TOTALRUNS;
    constexpr size_t reduced_runs = 1 ;
    constexpr bool run_advanced_tests = true;

    if (argc > 1) cache_directory = argv[1];

   {

#ifdef MAIN_32
#pragma message "32 bit version MAIN_32=YES"
rc::result_collector rc;


#ifdef BASELINES
#pragma message "BASELINES=YES"
//  benchmark_updates<hashtable_warpcore<key32, 20>>(rc, runs);
//  -------------benchmark_updates<cg_rtx_index<key32, 5>>(rc, runs);
    #ifdef HASHTABLE_WARPCORE
        #pragma message "HASHTABLE_WARPCORE=YES"
        benchmark_updates<hashtable_warpcore<key32, 80>>(rc, runs);
    #elif defined(HASHTABLE_SLAB)
        #pragma message "HASHTABLE_SLAB=YES"
        benchmark_updates<hashtable_slab<key32>>(rc, runs);
        //benchmark_updates<hashtable_warpcore<key32, 80>>(rc, runs);
    #elif defined(GPU_BTREE)
        #pragma message "GPU_BTREE=YES"
        benchmark_updates<tree_awad<key32, true>>(rc, runs);
    #elif defined(LSM_TREE)
    #pragma message "LSM_TREE=YES"
        benchmark_updates<lsm_tree_ashkiani<key32, lsm_chunk_size_log>>(rc, runs);
    #elif defined(GPULSMOPT)
        #pragma message "GPULSMOPT=YES"
        benchmark_updates<gpulsmopt<key32>>(rc, runs);
    #else
        printf("No Baselines Selected \n");
    #endif
    // benchmark_updates<cg_rtx_index_updates<key32, 6, 50, 120>>(rc, runs);
    // benchmark_updates<cg_rtx_index_updates<key32, 10, 0, 50, 200>>(rc, runs);
    // benchmark_updates<cg_rtx_index_updates<key32, 20, 0, 50, 200>>(rc, runs);
    // --- benchmark_updates<cg_rtx_index_updates<key32, 0, 5, 50, 200>>(rc, runs);
    // benchmark_updates<cg_rtx_index_updates<key32, 10, 0, 50, 200>>(rc, runs);
#else
#pragma message "BASELINES=NO"
         //benchmark_updates<hashtable_warpcore<key32, 30>>(rc, runs);
        //printf("nodesize ,cachelinesize %d, %d\n", input_node_size, cache_line_size);
        benchmark_updates<cg_rtx_index_updates<key32, cache_line_size, input_node_size, 50>>(rc, runs);
#endif
       // benchmark_updates<cg_rtx_index_updates<key32, 0, 4, 50, 100>>(rc, runs);
       // benchmark_updates<cg_rtx_index_updates<key32, 10, 0, 40, 200>>(rc, runs);

       // benchmark_updates<cg_rtx_index_updates<key64, 20, 0, 50, 150>>(rc, runs);
       // benchmark_updates<tree_awad<key32, 200>>(rc, runs);
       // 
       // benchmark_updates<hashtable_warpcore<key32, 30>>(rc, runs);
       // --benchmark_updates<rtx_index<key32>>(rc, runs);

#else
#pragma message "64 bit version MAIN_32=NO"
        rc::result_collector rc;

       //benchmark_updates<cg_rtx_index<key64, 3>>(rc, runs);

        //benchmark_updates<cg_rtx_index_updates<key32, 6, 50, 120>>(rc, runs);
        // benchmark_updates<cg_rtx_index_updates<key32, 10, 0, 50, 200>>(rc, runs);
        //benchmark_updates<cg_rtx_index_updates<key64, 0, 10, 50, 125>>(rc, runs);
        benchmark_updates<cg_rtx_index_updates<key64, 0, 4, 50>>(rc, runs);


       //benchmark_updates<cg_rtx_index_updates<key32, 10, 0, 40, 200>>(rc, runs);

       // benchmark_updates<cg_rtx_index_updates<key64, 20, 0, 50, 150>>(rc, runs);
       // ---> benchmark_updates<tree_awad<key64, 125>>(rc, runs);
       //
       // benchmark_updates<hashtable_warpcore_warpcore<key64, 40>>(rc, runs);
       // benchmark_updates<rtx_index<key32>>(rc, runs);

#endif


    
       
       /* SIMPLE RUN
        benchmark_updates<cg_rtx_index_updates<key32, 30, 50, 120>>(rc, runs);
        benchmark_updates<hashtable_warpcore<key32, 40>>(rc, runs);
        //benchmark_updates<hashtable_warpcore<key64, 40>>(rc, runs);
        benchmark_updates<tree_awad<key32, 125>>(rc, runs);
        benchmark_updates<rtx_index<key32>>(rc, runs);
        //benchmark_updates<rtx_index<key64>>(rc, runs);
        benchmark_updates<cg_rtx_index<key32, 5>>(rc, runs);
        //benchmark_updates<cg_rtx_index<key64, 5>>(rc, runs);
        benchmark_updates<cg_rtx_index<key32, 8>>(rc, runs);
        //benchmark_updates<cg_rtx_index<key64, 8>>(rc, runs);

        */
        std::ofstream result_file("updates.csv");
        rc.write_csv(result_file, rc::first_line_header, rc::long_form, rc::pack_columns);
   }

/*
    {
        rc::result_collector rc;
        test_point_query<key32>(runs, false, run_advanced_tests, rc);
        std::ofstream result_file("comparison-point-query-k32.csv");
        rc.write_csv(result_file, rc::first_line_header, rc::long_form, rc::pack_columns);
    }
    {
        rc::result_collector rc;
        test_point_query<key64>(runs, false, run_advanced_tests, rc);
        std::ofstream result_file("comparison-point-query-k64.csv");
        rc.write_csv(result_file, rc::first_line_header, rc::long_form, rc::pack_columns);
    }

*/
    /*{
        rc::result_collector rc;
        test_range_query<key32>(reduced_runs, rc);
        std::ofstream result_file("comparison-range-query-k32.csv");
        rc.write_csv(result_file, rc::first_line_header, rc::long_form, rc::pack_columns);
    }
    {
        rc::result_collector rc;
        test_range_query<key64>(reduced_runs, rc);
        std::ofstream result_file("comparison-range-query-k64.csv");
        rc.write_csv(result_file, rc::first_line_header, rc::long_form, rc::pack_columns);
    }

    // ignore this
    /*{
        rc::result_collector rc;
        test_sorted_array_variants<key32>(runs, rc);
        test_sorted_array_variants<key64>(runs, rc);
        std::ofstream result_file("comparison-sorted-array.csv");
        rc.write_csv(result_file, rc::first_line_header, rc::long_form, rc::pack_columns);
    }*/

    return 0;
}
