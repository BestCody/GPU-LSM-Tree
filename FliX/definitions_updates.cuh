// =============================================================================
// File: definitions_updates.cuh
// Author: Justus Henneberg, Rosina Kharal
// Description: Implements definitions_updates     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef DEFINITIONS_UPDATES_H
#define DEFINITIONS_UPDATES_H

#if !defined(GPULSMOPT)
#include <optix.h>
#include <optix_stubs.h>
#endif
#include <cuda_runtime.h>
#include <stdio.h>
#include <limits>
#include <cstdint>
#include <array>
#include "definitions.cuh"
#include "debug_definitions_updates.cuh"
#if !defined(GPULSMOPT)
#include "../ext/cudahelpers/cuda_helpers.cuh"
#endif

#include <cooperative_groups.h>
namespace coop_g = cooperative_groups;

#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x

constexpr bool alternate_updates = false;
#define PERFORM_SUCCESSOR_PROBES 0
////----------------------------------------------------------------
#pragma once

namespace compile_flags {

// -------------------- INSERTS --------------------
#if defined(INSERTS_TILE_BULK_ONLY)
inline constexpr smallsize inserts_tile_bulk_only = 1;
#else
inline constexpr smallsize inserts_tile_bulk_only = 0;
#endif

#if defined(INSERTS_TILE_BULK_HYBRID)
inline constexpr smallsize inserts_tile_bulk_hybrid = 1;
#else
inline constexpr smallsize inserts_tile_bulk_hybrid = 0;
#endif

#if defined(TILE_INSERTS)
inline constexpr smallsize tile_inserts = 1;
#else
inline constexpr smallsize tile_inserts = 0;
#endif

#if defined(TILE_INSERTS_C)
inline constexpr smallsize tile_inserts_c = 1;
#else
inline constexpr smallsize tile_inserts_c = 0;
#endif

// -------------------- DELETES --------------------
#if defined(TILE_DELETES)
inline constexpr smallsize tile_deletes = 1;
#else
inline constexpr smallsize tile_deletes = 0;
#endif

#if defined(DELETES_TILE_BULK)
inline constexpr smallsize deletes_tile_bulk = 1;
#else
inline constexpr smallsize deletes_tile_bulk = 0;
#endif

// -------------------- OTHER / BASELINES / HASHTABLE / WARPCORE --------------------
#if defined(BASELINES)
inline constexpr smallsize baselines = 1;
#else
inline constexpr smallsize baselines = 0;
#endif

#if defined(HASHTABLE_WARPCORE)
inline constexpr smallsize hashtable_warpcore = 1;
#else
inline constexpr smallsize hashtable_warpcore = 0;
#endif

#if defined(HASHTABLE_SLAB)
inline constexpr smallsize hashtable_slab = 1;
#else
inline constexpr smallsize hashtable_slab = 0;
#endif

#if defined(GPU_BTREE)
inline constexpr smallsize gpu_btree = 1;
#else
inline constexpr smallsize gpu_btree = 0;
#endif

#if defined(LSM_TREE)
inline constexpr smallsize lsm_tree = 1;
#else
inline constexpr smallsize lsm_tree = 0;
#endif

#if defined(GPULSMOPT)
inline constexpr smallsize gpulsmopt = 1;
#else
inline constexpr smallsize gpulsmopt = 0;
#endif



} // namespace compile_flags
//////-------------------------


#ifdef DIV
constexpr smallsize DIV_FACTOR = DIV;
#else
constexpr smallsize DIV_FACTOR = 8;
#endif

#ifdef LSM_CHUNK_SIZE_LOG
constexpr size_t lsm_chunk_size_log = LSM_CHUNK_SIZE_LOG;
#else
constexpr size_t lsm_chunk_size_log = 16;
#endif


#ifdef ROUNDS_NUMBER
constexpr size_t rounds = ROUNDS_NUMBER;
#else
constexpr size_t rounds = 8;
#endif

#ifdef INITIAL_BUILD_SIZE
constexpr size_t init_build_size_log = INITIAL_BUILD_SIZE;
#else
constexpr size_t init_build_size_log = 25;
#endif

#ifdef INITIAL_PROBE_SIZE
constexpr size_t init_probe_size_log = INITIAL_PROBE_SIZE;
#else
constexpr size_t init_probe_size_log = 26;
#endif

#ifdef ST_DELETE_TOMBSTONES
constexpr bool ST_Delete_Tombstones = true;
#else
constexpr bool ST_Delete_Tombstones = false;
#endif

#ifdef XVAL
constexpr size_t percentage_distribution_dense_keys = XVAL; // defaults to 50%
#else
constexpr size_t percentage_distribution_dense_keys = 25; // defaults to 50%
#endif

#ifdef YVAL
constexpr size_t percentage_new_keys_from_dense_region = YVAL; // defaults to 90%
#else
constexpr size_t percentage_new_keys_from_dense_region = 90; // defaults to 90%
#endif

#ifdef MIN_KEYS_INSERT
constexpr smallsize min_keys_insert = MIN_KEYS_INSERT;
#else
constexpr smallsize min_keys_insert = 3;
#endif

#ifdef NODE_THRESHOLD
constexpr smallsize node_threshold = NODE_THRESHOLD;
#else
constexpr smallsize node_threshold = 75;
#endif

#ifdef NODESIZE
constexpr smallsize input_node_size = NODESIZE;
#else
constexpr smallsize input_node_size = 4;
#endif

#ifdef CACHE_LINE_SIZE
constexpr uint8_t cache_line_size = CACHE_LINE_SIZE;
#else
constexpr uint8_t cache_line_size = 0; // Default to no cache line
#endif

#ifdef MAX_NODE
#pragma message("MAX_NODE is defined as " STRINGIFY(MAX_NODE))
//#pragma message("MAX_NODE_SIZE is defined as " STRINGIFY(MAX_NODE_SIZE))
constexpr bool MAX_NODE_SIZE = MAX_NODE;
#pragma message("MAX_NODE_SIZE is defined as " STRINGIFY(MAX_NODE_SIZE))
#else
#pragma message("MAX_NODE is not defined, using default 16")
constexpr bool MAX_NODE_SIZE = 16;
#endif

#ifdef HYBRID
constexpr bool hybrid = true;
#else
constexpr bool hybrid = false;
#endif

#ifdef ALL_SHIFT_RIGHT
constexpr bool all_shift_insertions = true;
#else
constexpr bool all_shift_insertions = false;
#endif

#ifdef DEFINE_TILE_SIZE
constexpr int TILE_SIZE = DEFINE_TILE_SIZE;
#else
constexpr int TILE_SIZE = 32;
#endif

constexpr int WARP_SIZE = 32;



/* Cache Line Computations:
Nodes support n key-offset pairs of size: sizeof(key_type) + 4 bytes for offset.
For sizeof(key_type) ==4, each pair is 8 bytes, for sizeof(key_type) ==8, each pair is 12 bytes.

Additional bytes required per node:
- +1 key-offset pair for max_size, size pair at index 0 in each node:
    - max_size is of type key_type, offset is 4 bytes (type smallsize)
- plus 4 bytes for last position ,next node pointer in each node
    - last position is of type smallsiz

Cache Line Computation 8 byte key-offset pair:
    - CL 0.5: 6 key-offset pairs, +1 max_size, size pair, +4 bytes
    --> 7x8 + 4 + 4(bytes padding required) = 64 bytes
    - CL 1: 14 key-offset pairs, +1 max_size, size pair, +4 bytes
    --> 15x8 + 4 + 4(bytes padding required) = 128 bytes
    - CL 2: 30 key-offset pairs, +1 max_size, size pair, +4 bytes
    --> 31x8 + 4 + 4(bytes padding required) = 256 bytes

Cache Line Computation 12 byte key-offset pair:
    - CL 0.5: 4 key-offset pairs, +1 max_size, size pair, +4 bytes
    --> 5x12 + 4 + 0(No padding required) = 64 bytes
    - CL 1: 9 key-offset pairs, +1 max_size, size pair, +4 bytes
    --> 10x12 + 4 + 4(bytes padding required) = 128 bytes
    - CL 2: 20 key-offset pairs, +1 max_size, size pair, +4 bytes
    --> 21x12 + 4 + 0(bytes padding required) = 256 bytes   

Note: A cache_line = 5 is 0.5 cache lines, 10 is 1 cache line, 20 is 2 cache lines
*/

//constexpr smallsize max_node_size = 14;

constexpr smallsize tombstone = 1;
constexpr key64 MAX_KEY = ~0ULL;
constexpr key64 MIN_KEY = 1ULL;

template <typename key_type>
__device__ __host__ constexpr smallsize compute_key_offset_bytes() {
    if constexpr (sizeof(key_type) == 8) {
        return sizeof(key64) + sizeof(smallsize);
    } else if constexpr (sizeof(key_type) == 4) {
        return sizeof(key32) + sizeof(smallsize);
    } else {
        static_assert(sizeof(key_type) == 8 || sizeof(key_type) == 4, "Unsupported key_type size");
    }
}

template <typename key_type>
__device__ __host__ constexpr smallsize compute_node_size(uint8_t cache_line, uint8_t node_size_log) {
    if (cache_line == 0) {
       
        return 1 << node_size_log;
    } else {
        // see cache_line computations above for details
        if constexpr (sizeof(key_type) == 4) {
            
            switch (cache_line) {
                case 10: return 14;
                case 20: return 30;
                default: return 6; // cache_line: 5 (0.5)
            }
        } else if constexpr (sizeof(key_type) == 8) {
            
            switch (cache_line) {
                case 10: return 9;
                case 20: return 20;
                default: return 4; // cache_line: 5 (0.5)
            }
        } else {
            static_assert(sizeof(key_type) == 4 || sizeof(key_type) == 8, "Unsupported key_type size");
        }
    }
}

template <typename key_type>
__device__ __host__ constexpr smallsize compute_nodestride_bytes(smallsize node_size, uint8_t cache_line) {
    smallsize keyoffsetbytes = compute_key_offset_bytes<key_type>();

    if (cache_line != 0) { // Padding required
        if (sizeof(key_type) == 4) {
            return (node_size + 1) * keyoffsetbytes + (sizeof(smallsize) + 4); 
            // 32-bit All cache lines (0.5, 1, 2) need 4 bytes padding
        } else if (sizeof(key_type) == 8) {
            if (cache_line == 5 || cache_line == 20) {
                return (node_size + 1) * keyoffsetbytes + sizeof(smallsize); 
                // No padding required
            } else if (cache_line == 10) {
                return (node_size + 1) * keyoffsetbytes + (sizeof(smallsize) + 4); 
                // 64-bit 1 CL needs 4 bytes padding
            }
        }
    } 

    // No padding: normal base_2 size nodes
    return (node_size + 1) * keyoffsetbytes + sizeof(smallsize);
}

template <typename key_type>
__device__ __host__ smallsize get_lastposition_bytes(smallsize node_size){
    return (node_size + 1) * compute_key_offset_bytes<key_type>();
}


#endif
