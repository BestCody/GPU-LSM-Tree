// =============================================================================
// File: lsm_debug_tree.cuh
// Author: Justus Henneberg
// Description: Implements lsm_debug_tree     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef LSM_DEBUG_TREE_CUH  
#define LSM_DEBUG_TREE_CUH


#ifndef LSM_DEBUG_PRINTF
#define LSM_DEBUG_PRINTF 1
#endif

#if LSM_DEBUG_PRINTF
#define LSM_DEVICE_PRINTF(...) printf(__VA_ARGS__)
#else
#define LSM_DEVICE_PRINTF(...) ((void)0)
#endif

template <typename KeyT>
__device__ __forceinline__ unsigned long long lsm_print_cast(KeyT v)
{
    if constexpr (sizeof(KeyT) <= sizeof(unsigned long long))
    {
        return static_cast<unsigned long long>(v);
    }
    else
    {
        // If KeyT isn't integral/<=64b, adjust printing to your type.
        return 0ull;
    }
}

/**
 * Print LSM levels (each level is a single contiguous key array).
 *
 * levels_keys: device array of pointers, one per level
 * levels_sizes: device array of sizes, one per level
 * num_levels: number of levels
 *
 * NOTE: call guarded by one thread to avoid spam.
 */
template <typename KeyT>
__device__ __forceinline__ void lsm_debug_print_levels_contiguous(
    const KeyT *const *__restrict__ levels_keys,
    const uint32_t *__restrict__ levels_sizes,
    uint32_t num_levels,
    const char *tag = "LSM")
{
    if (!(blockIdx.x == 0 && threadIdx.x == 0))
        return;

    LSM_DEVICE_PRINTF("\n[%s] ---- LSM TREE DUMP (contiguous) ----\n", tag);
    for (uint32_t lvl = 0; lvl < num_levels; ++lvl)
    {
        const KeyT *keys = levels_keys[lvl];
        const uint32_t n = levels_sizes[lvl];

        LSM_DEVICE_PRINTF("L%u (%u): ", lvl, n);
        for (uint32_t i = 0; i < n; ++i)
        {
            LSM_DEVICE_PRINTF("%llu", lsm_print_cast(keys[i]));
            if (i + 1 < n)
                LSM_DEVICE_PRINTF(" ");
        }
        LSM_DEVICE_PRINTF("\n");
    }
    LSM_DEVICE_PRINTF("[%s] -----------------------------------\n\n", tag);
}
// file: lsm_debug_print_flat.cuh
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

template <typename KeyT>
__device__ __forceinline__ unsigned long long lsm_print_u64(KeyT v)
{
    return static_cast<unsigned long long>(v);
}

/**
 * Prints LSM levels from a flat array layout:
 * level k: start = ((1<<k)-1) * chunk_size, size = (1<<k) * chunk_size
 *
 * inserted_chunk_counter bit k indicates whether level k is present.
 *
 * One level per line, all keys on that line.
 */
template <typename KeyT, uint32_t chunk_size>
__device__ __forceinline__ void lsm_debug_print_levels_flat(
    const KeyT *__restrict__ level_keys,
    uint32_t level_count,
    uint32_t inserted_chunk_counter,
    const char *tag = "LSM")
{
    if (!(blockIdx.x == 0 && threadIdx.x == 0))
        return;

    printf("\n[%s] ---- LSM TREE DUMP (flat) ----\n", tag);
    for (uint32_t k = 0; k < level_count; ++k)
    {
        if ((inserted_chunk_counter & (1u << k)) == 0)
            continue;

        const uint32_t n = (1u << k) * chunk_size;
        const uint32_t start = ((1u << k) - 1u) * chunk_size;

        printf("L%u (%u): ", k, n);
        for (uint32_t i = 0; i < n; ++i)
        {
            printf("%llu", lsm_print_u64(level_keys[start + i]));
            if (i + 1 < n)
                printf(" ");
        }
        printf("\n");
    }
    printf("[%s] -------------------------------\n\n", tag);
}


/**
 * Prints:
 *  1) Staging buffer (keys on one line; tombstones as -1)
 *  2) ALL LSM levels 0..level_count-1 from a flat layout
 *     Empty level prints: "Lk [k]"
 *     Non-empty prints keys; tombstones (level_values==tombstone_value) print as -1
 *
 * Flat layout:
 *   level k: start = ((1<<k)-1) * chunk_size, size = (1<<k) * chunk_size
 */
template <typename KeyT, typename OffT, uint32_t chunk_size>
__device__ __forceinline__ void lsm_debug_print_tree_and_staging_flat_all(
    // Tree
    const KeyT* __restrict__ level_keys,
    const OffT* __restrict__ level_values,
    uint32_t level_count,
    uint32_t inserted_chunk_counter,

    // Staging
    const KeyT* __restrict__ staged_keys,
    const OffT* __restrict__ staged_values,
    uint32_t staged_size,

    // Tombstone sentinel
    OffT tombstone_value,

    const char* tag = "LSM")
{
    if (!(blockIdx.x == 0 && threadIdx.x == 0)) return;

    printf("\n[%s] ---- LSM DUMP (staging + flat levels) ----\n", tag);

    // ---- Staging buffer ----
    printf("STAGED (%u): ", staged_size);
    for (uint32_t i = 0; i < staged_size; ++i) {
        const bool is_tombstone = (staged_values != nullptr) && (staged_values[i] == tombstone_value);
        if (is_tombstone) {
            printf("-1");
        } else {
            printf("%llu", lsm_print_u64(staged_keys[i]));
        }
        if (i + 1 < staged_size) printf(" ");
    }
    printf("\n");

    // ---- Levels ----
    for (uint32_t k = 0; k < level_count; ++k) {
        const bool occupied = (inserted_chunk_counter & (1u << k)) != 0;

        if (!occupied) {
            printf("L%u [%u]\n", k, k);
            continue;
        }

        const uint32_t n = (1u << k) * chunk_size;
        const uint32_t start = ((1u << k) - 1u) * chunk_size;

        printf("L%u (%u): ", k, n);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t idx = start + i;
            const bool is_tombstone = (level_values[idx] == tombstone_value);

            if (is_tombstone) {
                printf("-1");
            } else {
                printf("%llu", lsm_print_u64(level_keys[idx]));
            }
            if (i + 1 < n) printf(" ");
        }
        printf("\n");
    }

    printf("[%s] -----------------------------------------\n\n", tag);
}

/**
 * Prints ALL LSM levels 0..level_count-1 from a flat layout.
 * Empty level (bit not set): prints "Lk [k]"
 * Non-empty: prints keys on one line; tombstones print as -1.
 */
template <typename KeyT, typename OffT, uint32_t chunk_size>
__device__ __forceinline__ void lsm_debug_print_levels_flat_all(
    const KeyT* __restrict__ level_keys,
    const OffT* __restrict__ level_values,
    uint32_t level_count,
    uint32_t inserted_chunk_counter,
    OffT tombstone_value,
    const char* tag = "LSM")
{
    if (!(blockIdx.x == 0 && threadIdx.x == 0)) return;

    printf("\n[%s] ---- LSM TREE DUMP (flat, all levels) ----\n", tag);

    for (uint32_t k = 0; k < level_count; ++k) {
        const bool occupied = (inserted_chunk_counter & (1u << k)) != 0;

        if (!occupied) {
            // Your requested format for empty levels
            printf("L%u [%u]\n", k, k);
            continue;
        }

        const uint32_t n = (1u << k) * chunk_size;
        const uint32_t start = ((1u << k) - 1u) * chunk_size;

        printf("L%u (%u): ", k, n);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t idx = start + i;
            const bool is_tombstone = (level_values[idx] == tombstone_value);

            if (is_tombstone) {
                printf("-1");
            } else {
                printf("%llu", lsm_print_u64(level_keys[idx]));
            }

            if (i + 1 < n) printf(" ");
        }
        printf("\n");
    }

    printf("[%s] -----------------------------------------\n\n", tag);
}


/**
 * Prints LSM levels from a flat array layout:
 * level k: start = ((1<<k)-1) * chunk_size, size = (1<<k) * chunk_size
 *
 * If level_values[start+i] is a tombstone (== not_found), prints key as -1.
 */
template <typename KeyT, typename OffT, uint32_t chunk_size>
__device__ __forceinline__ void lsm_debug_print_levels_flat_tombstones(
    const KeyT* __restrict__ level_keys,
    const OffT* __restrict__ level_values,
    uint32_t level_count,
    uint32_t inserted_chunk_counter,
    OffT tombstone_value,            // pass not_found
    const char* tag = "LSM")
{
    if (!(blockIdx.x == 0 && threadIdx.x == 0)) return;

    printf("\n[%s] ---- LSM TREE DUMP (flat) ----\n", tag);

    for (uint32_t k = 0; k < level_count; ++k) {
        if ((inserted_chunk_counter & (1u << k)) == 0) continue;

        const uint32_t n = (1u << k) * chunk_size;
        const uint32_t start = ((1u << k) - 1u) * chunk_size;

        printf("L%u (%u): ", k, n);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t idx = start + i;
            const bool is_tombstone = (level_values[idx] == tombstone_value);

            if (is_tombstone) {
                printf("-1");
            } else {
                printf("%llu", lsm_print_u64(level_keys[idx]));
            }

            if (i + 1 < n) printf(" ");
        }
        printf("\n");
    }

    printf("[%s] -------------------------------\n\n", tag);



}


#endif 