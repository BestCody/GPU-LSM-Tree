// =============================================================================
// File: static_tree_index_layer.cuh
// Author: Justus Henneberg
// Description: Implements static_tree_index_layer     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================


#ifndef STATIC_TREE_INDEX_LAYER_CUH
#define STATIC_TREE_INDEX_LAYER_CUH


#include "definitions_updates.cuh"
//---- NEW BUILD STATIC TREE

template <typename key_type, uint16_t node_size_log, bool use_row_layout>
GLOBALQUALIFIER
void build_static_tree_kernel(
        mem::store_type<key_type, use_row_layout> sorted_entries,
        tree_metadata metadata,
        key_type* tree
) {
    constexpr size_t node_size = 1u << node_size_log;

    const smallsize tid = blockDim.x * blockIdx.x + threadIdx.x;
    const smallsize slot = tid & (node_size - 1);
    smallsize nid = tid >> node_size_log;

    if (tid >= (metadata.total_nodes << node_size_log)) return;

    // determine which level we are on
    smallsize level = 0;
    for (; level < metadata.level_count; ++level) {
        smallsize nodes_on_level = metadata.nodes_on_level[level];
        if (nid < nodes_on_level) break;
        nid -= nodes_on_level;
    }
    smallsize level_stride = node_size;
    for (; level < metadata.level_count - 1; ++level) {
        level_stride *= node_size + 1;
    }
    const smallsize offset_within_level = (nid << node_size_log) + slot;

    // determine from where to load entry
    // for level L, level count LC, node size N, thread id tid
    // we compute offset i within level by subtracting previous level sizes from tid -> done in loop
    // we compute level stride S = N(N+1)^(LC-L-1) -> done in loop to circumvent ipow
    // at tree buffer position tid, load the key from position [S + i*S + floor(i/N)*S - 1]
    const smallsize load_offset = level_stride * (1 + offset_within_level + offset_within_level / node_size) - 1;

    // usually, we would write load_offset < size to check for valid bounds
    // but using the very last entry for an inner node would imply there is a valid child node to the right,
    // which can never be the case. so we explicitly exclude the last entry
    bool exists = load_offset < sorted_entries.size() - 1;

    // each thread writes exactly one entry
    key_type write = exists ? sorted_entries.extract_key(load_offset) : std::numeric_limits<key_type>::max();
    tree[tid] = write;
}




//--------------- STATIC TREE BUILD ---------------

#endif