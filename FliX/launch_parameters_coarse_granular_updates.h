// =============================================================================
// File: launch_parameters_coarse_granular_updates.h
// Author: Rosina Kharal
// Description: Defines expressions and constants used in FliX
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef LAUNCH_PARAMETERS_COARSE_GRANULAR_UPDATES_H
#define LAUNCH_PARAMETERS_COARSE_GRANULAR_UPDATES_H

#include "definitions.cuh"
#include "impl_opt_static_tree.cuh"


template<typename key_type>
struct updatable_coarse_granular_launch_parameters {
    OptixTraversableHandle traversable;

    void* ordered_node_pairs;
    void* allocation_buffer;
    const key_type* tree_buffer;
   // tree_metadata metadata;
    smallsize tree_entries_count;

    void* copy_buffer;
   //opt_static_tree *static_tree;

    const key_type*  maxvalues;
    smallsize node_stride;
    const smallsize*  offset_list;
    const key_type*  update_list;
    const key_type*  delete_list;

    key_type*  copy_update_list;
    smallsize*  copy_offset_list;

    smallsize update_size;
    bool deletions;
    smallsize free_node;
    smallsize partition_count_with_overflow;
    smallsize allocation_buffer_count;
    smallsize* reuse_list; //support compaction and reuse of nodes
    smallsize reuse_list_count;

    bool long_keys;
    bool aggregate_results;

    smallsize stored_size;
    smallsize partition_size;
    smallsize node_size;
    smallsize partition_count;

    // either 32-bit or 64-bit keys, use void if irrelevant
    const key_type* query_lower;
    const key_type* query_upper;
    smallsize query_size;
    const smallsize* bucket_indices; // precomputed bucket indices for each query

    smallsize* result;
};
using updatable_cg_params = updatable_coarse_granular_launch_parameters<void>;

#endif
