// =============================================================================
// File: launch_parameters_coarse_granular.h
// Author: Justus Henneberg
// Description: Defines expressions and constants used in FliX     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef LAUNCH_PARAMETERS_COARSE_GRANULAR_H
#define LAUNCH_PARAMETERS_COARSE_GRANULAR_H

#include "definitions.cuh"


template<typename key_type>
struct coarse_granular_launch_parameters {
    OptixTraversableHandle traversable;

    const void* ordered_key_offset_pairs;

    bool long_keys;
    bool aggregate_results;
    bool find_partition_only;

    smallsize stored_size;
    smallsize partition_size;
    smallsize partition_count;

    // either 32-bit or 64-bit keys, use void if irrelevant
    const key_type* query_lower;
    const key_type* query_upper;

    smallsize* result;
};
using cg_params = coarse_granular_launch_parameters<void>;

#endif
