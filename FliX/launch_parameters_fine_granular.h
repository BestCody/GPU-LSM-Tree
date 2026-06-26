// =============================================================================
// File: launch_parameters_fine_granular.h
// Author: Justus Henneberg
// Description: Defines expressions and constants used in FliX     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef LAUNCH_PARAMETERS_FINE_GRANULAR_H
#define LAUNCH_PARAMETERS_FINE_GRANULAR_H

#include "definitions.cuh"


template<typename key_type>
struct fine_granular_launch_parameters {
    OptixTraversableHandle traversable;

    bool long_keys;
    bool has_range_queries;
    bool keys_are_unique;

    // either 32-bit or 64-bit keys, use void if irrelevant
    const key_type* query_lower;
    const key_type* query_upper;

    smallsize* result;
};
using fg_params = fine_granular_launch_parameters<void>;

#endif
