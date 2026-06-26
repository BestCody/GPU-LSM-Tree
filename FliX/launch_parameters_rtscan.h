// =============================================================================
// File: launch_parameters_rtscan.h
// Author: Justus Henneberg
// Description: Defines expressions and constants used in FliX     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef LAUNCH_PARAMETERS_RTSCAN_H
#define LAUNCH_PARAMETERS_RTSCAN_H

#include "definitions.cuh"


struct rtscan_params {
    OptixTraversableHandle traversable;

    const key32* stored_keys;

    double ray_spacing;

    uint32_t* bitmaps;
    smallsize bitmap_entries;

    const key32* query_lower;
    const key32* query_upper;
    smallsize* result;
    smallsize batch_size;
};

#endif
