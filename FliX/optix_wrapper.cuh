// =============================================================================
// File: optix_wrapper.cuh
// Author: Justus Henneberg
// Description: Implements optix_wrapper     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef OPTIX_WRAPPER_H
#define OPTIX_WRAPPER_H

#include "cuda_buffer.cuh"


struct optix_wrapper {
    explicit optix_wrapper(bool profiling = false);
    ~optix_wrapper();

    bool profiling;

    CUcontext cuda_context;
    OptixDeviceContext optix_context;
};

#endif
