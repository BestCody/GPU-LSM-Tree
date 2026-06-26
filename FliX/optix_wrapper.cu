// =============================================================================
// File: optix_wrapper.cu
// Author: Justus Henneberg
// Description: Launch experiments     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#include "optix_wrapper.cuh"

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>


static void context_log_cb(unsigned int level, const char *tag, const char *message, void *) {
    // ENABLE IF NEEDED
    //fprintf(stderr, "[optix_wrapper][%2d][%12s]: %s\n", (int)level, tag, message);
}


optix_wrapper::optix_wrapper(bool profiling) : profiling{profiling} {
    int num;
    cudaGetDeviceCount(&num);
    if (num == 0)
        throw std::runtime_error("no CUDA capable devices found!");
    cudaSetDevice(0);
    cudaFree(0);
    OPTIX_CHECK(optixInit())

    cuCtxGetCurrent(&cuda_context); CUERR
    OPTIX_CHECK(optixDeviceContextCreate(cuda_context, 0, &optix_context))
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optix_context, context_log_cb, nullptr, 4))
}


optix_wrapper::~optix_wrapper() {
    OPTIX_CHECK(optixDeviceContextDestroy(optix_context));
}
