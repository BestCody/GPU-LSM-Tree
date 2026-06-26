// =============================================================================
// File: optix_pipeline.cuh
// Author: Justus Henneberg
// Description: Implements optix_pipeline     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef OPTIX_PIPELINE_H
#define OPTIX_PIPELINE_H

#include "cuda_buffer.cuh"
#include "optix_wrapper.cuh"

#include <vector>


struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) raygen_sbt_record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) uint8_t header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) miss_sbt_record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) uint8_t header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) hitgroup_sbt_record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) uint8_t header[OPTIX_SBT_RECORD_HEADER_SIZE];
};


class optix_pipeline {
public:
    optix_pipeline(
        optix_wrapper* optix,
        const std::string& embedded_ptx_code,
        const std::string& launch_params_name,
        const std::string& raygen_name,
        const std::string& closesthit_name = "",
        const std::string& anyhit_name = "",
        const std::string& miss_name = "",
        uint32_t num_payload_values = 0,
        uint32_t num_attribute_values = 0,
        uint32_t max_register_count = 0,
        uint32_t max_trace_depth = 1);
    ~optix_pipeline();

private:
    void create_module(
        const std::string& embedded_ptx_code,
        const std::string& launch_parameters_variable_name,
        uint32_t num_payload_values,
        uint32_t num_attribute_values,
        uint32_t max_register_count,
        uint32_t max_trace_depth);
    void create_raygen_programs(const std::string& raygen_name);
    void create_miss_programs(const std::string& miss_name);
    void create_hitgroup_programs(const std::string& closesthit_name, const std::string& anyhit_name);
    void assemble_pipeline();
    void build_sbt();

public:
    optix_wrapper* optix;
    OptixPipeline pipeline;

    std::vector<OptixProgramGroup> raygen_program_groups;
    cuda_buffer<raygen_sbt_record> raygen_records_buffer;
    std::vector<OptixProgramGroup> miss_program_groups;
    cuda_buffer<miss_sbt_record> miss_records_buffer;
    std::vector<OptixProgramGroup> hitgroup_program_groups;
    cuda_buffer<hitgroup_sbt_record> hitgroup_records_buffer;
    OptixShaderBindingTable sbt = {};

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipelineLinkOptions    pipeline_link_options = {};

    OptixModule                 module;
    OptixModuleCompileOptions   module_compile_options = {};
};

#endif
