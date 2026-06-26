// =============================================================================
// File: optix_pipeline.cu
// Author: Justus Henneberg
// Description: Launch experiments     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#include "optix_pipeline.cuh"


static void print_log(const char *message) {
    // ENABLE IF NEEDED
    //std::cerr << "[optix_pipeline]" << message << std::endl;
}


optix_pipeline::optix_pipeline(
        optix_wrapper* optix,
        const std::string& embedded_ptx_code,
        const std::string& launch_parameters_variable_name,
        const std::string& raygen_name,
        const std::string& closesthit_name,
        const std::string& anyhit_name,
        const std::string& miss_name,
        uint32_t num_payload_values,
        uint32_t num_attribute_values,
        uint32_t max_register_count,
        uint32_t max_trace_depth) : optix{optix} {

    create_module(embedded_ptx_code, launch_parameters_variable_name, num_payload_values, num_attribute_values, max_register_count, max_trace_depth);
    create_raygen_programs(raygen_name);
    create_hitgroup_programs(closesthit_name, anyhit_name);
    create_miss_programs(miss_name);
    assemble_pipeline();
    build_sbt();
}


optix_pipeline::~optix_pipeline() {
    OPTIX_CHECK(optixPipelineDestroy(pipeline))
    for (auto pg : raygen_program_groups)
        OPTIX_CHECK(optixProgramGroupDestroy(pg))
    for (auto pg : miss_program_groups)
        OPTIX_CHECK(optixProgramGroupDestroy(pg))
    for (auto pg : hitgroup_program_groups)
        OPTIX_CHECK(optixProgramGroupDestroy(pg))
    OPTIX_CHECK(optixModuleDestroy(module))
}


void optix_pipeline::create_module(
        const std::string& embedded_ptx_code,
        const std::string& launch_parameters_variable_name,
        uint32_t num_payload_values,
        uint32_t num_attribute_values,
        uint32_t max_register_count,
        uint32_t max_trace_depth) {
    module_compile_options.maxRegisterCount  = max_register_count;
    module_compile_options.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    if (optix->profiling) {
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    } else {
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    }
    pipeline_compile_options = {};
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.usesMotionBlur        = false;
    pipeline_compile_options.numPayloadValues      = num_payload_values;
    pipeline_compile_options.numAttributeValues    = num_attribute_values;
    pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = launch_parameters_variable_name.c_str();

    pipeline_link_options.maxTraceDepth = max_trace_depth;

    const std::string ptx = embedded_ptx_code;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(
            optix->optix_context,
            &module_compile_options,
            &pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            log,&sizeof_log,
            &module
    ))
    if (sizeof_log > 1) print_log(log);
}


void optix_pipeline::create_raygen_programs(const std::string& raygen_name) {
    raygen_program_groups.resize(1);

    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc pg_desc       = {};
    pg_desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pg_desc.raygen.module               = module;
    pg_desc.raygen.entryFunctionName    = raygen_name.c_str();

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
            optix->optix_context,
            &pg_desc,
            1,
            &pg_options,
            log,&sizeof_log,
            &raygen_program_groups[0]
    ))
    if (sizeof_log > 1) print_log(log);
}


void optix_pipeline::create_miss_programs(const std::string& miss_name) {
    miss_program_groups.resize(1);

    {
        OptixProgramGroupOptions pg_options = {};
        OptixProgramGroupDesc pg_desc       = {};
        pg_desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_MISS;
        if (miss_name.empty()) {
            pg_desc.miss.module             = nullptr;
            pg_desc.miss.entryFunctionName  = nullptr;
        } else {
            pg_desc.miss.module             = module;
            pg_desc.miss.entryFunctionName  = miss_name.c_str();
        }

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
                optix->optix_context,
                &pg_desc,
                1,
                &pg_options,
                log,&sizeof_log,
                &miss_program_groups[0]
        ))
        if (sizeof_log > 1) print_log(log);
    }
}


void optix_pipeline::create_hitgroup_programs(const std::string& closesthit_name, const std::string& anyhit_name) {
    hitgroup_program_groups.resize(1);

    {
        OptixProgramGroupOptions pg_options  = {};
        OptixProgramGroupDesc pg_desc        = {};
        pg_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        if (closesthit_name.empty()) {
            pg_desc.hitgroup.moduleCH            = nullptr;
            pg_desc.hitgroup.entryFunctionNameCH = nullptr;
        } else {
            pg_desc.hitgroup.moduleCH            = module;
            pg_desc.hitgroup.entryFunctionNameCH = closesthit_name.c_str();
        }
        if (anyhit_name.empty()) {
            pg_desc.hitgroup.moduleAH            = nullptr;
            pg_desc.hitgroup.entryFunctionNameAH = nullptr;
        } else {
            pg_desc.hitgroup.moduleAH            = module;
            pg_desc.hitgroup.entryFunctionNameAH = anyhit_name.c_str();
        }

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
                optix->optix_context,
                &pg_desc,
                1,
                &pg_options,
                log,&sizeof_log,
                &hitgroup_program_groups[0]
        ))
        if (sizeof_log > 1) print_log(log);
    }
}


void optix_pipeline::assemble_pipeline() {
    std::vector<OptixProgramGroup> program_groups;
    for (auto pg : raygen_program_groups)
        program_groups.push_back(pg);
    for (auto pg : miss_program_groups)
        program_groups.push_back(pg);
    for (auto pg : hitgroup_program_groups)
        program_groups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
            optix->optix_context,
            &pipeline_compile_options,
            &pipeline_link_options,
            program_groups.data(),
            (int)program_groups.size(),
            log,&sizeof_log,
            &pipeline
    ))
    if (sizeof_log > 1) print_log(log);
}


void optix_pipeline::build_sbt() {
    std::vector<raygen_sbt_record> raygen_records;
    for (int i = 0; i < raygen_program_groups.size(); i++) {
        raygen_sbt_record rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_program_groups[i], &rec));
        raygen_records.push_back(rec);
    }
    raygen_records_buffer.alloc_and_upload(raygen_records);
    sbt.raygenRecord = raygen_records_buffer.cu_ptr();

    std::vector<miss_sbt_record> miss_records;
    for (int i = 0; i < miss_program_groups.size(); i++) {
        miss_sbt_record rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_program_groups[i], &rec));
        miss_records.push_back(rec);
    }
    miss_records_buffer.alloc_and_upload(miss_records);
    sbt.missRecordBase          = miss_records_buffer.cu_ptr();
    sbt.missRecordStrideInBytes = sizeof(miss_sbt_record);
    sbt.missRecordCount         = (int)miss_records.size();

    std::vector<hitgroup_sbt_record> hitgroup_records;
    for (int i = 0; i < hitgroup_program_groups.size(); i++) {
        hitgroup_sbt_record rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_program_groups[i], &rec));
        hitgroup_records.push_back(rec);
    }
    hitgroup_records_buffer.alloc_and_upload(hitgroup_records);
    sbt.hitgroupRecordBase          = hitgroup_records_buffer.cu_ptr();
    sbt.hitgroupRecordStrideInBytes = sizeof(hitgroup_sbt_record);
    sbt.hitgroupRecordCount         = (int)hitgroup_records.size();
}
