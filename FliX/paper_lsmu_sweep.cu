#include <iostream>

#include "utilities.cuh"

#if defined(PAPER_SWEEP_GPULSMOPT)
#include "impl_gpulsmopt.cuh"
#else
#include "impl_lsm_tree.cuh"
#endif

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef PAPER_LSM_BATCH_LOG
#define PAPER_LSM_BATCH_LOG 16
#endif

namespace
{

using key_type = std::uint32_t;
using clock_type = std::chrono::steady_clock;

#if defined(PAPER_SWEEP_GPULSMOPT)
constexpr const char *index_name = "GPULSMOpt";
template <unsigned BatchLog>
using paper_index_type = gpulsmopt<key_type>;
#else
constexpr const char *index_name = "LSMu";
template <unsigned BatchLog>
using paper_index_type = lsm_tree_ashkiani<key_type, BatchLog>;
#endif

constexpr unsigned batch_log = PAPER_LSM_BATCH_LOG;
constexpr std::uint64_t key_domain = std::uint64_t{1} << 30;
constexpr key_type key_mask = static_cast<key_type>(key_domain - 1);
constexpr std::uint64_t paper_insert_limit = std::uint64_t{1} << 27;
constexpr unsigned threads = 256;

void check_cuda(cudaError_t error, const char *expression)
{
    if (error != cudaSuccess)
        throw std::runtime_error(
            std::string(expression) + ": " + cudaGetErrorString(error));
}

#define PAPER_CUDA(expression) check_cuda((expression), #expression)

__host__ __device__ key_type key_for_index(std::uint32_t index)
{
    return (index * 747796405u + 289133645u) & key_mask;
}

__host__ __device__ std::uint32_t mix32(std::uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

__global__ void fill_insert_batch(
    key_type *keys,
    smallsize *values,
    std::uint32_t begin,
    std::uint32_t size)
{
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;
    keys[tid] = key_for_index(begin + tid);
    values[tid] = 1;
}

__global__ void fill_build_keys(key_type *keys, std::uint32_t size)
{
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
        keys[tid] = key_for_index(tid);
}

__global__ void fill_lookup_queries(
    key_type *queries,
    std::uint32_t size,
    std::uint32_t resident,
    bool hits,
    std::uint32_t seed)
{
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;
    const std::uint32_t random = mix32(tid + seed);
    if (hits)
        queries[tid] = key_for_index(random % resident);
    else
    {
        const std::uint32_t miss_begin = resident > paper_insert_limit
            ? resident : static_cast<std::uint32_t>(paper_insert_limit);
        queries[tid] = key_for_index(
            miss_begin +
            random % static_cast<std::uint32_t>(
                key_domain - miss_begin));
    }
}

__global__ void fill_range_queries(
    key_type *lower,
    key_type *upper,
    std::uint32_t size,
    std::uint32_t resident,
    std::uint32_t expected_hits,
    std::uint32_t seed)
{
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;
    std::uint64_t width =
        (std::uint64_t{expected_hits} * key_domain + resident - 1) /
        resident;
    width = width == 0 ? 1 : width;
    width = width > key_domain ? key_domain : width;
    const std::uint64_t starts = key_domain - width + 1;
    const std::uint64_t first = mix32(tid + seed) % starts;
    lower[tid] = static_cast<key_type>(first);
    upper[tid] = static_cast<key_type>(first + width - 1);
}

__global__ void validate_lookup_results(
    const smallsize *results,
    std::uint32_t size,
    bool hits,
    unsigned long long *errors)
{
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;
    const smallsize expected = hits ? 1u : not_found;
    if (results[tid] != expected)
        atomicAdd(errors, 1ull);
}

__global__ void validate_lookup_presence(
    const smallsize *results,
    std::uint32_t size,
    bool hits,
    unsigned long long *errors)
{
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;
    const bool found = results[tid] != not_found;
    if (found != hits)
        atomicAdd(errors, 1ull);
}

#if !defined(PAPER_SWEEP_GPULSMOPT)

__global__ void compare_results(
    const smallsize *left,
    const smallsize *right,
    std::uint32_t size,
    unsigned long long *errors)
{
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size && left[tid] != right[tid])
        atomicAdd(errors, 1ull);
}

#endif

__device__ std::uint64_t digest_mix(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

__global__ void digest_range_results(
    const smallsize *results,
    std::uint32_t size,
    std::uint32_t query_offset,
    unsigned long long *digest)
{
    __shared__ unsigned long long sums[threads];
    __shared__ unsigned long long xors[threads];
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = tid < size;
    const std::uint64_t query = std::uint64_t{query_offset} + tid;
    const std::uint64_t result = active ? results[tid] : 0;
    sums[threadIdx.x] = result;
    xors[threadIdx.x] = active
        ? digest_mix((query << 32) | result)
        : 0;
    __syncthreads();
    for (std::uint32_t stride = threads / 2; stride; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
            xors[threadIdx.x] ^= xors[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        atomicAdd(digest, sums[0]);
        atomicXor(digest + 1, xors[0]);
    }
}

[[maybe_unused]] __global__ void fill_cleanup_batch(
    key_type *keys,
    smallsize *values,
    std::uint32_t begin,
    std::uint32_t unique_count,
    std::uint32_t size)
{
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;
    const std::uint32_t index = begin + tid;
    const std::uint32_t source =
        index < unique_count ? index : (index - unique_count) % unique_count;
    keys[tid] = key_for_index(source);
    values[tid] = 1;
}

class gpu_timer
{
public:
    gpu_timer()
    {
        PAPER_CUDA(cudaEventCreate(&start_));
        PAPER_CUDA(cudaEventCreate(&stop_));
    }

    ~gpu_timer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    template <typename Function>
    double measure(Function &&function)
    {
        PAPER_CUDA(cudaEventRecord(start_));
        function();
        PAPER_CUDA(cudaEventRecord(stop_));
        PAPER_CUDA(cudaEventSynchronize(stop_));
        float milliseconds = 0;
        PAPER_CUDA(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

struct options
{
    std::filesystem::path output_directory = "paper_sweep_results";
    unsigned insert_limit_log = 27;
    unsigned query_limit_log = 24;
    unsigned range_chunk_log = 18;
    std::uint32_t stop_after_r = 0;
    bool main_sweep = true;
    bool cleanup_sweep = false;
    bool bulk_sweep = false;
    std::uint32_t profile_insert_r = 0;
    bool profile_all_inserts = false;
    bool forced_unified_validation = false;
    bool construction_only = false;
#if defined(PAPER_SWEEP_GPULSMOPT)
    bool include_count = false;
#else
    bool include_count = true;
#endif
};

unsigned parse_unsigned(const char *value, const char *name)
{
    const std::string text(value);
    size_t consumed = 0;
    const unsigned long result = std::stoul(text, &consumed);
    if (consumed != text.size() || result > 31)
        throw std::invalid_argument(std::string("invalid ") + name);
    return static_cast<unsigned>(result);
}

options parse_options(int argc, char **argv)
{
    options result;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument(argv[i]);
        auto require_value = [&]() -> const char * {
            if (++i >= argc)
                throw std::invalid_argument("missing value after " + argument);
            return argv[i];
        };
        if (argument == "--output")
            result.output_directory = require_value();
        else if (argument == "--insert-limit-log")
            result.insert_limit_log =
                parse_unsigned(require_value(), "insert limit log");
        else if (argument == "--query-limit-log")
            result.query_limit_log =
                parse_unsigned(require_value(), "query limit log");
        else if (argument == "--range-chunk-log")
            result.range_chunk_log =
                parse_unsigned(require_value(), "range chunk log");
        else if (argument == "--stop-after-r")
            result.stop_after_r = static_cast<std::uint32_t>(
                std::stoul(require_value()));
        else if (argument == "--profile-insert-r")
            result.profile_insert_r = static_cast<std::uint32_t>(
                std::stoul(require_value()));
        else if (argument == "--profile-all-inserts")
            result.profile_all_inserts = true;
        else if (argument == "--forced-unified-validation-only")
        {
            result.main_sweep = false;
            result.cleanup_sweep = false;
            result.bulk_sweep = false;
            result.forced_unified_validation = true;
        }
        else if (argument == "--construction-only")
        {
            result.main_sweep = false;
            result.cleanup_sweep = false;
            result.bulk_sweep = false;
            result.construction_only = true;
        }
        else if (argument == "--main-only")
        {
            result.main_sweep = true;
            result.cleanup_sweep = false;
            result.bulk_sweep = false;
        }
        else if (argument == "--cleanup-only")
        {
            result.main_sweep = false;
            result.cleanup_sweep = true;
            result.bulk_sweep = false;
        }
        else if (argument == "--bulk-only")
        {
            result.main_sweep = false;
            result.cleanup_sweep = false;
            result.bulk_sweep = true;
        }
        else if (argument == "--range-only")
            result.include_count = false;
        else
            throw std::invalid_argument("unknown argument: " + argument);
    }
    if (result.profile_insert_r != 0 && result.profile_all_inserts)
        throw std::invalid_argument(
            "--profile-insert-r and --profile-all-inserts are mutually exclusive");
    return result;
}

template <typename Function>
double measure_wall_ms(Function &&function)
{
    const auto start = clock_type::now();
    function();
    const auto stop = clock_type::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

void write_metadata(const options &configuration)
{
    std::filesystem::create_directories(configuration.output_directory);
    std::ofstream output(
        configuration.output_directory /
        ("metadata_b" + std::to_string(batch_log) + ".txt"));
    cudaDeviceProp properties{};
    PAPER_CUDA(cudaGetDeviceProperties(&properties, 0));
    output << "paper=1707.05354v2.pdf\n";
    output << "index=" << index_name << '\n';
    output << "gpu=" << properties.name << '\n';
    output << "batch_log=" << batch_log << '\n';
    output << "insert_limit_log=" << configuration.insert_limit_log << '\n';
    output << "query_limit_log=" << configuration.query_limit_log << '\n';
    output << "range_chunk_log=" << configuration.range_chunk_log << '\n';
    output << "count_enabled=" << configuration.include_count << '\n';
    output << "range_implementation=enumeration_checksum\n";
    output << "profile_insert_r=" << configuration.profile_insert_r << '\n';
    output << "profile_all_inserts=" << configuration.profile_all_inserts << '\n';
    output << "forced_unified_validation="
           << configuration.forced_unified_validation << '\n';
    output << "construction_only=" << configuration.construction_only << '\n';
#if defined(PAPER_SWEEP_GPULSMOPT)
    output << "gpulsmopt_canonical_carry=1\n";
    output << "gpulsmopt_canonical_local_epoch=1\n";
    output << "gpulsmopt_canonical_tournament_merge=1\n";
    output << "gpulsmopt_canonical_tournament_minimum_sources="
           << gpulsmopt2_detail::kCanonicalTournamentMinimumSources << '\n';
    output << "gpulsmopt_canonical_compact_multiway=1\n";
    output << "gpulsmopt_canonical_tournament_capacity_ceiling="
           << gpulsmopt2_detail::kCanonicalTournamentCapacityCeiling << '\n';
    output << "gpulsmopt_canonical_tournament_workspace="
           << "elastic_by_job_shape\n";
    output << "gpulsmopt_canonical_publication_graph=1\n";
    output << "gpulsmopt_maximum_batch_capacity="
           << gpulsmopt_adapter_detail::batch_capacity() << '\n';
    output << "gpulsmopt_level_zero_capacity="
           << gpulsmopt_adapter_detail::level_zero_capacity() << '\n';
#endif
}

#if defined(PAPER_SWEEP_GPULSMOPT)

[[maybe_unused]] key_type sparse_validation_key(std::uint32_t quotient)
{
    const std::uint32_t suffix =
        (quotient * 40503u + 17u) & 0xffffu;
    return (quotient << 16u) | suffix;
}

void run_forced_unified_validation(const options &configuration)
{
    using index_type = paper_index_type<batch_log>;
    constexpr std::uint32_t maximum_batch = 4096u;
    constexpr std::uint32_t sparse_batch = 2048u;
    constexpr std::uint32_t dense_batch = 4096u;
    constexpr std::uint32_t duplicate_batch = 4096u;
    constexpr std::uint32_t duplicate_keys = 1024u;
    constexpr std::uint32_t dense_quotient = 30000u;

    cuda_buffer<key_type> keys;
    cuda_buffer<smallsize> values;
    keys.alloc(maximum_batch);
    values.alloc(maximum_batch);

    size_t free_memory = 0;
    size_t total_memory = 0;
    PAPER_CUDA(cudaMemGetInfo(&free_memory, &total_memory));
    index_type index;
    index.build(
        nullptr, 0, std::uint32_t{1} << 20u, free_memory,
        nullptr, nullptr);

    std::unordered_map<key_type, smallsize> expected;
    std::vector<key_type> removed;
    std::ofstream results(
        configuration.output_directory / "forced_unified_validation.csv");
    results << "scenario,live_keys,queries,errors\n";

    auto validate = [&](const char *scenario) {
        std::vector<key_type> host_queries;
        std::vector<smallsize> host_expected;
        host_queries.reserve(expected.size() + removed.size() + 4096u);
        host_expected.reserve(host_queries.capacity());
        for (const auto &[key, value] : expected)
        {
            host_queries.push_back(key);
            host_expected.push_back(value);
        }
        for (const key_type key : removed)
        {
            host_queries.push_back(key);
            host_expected.push_back(not_found);
        }
        for (std::uint32_t suffix = 0u;
             host_queries.size() < expected.size() + removed.size() + 4096u;
             ++suffix)
        {
            const key_type key = (25000u << 16u) | (suffix & 0xffffu);
            if (expected.find(key) != expected.end())
                continue;
            host_queries.push_back(key);
            host_expected.push_back(not_found);
        }

        cuda_buffer<key_type> query_keys;
        cuda_buffer<smallsize> query_results;
        query_keys.alloc_and_upload(host_queries);
        query_results.alloc(host_queries.size());
        index.lookup(
            query_keys.ptr(), query_results.ptr(), host_queries.size(), 0);
        PAPER_CUDA(cudaDeviceSynchronize());
        const auto actual = query_results.download(host_queries.size());
        std::uint64_t errors = 0u;
        for (std::size_t i = 0u; i < actual.size(); ++i)
            errors += actual[i] != host_expected[i];
        results << scenario << ',' << expected.size() << ','
                << host_queries.size() << ',' << errors << '\n';
        results.flush();
        std::cout << "FORCED_UNIFIED_VALIDATION"
                  << " scenario=" << scenario
                  << " live_keys=" << expected.size()
                  << " queries=" << host_queries.size()
                  << " errors=" << errors << std::endl;
        if (errors)
            throw std::runtime_error(
                std::string("forced-unified validation failed in ") +
                scenario + ": " + std::to_string(errors));
    };

    std::vector<key_type> host_keys(maximum_batch);
    std::vector<smallsize> host_values(maximum_batch);

    // One key in each of 32,768 quotients: deliberately sparse rows.
    for (std::uint32_t batch = 0u; batch < 16u; ++batch)
    {
        for (std::uint32_t i = 0u; i < sparse_batch; ++i)
        {
            const std::uint32_t quotient = batch * sparse_batch + i;
            host_keys[i] = sparse_validation_key(quotient);
            host_values[i] = 0x01000000u + quotient;
            expected[host_keys[i]] = host_values[i];
        }
        keys.upload(host_keys.data(), sparse_batch);
        values.upload(host_values.data(), sparse_batch);
        index.insert(keys.ptr(), values.ptr(), sparse_batch, 0);
    }
    PAPER_CUDA(cudaDeviceSynchronize());
    validate("sparse_unique");

    // Fill every suffix of one quotient: deliberately dense and crowded.
    for (std::uint32_t batch = 0u; batch < 16u; ++batch)
    {
        for (std::uint32_t i = 0u; i < dense_batch; ++i)
        {
            const std::uint32_t suffix = batch * dense_batch + i;
            host_keys[i] = (dense_quotient << 16u) | suffix;
            host_values[i] = 0x02000000u + suffix;
            expected[host_keys[i]] = host_values[i];
        }
        keys.upload(host_keys.data(), dense_batch);
        values.upload(host_values.data(), dense_batch);
        index.insert(keys.ptr(), values.ptr(), dense_batch, 0);
    }
    PAPER_CUDA(cudaDeviceSynchronize());
    validate("dense_unique");

    // Four copies per key in every batch. The final position of the newest
    // batch is the expected winner.
    for (std::uint32_t batch = 0u; batch < 16u; ++batch)
    {
        for (std::uint32_t i = 0u; i < duplicate_batch; ++i)
        {
            const std::uint32_t source = i % duplicate_keys;
            host_keys[i] = sparse_validation_key(source);
            host_values[i] = 0x10000000u + batch * duplicate_batch + i;
            expected[host_keys[i]] = host_values[i];
        }
        keys.upload(host_keys.data(), duplicate_batch);
        values.upload(host_values.data(), duplicate_batch);
        index.insert(keys.ptr(), values.ptr(), duplicate_batch, 0);
    }
    PAPER_CUDA(cudaDeviceSynchronize());
    validate("duplicate_newest_wins");

    // Repeated tombstones must hide both the newest updates and older rows.
    removed.clear();
    removed.reserve(duplicate_keys);
    for (std::uint32_t i = 0u; i < duplicate_keys; ++i)
    {
        host_keys[i] = sparse_validation_key(i);
        removed.push_back(host_keys[i]);
        expected.erase(host_keys[i]);
    }
    keys.upload(host_keys.data(), duplicate_keys);
    for (std::uint32_t batch = 0u; batch < 16u; ++batch)
        index.remove(keys.ptr(), duplicate_keys, 0);
    PAPER_CUDA(cudaDeviceSynchronize());
    validate("tombstone_visibility");

    // Exercise rollover with several occupied levels. Repeated updates keep
    // the live set bounded while two complete hierarchy cycles are consumed.
    {
        DictionaryConfig rollover_config;
        rollover_config.max_elements = 512u;
        rollover_config.batch_capacity = 4u;
        rollover_config.level_zero_capacity = 64u;
        GPULSMOpt rollover_index(rollover_config);
        cuda_buffer<std::uint32_t> rollover_keys;
        cuda_buffer<std::uint32_t> rollover_values;
        rollover_keys.alloc(256u);
        rollover_values.alloc(256u);
        std::array<std::uint32_t, 4u> key_chunk{};
        std::array<std::uint32_t, 4u> value_chunk{};
        std::array<std::uint32_t, 256u> expected_values{};
        for (std::uint32_t batch = 0u; batch < 32u * 16u; ++batch)
        {
            for (std::uint32_t i = 0u; i < 4u; ++i)
            {
                const std::uint32_t ordinal = batch * 4u + i;
                key_chunk[i] = ordinal % expected_values.size();
                value_chunk[i] = 0x28000000u + ordinal;
                expected_values[key_chunk[i]] = value_chunk[i];
            }
            rollover_keys.upload(key_chunk.data(), key_chunk.size());
            rollover_values.upload(value_chunk.data(), value_chunk.size());
            rollover_index.insert(
                {rollover_keys.ptr(), rollover_values.ptr(), 4u}, 0);
        }
        PAPER_CUDA(cudaDeviceSynchronize());
        std::array<std::uint32_t, 256u> query_keys{};
        for (std::uint32_t i = 0u; i < query_keys.size(); ++i)
            query_keys[i] = i;
        rollover_keys.upload(query_keys.data(), query_keys.size());
        rollover_index.lookup(
            {rollover_keys.ptr(), query_keys.size(),
             rollover_values.ptr(), nullptr}, 0);
        PAPER_CUDA(cudaDeviceSynchronize());
        const auto actual_values =
            rollover_values.download(query_keys.size());
        std::uint32_t rollover_errors = 0u;
        for (std::uint32_t i = 0u; i < query_keys.size(); ++i)
            rollover_errors += actual_values[i] != expected_values[i];
        const auto rollover_status = rollover_index.canonical_carry_status();
        std::cout << "CANONICAL_MULTILEVEL_ROLLOVER_VALIDATION"
                  << " status=" << rollover_status
                  << " errors=" << rollover_errors << std::endl;
        if (rollover_status || rollover_errors)
            throw std::runtime_error(
                "canonical multi-level rollover validation failed");
    }

    // A one-level dictionary must recycle its top level for partial epochs,
    // then fail cleanly (without corrupting the resident level) once unique
    // survivors actually exceed its configured capacity.
    {
        DictionaryConfig capacity_config;
        capacity_config.max_elements = 1024u;
        capacity_config.batch_capacity = 4u;
        capacity_config.level_zero_capacity = 1024u;
        GPULSMOpt capacity_index(capacity_config);
        cuda_buffer<std::uint32_t> capacity_keys;
        cuda_buffer<std::uint32_t> capacity_values;
        capacity_keys.alloc(4u);
        capacity_values.alloc(4u);
        std::array<std::uint32_t, 4u> key_chunk{};
        std::array<std::uint32_t, 4u> value_chunk{};
        for (std::uint32_t batch = 0u; batch < 17u * 16u; ++batch)
        {
            for (std::uint32_t i = 0u; i < 4u; ++i)
            {
                key_chunk[i] = batch * 4u + i;
                value_chunk[i] = 0x30000000u + key_chunk[i];
            }
            capacity_keys.upload(key_chunk.data(), key_chunk.size());
            capacity_values.upload(value_chunk.data(), value_chunk.size());
            capacity_index.insert(
                {capacity_keys.ptr(), capacity_values.ptr(), 4u}, 0);
        }
        PAPER_CUDA(cudaDeviceSynchronize());
        const auto capacity_status = capacity_index.canonical_carry_status();
        const bool controlled_status =
            (capacity_status &
             gpulsmopt2_detail::kPublicationOutputOverflow) != 0u;
        bool controlled_exception = false;
        try
        {
            capacity_index.insert(
                {capacity_keys.ptr(), capacity_values.ptr(), 4u}, 0);
        }
        catch (const std::runtime_error &error)
        {
            controlled_exception =
                std::string(error.what()).find("capacity") !=
                std::string::npos;
        }
        std::cout << "CANONICAL_CAPACITY_VALIDATION"
                  << " status=" << capacity_status
                  << " controlled_status=" << controlled_status
                  << " controlled_exception=" << controlled_exception
                  << std::endl;
        if (!controlled_status || !controlled_exception)
            throw std::runtime_error(
                "canonical capacity exhaustion was not controlled");
    }

    std::ofstream complete(
        configuration.output_directory / "complete_forced_unified_validation");
    complete << "ok\n";
}

void run_construction_probe(const options &configuration)
{
    using index_type = paper_index_type<batch_log>;
    const std::uint32_t maximum_elements =
        std::uint32_t{1} << configuration.insert_limit_log;
    size_t free_before = 0;
    size_t total_memory = 0;
    PAPER_CUDA(cudaMemGetInfo(&free_before, &total_memory));
    index_type index;
    const double wall_ms = measure_wall_ms([&] {
        index.build(
            nullptr, 0, maximum_elements, free_before,
            nullptr, nullptr);
        PAPER_CUDA(cudaDeviceSynchronize());
    });
    const std::size_t reported_bytes = index.gpu_resident_bytes();
    size_t free_after = 0;
    PAPER_CUDA(cudaMemGetInfo(&free_after, &total_memory));
    std::ofstream output(
        configuration.output_directory / "construction_b20.csv");
    output << "maximum_elements,wall_ms,reported_gpu_resident_bytes,"
              "free_memory_delta_bytes\n";
    output << maximum_elements << ',' << std::setprecision(12) << wall_ms
           << ',' << reported_bytes << ',' << (free_before - free_after)
           << '\n';
    std::cout << "CONSTRUCTION_PROBE"
              << " maximum_elements=" << maximum_elements
              << " wall_ms=" << wall_ms
              << " reported_gpu_resident_bytes=" << reported_bytes
              << " free_memory_delta_bytes=" << (free_before - free_after)
              << std::endl;
}

#endif

template <typename Index>
double run_lookup(
    Index &index,
    cuda_buffer<key_type> &queries,
    cuda_buffer<smallsize> &results,
    cuda_buffer<unsigned long long> &errors,
    std::uint32_t count,
    std::uint32_t resident,
    bool hits,
    std::uint32_t seed,
    gpu_timer &timer)
{
    fill_lookup_queries<<<(count + threads - 1) / threads, threads>>>(
        queries.ptr(), count, resident, hits, seed);
    PAPER_CUDA(cudaGetLastError());
    const double milliseconds = timer.measure([&] {
        index.lookup(queries.ptr(), results.ptr(), count, 0);
    });
    validate_lookup_results<<<
        (count + threads - 1) / threads, threads>>>(
            results.ptr(), count, hits, errors.ptr());
    PAPER_CUDA(cudaGetLastError());
    return milliseconds;
}

struct range_measurement
{
    double count_ms = std::numeric_limits<double>::quiet_NaN();
    double range_ms = 0;
    std::uint64_t checksum_sum = 0;
    std::uint64_t checksum_xor = 0;
};

template <typename Index>
range_measurement run_count_and_range(
    Index &index,
    cuda_buffer<key_type> &lower,
    cuda_buffer<key_type> &upper,
    cuda_buffer<smallsize> &count_results,
    cuda_buffer<smallsize> &range_results,
    cuda_buffer<unsigned long long> &range_digest,
    cuda_buffer<unsigned long long> &errors,
    std::uint32_t total_queries,
    std::uint32_t resident,
    std::uint32_t expected_hits,
    std::uint32_t chunk_size,
    std::uint32_t seed,
    bool include_count,
    gpu_timer &timer)
{
#if defined(PAPER_SWEEP_GPULSMOPT)
    (void)count_results;
    (void)errors;
    (void)include_count;
#endif
    range_measurement measurement;
    range_digest.zero();
    for (std::uint32_t offset = 0; offset < total_queries;)
    {
        const std::uint32_t current =
            std::min(chunk_size, total_queries - offset);
        fill_range_queries<<<
            (current + threads - 1) / threads, threads>>>(
                lower.ptr(), upper.ptr(), current, resident,
                expected_hits, seed + offset);
        PAPER_CUDA(cudaGetLastError());
#if defined(PAPER_SWEEP_GPULSMOPT)
        measurement.range_ms += timer.measure([&] {
            index.range_lookup_sum(
                lower.ptr(), upper.ptr(), range_results.ptr(), current, 0);
        });
#else
        if (include_count)
        {
            if (!std::isfinite(measurement.count_ms))
                measurement.count_ms = 0;
            measurement.count_ms += timer.measure([&] {
                index.count(
                    lower.ptr(), upper.ptr(), count_results.ptr(), current, 0);
            });
        }
        measurement.range_ms += timer.measure([&] {
            index.range_lookup_sum(
                lower.ptr(), upper.ptr(), range_results.ptr(), current, 0);
        });
        if (include_count)
        {
            compare_results<<<
                (current + threads - 1) / threads, threads>>>(
                    count_results.ptr(), range_results.ptr(), current,
                    errors.ptr());
            PAPER_CUDA(cudaGetLastError());
        }
#endif
        digest_range_results<<<
            (current + threads - 1) / threads, threads>>>(
                range_results.ptr(), current, offset,
                range_digest.ptr());
        PAPER_CUDA(cudaGetLastError());
        offset += current;
    }
    const auto digest = range_digest.download(2);
    measurement.checksum_sum = digest[0];
    measurement.checksum_xor = digest[1];
    return measurement;
}

template <unsigned BatchLog>
void run_bulk_build(const options &configuration)
{
    using index_type = paper_index_type<BatchLog>;
    const std::uint32_t element_count =
        std::uint32_t{1} << configuration.insert_limit_log;
    cuda_buffer<key_type> keys;
    keys.alloc(element_count);
    fill_build_keys<<<(element_count + threads - 1) / threads, threads>>>(
        keys.ptr(), element_count);
    PAPER_CUDA(cudaGetLastError());
    PAPER_CUDA(cudaDeviceSynchronize());

    size_t free_memory = 0;
    size_t total_memory = 0;
    PAPER_CUDA(cudaMemGetInfo(&free_memory, &total_memory));
    index_type index;
    double gpu_time_ms = 0;
    size_t resident_bytes = 0;
    const double wall_time_ms = measure_wall_ms([&] {
        index.build(
            keys.ptr(), element_count, element_count, free_memory,
            &gpu_time_ms, &resident_bytes);
    });
    PAPER_CUDA(cudaDeviceSynchronize());

    const std::uint32_t validation_count =
        std::min<std::uint32_t>(element_count, std::uint32_t{1} << 18);
    cuda_buffer<key_type> queries;
    cuda_buffer<smallsize> results;
    cuda_buffer<unsigned long long> errors;
    queries.alloc(validation_count);
    results.alloc(validation_count);
    errors.alloc(1);
    errors.zero();
    for (const bool hits : {true, false})
    {
        fill_lookup_queries<<<
            (validation_count + threads - 1) / threads, threads>>>(
                queries.ptr(), validation_count, element_count, hits,
                hits ? 0x71000u : 0x72000u);
        index.lookup(queries.ptr(), results.ptr(), validation_count, 0);
        validate_lookup_presence<<<
            (validation_count + threads - 1) / threads, threads>>>(
                results.ptr(), validation_count, hits, errors.ptr());
    }
    PAPER_CUDA(cudaDeviceSynchronize());
    const unsigned long long error_count = errors.download_first_item();
    if (error_count != 0)
        throw std::runtime_error(
            "bulk-build lookup validation failures: " +
            std::to_string(error_count));

    const auto output_path = configuration.output_directory / "bulk_build.csv";
    std::ofstream output(output_path);
    output << "system,batch_log,elements,gpu_time_ms,wall_time_ms,"
              "rate_mops,gpu_resident_bytes\n";
    output << std::setprecision(12)
           << index_name << ',' << BatchLog << ',' << element_count << ','
           << gpu_time_ms << ',' << wall_time_ms << ','
           << element_count / gpu_time_ms / 1000.0 << ','
           << resident_bytes << '\n';
    output.flush();
    std::ofstream complete(
        configuration.output_directory / "complete_bulk");
    complete << "ok\n";
}

template <unsigned BatchLog>
void run_main_sweep(const options &configuration)
{
    using index_type = paper_index_type<BatchLog>;
    constexpr std::uint32_t batch_size = std::uint32_t{1} << BatchLog;
    if (configuration.insert_limit_log < BatchLog)
        return;

    const std::uint32_t maximum_elements =
        std::uint32_t{1} << configuration.insert_limit_log;
    const std::uint32_t paper_insertion_count =
        maximum_elements / batch_size;
    const std::uint32_t insertion_count = configuration.stop_after_r == 0
        ? paper_insertion_count
        : std::min(paper_insertion_count, configuration.stop_after_r);
    const bool run_lookups = BatchLog >= 16 && BatchLog <= 24 &&
        configuration.query_limit_log >= BatchLog;
    const bool run_ranges = BatchLog >= 16 && BatchLog <= 20 &&
        configuration.query_limit_log >= BatchLog;
    const std::uint32_t query_maximum = run_lookups
        ? std::uint32_t{1} << configuration.query_limit_log
        : 0;
    const std::uint32_t range_chunk =
        std::uint32_t{1} << configuration.range_chunk_log;

    std::filesystem::create_directories(configuration.output_directory);
    const std::string suffix = "_b" + std::to_string(BatchLog) + ".csv";
    std::ofstream insertion_file(
        configuration.output_directory / ("insertion" + suffix));
    std::ofstream lookup_file(
        configuration.output_directory / ("lookup" + suffix));
    const std::string range_prefix =
        configuration.include_count ? "count_range" : "range";
    std::ofstream range_file(
        configuration.output_directory / (range_prefix + suffix));
    insertion_file << "system,batch_log,r,resident_elements,time_ms,rate_mops,"
                      "cumulative_ms,effective_rate_mops\n";
    lookup_file << "system,batch_log,r,resident_elements,scenario,time_ms,"
                   "rate_mops\n";
    range_file << "system,batch_log,r,resident_elements,operation,expected_hits,"
                  "time_ms,rate_mops,chunk_size,checksum_sum,checksum_xor\n";
    insertion_file << std::setprecision(12);
    lookup_file << std::setprecision(12);
    range_file << std::setprecision(12);

    cuda_buffer<key_type> batch_keys;
    cuda_buffer<smallsize> batch_values;
    batch_keys.alloc(batch_size);
    batch_values.alloc(batch_size);

    // Complete-loop profiling pre-generates every input record before the
    // profiler range.  This keeps input-generation kernels outside the
    // insertion capture while preserving the normal one-batch path otherwise.
    cuda_buffer<key_type> profile_all_keys;
    cuda_buffer<smallsize> profile_all_values;
    if (configuration.profile_all_inserts)
    {
        profile_all_keys.alloc(maximum_elements);
        profile_all_values.alloc(maximum_elements);
        fill_insert_batch<<<
            (maximum_elements + threads - 1) / threads, threads>>>(
            profile_all_keys.ptr(), profile_all_values.ptr(), 0,
            maximum_elements);
        PAPER_CUDA(cudaGetLastError());
        PAPER_CUDA(cudaDeviceSynchronize());
    }

    cuda_buffer<key_type> lookup_queries;
    cuda_buffer<smallsize> lookup_results;
    const std::uint32_t validation_count =
        std::min(maximum_elements, std::uint32_t{1} << 18);
    const std::uint32_t lookup_capacity =
        std::max(query_maximum, validation_count);
    lookup_queries.alloc(lookup_capacity);
    lookup_results.alloc(lookup_capacity);

    cuda_buffer<key_type> lower;
    cuda_buffer<key_type> upper;
    cuda_buffer<smallsize> count_results;
    cuda_buffer<smallsize> range_results;
    cuda_buffer<unsigned long long> range_digest;
    if (run_ranges)
    {
        lower.alloc(range_chunk);
        upper.alloc(range_chunk);
        if (configuration.include_count)
            count_results.alloc(range_chunk);
        range_results.alloc(range_chunk);
        range_digest.alloc(2);
    }

    cuda_buffer<unsigned long long> errors;
    errors.alloc(1);
    errors.zero();

    size_t free_memory = 0;
    size_t total_memory = 0;
    PAPER_CUDA(cudaMemGetInfo(&free_memory, &total_memory));
    index_type index;
    index.build(
        nullptr, 0, maximum_elements, free_memory,
        nullptr, nullptr);

    gpu_timer timer;
    double cumulative_insert_ms = 0;
    const auto wall_start = clock_type::now();
    bool profiler_started = false;
    if (configuration.profile_all_inserts)
    {
        PAPER_CUDA(cudaProfilerStart());
        profiler_started = true;
    }

    for (std::uint32_t r = 1; r <= insertion_count; ++r)
    {
        const std::uint32_t begin = (r - 1) * batch_size;
        key_type *insert_keys = batch_keys.ptr();
        smallsize *insert_values = batch_values.ptr();
        if (configuration.profile_all_inserts)
        {
            insert_keys = profile_all_keys.ptr() + begin;
            insert_values = profile_all_values.ptr() + begin;
        }
        else
        {
            fill_insert_batch<<<
                (batch_size + threads - 1) / threads, threads>>>(
                insert_keys, insert_values, begin, batch_size);
            PAPER_CUDA(cudaGetLastError());
        }
        const bool capture_this_insert =
            configuration.profile_insert_r == r;
        if (capture_this_insert)
        {
            // The fill is deliberately outside a targeted insertion capture.
            PAPER_CUDA(cudaDeviceSynchronize());
            PAPER_CUDA(cudaProfilerStart());
            profiler_started = true;
        }
        const double insert_ms = timer.measure([&] {
            index.insert(
                insert_keys, insert_values, batch_size, 0);
            PAPER_CUDA(cudaDeviceSynchronize());
        });
        if (capture_this_insert)
        {
            PAPER_CUDA(cudaProfilerStop());
            profiler_started = false;
        }
        cumulative_insert_ms += insert_ms;
        const std::uint32_t resident = r * batch_size;
        const double insert_rate = batch_size / insert_ms / 1000.0;
        const double effective_rate =
            resident / cumulative_insert_ms / 1000.0;
        insertion_file << index_name << ',' << BatchLog << ',' << r << ','
                       << resident << ','
                       << insert_ms << ',' << insert_rate << ','
                       << cumulative_insert_ms << ',' << effective_rate << '\n';
        insertion_file.flush();

        if (run_lookups && resident <= query_maximum)
        {
            const double hit_ms = run_lookup(
                index, lookup_queries, lookup_results, errors,
                resident, resident, true, 0x10000u + r, timer);
            const double miss_ms = run_lookup(
                index, lookup_queries, lookup_results, errors,
                resident, resident, false, 0x20000u + r, timer);
            lookup_file << index_name << ',' << BatchLog << ',' << r << ','
                        << resident
                        << ",all_existing," << hit_ms << ','
                        << resident / hit_ms / 1000.0 << '\n';
            lookup_file << index_name << ',' << BatchLog << ',' << r << ','
                        << resident
                        << ",none_existing," << miss_ms << ','
                        << resident / miss_ms / 1000.0 << '\n';
            lookup_file.flush();
        }

        if (run_ranges && resident <= query_maximum)
        {
            for (const std::uint32_t expected_hits : {8u, 1024u})
            {
                const auto times = run_count_and_range(
                    index, lower, upper, count_results, range_results,
                    range_digest, errors, resident, resident, expected_hits,
                    range_chunk, 0x30000u + r + expected_hits,
                    configuration.include_count, timer);
                if (configuration.include_count)
                {
                    range_file << index_name << ',' << BatchLog << ',' << r
                               << ',' << resident << ",count," << expected_hits
                               << ',' << times.count_ms << ','
                               << resident / times.count_ms / 1000.0 << ','
                               << range_chunk << ",0,0\n";
                }
                range_file << index_name << ',' << BatchLog << ',' << r << ','
                           << resident
                           << ",range," << expected_hits << ','
                           << times.range_ms << ','
                           << resident / times.range_ms / 1000.0 << ','
                           << range_chunk << ',' << times.checksum_sum << ','
                           << times.checksum_xor << '\n';
                range_file.flush();
            }
        }

        if (r == 1 || r == insertion_count || (r & (r - 1)) == 0)
        {
            const double wall_seconds =
                std::chrono::duration<double>(clock_type::now() - wall_start)
                    .count();
            std::cout << "PROGRESS batch_log=" << BatchLog
                      << " r=" << r << '/' << insertion_count
                      << " resident=" << resident
                      << " wall_s=" << wall_seconds << std::endl;
        }
    }

    if (profiler_started)
    {
        PAPER_CUDA(cudaProfilerStop());
        profiler_started = false;
    }

    const std::uint32_t final_resident = insertion_count * batch_size;
    run_lookup(
        index, lookup_queries, lookup_results, errors,
        validation_count, final_resident, true, 0x61000u, timer);
    run_lookup(
        index, lookup_queries, lookup_results, errors,
        validation_count, final_resident, false, 0x62000u, timer);
    PAPER_CUDA(cudaDeviceSynchronize());
    const unsigned long long error_count = errors.download_first_item();
    if (error_count != 0)
    {
        std::cerr << "PAPER_VALIDATION_FAILURE count=" << error_count
                  << std::endl;
        throw std::runtime_error(
            "paper sweep validation failures: " +
            std::to_string(error_count));
    }
    const std::string marker_prefix = configuration.include_count
        ? ""
        : "range_";
    const std::string marker = insertion_count == paper_insertion_count
        ? "complete_" + marker_prefix + "b" + std::to_string(BatchLog)
        : "partial_" + marker_prefix + "b" + std::to_string(BatchLog) + "_r" +
              std::to_string(insertion_count);
    std::ofstream complete(configuration.output_directory / marker);
    complete << "ok\n";
}

#if !defined(PAPER_SWEEP_GPULSMOPT)

template <unsigned BatchLog>
void run_cleanup_case(
    const options &configuration,
    std::uint32_t resident_batches,
    unsigned stale_percent,
    bool query_benefit)
{
    using index_type = lsm_tree_ashkiani<key_type, BatchLog>;
    constexpr std::uint32_t batch_size = std::uint32_t{1} << BatchLog;
    const std::uint32_t resident = resident_batches * batch_size;
    const std::uint32_t stale =
        static_cast<std::uint32_t>(
            std::uint64_t{resident} * stale_percent / 100);
    const std::uint32_t unique = resident - stale;

    cuda_buffer<key_type> keys;
    cuda_buffer<smallsize> values;
    keys.alloc(batch_size);
    values.alloc(batch_size);
    size_t free_memory = 0;
    size_t total_memory = 0;
    PAPER_CUDA(cudaMemGetInfo(&free_memory, &total_memory));
    index_type index;
    index.build(nullptr, 0, resident, free_memory, nullptr, nullptr);
    for (std::uint32_t r = 0; r < resident_batches; ++r)
    {
        fill_cleanup_batch<<<
            (batch_size + threads - 1) / threads, threads>>>(
                keys.ptr(), values.ptr(), r * batch_size,
                unique, batch_size);
        PAPER_CUDA(cudaGetLastError());
        index.insert(keys.ptr(), values.ptr(), batch_size, 0);
    }
    PAPER_CUDA(cudaDeviceSynchronize());

    double before_lookup_ms = 0;
    double after_lookup_ms = 0;
    constexpr std::uint32_t benefit_queries = std::uint32_t{1} << 25;
    cuda_buffer<key_type> query_keys;
    cuda_buffer<smallsize> query_results;
    cuda_buffer<unsigned long long> errors;
    errors.alloc(1);
    errors.zero();
    gpu_timer timer;
    if (query_benefit)
    {
        query_keys.alloc(benefit_queries);
        query_results.alloc(benefit_queries);
        before_lookup_ms = run_lookup(
            index, query_keys, query_results, errors,
            benefit_queries, unique, true, 0x50000u, timer);
    }

    size_t survivors = 0;
    const double cleanup_wall_ms = measure_wall_ms([&] {
        survivors = index.cleanup(0);
    });
    PAPER_CUDA(cudaDeviceSynchronize());
    if (query_benefit)
    {
        after_lookup_ms = run_lookup(
            index, query_keys, query_results, errors,
            benefit_queries, unique, true, 0x50000u, timer);
    }
    PAPER_CUDA(cudaDeviceSynchronize());
    const unsigned long long error_count = errors.download_first_item();
    if (error_count != 0)
        throw std::runtime_error("cleanup lookup validation failed");

    const auto output_path = configuration.output_directory / "cleanup.csv";
    const bool write_header = !std::filesystem::exists(output_path);
    std::ofstream output(output_path, std::ios::app);
    if (write_header)
        output << "system,batch_log,resident_batches,resident_elements,"
                  "stale_percent,survivors,cleanup_wall_ms,"
                  "cleanup_rate_mops,query_count,before_lookup_ms,"
                  "after_lookup_ms,speedup,amortized_speedup\n";
    const double rate = resident / cleanup_wall_ms / 1000.0;
    const double speedup = query_benefit
        ? before_lookup_ms / after_lookup_ms
        : 0;
    const double amortized = query_benefit
        ? before_lookup_ms / (cleanup_wall_ms + after_lookup_ms)
        : 0;
    output << std::setprecision(12)
           << index_name << ',' << BatchLog << ',' << resident_batches << ','
           << resident << ','
           << stale_percent << ',' << survivors << ',' << cleanup_wall_ms
           << ',' << rate << ',' << (query_benefit ? benefit_queries : 0)
           << ',' << before_lookup_ms << ',' << after_lookup_ms << ','
           << speedup << ',' << amortized << '\n';
    output.flush();
    std::cout << "CLEANUP batch_log=" << BatchLog
              << " batches=" << resident_batches
              << " stale=" << stale_percent
              << " wall_ms=" << cleanup_wall_ms << std::endl;
}

void run_cleanup_sweep(const options &configuration)
{
    if constexpr (batch_log == 20)
    {
        run_cleanup_case<20>(configuration, 63, 10, false);
        run_cleanup_case<20>(configuration, 63, 50, false);
    }
    if constexpr (batch_log == 19)
    {
        run_cleanup_case<19>(configuration, 127, 10, false);
        run_cleanup_case<19>(configuration, 127, 50, false);
    }
    if constexpr (batch_log == 18)
        run_cleanup_case<18>(configuration, 127, 10, true);
}

#else

void run_cleanup_sweep(const options &)
{
    throw std::invalid_argument(
        "GPULSMOpt has no paper-equivalent explicit cleanup operation");
}

#endif

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const options configuration = parse_options(argc, argv);
        PAPER_CUDA(cudaSetDevice(0));
        PAPER_CUDA(cudaFree(0));
        write_metadata(configuration);
        if (configuration.bulk_sweep)
            run_bulk_build<batch_log>(configuration);
        if (configuration.main_sweep)
            run_main_sweep<batch_log>(configuration);
        if (configuration.cleanup_sweep)
            run_cleanup_sweep(configuration);
#if defined(PAPER_SWEEP_GPULSMOPT)
        if (configuration.forced_unified_validation)
            run_forced_unified_validation(configuration);
        if (configuration.construction_only)
            run_construction_probe(configuration);
#endif
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
