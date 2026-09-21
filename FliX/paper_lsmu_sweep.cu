#ifndef PAPER_SWEEP_COMMON
#error "paper_lsmu_sweep.cu only supports the common paper sweep"
#endif

// The selected adapter uses this value when sizing its first level, so the
// default must be visible before the adapter is included.
#ifndef PAPER_LSM_BATCH_LOG
#define PAPER_LSM_BATCH_LOG 16
#endif

#include "utilities.cuh"
#include "paper_backends.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using key_type = std::uint32_t;
using clock_type = std::chrono::steady_clock;

constexpr const char *index_name = paper_backend_name;
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
    return ((index * 747796405u + 289133645u) & key_mask) + 2u;
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
            random % static_cast<std::uint32_t>(key_domain - miss_begin));
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
    unsigned range_chunk_log = 0;
    std::uint32_t stop_after_r = 0;
    bool bulk_sweep = false;
    bool skip_ranges = false;
    bool skip_deletions = false;
};

unsigned parse_unsigned(const char *value, const char *name)
{
    const std::string text(value);
    size_t consumed = 0;
    const unsigned long result = std::stoul(text, &consumed);
    if (consumed != text.size() ||
        result > std::numeric_limits<unsigned>::max())
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
            result.stop_after_r =
                parse_unsigned(require_value(), "stop-after count");
        else if (argument == "--bulk-only")
            result.bulk_sweep = true;
        else if (argument == "--skip-ranges")
            result.skip_ranges = true;
        else if (argument == "--skip-deletions")
            result.skip_deletions = true;
        else
            throw std::invalid_argument("unknown argument: " + argument);
    }
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
    output << "paper=1707.05354v2.pdf\n"
              "protocol=common_initialized_v1\n"
              "lookup_timing=complete_unsorted_v1\n"
              "first_state=bulk_build_not_insertion\n"
              "values=original_insertion_ordinal\n";
    output << "index=" << index_name << '\n';
    output << "gpu=" << properties.name << '\n';
    output << "batch_log=" << batch_log << '\n';
    output << "insert_limit_log=" << configuration.insert_limit_log << '\n';
    output << "query_limit_log=" << configuration.query_limit_log << '\n';
    output << "range_chunk_log=" << configuration.range_chunk_log << '\n';
    output << "range_implementation=enumerate_records_sum_v1\n";
    output << "range_batching="
           << (configuration.range_chunk_log ? "explicit_chunks" : "full_batch")
           << '\n';
}

#include "paper_common_sweep.cuh"

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const options configuration = parse_options(argc, argv);
        PAPER_CUDA(cudaSetDevice(0));
        PAPER_CUDA(cudaFree(0));
        write_metadata(configuration);
        run_common_sweep(configuration);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
