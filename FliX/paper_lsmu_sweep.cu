#include <iostream>

#include "utilities.cuh"

#if defined(PAPER_SWEEP_GPULSMOPT)
#include "impl_gpulsmopt.cuh"
#else
#include "impl_lsm_tree.cuh"
#endif

#include <cuda_runtime.h>

#include <algorithm>
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
        queries[tid] = key_for_index(
            static_cast<std::uint32_t>(paper_insert_limit) +
            random % static_cast<std::uint32_t>(
                key_domain - paper_insert_limit));
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
#if defined(PAPER_SWEEP_GPULSMOPT)
    output << "gpulsmopt_maximum_batch_capacity="
           << gpulsmopt_adapter_detail::batch_capacity() << '\n';
    output << "gpulsmopt_level_zero_capacity="
           << gpulsmopt_adapter_detail::level_zero_capacity() << '\n';
#endif
}

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

    for (std::uint32_t r = 1; r <= insertion_count; ++r)
    {
        const std::uint32_t begin = (r - 1) * batch_size;
        fill_insert_batch<<<
            (batch_size + threads - 1) / threads, threads>>>(
                batch_keys.ptr(), batch_values.ptr(), begin, batch_size);
        PAPER_CUDA(cudaGetLastError());
        const double insert_ms = timer.measure([&] {
            index.insert(
                batch_keys.ptr(), batch_values.ptr(), batch_size, 0);
            PAPER_CUDA(cudaDeviceSynchronize());
        });
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
        throw std::runtime_error(
            "paper sweep validation failures: " +
            std::to_string(error_count));
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
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
