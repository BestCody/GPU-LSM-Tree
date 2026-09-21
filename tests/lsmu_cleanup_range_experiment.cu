#include "impl_lsm_tree.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using index_type = lsm_tree_ashkiani<key32, 20>;
using clock_type = std::chrono::steady_clock;

constexpr std::uint32_t initial_log = 27;
constexpr std::uint32_t batch_log = 20;
constexpr std::uint32_t query_limit_log = 24;
constexpr std::uint32_t batch_size = std::uint32_t{1} << batch_log;
constexpr std::uint32_t initial_size = std::uint32_t{1} << initial_log;
constexpr std::uint32_t maximum_query_size =
    std::uint32_t{1} << query_limit_log;
constexpr std::uint32_t minimum_query_size = batch_size;
constexpr std::uint64_t key_domain = std::uint64_t{1} << 30;
constexpr key32 key_mask = static_cast<key32>(key_domain - 1);
constexpr std::uint32_t threads = 256;
constexpr std::uint32_t validation_queries = 4096;

struct timing
{
    double gpu_ms = 0;
    double wall_ms = 0;
};

void checked(cudaError_t error, const char *where)
{
    if (error != cudaSuccess)
        throw std::runtime_error(
            std::string(where) + ": " + cudaGetErrorString(error));
}

__host__ __device__ key32 key_for_index(std::uint32_t index)
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

__global__ void fill_keys(key32 *keys, std::uint32_t begin,
                          std::uint32_t size)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        keys[index] = key_for_index(begin + index);
}

__global__ void fill_ranges(key32 *lower, key32 *upper, std::uint32_t size,
                            std::uint32_t live, std::uint32_t expected,
                            std::uint32_t seed)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;
    std::uint64_t width =
        (std::uint64_t{expected} * key_domain + live - 1) / live;
    width = width == 0 ? 1 : width;
    width = width > key_domain ? key_domain : width;
    const std::uint64_t starts = key_domain - width + 1;
    const std::uint64_t first = mix32(index + seed) % starts;
    lower[index] = static_cast<key32>(first);
    upper[index] = static_cast<key32>(first + width - 1);
}

template <typename Function>
timing measure(Function &&function)
{
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    checked(cudaEventCreate(&begin), "create begin event");
    checked(cudaEventCreate(&end), "create end event");
    const auto wall_begin = clock_type::now();
    checked(cudaEventRecord(begin), "record begin event");
    function();
    checked(cudaEventRecord(end), "record end event");
    checked(cudaEventSynchronize(end), "finish measured operation");
    const auto wall_end = clock_type::now();
    float gpu_ms = 0;
    checked(cudaEventElapsedTime(&gpu_ms, begin, end), "read GPU time");
    checked(cudaEventDestroy(begin), "destroy begin event");
    checked(cudaEventDestroy(end), "destroy end event");
    return {gpu_ms,
            std::chrono::duration<double, std::milli>(wall_end - wall_begin)
                .count()};
}

std::vector<smallsize> run_validation_ranges(
    index_type &index, cuda_buffer<key32> &lower,
    cuda_buffer<key32> &upper, cuda_buffer<smallsize> &answers)
{
    index.range_lookup_sum(lower.ptr(), upper.ptr(), answers.ptr(),
                           validation_queries, 0);
    checked(cudaStreamSynchronize(0), "finish validation range sum");
    return answers.download(validation_queries);
}

void compare_answers(const std::vector<smallsize> &before,
                     const std::vector<smallsize> &after)
{
    for (size_t index = 0; index < before.size(); ++index)
        if (before[index] != after[index])
            throw std::runtime_error(
                "cleanup changed range sum at query " +
                std::to_string(index) + ": before=" +
                std::to_string(before[index]) + ", after=" +
                std::to_string(after[index]));
}

void delete_batch(index_type &index, cuda_buffer<key32> &keys,
                  std::uint32_t begin)
{
    fill_keys<<<(batch_size + threads - 1) / threads, threads>>>(
        keys.ptr(), begin, batch_size);
    checked(cudaGetLastError(), "generate deletion batch");
    index.remove(keys.ptr(), batch_size, 0);
}

void run(std::uint32_t repetition)
{
    const size_t maximum_physical =
        size_t{2} * initial_size - maximum_query_size;

    cuda_buffer<key32> initial_keys;
    initial_keys.alloc(initial_size);
    fill_keys<<<(initial_size + threads - 1) / threads, threads>>>(
        initial_keys.ptr(), 0, initial_size);
    checked(cudaGetLastError(), "generate initial keys");
    checked(cudaStreamSynchronize(0), "finish initial key generation");

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    checked(cudaMemGetInfo(&free_bytes, &total_bytes), "read GPU memory");
    index_type index;
    index.build(initial_keys.ptr(), initial_size, maximum_physical, free_bytes,
                nullptr, nullptr);
    initial_keys.free();

    cuda_buffer<key32> update_keys;
    update_keys.alloc(batch_size);
    std::uint32_t live = initial_size;
    while (live > maximum_query_size)
    {
        live -= batch_size;
        delete_batch(index, update_keys, live);
    }
    checked(cudaStreamSynchronize(0), "finish initial deletion phase");

    cuda_buffer<key32> lower;
    cuda_buffer<key32> upper;
    cuda_buffer<smallsize> answers;
    lower.alloc(maximum_query_size);
    upper.alloc(maximum_query_size);
    answers.alloc(maximum_query_size);

    bool validate_cleanup = true;
    while (true)
    {
        const std::uint32_t state = live / batch_size + 1;
        std::vector<smallsize> before_cleanup;
        if (validate_cleanup)
        {
            fill_ranges<<<
                (validation_queries + threads - 1) / threads, threads>>>(
                    lower.ptr(), upper.ptr(), validation_queries, live, 8,
                    0x30000u + state + 8u);
            checked(cudaGetLastError(), "generate validation ranges");
            before_cleanup =
                run_validation_ranges(index, lower, upper, answers);
        }

        size_t valid_after_cleanup = 0;
        const timing cleanup_time = measure([&] {
            valid_after_cleanup = index.cleanup(0);
        });
        if (valid_after_cleanup != live)
            throw std::runtime_error(
                "cleanup retained " + std::to_string(valid_after_cleanup) +
                " elements, expected " + std::to_string(live));

        if (validate_cleanup)
        {
            const auto after_cleanup =
                run_validation_ranges(index, lower, upper, answers);
            compare_answers(before_cleanup, after_cleanup);
            validate_cleanup = false;
        }

        std::cout << std::setprecision(10)
                  << "RESULT operation=cleanup"
                  << " state=" << state
                  << " live=" << live
                  << " repetition=" << repetition
                  << " valid_after_cleanup=" << valid_after_cleanup
                  << " gpu_ms=" << cleanup_time.gpu_ms
                  << " wall_ms=" << cleanup_time.wall_ms << '\n';

        for (const std::uint32_t expected : {8u, 1024u})
        {
            fill_ranges<<<(live + threads - 1) / threads, threads>>>(
                lower.ptr(), upper.ptr(), live, live, expected,
                0x30000u + state + expected);
            checked(cudaGetLastError(), "generate measured ranges");
            const timing range_time = measure([&] {
                index.range_lookup_sum(
                    lower.ptr(), upper.ptr(), answers.ptr(), live, 0);
            });
            const auto sample = answers.download(
                std::min<std::uint32_t>(live, 1024));
            std::uint64_t checksum = 0;
            for (const smallsize value : sample)
                checksum += value;
            std::cout << std::setprecision(10)
                      << "RESULT operation=range_sum_after_cleanup"
                      << " state=" << state
                      << " live=" << live
                      << " repetition=" << repetition
                      << " expected=" << expected
                      << " queries=" << live
                      << " cleanup_gpu_ms=" << cleanup_time.gpu_ms
                      << " cleanup_wall_ms=" << cleanup_time.wall_ms
                      << " range_gpu_ms=" << range_time.gpu_ms
                      << " range_wall_ms=" << range_time.wall_ms
                      << " reported_gpu_ms="
                      << cleanup_time.gpu_ms + range_time.gpu_ms
                      << " reported_wall_ms="
                      << cleanup_time.wall_ms + range_time.wall_ms
                      << " sample_checksum=" << checksum << '\n';
        }

        std::cout << "PROGRESS state=" << state
                  << " live=" << live
                  << " repetition=" << repetition << '\n';
        if (live == minimum_query_size)
            break;
        live -= batch_size;
        delete_batch(index, update_keys, live);
        checked(cudaStreamSynchronize(0), "finish deletion batch");
    }

    index.destroy();
    std::cout << "PASS repetition=" << repetition << '\n';
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc != 3 || std::string(argv[1]) != "--repetition")
            throw std::invalid_argument(
                "usage: lsmu_cleanup_range_experiment --repetition N");
        checked(cudaSetDevice(0), "select GPU");
        run(static_cast<std::uint32_t>(std::stoul(argv[2])));
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
