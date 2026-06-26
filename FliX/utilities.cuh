// =============================================================================
// File: utilities.cuh
// Author: Justus Henneberg
// Description: Implements utilities     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef UTILITIES_CUH
#define UTILITIES_CUH

#include <algorithm>
#include <chrono>
#include <random>
#include <unordered_set>
#include <optional>

#include <cub/cub.cuh>

#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include "cuda_buffer.cuh"
#include "memory_layout.cuh"

namespace mem = memory_layout;

// #define UNSORTED_PROBES_CHECKS
// #define BASELINES
#define CUDA_CHECK(x)                                                                      \
    do                                                                                     \
    {                                                                                      \
        cudaError_t err = (x);                                                             \
        if (err != cudaSuccess)                                                            \
        {                                                                                  \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            asm("trap;");                                                                  \
        }                                                                                  \
    } while (0)

#define CUB_CHECK(x) CUDA_CHECK(x)

inline uint32_t get_max_optin_shmem_per_block_bytes()
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.sharedMemPerBlockOptin;
}

inline uint32_t get_max_shmem_per_block_bytes()
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.sharedMemPerBlock;
}

inline uint32_t interpolate_shmem_bytes(size_t shared_memory_occupancy)
{
    // 100% occupancy means 100% of the base shared memory (default)
    // 200% occupancy means 100% of the base and 100% of the opt-in shared memory (absolute maximum)
    // this is non-linear because usually, opt-in < 2 * base
    if (shared_memory_occupancy > 200)
        throw std::runtime_error("shared_memory_occupancy must be <= 200");
    uint32_t base_percentage = shared_memory_occupancy < 100 ? shared_memory_occupancy : 100;
    uint32_t base_bytes = get_max_shmem_per_block_bytes();
    uint32_t optin_percentage = shared_memory_occupancy > 100 ? shared_memory_occupancy - 100 : 0;
    uint32_t optin_bytes = get_max_optin_shmem_per_block_bytes();
    if (optin_bytes < base_bytes)
        throw std::runtime_error("opt-in shared memory must be >= base shared memory");
    return base_bytes * base_percentage / 100 + (optin_bytes - base_bytes) * optin_percentage / 100;
}

inline uint32_t get_number_of_sms()
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.multiProcessorCount;
}

inline smallsize ilog2_cpu(smallsize value)
{
    return value == 0 ? 0 : 31u - __builtin_clz(value);
}

using parameters_type = std::vector<std::pair<std::string, std::string>>;

std::string parameters_to_string(const parameters_type &params)
{
    std::string result;
    bool first = true;
    for (const auto &param : params)
    {
        if (first)
        {
            first = false;
        }
        else
        {
            result += "&";
        }
        result += param.first + "=" + param.second;
    }
    return result;
}

std::string parameters_to_name(const parameters_type &params)
{
    std::string result;
    bool first = true;
    for (const auto &param : params)
    {
        if (first)
        {
            first = false;
        }
        else
        {
            result += "_";
        }
        result += param.second;
    }
    return result;
}

template <typename key_type>
size_t footprint(size_t element_count)
{
    return element_count * sizeof(key_type);
}

class cuda_timer final
{
    cudaEvent_t timerstart, timerstop;
    cudaStream_t stream;

public:
    cuda_timer(cudaStream_t stream) : stream(stream)
    {
        cudaEventCreate(&timerstart);
        cudaEventCreate(&timerstop);
    }
    ~cuda_timer()
    {
        cudaEventDestroy(timerstart);
        cudaEventDestroy(timerstop);
    }
    void start()
    {
        cudaEventRecord(timerstart, stream);
    }
    void stop()
    {
        cudaEventRecord(timerstop, stream);
    }
    float time_ms()
    {
        float timerdelta;
        cudaEventSynchronize(timerstop);
        cudaEventElapsedTime(&timerdelta, timerstart, timerstop);
        return timerdelta;
    }
};

class scoped_cuda_timer final
{
    cuda_timer timer;
    double *result_location;

public:
    scoped_cuda_timer(cudaStream_t stream, double *result_location) : timer(stream), result_location(result_location)
    {
        if (result_location)
        {
            timer.start();
        }
    }
    ~scoped_cuda_timer()
    {
        if (result_location)
        {
            timer.stop();
            *result_location += timer.time_ms();
        }
    }
};

void rti_assert(bool predicate, const std::string &desc = {})
{
    if (!predicate)
        throw std::runtime_error(desc);
}

template <typename value_type>
void show_vector(const std::vector<value_type> &local_buffer, size_t max_output = std::numeric_limits<size_t>::max())
{
    for (size_t i = 0; i < std::min(max_output, local_buffer.size()); ++i)
    {
        std::cout << local_buffer[i] << "  ";
    }
    std::cout << std::endl;
}

template <typename key_type>
bool is_matching_result(const std::vector<key_type> &lower, const std::vector<key_type> &upper, const std::vector<smallsize> &expected, cuda_buffer<smallsize> &result_buffer, std::string &error_message)
{
    std::vector<smallsize> test_output(expected.size());
    result_buffer.download(test_output.data(), test_output.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        if (expected[i] != test_output[i])
        {
            if (lower[i] == upper[i])
            {
                error_message = "data mismatch at index " + std::to_string(i) + " for single-element lookup " + std::to_string(lower[i]) + ": expected " + std::to_string(expected[i]) + ", but received " + std::to_string(test_output[i]);
            }
            else
            {
                error_message = "data mismatch at index " + std::to_string(i) + " for range lookup [" + std::to_string(lower[i]) + ", " + std::to_string(upper[i]) + "]: expected " + std::to_string(expected[i]) + ", but received " + std::to_string(test_output[i]);
            }
            return false;
        }
    }
    return true;
}

template <typename key_type>
void check_result(const std::vector<key_type> &lower, const std::vector<key_type> &upper, const std::vector<smallsize> &expected, cuda_buffer<smallsize> &result_buffer)
{
    std::string error_message;
    auto ok = is_matching_result(lower, upper, expected, result_buffer, error_message);
    if (!ok)
    {
        throw std::logic_error("check_result failed: " + error_message);
    }
}

template <typename key_type>
size_t find_sort_buffer_size(size_t input_size)
{
    size_t temp_bytes_required = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes_required, (key_type *)nullptr, (key_type *)nullptr, input_size, 0, sizeof(key_type) * 8);
    return temp_bytes_required;
}

template <typename key_type, typename value_type>
size_t find_pair_sort_buffer_size(size_t input_size)
{
    size_t temp_bytes_required = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes_required, (key_type *)nullptr, (key_type *)nullptr, (value_type *)nullptr, (value_type *)nullptr, input_size, 0, sizeof(key_type) * 8);
    return temp_bytes_required;
}

template <typename key_type>
void untimed_sort(void *temp, size_t temp_bytes, const key_type *input, key_type *output, size_t input_size, cudaStream_t stream)
{
    // future work: use a different algo for small input buffers
    // maybe check out cub::BlockRadixSort or cub::StableOddEvenSort
    cub::DeviceRadixSort::SortKeys(temp, temp_bytes, input, output, input_size, 0, sizeof(key_type) * 8, stream);
}

template <typename key_type>
void timed_sort(void *temp, size_t temp_bytes, const key_type *input, key_type *output, size_t input_size, double *time_ms, cudaStream_t stream = 0)
{
    scoped_cuda_timer timer(stream, time_ms);
    untimed_sort(temp, temp_bytes, input, output, input_size, stream);
}

template <typename key_type, typename value_type>
void untimed_pair_sort(void *temp, size_t temp_bytes, const key_type *ki, key_type *ko, const value_type *vi, value_type *vo, size_t input_size, cudaStream_t stream)
{
    cub::DeviceRadixSort::SortPairs(temp, temp_bytes, ki, ko, vi, vo, input_size, 0, sizeof(key_type) * 8, stream);
}

template <typename key_type, typename value_type>
void timed_pair_sort(void *temp, size_t temp_bytes, const key_type *ki, key_type *ko, const value_type *vi, value_type *vo, size_t input_size, double *time_ms, cudaStream_t stream = 0)
{
    scoped_cuda_timer timer(stream, time_ms);
    untimed_pair_sort(temp, temp_bytes, ki, ko, vi, vo, input_size, stream);
}

GLOBALQUALIFIER
void init_offsets_kernel(smallsize *buffer, smallsize size, smallsize first_offset)
{
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= size)
        return;
    buffer[tid] = static_cast<smallsize>(tid) + first_offset;
}

void init_offsets(smallsize *buffer, size_t size, size_t first_offset, double *time_ms)
{
    scoped_cuda_timer timer(0, time_ms);
    init_offsets_kernel<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(buffer, size, first_offset);
}

void init_offsets(smallsize *buffer, size_t size, double *time_ms)
{
    init_offsets(buffer, size, 0, time_ms);
}

template <typename key_type>
GLOBALQUALIFIER void transform_into_row_layout_kernel(const key_type *keys, const smallsize *offsets, void *row_buffer, smallsize key_count)
{
    static_assert(sizeof(key_type) == 4 || sizeof(key_type) == 8, "key_type must be 4 or 8 bytes");

    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= key_count)
        return;

    smallsize byte_offset = tid * (sizeof(key_type) + sizeof(smallsize));
    mem::set(row_buffer, byte_offset, keys[tid]);
    mem::set(row_buffer, byte_offset + sizeof(key_type), offsets[tid]);
}

template <typename key_type>
void transform_into_row_layout(
    const key_type *keys,
    const smallsize *offsets,
    void *row_buffer,
    size_t key_count,
    double *time_ms)
{
    scoped_cuda_timer timer(0, time_ms);
    transform_into_row_layout_kernel<<<SDIV(key_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(keys, offsets, row_buffer, key_count);
    cudaStreamSynchronize(0);
    C2EX
}

template <uint16_t base, typename integer_type>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
    integer_type
    ilog(integer_type n)
{
    integer_type result = 0;
    integer_type agg = 1;
    while (agg < n)
    {
        agg *= base;
        ++result;
    }
    return result;
}

template <uint16_t base, typename integer_type>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
    integer_type
    ipow(integer_type n)
{
    integer_type result = 1;
    for (integer_type i = 0; i < n; ++i)
    {
        result *= base;
    }
    return result;
}

// -------------------------
// Stream-aware init_offsets
// -------------------------
GLOBALQUALIFIER
void init_offsets_kernel_u32_2(uint32_t *buffer, size_t size, uint32_t first_offset)
{
    const size_t tid = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (tid >= size)
        return;
    buffer[tid] = static_cast<uint32_t>(tid) + first_offset;
}

inline void init_offsets_u32_2(uint32_t *buffer, size_t size, uint32_t first_offset, cudaStream_t stream)
{
    if (size == 0)
        return;
    constexpr int TPB = 256;
    const int GRD = static_cast<int>((size + TPB - 1) / TPB);
    init_offsets_kernel_u32_2<<<GRD, TPB, 0, stream>>>(buffer, size, first_offset);
}

inline void init_offsets_u32_2(uint32_t *buffer, size_t size, cudaStream_t stream)
{
    init_offsets_u32_2(buffer, size, 0u, stream);
}

// ------------------------------------------
// Buffer-size helpers (as you already have)
// ------------------------------------------
template <typename key_type>
size_t find_sort_buffer_size2(size_t input_size)
{
    size_t temp_bytes_required = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, temp_bytes_required,
        (key_type *)nullptr, (key_type *)nullptr,
        input_size, 0, int(sizeof(key_type) * 8));
    return temp_bytes_required;
}

template <typename key_type, typename value_type>
size_t find_pair_sort_buffer_size2(size_t input_size)
{
    size_t temp_bytes_required = 0;
    auto st = cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes_required,
       (key_type*)nullptr, (key_type*)nullptr,
       (value_type*)nullptr, (value_type*)nullptr,
       input_size, 0, int(sizeof(key_type) * 8));
    //return temp_bytes_required;

   // size_t temp_bytes_required = 0;
    /* auto st = cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes_required,
        (key_type *)nullptr, (key_type *)nullptr,
        (value_type *)nullptr, (value_type *)nullptr,
        static_cast<int>(input_size),
        0);  */

    if (st != cudaSuccess)
    {
        printf("Temp-size query failed: %s\n", cudaGetErrorString(st));
       // asm("trap;");
    }
    return temp_bytes_required;
}

template <typename key_type, typename value_type>
size_t find_pair_sort_buffer_size3(size_t input_size, cudaStream_t stream = 0)
{
    size_t temp_bytes_required = 0;

    const int num_items = static_cast<int>(input_size);
    const int begin_bit = 0;
    const int end_bit   = 32; //int(sizeof(key_type) * 8);

    cudaError_t st = cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes_required,
        (key_type*)nullptr, (key_type*)nullptr,
        (value_type*)nullptr, (value_type*)nullptr,
        num_items,
        begin_bit, end_bit,
        stream);

    if (st != cudaSuccess)
        printf("Temp-size query failed (n=%zu -> %d): %s\n",
               input_size, num_items, cudaGetErrorString(st));

    return temp_bytes_required;
}

// ------------------------------------------
// Correct stream-aware timed_pair_sort
// ------------------------------------------
template <typename key_type, typename value_type>
void untimed_pair_sort2(void *temp,
                        size_t temp_bytes,
                        const key_type *ki,
                        key_type *ko,
                        const value_type *vi,
                        value_type *vo,
                        size_t input_size,
                        cudaStream_t stream)
{
    // CUB radix sort supports arbitrary sizes; no power-of-2 requirement.

    printf("Calling cub::DeviceRadixSort::SortPairs with input_size=%zu\n", input_size);
     auto st = cub::DeviceRadixSort::SortPairs(
      temp, temp_bytes,
       ki, ko,
      vi, vo,
        input_size, 0, int(sizeof(key_type) * 8),
        stream);

    /* auto st = cub::DeviceRadixSort::SortPairs(
        temp, temp_bytes,
        ki, ko,
        vi, vo,
        static_cast<int>(input_size),
        stream);  */

    if (st != cudaSuccess)
    {
        printf("SortPairs failed: %s\n", cudaGetErrorString(st));
       // asm("trap;");
    }

    auto st2 = cudaGetLastError();
    if (st2 != cudaSuccess)
    {
        printf("CUDA launch error after SortPairs: %s\n", cudaGetErrorString(st2));
       // asm("trap;");
    }
}

template <typename key_type, typename value_type>
void timed_pair_sort2(void *temp,
                      size_t temp_bytes,
                      const key_type *ki,
                      key_type *ko,
                      const value_type *vi,
                      value_type *vo,
                      size_t input_size,
                      double *time_ms,
                      cudaStream_t stream)
{
    scoped_cuda_timer timer(stream, time_ms);
    untimed_pair_sort2(temp, temp_bytes, ki, ko, vi, vo, input_size, stream);
}



#endif
