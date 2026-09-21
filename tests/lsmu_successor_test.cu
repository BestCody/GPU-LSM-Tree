#include "impl_lsm_tree.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t error, const char *where)
{
    if (error != cudaSuccess)
        throw std::runtime_error(
            std::string(where) + ": " + cudaGetErrorString(error));
}

template <typename T>
void upload(cuda_buffer<T> &buffer, const std::vector<T> &values)
{
    buffer.alloc(values.size());
    if (!values.empty())
        check_cuda(cudaMemcpy(buffer.ptr(), values.data(),
                              values.size() * sizeof(T),
                              cudaMemcpyHostToDevice),
                   "copy input to GPU");
}

template <typename Index>
void insert(Index &index, std::map<key32, smallsize> &visible,
            const std::vector<key32> &keys, smallsize value_base)
{
    std::vector<smallsize> values(keys.size());
    for (size_t i = 0; i < keys.size(); ++i)
        values[i] = value_base + static_cast<smallsize>(i);
    cuda_buffer<key32> device_keys;
    cuda_buffer<smallsize> device_values;
    upload(device_keys, keys);
    upload(device_values, values);
    index.insert(device_keys.ptr(), device_values.ptr(), keys.size(), 0);
    check_cuda(cudaDeviceSynchronize(), "finish insertion");
    for (size_t i = 0; i < keys.size(); ++i)
        visible[keys[i]] = values[i];
}

template <typename Index>
void erase(Index &index, std::map<key32, smallsize> &visible,
           const std::vector<key32> &keys)
{
    cuda_buffer<key32> device_keys;
    upload(device_keys, keys);
    index.remove(device_keys.ptr(), keys.size(), 0);
    check_cuda(cudaDeviceSynchronize(), "finish deletion");
    for (const key32 key : keys)
        visible.erase(key);
}

template <typename Index>
void check_successors(Index &index,
                      const std::map<key32, smallsize> &visible,
                      const std::vector<key32> &queries,
                      const char *phase)
{
    cuda_buffer<key32> device_queries;
    cuda_buffer<key32> device_results;
    upload(device_queries, queries);
    device_results.alloc(queries.size());
    index.lookups_successor(device_queries.ptr(), device_results.ptr(),
                            queries.size(), 0);
    check_cuda(cudaDeviceSynchronize(), "finish successor");
    const auto results = device_results.download(queries.size());
    for (size_t i = 0; i < queries.size(); ++i)
    {
        const auto expected_position = visible.lower_bound(queries[i]);
        const key32 expected = expected_position == visible.end()
                                   ? static_cast<key32>(not_found)
                                   : expected_position->first;
        if (results[i] != expected)
            throw std::runtime_error(
                std::string("successor mismatch in ") + phase +
                " for query " + std::to_string(queries[i]) +
                ": expected " + std::to_string(expected) +
                ", received " + std::to_string(results[i]));
    }
}

void run_correctness()
{
    using index_type = lsm_tree_ashkiani<key32, 8>;
    index_type index;
    constexpr size_t initial_count = 700;
    std::vector<key32> initial(initial_count);
    for (size_t i = 0; i < initial.size(); ++i)
        initial[i] = static_cast<key32>(2 * i);
    cuda_buffer<key32> device_initial;
    upload(device_initial, initial);
    index.build(device_initial.ptr(), initial.size(), 2048,
                std::numeric_limits<size_t>::max(), nullptr, nullptr);

    std::map<key32, smallsize> visible;
    for (size_t i = 0; i < initial.size(); ++i)
        visible[initial[i]] = static_cast<smallsize>(i);

    std::vector<key32> queries(1600);
    std::iota(queries.begin(), queries.end(), key32{0});
    queries.push_back(2000);
    queries.push_back(std::numeric_limits<key32>::max());
    check_successors(index, visible, queries, "initial levels");

    std::vector<key32> inserted(171);
    for (size_t i = 0; i < inserted.size(); ++i)
        inserted[i] = static_cast<key32>(4 * i + 1);
    insert(index, visible, inserted, 10000);

    std::vector<key32> deleted;
    for (key32 key = 0; key < 1200; key += 6)
        deleted.push_back(key);
    erase(index, visible, deleted);
    check_successors(index, visible, queries, "interleaved tombstones");

    insert(index, visible, {0, 6, 12, 18, 1599}, 20000);
    check_successors(index, visible, queries, "reinsertion");
    index.destroy();
    std::cout << "PASS LSMu successor across levels, tombstones, and "
                 "reinsertions\n";
}

void run_focused_timing(unsigned query_log)
{
    using index_type = lsm_tree_ashkiani<key32, 20>;
    constexpr size_t initial_count = size_t{1} << 25;
    constexpr size_t deletion_count = size_t{1} << 20;
    const size_t query_count = size_t{1} << query_log;

    std::vector<key32> initial(initial_count);
    std::iota(initial.begin(), initial.end(), key32{0});
    cuda_buffer<key32> device_initial;
    upload(device_initial, initial);

    index_type index;
    index.build(device_initial.ptr(), initial.size(),
                initial_count + deletion_count,
                std::numeric_limits<size_t>::max(), nullptr, nullptr);

    std::vector<key32> deleted(deletion_count);
    for (size_t i = 0; i < deleted.size(); ++i)
        deleted[i] = static_cast<key32>(32 * i);
    cuda_buffer<key32> device_deleted;
    upload(device_deleted, deleted);
    index.remove(device_deleted.ptr(), deleted.size(), 0);
    check_cuda(cudaDeviceSynchronize(), "prepare tombstone level");

    std::vector<key32> queries(query_count);
    for (size_t i = 0; i < queries.size(); ++i)
        queries[i] = deleted[(i * 40503u) & (deletion_count - 1)];
    cuda_buffer<key32> device_queries;
    cuda_buffer<key32> device_results;
    upload(device_queries, queries);
    device_results.alloc(query_count);

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    check_cuda(cudaEventCreate(&begin), "create timing event");
    check_cuda(cudaEventCreate(&end), "create timing event");
    check_cuda(cudaEventRecord(begin), "start timing");
    index.lookups_successor(device_queries.ptr(), device_results.ptr(),
                            query_count, 0);
    check_cuda(cudaEventRecord(end), "stop timing");
    check_cuda(cudaEventSynchronize(end), "finish focused successor");
    float milliseconds = 0;
    check_cuda(cudaEventElapsedTime(&milliseconds, begin, end),
               "read successor time");
    check_cuda(cudaEventDestroy(begin), "destroy timing event");
    check_cuda(cudaEventDestroy(end), "destroy timing event");

    const auto results = device_results.download(query_count);
    for (size_t i = 0; i < results.size(); ++i)
    {
        const key32 expected = queries[i] + 1;
        if (results[i] != expected)
            throw std::runtime_error(
                "focused successor mismatch for query " +
                std::to_string(queries[i]));
    }
    std::cout << "SUCCESSOR_TARGET initial_log=25 deletion_log=20 query_log="
              << query_log << " time_ms=" << milliseconds
              << " million_queries_per_second="
              << query_count / milliseconds / 1000.0 << '\n';
    index.destroy();
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        check_cuda(cudaSetDevice(0), "select GPU");
        if (argc == 1)
        {
            run_correctness();
        }
        else if (argc == 3 && std::string(argv[1]) == "--focused-query-log")
        {
            const unsigned query_log = static_cast<unsigned>(std::stoul(argv[2]));
            if (query_log > 24)
                throw std::invalid_argument("focused query log exceeds 24");
            run_focused_timing(query_log);
        }
        else
        {
            throw std::invalid_argument(
                "usage: lsmu_successor_test [--focused-query-log LOG]");
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
