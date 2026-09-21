// Regression for public updates larger than LSMu's internal batch.
// nvcc -std=c++17 -O3 -arch=sm_120 --extended-lambda \
//   --expt-relaxed-constexpr -IFliX tests/lsmu_large_updates.cu \
//   -o /tmp/lsmu_large_updates
#include "impl_lsm_tree.cuh"

#include <iostream>
#include <numeric>
#include <vector>

using index_type = lsm_tree_ashkiani<key32, 16>;
constexpr size_t batch_size = index_type::update_storage_granularity;

void checked(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

template <typename T>
void upload(cuda_buffer<T> &buffer, const std::vector<T> &values,
            cudaStream_t stream)
{
    buffer.alloc(values.size());
    checked(cudaMemcpyAsync(buffer.ptr(), values.data(), values.size() * sizeof(T),
                            cudaMemcpyHostToDevice, stream));
}

void verify(index_type &index, const std::vector<smallsize> &expected,
            cudaStream_t stream, const char *label)
{
    std::vector<key32> queries(expected.size());
    std::iota(queries.rbegin(), queries.rend(), key32{0});
    cuda_buffer<key32> device_queries;
    cuda_buffer<smallsize> device_results;
    upload(device_queries, queries, stream);
    device_results.alloc(queries.size());
    index.lookup(device_queries.ptr(), device_results.ptr(), queries.size(), stream);
    checked(cudaStreamSynchronize(stream));
    const auto answers = device_results.download(queries.size());
    for (size_t i = 0; i < queries.size(); ++i)
        if (answers[i] != expected[queries[i]])
            throw std::runtime_error(std::string(label) + " lookup key=" +
                std::to_string(queries[i]) + " actual=" + std::to_string(answers[i]) +
                " expected=" + std::to_string(expected[queries[i]]));

    std::vector<key32> lower, upper;
    for (size_t i = 0; i < 35; ++i)
    {
        const key32 begin = (i * (expected.size() - 1)) / 34;
        lower.push_back(begin);
        upper.push_back(std::min<size_t>(begin + 73, expected.size() - 1));
    }
    lower.push_back(7);
    upper.push_back(6);
    cuda_buffer<key32> device_lower, device_upper;
    upload(device_lower, lower, stream);
    upload(device_upper, upper, stream);
    for (bool count_only : {true, false})
    {
        if (count_only)
            index.count(device_lower.ptr(), device_upper.ptr(), device_results.ptr(),
                        lower.size(), stream);
        else
            index.range_lookup_sum(device_lower.ptr(), device_upper.ptr(),
                                   device_results.ptr(), lower.size(), stream);
        checked(cudaStreamSynchronize(stream));
        const auto sums = device_results.download(lower.size());
        for (size_t i = 0; i < lower.size(); ++i)
        {
            smallsize value = 0;
            for (key32 key = lower[i]; key <= upper[i]; ++key)
                if (expected[key] != not_found)
                    value += count_only ? 1 : expected[key];
            if (sums[i] != value)
                throw std::runtime_error(std::string(label) +
                    (count_only ? " count" : " range sum"));
        }
    }
    std::cout << "PASS " << label << " keys=" << expected.size() << std::endl;
}

void run_updates(size_t count, cudaStream_t stream)
{
    index_type index;
    const size_t padded = ((count + batch_size - 1) / batch_size) * batch_size;
    index.build(nullptr, 0, 7 * padded, std::numeric_limits<size_t>::max(),
                nullptr, nullptr);
    std::vector<key32> keys(count);
    std::vector<smallsize> values(count);
    std::vector<smallsize> expected(count + 4, not_found);
    std::iota(keys.rbegin(), keys.rend(), key32{2});
    // The later internal batch must win for this repeated key.
    if (count > batch_size)
        keys[batch_size] = keys[0];
    for (size_t i = 0; i < count; ++i)
        values[i] = static_cast<smallsize>(3 * i + 19);
    cuda_buffer<key32> device_keys;
    cuda_buffer<smallsize> device_values;
    upload(device_keys, keys, stream);
    upload(device_values, values, stream);

    index.update(nullptr, nullptr, 0, nullptr, 0, stream);
    index.insert(device_keys.ptr(), device_values.ptr(), count, stream);
    for (size_t i = 0; i < count; ++i)
        expected[keys[i]] = values[i];
    verify(index, expected, stream, "large insert");

    index.remove(device_keys.ptr(), count - 1, stream);
    for (size_t i = 0; i < count - 1; ++i)
        expected[keys[i]] = not_found;
    verify(index, expected, stream, "large delete with surviving tail");

    for (size_t i = 0; i < count; ++i)
        values[i] += 10000000u;
    checked(cudaMemcpyAsync(device_values.ptr(), values.data(),
                            values.size() * sizeof(smallsize),
                            cudaMemcpyHostToDevice, stream));
    index.insert(device_keys.ptr(), device_values.ptr(), count, stream);
    for (size_t i = 0; i < count; ++i)
        expected[keys[i]] = values[i];
    verify(index, expected, stream, "reinsert after tombstones");

    // The mixed batch crosses from insertions to overlapping deletions.
    const size_t deletes = count / 2 + 3;
    index.update(device_keys.ptr(), device_values.ptr(), count,
                 device_keys.ptr(), deletes, stream);
    for (size_t i = 0; i < deletes; ++i)
        expected[keys[i]] = not_found;
    verify(index, expected, stream, "unequal mixed update");

    index.insert_and_remove(device_keys.ptr(), device_values.ptr(), count,
                            device_keys.ptr(), stream);
    std::fill(expected.begin(), expected.end(), not_found);
    verify(index, expected, stream, "equal mixed update");
    index.destroy();
}

void check_capacity(cudaStream_t stream)
{
    index_type index;
    std::vector<key32> initial{1};
    cuda_buffer<key32> device_initial;
    device_initial.alloc_and_upload(initial);
    index.build(device_initial.ptr(), 1, 3 * batch_size,
                std::numeric_limits<size_t>::max(), nullptr, nullptr);
    bool rejected = false;
    try
    {
        // Null pointers are safe only if capacity is checked before launching.
        index.insert(nullptr, nullptr, 2 * batch_size + 1, stream);
    }
    catch (const std::overflow_error &)
    {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("oversized call was not rejected");
    rejected = false;
    try
    {
        index.update(nullptr, nullptr, std::numeric_limits<size_t>::max(),
                     nullptr, 1, stream);
    }
    catch (const std::overflow_error &)
    {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("size overflow was not rejected");
    std::vector<smallsize> expected(64, not_found);
    expected[1] = 0;
    verify(index, expected, stream, "failed update preserves original state");
    index.remove(device_initial.ptr(), 1, stream);
    expected[1] = not_found;
    verify(index, expected, stream, "reuse after rejected update");
    index.destroy();
}

int main()
{
    try
    {
        checked(cudaSetDevice(0));
        cudaStream_t stream;
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (size_t count : {batch_size - 1, batch_size, batch_size + 1,
                             3 * batch_size + 37, size_t{3355443}})
            run_updates(count, stream);
        check_capacity(stream);
        checked(cudaStreamDestroy(stream));
        std::cout << "PASS LSMu large public updates regression" << std::endl;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL " << error.what() << std::endl;
        return 1;
    }
}
