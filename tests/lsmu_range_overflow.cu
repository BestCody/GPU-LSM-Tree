// Standalone CUDA regression; --large additionally materializes query batches
// exceeding INT_MAX (COUNT) and UINT32_MAX (RANGE) candidate records.
// Build from the repository root (adjust -arch for the local GPU):
// nvcc -std=c++17 -O3 -arch=sm_120 --extended-lambda \
//   --expt-relaxed-constexpr -IFliX tests/lsmu_range_overflow.cu \
//   -o /tmp/lsmu_range_overflow
#include "impl_lsm_tree.cuh"

#include <iostream>
#include <map>
#include <numeric>
#include <vector>

using index_type = lsm_tree_ashkiani<key32, 15>;
constexpr size_t batch_size = index_type::update_storage_granularity;

void checked(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

void verify(index_type &index, const std::map<key32, smallsize> &oracle,
            const std::vector<key32> &lower, const std::vector<key32> &upper,
            bool count_only, cudaStream_t stream, const char *label)
{
    cuda_buffer<key32> device_lower, device_upper;
    cuda_buffer<smallsize> device_results;
    device_lower.alloc_and_upload(lower);
    device_upper.alloc_and_upload(upper);
    device_results.alloc(lower.size());
    checked(cudaMemsetAsync(device_results.ptr(), 0xa5,
                            device_results.size_in_bytes(), stream));
    if (count_only)
        index.count(device_lower.ptr(), device_upper.ptr(), device_results.ptr(),
                    lower.size(), stream);
    else
        index.range_lookup_sum(device_lower.ptr(), device_upper.ptr(),
                               device_results.ptr(), lower.size(), stream);
    checked(cudaStreamSynchronize(stream));
    const auto answers = device_results.download(lower.size());

    // CPU prefix sums avoid repeating work for the large overlapping batch.
    std::vector<key32> keys;
    std::vector<std::uint64_t> sums{0};
    for (const auto &entry : oracle)
    {
        keys.push_back(entry.first);
        sums.push_back(sums.back() + entry.second);
    }
    for (size_t i = 0; i < lower.size(); ++i)
    {
        smallsize expected = 0;
        if (lower[i] <= upper[i])
        {
            const size_t begin = std::lower_bound(keys.begin(), keys.end(),
                                                   lower[i]) - keys.begin();
            const size_t end = std::upper_bound(keys.begin(), keys.end(),
                                                 upper[i]) - keys.begin();
            expected = static_cast<smallsize>(
                count_only ? end - begin : sums[end] - sums[begin]);
        }
        if (answers[i] != expected)
            throw std::runtime_error(std::string(label) + " query " +
                std::to_string(i) + " returned " + std::to_string(answers[i]) +
                ", expected " + std::to_string(expected));
    }
    std::cout << "PASS " << label << " queries=" << lower.size() << std::endl;
}

void verify_mixed(index_type &index, const std::map<key32, smallsize> &oracle,
                  cudaStream_t stream)
{
    // Uneven warp/block tails, inclusive endpoints, reversed bounds, misses,
    // tombstone-only ranges, and full-domain ranges on a non-default stream.
    std::vector<key32> lower, upper;
    for (size_t i = 0; i < 259; ++i)
    {
        switch (i % 6)
        {
        case 0: lower.push_back(0); upper.push_back(batch_size - 1); break;
        case 1: lower.push_back(i); upper.push_back(i); break;
        case 2: lower.push_back(100); upper.push_back(0); break;
        case 3: lower.push_back(batch_size); upper.push_back(batch_size + 9); break;
        case 4: lower.push_back(0); upper.push_back(batch_size / 2 - 1); break;
        default: lower.push_back(batch_size / 2); upper.push_back(batch_size - 2);
        }
    }
    verify(index, oracle, lower, upper, true, stream, "mixed COUNT");
    verify(index, oracle, lower, upper, false, stream, "mixed RANGE");
    verify(index, oracle, {7, 8}, {6, 7}, false, stream, "zero candidates");
    index.count(nullptr, nullptr, nullptr, 0, stream);
    index.range_lookup_sum(nullptr, nullptr, nullptr, 0, stream);
    checked(cudaStreamSynchronize(stream));
}

int main(int argc, char **argv)
{
    try
    {
        if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--large"))
            throw std::invalid_argument("usage: lsmu_range_overflow [--large]");
        checked(cudaSetDevice(0));
        cudaStream_t stream;
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        size_t available_bytes = 0, total_bytes = 0;
        checked(cudaMemGetInfo(&available_bytes, &total_bytes));
        index_type index;
        std::map<key32, smallsize> oracle;
        index.build(nullptr, 0, 4 * batch_size, available_bytes, nullptr, nullptr);
        verify_mixed(index, oracle, stream);
        index.destroy();

        std::vector<key32> keys(batch_size);
        std::iota(keys.rbegin(), keys.rend(), key32{0});
        cuda_buffer<key32> device_keys;
        device_keys.alloc_and_upload(keys);
        index.build(device_keys.ptr(), keys.size(), 4 * batch_size,
                    available_bytes, nullptr, nullptr);
        for (size_t i = 0; i < keys.size(); ++i)
            oracle[keys[i]] = static_cast<smallsize>(i);
        verify_mixed(index, oracle, stream);

        std::vector<smallsize> values(keys.size());
        for (size_t i = 0; i < keys.size(); ++i)
            oracle[keys[i]] = values[i] = 0xffff0000u + keys[i];
        cuda_buffer<smallsize> device_values;
        device_values.alloc_and_upload(values);
        index.insert(device_keys.ptr(), device_values.ptr(), keys.size(), stream);
        verify_mixed(index, oracle, stream);

        // One partial deletion batch is padded with its last tombstone, leaving
        // three full physical batches of candidates but only half the live keys.
        std::iota(keys.begin(), keys.end(), key32{0});
        device_keys.upload(keys.data(), keys.size());
        index.remove(device_keys.ptr(), batch_size / 2, stream);
        for (key32 key = 0; key < batch_size / 2; ++key)
            oracle.erase(key);
        verify_mixed(index, oracle, stream);

        if (argc == 2)
        {
            constexpr std::uint64_t candidates_per_query = 3 * batch_size;
            for (bool count_only : {true, false})
            {
                const std::uint64_t old_limit = count_only
                    ? std::numeric_limits<int>::max()
                    : std::numeric_limits<smallsize>::max();
                const size_t query_count = old_limit / candidates_per_query + 1;
                std::vector<key32> lower(query_count, 0);
                std::vector<key32> upper(query_count, batch_size - 1);
                // Distinct answers after the overlapping queries also check
                // that recursively split calls write to the correct positions.
                lower.insert(lower.end(), {5, 0, batch_size - 1, batch_size});
                upper.insert(upper.end(), {4, 0, batch_size - 1, batch_size + 1});
                std::cout << "RUN " << (count_only ? "COUNT > INT_MAX" : "RANGE > UINT32_MAX")
                          << " candidates>=" << query_count * candidates_per_query
                          << std::endl;
                verify(index, oracle, lower, upper, count_only, stream,
                       count_only ? "COUNT > INT_MAX" : "RANGE > UINT32_MAX");
            }
        }

        index.insert(device_keys.ptr(), device_values.ptr(), 101, stream);
        for (size_t i = 0; i < 101; ++i)
            oracle[keys[i]] = values[i];
        verify_mixed(index, oracle, stream);
        index.destroy();
        checked(cudaStreamDestroy(stream));
        std::cout << "PASS LSMu range overflow regression" << std::endl;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL " << error.what() << std::endl;
        return 1;
    }
}
