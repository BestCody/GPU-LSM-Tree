// Regression for buckets whose lower bound exceeds every query.
#include "paper_backends.cuh"

#include <iostream>
#include <vector>

__global__ void query_bounds(int *results) {
    const key32 keys[] = {3, 7, 11, 11, 20};
    const key32 probes[] = {2, 3, 4, 11, 12, 20, 21, 100};
    const int thread = threadIdx.x;
    if (thread < 8)
        results[thread] = binarySearchIndex_leftmost_ge(keys, probes[thread], 0, 5);
    if (thread == 8)
        results[thread] = binarySearchIndex_leftmost_ge(keys, key32{12}, 1, 4);
    if (thread == 9)
        results[thread] = binarySearchIndex_leftmost_ge(keys, key32{11}, 1, 4);
    if (thread == 10)
        results[thread] = binarySearchIndex_leftmost_ge(keys, key32{3}, 0, 0);
}

bool check_bounds() {
    cuda_buffer<int> output;
    output.alloc(11);
    query_bounds<<<1, 32>>>(output.ptr());
    flix_benchmark::check(cudaDeviceSynchronize());
    const auto answers = output.download(11);
    const int expected[] = {0, 0, 1, 2, 4, 4, -1, -1, -1, 2, -1};
    bool correct = true;
    for (size_t i = 0; i < answers.size(); ++i) {
        if (expected[i] < 0 ? answers[i] >= 0 : answers[i] != expected[i]) {
            std::cerr << "FAIL lower bound case=" << i << " actual=" << answers[i]
                      << " expected=" << expected[i] << '\n';
            correct = false;
        }
    }
    if (correct) std::cout << "PASS lower bound edges and duplicates\n";
    return correct;
}

bool check_index(size_t count, key32 base) {
    std::vector<key32> keys(count);
    for (size_t i = 0; i < count; ++i) keys[i] = base + 2 * (count - i);
    cuda_buffer<key32> input;
    input.alloc(count);
    flix_benchmark::check(cudaMemcpy(input.ptr(), keys.data(), count * sizeof(key32),
                                    cudaMemcpyHostToDevice));
    selected_paper_backend index;
    size_t free_bytes, total_bytes;
    flix_benchmark::check(cudaMemGetInfo(&free_bytes, &total_bytes));
    paper_build(index, input.ptr(), count, count, free_bytes);
    flix_benchmark::lookup_workspace<key32> workspace;
    bool correct = true;
    const key32 end = static_cast<key32>(2 * count);
    std::vector<std::vector<key32>> batches = {
        {2}, {end / 2}, {8, 2, 4, 8}, {3, 2, 8, 7},
        {end, 2, end - 2}, {end + 2, end + 1}, {}};
    batches.emplace_back(33, 8);
    for (size_t batch = 0; batch < batches.size(); ++batch) {
        auto probes = batches[batch];
        for (auto &key : probes) key += base;
        cuda_buffer<key32> queries;
        cuda_buffer<smallsize> output;
        queries.alloc(std::max<size_t>(1, probes.size()));
        output.alloc(std::max<size_t>(1, probes.size()));
        if (!probes.empty())
            flix_benchmark::check(cudaMemcpy(queries.ptr(), probes.data(),
                probes.size() * sizeof(key32), cudaMemcpyHostToDevice));
        workspace.lookup(index, queries.ptr(), output.ptr(), probes.size());
        const auto answers = output.download(probes.size());
        for (size_t i = 0; i < probes.size(); ++i) {
            const key32 relative = probes[i] - base;
            const smallsize expected = relative >= 2 && relative <= end && relative % 2 == 0
                ? static_cast<smallsize>(count - relative / 2) : not_found;
            if (answers[i] != expected) {
                std::cerr << "FAIL lookup count=" << count << " base=" << base
                          << " batch=" << batch << " query=" << probes[i]
                          << " actual=" << answers[i] << " expected=" << expected << '\n';
                correct = false;
            }
        }
    }
    index.destroy();
    flix_benchmark::check(cudaDeviceSynchronize());
    if (correct) std::cout << "PASS sparse lookup count=" << count << " base=" << base << '\n';
    return correct;
}

int main() {
    try {
        flix_benchmark::check(cudaSetDevice(0));
        bool correct = check_bounds();
        for (size_t count : {size_t{16}, size_t{17}, size_t{4096}, size_t{1} << 20})
            correct = check_index(count, 0) && correct;
        correct = check_index(size_t{1} << 20, (key32{1} << 31) - (key32{1} << 21) - 64) && correct;
        return correct ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "FAIL " << error.what() << std::endl;
        return 1;
    }
}
