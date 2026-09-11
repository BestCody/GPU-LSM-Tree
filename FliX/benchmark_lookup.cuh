#ifndef FLIX_BENCHMARK_LOOKUP_CUH
#define FLIX_BENCHMARK_LOOKUP_CUH
#ifndef FLIX_COMPLETE_UNSORTED_LOOKUP
#define FLIX_COMPLETE_UNSORTED_LOOKUP 1
#endif

#include "utilities.cuh"

#include <array>
#include <chrono>
#include <type_traits>

namespace flix_benchmark {

inline void check(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(result));
}

struct lookup_times {
    double total_ms = 0;
    double wall_ms = 0;
    double prepare_ms = 0;
    double search_ms = 0;
    double restore_ms = 0;
};

template <typename Index, typename = void>
struct requires_ordered_lookup : std::false_type {};

template <typename Index>
struct requires_ordered_lookup<Index,
    std::void_t<decltype(Index::requires_sorted_lookup)>>
    : std::bool_constant<Index::requires_sorted_lookup> {};

static __global__ void make_lookup_permutation(uint32_t *ids, size_t n) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) ids[i] = static_cast<uint32_t>(i);
}

static __global__ void restore_lookup_answers(const smallsize *sorted,
    const uint32_t *permutation, smallsize *output, size_t n) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) output[permutation[i]] = sorted[i];
}

template <typename Key>
class lookup_workspace {
    cuda_buffer<uint32_t> permutation_in_, permutation_sorted_;
    cuda_buffer<uint8_t> scratch_;
    std::array<cudaEvent_t, 4> events_{};

    template <typename T>
    static void ensure(cuda_buffer<T> &buffer, size_t n) {
        if (buffer.num_elements < n) {
            buffer.resize(n);
            check(cudaGetLastError());
        }
    }

    double elapsed(unsigned begin, unsigned end) const {
        float ms = 0;
        check(cudaEventElapsedTime(&ms, events_[begin], events_[end]));
        return ms;
    }

public:
    cuda_buffer<Key> sorted_keys;
    cuda_buffer<smallsize> sorted_answers;

    lookup_workspace() {
        try {
            for (auto &event : events_) check(cudaEventCreate(&event));
        } catch (...) {
            for (auto event : events_) if (event) cudaEventDestroy(event);
            throw;
        }
    }
    lookup_workspace(const lookup_workspace &) = delete;
    lookup_workspace &operator=(const lookup_workspace &) = delete;
    ~lookup_workspace() {
        for (auto event : events_) cudaEventDestroy(event);
    }

    void prepare_ordered(const Key *keys, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        if (n > std::numeric_limits<uint32_t>::max())
            throw std::overflow_error("Lookup permutation exceeds 32-bit indices");
        ensure(sorted_keys, n);
        ensure(sorted_answers, n);
        ensure(permutation_in_, n);
        ensure(permutation_sorted_, n);
        make_lookup_permutation<<<(n + 255) / 256, 256, 0, stream>>>(
            permutation_in_.ptr(), n);
        check(cudaGetLastError());
        size_t scratch_bytes = 0;
        check(cub::DeviceRadixSort::SortPairs(nullptr, scratch_bytes,
            keys, sorted_keys.ptr(), permutation_in_.ptr(),
            permutation_sorted_.ptr(), n, 0, sizeof(Key) * 8, stream));
        ensure(scratch_, scratch_bytes);
        check(cub::DeviceRadixSort::SortPairs(scratch_.ptr(), scratch_bytes,
            keys, sorted_keys.ptr(), permutation_in_.ptr(),
            permutation_sorted_.ptr(), n, 0, sizeof(Key) * 8, stream));
        check(cudaMemsetAsync(sorted_answers.ptr(), 0,
                             n * sizeof(smallsize), stream));
    }

    template <typename Index>
    lookup_times lookup(Index &index, const Key *keys, smallsize *output,
                        size_t n, cudaStream_t stream = 0) {
        if (n == 0) return {};
        // Inputs must be ready before the common timing boundary.
        check(cudaStreamSynchronize(stream));
        const auto wall_start = std::chrono::steady_clock::now();
        check(cudaEventRecord(events_[0], stream));
        if constexpr (requires_ordered_lookup<Index>::value) {
            prepare_ordered(keys, n, stream);
        } else {
            check(cudaMemsetAsync(output, 0, n * sizeof(smallsize), stream));
        }
        check(cudaEventRecord(events_[1], stream));
        if constexpr (requires_ordered_lookup<Index>::value)
            index.lookups_ordered(sorted_keys.ptr(), sorted_answers.ptr(), n, stream);
        else
            index.lookup(keys, output, n, stream);
        check(cudaGetLastError());
        check(cudaEventRecord(events_[2], stream));
        if constexpr (requires_ordered_lookup<Index>::value) {
            restore_lookup_answers<<<(n + 255) / 256, 256, 0, stream>>>(
                sorted_answers.ptr(), permutation_sorted_.ptr(), output, n);
            check(cudaGetLastError());
        }
        check(cudaEventRecord(events_[3], stream));
        check(cudaEventSynchronize(events_[3]));
        const double wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall_start).count();
        return {elapsed(0, 3), wall_ms, elapsed(0, 1),
                elapsed(1, 2), elapsed(2, 3)};
    }
};

} // namespace flix_benchmark
#endif
