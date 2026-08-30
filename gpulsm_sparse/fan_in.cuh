#pragma once

#include "exact.cuh"

#include <thrust/execution_policy.h>
#include <thrust/merge.h>

namespace gpulsm_sparse {

struct ExactRun {
  std::uint32_t begin{};
  std::uint32_t count{};
};

template <class IncomingSource>
struct TerminalRefLess {
  TerminalSourceView<IncomingSource> source{};

  __device__ std::uint64_t age(std::uint32_t ref) const {
    const TerminalCandidate &candidate = source.candidates[ref];
    // This is the canonical GPULSMOpt source order: pending is source zero
    // and resident levels are visited in ascending level order, so a smaller
    // resident level is newer.  Express that order as an increasing age so
    // the prototype's mark-last rule selects the same winner as the original
    // carry without mirroring epoch metadata for ordinary roots.
    return candidate.source == kTerminalIncomingSource
        ? std::numeric_limits<std::uint64_t>::max()
        : std::numeric_limits<std::uint64_t>::max() - 1u -
              candidate.source;
  }

  __device__ bool operator()(std::uint32_t left,
                             std::uint32_t right) const {
    const int comparison = compare_keys(
        source_key(source, left), source_key(source, right));
    if (comparison != 0) return comparison < 0;
    const std::uint64_t left_age = age(left);
    const std::uint64_t right_age = age(right);
    if (left_age != right_age) return left_age < right_age;
    const TerminalCandidate &a = source.candidates[left];
    const TerminalCandidate &b = source.candidates[right];
    if (a.source != b.source) return a.source < b.source;
    if (a.locator != b.locator) return a.locator < b.locator;
    return left < right;
  }
};

__global__ void initialize_terminal_refs(
    std::uint32_t *refs, std::uint32_t count) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position < count) refs[position] = position;
}

template <class IncomingSource>
__global__ void mark_last_terminal_key(
    TerminalSourceView<IncomingSource> source,
    const std::uint32_t *merged, std::uint8_t *flags,
    std::uint32_t count) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= count) return;
  flags[position] = static_cast<std::uint8_t>(
      position + 1u == count ||
      compare_keys(source_key(source, merged[position]),
                   source_key(source, merged[position + 1u])) != 0);
}

// Direct production extraction of exact_fanin_gate.cu: sorted exact streams
// are merged pairwise with thrust::merge, then the last (newest) member of
// each equal-key group is selected.  The merge tree is allowed to have more
// than the benchmark's four leaves because a terminal radix carry can expose
// any number of already-sorted resident roots.
class ExactFanInWorkspace {
 public:
  struct Result {
    const std::uint32_t *winners{};
    std::uint32_t winner_count{};
  };

  template <class IncomingSource>
  Result merge(TerminalSourceView<IncomingSource> source,
               const std::vector<ExactRun> &runs,
               std::uint32_t candidate_count, cudaStream_t stream) {
    if (!candidate_count || runs.empty()) return {};
    ensure(candidate_count);
    std::uint64_t covered = 0u;
    for (const ExactRun &run : runs) {
      if (!run.count || run.begin != covered)
        throw std::invalid_argument("noncontiguous exact fan-in runs");
      covered += run.count;
    }
    if (covered != candidate_count)
      throw std::invalid_argument("exact fan-in coverage mismatch");

    initialize_terminal_refs<<<blocks(candidate_count), kThreads, 0,
                               stream>>>(refs_a_.data(), candidate_count);
    check(cudaGetLastError(), "initialize exact fan-in refs");
    std::vector<ExactRun> current = runs;
    bool input_a = true;
    TerminalRefLess<IncomingSource> less{source};
    auto policy = thrust::cuda::par_nosync.on(stream);
    while (current.size() > 1u) {
      const std::uint32_t *input = input_a
          ? refs_a_.data() : refs_b_.data();
      std::uint32_t *output = input_a
          ? refs_b_.data() : refs_a_.data();
      std::vector<ExactRun> next;
      next.reserve((current.size() + 1u) / 2u);
      for (std::size_t index = 0u; index < current.size(); index += 2u) {
        const ExactRun left = current[index];
        if (index + 1u == current.size()) {
          check(cudaMemcpyAsync(
                    output + left.begin, input + left.begin,
                    std::size_t{left.count} * sizeof(std::uint32_t),
                    cudaMemcpyDeviceToDevice, stream),
                "copy odd exact fan-in run");
          next.push_back(left);
          continue;
        }
        const ExactRun right = current[index + 1u];
        if (left.begin + left.count != right.begin)
          throw std::invalid_argument("disjoint exact fan-in pair");
        thrust::merge(policy,
                      input + left.begin,
                      input + left.begin + left.count,
                      input + right.begin,
                      input + right.begin + right.count,
                      output + left.begin, less);
        next.push_back({left.begin, left.count + right.count});
      }
      current = std::move(next);
      input_a = !input_a;
    }

    const std::uint32_t *merged = input_a
        ? refs_a_.data() : refs_b_.data();
    mark_last_terminal_key<<<blocks(candidate_count), kThreads, 0, stream>>>(
        source, merged, last_.data(), candidate_count);
    std::size_t bytes = select_bytes_;
    check(cub::DeviceSelect::Flagged(
              temporary_.data(), bytes, merged, last_.data(),
              winners_.data(), winner_count_.data(), candidate_count,
              stream),
          "select exact fan-in winners");
    std::uint32_t winners = 0u;
    check(cudaMemcpyAsync(&winners, winner_count_.data(), sizeof(winners),
                          cudaMemcpyDeviceToHost, stream),
          "copy exact fan-in winner count");
    check(cudaStreamSynchronize(stream), "wait exact fan-in winners");
    return {winners_.data(), winners};
  }

  std::size_t bytes() const {
    return refs_a_.bytes() + refs_b_.bytes() + last_.bytes() +
        winners_.bytes() + winner_count_.bytes() + temporary_.bytes();
  }

 private:
  void ensure(std::uint32_t capacity) {
    if (capacity <= capacity_) return;
    capacity_ = capacity;
    refs_a_.reset(capacity);
    refs_b_.reset(capacity);
    last_.reset(capacity);
    winners_.reset(capacity);
    winner_count_.reset(1u);
    std::size_t bytes = 0u;
    check(cub::DeviceSelect::Flagged(
              nullptr, bytes, refs_a_.data(), last_.data(),
              winners_.data(), winner_count_.data(), capacity),
          "size exact fan-in select");
    select_bytes_ = bytes;
    temporary_.reset(bytes);
  }

  std::uint32_t capacity_{};
  Buffer<std::uint32_t> refs_a_, refs_b_, winners_, winner_count_;
  Buffer<std::uint8_t> last_, temporary_;
  std::size_t select_bytes_{};
};

}  // namespace gpulsm_sparse
