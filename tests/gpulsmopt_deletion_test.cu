#include "GPULSMOpt.cuh"

#include <chrono>
#include <iostream>
#include <map>
#include <numeric>

// Inspect physical retention as well as the public results. A deleted key
// must not reappear, and the oldest ordinary output must shed its tombstones.
struct GPULSMOptDeletionTestAccess {
  static std::size_t shortcut_capacity(GPULSMOpt &index) {
    return index.sealed_workspace_ ? index.sealed_workspace_->capacity() : 0u;
  }
  static std::uint64_t stored(GPULSMOpt &index) {
    index.resolve_publication_receipt();
    return std::accumulate(std::begin(index.level_counts_),
                           std::end(index.level_counts_), std::uint64_t{0});
  }
  static std::uint64_t oldest_tombstones(GPULSMOpt &index) {
    index.resolve_publication_receipt();
    const auto mask = index.host_occupied_level_mask_;
    if (!mask) return 0u;
    const unsigned level = 63u - __builtin_clzll(mask);
    const auto snapshot = index.read_sealed_manifest();
    const auto generation = snapshot.manifests[snapshot.active]
        .levels[level].storage_generation;
    const auto begin = index.level_begin(level) +
        std::uint64_t{generation} * index.level_capacity(level);
    std::vector<std::uint32_t> flags(index.level_counts_[level]);
    if (!flags.empty())
      CUDA_CHECK(cudaMemcpy(flags.data(), index.resident_rows().key_flags + begin,
                            flags.size() * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost));
    return std::count_if(flags.begin(), flags.end(), [](std::uint32_t word) {
      return (word & (gpulsmopt2_detail::kTombstone << 16u)) != 0;
    });
  }
};

namespace {
template <class T> using Buffer = gpulsm_sparse::Buffer<T>;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <class T> Buffer<T> upload(const std::vector<T> &values) {
  Buffer<T> buffer(values.size());
  if (!values.empty())
    CUDA_CHECK(cudaMemcpy(buffer.data(), values.data(), buffer.bytes(),
                          cudaMemcpyHostToDevice));
  return buffer;
}

__host__ __device__ std::uint32_t key_for(std::uint32_t id) {
  return id * 2654435761u;
}

struct Fixture {
  static constexpr unsigned epoch = 512u;
  GPULSMOpt index;
  std::map<std::uint32_t, std::uint32_t> expected;
  Fixture() : index(config()) {}
  static DictionaryConfig config() {
    DictionaryConfig result;
    result.max_elements = 1u << 15u;
    result.batch_capacity = epoch / 16u;
    result.level_zero_capacity = epoch;
    return result;
  }
  void put(unsigned first, unsigned count, unsigned version, bool build = false) {
    std::vector<std::uint32_t> keys(count), values(count);
    for (unsigned i = 0; i < count; ++i) {
      keys[i] = key_for(first + i);
      values[i] = version + i;
      expected[keys[i]] = values[i];
    }
    auto dk = upload(keys);
    auto dv = upload(values);
    if (build) index.bulk_build(dk.data(), dv.data(), count, 0);
    else index.insert({dk.data(), dv.data(), count}, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  void erase(unsigned count, unsigned distinct, unsigned first = 0) {
    std::vector<std::uint32_t> keys(count);
    for (unsigned i = 0; i < count; ++i) {
      keys[i] = key_for(first + i % distinct);
      expected.erase(keys[i]);
    }
    auto dk = upload(keys);
    index.erase({dk.data(), count}, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    require(index.canonical_carry_status() == 0u, "publication failed");
  }
  void check(const char *phase) {
    constexpr unsigned count = 8192;
    std::vector<std::uint32_t> keys(count);
    for (unsigned i = 0; i < count; ++i) keys[i] = key_for(i);
    auto dk = upload(keys);
    Buffer<std::uint32_t> values(count);
    Buffer<std::uint8_t> found(count);
    index.lookup({dk.data(), count, values.data(), found.data()}, 0);
    std::vector<std::uint32_t> got(count);
    std::vector<std::uint8_t> present(count);
    CUDA_CHECK(cudaMemcpy(got.data(), values.data(), values.bytes(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(present.data(), found.data(), found.bytes(), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < count; ++i) {
      auto it = expected.find(keys[i]);
      if (bool(present[i]) != (it != expected.end()) ||
          (present[i] && got[i] != it->second))
        throw std::runtime_error(std::string(phase) + ": lookup mismatch at id " + std::to_string(i));
    }
    constexpr unsigned ordered_count = 32;
    Buffer<std::uint32_t> successors(ordered_count), sums(ordered_count);
    std::vector<std::uint32_t> upper(ordered_count);
    for (unsigned i = 0; i < ordered_count; ++i)
      upper[i] = static_cast<std::uint32_t>(std::min<std::uint64_t>(
          std::uint64_t{keys[i]} + 100000000u, 0xffffffffu));
    auto du = upload(upper);
    index.successor({dk.data(), ordered_count, successors.data()}, 0);
    index.range({dk.data(), du.data(), ordered_count, sums.data()}, 0);
    std::vector<std::uint32_t> gs(ordered_count), gr(ordered_count);
    CUDA_CHECK(cudaMemcpy(gs.data(), successors.data(), successors.bytes(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gr.data(), sums.data(), sums.bytes(), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < ordered_count; ++i) {
      auto it = expected.lower_bound(keys[i]);
      require(gs[i] == (it == expected.end() ? 0xffffffffu : it->first), "successor mismatch");
      std::uint32_t sum = 0u;
      for (; it != expected.end() && it->first <= upper[i]; ++it) sum += it->second;
      require(gr[i] == sum, "range sum mismatch");
    }
    std::cout << "PASS " << phase << '\n';
  }
};

void ordinary_checks() {
  {
    Fixture f;
    f.put(0, 4096, 100, true);
    f.erase(4 * Fixture::epoch, 1024); // Direct output, older run outside.
    require(GPULSMOptDeletionTestAccess::shortcut_capacity(f.index), "deletion missed shortcut");
    f.check("direct deletion output retains tombstones above an older run");
    // Three pending batches must fill and publish before the shortcut can
    // handle the remaining complete epochs and leave a partial tail.
    f.put(0, 7, 8000);
    f.erase(5 * Fixture::epoch + 19, 1400);
    f.check("pending data, merged outputs, duplicate deletions, partial tail");
    f.put(0, 4 * Fixture::epoch, 9000);
    f.check("large insertion after deletion and reinsertion");
  }
  {
    Fixture f;
    f.put(0, 128, 1, true);
    f.erase(3 * Fixture::epoch, 1024); // Merge the only older run.
    require(GPULSMOptDeletionTestAccess::shortcut_capacity(f.index), "all-run deletion missed shortcut");
    require(GPULSMOptDeletionTestAccess::stored(f.index) == 0, "unneeded tombstones retained");
    f.check("all older versions merged, zero surviving elements");
    f.put(0, 31, 10000);
    f.check("reinsertion after empty output");
    f.erase(8 * Fixture::epoch + 37, 2048);
    f.check("later publication after empty output");
  }
  {
    Fixture f;
    f.erase(9 * Fixture::epoch, 3000); // Several outputs, no original runs.
    require(GPULSMOptDeletionTestAccess::shortcut_capacity(f.index), "empty-tree deletion missed shortcut");
    require(GPULSMOptDeletionTestAccess::oldest_tombstones(f.index) == 0, "oldest direct output retained tombstones");
    f.check("multiple deletion outputs on an empty structure");
    f.put(0, 4 * Fixture::epoch, 20000);
    f.check("insertion after multiple empty-tree outputs");
  }
  {
    Fixture f;
    f.put(0, 128, 1, true);
    f.erase(Fixture::epoch, 128);
    f.put(256, 2 * Fixture::epoch, 40000);
    require(GPULSMOptDeletionTestAccess::oldest_tombstones(f.index) == 0,
            "insertion shortcut retained obsolete tombstones");
    f.check("insertion shortcut removes existing tombstones in an all-run merge");
  }
  {
    Fixture f;
    f.put(0, 128, 1, true);
    // Fill the hierarchy and exercise the replacement slot repeatedly.
    for (unsigned round = 0; round < 8; ++round)
      f.erase(16 * Fixture::epoch, 4096);
    f.check("deletion through final-slot replacement");
    f.put(0, 64, 30000);
    f.check("reinsertion after final-slot replacement");
  }
}

__global__ void generate(std::uint32_t *keys, std::uint32_t *values, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) { keys[i] = key_for(i); values[i] = i % 1000u + 1u; }
}

__global__ void check_large(const std::uint32_t *values, const std::uint8_t *found,
                            unsigned count, unsigned erased, unsigned *errors) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count && (bool(found[i]) != (i >= erased) ||
                   (found[i] && values[i] != i % 1000u + 1u)))
    atomicAdd(errors, 1u);
}

void time_one_case() {
  constexpr unsigned n = 1u << 27u, erased = 1u << 26u;
  DictionaryConfig config;
  config.max_elements = 1u << 28u;
  config.batch_capacity = 1u << 20u;
  config.level_zero_capacity = 1u << 24u;
  GPULSMOpt index(config);
  Buffer<std::uint32_t> keys(n), values(n);
  generate<<<(n + 255u) / 256u, 256>>>(keys.data(), values.data(), n);
  index.bulk_build(keys.data(), values.data(), n, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  const auto begin = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventRecord(start));
  index.erase({keys.data(), erased}, 0); // Exactly one measured deletion.
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  const double wall_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
  float event_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&event_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  require(GPULSMOptDeletionTestAccess::shortcut_capacity(index), "timing case missed shortcut");
  require(index.canonical_carry_status() == 0u, "timing publication failed");
  std::cout << "{\"kind\":\"timing\",\"initial_keys\":" << n
            << ",\"deleted_keys\":" << erased << ",\"repetitions\":1"
            << ",\"complete_wall_ms\":" << wall_ms << ",\"cuda_event_ms\":" << event_ms
            << ",\"million_deletions_per_s\":" << erased / (wall_ms * 1000.0)
            << "}" << std::endl;
  // Check every original key outside timing, transferring only an error count.
  Buffer<std::uint8_t> found(n);
  Buffer<unsigned> errors(1);
  CUDA_CHECK(cudaMemset(errors.data(), 0, sizeof(unsigned)));
  index.lookup({keys.data(), n, values.data(), found.data()}, 0);
  check_large<<<(n + 255u) / 256u, 256>>>(values.data(), found.data(), n, erased, errors.data());
  unsigned mistakes = 0;
  CUDA_CHECK(cudaMemcpy(&mistakes, errors.data(), sizeof(mistakes), cudaMemcpyDeviceToHost));
  require(mistakes == 0u, "large deletion result mismatch");
  std::cout << "PASS all " << n << " original keys checked outside timing\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    CUDA_CHECK(cudaSetDevice(0));
    if (argc == 2 && std::string(argv[1]) == "--timing") time_one_case();
    else if (argc == 1) ordinary_checks();
    else throw std::invalid_argument("usage: gpulsmopt_deletion_test [--timing]");
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
