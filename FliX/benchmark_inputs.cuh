#ifndef FLIX_BENCHMARK_INPUTS_CUH
#define FLIX_BENCHMARK_INPUTS_CUH

#include "definitions.cuh"
#include <iomanip>
#include <type_traits>
#include <vector>

#ifndef FLIX_BENCHMARK_KEY_BITS
#define FLIX_BENCHMARK_KEY_BITS 31
#endif

namespace flix_benchmark {

template <typename Index, typename = void>
struct supported_max_key {
    static constexpr auto value = max_usable_key<typename Index::key_type>();
};

template <typename Index>
struct supported_max_key<Index, std::void_t<decltype(Index::max_supported_key)>> {
    static constexpr auto value = Index::max_supported_key;
};

template <typename Index>
constexpr typename Index::key_type common_max_key() {
    using Key = typename Index::key_type;
    static_assert(FLIX_BENCHMARK_KEY_BITS > 0 &&
                  FLIX_BENCHMARK_KEY_BITS <= sizeof(Key) * 8,
                  "Benchmark key domain exceeds the key type");
    constexpr Key limit = max_usable_key<Key>(FLIX_BENCHMARK_KEY_BITS);
    static_assert(limit <= supported_max_key<Index>::value,
                  "Benchmark key domain is unsupported by this backend");
    return limit;
}

// A portable checksum of ordered requests, including counts.
class input_checksum {
    uint64_t value_ = UINT64_C(14695981039346656037);
public:
    void add(uint64_t value) {
        for (unsigned byte = 0; byte < 8; ++byte) {
            value_ ^= (value >> (8 * byte)) & 255;
            value_ *= UINT64_C(1099511628211);
        }
    }
    template <typename T>
    void add(const T *values, size_t n) {
        add(n);
        for (size_t i = 0; i < n; ++i) add(static_cast<uint64_t>(values[i]));
    }
    template <typename T>
    void add(const std::vector<T> &values) { add(values.data(), values.size()); }
    std::string str() const {
        std::ostringstream output;
        output << std::hex << std::setfill('0') << std::setw(16) << value_;
        return output.str();
    }
};

} // namespace flix_benchmark
#endif
