#ifndef FLIX_BENCHMARK_RANGE_CUH
#define FLIX_BENCHMARK_RANGE_CUH

#include <type_traits>

namespace flix_benchmark {

inline constexpr const char* enumeration_sum_contract = "enumerate_records_sum_v1";

template <typename Index, typename = void>
struct enumerates_range_records : std::false_type {};

template <typename Index>
struct enumerates_range_records<Index,
    std::void_t<decltype(Index::range_enumerates_records)>>
    : std::bool_constant<Index::range_enumerates_records> {};

template <typename Index>
constexpr const char* range_processing() {
    return enumerates_range_records<Index>::value
        ? enumeration_sum_contract : "aggregate_sum_unspecified";
}

}  // namespace flix_benchmark

#endif
