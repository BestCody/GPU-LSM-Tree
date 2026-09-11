#ifndef FLIX_COARSE_GRANULAR_RANGE_ENUMERATION_CUH
#define FLIX_COARSE_GRANULAR_RANGE_ENUMERATION_CUH

#include "coarse_granular_range_queries.cuh"

namespace flix_range {

struct SumSink {
    smallsize sum = 0;

    template <typename Key>
    DEVICEQUALIFIER void emit(Key, smallsize value) {
        sum += value;
    }
};

// Emit each qualifying record; the sink owns the output.
template <typename Key, typename Sink>
DEVICEQUALIFIER void enumerate_bucket(
    updatable_cg_params* params, smallsize bucket,
    Key lower, Key upper, smallsize query, Sink& sink)
{
    const smallsize link_offset = get_lastposition_bytes<Key>(params->node_size);
    void* node = get_bucket_head_node<Key>(params, bucket);
    while (node && cg::extract<Key>(node, 0) < lower)
        node = get_next_range_node<Key>(params, node, link_offset, query);

    while (node) {
        const Key maximum = cg::extract<Key>(node, 0);
        const smallsize size = cg::extract<smallsize>(node, sizeof(Key));
        for (smallsize slot = 1; slot <= size; ++slot) {
            const Key key = extract_key_node<Key>(node, slot);
            if (key > upper) return;
            if (key >= lower)
                sink.emit(key, extract_offset_node<Key>(node, slot));
        }
        if (maximum >= upper) return;
        node = get_next_range_node<Key>(params, node, link_offset, query);
    }
}

template <typename Key, typename Sink>
DEVICEQUALIFIER void enumerate(
    updatable_cg_params* params, Key lower, Key upper,
    smallsize query, Sink& sink)
{
    if (lower > upper) return;
    const auto* maxima = static_cast<const Key*>(params->maxvalues);
    const smallsize buckets = params->partition_count_with_overflow;
    const int first = find_range_bucket_from_maxvalues(maxima, lower, buckets);
    const int last = find_range_bucket_from_maxvalues(maxima, upper, buckets);
    if (first < 0 || last < 0) return;
    for (int bucket = first; bucket <= last; ++bucket)
        enumerate_bucket(params, static_cast<smallsize>(bucket), lower, upper,
                         query, sink);
}

template <typename Key>
GLOBALQUALIFIER void enumerate_sum_kernel(
    updatable_cg_params* params, const Key* lower, const Key* upper,
    smallsize* output, smallsize count)
{
    const smallsize query = blockIdx.x * blockDim.x + threadIdx.x;
    if (query >= count) return;
    SumSink sink;
    enumerate(params, lower[query], upper[query], query, sink);
    output[query] = sink.sum;
}

}  // namespace flix_range

#endif
