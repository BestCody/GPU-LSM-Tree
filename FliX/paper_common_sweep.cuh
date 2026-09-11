// Included inside the paper driver's namespace.
__global__ void fill_common_values(smallsize *values, uint32_t begin, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] = begin + i;
}

__global__ void validate_common_lookup(const smallsize *results, uint32_t n,
    uint32_t resident, bool hits, uint32_t seed, unsigned long long *errors) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && results[i] != (hits ? mix32(i + seed) % resident : not_found))
        atomicAdd(errors, 1ull);
}

struct common_rows {
    std::ofstream output;
    explicit common_rows(const std::filesystem::path &directory)
        : output(directory / "measurements.csv") {
        output << "system,protocol,operation,batch_log,state,resident_elements,"
                  "items,scenario,time_ms,wall_ms,prepare_ms,search_ms,restore_ms,"
                  "index_bytes,input_sum,input_xor,checksum_sum,checksum_xor\n";
        output << std::setprecision(12);
    }
    void add(const char *operation, uint32_t state, uint32_t resident,
             uint32_t items, const std::string &scenario,
             const flix_benchmark::lookup_times &times, size_t bytes,
             const std::vector<unsigned long long> &input = {0, 0},
             const std::vector<unsigned long long> &answer = {0, 0}) {
        output << index_name << ",common_initialized_v1," << operation << ','
               << batch_log << ',' << state << ',' << resident << ',' << items
               << ',' << scenario << ',' << times.total_ms << ',' << times.wall_ms
               << ',' << times.prepare_ms << ',' << times.search_ms << ','
               << times.restore_ms << ',' << bytes << ',' << input[0] << ','
               << input[1] << ',' << answer[0] << ',' << answer[1] << '\n';
        output.flush();
        if (!output) throw std::runtime_error("Cannot write paper measurements");
    }
};

template <typename Function>
flix_benchmark::lookup_times measure_common(gpu_timer &timer, Function function) {
    PAPER_CUDA(cudaDeviceSynchronize());
    flix_benchmark::lookup_times result;
    result.wall_ms = measure_wall_ms([&] { result.total_ms = timer.measure(function); });
    result.search_ms = result.total_ms;
    return result;
}

std::vector<unsigned long long> common_digest(const uint32_t *values, uint32_t n,
                                             cuda_buffer<unsigned long long> &digest,
                                             uint32_t offset = 0) {
    digest.zero();
    if (n) digest_range_results<<<(n + threads - 1) / threads, threads>>>(
        values, n, offset, digest.ptr());
    PAPER_CUDA(cudaGetLastError());
    return digest.download(2);
}

template <typename Index>
void run_common_impl(const options &configuration) {
    if (batch_log == 0 || batch_log > configuration.insert_limit_log ||
        configuration.insert_limit_log > 27 || configuration.query_limit_log == 0 ||
        configuration.query_limit_log > configuration.insert_limit_log ||
        configuration.range_chunk_log == 0 ||
        configuration.range_chunk_log > configuration.query_limit_log)
        throw std::invalid_argument("Invalid common paper size limits");
    constexpr uint32_t batch = uint32_t{1} << batch_log;
    const uint32_t maximum = uint32_t{1} << configuration.insert_limit_log;
    const uint32_t query_max = uint32_t{1} << configuration.query_limit_log;
    uint32_t states = maximum / batch;
    if (configuration.stop_after_r) states = std::min(states, configuration.stop_after_r);
    if (!paper_dynamic && !configuration.bulk_sweep)
        states = std::min(states, query_max / batch);
    if (states == 0) throw std::invalid_argument("No states fit the configured limits");
    if (configuration.cleanup_sweep || configuration.forced_unified_validation ||
        configuration.construction_only || configuration.profile_all_inserts ||
        configuration.profile_insert_r)
        throw std::invalid_argument("LSM-specific modes require the legacy experiment family");

    std::ofstream capability(configuration.output_directory / "capabilities.json");
    capability << "{\"protocol\":\"common_initialized_v1\",\"dynamic\":"
               << (paper_dynamic ? "true" : "false") << ",\"range\":"
               << (paper_range ? "true" : "false")
               << ",\"live_overwrite\":false,\"explicit_cleanup\":false,"
                  "\"lookup_timing\":\"complete_unsorted_v1\","
                  "\"range_timing\":\"complete_unsorted_range_v1\","
                  "\"key_mapping\":\"paper_permutation_plus_2\","
                  "\"key_min\":2,\"key_max\":1073741825}\n";
    common_rows rows(configuration.output_directory);
    gpu_timer timer;
    cuda_buffer<key_type> keys, queries, upper;
    cuda_buffer<smallsize> values, answers;
    cuda_buffer<unsigned long long> digest, errors;
    const uint32_t buffer_size = configuration.bulk_sweep ? maximum : batch;
    keys.alloc(buffer_size);
    values.alloc(buffer_size);
    const uint32_t query_capacity = std::max(batch, std::min(maximum, query_max));
    queries.alloc(query_capacity);
    answers.alloc(query_capacity);
    digest.alloc(2);
    errors.alloc(1);
    if constexpr (paper_range) {
        if (!configuration.skip_ranges) upper.alloc(query_capacity);
    }
    std::unique_ptr<Index> index;
    flix_benchmark::lookup_workspace<key_type> lookup;

    auto validate = [&] {
        PAPER_CUDA(cudaDeviceSynchronize());
        if (errors.download_first_item() != 0)
            throw std::runtime_error("Common paper lookup validation failed");
    };
    auto query_state = [&](uint32_t state, uint32_t resident, const char *phase) {
        const uint32_t count = std::min(resident, query_max);
        for (bool hits : {true, false}) {
            const uint32_t seed = (hits ? 0x10000u : 0x20000u) + state;
            fill_lookup_queries<<<(count + threads - 1) / threads, threads>>>(
                queries.ptr(), count, resident, hits, seed);
            const auto input = common_digest(queries.ptr(), count, digest);
            const auto times = lookup.lookup(*index, queries.ptr(), answers.ptr(), count);
            errors.zero();
            validate_common_lookup<<<(count + threads - 1) / threads, threads>>>(
                answers.ptr(), count, resident, hits, seed, errors.ptr());
            validate();
            rows.add(phase, state, resident, count, hits ? "all_existing" : "none_existing",
                     times, index->gpu_resident_bytes(), input);
        }
    };
    auto range_state = [&](uint32_t state, uint32_t resident, const char *phase) {
        if constexpr (paper_range) {
            if (configuration.skip_ranges || resident > query_max || batch_log > 20) return;
            for (uint32_t expected : {8u, 1024u}) {
                uint32_t remaining = resident, offset = 0;
                flix_benchmark::lookup_times times;
                std::vector<unsigned long long> input(2, 0), output(2, 0);
                while (remaining) {
                    const uint32_t count = std::min(remaining,
                        uint32_t{1} << configuration.range_chunk_log);
                    fill_range_queries<<<(count + threads - 1) / threads, threads>>>(
                        queries.ptr(), upper.ptr(), count, resident, expected,
                        0x30000u + state + expected + offset);
                    auto part = common_digest(queries.ptr(), count, digest, offset);
                    input[0] += part[0]; input[1] ^= part[1];
                    part = common_digest(upper.ptr(), count, digest, offset);
                    input[0] += part[0]; input[1] ^= (part[1] << 1) | (part[1] >> 63);
                    const auto measured = measure_common(timer, [&] {
                        index->range_lookup_sum(queries.ptr(), upper.ptr(), answers.ptr(), count, 0);
                    });
                    times.total_ms += measured.total_ms; times.wall_ms += measured.wall_ms;
                    part = common_digest(answers.ptr(), count, digest, offset);
                    output[0] += part[0]; output[1] ^= part[1];
                    remaining -= count; offset += count;
                }
                times.search_ms = times.total_ms;
                rows.add(phase, state, resident, resident, std::to_string(expected),
                         times, index->gpu_resident_bytes(), input, output);
            }
        }
    };
    const uint32_t first_state = configuration.bulk_sweep ? states : 1;
    for (uint32_t r = first_state; r <= states; ++r) {
        const uint32_t resident = configuration.bulk_sweep ? maximum : r * batch;
        const bool build = r == first_state || !paper_dynamic;
        const uint32_t n = build ? resident : batch;
        const uint32_t begin = build ? 0 : resident - batch;
        if (keys.num_elements < n) { keys.resize(n); values.resize(n); }
        fill_insert_batch<<<(n + threads - 1) / threads, threads>>>(
            keys.ptr(), values.ptr(), begin, n);
        fill_common_values<<<(n + threads - 1) / threads, threads>>>(values.ptr(), begin, n);
        const auto input = common_digest(keys.ptr(), n, digest);
        size_t free_bytes = 0, total_bytes = 0;
        PAPER_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (build) {
            index.reset();
            const auto times = measure_common(timer, [&] {
                index = std::make_unique<Index>();
                const size_t capacity = paper_tombstones && !configuration.skip_deletions
                    ? size_t(maximum) * 2 : maximum;
                paper_build(*index, keys.ptr(), n, capacity, free_bytes);
            });
            rows.add(configuration.bulk_sweep ? "bulk_build" : "build", r,
                     resident, n, "initial_keys", times, index->gpu_resident_bytes(), input);
        } else if constexpr (paper_dynamic) {
            const auto times = measure_common(timer, [&] {
                paper_prepare_updates<Index>(keys.ptr(), values.ptr(), n);
                index->insert(keys.ptr(), values.ptr(), n, 0);
            });
            rows.add("insert", r, resident, n, "distinct_growth", times,
                     index->gpu_resident_bytes(), input);
        }
        if (resident <= query_max || r == states || configuration.bulk_sweep)
            query_state(r, resident, "lookup");
        range_state(r, resident, "range_sum");
        std::cout << "COMMON_PROGRESS state=" << r << '/' << states << std::endl;
    }
    if constexpr (paper_dynamic) {
        if (!configuration.bulk_sweep && !configuration.skip_deletions) {
            for (uint32_t r = states; r > 1; --r) {
                const uint32_t resident = (r - 1) * batch;
                fill_insert_batch<<<(batch + threads - 1) / threads, threads>>>(
                    keys.ptr(), values.ptr(), resident, batch);
                const auto input = common_digest(keys.ptr(), batch, digest);
                // Keep the unsorted request for the post-delete check.
                PAPER_CUDA(cudaMemcpy(queries.ptr(), keys.ptr(), batch * sizeof(key_type), cudaMemcpyDeviceToDevice));
                const auto times = measure_common(timer, [&] {
                    paper_prepare_updates<Index>(keys.ptr(), nullptr, batch);
                    index->remove(keys.ptr(), batch, 0);
                });
                rows.add("delete", r, resident, batch, "reverse_growth", times,
                         index->gpu_resident_bytes(), input);
                const auto checked = lookup.lookup(*index, queries.ptr(), answers.ptr(), batch);
                errors.zero();
                validate_common_lookup<<<(batch + threads - 1) / threads, threads>>>(
                    answers.ptr(), batch, resident, false, 0, errors.ptr());
                validate();
                rows.add("lookup_deleted", r, resident, batch, "all_deleted", checked,
                         index->gpu_resident_bytes(), input);
                if (resident <= query_max || r == 2) query_state(r, resident, "lookup_after_delete");
                range_state(r, resident, "range_sum_after_delete");
            }
        }
    }
    const auto destruction = measure_common(timer, [&] { index.reset(); });
    rows.add("destroy", 0, 0, 0, "index_only", destruction, 0);
    std::ofstream complete(configuration.output_directory / "complete_common");
    complete << "ok\n";
}

void run_common_sweep(const options &configuration) {
    run_common_impl<selected_paper_backend>(configuration);
}
