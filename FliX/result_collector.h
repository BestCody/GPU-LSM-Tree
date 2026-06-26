// =============================================================================
// File: result_collector.h
// Author: Justus Henneberg
// Description: Defines expressions and constants used in FliX     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INFOSYS_RESULT_COLLECTOR_H
#define INFOSYS_RESULT_COLLECTOR_H

#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <string>
#include <sstream>
#include <utility>
#include <vector>


namespace rc {
namespace detail {

enum column_classification {
    parameter_column, measurement_column
};

template <typename value_type>
std::string stringify(value_type value) {
    std::stringstream converter;
    converter << std::setprecision(16) << value;
    return converter.str();
}

void error(int error_code, std::initializer_list<std::string> messages) {
    for (const auto& m : messages)
        std::cerr << m;
    std::exit(error_code);
}

struct result_entry {
    result_entry(std::string name, std::string value, column_classification cls)
        : name(std::move(name)), value(std::move(value)), cls(cls) {}
    std::string name;
    std::string value;
    column_classification cls;
};

}
}


namespace rc {

enum header_style {
    no_header, inline_header, first_line_header
};

enum column_padding_style {
    pad_columns, pack_columns
};

enum table_style {
    wide_form, long_form
};

class result_collector;

class result {
public:
    template <typename value_type>
    result& add_parameter(std::string name, value_type value) {
        entries.emplace_back(name, detail::stringify(std::move(value)), detail::parameter_column);
        return *this;
    }

    template <typename value_type>
    result& add_measurement(std::string name, value_type value) {
        entries.emplace_back(name, detail::stringify(std::move(value)), detail::measurement_column);
        return *this;
    }

    std::string to_string(bool inline_headers, const std::string& sep = ", ") const {
        bool is_first_entry = true;
        std::stringstream ss;
        for (const auto& e : entries) {
            if (!is_first_entry)
                ss << sep;
            is_first_entry = false;
            if (inline_headers)
                ss << e.name << "=";
            ss << e.value;
        }
        return ss.str();
    }

private:
    std::vector<detail::result_entry> entries;

    friend class result_collector;
};

class result_collector {
    std::vector<std::string> parameter_names;
    std::vector<std::string> measurement_names;
    std::unordered_map<std::string, size_t> column_widths;
    std::unordered_map<std::string, size_t> last_column_accesses;
    std::unordered_map<std::string, detail::column_classification> column_classifications;

    std::vector<result> rows;

    bool auto_save;
    bool was_saved;

    std::string description_column_name;
    std::string value_column_name;

public:
    explicit result_collector(
            bool auto_save = true,
            std::string description_column_name = "DESCRIPTION",
            std::string value_column_name = "VALUE"
    ) :
        auto_save(auto_save),
        was_saved(true),
        description_column_name(std::move(description_column_name)),
        value_column_name(std::move(value_column_name)) {}
    result_collector(const result_collector&) = delete;
    result_collector(result_collector&&) = delete;
    result_collector& operator=(const result_collector&) = delete;
    result_collector& operator=(result_collector&&) = delete;
    ~result_collector() {
        if (was_saved || !auto_save || rows.empty()) return;
        auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::ofstream file("autosave-" + std::to_string(timestamp) + ".csv");
        write_csv(file, first_line_header, long_form, pad_columns, ", ", "NULL");
    }

    void commit(const result& r) {
        for (const auto& entry : r.entries) {
            auto last_access = last_column_accesses.find(entry.name);
            if (last_access == last_column_accesses.end()) {
                // unknown column
                if (entry.cls == detail::parameter_column) {
                    parameter_names.push_back(entry.name);
                } else {
                    measurement_names.push_back(entry.name);
                }
                column_widths[entry.name] = entry.value.size();
                column_classifications[entry.name] = entry.cls;
            } else {
                // known column
                if (last_access->second == rows.size())
                    detail::error(9001, {"duplicate entry in record [", r.to_string(true), "]"});
                if (column_classifications[entry.name] != entry.cls && entry.cls == detail::parameter_column)
                    detail::error(9002, {"trying to add a parameter to measurement column '", entry.name, "'"});
                if (column_classifications[entry.name] != entry.cls && entry.cls == detail::measurement_column)
                    detail::error(9003, {"trying to add a measurement to parameter column '", entry.name, "'"});
                auto old_width = column_widths[entry.name];
                column_widths[entry.name] = std::max(old_width, entry.value.size());
            }
            last_column_accesses[entry.name] = rows.size();
        }
        rows.push_back(r);
        was_saved = false;
    }

    void write_csv(
            std::ostream& stream,
            const header_style hs,
            const table_style ts,
            const column_padding_style ps,
            const std::string& sep = ", ",
            const std::string& empty = "",
            const std::string& line_break = "\n"
    ) {
        auto all_column_names = parameter_names;
        auto locally_corrected_widths = column_widths;
        // correct column width to ensure room for empty entries
        for (const auto& column_name : parameter_names) {
            auto old_width = locally_corrected_widths[column_name];
            locally_corrected_widths[column_name] = std::max(old_width, empty.size());
        }
        size_t description_column_width = hs == first_line_header ? description_column_name.size() : 0;
        size_t measurement_column_width = hs == first_line_header ? value_column_name.size() : 0;
        if (ts == long_form) {
            all_column_names.insert(all_column_names.end(), {description_column_name, value_column_name});
            // the description column has to be wide enough to fit all measurement column names
            for (const auto& column_name : measurement_names) {
                description_column_width = std::max(description_column_width, column_name.size());
            }
            // the value column has to be wide enough to fit all measurement values
            for (const auto& column_name : measurement_names) {
                measurement_column_width = std::max(measurement_column_width, locally_corrected_widths[column_name]);
            }
        } else {
            all_column_names.insert(all_column_names.end(), measurement_names.begin(), measurement_names.end());
            // correct column width to ensure room for empty entries
            for (const auto& column_name : measurement_names) {
                auto old_width = locally_corrected_widths[column_name];
                locally_corrected_widths[column_name] = std::max(old_width, empty.size());
            }
        }

        // output the header
        if (hs == first_line_header) {
            for (const auto& column_name : all_column_names) {
                auto old_width = locally_corrected_widths[column_name];
                locally_corrected_widths[column_name] = std::max(old_width, column_name.size());
            }

            bool is_first_entry = true;
            for (const auto& column_name : all_column_names) {
                auto column_width = locally_corrected_widths[column_name];
                if (!is_first_entry)
                    stream << sep;
                is_first_entry = false;
                if (ps == pad_columns)
                    stream << std::string(column_width - column_name.size(), ' ');
                stream << column_name;
            }
            stream << line_break;
        }

        std::stringstream formatted_parameters_s;
        // do not write parameters to the output stream directly if we are dealing with long-form outputs
        auto& parameter_stream = ts == long_form ? formatted_parameters_s : stream;

        // output the rows
        for (const auto& current_row : rows) {
            bool is_first_entry = true;
            // handle parameters first
            for (const auto& column_name : parameter_names) {
                auto column_width = locally_corrected_widths[column_name];
                if (!is_first_entry)
                    parameter_stream << sep;
                is_first_entry = false;
                auto entry_it = std::find_if(current_row.entries.begin(), current_row.entries.end(),
                        [&column_name](const auto& e){return e.name == column_name;});
                const auto& entry = entry_it == current_row.entries.end() ? empty : entry_it->value;
                if (hs == inline_header)
                    parameter_stream << column_name << "=";
                if (ps == pad_columns)
                    parameter_stream << std::string(column_width - entry.size(), ' ');
                parameter_stream << entry;
            }

            if (ts == wide_form) {
                // for wide-form, just output the measurements
                for (const auto& column_name : measurement_names) {
                    auto column_width = locally_corrected_widths[column_name];
                    if (!is_first_entry)
                        stream << sep;
                    is_first_entry = false;
                    auto entry_it = std::find_if(current_row.entries.begin(), current_row.entries.end(),
                            [&column_name](const auto& e){return e.name == column_name;});
                    const auto& entry = entry_it == current_row.entries.end() ? empty : entry_it->value;
                    if (ps == pad_columns)
                        stream << std::string(column_width - entry.size(), ' ');
                    if (hs == inline_header)
                        stream << column_name << "=";
                    stream << entry;
                }
                stream << line_break;
            } else {
                // for long-form, output one line per measurement and re-use the parameters
                auto formatted_parameters = formatted_parameters_s.str();
                formatted_parameters_s.str("");

                for (const auto& entry : current_row.entries) {
                    if (entry.cls != detail::measurement_column) continue;
                    stream << formatted_parameters;
                    if (!formatted_parameters.empty())
                        stream << sep;
                    if (hs == inline_header)
                        stream << description_column_name << "=";
                    if (ps == pad_columns)
                        stream << std::string(description_column_width - entry.name.size(), ' ');
                    stream << entry.name << sep;
                    if (hs == inline_header)
                        stream << value_column_name << "=";
                    if (ps == pad_columns)
                        stream << std::string(measurement_column_width - entry.value.size(), ' ');
                    stream << entry.value << line_break;
                }
            }
        }
        stream << std::flush;
        was_saved = true;
    }

    void prevent_automatic_save() {
        was_saved = true;
    }
};

class auto_commit_result : public result {
    bool commit_on_destruction;
    result_collector* rc;

public:
    explicit auto_commit_result(result_collector& rc) : commit_on_destruction(true), rc(&rc) {}
    ~auto_commit_result() {
        if (commit_on_destruction) rc->commit(*this);
    }
    void prevent_automatic_commit() {
        commit_on_destruction = false;
    }
};

}

#endif
