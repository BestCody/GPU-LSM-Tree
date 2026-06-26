// =============================================================================
// File: definitions_fine_granular.cuh
// Author: Justus Henneberg
// Description: Implements definitions_fine_granular     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef DEFINITIONS_FINE_GRANULAR_H
#define DEFINITIONS_FINE_GRANULAR_H

#include "definitions.cuh"

namespace fg {

constexpr uint32_t x_bits = 23;
constexpr uint32_t y_bits = 23;
constexpr uint32_t z_bits = 18;

constexpr key64 x_mask = key64((size_t{1} << x_bits) - 1);
constexpr key64 y_mask = key64((size_t{1} << y_bits) - 1);

constexpr float eps = 0.5;

HOSTQUALIFIER DEVICEQUALIFIER INLINEQUALIFIER
float uint32_as_float(uint32_t i) {
    return static_cast<float>(i) + 0.5;
}

}

#endif
