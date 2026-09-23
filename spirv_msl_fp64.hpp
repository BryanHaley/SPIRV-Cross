/*
 * Copyright 2024 SPIRV-Cross Authors
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#ifndef SPIRV_CROSS_MSL_FP64_HPP
#define SPIRV_CROSS_MSL_FP64_HPP

#include <string>
#include <stdint.h>


// SPIRV_CROSS_NAMESPACE is provided by spirv_msl.hpp when included from there.
#ifndef SPIRV_CROSS_NAMESPACE
#define SPIRV_CROSS_NAMESPACE spirv_cross
#endif

namespace SPIRV_CROSS_NAMESPACE
{

// FP64Mode is declared in spirv_msl.hpp alongside CompilerMSL::Options.
// For standalone compilation we provide the definition here too.
enum class FP64Mode : uint8_t
{
	Disabled,
	DoubleFloat,
	TripleFloat
};

// Returns the MSL prelude string that must be emitted before any shader code.
const char *get_fp64_msl_prelude(FP64Mode mode);

// The vector and matrix half of the prelude. Emitted right after the scalar
// part, and only when the shader actually uses a dvec or a dmat.
std::string get_fp64_composite_prelude(FP64Mode mode, bool want_vectors, bool want_matrices);

// Name of the MSL type for a vecsize x columns emulated double.
std::string get_fp64_composite_type_name(FP64Mode mode, uint32_t vecsize, uint32_t columns);

// Returns the MSL scalar type name for the emulated double.
const char *get_fp64_type_name(FP64Mode mode);

// Splits an IEEE-754 binary64 value into its float components.
void split_double_df(double value, float &hi, float &lo);
void split_double_tf(double value, float &x, float &y, float &z);

// Returns the MSL function name for a given scalar fp64 operation.
const char *get_fp64_func_name(const char *op, FP64Mode mode);

// Vector helpers
const char *get_fp64_dot_name(int vecsize, FP64Mode mode);
const char *get_fp64_length_name(int vecsize, FP64Mode mode);
const char *get_fp64_distance_name(int vecsize, FP64Mode mode);
const char *get_fp64_normalize_name(int vecsize, FP64Mode mode);

// Buffer decode/encode
const char *get_fp64_decode_name(FP64Mode mode);
const char *get_fp64_encode_name(FP64Mode mode);

// Conversions: from/to can be "float", "int", "uint", "double"
const char *get_fp64_convert_name(const char *from, const char *to, FP64Mode mode);

const char *get_fp64_sign_name(FP64Mode mode);

} // namespace SPIRV_CROSS_NAMESPACE

#endif
