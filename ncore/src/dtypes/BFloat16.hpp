/**
 * @file BFloat16.hpp
 * @brief Thin wrapper exposing the BF16 software fallback for DTypes.cpp.
 *
 * Pulls in ncore::dtypes::detail::{bits_from_f32, f32_from_bits} when
 * native __bf16 is unavailable.
 */
#include <ncore/headeronly/dtypes/bfloat16.hh>
