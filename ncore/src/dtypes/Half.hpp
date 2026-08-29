/**
 * @file Half.hpp
 * @brief Thin wrapper exposing the FP16 software fallback for DTypes.cpp.
 *
 * Pulls in ncore::dtypes::detail::{fp16_ieee_from_fp32_value,
 * fp16_ieee_to_fp32_value} when native _Float16 is unavailable.
 */
#include <ncore/headeronly/dtypes/half.hh>
