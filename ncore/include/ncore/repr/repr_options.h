/**
 * @file repr_options.h
 * @brief Options and mode enum for tensor string representation.
 *
 * @details
 * ReprOptions controls every aspect of tensor formatting: display mode
 * (normal vs. debug), summarisation thresholds, line width, floating-
 * point precision, scientific-notation override, bool interpretation,
 * and quantized-value display.
 *
 * Every field has a sensible default; call repr_default_options() to
 * obtain a correctly initialised struct, then override specific fields.
 *
 * @see repr_context.h  Context built from these options.
 * @see tensor_repr.h   Top-level API consuming ReprOptions.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum ReprMode
 * @brief Display verbosity mode.
 *
 * REPR_MODE_NORMAL shows data values plus a dtype suffix only when the
 * dtype is not the default (Float32).  REPR_MODE_DEBUG always appends
 * dtype, shape, device, and gradient information.
 */
typedef enum {
  REPR_MODE_NORMAL, ///< Data + optional dtype suffix.
  REPR_MODE_DEBUG,  ///< Data + full metadata footer.
} ReprMode;

/**
 * @struct ReprOptions
 * @brief Configuration struct for tensor repr.
 *
 * All fields are initialised by repr_default_options().  Callers may
 * override any field after obtaining the defaults.
 */
typedef struct {
  ReprMode mode;         ///< NORMAL or DEBUG mode.
  size_t threshold;      ///< Element count above which summarisation kicks in.
                         ///< Default: 1000.
  size_t edge_items;     ///< Elements shown per edge when summarised.
                         ///< Default: 3.
  size_t linewidth;      ///< Soft maximum line width (reserved).  Default: 80.
  int precision;         ///< Number of decimal places for floats.  Default: 4.
  bool sci_mode;         ///< Force scientific notation.  Default: false.
  bool sci_mode_auto;    ///< Auto-detect sci notation (PyTorch heuristic).
                         ///< Default: true.
  bool show_dequantized; ///< Append dequantized value for qint types.
                         ///< Default: true.
  bool is_bool;          ///< Treat UnSigned8 values as True/False.
                         ///< Default: false.
} ReprOptions;

/**
 * @brief Return a ReprOptions struct with sensible defaults.
 *
 * Use this function instead of zero-initialising so that future fields
 * added to the struct are automatically covered.
 *
 * @return Default-initialised ReprOptions.
 */
ReprOptions repr_default_options(void);

#ifdef __cplusplus
}
#endif
