/**
 * @file repr_options.h
 * @brief Configuration options for tensor string representation.
 *
 * @details
 * Declares @ref ReprOptions and the @ref ReprMode enumeration that
 * control the visual behavior of the tensor representation module.
 * Users can customize verbosity, summarization thresholds, numeric
 * precision, and platform-specific interpretations (e.g., boolean
 * or quantized values).
 *
 * Every field in @ref ReprOptions has a sensible, library-standard
 * default value provided by @ref repr_default_options().
 *
 * @see repr_context.h  Internal context built from these options.
 * @see tensor_repr.h   Top-level public API.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum ReprMode
 * @brief Display verbosity mode for tensor representation.
 *
 * @details
 * Determines the amount of auxiliary information appended to the
 * tensor data block.
 */
typedef enum {
  /**
   * @brief Standard display mode.
   *
   * Shows only the multidimensional data block, optionally followed
   * by a `dtype` suffix if the type is not `Float32`.
   */
  ReprModeNormal,

  /**
   * @brief Verbose diagnostic mode.
   *
   * Shows the data block followed by a detailed footer containing
   * dtype, shape, device placement, and autograd state.
   */
  ReprModeDebug,
} ReprMode;

/**
 * @struct ReprOptions
 * @brief Primary configuration structure for the representation module.
 *
 * @details
 * Holds all user-tunable parameters. Obtain an instance via
 * @ref repr_default_options() and then modify only the fields
 * required for the specific representation call.
 */
typedef struct {
  ReprMode mode;     ///< Formatting mode (`ReprModeNormal` or `ReprModeDebug`).
  size_t threshold;  ///< Max elements before truncation (summarization).
  size_t edge_items; ///< Elements to show per edge when truncated.
  size_t linewidth;  ///< Target line width for wrapping (reserved).
  int precision;     ///< Fixed decimal places for floating-point output.
  bool sci_mode;     ///< If `true`, forces scientific (`%e`) notation.
  bool sci_mode_auto; ///< If `true`, auto-enables sci-notation based on data.
  bool show_dequantized; ///< If `true`, appends `(float)` value for quantized
                         ///< types.
  bool is_bool; ///< If `true`, renders UnSigned8 as `"True"`/`"False"`.
} ReprOptions;

/**
 * @brief Return a default-initialized ReprOptions structure.
 *
 * @details
 * Initialises all fields to library-standard, framework-agnostic
 * values:
 * - `mode`: `ReprModeNormal`
 * - `threshold`: `1000`
 * - `edge_items`: `3`
 * - `precision`: `4`
 * - `sci_mode_auto`: `true`
 *
 * @return A correctly initialised @ref ReprOptions structure.
 *
 * @see ReprOptions
 */
ReprOptions repr_default_options(void);

#ifdef __cplusplus
}
#endif
