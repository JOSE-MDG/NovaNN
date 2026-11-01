/**
 * @file repr_options.h
 * @brief Configuration options for tensor string representation.
 *
 * @details
 * Every field gets a library-standard default from
 * @ref repr_default_options().
 *
 * @see repr_context.h  Internal context built from these options.
 * @see tensor_repr.h   Top-level public API.
 */

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum ReprMode
 * @brief Display verbosity mode for tensor representation.
 */
typedef enum {
  /**
   * @brief Standard display mode.
   *
   * Shows only the multidimensional data block, optionally followed
   * by a @c dtype suffix if the type is not @c Float32.
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
 * Obtain an instance via @ref repr_default_options(), then modify
 * only the fields needed for the specific call.
 */
typedef struct {
  ReprMode mode; ///< Formatting mode (@c ReprModeNormal or @c ReprModeDebug).
  size_t threshold;      ///< Max elements before truncation. Exceeding it,
                         ///< or any single dimension exceeding twice
                         ///< @c edge_items, selects truncated rendering.
  size_t edge_items;     ///< Elements to show per edge when truncated. The
                         ///< builder may tighten this to bound total output
                         ///< on high-rank tensors; never loosened.
  int precision;         ///< Fixed decimal places for floating-point output.
  bool sci_mode;         ///< If @c true, forces scientific (@c %e) notation.
  bool sci_mode_auto;    ///< If @c true, enables sci-notation when the data
                         ///< range demands it (see context builder).
  bool show_dequantized; ///< If @c true, appends @c (float) value for quantized
                         ///< types.
  bool is_bool; ///< If @c true, renders UnSigned8 as @c "True"/@c "False".
} ReprOptions;

/**
 * @brief Return a default-initialized ReprOptions structure.
 *
 * @details
 * Initializes all fields to library-standard, framework-agnostic
 * values:
 * @li @c mode: @c ReprModeNormal
 * @li @c threshold: @c 1000
 * @li @c edge_items: @c 3
 * @li @c precision: @c 4
 * @li @c sci_mode_auto: @c true
 *
 * @return A default-initialized @ref ReprOptions structure.
 *
 * @see ReprOptions
 */
ReprOptions repr_default_options(void);

#ifdef __cplusplus
}
#endif
