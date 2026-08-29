/**
 * @file repr_options.c
 * @brief Default configuration factory for tensor representation.
 *
 * @details
 * Implements @ref repr_default_options(), which returns a
 * @ref ReprOptions struct initialized to library-standard values.
 *
 * @see repr_options.h  Structure definition and enums.
 * @see repr_context.h  Context built from these options.
 */

#include <ncore/repr/repr_options.h>

/**
 * @brief Return a ReprOptions struct with library-standard defaults.
 *
 * @details
 * Initialises all visual parameters to their default states. Users
 * should call this before overriding specific fields for custom
 * formatting.
 *
 * @return A fully initialised @ref ReprOptions structure.
 */
ReprOptions repr_default_options(void) {
  ReprOptions opts;
  opts.mode = ReprModeNormal;
  opts.threshold = 1000;
  opts.edge_items = 3;
  opts.linewidth = 80;
  opts.precision = 4;
  opts.sci_mode = false;
  opts.sci_mode_auto = true;
  opts.show_dequantized = true;
  opts.is_bool = false;
  return opts;
}
