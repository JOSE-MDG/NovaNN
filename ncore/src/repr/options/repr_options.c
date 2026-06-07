/**
 * @file repr_options.c
 * @brief Default options for tensor string representation.
 *
 * @details
 * Implements repr_default_options() which returns a fully initialised
 * ReprOptions struct with PyTorch-compatible defaults:
 *   - Normal mode (no debug footer).
 *   - Threshold of 1000 elements before summarisation.
 *   - 3 edge items when summarised.
 *   - 4 decimal places for floats.
 *   - Scientific-notation auto-detection enabled.
 *   - Quantized dequantized values shown.
 *   - Bool interpretation disabled.
 */

#include <ncore/repr/repr_options.h>

/**
 * @brief Return a ReprOptions struct with sensible defaults.
 *
 * All fields are set to PyTorch-compatible values.  Callers may
 * override any field after obtaining the result.
 *
 * @return Default-initialised ReprOptions.
 */
ReprOptions repr_default_options(void) {
  ReprOptions opts;
  opts.mode = REPR_MODE_NORMAL;
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
