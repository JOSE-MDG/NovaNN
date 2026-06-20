/**
 * @file repr_options.c
 * @brief implementation of the default configuration factory for tensor repr.
 *
 * @details
 * This module provides the standard initialization logic for formatting
 * options. By using a factory pattern (@ref repr_default_options()), NovaNN
 * ensures that all new representation parameters are initialized to sensible,
 * framework-standard values even as the module evolves.
 *
 * ## Architecture
 * - **PyTorch Compatibility**: Default values (e.g., precision=4,
 * threshold=1000) are chosen to mirror industry standards, ensuring a familiar
 * experience for research engineers.
 * - **Initialization Strategy**: Explicit field assignments ensure that no
 *   uninitialized data from the stack is used in the formatting pipeline.
 *
 * @see repr_options.h Structure definition and enums.
 */

#include <ncore/repr/repr_options.h>

/**
 * @brief Return a ReprOptions struct with library-standard defaults.
 *
 * @details
 * Initialises all visual parameters to their default states. Users should
 * call this before overriding specific fields for custom formatting.
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
