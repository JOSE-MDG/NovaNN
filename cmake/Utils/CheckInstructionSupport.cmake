#[=======================================================================[.rst:
.. module:: CheckInstructionSupport
   :synopsis: SIMD instruction detection utility macro.

Provides the ``check_simd()`` macro used by all CPU detection modules
to probe compiler support for specific SIMD instruction sets at
configure time.

The macro compiles and runs a small C++ snippet with the given compiler
flags.  On success it sets the result variable to ``1`` and appends the
corresponding flags to the ``SIMD_FLAGS`` list, which is later consumed
by ``nova_configure_cpu_target()`` to enable instruction-set optimisations
on a per-target basis.

.. function:: check_simd(VAR TEST_FLAGS APPEND_FLAGS SNIPPET)

   Compile and execute a C++ snippet to test instruction support.

   The macro saves ``CMAKE_REQUIRED_FLAGS``, temporarily overrides it
   with ``TEST_FLAGS``, calls ``check_cxx_source_runs()``, and restores
   the original value.  If the snippet compiles and runs without error,
   ``VAR`` is set to ``1`` and ``APPEND_FLAGS`` are tokenised and
   appended to ``SIMD_FLAGS``.

   :param VAR:          Result variable name (set to ``1`` on success).
   :type VAR:           ``variable name``
   :param TEST_FLAGS:   Compiler flags required for the instruction set
                        (e.g. ``"-mavx2"``).
   :type TEST_FLAGS:    ``string``
   :param APPEND_FLAGS: Flags to append to ``SIMD_FLAGS`` on success
                        (e.g. ``"-mavx2"``).
   :type APPEND_FLAGS:  ``string``
   :param SNIPPET:      C++ source code that exercises the instruction.
   :type SNIPPET:       ``string``

   .. code-block:: cmake

      check_simd(HAS_AVX2 "-mavx2" "-mavx2" "
          #include <immintrin.h>
          int main() { __m256i a = _mm256_set1_epi32(1); return 0; }
      ")

   .. note::

      The ``SIMD_FLAGS`` variable must be initialised (e.g.
      ``set(SIMD_FLAGS "")``) before the first call to this macro.
#]=======================================================================]

include(CheckCXXSourceRuns)

macro(check_simd VAR TEST_FLAGS APPEND_FLAGS SNIPPET)
    set(_saved_flags "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${TEST_FLAGS}")
    check_cxx_source_runs("${SNIPPET}" ${VAR})
    set(CMAKE_REQUIRED_FLAGS "${_saved_flags}")
    if(${VAR})
        separate_arguments(_simd_flags UNIX_COMMAND "${APPEND_FLAGS}")
        list(APPEND SIMD_FLAGS ${_simd_flags})
    endif()
endmacro()
