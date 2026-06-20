#[=======================================================================[.rst:
DetectOpenMP
------------

Detect OpenMP parallelism support for C and C++ compilers.  Uses
:command:`find_package` with the ``OpenMP`` module.

This module sets the following variables:

``NOVA_HAS_OPENMP``
  ``1`` if both ``OpenMP_C`` and ``OpenMP_CXX`` were found, ``0``
  otherwise.

The module is idempotent: if ``NOVA_HAS_OPENMP`` is already defined
the file returns immediately.

#]=======================================================================]

if(DEFINED NOVA_HAS_OPENMP)
  return()
endif()

find_package(OpenMP COMPONENTS C CXX)

if(OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
  set(NOVA_HAS_OPENMP 1)
  message(STATUS "Threading: OpenMP ${OpenMP_VERSION} found")
else()
  set(NOVA_HAS_OPENMP 0)
  message(STATUS "Threading: OpenMP NOT found")
endif()
