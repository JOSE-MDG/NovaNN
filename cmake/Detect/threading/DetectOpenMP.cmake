#[=======================================================================[.rst:
.. module:: DetectOpenMP
   :synopsis: Detect OpenMP availability for C and C++.

Uses ``find_package(OpenMP COMPONENTS C CXX)`` to locate OpenMP
support for both C and C++ compilers.  Sets ``NOVA_HAS_OPENMP`` to
``1`` only if both components are found.

**Result variables:**

- ``NOVA_HAS_OPENMP`` — Set to ``1`` if both C and CXX OpenMP support
  are found, ``0`` otherwise.

**Early exit:**

Returns immediately if ``NOVA_HAS_OPENMP`` is already defined.
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
