#[=======================================================================[.rst:
DetectPThreads
--------------

Detect POSIX threads (pthreads) support.  Uses
:command:`find_package` with the ``Threads`` module to resolve the
threading library.

This module sets the following variables:

``NOVA_HAS_PTHREADS``
  ``1`` if pthreads was found, ``0`` otherwise.

The module is idempotent: if ``NOVA_HAS_PTHREADS`` is already defined
the file returns immediately.

#]=======================================================================]

if(DEFINED NOVA_HAS_PTHREADS)
  return()
endif()

find_package(Threads REQUIRED)

if(CMAKE_USE_PTHREADS_INIT)
  set(NOVA_HAS_PTHREADS 1)
  message(STATUS "Threading: pthreads found")
else()
  set(NOVA_HAS_PTHREADS 0)
  message(STATUS "Threading: pthreads NOT found (using generic Threads)")
endif()
