#[=======================================================================[.rst:
DetectPThreads
--------------

Detect POSIX threads (pthreads) support, falling back to native Win32
threads on Windows.

This module sets the following variables:

``NOVA_HAS_PTHREADS``
  ``1`` if pthreads was found, ``0`` otherwise.

``NOVA_HAS_WIN32_THREADS``
  ``1`` on Windows, ``0`` otherwise.

The module is idempotent: if ``NOVA_HAS_PTHREADS`` is already defined
the file returns immediately.

#]=======================================================================]

if(DEFINED NOVA_HAS_PTHREADS)
  return()
endif()

if(WIN32)
  set(NOVA_HAS_PTHREADS 0)
  set(NOVA_HAS_WIN32_THREADS 1)
  message(STATUS "Threading: using native Windows threads")
else()
  set(NOVA_HAS_WIN32_THREADS 0)
  find_package(Threads REQUIRED)

  if(CMAKE_USE_PTHREADS_INIT)
    set(NOVA_HAS_PTHREADS 1)
    message(STATUS "Threading: pthreads found")
  else()
    set(NOVA_HAS_PTHREADS 0)
    message(STATUS "Threading: pthreads NOT found (using generic Threads)")
  endif()
endif()
