#[=======================================================================[.rst:
.. module:: DetectPThreads
   :synopsis: Detect POSIX threads (pthreads) availability.

Uses ``find_package(Threads REQUIRED)`` to locate the system threading
library.  If pthreads is the threading implementation, sets
``NOVA_HAS_PTHREADS`` to ``1``; otherwise sets it to ``0``.

**Result variables:**

- ``NOVA_HAS_PTHREADS`` — Set to ``1`` if pthreads is the threading
  backend, ``0`` otherwise.

**Early exit:**

Returns immediately if ``NOVA_HAS_PTHREADS`` is already defined,
allowing the module to be included multiple times without error.
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
