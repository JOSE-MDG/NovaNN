#[=======================================================================[.rst:
NovaNNReconfigureGuard
----------------------

Fail the configure when it is being re-run in a tight loop.

Ninja re-runs CMake whenever an input of the generated manifest
changes.  When the source tree keeps changing under the build, the
reconfigure never converges: Ninja repeats "Re-checking globbed
directories..." and "Re-running CMake..." without compiling anything.

This module records the timestamp of every successful configure in the
build tree.  When more than ``NOVA_RECONFIGURE_GUARD_MAX`` configures
happen within ``NOVA_RECONFIGURE_GUARD_WINDOW`` seconds, the configure
fails with a diagnostic naming the likely causes.

User Options
^^^^^^^^^^^^

``NOVA_RECONFIGURE_GUARD_WINDOW``
  Seconds within which consecutive configures count as a loop.  Default:
  ``60``.  Set it to ``0`` to disable the guard.  Every configure of the
  same build tree counts, manual ones included.

``NOVA_RECONFIGURE_GUARD_MAX``
  Number of consecutive configures inside the window that trips the
  guard.  Default: ``4``.

#]=======================================================================]

set(NOVA_RECONFIGURE_GUARD_WINDOW 60 CACHE STRING
    "Seconds within which consecutive configures count as a reconfigure loop")
set(NOVA_RECONFIGURE_GUARD_MAX 4 CACHE STRING
    "Consecutive configures inside the guard window that trip the guard")

#[=======================================================================[.rst:
.. command:: nova_guard_reconfigure_loop

  Fail the configure when it is being re-run in a tight loop:

  .. code-block:: cmake

    nova_guard_reconfigure_loop()

  The call reads the state left by the previous configure, increments the
  consecutive counter when the previous configure happened inside
  ``NOVA_RECONFIGURE_GUARD_WINDOW`` seconds, and aborts the configure
  when the counter exceeds ``NOVA_RECONFIGURE_GUARD_MAX``.

  The new state is written at the end of the directory scope, so a
  configure that fails earlier does not advance the counter.

#]=======================================================================]
function(nova_guard_reconfigure_loop)
  if(NOVA_RECONFIGURE_GUARD_WINDOW EQUAL 0)
    return()
  endif()

  set(_nova_guard_file
      "${CMAKE_BINARY_DIR}/CMakeFiles/nova_reconfigure_guard.txt")

  string(TIMESTAMP _nova_guard_now "%s" UTC)

  set(_nova_guard_previous "")
  set(_nova_guard_count 0)
  if(EXISTS "${_nova_guard_file}")
    file(READ "${_nova_guard_file}" _nova_guard_state)
    string(REGEX MATCH "([0-9]+) ([0-9]+)" _nova_guard_match
        "${_nova_guard_state}")
    if(_nova_guard_match)
      set(_nova_guard_previous "${CMAKE_MATCH_1}")
      set(_nova_guard_count "${CMAKE_MATCH_2}")
    endif()
  endif()

  set(_nova_guard_consecutive 1)
  if(_nova_guard_previous)
    math(EXPR _nova_guard_delta "${_nova_guard_now} - ${_nova_guard_previous}")
    if(_nova_guard_delta LESS_EQUAL NOVA_RECONFIGURE_GUARD_WINDOW)
      math(EXPR _nova_guard_consecutive "${_nova_guard_count} + 1")
    endif()
  endif()

  if(_nova_guard_consecutive GREATER NOVA_RECONFIGURE_GUARD_MAX)
    math(EXPR _nova_guard_elapsed "${_nova_guard_now} - ${_nova_guard_previous}")
    message(FATAL_ERROR
        "CMake reconfigure loop detected: ${_nova_guard_consecutive} "
        "configures in ${_nova_guard_elapsed} seconds.\n"
        "The build cannot make progress because CMake keeps being "
        "re-run: each reconfigure regenerates the manifest and the next "
        "Ninja pass finds it stale again.\n"
        "Likely causes:\n"
        "1. Files are added or removed under a directory watched by "
        "file(GLOB ... CONFIGURE_DEPENDS).  Ninja prints the affected "
        "files right after \"Re-checking globbed directories...\"; look "
        "for a \"-- GLOB mismatch!\" block in the build log.\n"
        "2. A file CMake depends on keeps a newer timestamp than the "
        "generated manifest.  The vcpkg manifest files (vcpkg.json, "
        "vcpkg-configuration.json) are the usual suspects: restoring them "
        "from an archive or checking them out updates their timestamp.\n"
        "3. The code generation step rewrites files inside the source "
        "tree that a glob matches.\n"
        "Stop whatever writes to the source tree and build again.\n"
        "The guard permits ${NOVA_RECONFIGURE_GUARD_MAX} configures per "
        "${NOVA_RECONFIGURE_GUARD_WINDOW} seconds, manual ones "
        "included: raise NOVA_RECONFIGURE_GUARD_MAX for a burst of "
        "them, or set NOVA_RECONFIGURE_GUARD_WINDOW to 0 to disable the "
        "guard."
    )
  endif()

  set_property(DIRECTORY PROPERTY NOVA_RECONFIGURE_GUARD_FILE
      "${_nova_guard_file}")
  set_property(DIRECTORY PROPERTY NOVA_RECONFIGURE_GUARD_STATE
      "${_nova_guard_now} ${_nova_guard_consecutive}")
  cmake_language(DEFER CALL nova_guard_reconfigure_loop_write)
endfunction()

#[=======================================================================[.rst:
.. command:: nova_guard_reconfigure_loop_write

  Persist the reconfigure counter for the next configure:

  .. code-block:: cmake

    nova_guard_reconfigure_loop_write()

  The state file, the timestamp and the counter are taken from the
  directory properties set by :command:`nova_guard_reconfigure_loop`.
  A deferred call expands its arguments when it runs, so the values have
  to travel through the directory scope.

#]=======================================================================]
function(nova_guard_reconfigure_loop_write)
  get_property(_nova_guard_file DIRECTORY
      PROPERTY NOVA_RECONFIGURE_GUARD_FILE)
  get_property(_nova_guard_state DIRECTORY
      PROPERTY NOVA_RECONFIGURE_GUARD_STATE)
  if(_nova_guard_file AND _nova_guard_state)
    file(WRITE "${_nova_guard_file}" "${_nova_guard_state}\n")
  endif()
endfunction()
