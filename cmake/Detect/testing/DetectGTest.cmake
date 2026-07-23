#[=======================================================================[.rst:
DetectGTest
-----------

Detect GoogleTest (GTest) testing framework.  Uses
:command:`find_package` with the ``GTest`` module (Config mode
first, fallback to Module mode).

This module sets the following variables:

``NOVA_HAS_GTEST``
  ``1`` if GTest was found, ``0`` otherwise.

Defined Functions
^^^^^^^^^^^^^^^^^

.. command:: nova_configure_gtest_target

  Link ``GTest::gtest`` and ``GTest::gtest_main`` to a target so that it
  can use the GoogleTest framework::

    if(NOVA_HAS_GTEST)
      nova_configure_gtest_target(my_test)
    endif()

The detection re-runs on every configure so that GTest is picked up if
installed after the initial configuration.

#]=======================================================================]

find_package(GTest QUIET CONFIG)

if(NOT GTest_FOUND)
  find_package(GTest QUIET MODULE)
endif()

if(GTest_FOUND)
  set(NOVA_HAS_GTEST 1 CACHE INTERNAL "")

  function(nova_configure_gtest_target TARGET)
    target_link_libraries(${TARGET} PRIVATE GTest::gtest GTest::gtest_main)
    target_include_directories(${TARGET} PRIVATE ${GTEST_INCLUDE_DIRS})
  endfunction()
endif()