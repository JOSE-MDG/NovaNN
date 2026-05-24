Threading Detection
===================

.. contents::
   :local:
   :depth: 1

----------

DetectPThreads
--------------

:file: ``cmake/detect/threading/DetectPThreads.cmake``

Detects POSIX threads via ``find_package(Threads REQUIRED)``.

Guarded against multiple inclusion (returns early if ``NOVA_HAS_PTHREADS``
is already defined).

Variables set:
   - ``NOVA_HAS_PTHREADS`` — 1 if pthreads are available, 0 otherwise

Targets exposed:
   - ``Threads::Threads`` — always available after this module runs

Usage::

   include(detect/threading/DetectPThreads)

Source
~~~~~~

.. literalinclude:: ../../cmake/detect/threading/DetectPThreads.cmake
   :language: cmake
   :linenos:

----------

DetectOpenMP
------------

:file: ``cmake/detect/threading/DetectOpenMP.cmake``

Detects OpenMP support for both C and C++ via
``find_package(OpenMP COMPONENTS C CXX)``.

Guarded against multiple inclusion.

Variables set:
   - ``NOVA_HAS_OPENMP`` — 1 if OpenMP is available for both C and CXX,
     0 otherwise

Targets exposed (conditional):
   - ``OpenMP::OpenMP_C``
   - ``OpenMP::OpenMP_CXX``

Usage::

   include(detect/threading/DetectOpenMP)

Source
~~~~~~

.. literalinclude:: ../../cmake/detect/threading/DetectOpenMP.cmake
   :language: cmake
   :linenos:
