# This package doesn't install ROCm. It instead verifies that ROCm is installed.
# Other packages can depend on this package to declare a dependency on ROCm.
# If this package is installed, we assume that ROCm is properly installed.

# note: this port is designed to follow the same pattern as the CUDA port.

include("${CMAKE_CURRENT_LIST_DIR}/vcpkg_find_rocm.cmake")

vcpkg_find_rocm(OUT_ROCM_ROOT ROCM_ROOT)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/vcpkg-port-config.cmake" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
file(COPY "${CMAKE_CURRENT_LIST_DIR}/vcpkg_find_rocm.cmake" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
file(INSTALL "${VCPKG_ROOT_DIR}/LICENSE.txt" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)

set(VCPKG_POLICY_CMAKE_HELPER_PORT enabled)
