function(vcpkg_find_rocm)
    cmake_parse_arguments(PARSE_ARGV 0 vfr "" "OUT_ROCM_ROOT;OUT_ROCM_VERSION" "")

    if(NOT vfr_OUT_ROCM_ROOT)
        message(FATAL_ERROR "vcpkg_find_rocm() requires an OUT_ROCM_ROOT argument")
    endif()

    set(ROCM_REQUIRED_VERSION "7.0")

    # Standard ROCm environment variables and default install locations
    set(ROCM_PATHS
            ENV ROCM_PATH
            ENV HIP_PATH
            ENV ROCM_ROOT
            /opt/rocm)

    # On Linux, hipcc is the compiler driver
    find_program(HIPCC
        NAMES hipcc
        PATHS
        ${ROCM_PATHS}
        PATH_SUFFIXES bin
        DOC "ROCm SDK location."
        NO_DEFAULT_PATH
    )

    # If hipcc wasn't found via standard paths, try a fallback glob for versioned folders
    if(NOT HIPCC)
        file(GLOB possible_paths "/opt/rocm-*")
        set(FOUND_PATH)
        foreach (p ${possible_paths})
            string(REGEX MATCH "[0-9]\\.[0-9]+(\\.[0-9]+)?" p_version ${p})
            if (IS_DIRECTORY ${p} AND p_version)
                if (p_version VERSION_GREATER_EQUAL ROCM_REQUIRED_VERSION)
                    set(FOUND_PATH ${p})
                    break()
                endif()
            endif()
        endforeach()

        if(FOUND_PATH)
            find_program(HIPCC
                NAMES hipcc
                PATHS ${FOUND_PATH}
                PATH_SUFFIXES bin
                NO_DEFAULT_PATH
            )
        endif()
    endif()

    set(error_code 1)
    if (HIPCC)
        execute_process(
            COMMAND ${HIPCC} --version
            OUTPUT_VARIABLE HIPCC_OUTPUT
            RESULT_VARIABLE error_code)
    endif()

    if (error_code OR NOT HIPCC)
        message(STATUS "Executing ${HIPCC} --version resulted in error: ${error_code}")
        message(FATAL_ERROR "Could not find ROCm. Before continuing, please download and install ROCm (v${ROCM_REQUIRED_VERSION} or higher) from:"
                            "\n    https://rocm.docs.amd.com/en/latest/deploy/linux/install.html\n")
    endif()

    # Sample hipcc --version output:
    # HIP version: 7.0.0
    # AMD clang version 17.0.0...
    string(REGEX MATCH "HIP version: ([0-9]+)\\.([0-9]+)\\.([0-9]+)" ROCM_VERSION ${HIPCC_OUTPUT})

    if(NOT CMAKE_MATCH_1)
        # Fallback regex for older or different output formats
        string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" ROCM_VERSION ${HIPCC_OUTPUT})
    endif()

    message(STATUS "Found ROCm ${ROCM_VERSION}")

    set(ROCM_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set(ROCM_VERSION_MINOR "${CMAKE_MATCH_2}")
    set(ROCM_VERSION_MAJOR_MINOR "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")

    if (ROCM_VERSION_MAJOR_MINOR VERSION_LESS ROCM_REQUIRED_VERSION)
        message(FATAL_ERROR "ROCm v${ROCM_VERSION_MAJOR_MINOR} found, but v${ROCM_REQUIRED_VERSION} is required. Please download and install a more recent version of ROCm from:"
                            "\n    https://rocm.docs.amd.com/en/latest/deploy/linux/install.html\n")
    endif()

    # Derive ROCm root from hipcc location (e.g., /opt/rocm-7.0.0/bin/hipcc -> /opt/rocm-7.0.0)
    get_filename_component(ROCM_ROOT "${HIPCC}" DIRECTORY)
    get_filename_component(ROCM_ROOT "${ROCM_ROOT}" DIRECTORY)

    set(${vfr_OUT_ROCM_ROOT} "${ROCM_ROOT}" PARENT_SCOPE)
    if(DEFINED vfr_OUT_ROCM_VERSION)
        set(${vfr_OUT_ROCM_VERSION} "${ROCM_VERSION_MAJOR_MINOR}" PARENT_SCOPE)
    endif()
endfunction()
