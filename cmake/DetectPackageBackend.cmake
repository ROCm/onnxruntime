# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

# Detect the packaging backend for ONNX Runtime ROCm builds.
#
# detect_package_backend() sets ORT_PACKAGE_BACKEND as a cache variable:
#   "therock"  - TheRock environment (amdrocm-xxx deb/rpm packages)
#   "default"  - Traditional ROCm with deb/rpm packages
#
# Preferred usage (explicit):
#   cmake -DORT_PACKAGE_BACKEND=therock -DORT_THEROCK_GPU_ARCH=gfx120x ..
#
# If ORT_PACKAGE_BACKEND is not set, falls back to auto-detection via
# dpkg/rpm to check for installed amdrocm-runtime packages.
#
# When ORT_PACKAGE_BACKEND=therock, ORT_THEROCK_GPU_ARCH must be set
# to the target GPU architecture family that follows TheRock packaging requirements.

function(_detect_therock_via_package_manager)
    set(_found FALSE)
    if(NOT WIN32)
        find_program(_ort_dpkg_exe dpkg)
        if(_ort_dpkg_exe)
            execute_process(
                COMMAND ${_ort_dpkg_exe} -s amdrocm-runtime
                RESULT_VARIABLE _result
                OUTPUT_QUIET ERROR_QUIET
            )
            if(_result EQUAL 0)
                set(_found TRUE)
            endif()
        endif()
        if(NOT _found)
            find_program(_ort_rpm_exe rpm)
            if(_ort_rpm_exe)
                execute_process(
                    COMMAND ${_ort_rpm_exe} -q amdrocm-runtime
                    RESULT_VARIABLE _result
                    OUTPUT_QUIET ERROR_QUIET
                )
                if(_result EQUAL 0)
                    set(_found TRUE)
                endif()
            endif()
        endif()
        unset(_ort_dpkg_exe CACHE)
        unset(_ort_rpm_exe CACHE)
    endif()
    set(_ORT_THEROCK_DETECTED ${_found} PARENT_SCOPE)
endfunction()

function(detect_package_backend)
    if(NOT DEFINED CACHE{ORT_PACKAGE_BACKEND})
        _detect_therock_via_package_manager()
        if(_ORT_THEROCK_DETECTED)
            set(_default_backend "therock")
            message(STATUS "ORT package backend auto-detected: therock (amdrocm-runtime found)")
            message(STATUS "  Hint: prefer explicit -DORT_PACKAGE_BACKEND=therock -DORT_THEROCK_GPU_ARCH=<arch>")
        else()
            set(_default_backend "default")
        endif()
        set(ORT_PACKAGE_BACKEND "${_default_backend}" CACHE STRING
            "Packaging backend: 'default' for traditional ROCm, 'therock' for TheRock amdrocm packages")
    endif()

    set_property(CACHE ORT_PACKAGE_BACKEND PROPERTY STRINGS "default" "therock")

    set(_valid_backends "default" "therock")
    if(NOT ORT_PACKAGE_BACKEND IN_LIST _valid_backends)
        message(FATAL_ERROR
            "ORT_PACKAGE_BACKEND='${ORT_PACKAGE_BACKEND}' is not valid. "
            "Must be one of: ${_valid_backends}")
    endif()

    if(ORT_PACKAGE_BACKEND STREQUAL "therock")
        if(DEFINED ENV{GPU_ARCH_FOR_THEROCK})
            set(_default_gpu_arch "$ENV{GPU_ARCH_FOR_THEROCK}")
        else()
            set(_default_gpu_arch "")
        endif()
        set(ORT_THEROCK_GPU_ARCH "${_default_gpu_arch}" CACHE STRING
            "TheRock GPU architecture family suffix (e.g. gfx120x)")

        message(STATUS "ORT package backend: therock (GPU arch: ${ORT_THEROCK_GPU_ARCH})")
    else()
        message(STATUS "ORT package backend: default (traditional ROCm)")
    endif()
endfunction()
