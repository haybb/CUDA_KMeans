cmake_minimum_required(VERSION 3.18)
project(proj_cuda LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

set (VECTORIZATION TRUE)

##############################################################
# Autodetect CUDA architecture(s)
##############################################################

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
message("-- CUDA Architecture(s): " ${CMAKE_CUDA_ARCHITECTURES})

##############################################################
# Build executables
##############################################################
include_directories(include)

add_subdirectory(third_party)

if (${VECTORIZATION})
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # using GCC/G++
        set (CMAKE_CXX_FLAGS "-O2 -ftree-vectorize -fopt-info-vec-optimized")
    endif()
endif()

add_subdirectory (src)
##############################################################
# Tests
##############################################################

add_subdirectory (test)

add_subdirectory (scripts)