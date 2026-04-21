cmake_minimum_required(VERSION 3.20)
project(midas_integrate_gpu LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compute capability — Ampere (A100) + Hopper (H100) + Ada (L40S) cover
# every GPU the paper benchmarks on. Add entries as new architectures
# become relevant.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
endif()

include(FetchContent)

# ---------------------------------------------------------------------------
# External dependencies
# ---------------------------------------------------------------------------
find_package(OpenMP COMPONENTS C)

# NLopt — same FetchContent fallback as the other packages.
find_package(NLopt QUIET)
if(NOT NLopt_FOUND AND NOT TARGET NLopt::nlopt AND NOT TARGET NLOPT::NLOPT)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(NLOPT_PC QUIET nlopt)
    endif()
    if(NLOPT_PC_FOUND)
        add_library(NLopt::nlopt INTERFACE IMPORTED)
        target_include_directories(NLopt::nlopt INTERFACE ${NLOPT_PC_INCLUDE_DIRS})
        target_link_libraries(NLopt::nlopt INTERFACE ${NLOPT_PC_LIBRARIES})
    else()
        FetchContent_Declare(
            nlopt
            GIT_REPOSITORY https://github.com/stevengj/nlopt.git
            GIT_TAG v2.10.1
        )
        set(NLOPT_CXX OFF CACHE BOOL "" FORCE)
        set(NLOPT_PYTHON OFF CACHE BOOL "" FORCE)
        set(NLOPT_TESTS OFF CACHE BOOL "" FORCE)
        set(NLOPT_FORTRAN OFF CACHE BOOL "" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(nlopt)
        if(NOT TARGET NLopt::nlopt)
            add_library(NLopt::nlopt ALIAS nlopt)
        endif()
    endif()
endif()
if(TARGET NLOPT::NLOPT AND NOT TARGET NLopt::nlopt)
    add_library(NLopt::nlopt ALIAS NLOPT::NLOPT)
endif()

# ---------------------------------------------------------------------------
# Generated version header
# ---------------------------------------------------------------------------
if(NOT DEFINED PACKAGE_VERSION)
    set(PACKAGE_VERSION "0.1.0")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/midas_version.h"
"#ifndef MIDAS_VERSION_H\n#define MIDAS_VERSION_H\n#define MIDAS_VERSION \"${PACKAGE_VERSION}\"\n#define MIDAS_GIT_HASH \"midas-integrate-gpu\"\n#define MIDAS_GIT_DATE \"\"\n#define MIDAS_VERSION_STRING \"midas-integrate-gpu v\" MIDAS_VERSION\n#endif\n")

# ---------------------------------------------------------------------------
# MIDASIntegratorGPU — CUDA streaming integrator.
# ---------------------------------------------------------------------------
add_executable(MIDASIntegratorGPU
    src/cuda/IntegratorFitPeaksGPUStream.cu
    src/c/PeakFit.c
    src/c/PeakFitIO.c
    src/c/MIDAS_Math.c
)
target_include_directories(MIDASIntegratorGPU PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/src/c
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_link_libraries(MIDASIntegratorGPU PRIVATE m NLopt::nlopt)
set_target_properties(MIDASIntegratorGPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
if(OpenMP_C_FOUND)
    target_link_libraries(MIDASIntegratorGPU PRIVATE OpenMP::OpenMP_C)
endif()

# pthreads — the .cu uses pthread_* directly.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(MIDASIntegratorGPU PRIVATE Threads::Threads)

# ---------------------------------------------------------------------------
# Install — binary lives alongside midas_integrate_gpu/_binaries.py.
# ---------------------------------------------------------------------------
install(TARGETS MIDASIntegratorGPU
    RUNTIME DESTINATION midas_integrate_gpu/_bin
)
