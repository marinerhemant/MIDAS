cmake_minimum_required(VERSION 3.20)
project(midas_auto_calibrate LANGUAGES C)

# Build with the same standards as the main MIDAS tree so behaviour matches.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)

# ---------------------------------------------------------------------------
# External dependencies
# ---------------------------------------------------------------------------
# OpenMP is optional; CalibrantIntegratorOMP uses #pragma omp when available.
find_package(OpenMP COMPONENTS C)

# HDF5 + libTIFF are required by FileReader.c. Use find_package (system libs,
# typically installed via apt/dnf/brew or via cibuildwheel's before-all).
# For local dev against an existing MIDAS build, point CMAKE_PREFIX_PATH at
# $MIDAS/build so find_package locates them: e.g.
#   export CMAKE_PREFIX_PATH="$HOME/opt/MIDAS/build:$CMAKE_PREFIX_PATH"
find_package(HDF5 REQUIRED COMPONENTS C HL)
find_package(TIFF REQUIRED)

# NLopt: try system first (brew / apt / dnf), then FetchContent + in-tree build.
# Matches the fallback pattern in the main MIDAS cmake/deps/NLOPT_custom.cmake.
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
        # Pinned tag for reproducibility; bump intentionally.
        message(STATUS "NLopt not found on system — building via FetchContent (pinned).")
        set(NLOPT_CXX OFF CACHE BOOL "" FORCE)
        set(NLOPT_PYTHON OFF CACHE BOOL "" FORCE)
        set(NLOPT_OCTAVE OFF CACHE BOOL "" FORCE)
        set(NLOPT_MATLAB OFF CACHE BOOL "" FORCE)
        set(NLOPT_GUILE OFF CACHE BOOL "" FORCE)
        set(NLOPT_SWIG OFF CACHE BOOL "" FORCE)
        set(NLOPT_TESTS OFF CACHE BOOL "" FORCE)
        set(NLOPT_FORTRAN OFF CACHE BOOL "" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

        # NLopt 2.10.x raised its CMake floor above the CMake-4 drop point.
        FetchContent_Declare(
            nlopt
            GIT_REPOSITORY https://github.com/stevengj/nlopt.git
            GIT_TAG v2.10.1
        )
        FetchContent_MakeAvailable(nlopt)

        # Create the NLopt::nlopt alias our targets link against.
        if(NOT TARGET NLopt::nlopt)
            add_library(NLopt::nlopt ALIAS nlopt)
        endif()
    endif()
endif()

# Support both CMake target-name conventions (NLopt::nlopt vs NLOPT::NLOPT).
if(TARGET NLOPT::NLOPT AND NOT TARGET NLopt::nlopt)
    add_library(NLopt::nlopt ALIAS NLOPT::NLOPT)
endif()
if(NOT TARGET NLopt::nlopt)
    message(FATAL_ERROR "Could not locate or build NLopt.")
endif()

# ---------------------------------------------------------------------------
# Generated version header
# ---------------------------------------------------------------------------
# CalibrantIntegratorOMP.c includes "midas_version.h", which in the main MIDAS
# tree is generated from git. We regenerate a stripped version here from the
# package version so the wheel carries a stable version string.
if(NOT DEFINED PACKAGE_VERSION)
    set(PACKAGE_VERSION "0.1.0")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/midas_version.h"
"#ifndef MIDAS_VERSION_H\n#define MIDAS_VERSION_H\n#define MIDAS_VERSION \"${PACKAGE_VERSION}\"\n#define MIDAS_GIT_HASH \"midas-auto-calibrate\"\n#define MIDAS_GIT_DATE \"\"\n#define MIDAS_VERSION_STRING \"midas-auto-calibrate v\" MIDAS_VERSION\n#endif\n")

# ---------------------------------------------------------------------------
# Shared integration static lib (matches FF_HEDM/CMakeLists.txt target
# `midas_integration`). Re-exported to midas-integrate via installed headers.
# ---------------------------------------------------------------------------
add_library(midas_integration STATIC
    src/c/MapperCore.c
    src/c/ImageUtils.c
    src/c/IntegrationCore.c
    src/c/CalibrationCore.c
    src/c/PeakFit.c
)
target_include_directories(midas_integration PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/c
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_link_libraries(midas_integration PUBLIC m)
if(OpenMP_C_FOUND)
    target_link_libraries(midas_integration PUBLIC OpenMP::OpenMP_C)
endif()
target_link_libraries(midas_integration PUBLIC NLopt::nlopt)

# ---------------------------------------------------------------------------
# Calibrant executable — ships inside the wheel at
# midas_auto_calibrate/_bin/MIDASCalibrant (located by _binaries.py at runtime).
# ---------------------------------------------------------------------------
add_executable(MIDASCalibrant
    src/c/CalibrantIntegratorOMP.c
    src/c/DetectorGeometry.c
    src/c/Panel.c
    src/c/FileReader.c
    src/c/MIDAS_Math.c
    src/c/MIDAS_ParamParser.c
    src/c/CalibPeakFit.c
)
target_include_directories(MIDASCalibrant PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/c
    ${CMAKE_CURRENT_BINARY_DIR}/generated
    ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(MIDASCalibrant PRIVATE
    midas_integration
    ${HDF5_C_LIBRARIES} ${HDF5_HL_LIBRARIES}
    TIFF::TIFF
)
if(OpenMP_C_FOUND)
    target_link_libraries(MIDASCalibrant PRIVATE OpenMP::OpenMP_C)
endif()

# ---------------------------------------------------------------------------
# GetHKLList helper — MIDASCalibrant system()-calls this for hkls.csv. Ships
# alongside MIDASCalibrant so the executable-dir fallback in midas_paths.h
# finds it inside the wheel's _bin/ directory.
# ---------------------------------------------------------------------------
add_executable(GetHKLList
    src/c/GetHKLList.c
    src/c/sgclib.c
    src/c/sgfind.c
    src/c/sghkl.c
    src/c/sgsi.c
    src/c/sgio.c
    src/c/MIDAS_ParamParser.c
)
target_include_directories(GetHKLList PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/c
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_link_libraries(GetHKLList PRIVATE m)

# ---------------------------------------------------------------------------
# Install — scikit-build-core places these inside the wheel's package tree.
# ---------------------------------------------------------------------------
install(TARGETS MIDASCalibrant GetHKLList
    RUNTIME DESTINATION midas_auto_calibrate/_bin
)
