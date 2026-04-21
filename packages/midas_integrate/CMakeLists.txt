cmake_minimum_required(VERSION 3.20)
project(midas_integrate LANGUAGES C)

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
find_package(OpenMP COMPONENTS C)
find_package(HDF5 REQUIRED COMPONENTS C HL)
find_package(TIFF REQUIRED)

# NLopt — same FetchContent fallback pattern as midas-auto-calibrate.
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
if(NOT TARGET NLopt::nlopt)
    message(FATAL_ERROR "Could not locate or build NLopt.")
endif()

# blosc2 + libzip — required by ZarrReader. System install is the cheap
# path (Homebrew / apt / dnf before-all); FetchContent is heavy for these
# and their zlib entanglement is fragile.
#
# Use pkg_check_modules(IMPORTED_TARGET) so the created CMake targets carry
# full library paths — plain `pkg_check_modules(BLOSC2 blosc2)` returns the
# library *name* without a `-L` prefix, which fails on keg-only Homebrew
# layouts (c-blosc2 isn't in /opt/homebrew/lib by default).
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(BLOSC2 IMPORTED_TARGET QUIET blosc2)
    pkg_check_modules(LIBZIP IMPORTED_TARGET QUIET libzip)
endif()

if(NOT BLOSC2_FOUND)
    # Try common Homebrew / package-manager install prefixes explicitly
    # before giving up — pkg-config isn't always on the user's PATH.
    set(_BLOSC2_HINTS)
    if(EXISTS "/opt/homebrew/opt/c-blosc2")
        list(APPEND _BLOSC2_HINTS "/opt/homebrew/opt/c-blosc2")
    endif()
    find_path(BLOSC2_INCLUDE_DIR blosc2.h HINTS ${_BLOSC2_HINTS} PATH_SUFFIXES include)
    find_library(BLOSC2_LIBRARY NAMES blosc2 HINTS ${_BLOSC2_HINTS} PATH_SUFFIXES lib)
    if(NOT BLOSC2_INCLUDE_DIR OR NOT BLOSC2_LIBRARY)
        message(FATAL_ERROR "BLOSC2 not found. Install c-blosc2 "
            "(brew install c-blosc2 / apt install libblosc2-dev / "
            "dnf install blosc2-devel). For keg-only Homebrew installs, "
            "export PKG_CONFIG_PATH=$(brew --prefix c-blosc2)/lib/pkgconfig.")
    endif()
    add_library(PkgConfig::BLOSC2 UNKNOWN IMPORTED)
    set_target_properties(PkgConfig::BLOSC2 PROPERTIES
        IMPORTED_LOCATION "${BLOSC2_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${BLOSC2_INCLUDE_DIR}")
endif()

if(NOT LIBZIP_FOUND)
    set(_LIBZIP_HINTS)
    if(EXISTS "/opt/homebrew/opt/libzip")
        list(APPEND _LIBZIP_HINTS "/opt/homebrew/opt/libzip")
    endif()
    find_path(LIBZIP_INCLUDE_DIR zip.h HINTS ${_LIBZIP_HINTS} PATH_SUFFIXES include)
    find_library(LIBZIP_LIBRARY NAMES zip HINTS ${_LIBZIP_HINTS} PATH_SUFFIXES lib)
    if(NOT LIBZIP_INCLUDE_DIR OR NOT LIBZIP_LIBRARY)
        message(FATAL_ERROR "libzip not found. Install "
            "(brew install libzip / apt install libzip-dev / "
            "dnf install libzip-devel).")
    endif()
    add_library(PkgConfig::LIBZIP UNKNOWN IMPORTED)
    set_target_properties(PkgConfig::LIBZIP PROPERTIES
        IMPORTED_LOCATION "${LIBZIP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LIBZIP_INCLUDE_DIR}")
endif()

# ---------------------------------------------------------------------------
# Generated version header — same layout as midas-auto-calibrate.
# ---------------------------------------------------------------------------
if(NOT DEFINED PACKAGE_VERSION)
    set(PACKAGE_VERSION "0.1.0")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/midas_version.h"
"#ifndef MIDAS_VERSION_H\n#define MIDAS_VERSION_H\n#define MIDAS_VERSION \"${PACKAGE_VERSION}\"\n#define MIDAS_GIT_HASH \"midas-integrate\"\n#define MIDAS_GIT_DATE \"\"\n#define MIDAS_VERSION_STRING \"midas-integrate v\" MIDAS_VERSION\n#endif\n")

# ---------------------------------------------------------------------------
# Shared integration static lib — mirrors the main MIDAS `midas_integration`
# target. Duplicated from midas-auto-calibrate; we bump both in lockstep on
# shared-C ABI breaks (see packages/midas_integrate/RELEASING.md).
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
target_link_libraries(midas_integration PUBLIC m NLopt::nlopt)
if(OpenMP_C_FOUND)
    target_link_libraries(midas_integration PUBLIC OpenMP::OpenMP_C)
endif()

# ---------------------------------------------------------------------------
# MIDASIntegrator — CPU radial integration driver (wraps IntegratorZarrOMP).
# ---------------------------------------------------------------------------
add_executable(MIDASIntegrator
    src/c/IntegratorZarrOMP.c
    src/c/ZarrReader.c
    src/c/PeakFitIO.c
    src/c/FileReader.c
    src/c/MIDAS_Math.c
    src/c/MIDAS_ParamParser.c
)
target_include_directories(MIDASIntegrator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/c
    ${CMAKE_CURRENT_BINARY_DIR}/generated
    ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(MIDASIntegrator PRIVATE
    midas_integration
    ${HDF5_C_LIBRARIES} ${HDF5_HL_LIBRARIES}
    TIFF::TIFF
    PkgConfig::BLOSC2
    PkgConfig::LIBZIP
)
if(OpenMP_C_FOUND)
    target_link_libraries(MIDASIntegrator PRIVATE OpenMP::OpenMP_C)
endif()

# ---------------------------------------------------------------------------
# MIDASDetectorMapper — pre-computes pixel→(R,η) binning. Paired with
# MIDASIntegrator: run Mapper once per geometry, integrate many frames.
# ---------------------------------------------------------------------------
add_executable(MIDASDetectorMapper
    src/c/DetectorMapper.c
    src/c/DetectorGeometry.c
    src/c/ZarrReader.c
    src/c/Panel.c
    src/c/FileReader.c
    src/c/MIDAS_Math.c
    src/c/MIDAS_ParamParser.c
)
target_include_directories(MIDASDetectorMapper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/c
    ${CMAKE_CURRENT_BINARY_DIR}/generated
    ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(MIDASDetectorMapper PRIVATE
    midas_integration
    ${HDF5_C_LIBRARIES} ${HDF5_HL_LIBRARIES}
    TIFF::TIFF
    PkgConfig::BLOSC2
    PkgConfig::LIBZIP
)
if(OpenMP_C_FOUND)
    target_link_libraries(MIDASDetectorMapper PRIVATE OpenMP::OpenMP_C)
endif()

# ---------------------------------------------------------------------------
# Install — inside the wheel's package tree, alongside _binaries.py.
# ---------------------------------------------------------------------------
install(TARGETS MIDASIntegrator MIDASDetectorMapper
    RUNTIME DESTINATION midas_integrate/_bin
)
