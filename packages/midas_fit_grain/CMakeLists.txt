# packages/midas_fit_grain/CMakeLists.txt
#
# scikit-build-core hook: compiles the unified C refiner
# (c_src/FitUnified.c) into a standalone binary `midas_fitgrain` and installs
# it under <site-packages>/midas_fit_grain/bin/. The Python wrapper
# (midas_fit_grain/backend_c.py) locates the binary via importlib.resources.
#
# Mirrors midas_index/CMakeLists.txt: requires OpenMP; if no C compiler or
# OpenMP is found the C target is silently skipped so `pip install` still
# succeeds with the Python-only (torch) path. All C sources are vendored under
# c_src/ — including forward.{c,h} (the shared forward model, canonical source
# packages/midas_ckernel) and the NLopt-free vendored Nelder-Mead.

cmake_minimum_required(VERSION 3.20)
project(midas_fit_grain_c NONE)

include(CheckLanguage)
check_language(C)
if(NOT CMAKE_C_COMPILER)
  message(WARNING
    "No C compiler found — the C refiner will not be built. The Python-only "
    "(torch) path remains available without this binary.")
  return()
endif()
enable_language(C)

set(MIDAS_FG_C_SRC ${CMAKE_CURRENT_SOURCE_DIR}/c_src)

find_package(OpenMP COMPONENTS C)
if(NOT OpenMP_C_FOUND)
  message(WARNING
    "OpenMP not found — the C refiner will not be built. On macOS install "
    "libomp via `brew install libomp`. The Python-only path remains available.")
  return()
endif()

set(MIDAS_PIP_VERSION "${SKBUILD_PROJECT_VERSION}" CACHE STRING
    "Version string baked into midas_fitgrain at build time")
if(NOT MIDAS_PIP_VERSION)
  set(MIDAS_PIP_VERSION "pip-build")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/midas_version.h"
"#ifndef MIDAS_VERSION_H\n"
"#define MIDAS_VERSION_H\n"
"#define MIDAS_VERSION \"${MIDAS_PIP_VERSION}\"\n"
"#define MIDAS_GIT_HASH \"\"\n"
"#define MIDAS_GIT_DATE \"\"\n"
"#define MIDAS_VERSION_STRING \"midas-fit-grain v\" MIDAS_VERSION\n"
"#endif\n")

# forward.c is the SHARED diffraction forward model (canonical source
# packages/midas_ckernel/c_src/forward.{c,h}); CalcDiffractionSpots.c is the
# refiner-side adapter over it. nelder_mead.c is the NLopt-free optimizer.
set(FITGRAIN_SOURCES
  ${MIDAS_FG_C_SRC}/FitUnified.c
  ${MIDAS_FG_C_SRC}/CalcDiffractionSpots.c
  ${MIDAS_FG_C_SRC}/forward.c
  ${MIDAS_FG_C_SRC}/MIDAS_Math.c
  ${MIDAS_FG_C_SRC}/GetMisorientation.c
  ${MIDAS_FG_C_SRC}/nelder_mead.c
  ${MIDAS_FG_C_SRC}/MIDAS_ParamParser.c
)

add_executable(midas_fitgrain ${FITGRAIN_SOURCES})
target_include_directories(midas_fitgrain PRIVATE
  ${MIDAS_FG_C_SRC}
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(midas_fitgrain PRIVATE OpenMP::OpenMP_C m)
target_compile_features(midas_fitgrain PRIVATE c_std_99)

install(TARGETS midas_fitgrain RUNTIME DESTINATION midas_fit_grain/bin)
