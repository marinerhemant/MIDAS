# packages/midas_index/CMakeLists.txt
#
# scikit-build-core hook: compiles the unified C indexer
# (c_src/IndexerUnified.c) into a standalone binary `midas_indexer` and
# installs it under <site-packages>/midas_index/bin/. The Python wrapper
# (midas_index/backend_c.py) locates the binary via importlib.resources.
#
# Requires OpenMP. If OpenMP is missing (e.g. macOS clang without
# `brew install libomp`), the C target is silently skipped — `pip install`
# still succeeds and users get the Python-only path.
#
# Source references reach into ../../FF_HEDM/src/ for shared math /
# misorientation primitives. This works for:
#   - editable installs (`pip install -e packages/midas_index`)
#   - git installs (`pip install "midas-index @ git+..."`)
# Per project_no_c_pypi_packages.md, PyPI sdists are not a target — the C
# binary is build-from-source from the MIDAS git tree only.

cmake_minimum_required(VERSION 3.20)
# Declare no languages up front: `LANGUAGES C` would make CMake probe for a C
# compiler at this line and hard-error if none exists (e.g. Windows without
# MSVC Build Tools, where the NMake generator can't find `nmake`). That would
# fail the whole wheel build before we reach the graceful fallbacks below.
project(midas_index_c NONE)

# Probe for a C compiler without aborting if one is missing. check_language()
# sets CMAKE_C_COMPILER to a NOTFOUND value instead of raising, so we can bail
# out cleanly and let `pip install` succeed with the Python-only path.
include(CheckLanguage)
check_language(C)
if(NOT CMAKE_C_COMPILER)
  message(WARNING
    "No C compiler found — the C indexer will not be built. On Windows install "
    "the Visual Studio Build Tools (\"Desktop development with C++\"). The "
    "Python-only path remains available without this binary.")
  return()
endif()
enable_language(C)

set(MIDAS_INDEX_C_SRC ${CMAKE_CURRENT_SOURCE_DIR}/c_src)
# All required C sources are vendored under c_src/ — no reach into
# FF_HEDM/src/ at build time. The vendored copies of MIDAS_Math.{c,h},
# GetMisorientation.{c,h}, IndexerConsolidatedIO.h are stripped of the
# upstream NLopt dependency to keep the pip install dependency-free.

find_package(OpenMP COMPONENTS C)
if(NOT OpenMP_C_FOUND)
  message(WARNING
    "OpenMP not found — the C indexer will not be built. On macOS install "
    "libomp via `brew install libomp`. On Linux install libgomp via your "
    "package manager (gcc usually pulls it in). The Python-only path "
    "remains available without this binary.")
  return()
endif()

# Generate a minimal midas_version.h. The legacy MIDAS build derives this
# from git; for the pip package we hard-code a placeholder that's overridden
# by `-DMIDAS_PIP_VERSION=...` at cmake-invoke time.
set(MIDAS_PIP_VERSION "${SKBUILD_PROJECT_VERSION}" CACHE STRING
    "Version string baked into midas_indexer at build time")
if(NOT MIDAS_PIP_VERSION)
  set(MIDAS_PIP_VERSION "pip-build")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/midas_version.h"
"#ifndef MIDAS_VERSION_H\n"
"#define MIDAS_VERSION_H\n"
"#define MIDAS_VERSION \"${MIDAS_PIP_VERSION}\"\n"
"#define MIDAS_GIT_HASH \"\"\n"
"#define MIDAS_GIT_DATE \"\"\n"
"#define MIDAS_VERSION_STRING \"midas-index v\" MIDAS_VERSION\n"
"#endif\n")

set(INDEXER_SOURCES
  ${MIDAS_INDEX_C_SRC}/IndexerUnified.c
  ${MIDAS_INDEX_C_SRC}/MIDAS_Math.c
  ${MIDAS_INDEX_C_SRC}/GetMisorientation.c
)

add_executable(midas_indexer ${INDEXER_SOURCES})
target_include_directories(midas_indexer PRIVATE
  ${MIDAS_INDEX_C_SRC}
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(midas_indexer PRIVATE OpenMP::OpenMP_C m)
target_compile_features(midas_indexer PRIVATE c_std_99)

install(TARGETS midas_indexer RUNTIME DESTINATION midas_index/bin)
