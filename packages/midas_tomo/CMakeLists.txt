# packages/midas_tomo/CMakeLists.txt
#
# scikit-build-core hook: compiles the gridrec CT engine (c_src/tomo_*.c) into
# a standalone binary `MIDAS_TOMO`, and — when CUDA is available — the GPU
# variant `MIDAS_TOMO_GPU`. Both install under <site-packages>/midas_tomo/bin/.
# The Python wrapper (midas_tomo/backend_c.py) locates them via
# importlib.resources.
#
# EVERY dependency probe here is FAIL-SOFT: if a compiler, FFTW, HDF5 or
# OpenMP is missing we `return()` with a message(WARNING) rather than erroring,
# so `pip install midas-tomo` still succeeds. backend_c.available() then
# reports False at runtime and tells the user what to install.
#
# The sources under c_src/ are a byte-identical mirror of TOMO/src/ at the
# fork commit (see FORK.txt). Do not edit them to satisfy a build problem —
# solve it here instead, so the drift check stays meaningful.

cmake_minimum_required(VERSION 3.20)

# Declare no languages up front: `LANGUAGES C` makes CMake probe for a C
# compiler at this line and hard-error if none exists (e.g. Windows without
# MSVC Build Tools). That would fail the wheel build before we reach the
# graceful fallbacks below.
project(midas_tomo_c NONE)

include(CheckLanguage)
check_language(C)
if(NOT CMAKE_C_COMPILER)
  message(WARNING
    "midas-tomo: no C compiler found — MIDAS_TOMO will not be built. On macOS "
    "install the Xcode command line tools; on Linux install gcc. The Python "
    "package still installs; backend_c.available() will report False.")
  return()
endif()
enable_language(C)

set(MIDAS_TOMO_C_SRC ${CMAKE_CURRENT_SOURCE_DIR}/c_src)

# ---------------------------------------------------------------- FFTW3f
# gridrec's CPU path uses the SINGLE-precision FFTW interface (fftwf_*), so we
# need the fftw3f library specifically, not fftw3. There is no FindFFTW module
# shipped with CMake, so: pkg-config first (how conda/homebrew/apt all ship
# it), then a plain find_library fallback for hand-built installs.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(FFTW3F QUIET fftw3f)
endif()

if(FFTW3F_FOUND)
  set(TOMO_FFTW_INCLUDE_DIRS ${FFTW3F_INCLUDE_DIRS})
  set(TOMO_FFTW_LIBRARIES ${FFTW3F_LINK_LIBRARIES})
  message(STATUS "midas-tomo: FFTW3f found via pkg-config (${FFTW3F_VERSION})")
else()
  find_path(TOMO_FFTW_INCLUDE_DIRS fftw3.h)
  find_library(TOMO_FFTW_LIBRARIES NAMES fftw3f libfftw3f)
  if(TOMO_FFTW_INCLUDE_DIRS AND TOMO_FFTW_LIBRARIES)
    message(STATUS "midas-tomo: FFTW3f found at ${TOMO_FFTW_LIBRARIES}")
    set(TOMO_HAVE_FFTW ON)
  else()
    message(STATUS
      "midas-tomo: FFTW3f not found — building with the vendored pocketfft "
      "backend only. This is fully supported; pocketfft is BSD-licensed and "
      "needs no external library. Install FFTW if you need bit-comparability "
      "with historical runs.")
    set(TOMO_FFTW_INCLUDE_DIRS "")
    set(TOMO_FFTW_LIBRARIES "")
    set(TOMO_HAVE_FFTW OFF)
  endif()
endif()
if(NOT DEFINED TOMO_HAVE_FFTW)
  set(TOMO_HAVE_FFTW ON)
endif()

# ---------------------------------------------------------------- HDF5
# OPTIONAL. midas-tomo reads HDF5 in Python (h5py, midas_tomo/hdf5.py) and
# hands the engine the staged binary layout, so the C's own readRawHDF5 path
# is unused by the package. Building without HDF5 costs nothing the package
# relies on; it only disables hand-written parameter files that set
# HDF5FileName, and that case gets an explicit runtime error pointing at the
# Python reader.
find_package(HDF5 QUIET COMPONENTS C)
if(HDF5_FOUND)
  message(STATUS "midas-tomo: HDF5 ${HDF5_VERSION} found — HDF5FileName input enabled")
else()
  message(STATUS
    "midas-tomo: HDF5 not found — building without the C HDF5 reader. This is "
    "fully supported: read HDF5 with midas_tomo.hdf5.read_exchange() instead.")
endif()

# ---------------------------------------------------------------- OpenMP
# gridrec parallelises over slices with OpenMP. Without it the binary would
# still be correct but single-threaded; the vendored tomo_init.c includes
# <omp.h> unconditionally, so it is a hard requirement for the build.
find_package(OpenMP QUIET COMPONENTS C)
if(NOT OpenMP_C_FOUND)
  message(WARNING
    "midas-tomo: OpenMP not found — MIDAS_TOMO will not be built. On macOS "
    "install libomp via `brew install libomp`; on Linux libgomp usually comes "
    "with gcc. The Python package still installs; backend_c.available() "
    "reports False.")
  return()
endif()

# ------------------------------------------------- generated midas_version.h
# The legacy MIDAS build derives this from git. For the pip package we bake in
# the package version, overridable at cmake-invoke time.
set(MIDAS_PIP_VERSION "${SKBUILD_PROJECT_VERSION}" CACHE STRING
    "Version string baked into MIDAS_TOMO at build time")
if(NOT MIDAS_PIP_VERSION)
  set(MIDAS_PIP_VERSION "pip-build")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/midas_version.h"
"#ifndef MIDAS_VERSION_H\n"
"#define MIDAS_VERSION_H\n"
"#define MIDAS_VERSION \"${MIDAS_PIP_VERSION}\"\n"
"#define MIDAS_GIT_HASH \"\"\n"
"#define MIDAS_GIT_DATE \"\"\n"
"#define MIDAS_VERSION_STRING \"midas-tomo v\" MIDAS_VERSION\n"
"#endif\n")

# pocketfft is a C++17 header, so the FFT shim is the one C++ file here.
# Enabling CXX is unconditional: pocketfft is what makes a build possible with
# no external FFT library, so it is always compiled in, and FFTW becomes the
# optional one.
enable_language(CXX)

set(TOMO_SOURCES
  ${MIDAS_TOMO_C_SRC}/tomo_init.c
  ${MIDAS_TOMO_C_SRC}/tomo_gridrec.c
  ${MIDAS_TOMO_C_SRC}/tomo_utils.c
  ${MIDAS_TOMO_C_SRC}/tomo_cleanup.c
  ${MIDAS_TOMO_C_SRC}/midas_fft_pocket.cpp
)

# ------------------------------------------------- legacy-matching C flags
# These reproduce the root MIDAS CMakeLists.txt (`-fPIC -O3 -w -g`, C99) and
# are applied unconditionally, independent of CMAKE_BUILD_TYPE. Two reasons,
# both load-bearing:
#
# 1. CORRECTNESS. tomo_gridrec.c defines Cnvlvnt() as C99 `inline` with a
#    matching `inline` declaration in tomo_heads.h and no `extern` definition
#    anywhere. Under C99 semantics that is an *inline definition* only: no
#    external symbol is emitted, so any call the optimiser does not inline is
#    an undefined reference at link time. At -O0 GCC inlines nothing and the
#    link fails outright. -O3 is therefore not a preference here, it is what
#    makes the vendored source link at all. Fixing this properly means
#    editing the C (adding an extern definition), which would break the
#    byte-identical mirror -- so we match the legacy build instead and record
#    the constraint here.
#
# 2. BIT-PARITY. Optimisation level changes instruction selection, including
#    FMA contraction and vectorisation of the gridrec inner loops, so it
#    changes the low-order bits of the reconstruction. Reproducing the 2023
#    reference output requires the same flags, not merely a working binary.
#    Anything that alters this list invalidates the golden-file test.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang|IntelLLVM")
  set(TOMO_LEGACY_C_FLAGS -fPIC -O3 -w -g)
else()
  set(TOMO_LEGACY_C_FLAGS "")
  message(WARNING
    "midas-tomo: unrecognised C compiler '${CMAKE_C_COMPILER_ID}' — cannot "
    "apply the legacy -O3 flags. The build may fail to link (Cnvlvnt) and "
    "will not be bit-comparable to the reference output.")
endif()

# ---------------------------------------------------------------- CPU target
add_executable(MIDAS_TOMO ${TOMO_SOURCES})
target_include_directories(MIDAS_TOMO PRIVATE
  ${MIDAS_TOMO_C_SRC}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${TOMO_FFTW_INCLUDE_DIRS}
)
# PI=M_PI matches the legacy TOMO/CMakeLists.txt definition — the sources rely
# on PI being defined by the build, not by math.h.
target_compile_definitions(MIDAS_TOMO PRIVATE PI=M_PI)
target_compile_options(MIDAS_TOMO PRIVATE ${TOMO_LEGACY_C_FLAGS})
target_link_libraries(MIDAS_TOMO PRIVATE
  ${TOMO_FFTW_LIBRARIES}
  OpenMP::OpenMP_C
  m
)
if(TOMO_HAVE_FFTW)
  target_compile_definitions(MIDAS_TOMO PRIVATE MIDAS_TOMO_HAVE_FFTW)
endif()
set_target_properties(MIDAS_TOMO PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(HDF5_FOUND)
  target_include_directories(MIDAS_TOMO PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(MIDAS_TOMO PRIVATE ${HDF5_C_LIBRARIES})
  target_compile_definitions(MIDAS_TOMO PRIVATE MIDAS_TOMO_HAVE_HDF5)
endif()
set_target_properties(MIDAS_TOMO PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)

install(TARGETS MIDAS_TOMO RUNTIME DESTINATION midas_tomo/bin)

# ------------------------------------------------------------ shared library
# The same sources built as a .so/.dylib so Python can call midas_tomo_run()
# in-process via ctypes instead of spawning the binary and round-tripping
# through disk. Built from the identical source list and the identical flags,
# so the two agree by construction; tests/test_library.py asserts that
# bitwise rather than trusting it.
#
# The CLI binary above is deliberately kept: dev/build_reference_binary.sh
# compares against it, which is what keeps the parity gate against a pre-fork
# build runnable.
add_library(midastomo SHARED ${TOMO_SOURCES})
target_include_directories(midastomo PRIVATE
  ${MIDAS_TOMO_C_SRC}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${TOMO_FFTW_INCLUDE_DIRS}
)
# MIDAS_TOMO_LIBRARY_BUILD compiles out main(): a shared object with an
# entry-point symbol is legal but confuses some loaders, and there is no
# reason to carry it.
target_compile_definitions(midastomo PRIVATE PI=M_PI MIDAS_TOMO_LIBRARY_BUILD)
target_compile_options(midastomo PRIVATE ${TOMO_LEGACY_C_FLAGS})
target_link_libraries(midastomo PRIVATE
  ${TOMO_FFTW_LIBRARIES}
  OpenMP::OpenMP_C
  m
)
set_target_properties(midastomo PROPERTIES
  C_STANDARD 99 C_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)
if(TOMO_HAVE_FFTW)
  target_compile_definitions(midastomo PRIVATE MIDAS_TOMO_HAVE_FFTW)
endif()
set_target_properties(midastomo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(HDF5_FOUND)
  target_include_directories(midastomo PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(midastomo PRIVATE ${HDF5_C_LIBRARIES})
  target_compile_definitions(midastomo PRIVATE MIDAS_TOMO_HAVE_HDF5)
endif()

install(TARGETS midastomo LIBRARY DESTINATION midas_tomo/bin
                          RUNTIME DESTINATION midas_tomo/bin)

# ---------------------------------------------------------------- GPU target
# Optional in every sense: no CUDA toolkit → skip with a status message, never
# an error. The Mac has no CUDA, so this path is exercised only on the
# beamline GPU hosts.
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    add_executable(MIDAS_TOMO_GPU ${TOMO_SOURCES} ${MIDAS_TOMO_C_SRC}/tomo_gpu.cu)
    target_include_directories(MIDAS_TOMO_GPU PRIVATE
      ${MIDAS_TOMO_C_SRC}
      ${CMAKE_CURRENT_BINARY_DIR}
      ${TOMO_FFTW_INCLUDE_DIRS}
    )
    target_compile_definitions(MIDAS_TOMO_GPU PRIVATE PI=M_PI ENABLE_CUDA)
    # Same legacy flags for the C sources; nvcc rejects them on .cu files, so
    # gate on COMPILE_LANGUAGE.
    target_compile_options(MIDAS_TOMO_GPU PRIVATE
      $<$<COMPILE_LANGUAGE:C>:${TOMO_LEGACY_C_FLAGS}>)
    if(HDF5_FOUND)
      target_include_directories(MIDAS_TOMO_GPU PRIVATE ${HDF5_INCLUDE_DIRS})
      target_link_libraries(MIDAS_TOMO_GPU PRIVATE ${HDF5_C_LIBRARIES})
      target_compile_definitions(MIDAS_TOMO_GPU PRIVATE MIDAS_TOMO_HAVE_HDF5)
    endif()
    target_link_libraries(MIDAS_TOMO_GPU PRIVATE
      ${TOMO_FFTW_LIBRARIES}
      CUDA::cufft
      CUDA::cudart
      OpenMP::OpenMP_C
      m
    )
    # nvcc does not pass -fopenmp to the host compiler on its own.
    target_compile_options(MIDAS_TOMO_GPU PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    set_target_properties(MIDAS_TOMO_GPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    install(TARGETS MIDAS_TOMO_GPU RUNTIME DESTINATION midas_tomo/bin)
    message(STATUS "midas-tomo: CUDA found — MIDAS_TOMO_GPU enabled")
  else()
    message(STATUS "midas-tomo: nvcc present but CUDAToolkit not found — GPU target skipped")
  endif()
else()
  message(STATUS "midas-tomo: no CUDA compiler — GPU target skipped (CPU build is unaffected)")
endif()
