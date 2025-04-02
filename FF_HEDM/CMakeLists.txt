cmake_minimum_required(VERSION 3.16)
# Set the CMP0135 policy to NEW
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(MIDAS_FF_HEDM C)
# project(MIDAS_FF_HEDM C CUDA)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(BUILD_CUDA "Build CUDA components" OFF)
option(BUILD_OMP "Build OpenMP components" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(DOWNLOAD_DEPENDENCIES "Download and build dependencies" ON)
option(SYSTEM_DEPENDENCIES "Use system dependencies if available" OFF)
option(APPLE_RPATH_FIX "Apply rpath fix for macOS" OFF)
option(USE_USER_HOME "Use user's home directory for config" ON)
option(INSTALL_PYTHON_DEPS "Install Python dependencies from requirements.txt" ON)
set(MIDAS_CONFIG_DIR "$ENV{HOME}/.MIDAS" CACHE PATH "Directory for MIDAS configuration files")
set(PYTHON_EXECUTABLE "" CACHE FILEPATH "Path to Python executable")

# Set default installation path
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation directory" FORCE)
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# RPATH settings for better portability
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Set RPATH for installed binaries
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@executable_path/../lib")
else()
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

# Find packages and dependencies
include(ExternalProject)
include(FetchContent)
include(GNUInstallDirs)

# Set up module path for custom find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/FindModules")

# Set default dependencies directory
set(DEPS_INSTALL_DIR "${CMAKE_BINARY_DIR}/deps" CACHE PATH "Dependencies installation directory")

# Add dependencies directory to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH ${DEPS_INSTALL_DIR})

# Check for OpenMP
if(BUILD_OMP)
  find_package(OpenMP)
  if(NOT OpenMP_C_FOUND)
    message(WARNING "OpenMP not found. Some components will be built without OpenMP support.")
  endif()
endif()

# Check for CUDA
if(BUILD_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -G -g -w -arch sm_52 -lm -rdc=true")
  else()
    message(WARNING "CUDA not found. CUDA components will not be built.")
    set(BUILD_CUDA OFF)
  endif()
endif()

# Function to handle dependencies
function(handle_dependency NAME URL DOWNLOAD_FLAG)
    if(NAME STREQUAL "LIBTIFF")
    if(TARGET TIFF::TIFF)
    message(STATUS "${NAME} target already exists as TIFF::TIFF, skipping")
    return()
    endif()
    elseif(TARGET ${NAME}::${NAME})
    message(STATUS "${NAME} target already exists, skipping")
    return()
    endif()
# Check if target already exists to avoid double-processing
  if(TARGET ${NAME}::${NAME})
    message(STATUS "${NAME} target already exists, skipping")
    return()
  endif()
  # Check for custom find module
  if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/FindModules/Find${NAME}.cmake")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/FindModules")
  endif()
  
  # Check for custom dependency file
  set(CUSTOM_DEP_FILE "${CMAKE_SOURCE_DIR}/cmake/deps/${NAME}_custom.cmake")
  if(EXISTS "${CUSTOM_DEP_FILE}")
    message(STATUS "Using custom dependency file for ${NAME}")
    include("${CUSTOM_DEP_FILE}")
    return()
  endif()
  
  if(${DOWNLOAD_FLAG})
    message(STATUS "Setting up ${NAME} for download and build")
    include(cmake/deps/${NAME}.cmake)
  else()
    if(SYSTEM_DEPENDENCIES)
      # Try to find the package in the system
      find_package(${NAME} QUIET)
      if(NOT ${NAME}_FOUND)
        # Also try with PkgConfig for better system library discovery
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
          string(TOLOWER "${NAME}" NAME_LOWER)
          pkg_check_modules(${NAME}_PC QUIET ${NAME_LOWER} lib${NAME_LOWER})
          if(${NAME}_PC_FOUND)
            message(STATUS "Found ${NAME} using PkgConfig")
            set(${NAME}_FOUND TRUE PARENT_SCOPE)
            # Create an imported target for consistent interface
            add_library(${NAME}::${NAME} INTERFACE IMPORTED)
            set_target_properties(${NAME}::${NAME} PROPERTIES
              INTERFACE_INCLUDE_DIRECTORIES "${${NAME}_PC_INCLUDE_DIRS}"
              INTERFACE_LINK_LIBRARIES "${${NAME}_PC_LIBRARIES}"
            )
          else()
            message(STATUS "${NAME} not found in system. Will download and build.")
            include(cmake/deps/${NAME}.cmake)
          endif()
        else()
          message(STATUS "${NAME} not found in system. Will download and build.")
          include(cmake/deps/${NAME}.cmake)
        endif()
      endif()
    else()
      include(cmake/deps/${NAME}.cmake)
    endif()
  endif()
    # Ensure the cmake file was actually included
    message(STATUS "Setting up ${NAME} dependency")
    include(cmake/deps/${NAME}.cmake)

  # Verify target was created
  if(NAME STREQUAL "LIBTIFF")
    if(NOT TARGET TIFF::TIFF)
      message(FATAL_ERROR "Failed to create TIFF::TIFF target")
    endif()
  elseif(NOT TARGET ${NAME}::${NAME})
    message(FATAL_ERROR "Failed to create ${NAME}::${NAME} target")
  endif()
  
endfunction()

# Create dependency directory structure
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/cmake/deps)

# Generate dependency CMake files
# NLOPT
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/deps/NLOPT_custom.cmake")
    file(WRITE ${CMAKE_SOURCE_DIR}/cmake/deps/NLOPT.cmake [[
    FetchContent_Declare(
    nlopt
    URL https://www.dropbox.com/scl/fi/ux4ccf23z7rotkgbqbrmk/nlopt-2.4.2.tar.gz?rlkey=afq6l6yyu9fnw1hpe62l4gwqq&dl=1
    URL_HASH MD5=d0b8f139a4acf29b76dbae69ade8ac54
    )

    FetchContent_GetProperties(nlopt)
    if(NOT nlopt_POPULATED)
    FetchContent_Populate(nlopt)
    
    # Set up NLOPT build options
    set(NLOPT_BUILD_SHARED ${BUILD_SHARED_LIBS} CACHE BOOL "Build NLopt as a shared library" FORCE)
    set(NLOPT_BUILD_STATIC NOT ${BUILD_SHARED_LIBS} CACHE BOOL "Build NLopt as a static library" FORCE)
    set(NLOPT_PYTHON OFF CACHE BOOL "Build python bindings" FORCE)
    set(NLOPT_OCTAVE OFF CACHE BOOL "Build octave bindings" FORCE)
    set(NLOPT_MATLAB OFF CACHE BOOL "Build matlab bindings" FORCE)
    set(NLOPT_GUILE OFF CACHE BOOL "Build guile bindings" FORCE)
    set(NLOPT_SWIG OFF CACHE BOOL "Use SWIG to build bindings" FORCE)
    set(NLOPT_TESTS OFF CACHE BOOL "Build tests" FORCE)
    
    add_subdirectory(${nlopt_SOURCE_DIR} ${nlopt_BINARY_DIR})

    # Export as a target
    add_library(NLOPT::NLOPT INTERFACE IMPORTED)
    if(BUILD_SHARED_LIBS)
        set_target_properties(NLOPT::NLOPT PROPERTIES
        INTERFACE_LINK_LIBRARIES nlopt
        )
    else()
        set_target_properties(NLOPT::NLOPT PROPERTIES
        INTERFACE_LINK_LIBRARIES nlopt-static
        )
    endif()
    set_target_properties(NLOPT::NLOPT PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${nlopt_SOURCE_DIR}/src/api"
    )
    endif()
    ]])
endif()

# LIBTIFF
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/deps/LIBTIFF_custom.cmake")
    file(WRITE ${CMAKE_SOURCE_DIR}/cmake/deps/LIBTIFF.cmake [[
    FetchContent_Declare(
    libtiff
    URL https://www.dropbox.com/scl/fi/tk3axrjtjgxmjj9hzsk13/tiff-4.6.0.tar.gz?rlkey=judqzxze5g4sg0bviyul8kqvp&dl=1
    )

    FetchContent_GetProperties(libtiff)
    if(NOT libtiff_POPULATED)
    FetchContent_Populate(libtiff)

    # Configure TIFF build options
    set(tiff_BUILD_SHARED ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared libraries" FORCE)
    set(tiff_BUILD_STATIC NOT ${BUILD_SHARED_LIBS} CACHE BOOL "Build static libraries" FORCE)

    # Disable unnecessary components
    set(tiff_BUILD_TOOLS OFF CACHE BOOL "Build TIFF tools" FORCE)
    set(tiff_BUILD_DOCS OFF CACHE BOOL "Build TIFF documentation" FORCE)
    set(tiff_BUILD_CONTRIB OFF CACHE BOOL "Build TIFF contrib" FORCE)
    set(tiff_BUILD_TESTS OFF CACHE BOOL "Build TIFF tests" FORCE)
    set(tiff_BUILD_ZLIB ON CACHE BOOL "Enable Zlib support" FORCE)

    # Find system zlib or one installed by blosc
    find_package(ZLIB)
    if(ZLIB_FOUND)
        set(ZLIB_INCLUDE_DIR ${ZLIB_INCLUDE_DIRS})
        set(ZLIB_LIBRARY ${ZLIB_LIBRARIES})
    endif()

    add_subdirectory(${libtiff_SOURCE_DIR} ${libtiff_BINARY_DIR})

    # Export TIFF library as a target
    add_library(TIFF::TIFF INTERFACE IMPORTED)
    if(BUILD_SHARED_LIBS)
        set_target_properties(TIFF::TIFF PROPERTIES
        INTERFACE_LINK_LIBRARIES tiff
        )
    else()
        set_target_properties(TIFF::TIFF PROPERTIES
        INTERFACE_LINK_LIBRARIES tiff-static
        )
    endif()
    set_target_properties(TIFF::TIFF PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${libtiff_SOURCE_DIR}/libtiff"
    )
    endif()
    ]])
endif()

# ZLIB
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/deps/ZLIB_custom.cmake")
    file(WRITE ${CMAKE_SOURCE_DIR}/cmake/deps/ZLIB.cmake [[
    FetchContent_Declare(
    zlib
    URL https://www.dropbox.com/scl/fi/vrgb8i9755eojx5eh1a2r/zlib-1.3.1.tar.gz?rlkey=9jwwjc1aqmflin5b75r2lc2yw&dl=1
    )

    FetchContent_GetProperties(zlib)
    if(NOT zlib_POPULATED)
    FetchContent_Populate(zlib)
    
    set(ZLIB_BUILD_SHARED ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared libraries" FORCE)
    set(ZLIB_BUILD_STATIC NOT ${BUILD_SHARED_LIBS} CACHE BOOL "Build static libraries" FORCE)
    
    add_subdirectory(${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
    
    # Export ZLIB as a target
    add_library(ZLIB::ZLIB INTERFACE IMPORTED)
    if(BUILD_SHARED_LIBS)
        set_target_properties(ZLIB::ZLIB PROPERTIES
        INTERFACE_LINK_LIBRARIES zlib
        )
    else()
        set_target_properties(ZLIB::ZLIB PROPERTIES
        INTERFACE_LINK_LIBRARIES zlibstatic
        )
    endif()
    set_target_properties(ZLIB::ZLIB PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${zlib_SOURCE_DIR};${zlib_BINARY_DIR}"
    )
    endif()
    ]])
endif()

# HDF5
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/deps/HDF5_custom.cmake")
    file(WRITE ${CMAKE_SOURCE_DIR}/cmake/deps/HDF5.cmake [[
    # HDF5 configuration
    FetchContent_Declare(
    hdf5
    URL https://www.dropbox.com/scl/fi/ib4wkq1s9jhm0oi9n6r7c/hdf5-1.14.2.tar.gz?rlkey=eq20hs7juecpwcn1vuumssjuf&dl=1
    )

    FetchContent_GetProperties(hdf5)
    if(NOT hdf5_POPULATED)
    FetchContent_Populate(hdf5)

    # Set HDF5 options
    set(HDF5_ENABLE_Z_LIB_SUPPORT ON CACHE BOOL "Enable ZLIB support" FORCE)
    set(HDF5_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared libraries" FORCE)
    set(HDF5_BUILD_STATIC_LIBS NOT ${BUILD_SHARED_LIBS} CACHE BOOL "Build static libraries" FORCE)
    set(HDF5_BUILD_TOOLS OFF CACHE BOOL "Build HDF5 tools" FORCE)
    set(HDF5_BUILD_EXAMPLES OFF CACHE BOOL "Build HDF5 examples" FORCE)
    set(HDF5_BUILD_UTILS OFF CACHE BOOL "Build HDF5 utilities" FORCE)
    set(HDF5_BUILD_HL_LIB ON CACHE BOOL "Build HDF5 high level APIs" FORCE)
    set(HDF5_BUILD_FORTRAN OFF CACHE BOOL "Build Fortran support" FORCE)
    set(HDF5_ENABLE_THREADSAFE OFF CACHE BOOL "Enable thread safety" FORCE)
    set(HDF5_ENABLE_PARALLEL OFF CACHE BOOL "Enable parallel HDF5" FORCE)

    # Find system zlib or one installed by blosc
    find_package(ZLIB)
    if(ZLIB_FOUND)
        set(ZLIB_INCLUDE_DIR ${ZLIB_INCLUDE_DIRS})
        set(ZLIB_LIBRARY ${ZLIB_LIBRARIES})
    endif()

    add_subdirectory(${hdf5_SOURCE_DIR} ${hdf5_BINARY_DIR})

    # Export HDF5 as targets
    add_library(HDF5::HDF5 INTERFACE IMPORTED)
    add_library(HDF5::HL INTERFACE IMPORTED)

    if(BUILD_SHARED_LIBS)
        set_target_properties(HDF5::HDF5 PROPERTIES
        INTERFACE_LINK_LIBRARIES hdf5-shared
        )
        set_target_properties(HDF5::HL PROPERTIES
        INTERFACE_LINK_LIBRARIES hdf5_hl-shared
        )
    else()
        set_target_properties(HDF5::HDF5 PROPERTIES
        INTERFACE_LINK_LIBRARIES hdf5-static
        )
        set_target_properties(HDF5::HL PROPERTIES
        INTERFACE_LINK_LIBRARIES hdf5_hl-static
        )
    endif()

    set_target_properties(HDF5::HDF5 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${hdf5_SOURCE_DIR}/src;${hdf5_BINARY_DIR}"
    )
    set_target_properties(HDF5::HL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${hdf5_SOURCE_DIR}/hl/src;${hdf5_BINARY_DIR}/hl"
    )
    endif()
    ]])
endif()

# BLOSC
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/deps/BLOSC_custom.cmake")
  file(WRITE ${CMAKE_SOURCE_DIR}/cmake/deps/BLOSC.cmake [[
  FetchContent_Declare(
    blosc
    GIT_REPOSITORY https://github.com/Blosc/c-blosc2.git
    GIT_TAG main
  )

  FetchContent_GetProperties(blosc)
  if(NOT blosc_POPULATED)
    FetchContent_Populate(blosc)
    
    # Set BLOSC options
    set(BLOSC_IS_SUBPROJECT ON CACHE BOOL "Build as subproject" FORCE)
    set(BLOSC_INSTALL OFF CACHE BOOL "Install blosc" FORCE)
    set(BLOSC_SHARED ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared" FORCE)
    set(BLOSC_STATIC NOT ${BUILD_SHARED_LIBS} CACHE BOOL "Build static" FORCE)
    set(BLOSC_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
    set(BLOSC_BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmarks" FORCE)
    set(BLOSC_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
    
    add_subdirectory(${blosc_SOURCE_DIR} ${blosc_BINARY_DIR})
    
    # Export BLOSC as a target
    add_library(BLOSC::BLOSC INTERFACE IMPORTED)
    if(BUILD_SHARED_LIBS)
      set_target_properties(BLOSC::BLOSC PROPERTIES
        INTERFACE_LINK_LIBRARIES blosc2_shared
      )
    else()
      set_target_properties(BLOSC::BLOSC PROPERTIES
        INTERFACE_LINK_LIBRARIES blosc2_static
      )
    endif()
    set_target_properties(BLOSC::BLOSC PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${blosc_SOURCE_DIR}/include"
    )
  endif()
  ]])
endif()

# BLOSC1
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/deps/BLOSC1_custom.cmake")
  file(WRITE ${CMAKE_SOURCE_DIR}/cmake/deps/BLOSC1.cmake [[
  FetchContent_Declare(
    blosc1
    GIT_REPOSITORY https://github.com/Blosc/c-blosc.git
    GIT_TAG main
  )

  FetchContent_GetProperties(blosc1)
  if(NOT blosc1_POPULATED)
    FetchContent_Populate(blosc1)
    
    # Set BLOSC1 options
    set(BLOSC_IS_SUBPROJECT ON CACHE BOOL "Build as subproject" FORCE)
    set(BLOSC_INSTALL OFF CACHE BOOL "Install blosc" FORCE)
    set(BUILD_SHARED ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared" FORCE)
    set(BUILD_STATIC NOT ${BUILD_SHARED_LIBS} CACHE BOOL "Build static" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmarks" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
    
    add_subdirectory(${blosc1_SOURCE_DIR} ${blosc1_BINARY_DIR})
    
    # Export BLOSC1 as a target
    add_library(BLOSC1::BLOSC1 INTERFACE IMPORTED)
    if(BUILD_SHARED_LIBS)
      set_target_properties(BLOSC1::BLOSC1 PROPERTIES
        INTERFACE_LINK_LIBRARIES blosc_shared
      )
    else()
      set_target_properties(BLOSC1::BLOSC1 PROPERTIES
        INTERFACE_LINK_LIBRARIES blosc_static
      )
    endif()
    set_target_properties(BLOSC1::BLOSC1 PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${blosc1_SOURCE_DIR}/blosc"
    )
  endif()
  ]])
endif()

# LIBZIP
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/deps/LIBZIP_custom.cmake")
    file(WRITE ${CMAKE_SOURCE_DIR}/cmake/deps/LIBZIP.cmake [[
    FetchContent_Declare(
    libzip
    URL https://www.dropbox.com/scl/fi/2mo9gzxi8ms3pp10pu6ad/libzip-1.10.1.tar.gz?rlkey=w7ph5tzczb2tfjatul31bs6x4&dl=1
    )

    FetchContent_GetProperties(libzip)
    if(NOT libzip_POPULATED)
    FetchContent_Populate(libzip)
    
    # Set LIBZIP options
    set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared libraries" FORCE)
    set(BUILD_TOOLS OFF CACHE BOOL "Build tools" FORCE)
    set(BUILD_REGRESS OFF CACHE BOOL "Build regression tests" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
    set(BUILD_DOC OFF CACHE BOOL "Build documentation" FORCE)
    
    add_subdirectory(${libzip_SOURCE_DIR} ${libzip_BINARY_DIR})
    
    # Export LIBZIP as a target
    add_library(LIBZIP::LIBZIP INTERFACE IMPORTED)
    set_target_properties(LIBZIP::LIBZIP PROPERTIES
        INTERFACE_LINK_LIBRARIES zip
        INTERFACE_INCLUDE_DIRECTORIES "${libzip_SOURCE_DIR}/lib;${libzip_BINARY_DIR}"
    )
    endif()
    ]])
endif()

# FFTW
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/deps/FFTW_custom.cmake")
    file(WRITE ${CMAKE_SOURCE_DIR}/cmake/deps/FFTW.cmake [[
    FetchContent_Declare(
    fftw
    URL https://www.dropbox.com/scl/fi/yugsuwobadxt5gvfsdz46/fftw-3.3.10.tar.gz?rlkey=cfo1rwazrr4gbm2k043np8skj&dl=1
    )

    FetchContent_GetProperties(fftw)
    if(NOT fftw_POPULATED)
    FetchContent_Populate(fftw)
    
    # FFTW is a bit tricky, we'll use ExternalProject
    # because FFTW doesn't have a CMakeLists.txt
    
    set(FFTW_SHARED_ARG)
    if(BUILD_SHARED_LIBS)
        set(FFTW_SHARED_ARG --enable-shared --disable-static)
    else()
        set(FFTW_SHARED_ARG --disable-shared --enable-static)
    endif()
    
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
        set(FFTW_OPTIONS --enable-float --disable-fortran)
    else()
        set(FFTW_OPTIONS --enable-float --disable-fortran --enable-sse --enable-sse2 --enable-avx --enable-avx2 --enable-avx-128-fma --enable-generic-simd128 --enable-generic-simd256)
    endif()
        
    ExternalProject_Add(fftw_external
        SOURCE_DIR ${fftw_SOURCE_DIR}
        BINARY_DIR ${fftw_BINARY_DIR}
        CONFIGURE_COMMAND ${fftw_SOURCE_DIR}/configure 
                        --prefix=${CMAKE_BINARY_DIR}/deps
                        ${FFTW_SHARED_ARG}
                        ${FFTW_OPTIONS}
        BUILD_COMMAND make -j8
        INSTALL_COMMAND make install
        BUILD_ALWAYS OFF
    )
    
    # Create interface library
    add_library(FFTW::FFTW INTERFACE IMPORTED)
    set_target_properties(FFTW::FFTW PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/deps/include"
    )
    
    if(BUILD_SHARED_LIBS)
        set_target_properties(FFTW::FFTW PROPERTIES
        INTERFACE_LINK_LIBRARIES "${CMAKE_BINARY_DIR}/deps/lib/libfftw3f${CMAKE_SHARED_LIBRARY_SUFFIX}"
        )
    else()
        set_target_properties(FFTW::FFTW PROPERTIES
        INTERFACE_LINK_LIBRARIES "${CMAKE_BINARY_DIR}/deps/lib/libfftw3f${CMAKE_STATIC_LIBRARY_SUFFIX}"
        )
    endif()
    
    # Add dependency
    add_dependencies(FFTW::FFTW fftw_external)
    endif()
    ]])
endif()

# Handle all dependencies
# handle_dependency(ZLIB "https://www.dropbox.com/scl/fi/vrgb8i9755eojx5eh1a2r/zlib-1.3.1.tar.gz?rlkey=9jwwjc1aqmflin5b75r2lc2yw&dl=0" DOWNLOAD_DEPENDENCIES)
handle_dependency(BLOSC "https://github.com/Blosc/c-blosc2" DOWNLOAD_DEPENDENCIES)
handle_dependency(LIBTIFF "https://www.dropbox.com/scl/fi/tk3axrjtjgxmjj9hzsk13/tiff-4.6.0.tar.gz?rlkey=judqzxze5g4sg0bviyul8kqvp&dl=0" DOWNLOAD_DEPENDENCIES)
handle_dependency(HDF5 "https://www.dropbox.com/scl/fi/ib4wkq1s9jhm0oi9n6r7c/hdf5-1.14.2.tar.gz?rlkey=eq20hs7juecpwcn1vuumssjuf&dl=0" DOWNLOAD_DEPENDENCIES)
handle_dependency(BLOSC1 "https://github.com/Blosc/c-blosc" DOWNLOAD_DEPENDENCIES)
handle_dependency(NLOPT "https://www.dropbox.com/scl/fi/ux4ccf23z7rotkgbqbrmk/nlopt-2.4.2.tar.gz?rlkey=afq6l6yyu9fnw1hpe62l4gwqq&dl=0" DOWNLOAD_DEPENDENCIES)
handle_dependency(LIBZIP "https://www.dropbox.com/scl/fi/2mo9gzxi8ms3pp10pu6ad/libzip-1.10.1.tar.gz?rlkey=w7ph5tzczb2tfjatul31bs6x4&dl=0" DOWNLOAD_DEPENDENCIES)
handle_dependency(FFTW "https://www.dropbox.com/scl/fi/yugsuwobadxt5gvfsdz46/fftw-3.3.10.tar.gz?rlkey=cfo1rwazrr4gbm2k043np8skj&dl=0" DOWNLOAD_DEPENDENCIES)

# Define common compiler flags
set(COMMON_C_FLAGS_LIST
  -fPIC
  -O3
  -w
)

# Platform-specific compiler flags
if(APPLE)
  list(APPEND COMMON_C_FLAGS_LIST
    -std=gnu99
  )
  # On macOS, some versions of Clang don't support -fgnu89-inline
  if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
    list(APPEND COMMON_C_FLAGS_LIST -fgnu89-inline)
  endif()
else()
  list(APPEND COMMON_C_FLAGS_LIST -fgnu89-inline)
endif()

# Convert list to string for older CMake versions
string(REPLACE ";" " " COMMON_C_FLAGS "${COMMON_C_FLAGS_LIST}")

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_C_FLAGS}")

# Add linker flags properly
if(NOT APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ldl -lm")
else()
  # macOS doesn't need explicit -ldl
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")
endif()

# Define link libraries
set(COMMON_LINK_LIBRARIES)

if(TARGET NLOPT::NLOPT)
  list(APPEND COMMON_LINK_LIBRARIES NLOPT::NLOPT)
endif()

# if(TARGET ZLIB::ZLIB)
#   list(APPEND COMMON_LINK_LIBRARIES ZLIB::ZLIB)
# endif()

if(TARGET LIBTIFF::LIBTIFF)
  list(APPEND COMMON_LINK_LIBRARIES LIBTIFF::LIBTIFF)
endif()

if(TARGET HDF5::HDF5 AND TARGET HDF5::HL)
  list(APPEND COMMON_LINK_LIBRARIES HDF5::HDF5 HDF5::HL)
endif()

if(TARGET BLOSC::BLOSC)
  list(APPEND COMMON_LINK_LIBRARIES BLOSC::BLOSC)
endif()

if(TARGET BLOSC1::BLOSC1)
  list(APPEND COMMON_LINK_LIBRARIES BLOSC1::BLOSC1)
endif()

if(TARGET LIBZIP::LIBZIP)
  list(APPEND COMMON_LINK_LIBRARIES LIBZIP::LIBZIP)
endif()

if(TARGET FFTW::FFTW)
  list(APPEND COMMON_LINK_LIBRARIES FFTW::FFTW)
endif()

# Create a directory for source file libraries
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/src)

# Common source files that are shared between multiple executables
set(COMMON_SOURCES
  src/CalcDiffractionSpots.c
  src/CalcPeakProfile.c
  src/GetMisorientation.c
  src/CalcStrains.c
  src/sgclib.c
  src/sgfind.c
  src/sghkl.c
  src/sgsi.c
  src/sgio.c
)

# Function to add an executable with the appropriate libraries
function(add_midas_executable TARGET_NAME)
  set(options CUDA OMP)
  set(oneValueArgs)
  set(multiValueArgs SOURCES LINK_LIBRARIES)
  cmake_parse_arguments(MIDAS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  if(MIDAS_CUDA AND NOT BUILD_CUDA)
    message(STATUS "Skipping CUDA executable ${TARGET_NAME} as CUDA is not enabled")
    return()
  endif()
  
  if(MIDAS_OMP AND NOT OpenMP_C_FOUND)
    message(STATUS "Skipping OpenMP executable ${TARGET_NAME} as OpenMP is not found")
    return()
  endif()
  
  if(MIDAS_CUDA)
    add_executable(${TARGET_NAME} ${MIDAS_SOURCES})
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  else()
    add_executable(${TARGET_NAME} ${MIDAS_SOURCES})
  endif()
  
  target_link_libraries(${TARGET_NAME} PRIVATE ${COMMON_LINK_LIBRARIES} ${MIDAS_LINK_LIBRARIES})
  
  if(MIDAS_OMP AND OpenMP_C_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_C)
  endif()
  
  # Add special handling for macOS rpath
  if(APPLE AND APPLE_RPATH_FIX)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND install_name_tool -add_rpath "${CMAKE_BINARY_DIR}/deps/lib" $<TARGET_FILE:${TARGET_NAME}>
      COMMENT "Adding rpath to ${TARGET_NAME}"
    )
  endif()
  
  install(TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

  # Also copy to bin in the source directory
  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/bin"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> "${CMAKE_SOURCE_DIR}/bin/"
    COMMENT "Copying ${TARGET_NAME} to ${CMAKE_SOURCE_DIR}/bin/"
  )
endfunction()

# Now add all the executables
add_midas_executable(Calibrant
  SOURCES src/Calibrant.c src/CalcPeakProfile.c
  LINK_LIBRARIES NLOPT::NLOPT TIFF::TIFF HDF5::HDF5 HDF5::HL
)

add_midas_executable(CalibrantOMP
  SOURCES src/CalibrantOMP.c src/CalcPeakProfile.c
  LINK_LIBRARIES NLOPT::NLOPT TIFF::TIFF HDF5::HDF5 HDF5::HL
  OMP
)

add_midas_executable(FitSetup
  SOURCES src/FitSetupParamsAll.c
  LINK_LIBRARIES NLOPT::NLOPT
)

add_midas_executable(FitTiltBCLsdSample
  SOURCES src/FitTiltBCLsdSampleOmegaCorrection.c
  LINK_LIBRARIES NLOPT::NLOPT
)

add_midas_executable(ForwardSimulation
  SOURCES src/ForwardSimulation.c
)

add_midas_executable(ForwardSimulationCompressed
  SOURCES src/ForwardSimulationCompressed.c
  LINK_LIBRARIES BLOSC1::BLOSC1 BLOSC::BLOSC LIBZIP::LIBZIP
)

add_midas_executable(FitPosOrStrains
  SOURCES src/FitPosOrStrains.c src/CalcDiffractionSpots.c
  LINK_LIBRARIES NLOPT::NLOPT
)

add_midas_executable(FitPosOrStrainsOMP
  SOURCES src/FitPosOrStrainsOMP.c src/CalcDiffractionSpots.c
  LINK_LIBRARIES NLOPT::NLOPT
  OMP
)

add_midas_executable(FitOrStrainsScanningOMP
  SOURCES src/FitOrStrainsScanningOMP.c src/CalcDiffractionSpots.c
  LINK_LIBRARIES NLOPT::NLOPT
  OMP
)

add_midas_executable(FitPosOrStrainsScanningHEDM
  SOURCES src/FitPosOrStrainsScanningHEDM.c src/CalcDiffractionSpots.c
  LINK_LIBRARIES NLOPT::NLOPT
)

add_midas_executable(PeaksFittingPerFile
  SOURCES src/PeaksFittingPerFile.c
  LINK_LIBRARIES NLOPT::NLOPT
)

add_midas_executable(PeaksFittingOMP
  SOURCES src/PeaksFittingOMP.c
  LINK_LIBRARIES NLOPT::NLOPT
  OMP
)

add_midas_executable(MapMultipleDetectors
  SOURCES src/MapMultipleDetectors.c
)

add_midas_executable(MergeOverlappingPeaks
  SOURCES src/MergeOverlappingPeaks.c
)

add_midas_executable(MergeOverlappingPeaksAll
  SOURCES src/MergeOverlappingPeaksAll.c
)

add_midas_executable(CalcRadius
  SOURCES src/CalcRadius.c
)

add_midas_executable(CalcRadiusAll
  SOURCES src/CalcRadiusAll.c
)

add_midas_executable(FindSaturatedPixels
  SOURCES src/FindSaturatedPixels.c
)

add_midas_executable(GrainTracking
  SOURCES src/GrainTracking.c
)

add_midas_executable(MergeMultipleRings
  SOURCES src/MergeMultipleRings.c
)

add_midas_executable(GenMedianDark
  SOURCES src/GenMedianDark.c
)

add_midas_executable(FitGrain
  SOURCES src/FitGrain.c src/CalcDiffractionSpots.c
  LINK_LIBRARIES NLOPT::NLOPT
)

add_midas_executable(FitGrainHydra
  SOURCES src/FitGrainHydra.c src/CalcDiffractionSpots.c
  LINK_LIBRARIES NLOPT::NLOPT
)

add_midas_executable(FitScanningGrain
  SOURCES src/FitScanningGrain.c
)

add_midas_executable(FitTiltX
  SOURCES src/FitTiltX.c
  LINK_LIBRARIES NLOPT::NLOPT
)

add_midas_executable(FitWedge
  SOURCES src/FitWedge.c
  LINK_LIBRARIES NLOPT::NLOPT
)

add_midas_executable(GetHKLList
  SOURCES src/GetHKLList.c src/sgclib.c src/sgfind.c src/sghkl.c src/sgsi.c src/sgio.c
)

add_midas_executable(PeaksFittingOMPZarr
  SOURCES src/PeaksFittingOMPZarr.c
  LINK_LIBRARIES NLOPT::NLOPT BLOSC::BLOSC LIBZIP::LIBZIP
  OMP
)

add_midas_executable(PeaksFittingOMPZarrRefactor
  SOURCES src/PeaksFittingOMPZarrRefactor.c
  LINK_LIBRARIES NLOPT::NLOPT BLOSC::BLOSC LIBZIP::LIBZIP
  OMP
)

add_midas_executable(GetHKLListZarr
  SOURCES src/GetHKLListZarr.c src/sgclib.c src/sgfind.c src/sghkl.c src/sgsi.c src/sgio.c
  LINK_LIBRARIES BLOSC::BLOSC LIBZIP::LIBZIP
)

add_midas_executable(MergeOverlappingPeaksAllZarr
  SOURCES src/MergeOverlappingPeaksAllZarr.c
  LINK_LIBRARIES BLOSC::BLOSC LIBZIP::LIBZIP
)

add_midas_executable(CalcRadiusAllZarr
  SOURCES src/CalcRadiusAllZarr.c
  LINK_LIBRARIES BLOSC::BLOSC LIBZIP::LIBZIP
)

add_midas_executable(FitSetupZarr
  SOURCES src/FitSetupParamsAllZarr.c
  LINK_LIBRARIES NLOPT::NLOPT BLOSC::BLOSC LIBZIP::LIBZIP
)

add_midas_executable(ProcessGrainsZarr
  SOURCES src/ProcessGrainsZarr.c src/GetMisorientation.c src/CalcStrains.c
  LINK_LIBRARIES NLOPT::NLOPT BLOSC::BLOSC LIBZIP::LIBZIP
)

add_midas_executable(IndexerLinuxArgsShm
  SOURCES src/IndexerLinuxArgsOptimizedShm.c
)

add_midas_executable(IndexerOMP
  SOURCES src/IndexerOMP.c
  OMP
)

add_midas_executable(IndexerScanningOMP
  SOURCES src/IndexerScanningOMP.c
  OMP
)

add_midas_executable(IndexScanningHEDM
  SOURCES src/IndexScanningHEDM.c src/CalcDiffractionSpots.c
)

add_midas_executable(SaveBinData
  SOURCES src/SaveBinData.c
)

add_midas_executable(SaveBinDataScanning
  SOURCES src/SaveBinDataScanning.c
)

add_midas_executable(MergeMultipleScans
  SOURCES src/MergeMultipleScans.c
)

add_midas_executable(ProcessGrains
  SOURCES src/ProcessGrains.c src/GetMisorientation.c src/CalcStrains.c
  LINK_LIBRARIES NLOPT::NLOPT
)

add_midas_executable(ProcessGrainsScanningHEDM
  SOURCES src/ProcessGrainsScanningHEDM.c src/GetMisorientation.c src/CalcStrains.c
  LINK_LIBRARIES NLOPT::NLOPT
)

add_midas_executable(MatchGrains
  SOURCES src/MatchGrains.c src/GetMisorientation.c
)

add_midas_executable(DetectorMapper
  SOURCES src/DetectorMapper.c
)

add_midas_executable(DetectorMapperZarr
  SOURCES src/DetectorMapperZarr.c
  LINK_LIBRARIES BLOSC::BLOSC LIBZIP::LIBZIP
)

add_midas_executable(Integrator
  SOURCES src/Integrator.c
  LINK_LIBRARIES TIFF::TIFF HDF5::HDF5 HDF5::HL
)

add_midas_executable(IntegratorZarr
  SOURCES src/IntegratorZarr.c
  LINK_LIBRARIES TIFF::TIFF HDF5::HDF5 HDF5::HL BLOSC::BLOSC LIBZIP::LIBZIP
)

add_midas_executable(ConvTiffToGE
  SOURCES src/ConvTiffToGE.c
)

add_midas_executable(MapBND
  SOURCES src/MapBND.c
)

add_midas_executable(findSingleSolutionPF
  SOURCES src/findSingleSolutionPF.c src/GetMisorientation.c
  OMP
)

add_midas_executable(findMultipleSolutionsPF
  SOURCES src/findMultipleSolutionsPF.c src/GetMisorientation.c
  OMP
)

add_midas_executable(mergeScansScanning
  SOURCES src/mergeScansScanning.c
  OMP
)

if(BUILD_CUDA)
  add_midas_executable(Indexer
    SOURCES src/IndexerRefineNLOPT.cu
    CUDA
  )
  
  add_midas_executable(Peaks
    SOURCES src/PeaksNLMD.cu
    CUDA
  )
endif()

# Add a custom target for building all non-CUDA MIDAS executables
add_custom_target(midas_executables DEPENDS 
    Calibrant
    CalibrantOMP
    FitSetup
    FitTiltBCLsdSample
    ForwardSimulation
    ForwardSimulationCompressed
    FitPosOrStrains
    FitPosOrStrainsOMP
    FitOrStrainsScanningOMP
    FitPosOrStrainsScanningHEDM
    PeaksFittingPerFile
    PeaksFittingOMP
    MapMultipleDetectors
    MergeOverlappingPeaks
    MergeOverlappingPeaksAll
    CalcRadius
    CalcRadiusAll
    FindSaturatedPixels
    GrainTracking
    MergeMultipleRings
    GenMedianDark
    FitGrain
    FitGrainHydra
    FitScanningGrain
    FitTiltX
    FitWedge
    GetHKLList
    PeaksFittingOMPZarr
    PeaksFittingOMPZarrRefactor
    GetHKLListZarr
    MergeOverlappingPeaksAllZarr
    CalcRadiusAllZarr
    FitSetupZarr
    ProcessGrainsZarr
    IndexerLinuxArgsShm
    IndexerOMP
    IndexerScanningOMP
    IndexScanningHEDM
    SaveBinData
    SaveBinDataScanning
    MergeMultipleScans
    ProcessGrains
    ProcessGrainsScanningHEDM
    MatchGrains
    DetectorMapper
    DetectorMapperZarr
    Integrator
    IntegratorZarr
    ConvTiffToGE
    MapBND
    findSingleSolutionPF
    findMultipleSolutionsPF
    mergeScansScanning
)

# Handle Python dependencies
if(INSTALL_PYTHON_DEPS)
  # Find Python
  if(NOT PYTHON_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
      set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE} CACHE FILEPATH "Path to Python executable" FORCE)
    else()
      find_program(PYTHON_EXECUTABLE python3)
      if(NOT PYTHON_EXECUTABLE)
        find_program(PYTHON_EXECUTABLE python)
        if(NOT PYTHON_EXECUTABLE)
          message(WARNING "Python not found. Cannot install Python dependencies.")
          set(INSTALL_PYTHON_DEPS OFF)
        endif()
      endif()
    endif()
  endif()

  if(INSTALL_PYTHON_DEPS AND PYTHON_EXECUTABLE)
    # Check for requirements.txt
    if(EXISTS "${CMAKE_SOURCE_DIR}/requirements.txt")
      # Create custom target for installing Python dependencies
      add_custom_target(python_deps
        COMMAND ${PYTHON_EXECUTABLE} -m pip install --user -r ${CMAKE_SOURCE_DIR}/requirements.txt
        COMMENT "Installing Python dependencies from requirements.txt"
        VERBATIM
      )
      # Note: we don't add a dependency to 'all' here to avoid
      # running pip install on every build
    else()
      message(WARNING "requirements.txt not found. Cannot install Python dependencies.")
      set(INSTALL_PYTHON_DEPS OFF)
    endif()
  endif()
endif()

# Generate environment setup script
configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/templates/setup.sh.in
  ${CMAKE_BINARY_DIR}/setup.sh
  @ONLY
)

install(PROGRAMS
  ${CMAKE_BINARY_DIR}/setup.sh
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Print configuration summary
message(STATUS "")
message(STATUS "MIDAS FF_HEDM Configuration Summary")
message(STATUS "------------------------------------")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build CUDA components: ${BUILD_CUDA}")
message(STATUS "Build OpenMP components: ${BUILD_OMP}")
message(STATUS "Build shared libraries: ${BUILD_SHARED_LIBS}")
message(STATUS "Download dependencies: ${DOWNLOAD_DEPENDENCIES}")
message(STATUS "Dependencies install directory: ${DEPS_INSTALL_DIR}")
message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")
message(STATUS "Install Python dependencies: ${INSTALL_PYTHON_DEPS}")
message(STATUS "Configuration directory: ${MIDAS_CONFIG_DIR}")
message(STATUS "------------------------------------")
message(STATUS "")