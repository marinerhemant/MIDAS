FetchContent_Declare(
  blosc1
  GIT_REPOSITORY https://github.com/Blosc/c-blosc.git
  GIT_TAG v1.21.6
)

FetchContent_GetProperties(blosc1)
if(NOT blosc1_POPULATED)
  FetchContent_Populate(blosc1)
  
  # Set BLOSC1 options
  set(BLOSC_IS_SUBPROJECT ON CACHE BOOL "Build as subproject" FORCE)
  set(BLOSC_INSTALL OFF CACHE BOOL "Install blosc" FORCE)
  
  # Set BUILD_SHARED based on BUILD_SHARED_LIBS
  set(BUILD_SHARED ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared" FORCE)
  
  # Explicitly set BUILD_STATIC to ON to satisfy BUILD_FUZZERS requirement
  set(BUILD_STATIC ON CACHE BOOL "Build static" FORCE)
  
  # Explicitly disable BUILD_FUZZERS
  set(BUILD_FUZZERS OFF CACHE BOOL "Build fuzzer programs" FORCE)
  
  set(BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
  set(BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmarks" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
  
  # Always use internal zlib in BLOSC1
  set(DEACTIVATE_ZLIB OFF CACHE BOOL "Do not include support for the Zlib library." FORCE)

  # Avoid issues with other dependencies
  set(DEACTIVATE_SNAPPY ON CACHE BOOL "Do not include support for the Snappy library." FORCE)
  # Use the already-fetched external zlib.  blosc1/blosc/CMakeLists.txt checks
  # ZLIB_FOUND (leaked from our top-level FetchContent cache) and links to
  # ZLIB_LIBRARY by file path.  Setting PREFER_EXTERNAL_ZLIB=ON ensures the
  # find_package(ZLIB) path runs, and since we already populated ZLIB_LIBRARY
  # and ZLIB_INCLUDE_DIR, it finds the fetched zlib properly.
  set(PREFER_EXTERNAL_ZLIB ON CACHE BOOL "Use external ZLIB for Blosc1" FORCE)

  # blosc1 v1.21.x has cmake_minimum_required(VERSION 2.8.12) which newer
  # CMake rejects.  Allow the old policy version without editing the source.
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
  add_subdirectory(${blosc1_SOURCE_DIR} ${blosc1_BINARY_DIR})

  # blosc1 links to ZLIB_LIBRARY by file path, not target.
  # Add an explicit target dependency so libz.so is built first.
  if(TARGET blosc_shared AND TARGET zlib)
    add_dependencies(blosc_shared zlib)
  endif()
  if(TARGET blosc_static AND TARGET zlib)
    add_dependencies(blosc_static zlib)
  endif()
  
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