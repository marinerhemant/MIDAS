FetchContent_Declare(
  blosc1
  GIT_REPOSITORY https://github.com/Blosc/c-blosc.git
  GIT_TAG main
)

FetchContent_GetProperties(blosc1)
if(NOT blosc1_POPULATED)
  FetchContent_Populate(blosc1)
  
  # Fix the problematic BUILD_FUZZERS option in CMakeLists.txt using CMake file operations
  file(READ "${blosc1_SOURCE_DIR}/CMakeLists.txt" cmake_content)
  string(REPLACE 
    "option(BUILD_FUZZERS \"Build fuzzer programs from the blosc compression library\" NOT ON)" 
    "option(BUILD_FUZZERS \"Build fuzzer programs\" OFF)" 
    cmake_content_fixed "${cmake_content}")
  file(WRITE "${blosc1_SOURCE_DIR}/CMakeLists.txt" "${cmake_content_fixed}")
  
  # Set BLOSC1 options
  set(BLOSC_IS_SUBPROJECT ON CACHE BOOL "Build as subproject" FORCE)
  set(BLOSC_INSTALL OFF CACHE BOOL "Install blosc" FORCE)
  set(BUILD_SHARED ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared" FORCE)
  set(BUILD_STATIC NOT ${BUILD_SHARED_LIBS} CACHE BOOL "Build static" FORCE)
  set(BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
  set(BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmarks" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
  
  # Always use internal zlib in BLOSC1
  set(DEACTIVATE_ZLIB OFF CACHE BOOL "Do not include support for the Zlib library." FORCE)
  
  # Avoid issues with other dependencies
  set(DEACTIVATE_SNAPPY ON CACHE BOOL "Do not include support for the Snappy library." FORCE)
  
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