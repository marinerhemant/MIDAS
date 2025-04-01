FetchContent_Declare(
  blosc1
  GIT_REPOSITORY https://github.com/Blosc/c-blosc.git
  GIT_TAG main
)

FetchContent_GetProperties(blosc1)
if(NOT blosc1_POPULATED)
  FetchContent_Populate(blosc1)
  
  # Fix the incorrect option syntax in CMakeLists.txt
  if(EXISTS "${blosc1_SOURCE_DIR}/CMakeLists.txt")
    file(READ "${blosc1_SOURCE_DIR}/CMakeLists.txt" BLOSC1_CMAKE_CONTENT)
    
    # Fix the option line with incorrect arguments
    string(REPLACE "option(BUILD_FUZZERS Build fuzzer programs from the blosc compression library NOT ON)" 
                   "option(BUILD_FUZZERS \"Build fuzzer programs from the blosc compression library\" OFF)" 
                   BLOSC1_CMAKE_CONTENT "${BLOSC1_CMAKE_CONTENT}")
    
    file(WRITE "${blosc1_SOURCE_DIR}/CMakeLists.txt" "${BLOSC1_CMAKE_CONTENT}")
  endif()
  
  # Set BLOSC1 options
  set(BLOSC_IS_SUBPROJECT ON CACHE BOOL "Build as subproject" FORCE)
  set(BLOSC_INSTALL OFF CACHE BOOL "Install blosc" FORCE)
  set(BUILD_SHARED ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared" FORCE)
  set(BUILD_STATIC NOT ${BUILD_SHARED_LIBS} CACHE BOOL "Build static" FORCE)
  set(BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
  set(BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmarks" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
  set(BUILD_FUZZERS OFF CACHE BOOL "Build fuzzer programs" FORCE)
  
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