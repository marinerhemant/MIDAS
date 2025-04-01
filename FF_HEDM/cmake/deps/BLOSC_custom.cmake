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
  
  # Enable using external zlib instead of internal one
  set(BLOSC_PREFER_EXTERNAL_ZLIB ON CACHE BOOL "Use external ZLIB" FORCE)
  set(BLOSC_PREFER_EXTERNAL_LZ4 ON CACHE BOOL "Use external LZ4" FORCE)
  
  # Disable internal zlib building completely
  set(DEACTIVATE_ZLIB ON CACHE BOOL "Deactivate internal zlib" FORCE)
  
  # Find system zlib
  find_package(ZLIB REQUIRED)
  
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