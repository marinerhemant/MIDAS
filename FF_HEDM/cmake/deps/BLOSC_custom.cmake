FetchContent_Declare(
  blosc
  GIT_REPOSITORY https://github.com/Blosc/c-blosc2.git
  GIT_TAG main
)

FetchContent_GetProperties(blosc)
if(NOT blosc_POPULATED)
  FetchContent_Populate(blosc)
  
  # Disable internal zlib by modifying its CMakeLists.txt before processing
  if(EXISTS "${blosc_SOURCE_DIR}/internal-complibs/zlib-ng-2.0.7/CMakeLists.txt")
    file(WRITE "${blosc_SOURCE_DIR}/internal-complibs/zlib-ng-2.0.7/CMakeLists.txt" "
      # Disabled to avoid conflict with the project's zlib
      message(STATUS \"Using system ZLIB instead of internal zlib-ng\")
      
      # Create dummy targets to satisfy blosc dependencies
      add_library(zlib INTERFACE)
      target_link_libraries(zlib INTERFACE ZLIB::ZLIB)
    ")
  endif()
  
  # Set BLOSC options
  set(BLOSC_IS_SUBPROJECT ON CACHE BOOL "Build as subproject" FORCE)
  set(BLOSC_INSTALL OFF CACHE BOOL "Install blosc" FORCE)
  set(BLOSC_SHARED ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared" FORCE)
  set(BLOSC_STATIC NOT ${BUILD_SHARED_LIBS} CACHE BOOL "Build static" FORCE)
  set(BLOSC_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
  set(BLOSC_BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmarks" FORCE)
  set(BLOSC_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
  
  # Force external zlib
  set(BLOSC_PREFER_EXTERNAL_ZLIB ON CACHE BOOL "Use external ZLIB" FORCE)
  set(DEACTIVATE_ZLIB OFF CACHE BOOL "Deactivate zlib compression" FORCE)
  
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