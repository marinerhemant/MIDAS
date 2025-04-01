FetchContent_Declare(
  blosc
  GIT_REPOSITORY https://github.com/Blosc/c-blosc2.git
  GIT_TAG main
)

FetchContent_GetProperties(blosc)
if(NOT blosc_POPULATED)
  FetchContent_Populate(blosc)
  
  # Rename the internal zlib to avoid conflict
  if(EXISTS "${blosc_SOURCE_DIR}/internal-complibs/zlib-ng-2.0.7/CMakeLists.txt")
    file(READ "${blosc_SOURCE_DIR}/internal-complibs/zlib-ng-2.0.7/CMakeLists.txt" ZLIB_CMAKE_CONTENT)
    
    # Replace the target name "zlib" with "blosc_internal_zlib" in all places
    string(REPLACE "add_library(zlib" 
                   "add_library(blosc_internal_zlib" 
                   ZLIB_CMAKE_CONTENT "${ZLIB_CMAKE_CONTENT}")
    
    # Replace any references to "zlib" target with "blosc_internal_zlib"
    string(REPLACE "TARGET zlib" 
                   "TARGET blosc_internal_zlib" 
                   ZLIB_CMAKE_CONTENT "${ZLIB_CMAKE_CONTENT}")
    
    # Update the output name to avoid filename conflicts
    string(REPLACE "OUTPUT_NAME z" 
                   "OUTPUT_NAME blosc_z" 
                   ZLIB_CMAKE_CONTENT "${ZLIB_CMAKE_CONTENT}")
    
    file(WRITE "${blosc_SOURCE_DIR}/internal-complibs/zlib-ng-2.0.7/CMakeLists.txt" "${ZLIB_CMAKE_CONTENT}")
  endif()
  
  # Also update BLOSC's main CMakeLists.txt to use the renamed zlib target
  if(EXISTS "${blosc_SOURCE_DIR}/CMakeLists.txt")
    file(READ "${blosc_SOURCE_DIR}/CMakeLists.txt" BLOSC_CMAKE_CONTENT)
    
    # Replace references to "zlib" with "blosc_internal_zlib" 
    string(REPLACE "zlib " 
                   "blosc_internal_zlib " 
                   BLOSC_CMAKE_CONTENT "${BLOSC_CMAKE_CONTENT}")
    
    string(REPLACE "zlib;" 
                   "blosc_internal_zlib;" 
                   BLOSC_CMAKE_CONTENT "${BLOSC_CMAKE_CONTENT}")
    
    file(WRITE "${blosc_SOURCE_DIR}/CMakeLists.txt" "${BLOSC_CMAKE_CONTENT}")
  endif()
  
  # Set BLOSC options
  set(BLOSC_IS_SUBPROJECT ON CACHE BOOL "Build as subproject" FORCE)
  set(BLOSC_INSTALL OFF CACHE BOOL "Install blosc" FORCE)
  set(BLOSC_SHARED ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared" FORCE)
  set(BLOSC_STATIC NOT ${BUILD_SHARED_LIBS} CACHE BOOL "Build static" FORCE)
  set(BLOSC_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
  set(BLOSC_BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmarks" FORCE)
  set(BLOSC_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
  
  # Disable external zlib to use internal one
  set(BLOSC_PREFER_EXTERNAL_ZLIB OFF CACHE BOOL "Use external ZLIB" FORCE)
  
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