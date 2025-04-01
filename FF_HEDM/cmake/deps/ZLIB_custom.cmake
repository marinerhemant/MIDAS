FetchContent_Declare(
  zlib
  URL https://www.dropbox.com/scl/fi/vrgb8i9755eojx5eh1a2r/zlib-1.3.1.tar.gz?rlkey=9jwwjc1aqmflin5b75r2lc2yw&dl=1
)

FetchContent_GetProperties(zlib)
if(NOT zlib_POPULATED)
  FetchContent_Populate(zlib)
  
  # Configure zlib build options
  set(ZLIB_BUILD_SHARED ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared libraries" FORCE)
  set(ZLIB_BUILD_STATIC NOT ${BUILD_SHARED_LIBS} CACHE BOOL "Build static libraries" FORCE)
  
  add_subdirectory(${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
  
  # Define a real imported library, not just an interface
  if(BUILD_SHARED_LIBS)
    add_library(ZLIB::ZLIB SHARED IMPORTED GLOBAL)
    set_target_properties(ZLIB::ZLIB PROPERTIES
      IMPORTED_LOCATION "${zlib_BINARY_DIR}/libz${CMAKE_SHARED_LIBRARY_SUFFIX}"
      IMPORTED_SONAME "libz${CMAKE_SHARED_LIBRARY_SUFFIX}"
      INTERFACE_INCLUDE_DIRECTORIES "${zlib_SOURCE_DIR};${zlib_BINARY_DIR}"
    )
  else()
    add_library(ZLIB::ZLIB STATIC IMPORTED GLOBAL)
    set_target_properties(ZLIB::ZLIB PROPERTIES
      IMPORTED_LOCATION "${zlib_BINARY_DIR}/libz${CMAKE_STATIC_LIBRARY_SUFFIX}"
      INTERFACE_INCLUDE_DIRECTORIES "${zlib_SOURCE_DIR};${zlib_BINARY_DIR}"
    )
  endif()
  
  # Add a dependency to ensure zlib is built before HDF5 tries to use it
  add_dependencies(ZLIB::ZLIB zlib)
endif()