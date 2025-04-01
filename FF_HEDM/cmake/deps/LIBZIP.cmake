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
