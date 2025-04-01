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
