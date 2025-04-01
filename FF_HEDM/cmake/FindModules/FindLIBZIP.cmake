# FindLIBZIP.cmake - Find LIBZIP library
#
# This module defines:
#  LIBZIP_FOUND        - True if LIBZIP is found
#  LIBZIP_INCLUDE_DIRS - LIBZIP include directories
#  LIBZIP_LIBRARIES    - LIBZIP libraries
#  LIBZIP::LIBZIP      - LIBZIP imported target

# Try to find LIBZIP in standard paths
find_path(LIBZIP_INCLUDE_DIR
  NAMES zip.h
  PATHS
    /usr/include
    /usr/local/include
    /opt/local/include
    ${CMAKE_PREFIX_PATH}/include
    ENV LIBZIP_DIR
  PATH_SUFFIXES libzip zip
)

# Find both shared and static libraries
find_library(LIBZIP_LIBRARY
  NAMES zip libzip
  PATHS
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /opt/local/lib
    ${CMAKE_PREFIX_PATH}/lib
    ENV LIBZIP_DIR
  PATH_SUFFIXES lib lib64
)

# Set standard variable names
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBZIP
  REQUIRED_VARS LIBZIP_LIBRARY LIBZIP_INCLUDE_DIR
)

if(LIBZIP_FOUND)
  set(LIBZIP_LIBRARIES ${LIBZIP_LIBRARY})
  set(LIBZIP_INCLUDE_DIRS ${LIBZIP_INCLUDE_DIR})
  
  if(NOT TARGET LIBZIP::LIBZIP)
    add_library(LIBZIP::LIBZIP INTERFACE IMPORTED)
    set_target_properties(LIBZIP::LIBZIP PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${LIBZIP_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${LIBZIP_LIBRARIES}"
    )
  endif()
endif()

mark_as_advanced(LIBZIP_INCLUDE_DIR LIBZIP_LIBRARY)