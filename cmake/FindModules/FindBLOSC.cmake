# FindBLOSC.cmake - Find BLOSC2 library
#
# This module defines:
#  BLOSC_FOUND        - True if BLOSC is found
#  BLOSC_INCLUDE_DIRS - BLOSC include directories
#  BLOSC_LIBRARIES    - BLOSC libraries
#  BLOSC::BLOSC       - BLOSC imported target

# Try to find BLOSC in standard paths
find_path(BLOSC_INCLUDE_DIR
  NAMES blosc2.h
  PATHS
    /usr/include
    /usr/local/include
    /opt/local/include
    ${CMAKE_PREFIX_PATH}/include
    ENV BLOSC_DIR
)

# Find both shared and static libraries
find_library(BLOSC_LIBRARY
  NAMES blosc2 libblosc2
  PATHS
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /opt/local/lib
    ${CMAKE_PREFIX_PATH}/lib
    ENV BLOSC_DIR
  PATH_SUFFIXES lib lib64
)

# Set standard variable names
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLOSC
  REQUIRED_VARS BLOSC_LIBRARY BLOSC_INCLUDE_DIR
)

if(BLOSC_FOUND)
  set(BLOSC_LIBRARIES ${BLOSC_LIBRARY})
  set(BLOSC_INCLUDE_DIRS ${BLOSC_INCLUDE_DIR})
  
  if(NOT TARGET BLOSC::BLOSC)
    add_library(BLOSC::BLOSC INTERFACE IMPORTED)
    set_target_properties(BLOSC::BLOSC PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${BLOSC_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${BLOSC_LIBRARIES}"
    )
  endif()
endif()

mark_as_advanced(BLOSC_INCLUDE_DIR BLOSC_LIBRARY)