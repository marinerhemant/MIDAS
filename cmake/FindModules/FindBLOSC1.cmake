# FindBLOSC1.cmake - Find BLOSC1 library
#
# This module defines:
#  BLOSC1_FOUND        - True if BLOSC1 is found
#  BLOSC1_INCLUDE_DIRS - BLOSC1 include directories
#  BLOSC1_LIBRARIES    - BLOSC1 libraries
#  BLOSC1::BLOSC1      - BLOSC1 imported target

# Try to find BLOSC1 in standard paths
find_path(BLOSC1_INCLUDE_DIR
  NAMES blosc.h
  PATHS
    /usr/include
    /usr/local/include
    /opt/local/include
    ${CMAKE_PREFIX_PATH}/include
    ENV BLOSC1_DIR
)

# Find both shared and static libraries
find_library(BLOSC1_LIBRARY
  NAMES blosc libblosc
  PATHS
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /opt/local/lib
    ${CMAKE_PREFIX_PATH}/lib
    ENV BLOSC1_DIR
  PATH_SUFFIXES lib lib64
)

# Set standard variable names
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLOSC1
  REQUIRED_VARS BLOSC1_LIBRARY BLOSC1_INCLUDE_DIR
)

if(BLOSC1_FOUND)
  set(BLOSC1_LIBRARIES ${BLOSC1_LIBRARY})
  set(BLOSC1_INCLUDE_DIRS ${BLOSC1_INCLUDE_DIR})
  
  if(NOT TARGET BLOSC1::BLOSC1)
    add_library(BLOSC1::BLOSC1 INTERFACE IMPORTED)
    set_target_properties(BLOSC1::BLOSC1 PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${BLOSC1_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${BLOSC1_LIBRARIES}"
    )
  endif()
endif()

mark_as_advanced(BLOSC1_INCLUDE_DIR BLOSC1_LIBRARY)