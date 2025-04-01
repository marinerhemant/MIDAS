# FindNLOPT.cmake - Find NLOPT library
#
# This module defines:
#  NLOPT_FOUND        - True if NLOPT is found
#  NLOPT_INCLUDE_DIRS - NLOPT include directories
#  NLOPT_LIBRARIES    - NLOPT libraries
#  NLOPT::NLOPT       - NLOPT imported target

# Try to find NLOPT in standard paths
find_path(NLOPT_INCLUDE_DIR
  NAMES nlopt.h
  PATHS
    /usr/include
    /usr/local/include
    /opt/local/include
    ${CMAKE_PREFIX_PATH}/include
    ENV NLOPT_DIR
  PATH_SUFFIXES nlopt
)

# Find both shared and static libraries
find_library(NLOPT_LIBRARY
  NAMES nlopt nlopt_cxx libnlopt libnlopt_cxx
  PATHS
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /opt/local/lib
    ${CMAKE_PREFIX_PATH}/lib
    ENV NLOPT_DIR
  PATH_SUFFIXES lib
)

# Set standard variable names
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NLOPT
  REQUIRED_VARS NLOPT_LIBRARY NLOPT_INCLUDE_DIR
)

if(NLOPT_FOUND)
  set(NLOPT_LIBRARIES ${NLOPT_LIBRARY})
  set(NLOPT_INCLUDE_DIRS ${NLOPT_INCLUDE_DIR})
  
  if(NOT TARGET NLOPT::NLOPT)
    add_library(NLOPT::NLOPT INTERFACE IMPORTED)
    set_target_properties(NLOPT::NLOPT PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${NLOPT_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${NLOPT_LIBRARIES}"
    )
  endif()
endif()

mark_as_advanced(NLOPT_INCLUDE_DIR NLOPT_LIBRARY)