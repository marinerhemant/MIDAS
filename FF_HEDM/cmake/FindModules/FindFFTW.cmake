# FindFFTW.cmake - Find FFTW library
#
# This module defines:
#  FFTW_FOUND        - True if FFTW is found
#  FFTW_INCLUDE_DIRS - FFTW include directories
#  FFTW_LIBRARIES    - FFTW libraries
#  FFTW::FFTW        - FFTW imported target
#
# The following components are supported:
#  FLOAT - Single precision (uses fftw3f library)
#  DOUBLE - Double precision (uses fftw3 library)
#  LONGDOUBLE - Long double precision (uses fftw3l library)

set(_FFTW_REQUIRED_VARS FFTW_INCLUDE_DIR)

# Try to find FFTW in standard paths
find_path(FFTW_INCLUDE_DIR
  NAMES fftw3.h
  PATHS
    /usr/include
    /usr/local/include
    /opt/local/include
    ${CMAKE_PREFIX_PATH}/include
    ENV FFTW_DIR
)

# Check for components or use default (FLOAT)
if(NOT FFTW_FIND_COMPONENTS)
  set(FFTW_FIND_COMPONENTS FLOAT)
endif()

# Find components
foreach(component ${FFTW_FIND_COMPONENTS})
  if(component STREQUAL "FLOAT")
    find_library(FFTW_FLOAT_LIBRARY
      NAMES fftw3f libfftw3f
      PATHS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/local/lib
        ${CMAKE_PREFIX_PATH}/lib
        ENV FFTW_DIR
      PATH_SUFFIXES lib lib64
    )
    list(APPEND _FFTW_REQUIRED_VARS FFTW_FLOAT_LIBRARY)
    list(APPEND FFTW_LIBRARIES ${FFTW_FLOAT_LIBRARY})
  endif()
  
  if(component STREQUAL "DOUBLE")
    find_library(FFTW_DOUBLE_LIBRARY
      NAMES fftw3 libfftw3
      PATHS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/local/lib
        ${CMAKE_PREFIX_PATH}/lib
        ENV FFTW_DIR
      PATH_SUFFIXES lib lib64
    )
    list(APPEND _FFTW_REQUIRED_VARS FFTW_DOUBLE_LIBRARY)
    list(APPEND FFTW_LIBRARIES ${FFTW_DOUBLE_LIBRARY})
  endif()
  
  if(component STREQUAL "LONGDOUBLE")
    find_library(FFTW_LONGDOUBLE_LIBRARY
      NAMES fftw3l libfftw3l
      PATHS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/local/lib
        ${CMAKE_PREFIX_PATH}/lib
        ENV FFTW_DIR
      PATH_SUFFIXES lib lib64
    )
    list(APPEND _FFTW_REQUIRED_VARS FFTW_LONGDOUBLE_LIBRARY)
    list(APPEND FFTW_LIBRARIES ${FFTW_LONGDOUBLE_LIBRARY})
  endif()
endforeach()

# Set standard variable names
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW
  REQUIRED_VARS ${_FFTW_REQUIRED_VARS}
  HANDLE_COMPONENTS
)

if(FFTW_FOUND)
  set(FFTW_INCLUDE_DIRS ${FFTW_INCLUDE_DIR})
  
  if(NOT TARGET FFTW::FFTW)
    add_library(FFTW::FFTW INTERFACE IMPORTED)
    set_target_properties(FFTW::FFTW PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${FFTW_LIBRARIES}"
    )
  endif()
  
  # Create component-specific targets
  if(FFTW_FLOAT_LIBRARY AND NOT TARGET FFTW::Float)
    add_library(FFTW::Float INTERFACE IMPORTED)
    set_target_properties(FFTW::Float PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${FFTW_FLOAT_LIBRARY}"
    )
  endif()
  
  if(FFTW_DOUBLE_LIBRARY AND NOT TARGET FFTW::Double)
    add_library(FFTW::Double INTERFACE IMPORTED)
    set_target_properties(FFTW::Double PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${FFTW_DOUBLE_LIBRARY}"
    )
  endif()
  
  if(FFTW_LONGDOUBLE_LIBRARY AND NOT TARGET FFTW::LongDouble)
    add_library(FFTW::LongDouble INTERFACE IMPORTED)
    set_target_properties(FFTW::LongDouble PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${FFTW_LONGDOUBLE_LIBRARY}"
    )
  endif()
endif()

mark_as_advanced(
  FFTW_INCLUDE_DIR
  FFTW_FLOAT_LIBRARY
  FFTW_DOUBLE_LIBRARY
  FFTW_LONGDOUBLE_LIBRARY
)