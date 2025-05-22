# FindHDF5.cmake - Find HDF5 library
#
# This module defines:
#  HDF5_FOUND        - True if HDF5 is found
#  HDF5_INCLUDE_DIRS - HDF5 include directories
#  HDF5_LIBRARIES    - HDF5 libraries
#  HDF5::HDF5        - HDF5 imported target
#  HDF5::HL          - HDF5 HL imported target

# Try to find HDF5 in standard paths
find_path(HDF5_INCLUDE_DIR
  NAMES hdf5.h
  PATHS
    /usr/include
    /usr/local/include
    /opt/local/include
    ${CMAKE_PREFIX_PATH}/include
    ENV HDF5_DIR
  PATH_SUFFIXES hdf5 hdf5/include
)

# Find both shared and static libraries
find_library(HDF5_LIBRARY
  NAMES hdf5 libhdf5
  PATHS
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /opt/local/lib
    ${CMAKE_PREFIX_PATH}/lib
    ENV HDF5_DIR
  PATH_SUFFIXES lib
)

find_library(HDF5_HL_LIBRARY
  NAMES hdf5_hl libhdf5_hl
  PATHS
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /opt/local/lib
    ${CMAKE_PREFIX_PATH}/lib
    ENV HDF5_DIR
  PATH_SUFFIXES lib
)

# Set standard variable names
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HDF5
  REQUIRED_VARS HDF5_LIBRARY HDF5_HL_LIBRARY HDF5_INCLUDE_DIR
)

if(HDF5_FOUND)
  set(HDF5_LIBRARIES ${HDF5_LIBRARY} ${HDF5_HL_LIBRARY})
  set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
  
  if(NOT TARGET HDF5::HDF5)
    add_library(HDF5::HDF5 INTERFACE IMPORTED)
    set_target_properties(HDF5::HDF5 PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${HDF5_LIBRARY}"
    )
  endif()
  
  if(NOT TARGET HDF5::HL)
    add_library(HDF5::HL INTERFACE IMPORTED)
    set_target_properties(HDF5::HL PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${HDF5_HL_LIBRARY}"
    )
  endif()
endif()

mark_as_advanced(HDF5_INCLUDE_DIR HDF5_LIBRARY HDF5_HL_LIBRARY)