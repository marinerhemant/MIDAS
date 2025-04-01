FetchContent_Declare(
  nlopt
  GIT_REPOSITORY https://github.com/stevengj/nlopt.git
  GIT_TAG master
)

FetchContent_GetProperties(nlopt)
if(NOT nlopt_POPULATED)
  FetchContent_Populate(nlopt)
  
  # Set up NLOPT build with CMake
  set(NLOPT_CXX OFF CACHE BOOL "disable C++ routines")
  set(NLOPT_PYTHON OFF CACHE BOOL "disable Python bindings")
  set(NLOPT_OCTAVE OFF CACHE BOOL "disable Octave bindings")
  set(NLOPT_MATLAB OFF CACHE BOOL "disable Matlab bindings")
  set(NLOPT_GUILE OFF CACHE BOOL "disable Guile bindings")
  set(NLOPT_SWIG OFF CACHE BOOL "disable SWIG bindings")
  set(NLOPT_TESTS OFF CACHE BOOL "disable tests")
  set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  
  # Use CMake for building instead of autotools
  add_subdirectory(${nlopt_SOURCE_DIR} ${nlopt_BINARY_DIR})
  
  # Create interface library
  add_library(NLOPT::NLOPT INTERFACE IMPORTED GLOBAL)
  
  # Set up include directories
  set_target_properties(NLOPT::NLOPT PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${nlopt_SOURCE_DIR}/src/api;${nlopt_BINARY_DIR}/src/api"
  )
  
  # NLOpt now provides its own target we can link to
  set_target_properties(NLOPT::NLOPT PROPERTIES
    INTERFACE_LINK_LIBRARIES "nlopt"
  )
  
  # Add dependency
  add_dependencies(NLOPT::NLOPT nlopt)
endif()