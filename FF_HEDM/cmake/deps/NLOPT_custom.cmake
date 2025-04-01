FetchContent_Declare(
  nlopt
  GIT_REPOSITORY https://github.com/stevengj/nlopt.git
  GIT_TAG master
)

FetchContent_GetProperties(nlopt)
if(NOT nlopt_POPULATED)
  FetchContent_Populate(nlopt)
  
  # Set up NLOPT build options with CMake
  option(NLOPT_CXX "enable C++ routines" OFF)
  option(NLOPT_PYTHON "enable Python bindings" OFF)
  option(NLOPT_OCTAVE "enable Octave bindings" OFF)
  option(NLOPT_MATLAB "enable Matlab bindings" OFF)
  option(NLOPT_GUILE "enable Guile bindings" OFF)
  option(NLOPT_SWIG "enable SWIG bindings" OFF)
  option(NLOPT_TESTS "enable tests" OFF)
  
  # Add nlopt subdirectory to build it using CMake
  add_subdirectory(${nlopt_SOURCE_DIR} ${nlopt_BINARY_DIR})
  
  # Create alias target to match expected naming convention
  if(NOT TARGET NLOPT::NLOPT)
    add_library(NLOPT::NLOPT ALIAS nlopt)
  endif()
endif()