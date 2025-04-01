FetchContent_Declare(
  nlopt
  URL https://www.dropbox.com/scl/fi/ux4ccf23z7rotkgbqbrmk/nlopt-2.4.2.tar.gz?rlkey=afq6l6yyu9fnw1hpe62l4gwqq&dl=1
  URL_HASH MD5=d0b8f139a4acf29b76dbae69ade8ac54
)

FetchContent_GetProperties(nlopt)
if(NOT nlopt_POPULATED)
  FetchContent_Populate(nlopt)
  
  # Set up NLOPT build options
  set(NLOPT_BUILD_SHARED ${BUILD_SHARED_LIBS} CACHE BOOL "Build NLopt as a shared library" FORCE)
  set(NLOPT_BUILD_STATIC NOT ${BUILD_SHARED_LIBS} CACHE BOOL "Build NLopt as a static library" FORCE)
  set(NLOPT_PYTHON OFF CACHE BOOL "Build python bindings" FORCE)
  set(NLOPT_OCTAVE OFF CACHE BOOL "Build octave bindings" FORCE)
  set(NLOPT_MATLAB OFF CACHE BOOL "Build matlab bindings" FORCE)
  set(NLOPT_GUILE OFF CACHE BOOL "Build guile bindings" FORCE)
  set(NLOPT_SWIG OFF CACHE BOOL "Use SWIG to build bindings" FORCE)
  set(NLOPT_TESTS OFF CACHE BOOL "Build tests" FORCE)
  
  add_subdirectory(${nlopt_SOURCE_DIR} ${nlopt_BINARY_DIR})

  # Export as a target
  add_library(NLOPT::NLOPT INTERFACE IMPORTED)
  if(BUILD_SHARED_LIBS)
    set_target_properties(NLOPT::NLOPT PROPERTIES
      INTERFACE_LINK_LIBRARIES nlopt
    )
  else()
    set_target_properties(NLOPT::NLOPT PROPERTIES
      INTERFACE_LINK_LIBRARIES nlopt-static
    )
  endif()
  set_target_properties(NLOPT::NLOPT PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${nlopt_SOURCE_DIR}/src/api"
  )
endif()
