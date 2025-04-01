FetchContent_Declare(
  nlopt
  URL https://www.dropbox.com/scl/fi/ux4ccf23z7rotkgbqbrmk/nlopt-2.4.2.tar.gz?rlkey=afq6l6yyu9fnw1hpe62l4gwqq&dl=1
  URL_HASH MD5=d0b8f139a4acf29b76dbae69ade8ac54
)

FetchContent_GetProperties(nlopt)
if(NOT nlopt_POPULATED)
  FetchContent_Populate(nlopt)
  
  # Check if CMakeLists.txt exists
  if(EXISTS "${nlopt_SOURCE_DIR}/CMakeLists.txt")
    # CMake build is available
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
  else()
    # Use Autotools build
    set(NLOPT_SHARED_ARG)
    if(BUILD_SHARED_LIBS)
      set(NLOPT_SHARED_ARG --enable-shared --disable-static)
    else()
      set(NLOPT_SHARED_ARG --disable-shared --enable-static)
    endif()
    
    ExternalProject_Add(nlopt_external
      SOURCE_DIR ${nlopt_SOURCE_DIR}
      BINARY_DIR ${nlopt_BINARY_DIR}
      CONFIGURE_COMMAND ${nlopt_SOURCE_DIR}/configure 
                        --prefix=${CMAKE_BINARY_DIR}/deps
                        ${NLOPT_SHARED_ARG}
                        --disable-python
                        --disable-octave
                        --disable-matlab
                        --disable-guile
                        --without-threadlocal
      BUILD_COMMAND make -j8
      INSTALL_COMMAND make install
      BUILD_ALWAYS OFF
    )
    
    # Create interface library
    add_library(NLOPT::NLOPT INTERFACE IMPORTED)
    set_target_properties(NLOPT::NLOPT PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/deps/include"
    )
    
    if(BUILD_SHARED_LIBS)
      set_target_properties(NLOPT::NLOPT PROPERTIES
        INTERFACE_LINK_LIBRARIES "${CMAKE_BINARY_DIR}/deps/lib/libnlopt${CMAKE_SHARED_LIBRARY_SUFFIX}"
      )
    else()
      set_target_properties(NLOPT::NLOPT PROPERTIES
        INTERFACE_LINK_LIBRARIES "${CMAKE_BINARY_DIR}/deps/lib/libnlopt${CMAKE_STATIC_LIBRARY_SUFFIX}"
      )
    endif()
    
    # Add dependency
    add_dependencies(NLOPT::NLOPT nlopt_external)
  endif()
endif()