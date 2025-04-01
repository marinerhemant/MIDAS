FetchContent_Declare(
  nlopt
  URL https://www.dropbox.com/scl/fi/ux4ccf23z7rotkgbqbrmk/nlopt-2.4.2.tar.gz?rlkey=afq6l6yyu9fnw1hpe62l4gwqq&dl=1
  URL_HASH MD5=d0b8f139a4acf29b76dbae69ade8ac54
)

FetchContent_GetProperties(nlopt)
if(NOT nlopt_POPULATED)
  FetchContent_Populate(nlopt)
  
  # Set up NLOPT build with autotools
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
    BUILD_COMMAND make -j8
    INSTALL_COMMAND make install
    BUILD_ALWAYS OFF
  )
  
  # Create interface library
  add_library(NLOPT::NLOPT INTERFACE IMPORTED GLOBAL)
  
  # Set up include directories
  set_target_properties(NLOPT::NLOPT PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/deps/include"
  )
  
  # Set up link libraries based on build type
  if(BUILD_SHARED_LIBS)
    if(APPLE)
      set_target_properties(NLOPT::NLOPT PROPERTIES
        INTERFACE_LINK_LIBRARIES "${CMAKE_BINARY_DIR}/deps/lib/libnlopt.dylib"
      )
    else()
      set_target_properties(NLOPT::NLOPT PROPERTIES
        INTERFACE_LINK_LIBRARIES "${CMAKE_BINARY_DIR}/deps/lib/libnlopt.so"
      )
    endif()
  else()
    set_target_properties(NLOPT::NLOPT PROPERTIES
      INTERFACE_LINK_LIBRARIES "${CMAKE_BINARY_DIR}/deps/lib/libnlopt.a"
    )
  endif()
  
  # Add dependency
  add_dependencies(NLOPT::NLOPT nlopt_external)
endif()