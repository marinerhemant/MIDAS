FetchContent_Declare(
  fftw
  URL https://www.dropbox.com/scl/fi/yugsuwobadxt5gvfsdz46/fftw-3.3.10.tar.gz?rlkey=cfo1rwazrr4gbm2k043np8skj&dl=1
)

FetchContent_GetProperties(fftw)
if(NOT fftw_POPULATED)
  FetchContent_Populate(fftw)
  
  # FFTW is a bit tricky, we'll use ExternalProject
  # because FFTW doesn't have a CMakeLists.txt
  
  set(FFTW_SHARED_ARG)
  if(BUILD_SHARED_LIBS)
    set(FFTW_SHARED_ARG --enable-shared --disable-static)
  else()
    set(FFTW_SHARED_ARG --disable-shared --enable-static)
  endif()
  
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
      set(FFTW_OPTIONS --enable-float --disable-fortran)
  else()
      set(FFTW_OPTIONS --enable-float --disable-fortran --enable-sse --enable-sse2 --enable-avx --enable-avx2 --enable-avx-128-fma --enable-generic-simd128 --enable-generic-simd256)
  endif()
  
  # Define the installation directory
  set(FFTW_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/fftw_install)
  
  ExternalProject_Add(fftw_external
    SOURCE_DIR ${fftw_SOURCE_DIR}
    BINARY_DIR ${fftw_BINARY_DIR}
    CONFIGURE_COMMAND ${fftw_SOURCE_DIR}/configure 
                      --prefix=${FFTW_INSTALL_DIR}
                      ${FFTW_SHARED_ARG}
                      ${FFTW_OPTIONS}
    BUILD_COMMAND make -j8
    INSTALL_COMMAND make install
    BUILD_ALWAYS OFF
  )
  
  # Create include directory to avoid errors during configuration
  file(MAKE_DIRECTORY ${FFTW_INSTALL_DIR}/include)
  
  # Create interface library
  add_library(FFTW::FFTW INTERFACE IMPORTED GLOBAL)
  set_target_properties(FFTW::FFTW PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INSTALL_DIR}/include"
  )
  
  # Set up library paths based on build type
  if(BUILD_SHARED_LIBS)
    if(APPLE)
      set(FFTW_LIB_PATH "${FFTW_INSTALL_DIR}/lib/libfftw3f.dylib")
    else()
      set(FFTW_LIB_PATH "${FFTW_INSTALL_DIR}/lib/libfftw3f.so")
    endif()
  else()
    set(FFTW_LIB_PATH "${FFTW_INSTALL_DIR}/lib/libfftw3f.a")
  endif()
  
  set_target_properties(FFTW::FFTW PROPERTIES
    INTERFACE_LINK_LIBRARIES "${FFTW_LIB_PATH}"
  )
  
  # Add dependency
  add_dependencies(FFTW::FFTW fftw_external)
  install(DIRECTORY ${CMAKE_BINARY_DIR}/deps/include/ DESTINATION include)
  install(DIRECTORY ${CMAKE_BINARY_DIR}/deps/lib/ DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()