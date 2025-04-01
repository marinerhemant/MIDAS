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
  
  set(FFTW_OPTIONS --enable-float --disable-fortran --enable-sse --enable-sse2 --enable-avx --enable-avx2 --enable-avx-128-fma --enable-generic-simd128 --enable-generic-simd256)
  
  ExternalProject_Add(fftw_external
    SOURCE_DIR ${fftw_SOURCE_DIR}
    BINARY_DIR ${fftw_BINARY_DIR}
    CONFIGURE_COMMAND ${fftw_SOURCE_DIR}/configure 
                      --prefix=${CMAKE_BINARY_DIR}/deps
                      ${FFTW_SHARED_ARG}
                      ${FFTW_OPTIONS}
    BUILD_COMMAND make -j8
    INSTALL_COMMAND make install
    BUILD_ALWAYS OFF
  )
  
  # Create interface library
  add_library(FFTW::FFTW INTERFACE IMPORTED)
  set_target_properties(FFTW::FFTW PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/deps/include"
  )
  
  if(BUILD_SHARED_LIBS)
    set_target_properties(FFTW::FFTW PROPERTIES
      INTERFACE_LINK_LIBRARIES "${CMAKE_BINARY_DIR}/deps/lib/libfftw3f${CMAKE_SHARED_LIBRARY_SUFFIX}"
    )
  else()
    set_target_properties(FFTW::FFTW PROPERTIES
      INTERFACE_LINK_LIBRARIES "${CMAKE_BINARY_DIR}/deps/lib/libfftw3f${CMAKE_STATIC_LIBRARY_SUFFIX}"
    )
  endif()
  
  # Add dependency
  add_dependencies(FFTW::FFTW fftw_external)
endif()
