# In cmake/deps/LIBTIFF_custom.cmake

if(NOT DEFINED BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON)
endif()

if(BUILD_SHARED_LIBS)
  set(LIBTIFF_BUILD_STATIC_LIBS OFF)
else()
  set(LIBTIFF_BUILD_STATIC_LIBS ON)
endif()

# Arguments to pass to libtiff's CMake configuration
set(LIBTIFF_CMAKE_ARGS
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
    -DBUILD_STATIC_LIBS=${LIBTIFF_BUILD_STATIC_LIBS}
    -DTIFF_TOOLS=OFF
    -DTIFF_CONTRIB=OFF
    -DTIFF_TESTS=OFF
    -Dzlib=ON
    # These flags might now be redundant if the patch works, but don't hurt
    -DTIFF_HTML=OFF
    -DTIFF_MAN=OFF
    -DTIFF_DOCS=OFF # General flag
    # We can try removing CMAKE_POLICY_VERSION_MINIMUM if the patch prevents doc/CMakeLists.txt from being processed
    # -DCMAKE_POLICY_VERSION_MINIMUM=3.5
)

if(TARGET ZLIB::ZLIB)
    get_target_property(ZLIB_INCLUDE_DIR_PROP ZLIB::ZLIB INTERFACE_INCLUDE_DIRECTORIES)
    if(ZLIB_INCLUDE_DIR_PROP)
      list(GET ZLIB_INCLUDE_DIR_PROP 0 _zlib_inc_dir_first)
      if(IS_DIRECTORY "${_zlib_inc_dir_first}")
        list(APPEND LIBTIFF_CMAKE_ARGS -DZLIB_INCLUDE_DIR=${_zlib_inc_dir_first})
      endif()
    endif()
endif()

FetchContent_Declare(
  libtiff
  URL https://download.osgeo.org/libtiff/tiff-4.7.0.tar.gz
  URL_HASH SHA256=67160e3457365ab96c5b3286a0903aa6e78bdc44c4bc737d2e486bcecb6ba976
  CMAKE_ARGS ${LIBTIFF_CMAKE_ARGS}
  PATCH_COMMAND patch -p1 -N --fuzz=0 < ${CMAKE_CURRENT_LIST_DIR}/libtiff-disable-doc.patch
  # -N: ignore already applied patches
  # --fuzz=0: apply only if exact match
)

FetchContent_MakeAvailable(libtiff)

# --- Target Aliasing Logic (remains the same) ---
if(NOT TARGET TIFF::TIFF)
    add_library(TIFF::TIFF INTERFACE IMPORTED GLOBAL)
    message(STATUS "LIBTIFF_custom.cmake: Creating TIFF::TIFF as INTERFACE IMPORTED.")
else()
    message(STATUS "LIBTIFF_custom.cmake: TIFF::TIFF target already exists.")
endif()

set(_libtiff_concrete_target "")
if(TARGET tiff)
    set(_libtiff_concrete_target tiff)
else()
    message(WARNING "LIBTIFF_custom.cmake: Primary libtiff target 'tiff' not found.")
endif()

if(_libtiff_concrete_target)
    set_property(TARGET TIFF::TIFF APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${_libtiff_concrete_target})
    message(STATUS "LIBTIFF_custom.cmake: Linking TIFF::TIFF to concrete target '${_libtiff_concrete_target}'.")
    get_target_property(_concrete_tiff_includes ${_libtiff_concrete_target} INTERFACE_INCLUDE_DIRECTORIES)
    if(_concrete_tiff_includes)
        set_property(TARGET TIFF::TIFF APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${_concrete_tiff_includes})
        message(STATUS "LIBTIFF_custom.cmake: Added include directories from '${_libtiff_concrete_target}' to TIFF::TIFF.")
    elseif(EXISTS "${libtiff_SOURCE_DIR}/libtiff")
        set_property(TARGET TIFF::TIFF APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${libtiff_SOURCE_DIR}/libtiff")
        message(STATUS "LIBTIFF_custom.cmake: Adding fallback include directory '${libtiff_SOURCE_DIR}/libtiff' to TIFF::TIFF.")
    else()
        message(WARNING "LIBTIFF_custom.cmake: Could not determine include directories for libtiff target '${_libtiff_concrete_target}'.")
    endif()
else()
    message(WARNING "LIBTIFF_custom.cmake: Could not determine appropriate concrete libtiff target to link against TIFF::TIFF. Linking for TIFF::TIFF might be incomplete.")
    if(EXISTS "${libtiff_SOURCE_DIR}/libtiff")
        set_property(TARGET TIFF::TIFF APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${libtiff_SOURCE_DIR}/libtiff")
        message(STATUS "LIBTIFF_custom.cmake: Adding fallback include directory '${libtiff_SOURCE_DIR}/libtiff' to TIFF::TIFF (no concrete target found).")
    endif()
endif()
message(STATUS "LIBTIFF_custom.cmake: Configured libtiff. Shared: ${BUILD_SHARED_LIBS}")