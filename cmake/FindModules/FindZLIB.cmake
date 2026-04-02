# FindZLIB.cmake — project-level override
#
# When MIDAS builds zlib via FetchContent, the "zlib" target exists but the
# library file (libz.so) is not yet on disk at configure time.  Downstream
# dependencies (blosc2, libtiff, hdf5, libzip) each call find_package(ZLIB)
# internally.  CMake's stock FindZLIB.cmake re-runs find_library(), which
# fails because the file doesn't exist yet.  This causes ZLIB_FOUND=FALSE
# inside those deps, leading to:
#   - blosc2 falling back to internal zlib-ng (target name collision)
#   - libtiff/hdf5/libzip silently disabling zlib or using a broken path
#
# Fix: if a ZLIB::ZLIB target already exists (from FetchContent or system
# find_package), report success immediately without re-searching.

if(TARGET ZLIB::ZLIB)
  # Populate the variables that callers expect from find_package(ZLIB).
  if(NOT ZLIB_INCLUDE_DIR)
    get_target_property(_zlib_inc ZLIB::ZLIB INTERFACE_INCLUDE_DIRECTORIES)
    if(_zlib_inc)
      list(GET _zlib_inc 0 ZLIB_INCLUDE_DIR)
    endif()
    unset(_zlib_inc)
  endif()
  set(ZLIB_INCLUDE_DIRS "${ZLIB_INCLUDE_DIR}")

  # ZLIB_LIBRARY / ZLIB_LIBRARIES: use the target name so that
  # target_link_libraries() resolves it as a CMake target dependency.
  if(NOT ZLIB_LIBRARY)
    set(ZLIB_LIBRARY ZLIB::ZLIB)
  endif()
  set(ZLIB_LIBRARIES ZLIB::ZLIB)

  set(ZLIB_FOUND TRUE)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(ZLIB
    REQUIRED_VARS ZLIB_LIBRARY ZLIB_INCLUDE_DIR
    VERSION_VAR ZLIB_VERSION_STRING)
  return()
endif()

# No pre-existing target — fall through to the standard CMake module.
include(${CMAKE_ROOT}/Modules/FindZLIB.cmake)
