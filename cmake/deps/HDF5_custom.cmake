    # HDF5 configuration
    FetchContent_Declare(
    hdf5
    URL "https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_1.14.6.tar.gz"
    )

    FetchContent_GetProperties(hdf5)
    if(NOT hdf5_POPULATED)
    FetchContent_Populate(hdf5)

    # Set HDF5 options
    set(HDF5_ENABLE_Z_LIB_SUPPORT ON CACHE BOOL "Enable ZLIB support" FORCE)
    set(HDF5_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared libraries" FORCE)
    set(HDF5_BUILD_STATIC_LIBS NOT ${BUILD_SHARED_LIBS} CACHE BOOL "Build static libraries" FORCE)
    set(HDF5_BUILD_TOOLS OFF CACHE BOOL "Build HDF5 tools" FORCE)
    set(HDF5_BUILD_EXAMPLES OFF CACHE BOOL "Build HDF5 examples" FORCE)
    set(HDF5_BUILD_UTILS OFF CACHE BOOL "Build HDF5 utilities" FORCE)
    set(HDF5_BUILD_HL_LIB ON CACHE BOOL "Build HDF5 high level APIs" FORCE)
    set(HDF5_BUILD_FORTRAN OFF CACHE BOOL "Build Fortran support" FORCE)
    set(HDF5_ENABLE_THREADSAFE OFF CACHE BOOL "Enable thread safety" FORCE)
    set(HDF5_ENABLE_PARALLEL OFF CACHE BOOL "Enable parallel HDF5" FORCE)
    set(HDF5_ENABLE_NONSTANDARD_FEATURE_FLOAT16 OFF CACHE BOOL "Disable _Float16 support" FORCE)

    # Find system zlib or one installed by blosc
    find_package(ZLIB)
    if(ZLIB_FOUND)
        set(ZLIB_INCLUDE_DIR ${ZLIB_INCLUDE_DIRS})
        set(ZLIB_LIBRARY ${ZLIB_LIBRARIES})
    endif()

    # Skip HDF5's install/export rules — we only need it as a build dep.
    # Without this, install(EXPORT "hdf5-targets") fails because it
    # references the fetched "zlib" target which isn't in any export set.
    set(HDF5_EXTERNALLY_CONFIGURED ON CACHE BOOL "" FORCE)
    set(HDF5_EXPORTED_TARGETS "" CACHE STRING "" FORCE)

    # Workaround: CMake prohibits set_target_properties on ALIAS targets.
    # When MIDAS builds zlib via FetchContent, ZLIB::ZLIB is created as an
    # ALIAS (see top-level CMakeLists.txt).  HDF5's CMakeFilters.cmake tries
    # set_target_properties(ZLIB::ZLIB PROPERTIES OUTPUT_NAME ...) which
    # fails on the ALIAS.  Pre-setting H5_ZLIB_HEADER tells HDF5 that zlib
    # is already externally configured, skipping its internal find+property
    # code path.  We also pre-populate LINK_COMP_LIBS so HDF5 still links
    # against zlib for compression support.
    if(TARGET ZLIB::ZLIB)
      get_target_property(_zlib_aliased ZLIB::ZLIB ALIASED_TARGET)
      if(_zlib_aliased)
        set(H5_ZLIB_HEADER "zlib.h")
        set(H5_ZLIB_INCLUDE_DIR_GEN "${ZLIB_INCLUDE_DIR}")
        set(H5_ZLIB_INCLUDE_DIRS "${ZLIB_INCLUDE_DIRS}")
        set(LINK_COMP_LIBS "ZLIB::ZLIB")
      endif()
    endif()

    add_subdirectory(${hdf5_SOURCE_DIR} ${hdf5_BINARY_DIR} EXCLUDE_FROM_ALL)

    # Ensure HDF5 targets wait for the FetchContent zlib target to be built.
    if(TARGET zlib)
      if(TARGET hdf5-shared)
        add_dependencies(hdf5-shared zlib)
      endif()
      if(TARGET hdf5-static)
        add_dependencies(hdf5-static zlib)
      endif()
    endif()

    # Export HDF5 as targets
    add_library(HDF5::HDF5 INTERFACE IMPORTED)
    add_library(HDF5::HL INTERFACE IMPORTED)

    if(BUILD_SHARED_LIBS)
        set_target_properties(HDF5::HDF5 PROPERTIES
        INTERFACE_LINK_LIBRARIES hdf5-shared
        )
        set_target_properties(HDF5::HL PROPERTIES
        INTERFACE_LINK_LIBRARIES hdf5_hl-shared
        )
    else()
        set_target_properties(HDF5::HDF5 PROPERTIES
        INTERFACE_LINK_LIBRARIES hdf5-static
        )
        set_target_properties(HDF5::HL PROPERTIES
        INTERFACE_LINK_LIBRARIES hdf5_hl-static
        )
    endif()

    set_target_properties(HDF5::HDF5 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${hdf5_SOURCE_DIR}/src;${hdf5_BINARY_DIR}"
    )
    set_target_properties(HDF5::HL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${hdf5_SOURCE_DIR}/hl/src;${hdf5_BINARY_DIR}/hl"
    )
    endif()