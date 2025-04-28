    # HDF5 configuration
    FetchContent_Declare(
    hdf5
    URL https://www.dropbox.com/scl/fi/ib4wkq1s9jhm0oi9n6r7c/hdf5-1.14.2.tar.gz?rlkey=eq20hs7juecpwcn1vuumssjuf&dl=1
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

    # Find system zlib or one installed by blosc
    find_package(ZLIB)
    if(ZLIB_FOUND)
        set(ZLIB_INCLUDE_DIR ${ZLIB_INCLUDE_DIRS})
        set(ZLIB_LIBRARY ${ZLIB_LIBRARIES})
    endif()

    include(GNUInstallDirs) # Ensure standard paths are defined
    set(CMAKE_INSTALL_INCLUDEDIR include CACHE PATH "Installation directory for header files" FORCE)
    set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries" FORCE)

    add_subdirectory(${hdf5_SOURCE_DIR} ${hdf5_BINARY_DIR})

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