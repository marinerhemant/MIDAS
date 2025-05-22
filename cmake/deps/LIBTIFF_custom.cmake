    FetchContent_Declare(
    libtiff
    URL https://www.dropbox.com/scl/fi/tk3axrjtjgxmjj9hzsk13/tiff-4.6.0.tar.gz?rlkey=judqzxze5g4sg0bviyul8kqvp&dl=1
    )

    FetchContent_GetProperties(libtiff)
    if(NOT libtiff_POPULATED)
    FetchContent_Populate(libtiff)

    # Configure TIFF build options
    set(tiff_BUILD_SHARED ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared libraries" FORCE)
    set(tiff_BUILD_STATIC NOT ${BUILD_SHARED_LIBS} CACHE BOOL "Build static libraries" FORCE)

    # Disable unnecessary components
    set(tiff_BUILD_TOOLS OFF CACHE BOOL "Build TIFF tools" FORCE)
    set(tiff_BUILD_DOCS OFF CACHE BOOL "Build TIFF documentation" FORCE)
    set(tiff_BUILD_CONTRIB OFF CACHE BOOL "Build TIFF contrib" FORCE)
    set(tiff_BUILD_TESTS OFF CACHE BOOL "Build TIFF tests" FORCE)
    set(tiff_BUILD_ZLIB ON CACHE BOOL "Enable Zlib support" FORCE)

    # Find system zlib or one installed by blosc
    find_package(ZLIB)
    if(ZLIB_FOUND)
        set(ZLIB_INCLUDE_DIR ${ZLIB_INCLUDE_DIRS})
        set(ZLIB_LIBRARY ${ZLIB_LIBRARIES})
    endif()

    add_subdirectory(${libtiff_SOURCE_DIR} ${libtiff_BINARY_DIR})

    # Export TIFF library as a target
    add_library(TIFF::TIFF INTERFACE IMPORTED)
    if(BUILD_SHARED_LIBS)
        set_target_properties(TIFF::TIFF PROPERTIES
        INTERFACE_LINK_LIBRARIES tiff
        )
    else()
        set_target_properties(TIFF::TIFF PROPERTIES
        INTERFACE_LINK_LIBRARIES tiff-static
        )
    endif()
    set_target_properties(TIFF::TIFF PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${libtiff_SOURCE_DIR}/libtiff"
    )
    endif()