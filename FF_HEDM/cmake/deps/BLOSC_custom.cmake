FetchContent_Declare(
  blosc
  GIT_REPOSITORY https://github.com/Blosc/c-blosc2.git
  GIT_TAG main # Consider using a specific tag/commit for reproducibility
)

FetchContent_GetProperties(blosc)
if(NOT blosc_POPULATED)
  FetchContent_Populate(blosc)

  # --- Configuration Options Passed Down to Blosc ---
  # Use standard CMake variables where possible and PARENT_SCOPE
  # These will be picked up by Blosc's CMakeLists.txt via add_subdirectory
  set(BUILD_TESTS OFF CACHE BOOL "Build Blosc tests" FORCE PARENT_SCOPE)
  set(BUILD_BENCHMARKS OFF CACHE BOOL "Build Blosc benchmarks" FORCE PARENT_SCOPE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "Build Blosc examples" FORCE PARENT_SCOPE)
  # Pass the global BUILD_SHARED_LIBS setting down
  set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared libraries" FORCE PARENT_SCOPE)

  # Control internal codec support (keep user's choice for Zlib)
  # Blosc's CMake uses DEACTIVATE_<CODEC>=ON to disable. Default is usually enabled (OFF).
  set(DEACTIVATE_ZLIB OFF CACHE BOOL "Deactivate Blosc's internal Zlib support" FORCE PARENT_SCOPE)
  # Example: set(DEACTIVATE_LZ4 OFF CACHE BOOL "Deactivate Blosc's internal LZ4 support" FORCE PARENT_SCOPE)
  # Example: set(DEACTIVATE_ZSTD OFF CACHE BOOL "Deactivate Blosc's internal ZSTD support" FORCE PARENT_SCOPE)

  # --- Build Blosc ---
  # Blosc's CMakeLists.txt will be processed here
  add_subdirectory(${blosc_SOURCE_DIR} ${blosc_BINARY_DIR})

  # Create our internal interface target linking to the actual Blosc target
  if(TARGET ${BLOSC_ACTUAL_TARGET_NAME} AND NOT TARGET BLOSC::BLOSC)
    add_library(BLOSC::BLOSC INTERFACE IMPORTED)
    set_target_properties(BLOSC::BLOSC PROPERTIES
      INTERFACE_LINK_LIBRARIES ${BLOSC_ACTUAL_TARGET_NAME} # Link to the 'blosc2' target
      INTERFACE_INCLUDE_DIRECTORIES "${blosc_SOURCE_DIR}/include" # Public headers are here
    )
    # If Blosc links system libraries (like ZLIB if external, pthreads, etc.),
    # they *should* be propagated correctly by the ${BLOSC_ACTUAL_TARGET_NAME} target.
  elseif(NOT TARGET BLOSC::BLOSC)
    message(WARNING "Could not find Blosc target '${BLOSC_ACTUAL_TARGET_NAME}' after add_subdirectory. Internal BLOSC::BLOSC target not fully configured.")
  endif()

  # --- INSTALLATION RULES ---
  # Install header files from the source directory
  install(DIRECTORY ${blosc_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
          FILES_MATCHING PATTERN "*.h"
          COMPONENT Development # Optional component specification
         )

  # Install the actual library target ('blosc2')
  if(TARGET ${BLOSC_ACTUAL_TARGET_NAME})
    install(TARGETS ${BLOSC_ACTUAL_TARGET_NAME}
            EXPORT BloscTargets # Associate with the export set named BloscTargets
            # Define installation destinations
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Libraries
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Libraries
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Runtime
           )

    # Install the CMake export file (BloscTargets.cmake)
    # Allows find_package(Blosc) to work for external projects
    install(EXPORT BloscTargets # Must match the EXPORT name above
            FILE BloscTargets.cmake # The generated CMake file name
            NAMESPACE Blosc::        # Namespace for imported targets (e.g., Blosc::blosc2)
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Blosc # Standard install location
            COMPONENT Development
           )
  else()
    message(WARNING "Blosc target '${BLOSC_ACTUAL_TARGET_NAME}' not found. Skipping installation rules for Blosc library.")
  endif()

endif()