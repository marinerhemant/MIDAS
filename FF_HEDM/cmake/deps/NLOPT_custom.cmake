FetchContent_Declare(
  nlopt
  GIT_REPOSITORY https://github.com/stevengj/nlopt.git
  GIT_TAG master # Consider using a specific stable tag/commit instead of master
)

FetchContent_GetProperties(nlopt)
if(NOT nlopt_POPULATED)
  FetchContent_Populate(nlopt)

  # --- Configuration Options Passed Down to NLOPT ---
  # Set options in the PARENT_SCOPE so add_subdirectory picks them up
  # Using CACHE FORCE ensures these settings override any defaults in nlopt's CMakeLists
  set(NLOPT_CXX OFF CACHE BOOL "enable C++ routines" FORCE PARENT_SCOPE)
  set(NLOPT_PYTHON OFF CACHE BOOL "enable Python bindings" FORCE PARENT_SCOPE)
  set(NLOPT_OCTAVE OFF CACHE BOOL "enable Octave bindings" FORCE PARENT_SCOPE)
  set(NLOPT_MATLAB OFF CACHE BOOL "enable Matlab bindings" FORCE PARENT_SCOPE)
  set(NLOPT_GUILE OFF CACHE BOOL "enable Guile bindings" FORCE PARENT_SCOPE)
  set(NLOPT_SWIG OFF CACHE BOOL "enable SWIG bindings" FORCE PARENT_SCOPE)
  set(NLOPT_TESTS OFF CACHE BOOL "enable tests" FORCE PARENT_SCOPE)
  # Ensure NLOPT builds the library type matching the main project
  set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "Build shared libraries" FORCE PARENT_SCOPE)

  # --- Build NLOPT ---
  # Add nlopt subdirectory to build it using its own CMakeLists.txt
  add_subdirectory(${nlopt_SOURCE_DIR} ${nlopt_BINARY_DIR})

  # --- Target Alias for Internal Use ---
  # Create an alias target for consistent namespacing within this build.
  # The actual target name created by nlopt's CMakeLists is typically 'nlopt'.
  if(TARGET nlopt AND NOT TARGET NLOPT::NLOPT)
    add_library(NLOPT::NLOPT ALIAS nlopt)
  endif()

  # --- INSTALLATION RULES ---
  # Install header files
  # nlopt.h is the main C header, nlopt.hpp for C++
  # nlopt_config.h is generated in the binary directory
  install(FILES
            ${nlopt_SOURCE_DIR}/nlopt.h      # Public C API Header
            # ${nlopt_SOURCE_DIR}/nlopt.hpp # Uncomment if NLOPT_CXX is ON
            ${nlopt_BINARY_DIR}/nlopt_config.h # Generated configuration header
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # Standard include directory (e.g., install/include)
          COMPONENT Development # Optional: Mark as development component
         )

  # Install the library target ('nlopt') created by add_subdirectory
  install(TARGETS nlopt # Use the actual target name created by NLOPT's CMake
          EXPORT NLOPTTargets # Associate with the export set 'NLOPTTargets'
          # Define installation destinations for different artifact types
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Libraries # For static libs (.a)
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Libraries # For shared libs (.so, .dylib)
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Runtime   # For shared libs on Windows (.dll)
         )

  # Install the CMake export file (NLOPTTargets.cmake)
  # This file allows external CMake projects to find and use the installed NLOPT library
  # using find_package(NLOPT)
  install(EXPORT NLOPTTargets # The name matching the EXPORT in install(TARGETS)
          FILE NLOPTTargets.cmake # The filename to generate
          NAMESPACE NLOPT::        # The namespace for imported targets (e.g., NLOPT::nlopt)
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NLOPT # Standard location for package files
          COMPONENT Development # Optional: Mark as development component
         )

endif()