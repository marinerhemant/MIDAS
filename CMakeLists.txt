cmake_minimum_required(VERSION 3.16)

# Define options first, before the project command
option(USE_CUDA "Build with CUDA support" OFF) # Default to ON, can be configured
option(BUILD_OMP "Build with OpenMP support" ON) # Default to ON
option(DOWNLOAD_DEPENDENCIES "Download and build dependencies if not found by system" ON)
option(USE_SYSTEM_DEPS "Prefer system-installed dependencies" OFF) # If ON, try find_package first

# Project definition
option(INSTALL_PYTHON_SCRIPTS "Install Python scripts from gui/ and utils/ subdirectories" ON)
option(INSTALL_PYTHON_DEPENDENCIES "Install Python dependencies using pip (if requirements.txt found)" ON)
set(PYTHON_EXECUTABLE "" CACHE FILEPATH "Path to Python executable for installing dependencies")

if(USE_CUDA)
  project(MIDAS LANGUAGES C CUDA)
  # Add CUDA specific configurations if necessary, e.g., CMAKE_CUDA_ARCHITECTURES
  # set(CMAKE_CUDA_ARCHITECTURES 75) # Example, adjust as needed
else()
  project(MIDAS LANGUAGES C)
endif()

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set default installation path if not specified by user
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation directory" FORCE)
endif()

# Output directories for libraries and binaries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# RPATH settings for better portability
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) # Important for packaging
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Set RPATH for installed binaries
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@executable_path/../lib")
else()
  # Use $ORIGIN for Linux, adjust if needed for other Unix-like systems
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

# Include directories for GNUInstallDirs
include(GNUInstallDirs)

# Define installation directories using GNUInstallDirs
set(MIDAS_INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR})
set(MIDAS_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR})
set(MIDAS_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})

# --- Dependency Management (Initial Placeholder - will be expanded) ---
# This section will be populated with logic from FF_HEDM/CMakeLists.txt,
# adapted to be globally available.
# For now, we'll just set up the module path for custom find modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindModules") # Assuming a similar structure
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindModules") # Create if doesn't exist
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps") # Create if doesn't exist

# Option for shared/static libs for dependencies
option(BUILD_SHARED_LIBS "Build dependencies as shared libraries" ON)

# Set default dependencies directory
set(DEPS_INSTALL_DIR "${CMAKE_BINARY_DIR}/deps" CACHE PATH "Dependencies installation directory")
list(APPEND CMAKE_PREFIX_PATH ${DEPS_INSTALL_DIR}) # Add to prefix path for find_package
include(FetchContent) # Ensure FetchContent is available
include(ExternalProject) # Ensure ExternalProject is available

# Function to handle dependencies (copied and adapted from FF_HEDM)
function(handle_dependency NAME URL DOWNLOAD_FLAG)
    # Check if target already exists (e.g. TIFF::TIFF or NAME::NAME)
    if(NAME STREQUAL "LIBTIFF")
      if(TARGET TIFF::TIFF)
        message(STATUS "${NAME} target already exists as TIFF::TIFF, skipping")
        return()
      endif()
    elseif(TARGET ${NAME}::${NAME})
      message(STATUS "${NAME} target '${NAME}::${NAME}' already exists, skipping")
      return()
    endif()

    # Check for custom find module
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindModules/Find${NAME}.cmake")
      # CMAKE_MODULE_PATH is already appended globally, no need to list(APPEND ...) here
      message(STATUS "Custom Find${NAME}.cmake found in CMAKE_MODULE_PATH.")
    endif()
    
    # Check for custom dependency file
    set(CUSTOM_DEP_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/${NAME}_custom.cmake")
    if(EXISTS "${CUSTOM_DEP_FILE}")
      message(STATUS "Using custom dependency file for ${NAME} from ${CUSTOM_DEP_FILE}")
      include("${CUSTOM_DEP_FILE}") # This custom file should define the target NAME::NAME or TIFF::TIFF
      # Verification after include
      if(NAME STREQUAL "LIBTIFF" AND NOT TARGET TIFF::TIFF)
          message(FATAL_ERROR "Custom dependency file ${CUSTOM_DEP_FILE} did not create TIFF::TIFF target.")
      elseif(NOT NAME STREQUAL "LIBTIFF" AND NOT TARGET ${NAME}::${NAME})
          message(FATAL_ERROR "Custom dependency file ${CUSTOM_DEP_FILE} did not create ${NAME}::${NAME} target.")
      endif()
      return()
    endif()
    
    if(USE_SYSTEM_DEPS) # Changed from SYSTEM_DEPENDENCIES to match global option
      # Try to find the package in the system
      find_package(${NAME} QUIET) # For some deps, this might need FindXXX.cmake, e.g. FindLibTIFF.cmake
      if(${NAME}_FOUND)
        message(STATUS "Found system ${NAME} via find_package.")
        # Ensure an imported target NAME::NAME or TIFF::TIFF is created by FindXXX.cmake
        # If FindXXX.cmake doesn't create it, we might need to create it here.
        if(NAME STREQUAL "LIBTIFF" AND NOT TARGET TIFF::TIFF AND TARGET ${NAME}::libtiff) # Common alias from FindLibTIFF
            add_library(TIFF::TIFF ALIAS ${NAME}::libtiff)
        elseif(NOT TARGET ${NAME}::${NAME} AND ${NAME}_LIBRARIES AND ${NAME}_INCLUDE_DIRS)
            add_library(${NAME}::${NAME} INTERFACE IMPORTED)
            set_target_properties(${NAME}::${NAME} PROPERTIES
              INTERFACE_INCLUDE_DIRECTORIES "${${NAME}_INCLUDE_DIRS}"
              INTERFACE_LINK_LIBRARIES "${${NAME}_LIBRARIES}"
            )
            message(STATUS "Created imported target ${NAME}::${NAME} from find_package results.")
        elseif(NOT TARGET ${NAME}::${NAME} AND NOT TARGET TIFF::TIFF)
             message(WARNING "Found system ${NAME} but no standard target (e.g. ${NAME}::${NAME} or TIFF::TIFF) was created by find_package. Manual configuration might be needed if linking fails.")
        endif()
        return()
      endif()
      
      # Also try with PkgConfig for better system library discovery
      find_package(PkgConfig QUIET)
      if(PkgConfig_FOUND)
        string(TOLOWER "${NAME}" NAME_LOWER)
        # Adjust pkg-config name if necessary (e.g., libtiff-4 for LIBTIFF)
        set(PKG_CONFIG_NAME ${NAME_LOWER})
        if(NAME STREQUAL "LIBTIFF")
            set(PKG_CONFIG_NAME "libtiff-4")
        elseif(NAME STREQUAL "LIBZIP")
            set(PKG_CONFIG_NAME "libzip")
        elseif(NAME STREQUAL "FFTW")
            set(PKG_CONFIG_NAME "fftw3f") # For single precision float
        endif()

        pkg_check_modules(${NAME}_PC QUIET ${PKG_CONFIG_NAME})
        if(${NAME}_PC_FOUND)
          message(STATUS "Found ${NAME} using PkgConfig (${PKG_CONFIG_NAME})")
          set(${NAME}_FOUND TRUE PARENT_SCOPE) # Propagate _FOUND variable
          # Create an imported target for consistent interface
          if(NAME STREQUAL "LIBTIFF")
            if(NOT TARGET TIFF::TIFF) 
              add_library(TIFF::TIFF INTERFACE IMPORTED) 
            endif()
            set_target_properties(TIFF::TIFF PROPERTIES
              INTERFACE_INCLUDE_DIRECTORIES "${${NAME}_PC_INCLUDE_DIRS}"
              INTERFACE_LINK_LIBRARIES "${${NAME}_PC_LIBRARIES}"
            )
          else()
            if(NOT TARGET ${NAME}::${NAME}) 
              add_library(${NAME}::${NAME} INTERFACE IMPORTED) 
            endif()
            set_target_properties(${NAME}::${NAME} PROPERTIES
              INTERFACE_INCLUDE_DIRECTORIES "${${NAME}_PC_INCLUDE_DIRS}"
              INTERFACE_LINK_LIBRARIES "${${NAME}_PC_LIBRARIES}"
            )
          endif()
          return()
        else()
          message(STATUS "${NAME} not found in system via PkgConfig (${PKG_CONFIG_NAME}).")
        endif()
      endif()
      message(STATUS "${NAME} not found in system (checked find_package and PkgConfig).")
    endif()
    
    # If we get here, we need to download and build
    if(${DOWNLOAD_FLAG})
      message(STATUS "Setting up ${NAME} for download and build from ${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/${NAME}.cmake")
      # Ensure the .cmake file (e.g., cmake/deps/NLOPT.cmake) is generated first
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/${NAME}.cmake")
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/${NAME}.cmake)
      else()
        message(FATAL_ERROR "Dependency .cmake file not found for ${NAME} at ${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/${NAME}.cmake. Please ensure file(WRITE...) block exists before handle_dependency call.")
      endif()
    else()
      message(STATUS "Skipping download for ${NAME} as DOWNLOAD_DEPENDENCIES is OFF and it was not found on the system.")
      message(FATAL_ERROR "Could not find system ${NAME} and downloading is disabled. Please install ${NAME} or enable DOWNLOAD_DEPENDENCIES and USE_SYSTEM_DEPS=OFF.")
    endif()
    
    # Verify target was created by the included .cmake file
    if(NAME STREQUAL "LIBTIFF") # Special case for LIBTIFF target name
      if(NOT TARGET TIFF::TIFF)
        message(FATAL_ERROR "Failed to create TIFF::TIFF target for ${NAME} after including ${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/${NAME}.cmake")
      endif()
    elseif(NOT TARGET ${NAME}::${NAME})
      message(FATAL_ERROR "Failed to create ${NAME}::${NAME} target for ${NAME} after including ${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/${NAME}.cmake")
    endif()
endfunction()


# Handle ZLIB explicitly if not handled reliably by Blosc or other dependencies.
# This is a fallback. Ideally, Blosc2 (if configured to build Zlib) or a system Zlib is used.
find_package(ZLIB QUIET)
if(NOT ZLIB_FOUND AND DOWNLOAD_DEPENDENCIES AND NOT TARGET ZLIB::ZLIB)
  message(STATUS "ZLIB not found by find_package. Attempting to build ZLIB via FetchContent.")
  FetchContent_Declare(
    zlib_fetch # Use zlib_fetch to avoid conflict with potential ZLIB::ZLIB from find_package
    URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz # Official Zlib source
    URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
  )
  FetchContent_GetProperties(zlib_fetch)
  if(NOT zlib_fetch_POPULATED)
    FetchContent_Populate(zlib_fetch)
    # Add zlib as a subdirectory. Its CMakeLists.txt should create ZLIB::ZLIB target.
    # Need to ensure its CMakeLists.txt supports BUILD_SHARED_LIBS or similar.
    # Zlib's own CMake (since 1.2.12) creates zlibstatic and zlib (shared).
    # It respects BUILD_SHARED_LIBS global property.
    add_subdirectory(${zlib_fetch_SOURCE_DIR} ${zlib_fetch_BINARY_DIR})
    if(TARGET ZLIB::ZLIB)
        message(STATUS "ZLIB::ZLIB target successfully created from fetched zlib.")
        # Ensure install locations are handled if other projects need to find it via CMAKE_PREFIX_PATH
        # By default, FetchContent subprojects don't install.
        # If DEPS_INSTALL_DIR is used, zlib should be configured to install there.
        # However, linking against ZLIB::ZLIB target directly is preferred.
    else()
        message(WARNING "Fetched zlib but ZLIB::ZLIB target was not created. Check zlib's CMakeLists.txt.")
    endif()
  endif()
  # Re-check ZLIB_FOUND after attempting to build
  if(NOT ZLIB_FOUND AND TARGET ZLIB::ZLIB)
    set(ZLIB_FOUND TRUE) # Manually set if target exists now
    # Populate ZLIB_INCLUDE_DIRS and ZLIB_LIBRARIES if needed by older CMake scripts, though ZLIB::ZLIB is preferred.
    if(TARGET ZLIB::ZLIB)
        get_target_property(ZLIB_INCLUDE_DIRS ZLIB::ZLIB INTERFACE_INCLUDE_DIRECTORIES)
    endif()
  endif()
elif(TARGET ZLIB::ZLIB)
    set(ZLIB_FOUND TRUE) # If find_package found it and created the target.
    message(STATUS "ZLIB found and ZLIB::ZLIB target exists.")
else()
    message(WARNING "ZLIB not found and DOWNLOAD_DEPENDENCIES is OFF or ZLIB::ZLIB target does not exist. Some dependencies might fail.")
endif()

# Handle all dependencies
# Order can matter: BLOSC may provide ZLIB, which HDF5, LIBTIFF, LIBZIP might need.
# FFTW is independent. NLOPT is independent.
handle_dependency(BLOSC "GIT_REPO https://github.com/Blosc/c-blosc2.git" DOWNLOAD_DEPENDENCIES)
handle_dependency(BLOSC1 "GIT_REPO https://github.com/Blosc/c-blosc.git" DOWNLOAD_DEPENDENCIES)
handle_dependency(LIBTIFF "URL_PLACEHOLDER_LIBTIFF" DOWNLOAD_DEPENDENCIES) 
handle_dependency(HDF5 "URL_PLACEHOLDER_HDF5" DOWNLOAD_DEPENDENCIES)    
handle_dependency(NLOPT "URL_PLACEHOLDER_NLOPT" DOWNLOAD_DEPENDENCIES)    
handle_dependency(LIBZIP "URL_PLACEHOLDER_LIBZIP" DOWNLOAD_DEPENDENCIES)   
handle_dependency(FFTW "URL_PLACEHOLDER_FFTW" DOWNLOAD_DEPENDENCIES)     


# Check for OpenMP
if(BUILD_OMP)
  find_package(OpenMP QUIET) # QUIET to avoid verbose output if not found initially
  if(OpenMP_C_FOUND)
    message(STATUS "OpenMP C found. Flags: ${OpenMP_C_FLAGS}, Libraries: ${OpenMP_C_LIB_NAMES}")
    # OpenMP::OpenMP_C target will be available for linking
  else()
    message(WARNING "OpenMP not found by find_package(OpenMP). OpenMP-enabled components might not build or might try to enable OpenMP via compiler flags directly if supported (e.g. -fopenmp).")
    # Some compilers enable OpenMP with just a flag, without needing explicit libraries.
    # Check if compiler supports -fopenmp (or equivalent) if find_package fails.
    # This is more advanced and typically handled by setting CMAKE_C_FLAGS += ${OpenMP_C_FLAGS} if find_package populated it.
  endif()
endif()

# CUDA configuration
if(USE_CUDA)
  # find_package(CUDA ...) should have been called by project(LANGUAGES CUDA)
  # but we can call it again to ensure variables are set or to get more info.
  find_package(CUDA QUIET) # Find existing CUDA toolkit chosen by project() or system
  if(CUDA_FOUND)
    message(STATUS "CUDA Toolkit found: version ${CUDA_VERSION_STRING}, compiler: ${CMAKE_CUDA_COMPILER}")
    # enable_language(CUDA) # Should be enabled by project() if listed
    if(NOT CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 11 CACHE STRING "CUDA standard to use (e.g., 11, 14, 17)" FORCE)
    endif()
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      # Set a default, but this should ideally be user-configurable e.g. via cache variable or toolchain file
      set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "Target CUDA architectures (e.g., 60;70;75;80)") # Example: Volta
      message(STATUS "CMAKE_CUDA_ARCHITECTURES not set, defaulting to ${CMAKE_CUDA_ARCHITECTURES}. Configure as needed.")
    endif()
    message(STATUS "CUDA enabled with standard ${CMAKE_CUDA_STANDARD} for architectures ${CMAKE_CUDA_ARCHITECTURES}.")
    # Example CUDA flags (from FF_HEDM, adjust as necessary)
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -G --generate-code arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(WARNING "USE_CUDA is ON, but CUDA Toolkit not found/configured by 'project' command or find_package(CUDA). Disabling CUDA for this build.")
    set(USE_CUDA OFF CACHE BOOL "CUDA support" FORCE) # Force disable if not found
    # Need to re-evaluate project languages if CUDA was mandatory and is now disabled.
    # This might require a re-configure if project() definition changes.
  endif()
endif()

# Define COMMON_LINK_LIBRARIES for subprojects to use
set(COMMON_LINK_LIBRARIES "") # Initialize as empty list

if(TARGET NLOPT::NLOPT)
  list(APPEND COMMON_LINK_LIBRARIES NLOPT::NLOPT)
endif()
if(TARGET TIFF::TIFF) 
  list(APPEND COMMON_LINK_LIBRARIES TIFF::TIFF)
endif()
if(TARGET HDF5::HDF5 AND TARGET HDF5::HL)
  list(APPEND COMMON_LINK_LIBRARIES HDF5::HDF5 HDF5::HL)
endif()
if(TARGET BLOSC::BLOSC)
  list(APPEND COMMON_LINK_LIBRARIES BLOSC::BLOSC)
endif()
if(TARGET BLOSC1::BLOSC1)
  list(APPEND COMMON_LINK_LIBRARIES BLOSC1::BLOSC1)
endif()
if(TARGET LIBZIP::LIBZIP)
  list(APPEND COMMON_LINK_LIBRARIES LIBZIP::LIBZIP)
endif()
if(TARGET FFTW::FFTW)
  list(APPEND COMMON_LINK_LIBRARIES FFTW::FFTW)
endif()
if(ZLIB_FOUND AND TARGET ZLIB::ZLIB) 
    list(APPEND COMMON_LINK_LIBRARIES ZLIB::ZLIB)
    message(STATUS "ZLIB::ZLIB added to COMMON_LINK_LIBRARIES.")
elseif(ZLIB_FOUND AND ZLIB_LIBRARIES) # Fallback if ZLIB::ZLIB target not available but find_package worked
    list(APPEND COMMON_LINK_LIBRARIES ${ZLIB_LIBRARIES})
    message(STATUS "ZLIB libraries (${ZLIB_LIBRARIES}) added to COMMON_LINK_LIBRARIES.")
endif()

# Propagate options to subdirectories
# Variables set with `set(... CACHE ...)` are automatically available.
# For others, or to ensure they are available for `if()` checks in subdirs:
# add_compile_definitions($<$<BOOL:${BUILD_OMP}>:ENABLE_OMP> $<$<BOOL:${USE_CUDA}>:ENABLE_CUDA>)
# Subdirectories can then check for these definitions or use the global CMake options directly.

# Common compiler flags
set(COMMON_C_FLAGS_LIST "") # Initialize as empty list
list(APPEND COMMON_C_FLAGS_LIST -fPIC -O3 -w -g)

if(APPLE)
  list(APPEND COMMON_C_FLAGS_LIST -std=gnu99)
  if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_C_COMPILER_ID MATCHES "AppleClang")
    # GNU GCC on macOS might support this, Clang might not or it's default.
    list(APPEND COMMON_C_FLAGS_LIST -fgnu89-inline)
  endif()
else() # Non-Apple (Linux, Windows GCC/Clang)
  list(APPEND COMMON_C_FLAGS_LIST -fgnu89-inline) # Common for GCC/Clang on Linux
endif()
string(REPLACE ";" " " COMMON_C_FLAGS_STRING "${COMMON_C_FLAGS_LIST}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_C_FLAGS_STRING}")

# Common linker flags (target specific is better, but for general system libs)
if(NOT APPLE)
  # -ldl for dlopen, -lm for math library.
  # These are often needed by various libraries or programs.
  # Prefer adding to specific targets if not universally needed.
  set(CMAKE_EXE_LINKER_FLAGS_INIT "${CMAKE_EXE_LINKER_FLAGS_INIT} -ldl -lm")
else()
  # macOS usually doesn't need explicit -ldl as it's part of System library.
  # -lm is also often automatically linked or part of System.
  set(CMAKE_EXE_LINKER_FLAGS_INIT "${CMAKE_EXE_LINKER_FLAGS_INIT} -lm")
endif()

# Global include directories
# For dependencies installed into DEPS_INSTALL_DIR
include_directories(SYSTEM BEFORE "${DEPS_INSTALL_DIR}/include")
# For fetched content source directories (e.g. _deps/name-src/include)
# This can be useful if a library is used as add_subdirectory but not installed to DEPS_INSTALL_DIR
# However, it's generally better if each dependency's target (e.g. NLOPT::NLOPT) exports its own include dirs.
# Example: include_directories("${CMAKE_BINARY_DIR}/_deps/nlopt-src/src/api") - but NLOPT::NLOPT should do this.

# --- Add Subdirectories ---
# These will be the main components of MIDAS

# Add FF_HEDM (assuming it has its own CMakeLists.txt)
add_subdirectory(FF_HEDM)

# Add DT (will need a CMakeLists.txt)
add_subdirectory(DT)

# Add NF_HEDM (will need a CMakeLists.txt)
add_subdirectory(NF_HEDM)

# Add TOMO (will need a CMakeLists.txt)
add_subdirectory(TOMO)

# --- Python Scripts Handling (Placeholder) ---
# option(INSTALL_PYTHON_SCRIPTS "Install Python scripts from gui/ and utils/" ON)
# if(INSTALL_PYTHON_SCRIPTS)
#   # Logic for finding and installing python scripts
#   # Logic for pip install requirements.txt
# endif()

# --- Top-Level setup.sh Generation (Placeholder) ---
# file(CONFIGURE cmake/templates/setup.sh.in ${CMAKE_BINARY_DIR}/setup.sh @ONLY)
# install(PROGRAMS ${CMAKE_BINARY_DIR}/setup.sh DESTINATION ${MIDAS_INSTALL_BINDIR})


# --- Python Scripts and Dependencies Handling ---

# Install Python scripts from gui/ and utils/
if(INSTALL_PYTHON_SCRIPTS)
    # Define a relative path for installation, CMAKE_INSTALL_PREFIX will be prepended.
    set(PYTHON_SCRIPTS_INSTALL_SUBDIR "python_scripts")
    set(PYTHON_SCRIPTS_FULL_INSTALL_DIR "\${CMAKE_INSTALL_PREFIX}/\${PYTHON_SCRIPTS_INSTALL_SUBDIR}")

    # Install scripts from gui/
    # Using CMAKE_CURRENT_SOURCE_DIR to correctly point to the project's root gui/ directory
    file(GLOB_RECURSE GUI_PYTHON_SCRIPTS
         RELATIVE "\${CMAKE_CURRENT_SOURCE_DIR}/gui"
         "\${CMAKE_CURRENT_SOURCE_DIR}/gui/*.py"
         "\${CMAKE_CURRENT_SOURCE_DIR}/gui/*/*.py") # Catches scripts in subdirs of gui

    if(GUI_PYTHON_SCRIPTS)
        message(STATUS "Found Python scripts in gui/: \${GUI_PYTHON_SCRIPTS}")
        foreach(script_file \${GUI_PYTHON_SCRIPTS})
            # script_file is already relative to CMAKE_CURRENT_SOURCE_DIR/gui
            get_filename_component(script_rel_dir "\${script_file}" DIRECTORY)
            install(FILES "\${CMAKE_CURRENT_SOURCE_DIR}/gui/\${script_file}"
                    DESTINATION "\${PYTHON_SCRIPTS_FULL_INSTALL_DIR}/gui/\${script_rel_dir}"
                    COMPONENT Runtime # Using Runtime component, or define a new one e.g., PythonScripts
            )
        endforeach()
        message(STATUS "Installing Python scripts from gui/ to \${PYTHON_SCRIPTS_FULL_INSTALL_DIR}/gui")
    else()
        message(STATUS "No Python scripts found in gui/ or gui/ directory does not exist.")
    endif()

    # Install scripts from utils/
    file(GLOB_RECURSE UTILS_PYTHON_SCRIPTS
         RELATIVE "\${CMAKE_CURRENT_SOURCE_DIR}/utils"
         "\${CMAKE_CURRENT_SOURCE_DIR}/utils/*.py"
         "\${CMAKE_CURRENT_SOURCE_DIR}/utils/*/*.py")

    if(UTILS_PYTHON_SCRIPTS)
        message(STATUS "Found Python scripts in utils/: \${UTILS_PYTHON_SCRIPTS}")
        foreach(script_file \${UTILS_PYTHON_SCRIPTS})
            get_filename_component(script_rel_dir "\${script_file}" DIRECTORY)
            install(FILES "\${CMAKE_CURRENT_SOURCE_DIR}/utils/\${script_file}"
                    DESTINATION "\${PYTHON_SCRIPTS_FULL_INSTALL_DIR}/utils/\${script_rel_dir}"
                    COMPONENT Runtime
            )
        endforeach()
        message(STATUS "Installing Python scripts from utils/ to \${PYTHON_SCRIPTS_FULL_INSTALL_DIR}/utils")
    else()
        message(STATUS "No Python scripts found in utils/ or utils/ directory does not exist.")
    endif()
    
    # The setup.sh.in script will need to know PYTHON_SCRIPTS_INSTALL_SUBDIR
    # to correctly add it to PYTHONPATH relative to SCRIPT_DIR.
    # We will pass PYTHON_SCRIPTS_INSTALL_SUBDIR to configure_file for setup.sh.
    # No direct modification here, but ensure configure_file for setup.sh uses it.
endif()

# Install Python dependencies using pip
if(INSTALL_PYTHON_DEPENDENCIES)
  # Find Python executable if not already specified
  if(NOT PYTHON_EXECUTABLE OR NOT EXISTS "\${PYTHON_EXECUTABLE}")
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND)
      set(PYTHON_EXECUTABLE "\${Python3_EXECUTABLE}" CACHE FILEPATH "Path to Python executable for installing dependencies" FORCE)
      message(STATUS "Found Python 3 interpreter: \${PYTHON_EXECUTABLE}")
    else()
      find_program(PYTHON_EXECUTABLE_FALLBACK NAMES python3 python QUIET)
      if(PYTHON_EXECUTABLE_FALLBACK)
        set(PYTHON_EXECUTABLE "\${PYTHON_EXECUTABLE_FALLBACK}" CACHE FILEPATH "Path to Python executable for installing dependencies" FORCE)
        message(STATUS "Found Python interpreter (fallback): \${PYTHON_EXECUTABLE}")
      else()
        message(WARNING "Python interpreter not found. Cannot install Python dependencies. Please set PYTHON_EXECUTABLE.")
        set(INSTALL_PYTHON_DEPENDENCIES OFF CACHE BOOL "Install Python dependencies" FORCE) # Disable if Python not found
      endif()
    endif()
  else()
     message(STATUS "Using PYTHON_EXECUTABLE: \${PYTHON_EXECUTABLE} from cache or pre-set.")
  endif()

  if(INSTALL_PYTHON_DEPENDENCIES AND PYTHON_EXECUTABLE AND EXISTS "\${PYTHON_EXECUTABLE}")
    # Check for FF_HEDM/requirements.txt (currently the only known one)
    set(FF_HEDM_REQUIREMENTS_FILE "\${CMAKE_CURRENT_SOURCE_DIR}/FF_HEDM/requirements.txt")
    if(EXISTS "\${FF_HEDM_REQUIREMENTS_FILE}")
      # Ensure the target is unique if this block is ever duplicated or put in a loop.
      if(NOT TARGET midas_python_deps_ff_hedm)
        add_custom_target(midas_python_deps_ff_hedm
          COMMAND "\${PYTHON_EXECUTABLE}" -m pip install --user -r "\${FF_HEDM_REQUIREMENTS_FILE}"
          COMMENT "Installing Python dependencies from FF_HEDM/requirements.txt"
          VERBATIM
        )
        message(STATUS "Added custom target 'midas_python_deps_ff_hedm' to install requirements from \${FF_HEDM_REQUIREMENTS_FILE}")
        message(STATUS "To build this target, run: cmake --build . --target midas_python_deps_ff_hedm")
      endif()
    else()
      message(STATUS "FF_HEDM/requirements.txt not found. Skipping pip install target creation for it.")
    endif()
    # Add other requirements files here if necessary in the future
    # Example:
    # set(TOMO_REQUIREMENTS_FILE "\${CMAKE_CURRENT_SOURCE_DIR}/TOMO/requirements.txt")
    # if(EXISTS "\${TOMO_REQUIREMENTS_FILE}")
    #   if(NOT TARGET midas_python_deps_tomo)
    #     add_custom_target(midas_python_deps_tomo
    #       COMMAND "\${PYTHON_EXECUTABLE}" -m pip install --user -r "\${TOMO_REQUIREMENTS_FILE}"
    #       COMMENT "Installing Python dependencies from TOMO/requirements.txt"
    #       VERBATIM
    #     )
    #     message(STATUS "Added custom target 'midas_python_deps_tomo' ...")
    #   endif()
    # else()
    #   message(STATUS "TOMO/requirements.txt not found.")
    # endif()

  elseif(INSTALL_PYTHON_DEPENDENCIES) # If PYTHON_EXECUTABLE was not found
     message(WARNING "Python executable not found (INSTALL_PYTHON_DEPENDENCIES is ON but PYTHON_EXECUTABLE is not set or invalid). Cannot install Python dependencies.")
     set(INSTALL_PYTHON_DEPENDENCIES OFF CACHE BOOL "Install Python dependencies" FORCE)
  endif()
endif()


# --- Top-Level setup.sh Generation ---
# This uses the MIDAS_INSTALL_BINDIR which is set from CMAKE_INSTALL_BINDIR (GNUInstallDirs)
# The paths in setup.sh.in are relative to CMAKE_INSTALL_PREFIX/bin

set(SETUP_SH_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/setup.sh.in")
set(SETUP_SH_OUTPUT "${CMAKE_BINARY_DIR}/setup.sh") # Generate in build dir first

# Configure the setup script
# Pass PYTHON_SCRIPTS_INSTALL_SUBDIR to the template if scripts are being installed.
# The @ONLY flag means only variables like @VAR@ or ${VAR} are replaced.
# If setup.sh.in uses shell variables like $VAR or ${VAR}, they will be preserved.
if(INSTALL_PYTHON_SCRIPTS)
  set(CONFIGURE_SETUP_SH_DEFINES "-DPYTHON_SCRIPTS_INSTALL_SUBDIR=\\"python_scripts\\"")
  # The above define isn't used by configure_file directly with @ONLY unless setup.sh.in uses #cmakedefine
  # Instead, we ensure PYTHON_SCRIPTS_INSTALL_SUBDIR is a normal CMake variable available during configure_file.
  # It's already set if INSTALL_PYTHON_SCRIPTS is true.
endif()

configure_file(
  "${SETUP_SH_TEMPLATE}"  # Input: cmake/templates/setup.sh.in
  "${SETUP_SH_OUTPUT}"    # Output: ${CMAKE_BINARY_DIR}/setup.sh
  @ONLY                   # Replace @CMAKE_INSTALL_PREFIX@, @PYTHON_SCRIPTS_INSTALL_SUBDIR@, etc.
)

# Install the configured setup.sh script to the installation's bin directory
install(PROGRAMS "${SETUP_SH_OUTPUT}" DESTINATION "${MIDAS_INSTALL_BINDIR}")

message(STATUS "Configured top-level setup.sh, will be installed to ${MIDAS_INSTALL_BINDIR}")

# --- Summary Message ---
message(STATUS "")
message(STATUS "MIDAS Project Configuration Summary")
message(STATUS "------------------------------------")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build CUDA components: ${USE_CUDA}")
message(STATUS "Build OpenMP components: ${BUILD_OMP}")
message(STATUS "Download dependencies: ${DOWNLOAD_DEPENDENCIES}")
message(STATUS "Prefer system dependencies: ${USE_SYSTEM_DEPS}")
message(STATUS "------------------------------------")
message(STATUS "")
