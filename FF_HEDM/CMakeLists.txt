# FF_HEDM specific CMakeLists.txt
# This file relies on the top-level CMakeLists.txt for project setup,
# options, dependency management, and global configurations.


# Function to add an FF_HEDM executable.
# Relies on variables/options from the top-level CMakeLists.txt:
# - COMMON_LINK_LIBRARIES (list of dependency targets)
# - USE_CUDA (global option to enable/disable CUDA)
# - BUILD_OMP (global option to enable/disable OpenMP)
# - OpenMP_C_FOUND (boolean, true if OpenMP C is found)
# - OpenMP::OpenMP_C (CMake target for OpenMP C libraries)
# - MIDAS_INSTALL_BINDIR (global installation directory for executables)
function(add_ff_hedm_executable TARGET_NAME)
  set(options CUDA OMP) # Options for this function: build with CUDA, build with OpenMP
  set(oneValueArgs "")  # No single-value arguments
  set(multiValueArgs SOURCES LINK_LIBRARIES) # Multi-value arguments: source files, specific libraries
  cmake_parse_arguments(FF_HEDM_EXEC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Conditional build for CUDA executables
  if(FF_HEDM_EXEC_CUDA AND NOT USE_CUDA)
    message(STATUS "FF_HEDM: Skipping CUDA target ${TARGET_NAME} as USE_CUDA is OFF.")
    return()
  endif()

  # Handle OpenMP
  set(USE_OMP_FOR_TARGET FALSE)
  if(FF_HEDM_EXEC_OMP AND BUILD_OMP AND OpenMP_C_FOUND)
    set(USE_OMP_FOR_TARGET TRUE)
    message(STATUS "FF_HEDM: Enabling OpenMP for ${TARGET_NAME}.")
  elseif(FF_HEDM_EXEC_OMP)
    message(STATUS "FF_HEDM: OpenMP requested for ${TARGET_NAME}, but OpenMP not available/enabled globally (BUILD_OMP: ${BUILD_OMP}, OpenMP_C_FOUND: ${OpenMP_C_FOUND}).")
  endif()

  # Add the executable target
  if(FF_HEDM_EXEC_CUDA)
    add_executable(${TARGET_NAME} ${FF_HEDM_EXEC_SOURCES})
    # CUDA specific properties are expected to be handled by top-level settings if needed
    # e.g. CMAKE_CUDA_ARCHITECTURES, CMAKE_CUDA_STANDARD
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # Example property
  else()
    add_executable(${TARGET_NAME} ${FF_HEDM_EXEC_SOURCES})
  endif()

  # Link libraries:
  # 1. COMMON_LINK_LIBRARIES (from top-level, e.g., NLOPT::NLOPT, HDF5::HDF5)
  # 2. FF_HEDM_EXEC_LINK_LIBRARIES (specific to this executable, if any)
  target_link_libraries(${TARGET_NAME} PRIVATE ${COMMON_LINK_LIBRARIES} ${FF_HEDM_EXEC_LINK_LIBRARIES})

  # Link OpenMP if enabled for this target
  if(USE_OMP_FOR_TARGET)
    target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_C)
  endif()

  # Install the executable to the globally defined binary directory
  install(TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR}
  )

  # Copy to a local bin directory in the FF_HEDM SOURCE subdir
  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"  # Changed to SOURCE_DIR
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> "${CMAKE_CURRENT_SOURCE_DIR}/bin/" # Changed to SOURCE_DIR
    COMMENT "Copied ${TARGET_NAME} to ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
    VERBATIM
  )
endfunction()


# --- Define FF_HEDM Executables ---
# Most executables link against COMMON_LINK_LIBRARIES implicitly via add_ff_hedm_executable.
# Any libraries listed in LINK_LIBRARIES are additional, target-specific dependencies.

add_ff_hedm_executable(Calibrant SOURCES src/Calibrant.c src/CalcPeakProfile.c)
add_ff_hedm_executable(CalibrantOMP SOURCES src/CalibrantOMP.c src/CalcPeakProfile.c OMP)
add_ff_hedm_executable(CalibrantPanelShiftsOMP SOURCES src/CalibrantPanelShiftsOMP.c src/CalcPeakProfile.c src/Panel.c src/FileReader.c OMP)
add_ff_hedm_executable(FitSetup SOURCES src/FitSetupParamsAll.c)
add_ff_hedm_executable(FitTiltBCLsdSample SOURCES src/FitTiltBCLsdSampleOmegaCorrection.c)
add_ff_hedm_executable(ForwardSimulation SOURCES src/ForwardSimulation.c)
add_ff_hedm_executable(ForwardSimulationCompressed SOURCES src/ForwardSimulationCompressed.c src/Panel.c OMP)

add_ff_hedm_executable(FitPosOrStrainsOMP SOURCES src/FitPosOrStrainsOMP.c src/CalcDiffractionSpots.c OMP)
add_ff_hedm_executable(FitPosOrStrainsDoubleDataset SOURCES src/FitPosOrStrainsDoubleDataset.c src/CalcDiffractionSpots.c OMP)
add_ff_hedm_executable(FitOrStrainsScanningOMP SOURCES src/FitOrStrainsScanningOMP.c src/CalcDiffractionSpots.c OMP)

add_ff_hedm_executable(PeaksFittingPerFile SOURCES src/PeaksFittingPerFile.c)
add_ff_hedm_executable(PeaksFittingOMP SOURCES src/PeaksFittingOMP.c OMP)
add_ff_hedm_executable(MapMultipleDetectors SOURCES src/MapMultipleDetectors.c)
add_ff_hedm_executable(MergeOverlappingPeaks SOURCES src/MergeOverlappingPeaks.c)
add_ff_hedm_executable(MergeOverlappingPeaksAll SOURCES src/MergeOverlappingPeaksAll.c)
add_ff_hedm_executable(CalcRadius SOURCES src/CalcRadius.c)
add_ff_hedm_executable(CalcRadiusAll SOURCES src/CalcRadiusAll.c)
add_ff_hedm_executable(FindSaturatedPixels SOURCES src/FindSaturatedPixels.c)
add_ff_hedm_executable(GrainTracking SOURCES src/GrainTracking.c)
add_ff_hedm_executable(MergeMultipleRings SOURCES src/MergeMultipleRings.c)
add_ff_hedm_executable(GenMedianDark SOURCES src/GenMedianDark.c)

add_ff_hedm_executable(FitGrain SOURCES src/FitGrain.c src/CalcDiffractionSpots.c src/Panel.c OMP)
add_ff_hedm_executable(FitMultipleGrains SOURCES src/FitMultipleGrains.c src/CalcDiffractionSpots.c src/Panel.c OMP)
add_ff_hedm_executable(FitGrainHydra SOURCES src/FitGrainHydra.c src/CalcDiffractionSpots.c)
add_ff_hedm_executable(FitScanningGrain SOURCES src/FitScanningGrain.c)
add_ff_hedm_executable(FitTiltX SOURCES src/FitTiltX.c)
add_ff_hedm_executable(FitWedge SOURCES src/FitWedge.c)
add_ff_hedm_executable(FitWedgeParallel SOURCES src/FitWedgeParallel.c src/Panel.c OMP)
add_ff_hedm_executable(MapDatasets SOURCES src/MapDatasets.c OMP)

add_ff_hedm_executable(GetHKLList SOURCES src/GetHKLList.c src/sgclib.c src/sgfind.c src/sghkl.c src/sgsi.c src/sgio.c)
add_ff_hedm_executable(PeaksFittingOMPZarr SOURCES src/PeaksFittingOMPZarr.c OMP)
add_ff_hedm_executable(PeaksFittingOMPZarrRefactor SOURCES src/PeaksFittingOMPZarrRefactor.c src/Panel.c OMP)
add_ff_hedm_executable(GetHKLListZarr SOURCES src/GetHKLListZarr.c src/sgclib.c src/sgfind.c src/sghkl.c src/sgsi.c src/sgio.c)
add_ff_hedm_executable(MergeOverlappingPeaksAllZarr SOURCES src/MergeOverlappingPeaksAllZarr.c)
add_ff_hedm_executable(CalcRadiusAllZarr SOURCES src/CalcRadiusAllZarr.c)
add_ff_hedm_executable(FitSetupZarr SOURCES src/FitSetupParamsAllZarr.c)
add_ff_hedm_executable(ProcessGrainsZarr SOURCES src/ProcessGrainsZarr.c src/GetMisorientation.c src/CalcStrains.c)

add_ff_hedm_executable(IndexerOMP SOURCES src/IndexerOMP.c OMP)
add_ff_hedm_executable(IndexerScanningOMP SOURCES src/IndexerScanningOMP.c OMP)

add_ff_hedm_executable(SaveBinData SOURCES src/SaveBinData.c)
add_ff_hedm_executable(SaveBinDataScanning SOURCES src/SaveBinDataScanning.c)
add_ff_hedm_executable(MergeMultipleScans SOURCES src/MergeMultipleScans.c)

add_ff_hedm_executable(ProcessGrains SOURCES src/ProcessGrains.c src/GetMisorientation.c src/CalcStrains.c)
add_ff_hedm_executable(ProcessGrainsScanningHEDM SOURCES src/ProcessGrainsScanningHEDM.c src/GetMisorientation.c src/CalcStrains.c)
add_ff_hedm_executable(MatchGrains SOURCES src/MatchGrains.c src/GetMisorientation.c)

add_ff_hedm_executable(DetectorMapper SOURCES src/DetectorMapper.c src/Panel.c src/FileReader.c)
add_ff_hedm_executable(DetectorMapperZarr SOURCES src/DetectorMapperZarr.c src/Panel.c src/FileReader.c)
add_ff_hedm_executable(Integrator SOURCES src/Integrator.c src/FileReader.c)
add_ff_hedm_executable(IntegratorZarr SOURCES src/IntegratorZarr.c)
add_ff_hedm_executable(IntegratorZarrOMP SOURCES src/IntegratorZarrOMP.c OMP)
add_ff_hedm_executable(ConvTiffToGE SOURCES src/ConvTiffToGE.c)
add_ff_hedm_executable(MapBND SOURCES src/MapBND.c)

# imageMax is a shared library loaded via ctypes from the Python GUI (ff_asym.py)
add_library(imageMax SHARED src/imageMax.c)
set_target_properties(imageMax PROPERTIES
  OUTPUT_NAME "imageMax"
  PREFIX ""       # produce imageMax.so, not libimageMax.so
  SUFFIX ".so"
)
install(TARGETS imageMax LIBRARY DESTINATION ${MIDAS_INSTALL_BINDIR})
add_custom_command(TARGET imageMax POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:imageMax> "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
  COMMENT "Copied imageMax.so to ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
  VERBATIM
)

add_ff_hedm_executable(findSingleSolutionPF SOURCES src/findSingleSolutionPF.c src/GetMisorientation.c OMP)
add_ff_hedm_executable(findSingleSolutionPFRefactored SOURCES src/findSingleSolutionPFRefactored.c src/GetMisorientation.c OMP)
add_ff_hedm_executable(findMultipleSolutionsPF SOURCES src/findMultipleSolutionsPF.c src/GetMisorientation.c OMP)
add_ff_hedm_executable(mergeScansScanning SOURCES src/mergeScansScanning.c OMP)

# CUDA specific targets
add_ff_hedm_executable(IntegratorCUDA SOURCES src/IntegratorFitPeaksGPUStream.cu CUDA OMP)
# Example for other CUDA targets if they were to be re-added:
# add_ff_hedm_executable(IndexerCUDA SOURCES src/IndexerRefineNLOPT.cu CUDA)
# add_ff_hedm_executable(PeaksCUDA SOURCES src/PeaksNLMD.cu CUDA)


# --- Aggregate target for all FF_HEDM executables (optional convenience) ---
set(FF_HEDM_ALL_NON_CUDA_TARGETS_LIST
    Calibrant CalibrantOMP FitSetup FitTiltBCLsdSample ForwardSimulation ForwardSimulationCompressed
    FitPosOrStrains FitPosOrStrainsOMP FitOrStrainsScanningOMP FitPosOrStrainsScanningHEDM
    PeaksFittingPerFile PeaksFittingOMP MapMultipleDetectors MergeOverlappingPeaks MergeOverlappingPeaksAll
    CalcRadius CalcRadiusAll FindSaturatedPixels GrainTracking MergeMultipleRings GenMedianDark
    FitGrain FitGrainHydra FitScanningGrain FitTiltX FitWedge GetHKLList PeaksFittingOMPZarr
    PeaksFittingOMPZarrRefactor GetHKLListZarr MergeOverlappingPeaksAllZarr CalcRadiusAllZarr
    FitSetupZarr ProcessGrainsZarr IndexerLinuxArgsShm IndexerOMP IndexerScanningOMP IndexScanningHEDM
    SaveBinData SaveBinDataScanning MergeMultipleScans ProcessGrains ProcessGrainsScanningHEDM MatchGrains
    DetectorMapper DetectorMapperZarr Integrator IntegratorZarr ConvTiffToGE MapBND
    findSingleSolutionPF findSingleSolutionPFRefactored findMultipleSolutionsPF mergeScansScanning
)

add_custom_target(midas_ff_hedm_all_executables DEPENDS ${FF_HEDM_ALL_NON_CUDA_TARGETS_LIST} imageMax)

if(USE_CUDA)
  add_dependencies(midas_ff_hedm_all_executables IntegratorCUDA)
  # Add other CUDA targets here if re-enabled:
  # add_dependencies(midas_ff_hedm_all_executables IndexerCUDA PeaksCUDA)
endif()

message(STATUS "FF_HEDM: Configured submodule targets. Build using 'midas_ff_hedm_all_executables' or individual target names.")
message(STATUS "FF_HEDM: Relies on top-level CMakeLists.txt for dependencies, options, and global settings.")