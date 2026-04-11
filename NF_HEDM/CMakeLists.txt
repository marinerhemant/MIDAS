# CMakeLists.txt for NF_HEDM component of MIDAS

set(NF_SRCDIR src)

# Allow NF_HEDM targets to include shared headers like midas_paths.h from FF_HEDM
include_directories(${CMAKE_SOURCE_DIR}/FF_HEDM/src)

# List of common source files for some targets
set(NF_SHARED_FIT_SOURCES
  ${NF_SRCDIR}/CalcDiffractionSpots.c
  ${NF_SRCDIR}/SharedFuncsFit.c
)

set(NF_SG_LIB_SOURCES # For GetHKLList
  ${NF_SRCDIR}/sgclib.c
  ${NF_SRCDIR}/sgfind.c
  ${NF_SRCDIR}/sghkl.c
  ${NF_SRCDIR}/sgsi.c
  ${NF_SRCDIR}/sgio.c
)

# Helper function to add executable and the custom command to copy to source/bin
function(add_nf_hedm_executable TARGET_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SOURCES LINK_LIBRARIES_EXTRA) # Allow extra libs if needed
  cmake_parse_arguments(NF_EXEC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_executable(${TARGET_NAME} ${NF_EXEC_SOURCES})
  target_link_libraries(${TARGET_NAME} PRIVATE ${COMMON_LINK_LIBRARIES} ${NF_EXEC_LINK_LIBRARIES_EXTRA})
  install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
   COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
   COMMENT "Copied ${TARGET_NAME} to ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
   VERBATIM
  )
endfunction()


# --- Executables (from 'all' rule in Makefile) ---

add_nf_hedm_executable(MakeDiffrSpots SOURCES ${NF_SRCDIR}/MakeDiffrSpots.c)
if(BUILD_OMP AND OpenMP_C_FOUND)
  target_link_libraries(MakeDiffrSpots PRIVATE OpenMP::OpenMP_C)
endif()
add_nf_hedm_executable(MakeHexGrid SOURCES ${NF_SRCDIR}/MakeHexGrid.c)
add_nf_hedm_executable(MedianImageLibTiff SOURCES ${NF_SRCDIR}/MedianImageLibTiff.c LINK_LIBRARIES_EXTRA TIFF::TIFF)
if(OpenMP_C_FOUND AND BUILD_OMP)
  target_link_libraries(MedianImageLibTiff PRIVATE OpenMP::OpenMP_C)
endif()
# add_nf_hedm_executable(ImageProcessingLibTiff SOURCES src/archive/ImageProcessingLibTiff.c LINK_LIBRARIES_EXTRA TIFF::TIFF)
# add_nf_hedm_executable(FitOrientation SOURCES src/archive/FitOrientation.c ${NF_SHARED_FIT_SOURCES} LINK_LIBRARIES_EXTRA NLOPT::NLOPT)
add_nf_hedm_executable(FitOrientationParameters SOURCES ${NF_SRCDIR}/FitOrientationParameters.c ${NF_SRCDIR}/CalcDiffractionSpots.c ${NF_SRCDIR}/SharedFuncsFit.c ${CMAKE_SOURCE_DIR}/FF_HEDM/src/MIDAS_Math.c LINK_LIBRARIES_EXTRA NLOPT::NLOPT)
target_link_libraries(FitOrientationParameters PRIVATE midas_orientation)
if(BUILD_OMP AND OpenMP_C_FOUND)
  target_link_libraries(FitOrientationParameters PRIVATE OpenMP::OpenMP_C)
endif()
add_nf_hedm_executable(GenSeedOrientationsFF2NFHEDM SOURCES ${NF_SRCDIR}/GenSeedOrientationsFF2NFHEDM.c LINK_LIBRARIES_EXTRA)
target_link_libraries(GenSeedOrientationsFF2NFHEDM PRIVATE midas_orientation)
add_nf_hedm_executable(SimulateDiffractionSpots SOURCES ${NF_SRCDIR}/SimulateDiffractionSpots.c)
# add_nf_hedm_executable(ProcessNFMicRemote SOURCES ${NF_SRCDIR}/ProcessNFMicRemote.c)
add_nf_hedm_executable(filterGridfromTomo SOURCES ${NF_SRCDIR}/filterGridfromTomo.c)
add_nf_hedm_executable(Mic2GrainsList SOURCES ${NF_SRCDIR}/Mic2GrainsList.c LINK_LIBRARIES_EXTRA)
target_link_libraries(Mic2GrainsList PRIVATE midas_orientation)
if(BUILD_OMP AND OpenMP_C_FOUND)
  target_link_libraries(Mic2GrainsList PRIVATE OpenMP::OpenMP_C)
endif()
add_nf_hedm_executable(MMapImageInfo SOURCES ${NF_SRCDIR}/MMapImageInfo.c)
if(BUILD_OMP AND OpenMP_C_FOUND)
  target_link_libraries(MMapImageInfo PRIVATE OpenMP::OpenMP_C)
endif()
add_nf_hedm_executable(GetHKLListNF SOURCES ${NF_SRCDIR}/GetHKLList.c ${NF_SRCDIR}/MIDAS_ParamParser.c ${NF_SG_LIB_SOURCES})
add_nf_hedm_executable(ParseMic SOURCES ${NF_SRCDIR}/ParseMic.c)
target_link_libraries(ParseMic PRIVATE midas_orientation)
if(BUILD_OMP AND OpenMP_C_FOUND)
  target_link_libraries(ParseMic PRIVATE OpenMP::OpenMP_C)
endif()
# add_nf_hedm_executable(FitOrientationSinglePoint SOURCES src/archive/FitOrientationSinglePoint.c ${NF_SHARED_FIT_SOURCES} LINK_LIBRARIES_EXTRA NLOPT::NLOPT)
add_nf_hedm_executable(ParseDeconvOutput SOURCES ${NF_SRCDIR}/ParseDeconvOutput.c LINK_LIBRARIES_EXTRA TIFF::TIFF)
add_nf_hedm_executable(compareNF SOURCES ${NF_SRCDIR}/compareNF.c ${NF_SRCDIR}/CalcDiffractionSpots.c ${NF_SRCDIR}/SharedFuncsFit.c ${CMAKE_SOURCE_DIR}/FF_HEDM/src/MIDAS_Math.c LINK_LIBRARIES_EXTRA NLOPT::NLOPT)
target_link_libraries(compareNF PRIVATE midas_orientation)
add_nf_hedm_executable(simulateNF SOURCES ${NF_SRCDIR}/simulateNF.c ${NF_SRCDIR}/CalcDiffractionSpots.c ${NF_SRCDIR}/SharedFuncsFit.c ${CMAKE_SOURCE_DIR}/FF_HEDM/src/MIDAS_Math.c LINK_LIBRARIES_EXTRA NLOPT::NLOPT)
target_link_libraries(simulateNF PRIVATE midas_orientation)
if(BUILD_OMP AND OpenMP_C_FOUND)
 target_link_libraries(simulateNF PRIVATE OpenMP::OpenMP_C)
endif()

# --- OpenMP Executables ---
# Note: The helper function doesn't handle OpenMP explicitly, OpenMP::OpenMP_C should be in COMMON_LINK_LIBRARIES or added if specific
add_executable(FitOrientationParametersMultiPoint ${NF_SRCDIR}/FitOrientationParametersMultiPoint.c ${NF_SRCDIR}/CalcDiffractionSpots.c ${NF_SRCDIR}/SharedFuncsFit.c ${CMAKE_SOURCE_DIR}/FF_HEDM/src/MIDAS_Math.c)
target_link_libraries(FitOrientationParametersMultiPoint PRIVATE midas_orientation)
target_link_libraries(FitOrientationParametersMultiPoint PRIVATE NLOPT::NLOPT ${COMMON_LINK_LIBRARIES})
if(OpenMP_C_FOUND AND BUILD_OMP)
  target_link_libraries(FitOrientationParametersMultiPoint PRIVATE OpenMP::OpenMP_C)
endif()
install(TARGETS FitOrientationParametersMultiPoint RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})
add_custom_command(TARGET FitOrientationParametersMultiPoint POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:FitOrientationParametersMultiPoint> "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
  COMMENT "Copied FitOrientationParametersMultiPoint to ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
  VERBATIM
)


add_executable(ProcessImagesCombined ${NF_SRCDIR}/ProcessImagesCombined.c)
target_link_libraries(ProcessImagesCombined PRIVATE TIFF::TIFF ${COMMON_LINK_LIBRARIES})
if(OpenMP_C_FOUND AND BUILD_OMP)
  target_link_libraries(ProcessImagesCombined PRIVATE OpenMP::OpenMP_C)
endif()
install(TARGETS ProcessImagesCombined RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})
add_custom_command(TARGET ProcessImagesCombined POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ProcessImagesCombined> "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
  COMMENT "Copied ProcessImagesCombined to ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
  VERBATIM
)

add_executable(FitOrientationOMP ${NF_SRCDIR}/FitOrientationOMP.c ${NF_SHARED_FIT_SOURCES} ${CMAKE_SOURCE_DIR}/FF_HEDM/src/MIDAS_Math.c)
target_link_libraries(FitOrientationOMP PRIVATE NLOPT::NLOPT ${COMMON_LINK_LIBRARIES} midas_orientation)
if(OpenMP_C_FOUND AND BUILD_OMP)
  target_link_libraries(FitOrientationOMP PRIVATE OpenMP::OpenMP_C)
endif()
install(TARGETS FitOrientationOMP RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})
add_custom_command(TARGET FitOrientationOMP POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:FitOrientationOMP> "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
  COMMENT "Copied FitOrientationOMP to ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
  VERBATIM
)

# --- GPU-accelerated FitOrientation (requires USE_CUDA=ON) ---
if(USE_CUDA AND CUDA_FOUND)
  set(NF_GPU_SOURCES
    ${NF_SRCDIR}/FitOrientationOMP.c
    ${NF_SHARED_FIT_SOURCES}
    ${CMAKE_SOURCE_DIR}/FF_HEDM/src/MIDAS_Math.c
    ${NF_SRCDIR}/FitOrientationGPU.cu
  )

  add_executable(FitOrientationGPU ${NF_GPU_SOURCES})
  target_link_libraries(FitOrientationGPU PRIVATE NLOPT::NLOPT ${COMMON_LINK_LIBRARIES})
  target_link_libraries(FitOrientationGPU PRIVATE cudart)
  target_compile_definitions(FitOrientationGPU PRIVATE ENABLE_CUDA)
  target_include_directories(FitOrientationGPU PRIVATE ${CMAKE_SOURCE_DIR}/utils)

  if(OpenMP_C_FOUND AND BUILD_OMP)
    target_link_libraries(FitOrientationGPU PRIVATE OpenMP::OpenMP_C)
    # nvcc needs explicit -fopenmp passed to host compiler for .cu files
    target_compile_options(FitOrientationGPU PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    )
  endif()

  set_target_properties(FitOrientationGPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  install(TARGETS FitOrientationGPU RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})
  add_custom_command(TARGET FitOrientationGPU POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:FitOrientationGPU> "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
    COMMENT "Copied FitOrientationGPU to ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
    VERBATIM
  )
  message(STATUS "NF_HEDM: GPU target FitOrientationGPU enabled")
else()
  message(STATUS "NF_HEDM: GPU target FitOrientationGPU skipped (USE_CUDA=${USE_CUDA})")
endif()

# --- Shared Library ---
add_library(NFGrainsCalc SHARED ${NF_SRCDIR}/NFGrainsCalc.c)
target_link_libraries(NFGrainsCalc PRIVATE ${COMMON_LINK_LIBRARIES} midas_orientation)
set_target_properties(NFGrainsCalc PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS NFGrainsCalc
  LIBRARY DESTINATION ${MIDAS_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${MIDAS_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR} # For DLLs on Windows
)
# Note: Copying shared libraries to source/bin is less common but can be done if needed:
# add_custom_command(TARGET NFGrainsCalc POST_BUILD
#   COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/lib" # or bin, depending on preference
#   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:NFGrainsCalc> "${CMAKE_CURRENT_SOURCE_DIR}/lib/"
#   COMMENT "Copied NFGrainsCalc to ${CMAKE_CURRENT_SOURCE_DIR}/lib/"
#   VERBATIM
# )


# --- HDF5 related executables (example if uncommented) ---
# add_executable(MedianImageHDF src/archive/MedianImageHDF.c)
# target_link_libraries(MedianImageHDF PRIVATE HDF5::HDF5 HDF5::HL ${COMMON_LINK_LIBRARIES})
# if(OpenMP_C_FOUND AND BUILD_OMP)
#     target_link_libraries(MedianImageHDF PRIVATE OpenMP::OpenMP_C)
# endif()
# install(TARGETS MedianImageHDF RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})
# add_custom_command(TARGET MedianImageHDF POST_BUILD
#   COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
#   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:MedianImageHDF> "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
#   VERBATIM)


# Optional: Add a custom target to build all things in this subdirectory
add_custom_target(midas_nf_hedm_all DEPENDS
  MakeDiffrSpots MakeHexGrid MedianImageLibTiff ImageProcessingLibTiff
  FitOrientation FitOrientationParameters FitOrientationParametersMultiPoint
  GenSeedOrientationsFF2NFHEDM SimulateDiffractionSpots 
  filterGridfromTomo MMapImageInfo GetHKLListNF ParseMic
  FitOrientationSinglePoint ParseDeconvOutput NFGrainsCalc
  compareNF simulateNF FitOrientationOMP
)