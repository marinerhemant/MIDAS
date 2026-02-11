# CMakeLists.txt for NF_HEDM component of MIDAS

set(NF_SRCDIR src)

# List of common source files for some targets
set(NF_SHARED_FIT_SOURCES
    ${NF_SRCDIR}/GetMisorientation.c
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
add_nf_hedm_executable(MakeHexGrid SOURCES ${NF_SRCDIR}/MakeHexGrid.c)
add_nf_hedm_executable(MedianImageLibTiff SOURCES ${NF_SRCDIR}/MedianImageLibTiff.c LINK_LIBRARIES_EXTRA TIFF::TIFF)
add_nf_hedm_executable(ImageProcessingLibTiff SOURCES ${NF_SRCDIR}/ImageProcessingLibTiff.c LINK_LIBRARIES_EXTRA TIFF::TIFF)
add_nf_hedm_executable(FitOrientation SOURCES ${NF_SRCDIR}/FitOrientation.c ${NF_SHARED_FIT_SOURCES} LINK_LIBRARIES_EXTRA NLOPT::NLOPT)
add_nf_hedm_executable(FitOrientationParameters SOURCES ${NF_SRCDIR}/FitOrientationParameters.c ${NF_SRCDIR}/CalcDiffractionSpots.c ${NF_SRCDIR}/SharedFuncsFit.c ${NF_SRCDIR}/GetMisorientation.c LINK_LIBRARIES_EXTRA NLOPT::NLOPT)
add_nf_hedm_executable(FitOrientationParametersMultiPoint SOURCES ${NF_SRCDIR}/FitOrientationParametersMultiPoint.c ${NF_SRCDIR}/CalcDiffractionSpots.c ${NF_SRCDIR}/SharedFuncsFit.c ${NF_SRCDIR}/GetMisorientation.c LINK_LIBRARIES_EXTRA NLOPT::NLOPT)
add_nf_hedm_executable(GenSeedOrientationsFF2NFHEDM SOURCES ${NF_SRCDIR}/GenSeedOrientationsFF2NFHEDM.c ${NF_SRCDIR}/GetMisorientation.c LINK_LIBRARIES_EXTRA)
add_nf_hedm_executable(SimulateDiffractionSpots SOURCES ${NF_SRCDIR}/SimulateDiffractionSpots.c)
add_nf_hedm_executable(ProcessNFMicRemote SOURCES ${NF_SRCDIR}/ProcessNFMicRemote.c)
add_nf_hedm_executable(filterGridfromTomo SOURCES ${NF_SRCDIR}/filterGridfromTomo.c)
add_nf_hedm_executable(Mic2GrainsList SOURCES ${NF_SRCDIR}/Mic2GrainsList.c ${NF_SRCDIR}/GetMisorientation.c LINK_LIBRARIES_EXTRA)
add_nf_hedm_executable(MMapImageInfo SOURCES ${NF_SRCDIR}/MMapImageInfo.c)
add_nf_hedm_executable(GetHKLListNF SOURCES ${NF_SRCDIR}/GetHKLList.c ${NF_SG_LIB_SOURCES})
add_nf_hedm_executable(ParseMic SOURCES ${NF_SRCDIR}/ParseMic.c)
add_nf_hedm_executable(FitOrientationSinglePoint SOURCES ${NF_SRCDIR}/FitOrientationSinglePoint.c ${NF_SHARED_FIT_SOURCES} LINK_LIBRARIES_EXTRA NLOPT::NLOPT)
add_nf_hedm_executable(ParseDeconvOutput SOURCES ${NF_SRCDIR}/ParseDeconvOutput.c LINK_LIBRARIES_EXTRA TIFF::TIFF)
add_nf_hedm_executable(compareNF SOURCES ${NF_SRCDIR}/compareNF.c ${NF_SRCDIR}/CalcDiffractionSpots.c ${NF_SRCDIR}/SharedFuncsFit.c LINK_LIBRARIES_EXTRA NLOPT::NLOPT)
add_nf_hedm_executable(simulateNF SOURCES ${NF_SRCDIR}/simulateNF.c ${NF_SRCDIR}/CalcDiffractionSpots.c ${NF_SRCDIR}/SharedFuncsFit.c)

# --- OpenMP Executables ---
# Note: The helper function doesn't handle OpenMP explicitly, OpenMP::OpenMP_C should be in COMMON_LINK_LIBRARIES or added if specific
add_executable(ImageProcessingLibTiffOMP ${NF_SRCDIR}/ImageProcessingLibTiffOMP.c)
target_link_libraries(ImageProcessingLibTiffOMP PRIVATE TIFF::TIFF ${COMMON_LINK_LIBRARIES})
if(OpenMP_C_FOUND AND BUILD_OMP)
    target_link_libraries(ImageProcessingLibTiffOMP PRIVATE OpenMP::OpenMP_C)
endif()
install(TARGETS ImageProcessingLibTiffOMP RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})
add_custom_command(TARGET ImageProcessingLibTiffOMP POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ImageProcessingLibTiffOMP> "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
    COMMENT "Copied ImageProcessingLibTiffOMP to ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
    VERBATIM
)

add_executable(FitOrientationOMP ${NF_SRCDIR}/FitOrientationOMP.c ${NF_SHARED_FIT_SOURCES})
target_link_libraries(FitOrientationOMP PRIVATE NLOPT::NLOPT ${COMMON_LINK_LIBRARIES})
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

# --- Shared Library ---
add_library(NFGrainsCalc SHARED ${NF_SRCDIR}/NFGrainsCalc.c ${NF_SRCDIR}/GetMisorientation.c)
target_link_libraries(NFGrainsCalc PRIVATE ${COMMON_LINK_LIBRARIES})
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
# add_executable(MedianImageHDF ${NF_SRCDIR}/MedianImageHDF.c)
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
    GenSeedOrientationsFF2NFHEDM SimulateDiffractionSpots ProcessNFMicRemote
    filterGridfromTomo MMapImageInfo GetHKLListNF ParseMic
    FitOrientationSinglePoint ParseDeconvOutput NFGrainsCalc
    compareNF simulateNF ImageProcessingLibTiffOMP FitOrientationOMP
)