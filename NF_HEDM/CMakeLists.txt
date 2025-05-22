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

# --- Executables (from 'all' rule in Makefile) ---

add_executable(MakeDiffrSpots ${NF_SRCDIR}/MakeDiffrSpots.c)
install(TARGETS MakeDiffrSpots RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(MakeHexGrid ${NF_SRCDIR}/MakeHexGrid.c)
install(TARGETS MakeHexGrid RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(MedianImageLibTiff ${NF_SRCDIR}/MedianImageLibTiff.c)
target_link_libraries(MedianImageLibTiff PRIVATE TIFF::TIFF) # From COMMON_LINK_LIBRARIES
install(TARGETS MedianImageLibTiff RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(ImageProcessingLibTiff ${NF_SRCDIR}/ImageProcessingLibTiff.c)
target_link_libraries(ImageProcessingLibTiff PRIVATE TIFF::TIFF)
install(TARGETS ImageProcessingLibTiff RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(FitOrientation ${NF_SRCDIR}/FitOrientation.c ${NF_SHARED_FIT_SOURCES})
target_link_libraries(FitOrientation PRIVATE NLOPT::NLOPT) # From COMMON_LINK_LIBRARIES
install(TARGETS FitOrientation RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(FitOrientationParameters ${NF_SRCDIR}/FitOrientationParameters.c ${NF_SRCDIR}/CalcDiffractionSpots.c ${NF_SRCDIR}/SharedFuncsFit.c)
target_link_libraries(FitOrientationParameters PRIVATE NLOPT::NLOPT)
install(TARGETS FitOrientationParameters RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(FitOrientationParametersMultiPoint ${NF_SRCDIR}/FitOrientationParametersMultiPoint.c ${NF_SRCDIR}/CalcDiffractionSpots.c ${NF_SRCDIR}/SharedFuncsFit.c)
target_link_libraries(FitOrientationParametersMultiPoint PRIVATE NLOPT::NLOPT)
install(TARGETS FitOrientationParametersMultiPoint RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(GenSeedOrientationsFF2NFHEDM ${NF_SRCDIR}/GenSeedOrientationsFF2NFHEDM.c)
install(TARGETS GenSeedOrientationsFF2NFHEDM RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(SimulateDiffractionSpots ${NF_SRCDIR}/SimulateDiffractionSpots.c)
install(TARGETS SimulateDiffractionSpots RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(ProcessNFMicRemote ${NF_SRCDIR}/ProcessNFMicRemote.c)
install(TARGETS ProcessNFMicRemote RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(filterGridfromTomo ${NF_SRCDIR}/filterGridfromTomo.c) # Renamed from GridFilterTomo for consistency
install(TARGETS filterGridfromTomo RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(MMapImageInfo ${NF_SRCDIR}/MMapImageInfo.c)
install(TARGETS MMapImageInfo RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(GetHKLListNF ${NF_SRCDIR}/GetHKLList.c ${NF_SG_LIB_SOURCES})
install(TARGETS GetHKLListNF RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(ParseMic ${NF_SRCDIR}/ParseMic.c)
install(TARGETS ParseMic RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(FitOrientationSinglePoint ${NF_SRCDIR}/FitOrientationSinglePoint.c ${NF_SHARED_FIT_SOURCES})
target_link_libraries(FitOrientationSinglePoint PRIVATE NLOPT::NLOPT)
install(TARGETS FitOrientationSinglePoint RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(ParseDeconvOutput ${NF_SRCDIR}/ParseDeconvOutput.c)
target_link_libraries(ParseDeconvOutput PRIVATE TIFF::TIFF)
install(TARGETS ParseDeconvOutput RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(compareNF ${NF_SRCDIR}/compareNF.c ${NF_SRCDIR}/CalcDiffractionSpots.c ${NF_SRCDIR}/SharedFuncsFit.c) # Renamed from CompareNF
target_link_libraries(compareNF PRIVATE NLOPT::NLOPT)
install(TARGETS compareNF RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(simulateNF ${NF_SRCDIR}/simulateNF.c ${NF_SRCDIR}/CalcDiffractionSpots.c ${NF_SRCDIR}/SharedFuncsFit.c) # Renamed from SimulateNF
install(TARGETS simulateNF RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

# --- OpenMP Executables ---
add_executable(ImageProcessingLibTiffOMP ${NF_SRCDIR}/ImageProcessingLibTiffOMP.c)
target_link_libraries(ImageProcessingLibTiffOMP PRIVATE TIFF::TIFF)
if(OpenMP_C_FOUND AND BUILD_OMP)
    target_link_libraries(ImageProcessingLibTiffOMP PRIVATE OpenMP::OpenMP_C)
endif()
install(TARGETS ImageProcessingLibTiffOMP RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

add_executable(FitOrientationOMP ${NF_SRCDIR}/FitOrientationOMP.c ${NF_SHARED_FIT_SOURCES})
target_link_libraries(FitOrientationOMP PRIVATE NLOPT::NLOPT)
if(OpenMP_C_FOUND AND BUILD_OMP)
    target_link_libraries(FitOrientationOMP PRIVATE OpenMP::OpenMP_C)
endif()
install(TARGETS FitOrientationOMP RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

# --- Shared Library ---
add_library(NFGrainsCalc SHARED ${NF_SRCDIR}/NFGrainsCalc.c)
# No specific dependencies mentioned for the .so other than CFLAGS in Makefile
# If it needs m or dl, those are handled by CMAKE_SHARED_LINKER_FLAGS from top level or CMAKE_C_FLAGS.
set_target_properties(NFGrainsCalc PROPERTIES POSITION_INDEPENDENT_CODE ON) # Standard for shared libs
install(TARGETS NFGrainsCalc
    LIBRARY DESTINATION ${MIDAS_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${MIDAS_INSTALL_LIBDIR} # For static lib part on some platforms
    RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR} # For DLLs on Windows
)

# --- HDF5 related executables (not in 'all' in Makefile, but present) ---
# These used H5CC. With CMake, we link against HDF5 target which provides flags.
# Uncomment and adjust if these are to be included.
# add_executable(MedianImageHDF ${NF_SRCDIR}/MedianImageHDF.c)
# target_link_libraries(MedianImageHDF PRIVATE HDF5::HDF5 HDF5::HL) # From COMMON_LINK_LIBRARIES
# if(OpenMP_C_FOUND AND BUILD_OMP) # Makefile showed -fopenmp
#     target_link_libraries(MedianImageHDF PRIVATE OpenMP::OpenMP_C)
# endif()
# install(TARGETS MedianImageHDF RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

# add_executable(ImageProcessingHDF ${NF_SRCDIR}/ImageProcessingHDF.c)
# target_link_libraries(ImageProcessingHDF PRIVATE HDF5::HDF5 HDF5::HL)
# if(OpenMP_C_FOUND AND BUILD_OMP) # Makefile showed -fopenmp
#     target_link_libraries(ImageProcessingHDF PRIVATE OpenMP::OpenMP_C)
# endif()
# install(TARGETS ImageProcessingHDF RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})


# Optional: Add a custom target to build all things in this subdirectory that are part of 'all'
add_custom_target(midas_nf_hedm_all DEPENDS
    MakeDiffrSpots
    MakeHexGrid
    MedianImageLibTiff
    ImageProcessingLibTiff
    FitOrientation
    FitOrientationParameters
    FitOrientationParametersMultiPoint
    GenSeedOrientationsFF2NFHEDM
    SimulateDiffractionSpots
    ProcessNFMicRemote
    filterGridfromTomo
    MMapImageInfo
    GetHKLList
    ParseMic
    FitOrientationSinglePoint
    ParseDeconvOutput
    NFGrainsCalc # Shared library
    compareNF
    simulateNF
    ImageProcessingLibTiffOMP
    FitOrientationOMP
)

# Note: CUDA targets are not explicitly defined as no .cu files or nvcc compilation rules
# were found in the 'all' target list of the Makefile. If C files need to be compiled with nvcc
# as the C compiler (via CMAKE_C_COMPILER set to nvcc), that would be a global setting.
# If specific files use CUDA, they should be identified and compiled with CUDA language if needed.
# The top-level USE_CUDA option enables CUDA language support project-wide if project(MIDAS LANGUAGES C CUDA) is used.
