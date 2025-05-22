# CMakeLists.txt for DT component of MIDAS

# Ensure sources are relative to the current directory (DT/)
set(DT_SRCDIR src)

# --- Executable: DetectorMapperDT ---
add_executable(DetectorMapperDT ${DT_SRCDIR}/DetectorMapper.c)
# Link common libraries, including math library (m) and others from the top-level.
target_link_libraries(DetectorMapperDT PRIVATE ${COMMON_LINK_LIBRARIES})

install(TARGETS DetectorMapperDT
    RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR}
)

# --- Executable: IntegratorPeakFitOMP ---
add_executable(IntegratorPeakFitOMP ${DT_SRCDIR}/IntegratorPeakFitOMP.c)

# Dependencies: OpenMP, and others from COMMON_LINK_LIBRARIES (e.g., NLOPT, TIFF, m, dl)
target_link_libraries(IntegratorPeakFitOMP PRIVATE ${COMMON_LINK_LIBRARIES})

# Enable OpenMP if available
if(OpenMP_C_FOUND AND BUILD_OMP)
    target_link_libraries(IntegratorPeakFitOMP PRIVATE OpenMP::OpenMP_C)
endif()

install(TARGETS IntegratorPeakFitOMP
    RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR}
)

# --- Executable: PeakFit ---
add_executable(PeakFit ${DT_SRCDIR}/PeakFit.c)

# Dependencies: OpenMP, and others from COMMON_LINK_LIBRARIES
target_link_libraries(PeakFit PRIVATE ${COMMON_LINK_LIBRARIES})

# Enable OpenMP if available
if(OpenMP_C_FOUND AND BUILD_OMP)
    target_link_libraries(PeakFit PRIVATE OpenMP::OpenMP_C)
endif()

install(TARGETS PeakFit
    RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR}
)

# Optional: Add a custom target to build all executables in this subdirectory
add_custom_target(midas_dt_all_executables DEPENDS
    DetectorMapperDT # Corrected from DetectorMapper
    IntegratorPeakFitOMP
    PeakFit
)