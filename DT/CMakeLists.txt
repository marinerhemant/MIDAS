# CMakeLists.txt for DT component of MIDAS

# Ensure sources are relative to the current directory (DT/)
set(DT_SRCDIR src)

# --- Executable: DetectorMapper ---
add_executable(DetectorMapper ${DT_SRCDIR}/DetectorMapper.c)
# No specific dependencies from Makefile other than standard CFLAGS
# Link common libraries if any are universally needed, or keep minimal.
# For now, assume it doesn't need the COMMON_LINK_LIBRARIES from the top-level.
# If it needs m or dl, those are handled by CMAKE_EXE_LINKER_FLAGS from top level.

install(TARGETS DetectorMapper
    RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR}
)

# --- Executable: IntegratorPeakFitOMP ---
add_executable(IntegratorPeakFitOMP ${DT_SRCDIR}/IntegratorPeakFitOMP.c)

# Dependencies: OpenMP, NLOPT, TIFF
target_link_libraries(IntegratorPeakFitOMP PRIVATE
    ${COMMON_LINK_LIBRARIES} # This includes NLOPT::NLOPT, TIFF::TIFF, etc.
)

# Enable OpenMP if available
if(OpenMP_C_FOUND AND BUILD_OMP)
    target_link_libraries(IntegratorPeakFitOMP PRIVATE OpenMP::OpenMP_C)
endif()

install(TARGETS IntegratorPeakFitOMP
    RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR}
)

# --- Executable: PeakFit ---
add_executable(PeakFit ${DT_SRCDIR}/PeakFit.c)

# Dependencies: OpenMP, NLOPT, TIFF
target_link_libraries(PeakFit PRIVATE
    ${COMMON_LINK_LIBRARIES}
)

# Enable OpenMP if available
if(OpenMP_C_FOUND AND BUILD_OMP)
    target_link_libraries(PeakFit PRIVATE OpenMP::OpenMP_C)
endif()

install(TARGETS PeakFit
    RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR}
)

# Optional: Add a custom target to build all executables in this subdirectory
add_custom_target(midas_dt_all_executables DEPENDS
    DetectorMapper
    IntegratorPeakFitOMP
    PeakFit
)
