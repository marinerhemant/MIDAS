# CMakeLists.txt for TOMO component of MIDAS

set(TOMO_SRCDIR src)

set(TOMO_SOURCES
    ${TOMO_SRCDIR}/tomo_init.c
    ${TOMO_SRCDIR}/tomo_gridrec.c
    ${TOMO_SRCDIR}/tomo_utils.c
)

# --- Executable: MIDAS_TOMO ---
# The Makefile had two targets, 'tomo' and 'tomodebug', differing mainly by CFLAGS vs CDFLAGS (-g).
# CMake handles Debug/Release flags via CMAKE_BUILD_TYPE.
# We will define one target and let the build type manage debug symbols.
add_executable(MIDAS_TOMO ${TOMO_SOURCES})

# Dependencies: FFTW, OpenMP, and standard libraries (like math lib 'm').
# FFTW::FFTW, m, dl etc. should be in COMMON_LINK_LIBRARIES from the top-level.
target_link_libraries(MIDAS_TOMO PRIVATE ${COMMON_LINK_LIBRARIES}) # For FFTW and other common dependencies

# Add compile definition -DPI=M_PI
# This ensures 'PI' macro is available, expanding to M_PI (from math.h)
target_compile_definitions(MIDAS_TOMO PRIVATE PI=M_PI)

# Enable OpenMP if available
if(OpenMP_C_FOUND AND BUILD_OMP)
    target_link_libraries(MIDAS_TOMO PRIVATE OpenMP::OpenMP_C)
endif()

install(TARGETS MIDAS_TOMO
    RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR}
)

# Optional: Add a custom target to build all executables in this subdirectory
add_custom_target(midas_tomo_all_executables DEPENDS
    MIDAS_TOMO
)