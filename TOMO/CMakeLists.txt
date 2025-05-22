# CMakeLists.txt for TOMO component of MIDAS

set(TOMO_SRCDIR src)

set(TOMO_SOURCES
    ${TOMO_SRCDIR}/tomo_init.c
    ${TOMO_SRCDIR}/tomo_gridrec.c
    ${TOMO_SRCDIR}/tomo_utils.c
)

# --- Executable: MIDAS_TOMO ---
add_executable(MIDAS_TOMO ${TOMO_SOURCES})
target_link_libraries(MIDAS_TOMO PRIVATE ${COMMON_LINK_LIBRARIES})
target_compile_definitions(MIDAS_TOMO PRIVATE PI=M_PI)

if(OpenMP_C_FOUND AND BUILD_OMP)
    target_link_libraries(MIDAS_TOMO PRIVATE OpenMP::OpenMP_C)
endif()

install(TARGETS MIDAS_TOMO
    RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR}
)

add_custom_command(TARGET MIDAS_TOMO POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:MIDAS_TOMO> "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
  COMMENT "Copied MIDAS_TOMO to ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
  VERBATIM
)

# Optional: Add a custom target to build all executables in this subdirectory
add_custom_target(midas_tomo_all_executables DEPENDS
    MIDAS_TOMO
)