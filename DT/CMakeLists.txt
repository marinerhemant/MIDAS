# CMakeLists.txt for DT component of MIDAS

set(DT_SRCDIR src)

# Helper function to add executable and the custom command to copy to source/bin
function(add_dt_executable TARGET_NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SOURCES LINK_LIBRARIES_EXTRA)
    cmake_parse_arguments(DT_EXEC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_executable(${TARGET_NAME} ${DT_EXEC_SOURCES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${COMMON_LINK_LIBRARIES} ${DT_EXEC_LINK_LIBRARIES_EXTRA})
    install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
      COMMENT "Copied ${TARGET_NAME} to ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
      VERBATIM
    )
endfunction()

# --- Executable: DetectorMapperDT ---
add_dt_executable(DetectorMapperDT SOURCES ${DT_SRCDIR}/DetectorMapper.c)

# --- Executable: IntegratorPeakFitOMP ---
add_executable(IntegratorPeakFitOMP ${DT_SRCDIR}/IntegratorPeakFitOMP.c)
target_link_libraries(IntegratorPeakFitOMP PRIVATE ${COMMON_LINK_LIBRARIES})
if(OpenMP_C_FOUND AND BUILD_OMP)
    target_link_libraries(IntegratorPeakFitOMP PRIVATE OpenMP::OpenMP_C)
endif()
install(TARGETS IntegratorPeakFitOMP RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})
add_custom_command(TARGET IntegratorPeakFitOMP POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:IntegratorPeakFitOMP> "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
    COMMENT "Copied IntegratorPeakFitOMP to ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
    VERBATIM
)

# --- Executable: PeakFit ---
add_executable(PeakFit ${DT_SRCDIR}/PeakFit.c)
target_link_libraries(PeakFit PRIVATE ${COMMON_LINK_LIBRARIES})
if(OpenMP_C_FOUND AND BUILD_OMP)
    target_link_libraries(PeakFit PRIVATE OpenMP::OpenMP_C)
endif()
install(TARGETS PeakFit RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR})
add_custom_command(TARGET PeakFit POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:PeakFit> "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
    COMMENT "Copied PeakFit to ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
    VERBATIM
)

# Optional: Add a custom target to build all executables in this subdirectory
add_custom_target(midas_dt_all_executables DEPENDS
    DetectorMapperDT
    IntegratorPeakFitOMP
    PeakFit
)