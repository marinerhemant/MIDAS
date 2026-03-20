# CMakeLists.txt for TOMO component of MIDAS

set(TOMO_SRCDIR src)

set(TOMO_SOURCES
    ${TOMO_SRCDIR}/tomo_init.c
    ${TOMO_SRCDIR}/tomo_gridrec.c
    ${TOMO_SRCDIR}/tomo_utils.c
    ${TOMO_SRCDIR}/tomo_cleanup.c
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

# --- GPU-accelerated target: MIDAS_TOMO_GPU (requires USE_CUDA=ON) ---
if(USE_CUDA AND CUDA_FOUND)
    set(TOMO_GPU_SOURCES
        ${TOMO_SRCDIR}/tomo_init.c
        ${TOMO_SRCDIR}/tomo_gridrec.c
        ${TOMO_SRCDIR}/tomo_utils.c
        ${TOMO_SRCDIR}/tomo_cleanup.c
        ${TOMO_SRCDIR}/tomo_gpu.cu
    )

    add_executable(MIDAS_TOMO_GPU ${TOMO_GPU_SOURCES})
    target_link_libraries(MIDAS_TOMO_GPU PRIVATE ${COMMON_LINK_LIBRARIES})
    target_link_libraries(MIDAS_TOMO_GPU PRIVATE cufft cudart)
    target_compile_definitions(MIDAS_TOMO_GPU PRIVATE PI=M_PI ENABLE_CUDA)

    if(OpenMP_C_FOUND AND BUILD_OMP)
        target_link_libraries(MIDAS_TOMO_GPU PRIVATE OpenMP::OpenMP_C)
        # nvcc needs explicit -fopenmp passed to host compiler for .cu files
        target_compile_options(MIDAS_TOMO_GPU PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
        )
    endif()

    # Set CUDA properties
    set_target_properties(MIDAS_TOMO_GPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    install(TARGETS MIDAS_TOMO_GPU
        RUNTIME DESTINATION ${MIDAS_INSTALL_BINDIR}
    )

    add_custom_command(TARGET MIDAS_TOMO_GPU POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:MIDAS_TOMO_GPU> "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
        COMMENT "Copied MIDAS_TOMO_GPU to ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
        VERBATIM
    )

    message(STATUS "TOMO: GPU target MIDAS_TOMO_GPU enabled")
else()
    message(STATUS "TOMO: GPU target MIDAS_TOMO_GPU skipped (USE_CUDA=${USE_CUDA})")
endif()

# Optional: Add a custom target to build all executables in this subdirectory
set(TOMO_ALL_TARGETS MIDAS_TOMO)
if(TARGET MIDAS_TOMO_GPU)
    list(APPEND TOMO_ALL_TARGETS MIDAS_TOMO_GPU)
endif()
add_custom_target(midas_tomo_all_executables DEPENDS
    ${TOMO_ALL_TARGETS}
)