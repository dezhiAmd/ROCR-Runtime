include_guard()

option(HSA_EXAMPLES_WERROR "Make all warnings into errors." ON)

# HSA examples add_executable
function(hsa_examples_add_executable TARGET)
  add_executable(${TARGET} ${ARGN})

  set_target_properties(
    ${TARGET}
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
               POSITION_INDEPENDENT_CODE ON)

  target_compile_options(
    ${TARGET}
    PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic
            $<$<BOOL:${HSA_EXAMPLES_WERROR}>:-Werror>>)

  target_link_libraries(${TARGET} PRIVATE hsa-runtime64::hsa-runtime64)
endfunction()

# HIP kernel compilation
function(hsa_examples_add_kernel TARGET SOURCE GPU_ARCH)
  get_filename_component(SOURCE_WE ${SOURCE} NAME_WE)

  set(ASSEMBLY_FILE "${SOURCE_WE}-hip-amdgcn-amd-amdhsa-${GPU_ARCH}.s")
  set(HSACO_FILE "${SOURCE_WE}.hsaco")

  add_custom_command(
    OUTPUT ${ASSEMBLY_FILE}
    COMMAND ${HIP_HIPCC_EXECUTABLE} -S --cuda-device-only
            --offload-arch=${GPU_ARCH} ${SOURCE} -o ${ASSEMBLY_FILE}
    VERBATIM)

  add_custom_target(${TARGET}-assembly DEPENDS ${ASSEMBLY_FILE})

  add_custom_command(
    OUTPUT ${HSACO_FILE}
    COMMAND ${CMAKE_HIP_COMPILER} -target amdgcn-amd-amdhsa -mcpu=${GPU_ARCH}
            ${ASSEMBLY_FILE} -o ${HSACO_FILE}
    DEPENDS ${TARGET}-assembly
    VERBATIM)

  add_custom_target(${TARGET} DEPENDS ${HSACO_FILE})

endfunction()
