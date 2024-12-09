include(hsa_support)

find_package(Boost REQUIRED COMPONENTS program_options)

hsa_examples_add_executable(dispatch_multi_gpu dispatch_multi_gpu.cpp)
hsa_examples_add_kernel(
  add_one_kernel ${CMAKE_CURRENT_SOURCE_DIR}/add_one_kernel.cpp
  ${HSA_EXAMPLES_GPU_ARCH})
add_dependencies(dispatch_multi_gpu add_one_kernel)
target_link_libraries(dispatch_multi_gpu PRIVATE Boost::program_options)
