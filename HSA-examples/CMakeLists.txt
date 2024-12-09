cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(
  HSA-examples
  VERSION 0.1.0
  DESCRIPTION "HSA heterogeneous tests"
  HOMEPAGE_URL "https://github.com/ypapadop-amd/HSA-examples"
  LANGUAGES CXX HIP)

set(ROCM_ROOT "/home/dezhliao/develop/ROCR-Install")
set(hip_DIR "/opt/rocm/lib/cmake/hip")


list(APPEND CMAKE_PREFIX_PATH ${ROCM_ROOT})

message(STATUS "CMAKE_PREFIX_PATH is set to ${CMAKE_PREFIX_PATH}")

find_package(hsa-runtime64 1.0 REQUIRED)
find_package(hip REQUIRED)

# CMake modules
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_LIST_DIR}/cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# HIP compiler support
if(NOT DEFINED ${HSA_EXAMPLES_GPU_ARCH})
  set(HSA_EXAMPLES_GPU_ARCH
      "gfx942"
      CACHE STRING "Target GPU architecture")
endif()

# sources
add_subdirectory(iterate_agents)
add_subdirectory(dispatch_multi_gpu)
