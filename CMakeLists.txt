cmake_minimum_required(VERSION 3.27)
project(SDFoam VERSION 1.0.0)

cmake_policy(SET CMP0060 NEW)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GPU_DEBUG
    ON
    CACHE BOOL "Enable GPU debug features")
add_definitions(-DGPU_DEBUG=$<BOOL:${GPU_DEBUG}>)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "Build type")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external"
     "${CMAKE_SOURCE_DIR}/external/submodules")

set(Python3_FIND_VIRTUALENV FIRST)

find_package(
  Python3
  COMPONENTS Interpreter Development.Module
  REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE _pybind11_cmake_dir
  RESULT_VARIABLE _pybind11_cmake_result
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(_pybind11_cmake_result EQUAL 0
   AND EXISTS "${_pybind11_cmake_dir}/pybind11Config.cmake")
  set(pybind11_DIR
      "${_pybind11_cmake_dir}"
      CACHE PATH "Path to pybind11 CMake package" FORCE)
endif()

find_package(pybind11 REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import pathlib, torch; print(pathlib.Path(torch.__file__).parent / 'share' / 'cmake' / 'Torch')"
  OUTPUT_VARIABLE _torch_cmake_dir
  RESULT_VARIABLE _torch_cmake_result
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(_torch_cmake_result EQUAL 0
   AND EXISTS "${_torch_cmake_dir}/TorchConfig.cmake")
  set(Torch_DIR
      "${_torch_cmake_dir}"
      CACHE PATH "Path to Torch CMake package" FORCE)
endif()

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH
             "${TORCH_INSTALL_PREFIX}/lib")

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import pathlib, torch; print(pathlib.Path(torch.__file__).parent / 'include')"
  OUTPUT_VARIABLE _torch_include_dir
  RESULT_VARIABLE _torch_include_result
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(_torch_include_result EQUAL 0)
  set(_torch_cuda_macros_h "${_torch_include_dir}/c10/cuda/CUDAMacros.h")
  set(_torch_cuda_cmake_macros_h
      "${_torch_include_dir}/c10/cuda/impl/cuda_cmake_macros.h")
  if(EXISTS "${_torch_cuda_macros_h}"
     AND NOT EXISTS "${_torch_cuda_cmake_macros_h}")
    message(STATUS "Creating missing PyTorch CUDA CMake macros header")
    file(MAKE_DIRECTORY "${_torch_include_dir}/c10/cuda/impl")
    file(
      WRITE "${_torch_cuda_cmake_macros_h}"
      "#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)\n"
      "#pragma once\n\n"
      "#define C10_CUDA_BUILD_SHARED_LIBS\n\n"
      "#else\n"
      "#error \"This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined.\"\n"
      "#endif\n")
  endif()
endif()

option(SDFOAM_UPDATE_SUBMODULES
       "Automatically initialize missing git submodules during CMake configure"
       ON)

set(SDFOAM_REQUIRED_SUBMODULE_FILES
    external/submodules/glfw/CMakeLists.txt
    external/submodules/tbb/CMakeLists.txt
    external/submodules/imgui/imgui.cpp
    external/submodules/eigen/Eigen/Core
    external/submodules/atomic_queue/include/atomic_queue/atomic_queue.h
    external/submodules/mesa/include/GL/gl.h)

set(_sdfoam_missing_submodules FALSE)
foreach(_sdfoam_submodule_file IN LISTS SDFOAM_REQUIRED_SUBMODULE_FILES)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/${_sdfoam_submodule_file}")
    set(_sdfoam_missing_submodules TRUE)
  endif()
endforeach()

if(_sdfoam_missing_submodules)
  if(SDFOAM_UPDATE_SUBMODULES)
    find_package(Git QUIET)
    if(NOT Git_FOUND)
      message(
        FATAL_ERROR
          "SDFoam git submodules are missing and Git was not found. Run "
          "`git submodule update --init --recursive` from ${CMAKE_SOURCE_DIR}.")
    endif()

    message(STATUS "SDFoam submodules are missing; initializing them with Git")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      RESULT_VARIABLE _sdfoam_submodule_result)
    if(NOT _sdfoam_submodule_result EQUAL 0)
      message(
        FATAL_ERROR
          "Failed to initialize SDFoam submodules. Run "
          "`git submodule update --init --recursive` from ${CMAKE_SOURCE_DIR}.")
    endif()
  else()
    message(
      FATAL_ERROR
        "SDFoam git submodules are missing. Run "
        "`git submodule update --init --recursive` from ${CMAKE_SOURCE_DIR}.")
  endif()
endif()

set(_sdfoam_eigen_matrix_h
    "${CMAKE_SOURCE_DIR}/external/submodules/eigen/Eigen/src/Core/Matrix.h")
if(EXISTS "${_sdfoam_eigen_matrix_h}")
  file(READ "${_sdfoam_eigen_matrix_h}" _sdfoam_eigen_matrix)
  set(_sdfoam_eigen_matrix_patched "${_sdfoam_eigen_matrix}")
  string(REPLACE "EIGEN_DEVICE_FUNC explicit constexpr EIGEN_STRONG_INLINE Matrix("
                 "EIGEN_DEVICE_FUNC explicit EIGEN_STRONG_INLINE Matrix("
                 _sdfoam_eigen_matrix_patched "${_sdfoam_eigen_matrix_patched}")
  string(REPLACE "EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE explicit Matrix(const T& x)"
                 "EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE explicit Matrix(const T& x)"
                 _sdfoam_eigen_matrix_patched "${_sdfoam_eigen_matrix_patched}")
  string(REPLACE "EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Matrix(const T0& x, const T1& y)"
                 "EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Matrix(const T0& x, const T1& y)"
                 _sdfoam_eigen_matrix_patched "${_sdfoam_eigen_matrix_patched}")
  string(REPLACE "EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Matrix(const Scalar& x, const Scalar& y, const Scalar& z)"
                 "EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Matrix(const Scalar& x, const Scalar& y, const Scalar& z)"
                 _sdfoam_eigen_matrix_patched "${_sdfoam_eigen_matrix_patched}")
  string(REPLACE "EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Matrix(const Scalar& x, const Scalar& y, const Scalar& z,"
                 "EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Matrix(const Scalar& x, const Scalar& y, const Scalar& z,"
                 _sdfoam_eigen_matrix_patched "${_sdfoam_eigen_matrix_patched}")
  if(NOT _sdfoam_eigen_matrix STREQUAL _sdfoam_eigen_matrix_patched)
    message(STATUS "Patching Eigen Matrix constructors for CUDA/MSVC compatibility")
    file(WRITE "${_sdfoam_eigen_matrix_h}" "${_sdfoam_eigen_matrix_patched}")
  endif()
endif()

add_subdirectory(external)
include_directories(${SDFOAM_EXTERNAL_INCLUDES})

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX
      "${CMAKE_SOURCE_DIR}/sdfoam"
      CACHE PATH "..." FORCE)
endif()

set(SDFOAM_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

add_subdirectory(src)
add_subdirectory(torch_bindings)
