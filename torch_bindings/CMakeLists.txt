execute_process(
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/torch_info.py torch
  OUTPUT_VARIABLE TORCH_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/torch_info.py cuda
  OUTPUT_VARIABLE TORCH_CUDA_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(TORCH_CUDA_VERSION VERSION_EQUAL CUDA_VERSION_STRING)
  message(STATUS "CUDA version is: ${CUDA_VERSION_STRING}")
else()
  message(
    FATAL_ERROR
      "CUDA version found (${CUDA_VERSION_STRING}) does not match the version required by PyTorch (${TORCH_CUDA_VERSION})."
  )
endif()

pybind11_add_module(torch_bindings MODULE torch_bindings.cpp
                    pipeline_bindings.cpp triangulation_bindings.cpp)

target_include_directories(torch_bindings PUBLIC ${CMAKE_SOURCE_DIR}/src)

find_library(TORCH_CUDA_LIBRARY torch_cuda PATHS "${TORCH_INSTALL_PREFIX}/lib")
find_library(C10_CUDA_LIBRARY c10_cuda PATHS "${TORCH_INSTALL_PREFIX}/lib")

set(SDFOAM_TORCH_CUDA_LIBRARIES)
if(TORCH_CUDA_LIBRARY)
  list(APPEND SDFOAM_TORCH_CUDA_LIBRARIES ${TORCH_CUDA_LIBRARY})
endif()
if(C10_CUDA_LIBRARY)
  list(APPEND SDFOAM_TORCH_CUDA_LIBRARIES ${C10_CUDA_LIBRARY})
else()
  message(
    FATAL_ERROR
      "Could not find c10_cuda library in ${TORCH_INSTALL_PREFIX}/lib. "
      "Please install a CUDA-enabled PyTorch wheel.")
endif()

target_link_libraries(torch_bindings PRIVATE torch ${TORCH_PYTHON_LIBRARY}
                                             ${SDFOAM_TORCH_CUDA_LIBRARIES} sdfoam
                                             ${GLFW_LIBRARY})

install(
  TARGETS torch_bindings
  COMPONENT torch_bindings
  LIBRARY DESTINATION ${SDFOAM_INSTALL_PREFIX}
  ARCHIVE DESTINATION ${SDFOAM_INSTALL_PREFIX}
  RUNTIME DESTINATION ${SDFOAM_INSTALL_PREFIX})

configure_file(${CMAKE_SOURCE_DIR}/torch_bindings/sdfoam/__init__.py.in
               ${CMAKE_BINARY_DIR}/__init__.py @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/__init__.py
        DESTINATION ${SDFOAM_INSTALL_PREFIX})
