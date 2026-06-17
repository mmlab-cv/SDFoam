find_package(CUDAToolkit REQUIRED)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -D_USE_MATH_DEFINES)

list(
  APPEND
  SDFOAM_INCLUDES
  ${CMAKE_SOURCE_DIR}/external/submodules/termcolor/include
  ${CMAKE_SOURCE_DIR}/external/submodules/eigen
  ${CMAKE_SOURCE_DIR}/external/submodules/atomic_queue/include
  ${GLFW_INCLUDES})

list(APPEND SDFOAM_CXX_SOURCES viewer/viewer.cpp utils/batch_fetcher.cpp)
list(
  APPEND
  SDFOAM_CUDA_SOURCES
  aabb_tree/aabb_tree.cu
  delaunay/delaunay.cu
  delaunay/sample_initial_tets.cu
  delaunay/growth_iteration.cu
  delaunay/delete_violations.cu
  delaunay/triangulation_ops.cu
  tracing/pipeline.cu)

add_library(sdfoam STATIC ${SDFOAM_CXX_SOURCES} ${SDFOAM_CUDA_SOURCES})
target_include_directories(sdfoam PUBLIC ${SDFOAM_INCLUDES}
                                         ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(sdfoam PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_definitions(sdfoam PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:EIGEN_HAS_CONSTEXPR=0>)
target_link_libraries(
  sdfoam PUBLIC CUDA::cudart CUDA::cuda_driver ${GLFW_LIBRARY} gl3w imgui TBB::tbb)
if(MSVC)
  target_compile_options(
    sdfoam PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --extended-lambda --expt-relaxed-constexpr -Xcompiler=/O2 -Xcompiler=/W3 -Xcompiler=/EHsc -Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /W3 /EHsc /Zc:preprocessor>)
else()
  target_compile_options(
    sdfoam PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --extended-lambda --expt-relaxed-constexpr -O3 -Wall -Wno-unknown-pragmas> -O3)
endif()
