cmake_minimum_required(VERSION 3.28)

project(emx_ort_runtime_extractor CXX)

if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

set(onnxruntime_BUILD_UNIT_TESTS ON CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(onnxruntime_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(onnxruntime_RUN_ONNX_TESTS OFF CACHE BOOL "" FORCE)
set(onnxruntime_USE_CUDA OFF CACHE BOOL "" FORCE)
set(onnxruntime_USE_DML OFF CACHE BOOL "" FORCE)
set(onnxruntime_USE_OPENVINO OFF CACHE BOOL "" FORCE)
set(onnxruntime_USE_TENSORRT OFF CACHE BOOL "" FORCE)
set(onnxruntime_USE_WEBGPU OFF CACHE BOOL "" FORCE)
set(onnxruntime_USE_XNNPACK OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "" FORCE)

set(EMX_ORT_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
get_filename_component(EMX_ORT_REPO_ROOT "${EMX_ORT_REPO_ROOT}" ABSOLUTE)

add_subdirectory("${EMX_ORT_REPO_ROOT}/onnxruntime-org/cmake" "${CMAKE_BINARY_DIR}/onnxruntime")

add_executable(
  ort_cpp_test_runtime_extractor
  "${EMX_ORT_REPO_ROOT}/cpp/runtime_extractor/ort_runtime_capture.cpp"
  "${EMX_ORT_REPO_ROOT}/cpp/runtime_extractor/ort_runtime_extractor.cpp"
)

target_compile_features(ort_cpp_test_runtime_extractor PRIVATE cxx_std_20)
target_include_directories(
  ort_cpp_test_runtime_extractor
  PRIVATE
    "${EMX_ORT_REPO_ROOT}"
    "${EMX_ORT_REPO_ROOT}/onnxruntime-org/onnxruntime"
    "${EMX_ORT_REPO_ROOT}/onnxruntime-org/include/onnxruntime"
    "${CMAKE_BINARY_DIR}/onnxruntime"
    "${CMAKE_BINARY_DIR}/generated"
)
target_compile_definitions(
  ort_cpp_test_runtime_extractor
  PRIVATE
    _USE_MATH_DEFINES
    ORT_ENABLE_STREAM
)

if(WIN32)
  target_compile_definitions(
    ort_cpp_test_runtime_extractor
    PRIVATE
      NOMINMAX
      WIN32_LEAN_AND_MEAN
  )
endif()

if(MSVC)
  target_compile_options(ort_cpp_test_runtime_extractor PRIVATE /MP /utf-8)
endif()

target_link_libraries(
  ort_cpp_test_runtime_extractor
  PRIVATE
    onnxruntime_unittest_utils
    onnxruntime_session
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_util
    onnxruntime_lora
    onnxruntime_framework
    onnxruntime_graph
    onnxruntime_mlas
    onnxruntime_common
    onnxruntime_flatbuffers
    GTest::gtest
    GTest::gmock
)
