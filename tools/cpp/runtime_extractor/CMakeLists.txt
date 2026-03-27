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

set(EMX_ORT_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../..")
get_filename_component(EMX_ORT_REPO_ROOT "${EMX_ORT_REPO_ROOT}" ABSOLUTE)
if(NOT DEFINED EMX_ORT_SOURCE_ROOT)
  message(FATAL_ERROR "EMX_ORT_SOURCE_ROOT must point to a local ONNX Runtime source checkout.")
endif()
get_filename_component(EMX_ORT_SOURCE_ROOT "${EMX_ORT_SOURCE_ROOT}" ABSOLUTE)

add_subdirectory("${EMX_ORT_SOURCE_ROOT}/cmake" "${CMAKE_BINARY_DIR}/onnxruntime")

function(emx_apply_runtime_extractor_target_settings target_name)
  target_compile_features("${target_name}" PRIVATE cxx_std_20)
  target_include_directories(
      "${target_name}"
    PRIVATE
      "${EMX_ORT_REPO_ROOT}"
      "${EMX_ORT_SOURCE_ROOT}/onnxruntime"
      "${EMX_ORT_SOURCE_ROOT}/include/onnxruntime"
      "${CMAKE_BINARY_DIR}/onnxruntime"
      "${CMAKE_BINARY_DIR}/generated"
  )
  target_compile_definitions(
    "${target_name}"
    PRIVATE
      _USE_MATH_DEFINES
      ORT_ENABLE_STREAM
  )

  if(onnxruntime_DISABLE_RTTI)
    target_compile_definitions(
      "${target_name}"
      PRIVATE
        ORT_NO_RTTI
    )
    target_compile_options(
      "${target_name}"
      PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>"
    )
  endif()

  if(WIN32)
    target_compile_definitions(
      "${target_name}"
      PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
  endif()

  if(MSVC)
    target_compile_options("${target_name}" PRIVATE /MP /utf-8)
  endif()

  target_link_libraries(
    "${target_name}"
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
endfunction()

function(emx_add_runtime_extractor_target target_name source_file source_file_rel source_root_rel extra_includes_header)
  add_executable(
    "${target_name}"
    "${EMX_ORT_REPO_ROOT}/tools/cpp/runtime_extractor/ort_runtime_capture.cpp"
    "${EMX_ORT_REPO_ROOT}/tools/cpp/runtime_extractor/ort_runtime_extractor.cpp"
  )

  target_compile_definitions(
    "${target_name}"
    PRIVATE
      EMX_ORT_CAPTURE_TEST_SOURCE="${source_file}"
      EMX_ORT_CAPTURE_SOURCE_FILE_REL="${source_file_rel}"
      EMX_ORT_CAPTURE_SOURCE_ROOT_REL="${source_root_rel}"
      EMX_ORT_CAPTURE_EXTRA_INCLUDES_HEADER="${extra_includes_header}"
  )

  emx_apply_runtime_extractor_target_settings("${target_name}")
endfunction()

set(EMX_RUNTIME_TARGETS_FILE "${CMAKE_BINARY_DIR}/generated/emx_runtime_targets.cmake")
if(NOT EXISTS "${EMX_RUNTIME_TARGETS_FILE}")
  message(FATAL_ERROR "Missing generated runtime target manifest: ${EMX_RUNTIME_TARGETS_FILE}")
endif()

include("${EMX_RUNTIME_TARGETS_FILE}")
