// Runtime extractor entry point. This translation unit compiles a selected ORT
// test source into a standalone runner and redirects OpTester usage through a
// capturing subclass that exports JSON metadata.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#ifdef ERROR
#undef ERROR
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

#ifdef USE_WEBGPU
#undef USE_WEBGPU
#endif

#include <google/protobuf/message_lite.h>

#include "core/session/onnxruntime_cxx_api.h"
#include "gtest/gtest.h"

#include "cpp/runtime_extractor/ort_runtime_capture.h"

#ifndef EMX_ORT_CAPTURE_TEST_SOURCE
#error "EMX_ORT_CAPTURE_TEST_SOURCE must be defined for the runtime extractor target."
#endif

#ifndef EMX_ORT_CAPTURE_SOURCE_ROOT_REL
#error "EMX_ORT_CAPTURE_SOURCE_ROOT_REL must be defined for the runtime extractor target."
#endif

#ifndef EMX_ORT_CAPTURE_EXTRA_INCLUDES_HEADER
#define EMX_ORT_CAPTURE_EXTRA_INCLUDES_HEADER "cpp/runtime_extractor/ort_runtime_capture_extra_includes_default.h"
#endif

#include EMX_ORT_CAPTURE_EXTRA_INCLUDES_HEADER

std::unique_ptr<Ort::Env> ort_env;

namespace emx::ort_runtime {

void SetupOrtEnvironment() {
  OrtThreadingOptions threading_options;
  ort_env = std::make_unique<Ort::Env>(&threading_options, ORT_LOGGING_LEVEL_WARNING, "emx_ort_runtime_extractor");
}

void TearDownOrtEnvironment() {
  ort_env.reset();
  ::google::protobuf::ShutdownProtobufLibrary();
}

std::vector<char*> StripCustomArgs(
    int argc,
    char** argv,
    std::filesystem::path& output_path,
    std::filesystem::path& artifact_root) {
  std::vector<char*> forwarded_args;
  forwarded_args.reserve(static_cast<size_t>(argc));
  forwarded_args.push_back(argv[0]);

  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--emx_output_json" && index + 1 < argc) {
      output_path = argv[++index];
      continue;
    }

    if (argument == "--emx_artifact_root" && index + 1 < argc) {
      artifact_root = argv[++index];
      continue;
    }

    forwarded_args.push_back(argv[index]);
  }

  return forwarded_args;
}

}  // namespace emx::ort_runtime

namespace onnxruntime::test {
using EMX_CAPTURE_ORT_OPTESTER = ::emx::ort_runtime::CapturingOpTester;
}  // namespace onnxruntime::test

#define OpTester EMX_CAPTURE_ORT_OPTESTER
#include EMX_ORT_CAPTURE_TEST_SOURCE
#undef OpTester

int main(int argc, char** argv) {
  int status = 0;
  std::filesystem::path output_path;
  std::filesystem::path artifact_root;

  try {
    std::vector<char*> forwarded_args = emx::ort_runtime::StripCustomArgs(argc, argv, output_path, artifact_root);
    int forwarded_argc = static_cast<int>(forwarded_args.size());
    emx::ort_runtime::CaptureCollector::Instance().Reset(
        EMX_ORT_CAPTURE_SOURCE_ROOT_REL,
        artifact_root);

    emx::ort_runtime::SetupOrtEnvironment();
    ::testing::InitGoogleTest(&forwarded_argc, forwarded_args.data());
    status = RUN_ALL_TESTS();

    if (!output_path.empty()) {
      emx::ort_runtime::CaptureCollector::Instance().WriteJson(output_path);
      std::cout << "Wrote " << emx::ort_runtime::CaptureCollector::Instance().RecordCount()
                << " runtime records to " << std::filesystem::absolute(output_path).string() << "\n";
    }

    emx::ort_runtime::TearDownOrtEnvironment();
  } catch (const std::exception& exception) {
    std::cerr << "ort_cpp_test_runtime_extractor failed: " << exception.what() << "\n";
    status = 1;
  }

  return status;
}
