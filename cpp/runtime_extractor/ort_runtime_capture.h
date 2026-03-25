// Runtime capture helpers for executing ORT OpTester-based unit tests and
// exporting reusable JSON metadata plus ONNX/TensorProto artifacts.

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "test/unittest_util/op_tester.h"

namespace emx::ort_runtime {

struct CapturedTensor {
  std::string name;
  std::string value_kind;
  int element_type = 0;
  std::vector<int64_t> shape;
  bool is_initializer = false;
  bool has_data = false;
  std::optional<double> relative_error;
  std::optional<double> absolute_error;
  bool sort_output = false;
  std::string data_encoding;
  std::string data_base64;
  std::vector<std::string> string_data;
  std::vector<std::string> warnings;
};

struct CapturedAttribute {
  std::string name;
  std::string value_kind;
  int tensor_element_type = 0;
  std::vector<int64_t> tensor_shape;
  std::string tensor_data_encoding;
  std::string tensor_data_base64;
  std::vector<std::string> tensor_string_data;
  std::optional<int64_t> int_value;
  std::optional<double> float_value;
  std::optional<std::string> string_value;
  std::vector<int64_t> ints_value;
  std::vector<double> floats_value;
  std::vector<std::string> strings_value;
  std::vector<std::string> warnings;
};

struct CapturedRecord {
  std::string source_file;
  std::string test_suite;
  std::string test_name;
  int run_index = 0;
  std::string op_name;
  int64_t opset = 0;
  std::string domain;
  bool expects_failure = false;
  std::string expected_failure_substring;
  std::vector<std::string> included_providers;
  std::vector<std::string> excluded_providers;
  bool saw_run_call = false;
  int node_count = 0;
  std::string artifact_directory;
  std::string model_path;
  std::vector<std::string> input_files;
  std::vector<std::string> output_files;
  std::vector<CapturedTensor> inputs;
  std::vector<CapturedTensor> outputs;
  std::vector<CapturedAttribute> attributes;
  std::vector<std::string> warnings;
};

class CaptureCollector {
 public:
  static CaptureCollector& Instance();

  void Reset(std::string source_root_relative, std::string source_file_relative, std::filesystem::path artifact_root);
  void AddRecord(CapturedRecord record);
  int AllocateRunIndex(std::string_view test_suite, std::string_view test_name);
  size_t RecordCount() const noexcept;
  void WriteJson(const std::filesystem::path& output_path) const;
  const std::filesystem::path& ArtifactRoot() const noexcept;

 private:
  std::string source_root_relative_;
  std::string source_file_relative_;
  std::filesystem::path artifact_root_;
  std::vector<CapturedRecord> records_;
  std::unordered_map<std::string, int> next_run_index_by_test_case_;
};

class CapturingOpTester : public onnxruntime::test::OpTester {
 public:
  using onnxruntime::test::OpTester::OpTester;
  using ExpectResult = onnxruntime::test::BaseTester::ExpectResult;
  using onnxruntime::test::OpTester::Run;
  using onnxruntime::test::OpTester::RunWithConfig;

  ~CapturingOpTester() override;

  void Run(ExpectResult expect_result = ExpectResult::kExpectSuccess,
           const std::string& expected_failure_string = "",
           const std::unordered_set<std::string>& excluded_provider_types = {},
           const onnxruntime::RunOptions* run_options = nullptr,
           std::vector<std::unique_ptr<onnxruntime::IExecutionProvider>>* execution_providers = nullptr,
           ExecutionMode execution_mode = ExecutionMode::ORT_SEQUENTIAL,
           const onnxruntime::Graph::ResolveOptions& resolve_options = {});

  void Run(onnxruntime::SessionOptions session_options,
           ExpectResult expect_result = ExpectResult::kExpectSuccess,
           const std::string& expected_failure_string = "",
           const std::unordered_set<std::string>& excluded_provider_types = {},
           const onnxruntime::RunOptions* run_options = nullptr,
           std::vector<std::unique_ptr<onnxruntime::IExecutionProvider>>* execution_providers = nullptr,
           const onnxruntime::Graph::ResolveOptions& resolve_options = {},
           size_t* number_of_pre_packed_weights_counter = nullptr,
           size_t* number_of_shared_pre_packed_weights_counter = nullptr);

  void RunWithConfig(size_t* number_of_pre_packed_weights_counter = nullptr,
                     size_t* number_of_shared_pre_packed_weights_counter = nullptr);

  const std::string& CapturedOpName() const noexcept { return Op(); }
  const std::string& CapturedDomain() const noexcept { return Domain(); }
  int CapturedOpset() const noexcept { return Opset(); }
  std::vector<onnxruntime::test::BaseTester::Data>& CapturedInputData() { return GetInputData(); }
  std::vector<onnxruntime::test::BaseTester::Data>& CapturedOutputData() { return GetOutputData(); }
  std::vector<size_t>& CapturedInitializerIndexes() { return GetInitializerIndexes(); }
  std::vector<std::string> CapturedConfiguredExecutionProviders() const;
  std::vector<std::string> CapturedExcludedProviderTypes() const;

 private:
  void CaptureSnapshot(
      bool saw_run_call,
      ExpectResult expect_result = ExpectResult::kExpectSuccess,
      std::string expected_failure_string = {},
      const std::unordered_set<std::string>* excluded_provider_types = nullptr,
      const std::vector<std::unique_ptr<onnxruntime::IExecutionProvider>>* execution_providers = nullptr);

  bool has_captured_snapshot_ = false;
};

}  // namespace emx::ort_runtime
