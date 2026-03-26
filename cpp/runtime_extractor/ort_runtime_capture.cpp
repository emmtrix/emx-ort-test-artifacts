// Runtime capture helpers for executing ORT OpTester-based unit tests and
// exporting reusable JSON metadata plus ONNX/TensorProto artifacts.

#include "cpp/runtime_extractor/ort_runtime_capture.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <optional>
#include <ostream>
#include <set>
#include <map>
#include <unordered_set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <google/protobuf/message_lite.h>
#include "gtest/gtest.h"

#include "core/framework/tensorprotoutils.h"
#include "emx_runtime_capture_config.h"

namespace fs = std::filesystem;

namespace emx::ort_runtime {
namespace {

bool IsQuantizeLstmReferenceHelper(const CapturingOpTester& tester) {
  constexpr std::string_view kQuantizeLstmSource =
      "onnxruntime/test/contrib_ops/quantize_lstm_op_test.cc";
  return std::string_view(EMX_ORT_CAPTURE_SOURCE_FILE_REL) == kQuantizeLstmSource &&
         tester.CapturedOpName() == "LSTM" &&
         tester.CapturedDomain() == onnxruntime::kOnnxDomain;
}

std::string JsonEscape(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);

  for (const unsigned char c : value) {
    switch (c) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (c >= 0x20 && c <= 0x7E) {
          escaped.push_back(static_cast<char>(c));
        } else {
          constexpr char kHexDigits[] = "0123456789ABCDEF";
          escaped += "\\u00";
          escaped.push_back(kHexDigits[(c >> 4) & 0x0F]);
          escaped.push_back(kHexDigits[c & 0x0F]);
        }
        break;
    }
  }

  return escaped;
}

std::string NormalizeProviderName(std::string_view provider_type) {
  static const std::map<std::string, std::string, std::less<>> kProviderAliases = {
      {"CPUExecutionProvider", "CPU"},
      {"CUDAExecutionProvider", "CUDA"},
      {"DmlExecutionProvider", "DML"},
      {"TensorrtExecutionProvider", "TensorRT"},
      {"ROCMExecutionProvider", "ROCM"},
      {"OpenVINOExecutionProvider", "OpenVINO"},
      {"CoreMLExecutionProvider", "CoreML"},
      {"XnnpackExecutionProvider", "XNNPACK"},
      {"QNNExecutionProvider", "QNN"},
      {"WebGpuExecutionProvider", "WebGPU"},
      {"MIGraphXExecutionProvider", "MIGraphX"},
      {"NnapiExecutionProvider", "NNAPI"},
      {"RknpuExecutionProvider", "RKNPU"},
      {"VitisAIExecutionProvider", "VitisAI"},
      {"JsExecutionProvider", "JS"},
      {"AzureExecutionProvider", "Azure"},
      {"CannExecutionProvider", "CANN"},
      {"ArmNNExecutionProvider", "ArmNN"},
      {"AclExecutionProvider", "ACL"},
      {"SystolicExecutionProvider", "Systolic"},
      {"SNPEExecutionProvider", "SNPE"},
      {"MsnpuExecutionProvider", "MSNPU"},
  };

  if (const auto it = kProviderAliases.find(provider_type); it != kProviderAliases.end()) {
    return it->second;
  }

  constexpr std::string_view suffix = "ExecutionProvider";
  if (provider_type.size() > suffix.size() &&
      provider_type.substr(provider_type.size() - suffix.size()) == suffix) {
    return std::string(provider_type.substr(0, provider_type.size() - suffix.size()));
  }

  return std::string(provider_type);
}

void WriteIndent(std::ostream& out, int indent) {
  for (int i = 0; i < indent; ++i) {
    out.put(' ');
  }
}

void WriteStringArray(std::ostream& out, const std::vector<std::string>& values, int indent) {
  out << "[";
  if (!values.empty()) {
    out << "\n";
  }

  for (size_t index = 0; index < values.size(); ++index) {
    WriteIndent(out, indent + 2);
    out << "\"" << JsonEscape(values[index]) << "\"";
    if (index + 1 < values.size()) {
      out << ",";
    }
    out << "\n";
  }

  if (!values.empty()) {
    WriteIndent(out, indent);
  }
  out << "]";
}

void WriteIntArray(std::ostream& out, const std::vector<int64_t>& values, int indent) {
  out << "[";
  if (!values.empty()) {
    out << "\n";
  }

  for (size_t index = 0; index < values.size(); ++index) {
    WriteIndent(out, indent + 2);
    out << values[index];
    if (index + 1 < values.size()) {
      out << ",";
    }
    out << "\n";
  }

  if (!values.empty()) {
    WriteIndent(out, indent);
  }
  out << "]";
}

void WriteDoubleArray(std::ostream& out, const std::vector<double>& values, int indent) {
  out << "[";
  if (!values.empty()) {
    out << "\n";
  }

  for (size_t index = 0; index < values.size(); ++index) {
    WriteIndent(out, indent + 2);
    out << std::setprecision(17) << values[index];
    if (index + 1 < values.size()) {
      out << ",";
    }
    out << "\n";
  }

  if (!values.empty()) {
    WriteIndent(out, indent);
  }
  out << "]";
}

std::string Base64Encode(const std::byte* data, size_t size) {
  static constexpr char kAlphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  std::string encoded;
  encoded.reserve(((size + 2) / 3) * 4);

  for (size_t index = 0; index < size; index += 3) {
    const size_t remaining = std::min<size_t>(3, size - index);
    uint32_t block = static_cast<uint32_t>(std::to_integer<unsigned char>(data[index])) << 16;
    if (remaining > 1) {
      block |= static_cast<uint32_t>(std::to_integer<unsigned char>(data[index + 1])) << 8;
    }
    if (remaining > 2) {
      block |= static_cast<uint32_t>(std::to_integer<unsigned char>(data[index + 2]));
    }

    encoded.push_back(kAlphabet[(block >> 18) & 0x3F]);
    encoded.push_back(kAlphabet[(block >> 12) & 0x3F]);
    encoded.push_back(remaining > 1 ? kAlphabet[(block >> 6) & 0x3F] : '=');
    encoded.push_back(remaining > 2 ? kAlphabet[block & 0x3F] : '=');
  }

  return encoded;
}

std::string SanitizePathComponent(std::string_view value) {
  std::string sanitized;
  sanitized.reserve(value.size());

  for (const unsigned char c : value) {
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-') {
      sanitized.push_back(static_cast<char>(c));
    } else {
      sanitized.push_back('_');
    }
  }

  if (sanitized.empty()) {
    sanitized = "unnamed";
  }

  return sanitized;
}

void NormalizeGraphProtoForSerialization(ONNX_NAMESPACE::GraphProto* graph);

void NormalizeAttributeProtoForSerialization(ONNX_NAMESPACE::AttributeProto* attribute) {
  if (attribute->has_g()) {
    NormalizeGraphProtoForSerialization(attribute->mutable_g());
  }

  for (int index = 0; index < attribute->graphs_size(); ++index) {
    NormalizeGraphProtoForSerialization(attribute->mutable_graphs(index));
  }
}

void NormalizeNodeProtoForSerialization(ONNX_NAMESPACE::NodeProto* node) {
  auto* attributes = node->mutable_attribute();
  for (auto& attribute : *attributes) {
    NormalizeAttributeProtoForSerialization(&attribute);
  }

  // Keep ONNX serialization stable across platforms/runs so regenerated artifacts
  // do not churn just because attribute iteration order changed.
  std::stable_sort(
      attributes->begin(),
      attributes->end(),
      [](const ONNX_NAMESPACE::AttributeProto& left, const ONNX_NAMESPACE::AttributeProto& right) {
        if (left.name() != right.name()) {
          return left.name() < right.name();
        }

        return left.type() < right.type();
      });
}

void NormalizeGraphProtoForSerialization(ONNX_NAMESPACE::GraphProto* graph) {
  for (auto& node : *graph->mutable_node()) {
    NormalizeNodeProtoForSerialization(&node);
  }
}

void NormalizeModelProtoForSerialization(ONNX_NAMESPACE::ModelProto* model) {
  if (model->has_graph()) {
    NormalizeGraphProtoForSerialization(model->mutable_graph());
  }

  for (auto& function : *model->mutable_functions()) {
    for (auto& node : *function.mutable_node()) {
      NormalizeNodeProtoForSerialization(&node);
    }
  }
}

void WriteBinaryProtoToFile(const google::protobuf::MessageLite& proto, const fs::path& output_path) {
  if (output_path.has_parent_path()) {
    fs::create_directories(output_path.parent_path());
  }

  std::ofstream output(output_path, std::ios::binary);
  ORT_ENFORCE(output.good(), "Failed to open output file: ", output_path.string());
  ORT_ENFORCE(proto.SerializeToOstream(&output), "Failed to serialize proto to file: ", output_path.string());
}

void WriteValidationMetadataToFile(const CapturedRecord& record, const fs::path& output_path) {
  if (output_path.has_parent_path()) {
    fs::create_directories(output_path.parent_path());
  }

  std::ofstream output(output_path);
  ORT_ENFORCE(output.good(), "Failed to open validation metadata file: ", output_path.string());

  output << "{\n";
  WriteIndent(output, 2);
  output << "\"expects_failure\": " << (record.expects_failure ? "true" : "false") << ",\n";
  WriteIndent(output, 2);
  output << "\"expected_failure_substring\": \"" << JsonEscape(record.expected_failure_substring) << "\",\n";
  if (!record.included_providers.empty()) {
    WriteIndent(output, 2);
    output << "\"included_providers\": ";
    WriteStringArray(output, record.included_providers, 2);
    output << ",\n";
  }
  if (!record.excluded_providers.empty()) {
    WriteIndent(output, 2);
    output << "\"excluded_providers\": ";
    WriteStringArray(output, record.excluded_providers, 2);
    output << ",\n";
  }
  WriteIndent(output, 2);
  output << "\"outputs\": [";
  if (!record.outputs.empty()) {
    output << "\n";
  }

  for (size_t index = 0; index < record.outputs.size(); ++index) {
    const auto& tensor = record.outputs[index];
    WriteIndent(output, 4);
    output << "{\n";
    WriteIndent(output, 6);
    output << "\"name\": \"" << JsonEscape(tensor.name) << "\",\n";
    WriteIndent(output, 6);
    output << "\"relative_error\": ";
    if (tensor.relative_error.has_value()) {
      output << std::setprecision(17) << *tensor.relative_error;
    } else {
      output << "null";
    }
    output << ",\n";
    WriteIndent(output, 6);
    output << "\"absolute_error\": ";
    if (tensor.absolute_error.has_value()) {
      output << std::setprecision(17) << *tensor.absolute_error;
    } else {
      output << "null";
    }
    output << ",\n";
    WriteIndent(output, 6);
    output << "\"sort_output\": " << (tensor.sort_output ? "true" : "false") << "\n";
    WriteIndent(output, 4);
    output << "}";
    if (index + 1 < record.outputs.size()) {
      output << ",";
    }
    output << "\n";
  }

  if (!record.outputs.empty()) {
    WriteIndent(output, 2);
  }
  output << "]\n";
  output << "}\n";
}

fs::path BuildArtifactDirectory(const CapturedRecord& record, const fs::path& artifact_root) {
  const fs::path source_file(record.source_file);
  return artifact_root / source_file.parent_path() / source_file.stem() /
         (SanitizePathComponent(record.test_name) + "_run" + std::to_string(record.run_index));
}

bool ShouldSerializeInputArtifact(
    const onnxruntime::test::BaseTester::Data& input,
    size_t index,
    const std::set<size_t>& initializer_indexes,
    const std::unordered_set<std::string>& model_input_names) {
  return initializer_indexes.count(index) == 0 &&
         input.def.Exists() &&
         input.data.IsAllocated() &&
         model_input_names.count(input.def.Name()) > 0;
}

bool ShouldSerializeOutputArtifact(
    const onnxruntime::test::BaseTester::Data& output,
    const std::unordered_set<std::string>& model_output_names) {
  return output.def.Exists() &&
         output.data.IsAllocated() &&
         model_output_names.count(output.def.Name()) > 0;
}

std::map<std::string, onnxruntime::test::ValidateOutputParams> BuildOutputValidationParamsByName(
    const std::vector<onnxruntime::test::BaseTester::Data>& output_data) {
  std::map<std::string, onnxruntime::test::ValidateOutputParams> params_by_name;
  for (const auto& output : output_data) {
    if (!output.def.Exists()) {
      continue;
    }
    params_by_name.emplace(output.def.Name(), output.validation_params);
  }
  return params_by_name;
}

void WriteArtifacts(CapturingOpTester& tester, CapturedRecord& record) {
  const fs::path& artifact_root = CaptureCollector::Instance().ArtifactRoot();
  if (artifact_root.empty()) {
    return;
  }

  const fs::path artifact_dir = BuildArtifactDirectory(record, artifact_root);
  fs::remove_all(artifact_dir);
  const fs::path dataset_dir = artifact_dir / "test_data_set_0";
  fs::create_directories(dataset_dir);
  std::unordered_set<std::string> model_input_names;
  std::unordered_set<std::string> model_output_names;

  try {
    onnxruntime::Model& model = tester.BuildModel();
    const auto resolve_status = model.MainGraph().Resolve({});
    if (!resolve_status.IsOK()) {
      record.warnings.push_back("Graph resolve failed before serializing model: " + resolve_status.ErrorMessage());
    }

    auto model_proto = model.ToProto();
    for (const auto& input : model_proto.graph().input()) {
      model_input_names.insert(input.name());
    }
    for (const auto& output : model_proto.graph().output()) {
      model_output_names.insert(output.name());
    }

    NormalizeModelProtoForSerialization(&model_proto);

    const fs::path model_path = artifact_dir / "model.onnx";
    WriteBinaryProtoToFile(model_proto, model_path);
    record.artifact_directory = fs::relative(artifact_dir, artifact_root).generic_string();
    record.model_path = fs::relative(model_path, artifact_root).generic_string();
  } catch (const std::exception& exception) {
    record.warnings.push_back("Failed to serialize model artifact: " + std::string(exception.what()));
  }

  const auto initializer_indexes = std::set<size_t>(
      tester.CapturedInitializerIndexes().begin(),
      tester.CapturedInitializerIndexes().end());

  size_t input_index = 0;
  const auto& input_data = tester.CapturedInputData();
  for (size_t index = 0; index < input_data.size(); ++index) {
    const auto& input = input_data[index];
    if (!ShouldSerializeInputArtifact(input, index, initializer_indexes, model_input_names)) {
      continue;
    }

    if (!input.data.IsTensor()) {
      record.warnings.push_back("Skipping non-tensor input artifact for " + input.def.Name());
      continue;
    }

    const fs::path input_path = dataset_dir / ("input_" + std::to_string(input_index++) + ".pb");
    try {
      const auto& tensor = input.data.Get<onnxruntime::Tensor>();
      auto tensor_proto = onnxruntime::utils::TensorToTensorProto(tensor, input.def.Name());
      WriteBinaryProtoToFile(tensor_proto, input_path);
      record.input_files.push_back(fs::relative(input_path, artifact_root).generic_string());
    } catch (const std::exception& exception) {
      record.warnings.push_back("Failed to serialize input artifact for " + input.def.Name() + ": " + exception.what());
    }
  }

  size_t output_index = 0;
  const auto& output_data = tester.CapturedOutputData();
  const auto output_validation_params = BuildOutputValidationParamsByName(output_data);
  for (const auto& output : output_data) {
    if (!ShouldSerializeOutputArtifact(output, model_output_names)) {
      continue;
    }

    if (!output.data.IsTensor()) {
      record.warnings.push_back("Skipping non-tensor output artifact for " + output.def.Name());
      continue;
    }

    const fs::path output_path = dataset_dir / ("output_" + std::to_string(output_index++) + ".pb");
    try {
      const auto& tensor = output.data.Get<onnxruntime::Tensor>();
      auto tensor_proto = onnxruntime::utils::TensorToTensorProto(tensor, output.def.Name());
      WriteBinaryProtoToFile(tensor_proto, output_path);
      record.output_files.push_back(fs::relative(output_path, artifact_root).generic_string());
    } catch (const std::exception& exception) {
      record.warnings.push_back("Failed to serialize output artifact for " + output.def.Name() + ": " + exception.what());
    }
  }

}

CapturedTensor CaptureOrtValue(
    const std::string& name,
    const OrtValue& value,
    bool is_initializer) {
  CapturedTensor tensor;
  tensor.name = name;
  tensor.is_initializer = is_initializer;

  if (!value.IsAllocated()) {
    tensor.value_kind = "none";
    return tensor;
  }

  if (!value.IsTensor()) {
    tensor.value_kind = "unsupported";
    tensor.warnings.push_back("Only tensor OrtValue inputs are currently serialized in runtime mode.");
    return tensor;
  }

  const auto& ort_tensor = value.Get<onnxruntime::Tensor>();
  tensor.value_kind = "tensor";
  tensor.element_type = ort_tensor.GetElementType();
  const auto dims = ort_tensor.Shape().GetDims();
  tensor.shape.assign(dims.begin(), dims.end());
  tensor.has_data = true;

  if (ort_tensor.IsDataTypeString()) {
    const std::string* strings = ort_tensor.Data<std::string>();
    tensor.string_data.assign(strings, strings + ort_tensor.Shape().Size());
    return tensor;
  }

  tensor.data_encoding = "base64";
  tensor.data_base64 = Base64Encode(
      reinterpret_cast<const std::byte*>(ort_tensor.DataRaw()),
      ort_tensor.SizeInBytes());
  return tensor;
}

CapturedAttribute CaptureAttribute(const ONNX_NAMESPACE::AttributeProto& attribute) {
  CapturedAttribute captured;
  captured.name = attribute.name();

  switch (attribute.type()) {
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INT:
      captured.value_kind = "int";
      captured.int_value = attribute.i();
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT:
      captured.value_kind = "float";
      captured.float_value = attribute.f();
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRING:
      captured.value_kind = "string";
      captured.string_value = attribute.s();
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INTS:
      captured.value_kind = "ints";
      captured.ints_value.assign(attribute.ints().begin(), attribute.ints().end());
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS:
      captured.value_kind = "floats";
      captured.floats_value.assign(attribute.floats().begin(), attribute.floats().end());
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS:
      captured.value_kind = "strings";
      captured.strings_value.assign(attribute.strings().begin(), attribute.strings().end());
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR: {
      captured.value_kind = "tensor";
      const auto& tensor = attribute.t();
      captured.tensor_element_type = tensor.data_type();
      captured.tensor_shape.assign(tensor.dims().begin(), tensor.dims().end());

      if (!tensor.string_data().empty()) {
        captured.tensor_string_data.assign(tensor.string_data().begin(), tensor.string_data().end());
      } else if (!tensor.raw_data().empty()) {
        captured.tensor_data_encoding = "base64";
        captured.tensor_data_base64 = Base64Encode(
            reinterpret_cast<const std::byte*>(tensor.raw_data().data()),
            tensor.raw_data().size());
      } else {
        captured.warnings.push_back("Tensor attribute does not contain raw_data or string_data.");
      }
      break;
    }
    default:
      captured.value_kind = "unsupported";
      captured.warnings.push_back("Unsupported attribute type in runtime capture.");
      break;
  }

  return captured;
}

std::string CompletenessOf(const CapturedRecord& record) {
  if (record.inputs.empty()) {
    return "metadata_only";
  }

  const bool all_inputs_serialized = std::all_of(
      record.inputs.begin(),
      record.inputs.end(),
      [](const CapturedTensor& tensor) { return tensor.has_data || tensor.value_kind == "none"; });

  return all_inputs_serialized ? "complete" : "partial";
}

void WriteTensor(std::ostream& out, const CapturedTensor& tensor, int indent) {
  WriteIndent(out, indent);
  out << "{\n";
  WriteIndent(out, indent + 2);
  out << "\"name\": \"" << JsonEscape(tensor.name) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"value_kind\": \"" << JsonEscape(tensor.value_kind) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"element_type\": " << tensor.element_type << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"shape\": ";
  WriteIntArray(out, tensor.shape, indent + 2);
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"is_initializer\": " << (tensor.is_initializer ? "true" : "false") << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"has_data\": " << (tensor.has_data ? "true" : "false") << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"relative_error\": ";
  if (tensor.relative_error.has_value()) {
    out << std::setprecision(17) << *tensor.relative_error;
  } else {
    out << "null";
  }
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"absolute_error\": ";
  if (tensor.absolute_error.has_value()) {
    out << std::setprecision(17) << *tensor.absolute_error;
  } else {
    out << "null";
  }
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"sort_output\": " << (tensor.sort_output ? "true" : "false") << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"data_encoding\": \"" << JsonEscape(tensor.data_encoding) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"data_base64\": \"" << JsonEscape(tensor.data_base64) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"string_data\": ";
  WriteStringArray(out, tensor.string_data, indent + 2);
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"warnings\": ";
  WriteStringArray(out, tensor.warnings, indent + 2);
  out << "\n";
  WriteIndent(out, indent);
  out << "}";
}

void WriteTensorList(std::ostream& out, const std::vector<CapturedTensor>& tensors, int indent) {
  out << "[";
  if (!tensors.empty()) {
    out << "\n";
  }

  for (size_t index = 0; index < tensors.size(); ++index) {
    WriteTensor(out, tensors[index], indent + 2);
    if (index + 1 < tensors.size()) {
      out << ",";
    }
    out << "\n";
  }

  if (!tensors.empty()) {
    WriteIndent(out, indent);
  }
  out << "]";
}

void WriteAttribute(std::ostream& out, const CapturedAttribute& attribute, int indent) {
  WriteIndent(out, indent);
  out << "{\n";
  WriteIndent(out, indent + 2);
  out << "\"name\": \"" << JsonEscape(attribute.name) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"value_kind\": \"" << JsonEscape(attribute.value_kind) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"int_value\": ";
  if (attribute.int_value.has_value()) {
    out << *attribute.int_value;
  } else {
    out << "null";
  }
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"float_value\": ";
  if (attribute.float_value.has_value()) {
    out << std::setprecision(17) << *attribute.float_value;
  } else {
    out << "null";
  }
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"string_value\": ";
  if (attribute.string_value.has_value()) {
    out << "\"" << JsonEscape(*attribute.string_value) << "\"";
  } else {
    out << "null";
  }
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"ints_value\": ";
  WriteIntArray(out, attribute.ints_value, indent + 2);
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"floats_value\": ";
  WriteDoubleArray(out, attribute.floats_value, indent + 2);
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"strings_value\": ";
  WriteStringArray(out, attribute.strings_value, indent + 2);
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"tensor_element_type\": " << attribute.tensor_element_type << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"tensor_shape\": ";
  WriteIntArray(out, attribute.tensor_shape, indent + 2);
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"tensor_data_encoding\": \"" << JsonEscape(attribute.tensor_data_encoding) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"tensor_data_base64\": \"" << JsonEscape(attribute.tensor_data_base64) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"tensor_string_data\": ";
  WriteStringArray(out, attribute.tensor_string_data, indent + 2);
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"warnings\": ";
  WriteStringArray(out, attribute.warnings, indent + 2);
  out << "\n";
  WriteIndent(out, indent);
  out << "}";
}

void WriteAttributeList(std::ostream& out, const std::vector<CapturedAttribute>& attributes, int indent) {
  out << "[";
  if (!attributes.empty()) {
    out << "\n";
  }

  for (size_t index = 0; index < attributes.size(); ++index) {
    WriteAttribute(out, attributes[index], indent + 2);
    if (index + 1 < attributes.size()) {
      out << ",";
    }
    out << "\n";
  }

  if (!attributes.empty()) {
    WriteIndent(out, indent);
  }
  out << "]";
}

void WriteRecord(std::ostream& out, const CapturedRecord& record, int indent) {
  WriteIndent(out, indent);
  out << "{\n";
  WriteIndent(out, indent + 2);
  out << "\"source_file\": \"" << JsonEscape(record.source_file) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"test_suite\": \"" << JsonEscape(record.test_suite) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"test_name\": \"" << JsonEscape(record.test_name) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"run_index\": " << record.run_index << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"op_name\": \"" << JsonEscape(record.op_name) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"opset\": " << record.opset << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"domain\": \"" << JsonEscape(record.domain) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"expects_failure\": " << (record.expects_failure ? "true" : "false") << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"expected_failure_substring\": \"" << JsonEscape(record.expected_failure_substring) << "\",\n";
  if (!record.included_providers.empty()) {
    WriteIndent(out, indent + 2);
    out << "\"included_providers\": ";
    WriteStringArray(out, record.included_providers, indent + 2);
    out << ",\n";
  }
  if (!record.excluded_providers.empty()) {
    WriteIndent(out, indent + 2);
    out << "\"excluded_providers\": ";
    WriteStringArray(out, record.excluded_providers, indent + 2);
    out << ",\n";
  }
  WriteIndent(out, indent + 2);
  out << "\"saw_run_call\": " << (record.saw_run_call ? "true" : "false") << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"node_count\": " << record.node_count << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"artifact_directory\": \"" << JsonEscape(record.artifact_directory) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"model_path\": \"" << JsonEscape(record.model_path) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"input_files\": ";
  WriteStringArray(out, record.input_files, indent + 2);
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"output_files\": ";
  WriteStringArray(out, record.output_files, indent + 2);
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"completeness\": \"" << CompletenessOf(record) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"inputs\": ";
  WriteTensorList(out, record.inputs, indent + 2);
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"outputs\": ";
  WriteTensorList(out, record.outputs, indent + 2);
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"attributes\": ";
  WriteAttributeList(out, record.attributes, indent + 2);
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"warnings\": ";
  WriteStringArray(out, record.warnings, indent + 2);
  out << "\n";
  WriteIndent(out, indent);
  out << "}";
}

CapturedRecord BuildRecordFromTester(
    CapturingOpTester& tester,
    bool saw_run_call,
    int run_index,
    onnxruntime::test::BaseTester::ExpectResult expect_result,
    std::string_view expected_failure_string,
    const std::unordered_set<std::string>* excluded_provider_types,
    const std::vector<std::unique_ptr<onnxruntime::IExecutionProvider>>* execution_providers) {
  CapturedRecord record;
  record.source_file = EMX_ORT_CAPTURE_SOURCE_FILE_REL;
  record.run_index = run_index;
  record.saw_run_call = saw_run_call;
  record.op_name = tester.CapturedOpName();
  record.opset = tester.CapturedOpset();
  record.domain = tester.CapturedDomain();
  record.expects_failure = expect_result == onnxruntime::test::BaseTester::ExpectResult::kExpectFailure;
  record.expected_failure_substring = std::string(expected_failure_string);
  if (execution_providers != nullptr) {
    for (const auto& execution_provider : *execution_providers) {
      if (execution_provider) {
        record.included_providers.push_back(NormalizeProviderName(execution_provider->Type()));
      }
    }
  } else {
    record.included_providers = tester.CapturedConfiguredExecutionProviders();
  }
  if (excluded_provider_types != nullptr) {
    record.excluded_providers.reserve(excluded_provider_types->size());
    for (const auto& provider_type : *excluded_provider_types) {
      record.excluded_providers.push_back(NormalizeProviderName(provider_type));
    }
    std::sort(record.excluded_providers.begin(), record.excluded_providers.end());
  } else {
    record.excluded_providers = tester.CapturedExcludedProviderTypes();
  }

  if (const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info(); test_info != nullptr) {
    record.test_suite = test_info->test_suite_name();
    record.test_name = test_info->name();
  }

  const auto initializer_indexes = std::set<size_t>(
      tester.CapturedInitializerIndexes().begin(),
      tester.CapturedInitializerIndexes().end());

  const auto& input_data = tester.CapturedInputData();
  for (size_t index = 0; index < input_data.size(); ++index) {
    const bool is_initializer = initializer_indexes.count(index) > 0;
    record.inputs.push_back(CaptureOrtValue(input_data[index].def.Name(), input_data[index].data, is_initializer));
  }

  const auto& output_data = tester.CapturedOutputData();
  for (const auto& output : output_data) {
    auto captured = CaptureOrtValue(output.def.Name(), output.data, false);
    if (output.validation_params.relative_error.has_value()) {
      captured.relative_error = *output.validation_params.relative_error;
    }
    if (output.validation_params.absolute_error.has_value()) {
      captured.absolute_error = *output.validation_params.absolute_error;
    }
    captured.sort_output = output.validation_params.sort_output;
    record.outputs.push_back(std::move(captured));
  }

  record.warnings.push_back(
      "Runtime mode currently captures inputs and expected outputs directly from OpTester. "
      "Attributes remain empty until a later merge step enriches them.");

  WriteArtifacts(tester, record);

  if (!record.artifact_directory.empty()) {
    const fs::path validation_path = BuildArtifactDirectory(record, CaptureCollector::Instance().ArtifactRoot()) / "validation.json";
    try {
      WriteValidationMetadataToFile(record, validation_path);
    } catch (const std::exception& exception) {
      record.warnings.push_back("Failed to serialize validation metadata: " + std::string(exception.what()));
    }
  }

  return record;
}

}  // namespace

CaptureCollector& CaptureCollector::Instance() {
  static CaptureCollector collector;
  return collector;
}

void CaptureCollector::Reset(std::string source_root_relative, std::string source_file_relative, fs::path artifact_root) {
  source_root_relative_ = std::move(source_root_relative);
  source_file_relative_ = std::move(source_file_relative);
  artifact_root_ = std::move(artifact_root);
  records_.clear();
  next_run_index_by_test_case_.clear();
}

void CaptureCollector::AddRecord(CapturedRecord record) {
  if (record.source_file.empty()) {
    record.source_file = source_file_relative_;
  }

  records_.push_back(std::move(record));
}

int CaptureCollector::AllocateRunIndex(std::string_view test_suite, std::string_view test_name) {
  const std::string key = std::string(test_suite) + "::" + std::string(test_name);
  int& next_run_index = next_run_index_by_test_case_[key];
  const int run_index = next_run_index;
  ++next_run_index;
  return run_index;
}

size_t CaptureCollector::RecordCount() const noexcept {
  return records_.size();
}

const fs::path& CaptureCollector::ArtifactRoot() const noexcept {
  return artifact_root_;
}

void CaptureCollector::WriteJson(const fs::path& output_path) const {
  if (output_path.has_parent_path()) {
    fs::create_directories(output_path.parent_path());
  }

  std::ofstream output(output_path);
  output << "{\n";
  WriteIndent(output, 2);
  output << "\"capture_mode\": \"runtime\",\n";
  WriteIndent(output, 2);
  output << "\"source_root\": \"" << JsonEscape(source_root_relative_) << "\",\n";
  WriteIndent(output, 2);
  output << "\"source_file\": \"" << JsonEscape(source_file_relative_) << "\",\n";
  WriteIndent(output, 2);
  output << "\"artifact_root\": \"" << JsonEscape(artifact_root_.generic_string()) << "\",\n";
  WriteIndent(output, 2);
  output << "\"record_count\": " << records_.size() << ",\n";
  WriteIndent(output, 2);
  output << "\"records\": [\n";

  for (size_t index = 0; index < records_.size(); ++index) {
    WriteRecord(output, records_[index], 4);
    if (index + 1 < records_.size()) {
      output << ",";
    }
    output << "\n";
  }

  WriteIndent(output, 2);
  output << "]\n";
  output << "}\n";
}

CapturingOpTester::~CapturingOpTester() {
  try {
    if (!has_captured_snapshot_) {
      const auto& run_context = GetRunContext();
      CaptureSnapshot(
          false,
          run_context.expect_result,
          run_context.expected_failure_string,
          &run_context.excluded_provider_types);
    }
  } catch (...) {
  }
}

std::vector<std::string> CapturingOpTester::CapturedConfiguredExecutionProviders() const {
  std::vector<std::string> execution_provider_types;
  const auto& execution_providers = GetRunContext().execution_providers;
  execution_provider_types.reserve(execution_providers.size());
  for (const auto& execution_provider : execution_providers) {
    if (execution_provider) {
      execution_provider_types.push_back(NormalizeProviderName(execution_provider->Type()));
    }
  }
  return execution_provider_types;
}

std::vector<std::string> CapturingOpTester::CapturedExcludedProviderTypes() const {
  std::vector<std::string> excluded_provider_types;
  excluded_provider_types.reserve(GetRunContext().excluded_provider_types.size());
  for (const auto& provider_type : GetRunContext().excluded_provider_types) {
    excluded_provider_types.push_back(NormalizeProviderName(provider_type));
  }
  std::sort(excluded_provider_types.begin(), excluded_provider_types.end());
  return excluded_provider_types;
}

void CapturingOpTester::Run(
    ExpectResult expect_result,
    const std::string& expected_failure_string,
    const std::unordered_set<std::string>& excluded_provider_types,
    const onnxruntime::RunOptions* run_options,
    std::vector<std::unique_ptr<onnxruntime::IExecutionProvider>>* execution_providers,
    ExecutionMode execution_mode,
    const onnxruntime::Graph::ResolveOptions& resolve_options) {
  CaptureSnapshot(
      true,
      expect_result,
      expected_failure_string,
      &excluded_provider_types,
      execution_providers);
  onnxruntime::test::OpTester::Run(
      expect_result,
      expected_failure_string,
      excluded_provider_types,
      run_options,
      execution_providers,
      execution_mode,
      resolve_options);
}

void CapturingOpTester::Run(
    onnxruntime::SessionOptions session_options,
    ExpectResult expect_result,
    const std::string& expected_failure_string,
    const std::unordered_set<std::string>& excluded_provider_types,
    const onnxruntime::RunOptions* run_options,
    std::vector<std::unique_ptr<onnxruntime::IExecutionProvider>>* execution_providers,
    const onnxruntime::Graph::ResolveOptions& resolve_options,
    size_t* number_of_pre_packed_weights_counter,
    size_t* number_of_shared_pre_packed_weights_counter) {
  CaptureSnapshot(
      true,
      expect_result,
      expected_failure_string,
      &excluded_provider_types,
      execution_providers);
  onnxruntime::test::OpTester::Run(
      std::move(session_options),
      expect_result,
      expected_failure_string,
      excluded_provider_types,
      run_options,
      execution_providers,
      resolve_options,
      number_of_pre_packed_weights_counter,
      number_of_shared_pre_packed_weights_counter);
}

void CapturingOpTester::RunWithConfig(
    size_t* number_of_pre_packed_weights_counter,
    size_t* number_of_shared_pre_packed_weights_counter) {
  const auto& run_context = GetRunContext();
  CaptureSnapshot(
      true,
      run_context.expect_result,
      run_context.expected_failure_string,
      &run_context.excluded_provider_types);
  onnxruntime::test::OpTester::RunWithConfig(
      number_of_pre_packed_weights_counter,
      number_of_shared_pre_packed_weights_counter);
}

void CapturingOpTester::CaptureSnapshot(
    bool saw_run_call,
    ExpectResult expect_result,
    std::string expected_failure_string,
    const std::unordered_set<std::string>* excluded_provider_types,
    const std::vector<std::unique_ptr<onnxruntime::IExecutionProvider>>* execution_providers) {
  if (IsQuantizeLstmReferenceHelper(*this)) {
    has_captured_snapshot_ = true;
    return;
  }

  std::string test_suite;
  std::string test_name;
  if (const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info(); test_info != nullptr) {
    test_suite = test_info->test_suite_name();
    test_name = test_info->name();
  }

  const int run_index = CaptureCollector::Instance().AllocateRunIndex(test_suite, test_name);
  CaptureCollector::Instance().AddRecord(
      BuildRecordFromTester(
          *this,
          saw_run_call,
          run_index,
          expect_result,
          expected_failure_string,
          excluded_provider_types,
          execution_providers));
  has_captured_snapshot_ = true;
}

}  // namespace emx::ort_runtime
