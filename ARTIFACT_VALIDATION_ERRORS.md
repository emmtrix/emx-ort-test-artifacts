<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_artifact_validation_docs.py::test_artifact_validation_error_doc -->

# ORT artifact validation errors

Aggregates non-OK artifact validation outcomes.

Expectation source: `tests/artifact_validation_expected.json`

Validated cases: 3244 / 3259 OK, 15 non-OK.

| Error message | Count | Sources |
| --- | --- | --- |
| Input count mismatch | 8 | artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test |
| ONNX Runtime error | 7 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test, artifacts/onnxruntime/test/contrib_ops/moe_test |

## Error frequency by source

| Error message | Source | Count |
| --- | --- | --- |
| ONNX Runtime error | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | 5 |
| Input count mismatch | artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test | 8 |
| ONNX Runtime error | artifacts/onnxruntime/test/contrib_ops/moe_test | 2 |

## Failing artifact cases

Lists every artifact case with a non-OK expected validation result.

| Case | Source | Error |
| --- | --- | --- |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_BFloat16Input_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | [ONNXRuntimeError] : 10 : INVALID_GRAPH : This is an invalid model. Type Error: Type 'tensor(bfloat16)' of input parameter (x) of operator (Shape) in node () is invalid. |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Float16Input_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | [ONNXRuntimeError] : 1 : FAIL : Node () Op (Flatten) [ShapeInferenceError] Invalid value(-1) for attribute 'axis' |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Float16ScaleBiasOutput_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | [ONNXRuntimeError] : 1 : FAIL : Node () Op (Flatten) [ShapeInferenceError] Invalid value(-1) for attribute 'axis' |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Float16Input_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | [ONNXRuntimeError] : 1 : FAIL : Node () Op (Flatten) [ShapeInferenceError] Invalid value(-1) for attribute 'axis' |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Float16ScaleOutput_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | [ONNXRuntimeError] : 1 : FAIL : Node () Op (Flatten) [ShapeInferenceError] Invalid value(-1) for attribute 'axis' |
| artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test/TestCoo_run0 | artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test | test_data_set_0: input count mismatch: files=1, model_inputs=2 |
| artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test/TestCoo_run1 | artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test | test_data_set_0: input count mismatch: files=1, model_inputs=2 |
| artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test/TestCoo_run2 | artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test | test_data_set_0: input count mismatch: files=1, model_inputs=2 |
| artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test/TestCoo_run3 | artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test | test_data_set_0: input count mismatch: files=1, model_inputs=2 |
| artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test/TestCsr_run0 | artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test | test_data_set_0: input count mismatch: files=1, model_inputs=2 |
| artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test/TestCsr_run1 | artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test | test_data_set_0: input count mismatch: files=1, model_inputs=2 |
| artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test/TestCsr_run2 | artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test | test_data_set_0: input count mismatch: files=1, model_inputs=2 |
| artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test/TestCsr_run3 | artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test | test_data_set_0: input count mismatch: files=1, model_inputs=2 |
| artifacts/onnxruntime/test/contrib_ops/moe_test/QMoETest_CPU_RouterWeights_run0 | artifacts/onnxruntime/test/contrib_ops/moe_test | [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from /home/runner/work/emx-ort-test-materializer/emx-ort-test-materializer/artifacts/onnxruntime/test/contrib_ops/moe_test/QMoETest_CPU_RouterWeights_run0/model.onnx failed:This is an invalid model. In Node, ("node1", QMoE, "com.microsoft", -1) : ("input": tensor(float16),"router_probs": tensor(float16),"fc1_experts_weights": tensor(uint8),"fc1_scales": tensor(float),"","fc2_experts_weights": tensor(uint8),"fc2_scales": tensor(float),"fc2_experts_bias": tensor(float16),"","","","","","","router_weights": tensor(float16),) -> ("output": tensor(float16),) , Error Node(node1) with schema(com.microsoft::QMoE:1) has input size 15 not in range [min=7, max=14]. |
| artifacts/onnxruntime/test/contrib_ops/moe_test/QMoETest_CPU_RouterWeights_run1 | artifacts/onnxruntime/test/contrib_ops/moe_test | [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from /home/runner/work/emx-ort-test-materializer/emx-ort-test-materializer/artifacts/onnxruntime/test/contrib_ops/moe_test/QMoETest_CPU_RouterWeights_run1/model.onnx failed:This is an invalid model. In Node, ("node1", QMoE, "com.microsoft", -1) : ("input": tensor(float16),"router_probs": tensor(float16),"fc1_experts_weights": tensor(uint8),"fc1_scales": tensor(float),"","fc2_experts_weights": tensor(uint8),"fc2_scales": tensor(float),"fc2_experts_bias": tensor(float16),"","","","","","","router_weights": tensor(float16),) -> ("output": tensor(float16),) , Error Node(node1) with schema(com.microsoft::QMoE:1) has input size 15 not in range [min=7, max=14]. |
