<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_artifact_validation_docs.py::test_artifact_validation_error_doc -->

# ORT artifact validation errors

Aggregates non-OK artifact validation outcomes.

Expectation source: `tests/artifact_validation_expected.json`

Validated cases: 4182 / 4189 OK, 7 non-OK.

| Error message | Count | Sources |
| --- | --- | --- |
| ONNX Runtime error | 7 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test, artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test |

## Error frequency by source

| Error message | Source | Count |
| --- | --- | --- |
| ONNX Runtime error | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | 5 |
| ONNX Runtime error | artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test | 2 |

## Failing artifact cases

Lists every artifact case with a non-OK expected validation result.

| Case | Source | Error |
| --- | --- | --- |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_BFloat16Input_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | [ONNXRuntimeError] : 10 : INVALID_GRAPH : This is an invalid model. Type Error: Type 'tensor(bfloat16)' of input parameter (x) of operator (Shape) in node () is invalid. |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Float16Input_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | [ONNXRuntimeError] : 1 : FAIL : Node () Op (Flatten) [ShapeInferenceError] Invalid value(-1) for attribute 'axis' |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Float16ScaleBiasOutput_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | [ONNXRuntimeError] : 1 : FAIL : Node () Op (Flatten) [ShapeInferenceError] Invalid value(-1) for attribute 'axis' |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Float16Input_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | [ONNXRuntimeError] : 1 : FAIL : Node () Op (Flatten) [ShapeInferenceError] Invalid value(-1) for attribute 'axis' |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Float16ScaleOutput_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | [ONNXRuntimeError] : 1 : FAIL : Node () Op (Flatten) [ShapeInferenceError] Invalid value(-1) for attribute 'axis' |
| artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test/MatMul2DBlkZp_run0 | artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test | [ONNXRuntimeError] : 1 : FAIL : Load model from D:/emmtrix/git/emx-ort-test-materializer/artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test/MatMul2DBlkZp_run0/model.onnx failed:Node (node1) Op (MatMulFpQ4) [ShapeInferenceError] 4b quantization not yet supported on this hardware platform! |
| artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test/MatMul2DSym_run0 | artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test | [ONNXRuntimeError] : 1 : FAIL : Load model from D:/emmtrix/git/emx-ort-test-materializer/artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test/MatMul2DSym_run0/model.onnx failed:Node (node1) Op (MatMulFpQ4) [ShapeInferenceError] 4b quantization not yet supported on this hardware platform! |
