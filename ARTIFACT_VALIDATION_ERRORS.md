<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_artifact_validation_docs.py::test_artifact_validation_error_doc -->

# ORT artifact validation errors

Aggregates non-OK artifact validation outcomes.

Expectation source: `tests/artifact_validation_expected.json`

Validated cases: 3130 / 3257 OK, 127 non-OK.

| Error message | Count | Sources |
| --- | --- | --- |
| Values differ | 112 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test |
| Input count mismatch | 8 | artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test |
| ONNX Runtime error | 7 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test, artifacts/onnxruntime/test/contrib_ops/moe_test |

## Error frequency by source

| Error message | Source | Count |
| --- | --- | --- |
| Values differ | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | 112 |
| ONNX Runtime error | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | 5 |
| Input count mismatch | artifacts/onnxruntime/test/contrib_ops/math/matmul_sparse_test | 8 |
| ONNX Runtime error | artifacts/onnxruntime/test/contrib_ops/moe_test | 2 |

## Failing artifact cases

Lists every artifact case with a non-OK expected validation result.

| Case | Source | Error |
| --- | --- | --- |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run0 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run1 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run10 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run11 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run12 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run13 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run14 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run15 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run2 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run3 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run4 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run5 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run6 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run7 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run8 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run9 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run0 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run1 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run10 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run11 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run12 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run13 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run14 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run15 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run16 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run17 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run18 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run19 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run2 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run20 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run21 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run22 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run23 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run24 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run25 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run26 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run27 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run28 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run29 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run3 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run30 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run31 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run4 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run5 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run6 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run7 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run8 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run9 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run0 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run1 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run10 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run11 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run12 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run13 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run14 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run15 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run16 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run17 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run18 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run19 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run2 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run20 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run21 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run22 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run23 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run24 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run25 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run26 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run27 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run28 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run29 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run3 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run30 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run31 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run4 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run5 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run6 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run7 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run8 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run9 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run0 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run1 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run10 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run11 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run12 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run13 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run14 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run15 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run16 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run17 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run18 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run19 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run2 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run20 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run21 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run22 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run23 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run24 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run25 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run26 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run27 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run28 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run29 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run3 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run30 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run31 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run4 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run5 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run6 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run7 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run8 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
| artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run9 | artifacts/onnxruntime/test/contrib_ops/gather_block_quantized_op_test | test_data_set_0 output_0 (output): FAIL - values differ |
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
| artifacts/onnxruntime/test/contrib_ops/moe_test/QMoETest_CPU_RouterWeights_run0 | artifacts/onnxruntime/test/contrib_ops/moe_test | [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from D:/emmtrix/git/emx-ort-test-materializer/artifacts/onnxruntime/test/contrib_ops/moe_test/QMoETest_CPU_RouterWeights_run0/model.onnx failed:This is an invalid model. In Node, ("node1", QMoE, "com.microsoft", -1) : ("input": tensor(float16),"router_probs": tensor(float16),"fc1_experts_weights": tensor(uint8),"fc1_scales": tensor(float),"","fc2_experts_weights": tensor(uint8),"fc2_scales": tensor(float),"fc2_experts_bias": tensor(float16),"","","","","","","router_weights": tensor(float16),) -> ("output": tensor(float16),) , Error Node(node1) with schema(com.microsoft::QMoE:1) has input size 15 not in range [min=7, max=14]. |
| artifacts/onnxruntime/test/contrib_ops/moe_test/QMoETest_CPU_RouterWeights_run1 | artifacts/onnxruntime/test/contrib_ops/moe_test | [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from D:/emmtrix/git/emx-ort-test-materializer/artifacts/onnxruntime/test/contrib_ops/moe_test/QMoETest_CPU_RouterWeights_run1/model.onnx failed:This is an invalid model. In Node, ("node1", QMoE, "com.microsoft", -1) : ("input": tensor(float16),"router_probs": tensor(float16),"fc1_experts_weights": tensor(uint8),"fc1_scales": tensor(float),"","fc2_experts_weights": tensor(uint8),"fc2_scales": tensor(float),"fc2_experts_bias": tensor(float16),"","","","","","","router_weights": tensor(float16),) -> ("output": tensor(float16),) , Error Node(node1) with schema(com.microsoft::QMoE:1) has input size 15 not in range [min=7, max=14]. |
