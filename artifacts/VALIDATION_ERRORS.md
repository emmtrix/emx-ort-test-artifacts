<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: python tools/scripts/generate_artifact_validation_overview.py -->

# ORT artifact validation errors

Aggregates non-OK artifact validation outcomes.

Expectation source: `tests/artifact_validation_expected.json`

Validated cases: 4184 / 4184 OK, 0 non-OK.

Ignored artifact generation cases: 5.

| Error message | Count | Sources |
| --- | --- | --- |

## Error frequency by source

| Error message | Source | Count |
| --- | --- | --- |

## Failing artifact cases

Lists every artifact case with a non-OK expected validation result.

| Case | Source | Error |
| --- | --- | --- |

## Ignored Artifact Generation Cases

Lists configured artifact cases that generation skips, together with the tracked reason.

| Case | Source | Reason |
| --- | --- | --- |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_BFloat16Input_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | Ignored until the runtime artifact pipeline can replay this legacy LayerNormalization bfloat16 case without surfacing a known CPU environment limitation as an artifact failure. |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Float16Input_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | Ignored until the runtime artifact pipeline preserves a compatible ONNX opset import for legacy mixed-precision LayerNormalization exports. |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Float16ScaleBiasOutput_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | Ignored until the runtime artifact pipeline preserves a compatible ONNX opset import for legacy mixed-precision LayerNormalization exports. |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Float16Input_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | Ignored until the runtime artifact pipeline preserves a compatible ONNX opset import for legacy mixed-precision LayerNormalization exports. |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Float16ScaleOutput_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | Ignored until the runtime artifact pipeline preserves a compatible ONNX opset import for legacy mixed-precision LayerNormalization exports. |
