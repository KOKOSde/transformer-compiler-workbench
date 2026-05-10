# Transformer Compiler Workbench Report

This report is CPU-first. It does not claim GPU or CUDA validation.

![Compiler workbench pipeline](assets/pipeline.svg)

## Graph Summary

| Graph | Nodes | Cast | Transpose | Reshape | CPU latency p50 | Max output diff |
|---|---:|---:|---:|---:|---:|---:|
| Original | 51 | 0 | 3 | 0 |  | 0 |
| ORT optimized | 31 | 0 | 1 | 12 | 0.019 ms | 0 |
| Custom optimized | 48 | 0 | 1 | 0 | 0.018 ms | 0 |

![Node count by graph](assets/node_counts.svg)

![Custom rewrite pass effects](assets/pass_effects.svg)

![ORT graph rewrite footprint](assets/ort_op_delta.svg)


## Rewrite Passes

- `remove_identity`: 1 rewrite(s)
- `remove_dropout_inference`: 0 rewrite(s)
- `collapse_cast_chains`: 0 rewrite(s)
- `collapse_transpose_pairs`: 1 rewrite(s)
- `remove_noop_reshape`: 0 rewrite(s)

## Optimization Opportunities

- Remove Identity nodes when outputs are rewired safely.
- Cancel adjacent Transpose pairs when permutations invert.
- Report-only: MatMul + Add can feed future linear fusion passes.
- Report-only: MatMul + Add + GELU-like regions can feed future MLP fusion passes.
- Report-only: GELU-like regions can feed future activation fusion.
- Report-only: LayerNorm-like subgraphs can feed canonicalization.

## Output Parity

- Standalone validation parity: True
- ORT parity: True
- Custom parity: True

## ONNX-MLIR Lowering

- Status: skipped
- Reason: onnx-mlir binary not found on PATH

## Report Files

- `reports/assets/node_counts.svg`
- `reports/assets/ort_op_delta.svg`
- `reports/assets/pass_effects.svg`
- `reports/assets/pipeline.svg`
- `reports/baseline.json`
- `reports/baseline.md`
- `reports/lowering.json`
- `reports/opt.json`
- `reports/validate.json`
