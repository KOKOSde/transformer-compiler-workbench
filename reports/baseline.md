# ONNX Graph Analysis: `model.onnx`

- Nodes: 51
- Initializers: 16

## Op Histogram

| Op | Count |
|---|---:|
| Add | 13 |
| Constant | 7 |
| Div | 3 |
| Erf | 1 |
| Identity | 1 |
| MatMul | 8 |
| Mul | 4 |
| Pow | 2 |
| ReduceMean | 4 |
| Softmax | 1 |
| Sqrt | 2 |
| Sub | 2 |
| Transpose | 3 |

## Suspected Opportunities

- Remove Identity nodes when outputs are rewired safely.
- Cancel adjacent Transpose pairs when permutations invert.
- Report-only: MatMul + Add can feed future linear fusion passes.
- Report-only: MatMul + Add + GELU-like regions can feed future MLP fusion passes.
- Report-only: GELU-like regions can feed future activation fusion.
- Report-only: LayerNorm-like subgraphs can feed canonicalization.

## Pattern Counts

- matmul_add: 6
- matmul_add_gelu_like: 1
- gelu_like: 1
- layer_norm_like: 2
- attention_like: 1
- transpose_reshape_chains: 1
- cast_heavy_regions: 0
