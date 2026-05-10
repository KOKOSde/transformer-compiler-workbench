# transformer-compiler-workbench

[![CI](https://github.com/KOKOSde/transformer-compiler-workbench/actions/workflows/ci.yml/badge.svg)](https://github.com/KOKOSde/transformer-compiler-workbench/actions/workflows/ci.yml)

`transformer-compiler-workbench` is a CPU-first ONNX/MLIR workbench for
studying how transformer inference graphs change through analysis, ONNX Runtime
optimization, conservative graph rewrites, validation, and optional ONNX-MLIR
lowering.

It is not a replacement for ONNX Runtime, ONNX Runtime transformer optimizer,
Hugging Face Optimum, NVIDIA ONNX GraphSurgeon, or ONNX-MLIR. The goal is to
connect compiler tools into a clear research workflow and make graph changes
visible, testable, and reproducible.

The NVIDIA LPX-specific positioning is in
[docs/NVIDIA_LPX_ALIGNMENT.md](docs/NVIDIA_LPX_ALIGNMENT.md).

## Architecture

```mermaid
flowchart LR
    A["Export tiny transformer-like ONNX"] --> B["Analyze graph"]
    B --> C["ONNX Runtime offline optimization"]
    C --> D["Conservative custom rewrites"]
    D --> E["ONNX checker + ORT CPU parity"]
    D --> F["Optional ONNX-MLIR lowering"]
    B --> G["Markdown report"]
    C --> G
    E --> G
    F --> G
```

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

python -m tcw export --model distilbert-base-uncased --out artifacts/model.onnx
python -m tcw analyze artifacts/model.onnx --out reports/baseline.json
python -m tcw optimize artifacts/model.onnx --out artifacts/model.opt.onnx --report reports/opt.json --sample-inputs artifacts/model.inputs.npz
python -m tcw validate artifacts/model.onnx artifacts/model.opt.onnx --out reports/validate.json --sample-inputs artifacts/model.inputs.npz
python -m tcw lower artifacts/model.opt.onnx --emit-dir artifacts/mlir
cp artifacts/mlir/lowering.json reports/lowering.json
python -m tcw report --reports reports --out reports/index.md
```

Or run the full workflow:

```bash
scripts/reproduce.sh
```

The MVP export command writes a deterministic tiny transformer-like ONNX graph.
It intentionally avoids model downloads so the first workflow works on macOS
CPU. The graph uses standard ONNX operators and includes attention-like,
MatMul/Add, GELU-like, LayerNorm-like, Identity, and canceling Transpose
patterns for analysis and rewrite experiments.

## Example Report

```markdown
| Graph | Nodes | Cast | Transpose | Reshape | CPU latency p50 | Max output diff |
|---|---:|---:|---:|---:|---:|---:|
| Original | X | X | X | X |  | 0 |
| ORT optimized | X | X | X | X | X ms | Y |
| Custom optimized | X | X | X | X | X ms | Y |
```

Latency is ONNX Runtime CPU latency only. The project does not claim GPU,
CUDA, TensorRT, or real deployment speedups.

The report also emits SVG visualizations under `reports/assets/`, including the
compiler workflow, node-count changes, custom pass effects, and ORT op deltas.

## Supported Rewrites

All rewrite passes are conservative and must pass `onnx.checker` plus ONNX
Runtime output parity tests.

- Remove `Identity` nodes by rewiring uses.
- Remove inference `Dropout` when the mask output is unused and no
  `training_mode` input is present.
- Collapse safe `Cast -> Cast` round trips when the source dtype and final dtype
  are statically known to match.
- Cancel adjacent `Transpose -> Transpose` pairs when permutations compose to
  identity.
- Remove statically provable no-op `Reshape` nodes when input and output shapes
  match after shape inference.

## Report-Only Detectors

These detectors do not rewrite the graph in the MVP:

- `MatMul + Add`
- GELU-like regions, including `Erf`-based GELU
- LayerNorm-like subgraphs
- Attention-like MatMul/Softmax regions
- Transpose/Reshape chains around attention
- Cast-heavy regions

Report-only findings are labeled as future compiler-pass opportunities.

## ONNX-MLIR

`python -m tcw lower ...` uses `onnx-mlir` only if the binary is installed on
`PATH`. If it is missing, the command writes `ONNX_MLIR_SETUP.md` and
`lowering.json` with a skipped status. Unit tests do not require ONNX-MLIR.

## Tests

```bash
pytest
ruff check .
ruff format --check .
```

The test suite is CPU-safe and uses tiny synthetic ONNX graphs.

## Optional GPU Measurement

The repo does not require a GPU. On a GPU machine with `onnxruntime-gpu`
installed and a visible CUDA provider, validation can be rerun with:

```bash
python -m tcw validate artifacts/model.onnx artifacts/model.opt.onnx \
  --out reports/validate.cuda.json \
  --sample-inputs artifacts/model.inputs.npz \
  --provider CUDAExecutionProvider
```

If the requested provider is unavailable, the tool falls back to
`CPUExecutionProvider` and records the available providers in the JSON report.

## Potential Upstream PRs

- ONNX-MLIR: add transformer-block lowering test case.
- ONNX-MLIR: add canonicalization test for redundant reshape/transpose pattern.
- ONNX Runtime: improve transformer optimizer report/debug visibility.
- NVIDIA TensorRT / onnx-graphsurgeon docs: example graph surgery workflow for
  transformer blocks.

## Design Notes

- macOS/CPU first.
- No CUDA, TensorRT, or NVIDIA GPU requirement.
- Netron-compatible ONNX artifacts.
- Standard ONNX operators in generated and rewritten graphs.
- No custom fused operators in the MVP.
- No external services.
