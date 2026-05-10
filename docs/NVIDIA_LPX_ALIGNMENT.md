# NVIDIA LPX Alignment

This project is built to demonstrate the parts of inference compiler work that
are hard to show with a generic model-serving demo.

## What It Shows

- Portable graph work with ONNX instead of framework-only code.
- Compiler-style analysis: op histograms, pattern detectors, shape-only chains,
  and report-only future fusion opportunities.
- Runtime/compiler boundary work: ONNX Runtime offline optimization and custom
  graph rewrites are measured separately.
- Correctness discipline: every rewrite must pass ONNX checker and ORT output
  parity.
- Multilevel IR awareness: ONNX-MLIR is integrated as an optional lowering
  target when the external compiler is installed.
- Performance discipline: latency reports include runtime/provider provenance
  and do not claim GPU validation unless run with a GPU provider.

## Why It Maps To LPX

LPX-style inference work sits between model graphs, runtime execution,
compiler passes, and hardware feedback. The first version of this workbench is
small, but it exercises that interface:

1. Start with a transformer-like inference graph.
2. Identify graph regions that are compiler opportunities.
3. Run a production runtime optimizer.
4. Apply conservative graph rewrites with validation.
5. Optionally lower to MLIR for deeper compiler inspection.
6. Produce a report that an ML systems engineer can use to debug what changed.

## What Is Deliberately Not Claimed

- No TensorRT path is claimed.
- No CUDA path is required.
- No GPU result is reported unless collected on a machine with an installed GPU
  ONNX Runtime provider.
- No custom fused operators are introduced in the MVP.
- No speedup claim is made without showing graph, provider, and parity context.

## Next Compiler-Focused Extensions

- Add a real ONNX-MLIR test case for a transformer block.
- Add canonicalization tests for transpose/reshape chains around attention.
- Emit a pass-decision trace that explains why each rewrite was or was not
  applied.
- Add optional CUDAExecutionProvider benchmarking on GPU machines.
- Add an MLIR pattern-matching playground for LayerNorm and MLP blocks.
