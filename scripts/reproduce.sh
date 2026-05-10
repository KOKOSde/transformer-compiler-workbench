#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" -m tcw export --model distilbert-base-uncased --out artifacts/model.onnx
"$PYTHON_BIN" -m tcw analyze artifacts/model.onnx --out reports/baseline.json
"$PYTHON_BIN" -m tcw optimize artifacts/model.onnx \
  --out artifacts/model.opt.onnx \
  --report reports/opt.json \
  --sample-inputs artifacts/model.inputs.npz
"$PYTHON_BIN" -m tcw validate artifacts/model.onnx artifacts/model.opt.onnx \
  --out reports/validate.json \
  --sample-inputs artifacts/model.inputs.npz
"$PYTHON_BIN" -m tcw benchmark \
  artifacts/model.onnx artifacts/model.opt.onnx artifacts/model.opt.ort.onnx \
  --labels Original Custom ORT \
  --sample-inputs artifacts/model.inputs.npz \
  --providers CPUExecutionProvider \
  --out reports/benchmark.json \
  --warmup 3 \
  --runs 20
"$PYTHON_BIN" -m tcw lower artifacts/model.opt.onnx --emit-dir artifacts/mlir
mkdir -p reports
cp artifacts/mlir/lowering.json reports/lowering.json
"$PYTHON_BIN" -m tcw report --reports reports --out reports/index.md
