#!/usr/bin/env bash
set -euo pipefail

python -m tcw export --model distilbert-base-uncased --out artifacts/model.onnx
python -m tcw analyze artifacts/model.onnx --out reports/baseline.json
python -m tcw optimize artifacts/model.onnx \
  --out artifacts/model.opt.onnx \
  --report reports/opt.json \
  --sample-inputs artifacts/model.inputs.npz
python -m tcw validate artifacts/model.onnx artifacts/model.opt.onnx \
  --out reports/validate.json \
  --sample-inputs artifacts/model.inputs.npz
python -m tcw lower artifacts/model.opt.onnx --emit-dir artifacts/mlir
mkdir -p reports
cp artifacts/mlir/lowering.json reports/lowering.json
python -m tcw report --reports reports --out reports/index.md
