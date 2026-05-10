# ONNX-MLIR lowering skipped

`onnx-mlir` was not found on PATH.

Install ONNX-MLIR from https://github.com/onnx/onnx-mlir and
make sure the `onnx-mlir` binary is available on PATH, then rerun:

```bash
python -m tcw lower artifacts/model.opt.onnx --emit-dir artifacts/mlir
```
