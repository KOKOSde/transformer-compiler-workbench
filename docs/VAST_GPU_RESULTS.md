# Vast GPU Run

This run verifies that the workbench can execute on a rented NVIDIA GPU
instance while keeping the benchmark claims honest.

## Machine

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Driver: 590.48.01
- GPU memory: 97887 MiB
- OS: Linux 6.8.0-90-generic x86_64
- Python: 3.12.3

## Runtime

- `onnxruntime-gpu==1.26.0`
- `onnx==1.21.0`
- CUDA provider dependencies installed from Python wheels:
  - `nvidia-cublas-cu12==12.9.2.10`
  - `nvidia-cuda-runtime-cu12==12.9.79`
  - `nvidia-cudnn-cu12==9.22.0.52`
  - `nvidia-cufft-cu12==11.4.1.4`
  - `nvidia-curand-cu12==10.3.10.19`

## Command

```bash
LD_LIBRARY_PATH="$CUDA_LIBS:${LD_LIBRARY_PATH:-}" \
python -m tcw validate artifacts/model.onnx artifacts/model.opt.onnx \
  --out reports/validate.cuda.json \
  --sample-inputs artifacts/model.inputs.npz \
  --provider CUDAExecutionProvider
```

## Result

| Graph | Provider | p50 latency | Max output diff | Parity |
|---|---|---:|---:|---|
| Original | CUDAExecutionProvider + CPUExecutionProvider | 0.177 ms | 0 | true |
| Custom optimized | CUDAExecutionProvider + CPUExecutionProvider | 0.177 ms | 0 | true |

The graph is intentionally tiny, so GPU launch overhead dominates. This is not
presented as a speedup. The useful result is that the same artifacts validate
under an NVIDIA GPU provider stack with exact output parity, and the report
records the actual session providers.

Full JSON: [`reports/validate.cuda.json`](../reports/validate.cuda.json)
