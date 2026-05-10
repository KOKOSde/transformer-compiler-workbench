from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from tcw.io import write_json, write_text


def _run_onnx_mlir(
    binary: str, model_path: Path, emit_dir: Path, flag: str
) -> dict[str, Any]:
    stem = model_path.stem
    out_base = emit_dir / f"{stem}.{flag.lower()}"
    command = [binary, f"--Emit{flag}", str(model_path), "-o", str(out_base)]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {"flag": flag, "ok": False, "error": str(exc), "command": command}
    outputs = sorted(str(path) for path in emit_dir.glob(f"{out_base.name}*"))
    return {
        "flag": flag,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "outputs": outputs,
        "command": command,
    }


def lower_model(model_path: str | Path, emit_dir: str | Path) -> dict[str, Any]:
    model_path = Path(model_path)
    emit_dir = Path(emit_dir)
    emit_dir.mkdir(parents=True, exist_ok=True)

    binary = shutil.which("onnx-mlir")
    if binary is None:
        instructions = "\n".join(
            [
                "# ONNX-MLIR lowering skipped",
                "",
                "`onnx-mlir` was not found on PATH.",
                "",
                "Install ONNX-MLIR from https://github.com/onnx/onnx-mlir and",
                "make sure the `onnx-mlir` binary is available on PATH, then rerun:",
                "",
                f"```bash\npython -m tcw lower {model_path} --emit-dir {emit_dir}\n```",
                "",
            ]
        )
        write_text(emit_dir / "ONNX_MLIR_SETUP.md", instructions)
        report = {
            "model_path": str(model_path),
            "status": "skipped",
            "reason": "onnx-mlir binary not found on PATH",
            "setup_instructions": str(emit_dir / "ONNX_MLIR_SETUP.md"),
        }
        write_json(emit_dir / "lowering.json", report)
        return report

    runs = [
        _run_onnx_mlir(binary, model_path, emit_dir, flag)
        for flag in ("ONNXIR", "MLIR", "LLVMIR")
    ]
    report = {
        "model_path": str(model_path),
        "status": "ok" if any(run["ok"] for run in runs) else "failed",
        "binary": binary,
        "runs": runs,
    }
    write_json(emit_dir / "lowering.json", report)
    return report
