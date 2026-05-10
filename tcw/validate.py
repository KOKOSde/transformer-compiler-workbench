from __future__ import annotations

import statistics
import time
import platform
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from tcw.io import write_json
from tcw.onnx_helpers import input_feeds, load_model


def _session(
    path: str | Path, provider: str = "CPUExecutionProvider"
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    providers = [provider]
    if provider not in ort.get_available_providers():
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=providers,
    )


def load_npz_inputs(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def run_outputs(
    path: str | Path,
    feeds: dict[str, np.ndarray],
    *,
    provider: str = "CPUExecutionProvider",
) -> list[np.ndarray]:
    session = _session(path, provider=provider)
    session_inputs = {item.name for item in session.get_inputs()}
    filtered = {key: value for key, value in feeds.items() if key in session_inputs}
    return session.run(None, filtered)


def compare_outputs(
    expected: list[np.ndarray],
    actual: list[np.ndarray],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    if len(expected) != len(actual):
        return {
            "parity": False,
            "reason": f"output count mismatch: {len(expected)} != {len(actual)}",
            "max_abs_diff": None,
            "max_rel_diff": None,
        }
    max_abs = 0.0
    max_rel = 0.0
    for left, right in zip(expected, actual):
        if left.shape != right.shape:
            return {
                "parity": False,
                "reason": f"shape mismatch: {left.shape} != {right.shape}",
                "max_abs_diff": None,
                "max_rel_diff": None,
            }
        diff = np.abs(left - right)
        denom = np.maximum(np.abs(left), 1e-12)
        max_abs = max(max_abs, float(diff.max()) if diff.size else 0.0)
        rel = diff / denom
        max_rel = max(max_rel, float(rel.max()) if rel.size else 0.0)
    return {
        "parity": bool(max_abs <= atol or max_rel <= rtol),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "atol": atol,
        "rtol": rtol,
    }


def latency_ms(
    path: str | Path,
    feeds: dict[str, np.ndarray],
    *,
    provider: str = "CPUExecutionProvider",
    warmup: int = 3,
    runs: int = 20,
) -> dict[str, float]:
    session = _session(path, provider=provider)
    session_inputs = {item.name for item in session.get_inputs()}
    filtered = {key: value for key, value in feeds.items() if key in session_inputs}
    for _ in range(warmup):
        session.run(None, filtered)
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, filtered)
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "p50": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "runs": float(runs),
    }


def validate_models(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    sample_input_path: str | Path | None = None,
    out: str | Path | None = None,
    provider: str = "CPUExecutionProvider",
) -> dict[str, Any]:
    baseline_model = load_model(baseline_path)
    onnx.checker.check_model(baseline_model)
    onnx.checker.check_model(load_model(candidate_path))

    feeds = (
        load_npz_inputs(sample_input_path)
        if sample_input_path is not None
        else input_feeds(baseline_model)
    )
    baseline_outputs = run_outputs(baseline_path, feeds, provider=provider)
    candidate_outputs = run_outputs(candidate_path, feeds, provider=provider)
    comparison = compare_outputs(baseline_outputs, candidate_outputs)
    report = {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "sample_input_path": str(sample_input_path) if sample_input_path else None,
        "runtime": {
            "requested_provider": provider,
            "available_providers": ort.get_available_providers(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "output_parity": comparison,
        "latency_ms": {
            "baseline": latency_ms(baseline_path, feeds, provider=provider),
            "candidate": latency_ms(candidate_path, feeds, provider=provider),
        },
    }
    if out is not None:
        write_json(out, report)
    return report
