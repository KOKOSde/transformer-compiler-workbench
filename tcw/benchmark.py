from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort

from tcw.analyze import analyze_model
from tcw.io import write_json
from tcw.validate import compare_outputs, latency_ms, load_npz_inputs, run_outputs


def _model_summary(path: Path, label: str) -> dict[str, Any]:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    analysis = analyze_model(path)
    return {
        "label": label,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "node_count": analysis["node_count"],
        "selected_op_counts": analysis["selected_op_counts"],
    }


def _provider_active(requested_provider: str, session_providers: list[str]) -> bool:
    if requested_provider == "CPUExecutionProvider":
        return "CPUExecutionProvider" in session_providers
    return requested_provider in session_providers


def benchmark_models(
    model_paths: list[str | Path],
    *,
    sample_input_path: str | Path,
    providers: list[str],
    labels: list[str] | None = None,
    out: str | Path | None = None,
    warmup: int = 10,
    runs: int = 50,
) -> dict[str, Any]:
    if not model_paths:
        raise ValueError("at least one model path is required")
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    paths = [Path(path) for path in model_paths]
    resolved_labels = labels or [path.stem for path in paths]
    if len(resolved_labels) != len(paths):
        raise ValueError("labels length must match model paths length")

    feeds = load_npz_inputs(sample_input_path)
    model_rows = [
        _model_summary(path, label) for path, label in zip(paths, resolved_labels)
    ]

    report: dict[str, Any] = {
        "sample_input_path": str(sample_input_path),
        "runtime": {
            "available_providers": ort.get_available_providers(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "models": model_rows,
        "providers": {},
    }

    for provider in providers:
        provider_report = _benchmark_provider(
            paths,
            resolved_labels,
            feeds,
            provider=provider,
            warmup=warmup,
            runs=runs,
        )
        report["providers"][provider] = provider_report

    if out is not None:
        write_json(out, report)
    return report


def _benchmark_provider(
    paths: list[Path],
    labels: list[str],
    feeds: dict[str, Any],
    *,
    provider: str,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    try:
        baseline_outputs = run_outputs(paths[0], feeds, provider=provider)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    rows: dict[str, Any] = {}
    baseline_p50: float | None = None
    baseline_active = False
    for index, (path, label) in enumerate(zip(paths, labels)):
        try:
            current_outputs = (
                baseline_outputs
                if index == 0
                else run_outputs(path, feeds, provider=provider)
            )
            timing = latency_ms(
                path,
                feeds,
                provider=provider,
                warmup=warmup,
                runs=runs,
            )
            active = _provider_active(provider, timing["session_providers"])
            if index == 0:
                baseline_p50 = float(timing["p50"])
                baseline_active = active
            speedup = None
            if baseline_p50 and active and baseline_active:
                speedup = baseline_p50 / float(timing["p50"])
            rows[label] = {
                "latency_ms": timing,
                "effective_provider": active,
                "speedup_vs_first_model": speedup,
                "output_parity_vs_first_model": compare_outputs(
                    baseline_outputs, current_outputs
                ),
            }
        except Exception as exc:
            rows[label] = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "requested_provider": provider,
        "models": rows,
    }
