from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort

from tcw.analyze import analyze_model
from tcw.io import write_json
from tcw.rewrite import rewrite_model
from tcw.validate import validate_models


def _histogram_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(before) | set(after))
    return {key: after.get(key, 0) - before.get(key, 0) for key in keys}


def _op_changes(
    before: dict[str, int], after: dict[str, int]
) -> dict[str, dict[str, int]]:
    delta = _histogram_delta(before, after)
    return {
        "removed_ops": {key: -value for key, value in delta.items() if value < 0},
        "added_ops": {key: value for key, value in delta.items() if value > 0},
    }


def run_ort_offline_optimization(in_path: str | Path, out_path: str | Path) -> None:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(out_path)
    ort.InferenceSession(
        str(in_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    onnx.checker.check_model(onnx.load(str(out_path)))


def optimize_model(
    in_path: str | Path,
    out_path: str | Path,
    *,
    report_path: str | Path | None = None,
    sample_input_path: str | Path | None = None,
) -> dict[str, Any]:
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ort_path = out_path.with_suffix(".ort.onnx")

    original = analyze_model(in_path)
    run_ort_offline_optimization(in_path, ort_path)
    ort_report = analyze_model(ort_path)
    pass_results = rewrite_model(in_path, out_path)
    custom = analyze_model(out_path)

    validation_ort = validate_models(
        in_path,
        ort_path,
        sample_input_path=sample_input_path,
    )
    validation_custom = validate_models(
        in_path,
        out_path,
        sample_input_path=sample_input_path,
    )

    report = {
        "input_model": str(in_path),
        "ort_optimized_model": str(ort_path),
        "custom_optimized_model": str(out_path),
        "node_count_delta": {
            "ort": ort_report["node_count"] - original["node_count"],
            "custom": custom["node_count"] - original["node_count"],
        },
        "op_histogram_delta": {
            "ort": _histogram_delta(
                original["op_histogram"], ort_report["op_histogram"]
            ),
            "custom": _histogram_delta(
                original["op_histogram"], custom["op_histogram"]
            ),
        },
        "op_changes": {
            "ort": _op_changes(original["op_histogram"], ort_report["op_histogram"]),
            "custom": _op_changes(original["op_histogram"], custom["op_histogram"]),
        },
        "graphs": {
            "original": original,
            "ort_optimized": ort_report,
            "custom_optimized": custom,
        },
        "rewrite_passes": [result.as_dict() for result in pass_results],
        "validation": {
            "ort": validation_ort,
            "custom": validation_custom,
        },
    }
    if report_path is not None:
        write_json(report_path, report)
    return report
