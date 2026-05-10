from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tcw.analyze import analyze_model, write_analysis
from tcw.benchmark import benchmark_models
from tcw.io import read_json
from tcw.lower import lower_model
from tcw.optimize import optimize_model
from tcw.report import generate_report
from tcw.rewrite import rewrite_model
from tcw.validate import compare_outputs, validate_models


def _save(model: onnx.ModelProto, path: Path) -> Path:
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def _linear_graph(path: Path) -> Path:
    weight = numpy_helper.from_array(np.eye(2, dtype=np.float32), name="w")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="matmul")],
        "linear",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        [weight],
    )
    return _save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)]),
        path,
    )


def _identity_graph(path: Path) -> Path:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"], name="identity")],
        "identity",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
    )
    return _save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)]),
        path,
    )


def _transpose_pair_graph(path: Path) -> Path:
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["x"], ["t1"], name="t1", perm=[0, 2, 1]),
            helper.make_node("Transpose", ["t1"], ["y"], name="t2", perm=[0, 2, 1]),
        ],
        "transpose_pair",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3])],
    )
    return _save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)]),
        path,
    )


def _cast_chain_graph(path: Path, final_type: int = TensorProto.FLOAT) -> Path:
    graph = helper.make_graph(
        [
            helper.make_node(
                "Cast", ["x"], ["c1"], name="cast_to_double", to=TensorProto.DOUBLE
            ),
            helper.make_node("Cast", ["c1"], ["y"], name="cast_final", to=final_type),
        ],
        "cast_chain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", final_type, [1, 2])],
    )
    return _save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)]),
        path,
    )


def test_graph_analyzer_works_on_tiny_synthetic_graph(tmp_path: Path):
    model_path = _linear_graph(tmp_path / "linear.onnx")
    report = analyze_model(model_path)
    assert report["node_count"] == 1
    assert report["op_histogram"]["MatMul"] == 1
    assert report["initializer_count"] == 1


def test_identity_removal_preserves_output(tmp_path: Path):
    model_path = _identity_graph(tmp_path / "identity.onnx")
    out_path = tmp_path / "identity.opt.onnx"
    rewrite_model(model_path, out_path)
    onnx.checker.check_model(onnx.load(out_path))
    report = validate_models(model_path, out_path)
    assert report["output_parity"]["parity"]
    assert analyze_model(out_path)["op_histogram"].get("Identity", 0) == 0


def test_canceling_transpose_pair_preserves_output(tmp_path: Path):
    model_path = _transpose_pair_graph(tmp_path / "transpose.onnx")
    out_path = tmp_path / "transpose.opt.onnx"
    rewrite_model(model_path, out_path)
    onnx.checker.check_model(onnx.load(out_path))
    report = validate_models(model_path, out_path)
    assert report["output_parity"]["parity"]
    assert analyze_model(out_path)["op_histogram"].get("Transpose", 0) == 0


def test_cast_chain_only_rewrites_safe_cases(tmp_path: Path):
    safe_path = _cast_chain_graph(tmp_path / "safe.onnx")
    safe_out = tmp_path / "safe.opt.onnx"
    rewrite_model(safe_path, safe_out)
    assert analyze_model(safe_out)["op_histogram"].get("Cast", 0) == 0
    assert validate_models(safe_path, safe_out)["output_parity"]["parity"]

    unsafe_path = _cast_chain_graph(
        tmp_path / "unsafe.onnx", final_type=TensorProto.INT64
    )
    unsafe_out = tmp_path / "unsafe.opt.onnx"
    rewrite_model(unsafe_path, unsafe_out)
    assert analyze_model(unsafe_out)["op_histogram"].get("Cast", 0) == 2


def test_optimizer_output_passes_checker(tmp_path: Path):
    model_path = _identity_graph(tmp_path / "identity.onnx")
    out_path = tmp_path / "identity.opt.onnx"
    report_path = tmp_path / "opt.json"
    optimize_model(model_path, out_path, report_path=report_path)
    onnx.checker.check_model(onnx.load(out_path))
    assert read_json(report_path)["validation"]["custom"]["output_parity"]["parity"]


def test_benchmark_generates_cpu_report(tmp_path: Path):
    model_path = _linear_graph(tmp_path / "linear.onnx")
    sample_path = tmp_path / "inputs.npz"
    np.savez(sample_path, x=np.array([[1.0, 2.0]], dtype=np.float32))
    report_path = tmp_path / "benchmark.json"

    report = benchmark_models(
        [model_path],
        sample_input_path=sample_path,
        providers=["CPUExecutionProvider"],
        labels=["linear"],
        out=report_path,
        warmup=0,
        runs=1,
    )

    result = report["providers"]["CPUExecutionProvider"]["models"]["linear"]
    assert report_path.exists()
    assert result["effective_provider"]
    assert result["output_parity_vs_first_model"]["parity"]
    assert result["latency_ms"]["p50"] >= 0


def test_validation_detects_numerical_mismatch():
    left = [np.array([1.0], dtype=np.float32)]
    right = [np.array([2.0], dtype=np.float32)]
    result = compare_outputs(left, right, atol=1e-6, rtol=1e-6)
    assert not result["parity"]


def test_report_generation_works(tmp_path: Path):
    model_path = _linear_graph(tmp_path / "linear.onnx")
    reports = tmp_path / "reports"
    reports.mkdir()
    write_analysis(model_path, reports / "baseline.json")
    out = reports / "index.md"
    content = generate_report(reports, out)
    assert out.exists()
    assert "Graph Summary" in content


def test_onnx_mlir_integration_skips_cleanly_when_missing(tmp_path: Path, monkeypatch):
    model_path = _linear_graph(tmp_path / "linear.onnx")
    monkeypatch.setenv("PATH", "")
    result = lower_model(model_path, tmp_path / "mlir")
    assert result["status"] == "skipped"
    assert (tmp_path / "mlir" / "ONNX_MLIR_SETUP.md").exists()
