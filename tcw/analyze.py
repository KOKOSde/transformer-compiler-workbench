from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import Any

import onnx

from tcw.io import write_json, write_text
from tcw.onnx_helpers import consumers, load_model, value_shapes

SHAPE_ONLY_OPS = {
    "Cast",
    "Concat",
    "Constant",
    "Gather",
    "Reshape",
    "Shape",
    "Squeeze",
    "Transpose",
    "Unsqueeze",
}


def _op_histogram(model: onnx.ModelProto) -> Counter[str]:
    return Counter(node.op_type for node in model.graph.node)


def _shape_summary(model: onnx.ModelProto) -> dict[str, Any]:
    shapes = value_shapes(model)
    return {
        "inputs": {
            value.name: shapes.get(value.name, []) for value in model.graph.input
        },
        "outputs": {
            value.name: shapes.get(value.name, []) for value in model.graph.output
        },
    }


def _matmul_add_patterns(model: onnx.ModelProto) -> list[dict[str, str]]:
    by_input = consumers(model)
    patterns = []
    for node in model.graph.node:
        if node.op_type != "MatMul" or not node.output:
            continue
        for consumer in by_input.get(node.output[0], []):
            if consumer.op_type == "Add":
                patterns.append({"matmul": node.name, "add": consumer.name})
    return patterns


def _gelu_patterns(model: onnx.ModelProto) -> list[dict[str, str]]:
    patterns = []
    nodes = list(model.graph.node)
    for index, node in enumerate(nodes):
        if node.op_type in {"Gelu", "FastGelu"}:
            patterns.append({"kind": node.op_type, "node": node.name})
        if node.op_type == "Erf":
            window = nodes[max(0, index - 3) : min(len(nodes), index + 5)]
            if any(candidate.op_type == "Mul" for candidate in window):
                patterns.append({"kind": "ErfGeluLike", "node": node.name})
    return patterns


def _matmul_add_gelu_patterns(model: onnx.ModelProto) -> list[dict[str, Any]]:
    by_input = consumers(model)
    patterns = []
    for matmul in model.graph.node:
        if matmul.op_type != "MatMul" or not matmul.output:
            continue
        add_nodes = [
            node for node in by_input.get(matmul.output[0], []) if node.op_type == "Add"
        ]
        for add in add_nodes:
            queue = deque([(add, 0)])
            seen = {add.name}
            while queue:
                node, depth = queue.popleft()
                if depth > 6 or not node.output:
                    continue
                if node.op_type in {"Gelu", "FastGelu", "Erf"} and node is not add:
                    patterns.append(
                        {
                            "matmul": matmul.name,
                            "add": add.name,
                            "activation": node.name,
                            "kind": node.op_type,
                            "report_only": True,
                        }
                    )
                    break
                for output in node.output:
                    for next_node in by_input.get(output, []):
                        if next_node.name not in seen:
                            seen.add(next_node.name)
                            queue.append((next_node, depth + 1))
    return patterns


def _layer_norm_patterns(model: onnx.ModelProto) -> list[dict[str, str]]:
    patterns = []
    nodes = list(model.graph.node)
    for index, node in enumerate(nodes):
        if node.op_type != "ReduceMean":
            continue
        window = nodes[index : index + 12]
        ops = {candidate.op_type for candidate in window}
        if {"Sub", "Sqrt", "Div", "Mul", "Add"}.issubset(ops):
            patterns.append({"kind": "LayerNormLike", "start": node.name})
    return patterns


def _attention_patterns(model: onnx.ModelProto) -> list[dict[str, Any]]:
    matmuls = [node for node in model.graph.node if node.op_type == "MatMul"]
    softmaxes = [node for node in model.graph.node if node.op_type == "Softmax"]
    if len(matmuls) >= 4 and softmaxes:
        return [
            {
                "kind": "AttentionLike",
                "matmul_count": len(matmuls),
                "softmax_nodes": [node.name for node in softmaxes],
                "report_only": True,
            }
        ]
    return []


def _transpose_reshape_chains(model: onnx.ModelProto) -> list[dict[str, Any]]:
    by_input = consumers(model)
    chains = []
    for node in model.graph.node:
        if node.op_type not in {"Transpose", "Reshape"} or not node.output:
            continue
        chain = [node]
        cursor = node
        while cursor.output:
            next_nodes = [
                item
                for item in by_input.get(cursor.output[0], [])
                if item.op_type in {"Transpose", "Reshape"}
            ]
            if len(next_nodes) != 1:
                break
            cursor = next_nodes[0]
            chain.append(cursor)
        if len(chain) > 1:
            chains.append(
                {
                    "nodes": [item.name for item in chain],
                    "ops": [item.op_type for item in chain],
                    "report_only": True,
                }
            )
    return chains


def _cast_heavy_regions(model: onnx.ModelProto) -> list[dict[str, Any]]:
    nodes = list(model.graph.node)
    regions = []
    for index in range(len(nodes)):
        window = nodes[index : index + 8]
        cast_nodes = [node.name for node in window if node.op_type == "Cast"]
        if len(cast_nodes) >= 3:
            regions.append({"start_index": index, "cast_nodes": cast_nodes})
    return regions


def _longest_shape_only_chain(model: onnx.ModelProto) -> dict[str, Any]:
    by_input = consumers(model)
    best: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.op_type not in SHAPE_ONLY_OPS:
            continue
        queue: deque[list[onnx.NodeProto]] = deque([[node]])
        while queue:
            chain = queue.popleft()
            if len(chain) > len(best):
                best = chain
            tail = chain[-1]
            for output in tail.output:
                for next_node in by_input.get(output, []):
                    if next_node.op_type in SHAPE_ONLY_OPS and next_node not in chain:
                        queue.append([*chain, next_node])
    return {
        "length": len(best),
        "nodes": [node.name for node in best],
        "ops": [node.op_type for node in best],
    }


def _opportunity_report(model: onnx.ModelProto, histogram: Counter[str]) -> list[str]:
    opportunities = []
    if histogram["Identity"]:
        opportunities.append("Remove Identity nodes when outputs are rewired safely.")
    if histogram["Dropout"]:
        opportunities.append(
            "Remove inference Dropout nodes when mask output is unused."
        )
    if histogram["Cast"] >= 2:
        opportunities.append("Collapse Cast chains when source and final dtype match.")
    if histogram["Transpose"] >= 2:
        opportunities.append(
            "Cancel adjacent Transpose pairs when permutations invert."
        )
    if _matmul_add_patterns(model):
        opportunities.append(
            "Report-only: MatMul + Add can feed future linear fusion passes."
        )
    if _matmul_add_gelu_patterns(model):
        opportunities.append(
            "Report-only: MatMul + Add + GELU-like regions can feed future "
            "MLP fusion passes."
        )
    if _gelu_patterns(model):
        opportunities.append(
            "Report-only: GELU-like regions can feed future activation fusion."
        )
    if _layer_norm_patterns(model):
        opportunities.append(
            "Report-only: LayerNorm-like subgraphs can feed canonicalization."
        )
    return opportunities


def analyze_model(path: str | Path) -> dict[str, Any]:
    model = load_model(path)
    histogram = _op_histogram(model)
    selected = ["Cast", "Reshape", "Transpose", "Identity", "Dropout", "Constant"]
    patterns = {
        "matmul_add": _matmul_add_patterns(model),
        "matmul_add_gelu_like": _matmul_add_gelu_patterns(model),
        "gelu_like": _gelu_patterns(model),
        "layer_norm_like": _layer_norm_patterns(model),
        "attention_like": _attention_patterns(model),
        "transpose_reshape_chains": _transpose_reshape_chains(model),
        "cast_heavy_regions": _cast_heavy_regions(model),
    }
    return {
        "model_path": str(path),
        "graph_name": model.graph.name,
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_histogram": dict(sorted(histogram.items())),
        "selected_op_counts": {op: histogram[op] for op in selected},
        "shapes": _shape_summary(model),
        "patterns": patterns,
        "longest_shape_only_chain": _longest_shape_only_chain(model),
        "suspected_opportunities": _opportunity_report(model, histogram),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# ONNX Graph Analysis: `{Path(report['model_path']).name}`",
        "",
        f"- Nodes: {report['node_count']}",
        f"- Initializers: {report['initializer_count']}",
        "",
        "## Op Histogram",
        "",
        "| Op | Count |",
        "|---|---:|",
    ]
    for op_type, count in report["op_histogram"].items():
        lines.append(f"| {op_type} | {count} |")
    lines.extend(["", "## Suspected Opportunities", ""])
    for item in report["suspected_opportunities"]:
        lines.append(f"- {item}")
    if not report["suspected_opportunities"]:
        lines.append("- None detected.")
    lines.extend(["", "## Pattern Counts", ""])
    for name, values in report["patterns"].items():
        lines.append(f"- {name}: {len(values)}")
    lines.append("")
    return "\n".join(lines)


def write_analysis(path: str | Path, out: str | Path) -> dict[str, Any]:
    report = analyze_model(path)
    write_json(out, report)
    markdown_path = Path(out).with_suffix(".md")
    write_text(markdown_path, _markdown(report))
    return report
