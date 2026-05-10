from __future__ import annotations

from pathlib import Path
from typing import Any

from tcw.io import read_json, write_text


def _find_report(reports_dir: Path, name: str) -> dict[str, Any] | None:
    path = reports_dir / name
    return read_json(path) if path.exists() else None


def _graph_row(name: str, graph: dict[str, Any], latency: Any, diff: Any) -> str:
    selected = graph.get("selected_op_counts", {})
    p50 = ""
    if isinstance(latency, dict):
        p50 = f"{latency.get('p50', 0):.3f} ms"
    max_diff = ""
    if isinstance(diff, dict) and diff.get("max_abs_diff") is not None:
        max_diff = f"{diff['max_abs_diff']:.3g}"
    return (
        f"| {name} | {graph.get('node_count', '')} | {selected.get('Cast', 0)} | "
        f"{selected.get('Transpose', 0)} | {selected.get('Reshape', 0)} | "
        f"{p50} | {max_diff} |"
    )


def generate_report(reports_dir: str | Path, out: str | Path) -> str:
    reports_dir = Path(reports_dir)
    baseline = _find_report(reports_dir, "baseline.json")
    opt = _find_report(reports_dir, "opt.json")
    validate = _find_report(reports_dir, "validate.json")
    lowering = None
    for candidate in reports_dir.glob("**/lowering.json"):
        lowering = read_json(candidate)
        break

    lines = [
        "# Transformer Compiler Workbench Report",
        "",
        "This report is CPU-first. It does not claim GPU or CUDA validation.",
        "",
        "## Graph Summary",
        "",
        "| Graph | Nodes | Cast | Transpose | Reshape | CPU latency p50 | Max output diff |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if baseline:
        lines.append(_graph_row("Original", baseline, None, {"max_abs_diff": 0.0}))
    if opt:
        graphs = opt.get("graphs", {})
        validation = opt.get("validation", {})
        if "ort_optimized" in graphs:
            lines.append(
                _graph_row(
                    "ORT optimized",
                    graphs["ort_optimized"],
                    validation.get("ort", {}).get("latency_ms", {}).get("candidate"),
                    validation.get("ort", {}).get("output_parity"),
                )
            )
        if "custom_optimized" in graphs:
            lines.append(
                _graph_row(
                    "Custom optimized",
                    graphs["custom_optimized"],
                    validation.get("custom", {}).get("latency_ms", {}).get("candidate"),
                    validation.get("custom", {}).get("output_parity"),
                )
            )
    elif validate and baseline:
        lines.append(
            _graph_row(
                "Candidate",
                baseline,
                validate.get("latency_ms", {}).get("candidate"),
                validate.get("output_parity"),
            )
        )

    lines.extend(["", "## Rewrite Passes", ""])
    if opt:
        for item in opt.get("rewrite_passes", []):
            lines.append(f"- `{item['name']}`: {item['changed']} rewrite(s)")
    else:
        lines.append("- No optimization report found.")

    lines.extend(["", "## Optimization Opportunities", ""])
    source = baseline or (opt or {}).get("graphs", {}).get("original")
    if source:
        for item in source.get("suspected_opportunities", []):
            lines.append(f"- {item}")
    else:
        lines.append("- No analysis report found.")

    lines.extend(["", "## Output Parity", ""])
    if validate:
        lines.append(
            f"- Standalone validation parity: {validate['output_parity'].get('parity')}"
        )
    if opt:
        lines.append(
            f"- ORT parity: {opt['validation']['ort']['output_parity'].get('parity')}"
        )
        lines.append(
            f"- Custom parity: {opt['validation']['custom']['output_parity'].get('parity')}"
        )

    lines.extend(["", "## ONNX-MLIR Lowering", ""])
    if lowering:
        lines.append(f"- Status: {lowering.get('status')}")
        if lowering.get("reason"):
            lines.append(f"- Reason: {lowering['reason']}")
    else:
        lines.append("- No lowering report found.")

    lines.extend(["", "## Report Files", ""])
    for path in sorted(reports_dir.glob("**/*")):
        if path.is_file():
            lines.append(f"- `{path}`")
    lines.append("")

    content = "\n".join(lines)
    write_text(out, content)
    return content
