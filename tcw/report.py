from __future__ import annotations

from pathlib import Path
from typing import Any

from tcw.io import read_json, write_text
from tcw.visualize import (
    write_benchmark_chart,
    write_provider_speedup_chart,
    write_visual_assets,
)


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
    benchmark = _find_report(reports_dir, "benchmark.json")
    if benchmark is None:
        for candidate in sorted(reports_dir.glob("benchmark*.json")):
            benchmark = read_json(candidate)
            break
    lowering = None
    for candidate in reports_dir.glob("**/lowering.json"):
        lowering = read_json(candidate)
        break

    lines = [
        "# Transformer Compiler Workbench Report",
        "",
        "This report is CPU-first. It does not claim GPU or CUDA validation.",
        "",
        "![Compiler workbench pipeline](assets/pipeline.svg)",
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

    if opt:
        write_visual_assets(reports_dir, opt)
        lines.extend(
            [
                "",
                "![Node count by graph](assets/node_counts.svg)",
                "",
                "![Custom rewrite pass effects](assets/pass_effects.svg)",
                "",
                "![ORT graph rewrite footprint](assets/ort_op_delta.svg)",
                "",
            ]
        )

    if benchmark:
        assets = reports_dir / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        write_benchmark_chart(benchmark, assets / "benchmark_latency.svg")
        lines.extend(
            [
                "",
                "## Provider Benchmark",
                "",
                "![Provider latency benchmark](assets/benchmark_latency.svg)",
                "",
                "| Provider | Graph | Effective provider | p50 latency | Parity vs first graph |",
                "|---|---|---|---:|---|",
            ]
        )
        for provider, provider_report in benchmark.get("providers", {}).items():
            models = provider_report.get("models", {})
            labels = [item["label"] for item in benchmark.get("models", [])]
            for label in labels:
                result = models.get(label, {})
                if "error" in result:
                    lines.append(f"| {provider} | {label} | error |  | false |")
                    continue
                latency = result.get("latency_ms", {})
                parity = result.get("output_parity_vs_first_model", {}).get("parity")
                lines.append(
                    f"| {provider} | {label} | {result.get('effective_provider')} | "
                    f"{latency.get('p50', 0):.3f} ms | {parity} |"
                )
        speedups = benchmark.get("provider_speedups_vs_cpu", {})
        if speedups:
            write_provider_speedup_chart(benchmark, assets / "provider_speedups.svg")
            lines.extend(
                [
                    "",
                    "![Provider speedup vs CPU](assets/provider_speedups.svg)",
                    "",
                    "| Provider | Graph | p50 speedup vs CPU |",
                    "|---|---|---:|",
                ]
            )
            labels = [item["label"] for item in benchmark.get("models", [])]
            for provider, provider_speedups in speedups.items():
                for label in labels:
                    value = provider_speedups.get(label)
                    if value is not None:
                        lines.append(f"| {provider} | {label} | {value:.2f}x |")

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
