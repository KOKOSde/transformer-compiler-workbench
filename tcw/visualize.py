from __future__ import annotations

from pathlib import Path
from typing import Any

from tcw.io import write_text


COLORS = {
    "original": "#4f46e5",
    "ort": "#0891b2",
    "custom": "#16a34a",
    "accent": "#f59e0b",
    "text": "#111827",
    "muted": "#6b7280",
    "grid": "#e5e7eb",
}

BAR_COLORS = ["#4f46e5", "#0891b2", "#16a34a", "#f59e0b", "#db2777", "#475569"]


def _svg(width: int, height: int, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img">',
            '<rect width="100%" height="100%" fill="white"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def _text(x: int, y: int, content: str, *, size: int = 13, weight: int = 400) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Inter, ui-sans-serif, system-ui, '
        f'-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{COLORS["text"]}">'
        f"{content}</text>"
    )


def write_node_count_chart(graphs: dict[str, Any], out: Path) -> None:
    rows = [
        ("Original", graphs["original"]["node_count"], COLORS["original"]),
        ("ORT optimized", graphs["ort_optimized"]["node_count"], COLORS["ort"]),
        ("Custom rewrites", graphs["custom_optimized"]["node_count"], COLORS["custom"]),
    ]
    max_value = max(value for _, value, _ in rows)
    body = [_text(24, 32, "Node count by graph", size=18, weight=700)]
    for index, (label, value, color) in enumerate(rows):
        y = 70 + index * 46
        width = int((value / max_value) * 360)
        body.append(_text(24, y + 18, label))
        body.append(
            f'<rect x="150" y="{y}" width="{width}" height="24" rx="4" fill="{color}"/>'
        )
        body.append(_text(150 + width + 10, y + 18, str(value), weight=700))
    write_text(out, _svg(560, 220, body))


def write_pass_effect_chart(passes: list[dict[str, Any]], out: Path) -> None:
    max_value = max([item["changed"] for item in passes] + [1])
    body = [_text(24, 32, "Custom rewrite pass effects", size=18, weight=700)]
    for index, item in enumerate(passes):
        y = 66 + index * 38
        width = int((item["changed"] / max_value) * 300) if item["changed"] else 4
        color = COLORS["custom"] if item["changed"] else COLORS["grid"]
        body.append(_text(24, y + 17, item["name"].replace("_", " ")))
        body.append(
            f'<rect x="260" y="{y}" width="{width}" height="22" rx="4" fill="{color}"/>'
        )
        body.append(_text(270 + width, y + 17, str(item["changed"]), weight=700))
    write_text(out, _svg(620, max(260, 86 + len(passes) * 38), body))


def write_op_delta_chart(opt_report: dict[str, Any], out: Path) -> None:
    changes = opt_report.get("op_changes", {}).get("ort", {})
    removed = changes.get("removed_ops", {})
    added = changes.get("added_ops", {})
    rows = [(f"-{op}", count, "#dc2626") for op, count in sorted(removed.items())] + [
        (f"+{op}", count, "#2563eb") for op, count in sorted(added.items())
    ]
    rows = rows[:10]
    body = [_text(24, 32, "ORT graph rewrite footprint", size=18, weight=700)]
    if not rows:
        body.append(_text(24, 70, "No op histogram changes detected.", size=14))
        write_text(out, _svg(640, 120, body))
        return
    max_value = max(value for _, value, _ in rows)
    for index, (label, value, color) in enumerate(rows):
        y = 66 + index * 34
        width = int((value / max_value) * 300)
        body.append(_text(24, y + 16, label))
        body.append(
            f'<rect x="180" y="{y}" width="{width}" height="20" rx="4" fill="{color}"/>'
        )
        body.append(_text(190 + width, y + 16, str(value), weight=700))
    write_text(out, _svg(640, 96 + len(rows) * 34, body))


def write_pipeline_diagram(out: Path) -> None:
    stages = [
        ("Export", "standard ONNX graph"),
        ("Analyze", "patterns + opportunities"),
        ("ORT", "offline optimized graph"),
        ("Rewrite", "safe custom passes"),
        ("Validate", "checker + parity"),
        ("Lower", "optional ONNX-MLIR"),
    ]
    body = [_text(24, 32, "Compiler workbench pipeline", size=18, weight=700)]
    x = 24
    for index, (title, subtitle) in enumerate(stages):
        y = 70
        body.append(
            f'<rect x="{x}" y="{y}" width="142" height="76" rx="8" fill="#f9fafb" stroke="{COLORS["grid"]}"/>'
        )
        body.append(_text(x + 14, y + 30, title, size=15, weight=700))
        body.append(
            f'<text x="{x + 14}" y="{y + 54}" font-family="Inter, ui-sans-serif, system-ui" font-size="11" fill="{COLORS["muted"]}">{subtitle}</text>'
        )
        if index != len(stages) - 1:
            body.append(
                f'<path d="M{x + 142} {y + 38} L{x + 170} {y + 38}" stroke="{COLORS["muted"]}" stroke-width="2"/>'
            )
            body.append(
                f'<path d="M{x + 170} {y + 38} l-7 -5 v10 z" fill="{COLORS["muted"]}"/>'
            )
        x += 170
    write_text(out, _svg(1060, 180, body))


def write_benchmark_chart(benchmark: dict[str, Any], out: Path) -> None:
    rows: list[tuple[str, float, str]] = []
    labels = [item["label"] for item in benchmark.get("models", [])]
    for provider, provider_report in benchmark.get("providers", {}).items():
        models = provider_report.get("models", {})
        for index, label in enumerate(labels):
            result = models.get(label, {})
            latency = result.get("latency_ms", {})
            p50 = latency.get("p50")
            if p50 is None:
                continue
            active_marker = "" if result.get("effective_provider") else " fallback"
            rows.append(
                (
                    f"{provider.replace('ExecutionProvider', '')}: {label}{active_marker}",
                    float(p50),
                    BAR_COLORS[index % len(BAR_COLORS)],
                )
            )

    body = [_text(24, 32, "Provider latency benchmark", size=18, weight=700)]
    if not rows:
        body.append(_text(24, 70, "No successful latency rows found.", size=14))
        write_text(out, _svg(720, 120, body))
        return

    max_value = max(value for _, value, _ in rows)
    for index, (label, value, color) in enumerate(rows):
        y = 66 + index * 36
        width = max(3, int((value / max_value) * 330))
        body.append(_text(24, y + 16, label, size=12))
        body.append(
            f'<rect x="310" y="{y}" width="{width}" height="20" rx="4" fill="{color}"/>'
        )
        body.append(_text(320 + width, y + 16, f"{value:.3f} ms", weight=700))
    write_text(out, _svg(780, 96 + len(rows) * 36, body))


def write_provider_speedup_chart(benchmark: dict[str, Any], out: Path) -> None:
    rows: list[tuple[str, float, str]] = []
    labels = [item["label"] for item in benchmark.get("models", [])]
    for provider, provider_speedups in benchmark.get(
        "provider_speedups_vs_cpu", {}
    ).items():
        for index, label in enumerate(labels):
            value = provider_speedups.get(label)
            if value is not None:
                rows.append(
                    (
                        f"{provider.replace('ExecutionProvider', '')}: {label}",
                        float(value),
                        BAR_COLORS[index % len(BAR_COLORS)],
                    )
                )

    body = [_text(24, 32, "Provider speedup vs CPU", size=18, weight=700)]
    if not rows:
        body.append(_text(24, 70, "No cross-provider speedups recorded.", size=14))
        write_text(out, _svg(720, 120, body))
        return

    max_value = max(value for _, value, _ in rows)
    for index, (label, value, color) in enumerate(rows):
        y = 66 + index * 36
        width = max(3, int((value / max_value) * 360))
        body.append(_text(24, y + 16, label, size=12))
        body.append(
            f'<rect x="260" y="{y}" width="{width}" height="20" rx="4" fill="{color}"/>'
        )
        body.append(_text(270 + width, y + 16, f"{value:.2f}x", weight=700))
    write_text(out, _svg(760, 96 + len(rows) * 36, body))


def write_visual_assets(
    reports_dir: str | Path, opt_report: dict[str, Any]
) -> list[Path]:
    reports_dir = Path(reports_dir)
    assets = reports_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    outputs = [
        assets / "pipeline.svg",
        assets / "node_counts.svg",
        assets / "pass_effects.svg",
        assets / "ort_op_delta.svg",
    ]
    write_pipeline_diagram(outputs[0])
    write_node_count_chart(opt_report["graphs"], outputs[1])
    write_pass_effect_chart(opt_report.get("rewrite_passes", []), outputs[2])
    write_op_delta_chart(opt_report, outputs[3])
    return outputs
