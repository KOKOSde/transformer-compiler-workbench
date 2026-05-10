from __future__ import annotations

import argparse
from pathlib import Path

from tcw.analyze import write_analysis
from tcw.benchmark import benchmark_models
from tcw.export import export_model
from tcw.lower import lower_model
from tcw.optimize import optimize_model
from tcw.report import generate_report
from tcw.validate import validate_models


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcw",
        description="Transformer Compiler Workbench for ONNX/MLIR graph workflows.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    export = subcommands.add_parser("export", help="Export a reproducible ONNX model.")
    export.add_argument("--model", default="distilbert-base-uncased")
    export.add_argument("--out", required=True)
    export.add_argument(
        "--preset",
        choices=["tiny", "benchmark"],
        default="tiny",
        help="Export shape preset. 'tiny' is for quick compiler-pass demos.",
    )
    export.add_argument("--batch", type=int)
    export.add_argument("--seq", type=int)
    export.add_argument("--hidden", type=int)
    export.add_argument("--ffn", type=int)
    export.add_argument("--layers", type=int)

    analyze = subcommands.add_parser("analyze", help="Analyze an ONNX graph.")
    analyze.add_argument("model")
    analyze.add_argument("--out", required=True)

    optimize = subcommands.add_parser("optimize", help="Run ORT and custom rewrites.")
    optimize.add_argument("model")
    optimize.add_argument("--out", required=True)
    optimize.add_argument("--report", required=True)
    optimize.add_argument("--sample-inputs")
    optimize.add_argument("--provider", default="CPUExecutionProvider")

    validate = subcommands.add_parser("validate", help="Validate output parity.")
    validate.add_argument("baseline")
    validate.add_argument("candidate")
    validate.add_argument("--out", required=True)
    validate.add_argument("--sample-inputs")
    validate.add_argument("--provider", default="CPUExecutionProvider")

    benchmark = subcommands.add_parser(
        "benchmark", help="Measure ORT latency and parity for one or more models."
    )
    benchmark.add_argument("models", nargs="+")
    benchmark.add_argument("--sample-inputs", required=True)
    benchmark.add_argument("--out", required=True)
    benchmark.add_argument("--labels", nargs="+")
    benchmark.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="Requested ONNX Runtime providers to measure.",
    )
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--runs", type=int, default=50)

    lower = subcommands.add_parser("lower", help="Run optional ONNX-MLIR lowering.")
    lower.add_argument("model")
    lower.add_argument("--emit-dir", required=True)

    report = subcommands.add_parser("report", help="Generate Markdown summary.")
    report.add_argument("--reports", required=True)
    report.add_argument("--out", required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "export":
        result = export_model(
            args.model,
            args.out,
            preset=args.preset,
            batch=args.batch,
            seq=args.seq,
            hidden=args.hidden,
            ffn=args.ffn,
            layers=args.layers,
        )
        print(f"Wrote ONNX model: {result['onnx_path']}")
        print(f"Wrote sample inputs: {result['sample_input_path']}")
        return
    if args.command == "analyze":
        write_analysis(args.model, args.out)
        print(f"Wrote analysis: {args.out}")
        print(f"Wrote Markdown: {Path(args.out).with_suffix('.md')}")
        return
    if args.command == "optimize":
        optimize_model(
            args.model,
            args.out,
            report_path=args.report,
            sample_input_path=args.sample_inputs,
            provider=args.provider,
        )
        print(f"Wrote optimized model: {args.out}")
        print(f"Wrote optimization report: {args.report}")
        return
    if args.command == "validate":
        validate_models(
            args.baseline,
            args.candidate,
            sample_input_path=args.sample_inputs,
            out=args.out,
            provider=args.provider,
        )
        print(f"Wrote validation report: {args.out}")
        return
    if args.command == "benchmark":
        benchmark_models(
            args.models,
            sample_input_path=args.sample_inputs,
            providers=args.providers,
            labels=args.labels,
            out=args.out,
            warmup=args.warmup,
            runs=args.runs,
        )
        print(f"Wrote benchmark report: {args.out}")
        return
    if args.command == "lower":
        result = lower_model(args.model, args.emit_dir)
        print(f"Lowering status: {result['status']}")
        print(f"Wrote lowering report: {Path(args.emit_dir) / 'lowering.json'}")
        return
    if args.command == "report":
        generate_report(args.reports, args.out)
        print(f"Wrote report: {args.out}")
        return

    parser.error(f"Unhandled command: {args.command}")
