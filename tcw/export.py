from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tcw.io import write_json
from tcw.onnx_helpers import save_checked_model


@dataclass(frozen=True)
class TransformerSpec:
    batch: int
    seq: int
    hidden: int
    ffn: int
    layers: int
    name: str


PRESETS = {
    "tiny": TransformerSpec(batch=1, seq=4, hidden=8, ffn=16, layers=1, name="tiny"),
    "benchmark": TransformerSpec(
        batch=8,
        seq=128,
        hidden=256,
        ffn=1024,
        layers=4,
        name="benchmark",
    ),
}


def _tensor(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array.astype(np.float32), name=name)


def _const(name: str, value: np.ndarray) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        [],
        [name],
        name=f"const_{name}",
        value=numpy_helper.from_array(value.astype(np.float32)),
    )


def _matmul_add(
    x: str,
    weight: str,
    bias: str,
    prefix: str,
) -> list[onnx.NodeProto]:
    return [
        helper.make_node(
            "MatMul", [x, weight], [f"{prefix}_matmul"], name=f"{prefix}_matmul"
        ),
        helper.make_node(
            "Add", [f"{prefix}_matmul", bias], [prefix], name=f"{prefix}_add"
        ),
    ]


def _layer_norm(x: str, prefix: str) -> list[onnx.NodeProto]:
    return [
        _const(f"{prefix}_two", np.array(2.0, dtype=np.float32)),
        _const(f"{prefix}_eps", np.array(1e-5, dtype=np.float32)),
        helper.make_node(
            "ReduceMean",
            [x],
            [f"{prefix}_mean"],
            name=f"{prefix}_mean",
            axes=[-1],
            keepdims=1,
        ),
        helper.make_node(
            "Sub", [x, f"{prefix}_mean"], [f"{prefix}_centered"], name=f"{prefix}_sub"
        ),
        helper.make_node(
            "Pow",
            [f"{prefix}_centered", f"{prefix}_two"],
            [f"{prefix}_square"],
            name=f"{prefix}_pow",
        ),
        helper.make_node(
            "ReduceMean",
            [f"{prefix}_square"],
            [f"{prefix}_var"],
            name=f"{prefix}_var",
            axes=[-1],
            keepdims=1,
        ),
        helper.make_node(
            "Add",
            [f"{prefix}_var", f"{prefix}_eps"],
            [f"{prefix}_var_eps"],
            name=f"{prefix}_add_eps",
        ),
        helper.make_node(
            "Sqrt", [f"{prefix}_var_eps"], [f"{prefix}_std"], name=f"{prefix}_sqrt"
        ),
        helper.make_node(
            "Div",
            [f"{prefix}_centered", f"{prefix}_std"],
            [f"{prefix}_norm"],
            name=f"{prefix}_div",
        ),
        helper.make_node(
            "Mul",
            [f"{prefix}_norm", f"{prefix}_gamma"],
            [f"{prefix}_scaled"],
            name=f"{prefix}_mul",
        ),
        helper.make_node(
            "Add",
            [f"{prefix}_scaled", f"{prefix}_beta"],
            [prefix],
            name=f"{prefix}_add_beta",
        ),
    ]


def _gelu_erf(x: str, prefix: str) -> list[onnx.NodeProto]:
    return [
        _const(f"{prefix}_sqrt2", np.array(1.41421356237, dtype=np.float32)),
        _const(f"{prefix}_one", np.array(1.0, dtype=np.float32)),
        _const(f"{prefix}_half", np.array(0.5, dtype=np.float32)),
        helper.make_node(
            "Div", [x, f"{prefix}_sqrt2"], [f"{prefix}_div"], name=f"{prefix}_div"
        ),
        helper.make_node(
            "Erf", [f"{prefix}_div"], [f"{prefix}_erf"], name=f"{prefix}_erf"
        ),
        helper.make_node(
            "Add",
            [f"{prefix}_erf", f"{prefix}_one"],
            [f"{prefix}_one_plus"],
            name=f"{prefix}_add",
        ),
        helper.make_node(
            "Mul",
            [x, f"{prefix}_one_plus"],
            [f"{prefix}_x_mul"],
            name=f"{prefix}_mul_x",
        ),
        helper.make_node(
            "Mul",
            [f"{prefix}_x_mul", f"{prefix}_half"],
            [prefix],
            name=f"{prefix}_mul_half",
        ),
    ]


def _add_layer_initializers(
    initializers: list[onnx.TensorProto],
    rng: np.random.Generator,
    spec: TransformerSpec,
    prefix: str,
) -> None:
    hidden = spec.hidden
    ffn = spec.ffn
    for name in ("q", "k", "v", "o"):
        initializers.append(
            _tensor(f"{prefix}_w_{name}", rng.normal(0, 0.02, size=(hidden, hidden)))
        )
        initializers.append(
            _tensor(f"{prefix}_b_{name}", rng.normal(0, 0.002, size=(hidden,)))
        )
    for name, shape in (("w_ff1", (hidden, ffn)), ("w_ff2", (ffn, hidden))):
        initializers.append(
            _tensor(f"{prefix}_{name}", rng.normal(0, 0.02, size=shape))
        )
    for name, shape in (("b_ff1", (ffn,)), ("b_ff2", (hidden,))):
        initializers.append(
            _tensor(f"{prefix}_{name}", rng.normal(0, 0.002, size=shape))
        )
    for ln_name in ("ln1", "ln2"):
        initializers.append(
            _tensor(f"{prefix}_{ln_name}_gamma", np.ones((hidden,), dtype=np.float32))
        )
        initializers.append(
            _tensor(f"{prefix}_{ln_name}_beta", np.zeros((hidden,), dtype=np.float32))
        )


def _transformer_layer(
    x: str, prefix: str, *, include_demo_rewrites: bool
) -> list[onnx.NodeProto]:
    nodes: list[onnx.NodeProto] = []
    layer_input = x
    if include_demo_rewrites:
        layer_input = f"{prefix}_input_identity"
        nodes.append(
            helper.make_node("Identity", [x], [layer_input], name=f"{prefix}_identity")
        )

    nodes.extend(
        _matmul_add(layer_input, f"{prefix}_w_q", f"{prefix}_b_q", f"{prefix}_q")
    )
    nodes.extend(
        _matmul_add(layer_input, f"{prefix}_w_k", f"{prefix}_b_k", f"{prefix}_k")
    )
    nodes.extend(
        _matmul_add(layer_input, f"{prefix}_w_v", f"{prefix}_b_v", f"{prefix}_v")
    )
    nodes.append(
        helper.make_node(
            "Transpose",
            [f"{prefix}_k"],
            [f"{prefix}_k_t"],
            name=f"{prefix}_k_transpose",
            perm=[0, 2, 1],
        )
    )
    nodes.append(
        helper.make_node(
            "MatMul",
            [f"{prefix}_q", f"{prefix}_k_t"],
            [f"{prefix}_attention_scores"],
            name=f"{prefix}_attention_scores",
        )
    )
    nodes.append(
        helper.make_node(
            "Softmax",
            [f"{prefix}_attention_scores"],
            [f"{prefix}_attention_probs"],
            name=f"{prefix}_attention_softmax",
            axis=-1,
        )
    )
    nodes.append(
        helper.make_node(
            "MatMul",
            [f"{prefix}_attention_probs", f"{prefix}_v"],
            [f"{prefix}_context"],
            name=f"{prefix}_attention_context",
        )
    )
    nodes.extend(
        _matmul_add(
            f"{prefix}_context",
            f"{prefix}_w_o",
            f"{prefix}_b_o",
            f"{prefix}_attn_out",
        )
    )
    nodes.append(
        helper.make_node(
            "Add",
            [layer_input, f"{prefix}_attn_out"],
            [f"{prefix}_resid1"],
            name=f"{prefix}_resid1",
        )
    )
    nodes.extend(_layer_norm(f"{prefix}_resid1", f"{prefix}_ln1"))
    nodes.extend(
        _matmul_add(
            f"{prefix}_ln1", f"{prefix}_w_ff1", f"{prefix}_b_ff1", f"{prefix}_ff1"
        )
    )
    nodes.extend(_gelu_erf(f"{prefix}_ff1", f"{prefix}_gelu"))
    nodes.extend(
        _matmul_add(
            f"{prefix}_gelu", f"{prefix}_w_ff2", f"{prefix}_b_ff2", f"{prefix}_ff2"
        )
    )
    nodes.append(
        helper.make_node(
            "Add",
            [f"{prefix}_ln1", f"{prefix}_ff2"],
            [f"{prefix}_resid2"],
            name=f"{prefix}_resid2",
        )
    )
    ln2_input = f"{prefix}_resid2"
    if include_demo_rewrites:
        ln2_input = f"{prefix}_resid2_back"
        nodes.append(
            helper.make_node(
                "Transpose",
                [f"{prefix}_resid2"],
                [f"{prefix}_tr1"],
                name=f"{prefix}_cancel_transpose_1",
                perm=[0, 2, 1],
            )
        )
        nodes.append(
            helper.make_node(
                "Transpose",
                [f"{prefix}_tr1"],
                [ln2_input],
                name=f"{prefix}_cancel_transpose_2",
                perm=[0, 2, 1],
            )
        )
    nodes.extend(_layer_norm(ln2_input, f"{prefix}_ln2"))
    return nodes


def build_transformer_like_model(
    spec: TransformerSpec,
    *,
    seed: int = 0,
    include_demo_rewrites: bool = True,
) -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    sample = {
        "hidden": rng.normal(0, 0.2, size=(spec.batch, spec.seq, spec.hidden)).astype(
            np.float32
        )
    }

    x = "hidden"
    for layer in range(spec.layers):
        prefix = f"layer{layer}"
        _add_layer_initializers(initializers, rng, spec, prefix)
        nodes.extend(
            _transformer_layer(
                x,
                prefix,
                include_demo_rewrites=include_demo_rewrites,
            )
        )
        x = f"{prefix}_ln2"

    graph = helper.make_graph(
        nodes,
        f"tcw_{spec.name}_transformer_like",
        [
            helper.make_tensor_value_info(
                "hidden", TensorProto.FLOAT, [spec.batch, spec.seq, spec.hidden]
            )
        ],
        [
            helper.make_tensor_value_info(
                x, TensorProto.FLOAT, [spec.batch, spec.seq, spec.hidden]
            )
        ],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="transformer-compiler-workbench",
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    model.ir_version = 9
    metadata = model.metadata_props.add()
    metadata.key = "tcw_model"
    metadata.value = f"{spec.name}-transformer-like"
    return model, sample


def build_tiny_transformer_like_model(
    seed: int = 0,
) -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    return build_transformer_like_model(PRESETS["tiny"], seed=seed)


def export_model(
    model_name: str,
    out: str | Path,
    *,
    preset: str = "tiny",
    batch: int | None = None,
    seq: int | None = None,
    hidden: int | None = None,
    ffn: int | None = None,
    layers: int | None = None,
) -> dict[str, Any]:
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if preset not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown preset '{preset}'. Available presets: {available}")
    base = PRESETS[preset]
    spec = TransformerSpec(
        batch=batch or base.batch,
        seq=seq or base.seq,
        hidden=hidden or base.hidden,
        ffn=ffn or base.ffn,
        layers=layers or base.layers,
        name=base.name
        if (batch, seq, hidden, ffn, layers) == (None,) * 5
        else "custom",
    )
    model, sample = build_transformer_like_model(spec)
    save_checked_model(model, output_path)

    sample_path = output_path.with_suffix(".inputs.npz")
    np.savez(sample_path, **sample)

    metadata = {
        "requested_model": model_name,
        "exported_model": f"{spec.name}-transformer-like",
        "shape": {
            "batch": spec.batch,
            "seq": spec.seq,
            "hidden": spec.hidden,
            "ffn": spec.ffn,
            "layers": spec.layers,
        },
        "onnx_path": str(output_path),
        "sample_input_path": str(sample_path),
        "note": _export_note(spec),
    }
    write_json(output_path.with_suffix(".export.json"), metadata)
    return metadata


def _export_note(spec: TransformerSpec) -> str:
    if spec.name == "tiny":
        return (
            "The default export uses a deterministic tiny transformer-like ONNX "
            "graph so the workbench is reproducible on macOS/CPU without downloads."
        )
    return (
        "This export uses a larger deterministic transformer-like ONNX graph for "
        "provider benchmarking. It still uses standard ONNX operators and does "
        "not require CUDA."
    )
