from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tcw.io import write_json
from tcw.onnx_helpers import save_checked_model


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


def build_tiny_transformer_like_model(
    seed: int = 0,
) -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    batch, seq, hidden, ffn = 1, 4, 8, 16

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    sample = {
        "hidden": rng.normal(0, 0.2, size=(batch, seq, hidden)).astype(np.float32)
    }

    for name in ("q", "k", "v", "o"):
        initializers.append(
            _tensor(f"w_{name}", rng.normal(0, 0.2, size=(hidden, hidden)))
        )
        initializers.append(_tensor(f"b_{name}", rng.normal(0, 0.02, size=(hidden,))))
    for name, shape in (("w_ff1", (hidden, ffn)), ("w_ff2", (ffn, hidden))):
        initializers.append(_tensor(name, rng.normal(0, 0.2, size=shape)))
    for name, shape in (("b_ff1", (ffn,)), ("b_ff2", (hidden,))):
        initializers.append(_tensor(name, rng.normal(0, 0.02, size=shape)))
    for prefix in ("ln1", "ln2"):
        initializers.append(
            _tensor(f"{prefix}_gamma", np.ones((hidden,), dtype=np.float32))
        )
        initializers.append(
            _tensor(f"{prefix}_beta", np.zeros((hidden,), dtype=np.float32))
        )

    nodes.append(
        helper.make_node("Identity", ["hidden"], ["hidden_id"], name="drop_in_identity")
    )
    nodes.extend(_matmul_add("hidden_id", "w_q", "b_q", "q"))
    nodes.extend(_matmul_add("hidden_id", "w_k", "b_k", "k"))
    nodes.extend(_matmul_add("hidden_id", "w_v", "b_v", "v"))
    nodes.append(
        helper.make_node(
            "Transpose", ["k"], ["k_t"], name="k_transpose", perm=[0, 2, 1]
        )
    )
    nodes.append(
        helper.make_node(
            "MatMul", ["q", "k_t"], ["attention_scores"], name="attention_scores"
        )
    )
    nodes.append(
        helper.make_node(
            "Softmax",
            ["attention_scores"],
            ["attention_probs"],
            name="attention_softmax",
            axis=-1,
        )
    )
    nodes.append(
        helper.make_node(
            "MatMul", ["attention_probs", "v"], ["context"], name="attention_context"
        )
    )
    nodes.extend(_matmul_add("context", "w_o", "b_o", "attn_out"))
    nodes.append(
        helper.make_node("Add", ["hidden_id", "attn_out"], ["resid1"], name="resid1")
    )
    nodes.extend(_layer_norm("resid1", "ln1"))
    nodes.extend(_matmul_add("ln1", "w_ff1", "b_ff1", "ff1"))
    nodes.extend(_gelu_erf("ff1", "gelu"))
    nodes.extend(_matmul_add("gelu", "w_ff2", "b_ff2", "ff2"))
    nodes.append(helper.make_node("Add", ["ln1", "ff2"], ["resid2"], name="resid2"))
    nodes.append(
        helper.make_node(
            "Transpose", ["resid2"], ["tr1"], name="cancel_transpose_1", perm=[0, 2, 1]
        )
    )
    nodes.append(
        helper.make_node(
            "Transpose",
            ["tr1"],
            ["resid2_back"],
            name="cancel_transpose_2",
            perm=[0, 2, 1],
        )
    )
    nodes.extend(_layer_norm("resid2_back", "ln2"))

    graph = helper.make_graph(
        nodes,
        "tcw_tiny_transformer_like",
        [
            helper.make_tensor_value_info(
                "hidden", TensorProto.FLOAT, [batch, seq, hidden]
            )
        ],
        [helper.make_tensor_value_info("ln2", TensorProto.FLOAT, [batch, seq, hidden])],
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
    metadata.value = "tiny-transformer-like"
    return model, sample


def export_model(model_name: str, out: str | Path) -> dict[str, str]:
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, sample = build_tiny_transformer_like_model()
    save_checked_model(model, output_path)

    sample_path = output_path.with_suffix(".inputs.npz")
    np.savez(sample_path, **sample)

    metadata = {
        "requested_model": model_name,
        "exported_model": "tiny-transformer-like",
        "onnx_path": str(output_path),
        "sample_input_path": str(sample_path),
        "note": (
            "The MVP uses a deterministic tiny transformer-like ONNX graph so "
            "the workbench is reproducible on macOS/CPU without downloads."
        ),
    }
    write_json(output_path.with_suffix(".export.json"), metadata)
    return metadata
