from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx

from tcw.onnx_helpers import (
    consumers,
    get_attribute,
    graph_output_names,
    infer_shapes,
    load_model,
    remove_nodes,
    replace_value_uses,
    save_checked_model,
    value_elem_types,
    value_shapes,
)


@dataclass(frozen=True)
class PassResult:
    name: str
    changed: int
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "changed": self.changed, "notes": self.notes}


def _remove_identity(model: onnx.ModelProto) -> PassResult:
    removed = []
    for node in list(model.graph.node):
        if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
            continue
        replace_value_uses(model, node.output[0], node.input[0])
        removed.append(node)
    remove_nodes(model, removed)
    return PassResult("remove_identity", len(removed), [])


def _remove_dropout(model: onnx.ModelProto) -> PassResult:
    by_input = consumers(model)
    outputs = graph_output_names(model)
    removed = []
    notes = []
    for node in list(model.graph.node):
        if node.op_type != "Dropout" or not node.input or not node.output:
            continue
        if len(node.input) >= 3:
            notes.append(f"Skipped {node.name}: explicit training_mode input present.")
            continue
        mask_output = node.output[1] if len(node.output) > 1 else ""
        if mask_output and (mask_output in outputs or by_input.get(mask_output)):
            notes.append(f"Skipped {node.name}: dropout mask output is used.")
            continue
        replace_value_uses(model, node.output[0], node.input[0])
        removed.append(node)
    remove_nodes(model, removed)
    return PassResult("remove_dropout_inference", len(removed), notes)


def _collapse_transpose_pairs(model: onnx.ModelProto) -> PassResult:
    by_input = consumers(model)
    removed = []
    notes = []
    for first in list(model.graph.node):
        if first.op_type != "Transpose" or not first.output:
            continue
        users = by_input.get(first.output[0], [])
        if len(users) != 1:
            continue
        second = users[0]
        if second.op_type != "Transpose" or not second.output:
            continue
        first_perm = list(get_attribute(first, "perm", []))
        second_perm = list(get_attribute(second, "perm", []))
        if not first_perm or not second_perm or len(first_perm) != len(second_perm):
            continue
        composed = [first_perm[index] for index in second_perm]
        if composed != list(range(len(first_perm))):
            continue
        replace_value_uses(model, second.output[0], first.input[0])
        removed.extend([first, second])
        notes.append(f"Removed canceling pair {first.name} -> {second.name}.")
    remove_nodes(model, removed)
    return PassResult("collapse_transpose_pairs", len(removed) // 2, notes)


def _collapse_cast_chains(model: onnx.ModelProto) -> PassResult:
    typed = value_elem_types(model)
    by_input = consumers(model)
    removed = []
    notes = []
    for first in list(model.graph.node):
        if first.op_type != "Cast" or not first.input or not first.output:
            continue
        users = by_input.get(first.output[0], [])
        if len(users) != 1:
            continue
        second = users[0]
        if second.op_type != "Cast" or not second.output:
            continue
        source_type = typed.get(first.input[0])
        final_type = get_attribute(second, "to")
        if source_type is None or final_type is None:
            notes.append(f"Skipped {first.name} -> {second.name}: dtype unknown.")
            continue
        if source_type != final_type:
            continue
        replace_value_uses(model, second.output[0], first.input[0])
        removed.extend([first, second])
        notes.append(f"Removed safe cast round-trip {first.name} -> {second.name}.")
    remove_nodes(model, removed)
    return PassResult("collapse_cast_chains", len(removed) // 2, notes)


def _remove_noop_reshape(model: onnx.ModelProto) -> PassResult:
    shaped = value_shapes(infer_shapes(model))
    removed = []
    notes = []
    for node in list(model.graph.node):
        if node.op_type != "Reshape" or len(node.input) < 2 or not node.output:
            continue
        input_shape = shaped.get(node.input[0])
        output_shape = shaped.get(node.output[0])
        if input_shape and output_shape and input_shape == output_shape:
            replace_value_uses(model, node.output[0], node.input[0])
            removed.append(node)
            notes.append(f"Removed statically no-op Reshape {node.name}.")
    remove_nodes(model, removed)
    return PassResult("remove_noop_reshape", len(removed), notes)


PASSES = [
    _remove_identity,
    _remove_dropout,
    _collapse_cast_chains,
    _collapse_transpose_pairs,
    _remove_noop_reshape,
]


def apply_rewrite_passes(
    model: onnx.ModelProto,
) -> tuple[onnx.ModelProto, list[PassResult]]:
    model = onnx.load_from_string(model.SerializeToString())
    results = []
    for pass_fn in PASSES:
        result = pass_fn(model)
        results.append(result)
        onnx.checker.check_model(model)
    return model, results


def rewrite_model(in_path: str | Path, out_path: str | Path) -> list[PassResult]:
    model = load_model(in_path)
    rewritten, results = apply_rewrite_passes(model)
    save_checked_model(rewritten, out_path)
    return results
