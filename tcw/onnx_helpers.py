from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper, numpy_helper


FLOAT_TYPES = {
    TensorProto.FLOAT,
    TensorProto.FLOAT16,
    TensorProto.DOUBLE,
    TensorProto.BFLOAT16,
}
INT_TYPES = {
    TensorProto.INT8,
    TensorProto.INT16,
    TensorProto.INT32,
    TensorProto.INT64,
    TensorProto.UINT8,
    TensorProto.UINT16,
    TensorProto.UINT32,
    TensorProto.UINT64,
}


def load_model(path: str | Path) -> ModelProto:
    return onnx.load(str(path))


def save_checked_model(model: ModelProto, path: str | Path) -> None:
    onnx.checker.check_model(model)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def infer_shapes(model: ModelProto) -> ModelProto:
    try:
        return onnx.shape_inference.infer_shapes(model)
    except Exception:
        return model


def attribute_value(attribute: onnx.AttributeProto) -> Any:
    return helper.get_attribute_value(attribute)


def get_attribute(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return attribute_value(attribute)
    return default


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    shape = value_info.type.tensor_type.shape
    dims: list[int | str | None] = []
    for dim in shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def value_shapes(model: ModelProto) -> dict[str, list[int | str | None]]:
    shaped = infer_shapes(model)
    values = [
        *shaped.graph.input,
        *shaped.graph.value_info,
        *shaped.graph.output,
    ]
    return {
        value.name: tensor_shape(value)
        for value in values
        if value.type.HasField("tensor_type")
    }


def value_elem_types(model: ModelProto) -> dict[str, int]:
    shaped = infer_shapes(model)
    result: dict[str, int] = {}
    for value in [*shaped.graph.input, *shaped.graph.value_info, *shaped.graph.output]:
        if value.type.HasField("tensor_type"):
            result[value.name] = value.type.tensor_type.elem_type
    for initializer in shaped.graph.initializer:
        result[initializer.name] = initializer.data_type
    return result


def graph_output_names(model: ModelProto) -> set[str]:
    return {output.name for output in model.graph.output}


def initializer_map(model: ModelProto) -> dict[str, np.ndarray]:
    return {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }


def consumers(model: ModelProto) -> dict[str, list[onnx.NodeProto]]:
    result: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            if name:
                result[name].append(node)
    return result


def producer_map(model: ModelProto) -> dict[str, onnx.NodeProto]:
    result: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for output in node.output:
            if output:
                result[output] = node
    return result


def replace_value_uses(
    model: ModelProto,
    old: str,
    new: str,
    *,
    replace_graph_outputs: bool = True,
) -> None:
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new
    if replace_graph_outputs:
        for output in model.graph.output:
            if output.name == old:
                output.name = new


def remove_nodes(model: ModelProto, nodes: Iterable[onnx.NodeProto]) -> None:
    to_remove = {id(node) for node in nodes}
    kept = [node for node in model.graph.node if id(node) not in to_remove]
    del model.graph.node[:]
    model.graph.node.extend(kept)


def input_feeds(
    model: ModelProto,
    *,
    seed: int = 0,
    overrides: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    feeds: dict[str, np.ndarray] = {}
    for graph_input in model.graph.input:
        if graph_input.name in initializer_names:
            continue
        if overrides and graph_input.name in overrides:
            feeds[graph_input.name] = overrides[graph_input.name]
            continue
        tensor_type = graph_input.type.tensor_type
        elem_type = tensor_type.elem_type
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(2)
        if elem_type in FLOAT_TYPES:
            feeds[graph_input.name] = rng.normal(0, 0.2, size=shape).astype(np.float32)
        elif elem_type in INT_TYPES:
            feeds[graph_input.name] = rng.integers(0, 8, size=shape, dtype=np.int64)
        else:
            feeds[graph_input.name] = rng.normal(0, 0.2, size=shape).astype(np.float32)
    return feeds


def make_value_info(name: str, shape: list[int], dtype: int = TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)
