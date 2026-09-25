"""Reusable graph IR, lifetime allocation, and lowering for HASLAB M6."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .c2f import (
    C2fBlockSpec,
    C2fConvSpec,
    _Builder,
    _TensorRef,
    _extract_hxb,
    _prepare,
)
from .pipeline import ConvSiluLayerSpec, compile_conv_silu_pipeline
from .vertical_slice import CompileError, _align

GRAPH_SCHEMA = "haslab.graph-schedule.v1"
AllocationMode = Literal["diagnostic", "release"]


@dataclass(frozen=True)
class GraphTensor:
    name: str
    height: int
    width: int
    channels: int
    scale: float
    alias_of: str | None = None
    channel_group_offset: int = 0

    @property
    def bytes(self) -> int:
        return self.height * self.width * self.channels


@dataclass(frozen=True)
class ConvSiluOp:
    source: str
    destination: str
    layer: C2fConvSpec


@dataclass(frozen=True)
class AddOp:
    left: str
    right: str
    destination: str
    node_name: str


@dataclass(frozen=True)
class ConcatOp:
    sources: tuple[str, ...]
    destination: str
    node_name: str


@dataclass(frozen=True)
class MaxPoolOp:
    source: str
    destination: str
    node_name: str


GraphOp = ConvSiluOp | AddOp | ConcatOp | MaxPoolOp


@dataclass(frozen=True)
class GraphIR:
    tensors: tuple[GraphTensor, ...]
    operations: tuple[GraphOp, ...]
    source_nodes: tuple[str, ...]
    final_outputs: tuple[str, ...]


@dataclass(frozen=True)
class C2fStageSpec:
    cv1: C2fConvSpec
    bottlenecks: tuple[tuple[C2fConvSpec, C2fConvSpec], ...]
    cv2: C2fConvSpec
    residual_scales: tuple[float, ...]
    concat_scale: float
    split_nodes: tuple[str, str]
    add_nodes: tuple[str, ...]
    concat_node: str


@dataclass(frozen=True)
class C2fExtension:
    downsample: C2fConvSpec
    stage: C2fStageSpec


@dataclass(frozen=True)
class SPPFSpec:
    cv1: C2fConvSpec
    pool_nodes: tuple[str, str, str]
    concat_node: str
    concat_scale: float
    cv2: C2fConvSpec


def _material(name: str, layer: C2fConvSpec) -> GraphTensor:
    weights = np.asarray(layer.weights)
    output_channels = int(weights.shape[0])
    output_h = (layer.input_h + 2 * layer.padding - layer.kernel) // layer.stride + 1
    output_w = (layer.input_w + 2 * layer.padding - layer.kernel) // layer.stride + 1
    return GraphTensor(name, output_h, output_w, output_channels, layer.output_scale)


def first_stage(block: C2fBlockSpec) -> C2fStageSpec:
    return C2fStageSpec(
        cv1=block.cv1,
        bottlenecks=((block.bottleneck_cv1, block.bottleneck_cv2),),
        cv2=block.cv2,
        residual_scales=(block.residual_scale,),
        concat_scale=block.concat_scale,
        split_nodes=block.split_nodes,
        add_nodes=(block.add_node,),
        concat_node=block.concat_node,
    )


def build_two_c2f_graph(
    *, first: C2fStageSpec, downsample: C2fConvSpec, second: C2fStageSpec
) -> GraphIR:
    """Create the reusable IR for pinned YOLOv8n nodes 6..47."""

    return build_c2f_graph(
        first=first,
        extensions=(C2fExtension(downsample=downsample, stage=second),),
    )


def build_c2f_graph(
    *, first: C2fStageSpec, extensions: Sequence[C2fExtension]
) -> GraphIR:
    """Create a reusable stem-following chain of downsample and C2f stages."""

    tensors: list[GraphTensor] = []
    operations: list[GraphOp] = []

    def add_stage(stage: C2fStageSpec, source: str, prefix: str) -> str:
        cv1_name = f"{prefix}.cv1"
        tensors.append(_material(cv1_name, stage.cv1))
        operations.append(ConvSiluOp(source, cv1_name, stage.cv1))
        hidden_channels = int(np.asarray(stage.cv1.weights).shape[0]) // 2
        split0 = f"{prefix}.split0"
        split1 = f"{prefix}.split1"
        cv1_tensor = tensors[-1]
        tensors.extend(
            [
                GraphTensor(
                    split0,
                    cv1_tensor.height,
                    cv1_tensor.width,
                    hidden_channels,
                    cv1_tensor.scale,
                    alias_of=cv1_name,
                    channel_group_offset=0,
                ),
                GraphTensor(
                    split1,
                    cv1_tensor.height,
                    cv1_tensor.width,
                    hidden_channels,
                    cv1_tensor.scale,
                    alias_of=cv1_name,
                    channel_group_offset=hidden_channels // 8,
                ),
            ]
        )
        branch = split1
        concat_sources = [split0, split1]
        if not (
            len(stage.bottlenecks)
            == len(stage.residual_scales)
            == len(stage.add_nodes)
        ):
            raise CompileError(f"{prefix} bottleneck metadata lengths disagree")
        for index, ((conv1, conv2), residual_scale, add_node) in enumerate(
            zip(stage.bottlenecks, stage.residual_scales, stage.add_nodes)
        ):
            conv1_name = f"{prefix}.m.{index}.cv1"
            conv2_name = f"{prefix}.m.{index}.cv2"
            add_name = f"{prefix}.m.{index}.add"
            tensors.append(_material(conv1_name, conv1))
            operations.append(ConvSiluOp(branch, conv1_name, conv1))
            tensors.append(_material(conv2_name, conv2))
            operations.append(ConvSiluOp(conv1_name, conv2_name, conv2))
            conv2_tensor = tensors[-1]
            tensors.append(
                GraphTensor(
                    add_name,
                    conv2_tensor.height,
                    conv2_tensor.width,
                    conv2_tensor.channels,
                    residual_scale,
                )
            )
            operations.append(AddOp(branch, conv2_name, add_name, add_node))
            branch = add_name
            concat_sources.append(add_name)
        concat_name = f"{prefix}.concat"
        tensors.append(
            GraphTensor(
                concat_name,
                cv1_tensor.height,
                cv1_tensor.width,
                sum(next(item.channels for item in tensors if item.name == name) for name in concat_sources),
                stage.concat_scale,
            )
        )
        operations.append(ConcatOp(tuple(concat_sources), concat_name, stage.concat_node))
        output_name = prefix
        tensors.append(_material(output_name, stage.cv2))
        operations.append(ConvSiluOp(concat_name, output_name, stage.cv2))
        return output_name

    if not first.cv1.name.endswith(".cv1"):
        raise CompileError("first C2f stage name must end in .cv1")
    first_prefix = first.cv1.name.removesuffix(".cv1")
    output = add_stage(first, "model.1", first_prefix)
    source_nodes = _stage_source_nodes(first)
    for extension in extensions:
        downsample_name = extension.downsample.name
        tensors.append(_material(downsample_name, extension.downsample))
        operations.append(ConvSiluOp(output, downsample_name, extension.downsample))
        source_nodes.extend(extension.downsample.node_names)
        if not extension.stage.cv1.name.endswith(".cv1"):
            raise CompileError("C2f stage name must end in .cv1")
        prefix = extension.stage.cv1.name.removesuffix(".cv1")
        output = add_stage(extension.stage, downsample_name, prefix)
        source_nodes.extend(_stage_source_nodes(extension.stage))
    return GraphIR(tuple(tensors), tuple(operations), tuple(source_nodes), (output,))


def extend_with_sppf(graph: GraphIR, sppf: SPPFSpec) -> GraphIR:
    """Append the pinned three-pool SPPF pattern to a validated graph shape."""

    if len(graph.final_outputs) != 1:
        raise CompileError("SPPF requires one graph input")
    if not sppf.cv1.name.endswith(".cv1") or not sppf.cv2.name.endswith(".cv2"):
        raise CompileError("SPPF convolution names must end in .cv1 and .cv2")
    prefix = sppf.cv1.name.removesuffix(".cv1")
    if sppf.cv2.name.removesuffix(".cv2") != prefix:
        raise CompileError("SPPF convolution names must share a prefix")
    source = graph.final_outputs[0]
    tensors = list(graph.tensors)
    operations = list(graph.operations)
    cv1_name = sppf.cv1.name
    cv1_tensor = _material(cv1_name, sppf.cv1)
    tensors.append(cv1_tensor)
    operations.append(ConvSiluOp(source, cv1_name, sppf.cv1))
    pool_sources = [cv1_name]
    for index, node_name in enumerate(sppf.pool_nodes):
        destination = f"{prefix}.pool{index}"
        tensors.append(
            GraphTensor(
                destination,
                cv1_tensor.height,
                cv1_tensor.width,
                cv1_tensor.channels,
                cv1_tensor.scale,
            )
        )
        operations.append(MaxPoolOp(pool_sources[-1], destination, node_name))
        pool_sources.append(destination)
    concat_name = f"{prefix}.concat"
    tensors.append(
        GraphTensor(
            concat_name,
            cv1_tensor.height,
            cv1_tensor.width,
            cv1_tensor.channels * len(pool_sources),
            sppf.concat_scale,
        )
    )
    operations.append(ConcatOp(tuple(pool_sources), concat_name, sppf.concat_node))
    output_name = prefix
    tensors.append(_material(output_name, sppf.cv2))
    operations.append(ConvSiluOp(concat_name, output_name, sppf.cv2))
    source_nodes = list(graph.source_nodes)
    source_nodes.extend(sppf.cv1.node_names)
    source_nodes.extend(sppf.pool_nodes)
    source_nodes.append(sppf.concat_node)
    source_nodes.extend(sppf.cv2.node_names)
    return GraphIR(tuple(tensors), tuple(operations), tuple(source_nodes), (output_name,))


def build_backbone_graph(
    *,
    first: C2fStageSpec,
    extensions: Sequence[C2fExtension],
    sppf: SPPFSpec,
) -> GraphIR:
    """Build the pinned YOLOv8n backbone through its SPPF output."""

    return extend_with_sppf(
        build_c2f_graph(first=first, extensions=extensions),
        sppf,
    )


def _stage_source_nodes(stage: C2fStageSpec) -> list[str]:
    nodes = list(stage.cv1.node_names) + list(stage.split_nodes)
    for pair, add_node in zip(stage.bottlenecks, stage.add_nodes):
        nodes.extend(pair[0].node_names)
        nodes.extend(pair[1].node_names)
        nodes.append(add_node)
    nodes.append(stage.concat_node)
    nodes.extend(stage.cv2.node_names)
    return nodes


def _operation_sources(operation: GraphOp) -> tuple[str, ...]:
    if isinstance(operation, (ConvSiluOp, MaxPoolOp)):
        return (operation.source,)
    if isinstance(operation, AddOp):
        return (operation.left, operation.right)
    return operation.sources


def _operation_destination(operation: GraphOp) -> str:
    return operation.destination


def _validate_graph(graph: GraphIR, base_names: set[str]) -> dict[str, GraphTensor]:
    tensors: dict[str, GraphTensor] = {}
    for tensor in graph.tensors:
        if tensor.name in tensors or tensor.name in base_names:
            raise CompileError(f"duplicate graph tensor: {tensor.name}")
        if min(tensor.height, tensor.width, tensor.channels) <= 0 or tensor.channels % 8:
            raise CompileError(f"{tensor.name} has an invalid HWC8 shape")
        if not np.isfinite(tensor.scale) or tensor.scale <= 0:
            raise CompileError(f"{tensor.name} has an invalid scale")
        if tensor.alias_of is not None and tensor.alias_of not in tensors:
            raise CompileError(f"{tensor.name} aliases an unavailable tensor")
        tensors[tensor.name] = tensor
    produced = set(base_names)
    for operation in graph.operations:
        missing = [name for name in _operation_sources(operation) if name not in produced]
        if missing:
            raise CompileError(f"graph operation reads unavailable tensors: {missing}")
        destination = _operation_destination(operation)
        if destination not in tensors or tensors[destination].alias_of is not None:
            raise CompileError(f"graph operation has invalid destination: {destination}")
        if destination in produced:
            raise CompileError(f"graph tensor is produced twice: {destination}")
        produced.add(destination)
        produced.update(
            tensor.name
            for tensor in graph.tensors
            if tensor.alias_of == destination
        )
    if any(name not in produced for name in graph.final_outputs):
        raise CompileError("graph final output is not produced")
    return tensors


def _allocation_plan(
    *,
    graph: GraphIR,
    base_records: Sequence[dict[str, object]],
    mode: AllocationMode,
) -> tuple[dict[str, int], list[dict[str, object]], int, int]:
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    material = [tensor for tensor in graph.tensors if tensor.alias_of is None]
    producer = {
        _operation_destination(operation): index
        for index, operation in enumerate(graph.operations)
    }

    def root(name: str) -> str:
        while name in tensors and tensors[name].alias_of is not None:
            name = tensors[name].alias_of  # type: ignore[assignment]
        return name

    last_use: dict[str, int] = {record["name"]: -1 for record in base_records}
    last_use.update({tensor.name: producer[tensor.name] for tensor in material})
    for index, operation in enumerate(graph.operations):
        for name in _operation_sources(operation):
            resolved = root(name)
            last_use[resolved] = max(last_use.get(resolved, -1), index)
    terminal = len(graph.operations)
    for name in graph.final_outputs:
        last_use[root(name)] = terminal

    records: list[dict[str, object]] = []
    allocations: dict[str, int] = {}
    base_starts = {record["name"]: -2 + index for index, record in enumerate(base_records)}
    for record in base_records:
        name = str(record["name"])
        allocations[name] = int(record["addend"])
        records.append(
            {
                "addend": int(record["addend"]),
                "bytes": int(record["bytes"]),
                "first_operation": base_starts[name],
                "last_operation": terminal if mode == "diagnostic" else last_use[name],
                "name": name,
            }
        )
    cursor = max(int(record["addend"]) + int(record["bytes"]) for record in base_records)
    for tensor in material:
        start = producer[tensor.name]
        end = terminal if mode == "diagnostic" else last_use[tensor.name]
        size = tensor.bytes
        if mode == "diagnostic":
            addend = _align(cursor, 64)
            cursor = addend + size
        else:
            active = sorted(
                (
                    int(record["addend"]),
                    int(record["addend"]) + int(record["bytes"]),
                )
                for record in records
                if int(record["last_operation"]) >= start
            )
            addend = 0
            for low, high in active:
                addend = _align(addend, 64)
                if addend + size <= low:
                    break
                addend = max(addend, high)
            addend = _align(addend, 64)
        allocations[tensor.name] = addend
        records.append(
            {
                "addend": addend,
                "bytes": size,
                "first_operation": start,
                "last_operation": end,
                "name": tensor.name,
            }
        )
    peak = max(int(record["addend"]) + int(record["bytes"]) for record in records)
    logical = sum(int(record["bytes"]) for record in records)
    return allocations, records, peak, logical


def compile_graph(
    *,
    stem_layers: Sequence[ConvSiluLayerSpec],
    graph: GraphIR,
    source_model_sha256: str,
    allocation_mode: AllocationMode,
) -> bytes:
    """Lower a validated graph using diagnostic or lifetime-reuse allocation."""

    if allocation_mode not in ("diagnostic", "release"):
        raise CompileError("allocation mode must be diagnostic or release")
    if len(stem_layers) != 2:
        raise CompileError("the graph profile requires the two-layer stem")
    base = compile_conv_silu_pipeline(layers=stem_layers, source_model_sha256=source_model_sha256)
    manifest, commands, constants, debug = _extract_hxb(base)
    builder = _Builder(manifest, commands, constants, debug)
    builder.schema = GRAPH_SCHEMA
    builder.profile = "M6_REUSABLE_GRAPH"
    builder.operation_schedule_key = "graph_operations"
    builder.allocation_description = (
        "all materialized boundaries retained"
        if allocation_mode == "diagnostic"
        else "deterministic first-fit reuse by inclusive operation lifetime"
    )
    base_records = [dict(item) for item in builder.tensors]
    base_names = {str(item["name"]) for item in base_records}
    tensor_specs = _validate_graph(graph, base_names)
    allocations, plan, peak, logical = _allocation_plan(
        graph=graph, base_records=base_records, mode=allocation_mode
    )
    _, _, diagnostic_peak, _ = _allocation_plan(
        graph=graph, base_records=base_records, mode="diagnostic"
    )
    _, _, release_peak, _ = _allocation_plan(
        graph=graph, base_records=base_records, mode="release"
    )

    refs: dict[str, _TensorRef] = {}
    for record in base_records:
        refs[str(record["name"])] = _TensorRef(
            str(record["name"]),
            int(record["addend"]),
            int(record["physical_shape"][0]),
            int(record["physical_shape"][1]),
            int(record["logical_shape"][1]),
            int(record["physical_shape"][2]),
            0,
            float(record["scale_binary32"]),
        )
    material_records: dict[str, dict[str, object]] = {}
    for tensor in graph.tensors:
        if tensor.alias_of is not None:
            source = refs[tensor.alias_of]
            refs[tensor.name] = builder.add_view(
                tensor.name, source, tensor.channel_group_offset, tensor.channels
            )
            continue
        addend = allocations[tensor.name]
        groups = tensor.channels // 8
        record = {
            "addend": addend,
            "bytes": tensor.bytes,
            "dtype": "int8",
            "layout": "HWC8",
            "logical_shape": [1, tensor.channels, tensor.height, tensor.width],
            "name": tensor.name,
            "physical_shape": [tensor.height, tensor.width, groups, 8],
            "scale_binary32": float(np.float32(tensor.scale)),
        }
        material_records[tensor.name] = record
        refs[tensor.name] = _TensorRef(
            tensor.name,
            addend,
            tensor.height,
            tensor.width,
            tensor.channels,
            groups,
            0,
            tensor.scale,
        )
    if allocation_mode == "diagnostic":
        builder.tensors.extend(material_records[tensor.name] for tensor in graph.tensors if tensor.alias_of is None)
    else:
        builder.tensors.clear()
        builder.tensors.extend(material_records[name] for name in graph.final_outputs)
    builder.output_bytes = peak
    manifest["output"]["allocation_mode"] = allocation_mode
    manifest["output"]["allocation_plan"] = plan
    manifest["schedule"]["allocation"] = {
        "diagnostic_peak_output_bytes": diagnostic_peak,
        "logical_materialized_tensor_bytes": logical,
        "mode": allocation_mode,
        "release_peak_output_bytes": release_peak,
        "reuse_savings_bytes": diagnostic_peak - release_peak,
        "selected_peak_output_bytes": peak,
    }

    for operation in graph.operations:
        if isinstance(operation, ConvSiluOp):
            builder.schedule_conv(
                refs[operation.source], refs[operation.destination], _prepare(operation.layer)
            )
        elif isinstance(operation, AddOp):
            builder.schedule_add(
                refs[operation.left],
                refs[operation.right],
                refs[operation.destination],
                operation.node_name,
            )
        elif isinstance(operation, ConcatOp):
            builder.schedule_concat(
                tuple(refs[name] for name in operation.sources),
                refs[operation.destination],
                operation.node_name,
            )
        else:
            builder.schedule_maxpool(
                refs[operation.source],
                refs[operation.destination],
                operation.node_name,
            )
    if allocation_mode == "release":
        builder.views.clear()
    source_nodes = [name for layer in stem_layers for name in layer.node_names]
    source_nodes.extend(graph.source_nodes)
    return builder.finish(source_nodes)


def compile_pinned_through_second_c2f(
    model_path: str,
    calibration_path: str,
    lut_path: str,
    *,
    allocation_mode: AllocationMode,
) -> bytes:
    """Compile pinned nodes 0..47 with the reusable graph scheduler."""

    from .c2f import load_pinned_through_second_c2f

    stem, first, downsample, second, model_hash = load_pinned_through_second_c2f(
        model_path, calibration_path, lut_path
    )
    graph = build_two_c2f_graph(
        first=first_stage(first), downsample=downsample, second=second
    )
    return compile_graph(
        stem_layers=stem,
        graph=graph,
        source_model_sha256=model_hash,
        allocation_mode=allocation_mode,
    )


def compile_pinned_through_third_c2f(
    model_path: str,
    calibration_path: str,
    lut_path: str,
    *,
    allocation_mode: AllocationMode,
) -> bytes:
    """Compile pinned nodes 0..73 with the reusable graph scheduler."""

    from .c2f import load_pinned_through_third_c2f

    stem, first, downsample2, second, downsample3, third, model_hash = (
        load_pinned_through_third_c2f(model_path, calibration_path, lut_path)
    )
    graph = build_c2f_graph(
        first=first_stage(first),
        extensions=(
            C2fExtension(downsample=downsample2, stage=second),
            C2fExtension(downsample=downsample3, stage=third),
        ),
    )
    return compile_graph(
        stem_layers=stem,
        graph=graph,
        source_model_sha256=model_hash,
        allocation_mode=allocation_mode,
    )


def compile_pinned_through_backbone(
    model_path: str,
    calibration_path: str,
    lut_path: str,
    *,
    allocation_mode: AllocationMode,
) -> bytes:
    """Compile pinned nodes 0..102 through the complete backbone and SPPF."""

    from .c2f import load_pinned_through_backbone

    (
        stem,
        first,
        downsample2,
        second,
        downsample3,
        third,
        downsample4,
        fourth,
        sppf,
        model_hash,
    ) = load_pinned_through_backbone(model_path, calibration_path, lut_path)
    graph = build_backbone_graph(
        first=first_stage(first),
        extensions=(
            C2fExtension(downsample=downsample2, stage=second),
            C2fExtension(downsample=downsample3, stage=third),
            C2fExtension(downsample=downsample4, stage=fourth),
        ),
        sppf=sppf,
    )
    return compile_graph(
        stem_layers=stem,
        graph=graph,
        source_model_sha256=model_hash,
        allocation_mode=allocation_mode,
    )
