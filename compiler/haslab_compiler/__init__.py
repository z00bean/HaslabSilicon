"""Minimal HASLAB compiler components for the M6 vertical slice."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from .hxb import Section, SectionType, build_hxb
from .c2f import (
    C2fBlockSpec,
    C2fConvSpec,
    compile_first_c2f,
    compile_pinned_first_c2f,
    load_pinned_first_c2f,
)
from .first_layer import compile_conv_silu_first_layer, compile_pinned_first_layer
from .graph import (
    AddOp,
    C2fStageSpec,
    ConcatOp,
    ConvSiluOp,
    GraphIR,
    GraphTensor,
    build_two_c2f_graph,
    compile_graph,
    compile_pinned_through_second_c2f,
    first_stage,
)
from .pipeline import (
    ConvSiluLayerSpec,
    compile_conv_silu_pipeline,
    compile_pinned_first_two_layers,
    load_pinned_first_two_layers,
)
from .vertical_slice import (
    CompileError,
    compile_conv_silu_slice,
    compile_pinned_first_block,
    extract_input_patch,
)

__all__ = [
    "CompileError",
    "C2fBlockSpec",
    "C2fConvSpec",
    "C2fStageSpec",
    "AddOp",
    "ConcatOp",
    "ConvSiluOp",
    "ConvSiluLayerSpec",
    "GraphIR",
    "GraphTensor",
    "Section",
    "SectionType",
    "build_hxb",
    "build_two_c2f_graph",
    "compile_conv_silu_first_layer",
    "compile_conv_silu_pipeline",
    "compile_conv_silu_slice",
    "compile_first_c2f",
    "compile_graph",
    "compile_pinned_first_block",
    "compile_pinned_first_c2f",
    "compile_pinned_first_layer",
    "compile_pinned_first_two_layers",
    "compile_pinned_through_second_c2f",
    "extract_input_patch",
    "first_stage",
    "load_pinned_first_two_layers",
    "load_pinned_first_c2f",
]
