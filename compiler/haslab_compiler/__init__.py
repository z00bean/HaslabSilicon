"""Minimal HASLAB compiler components for the M6 vertical slice."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from .hxb import Section, SectionType, build_hxb
from .first_layer import compile_conv_silu_first_layer, compile_pinned_first_layer
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
    "ConvSiluLayerSpec",
    "Section",
    "SectionType",
    "build_hxb",
    "compile_conv_silu_first_layer",
    "compile_conv_silu_pipeline",
    "compile_conv_silu_slice",
    "compile_pinned_first_block",
    "compile_pinned_first_layer",
    "compile_pinned_first_two_layers",
    "extract_input_patch",
    "load_pinned_first_two_layers",
]
