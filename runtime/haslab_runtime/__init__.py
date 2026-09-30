"""Minimal HASLAB runtime and strict .hxb loader for the simulator backend."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from .package import HxbFormatError, HxbPackage, load_hxb
from .simulator import (
    Completion,
    RuntimeErrorCode,
    RuntimeFailure,
    RuntimeStatus,
    SimulatorRuntime,
    SubmissionStats,
    SubmissionToken,
)
from .yolov8n_tail import (
    Letterbox,
    decode_host_tail,
    map_boxes_to_original,
    select_detections,
    unpack_boundaries,
)

__all__ = [
    "Completion",
    "HxbFormatError",
    "HxbPackage",
    "RuntimeErrorCode",
    "RuntimeFailure",
    "RuntimeStatus",
    "SimulatorRuntime",
    "SubmissionStats",
    "SubmissionToken",
    "Letterbox",
    "decode_host_tail",
    "map_boxes_to_original",
    "select_detections",
    "unpack_boundaries",
    "load_hxb",
]
