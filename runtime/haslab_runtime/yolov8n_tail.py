"""Explicit host implementation of pinned YOLOv8n ONNX nodes 218..260.

This module does not run ONNX Runtime or silently delegate unsupported nodes.
It accepts the three final INT32 HWC8 accelerator tensors and their per-channel
binary32 scales, then performs the fixed graph's FLOAT32 host operations.
"""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOUNDARY_SHAPES = ((1, 144, 40, 40), (1, 144, 20, 20), (1, 144, 10, 10))
BOUNDARY_NAMES = tuple(f"model.22.boundary{i}" for i in range(3))
STRIDES = (8, 16, 32)


def unpack_boundaries(output: bytes, records: list[dict]) -> tuple[np.ndarray, ...]:
    """Read release-package INT32 HWC8 records as dequantized NCHW FLOAT32."""
    by_name = {record["name"]: record for record in records}
    if len(by_name) != 3 or set(by_name) != set(BOUNDARY_NAMES):
        raise ValueError("expected exactly the three pinned learned-head boundaries")
    result = []
    for name, shape in zip(BOUNDARY_NAMES, BOUNDARY_SHAPES):
        record = by_name[name]
        _, channels, height, width = shape
        physical = (height, width, channels // 8, 8)
        start, size = int(record["addend"]), int(record["bytes"])
        if (record["dtype"] != "int32" or record["layout"] != "HWC8"
                or tuple(record["logical_shape"]) != shape
                or tuple(record["physical_shape"]) != physical
                or size != height * width * channels * 4
                or start < 0 or start + size > len(output)):
            raise ValueError(f"invalid pinned boundary record: {name}")
        scales = np.asarray(record["scales_binary32_per_channel"], dtype=np.float32)
        if scales.shape != (channels,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError(f"invalid per-channel scales: {name}")
        raw = np.frombuffer(output, dtype="<i4", count=size // 4, offset=start)
        nchw = raw.reshape(physical).reshape(height, width, channels).transpose(2, 0, 1)[None]
        result.append(nchw.astype(np.float32) * scales.reshape(1, channels, 1, 1))
    return tuple(result)


def _anchors_and_strides() -> tuple[np.ndarray, np.ndarray]:
    grids, stride_vectors = [], []
    for stride, (_, _, height, width) in zip(STRIDES, BOUNDARY_SHAPES):
        yy, xx = np.meshgrid(np.arange(height, dtype=np.float32),
                             np.arange(width, dtype=np.float32), indexing="ij")
        grids.append(np.stack((xx.reshape(-1) + np.float32(0.5),
                               yy.reshape(-1) + np.float32(0.5)), axis=0))
        stride_vectors.append(np.full(height * width, stride, dtype=np.float32))
    return np.concatenate(grids, axis=1)[None], np.concatenate(stride_vectors)[None]


def decode_host_tail(boundaries: tuple[np.ndarray, ...], *, trace: bool = False):
    """Run nodes 218..260; return [1,84,2100] and optional named stage trace.

    All arrays are binary32, including DFL softmax and its 0..15 expectation.
    Candidate order is stride 8, 16, 32, then row-major y/x. The first four
    output channels are cx, cy, w, h in 320-pixel letterbox coordinates.
    """
    if len(boundaries) != 3:
        raise ValueError("expected three boundary tensors")
    for values, shape in zip(boundaries, BOUNDARY_SHAPES):
        if values.shape != shape or values.dtype != np.float32 or not np.all(np.isfinite(values)):
            raise ValueError("invalid boundary shape, dtype, or finite values")
    stages: dict[str, np.ndarray] = {}
    flattened = [values.reshape(1, 144, -1) for values in boundaries]  # 218..223
    logits = np.concatenate(flattened, axis=2)  # 224
    regression, classes = np.split(logits, [64], axis=1)  # 225..226
    dfl = regression.reshape(1, 4, 16, 2100).transpose(0, 2, 1, 3)  # 227..229
    shifted = dfl - np.max(dfl, axis=1, keepdims=True)
    exponent = np.exp(shifted).astype(np.float32)
    probability = exponent / np.sum(exponent, axis=1, keepdims=True, dtype=np.float32)  # 230
    distances = np.sum(probability * np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1),
                       axis=1, dtype=np.float32)  # 231..233
    left_top, right_bottom = distances[:, :2], distances[:, 2:]  # 234..247
    anchors, strides = _anchors_and_strides()  # 248,250,257
    xy1 = anchors - left_top  # 249
    xy2 = anchors + right_bottom  # 251
    center = (xy1 + xy2) / np.float32(2.0)  # 252..254
    size = xy2 - xy1  # 255
    boxes = np.concatenate((center, size), axis=1) * strides  # 256,258
    # Equivalent stable sigmoid avoids overflow for very large finite logits.
    scores = np.empty_like(classes)
    positive = classes >= 0
    scores[positive] = 1 / (1 + np.exp(-classes[positive]))
    negative_exp = np.exp(classes[~positive])
    scores[~positive] = negative_exp / (1 + negative_exp)  # 259
    decoded = np.concatenate((boxes, scores), axis=1)  # 260
    if trace:
        stages.update(concat=logits, regression=regression, classes=classes,
                      dfl_softmax=probability, dfl_distances=distances,
                      anchors=anchors, strides=strides, xy1=xy1, xy2=xy2,
                      boxes=boxes, scores=scores)
        return decoded, stages
    return decoded


@dataclass(frozen=True)
class Letterbox:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    top: int
    left: int

    @classmethod
    def centered(cls, height: int, width: int) -> "Letterbox":
        if height <= 0 or width <= 0:
            raise ValueError("original image dimensions must be positive")
        gain = min(320 / height, 320 / width, 1.0)
        resized_height, resized_width = round(height * gain), round(width * gain)
        return cls(height, width, resized_height, resized_width,
                   round((320 - resized_height) / 2 - 0.1),
                   round((320 - resized_width) / 2 - 0.1))


def map_boxes_to_original(boxes_xyxy: np.ndarray, letterbox: Letterbox) -> np.ndarray:
    """Reverse the recorded 320x320 letterbox and clip xyxy to source bounds."""
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
        raise ValueError("expected N x 4 xyxy boxes")
    if min(letterbox.original_height, letterbox.original_width,
           letterbox.resized_height, letterbox.resized_width) <= 0:
        raise ValueError("invalid letterbox dimensions")
    mapped = boxes_xyxy.astype(np.float32, copy=True)
    mapped[:, (0, 2)] -= np.float32(letterbox.left)
    mapped[:, (1, 3)] -= np.float32(letterbox.top)
    mapped[:, (0, 2)] /= np.float32(letterbox.resized_width / letterbox.original_width)
    mapped[:, (1, 3)] /= np.float32(letterbox.resized_height / letterbox.original_height)
    mapped[:, (0, 2)] = np.clip(mapped[:, (0, 2)], 0, letterbox.original_width)
    mapped[:, (1, 3)] = np.clip(mapped[:, (1, 3)], 0, letterbox.original_height)
    return mapped


def select_detections(decoded: np.ndarray, letterbox: Letterbox, *,
                      confidence: float = 0.25, iou: float = 0.7,
                      max_detections: int = 300) -> np.ndarray:
    """Best-class filtering and class-aware greedy NMS; rows are xyxy,score,class.

    Threshold comparison is strict (>); equal-score candidates keep lower
    candidate index. IoU suppresses only when greater than the threshold.
    """
    if decoded.shape != (1, 84, 2100) or not np.all(np.isfinite(decoded)):
        raise ValueError("expected finite [1,84,2100] decoded output")
    if not (0 <= confidence <= 1 and 0 <= iou <= 1) or max_detections <= 0:
        raise ValueError("invalid filtering thresholds")
    class_id = np.argmax(decoded[0, 4:], axis=0)
    score = np.max(decoded[0, 4:], axis=0)
    candidates = np.flatnonzero(score > confidence)
    order = candidates[np.lexsort((candidates, -score[candidates]))]
    xywh = decoded[0, :4][:, order].T
    xyxy = np.empty_like(xywh)
    xyxy[:, :2] = xywh[:, :2] - xywh[:, 2:] / np.float32(2)
    xyxy[:, 2:] = xywh[:, :2] + xywh[:, 2:] / np.float32(2)
    kept: list[int] = []
    for index in range(len(order)):
        box = xyxy[index]
        area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        suppressed = False
        for previous in kept:
            if class_id[order[index]] != class_id[order[previous]]:
                continue
            other = xyxy[previous]
            intersection = max(0.0, float(min(box[2], other[2]) - max(box[0], other[0]))) * max(
                0.0, float(min(box[3], other[3]) - max(box[1], other[1])))
            other_area = max(0.0, float(other[2] - other[0])) * max(0.0, float(other[3] - other[1]))
            union = area + other_area - intersection
            if union > 0 and intersection / union > iou:
                suppressed = True
                break
        if not suppressed:
            kept.append(index)
            if len(kept) == max_detections:
                break
    if not kept:
        return np.empty((0, 6), dtype=np.float32)
    boxes = map_boxes_to_original(xyxy[kept], letterbox)
    return np.column_stack((boxes, score[order[kept]], class_id[order[kept]])).astype(np.float32)
