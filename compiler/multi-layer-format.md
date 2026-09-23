# Experimental M6 sequential Conv-SiLU package

Copyright (C) 2026 Zubin Bhuyan.
SPDX-License-Identifier: GPL-3.0-or-later

`haslab.conv-silu-pipeline.v1` is the first reusable multi-layer lowering profile. It compiles a connected sequence of 3×3, stride-two, pad-one `Conv → Sigmoid → Mul` blocks into one deterministic package. It remains experimental and does not define the stable HASLAB package schema.

## Scheduling model

Every layer uses 8×8 output tiles, eight-output-channel groups, and eight-input-channel reduction chunks. The compiler validates tensor and scale connectivity between layers. It packs each output-group/input-chunk weight block independently, keeps all weights and epilogue records for the active layer in local SRAM, loads each spatial input tile once, and reuses that input across all output groups.

The second YOLOv8n block consumes the retained 160×160×2×8 first-layer HWC8 tensor. Because its two input groups are interleaved at each pixel, the scheduler gathers each group with strided eight-byte DMA transfers before issuing the two-command accumulator chain. `FIRST` is set for channels 0–7 and `LAST` for channels 8–15. The four output groups are scattered into the canonical 80×80×4×8 result.

Both outputs remain live in the external `output` buffer:

| Tensor | Addend | Bytes | Logical shape | Physical HWC8 shape |
|---|---:|---:|---|---|
| `model.0` | 0 | 409,600 | 1×16×160×160 | 160×160×2×8 |
| `model.1` | 409,600 | 204,800 | 1×32×80×80 | 80×80×4×8 |

## Exact schedule

| Quantity | First block | Second block | Cumulative |
|---|---:|---:|---:|
| Spatial tiles | 400 | 100 | 500 |
| Output-group tile executions | 800 | 400 | 1,200 |
| Input-channel chunks | 1 | 2 | — |
| Commands excluding `END` | 8,442 | 7,802 | 16,244 |
| `DMA_COPY2D` | 6,803 | 6,583 | 13,386 |
| `FILL8` | 39 | 19 | 58 |
| `CONV_I8` | 800 | 800 | 1,600 |
| `EPILOGUE` | 800 | 400 | 1,200 |
| Constant DMA bytes | 2,432 | 6,144 | 8,576 |
| Input DMA bytes | 919,368 | 456,976 | 1,376,344 |
| Output DMA bytes | 409,600 | 204,800 | 614,400 |
| Total DMA bytes | 1,331,400 | 667,920 | 1,999,320 |
| Boundary fill bytes | 90,168 | 87,856 | 178,024 |
| INT8 MACs | 11,059,200 | 29,491,200 | 40,550,400 |

One final `END` command makes the package total 16,245 commands. The first-block traffic is lower than the earlier `haslab.first-layer.v1` proof because this scheduler loads each input tile once and reuses it across both output groups.

## Allocation and runtime evidence

The package declares 819,200 input bytes, 8,576 constant bytes, and 614,400 retained-output bytes, for 1,442,176 external bytes. Peak local use is 4,624 input bytes, 4,608 weight bytes, 2,048 accumulator bytes, 512 output bytes, and 1,536 parameter bytes.

The functional runtime accepts all 16,245 commands. With its modeled eight-entry FIFO it records 16,237 `BUSY` responses and commands retired during refill, followed by an eight-command final drain. These counts describe the synchronous submission algorithm; they do not model transport cycles or throughput.

The pinned integration compares all 614,400 retained INT8 values with the independent numerical model and reports zero mismatches. It also compares each dequantized boundary with ONNX Runtime. See the [machine-readable result](../benchmarks/manifests/yolov8n-320-opset13/m6-first-two-conv-silu-layers.json).

## Remaining work

The first two C2f increments and reusable lifetime scheduling are now documented in the [reusable graph format note](reusable-graph-format.md). The remaining accelerator graph and declared host tail remain open.
