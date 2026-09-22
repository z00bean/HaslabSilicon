# Experimental M6 first-layer package

Copyright (C) 2026 Zubin Bhuyan.
SPDX-License-Identifier: GPL-3.0-or-later

`haslab.first-layer.v1` extends the one-tile proof to the complete first YOLOv8n Conv-SiLU layer. It remains an experimental manifest schema rather than a stable public package format.

## Tensor coverage

The input binding is the complete 320×320 HWC8 tensor: three logical INT8 channels plus five zero padding lanes. The output is the canonical 160×160×2×8 HWC8 representation of all 16 logical channels. The compiler emits 400 spatial 8×8 tiles for each of two output-channel groups, for 800 convolution executions.

Each interior tile copies a 17×17 input patch with one strided two-dimensional DMA. The top row and left column need explicit zero halo bytes. Their union contains 39 unique boundary tiles per channel group, so the schedule emits 78 `FILL8` commands. The even 320-pixel input extent requires no bottom or right halo.

The output SRAM holds one contiguous 8×8×8 group tile. Canonical HWC8 output interleaves its two channel groups at every pixel, so each tile uses eight DMA commands. Each command scatters one local tile row with 8-byte transfers, eight rows, an 8-byte source stride, and a 16-byte destination stride.

## Generated schedule

| Quantity | Exact value |
|---|---:|
| Spatial tiles | 400 |
| Channel-group tile executions | 800 |
| Total commands | 8,884 |
| `DMA_COPY2D` | 7,205 |
| `FILL8` | 78 |
| `CONV_I8` | 800 |
| `EPILOGUE` | 800 |
| `END` | 1 |
| Constant DMA bytes | 2,432 |
| Input DMA bytes | 1,838,736 |
| Output DMA bytes | 409,600 |
| Total DMA bytes | 2,250,768 |
| Boundary fill bytes | 180,336 |
| INT8 MACs | 11,059,200 |

Weights and epilogue records are loaded once per output group. The shared 1,024-byte SiLU table is loaded once. The schedule is intentionally serialized because the functional ABI model has no overlap or timing semantics.

## FIFO accounting

The functional runtime submits into the modeled eight-entry FIFO and retires one command whenever submission returns `BUSY`. Execution of this package records 8,884 accepted commands, 8,876 `BUSY` responses, 8,876 commands retired during refill, and eight commands in the final drain. These are exact host/simulator interactions, not cycle counts or a throughput measurement.

## Remaining work

The next compiler step should replace first-layer constants with a reusable layer plan and lower the second Conv-SiLU block while preserving the first-layer tensor as a checked intermediate. That step must handle 16 logical input channels, 32 outputs, four output groups, the 80×80 result, and cumulative external-memory liveness and traffic.

Before a stable package release, define required manifest keys, size limits, unknown-field behavior, buffer alias/lifetime rules, complete tensor descriptors, host-tail metadata, and negative relocation/parser cases.
