# Experimental M6 first-C2f package

Copyright (C) 2026 Zubin Bhuyan.  
SPDX-License-Identifier: GPL-3.0-or-later

`haslab.first-c2f.v1` is the first branch-sensitive HASLAB package profile. It executes pinned YOLOv8n nodes 0–21: the two-layer stem followed by the complete first C2f block. It remains an experimental workload proof and does not define the stable package schema.

## Graph lowering

The first C2f block lowers as:

```text
model.1 output
  → 1×1 Conv-SiLU (32→32)
  → two 16-channel split views
       ├─ split0 ──────────────────────────────┐
       └─ split1 → 3×3 Conv-SiLU → 3×3 Conv-SiLU
                    └──────── scaled residual add
  → three-way channel concat (16 + 16 + 16)
  → 1×1 Conv-SiLU (48→32)
```

Both split outputs alias the `model.2.cv1` allocation at channel-group offsets zero and two. They allocate no external bytes. The residual command applies a shared shift with separate input multipliers, so its two input scales are explicit. Concat materialization maps all three sources into the recorded common concat scale before the final convolution. This includes an exact-scale mapping for the residual branch; it keeps the command path uniform and testable.

All ten validation boundaries are retained or exposed as views. The eight materialized tensors use 1,638,400 external output bytes. Together with the input and constants, the package declares 2,480,256 external bytes.

## Exact pinned schedule

| Quantity | Nodes 0–5 | First C2f increment | Cumulative |
|---|---:|---:|---:|
| Commands excluding final `END` | 16,244 | 42,807 | 59,051 |
| `DMA_COPY2D` | 13,386 | 35,935 | 49,321 |
| `CONV_I8` | 1,600 | 4,800 | 6,400 |
| `EPILOGUE` | 1,200 | 1,200 | 2,400 |
| `ADD_I8` | 0 | 200 | 200 |
| `MAP_I8` | 0 | 600 | 600 |
| `FILL8` | 58 | 72 | 130 |
| DMA bytes | 1,999,320 | 2,369,408 | 4,368,728 |
| INT8 MACs | 40,550,400 | 45,875,200 | 86,425,600 |

The final `END` makes 59,052 commands. The functional runtime reports 59,044 FIFO `BUSY`/refill events and an eight-command final drain. These are synchronous submission counts, not transport timing.

The independent graph evaluation compares 1,843,200 values across the two stem outputs, four C2f convolution outputs, two split views, residual output, and concat output. Every value matches the command simulator. The [machine-readable report](../benchmarks/manifests/yolov8n-320-opset13/m6-first-c2f.json) records allocation, per-operation traffic, fixed-point coefficients, hashes, FIFO behavior, and FLOAT-reference errors.

## Current limits and next increment

The implementation validates the exact first C2f topology rather than accepting arbitrary ONNX graphs. It retains every boundary for differential testing and uses a deliberately direct DMA schedule. The command count therefore exposes a real scaling risk: extending this special-case schedule unchanged could exceed the current package limit or impose impractical command-dispatch traffic.

That next increment is complete in the [reusable graph schedule](reusable-graph-format.md), which covers nodes 22–47 in diagnostic and lifetime-reuse modes. This document remains the record for the earlier first-C2f package profile.
