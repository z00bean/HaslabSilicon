# Reusable graph schedule

The M6 reusable scheduler lowers pinned YOLOv8n nodes 0–47 into the experimental `haslab.graph-schedule.v1` package profile. It covers the two-layer stem, the first C2f block, the `model.3` stride-two convolution, and the second C2f block with two residual bottlenecks. The implementation and schema remain experimental.

## Graph representation

`GraphIR` is an ordered, static-shape graph. Each `GraphTensor` fixes its HWC8 dimensions, binary32 scale, and optional alias relationship. Operations are `ConvSiluOp`, `AddOp`, or `ConcatOp`; split is represented by zero-allocation channel views. Validation rejects duplicate tensors, unavailable inputs, invalid HWC8 shapes or scales, duplicate producers, and missing final outputs before commands are emitted.

The current ONNX adapter validates the exact pinned nodes and extracts calibrated convolution and merge metadata. The IR itself is reusable across repeated C2f stages, but complete ONNX graph import remains unfinished.

## Allocation modes

Diagnostic mode gives every material tensor a distinct 64-byte-aligned range and retains every tensor and split view in the package manifest. This mode supports layerwise differential checking.

Release mode computes each material tensor's producing operation and last consuming operation, including alias uses. Its deterministic first-fit allocator permits two ranges to overlap only when their inclusive operation lifetimes are disjoint. Only declared final outputs are exposed. The full lifetime plan remains in the manifest so reuse can be audited.

For nodes 0–47, diagnostic output storage is 2,457,600 bytes. Release storage is 614,400 bytes, saving 1,843,200 bytes. Diagnostic and release compilation are separately byte deterministic.

## Scheduling behavior

Convolutions use 8×8 output tiles, eight-channel groups, explicit halo fills, and INT32 chunk accumulation. A layer whose complete weight set exceeds the 16 KiB weight SRAM is legal when one output group's weights fit; the scheduler loads that group for each spatial tile while keeping the tile's complete input patch resident. The manifest records this behavior as `one output group streamed per spatial tile`.

Residual addition and concat store explicit fixed-point rescaling parameters. Concat is materialized at a common scale. Split aliases allocate no additional bytes.

## Recorded evidence

The tracked [nodes 0–47 report](../benchmarks/manifests/yolov8n-320-opset13/m6-through-second-c2f.json) records:

| Measurement | Result |
|---|---:|
| Commands | 97,670 |
| DMA bytes | 6,951,768 |
| INT8 MACs | 194,560,000 |
| Diagnostic package bytes | 18,596,288 |
| Release package bytes | 18,530,880 |
| Diagnostic peak output bytes | 2,457,600 |
| Release peak output bytes | 614,400 |
| Diagnostic values compared | 2,764,800 |
| Diagnostic mismatches | 0 |
| Release final values compared | 102,400 |
| Release final mismatches | 0 |

Both packages reached the eight-command FIFO high-water mark. Each accepted all 97,670 commands, observed 97,662 full-FIFO responses and refills, and drained eight commands at the end. These are functional submission counts, not cycle, latency, or transport-throughput measurements.

## Remaining work

The next bounded extension is nodes 48–73: the `model.5` downsample and `model.6` third C2f block. Whole-model completion also requires the later backbone and neck branches, upsampling and pooling, the INT32 learned-head boundary, and the declared host tail. Command compression, transport timing, and hardware resource measurements remain separate work.
