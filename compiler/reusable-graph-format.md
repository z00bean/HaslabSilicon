# Reusable graph schedule

The M6 reusable scheduler lowers pinned YOLOv8n nodes 0–73 into the experimental `haslab.graph-schedule.v1` package profile. It covers the two-layer stem, three C2f blocks, and the two intervening stride-two convolutions. The implementation and schema remain experimental.

## Graph representation

`GraphIR` is an ordered, static-shape graph. Each `GraphTensor` fixes its HWC8 dimensions, binary32 scale, and optional alias relationship. Operations are `ConvSiluOp`, `AddOp`, or `ConcatOp`; split is represented by zero-allocation channel views. Validation rejects duplicate tensors, unavailable inputs, invalid HWC8 shapes or scales, duplicate producers, and missing final outputs before commands are emitted.

The current ONNX adapter validates the exact pinned nodes and extracts calibrated convolution and merge metadata. The IR itself is reusable across repeated C2f stages, but complete ONNX graph import remains unfinished.

## Allocation modes

Diagnostic mode gives every material tensor a distinct 64-byte-aligned range and retains every tensor and split view in the package manifest. This mode supports layerwise differential checking.

Release mode computes each material tensor's producing operation and last consuming operation, including alias uses. Its deterministic first-fit allocator permits two ranges to overlap only when their inclusive operation lifetimes are disjoint. Only declared final outputs are exposed. The full lifetime plan remains in the manifest so reuse can be audited.

For nodes 0–73, diagnostic output storage is 2,867,200 bytes. Release storage is 614,400 bytes, saving 2,252,800 bytes. Diagnostic and release compilation are separately byte deterministic.

## Scheduling behavior

Convolutions use output tiles no larger than 8×8, eight-channel groups, explicit halo fills, and INT32 chunk accumulation. The scheduler selects the largest height/width divisors whose complete input-group patch fits the 16 KiB input SRAM. Earlier regions remain 8×8; the 20×20 region uses 5×5 tiles. A layer whose complete weight set exceeds the 16 KiB weight SRAM is legal when one output group's weights fit; the scheduler loads that group for each spatial tile while keeping the tile's complete input patch resident. The manifest records this behavior as `one output group streamed per spatial tile`.

Residual addition and concat store explicit fixed-point rescaling parameters. Concat is materialized at a common scale. When all concat mapping records exceed the 3 KiB parameter SRAM, the scheduler loads one source's records at a time and records `one concat source at a time`. Split aliases allocate no additional bytes.

## Recorded evidence

The tracked [nodes 0–73 report](../benchmarks/manifests/yolov8n-320-opset13/m6-through-third-c2f.json) records:

| Measurement | Result |
|---|---:|
| Commands | 143,156 |
| DMA bytes | 12,116,376 |
| INT8 MACs | 302,694,400 |
| Diagnostic package bytes | 26,651,136 |
| Release package bytes | 26,543,616 |
| Diagnostic peak output bytes | 2,867,200 |
| Release peak output bytes | 614,400 |
| Diagnostic values compared | 3,225,600 |
| Diagnostic mismatches | 0 |
| Release final values compared | 51,200 |
| Release final mismatches | 0 |

Both packages reached the eight-command FIFO high-water mark. Each accepted all 143,156 commands, observed 143,148 full-FIFO responses and refills, and drained eight commands at the end. These are functional submission counts, not cycle, latency, or transport-throughput measurements.

## Remaining work

The next bounded extension is nodes 74–102, completing the backbone through the `model.7` downsample, `model.8` fourth C2f block, and `model.9` SPPF. This requires native MaxPool lowering and pooled-branch liveness. Whole-model completion also requires the neck branches, upsampling, the INT32 learned-head boundary, and the declared host tail. Command compression, transport timing, and hardware resource measurements remain separate work.
