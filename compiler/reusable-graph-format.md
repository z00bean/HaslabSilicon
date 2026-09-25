# Reusable graph schedule

The M6 reusable scheduler lowers pinned YOLOv8n nodes 0–119 into the experimental `haslab.graph-schedule.v1` package profile. It covers the complete backbone and first top-down neck stage: the two-layer stem, five C2f blocks, three stride-two convolutions, SPPF, nearest-neighbor upsampling, and a backbone skip concat. The implementation and schema remain experimental.

## Graph representation

`GraphIR` is an ordered, static-shape graph. Each `GraphTensor` fixes its HWC8 dimensions, binary32 scale, and optional alias relationship. Operations are `ConvSiluOp`, `AddOp`, `ConcatOp`, `MaxPoolOp`, or `Upsample2Op`; split is represented by zero-allocation channel views. C2f bottlenecks explicitly declare whether a residual add exists, which permits both backbone residual blocks and neck non-residual blocks. Validation rejects duplicate tensors, unavailable inputs, invalid HWC8 shapes or scales, duplicate producers, and missing final outputs before commands are emitted.

The current ONNX adapter validates the exact pinned nodes and extracts calibrated convolution and merge metadata. The IR itself is reusable across repeated C2f stages, but complete ONNX graph import remains unfinished.

## Allocation modes

Diagnostic mode gives every material tensor a distinct 64-byte-aligned range and retains every tensor and split view in the package manifest. This mode supports layerwise differential checking.

Release mode computes each material tensor's producing operation and last consuming operation, including alias uses. Its deterministic first-fit allocator permits two ranges to overlap only when their inclusive operation lifetimes are disjoint. Only declared final outputs are exposed. The full lifetime plan remains in the manifest so reuse can be audited.

For nodes 0–119, diagnostic output storage is 3,635,200 bytes. Release storage is 614,400 bytes, saving 3,020,800 bytes. The release lifetime plan retains `model.6` from operation 25 through its neck concat use at operation 40 without increasing the prior peak. Diagnostic and release compilation are separately byte deterministic.

## Scheduling behavior

Convolutions use output tiles no larger than 8×8, eight-channel groups, explicit halo fills, and INT32 chunk accumulation. The scheduler selects the largest height/width divisors whose complete input-group patch fits the 16 KiB input SRAM. Earlier regions remain 8×8; the 20×20 region uses 5×5 tiles. A layer whose complete weight set exceeds the 16 KiB weight SRAM is legal when one output group's weights fit; the scheduler loads that group for each spatial tile while keeping the tile's complete input patch resident. The manifest records this behavior as `one output group streamed per spatial tile`.

Residual addition and concat store explicit fixed-point rescaling parameters. Concat is materialized at a common scale. When all concat mapping records exceed the 3 KiB parameter SRAM, the scheduler loads one source's records at a time. If one source is itself too wide, it streams one 128-byte channel-group record per spatial tile. Wide residual additions stream one left/right channel-group pair per spatial tile. Wide convolutions retain the 1 KiB SiLU LUT and stream one 128-byte output-group epilogue record per spatial tile. Split aliases allocate no additional bytes.

`MaxPoolOp` lowers the pinned 5×5, stride-one, pad-two SPPF operation to `MAXPOOL5_I8`. Each 5×5 output tile and eight-channel group receives its actual neighboring pixels plus an explicit INT8 `-128` boundary halo. Pooling retains the input scale. The three material pool tensors remain live until the four-way SPPF concat consumes them.

`Upsample2Op` lowers the pinned asymmetric, nearest, floor-mode ONNX Resize to exact `UPSAMPLE2_I8`. Each 5×5 input tile becomes a 10×10 output tile, and the operation retains its input scale. The first neck stage concatenates that output with the earlier `model.6` tensor and then executes a non-residual C2f block.

## Recorded evidence

The tracked [nodes 0–119 report](../benchmarks/manifests/yolov8n-320-opset13/m6-through-first-neck.json) records:

| Measurement | Result |
|---|---:|
| Commands | 235,474 |
| DMA bytes | 20,101,656 |
| INT8 MACs | 453,427,200 |
| `MAXPOOL5_I8` commands | 192 |
| `UPSAMPLE2_I8` commands | 128 |
| Diagnostic package bytes | 42,995,264 |
| Release package bytes | 42,819,200 |
| Diagnostic peak output bytes | 3,635,200 |
| Release peak output bytes | 614,400 |
| Diagnostic values compared | 4,070,400 |
| Diagnostic mismatches | 0 |
| Release final values compared | 51,200 |
| Release final mismatches | 0 |

Both packages reached the eight-command FIFO high-water mark. Each accepted all 235,474 commands, observed 235,466 full-FIFO responses and refills, and drained eight commands at the end. These are functional submission counts, not cycle, latency, or transport-throughput measurements.

## Remaining work

The next bounded extension is nodes 120–136: exact nearest-neighbor `model.13` upsampling, `model.14` channel concat with the long-lived `model.4` skip tensor, and the non-residual `model.15` C2f block. Whole-model completion also requires the bottom-up neck branches, INT32 learned-head boundary, and declared host tail. Command compression, transport timing, and hardware resource measurements remain separate work.
