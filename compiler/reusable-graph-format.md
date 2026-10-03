# Reusable graph schedule

The M6 reusable scheduler lowers the complete pinned YOLOv8n accelerator partition, nodes 0–217, into the experimental `haslab.graph-schedule.v1` package profile. It covers the two-layer stem, eight C2f blocks, five stride-two convolutions, SPPF, two nearest-neighbor upsampling operations, four skip concats, six learned class/box branches, six raw INT32 convolutions, and three exact INT32 boundary concats. The implementation and schema remain experimental.

## Graph representation

`GraphIR` is an ordered, static-shape graph. Each `GraphTensor` fixes its HWC8 dimensions, binary32 scale, and optional alias relationship. Operations are `ConvSiluOp`, `AddOp`, `ConcatOp`, `MaxPoolOp`, or `Upsample2Op`; split is represented by zero-allocation channel views. C2f bottlenecks explicitly declare whether a residual add exists, which permits both backbone residual blocks and neck non-residual blocks. Validation rejects duplicate tensors, unavailable inputs, invalid HWC8 shapes or scales, duplicate producers, and missing final outputs before commands are emitted.

The current ONNX adapter validates the exact pinned nodes and extracts calibrated convolution and merge metadata. The IR itself is reusable across repeated C2f stages, but complete ONNX graph import remains unfinished.

## Allocation modes

Diagnostic mode gives every material tensor a distinct 64-byte-aligned range and retains every tensor and split view in the package manifest. This mode supports layerwise differential checking.

Release mode computes each material tensor's producing operation and last consuming operation, including alias uses. Its deterministic first-fit allocator permits two ranges to overlap only when their inclusive operation lifetimes are disjoint. Only declared final outputs are exposed. The full lifetime plan remains in the manifest so reuse can be audited.

For nodes 0–217, diagnostic output storage is 8,131,200 bytes. Release storage is 2,355,200 bytes, saving 5,776,000 bytes. The release lifetime plan retains the earlier neck skips only through their final consumers, then holds the three terminal INT32 tensors through graph completion. Diagnostic and release compilation are separately byte deterministic.

## Scheduling behavior

Convolutions use output tiles no larger than 8×8, eight-channel groups, explicit halo fills, and INT32 chunk accumulation. The scheduler selects the largest height/width divisors whose complete input-group patch fits the 16 KiB input SRAM. Earlier regions remain 8×8; the 20×20 region uses 5×5 tiles. A layer whose complete weight set exceeds the 16 KiB weight SRAM is legal when one output group's weights fit; the scheduler loads that group for each spatial tile while keeping the tile's complete input patch resident. The manifest records this behavior as `one output group streamed per spatial tile`.

Residual addition and concat store explicit fixed-point rescaling parameters. Concat is materialized at a common scale. When all concat mapping records exceed the 3 KiB parameter SRAM, the scheduler loads one source's records at a time. If one source is itself too wide, it streams one 128-byte channel-group record per spatial tile. Wide residual additions stream one left/right channel-group pair per spatial tile. Wide convolutions retain the 1 KiB SiLU LUT and stream one 128-byte output-group epilogue record per spatial tile. Split aliases allocate no additional bytes.

`MaxPoolOp` lowers the pinned 5×5, stride-one, pad-two SPPF operation to `MAXPOOL5_I8`. Each 5×5 output tile and eight-channel group receives its actual neighboring pixels plus an explicit INT8 `-128` boundary halo. Pooling retains the input scale. The three material pool tensors remain live until the four-way SPPF concat consumes them.

`Upsample2Op` lowers the pinned asymmetric, nearest, floor-mode ONNX Resize to exact `UPSAMPLE2_I8`. Each 5×5 input tile becomes a 10×10 output tile, and the operation retains its input scale. The generic top-down neck extension concatenates the result with a named earlier tensor and appends a non-residual C2f block. It is used for both `model.6 → model.12` and `model.4 → model.15` skip paths.

The generic bottom-up neck extension applies a stride-two convolution to the current output, concatenates it with a named retained tensor at the resulting spatial shape, and appends a C2f block. Nodes 137–154 use it for `model.15 → model.16`, the retained `model.12` skip, and non-residual `model.18`; nodes 155–172 reuse it for `model.18 → model.19`, the retained `model.9` skip, and non-residual `model.21`.

`ConvRawOp` uses EPILOGUE mode 0 to add bias once and store little-endian INT32 `HWC8` values without requantization. Each logical channel records the exact binary32 product of its input activation scale and weight scale. `ConcatI32Op` copies the 64 regression channels followed by the 80 classification channels without arithmetic. For the 256-input-channel 3×3 branch, one complete output-group weight set is 18 KiB, so the scheduler streams one 8×8 input/output-channel weight chunk per reduction step to stay within the 16 KiB weight SRAM.

## Recorded evidence

The tracked [nodes 0–217 report](../benchmarks/manifests/yolov8n-320-opset13/m6-through-detection-head.json) records:

| Measurement | Result |
|---|---:|
| Commands | 412,092 |
| Command bytes | 52,747,776 |
| DMA bytes | 44,598,232 |
| INT8 MACs | 1,092,864,000 |
| Diagnostic package bytes | 75,688,384 |
| Release package bytes | 75,409,728 |
| Diagnostic peak output bytes | 8,131,200 |
| Release peak output bytes | 2,355,200 |
| Diagnostic values compared | 6,931,200 |
| Diagnostic mismatches | 0 |
| Release final values compared | 302,400 |
| Release final mismatches | 0 |

Both packages reached the eight-command FIFO high-water mark. Each accepted all 412,092 commands, observed 412,084 full-FIFO responses and refills, and drained eight commands at the end. These are functional submission counts, not cycle, latency, or transport-throughput measurements.

## Remaining work

The declared host tail at nodes 218–260 consumes the three INT32 HWC8 tensors with their per-channel scales and compares decoded outputs against an isolated pinned ONNX Runtime tail. The [host-tail report](../benchmarks/manifests/yolov8n-320-opset13/m6-host-tail.json) covers the initial synthetic output; [multi-input evidence](../benchmarks/manifests/yolov8n-320-opset13/m6-multi-input.json) now covers six diagnostic inputs and one real-image release replay. COCO command-path accuracy remains open. Command/relocation compaction, transport timing, and hardware resource measurements remain separate pre-RTL work.
