# HASLAB v0 software/hardware contract

Copyright (C) 2026 Zubin Bhuyan.
SPDX-License-Identifier: CERN-OHL-S-2.0

**Status:** proposed specification, revision 0.1, September 20, 2026. This defines an intended hardware/software interface, not existing accelerator functionality. The release profile is `haslab-v0-int8`. The [revised architecture](haslab-v0-revised-architecture.md) supplies its rationale; this document takes precedence for interface details. The executable [Python numerical reference model](../reference/numerical-semantics.md) defines the arithmetic subset, and the [functional command simulator](../simulation/command-simulator.md) exercises the byte-level ABI and state behavior. No compiler, runtime, model export, cycle model, or RTL is implemented here.

The intended workflow is:

**PyTorch/framework export → ONNX → HASLAB compiler → `.hxb` model package → HASLAB runtime → FPGA; later, recompile for ASIC.**

A package includes the accelerator schedule and an explicit host detection tail. v0 is a complete host+FPGA detector pipeline, not a standalone FPGA detector. The initial workload remains a pinned YOLOv8n, batch one, 320×320. All backbone, neck, and learned detection-head convolutions run on the accelerator; image preparation and the specified final decoding/DFL/NMS stage run on the host.

“MUST” denotes a requirement of the proposed profile. Fields and IDs below are draft allocations to validate before ABI freeze. A future target may reuse the frontend and public API without accepting v0 binaries.

| Boundary | Producer → consumer | Representation |
|---|---|---|
| Model | Framework → compiler | Validated static ONNX graph |
| Compilation | Compiler → runtime | Versioned `.hxb` container, manifest, data, commands, host-tail graph |
| Device work | Runtime → FPGA controller | Fixed 128-byte little-endian commands |
| Numeric data | DMA ↔ compute/utility | Packed INT8 or INT32 tiles |
| Completion | Controller → runtime | Ordered sequence counters, fault registers |
| Results | Runtime → application | Typed outputs or declared decoded detections, with timing and provenance |

## 1. Tensor representation

### Logical shape and physical layout

Logical activations use named axes `N,C,H,W`, regardless of the frontend's storage layout. v0 requires `N=1`, positive static dimensions, and no dynamic ranks. Logical device dimensions and descriptor counts are unsigned 32-bit values unless a stricter bound is stated; the compiler rejects overflowing products or extents. Define `Cpad = 8 × ceil(C/8)`. Physical activation layout `HWC8` stores `[H][W][Cpad/8][8]`, with the last lane contiguous:

`byte_address(h,w,c) = base + ((h × W + w) × Cpad + c) × element_bytes`.

This is channel-last storage with channels padded to eight, not NCHW storage. Padding lanes MUST be zero on produced tensors and MUST not contribute to logical results. The host/compiler handles NCHW↔HWC8 conversion; hardware receives physical tiles. Three-channel RGB input therefore occupies eight lanes per pixel.

Convolution processes an output-channel group of at most eight. Packed tile weights `KHWCI8` are `[kernel_y][kernel_x][Cchunk_pad][8 output lanes]`, with `Cchunk_pad=round_up(Cchunk,8)`. Weight address in bytes is:

`weight_base + (((ky × kernel_w + kx) × Cchunk_pad + ci) × 8 + output_lane)`.

Unused input/output lanes in weights MUST be zero. A 64-bit word supplies eight output-channel weights. The compiler repacks ordinary ONNX `[Cout,Cin,Kh,Kw]` weights; model-level tensors do not impose a CPU/FPGA lane count.

Accumulator and epilogue tiles use `HWC8`, eight physical channels, INT32 or INT8 as declared. All multibyte values are little-endian; signed integers are two's complement. Floating-point scale metadata is IEEE binary32 on disk and never interpreted as executable arithmetic by v0 hardware.

### Views and padding

The host IR may express views with byte strides. The device accepts only dense local layouts described here. Materialize unsupported views with compiled copies, or reject them; arbitrary tensor strides do not appear in the compute command. Padded channels are storage only and do not alter logical shape.

Convolution boundary padding is explicitly materialized as INT8 zero in the local patch. Pooling boundary padding is INT8 -128. These are distinct semantics. Pooling outputs still zero their unused channel lanes. A temporary pool halo fill may place -128 in unused input lanes, which the kernel masks; this scratch pattern is not a logical tensor output. Compiler-generated copies MUST not read outside source allocations to obtain halos.

## 2. Memory representation

References are `(space:u32, offset:u32)`, where offset is a byte offset into a bounded region. They are not host pointers, physical system addresses, or virtual addresses. No v0 access can exceed the runtime's assigned external-memory aperture.

| Space ID | Name | Logical capacity | Role |
|---:|---|---:|---|
| 0 | EXT | Runtime-reported, maximum representable 32-bit byte capacity | Board DRAM aperture: weights, live activations, scratch |
| 1 | INPUT | 16,384 bytes | Dense patches and utility sources |
| 2 | WEIGHT | 16,384 bytes | Packed INT8 weight tiles |
| 3 | ACC | 8,192 bytes | INT32 partial sums; only convolution writes |
| 4 | OUTPUT | 8,192 bytes | Utility/epilogue destinations |
| 5 | PARAM | 3,072 bytes addressable | Bias/scaling records and SiLU LUT |
| — | Command FIFO | 1,024 bytes, not DMA-addressable | Eight 128-byte commands |

PARAM plus FIFO implements the earlier 4 KiB metadata budget, keeping **52 KiB total logical local storage**. A 128-byte MMIO staging register block and small controller/FIFO registers are additional; report actual BRAM/register utilization separately. Scratch memory contents after reset are unspecified; validity is cleared. Software MUST initialize anything it reads.

All DMA row starts, strides, and lengths are multiples of eight bytes. All local tensor/parameter bases are eight-byte aligned. EXT allocations start at 64-byte boundaries. Runtime allocation lengths include channel/tile padding. Hardware does not handle arbitrary byte tails or perform read-modify-write on neighboring allocations. This intentionally narrows the earlier architecture's open question about unaligned DMA.

Address arithmetic MUST use widened intermediates to detect overflow. For a 2-D transfer with positive `rows`, check `offset + (rows-1)×stride + row_bytes ≤ capacity` before issuing a bus access. Check source and destination independently. Zero-size hardware operations are invalid; the compiler removes them.

Only one execution context exists. Tensor, utility, and DMA operations are serialized. Live branch tensors reside in EXT until their last consumer; a two-buffer whole-model allocation is insufficient. A trusted compiler proves object-level liveness and nonaliasing; hardware validates region bounds and legal spaces, not a per-tensor protection table.

## 3. Tensor descriptors

Tensor descriptors are **host-side manifest/IR records**. FPGA logic does not traverse a tensor descriptor table or parse strings. The runtime resolves them to immediate command operands.

| Field | Required semantics |
|---|---|
| `tensor_id`, `name` | Stable numeric ID and graph/debug name |
| `logical_shape` | Named dimensions; activations N,C,H,W, weights Cout,Cin,Kh,Kw |
| `dtype` | `I8=1`, `I32=2`; `F8E4M3FN=16`, `F8E5M2=17`, `F16=18`, `F32=19` reserved for future device profiles |
| `layout` | HWC8, KHWCI8, or host-only layout; no implicit reinterpretation |
| `physical_shape`, `byte_strides` | Full padded extent and byte addressing |
| `buffer_id`, `byte_offset`, `storage_bytes` | Allocation and view extent, checked at load time |
| `quantization` | Scale data reference, zero point, logical scale axis, and numeric profile |
| `access` | Input, immutable constant, intermediate, or output; not a security permission |
| `producer`, `last_consumer` | Compiler liveness/debug evidence |
| `location` | Host, EXT allocation, or temporary local tile |

Scales are positive finite binary32. Activation scale count is one; weight scales have Cout entries indexed by logical output channel. Zero point MUST be zero. Final INT32 logits use per-channel scale `s_input × s_weight[channel]`; padding has no logical scale. Scale records and tensor names stay in the host package, while integer hardware multipliers are separately derived and serialized.

Unknown dtype/layout/profile, inconsistent extents, negative strides, shape overflow, scale-count mismatch, and conflicting buffer lifetimes are load/compile errors. A reserved dtype ID is not a capability promise.

## 4. DMA descriptors

`DMA_COPY2D` embeds this nine-word descriptor directly in the command payload; all fields are u32:

| Payload word | Field |
|---:|---|
| P0, P1 | Source space, source byte offset |
| P2, P3 | Destination space, destination byte offset |
| P4 | Bytes per row |
| P5 | Number of rows |
| P6, P7 | Source stride, destination stride, in bytes |
| P8 | Reserved, MUST be zero |

Contiguous DMA uses `rows=1` and both strides equal to row length. For multiple rows, each stride MUST be at least the row length. Strides/offsets/length obey eight-byte alignment. One endpoint MUST be EXT, the other INPUT, WEIGHT, OUTPUT, or PARAM. Both directions are available. DMA cannot address ACC, the command FIFO, or MMIO; no EXT→EXT or local→local DMA is supported.

The proposed board adapter presents a 64-bit data path, aligned incrementing bursts of at most 16 beats, split at row and 4 KiB boundaries. Exactly one burst transaction is outstanding, with read/write phases serialized. The adapter may translate to another board bus while preserving these semantics. DMA completes only after all read data has been committed locally or all writes have received successful responses. A bus error halts the queue; partial destination bytes are unspecified and outputs are invalid.

A separate `COPY2D` utility command uses the same payload for local copies between INPUT/OUTPUT. Source/destination ranges MUST not overlap; it is not `memmove`. PARAM and ACC are inaccessible to this utility copy. Rescaled copies use `MAP_I8` instead. These restrictions remove hidden coherence and port-arbitration requirements.

## 5. Accelerator commands

### Framing and compatibility

A command is 128 bytes: 32 little-endian u32 words, no native-language structure padding. Word zero contains ABI major in bits 31:24, ABI minor in bits 23:16, opcode in bits 15:0. This draft uses ABI **0.1**. Word one contains flags, word two a nonzero sequence number, and word three is reserved zero. Words 4–31 are payload words P0–P27. Unused payload words and reserved flag bits MUST be zero.

Sequence numbers increase by one from 1 after reset, without wraparound. Before exhausting u32 sequence numbers, drain and reset; runtime tokens also include a reset generation. A command with the wrong version, invalid flag, or sequence gap MUST not execute.

| Opcode | Command | Purpose |
|---:|---|---|
| 0x0001 | DMA_COPY2D | EXT↔local transfer |
| 0x0002 | FILL8 | Initialize INPUT/WEIGHT/OUTPUT/PARAM bytes |
| 0x0010 | CONV_I8 | One output tile, one input-channel reduction chunk |
| 0x0011 | EPILOGUE | Bias and conversion of a completed accumulator tile |
| 0x0020 | MAP_I8 | Same-shape INT8 rescale or identity copy |
| 0x0021 | ADD_I8 | Equal-shape scaled residual add |
| 0x0022 | MAXPOOL5_I8 | 5×5 stride-one max pooling |
| 0x0023 | UPSAMPLE2_I8 | Exact nearest-neighbor 2× spatial replication |
| 0x0024 | COPY2D | Local byte copy described in section 4 |
| 0x0030 | FENCE | In-order visibility point |
| 0x0031 | END | Successful inference boundary |

There are no branches, jumps, arbitrary microcode, CPU custom instructions, or YOLO-specific opcodes. Tiling loops are expanded by the compiler; kernel-internal loops use bounded descriptor dimensions.

### Fill and convolution payloads

FILL8 uses P0 space, P1 offset, P2 byte length, P3 fill byte represented as u32 in 0–255. Length is a positive multiple of eight; access is bounded. No flags are permitted.

CONV_I8 uses P0 INPUT offset, P1 WEIGHT offset, P2 ACC offset, P3 output height, P4 output width, P5 valid output lanes, P6 logical input channels in this chunk, P7 first logical input-channel index, P8 total logical input channels, P9 kernel size, P10 stride. All are u32; other fields are zero.

Constraints: output H/W in 1–8; valid output lanes 1–8; chunk channels 1–32; total channels 1–65,535; kernel 1 or 3; stride 1 or 2, equal in both spatial dimensions. Chunk start is a multiple of eight; every nonfinal chunk has a multiple-of-eight channel count. Dense convolution only, groups=1 and dilation=1.

The local input patch is dense HWC8 with height `(out_h-1)×stride+kernel`, analogous width, and padded chunk channels. It includes any required zero halo already. The command does not infer global padding or read EXT. The weight and accumulator layouts follow section 1; bounds are calculated from these dimensions.

Flag bit 0 is FIRST; bit 1 is LAST. FIRST requires chunk start zero and no open accumulator context, and initializes all physical accumulator lanes to zero. A non-FIRST command MUST match the active ACC offset, output shape, valid lanes, total channels, kernel, and stride, with chunk start exactly equal to the previously accumulated channel count. Only one such context exists. LAST is required exactly when `chunk_start+chunk_channels=total_channels`; it marks the context ready for EPILOGUE. A single-chunk tile sets both bits.

DMA/fill commands may replace inputs, weights, and parameters between chunks. Another tile cannot interleave its convolution or epilogue. Utility operations are rejected while an accumulator context exists. EPILOGUE requires a ready context, consumes it, and releases ACC. FENCE may observe an open context; END must reject one. This small context tracker prevents missing chunks and accidental double finalization without a general dependency scheduler.

### Epilogue and utility payloads

All operations below use one group of eight physical channels and 1–8 valid lanes. Sources/destinations are dense local HWC8, not general-strided views. Every output padding lane is zeroed. Integer dimensions are positive; memory extents must fit their spaces.

| Command | Ordered payload words beginning at P0 |
|---|---|
| EPILOGUE | ACC offset; OUTPUT offset; PARAM record offset; mode; LUT offset |
| MAP_I8 | INPUT offset; OUTPUT offset; H; W; valid lanes; PARAM record offset |
| ADD_I8 | INPUT source A offset; INPUT source B offset; OUTPUT offset; H; W; valid lanes; PARAM A offset; PARAM B offset |
| MAXPOOL5_I8 | INPUT offset; OUTPUT offset; output H; output W; valid lanes |
| UPSAMPLE2_I8 | INPUT offset; OUTPUT offset; input H; input W; valid lanes |
| FENCE / END | All payload zero |

EPILOGUE obtains shape from its active context. Mode 0 emits biased raw INT32 logits; mode 1 emits rescaled INT8; mode 2 emits INT8 through the SiLU LUT. For modes 0/1, LUT offset is zero. Mode 0 ignores scale parameters only after validating their required zero values. It still zeroes invalid output lanes. No epilogue modifies or persists biased ACC for subsequent reductions.

Utility H/W is limited to 1–64 as well as memory bounds; upsampling produces `2H×2W`. MAXPOOL5 takes a pre-padded `(H+4)×(W+4)` patch. The compiler provides -128 halo and actual neighboring pixels. Upsampling uses `output[y,x,c]=input[floor(y/2),floor(x/2),c]`. Pooling/upsampling retain the input scale. ADD requires equal logical shapes; generic broadcasting is not a device capability.

PARAM references point to eight 16-byte channel records: word 0 signed INT32 bias; word 1 nonnegative multiplier M ≤ 2^31-1; word 2 unsigned shift R ≤ 62; word 3 zero. Records are eight-byte aligned. Bias is zero for MAP/ADD. EPILOGUE mode 0 requires M=R=0; other modes use the specified scale pair. Padded-lane records are all zero. A SiLU LUT is 1,024 signed bytes, eight-byte aligned, wholly within PARAM, and disjoint from the referenced records.

FENCE completes after all preceding effects are visible within the device memory system. END adds the requirement that no accumulator context remains and marks model success. All commands remain serialized; neither command introduces concurrent execution.

## 6. Synchronization and submission

v0 has one queue and one model execution in flight. The runtime has exclusive device ownership during a session; sharing across processes is a host responsibility. Commands execute and retire strictly in sequence. No dependency tokens, cache-coherent tensor access, or out-of-order completion exists.

The host writes a complete 128-byte staging command, executes the platform's MMIO write barrier, then writes COMMIT with the same sequence number. If FIFO space exists, hardware atomically snapshots the staging words and increments LAST_ACCEPTED. If full, COMMIT is rejected without side effects and SUBMIT_RESULT reports BUSY; the runtime retries after checking capacity. There is no partial command in the execution FIFO. MMIO reads must flush posted writes according to the platform backend.

A staging command must not be overwritten until its commit result has been observed. Submitted external constants and inputs remain immutable until END or a completed abort/reset. Local memory is modified only by serialized commands. Writes to PARAM do not retroactively alter previous commands; hardware consumes parameters when a command executes.

Before submission, the runtime synchronizes host-written EXT buffers for device access. After successful END and DMA write completion, it synchronizes device-written buffers for CPU access. FENCE is not a replacement for host cache maintenance. A CPU host pointer is never placed into a device descriptor.

Polling is the required completion mechanism; interrupts are optional later. Waiting with a timeout does not cancel work or transfer buffer ownership. To abort, the runtime requests reset and waits for RESET_DONE. The board adapter must drain or safely terminate outstanding transactions before reuse; software must never free a buffer merely because a timeout occurred.

## 7. Status and error reporting

### Proposed MMIO map

Registers are aligned 32-bit words; unknown/reserved writes are rejected or ignored as documented by the platform, never interpreted as commands. The portable register window uses these byte offsets:

| Offset | Register | Access / semantics |
|---:|---|---|
| 0x000 | ID | Read: 0x48415330, numerical identifier for HAS0 |
| 0x004 | ABI | Read: major in bits 31:16, minor in 15:0; 0x00000001 for draft 0.1 |
| 0x008 | CAPABILITIES | Read: bits 0 INT8, 1 INT32 accumulate, 2 DMA2D, 3 utility, 4 serialized queue set; bit 8 FP8 clear; other bits zero |
| 0x00C | STATE | Read: 0 RESETTING, 1 IDLE, 2 RUNNING, 3 FAULT |
| 0x010 | FIFO_FREE | Read: number of complete command slots free, 0–8 |
| 0x014 | LAST_ACCEPTED | Read: last successfully enqueued sequence, zero after reset |
| 0x018 | LAST_COMPLETED | Read: last successfully retired sequence, zero after reset |
| 0x01C | LAST_END | Read: sequence of last successful END, zero after reset |
| 0x020 | ERROR_CODE | Read: sticky first execution fault code |
| 0x024 | ERROR_SEQ | Read: failing command sequence |
| 0x028 | ERROR_SPACE | Read: implicated space or 0xFFFFFFFF if not applicable |
| 0x02C | ERROR_OFFSET | Read: implicated offset where available, else zero |
| 0x030 | ERROR_FIELD | Read: command word index, or 0xFFFFFFFF if not applicable |
| 0x034 | CONTROL | Write bit 0 requests reset; other bits must be zero |
| 0x038 | RESET_GENERATION | Read: increments after successful reset; not a persistent identity |
| 0x03C | EXT_BYTES | Read: configured aperture capacity |
| 0x040 | SUBMIT_RESULT | Read: 0 accepted/no result, 1 FIFO busy, 2 fault/reset blocked, 3 invalid framing or sequence |
| 0x044 | COMMIT | Write: atomically submit staged record using this sequence number |
| 0x048 | CYCLES_LO | Read: low execution-cycle count; latches high word until next low read |
| 0x04C | CYCLES_HI | Read: latched high word |
| 0x050 | INPUT_BYTES | Read: 16,384 |
| 0x054 | WEIGHT_BYTES | Read: 16,384 |
| 0x058 | ACC_BYTES | Read: 8,192 |
| 0x05C | OUTPUT_BYTES | Read: 8,192 |
| 0x060 | PARAM_BYTES | Read: 3,072 |
| 0x100–0x17C | CMD_STAGING[0..31] | Write: command words |

RUNNING means an inference has begun and has not reached END, even if the FIFO temporarily empties while the host refills it. The first accepted work command after IDLE starts RUNNING. END returns to IDLE only after its effects are complete; queued work may start the next invocation. The v0 runtime does not queue a second invocation before the first completes. The cycle counter counts RUNNING cycles, including host-refill stalls, and resets on device reset. Measure submission/transfer/wall time separately.

COMMIT framing/version/sequence checks occur before enqueue and can reject a submission without poisoning earlier work. LAST_ACCEPTED advances only on accepted commands. Opcode-specific validation occurs before that command's effects; invalid payloads create a sticky execution fault. After an execution fault, no later command executes and COMMIT is blocked until reset. An unknown opcode may be accepted as framed data but faults before execution.

| Code | Execution fault | Meaning |
|---:|---|---|
| 0 | NONE | No fault |
| 1 | BAD_OPCODE | Unknown/unsupported operation |
| 2 | BAD_FIELD | Reserved bits, invalid enum, dimension, or mode |
| 3 | ALIGNMENT | Offset, stride, or length violates alignment |
| 4 | BOUNDS | Overflow or out-of-region access |
| 5 | BAD_SPACE | Illegal source/destination space |
| 6 | BAD_STATE | Invalid accumulator sequence or END during unfinished reduction |
| 7 | ARITH_OVERFLOW | INT32 accumulation/bias outside range |
| 8 | DMA_BUS | Bus response error |
| 9 | INTERNAL | Internal assertion/state fault exposed by implementation |

Overflow detection must prevent successful retirement of an invalid numeric operation; destination state may be partial. There is no recovery by saturating an accumulator. The compiler's static bound proof is required in addition to hardware overflow checks.

Host/runtime errors are separate from device fault codes: invalid model package, ABI/capability mismatch, unsupported ONNX semantics, allocation failure, device busy, transport failure, and wait timeout. Reports include the graph node/tensor mapping from sequence IDs. A timeout is not automatically a hardware fault; a lost completion must be distinguished from a numerical mismatch.

On fault/reset, outstanding output contents are invalid and the runtime must not return them as detections. Reset empties the queue, clears context and completion/error counters, and invalidates local contents after transactions are quiescent. Weights/parameters are reloaded before execution. If quiescence cannot be established, the backend marks the device unusable until a platform reset.

## 8. FP8 formats and forward compatibility

**v0 has no native FP8 capability.** Compilation for v0 MUST reject FP8 device execution. Storage being eight bits wide does not allow FP8 bytes to be fed to INT8 MACs. Explicit recompilation/calibration to INT8 is a different package with a reported numerical change.

For v1, reserve ONNX-compatible E4M3FN as the first FP8 representation and E5M2 as a later optional representation. E4M3FN has sign/exponent/fraction widths 1/4/3, exponent bias 7, and no infinity encoding. E5M2 uses 1/5/2 and bias 15, and includes infinities. FNUZ variants are distinct and must not be silently reinterpreted. [ONNX FP8 encoding reference](https://onnx.ai/onnx/technical/float8.html)

A future profile must specify subnormal decoding, signed zero, canonical NaN, nearest-even conversion, finite overflow saturation, infinity conversion, scale placement, and accumulation order. Default intent is FP32 accumulation; FP16 accumulation requires a separately validated mode. No FP8 opcode or payload is frozen in the v0 ABI. Future support requires both a device capability and a compatible numeric-profile/package version.

## 9. INT8 format and exact conversion semantics

The device interprets INT8 as signed -128…127, with zero point zero. The compiler may select symmetric clipping limits during calibration, but the arithmetic supports the entire signed range. Quantization means `q = clamp[-128,127](RNE(real/scale))`, with positive finite scale.

Define RNE for a signed rational `z / 2^R` mathematically: round to nearest integer, breaking an exact half tie toward the even integer, symmetrically for positive and negative inputs. R=0 returns z. Thus RNE(5/2)=2, RNE(7/2)=4, RNE(-5/2)=-2, and RNE(-7/2)=-4. Neither truncation toward zero nor “add half then arithmetic shift” is a substitute.

Hardware requantization uses:

`requant(x,M,R) = clamp_int8(RNE((signed_64(x) × M) / 2^R))`.

M is 0…2^31-1 and R is 0…62. M=0 is valid for an intentionally zeroed output but must be declared by the compiler. The compiler approximates a positive real scale ratio with M/2^R, records both the ideal ratio and the chosen integers, and reports approximation error. A deterministic selection rule chooses the smallest absolute ratio error among legal pairs, breaking ties with smaller R then smaller M; overflow and underflow-to-zero constraints are checked before acceptance. The serialized integers are authoritative for hardware/simulator equivalence.

For a completed convolution sum S in output channel c:

- Bias is precomputed as `RNE(real_bias[c]/(s_input×s_weight[c]))` and must fit INT32.
- Form X=S+bias exactly, with INT32 overflow detection.
- Linear INT8 output uses M/2^R approximating `(s_input×s_weight[c])/s_output`.
- Raw INT32 output writes X; the host dequantizes using the corresponding channel scale.

ADD uses a single final rounding:

`out = clamp_int8(RNE((qa×Ma + qb×Mb) / 2^R))`.

Parameter pairs A/B MUST have equal R for corresponding lanes. Ratios approximate `sa/sout` and `sb/sout` with a shared R; choose the legal shared shift minimizing maximum relative ratio error, then smaller R, then smaller multipliers. Positive ratios that cannot be represented within the agreed compiler error budget fail compilation. Use signed 64-bit intermediates; do not saturate either contribution before addition. MAP is the single-input formula with no bias.

For SiLU, the manifest stores a positive grid step delta and output scale. Channel coefficients map X into `j=clamp[-512,511](RNE(X×M/2^R))+512`, with ratio approximately `(s_input×s_weight[c])/delta`. Byte j in the serialized LUT is `clamp_int8(RNE(SiLU((j-512)×delta)/s_output))`. The compiler generates the table in sufficiently high precision, records its bytes/hash, and evaluates clipping/approximation error on the model. Hardware uses the bytes, not an exponential function. No floating-point scalar interpretation occurs in v0.

These are HASLAB numerical semantics. They are not a claim that arbitrary floating-point ONNX Q/DQ evaluation or arbitrary vendor INT8 kernels are bit-identical. ONNX's own QuantizeLinear also specifies nearest-even rounding, but scale representation and fusion remain distinct concerns. [ONNX QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)

## 10. Accumulator format and lifecycle

INT8 products accumulate exactly in signed INT32, including across input-channel chunks. No saturation, wraparound, or intermediate requantization is permitted. Reduction order is kernel y, kernel x, then input channel within each chunk; chunk order increases logical input channel. Exact integer arithmetic makes the result order-independent only while bounds hold.

For each output channel the compiler proves a conservative bound such as `128 × sum(abs(q_weight[k])) + abs(bias) ≤ INT32_MAX`, including every real reduction term. A uniform `K×16,384+abs(bias)` bound is acceptable but more conservative. Padded weights are zero. Reject unsafe operations; chunking does not make an overflowing final sum safe. Independent hardware checks catch overflow during accumulation and bias addition.

FIRST opens a context and clears its tile; intermediate CONV commands add contributions; LAST closes the reduction; EPILOGUE adds bias once and retires the context. ACC storage cannot be modified by DMA, fill, or arbitrary host writes. This avoids stale partial-sum reuse. Final logits leave through OUTPUT in INT32 mode, then DMA to EXT; the runtime does not directly expose ACC.

FP32/FP16 accumulation is outside v0. Future floating-point accumulation must have a separate numeric contract and cannot inherit the integer bitwise-equivalence rule.

## 11. Supported tensor operations

| Operation | v0 limits | Execution mechanism |
|---|---|---|
| Dense convolution | 1×1/3×3, stride 1/2, dilation 1, groups 1, static N=1 | Chunked CONV_I8 + EPILOGUE |
| Bias | Constant, per output channel | EPILOGUE, exactly once |
| Linear requantization | Per-tensor activations, lane-specific conversion constants | EPILOGUE or MAP_I8 |
| SiLU | Recognized x×sigmoid(x), calibrated LUT approximation | EPILOGUE mode 2 |
| Residual add | Equal shapes, common output quantization domain | ADD_I8 |
| Maximum pooling | 5×5, stride 1, symmetric two-pixel boundary halo | MAXPOOL5_I8 with compiled halo |
| Nearest upsampling | Exactly 2× height/width | UPSAMPLE2_I8 |
| Channel concat/split | Static offsets, compatible quantization or explicit rescaling | Copy/MAP schedules or compiler-only views |
| Constant/shape computations | Fully evaluable at compilation | No device execution |
| Fill/copy | Aligned bounded rows; no overlapping local copy | FILL8, COPY2D, DMA_COPY2D |

For v0, split/concat channel boundaries must be multiples of eight except the final channel tail. This restriction matches the intended blocked workload profile but must be confirmed from the export. Arbitrary byte channel gathers are not secretly implemented by DMA. If a real graph needs them, either revise the bounded utility command before ABI freeze or fail the profile explicitly.

Pure reshape/view removal is legal only when it preserves logical element order. A transposition that changes bytes requires a supported copy schedule; v0 has no general transpose instruction. Empty operations are folded on the host. The compiler may express a compatible fixed dot product as a 1×1 convolution, but v0 does not promise arbitrary ONNX MatMul/Gemm.

## 12. Unsupported operations and rejection policy

Device v0 does not support FP8/FP16/FP32 arithmetic, unsigned/asymmetric quantized arithmetic, arbitrary per-channel activation quantization, general broadcast arithmetic, depthwise/grouped/dilated/transposed convolution, attention, softmax, LayerNorm, GELU, arbitrary interpolation/ROI resize, recurrent operators, random sampling, training, sparsity, dynamic control flow, dynamic shapes, or batch sizes above one.

These are unsupported **device** operations. Selected softmax/shape/box operations in the declared host detection tail remain allowed. That exception is a named graph partition containing terminal DFL/class/box decoding; it is not permission to run an unsupported backbone or head layer on a CPU. All other unsupported nodes fail compilation with graph node ID, operator domain/version, shape/type, unsupported attribute, and available remedy.

No silent approximation of an unsupported operator is permitted. SiLU LUT and fixed-point scaling are explicitly versioned approximations covered by model accuracy evaluation. A compiler cannot delete attention, replace an activation, or change input resolution merely to make compilation succeed.

## 13. ONNX operator mapping

Proposed initial frontend profile: ONNX default-domain **opset 13**, static shapes, FLOAT inputs/initializers, no training graph, and an explicit external-data manifest if weights are separate. Pin the ONNX IR version and exporter build after the workload audit. Other opsets are rejected unless a separately tested conversion yields this profile with equivalent semantics; merely changing the opset number is invalid.

| ONNX form | Lowering / constraint |
|---|---|
| Conv | Constant weights, optional constant bias, valid v0 kernel/stride/group/dilation; pack weights and tile; normalize legal padding to explicit local halo |
| BatchNormalization | Fold only constant inference parameters into a preceding compatible Conv; reject unfused runtime BN |
| Sigmoid then Mul of the same input | Recognize exact SiLU dataflow, preserve any other uses, emit calibrated epilogue LUT |
| Add | Equal-shaped tensors → scaled ADD; safe constant-bias patterns may fuse into Conv before quantization |
| MaxPool | 5×5, stride 1, pads 2, dilation 1, ceil_mode=0, no used indices output |
| Resize | Constant 2× spatial nearest resize, coordinate_transformation_mode=asymmetric, nearest_mode=floor; other forms require a proven equivalent rewrite or rejection |
| Concat | Logical channel axis, static shapes, eight-aligned boundaries; rescale inputs to common output scale if required |
| Split / Slice | Static channel regions within the supported alignment profile; views or copy schedules |
| Reshape / Transpose | Compile-time logical/layout transformation if supported by the copy/layout plan; never blindly reinterpret blocked bytes |
| Constant / Shape / Gather / Unsqueeze / Squeeze / Cast | Fold only where inputs and resulting values are known at compile time, otherwise reject outside the host tail |
| Identity / inference Dropout | Eliminate only when semantics and used outputs permit |
| Softmax, final Sigmoid, terminal DFL arithmetic | Declared host-tail graph only |
| NonMaxSuppression | Host implementation or declared host graph with matched threshold, ordering, and box conventions |
| QuantizeLinear / DequantizeLinear / QLinearConv | Not accepted as arbitrary v0 input; v0 generates its own calibrated profile from FLOAT ONNX |
| MatMul / Gemm | Reject unless a specifically validated static rewrite fits the dense 1×1 profile; no general promise |
| Custom-domain operators | Reject |

An operator name alone is insufficient: support includes attributes, broadcasting, ranks, numerical types, and the operator schema active at the model's opset. Official Conv and Resize definitions are the reference for frontend semantics. [ONNX Conv](https://onnx.ai/onnx/operators/onnx__Conv.html), [ONNX Resize](https://onnx.ai/onnx/operators/onnx__Resize.html)

This proposed opset/profile must be tested against the pinned export before RTL. If the actual exporter uses another nearest-resize convention, prove its index mapping equivalent for the fixed shape or amend the supported utility profile; do not silently substitute asymmetric/floor.

## 14. Model compilation, lowering, and package format

### Compilation pipeline

1. **Import and validate.** Check model structure, shapes, opset/IR, external weights, hashes, and operator attributes. Produce coverage and proposed partition reports first.
2. **Establish the reference.** Run the exported floating-point graph with the exact preprocessing and output conventions. Record weight/exporter/model identities. This is future work, not a result claimed by this specification.
3. **Normalize and partition.** Fold constants/inference BN, recognize SiLU, determine the terminal host decoding partition, and retain node provenance. Hardware executes all learned detector convolutions.
4. **Calibrate.** Use a recorded representative calibration dataset distinct from the evaluation set. Choose tensor/channel scales and LUT grids, derive integer parameters, and evaluate against floating-point accuracy before proceeding.
5. **Lower.** Emit explicit quantized Conv, Add, pool, resize, copy, and conversion IR. Enforce equal-scale requirements at merges and define final INT32 logit dequantization.
6. **Allocate and tile.** Use a deterministic liveness allocator for EXT; allocate fixed local tile offsets. Fit halos and both residual sources. Prove accumulator bounds, parameter footprints, zero padding, and every access range.
7. **Schedule.** Expand the serialized command sequence. Keep an output tile in ACC across reduction chunks, load the correct weights/inputs, execute epilogue once, and store it. Preserve tensors reused by branches.
8. **Package and validate.** Emit `.hxb`, operator/numeric/coverage reports, memory allocation, and sequence-to-node debug mapping. A future command simulator must reproduce the package before it is accepted for FPGA release.

A representative tile schedule is: FILL input halo → DMA valid input rows → DMA packed weights → CONV FIRST → repeat input/weight loads and CONV continuation → CONV LAST → DMA epilogue parameters/LUT → EPILOGUE → DMA output rows. FIRST/LAST may be combined for one chunk. At frame end issue END after all output stores. Commands are bounded general operations, not hand-authored YOLO-layer opcodes.

### `.hxb` container proposal

Use a simple sectioned file with no executable host code and no hardware-side parser. All integer fields are little-endian. Header is 64 bytes:

| Byte offset | Type / field |
|---:|---|
| 0 | 8 bytes ASCII `HSLBHXB0` |
| 8, 10 | u16 format major=0, minor=1 |
| 12 | u32 header_bytes=64 |
| 16 | u32 flags=0 |
| 20 | u32 section_count |
| 24 | u64 section_table_offset |
| 32 | u64 total_file_bytes |
| 40 | u32 manifest_section_index, zero-based |
| 44–63 | Reserved zero bytes |

Section table entries are 64 bytes: u32 type at 0, u32 flags=0 at 4, u64 file offset at 8, u64 byte length at 16, 32 raw SHA-256 bytes at 24, eight zero bytes at 56. Section starts and the table start are 64-byte aligned. Hashes cover exact section bytes, excluding alignment padding. Padding is zero. Sections/header/table must not overlap, and all bounds are validated with widened arithmetic.

Section types: 1 MANIFEST_UTF8_JSON, 2 COMMANDS, 3 CONSTANT_DATA, 4 HOST_TAIL_ONNX, 5 DEBUG_UTF8_JSON. One manifest, command section, and constant section are required; at most one host-tail/debug section. Unknown types/flags fail this v0 loader. JSON disallows duplicate keys and nonfinite numbers. The runtime rejects malformed, truncated, oversized, or incompatible packages before device submission. Hashes detect corruption; they do not authenticate a publisher.

The manifest contains profile/ABI requirements; model/export/tool hashes; preprocessing contract; tensor descriptors; external allocation requirements; constant placement; integer/LUT profile; the expected host-tail partition; output decoding/NMS configuration; calibration/evaluation provenance; and command relocation entries. Binary scale arrays are stored in CONSTANT_DATA with a declared little-endian F32 type; JSON does not redefine their numerical bits.

COMMANDS contains 128-byte records with contiguous relative sequence numbers starting at one. Runtime patches sequence numbers to the current device epoch and applies only declared EXT-address relocations. Runtime-generated constant-upload commands also receive sequence numbers and are marked as runtime setup in diagnostics. A relocation identifies command index, payload word, buffer ID, and byte addend. It must target an EXT offset field in DMA_COPY2D, resolve inside the buffer/aperture, and never modify opcodes, sizes, local references, or parameters. Input binding cannot change dimensions or allocation requirements. Runtime verifies original hashes before relocation.

CONSTANT_DATA holds packed weights, integer parameters, LUT bytes, and scale metadata, each with explicit offsets/lengths and allocation placement. HOST_TAIL_ONNX is data for a supported host evaluator; arbitrary plugins or embedded scripts are not loaded. The manifest identifies dequantized tail input names and original graph outputs. External resources are resolved only from declared packaged data or explicit runtime paths, not downloaded implicitly.

Preprocessing is data, not a hidden default: input size, RGB/BGR order, image resize/letterbox policy, interpolation, pad values, normalization, scale, and tensor layout must be specified and versioned. Output metadata defines original-image coordinate mapping, class ordering, confidence thresholds, NMS IoU threshold, maximum detections, and tie/order policy. Their exact values are frozen with the workload audit.

The package is target-specific. ASIC migration may change memory sizes, command ABI, numeric modes, and tiling; recompilation is expected. ONNX, logical tensor descriptors, and the high-level runtime lifecycle are the stable layers.

## 15. Runtime API

Define a small C-compatible ownership/lifecycle model, with a future Python wrapper. The names below are proposed API contracts, not implemented functions or binding code.

| API operation | Contract |
|---|---|
| `haslab_open` | Open a platform device/backend and establish exclusive session ownership |
| `haslab_query_capabilities` | Return ABI, numeric/operation capabilities, region capacities, alignment, and backend identity |
| `haslab_load_model` | Validate `.hxb`, expose required host stages, allocate EXT workspace, place constants, return model handle |
| `haslab_get_io_info` | Return input/output shapes, types, quantization, preprocessing, and partition metadata |
| `haslab_prepare_input` | Optional host helper implementing the declared image preparation; returns a bindable input tensor |
| `haslab_bind_input` | Bind an exactly typed/shaped buffer; copy/repack explicitly where necessary; no inferred resize |
| `haslab_submit` | Start one inference and return token `(session, reset_generation, final_END_sequence)`; BUSY if another is active |
| `haslab_query` | Return pending, accelerator complete/host tail pending, success, or failure |
| `haslab_wait` | Wait up to timeout; full success includes the declared host tail; timeout leaves work/buffer ownership active |
| `haslab_get_outputs` | Return valid final outputs only after full success; optional raw logits via an explicitly named diagnostic mode |
| `haslab_get_profile` | Return hardware cycles, submit/refill time, transfers, host preprocessing/tail time, and total latency |
| `haslab_get_error` | Return structured runtime/device error and graph/command location |
| `haslab_reset` | Quiesce DMA, invalidate pending work, reset device, advance generation; force reload before reuse |
| `haslab_unload_model` | Release model/workspace only after no active job owns them |
| `haslab_close` | Drain or safely abort before releasing transport/device ownership |

The model handle owns constants and workspace; application input/output buffer lifetimes extend through completion or confirmed reset. Runtime-owned returned buffers remain valid until explicitly released or the model is unloaded. Bindings must document whether submission copies inputs or retains them; the default v0 contract retains bindings through job completion. No zero-copy promise is made for every host platform.

Loading the v0 detector requires explicit acceptance of its declared host stages through the API options; otherwise return HOST_PARTITION_REQUIRED. This is a one-time model-loading choice, not per-layer fallback. The runtime reports those stages in model information and profiling. A raw-accelerator diagnostic request is distinct from a completed detector result.

Input tensors may come from a host perception pipeline, allowing later policy integration without changing the device command interface. v0 does not expose a recurrent-policy state API; v1 may add state handles backed by SRAM/EXT with explicit reset semantics. RISC-V firmware can later assume command submission while the public model API remains similar.

## 16. Ambiguities and pre-RTL resolution gates

The following are still unmeasured or intentionally pending. They must not be confused with implemented guarantees.

| Item | Proposed default | Resolution required before relevant RTL freeze |
|---|---|---|
| Exact workload artifact | YOLOv8n, 1×3×320×320, opset 13 | Pin weights, exporter, ONNX IR, graph hashes; inspect all operators/attributes and terminal partition |
| Detection boundary | FPGA learned head; host DFL/class/box/NMS | Verify actual exported graph, tail input names/scales, ordering and coordinate conventions |
| Quantization accuracy | Symmetric INT8; 1,024-byte SiLU LUT | Calibrate/evaluate; proposed ≤1 absolute mAP point loss versus identical FLOAT model; agree actual acceptance budget |
| Scale approximation | Integer M/R formulas in section 9 | Freeze acceptable relative/absolute conversion error and deterministic high-precision coefficient/table generation |
| Quantization merges | One output scale for residual/concat | Prove scales and eight-aligned split boundaries fit the selected graph without hidden requantization errors |
| Board/transport | Existing bridge and external DRAM | Choose board; verify MMIO ordering, DMA response semantics, aperture size, coherent/noncoherent host buffers, reset quiescence |
| Physical memory mapping | 52 KiB logical stores, serial accesses | Check BRAM widths/depth rounding, synchronous latency, utilization, clock crossing and parameter/LUT access timing |
| Throughput versus command traffic | Serialized 128-byte FIFO commands | Estimate per-frame commands, host refill time, patch copies, accumulator spill cycles; decide if adequate for a useful v0 |
| Convolution chunking | 8-lane output group, ≤32 input channels/chunk | Verify all layer bounds, tail layouts, and tile halos; first-layer RGB padding must match weight packing |
| ABI/register allocations | Draft 0.1 tables in this document | Review command encodings, invalid-field behavior, reset/commit races, and test vectors together before freezing |
| Package/preprocessing schema | Sectioned `.hxb`, explicit manifest | Freeze manifest required keys, per-section size limits, constants placement, and exact image/box semantics before compiler/runtime implementation |
| Future FP8 | E4M3FN + FP32 accumulation | Separate v1 numerical contract and arithmetic feasibility; does not block INT8 v0 RTL |
| ASIC SRAM/PHY | No FPGA-IP portability assumption | Resolve at ASIC feasibility stage, not by adding speculative v0 features |

The proposed command arithmetic/layout choices are sufficiently concrete for a reference-model review. The workload and numerical gates still block responsible RTL freeze: a byte-level interface is not evidence that a calibrated modern detector meets accuracy or fits the selected FPGA.

## 17. Required conformance evidence before a v0 release

Future tests must cover independent numeric reference agreement; positive/negative rounding ties; accumulator overflow and cross-chunk continuity; padded RGB/channels; pool boundary sentinels; mismatched residual scales; concat liveness; malformed DMA bounds; FIFO full retries; sequence/version rejection; partial DMA failure; reset with outstanding bus work; stale generation tokens; host cache maintenance; and a full pinned detector accuracy/latency run.

A representative byte-layout test should show that pixel (0,1), logical channel 2 in an HWC8 tensor with C=3 is at byte offset 10, while the next row begins at W×8. A 3×3, 32-input-channel, eight-output-lane weight tile occupies 2,304 bytes. An 8×8 output tile uses 2,048 ACC bytes and 512 INT8 OUTPUT bytes. These examples are specification checks, not test implementation.

The Python golden model and functional command simulator now accompany the numerical and byte-level portions of this contract. Remaining pre-RTL work includes resolving the pinned workload and quantization-accuracy gates and generating independent conformance vectors for a future RTL implementation. No RTL has been written.
