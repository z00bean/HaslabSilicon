# HASLAB reference-model numerical semantics

Copyright (C) 2026 Zubin Bhuyan.
SPDX-License-Identifier: CERN-OHL-S-2.0

This document describes the executable Python golden model in `reference/haslab_ref`. The model defines intended numerical results for future compiler, simulator, and RTL verification. It does not claim that v0 hardware, FPGA execution, or a supported model pipeline exists.

## General rules

- Integer storage uses two's-complement signed INT8 or INT32.
- Multibyte hardware storage is little-endian; NumPy array byte order is irrelevant until data is serialized.
- Operations reject unsupported shapes, nonfinite quantization inputs, invalid scale metadata, and overflow rather than silently changing semantics.
- Elementwise device operations require identical logical shapes. Generic NumPy broadcasting is not part of v0.
- Channel parameters are scalar or one value per final physical channel. HWC8 operations normally use eight values per channel group.
- Random sampling is unsupported. `random_sample` always raises `UnsupportedOperationError`; v0 defines no RNG, seed, or probability-distribution contract.

## Rounding

HASLAB uses round-to-nearest, ties-to-even (RNE). For signed integer fixed-point conversion, `round_shift_rne(z, R)` computes the mathematical value `RNE(z / 2^R)` for `0 ≤ R ≤ 62`. Sign is handled symmetrically:

| Input | Result |
|---|---:|
| 5 / 2 | 2 |
| 7 / 2 | 4 |
| -5 / 2 | -2 |
| -7 / 2 | -4 |

No implementation may replace this with unconditional half-up rounding or truncation toward zero.

## INT8 quantization and scaling

Scales are first rounded to IEEE binary32 and must be finite and positive. A scale may be scalar or an array that NumPy can broadcast against the source tensor; a non-broadcastable shape is rejected. Compiler-generated profiles use a scalar for activations and an explicitly shaped per-output-channel array for weights. Symmetric quantization has zero point zero:

`q = clamp[-128,127](RNE(real / scale_binary32))`

The reference calculates the division in binary64 after promoting the exact binary32 scale, then applies RNE. Nonfinite real input is rejected. Dequantization computes `float32(q) × scale_binary32` and rounds the result to binary32.

Fixed-point requantization uses an integer multiplier `0 ≤ M ≤ 2^31-1` and shift `0 ≤ R ≤ 62`:

`qout = clamp[-128,127](RNE((int32_value × M) / 2^R))`

The intermediate is evaluated as an exact Python integer. `ADD_I8` forms both scaled terms in one exact numerator and rounds only once:

`qout = clamp[-128,127](RNE((qa×Ma + qb×Mb) / 2^R))`

This matters at half ties. Inputs must have equal shapes and both channel multipliers share the same shift for a channel.

Elementwise INT8 multiplication widens to INT32 without saturation. Elementwise INT32 addition rejects overflow and does not wrap.

## INT32 accumulation

`accumulate_i8` multiplies paired INT8 values and adds each exact product to a signed INT32 accumulator in increasing vector index. Every addition is checked. `matmul_i8` reduces K in increasing index; optional INT32 bias is added once after the complete reduction and is checked separately.

Logical convolution accepts NCHW input with batch one and OIHW weights. It supports dense 1×1 or 3×3 kernels, equal spatial stride one or two, and nonnegative zero padding. Reduction order is kernel row, kernel column, then input channel. Out-of-image activation values are exact INT8 zero. Bias is added after the complete reduction. Groups, dilation, depthwise convolution, and non-square kernels are rejected.

Partial reduction chunks in future command-level work must yield the same mathematical sum and must retain INT32 values between chunks; intermediate requantization is forbidden.

## FP8 conversion

FP8 is not a v0 device capability. The golden model includes conversions for future profile design and verification.

`E4M3FN` has one sign bit, four exponent bits, three fraction bits, and exponent bias 7. Exponent 0 encodes zero/subnormal values. Exponent 15 with fraction 7 is NaN; exponent 15 with fractions 0–6 remains finite. Maximum finite magnitude is 448. There is no infinity encoding.

`E5M2` has one sign bit, five exponent bits, two fraction bits, and exponent bias 15. Exponent 31 with fraction 0 is infinity and nonzero fractions are NaN. Maximum finite magnitude is 57,344.

Encoding rules are:

- Round finite magnitudes to the nearest representable value, breaking exact ties toward the encoding whose stored fraction least-significant bit is zero.
- Saturate finite overflow and positive/negative infinity to the format's signed maximum finite value.
- Preserve the sign of zero.
- Encode every NaN as canonical positive byte `0x7f`.
- Decode every format-defined NaN as a Python/IEEE NaN; a NaN payload or sign is not preserved.

FP8 quantization first divides by a positive binary32 scale. Dequantization decodes, multiplies by that binary32 scale, and rounds to binary32. Array scales follow the same explicit NumPy broadcasting rule as INT8 reference quantization.

`matmul_fp8` decodes bytes to binary32, multiplies each pair with a binary32 rounding, and adds K terms sequentially with a binary32 rounding after every addition. Optional binary32 bias is added last. This future-looking behavior is deterministic for a fixed Python/NumPy IEEE environment, but no v0 hardware opcode is implied.

## Activation functions

`sigmoid` is evaluated in binary64 with a branch-stable formula: `1/(1+exp(-x))` for nonnegative x and `exp(x)/(1+exp(x))` for negative x. `silu(x)` is binary64 `x × sigmoid(x)`.

The device-oriented SiLU path contains exactly 1,024 signed-byte entries indexed by integers -512 through 511. LUT construction first rounds `delta` and the output scale to binary32, evaluates binary64 SiLU at `index × delta`, then applies the INT8 quantization rule. Runtime lookup calculates each channel index as:

`j = clamp[-512,511](RNE(accumulator × M / 2^R))`

and returns LUT byte `j+512`. The table bytes, delta, scale, multiplier, and shift are part of the model profile; the hardware does not evaluate an exponential.

## Tensor layouts

Logical activations use NCHW with `N=1`. `nchw_to_hwc8` produces `[H,W,ceil(C/8),8]`. Logical channel `c` is at group `c//8`, lane `c%8`; unused lanes receive the explicit pad value, zero by default. The inverse requires the logical channel count and discards padded lanes.

Logical weights use `[Cout,Cin,Kh,Kw]`. `oihw_to_khwci8` produces `[Kh,Kw,CinPad,CoutGroups,8]`, where `CinPad=round_up(Cin,8)`. The final axis contains eight adjacent output-channel weights for one kernel position and input channel. Both input and output padding use the explicit pad value, zero by default.

The converters preserve input dtype and reject nonnumeric, nonpositive, or structurally invalid tensors. Layout conversion never changes logical numerical values.

## Pooling, upsampling, clamping, and argmax

`maxpool5_i8` accepts a pre-padded HWC8 tensor with two halo pixels on each spatial side and returns 5×5, stride-one maxima. The caller uses -128 for spatial halo. The result explicitly writes zero to physical channel lanes beyond the declared logical channel count.

`upsample2_nearest_i8` duplicates every HWC8 spatial element into a 2×2 block. Equivalently, output `(y,x)` reads input `(floor(y/2),floor(x/2))`. It also zeroes unused physical channel lanes.

`clamp` includes both bounds and rejects an inverted interval. `clamp_int8` saturates integers to -128…127.

`argmax` returns the lowest row-major index among equal maxima, matching NumPy's first-occurrence rule. With an axis, it returns the lowest index on that axis. Empty and NaN-bearing inputs are rejected so NaN ordering is never implicit.

## Verification scope

The test suite exhaustively round-trips every finite encoding of both FP8 formats and separately tests special values, subnormal and normal ties, saturation, scaling, integer rounding, accumulator overflow, layout padding, convolution, pooling, upsampling, activation LUT behavior, argmax ties, and unsupported sampling.

These tests establish the Python definition. The repository now also contains byte-level command conformance vectors, an audited pinned ONNX export, and measured calibration accuracy from a signed-symmetric INT8 software proxy. Before RTL can claim workload agreement, the compiler and simulated runtime must execute the pinned model with these exact HASLAB numerical rules and pass layerwise differential checks.
