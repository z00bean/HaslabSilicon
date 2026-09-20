# Python reference model

`haslab_ref` is the executable numerical specification for HASLAB v0. It is deliberately scalar and explicit where operation order matters. It is a correctness oracle for future compiler, simulator, and RTL work, not a performance model.

Implemented semantics include:

- ONNX-compatible FP8 E4M3FN and E5M2 encode/decode, with HASLAB's documented saturation and canonical-NaN policy
- symmetric INT8 quantization, dequantization, exact fixed-point requantization, and ties-to-even rounding
- overflow-checked INT32 accumulation, INT8 matrix multiplication, and logical convolution
- scaled residual addition, mapping/scaling, clamping, and widening elementwise multiplication
- SiLU/sigmoid reference functions and the v0 1,024-entry SiLU LUT path
- HWC8 activation and KHWCI8 weight layout conversion
- v0 5×5 max pooling, 2× nearest upsampling, and deterministic argmax

Random sampling remains unsupported in v0. Calling the explicit sampling stub raises `UnsupportedOperationError`; no random-number algorithm or seed contract is implied.

Run the tests from the repository root:

```sh
make test
```

The exact behavior and edge cases are documented in [numerical-semantics.md](numerical-semantics.md). The v0 device still rejects FP8 commands; FP8 utilities exist to freeze conversion behavior for future profiles.
