# HASLAB runtime

The runtime area now contains a minimal Python lifecycle for the M6 single-layer, sequential, C2f, and reusable graph packages. It strictly parses the proposed `.hxb` framing, verifies section hashes and zero padding, rejects duplicate JSON keys and incompatible manifests, allocates declared external buffers, applies only validated DMA relocations, binds an exact-size input, submits commands to `haslab_sim`, and returns a generation-scoped completion token with structured device faults. Refill-time faults include the architectural sequence, code, and message. The nodes 0–119 first-neck package executes 235,474 commands; diagnostic mode exposes all comparison boundaries while release mode allocates 614,400 output bytes and exposes only the final tensor.

`haslab_runtime.SimulatorRuntime` provides `load_model`, `bind_input`, `submit`, `wait`, and `reset`. It records accepted commands, FIFO `BUSY` responses, commands retired while refilling, high-water mark, and final drain. Unit tests cover bit-exact compiler-to-simulator execution, corrupt-package rejection, binding size, and stale-token invalidation.

This is a synchronous Python research backend for four experimental package profiles. It is not the planned C ABI, an FPGA driver, a multi-model scheduler, a host-tail evaluator, or a complete YOLO runtime.
