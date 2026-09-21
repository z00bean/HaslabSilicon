# HASLAB runtime

The runtime area now contains a minimal Python lifecycle for the M6 vertical slice. It strictly parses the proposed `.hxb` framing, verifies section hashes and zero padding, rejects duplicate JSON keys and incompatible manifests, allocates declared external buffers, applies only validated DMA relocations, binds an exact-size input, submits commands to `haslab_sim`, and returns a generation-scoped completion token with structured device faults.

`haslab_runtime.SimulatorRuntime` provides `load_model`, `bind_input`, `submit`, `wait`, and `reset`. Unit tests cover bit-exact compiler-to-simulator execution, corrupt-package rejection, binding size, and stale-token invalidation.

This is a synchronous Python research backend for one experimental package profile. It is not the planned C ABI, an FPGA driver, a multi-model scheduler, a host-tail evaluator, or a complete YOLO runtime.
