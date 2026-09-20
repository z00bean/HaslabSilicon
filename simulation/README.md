# Simulation

`haslab_sim` is the functional, byte-level simulator for the proposed v0 command ABI. It models:

- 128-byte little-endian command encoding and FIFO submission rules
- EXT and bounded local memory spaces
- serialized DMA, fill, copy, convolution, epilogue, utility, fence, and end commands
- accumulator FIRST/LAST lifecycle
- ordered completion, reset generation, and sticky architectural faults

It calls the Python golden model for numerical operations. It intentionally does not model cycles, burst timing, host cache behavior, interrupts, or RTL signal timing.

Run its tests with `make sim`, or run every Python test with `make test`.
