# Compiler tests

The tests cover deterministic `.hxb` framing, required-section validation, and exact complete-first-layer schedule counts, buffer extents, DMA traffic, and package reproducibility. The cross-component synthetic Conv-SiLU execution test is under `runtime/tests/`; the pinned third-party model is exercised by reproducible benchmark tools without checking its derived weights into Git.

Checked-in test inputs must remain small, license-cleared, and reproducible.
