# Compiler tests

The tests cover deterministic `.hxb` framing, required-section validation, exact first-layer and two-layer schedule counts, retained buffer extents, input-channel chunks, DMA traffic, connectivity rejection, and package reproducibility. Cross-component synthetic execution tests are under `runtime/tests/`; the pinned third-party model is exercised by reproducible benchmark tools without checking its derived weights into Git.

Checked-in test inputs must remain small, license-cleared, and reproducible.
