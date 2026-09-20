# HASLAB runtime

The future runtime will load validated target-specific packages, bind input/output buffers, submit commands to an FPGA or ASIC backend, and report completion, profile data, and structured errors.

The proposed API is specified in `docs/haslab-v0-contract.md`. No C library, Python binding, or hardware backend is implemented.
