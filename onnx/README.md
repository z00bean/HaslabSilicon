# ONNX support

This area will contain the supported ONNX profile, graph-validation rules, operator mapping, export recipes, and small fixtures. It does not contain model weights.

The proposed v0 profile is a static floating-point ONNX input graph that the future compiler calibrates into HASLAB's INT8 profile. Supported and rejected operator mappings are documented in `docs/haslab-v0-contract.md`.
