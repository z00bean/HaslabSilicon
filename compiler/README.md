# HASLAB compiler

The future compiler will validate a bounded ONNX profile, establish quantization metadata, lower supported graphs to the v0 command representation, allocate buffers, and emit target-specific `.hxb` packages.

The planned source interface is ONNX rather than manually rewritten networks. No importer, lowering pass, or package generator exists yet.
