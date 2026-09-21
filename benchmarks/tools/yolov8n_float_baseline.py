#!/usr/bin/env python3
"""Generate the deterministic FLOAT smoke baseline for the pinned YOLOv8n export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    values = np.arange(3 * 320 * 320, dtype=np.uint32) % 256
    input_tensor = values.astype(np.float32).reshape(1, 3, 320, 320) / np.float32(255.0)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"images": input_tensor})[0]

    report: dict[str, object] = {
        "schema_version": 1,
        "purpose": "deterministic exporter/runtime smoke check; not a detection-accuracy measurement",
        "input": {
            "algorithm": "reshape(arange(3*320*320) mod 256) as [1,3,320,320] FLOAT32, then divide by FLOAT32(255)",
            "shape": list(input_tensor.shape),
            "dtype": str(input_tensor.dtype),
            "sha256": sha256_bytes(input_tensor),
        },
        "onnxruntime": {
            "version": ort.__version__,
            "providers": session.get_providers(),
            "output_shape": list(onnx_output.shape),
            "output_dtype": str(onnx_output.dtype),
            "output_sha256": sha256_bytes(onnx_output),
            "minimum": float(onnx_output.min()),
            "maximum": float(onnx_output.max()),
            "mean": float(onnx_output.mean()),
        },
    }

    if args.weights:
        import torch
        import ultralytics
        from ultralytics import YOLO

        torch_model = YOLO(str(args.weights)).model.eval()
        with torch.no_grad():
            torch_output = torch_model(torch.from_numpy(input_tensor))[0].numpy()
        difference = np.abs(onnx_output - torch_output)
        report["pytorch_reference"] = {
            "torch_version": torch.__version__,
            "ultralytics_version": ultralytics.__version__,
            "output_shape": list(torch_output.shape),
            "output_sha256": sha256_bytes(torch_output),
            "minimum": float(torch_output.min()),
            "maximum": float(torch_output.max()),
            "mean": float(torch_output.mean()),
            "comparison": {
                "max_absolute_error": float(difference.max()),
                "mean_absolute_error": float(difference.mean()),
                "rtol": 1e-4,
                "atol": 1e-5,
                "allclose": bool(np.allclose(onnx_output, torch_output, rtol=1e-4, atol=1e-5)),
            },
        }

    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
