#!/usr/bin/env python3
"""Run the pinned YOLOv8n FLOAT ONNX baseline on official COCO val2017."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
import ultralytics
from ultralytics import YOLO


EXPECTED_MODEL_SHA256 = "5b232bf21720cac4264463896ace02b919d8bec3f3fd1e4b9d36695493fa1718"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_hash = sha256_file(args.model)
    if model_hash != EXPECTED_MODEL_SHA256:
        raise SystemExit(f"model hash mismatch: {model_hash}")
    root = args.dataset_root.resolve()
    preparation_path = root / "preparation.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    names = preparation["category_names"]

    dataset_yaml = root / "haslab-coco-val2017.yaml"
    yaml_lines = [
        f"path: {root}",
        "train: val2017.txt",
        "val: val2017.txt",
        "names:",
        *(f"  {index}: {name}" for index, name in enumerate(names)),
    ]
    dataset_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    run_root = root / "runs"
    run_name = "yolov8n-320-opset13-float"
    if "CPUExecutionProvider" not in onnxruntime.get_available_providers():
        raise SystemExit("ONNX Runtime CPUExecutionProvider is unavailable")
    original_provider_query = onnxruntime.get_available_providers
    onnxruntime.get_available_providers = lambda: ["CPUExecutionProvider"]
    try:
        metrics = YOLO(str(args.model), task="detect").val(
            data=str(dataset_yaml),
            split="val",
            imgsz=320,
            batch=1,
            device="cpu",
            half=False,
            rect=False,
            conf=0.001,
            iou=0.7,
            max_det=300,
            save_json=True,
            plots=False,
            verbose=False,
            workers=0,
            seed=0,
            deterministic=True,
            project=str(run_root),
            name=run_name,
            exist_ok=True,
        )
    finally:
        onnxruntime.get_available_providers = original_provider_query
    predictions = run_root / run_name / "predictions.json"
    per_class = []
    class_indices = list(getattr(metrics.box, "ap_class_index", []))
    ap_values = list(getattr(metrics.box, "ap", []))
    ap50_values = list(getattr(metrics.box, "ap50", []))
    for position, class_index in enumerate(class_indices):
        per_class.append(
            {
                "class_id": int(class_index),
                "name": names[int(class_index)],
                "ap50_95": float(ap_values[position]),
                "ap50": float(ap50_values[position]),
            }
        )

    report = {
        "schema_version": 1,
        "status": "complete",
        "model": {
            "filename": args.model.name,
            "sha256": model_hash,
            "backend": "ONNX Runtime CPUExecutionProvider via Ultralytics AutoBackend",
        },
        "dataset": preparation,
        "evaluation": {
            "images": 5_000,
            "image_size": 320,
            "batch": 1,
            "confidence_threshold": 0.001,
            "iou_threshold": 0.7,
            "nms_maximum_detections": 300,
            "coco_ap_maximum_detections": 100,
            "rectangular_batches": False,
            "half_precision": False,
            "seed": 0,
            "deterministic": True,
            "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
            "overall_ap_source": "pycocotools COCOeval bbox, with mAP50-95 and mAP50 copied into the Ultralytics result dictionary",
            "precision_recall_source": "Ultralytics detection validator matching",
            "per_class_source": "Ultralytics detection validator AP arrays",
            "per_class": per_class,
            "speed_ms_per_image": {
                key: float(value) for key, value in metrics.speed.items()
            },
            "predictions_sha256": sha256_file(predictions),
            "predictions_bytes": predictions.stat().st_size,
            "command_profile": "static ONNX, CPUExecutionProvider, batch 1, 320x320, conf 0.001, NMS IoU 0.7, max_det 300, square validation batches",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "numpy": np.__version__,
            "pycocotools": importlib.metadata.version("pycocotools"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["evaluation"]["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
