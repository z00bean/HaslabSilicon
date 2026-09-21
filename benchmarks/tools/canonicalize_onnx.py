#!/usr/bin/env python3
"""Remove known volatile Ultralytics metadata from a pinned ONNX export.

Ultralytics 8.3.40 embeds the wall-clock export time and derives the model
description from the input filename. Those fields make otherwise identical
graphs hash differently. This tool fixes the description, removes the date,
sorts metadata keys, checks the model, and uses deterministic protobuf output.
It does not change graph nodes, tensors, initializers, or opsets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


DESCRIPTION = "Ultralytics YOLOv8n model trained on coco.yaml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.input)
    properties = {
        item.key: item.value
        for item in model.metadata_props
        if item.key not in {"date", "description", "haslab_canonicalization"}
    }
    properties["description"] = DESCRIPTION
    properties["haslab_canonicalization"] = "ultralytics-onnx-metadata-v1"
    del model.metadata_props[:]
    for key, value in sorted(properties.items()):
        item = model.metadata_props.add()
        item.key = key
        item.value = value

    onnx.checker.check_model(model, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model.SerializeToString(deterministic=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
