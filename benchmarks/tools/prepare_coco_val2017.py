#!/usr/bin/env python3
"""Prepare official COCO val2017 annotations for pinned HASLAB evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


COCO80_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_fingerprint(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "little"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-archive", type=Path, required=True)
    parser.add_argument("--annotation-archive", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    images_dir = root / "images" / "val2017"
    annotation = root / "annotations" / "instances_val2017.json"
    labels_dir = root / "labels" / "val2017"
    if not images_dir.is_dir() or not annotation.is_file():
        raise SystemExit("expected extracted images/val2017 and annotations/instances_val2017.json")

    data = json.loads(annotation.read_text(encoding="utf-8"))
    images = sorted(data["images"], key=lambda item: item["file_name"])
    if len(images) != 5_000:
        raise SystemExit(f"expected 5000 validation images, found {len(images)}")
    category_ids = tuple(sorted(item["id"] for item in data["categories"]))
    if category_ids != COCO80_CATEGORY_IDS:
        raise SystemExit("COCO category IDs do not match the pinned 80-class mapping")
    category_to_class = {category: index for index, category in enumerate(category_ids)}
    category_names = [
        item["name"] for item in sorted(data["categories"], key=lambda item: item["id"])
    ]

    annotations: dict[int, list[dict]] = defaultdict(list)
    for item in data["annotations"]:
        annotations[item["image_id"]].append(item)

    labels_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    label_paths: list[Path] = []
    kept_boxes = 0
    ignored_crowd = 0
    duplicate_boxes = 0
    for image in images:
        image_path = images_dir / image["file_name"]
        if not image_path.is_file():
            raise SystemExit(f"missing validation image: {image_path}")
        image_paths.append(image_path)
        width, height = image["width"], image["height"]
        lines: list[tuple[float, ...]] = []
        seen: set[tuple[float, ...]] = set()
        for item in annotations[image["id"]]:
            if item.get("iscrowd", False):
                ignored_crowd += 1
                continue
            x, y, box_width, box_height = (float(value) for value in item["bbox"])
            if box_width <= 0 or box_height <= 0:
                continue
            record = (
                float(category_to_class[item["category_id"]]),
                (x + box_width / 2.0) / width,
                (y + box_height / 2.0) / height,
                box_width / width,
                box_height / height,
            )
            if record in seen:
                duplicate_boxes += 1
                continue
            seen.add(record)
            lines.append(record)
        label_path = labels_dir / Path(image["file_name"]).with_suffix(".txt")
        label_path.write_text(
            "".join(("%g " * len(line)).rstrip() % line + "\n" for line in lines),
            encoding="utf-8",
        )
        label_paths.append(label_path)
        kept_boxes += len(lines)

    val_list = root / "val2017.txt"
    val_list.write_text(
        "".join(f"./images/val2017/{path.name}\n" for path in image_paths),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "source": "COCO 2017 validation images and instances_val2017 annotations",
        "image_archive": {
            "filename": args.image_archive.name,
            "bytes": args.image_archive.stat().st_size,
            "sha256": sha256_file(args.image_archive),
        },
        "annotation_archive": {
            "filename": args.annotation_archive.name,
            "bytes": args.annotation_archive.stat().st_size,
            "sha256": sha256_file(args.annotation_archive),
        },
        "instances_annotation": {
            "filename": annotation.name,
            "bytes": annotation.stat().st_size,
            "sha256": sha256_file(annotation),
        },
        "image_count": len(image_paths),
        "image_tree_sha256": tree_fingerprint(image_paths, root),
        "label_file_count": len(label_paths),
        "label_tree_sha256": tree_fingerprint(label_paths, root),
        "val_list_sha256": sha256_file(val_list),
        "kept_non_crowd_boxes": kept_boxes,
        "ignored_crowd_annotations": ignored_crowd,
        "ignored_duplicate_boxes": duplicate_boxes,
        "category_ids": list(category_ids),
        "category_names": category_names,
        "conversion": "COCO xywh top-left pixels to YOLO class,cx,cy,w,h normalized; iscrowd ignored; duplicate records removed",
    }
    (root / "preparation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
