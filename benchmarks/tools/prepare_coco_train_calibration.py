#!/usr/bin/env python3
"""Select and verify the pinned 512-image COCO train2017 calibration set."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


EXPECTED_ANNOTATION_ARCHIVE_SHA256 = (
    "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
)
EXPECTED_TRAIN_ANNOTATION_SHA256 = (
    "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
)
CALIBRATION_IMAGE_COUNT = 512
COCO_IMAGE_BASE_URL = "http://images.cocodataset.org/train2017"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_fingerprint(records: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        filename = str(record["filename"]).encode("utf-8")
        digest.update(len(filename).to_bytes(4, "little"))
        digest.update(filename)
        digest.update(int(record["bytes"]).to_bytes(8, "little"))
        digest.update(bytes.fromhex(str(record["sha256"])))
    return digest.hexdigest()


def ordered_filename_fingerprint(filenames: list[str]) -> str:
    digest = hashlib.sha256()
    for filename in filenames:
        encoded = filename.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def download_one(filename: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    url = f"{COCO_IMAGE_BASE_URL}/{filename}"
    temporary = destination.with_suffix(destination.suffix + ".partial")
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "HASLAB/0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                temporary.write_bytes(response.read())
            if temporary.stat().st_size == 0:
                raise RuntimeError("downloaded image is empty")
            temporary.replace(destination)
            return
        except Exception as error:  # network errors vary by platform
            last_error = error
            temporary.unlink(missing_ok=True)
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation-archive", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    annotation = root / "annotations" / "instances_train2017.json"
    archive_hash = sha256_file(args.annotation_archive)
    if archive_hash != EXPECTED_ANNOTATION_ARCHIVE_SHA256:
        raise SystemExit(f"annotation archive hash mismatch: {archive_hash}")
    annotation_hash = sha256_file(annotation)
    if annotation_hash != EXPECTED_TRAIN_ANNOTATION_SHA256:
        raise SystemExit(f"train annotation hash mismatch: {annotation_hash}")

    data = json.loads(annotation.read_text(encoding="utf-8"))
    images_by_name = {item["file_name"]: item for item in data["images"]}
    ordered = sorted(
        images_by_name,
        key=lambda filename: (hashlib.sha256(filename.encode("utf-8")).hexdigest(), filename),
    )
    filenames = ordered[:CALIBRATION_IMAGE_COUNT]
    images_dir = root / "images" / "train2017-calibration-512"
    images_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_one, filename, images_dir / filename): filename
                for filename in filenames
            }
            for future in as_completed(futures):
                future.result()

    records: list[dict[str, object]] = []
    for rank, filename in enumerate(filenames):
        path = images_dir / filename
        if not path.is_file():
            raise SystemExit(f"missing calibration image: {path}")
        metadata = images_by_name[filename]
        records.append(
            {
                "rank": rank,
                "image_id": metadata["id"],
                "filename": filename,
                "filename_sha256": hashlib.sha256(filename.encode("utf-8")).hexdigest(),
                "width": metadata["width"],
                "height": metadata["height"],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    report = {
        "schema_version": 1,
        "selection_rule": "sort every COCO train2017 filename by SHA256(UTF-8 filename), then filename as an explicit tie-breaker; take the first 512",
        "source": {
            "annotation_archive": {
                "filename": args.annotation_archive.name,
                "bytes": args.annotation_archive.stat().st_size,
                "sha256": archive_hash,
            },
            "instances_annotation": {
                "filename": annotation.name,
                "bytes": annotation.stat().st_size,
                "sha256": annotation_hash,
            },
            "image_base_url": COCO_IMAGE_BASE_URL,
            "train_image_count": len(images_by_name),
        },
        "selection": {
            "image_count": len(records),
            "ordered_filename_sha256": ordered_filename_fingerprint(filenames),
            "image_collection_sha256": collection_fingerprint(records),
            "total_image_bytes": sum(int(record["bytes"]) for record in records),
            "images": records,
        },
    }
    output = args.output or (root / "calibration-train2017-512.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report["selection"].items() if key != "images"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
