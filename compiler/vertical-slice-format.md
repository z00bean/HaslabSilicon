# Experimental M6 vertical-slice package

Copyright (C) 2026 Zubin Bhuyan.
SPDX-License-Identifier: GPL-3.0-or-later

This document describes `haslab.vertical-slice.v1`, the deliberately narrow manifest used to prove the first compiler/runtime path. It is implementation evidence for the proposed `.hxb` container, not a stable public package schema.

## Compiled work

The package represents one interior 8×8 output tile at output origin `(1,1)` and output channels 0–7 of the pinned YOLOv8n first block. The logical convolution is 3×3, stride two, with three input channels. The selected interior tile needs a 17×17 input patch and no explicit padding.

The compiler quantizes the first eight OIHW weight channels with the recorded per-output-channel binary32 scales, packs them as KHWCI8, converts FLOAT bias to INT32 with round-to-nearest ties-to-even using `input_scale × weight_scale[channel]`, and serializes eight 16-byte epilogue records plus the recorded 1,024-byte SiLU table.

## Sections

The writer follows the header, 64-byte section entries, alignment, and SHA-256 rules in the v0 contract. It emits sections in numeric type order:

1. canonical UTF-8 JSON manifest;
2. eight 128-byte command records;
3. 1,728 bytes of packed constants;
4. canonical UTF-8 JSON debug provenance.

The ignored local package is 5,056 bytes. It contains model-derived weights and is not committed. The tracked report records its hash and result.

## Command schedule

| Relative sequence | Command | Purpose |
|---:|---|---|
| 1 | `DMA_COPY2D` | Bound 17×17 HWC8 input patch to INPUT SRAM |
| 2 | `DMA_COPY2D` | Packed KHWCI8 weights to WEIGHT SRAM |
| 3 | `DMA_COPY2D` | Bias/multiplier/shift records to PARAM SRAM |
| 4 | `DMA_COPY2D` | SiLU table to PARAM SRAM offset 128 |
| 5 | `CONV_I8 FIRST|LAST` | Exact INT8 products and INT32 accumulation |
| 6 | `EPILOGUE mode=2` | Add bias and apply the fixed-point SiLU lookup |
| 7 | `DMA_COPY2D` | Copy the 512-byte HWC8 result to EXT |
| 8 | `END` | Mark the submission complete |

The manifest declares three external buffers: `input`, `constants`, and `output`. Relocations can change only the applicable EXT offset word of a DMA command. The runtime verifies the command target, zero placeholder, transfer extent, declared buffer bounds, and unique relocation before patching.

## Runtime validation

The current loader bounds the complete package at 64 MiB, accepts container version 0.1, rejects unknown or duplicate sections, validates all section hashes and zero alignment padding, rejects duplicate JSON keys and nonfinite JSON numbers, and requires the exact vertical-slice manifest and ABI 0.1.

The runtime API is synchronous. Submission tokens contain a session, reset generation, and final sequence. Reset invalidates earlier tokens. Simulator arithmetic faults are returned as structured failed completions.

## Required replacement

The [complete first-layer package](first-layer-format.md) now replaces the fixed tile assumptions with generated boundary halos, both output groups, assembled output storage, and measured schedule traffic. Before a stable M6 package release, define and test required manifest keys, unknown-field handling, per-section limits, buffer lifetime and alias rules, complete tensor descriptors, host-tail metadata, all constant types, relocation coverage, and malformed-package cases.
