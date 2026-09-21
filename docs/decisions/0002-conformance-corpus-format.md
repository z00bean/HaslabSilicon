# ADR 0002: independent binary conformance corpus

Copyright (C) 2026 Zubin Bhuyan.
SPDX-License-Identifier: CERN-OHL-S-2.0

Date: 2026-09-20

Status: accepted for candidate fixture development; stable ABI freeze remains pending independent review.

## Context

M3 produced a reviewed command-ABI candidate, but unit tests coupled to a model are insufficient as a portable contract for another implementation. M4 must preserve expected bytes and control states independently of the simulator and make corruption, stale versions, and unintended fixture regeneration visible.

## Decision

- Use fixture schema version 1: JSON for metadata/actions/checkpoints, raw binary for command and memory artifacts. Pin the exact contract and ABI-definition hashes.
- Store human-readable derivations alongside fixed expected spans, counters, and errors. Use a standard-library-only recipe serializer with no simulator or numerical-model imports.
- Check reproducibility without rewriting files during normal tests. Regeneration requires explicit `--write` and review of changed expectations.
- Validate the complete corpus before executing a device. Fail closed on unknown fields/actions, bad types, duplicate JSON keys, invalid paths/extents, and hash or version mismatches.
- Mask optional fault diagnostics explicitly and omit unspecified memory after arithmetic/bus faults. Do not require hardware to reproduce incidental Python fault messages or reset zeros.
- Publish `conformance/abi-v0.1.json` as the machine-readable allocation registry. Check existing Python constants against it; future compiler/RTL consumers must use generated or consistency-checked definitions.
- Include the corpus and runner-failure tests in `make test` and therefore CI. Keep the functional adapter distinct from future MMIO/bus/cycle adapters.

## Alternatives

Generating expected results from the simulator would preserve shared bugs. Using only Python test functions would make reuse by RTL or another software implementation harder. Embedding executable expressions in the manifest would complicate portability and allow an expectation to become another copy of the implementation.

## Consequences and evidence

The initial 46 fixtures are independent of implementation helpers, but remain authored and reviewed within this development process. They are not an external review or complete verification. Stable M3/M4 release still requires independent scrutiny of the candidate and its derivations. Runtime stale tokens, bus errors, and reset races with outstanding transfers await the corresponding implementations.

See the [package guide](../../conformance/README.md), [format](../../conformance/FORMAT.md), and [runner tests](../../conformance/tests/test_corpus.py).
