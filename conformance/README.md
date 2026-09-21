# HASLAB v0 conformance package

Copyright (C) 2026 Zubin Bhuyan.
SPDX-License-Identifier: CERN-OHL-S-2.0

This package contains **46 independently authored command fixtures** for the experimental `haslab-v0-int8` ABI 0.1, contract document revision 0.2. A fixture contains actual command bytes, initial memory images, stored expected results, and a sequence of host actions and architectural checkpoints. It can be replayed against another implementation without importing the Python numerical model.

The corpus is a candidate, not a frozen release ABI or an independent external review. The expected values were derived separately from the implementation using explicit arithmetic and contract rules. Passing the corpus is evidence for its covered cases, not a proof of every possible command sequence or a claim about RTL timing.

## Run it

From the repository root, with the existing Python/NumPy dependencies installed:

```sh
make conformance
```

This checks byte-for-byte reproducibility, ABI consistency, schema/runner tests, and all stored fixtures. `make test` includes the same checks, so the GitHub Actions test job also runs them.

Validation on 2026-09-20: all 46 fixture cases and 104 unit tests passed (43 numerical-model, 42 simulator, 19 conformance-infrastructure tests). Repository and whitespace checks also passed.

Run one case while debugging:

```sh
PYTHONPATH=reference:simulation:. python3 -m conformance.runner --case conv-raw
```

## Files

| File | Role |
|---|---|
| [abi-v0.1.json](abi-v0.1.json) | Machine-readable command allocations, payload order, flags, spaces, capacities, header bits, types, state/errors, and candidate MMIO addresses |
| [FORMAT.md](FORMAT.md) | Versioned fixture schema, execution rules, comparisons, and portability contract |
| [corpus/manifest.json](corpus/manifest.json) | Corpus inventory, contract/ABI hashes, artifact hashes, case derivations, and expected checkpoints |
| `corpus/<case>/commands.bin` | Exact little-endian bytes submitted to the device; malformed framing cases use explicit byte slices |
| `corpus/<case>/initial-*.bin` | Testbench memory initialization, independent of reset contents |
| `corpus/<case>/expected-*.bin` | Stored expected byte spans, never calculated by the device under test |
| [generate.py](generate.py) | Standard-library-only recipes and serializer for reviewing/reproducing the corpus |
| [runner.py](runner.py) | Strict pre-execution validation and functional-simulator adapter |
| [tests/test_corpus.py](tests/test_corpus.py) | ABI drift, corruption rejection, schema validation, and deliberately wrong-answer tests |

`abi-v0.1.json` is the machine-readable allocation registry for future consumers. CI checks the existing simulator enums, header packing, payload lengths, capacities, and flags against it. The simulator still uses Python constants; it does not load JSON at runtime. Candidate MMIO/type metadata with no executable consumer remains specification-only. New compiler/RTL constants should be generated or checked against this registry rather than maintained as unchecked copies. The normative behavioral rules remain in the [contract](../docs/haslab-v0-contract.md).

## Coverage and derivation

| Category | Evidence |
|---|---|
| Every v0 opcode | Valid fill/DMA/copy, convolution/epilogue, map/add/pool/upsample, FENCE, and END |
| Tensor arithmetic | Hand-computed 1×1 and 3×3 convolutions, stride two, zero halo, multi-chunk accumulation, bias, zero padding |
| Integer boundaries | Positive/negative ties-to-even, saturation, M=0, positive bias overflow, real accumulated overflow from legal chunks |
| LUT indexing | Explicit synthetic table values at clamped bounds and tie-selected indices; this validates lookup, not SiLU approximation accuracy |
| Memory | Two-dimensional DMA both directions, guard bytes, legal interleaved copy rows, illegal cross-row overlap, alignment, bounds, widened address arithmetic |
| Control | FIFO full/retry, malformed framing, exact ABI mismatch, sequence gap/zero, reserved flags/header/payload, sticky first fault, reset dropping queued work |
| Context/parameters | Missing or incorrect convolution continuation, END/utility with open ACC, epilogue without context, mismatched ADD shifts, invalid padding parameters, LUT overlap |

Each case's `derivation` explains its answer. For example, `conv-raw` stores 25 and 5 from `1×2 + 2×3 + 3×4 + 5` and `1 + 2 + 3 − 1`. `accumulator-overflow` reaches the signed INT32 limit through 456 legal convolution chunks: 455 chunks retire; chunk 456 faults. Expected failed-accumulator contents are deliberately unspecified.

The generator imports no simulator, reference-model, NumPy, or framework arithmetic. ABI consistency tests and runner tests have a separate purpose: checking the machinery which consumes the corpus. Neither is used to manufacture golden outputs.

## Updating fixtures

1. Explain the contract rule and derive the expected answer independently.
2. Edit the recipe in `generate.py` and, if applicable, the ABI registry through a reviewed decision.
3. Explicitly regenerate the assets:

   ```sh
   python3 -m conformance.generate --write
   ```

4. Review the derivation, manifest changes, and byte images. A simulator disagreement must be investigated; do not replace expected bytes with simulator output.
5. Run `make conformance` and `make test`, and record changed coverage in the development plan.

Normal tests never rewrite fixtures. `python3 -m conformance.generate` without `--write` compares the complete file set and every byte, including manifest hashes. Contract or registry edits invalidate the pins until explicitly reviewed and regenerated. Hashes detect drift/corruption; they do not establish correctness or authenticate a reviewer.

## Remaining gates

- Independent review of the candidate tables, transition rules, and fixture derivations before stable M3/M4 release.
- The M5 pinned YOLO graph, calibration, preprocessing/host-tail boundary, and model accuracy evidence.
- Transport/MMIO staging, reset races with outstanding bus work, DMA_BUS injection, cache maintenance, and cycle counters. The functional simulator does not model them.
- Runtime session/stale-token handling, which requires a runtime implementation. Reset-generation increments are covered, but this is not stale-token verification.
- More exhaustive parameter/shape combinations and seed-recorded randomized differential tests when RTL is available. Existing simulator unit tests supplement the corpus but are not implicitly portable binary fixtures.

No RTL or end-to-end detector is implemented by this package.
