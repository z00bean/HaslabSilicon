# Conformance fixture format, version 1

Copyright (C) 2026 Zubin Bhuyan.
SPDX-License-Identifier: CERN-OHL-S-2.0

The corpus consists of one UTF-8 JSON manifest and raw `.bin` artifacts below its directory. `runner.load_suite` is the strict executable validator for this schema. No third-party JSON-schema package is required. Unknown fields/versions/actions, duplicate keys, nonfinite JSON numbers, wrong types, invalid extents, missing artifacts, stale pins, and hash/length mismatches fail before device initialization.

All numeric schema fields are unsigned JSON integers, never booleans, strings, or fractions. Unless narrowed below they fit u32. Byte offsets are offsets into a memory region or an artifact, never host pointers. Command and INT32 memory bytes are little-endian. The manifest's `abi` and `contract_revision` are independent from `schema_version`.

## Manifest fields

| Field | Required value / meaning |
|---|---|
| `schema_version` | Integer 1 |
| `profile` | `haslab-v0-int8` |
| `abi` | `[0, 1]` |
| `contract_revision` | String `0.2` |
| `contract_sha256` | SHA-256 of exact repository `docs/haslab-v0-contract.md` bytes |
| `abi_sha256` | SHA-256 of exact `conformance/abi-v0.1.json` bytes |
| `byte_order` | `little` |
| `provenance` | Nonempty description of how expectations were obtained |
| `artifacts` | Map from relative artifact path to `{bytes, sha256}` |
| `cases` | Nonempty array of case objects |

Artifact paths must match `<case-id>/<lowercase-name>.bin`, with lowercase ASCII letters, digits, and hyphens in each name. Absolute paths, traversal, and symlinks resolving outside the corpus are rejected. Each artifact must be referenced by its own case. `bytes` is the exact file length and `sha256` is 64 lowercase hex digits over those raw bytes. Hashes exclude the manifest itself. There are no compression, network fetch, plugin, executable-code, or expression fields.

Validator resource limits are 4 MiB for the manifest, 4 MiB per artifact, 64 MiB total artifacts, and 16 MiB modeled EXT per case. These are test-harness limits, not new device limits. Wider device addresses can still be exercised as invalid command operands.

## Case fields

| Field | Meaning |
|---|---|
| `id` | Unique lowercase letter/digit/hyphen name |
| `derivation` | Nonempty independently written explanation of the result or violated rule |
| `tags` | Nonempty array of descriptive coverage strings |
| `tensors` | Array of `{layout, physical_shape, dtype}`; may be empty for control/byte-only cases |
| `ext_bytes` | Positive EXT capacity for this case, at most 16 MiB |
| `commands` | Artifact holding submission bytes |
| `initial` | Array of initial memory spans |
| `steps` | Nonempty ordered action array ending in `check` |

Tensor metadata uses `HWC8` or tile `KHWCI8`, `I8` or `I32`, and four positive physical dimensions with last dimension eight. HWC8 shape is `[H,W,Cgroups,8]`; KHWCI8 tile shape is `[Kh,Kw,Cchunk_pad,8]`. Metadata helps review; command operands are still authoritative. The validator does not perform graph inference or prove tensor metadata matches a workload.

An initial span is `{space, offset, artifact}`. Region IDs are EXT=0, INPUT=1, WEIGHT=2, ACC=3, OUTPUT=4, PARAM=5. Spans have positive length and fit the selected region. They are loaded in listed order before any command. Loading initial bytes is a testbench facility, not a device opcode. Omitted memory has unspecified contents; fixtures must initialize every operand read before observing it. Reset invalidates local memory, so tests must reinitialize it before later reads. Only explicitly expected spans are compared.

## Actions

| `action` | Other fields | Behavior |
|---|---|---|
| `submit` | `offset`, `length`, `result` | Submit exactly this byte slice of `commands`; compare submission result 0 accepted, 1 busy, 2 blocked, or 3 invalid |
| `execute` | `offset`, `count`, `retired` | Require an empty FIFO; submit and execute `count` consecutive 128-byte records one at a time; every attempted submission must be accepted; stop at the first failed execution and compare successful retirement count |
| `run` | `count`, `retired` | Execute up to `count` already queued records, stopping at the first failure/empty queue; compare successful retirement count |
| `reset` | None | Request reset and wait for its architectural completion before the next step |
| `check` | `status`, `diagnostic_masks`, `memory`; optional `allowed_error_codes` | Compare architectural state and expected byte spans |

`count` is 1…65,536; `retired` is 0…count. `submit.length` is 0…129 so malformed short/long records can be tested. All slices must fit their artifact. Because framing tests can store malformed records, `commands.bin` is not necessarily globally aligned or a simple array of records; follow the action offsets exactly. Each record submitted by `execute` is 128 bytes regardless of its artifact offset.

The functional adapter maps reset to synchronous `HaslabDevice.reset()`. A future RTL adapter must wait for the contract's generation/state acknowledgement and drain required bus work. For `execute`, wait for one record's completion/fault before submitting the next. For `run`, the adapter may need simulation stepping or a controlled queue-execution gate to reproduce a checkpoint after a particular retirement; this action format does not claim to be a production host runtime API.

## Checkpoints

Every `status` object has exactly these fields:

```text
state, last_accepted, last_completed, last_end, reset_generation,
error_code, error_seq, error_space, error_offset, error_field
```

State values are RESETTING=0, IDLE=1, RUNNING=2, FAULT=3. Error codes are 0…9 as defined by the ABI registry. All actual values must fit u32; comparisons never hide invalid high bits by truncating status.

`diagnostic_masks` always contains `error_space`, `error_offset`, and `error_field` u32 masks. Comparison is `(actual & mask) == (expected & mask)` for these three optional diagnostic fields only. Mask zero ignores a diagnostic; `4294967295` requires exact agreement. Main counters, state, error code, and failing sequence are compared exactly. If supplied, `allowed_error_codes` is a nonempty array of applicable codes including the primary expected code; actual error code must belong to that set. Prefer isolated violations with exact codes.

An expected memory span is `{space, offset, artifact}` with optional `mask` naming another artifact of identical byte length. Compare each byte with the same bit-mask rule, defaulting to `0xff` for every byte. Bytes outside expected spans are unasserted. Omit arithmetic/bus-fault destination spans when the contract declares partial contents unspecified; a mask must not hide a mandatory result. The candidate corpus uses exact memory spans and masks optional diagnostics to zero.

There is no free-text fault-message comparison, cycle comparison, implicit full-memory comparison, or expectation calculated from a command at test time. A valid corpus can be consumed by a C++, Rust, Python, or RTL testbench adapter using only this format and the published contract.
