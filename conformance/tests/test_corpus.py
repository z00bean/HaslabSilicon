"""Runner failure tests and cross-language ABI drift checks."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import ast
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from conformance import generate
from conformance.runner import (
    CAPACITY, DEFAULT_CORPUS, ROOT, ConformanceError, CorpusError, load_json, load_suite, run_case,
)
from haslab_sim import Command, HaslabDevice
from haslab_sim import abi, device, errors, memory


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.blobs = load_suite()

    def case(self, name):
        return copy.deepcopy(next(c for c in self.manifest["cases"] if c["id"] == name))

    def mutate_manifest(self, mutate):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            shutil.copytree(DEFAULT_CORPUS, root)
            manifest = copy.deepcopy(self.manifest)
            mutate(manifest)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(CorpusError):
                load_suite(root)

    def test_independent_recipes_reproduce_checked_in_bytes(self):
        generated = generate.render()
        actual = {p.relative_to(DEFAULT_CORPUS).as_posix() for p in DEFAULT_CORPUS.rglob("*") if p.is_file()}
        self.assertEqual(actual, set(generated))
        for name, data in generated.items():
            with self.subTest(artifact=name):
                self.assertEqual((DEFAULT_CORPUS / name).read_bytes(), data)

    def test_recipes_have_no_implementation_or_numeric_library_imports(self):
        tree = ast.parse((ROOT / "conformance/generate.py").read_text())
        allowed = {"argparse", "hashlib", "json", "struct", "pathlib"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name, allowed)
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)

    def test_abi_definition_matches_executable_constants(self):
        spec = load_json(ROOT / "conformance/abi-v0.1.json")
        self.assertEqual(spec["abi"], {"major": abi.ABI_MAJOR, "minor": abi.ABI_MINOR})
        self.assertEqual(spec["command_bytes"], abi.COMMAND_BYTES)
        self.assertEqual(spec["command_words"], abi.COMMAND_WORDS)
        self.assertEqual(spec["payload_words"], abi.PAYLOAD_WORDS)
        self.assertEqual(spec["fifo_depth"], device.FIFO_DEPTH)
        self.assertEqual(spec["ext_capacity_max"], abi.UINT32_MAX)
        for key, enum in (("spaces", abi.MemorySpace), ("errors", errors.ErrorCode),
                          ("states", device.DeviceState), ("submit_results", device.SubmitResult)):
            self.assertEqual(spec[key], {item.name: int(item) for item in enum})
        self.assertEqual({name: value["opcode"] for name, value in spec["commands"].items()},
                         {item.name: int(item) for item in abi.Opcode})
        self.assertEqual(spec["local_bytes"], {space.name: size for space, size in memory.LOCAL_CAPACITIES.items()})
        self.assertEqual(CAPACITY, {spec["spaces"][name]: size for name, size in spec["local_bytes"].items()})
        self.assertEqual(spec["commands"]["CONV_I8"]["flags"],
                         {"FIRST": device.FIRST_FLAG, "LAST": device.LAST_FLAG})
        sample = Command(0x1234, 0x89ABCDEF, flags=0x12345678, abi_major=5, abi_minor=6)
        words = sample.to_words()
        values = dict(abi_major=5, abi_minor=6, opcode=0x1234, flags=0x12345678,
                      sequence=0x89ABCDEF, reserved_zero=0)
        for name, (word, shift, width) in spec["header"].items():
            self.assertEqual((words[word] >> shift) & ((1 << width) - 1), values[name])

    def test_registry_payload_lengths_match_unused_word_rejection(self):
        spec = load_json(ROOT / "conformance/abi-v0.1.json")
        for name, entry in spec["commands"].items():
            with self.subTest(command=name):
                d = HaslabDevice(ext_bytes=256)
                d.submit(Command.build(entry["opcode"], 1, [0]*len(entry["payload"])+[1]))
                self.assertFalse(d.run_next())
                self.assertEqual(d.fault.code, errors.ErrorCode.BAD_FIELD)
                self.assertEqual(d.fault.field, 4+len(entry["payload"]))

    def test_corrupted_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            shutil.copytree(DEFAULT_CORPUS, root)
            path = root / self.case("conv-raw")["commands"]
            raw = bytearray(path.read_bytes()); raw[0] ^= 1
            path.write_bytes(raw)
            with self.assertRaisesRegex(CorpusError, "hash mismatch"):
                load_suite(root)

    def test_stale_contract_or_abi_pin_is_rejected(self):
        for key in ("contract_sha256", "abi_sha256"):
            with self.subTest(key=key):
                self.mutate_manifest(lambda m: m.update({key: "0"*64}))

    def test_unsupported_schema_and_boolean_version_are_rejected(self):
        for value in (2, True):
            with self.subTest(value=value):
                self.mutate_manifest(lambda m: m.update(schema_version=value))

    def test_unknown_fields_and_actions_are_rejected(self):
        self.mutate_manifest(lambda m: m.update(typo=1))
        self.mutate_manifest(lambda m: m["cases"][0]["steps"][0].update(action="typo"))

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            for value in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
                with self.subTest(value=value):
                    path.write_text(value, encoding="utf-8")
                    with self.assertRaises(CorpusError):
                        load_json(path)

    def test_path_traversal_is_rejected(self):
        self.mutate_manifest(lambda m: m["artifacts"].update({"../escape.bin": dict(bytes=8, sha256="0"*64)}))

    def test_bad_command_and_memory_extents_are_rejected(self):
        self.mutate_manifest(lambda m: m["cases"][0]["steps"][0].update(offset=0xFFFFFFFF))
        self.mutate_manifest(lambda m: m["cases"][0]["initial"][0].update(offset=0xFFFFFFFF))

    def test_missing_status_check_is_rejected(self):
        self.mutate_manifest(lambda m: m["cases"][0]["steps"].pop())

    def test_duplicate_case_and_missing_artifact_are_rejected(self):
        self.mutate_manifest(lambda m: m["cases"].append(m["cases"][0]))
        self.mutate_manifest(lambda m: m["cases"][0]["initial"][0].update(artifact="missing/data.bin"))

    def test_wrong_memory_answer_fails_with_byte_location(self):
        case = self.case("map-ties")
        blobs = dict(self.blobs)
        path = case["steps"][-1]["memory"][0]["artifact"]
        blobs[path] = b"\x03" + blobs[path][1:]
        with self.assertRaisesRegex(ConformanceError, "space 4 byte 0"):
            run_case(case, blobs)

    def test_wrong_counter_or_error_answer_fails(self):
        for name, key in (("conv-raw", "last_completed"), ("bias-overflow", "error_code")):
            with self.subTest(key=key):
                case = self.case(name)
                case["steps"][-1]["status"][key] += 1
                with self.assertRaisesRegex(ConformanceError, key):
                    run_case(case, self.blobs)

    def test_optional_diagnostics_are_masked_but_code_is_not(self):
        case = self.case("dma-bounds")
        check = case["steps"][-1]
        check["status"]["error_space"] = 99
        run_case(case, self.blobs)
        check["diagnostic_masks"]["error_space"] = 0xFFFFFFFF
        with self.assertRaisesRegex(ConformanceError, "error_space"):
            run_case(case, self.blobs)

    def test_memory_bit_masks_only_ignore_declared_bits(self):
        case = self.case("map-ties")
        span = case["steps"][-1]["memory"][0]
        blobs = dict(self.blobs)
        span["mask"] = "map-ties/mask.bin"
        blobs[span["mask"]] = b"\xfe" + b"\xff"*7
        blobs[span["artifact"]] = b"\x03" + blobs[span["artifact"]][1:]
        run_case(case, blobs)
        blobs[span["artifact"]] = b"\x04" + blobs[span["artifact"]][1:]
        with self.assertRaises(ConformanceError):
            run_case(case, blobs)

    def test_allowed_structural_error_set_is_explicit(self):
        case = self.case("dma-bounds")
        case["steps"][-1]["allowed_error_codes"] = [3, 4]
        run_case(case, self.blobs)
        case["steps"][-1]["allowed_error_codes"] = [3]
        with self.assertRaisesRegex(ConformanceError, "error_code"):
            run_case(case, self.blobs)

    def test_out_of_range_actual_status_cannot_hide_behind_a_mask(self):
        class BrokenCounter(HaslabDevice):
            def run_next(self):
                result = super().run_next()
                self.last_completed += 1 << 32
                return result
        with self.assertRaisesRegex(ConformanceError, "fits u32"):
            run_case(self.case("map-ties"), self.blobs, BrokenCounter)


if __name__ == "__main__":
    unittest.main()
