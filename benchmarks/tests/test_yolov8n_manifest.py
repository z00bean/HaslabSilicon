import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = ROOT / "benchmarks" / "manifests" / "yolov8n-320-opset13"


class TestYolov8nManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((WORKLOAD / "manifest.json").read_text())
        cls.audit = json.loads((WORKLOAD / "graph-audit.json").read_text())
        cls.baseline = json.loads((WORKLOAD / "float-smoke-baseline.json").read_text())
        cls.coco_baseline = json.loads(
            (WORKLOAD / "float-coco-baseline.json").read_text()
        )
        cls.selection = json.loads((WORKLOAD / "calibration-selection.json").read_text())
        cls.int8_calibration = json.loads((WORKLOAD / "int8-calibration.json").read_text())
        cls.int8_diagnostics = json.loads((WORKLOAD / "int8-diagnostics.json").read_text())
        cls.int8_baseline = json.loads((WORKLOAD / "int8-coco-baseline.json").read_text())
        cls.m6_slice = json.loads((WORKLOAD / "m6-first-conv-silu-slice.json").read_text())
        cls.m6_first_layer = json.loads(
            (WORKLOAD / "m6-first-conv-silu-layer.json").read_text()
        )
        cls.m6_first_two_layers = json.loads(
            (WORKLOAD / "m6-first-two-conv-silu-layers.json").read_text()
        )
        cls.m6_first_c2f = json.loads((WORKLOAD / "m6-first-c2f.json").read_text())

    def test_artifact_identity_is_consistent(self):
        self.assertEqual(
            self.manifest["export"]["onnx_sha256"],
            self.audit["model"]["sha256"],
        )
        self.assertEqual(self.manifest["export"]["onnx_ir_version"], 7)
        self.assertEqual(self.manifest["export"]["opset"], 13)

    def test_partition_is_explicit_and_complete(self):
        self.assertEqual(self.manifest["partition"]["accelerator_nodes"], [0, 217])
        self.assertEqual(self.manifest["partition"]["host_tail_nodes"], [218, 260])
        self.assertEqual(self.audit["inventory"]["node_count"], 261)
        self.assertEqual(
            self.audit["inventory"]["classification_counts"]["host_tail"], 43
        )
        self.assertEqual(
            self.audit["result"]["unsupported_accelerator_nodes"], []
        )

    def test_v0_static_feasibility_claims_have_evidence(self):
        self.assertTrue(self.audit["result"]["passes_structural_v0_audit"])
        self.assertTrue(self.audit["local_memory"]["all_convolution_tiles_fit"])
        self.assertEqual(len(self.audit["local_memory"]["convolutions"]), 63)
        for space, used in self.audit["local_memory"]["worst_case_bytes"].items():
            self.assertLessEqual(used, self.audit["local_memory"]["capacities"][space])

    def test_smoke_and_accuracy_baselines_are_distinguished(self):
        self.assertIn("not a detection-accuracy measurement", self.baseline["purpose"])
        comparison = self.baseline["pytorch_reference"]["comparison"]
        self.assertTrue(comparison["allclose"])
        self.assertEqual(self.manifest["evaluation"]["status"], "complete")
        self.assertEqual(self.coco_baseline["status"], "complete")
        self.assertIn("proxy_passes", self.manifest["quantization"]["status"])

    def test_coco_baseline_matches_manifest(self):
        evaluation = self.manifest["evaluation"]
        result = self.coco_baseline
        metrics = result["evaluation"]["metrics"]

        self.assertEqual(
            result["model"]["sha256"], self.manifest["export"]["onnx_sha256"]
        )
        self.assertEqual(result["dataset"]["image_count"], 5000)
        self.assertEqual(result["dataset"]["kept_non_crowd_boxes"], 36335)
        self.assertEqual(len(result["evaluation"]["per_class"]), 80)
        self.assertAlmostEqual(
            metrics["metrics/mAP50-95(B)"], evaluation["metrics"]["mAP50_95"]
        )
        self.assertAlmostEqual(
            metrics["metrics/mAP50(B)"], evaluation["metrics"]["mAP50"]
        )
        self.assertEqual(
            result["evaluation"]["predictions_sha256"],
            evaluation["predictions_sha256"],
        )
        self.assertEqual(
            result["dataset"]["image_archive"]["sha256"],
            evaluation["image_archive"]["sha256"],
        )
        self.assertEqual(
            result["dataset"]["annotation_archive"]["sha256"],
            evaluation["annotation_archive"]["sha256"],
        )

    def test_int8_calibration_package_is_internally_consistent(self):
        quantization = self.manifest["quantization"]
        calibration = self.int8_calibration
        selection = self.selection["selection"]

        self.assertEqual(selection["image_count"], 512)
        filename_hashes = [item["filename_sha256"] for item in selection["images"]]
        self.assertEqual(filename_hashes, sorted(filename_hashes))
        self.assertEqual(len({item["filename"] for item in selection["images"]}), 512)
        self.assertEqual(
            selection["ordered_filename_sha256"],
            quantization["selection"]["ordered_filename_sha256"],
        )
        self.assertEqual(calibration["calibration"]["images"], 512)
        self.assertEqual(calibration["scale_counts"]["activation_per_tensor"], 205)
        self.assertEqual(calibration["scale_counts"]["weight_per_output_channel"], 63)
        self.assertTrue(calibration["all_zero_points_are_zero"])
        self.assertEqual(calibration["silu"]["tables"], 57)
        lut_bytes = (WORKLOAD / "int8-silu-luts.bin").read_bytes()
        self.assertEqual(
            hashlib.sha256(lut_bytes).hexdigest(),
            calibration["silu"]["binary_sha256"],
        )

    def test_int8_proxy_passes_budget_without_claiming_target_equivalence(self):
        quantization = self.manifest["quantization"]
        proxy = quantization["proxy_evaluation"]
        metrics = self.int8_baseline["evaluation"]["metrics"]

        self.assertEqual(
            self.int8_baseline["model"]["sha256"],
            quantization["calibration"]["quantized_proxy_model_sha256"],
        )
        self.assertAlmostEqual(proxy["mAP50_95"], metrics["metrics/mAP50-95(B)"])
        expected_loss = (
            self.coco_baseline["evaluation"]["metrics"]["metrics/mAP50-95(B)"]
            - metrics["metrics/mAP50-95(B)"]
        )
        self.assertAlmostEqual(proxy["absolute_mAP50_95_loss"], expected_loss)
        self.assertLessEqual(
            proxy["absolute_mAP50_95_loss"],
            quantization["acceptance_budget"]["maximum_absolute_loss"],
        )
        self.assertTrue(proxy["passes_budget"])
        self.assertEqual(len(self.int8_baseline["evaluation"]["per_class"]), 80)
        self.assertIn("not final HASLAB acceptance", quantization["acceptance_scope"])
        self.assertEqual(self.int8_diagnostics["images"], 512)
        self.assertEqual(self.int8_diagnostics["tensor_count"], 215)
        self.assertEqual(
            self.int8_diagnostics["weights"]["aggregate"]["clipped_fraction"], 0.0
        )

    def test_m6_slice_records_exact_command_execution_without_overclaiming(self):
        result = self.m6_slice
        self.assertEqual(result["source_model"]["sha256"], self.manifest["export"]["onnx_sha256"])
        self.assertEqual(result["package"]["schema"], "haslab.vertical-slice.v1")
        self.assertEqual(result["package"]["commands"], 8)
        self.assertTrue(result["package"]["repeated_compilation_byte_identical"])
        self.assertTrue(result["exact_integer_comparison"]["pass"])
        self.assertEqual(result["exact_integer_comparison"]["mismatch_count"], 0)
        self.assertEqual(result["exact_integer_comparison"]["compared_values"], 512)
        self.assertIn("not whole-layer", result["scope"]["limitation"])

    def test_m6_complete_first_layer_has_exact_schedule_and_output(self):
        result = self.m6_first_layer
        self.assertEqual(result["source_model"]["sha256"], self.manifest["export"]["onnx_sha256"])
        self.assertEqual(result["package"]["schema"], "haslab.first-layer.v1")
        self.assertTrue(result["package"]["repeated_compilation_byte_identical"])
        self.assertEqual(result["schedule"]["spatial_tiles"], 400)
        self.assertEqual(result["schedule"]["tile_executions"], 800)
        self.assertEqual(sum(result["schedule"]["opcode_counts"].values()), 8884)
        self.assertEqual(result["schedule"]["dma_bytes"]["total"], 2250768)
        self.assertEqual(result["exact_integer_comparison"]["compared_values"], 409600)
        self.assertEqual(result["exact_integer_comparison"]["mismatch_count"], 0)
        self.assertTrue(result["exact_integer_comparison"]["pass"])
        fifo = result["runtime"]["fifo"]
        self.assertEqual(fifo["accepted_commands"], 8884)
        self.assertEqual(fifo["busy_responses"], 8876)
        self.assertEqual(fifo["fifo_high_watermark"], 8)
        self.assertEqual(fifo["final_drain_commands"], 8)
        self.assertIn("not the remaining accelerator graph", result["scope"]["limitation"])

    def test_m6_first_two_layers_retain_and_compare_both_boundaries(self):
        result = self.m6_first_two_layers
        record = self.manifest["m6_first_two_layers"]
        self.assertEqual(result["source_model"]["sha256"], self.manifest["export"]["onnx_sha256"])
        self.assertEqual(result["package"]["schema"], "haslab.conv-silu-pipeline.v1")
        self.assertEqual(result["package"]["sha256"], record["package_sha256"])
        self.assertTrue(result["package"]["repeated_compilation_byte_identical"])
        self.assertEqual(result["allocation"]["external_bytes"], 1442176)
        self.assertEqual(
            [item["bytes"] for item in result["allocation"]["retained_tensors"]],
            [409600, 204800],
        )
        self.assertEqual(result["schedule"]["opcode_counts"]["CONV_I8"], 1600)
        self.assertEqual(sum(result["schedule"]["opcode_counts"].values()), 16245)
        self.assertEqual(result["schedule"]["dma_bytes"]["total"], 1999320)
        self.assertEqual(result["schedule"]["layers"][1]["input_channel_chunks"], 2)
        self.assertEqual(result["exact_integer_totals"]["compared_values"], 614400)
        self.assertEqual(result["exact_integer_totals"]["mismatch_count"], 0)
        self.assertTrue(result["exact_integer_totals"]["pass"])
        self.assertEqual(len(result["exact_integer_comparisons"]), 2)
        fifo = result["runtime"]["fifo"]
        self.assertEqual(fifo["accepted_commands"], 16245)
        self.assertEqual(fifo["busy_responses"], 16237)
        self.assertEqual(fifo["final_drain_commands"], 8)
        self.assertIn("not the remaining accelerator graph", result["scope"]["limitation"])

    def test_m6_first_c2f_records_exact_branch_semantics(self):
        result = self.m6_first_c2f
        record = self.manifest["m6_first_c2f"]
        self.assertEqual(result["source_model"]["sha256"], self.manifest["export"]["onnx_sha256"])
        self.assertEqual(result["package"]["schema"], "haslab.first-c2f.v1")
        self.assertEqual(result["package"]["sha256"], record["package_sha256"])
        self.assertTrue(result["package"]["repeated_compilation_byte_identical"])
        self.assertEqual(result["allocation"]["external_bytes"], 2480256)
        self.assertEqual(len(result["allocation"]["retained_tensors"]), 8)
        self.assertEqual(len(result["allocation"]["zero_allocation_views"]), 2)
        self.assertTrue(
            all(view["byte_allocation"] == 0 for view in result["allocation"]["zero_allocation_views"])
        )
        self.assertEqual(result["schedule"]["opcode_counts"]["ADD_I8"], 200)
        self.assertEqual(result["schedule"]["opcode_counts"]["MAP_I8"], 600)
        self.assertEqual(sum(result["schedule"]["opcode_counts"].values()), 59052)
        self.assertEqual(result["schedule"]["dma_bytes"]["total"], 4368728)
        self.assertEqual(result["schedule"]["macs"], 86425600)
        self.assertEqual(result["exact_integer_totals"]["compared_values"], 1843200)
        self.assertEqual(result["exact_integer_totals"]["mismatch_count"], 0)
        self.assertTrue(result["exact_integer_totals"]["pass"])
        self.assertEqual(len(result["exact_integer_comparisons"]), 10)
        fifo = result["runtime"]["fifo"]
        self.assertEqual(fifo["accepted_commands"], 59052)
        self.assertEqual(fifo["busy_responses"], 59044)
        self.assertEqual(fifo["final_drain_commands"], 8)
        self.assertIn("not the remaining accelerator graph", result["scope"]["limitation"])


if __name__ == "__main__":
    unittest.main()
