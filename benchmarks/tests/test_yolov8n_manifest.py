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
        self.assertEqual(self.manifest["quantization"]["status"], "pending")

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


if __name__ == "__main__":
    unittest.main()
