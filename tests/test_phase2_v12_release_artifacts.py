import csv
import hashlib
import json
import re
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / "results/optimization"
MANIFEST = OPT / "phase2_v12_release_artifacts_manifest.json"


def _strict_json(text):
    def reject_constant(value):
        raise ValueError(f"Non-standard JSON constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


class ReleaseArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MANIFEST.is_file():
            raise AssertionError("Run scripts/build_phase2_v12_release_artifacts.py before the release tests")
        cls.manifest = _strict_json(MANIFEST.read_text(encoding="utf-8"))
        cls.registry_path = ROOT / cls.manifest["outputs"]["experiment_registry"]["path"]
        cls.calibration_path = ROOT / cls.manifest["outputs"]["statistical_calibration"]["path"]
        cls.registry = [
            _strict_json(line)
            for line in cls.registry_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        with cls.calibration_path.open(encoding="utf-8", newline="") as stream:
            cls.calibration = list(csv.DictReader(stream))

    def test_registry_lines_are_strict_json_with_unique_ids(self):
        ids = [record["experiment_id"] for record in self.registry]
        self.assertEqual(len(ids), len(set(ids)))
        counts = Counter(record["record_type"] for record in self.registry)
        self.assertEqual(counts["predictive_candidate"], 106)
        self.assertEqual(counts["statistical_calibration_scenario"], 104)
        self.assertGreater(counts["sealed_temporal_evaluation"], 0)

    def test_incomplete_selection_is_never_reported_as_independent_validation(self):
        self.assertIn("INCOMPLETE", self.manifest["status"])
        self.assertIn("NOT_STARTED", self.manifest["independent_validation_status"])
        progress = [row for row in self.calibration if row["status"] == "RUNNING_INCOMPLETE"]
        self.assertTrue(progress)
        self.assertEqual(progress[0]["decision"], "INCONCLUSIVE")
        independent = [row for row in self.registry if row["record_type"] == "v1_2_independent_validation_phase"]
        self.assertEqual(len(independent), 1)
        self.assertIn("NOT_STARTED", independent[0]["run_status"])

    def test_output_hashes_match_manifest(self):
        for artifact in self.manifest["outputs"].values():
            path = ROOT / artifact["path"]
            self.assertTrue(path.is_file(), artifact["path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, artifact["sha256"], artifact["path"])

    def test_public_aggregate_artifacts_exclude_credentials_and_private_paths(self):
        paths = [
            self.registry_path,
            self.calibration_path,
            ROOT / "results/optimization/model_comparison.csv",
            ROOT / "results/optimization/metric_history.parquet",
            ROOT / "reports/optimization/metric_dashboard.html",
            ROOT / "results/optimization/native_grid_collocation_summary.json",
            ROOT / "results/optimization/native_grid_collocation_validation.json",
            ROOT / "results/optimization/round_03_augmentation_robustness/noise_robustness_by_scheme.csv",
            ROOT / "results/optimization/round_06_2025_test_score/sealed_2025_score.json",
            ROOT / "results/optimization/round_06_frozen_test/frozen_test_manifest.json",
            ROOT / "results/statistical_correction/real_series_v12/descriptive_trend_estimates.csv",
            ROOT / "results/statistical_correction/real_series_v12/descriptive_trend_manifest.json",
        ]
        patterns = re.compile(r"(?i)(earthdata\.token|urs\.earthdata|/home/diego|C:\\Users\\Pc)")
        for path in paths:
            payload = path.read_bytes()
            if path.suffix == ".parquet":
                # Parquet is a binary columnar aggregate; scan its raw bytes for private path markers.
                text = payload.decode("latin-1")
            else:
                text = payload.decode("utf-8")
            self.assertIsNone(patterns.search(text), path.name)

    def test_dashboard_is_local_and_self_contained(self):
        dashboard = (ROOT / "reports/optimization/metric_dashboard.html").read_text(encoding="utf-8")
        self.assertIn("KUNTUR v1.2 continuous optimization", dashboard)
        self.assertIn("v1.2 statistical-calibration gate", dashboard)
        self.assertNotRegex(dashboard, r"(?i)<script[^>]+src=")
        self.assertNotRegex(dashboard, r"(?i)(src|href)=['\"]https?://")

    def test_native_grid_collocation_is_recorded_as_descriptive_external_evidence(self):
        records = [row for row in self.registry if row["record_type"] == "external_point_to_grid_assessment"]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["run_status"], "VALIDATED_DESCRIPTIVE_POINT_TO_GRID_AGREEMENT")
        self.assertIn("assimilate stations", record["decision_reason"])
        self.assertEqual(record["metrics_by_source"]["NOAA_GHCN_M_QCU"]["matched_months"], 276)
        self.assertEqual(record["metrics_by_source"]["INAMHI_yearbook_temperature"]["matched_months"], 132)

    def test_real_series_trends_are_published_as_descriptive_with_inference_withheld(self):
        records = [row for row in self.registry if row["record_type"] == "real_series_descriptive_trend_summary"]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(len(record["metrics_by_variable"]), 8)
        self.assertEqual(record["run_status"], "DESCRIPTIVE_ESTIMATES_COMPLETE_INFERENCE_WITHHELD")
        self.assertTrue(all(not row["inference_status"].startswith("VALIDATED") for row in record["metrics_by_variable"].values()))


if __name__ == "__main__":
    unittest.main()
