import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from kuntur_data.phase2.analytical import (
    analyze_variable,
    cached_analyze_variable,
    monthly_anomalies,
    perturb_analytical_series,
)
from kuntur_data.phase2.ingestion import GlobalAnalyticalCache


class AnalyticalEngineTests(unittest.TestCase):
    def fixture(self):
        months = pd.period_range("2000-01", "2009-12", freq="M")
        seasonal = np.tile(np.arange(12, dtype=float), 10)
        trend = np.arange(len(months), dtype=float) * 0.1
        return pd.DataFrame({
            "month": months.astype(str),
            "variable_id": "fixture_temperature",
            "value": 20.0 + seasonal + trend,
            "units": "degC",
            "valid_area_fraction_bbox": 0.9,
            "region_id": "fixture_bbox",
            "source_sha256": "a" * 64,
        })

    def test_monthly_anomalies_remove_calendar_climatology_and_respect_coverage(self):
        frame = self.fixture()
        frame.loc[0, "valid_area_fraction_bbox"] = 0.2
        result = monthly_anomalies(frame, "fixture_temperature", coverage_threshold=0.5)
        self.assertTrue(np.isnan(result.loc[0, "anomaly"]))
        self.assertAlmostEqual(result.loc[12, "anomaly"], -4.8, places=8)
        self.assertEqual(result.loc[12, "climatology_sample_count"], 9)

    def test_seasonally_adjusted_ols_recovers_known_decadal_slope(self):
        result = analyze_variable(self.fixture(), "fixture_temperature", coverage_threshold=0.5)
        self.assertAlmostEqual(result["trend"]["ols_hac_slope_per_decade"], 12.0, places=8)
        self.assertAlmostEqual(result["trend"]["theil_sen_slope_per_decade"], 12.0, places=8)
        self.assertEqual(result["trend"]["n_months"], 120)
        self.assertEqual(result["trend"]["coverage_threshold"], 0.5)

    def test_analytical_perturbations_are_seeded_and_do_not_mutate_gold(self):
        frame = self.fixture()
        original = frame["value"].to_numpy(copy=True)
        a = perturb_analytical_series(frame, "fixture_temperature", family="gaussian", intensity=0.25, seed=7)
        b = perturb_analytical_series(frame, "fixture_temperature", family="gaussian", intensity=0.25, seed=7)
        np.testing.assert_array_equal(a["value"], b["value"])
        np.testing.assert_array_equal(frame["value"], original)
        self.assertTrue(a["derivative_version"].str.startswith("phase2-analytical-robustness").all())

    def test_combined_perturbation_uses_a_normalized_total_budget(self):
        frame = self.fixture()
        result = perturb_analytical_series(frame, "fixture_temperature", family="gaussian+drift", intensity=0.25, seed=11)
        self.assertEqual(result["analytical_noise_components"].iloc[0], '["gaussian", "drift"]')
        self.assertAlmostEqual(result["analytical_effective_intensity"].iloc[0], 0.25, places=12)

    def test_missingness_targets_only_rows_passing_the_declared_coverage_rule(self):
        frame = self.fixture().iloc[:24].copy()
        frame.loc[12:, "valid_area_fraction_bbox"] = 0.2
        result = perturb_analytical_series(frame, "fixture_temperature", family="missingness", intensity=0.5, seed=19, coverage_threshold=0.5)
        self.assertEqual(int(result.value.iloc[:12].isna().sum()), 6)
        self.assertEqual(int(result.value.iloc[12:].isna().sum()), 0)

    def test_cached_trend_uses_scientific_hash_context_and_returns_same_result(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = GlobalAnalyticalCache(Path(temp), max_disk_bytes=1_000_000)
            frame = self.fixture()
            first = cached_analyze_variable(frame, "fixture_temperature", cache=cache, coverage_threshold=0.5, key_context={"gold_sha256": "a" * 64, "contract_version": "1.0.0", "region_geometry_hash": "r1"})
            second = cached_analyze_variable(frame, "fixture_temperature", cache=cache, coverage_threshold=0.5, key_context={"gold_sha256": "a" * 64, "contract_version": "1.0.0", "region_geometry_hash": "r1"})
            changed = cached_analyze_variable(frame, "fixture_temperature", cache=cache, coverage_threshold=0.5, key_context={"gold_sha256": "b" * 64, "contract_version": "1.0.0", "region_geometry_hash": "r1"})
            disk_cache = GlobalAnalyticalCache(Path(temp), max_memory_bytes=0, max_disk_bytes=1_000_000)
            disk_result = cached_analyze_variable(frame, "fixture_temperature", cache=disk_cache, coverage_threshold=0.5, key_context={"gold_sha256": "a" * 64, "contract_version": "1.0.0", "region_geometry_hash": "r1"})
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertFalse(changed["cache_hit"])
            self.assertTrue(disk_result["cache_hit"])
            self.assertEqual(disk_cache.stats()["json_disk_hits"], 1)
            self.assertEqual(first["trend"], second["trend"])
            self.assertEqual(first["trend"], changed["trend"])


if __name__ == "__main__":
    unittest.main()
