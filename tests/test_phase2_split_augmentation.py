import unittest

import numpy as np
import pandas as pd

from kuntur_data.phase2.augmentation import (
    FEATURE_COLUMNS,
    SINGLE_FAMILIES,
    history_array,
    perturb_sample,
    training_augmentation,
    training_location_scale,
)
from kuntur_data.phase2.splits import phase2_split


class Phase2SplitTests(unittest.TestCase):
    def test_80_20_split_is_chronological_and_retains_other_roles(self):
        rows = []
        for i in range(262):
            partition = "train" if i < 190 else ("validation" if i < 226 else "test")
            rows.append({
                "sample_id": f"s{i:03d}", "target_month": f"{2003 + (2 + i) // 12:04d}-{(2 + i) % 12 + 1:02d}",
                "forecast_origin_month": "2003-01", "issue_date_utc": "2003-01-01T00:00:00Z",
                "history_start_month": "2001-12", "history_end_month": "2002-11", "partition": partition,
            })
        # The sample dates must be unique and monotonic; use explicit dates for the 190/36/36 contract.
        dates = pd.period_range("2003-03", periods=190, freq="M").astype(str).tolist()
        dates += pd.period_range("2019-01", periods=36, freq="M").astype(str).tolist()
        dates += pd.period_range("2022-01", periods=36, freq="M").astype(str).tolist()
        for row, month in zip(rows, dates):
            row["target_month"] = month
        split = phase2_split(pd.DataFrame(rows))
        self.assertEqual(split.phase2_role.value_counts().to_dict(), {
            "model_fit": 152, "internal_robustness_holdout": 38, "validation": 36, "test": 36,
        })
        fit = split[split.phase2_role == "model_fit"]
        robust = split[split.phase2_role == "internal_robustness_holdout"]
        self.assertLess(fit.target_month.max(), robust.target_month.min())
        self.assertEqual(split.loc[split.partition == "validation", "phase2_role"].unique().tolist(), ["validation"])
        self.assertEqual(split.loc[split.partition == "test", "phase2_role"].unique().tolist(), ["test"])


class Phase2AugmentationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        base = np.zeros((10, 12, 5), dtype=np.float64)
        base[:, :, 0] = 20 + rng.normal(0, 2, (10, 12))
        base[:, :, 1] = 0.015 + rng.normal(0, 0.002, (10, 12))
        base[:, :, 2] = rng.normal(0, 1.5, (10, 12))
        base[:, :, 3] = rng.normal(0, 1.5, (10, 12))
        base[:, :, 4] = 92000 + rng.normal(0, 1000, (10, 12))
        self.x = base
        self.location, self.scale = training_location_scale(base[:8])

    def test_history_schema_uses_oldest_to_newest_and_60_values(self):
        frame = pd.DataFrame({column: [float(i)] for i, column in enumerate(FEATURE_COLUMNS)})
        values = history_array(frame)
        self.assertEqual(values.shape, (1, 12, 5))
        self.assertEqual(values[0, 0, 0], frame["merra2_t2m_c_lag_12"].iloc[0])
        self.assertEqual(values[0, -1, 0], frame["merra2_t2m_c_lag_01"].iloc[0])

    def test_gaussian_noise_is_deterministic_and_has_training_scaled_rms(self):
        kwargs = {"scenario": "gaussian", "intensity": 0.10, "location": self.location, "scale": self.scale, "seed": 11}
        one = perturb_sample(self.x[0], **kwargs)
        two = perturb_sample(self.x[0], **kwargs)
        np.testing.assert_array_equal(one["values"], two["values"])
        realized = np.sqrt(np.mean(((one["values"] - self.x[0]) / self.scale.reshape(1, -1)) ** 2))
        self.assertAlmostEqual(realized, 0.10, delta=0.04)

    def test_all_noise_families_preserve_target_and_are_labeled(self):
        frame = pd.DataFrame({"sample_id": ["a", "b"], "target_value": [20.0, 21.0]})
        for i, column in enumerate(FEATURE_COLUMNS):
            frame[column] = self.x[:2].reshape(2, -1)[:, i]
        augmented, meta = training_augmentation(frame, seed=12)
        expected_scenarios = 1 + len(SINGLE_FAMILIES) * 4 + 5 * 4
        self.assertEqual(meta["scenario_count"], expected_scenarios)
        self.assertEqual(len(augmented), 2 * expected_scenarios - len(meta["rejected"]))
        self.assertTrue(set(augmented["source_partition"]) == {"train_fit"})
        for sample_id, target in zip(frame.sample_id, frame.target_value):
            rows = augmented[augmented.base_sample_id == sample_id]
            np.testing.assert_allclose(rows.target_value.to_numpy(), target)
        self.assertTrue(set(SINGLE_FAMILIES).issubset(set(augmented.noise_type)))
        self.assertTrue(augmented.is_combined_noise.any())

    def test_missingness_is_explicit_and_not_zero_filled_without_indicator(self):
        item = perturb_sample(self.x[0], scenario="missingness", intensity=0.25, location=self.location, scale=self.scale, seed=9)
        self.assertAlmostEqual(item["missing"].mean(), 0.25, delta=0.02)
        self.assertTrue(np.isnan(item["values"][item["missing"]]).all())
        self.assertEqual(item["parameters"]["realized_missing_fraction"], item["missing"].mean())

    def test_combined_noise_effective_budget_is_controlled(self):
        item = perturb_sample(self.x[0], scenario="gaussian+drift", intensity=0.50, location=self.location, scale=self.scale, seed=10)
        self.assertAlmostEqual(item["parameters"]["realized_additive_rms_over_training_sd"], 0.50, places=6)
        self.assertAlmostEqual(item["effective_intensity"], 0.50, places=6)


if __name__ == "__main__":
    unittest.main()
