import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from scripts.run_phase2_v12_native_grid_collocation import (
    _load_exported_station_table,
    _metrics,
    _one_record_per_month,
    nearest_grid_cell,
)
from scripts.verify_phase2_v12_native_grid_collocation import metric_summary, month_range


class NativeGridCollocationTests(unittest.TestCase):
    def test_verifier_calendar_range_is_inclusive(self):
        self.assertEqual(month_range("2002-11", "2003-02"), ["2002-11", "2002-12", "2003-01", "2003-02"])

    def test_independent_verifier_recomputes_hand_calculated_metrics(self):
        rows = [
            {"merra2_minus_station_c": "1", "station_temperature_c": "10", "merra2_t2m_c": "11",
             "cell_center_distance_km": "10", "month": "2002-01", "station_id": "A", "grid_lat_index": "0", "grid_lon_index": "0"},
            {"merra2_minus_station_c": "-2", "station_temperature_c": "20", "merra2_t2m_c": "18",
             "cell_center_distance_km": "20", "month": "2002-02", "station_id": "B", "grid_lat_index": "0", "grid_lon_index": "1"},
        ]
        result = metric_summary(rows)
        self.assertEqual(result["matched_station_months"], 2)
        self.assertEqual(result["matched_months"], 2)
        self.assertAlmostEqual(result["mae_c"], 1.5)
        self.assertAlmostEqual(result["rmse_c"], 2.5 ** 0.5)
        self.assertAlmostEqual(result["bias_c"], -0.5)
        self.assertAlmostEqual(result["pearson_r"], 1.0)
        self.assertAlmostEqual(result["p95_cell_distance_km"], 19.5)

    def test_nearest_cell_uses_native_centers_and_great_circle_distance(self):
        row, column, distance = nearest_grid_cell(
            0.9,
            10.9,
            np.array([0.0, 1.0]),
            np.array([10.0, 11.0]),
        )
        self.assertEqual((row, column), (1, 1))
        self.assertGreater(distance, 0)
        self.assertLess(distance, 20)

    def test_nearest_cell_rejects_invalid_coordinates(self):
        with self.assertRaises(ValueError):
            nearest_grid_cell(91, 0, np.array([0.0]), np.array([0.0]))

    def test_metrics_match_hand_calculated_pair_errors(self):
        rows = [
            {"merra2_minus_station_c": -1.0, "station_temperature_c": 10.0, "merra2_t2m_c": 9.0,
             "station_id": "a", "grid_lat_index": 0, "grid_lon_index": 0, "cell_center_distance_km": 5.0},
            {"merra2_minus_station_c": 1.0, "station_temperature_c": 12.0, "merra2_t2m_c": 13.0,
             "station_id": "b", "grid_lat_index": 0, "grid_lon_index": 1, "cell_center_distance_km": 15.0},
        ]
        result = _metrics(rows)
        self.assertEqual(result["matched_station_months"], 2)
        self.assertEqual(result["matched_stations"], 2)
        self.assertEqual(result["unique_grid_cells"], 2)
        self.assertAlmostEqual(result["mae_c"], 1.0)
        self.assertAlmostEqual(result["rmse_c"], 1.0)
        self.assertAlmostEqual(result["bias_c"], 0.0)
        self.assertAlmostEqual(result["pearson_r"], 1.0)
        self.assertAlmostEqual(result["mean_cell_distance_km"], 10.0)

    def test_manifest_requires_one_unambiguous_validated_record_per_month(self):
        row = {
            "source_id": "merra2_monthly_slv",
            "region_id": "ecuador_pilot_bbox",
            "month": "2002-01",
            "status": "VALIDATED",
            "processed_path": "data/silver/month.npz",
            "processed_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "regional_input_sha256": "c" * 64,
        }
        selected, superseded = _one_record_per_month([row], ["2002-01"], {"data/silver/month.npz": "a" * 64})
        self.assertEqual(selected["2002-01"], row)
        self.assertEqual(superseded, 0)
        with self.assertRaises(ValueError):
            _one_record_per_month([], ["2002-01"], {})
        conflict = dict(row, processed_sha256="d" * 64)
        selected, superseded = _one_record_per_month([row, conflict], ["2002-01"], {"data/silver/month.npz": "a" * 64})
        self.assertEqual(selected["2002-01"], row)
        self.assertEqual(superseded, 1)
        conflicting_lineage = dict(row, source_sha256="e" * 64)
        with self.assertRaises(ValueError):
            _one_record_per_month(
                [row, conflicting_lineage], ["2002-01"], {"data/silver/month.npz": "a" * 64}
            )

    def test_station_csv_requires_matching_parquet_and_manifest_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "station.csv"
            parquet_path = root / "station.parquet"
            manifest_path = root / "source.json"
            sidecar_path = Path(str(csv_path) + ".provenance.json")
            csv_bytes = b"station_id,month,temperature\nA,2002-01,12.5\n"
            parquet_bytes = b"synthetic parquet provenance fixture"
            manifest_bytes = b"{\"status\":\"VALIDATED\"}\n"
            csv_path.write_bytes(csv_bytes)
            parquet_path.write_bytes(parquet_bytes)
            manifest_path.write_bytes(manifest_bytes)
            sha = lambda payload: hashlib.sha256(payload).hexdigest()
            sidecar_path.write_text(json.dumps({
                "source_id": "NOAA_GHCN_M_QCU",
                "csv_sha256": sha(csv_bytes),
                "source_table_sha256": sha(parquet_bytes),
                "source_manifest_sha256": sha(manifest_bytes),
                "row_count": 1,
                "columns": ["station_id", "month", "temperature"],
            }), encoding="utf-8")
            frame, hashes = _load_exported_station_table(
                csv_path, parquet_path, manifest_path, source="NOAA_GHCN_M_QCU"
            )
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["station_id"], "A")
            self.assertEqual(hashes["csv_sha256"], sha(csv_bytes))
            csv_path.write_bytes(csv_bytes + b"\n")
            with self.assertRaises(ValueError):
                _load_exported_station_table(
                    csv_path, parquet_path, manifest_path, source="NOAA_GHCN_M_QCU"
                )


if __name__ == "__main__":
    unittest.main()
