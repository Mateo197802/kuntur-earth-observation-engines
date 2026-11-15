import datetime as dt
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kuntur_data.preprocessing import (  # noqa: E402
    area_metrics,
    area_weights,
    calendar_hours,
    forecast_rows,
    modis_qa_mask,
    month_end_exclusive,
    month_range,
    process_one,
    sha256_file,
    write_parquet,
)


class PreprocessingMathTests(unittest.TestCase):
    def test_calendar_month_hours_include_leap_february(self):
        self.assertEqual(calendar_hours("2020-02"), 696)
        self.assertEqual(calendar_hours("2021-02"), 672)
        self.assertEqual(calendar_hours("2002-01"), 744)
        self.assertEqual(month_end_exclusive("2024-12"), dt.date(2025, 1, 1))

    def test_spherical_area_weights_cover_clipped_bbox_without_regridding(self):
        lat = np.array([-0.5, 0.5])
        lon = np.array([-0.5, 0.5])
        weights, bbox_area = area_weights(lat, lon, [-1, -1, 1, 1])
        self.assertEqual(weights.shape, (2, 2))
        self.assertAlmostEqual(float(weights.sum()), bbox_area, delta=bbox_area * 1e-12)
        self.assertAlmostEqual(float(weights[0, 0]), float(weights[1, 1]), delta=bbox_area * 1e-12)

    def test_weighted_metrics_keep_missing_area_visible(self):
        values = np.array([[1.0, 3.0], [5.0, 7.0]])
        weights = np.array([[1.0, 1.0], [1.0, 1.0]])
        valid = np.array([[True, True], [False, False]])
        fill = np.array([[False, False], [True, False]])
        rejected = np.array([[False, False], [False, True]])
        result = area_metrics(values, valid, fill, rejected, weights, bbox_area=8.0)
        self.assertEqual(result["value"], 2.0)
        self.assertEqual(result["valid_area_fraction_bbox"], 0.25)
        self.assertEqual(result["fill_fraction_of_represented_grid"], 0.25)
        self.assertEqual(result["range_reject_fraction_of_represented_grid"], 0.25)

    def test_modis_c2_qa_mask_uses_geospatial_bits_as_context(self):
        good_land = 1 << 11
        qa = np.array([
            good_land,
            good_land | 1,                         # MODLAND = not best
            good_land | (2 << 2),                  # usefulness > 1
            0,                                      # water
            good_land | (1 << 8),                  # adjacent cloud
            good_land | (1 << 10),                 # mixed cloud
            good_land | (3 << 14),                 # geospatial quality retained
            65535,                                  # QA fill
        ], dtype=np.uint16)
        reliability = np.array([0, 0, 0, 0, 0, 0, 0, -1], dtype=np.int16)
        passed, decoded = modis_qa_mask(qa, reliability)
        self.assertEqual(passed.tolist(), [True, False, False, False, False, False, True, False])
        self.assertEqual(int(decoded["geospatial_quality"][6]), 3)


class ForecastContractTests(unittest.TestCase):
    def test_forecast_windows_are_leakage_safe_and_chronologically_partitioned(self):
        contract = json.loads((PROJECT_ROOT / "configs" / "data_contract_v1.json").read_text(encoding="utf-8"))
        months = month_range("2002-01", "2024-12")
        gold_rows = []
        variable_ids = contract["forecast"]["predictor_variable_ids"]
        for index, month in enumerate(months):
            for variable_id in variable_ids:
                gold_rows.append({
                    "month": month, "variable_id": variable_id, "value": float(index),
                    "units": "degC" if variable_id == "merra2_t2m_c" else "unit-test",
                    "source_sha256": f"{index:064x}", "valid_area_fraction_bbox": 0.95,
                    "month_start_utc": f"{month}-01T00:00:00Z",
                    "month_end_exclusive_utc": "unused",
                })
        samples, split_rows, audit = forecast_rows(gold_rows, contract)
        self.assertEqual(len(samples), 262)
        self.assertEqual(audit["partition_counts"], {"train": 190, "validation": 36, "test": 36})
        self.assertEqual(samples[0]["target_month"], "2003-03")
        self.assertEqual(samples[0]["forecast_origin_month"], "2003-02")
        self.assertEqual(samples[0]["history_start_month"], "2002-01")
        self.assertEqual(samples[0]["history_end_month"], "2002-12")
        self.assertEqual(samples[0]["merra2_t2m_c_lag_01"], 11.0)
        self.assertEqual(len(json.loads(samples[0]["history_lineage_json"])), 60)
        self.assertTrue(all(row["history_end_month"] < row["forecast_origin_month"] for row in samples))
        self.assertEqual(len(split_rows), len(samples))


class RegionalIntegrationTests(unittest.TestCase):
    def _copy_processing_code(self, root):
        for relative in ("src/kuntur_data/preprocessing.py", "scripts/preprocess_kuntur_data.py"):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROJECT_ROOT / relative, destination)

    def test_merra_temperature_preprocessing_checks_lineage_and_kelvin_conversion(self):
        contract = json.loads((PROJECT_ROOT / "configs" / "data_contract_v1.json").read_text(encoding="utf-8"))
        contract["region"]["bbox_wsen"] = [-81, -1, -79, 1]
        month = "2002-01"
        source_id = "merra2_monthly_slv"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._copy_processing_code(root)
            input_path = root / "data/silver/regions/ecuador_pilot_bbox" / source_id / "2002" / "sample.h5"
            input_path.parent.mkdir(parents=True)
            start_days = (dt.date(2002, 1, 1) - dt.date(1970, 1, 1)).days
            end_days = (dt.date(2002, 2, 1) - dt.date(1970, 1, 1)).days
            with h5py.File(input_path, "w") as handle:
                handle.attrs["source_period_start"] = "2002-01-01"
                handle.attrs["source_period_end"] = "2002-01-31"
                coords = handle.create_group("coordinates")
                coords.create_dataset("lat", data=np.array([-0.5, 0.5]))
                coords.create_dataset("lon", data=np.array([-80.5, -79.5]))
                time = coords.create_dataset("time", data=np.array([start_days + 14]))
                time.attrs["units"] = "days since 1970-01-01 00:00:00 UTC"
                bounds = coords.create_dataset("time_bounds", data=np.array([[start_days, end_days]]))
                bounds.attrs["units"] = "days since 1970-01-01 00:00:00 UTC"
                variables = handle.create_group("variables")
                definitions = {
                    "T2M": (np.array([[[1.0e15, 274.15], [275.15, 276.15]]]), "K"),
                    "QV2M": (np.full((1, 2, 2), 0.01), "kg kg-1"),
                    "U2M": (np.ones((1, 2, 2)), "m s-1"),
                    "V2M": (np.ones((1, 2, 2)) * -1, "m s-1"),
                    "PS": (np.ones((1, 2, 2)) * 90000, "Pa"),
                }
                for name, (values, units) in definitions.items():
                    dataset = variables.create_dataset(name, data=values)
                    dataset.attrs["source_variable_name"] = name
                    dataset.attrs["units"] = units
                    dataset.attrs["_FillValue"] = 1.0e15
            region_sha = sha256_file(input_path)
            source_sha = "a" * 64
            acquisition = {(source_id, month): {"status": "VALIDATED", "sha256": source_sha, "object_id": "fixture-granule"}}
            regional = {(source_id, month): {
                "status": "VALIDATED", "source_sha256": source_sha,
                "output_path": input_path.relative_to(root).as_posix(),
                "output_bytes": input_path.stat().st_size, "output_sha256": region_sha,
            }}
            record, rows = process_one(root, contract, acquisition, regional, source_id, month, "b" * 64)
            by_id = {row["variable_id"]: row for row in rows}
            self.assertEqual(record["status"], "VALIDATED")
            self.assertEqual(len(rows), 6)
            self.assertAlmostEqual(by_id["merra2_t2m_k"]["value"], 275.15, places=8)
            self.assertAlmostEqual(by_id["merra2_t2m_c"]["value"], 2.0, places=8)
            self.assertEqual(by_id["merra2_t2m_k"]["fill_fraction_of_represented_grid"], 0.25)
            self.assertEqual(record["regional_input_sha256"], region_sha)
            with np.load(root / record["processed_path"], allow_pickle=False) as output:
                self.assertEqual(output["latitude"].tolist(), [-0.5, 0.5])
                self.assertAlmostEqual(float(output["t2m_c"][0, 1]), 1.0, places=8)

    def test_imerg_leap_february_total_uses_exact_calendar_hours(self):
        contract = json.loads((PROJECT_ROOT / "configs" / "data_contract_v1.json").read_text(encoding="utf-8"))
        contract["region"]["bbox_wsen"] = [-0.1, -0.1, 0.1, 0.1]
        month = "2020-02"
        source_id = "gpm_imerg_monthly_final"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._copy_processing_code(root)
            input_path = root / "data/silver/regions/ecuador_pilot_bbox" / source_id / "2020" / "sample.h5"
            input_path.parent.mkdir(parents=True)
            start_days = (dt.date(2020, 2, 1) - dt.date(1970, 1, 1)).days
            end_days = (dt.date(2020, 3, 1) - dt.date(1970, 1, 1)).days
            with h5py.File(input_path, "w") as handle:
                handle.attrs["source_period_start"] = "2020-02-01"
                handle.attrs["source_period_end"] = "2020-02-29"
                coords = handle.create_group("coordinates")
                coords.create_dataset("lat", data=np.array([-0.05, 0.05]))
                coords.create_dataset("lon", data=np.array([-0.05, 0.05]))
                coords.create_dataset("time", data=np.array([start_days + 14]))
                coords.create_dataset("time_bounds", data=np.array([[start_days, end_days]]))
                variables = handle.create_group("variables")
                for name, values, units in (
                    ("precipitation", np.array([[[1.0, 2.0], [3.0, 4.0]]]), "mm/hr"),
                    ("randomError", np.full((1, 2, 2), 0.1), "mm/hr"),
                    ("precipitationQualityIndex", np.ones((1, 2, 2)) * 2, "index"),
                ):
                    dataset = variables.create_dataset(name, data=values)
                    dataset.attrs["source_variable_name"] = name
                    dataset.attrs["units"] = units
                    dataset.attrs["_FillValue"] = -9999.9
            region_sha = sha256_file(input_path)
            source_sha = "c" * 64
            acquisition = {(source_id, month): {"status": "VALIDATED", "sha256": source_sha, "object_id": "fixture-imerg"}}
            regional = {(source_id, month): {
                "status": "VALIDATED", "source_sha256": source_sha,
                "output_path": input_path.relative_to(root).as_posix(),
                "output_bytes": input_path.stat().st_size, "output_sha256": region_sha,
            }}
            _record, rows = process_one(root, contract, acquisition, regional, source_id, month, "d" * 64)
            by_id = {row["variable_id"]: row for row in rows}
            self.assertAlmostEqual(by_id["imerg_precip_rate"]["value"], 2.5, places=10)
            self.assertEqual(by_id["imerg_precip_monthly_total"]["value"], 1740.0)
            with np.load(root / "data/silver/processed_native_grid/v1/gpm_imerg_monthly_final/2020/202002.npz", allow_pickle=False) as output:
                metadata = json.loads(str(output["metadata_json"].item()))
                self.assertEqual(metadata["transforms"]["imerg_precip_monthly_total"]["hours_in_month"], 696)

    def test_modis_scale_and_strict_quality_mask_are_applied_to_regional_mean(self):
        contract = json.loads((PROJECT_ROOT / "configs" / "data_contract_v1.json").read_text(encoding="utf-8"))
        contract["region"]["bbox_wsen"] = [0.0, -0.05, 0.1, 0.05]
        source_id = "modis_mod13c2_monthly_ndvi"
        month = "2002-01"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._copy_processing_code(root)
            input_path = root / "data/silver/regions/ecuador_pilot_bbox" / source_id / "2002" / "sample.npz"
            input_path.parent.mkdir(parents=True)
            names = {
                "ndvi": "CMG_0.05_Deg_Monthly_NDVI",
                "evi": "CMG_0.05_Deg_Monthly_EVI",
                "vi_quality": "CMG_0.05_Deg_Monthly_VI_Quality",
                "pixel_reliability": "CMG_0.05_Deg_Monthly_pixel_reliability",
            }
            attributes = {
                names["ndvi"]: {"_FillValue": -3000, "valid_range": [-2000, 10000], "scale_factor": 10000},
                names["evi"]: {"_FillValue": -3000, "valid_range": [-2000, 10000], "scale_factor": 10000},
                names["vi_quality"]: {"_FillValue": 65535, "valid_range": [0, 65534]},
                names["pixel_reliability"]: {"_FillValue": -1, "valid_range": [0, 4]},
            }
            metadata = {"source_period_start": "2002-01-01", "source_period_end": "2002-01-31", "variable_attributes": attributes}
            qa = np.array([[1 << 11, 1 << 11], [(1 << 11) | (1 << 8), 1 << 11]], dtype=np.uint16)
            with input_path.open("wb") as stream:
                np.savez_compressed(stream, **{
                    "latitude": np.array([-0.025, 0.025]), "longitude": np.array([0.025, 0.075]),
                    "time": np.array(["2002-01-15"]), "metadata_json": np.array(json.dumps(metadata)),
                    "CMG_0.05_Deg_Monthly_NDVI": np.array([[5000, 10000], [-2000, -3000]], dtype=np.int16),
                    "CMG_0.05_Deg_Monthly_EVI": np.array([[2000, 4000], [6000, -3000]], dtype=np.int16),
                    "CMG_0.05_Deg_Monthly_VI_Quality": qa,
                    "CMG_0.05_Deg_Monthly_pixel_reliability": np.zeros((2, 2), dtype=np.int8),
                })
            region_sha = sha256_file(input_path)
            source_sha = "e" * 64
            acquisition = {(source_id, month): {"status": "VALIDATED", "sha256": source_sha, "object_id": "fixture-modis"}}
            regional = {(source_id, month): {
                "status": "VALIDATED", "source_sha256": source_sha,
                "output_path": input_path.relative_to(root).as_posix(),
                "output_bytes": input_path.stat().st_size, "output_sha256": region_sha,
            }}
            record, rows = process_one(root, contract, acquisition, regional, source_id, month, "f" * 64)
            by_id = {row["variable_id"]: row for row in rows}
            self.assertAlmostEqual(by_id["modis_ndvi"]["value"], 0.75, places=10)
            self.assertAlmostEqual(by_id["modis_ndvi"]["qa_pass_fraction_of_candidates"], 2 / 3, places=10)
            self.assertAlmostEqual(by_id["modis_ndvi"]["valid_area_fraction_bbox"], 0.5, places=12)
            self.assertEqual(record["summary_metrics"]["modis_evi"]["valid_cell_count"], 2)

    def test_parquet_writer_reopens_and_hashes_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.parquet"
            rows, digest, size = write_parquet(path, [{"month": "2002-01", "value": 1.5}], {"contract": "test"})
            self.assertEqual(rows, 1)
            self.assertEqual(len(digest), 64)
            self.assertEqual(size, path.stat().st_size)
            self.assertGreater(size, 0)


if __name__ == "__main__":
    unittest.main()
