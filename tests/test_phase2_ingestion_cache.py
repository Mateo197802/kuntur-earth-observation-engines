import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from kuntur_data.phase2.ingestion import GlobalAnalyticalCache, PartitionedGoldStore


class PartitionAndCacheTests(unittest.TestCase):
    def gold_fixture(self):
        return pd.DataFrame({
            "dataset_id": ["M2", "M2", "M2", "GPM"],
            "variable_id": ["t2m", "t2m", "qv", "rain"],
            "month": ["2020-01", "2021-03", "2021-03", "2021-03"],
            "region_id": ["ecuador"] * 4,
            "value": [20.0, 21.0, 0.01, 10.0],
            "units": ["degC", "degC", "kg kg-1", "mm"],
            "valid_area_fraction_bbox": [1.0] * 4,
            "contract_sha256": ["a" * 64] * 4,
        })

    def test_query_prunes_to_requested_variable_and_month_range(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gold_path = root / "gold.parquet"
            self.gold_fixture().to_parquet(gold_path, index=False)
            store = PartitionedGoldStore(gold_path, root / "partitioned")
            store.materialize()
            got = store.query(variable_ids=["t2m"], start_month="2021-01", end_month="2021-12", columns=["month", "variable_id", "value"])
            self.assertEqual(got.to_dict("records"), [{"month": "2021-03", "variable_id": "t2m", "value": 21.0}])

    def test_cache_returns_equal_frames_and_invalidates_when_inputs_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = pd.DataFrame({"month": ["2020-01"], "value": [3.5]})
            cache = GlobalAnalyticalCache(root, max_memory_bytes=64, max_disk_bytes=1_000_000)
            key = cache.key({"id": "key-a"})
            cache.put_frame(key, frame)
            pd.testing.assert_frame_equal(cache.get_frame(key), frame)
            cache.invalidate(key)
            self.assertIsNone(cache.get_frame(key))
            self.assertNotEqual(cache.key({"gold_sha256": "a", "contract_version": "1"}), cache.key({"gold_sha256": "b", "contract_version": "1"}))

    def test_concurrent_writes_leave_a_readable_complete_cache_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = GlobalAnalyticalCache(Path(temp), max_disk_bytes=1_000_000)
            frame = pd.DataFrame({"value": list(range(200))})
            key = cache.key({"id": "shared-key"})
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda _: cache.put_frame(key, frame), range(4)))
            pd.testing.assert_frame_equal(cache.get_frame(key), frame)


if __name__ == "__main__":
    unittest.main()
