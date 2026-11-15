import unittest

from scripts.run_downloads import TransferPlanError, resolve_transfer_plan


class TransferPlanTests(unittest.TestCase):
    pilot = {
        "require_spatial_subset_when_available": True,
        "fallback_to_full_granules_only_if_within_budget": True,
    }
    source = {"allowed_hosts": ["data.example.nasa.gov"]}

    def test_prepared_subset_is_not_downloadable_until_response_validation(self):
        row = {
            "planned_transfer_mode": "HARMONY_EDR_CUBE_REQUEST_PREPARED_NOT_SUBMITTED",
            "subset_request_status": "PREPARED_NOT_SUBMITTED",
            "spatial_subsetting_advertised": True,
            "harmony_bbox_time_variable_subset_supported": True,
            "source_url": "https://data.example.nasa.gov/full.nc",
            "expected_full_granule_size_bytes": 1000,
        }
        with self.assertRaises(TransferPlanError):
            resolve_transfer_plan(row, self.pilot, self.source, 2000)

    def test_subset_requires_matching_request_hash_and_validated_exact_size(self):
        row = {
            "planned_transfer_mode": "SUBSET_RESULT_READY",
            "subset_request_status": "RESPONSE_VALIDATED",
            "subset_request_hash": "abc123",
            "subset_result_request_hash": "wrong-hash",
            "validated_subset_download_url": "https://harmony.earthdata.nasa.gov/jobs/123/result",
            "subset_validated_output_size_bytes": 500,
            "spatial_subsetting_advertised": True,
        }
        with self.assertRaises(TransferPlanError):
            resolve_transfer_plan(row, self.pilot, self.source, 2000)
        row["subset_result_request_hash"] = "abc123"
        plan = resolve_transfer_plan(row, self.pilot, self.source, 2000)
        self.assertEqual(plan["url"], row["validated_subset_download_url"])
        self.assertEqual(plan["expected_size_bytes"], 500)
        self.assertIn("harmony.earthdata.nasa.gov", plan["allowed_hosts"])

    def test_subset_result_over_budget_is_blocked(self):
        row = {
            "planned_transfer_mode": "SUBSET_RESULT_READY",
            "subset_request_status": "RESPONSE_VALIDATED",
            "subset_request_hash": "abc123",
            "subset_result_request_hash": "abc123",
            "validated_subset_download_url": "https://harmony.earthdata.nasa.gov/jobs/123/result",
            "subset_validated_output_size_bytes": 500,
            "harmony_bbox_time_variable_subset_supported": True,
        }
        with self.assertRaises(TransferPlanError):
            resolve_transfer_plan(row, self.pilot, self.source, 400)

    def test_full_file_fallback_requires_explicit_mode_and_budget(self):
        row = {
            "planned_transfer_mode": "SPATIAL_SUBSET_ROUTE_UNVERIFIED",
            "source_url": "https://data.example.nasa.gov/full.nc",
            "expected_full_granule_size_bytes": 1000,
        }
        with self.assertRaises(TransferPlanError):
            resolve_transfer_plan(row, self.pilot, self.source, 2000)
        row.update(
            {
                "planned_transfer_mode": "FULL_OBJECT_FALLBACK_APPROVED",
                "spatial_subsetting_advertised": False,
                "harmony_bbox_time_variable_subset_supported": False,
            }
        )
        plan = resolve_transfer_plan(row, self.pilot, self.source, 2000)
        self.assertEqual(plan["url"], row["source_url"])
        self.assertEqual(plan["expected_size_bytes"], 1000)
        with self.assertRaises(TransferPlanError):
            resolve_transfer_plan(row, self.pilot, self.source, 999)


if __name__ == "__main__":
    unittest.main()
