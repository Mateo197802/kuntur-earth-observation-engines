import unittest

from kuntur_data.discovery.cmr import extract_data_url, granule_size_bytes, temporal_summary
from kuntur_data.discovery.harmony import build_edr_cube_subset_url


class CMRHelpersTests(unittest.TestCase):
    def test_temporal_summary_preserves_open_end(self):
        value = {
            "TemporalExtents": [
                {"RangeDateTimes": [{"BeginningDateTime": "2000-01-01T00:00:00Z"}], "EndsAtPresentFlag": True}
            ]
        }
        self.assertEqual(temporal_summary(value), ("2000-01-01T00:00:00Z", None, True))

    def test_granule_size_converts_mb_to_bytes(self):
        self.assertEqual(
            granule_size_bytes({"DataGranule": {"ArchiveAndDistributionInformation": [{"Size": 2.5, "SizeUnit": "MB"}]}}),
            2_500_000,
        )

    def test_granule_size_uses_primary_data_file_not_sidecar(self):
        umm = {
            "RelatedUrls": [{"Type": "GET DATA", "URL": "https://example.nasa.gov/data/product.h5"}],
            "DataGranule": {
                "ArchiveAndDistributionInformation": [
                    {"Name": "product.h5.iso.xml", "SizeInBytes": 90_000},
                    {"Name": "product.h5", "SizeInBytes": 155_336_252},
                    {"Name": "product.qa", "SizeInBytes": 7_249},
                ]
            },
        }
        self.assertEqual(granule_size_bytes(umm), 155_336_252)

    def test_extract_data_url_chooses_https_get_data(self):
        record = {
            "RelatedUrls": [
                {"URL": "s3://private/object", "Type": "GET DATA VIA DIRECT ACCESS"},
                {"URL": "https://data.example.test/object", "Type": "GET DATA"},
            ]
        }
        self.assertEqual(extract_data_url(record), "https://data.example.test/object")

    def test_harmony_subset_url_targets_one_granule_and_requested_variables(self):
        url = build_edr_cube_subset_url(
            "C1276812859-GES_DISC",
            "G4379178609-GES_DISC",
            [-81.5, -5.1, -75.0, 1.7],
            "2022-01-01T00:00:00Z",
            "2022-01-31T23:59:59Z",
            ["T2M", "QV2M"],
        )
        self.assertTrue(url.startswith("https://harmony.earthdata.nasa.gov/ogc-api-edr/1.1.0/collections/C1276812859-GES_DISC/cube?"))
        self.assertIn("granuleId=G4379178609-GES_DISC", url)
        self.assertIn("parameter-name=T2M%2CQV2M", url)
        self.assertIn("maxResults=1", url)

    def test_harmony_subset_url_rejects_invalid_bbox(self):
        with self.assertRaises(ValueError):
            build_edr_cube_subset_url(
                "C1276812859-GES_DISC",
                "G4379178609-GES_DISC",
                [-81.5, 1.7, -75.0, -5.1],
                "2022-01-01T00:00:00Z",
                "2022-01-31T23:59:59Z",
                ["T2M"],
            )


if __name__ == "__main__":
    unittest.main()
