import unittest

from kuntur_data.phase2.external import (
    parse_ghcnm_inventory_line,
    parse_ghcnm_tavg_line,
    station_monthly_composite,
)


class GHCNMParserTests(unittest.TestCase):
    def test_inventory_fixed_width_fields(self):
        line = f"EC000000001{'':1}{-1.25:8.2f}{'':1}{-78.50:9.2f}{'':1}{120.0:6.1f}{'':1}{'TEST STATION':30s}"
        parsed = parse_ghcnm_inventory_line(line)
        self.assertEqual(parsed["station_id"], "EC000000001")
        self.assertAlmostEqual(parsed["latitude"], -1.25)
        self.assertAlmostEqual(parsed["longitude"], -78.5)
        self.assertEqual(parsed["station_name"], "TEST STATION")

    def test_data_units_missing_and_flags(self):
        months = []
        for month in range(1, 13):
            value = 2312 if month == 1 else (-9999 if month == 2 else 2000)
            qc = "R" if month == 3 else " "
            months.append(f"{value:5d}{' '}{qc}{'S' if month == 4 else ' '}")
        parsed = parse_ghcnm_tavg_line("EC0000000012020TAVG" + "".join(months))
        self.assertEqual(len(parsed), 12)
        self.assertAlmostEqual(parsed[0]["value_c"], 23.12)
        self.assertTrue(parsed[1]["missing_sentinel"])
        self.assertEqual(parsed[2]["qcflag"], "R")
        self.assertEqual(parsed[3]["dsflag"], "S")

    def test_equal_station_composite_requires_minimum_and_blank_qc(self):
        observations = [
            {"station_id": f"S{i}", "month": "2020-01", "value_c": float(i), "qcflag": "", "missing_sentinel": False}
            for i in range(1, 4)
        ]
        observations.append({"station_id": "S4", "month": "2020-01", "value_c": 99.0, "qcflag": "R", "missing_sentinel": False})
        observations.append({"station_id": "S1", "month": "2020-02", "value_c": 1.0, "qcflag": "", "missing_sentinel": False})
        result = station_monthly_composite(observations)
        self.assertEqual(result, [{"month": "2020-01", "tavg_equal_station_mean_c": 2.0, "eligible_station_count": 3, "station_min_c": 1.0, "station_max_c": 3.0}])

    def test_non_tavg_records_are_ignored(self):
        record = "EC0000000012020TMAX" + "".join(f"{i:5d}   " for i in range(12))
        self.assertEqual(parse_ghcnm_tavg_line(record), [])


if __name__ == "__main__":
    unittest.main()
