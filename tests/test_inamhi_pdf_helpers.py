import unittest

from kuntur_data.phase2.inamhi_pdf import parse_dms_words, parse_monthly_station_words, parse_station_index_words


class InamhiPdfHelperTests(unittest.TestCase):
    def test_dms_columns_convert_to_signed_decimal_degrees(self):
        words = [
            {"x0": 251.8, "text": "1"}, {"x0": 257.4, "text": "°"},
            {"x0": 268.1, "text": "16"}, {"x0": 277.1, "text": "'"},
            {"x0": 285.4, "text": "44"}, {"x0": 294.4, "text": '"'},
            {"x0": 299.2, "text": "S"},
            {"x0": 312.8, "text": "78"}, {"x0": 321.8, "text": "°"},
            {"x0": 330.0, "text": "22"}, {"x0": 339.0, "text": "'"},
            {"x0": 350.2, "text": "0"}, {"x0": 355.8, "text": '"'},
            {"x0": 360.6, "text": "W"},
        ]
        self.assertAlmostEqual(parse_dms_words(words, axis="latitude"), -(1 + 16 / 60 + 44 / 3600))
        self.assertAlmostEqual(parse_dms_words(words, axis="longitude"), -(78 + 22 / 60))

    def test_index_row_extracts_station_and_coordinates(self):
        words = [
            {"x0": 71.7, "top": 140.3, "text": "M002"},
            {"x0": 91.4, "top": 140.3, "text": "LA"}, {"x0": 104.0, "top": 140.3, "text": "TOLA"},
            {"x0": 251.8, "top": 140.3, "text": "0"}, {"x0": 268.1, "top": 140.3, "text": "13"},
            {"x0": 285.4, "top": 140.3, "text": "46"}, {"x0": 299.2, "top": 140.3, "text": "S"},
            {"x0": 312.8, "top": 140.3, "text": "78"}, {"x0": 330.0, "top": 140.3, "text": "22"},
            {"x0": 350.2, "top": 140.3, "text": "0"}, {"x0": 360.6, "top": 140.3, "text": "W"},
            {"x0": 378.0, "top": 140.3, "text": "2480"},
        ]
        parsed = parse_station_index_words(words)[0]
        self.assertEqual(parsed["station_id"], "M002")
        self.assertEqual(parsed["station_name"], "LA TOLA")
        self.assertAlmostEqual(parsed["latitude"], -(13 / 60 + 46 / 3600))
        self.assertAlmostEqual(parsed["longitude"], -(78 + 22 / 60))
        self.assertEqual(parsed["elevation_m"], 2480.0)

    def test_index_layout_with_tight_hemisphere_columns_is_supported(self):
        # 2006–2012 yearbooks place S/W immediately before the nominal column
        # boundaries; the parser must retain those valid station coordinates.
        words = [
            {"x0": 66.9, "top": 100.0, "text": "M447"},
            {"x0": 86.6, "top": 100.0, "text": "24"},
            {"x0": 94.9, "top": 100.0, "text": "DE"},
            {"x0": 104.9, "top": 100.0, "text": "MAYO"},
            {"x0": 250.4, "top": 100.0, "text": "1"},
            {"x0": 266.7, "top": 100.0, "text": "16"},
            {"x0": 284.0, "top": 100.0, "text": "44"},
            {"x0": 297.8, "top": 100.0, "text": "S"},
            {"x0": 311.3, "top": 100.0, "text": "80"},
            {"x0": 328.6, "top": 100.0, "text": "25"},
            {"x0": 348.8, "top": 100.0, "text": "6"},
            {"x0": 359.2, "top": 100.0, "text": "W"},
            {"x0": 377.6, "top": 100.0, "text": "115"},
            {"x0": 481.7, "top": 100.0, "text": "100"},
        ]
        parsed = parse_station_index_words(words)[0]
        self.assertEqual(parsed["station_id"], "M447")
        self.assertEqual(parsed["station_name"], "24 DE MAYO")
        self.assertAlmostEqual(parsed["latitude"], -(1 + 16 / 60 + 44 / 3600))
        self.assertAlmostEqual(parsed["longitude"], -(80 + 25 / 60 + 6 / 3600))
        self.assertEqual(parsed["climatology_page_printed"], 100)

    def test_shifted_index_layout_with_zero_padded_station_codes_is_supported(self):
        # 2011–2012 books shift the coordinate columns left and publish
        # zero-padded station IDs (M0447 corresponds to M447 in older tables).
        words = [
            {"x0": 56.1, "top": 100.0, "text": "M0447"},
            {"x0": 79.0, "top": 100.0, "text": "24"},
            {"x0": 87.4, "top": 100.0, "text": "DE"},
            {"x0": 97.4, "top": 100.0, "text": "MAYO(JABONCILLO)"},
            {"x0": 238.3, "top": 100.0, "text": "1"},
            {"x0": 254.4, "top": 100.0, "text": "16"},
            {"x0": 271.3, "top": 100.0, "text": "44"},
            {"x0": 285.5, "top": 100.0, "text": "S"},
            {"x0": 298.5, "top": 100.0, "text": "80"},
            {"x0": 315.4, "top": 100.0, "text": "25"},
            {"x0": 335.3, "top": 100.0, "text": "6"},
            {"x0": 346.1, "top": 100.0, "text": "W"},
            {"x0": 363.0, "top": 100.0, "text": "115"},
            {"x0": 530.0, "top": 100.0, "text": "129"},
        ]
        parsed = parse_station_index_words(words)[0]
        self.assertEqual(parsed["station_id"], "M447")
        self.assertEqual(parsed["station_code_source"], "M0447")
        self.assertEqual(parsed["station_name"], "24 DE MAYO(JABONCILLO)")
        self.assertAlmostEqual(parsed["latitude"], -(1 + 16 / 60 + 44 / 3600))
        self.assertAlmostEqual(parsed["longitude"], -(80 + 25 / 60 + 6 / 3600))
        self.assertEqual(parsed["elevation_m"], 115.0)
        self.assertEqual(parsed["climatology_page_printed"], 129)

    def test_monthly_rows_extract_temperature_and_precipitation(self):
        words = [
            {"x0": 59.5, "top": 160.1, "text": "ENERO"},
            {"x0": 268.1, "top": 160.1, "text": "16.0"},
            {"x0": 455.6, "top": 160.1, "text": "44.7"},
        ]
        station = {"station_id": "M002", "station_name": "LA TOLA", "latitude": -0.2, "longitude": -78.4, "elevation_m": 2480.0}
        parsed = parse_monthly_station_words(words, year=2002, station=station, source_page=101)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["month"], "2002-01")
        self.assertEqual(parsed[0]["tavg_c"], 16.0)
        self.assertEqual(parsed[0]["precipitation_mm_month"], 44.7)


if __name__ == "__main__":
    unittest.main()
