import tempfile
import unittest
from pathlib import Path

from kuntur_data.validation.files import HDF5_MAGIC, inspect_container


class FileValidationTests(unittest.TestCase):
    def test_recognizes_hdf5_container_and_hashes_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny.h5"
            path.write_bytes(HDF5_MAGIC + b"software-test-fixture")
            result = inspect_container(path, "HDF5")
            self.assertEqual(result["detected_format"], "HDF5/NetCDF-4")
            self.assertEqual(result["size_bytes"], len(HDF5_MAGIC) + len(b"software-test-fixture"))
            self.assertEqual(len(result["sha256"]), 64)

    def test_rejects_wrong_container_signature(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "not-netcdf.nc"
            path.write_bytes(b"not a scientific container")
            with self.assertRaises(ValueError):
                inspect_container(path, "NetCDF")


if __name__ == "__main__":
    unittest.main()
