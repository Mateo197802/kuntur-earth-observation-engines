"""Byte-level integrity checks; these do not replace scientific metadata checks."""

from __future__ import annotations

from pathlib import Path

from kuntur_data.download.http import sha256_file

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
NETCDF_CLASSIC_MAGIC = (b"CDF\x01", b"CDF\x02", b"CDF\x05")


def inspect_container(path: Path, expected_format: str) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise ValueError("File does not exist or is not a regular file.")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("File is empty.")
    with path.open("rb") as stream:
        header = stream.read(1024 * 1024)
    label = expected_format.casefold()
    if label in {"hdf5", "hdf", "netcdf4", "netcdf-4", "hdf5/netcdf4"}:
        valid = HDF5_MAGIC in header
        detected = "HDF5/NetCDF-4" if valid else "unknown"
    elif label in {"netcdf", "netcdf-classic", "netcdf3"}:
        valid = any(header.startswith(signature) for signature in NETCDF_CLASSIC_MAGIC) or HDF5_MAGIC in header
        detected = "NetCDF" if valid else "unknown"
    else:
        raise ValueError(f"Unsupported expected format hint: {expected_format}")
    if not valid:
        raise ValueError(f"Container signature does not match expected format {expected_format}.")
    return {"path": str(path), "size_bytes": size, "sha256": sha256_file(path), "detected_format": detected}
