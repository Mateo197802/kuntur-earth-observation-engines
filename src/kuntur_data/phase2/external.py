"""Small, testable parsers and aggregation helpers for external validation data."""

from __future__ import annotations

from collections.abc import Iterable


def parse_ghcnm_inventory_line(line: str) -> dict[str, object]:
    """Parse one GHCN-M v4 .inv fixed-width station metadata record."""
    record = line.rstrip("\r\n")
    if len(record) < 68:
        raise ValueError(f"GHCN-M inventory record is too short: {len(record)}")
    return {
        "station_id": record[0:11].strip(),
        "latitude": float(record[12:20]),
        "longitude": float(record[21:30]),
        "elevation_m": float(record[31:37]),
        "station_name": record[38:68].strip(),
    }


def parse_ghcnm_tavg_line(line: str) -> list[dict[str, object]]:
    """Parse TAVG monthly entries from one GHCN-M v4 .dat record.

    NOAA stores values in hundredths of degrees Celsius and uses -9999 for
    missing. QC and source flags are retained verbatim for downstream policy.
    """
    record = line.rstrip("\r\n")
    if len(record) < 115:
        raise ValueError(f"GHCN-M data record is too short: {len(record)}")
    if record[15:19] != "TAVG":
        return []
    station_id = record[0:11].strip()
    year = int(record[11:15])
    rows: list[dict[str, object]] = []
    for month in range(1, 13):
        base = 19 + (month - 1) * 8
        raw_value = int(record[base : base + 5])
        rows.append(
            {
                "station_id": station_id,
                "year": year,
                "month_number": month,
                "month": f"{year:04d}-{month:02d}",
                "raw_value_hundredth_c": raw_value,
                "value_c": None if raw_value == -9999 else raw_value / 100.0,
                "dmflag": record[base + 5 : base + 6].strip(),
                "qcflag": record[base + 6 : base + 7].strip(),
                "dsflag": record[base + 7 : base + 8].strip(),
                "missing_sentinel": raw_value == -9999,
            }
        )
    return rows


def station_monthly_composite(
    observations: Iterable[dict[str, object]], *, min_stations: int = 3
) -> list[dict[str, object]]:
    """Equal-station composite using present TAVG values with blank QC flags.

    Blank QC denotes no QC failure or a value not evaluable by a QC check in
    the NOAA documentation. This is therefore a conservative eligibility rule,
    not a claim that every retained value passed every possible test.
    """
    if min_stations < 1:
        raise ValueError("min_stations must be positive")
    groups: dict[str, dict[str, float]] = {}
    for obs in observations:
        value = obs.get("value_c")
        if obs.get("missing_sentinel") or obs.get("qcflag", "") or value is None:
            continue
        month = str(obs["month"])
        groups.setdefault(month, {})[str(obs["station_id"])] = float(value)
    rows: list[dict[str, object]] = []
    for month, station_values in sorted(groups.items()):
        count = len(station_values)
        if count < min_stations:
            continue
        values = list(station_values.values())
        rows.append(
            {
                "month": month,
                "tavg_equal_station_mean_c": sum(values) / count,
                "eligible_station_count": count,
                "station_min_c": min(values),
                "station_max_c": max(values),
            }
        )
    return rows
