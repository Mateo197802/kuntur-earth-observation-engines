"""Coordinate-based extraction helpers for INAMHI annual station tables."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable


MONTHS = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4, "MAYO": 5, "JUNIO": 6,
    "JULIO": 7, "AGOSTO": 8, "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11, "DICIEMBRE": 12,
}
STATION_CODE = re.compile(r"^M[A-Z0-9]{2,4}$")
NUMBER = re.compile(r"^-?\d+(?:[.,]\d+)?$")


def normalize_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Z0-9]+", " ", normalized.upper()).strip()


def _numbers_in_x(words: Iterable[dict], left: float, right: float) -> list[float]:
    values = []
    for word in words:
        if left <= float(word["x0"]) < right:
            token = str(word["text"]).strip()
            if NUMBER.fullmatch(token):
                values.append(float(token.replace(",", ".")))
    return values


def parse_dms_words(words: Iterable[dict], *, axis: str) -> float | None:
    """Read a DMS coordinate from the fixed columns in INAMHI index tables."""
    if axis == "latitude":
        layouts = (
            ((250, 265), (265, 282), (282, 298), (295, 310)),
            ((237, 252), (252, 268), (268, 284), (282, 297)),
        )
    elif axis == "longitude":
        layouts = (
            ((310, 327), (327, 344), (344, 360), (355, 380)),
            ((297, 313), (313, 329), (329, 346), (345, 359)),
        )
    else:
        raise ValueError("axis must be latitude or longitude")
    words = list(words)
    for columns in layouts:
        components = [_numbers_in_x(words, *column) for column in columns[:3]]
        if not all(component for component in components):
            continue
        degree, minute, second = (component[0] for component in components)
        hemisphere_words = [str(w["text"]).strip().upper() for w in words if columns[3][0] <= float(w["x0"]) < columns[3][1]]
        hemisphere = next((h for h in hemisphere_words if h in {"N", "S", "E", "W"}), None)
        if hemisphere is None or minute >= 60 or second >= 60:
            continue
        value = degree + minute / 60.0 + second / 3600.0
        return -value if hemisphere in {"S", "W"} else value
    return None


def canonical_station_id(value: str) -> str:
    """Normalize yearbook codes with left-zero padding while preserving 3-digit codes."""
    code = str(value).strip().upper()
    match = re.fullmatch(r"M(\d+)", code)
    if not match:
        return code
    digits = match.group(1)
    if len(digits) > 3 and digits.startswith("0"):
        digits = digits.lstrip("0").zfill(3)
    return f"M{digits}"


def group_words_by_row(words: Iterable[dict], *, precision: int = 1) -> dict[float, list[dict]]:
    rows: dict[float, list[dict]] = defaultdict(list)
    for word in words:
        top = round(float(word["top"]), precision)
        rows[top].append(word)
    return {top: sorted(row, key=lambda item: float(item["x0"])) for top, row in rows.items()}


def parse_station_index_words(words: Iterable[dict]) -> list[dict[str, object]]:
    """Parse station rows on an INAMHI yearbook's published-station index."""
    records: list[dict[str, object]] = []
    for row in group_words_by_row(words).values():
        code_candidates = [(float(w["x0"]), str(w["text"]).strip().upper()) for w in row if 45 <= float(w["x0"]) < 105 and STATION_CODE.fullmatch(str(w["text"]).strip().upper())]
        if not code_candidates:
            continue
        code_x, code_source = min(code_candidates, key=lambda item: item[0])
        code = canonical_station_id(code_source)
        name = " ".join(str(w["text"]) for w in row if code_x + 15 <= float(w["x0"]) < 205).strip()
        lat = parse_dms_words(row, axis="latitude")
        lon = parse_dms_words(row, axis="longitude")
        shifted_layout = code_x < 65
        elevations = _numbers_in_x(row, 357, 380) if shifted_layout else _numbers_in_x(row, 372, 398)
        climatology_pages = _numbers_in_x(row, 515, 550) if shifted_layout else _numbers_in_x(row, 468, 550)
        if lat is None or lon is None:
            continue
        records.append({
            "station_id": code,
            "station_code_source": code_source,
            "station_name": name,
            "latitude": lat,
            "longitude": lon,
            "elevation_m": elevations[0] if elevations else None,
            "climatology_page_printed": int(climatology_pages[0]) if climatology_pages else None,
        })
    return records


def parse_monthly_station_words(words: Iterable[dict], *, year: int, station: dict[str, object], source_page: int) -> list[dict[str, object]]:
    """Extract monthly mean shade-air temperature and total precipitation.

    Annual-book page layout columns are fixed: monthly mean temperature is at
    x=260..280 pt; monthly precipitation sum is at x=450..480 pt. Blank cells
    remain missing. Source page and station metadata are copied to each row.
    """
    result: list[dict[str, object]] = []
    seen_months: set[int] = set()
    for top, row in group_words_by_row(words).items():
        month_word = next((str(w["text"]).strip() for w in row if float(w["x0"]) < 100 and normalize_label(str(w["text"])) in MONTHS), None)
        if month_word is None:
            continue
        month_number = MONTHS[normalize_label(month_word)]
        # The same page contains a second table (wind) with another set of
        # Spanish month labels. Retain the first monthly-climate table only.
        if month_number in seen_months:
            continue
        seen_months.add(month_number)
        temp_values = _numbers_in_x(row, 260, 280)
        precipitation_values = _numbers_in_x(row, 450, 480)
        temp = temp_values[0] if temp_values else None
        precipitation = precipitation_values[0] if precipitation_values else None
        result.append({
            **station,
            "year": int(year),
            "month_number": month_number,
            "month": f"{year:04d}-{month_number:02d}",
            "tavg_c": temp,
            "precipitation_mm_month": precipitation,
            "source_pdf_page": int(source_page),
            "tavg_physical_range_ok": temp is None or -90.0 <= float(temp) <= 60.0,
            "precipitation_physical_range_ok": precipitation is None or 0.0 <= float(precipitation) <= 3000.0,
        })
    return result
