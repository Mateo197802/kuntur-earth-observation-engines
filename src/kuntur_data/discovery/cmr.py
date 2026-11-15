"""Small read-only client for NASA's Common Metadata Repository (CMR)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


class CMRRequestError(RuntimeError):
    """Raised for an unsuccessful or malformed official CMR response."""


@dataclass(frozen=True)
class CMRClient:
    base_url: str = "https://cmr.earthdata.nasa.gov/search"
    timeout_seconds: float = 30.0
    user_agent: str = "kuntur-data-catalog/0.1 (read-only metadata discovery)"

    def _get_json(self, endpoint: str, params: Iterable[tuple[str, str]]) -> dict[str, Any]:
        query = urlencode(list(params))
        url = urljoin(self.base_url.rstrip("/") + "/", endpoint.lstrip("/"))
        if query:
            url = f"{url}?{query}"
        request = Request(url, headers={"Accept": "application/json", "User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise CMRRequestError(f"CMR metadata request failed ({type(exc).__name__}).") from exc
        try:
            result = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CMRRequestError("CMR returned a response that was not valid UTF-8 JSON.") from exc
        if not isinstance(result, dict):
            raise CMRRequestError("CMR returned an unexpected JSON structure.")
        return result

    def collection(self, short_name: str, version: str) -> dict[str, Any]:
        result = self._get_json(
            "collections.umm_json",
            [("short_name[]", short_name), ("version", version), ("page_size", "10")],
        )
        for item in result.get("items", []):
            umm = item.get("umm", {})
            if umm.get("ShortName") == short_name and str(umm.get("Version")) == version:
                return item
        raise CMRRequestError(f"No exact CMR collection found for {short_name} version {version}.")

    def variables(self, concept_ids: list[str], batch_size: int = 40) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for start in range(0, len(concept_ids), batch_size):
            batch = concept_ids[start : start + batch_size]
            params = [("concept_id[]", concept_id) for concept_id in batch]
            params.append(("page_size", str(max(100, len(batch)))))
            result = self._get_json("variables.umm_json", params)
            records.extend(result.get("items", []))
        return records

    def latest_granule(self, collection_concept_id: str) -> dict[str, Any] | None:
        result = self._get_json(
            "granules.umm_json",
            [
                ("collection_concept_id", collection_concept_id),
                ("sort_key[]", "-start_date"),
                ("page_size", "1"),
            ],
        )
        items = result.get("items", [])
        return items[0] if items else None

    def granules_for_region_period(
        self,
        collection_concept_id: str,
        start: str,
        end: str,
        bbox: list[float],
        page_size: int = 2000,
    ) -> list[dict[str, Any]]:
        if len(bbox) != 4:
            raise ValueError("bbox must be [west, south, east, north].")
        bbox_value = ",".join(str(value) for value in bbox)
        result = self._get_json(
            "granules.umm_json",
            [
                ("collection_concept_id", collection_concept_id),
                ("temporal", f"{start},{end}"),
                ("bounding_box", bbox_value),
                ("page_size", str(page_size)),
            ],
        )
        items = result.get("items", [])
        hits = int(result.get("hits", len(items)))
        if hits > len(items):
            raise CMRRequestError(
                f"CMR returned only {len(items)} of {hits} matching granules; paging is required."
            )
        return items


def temporal_summary(umm: dict[str, Any]) -> tuple[str | None, str | None, bool]:
    starts: list[str] = []
    ends: list[str] = []
    ongoing = False
    extents = list(umm.get("TemporalExtents", []) or [])
    singular_extent = umm.get("TemporalExtent")
    if isinstance(singular_extent, dict):
        extents.append(singular_extent)
    for extent in extents:
        ongoing = ongoing or bool(extent.get("EndsAtPresentFlag"))
        ranges = extent.get("RangeDateTimes", []) or []
        singular_range = extent.get("RangeDateTime")
        if isinstance(singular_range, dict):
            ranges = [*ranges, singular_range]
        elif isinstance(ranges, dict):
            ranges = [ranges]
        for interval in ranges:
            if interval.get("BeginningDateTime"):
                starts.append(str(interval["BeginningDateTime"]))
            if interval.get("EndingDateTime"):
                ends.append(str(interval["EndingDateTime"]))
        instants = extent.get("SingleDateTimes", []) or []
        singular_instant = extent.get("SingleDateTime")
        if isinstance(singular_instant, dict):
            instants = [*instants, singular_instant]
        elif isinstance(instants, dict):
            instants = [instants]
        for instant in instants:
            if instant.get("DateTime"):
                starts.append(str(instant["DateTime"]))
                ends.append(str(instant["DateTime"]))
    return (min(starts) if starts else None, max(ends) if ends else None, ongoing)


def granule_size_bytes(umm: dict[str, Any]) -> int | None:
    """Return the primary data object's size from CMR archive metadata.

    A granule can list auxiliary QA/XML objects before or after its science file.
    Prefer the archive item whose name matches the HTTPS GET DATA URL; if CMR does
    not expose a direct file URL, use the largest listed object as a conservative
    estimate of the primary payload.
    """
    archive = umm.get("DataGranule", {}).get("ArchiveAndDistributionInformation", [])
    if not archive:
        return None

    data_url_name = None
    for related in umm.get("RelatedUrls", []):
        if str(related.get("Type", "")).upper() != "GET DATA":
            continue
        url = str(related.get("URL", ""))
        if url.startswith("https://"):
            data_url_name = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1].casefold()
            break

    multipliers = {
        "B": 1,
        "BYTE": 1,
        "BYTES": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    candidates: list[tuple[int, str]] = []
    for entry in archive:
        try:
            exact_bytes = entry.get("SizeInBytes")
            if exact_bytes is not None:
                size_bytes = int(exact_bytes)
            else:
                size = float(entry["Size"])
                unit = str(entry.get("SizeUnit", "")).strip().upper()
                multiplier = multipliers.get(unit)
                if multiplier is None:
                    continue
                size_bytes = round(size * multiplier)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if size_bytes < 0:
            continue
        name = str(entry.get("Name", "")).casefold()
        candidates.append((size_bytes, name))
        if data_url_name and name.rsplit("/", 1)[-1] == data_url_name:
            return size_bytes

    return max((size for size, _ in candidates), default=None)


def extract_data_url(umm: dict[str, Any]) -> str | None:
    for related in umm.get("RelatedUrls", []):
        kind = str(related.get("Type", "")).upper()
        url = related.get("URL")
        if url and kind == "GET DATA" and str(url).startswith("https://"):
            return str(url)
    for related in umm.get("RelatedUrls", []):
        url = related.get("URL")
        if url and str(url).startswith("https://") and "OPENDAP DATA" in str(related.get("Subtype", "")).upper():
            return str(url)
    return None
