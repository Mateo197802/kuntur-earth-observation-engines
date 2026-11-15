"""Read-only NASA Harmony capability discovery and safe EDR request planning."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


class HarmonyRequestError(RuntimeError):
    """Raised when public Harmony capability metadata cannot be verified."""


@dataclass(frozen=True)
class HarmonyClient:
    base_url: str = "https://harmony.earthdata.nasa.gov"
    timeout_seconds: float = 30.0
    user_agent: str = "kuntur-data/0.1 (read-only NASA Harmony capability discovery)"

    def capabilities(self, collection_concept_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"C\d+-[A-Za-z0-9_]+", collection_concept_id):
            raise ValueError("Expected an exact NASA CMR collection concept ID.")
        parts = urlsplit(self.base_url)
        if parts.scheme != "https" or parts.hostname != "harmony.earthdata.nasa.gov":
            raise ValueError("Harmony metadata must use the official HTTPS host.")
        url = f"{self.base_url.rstrip('/')}/capabilities?{urlencode([('collectionId', collection_concept_id), ('version', '3')])}"
        request = Request(url, headers={"Accept": "application/json", "User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read(5 * 1024 * 1024 + 1)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise HarmonyRequestError(f"Harmony capabilities request failed ({type(exc).__name__}).") from exc
        if len(payload) > 5 * 1024 * 1024:
            raise HarmonyRequestError("Harmony capability response exceeded the 5 MiB metadata limit.")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HarmonyRequestError("Harmony returned invalid capability JSON.") from exc
        if not isinstance(value, dict) or value.get("conceptId") != collection_concept_id:
            raise HarmonyRequestError("Harmony returned capability metadata for a different collection.")

        supported_services: list[dict[str, Any]] = []
        for service in value.get("services", []):
            caps = service.get("capabilities", {})
            subset = caps.get("subsetting", {})
            supported_services.append(
                {
                    "name": service.get("name"),
                    "href": service.get("href"),
                    "bbox": bool(subset.get("bbox")),
                    "temporal": bool(subset.get("temporal")),
                    "variable": bool(subset.get("variable")),
                    "output_formats": [item.get("mimeType") for item in caps.get("outputFormats", [])],
                }
            )
        compatible = [
            service
            for service in supported_services
            if service["bbox"] and service["temporal"] and service["variable"]
        ]
        summary = value.get("summary", {})
        return {
            "status": "VERIFIED",
            "provider": "NASA Harmony",
            "concept_id": collection_concept_id,
            "short_name": value.get("shortName"),
            "capabilities_version": value.get("capabilitiesVersion"),
            "snapshot_utc": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "services": supported_services,
            "compatible_bbox_time_variable_services": [item["name"] for item in compatible],
            "supports_combined_subset": bool(compatible),
        }


def supports_combined_subset(capabilities: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(capabilities, dict)
        and capabilities.get("status") == "VERIFIED"
        and capabilities.get("supports_combined_subset") is True
        and capabilities.get("compatible_bbox_time_variable_services")
    )


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Subset period datetimes must include an explicit timezone.")
    return parsed.astimezone(timezone.utc)


def build_edr_cube_subset_url(
    collection_concept_id: str,
    granule_concept_id: str,
    bbox_wsen: list[float],
    period_start: str,
    period_end: str,
    variable_names: list[str],
    *,
    base_url: str = "https://harmony.earthdata.nasa.gov",
) -> str:
    """Build (but do not submit) a one-granule Harmony OGC EDR subset request.

    The URL is only a plan until an authenticated, bounded request is explicitly
    authorized. It intentionally contains no credential, signed URL, or callback.
    """
    if not re.fullmatch(r"C\d+-[A-Za-z0-9_]+", collection_concept_id):
        raise ValueError("Expected an exact NASA CMR collection concept ID.")
    if not granule_concept_id or any(ord(char) < 32 for char in granule_concept_id):
        raise ValueError("A non-empty CMR granule concept ID is required.")
    if len(bbox_wsen) != 4 or not all(math.isfinite(float(value)) for value in bbox_wsen):
        raise ValueError("bbox must contain four finite numbers in west,south,east,north order.")
    west, south, east, north = (float(value) for value in bbox_wsen)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox bounds are invalid for EPSG:4326.")
    start = _parse_utc(period_start)
    end = _parse_utc(period_end)
    if end < start:
        raise ValueError("Subset period end must not precede its start.")
    if not variable_names or any(not name or "," in name or any(ord(char) < 32 for char in name) for name in variable_names):
        raise ValueError("At least one valid, comma-free variable name is required.")
    parts = urlsplit(base_url)
    if parts.scheme != "https" or parts.hostname != "harmony.earthdata.nasa.gov":
        raise ValueError("Subset requests must target the official Harmony HTTPS host.")
    params = [
        ("bbox", ",".join(format(value, ".12g") for value in (west, south, east, north))),
        ("datetime", f"{start.isoformat().replace('+00:00', 'Z')}/{end.isoformat().replace('+00:00', 'Z')}"),
        ("parameter-name", ",".join(variable_names)),
        ("granuleId", granule_concept_id),
        ("f", "application/netcdf"),
        ("maxResults", "1"),
        ("ignoreErrors", "false"),
        ("forceAsync", "true"),
    ]
    endpoint = f"{base_url.rstrip('/')}/ogc-api-edr/1.1.0/collections/{collection_concept_id}/cube"
    return f"{endpoint}?{urlencode(params)}"
