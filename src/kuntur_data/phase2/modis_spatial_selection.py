"""Small, deterministic helpers for MOD13C2 QA and spatial-support audits."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


EARTH_RADIUS_M = 6_371_008.8
NDVI_EVI_FILL = -3000
NDVI_EVI_VALID_RANGE = (-2000, 10_000)


def build_quality_masks(
    ndvi_raw: np.ndarray,
    evi_raw: np.ndarray,
    vi_quality: np.ndarray,
    pixel_reliability: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return product-valid, contract-strict and relaxed QA masks.

    ``strict`` reproduces the frozen MOD13C2 contract: MODLAND=0,
    VI usefulness<=1, reliability=0, land-only, and no adjacent or mixed
    cloud flags. ``relaxed`` is a separately named sensitivity tier that
    includes marginal MODLAND/reliability values only when usefulness<=4;
    it does not replace the frozen contract.
    """
    ndvi = np.asarray(ndvi_raw)
    evi = np.asarray(evi_raw)
    qa = np.asarray(vi_quality)
    reliability = np.asarray(pixel_reliability)
    if not (ndvi.shape == evi.shape == qa.shape == reliability.shape):
        raise ValueError("MODIS science and QA arrays must have identical shapes")
    if any(array.dtype.kind not in "iuf" for array in (ndvi, evi, qa, reliability)):
        raise ValueError("MODIS science and QA arrays must be numeric")
    if any(array.dtype.kind == "f" and not np.isfinite(array).all() for array in (ndvi, evi, qa, reliability)):
        raise ValueError("MODIS science and QA arrays cannot contain nonfinite values")

    raw_low, raw_high = NDVI_EVI_VALID_RANGE
    ndvi_valid = (ndvi != NDVI_EVI_FILL) & (ndvi >= raw_low) & (ndvi <= raw_high)
    evi_valid = (evi != NDVI_EVI_FILL) & (evi >= raw_low) & (evi <= raw_high)
    qa_valid = (qa != 65535) & (qa >= 0) & (qa <= 65534)
    reliability_valid = (reliability >= 0) & (reliability <= 4)

    qa_uint = qa.astype(np.uint32, copy=False)
    modland = qa_uint & 0b11
    usefulness = (qa_uint >> 2) & 0b1111
    adjacent_cloud = (qa_uint >> 8) & 1
    mixed_cloud = (qa_uint >> 10) & 1
    land_water = (qa_uint >> 11) & 0b111

    common = qa_valid & reliability_valid & (land_water == 1) & (adjacent_cloud == 0) & (mixed_cloud == 0)
    strict_qa = common & (modland == 0) & (usefulness <= 1) & (reliability == 0)
    relaxed_qa = common & (modland <= 1) & (usefulness <= 4) & (reliability <= 1)

    return {
        "ndvi_product_valid": ndvi_valid,
        "evi_product_valid": evi_valid,
        "qa_strict": strict_qa,
        "qa_relaxed": relaxed_qa,
        "ndvi_strict": ndvi_valid & strict_qa,
        "evi_strict": evi_valid & strict_qa,
        "ndvi_relaxed": ndvi_valid & relaxed_qa,
        "evi_relaxed": evi_valid & relaxed_qa,
        "qa_valid": qa_valid,
        "pixel_reliability_valid": reliability_valid,
    }


def _cell_edges(centers: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.asarray(centers, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite cell centers are required")
    differences = np.diff(values)
    if not np.all(differences > 0):
        raise ValueError("Latitude and longitude centers must be strictly ascending")
    step = float(np.median(differences))
    if not np.allclose(differences, step, rtol=1e-4, atol=1e-8):
        raise ValueError("Nonuniform centers require authoritative cell bounds")
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = (values[:-1] + values[1:]) / 2
    edges[0] = max(low, values[0] - differences[0] / 2)
    edges[-1] = min(high, values[-1] + differences[-1] / 2)
    return np.clip(edges, low, high)


def area_weights_from_centers(
    latitude: np.ndarray,
    longitude: np.ndarray,
    bbox_wsen: Sequence[float],
) -> tuple[np.ndarray, float]:
    """Compute spherical cell areas clipped to a west/south/east/north bbox."""
    if len(bbox_wsen) != 4:
        raise ValueError("bbox must contain west, south, east, north")
    west, south, east, north = map(float, bbox_wsen)
    if not (west < east and south < north and -90 <= south < north <= 90):
        raise ValueError("bbox bounds are invalid")
    lat_edges = _cell_edges(latitude, south, north)
    lon_edges = _cell_edges(longitude, west, east)
    north_sine = np.sin(np.deg2rad(lat_edges[1:]))
    south_sine = np.sin(np.deg2rad(lat_edges[:-1]))
    delta_lon = np.deg2rad(np.diff(lon_edges))
    weights = EARTH_RADIUS_M**2 * (north_sine - south_sine)[:, None] * delta_lon[None, :]
    bbox_area = (
        EARTH_RADIUS_M**2
        * math.radians(east - west)
        * (math.sin(math.radians(north)) - math.sin(math.radians(south)))
    )
    if not np.isfinite(weights).all() or np.any(weights < 0) or bbox_area <= 0:
        raise ValueError("Derived spherical cell areas are invalid")
    return weights, bbox_area


def temporal_support_summary(
    eligible_by_month: np.ndarray,
    cell_area_weights: np.ndarray,
    *,
    bbox_area: float,
    months: Sequence[str],
    support_frequency_threshold: float = 0.5,
) -> dict[str, object]:
    """Summarize stable support and early/late spatial-selection drift.

    Early support is January 2002 through December 2010; late support is
    January 2016 through December 2024. The 2011-2015 middle period remains
    in the all-period stable/union metrics and is not silently discarded.
    """
    masks = np.asarray(eligible_by_month, dtype=bool)
    weights = np.asarray(cell_area_weights, dtype=np.float64).reshape(-1)
    if masks.ndim != 2 or masks.shape[0] != len(months) or masks.shape[1] != weights.size:
        raise ValueError("eligibility must be months by cells and match months/area weights")
    if not np.isfinite(weights).all() or np.any(weights < 0) or not math.isfinite(bbox_area) or bbox_area <= 0:
        raise ValueError("area weights and bbox area must be finite and nonnegative")
    if not (0 < support_frequency_threshold <= 1):
        raise ValueError("support_frequency_threshold must be in (0, 1]")
    years = []
    for month in months:
        try:
            years.append(int(str(month)[:4]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid month identifier: {month!r}") from exc

    monthly_area = masks @ weights
    stable = masks.all(axis=0)
    union = masks.any(axis=0)
    summary: dict[str, object] = {
        "month_count": int(masks.shape[0]),
        "cell_count": int(masks.shape[1]),
        "stable_pixel_count": int(np.count_nonzero(stable)),
        "union_pixel_count": int(np.count_nonzero(union)),
        "stable_area_fraction_bbox": float(weights[stable].sum() / bbox_area),
        "union_area_fraction_bbox": float(weights[union].sum() / bbox_area),
        "mean_monthly_area_fraction_bbox": float(np.mean(monthly_area / bbox_area)),
        "minimum_monthly_area_fraction_bbox": float(np.min(monthly_area / bbox_area)),
        "maximum_monthly_area_fraction_bbox": float(np.max(monthly_area / bbox_area)),
        "support_frequency_threshold": float(support_frequency_threshold),
    }

    early_rows = np.asarray([2002 <= year <= 2010 for year in years], dtype=bool)
    late_rows = np.asarray([2016 <= year <= 2024 for year in years], dtype=bool)
    if not early_rows.any() or not late_rows.any():
        summary.update(
            {
                "early_month_count": int(early_rows.sum()),
                "late_month_count": int(late_rows.sum()),
                "early_late_frequency_weighted_l1_bbox": None,
                "early_late_area_fraction_changed_by_10pp": None,
                "early_late_jaccard_area": None,
            }
        )
        return summary

    early_frequency = masks[early_rows].mean(axis=0)
    late_frequency = masks[late_rows].mean(axis=0)
    changed = np.abs(late_frequency - early_frequency)
    early_core = early_frequency >= support_frequency_threshold
    late_core = late_frequency >= support_frequency_threshold
    intersection_area = float(weights[early_core & late_core].sum())
    union_area = float(weights[early_core | late_core].sum())
    summary.update(
        {
            "early_month_count": int(early_rows.sum()),
            "late_month_count": int(late_rows.sum()),
            "early_mean_area_fraction_bbox": float(np.sum(weights * early_frequency) / bbox_area),
            "late_mean_area_fraction_bbox": float(np.sum(weights * late_frequency) / bbox_area),
            "early_late_frequency_weighted_l1_bbox": float(np.sum(weights * changed) / bbox_area),
            "early_late_area_fraction_changed_by_10pp": float(np.sum(weights[changed >= 0.1]) / bbox_area),
            "early_late_jaccard_area": intersection_area / union_area if union_area else None,
        }
    )
    return summary


def geospatial_quality_fractions(
    eligible: np.ndarray,
    qa_valid: np.ndarray,
    quality_codes: np.ndarray,
    cell_area_weights: np.ndarray,
) -> dict[str, float | None]:
    """Summarize MODIS geospatial quality only where the QA word is valid.

    Returned class fractions are conditional on valid QA among selected science
    pixels. ``valid_qa_fraction_selected`` exposes how much selected area had a
    readable QA word, so QA fill values are never assigned to bit-field class 3.
    """
    selected = np.asarray(eligible, dtype=bool)
    valid = np.asarray(qa_valid, dtype=bool)
    codes = np.asarray(quality_codes)
    weights = np.asarray(cell_area_weights, dtype=np.float64)
    if not (selected.shape == valid.shape == codes.shape == weights.shape):
        raise ValueError("MODIS QA masks, quality classes and area weights must have identical shapes")
    if codes.dtype.kind not in "iu" or np.any((codes < 0) | (codes > 3)):
        raise ValueError("MODIS geospatial quality classes must be integers in [0, 3]")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("MODIS area weights must be finite and nonnegative")
    selected_area = float(weights[selected].sum())
    selected_valid_qa_area = float(weights[selected & valid].sum())
    result: dict[str, float | None] = {
        "valid_qa_fraction_selected": (
            selected_valid_qa_area / selected_area if selected_area > 0 else None
        )
    }
    for quality_class in range(4):
        class_area = float(weights[selected & valid & (codes == quality_class)].sum())
        result[f"quality_class_{quality_class}_fraction_given_valid_qa"] = (
            class_area / selected_valid_qa_area if selected_valid_qa_area > 0 else None
        )
    return result
