from __future__ import annotations

import numpy as np

from kuntur_data.phase2.modis_spatial_selection import (
    area_weights_from_centers,
    build_quality_masks,
    geospatial_quality_fractions,
    temporal_support_summary,
)


def _qa(*, modland=0, usefulness=0, adjacent=0, mixed=0, land_water=1, geospatial=0):
    return (
        modland
        | (usefulness << 2)
        | (adjacent << 8)
        | (mixed << 10)
        | (land_water << 11)
        | (geospatial << 14)
    )


def test_quality_tiers_reproduce_contract_and_keep_a_named_relaxed_sensitivity():
    ndvi = np.array([1000, 1000, 1000, 1000, -3000, 10001], dtype=np.int16)
    evi = np.array([2000, 2000, 2000, 2000, 2000, 2000], dtype=np.int16)
    qa = np.array(
        [
            _qa(),
            _qa(modland=1, usefulness=4),
            _qa(modland=1, usefulness=5),
            _qa(adjacent=1),
            _qa(),
            _qa(),
        ],
        dtype=np.uint16,
    )
    reliability = np.array([0, 1, 1, 0, 0, 0], dtype=np.int8)

    masks = build_quality_masks(ndvi, evi, qa, reliability)

    assert masks["ndvi_product_valid"].tolist() == [True, True, True, True, False, False]
    assert masks["ndvi_strict"].tolist() == [True, False, False, False, False, False]
    assert masks["ndvi_relaxed"].tolist() == [True, True, False, False, False, False]
    assert masks["evi_strict"].tolist() == [True, False, False, False, True, True]


def test_spherical_cell_weights_clip_edges_to_the_declared_bbox():
    latitude = np.array([0.25, 0.75])
    longitude = np.array([10.25, 10.75])
    weights, bbox_area = area_weights_from_centers(
        latitude, longitude, [10.0, 0.0, 11.0, 1.0]
    )

    assert weights.shape == (2, 2)
    assert np.all(weights > 0)
    assert np.isclose(weights.sum(), bbox_area, rtol=1e-12)


def test_temporal_support_reports_stable_union_and_early_late_composition():
    eligible = np.array(
        [[True, True, False], [False, True, True]],
        dtype=bool,
    )
    weights = np.array([1.0, 2.0, 3.0])

    summary = temporal_support_summary(
        eligible,
        weights,
        bbox_area=6.0,
        months=["2002-01", "2016-01"],
    )

    assert summary["stable_pixel_count"] == 1
    assert summary["union_pixel_count"] == 3
    assert np.isclose(summary["stable_area_fraction_bbox"], 2 / 6)
    assert np.isclose(summary["union_area_fraction_bbox"], 1.0)
    assert np.isclose(summary["early_late_frequency_weighted_l1_bbox"], 4 / 6)
    assert np.isclose(summary["early_late_jaccard_area"], 2 / 6)


def test_geospatial_quality_fractions_do_not_count_qa_fill_as_class_three():
    result = geospatial_quality_fractions(
        eligible=np.array([True, True, True, True]),
        qa_valid=np.array([True, True, True, False]),
        quality_codes=np.array([0, 1, 2, 3], dtype=np.uint8),
        cell_area_weights=np.ones(4),
    )

    assert np.isclose(result["valid_qa_fraction_selected"], 0.75)
    assert np.allclose(
        [result[f"quality_class_{index}_fraction_given_valid_qa"] for index in range(4)],
        [1 / 3, 1 / 3, 1 / 3, 0],
    )
