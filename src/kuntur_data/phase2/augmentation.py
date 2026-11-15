"""Deterministic, unit-aware time-series perturbations for Phase 2."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np
import pandas as pd


AUGMENTATION_VERSION = "kuntur-noise-augmentation-1.0.0"
CHANNELS = ("merra2_t2m_c", "merra2_qv2m", "merra2_u2m", "merra2_v2m", "merra2_ps")
LAGS = tuple(range(12, 0, -1))  # oldest to newest; newest remains T-3
FEATURE_COLUMNS = tuple(f"{channel}_lag_{lag:02d}" for lag in LAGS for channel in CHANNELS)
MISSING_COLUMNS = tuple(f"missing__{name}" for name in FEATURE_COLUMNS)
SINGLE_FAMILIES = ("gaussian", "laplace", "impulsive", "drift", "heteroscedastic", "missingness")
COMBINED = {
    "gaussian+drift": ("gaussian", "drift"),
    "gaussian+impulsive": ("gaussian", "impulsive"),
    "drift+missingness": ("drift", "missingness"),
    "gaussian+heteroscedastic": ("gaussian", "heteroscedastic"),
    "mixed_multi_family": ("gaussian", "laplace", "impulsive", "drift", "heteroscedastic", "missingness"),
}
TRAIN_INTENSITIES = (0.05, 0.10, 0.15, 0.20)
PHYSICAL_RANGES = {
    "merra2_t2m_c": (-123.15, 86.85),
    "merra2_qv2m": (0.0, 0.1),
    "merra2_u2m": (-150.0, 150.0),
    "merra2_v2m": (-150.0, 150.0),
    "merra2_ps": (10000.0, 120000.0),
}


def training_location_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate channel statistics from model-fit samples and their histories only."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (12, len(CHANNELS)):
        raise ValueError("Expected (samples, 12 history months, 5 channels)")
    flat = x.reshape(-1, len(CHANNELS))
    location = np.mean(flat, axis=0)
    scale = np.std(flat, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return location, scale


def _seed(base_seed: int, sample_id: str, scenario: str, intensity: float) -> int:
    key = f"{base_seed}|{sample_id}|{scenario}|{intensity:.6f}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def _single_component(family: str, x: np.ndarray, intensity: float, location: np.ndarray, scale: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    noise = np.zeros_like(x, dtype=np.float64)
    missing = np.zeros_like(x, dtype=bool)
    sigma = scale.reshape(1, -1)
    if family == "gaussian":
        noise = rng.normal(size=x.shape) * (intensity * sigma)
    elif family == "laplace":
        noise = rng.laplace(size=x.shape) * (intensity * sigma / np.sqrt(2.0))
    elif family == "impulsive":
        duty = 0.05
        events = rng.random(x.shape) < duty
        signs = rng.choice(np.array([-1.0, 1.0]), size=x.shape)
        noise = events * signs * (intensity / np.sqrt(duty)) * sigma
    elif family == "drift":
        rho = 0.8
        raw = rng.normal(size=x.shape)
        for t in range(1, raw.shape[0]):
            raw[t] = rho * raw[t - 1] + np.sqrt(1.0 - rho**2) * raw[t]
        raw -= raw.mean(axis=0, keepdims=True)
        rms = np.sqrt(np.mean(raw**2, axis=0, keepdims=True))
        raw /= np.where(rms > 1e-12, rms, 1.0)
        noise = raw * (intensity * sigma)
    elif family == "heteroscedastic":
        z = (x - location.reshape(1, -1)) / sigma
        raw = rng.normal(size=x.shape) * np.sqrt(1.0 + np.abs(z))
        rms = np.sqrt(np.mean(raw**2, axis=0, keepdims=True))
        raw /= np.where(rms > 1e-12, rms, 1.0)
        noise = raw * (intensity * sigma)
    elif family == "missingness":
        eligible = x.size
        count = int(round(intensity * eligible))
        if intensity > 0 and count == 0:
            count = 1
        indices = rng.choice(eligible, size=min(count, eligible), replace=False)
        missing.flat[indices] = True
    else:
        raise ValueError(f"Unknown noise family: {family}")
    params = {"family": family, "intensity_definition": _INTENSITY_DEFINITIONS[family]}
    if family == "drift":
        params["ar1_rho"] = 0.8
    if family == "impulsive":
        params["spike_duty_fraction"] = 0.05
    if family == "missingness":
        params["missing_cells"] = int(missing.sum())
        params["eligible_cells"] = int(missing.size)
    return noise, missing, params


_INTENSITY_DEFINITIONS = {
    "gaussian": "RMS additive perturbation / model-fit training SD = intensity; Gaussian zero mean.",
    "laplace": "RMS additive perturbation / model-fit training SD = intensity; Laplace scale intensity*SD/sqrt(2).",
    "impulsive": "5% of eligible feature cells receive signed spikes; spike magnitude=intensity*training SD/sqrt(0.05), so expected RMS equals intensity*SD.",
    "drift": "Centered AR(1) history perturbation with rho=0.8 and per-channel RMS=intensity*training SD.",
    "heteroscedastic": "Zero-mean Gaussian noise with cellwise variance proportional to 1+absolute training-standardized value, normalized to per-channel RMS=intensity*training SD.",
    "missingness": "Proportion of eligible input cells set missing equals intensity; targets are never changed; model receives training-mean imputation and an indicator per cell.",
}


def perturb_sample(
    x: np.ndarray,
    *,
    scenario: str,
    intensity: float,
    location: np.ndarray,
    scale: np.ndarray,
    seed: int,
) -> dict:
    """Corrupt a single (12, 5) sample; no clipping or target modification."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (12, len(CHANNELS)) or not (0 <= intensity <= 0.50):
        raise ValueError("Expected a 12x5 history and intensity between 0 and 0.50")
    if scenario == "clean":
        return {"values": x.copy(), "missing": np.zeros_like(x, dtype=bool), "components": {}, "parameters": {}, "seed": seed, "effective_intensity": 0.0}
    families = COMBINED.get(scenario, (scenario,))
    if any(name not in SINGLE_FAMILIES for name in families):
        raise ValueError(f"Unknown augmentation scenario: {scenario}")
    rngs = [np.random.default_rng(seed + i * 7919) for i in range(len(families))]
    component_budget = intensity / np.sqrt(len(families))
    total_noise = np.zeros_like(x, dtype=np.float64)
    total_missing = np.zeros_like(x, dtype=bool)
    parameters = {"scenario": scenario, "total_intensity": intensity, "normalization": "L2 effective budget across additive RMS and missing fraction"}
    additive_budgets = []
    requested_components = {}
    for family, rng in zip(families, rngs):
        noise, missing, params = _single_component(family, x, component_budget, location, scale, rng)
        total_noise += noise
        total_missing |= missing
        params["component_intensity"] = component_budget
        requested_components[family] = component_budget
        parameters[family] = params
        if family != "missingness":
            additive_budgets.append(component_budget)
    missing_fraction = float(total_missing.mean())
    target_additive_rms = float(np.sqrt(sum(v * v for v in additive_budgets)))
    if additive_budgets and np.any(total_missing):
        # Discrete missing cells are quantized on 60 inputs; rescale the additive part
        # so the realized L2 budget remains at the requested total intensity.
        target_additive_rms = float(np.sqrt(max(0.0, intensity**2 - missing_fraction**2)))
    elif not additive_budgets:
        target_additive_rms = 0.0
    if target_additive_rms > 0:
        standardized_rms = float(np.sqrt(np.mean((total_noise / scale.reshape(1, -1)) ** 2)))
        if standardized_rms > 1e-12:
            total_noise *= target_additive_rms / standardized_rms
    values = x + total_noise
    values[total_missing] = np.nan
    additive_rms = float(np.sqrt(np.nanmean((total_noise / scale.reshape(1, -1)) ** 2))) if target_additive_rms else 0.0
    effective = float(np.sqrt(additive_rms**2 + missing_fraction**2))
    invalid = []
    for j, channel in enumerate(CHANNELS):
        low, high = PHYSICAL_RANGES[channel]
        observed = values[:, j][~np.isnan(values[:, j])]
        if np.any((observed < low) | (observed > high)):
            invalid.append(channel)
    parameters.update({"realized_additive_rms_over_training_sd": additive_rms, "realized_missing_fraction": missing_fraction, "realized_effective_intensity": effective, "physical_range_violations": invalid})
    return {"values": values, "missing": total_missing, "components": requested_components, "parameters": parameters, "seed": int(seed), "effective_intensity": effective}


def history_array(frame: pd.DataFrame) -> np.ndarray:
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Forecast rows missing required features: {sorted(missing)}")
    return frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64).reshape(len(frame), 12, len(CHANNELS))


def model_matrix(x: np.ndarray, missing: np.ndarray, location: np.ndarray, *, add_mask: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    missing = np.asarray(missing, dtype=bool)
    x[missing] = np.broadcast_to(location.reshape(1, 1, -1), x.shape)[missing]
    flat = x.reshape(len(x), -1)
    if add_mask:
        flat = np.concatenate([flat, missing.reshape(len(x), -1).astype(np.float64)], axis=1)
    return flat


def training_augmentation(
    frame: pd.DataFrame,
    *,
    seed: int = 20261115,
    source_partition: str = "train_fit",
) -> tuple[pd.DataFrame, dict]:
    """Create balanced clean + 24 single + 20 combined fit-only examples per original."""
    x = history_array(frame)
    y = frame["target_value"].to_numpy(dtype=np.float64)
    ids = frame["sample_id"].astype(str).to_numpy()
    location, scale = training_location_scale(x)
    scenarios = [(name, 0.0) for name in ("clean",)]
    scenarios += [(family, level) for family in SINGLE_FAMILIES for level in TRAIN_INTENSITIES]
    scenarios += [(name, level) for name in COMBINED for level in TRAIN_INTENSITIES]
    records = []
    rejected = []
    for row, sample_id in enumerate(ids):
        for scenario, intensity in scenarios:
            scenario_seed = _seed(seed, sample_id, scenario, intensity)
            item = perturb_sample(x[row], scenario=scenario, intensity=intensity, location=location, scale=scale, seed=scenario_seed)
            violations = item["parameters"].get("physical_range_violations", [])
            if violations:
                rejected.append({"base_sample_id": sample_id, "scenario": scenario, "intensity": intensity, "reason": "broad physical screen exceeded", "channels": violations})
                continue
            values = item["values"].copy()
            miss = item["missing"].copy()
            values[miss] = np.broadcast_to(location.reshape(1, -1), values.shape)[miss]
            record = {
                "base_sample_id": sample_id,
                "augmentation_id": f"{sample_id}:{scenario}:{int(round(intensity * 100)):02d}",
                "noise_type": scenario,
                "noise_intensity": float(intensity),
                "noise_components": json.dumps(item["components"], sort_keys=True),
                "is_combined_noise": bool(scenario in COMBINED),
                "random_seed": item["seed"],
                "noise_parameters": json.dumps(item["parameters"], sort_keys=True),
                "source_partition": source_partition,
                "target_variable": "merra2_t2m_c",
                "target_value": float(y[row]),
                "augmentation_version": AUGMENTATION_VERSION,
            }
            for k, col in enumerate(FEATURE_COLUMNS):
                record[col] = float(values.reshape(-1)[k])
                record[MISSING_COLUMNS[k]] = int(miss.reshape(-1)[k])
            records.append(record)
    return pd.DataFrame.from_records(records), {"location": location, "scale": scale, "rejected": rejected, "scenario_count": len(scenarios), "expected_rows": len(frame) * len(scenarios)}
