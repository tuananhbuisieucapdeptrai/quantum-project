"""Tides: hourly sea level at a NOAA tide gauge.

Formulation
-----------
Input  : time t, represented as the phases phi_k(t) = (omega_k * t) mod 2pi of K tidal
         constituents (angle features, one qubit each in the default quantum encoding).
Target : observed hourly water level (m, relative to mean sea level).

Classical harmonic analysis models h(t) = sum_k A_k cos(omega_k t - g_k), i.e. a sparse
Fourier series in t. A quantum model with one Pauli-rotation encoding per phase is also a
Fourier series in the phases, so this domain directly tests whether the quantum model's
frequency spectrum matches the physics.

Domain baselines kept in ``extras``: NOAA's official harmonic prediction for the same hours.
Residual weather effects (storm surge) are not predictable from time alone, so there is an
irreducible error floor -- report it rather than hide it.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from .base import RegressionDataset, read_processed, wrap_angle, write_processed
from .io import RAW_DIR, cached_get

API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
APPLICATION = "qmlbench"

# Standard constituent angular speeds in degrees per (solar) hour.
CONSTITUENT_SPEEDS_DEG_PER_HOUR = {
    "M2": 28.9841042,
    "S2": 30.0000000,
    "N2": 28.4397295,
    "K2": 30.0821373,
    "K1": 15.0410686,
    "O1": 13.9430356,
    "P1": 14.9589314,
    "Q1": 13.3986609,
    "M4": 57.9682084,
    "M6": 86.9523127,
    "MK3": 44.0251729,
    "Sa": 0.0410686,
    "Ssa": 0.0821373,
}
EPOCH = pd.Timestamp("2000-01-01T00:00:00Z")


def _fetch_year(station: str, year: int, product: str, datum: str,
                refresh: bool = False) -> pd.DataFrame:
    params = {
        "product": product,
        "station": station,
        "begin_date": f"{year}0101",
        "end_date": f"{year}1231 23:59",
        "datum": datum,
        "units": "metric",
        "time_zone": "gmt",
        "format": "csv",
        "application": APPLICATION,
    }
    if product == "predictions":
        params["interval"] = "h"
    cache = RAW_DIR / "tides" / f"{station}_{product}_{year}.csv"
    body = cached_get(API, cache, params=params, refresh=refresh)
    df = pd.read_csv(io.BytesIO(body), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    if "Date Time" not in df.columns:
        raise RuntimeError(f"Unexpected NOAA response for {product} {year}: {body[:300]!r}")
    df["time"] = pd.to_datetime(df["Date Time"], utc=True)
    return df


def fetch(cfg: dict, refresh: bool = False) -> Path:
    station, datum = str(cfg["station"]), cfg.get("datum", "MSL")
    obs, pred = [], []
    for year in cfg["years"]:
        o = _fetch_year(station, year, "hourly_height", datum, refresh)
        obs.append(o[["time", "Water Level", "Sigma"]])
        p = _fetch_year(station, year, "predictions", datum, refresh)
        pred.append(p[["time", "Prediction"]])
    obs_df = pd.concat(obs).rename(columns={"Water Level": "water_level", "Sigma": "sigma"})
    pred_df = pd.concat(pred).rename(columns={"Prediction": "noaa_prediction"})
    df = obs_df.merge(pred_df, on="time", how="left")
    df = df.dropna(subset=["water_level"]).drop_duplicates("time").sort_values("time")
    df["hours_since_epoch"] = (df["time"] - EPOCH).dt.total_seconds() / 3600.0
    df["station"] = station
    return write_processed(df.reset_index(drop=True), cfg["name"])


def constituent_phases(hours: np.ndarray, constituents: list[str]) -> np.ndarray:
    """Phase of each constituent at each time, in radians wrapped to [-pi, pi)."""
    speeds = np.deg2rad([CONSTITUENT_SPEEDS_DEG_PER_HOUR[c] for c in constituents])
    return wrap_angle(np.outer(np.asarray(hours, dtype=float), speeds))


def load(cfg: dict) -> RegressionDataset:
    df = read_processed(cfg["name"])
    constituents = cfg["constituents"]
    X = constituent_phases(df["hours_since_epoch"].to_numpy(), constituents)
    return RegressionDataset(
        name="tides",
        X=X,
        y=df["water_level"].to_numpy(dtype=float),
        feature_names=[f"phase_{c}" for c in constituents],
        feature_types=["angle"] * len(constituents),
        task="regression",
        split_kind="temporal",
        time=df["time"].to_numpy(),
        extras=df[["noaa_prediction", "hours_since_epoch"]].reset_index(drop=True),
        target_name="water_level_m",
        meta={"station": cfg["station"], "constituents": constituents,
              "latitude": cfg.get("latitude"),
              "harmonic_constituents": cfg.get("harmonic_constituents", constituents),
              "test_start": cfg["split"]["test_start"]},
    )
