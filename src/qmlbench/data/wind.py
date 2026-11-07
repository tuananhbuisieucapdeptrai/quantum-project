"""Wind: power output of one Kelmarsh wind turbine (10-minute SCADA averages).

Formulation
-----------
Input  : wind speed (real), wind direction (angle), and optionally nacelle ambient
         temperature (real, a proxy for air density; ``use_ambient_temp`` in the config).
         Note: the 2016 archive has no wind direction before May, so the config uses 2017.
Target : normalised power P / P_rated in [0, ~1] (capacity factor of the 10-min interval).

Direction enters through wakes from neighbouring turbines and terrain, so the
direction-dependence is subtle and periodic -- a natural fit for rotation-gate encoding,
while classical models need explicit sin/cos features to respect the wrap-around at 360 deg.
A useful secondary target is the residual from the binned power curve (IEC 61400-12-1
style), which isolates the direction effect; it is computed in ``qmlbench.models.domain``.

Cleaning uses the operator's own flags rather than heuristics:
  * 'Data Availability' == 1 (full 10 minutes of valid data),
  * 'Lost Production to Downtime and Curtailment Total (kWh)' == 0.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .base import RegressionDataset, read_processed, wrap_angle, write_processed
from .io import RAW_DIR, HTTPRangeFile

COLS = {
    "# Date and time": "time",
    "Wind speed (m/s)": "wind_speed",
    "Wind direction (°)": "wind_dir_deg",
    "Nacelle position (°)": "nacelle_deg",
    "Power (kW)": "power_kw",
    "Nacelle ambient temperature (°C)": "ambient_temp_c",
    "Data Availability": "availability",
    "Lost Production to Downtime and Curtailment Total (kWh)": "lost_downtime_curtail_kwh",
    "Blade angle (pitch position) A (°)": "pitch_deg",
}


def _zenodo_url(cfg: dict) -> str:
    return f"https://zenodo.org/records/{cfg['zenodo_record']}/files/{cfg['archive']}?download=1"


def _raw_csv(cfg: dict, refresh: bool) -> Path:
    """Pull just one turbine's CSV out of the remote zip and cache it."""
    out_dir = RAW_DIR / "wind"
    prefix = f"Turbine_Data_Kelmarsh_{cfg['turbine']}_"
    cached = sorted(out_dir.glob(prefix + "*.csv"))
    if cached and not refresh:
        return cached[0]
    remote = HTTPRangeFile(_zenodo_url(cfg))
    with zipfile.ZipFile(remote) as zf:
        member = next(n for n in zf.namelist() if n.startswith(prefix))
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / member
        tmp = target.with_suffix(".csv.part")
        with zf.open(member) as src, open(tmp, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        tmp.replace(target)
    print(f"wind: downloaded {remote.bytes_downloaded / 1e6:.1f} MB of {remote.size / 1e6:.0f} MB archive")
    return target


def fetch(cfg: dict, refresh: bool = False) -> Path:
    path = _raw_csv(cfg, refresh)
    df = pd.read_csv(path, comment=None, skiprows=9, usecols=list(COLS), low_memory=False)
    df = df.rename(columns=COLS)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    n0 = len(df)
    df = df.dropna(subset=["wind_speed", "wind_dir_deg", "power_kw"])
    clean = cfg.get("cleaning", {})
    if clean.get("require_full_availability", True):
        df = df[df["availability"] == 1]
    if clean.get("drop_downtime_curtailment", True):
        df = df[df["lost_downtime_curtail_kwh"].fillna(0) == 0]
    df = df[df["wind_speed"] >= clean.get("min_wind_speed", 0.0)]
    df = df.assign(turbine=int(cfg["turbine"]), rated_power_kw=float(cfg["rated_power_kw"]))
    print(f"wind: kept {len(df)} / {n0} 10-min rows after cleaning")
    return write_processed(df.sort_values("time").reset_index(drop=True), cfg["name"])


def load(cfg: dict) -> RegressionDataset:
    df = read_processed(cfg["name"])
    cols = [df["wind_speed"].to_numpy(dtype=float),
            wrap_angle(np.deg2rad(df["wind_dir_deg"].to_numpy(dtype=float)))]
    names, types = ["wind_speed", "wind_direction"], ["real", "angle"]
    if cfg.get("use_ambient_temp", False):
        df = df.dropna(subset=["ambient_temp_c"]).reset_index(drop=True)
        cols = [df["wind_speed"].to_numpy(dtype=float),
                wrap_angle(np.deg2rad(df["wind_dir_deg"].to_numpy(dtype=float))),
                df["ambient_temp_c"].to_numpy(dtype=float)]
        names, types = names + ["ambient_temp"], types + ["real"]
    X = np.column_stack(cols)
    y = df["power_kw"].to_numpy(dtype=float) / float(cfg["rated_power_kw"])
    return RegressionDataset(
        name="wind",
        X=X,
        y=y,
        feature_names=names,
        feature_types=types,
        task="regression",
        split_kind="temporal",
        time=df["time"].to_numpy(),
        extras=df[["power_kw", "pitch_deg", "nacelle_deg", "ambient_temp_c"]].reset_index(drop=True),
        target_name="power_fraction_of_rated",
        meta={"turbine": cfg["turbine"], "rated_power_kw": cfg["rated_power_kw"],
              "test_start": cfg["split"]["test_start"]},
    )
