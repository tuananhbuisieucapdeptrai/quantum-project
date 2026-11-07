"""Astronomy: photometric redshift of SDSS galaxies.

Formulation
-----------
Input  : 4 colours (u-g, g-r, r-i, i-z) and the r-band magnitude, from dereddened SDSS
         photometry. 5 real-valued features -> 5 qubits.
Target : spectroscopic redshift z (treated as ground truth).

This is the "ordinary tabular" control domain: no periodic inputs, a smooth but strongly
non-linear target, plenty of data. If quantum models only shine on periodic data, this is
where it should show.

Field-standard metrics (see ``qmlbench.metrics``): sigma_NMAD of dz/(1+z) and the
catastrophic-outlier fraction |dz|/(1+z) > 0.15.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import RegressionDataset, read_processed, write_processed
from .io import RAW_DIR, cached_get

BANDS = ["u", "g", "r", "i", "z"]


def _api(release: str) -> str:
    return f"https://skyserver.sdss.org/{release}/SkyServerWS/SearchTools/SqlSearch"


def _query(cfg: dict, after_id: int, n: int) -> str:
    mags = ", ".join(f"p.dered_{b}" for b in BANDS)
    return (
        f"SELECT TOP {n} s.specObjID, s.bestObjID, s.ra, s.dec, {mags}, "
        f"s.z AS zspec, s.zErr "
        f"FROM SpecObj s JOIN PhotoObj p ON s.bestObjID = p.objID "
        f"WHERE s.class = 'GALAXY' AND s.zWarning = 0 AND p.clean = 1 "
        f"AND p.dered_r < {cfg['r_max']} AND s.z > 0 AND s.z < {cfg['z_max']} "
        f"AND s.fiberID % {cfg['fiber_mod']} = {cfg['fiber_rem']} "
        f"AND s.specObjID > {after_id} "
        f"ORDER BY s.specObjID"
    )


def fetch(cfg: dict, refresh: bool = False) -> Path:
    api, batch = _api(cfg["release"]), int(cfg["batch_size"])
    frames, after, page = [], 0, 0
    with tqdm(desc="SDSS batches", unit="batch") as bar:
        while True:
            cache = RAW_DIR / "photoz" / (
                f"{cfg['release']}_mod{cfg['fiber_mod']}_{cfg['fiber_rem']}"
                f"_r{cfg['r_max']}_page{page:03d}.csv")
            body = cached_get(api, cache, params={"cmd": _query(cfg, after, batch),
                                                  "format": "csv"}, refresh=refresh)
            # Response starts with a "#Table1" line, then a CSV header.
            df = pd.read_csv(io.BytesIO(body), skiprows=1)
            if "specObjID" not in df.columns:
                raise RuntimeError(f"Unexpected SkyServer response: {body[:300]!r}")
            frames.append(df)
            bar.update(1)
            if len(df) < batch:
                break
            after, page = int(df["specObjID"].iloc[-1]), page + 1
    out = pd.concat(frames, ignore_index=True)
    # Repeat spectra of the same photometric object: keep the most precise redshift.
    out = out.sort_values("zErr").drop_duplicates("bestObjID").sort_values("specObjID")
    mags = [f"dered_{b}" for b in BANDS]
    out = out[(out[mags] > 0).all(axis=1) & (out[mags] < 30).all(axis=1)]
    return write_processed(out.reset_index(drop=True), cfg["name"])


def load(cfg: dict) -> RegressionDataset:
    df = read_processed(cfg["name"])
    m = {b: df[f"dered_{b}"].to_numpy(dtype=float) for b in BANDS}
    X = np.column_stack([m["u"] - m["g"], m["g"] - m["r"], m["r"] - m["i"],
                         m["i"] - m["z"], m["r"]])
    return RegressionDataset(
        name="photoz",
        X=X,
        y=df["zspec"].to_numpy(dtype=float),
        feature_names=["u_g", "g_r", "r_i", "i_z", "r_mag"],
        feature_types=["real"] * 5,
        task="regression",
        split_kind="random",
        extras=df[["ra", "dec", "zErr"]].reset_index(drop=True),
        target_name="redshift",
        meta={"release": cfg["release"], "r_max": cfg["r_max"]},
    )
