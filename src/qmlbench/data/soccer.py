"""Soccer: expected goals (xG) from StatsBomb Open Data.

Formulation
-----------
Input  : shot geometry and context. Pitch is 120 x 80 yards; the goal is centred at
         (120, 40) with posts at y = 36 and y = 44.
           distance      -- yards to goal centre (real)
           goal_angle    -- angle the goal mouth subtends from the shot location, in [0, pi] (real)
           bearing       -- direction to goal centre relative to the goal line normal (angle;
                            note it spans only about (-pi/2, pi/2), so unlike tides/wind it never wraps)
           is_header, first_time, under_pressure, free_kick -- context flags (binary)
Target : 1 if the shot was a goal else 0. This is a *probability* regression: models
         output P(goal) and are scored with Brier score, log-loss and calibration.

Penalties and shoot-out kicks (period 5) are excluded by default: their xG is essentially a
constant and would dominate the easy part of the problem.

StatsBomb's own xG is kept in ``extras`` as an industry reference model.
Licence: StatsBomb Open Data -- credit StatsBomb in any public output.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import RegressionDataset, read_processed, write_processed
from .io import RAW_DIR, cached_get, get

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
GOAL_X, GOAL_Y, HALF_WIDTH = 120.0, 40.0, 4.0  # yards

FEATURES = ["distance", "goal_angle", "bearing", "is_header", "first_time",
            "under_pressure", "free_kick"]
FEATURE_TYPES = ["real", "real", "angle", "binary", "binary", "binary", "binary"]


def shot_geometry(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (distance, goal_angle, bearing) for shot locations in StatsBomb coordinates."""
    dx = GOAL_X - np.asarray(x, dtype=float)
    dy = np.asarray(y, dtype=float) - GOAL_Y
    distance = np.hypot(dx, dy)
    # Angle between the vectors to the two posts; atan2 form stays valid for dx <= 0.
    goal_angle = np.arctan2(2 * HALF_WIDTH * dx, dx ** 2 + dy ** 2 - HALF_WIDTH ** 2)
    goal_angle = np.where(goal_angle < 0, goal_angle + np.pi, goal_angle)
    bearing = np.arctan2(dy, dx)
    return distance, goal_angle, bearing


def _shots_from_events(events: list[dict], match: dict) -> list[dict]:
    rows = []
    for ev in events:
        if ev.get("type", {}).get("name") != "Shot" or ev.get("period") == 5:
            continue
        shot = ev.get("shot", {})
        x, y = ev["location"][:2]
        rows.append({
            "match_id": match["match_id"],
            "competition": match["competition"]["competition_name"],
            "season": match["season"]["season_name"],
            "event_id": ev["id"],
            "period": ev.get("period"),
            "minute": ev.get("minute"),
            "x": x,
            "y": y,
            "shot_type": shot.get("type", {}).get("name"),
            "body_part": shot.get("body_part", {}).get("name"),
            "technique": shot.get("technique", {}).get("name"),
            "play_pattern": ev.get("play_pattern", {}).get("name"),
            "first_time": bool(shot.get("first_time", False)),
            "under_pressure": bool(ev.get("under_pressure", False)),
            "outcome": shot.get("outcome", {}).get("name"),
            "statsbomb_xg": shot.get("statsbomb_xg"),
        })
    return rows


def _match_shots(match: dict, refresh: bool) -> list[dict]:
    """Shots of one match, cached as a small JSON (the ~3 MB event file is not kept)."""
    cache = RAW_DIR / "soccer" / "shots" / f"{match['match_id']}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    events = get(f"{BASE}/events/{match['match_id']}.json").json()
    rows = _shots_from_events(events, match)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    return rows


def fetch(cfg: dict, refresh: bool = False) -> Path:
    matches = []
    for comp_id, season_id in cfg["competitions"]:
        body = cached_get(f"{BASE}/matches/{comp_id}/{season_id}.json",
                          RAW_DIR / "soccer" / "matches" / f"{comp_id}_{season_id}.json",
                          refresh=refresh)
        matches += json.loads(body)
    rows = []
    with ThreadPoolExecutor(max_workers=int(cfg.get("download_workers", 4))) as pool:
        futures = [pool.submit(_match_shots, m, refresh) for m in matches]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="StatsBomb matches"):
            rows += fut.result()
    df = pd.DataFrame(rows)
    if cfg.get("exclude_penalties", True):
        df = df[df["shot_type"] != "Penalty"]
    df["is_goal"] = (df["outcome"] == "Goal").astype(int)
    df = df.sort_values(["match_id", "period", "minute", "event_id"]).reset_index(drop=True)
    print(f"soccer: {df['match_id'].nunique()} matches, {len(df)} shots, "
          f"goal rate {df['is_goal'].mean():.3f}")
    return write_processed(df, cfg["name"])


def load(cfg: dict) -> RegressionDataset:
    df = read_processed(cfg["name"])
    distance, goal_angle, bearing = shot_geometry(df["x"].to_numpy(), df["y"].to_numpy())
    X = np.column_stack([
        distance,
        goal_angle,
        bearing,
        (df["body_part"] == "Head").to_numpy(dtype=float),
        df["first_time"].to_numpy(dtype=float),
        df["under_pressure"].to_numpy(dtype=float),
        (df["shot_type"] == "Free Kick").to_numpy(dtype=float),
    ])
    return RegressionDataset(
        name="soccer",
        X=X,
        y=df["is_goal"].to_numpy(dtype=float),
        feature_names=list(FEATURES),
        feature_types=list(FEATURE_TYPES),
        task="probability",
        split_kind="group",
        groups=df["match_id"].to_numpy(),
        extras=df[["statsbomb_xg", "competition", "x", "y"]].reset_index(drop=True),
        target_name="is_goal",
        meta={"competitions": cfg["competitions"],
              "attribution": "Data: StatsBomb Open Data (https://github.com/statsbomb/open-data)"},
    )
