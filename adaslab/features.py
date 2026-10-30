"""Model inputs: silver features from the lake, rendered-frame features, telemetry shape embeddings, truth."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from deltalake import DeltaTable
from sklearn.decomposition import PCA

from .frames import image_features, render

ROOT = Path(__file__).resolve().parents[1]
NUM = ["speed_pre", "speed_drop", "max_decel", "max_brake", "max_abs_lat", "max_abs_torque", "lane_min",
       "lane_drop", "lead_frac", "lead_min", "ttc_min", "vis_min", "vis_drop", "vis_low_s", "vis_shift"]
CAT = ["event_type", "road", "weather", "light"]
SCENE_OBJECTS = ["car_adjacent_close", "lead_car", "pedestrian", "cones", "overpass", "tunnel_exit", "sun", "curve"]


def load() -> pd.DataFrame:
    s = DeltaTable(str(ROOT / "data" / "lake" / "silver_event_features")).to_pandas()
    t = pd.read_parquet(ROOT / "data" / "truth" / "events_truth.parquet")
    raw = pd.read_parquet(ROOT / "data" / "raw" / "events.parquet")[["event_id"]].reset_index()  # row in windows.npy
    return s.merge(t, on="event_id").merge(raw, on="event_id").sort_values("index").reset_index(drop=True)


def image_matrix(df: pd.DataFrame, cache: Path = ROOT / "data" / "image_features.npy") -> np.ndarray:
    """Classical features of every event's rendered frame (cached; ~1 min the first time)."""
    if cache.exists():
        F = np.load(cache)
        if len(F) == len(df):
            return F
    F = np.stack([image_features(render(r, int(r["index"]), 256, 144)) for _, r in df.iterrows()])
    np.save(cache, F)
    return F


def scene_matrix(df: pd.DataFrame) -> np.ndarray:
    """Scene attributes as a perfect vision model would report them (an upper bound for the VLM step)."""
    objs = df.objects.fillna("").str.split("|")
    cols = [objs.map(lambda o, k=k: float(k in o)) for k in SCENE_OBJECTS]
    cols.append((df.lane_markings == "faded").astype(float))
    return np.column_stack(cols)


def shape_embedding(df: pd.DataFrame, dims: int = 16, fit_rows=None) -> np.ndarray:
    """PCA of the down-sampled, per-channel-standardised telemetry window."""
    W = np.load(ROOT / "data" / "raw" / "windows.npy", mmap_mode="r")[df["index"].values]
    W = np.nan_to_num(np.asarray(W), nan=0.0)
    W = W.reshape(len(W), W.shape[1], 20, -1).mean(-1)  # 9 channels x 20 points
    mu, sd = W.mean((0, 2), keepdims=True), W.std((0, 2), keepdims=True) + 1e-6
    Z = ((W - mu) / sd).reshape(len(W), -1)
    pca = PCA(dims, random_state=0).fit(Z if fit_rows is None else Z[fit_rows])
    return pca.transform(Z)


def matrix(df: pd.DataFrame, kind: str, img: np.ndarray | None = None) -> pd.DataFrame:
    """kind: telemetry | telemetry+context | +image | +scene (perfect vision)"""
    X = df[NUM].copy()
    if kind != "telemetry":
        X = X.join(pd.get_dummies(df[CAT], dtype=float))
    if kind == "+image":
        X = X.join(pd.DataFrame(img, index=df.index, columns=[f"img{i}" for i in range(img.shape[1])]))
    if kind == "+scene":
        X = X.join(pd.DataFrame(scene_matrix(df), index=df.index,
                                columns=[f"scene_{k}" for k in SCENE_OBJECTS + ["faded"]]))
    return X
