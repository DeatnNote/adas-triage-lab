"""Analytics API: event detail, similar events, today's triage queue, firmware KPIs.

    uvicorn adaslab.api:default_app --factory --port 8080   (after the pipeline and study have run)
"""
from __future__ import annotations

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from .llm import card


class Analytics:
    def __init__(self, events: pd.DataFrame, expected_severity: np.ndarray, embedding: np.ndarray,
                 drives: pd.DataFrame):
        self.ev = events.reset_index(drop=True).assign(expected_severity=expected_severity)
        self.pos = {e: i for i, e in enumerate(self.ev.event_id)}
        self.index = faiss.IndexFlatIP(embedding.shape[1])
        self.index.add(np.ascontiguousarray(embedding, dtype="float32"))
        self.Z = embedding
        self.drives = drives

    def event(self, eid: str) -> dict:
        r = self.ev.iloc[self.pos[eid]]
        return {"event_id": eid, "day": int(r.day), "event_type": r.event_type, "firmware": r.firmware,
                "expected_severity": round(float(r.expected_severity), 2), "card": card(r)}

    def similar(self, eid: str, k: int = 5) -> list[dict]:
        i = self.pos[eid]
        D, I = self.index.search(self.Z[i:i + 1].astype("float32"), k + 1)
        return [{"event_id": self.ev.event_id[j], "similarity": round(float(d), 3), "event_type": self.ev.event_type[j]}
                for d, j in zip(D[0], I[0]) if j != i][:k]

    def queue(self, day: int, k: int = 20) -> list[dict]:
        g = self.ev[self.ev.day == day].nlargest(k, "expected_severity")
        return [{"event_id": r.event_id, "expected_severity": round(float(r.expected_severity), 2),
                 "event_type": r.event_type} for r in g.itertuples()]

    def kpi(self) -> list[dict]:
        miles = self.drives.groupby("firmware").engaged_miles.sum()
        cnt = self.ev.groupby(["firmware", "event_type"]).size().unstack(fill_value=0)
        rate = (cnt.div(miles, axis=0) * 1000).round(3)
        return [{"firmware": fw, **{f"{c}_per_1000_engaged_miles": float(v) for c, v in row.items()}}
                for fw, row in rate.iterrows()]


def create_app(a: Analytics) -> FastAPI:
    app = FastAPI(title="ADAS event analytics")

    def _get(eid):
        if eid not in a.pos:
            raise HTTPException(404, "unknown event")
        return eid

    @app.get("/events/{eid}")
    def event(eid: str):
        return a.event(_get(eid))

    @app.get("/events/{eid}/similar")
    def similar(eid: str, k: int = 5):
        return {"results": a.similar(_get(eid), k)}

    @app.get("/triage/queue")
    def queue(day: int, k: int = 20):
        return {"day": day, "results": a.queue(day, k)}

    @app.get("/kpi/firmware")
    def kpi():
        return {"results": a.kpi()}

    return app


def default_app() -> FastAPI:
    from deltalake import DeltaTable

    from . import discover, features as F, triage
    from .pipeline import LAKE
    df = F.load()
    img = F.image_matrix(df)
    X = F.matrix(df, "+image", img)
    lab = (df.day < 80).values & (np.random.default_rng(0).random(len(df)) < 0.25)
    m = triage.clf().fit(X[lab], df.severity[lab])
    exp = m.predict_proba(X) @ m.classes_
    Z = discover.embed(df, F.shape_embedding(df), img)
    drop = ["scenario", "root_cause", "severity", "lane_markings", "objects"]  # truth never served
    return create_app(Analytics(df.drop(columns=drop), exp, Z, DeltaTable(str(LAKE / "bronze_drives")).to_pandas()))
