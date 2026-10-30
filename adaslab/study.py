"""Every number in the README -> results/study.json (run after `python -m adaslab.pipeline`)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from deltalake import DeltaTable

from . import discover, features as F, firmware, triage
from .sim import FW_EFFECT, vin_hash

ROOT = Path(__file__).resolve().parents[1]
LAKE = ROOT / "data" / "lake"
REVIEWS_PER_DAY = 20  # two engineer-hours a day at 6 minutes an event


def main():
    df = F.load()
    img = F.image_matrix(df)
    rng = np.random.default_rng(0)
    hist = (df.day < 80).values
    test = ~hist
    labelled = hist & (rng.random(len(df)) < 0.25)
    out = {"data": {"events": len(df), "vehicles": int(df.vin.nunique()),
                    "drives": len(DeltaTable(str(LAKE / "bronze_drives")).to_pandas()),
                    "events_per_day_test": float(df[test].groupby("day").size().mean()),
                    "labelled_history": int(labelled.sum()), "test_events": int(test.sum()),
                    "scenario_counts": df.scenario.value_counts().to_dict(),
                    "severity_counts": df.severity.value_counts().sort_index().to_dict()}}

    # 1. classifiers by input set
    cls = {}
    for kind in ("telemetry", "telemetry+context", "+image", "+scene"):
        r = triage.evaluate_classifiers(df, F.matrix(df, kind, img), labelled, test)
        r.pop("predicted_scenario")
        cls[kind] = r
    out["classifiers"] = cls

    # 2. daily review queue
    X = F.matrix(df, "+image", img)
    m = triage.clf().fit(X[labelled], df.severity[labelled])
    expected = pd.Series(m.predict_proba(X) @ m.classes_, index=df.index)
    policies = {
        "chronological": pd.Series(-(df.day * 1e6 + df.t_offset_s).values, index=df.index),
        "safety events first, by deceleration": (df.event_type == "safety") * 100 + df.max_decel,
        "model: expected severity": expected,
        "oracle (true severity)": df.severity + pd.Series(rng.random(len(df)) * 0.1, index=df.index),
    }
    out["queue"] = {k: triage.daily_queue(df, v, test, REVIEWS_PER_DAY) for k, v in policies.items()}
    out["queue_budget_per_day"] = REVIEWS_PER_DAY

    # 3. active learning
    out["active_learning"] = {s: triage.active_learning(df, X, hist, test, strategy=s, rounds=14)
                              for s in ("random", "uncertainty")}

    # 4. discovery and retrieval
    shape = F.shape_embedding(df)
    disc = {}
    for name, Z in (("telemetry", discover.embed(df, shape)), ("telemetry+frame", discover.embed(df, shape, img))):
        c = discover.clusters(Z, df.scenario)
        c.pop("labels")
        disc[name] = {**c, **discover.retrieval_precision(Z, df.scenario)}
    out["discovery"] = disc

    # 5. firmware: labels from the scenario model trained on pre-rollout history only
    early = (df.day < 40).values & (rng.random(len(df)) < 0.3)
    df["pred_scenario"] = triage.clf().fit(X[early], df.scenario[early]).predict(X)
    veh = pd.read_parquet(ROOT / "data" / "raw" / "vehicles.parquet")
    waves = pd.Series(veh.update_day.values, index=veh.vin.map(vin_hash))
    drives = DeltaTable(str(LAKE / "bronze_drives")).to_pandas()
    fw = {}
    for scen in ("sharp_curve", "phantom_brake", "cut_in", "faded_lanes", "sensor_blockage_snow"):
        row = {"true_rate_ratio": FW_EFFECT.get(scen, 1.0)}
        for col, tag in (("scenario", "true_labels"), ("pred_scenario", "model_labels")):
            g = firmware.vehicle_week(df, col, scen, LAKE)
            row[f"did_{tag}"] = firmware.did_poisson(g, waves)
            row[f"naive_{tag}"] = firmware.naive_before_after(g, 40 // 7, 80 // 7)["rate_ratio"]
        row["monitor"] = firmware.sequential_monitor(df, drives, "pred_scenario", scen, 40)
        fw[scen] = row
    out["firmware"] = fw
    out["firmware_label_model_accuracy_after_rollout"] = float((df.pred_scenario == df.scenario)[df.day >= 40].mean())

    (ROOT / "results" / "study.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("data",)}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
