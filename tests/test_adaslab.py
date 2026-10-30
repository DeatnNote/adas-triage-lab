import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from adaslab import firmware, triage
from adaslab.api import Analytics, create_app
from adaslab.frames import image_features, render
from adaslab.llm import card, rules_triage, scene_score
from adaslab.sim import CHANNELS, T, generate, save, telemetry, vin_hash


@pytest.fixture(scope="module")
def small():
    return generate(n_vehicles=30, days=12, seed=3)


def test_telemetry_signatures():
    rng = np.random.default_rng(0)
    X, key = telemetry("phantom_brake", 30.0, "clear", "day", rng)
    assert X.shape == (len(CHANNELS), len(T))
    assert np.isnan(X[CHANNELS.index("lead_dist")]).all() and key["max_decel"] > 2  # braking with nothing ahead
    X, _ = telemetry("sharp_curve", 25.0, "clear", "day", rng)
    assert np.abs(X[CHANNELS.index("lat_accel")]).max() > 2


def test_vins_are_pseudonymised_and_stable():
    assert vin_hash("W1K123") == vin_hash("W1K123") != vin_hash("W1K124")
    assert "W1K" not in vin_hash("W1K123")


def test_raw_files_carry_no_truth(small, tmp_path):
    save(small, tmp_path)
    raw = pd.read_parquet(tmp_path / "raw" / "events.parquet")
    assert not {"scenario", "severity", "root_cause"} & set(raw.columns)


def test_frames_render_and_features():
    sc = {"weather": "snow", "light": "night", "lane_markings": "faded", "objects": "pedestrian|cones"}
    img = render(sc, 1)
    assert img.size == (512, 288)
    assert image_features(img).shape == (43,)


def test_queue_counts_severity4_seen_same_day():
    df = pd.DataFrame({"day": [0, 0, 0, 1, 1], "severity": [4, 1, 4, 4, 2]})
    s = pd.Series([0.9, 0.8, 0.1, 0.5, 0.6])
    q = triage.daily_queue(df, s, np.ones(5, bool), budget_per_day=1)
    assert q["sev4_reviewed_same_day"] == pytest.approx(1 / 3)


def test_did_recovers_a_known_rate_ratio():
    rng = np.random.default_rng(0)
    rows = []
    for v in range(300):
        wave = [3, 6, 99][v % 3]
        for w in range(12):
            miles = rng.uniform(50, 150)
            season = 1 + 0.08 * w  # rates rise over time for everyone
            new = w >= wave
            rows.append({"vin": f"v{v}", "week": w, "miles": miles, "new": float(new),
                         "n": rng.poisson(0.02 * miles * season * (0.6 if new else 1.0))})
    g = pd.DataFrame(rows)
    waves = pd.Series({f"v{v}": [3, 6, 99][v % 3] for v in range(300)})
    r = firmware.did_poisson(g, waves)
    assert 0.5 < r["rate_ratio"] < 0.7 and r["ci"][1] < 0.8
    assert firmware.naive_before_after(g, 3, 6)["rate_ratio"] > 0.75  # winter hides the improvement


def test_event_card_uses_no_truth_fields():
    r = pd.Series({"event_type": "safety", "road": "highway", "weather": "clear", "light": "day", "firmware": "4.3.0",
                   "speed_pre": 30, "speed_drop": 8, "max_decel": 5.0, "max_brake": 60, "max_abs_lat": 0.3,
                   "max_abs_torque": 0.4, "lane_min": 0.9, "lane_drop": 0.0, "vis_min": 0.9, "vis_low_s": 0.0,
                   "lead_frac": 0.0, "lead_min": 200, "ttc_min": 99, "scenario": "SECRET", "severity": 4})
    c = card(r)
    assert "SECRET" not in c and "no lead object" in c
    assert rules_triage(r)["scenario"] == "phantom_brake"


def test_scene_score_object_f1():
    sub = pd.DataFrame({"weather": ["rain"], "light": ["day"], "lane_markings": ["clear"], "objects": ["cones|lead_car"]})
    s = scene_score([{"weather": "rain", "light": "day", "lane_markings": "clear", "objects": ["cones"]}], sub)
    assert s["objects_precision"] == 1.0 and s["objects_recall"] == 0.5


def test_api_endpoints():
    ev = pd.DataFrame({"event_id": ["E1", "E2", "E3"], "day": [0, 0, 1], "event_type": ["safety", "takeover", "safety"],
                       "firmware": ["4.2.1", "4.3.0", "4.3.0"], "road": "highway", "weather": "clear", "light": "day",
                       "speed_pre": 30.0, "speed_drop": 1.0, "max_decel": 1.0, "max_brake": 0.0, "max_abs_lat": 0.1,
                       "max_abs_torque": 0.1, "lane_min": 0.9, "lane_drop": 0.0, "vis_min": 0.9, "vis_low_s": 0.0,
                       "lead_frac": 0.0, "lead_min": 200.0, "ttc_min": 99.0})
    Z = np.eye(3, dtype="float32")
    drives = pd.DataFrame({"firmware": ["4.2.1", "4.3.0"], "engaged_miles": [100.0, 200.0]})
    cl = TestClient(create_app(Analytics(ev, np.array([3.5, 1.2, 2.0]), Z, drives)))
    assert cl.get("/events/E1").json()["expected_severity"] == 3.5
    assert cl.get("/events/nope").status_code == 404
    assert cl.get("/triage/queue", params={"day": 0, "k": 1}).json()["results"][0]["event_id"] == "E1"
    assert len(cl.get("/events/E1/similar", params={"k": 2}).json()["results"]) == 2
    assert cl.get("/kpi/firmware").json()["results"]


def test_spark_pipeline_end_to_end(small, tmp_path):
    pytest.importorskip("pyspark")
    from adaslab import pipeline
    save(small, tmp_path)
    lake = tmp_path / "lake"
    pipeline.bronze(tmp_path / "raw", lake)
    spark = pipeline.spark_session()
    try:
        s = pipeline.silver(spark, lake)
        g = pipeline.gold(spark, lake)
    finally:
        spark.stop()
    assert len(s) == len(small["events"]) and s.vin.str.len().eq(16).all()
    loc = pipeline.silver_local(lake).set_index("event_id").sort_index()
    sp = s.set_index("event_id").sort_index()
    for c in ("max_decel", "lane_drop", "lead_frac", "ttc_min", "vis_low_s", "speed_drop"):
        np.testing.assert_allclose(sp[c].values, loc[c].values, rtol=1e-4, atol=1e-4)  # Spark == NumPy
    pb = s.merge(small["events"][["event_id", "scenario"]], on="event_id")
    assert (pb[pb.scenario == "phantom_brake"].lead_frac == 0).all()
    assert g.engaged_miles.sum() == pytest.approx(small["drives"].engaged_miles.sum(), rel=1e-6)
