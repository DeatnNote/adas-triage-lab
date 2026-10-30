"""Simulated ADAS fleet: drives, events, 10 Hz telemetry windows, scene attributes, and a staggered firmware rollout.

Everything is synthetic; the simulator writes the truth (scenario, severity, root cause) to data/truth/ for
scoring only. The pipeline reads data/raw/.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HZ, WIN = 10, 10  # 10 Hz, +-10 s around the event
T = np.arange(-WIN * HZ, WIN * HZ) / HZ
CHANNELS = ["speed", "long_accel", "lat_accel", "steer_torque", "brake_pressure", "lane_conf", "lead_dist",
            "ttc", "cam_visibility"]
ROADS = ["highway", "urban", "rural"]
WEATHER = ["clear", "rain", "snow", "fog"]
LIGHT = ["day", "low_sun", "night"]
COHORTS = ["commuter", "new_owner", "long_haul"]

# scenario -> (root cause, event type, base weight, base severity, scene objects, conditions)
SCENARIOS = {
    "cut_in":               ("other_road_user", "safety",       1.0, 3.0, ["car_adjacent_close"]),
    "pedestrian_crossing":  ("other_road_user", "safety",       0.35, 3.6, ["pedestrian"]),
    "faded_lanes":          ("perception",      "deactivation", 1.2, 1.6, ["faded_lanes"]),
    "construction_zone":    ("environment",     "takeover",     0.8, 2.0, ["cones"]),
    "sharp_curve":          ("planning",        "deactivation", 1.0, 2.2, ["curve"]),
    "sun_glare":            ("perception",      "deactivation", 0.6, 2.0, ["sun"]),
    "phantom_brake":        ("perception",      "safety",       0.5, 2.8, ["overpass"]),
    "sensor_blockage_snow": ("environment",     "deactivation", 0.5, 1.8, []),
    "comfort_override":     ("driver_preference", "takeover",   1.4, 1.0, []),
    "tunnel_exit_glare":    ("perception",      "deactivation", 0.08, 2.9, ["tunnel_exit"]),  # rare edge case
}
SCEN = list(SCENARIOS)
ROOT_CAUSES = sorted({v[0] for v in SCENARIOS.values()})
FW_OLD, FW_NEW = "4.2.1", "4.3.0"
FW_EFFECT = {"sharp_curve": 0.65, "phantom_brake": 1.6}  # rate multipliers under the new firmware (truth)


def vin_hash(vin: str, salt: str = "lab-salt") -> str:
    """Pseudonymise VINs before anything leaves the raw zone."""
    return hashlib.sha256((salt + vin).encode()).hexdigest()[:16]


def season(day: np.ndarray) -> dict:
    """Day 0 = Oct 1. Rain and snow become more common through the winter; low sun too."""
    f = np.clip(day / 120, 0, 1)
    return {"rain": 0.15 + 0.10 * f, "snow": 0.02 + 0.18 * f, "fog": 0.05, "low_sun": 0.12 + 0.10 * f}


def scenario_rates(road, weather, light, cohort, fw_new):
    """Unnormalised event rates per engaged mile for each scenario given the context."""
    r = {}
    for s, (_, _, w, _, _) in SCENARIOS.items():
        m = w
        if s == "pedestrian_crossing":
            m *= 3.0 if road == "urban" else 0.2
        if s == "cut_in":
            m *= 1.5 if road == "highway" else 0.8
        if s == "faded_lanes":
            m *= 2.0 if road == "rural" else 0.7
            m *= 1.8 if weather in ("rain", "snow") else 1.0
        if s == "sharp_curve":
            m *= 2.0 if road == "rural" else 0.6
            m *= 1.5 if weather != "clear" else 1.0
        if s == "sun_glare":
            m *= 5.0 if light == "low_sun" else 0.1
        if s == "phantom_brake":
            m *= 1.8 if road == "highway" else 0.5
        if s == "sensor_blockage_snow":
            m *= 12.0 if weather == "snow" else 0.02
        if s == "construction_zone":
            m *= 1.3 if road != "rural" else 0.7
        if s == "comfort_override":
            m *= 1.8 if cohort == "new_owner" else 0.8
        if s == "tunnel_exit_glare":
            m *= 3.0 if (road == "urban" and light != "night") else 0.5
        if fw_new and s in FW_EFFECT:
            m *= FW_EFFECT[s]
        r[s] = m * 0.0015  # events per engaged mile
    return r


def telemetry(scenario: str, speed0: float, weather: str, light: str, rng) -> tuple[np.ndarray, dict]:
    """(channels, n) window and the event's key quantities (min TTC etc.)."""
    n = len(T)
    pre, post = T < 0, T >= 0
    bump = np.exp(-0.5 * (T / 1.2) ** 2)
    step = 1 / (1 + np.exp(-3 * T))
    v = np.full(n, speed0) + rng.normal(0, 0.3, n).cumsum() * 0.05
    lon = rng.normal(0, 0.15, n)
    lat = rng.normal(0, 0.1, n)
    torque = rng.normal(0, 0.2, n)
    brake = np.zeros(n)
    lane = np.clip(rng.normal(0.9, 0.03, n), 0, 1)
    lead = np.full(n, np.nan)
    vis = np.clip(rng.normal(0.92 if weather == "clear" else 0.7, 0.03, n), 0, 1)
    ttc_min = np.inf
    if scenario in ("cut_in", "pedestrian_crossing"):
        gap0 = rng.uniform(6, 25) if scenario == "cut_in" else rng.uniform(8, 30)
        lead = np.where(T > -2, gap0 + np.maximum(T, 0) * rng.uniform(0.5, 3), np.nan)
        decel = rng.uniform(3, 9)
        lon -= decel * bump * (T > -1)
        brake += decel * 12 * bump * (T > -1)
        v = v - np.cumsum(np.where(T > -1, decel * bump, 0)) / HZ
        ttc_min = float(gap0 / max(speed0 * rng.uniform(0.15, 0.5), 0.5))
    elif scenario == "phantom_brake":
        decel = rng.uniform(2.5, 6.5)
        lon -= decel * bump
        brake += decel * 12 * bump
        v = v - np.cumsum(decel * bump) / HZ
        ttc_min = np.inf  # nothing ahead: that is the signature
    elif scenario in ("faded_lanes", "sensor_blockage_snow"):
        lane = np.clip(lane - 0.6 * step * rng.uniform(0.7, 1.0), 0, 1)
        if scenario == "sensor_blockage_snow":
            vis = np.clip(vis - 0.5 * step, 0, 1)
    elif scenario == "sharp_curve":
        a = rng.uniform(2.5, 4.5)
        lat += a * np.exp(-0.5 * ((T + 1) / 2.5) ** 2)
        torque += 1.5 * bump * rng.choice([-1, 1])
    elif scenario == "sun_glare":
        vis = np.clip(vis - 0.55 * bump * rng.uniform(0.8, 1.0), 0, 1)
        lane = np.clip(lane - 0.3 * bump, 0, 1)
    elif scenario == "tunnel_exit_glare":
        vis = np.clip(vis - 0.7 * np.exp(-0.5 * (T / 0.6) ** 2), 0, 1)  # sharp, short whiteout
        lane = np.clip(lane - 0.5 * np.exp(-0.5 * (T / 0.8) ** 2), 0, 1)
        lead = np.where(T > -5, rng.uniform(20, 40) + 0 * T, np.nan) if rng.random() < 0.6 else lead
    elif scenario == "construction_zone":
        lane = np.clip(lane - 0.35 * step, 0, 1)
        v = v - 4 * step
        torque += 2.5 * bump * rng.choice([-1, 1])
    elif scenario == "comfort_override":
        torque += rng.uniform(2, 4) * bump * rng.choice([-1, 1])
    ttc = np.where(np.isfinite(lead) & (v > 1), lead / np.maximum(v * 0.3, 0.5), np.nan)
    if np.isfinite(ttc_min):
        ttc = np.where(np.abs(T) < 0.5, ttc_min, ttc)
    X = np.vstack([np.maximum(v, 0), lon, lat, torque, np.maximum(brake, 0), lane, lead, ttc, vis]).astype("float32")
    return X, {"min_ttc": float(np.nanmin(ttc)) if np.isfinite(ttc).any() else np.inf,
               "max_decel": float(-lon.min()), "speed": float(speed0)}


def severity(scenario: str, key: dict, rng) -> int:
    base = SCENARIOS[scenario][3]
    s = base + rng.normal(0, 0.45)
    if np.isfinite(key["min_ttc"]):
        s += 1.2 if key["min_ttc"] < 1.5 else (0.5 if key["min_ttc"] < 2.5 else -0.3)
    s += 0.6 if key["max_decel"] > 6 else 0.0
    s += 0.3 if key["speed"] > 28 else 0.0
    return int(np.clip(np.round(s), 1, 4))


def scene(scenario: str, road: str, weather: str, light: str, rng) -> dict:
    objs = set(SCENARIOS[scenario][4])
    if rng.random() < 0.4:
        objs.add("lead_car")
    if scenario not in ("construction_zone",) and rng.random() < 0.04:
        objs.add("cones")  # incidental
    if scenario != "phantom_brake" and road == "highway" and rng.random() < 0.08:
        objs.add("overpass")  # incidental overpass: the scene alone does not prove a phantom brake
    lanes = "faded" if "faded_lanes" in objs or (weather == "snow" and rng.random() < 0.5) else "clear"
    objs.discard("faded_lanes")
    if light == "night" and "sun" in objs:
        light = "low_sun"
    return {"road": road, "weather": weather, "light": light, "lane_markings": lanes, "objects": sorted(objs)}


def generate(n_vehicles: int = 600, days: int = 120, seed: int = 21, out: Path | None = None) -> dict:
    rng = np.random.default_rng(seed)
    vins = [f"W1K{rng.integers(10**13, 10**14)}" for _ in range(n_vehicles)]
    cohort = rng.choice(COHORTS, n_vehicles, p=[0.5, 0.3, 0.2])
    region = rng.choice(["CA", "WA", "MI", "NY", "TX"], n_vehicles)
    # staggered over-the-air rollout: three waves, vehicles assigned at random
    wave = rng.choice([40, 60, 80, 999], n_vehicles, p=[0.25, 0.25, 0.25, 0.25])
    drives, events, windows = [], [], []
    for d in range(days):
        sea = season(np.array(d))
        for i in range(n_vehicles):
            for _ in range(rng.poisson(1.3 if cohort[i] != "long_haul" else 2.0)):
                road = rng.choice(ROADS, p=[0.5, 0.3, 0.2])
                snow_mult = 2.0 if region[i] in ("MI", "NY") else (0.1 if region[i] in ("CA", "TX") else 1.0)
                pw = np.array([0, sea["rain"], sea["snow"] * snow_mult, sea["fog"]])
                pw[0] = max(1 - pw[1:].sum(), 0.05)
                weather = rng.choice(WEATHER, p=pw / pw.sum())
                light = rng.choice(LIGHT, p=[1 - sea["low_sun"] - 0.2, sea["low_sun"], 0.2])
                miles = float(rng.gamma(2.0, 9.0 if cohort[i] != "long_haul" else 25.0))
                engaged = miles * rng.uniform(0.3, 0.9)
                fw_new = d >= wave[i]
                drive_id = f"D{len(drives):07d}"
                drives.append((drive_id, vins[i], d, road, weather, light, cohort[i], region[i],
                               FW_NEW if fw_new else FW_OLD, miles, engaged))
                rates = scenario_rates(road, weather, light, cohort[i], fw_new)
                for s, lam in rates.items():
                    for _ in range(rng.poisson(lam * engaged)):
                        speed0 = {"highway": rng.uniform(25, 36), "urban": rng.uniform(8, 18),
                                  "rural": rng.uniform(15, 27)}[road]
                        X, key = telemetry(s, speed0, weather, light, rng)
                        sev = severity(s, key, rng)
                        sc = scene(s, road, weather, light, rng)
                        eid = f"E{len(events):06d}"
                        events.append({"event_id": eid, "drive_id": drive_id, "vin": vins[i], "day": d,
                                       "t_offset_s": float(rng.uniform(0, miles / max(speed0, 1) * 1609)),
                                       "event_type": SCENARIOS[s][1], "road": road, "weather": weather,
                                       "light": sc["light"], "cohort": cohort[i], "region": region[i],
                                       "firmware": FW_NEW if fw_new else FW_OLD,
                                       "scenario": s, "root_cause": SCENARIOS[s][0], "severity": sev,
                                       "lane_markings": sc["lane_markings"], "objects": "|".join(sc["objects"])})
                        windows.append(X)
    dr = pd.DataFrame(drives, columns=["drive_id", "vin", "day", "road", "weather", "light", "cohort", "region",
                                       "firmware", "miles", "engaged_miles"])
    ev = pd.DataFrame(events)
    W = np.stack(windows)
    res = {"drives": dr, "events": ev, "windows": W,
           "vehicles": pd.DataFrame({"vin": vins, "cohort": cohort, "region": region, "update_day": wave})}
    if out is not None:
        save(res, out)
    return res


TRUTH_COLS = ["scenario", "root_cause", "severity", "lane_markings", "objects"]


def save(res: dict, out: Path):
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    ev = res["events"]
    res["drives"].to_parquet(out / "raw" / "drives.parquet", index=False)
    ev.drop(columns=TRUTH_COLS).to_parquet(out / "raw" / "events.parquet", index=False)
    np.save(out / "raw" / "windows.npy", res["windows"])  # row i belongs to events row i
    res["vehicles"].to_parquet(out / "raw" / "vehicles.parquet", index=False)
    ev[["event_id"] + TRUTH_COLS].to_parquet(out / "truth" / "events_truth.parquet", index=False)


if __name__ == "__main__":
    r = generate(out=ROOT / "data")
    ev = r["events"]
    print(len(r["drives"]), "drives", len(ev), "events", r["windows"].shape)
    print(ev.scenario.value_counts().to_string())
    print(ev.severity.value_counts().sort_index().to_string())
