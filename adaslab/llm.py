"""Foundation-model steps, run with your own key:
  * VLM scene description: Claude reads the rendered dashcam frame -> weather, light, lane markings, objects.
  * LLM triage: an event card (telemetry summary + context [+ the VLM's scene]) -> severity, root cause,
    scenario and a two-sentence narrative, as a forced tool call with a JSON schema.

    PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
    python -m adaslab.llm --reuse-scenes                # prompt v2: 300 triage calls, scenes reused from v1's log
    python -m adaslab.llm --prompt v1                   # reproduce v1: 150 VLM + 300 triage calls
    python -m adaslab.llm --per-scenario 2 --dry-run    # no API calls: offline stand-ins, checks the plumbing

Writes results/llm_eval_<prompt>.json (scores against the truth) and results/llm_outputs_<prompt>.jsonl.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from . import features as F
from .frames import render
from .sim import LIGHT, ROOT_CAUSES, SCEN, WEATHER
from .triage import clf

ROOT = Path(__file__).resolve().parents[1]
OBJECTS = F.SCENE_OBJECTS

SCENE_TOOL = {
    "name": "describe_scene",
    "description": "Describe the dashcam frame.",
    "input_schema": {"type": "object", "properties": {
        "weather": {"type": "string", "enum": WEATHER},
        "light": {"type": "string", "enum": LIGHT},
        "lane_markings": {"type": "string", "enum": ["clear", "faded"]},
        "objects": {"type": "array", "items": {"type": "string", "enum": OBJECTS}},
        "summary": {"type": "string"}}, "required": ["weather", "light", "lane_markings", "objects", "summary"]},
}
SCENE_SYSTEM = (
    "You review frames from a car's forward camera. Report what is visible. Objects: car_adjacent_close (a "
    "vehicle very close in the next lane), lead_car (vehicle ahead in our lane), pedestrian, cones, overpass (bridge "
    "across the road), tunnel_exit (dark tunnel with a bright opening), sun (low sun in view), curve (road bends). "
    "light: day, low_sun (orange/low sun), night. lane_markings: faded if lines are missing or barely visible.")

TRIAGE_TOOL = {
    "name": "triage_event",
    "description": "Record the triage decision for one ADAS event.",
    "input_schema": {"type": "object", "properties": {
        "severity": {"type": "integer", "minimum": 1, "maximum": 4},
        "root_cause": {"type": "string", "enum": ROOT_CAUSES},
        "scenario": {"type": "string", "enum": SCEN},
        "narrative": {"type": "string", "description": "Two sentences for an engineer: what happened and why."}},
        "required": ["severity", "root_cause", "scenario", "narrative"]},
}
TRIAGE_SYSTEM = """You triage ADAS events (takeovers, system deactivations, safety interventions) for a fleet
forensics team. Use the telemetry summary and context; a scene description may be included.
Severity: 4 = imminent collision risk (very short time-to-collision or very hard braking at speed);
3 = significant safety relevance (hard braking, close object, or unexpected braking at speed);
2 = degraded system performance without immediate danger; 1 = benign (comfort or preference).
Root cause: perception (sensing/recognition limits or false detections), planning (vehicle control or path
decisions), other_road_user (another vehicle or pedestrian created the situation), environment (road works,
blocked sensors), driver_preference (driver simply chose to take over).
A hard brake with no lead object detected is a likely false detection."""

# v2: written after the v1 run (results/llm_eval_v1.json). v1 named the scenarios but never defined them
# (scenario accuracy 26% from telemetry alone) and called too few events severe (severe recall 42-54%).
# v2 defines each scenario and adds the five most similar *labelled* historical events as worked examples,
# retrieved by embedding similarity, so the model sees how this team grades severity.
SCENARIO_DEFS = """Scenarios:
- cut_in: another vehicle moves into our lane close ahead; hard braking with a lead object appearing.
- pedestrian_crossing: a person crosses in front, usually in towns; hard braking with an object ahead.
- faded_lanes: lane lines are worn or covered; lane-detection confidence drops and the system hands over.
- construction_zone: cones or road works; lane confidence dips, the car slows, the driver steers.
- sharp_curve: high lateral acceleration in a bend; the system hands over.
- sun_glare: low sun blinds the camera for a few seconds.
- phantom_brake: braking with no object ahead, often under bridges or overpasses (a false detection).
- sensor_blockage_snow: snow covers the sensors; visibility stays low for many seconds.
- comfort_override: the driver steers or takes over with no hazard in the signals or the scene.
- tunnel_exit_glare: a brief, near-total visibility drop at a tunnel exit."""
TRIAGE_SYSTEM_V2 = TRIAGE_SYSTEM + "\n\n" + SCENARIO_DEFS + """

You will also see the most similar past events with the labels this team's engineers gave them. Use them to
calibrate severity, but decide from this event's own evidence."""


def card(r: pd.Series, scene: dict | None = None) -> str:
    mph = 2.237
    lead = (f"lead object detected {r.lead_frac * 100:.0f}% of the window, closest {r.lead_min:.0f} m"
            if r.lead_frac > 0 else "no lead object detected")
    ttc = f"{r.ttc_min:.1f} s" if r.ttc_min < 50 else "n/a"
    lines = [f"Event type: {r.event_type}. Road: {r.road}. Weather: {r.weather}. Light: {r.light}. "
             f"Firmware: {r.firmware}.",
             f"Speed before: {r.speed_pre * mph:.0f} mph; speed drop {r.speed_drop * mph:.0f} mph; "
             f"max deceleration {r.max_decel:.1f} m/s^2; brake pressure peak {r.max_brake:.0f} bar.",
             f"Peak lateral acceleration {r.max_abs_lat:.1f} m/s^2; peak steering torque {r.max_abs_torque:.1f} Nm.",
             f"Lane-detection confidence min {r.lane_min:.2f} (drop {r.lane_drop:.2f}); camera visibility min "
             f"{r.vis_min:.2f}, below 0.45 for {r.vis_low_s:.1f} s.",
             f"{lead}; minimum time-to-collision {ttc}."]
    if scene:
        lines.append(f"Scene (from camera): {scene.get('summary', '')} Objects: {', '.join(scene.get('objects', [])) or 'none'}; "
                     f"lane markings {scene.get('lane_markings')}.")
    return "\n".join(lines)


def examples_text(rows: pd.DataFrame) -> str:
    parts = []
    for i, (_, e) in enumerate(rows.iterrows(), 1):
        parts.append(f"Example {i}:\n{card(e)}\nEngineer label: severity {int(e.severity)}, root cause "
                     f"{e.root_cause}, scenario {e.scenario}.")
    return "Similar past events (labelled):\n\n" + "\n\n".join(parts)


def frame_b64(r: pd.Series) -> str:
    buf = io.BytesIO()
    render(r, int(r["index"])).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def tool_call(client, model, system, tool, content, usage):
    resp = client.messages.create(model=model, max_tokens=800, system=system, tools=[tool],
                                  tool_choice={"type": "tool", "name": tool["name"]},
                                  messages=[{"role": "user", "content": content}])
    usage["calls"] += 1
    usage["input_tokens"] += resp.usage.input_tokens
    usage["output_tokens"] += resp.usage.output_tokens
    return next(b for b in resp.content if getattr(b, "type", "") == "tool_use").input


# ---------------- offline stand-ins and supervised baselines ----------------
class ClassicalVision:
    """Logistic regressions on hand-crafted image features, trained on labelled frames: the non-foundation-model
    baseline for the scene step (it needs labels; the VLM does not)."""

    def fit(self, img: np.ndarray, df: pd.DataFrame):
        self.m = {}
        y = {"weather": df.weather, "light": df.light, "lane_markings": df.lane_markings}
        objs = df.objects.fillna("").str.split("|")
        for o in OBJECTS:
            y[o] = objs.map(lambda s, o=o: o in s)
        for k, v in y.items():
            self.m[k] = LogisticRegression(max_iter=2000, C=1.0).fit(img, v)
        return self

    def describe(self, img_row: np.ndarray) -> dict:
        x = img_row[None]
        objs = [o for o in OBJECTS if self.m[o].predict(x)[0]]
        return {"weather": self.m["weather"].predict(x)[0], "light": self.m["light"].predict(x)[0],
                "lane_markings": self.m["lane_markings"].predict(x)[0], "objects": objs, "summary": ""}


def rules_triage(r: pd.Series) -> dict:
    if r.event_type == "safety":
        sev = 4 if (r.ttc_min < 1.5 or r.max_decel > 6) else 3
        rc, sc = ("perception", "phantom_brake") if r.lead_frac == 0 else ("other_road_user", "cut_in")
    elif r.vis_low_s > 3:
        sev, rc, sc = 2, "environment", "sensor_blockage_snow"
    elif r.vis_low_s > 0:
        sev, rc, sc = 2, "perception", "sun_glare"
    elif r.max_abs_lat > 2:
        sev, rc, sc = 2, "planning", "sharp_curve"
    elif r.lane_drop > 0.4:
        sev, rc, sc = 2, "perception", "faded_lanes"
    elif r.speed_drop > 2:
        sev, rc, sc = 2, "environment", "construction_zone"
    else:
        sev, rc, sc = 1, "driver_preference", "comfort_override"
    return {"severity": sev, "root_cause": rc, "scenario": sc, "narrative": ""}


def score(pred: list[dict], sub: pd.DataFrame) -> dict:
    sev = np.array([p["severity"] for p in pred])
    return {"severity_acc": accuracy_score(sub.severity, sev),
            "severity_within_1": float(np.mean(np.abs(sev - sub.severity.values) <= 1)),
            "severe_recall": float(np.mean(sev[sub.severity.values >= 3] >= 3)),
            "severe_precision": float(np.mean(sub.severity.values[sev >= 3] >= 3)) if (sev >= 3).any() else None,
            "root_cause_acc": accuracy_score(sub.root_cause, [p["root_cause"] for p in pred]),
            "scenario_acc": accuracy_score(sub.scenario, [p["scenario"] for p in pred]),
            "scenario_macro_f1": f1_score(sub.scenario, [p["scenario"] for p in pred], average="macro")}


def scene_score(pred: list[dict], sub: pd.DataFrame) -> dict:
    out = {k: accuracy_score(sub[k], [p[k] for p in pred]) for k in ("weather", "light", "lane_markings")}
    tobj = sub.objects.fillna("").str.split("|").map(set)
    tp = fp = fn = 0
    for p, t in zip(pred, tobj):
        ps = set(p["objects"]) - {""}
        t = t - {""}
        tp, fp, fn = tp + len(ps & t), fp + len(ps - t), fn + len(t - ps)
    out["objects_precision"], out["objects_recall"] = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=15)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--prompt", default="v2", choices=["v1", "v2"])
    ap.add_argument("--reuse-scenes", action="store_true",
                    help="reuse VLM scene outputs from results/llm_outputs*.jsonl instead of calling the VLM again")
    ap.add_argument("--k", type=int, default=5, help="retrieved labelled examples per event (v2)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    df = F.load()
    img = F.image_matrix(df)
    hist = df.day < 80
    rng = np.random.default_rng(0)
    labelled = hist & (rng.random(len(df)) < 0.25)  # the same labelled history the supervised models use
    sub = df[~hist].groupby("scenario").sample(a.per_scenario, random_state=0)
    vision = ClassicalVision().fit(img[labelled.values][:3000], df[labelled].iloc[:3000])
    X = F.matrix(df, "+image", img)
    sup = {t: clf().fit(X[labelled], df.loc[labelled, t]) for t in ("severity", "root_cause", "scenario")}

    # retrieval over the labelled history (v2 examples, and a k-NN vote baseline that uses them without an LLM)
    import faiss
    from collections import Counter
    from . import discover
    Z = discover.embed(df, F.shape_embedding(df), img)
    lab_idx = np.flatnonzero(labelled.values)
    index = faiss.IndexFlatIP(Z.shape[1])
    index.add(np.ascontiguousarray(Z[lab_idx]))
    _, nn = index.search(np.ascontiguousarray(Z[sub.index.values]), a.k)
    neighbours = {eid: df.iloc[lab_idx[row]] for eid, row in zip(sub.event_id, nn)}

    def knn_vote(eid):
        nb = neighbours[eid]
        return {"severity": int(round(nb.severity.median())), "root_cause": Counter(nb.root_cause).most_common(1)[0][0],
                "scenario": Counter(nb.scenario).most_common(1)[0][0]}

    cached = {}
    if a.reuse_scenes:
        for name in ("llm_outputs_v2.jsonl", "llm_outputs.jsonl", "llm_outputs_v1.jsonl"):
            f = ROOT / "results" / name
            if f.exists():
                for line in open(f, encoding="utf-8"):
                    rec = json.loads(line)
                    if "error" not in rec.get("scene", {}):
                        cached.setdefault(rec["event_id"], rec["scene"])
        print(f"reusing {len(cached)} cached scene descriptions")

    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0}
    if not a.dry_run:
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit('Set ANTHROPIC_API_KEY first (PowerShell: $env:ANTHROPIC_API_KEY = "sk-ant-...")')
        client = anthropic.Anthropic()
    (ROOT / "results").mkdir(exist_ok=True)
    tag = f"{a.prompt}-dryrun" if a.dry_run else a.prompt
    log = ROOT / "results" / f"llm_outputs_{tag}.jsonl"
    log.unlink(missing_ok=True)
    system = TRIAGE_SYSTEM if a.prompt == "v1" else TRIAGE_SYSTEM_V2
    scenes, tri_card, tri_scene, classical_scenes = [], [], [], []
    for _, r in sub.iterrows():
        cs = vision.describe(img[r.name])
        classical_scenes.append(cs)
        ex = "" if a.prompt == "v1" else "\n\n" + examples_text(neighbours[r.event_id])
        try:
            if a.dry_run:
                sc, t1, t2 = cs, rules_triage(r), rules_triage(r)
            else:
                sc = cached.get(r.event_id) or tool_call(client, a.model, SCENE_SYSTEM, SCENE_TOOL, [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": frame_b64(r)}},
                    {"type": "text", "text": "Describe this frame."}], usage)
                t1 = tool_call(client, a.model, system, TRIAGE_TOOL, "Event to triage:\n" + card(r) + ex, usage)
                t2 = tool_call(client, a.model, system, TRIAGE_TOOL, "Event to triage:\n" + card(r, sc) + ex, usage)
        except Exception as e:  # noqa: BLE001
            usage["errors"] += 1
            sc, t1, t2 = cs, rules_triage(r), rules_triage(r)
            sc["error"] = repr(e)
        scenes.append(sc)
        tri_card.append(t1)
        tri_scene.append(t2)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event_id": r.event_id, "prompt": a.prompt, "scene": sc, "triage_card": t1,
                                 "triage_card_scene": t2}, default=str) + "\n")
    Xs = X.loc[sub.index]
    supervised = [{"severity": int(s), "root_cause": rc, "scenario": sc} for s, rc, sc in
                  zip(sup["severity"].predict(Xs), sup["root_cause"].predict(Xs), sup["scenario"].predict(Xs))]
    res = {"model": "stub" if a.dry_run else a.model, "prompt": a.prompt, "events": len(sub),
           "per_scenario": a.per_scenario, "usage": usage, "scenes_reused": len(cached),
           "scene": {"vlm_zero_shot": scene_score(scenes, sub),
                     "classical_supervised_3000_labels": scene_score(classical_scenes, sub)},
           "triage": {"llm_card_only": score(tri_card, sub), "llm_card_plus_vlm_scene": score(tri_scene, sub),
                      "rules": score([rules_triage(r) for _, r in sub.iterrows()], sub),
                      f"knn_vote_{a.k}_labelled_neighbours": score([knn_vote(e) for e in sub.event_id], sub),
                      f"lightgbm_supervised_{int(labelled.sum())}_labels": score(supervised, sub)},
           "example_narratives": [{"event_id": sub.iloc[i].event_id, "true_scenario": sub.iloc[i].scenario,
                                   "true_severity": int(sub.iloc[i].severity), "narrative": tri_scene[i].get("narrative")}
                                  for i in range(0, len(sub), max(1, len(sub) // 6))]}
    name = f"llm_eval_{tag}.json"
    (ROOT / "results" / name).write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    t = res["triage"]
    print(f"{res['model']} prompt {a.prompt}: severity acc card {t['llm_card_only']['severity_acc']:.1%} / +scene "
          f"{t['llm_card_plus_vlm_scene']['severity_acc']:.1%}; scenario acc card {t['llm_card_only']['scenario_acc']:.1%} / "
          f"+scene {t['llm_card_plus_vlm_scene']['scenario_acc']:.1%}; {usage['calls']} calls, {usage['errors']} errors "
          f"-> results/{name}")


if __name__ == "__main__":
    main()
