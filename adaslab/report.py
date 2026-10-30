"""Figures -> results/firmware.png, results/triage_queue.png, results/active_learning.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
INK, MUTED, GRID = "#1c1c1a", "#62625c", "#e4e3de"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
NAMES = {"sharp_curve": "Deactivations on sharp curves", "phantom_brake": "Phantom braking (no object ahead)",
         "cut_in": "Cut-ins (unaffected)", "faded_lanes": "Faded-lane deactivations (unaffected)",
         "sensor_blockage_snow": "Snow sensor blockage (unaffected)"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def fig_firmware(d):
    fw = d["firmware"]
    keys = list(NAMES)
    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    for i, k in enumerate(keys):
        r = fw[k]
        lo, hi = r["did_model_labels"]["ci"]
        ax.plot([lo, hi], [i, i], color=S1, lw=2.5, solid_capstyle="round")
        ax.plot(r["did_model_labels"]["rate_ratio"], i, "o", color=S1, ms=7,
                label="Difference-in-differences (95% CI)" if i == 0 else None)
        ax.plot(min(r["naive_model_labels"], 2.55), i, "s", color=S2, ms=7,
                label="Naive before/after" if i == 0 else None)
        ax.plot(r["true_rate_ratio"], i, "|", color=INK, ms=16, mew=2.5, label="True effect" if i == 0 else None)
        if r["naive_model_labels"] > 2.6:
            ax.text(2.63, i, f"{r['naive_model_labels']:.1f}x", va="center", fontsize=8, color=S2)
    ax.axvline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([NAMES[k] for k in keys], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0.4, 2.9)
    ax.set_xlabel("Event rate with firmware 4.3.0 / rate with 4.2.1 (per engaged mile)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower right", bbox_to_anchor=(1, 0.99))
    ax.set_title("Firmware impact: staggered rollout across a winter", loc="left", fontsize=10.5, color=INK, pad=22)
    style(ax)
    fig.tight_layout()
    fig.savefig(RES / "firmware.png", dpi=160)


def fig_queue(d):
    q = d["queue"]
    rows = [("chronological", "Chronological"), ("safety events first, by deceleration", "Rule: safety events first"),
            ("model: expected severity", "Model: expected severity"), ("oracle (true severity)", "Perfect ranking")]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    vals = [q[k]["hours_to_review_90pct_sev4"] for k, _ in rows]
    cols = [GRID, S2, S1, MUTED]
    ax.barh([r[1] for r in rows], vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 8, i, f"{v:.0f} h  ·  {q[rows[i][0]]['sev4_reviewed_same_day'] * 100:.0f}% seen same day",
                va="center", fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.55)
    ax.set_xlabel(f"Engineer hours to review 90% of severity-4 events (all {q['chronological']['events_in_period']:,} "
                  f"events = {q['chronological']['hours_to_review_all']:.0f} h)", color=MUTED, fontsize=9)
    ax.set_title("Daily triage queue, 40 test days", loc="left", fontsize=10.5, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(RES / "triage_queue.png", dpi=160)


def fig_al(d):
    al = d["active_learning"]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for k, c, lab in (("random", S2, "Random"), ("uncertainty", S1, "Uncertainty sampling")):
        ax.plot([r["labels"] for r in al[k]], [r["auc"] for r in al[k]], color=c, lw=2, marker="o", ms=4, label=lab)
    ax.set_xlabel("Engineer labels", color=MUTED, fontsize=9)
    ax.set_ylabel("Severe-event AUC (test days)", color=MUTED, fontsize=9)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Active learning did not beat random labelling here", loc="left", fontsize=10.5, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(RES / "active_learning.png", dpi=160)


def main():
    d = json.loads((RES / "study.json").read_text(encoding="utf-8"))
    fig_firmware(d)
    fig_queue(d)
    fig_al(d)
    print("figures written to", RES)


if __name__ == "__main__":
    main()
