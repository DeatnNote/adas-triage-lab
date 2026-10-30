"""Firmware impact: staggered over-the-air rollout analysed as a difference-in-differences (Poisson, two-way
fixed effects), compared with a naive before/after; plus a sequential monitor for a regression."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from deltalake import DeltaTable
from scipy import stats

from .sim import FW_NEW


def vehicle_week(events: pd.DataFrame, label_col: str, scenario: str, lake) -> pd.DataFrame:
    """Engaged miles per vehicle-week (from the lake) with counts of one scenario (true or model-labelled)."""
    d = DeltaTable(str(lake / "bronze_drives")).to_pandas()
    d["week"] = d.day // 7
    expo = d.groupby(["vin", "week"]).agg(miles=("engaged_miles", "sum"),
                                          new=("firmware", lambda s: (s == FW_NEW).mean())).reset_index()
    e = events[events[label_col] == scenario].assign(week=lambda x: x.day // 7)
    cnt = e.groupby(["vin", "week"]).size().rename("n").reset_index()
    g = expo.merge(cnt, on=["vin", "week"], how="left").fillna({"n": 0})
    return g[(g.new == 0) | (g.new == 1)]  # drop the week each car switched mid-week


def did_poisson(g: pd.DataFrame, waves: pd.Series) -> dict:
    """log E[n] = log(miles) + rollout-wave FE + week FE + beta * new_firmware; SEs clustered by vehicle.
    exp(beta) = rate ratio. (Wave FE rather than vehicle FE: cars were assigned to waves at random, and it keeps
    the design matrix small enough for a laptop.)"""
    g = g.assign(wave=g.vin.map(waves).astype(str))
    X = pd.get_dummies(g[["wave", "week"]].astype(str), drop_first=True, dtype=float)
    X.insert(0, "new_fw", g.new.values.astype(float))
    X = sm.add_constant(X)
    m = sm.GLM(g.n.values, X, family=sm.families.Poisson(), offset=np.log(g.miles.values)).fit(cov_type="cluster",
                                                                                         cov_kwds={"groups": pd.factorize(g.vin)[0]})
    b, se = m.params["new_fw"], m.bse["new_fw"]
    return {"rate_ratio": float(np.exp(b)), "ci": [float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))]}


def naive_before_after(g: pd.DataFrame, first_update_week: int, last_update_week: int) -> dict:
    """Fleet rate before any car updated vs after every wave: what a dashboard would show."""
    pre = g[g.week < first_update_week]
    post = g[g.week >= last_update_week + 1]
    r = (post.n.sum() / post.miles.sum()) / (pre.n.sum() / pre.miles.sum())
    return {"rate_ratio": float(r)}


def sequential_monitor(events: pd.DataFrame, drives: pd.DataFrame, label_col: str, scenario: str,
                       start_day: int, z_alert: float = 3.0) -> dict:
    """Each day after the first wave, compare updated vs not-yet-updated cars on that day's cumulative data
    (same calendar days, so season cancels). Alert when the rate-ratio z-score passes `z_alert` (a strict
    threshold because it is checked every day)."""
    e = events[(events[label_col] == scenario) & (events.day >= start_day)]
    d = drives[drives.day >= start_day]
    for day in range(start_day, int(d.day.max()) + 1):
        de, dd = e[e.day <= day], d[d.day <= day]
        n1, n0 = (de.firmware == FW_NEW).sum(), (de.firmware != FW_NEW).sum()
        m1 = dd.loc[dd.firmware == FW_NEW, "engaged_miles"].sum()
        m0 = dd.loc[dd.firmware != FW_NEW, "engaged_miles"].sum()
        if n1 < 5 or n0 < 5:
            continue
        lr = np.log((n1 / m1) / (n0 / m0))
        z = lr / np.sqrt(1 / n1 + 1 / n0)
        if abs(z) > z_alert:
            return {"alert_day": int(day), "days_after_first_update": int(day - start_day),
                    "rate_ratio_at_alert": float(np.exp(lr)), "direction": "increase" if z > 0 else "decrease",
                    "p_one_check": float(2 * stats.norm.sf(abs(z)))}
    return {"alert_day": None}
