"""Triage: event classifiers, a daily review queue, and an active-learning loop with simulated engineer labels."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

MINUTES_PER_REVIEW = 6


def clf(n=300):
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.05, num_leaves=31, min_child_samples=10,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)


def evaluate_classifiers(df, X, train, test) -> dict:
    out = {}
    m = clf().fit(X[train], df.scenario[train])
    ps = m.predict(X[test])
    out["scenario_macro_f1"] = f1_score(df.scenario[test], ps, average="macro")
    out["scenario_per_class_f1"] = dict(zip(sorted(df.scenario.unique()),
                                            f1_score(df.scenario[test], ps, average=None, labels=sorted(df.scenario.unique()))))
    m = clf().fit(X[train], df.root_cause[train])
    out["root_cause_acc"] = accuracy_score(df.root_cause[test], m.predict(X[test]))
    m = clf().fit(X[train], df.severity[train])
    p = m.predict(X[test])
    out["severity_acc"] = accuracy_score(df.severity[test], p)
    out["severity_within_1"] = float(np.mean(np.abs(p - df.severity[test]) <= 1))
    m = clf().fit(X[train], df.severity[train] >= 3)
    out["severe_auc"] = roc_auc_score(df.severity[test] >= 3, m.predict_proba(X[test])[:, 1])
    out["predicted_scenario"] = pd.Series(ps, index=df.index[test])
    return out


def daily_queue(df: pd.DataFrame, score: pd.Series, test_mask, budget_per_day: int) -> dict:
    """Engineers review `budget_per_day` events a day, highest score first. Share of severity-4 events reviewed
    on the day they happened, and the engineer hours needed to see 90% of them."""
    d = df[test_mask].assign(score=score[test_mask].values)
    seen = 0
    for _, g in d.groupby("day"):
        top = g.nlargest(budget_per_day, "score")
        seen += int((top.severity == 4).sum())
    total4 = int((d.severity == 4).sum())
    # global ordering within each day: reviews needed to reach 90% of sev-4, summed over days
    need = 0
    for _, g in d.groupby("day"):
        s4 = (g.sort_values("score", ascending=False).severity == 4).values
        if s4.sum() == 0:
            continue
        k = int(np.searchsorted(np.cumsum(s4), np.ceil(0.9 * s4.sum())) + 1)
        need += k
    return {"sev4_reviewed_same_day": seen / max(total4, 1), "sev4_total": total4,
            "hours_to_review_90pct_sev4": need * MINUTES_PER_REVIEW / 60,
            "events_in_period": len(d), "hours_to_review_all": len(d) * MINUTES_PER_REVIEW / 60}


def active_learning(df, X, pool_mask, test_mask, start=100, step=100, rounds=14, strategy="uncertainty", seed=0):
    """Severe-event classifier trained on labels requested from engineers. Returns AUC after each round."""
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(pool_mask)
    labelled = list(rng.choice(pool, start, replace=False))
    y = (df.severity >= 3).values
    curve = []
    for r in range(rounds + 1):
        m = clf(200).fit(X.iloc[labelled], y[labelled])
        p_test = m.predict_proba(X[test_mask])[:, 1]
        curve.append({"labels": len(labelled), "auc": roc_auc_score(y[test_mask], p_test)})
        if r == rounds:
            break
        rest = np.setdiff1d(pool, labelled)
        if strategy == "random":
            pick = rng.choice(rest, step, replace=False)
        else:  # closest to the decision boundary, with a little randomness to avoid near-duplicates
            p = m.predict_proba(X.iloc[rest])[:, 1]
            u = -np.abs(p - 0.5) + rng.normal(0, 0.02, len(p))
            pick = rest[np.argsort(-u)[:step]]
        labelled += list(pick)
    return curve
