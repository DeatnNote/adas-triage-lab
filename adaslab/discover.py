"""Scenario discovery (HDBSCAN on event embeddings) and similar-event retrieval (FAISS)."""
from __future__ import annotations

import faiss
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure
from sklearn.preprocessing import StandardScaler

from .features import NUM


def embed(df: pd.DataFrame, shape: np.ndarray, img: np.ndarray | None = None) -> np.ndarray:
    parts = [StandardScaler().fit_transform(np.log1p(np.abs(df[NUM].values)) * np.sign(df[NUM].values)),
             StandardScaler().fit_transform(shape)]
    if img is not None:
        parts.append(StandardScaler().fit_transform(PCA(8, random_state=0).fit_transform(img)))
    Z = np.hstack(parts).astype("float32")
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def clusters(Z: np.ndarray, truth: pd.Series, rare: str = "tunnel_exit_glare", min_cluster_size: int = 40) -> dict:
    lab = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10, copy=True).fit_predict(Z)
    km = KMeans(truth.nunique(), n_init=10, random_state=0).fit_predict(Z)
    clustered = lab >= 0
    out = {"hdbscan_clusters": int(lab.max() + 1), "noise_share": float((~clustered).mean()),
           "hdbscan_ari_clustered": float(adjusted_rand_score(truth[clustered], lab[clustered])),
           "kmeans_ari": float(adjusted_rand_score(truth, km))}
    h, c, _ = homogeneity_completeness_v_measure(truth[clustered], lab[clustered])
    hk, ck, _ = homogeneity_completeness_v_measure(truth, km)
    out.update({"hdbscan_homogeneity": float(h), "hdbscan_completeness": float(c),
                "kmeans_homogeneity": float(hk), "kmeans_completeness": float(ck)})
    # did a cluster isolate the rare edge case?
    is_rare = (truth == rare).values
    best = None
    for c in range(lab.max() + 1):
        m = lab == c
        prec, rec = is_rare[m].mean(), (m & is_rare).sum() / max(is_rare.sum(), 1)
        if best is None or rec * prec > best[1] * best[2]:
            best = (c, prec, rec, int(m.sum()))
    out["rare_family"] = {"events": int(is_rare.sum()), "share_of_fleet_events": float(is_rare.mean()),
                          "best_cluster_size": best[3], "precision": float(best[1]), "recall": float(best[2])}
    km_best = max(((is_rare[km == c].mean(), (is_rare & (km == c)).sum() / is_rare.sum()) for c in range(km.max() + 1)),
                  key=lambda t: t[0] * t[1])
    out["rare_family_kmeans"] = {"precision": float(km_best[0]), "recall": float(km_best[1])}
    out["labels"] = lab
    return out


def retrieval_precision(Z: np.ndarray, truth: pd.Series, k: int = 10, n: int = 2000, seed: int = 0) -> dict:
    index = faiss.IndexFlatIP(Z.shape[1])
    index.add(Z)
    rng = np.random.default_rng(seed)
    q = rng.choice(len(Z), n, replace=False)
    _, I = index.search(Z[q], k + 1)
    t = truth.values
    prec = [(t[I[i, 1:]] == t[qi]).mean() for i, qi in enumerate(q)]
    rare = [p for p, qi in zip(prec, q) if t[qi] == "tunnel_exit_glare"]
    return {"precision_at_10": float(np.mean(prec)), "queries": n,
            "precision_at_10_rare": float(np.mean(rare)) if rare else None}
