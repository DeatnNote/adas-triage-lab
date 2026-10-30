"""Bronze -> silver -> gold on Delta tables, with the heavy lifting in PySpark.

* bronze: raw drives, events (VINs pseudonymised) and 10 Hz telemetry windows as array columns
* silver: one row per event with features computed in Spark from the arrays (higher-order functions)
* gold:   vehicle-week exposure and event counts, the base for KPIs and the firmware study

Delta tables are written with delta-rs (`deltalake`); Spark reads each table's current snapshot files. On a
Databricks/Synapse cluster with delta-spark the reads become `spark.read.format("delta")` and nothing else changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from .sim import CHANNELS, HZ, WIN, vin_hash

ROOT = Path(__file__).resolve().parents[1]
LAKE = ROOT / "data" / "lake"


def spark_session():
    from pyspark.sql import SparkSession
    s = (SparkSession.builder.master("local[4]").appName("adaslab")
         .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "8")
         .config("spark.driver.memory", "3g").getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    return s


def read_delta(spark, name: str, lake: Path = LAKE):
    files = DeltaTable(str(lake / name)).file_uris()
    return spark.read.parquet(*files)


def bronze(raw: Path = ROOT / "data" / "raw", lake: Path = LAKE):
    drives = pd.read_parquet(raw / "drives.parquet")
    events = pd.read_parquet(raw / "events.parquet")
    W = np.load(raw / "windows.npy")
    for df in (drives, events):
        df["vin"] = df.vin.map(vin_hash)  # no raw VINs past this point
    write_deltalake(str(lake / "bronze_drives"), drives, mode="overwrite")
    write_deltalake(str(lake / "bronze_events"), events, mode="overwrite")
    cols = {"event_id": pa.array(events.event_id)}
    for c, name in enumerate(CHANNELS):
        cols[name] = pa.array(list(W[:, c, :].astype("float32")), type=pa.list_(pa.float32()))
    write_deltalake(str(lake / "bronze_windows"), pa.table(cols), mode="overwrite")
    return len(drives), len(events)


PRE = WIN * HZ  # index of t = 0


def silver(spark, lake: Path = LAKE) -> pd.DataFrame:
    from pyspark.sql import functions as F
    w = read_delta(spark, "bronze_windows", lake)
    ev = read_delta(spark, "bronze_events", lake)

    def finite(c):
        return F.filter(F.col(c), lambda x: ~F.isnan(x))

    def mean(arr):
        return F.aggregate(arr, F.lit(0.0), lambda acc, x: acc + x) / F.greatest(F.size(arr), F.lit(1))

    pre = lambda c: F.slice(F.col(c), 1, PRE)  # noqa: E731
    post = lambda c: F.slice(F.col(c), PRE + 1, PRE)  # noqa: E731
    feats = w.select(
        "event_id",
        mean(F.slice(F.col("speed"), PRE - 20, 20)).alias("speed_pre"),
        (mean(F.slice(F.col("speed"), PRE - 20, 20)) - F.array_min("speed")).alias("speed_drop"),
        (-F.array_min("long_accel")).alias("max_decel"),
        F.array_max("brake_pressure").alias("max_brake"),
        F.array_max(F.transform("lat_accel", F.abs)).alias("max_abs_lat"),
        F.array_max(F.transform("steer_torque", F.abs)).alias("max_abs_torque"),
        F.array_min("lane_conf").alias("lane_min"),
        (mean(pre("lane_conf")) - mean(post("lane_conf"))).alias("lane_drop"),
        (F.size(finite("lead_dist")) / F.lit(2 * PRE)).alias("lead_frac"),
        F.coalesce(F.array_min(finite("lead_dist")), F.lit(200.0)).alias("lead_min"),
        F.coalesce(F.array_min(finite("ttc")), F.lit(99.0)).alias("ttc_min"),
        F.array_min("cam_visibility").alias("vis_min"),
        (mean(pre("cam_visibility")) - F.array_min("cam_visibility")).alias("vis_drop"),
        (F.size(F.filter("cam_visibility", lambda x: x < 0.45)) / F.lit(HZ)).alias("vis_low_s"),
        (mean(post("cam_visibility")) - mean(pre("cam_visibility"))).alias("vis_shift"),
    )
    out = ev.join(feats, "event_id").toPandas()
    write_deltalake(str(lake / "silver_event_features"), out, mode="overwrite")
    return out


def silver_local(lake: Path = LAKE) -> pd.DataFrame:
    """The same silver features in NumPy, for laptops without Java (and as a parity check on the Spark job)."""
    w = DeltaTable(str(lake / "bronze_windows")).to_pyarrow_table()
    ev = DeltaTable(str(lake / "bronze_events")).to_pandas()
    A = {c: np.stack(w.column(c).to_numpy(zero_copy_only=False)).astype("float64") for c in CHANNELS}
    pre, post = slice(0, PRE), slice(PRE, 2 * PRE)
    lead, ttc = A["lead_dist"], A["ttc"]
    with np.errstate(all="ignore"):
        f = pd.DataFrame({
            "event_id": w.column("event_id").to_pylist(),
            "speed_pre": A["speed"][:, PRE - 21:PRE - 1].mean(1),
            "speed_drop": A["speed"][:, PRE - 21:PRE - 1].mean(1) - A["speed"].min(1),
            "max_decel": -A["long_accel"].min(1), "max_brake": A["brake_pressure"].max(1),
            "max_abs_lat": np.abs(A["lat_accel"]).max(1), "max_abs_torque": np.abs(A["steer_torque"]).max(1),
            "lane_min": A["lane_conf"].min(1),
            "lane_drop": A["lane_conf"][:, pre].mean(1) - A["lane_conf"][:, post].mean(1),
            "lead_frac": np.isfinite(lead).sum(1) / (2 * PRE),
            "lead_min": np.where(np.isfinite(lead).any(1), np.nanmin(np.where(np.isfinite(lead), lead, np.inf), 1), 200.0),
            "ttc_min": np.where(np.isfinite(ttc).any(1), np.nanmin(np.where(np.isfinite(ttc), ttc, np.inf), 1), 99.0),
            "vis_min": A["cam_visibility"].min(1),
            "vis_drop": A["cam_visibility"][:, pre].mean(1) - A["cam_visibility"].min(1),
            "vis_low_s": (A["cam_visibility"] < 0.45).sum(1) / HZ,
            "vis_shift": A["cam_visibility"][:, post].mean(1) - A["cam_visibility"][:, pre].mean(1),
        })
    out = ev.merge(f, on="event_id")
    write_deltalake(str(lake / "silver_event_features"), out, mode="overwrite")
    return out


def gold(spark, lake: Path = LAKE) -> pd.DataFrame:
    """Vehicle-week exposure (engaged miles) and event counts by event type."""
    from pyspark.sql import functions as F
    d = read_delta(spark, "bronze_drives", lake).withColumn("week", (F.col("day") / 7).cast("int"))
    e = read_delta(spark, "bronze_events", lake).withColumn("week", (F.col("day") / 7).cast("int"))
    expo = d.groupBy("vin", "week").agg(F.sum("engaged_miles").alias("engaged_miles"),
                                         F.max(F.col("firmware") == "4.3.0").alias("new_fw_any"),
                                         F.min(F.col("firmware") == "4.3.0").alias("new_fw_all"))
    cnt = e.groupBy("vin", "week").pivot("event_type", ["takeover", "deactivation", "safety"]).count()
    g = expo.join(cnt, ["vin", "week"], "left").fillna(0).toPandas()
    write_deltalake(str(lake / "gold_vehicle_week"), g, mode="overwrite")
    return g


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-spark", action="store_true", help="compute silver features with NumPy (no Java needed)")
    a = ap.parse_args()
    n_d, n_e = bronze()
    if a.no_spark:
        s = silver_local()
        print(f"bronze: {n_d} drives, {n_e} events; silver (NumPy): {s.shape}; gold skipped")
        return
    spark = spark_session()
    s = silver(spark)
    g = gold(spark)
    spark.stop()
    print(f"bronze: {n_d} drives, {n_e} events; silver: {s.shape}; gold: {g.shape}")


if __name__ == "__main__":
    main()
