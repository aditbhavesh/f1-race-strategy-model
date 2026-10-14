"""Estimate the time lost by a pit stop at each circuit (green-flag stops only).

Pit loss = (in-lap + out-lap) - 2 * (the driver's normal lap pace around the stop).
It includes the slow pit-lane drive, the stationary time and the cold-tyre out-lap.
Stops made under safety car / VSC are much cheaper, so they are excluded - they would
make pitting look artificially attractive.
"""
import numpy as np
import pandas as pd

from data import wet_races

PRIOR_N = 15  # circuits with few stops are pulled towards the all-circuit median


def measure_stops(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    raw = raw[~raw["Race"].isin(wet_races(raw))].sort_values(["Race", "Driver", "LapNumber"])
    ref_by = {k: v for k, v in clean.groupby(["Race", "Driver"])}
    rows = []
    for (race, drv), x in raw.groupby(["Race", "Driver"]):
        x = x.set_index("LapNumber")
        ref_all = ref_by.get((race, drv))
        if ref_all is None:
            continue
        for ln in x.index[x["PitInTime"].notna()]:
            out_lap = ln + 1
            if out_lap not in x.index:
                continue
            a, b = x.loc[ln], x.loc[out_lap]
            if a["TrackStatus"] != 1 or b["TrackStatus"] != 1:
                continue
            if pd.isna(a["LapTimeSec"]) or pd.isna(b["LapTimeSec"]):
                continue
            ref = ref_all[ref_all["LapNumber"].between(ln - 5, out_lap + 5)]["LapTimeSec"]
            if len(ref) < 4:
                continue
            rows.append((race, drv, int(ln), a["LapTimeSec"] + b["LapTimeSec"] - 2 * ref.median()))
    return pd.DataFrame(rows, columns=["Race", "Driver", "Lap", "PitLoss_s"])


def pit_loss_table(stops: pd.DataFrame, races: list) -> pd.DataFrame:
    """Per-circuit median, shrunk towards the global median when few stops were measured."""
    gmed = stops["PitLoss_s"].median()
    g = stops.groupby("Race")["PitLoss_s"].agg(n_stops="size", median_s="median")
    t = pd.DataFrame({"Race": races}).merge(g, on="Race", how="left")
    t["n_stops"] = t["n_stops"].fillna(0)
    t["median_s"] = t["median_s"].fillna(gmed)
    t["pit_loss_s"] = (t["n_stops"] * t["median_s"] + PRIOR_N * gmed) / (t["n_stops"] + PRIOR_N)
    t["global_median_s"] = gmed
    return t
