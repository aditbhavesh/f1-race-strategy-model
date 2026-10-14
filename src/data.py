"""Load and clean the 2024 race lap data.

Key idea: strategy models should be fitted on *representative racing laps* only.
We therefore drop laps that are not a fair measure of car + tyre pace:
  - lap 1 (standing start), in-laps and out-laps (pit lane), FastF1 "inaccurate" laps
  - any lap not run fully under green flag (safety car, VSC, yellow flags, red flag)
  - wet-weather laps and entire races that were mostly wet/mixed
  - laps far slower than the car's typical pace (traffic, mistakes, lock-ups)
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "laps_2024.csv"
DRY = ["SOFT", "MEDIUM", "HARD"]
WET_RACE_SHARE = 0.10   # races with >10% inter/wet laps are excluded entirely
SLOW_CAP = 1.07         # drop laps >7% slower than the driver's median green lap in that race


def load_raw(path=DATA) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["Driver"] = d["Driver"].astype(str)
    d["Race"] = d["Race"].astype(str)
    return d


def wet_races(d: pd.DataFrame) -> list:
    share = d.groupby("Race")["Compound"].apply(lambda c: c.isin(["INTERMEDIATE", "WET"]).mean())
    return share[share > WET_RACE_SHARE].index.tolist()


def clean_laps(d: pd.DataFrame) -> pd.DataFrame:
    """Return representative dry, green-flag racing laps with modelling features."""
    d = d.copy()
    d = d[~d["Race"].isin(wet_races(d))]
    d = d[d["Compound"].isin(DRY)]
    d = d[d["LapTimeSec"].notna()]
    d = d[(d["LapNumber"] > 1) & d["IsAccurate"] & (d["TrackStatus"] == 1)]
    d = d[d["PitInTime"].isna() & d["PitOutTime"].isna()]          # drop in/out laps explicitly
    d = d[d["TyreLife"].notna()]

    d["RaceDriver"] = d["Race"] + "|" + d["Driver"]
    med = d.groupby("RaceDriver")["LapTimeSec"].transform("median")
    d = d[d["LapTimeSec"] <= SLOW_CAP * med]

    # drivers with very few laps in a race give unreliable fixed effects
    n = d.groupby("RaceDriver")["LapTimeSec"].transform("size")
    d = d[n >= 15].copy()

    d["LapFrac"] = d["LapNumber"] / d["TotalLaps"]
    d["RelPace"] = d["LapTimeSec"] - d.groupby("RaceDriver")["LapTimeSec"].transform("mean")
    return d.reset_index(drop=True)


def stints(d_raw: pd.DataFrame) -> pd.DataFrame:
    """One row per driver stint in the full raw data (used for actual strategies)."""
    g = (d_raw[d_raw["Compound"].notna()]
         .groupby(["Race", "Driver", "Stint"])
         .agg(Compound=("Compound", lambda s: s.mode().iat[0]),
              StartLap=("LapNumber", "min"), EndLap=("LapNumber", "max"),
              Team=("Team", "first"), TotalLaps=("TotalLaps", "first"))
         .reset_index())
    g["Laps"] = g["EndLap"] - g["StartLap"] + 1
    return g


if __name__ == "__main__":
    raw = load_raw()
    clean = clean_laps(raw)
    print("raw laps:", len(raw), "| races excluded as wet:", wet_races(raw))
    print("clean laps:", len(clean), "| races:", clean["Race"].nunique())
    print(clean.groupby("Compound")["LapTimeSec"].agg(["size", "mean"]).round(2))
