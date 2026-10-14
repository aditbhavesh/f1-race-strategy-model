"""Download race lap data for one F1 season with FastF1 and save it as one CSV.

Usage:
    python download_laps.py            # 2024 season (default)
    python download_laps.py 2023       # another season

Output: laps_<year>.csv (attach this file back to Claude).
Notes:
  - First run takes 10-20 minutes (downloads ~24 races). A local "cache" folder
    is created so re-runs are fast. Do NOT commit that folder to GitHub.
  - If a race fails, it is skipped and listed at the end; the rest still saves.
"""
import sys
import pandas as pd
import fastf1

year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024

import os
os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache("cache")

KEEP = ["Driver", "Team", "LapNumber", "LapTime", "Stint", "Compound", "TyreLife",
        "FreshTyre", "PitInTime", "PitOutTime", "TrackStatus", "Position", "IsAccurate"]

schedule = fastf1.get_event_schedule(year, include_testing=False)
frames, failed = [], []

for _, ev in schedule.iterrows():
    rnd, name = int(ev["RoundNumber"]), ev["EventName"]
    print(f"[{rnd}] {name} ...", flush=True)
    try:
        s = fastf1.get_session(year, rnd, "R")
        s.load(telemetry=False, weather=False, messages=False)
        laps = s.laps
        cols = [c for c in KEEP if c in laps.columns]
        df = laps[cols].copy()
        df["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
        df["Round"] = rnd
        df["Race"] = name
        df["TotalLaps"] = int(laps["LapNumber"].max())
        frames.append(df)
    except Exception as e:  # keep going if one race fails
        print(f"   skipped: {e}")
        failed.append(f"{rnd} {name}")

if not frames:
    sys.exit("No races downloaded. Check your internet connection and fastf1 version.")

out = pd.concat(frames, ignore_index=True)
for c in ("PitInTime", "PitOutTime", "LapTime"):
    if c in out.columns:
        out[c] = out[c].dt.total_seconds()  # timedelta -> seconds for easy CSV use
fname = f"laps_{year}.csv"
out.to_csv(fname, index=False)
print(f"\nSaved {len(out)} laps from {len(frames)} races to {fname}")
if failed:
    print("Skipped:", ", ".join(failed))
