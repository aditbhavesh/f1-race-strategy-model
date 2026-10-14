"""Turn fitted tables into simulator inputs, and extract the strategies teams actually ran."""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from data import DRY, wet_races
from simulator import RaceParams

CAP_QUANTILE = 0.99      # longest stint we allow = 99th percentile of observed tyre age
MIN_COMPOUND_LAPS = 40   # fewer laps than this on a compound at a circuit -> use the all-circuit cap


def _ordered_offsets(soft: float, medium: float) -> tuple:
    """Enforce the physical ordering  SOFT <= MEDIUM <= HARD(=0)  (pool-adjacent-violators).

    Raw compound offsets are the noisiest estimates in the project (see README), and sometimes come out
    in the wrong order. Isotonic regression is the smallest correction that restores the ordering.
    """
    ir = IsotonicRegression(increasing=True)
    y = ir.fit_transform([0, 1, 2], [soft, medium, 0.0], sample_weight=[1, 1, 1e6])
    return float(y[0] - y[2]), float(y[1] - y[2])


def stint_caps(raw: pd.DataFrame) -> dict:
    r = raw[~raw["Race"].isin(wet_races(raw)) & raw["Compound"].isin(DRY)]
    glob = r.groupby("Compound")["TyreLife"].quantile(CAP_QUANTILE).to_dict()
    caps = {}
    for race, x in r.groupby("Race"):
        caps[race] = {}
        for c in DRY:
            y = x.loc[x["Compound"] == c, "TyreLife"]
            v = y.quantile(CAP_QUANTILE) if len(y) >= MIN_COMPOUND_LAPS else glob[c]
            caps[race][c] = int(max(12, round(v)))
    return caps


def build_params(clean: pd.DataFrame, raw: pd.DataFrame, deg_table: pd.DataFrame, pit_tbl: pd.DataFrame) -> dict:
    caps = stint_caps(raw)
    out = {}
    for _, row in deg_table.iterrows():
        race = row["Race"]
        s, m = _ordered_offsets(row["offset_SOFT"], row["offset_MEDIUM"])
        n_laps = int(clean.loc[clean["Race"] == race, "TotalLaps"].iat[0])
        out[race] = RaceParams(
            race=race, n_laps=n_laps,
            deg={c: max(float(row[f"deg_{c}"]), 0.0) for c in DRY},     # tyres never get faster with age
            offset={"SOFT": s, "MEDIUM": m, "HARD": 0.0},
            pit_loss=float(pit_tbl.loc[pit_tbl["Race"] == race, "pit_loss_s"].iat[0]),
            cap=caps[race])
    return out


def actual_strategies(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per classified finisher: final position, stops, and the stints they ran."""
    raw = raw[~raw["Race"].isin(wet_races(raw))]
    rows = []
    for (race, drv), x in raw.groupby(["Race", "Driver"]):
        total = int(x["TotalLaps"].iat[0])
        if x["LapNumber"].max() < total - 1:      # retired / not classified
            continue
        x = x.sort_values("LapNumber")
        final_pos = x["Position"].dropna().iat[-1] if x["Position"].notna().any() else np.nan
        stints = []
        for _, s in x.groupby("Stint"):
            comp = s["Compound"].mode().iat[0]
            stints.append((comp, int(s["LapNumber"].min()), int(s["LapNumber"].max())))
        rows.append({"Race": race, "Driver": drv, "Team": x["Team"].iat[0], "final_pos": final_pos,
                     "stops": len(stints) - 1, "stints": stints,
                     "sequence": "-".join(c[0] for c, _, _ in stints)})
    return pd.DataFrame(rows)


def neutralisation_share(raw: pd.DataFrame) -> pd.Series:
    """Share of race laps run under safety car / red flag / VSC (TrackStatus codes 4,5,6,7)."""
    s = raw["TrackStatus"].astype(str).str.contains("[4567]")
    return s.groupby(raw["Race"]).mean()
