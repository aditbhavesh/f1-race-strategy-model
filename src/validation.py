"""Does the strategy simulator recommend what winning teams actually did?

Two tests, both scored on the *number of stops* of the race winner (the cleanest, least ambiguous
outcome to compare - exact pit laps depend on safety cars and traffic that we do not model):

  A. "Calibrated"  - circuit parameters estimated from that weekend's own race laps
                     (what a team knows after practice / early in the race). Tyre-age caps use
                     the longest stints seen, so this test is optimistic.
  B. "Blind"       - leave-one-race-out: parameters for the race come ONLY from the other 20 circuits
                     (typical degradation, typical pit loss, typical stint caps). A genuine out-of-sample test.
  C. "Calibrated, blind caps" - like A, but tyre-age caps come from the other circuits only. This removes
                     the optimism in A (caps in A are taken from the stints teams actually ran).

Baselines: always predict the most common winner strategy (1 stop).
"""
import numpy as np
import pandas as pd

from data import DRY
from degradation import fit_per_race
from params import _ordered_offsets, actual_strategies, build_params, stint_caps
from pit_loss import measure_stops, pit_loss_table
from simulator import RaceParams, best_overall, optimise


def winners(raw):
    a = actual_strategies(raw)
    return a[a["final_pos"] == 1].set_index("Race")


def blind_params(clean, raw, race, stops_all):
    """Parameters for `race` estimated without looking at that race at all."""
    c_tr = clean[clean["Race"] != race]
    summ, _ = fit_per_race(c_tr)
    mu = summ.set_index("coef")["typical_mu"]
    s, m = _ordered_offsets(mu["offset_SOFT"], mu["offset_MEDIUM"])
    others = stops_all[stops_all["Race"] != race]["PitLoss_s"].median()
    r = raw[~raw["Race"].eq(race) & raw["Compound"].isin(DRY)]
    caps = {c: int(max(12, round(r.loc[r["Compound"] == c, "TyreLife"].quantile(0.99)))) for c in DRY}
    return RaceParams(race=race, n_laps=int(clean.loc[clean["Race"] == race, "TotalLaps"].iat[0]),
                      deg={c: max(float(mu[f"deg_{c}"]), 0.0) for c in DRY},
                      offset={"SOFT": s, "MEDIUM": m, "HARD": 0.0}, pit_loss=float(others), cap=caps)


def run_validation(clean, raw, params_calibrated, neutral):
    win = winners(raw)
    stops_all = measure_stops(raw, clean)
    rows = []
    for race, p in params_calibrated.items():
        if race not in win.index:
            continue
        a = optimise(p); ka, ba = best_overall(a)
        bp = blind_params(clean, raw, race, stops_all)
        b = optimise(bp); kb, bb = best_overall(b)
        c_ = optimise(p.replace(cap=bp.cap)); kc, bc = best_overall(c_)
        w = win.loc[race]
        rows.append({"Race": race, "laps": p.n_laps, "winner": w["Driver"], "winner_stops": int(w["stops"]),
                     "winner_sequence": w["sequence"],
                     "calibrated_stops": ka, "calibrated_sequence": "-".join(s[0][0] for s in ba["stints"]),
                     "calibrated_blindcaps_stops": kc, "calibrated_blindcaps_sequence": "-".join(s[0][0] for s in bc["stints"]),
                     "blind_stops": kb, "blind_sequence": "-".join(s[0][0] for s in bb["stints"]),
                     "neutralised_share": float(neutral.get(race, np.nan))})
    v = pd.DataFrame(rows)
    v["calibrated_match"] = v["calibrated_stops"] == v["winner_stops"]
    v["blind_match"] = v["blind_stops"] == v["winner_stops"]
    v["calibrated_blindcaps_match"] = v["calibrated_blindcaps_stops"] == v["winner_stops"]
    v["always_one_stop_match"] = v["winner_stops"] == 1
    return v


def summarise(v):
    return pd.DataFrame([
        {"test": "Always predict 1 stop (baseline)", "races": len(v), "correct": int(v["always_one_stop_match"].sum())},
        {"test": "Simulator, blind (leave-one-race-out)", "races": len(v), "correct": int(v["blind_match"].sum())},
        {"test": "Simulator, calibrated tyre data + blind stint caps", "races": len(v), "correct": int(v["calibrated_blindcaps_match"].sum())},
        {"test": "Simulator, fully calibrated to the weekend", "races": len(v), "correct": int(v["calibrated_match"].sum())},
    ]).assign(accuracy=lambda d: d["correct"] / d["races"])


def sensitivity(params_calibrated, scenarios=None):
    """Does the recommended number of stops survive plausible errors in the model inputs?"""
    scen = scenarios or {
        "deg x0.8": lambda p: p.replace(deg={c: v * 0.8 for c, v in p.deg.items()}),
        "deg x1.2": lambda p: p.replace(deg={c: v * 1.2 for c, v in p.deg.items()}),
        "pit loss -3s": lambda p: p.replace(pit_loss=p.pit_loss - 3),
        "pit loss +3s": lambda p: p.replace(pit_loss=p.pit_loss + 3),
        "soft/medium 0.3s faster": lambda p: p.replace(offset={"SOFT": p.offset["SOFT"] - 0.3, "MEDIUM": p.offset["MEDIUM"] - 0.3, "HARD": 0.0}),
        "soft/medium 0.3s slower": lambda p: p.replace(offset={"SOFT": p.offset["SOFT"] + 0.3, "MEDIUM": p.offset["MEDIUM"] + 0.3, "HARD": 0.0}),
        "tyre caps -15%": lambda p: p.replace(cap={c: max(12, int(v * 0.85)) for c, v in p.cap.items()}),
        "tyre caps +15%": lambda p: p.replace(cap={c: int(v * 1.15) for c, v in p.cap.items()}),
    }
    rows = []
    for race, p in params_calibrated.items():
        k0, b0 = best_overall(optimise(p))
        seq0 = "-".join(s[0][0] for s in b0["stints"])
        for name, f in scen.items():
            o = optimise(f(p))
            if not o:
                continue
            k1, b1 = best_overall(o)
            rows.append({"Race": race, "scenario": name, "base_stops": k0, "new_stops": k1,
                         "stops_unchanged": k0 == k1,
                         "sequence_unchanged": "-".join(s[0][0] for s in b1["stints"]) == seq0})
    return pd.DataFrame(rows)
