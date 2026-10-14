"""Run the whole project end to end:  python run_all.py   (add --quick to skip the slow safety-car sensitivity runs)

Reads  data/laps_2024.csv   (created by download_laps.py)
Writes outputs/*.csv and outputs/figures/*.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from data import clean_laps, load_raw, wet_races
from degradation import fit_per_race
from figures import make_all
from pace_models import cross_validate
from params import actual_strategies, build_params, neutralisation_share
from pit_loss import measure_stops, pit_loss_table
from simulator import best_overall, breakeven_pit_loss, optimise
from validation import run_validation, sensitivity, summarise
import safety_car as sc
from sc_validation import run_sc_validation, sc_sensitivity, summarise_sc

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main(quick=False):
    raw = load_raw()
    clean = clean_laps(raw)
    print(f"Laps: {len(raw):,} raw -> {len(clean):,} clean | wet races dropped: {', '.join(wet_races(raw))}")

    # 1. tyre degradation (random-effects, circuit-specific)
    summ, table = fit_per_race(clean)
    summ.to_csv(OUT / "degradation_typical.csv", index=False)
    table.to_csv(OUT / "degradation_by_circuit.csv", index=False)

    # 2. pace model comparison (leave-drivers-out cross-validation)
    metrics, preds = cross_validate(clean)
    metrics.to_csv(OUT / "pace_model_comparison.csv", index=False)
    print("\nPace model (unseen drivers):\n", metrics.round(3).to_string(index=False))

    # 3. pit loss + simulator inputs
    stops = measure_stops(raw, clean)
    pit_tbl = pit_loss_table(stops, sorted(clean["Race"].unique()))
    pit_tbl.to_csv(OUT / "pit_loss_by_circuit.csv", index=False)
    params = build_params(clean, raw, table, pit_tbl)

    # 4. optimal strategies per circuit
    rows, be_rows = [], []
    for race, p in params.items():
        opt = optimise(p)
        k, best = best_overall(opt)
        for stops_n, o in opt.items():
            rows.append({"Race": race, "laps": p.n_laps, "stops": stops_n, "delta_s": o["total_s"] - best["total_s"],
                         "is_best": stops_n == k,
                         "strategy": " | ".join(f"{c[0]}{s}-{e}" for c, s, e in o["stints"]),
                         "pit_laps": ",".join(map(str, o["pit_laps"]))})
        be_rows.append({"Race": race, "pit_loss_s": p.pit_loss, "breakeven_pit_loss_s": breakeven_pit_loss(p)})
    opt_df = pd.DataFrame(rows)
    opt_df.to_csv(OUT / "optimal_strategies.csv", index=False)
    breakeven = pd.DataFrame(be_rows)
    breakeven.to_csv(OUT / "pit_loss_breakeven.csv", index=False)

    # 5. validation against what winners did + sensitivity
    val = run_validation(clean, raw, params, neutralisation_share(raw))
    val.to_csv(OUT / "validation_by_race.csv", index=False)
    val_summary = summarise(val)
    val_summary.to_csv(OUT / "validation_summary.csv", index=False)
    print("\nDoes the simulator predict the winner's number of stops?\n", val_summary.round(2).to_string(index=False))
    sens = sensitivity(params)
    sens.to_csv(OUT / "sensitivity.csv", index=False)
    print("\nRecommendation unchanged under input perturbations:",
          f"stops {sens['stops_unchanged'].mean():.0%}, stops+tyre order {sens['sequence_unchanged'].mean():.0%}")

    # 6. safety-car / VSC risk (Monte Carlo)
    laps_kind = sc.neutral_laps(raw)
    neutral_losses = sc.neutral_pit_losses(raw, laps_kind)
    green_loss = float(stops["PitLoss_s"].median())
    desc = sc.describe_model(raw, green_loss, laps_kind, neutral_losses)
    desc.to_csv(OUT / "safety_car_model.csv", index=False)
    print("\nSafety car / VSC model:\n", desc.round(2).to_string(index=False))
    cache = {"laps": laps_kind, "pl": neutral_losses}
    sc_val = run_sc_validation(raw, params, val, n_scen=300 if quick else 1500, cache=cache)
    sc_val.to_csv(OUT / "safety_car_validation_by_race.csv", index=False)
    sc_summary = summarise_sc(sc_val)
    sc_summary.to_csv(OUT / "safety_car_validation_summary.csv", index=False)
    print("\nDoes safety-car awareness change the answer?\n", sc_summary.round(2).to_string(index=False))
    print("Recommended stop count changed in", int(sc_val["expected_changed"].sum()), "of", len(sc_val), "races")
    if not quick:
        sc_sens = sc_sensitivity(raw, params, val, cache=cache)
        sc_sens.to_csv(OUT / "safety_car_sensitivity.csv", index=False)
        print("\nSafety-car assumptions:\n", sc_sens.to_string(index=False))

    # 7. charts
    make_all({"summ": summ, "table": table, "metrics": metrics, "preds": preds, "opt": opt_df,
              "winner_stops": val.set_index("Race")["winner_stops"].to_dict(), "val_summary": val_summary,
              "sens": sens, "breakeven": breakeven, "sc_val": sc_val,
              "green_losses": stops["PitLoss_s"], "neutral_losses": neutral_losses,
              "fuel": float(summ.set_index("coef").loc["fuel_per_lap", "typical_mu"])})
    print(f"\nDone. Tables in {OUT}, charts in {OUT / 'figures'}")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
