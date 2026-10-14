"""Does adding safety-car risk change the recommended number of stops, and does it match winners better?

Three versions of the recommendation are compared with the race winner's number of stops:

  deterministic : the original simulator (empty track, green flag all race)
  expected      : average over random SC/VSC scenarios drawn from rates fitted on the OTHER 20 races only
                  (a fair, out-of-sample forecast: the team does not know what will happen)
  hindsight     : the race's REAL safety cars / VSCs are given to the simulator in advance.
                  This is NOT a forecast. It answers "would knowing the safety cars explain what the winner did?"

All versions use the same weekend-calibrated tyre parameters as the "fully calibrated" test in validation.py.
"""
import numpy as np
import pandas as pd

import safety_car as sc
from simulator import best_overall, optimise

N_SCEN = 1500


def _extra(p):
    """Exact deterministic optimum for each stop count, so the candidate pool always contains it."""
    out = []
    for k, o in optimise(p).items():
        out.append(sc.strategy_to_arrays(o["stints"]))
    return out


def _mean_by_stops(best):
    return {k: v["mean_s"] for k, v in best.items()}


def run_sc_validation(raw, params, val, n_scen=N_SCEN, window=sc.WINDOW, rate_scale=1.0, save_scale=1.0,
                      cache=None):
    """One row per race. `cache` (dict) lets repeated calls reuse laps, pit losses and candidate lists."""
    cache = {} if cache is None else cache
    if "laps" not in cache:
        cache["laps"] = sc.neutral_laps(raw)
        cache["pl"] = sc.neutral_pit_losses(raw, cache["laps"])
    laps, pl = cache["laps"], cache["pl"]
    rows = []
    for _, v in val.iterrows():
        race = v["Race"]
        p = params[race]
        if ("cands", race) not in cache:
            cache[("cands", race)] = sc.candidate_strategies(p, max_gap=25.0)
        cands = cache[("cands", race)]
        extra = _extra(p)
        model = sc.fit_neutral_model(raw, green_pit_loss=p.pit_loss, exclude_race=race, laps=laps, pit_losses=pl)
        model = model.scaled(rate_scale, save_scale)
        kinds = sc.sample_scenarios(model, p.n_laps, n_scen, seed=7)
        exp = sc.best_under_scenarios(p, kinds, model.save, window, extra=extra, cands=cands)
        hind = sc.best_under_scenarios(p, sc.actual_scenario(laps, race, p.n_laps), model.save, window,
                                       extra=extra, cands=cands)
        k_exp = min(exp, key=lambda k: exp[k]["mean_s"])
        k_hind = min(hind, key=lambda k: hind[k]["mean_s"])
        m_exp, m_hind = _mean_by_stops(exp), _mean_by_stops(hind)
        det_k, det_best = best_overall(optimise(p))
        rows.append({
            "Race": race, "winner_stops": int(v["winner_stops"]), "deterministic_stops": det_k,
            "expected_stops": k_exp, "hindsight_stops": k_hind,
            "p_any_neutralisation": float((kinds[:, 2:p.n_laps] > 0).any(axis=1).mean()),
            "actual_neutral_laps": int((laps[(laps["Race"] == race)]["kind"] > 0).sum()),
            **{f"expected_gap_{k}stop_s": m_exp[k] - m_exp[k_exp] for k in sorted(m_exp)},
            **{f"hindsight_gap_{k}stop_s": m_hind[k] - m_hind[k_hind] for k in sorted(m_hind)},
        })
    d = pd.DataFrame(rows)
    d["deterministic_match"] = d["deterministic_stops"] == d["winner_stops"]
    d["expected_match"] = d["expected_stops"] == d["winner_stops"]
    d["hindsight_match"] = d["hindsight_stops"] == d["winner_stops"]
    d["expected_changed"] = d["expected_stops"] != d["deterministic_stops"]
    return d


def summarise_sc(d):
    n = len(d)
    return pd.DataFrame([
        {"test": "Deterministic simulator (no safety cars)", "races": n, "correct": int(d["deterministic_match"].sum())},
        {"test": "Safety-car aware, forecast (rates from other races)", "races": n, "correct": int(d["expected_match"].sum())},
        {"test": "Safety-car aware, hindsight (real SC/VSC known)", "races": n, "correct": int(d["hindsight_match"].sum())},
    ]).assign(accuracy=lambda x: x["correct"] / x["races"])


def sc_sensitivity(raw, params, val, cache=None):
    """How much do the conclusions depend on the assumptions that are weakest in the data?"""
    cache = {} if cache is None else cache
    settings = {
        "baseline (window 10, rates as fitted)": dict(),
        "SC/VSC twice as frequent": dict(rate_scale=2.0),
        "SC/VSC half as frequent": dict(rate_scale=0.5),
        "stop saving 50% larger": dict(save_scale=1.5),
        "stop saving 50% smaller": dict(save_scale=0.5),
        "no reaction (window 0)": dict(window=0),
        "wider reaction (window 15)": dict(window=15),
    }
    rows = []
    for name, kw in settings.items():
        d = run_sc_validation(raw, params, val, n_scen=800, cache=cache, **kw)
        rows.append({"setting": name, "stops_changed_vs_deterministic": int(d["expected_changed"].sum()),
                     "matches_winner": int(d["expected_match"].sum()), "races": len(d)})
    return pd.DataFrame(rows)
