"""Sanity tests:  python -m pytest tests   (or just: python tests/test_core.py)

1. The strategy optimiser (dynamic programming) must equal an exhaustive brute-force search.
2. The degradation fit must recover known coefficients from synthetic data.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from degradation import fit_fe
from simulator import COMPOUNDS, MIN_STINT, RaceParams, evaluate, optimise


def brute_force(p: RaceParams, stops: int):
    N, best = p.n_laps, (np.inf, None)
    for cuts in itertools.combinations(range(1, N), stops):
        bounds = [0, *cuts, N]
        lens = [b - a for a, b in zip(bounds[:-1], bounds[1:])]
        for comps in itertools.product(COMPOUNDS, repeat=stops + 1):
            if len(set(comps)) < 2:
                continue
            if any(L < MIN_STINT or L > p.cap[c] for L, c in zip(lens, comps)):
                continue
            stints = [(c, a + 1, b) for c, a, b in zip(comps, bounds[:-1], bounds[1:])]
            t = evaluate(p, stints)
            if t < best[0] - 1e-9:
                best = (t, stints)
    return best


def test_optimiser_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(6):
        p = RaceParams(
            race="synthetic", n_laps=int(rng.integers(24, 31)),
            deg={"SOFT": rng.uniform(0.05, 0.15), "MEDIUM": rng.uniform(0.03, 0.10), "HARD": rng.uniform(0.02, 0.08)},
            offset={"SOFT": -rng.uniform(0.2, 0.8), "MEDIUM": -rng.uniform(0.0, 0.3), "HARD": 0.0},
            pit_loss=float(rng.uniform(18, 26)),
            cap={"SOFT": int(rng.integers(10, 16)), "MEDIUM": int(rng.integers(14, 22)), "HARD": int(rng.integers(18, 30))})
        opt = optimise(p, max_stops=2)
        for k in (1, 2):
            bf_t, _ = brute_force(p, k)
            if k in opt:
                assert abs(opt[k]["total_s"] - bf_t) < 1e-6, (k, opt[k]["total_s"], bf_t)
            else:
                assert not np.isfinite(bf_t)


def test_fixed_effects_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    rows = []
    for rd in range(60):
        base = rng.uniform(80, 100)                      # each driver-race has its own baseline
        age0 = 0
        for lap in range(2, 60):
            if lap % 20 == 2:
                age0, comp = lap, ["SOFT", "MEDIUM", "HARD"][(lap // 20 + rd) % 3]
            age = lap - age0 + 1
            deg = {"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.05}[comp]
            off = {"SOFT": -0.4, "MEDIUM": -0.15, "HARD": 0.0}[comp]
            rows.append((f"r|{rd}", lap, comp, age, base - 0.05 * lap + off + deg * age + rng.normal(0, 0.15)))
    df = pd.DataFrame(rows, columns=["RaceDriver", "LapNumber", "Compound", "TyreLife", "LapTimeSec"])
    c = fit_fe(df)
    assert abs(c["fuel_per_lap"] + 0.05) < 0.01
    assert abs(c["deg_SOFT"] - 0.09) < 0.015 and abs(c["deg_HARD"] - 0.05) < 0.01
    assert abs(c["offset_SOFT"] + 0.4) < 0.15


def _toy_params():
    return RaceParams(race="toy", n_laps=40,
                      deg={"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.05},
                      offset={"SOFT": -0.5, "MEDIUM": -0.2, "HARD": 0.0},
                      pit_loss=22.0, cap={"SOFT": 15, "MEDIUM": 25, "HARD": 35})


def test_safety_car_scorer_matches_simulator_when_all_green():
    """With no safety cars the scenario scorer must give exactly the base simulator's race time."""
    import safety_car as sc
    p = _toy_params()
    save = {sc.SC: 0.3, sc.VSC: 0.25}
    green = np.zeros((3, p.n_laps + 2), dtype=np.int8)
    for k, o in optimise(p, max_stops=2).items():
        comps, pits = sc.strategy_to_arrays(o["stints"])
        t = sc.scenario_times(p, comps, pits, green, save, window=sc.WINDOW)
        assert np.allclose(t, o["total_s"]), (k, t, o["total_s"])


def test_safety_car_makes_a_stop_cheaper_and_never_slower():
    """A safety car lap can only help a team that is free to move its stop, and the saving is bounded."""
    import safety_car as sc
    p = _toy_params()
    save = {sc.SC: 0.3, sc.VSC: 0.25}
    best = optimise(p, max_stops=1)[1]
    comps, pits = sc.strategy_to_arrays(best["stints"])
    green = np.zeros((1, p.n_laps + 2), dtype=np.int8)
    neutral = green.copy()
    neutral[0, pits[0] + 1: pits[0] + 4] = sc.SC               # 3 SC laps just after the planned stop
    t_green = sc.scenario_times(p, comps, pits, green, save)[0]
    t_sc = sc.scenario_times(p, comps, pits, neutral, save)[0]
    assert t_sc <= t_green + 1e-9
    assert t_green - t_sc <= save[sc.SC] * p.pit_loss + 1e-9    # cannot save more than the stop's discount
    t_still = sc.scenario_times(p, comps, pits, neutral, save, window=0)[0]
    assert t_still >= t_sc - 1e-9                               # reacting is never worse than not reacting


def test_sampled_scenarios_follow_the_fitted_rate():
    import safety_car as sc
    model = sc.NeutralModel(rate={sc.SC: 0.01, sc.VSC: 0.0}, durations={sc.SC: np.array([4]), sc.VSC: np.array([2])},
                            save={sc.SC: 0.3, sc.VSC: 0.25}, n_races=1, n_episodes={sc.SC: 1, sc.VSC: 0})
    k = sc.sample_scenarios(model, n_laps=60, n=4000, seed=3)
    episodes_per_race = ((k[:, 1:] == sc.SC) & (k[:, :-1] != sc.SC)).sum(axis=1).mean()
    assert abs(episodes_per_race - 0.6) < 0.1                    # 0.01 per lap * 60 laps
    assert (k == sc.VSC).sum() == 0


if __name__ == "__main__":
    test_optimiser_matches_brute_force(); print("optimiser == brute force: OK")
    test_fixed_effects_recovers_known_coefficients(); print("fixed-effects recovery: OK")
    test_safety_car_scorer_matches_simulator_when_all_green(); print("safety-car scorer == simulator when all green: OK")
    test_safety_car_makes_a_stop_cheaper_and_never_slower(); print("safety-car saving bounded and never slower: OK")
    test_sampled_scenarios_follow_the_fitted_rate(); print("sampled scenarios follow fitted rate: OK")
