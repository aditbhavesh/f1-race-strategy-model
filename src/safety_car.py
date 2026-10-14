"""Safety-car and VSC risk for the strategy simulator (Monte Carlo).

The base simulator (simulator.py) assumes an empty track under green flags all race. In reality a safety
car (SC) or virtual safety car (VSC) makes a pit stop cheaper, and teams react to it. This module:

  1. finds every SC / VSC period in the 2024 lap data (a lap counts as neutralised when at least half of
     the drivers' laps carry the SC or VSC flag),
  2. measures how much cheaper a stop is under each (time lost relative to cars that did not stop on the same laps),
  3. fits simple arrival rates and durations, and draws random race scenarios from them,
  4. scores any strategy by its AVERAGE race time over those scenarios, where a stop may be moved up to
     `window` laps to coincide with a neutralisation if that is cheaper (a simple model of a team reacting).

Simplifications (also listed in the README): arrivals are pooled across circuits (no circuit-specific risk),
start laps are uniform, tyres do not wear during neutralised laps, only the two adjacent stints are
considered when a stop is moved, and track position / bunching-up of the field is not modelled.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import wet_races
from simulator import COMPOUNDS, MIN_STINT, RaceParams, stint_cost

GREEN, VSC, SC = 0, 1, 2
WINDOW = 10          # laps a stop may be moved to catch a neutralisation
SAVE_MIN, SAVE_MAX = 0.0, 0.9


# ---------------------------------------------------------------------------------------------- data
def neutral_laps(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per race lap: kind in {GREEN, VSC, SC}. Red-flag laps count as SC."""
    ts = raw["TrackStatus"].astype(str)
    d = raw.assign(_sc=ts.str.contains("[45]"), _vsc=ts.str.contains("[67]"))
    d["_vsc"] = d["_vsc"] & ~d["_sc"]
    g = d.groupby(["Race", "LapNumber"]).agg(sc=("_sc", "mean"), vsc=("_vsc", "mean")).reset_index()
    g["kind"] = np.where(g["sc"] >= 0.5, SC, np.where(g["vsc"] >= 0.5, VSC, GREEN))
    g["LapNumber"] = g["LapNumber"].astype(int)
    return g[["Race", "LapNumber", "kind"]]


def episodes(laps: pd.DataFrame) -> pd.DataFrame:
    """Contiguous neutralised periods: Race, kind, start lap, duration (laps)."""
    rows = []
    for race, x in laps.sort_values(["Race", "LapNumber"]).groupby("Race"):
        k, l = x["kind"].to_numpy(), x["LapNumber"].to_numpy()
        i = 0
        while i < len(k):
            if k[i] != GREEN:
                j = i
                while j + 1 < len(k) and k[j + 1] == k[i] and l[j + 1] == l[j] + 1:
                    j += 1
                rows.append((race, int(k[i]), int(l[i]), j - i + 1))
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(rows, columns=["Race", "kind", "start", "dur"])


def neutral_pit_losses(raw: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """Time lost by stops whose in-lap was neutralised, relative to the field.

    loss = (in-lap + out-lap of the stopper) - median(in-lap + out-lap of cars that did not stop on those two laps).
    Comparing with the field removes the slow SC lap times themselves.
    """
    kind = laps.set_index(["Race", "LapNumber"])["kind"]
    rr = raw[~raw["Race"].isin(wet_races(raw))].copy()
    rr["_pit"] = rr["PitInTime"].notna()
    rows = []
    for race, x in rr.groupby("Race"):
        piv = x.pivot_table(index="LapNumber", columns="Driver", values="LapTimeSec")
        pit = (x.pivot_table(index="LapNumber", columns="Driver", values="_pit", aggfunc="max")
               .reindex_like(piv).fillna(False).astype(bool))
        busy = pit | pit.shift(1, fill_value=False)
        for lap_i, drv_i in zip(*np.where(pit.to_numpy())):
            lap = int(piv.index[lap_i])
            k = int(kind.get((race, lap), GREEN))
            if k == GREEN or lap_i + 1 >= len(piv):
                continue
            t = piv.iloc[lap_i, drv_i] + piv.iloc[lap_i + 1, drv_i]
            if np.isnan(t):
                continue
            free = ~busy.iloc[lap_i].to_numpy() & ~busy.iloc[lap_i + 1].to_numpy()
            f = (piv.iloc[lap_i].to_numpy() + piv.iloc[lap_i + 1].to_numpy())[free]
            f = f[~np.isnan(f)]
            if len(f) >= 5:
                rows.append((race, piv.columns[drv_i], lap, k, t - np.median(f)))
    return pd.DataFrame(rows, columns=["Race", "Driver", "Lap", "kind", "loss_s"])


# ------------------------------------------------------------------------------------------- the model
@dataclass
class NeutralModel:
    rate: dict          # expected episodes per race lap, {SC: .., VSC: ..}
    durations: dict     # observed durations (laps), {SC: array, VSC: array}
    save: dict          # fraction of the green-flag pit loss saved, {SC: .., VSC: ..}
    n_races: int
    n_episodes: dict

    def scaled(self, rate_scale=1.0, save_scale=1.0) -> "NeutralModel":
        return NeutralModel({k: v * rate_scale for k, v in self.rate.items()}, self.durations,
                            {k: float(np.clip(v * save_scale, SAVE_MIN, SAVE_MAX)) for k, v in self.save.items()},
                            self.n_races, self.n_episodes)


def fit_neutral_model(raw: pd.DataFrame, green_pit_loss: float, exclude_race: str = None,
                      laps: pd.DataFrame = None, pit_losses: pd.DataFrame = None) -> NeutralModel:
    """Fit on the dry races (optionally leaving one out, for honest out-of-sample tests).

    `laps` / `pit_losses` can be passed in (computed once) to avoid recomputing them for every left-out race.
    """
    laps = neutral_laps(raw) if laps is None else laps
    dry = laps[~laps["Race"].isin(wet_races(raw))]
    if exclude_race is not None:
        dry = dry[dry["Race"] != exclude_race]
    ep = episodes(dry)
    n_laps = len(dry)
    rate, dur, n_ep = {}, {}, {}
    for k in (SC, VSC):
        d = ep.loc[ep["kind"] == k, "dur"].to_numpy()
        n_ep[k] = len(d)
        rate[k] = len(d) / n_laps
        dur[k] = d if len(d) else np.array([2 if k == VSC else 4])
    pl = neutral_pit_losses(raw, laps) if pit_losses is None else pit_losses
    if exclude_race is not None:
        pl = pl[pl["Race"] != exclude_race]
    save = {}
    for k in (SC, VSC):
        x = pl.loc[pl["kind"] == k, "loss_s"]
        save[k] = float(np.clip(1 - x.median() / green_pit_loss, SAVE_MIN, SAVE_MAX)) if len(x) >= 3 else 0.25
    return NeutralModel(rate, dur, save, dry["Race"].nunique(), n_ep)


def sample_scenarios(model: NeutralModel, n_laps: int, n: int, seed: int = 0) -> np.ndarray:
    """(n, n_laps + 2) array of lap kinds (GREEN / VSC / SC). Index = lap number (0 and n_laps+1 unused)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n_laps + 2), dtype=np.int8)
    for kind in (VSC, SC):                                    # SC written last so it overrides VSC
        counts = rng.poisson(model.rate[kind] * n_laps, size=n)
        for s in np.where(counts > 0)[0]:
            for _ in range(counts[s]):
                start = int(rng.integers(2, n_laps))          # uniform over laps 2..n_laps-1
                dur = int(rng.choice(model.durations[kind]))
                out[s, start:min(start + dur, n_laps + 1)] = kind
    return out


def actual_scenario(laps: pd.DataFrame, race: str, n_laps: int) -> np.ndarray:
    """The race's real neutralisations as a single scenario, shape (1, n_laps + 2)."""
    out = np.zeros((1, n_laps + 2), dtype=np.int8)
    x = laps[laps["Race"] == race]
    out[0, x["LapNumber"].to_numpy()] = x["kind"].to_numpy()
    return out


# ----------------------------------------------------------------------------------------- scoring
def scenario_times(p: RaceParams, comps, pits, kinds: np.ndarray, save: dict, window: int = WINDOW) -> np.ndarray:
    """Strategy-relevant race time (same units as simulator.evaluate) in every scenario.

    comps: compound per stint; pits: last lap of every stint but the last (planned stop laps).
    Each stop may move by up to `window` laps if that lowers cost, given the stint-length rules.
    Stops are placed in order, so two stops can never share a lap.
    """
    S = kinds.shape[0]
    N, k = p.n_laps, len(pits)
    pit_cost_by_kind = np.array([p.pit_loss, p.pit_loss * (1 - save[VSC]), p.pit_loss * (1 - save[SC])])
    d = np.arange(-window, window + 1)
    prev = np.zeros(S, dtype=int)
    total = np.zeros(S)
    for j in range(k):
        b = pits[j] + d                                             # candidate stop laps, shape (D,)
        ok_lap = (b >= 1) & (b <= N - 1)
        bb = np.clip(b, 1, N - 1)
        nxt = pits[j + 1] if j + 1 < k else N
        Lp = bb[None, :] - prev[:, None]
        Ln = nxt - bb[None, :]
        feas = ok_lap[None, :] & (Lp >= MIN_STINT) & (Lp <= p.cap[comps[j]]) & (Ln >= MIN_STINT) & (Ln <= p.cap[comps[j + 1]])
        pc = pit_cost_by_kind[kinds[:, bb]]
        cost = (stint_cost_L(p, comps[j], Lp) + stint_cost_L(p, comps[j + 1], Ln) + pc
                + 1e-6 * np.abs(d)[None, :])                         # tie-break: stay near the plan
        cost = np.where(feas, cost, np.inf)
        choice = np.argmin(cost, axis=1)
        none = ~np.isfinite(cost[np.arange(S), choice])
        choice = np.where(none, window, choice)                      # no legal option: keep the planned lap
        chosen_b = bb[choice]
        Lp_c = chosen_b - prev
        total += stint_cost_L(p, comps[j], Lp_c) + pit_cost_by_kind[kinds[np.arange(S), chosen_b]]
        prev = chosen_b
    total += stint_cost_L(p, comps[k], N - prev)
    return total


def stint_cost_L(p: RaceParams, c: str, L):
    L = np.asarray(L, dtype=float)
    return L * p.offset[c] + p.deg[c] * L * (L + 1) / 2


def strategy_to_arrays(stints):
    return [s[0] for s in stints], [s[2] for s in stints[:-1]]


# -------------------------------------------------------------------------------------- candidates
def candidate_strategies(p: RaceParams, max_gap: float, per_stop: int = 250, grid3: int = 2):
    """Plausible strategies (compounds + pit laps) within `max_gap` seconds of the best deterministic one.

    1- and 2-stop plans use every pit lap; 3-stop plans use every `grid3`-th lap (plus the exact optimum is
    always added by the caller) to keep the search small.
    """
    N = p.n_laps
    cands = {1: [], 2: [], 3: []}
    import itertools

    def ok(c, L):
        return (L >= MIN_STINT) & (L <= p.cap[c])

    for comps in itertools.product(COMPOUNDS, repeat=2):
        if len(set(comps)) < 2:
            continue
        a = np.arange(MIN_STINT, N - MIN_STINT + 1)
        m = ok(comps[0], a) & ok(comps[1], N - a)
        for x in a[m]:
            cands[1].append((float(stint_cost_L(p, comps[0], x) + stint_cost_L(p, comps[1], N - x) + p.pit_loss), comps, (int(x),)))
    for comps in itertools.product(COMPOUNDS, repeat=3):
        if len(set(comps)) < 2:
            continue
        a = np.arange(MIN_STINT, N - 2 * MIN_STINT + 1)
        A, B = np.meshgrid(a, np.arange(2 * MIN_STINT, N - MIN_STINT + 1), indexing="ij")
        L0, L1, L2 = A, B - A, N - B
        m = ok(comps[0], L0) & ok(comps[1], L1) & ok(comps[2], L2)
        cost = stint_cost_L(p, comps[0], L0) + stint_cost_L(p, comps[1], L1) + stint_cost_L(p, comps[2], L2) + 2 * p.pit_loss
        for x, y, c in zip(A[m], B[m], cost[m]):
            cands[2].append((float(c), comps, (int(x), int(y))))
    g = np.arange(MIN_STINT, N - 3 * MIN_STINT + 1, grid3)
    for comps in itertools.product(COMPOUNDS, repeat=4):
        if len(set(comps)) < 2:
            continue
        for x in g:
            for y in np.arange(x + MIN_STINT, N - 2 * MIN_STINT + 1, grid3):
                for z in np.arange(y + MIN_STINT, N - MIN_STINT + 1, grid3):
                    L = (x, y - x, z - y, N - z)
                    if all(ok(c, l) for c, l in zip(comps, L)):
                        cands[3].append((float(sum(stint_cost_L(p, c, l) for c, l in zip(comps, L)) + 3 * p.pit_loss),
                                         comps, (int(x), int(y), int(z))))
    best = min(c[0] for v in cands.values() for c in v)
    out = []
    for k, v in cands.items():
        v = sorted(v, key=lambda t: t[0])
        out += [(k, c[1], c[2], c[0]) for c in v if c[0] <= best + max_gap][:per_stop]
    return out


def best_under_scenarios(p: RaceParams, kinds: np.ndarray, save: dict, window: int = WINDOW,
                         max_gap: float = 25.0, extra=(), cands=None):
    """Score candidate strategies by mean race time over `kinds`; return the best overall and per stop count.

    `extra` = additional (comps, pits) to include, e.g. the exact deterministic optimum per stop count.
    """
    if cands is None:
        cands = candidate_strategies(p, max_gap)
    cand = [(len(pits), comps, pits, det) for k, comps, pits, det in cands]
    cand += [(len(pits), tuple(comps), tuple(pits), np.nan) for comps, pits in extra]
    best = {}
    for k, comps, pits, det in cand:
        t = scenario_times(p, comps, pits, kinds, save, window)
        m = float(t.mean())
        if k not in best or m < best[k]["mean_s"]:
            best[k] = {"stops": k, "mean_s": m, "comps": comps, "pits": pits, "det_s": det}
    return best


def describe_model(raw: pd.DataFrame, green_pit_loss: float, laps=None, pit_losses=None, n_boot=2000, seed=0) -> pd.DataFrame:
    """Plain-language summary of what was measured, with a bootstrap interval on the stop saving."""
    laps = neutral_laps(raw) if laps is None else laps
    pl = neutral_pit_losses(raw, laps) if pit_losses is None else pit_losses
    m = fit_neutral_model(raw, green_pit_loss, laps=laps, pit_losses=pl)
    rng = np.random.default_rng(seed)
    rows = []
    for k, name in ((SC, "Safety car / red flag"), (VSC, "Virtual safety car")):
        x = pl.loc[pl["kind"] == k, "loss_s"].to_numpy()
        boots = [1 - np.median(rng.choice(x, len(x))) / green_pit_loss for _ in range(n_boot)] if len(x) else [np.nan]
        rows.append({"kind": name, "episodes": m.n_episodes[k], "dry_races": m.n_races,
                     "episodes_per_race": m.rate[k] * len(laps[~laps["Race"].isin(wet_races(raw))]) / m.n_races,
                     "median_duration_laps": float(np.median(m.durations[k])),
                     "stops_measured": len(x), "median_stop_loss_s": float(np.median(x)) if len(x) else np.nan,
                     "green_stop_loss_s": green_pit_loss, "saving_share": m.save[k],
                     "saving_p5": float(np.nanpercentile(boots, 5)), "saving_p95": float(np.nanpercentile(boots, 95))})
    return pd.DataFrame(rows)
