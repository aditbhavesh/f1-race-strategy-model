"""Pit-stop strategy simulator.

For one circuit we know (from the degradation model) how many seconds each compound loses per
lap of tyre age, how much faster/slower each compound is when fresh, and how long a stop costs.
The simulator finds the race strategy (compounds + pit laps) with the lowest total time using
dynamic programming, under realistic rules:

  - dry race rule: at least two *different* compounds must be used
  - a stint must be at least MIN_STINT laps
  - a stint cannot be longer than the longest stint we actually *observed* on that compound at
    that circuit (we do not extrapolate degradation beyond the data - no "tyre cliff" is modelled)

Lap-time model for lap n on a tyre set of age a (a = 1 on the first lap of the set):
    t = base + fuel * n + offset[compound] + deg[compound] * a
`base` and the fuel term are identical for every strategy, so only offset, deg and pit loss decide.

NOT modelled: safety cars, traffic / track position, tyre-warm-up, driver-specific tyre management,
used (scrubbed) tyres, weather. See README "Limitations".
"""
from dataclasses import dataclass, field
import numpy as np

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
MIN_STINT = 5
INF = 1e12


@dataclass
class RaceParams:
    race: str
    n_laps: int
    deg: dict          # s per lap of tyre age
    offset: dict       # s, fresh-tyre pace vs HARD (negative = faster)
    pit_loss: float    # s
    cap: dict          # max stint length (laps) per compound
    meta: dict = field(default_factory=dict)

    def replace(self, **kw):
        d = dict(race=self.race, n_laps=self.n_laps, deg=dict(self.deg), offset=dict(self.offset),
                 pit_loss=self.pit_loss, cap=dict(self.cap), meta=dict(self.meta))
        d.update(kw)
        return RaceParams(**d)


def stint_cost(p: RaceParams, c: str, start: int, end: int) -> float:
    """Total (strategy-relevant) time for laps start..end on one set of compound c."""
    L = end - start + 1
    return L * p.offset[c] + p.deg[c] * L * (L + 1) / 2


def _cost_matrix(p: RaceParams, c: str):
    """M[e, e2] = cost of a stint covering laps e+1..e2 on compound c (inf if not allowed)."""
    N = p.n_laps
    M = np.full((N + 1, N + 1), INF)
    for e in range(N):
        lo, hi = e + MIN_STINT, min(N, e + p.cap[c])
        if lo > N:
            continue
        L = np.arange(lo - e, hi - e + 1)
        M[e, e + L] = L * p.offset[c] + p.deg[c] * L * (L + 1) / 2
    return M


def optimise(p: RaceParams, max_stops: int = 3):
    """Best strategy for each stop count 1..max_stops.

    Returns {stops: dict(total_s, stints=[(compound, start, end)], pit_laps=[...])}, total_s being
    the strategy-relevant time (includes pit losses); compare values *between* strategies only.
    """
    N = p.n_laps
    mats = {c: _cost_matrix(p, c) for c in COMPOUNDS}
    bit = {c: 1 << i for i, c in enumerate(COMPOUNDS)}
    # dp[k][mask] = vector over lap e (finished laps) of best cost with k stints used
    dp = [dict() for _ in range(max_stops + 2)]
    par = [dict() for _ in range(max_stops + 2)]
    zero = np.full(N + 1, INF); zero[0] = 0.0
    dp[0][0] = zero
    for k in range(max_stops + 1):
        for mask, vec in list(dp[k].items()):
            for c in COMPOUNDS:
                cand = vec[:, None] + mats[c] + (p.pit_loss if k >= 1 else 0.0)
                best = cand.min(axis=0)
                arg = cand.argmin(axis=0)
                nm = mask | bit[c]
                key = (nm)
                if key not in dp[k + 1]:
                    dp[k + 1][key] = np.full(N + 1, INF); par[k + 1][key] = {}
                better = best < dp[k + 1][key]
                for e2 in np.where(better)[0]:
                    par[k + 1][key][e2] = (mask, c, int(arg[e2]))
                dp[k + 1][key] = np.where(better, best, dp[k + 1][key])
    out = {}
    for k in range(2, max_stops + 2):               # k stints = k-1 stops
        cands = [(v[N], m) for m, v in dp[k].items() if bin(m).count("1") >= 2 and v[N] < INF / 2]
        if not cands:
            continue
        total, mask = min(cands)
        stints, e2, m, kk = [], N, mask, k
        while kk > 0:
            pm, c, e = par[kk][m][e2]
            stints.append((c, e + 1, e2)); e2, m, kk = e, pm, kk - 1
        stints.reverse()
        out[k - 1] = {"total_s": float(total), "stints": stints,
                      "pit_laps": [s[2] for s in stints[:-1]]}
    return out


def evaluate(p: RaceParams, stints) -> float:
    """Model cost of any given strategy (no rule checks) - used to score the strategies teams really ran."""
    return sum(stint_cost(p, c, s, e) for c, s, e in stints) + p.pit_loss * (len(stints) - 1)


def best_overall(opt: dict):
    k = min(opt, key=lambda k: opt[k]["total_s"])
    return k, opt[k]


def breakeven_pit_loss(p: RaceParams, lo=5.0, hi=45.0):
    """Pit loss at which the best 1-stop and best 2-stop strategies take equal time (bisection)."""
    def diff(P):
        o = optimise(p.replace(pit_loss=P), max_stops=2)
        if 1 not in o or 2 not in o:
            return np.nan
        return o[1]["total_s"] - o[2]["total_s"]    # >0 -> 2-stop faster
    a, b = diff(lo), diff(hi)
    if np.isnan(a) or np.isnan(b) or a * b > 0:
        return np.nan
    for _ in range(25):
        mid = (lo + hi) / 2
        if diff(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
