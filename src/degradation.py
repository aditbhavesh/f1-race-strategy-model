"""Tyre degradation + fuel-burn model using driver-race fixed effects.

Lap time of a driver in a race is modelled as

    LapTime = (driver-in-this-race baseline)            <- absorbs car, driver, circuit, fuel-at-start
              + fuel * LapNumber                         <- car gets lighter / track rubbers in (negative)
              + compound offset (vs HARD)
              + deg_compound * TyreLife                  <- seconds lost per extra lap of tyre age

"Fixed effects" means every driver in every race gets his own baseline, so we only learn from
*changes within one driver's race*: how his lap time moves as tyres age and fuel burns off.
That removes the (huge) car and circuit differences and isolates the thing strategy cares about.
"""
import numpy as np
import pandas as pd

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def design(df: pd.DataFrame, quad: bool = False, compounds=COMPOUNDS):
    """Return (X, names). Reference compound for offsets is HARD."""
    cols, names = [df["LapNumber"].to_numpy(float)], ["fuel_per_lap"]
    for c in ("SOFT", "MEDIUM"):
        cols.append((df["Compound"] == c).to_numpy(float)); names.append(f"offset_{c}")
    for c in compounds:
        m = (df["Compound"] == c).to_numpy(float)
        cols.append(m * df["TyreLife"].to_numpy(float)); names.append(f"deg_{c}")
        if quad:
            cols.append(m * df["TyreLife"].to_numpy(float) ** 2); names.append(f"deg2_{c}")
    return np.column_stack(cols), names


def _demean(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    a = a.astype(float)
    codes, uniq = pd.factorize(groups)
    sums = np.zeros((len(uniq),) + a.shape[1:]); np.add.at(sums, codes, a)
    cnt = np.bincount(codes).astype(float)
    mean = sums / (cnt if a.ndim == 1 else cnt[:, None])
    return a - mean[codes]


def fit_fe(df: pd.DataFrame, quad: bool = False, min_compound_laps: int = 40):
    """Fit the fixed-effects model. Returns dict name->coef (+ '_n', '_rmse').

    Compounds with fewer than `min_compound_laps` laps in the data get NaN coefficients
    (the caller falls back to the global estimate) instead of an unstable fit.
    """
    X, names = design(df, quad)
    y = df["LapTimeSec"].to_numpy(float)
    g = df["RaceDriver"].to_numpy()
    Xd, yd = _demean(X, g), _demean(y, g)
    counts = df["Compound"].value_counts()
    keep = np.abs(Xd).sum(0) > 1e-9
    for j, n in enumerate(names):
        parts = n.split("_")
        if len(parts) > 1 and parts[1] in COMPOUNDS and counts.get(parts[1], 0) < min_compound_laps:
            keep[j] = False
    A = Xd[:, keep]
    beta_k = np.linalg.lstsq(A, yd, rcond=1e-10)[0]
    beta = np.full(X.shape[1], np.nan); beta[keep] = beta_k
    out = dict(zip(names, beta)); out["_n"] = len(df)
    resid = yd - A @ beta_k
    out["_rmse"] = float(np.sqrt((resid ** 2).mean()))
    # classical standard errors (dof corrected for the demeaning); laps are autocorrelated so we
    # inflate them later in fit_per_race
    dof = max(len(yd) - len(np.unique(g)) - A.shape[1], 1)
    cov = np.linalg.pinv(A.T @ A) * (resid ** 2).sum() / dof
    se = np.full(X.shape[1], np.nan); se[keep] = np.sqrt(np.diag(cov))
    out["_se"] = dict(zip(names, se))
    return out


def predict_relative(df: pd.DataFrame, coef: dict, quad: bool = False) -> np.ndarray:
    """Predict lap time *relative to the driver-race mean* (baseline removed)."""
    X, names = design(df, quad)
    beta = np.array([0.0 if np.isnan(coef[n]) else coef[n] for n in names])
    Z = X @ beta
    return Z - pd.Series(Z).groupby(df["RaceDriver"].to_numpy()).transform("mean").to_numpy()


def bootstrap_global(df: pd.DataFrame, n_boot: int = 200, seed: int = 0, quad: bool = False):
    """Cluster bootstrap over *races* -> confidence intervals that respect circuit-to-circuit variation."""
    rng = np.random.default_rng(seed)
    races = df["Race"].unique()
    groups = {r: df[df["Race"] == r] for r in races}
    rows = []
    for _ in range(n_boot):
        pick = rng.choice(races, size=len(races), replace=True)
        # relabel RaceDriver per draw so repeated races keep their own fixed effects
        parts = [groups[r].assign(RaceDriver=groups[r]["RaceDriver"] + f"#{i}") for i, r in enumerate(pick)]
        c = fit_fe(pd.concat(parts), quad)
        rows.append({k: v for k, v in c.items() if not k.startswith("_")})
    return pd.DataFrame(rows)


def fit_per_race(df: pd.DataFrame, min_laps: int = 120, se_inflate: float = 2.0, quad: bool = False):
    """Circuit-specific coefficients with random-effects (empirical-Bayes) shrinkage.

    1. Fit the fixed-effects model separately for each circuit (noisy, esp. for soft tyres).
    2. Combine circuits with a DerSimonian-Laird random-effects meta-analysis: this gives the
       typical value `mu` and how much circuits genuinely differ (`tau`).
    3. Shrink each circuit's estimate towards `mu`. A circuit with a precise estimate keeps its own
       value; a circuit with a noisy estimate (or none) is pulled to the typical value.

    Standard errors are multiplied by `se_inflate` because consecutive laps are correlated.
    Returns (summary_df_of_mu_tau, per_race_table).
    """
    races = sorted(df["Race"].unique())
    local = {r: fit_fe(df[df["Race"] == r], quad) for r in races if (df["Race"] == r).sum() >= min_laps}
    names = [n for n in design(df.head(5), quad)[1]]
    table = pd.DataFrame({"Race": races})
    table["n_laps"] = [int((df["Race"] == r).sum()) for r in races]
    summ = []
    for k in names:
        b = np.array([local[r][k] if r in local else np.nan for r in races])
        se = np.array([local[r]["_se"][k] * se_inflate if r in local else np.nan for r in races])
        ok = np.isfinite(b) & np.isfinite(se) & (se > 0)
        if ok.sum() < 3:                      # not enough circuits: fall back to the pooled fit
            mu, tau2 = fit_fe(df, quad)[k], 0.0
            shr = np.full(len(races), mu)
        else:
            w = 1 / se[ok] ** 2
            mu_fe = (w * b[ok]).sum() / w.sum()
            Q = (w * (b[ok] - mu_fe) ** 2).sum()
            c_ = w.sum() - (w ** 2).sum() / w.sum()
            tau2 = max((Q - (ok.sum() - 1)) / c_, 1e-6)
            w_re = 1 / (se[ok] ** 2 + tau2)
            mu = (w_re * b[ok]).sum() / w_re.sum()
            shr = np.full(len(races), mu)
            shr[ok] = (b[ok] / se[ok] ** 2 + mu / tau2) / (1 / se[ok] ** 2 + 1 / tau2)
        table[k] = shr
        table[k + "_raw"] = b
        se_mu = float(np.sqrt(1 / w_re.sum())) if ok.sum() >= 3 else np.nan
        summ.append({"coef": k, "typical_mu": mu, "se_mu": se_mu, "between_circuit_sd_tau": np.sqrt(tau2),
                     "circuits_used": int(ok.sum())})
    return pd.DataFrame(summ), table
