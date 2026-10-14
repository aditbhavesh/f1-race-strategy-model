"""Race-pace prediction: how well can we predict how a driver's lap times *evolve* during a race?

Target ("RelPace") = lap time minus that driver's own average lap time in that race.
It strips out car/driver/circuit speed and keeps what strategy depends on: the effect of
tyre age, compound and fuel/track evolution on lap time.

Validation is "leave drivers out": the model is trained on some drivers and tested on drivers it
has never seen (all their laps), so it cannot memorise a driver's own stint.

Models compared (all trained only on the training drivers):
  0. Naive          - predict zero change (every lap equal to the driver's average)
  1. Global linear  - one fuel + tyre-degradation model for all circuits
  2. Circuit linear - circuit-specific degradation (shrunk towards global)
  3. Gradient boost - non-linear model (HistGradientBoosting) with circuit as a feature
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from degradation import fit_fe, fit_per_race, predict_relative

N_FOLDS = 5


def _demean_pred(pred: np.ndarray, rd: np.ndarray) -> np.ndarray:
    return pred - pd.Series(pred).groupby(rd).transform("mean").to_numpy()


def _gbm_features(df: pd.DataFrame, races: list) -> pd.DataFrame:
    X = pd.DataFrame({
        "TyreLife": df["TyreLife"].to_numpy(), "LapNumber": df["LapNumber"].to_numpy(),
        "LapFrac": df["LapFrac"].to_numpy(), "TotalLaps": df["TotalLaps"].to_numpy(),
        "Compound": df["Compound"].map({"SOFT": 0, "MEDIUM": 1, "HARD": 2}).to_numpy(),
        "Race": pd.Categorical(df["Race"], categories=races).codes,
    })
    return X


def cross_validate(clean: pd.DataFrame, seed: int = 0):
    """Return (metrics_df, preds_df)."""
    clean = clean.reset_index(drop=True)
    y = clean["RelPace"].to_numpy()
    rd = clean["RaceDriver"].to_numpy()
    races = sorted(clean["Race"].unique())
    preds = {k: np.zeros(len(clean)) for k in ["naive", "global_linear", "circuit_linear", "gradient_boost"]}

    for tr, te in GroupKFold(n_splits=N_FOLDS).split(clean, y, clean["Driver"]):
        train, test = clean.iloc[tr], clean.iloc[te]
        # 1. global linear
        g = fit_fe(train)
        preds["global_linear"][te] = predict_relative(test, g)
        # 2. circuit-specific linear
        _, table = fit_per_race(train)
        out = np.zeros(len(test))
        for race, idx in test.groupby("Race").indices.items():
            coef = table[table["Race"] == race].iloc[0].to_dict()
            out[idx] = predict_relative(test.iloc[idx], coef)
        preds["circuit_linear"][te] = out
        # 3. gradient boosting (categorical-ish circuit id, monotone in nothing -> let it learn)
        gb = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300,
                                           categorical_features=[5], random_state=seed)
        gb.fit(_gbm_features(train, races), y[tr])
        preds["gradient_boost"][te] = _demean_pred(gb.predict(_gbm_features(test, races)), rd[te])

    rows = []
    for name, p in preds.items():
        err = y - p
        rows.append({"model": name, "MAE_s": np.abs(err).mean(), "RMSE_s": np.sqrt((err ** 2).mean()),
                     "R2": 1 - (err ** 2).sum() / (y ** 2).sum()})
    m = pd.DataFrame(rows)
    base = m.loc[m["model"] == "naive", "MAE_s"].iat[0]
    m["MAE_gain_vs_naive_%"] = 100 * (1 - m["MAE_s"] / base)
    return m, pd.DataFrame(preds).assign(actual=y, Race=clean["Race"].to_numpy(),
                                         Driver=clean["Driver"].to_numpy(),
                                         Compound=clean["Compound"].to_numpy(),
                                         LapNumber=clean["LapNumber"].to_numpy(),
                                         RaceDriver=clean["RaceDriver"].to_numpy(),
                                         TyreLife=clean["TyreLife"].to_numpy())


if __name__ == "__main__":
    from data import load_raw, clean_laps
    c = clean_laps(load_raw())
    m, p = cross_validate(c)
    print(m.round(3).to_string(index=False))
