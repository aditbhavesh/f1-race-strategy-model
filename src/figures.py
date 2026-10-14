"""Charts for the README. Palette: validated default categorical order (blue, orange, aqua), light surface.

Design rules applied: one axis per chart, direct labels, thin marks, recessive grid, text in ink colours
(never the series colour), identity never colour-alone (labels / markers), values also saved as CSV tables.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COMP_COLOR = {"HARD": BLUE, "MEDIUM": ORANGE, "SOFT": AQUA}
STOP_COLOR = {1: BLUE, 2: ORANGE, 3: AQUA}
OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 9,
    })


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def short(race):
    return race.replace(" Grand Prix", "")


def fig_typical_degradation(summ):
    s = summ.set_index("coef")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, c in enumerate(["SOFT", "MEDIUM", "HARD"]):
        mu, se = s.loc[f"deg_{c}", "typical_mu"], s.loc[f"deg_{c}", "se_mu"]
        ax.plot([mu - 1.96 * se, mu + 1.96 * se], [i, i], color=COMP_COLOR[c], lw=2.5, solid_capstyle="round")
        ax.plot(mu, i, "o", ms=9, color=COMP_COLOR[c], mec=SURFACE, mew=2)
        ax.text(mu + 1.96 * se + 0.004, i, f"{mu:.3f} s/lap", va="center", color=INK, fontsize=10)
    ax.set_yticks(range(3)); ax.set_yticklabels(["Soft", "Medium", "Hard"]); ax.invert_yaxis()
    ax.set_xlim(0, 0.14); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Lap time lost per extra lap of tyre age (seconds), typical circuit, 95% interval")
    ax.set_title("Softer tyres wear faster, but the intervals overlap")
    _save(fig, "01_typical_degradation.png")


def fig_circuit_degradation(table):
    t = table.copy()
    t["avg"] = t[["deg_MEDIUM", "deg_HARD"]].mean(axis=1)
    t = t.sort_values("avg")
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    y = np.arange(len(t))
    ax.hlines(y, t["deg_MEDIUM"], t["deg_HARD"], color=GRID, lw=2, zorder=1)
    ax.scatter(t["deg_HARD"], y, s=55, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="Hard")
    ax.scatter(t["deg_MEDIUM"], y, s=55, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="Medium")
    ax.set_yticks(y); ax.set_yticklabels([short(r) for r in t["Race"]])
    ax.set_xlabel("Seconds lost per lap of tyre age (shrunk circuit estimate)")
    ax.set_title("Tyre wear depends heavily on the circuit")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _save(fig, "02_degradation_by_circuit.png")


def fig_model_comparison(metrics):
    names = {"naive": "Naive (no change)", "global_linear": "Global linear", "circuit_linear": "Circuit linear",
             "gradient_boost": "Gradient boosting"}
    m = metrics.copy(); m["label"] = m["model"].map(names)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(m))[::-1]
    ax.barh(y, m["MAE_s"], color=BLUE, height=0.55)
    for yi, v, g in zip(y, m["MAE_s"], m["MAE_gain_vs_naive_%"]):
        txt = f"{v:.2f} s" + ("" if g == 0 else f"  ({g:.0f}% better)")
        ax.text(v + 0.01, yi, txt, va="center", color=INK, fontsize=9.5)
    ax.set_yticks(y); ax.set_yticklabels(m["label"]); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Mean absolute error in predicted lap-time change (seconds), unseen drivers")
    ax.set_title("Pace model accuracy")
    _save(fig, "03_pace_model_comparison.png")


def fig_pace_by_age(preds, fuel_per_lap):
    """Tyre-only degradation curve: fuel burn removed, each compound shown relative to its own age-3 value."""
    p = preds[preds["TyreLife"].between(3, 35)].copy()
    lap_dev = p["LapNumber"] - p.groupby("RaceDriver")["LapNumber"].transform("mean")
    p["actual_fc"] = p["actual"] - fuel_per_lap * lap_dev
    p["pred_fc"] = p["gradient_boost"] - fuel_per_lap * lap_dev
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for c in ["SOFT", "MEDIUM", "HARD"]:
        x = p[p["Compound"] == c]
        g = x.groupby("TyreLife").agg(actual=("actual_fc", "mean"), pred=("pred_fc", "mean"), n=("actual", "size"))
        g = g[g["n"] >= 60]
        if len(g) < 5:
            continue
        g = g - g.iloc[0][["actual", "pred"]].reindex(g.columns).fillna(0)
        ax.plot(g.index, g["pred"], color=COMP_COLOR[c], lw=2)
        ax.plot(g.index, g["actual"], "o", color=COMP_COLOR[c], ms=4, alpha=0.75)
        ax.text(g.index.max() + 0.5, g["pred"].iloc[-1], c.title(), color=INK, va="center", fontsize=9.5)
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Extra lap time vs the same tyre at age 3 (s)")
    ax.set_title("Tyre degradation: model vs reality (fuel burn removed)")
    ax.text(0.99, 0.04, "Dots = actual average, line = model prediction (unseen drivers)",
            transform=ax.transAxes, color=INK2, fontsize=8.5, ha="right")
    _save(fig, "04_pace_vs_tyre_age.png")


def fig_strategy_gaps(opt_df, winners):
    d = opt_df.copy()
    d["second"] = d.groupby("Race")["delta_s"].transform(lambda s: s[s > 0].min() if (s > 0).any() else np.nan)
    order = d.drop_duplicates("Race").sort_values("second")["Race"].tolist()[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    for yi, race in enumerate(order):
        r = d[d["Race"] == race]
        ax.hlines(yi, 0, min(r["delta_s"].max(), 30), color=GRID, lw=1.5, zorder=1)
        for _, row in r.iterrows():
            x = min(row["delta_s"], 30)
            ax.scatter(x, yi, s=60, color=STOP_COLOR[int(row["stops"])], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            if winners.get(race) == int(row["stops"]):
                ax.scatter(x, yi, s=190, facecolor="none", edgecolor=INK, linewidth=1.4, zorder=4)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([short(r) for r in order])
    ax.set_xlim(-1, 31); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Seconds slower than the model's best strategy (capped at 30 s)")
    ax.set_title("How much slower is each stop count? (ringed = winner's actual)")
    for s in (1, 2, 3):
        ax.scatter([], [], s=60, color=STOP_COLOR[s], label=f"{s}-stop")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3)
    _save(fig, "05_strategy_gap_by_race.png")


def fig_validation(summary):
    s = summary.copy()
    lab = {"Always predict 1 stop (baseline)": "Baseline:\nalways 1 stop",
           "Simulator, blind (leave-one-race-out)": "Simulator,\nblind",
           "Simulator, calibrated tyre data + blind stint caps": "Calibrated tyres,\nblind stint caps",
           "Simulator, fully calibrated to the weekend": "Fully calibrated\nto the weekend"}
    s["label"] = s["test"].map(lab)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(s))
    colors = [GRID if i == 0 else BLUE for i in range(len(s))]
    ax.bar(x, s["accuracy"] * 100, color=colors, width=0.6)
    ax.bar(0, s["accuracy"].iat[0] * 100, color="#b9b8b2", width=0.6)
    for xi, a, c, n in zip(x, s["accuracy"] * 100, s["correct"], s["races"]):
        ax.text(xi, a + 1.5, f"{a:.0f}%\n({c}/{n})", ha="center", color=INK, fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(s["label"], fontsize=9)
    ax.set_ylim(0, 105); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Races where predicted stop count = winner's (%)")
    ax.set_title("Does the simulator pick the winner's number of stops?")
    _save(fig, "06_validation.png")


def fig_sensitivity(sens):
    g = sens.groupby("scenario")[["stops_unchanged", "sequence_unchanged"]].mean() * 100
    g = g.sort_values("stops_unchanged")
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    y = np.arange(len(g)); h = 0.36
    ax.barh(y + h / 2, g["stops_unchanged"], height=h, color=BLUE, label="Same number of stops")
    ax.barh(y - h / 2, g["sequence_unchanged"], height=h, color=ORANGE, label="Same stops and tyre order")
    for yi, a, b in zip(y, g["stops_unchanged"], g["sequence_unchanged"]):
        ax.text(a + 1, yi + h / 2, f"{a:.0f}%", va="center", fontsize=8.5, color=INK)
        ax.text(b + 1, yi - h / 2, f"{b:.0f}%", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(g.index); ax.set_xlim(0, 112); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of the 21 races where the recommendation does not change (%)")
    ax.set_title("How fragile is the recommendation?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _save(fig, "07_sensitivity.png")


def fig_breakeven(be):
    d = be.dropna(subset=["breakeven_pit_loss_s"]).sort_values("breakeven_pit_loss_s")
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    y = np.arange(len(d))
    ax.hlines(y, d["pit_loss_s"], d["breakeven_pit_loss_s"], color=GRID, lw=2, zorder=1)
    ax.scatter(d["pit_loss_s"], y, s=55, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="Measured pit loss")
    ax.scatter(d["breakeven_pit_loss_s"], y, s=55, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="Break-even: 1-stop = 2-stop")
    ax.set_yticks(y); ax.set_yticklabels([short(r) for r in d["Race"]]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Time lost per pit stop (seconds)")
    ax.set_title("Pit-loss break-even between 1-stop and 2-stop")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    _save(fig, "08_pit_loss_breakeven.png")


def fig_stop_cost_by_condition(green, neutral):
    """Time lost per stop under green flag, VSC and SC (each dot = one stop)."""
    rng = np.random.default_rng(3)
    groups = [("Green flag", green, BLUE), ("Virtual safety car", neutral.loc[neutral["kind"] == 1, "loss_s"], ORANGE),
              ("Safety car", neutral.loc[neutral["kind"] == 2, "loss_s"], AQUA)]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for i, (name, v, col) in enumerate(groups):
        v_all = np.asarray(v, dtype=float)
        v_all = v_all[~np.isnan(v_all)]
        v = v_all[(v_all > -5) & (v_all < 45)]          # only the dots are clipped to the axis
        ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=14 if len(v_all) > 100 else 28, color=col, alpha=0.5, linewidth=0)
        med = np.median(v_all)                           # median and n use every measured stop (matches the tables)
        ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, lw=2)
        extra = f", {len(v_all) - len(v)} off-axis" if len(v_all) > len(v) else ""
        ax.text(46, i, f"median {med:.1f} s  (n={len(v_all)}{extra})", va="center", color=INK, fontsize=9.5)
    ax.set_yticks(range(3)); ax.set_yticklabels([g[0] for g in groups]); ax.invert_yaxis()
    ax.set_xlim(-5, 62); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Time lost by a stop relative to cars that did not stop (seconds)")
    ax.set_title("A stop under a safety car is cheaper, but not free")
    _save(fig, "09_stop_cost_by_condition.png")


def fig_sc_gap_shift(opt_df, sc_val):
    """Margin between best and second-best stop count: no safety cars vs expected over random safety cars."""
    d = opt_df.copy()
    det = d[d["delta_s"] > 0].groupby("Race")["delta_s"].min()
    gap_cols = [c for c in sc_val.columns if c.startswith("expected_gap_")]
    exp = sc_val.set_index("Race")[gap_cols].where(lambda x: x > 1e-9).min(axis=1)
    t = pd.DataFrame({"det": det, "exp": exp}).dropna().sort_values("det")
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    y = np.arange(len(t))
    ax.hlines(y, t["det"], t["exp"], color=GRID, lw=2, zorder=1)
    ax.scatter(t["det"], y, s=55, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="No safety cars")
    ax.scatter(t["exp"], y, s=55, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="Average over random safety cars")
    ax.set_yticks(y); ax.set_yticklabels([short(r) for r in t.index]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Seconds the runner-up stop count is slower than the best")
    ax.set_title("Safety-car risk barely moves the margins")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)
    _save(fig, "10_safety_car_margin_shift.png")


def make_all(ctx):
    _style()
    fig_typical_degradation(ctx["summ"])
    fig_circuit_degradation(ctx["table"])
    fig_model_comparison(ctx["metrics"])
    fig_pace_by_age(ctx["preds"], ctx["fuel"])
    fig_strategy_gaps(ctx["opt"], ctx["winner_stops"])
    fig_validation(ctx["val_summary"])
    fig_sensitivity(ctx["sens"])
    fig_breakeven(ctx["breakeven"])
    if "sc_val" in ctx:
        fig_stop_cost_by_condition(ctx["green_losses"], ctx["neutral_losses"])
        fig_sc_gap_shift(ctx["opt"], ctx["sc_val"])
