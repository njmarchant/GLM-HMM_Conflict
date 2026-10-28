"""
5.3_alcohol_dwell_times.py
--------------------------
State persistence (dwell times) in the alcohol tests (Fig. S5C-D): how many consecutive trials a rat stays
in a latent state, by dose, in the first 15 min and after the first 15 min of the test sessions.

Input: the table written by 5.1_alcohol_states_to_trials.py. States are re-derived here from Prob_State_1..3
at THRESHOLD (0.65), exactly as in 5.2.

Run definition (identical to Phase 3, 2.3_plot_occupancy_and_dwell_times.py, with the 3-trial gap of the Methods):
  Within a session, classified trials form runs of the same state. A run may bridge up to MAX_GAP (3) consecutive
  unclassified trials if the same state resumes on both sides; bridged trials count towards its length. A run ends
  when a different state is classified, after more than MAX_GAP consecutive unclassified trials, or at the end of
  the trial sequence it is computed on.
Epochs: runs are computed SEPARATELY within each epoch (first 15 min: trial start <= 900 s; after: > 900 s), so a
  run cannot cross the boundary. Runs that touch an epoch edge are flagged (Starts_at_epoch_start, Ended_by =
  'epoch_end') because their true length is censored. The first epoch holds only about 16 trials per session,
  so its runs cannot exceed that; the number of trials in the epoch also depends on the dose (trials last longer
  when a rat omits), which is printed below.

Outputs (OUTPUT_DIR, default HMM_Outputs/Alcohol_n15 so that they sit with the 5.2 results)
  dwell_runs_{epoch}.csv            one row per run (Rat, Dose, Session_ID, State, Run_Length, flags)
  dwell_per_rat_{epoch}.csv         per rat x dose x state: number of runs, median and mean run length
  dwell_median_wide_{epoch}.csv     rat x state rows, one column per dose (median run length); blanks where a rat has
  dwell_mean_wide_{epoch}.csv       no run of that state at that dose. For Prism (mixed-effects model handles blanks)
  dwell_summary.csv / SUMMARY_dwell.txt   pooled-run counts, medians and quartiles per epoch x state x dose
  dwell_tests.csv                   cross-check: mixed model of log run length by dose within each state/epoch,
                                    random intercept for rat, Holm-corrected vs saline
  Fig_S5CD_DwellTimes.{svg,png}

Usage:  python 5.3_alcohol_dwell_times.py [--input F] [--exclude none | R18 R21] [--out DIR] [--max-gap 3]
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

VERSION = "5.3 v2"
INPUT_PATH = os.path.join("Data", "collated_2026_conflict_alcohol_with_Probabilities_065.csv")
OUTPUT_DIR = os.path.join("HMM_Outputs", "Alcohol_n15")
DOSE_COL = "Session_Type"
DOSES = ["Saline", "Alcohol_Low", "Alcohol_High"]
DOSE_LABELS = {"Saline": "Saline", "Alcohol_Low": "0.5 g/kg", "Alcohol_High": "1.0 g/kg"}
THRESHOLD = 0.65
SPLIT_S = 900
MAX_GAP = 3
EXCLUDE_RATS = []
YMAX = 90                  # y-axis maximum of both dwell-time panels (Fig. S5C-D)
STATES = [1, 2, 3]
STATE_NAMES = {1: "Risk-\nTaking", 2: "Risk-\nAverse", 3: "Repeating"}
COLORS = {
    1: dict(zip(DOSES, ["#E2E4A8", "#C4C742", "#B0B300"])),
    2: dict(zip(DOSES, ["#E6AEE5", "#CA3CC6", "#B500B3"])),
    3: dict(zip(DOSES, ["#A8E2E4", "#42C4C7", "#00B0B3"])),
}


# =========================================================
# RUNS
# =========================================================
def compute_runs(states, max_gap=MAX_GAP):
    """states: 1-D array with 1/2/3 or NaN. Returns a list of dicts (State, Run_Length, Start_Index, Ended_by).
    Same logic as Phase 3 (2.3), with start index and the reason each run ended."""
    out = []
    cur, length, gap, start = None, 0, 0, None
    for i, s in enumerate(states):
        if np.isnan(s):
            if cur is not None:
                gap += 1
                if gap > max_gap:                                   # gap too long: the run ended before the gap
                    out.append(dict(State=int(cur), Run_Length=length, Start_Index=start, Ended_by="gap"))
                    cur, length, gap, start = None, 0, 0, None
            continue
        if cur is not None and s == cur:                            # same state resumes, possibly after a short gap
            length += 1 + gap
        else:
            if cur is not None:
                out.append(dict(State=int(cur), Run_Length=length, Start_Index=start,
                                Ended_by="switch" if gap == 0 else "switch_after_gap"))
            cur, length, start = s, 1, i
        gap = 0
    if cur is not None:
        out.append(dict(State=int(cur), Run_Length=length, Start_Index=start, Ended_by="epoch_end"))
    return out


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    run = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok])]):
        run = max(run, (len(ok) - rank) * p[i])
        out[i] = min(1.0, run)
    return out


# =========================================================
# MAIN
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Dwell times of latent states in the alcohol tests (Fig. S5C-D).")
    ap.add_argument("--input", default=INPUT_PATH)
    ap.add_argument("--exclude", nargs="*", default=None, help="rat IDs to exclude, or 'none' (default: none)")
    ap.add_argument("--out", default=OUTPUT_DIR)
    ap.add_argument("--max-gap", type=int, default=MAX_GAP, help="unclassified trials a run may bridge (default 3)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    excl = [] if (args.exclude is None or [x.lower() for x in args.exclude] == ["none"]) else args.exclude
    os.makedirs(args.out, exist_ok=True)
    print(f"[{VERSION}] input: {args.input} | output: {args.out} | max gap {args.max_gap} | excluded: {excl}")

    df = pd.read_csv(args.input, dtype={"Session_ID": str})
    need = ["Rat", DOSE_COL, "Session_ID", "Trial", "Timestamp", "Prob_State_1", "Prob_State_2", "Prob_State_3"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"Input is missing column(s): {miss}")
    df[DOSE_COL] = df[DOSE_COL].astype(str).str.strip()
    if set(df[DOSE_COL]) - set(DOSES):
        sys.exit(f"Unrecognised dose labels {set(df[DOSE_COL]) - set(DOSES)}")
    df = df[~df["Rat"].isin(excl)]
    have = df.groupby("Rat")[DOSE_COL].agg(set)
    incomplete = [r for r, s in have.items() if not set(DOSES) <= s]
    if incomplete:
        print(f"Dropping rats without all three doses: {incomplete}")
        df = df[~df["Rat"].isin(incomplete)]
    P = df[["Prob_State_1", "Prob_State_2", "Prob_State_3"]].values
    df["State"] = np.where(np.nanmax(P, axis=1) >= THRESHOLD, np.nanargmax(P, axis=1) + 1, np.nan)
    print(f"{df['Rat'].nunique()} rats, {df.groupby(['Rat', 'Session_ID']).ngroups} sessions, {len(df)} trials")

    # ---- runs, per epoch, per session ----
    rows, epoch_len = [], []
    for (rat, sid), s in df.sort_values(["Rat", "Session_ID", "Trial"]).groupby(["Rat", "Session_ID"]):
        dose = s[DOSE_COL].iloc[0]
        for epoch, sub in [("first_15min", s[s["Timestamp"] <= SPLIT_S]), ("after_15min", s[s["Timestamp"] > SPLIT_S])]:
            if len(sub) == 0:
                continue
            epoch_len.append(dict(Rat=rat, Dose=dose, Epoch=epoch, N_trials=len(sub)))
            trials = sub["Trial"].values
            for r in compute_runs(sub["State"].values.astype(float), args.max_gap):
                rows.append(dict(Rat=rat, Dose=dose, Session_ID=sid, Epoch=epoch, State=r["State"],
                                 Run_Length=r["Run_Length"], Start_Trial=int(trials[r["Start_Index"]]),
                                 Starts_at_epoch_start=bool(r["Start_Index"] == 0), Ended_by=r["Ended_by"]))
    runs = pd.DataFrame(rows)
    elen = pd.DataFrame(epoch_len)
    lines = [f"Alcohol dwell times. Threshold {THRESHOLD}; split at {SPLIT_S} s; max gap {args.max_gap}; "
             f"{df['Rat'].nunique()} rats. Excluded: {excl}", f"Rats: {sorted(df['Rat'].unique())}"]

    lines.append("\nTrials per session in each epoch (mean over sessions), by dose:")
    lines.append(elen.groupby(["Epoch", "Dose"]).N_trials.mean().unstack().reindex(columns=DOSES).round(1).to_string())

    summ_rows = []
    for epoch in ["first_15min", "after_15min"]:
        r = runs[runs.Epoch == epoch]
        r.to_csv(os.path.join(args.out, f"dwell_runs_{epoch}.csv"), index=False)
        pr = (r.groupby(["Rat", "Dose", "State"]).Run_Length.agg(N_runs="size", Median_run="median", Mean_run="mean").reset_index())
        pr.to_csv(os.path.join(args.out, f"dwell_per_rat_{epoch}.csv"), index=False)
        for stat, name in [("Median_run", "median"), ("Mean_run", "mean")]:
            wide = pr.pivot_table(index=["Rat", "State"], columns="Dose", values=stat).reindex(columns=DOSES)
            full = pd.MultiIndex.from_product([sorted(df["Rat"].unique()), STATES], names=["Rat", "State"])
            wide.reindex(full).reset_index().to_csv(os.path.join(args.out, f"dwell_{name}_wide_{epoch}.csv"), index=False)
        lines.append(f"\n=== {epoch}: {len(r)} runs; runs ending at the epoch edge: {100 * (r.Ended_by == 'epoch_end').mean():.0f}%, "
                     f"starting at its first trial: {100 * r.Starts_at_epoch_start.mean():.0f}% ===")
        lines.append("Pooled runs: n runs [n rats] | median (Q1-Q3) trials")
        for st in STATES:
            cells = []
            for d in DOSES:
                v = r[(r.State == st) & (r.Dose == d)]
                if len(v):
                    q1, med, q3 = v.Run_Length.quantile([.25, .5, .75])
                    cells.append(f"{DOSE_LABELS[d]}: {len(v):3d} [{v.Rat.nunique():2d}] {med:4.1f} ({q1:.0f}-{q3:.0f})")
                else:
                    cells.append(f"{DOSE_LABELS[d]}: none")
                summ_rows.append(dict(Epoch=epoch, State=st, Dose=d, N_runs=len(v), N_rats=v.Rat.nunique(),
                                      Median=v.Run_Length.median() if len(v) else np.nan,
                                      Q1=v.Run_Length.quantile(.25) if len(v) else np.nan,
                                      Q3=v.Run_Length.quantile(.75) if len(v) else np.nan,
                                      Mean=v.Run_Length.mean() if len(v) else np.nan))
            lines.append(f"  State {st}: " + " | ".join(cells))
    pd.DataFrame(summ_rows).to_csv(os.path.join(args.out, "dwell_summary.csv"), index=False)

    # ---- cross-check: mixed model on log run length, random intercept for rat, per state and epoch ----
    test_rows = []
    try:
        import statsmodels.formula.api as smf
        for epoch in ["first_15min", "after_15min"]:
            for st in STATES:
                d = runs[(runs.Epoch == epoch) & (runs.State == st)].copy()
                d["logL"] = np.log(d.Run_Length)
                d["Dose"] = pd.Categorical(d["Dose"], categories=DOSES)
                counts = d.groupby("Dose", observed=True).Run_Length.size()
                if d.Rat.nunique() < 4 or counts.get("Saline", 0) < 3 or (counts.reindex(DOSES).fillna(0) >= 3).sum() < 2:
                    test_rows.append(dict(Epoch=epoch, State=st, Note="too few runs for a mixed model"))
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = smf.mixedlm("logL ~ C(Dose, Treatment('Saline'))", d, groups=d["Rat"]).fit(reml=True)
                    cs = []
                    for dose in ["Alcohol_Low", "Alcohol_High"]:
                        k = [n for n in m.params.index if dose in n]
                        if k and counts.get(dose, 0) >= 3:
                            b, se, p = m.params[k[0]], m.bse[k[0]], m.pvalues[k[0]]
                            cs.append(dict(Epoch=epoch, State=st, Comparison=f"{dose} vs Saline", Ratio_of_geometric_means=np.exp(b),
                                           CI_low=np.exp(b - 1.96 * se), CI_high=np.exp(b + 1.96 * se), p=p,
                                           N_runs=int(len(d)), N_rats=int(d.Rat.nunique())))
                    adj = holm([c["p"] for c in cs])
                    for c, a in zip(cs, adj):
                        c["Holm_p"] = a
                    test_rows += cs
                except Exception as e:
                    test_rows.append(dict(Epoch=epoch, State=st, Note=f"model failed: {type(e).__name__}"))
    except ImportError:
        print("statsmodels not installed: skipping the mixed-model cross-check.")
    tests = pd.DataFrame(test_rows)
    tests.to_csv(os.path.join(args.out, "dwell_tests.csv"), index=False)
    if len(tests):
        lines.append("\nCross-check (mixed model, log run length, random intercept for rat; Holm within state/epoch):")
        lines.append(tests.round(4).to_string(index=False))

    txt = "\n".join(lines)
    with open(os.path.join(args.out, "SUMMARY_dwell.txt"), "w") as f:
        f.write(txt)
    print("\n" + txt)

    # ---- figure (same layout as Fig. S5C-D) ----
    rng = np.random.default_rng(14)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, epoch, title in zip(axes, ["first_15min", "after_15min"], ["First 15 minutes", "After 15 minutes"]):
        w = 0.25
        for st in STATES:
            for i, dose in enumerate(DOSES):
                v = runs[(runs.Epoch == epoch) & (runs.State == st) & (runs.Dose == dose)].Run_Length.values
                if len(v) == 0:
                    continue
                x = (st - 1) + (i - 1) * w
                bp = ax.boxplot(v, positions=[x], widths=w * 0.9, patch_artist=True, showfliers=False,
                                medianprops=dict(color="black", linewidth=1.2), whiskerprops=dict(linewidth=1.0),
                                capprops=dict(linewidth=1.0), boxprops=dict(linewidth=1.2))
                bp["boxes"][0].set_facecolor(COLORS[st][dose])
                ax.scatter(rng.normal(x, w * 0.12, len(v)), v, s=6, color="black", alpha=0.35, zorder=3, linewidths=0)
        ax.set_xticks(range(3))
        ax.set_xticklabels([STATE_NAMES[s] for s in STATES])
        ax.set_xlabel("State", fontweight="bold")
        ax.set_ylabel("Consecutive runs (number of trials)", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, YMAX)
        ax.set_yticks(range(0, YMAX + 1, 10))
        n_over = int((runs[runs.Epoch == epoch].Run_Length > YMAX).sum())
        if n_over:
            print(f"WARNING: {n_over} run(s) longer than {YMAX} trials in {epoch} are cut off by the y-axis limit.")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="black") for c in ["#DDDDDD", "#999999", "#444444"]]
    axes[1].legend(handles, [DOSE_LABELS[d] for d in DOSES], frameon=False, title="Dose (pale to dark)",
                   loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    for ext, kw in [("svg", {}), ("png", {"dpi": 300})]:
        plt.savefig(os.path.join(args.out, f"Fig_S5CD_DwellTimes.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"\nSaved to {args.out}/")
