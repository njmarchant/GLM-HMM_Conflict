"""
5.2_alcohol_state_occupancy.py
------------------------------
State occupancy under acute alcohol (Fig. 5B; first-15-min epoch Fig. S5B), plus the
per-rat behaviour tables behind Fig. 5A / S5A.

Input: the table written by 5.1_alcohol_states_to_trials.py (smoothed state posteriors
from the FROZEN 3-state model, decoded on full sessions). The state label is
re-derived here from Prob_State_1..3 at THRESHOLD, so the threshold is explicit and does
not depend on whatever the input file was thresholded at.

Epochs (Timestamp = seconds from session start; sessions are 100 trials, ~90 min):
  first_15min  Timestamp <= 900   (Fig. S5)
  after_15min  Timestamp >  900   (main analysis, Fig. 5)

Occupancy = % of a rat's ASSIGNED trials (posterior >= THRESHOLD) in each state, with the
rat's two sessions per dose pooled. A rat with no assigned trials in a cell is NaN, not 0.

Statistics (rat is the unit; complete cases only):
  Friedman test across the three doses, and Wilcoxon signed-rank tests of saline vs each
  dose, Holm-corrected within each state (or cue). These are a nonparametric check.
  The repeated-measures ANOVAs reported in the paper (State x Dose; Cue x Dose) are run
  in GraphPad Prism on the exported *_wide.csv tables, as for the other behavioural data.

Usage (from the repository root)
  python 5.2_alcohol_state_occupancy.py                                   # n = 15, HMM_Outputs/Alcohol_n15
  python 5.2_alcohol_state_occupancy.py --exclude R18 R21 --out HMM_Outputs/Alcohol_n13
--exclude overrides EXCLUDE_RATS ('none' = exclude nobody); --out overrides the output folder, so
several rat sets can be run side by side without overwriting each other.

Outputs (HMM_Outputs/Alcohol_n15/ by default)
  occupancy_per_rat_{epoch}.csv, occupancy_wide_{epoch}.csv
  behaviour_per_rat_{epoch}.csv, behaviour_wide_{epoch}.csv
  dose_tests.csv, SUMMARY.txt
  Fig5B_StateOccupancy_after15min.{svg,png}, FigS5B_StateOccupancy_first15min.{svg,png}
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

# =========================================================
# CONFIGURATION
# =========================================================
VERSION = "5.2 v4"
INPUT_PATH = os.path.join("Data", "collated_2026_conflict_alcohol_with_Probabilities_065.csv")
OUTPUT_DIR = os.path.join("HMM_Outputs", "Alcohol_n15")

DOSE_COL = "Session_Type"
DOSES = ["Saline", "Alcohol_Low", "Alcohol_High"]
DOSE_LABELS = {"Saline": "Saline", "Alcohol_Low": "0.5 g/kg", "Alcohol_High": "1.0 g/kg"}

THRESHOLD = 0.65          # same as the rest of the pipeline
SPLIT_S = 900             # first 15 min
EXCLUDE_RATS = []        # behaviour and state occupancy use every rat with all three doses. The histology exclusions
                         # (R18, R21) apply to the photometry analyses only (4.1, MATLAB); pass --exclude R18 R21 for a
                         # sensitivity analysis.
REQUIRE_COMPLETE = True   # drop rats missing any of the three doses
EXPECTED_N = None         # set to an integer to be warned when the analysed n differs

STATES = [1, 2, 3]
STATE_NAMES = {1: "State 1:\nRisk-taking", 2: "State 2:\nRisk-averse", 3: "State 3:\nRepeating"}
CUES = {"reward": "rewDS", "punish": "punDS", "conflict": "Conflict"}
COLORS = {  # pale = saline, mid = 0.5 g/kg, dark = 1.0 g/kg (as in Fig. 5B)
    1: dict(zip(DOSES, ["#E2E4A8", "#C4C742", "#B0B300"])),
    2: dict(zip(DOSES, ["#E6AEE5", "#CA3CC6", "#B500B3"])),
    3: dict(zip(DOSES, ["#A8E2E4", "#42C4C7", "#00B0B3"])),
}


# =========================================================
# 1. LOAD, DERIVE STATE, DEFINE THE ANALYSED RATS
# =========================================================
def load():
    df = pd.read_csv(INPUT_PATH, dtype={"Session_ID": str})
    need = ["Rat", DOSE_COL, "Session_ID", "Trial", "Timestamp", "Cue", "Outcome",
            "Prob_State_1", "Prob_State_2", "Prob_State_3"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing column(s): {missing}")
    df[DOSE_COL] = df[DOSE_COL].astype(str).str.strip()
    unknown = set(df[DOSE_COL]) - set(DOSES)
    if unknown:
        raise ValueError(f"Unrecognised dose labels {unknown}; expected {DOSES}")

    P = df[["Prob_State_1", "Prob_State_2", "Prob_State_3"]].values
    df["State"] = np.where(np.nanmax(P, axis=1) >= THRESHOLD, np.nanargmax(P, axis=1) + 1, np.nan)
    df["Lever_Press"] = df["Outcome"].astype(float)
    return df


def design_report(df):
    print(f"\nFile: {df['Rat'].nunique()} rats, {df.groupby(['Rat', 'Session_ID']).ngroups} sessions, "
          f"{len(df)} trials. Unassigned at {THRESHOLD}: {100 * df['State'].isna().mean():.1f}%")
    if "Sex" in df.columns:
        print("Sex values:", df.drop_duplicates("Rat")["Sex"].value_counts(dropna=False).to_dict())
    sess = df.groupby(["Rat", DOSE_COL])["Session_ID"].nunique().unstack().reindex(columns=DOSES)
    odd = sess[(sess.isna().any(axis=1)) | ((sess.fillna(0) != 2).any(axis=1))]
    if len(odd):
        print("Rats with a dose missing or not exactly 2 sessions (sessions per dose):")
        print(odd.to_string())


def select_rats(df):
    if EXCLUDE_RATS:
        present = [r for r in EXCLUDE_RATS if r in set(df["Rat"])]
        print(f"Excluding {present} (configured: {EXCLUDE_RATS})")
        df = df[~df["Rat"].isin(EXCLUDE_RATS)]
    if REQUIRE_COMPLETE:
        have = df.groupby("Rat")[DOSE_COL].agg(lambda s: set(s))
        incomplete = [r for r, s in have.items() if not set(DOSES) <= s]
        if incomplete:
            print(f"Dropping rats without all three doses: {incomplete}")
            df = df[~df["Rat"].isin(incomplete)]
    n = df["Rat"].nunique()
    flag = "" if EXPECTED_N is None or n == EXPECTED_N else f"   <-- expected n = {EXPECTED_N}; check the rat set (see --exclude)"
    print(f"Rats analysed: {n}{flag}")
    if n == 0:
        sys.exit("No rats left to analyse (none has all three doses after the exclusions). "
                 "Check the input table and --exclude.")
    return df


# =========================================================
# 2. PER-RAT METRICS
# =========================================================
def occupancy(d):
    rows = []
    for (rat, dose), s in d.groupby(["Rat", DOSE_COL]):
        n = int(s["State"].notna().sum())
        for k in STATES:
            rows.append({"Rat": rat, "Dose": dose, "State": k, "N_assigned": n,
                         "Occupancy": np.nan if n == 0 else 100 * (s["State"] == k).sum() / n})
    return pd.DataFrame(rows)


def behaviour(d):
    g = d.groupby(["Rat", DOSE_COL, "Cue"])["Lever_Press"].agg(["mean", "size"]).reset_index()
    g["Press_Rate"] = g["mean"] * 100
    g["Cue"] = g["Cue"].str.lower().map(CUES)
    return g.rename(columns={DOSE_COL: "Dose", "size": "N_trials"})[["Rat", "Dose", "Cue", "N_trials", "Press_Rate"]]


def to_wide(long, index, value):
    return long.pivot(index=index, columns="Dose", values=value).reindex(columns=DOSES).reset_index()


# =========================================================
# 3. TESTS
# =========================================================
def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    run = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok])]):
        run = max(run, (len(ok) - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def dose_tests(wide, measure, epoch):
    """wide: rows = rats, columns = DOSES."""
    w = wide[DOSES].dropna()
    try:
        fr = stats.friedmanchisquare(*[w[d] for d in DOSES]) if len(w) >= 3 else None
        chi, p_fr = (fr.statistic, fr.pvalue) if fr is not None else (np.nan, np.nan)
    except ValueError:
        chi, p_fr = np.nan, np.nan
    comps = [("Saline", "Alcohol_Low"), ("Saline", "Alcohol_High")]
    raw = []
    for a, b in comps:
        try:
            raw.append(stats.wilcoxon(w[a], w[b]).pvalue)
        except ValueError:      # all differences zero
            raw.append(np.nan)
    adj = holm(raw)
    return [{"Epoch": epoch, "Measure": measure, "N_rats": len(w), "Friedman_chi2": chi, "Friedman_p": p_fr,
             "Comparison": f"{a} vs {b}", "Median_diff": float(np.median(w[b] - w[a])) if len(w) else np.nan,
             "Wilcoxon_p": r, "Holm_p": q} for (a, b), r, q in zip(comps, raw, adj)]


# =========================================================
# 4. FIGURE
# =========================================================
def plot_occupancy(occ, title, stem, rng):
    fig, ax = plt.subplots(figsize=(7, 5))
    w = 0.23
    for i, dose in enumerate(DOSES):
        for k in STATES:
            v = occ[(occ["State"] == k) & (occ["Dose"] == dose)]["Occupancy"].dropna()
            if len(v) == 0:
                continue
            x = (k - 1) + (i - 1) * w
            ax.bar(x, v.mean(), w, color=COLORS[k][dose], edgecolor="black", linewidth=1.2, zorder=2)
            ax.errorbar(x, v.mean(), yerr=v.sem() if len(v) > 1 else 0, fmt="none", color="black",
                        capsize=3, linewidth=1.2, zorder=3)
            ax.scatter(rng.normal(x, 0.03, len(v)), v, s=14, color="white", edgecolor="black",
                       linewidth=0.6, zorder=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels([STATE_NAMES[k] for k in STATES])
    ax.set_ylabel("% of classified trials", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_title(title, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="black") for c in ["#DDDDDD", "#999999", "#444444"]]
    ax.legend(handles, [DOSE_LABELS[d] for d in DOSES], frameon=False, title="Dose (pale to dark)",
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    for ext, kw in [("svg", {}), ("png", {"dpi": 300})]:
        plt.savefig(os.path.join(OUTPUT_DIR, f"{stem}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)


# =========================================================
# 5. MAIN
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Alcohol state occupancy (Fig. 5B / S5B).")
    ap.add_argument("--input", default=None, help="Trial table from 5.1 (default: INPUT_PATH)")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="Rat IDs to exclude, or 'none' (default: EXCLUDE_RATS)")
    ap.add_argument("--out", default=None, help="Output folder (default: OUTPUT_DIR)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.input:
        INPUT_PATH = args.input
    if args.exclude is not None:
        EXCLUDE_RATS = [] if [x.lower() for x in args.exclude] == ["none"] else args.exclude
    if args.out:
        OUTPUT_DIR = args.out
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(14)
    print(f"[{VERSION}] input: {INPUT_PATH} | output: {OUTPUT_DIR}")

    df = load()
    design_report(df)
    df = select_rats(df)

    epochs = {"first_15min": df[df["Timestamp"] <= SPLIT_S], "after_15min": df[df["Timestamp"] > SPLIT_S]}
    figs = {"first_15min": ("Alcohol: first 15 min", "FigS5B_StateOccupancy_first15min"),
            "after_15min": ("Alcohol: after the first 15 min", "Fig5B_StateOccupancy_after15min")}
    test_rows, lines = [], [f"Alcohol state occupancy. Threshold {THRESHOLD}; split at {SPLIT_S} s; "
                            f"{df['Rat'].nunique()} rats.",
                            f"Rats analysed: {sorted(df['Rat'].unique())}",
                            f"Excluded by configuration: {EXCLUDE_RATS}"]

    for name, d in epochs.items():
        occ, beh = occupancy(d), behaviour(d)
        occ.to_csv(os.path.join(OUTPUT_DIR, f"occupancy_per_rat_{name}.csv"), index=False)
        beh.to_csv(os.path.join(OUTPUT_DIR, f"behaviour_per_rat_{name}.csv"), index=False)
        ow = occ.pivot_table(index=["Rat", "State"], columns="Dose", values="Occupancy", dropna=False)\
                .reindex(columns=DOSES).reset_index()
        ow.to_csv(os.path.join(OUTPUT_DIR, f"occupancy_wide_{name}.csv"), index=False)
        bw = beh.pivot_table(index=["Rat", "Cue"], columns="Dose", values="Press_Rate", dropna=False)\
                .reindex(columns=DOSES).reset_index()
        bw.to_csv(os.path.join(OUTPUT_DIR, f"behaviour_wide_{name}.csv"), index=False)

        lines.append(f"\n=== {name}: {len(d)} trials, {100 * d['State'].isna().mean():.1f}% unassigned ===")
        lines.append("Occupancy (% of assigned trials), mean +/- SEM over rats [n]:")
        for k in STATES:
            cells = []
            for dose in DOSES:
                v = occ[(occ.State == k) & (occ.Dose == dose)]["Occupancy"].dropna()
                cells.append(f"{DOSE_LABELS[dose]} {v.mean():5.1f} +/- {v.sem():4.1f} [{len(v)}]")
            lines.append(f"  State {k}: " + " | ".join(cells))
            test_rows += dose_tests(ow[ow.State == k], f"Occupancy, State {k}", name)
        lines.append("Press rate (%), mean +/- SEM over rats [n]:")
        for cue in CUES.values():
            cells = []
            for dose in DOSES:
                v = beh[(beh.Cue == cue) & (beh.Dose == dose)]["Press_Rate"].dropna()
                cells.append(f"{DOSE_LABELS[dose]} {v.mean():5.1f} +/- {v.sem():4.1f} [{len(v)}]")
            lines.append(f"  {cue:9s}: " + " | ".join(cells))
            test_rows += dose_tests(bw[bw.Cue == cue], f"Press rate, {cue}", name)

        title, stem = figs[name]
        plot_occupancy(occ, title, stem, rng)

    tests = pd.DataFrame(test_rows)
    tests.to_csv(os.path.join(OUTPUT_DIR, "dose_tests.csv"), index=False)
    lines.append("\nNonparametric dose tests (rat = unit; Holm within each state/cue):")
    lines.append(tests.round(4).to_string(index=False))
    txt = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "SUMMARY.txt"), "w") as f:
        f.write(txt)
    print("\n" + txt)
    print(f"\nSaved to {OUTPUT_DIR}/")
