"""
1.1f_four_state_comparison.py
-----------------------------
Compares the retained 3-state model with the 4-state solution (Supplementary Figure 2E-H). Both models are the
saved fits from 1.1 and are applied to the full Phase 3 dataset; nothing is refitted.

  E  Cosine similarity between the GLM weight vectors of the 3-state states (rows) and the 4-state states (columns).
     States are matched by optimal (Hungarian) assignment; matched cells are bold and outlined. The 4-state states
     are ordered as: match of Risk-taking, match of Risk-averse, match of Repeating, then the unmatched (extra) state.
  F  Correspondence of trial assignments. Each trial is assigned in each model to its most probable state if that
     smoothed posterior is >= CONFIDENCE_THRESHOLD (as in 2.1), otherwise "Unassigned". Rows: 3-state assignment;
     columns: 4-state assignment; cells: % of the row.
  G  GLM weights of the 4-state solution (positive = drive to press). Weights are selected BY NAME from the fit
     order (rewDS, punDS, past pellet, past shock, bias), so the display order cannot swap the history terms.
  H  Percentage of rewDS, punDS and conflict trials completed in each 4-state state (assigned trials only;
     mean +/- SEM across rats, points = rats). The per-rat table is exported for statistics in Prism.

Inputs   Data/collated_2026_conflict_data_raw.csv, final_3_state_model_v5.pkl, final_4_state_model_v5.pkl
Outputs  HMM_Outputs/FourState/
           SuppFig2E_Cosine_3v4.{svg,png}, SuppFig2F_Assignment_3v4.{svg,png},
           SuppFig2G_Weights_4state.{svg,png}, SuppFig2H_Behaviour_4state.{svg,png}
           cosine_3v4.csv, state_matching.csv, assignment_counts.csv, assignment_row_percent.csv,
           4_State_GLM_Weights_named.csv, behaviour_4state_per_rat.csv, behaviour_4state_wide.csv, SUMMARY.txt
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

# =========================================================
# 0. CONFIGURATION
# =========================================================
VERSION              = "1.1f v1"
INPUT_PATH           = os.path.join("Data", "collated_2026_conflict_data_raw.csv")
MODEL_DIR            = ""                                  # models live in the repository root ("" = here)
REF_MODEL            = "final_3_state_model_v5.pkl"
TEST_MODEL           = "final_4_state_model_v5.pkl"
OUTPUT_DIR           = os.path.join("HMM_Outputs", "FourState")

CONFIDENCE_THRESHOLD = 0.65                                # same as 2.1
MIN_TRIALS           = 5                                   # same session exclusion as 1.1

# Names. REF_NAMES follow the 3-state model's state order (1, 2, 3).
# FOUR_STATE_NAMES are in display order: match of state 1, match of state 2, match of state 3, extra state.
# Use the same names in every panel and in the legend (e.g. "Repeating 1" / "Repeating 2" instead of "Off-task").
REF_NAMES            = ["Risk-Taking", "Risk-Averse", "Repeating"]
FOUR_STATE_NAMES     = ["Risk-Taking", "Risk-Averse", "Repeating 1", "Repeating 2"]   # as in Fig. S2E-H
UNASSIGNED           = "Unassigned"

# Colours: states in display order (as in Fig. 3 and the 4-state panels), cues as in Fig. 3E.
FOUR_STATE_COLORS    = ["#B0B300", "#B500B3", "#00B0B3", "#1E9BD7"]
CUE_COLORS           = {"rewDS": (0.12, 0.80, 0.01), "punDS": (0.87, 0.01, 0.12), "Conf": (0.01, 0.11, 0.86)}

# GLM inputs in FIT order (as built by prepare_hmm_data) and the order in which panel G shows them.
FIT_ORDER            = ["rewDS", "punDS", "Past Pellet", "Past Shock", "Baseline Bias"]
DISPLAY_ORDER        = ["rewDS", "punDS", "Past Shock", "Past Pellet", "Baseline Bias"]   # as in Fig. 3B


# =========================================================
# 1. DATA PREPARATION (identical to 1.1 / 2.1)
# =========================================================
def prepare_hmm_data(df):
    df = df.sort_values(by=['Rat', 'Session', 'Trial']).reset_index(drop=True)   # index = row position
    df['past_pellet'] = (df['Response'] == 'Response-Pellet').astype(int).groupby([df['Rat'], df['Session']]).shift(1).fillna(0)
    df['past_shock'] = (df['Response'] == 'Response-Shock').astype(int).groupby([df['Rat'], df['Session']]).shift(1).fillna(0)
    df['rew_cue_val'] = (df['Cue'] == 'reward') | (df['Cue'] == 'conflict')
    df['pun_cue_val'] = (df['Cue'] == 'punish') | (df['Cue'] == 'conflict')
    df['bias'] = 1.0
    df['Lever_Press'] = df['Outcome'].astype(int)
    datas, inputs, meta = [], [], []
    for (rat, session), sd in df.groupby(['Rat', 'Session']):
        if len(sd) < MIN_TRIALS:
            continue
        datas.append(sd['Lever_Press'].values.reshape(-1, 1).astype(int))
        inputs.append(sd[['rew_cue_val', 'pun_cue_val', 'past_pellet', 'past_shock', 'bias']].astype(float).values)
        meta.append({'Rat': rat, 'Session': session, 'Indices': sd.index})
    print(f"Processed {len(datas)} sessions, {sum(len(d) for d in datas)} trials (paper: 233 sessions, 23,126 trials).")
    return datas, inputs, meta, df


# =========================================================
# 2. HELPERS
# =========================================================
def load_model(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Put the saved model files from 1.1 in MODEL_DIR.")
    with open(path, "rb") as f:
        return pickle.load(f)


def press_weights(hmm):
    """GLM weights with positive = drive to press, shape (K, 5), columns in FIT_ORDER."""
    raw = hmm.observations.Wk
    return np.atleast_2d(raw[:, 0, :] * -1 if raw.ndim == 3 else raw.squeeze() * -1)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def assign_states(hmm, datas, inputs, meta, n_rows):
    """Smoothed posterior per trial -> assigned state (1-based) or NaN below the threshold."""
    out = np.full(n_rows, np.nan)
    for data, inp, m in zip(datas, inputs, meta):
        p, _, _ = hmm.expected_states(data, input=inp)
        out[m['Indices']] = np.where(p.max(axis=1) >= CONFIDENCE_THRESHOLD, p.argmax(axis=1) + 1, np.nan)
    return out


def save(fig, stem):
    for ext, kw in [("svg", {}), ("png", {"dpi": 300})]:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{stem}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)


# =========================================================
# 3. MAIN
# =========================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[{VERSION}]")
    df = pd.read_csv(INPUT_PATH, dtype={'Session': str})
    datas, inputs, meta, df = prepare_hmm_data(df)

    ref, test = load_model(REF_MODEL), load_model(TEST_MODEL)
    W3, W4 = press_weights(ref), press_weights(test)
    if W3.shape[0] != 3 or W4.shape[0] != 4:
        raise SystemExit(f"Expected a 3-state and a 4-state model; got {W3.shape[0]} and {W4.shape[0]} states.")

    # ---------- E: cosine similarity and matching ----------
    C = np.array([[cosine(W3[i], W4[j]) for j in range(4)] for i in range(3)])     # (3 ref, 4 test)
    r_ind, c_ind = linear_sum_assignment(-C)                                          # maximise similarity
    match = dict(zip(r_ind, c_ind))                                                   # ref state -> test state (0-based)
    extra = [j for j in range(4) if j not in match.values()]
    order = [match[0], match[1], match[2]] + extra                                    # 4-state display order
    names4 = {order[k]: FOUR_STATE_NAMES[k] for k in range(4)}
    Cd = C[:, order]

    cos_df = pd.DataFrame(Cd, index=REF_NAMES, columns=FOUR_STATE_NAMES).round(3)
    cos_df.to_csv(os.path.join(OUTPUT_DIR, "cosine_3v4.csv"))
    match_df = pd.DataFrame({"Display_name": FOUR_STATE_NAMES,
                             "Model_state_index_4": [j + 1 for j in order],
                             "Matched_3state": REF_NAMES + ["(extra)"],
                             "Cosine_to_match": [round(C[i, match[i]], 3) for i in range(3)] + [np.nan],
                             "Closest_3state": [REF_NAMES[int(np.argmax(C[:, j]))] for j in order],
                             "Cosine_to_closest": [round(C[:, j].max(), 3) for j in order]})
    match_df.to_csv(os.path.join(OUTPUT_DIR, "state_matching.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(Cd, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(3):
        for k in range(4):
            matched = (k == i)                                                         # matched pairs sit on the diagonal
            ax.text(k, i, f"{Cd[i, k]:.3f}", ha="center", va="center", fontsize=11,
                    color="white" if abs(Cd[i, k]) > 0.6 else "black", fontweight="bold" if matched else "normal")
            if matched:
                ax.add_patch(Rectangle((k - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.5))
    ax.set_xticks(range(4)); ax.set_xticklabels(FOUR_STATE_NAMES)
    ax.set_yticks(range(3)); ax.set_yticklabels(REF_NAMES, rotation=90, va="center")
    ax.set_xlabel("4-state model"); ax.set_ylabel("3-state model")
    ax.set_title("3-state vs 4-state solution")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("Cosine similarity of GLM weights"); cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    save(fig, "SuppFig2E_Cosine_3v4")

    # ---------- F: trial assignment correspondence ----------
    s3 = assign_states(ref, datas, inputs, meta, len(df))
    s4 = assign_states(test, datas, inputs, meta, len(df))
    decoded = np.zeros(len(df), bool)
    for m in meta:
        decoded[m['Indices']] = True
    lab3 = pd.Series([REF_NAMES[int(x) - 1] if not np.isnan(x) else UNASSIGNED for x in s3])[decoded]
    lab4 = pd.Series([names4[int(x) - 1] if not np.isnan(x) else UNASSIGNED for x in s4])[decoded]
    rows, cols = REF_NAMES + [UNASSIGNED], FOUR_STATE_NAMES + [UNASSIGNED]
    counts = (pd.crosstab(lab3.rename("3-state"), lab4.rename("4-state"))
                .reindex(index=rows, columns=cols, fill_value=0))
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    counts.to_csv(os.path.join(OUTPUT_DIR, "assignment_counts.csv"))
    pct.round(2).to_csv(os.path.join(OUTPUT_DIR, "assignment_row_percent.csv"))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(pct.values, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    for i in range(len(rows)):
        rmax = np.nanargmax(pct.values[i])
        for k in range(len(cols)):
            v = pct.values[i, k]
            ax.text(k, i, f"{v:.2f}%", ha="center", va="center", fontsize=10,
                    color="white" if v > 60 else "black", fontweight="bold" if k == rmax else "normal")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, rotation=90, va="center")
    ax.set_xlabel("4-State Model Assignment"); ax.set_ylabel("3-State Model Assignment")
    ax.set_title("% of 3-state assignments per state of the 4-state solution")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    save(fig, "SuppFig2F_Assignment_3v4")

    # ---------- G: 4-state weights (selected by name, shown in DISPLAY_ORDER) ----------
    W4d = pd.DataFrame(W4[order], index=FOUR_STATE_NAMES, columns=FIT_ORDER)
    W4d.round(4).to_csv(os.path.join(OUTPUT_DIR, "4_State_GLM_Weights_named.csv"))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x, w = np.arange(len(DISPLAY_ORDER)), 0.2
    for k, name in enumerate(FOUR_STATE_NAMES):
        ax.bar(x + (k - 1.5) * w, W4d.loc[name, DISPLAY_ORDER].values, w, color=FOUR_STATE_COLORS[k],
               edgecolor="black", linewidth=1.0, label=f"State {k + 1}: {name}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" ", "\n") for n in DISPLAY_ORDER])
    ax.set_ylabel("GLM weight (β)\n(positive = drive to press)")
    ax.set_title("GLM weights for the 4-state solution")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    save(fig, "SuppFig2G_Weights_4state")

    # ---------- H: behaviour per 4-state state ----------
    d = df.loc[decoded].copy()
    d["State4"] = lab4.values
    d = d[d["State4"] != UNASSIGNED]
    d["Cue_Label"] = d["Cue"].str.lower().map({"reward": "rewDS", "punish": "punDS", "conflict": "Conf"})
    per_rat = (d.groupby(["Rat", "State4", "Cue_Label"])["Lever_Press"].agg(["mean", "size"]).reset_index()
                 .rename(columns={"mean": "P_Press", "size": "N_trials"}))
    per_rat["Percent_Completed"] = per_rat["P_Press"] * 100
    per_rat.to_csv(os.path.join(OUTPUT_DIR, "behaviour_4state_per_rat.csv"), index=False)
    full = pd.MultiIndex.from_product([sorted(d["Rat"].unique()), FOUR_STATE_NAMES], names=["Rat", "State4"])
    wide = (per_rat.pivot_table(index=["Rat", "State4"], columns="Cue_Label", values="Percent_Completed")
                   .reindex(full).reindex(columns=["rewDS", "punDS", "Conf"]).reset_index())
    wide.to_csv(os.path.join(OUTPUT_DIR, "behaviour_4state_wide.csv"), index=False)

    rng = np.random.default_rng(14)
    fig, ax = plt.subplots(figsize=(10, 6))
    bw = 0.26
    for k, name in enumerate(FOUR_STATE_NAMES):
        for c, cue in enumerate(["rewDS", "punDS", "Conf"]):
            v = per_rat[(per_rat.State4 == name) & (per_rat.Cue_Label == cue)]["Percent_Completed"].dropna()
            if v.empty:
                continue
            xx = k + (c - 1) * bw
            ax.bar(xx, v.mean(), bw, color=CUE_COLORS[cue], edgecolor="black", linewidth=1.2,
                   label=cue if k == 0 else None, zorder=2)
            ax.errorbar(xx, v.mean(), yerr=v.sem() if len(v) > 1 else 0, fmt="none", color="black", capsize=3, zorder=3)
            ax.scatter(rng.normal(xx, 0.03, len(v)), v, s=14, color="white", edgecolor="black", linewidth=0.7, zorder=4)
    ax.set_xticks(range(4)); ax.set_xticklabels(FOUR_STATE_NAMES)
    ax.set_ylim(0, 105); ax.set_ylabel("Percentage completed"); ax.set_xlabel("Inferred State")
    ax.set_title("Behavioral profile by latent state (4-state solution)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "SuppFig2H_Behaviour_4state")

    # ---------- summary ----------
    means = per_rat.groupby(["State4", "Cue_Label"])["Percent_Completed"].agg(["mean", "sem", "count"]).round(1)
    lines = [f"[{VERSION}] 3-state vs 4-state comparison. Threshold {CONFIDENCE_THRESHOLD}; "
             f"{int(decoded.sum())} trials, {d['Rat'].nunique()} rats.",
             "\nE. Cosine similarity (rows: 3-state; columns: 4-state, display order):", cos_df.to_string(),
             "\nState matching (4-state model index -> display name):", match_df.to_string(index=False),
             "\nF. Assignment correspondence (% of row):", pct.round(2).to_string(),
             "\nF. Assignment counts:", counts.to_string(),
             "\nG. 4-state weights (fit order; positive = drive to press):", W4d.round(3).to_string(),
             "\nH. Percentage completed per state and cue (mean, SEM, n rats):", means.to_string()]
    txt = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "SUMMARY.txt"), "w") as f:
        f.write(txt)
    print(txt)
    print(f"\nSaved to {OUTPUT_DIR}/")
