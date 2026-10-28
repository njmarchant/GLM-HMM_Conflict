"""
5.1_alcohol_states_to_trials.py
-------------------------------
Decodes latent states in the acute-alcohol test sessions (saline, 0.5 g/kg,
1.0 g/kg) with the FROZEN Phase 3 three-state GLM-HMM. Nothing is refitted.

Same procedure as 2.1_States_to_trials.py: full sessions are passed through
hmm.expected_states() (forward-backward smoothing), and each trial is assigned to
its most probable state only if that probability is >= CONFIDENCE_THRESHOLD.

Sessions are decoded in full. The first 15 minutes of each test session are analysed
separately (Fig. S5); that split is applied AFTER decoding, in 5.2 / 5.3, using the
Timestamp column carried through from the input file.

Inputs
  Data/collated_2026_conflict_alcohol_raw.csv   (from 5.0) one row per trial; needs columns
      Rat, Session_ID, Trial, Cue, Outcome, Response
      (+ dose / time-in-session columns, which are passed through unchanged)
  final_3_state_model_v5.pkl                    frozen model from 1.1_model_fit.py
Output
  Data/collated_2026_conflict_alcohol_with_Probabilities_065.csv
      input columns + Prob_State_1..3, State_Confidence, Inferred_State
"""

import argparse
import os
import pickle
import time
import numpy as np
import pandas as pd

# =========================================================
# CONFIGURATION
# =========================================================
VERSION = "5.1 v2"
INPUT_PATH = os.path.join("Data", "collated_2026_conflict_alcohol_raw.csv")
MODEL_PATH = "final_3_state_model_v5.pkl"
OUTPUT_PATH = os.path.join("Data", "collated_2026_conflict_alcohol_with_Probabilities_065.csv")

SESSION_COL = "Session_ID"     # must be unique per physical test session within a rat
CONFIDENCE_THRESHOLD = 0.65    # same as 2.1
MIN_TRIALS = 5                 # sessions shorter than this are skipped (same as 1.1 / 2.1)
MAX_TRIALS_EXPECTED = 100      # sessions have 100 trials; more suggests merged sessions

PELLET, SHOCK = "Response-Pellet", "Response-Shock"
VALID_CUES = {"reward", "punish", "conflict"}


# =========================================================
# 1. INPUT CHECKS (fail loudly instead of decoding with silently wrong inputs)
# =========================================================
def validate_input(df):
    required = ["Rat", SESSION_COL, "Trial", "Cue", "Outcome", "Response"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required column(s): {missing}")

    # Cue labels: unrecognised labels would give all-zero cue regressors.
    bad_cues = set(df["Cue"].dropna().unique()) - VALID_CUES
    if bad_cues:
        raise ValueError(f"Unrecognised Cue labels {bad_cues}; expected {VALID_CUES}.")

    # A press always produced an outcome, so (Response is pellet or shock) must equal
    # (Outcome == 1). If not, the history regressors do not mean what the model assumes.
    pressed = df["Outcome"].astype(int) == 1
    has_outcome = df["Response"].isin([PELLET, SHOCK])
    n_mismatch = int((pressed != has_outcome).sum())
    if n_mismatch:
        raise ValueError(
            f"{n_mismatch} trials where Outcome and Response disagree "
            f"(expected Response in {{'{PELLET}', '{SHOCK}'}} exactly when Outcome == 1). "
            f"Response labels found: {sorted(df['Response'].dropna().unique())}")

    # Duplicate trial numbers within a Rat x session indicate that two physical
    # sessions share one ID, so history regressors would cross sessions.
    dup = df.duplicated(["Rat", SESSION_COL, "Trial"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicated (Rat, {SESSION_COL}, Trial) rows: "
                         f"{SESSION_COL} is not unique per session.")

    sizes = df.groupby(["Rat", SESSION_COL]).size()
    print(f"{df['Rat'].nunique()} rats, {len(sizes)} sessions, {len(df)} trials. "
          f"Trials per session: min {sizes.min()}, median {sizes.median():.0f}, max {sizes.max()}.")
    print(f"Rats: {sorted(df['Rat'].unique())}")
    if sizes.max() > MAX_TRIALS_EXPECTED:
        print(f"WARNING: a session has {sizes.max()} trials (> {MAX_TRIALS_EXPECTED}); "
              f"check that {SESSION_COL} is unique per session.")

    if "Timestamp" not in df.columns:
        print("WARNING: no Timestamp column. 5.2 / 5.3 need the trial start time to split "
              "off the first 15 minutes.")


# =========================================================
# 2. DATA PREPARATION (identical covariates and order to 1.1 / 2.1)
# =========================================================
def prepare_hmm_data(df):
    df = df.sort_values(by=["Rat", SESSION_COL, "Trial"]).copy()
    grp = [df["Rat"], df[SESSION_COL]]

    df["past_pellet"] = (df["Response"] == PELLET).astype(int).groupby(grp).shift(1).fillna(0)
    df["past_shock"] = (df["Response"] == SHOCK).astype(int).groupby(grp).shift(1).fillna(0)
    df["rew_cue_val"] = (df["Cue"] == "reward") | (df["Cue"] == "conflict")
    df["pun_cue_val"] = (df["Cue"] == "punish") | (df["Cue"] == "conflict")
    df["bias"] = 1.0
    df["Lever_Press"] = df["Outcome"].astype(int)   # 1 = press, 0 = omission

    datas, inputs, meta, skipped = [], [], [], []
    for (rat, session), sd in df.groupby(["Rat", SESSION_COL]):
        if len(sd) < MIN_TRIALS:
            skipped.append((rat, session, len(sd)))
            continue
        datas.append(sd["Lever_Press"].values.reshape(-1, 1).astype(int))
        inputs.append(sd[["rew_cue_val", "pun_cue_val", "past_pellet",
                          "past_shock", "bias"]].astype(float).values)
        meta.append({"Rat": rat, "Session": session, "Indices": sd.index})

    print(f"Prepared {len(datas)} sessions; skipped {len(skipped)} with < {MIN_TRIALS} trials.")
    for rat, session, n in skipped:
        print(f"   skipped {rat} / {session} ({n} trials)")
    return datas, inputs, meta, df


# =========================================================
# 3. DECODE WITH THE FROZEN MODEL
# =========================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Decode alcohol-test sessions with the frozen 3-state GLM-HMM.")
    ap.add_argument("--input", default=INPUT_PATH, help="trial table written by 5.0 (default: %(default)s)")
    ap.add_argument("--output", default=OUTPUT_PATH, help="table with state probabilities (default: %(default)s)")
    ap.add_argument("--model", default=MODEL_PATH, help="frozen model (default: %(default)s)")
    args = ap.parse_args()
    INPUT_PATH, OUTPUT_PATH, MODEL_PATH = args.input, args.output, args.model

    print(f"[{VERSION}]")
    print(f"Loading dataset {INPUT_PATH} (modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(INPUT_PATH)))})...")
    df = pd.read_csv(INPUT_PATH, dtype={SESSION_COL: str})
    validate_input(df)
    datas, inputs, meta, df = prepare_hmm_data(df)

    print(f"Loading frozen model from {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f:
        hmm = pickle.load(f)

    print(f"Inferring state probabilities (threshold {CONFIDENCE_THRESHOLD})...")
    for c in ["Inferred_State", "State_Confidence", "Prob_State_1", "Prob_State_2", "Prob_State_3"]:
        df[c] = np.nan

    for data, inp, m in zip(datas, inputs, meta):
        state_probs, _, _ = hmm.expected_states(data, input=inp)   # smoothed posterior (T, 3)
        best = np.argmax(state_probs, axis=1)
        conf = np.max(state_probs, axis=1)
        idx = m["Indices"]
        df.loc[idx, "Inferred_State"] = np.where(conf >= CONFIDENCE_THRESHOLD, best + 1, np.nan)
        df.loc[idx, "State_Confidence"] = conf
        for k in range(3):
            df.loc[idx, f"Prob_State_{k+1}"] = state_probs[:, k]

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved probabilities and thresholded states to '{OUTPUT_PATH}'")

    # Summary: only trials that were actually decoded
    decoded = df["Prob_State_1"].notna()
    n_dec = int(decoded.sum())
    n_unassigned = int((decoded & df["Inferred_State"].isna()).sum())
    print(f"\nDecoded trials: {n_dec} of {len(df)}")
    print(f"Unassigned at {CONFIDENCE_THRESHOLD}: {n_unassigned} ({100 * n_unassigned / n_dec:.1f}%)")
