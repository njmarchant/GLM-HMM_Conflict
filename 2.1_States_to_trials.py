"""
2.1_States_to_trials.py
-----------------------
Decodes the latent state of every Phase 3 trial with the frozen 3-state model.

Each Rat x Session is passed through hmm.expected_states (forward-backward smoothing,
p(z_t | y_1:T)). A trial is assigned to its most probable state if that probability is
at least CONFIDENCE_THRESHOLD (0.65); otherwise Inferred_State is left empty.

Input   Data/collated_2026_conflict_data_raw.csv, final_3_state_model_v5.pkl
Output  Data/collated_2026_conflict_data_with_Probabilities_065_Past_PelletShock_v5.csv
        = the input columns (including the photometry columns, passed through unchanged)
          + Prob_State_1..3, State_Confidence, Inferred_State, and the model inputs.
        This table is the input to 2.2, 2.3, 2.4, 3.1, 3.2, 4.1 and the MATLAB
        state-sorted photometry analyses.

If the output file already exists and contains columns this script does not produce
(e.g. values merged in by hand), the script stops rather than overwrite them.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# =========================================================
# 0. CONFIGURATION
# =========================================================
INPUT_PATH  = os.path.join("Data", "collated_2026_conflict_data_raw.csv")
MODEL_PATH  = "final_3_state_model_v5.pkl"
OUTPUT_PATH = os.path.join("Data", "collated_2026_conflict_data_with_Probabilities_065_Past_PelletShock_v5.csv")
CONFIDENCE_THRESHOLD = 0.65
MIN_TRIALS = 5                 # same session filter as 1.1
N_STATES = 3


def prepare_hmm_data(df):
    """Same inputs, order and session filter as 1.1_model_fit.py."""
    print("Preparing data: separating by Rat and Session...")
    df = df.sort_values(by=['Rat', 'Session', 'Trial']).copy()
    g = [df['Rat'], df['Session']]
    df['Lever_Press'] = df['Outcome'].astype(int)          # 1 = press, 0 = omission
    df['past_pellet'] = (df['Response'] == 'Response-Pellet').astype(int).groupby(g).shift(1).fillna(0)
    df['past_shock'] = (df['Response'] == 'Response-Shock').astype(int).groupby(g).shift(1).fillna(0)
    df['rew_cue_val'] = (df['Cue'] == 'reward') | (df['Cue'] == 'conflict')
    df['pun_cue_val'] = (df['Cue'] == 'punish') | (df['Cue'] == 'conflict')
    df['bias'] = 1.0

    datas, inputs, session_meta = [], [], []
    for (rat, session), sd in df.groupby(['Rat', 'Session']):
        if len(sd) < MIN_TRIALS:
            continue
        datas.append(sd['Lever_Press'].values.reshape(-1, 1).astype(int))
        inputs.append(sd[['rew_cue_val', 'pun_cue_val', 'past_pellet', 'past_shock', 'bias']].astype(float).values)
        session_meta.append({'Rat': rat, 'Session': session, 'Indices': sd.index})
    print(f"Successfully processed {len(datas)} independent sessions.")
    return datas, inputs, session_meta, df


if __name__ == "__main__":
    print(f"Loading {INPUT_PATH} ...")
    raw = pd.read_csv(INPUT_PATH, dtype={"Session": str})
    datas, inputs, session_meta, df = prepare_hmm_data(raw)

    new_cols = ['Inferred_State', 'State_Confidence'] + [f'Prob_State_{k+1}' for k in range(N_STATES)]
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH, nrows=1).columns
        foreign = [c for c in existing if c not in set(df.columns) | set(new_cols)]
        if foreign:
            sys.exit(f"STOP: {OUTPUT_PATH} contains columns this script does not produce:\n  {foreign}\n"
                     "They were probably added after 2.1. Move or rename that file (or merge those columns "
                     "into the raw input) before re-running.")

    print(f"Loading frozen model {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        hmm = pickle.load(f)

    print(f"Inferring state probabilities (threshold {CONFIDENCE_THRESHOLD})...")
    for c in new_cols:
        df[c] = np.nan
    for data, inp, meta in zip(datas, inputs, session_meta):
        state_probs, _, _ = hmm.expected_states(data, input=inp)    # smoothed posterior (T, K)
        best = np.argmax(state_probs, axis=1)
        conf = np.max(state_probs, axis=1)
        idx = meta['Indices']
        df.loc[idx, 'Inferred_State'] = np.where(conf >= CONFIDENCE_THRESHOLD, best + 1, np.nan)
        df.loc[idx, 'State_Confidence'] = conf
        for k in range(N_STATES):
            df.loc[idx, f'Prob_State_{k+1}'] = state_probs[:, k]

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"Saved state probabilities and labels to '{OUTPUT_PATH}'")

    decoded = df['Prob_State_1'].notna()
    n_dec = int(decoded.sum())
    n_un = int((decoded & df['Inferred_State'].isna()).sum())
    print(f"\nDecoded trials: {n_dec} of {len(df)} ({len(datas)} sessions, {df['Rat'].nunique()} rats)")
    print(f"Unassigned at {CONFIDENCE_THRESHOLD}: {n_un} ({100 * n_un / n_dec:.1f}%)")
