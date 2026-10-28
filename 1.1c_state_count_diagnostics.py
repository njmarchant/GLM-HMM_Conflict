"""
1.1c_state_count_diagnostics.py
-------------------------------
Diagnostic (not in the paper): characterises the EXTRA state(s) that appear in the
4- (and optionally 5-) state solutions.

For each candidate model it reports, per state:
  * threshold-free occupancy  (mean posterior mass, chance = 1/K)
  * thresholded occupancy      (0.65 rule, matching 2.1 / 2.3)
  * per-rat trial counts, and how many rats reach the photometry minimum
  * alignment to the 3-state reference via GLM-weight cosine similarity, so the
    "extra" state is identified unambiguously and we can see whether it is a
    redundant split of an existing state.

Run AFTER 1.1 (needs final_{3,4,5}_state_model_v5.pkl in the repository root).
Reads the same raw CSV and uses the same data prep / sign-flip / threshold as
the rest of the pipeline, so numbers are directly comparable to the figures.
The published comparison of the 3- and 4-state solutions (Fig. S2E-H) is made by
1.1f_four_state_comparison.py.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# 0. CONFIGURATION
# =========================================================
INPUT_DIR        = "Data"          # folder holding the raw CSV
MODEL_DIR        = ""              # folder holding the .pkl files ("" = repository root)
OUTPUT_DIR       = "HMM_Outputs"
INPUT_CSV        = "collated_2026_conflict_data_raw.csv"
MODEL_TEMPLATE   = "final_{k}_state_model_v5.pkl"

REFERENCE_STATES = 3           # the retained model
TEST_STATES      = [4, 5]      # candidates to scrutinise (missing pkls are skipped)

CONFIDENCE_THRESHOLD    = 0.65 # matches 2.1 / 2.3
MIN_TRIALS_PHOTOMETRY   = 5    # a state is "photometry-viable" in a rat if it has >= this many assigned trials

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

WEIGHT_NAMES = ['Rew_Cue', 'Pun_Cue', 'Past_Pellet', 'Past_Shock', 'Bias']


# =========================================================
# 1. DATA PREP  (verbatim from 1.1 / 2.1)
# =========================================================
def prepare_hmm_data(df):
    print("Preparing data: separating by Rat and Session...")
    df = df.sort_values(by=['Rat', 'Session', 'Trial']).copy()
    df['past_pellet'] = (df['Response'] == 'Response-Pellet').astype(int)\
        .groupby([df['Rat'], df['Session']]).shift(1).fillna(0)
    df['past_shock'] = (df['Response'] == 'Response-Shock').astype(int)\
        .groupby([df['Rat'], df['Session']]).shift(1).fillna(0)
    df['rew_cue_val'] = (df['Cue'] == 'reward') | (df['Cue'] == 'conflict')
    df['pun_cue_val'] = (df['Cue'] == 'punish') | (df['Cue'] == 'conflict')
    df['bias'] = 1.0
    df['Lever_Press'] = df['Outcome'].astype(int)

    datas, inputs, meta = [], [], []
    for (rat, session), sd in df.groupby(['Rat', 'Session']):
        if len(sd) < 5:
            continue
        Y = sd['Lever_Press'].values.reshape(-1, 1).astype(int)
        X = sd[['rew_cue_val', 'pun_cue_val', 'past_pellet', 'past_shock', 'bias']].astype(float).values
        datas.append(Y); inputs.append(X)
        meta.append({'Rat': rat, 'Session': session, 'Indices': sd.index})
    print(f"Successfully processed {len(datas)} independent sessions.")
    return datas, inputs, meta, df


# =========================================================
# 2. HELPERS
# =========================================================
def get_press_weights(hmm):
    """Sign-flipped GLM weights (positive = drive to press), matching 1.1's export."""
    raw = hmm.observations.Wk
    W = raw[:, 0, :] * -1 if raw.ndim == 3 else raw.squeeze() * -1
    return np.atleast_2d(W)


def infer_posteriors(hmm, datas, inputs, meta, K, base_df):
    """Attach Prob_State_1..K and a thresholded Inferred_State to a copy of base_df."""
    df = base_df.copy()
    prob_cols = [f'Prob_State_{i+1}' for i in range(K)]
    for c in prob_cols:
        df[c] = np.nan
    df['Inferred_State'] = np.nan

    for data, inp, m in zip(datas, inputs, meta):
        state_probs, _, _ = hmm.expected_states(data, input=inp)   # (T, K), smoothed posterior
        idx = m['Indices']
        for i, c in enumerate(prob_cols):
            df.loc[idx, c] = state_probs[:, i]
        best = np.argmax(state_probs, axis=1)
        conf = np.max(state_probs, axis=1)
        df.loc[idx, 'Inferred_State'] = np.where(conf >= CONFIDENCE_THRESHOLD, best + 1, np.nan)
    return df, prob_cols


def align_to_reference(W_test, W_ref):
    """
    Greedy (Hungarian if scipy present) matching of reference states to test states
    by cosine similarity of GLM weights. Returns:
      match     : dict {ref_state (1-based): test_state (1-based)}
      extras    : list of test states (1-based) not matched to any reference state
      cos       : cosine similarity matrix, shape (K_test, K_ref)
    """
    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    Kt, Kr = W_test.shape[0], W_ref.shape[0]
    cosmat = np.array([[cos(W_test[t], W_ref[r]) for r in range(Kr)] for t in range(Kt)])  # (Kt, Kr)

    try:
        from scipy.optimize import linear_sum_assignment
        # rows = reference, cols = test; maximise similarity -> minimise (1 - cos)
        r_ind, t_ind = linear_sum_assignment(1.0 - cosmat.T)
        match = {int(r + 1): int(t + 1) for r, t in zip(r_ind, t_ind)}
    except Exception:
        match, used = {}, set()
        for r in range(Kr):
            order = np.argsort(-cosmat[:, r])
            for t in order:
                if t not in used:
                    match[r + 1] = int(t + 1); used.add(t); break

    extras = [t + 1 for t in range(Kt) if (t + 1) not in match.values()]
    return match, extras, cosmat


# =========================================================
# 3. PER-MODEL DIAGNOSTIC
# =========================================================
def diagnose(hmm, K, datas, inputs, meta, base_df, W_ref, ref_K):
    df, prob_cols = infer_posteriors(hmm, datas, inputs, meta, K, base_df)
    rats = sorted(df['Rat'].dropna().unique())
    n_rats = len(rats)

    # ---- threshold-free occupancy: per-rat mean posterior, then group mean +/- SEM ----
    per_rat_tf = df.groupby('Rat')[prob_cols].mean()          # (rats, K), rows ~sum to 1
    tf_mean = per_rat_tf.mean().values * 100
    tf_sem = per_rat_tf.sem().values * 100

    # ---- thresholded occupancy (matches 2.3): per-rat % of ASSIGNED trials in each state ----
    th_occ = np.full((n_rats, K), np.nan)
    counts = np.zeros((n_rats, K), dtype=int)                 # raw assigned-trial counts per rat/state
    for ri, rat in enumerate(rats):
        rd = df[df['Rat'] == rat]
        assigned = rd['Inferred_State'].notna().sum()
        for s in range(1, K + 1):
            c = int((rd['Inferred_State'] == s).sum())
            counts[ri, s - 1] = c
            if assigned > 0:
                th_occ[ri, s - 1] = c / assigned * 100
    th_mean = np.nanmean(th_occ, axis=0)
    th_sem = np.nanstd(th_occ, axis=0, ddof=1) / np.sqrt(n_rats)

    median_trials = np.median(counts, axis=0)
    n_viable = (counts >= MIN_TRIALS_PHOTOMETRY).sum(axis=0)   # rats with enough trials for photometry
    n_expressing = (counts > 0).sum(axis=0)

    # ---- alignment to reference ----
    W_test = get_press_weights(hmm)
    match, extras, cosmat = align_to_reference(W_test, W_ref)
    # best-matching reference state for each test state (for the "split" story)
    best_ref = np.argmax(cosmat, axis=1) + 1
    best_ref_cos = np.max(cosmat, axis=1)

    # ---- assemble summary table ----
    rows = []
    for s in range(1, K + 1):
        rows.append({
            'State': s,
            'ThreshFree_Occ_%': round(tf_mean[s - 1], 2),
            'ThreshFree_SEM': round(tf_sem[s - 1], 2),
            'Thresholded_Occ_%': round(th_mean[s - 1], 2),
            'Thresholded_SEM': round(th_sem[s - 1], 2),
            'Median_trials_per_rat': int(median_trials[s - 1]),
            f'N_rats_>= {MIN_TRIALS_PHOTOMETRY}_trials': int(n_viable[s - 1]),
            'N_rats_expressing': int(n_expressing[s - 1]),
            'Best_match_ref_state': int(best_ref[s - 1]),
            'Best_match_cosine': round(float(best_ref_cos[s - 1]), 3),
            'Is_EXTRA_state': s in extras,
        })
    summary = pd.DataFrame(rows)

    cos_df = pd.DataFrame(cosmat,
                          index=[f'Test_State_{i+1}' for i in range(K)],
                          columns=[f'Ref_State_{j+1}' for j in range(ref_K)]).round(3)
    return summary, cos_df, match, extras, n_rats


def make_figure(summary, K, extras, n_rats, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    states = summary['State'].values
    colors = ['#cc2b2b' if s in extras else '#23395B' for s in states]

    ax = axes[0]
    ax.bar(states, summary['ThreshFree_Occ_%'], yerr=summary['ThreshFree_SEM'],
           color=colors, edgecolor='black', capsize=4)
    ax.axhline(100.0 / K, color='gray', linestyle=':', linewidth=1.2)
    ax.text(states[-1] + 0.3, 100.0 / K, f'chance\n({100.0/K:.0f}%)',
            color='gray', fontsize=8, va='center')
    ax.set_title(f'{K}-state model: threshold-free occupancy', fontweight='bold', fontsize=11)
    ax.set_xlabel('State'); ax.set_ylabel('Mean posterior occupancy (%)')
    ax.set_xticks(states)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[1]
    ax.bar(states, summary[f'N_rats_>= {MIN_TRIALS_PHOTOMETRY}_trials'],
           color=colors, edgecolor='black')
    ax.axhline(n_rats, color='gray', linestyle=':', linewidth=1.0)
    ax.set_title(f'Rats with >= {MIN_TRIALS_PHOTOMETRY} assigned trials (of {n_rats})',
                 fontweight='bold', fontsize=11)
    ax.set_xlabel('State'); ax.set_ylabel('Number of rats')
    ax.set_xticks(states); ax.set_ylim(0, n_rats + 1)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    if extras:
        fig.suptitle(f"Extra state(s) shown in red: State {extras}", fontsize=10, y=1.02)
    plt.tight_layout()
    base = os.path.join(output_dir, f'StateCount_Diagnostic_{K}state')
    plt.savefig(base + '.svg', bbox_inches='tight')
    plt.savefig(base + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# =========================================================
# 4. EXECUTION
# =========================================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df_raw = pd.read_csv(os.path.join(INPUT_DIR, INPUT_CSV), dtype={"Session": str})
    datas, inputs, meta, base_df = prepare_hmm_data(df_raw)

    # Reference model + its weights
    ref_path = os.path.join(MODEL_DIR, MODEL_TEMPLATE.format(k=REFERENCE_STATES))
    with open(ref_path, "rb") as f:
        ref_hmm = pickle.load(f)
    W_ref = get_press_weights(ref_hmm)
    print(f"\nLoaded {REFERENCE_STATES}-state reference model from {ref_path}")

    for K in TEST_STATES:
        path = os.path.join(MODEL_DIR, MODEL_TEMPLATE.format(k=K))
        if not os.path.exists(path):
            print(f"\n[skip] {path} not found."); continue

        print(f"\n{'='*60}\n{K}-STATE MODEL DIAGNOSTIC\n{'='*60}")
        with open(path, "rb") as f:
            hmm = pickle.load(f)

        summary, cos_df, match, extras, n_rats = diagnose(
            hmm, K, datas, inputs, meta, base_df, W_ref, REFERENCE_STATES)

        print("\n-- Occupancy & photometry viability --")
        print(summary.to_string(index=False))
        print("\n-- Weight cosine similarity to reference states --")
        print(cos_df.to_string())
        print(f"\nReference->test state match: {match}")
        print(f"EXTRA state(s) (not matched to any reference state): {extras}")
        for e in extras:
            row = summary[summary.State == e].iloc[0]
            print(f"  State {e}: threshold-free occ = {row['ThreshFree_Occ_%']}%, "
                  f"{row[f'N_rats_>= {MIN_TRIALS_PHOTOMETRY}_trials']}/{n_rats} rats photometry-viable, "
                  f"nearest reference = State {row['Best_match_ref_state']} "
                  f"(cosine {row['Best_match_cosine']})")

        summary.to_csv(os.path.join(OUTPUT_DIR, f'StateCount_Diagnostic_{K}state_summary.csv'), index=False)
        cos_df.to_csv(os.path.join(OUTPUT_DIR, f'StateCount_Diagnostic_{K}state_cosine.csv'))
        make_figure(summary, K, extras, n_rats, OUTPUT_DIR)

    print("\nDone. Tables and figures written to 'HMM_Outputs/'.")
