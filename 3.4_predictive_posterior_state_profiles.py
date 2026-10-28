"""
3.4_predictive_posterior_state_profiles.py
------------------------------------------
Are the state-conditioned response rates in Fig. 3D/E inflated because trials are
assigned with SMOOTHED posteriors, which condition on the very choice being summarised?

The smoothed posterior p(z_t | y_1:T) uses choice y_t, so a press on a conflict
trial pulls that trial towards the Risk-taking state and an omission pulls it
towards the Risk-averse state. Response rates computed from those assignments are
polarised partly by construction. This script recomputes the Fig. 3E profiles with
assignments that do NOT use the current choice, and reports the model-implied
emission probabilities, which do not depend on assignment at all.

Source of the Results values: model-implied p(press) on conflict trials in the
Risk-taking (0.62-0.72) and Risk-averse (0.09-0.24) states, and the Repeating-state
emissions (0.99 after a pellet, 0.89 after a shock, 0.005 after an omission), all
in emission_table.csv.

Posteriors computed for every Phase 3 trial (same data prep as 1.1 / 2.1):
  smoothed    p(z_t | y_1:T)                  -> what 2.1 / Fig. 3 use (reference)
  predictive  p(z_t | y_1:t-1)                -> past choices only; fully non-circular
  loo         p(z_t | y_1:t-1, y_t+1:T)       -> all choices EXCEPT y_t
                                                 (caveat: y_t leaks weakly through the
                                                 history inputs of trial t+1, mainly in
                                                 the Repeating state; see note below)

For each posterior type the script reports, per rat x state x cue:
  * response rate on trials assigned to the state (posterior >= 0.65), with n trials
  * soft response rate (posterior-weighted, no threshold)
  * model-implied p(press): mean of sigmoid(w_k . x_t) over the rat's trials of that cue
    (uses the actual history inputs; independent of assignment)

It also summarises the one-step-ahead predicted p(press) on conflict trials.

OUTPUTS (in HMM_Outputs/3.4_predictive_posteriors/):
  trial_posteriors_all_types.csv            per-trial posteriors (all three types)
  state_profiles_long.csv                   rat x state x cue x method, with n trials
  Prism_Fig3E_<method>.csv                  rows = rats, columns = State x Cue (paste into Prism)
  cell_counts_<method>.csv                  trials per rat x state x cue (flags cells < MIN_TRIALS)
  emission_table.csv                        model-implied p(press) by state x cue x previous outcome
  conflict_pred_ppress_summary.csv / hist   distribution of one-step-ahead p(press) on conflict trials
  summary_printout.txt                      everything printed to the console

Run from the repository root (needs the raw CSV and final_3_state_model_v5.pkl).
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

# =========================================================
# 0. CONFIGURATION
# =========================================================
INPUT_DIR   = "Data"
INPUT_CSV   = "collated_2026_conflict_data_raw.csv"      # same raw file as 1.1 / 2.1
MODEL_PKL   = "final_3_state_model_v5.pkl"
OUTPUT_DIR  = os.path.join("HMM_Outputs", "3.4_predictive_posteriors")

CONFIDENCE_THRESHOLD = 0.65     # same as 2.1
MIN_TRIALS           = 5        # cells below this are flagged (and set to NaN in the Prism tables)
STATE_NAMES = {1: 'Risk-taking', 2: 'Risk-averse', 3: 'Repeating'}
CUE_ORDER   = ['reward', 'punish', 'conflict']
CUE_LABEL   = {'reward': 'rewDS', 'punish': 'punDS', 'conflict': 'Conf'}
METHODS     = ['smoothed', 'predictive', 'loo']


# =========================================================
# 1. DATA PREP (identical to 1.1 / 2.1)
# =========================================================
def prepare_hmm_data(df):
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
        datas.append(sd['Lever_Press'].values.reshape(-1, 1).astype(int))
        inputs.append(sd[['rew_cue_val', 'pun_cue_val', 'past_pellet',
                          'past_shock', 'bias']].astype(float).values)
        meta.append({'Rat': rat, 'Session': session, 'Indices': sd.index})
    return datas, inputs, meta, df


# =========================================================
# 2. HMM HELPERS (same conventions as 1.1b)
# =========================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def get_press_weights(hmm):
    """(K, D) weights, positive = drive to press (sign flip as in 1.1)."""
    raw = hmm.observations.Wk
    W = raw[:, 0, :] * -1 if raw.ndim == 3 else raw.squeeze() * -1
    return np.atleast_2d(W)


def get_initial_distn(hmm, K):
    try:
        pi0 = np.asarray(hmm.init_state_distn.initial_state_distn, dtype=float)
    except Exception:
        pi0 = np.ones(K)
    pi0 = np.clip(pi0, 1e-300, None)
    return pi0 / pi0.sum()


def forward_backward(emis, A, pi0):
    """
    Scaled forward-backward for one session.
      emis : (T, K) p(y_t | z_t = k, x_t) for the OBSERVED choice
    Returns dict with
      pred     p(z_t | y_1:t-1)
      filt     p(z_t | y_1:t)
      smooth   p(z_t | y_1:T)
      loo      p(z_t | y_1:t-1, y_t+1:T)  (current choice excluded)
      logc     log p(y_t | y_1:t-1) per trial (one-step-ahead log predictive)
    """
    T, K = emis.shape
    pred = np.zeros((T, K)); filt = np.zeros((T, K)); c = np.zeros(T)
    for t in range(T):
        pred[t] = pi0 if t == 0 else filt[t - 1] @ A
        u = pred[t] * emis[t]
        c[t] = max(u.sum(), 1e-300)
        filt[t] = u / c[t]
    beta = np.ones((T, K))                       # scaled p(y_t+1:T | z_t)
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (emis[t + 1] * beta[t + 1]) / c[t + 1]
    smooth = filt * beta
    smooth /= smooth.sum(axis=1, keepdims=True)
    loo = pred * beta
    loo /= loo.sum(axis=1, keepdims=True)
    return dict(pred=pred, filt=filt, smooth=smooth, loo=loo, logc=np.log(c))


# =========================================================
# 3. MAIN
# =========================================================
class Tee:
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8'); self.o = sys.stdout
    def write(self, s):
        self.o.write(s); self.f.write(s)
    def flush(self):
        self.o.flush(); self.f.flush()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sys.stdout = Tee(os.path.join(OUTPUT_DIR, 'summary_printout.txt'))

    df_raw = pd.read_csv(os.path.join(INPUT_DIR, INPUT_CSV), dtype={"Session": str})
    datas, inputs, meta, df = prepare_hmm_data(df_raw)
    with open(MODEL_PKL, 'rb') as f:
        hmm = pickle.load(f)

    W = get_press_weights(hmm)
    A = np.atleast_2d(hmm.transitions.transition_matrix)
    K = A.shape[0]
    pi0 = get_initial_distn(hmm, K)
    print(f"Loaded {MODEL_PKL}: K = {K}; {len(datas)} sessions, "
          f"{df['Rat'].nunique()} rats\n")
    print("Press weights (rew, pun, past pellet, past shock, bias):")
    print(pd.DataFrame(W, index=[STATE_NAMES.get(k + 1, k + 1) for k in range(K)],
                       columns=['rewDS', 'punDS', 'past_pellet', 'past_shock', 'bias']).round(2))

    # ---- per-trial posteriors ----
    for m in METHODS:
        for k in range(K):
            df[f'{m}_P{k+1}'] = np.nan
    for k in range(K):
        df[f'pPress_given_S{k+1}'] = np.nan
    df['pred_pPress'] = np.nan

    max_diff = 0.0
    for Y, X, mt in zip(datas, inputs, meta):
        y = Y.ravel()
        pk = sigmoid(X @ W.T)                                  # (T, K) p(press | state)
        emis = np.where(y[:, None] == 1, pk, 1 - pk)
        fb = forward_backward(emis, A, pi0)
        # sanity check against ssm's own smoother
        ssm_smooth, _, _ = hmm.expected_states(Y, input=X)
        max_diff = max(max_diff, np.abs(ssm_smooth - fb['smooth']).max())
        idx = mt['Indices']
        for k in range(K):
            df.loc[idx, f'smoothed_P{k+1}']   = fb['smooth'][:, k]
            df.loc[idx, f'predictive_P{k+1}'] = fb['pred'][:, k]
            df.loc[idx, f'loo_P{k+1}']        = fb['loo'][:, k]
            df.loc[idx, f'pPress_given_S{k+1}'] = pk[:, k]
        df.loc[idx, 'pred_pPress'] = np.sum(fb['pred'] * pk, axis=1)
    print(f"\nCheck: max |own smoother - ssm.expected_states| = {max_diff:.2e} "
          f"({'OK' if max_diff < 1e-6 else 'MISMATCH - check pi0 / sign convention'})")

    df = df[df['smoothed_P1'].notna()].copy()           # sessions that entered the model
    for m in METHODS:
        P = df[[f'{m}_P{k+1}' for k in range(K)]].values
        conf = P.max(axis=1)
        df[f'{m}_state'] = np.where(conf >= CONFIDENCE_THRESHOLD, P.argmax(axis=1) + 1, np.nan)
        un = np.isnan(df[f'{m}_state']).mean() * 100
        print(f"{m:10s}: {un:5.1f}% of trials unassigned at >= {CONFIDENCE_THRESHOLD}")

    # agreement between assignment methods
    print("\nAgreement with smoothed assignment (trials assigned under both):")
    for m in ['predictive', 'loo']:
        both = df['smoothed_state'].notna() & df[f'{m}_state'].notna()
        agree = (df.loc[both, 'smoothed_state'] == df.loc[both, f'{m}_state']).mean() * 100
        print(f"  {m:10s}: {agree:.1f}% agree (n = {both.sum()})")

    keep = ['Rat', 'Session', 'Trial', 'Cue', 'Lever_Press', 'pred_pPress'] + \
           [c for c in df.columns if c.endswith(tuple(f'_P{k+1}' for k in range(K)))] + \
           [f'{m}_state' for m in METHODS] + [f'pPress_given_S{k+1}' for k in range(K)]
    df[keep].to_csv(os.path.join(OUTPUT_DIR, 'trial_posteriors_all_types.csv'), index=False)

    # ---- rat x state x cue profiles ----
    rows = []
    rats = sorted(df['Rat'].unique())
    for rat in rats:
        dr = df[df['Rat'] == rat]
        for k in range(1, K + 1):
            for cue in CUE_ORDER:
                dc = dr[dr['Cue'] == cue]
                implied_all = dc[f'pPress_given_S{k}'].mean() * 100   # assignment-free
                for m in METHODS:
                    sel = dc[dc[f'{m}_state'] == k]
                    w = dc[f'{m}_P{k}'].values
                    soft = np.nan if w.sum() < 1e-9 else \
                        np.sum(w * dc['Lever_Press'].values) / w.sum() * 100
                    rows.append({
                        'Rat': rat, 'State': k, 'State_name': STATE_NAMES.get(k, k),
                        'Cue': CUE_LABEL[cue], 'Method': m,
                        'n_trials': len(sel),
                        'Resp_rate_assigned': sel['Lever_Press'].mean() * 100 if len(sel) else np.nan,
                        'Model_implied_on_assigned': sel[f'pPress_given_S{k}'].mean() * 100 if len(sel) else np.nan,
                        'Resp_rate_soft': soft,
                        'Effective_n_soft': w.sum(),
                        'Model_implied_all_trials': implied_all,
                    })
    prof = pd.DataFrame(rows)
    prof.to_csv(os.path.join(OUTPUT_DIR, 'state_profiles_long.csv'), index=False)

    # ---- Prism-ready tables (rows = rats; columns = State|Cue), cells < MIN_TRIALS -> NaN ----
    for m in METHODS:
        sub = prof[prof['Method'] == m].copy()
        sub.loc[sub['n_trials'] < MIN_TRIALS, 'Resp_rate_assigned'] = np.nan
        wide = sub.pivot_table(index='Rat', columns=['State_name', 'Cue'],
                               values='Resp_rate_assigned', dropna=False)
        wide = wide.reindex(columns=pd.MultiIndex.from_product(
            [[STATE_NAMES[k] for k in range(1, K + 1)], [CUE_LABEL[c] for c in CUE_ORDER]]))
        wide.columns = [f'{s} | {c}' for s, c in wide.columns]
        wide.to_csv(os.path.join(OUTPUT_DIR, f'Prism_Fig3E_{m}.csv'))
        cnt = sub.pivot_table(index='Rat', columns=['State_name', 'Cue'], values='n_trials')
        cnt.columns = [f'{s} | {c}' for s, c in cnt.columns]
        cnt.to_csv(os.path.join(OUTPUT_DIR, f'cell_counts_{m}.csv'))
        n_missing = int(wide.isna().sum().sum())
        print(f"\n[{m}] Prism table: {n_missing} empty cells (rat x state x cue with < "
              f"{MIN_TRIALS} trials) -> "
              f"{'full RM ANOVA possible' if n_missing == 0 else 'use mixed-effects (REML) in Prism'}")

    # model-implied, assignment-free table (one per rat; no missing cells by construction)
    imp = prof[prof['Method'] == 'smoothed'].pivot_table(
        index='Rat', columns=['State_name', 'Cue'], values='Model_implied_all_trials')
    imp = imp.reindex(columns=pd.MultiIndex.from_product(
        [[STATE_NAMES[k] for k in range(1, K + 1)], [CUE_LABEL[c] for c in CUE_ORDER]]))
    imp.columns = [f'{s} | {c}' for s, c in imp.columns]
    imp.to_csv(os.path.join(OUTPUT_DIR, 'Prism_Fig3E_model_implied.csv'))

    # ---- group summary ----
    print("\n=== Group mean (SEM) response rate (%) by state x cue ===")
    print("(assigned = rate on trials assigned at >= 0.65; soft = posterior-weighted; "
          "implied = mean model p(press | state, x_t) over ALL the rat's trials of that cue)\n")
    out = []
    for k in range(1, K + 1):
        for cue in [CUE_LABEL[c] for c in CUE_ORDER]:
            r = {'State': STATE_NAMES[k], 'Cue': cue}
            for m in METHODS:
                s = prof[(prof.Method == m) & (prof.State == k) & (prof.Cue == cue) &
                         (prof.n_trials >= MIN_TRIALS)]['Resp_rate_assigned']
                r[f'{m}_assigned'] = f"{s.mean():.1f} ({s.sem():.1f}) n={s.count()}"
            s = prof[(prof.Method == 'predictive') & (prof.State == k) & (prof.Cue == cue)]['Resp_rate_soft']
            r['predictive_soft'] = f"{s.mean():.1f} ({s.sem():.1f})"
            s = prof[(prof.Method == 'smoothed') & (prof.State == k) & (prof.Cue == cue)]['Model_implied_all_trials']
            r['model_implied'] = f"{s.mean():.1f} ({s.sem():.1f})"
            out.append(r)
    summ = pd.DataFrame(out)
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(summ.to_string(index=False))
    summ.to_csv(os.path.join(OUTPUT_DIR, 'group_summary_by_method.csv'), index=False)

    # ---- emission table: state x cue x previous outcome (global, assignment-free) ----
    prev = {'after omission': (0, 0), 'after pellet': (1, 0), 'after shock': (0, 1)}
    cues = {'rewDS': (1, 0), 'punDS': (0, 1), 'Conf': (1, 1)}
    em = []
    for k in range(K):
        for cname, (r_, p_) in cues.items():
            row = {'State': STATE_NAMES.get(k + 1, k + 1), 'Cue': cname}
            for pname, (pp, ps) in prev.items():
                x = np.array([r_, p_, pp, ps, 1.0])
                row[pname] = round(float(sigmoid(x @ W[k])), 3)
            em.append(row)
    em = pd.DataFrame(em)
    em.to_csv(os.path.join(OUTPUT_DIR, 'emission_table.csv'), index=False)
    print("\n=== Model-implied p(press) by state, cue and previous outcome ===")
    print(em.to_string(index=False))

    # ---- one-step-ahead predicted p(press) on conflict trials ----
    conf = df[df['Cue'] == 'conflict']
    v = conf['pred_pPress'].values
    bins = {'< 0.3': (v < 0.3).mean() * 100,
            '0.3-0.7': ((v >= 0.3) & (v <= 0.7)).mean() * 100,
            '> 0.7': (v > 0.7).mean() * 100}
    n = len(v)
    g = pd.Series(v).skew(); kurt = pd.Series(v).kurt()
    bc = (g ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    print("\n=== One-step-ahead predicted p(press) on conflict trials (no current choice used) ===")
    print(f"  n = {n} trials; median = {np.median(v):.2f}")
    for b, p in bins.items():
        print(f"  {b:8s}: {p:5.1f}% of conflict trials")
    print(f"  Sarle's bimodality coefficient = {bc:.3f} (> 0.555 suggests bimodality)")
    try:
        import diptest
        dip, pdip = diptest.diptest(v)
        print(f"  Hartigan's dip = {dip:.4f}, p = {pdip:.4g}")
    except ImportError:
        print("  (pip install diptest for Hartigan's dip test)")
    print("  NOTE: the predicted distribution is shaped by the model; treat it as a description\n"
          "  of the fitted mixture, not as independent evidence of bimodality. The independent\n"
          "  evidence is the cross-validated gain over the single-state GLM (Fig. S2A).")
    per_rat = conf.groupby('Rat')['pred_pPress'].agg(
        median='median',
        pct_below_0_3=lambda s: (s < 0.3).mean() * 100,
        pct_0_3_to_0_7=lambda s: ((s >= 0.3) & (s <= 0.7)).mean() * 100,
        pct_above_0_7=lambda s: (s > 0.7).mean() * 100)
    per_rat.to_csv(os.path.join(OUTPUT_DIR, 'conflict_pred_ppress_summary.csv'))

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.hist(v, bins=np.linspace(0, 1, 41), color='#011186', edgecolor='white')
    ax.set_xlabel('One-step-ahead predicted p(press)\non conflict trials')
    ax.set_ylabel('Trials')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'conflict_pred_ppress_hist.svg'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'conflict_pred_ppress_hist.png'), dpi=300)
    plt.close(fig)

    print("\nNOTE on 'loo': the history inputs of trial t+1 (past pellet / past shock) reveal\n"
          "whether the rat pressed on trial t, so y_t leaks into the loo posterior through\n"
          "the Repeating state's large history weights. For the two on-task states the leak is\n"
          "much smaller (history weights <= ~1). 'predictive' is the fully non-circular version.")
    print(f"\nDone. Outputs in '{OUTPUT_DIR}/'.")
    tee = sys.stdout
    sys.stdout = tee.o
    tee.f.close()


if __name__ == '__main__':
    main()
