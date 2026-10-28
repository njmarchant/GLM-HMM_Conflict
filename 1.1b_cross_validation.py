"""
1.1b_cross_validation.py
------------------------
Leave-one-rat-out cross-validation of the GLM-HMM (K = 1..5) and of a single-state
logistic GLM baseline (Supplementary Fig. 2A-B).

For each held-out rat, every K is refit on the other 15 rats (same multi-start EM as 1.1)
and scored on the held-out rat with a scaled forward filter: held-out log-likelihood per
trial, and one-step-ahead choice accuracy, in which the state distribution at trial t is
the filtered posterior from t-1 propagated through the transition matrix (no prediction
is conditioned on the choice it predicts).

Inputs   Data/collated_2026_conflict_data_raw.csv
Outputs  (HMM_Outputs/)
         CV_HeldOut_LL_by_State.csv, CV_OneStepAhead_Accuracy.csv
         SuppFig_CrossValidation_Accuracy          Fig. S2A  per-cue accuracy, 3-state vs GLM
         SuppFig_CrossValidation_AccuracyByState   Fig. S2B  accuracy vs number of states
         SuppFig_CrossValidation_LL                held-out LL vs number of states (not in the paper)
         SuppFig2_stats.txt, SuppFig2_stats_table.csv   legend-ready statistics
           (also AIC/BIC differences for Fig. S2C-D if 1.1 has written Model_Selection_Metrics.csv)

The full run is 16 folds x 5 K x 10 initialisations = 800 fits (the slowest step).
To recompute the statistics and figures from the saved CSVs without refitting:
    python 1.1b_cross_validation.py --stats-only
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# ssm is only needed for the full cross-validation run, not for --stats-only
try:
    import ssm
    SSM_IMPORT_ERROR = None
except ImportError as e:
    ssm = None
    SSM_IMPORT_ERROR = e
SSM_HELP = ("If the error mentions NumPy 1.x / 2.x or _ARRAY_API, ssm was compiled against a different NumPy "
            "than the one installed: rebuild it (README, Installation > Troubleshooting).")

# =========================================================
# 0. REPRODUCIBILITY & PLOT STYLE
# =========================================================
# Same seed convention as 1.1 so multi-start behaviour is reproducible.
np.random.seed(14)

# Illustrator-editable vector text
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

# ---- Configuration knobs -------------------------------------------------
INPUT_DIR        = "Data"                                 # folder holding the raw CSV
INPUT_CSV        = "collated_2026_conflict_data_raw.csv"  # same raw file as 1.1
INPUT_PATH       = os.path.join(INPUT_DIR, INPUT_CSV)     # full path to the data
OUTPUT_DIR       = "HMM_Outputs"
MAX_STATES       = 5      # evaluate 1..MAX_STATES, mirroring 1.1
CHOSEN_STATES    = 3      # the model that feeds the per-cue accuracy panel (Fig S2A)
CV_NUM_INITS     = 10     # multi-start inits per fold-fit (1.1 uses 10).
                          #   3-5 is usually sufficient for CV and much faster.
CV_NUM_ITERS     = 200    # EM iterations per fit (matches 1.1)
CV_TOLERANCE     = 1e-4   # EM tolerance (matches 1.1)
# -------------------------------------------------------------------------


# =========================================================
# 1. DATA PREPARATION  (identical logic to 1.1_model_fit.py)
# =========================================================
def prepare_hmm_data(df):
    """
    Convert the clean dataframe into the list-of-arrays format ssm expects,
    with strict Rat x Session boundaries. This is copied from 1.1 so that the
    CV models see EXACTLY the same covariates, in the same order, as the
    full-data reporting model.

    NOTE on column order: the input feature order here is
        [rew_cue_val, pun_cue_val, past_pellet, past_shock, bias]
    which matches the order the model is FIT in (1.1).
    """
    print("Preparing data: separating by Rat and Session...")
    df = df.sort_values(by=['Rat', 'Session', 'Trial']).copy()

    # Past feedback variables (shifted by 1 trial, within session)
    df['past_pellet'] = (df['Response'] == 'Response-Pellet').astype(int)\
        .groupby([df['Rat'], df['Session']]).shift(1).fillna(0)
    df['past_shock'] = (df['Response'] == 'Response-Shock').astype(int)\
        .groupby([df['Rat'], df['Session']]).shift(1).fillna(0)

    # Binary cue indicators (conflict activates BOTH cues)
    df['rew_cue_val'] = (df['Cue'] == 'reward') | (df['Cue'] == 'conflict')
    df['pun_cue_val'] = (df['Cue'] == 'punish') | (df['Cue'] == 'conflict')

    # Intercept
    df['bias'] = 1.0

    # Binary response (1 = Press, 0 = Omission)
    df['Lever_Press'] = df['Outcome'].astype(int)

    datas, inputs, session_meta = [], [], []
    for (rat, session), session_data in df.groupby(['Rat', 'Session']):
        if len(session_data) < 5:
            continue
        Y = session_data['Lever_Press'].values.reshape(-1, 1).astype(int)
        X = session_data[['rew_cue_val', 'pun_cue_val',
                          'past_pellet', 'past_shock', 'bias']].astype(float).values
        datas.append(Y)
        inputs.append(X)
        session_meta.append({'Rat': rat, 'Session': session})

    print(f"Successfully processed {len(datas)} independent sessions "
          f"from {df['Rat'].nunique()} rats.")
    return datas, inputs, session_meta


# =========================================================
# 2. HELPERS: emissions, weights, and a causal forward filter
# =========================================================
def sigmoid(z):
    # Clipped for numerical stability.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


def get_press_weights(hmm):
    """
    Extract per-state GLM weights oriented so that POSITIVE = drive to press,
    replicating the -1 sign flip used in 1.1's export block. Returns (K, D).
    """
    raw = hmm.observations.Wk
    if raw.ndim == 3:            # (K, C-1, D) with C=2 -> take the single logit row
        W = raw[:, 0, :] * -1
    else:
        W = raw.squeeze() * -1
    return np.atleast_2d(W)


def get_initial_distn(hmm, K):
    """Robustly fetch the fitted initial-state distribution; fall back to uniform.
    (For ~100-trial sessions the initial distribution has negligible impact.)"""
    try:
        pi0 = np.asarray(hmm.init_state_distn.initial_state_distn, dtype=float)
    except Exception:
        try:
            log_pi0 = np.asarray(hmm.init_state_distn.log_pi0, dtype=float)
            pi0 = np.exp(log_pi0 - log_pi0.max())
        except Exception:
            pi0 = np.ones(K)
    pi0 = np.clip(pi0, 1e-12, None)
    return pi0 / pi0.sum()


def forward_filter(emis, A, pi0):
    """
    Scaled forward algorithm on a single session.

    emis : (T, K) likelihood of the OBSERVED choice under each state, p(y_t | z_t=k)
    A    : (K, K) transition matrix (rows sum to 1)
    pi0  : (K,)   initial state distribution

    Returns
      pred    : (T, K) one-step-ahead state prior p(z_t | y_{1:t-1})   [causal]
      loglik  : float  marginal log-likelihood of the session (sum of log scalers)
    """
    T, K = emis.shape
    alpha = np.zeros((T, K))   # filtered posterior p(z_t | y_{1:t})
    pred = np.zeros((T, K))    # one-step-ahead prior p(z_t | y_{1:t-1})
    loglik = 0.0
    for t in range(T):
        prior = pi0 if t == 0 else alpha[t - 1] @ A
        pred[t] = prior
        unnorm = prior * emis[t]
        c = unnorm.sum()
        c = c if c > 0 else 1e-300
        alpha[t] = unnorm / c
        loglik += np.log(c)
    return pred, loglik


def score_heldout(hmm, test_datas, test_inputs):
    """
    For a fitted model and a held-out rat's sessions, compute:
      - total held-out log-likelihood and total trial count
      - per-trial one-step-ahead press predictions + observed choices + cue codes

    Cue codes are derived from the input columns (rew=col0, pun=col1):
      (1,0)->'reward'  (0,1)->'punish'  (1,1)->'conflict'
    """
    W = get_press_weights(hmm)                 # (K, D)
    A = np.atleast_2d(hmm.transitions.transition_matrix)
    K = A.shape[0]
    pi0 = get_initial_distn(hmm, K)

    total_ll, total_n = 0.0, 0
    all_pred, all_obs, all_cue = [], [], []

    for Y, X in zip(test_datas, test_inputs):
        y = Y.ravel().astype(int)
        p_press_states = sigmoid(X @ W.T)                    # (T, K)
        emis = np.where(y[:, None] == 1,
                        p_press_states, 1.0 - p_press_states)  # p(observed | state)

        pred, ll = forward_filter(emis, A, pi0)
        total_ll += ll
        total_n += len(y)

        # One-step-ahead predicted press probability (uses only past choices)
        p_press_osa = np.sum(pred * p_press_states, axis=1)
        all_pred.append((p_press_osa >= 0.5).astype(int))
        all_obs.append(y)

        rew = X[:, 0] > 0.5
        pun = X[:, 1] > 0.5
        cue = np.where(rew & pun, 'conflict', np.where(rew, 'reward', 'punish'))
        all_cue.append(cue)

    return (total_ll, total_n,
            np.concatenate(all_pred) if all_pred else np.array([]),
            np.concatenate(all_obs) if all_obs else np.array([]),
            np.concatenate(all_cue) if all_cue else np.array([]))


# =========================================================
# 3. LEAVE-ONE-RAT-OUT CROSS-VALIDATION
# =========================================================
def run_loro_cv(datas, inputs, meta):
    obs_dim = 1
    input_dim = inputs[0].shape[1]
    rats = sorted({m['Rat'] for m in meta})
    print(f"\n--- Leave-one-rat-out CV over {len(rats)} rats, states 1..{MAX_STATES} ---")

    ll_rows = []       # per fold x state: held-out LL/trial
    acc_rows = []      # per fold x cue: chosen-K HMM vs 1-state GLM accuracy

    for held in rats:
        print(f"\n=== Fold: holding out rat {held} ===")
        tr = [i for i, m in enumerate(meta) if m['Rat'] != held]
        te = [i for i, m in enumerate(meta) if m['Rat'] == held]

        train_datas, train_inputs = [datas[i] for i in tr], [inputs[i] for i in tr]
        test_datas,  test_inputs  = [datas[i] for i in te], [inputs[i] for i in te]

        # ---- 1-state logistic-regression baseline (K-independent, causal) ----
        Xtr = np.vstack(train_inputs); Ytr = np.concatenate([d.ravel() for d in train_datas])
        Xte = np.vstack(test_inputs);  Yte = np.concatenate([d.ravel() for d in test_datas])
        glm = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        glm.fit(Xtr, Ytr)
        glm_pred = (glm.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
        rew = Xte[:, 0] > 0.5; pun = Xte[:, 1] > 0.5
        glm_cue = np.where(rew & pun, 'conflict', np.where(rew, 'reward', 'punish'))

        # ---- Fit GLM-HMM for each state count on the training rats ----
        for k in range(1, MAX_STATES + 1):
            best_ll, best_hmm = -np.inf, None
            for _ in range(CV_NUM_INITS):
                hmm = ssm.HMM(k, obs_dim, input_dim,
                              observations="input_driven_obs",
                              observation_kwargs=dict(C=2),
                              transitions="standard")
                lls = hmm.fit(train_datas, inputs=train_inputs, method="em",
                              num_iters=CV_NUM_ITERS, tolerance=CV_TOLERANCE)
                if lls[-1] > best_ll:
                    best_ll, best_hmm = lls[-1], hmm

            total_ll, total_n, pred, obs, cue = score_heldout(best_hmm, test_datas, test_inputs)
            ll_per_trial = total_ll / total_n
            overall_acc = float(np.mean(pred == obs)) * 100.0
            ll_rows.append({'HeldOut_Rat': held, 'Num_States': k,
                            'HeldOut_LL_per_trial': ll_per_trial,
                            'HeldOut_Accuracy_overall': overall_acc,
                            'N_test_trials': total_n})
            print(f"  {k}-state | held-out LL/trial = {ll_per_trial:.4f} | "
                  f"one-step acc = {overall_acc:.1f}%")

            # ---- Per-cue accuracy for the chosen model only (Fig S2A) ----
            if k == CHOSEN_STATES:
                for cue_name in ['reward', 'punish', 'conflict']:
                    hm = cue == cue_name
                    gm = glm_cue == cue_name
                    if hm.sum() > 0:
                        acc_rows.append({'HeldOut_Rat': held, 'Cue': cue_name,
                                         'Model': '3-State GLM-HMM',
                                         'Accuracy': float(np.mean(pred[hm] == obs[hm])) * 100.0})
                    if gm.sum() > 0:
                        acc_rows.append({'HeldOut_Rat': held, 'Cue': cue_name,
                                         'Model': '1-State GLM Baseline',
                                         'Accuracy': float(np.mean(glm_pred[gm] == Yte[gm])) * 100.0})

    return pd.DataFrame(ll_rows), pd.DataFrame(acc_rows)


# =========================================================
# 4. STATISTICS FOR SUPPLEMENTARY FIGURE 2
# =========================================================
def _paired(a, b):
    """Paired comparison a vs b across rats. Returns a dict of statistics."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    d = a - b
    n = len(d)
    t, p = stats.ttest_rel(a, b)
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    try:
        w_p = stats.wilcoxon(a, b).pvalue
    except ValueError:            # all differences zero
        w_p = np.nan
    return {'n': n, 'mean_a': a.mean(), 'sem_a': a.std(ddof=1) / np.sqrt(n),
            'mean_b': b.mean(), 'sem_b': b.std(ddof=1) / np.sqrt(n),
            'mean_diff': d.mean(), 'ci_low': d.mean() - tcrit * se, 'ci_high': d.mean() + tcrit * se,
            'n_improved': int((d > 0).sum()), 't': t, 'df': n - 1, 'p': p, 'wilcoxon_p': w_p}


def _fmt_p(p):
    if np.isnan(p):
        return "n/a"
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}" if p < 0.001 else f"p = {p:.3f}"


def write_s2_stats(ll_df, acc_df, output_dir):
    """
    Writes legend-ready statistics for Supp Fig 2A/B (and AIC/BIC deltas for C/D
    if 1.1's Model_Selection_Metrics.csv is present). All tests are paired across
    held-out rats (each rat contributes one value per model).
    """
    lines, rows = [], []
    cue_names = {'reward': 'rewDS', 'punish': 'punDS', 'conflict': 'conflict'}
    hmm_col = f'{CHOSEN_STATES}-State GLM-HMM'

    # ---- S2A: per-cue accuracy, HMM vs 1-state GLM ----
    lines.append(f"S2A  Held-out one-step-ahead accuracy, {CHOSEN_STATES}-state GLM-HMM vs "
                 f"1-state GLM (mean +/- SEM across rats; paired t-test across held-out rats)")
    pieces = []
    for cue in ['reward', 'punish', 'conflict']:
        sub = acc_df[acc_df.Cue == cue].pivot(index='HeldOut_Rat', columns='Model', values='Accuracy')
        if hmm_col not in sub.columns or '1-State GLM Baseline' not in sub.columns:
            continue
        s = _paired(sub[hmm_col], sub['1-State GLM Baseline'])
        txt = (f"{cue_names[cue]}: {s['mean_a']:.1f} +/- {s['sem_a']:.1f}% vs "
               f"{s['mean_b']:.1f} +/- {s['sem_b']:.1f}% "
               f"(difference {s['mean_diff']:+.1f} pp, 95% CI {s['ci_low']:+.1f} to {s['ci_high']:+.1f}; "
               f"higher in {s['n_improved']}/{s['n']} rats; t({s['df']}) = {s['t']:.2f}, {_fmt_p(s['p'])}; "
               f"Wilcoxon {_fmt_p(s['wilcoxon_p'])})")
        lines.append("  " + txt)
        pieces.append(f"{cue_names[cue]}: {s['mean_a']:.1f} ± {s['sem_a']:.1f}% vs "
                      f"{s['mean_b']:.1f} ± {s['sem_b']:.1f}%, t({s['df']}) = {s['t']:.2f}, {_fmt_p(s['p'])}")
        rows.append({'Panel': 'S2A', 'Comparison': f'{cue_names[cue]}: HMM vs GLM', **s})
    lines.append("  Legend text:  [" + "; ".join(pieces) + "]")

    # ---- S2B: accuracy by number of states ----
    lines.append("\nS2B  Held-out one-step-ahead accuracy by number of states (mean +/- SEM across rats)")
    acc_piv = ll_df.pivot(index='HeldOut_Rat', columns='Num_States', values='HeldOut_Accuracy_overall')
    ll_piv = ll_df.pivot(index='HeldOut_Rat', columns='Num_States', values='HeldOut_LL_per_trial')
    for k in sorted(acc_piv.columns):
        a, l = acc_piv[k], ll_piv[k]
        lines.append(f"  {k} state(s): accuracy {a.mean():.1f} +/- {a.sem():.1f}%; "
                     f"held-out LL/trial {l.mean():.4f} +/- {l.sem():.4f}")
        rows.append({'Panel': 'S2B', 'Comparison': f'{k} states (descriptive)', 'n': a.count(),
                     'mean_a': a.mean(), 'sem_a': a.sem()})
    lines.append("  Consecutive state numbers (paired across held-out rats):")
    ks = sorted(acc_piv.columns)
    for a_k, b_k in zip(ks[:-1], ks[1:]):
        for metric, piv, unit in [('accuracy', acc_piv, 'pp'), ('LL/trial', ll_piv, '')]:
            s = _paired(piv[b_k], piv[a_k])
            dec = 2 if metric == 'accuracy' else 4
            lines.append(f"    {a_k} -> {b_k} states, {metric}: {s['mean_diff']:+.{dec}f} {unit} "
                         f"(95% CI {s['ci_low']:+.{dec}f} to {s['ci_high']:+.{dec}f}); higher in "
                         f"{s['n_improved']}/{s['n']} rats; t({s['df']}) = {s['t']:.2f}, {_fmt_p(s['p'])}")
            rows.append({'Panel': 'S2B', 'Comparison': f'{a_k}->{b_k} states, {metric}', **s})
    if CHOSEN_STATES in ks:
        for k in [k for k in ks if k > CHOSEN_STATES]:
            s = _paired(acc_piv[k], acc_piv[CHOSEN_STATES])
            lines.append(f"    {CHOSEN_STATES} -> {k} states, accuracy: {s['mean_diff']:+.2f} pp; "
                         f"t({s['df']}) = {s['t']:.2f}, {_fmt_p(s['p'])}")
            rows.append({'Panel': 'S2B', 'Comparison': f'{CHOSEN_STATES}->{k} states, accuracy', **s})
        extra = [f"{k} states: {acc_piv[k].mean():.1f} ± {acc_piv[k].sem():.1f}%"
                 for k in ks if k > CHOSEN_STATES]
        lines.append("  Legend text:  [" + "; ".join(extra) + "]")

    # ---- S2C/D: AIC/BIC deltas, if available ----
    msm = os.path.join(output_dir, 'Model_Selection_Metrics.csv')
    if os.path.exists(msm):
        m = pd.read_csv(msm).set_index('Num_States')
        lines.append("\nS2C/D  Information criteria (full dataset; from 1.1). Delta = model minus "
                     f"{CHOSEN_STATES}-state (negative = better than {CHOSEN_STATES}-state)")
        for k, r in m.iterrows():
            lines.append(f"  {k} states: LL = {r['Log_Likelihood']:.1f}, k = {int(r['Free_Parameters'])}, "
                         f"AIC = {r['AIC']:.1f} (delta {r['AIC'] - m.loc[CHOSEN_STATES, 'AIC']:+.1f}), "
                         f"BIC = {r['BIC']:.1f} (delta {r['BIC'] - m.loc[CHOSEN_STATES, 'BIC']:+.1f})")

    txt = "\n".join(lines)
    print("\n" + "=" * 70 + "\nSUPPLEMENTARY FIGURE 2 STATISTICS\n" + "=" * 70 + "\n" + txt)
    with open(os.path.join(output_dir, 'SuppFig2_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(txt + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'SuppFig2_stats_table.csv'), index=False)
    print(f"\nSaved {os.path.join(output_dir, 'SuppFig2_stats.txt')} and SuppFig2_stats_table.csv")


# =========================================================
# 5. PLOTS
# =========================================================
def plot_cv_loglik(ll_df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    g = ll_df.groupby('Num_States')['HeldOut_LL_per_trial']
    mean, sem = g.mean(), g.sem()
    ks = mean.index.values

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(ks, mean.values, yerr=sem.values, marker='o',
                color='black', capsize=3, linewidth=1.5)
    ax.axvline(CHOSEN_STATES, color='red', linestyle='--', linewidth=1.2,
               label=f'Selected ({CHOSEN_STATES} states)')
    ax.set_title('Cross-validated held-out log-likelihood', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of latent states', fontsize=11)
    ax.set_ylabel('Held-out LL per trial\n(higher = better)', fontsize=11)
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'SuppFig_CrossValidation_LL.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'SuppFig_CrossValidation_LL.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved held-out LL figure (mean +/- SEM across rats).")


def plot_cv_accuracy_by_state(ll_df, output_dir):
    """
    Supplementary panel S2B: overall one-step-ahead held-out accuracy as a
    function of the number of latent states.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    g = ll_df.groupby('Num_States')['HeldOut_Accuracy_overall']
    mean, sem = g.mean(), g.sem()
    ks = mean.index.values

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(ks, mean.values, yerr=sem.values, marker='o',
                color='black', capsize=3, linewidth=1.5)
    ax.axvline(CHOSEN_STATES, color='red', linestyle='--', linewidth=1.2,
               label=f'Selected ({CHOSEN_STATES} states)')
    ax.set_title('Cross-validated choice prediction accuracy',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of latent states', fontsize=11)
    ax.set_ylabel('Held-out predictive accuracy (%)\n(one-step-ahead)', fontsize=11)
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'SuppFig_CrossValidation_AccuracyByState.svg'),
                bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'SuppFig_CrossValidation_AccuracyByState.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved accuracy-vs-state-number figure (mean +/- SEM across rats).")


def plot_cv_accuracy(acc_df, output_dir):
    """Supplementary panel S2A: per-cue accuracy, HMM vs 1-state GLM."""
    if acc_df.empty:
        print("No accuracy rows to plot."); return
    import seaborn as sns
    palette = {'1-State GLM Baseline': '#c9c9c9', '3-State GLM-HMM': '#23395B'}
    order = ['reward', 'punish', 'conflict']
    labels = {'reward': 'Reward Cue', 'punish': 'Punishment Cue', 'conflict': 'Conflict Cue'}
    acc_df = acc_df.copy(); acc_df['Cue'] = acc_df['Cue'].map(labels)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.barplot(data=acc_df, x='Cue', y='Accuracy', hue='Model',
                order=[labels[o] for o in order],
                palette=palette, edgecolor='black', linewidth=1.3,
                errorbar='se', capsize=0.1, ax=ax)
    ax.set_title('Cross-validated choice prediction accuracy (one-step-ahead)',
                 fontsize=12, fontweight='bold', pad=14)
    ax.set_ylabel('Held-out predictive accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Trial context', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.axhline(50, color='gray', linestyle=':', linewidth=1.2)
    ax.text(2.45, 51, 'Chance\n(50%)', color='gray', fontsize=8, ha='center', va='bottom')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.legend(frameon=False, fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'SuppFig_CrossValidation_Accuracy.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'SuppFig_CrossValidation_Accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved cross-validated accuracy figure (mean +/- SEM across rats).")


# =========================================================
# 6. EXECUTION BLOCK
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leave-one-rat-out CV for the GLM-HMM (Supp Fig 2A/B).")
    parser.add_argument('--stats-only', action='store_true',
                        help="skip the CV; recompute statistics (and replot) from the saved CSVs")
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    ll_path = os.path.join(OUTPUT_DIR, 'CV_HeldOut_LL_by_State.csv')
    acc_path = os.path.join(OUTPUT_DIR, 'CV_OneStepAhead_Accuracy.csv')

    if args.stats_only:
        if not (os.path.exists(ll_path) and os.path.exists(acc_path)):
            sys.exit(f"--stats-only needs {ll_path} and {acc_path}. Run without the flag first.")
        ll_df, acc_df = pd.read_csv(ll_path), pd.read_csv(acc_path)
        print(f"Loaded saved CV results: {ll_df['HeldOut_Rat'].nunique()} held-out rats.")
    else:
        if ssm is None:
            sys.exit(f"ssm could not be imported: {SSM_IMPORT_ERROR}\n{SSM_HELP}\n"
                     "(--stats-only reuses the saved CSVs without ssm.)")
        df = pd.read_csv(INPUT_PATH, dtype={"Session": str})
        datas, inputs, meta = prepare_hmm_data(df)
        ll_df, acc_df = run_loro_cv(datas, inputs, meta)
        # Save raw per-fold metrics for supplementary tables
        ll_df.to_csv(ll_path, index=False)
        acc_df.to_csv(acc_path, index=False)

    # Statistics (legend-ready) for Supp Fig 2A/B
    write_s2_stats(ll_df, acc_df, OUTPUT_DIR)

    plot_cv_loglik(ll_df, OUTPUT_DIR)                 # held-out LL (not in the paper)
    plot_cv_accuracy(acc_df, OUTPUT_DIR)              # Supp panel S2A
    plot_cv_accuracy_by_state(ll_df, OUTPUT_DIR)      # Supp panel S2B

    print("\nDone. Outputs written to 'HMM_Outputs/'.")
