"""
4.1_aIC_posthoc_choice_prediction.py
------------------------------------
Does aIC calcium carry information about the upcoming choice that the frozen
3-state GLM-HMM does not already capture? (Supplementary Fig. 4)

APPROACH (post-hoc, model frozen)
---------------------------------
1. Load the trained model (final_3_state_model_v5.pkl). Nothing is refit.
2. Re-run it CAUSALLY over every session: for trial t the predicted state
   distribution uses only choices 1..t-1 (one-step-ahead, identical forward
   filter to 1.1b). This gives p_press_OSA(t) = the model's best guess before
   it sees the choice.

   !! The Prob_State_* columns already in the CSV are SMOOTHED posteriors from
   !! hmm.expected_states(); they condition on the choice at trial t itself and
   !! must NOT be used as the baseline prediction (they would leak the answer).

3. Second-stage logistic regression, with the model's log-odds as a fixed
   OFFSET (coefficient fixed at 1):
       Baseline  : choice ~ 1                         + offset(logit p_OSA)
       aIC       : choice ~ 1 + aIC                   + offset
       aIC x cue : choice ~ 1 + aIC:rew + aIC:pun + aIC:conf + offset
       aIC x state: choice ~ 1 + aIC:P(S1) + aIC:P(S2) + aIC:P(S3) + offset
       aIC_conf_x_state (HYPOTHESIS MODEL):
                   choice ~ 1 + aIC:rew + aIC:pun
                              + aIC:conf:P(S1) + aIC:conf:P(S2) + aIC:conf:P(S3) + offset
         Prediction: conflict slope > 0 in State 1, < 0 in State 2.
         Tested by the S1 - S2 contrast (rat-clustered Wald), a state-shuffle
         permutation null, and CV gain over the single-conflict-slope model.
       aIC_full_x_state: every cue x state has its own slope. Used to compare
         conflict with rewDS WITHIN State 1 and with punDS WITHIN State 2
         (Wald tests on the full rat-clustered covariance), plus the
         'mismatched' comparisons as a check that the match is specific.
   aIC is z-scored within session. P(Sk) are the causal predicted state probs.

4. Evaluation
   a) Leave-one-rat-out CV of the second stage: held-out log-likelihood per
      trial and accuracy, overall and split by cue. Paired across rats.
   b) Shuffle control: aIC permuted within Rat x Session x Cue (keeps any
      cue-related mean difference, destroys trial-by-trial information).
   c) Full-data fits with rat-clustered robust SEs for the coefficients.
   d) Two windows: ZMean_CueFull (pre-specified, 0-10 s cue) and ZMean_BL
      (5 s before trial onset) as a specificity control.

Figures for the paper (in OUTPUT_DIR):
   aIC_PostHoc_Window_BL_vs_CueFull             Fig. S4A  does aIC add information? (+ shuffle null)
   aIC_SlopeGrid_CueByState_ZMean_CueFull       Fig. S4B  slopes for every cue x state
   aIC_SlopeMatching_WithinState_ZMean_CueFull  Fig. S4C-D  conflict vs matched / mismatched cue
Other figures/CSVs are diagnostics and supporting statistics.

Input: the 2.1 output table (Phase 3 trials with state probabilities; it carries the
photometry columns ZMean_BL and ZMean_CueFull from the raw table) and the model file.
Run from the repository root.
Requires: numpy, pandas, scipy, statsmodels, matplotlib, and ssm (only so the
pickle can be loaded).
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(14)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

# =========================================================
# 0. CONFIGURATION
# =========================================================
INPUT_PATH  = os.path.join("Data", "collated_2026_conflict_data_with_Probabilities_065_Past_PelletShock_v5.csv")
MODEL_PATH  = "final_3_state_model_v5.pkl"     # the FROZEN 3-state model object is required
OUTPUT_DIR  = os.path.join("HMM_Outputs", "aIC_PostHoc")

# Roles of the 3-state model (fixed a priori, from the behavioural characterisation)
ROLES = {'risk_taking': 1, 'risk_averse': 2}
STATE_NAMES = {1: 'Risk-taking', 2: 'Risk-averse', 3: 'Repeating'}

PRIMARY_WINDOW = "ZMean_CueFull"
# ZMean_CueFull is the pre-specified window; ZMean_BL is the specificity control:
# pre-trial activity should NOT predict choice beyond the model.
WINDOWS = ["ZMean_BL", "ZMean_CueFull"]

# LEVER note: ZMean_Lever (10 s before the action) overlaps the press itself and
# its meaning differs for omitted trials, so it is not a clean *pre-choice*
# predictor. Set True only for exploratory comparison; interpret with caution.
INCLUDE_LEVER = False

# Rats dropped from the photometry analyses (histology; 2 of 16).
EXCLUDE_RATS = ['R18', 'R21']

# Optional row filter applied BEFORE the second stage (the HMM always runs on
# full sessions). E.g. lambda d: d['Phase'] == 3   -- adjust to your columns.
ROW_FILTER = None

N_PERMUTATIONS = 1000     # within Rat x Session x Cue shuffles (primary window)
MIN_TRIALS_PER_SESSION_AIC = 5   # sessions with fewer aIC trials are dropped
P_CLIP = 1e-6
CUE_ORDER = ['reward', 'punish', 'conflict']

# =========================================================
# 1. DATA PREPARATION (identical covariates / order to 1.1 and 2.1)
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

    sessions = []
    for (rat, session), sd in df.groupby(['Rat', 'Session']):
        if len(sd) < 5:
            continue
        Y = sd['Lever_Press'].values.astype(int)
        X = sd[['rew_cue_val', 'pun_cue_val', 'past_pellet',
                'past_shock', 'bias']].astype(float).values
        sessions.append({'Rat': rat, 'Session': session, 'Indices': sd.index, 'Y': Y, 'X': X})
    print(f"Prepared {len(sessions)} sessions from {df['Rat'].nunique()} rats.")
    return df, sessions


# =========================================================
# 2. FROZEN MODEL -> CAUSAL (ONE-STEP-AHEAD) PREDICTIONS
# =========================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


def get_press_weights(hmm):
    """Same sign convention as 1.1 / 1.1b: positive = drive to press. (K, D)"""
    raw = hmm.observations.Wk
    W = raw[:, 0, :] * -1 if raw.ndim == 3 else raw.squeeze() * -1
    return np.atleast_2d(W)


def get_initial_distn(hmm, K):
    try:
        pi0 = np.asarray(hmm.init_state_distn.initial_state_distn, dtype=float)
    except Exception:
        try:
            lp = np.asarray(hmm.init_state_distn.log_pi0, dtype=float)
            pi0 = np.exp(lp - lp.max())
        except Exception:
            pi0 = np.ones(K)
    pi0 = np.clip(pi0, 1e-12, None)
    return pi0 / pi0.sum()


def forward_filter(emis, A, pi0):
    """Returns pred (T,K) = p(z_t | y_1:t-1): causal, never sees choice t."""
    T, K = emis.shape
    alpha = np.zeros((T, K))
    pred = np.zeros((T, K))
    for t in range(T):
        prior = pi0 if t == 0 else alpha[t - 1] @ A
        pred[t] = prior
        u = prior * emis[t]
        alpha[t] = u / max(u.sum(), 1e-300)
    return pred


def add_causal_predictions(df, sessions, hmm):
    W = get_press_weights(hmm)
    A = np.atleast_2d(hmm.transitions.transition_matrix)
    K = A.shape[0]
    pi0 = get_initial_distn(hmm, K)

    df['P_press_OSA'] = np.nan
    for k in range(K):
        df[f'PredState_{k+1}'] = np.nan

    for s in sessions:
        p_states = sigmoid(s['X'] @ W.T)                         # (T, K)
        emis = np.where(s['Y'][:, None] == 1, p_states, 1 - p_states)
        pred = forward_filter(emis, A, pi0)
        df.loc[s['Indices'], 'P_press_OSA'] = np.sum(pred * p_states, axis=1)
        for k in range(K):
            df.loc[s['Indices'], f'PredState_{k+1}'] = pred[:, k]

    ok = df['P_press_OSA'].notna()
    acc = np.mean((df.loc[ok, 'P_press_OSA'] >= 0.5) == (df.loc[ok, 'Lever_Press'] == 1)) * 100
    print(f"Frozen-model one-step-ahead accuracy on all trials: {acc:.1f}% "
          f"(should be close to the in-sample value behind Fig. S2).")
    return df, K


# =========================================================
# 2b. STATE LABELS
# =========================================================
def state_label(k):
    return f'State {k}\n{STATE_NAMES.get(k, "")}'


# =========================================================
# 3. LOGISTIC REGRESSION WITH A FIXED OFFSET (Newton / IRLS)
# =========================================================
def fit_logit_offset(X, y, offset, l2=1e-6, max_iter=100, tol=1e-9):
    b = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = sigmoid(X @ b + offset)
        w = p * (1 - p)
        g = X.T @ (y - p) - l2 * b
        H = X.T @ (X * w[:, None]) + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < tol:
            break
    return b


def loglik(y, p):
    p = np.clip(p, P_CLIP, 1 - P_CLIP)
    return y * np.log(p) + (1 - y) * np.log(1 - p)


def design(d, model, aic_col, K):
    n = len(d)
    one = np.ones((n, 1))
    a = d[aic_col].values[:, None]
    if model == 'Baseline':
        return one
    if model == 'aIC':
        return np.hstack([one, a])
    if model == 'aIC_x_cue':
        cols = [(d['Cue'].values == c).astype(float)[:, None] * a for c in CUE_ORDER]
        return np.hstack([one] + cols)
    if model == 'aIC_x_state':
        cols = [d[f'PredState_{k+1}'].values[:, None] * a for k in range(K)]
        return np.hstack([one] + cols)
    if model == 'aIC_conf_x_state':
        # rewDS and punDS keep their own slopes (so they cannot contaminate the
        # conflict terms); the conflict slope is split by causal state probability.
        conf = (d['Cue'].values == 'conflict').astype(float)[:, None]
        cols = [(d['Cue'].values == 'reward').astype(float)[:, None] * a,
                (d['Cue'].values == 'punish').astype(float)[:, None] * a]
        cols += [conf * d[f'PredState_{k+1}'].values[:, None] * a for k in range(K)]
        return np.hstack([one] + cols)
    if model == 'aIC_full_x_state':
        # every cue x state gets its own slope -> like-for-like within-state comparison
        cols = []
        for c in CUE_ORDER:
            cm = (d['Cue'].values == c).astype(float)[:, None]
            cols += [cm * d[f'PredState_{k+1}'].values[:, None] * a for k in range(K)]
        return np.hstack([one] + cols)
    raise ValueError(model)


def design_names(model, K):
    if model == 'Baseline':
        return ['Intercept']
    if model == 'aIC':
        return ['Intercept', 'aIC']
    if model == 'aIC_x_cue':
        return ['Intercept'] + [f'aIC:{c}' for c in CUE_ORDER]
    if model == 'aIC_x_state':
        return ['Intercept'] + [f'aIC:P(State{k+1})' for k in range(K)]
    if model == 'aIC_conf_x_state':
        return (['Intercept', 'aIC:reward', 'aIC:punish'] +
                [f'aIC:conflict:P(State{k+1})' for k in range(K)])
    return ['Intercept'] + [f'aIC:{c}:P(State{k+1})' for c in CUE_ORDER for k in range(K)]


# =========================================================
# 4. BUILD THE SECOND-STAGE DATASET FOR ONE WINDOW
# =========================================================
def build_stage2(df, raw_col):
    d = df[df['P_press_OSA'].notna() & df[raw_col].notna()].copy()
    if EXCLUDE_RATS:
        d = d[~d['Rat'].isin(EXCLUDE_RATS)]
    if ROW_FILTER is not None:
        d = d[ROW_FILTER(d)]

    # z-score within session (removes bleaching / between-rat scale differences)
    g = d.groupby(['Rat', 'Session'])[raw_col]
    sd = g.transform('std')
    d['aIC_z'] = (d[raw_col] - g.transform('mean')) / sd.replace(0, np.nan)
    counts = g.transform('count')
    d = d[(counts >= MIN_TRIALS_PER_SESSION_AIC) & d['aIC_z'].notna()]

    p = np.clip(d['P_press_OSA'].values, P_CLIP, 1 - P_CLIP)
    d['offset'] = np.log(p / (1 - p))
    d['y'] = d['Lever_Press'].values.astype(float)
    return d.reset_index(drop=True)


# =========================================================
# 5. LEAVE-ONE-RAT-OUT CV OF THE SECOND STAGE
# =========================================================
MODELS = ['Baseline', 'aIC', 'aIC_x_cue', 'aIC_x_state', 'aIC_conf_x_state', 'aIC_full_x_state']


def loro_cv(d, K, aic_col='aIC_z', models=MODELS):
    """Returns per-rat x cue x model held-out LL/trial and accuracy."""
    rows = []
    rats = sorted(d['Rat'].unique())
    for held in rats:
        tr, te = d[d['Rat'] != held], d[d['Rat'] == held]
        for m in models:
            b = fit_logit_offset(design(tr, m, aic_col, K), tr['y'].values, tr['offset'].values)
            p = sigmoid(design(te, m, aic_col, K) @ b + te['offset'].values)
            ll = loglik(te['y'].values, p)
            correct = ((p >= 0.5) == (te['y'].values == 1))
            for cue in ['all'] + CUE_ORDER:
                mask = np.ones(len(te), bool) if cue == 'all' else (te['Cue'].values == cue)
                if mask.sum() == 0:
                    continue
                rows.append({'Rat': held, 'Model': m, 'Cue': cue, 'N': int(mask.sum()),
                             'LL_per_trial': ll[mask].mean(),
                             'Accuracy': correct[mask].mean() * 100})
    return pd.DataFrame(rows)


def frozen_reference(d):
    """Pure frozen model (no free parameters at all) for reference."""
    rows = []
    p = sigmoid(d['offset'].values)
    ll = loglik(d['y'].values, p)
    correct = (p >= 0.5) == (d['y'].values == 1)
    for rat, idx in d.groupby('Rat').groups.items():
        sub = d.loc[idx]
        for cue in ['all'] + CUE_ORDER:
            mask = np.ones(len(sub), bool) if cue == 'all' else (sub['Cue'].values == cue)
            if mask.sum():
                rows.append({'Rat': rat, 'Model': 'Frozen_GLMHMM', 'Cue': cue, 'N': int(mask.sum()),
                             'LL_per_trial': ll[idx][mask].mean(),
                             'Accuracy': correct[idx][mask].mean() * 100})
    return pd.DataFrame(rows)


def paired_gain(cv, model, cue, metric='LL_per_trial', reference='Baseline'):
    piv = cv[cv['Cue'] == cue].pivot(index='Rat', columns='Model', values=metric)
    if model not in piv or reference not in piv:
        return None
    delta = (piv[model] - piv[reference]).dropna()
    t, p_t = stats.ttest_1samp(delta, 0.0)
    try:
        p_w = stats.wilcoxon(delta).pvalue
    except ValueError:
        p_w = np.nan
    return {'Model': model, 'Reference': reference, 'Cue': cue, 'Metric': metric,
            'Mean_delta': delta.mean(),
            'SEM_delta': delta.sem(), 'N_rats': len(delta),
            'N_improved': int((delta > 0).sum()), 't': t, 'p_ttest': p_t, 'p_wilcoxon': p_w}


# =========================================================
# 6. SHUFFLE CONTROL (within Rat x Session x Cue)
# =========================================================
def shuffle_null(d, K, n_perm, rng):
    """Null distribution of the CV gain (aIC vs Baseline) in mean LL/trial,
    overall and on conflict trials. Baseline CV does not depend on aIC, so it
    is computed once."""
    base = loro_cv(d, K, models=['Baseline'])
    b_all = base[base.Cue == 'all'].set_index('Rat')['LL_per_trial']
    b_conf = base[base.Cue == 'conflict'].set_index('Rat')['LL_per_trial']
    groups = d.groupby(['Rat', 'Session', 'Cue']).indices
    null_all, null_conf = np.zeros(n_perm), np.zeros(n_perm)
    for i in range(n_perm):
        shuf = d['aIC_z'].values.copy()
        for idx in groups.values():
            shuf[idx] = shuf[idx][rng.permutation(len(idx))]
        d['aIC_shuf'] = shuf
        cv = loro_cv(d, K, aic_col='aIC_shuf', models=['aIC'])
        a_all = cv[cv.Cue == 'all'].set_index('Rat')['LL_per_trial']
        a_conf = cv[cv.Cue == 'conflict'].set_index('Rat')['LL_per_trial']
        null_all[i] = (a_all - b_all).mean()
        null_conf[i] = (a_conf - b_conf).dropna().mean()
        if (i + 1) % 100 == 0:
            print(f"    permutation {i+1}/{n_perm}")
    d.drop(columns='aIC_shuf', inplace=True)
    return null_all, null_conf


# =========================================================
# 7. FULL-DATA COEFFICIENTS WITH RAT-CLUSTERED SEs
# =========================================================
def full_data_coefficients(d, K):
    import statsmodels.api as sm
    rows = []
    for m in MODELS[1:]:
        X = design(d, m, 'aIC_z', K)
        fit = sm.GLM(d['y'].values, X, family=sm.families.Binomial(),
                     offset=d['offset'].values).fit(
            cov_type='cluster', cov_kwds={'groups': pd.factorize(d['Rat'])[0]})
        ci = fit.conf_int()
        for j, name in enumerate(design_names(m, K)):
            rows.append({'Model': m, 'Term': name, 'Beta': fit.params[j], 'SE_cluster': fit.bse[j],
                         'CI_low': ci[j, 0], 'CI_high': ci[j, 1], 'z': fit.tvalues[j],
                         'p': fit.pvalues[j]})
    return pd.DataFrame(rows)



# =========================================================
# 7b. STATE-DEPENDENT CONFLICT TEST (the hypothesis-driven model)
# =========================================================
# PRIMARY criterion is the SIGN pattern (beta_S1 > 0 AND beta_S2 < 0), because
# logit slopes can be scaled differently between states (floor effects, soft
# state probabilities) but such scaling cannot flip a sign. The S1 - S2 contrast
# is reported as secondary.
# Hypothesis: on conflict trials, aIC should push toward PRESSING in State 1
# (conflict treated like rewDS) and toward OMITTING in State 2 (conflict treated
# like punDS). Prediction: beta[conflict:S1] > 0, beta[conflict:S2] < 0, so the
# contrast S1 - S2 > 0. A single conflict slope would average these away.
STATE_MODEL = 'aIC_conf_x_state'
def _role_cols(K):
    names = design_names(STATE_MODEL, K)
    return (names.index(f"aIC:conflict:P(State{ROLES['risk_taking']})"),
            names.index(f"aIC:conflict:P(State{ROLES['risk_averse']})"))


def contrast_S1_minus_S2(d, aic_col, K):
    S1_COL, S2_COL = _role_cols(K)
    b = fit_logit_offset(design(d, STATE_MODEL, aic_col, K), d['y'].values, d['offset'].values)
    return b[S1_COL] - b[S2_COL], b[S1_COL], b[S2_COL]


def state_contrast_full(d, K):
    """Cluster-robust (by rat) Wald test of beta_S1 - beta_S2 on conflict trials."""
    import statsmodels.api as sm
    X = design(d, STATE_MODEL, 'aIC_z', K)
    fit = sm.GLM(d['y'].values, X, family=sm.families.Binomial(),
                 offset=d['offset'].values).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(d['Rat'])[0]})
    S1_COL, S2_COL = _role_cols(K)
    L = np.zeros(X.shape[1]); L[S1_COL], L[S2_COL] = 1, -1
    tt = fit.t_test(L)
    b, se = fit.params, fit.bse
    return {'beta_S1': b[S1_COL], 'p_one_sided_S1>0': float(stats.norm.sf(b[S1_COL] / se[S1_COL])),
            'beta_S2': b[S2_COL], 'p_one_sided_S2<0': float(stats.norm.cdf(b[S2_COL] / se[S2_COL])),
            'Contrast': 'conflict: beta_S1 - beta_S2',
            'Estimate': float(np.squeeze(tt.effect)), 'SE_cluster': float(np.squeeze(tt.sd)),
            'z': float(np.squeeze(tt.tvalue)), 'p_two_sided': float(np.squeeze(tt.pvalue)),
            'p_one_sided_S1>S2': float(stats.norm.sf(np.squeeze(tt.tvalue)))}


def effective_state_trials(d, K):
    """Sum of causal P(state) over conflict trials = effective n per state, per rat."""
    c = d[d['Cue'] == 'conflict']
    out = c.groupby('Rat')[[f'PredState_{k+1}' for k in range(K)]].sum()
    out.columns = [f'EffN_conflict_S{k+1}' for k in range(K)]
    return out


def state_nulls(d, K, n_perm, rng):
    """
    Two permutation nulls for the state-split conflict model:

    (A) aIC shuffled within Rat x Session x Cue -> 'no trial-level aIC information'.
        Tests whether the state-split conflict terms beat an uninformative aIC.
    (B) Causal state probabilities shuffled among CONFLICT trials within
        Rat x Session (aIC, choice and the model offset stay paired) ->
        'aIC informative, but not state-dependent'. This is the null for the
        claim that STATE sorting resolves conflict. Offsets are left intact so
        the frozen model's own use of state is unchanged.

    Statistics per permutation: full-data contrast beta_S1 - beta_S2, and the
    CV conflict-trial gain of the state model over the aIC_x_cue model.
    """
    ref = loro_cv(d, K, models=['aIC_x_cue'])
    ref_conf = ref[ref.Cue == 'conflict'].set_index('Rat')['LL_per_trial']

    def cv_gain(dd, aic_col):
        cv = loro_cv(dd, K, aic_col=aic_col, models=[STATE_MODEL])
        g = cv[cv.Cue == 'conflict'].set_index('Rat')['LL_per_trial']
        return (g - ref_conf).dropna().mean()

    cue_groups = d.groupby(['Rat', 'Session', 'Cue']).indices
    conf = d['Cue'].values == 'conflict'
    conf_groups = d[conf].groupby(['Rat', 'Session']).indices
    conf_idx = np.where(conf)[0]
    pcols = [f'PredState_{k+1}' for k in range(K)]
    P = d[pcols].values

    rows = []
    for i in range(n_perm):
        # (A) aIC shuffle
        shuf = d['aIC_z'].values.copy()
        for idx in cue_groups.values():
            shuf[idx] = shuf[idx][rng.permutation(len(idx))]
        d['aIC_shuf'] = shuf
        cA, b1A, b2A = contrast_S1_minus_S2(d, 'aIC_shuf', K)
        # CV gain vs aIC_x_cue must use the same shuffled aIC for the reference
        refA = loro_cv(d, K, aic_col='aIC_shuf', models=['aIC_x_cue'])
        refA = refA[refA.Cue == 'conflict'].set_index('Rat')['LL_per_trial']
        cvA = loro_cv(d, K, aic_col='aIC_shuf', models=[STATE_MODEL])
        gA = (cvA[cvA.Cue == 'conflict'].set_index('Rat')['LL_per_trial'] - refA).dropna().mean()

        # (B) state-probability shuffle among conflict trials
        Pb = P.copy()
        for local in conf_groups.values():
            rows_ = conf_idx[local]
            Pb[rows_] = P[rows_][rng.permutation(len(rows_))]
        dB = d.copy(); dB[pcols] = Pb
        cB, _, _ = contrast_S1_minus_S2(dB, 'aIC_z', K)
        gB = cv_gain(dB, 'aIC_z')

        rows.append({'A_contrast': cA, 'A_beta_S1': b1A, 'A_beta_S2': b2A,
                     'A_cvgain_conf_vs_cue': gA,
                     'B_contrast': cB, 'B_cvgain_conf_vs_cue': gB})
        if (i + 1) % 100 == 0:
            print(f"    state-null permutation {i+1}/{n_perm}")
    d.drop(columns='aIC_shuf', inplace=True)
    return pd.DataFrame(rows)


def plot_state_test(coefs, nulls, obs_contrast, window, out):
    c = coefs[(coefs.Model == STATE_MODEL) & coefs.Term.str.startswith('aIC:conflict')]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    ax = axes[0]
    x = np.arange(len(c))
    cols = [(176/255, 179/255, 0) if k + 1 == ROLES['risk_taking'] else
            (181/255, 0, 179/255) if k + 1 == ROLES['risk_averse'] else (0.6, 0.6, 0.6)
            for k in range(len(c))]
    ax.bar(x, c['Beta'], color=cols, edgecolor='black',
           yerr=[c['Beta'] - c['CI_low'], c['CI_high'] - c['Beta']], capsize=4)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([state_label(k + 1) for k in range(len(c))])
    ax.set_ylabel('β aIC on conflict trials\n(log-odds press per SD; 95% CI, rat-clustered)')
    ax.set_title('Conflict-trial aIC slope by latent state', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    if nulls is not None:
        ax.hist(nulls['B_contrast'], bins=40, color='grey', alpha=0.6, label='State-shuffle null')
        ax.hist(nulls['A_contrast'], bins=40, color='tab:red', alpha=0.35, label='aIC-shuffle null')
    ax.axvline(obs_contrast, color='black', lw=2, label='Observed')
    ax.set_xlabel('β(risk-taking) − β(risk-averse), conflict trials')
    ax.set_ylabel('Permutations')
    ax.set_title('State-dependence test', fontweight='bold')
    ax.legend(frameon=False, fontsize=8); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    for ext in ['svg', 'png']:
        plt.savefig(os.path.join(out, f'aIC_PostHoc_StateConflictTest_{window}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


# =========================================================
# 7c. WITHIN-STATE SLOPE MATCHING (conflict vs rewDS / punDS in the SAME state)
# =========================================================
FULL_MODEL = 'aIC_full_x_state'
# (label, term A, term B). "Matched" pairs: hypothesis predicts NO difference.
# "Mismatched" pairs: hypothesis predicts a difference (conflict should look
# like the state-appropriate cue, NOT the other one).
def match_tests():
    a, v = ROLES['risk_taking'], ROLES['risk_averse']
    return [
        (f'State {a} (risk-taking): conflict vs rewDS (matched)',    f'aIC:conflict:P(State{a})', f'aIC:reward:P(State{a})'),
        (f'State {v} (risk-averse): conflict vs punDS (matched)',    f'aIC:conflict:P(State{v})', f'aIC:punish:P(State{v})'),
        (f'State {a} (risk-taking): conflict vs punDS (mismatched)', f'aIC:conflict:P(State{a})', f'aIC:punish:P(State{a})'),
        (f'State {v} (risk-averse): conflict vs rewDS (mismatched)', f'aIC:conflict:P(State{v})', f'aIC:reward:P(State{v})'),
    ]


def fit_full_state_model(d, K):
    import statsmodels.api as sm
    X = design(d, FULL_MODEL, 'aIC_z', K)
    fit = sm.GLM(d['y'].values, X, family=sm.families.Binomial(),
                 offset=d['offset'].values).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(d['Rat'])[0]})
    return fit, design_names(FULL_MODEL, K)


def slope_matching_tests(d, K):
    """Wald tests using the FULL rat-clustered covariance (slopes share rats,
    so they are not independent)."""
    fit, names = fit_full_state_model(d, K)
    idx = {n: i for i, n in enumerate(names)}
    ci = fit.conf_int()
    coef = pd.DataFrame({'Term': names, 'Beta': fit.params, 'SE_cluster': fit.bse,
                         'CI_low': ci[:, 0], 'CI_high': ci[:, 1], 'p': fit.pvalues})
    rows = []
    for lab, a, b in match_tests():
        L = np.zeros(len(names)); L[idx[a]], L[idx[b]] = 1, -1
        tt = fit.t_test(L)
        rows.append({'Test': lab, 'Beta_A': fit.params[idx[a]], 'Beta_B': fit.params[idx[b]],
                     'Difference': float(np.squeeze(tt.effect)), 'SE': float(np.squeeze(tt.sd)),
                     'z': float(np.squeeze(tt.tvalue)), 'p_two_sided': float(np.squeeze(tt.pvalue))})
    return coef, pd.DataFrame(rows)


def effective_cue_state_trials(d, K):
    pcols = [f'PredState_{k+1}' for k in range(K)]
    out = d.groupby('Cue')[pcols].sum().reindex(CUE_ORDER)
    out.columns = [f'EffN_S{k+1}' for k in range(K)]
    return out


def plot_slope_matching(coef, window, out):
    """Per state: matched cue + conflict (bold) and the mismatched cue (thin)."""
    c = coef.set_index('Term')
    G, R = '#2DB84B', '#E0202A'
    S1, S2 = (176/255, 179/255, 0), (181/255, 0, 179/255)
    a, v = ROLES['risk_taking'], ROLES['risk_averse']
    x = np.linspace(-2, 2, 100)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    spec = [(axes[0], f'State {a}: {STATE_NAMES[a]}',
             [(f'aIC:reward:P(State{a})', G, 'rewDS', '-', 2.2, 0.18),
              (f'aIC:conflict:P(State{a})', S1, 'Conflict', '--', 2.2, 0.18),
              (f'aIC:punish:P(State{a})', R, 'punDS', '-', 1.4, 0.08)]),
            (axes[1], f'State {v}: {STATE_NAMES[v]}',
             [(f'aIC:punish:P(State{v})', R, 'punDS', '-', 2.2, 0.18),
              (f'aIC:conflict:P(State{v})', S2, 'Conflict', '--', 2.2, 0.18),
              (f'aIC:reward:P(State{v})', G, 'rewDS', '-', 1.4, 0.08)])]
    for ax, title, lines in spec:
        for term, col, lab, ls, lw, alpha in lines:
            b, se = c.loc[term, 'Beta'], c.loc[term, 'SE_cluster']
            ax.fill_between(x, (b - 1.96 * se) * x, (b + 1.96 * se) * x, color=col, alpha=alpha, lw=0)
            ax.plot(x, b * x, ls, color=col, lw=lw, alpha=1 if lw > 2 else 0.75,
                    label=f'{lab}  β = {b:+.2f}')
        ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Cue-period aIC (SD from session mean)')
        ax.legend(frameon=False, fontsize=8.5, loc='upper left')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Change in log-odds of pressing\n(relative to GLM-HMM prediction)')
    lim = max(abs(np.array(axes[0].get_ylim())).max(), abs(np.array(axes[1].get_ylim())).max())
    axes[0].set_ylim(-lim, lim)
    plt.tight_layout()
    for ext in ['svg', 'png']:
        plt.savefig(os.path.join(out, f'aIC_SlopeMatching_WithinState_{window}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_slope_grid(coef, K, window, out):
    """All cue x state slopes, grouped by state."""
    c = coef.set_index('Term')
    cue_col = {'reward': '#2DB84B', 'punish': '#E0202A', 'conflict': '#3050E0'}
    cue_lab = {'reward': 'rewDS', 'punish': 'punDS', 'conflict': 'Conflict'}
    snames = [state_label(k + 1) for k in range(K)]
    fig, ax = plt.subplots(figsize=(2.2 * K, 3.8))
    w = 0.26
    for j, cue in enumerate(CUE_ORDER):
        terms = [f'aIC:{cue}:P(State{k+1})' for k in range(K)]
        b = c.loc[terms, 'Beta'].values
        lo, hi = c.loc[terms, 'CI_low'].values, c.loc[terms, 'CI_high'].values
        xs = np.arange(K) + (j - 1) * w
        ax.bar(xs, b, w, color=cue_col[cue], edgecolor='black', label=cue_lab[cue])
        ax.errorbar(xs, b, yerr=[b - lo, hi - b], fmt='none', ecolor='black', capsize=3, lw=1)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(K)); ax.set_xticklabels(snames)
    ax.set_ylabel('aIC → choice slope (β)\n95% CI, rat-clustered')
    ax.set_title('aIC–choice slopes by cue and latent state', fontweight='bold')
    ax.legend(frameon=False, fontsize=8.5); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    for ext in ['svg', 'png']:
        plt.savefig(os.path.join(out, f'aIC_SlopeGrid_CueByState_{window}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


# =========================================================
# 8. PLOTS
# =========================================================
def plot_primary(cv, window, null_all, null_conf, out):
    labels = {'all': 'All trials', 'reward': 'rewDS', 'punish': 'punDS', 'conflict': 'Conflict'}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, cue in enumerate(['all'] + CUE_ORDER):
        piv = cv[cv.Cue == cue].pivot(index='Rat', columns='Model', values='LL_per_trial')
        delta = (piv['aIC'] - piv['Baseline']).dropna() * 1000
        jitter = np.random.uniform(-0.12, 0.12, len(delta))
        ax.scatter(np.full(len(delta), i) + jitter, delta, color='grey', s=18, zorder=2)
        ax.errorbar(i, delta.mean(), yerr=delta.sem(), fmt='o', color='black',
                    capsize=4, markersize=7, zorder=3)
    for i, null in [(0, null_all), (3, null_conf)]:
        if null is not None:
            lo, hi = np.percentile(null * 1000, [2.5, 97.5])
            ax.fill_between([i - 0.3, i + 0.3], lo, hi, color='tab:red', alpha=0.2, lw=0,
                            label='Shuffle 95%' if i == 0 else None)
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.set_xticks(range(4)); ax.set_xticklabels([labels[c] for c in ['all'] + CUE_ORDER])
    ax.set_ylabel('Held-out ΔLL per trial (×10⁻³ nats)\naIC model − frozen GLM-HMM')
    ax.set_title(f'Post-hoc aIC gain ({window})', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    if null_all is not None:
        ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    for ext in ['svg', 'png']:
        plt.savefig(os.path.join(out, f'aIC_PostHoc_Gain_{window}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_windows(gains, out, null_all=None, null_conf=None, p_all=None, p_conf=None):
    """BL (control) vs CueFull, single-slope aIC model, held-out gain over the
    frozen GLM-HMM. Bars = mean +/- SEM across rats. Yellow line = 95th
    percentile of the within Rat x Session x Cue shuffle null (CueFull only)."""
    g = gains[(gains.Model == 'aIC') & (gains.Metric == 'LL_per_trial') &
              gains.Cue.isin(['all', 'conflict'])]
    if 'Reference' in g:
        g = g[g.Reference == 'Baseline']
    wins = ['ZMean_BL', PRIMARY_WINDOW]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    x = np.arange(len(wins))
    top = 0
    for j, (cue, col, lab, null, p) in enumerate([
            ('all', '#23395B', 'All trials', null_all, p_all),
            ('conflict', '#3050E0', 'Conflict trials', null_conf, p_conf)]):
        sub = g[g.Cue == cue].set_index('Window').reindex(wins)
        xs = x + (j - 0.5) * 0.36
        m, e = sub['Mean_delta'].values * 1e3, sub['SEM_delta'].values * 1e3
        ax.bar(xs, m, 0.36, yerr=e, capsize=3, color=col, edgecolor='black', label=lab, zorder=2)
        top = max(top, np.nanmax(m + e))
        if null is not None:
            p95 = np.percentile(null * 1e3, 95)
            ax.plot([xs[1] - 0.17, xs[1] + 0.17], [p95, p95], color='#FFB000', lw=2.2, zorder=4,
                    label='Shuffle null, 95th percentile' if j == 0 else None)
        if p is not None:
            ax.text(xs[1], m[1] + e[1] + 0.06 * top, f'p = {p:.3f}', ha='center', fontsize=8)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(['Pre-trial\n(BL)', 'Cue period\n(0–10 s)'])
    ax.set_ylabel('Held-out Δ log-likelihood per trial (×10⁻³)\n'
                  'aIC + model vs model alone')
    ax.set_title('aIC improves choice prediction', fontweight='bold')
    ax.set_ylim(min(-0.4, ax.get_ylim()[0]), top * 1.25)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    for ext in ['svg', 'png']:
        plt.savefig(os.path.join(out, f'aIC_PostHoc_Window_BL_vs_CueFull.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_coefficients(coefs, window, out):
    c = coefs[coefs.Model.isin(['aIC_x_cue', 'aIC_x_state']) & (coefs.Term != 'Intercept')]
    fig, ax = plt.subplots(figsize=(6, 4))
    y = np.arange(len(c))[::-1]
    ax.errorbar(c['Beta'], y, xerr=[c['Beta'] - c['CI_low'], c['CI_high'] - c['Beta']],
                fmt='o', color='black', capsize=3)
    ax.axvline(0, color='k', ls=':', lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(c['Term'])
    ax.set_xlabel('β (log-odds of press per SD aIC), 95% CI (rat-clustered)')
    ax.set_title(f'Where does aIC add information? ({window})', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    for ext in ['svg', 'png']:
        plt.savefig(os.path.join(out, f'aIC_PostHoc_Coefficients_{window}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# =========================================================
# 9. MAIN
# =========================================================
def main(hmm=None, df=None, n_perm=N_PERMUTATIONS):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if df is None:
        df = pd.read_csv(INPUT_PATH, dtype={"Session": str})
    if hmm is None:
        with open(MODEL_PATH, 'rb') as f:
            hmm = pickle.load(f)

    df, sessions = prepare_hmm_data(df)
    df, K = add_causal_predictions(df, sessions, hmm)
    if K != 3:
        print(f"WARNING: expected a 3-state model, loaded {K} states.")

    windows = WINDOWS + (['ZMean_Lever'] if INCLUDE_LEVER else [])
    windows = [w for w in windows if w in df.columns]
    if not windows:
        print(f"WARNING: none of the photometry columns {WINDOWS} is in {INPUT_PATH}; nothing to do.")
    rng = np.random.default_rng(14)

    all_cv, all_gains = [], []
    null_all = null_conf = None
    p_shuffle = {}
    for w in windows:
        d = build_stage2(df, w)
        print(f"\n=== Window {w}: {len(d)} trials, {d['Rat'].nunique()} rats, "
              f"{d.groupby(['Rat','Session']).ngroups} sessions ===")
        print(d['Cue'].value_counts().to_string())

        cv = pd.concat([loro_cv(d, K), frozen_reference(d)], ignore_index=True)
        cv['Window'] = w
        all_cv.append(cv)

        for m in MODELS[1:]:
            for cue in ['all'] + CUE_ORDER:
                for metric in ['LL_per_trial', 'Accuracy']:
                    r = paired_gain(cv, m, cue, metric)
                    if r:
                        r['Window'] = w
                        all_gains.append(r)
        # Does splitting the conflict slope by state beat a single conflict slope?
        for cue in ['all', 'conflict']:
            for metric in ['LL_per_trial', 'Accuracy']:
                r = paired_gain(cv, STATE_MODEL, cue, metric, reference='aIC_x_cue')
                if r:
                    r['Window'] = w
                    all_gains.append(r)

        if w == PRIMARY_WINDOW:
            if n_perm > 0:
                print(f"  Running {n_perm} within Rat x Session x Cue shuffles...")
                null_all, null_conf = shuffle_null(d, K, n_perm, rng)
                g = pd.DataFrame(all_gains)
                obs = g[(g.Window == w) & (g.Model == 'aIC') & (g.Metric == 'LL_per_trial')].set_index('Cue')
                for cue, null in [('all', null_all), ('conflict', null_conf)]:
                    o = obs.loc[cue, 'Mean_delta']
                    p = (np.sum(null >= o) + 1) / (len(null) + 1)
                    p_shuffle[cue] = p
                    print(f"  Shuffle test ({cue}): observed ΔLL = {o*1000:+.3f}e-3, "
                          f"null 95th pct = {np.percentile(null,95)*1000:+.3f}e-3, p = {p:.4f}")
                pd.DataFrame({'null_all': null_all, 'null_conflict': null_conf}).to_csv(
                    os.path.join(OUTPUT_DIR, f'Shuffle_Null_{w}.csv'), index=False)

            coefs = full_data_coefficients(d, K)
            coefs.to_csv(os.path.join(OUTPUT_DIR, f'FullData_Coefficients_{w}.csv'), index=False)
            print("\n  Full-data coefficients (rat-clustered SE):")
            print(coefs.round(4).to_string(index=False))
            plot_primary(cv, w, null_all, null_conf, OUTPUT_DIR)
            plot_coefficients(coefs, w, OUTPUT_DIR)

            # ---- Hypothesis test: state-dependent conflict slopes ----
            print("\n  === State-dependent conflict test ===")
            eff = effective_state_trials(d, K)
            eff.to_csv(os.path.join(OUTPUT_DIR, f'Conflict_EffectiveTrials_byState_{w}.csv'))
            print("  Effective conflict trials per state (sum of causal P(state)), per rat:")
            print(eff.round(1).to_string())
            con = state_contrast_full(d, K)
            print(f"  SIGN TEST (primary): beta_risk-taking(S{ROLES['risk_taking']}) = {con['beta_S1']:+.3f} (1-sided p>0 = {con['p_one_sided_S1>0']:.4f}); "
                  f"beta_risk-averse(S{ROLES['risk_averse']}) = {con['beta_S2']:+.3f} (1-sided p<0 = {con['p_one_sided_S2<0']:.4f}). "
                  f"Hypothesis supported only if BOTH are significant (intersection-union test).")
            print(f"  beta_S1 - beta_S2 (risk-taking - risk-averse) = {con['Estimate']:+.3f} (SE {con['SE_cluster']:.3f}), "
                  f"z = {con['z']:.2f}, p(2-sided) = {con['p_two_sided']:.4f}, "
                  f"p(1-sided, S1>S2) = {con['p_one_sided_S1>S2']:.4f}")
            nulls = None
            if n_perm > 0:
                print(f"  Running {n_perm} permutations for the two state-test nulls...")
                nulls = state_nulls(d, K, n_perm, rng)
                nulls.to_csv(os.path.join(OUTPUT_DIR, f'StateTest_Nulls_{w}.csv'), index=False)
                g = pd.DataFrame(all_gains)
                gobs = g[(g.Window == w) & (g.Model == STATE_MODEL) & (g.Reference == 'aIC_x_cue')
                         & (g.Cue == 'conflict') & (g.Metric == 'LL_per_trial')]['Mean_delta'].iloc[0]
                for tag, lab in [('A', 'aIC shuffle'), ('B', 'state shuffle')]:
                    pc = (np.sum(nulls[f'{tag}_contrast'] >= con['Estimate']) + 1) / (n_perm + 1)
                    pg = (np.sum(nulls[f'{tag}_cvgain_conf_vs_cue'] >= gobs) + 1) / (n_perm + 1)
                    con[f'p_perm_contrast_{tag}'] = pc
                    con[f'p_perm_cvgain_{tag}'] = pg
                    if tag == 'A':
                        con['p_perm_S1>0_A'] = (np.sum(nulls['A_beta_S1'] >= con['beta_S1']) + 1) / (n_perm + 1)
                        con['p_perm_S2<0_A'] = (np.sum(nulls['A_beta_S2'] <= con['beta_S2']) + 1) / (n_perm + 1)
                        print(f"  aIC shuffle  : beta_S1 p = {con['p_perm_S1>0_A']:.4f}, "
                              f"beta_S2 p = {con['p_perm_S2<0_A']:.4f}")
                    print(f"  {lab:13s}: contrast p = {pc:.4f} | CV conflict gain over "
                          f"aIC_x_cue ({gobs*1000:+.3f}e-3) p = {pg:.4f}")
            pd.DataFrame([con]).to_csv(os.path.join(OUTPUT_DIR, f'StateTest_Contrast_{w}.csv'), index=False)
            plot_state_test(coefs, nulls, con['Estimate'], w, OUTPUT_DIR)

            # ---- Within-state slope matching ----
            print("\n  === Within-state slope matching (all cues split by state) ===")
            effcs = effective_cue_state_trials(d, K)
            print("  Effective trials per cue x state (pooled over rats):")
            print(effcs.round(0).to_string())
            fcoef, mtests = slope_matching_tests(d, K)
            fcoef.to_csv(os.path.join(OUTPUT_DIR, f'WithinState_Slopes_{w}.csv'), index=False)
            mtests.to_csv(os.path.join(OUTPUT_DIR, f'WithinState_MatchingTests_{w}.csv'), index=False)
            print(fcoef.round(4).to_string(index=False))
            print("\n  Matched pairs: hypothesis predicts NO difference. "
                  "Mismatched pairs: hypothesis predicts a difference.")
            print(mtests.round(4).to_string(index=False))
            plot_slope_matching(fcoef, w, OUTPUT_DIR)
            plot_slope_grid(fcoef, K, w, OUTPUT_DIR)

    if not all_cv:
        return None, None
    cv_df = pd.concat(all_cv, ignore_index=True)
    gains = pd.DataFrame(all_gains)
    cv_df.to_csv(os.path.join(OUTPUT_DIR, 'CV_PerRat_PerCue.csv'), index=False)
    gains.to_csv(os.path.join(OUTPUT_DIR, 'CV_Paired_Gains.csv'), index=False)
    if PRIMARY_WINDOW in windows and 'ZMean_BL' in windows:
        plot_windows(gains, OUTPUT_DIR, null_all, null_conf,
                     p_shuffle.get('all'), p_shuffle.get('conflict'))

    print("\n--- Held-out gain over frozen GLM-HMM (LL/trial, x1e-3; paired across rats) ---")
    show = gains[gains.Metric == 'LL_per_trial'].copy()
    show['Mean_x1e3'] = show['Mean_delta'] * 1000
    show['SEM_x1e3'] = show['SEM_delta'] * 1000
    print(show[['Window', 'Model', 'Cue', 'Mean_x1e3', 'SEM_x1e3', 'N_improved', 'N_rats',
                'p_ttest', 'p_wilcoxon']].round(4).to_string(index=False))
    print(f"\nDone. Outputs in '{OUTPUT_DIR}/'.")
    return cv_df, gains


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Post-hoc aIC choice prediction with the frozen GLM-HMM (Fig. S4).")
    ap.add_argument("--n-perm", type=int, default=N_PERMUTATIONS,
                    help="permutations for the shuffle nulls (default %(default)s; 0 = skip)")
    a = ap.parse_args()
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    main(n_perm=a.n_perm)
