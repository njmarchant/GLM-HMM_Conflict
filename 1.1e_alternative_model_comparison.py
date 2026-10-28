"""
1.1e_alternative_model_comparison.py   (v2.2)
============================================

Does the 3-state GLM-HMM beat models in which a SINGLE policy integrates the
reward and punishment cues, but is allowed to change over time?

Every model is scored by the same quantity the forward algorithm gives the HMM
in 1.1b: the causal one-step-ahead log-likelihood sum_t log p(y_t | y_1:t-1),
on rats (or sessions) not used for fitting. The HMM is NOT refit here; its
leave-one-rat-out results are read from 1.1b and paired with these by rat.

Models (keys for --models in brackets)
--------------------------------------
Fixed integration
  [glm_static]          1-state logistic GLM on the HMM's five inputs (rewDS,
                        punDS, past pellet, past shock, bias). One policy for all
                        trials and rats.

Learned integration: Q-learning with a Pavlovian bias (Guitart-Masip et al.,
2012, NeuroImage; Cavanagh et al., 2013, J Neurosci), adapted to a discriminated
operant task. Two learned quantities per cue feature f:
    Q_f  instrumental value of pressing, updated only after a press, towards the
         outcome that press produced;
    V_f  Pavlovian value of the cue, updated on EVERY trial towards whatever
         followed it (pellet, shock, or nothing on withheld trials). The
         nothing-followed update is the negative prediction error on an omitted
         rewDS trial, and lets successful avoidance on punDS extinguish fear.
    outcome utility  u = rho_r * pellet - rho_p * shock   (0 when nothing)
    learning rate by outcome TYPE: a_r (pellet), a_p (shock), a_0 (nothing)
    p(press) = sigmoid( Q(cue) + pi * V(cue) + b + h_pel*pastPellet + h_shk*pastShock )
  The value of withholding is 0 (it never yields an outcome). Values carry over
  between sessions within a rat.
  [qpav_elemental]      conflict = rewDS + punDS elements (summation; both
                        elements share the prediction error).
  [qpav_configural]     conflict has its own Q and V (no summation).
  [qpav_elemental_rat]  as above, but parameters fitted PER RAT, starting from and
  [qpav_configural_rat] weakly shrunk towards the population fit (Gaussian prior,
                        SD 2 in unconstrained space). Cross-validated WITHIN rat:
                        5 folds of sessions (sessions 1,6,11 | 2,7,12 | ...). Each
                        test session is predicted causally from a model that never
                        fitted its choices. NB these models see the test rat's other
                        sessions, which the leave-one-rat-out HMM never does, so the
                        comparison is deliberately biased in their favour.

Gradual drift: dynamic logistic GLM on the same five inputs, weights estimated
online by an approximate Bayesian filter (Laplace update, probit predictive).
  [drift_session]       random walk  w_t = w_{t-1} + e,  reset to N(w0, s0^2) each session
  [drift_carry]         random walk carried over days (+ between-session variance)
  [drift_ar1_session]   mean-reverting AR(1): w_t = mu + phi*(w_{t-1} - mu) + e, with
                        stationary SD s0 per weight; each session starts at the
                        stationary distribution. Unlike the random walk, its
                        variance does not grow over a session.
  [drift_ar1_carry]     AR(1) carried over days (extra reversion phi_day between sessions)

Controls (no drift, separate weights per session or per rat, learned online)
  [glm_session_re], [glm_rat_re]

Inputs (same folder layout as 1.1 / 1.1b)
-----------------------------------------
  Data/collated_2026_conflict_data_raw.csv
  HMM_Outputs/CV_HeldOut_LL_by_State.csv, CV_OneStepAhead_Accuracy.csv   (1.1b)
  HMM_Outputs/3_State_GLM_Weights.csv, 3_State_Transitions.csv           (1.1; for --ppc)

Outputs (HMM_Outputs/; new runs MERGE into saved results, re-run models replaced)
---------------------------------------------------------------------------------
  AltModels_CV_per_rat.csv, AltModels_CV_per_cue.csv
  AltModels_fullfit_params.csv        population fits (per-rat models: median over rats)
  AltModels_perrat_params.csv         per-rat fits of the *_rat models
  AltModels_PPC_windows.csv           simulated block response rates (with --ppc)
  AltModels_PPC_summary.csv           whole-distribution check per model (distance, p)
  AltModels_PPC_behaviour.csv / _summary.csv   outcome effect on the next conflict choice and
                                      coupling of conflict with punDS/rewDS responding,
                                      observed vs simulated (see behaviour_stats)
  AltModels_stats.txt / _stats_table.csv   includes the PPC section if its files exist
  SuppFig_AltModels_LL / _Accuracy / _PPC  (.svg/.png)
The main comparison figure is drawn from these files by 1.1g_plot_model_comparison_figure.py.

Usage
-----
  python 1.1e_alternative_model_comparison.py --ppc            # all models, with the predictive check
  python 1.1e_alternative_model_comparison.py --models qpav_configural_rat --ppc
      (only the listed models; results are MERGED into the saved files)
  python 1.1e_alternative_model_comparison.py --stats-only     # stats/plots from saved files
  python 1.1e_alternative_model_comparison.py --stats-only --ppc   # also (re)simulate default models

Requires numpy, scipy, pandas, matplotlib and jax (recommended; `pip install jax`)
or autograd (installed with ssm; much slower).
"""

import os
import sys
import time
import argparse

import numpy as onp                      # plain numpy for data handling
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

# ---- Differentiation backend --------------------------------------------
try:
    if os.environ.get('ALTMODELS_BACKEND', '').lower() == 'autograd':
        raise ImportError                       # force the autograd fallback
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as np
    BACKEND = 'jax'

    def make_value_and_grad(f):
        return jax.jit(jax.value_and_grad(f))

    def jit(f):
        return jax.jit(f)

    def scan(step, carry, xs):
        return jax.lax.scan(step, carry, xs)
except ImportError:
    import autograd.numpy as np
    from autograd import value_and_grad as _ag_vg
    BACKEND = 'autograd'

    def make_value_and_grad(f):
        return _ag_vg(f)

    def jit(f):
        return f

    def scan(step, carry, xs):
        """Python-loop equivalent of jax.lax.scan (time on axis 0 of every xs)."""
        outs = []
        for t in range(xs[0].shape[0]):
            carry, o = step(carry, tuple(x[t] for x in xs))
            outs.append(o)
        return carry, tuple(np.stack([o[i] for o in outs]) for i in range(len(outs[0])))

# =========================================================
# 0. CONFIGURATION
# =========================================================
onp.random.seed(14)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

INPUT_PATH  = os.path.join("Data", "collated_2026_conflict_data_raw.csv")
OUTPUT_DIR  = "HMM_Outputs"
CHOSEN_STATES = 3

MAXITER       = 500      # L-BFGS iterations per fit
NEWTON_STEPS  = 6        # inner Newton steps for the Laplace update (drift models)
PPC_N_SIMS    = 50       # simulated datasets per model for the PPC
PPC_WINDOW    = 8        # consecutive conflict trials per PPC block
PERRAT_FOLDS  = 5        # within-rat session folds for the *_rat models
PERRAT_PRIOR_SD = 2.0    # shrinkage of per-rat parameters towards the population fit

INPUT_NAMES = ['rewDS', 'punDS', 'past_pellet', 'past_shock', 'bias']

MODEL_LABELS = {
    'glm_static':          'GLM static',
    'qpav_elemental':      'Q-learning + Pavlovian (elemental)',
    'qpav_configural':     'Q-learning + Pavlovian (configural)',
    'qpav_elemental_rat':  'Q-learning + Pavlovian (elemental, per rat)',
    'qpav_configural_rat': 'Q-learning + Pavlovian (configural, per rat)',
    'drift_session':       'GLM drift (session reset)',
    'drift_carry':         'GLM drift (carry over days)',
    'drift_ar1_session':   'GLM drift AR(1) (session reset)',
    'drift_ar1_carry':     'GLM drift AR(1) (carry over days)',
    'glm_session_re':      'GLM per-session weights (no drift)',
    'glm_rat_re':          'GLM per-rat weights (no drift)',
}
DEFAULT_MODELS = list(MODEL_LABELS)
HMM_LABEL = f'{CHOSEN_STATES}-state GLM-HMM'


def is_per_rat(name):
    return name.endswith('_rat')


def base_of(name):
    return name[:-4] if is_per_rat(name) else name


# =========================================================
# 1. DATA PREPARATION  (covariates identical to 1.1 / 1.1b)
# =========================================================
def prepare_trials(df):
    df = df.sort_values(by=['Rat', 'Session', 'Trial']).copy()
    g = [df['Rat'], df['Session']]
    df['pellet_now'] = (df['Response'] == 'Response-Pellet').astype(int)
    df['shock_now']  = (df['Response'] == 'Response-Shock').astype(int)
    df['past_pellet'] = df['pellet_now'].groupby(g).shift(1).fillna(0)
    df['past_shock']  = df['shock_now'].groupby(g).shift(1).fillna(0)
    df['rew_cue_val'] = ((df['Cue'] == 'reward') | (df['Cue'] == 'conflict')).astype(float)
    df['pun_cue_val'] = ((df['Cue'] == 'punish') | (df['Cue'] == 'conflict')).astype(float)
    df['bias'] = 1.0
    df['Lever_Press'] = df['Outcome'].astype(int)

    sessions = []
    for (rat, sess), s in df.groupby(['Rat', 'Session'], sort=True):
        if len(s) < 5:                                     # same rule as 1.1
            continue
        sessions.append({
            'Rat': rat, 'Session': sess,
            'X': s[['rew_cue_val', 'pun_cue_val', 'past_pellet', 'past_shock', 'bias']].values.astype(float),
            'y': s['Lever_Press'].values.astype(float),
            'pellet': s['pellet_now'].values.astype(float),
            'shock': s['shock_now'].values.astype(float),
            'cue': s['Cue'].values.astype(str),
        })
    rats = sorted({s['Rat'] for s in sessions})
    print(f"Prepared {len(sessions)} sessions, {sum(len(s['y']) for s in sessions)} trials, "
          f"{len(rats)} rats.")
    return sessions, rats


def pack(sessions, by='session'):
    """
    Pack sessions into padded arrays (S sequences x T trials) for batched filtering.
      by='session' : one sequence per session (state/values reset each session)
      by='rat'     : one sequence per rat, sessions concatenated in order;
                     'newsess' marks the first trial of every session after the first.
    'sess_pos' gives each trial's session position within its sequence.
    """
    if by == 'session':
        groups = [[s] for s in sessions]
    else:
        order = {}
        for s in sessions:
            order.setdefault(s['Rat'], []).append(s)
        groups = [order[r] for r in sorted(order)]

    S = len(groups)
    T = max(sum(len(s['y']) for s in g) for g in groups)
    D = sessions[0]['X'].shape[1]
    out = {k: onp.zeros((S, T)) for k in ['y', 'mask', 'newsess', 'pellet', 'shock',
                                         'is_rew', 'is_pun', 'is_conf', 'sess_pos']}
    out['X'] = onp.zeros((S, T, D))
    for i, g in enumerate(groups):
        t0 = 0
        for j, s in enumerate(g):
            n = len(s['y'])
            sl = slice(t0, t0 + n)
            out['X'][i, sl] = s['X']
            out['y'][i, sl] = s['y']
            out['mask'][i, sl] = 1.0
            out['pellet'][i, sl] = s['pellet']
            out['shock'][i, sl] = s['shock']
            out['is_rew'][i, sl] = (s['cue'] == 'reward')
            out['is_pun'][i, sl] = (s['cue'] == 'punish')
            out['is_conf'][i, sl] = (s['cue'] == 'conflict')
            out['sess_pos'][i, sl] = j
            if j > 0:
                out['newsess'][i, t0] = 1.0
            t0 += n
    out['F_el'] = onp.stack([out['X'][:, :, 0], out['X'][:, :, 1]], axis=-1)            # cue elements
    out['F_conf'] = onp.stack([out['is_rew'], out['is_pun'], out['is_conf']], axis=-1)  # trial types
    U = onp.stack([out['is_conf'] * out['pellet'], out['is_conf'] * out['shock'],
                   out['is_rew'] * out['pellet'], out['is_pun'] * out['shock']], axis=-1)
    out['U_prev'] = onp.zeros_like(U)                  # events on t-1 drive the t-1 -> t transition
    out['U_prev'][:, 1:] = U[:, :-1]
    out['U_prev'] *= (1 - out['newsess'])[..., None]  # never across a session boundary
    return out


# =========================================================
# 2. MODEL DEFINITIONS (step functions run with scan; no in-place assignment)
# =========================================================
def log_sigmoid(z):
    return -np.logaddexp(0.0, -z)


def sigmoid(z):
    return 0.5 * (np.tanh(0.5 * z) + 1.0)


def bernoulli_ll(z, y, mk):
    return mk * (y * log_sigmoid(z) + (1.0 - y) * log_sigmoid(-z))


def time_major(*arrs):
    return tuple(np.swapaxes(a, 0, 1) for a in arrs)


def _laplace_update(m, P, x, y):
    """Predictive logit and exact single-trial Laplace update for logistic regression
    with prior N(m, P). Returns (z_pred, m_post, P_post)."""
    Px = np.einsum('sij,sj->si', P, x)
    mu = np.sum(m * x, axis=1)
    s2 = np.sum(x * Px, axis=1)
    z = mu / np.sqrt(1.0 + np.pi * s2 / 8.0)     # probit approximation to E[sigmoid(w.x)]
    a = np.zeros_like(mu)                         # posterior mode: w = m + P x a
    for _ in range(NEWTON_STEPS):
        sg = sigmoid(mu + s2 * a)
        a = a - (a - y + sg) / (1.0 + s2 * sg * (1.0 - sg))
    sg = sigmoid(mu + s2 * a)
    r = sg * (1.0 - sg)
    m_post = m + Px * a[:, None]
    P_post = P - (r / (1.0 + r * s2))[:, None, None] * np.einsum('si,sj->sij', Px, Px)
    return z, m_post, P_post


# ---------- 2a. Random-walk drift ----------
def drift_unpack(theta, D, carry):
    w0 = theta[:D]
    s0 = np.exp(theta[D:2 * D])                 # prior SD at session (or rat) start
    sd = np.exp(theta[2 * D:3 * D])             # trial-to-trial drift SD
    sday = np.exp(theta[3 * D]) if carry else 0.0
    return w0, s0, sd, sday


def drift_init(D, carry):
    th = [0.0] * D + [onp.log(1.0)] * D + [onp.log(0.05)] * D
    if carry:
        th += [onp.log(0.3)]
    return onp.array(th)


def drift_filter(theta, A, carry):
    S, T, D = A['X'].shape
    w0, s0, sd, sday = drift_unpack(theta, D, carry)
    m0 = np.ones((S, 1)) * w0[None, :]
    P0 = np.ones((S, 1, 1)) * np.diag(s0 ** 2)[None, :, :]
    Q = np.diag(sd ** 2)[None, :, :]
    Iday = (sday ** 2) * np.eye(D)[None, :, :]

    def step(c, inp):
        m, P = c
        x, y, mk, ns = inp
        P = P + ns[:, None, None] * Iday
        z, m_new, P_new = _laplace_update(m, P, x, y)
        P_new = P_new + Q
        k1, k2 = mk[:, None], mk[:, None, None]
        return (k1 * m_new + (1.0 - k1) * m, k2 * P_new + (1.0 - k2) * P), (bernoulli_ll(z, y, mk), sigmoid(z))

    xs = time_major(A['X'], A['y'], A['mask'], A['newsess'] * (1.0 if carry else 0.0))
    _, (ll, p) = scan(step, (m0, P0), xs)
    return np.sum(ll), np.swapaxes(p, 0, 1)


# ---------- 2b. Mean-reverting AR(1) drift ----------
def ar1_unpack(theta, D, carry):
    mu = theta[:D]
    s0 = np.exp(theta[D:2 * D])                 # stationary SD of each weight
    phi = sigmoid(theta[2 * D:3 * D])           # trial-to-trial autocorrelation (0-1)
    phi_day = sigmoid(theta[3 * D]) if carry else 0.0   # session-to-session autocorrelation
    return mu, s0, phi, phi_day


def ar1_init(D, carry):
    th = [0.0] * D + [onp.log(1.0)] * D + [onp.log(0.95 / 0.05)] * D
    if carry:
        th += [0.0]
    return onp.array(th)


def ar1_filter(theta, A, carry):
    """w_t = mu + phi (w_{t-1} - mu) + e,  Var(e) = s0^2 (1 - phi^2): stationary at N(mu, s0^2)."""
    S, T, D = A['X'].shape
    mu, s0, phi, phi_day = ar1_unpack(theta, D, carry)
    S0 = np.diag(s0 ** 2)[None, :, :]
    Qd = np.diag(s0 ** 2 * (1.0 - phi ** 2))[None, :, :]
    PP = np.outer(phi, phi)[None, :, :]
    m0 = np.ones((S, 1)) * mu[None, :]
    P0 = np.ones((S, 1, 1)) * S0

    def step(c, inp):
        m, P = c
        x, y, mk, ns = inp
        g = 1.0 - ns * (1.0 - phi_day)              # reversion at session boundaries (carry only)
        m = mu[None, :] + g[:, None] * (m - mu[None, :])
        P = (g ** 2)[:, None, None] * P + (1.0 - g ** 2)[:, None, None] * S0
        z, m_post, P_post = _laplace_update(m, P, x, y)
        m_new = mu[None, :] + phi[None, :] * (m_post - mu[None, :])
        P_new = P_post * PP + Qd
        k1, k2 = mk[:, None], mk[:, None, None]
        return (k1 * m_new + (1.0 - k1) * m, k2 * P_new + (1.0 - k2) * P), (bernoulli_ll(z, y, mk), sigmoid(z))

    xs = time_major(A['X'], A['y'], A['mask'], A['newsess'] * (1.0 if carry else 0.0))
    _, (ll, p) = scan(step, (m0, P0), xs)
    return np.sum(ll), np.swapaxes(p, 0, 1)


# ---------- 2c. Q-learning with a Pavlovian bias ----------
def qpav_unpack(theta, configural):
    nF = 3 if configural else 2
    a_r, a_p, a_0 = sigmoid(theta[0]), sigmoid(theta[1]), sigmoid(theta[2])
    rho_r, rho_p = np.exp(theta[3]), np.exp(theta[4])
    pav, bias, h_pel, h_shk = theta[5], theta[6], theta[7], theta[8]
    Q0, V0 = theta[9:9 + nF], theta[9 + nF:9 + 2 * nF]
    return a_r, a_p, a_0, rho_r, rho_p, pav, bias, h_pel, h_shk, Q0, V0


def qpav_init(configural):
    lr = onp.log(0.2 / 0.8)
    v0 = [0.5, -0.5, 0.0] if configural else [0.5, -0.5]
    return onp.array([lr, lr, lr, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0] + v0 + v0)


def qpav_filter(theta, A, configural):
    F = A['F_conf'] if configural else A['F_el']
    S = A['y'].shape[0]
    a_r, a_p, a_0, rho_r, rho_p, pav, bias, h_pel, h_shk, Q0, V0 = qpav_unpack(theta, configural)

    def step(c, inp):
        Qv, Vv = c
        f, x, y, mk, pel, shk = inp
        q = np.sum(Qv * f, axis=1)
        v = np.sum(Vv * f, axis=1)
        z = q + pav * v + bias + h_pel * x[:, 2] + h_shk * x[:, 3]
        u = rho_r * pel - rho_p * shk                     # 0 when nothing was delivered
        alpha = pel * a_r + shk * a_p + (1.0 - pel - shk) * a_0
        Qv = Qv + (mk * y * alpha * (u - q))[:, None] * f   # instrumental: after a press only
        Vv = Vv + (mk * alpha * (u - v))[:, None] * f       # Pavlovian: every trial
        return (Qv, Vv), (bernoulli_ll(z, y, mk), sigmoid(z))

    xs = time_major(F, A['X'], A['y'], A['mask'], A['pellet'], A['shock'])
    init = (np.ones((S, 1)) * Q0[None, :], np.ones((S, 1)) * V0[None, :])
    _, (ll, p) = scan(step, init, xs)
    return np.sum(ll), np.swapaxes(p, 0, 1)


# ---------- 2d. Static GLM and random-effects controls ----------
def static_filter(theta, A):
    z = np.einsum('std,d->st', A['X'], theta)
    return np.sum(bernoulli_ll(z, A['y'], A['mask'])), sigmoid(z)


_OFF = -30.0     # log-SD fixed at -30: variance switched off


def expand_theta(name, theta, D):
    """Controls are the random-walk filter with drift switched off."""
    if name == 'glm_session_re':
        return 'drift_session', np.concatenate([theta, _OFF * np.ones(D)])
    if name == 'glm_rat_re':
        return 'drift_carry', np.concatenate([theta, _OFF * np.ones(D), _OFF * np.ones(1)])
    return name, theta


# ---------- registry ----------
def model_spec(name, D):
    name = base_of(name)
    specs = {
        'glm_static':        ('session', onp.zeros(D), static_filter),
        'drift_session':     ('session', drift_init(D, False), lambda th, A: drift_filter(th, A, False)),
        'drift_carry':       ('rat', drift_init(D, True), lambda th, A: drift_filter(th, A, True)),
        'drift_ar1_session': ('session', ar1_init(D, False), lambda th, A: ar1_filter(th, A, False)),
        'drift_ar1_carry':   ('rat', ar1_init(D, True), lambda th, A: ar1_filter(th, A, True)),
        'qpav_elemental':    ('rat', qpav_init(False), lambda th, A: qpav_filter(th, A, False)),
        'qpav_configural':   ('rat', qpav_init(True), lambda th, A: qpav_filter(th, A, True)),
        'glm_session_re':    ('session', drift_init(D, False)[:2 * D],
                              lambda th, A: drift_filter(expand_theta('glm_session_re', th, D)[1], A, False)),
        'glm_rat_re':        ('rat', drift_init(D, False)[:2 * D],
                              lambda th, A: drift_filter(expand_theta('glm_rat_re', th, D)[1], A, True)),
    }
    pack_by, init, ll = specs[name]
    return dict(pack_by=pack_by, init=init, ll=ll)


def n_params(name, D):
    return len(model_spec(name, D)['init'])


def describe_params(name, theta, D):
    """Human-readable parameters for the parameter tables."""
    name, theta = expand_theta(base_of(name), theta, D)
    th = onp.asarray(theta)
    rows = {}
    if name == 'glm_static':
        rows.update({f'w_{n}': th[i] for i, n in enumerate(INPUT_NAMES)})
    elif name.startswith('drift_ar1'):
        mu, s0, phi, phi_day = [onp.asarray(v) for v in ar1_unpack(th, D, name.endswith('carry'))]
        for i, n in enumerate(INPUT_NAMES):
            rows[f'mean_{n}'] = mu[i]; rows[f'sd_stationary_{n}'] = s0[i]; rows[f'phi_{n}'] = phi[i]
            rows[f'halflife_trials_{n}'] = onp.log(0.5) / onp.log(phi[i]) if 0 < phi[i] < 1 else onp.nan
        if name.endswith('carry'):
            rows['phi_between_sessions'] = float(phi_day)
    elif name.startswith('drift'):
        carry = name == 'drift_carry'
        w0, s0, sd, sday = [onp.asarray(v) for v in drift_unpack(th, D, carry)]
        for i, n in enumerate(INPUT_NAMES):
            rows[f'w0_{n}'] = w0[i]; rows[f'sd_start_{n}'] = s0[i]; rows[f'sd_drift_{n}'] = sd[i]
        if carry:
            rows['sd_between_sessions'] = float(sday)
    elif name.startswith('qpav'):
        configural = name.endswith('configural')
        a_r, a_p, a_0, rho_r, rho_p, pav, bias, h_pel, h_shk, Q0, V0 = qpav_unpack(th, configural)
        rows.update(dict(alpha_pellet=float(a_r), alpha_shock=float(a_p), alpha_nothing=float(a_0),
                         rho_pellet=float(rho_r), rho_shock=float(rho_p), pavlovian_bias=float(pav),
                         bias=float(bias), w_past_pellet=float(h_pel), w_past_shock=float(h_shk)))
        feats = ['rewDS', 'punDS', 'conflict'] if configural else ['rew_element', 'pun_element']
        rows.update({f'Q0_{f}': float(v) for f, v in zip(feats, onp.asarray(Q0))})
        rows.update({f'V0_{f}': float(v) for f, v in zip(feats, onp.asarray(V0))})
    return rows


# =========================================================
# 3. FITTING AND SCORING
# =========================================================
_COMPILED = {}


def _objective(name, D):
    """Mean negative log-likelihood per trial + gradient, and the jitted filter."""
    name = base_of(name)
    if name not in _COMPILED:
        ll = model_spec(name, D)['ll']
        f = make_value_and_grad(lambda th, A: -ll(th, A)[0] / np.sum(A['mask']))
        _COMPILED[name] = (f, jit(ll))
    return _COMPILED[name]


def fit_model(name, sessions, D, init=None, prior_mean=None, prior_sd=None, verbose=True, tag=''):
    """Maximum likelihood (or MAP, with an isotropic Gaussian prior in unconstrained space)."""
    spec = model_spec(name, D)
    A = pack(sessions, spec['pack_by'])
    n = A['mask'].sum()
    f, _ = _objective(name, D)

    def obj(th):
        v, g = f(th, A)
        v, g = float(v), onp.asarray(g, dtype=float)
        if prior_mean is not None:
            d = th - prior_mean
            v += 0.5 * onp.sum(d ** 2) / prior_sd ** 2 / n
            g = g + d / prior_sd ** 2 / n
        return v, g

    th0 = spec['init'] if init is None else onp.asarray(init, dtype=float)
    t0 = time.time()
    res = minimize(obj, th0, jac=True, method='L-BFGS-B', options=dict(maxiter=MAXITER))
    if verbose:
        flag = '' if res.success else f'  [not converged: {res.message}]'
        print(f"    fit {MODEL_LABELS[name] + tag:<46s} train obj/trial = {res.fun:.4f} "
              f"({res.nit} it, {time.time() - t0:.0f}s){flag}", flush=True)
    return res.x


def predict(name, theta, sessions, D):
    """Run the model's filter over the sessions; returns the packed arrays and p(press)."""
    spec = model_spec(name, D)
    A = pack(sessions, spec['pack_by'])
    _, llf = _objective(name, D)
    _, p = llf(theta, A)
    return A, onp.asarray(p)


def _trial_ll(p, y):
    p = onp.clip(p, 1e-12, 1 - 1e-12)
    return y * onp.log(p) + (1 - y) * onp.log(1 - p)


def score_model(name, theta, sessions, D):
    """Causal LL and one-step-ahead predictions on all trials of the given sessions."""
    A, p = predict(name, theta, sessions, D)
    mk = A['mask'] > 0
    p, y = p[mk], A['y'][mk]
    cue = onp.where(A['is_conf'][mk] > 0, 'conflict', onp.where(A['is_rew'][mk] > 0, 'reward', 'punish'))
    return float(_trial_ll(p, y).sum()), int(mk.sum()), (p >= 0.5).astype(int), y.astype(int), cue


def fit_and_score_per_rat(name, theta_pop, rat_sessions, D):
    """
    Within-rat cross-validation for a *_rat model. Sessions (in order) are split into
    PERRAT_FOLDS interleaved folds. For each fold, parameters are fitted (MAP, prior
    centred on theta_pop) to the rat's OTHER sessions only, then the model is run over
    all of the rat's sessions in order and scored on the held-out fold's trials. The
    test sessions' choices therefore never enter the fit, but their outcomes do update
    the learned values used to predict later trials, as they would for the rat.
    """
    folds = onp.arange(len(rat_sessions)) % PERRAT_FOLDS
    ll_tot, n_tot, preds, obs, cues = 0.0, 0, [], [], []
    for k in range(PERRAT_FOLDS):
        train = [s for s, fk in zip(rat_sessions, folds) if fk != k]
        if not train or k not in folds:
            continue
        th = fit_model(name, train, D, init=theta_pop, prior_mean=theta_pop,
                       prior_sd=PERRAT_PRIOR_SD, verbose=False)
        A, p = predict(name, th, rat_sessions, D)
        test = (A['mask'] > 0) & onp.isin(A['sess_pos'], onp.where(folds == k)[0])
        ll_tot += float(_trial_ll(p[test], A['y'][test]).sum())
        n_tot += int(test.sum())
        preds.append((p[test] >= 0.5).astype(int)); obs.append(A['y'][test].astype(int))
        cues.append(onp.where(A['is_conf'][test] > 0, 'conflict',
                              onp.where(A['is_rew'][test] > 0, 'reward', 'punish')))
    return ll_tot, n_tot, onp.concatenate(preds), onp.concatenate(obs), onp.concatenate(cues)


def run_cv(sessions, rats, models, D):
    per_rat, per_cue = [], []
    print(f"\n--- Leave-one-rat-out CV over {len(rats)} rats (backend: {BACKEND}) ---")
    for held in rats:
        print(f"\n=== Fold: holding out rat {held} ===")
        train = [s for s in sessions if s['Rat'] != held]
        test = [s for s in sessions if s['Rat'] == held]
        pop = {}                                            # population fits in this fold
        for name in models:
            base = base_of(name)
            if base not in pop:
                pop[base] = fit_model(base, train, D)
            if is_per_rat(name):
                ll, n, pred, obs, cue = fit_and_score_per_rat(name, pop[base], test, D)
            else:
                ll, n, pred, obs, cue = score_model(name, pop[base], test, D)
            per_rat.append({'HeldOut_Rat': held, 'Model': MODEL_LABELS[name],
                            'HeldOut_LL_per_trial': ll / n,
                            'HeldOut_Accuracy_overall': 100.0 * onp.mean(pred == obs),
                            'N_test_trials': n, 'N_params': n_params(name, D)})
            for c in ['reward', 'punish', 'conflict']:
                k = cue == c
                if k.sum():
                    per_cue.append({'HeldOut_Rat': held, 'Model': MODEL_LABELS[name], 'Cue': c,
                                    'Accuracy': 100.0 * onp.mean(pred[k] == obs[k])})
            print(f"      {MODEL_LABELS[name]:<46s} held-out LL/trial = {ll / n:.4f} | "
                  f"acc = {per_rat[-1]['HeldOut_Accuracy_overall']:.1f}%", flush=True)
    return pd.DataFrame(per_rat), pd.DataFrame(per_cue)


def fit_full(name, sessions, rats, D, cache):
    """Fit to all rats. Per-rat models return {rat: theta}, shrunk to the population fit."""
    base = base_of(name)
    if base not in cache:
        cache[base] = fit_model(base, sessions, D, tag=' (all rats)')
    if not is_per_rat(name):
        return cache[base]
    out = {}
    for r in rats:
        rs = [s for s in sessions if s['Rat'] == r]
        out[r] = fit_model(name, rs, D, init=cache[base], prior_mean=cache[base],
                           prior_sd=PERRAT_PRIOR_SD, verbose=False)
    print(f"    fit {MODEL_LABELS[name]:<46s} ({len(rats)} rats)", flush=True)
    return out


def load_hmm_cv(output_dir):
    llp = os.path.join(output_dir, 'CV_HeldOut_LL_by_State.csv')
    accp = os.path.join(output_dir, 'CV_OneStepAhead_Accuracy.csv')
    if not (os.path.exists(llp) and os.path.exists(accp)):
        sys.exit(f"Need {llp} and {accp} from 1.1b_cross_validation.py. Run that first.")
    ll = pd.read_csv(llp)
    ll = ll[ll.Num_States >= CHOSEN_STATES].copy()
    ll['Model'] = ll.Num_States.map(lambda k: f'{k}-state GLM-HMM')
    ll['N_params'] = ll.Num_States.map(lambda k: (k - 1) + k * (k - 1) + 5 * k)
    acc = pd.read_csv(accp)
    acc = acc[acc.Model.str.contains('HMM')].copy()
    acc['Model'] = HMM_LABEL
    return (ll[['HeldOut_Rat', 'Model', 'HeldOut_LL_per_trial', 'HeldOut_Accuracy_overall',
                'N_test_trials', 'N_params']],
            acc[['HeldOut_Rat', 'Model', 'Cue', 'Accuracy']])


def read_saved(path, col):
    """Read a saved results CSV, keeping only models defined in this script (results of
    models from earlier versions, e.g. v1 Q-learning or the GLM-HMM variants, are dropped)."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    keep = set(MODEL_LABELS.values()) | {HMM_LABEL, 'Observed'}
    drop = sorted(set(df[col]) - keep)
    if drop:
        print(f"  NOTE: {os.path.basename(path)}: ignoring saved results of models no longer in "
              f"this script: {drop}")
    return df[df[col].isin(keep)].reset_index(drop=True)


def merge_saved(new, path, col):
    old = read_saved(path, col)
    if old is not None:
        new = pd.concat([old[~old[col].isin(set(new[col]))], new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def ordered(labels):
    """Display order: larger HMMs, then MODEL_LABELS order, then anything else."""
    labels = list(labels)
    hmm = sorted([m for m in labels if m.endswith('GLM-HMM') and m != HMM_LABEL])
    known = [m for m in MODEL_LABELS.values() if m in labels]
    rest = [m for m in labels if m not in hmm + known + [HMM_LABEL, 'Observed']]
    return hmm + known + rest


# =========================================================
# 4. STATISTICS
# =========================================================
def _fmt_p(p):
    if p is None or onp.isnan(p):
        return "n/a"
    return "p < 0.0001" if p < 0.0001 else (f"p = {p:.4f}" if p < 0.001 else f"p = {p:.3f}")


def _paired(a, b):
    a, b = onp.asarray(a, float), onp.asarray(b, float)
    ok = ~onp.isnan(a) & ~onp.isnan(b)
    a, b = a[ok], b[ok]
    d = a - b
    n = len(d)
    t, p = stats.ttest_rel(a, b)
    try:
        wp = stats.wilcoxon(a, b).pvalue
    except ValueError:
        wp = onp.nan
    se = d.std(ddof=1) / onp.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    return dict(n=n, mean_diff=d.mean(), ci_low=d.mean() - tc * se, ci_high=d.mean() + tc * se,
                n_better=int((d > 0).sum()), t=t, df=n - 1, p=p, wilcoxon_p=wp)


def _block_hist(r):
    """Distribution of block response rates over the PPC_WINDOW+1 possible values."""
    k = onp.rint(onp.asarray(r) * PPC_WINDOW).astype(int)
    return onp.bincount(k, minlength=PPC_WINDOW + 1) / len(k)


def ppc_distance(df):
    """
    Whole-distribution posterior predictive check, per model.
      D_obs  total-variation distance between the rats' block distribution and the
             model's expected distribution (mean over simulations): the fraction of
             blocks that would have to move to make the two match.
      D_null the same distance for each simulated dataset against the mean of the
             OTHER simulations: the mismatch expected from sampling alone if the model
             were true.
      p      fraction of D_null >= D_obs (posterior predictive p-value).
    Returns a DataFrame with one row per model.
    """
    obs_h = _block_hist(df[df.Source == 'Observed'].Rate)
    lo_k, hi_k = int(onp.floor(0.25 * PPC_WINDOW)), int(onp.ceil(0.75 * PPC_WINDOW))
    ext = lambda h: h[:lo_k + 1].sum() + h[hi_k:].sum()      # blocks <=25% or >=75% pressed
    out = []
    for label in [HMM_LABEL] + ordered(df.Source.unique()):
        sub = df[df.Source == label]
        if sub.empty:
            continue
        H = onp.vstack([_block_hist(g.Rate) for _, g in sub.groupby('Sim')])
        n = len(H)
        mean_h = H.mean(0)
        d_obs = 0.5 * onp.abs(obs_h - mean_h).sum()
        loo = (H.sum(0)[None, :] - H) / (n - 1)
        d_null = 0.5 * onp.abs(H - loo).sum(1)
        ext_sims = onp.array([ext(h) for h in H])
        out.append(dict(Model=label, D_obs=d_obs, D_null_95=onp.percentile(d_null, 95),
                        D_null_lo=onp.percentile(d_null, 2.5), D_null_hi=onp.percentile(d_null, 97.5),
                        p=(1 + onp.sum(d_null >= d_obs)) / (1 + n), n_exceed=int(onp.sum(d_null >= d_obs)),
                        n_sims=n,
                        extreme_mean=ext_sims.mean(), extreme_lo=onp.percentile(ext_sims, 2.5),
                        extreme_hi=onp.percentile(ext_sims, 97.5)))
    res = pd.DataFrame(out)
    res.attrs['obs_extreme'] = ext(obs_h)
    res.attrs['n_blocks'] = int((df.Source == 'Observed').sum())
    return res


def ppc_summary(df, output_dir):
    res = ppc_distance(df)
    lines = [f"Blocks of {PPC_WINDOW} consecutive conflict trials (non-overlapping), {res.attrs['n_blocks']} observed blocks",
             "Distance between the rats' and each model's distribution of block response rates",
             "(total variation = fraction of blocks that would have to move; 'sampling only' = 95% range",
             " expected if the model were true; p = posterior predictive p-value, small = model fails)"]
    for _, r in res.iterrows():
        ptxt = (f"p <= {r.p:.3f} (no simulation of {int(r.n_sims)} as far off; minimum possible p)"
                if r.n_exceed == 0 else f"p = {r.p:.3f}")
        lines.append(f"  {r.Model:<46s} distance = {r.D_obs:.3f} (sampling only: {r.D_null_lo:.3f}-{r.D_null_hi:.3f}); {ptxt}")
    lines.append(f"\nCoarser summary: fraction of blocks <=25% or >=75% pressed (rats: {res.attrs['obs_extreme']:.3f})")
    for _, r in res.iterrows():
        lines.append(f"  {r.Model:<46s} {r.extreme_mean:.3f} (95% sim. interval {r.extreme_lo:.3f}-{r.extreme_hi:.3f})")
    res.to_csv(os.path.join(output_dir, 'AltModels_PPC_summary.csv'), index=False)

    bpath = os.path.join(output_dir, 'AltModels_PPC_behaviour.csv')
    if os.path.exists(bpath):
        b = read_saved(bpath, 'Source')
        ob = b[b.Source == 'Observed'].iloc[0]
        brow = []
        lines.append("\nBehavioural signatures (observed vs each model's simulations; two-sided p)")
        for stat, desc in BEHAV_STATS.items():
            lines.append(f"  {stat}: {desc}")
            lines.append(f"    {'Rats':<46s} {ob[stat]:+.3f}")
            for label in [HMM_LABEL] + ordered(b.Source.unique()):
                sim = b[b.Source == label][stat].dropna().values
                if len(sim) == 0:
                    continue
                n = len(sim)
                p2 = min(1.0, 2 * min((1 + onp.sum(sim >= ob[stat])) / (n + 1), (1 + onp.sum(sim <= ob[stat])) / (n + 1)))
                lo, hi = onp.percentile(sim, [2.5, 97.5])
                lines.append(f"    {label:<46s} {sim.mean():+.3f} (95% sim. interval {lo:+.3f} to {hi:+.3f}); p = {p2:.3f}")
                brow.append(dict(Model=label, Stat=stat, Observed=ob[stat], Sim_mean=sim.mean(),
                                 Sim_lo=lo, Sim_hi=hi, p_two_sided=p2, n_sims=n))
        pd.DataFrame(brow).to_csv(os.path.join(output_dir, 'AltModels_PPC_behaviour_summary.csv'), index=False)
    return "\n".join(lines)


def write_stats(per_rat, per_cue, output_dir):
    lines = []
    piv = per_rat.pivot(index='HeldOut_Rat', columns='Model', values='HeldOut_LL_per_trial')
    ntr = per_rat.pivot(index='HeldOut_Rat', columns='Model', values='N_test_trials')[HMM_LABEL]
    npar = per_rat.groupby('Model').N_params.first()
    models = ordered(m for m in per_rat.Model.unique() if m != HMM_LABEL)
    pr = lambda m: ' per rat' if 'per rat' in m else ''

    lines.append("Held-out causal log-likelihood per trial, mean +/- SEM across rats")
    lines.append("  (leave-one-rat-out; '(per rat)' models: within-rat 5-fold session CV, which lets them "
                 "use the test rat's other sessions; the HMM never sees the test rat)")
    for m in [HMM_LABEL] + models:
        lines.append(f"  {m:<46s} {piv[m].mean():.4f} +/- {piv[m].sem():.4f}   ({int(npar[m])} parameters{pr(m)})")

    lines.append(f"\n{HMM_LABEL} minus each alternative (positive = HMM predicts held-out choices better)")
    rows = []
    for m in models:
        s = _paired(piv[HMM_LABEL], piv[m])
        total = float(((piv[HMM_LABEL] - piv[m]) * ntr).sum())
        lines.append(f"  vs {m}: dLL/trial = {s['mean_diff']:+.4f} (95% CI {s['ci_low']:+.4f} to "
                     f"{s['ci_high']:+.4f}); HMM better in {s['n_better']}/{s['n']} rats; "
                     f"t({s['df']}) = {s['t']:.2f}, {_fmt_p(s['p'])}; Wilcoxon {_fmt_p(s['wilcoxon_p'])}; "
                     f"summed dLL = {total:+.1f} nats")
        rows.append({'Comparison': f'HMM vs {m}', 'Metric': 'LL/trial', 'summed_dLL_nats': total, **s})

    lines.append("\nHeld-out one-step-ahead accuracy by cue (mean +/- SEM across rats); HMM minus alternative")
    cue_names = {'reward': 'rewDS', 'punish': 'punDS', 'conflict': 'conflict'}
    for c in ['reward', 'punish', 'conflict']:
        sub = per_cue[per_cue.Cue == c].pivot(index='HeldOut_Rat', columns='Model', values='Accuracy')
        txt = [f"HMM {sub[HMM_LABEL].mean():.1f} +/- {sub[HMM_LABEL].sem():.1f}%"]
        for m in models:
            if m not in sub:
                continue
            s = _paired(sub[HMM_LABEL], sub[m])
            txt.append(f"{m} {sub[m].mean():.1f}% (diff {s['mean_diff']:+.1f} pp, {s['n_better']}/{s['n']} rats, {_fmt_p(s['p'])})")
            rows.append({'Comparison': f'HMM vs {m}', 'Metric': f'accuracy {cue_names[c]}', **s})
        lines.append(f"  {cue_names[c]}: " + "; ".join(txt))

    ppc_path = os.path.join(output_dir, 'AltModels_PPC_windows.csv')
    if os.path.exists(ppc_path):
        lines.append("\nPOSTERIOR PREDICTIVE CHECK\n" + ppc_summary(read_saved(ppc_path, 'Source'), output_dir))
    txt = "\n".join(lines)
    print("\n" + "=" * 70 + "\nALTERNATIVE MODEL COMPARISON\n" + "=" * 70 + "\n" + txt)
    with open(os.path.join(output_dir, 'AltModels_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(txt + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'AltModels_stats_table.csv'), index=False)


# =========================================================
# 5. PLOTS (supplementary; the main figure is made by 1.1g)
# =========================================================
def plot_ll(per_rat, output_dir):
    piv = per_rat.pivot(index='HeldOut_Rat', columns='Model', values='HeldOut_LL_per_trial')
    models = ordered(m for m in piv.columns if m != HMM_LABEL)
    fig, ax = plt.subplots(figsize=(1.1 * len(models) + 2, 4.5))
    for i, m in enumerate(models):
        d = (piv[HMM_LABEL] - piv[m]).values
        ax.scatter(i + onp.random.uniform(-0.12, 0.12, len(d)), d, s=14, color='#23395B', alpha=0.6, zorder=3)
        ax.errorbar(i + 0.28, d.mean(), yerr=d.std(ddof=1) / onp.sqrt(len(d)), fmt='o', color='black', capsize=3, zorder=4)
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel(f'Held-out LL per trial\n{HMM_LABEL} minus model')
    ax.set_title('HMM advantage over single-policy models (> 0 favours the HMM)', fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    for ext in ['svg', 'png']:
        plt.savefig(os.path.join(output_dir, f'SuppFig_AltModels_LL.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_accuracy(per_cue, output_dir):
    order_m = [HMM_LABEL] + ordered(m for m in per_cue.Model.unique() if m != HMM_LABEL)
    cues = ['reward', 'punish', 'conflict']
    lab = {'reward': 'rewDS', 'punish': 'punDS', 'conflict': 'Conflict'}
    cmap = plt.get_cmap('tab20')
    w = 0.85 / len(order_m)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for j, m in enumerate(order_m):
        for i, c in enumerate(cues):
            v = per_cue[(per_cue.Model == m) & (per_cue.Cue == c)].Accuracy
            ax.bar(i + (j - (len(order_m) - 1) / 2) * w, v.mean(), w, yerr=v.sem(), capsize=1.5,
                   color=cmap(j % 20), edgecolor='black', linewidth=0.5, label=m if i == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels([lab[c] for c in cues])
    ax.set_ylim(0, 105); ax.axhline(50, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel('Held-out accuracy (%)\n(one-step-ahead)')
    ax.legend(frameon=False, fontsize=7, ncol=3, loc='lower left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    for ext in ['svg', 'png']:
        plt.savefig(os.path.join(output_dir, f'SuppFig_AltModels_Accuracy.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_ppc(df, output_dir):
    labels = [HMM_LABEL] + ordered(df.Source.unique())
    labels = [l for l in labels if l in set(df.Source)]
    obs = df[df.Source == 'Observed'].Rate.values
    bins = (onp.arange(PPC_WINDOW + 2) - 0.5) / PPC_WINDOW
    ncol = min(4, len(labels)); nrow = int(onp.ceil(len(labels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.8 * ncol, 2.6 * nrow), sharey=True, squeeze=False)
    for ax, label in zip(axes.flat, labels):
        ax.hist(df[df.Source == label].Rate.values, bins=bins, density=True, color='#c9c9c9', edgecolor='white')
        ax.hist(obs, bins=bins, density=True, histtype='step', color='black', linewidth=1.5)
        ax.set_title(label, fontsize=8)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for ax in axes.flat[len(labels):]:
        ax.axis('off')
    fig.supxlabel(f'Fraction pressed per block of {PPC_WINDOW} conflict trials (black: rats; grey: model)', fontsize=9)
    plt.tight_layout()
    for ext in ['svg', 'png']:
        plt.savefig(os.path.join(output_dir, f'SuppFig_AltModels_PPC.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# =========================================================
# 6. POSTERIOR PREDICTIVE CHECK (--ppc)
# =========================================================
# Each model fitted to all rats is simulated on every rat's real cue sequence. A
# simulated press on a conflict trial gets the rat's real outcome when the rat also
# pressed, otherwise a 50/50 draw. History inputs, learned values and drifting
# weights all evolve from the SIMULATED choices.

def _outcome_for_press(s, t, rng):
    if s['cue'][t] == 'reward':
        return 1.0, 0.0
    if s['cue'][t] == 'punish':
        return 0.0, 1.0
    if s['y'][t] == 1:
        return s['pellet'][t], s['shock'][t]
    pel = float(rng.random() < 0.5)
    return pel, 1.0 - pel


def _sim_one_rat(name, th, rat_sessions, D, rng, hmm=None):
    """Simulate one rat's sessions in order; returns list of choice vectors."""
    name, th = expand_theta(name, th, D) if th is not None else (name, None)
    th = onp.asarray(th) if th is not None else None
    out = []
    if name.startswith('drift_ar1'):
        mu, s0, phi, phi_day = [onp.asarray(v) for v in ar1_unpack(th, D, name.endswith('carry'))]
        w = mu + s0 * rng.standard_normal(D)
    elif name.startswith('drift'):
        w0, s0, sd, sday = [onp.asarray(v) for v in drift_unpack(th, D, name == 'drift_carry')]
        w = w0 + s0 * rng.standard_normal(D)
    elif name.startswith('qpav'):
        configural = name.endswith('configural')
        a_r, a_p, a_0, rho_r, rho_p, pav, bias, h_pel, h_shk, Q0, V0 = [onp.asarray(v) for v in qpav_unpack(th, configural)]
        Qv, Vv = Q0.astype(float).copy(), V0.astype(float).copy()

    for j, s in enumerate(rat_sessions):
        T = len(s['y'])
        ysim, pelsim, shksim = onp.zeros(T), onp.zeros(T), onp.zeros(T)
        past_pel = past_shk = 0.0
        if j > 0:     # session boundary
            if name == 'drift_session':
                w = w0 + s0 * rng.standard_normal(D)
            elif name == 'drift_carry':
                w = w + sday * rng.standard_normal(D)
            elif name == 'drift_ar1_session':
                w = mu + s0 * rng.standard_normal(D)
            elif name == 'drift_ar1_carry':
                w = mu + phi_day * (w - mu) + onp.sqrt(1 - phi_day ** 2) * s0 * rng.standard_normal(D)
        if name == 'hmm':
            W, Atr, pi0 = hmm
            z = rng.choice(len(pi0), p=pi0)
        for t in range(T):
            x = onp.array([s['X'][t, 0], s['X'][t, 1], past_pel, past_shk, 1.0])
            if name.startswith('qpav'):
                f = (onp.array([s['cue'][t] == 'reward', s['cue'][t] == 'punish', s['cue'][t] == 'conflict'], float)
                     if configural else x[:2])
            if name == 'glm_static':
                logit = x @ th
            elif name.startswith('drift'):
                logit = x @ w
            elif name == 'hmm':
                if t > 0:
                    z = rng.choice(len(pi0), p=Atr[z])
                logit = x @ W[z]
            elif name.startswith('qpav'):
                q, v = Qv @ f, Vv @ f
                logit = q + pav * v + bias + h_pel * past_pel + h_shk * past_shk
            y = float(rng.random() < 1 / (1 + onp.exp(-logit)))
            ysim[t] = y
            pel, shk = _outcome_for_press(s, t, rng) if y else (0.0, 0.0)
            pelsim[t], shksim[t] = pel, shk
            if name.startswith('qpav'):
                u = rho_r * pel - rho_p * shk
                alpha = a_r if pel else (a_p if shk else a_0)
                if y:
                    Qv = Qv + alpha * (u - q) * f
                Vv = Vv + alpha * (u - v) * f
            if name.startswith('drift_ar1'):
                w = mu + phi * (w - mu) + onp.sqrt(1 - phi ** 2) * s0 * rng.standard_normal(D)
            elif name.startswith('drift'):
                w = w + sd * rng.standard_normal(D)
            past_pel, past_shk = pel, shk
        out.append((ysim, pelsim, shksim))
    return out


def simulate(name, theta, sessions, D, rng, hmm=None):
    """Simulated (choices, pellets, shocks), one tuple per session (same order as
    `sessions`). `theta` may be a {rat: theta} dict for per-rat models."""
    by_rat = {}
    for idx, s in enumerate(sessions):
        by_rat.setdefault(s['Rat'], []).append(idx)
    sims = [None] * len(sessions)
    for rat, idxs in by_rat.items():
        th = theta[rat] if isinstance(theta, dict) else theta
        for idx, sim in zip(idxs, _sim_one_rat(base_of(name), th, [sessions[i] for i in idxs], D, rng, hmm)):
            sims[idx] = sim
    return sims


def conflict_windows(sessions, choices, k=PPC_WINDOW):
    rates = []
    for s, y in zip(sessions, choices):
        c = onp.asarray(y)[s['cue'] == 'conflict']
        for i in range(0, len(c) - k + 1, k):
            rates.append(c[i:i + k].mean())
    return onp.array(rates)


BEHAV_BLOCK = 24      # trials per block for the cross-trial-type coupling statistic

BEHAV_STATS = {  # name: description used in the stats file
    'p_after_pellet': 'P(press on next conflict trial | pressed last conflict trial, got pellet)',
    'p_after_shock':  'P(press on next conflict trial | pressed last conflict trial, got shock)',
    'outcome_effect': 'pellet minus shock: effect of the last conflict outcome on the next conflict choice',
    'outcome_effect_conf_lag2': 'same, on the 2nd conflict trial after the outcome (persistence)',
    'outcome_effect_conf_lag3': 'same, on the 3rd conflict trial after the outcome (persistence)',
    'outcome_effect_next_rewDS': 'pellet minus shock: effect of the last conflict outcome on the next rewDS choice',
    'outcome_effect_next_punDS': 'pellet minus shock: effect of the last conflict outcome on the next punDS choice',
    'r_conflict_punDS': f'within-session correlation of conflict and punDS responding across {BEHAV_BLOCK}-trial blocks',
    'r_conflict_rewDS': f'within-session correlation of conflict and rewDS responding across {BEHAV_BLOCK}-trial blocks',
}


def behaviour_stats(sessions, ys, pels, shks):
    """
    Two signatures that separate outcome-driven learning from outcome-independent states.
    1. Outcome effect. Pellet vs shock after a conflict press is randomised by the task,
       so the difference in responding on the NEXT conflict trial (whatever lies between)
       is a causal estimate of how much a single outcome changes the conflict policy.
       Learning models predict a large effect; states with outcome-independent
       transitions predict little beyond the one-trial history weights.
    2. Coupling across trial types. Within a session, do conflict and punDS (rewDS)
       responding rise and fall together across blocks? A state that changes the
       weights of all cues predicts positive coupling; a configural learning model,
       whose conflict value is learned separately, predicts little.
    """
    after_pel, after_shk, blocks = [], [], []
    keys = ['conf_lag2', 'conf_lag3', 'next_rewDS', 'next_punDS']
    eff = {k: ([], []) for k in keys}                    # (after pellet, after shock)
    for s, y, pe, sh in zip(sessions, ys, pels, shks):
        y = onp.asarray(y)
        conf = onp.where(s['cue'] == 'conflict')[0]
        rew = onp.where(s['cue'] == 'reward')[0]
        pun = onp.where(s['cue'] == 'punish')[0]
        for i, a in enumerate(conf[:-1]):
            if y[a] != 1:
                continue
            g = 0 if pe[a] == 1 else 1
            (after_pel if g == 0 else after_shk).append(y[conf[i + 1]])
            for lag, key in [(2, 'conf_lag2'), (3, 'conf_lag3')]:
                if i + lag < len(conf):
                    eff[key][g].append(y[conf[i + lag]])
            for idx, key in [(rew, 'next_rewDS'), (pun, 'next_punDS')]:
                nxt = idx[idx > a]
                if len(nxt):
                    eff[key][g].append(y[nxt[0]])
        rows = []
        for i in range(0, len(y) - BEHAV_BLOCK + 1, BEHAV_BLOCK):
            cue, yy = s['cue'][i:i + BEHAV_BLOCK], y[i:i + BEHAV_BLOCK]
            if all((cue == c).sum() >= 3 for c in ['conflict', 'punish', 'reward']):
                rows.append([yy[cue == c].mean() for c in ['conflict', 'punish', 'reward']])
        if len(rows) >= 2:
            rows = onp.array(rows)
            blocks.append(rows - rows.mean(0))           # within-session deviations
    B = onp.vstack(blocks)
    corr = lambda a, b: onp.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else onp.nan
    pp, ps = onp.mean(after_pel), onp.mean(after_shk)
    out = dict(p_after_pellet=pp, p_after_shock=ps, outcome_effect=pp - ps,
               r_conflict_punDS=corr(B[:, 0], B[:, 1]), r_conflict_rewDS=corr(B[:, 0], B[:, 2]))
    for k in keys:
        a, b = eff[k]
        out['outcome_effect_' + k] = (onp.mean(a) - onp.mean(b)) if len(a) and len(b) else onp.nan
    return out


def run_ppc(sessions, rats, models, D, output_dir, cache):
    print("\n--- Posterior predictive check ---")
    rng = onp.random.default_rng(14)
    obs = conflict_windows(sessions, [s['y'] for s in sessions])
    rows = [pd.DataFrame({'Source': 'Observed', 'Sim': 0, 'Rate': obs})]
    brows = [{'Source': 'Observed', 'Sim': 0, **behaviour_stats(
        sessions, [s['y'] for s in sessions], [s['pellet'] for s in sessions], [s['shock'] for s in sessions])}]
    jobs = []
    wp = os.path.join(output_dir, f'{CHOSEN_STATES}_State_GLM_Weights.csv')
    tp = os.path.join(output_dir, f'{CHOSEN_STATES}_State_Transitions.csv')
    if os.path.exists(wp) and os.path.exists(tp):
        W = pd.read_csv(wp, index_col=0).values            # press-oriented (sign flipped in 1.1)
        Atr = pd.read_csv(tp, index_col=0).values
        Atr = Atr / Atr.sum(1, keepdims=True)
        evals, evecs = onp.linalg.eig(Atr.T)
        pi0 = onp.real(evecs[:, onp.argmin(onp.abs(evals - 1))]); pi0 = pi0 / pi0.sum()
        jobs.append(('hmm', HMM_LABEL, None, (W, Atr, pi0)))
    else:
        print(f"  (HMM weights/transitions from 1.1 not found in {output_dir}; HMM not simulated)")
    for name in models:
        jobs.append((name, MODEL_LABELS[name], fit_full(name, sessions, rats, D, cache), None))
    for name, label, theta, hmm in jobs:
        t0 = time.time()
        for i in range(PPC_N_SIMS):
            sims = simulate(name, theta, sessions, D, rng, hmm)
            ys, pels, shks = [x[0] for x in sims], [x[1] for x in sims], [x[2] for x in sims]
            rows.append(pd.DataFrame({'Source': label, 'Sim': i, 'Rate': conflict_windows(sessions, ys)}))
            brows.append({'Source': label, 'Sim': i, **behaviour_stats(sessions, ys, pels, shks)})
        print(f"  simulated {label} ({time.time() - t0:.0f}s)", flush=True)
    new = pd.concat(rows, ignore_index=True)
    path = os.path.join(output_dir, 'AltModels_PPC_windows.csv')
    old = read_saved(path, 'Source')
    if old is not None:                         # keep earlier simulations of models not re-run
        new = pd.concat([old[~old.Source.isin(set(new.Source))], new], ignore_index=True)
    new.to_csv(path, index=False)
    bnew = pd.DataFrame(brows)
    bpath = os.path.join(output_dir, 'AltModels_PPC_behaviour.csv')
    bold = read_saved(bpath, 'Source')
    if bold is not None:
        bnew = pd.concat([bold[~bold.Source.isin(set(bnew.Source))], bnew], ignore_index=True)
    bnew.to_csv(bpath, index=False)
    return new


# =========================================================
# 7. EXECUTION BLOCK
# =========================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare the GLM-HMM with single-policy models.")
    ap.add_argument('--models', nargs='+', default=DEFAULT_MODELS, choices=list(MODEL_LABELS))
    ap.add_argument('--stats-only', action='store_true', help="reuse saved CV CSVs; redo stats and plots")
    ap.add_argument('--ppc', action='store_true', help="(re)run the posterior predictive check for --models")
    ap.add_argument('--input', default=INPUT_PATH)
    ap.add_argument('--outdir', default=OUTPUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    pr_path = os.path.join(args.outdir, 'AltModels_CV_per_rat.csv')
    pc_path = os.path.join(args.outdir, 'AltModels_CV_per_cue.csv')

    hmm_rat, hmm_cue = load_hmm_cv(args.outdir)
    if (not args.stats_only) or args.ppc:
        sessions, rats = prepare_trials(pd.read_csv(args.input, dtype={'Session': str}))
        D = sessions[0]['X'].shape[1]
    cache = {}                                   # full-data population fits

    if args.stats_only:
        alt_rat, alt_cue = read_saved(pr_path, 'Model'), read_saved(pc_path, 'Model')
    else:
        alt_rat, alt_cue = run_cv(sessions, rats, args.models, D)
        alt_rat = merge_saved(alt_rat, pr_path, 'Model')
        alt_cue = merge_saved(alt_cue, pc_path, 'Model')
        print("\n--- Full-data fits (parameters for reporting) ---")
        prow, rrow = [], []
        for name in args.models:
            th = fit_full(name, sessions, rats, D, cache)
            if isinstance(th, dict):
                per = pd.DataFrame([{'Rat': r, 'Model': MODEL_LABELS[name], **describe_params(name, t, D)}
                                    for r, t in th.items()])
                rrow.append(per)
                prow.append({'Model': MODEL_LABELS[name] + ' [median over rats]',
                             **per.drop(columns=['Rat', 'Model']).median().to_dict()})
            else:
                prow.append({'Model': MODEL_LABELS[name], **describe_params(name, th, D)})
        merge_saved(pd.DataFrame(prow), os.path.join(args.outdir, 'AltModels_fullfit_params.csv'), 'Model')
        if rrow:
            merge_saved(pd.concat(rrow, ignore_index=True),
                        os.path.join(args.outdir, 'AltModels_perrat_params.csv'), 'Model')

    if args.ppc:
        run_ppc(sessions, rats, args.models, D, args.outdir, cache)

    per_rat = pd.concat([hmm_rat, alt_rat], ignore_index=True)
    per_cue = pd.concat([hmm_cue, alt_cue], ignore_index=True)
    write_stats(per_rat, per_cue, args.outdir)
    plot_ll(per_rat, args.outdir)
    plot_accuracy(per_cue, args.outdir)
    ppc_path = os.path.join(args.outdir, 'AltModels_PPC_windows.csv')
    if os.path.exists(ppc_path):
        plot_ppc(read_saved(ppc_path, 'Source'), args.outdir)
    print(f"\nDone. Outputs written to '{args.outdir}/'.")