"""
5.4_alcohol_frozen_model_checks.py
----------------------------------
Additional analysis (not reported in the paper): does alcohol shift state OCCUPANCY,
or does it change the policy WITHIN a state?

The alcohol sessions were decoded with the Phase 3 model held fixed. If alcohol
weakened, say, the punDS weight inside the Risk-averse state, a frozen decoder would
re-assign those trials to other states, and the result would look exactly like
"loss of the Risk-averse state". This script runs three checks.

  CHECK 1  State-conditioned response profiles by dose
           Per rat x dose x state x cue: observed response rate on assigned trials,
           and the rate the frozen model predicts for those same trials. If the
           policy inside a state changed, observed and model-implied rates diverge
           under alcohol (e.g. Risk-averse punDS/conflict responding above prediction).
           Done with smoothed assignment (as in Fig. 5B) and with one-step-ahead
           predictive assignment (see 3.4).

  CHECK 2  Fit of the frozen model by dose
           Per rat x dose: one-step-ahead log-likelihood per trial and accuracy of the
           frozen model. A selective drop under alcohol suggests the Phase 3 policies no
           longer describe behaviour. (Raw LL also falls if behaviour is simply noisier,
           so CHECK 3 is the decisive comparison.)

  CHECK 3  Refitting with dose-specific parameters (all warm-started from the frozen model)
           R  shared refit       one 3-state model for all alcohol sessions
           D  dose-specific      a separate 3-state model per dose (weights AND transitions)
           X  dose as input      one 3-state model with extra inputs:
                                 Low, High (state-specific bias shifts) and
                                 Low x punDS, High x punDS (state-specific change in punDS weight)
           Compared by leave-one-rat-out cross-validated LL per trial, plus AIC/BIC on the
           full data. If D and X do not beat R out of sample, and X's dose terms inside the
           Risk-averse state are near zero, the occupancy account is sufficient. Also reports
           occupancy decoded with each dose's OWN refit model (D): if Risk-averse occupancy
           still falls under alcohol when weights are free, the result does not depend on
           freezing the model. Optional rat-level bootstrap of the D weights.

  EXTRA    Full-session occupancy and response rates (for the supplement), plus the
           first-15-min and post-15-min epochs, using the same decoding as 5.1.

INPUT  Data/collated_2026_conflict_alcohol_with_Probabilities_065.csv (from 5.1; the raw
       table from 5.0 also works, without the cross-check against Prob_State_*).
OUTPUTS (HMM_Outputs/Alcohol_checks/): Prism-ready CSVs, weight / transition tables,
CV results and summary_printout.txt. See the INTERPRETATION block at the end of the printout.

Usage:  python 5.4_alcohol_frozen_model_checks.py [--skip-cv] [--n-boot N]
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

try:
    import ssm
    SSM_IMPORT_ERROR = None
except ImportError as e:
    ssm = None
    SSM_IMPORT_ERROR = e
SSM_HELP = ("If the error mentions NumPy 1.x / 2.x or _ARRAY_API, ssm was compiled against a different NumPy "
            "than the one installed: rebuild it (README, Installation > Troubleshooting).")

# =========================================================
# 0. CONFIGURATION
# =========================================================
ALC_CSV    = os.path.join("Data", "collated_2026_conflict_alcohol_with_Probabilities_065.csv")
MODEL_PKL  = "final_3_state_model_v5.pkl"
OUTPUT_DIR = os.path.join("HMM_Outputs", "Alcohol_checks")

DOSE_COL   = "Session_Type"                                  # column holding the test condition (from 5.0)
DOSE_MAP   = {"Saline": "Saline",                            # raw value -> label (order = Saline, Low, High)
              "Alcohol_Low": "Low",
              "Alcohol_High": "High"}
TIME_COL   = "Timestamp"                                     # trial start, seconds from session start
SESSION_COL = "Session_ID"                                   # identifies a session uniquely within rat

EPOCH_CUTOFF_S = 900          # first 15 min
FIT_EPOCH      = "post15"     # trials used for CHECK 2/3 fitting and scoring: "post15" or "full"
                              # (history inputs are always computed on the full session first)
CONFIDENCE_THRESHOLD = 0.65
MIN_TRIALS     = 5            # min trials per rat x dose x state x cue cell in CHECK 1
NUM_ITERS      = 200
TOLERANCE      = 1e-4
N_BOOT         = 100          # rat-level bootstrap of the D weights (0 to skip; ~1-3 s per fit)
SEED           = 14

STATE_NAMES = {1: "Risk-taking", 2: "Risk-averse", 3: "Repeating"}
DOSES       = ["Saline", "Low", "High"]
CUE_LABEL   = {"reward": "rewDS", "punish": "punDS", "conflict": "Conf"}
BASE_INPUTS = ["rew_cue_val", "pun_cue_val", "past_pellet", "past_shock", "bias"]
BASE_NAMES  = ["rewDS", "punDS", "past_pellet", "past_shock", "bias"]
X_INPUTS    = BASE_INPUTS + ["dose_low", "dose_high", "low_x_pun", "high_x_pun"]
X_NAMES     = BASE_NAMES + ["Low", "High", "Low x punDS", "High x punDS"]


# =========================================================
# 1. DATA
# =========================================================
def load_alcohol(path):
    df = pd.read_csv(path, dtype={SESSION_COL: str})
    print(f"Loaded {path}: {len(df)} rows. Columns:\n  {list(df.columns)}\n")
    need = ["Rat", SESSION_COL, "Trial", "Cue", "Response", "Outcome", DOSE_COL, TIME_COL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"STOP: column(s) {missing} not found. Edit DOSE_COL / TIME_COL / SESSION_COL in CONFIG.")
    vals = sorted(df[DOSE_COL].dropna().astype(str).unique())
    unmapped = [v for v in vals if v not in DOSE_MAP]
    if unmapped:
        sys.exit(f"STOP: {DOSE_COL} values {unmapped} are not in DOSE_MAP. Values found: {vals}")
    df["Dose_label"] = df[DOSE_COL].astype(str).map(DOSE_MAP)
    df = df.sort_values(["Rat", SESSION_COL, "Trial"]).reset_index(drop=True)

    g = [df["Rat"], df[SESSION_COL]]
    df["past_pellet"] = (df["Response"] == "Response-Pellet").astype(int).groupby(g).shift(1).fillna(0)
    df["past_shock"] = (df["Response"] == "Response-Shock").astype(int).groupby(g).shift(1).fillna(0)
    df["rew_cue_val"] = ((df["Cue"] == "reward") | (df["Cue"] == "conflict")).astype(float)
    df["pun_cue_val"] = ((df["Cue"] == "punish") | (df["Cue"] == "conflict")).astype(float)
    df["bias"] = 1.0
    df["Lever_Press"] = df["Outcome"].astype(int)
    df["dose_low"] = (df["Dose_label"] == "Low").astype(float)
    df["dose_high"] = (df["Dose_label"] == "High").astype(float)
    df["low_x_pun"] = df["dose_low"] * df["pun_cue_val"]
    df["high_x_pun"] = df["dose_high"] * df["pun_cue_val"]
    df["Epoch"] = np.where(df[TIME_COL] <= EPOCH_CUTOFF_S, "first15", "post15")

    n_dose = df.groupby("Rat")["Dose_label"].nunique()
    print(f"{df['Rat'].nunique()} rats; sessions per dose:\n"
          f"{df.groupby(['Dose_label'])[SESSION_COL].apply(lambda s: s.astype(str).nunique()).to_string()}")
    if (n_dose < 3).any():
        print(f"  NOTE: rats without all three doses: {list(n_dose[n_dose < 3].index)}")
    return df


def sequences(df, cols, epoch=None, dose=None, rats=None):
    """List of (Y, X, index) per session, optionally restricted to an epoch / dose / rats."""
    out = []
    for (rat, sess), sd in df.groupby(["Rat", SESSION_COL], sort=False):
        if dose is not None and sd["Dose_label"].iloc[0] != dose:
            continue
        if rats is not None and rat not in rats:
            continue
        if epoch is not None and epoch != "full":
            sd = sd[sd["Epoch"] == epoch]
        if len(sd) < 5:
            continue
        out.append((sd["Lever_Press"].values.reshape(-1, 1).astype(int),
                    sd[cols].astype(float).values, sd.index))
    return out


# =========================================================
# 2. HMM HELPERS
# =========================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def press_weights(hmm):
    raw = hmm.observations.Wk
    return np.atleast_2d(raw[:, 0, :] * -1 if raw.ndim == 3 else raw.squeeze() * -1)


def init_distn(hmm, K):
    try:
        p = np.asarray(hmm.init_state_distn.initial_state_distn, float)
    except Exception:
        p = np.ones(K)
    p = np.clip(p, 1e-300, None)
    return p / p.sum()


def params_of(hmm):
    W = press_weights(hmm)
    A = np.atleast_2d(hmm.transitions.transition_matrix)
    return W, A, init_distn(hmm, A.shape[0])


def forward_backward(emis, A, pi0):
    T, K = emis.shape
    pred = np.zeros((T, K)); filt = np.zeros((T, K)); c = np.zeros(T)
    for t in range(T):
        pred[t] = pi0 if t == 0 else filt[t - 1] @ A
        u = pred[t] * emis[t]
        c[t] = max(u.sum(), 1e-300)
        filt[t] = u / c[t]
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (emis[t + 1] * beta[t + 1]) / c[t + 1]
    sm = filt * beta
    sm /= sm.sum(axis=1, keepdims=True)
    return dict(pred=pred, smooth=sm, logc=np.log(c))


def run_model(W, A, pi0, Y, X):
    y = Y.ravel()
    pk = sigmoid(X @ W.T)
    emis = np.where(y[:, None] == 1, pk, 1 - pk)
    fb = forward_backward(emis, A, pi0)
    fb["pk"] = pk
    fb["pred_ppress"] = np.sum(fb["pred"] * pk, axis=1)
    return fb


def score(W, A, pi0, seqs):
    """Total LL, n trials, n correct (one-step-ahead) over a list of sequences."""
    ll = n = corr = 0.0
    for Y, X, _ in seqs:
        fb = run_model(W, A, pi0, Y, X)
        ll += fb["logc"].sum(); n += len(Y)
        corr += np.sum((fb["pred_ppress"] >= 0.5).astype(int) == Y.ravel())
    return ll, n, corr


def fit_warm(seqs, W_init, A_init, pi_init, M):
    """EM fit warm-started at the given parameters (initialize=False keeps them)."""
    K = W_init.shape[0]
    h = ssm.HMM(K, 1, M, observations="input_driven_obs",
                observation_kwargs=dict(C=2), transitions="standard")
    h.observations.Wk = (-W_init)[:, None, :].copy()
    h.transitions.log_Ps = np.log(np.clip(A_init, 1e-12, None))
    h.init_state_distn.log_pi0 = np.log(np.clip(pi_init, 1e-12, None))
    datas = [s[0] for s in seqs]; inputs = [s[1] for s in seqs]
    h.fit(datas, inputs=inputs, method="em", num_iters=NUM_ITERS,
          tolerance=TOLERANCE, initialize=False, verbose=0)
    W, A, pi = params_of(h)
    # guard against label switching: match states to the frozen model by weight cosine
    perm = match_states(W[:, :5], W_init[:, :5])
    return W[perm], A[np.ix_(perm, perm)], pi[perm]


def match_states(W_fit, W_ref):
    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    C = np.array([[cos(W_fit[i], W_ref[j]) for j in range(len(W_ref))] for i in range(len(W_fit))])
    r, c = linear_sum_assignment(-C)          # r: fit rows, c: ref cols
    perm = np.empty(len(W_ref), int)
    perm[c] = r                                # perm[ref_state] = fitted state index
    return perm


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def n_params(K, M):
    return (K - 1) + K * (K - 1) + K * M


def pad(W5, M):
    W = np.zeros((W5.shape[0], M)); W[:, :5] = W5
    return W


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8"); self.o = sys.stdout
    def write(self, s):
        self.o.write(s); self.f.write(s)
    def flush(self):
        self.o.flush(); self.f.flush()


def paired(a, b, label):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        print(f"  {label}: too few rats"); return
    d = a - b
    t, p = stats.ttest_rel(a, b)
    try:
        w_p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        w_p = np.nan
    print(f"  {label}: mean diff = {d.mean():+.4f} (SEM {d.std(ddof=1)/np.sqrt(len(d)):.4f}), "
          f"higher in {int((d > 0).sum())}/{len(d)} rats, t({len(d)-1}) = {t:.2f}, p = {p:.4f}; "
          f"Wilcoxon p = {w_p:.4f}")


# =========================================================
# 3. MAIN
# =========================================================
def main(skip_cv=False, n_boot=N_BOOT):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sys.stdout = Tee(os.path.join(OUTPUT_DIR, "summary_printout.txt"))
    rng = np.random.default_rng(SEED)

    df = load_alcohol(ALC_CSV)
    with open(MODEL_PKL, "rb") as f:
        frozen = pickle.load(f)
    W0, A0, pi0 = params_of(frozen)
    K = A0.shape[0]
    print(f"\nFrozen model: K = {K}")
    print(pd.DataFrame(W0, index=[STATE_NAMES[k + 1] for k in range(K)], columns=BASE_NAMES).round(2))

    # ---------------- decode whole sessions with the frozen model (as in 5.1) ----------------
    for k in range(K):
        df[f"sm_P{k+1}"] = np.nan; df[f"pr_P{k+1}"] = np.nan; df[f"pPress_S{k+1}"] = np.nan
    df["logc_frozen"] = np.nan; df["pred_ppress_frozen"] = np.nan
    for Y, X, idx in sequences(df, BASE_INPUTS, epoch="full"):
        fb = run_model(W0, A0, pi0, Y, X)
        for k in range(K):
            df.loc[idx, f"sm_P{k+1}"] = fb["smooth"][:, k]
            df.loc[idx, f"pr_P{k+1}"] = fb["pred"][:, k]
            df.loc[idx, f"pPress_S{k+1}"] = fb["pk"][:, k]
        df.loc[idx, "pred_ppress_frozen"] = fb["pred_ppress"]
    for tag in ["sm", "pr"]:
        P = df[[f"{tag}_P{k+1}" for k in range(K)]].values
        df[f"{tag}_state"] = np.where(np.nanmax(P, axis=1) >= CONFIDENCE_THRESHOLD,
                                      np.nanargmax(np.nan_to_num(P, nan=-1), axis=1) + 1, np.nan)
    if "Prob_State_1" in df.columns:
        d = np.nanmax(np.abs(df[[f"Prob_State_{k+1}" for k in range(K)]].values -
                             df[[f"sm_P{k+1}" for k in range(K)]].values))
        print(f"\nCheck vs existing Prob_State columns (5.1 output): max abs diff = {d:.2e}")
    un = df["sm_state"].isna().mean() * 100
    print(f"Unassigned (smoothed, >= {CONFIDENCE_THRESHOLD}): {un:.1f}% of trials")

    # ---------------- EXTRA: occupancy + response rates by epoch ----------------
    print("\n" + "=" * 70 + "\nEXTRA  Occupancy and response rates by epoch (frozen decoding)\n" + "=" * 70)
    for epoch in ["full", "first15", "post15"]:
        de = df if epoch == "full" else df[df["Epoch"] == epoch]
        occ = (de.dropna(subset=["sm_state"])
                 .groupby(["Rat", "Dose_label"])["sm_state"]
                 .value_counts(normalize=True).mul(100).unstack(fill_value=0))
        occ = occ.reindex(columns=range(1, K + 1), fill_value=0)
        occ.columns = [STATE_NAMES[c] for c in occ.columns]
        occ_w = occ.unstack("Dose_label").reindex(
            columns=pd.MultiIndex.from_product([[STATE_NAMES[k] for k in range(1, K + 1)], DOSES]))
        occ_w.columns = [f"{s} | {d}" for s, d in occ_w.columns]
        occ_w.to_csv(os.path.join(OUTPUT_DIR, f"Prism_occupancy_{epoch}.csv"))
        rr = de.groupby(["Rat", "Dose_label", "Cue"])["Lever_Press"].mean().mul(100).unstack(["Cue", "Dose_label"])
        rr = rr.reindex(columns=pd.MultiIndex.from_product([["reward", "punish", "conflict"], DOSES]))
        rr.columns = [f"{CUE_LABEL[c]} | {d}" for c, d in rr.columns]
        rr.to_csv(os.path.join(OUTPUT_DIR, f"Prism_response_rates_{epoch}.csv"))
        print(f"\n[{epoch}] occupancy, % of assigned trials (mean across rats):")
        print(occ.groupby("Dose_label").mean().reindex(DOSES).round(1).to_string())
        print(f"[{epoch}] response rate, %:")
        print(de.groupby(["Dose_label", "Cue"])["Lever_Press"].mean().mul(100)
                .unstack().reindex(DOSES)[["reward", "punish", "conflict"]].round(1).to_string())

    # ---------------- CHECK 1: state-conditioned profiles by dose ----------------
    print("\n" + "=" * 70 + "\nCHECK 1  State-conditioned response profiles by dose (post-15 min)\n" + "=" * 70)
    dp = df[df["Epoch"] == "post15"]
    rows = []
    for tag, label in [("sm", "smoothed"), ("pr", "predictive")]:
        for (rat, dose, k, cue), g in dp.dropna(subset=[f"{tag}_state"]).groupby(
                ["Rat", "Dose_label", f"{tag}_state", "Cue"]):
            k = int(k)
            rows.append(dict(Assignment=label, Rat=rat, Dose=dose, State=STATE_NAMES[k],
                             Cue=CUE_LABEL[cue], n_trials=len(g),
                             Observed=g["Lever_Press"].mean() * 100,
                             Model_implied=g[f"pPress_S{k}"].mean() * 100))
    prof = pd.DataFrame(rows)
    prof["Obs_minus_model"] = prof["Observed"] - prof["Model_implied"]
    prof.to_csv(os.path.join(OUTPUT_DIR, "check1_state_profiles_by_dose_long.csv"), index=False)
    ok = prof[prof["n_trials"] >= MIN_TRIALS]
    for label in ["smoothed", "predictive"]:
        s = ok[ok["Assignment"] == label]
        print(f"\n[{label} assignment] observed % (model-implied %) [n rats with >= {MIN_TRIALS} trials]")
        tab = s.groupby(["State", "Cue", "Dose"]).agg(obs=("Observed", "mean"), mdl=("Model_implied", "mean"),
                                                      n=("Rat", "nunique"))
        tab["txt"] = tab.apply(lambda r: f"{r.obs:5.1f} ({r.mdl:5.1f}) [{int(r.n)}]", axis=1)
        print(tab["txt"].unstack("Dose").reindex(columns=DOSES).to_string())
        for st in [STATE_NAMES[k] for k in range(1, K + 1)]:
            w = s[s["State"] == st].pivot_table(index="Rat", columns=["Cue", "Dose"], values="Observed")
            w = w.reindex(columns=pd.MultiIndex.from_product([["rewDS", "punDS", "Conf"], DOSES]))
            w.columns = [f"{c} | {d}" for c, d in w.columns]
            w.to_csv(os.path.join(OUTPUT_DIR, f"Prism_check1_{label}_{st}.csv"))
        # quick test: residual (observed - model) in each state, saline vs high
        print(f"\n  Residual (observed - model-implied), High vs Saline, paired across rats [{label}]:")
        for st in [STATE_NAMES[k] for k in range(1, K + 1)]:
            for cue in ["rewDS", "punDS", "Conf"]:
                r = s[(s.State == st) & (s.Cue == cue)].pivot_table(index="Rat", columns="Dose",
                                                                    values="Obs_minus_model")
                if {"Saline", "High"} <= set(r.columns):
                    paired(r["High"], r["Saline"], f"{st:11s} {cue:5s}")

    # ---------------- CHECK 2: frozen-model fit by dose ----------------
    print("\n" + "=" * 70 + f"\nCHECK 2  Frozen-model one-step-ahead fit by dose ({FIT_EPOCH})\n" + "=" * 70)
    fit_rows = []
    for rat in sorted(df["Rat"].unique()):
        for dose in DOSES:
            seqs = sequences(df, BASE_INPUTS, epoch=FIT_EPOCH, dose=dose, rats={rat})
            if not seqs:
                continue
            ll, n, corr = score(W0, A0, pi0, seqs)
            ent = df.loc[np.concatenate([s[2] for s in seqs]), "Lever_Press"].mean()
            fit_rows.append(dict(Rat=rat, Dose=dose, n_trials=int(n), LL_per_trial=ll / n,
                                 Accuracy=corr / n * 100, Press_rate=ent * 100))
    fit = pd.DataFrame(fit_rows)
    fit.to_csv(os.path.join(OUTPUT_DIR, "check2_frozen_fit_by_dose.csv"), index=False)
    for v in ["LL_per_trial", "Accuracy"]:
        w = fit.pivot(index="Rat", columns="Dose", values=v).reindex(columns=DOSES)
        w.to_csv(os.path.join(OUTPUT_DIR, f"Prism_check2_{v}.csv"))
        print(f"\n{v}: " + ", ".join(f"{d} {w[d].mean():.3f} ({w[d].sem():.3f})" for d in DOSES))
        for d in ["Low", "High"]:
            paired(w[d], w["Saline"], f"{d} vs Saline")
    print("  (Lower LL under alcohol can mean either a poorer fit or noisier behaviour; see CHECK 3.)")

    if ssm is None:
        print(f"\nssm could not be imported ({SSM_IMPORT_ERROR}): skipping CHECK 3.\n{SSM_HELP}")
        return

    # ---------------- CHECK 3: refits ----------------
    print("\n" + "=" * 70 + f"\nCHECK 3  Refits warm-started from the frozen model ({FIT_EPOCH})\n" + "=" * 70)
    all_b = sequences(df, BASE_INPUTS, epoch=FIT_EPOCH)
    all_x = sequences(df, X_INPUTS, epoch=FIT_EPOCH)
    n_tot = sum(len(s[0]) for s in all_b)

    ll_F = score(W0, A0, pi0, all_b)[0]
    WR, AR, piR = fit_warm(all_b, W0, A0, pi0, 5)
    ll_R = score(WR, AR, piR, all_b)[0]
    WX, AX, piX = fit_warm(all_x, pad(W0, 9), A0, pi0, 9)
    ll_X = score(WX, AX, piX, all_x)[0]
    D = {}
    ll_D = 0.0
    for dose in DOSES:
        sb = sequences(df, BASE_INPUTS, epoch=FIT_EPOCH, dose=dose)
        D[dose] = fit_warm(sb, W0, A0, pi0, 5)
        ll_D += score(*D[dose], sb)[0]
    k5, k9 = n_params(K, 5), n_params(K, 9)
    ic = pd.DataFrame([
        dict(Model="F frozen (0 free params)", LL=ll_F, k=0),
        dict(Model="R shared refit", LL=ll_R, k=k5),
        dict(Model="X dose as input", LL=ll_X, k=k9),
        dict(Model="D dose-specific", LL=ll_D, k=3 * k5)])
    ic["AIC"] = 2 * ic["k"] - 2 * ic["LL"]
    ic["BIC"] = ic["k"] * np.log(n_tot) - 2 * ic["LL"]
    ic["LL_per_trial"] = ic["LL"] / n_tot
    print(f"\nIn-sample fit ({n_tot} trials):")
    print(ic.round(2).to_string(index=False))
    ic.to_csv(os.path.join(OUTPUT_DIR, "check3_information_criteria.csv"), index=False)

    # weights
    wt = []
    for k in range(K):
        wt.append(dict(Model="Frozen", State=STATE_NAMES[k + 1], **dict(zip(BASE_NAMES, W0[k]))))
        wt.append(dict(Model="R shared", State=STATE_NAMES[k + 1], **dict(zip(BASE_NAMES, WR[k]))))
        for dose in DOSES:
            wt.append(dict(Model=f"D {dose}", State=STATE_NAMES[k + 1],
                           **dict(zip(BASE_NAMES, D[dose][0][k])),
                           cos_to_frozen=cosine(D[dose][0][k], W0[k])))
        wt.append(dict(Model="X dose-input", State=STATE_NAMES[k + 1], **dict(zip(X_NAMES, WX[k]))))
    wt = pd.DataFrame(wt)
    wt.to_csv(os.path.join(OUTPUT_DIR, "check3_weights.csv"), index=False)
    print("\nWeights (positive = drive to press):")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(wt.round(2).to_string(index=False))
    tr = []
    for name, A_ in [("Frozen", A0), ("R shared", AR), ("X dose-input", AX)] + \
                    [(f"D {d}", D[d][1]) for d in DOSES]:
        for k in range(K):
            tr.append(dict(Model=name, From=STATE_NAMES[k + 1],
                           **{f"to {STATE_NAMES[j+1]}": A_[k, j] for j in range(K)}))
    tr = pd.DataFrame(tr)
    tr.to_csv(os.path.join(OUTPUT_DIR, "check3_transitions.csv"), index=False)
    print("\nTransition matrices:")
    print(tr.round(3).to_string(index=False))

    # occupancy decoded with each dose's own model (D)
    print(f"\nOccupancy ({FIT_EPOCH}, smoothed >= {CONFIDENCE_THRESHOLD}) decoded with each dose's OWN refit model:")
    df["D_state"] = np.nan
    for dose in DOSES:
        Wd, Ad, pid = D[dose]
        for Y, X, idx in sequences(df, BASE_INPUTS, epoch="full", dose=dose):
            fb = run_model(Wd, Ad, pid, Y, X)
            conf = fb["smooth"].max(axis=1)
            st = np.where(conf >= CONFIDENCE_THRESHOLD, fb["smooth"].argmax(axis=1) + 1, np.nan)
            df.loc[idx, "D_state"] = st
    de = df if FIT_EPOCH == "full" else df[df["Epoch"] == FIT_EPOCH]
    occD = (de.dropna(subset=["D_state"]).groupby(["Rat", "Dose_label"])["D_state"]
              .value_counts(normalize=True).mul(100).unstack(fill_value=0)
              .reindex(columns=range(1, K + 1), fill_value=0))
    occD.columns = [STATE_NAMES[c] for c in occD.columns]
    print(occD.groupby("Dose_label").mean().reindex(DOSES).round(1).to_string())
    w = occD.unstack("Dose_label").reindex(
        columns=pd.MultiIndex.from_product([[STATE_NAMES[k] for k in range(1, K + 1)], DOSES]))
    w.columns = [f"{s} | {d}" for s, d in w.columns]
    w.to_csv(os.path.join(OUTPUT_DIR, "Prism_check3_occupancy_dose_specific_model.csv"))
    for k in range(1, K + 1):
        col = STATE_NAMES[k]
        paired(w[f"{col} | High"], w[f"{col} | Saline"], f"{col:11s} High vs Saline")

    # ---------- LOO-rat CV ----------
    if not skip_cv:
        print("\nLeave-one-rat-out CV (held-out LL per trial; higher = better):")
        rats = sorted(df["Rat"].unique())
        cv = []
        for i, held in enumerate(rats):
            train = set(rats) - {held}
            tb = sequences(df, BASE_INPUTS, epoch=FIT_EPOCH, rats=train)
            tx = sequences(df, X_INPUTS, epoch=FIT_EPOCH, rats=train)
            pR = fit_warm(tb, W0, A0, pi0, 5)
            pX = fit_warm(tx, pad(W0, 9), A0, pi0, 9)
            pD = {d: fit_warm(sequences(df, BASE_INPUTS, epoch=FIT_EPOCH, dose=d, rats=train),
                              W0, A0, pi0, 5) for d in DOSES}
            for dose in DOSES:
                hb = sequences(df, BASE_INPUTS, epoch=FIT_EPOCH, dose=dose, rats={held})
                hx = sequences(df, X_INPUTS, epoch=FIT_EPOCH, dose=dose, rats={held})
                if not hb:
                    continue
                n = sum(len(s[0]) for s in hb)
                cv.append(dict(Rat=held, Dose=dose, n_trials=n,
                               F=score(W0, A0, pi0, hb)[0] / n,
                               R=score(*pR, hb)[0] / n,
                               X=score(*pX, hx)[0] / n,
                               D=score(*pD[dose], hb)[0] / n))
            print(f"  fold {i+1}/{len(rats)} done (held out {held})")
        cv = pd.DataFrame(cv)
        cv.to_csv(os.path.join(OUTPUT_DIR, "check3_LORO_CV.csv"), index=False)
        # per rat, trial-weighted across doses
        cvr = cv.assign(**{m: cv[m] * cv["n_trials"] for m in "FRXD"}).groupby("Rat")[
            ["F", "R", "X", "D", "n_trials"]].sum()
        for m in "FRXD":
            cvr[m] = cvr[m] / cvr["n_trials"]
        print("\nMean held-out LL per trial: " +
              ", ".join(f"{m} {cvr[m].mean():.4f}" for m in "FRXD"))
        paired(cvr["R"], cvr["F"], "R shared refit  vs F frozen     ")
        paired(cvr["X"], cvr["R"], "X dose-as-input vs R shared refit")
        paired(cvr["D"], cvr["R"], "D dose-specific vs R shared refit")
        print("\n  By dose (D - R, X - R):")
        for dose in DOSES:
            s = cv[cv.Dose == dose]
            paired(s["D"].values, s["R"].values, f"  {dose:6s} D vs R")
            paired(s["X"].values, s["R"].values, f"  {dose:6s} X vs R")

    # ---------- bootstrap of D weights ----------
    if n_boot and n_boot > 0:
        print(f"\nRat-level bootstrap of dose-specific weights ({n_boot} resamples):")
        rats = sorted(df["Rat"].unique())
        boot = []
        for b in range(n_boot):
            samp = rng.choice(rats, size=len(rats), replace=True)
            Wb = {}
            for dose in DOSES:
                seqs = []
                for r in samp:                      # duplicates allowed: each copy added again
                    seqs += sequences(df, BASE_INPUTS, epoch=FIT_EPOCH, dose=dose, rats={r})
                Wb[dose] = fit_warm(seqs, W0, A0, pi0, 5)[0]
            for k in range(K):
                for j, nm in enumerate(BASE_NAMES):
                    boot.append(dict(b=b, State=STATE_NAMES[k + 1], Weight=nm,
                                     Low_minus_Saline=Wb["Low"][k, j] - Wb["Saline"][k, j],
                                     High_minus_Saline=Wb["High"][k, j] - Wb["Saline"][k, j]))
            if (b + 1) % 10 == 0:
                print(f"  {b+1}/{n_boot}")
        boot = pd.DataFrame(boot)
        boot.to_csv(os.path.join(OUTPUT_DIR, "check3_bootstrap_draws.csv"), index=False)
        summ = boot.groupby(["State", "Weight"]).agg(
            High_minus_Saline_median=("High_minus_Saline", "median"),
            High_CI_low=("High_minus_Saline", lambda s: s.quantile(0.025)),
            High_CI_high=("High_minus_Saline", lambda s: s.quantile(0.975)),
            Low_minus_Saline_median=("Low_minus_Saline", "median"),
            Low_CI_low=("Low_minus_Saline", lambda s: s.quantile(0.025)),
            Low_CI_high=("Low_minus_Saline", lambda s: s.quantile(0.975)))
        summ.to_csv(os.path.join(OUTPUT_DIR, "check3_bootstrap_weight_differences.csv"))
        print(summ.round(2).to_string())
        print("  (95% CI excluding 0 = weight changed with dose. Weights in rarely occupied\n"
              "   states, e.g. Risk-averse under High, are poorly constrained: expect wide CIs.)")

    print("\n" + "=" * 70 + "\nINTERPRETATION\n" + "=" * 70 + """
Occupancy account supported if:
  - CHECK 1: within each state, observed rates under alcohol match the model-implied
    rates about as well as under saline (residuals High vs Saline n.s.), especially
    punDS and conflict in Risk-averse and Risk-taking;
  - CHECK 3: D and X do not improve held-out LL over R, X's Low/High and dose x punDS
    terms inside Risk-averse are small, and occupancy decoded with the dose-specific
    model (D) still shows less Risk-averse under alcohol.
Within-state policy change indicated if:
  - Risk-averse (or Risk-taking) punDS/conflict responding exceeds the model under
    alcohol, D or X beat R out of sample, and D's punDS weight in Risk-averse shrinks
    under alcohol with a bootstrap CI excluding 0.
Either outcome is reportable; the second would change the alcohol section from
"loss of the Risk-averse state" to "weakened punishment weighting within it".
""")
    df.to_csv(os.path.join(OUTPUT_DIR, "alcohol_trials_with_all_posteriors.csv"), index=False)
    print(f"Done. Outputs in '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cv", action="store_true", help="skip leave-one-rat-out CV (slowest part)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="bootstrap resamples (0 = skip)")
    a = ap.parse_args()
    try:
        main(skip_cv=a.skip_cv, n_boot=a.n_boot)
    finally:
        if isinstance(sys.stdout, Tee):
            tee = sys.stdout
            sys.stdout = tee.o
            tee.f.close()
