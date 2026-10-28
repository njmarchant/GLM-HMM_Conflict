"""
1.1_model_fit.py
----------------
Fits Bernoulli GLM-HMMs with K = 1..MAX_STATES latent states to the Phase 3
trial-by-trial press/omit choices, reports log-likelihood, AIC and BIC for every K,
and exports the weights and transition matrix of the retained model (K = 3).

Inputs   Data/collated_2026_conflict_data_raw.csv
Outputs  final_{K}_state_model_v5.pkl                      (K = 1..5; only when refitting)
         HMM_Outputs/Model_Selection_Metrics.csv            (Fig. S2C-D values)
         HMM_Outputs/Model_Selection_AIC_BIC.{png,svg}
         HMM_Outputs/3_State_GLM_Weights.csv                (Fig. 3B; read by 1.2)
         HMM_Outputs/3_State_Transitions.csv                (Fig. 3A, 3C)

REFIT_MODELS = False (default) evaluates the shipped model files without refitting, so
the published numbers are reproduced exactly. REFIT_MODELS = "missing" loads the model
files that exist and fits only the K for which no file exists (e.g. K = 1 and 2 when only
the 3-, 4- and 5-state files were kept). REFIT_MODELS = True refits every K by
multi-start EM and saves each model; existing model files are never overwritten unless
OVERWRITE_MODELS = True. After a refit, state numbering may differ from the published
model: compare the exported weights with the README before running Stage 2.

AIC and BIC use the data log-likelihood (hmm.log_likelihood), not the value returned by
hmm.fit(), which also contains ssm's N(0, 1000^2) prior on the GLM weights (about -39.1
per state). The best of the random initialisations is still chosen on the fit objective.
Free parameters: (K - 1) initial-state + K(K - 1) transition + K x 5 GLM weights.

Run from the repository root.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import ssm
    SSM_IMPORT_ERROR = None
except ImportError as e:
    ssm = None
    SSM_IMPORT_ERROR = e
SSM_HELP = ("If the error mentions NumPy 1.x / 2.x or _ARRAY_API, ssm was compiled against a different NumPy "
            "than the one installed: rebuild it (README, Installation > Troubleshooting).")

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

# =========================================================
# 0. CONFIGURATION
# =========================================================
INPUT_PATH       = os.path.join("Data", "collated_2026_conflict_data_raw.csv")
OUTPUT_DIR       = "HMM_Outputs"
MODEL_TEMPLATE   = "final_{k}_state_model_v5.pkl"

MAX_STATES       = 5       # fit / evaluate K = 1..MAX_STATES
CHOSEN_STATES    = 3       # model whose weights and transitions are exported
NUM_INITS        = 10      # random initialisations per K (refit only)
NUM_ITERS        = 200     # EM iterations
TOLERANCE        = 1e-4    # EM tolerance
MIN_TRIALS       = 5       # sessions with fewer trials are excluded
SEED             = 14

REFIT_MODELS     = False   # False: shipped pkl files only; "missing": fit only absent K; True: refit every K
OVERWRITE_MODELS = False   # only with REFIT_MODELS: replace existing pkl files

INPUT_COLS   = ['rew_cue_val', 'pun_cue_val', 'past_pellet', 'past_shock', 'bias']
WEIGHT_COLS  = ['Rew_Cue_Weight', 'Pun_Cue_Weight', 'Past_Pellet_Weight',
                'Past_Shock_Weight', 'Baseline_Bias']


# =========================================================
# 1. DATA PREPARATION
# =========================================================
def prepare_hmm_data(df):
    """Converts the trial table into ssm's list-of-arrays format, one sequence per
    Rat x Session. History inputs are shifted within session (0 on the first trial)."""
    print("Preparing data: separating by Rat and Session...")
    df = df.sort_values(by=['Rat', 'Session', 'Trial']).copy()
    g = [df['Rat'], df['Session']]
    df['past_pellet'] = (df['Response'] == 'Response-Pellet').astype(int).groupby(g).shift(1).fillna(0)
    df['past_shock'] = (df['Response'] == 'Response-Shock').astype(int).groupby(g).shift(1).fillna(0)
    df['rew_cue_val'] = (df['Cue'] == 'reward') | (df['Cue'] == 'conflict')
    df['pun_cue_val'] = (df['Cue'] == 'punish') | (df['Cue'] == 'conflict')
    df['bias'] = 1.0
    df['Lever_Press'] = df['Outcome'].astype(int)          # 1 = press, 0 = omission

    datas, inputs, session_meta = [], [], []
    for (rat, session), sd in df.groupby(['Rat', 'Session']):
        if len(sd) < MIN_TRIALS:
            continue
        datas.append(sd['Lever_Press'].values.reshape(-1, 1).astype(int))
        inputs.append(sd[INPUT_COLS].astype(float).values)
        session_meta.append({'Rat': rat, 'Session': session, 'Indices': sd.index})

    n_trials = sum(len(d) for d in datas)
    print(f"Successfully processed {len(datas)} independent sessions, {n_trials} trials, "
          f"{df['Rat'].nunique()} rats.")
    return datas, inputs, session_meta, df


# =========================================================
# 2. FITTING / LOADING
# =========================================================
def n_free_params(k, input_dim):
    return (k - 1) + k * (k - 1) + k * input_dim


def fit_best_of_inits(k, datas, inputs):
    """Multi-start EM; returns the model with the highest final fit objective."""
    input_dim = inputs[0].shape[1]
    best_obj, best_hmm = -np.inf, None
    for init in range(NUM_INITS):
        hmm = ssm.HMM(k, 1, input_dim,
                      observations="input_driven_obs",
                      observation_kwargs=dict(C=2),
                      transitions="standard")
        train_lls = hmm.fit(datas, inputs=inputs, method="em",
                            num_iters=NUM_ITERS, tolerance=TOLERANCE)
        if train_lls[-1] > best_obj:
            best_obj, best_hmm = train_lls[-1], hmm
    print(f"  best fit objective (log-likelihood + weight prior): {best_obj:.2f}")
    return best_hmm


def get_models(datas, inputs):
    models = {}
    for k in range(1, MAX_STATES + 1):
        path = MODEL_TEMPLATE.format(k=k)
        refit = REFIT_MODELS is True or (REFIT_MODELS == "missing" and not os.path.exists(path))
        if refit:
            print(f"\nFitting {k}-state model ({NUM_INITS} random initialisations)...")
            hmm = fit_best_of_inits(k, datas, inputs)
            if os.path.exists(path) and not OVERWRITE_MODELS:
                alt = path.replace(".pkl", "_refit.pkl")
                print(f"  {path} exists and OVERWRITE_MODELS is False: saved as {alt}")
                path = alt
            with open(path, "wb") as f:
                pickle.dump(hmm, f)
            print(f"  saved {path}")
        else:
            if not os.path.exists(path):
                print(f"[skip] {path} not found (set REFIT_MODELS = \"missing\" to fit it).")
                continue
            with open(path, "rb") as f:
                hmm = pickle.load(f)
            print(f"Loaded {path}")
        models[k] = hmm
    return models


def model_selection_table(models, datas, inputs):
    input_dim = inputs[0].shape[1]
    n = sum(len(d) for d in datas)
    rows = []
    for k, hmm in sorted(models.items()):
        ll = float(hmm.log_likelihood(datas, inputs=inputs))     # data log-likelihood, no prior
        p = n_free_params(k, input_dim)
        rows.append({'Num_States': k, 'Log_Likelihood': ll, 'Free_Parameters': p,
                     'AIC': 2 * p - 2 * ll, 'BIC': p * np.log(n) - 2 * ll, 'N_trials': n})
        print(f"  K = {k}: LL = {ll:.2f}, parameters = {p}, AIC = {rows[-1]['AIC']:.1f}, "
              f"BIC = {rows[-1]['BIC']:.1f}")
    return pd.DataFrame(rows)


# =========================================================
# 3. OUTPUTS
# =========================================================
def plot_model_selection(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title, color in zip(
            axes, ['Log_Likelihood', 'AIC', 'BIC'],
            ['Log-likelihood', 'Akaike Information Criterion (AIC)',
             'Bayesian Information Criterion (BIC)'],
            ['black', 'blue', 'red']):
        ax.plot(results_df['Num_States'], results_df[col], marker='o', color=color)
        ax.axvline(CHOSEN_STATES, color='gray', linestyle=':', linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel('Number of States')
        ax.set_ylabel(col.replace('_', '-'))
        ax.set_xticks(results_df['Num_States'])
    plt.tight_layout()
    for ext in ['png', 'svg']:
        plt.savefig(os.path.join(OUTPUT_DIR, f"Model_Selection_AIC_BIC.{ext}"), dpi=300)
    plt.close(fig)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "Model_Selection_Metrics.csv"), index=False)
    print(f"Saved Model_Selection_Metrics.csv and Model_Selection_AIC_BIC to {OUTPUT_DIR}/")


def export_model_parameters(hmm, num_states):
    trans = pd.DataFrame(hmm.transitions.transition_matrix,
                         columns=[f"To_State_{i+1}" for i in range(num_states)],
                         index=[f"From_State_{i+1}" for i in range(num_states)])
    trans.to_csv(os.path.join(OUTPUT_DIR, f"{num_states}_State_Transitions.csv"))

    # ssm models the log-odds of class 0 (omission) relative to the last class (press);
    # multiply by -1 so that positive weights drive pressing.
    raw = hmm.observations.Wk
    W = raw[:, 0, :] * -1 if raw.ndim == 3 else raw.squeeze() * -1
    weights = pd.DataFrame(np.atleast_2d(W), columns=WEIGHT_COLS,
                           index=[f"State_{i+1}" for i in range(num_states)])
    weights.to_csv(os.path.join(OUTPUT_DIR, f"{num_states}_State_GLM_Weights.csv"))

    print(f"\n{num_states}-state GLM weights (positive = drive to press):")
    print(weights.round(2).to_string())
    print(f"\n{num_states}-state transition matrix:")
    print(trans.round(3).to_string())


# =========================================================
# 4. MAIN
# =========================================================
if __name__ == "__main__":
    np.random.seed(SEED)
    if ssm is None:
        raise SystemExit(f"ssm could not be imported: {SSM_IMPORT_ERROR}\n{SSM_HELP}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, dtype={"Session": str})
    datas, inputs, session_meta, df = prepare_hmm_data(df)

    models = get_models(datas, inputs)
    if not models:
        raise SystemExit("No models to evaluate.")

    print("\nModel selection (data log-likelihood, no prior):")
    results_df = model_selection_table(models, datas, inputs)
    plot_model_selection(results_df)

    if CHOSEN_STATES in models:
        export_model_parameters(models[CHOSEN_STATES], CHOSEN_STATES)
    else:
        print(f"WARNING: no {CHOSEN_STATES}-state model; weights and transitions not exported.")
    print("\nDone.")
