"""
1.2_GLM_wts.py
--------------
Plots the per-state GLM weights of the retained 3-state model (Fig. 3B).

Input   HMM_Outputs/3_State_GLM_Weights.csv   (written by 1.1)
Output  HMM_Outputs/Fig_GLM_Weights.{svg,png}

Weights are selected BY COLUMN NAME, so the display order (PLOT_ORDER) cannot swap
the history terms. The script refuses the legacy hand-edited weights file, whose
Past Pellet / Past Shock headers were transposed relative to the fit order.
Model-implied press probabilities per state, cue and previous outcome (e.g. the
Repeating-state values 0.99 / 0.89 / 0.005 quoted in the Results) are printed by
3.4_predictive_posterior_state_profiles.py (emission_table.csv).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'   # editable text in SVG
plt.rcParams['pdf.fonttype'] = 42

# =========================================================
# 0. CONFIGURATION
# =========================================================
NUM_STATES   = 3
OUTPUT_DIR   = "HMM_Outputs"
WEIGHTS_FILE = os.path.join(OUTPUT_DIR, f'{NUM_STATES}_State_GLM_Weights.csv')
OUT_STEM     = os.path.join(OUTPUT_DIR, 'Fig_GLM_Weights')

# Canonical column names written by 1.1 (fit order: rewDS, punDS, past pellet, past shock, bias)
FEATURES = ['Rew_Cue_Weight', 'Pun_Cue_Weight',
            'Past_Pellet_Weight', 'Past_Shock_Weight', 'Baseline_Bias']
DISPLAY_NAMES = {'Rew_Cue_Weight': 'rewDS', 'Pun_Cue_Weight': 'punDS',
                 'Past_Pellet_Weight': 'Past\nPellet', 'Past_Shock_Weight': 'Past\nShock',
                 'Baseline_Bias': 'Baseline\nBias'}
# Left-to-right order of the bars. Fig. 3B shows Past Shock before Past Pellet.
PLOT_ORDER = ['Rew_Cue_Weight', 'Pun_Cue_Weight',
              'Past_Shock_Weight', 'Past_Pellet_Weight', 'Baseline_Bias']

# Headers of the old hand-edited file. If present, abort.
LEGACY_HEADERS = {'rewDS', 'punDS', 'Past Shock', 'Past Pellet'}

STATE_COLORS = {'State_1': '#B0B300', 'State_2': '#B500B3', 'State_3': '#00B0B3'}
STATE_LABELS = {'State_1': 'State 1: Risk-taking',
                'State_2': 'State 2: Risk-averse',
                'State_3': 'State 3: Repeating'}

# =========================================================
# 1. LOAD AND VALIDATE
# =========================================================
if not os.path.exists(WEIGHTS_FILE):
    raise SystemExit(f"Error: could not find '{WEIGHTS_FILE}'.\n"
                     f"Run 1.1_model_fit.py (CHOSEN_STATES = {NUM_STATES}) to generate it.")
df_weights = pd.read_csv(WEIGHTS_FILE, index_col=0)
cols = set(df_weights.columns)

if LEGACY_HEADERS & cols:
    raise SystemExit(
        "\n*** ABORTED: legacy weights file detected. ***\n"
        f"'{WEIGHTS_FILE}' uses the old hand-edited headers (rewDS / punDS / Past Shock / Past Pellet),\n"
        "in which the two history columns are TRANSPOSED relative to the model's fit order.\n"
        "Delete this file and re-run 1.1_model_fit.py to regenerate it.\n")

missing = [f for f in FEATURES if f not in cols]
if missing:
    raise SystemExit(f"\nError: '{WEIGHTS_FILE}' is missing column(s) {missing}.\n"
                     f"Found: {list(df_weights.columns)}. This file should be written by 1.1_model_fit.py.\n")

state_index = [f'State_{i+1}' for i in range(NUM_STATES)]
missing_rows = [s for s in state_index if s not in df_weights.index]
if missing_rows:
    raise SystemExit(f"Error: missing row(s) {missing_rows} in '{WEIGHTS_FILE}'.")

print(f"Loaded weights from '{WEIGHTS_FILE}' (positive = drive to press):\n")
print(df_weights.loc[state_index, FEATURES].round(3).to_string())

# =========================================================
# 2. PLOT
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(PLOT_ORDER))
width = 0.23

for i, state in enumerate(state_index):
    weights = df_weights.loc[state, PLOT_ORDER].values.astype(float)   # selected by name
    ax.bar(x + width * i, weights, width, label=STATE_LABELS[state],
           color=STATE_COLORS[state], edgecolor='black', linewidth=1.2)

ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
ax.set_ylabel('GLM weight\n(positive = drive to press, negative = drive to omit)',
              fontweight='bold', fontsize=12)
ax.set_title('GLM Weights Per State', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels([DISPLAY_NAMES[f] for f in PLOT_ORDER], fontweight='bold', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=11)
plt.tight_layout()

plt.savefig(OUT_STEM + '.svg', format='svg', bbox_inches='tight')
plt.savefig(OUT_STEM + '.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved {OUT_STEM}.svg and .png")
