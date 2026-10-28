"""
1.3_model_selection_metrics.py
------------------------------
Plots log-likelihood, AIC and BIC against the number of states (Fig. S2C-D).

Input   HMM_Outputs/Model_Selection_Metrics.csv   (written by 1.1)
Output  HMM_Outputs/SuppFig_Model_Selection.{svg,png}
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

OUTPUT_DIR    = "HMM_Outputs"
METRICS_FILE  = os.path.join(OUTPUT_DIR, "Model_Selection_Metrics.csv")
CHOSEN_STATES = 3

# 1. Load data
if not os.path.exists(METRICS_FILE):
    sys.exit(f"Error: could not find '{METRICS_FILE}'. Run 1.1_model_fit.py first.")
df = pd.read_csv(METRICS_FILE)
states = df['Num_States'].values

# 2. Multi-panel plot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5))
colors = ['#494949', '#011186', '#87010c']
metrics = ['Log_Likelihood', 'AIC', 'BIC']
titles = ['Log-Likelihood (LL)', 'Akaike Info Criterion (AIC)', 'Bayesian Info Criterion (BIC)']
ylabels = ['Log-Likelihood Value', 'AIC Value', 'BIC Value']

for i, ax in enumerate(axes):
    metric = metrics[i]
    ax.plot(states, df[metric], marker='o', color=colors[i], linewidth=2, markersize=6, zorder=3)
    ax.axvline(CHOSEN_STATES, color='red', linestyle='--', alpha=0.7, linewidth=1.5, zorder=1)
    ax.set_title(titles[i], fontweight='bold', fontsize=12, pad=12)
    ax.set_xlabel('Number of States', fontweight='bold', fontsize=10)
    ax.set_ylabel(ylabels[i], fontweight='bold', fontsize=10)
    ax.set_xticks(states)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# 3. Save
plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUTPUT_DIR, 'SuppFig_Model_Selection.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'SuppFig_Model_Selection.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print(df.round(1).to_string(index=False))
print(f"\nSaved {OUTPUT_DIR}/SuppFig_Model_Selection.svg and .png")
