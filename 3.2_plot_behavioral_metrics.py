"""
3.2_plot_behavioral_metrics.py
------------------------------
Per-rat response rate (and latency on pressed trials) for each state x cue (Fig. 3E).

Input   Data/collated_2026_conflict_data_with_Probabilities_065_Past_PelletShock_v5.csv (from 2.1)
Output  HMM_Outputs/summary_statistics_by_rat_state_cue.csv
            complete Rat x State x Cue grid (empty where a rat has no trials of that
            combination); input to the State x Cue ANOVA in Prism
        HMM_Outputs/Fig3E_Response_Rate_Bar.{svg,png}     Fig. 3E
        HMM_Outputs/Latency_by_State_Bar.{svg,png}        not in the paper

Response rates by state use smoothed state assignments, which condition on the choice
being summarised; see 3.4 for assignment-free and one-step-ahead versions.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

# =========================================================
# 0. CONFIGURATION & COLOURS
# =========================================================
INPUT_PATH  = os.path.join("Data", "collated_2026_conflict_data_with_Probabilities_065_Past_PelletShock_v5.csv")
OUTPUT_DIR  = "HMM_Outputs"
LATENCY_COL = 'Latency'

CUE_PALETTE = {'rewDS': (0.12, 0.80, 0.01), 'punDS': (0.87, 0.01, 0.12), 'Conf': (0.01, 0.11, 0.86)}
CUE_ORDER = ['rewDS', 'punDS', 'Conf']
CUE_MAP = {'reward': 'rewDS', 'punish': 'punDS', 'conflict': 'Conf'}
STATE_NAMES = {1: 'Risk-taking', 2: 'Risk-averse', 3: 'Repeating'}

# =========================================================
# 1. LOAD DATA
# =========================================================
if not os.path.exists(INPUT_PATH):
    sys.exit(f"Error: {INPUT_PATH} not found. Run 2.1_States_to_trials.py first.")
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT_PATH, dtype={"Session": str})

df = df.dropna(subset=['Inferred_State']).copy()          # assigned trials only
df['Inferred_State'] = df['Inferred_State'].astype(int)
df['Cue_Label'] = pd.Categorical(df['Cue'].str.lower().map(CUE_MAP), categories=CUE_ORDER, ordered=True)
df['Lever_Press'] = df['Outcome'].astype(int)

# =========================================================
# 2. SUBJECT-LEVEL AVERAGES
# =========================================================
keys = ['Rat', 'Inferred_State', 'Cue_Label']
df_resp = df.groupby(keys, observed=True)['Lever_Press'].agg(['mean', 'size']).reset_index()
df_resp['Response_Percentage'] = df_resp['mean'] * 100
df_resp = df_resp.rename(columns={'size': 'N_trials'}).drop(columns='mean')

if LATENCY_COL in df.columns:
    df_lat = (df[df['Lever_Press'] == 1].groupby(keys, observed=True)[LATENCY_COL]
              .mean().reset_index())
else:
    print(f"Warning: latency column '{LATENCY_COL}' not found; latency plot skipped.")
    df_lat = pd.DataFrame(columns=keys + [LATENCY_COL])

# Complete Rat x State x Cue grid (a row even where a rat has no trials of a combination)
full = pd.MultiIndex.from_product([sorted(df['Rat'].unique()), sorted(df['Inferred_State'].unique()), CUE_ORDER],
                                  names=keys).to_frame(index=False)
full['Cue_Label'] = pd.Categorical(full['Cue_Label'], categories=CUE_ORDER, ordered=True)
summary_df = (full.merge(df_resp, on=keys, how='left')
                  .merge(df_lat, on=keys, how='left'))
summary_df['State_Name'] = summary_df['Inferred_State'].map(STATE_NAMES)
summary_file = os.path.join(OUTPUT_DIR, "summary_statistics_by_rat_state_cue.csv")
summary_df.to_csv(summary_file, index=False)
print(f"Saved {summary_file}")
print(df_resp.groupby(['Inferred_State', 'Cue_Label'], observed=True)['Response_Percentage']
      .agg(['mean', 'sem', 'count']).round(1).to_string())


def bar_with_points(data, value, ylabel, title, stem, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=data, x='Inferred_State', y=value, hue='Cue_Label', hue_order=CUE_ORDER,
                palette=CUE_PALETTE, errorbar='se', capsize=0.1,
                err_kws={'linewidth': 1.5, 'color': 'black'}, edgecolor='black', linewidth=1.5, ax=ax)
    sns.stripplot(data=data, x='Inferred_State', y=value, hue='Cue_Label', hue_order=CUE_ORDER,
                  palette={c: 'k' for c in CUE_ORDER}, dodge=True, alpha=0.6, jitter=True, size=5,
                  ax=ax, legend=False)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_xlabel("Inferred State", fontsize=12, fontweight='bold')
    if ylim:
        ax.set_ylim(*ylim)
    states = sorted(data['Inferred_State'].unique())
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels([f"State {s}\n{STATE_NAMES.get(s, '')}" for s in states], fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], title="", frameon=False, fontsize=11, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, stem + ".svg"), format="svg")
    plt.savefig(os.path.join(OUTPUT_DIR, stem + ".png"), dpi=300)
    plt.close(fig)
    print(f"Saved {os.path.join(OUTPUT_DIR, stem)}.svg / .png")


# =========================================================
# 3. FIGURES
# =========================================================
bar_with_points(df_resp, 'Response_Percentage', "Response Rate (%)",
                "Behavioral Profile by Latent State", "Fig3E_Response_Rate_Bar", ylim=(0, 105))
if not df_lat.empty:
    bar_with_points(df_lat, LATENCY_COL, "Latency (s)",
                    "Response Latency by Latent State (pressed trials only)", "Latency_by_State_Bar")
