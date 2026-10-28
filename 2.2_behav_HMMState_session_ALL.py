"""
2.2_behav_HMMState_session_ALL.py
---------------------------------
One figure per Phase 3 session (Fig. 3F shows two of them):
  top     posterior probability of each state across trials (smoothed, from 2.1)
  bottom  trial type (background colour), choice (press / omission) and the
          assigned state (posterior >= 0.65; unassigned trials are left blank)

Input   Data/collated_2026_conflict_data_with_Probabilities_065_Past_PelletShock_v5.csv (from 2.1)
Output  Session_Traces_Library_065/Rat_<rat>_Session_<session>.png
        Sessions listed in ONLY_SESSIONS are also saved as SVG (editable text). Leave
        ONLY_SESSIONS empty to save every session as PNG only.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

# =========================================================
# 0. CONFIGURATION
# =========================================================
INPUT_PATH = os.path.join("Data", "collated_2026_conflict_data_with_Probabilities_065_Past_PelletShock_v5.csv")
OUTPUT_DIR = "Session_Traces_Library_065"
# (Rat, Session) pairs to also save as SVG; Fig. 3F uses R13 sessions 3.03 and 3.10.
ONLY_SESSIONS = [("R13", "3.03"), ("R13", "3.10")]
PNG_DPI = 150

STATE_COLORS = {1: '#B0B300', 2: '#B500B3', 3: '#00B0B3'}
STATE_NAMES = {1: 'Risk-taking', 2: 'Risk-averse', 3: 'Repeating'}
CUE_COLORS = {'reward': '#90ee90', 'punish': '#ffcccb', 'conflict': '#add8e6'}
# Vertical position of each state in the bottom panel: State 1 top, 3 middle, 2 bottom
Y_POSITION = {1: 3, 3: 2, 2: 1}


def plot_session(d, rat, session, out_stem, save_svg):
    d = d.sort_values('Trial')
    trials = d['Trial'].values
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [1.2, 1], 'hspace': 0.08})

    # --- top: posterior probability of each state ---
    for k in (1, 2, 3):
        p = d[f'Prob_State_{k}'].values
        ax0.plot(trials, p, color=STATE_COLORS[k], lw=1.6, label=f'p(state {k}) {STATE_NAMES[k]}')
        ax0.fill_between(trials, 0, p, color=STATE_COLORS[k], alpha=0.12, lw=0)
    ax0.set_ylim(0, 1.02)
    ax0.set_yticks([0, 0.5, 1])
    ax0.set_ylabel('Probability', fontweight='bold')
    ax0.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    ax0.set_title(f'Rat {rat} | Session {session}', loc='left', fontsize=12, fontweight='bold')
    ax0.spines[['top', 'right']].set_visible(False)

    # --- bottom: cue background, choice, assigned state ---
    for t, cue in zip(trials, d['Cue']):
        ax1.axvspan(t - 0.5, t + 0.5, color=CUE_COLORS.get(cue, 'white'), alpha=0.35, lw=0)
    ax1.scatter(trials, d['Outcome'], color='black', s=18, zorder=5)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Omit', 'Press'], fontweight='bold')
    ax1.set_ylim(-0.15, 1.15)
    ax1.set_ylabel('Action', fontweight='bold')
    ax1.set_xlabel('Trial #', fontweight='bold')
    ax1.set_xlim(trials.min() - 1, trials.max() + 1)

    ax2 = ax1.twinx()
    y_state = d['Inferred_State'].map(Y_POSITION)
    known = y_state.notna().values
    ax2.plot(trials[known], y_state[known], color='gray', alpha=0.4, lw=1, zorder=4)
    for k, col in STATE_COLORS.items():
        sel = (d['Inferred_State'] == k).values
        ax2.scatter(trials[sel], y_state[sel], color=col, s=22, zorder=6, edgecolors='white', linewidths=0.4)
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(['2', '3', '1'])
    ax2.set_ylim(0.5, 3.5)
    ax2.set_ylabel('Inferred state', fontweight='bold')

    legend = [mpatches.Patch(color=CUE_COLORS['reward'], alpha=0.5, label='rewDS'),
              mpatches.Patch(color=CUE_COLORS['punish'], alpha=0.5, label='punDS'),
              mpatches.Patch(color=CUE_COLORS['conflict'], alpha=0.5, label='Conflict'),
              plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=7, label='Action')]
    ax1.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=4, frameon=False)

    plt.savefig(out_stem + '.png', dpi=PNG_DPI, bbox_inches='tight')
    if save_svg:
        plt.savefig(out_stem + '.svg', bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        sys.exit(f"Error: {INPUT_PATH} not found. Run 2.1_States_to_trials.py first.")
    df = pd.read_csv(INPUT_PATH, dtype={"Session": str})
    need = ['Rat', 'Session', 'Trial', 'Cue', 'Outcome', 'Inferred_State',
            'Prob_State_1', 'Prob_State_2', 'Prob_State_3']
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"Error: {INPUT_PATH} is missing column(s) {missing}.")
    df = df.dropna(subset=['Prob_State_1'])          # sessions that were decoded
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    svg_set = {(str(r), str(s)) for r, s in ONLY_SESSIONS}
    found = set()
    rats = sorted(df['Rat'].unique())
    print(f"Exporting session traces for {len(rats)} rats...")
    for rat in rats:
        dr = df[df['Rat'] == rat]
        for session in sorted(dr['Session'].unique()):
            save_svg = (str(rat), str(session)) in svg_set
            found |= {(str(rat), str(session))} if save_svg else set()
            stem = os.path.join(OUTPUT_DIR, f"Rat_{rat}_Session_{session}")
            plot_session(dr[dr['Session'] == session], rat, session, stem, save_svg)
        print(f"  {rat}: {dr['Session'].nunique()} sessions")
    for r, s in sorted(svg_set - found):
        print(f"WARNING: ({r}, {s}) in ONLY_SESSIONS was not found in the table.")
    print(f"\nSaved to '{OUTPUT_DIR}/'.")
