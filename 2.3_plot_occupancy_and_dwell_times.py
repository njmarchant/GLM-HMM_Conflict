"""
2.3_plot_occupancy_and_dwell_times.py
-------------------------------------
State occupancy per rat (Fig. S3A) and state persistence / dwell times (Fig. S3B).

Occupancy = percentage of a rat's assigned trials (posterior >= 0.65) in each state.
Runs: within a session, assigned trials form runs of the same state. A run may bridge up
to MAX_GAP (3) consecutive unassigned trials if the same state resumes on both sides;
bridged trials count towards its length. A run ends when a different state is assigned,
after more than MAX_GAP unassigned trials, or at the end of the session (right-censored).

Input   Data/collated_2026_conflict_data_with_Probabilities_065_Past_PelletShock_v5.csv (from 2.1)
Output  HMM_Outputs/SuppFig_Occupancy_Dwell_gap3.{svg,png}
        HMM_Outputs/occupancy_per_rat.csv   (rows = rats, one column per state; for Prism)
        HMM_Outputs/dwell_runs_gap3.csv     (one row per run)
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
# 0. CONFIGURATION
# =========================================================
DATA_FILE = os.path.join("Data", "collated_2026_conflict_data_with_Probabilities_065_Past_PelletShock_v5.csv")
OUTPUT_DIR = "HMM_Outputs"
STATE_THRESHOLD = 0.65    # confidence threshold for assigning a trial to a state
MAX_GAP = 3               # consecutive unassigned trials a run may bridge (0 = no bridging)
COUNT_GAP_TRIALS = True   # bridged trials count towards run length

STATES = ['State 1', 'State 2', 'State 3']
STATE_PALETTE = {'State 1': '#B0B300', 'State 2': '#B500B3', 'State 3': '#00B0B3'}
STATE_LABELS = ['State 1:\nRisk-Taking', 'State 2:\nRisk-Averse', 'State 3:\nRepeating']

# =========================================================
# 1. LOAD AND ASSIGN STATES
# =========================================================
print(f"Loading dataset: {DATA_FILE}...")
if not os.path.exists(DATA_FILE):
    sys.exit(f"Error: could not find '{DATA_FILE}'. Run 2.1_States_to_trials.py first.")
df = pd.read_csv(DATA_FILE, dtype={"Session": str})
prob_cols = ['Prob_State_1', 'Prob_State_2', 'Prob_State_3']
df = df.dropna(subset=prob_cols)
df = df.sort_values(by=['Rat', 'Session', 'Trial']).copy()   # chronological order within session

print("Assigning trials to states based on threshold...")
df['Highest_Prob_State'] = df[prob_cols].idxmax(axis=1)
df['Max_Prob_Value'] = df[prob_cols].max(axis=1)
df['Inferred_State'] = df['Highest_Prob_State'].str.extract(r'(\d+)', expand=False).astype(float)
df.loc[df['Max_Prob_Value'] < STATE_THRESHOLD, 'Inferred_State'] = np.nan

# =========================================================
# 2. OCCUPANCY PER RAT
# =========================================================
print("Calculating state occupancy per rat...")
rows = []
for rat, rat_df in df.groupby('Rat'):
    total_assigned = rat_df['Inferred_State'].notna().sum()
    if total_assigned == 0:
        continue
    for s in [1, 2, 3]:
        rows.append({'Rat': rat, 'State': f'State {s}',
                     'Occupancy': 100 * (rat_df['Inferred_State'] == s).sum() / total_assigned})
df_rat_occupancy = pd.DataFrame(rows)
for s in STATES:
    print(f" -> {s} group mean occupancy: "
          f"{df_rat_occupancy.loc[df_rat_occupancy['State'] == s, 'Occupancy'].mean():.2f}%")


# =========================================================
# 3. DWELL TIMES (RUN LENGTHS)
# =========================================================
def compute_runs(states, max_gap=0, count_gap_trials=True):
    """Run lengths per state within one session.
    max_gap: number of consecutive unassigned (NaN) trials allowed inside a run
             when the SAME state resumes afterwards.
    count_gap_trials: if True, bridged NaN trials are added to the run length."""
    runs = {1: [], 2: [], 3: []}
    reasons = {'switch': 0, 'switch_after_gap': 0, 'gap': 0, 'session_end': 0}
    cur, length, gap = None, 0, 0
    for s in states:
        if np.isnan(s):
            if cur is not None:
                gap += 1
                if gap > max_gap:                       # gap too long: run ends here
                    runs[int(cur)].append(length); reasons['gap'] += 1
                    cur, length, gap = None, 0, 0
            continue
        if cur is not None and s == cur:                # same state resumes (possibly after short gap)
            length += 1 + (gap if count_gap_trials else 0)
        else:                                           # new run
            if cur is not None:
                runs[int(cur)].append(length)
                reasons['switch' if gap == 0 else 'switch_after_gap'] += 1
            cur, length = s, 1
        gap = 0
    if cur is not None:
        runs[int(cur)].append(length); reasons['session_end'] += 1
    return runs, reasons


print(f"Calculating run lengths (MAX_GAP = {MAX_GAP})...")
dwell_times = {1: [], 2: [], 3: []}
dwell_rows = []   # one row per run, keeps the Rat ID for per-rat summaries
end_reasons = {'switch': 0, 'switch_after_gap': 0, 'gap': 0, 'session_end': 0}

for (rat, session), session_df in df.groupby(['Rat', 'Session']):
    runs, reasons = compute_runs(session_df['Inferred_State'].values.astype(float),
                                 max_gap=MAX_GAP, count_gap_trials=COUNT_GAP_TRIALS)
    for s in (1, 2, 3):
        dwell_times[s].extend(runs[s])
        dwell_rows += [{'Rat': rat, 'Session': session, 'State': f'State {s}', 'Run_Length': r} for r in runs[s]]
    for k in end_reasons:
        end_reasons[k] += reasons[k]

total_runs = sum(end_reasons.values())
print("\nHow runs terminate:")
for k, v in end_reasons.items():
    print(f"   {k:<17} {v:>5}  ({100 * v / total_runs:.1f}%)")
for s in (1, 2, 3):
    r = np.array(dwell_times[s])
    print(f"   State {s}: {np.mean(r <= 3) * 100:.1f}% of runs are <= 3 trials")
print("   (Note: 'session_end' runs are right-censored - true dwell is longer.)\n")

df_dwell = pd.DataFrame(dwell_rows)
for s_num in (1, 2, 3):
    r = dwell_times[s_num]
    print(f" -> State {s_num}: median dwell = {np.median(r):.1f} trials, "
          f"mean = {np.mean(r):.2f} (N = {len(r)} runs)")

os.makedirs(OUTPUT_DIR, exist_ok=True)
df_rat_occupancy.pivot(index='Rat', columns='State', values='Occupancy').reindex(columns=STATES)\
    .to_csv(os.path.join(OUTPUT_DIR, 'occupancy_per_rat.csv'))
df_dwell.to_csv(os.path.join(OUTPUT_DIR, f'dwell_runs_gap{MAX_GAP}.csv'), index=False)

# =========================================================
# 4. FIGURE
# =========================================================
print("Generating figure...")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5.5))


def paired_rat_plot(ax, data, value_col):
    """Mean bar + SEM, one point per rat aligned in each column, grey line per rat."""
    wide = data.pivot(index='Rat', columns='State', values=value_col).reindex(columns=STATES)
    x = np.arange(len(STATES))
    means, sems = wide.mean(), wide.sem()
    ax.bar(x, means, width=0.6, color=[STATE_PALETTE[s] for s in STATES], alpha=0.35,
           edgecolor='#999999', linewidth=1.5, zorder=1)
    for _, row in wide.iterrows():
        ax.plot(x, row.values, color='grey', alpha=0.5, linewidth=1, zorder=2)
    for i, s in enumerate(STATES):
        y = wide[s].dropna()
        ax.scatter(np.full(len(y), i), y, s=60, color=STATE_PALETTE[s],
                   edgecolor='black', linewidth=1, zorder=3)
    ax.errorbar(x, means, yerr=sems, fmt='none', ecolor='black', elinewidth=2,
                capsize=10, capthick=2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_LABELS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    return means


# Panel A: occupancy per rat
ax1 = axes[0]
paired_rat_plot(ax1, df_rat_occupancy, 'Occupancy')
ax1.set_title('State Occupancy', fontweight='bold', fontsize=12, pad=12)
ax1.set_xlabel('Inferred State', fontweight='bold', fontsize=10)
ax1.set_ylabel('Percentage of classified trials (%)', fontweight='bold', fontsize=10)
ax1.set_ylim(0, 105)

# Panel B: dwell times, all runs pooled across rats
ax2 = axes[1]
sns.stripplot(
    data=df_dwell, x='State', y='Run_Length', order=STATES, hue='State', palette=STATE_PALETTE,
    ax=ax2, size=4, alpha=0.4, jitter=0.2, linewidth=0.5, edgecolor='black', dodge=False,
    legend=False, zorder=1
)
for i, st in enumerate(STATES):
    col = STATE_PALETTE[st]
    ax2.boxplot(
        df_dwell.loc[df_dwell['State'] == st, 'Run_Length'], positions=[i], widths=0.5,
        showfliers=False, patch_artist=True, manage_ticks=False, zorder=3,
        boxprops=dict(facecolor='none', edgecolor=col, linewidth=2),
        medianprops=dict(color=col, linewidth=2),
        whiskerprops=dict(color='black', linewidth=2),
        capprops=dict(color='black', linewidth=2),
    )
ax2.set_title('State Persistence (Dwell Times)', fontweight='bold', fontsize=12, pad=12)
ax2.set_xlabel('Inferred State', fontweight='bold', fontsize=10)
ax2.set_ylabel('Consecutive Runs (Number of Trials)', fontweight='bold', fontsize=10)
ax2.set_xticks(range(len(STATES)))
ax2.set_xticklabels(STATE_LABELS)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
ax2.set_ylim(0, df_dwell['Run_Length'].max() * 1.15)   # headroom for n labels

ymax = ax2.get_ylim()[1]
for idx, s_num in enumerate([1, 2, 3]):
    ax2.text(idx, ymax, f"n={len(dwell_times[s_num])} blocks\nmed={np.median(dwell_times[s_num]):g} trials",
             ha='center', va='top', fontsize=8, color='gray', fontweight='bold')

plt.tight_layout()
stem = os.path.join(OUTPUT_DIR, f'SuppFig_Occupancy_Dwell_gap{MAX_GAP}')
plt.savefig(stem + '.svg', format='svg', bbox_inches='tight')
plt.savefig(stem + '.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved {stem}.svg / .png, occupancy_per_rat.csv and dwell_runs_gap{MAX_GAP}.csv")
