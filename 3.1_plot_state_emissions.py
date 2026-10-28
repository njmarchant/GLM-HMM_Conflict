"""
3.1_plot_state_emissions.py
---------------------------
Per-rat response profile of each latent state (Fig. 3D): for each rat and state, the
probability of pressing on rewDS, punDS and conflict trials (assigned trials only), and
the probability of repeating the previous choice of the session.

Input   Data/collated_2026_conflict_data_with_Probabilities_065_Past_PelletShock_v5.csv (from 2.1)
Output  HMM_Outputs/Emissions/
          state_emissions_per_rat.csv          rat x state table behind every panel
          interactive_3d_states.html           source of Fig. 3D
          3d_state_emissions.pdf               static 3D view
          pure_vector_behavior_matrix.{svg,pdf} pairwise matrix
          ssm_rew_vs_pun.pdf, ssm_conf_vs_repeating.pdf   2D slices
          powerpoint_3d_states.glb             3D model for PowerPoint (needs trimesh)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
sns.set_theme(style="ticks")

# =========================================================
# 0. CONFIGURATION & COLOURS
# =========================================================
INPUT_PATH = os.path.join("Data", "collated_2026_conflict_data_with_Probabilities_065_Past_PelletShock_v5.csv")
OUT = os.path.join("HMM_Outputs", "Emissions")

PALETTE = {"State 1": "#B0B300",   # Risk-taking
           "State 2": "#B500B3",   # Risk-averse
           "State 3": "#00B0B3"}   # Repeating
STATE_COLORS = {1: "#B0B300", 2: "#B500B3", 3: "#00B0B3"}


def hex_to_rgba(hex_color):
    return [int(c * 255) for c in mcolors.to_rgb(hex_color)] + [255]


RGBA = {k: hex_to_rgba(v) for k, v in PALETTE.items()}

# =========================================================
# 1. LOAD DATA & CALCULATE PROBABILITIES
# =========================================================
if not os.path.exists(INPUT_PATH):
    sys.exit(f"Error: {INPUT_PATH} not found. Run 2.1_States_to_trials.py first.")
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(INPUT_PATH, dtype={"Session": str})
df = df.sort_values(['Rat', 'Session', 'Trial']).copy()
df['Lever_Press'] = df['Outcome'].astype(int)

# Repeated choice: same choice as the PREVIOUS TRIAL OF THE SESSION (computed before
# unassigned trials are dropped; NaN on the first trial of each session)
prev = df.groupby(['Rat', 'Session'])['Lever_Press'].shift(1)
df['Repeated_Choice'] = np.where(prev.isna(), np.nan, (df['Lever_Press'] == prev).astype(float))

df = df.dropna(subset=['Inferred_State']).copy()      # assigned trials only
states = sorted(df['Inferred_State'].unique())

print("Calculating subject-level state emissions...")
keys = ['Rat', 'Inferred_State']
rew_df = df[df['Cue'] == 'reward'].groupby(keys)['Lever_Press'].mean().reset_index(name='P_Rew')
pun_df = df[df['Cue'] == 'punish'].groupby(keys)['Lever_Press'].mean().reset_index(name='P_Pun')
conf_df = df[df['Cue'] == 'conflict'].groupby(keys)['Lever_Press'].mean().reset_index(name='P_Conf')
stick_df = df.groupby(keys)['Repeated_Choice'].mean().reset_index(name='P_Sticky')
n_df = df.groupby(keys).size().reset_index(name='N_trials')

merged_df = rew_df.merge(pun_df, on=keys, how='inner').merge(conf_df, on=keys, how='inner')
merged_df = merged_df.merge(stick_df, on=keys, how='left').merge(n_df, on=keys, how='left')
merged_df['Inferred_State_Label'] = "State " + merged_df['Inferred_State'].astype(int).astype(str)
merged_df = merged_df.sort_values('Inferred_State_Label')
merged_df.to_csv(os.path.join(OUT, 'state_emissions_per_rat.csv'), index=False)
print(merged_df.groupby('Inferred_State_Label')[['P_Rew', 'P_Pun', 'P_Conf', 'P_Sticky']].mean().round(3))
if merged_df.empty:
    sys.exit("No rat has assigned trials of every cue type in any state; nothing to plot.")

# =========================================================
# 2. 2D SLICE PLOTS
# =========================================================
print("Generating 2D slice plots...")
plt.figure(figsize=(6, 5))
for state in states:
    d = merged_df[merged_df['Inferred_State'] == state]
    plt.scatter(d['P_Pun'], d['P_Rew'], c=STATE_COLORS.get(int(state), 'k'),
                label=f'State {int(state)}', alpha=0.7, s=50, edgecolor='white')
plt.title("State Emissions: P(Press) for Reward vs. Punishment")
plt.xlabel("P(Press | Punish)"); plt.ylabel("P(Press | Reward)")
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
plt.xlim(-0.05, 1.05); plt.ylim(-0.05, 1.05)
plt.legend(title="Hidden State", frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "ssm_rew_vs_pun.pdf"), format="pdf")
plt.close()

plt.figure(figsize=(6, 5))
for state in states:
    d = merged_df[merged_df['Inferred_State'] == state]
    plt.scatter(d['P_Sticky'], d['P_Conf'], c=STATE_COLORS.get(int(state), 'k'),
                label=f'State {int(state)}', alpha=0.7, s=50, edgecolor='white')
plt.title("State Emissions: Stickiness vs. Conflict Choice Bias")
plt.xlabel("P(Choice = Previous Choice | State k)"); plt.ylabel("P(Press | Conflict)")
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
plt.xlim(0.3, 1.05); plt.ylim(-0.05, 1.05)
plt.legend(title="Hidden State", frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "ssm_conf_vs_repeating.pdf"), format="pdf")
plt.close()

# =========================================================
# 3. PAIRWISE MATRIX
# =========================================================
print("Generating pairwise matrix...")
labels_present = sorted(merged_df['Inferred_State_Label'].unique())
g = sns.pairplot(
    merged_df, vars=['P_Pun', 'P_Rew', 'P_Conf'], hue='Inferred_State_Label',
    hue_order=labels_present, palette=PALETTE,
    markers=["o", "s", "D"][:len(labels_present)],
    plot_kws={'alpha': 0.7, 's': 50, 'edgecolor': 'none'},
    diag_kws={'alpha': 0.4, 'fill': True})
for ax in g.axes.flatten():
    if ax is not None:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axis_labels = {"P_Pun": "P(Press | Punish)", "P_Rew": "P(Press | Reward)", "P_Conf": "P(Press | Conflict)"}
for i, var in enumerate(['P_Pun', 'P_Rew', 'P_Conf']):
    g.axes[2, i].set_xlabel(axis_labels[var], fontsize=10)
    g.axes[i, 0].set_ylabel(axis_labels[var], fontsize=10)
plt.suptitle("Behavioral Phenotypes Pairwise Matrix", y=1.02, fontsize=12)
plt.savefig(os.path.join(OUT, "pure_vector_behavior_matrix.svg"), format="svg", bbox_inches='tight', transparent=True)
plt.savefig(os.path.join(OUT, "pure_vector_behavior_matrix.pdf"), format="pdf", bbox_inches='tight', transparent=True)
plt.close()

# =========================================================
# 4. STATIC 3D PLOT
# =========================================================
print("Generating static 3D plot...")
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for state in states:
    d = merged_df[merged_df['Inferred_State'] == state]
    ax.scatter(d['P_Pun'], d['P_Rew'], d['P_Conf'], c=STATE_COLORS.get(int(state), 'k'),
               label=f'State {int(state)}', alpha=0.7, s=60)
ax.set_title("3D State Emissions: Propensity to Press by Cue")
ax.set_xlabel("P(Press | Punish)"); ax.set_ylabel("P(Press | Reward)"); ax.set_zlabel("P(Press | Conflict)")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])
plt.legend(title="Hidden State", frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "3d_state_emissions.pdf"), format="pdf")
plt.close(fig)

# =========================================================
# 5. INTERACTIVE 3D (source of Fig. 3D)
# =========================================================
try:
    import plotly.express as px
    print("Generating interactive 3D HTML...")
    plotly_colors = {k: f"rgb({v[0]},{v[1]},{v[2]})" for k, v in RGBA.items()}
    fig = px.scatter_3d(
        merged_df, x='P_Pun', y='P_Rew', z='P_Conf', color='Inferred_State_Label',
        color_discrete_map=plotly_colors,
        title="3D Behavioral Phenotypes Across Hidden States",
        labels={"P_Pun": "P(Press | Punish)", "P_Rew": "P(Press | Reward)", "P_Conf": "P(Press | Conflict)"},
        opacity=0.8)
    fig.update_layout(scene=dict(xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), zaxis=dict(range=[0, 1])))
    fig.write_html(os.path.join(OUT, "interactive_3d_states.html"))
except ImportError:
    print("plotly not installed: interactive_3d_states.html (Fig. 3D) skipped. pip install plotly")

# =========================================================
# 6. 3D MODEL FOR POWERPOINT (.glb)
# =========================================================
try:
    import trimesh
    print("Generating 3D model (.glb)...")
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.axis(origin_size=0.02, axis_radius=0.005, axis_length=1.0))
    for _, row in merged_df.iterrows():
        sphere = trimesh.creation.icosphere(radius=0.02)
        sphere.apply_translation([row['P_Pun'], row['P_Rew'], row['P_Conf']])
        sphere.visual.vertex_colors = RGBA.get(row['Inferred_State_Label'], [128, 128, 128, 255])
        scene.add_geometry(sphere)
    scene.export(os.path.join(OUT, "powerpoint_3d_states.glb"))
except ImportError:
    print("trimesh not installed: powerpoint_3d_states.glb skipped (optional).")

print(f"Done. Outputs in {OUT}/")
