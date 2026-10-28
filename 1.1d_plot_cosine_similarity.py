"""
1.1d_plot_cosine_similarity.py
------------------------------
Diagnostic (not in the paper): plots the GLM-weight cosine similarity matrices
produced by 1.1c_state_count_diagnostics.py.

Each panel shows how the states of a larger model (4- or 5-state; rows) map onto
the retained 3-state solution (columns). Cells are annotated with the cosine
similarity between the two states' GLM weight vectors:
    1.0  = identical strategy
    0.0  = unrelated
   -1.0  = opposite

Matched pairs (the optimal assignment) are outlined in black. Rows corresponding
to unmatched "extra" states are labelled in red - these are the states that
appear only in the larger model.

PREREQUISITE: run 1.1c_state_count_diagnostics.py first. It writes
    HMM_Outputs/StateCount_Diagnostic_{K}state_cosine.csv
    HMM_Outputs/StateCount_Diagnostic_{K}state_summary.csv
1.1c must be run against the CURRENT 3-state pkl. If the 3-state model has been
refit since 1.1c was last run, re-run 1.1c before plotting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- Illustrator-editable vector text (matches the rest of the pipeline) ---
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

# =========================================================
# CONFIGURATION
# =========================================================
OUTPUT_DIR = "HMM_Outputs"
TEST_STATES = [4, 5]          # which models to plot (must match 1.1c)
REFERENCE_STATES = 3

# Readable names for the reference (retained) states. Order = State_1..State_3.
REF_LABELS = ['Risk-Taking\n(State 1)',
              'Risk-Averse\n(State 2)',
              'Repeating\n(State 3)']

# Colour scale. Cosine is bounded [-1, 1]; a diverging map centred on 0 keeps
# negative similarities visually distinct from weak positive ones.
CMAP = 'RdBu_r'
VMIN, VMAX = -1.0, 1.0

# Cells at or above this are called a "match" in the printed summary.
MATCH_HIGHLIGHT = 0.90


# =========================================================
# PLOTTING
# =========================================================
def plot_cosine_panel(ax, cos_df, summary_df, K):
    """Draw one annotated cosine-similarity heatmap."""
    M = cos_df.values                      # (K_test, K_ref)
    n_test, n_ref = M.shape

    im = ax.imshow(M, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect='auto')

    # Which test states are the unmatched "extra" ones?
    extras = set(summary_df.loc[summary_df['Is_EXTRA_state'] == True, 'State'].astype(int))

    # Optimal assignment: for each reference state, the matched test state.
    # Recovered from the summary table (Best_match_ref_state on non-extra rows).
    matched_cells = set()
    for _, row in summary_df.iterrows():
        s = int(row['State'])
        if s not in extras:
            matched_cells.add((s - 1, int(row['Best_match_ref_state']) - 1))

    # Annotate every cell; outline matched pairs.
    for i in range(n_test):
        for j in range(n_ref):
            val = M[i, j]
            # White text on saturated cells, black on pale ones.
            txt_col = 'white' if abs(val) > 0.6 else 'black'
            weight = 'bold' if (i, j) in matched_cells else 'normal'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color=txt_col, fontsize=10, fontweight=weight)
            if (i, j) in matched_cells:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor='black', linewidth=2.5))

    ax.set_xticks(np.arange(n_ref))
    ax.set_xticklabels(REF_LABELS[:n_ref], fontsize=9)
    ax.set_yticks(np.arange(n_test))

    # Red labels flag the extra states.
    ylabels = [f'State {i+1}' for i in range(n_test)]
    ax.set_yticklabels(ylabels, fontsize=10)
    for i, lab in enumerate(ax.get_yticklabels()):
        if (i + 1) in extras:
            lab.set_color('#cc2b2b')
            lab.set_fontweight('bold')

    ax.set_xlabel(f'{REFERENCE_STATES}-state model (retained)',
                  fontsize=10, fontweight='bold')
    ax.set_ylabel(f'{K}-state model', fontsize=10, fontweight='bold')
    ax.set_title(f'{K}-state solution vs retained model',
                 fontsize=11, fontweight='bold', pad=10)

    # Thin gridlines between cells
    ax.set_xticks(np.arange(-0.5, n_ref, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_test, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)

    return im, extras


def main():
    panels = []
    for K in TEST_STATES:
        cos_path = os.path.join(OUTPUT_DIR, f'StateCount_Diagnostic_{K}state_cosine.csv')
        sum_path = os.path.join(OUTPUT_DIR, f'StateCount_Diagnostic_{K}state_summary.csv')
        if not (os.path.exists(cos_path) and os.path.exists(sum_path)):
            print(f"[skip] missing output for K={K}. Run 1.1c first.")
            continue
        cos_df = pd.read_csv(cos_path, index_col=0)
        sum_df = pd.read_csv(sum_path)
        panels.append((K, cos_df, sum_df))

    if not panels:
        raise SystemExit("No 1.1c cosine output found in "
                         f"'{OUTPUT_DIR}'. Run 1.1c_state_count_diagnostics.py first.")

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))
    if len(panels) == 1:
        axes = [axes]

    im = None
    for ax, (K, cos_df, sum_df) in zip(axes, panels):
        im, extras = plot_cosine_panel(ax, cos_df, sum_df, K)

        # Console summary so the values can be checked against the manuscript.
        print(f"\n--- {K}-state model: cosine similarity to {REFERENCE_STATES}-state solution ---")
        print(cos_df.round(3).to_string())
        for _, row in sum_df.iterrows():
            s = int(row['State'])
            tag = "EXTRA" if row['Is_EXTRA_state'] else "matched"
            print(f"  State {s} [{tag}]: nearest reference = State "
                  f"{int(row['Best_match_ref_state'])}, cosine = {row['Best_match_cosine']:.3f}")

    # Shared colourbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
    cbar.set_label('Cosine similarity of GLM weights', fontsize=10, fontweight='bold')
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    out_base = os.path.join(OUTPUT_DIR, 'SuppFig_CosineSimilarity')
    plt.savefig(out_base + '.svg', bbox_inches='tight')
    plt.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved '{out_base}.svg' and '.png'.")
    print("Black outline = optimal assignment; red row label = unmatched 'extra' state.")


if __name__ == "__main__":
    main()
