"""
1.1g_plot_model_comparison_figure.py   (v3: four models x three tests)
=====================================================================
Figure: "Only the state account is consistent with all three features of conflict
behaviour."

Four accounts of how a rat ends up pressing on about half of conflict trials, one
model each (the best-performing version of each family in 1.1e):

  Fixed weighing          GLM static
  Learning from outcomes  Q-learning + Pavlovian bias (configural, fitted per rat)
  Gradual drift           dynamic GLM, random-walk weights (reset each session)
  Switching states        3-state GLM-HMM (the 1.1 model; held-out results from 1.1b)

Layout (180 mm wide, editable SVG text):
  Row 1  A  What each account says: schematic P(press | conflict) over one session
            (illustrative traces, not model output)
  Row 2  Three tests, same four models in the same order and colours as row A
         B  Predicting choices of unseen rats: held-out log-likelihood per trial
            relative to the state model (one point per held-out rat; mean +/- SEM)
         C  Stretches of pressing or withholding: % of 8-conflict-trial blocks that
            were mostly pressed or mostly withheld (<=25% or >=75%); rats = black
            line; each model = mean and 95% interval of 50 simulations
         D  Cues rise and fall together: within-session correlation of conflict
            responding with rewDS and punDS responding (24-trial blocks, mean of the
            two correlations); rats = black line; models = mean and 95% interval

Inputs (HMM_Outputs/, all written by 1.1b / 1.1e --ppc):
  CV_HeldOut_LL_by_State.csv, AltModels_CV_per_rat.csv,
  AltModels_PPC_windows.csv, AltModels_PPC_behaviour.csv
Outputs:
  Fig_ModelComparison.svg / .png
  Fig_ModelComparison_values.csv   every number plotted (for the text and legend)

Usage:
  python 1.1g_plot_model_comparison_figure.py            # reads/writes HMM_Outputs/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
})

INK, MUTED, GRID = '#222222', '#6b6b6b', '#d9d9d9'
WINDOW = 8
HMM3 = '3-state GLM-HMM'

# (short label, panel-A title, one-line description, candidate model labels in order of
#  preference, colour). Colours: validated categorical palette, slots 1-4; every mark is
#  also labelled, so identity never depends on colour alone.
MODELS = [
    ('Fixed',    'Fixed weighing',         'same balance on every trial',
     ['GLM static'], '#eda100'),
    ('Learning', 'Learning from outcomes', 'cue values updated after each outcome',
     ['Q-learning + Pavlovian (configural, per rat)', 'Q-learning + Pavlovian (configural)',
      'Q-learning + Pavlovian (elemental, per rat)', 'Q-learning + Pavlovian (elemental)'], '#eb6834'),
    ('Drift',    'Gradual drift',          'willingness to press wanders slowly',
     ['GLM drift (session reset)', 'GLM drift AR(1) (session reset)'], '#1baf7a'),
    ('States',   'Switching states',       'jumps between distinct modes',
     [HMM3], '#2a78d6'),
]


def resolve(sources):
    """For each account, the first candidate model present in every data source."""
    out = []
    for short, title, desc, cands, col in MODELS:
        hit = next((c for c in cands if all(c in s for s in sources)), None)
        if hit is None:
            raise SystemExit(f"No model for '{title}' found in all inputs (tried {cands}). "
                             "Run 1.1e with --ppc for it first.")
        out.append(dict(short=short, title=title, desc=desc, src=hit, col=col))
    return out


def panel_letter(ax, letter, x=-0.30, y=1.12):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')


def model_axis(ax, models):
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m['short'] for m in models])
    for t, m in zip(ax.get_xticklabels(), models):
        t.set_color(INK)
    ax.set_xlim(-0.6, len(models) - 0.4)


# ---------------------------------------------------------------- row A
def schematic_traces(rng, T=100):
    t = np.arange(T)
    fixed = np.full(T, 0.45)
    # learning: a value nudged up by pellets and down by shocks after each press
    v, learned = 0.0, []
    for _ in t:
        learned.append(1 / (1 + np.exp(-(v - 0.2) * 2.2)))
        if rng.random() < learned[-1]:
            v += 0.25 * ((1.0 if rng.random() < 0.5 else -1.3) - v)
    learned = np.array(learned)
    walk = np.cumsum(rng.normal(0, 0.22, T)); walk -= walk.mean()
    drift = 1 / (1 + np.exp(-(walk - 0.2)))
    states = np.zeros(T, int); states[12:50] = 1; states[63:88] = 1
    discrete = np.where(states == 1, 0.08, 0.72)
    return t, [fixed, learned, drift, discrete]


def panel_A(axes, models, rng):
    t, traces = schematic_traces(rng)
    for i, (ax, m, y) in enumerate(zip(axes, models, traces)):
        ax.plot(t, y, color=m['col'], lw=1.6, solid_capstyle='round',
                drawstyle='steps-post' if i == 3 else 'default')
        ax.axhline(0.45, color=GRID, lw=0.6, ls=(0, (2, 2)), zorder=0)
        ax.set_ylim(-0.02, 1.02); ax.set_xlim(0, t[-1])
        ax.set_xticks([]); ax.set_yticks([0, 0.5, 1])
        ax.set_title(m['title'], color=INK, pad=11, fontweight='bold')
        ax.text(0.5, 1.03, m['desc'], transform=ax.transAxes, ha='center', va='bottom',
                fontsize=6, color=MUTED)
        ax.set_xlabel('Trials within a session', color=MUTED, labelpad=2)
        if i == 0:
            ax.set_ylabel('P(press | conflict)')
        else:
            ax.set_yticklabels([])
    panel_letter(axes[0], 'A', x=-0.36, y=1.16)
    axes[3].text(0.02, 0.77, 'risk-taking', transform=axes[3].transAxes, fontsize=6, color=MUTED)
    axes[3].text(0.15, 0.13, 'risk-averse', transform=axes[3].transAxes, fontsize=6, color=MUTED)


# ---------------------------------------------------------------- row B
def panel_B(ax, models, alt, hmm):
    h = hmm[hmm.Num_States == 3].set_index('HeldOut_Rat').HeldOut_LL_per_trial
    a = alt.pivot(index='HeldOut_Rat', columns='Model', values='HeldOut_LL_per_trial')
    a[HMM3] = h
    rng = np.random.default_rng(3)
    rows = []
    for i, m in enumerate(models):
        d = (a[m['src']] - a[HMM3]).dropna()
        if m['src'] == HMM3:
            ax.plot(i, 0, 'o', ms=5, color=m['col'], mec='white', mew=0.6, zorder=4)
            continue
        x = i + rng.uniform(-0.12, 0.12, len(d))
        ax.scatter(x, d.values, s=9, color=m['col'], alpha=0.55, lw=0, zorder=3)
        sem = d.std(ddof=1) / np.sqrt(len(d))
        ax.errorbar(i + 0.3, d.mean(), yerr=sem, fmt='o', ms=4.5, color=m['col'],
                    mec='white', mew=0.5, elinewidth=1.0, capsize=0, zorder=4)
        rows.append(dict(Panel='B', Account=m['short'], Model=m['src'], Mean=d.mean(), SEM=sem,
                         n_rats=len(d), n_better_than_states=int((d > 0).sum())))
    ax.axhline(0, color=models[-1]['col'], lw=0.8, ls=(0, (3, 2)), zorder=1)
    model_axis(ax, models)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, max(hi, 0) + 0.12 * (max(hi, 0) - lo))     # headroom above the reference line
    ax.set_ylabel('Held-out log-likelihood vs states\n(nats/trial; above 0 = better)')
    ax.set_title('Predicting unseen rats', fontweight='bold', pad=4)
    panel_letter(ax, 'B')
    return rows


def _interval_points(ax, models, sims, obs, fmt_obs):
    rows = []
    for i, m in enumerate(models):
        v = sims[m['src']]
        mu, lo, hi = np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)
        ax.plot([i, i], [lo, hi], color=m['col'], lw=1.4, solid_capstyle='round', zorder=2)
        ax.plot(i, mu, 'o', ms=5.5, color=m['col'], mec='white', mew=0.7, zorder=3)
        rows.append(dict(Account=m['short'], Model=m['src'], Mean=mu, Lo95=lo, Hi95=hi,
                         Rats=obs, Rats_inside_95=bool(lo <= obs <= hi), n_sims=len(v)))
    ax.axhline(obs, color=INK, lw=1.0, zorder=1)
    ax.text(-0.55, obs, f' rats {fmt_obs(obs)}', ha='left', va='bottom', fontsize=6, color=INK)
    model_axis(ax, models)
    return rows


# ---------------------------------------------------------------- row C
def panel_C(ax, models, ppc):
    def extreme(r):
        r = np.asarray(r)
        return np.mean((r <= 0.25) | (r >= 0.75))
    obs = extreme(ppc[ppc.Source == 'Observed'].Rate)
    sims = {m['src']: ppc[ppc.Source == m['src']].groupby('Sim').Rate.apply(extreme).values
            for m in models}
    rows = _interval_points(ax, models, sims, obs, lambda v: f'{v * 100:.0f}%')
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel(f'Blocks of {WINDOW} conflict trials\nmostly pressed or mostly withheld')
    ax.set_title('Runs of pressing or withholding', fontweight='bold', pad=4)
    panel_letter(ax, 'C')
    return [dict(Panel='C', **r) for r in rows]


# ---------------------------------------------------------------- row D
def panel_D(ax, models, beh):
    beh = beh.assign(coupling=beh[['r_conflict_rewDS', 'r_conflict_punDS']].mean(axis=1))
    obs = float(beh[beh.Source == 'Observed'].coupling.iloc[0])
    sims = {m['src']: beh[beh.Source == m['src']].coupling.dropna().values for m in models}
    rows = _interval_points(ax, models, sims, obs, lambda v: f'{v:.2f}')
    ax.set_ylim(0, 0.65)
    ax.set_ylabel('Correlation of conflict responding\nwith rewDS and punDS responding')
    ax.set_title('Cues move together', fontweight='bold', pad=4)
    panel_letter(ax, 'D')
    # the two correlations separately, for the legend and text
    for r in rows:
        sub = beh[beh.Source == r['Model']]
        ob = beh[beh.Source == 'Observed'].iloc[0]
        for k in ['r_conflict_rewDS', 'r_conflict_punDS']:
            r[f'{k}_mean'] = sub[k].mean()
            r[f'{k}_lo95'], r[f'{k}_hi95'] = np.percentile(sub[k].dropna(), [2.5, 97.5])
            r[f'{k}_rats'] = ob[k]
    return [dict(Panel='D', **r) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='HMM_Outputs')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or a.dir
    alt = pd.read_csv(os.path.join(a.dir, 'AltModels_CV_per_rat.csv'))
    hmm = pd.read_csv(os.path.join(a.dir, 'CV_HeldOut_LL_by_State.csv'))
    ppc = pd.read_csv(os.path.join(a.dir, 'AltModels_PPC_windows.csv'))
    beh = pd.read_csv(os.path.join(a.dir, 'AltModels_PPC_behaviour.csv'))

    models = resolve([set(alt.Model) | {HMM3}, set(ppc.Source), set(beh.Source)])
    for m in models:
        print(f"  {m['title']:<24s} -> {m['src']}")

    fig = plt.figure(figsize=(180 / 25.4, 118 / 25.4))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.8, 1.0], hspace=0.55,
                  left=0.105, right=0.98, top=0.90, bottom=0.07)
    gA = gs[0].subgridspec(1, 4, wspace=0.18)
    panel_A([fig.add_subplot(gA[i]) for i in range(4)], models, np.random.default_rng(7))
    gB = gs[1].subgridspec(1, 3, wspace=0.62)
    rows = panel_B(fig.add_subplot(gB[0]), models, alt, hmm)
    rows += panel_C(fig.add_subplot(gB[1]), models, ppc)
    rows += panel_D(fig.add_subplot(gB[2]), models, beh)

    for ext in ['svg', 'png']:
        fig.savefig(os.path.join(out, f'Fig_ModelComparison.{ext}'), dpi=300)
    pd.DataFrame(rows).to_csv(os.path.join(out, 'Fig_ModelComparison_values.csv'), index=False)
    print(f"Saved Fig_ModelComparison.svg/.png and Fig_ModelComparison_values.csv to {out}/")


if __name__ == '__main__':
    main()