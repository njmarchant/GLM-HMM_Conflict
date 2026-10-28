"""
2.4_state_dynamics_shock.py
---------------------------
Do rats alternate between latent states WITHIN sessions, and do the states simply
track the experimenter-set shock intensity?

ANALYSES
  1. Within-session switching (session level)
       - confident runs of each state (posterior >= 0.65), bridging gaps of up to
         MAX_GAP unassigned trials, exactly as in Supplementary Fig. 3B
       - a switch is counted between consecutive runs of different states, each
         at least MIN_RUN trials long (shorter runs are treated as flicker)
       - % sessions with >= 1 switch; % sessions containing both on-task states;
         number of rats with >= 1 session containing both on-task states
       - switch types: direct S1<->S2 vs. routed through S3 (S1->S3->S2, S2->S3->S1)
  2. Variance decomposition of trial-level state membership into
       between-rat / between-session (within rat) / within-session components.
       Shock intensity is constant within a session, so the within-session
       component is variance that intensity cannot explain.
  3. Occupancy across Phase 3 sessions (session number 1..N within each rat)
       LMM: occupancy ~ session_c + (1 | Rat)
  4. Occupancy vs shock intensity
       LMM: occupancy ~ intensity_within + intensity_between + session_c + (1 | Rat)
         intensity_within  = session intensity minus the rat's mean intensity
                             (the titration effect, tested within rat)
         intensity_between = the rat's mean intensity (individual differences)
         session_c         = session number, centred (controls for training time,
                             which is confounded with titration)
       Robustness: session-to-session change model
         d_occupancy ~ d_intensity + (1 | Rat)

INPUT   Data/collated_2026_conflict_data_with_Probabilities_065_Past_PelletShock_v5.csv (from 2.1)
OUTPUTS (HMM_Outputs/StateDynamics/)
  session_table.csv, switch_types.csv, variance_decomposition.csv,
  occupancy_by_session_number.csv, lmm_results.csv, SUMMARY.txt,
  Fig_StateDynamics.svg / .png  (5 panels; Supplementary Fig. 3C-G)
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import statsmodels.formula.api as smf

plt.rcParams['svg.fonttype'] = 'none'   # Illustrator-editable text
plt.rcParams['pdf.fonttype'] = 42
warnings.filterwarnings('ignore', category=UserWarning)          # statsmodels convergence chatter
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =========================================================
# 0. CONFIGURATION
# =========================================================
DATA_FILE        = "Data/collated_2026_conflict_data_with_Probabilities_065_Past_PelletShock_v5.csv"
OUTPUT_DIR       = os.path.join("HMM_Outputs", "StateDynamics")
STATE_THRESHOLD  = 0.65   # same as 2.1 / 2.3
MAX_GAP          = 3      # bridge up to 3 unassigned trials (same as Supp. Fig. 3B)
MIN_RUN          = 3      # runs shorter than this are not counted as a visit/switch
MIN_TRIALS_STATE = 5      # a session "contains" a state if >= this many assigned trials
MIN_ASSIGNED     = 10     # sessions with fewer assigned trials are dropped from occupancy models
INTENSITY_COL    = "Intensity"
STATES           = [1, 2, 3]
STATE_NAMES      = {1: 'Risk-taking', 2: 'Risk-averse', 3: 'Repeating'}
STATE_COLORS     = {1: '#B0B300', 2: '#B500B3', 3: '#00B0B3'}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 1. LOAD + STATE ASSIGNMENT
# =========================================================
def natural_key(x):
    """Sort '3.2' before '3.10', and numbers numerically."""
    s = str(x)
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


print(f"Loading {DATA_FILE} ...")
df = pd.read_csv(DATA_FILE, dtype={"Session": str})   # keep "3.10" distinct from "3.1"
prob_cols = ['Prob_State_1', 'Prob_State_2', 'Prob_State_3']
df = df.dropna(subset=prob_cols).copy()

# Chronological order within each rat
df['_sess_key'] = df['Session'].map(lambda s: tuple(natural_key(s)))
df = df.sort_values(['Rat', '_sess_key', 'Trial']).reset_index(drop=True)

df['MaxProb'] = df[prob_cols].max(axis=1)
df['MAP_State'] = df[prob_cols].values.argmax(axis=1) + 1
df['State'] = np.where(df['MaxProb'] >= STATE_THRESHOLD, df['MAP_State'], np.nan)

# Session number (1..N) within each rat
sess_order = (df[['Rat', 'Session', '_sess_key']].drop_duplicates()
              .sort_values(['Rat', '_sess_key']))
sess_order['Session_N'] = sess_order.groupby('Rat').cumcount() + 1
df = df.merge(sess_order[['Rat', 'Session', 'Session_N']], on=['Rat', 'Session'])

has_intensity = INTENSITY_COL in df.columns and df[INTENSITY_COL].notna().any()
if not has_intensity:
    print(f"WARNING: column '{INTENSITY_COL}' not found - shock analyses will be skipped.")

n_total = len(df)
n_unassigned = df['State'].isna().sum()
print(f"{df['Rat'].nunique()} rats, {df.groupby(['Rat','Session']).ngroups} sessions, "
      f"{n_total} trials; unassigned at {STATE_THRESHOLD}: {n_unassigned} "
      f"({100*n_unassigned/n_total:.1f}%)")


# =========================================================
# 2. RUNS WITH GAP BRIDGING (identical rule to Supp. Fig. 3B)
# =========================================================
def bridged_runs(states, max_gap=MAX_GAP):
    """
    Returns a list of (state, start_idx, end_idx, length) for confident runs.
    Gaps of <= max_gap unassigned trials are bridged when the same state resumes
    on both sides; bridged trials count towards the run length.
    """
    runs = []
    cur_state, start, last_assigned, gap = None, None, None, 0
    for i, s in enumerate(states):
        if np.isnan(s):
            if cur_state is not None:
                gap += 1
                if gap > max_gap:
                    runs.append((cur_state, start, last_assigned, last_assigned - start + 1))
                    cur_state, start, gap = None, None, 0
            continue
        s = int(s)
        if cur_state is None:
            cur_state, start, last_assigned, gap = s, i, i, 0
        elif s == cur_state:
            last_assigned, gap = i, 0
        else:
            runs.append((cur_state, start, last_assigned, last_assigned - start + 1))
            cur_state, start, last_assigned, gap = s, i, i, 0
    if cur_state is not None:
        runs.append((cur_state, start, last_assigned, last_assigned - start + 1))
    return runs


# =========================================================
# 3. SESSION TABLE + SWITCH COUNTS
# =========================================================
session_rows, switch_rows = [], []
for (rat, sess), g in df.groupby(['Rat', 'Session'], sort=False):
    st = g['State'].values.astype(float)
    n_assigned = int(np.sum(~np.isnan(st)))
    runs = [r for r in bridged_runs(st) if r[3] >= MIN_RUN]

    # merge consecutive runs of the same state (separated only by a dropped short run / long gap)
    seq = []
    for r in runs:
        if seq and seq[-1] == r[0]:
            continue
        seq.append(r[0])
    switches = list(zip(seq[:-1], seq[1:]))
    for a, b in switches:
        switch_rows.append({'Rat': rat, 'Session': sess, 'From': a, 'To': b})
    # routed S1->S3->S2 and S2->S3->S1 sequences
    via3 = sum(1 for i in range(len(seq) - 2)
               if seq[i + 1] == 3 and {seq[i], seq[i + 2]} == {1, 2})

    row = {'Rat': rat, 'Session': sess, 'Session_N': int(g['Session_N'].iloc[0]),
           'N_trials': len(g), 'N_assigned': n_assigned,
           'N_switches': len(switches),
           'N_direct_S1_S2': sum(1 for a, b in switches if {a, b} == {1, 2}),
           'N_via_S3_S1_S2': via3,
           'Sequence': '-'.join(map(str, seq))}
    if has_intensity:
        row['Intensity'] = float(g[INTENSITY_COL].iloc[0])
    for k in STATES:
        cnt = int(np.sum(st == k))
        row[f'Count_S{k}'] = cnt
        row[f'Occ_S{k}'] = 100 * cnt / n_assigned if n_assigned else np.nan          # % of assigned
        row[f'PostOcc_S{k}'] = 100 * g[f'Prob_State_{k}'].mean()                      # threshold-free
        row[f'Has_S{k}'] = cnt >= MIN_TRIALS_STATE
    row['Has_S1_and_S2'] = row['Has_S1'] and row['Has_S2']
    row['N_states_visited'] = sum(row[f'Has_S{k}'] for k in STATES)
    session_rows.append(row)

sess_df = pd.DataFrame(session_rows)
sw_df = pd.DataFrame(switch_rows, columns=['Rat', 'Session', 'From', 'To'])
sess_df.to_csv(os.path.join(OUTPUT_DIR, 'session_table.csv'), index=False)

switch_types = (sw_df.groupby(['From', 'To']).size().rename('N').reset_index()
                if len(sw_df) else pd.DataFrame(columns=['From', 'To', 'N']))
switch_types.to_csv(os.path.join(OUTPUT_DIR, 'switch_types.csv'), index=False)

# Sensitivity of switch counts to the minimum run length (flicker control)
def state_sequence(st, min_run):
    seq = []
    for r in bridged_runs(st):
        if r[3] >= min_run and not (seq and seq[-1] == r[0]):
            seq.append(r[0])
    return seq

sens_rows = []
for mr in [1, 3, 5, 10]:
    n_sw = [len(state_sequence(g['State'].values.astype(float), mr)) - 1
            for _, g in df.groupby(['Rat', 'Session'], sort=False)]
    n_sw = np.clip(n_sw, 0, None)
    sens_rows.append({'Min_run': mr, 'Pct_sessions_with_switch': 100 * np.mean(np.array(n_sw) > 0),
                      'Median_switches': float(np.median(n_sw))})
sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(os.path.join(OUTPUT_DIR, 'switch_sensitivity_min_run.csv'), index=False)

# Rat-level summaries
rat_sw = sess_df.groupby('Rat').agg(
    sessions=('Session', 'size'),
    sessions_with_switch=('N_switches', lambda x: int((x > 0).sum())),
    sessions_with_S1_and_S2=('Has_S1_and_S2', 'sum'),
    mean_switches=('N_switches', 'mean')).reset_index()
rat_sw.to_csv(os.path.join(OUTPUT_DIR, 'rat_switch_summary.csv'), index=False)


# =========================================================
# 4. VARIANCE DECOMPOSITION (rat / session-within-rat / within-session)
# =========================================================
assigned = df.dropna(subset=['State']).copy()
vd_rows = []
for k in STATES:
    y = (assigned['State'] == k).astype(float)
    grand = y.mean()
    rat_mean = y.groupby(assigned['Rat']).transform('mean')
    ses_mean = y.groupby([assigned['Rat'], assigned['Session']]).transform('mean')
    ss_tot = ((y - grand) ** 2).sum()
    ss_rat = ((rat_mean - grand) ** 2).sum()
    ss_ses = ((ses_mean - rat_mean) ** 2).sum()
    ss_wit = ((y - ses_mean) ** 2).sum()
    vd_rows.append({'State': k, 'Name': STATE_NAMES[k],
                    'Between_rat_%': 100 * ss_rat / ss_tot,
                    'Between_session_within_rat_%': 100 * ss_ses / ss_tot,
                    'Within_session_%': 100 * ss_wit / ss_tot})
vd_df = pd.DataFrame(vd_rows)
vd_df.to_csv(os.path.join(OUTPUT_DIR, 'variance_decomposition.csv'), index=False)


# =========================================================
# 5. OCCUPANCY ACROSS SESSIONS + SHOCK INTENSITY (LMMs)
# =========================================================
occ = sess_df[sess_df['N_assigned'] >= MIN_ASSIGNED].copy()
occ['session_c'] = occ['Session_N'] - occ['Session_N'].mean()

by_session = (occ.melt(id_vars=['Rat', 'Session_N'],
                       value_vars=[f'Occ_S{k}' for k in STATES],
                       var_name='State', value_name='Occupancy')
              .groupby(['State', 'Session_N'])['Occupancy']
              .agg(['mean', 'sem', 'count']).reset_index())
by_session.to_csv(os.path.join(OUTPUT_DIR, 'occupancy_by_session_number.csv'), index=False)


def fit_lmm(formula, data, term):
    """Random-intercept LMM (REML). If the optimiser fails or the random-effect
    variance is on the boundary (NaN SEs), try other optimisers, then fall back
    to OLS with rat-clustered standard errors. The method used is reported."""
    data = data.dropna(subset=[c for c in data.columns if c in formula]).copy()
    out = {'n_obs': len(data), 'n_rats': data['Rat'].nunique()}
    for meth in ['lbfgs', 'powell', 'nm']:
        try:
            res = smf.mixedlm(formula, data, groups=data['Rat']).fit(reml=True, method=meth)
            if np.isfinite(res.bse[term]):
                ci = res.conf_int().loc[term]
                return {'beta': res.params[term], 'se': res.bse[term], 'z': res.tvalues[term],
                        'p': res.pvalues[term], 'ci_lo': ci[0], 'ci_hi': ci[1],
                        'method': f'LMM ({meth})', **out}
        except Exception:
            pass
    res = smf.ols(formula, data).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(data['Rat'])[0]})
    ci = res.conf_int().loc[term]
    print(f"  NOTE: LMM did not converge for '{formula}'; used OLS with rat-clustered SEs.")
    return {'beta': res.params[term], 'se': res.bse[term], 'z': res.tvalues[term], 'p': res.pvalues[term],
            'ci_lo': ci[0], 'ci_hi': ci[1], 'method': 'OLS, rat-clustered SE', **out}


lmm_rows = []
for k in STATES:
    r = fit_lmm(f'Occ_S{k} ~ session_c', occ, 'session_c')
    lmm_rows.append({'State': k, 'Model': 'Occupancy ~ session number', 'Term': 'session (per session)', **r})

shock_ok = False
if has_intensity:
    occ['int_between'] = occ.groupby('Rat')['Intensity'].transform('mean')
    occ['int_within'] = occ['Intensity'] - occ['int_between']
    n_rats_varying = int((occ.groupby('Rat')['Intensity'].nunique() > 1).sum())
    within_r = (occ.groupby('Rat')[['Intensity', 'Session_N']]
                .apply(lambda g: g['Intensity'].corr(g['Session_N']) if g['Intensity'].nunique() > 1 else np.nan)
                .dropna())
    shock_ok = n_rats_varying >= 3
    if shock_ok:
        # report per 0.05 mA, the typical titration step
        occ['int_within_005'] = occ['int_within'] / 0.05
        occ['int_between_005'] = (occ['int_between'] - occ['int_between'].mean()) / 0.05
        for k in STATES:
            f = f'Occ_S{k} ~ int_within_005 + int_between_005 + session_c'
            for term, lab in [('int_within_005', 'intensity, within rat (per 0.05 mA)'),
                              ('int_between_005', 'intensity, between rats (per 0.05 mA)'),
                              ('session_c', 'session (per session), adjusted')]:
                lmm_rows.append({'State': k, 'Model': 'Occupancy ~ intensity + session', 'Term': lab,
                                 **fit_lmm(f, occ, term)})

        # Robustness: consecutive-session changes
        ch = occ.sort_values(['Rat', 'Session_N']).copy()
        for k in STATES:
            ch[f'd_Occ_S{k}'] = ch.groupby('Rat')[f'Occ_S{k}'].diff()
        ch['d_int_005'] = ch.groupby('Rat')['Intensity'].diff() / 0.05
        ch = ch.dropna(subset=['d_int_005'])
        for k in STATES:
            lmm_rows.append({'State': k, 'Model': 'Change model', 'Term': 'd intensity (per 0.05 mA)',
                             **fit_lmm(f'd_Occ_S{k} ~ d_int_005', ch, 'd_int_005')})
        ch['Direction'] = np.sign(ch['d_int_005']).map({-1: 'decrease', 0: 'unchanged', 1: 'increase'})
        change_summary = (ch.melt(id_vars=['Rat', 'Direction'],
                                  value_vars=[f'd_Occ_S{k}' for k in STATES],
                                  var_name='State', value_name='d_Occ')
                          .groupby(['State', 'Direction'])['d_Occ'].agg(['mean', 'sem', 'count']).reset_index())
        change_summary.to_csv(os.path.join(OUTPUT_DIR, 'occupancy_change_by_intensity_change.csv'), index=False)

lmm_df = pd.DataFrame(lmm_rows)
lmm_df.to_csv(os.path.join(OUTPUT_DIR, 'lmm_results.csv'), index=False)


# =========================================================
# 6. SUMMARY TEXT (numbers in sentence form for the Results)
# =========================================================
def pfmt(p):
    return 'p < 0.0001' if p < 0.0001 else f'p = {p:.3f}' if p < 0.01 else f'p = {p:.2f}'


ns = len(sess_df)
pct_switch = 100 * (sess_df['N_switches'] > 0).mean()
pct_s1s2 = 100 * sess_df['Has_S1_and_S2'].mean()
rats_s1s2 = int((rat_sw['sessions_with_S1_and_S2'] > 0).sum())
n_rats = sess_df['Rat'].nunique()
n_direct = int(sess_df['N_direct_S1_S2'].sum())
n_via3 = int(sess_df['N_via_S3_S1_S2'].sum())
med_sw = sess_df['N_switches'].median()
iqr_sw = sess_df['N_switches'].quantile([.25, .75]).values

L = []
L.append("STATE DYNAMICS AND SHOCK INTENSITY - SUMMARY")
L.append(f"Settings: threshold {STATE_THRESHOLD}, gap bridging <= {MAX_GAP} trials, "
         f"minimum run {MIN_RUN} trials, session 'contains' a state if >= {MIN_TRIALS_STATE} trials.\n")
L.append(f"Trials left unassigned at {STATE_THRESHOLD}: {100*n_unassigned/n_total:.1f}% ({n_unassigned}/{n_total}).")
L.append(f"Sessions: {ns} from {n_rats} rats.")
L.append(f"Sessions with >= 1 state switch: {pct_switch:.1f}%; median switches per session "
         f"{med_sw:.0f} (IQR {iqr_sw[0]:.0f}-{iqr_sw[1]:.0f}).")
L.append(f"Sessions containing both on-task states (S1 and S2): {pct_s1s2:.1f}%; "
         f"rats with >= 1 such session: {rats_s1s2}/{n_rats}.")
for _, r in sens_df.iterrows():
    L.append(f"   sensitivity, minimum run {int(r.Min_run):>2} trials: {r.Pct_sessions_with_switch:.1f}% of sessions switch, median {r.Median_switches:.0f}")
L.append(f"Sessions per rat: min {rat_sw['sessions'].min()}, max {rat_sw['sessions'].max()}; rats with fewer than the maximum: " + ", ".join(rat_sw.loc[rat_sw['sessions'] < rat_sw['sessions'].max(), 'Rat'].astype(str)))
L.append(f"S1<->S2 changes: {n_direct} direct, {n_via3} routed through the Repeating state.")
L.append("Switch counts (From -> To):")
for _, r in switch_types.iterrows():
    L.append(f"   S{int(r.From)} -> S{int(r.To)}: {int(r.N)}")
L.append("\nVariance in trial-level state membership:")
for _, r in vd_df.iterrows():
    L.append(f"   {r.Name}: between rats {r['Between_rat_%']:.1f}%, between sessions (within rat) "
             f"{r['Between_session_within_rat_%']:.1f}%, within sessions {r['Within_session_%']:.1f}%")
L.append("\nOccupancy across Phase 3 sessions (LMM, % per session):")
for _, r in lmm_df[lmm_df['Model'] == 'Occupancy ~ session number'].iterrows():
    L.append(f"   {STATE_NAMES[r.State]}: beta = {r.beta:+.2f}%/session "
             f"(95% CI {r.ci_lo:+.2f} to {r.ci_hi:+.2f}), z = {r.z:.2f}, {pfmt(r.p)}")
if has_intensity:
    L.append(f"\nShock intensity: {n_rats_varying}/{occ['Rat'].nunique()} rats had > 1 intensity in Phase 3; "
             f"range {occ['Intensity'].min():.2f}-{occ['Intensity'].max():.2f} mA.")
    if len(within_r):
        L.append(f"   Within-rat correlation of intensity with session number: median r = "
                 f"{within_r.median():.2f} (range {within_r.min():.2f} to {within_r.max():.2f}).")
    if shock_ok:
        L.append("   Occupancy ~ intensity_within + intensity_between + session (LMM, % per 0.05 mA):")
        for _, r in lmm_df[lmm_df['Model'] == 'Occupancy ~ intensity + session'].iterrows():
            L.append(f"   {STATE_NAMES[r.State]:12s} {r.Term:42s} beta = {r.beta:+.2f} "
                     f"(95% CI {r.ci_lo:+.2f} to {r.ci_hi:+.2f}), z = {r.z:.2f}, {pfmt(r.p)}")
        L.append("   Change model, d_occupancy ~ d_intensity (LMM, % per 0.05 mA):")
        for _, r in lmm_df[lmm_df['Model'] == 'Change model'].iterrows():
            L.append(f"   {STATE_NAMES[r.State]:12s} beta = {r.beta:+.2f} "
                     f"(95% CI {r.ci_lo:+.2f} to {r.ci_hi:+.2f}), z = {r.z:.2f}, {pfmt(r.p)}")
    else:
        L.append("   Too few rats with varying intensity for the within-rat model.")
txt = "\n".join(L)
with open(os.path.join(OUTPUT_DIR, 'SUMMARY.txt'), 'w') as f:
    f.write(txt)
print("\n" + txt)


# =========================================================
# 7. FIGURE (Supplementary Fig. 3C-G)
# =========================================================
fig, ax = plt.subplots(1, 5, figsize=(25, 4.8), gridspec_kw={'width_ratios': [1, 1.4, 1, 1.2, 1.2]})

# C: switches per session
a = ax[0]
mx = int(sess_df['N_switches'].max()) if ns else 0
a.hist(sess_df['N_switches'], bins=np.arange(-0.5, mx + 1.5, 1), color='#777777', edgecolor='black')
a.set_xlabel('State switches per session'); a.set_ylabel('Sessions')
a.xaxis.set_major_locator(MaxNLocator(integer=True))

# D: per rat - sessions containing both on-task states / other switch / no switch
a = ax[1]
r = rat_sw.copy()
r['both'] = r['sessions_with_S1_and_S2']
r['switch_only'] = (sess_df.assign(sw=sess_df['N_switches'] > 0, both=sess_df['Has_S1_and_S2'])
                    .groupby('Rat')[['sw', 'both']].apply(lambda g: int((g['sw'] & ~g['both']).sum()))
                    .reindex(r['Rat']).values)
r['none'] = r['sessions'] - r['both'] - r['switch_only']
r = r.sort_values('both', ascending=False)
x = np.arange(len(r))
a.bar(x, r['both'], color='#333333', edgecolor='black', label='Both on-task states')
a.bar(x, r['switch_only'], bottom=r['both'], color='#888888', edgecolor='black', label='Other switches')
a.bar(x, r['none'], bottom=r['both'] + r['switch_only'], color='white', edgecolor='black', label='No switch')
a.set_xticks(x); a.set_xticklabels(r['Rat'], rotation=90, fontsize=7)
a.set_ylabel('Sessions'); a.set_xlabel('Rat')
a.legend(frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)

# E: variance decomposition
a = ax[2]
comp = ['Between_rat_%', 'Between_session_within_rat_%', 'Within_session_%']
lab = ['Between rats', 'Between sessions', 'Within session']
cols = ['#dddddd', '#888888', '#333333']
bottom = np.zeros(len(vd_df))
for c, l, col in zip(comp, lab, cols):
    a.bar([STATE_NAMES[k] for k in vd_df['State']], vd_df[c], bottom=bottom, color=col, edgecolor='black', label=l)
    bottom += vd_df[c].values
a.set_ylabel('Variance in state membership (%)'); a.set_ylim(0, 100)
a.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))

# F: occupancy across sessions
a = ax[3]
for k in STATES:
    d = by_session[by_session['State'] == f'Occ_S{k}']
    a.errorbar(d['Session_N'], d['mean'], yerr=d['sem'], color=STATE_COLORS[k], marker='o', ms=4, capsize=2,
               label=STATE_NAMES[k])
a.set_xlabel('Phase 3 session'); a.set_ylabel('Occupancy (% of assigned trials)')
a.set_ylim(0, 100); a.legend(frameon=False, fontsize=8)
a.xaxis.set_major_locator(MaxNLocator(integer=True))

# G: coefficient plot - within-rat intensity and session, from the adjusted LMM
a = ax[4]
if shock_ok:
    m = lmm_df[lmm_df['Model'] == 'Occupancy ~ intensity + session']
    terms = [('intensity, within rat (per 0.05 mA)', 'Shock intensity\n(per 0.05 mA, within rat)'),
             ('session (per session), adjusted', 'Session\n(per 5 sessions)')]
    for ti, (term, tlab) in enumerate(terms):
        for si, k in enumerate(STATES):
            rr = m[(m['State'] == k) & (m['Term'] == term)].iloc[0]
            scale = 5 if term.startswith('session') else 1
            y = ti + (si - 1) * 0.22
            a.errorbar(rr.beta * scale, y, xerr=[[(rr.beta - rr.ci_lo) * scale], [(rr.ci_hi - rr.beta) * scale]],
                       fmt='o', color=STATE_COLORS[k], capsize=3, label=STATE_NAMES[k] if ti == 0 else None)
    a.axvline(0, color='black', lw=0.8)
    a.set_yticks([0, 1]); a.set_yticklabels([t[1] for t in terms]); a.set_ylim(1.5, -0.5)
    a.set_xlabel('Change in occupancy (%), 95% CI')
    a.legend(frameon=False, fontsize=8, loc='lower right')
else:
    a.text(0.5, 0.5, 'No intensity data', ha='center', va='center', transform=a.transAxes)

for i, a in enumerate(ax):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.text(-0.15, 1.12, 'CDEFG'[i], transform=a.transAxes, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig_StateDynamics.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig_StateDynamics.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved outputs to {OUTPUT_DIR}/")
