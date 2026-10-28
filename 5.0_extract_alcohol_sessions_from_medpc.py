"""
5.0_extract_alcohol_sessions_from_medpc.py
------------------------------------------
Builds the trial-level alcohol-test table (the input of 5.1) directly from the raw MedPC
output files, so that every test session that was run is included, whether or not it
survived the MATLAB/photometry processing.

WHAT IS IN A MedPC FILE (program "Food Conflict_2023", 100 trials per session)
  Header      Subject, Start Date (MM/DD/YY), Start/End Time, Box, MSN
  Z array     one record of 6 numbers per trial, in trial order:
                [trial, cue, result, latency_s, trial_end_min, 0]
                cue    1 = rewDS, 2 = punDS, 3 = conflict
                result 1 = lever press, 4 = omission
                latency_s is measured from CUE onset (10 s cue + latency from lever insertion);
                it is 40 for an omission (10 s cue + 30 s lever)
                trial_end_min is minutes on the MedPC clock when the press / retraction happened
  Y array     [trial, cue, press(1)/omit(2), 0, mag_entry_min, mag_exit_min]. The first
              four columns repeat the trial list; the last two are a separate list of
              magazine-entry (beam break) times, in minutes. Only present in the "mag"
              version of the program.
  Not stored  the drug condition, and which outcome (pellet or shock) a conflict press produced.

HOW THE OUTPUT TABLE IS DEFINED (checked against the MATLAB table; see the validation report)
  Latency          = latency_s - 10  (from lever insertion); empty on omissions
  Time_MedPC_s     = trial start on the MedPC clock = trial_end_min*60 - latency_s
  Timestamp        = Time_MedPC_s + a per-session offset (the photometry start lag, about 55-75 s, constant
                     within a session to 0.02 s). Rows that agree with the MATLAB table keep the table's
                     Timestamp; otherwise the offset is the session's own (if other trials of the session
                     agree) or the median over validated sessions (typically about 7 s off, so about
                     0.1 trials per session can fall on the other side of the 900 s epoch boundary).
  Response         omission -> Omission; reward press -> Response-Pellet; punishment press ->
                   Response-Shock; conflict press -> taken from the MATLAB table when the trial agrees
                   with it, otherwise INFERRED from the magazine: a pellet is followed by a magazine
                   entry MAG_MIN-MAG_MAX s after the press, a shock usually is not (89.5% correct in
                   leave-one-rat-out testing on 1,565 validated conflict presses; 17.5% of shocks are
                   called pellets, 3.1% of pellets shocks; simulated label error of up to 30% did not
                   switch any trial between states). Sessions without magazine data get 'Response-Unknown'.
  Condition        from the MATLAB table for sessions in it, otherwise from --conditions (built from the lab
                   journal: injection record per rat and date; "EXCLUDE" marks a session that was not a
                   test, e.g. 'not tested'). Where a session is in both, the two are compared and any
                   disagreement is reported. A design-based guess is NOT used: it would have given R19 a
                   dose on 30 Jul, when the journal says R19 was not tested.
  Session_ID       <condition>_<k>, k = order of that condition within the rat.

RECONCILIATION WITH THE MATLAB TABLE (per session, before anything is taken from the table)
  The table's rows are matched to MedPC trials on cue, press/omission and latency (0.02 s).
  - same number of trials, all agree        -> table values are used
  - table has one extra row ("ghost" trial) -> the row whose removal makes the rest agree is dropped and
                                               the remaining rows are renumbered (found in R20, 9 Jul)
  - a few single trials disagree            -> MedPC wins for those trials; Response is re-derived, the
                                               photometry values of those trials are left empty
  - anything else                           -> the session is 'unreconciled': MedPC only, no table values
  All of this is listed in trial_discrepancies.csv and validation_vs_matlab_table.csv.

Usage (from the repository root, with the session folders in ./Alc_data)
  python 5.0_extract_alcohol_sessions_from_medpc.py --ignore-rats R09
  (run_alcohol_stage.sh runs it with the settings used for the paper.)
  The journal, the old MATLAB table and the conditions file are found automatically (in ., Data/ or the
  session folder); pass --root / --journal / --table / --conditions to override. Windows-style paths
  (a \\\\wsl.localhost share or a C: drive path) are converted for WSL.
  To see unmapped sessions first, remove or rename the conditions file: sessions with no known condition are listed in
  <report-dir>/unmapped_sessions_TEMPLATE.csv. Fill in Session_Type and pass it back with --conditions.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DOSES = ["Saline", "Alcohol_Low", "Alcohol_High"]
DOSE_ALIASES = {"saline": "Saline", "sal": "Saline", "veh": "Saline", "vehicle": "Saline",
                "low": "Alcohol_Low", "alcohol_low": "Alcohol_Low", "0.5": "Alcohol_Low", "0.5 g/kg": "Alcohol_Low",
                "high": "Alcohol_High", "alcohol_high": "Alcohol_High", "1.0": "Alcohol_High", "1": "Alcohol_High",
                "1.0 g/kg": "Alcohol_High"}
EXCLUDE_TOKENS = {"exclude", "not tested", "not_tested", "none", "no injection"}
CUES = {1: "reward", 2: "punish", 3: "conflict"}
PHOTOMETRY_COLS = ["ZMean_BL", "ZMean_Cue5s", "ZMean_Cue10s", "ZMean_CueFull", "ZMean_Lever",
                   "mean_postResponse_0_5", "mean_preResponse_10_0"]
CUE_PERIOD_S = 10.0
VERSION = "5.0 v4"
MAG_MIN, MAG_MAX = 0.25, 3.0    # s after a conflict press; a magazine entry in this window = pellet
                                # (chosen on 1,565 validated conflict presses; 89.5% correct in leave-one-rat-out testing)


# =========================================================
# 1. PARSE ONE MedPC FILE
# =========================================================
def parse_medpc(path):
    """Returns (header dict, scalar dict, array dict). MedPC writes decimal commas."""
    with open(path, newline="", errors="replace") as fh:
        txt = fh.read().replace("\r", "")
    if txt.count("Start Date:") > 1:
        print(f"  WARNING: {path} contains more than one session; only the last block is parsed.")
        txt = txt[txt.rfind("File:"):] if "File:" in txt else txt[txt.rfind("Start Date:"):]
    hdr, scal, arrays, cur = {}, {}, {}, None
    for line in txt.split("\n"):
        m = re.match(r"^(Start Date|End Date|Subject|Experiment|Group|Box|Start Time|End Time|MSN):\s*(.*)$", line)
        if m:
            hdr[m.group(1)] = m.group(2).strip()
            continue
        m = re.match(r"^([A-Z]):\s*(\S+)\s*$", line)
        if m:
            scal[m.group(1)] = float(m.group(2).replace(",", "."))
            cur = None
            continue
        m = re.match(r"^([A-Z]):\s*$", line)
        if m:
            cur = m.group(1)
            arrays[cur] = []
            continue
        m = re.match(r"^\s*\d+:\s+(.*)$", line)
        if m and cur:
            arrays[cur] += [float(x.replace(",", ".")) for x in m.group(1).split()]
    return hdr, scal, {k: np.array(v) for k, v in arrays.items()}


def session_from_file(path):
    """(trial table, magazine-entry array) for one file, or None if it does not look like a session."""
    hdr, scal, arr = parse_medpc(path)
    if "Z" not in arr or "Subject" not in hdr or "Start Date" not in hdr:
        return None
    Z = arr["Z"]
    Z = Z[: len(Z) // 6 * 6].reshape(-1, 6)
    Z = Z[Z[:, 0] > 0]
    if len(Z) == 0:
        return None
    date = pd.to_datetime(hdr["Start Date"], format="%m/%d/%y")
    trial_end_s = Z[:, 4] * 60.0
    d = pd.DataFrame({
        "Rat": hdr["Subject"], "Date_dt": date, "Box": hdr.get("Box"), "Trial": Z[:, 0].astype(int),
        "Cue": [CUES.get(int(c), "?") for c in Z[:, 1]],
        "Outcome": (Z[:, 2] == 1).astype(int),
        "_result": Z[:, 2].astype(int),
        "Latency": np.round(np.where(Z[:, 2] == 1, Z[:, 3] - CUE_PERIOD_S, np.nan), 3),
        "Time_MedPC_s": np.round(trial_end_s - Z[:, 3], 3),
        "_press_s": trial_end_s,
        "MedPC_File": str(path),
    })
    # magazine-entry list (last two columns of Y); absent in versions without the magazine recording
    mag = np.empty((0, 2))
    if "Y" in arr:
        Y = arr["Y"]
        Y = Y[: len(Y) // 6 * 6].reshape(-1, 6)
        mag = Y[:, 4:6][Y[:, 4] > 0] * 60.0
    return d, mag


# =========================================================
# 2. RESPONSE LABELS
# =========================================================
def mag_features(mag, press_s):
    """Delay (s) from a press to the next magazine entry, that entry's duration, the number of entries
    within 5 s after the press, and whether the rat was already in the magazine at the press."""
    nxt = mag[mag[:, 0] >= press_s - 0.05]
    delay = nxt[0, 0] - press_s if len(nxt) else np.inf
    dur = nxt[0, 1] - nxt[0, 0] if len(nxt) else np.nan
    n5 = int(((nxt[:, 0] - press_s) <= 5).sum()) if len(nxt) else 0
    inmag = bool(((mag[:, 0] < press_s - 0.05) & (mag[:, 1] > press_s)).any())
    return delay, dur, n5, inmag


def label_responses(d, mag):
    """Adds Response / Response_Source (MedPC-only labels) and the magazine features for one session."""
    has_mag = len(mag) >= 5
    resp, src, feats = [], [], []
    for _, r in d.iterrows():
        f = mag_features(mag, r["_press_s"]) if (has_mag and r["Outcome"] == 1) else (np.nan, np.nan, np.nan, np.nan)
        feats.append(f)
        if r["Outcome"] == 0:
            resp.append("Omission"); src.append("omission")
        elif r["Cue"] == "reward":
            resp.append("Response-Pellet"); src.append("rule")
        elif r["Cue"] == "punish":
            resp.append("Response-Shock"); src.append("rule")
        elif has_mag:
            resp.append("Response-Pellet" if MAG_MIN <= f[0] <= MAG_MAX else "Response-Shock")
            src.append("inferred_mag")
        else:
            resp.append("Response-Unknown"); src.append("unknown_no_mag_data")
    d["Response"], d["Response_Source"] = resp, src
    d["_delay"], d["_dur"], d["_n5"], d["_inmag"] = zip(*feats) if feats else ([], [], [], [])
    return d


# =========================================================
# 2b. RECONCILE A SESSION WITH THE MATLAB TABLE
# =========================================================
def trial_match(m, t):
    """Per-trial agreement between MedPC rows m and table rows t (same length, same order)."""
    cue_ok = m["Cue"].values == t["Cue"].values
    out_ok = m["Outcome"].values == t["Outcome"].astype(int).values
    ml, tl = m["Latency"].values.astype(float), t["Latency"].values.astype(float)
    lat_ok = np.where(np.isnan(ml) | np.isnan(tl), np.isnan(ml) & np.isnan(tl), np.abs(ml - tl) < 0.02)
    return cue_ok & out_ok & lat_ok


def align_session(m, t):
    """Match MedPC trials to table rows. Returns (status, t_idx, ok): t_idx[i] is the table row (position)
    for MedPC trial i or -1; ok[i] says the pair agrees on cue, press and latency."""
    n = len(m)
    if len(t) == n:
        ok = trial_match(m, t)
        return ("exact" if ok.all() else f"{int((~ok).sum())} discrepant trial(s)"), np.arange(n), ok
    if len(t) == n + 1:
        best = None
        for k in range(len(t)):
            keep = np.delete(np.arange(len(t)), k)
            ok = trial_match(m, t.iloc[keep])
            if best is None or ok.sum() > best[0].sum():
                best = (ok, keep, k)
        ok, keep, k = best
        if (~ok).sum() <= 3:
            status = f"extra table row (trial {int(t.iloc[k]['Trial'])}) removed"
            if not ok.all():
                status += f"; {int((~ok).sum())} discrepant trial(s)"
            return status, keep, ok
    return "unreconciled", np.full(n, -1), np.zeros(n, bool)


# =========================================================
# 3. INPUT HELPERS
# =========================================================
def find_sessions(root):
    sessions, bad = [], []
    for p in sorted(Path(root).rglob("Subject_R*_Experiment_*_Group_*.txt")):
        try:
            res = session_from_file(p)
        except Exception as e:     # keep going, report at the end
            bad.append((str(p), repr(e)))
            continue
        if res is None:
            bad.append((str(p), "no Z array / header"))
            continue
        d, mag = res
        m = re.match(r"^(\d{6})", p.parent.name)
        if m and m.group(1) != d["Date_dt"].iloc[0].strftime("%y%m%d"):
            print(f"  WARNING: folder '{p.parent.name}' starts with {m.group(1)} but the file's Start Date is "
                  f"{d['Date_dt'].iloc[0].date()} ({p.name})")
        sessions.append((d, mag))
    return sessions, bad


def read_journal(path):
    j = pd.read_csv(path)
    long = j.melt(id_vars="Date", var_name="Rat", value_name="Test").dropna()
    long["Date_dt"] = pd.to_datetime(long["Date"], format="%d/%m/%Y")
    return long[long["Test"].astype(str).str.upper().str.startswith("TEST")][["Rat", "Date_dt", "Test"]]


def to_wsl_path(p):
    """Accept Windows-style paths when running under WSL (wsl.localhost shares and drive letters)."""
    if p is None:
        return None
    q = str(p).strip().strip('"').strip("'")
    m = re.match(r"^\\\\(?:wsl\.localhost|wsl\$)\\[^\\]+\\(.*)$", q)
    if m:
        return "/" + m.group(1).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", q)
    if m:
        return f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")
    return q


def find_file(explicit, names, dirs):
    """The explicit path if given, otherwise the first existing <dir>/<name>."""
    if explicit:
        path = to_wsl_path(explicit)
        if not os.path.exists(path):
            sys.exit(f"File not found: {path}")
        return path
    for d in dirs:
        for n in names:
            cand = os.path.join(d, n)
            if os.path.exists(cand):
                return cand
    return None


def parse_date_any(x):
    s = str(x).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%y%m%d", "%d-%m-%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse date '{x}' (use YYYY-MM-DD, DD/MM/YYYY or YYMMDD)")


def read_conditions(path):
    c = pd.read_csv(path, dtype=str)
    need = {"Rat", "Date", "Session_Type"}
    if not need <= set(c.columns):
        raise ValueError(f"--conditions needs columns {need}; found {list(c.columns)}")
    c = c.dropna(subset=["Session_Type"])
    c["Date_dt"] = c["Date"].map(parse_date_any)
    st = c["Session_Type"].str.strip().str.lower()
    c["Session_Type"] = np.where(st.isin(EXCLUDE_TOKENS), "EXCLUDE", st.map(DOSE_ALIASES))
    if pd.isna(c["Session_Type"]).any():
        raise ValueError("Unrecognised Session_Type in --conditions; use Saline, Alcohol_Low, Alcohol_High or EXCLUDE.")
    c = c[["Rat", "Date_dt", "Session_Type"]]
    dup = c[c.duplicated(["Rat", "Date_dt"], keep=False)]
    if len(dup):
        conflicting = dup.groupby(["Rat", "Date_dt"])["Session_Type"].nunique()
        if (conflicting > 1).any():
            bad = conflicting[conflicting > 1].reset_index()
            raise ValueError("--conditions gives different doses for the same rat and date:\n"
                             + bad.assign(Date_dt=bad.Date_dt.dt.date).to_string(index=False))
        print(f"  NOTE: {len(dup) - dup.drop_duplicates(['Rat', 'Date_dt']).shape[0]} identical duplicate rows in --conditions dropped.")
        c = c.drop_duplicates(["Rat", "Date_dt"])
    return c


# =========================================================
# 4. MAIN
# =========================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="Alc_data",
                    help="folder containing the session folders / MedPC files (default: Alc_data; Windows or WSL paths accepted)")
    ap.add_argument("--journal", default=None, help="Master_Labjournal_Sessions.csv (default: looked for in . , Data/ and --root)")
    ap.add_argument("--table", default=None, help="existing MATLAB-derived alcohol table (default: Conflict_02_Alc_tests_UPDATED_v2.csv in . , Data/ or --root)")
    ap.add_argument("--conditions", default=None, help="CSV with Rat, Date, Session_Type from the lab journal (default: conditions_from_lab_journal.csv in . )")
    ap.add_argument("--ignore-rats", nargs="*", default=[], help="rat IDs to leave out entirely (e.g. R09, removed before the tests were completed)")
    ap.add_argument("--out", default=os.path.join("Data", "collated_2026_conflict_alcohol_raw.csv"))
    ap.add_argument("--report-dir", default=os.path.join("HMM_Outputs", "Alcohol_extract"))
    args = ap.parse_args()
    args.root = to_wsl_path(args.root)
    if not os.path.isdir(args.root):
        sys.exit(f"--root is not a folder: {args.root}\nRun from the repository root, or pass --root with the full path.")
    search = [".", "Data", args.root]
    args.journal = find_file(args.journal, ["Master_Labjournal_Sessions.csv"], search)
    args.table = find_file(args.table, ["Conflict_02_Alc_tests_UPDATED_v2.csv"], search)
    args.conditions = find_file(args.conditions, ["conditions_from_lab_journal.csv", "conditions_missing_sessions.csv"], ["."])
    print(f"[{VERSION}]")
    print("Using:")
    for lab, v in [("session folders", args.root), ("journal", args.journal),
                   ("existing MATLAB table", args.table), ("conditions file", args.conditions)]:
        print(f"  {lab:22s} {v if v else '(not found - skipped)'}")
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ---- read files ----
    print(f"Scanning {args.root} ...")
    sessions, bad = find_sessions(args.root)
    for p, why in bad:
        print(f"  skipped {p}: {why}")
    if not sessions:
        sys.exit("No MedPC sessions found. Check --root.")
    # one file per rat x date; if duplicates, keep the one with the most trials
    best, MAG = {}, {}
    for d, mag in sessions:
        k = (d["Rat"].iloc[0], d["Date_dt"].iloc[0])
        if k in best:
            print(f"  WARNING: duplicate files for {k[0]} {k[1].date()}; keeping the longer one.")
            if len(d) <= len(best[k]):
                continue
        best[k], MAG[k] = d, mag
    if args.ignore_rats:
        dropped = [k for k in best if k[0] in args.ignore_rats]
        print(f"Ignoring {args.ignore_rats}: {len(dropped)} sessions left out.")
        best = {k: v for k, v in best.items() if k[0] not in args.ignore_rats}
    sessions = list(best.values())
    print(f"Found {len(sessions)} sessions from {len({k[0] for k in best})} rats.")

    # ---- restrict to test sessions via the journal ----
    test_label = {}
    if args.journal:
        j = read_journal(args.journal)
        test_label = {(r.Rat, r.Date_dt): r.Test for r in j.itertuples()}
        missing = j[[(r.Rat, r.Date_dt) not in best for r in j.itertuples()]]
        missing.to_csv(os.path.join(args.report_dir, "journal_sessions_without_medpc_file.csv"), index=False)
        if len(missing):
            print("\nJournal TEST sessions with no MedPC file (not run, or file not in --root):")
            print(missing.assign(Date_dt=missing.Date_dt.dt.date).to_string(index=False))
        sessions = [d for d in sessions if (d["Rat"].iloc[0], d["Date_dt"].iloc[0]) in test_label]
        print(f"{len(sessions)} MedPC sessions are journal TEST sessions.")

    # ---- structural checks ----
    for d in sessions:
        rat, dt = d["Rat"].iloc[0], d["Date_dt"].iloc[0].date()
        if d["Trial"].tolist() != list(range(1, len(d) + 1)):
            print(f"  WARNING: {rat} {dt}: trial numbers are not 1..N")
        if len(d) != 100:
            print(f"  NOTE: {rat} {dt}: {len(d)} trials (expected 100)")
        if (d["Cue"] == "?").any() or not set(d["_result"]) <= {1, 4}:
            print(f"  WARNING: {rat} {dt}: unexpected cue or result codes {sorted(set(d['_result']))}")
        label_responses(d, MAG[(rat, d["Date_dt"].iloc[0])])
        d["Test"] = test_label.get((rat, d["Date_dt"].iloc[0]), "")
    M = pd.concat(sessions, ignore_index=True)
    M["Date"] = M["Date_dt"].dt.strftime("%y%m%d").astype(int)

    # ---- reconcile with the existing MATLAB table (validation, conditions, photometry) ----
    M = M.sort_values(["Rat", "Date_dt", "Trial"]).reset_index(drop=True)
    M["Response_MedPC"], M["Response_Source_MedPC"] = M["Response"], M["Response_Source"]
    M["Session_Type"] = pd.Series([None] * len(M), index=M.index, dtype=object)
    M["Response_T"] = pd.Series([None] * len(M), index=M.index, dtype=object)
    M["Table_Status"] = pd.Series(["not in table"] * len(M), index=M.index, dtype=object)
    M["Timestamp_T"] = np.nan
    for c in PHOTOMETRY_COLS:
        M[c] = np.nan
    M["Table_Row_Used"] = False
    offsets, disc, vrows, calib = {}, [], [], []
    T = None
    if args.table:
        T = pd.read_csv(args.table, dtype={"Session_ID": str})
        T["Date_dt"] = pd.to_datetime(T["Date"].astype(str), format="%y%m%d")
        pcols = [c for c in PHOTOMETRY_COLS if c in T.columns]
        for (rat, dt), g in M.groupby(["Rat", "Date_dt"], sort=False):
            t = T[(T["Rat"] == rat) & (T["Date_dt"] == dt)].sort_values("Trial").reset_index(drop=True)
            if t.empty:
                continue
            ix, m = g.index, g.reset_index(drop=True)
            status, tidx, ok = align_session(m, t)
            M.loc[ix, "Session_Type"] = t["Session_Type"].iloc[0]
            M.loc[ix, "Table_Status"] = status
            rows = ix[ok]
            trow = t.iloc[tidx[ok]]
            M.loc[rows, "Response_T"] = trow["Response"].values
            M.loc[rows, "Timestamp_T"] = trow["Timestamp"].values
            for c in pcols:
                M.loc[rows, c] = trow[c].values
            M.loc[rows, "Table_Row_Used"] = True
            # trials that exist on both sides but disagree; table rows with no MedPC trial
            for i_, (bad, ti) in enumerate(zip(~ok, tidx)):
                if bad and ti >= 0:
                    tr_ = t.iloc[ti]
                    disc.append({"Rat": rat, "Date": dt.date(), "Kind": "trial disagrees", "Trial_medpc": int(m.Trial[i_]),
                                 "Trial_table": int(tr_["Trial"]), "Cue_medpc": m.Cue[i_], "Cue_table": tr_["Cue"],
                                 "Outcome_medpc": int(m.Outcome[i_]), "Outcome_table": int(tr_["Outcome"]),
                                 "Latency_medpc": m.Latency[i_], "Latency_table": tr_["Latency"], "Response_table": tr_["Response"]})
            for k in sorted(set(range(len(t))) - set(tidx[tidx >= 0])) if status != "unreconciled" else []:
                tr_ = t.iloc[k]
                disc.append({"Rat": rat, "Date": dt.date(), "Kind": "extra table row (no MedPC trial)", "Trial_medpc": np.nan,
                             "Trial_table": int(tr_["Trial"]), "Cue_table": tr_["Cue"], "Outcome_table": int(tr_["Outcome"]),
                             "Latency_table": tr_["Latency"], "Response_table": tr_["Response"]})
            if ok.any():
                dif = M.loc[rows, "Timestamp_T"].values - M.loc[rows, "Time_MedPC_s"].values
                offsets[(rat, dt)] = (float(np.median(dif)), float(np.std(dif)))
            g_ok = M.loc[rows]
            conf = (g_ok["Cue"] == "conflict") & (g_ok["Outcome"] == 1)
            nonconf = g_ok[~conf]
            vrows.append({
                "Rat": rat, "Date": dt.date(), "n_trials_medpc": len(m), "n_trials_table": len(t), "alignment": status,
                "n_trials_agreeing": int(ok.sum()),
                "latency_max_abs_diff_s": float(np.nanmax(np.abs(g_ok["Latency"].values - trow["Latency"].values))) if ok.any() and g_ok["Latency"].notna().any() else np.nan,
                "response_mismatch_nonconflict": int((nonconf["Response_MedPC"] != nonconf["Response_T"]).sum()),
                "conflict_presses": int(conf.sum()),
                "mag_rule_correct_on_conflict": int((g_ok.loc[conf, "Response_MedPC"] == g_ok.loc[conf, "Response_T"]).sum()),
                "timestamp_offset_median_s": offsets.get((rat, dt), (np.nan, np.nan))[0],
                "timestamp_offset_sd_s": offsets.get((rat, dt), (np.nan, np.nan))[1]})
            cp = g_ok[g_ok["Outcome"] == 1]
            calib.append(pd.DataFrame({"Rat": rat, "Date": dt.date(), "Trial": cp["Trial"], "Cue": cp["Cue"],
                                       "True_Response": cp["Response_T"], "Delay_s": cp["_delay"].replace(np.inf, np.nan),
                                       "Entry_duration_s": cp["_dur"], "Entries_within_5s": cp["_n5"],
                                       "Already_in_magazine": cp["_inmag"]}))
        V = pd.DataFrame(vrows)
        V.to_csv(os.path.join(args.report_dir, "validation_vs_matlab_table.csv"), index=False)
        pd.DataFrame(disc).to_csv(os.path.join(args.report_dir, "trial_discrepancies.csv"), index=False)
        if calib:
            pd.concat(calib).to_csv(os.path.join(args.report_dir, "mag_calibration_presses.csv"), index=False)

        print(f"\n{int(M['Table_Row_Used'].sum())} of {len(M)} MedPC trials matched a MATLAB-table row "
              f"({len(V)} sessions in both).")
        print("\n=== Validation against the MATLAB table ===")
        print("session alignment:", V["alignment"].map(lambda a: a if a == "exact" else "NOT exact").value_counts().to_dict())
        odd = V[V["alignment"] != "exact"]
        if len(odd):
            print("Sessions that needed reconciliation:")
            print(odd[["Rat", "Date", "n_trials_medpc", "n_trials_table", "alignment"]].to_string(index=False))
        print(f"agreeing trials: {int(V.n_trials_agreeing.sum())} of {int(V.n_trials_medpc.sum())}; "
              f"max latency difference on agreeing trials: {V.latency_max_abs_diff_s.max():.4f} s; "
              f"non-conflict Response mismatches: {int(V.response_mismatch_nonconflict.sum())}")
        cpn = int(V.conflict_presses.sum())
        print(f"magazine rule on conflict presses: {int(V.mag_rule_correct_on_conflict.sum())}/{cpn} correct "
              f"({100 * V.mag_rule_correct_on_conflict.sum() / max(cpn, 1):.1f}%)")
        good = V[(V.timestamp_offset_sd_s < 0.1)]
        offset_default = float(good.timestamp_offset_median_s.median())
        print(f"Timestamp offset (table - MedPC clock): median {offset_default:.1f} s, "
              f"range {good.timestamp_offset_median_s.min():.1f} to {good.timestamp_offset_median_s.max():.1f}; "
              f"within-session SD up to {good.timestamp_offset_sd_s.max():.3f} s ({len(good)} sessions)")
        if len(disc):
            print(f"{len(disc)} trial-level discrepancies written to trial_discrepancies.csv")
    else:
        offset_default = np.nan
        print("\nNo --table given: no validation, no Timestamp offset (Timestamp left empty).")

    # ---- final values: the table where its row agrees with MedPC, otherwise MedPC ----
    used = M["Table_Row_Used"]
    M["Response"] = np.where(used, M["Response_T"], M["Response_MedPC"])
    M["Response_Source"] = np.where(used, "matlab_table", M["Response_Source_MedPC"])
    off = [offsets.get((r, d), (np.nan, np.nan))[0] for r, d in zip(M["Rat"], M["Date_dt"])]
    off = np.array(off, float)
    M["Timestamp"] = np.where(used, M["Timestamp_T"],
                              M["Time_MedPC_s"] + np.where(np.isnan(off), offset_default, off))
    M["Timestamp"] = M["Timestamp"].round(3)
    M["Timestamp_Source"] = np.where(used, "matlab_table", np.where(np.isnan(off), "median_offset", "session_offset"))
    if T is None:
        M["Timestamp_Source"] = "none (no --table)"
    cond_from_table = M["Session_Type"].copy()
    if T is not None:
        sex_hemi = T.drop_duplicates("Rat").set_index("Rat")[[c for c in ["Sex", "Hemi"] if c in T.columns]]
        M = M.join(sex_hemi, on="Rat")

    # ---- conditions ----
    M["Session_Type"] = cond_from_table
    M["Condition_Source"] = np.where(M["Session_Type"].notna(), "matlab_table", None)
    if args.conditions is None:
        print("\nNo conditions file found (looked for conditions_from_lab_journal.csv in the current folder; "
              "pass --conditions PATH). Sessions not in the MATLAB table cannot be restored without one.")
    if args.conditions:
        C = read_conditions(args.conditions)
        M = M.merge(C.rename(columns={"Session_Type": "_cond"}), on=["Rat", "Date_dt"], how="left")
        sess = M[["Rat", "Date_dt", "Session_Type", "_cond"]].drop_duplicates(["Rat", "Date_dt"])
        both = sess[sess["Session_Type"].notna() & sess["_cond"].notna() & (sess["_cond"] != "EXCLUDE")]
        clash = both[both["Session_Type"] != both["_cond"]]
        print(f"\nConditions file vs MATLAB table: {len(both) - len(clash)} of {len(both)} sessions in both agree"
              + ("." if len(clash) == 0 else ";"))
        if len(clash):
            print("  DISAGREE (the table's dose is kept; check the journal):")
            print(clash.assign(Date_dt=clash.Date_dt.dt.date).rename(columns={"Session_Type": "table", "_cond": "conditions_file"}).to_string(index=False))
        exc = M["_cond"] == "EXCLUDE"
        if exc.any():
            ex = M[exc][["Rat", "Date_dt", "Test"]].drop_duplicates()
            ex.assign(Date=ex.Date_dt.dt.date, Reason="conditions file: EXCLUDE").drop(columns="Date_dt").to_csv(
                os.path.join(args.report_dir, "excluded_sessions.csv"), index=False)
            print("Sessions EXCLUDED by the conditions file (not tests):")
            print(ex.assign(Date_dt=ex.Date_dt.dt.date).to_string(index=False))
            if (M[exc]["Table_Status"] != "not in table").any():
                print("  WARNING: an excluded session is present in the MATLAB table.")
            M = M[~exc]
        newly = M["Session_Type"].isna() & M["_cond"].notna()
        M["Condition_Source"] = np.where(newly, "conditions_file", M["Condition_Source"])
        M["Session_Type"] = M["Session_Type"].fillna(M["_cond"])
        M = M.drop(columns="_cond")
    unmapped = M[M["Session_Type"].isna()][["Rat", "Date_dt", "Test"]].drop_duplicates()
    if len(unmapped):
        t_ = unmapped.assign(Date=unmapped.Date_dt.dt.strftime("%Y-%m-%d"), Session_Type="")[["Rat", "Date", "Test", "Session_Type"]]
        t_.to_csv(os.path.join(args.report_dir, "unmapped_sessions_TEMPLATE.csv"), index=False)
        print("\nSessions with NO KNOWN CONDITION (excluded from the output). Fill in Session_Type "
              f"(Saline / Alcohol_Low / Alcohol_High) in {args.report_dir}/unmapped_sessions_TEMPLATE.csv and "
              "pass it with --conditions:")
        print(t_.to_string(index=False))
    M = M[M["Session_Type"].notna()].copy()

    # ---- session IDs, order, output ----
    order = (M[["Rat", "Date_dt", "Session_Type"]].drop_duplicates().sort_values(["Rat", "Date_dt"]))
    order["k"] = order.groupby(["Rat", "Session_Type"]).cumcount() + 1
    order["Session_ID"] = order["Session_Type"] + "_" + order["k"].astype(str)
    M = M.merge(order[["Rat", "Date_dt", "Session_ID"]], on=["Rat", "Date_dt"])
    if T is not None:
        chk = M[["Rat", "Date_dt", "Session_ID"]].drop_duplicates().merge(
            T[["Rat", "Date_dt", "Session_ID"]].drop_duplicates(), on=["Rat", "Date_dt"], suffixes=("", "_table"))
        diff = chk[chk.Session_ID != chk.Session_ID_table]
        if len(diff):
            print("\nNOTE: Session_ID numbering differs from the old table for (restored sessions shift k):")
            print(diff.assign(Date_dt=diff.Date_dt.dt.date).to_string(index=False))

    M = M.sort_values(["Rat", "Date_dt", "Trial"])
    cols = (["Rat", "Sex", "Hemi", "Session_Type", "Session_ID", "Date", "Trial", "Timestamp", "Cue",
             "Response", "Outcome", "Latency"] + [c for c in PHOTOMETRY_COLS if c in M.columns] +
            ["Test", "Box", "Time_MedPC_s", "Response_Source", "Timestamp_Source", "Table_Status",
             "Table_Row_Used", "Condition_Source", "MedPC_File"])
    M[[c for c in cols if c in M.columns]].to_csv(args.out, index=False)

    # ---- summary ----
    inv = (M.groupby(["Rat", "Date", "Test", "Session_Type", "Session_ID"], dropna=False)
             .agg(n_trials=("Trial", "size"), table_status=("Table_Status", "first"),
                  condition_source=("Condition_Source", "first"),
                  response_source=("Response_Source", lambda x: ",".join(sorted(set(x)))),
                  timestamp_source=("Timestamp_Source", lambda x: ",".join(sorted(set(x))))).reset_index())
    inv.to_csv(os.path.join(args.report_dir, "session_inventory.csv"), index=False)
    print(f"\nWrote {len(M)} trials, {len(inv)} sessions, {M['Rat'].nunique()} rats to {args.out}")
    print(pd.crosstab(inv["Rat"], inv["Session_Type"]).to_string())
    new = inv[(inv["table_status"] != "exact") | inv["response_source"].str.contains("inferred|unknown")]
    if len(new):
        print("\nSessions NOT taken wholly from the MATLAB table (restored, reconciled, or with inferred outcomes):")
        print(new[["Rat", "Date", "Session_Type", "table_status", "condition_source", "response_source", "timestamp_source"]].to_string(index=False))
    if (M["Response"] == "Response-Unknown").any():
        print("\nWARNING: some conflict presses have Response-Unknown (no magazine data in that session). "
              "5.1 will refuse to run until these are resolved from another source.")
    print(f"\nReport files are in {args.report_dir}/")


if __name__ == "__main__":
    main()
