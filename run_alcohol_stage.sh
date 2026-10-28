#!/usr/bin/env bash
# Runs the alcohol-test stage (5.0 -> 5.3) with the settings used for the paper, from the
# repository root. Every file path is passed explicitly, each script's version string is
# checked before anything runs, and each step's console output is saved in
# HMM_Outputs/Alcohol_logs/.
#
#   bash run_alcohol_stage.sh            # 5.0 to 5.3
#   bash run_alcohol_stage.sh --checks   # also 5.4 (frozen-model checks; slow, not in the paper)
#
# Needs: Alc_data/ (MedPC files), Data/Conflict_02_Alc_tests_UPDATED_v2.csv (photometry-derived
# table), conditions_from_lab_journal.csv and final_3_state_model_v5.pkl.
set -euo pipefail

PY="${PYTHON:-python}"
LOG="HMM_Outputs/Alcohol_logs"
RAW="Data/collated_2026_conflict_alcohol_raw.csv"
DECODED="Data/collated_2026_conflict_alcohol_with_Probabilities_065.csv"
MODEL="final_3_state_model_v5.pkl"
mkdir -p "$LOG"

# Refuse to run older copies of the scripts.
check_version () {   # file, expected VERSION string
  grep -q "VERSION = \"$2\"" "$1" || { echo "ERROR: $1 is not version '$2'. Use the repository copy."; exit 1; }
}
check_version 5.0_extract_alcohol_sessions_from_medpc.py "5.0 v4"
check_version 5.1_alcohol_states_to_trials.py            "5.1 v2"
check_version 5.2_alcohol_state_occupancy.py             "5.2 v4"
check_version 5.3_alcohol_dwell_times.py                 "5.3 v2"
for f in Alc_data Data/Conflict_02_Alc_tests_UPDATED_v2.csv conditions_from_lab_journal.csv "$MODEL"; do
  [ -e "$f" ] || { echo "ERROR: $f not found (run from the repository root)."; exit 1; }
done

run () {   # log name, command...
  local name="$1"; shift
  echo "=== $name ==="
  "$@" 2>&1 | tee "$LOG/$name.log"
}

# R09 left the cohort before the tests were completed (Methods); 5.2 / 5.3 would drop it anyway.
run 5.0_extract "$PY" 5.0_extract_alcohol_sessions_from_medpc.py \
    --root Alc_data --table Data/Conflict_02_Alc_tests_UPDATED_v2.csv \
    --conditions conditions_from_lab_journal.csv --ignore-rats R09 \
    --out "$RAW" --report-dir HMM_Outputs/Alcohol_extract
run 5.1_decode "$PY" 5.1_alcohol_states_to_trials.py --input "$RAW" --output "$DECODED" --model "$MODEL"
# Primary analysis: every rat with all three doses (n = 15)
run 5.2_occupancy_n15 "$PY" 5.2_alcohol_state_occupancy.py --input "$DECODED" --exclude none --out HMM_Outputs/Alcohol_n15
run 5.3_dwell_n15     "$PY" 5.3_alcohol_dwell_times.py      --input "$DECODED" --exclude none --out HMM_Outputs/Alcohol_n15
# Sensitivity analysis: the photometry rats only (R18 and R21 excluded after histology; n = 13)
run 5.2_occupancy_n13 "$PY" 5.2_alcohol_state_occupancy.py --input "$DECODED" --exclude R18 R21 --out HMM_Outputs/Alcohol_n13

if [ "${1:-}" = "--checks" ]; then
  run 5.4_checks "$PY" 5.4_alcohol_frozen_model_checks.py
fi
echo "Done. Logs in $LOG/"
