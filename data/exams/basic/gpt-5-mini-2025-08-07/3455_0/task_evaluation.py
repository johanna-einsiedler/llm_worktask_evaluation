#!/usr/bin/env python3
"""
task_evaluation.py

Automated grader for the Basic Practical Telemetry exam (basic level).

Usage:
    python task_evaluation.py <candidate_submission.json> <answer_key.json>

Output:
    Creates/overwrites test_results.json in the current working directory with:
    - per-task score breakdown
    - detailed messages for any mismatches/deductions
    - total score, max score, percentage and overall_score key (numeric 0-100)
"""

import json
import sys
import os
from math import isfinite
from datetime import datetime

# ---------- Configuration: scoring weights (total 100) ----------
WEIGHTS = {
    "task1": 30,   # Ingest & DB creation
    "task2": 40,   # Query & Analysis
    "task3": 15,   # Insert & recompute
    "task4": 10,   # Documentation & reproducibility
    "code_hygiene": 5  # Minor points for env/run details presence
}

# Sub-weights inside tasks for detailed messages (sum to parent weight)
SUB_WEIGHTS = {
    "task1": {
        "db_path": 10,
        "schema_description": 10,
        "rows_counts": 10
    },
    "task2": {
        "total_rows": 5,
        "top_hosts": 10,
        "hourly_stats": 10,
        "cpu_spikes": 15
    },
    "task3": {
        "insert_success": 10,
        "insert_validation": 5
    },
    "task4": {
        "run_instructions": 5,
        "schema_and_improvements": 5
    }
}

# Numeric comparison tolerances
NUM_TOL = 0.01  # tolerance for average comparisons (two decimals)
AVG_TOL = 0.0001  # small tolerance for exact expected floats like 74.0

# Helper functions
def load_json_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)

def safe_get(dct, path_list, default=None):
    cur = dct
    for k in path_list:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and isfinite(x)

def float_eq(a, b, tol=NUM_TOL):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False

def normalize_ts(ts):
    """
    Normalize ISO8601 timestamps used in exam (expects trailing 'Z' UTC).
    Accept formats like 'YYYY-MM-DDTHH:MM:SSZ' or variations parseable.
    Returns canonical 'YYYY-MM-DDTHH:MM:SSZ' or raises ValueError.
    """
    if not isinstance(ts, str):
        raise ValueError("timestamp not string")
    # Try common ISO with Z
    fmts = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z"
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(ts, fmt)
            # produce canonical Z format (no offset)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            continue
    # Try python's fromisoformat (doesn't accept Z), convert Z to +00:00
    try:
        s = ts
        if ts.endswith("Z"):
            s = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        # output in canonical Z (UTC) if offset is zero
        if dt.utcoffset() is not None:
            # convert to UTC naive
            dt_utc = dt - dt.utcoffset()
            return dt_utc.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        raise ValueError("unsupported timestamp format: {}".format(ts))

def compare_top_hosts(expected_list, actual_list, messages):
    """
    expected_list and actual_list are lists of dicts with 'host' and 'avg_cpu'.
    Order matters per exam (descending).
    Returns (points_awarded, max_points, messages_added)
    """
    max_points = SUB_WEIGHTS['task2']['top_hosts']
    awarded = 0
    msgs = []
    if not isinstance(actual_list, list):
        msgs.append("top_hosts_by_avg_cpu missing or not a list.")
        return 0, max_points, msgs
    if len(actual_list) != len(expected_list):
        msgs.append("top_hosts_by_avg_cpu length mismatch: expected {}, got {}.".format(len(expected_list), len(actual_list)))
        # still attempt to compare min length
    correct = 0
    total = len(expected_list)
    for idx, exp in enumerate(expected_list):
        try:
            cand = actual_list[idx]
        except Exception:
            msgs.append("Missing candidate entry for rank {} expected host {}".format(idx+1, exp.get('host')))
            continue
        exp_host = exp.get('host')
        cand_host = cand.get('host')
        exp_avg = exp.get('avg_cpu')
        cand_avg = cand.get('avg_cpu')
        host_ok = (str(exp_host).strip().lower() == str(cand_host).strip().lower())
        avg_ok = False
        if is_number(exp_avg) and is_number(cand_avg):
            avg_ok = float_eq(exp_avg, cand_avg, NUM_TOL)
        if host_ok and avg_ok:
            correct += 1
        else:
            msgs.append("top_hosts rank {} mismatch: expected ({},{:.2f}), got ({},{})".format(
                idx+1, exp_host, exp_avg if is_number(exp_avg) else exp_avg, cand_host, cand_avg))
    # award proportional points
    if total > 0:
        awarded = int(round(max_points * (correct / total)))
    return awarded, max_points, msgs

def compare_hourly_stats(expected_list, actual_list, messages):
    """
    expected_list and actual_list: lists of dicts with hour_start, avg_cpu, max_mem_mb.
    Need chronological order; exam expects exact set and order.
    Award points proportional to matched buckets.
    """
    max_points = SUB_WEIGHTS['task2']['hourly_stats']
    msgs = []
    if not isinstance(actual_list, list):
        msgs.append("hourly_stats_for_host missing or not a list.")
        return 0, max_points, msgs
    exp_len = len(expected_list)
    cand_len = len(actual_list)
    matched = 0
    # Build mapping from hour_start -> tuple(avg_cpu, max_mem_mb)
    exp_map = {}
    for e in expected_list:
        try:
            key = normalize_ts(e['hour_start'])
        except Exception:
            key = e.get('hour_start')
        exp_map[key] = (e.get('avg_cpu'), e.get('max_mem_mb'))
    cand_map = {}
    for c in actual_list:
        try:
            key = normalize_ts(c['hour_start'])
        except Exception:
            key = c.get('hour_start')
        cand_map[key] = (c.get('avg_cpu'), c.get('max_mem_mb'))
    # Check each expected bucket exists and values match
    for k, (exp_avg, exp_max) in exp_map.items():
        if k not in cand_map:
            msgs.append("Missing hourly bucket for hour_start {}".format(k))
            continue
        cand_avg, cand_max = cand_map[k]
        avg_ok = is_number(exp_avg) and is_number(cand_avg) and float_eq(exp_avg, cand_avg, NUM_TOL)
        max_ok = is_number(exp_max) and is_number(cand_max) and (abs(float(exp_max) - float(cand_max)) <= 1e-6)
        if avg_ok and max_ok:
            matched += 1
        else:
            msgs.append("Hourly bucket {} value mismatch: expected avg {:.2f}, max {} ; got avg {}, max {}".format(
                k, exp_avg, exp_max, cand_avg, cand_max))
    if exp_len > 0:
        awarded = int(round(max_points * (matched / exp_len)))
    else:
        awarded = max_points if cand_len == 0 else 0
    return awarded, max_points, msgs

def compare_cpu_spikes(expected_list, actual_list, messages):
    """
    expected_list: authoritative list of spike dicts. Candidate may include more; must include all expected.
    Award full points if all expected periods are present (matching host and normalized ts).
    Partial credit for subset.
    """
    max_points = SUB_WEIGHTS['task2']['cpu_spikes']
    msgs = []
    if not isinstance(actual_list, list):
        msgs.append("cpu_spike_periods missing or not a list.")
        return 0, max_points, msgs
    expected_norm = []
    for e in expected_list:
        try:
            start = normalize_ts(e['start_ts'])
            end = normalize_ts(e['end_ts'])
        except Exception:
            # fallback to raw strings
            start = e.get('start_ts')
            end = e.get('end_ts')
        expected_norm.append((e.get('host'), start, end))
    actual_norm = set()
    for a in actual_list:
        try:
            start = normalize_ts(a.get('start_ts'))
            end = normalize_ts(a.get('end_ts'))
        except Exception:
            start = a.get('start_ts')
            end = a.get('end_ts')
        actual_norm.add((a.get('host'), start, end))
    matched = 0
    for exp in expected_norm:
        if exp in actual_norm:
            matched += 1
        else:
            msgs.append("Missing expected CPU spike period: host {}, start {}, end {}".format(exp[0], exp[1], exp[2]))
    if len(expected_norm) > 0:
        awarded = int(round(max_points * (matched / len(expected_norm))))
    else:
        awarded = max_points if len(actual_norm) == 0 else 0
    return awarded, max_points, msgs

def compare_top_level_number(field_path, expected, actual, max_points, tol=NUM_TOL):
    msgs = []
    awarded = 0
    if actual is None:
        msgs.append("Missing value for {}".format(".".join(field_path)))
        return awarded, max_points, msgs
    if not is_number(expected) or not is_number(actual):
        msgs.append("Non-numeric value for {}".format(".".join(field_path)))
        return awarded, max_points, msgs
    if float_eq(expected, actual, tol):
        awarded = max_points
    else:
        # partial credit if close (relaxed)
        diff = abs(float(expected) - float(actual))
        if diff <= 1:
            awarded = int(round(max_points * 0.5))
            msgs.append("Value for {} close but not exact: expected {}, got {}".format(".".join(field_path), expected, actual))
        else:
            msgs.append("Value for {} mismatch: expected {}, got {}".format(".".join(field_path), expected, actual))
    return awarded, max_points, msgs

def safe_lower_str(s):
    return (s or "").strip().lower()

def main():
    if len(sys.argv) != 3:
        print("Usage: python task_evaluation.py <candidate_submission.json> <answer_key.json>")
        sys.exit(2)

    cand_path = sys.argv[1]
    key_path = sys.argv[2]

    cand_json, err = load_json_file(cand_path)
    if err:
        print("Error loading candidate submission JSON '{}': {}".format(cand_path, err))
        sys.exit(2)
    key_json, err = load_json_file(key_path)
    if err:
        print("Error loading answer key JSON '{}': {}".format(key_path, err))
        sys.exit(2)

    results = {
        "per_task": {},
        "total_points": 0,
        "max_points": sum(WEIGHTS.values()),
        "percentage": 0.0,
        "overall_score": 0.0,
        "messages": []
    }

    total_awarded = 0

    # -------------- Task 1: Ingest & DB creation (30) --------------
    task1_msgs = []
    task1_awarded = 0
    # 1.1 db_path check (expected db file)
    expected_db = safe_get(key_json, ['env', 'db_file'])
    cand_db = safe_get(cand_json, ['env', 'db_file'])
    db_points = SUB_WEIGHTS['task1']['db_path']
    if not cand_db:
        task1_msgs.append("env.db_file missing in submission.")
    else:
        # compare with expected
        if expected_db and os.path.basename(str(cand_db)) == os.path.basename(str(expected_db)):
            task1_awarded += db_points
        else:
            # partial credit if candidate provided some db file name
            task1_awarded += int(round(db_points * 0.5))
            task1_msgs.append("env.db_file differs from expected. Expected '{}', got '{}' (partial credit).".format(expected_db, cand_db))
    # 1.2 schema_description presence & non-empty
    schema_points = SUB_WEIGHTS['task1']['schema_description']
    schema_desc = safe_get(cand_json, ['schema_and_choices', 'schema_description'])
    if isinstance(schema_desc, str) and schema_desc.strip():
        # lightweight heuristic: must mention 'table' and some columns or 'telemetry'
        lc = safe_lower_str(schema_desc)
        if 'table' in lc or 'telemetry' in lc or 'timestamp' in lc:
            task1_awarded += schema_points
        else:
            # partial credit if present but terse
            task1_awarded += int(round(schema_points * 0.5))
            task1_msgs.append("schema_description provided but terse or missing key terms; partial credit.")
    else:
        task1_msgs.append("schema_description missing or empty; 0 for schema_description subtask.")

    # 1.3 rows_loaded & rows_skipped correctness
    rows_points = SUB_WEIGHTS['task1']['rows_counts']
    expected_rows_loaded = safe_get(key_json, ['ingestion', 'rows_loaded'])
    expected_rows_skipped = safe_get(key_json, ['ingestion', 'rows_skipped'])
    cand_rows_loaded = safe_get(cand_json, ['ingestion', 'rows_loaded'])
    cand_rows_skipped = safe_get(cand_json, ['ingestion', 'rows_skipped'])
    if is_number(expected_rows_loaded) and is_number(expected_rows_skipped):
        if cand_rows_loaded == expected_rows_loaded and cand_rows_skipped == expected_rows_skipped:
            task1_awarded += rows_points
        else:
            # partial credit: closeness
            msg = "Ingestion rows mismatch: expected loaded={}, skipped={}; got loaded={}, skipped={}".format(
                expected_rows_loaded, expected_rows_skipped, cand_rows_loaded, cand_rows_skipped)
            task1_msgs.append(msg)
            # award points proportionally by matching values
            score = 0
            if cand_rows_loaded == expected_rows_loaded:
                score += 0.5
            if cand_rows_skipped == expected_rows_skipped:
                score += 0.5
            task1_awarded += int(round(rows_points * score))
    else:
        task1_msgs.append("Answer key ingestion.expected counts missing; cannot grade ingestion rows reliably.")

    # Clamp and record
    task1_awarded = int(min(task1_awarded, WEIGHTS['task1']))
    results['per_task']['task1'] = {
        "points_awarded": task1_awarded,
        "points_available": WEIGHTS['task1'],
        "messages": task1_msgs
    }
    total_awarded += task1_awarded

    # -------------- Task 2: Query & Analysis (40) --------------
    task2_msgs = []
    task2_awarded = 0
    exp_results = safe_get(key_json, ['results'], {})
    cand_results = safe_get(cand_json, ['results'], {})

    # 2.1 total_rows (5)
    exp_total = safe_get(exp_results, ['total_rows'])
    cand_total = safe_get(cand_results, ['total_rows'])
    pts, mx, m = compare_top_level_number(['results', 'total_rows'], exp_total, cand_total, SUB_WEIGHTS['task2']['total_rows'], tol=0)
    task2_awarded += pts
    task2_msgs.extend(m)

    # 2.2 top_hosts_by_avg_cpu (10)
    exp_top = safe_get(exp_results, ['top_hosts_by_avg_cpu'], [])
    cand_top = safe_get(cand_results, ['top_hosts_by_avg_cpu'], [])
    pts, mx, m = compare_top_hosts(exp_top, cand_top, task2_msgs)
    task2_awarded += pts
    task2_msgs.extend(m)

    # 2.3 hourly_stats_for_host (10)
    exp_hourly = safe_get(exp_results, ['hourly_stats_for_host'], [])
    cand_hourly = safe_get(cand_results, ['hourly_stats_for_host'], [])
    pts, mx, m = compare_hourly_stats(exp_hourly, cand_hourly, task2_msgs)
    task2_awarded += pts
    task2_msgs.extend(m)

    # 2.4 cpu_spike_periods (15)
    exp_spikes = safe_get(exp_results, ['cpu_spike_periods'], [])
    cand_spikes = safe_get(cand_results, ['cpu_spike_periods'], [])
    pts, mx, m = compare_cpu_spikes(exp_spikes, cand_spikes, task2_msgs)
    task2_awarded += pts
    task2_msgs.extend(m)

    task2_awarded = int(min(task2_awarded, WEIGHTS['task2']))
    results['per_task']['task2'] = {
        "points_awarded": task2_awarded,
        "points_available": WEIGHTS['task2'],
        "messages": task2_msgs
    }
    total_awarded += task2_awarded

    # -------------- Task 3: Insert & recompute (15) --------------
    task3_msgs = []
    task3_awarded = 0
    exp_after = safe_get(key_json, ['after_insert'], {})
    cand_after = safe_get(cand_json, ['after_insert'], {})

    # 3.1 insert_success and updated_avg_cpu_for_host correctness (10)
    exp_insert_ok = safe_get(exp_after, ['inserted_record_ok'])
    cand_insert_ok = safe_get(cand_after, ['inserted_record_ok'])
    exp_updated_avg = safe_get(exp_after, ['updated_avg_cpu_for_host'])
    cand_updated_avg = safe_get(cand_after, ['updated_avg_cpu_for_host'])
    # Check inserted_record_ok first
    if isinstance(exp_insert_ok, bool):
        if cand_insert_ok is True and exp_insert_ok is True:
            task3_awarded += SUB_WEIGHTS['task3']['insert_success']
        else:
            task3_msgs.append("inserted_record_ok mismatch: expected {}, got {}".format(exp_insert_ok, cand_insert_ok))
            # small partial credit if candidate reports insertion attempted (True/False) but updated_avg correct
            # Continue to check updated_avg below for partial credit
    else:
        task3_msgs.append("Answer key missing after_insert.inserted_record_ok; cannot fully grade insert success.")

    # updated_avg_cpu_for_host check (10 points in spec; but within task3 we have 10 for insert_success and 5 for validation)
    # We'll give full credit for update if numeric and matches expected (tolerance).
    if is_number(exp_updated_avg) and is_number(cand_updated_avg):
        if float_eq(exp_updated_avg, cand_updated_avg, AVG_TOL):
            # award part of insert_success if not already awarded (but don't double count)
            # We already gave insert_success points above. If not, give them here proportionally.
            # For clarity, award the points for correctness now if insertion wasn't awarded earlier.
            if cand_insert_ok is True and exp_insert_ok is True:
                # already counted insert_success full points; give no extra here
                pass
            else:
                # give partial credit for correct updated_avg even if candidate flagged insert incorrectly
                task3_awarded += int(round(SUB_WEIGHTS['task3']['insert_success'] * 0.5))
            # full credit towards updated average correctness is captured by the insert_success bucket above
        else:
            task3_msgs.append("updated_avg_cpu_for_host mismatch: expected {}, got {}".format(exp_updated_avg, cand_updated_avg))
    else:
        task3_msgs.append("updated_avg_cpu_for_host missing or non-numeric in submission.")

    # 3.2 insert_validation / duplicates handling (5)
    # Heuristic: check if ingestion.rows_skipped equals expected and/or ingestion.note mentions 'duplicate'
    cand_rows_skipped = safe_get(cand_json, ['ingestion', 'rows_skipped'])
    exp_rows_skipped = safe_get(key_json, ['ingestion', 'rows_skipped'])
    note = safe_get(cand_json, ['ingestion', 'note'], '') or ''
    opt_errors = safe_get(cand_json, ['optional', 'errors'], []) or []
    dup_mentioned = False
    if isinstance(note, str) and 'duplicate' in note.lower():
        dup_mentioned = True
    else:
        for e in opt_errors:
            if isinstance(e, str) and 'duplicate' in e.lower():
                dup_mentioned = True
                break
    if cand_rows_skipped == exp_rows_skipped and dup_mentioned:
        task3_awarded += SUB_WEIGHTS['task3']['insert_validation']
    else:
        # partial credit if either the counts match or duplicate mention present
        partial = 0.0
        if cand_rows_skipped == exp_rows_skipped:
            partial += 0.5
        if dup_mentioned:
            partial += 0.5
        awarded = int(round(SUB_WEIGHTS['task3']['insert_validation'] * partial))
        task3_awarded += awarded
        if awarded == 0:
            task3_msgs.append("No evidence of duplicate handling in ingestion.note or optional.errors and/or rows_skipped differs from expected.")

    # Clamp and record
    task3_awarded = int(min(task3_awarded, WEIGHTS['task3']))
    results['per_task']['task3'] = {
        "points_awarded": task3_awarded,
        "points_available": WEIGHTS['task3'],
        "messages": task3_msgs
    }
    total_awarded += task3_awarded

    # -------------- Task 4: Documentation & reproducibility (10) --------------
    task4_msgs = []
    task4_awarded = 0
    # 4.1 run_instructions (5) — must be present and match entrypoint
    cand_run = safe_get(cand_json, ['run_instructions'])
    cand_entry = safe_get(cand_json, ['env', 'entrypoint'])
    exp_run = safe_get(key_json, ['run_instructions'])
    ri_points = SUB_WEIGHTS['task4']['run_instructions']
    if isinstance(cand_run, str) and cand_run.strip():
        # Basic check: cand_run should mention the entrypoint or be non-empty
        if cand_entry and (cand_entry in cand_run or cand_run.strip() == exp_run):
            task4_awarded += ri_points
        else:
            # partial credit if non-empty
            task4_awarded += int(round(ri_points * 0.5))
            task4_msgs.append("run_instructions present but does not reference the provided entrypoint; partial credit.")
    else:
        task4_msgs.append("run_instructions missing or empty.")

    # 4.2 schema_description + two suggested improvements (5)
    si_points = SUB_WEIGHTS['task4']['schema_and_improvements']
    schema_desc = safe_get(cand_json, ['schema_and_choices', 'schema_description'])
    suggested = safe_get(cand_json, ['schema_and_choices', 'suggested_improvements'])
    if isinstance(schema_desc, str) and schema_desc.strip() and isinstance(suggested, list) and len(suggested) == 2:
        task4_awarded += si_points
    else:
        # partial credit if either schema_desc present or at least one suggestion
        partial = 0.0
        if isinstance(schema_desc, str) and schema_desc.strip():
            partial += 0.5
        if isinstance(suggested, list) and len(suggested) >= 1:
            partial += 0.5
        task4_awarded += int(round(si_points * partial))
        task4_msgs.append("schema_description or suggested_improvements missing/incomplete; expected 2 suggested improvements.")

    task4_awarded = int(min(task4_awarded, WEIGHTS['task4']))
    results['per_task']['task4'] = {
        "points_awarded": task4_awarded,
        "points_available": WEIGHTS['task4'],
        "messages": task4_msgs
    }
    total_awarded += task4_awarded

    # -------------- Code hygiene (5) --------------
    code_msgs = []
    code_awarded = 0
    # Heuristic: award points if env.language present, env.entrypoint present and run_instructions present
    if safe_get(cand_json, ['env', 'language']) and safe_get(cand_json, ['env', 'entrypoint']) and safe_get(cand_json, ['run_instructions']):
        code_awarded = WEIGHTS['code_hygiene']
    else:
        # partial depending on which are present
        present = 0
        if safe_get(cand_json, ['env', 'language']):
            present += 1
        if safe_get(cand_json, ['env', 'entrypoint']):
            present += 1
        if safe_get(cand_json, ['run_instructions']):
            present += 1
        code_awarded = int(round(WEIGHTS['code_hygiene'] * (present / 3.0)))
        code_msgs.append("Code hygiene heuristics: awarded {} of {} (language/entrypoint/run_instructions presence).".format(code_awarded, WEIGHTS['code_hygiene']))

    results['per_task']['code_hygiene'] = {
        "points_awarded": code_awarded,
        "points_available": WEIGHTS['code_hygiene'],
        "messages": code_msgs
    }
    total_awarded += code_awarded

    # -------------- Summary --------------
    results['total_points'] = total_awarded
    results['max_points'] = sum(WEIGHTS.values())
    pct = (total_awarded / results['max_points']) * 100.0 if results['max_points'] > 0 else 0.0
    results['percentage'] = round(pct, 2)
    results['overall_score'] = round(pct, 2)

    # Consolidate messages for top-level human-readable output
    # Gather all messages from per_task
    all_messages = []
    for k, v in results['per_task'].items():
        msgs = v.get('messages', [])
        if msgs:
            header = "Task {}:".format(k)
            all_messages.append(header)
            for m in msgs:
                all_messages.append("- " + str(m))
    results['messages'] = all_messages

    # Save test_results.json in same directory as script (current working directory)
    out_path = os.path.join(os.getcwd(), "test_results.json")
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print("Grading complete. Results written to {}".format(out_path))
    except Exception as e:
        print("Failed to write results file '{}': {}".format(out_path, e))
        sys.exit(2)

if __name__ == "__main__":
    main()