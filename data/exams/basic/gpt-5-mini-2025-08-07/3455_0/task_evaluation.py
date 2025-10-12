#!/usr/bin/env python3
"""
task_evaluation.py

Usage:
    python3 task_evaluation.py <candidate_test_submission.json> <answer_key.json>

Produces:
    test_results.json in the current working directory.

Author: Automated grader for the System Telemetry Basic Practical Exam.
"""

import json
import sys
import os
from datetime import datetime, timezone, timedelta

# ----------------------
# Helper utilities
# ----------------------

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f), None
    except Exception as e:
        return None, f"Failed to load JSON from {path}: {e}"

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def safe_get(dct, *keys):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def try_parse_iso_to_hour(s):
    """
    Parse an ISO-like timestamp and return a datetime truncated to hour (tz-aware UTC).
    Tolerates:
      - 'Z' suffix (converted to +00:00)
      - timezone offsets like -05:00
      - naive timestamps (interpreted as UTC)
    Returns datetime object (UTC, tzinfo=timezone.utc) truncated to hour, or None on failure.
    """
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    # Normalize 'Z'
    if s.endswith('Z'):
        s2 = s[:-1] + '+00:00'
    else:
        s2 = s
    # If no timezone offset and no explicit offset, treat as naive UTC
    try:
        # datetime.fromisoformat supports offsets like +00:00
        dt = datetime.fromisoformat(s2)
    except Exception:
        # Try to handle if seconds missing or other minor formatting issues
        # Attempt to split timezone manually
        # As a fallback, attempt parsing common forms manually
        try:
            # If there's a space-separated timezone, replace space with T
            s3 = s2.replace(' ', 'T')
            dt = datetime.fromisoformat(s3)
        except Exception:
            return None
    # If dt is naive, set tzinfo=UTC per rules (timestamps without timezone are interpreted as UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # Convert to UTC
    try:
        dt_utc = dt.astimezone(timezone.utc)
    except Exception:
        return None
    # Truncate to hour
    dt_hour = dt_utc.replace(minute=0, second=0, microsecond=0)
    return dt_hour

def normalize_hour_str_for_compare(s):
    """
    Convert a candidate hour string to canonical "YYYY-MM-DDTHH:00:00" by parsing and formatting.
    Returns string or None.
    """
    dt = try_parse_iso_to_hour(s)
    if dt is None:
        return None
    return dt.strftime('%Y-%m-%dT%H:00:00')

def parse_timestamp_to_utc_iso_no_tz(s):
    """
    Parse timestamp and return canonical UTC timestamp string "YYYY-MM-DDTHH:MM:SSZ".
    Returns None if cannot parse.
    """
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    if s.endswith('Z'):
        s2 = s[:-1] + '+00:00'
    else:
        s2 = s
    try:
        dt = datetime.fromisoformat(s2)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

def float_eq(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False

def ensure_list(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]

# ----------------------
# Scoring logic
# ----------------------

def score_unique_hosts(candidate, expected):
    max_points = 5
    reasons = []
    c = safe_get(candidate, 'outputs', 'unique_hosts')
    e = safe_get(expected, 'outputs', 'unique_hosts')
    if c is None:
        reasons.append("unique_hosts missing in candidate outputs.")
        return 0, max_points, reasons
    if e is None:
        reasons.append("unique_hosts missing in answer key.")
        return 0, max_points, reasons
    try:
        if int(c) == int(e):
            return max_points, max_points, ["unique_hosts matches expected."]
        else:
            reasons.append(f"unique_hosts mismatch: candidate={c}, expected={e}.")
            # partial credit if close? No, keep 0 for mismatch per rubric.
            return 0, max_points, reasons
    except Exception:
        reasons.append("unique_hosts not an integer.")
        return 0, max_points, reasons

def score_topk_list(candidate_list, expected_list, k, max_points, key_fields):
    """
    Generic scoring for ordered top-K lists.
    candidate_list and expected_list are lists of dicts.
    key_fields: tuple of fields to compare in order (e.g., ('host','error_count'))
    Scoring: per-position matching both fields => proportional credit.
    Returns (score, max_points, reasons).
    """
    reasons = []
    cand = ensure_list(candidate_list)
    exp = ensure_list(expected_list)
    # Ensure lengths
    if len(cand) < k:
        reasons.append(f"Candidate list has {len(cand)} entries; expected {k}. Missing entries will be treated as mismatches.")
    if len(exp) < k:
        reasons.append(f"Answer key expected list has {len(exp)} entries; expected {k}.")
    matches = 0
    for i in range(k):
        try:
            c_item = cand[i]
        except IndexError:
            c_item = None
        try:
            e_item = exp[i]
        except IndexError:
            e_item = None
        if c_item is None or e_item is None:
            continue
        ok = True
        for f in key_fields:
            if f not in c_item or f not in e_item:
                ok = False
                break
            # allow numeric comparison for counts
            if isinstance(e_item[f], (int, float)):
                try:
                    if int(c_item[f]) != int(e_item[f]):
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
            else:
                # string compare tolerant to whitespace
                if str(c_item[f]).strip() != str(e_item[f]).strip():
                    ok = False
                    break
        if ok:
            matches += 1
        else:
            reasons.append(f"Position {i+1} mismatch: candidate={c_item}, expected={e_item}")
    score = (matches / k) * max_points
    return score, max_points, reasons

def score_hourly_buckets(candidate_buckets, expected_buckets, max_points):
    reasons = []
    cand = ensure_list(candidate_buckets)
    exp = ensure_list(expected_buckets)
    if len(exp) != 24:
        reasons.append(f"Answer key hourly buckets length is {len(exp)} != 24 (unexpected).")
    # We'll compare by normalized hour string and counts.
    matches = 0
    total = len(exp)
    for i, e in enumerate(exp):
        e_hour_raw = e.get('hour')
        e_count = e.get('count')
        # Candidate must have same ordering
        try:
            c = cand[i]
        except Exception:
            reasons.append(f"Missing candidate bucket at position {i}.")
            continue
        c_hour_raw = c.get('hour')
        c_count = c.get('count')
        e_hour_norm = normalize_hour_str_for_compare(e_hour_raw)
        c_hour_norm = normalize_hour_str_for_compare(c_hour_raw)
        if e_hour_norm is None:
            reasons.append(f"Could not parse expected hour '{e_hour_raw}' at index {i}.")
            continue
        if c_hour_norm is None:
            reasons.append(f"Could not parse candidate hour '{c_hour_raw}' at index {i}.")
            continue
        if e_hour_norm != c_hour_norm:
            reasons.append(f"Hour mismatch at position {i}: candidate='{c_hour_norm}', expected='{e_hour_norm}'.")
            continue
        # Compare counts
        try:
            if int(c_count) == int(e_count):
                matches += 1
            else:
                reasons.append(f"Count mismatch for hour {e_hour_norm}: candidate={c_count}, expected={e_count}.")
        except Exception:
            reasons.append(f"Non-integer count at hour {e_hour_norm}.")
    if total == 0:
        score = 0
    else:
        score = (matches / total) * max_points
    return score, max_points, reasons

def score_avg_cpu_by_component(candidate_arr, expected_arr, max_points):
    reasons = []
    cand = ensure_list(candidate_arr)
    exp = ensure_list(expected_arr)
    # Build map of candidate by component
    cand_map = {}
    for item in cand:
        comp = item.get('component')
        val = item.get('avg_cpu')
        if comp is not None:
            try:
                cand_map[str(comp)] = float(val) if val is not None else None
            except Exception:
                cand_map[str(comp)] = None
    matches = 0
    total = len(exp)
    if total == 0:
        reasons.append("No expected components in answer key.")
        return 0, max_points, reasons
    for e in exp:
        comp = e.get('component')
        e_val = e.get('avg_cpu')
        if comp not in cand_map:
            reasons.append(f"Missing component '{comp}' in candidate avg_cpu_by_component.")
            continue
        c_val = cand_map.get(comp)
        if c_val is None and (e_val is None):
            matches += 1
        elif c_val is None and e_val is not None:
            reasons.append(f"Component '{comp}': candidate avg_cpu is NULL but expected {e_val}.")
        else:
            # Compare floats with tolerance
            try:
                if float_eq(c_val, float(e_val), tol=1e-6):
                    matches += 1
                else:
                    reasons.append(f"Component '{comp}': avg_cpu mismatch candidate={c_val}, expected={e_val}.")
            except Exception:
                reasons.append(f"Component '{comp}': could not compare values candidate={c_val}, expected={e_val}.")
    score = (matches / total) * max_points
    return score, max_points, reasons

def score_topk_event_ids(candidate_list, expected_list, max_points):
    # Similar to topk scoring but for event_id and count; expected_list length may be 10
    k = len(expected_list) if isinstance(expected_list, list) else 0
    if k == 0:
        return 0, max_points, ["Answer key has empty top10_event_ids."]
    return score_topk_list(candidate_list, expected_list, k, max_points, ('event_id','count'))

def score_error_csv_sample(candidate_rows, expected_rows, max_points):
    """
    Compare sample rows up to expected length. Each row expected to have:
    timestamp, host, component, event_id, message, metric_value
    Tolerant timestamp parsing and numeric comparison for metric_value.
    Score proportional to number of matching rows.
    """
    reasons = []
    cand = ensure_list(candidate_rows)
    exp = ensure_list(expected_rows)
    if len(exp) == 0:
        return 0, max_points, ["No expected sample rows provided in answer key."]
    matches = 0
    total = len(exp)
    for i, e in enumerate(exp):
        try:
            c = cand[i]
        except Exception:
            reasons.append(f"Missing candidate sample row at position {i}.")
            continue
        # Compare fields
        ok = True
        # timestamp: accept either Z or +00:00 variations; compare by normalized UTC timestamp
        e_ts = e.get('timestamp')
        c_ts = c.get('timestamp')
        e_ts_norm = parse_timestamp_to_utc_iso_no_tz(e_ts)
        c_ts_norm = parse_timestamp_to_utc_iso_no_tz(c_ts)
        if e_ts_norm is None or c_ts_norm is None:
            reasons.append(f"Could not parse timestamps for sample row {i}: candidate='{c_ts}', expected='{e_ts}'.")
            ok = False
        else:
            if e_ts_norm != c_ts_norm:
                reasons.append(f"Timestamp mismatch at sample row {i}: candidate='{c_ts_norm if (c_ts_norm:=c_ts_norm) else c_ts}', expected='{e_ts_norm}'.")
                ok = False
        # host/component/event_id/message exact string compare tolerant to whitespace
        for fld in ('host','component','event_id','message'):
            if str(c.get(fld, '')).strip() != str(e.get(fld, '')).strip():
                reasons.append(f"Sample row {i} field '{fld}' mismatch: candidate='{c.get(fld)}', expected='{e.get(fld)}'.")
                ok = False
        # metric_value numeric or null
        e_mv = e.get('metric_value')
        c_mv = c.get('metric_value')
        if e_mv is None:
            if c_mv is not None:
                reasons.append(f"Sample row {i} metric_value expected NULL but candidate has {c_mv}.")
                ok = False
        else:
            try:
                if not float_eq(float(c_mv), float(e_mv), tol=1e-6):
                    reasons.append(f"Sample row {i} metric_value mismatch: candidate={c_mv}, expected={e_mv}.")
                    ok = False
            except Exception:
                reasons.append(f"Sample row {i} metric_value parse error: candidate={c_mv}, expected={e_mv}.")
                ok = False
        if ok:
            matches += 1
    score = (matches / total) * max_points
    return score, max_points, reasons

def score_summary_report(candidate, expected, max_points):
    """
    summary_report.json should contain top_hosts (5 items) and hourly_errors (24 items).
    We'll compare components already scored: top5 and hourly. Here, we compute matches across both arrays combined.
    """
    reasons = []
    cand_top = safe_get(candidate, 'outputs', 'top5_hosts_error') or []
    exp_top = safe_get(expected, 'outputs', 'top5_hosts_error') or []
    cand_hour = safe_get(candidate, 'outputs', 'hourly_errors_last24') or []
    exp_hour = safe_get(expected, 'outputs', 'hourly_errors_last24') or []
    # Combine counts
    total_elements = len(exp_top) + len(exp_hour)
    if total_elements == 0:
        return 0, max_points, ["Answer key has no summary elements."]
    matches = 0
    # top hosts: position-wise comparison of host and error_count
    for i, e in enumerate(exp_top):
        try:
            c = cand_top[i]
        except Exception:
            continue
        if str(c.get('host','')).strip() == str(e.get('host','')).strip():
            try:
                if int(c.get('error_count')) == int(e.get('error_count')):
                    matches += 1
                else:
                    reasons.append(f"Summary top_host count mismatch at pos {i}: candidate={c.get('error_count')}, expected={e.get('error_count')}")
            except Exception:
                reasons.append(f"Summary top_host count parse error at pos {i}.")
        else:
            reasons.append(f"Summary top_host mismatch at pos {i}: candidate={c.get('host')}, expected={e.get('host')}")
    # hourly: compare normalized hours and counts; assume same ordering
    for i, e in enumerate(exp_hour):
        try:
            c = cand_hour[i]
        except Exception:
            continue
        e_hour_norm = normalize_hour_str_for_compare(e.get('hour'))
        c_hour_norm = normalize_hour_str_for_compare(c.get('hour'))
        if e_hour_norm is None or c_hour_norm is None:
            reasons.append(f"Could not parse hours in summary at index {i}.")
            continue
        if e_hour_norm != c_hour_norm:
            reasons.append(f"Summary hour mismatch at pos {i}: candidate='{c_hour_norm}', expected='{e_hour_norm}'.")
            continue
        try:
            if int(c.get('count')) == int(e.get('count')):
                matches += 1
            else:
                reasons.append(f"Summary hourly count mismatch at {e_hour_norm}: candidate={c.get('count')}, expected={e.get('count')}.")
        except Exception:
            reasons.append(f"Summary hourly count parse error at pos {i}.")
    score = (matches / total_elements) * max_points
    return score, max_points, reasons

def score_ingestion_and_storage(candidate, expected, max_points=30):
    """
    Best-effort checks using presence of files_produced, notes content, and data-driven signals
    (e.g., top10_event_ids all 1 indicates dedup & uniqueness).
    This is heuristic since we don't inspect the DB file here.
    Breakdown internal:
      - DB file present and non-empty: up to 8 points
      - Notes mention schema and uniqueness: up to 6 points
      - Duplicates removed correctly: 5 points (uses top10_event_ids & notes)
      - Malformed timestamps handled: 5 points (notes must state count equals expected)
      - Single-command run / files present: 6 points (end-to-end)
    """
    reasons = []
    cand = candidate
    exp = expected
    score = 0.0
    # Requirements
    files = ensure_list(safe_get(candidate, 'files_produced') or [])
    files_map = { os.path.basename(f.get('file')): f for f in files if isinstance(f, dict) and 'file' in f }
    # 1) DB file present and non-empty (8)
    db_part = 8
    db_file = files_map.get('system_logs.db')
    if db_file and isinstance(db_file.get('bytes', None), int) and db_file.get('bytes', 0) > 0:
        score += db_part
        reasons.append("system_logs.db present and non-empty.")
    else:
        reasons.append("system_logs.db missing or empty in files_produced.")
    # 2) Notes mention schema and event_id uniqueness (6)
    notes = safe_get(candidate, 'notes') or ""
    notes_lower = notes.lower()
    notes_part = 6
    if 'event_id' in notes_lower and ('unique' in notes_lower or 'unique' in notes):
        score += notes_part
        reasons.append("Notes mention event_id uniqueness.")
    else:
        reasons.append("Notes do not clearly mention event_id uniqueness.")
    # 3) Duplicates removed correctly (5)
    dup_part = 5
    # Heuristic: expected top10_event_ids all counts == 1 per answer key indicates dedup enforced.
    exp_top10 = safe_get(exp, 'outputs', 'top10_event_ids') or []
    cand_top10 = safe_get(cand, 'outputs', 'top10_event_ids') or []
    dedup_ok = True
    # If candidate top10 exists, check if all counts == 1 (suggests uniqueness)
    if ensure_list(cand_top10):
        try:
            for item in cand_top10:
                if int(item.get('count', 0)) != 1:
                    dedup_ok = False
                    break
        except Exception:
            dedup_ok = False
        if dedup_ok:
            score += dup_part
            reasons.append("Candidate top10_event_ids shows unique counts indicating dedup applied.")
        else:
            reasons.append("Candidate top10_event_ids suggests duplicate event_id counts present.")
    else:
        reasons.append("Candidate top10_event_ids missing; cannot infer dedup handling.")
    # 4) Malformed timestamps handled (5)
    malformed_part = 5
    # Answer key often mentions how many malformed dropped; try to find "malformed" and a number in notes
    exp_malformed = None
    # try to find in expected notes phrase like 'Malformed timestamps dropped: 2' or 'malformed timestamps dropped: 2'
    exp_notes = safe_get(exp, 'notes') or ""
    # attempt to extract integer from expected notes if present
    import re
    m = re.search(r"malformed.*?(\d+)", exp_notes, re.IGNORECASE)
    if m:
        try:
            exp_malformed = int(m.group(1))
        except:
            exp_malformed = None
    # Extract candidate reported malformed number
    m2 = re.search(r"malformed.*?(\d+)", notes, re.IGNORECASE)
    cand_malformed = None
    if m2:
        try:
            cand_malformed = int(m2.group(1))
        except:
            cand_malformed = None
    if exp_malformed is not None and cand_malformed is not None:
        if cand_malformed == exp_malformed:
            score += malformed_part
            reasons.append(f"Notes report malformed timestamps dropped = {cand_malformed}, matches answer key.")
        else:
            reasons.append(f"Notes report malformed timestamps dropped = {cand_malformed}, expected {exp_malformed}.")
    else:
        # If we can't find numbers, give partial credit if notes mention 'malformed' and 'dropped'
        if 'malformed' in notes_lower and ('drop' in notes_lower or 'dropped' in notes_lower):
            score += (malformed_part * 0.5)
            reasons.append("Notes mention malformed timestamps being dropped (partial credit).")
        else:
            reasons.append("Notes do not state malformed timestamp handling.")
    # 5) Single-command run / files present (6)
    run_part = 6
    run_cmd = safe_get(candidate, 'run_command')
    required_files = ['system_logs.db','error_events_7days.csv','summary_report.json','test_submission.json']
    missing_files = [f for f in required_files if f not in files_map]
    if run_cmd and isinstance(run_cmd, str) and run_cmd.strip():
        if not missing_files:
            score += run_part
            reasons.append("run_command provided and all required output files listed in files_produced.")
        else:
            # partial if run_command present but missing files
            score += run_part * 0.5
            reasons.append(f"run_command provided but some required files missing in files_produced: {missing_files}")
    else:
        reasons.append("run_command missing or empty.")
    return score, max_points, reasons

def score_code_quality(candidate, expected, max_points=10):
    """
    Heuristics: check run_command presence, language present, test_submission listed, and notes length <=200
    """
    reasons = []
    score = 0.0
    # run_command non-empty: 3 pts
    run_cmd = safe_get(candidate, 'run_command')
    if run_cmd and isinstance(run_cmd, str) and run_cmd.strip():
        score += 3
        reasons.append("run_command present.")
    else:
        reasons.append("run_command missing or empty.")
    # files_produced includes test_submission.json (2 pts)
    files = ensure_list(safe_get(candidate, 'files_produced') or [])
    files_map = { os.path.basename(f.get('file')): f for f in files if isinstance(f, dict) and 'file' in f }
    if 'test_submission.json' in files_map:
        score += 2
        reasons.append("test_submission.json listed in files_produced.")
    else:
        reasons.append("test_submission.json not listed in files_produced.")
    # language field present (2 pts)
    lang = safe_get(candidate, 'language')
    if lang and isinstance(lang, str) and lang.strip():
        score += 2
        reasons.append(f"language field present: {lang}")
    else:
        reasons.append("language field missing or empty.")
    # notes length <=200 (3 pts)
    notes = safe_get(candidate, 'notes') or ""
    # count words or simply characters? Requirement is <=200 words. We'll approximate by splitting whitespace and count words.
    words = len(notes.split())
    if words <= 200 and words > 0:
        score += 3
        reasons.append(f"notes provided with {words} words (<=200).")
    elif words == 0:
        reasons.append("notes empty.")
    else:
        reasons.append(f"notes too long ({words} words > 200).")
    return score, max_points, reasons

def score_documentation(candidate, expected, max_points=5):
    """
    Check notes includes required elements:
      - mention of deduplication ('duplicate'/'dedup')
      - mention of event_id collision handling ('event_id' & 'earliest' or 'kept earliest')
      - mention of timezone/UTC handling ('UTC' or 'timezone' or 'converted to UTC')
    Award proportionally.
    """
    notes = safe_get(candidate, 'notes') or ""
    notes_low = notes.lower()
    reasons = []
    checks = [
        ('duplicate', ['duplicate', 'dedup', 'deduplicate']),
        ('event_id_earliest', ['event_id', 'earliest', 'keep the earliest', 'kept the earliest']),
        ('utc_timezone', ['utc', 'timezone', 'converted to utc', 'interpreted as utc'])
    ]
    matched = 0
    for name, tokens in checks:
        ok = False
        for t in tokens:
            if t in notes_low:
                ok = True
                break
        if ok:
            matched += 1
            reasons.append(f"Notes mention '{name}'.")
        else:
            reasons.append(f"Notes do not clearly mention '{name}'.")
    score = (matched / len(checks)) * max_points
    return score, max_points, reasons

# ----------------------
# Main grading orchestration
# ----------------------

def grade(candidate_json, answer_key_json):
    """
    Returns a dictionary with detailed scoring and overall_score numeric.
    """
    results = {}
    total_points = 100.0
    breakdown = []

    # 1) Ingestion & Storage (30)
    ingestion_score, ingestion_max, ingestion_reasons = score_ingestion_and_storage(candidate_json, answer_key_json, max_points=30)
    breakdown.append({
        "section": "Ingestion & Storage",
        "score": round(ingestion_score, 3),
        "max_score": ingestion_max,
        "reasons": ingestion_reasons
    })

    # 2) Queries & Analysis (40)
    q_reasons = []
    q_score = 0.0
    # unique_hosts (5)
    s, m, r = score_unique_hosts(candidate_json, answer_key_json)
    q_score += s
    q_reasons += ["unique_hosts:"] + r
    breakdown.append({
        "section": "Queries & Analysis: unique_hosts",
        "score": round(s,3),
        "max_score": m,
        "reasons": r
    })
    # top5_hosts_error (10)
    cand_top5 = safe_get(candidate_json, 'outputs', 'top5_hosts_error') or []
    exp_top5 = safe_get(answer_key_json, 'outputs', 'top5_hosts_error') or []
    s, m, r = score_topk_list(cand_top5, exp_top5, 5, 10, ('host','error_count'))
    q_score += s
    breakdown.append({
        "section": "Queries & Analysis: top5_hosts_error",
        "score": round(s,3),
        "max_score": m,
        "reasons": r
    })
    # hourly_errors_last24 (10)
    s, m, r = score_hourly_buckets(safe_get(candidate_json, 'outputs', 'hourly_errors_last24') or [],
                                   safe_get(answer_key_json, 'outputs', 'hourly_errors_last24') or [],
                                   10)
    q_score += s
    breakdown.append({
        "section": "Queries & Analysis: hourly_errors_last24",
        "score": round(s,3),
        "max_score": m,
        "reasons": r
    })
    # avg_cpu_by_component (10)
    s, m, r = score_avg_cpu_by_component(safe_get(candidate_json, 'outputs', 'avg_cpu_by_component') or [],
                                         safe_get(answer_key_json, 'outputs', 'avg_cpu_by_component') or [],
                                         10)
    q_score += s
    breakdown.append({
        "section": "Queries & Analysis: avg_cpu_by_component",
        "score": round(s,3),
        "max_score": m,
        "reasons": r
    })
    # top10_event_ids (5)
    s, m, r = score_topk_event_ids(safe_get(candidate_json, 'outputs', 'top10_event_ids') or [],
                                   safe_get(answer_key_json, 'outputs', 'top10_event_ids') or [],
                                   5)
    q_score += s
    breakdown.append({
        "section": "Queries & Analysis: top10_event_ids",
        "score": round(s,3),
        "max_score": m,
        "reasons": r
    })

    # Append aggregate Queries & Analysis summary
    breakdown.append({
        "section": "Queries & Analysis: subtotal",
        "score": round(q_score,3),
        "max_score": 40,
        "reasons": ["Aggregate of unique_hosts, top5, hourly, avg_cpu, top10"]
    })

    # 3) Exports & Reporting (15)
    exp_reasons = []
    exp_score = 0.0
    # error_events_7days.csv sample (8)
    s, m, r = score_error_csv_sample(safe_get(candidate_json, 'sample_error_csv_rows') or [],
                                     safe_get(answer_key_json, 'sample_error_csv_rows') or [],
                                     8)
    exp_score += s
    breakdown.append({
        "section": "Exports & Reporting: error_events_7days.csv (sample)",
        "score": round(s,3),
        "max_score": m,
        "reasons": r
    })
    # summary_report.json (7)
    s, m, r = score_summary_report(candidate_json, answer_key_json, 7)
    exp_score += s
    breakdown.append({
        "section": "Exports & Reporting: summary_report.json",
        "score": round(s,3),
        "max_score": m,
        "reasons": r
    })
    breakdown.append({
        "section": "Exports & Reporting: subtotal",
        "score": round(exp_score,3),
        "max_score": 15,
        "reasons": ["Aggregate of CSV sample and summary JSON"]
    })

    # 4) Code quality & reproducibility (10)
    cq_score, cq_max, cq_reasons = score_code_quality(candidate_json, answer_key_json, max_points=10)
    breakdown.append({
        "section": "Code quality & reproducibility",
        "score": round(cq_score,3),
        "max_score": cq_max,
        "reasons": cq_reasons
    })

    # 5) Documentation & rationale (5)
    doc_score, doc_max, doc_reasons = score_documentation(candidate_json, answer_key_json, max_points=5)
    breakdown.append({
        "section": "Documentation & rationale",
        "score": round(doc_score,3),
        "max_score": doc_max,
        "reasons": doc_reasons
    })

    # Total up scores
    total_awarded = 0.0
    total_possible = 0.0
    for b in breakdown:
        total_awarded += float(b.get('score', 0.0))
        total_possible += float(b.get('max_score', 0.0))
    # Some breakdown entries include subtotals and per-sub items; ensure we don't double-count.
    # We set total_possible to 100 explicitly for consistency with rubric.
    total_possible = 100.0
    # Sum of per-section may not equal 100 because we appended subtotals; compute overall by summing the main scoring components we intended:
    # To avoid mis-sum due to subtotals, recompute overall_score by summing the intended pieces:
    # Intended structure:
    # - ingestion_score (30)
    # - q_score (40)
    # - exp_score (15)
    # - cq_score (10)
    # - doc_score (5)
    total_awarded = round(ingestion_score + q_score + exp_score + cq_score + doc_score, 6)
    overall_pct = (total_awarded / total_possible) * 100.0 if total_possible > 0 else 0.0

    results = {
        "breakdown": breakdown,
        "summary": {
            "ingestion_and_storage": round(ingestion_score,3),
            "queries_and_analysis": round(q_score,3),
            "exports_and_reporting": round(exp_score,3),
            "code_quality_and_reproducibility": round(cq_score,3),
            "documentation_and_rationale": round(doc_score,3)
        },
        "total_awarded": round(total_awarded,3),
        "total_possible": total_possible,
        "percentage": round(overall_pct,3),
        "overall_score": round(overall_pct,3)
    }
    return results

# ----------------------
# CLI entrypoint
# ----------------------

def main(argv):
    if len(argv) != 3:
        print("Usage: python3 task_evaluation.py <candidate_test_submission.json> <answer_key.json>")
        sys.exit(2)
    cand_path = argv[1]
    key_path = argv[2]
    cand_json, err = load_json(cand_path)
    if err:
        print(err)
        sys.exit(1)
    key_json, err = load_json(key_path)
    if err:
        print(err)
        sys.exit(1)
    # Grade
    try:
        results = grade(cand_json, key_json)
    except Exception as e:
        print(f"Unexpected error during grading: {e}")
        # produce a minimal error results file
        results = {
            "error": str(e)
        }
    # Write test_results.json next to this script (current working directory)
    out_path = os.path.join(os.getcwd(), 'test_results.json')
    try:
        save_json(results, out_path)
        print(f"Grading complete. Results written to {out_path}")
    except Exception as e:
        print(f"Failed to write results to {out_path}: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main(sys.argv)