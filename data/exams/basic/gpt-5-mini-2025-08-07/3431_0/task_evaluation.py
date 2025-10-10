#!/usr/bin/env python3
"""
task_evaluation.py

Automated grader for the basic practical exam (Software Developer - Applications).
Usage:
    python3 task_evaluation.py <candidate_submission.json> <answer_key.json>

Produces:
    test_results.json - detailed grading report in the same directory as this script.

Only uses Python standard library.
"""

import json
import sys
import os
import math

# ---- Helper functions ----

def safe_load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)

def contains_failure_text(text):
    if not text:
        return False
    low = text.lower()
    # heuristics for failure presence
    return ("some tests failed" in low) or ("fail" in low and "ok" not in low) \
           or ("traceback" in low) or ("assertionerror" in low)

def contains_all_pass_text(text):
    if not text:
        return False
    low = text.lower()
    # Either explicit all tests pass or unittest OK at end
    return ("all tests pass" in low) or ("\nok\n" in low) or ("\nok\n" in low) or ("ok\nall tests pass" in low) \
           or (low.strip().endswith("ok")) or ("ok\n" in low and "fail" not in low) or ("ok" in low and "fail" not in low and "error" not in low)

def numeric_improvement(before, after):
    # returns improvement fraction (positive means faster)
    try:
        if before <= 0:
            return None
        return (before - after) / before
    except Exception:
        return None

def text_has_keywords(text, keywords):
    if not text:
        return False
    low = text.lower()
    return any(k.lower() in low for k in keywords)

def approx_equal(a, b, rel_tol=1e-6):
    try:
        return abs(a - b) <= rel_tol * max(1.0, abs(a), abs(b))
    except Exception:
        return False

# ---- Grading configuration (weights) ----
MAX_TOTAL = 100.0
WEIGHTS = {
    'A': 50.0,
    'B': 20.0,
    'C': 20.0,
    'COMM': 10.0
}

# Sub-component weights for Task A (sum to WEIGHTS['A'])
A_SUB = {
    'pre_repro': 5.0,
    'fix_correct': 30.0,
    'minimal_patch': 10.0,
    'explanation': 5.0
}

# Task B sub-weights (sum to WEIGHTS['B'])
B_SUB = {
    'run_demo': 12.0,
    'explain_risks': 8.0
}

# Task C sub-weights (sum to WEIGHTS['C'])
C_SUB = {
    'measure_report': 8.0,
    'meaningful_improve': 8.0,
    'explanation': 4.0
}

# Communication & Packaging subweights (sum to WEIGHTS['COMM'])
COMM_SUB = {
    'json_format': 3.0,
    'patch_clarity': 4.0,
    'run_instructions': 3.0
}

def grade(candidate, answer_key):
    results = {
        'per_task': {},
        'deductions': [],
        'total_points_earned': 0.0,
        'max_points': MAX_TOTAL,
    }

    earned_total = 0.0
    max_total = MAX_TOTAL

    # Basic validation of candidate JSON structure
    if not isinstance(candidate, dict):
        results['error'] = 'Candidate submission is not a JSON object.'
        return results

    # Ensure task_results exists and has three tasks
    task_list = candidate.get('task_results')
    if not isinstance(task_list, list) or len(task_list) < 3:
        results['error'] = 'task_results missing or not an array of length >= 3.'
        return results

    # Create a mapping by task_id for convenience
    tasks_by_id = {}
    for t in task_list:
        tid = t.get('task_id')
        if tid:
            tasks_by_id[tid] = t

    # ---- Task A grading ----
    a_max = WEIGHTS['A']
    a_earned = 0.0
    a_comments = []

    a_task = tasks_by_id.get('A-bugfix')
    if not a_task:
        a_comments.append('Task A entry missing.')
        results['per_task']['A-bugfix'] = {
            'points_earned': 0.0,
            'points_max': a_max,
            'comments': a_comments
        }
    else:
        # 1) Pre-fix reproduction (5)
        pre_out = a_task.get('pre_fix_test_output', '')
        pre_ok = contains_failure_text(pre_out)
        if pre_ok:
            a_earned += A_SUB['pre_repro']
            a_comments.append(f"Pre-fix failing output detected (+{A_SUB['pre_repro']:.1f}).")
        else:
            a_comments.append("Pre-fix failing output not convincingly present (0/5).")

        # 2) Fix correctness (30)
        post_out = a_task.get('post_fix_test_output', '')
        post_ok = contains_all_pass_text(post_out)
        if post_ok:
            a_earned += A_SUB['fix_correct']
            a_comments.append(f"Post-fix tests show passing (+{A_SUB['fix_correct']:.1f}).")
        else:
            a_comments.append("Post-fix tests do not show successful pass (0/30).")

        # 3) Minimal and appropriate patch (10)
        files_modified = a_task.get('files_modified', []) or []
        patch = a_task.get('patch_unified_diff', '') or ''
        # If answer key lists expected files for Task A, prefer that check; else use sensible heuristics
        expected_a_files = None
        try:
            expected_a_files = answer_key.get('task_results', [])
            # attempt to find the A-bugfix entry
            for e in expected_a_files:
                if e.get('task_id') == 'A-bugfix':
                    expected_a_files = e.get('files_modified', [])
                    break
        except Exception:
            expected_a_files = None

        minimal_score = 0.0
        # Check files_modified against expected
        if expected_a_files and isinstance(expected_a_files, list):
            # award full if candidate modified only expected files (subset)
            if set(files_modified).issubset(set(expected_a_files)):
                minimal_score = A_SUB['minimal_patch']
                a_comments.append(f"Files modified are a subset of expected files (+{minimal_score:.1f}).")
            else:
                # partial credit if includes at least one expected file and not many other files
                if any(f in (expected_a_files or []) for f in files_modified):
                    # small diff heuristic
                    if len(files_modified) <= 2:
                        minimal_score = round(A_SUB['minimal_patch'] * 0.7, 2)
                        a_comments.append(f"Modified expected file(s) but also others; partial for patch (+{minimal_score:.1f}).")
                    else:
                        minimal_score = round(A_SUB['minimal_patch'] * 0.4, 2)
                        a_comments.append(f"Modified extra files; small partial credit (+{minimal_score:.1f}).")
                else:
                    minimal_score = 0.0
                    a_comments.append("Files modified do not match expected; no points for minimal patch (0/10).")
        else:
            # No expected list: heuristics
            if files_modified and len(files_modified) <= 2 and any('processor' in f.lower() for f in files_modified):
                minimal_score = A_SUB['minimal_patch']
                a_comments.append(f"Small patch touching processor file(s) considered minimal (+{minimal_score:.1f}).")
            elif patch and ('--- a/' in patch or '+++ b/' in patch or patch.strip().startswith('=== file:')):
                minimal_score = round(A_SUB['minimal_patch'] * 0.8, 2)
                a_comments.append(f"Unified diff present; awarding most of patch score (+{minimal_score:.1f}).")
            elif patch == 'N/A' and files_modified:
                minimal_score = round(A_SUB['minimal_patch'] * 0.5, 2)
                a_comments.append("No unified diff provided but files_modified present; partial credit (+{:.1f}).".format(minimal_score))
            else:
                minimal_score = 0.0
                a_comments.append("No patch info provided; no points for minimal patch (0/10).")

        a_earned += minimal_score

        # 4) Explanation (5)
        explanation = (a_task.get('explanation') or '').strip()
        if explanation:
            # check for root cause keywords
            if text_has_keywords(explanation, ['sort', 'descending', 'reverse', 'ascending', 'order']):
                a_earned += A_SUB['explanation']
                a_comments.append(f"Explanation mentions root cause keywords (+{A_SUB['explanation']:.1f}).")
            else:
                a_earned += round(A_SUB['explanation'] * 0.6, 2)
                a_comments.append(f"Explanation present but missing clear root-cause keywords; partial (+{round(A_SUB['explanation'] * 0.6, 2):.1f}).")
        else:
            a_comments.append("No explanation provided (0/5).")

        # Store task A results
        results['per_task']['A-bugfix'] = {
            'points_earned': round(a_earned, 2),
            'points_max': a_max,
            'comments': a_comments
        }

    earned_total += a_earned

    # If Task A fix is not correct, apply major penalty: cap final possible score to 40% of max
    # (This enforces the guidance that Task A must be fixed to be competent.)
    post_ok_final = False
    a_task = tasks_by_id.get('A-bugfix')
    if a_task:
        post_ok_final = contains_all_pass_text(a_task.get('post_fix_test_output', ''))
    cap_applied = False
    cap_value_points = None
    if not post_ok_final:
        cap_points = round(MAX_TOTAL * 0.4, 2)  # 40% cap
        cap_applied = True
        cap_value_points = cap_points
        # We'll still compute raw earned_total, but at the end we'll cap the final awarded points.
        results['deductions'].append("Task A post-fix tests did not show passing. Per policy, final awarded points will be capped to {:.1f}% of max ({} points).".format(40.0, cap_points))

    # ---- Task B grading ----
    b_max = WEIGHTS['B']
    b_earned = 0.0
    b_comments = []
    b_task = tasks_by_id.get('B-adapt_env')
    if not b_task:
        b_comments.append('Task B entry missing.')
        results['per_task']['B-adapt_env'] = {
            'points_earned': 0.0,
            'points_max': b_max,
            'comments': b_comments
        }
    else:
        # 1) run_demo (12)
        run_output = b_task.get('run_output_after_adaptation', '') or ''
        # Attempt to compare to answer_key expected output for B if present
        expected_run = None
        try:
            # locate in answer key task's expected output if provided
            ak_tr = answer_key.get('task_results', [])
            for e in ak_tr:
                if e.get('task_id') == 'B-adapt_env':
                    expected_run = e.get('run_output_after_adaptation')
                    break
        except Exception:
            expected_run = None

        if expected_run:
            # Normalize whitespace and compare whether candidate output contains all expected non-empty lines
            expected_lines = [l.strip() for l in expected_run.strip().splitlines() if l.strip()]
            got_lines = [l.strip() for l in run_output.strip().splitlines() if l.strip()]
            # Simple matching: all expected lines must appear in candidate output in same order
            match_all = True
            idx = 0
            for el in expected_lines:
                found = False
                while idx < len(got_lines):
                    if got_lines[idx] == el:
                        found = True
                        idx += 1
                        break
                    idx += 1
                if not found:
                    match_all = False
                    break
            if match_all:
                b_earned += B_SUB['run_demo']
                b_comments.append(f"Run output after adaptation matches expected (+{B_SUB['run_demo']:.1f}).")
            else:
                # partial: some expected lines present?
                hits = sum(1 for el in expected_lines if el in run_output)
                if hits > 0:
                    frac = hits / max(1, len(expected_lines))
                    add = round(B_SUB['run_demo'] * frac, 2)
                    b_earned += add
                    b_comments.append(f"Run output contains {hits}/{len(expected_lines)} expected lines; partial credit (+{add:.2f}).")
                else:
                    b_comments.append("Run output does not match expected (0/12).")
        else:
            # No expected in answer key: heuristics: non-empty output that prints product codes is acceptable
            if run_output and any(c.isalpha() for c in run_output):
                b_earned += B_SUB['run_demo']
                b_comments.append(f"Non-empty run output detected (+{B_SUB['run_demo']:.1f}).")
            else:
                b_comments.append("No run output captured (0/12).")

        # 2) explanation + risks (8)
        expl = (b_task.get('explanation') or '').strip()
        risks = (b_task.get('risks_or_limitations') or '').strip()
        if expl and risks:
            b_earned += B_SUB['explain_risks']
            b_comments.append(f"Explanation and risks provided (+{B_SUB['explain_risks']:.1f}).")
        elif expl or risks:
            b_earned += round(B_SUB['explain_risks'] * 0.6, 2)
            b_comments.append("Partial explanation or partial risks provided; partial credit.")
        else:
            b_comments.append("No explanation or risks provided (0/8).")

        results['per_task']['B-adapt_env'] = {
            'points_earned': round(b_earned, 2),
            'points_max': b_max,
            'comments': b_comments
        }

    earned_total += b_earned

    # ---- Task C grading ----
    c_max = WEIGHTS['C']
    c_earned = 0.0
    c_comments = []
    c_task = tasks_by_id.get('C-performance')
    if not c_task:
        c_comments.append('Task C entry missing.')
        results['per_task']['C-performance'] = {
            'points_earned': 0.0,
            'points_max': c_max,
            'comments': c_comments
        }
    else:
        # 1) measure and report (8)
        before = c_task.get('benchmark_before_ms')
        after = c_task.get('benchmark_after_ms')
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            # Accept positive numeric values
            if before > 0 and after >= 0:
                c_earned += C_SUB['measure_report']
                c_comments.append(f"Benchmark numbers present (+{C_SUB['measure_report']:.1f}).")
            else:
                c_comments.append("Benchmark values present but non-positive (0/8 for measure/report).")
        else:
            c_comments.append("Benchmark before/after numbers missing or not numeric (0/8).")

        # 2) meaningful improvement (8)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before > 0:
            improv = numeric_improvement(before, after)
            if improv is None:
                c_comments.append("Could not compute improvement (0/8).")
            else:
                pct = improv * 100.0
                if after < before:
                    # positive improvement
                    if pct >= 15.0:
                        c_earned += C_SUB['meaningful_improve']
                        c_comments.append(f"Improvement {pct:.1f}% >= 15%: full credit (+{C_SUB['meaningful_improve']:.1f}).")
                    elif pct >= 5.0:
                        add = round(C_SUB['meaningful_improve'] * 0.5, 2)
                        c_earned += add
                        c_comments.append(f"Improvement {pct:.1f}% is modest (>=5%); partial credit (+{add:.2f}).")
                    else:
                        c_comments.append(f"Improvement {pct:.1f}% under threshold; no points (0/8).")
                elif approx_equal(after, before):
                    c_comments.append("No measurable improvement (0/8).")
                else:
                    c_comments.append(f"Regression observed ({pct:.1f}% worse); no points (0/8).")
        else:
            c_comments.append("Insufficient numeric benchmark data for improvement calculation (0/8).")

        # 3) explanation (4)
        c_expl = (c_task.get('explanation') or '').strip()
        if c_expl:
            if text_has_keywords(c_expl, ['single-pass', 'dict', 'accumul', 'optimi', 'complexit', 'one-pass', 'linear', 'o(n)']):
                c_earned += C_SUB['explanation']
                c_comments.append(f"Explanation contains optimization details/trade-offs (+{C_SUB['explanation']:.1f}).")
            else:
                c_earned += round(C_SUB['explanation'] * 0.6, 2)
                c_comments.append("Explanation present but lacking clear optimization detail; partial credit.")
        else:
            c_comments.append("No explanation provided for optimization (0/4).")

        results['per_task']['C-performance'] = {
            'points_earned': round(c_earned, 2),
            'points_max': c_max,
            'comments': c_comments
        }

    earned_total += c_earned

    # ---- Communication & Packaging grading (10) ----
    comm_max = WEIGHTS['COMM']
    comm_earned = 0.0
    comm_comments = []

    # 1) JSON format (3) - check for required top-level keys
    required_top = ['candidate_name', 'time_taken_minutes', 'task_results']
    missing_keys = [k for k in required_top if k not in candidate]
    if not missing_keys and isinstance(candidate.get('task_results'), list) and len(candidate.get('task_results')) >= 3:
        comm_earned += COMM_SUB['json_format']
        comm_comments.append(f"test_submission.json contains required top-level keys (+{COMM_SUB['json_format']:.1f}).")
    else:
        comm_comments.append(f"Missing or malformed top-level keys: {missing_keys} (0/{COMM_SUB['json_format']}).")

    # 2) Patch/diff clarity (4)
    any_diff_good = False
    any_files_modified = False
    for t in task_list:
        files_mod = t.get('files_modified', []) or []
        any_files_modified = any_files_modified or bool(files_mod)
        patch = (t.get('patch_unified_diff') or '').strip()
        if patch and ('--- a/' in patch or '+++ b/' in patch or patch.startswith('*** ') or patch.startswith('=== file:')):
            any_diff_good = True
            break
    if any_diff_good:
        comm_earned += COMM_SUB['patch_clarity']
        comm_comments.append(f"Unified diffs found and look like git-format (+{COMM_SUB['patch_clarity']:.1f}).")
    elif any_files_modified:
        # candidate provided modified file list but no unified diff
        comm_earned += round(COMM_SUB['patch_clarity'] * 0.6, 2)
        comm_comments.append("Files_modified present but unified diff not found; partial credit for patch clarity.")
    else:
        comm_comments.append("No patch/diff or files_modified information provided (0/4).")

    # 3) run instructions & logs (3)
    has_run_instructions = False
    # bench command field or pre/post outputs presence
    bench_cmds = []
    for t in task_list:
        if 'bench_command_run' in t and t.get('bench_command_run'):
            bench_cmds.append(t.get('bench_command_run'))
    if bench_cmds and any(isinstance(x, str) and x.strip() for x in bench_cmds):
        has_run_instructions = True
    # also require tests outputs present
    has_test_logs = any(t.get('pre_fix_test_output') or t.get('post_fix_test_output') for t in task_list)
    if has_run_instructions and has_test_logs:
        comm_earned += COMM_SUB['run_instructions']
        comm_comments.append(f"Bench command and test logs present (+{COMM_SUB['run_instructions']:.1f}).")
    elif has_test_logs:
        comm_earned += round(COMM_SUB['run_instructions'] * 0.6, 2)
        comm_comments.append("Test logs present but bench command missing; partial credit.")
    else:
        comm_comments.append("No run instructions or logs present (0/3).")

    results['per_task']['Communication'] = {
        'points_earned': round(comm_earned, 2),
        'points_max': comm_max,
        'comments': comm_comments
    }

    earned_total += comm_earned

    # ---- Final aggregation and cap if necessary ----
    raw_points = earned_total
    final_points = raw_points
    if cap_applied:
        cap_points = cap_value_points
        if final_points > cap_points:
            results['deductions'].append("Applied Task A failure cap: raw points {:.2f} reduced to cap {:.2f}.".format(final_points, cap_points))
            final_points = cap_points

    # Round and compute percentage
    final_points = round(final_points, 2)
    percent = round((final_points / max_total) * 100.0, 2) if max_total > 0 else 0.0

    results['total_points_earned'] = final_points
    results['max_points'] = max_total
    results['percentage'] = percent
    results['overall_score'] = percent  # numeric variable as requested

    # Include raw breakdown and any helpful info
    results['raw_points_before_cap'] = round(raw_points, 2)
    results['cap_applied'] = cap_applied
    if cap_applied:
        results['cap_points'] = cap_value_points

    return results

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 task_evaluation.py <candidate_submission.json> <answer_key.json>")
        sys.exit(2)

    cand_path = sys.argv[1]
    key_path = sys.argv[2]

    cand_json, err = safe_load_json(cand_path)
    if err:
        out = {
            'error': f"Failed to load candidate submission JSON: {err}"
        }
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print("Error loading candidate JSON. Wrote test_results.json with error info.")
        sys.exit(1)

    key_json, err = safe_load_json(key_path)
    if err:
        out = {
            'error': f"Failed to load answer key JSON: {err}"
        }
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print("Error loading answer key JSON. Wrote test_results.json with error info.")
        sys.exit(1)

    # Perform grading
    results = grade(cand_json, key_json)

    # Write results file
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("Grading complete. Results written to test_results.json")
    print("Overall score: {}%".format(results.get('overall_score', 0.0)))

if __name__ == '__main__':
    main()