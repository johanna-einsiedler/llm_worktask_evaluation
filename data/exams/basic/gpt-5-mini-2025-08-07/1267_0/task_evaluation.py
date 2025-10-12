# task_evaluation.py
"""
Automated grading script for the Debug & Fix practical exam (Basic).

Usage:
    python task_evaluation.py path/to/test_submission.json path/to/answer_key.json

This script compares the candidate's submission JSON to the provided answer key JSON,
applies the grading logic described in the evaluator guidelines, and writes a detailed
result file test_results.json (next to this script).

Output (test_results.json) includes:
 - per-task awarded points and explanations
 - explanation & reproducibility score breakdown
 - code hygiene & minimality score
 - total points, maximum points, percentage
 - overall_score numeric variable (0..100)
"""

import json
import sys
import os
import re
from pathlib import Path

# Scoring configuration (per evaluator guidelines)
TASK_POINTS = {1: 15, 2: 20, 3: 20, 4: 15}  # automated correctness points per task
AUTOMATED_TOTAL = sum(TASK_POINTS.values())  # 70
EXPLANATION_TOTAL = 20
HYGIENE_TOTAL = 10
MAX_TOTAL = AUTOMATED_TOTAL + EXPLANATION_TOTAL + HYGIENE_TOTAL  # 100

# Helper utilities -----------------------------------------------------------

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid JSON or unreadable file '{path}': {e}")

def find_task_pass_in_text(text, task_id):
    """
    Search for a line like 'Task1: PASS' (case-insensitive) in given text.
    Returns True if a PASS for that task is found, False otherwise.
    """
    if not text:
        return False
    # Look for "Task{n}:" then whether "PASS" appears on same line
    pattern = re.compile(rf"Task\s*{task_id}\s*:\s*(PASS|FAIL)", re.IGNORECASE)
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            return m.group(1).strip().upper() == "PASS"
    return False

def tokenize_short(s):
    """Lowercase, split on non-alphanumeric, remove short tokens."""
    if not s:
        return set()
    parts = re.split(r'\W+', s.lower())
    tokens = {p for p in parts if len(p) >= 3}
    return tokens

def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0

def safe_get_task(candidate, tid):
    """
    Retrieve the task object for task_id from candidate['task_results'].
    Returns None if not present or malformed.
    """
    tr = candidate.get("task_results")
    if not isinstance(tr, list):
        return None
    for t in tr:
        try:
            if int(t.get("task_id", -1)) == tid:
                return t
        except Exception:
            continue
    return None

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

# Main grading logic ---------------------------------------------------------

def grade(candidate_json, answer_json):
    results = {
        "per_task": {},
        "explanation_and_repro": {},
        "hygiene": {},
        "total_points": 0.0,
        "max_points": float(MAX_TOTAL),
        "percentage": 0.0,
        "overall_score": 0.0,
        "messages": []
    }

    # Basic validation of loaded JSON structures
    if not isinstance(candidate_json, dict):
        raise ValueError("Candidate submission JSON must be an object at top level.")
    if not isinstance(answer_json, dict):
        raise ValueError("Answer key JSON must be an object at top level.")

    # 1) Automated task correctness scoring (70 points)
    automated_awarded = 0.0
    task_details_msgs = []

    # Prefer final_test_output in candidate as authoritative place to look for PASS lines.
    candidate_final_output = candidate_json.get("final_test_output", "")
    answer_final_output = answer_json.get("final_test_output", "")

    for tid, max_pts in TASK_POINTS.items():
        task_score = 0.0
        detail = {"task_id": tid, "max_points": max_pts, "awarded_points": 0.0, "notes": []}

        # Determine PASS/FAIL from candidate final_test_output or task how_tested
        passed = find_task_pass_in_text(candidate_final_output, tid)

        # Fallback: check the task-specific how_tested field
        if not passed:
            task_obj = safe_get_task(candidate_json, tid)
            if task_obj:
                ht = task_obj.get("how_tested", "")
                if isinstance(ht, str) and find_task_pass_in_text(ht, tid):
                    passed = True

        if passed:
            task_score = float(max_pts)
            detail["awarded_points"] = task_score
            detail["notes"].append(f"Task{tid}: PASS detected in candidate's submitted outputs.")
        else:
            # Task failed according to candidate-provided outputs. Consider partial credit.
            # Partial credit rule: if candidate correctly diagnosed root cause & provided reasonable change_summary,
            # award up to 50% of task points based on similarity to expected root cause/change_summary.
            # We need answer key task expected text to compare.
            ans_task = safe_get_task(answer_json, tid)
            cand_task = safe_get_task(candidate_json, tid)

            if not cand_task:
                detail["notes"].append("No task entry found in candidate submission; no partial credit awarded.")
                task_score = 0.0
            else:
                # Compare root_cause and change_summary similarity against answer key (if available).
                ans_root = ans_task.get("root_cause", "") if ans_task else ""
                ans_change = ans_task.get("change_summary", "") if ans_task else ""

                cand_root = cand_task.get("root_cause", "") or ""
                cand_change = cand_task.get("change_summary", "") or ""

                tok_ans_root = tokenize_short(ans_root)
                tok_ans_change = tokenize_short(ans_change)
                tok_cand_root = tokenize_short(cand_root)
                tok_cand_change = tokenize_short(cand_change)

                sim_root = jaccard(tok_ans_root, tok_cand_root)
                sim_change = jaccard(tok_ans_change, tok_cand_change)
                avg_sim = (sim_root + sim_change) / 2.0

                # Cap partial at 50% of max_pts.
                partial_proportion = clamp(avg_sim * 0.5, 0.0, 0.5)
                task_score = round(max_pts * partial_proportion, 2)

                detail["notes"].append(
                    f"Task{tid}: reported FAIL (no PASS line). "
                    f"Similarity root:{sim_root:.2f} change:{sim_change:.2f} avg:{avg_sim:.2f}. "
                    f"Partial proportion applied: {partial_proportion:.3f}."
                )

                if task_score > 0:
                    detail["notes"].append(f"Awarded partial credit: {task_score}/{max_pts} for diagnostic explanations.")
                else:
                    detail["notes"].append("No partial credit awarded (low similarity / missing explanations).")

        automated_awarded += task_score
        detail["awarded_points"] = task_score
        results["per_task"][f"Task{tid}"] = detail

    # 2) Explanation & reproducibility scoring (20 points)
    # Split into two subparts: (A) Clear root_cause & change_summary for each task (10 pts),
    # (B) test runner output & reproduction steps included (10 pts).
    expl_awarded = 0.0
    expl_notes = {"per_task": {}, "summary": []}

    per_task_quota = 10.0 / 4.0  # 2.5 per task for part A
    per_task_quota_b = 10.0 / 4.0  # 2.5 per task for part B

    partA_awarded = 0.0
    partB_awarded = 0.0

    for tid in (1, 2, 3, 4):
        t_obj = safe_get_task(candidate_json, tid)
        a_score = 0.0
        b_score = 0.0
        notes = []

        # Part A: root_cause & change_summary presence/quality
        if t_obj:
            root = (t_obj.get("root_cause") or "")
            change = (t_obj.get("change_summary") or "")
            # Heuristic: require non-empty & some length to count as "clear"
            if root.strip() and change.strip():
                # length-based quality
                if len(root.strip()) >= 20 and len(change.strip()) >= 20:
                    a_score = per_task_quota
                    notes.append("Both root_cause and change_summary present and sufficiently detailed.")
                else:
                    a_score = per_task_quota * 0.5
                    notes.append("root_cause/change_summary present but short; awarded partial of this subcomponent.")
            else:
                a_score = 0.0
                notes.append("Missing or empty root_cause/change_summary; no points for this subcomponent.")
        else:
            notes.append("Task entry missing in submission; no points for explanation subcomponent.")

        # Part B: how_tested presence & contains useful info (command and task lines)
        if t_obj:
            how = (t_obj.get("how_tested") or "")
            hw = how.lower()
            # check presence of a run command (python or ./run_tests) and reference to TaskX line
            has_command = ("run_tests" in hw) or ("python" in hw) or ("./" in hw)
            has_task_line = bool(re.search(rf"task\s*{tid}\s*:", how, flags=re.IGNORECASE))
            if has_command and has_task_line:
                b_score = per_task_quota_b
                notes.append("how_tested includes run command and Task result lines.")
            elif has_task_line:
                b_score = per_task_quota_b * 0.6
                notes.append("how_tested contains Task result lines but run command missing or unclear.")
            elif has_command:
                b_score = per_task_quota_b * 0.4
                notes.append("how_tested contains run command but Task result lines not found.")
            else:
                b_score = 0.0
                notes.append("how_tested missing or does not include run command/Task lines.")
        else:
            notes.append("Task entry missing in submission; no points for how_tested subcomponent.")

        partA_awarded += a_score
        partB_awarded += b_score
        expl_notes["per_task"][f"Task{tid}"] = {
            "explanation_points_awarded": round(a_score, 2),
            "repro_points_awarded": round(b_score, 2),
            "notes": notes
        }

    expl_awarded = partA_awarded + partB_awarded
    results["explanation_and_repro"]["awarded_points"] = round(expl_awarded, 2)
    results["explanation_and_repro"]["max_points"] = EXPLANATION_TOTAL
    results["explanation_and_repro"]["details"] = expl_notes

    # 3) Code hygiene & minimality (10 points)
    # Heuristic: Compare files_changed lists against expected files_changed in answer key.
    hygiene_awarded = 0.0
    hygiene_notes = []

    # Collect expected files from answer key (if present)
    expected_files = set()
    ans_task_list = answer_json.get("task_results") if isinstance(answer_json.get("task_results"), list) else []
    for t in ans_task_list:
        fl = t.get("files_changed")
        if isinstance(fl, list):
            for f in fl:
                expected_files.add(str(f))

    # Candidate files changed
    candidate_files = set()
    cand_task_list = candidate_json.get("task_results") if isinstance(candidate_json.get("task_results"), list) else []
    for t in cand_task_list:
        fl = t.get("files_changed")
        if isinstance(fl, list):
            for f in fl:
                candidate_files.add(str(f))

    # If no candidate files listed at all, that's a red flag; award 0 but note it.
    if not candidate_files:
        hygiene_awarded = 0.0
        hygiene_notes.append("No files_changed listed by candidate; cannot verify minimality. No points awarded.")
    else:
        # Extra files = candidate_files - expected_files
        if expected_files:
            extra_files = candidate_files - expected_files
            # If all candidate files are subset of expected and number reasonable -> full points
            if candidate_files.issubset(expected_files) and len(candidate_files) <= max(1, len(expected_files) + 1):
                hygiene_awarded = float(HYGIENE_TOTAL)
                hygiene_notes.append("Files changed are within expected set; awarded full hygiene points.")
            else:
                # penalize proportionally to number of extra files
                num_extra = len(extra_files)
                proportion_kept = 1.0 - (num_extra / max(1, len(candidate_files)))
                # Ensure proportion_kept >= 0
                proportion_kept = clamp(proportion_kept, 0.0, 1.0)
                hygiene_awarded = round(HYGIENE_TOTAL * proportion_kept, 2)
                hygiene_notes.append(
                    f"Candidate changed {len(candidate_files)} files; {num_extra} file(s) not in expected set. "
                    f"Partial hygiene score: {hygiene_awarded}/{HYGIENE_TOTAL}."
                )
                if extra_files:
                    hygiene_notes.append(f"Extra files: {sorted(list(extra_files))}")
        else:
            # No expected file list provided in answer key; use heuristic based on number of files changed
            if len(candidate_files) <= 4:
                hygiene_awarded = float(HYGIENE_TOTAL)
                hygiene_notes.append("Answer key does not list expected modified files; assumed small changes -> full points.")
            elif len(candidate_files) <= 8:
                hygiene_awarded = round(HYGIENE_TOTAL * 0.5, 2)
                hygiene_notes.append("Many files changed; awarded partial hygiene points.")
            else:
                hygiene_awarded = 0.0
                hygiene_notes.append("Too many files changed; hygiene points = 0.")

    results["hygiene"]["awarded_points"] = round(hygiene_awarded, 2)
    results["hygiene"]["max_points"] = HYGIENE_TOTAL
    results["hygiene"]["notes"] = hygiene_notes
    results["hygiene"]["candidate_files_changed"] = sorted(list(candidate_files))
    results["hygiene"]["expected_files"] = sorted(list(expected_files))

    # 4) Totals
    total_awarded = round(automated_awarded + expl_awarded + hygiene_awarded, 2)
    results["total_points"] = total_awarded
    results["max_points"] = float(MAX_TOTAL)
    pct = (total_awarded / MAX_TOTAL) * 100.0 if MAX_TOTAL > 0 else 0.0
    results["percentage"] = round(pct, 2)
    results["overall_score"] = round(pct, 2)

    # Add per-task automated scores into results summary
    # Also include the automated_awarded subtotal
    results["automated_subtotal"] = round(automated_awarded, 2)
    results["automated_max"] = AUTOMATED_TOTAL
    results["messages"].append(
        f"Automated correctness subtotal: {automated_awarded}/{AUTOMATED_TOTAL}."
    )
    results["messages"].append(
        f"Explanation & reproducibility subtotal: {round(expl_awarded,2)}/{EXPLANATION_TOTAL}."
    )
    results["messages"].append(
        f"Code hygiene subtotal: {round(hygiene_awarded,2)}/{HYGIENE_TOTAL}."
    )
    results["messages"].append(
        f"Total awarded: {total_awarded}/{MAX_TOTAL} -> {results['percentage']}%."
    )

    # Add basic pass/fail recommendation and any major red flags
    pass_threshold = 80.0
    if results["overall_score"] >= pass_threshold:
        results["recommendation"] = "PASS"
        results["messages"].append("Candidate meets the passing threshold.")
    else:
        results["recommendation"] = "FAIL"
        results["messages"].append("Candidate does not meet the passing threshold (80%).")

    return results

# Script entry point ---------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python task_evaluation.py path/to/test_submission.json path/to/answer_key.json")
        sys.exit(2)

    cand_path = sys.argv[1]
    ans_path = sys.argv[2]

    # Resolve output path to same directory as this script
    out_path = Path(__file__).parent / "test_results.json"

    # Load JSON input files with robust error handling
    try:
        candidate_json = load_json(cand_path)
    except FileNotFoundError:
        print(f"Candidate submission file not found: {cand_path}")
        sys.exit(3)
    except ValueError as e:
        print(f"Error loading candidate submission: {e}")
        sys.exit(4)

    try:
        answer_json = load_json(ans_path)
    except FileNotFoundError:
        print(f"Answer key file not found: {ans_path}")
        sys.exit(5)
    except ValueError as e:
        print(f"Error loading answer key: {e}")
        sys.exit(6)

    # Perform grading
    try:
        results = grade(candidate_json, answer_json)
    except Exception as e:
        # Capture unexpected errors and write a minimal results file
        tb = str(e)
        minimal = {
            "error": "Grading failed due to an internal error.",
            "exception": tb
        }
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(minimal, f, indent=2)
        except Exception:
            pass
        print("Grading failed unexpectedly:", e)
        sys.exit(10)

    # Save results
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Failed to write results to {out_path}: {e}")
        sys.exit(11)

    # Print summary to stdout
    print(f"Grading complete. Results written to: {out_path}")
    print(f"Candidate overall score: {results.get('overall_score')}% ({results.get('total_points')}/{results.get('max_points')})")
    print(f"Recommendation: {results.get('recommendation')}")
    sys.exit(0)


if __name__ == "__main__":
    main()