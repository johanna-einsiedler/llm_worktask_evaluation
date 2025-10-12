#!/usr/bin/env python3
"""
task_evaluation.py

Automated grader for the Practical Maintenance & Optimization — Basic exam.

Usage:
    python task_evaluation.py <candidate_submission.json> <answer_key.json>

This script:
- Loads the candidate's test_submission.json and the official answer key (answer_key.json).
- Applies the grading rubric described in the exam materials.
- Produces a detailed JSON report named test_results.json saved next to this script.

Implementation notes:
- Uses only Python standard library.
- Robust to missing/malformed fields (will report problems in the output).
- Deterministic scoring based on evidence fields, commands run, and textual explanations.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

# ---------- Helper utilities ----------

def load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON from {path}: {e}")

def find_task(tasks: List[Dict[str, Any]], task_id: str) -> Optional[Dict[str, Any]]:
    for t in tasks:
        if str(t.get("id", "")).strip().lower() == task_id.lower():
            return t
    return None

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def to_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        s = str(value).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def contains_keyword(text: str, keywords: List[str]) -> bool:
    if not isinstance(text, str):
        return False
    low = text.lower()
    return any(k.lower() in low for k in keywords)

def cmd_contains(cmd_text: str, substr: str) -> bool:
    if not isinstance(cmd_text, str):
        return False
    return substr.lower() in cmd_text.lower()

# ---------- Grading logic for each task ----------

def grade_task_a(candidate_task: Dict[str, Any], answer_task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task A (Bug Fix) scoring:
    - Correctness (unit tests pass): 25
    - Quality of fix (files_changed includes src/processor.py): 5
    - Explanation (sufficient content): 5
    Total: 35
    """
    max_points = 35
    breakdown = {
        "tests_passed": {"score": 0, "max": 25, "reason": ""},
        "quality_of_fix": {"score": 0, "max": 5, "reason": ""},
        "explanation": {"score": 0, "max": 5, "reason": ""},
    }
    reasons = []

    # Evidence tests_passed
    tests_passed = safe_get(candidate_task, "evidence", "tests_passed")
    # allow string "true"/"false"
    tp_bool = None
    if isinstance(tests_passed, bool):
        tp_bool = tests_passed
    elif isinstance(tests_passed, str):
        tp_bool = tests_passed.strip().lower() == "true"
    else:
        tp_bool = False

    if tp_bool:
        breakdown["tests_passed"]["score"] = 25
        breakdown["tests_passed"]["reason"] = "Candidate reports tests passed."
    else:
        # partial credit if they modified expected file
        files = candidate_task.get("files_changed", []) or []
        if any(f.strip() == "src/processor.py" for f in files):
            breakdown["tests_passed"]["score"] = 10
            breakdown["tests_passed"]["reason"] = "Tests not reported passing, but src/processor.py was changed -> partial credit."
        else:
            breakdown["tests_passed"]["score"] = 0
            breakdown["tests_passed"]["reason"] = "Tests not passed and no relevant file changed."

    # Quality of fix: heuristic check for src/processor.py in files_changed
    files = candidate_task.get("files_changed", []) or []
    if any(f.strip() == "src/processor.py" for f in files):
        breakdown["quality_of_fix"]["score"] = 5
        breakdown["quality_of_fix"]["reason"] = "src/processor.py modified (expected for this fix)."
    else:
        breakdown["quality_of_fix"]["score"] = 0
        breakdown["quality_of_fix"]["reason"] = "src/processor.py not listed in files_changed."

    # Explanation: require non-empty and mention the bug domain (strip/whitespace/parse)
    explanation = candidate_task.get("explanation", "") or ""
    if len(explanation.strip()) >= 20 or contains_keyword(explanation, ["strip", "whitespace", "parse", "parse_line"]):
        breakdown["explanation"]["score"] = 5
        breakdown["explanation"]["reason"] = "Sufficient explanation present."
    elif len(explanation.strip()) >= 5:
        breakdown["explanation"]["score"] = 2
        breakdown["explanation"]["reason"] = "Brief explanation provided; partial credit."
    else:
        breakdown["explanation"]["score"] = 0
        breakdown["explanation"]["reason"] = "No explanation or too short."

    total = sum(breakdown[k]["score"] for k in breakdown)
    # Collect deduction reasons
    for k, v in breakdown.items():
        if v["score"] < v["max"]:
            reasons.append(f"{k}: {v['reason']} (awarded {v['score']}/{v['max']})")

    return {
        "id": "A-bugfix",
        "points_awarded": total,
        "points_max": max_points,
        "breakdown": breakdown,
        "deductions": reasons
    }

def grade_task_b(candidate_task: Dict[str, Any], answer_task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task B (Low-memory adaptation) scoring:
    - Adaptation correctness: 15
    - Evidence: 10
    - Explanation: 5
    Total: 30
    """
    max_points = 30
    breakdown = {
        "adaptation_correctness": {"score": 0, "max": 15, "reason": ""},
        "evidence": {"score": 0, "max": 10, "reason": ""},
        "explanation": {"score": 0, "max": 5, "reason": ""},
    }
    reasons = []

    # Check evidence.low_mem_mode_ran
    low_mem_ran = safe_get(candidate_task, "evidence", "low_mem_mode_ran")
    lm_bool = None
    if isinstance(low_mem_ran, bool):
        lm_bool = low_mem_ran
    elif isinstance(low_mem_ran, str):
        lm_bool = low_mem_ran.strip().lower() == "true"
    else:
        lm_bool = False

    if lm_bool:
        breakdown["adaptation_correctness"]["score"] = 15
        breakdown["adaptation_correctness"]["reason"] = "Candidate reports low-memory mode ran successfully."
    else:
        # partial if file changed
        files = candidate_task.get("files_changed", []) or []
        if any(f.strip() == "src/processor.py" for f in files):
            breakdown["adaptation_correctness"]["score"] = 7
            breakdown["adaptation_correctness"]["reason"] = "Low-memory mode not reported running, but src/processor.py modified -> partial credit."
        else:
            breakdown["adaptation_correctness"]["score"] = 0
            breakdown["adaptation_correctness"]["reason"] = "No low-memory evidence and no relevant file change."

    # Evidence via commands_run: look for a command containing '--low-mem' or 'low-mem' with exit_code 0 and stdout indicating success.
    cmd_entries = candidate_task.get("commands_run", []) or []
    evidence_ok = False
    evidence_partial = False
    evid_notes = []
    for cmd_obj in cmd_entries:
        cmd = str(cmd_obj.get("cmd", "") or "")
        stdout = str(cmd_obj.get("stdout", "") or "")
        exit_code = cmd_obj.get("exit_code", None)
        if "--low-mem" in cmd or "low-mem" in cmd.lower():
            # prefer exit_code == 0 and success markers
            if exit_code == 0 and ("LOW-MEM MODE OK: True".lower() in stdout.lower() or "low-mem mode ok: true" in stdout.lower() or "LOW-MEM MODE OK: true".lower() in stdout.lower()):
                evidence_ok = True
                evid_notes.append(f"Found successful low-mem command: {cmd}")
                break
            # maybe returned result but OK False
            if "LOW-MEM MODE RESULT" in stdout or "LOW-MEM MODE OK" in stdout:
                evidence_partial = True
                evid_notes.append(f"Found low-mem command but not clearly successful: {cmd}")
    if evidence_ok:
        breakdown["evidence"]["score"] = 10
        breakdown["evidence"]["reason"] = "Clear low-memory run found with success indicator in stdout and exit_code 0."
    elif evidence_partial:
        breakdown["evidence"]["score"] = 5
        breakdown["evidence"]["reason"] = "Low-memory command present but stdout did not clearly indicate OK or exit code != 0."
    else:
        breakdown["evidence"]["score"] = 0
        breakdown["evidence"]["reason"] = "No low-memory command evidence found in commands_run."

    # Explanation: similar heuristic to Task A
    explanation = candidate_task.get("explanation", "") or ""
    if len(explanation.strip()) >= 20 or contains_keyword(explanation, ["stream", "streaming", "low-mem", "memory", "stream=True", "fallback"]):
        breakdown["explanation"]["score"] = 5
        breakdown["explanation"]["reason"] = "Sufficient explanation about streaming/low-memory adaptation."
    elif len(explanation.strip()) >= 5:
        breakdown["explanation"]["score"] = 2
        breakdown["explanation"]["reason"] = "Brief explanation; partial credit."
    else:
        breakdown["explanation"]["score"] = 0
        breakdown["explanation"]["reason"] = "No explanation provided."

    total = sum(breakdown[k]["score"] for k in breakdown)
    for k, v in breakdown.items():
        if v["score"] < v["max"]:
            reasons.append(f"{k}: {v['reason']} (awarded {v['score']}/{v['max']})")
    if evid_notes:
        reasons.append("evidence_notes: " + " | ".join(evid_notes))

    return {
        "id": "B-lowmem",
        "points_awarded": total,
        "points_max": max_points,
        "breakdown": breakdown,
        "deductions": reasons
    }

def grade_task_c(candidate_task: Dict[str, Any], answer_task: Dict[str, Any], candidate_task_a: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Task C (Performance) scoring:
    - Performance improvement: 20 (full if improvement_factor >= 2.0)
      - partial credit linearly for 1.0 < factor < 2.0
    - Regression-free (tests still pass): 10 (requires Task A tests_passed true)
    - Explanation: 5
    Total: 35
    """
    max_points = 35
    breakdown = {
        "performance_improvement": {"score": 0, "max": 20, "reason": ""},
        "regression_free": {"score": 0, "max": 10, "reason": ""},
        "explanation": {"score": 0, "max": 5, "reason": ""},
    }
    reasons = []

    evidence = candidate_task.get("evidence", {}) or {}
    baseline = to_float(evidence.get("baseline_time_s"))
    optimized = to_float(evidence.get("optimized_time_s"))
    reported_factor = to_float(evidence.get("improvement_factor"))

    calc_factor = None
    if baseline is not None and optimized is not None and optimized > 0:
        calc_factor = baseline / optimized if optimized > 0 else None

    # Determine the improvement factor to use (prefer calculated if possible)
    use_factor = None
    factor_source = ""
    if calc_factor is not None:
        use_factor = calc_factor
        factor_source = "calculated"
    elif reported_factor is not None:
        use_factor = reported_factor
        factor_source = "reported"
    else:
        use_factor = None

    if use_factor is None:
        breakdown["performance_improvement"]["score"] = 0
        breakdown["performance_improvement"]["reason"] = "No valid baseline/optimized times provided to compute improvement."
        reasons.append("performance_improvement: missing baseline/optimized times.")
    else:
        # Cap unrealistic factors and handle edge cases
        if use_factor <= 1.0:
            breakdown["performance_improvement"]["score"] = 0
            breakdown["performance_improvement"]["reason"] = f"No improvement (factor={use_factor:.3f}, source={factor_source})."
            reasons.append(breakdown["performance_improvement"]["reason"])
        else:
            # Full credit when factor >= 2.0
            if use_factor >= 2.0:
                breakdown["performance_improvement"]["score"] = 20
                breakdown["performance_improvement"]["reason"] = f"Improvement factor {use_factor:.3f} (>=2.0) => full credit."
            else:
                # linear scaling between 1.0 -> 0 pts and 2.0 -> full pts
                # factor in (1,2) gives 0..20 linearly
                scaled = 20.0 * (use_factor - 1.0) / 1.0
                # round to 2 decimals points
                scaled = round(scaled, 2)
                breakdown["performance_improvement"]["score"] = float(scaled)
                breakdown["performance_improvement"]["reason"] = f"Improvement factor {use_factor:.3f} => partial credit ({scaled}/{20})."
                reasons.append(breakdown["performance_improvement"]["reason"])

    # Regression-free: require Task A tests_passed true
    tests_passed_A = False
    if candidate_task_a:
        tests_passed_A = bool(safe_get(candidate_task_a, "evidence", "tests_passed") is True)
    if tests_passed_A:
        breakdown["regression_free"]["score"] = 10
        breakdown["regression_free"]["reason"] = "Tests pass after changes (Task A evidence)."
    else:
        # Also try to detect in candidate_task commands that tests passed
        cmd_entries = candidate_task.get("commands_run", []) or []
        tests_ok_found = False
        for c in cmd_entries:
            stdout = str(c.get("stdout", "") or "")
            if "ALL TESTS PASSED" in stdout or "ALL TESTS PASSED".lower() in stdout.lower():
                tests_ok_found = True
                break
        if tests_ok_found:
            breakdown["regression_free"]["score"] = 10
            breakdown["regression_free"]["reason"] = "Found 'ALL TESTS PASSED' in Task C commands_run stdout."
        else:
            breakdown["regression_free"]["score"] = 0
            breakdown["regression_free"]["reason"] = "Tests not shown as passing after optimization."

    # Explanation: simple heuristics
    explanation = candidate_task.get("explanation", "") or ""
    if len(explanation.strip()) >= 20 or contains_keyword(explanation, ["optimiz", "complex", "dict", "set", "k*(k-1)", "combin"]):
        breakdown["explanation"]["score"] = 5
        breakdown["explanation"]["reason"] = "Sufficient explanation about optimization/algorithmic change."
    elif len(explanation.strip()) >= 5:
        breakdown["explanation"]["score"] = 2
        breakdown["explanation"]["reason"] = "Brief explanation; partial credit."
    else:
        breakdown["explanation"]["score"] = 0
        breakdown["explanation"]["reason"] = "No explanation provided."

    total = sum(float(breakdown[k]["score"]) for k in breakdown)
    # Format reasons from components with incomplete/full credit info
    for k, v in breakdown.items():
        if float(v["score"]) < float(v["max"]):
            reasons.append(f"{k}: {v['reason']} (awarded {v['score']}/{v['max']})")

    # Additional note: if reported_factor is present but differs significantly from calc_factor, note it
    if reported_factor is not None and calc_factor is not None:
        if abs(reported_factor - calc_factor) / calc_factor > 0.05:  # >5% mismatch
            reasons.append(f"Reported improvement_factor ({reported_factor}) differs from calculated ({calc_factor:.6f}) by >5%.")

    return {
        "id": "C-performance",
        "points_awarded": total,
        "points_max": max_points,
        "breakdown": breakdown,
        "deductions": reasons,
        "computed_improvement_factor": (calc_factor if calc_factor is not None else reported_factor)
    }

# ---------- Main orchestration ----------

def main():
    if len(sys.argv) != 3:
        print("Usage: python task_evaluation.py <candidate_submission.json> <answer_key.json>")
        sys.exit(2)

    cand_path = sys.argv[1]
    ans_path = sys.argv[2]

    # Load files with robust error handling
    try:
        candidate = load_json(cand_path)
    except Exception as e:
        print(f"Error loading candidate submission: {e}")
        sys.exit(1)
    try:
        answer = load_json(ans_path)
    except Exception as e:
        print(f"Error loading answer key: {e}")
        sys.exit(1)

    results = {
        "candidate_file": os.path.abspath(cand_path),
        "answer_key_file": os.path.abspath(ans_path),
        "tasks": [],
        "total_points_awarded": 0.0,
        "total_points_max": 100.0,
        "percentage_score": 0.0,
        "overall_score": 0.0,
        "notes": []
    }

    # Validate basic structure
    cand_tasks = candidate.get("tasks", [])
    ans_tasks = answer.get("tasks", [])

    if not isinstance(cand_tasks, list):
        results["notes"].append("Candidate JSON 'tasks' field missing or not an array.")
        cand_tasks = []
    if not isinstance(ans_tasks, list):
        results["notes"].append("Answer key 'tasks' field missing or not an array.")
        ans_tasks = []

    # Find tasks by id
    cand_task_A = find_task(cand_tasks, "A-bugfix")
    cand_task_B = find_task(cand_tasks, "B-lowmem")
    cand_task_C = find_task(cand_tasks, "C-performance")

    ans_task_A = find_task(ans_tasks, "A-bugfix")
    ans_task_B = find_task(ans_tasks, "B-lowmem")
    ans_task_C = find_task(ans_tasks, "C-performance")

    # Basic presence checks
    if cand_task_A is None:
        results["notes"].append("Candidate submission missing task A-bugfix.")
        cand_task_A = {}
    if cand_task_B is None:
        results["notes"].append("Candidate submission missing task B-lowmem.")
        cand_task_B = {}
    if cand_task_C is None:
        results["notes"].append("Candidate submission missing task C-performance.")
        cand_task_C = {}

    # Grade each task
    graded_A = grade_task_a(cand_task_A, ans_task_A or {})
    graded_B = grade_task_b(cand_task_B, ans_task_B or {})
    graded_C = grade_task_c(cand_task_C, ans_task_C or {}, cand_task_A)

    # Collect and sum
    results["tasks"].append(graded_A)
    results["tasks"].append(graded_B)
    results["tasks"].append(graded_C)

    total_awarded = float(graded_A["points_awarded"]) + float(graded_B["points_awarded"]) + float(graded_C["points_awarded"])
    results["total_points_awarded"] = round(total_awarded, 2)
    results["total_points_max"] = 100.0
    percent = (total_awarded / 100.0) * 100.0
    results["percentage_score"] = round(percent, 2)
    # also set the required variable overall_score (numeric)
    results["overall_score"] = round(percent, 2)

    # Add human-readable summary notes
    if results["overall_score"] >= 80.0:
        results["notes"].append("PASS threshold (>=80%) met.")
    else:
        results["notes"].append("PASS threshold not met (>=80%).")

    # Save test_results.json in same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "test_results.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Grading complete. Results written to {out_path}")
    except Exception as e:
        print(f"Failed to write results to {out_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Expose overall_score variable as required by the spec (value assigned at runtime in results).
overall_score = None