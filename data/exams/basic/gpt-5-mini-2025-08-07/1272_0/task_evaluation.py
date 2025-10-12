#!/usr/bin/env python3
"""
task_evaluation.py

Automated grader for the Practical Code Maintenance basic exam.

Usage:
    python task_evaluation.py path/to/test_submission.json path/to/answer_key.json

Produces:
    test_results.json in the same directory as this script.

Notes:
- Uses only Python standard library.
- Designed to be deterministic and robust to minor formatting differences.
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

# ----------------------
# Helper utility functions
# ----------------------


def load_json(path: str) -> Any:
    """Load JSON file and return parsed object, raising a ValueError on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")


def safe_get(d: Dict, key: str, default=None):
    return d.get(key, default)


def tests_report_passed(test_output: str) -> bool:
    """
    Heuristic: consider tests passing if output contains '<num> passed' and does not
    contain 'FAILED' or 'failed' in a prominent way. This tolerates pytest output.
    """
    if not test_output:
        return False
    # If output explicitly contains 'FAILED' or 'failed' alone (case-insensitive) -> fail
    if re.search(r"\bFAILED\b", test_output, flags=re.IGNORECASE):
        return False
    # Check for pattern like '4 passed' or 'passed in'
    if re.search(r"\b\d+\s+passed\b", test_output) or re.search(r"\bpassed in\b", test_output):
        return True
    # Also accept 'All tests passed' etc.
    if re.search(r"all tests passed", test_output, flags=re.IGNORECASE):
        return True
    return False


def extract_benchmark_times(test_output: str) -> Dict[str, float]:
    """
    Parse bench.py output lines like:
      BENCHMARK: small time_seconds=0.000312
      duplicate_count: 4
    Returns dict mapping 'small' and/or 'large' -> time_seconds (float) when found.
    """
    times = {}
    if not test_output:
        return times
    for m in re.finditer(r"BENCHMARK:\s*(\w+)\s+time_seconds=([0-9.]+)", test_output):
        kind = m.group(1)
        val = float(m.group(2))
        times[kind] = val
    return times


def count_sentences(text: str) -> int:
    """
    Rough sentence counter using periods, exclamation marks, and question marks.
    """
    if not text:
        return 0
    # Normalize whitespace
    s = re.sub(r"\s+", " ", text.strip())
    # Split on sentence-ending punctuation followed by space or end-of-string
    parts = re.split(r"[.!?]+\s*", s)
    parts = [p for p in (p.strip() for p in parts) if p]
    return len(parts)


def contains_keywords(text: str, keywords: List[str]) -> bool:
    if not text:
        return False
    low = text.lower()
    for kw in keywords:
        if kw.lower() in low:
            return True
    return False


def safe_number(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        return float(str(x))
    except Exception:
        return None


# ----------------------
# Grading logic
# ----------------------


def grade_submission(candidate: Dict, answer_key: Dict) -> Dict:
    """
    Main grading function. Returns a structured results dict to be serialized.
    """
    results: Dict[str, Any] = {
        "per_task": {},
        "deductions": [],
        "total_points_awarded": 0,
        "max_points": 100,
    }

    # Scoring weights
    weights = {
        "A_correctness": 30,
        "A_minimal": 10,
        "B_feature": 20,
        "B_integration_test": 10,
        "C_improvement": 15,
        "C_explanation": 10,
        "submission_quality": 5,
    }

    # Basic sanity checks for required top-level fields
    required_top = ["name", "language", "time_taken_minutes", "run_commands", "tasks"]
    missing_keys = [k for k in required_top if k not in candidate]
    submission_quality_score = 0
    submission_quality_notes: List[str] = []

    if missing_keys:
        submission_quality_notes.append(f"Missing required top-level keys: {missing_keys}")
    else:
        # Validate tasks: must be array of 3 with ids A,B,C
        tasks = candidate.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 3:
            submission_quality_notes.append("tasks must be an array of 3 task objects (A,B,C).")
        else:
            ids = {safe_get(t, "id") for t in tasks}
            if ids != {"A", "B", "C"}:
                submission_quality_notes.append(f"tasks must contain ids 'A','B','C'. Found ids: {ids}")

    # Award submission quality points if top-level structure is fine
    if not missing_keys and not submission_quality_notes:
        submission_quality_score = weights["submission_quality"]
    else:
        # Partial credit: if name, tasks present but minor issues, give 2 pts
        if "name" in candidate and "tasks" in candidate:
            submission_quality_score = 2
        else:
            submission_quality_score = 0

    results["per_task"]["submission_quality"] = {
        "points_awarded": submission_quality_score,
        "points_available": weights["submission_quality"],
        "notes": submission_quality_notes or ["Top-level submission structure validated."],
    }

    # Map tasks by id for convenience
    task_map = {}
    tasks_list = candidate.get("tasks", []) if isinstance(candidate.get("tasks"), list) else []
    for t in tasks_list:
        tid = safe_get(t, "id")
        if tid:
            task_map[tid] = t

    # -------------------
    # Task A grading
    # -------------------
    a_awarded = 0
    a_notes: List[str] = []
    taskA = task_map.get("A", {})
    a_test_output = safe_get(taskA, "test_output", "")
    a_modified_files = safe_get(taskA, "modified_files", {}) or {}

    # Correctness (30): Task A tests pass
    if tests_report_passed(a_test_output):
        a_awarded += weights["A_correctness"]
        a_notes.append("Task A tests passed (correctness).")
    else:
        a_notes.append("Task A tests did not pass in submitted test_output. No correctness points awarded.")

    # Minimal & well-scoped changes (10)
    minimal_award = 0
    # Expect processor.py to be among modified files for Task A
    if "processor.py" in a_modified_files and len(a_modified_files) <= 2:
        # also require a non-empty explanation (3-6 sentences preferred)
        expl = safe_get(taskA, "explanation", "")
        sent_count = count_sentences(expl)
        if sent_count >= 2:
            minimal_award = weights["A_minimal"]
            a_notes.append("Task A modifications look localized (processor.py present, <=2 files) and explanation provided.")
        else:
            minimal_award = int(weights["A_minimal"] * 0.6)
            a_notes.append("processor.py modified but explanation is short; partial minimality credit awarded.")
    elif "processor.py" in a_modified_files:
        minimal_award = int(weights["A_minimal"] * 0.6)
        a_notes.append("processor.py modified but multiple files changed for Task A; partial minimality credit awarded.")
    else:
        minimal_award = 0
        a_notes.append("processor.py not found among Task A modified_files; minimality credit denied.")

    a_awarded += minimal_award

    results["per_task"]["A"] = {
        "points_awarded": a_awarded,
        "points_available": weights["A_correctness"] + weights["A_minimal"],
        "notes": a_notes,
        "details": {
            "test_output": a_test_output,
            "modified_files_list": list(a_modified_files.keys()),
            "explanation": safe_get(taskA, "explanation"),
        },
    }
    # -------------------
    # Task B grading
    # -------------------
    b_awarded = 0
    b_notes: List[str] = []
    taskB = task_map.get("B", {})
    b_test_output = safe_get(taskB, "test_output", "")
    b_modified_files = safe_get(taskB, "modified_files", {}) or {}
    b_expl = safe_get(taskB, "explanation", "")

    # Feature works (20): tests pass AND test for min_length present in modified test file
    feature_award = 0
    integration_award = 0

    tests_pass = tests_report_passed(b_test_output)

    # Check that candidate added or modified tests/test_task_b_spec.py and that it contains min_length usage
    testfile_present = False
    testfile_contains_min = False
    for fname, content in b_modified_files.items():
        if fname.endswith("tests/test_task_b_spec.py") or fname.endswith("test_task_b_spec.py") or "/test_task_b_spec.py" in fname:
            testfile_present = True
            if "min_length" in (content or ""):
                testfile_contains_min = True

    # Check processor.py modified for feature
    processor_modified = "processor.py" in b_modified_files

    # Determine awarding for feature:
    if tests_pass and testfile_contains_min and processor_modified:
        feature_award = weights["B_feature"]
        integration_award = weights["B_integration_test"]
        b_notes.append("All B tests passed and min_length test present; feature and integration points awarded.")
    else:
        # Partial cases
        if processor_modified and testfile_contains_min:
            # If tests didn't pass but changes present -> partial credit
            feature_award = int(weights["B_feature"] * 0.5)
            integration_award = weights["B_integration_test"]
            b_notes.append("Processor and B test updated for min_length but tests did not pass; partial feature credit awarded.")
        elif processor_modified and testfile_present:
            # test present but maybe not using min_length
            feature_award = int(weights["B_feature"] * 0.5)
            integration_award = int(weights["B_integration_test"] * 0.7)
            b_notes.append("Processor modified and a test for Task B was provided but it may not use min_length; partial credit awarded.")
        elif processor_modified:
            # Changed processor but no new/updated test
            feature_award = int(weights["B_feature"] * 0.6)
            integration_award = 0
            b_notes.append("Processor modified but no Task B test added; partial feature credit awarded (no integration/test credit).")
        else:
            feature_award = 0
            integration_award = 0
            b_notes.append("No evidence of min_length implementation in submitted Task B artifacts; no points awarded.")

    b_awarded = feature_award + integration_award

    results["per_task"]["B"] = {
        "points_awarded": b_awarded,
        "points_available": weights["B_feature"] + weights["B_integration_test"],
        "notes": b_notes,
        "details": {
            "test_output": b_test_output,
            "modified_files_list": list(b_modified_files.keys()),
            "processor_modified": processor_modified,
            "b_testfile_present": testfile_present,
            "b_testfile_contains_min_length": testfile_contains_min,
            "explanation": b_expl,
        },
    }

    # -------------------
    # Task C grading
    # -------------------
    c_awarded = 0
    c_notes: List[str] = []
    taskC = task_map.get("C", {})
    c_test_output = safe_get(taskC, "test_output", "")
    c_modified_files = safe_get(taskC, "modified_files", {}) or {}
    c_expl = safe_get(taskC, "explanation", "")
    bench_before = safe_number(safe_get(taskC, "benchmark_before"))
    bench_after = safe_number(safe_get(taskC, "benchmark_after"))

    # Parse any bench times from test_output as fallback if benchmark_before/after missing.
    bench_times_in_output = extract_benchmark_times(c_test_output)

    # Improvement (15): benchmark_before & after provided and after < before
    improvement_award = 0
    explanation_award = 0

    if bench_before is not None and bench_after is not None:
        if bench_after < bench_before:
            improvement_award = weights["C_improvement"]
            c_notes.append(
                f"benchmark provided and improved: before={bench_before}, after={bench_after} -> improvement awarded."
            )
        elif bench_after == bench_before:
            improvement_award = 0
            c_notes.append("benchmark provided but no improvement (after == before); no improvement credit.")
        else:
            improvement_award = 0
            c_notes.append("benchmark provided but no improvement (after >= before); no improvement credit.")
    else:
        # Try to infer from output: if output contains 'BENCHMARK: large time_seconds=...' we can only grade if
        # the value is smaller than the answer_key's baseline (if present).
        # Get answer key baseline large time if available
        ak_taskC = None
        for t in answer_key.get("tasks", []):
            if safe_get(t, "id") == "C":
                ak_taskC = t
                break
        ak_large = None
        if ak_taskC:
            ak_large = safe_number(safe_get(ak_taskC, "benchmark_before") or None)

        # If candidate's output contains a 'large' time and that is less than ak_large (if known), award improvement
        if "large" in bench_times_in_output and ak_large is not None:
            cand_large = bench_times_in_output.get("large")
            if cand_large is not None and cand_large < ak_large:
                improvement_award = weights["C_improvement"]
                c_notes.append(
                    f"Inferred benchmark improvement from test_output: candidate large={cand_large} < answer_key baseline={ak_large}"
                )
            else:
                improvement_award = 0
                c_notes.append("No clear benchmark improvement inferred from output.")
        else:
            # No numeric benchmarks; check if explanation claims robustness (e.g., prevented failure on large input)
            if contains_keywords(c_expl, ["prevent", "robust", "handle", "fail", "failure", "memory", "crash"]):
                improvement_award = int(weights["C_improvement"] * 0.6)
                c_notes.append("No numeric benchmarks provided but explanation claims improved robustness: partial credit.")
            else:
                improvement_award = 0
                c_notes.append("No benchmark numbers provided and no clear robustness claim: no improvement credit.")

    # Explanation quality (10): check for mention of algorithmic improvement keywords or clear justification
    keywords = ["counter", "collections.Counter", "set", "linear", "o(n)", "O(n)", "algorithm", "complex", "cache", "guard", "optimiz"]
    if contains_keywords(c_expl or "", keywords) and count_sentences(c_expl) >= 1:
        explanation_award = weights["C_explanation"]
        c_notes.append("Explanation contains algorithmic keywords and is present: full explanation credit.")
    elif c_expl and len(c_expl.strip()) > 20:
        explanation_award = int(weights["C_explanation"] * 0.6)
        c_notes.append("Explanation present but missing keywords: partial explanation credit.")
    else:
        explanation_award = 0
        c_notes.append("Explanation missing or too short: no explanation credit.")

    c_awarded = improvement_award + explanation_award

    results["per_task"]["C"] = {
        "points_awarded": c_awarded,
        "points_available": weights["C_improvement"] + weights["C_explanation"],
        "notes": c_notes,
        "details": {
            "test_output": c_test_output,
            "modified_files_list": list(c_modified_files.keys()),
            "benchmark_before": bench_before,
            "benchmark_after": bench_after,
            "bench_times_in_output": bench_times_in_output,
            "explanation": c_expl,
        },
    }

    # -------------------
    # Aggregate totals and final notes
    # -------------------
    total_awarded = (
        results["per_task"]["A"]["points_awarded"]
        + results["per_task"]["B"]["points_awarded"]
        + results["per_task"]["C"]["points_awarded"]
        + submission_quality_score
    )

    results["total_points_awarded"] = total_awarded
    results["max_points"] = 100
    percentage = (total_awarded / results["max_points"]) * 100.0
    results["percentage"] = round(percentage, 2)
    results["overall_score"] = results["percentage"]  # required variable in output

    # Compose deductions/reasons list
    deductions: List[str] = []

    # Common deductions messages derived from notes
    # For Task A correctness
    if results["per_task"]["A"]["points_awarded"] < (weights["A_correctness"] + weights["A_minimal"]):
        deductions.append(f"Task A partial/failed: {results['per_task']['A']['notes']}")

    if results["per_task"]["B"]["points_awarded"] < (weights["B_feature"] + weights["B_integration_test"]):
        deductions.append(f"Task B partial/failed: {results['per_task']['B']['notes']}")

    if results["per_task"]["C"]["points_awarded"] < (weights["C_improvement"] + weights["C_explanation"]):
        deductions.append(f"Task C partial/failed: {results['per_task']['C']['notes']}")

    if submission_quality_score < weights["submission_quality"]:
        deductions.append("Submission quality issues: " + "; ".join(submission_quality_notes) if submission_quality_notes else "Top-level format issues.")

    results["deductions"] = deductions

    # Include candidate metadata and run_commands for evaluator convenience
    results["candidate_metadata"] = {
        "name": candidate.get("name"),
        "email": candidate.get("email"),
        "language": candidate.get("language"),
        "time_taken_minutes": candidate.get("time_taken_minutes"),
        "run_commands": candidate.get("run_commands"),
    }

    # Keep answer key small excerpt for manual review
    results["answer_key_summary"] = {
        "expected_tasks": [safe_get(t, "id") for t in answer_key.get("tasks", []) if safe_get(t, "id")],
    }

    return results


# ----------------------
# Main entrypoint
# ----------------------


def main(argv):
    if len(argv) < 3:
        print("Usage: python task_evaluation.py path/to/test_submission.json path/to/answer_key.json")
        sys.exit(2)

    submission_path = argv[1]
    answer_key_path = argv[2]

    # Load inputs with error handling
    try:
        candidate = load_json(submission_path)
    except ValueError as e:
        err = {"error": str(e)}
        with open("test_results.json", "w", encoding="utf-8") as outf:
            json.dump(err, outf, indent=2)
        print(f"Error loading submission: {e}")
        sys.exit(1)

    try:
        answer_key = load_json(answer_key_path)
    except ValueError as e:
        err = {"error": str(e)}
        with open("test_results.json", "w", encoding="utf-8") as outf:
            json.dump(err, outf, indent=2)
        print(f"Error loading answer key: {e}")
        sys.exit(1)

    # Perform grading
    try:
        results = grade_submission(candidate, answer_key)
    except Exception as e:
        # Unexpected internal error: produce an error output file for debugging
        err = {"error": f"Internal grader error: {e}"}
        with open("test_results.json", "w", encoding="utf-8") as outf:
            json.dump(err, outf, indent=2)
        print(f"Internal grading error: {e}")
        sys.exit(1)

    # Save results
    try:
        with open("test_results.json", "w", encoding="utf-8") as outf:
            json.dump(results, outf, indent=2)
    except Exception as e:
        print(f"Failed to write test_results.json: {e}")
        sys.exit(1)

    # Print a concise summary to stdout
    print("Grading complete.")
    print(f"Total: {results['total_points_awarded']}/{results['max_points']}  ({results['percentage']}%)")
    print("Per-task awarded points:")
    for tid in ("A", "B", "C"):
        t = results["per_task"].get(tid, {})
        print(f"  Task {tid}: {t.get('points_awarded', 0)} / {t.get('points_available', 0)}")
    print(f"Submission quality: {results['per_task']['submission_quality']['points_awarded']} / {results['per_task']['submission_quality']['points_available']}")
    print("Detailed results written to test_results.json")


if __name__ == "__main__":
    main(sys.argv)