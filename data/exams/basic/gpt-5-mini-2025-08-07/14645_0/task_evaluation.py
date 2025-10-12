# task_evaluation.py
"""
Automated grader for the "Test Planning Practical — Basic" exam.
Usage:
    python task_evaluation.py <candidate_submission.json> <answer_key.json>

Outputs:
    test_results.json in the same directory as this script.

Notes:
- Uses only standard libraries.
- Implements rubric and mandatory checks as described in the exam materials.
"""

import json
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict

# ---------- Configuration / Constants ----------
MAX_POINTS = 100.0
# Rubric breakdown (points)
RUBRIC = {
    "schedule_and_resourcing": 40.0,  # 5 sub-items (8 points each)
    "test_strategy": 30.0,            # 5 sub-items (6 points each)
    "risk_and_contingency": 15.0,     # 3 sub-items (6,5,4)
    "assumptions_and_json": 10.0,     # assumptions 3, json/schema 7
    "metrics_and_reporting": 5.0
}

# Sub-splits used for more granular scoring
SCHEDULE_SUB = {
    "tasks_coverage": 8.0,
    "estimates_realism": 8.0,
    "dependencies": 8.0,
    "resource_allocation": 8.0,
    "critical_path_and_dates": 8.0
}

STRATEGY_SUB = {
    "scope_objectives": 6.0,
    "entry_exit_criteria": 6.0,
    "test_types": 6.0,
    "automation_approach": 6.0,
    "reporting_metrics": 6.0
}

RISK_SUB = {
    "mapping": 6.0,
    "contingency_plan": 5.0,
    "defect_env_contingency": 4.0
}

ASSUMPTIONS_SUB = {
    "assumptions": 3.0,
    "json_schema": 7.0
}

METRICS_POINTS = 5.0

# Fixed project constraints from the brief (these are authoritative)
PROJECT_START_DATE_STR = "2025-11-03"
CODE_FREEZE_DATE_STR = "2025-11-21"
RELEASE_DATE_STR = "2025-11-28"

PROJECT_START_DATE = datetime.strptime(PROJECT_START_DATE_STR, "%Y-%m-%d").date()
CODE_FREEZE_DATE = datetime.strptime(CODE_FREEZE_DATE_STR, "%Y-%m-%d").date()
RELEASE_DATE = datetime.strptime(RELEASE_DATE_STR, "%Y-%m-%d").date()

# Environment gating
INTEGRATION_ENV_USABLE_OFFSET = 5  # offset days when integration env is usable
PREPROD_USABLE_OFFSET = 20

# Allowed owners
ALLOWED_OWNERS = {"Tester A", "Tester B", "Automation"}

# Required top-level keys per schema
REQUIRED_TOP_KEYS = {
    "candidate_name",
    "exam_start",
    "project_start_date",
    "assumptions",
    "test_strategy",
    "tasks",
    "schedule_summary",
    "risk_prioritization",
    "contingency_plan",
    "metrics_and_reporting",
    "final_comments"
}

# Required task types set
REQUIRED_TASK_TYPES = {"test-design", "env-setup", "test-execution", "automation", "regression", "UAT-support", "reporting"}

# Minimal helper functions
def safe_load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iso_to_date(d):
    # Accepts YYYY-MM-DD or returns None
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        return None

def offset_to_date(offset_int):
    return PROJECT_START_DATE + timedelta(days=int(offset_int))

def date_range(start_date, duration_days):
    # yields inclusive date list
    return [start_date + timedelta(days=i) for i in range(duration_days)]

def approx_equal(a, b, tol=1e-6):
    return abs(a - b) <= tol

# ---------- Main grading logic ----------
def main():
    # Argument handling
    if len(sys.argv) != 3:
        print("Usage: python task_evaluation.py <candidate_submission.json> <answer_key.json>")
        sys.exit(2)

    candidate_path = sys.argv[1]
    answer_key_path = sys.argv[2]
    results = {
        "scores": {},
        "details": [],
        "total_points": 0.0,
        "max_points": MAX_POINTS,
        "percentage": 0.0,
        "overall_score": 0.0,
        "pass": False,
        "mandatory_fail_reasons": []
    }

    # Load JSON files with error handling
    try:
        candidate = safe_load_json(candidate_path)
    except Exception as e:
        results["details"].append(f"ERROR: Could not load candidate JSON: {e}")
        write_results(results)
        sys.exit(1)

    try:
        answer_key = safe_load_json(answer_key_path)
    except Exception as e:
        results["details"].append(f"ERROR: Could not load answer-key JSON: {e}")
        write_results(results)
        sys.exit(1)

    # Basic validation of candidate top-level keys
    missing_keys = [k for k in REQUIRED_TOP_KEYS if k not in candidate]
    if missing_keys:
        results["details"].append(f"Missing top-level required keys: {missing_keys}")
        # Deduct full JSON/schema points
        results["scores"]["assumptions_and_json"] = 0.0
        results["details"].append("Assigned 0 points for Assumptions & JSON due to missing top-level keys.")
        # Still continue to try evaluate what is present
    else:
        # Evaluate assumptions & JSON (10 points)
        assump_score, assump_msg = evaluate_assumptions_and_json(candidate)
        results["scores"]["assumptions_and_json"] = assump_score
        results["details"].extend(assump_msg)

    # Test strategy scoring (30)
    if "test_strategy" in candidate:
        strat_score, strat_msgs = evaluate_test_strategy(candidate.get("test_strategy", ""))
        results["scores"]["test_strategy"] = strat_score
        results["details"].extend(strat_msgs)
    else:
        results["scores"]["test_strategy"] = 0.0
        results["details"].append("Missing test_strategy: 0 points for Test Strategy.")

    # Schedule & resourcing (40)
    schedule_score, sched_msgs, schedule_info = evaluate_schedule(candidate)
    results["scores"]["schedule_and_resourcing"] = schedule_score
    results["details"].extend(sched_msgs)

    # Risk & contingency (15)
    risk_score, risk_msgs = evaluate_risk_and_contingency(candidate)
    results["scores"]["risk_and_contingency"] = risk_score
    results["details"].extend(risk_msgs)

    # Metrics & reporting (5)
    metrics_score, metrics_msgs = evaluate_metrics(candidate)
    results["scores"]["metrics_and_reporting"] = metrics_score
    results["details"].extend(metrics_msgs)

    # Sum total points
    total = 0.0
    for k, v in results["scores"].items():
        total += v
    results["total_points"] = round(total, 2)
    results["percentage"] = round((total / MAX_POINTS) * 100.0, 2)
    results["overall_score"] = results["percentage"]

    # Mandatory element checks - if missing any -> automatic fail
    mandatory_fail_reasons = check_mandatory_elements(candidate, schedule_info)
    if mandatory_fail_reasons:
        results["pass"] = False
        results["mandatory_fail_reasons"] = mandatory_fail_reasons
        results["details"].append("MANDATORY ELEMENTS MISSING or VIOLATED: " + "; ".join(mandatory_fail_reasons))
        # According to instructions, failure to include mandatory elements results in automatic fail regardless of numeric score.
    else:
        # Determine pass/fail by >=80%
        results["pass"] = results["overall_score"] >= 80.0

    # Add pass summary
    if results["pass"]:
        results["details"].append(f"PASS: Candidate achieved {results['overall_score']}% (>=80%).")
    else:
        results["details"].append(f"FAIL: Candidate achieved {results['overall_score']}% (<80% or missing mandatory elements).")

    # Save results
    write_results(results)
    # Also print summary to stdout
    print(json.dumps({
        "overall_score": results["overall_score"],
        "pass": results["pass"],
        "total_points": results["total_points"],
        "max_points": results["max_points"]
    }, indent=2))


# ---------- Evaluation subroutines ----------

def evaluate_assumptions_and_json(candidate):
    msgs = []
    score = 0.0
    # Assumptions check (3 points)
    assumps = candidate.get("assumptions")
    if isinstance(assumps, list) and 4 <= len(assumps) <= 8:
        score += ASSUMPTIONS_SUB["assumptions"]
        msgs.append(f"Assumptions: {len(assumps)} items present -> full {ASSUMPTIONS_SUB['assumptions']} points.")
    elif isinstance(assumps, list) and len(assumps) > 0:
        # partial credit
        pts = ASSUMPTIONS_SUB["assumptions"] * (len(assumps) / 4.0)
        pts = min(ASSUMPTIONS_SUB["assumptions"], pts)
        score += pts
        msgs.append(f"Assumptions: {len(assumps)} items present -> {round(pts,2)} points (partial).")
    else:
        msgs.append("Assumptions missing or empty -> 0 points for assumptions.")

    # JSON Schema / format checks (7 points)
    schema_ok = True
    schema_msgs = []
    # Check required top keys presence
    missing_keys = [k for k in REQUIRED_TOP_KEYS if k not in candidate]
    if missing_keys:
        schema_ok = False
        schema_msgs.append(f"Missing required top-level keys: {missing_keys}")
    # Check tasks array presence and structure
    tasks = candidate.get("tasks")
    if not isinstance(tasks, list) or len(tasks) == 0:
        schema_ok = False
        schema_msgs.append("tasks array missing or empty.")
    else:
        # check each task fields and owners validity
        for idx, t in enumerate(tasks, start=1):
            for field in ["id", "description", "owner", "type", "estimate_hours", "duration_days", "earliest_start_offset", "dependencies", "parallelizable"]:
                if field not in t:
                    schema_ok = False
                    schema_msgs.append(f"Task {t.get('id','T?')} missing field '{field}'.")
            owner = t.get("owner")
            if owner not in ALLOWED_OWNERS:
                schema_ok = False
                schema_msgs.append(f"Task {t.get('id','T?')} has invalid owner '{owner}' (allowed: {ALLOWED_OWNERS}).")
            # dependencies should be list
            if "dependencies" in t and not isinstance(t.get("dependencies"), list):
                schema_ok = False
                schema_msgs.append(f"Task {t.get('id','T?')} dependencies must be an array.")
    if schema_ok:
        score += ASSUMPTIONS_SUB["json_schema"]
        msgs.append(f"JSON schema and task structure valid -> full {ASSUMPTIONS_SUB['json_schema']} points.")
    else:
        msgs.append("Schema/format issues: " + "; ".join(schema_msgs))
        # give partial credit if some elements valid
        # crude partial: if tasks array exists and owner strings valid for most tasks, give half
        if isinstance(tasks, list) and len(tasks) > 0:
            valid_owner_count = sum(1 for t in tasks if t.get("owner") in ALLOWED_OWNERS)
            frac = valid_owner_count / len(tasks)
            pts = round(ASSUMPTIONS_SUB["json_schema"] * frac, 2)
            score += pts
            msgs.append(f"Partial JSON/schema points awarded: {pts} (based on owner validity).")
        else:
            msgs.append("No partial JSON/schema points awarded.")

    # cap score to ASSUMPTIONS_SUB total
    score = min(score, sum(ASSUMPTIONS_SUB.values()))
    return round(score, 2), msgs

def evaluate_test_strategy(strategy_text):
    msgs = []
    score = 0.0
    text = strategy_text or ""
    length = len(text)
    if length < 300:
        msgs.append(f"test_strategy length {length} chars: below 300 requirement -> may be deducting points.")
    elif length > 800:
        msgs.append(f"test_strategy length {length} chars: above 800 recommended limit.")

    # Scope & objectives (6 pts) - look for 'scope', 'objective', feature ids or 'checkout' keyword
    if any(k in text.lower() for k in ["scope", "objective", "objectives", "f1", "checkout", "features"]):
        score += STRATEGY_SUB["scope_objectives"]
        msgs.append(f"Found scope/objectives indicators -> +{STRATEGY_SUB['scope_objectives']} pts.")
    else:
        msgs.append("Scope/objectives not clearly stated -> 0 for that sub-item.")

    # Entry/Exit criteria (6 pts) - check for 'entry', 'exit', 'criteria', 'p0', 'p1', 'sign-off', 'no more than'
    if any(k in text.lower() for k in ["entry", "exit", "criteria", "p0", "p1", "sign-off", "uAT".lower(), "no more than", "go/no-go", "go/no go"]):
        score += STRATEGY_SUB["entry_exit_criteria"]
        msgs.append(f"Entry/Exit criteria detected -> +{STRATEGY_SUB['entry_exit_criteria']} pts.")
    else:
        msgs.append("Entry/Exit criteria not detected or too vague -> 0 for that sub-item.")

    # Test types (6 pts) - check presence of smoke, functional, integration, regression, performance, UAT
    types_found = sum(1 for t in ["smoke", "functional", "integration", "regression", "performance", "uat"] if t in text.lower())
    if types_found >= 3:
        score += STRATEGY_SUB["test_types"]
        msgs.append(f"Multiple test types mentioned ({types_found}) -> +{STRATEGY_SUB['test_types']} pts.")
    else:
        partial = STRATEGY_SUB["test_types"] * (types_found / 3.0) if types_found > 0 else 0.0
        score += partial
        msgs.append(f"Test types found: {types_found} -> +{round(partial,2)} pts (partial).")

    # Automation approach (6 pts) - look for automation, 40%, nightly, 2h
    if any(k in text.lower() for k in ["automation", "40%", "40 percent", "night", "nightly", "2h", "2 h", "2 hours"]):
        score += STRATEGY_SUB["automation_approach"]
        msgs.append(f"Automation approach referenced -> +{STRATEGY_SUB['automation_approach']} pts.")
    else:
        msgs.append("Automation approach not clearly described -> 0 for that sub-item.")

    # Reporting metrics (6 pts) - daily, defects, pass/fail, cadence
    if any(k in text.lower() for k in ["daily", "defect", "open defects", "pass/fail", "cadence", "report", "reporting"]):
        score += STRATEGY_SUB["reporting_metrics"]
        msgs.append(f"Reporting metrics/cadence described -> +{STRATEGY_SUB['reporting_metrics']} pts.")
    else:
        msgs.append("Reporting metrics/cadence not clearly described -> 0 for that sub-item.")

    # Cap and return
    score = min(score, sum(STRATEGY_SUB.values()))
    return round(score, 2), msgs

def evaluate_schedule(candidate):
    msgs = []
    score = 0.0
    schedule_info = {
        "tasks": [],
        "task_map": {},
        "schedule_start": None,
        "schedule_end": None,
        "total_effort_hours": None
    }

    tasks = candidate.get("tasks")
    if not isinstance(tasks, list) or len(tasks) == 0:
        msgs.append("No tasks provided: 0 points for schedule & resourcing.")
        return 0.0, msgs, schedule_info

    # Convert tasks and basic validations
    task_map = {}
    for t in tasks:
        tid = t.get("id")
        # minimal validation
        if not tid:
            msgs.append("A task is missing an id. This will affect dependency checks.")
            continue
        task_map[tid] = t

    schedule_info["task_map"] = task_map

    # 1) Tasks coverage (8 points) - ensure required task types present
    types_present = set()
    for t in tasks:
        typ = t.get("type")
        if isinstance(typ, str):
            types_present.add(typ)
    missing_types = REQUIRED_TASK_TYPES - types_present
    if not missing_types:
        score += SCHEDULE_SUB["tasks_coverage"]
        msgs.append(f"All required task types present -> +{SCHEDULE_SUB['tasks_coverage']} pts.")
    else:
        # award partial based on fraction present
        present = len(REQUIRED_TASK_TYPES) - len(missing_types)
        pts = round(SCHEDULE_SUB["tasks_coverage"] * (present / len(REQUIRED_TASK_TYPES)), 2)
        score += pts
        msgs.append(f"Missing task types: {sorted(list(missing_types))} -> +{pts} pts (partial).")

    # 2) Estimates realism (8 points)
    # Check total_effort_hours matches sum of estimate_hours
    sum_estimates = 0.0
    bad_estimate_msgs = []
    for t in tasks:
        eh = t.get("estimate_hours")
        dur = t.get("duration_days")
        tid = t.get("id", "T?")
        try:
            eh_f = float(eh)
            sum_estimates += eh_f
            if isinstance(dur, int) and dur > 0:
                # estimate_hours should not exceed duration_days*8 by a large amount
                if eh_f > dur * 10:  # unrealistic >10h/day
                    bad_estimate_msgs.append(f"{tid}: estimate_hours {eh_f} >> duration_days {dur} (unrealistic >10h/day).")
        except Exception:
            bad_estimate_msgs.append(f"{tid}: invalid estimate_hours '{eh}'")
    # check candidate schedule_summary.total_effort_hours if provided
    schedule_summary = candidate.get("schedule_summary", {})
    candidate_total_effort = schedule_summary.get("total_effort_hours")
    if candidate_total_effort is not None:
        try:
            candidate_total_effort_f = float(candidate_total_effort)
            if not approx_equal(candidate_total_effort_f, sum_estimates):
                msgs.append(f"schedule_summary.total_effort_hours ({candidate_total_effort_f}) != sum of task estimate_hours ({sum_estimates}) -> discrepancy.")
                # small deduction: 50% of realism points if mismatch
                pts = SCHEDULE_SUB["estimates_realism"] * 0.5
                score += pts
                msgs.append(f"Awarded {pts} pts for estimates_realism (partial) due to discrepancy.")
            else:
                # Compare to answer_key expected effort if available to judge realism
                # Use answer_key schedule_summary if present
                answer_total = None
                try:
                    answer_total = float(answer_key.get("schedule_summary", {}).get("total_effort_hours", 0))
                except Exception:
                    answer_total = None
                if answer_total:
                    # Accept if sum_estimates within 0.5x - 2x of answer_total
                    if 0.5 * answer_total <= sum_estimates <= 2.0 * answer_total:
                        score += SCHEDULE_SUB["estimates_realism"]
                        msgs.append(f"Total effort {sum_estimates} within reasonable range of answer-key {answer_total} -> full {SCHEDULE_SUB['estimates_realism']} pts.")
                    else:
                        # partial credit proportional to closeness (clamped)
                        ratio = min(sum_estimates / answer_total, answer_total / sum_estimates)
                        pts = round(SCHEDULE_SUB["estimates_realism"] * max(0.0, ratio), 2)
                        score += pts
                        msgs.append(f"Total effort {sum_estimates} vs answer {answer_total} -> {pts} pts (partial).")
                else:
                    # no answer_key total -> accept equality
                    score += SCHEDULE_SUB["estimates_realism"]
                    msgs.append(f"Estimates consistent with schedule_summary -> +{SCHEDULE_SUB['estimates_realism']} pts.")
        except Exception:
            # invalid number in summary -> partial
            msgs.append("schedule_summary.total_effort_hours invalid -> partial credit for estimates.")
            pts = SCHEDULE_SUB["estimates_realism"] * 0.5
            score += pts
            msgs.append(f"Awarded {pts} pts (partial).")
    else:
        # no schedule_summary total provided -> partial credit if sum_estimates seems reasonable vs answer_key
        answer_total = None
        try:
            answer_total = float(answer_key.get("schedule_summary", {}).get("total_effort_hours", 0))
        except Exception:
            answer_total = None
        if answer_total:
            if 0.5 * answer_total <= sum_estimates <= 2.0 * answer_total:
                score += SCHEDULE_SUB["estimates_realism"] * 0.9
                msgs.append(f"No schedule_summary.total_effort_hours but sum estimates {sum_estimates} in reasonable range -> +{round(SCHEDULE_SUB['estimates_realism']*0.9,2)} pts.")
            else:
                pts = SCHEDULE_SUB["estimates_realism"] * 0.5
                score += pts
                msgs.append(f"No summary; sum estimates {sum_estimates} far from expected -> +{pts} pts (partial).")
        else:
            # no reference -> give partial based on absence/presence
            pts = SCHEDULE_SUB["estimates_realism"] * 0.5
            score += pts
            msgs.append(f"No reference answer total; awarding {pts} pts (partial).")

    if bad_estimate_msgs:
        msgs.append("Estimate concerns: " + "; ".join(bad_estimate_msgs))

    schedule_info["total_effort_hours"] = sum_estimates

    # 3) Dependencies defined and feasible (8 points)
    # Check that all dependencies reference existing IDs and no dependency refers to non-existent
    dep_issues = []
    for t in tasks:
        deps = t.get("dependencies", [])
        if not isinstance(deps, list):
            dep_issues.append(f"{t.get('id','T?')}: dependencies not an array.")
            continue
        for d in deps:
            if d not in task_map:
                dep_issues.append(f"{t.get('id','T?')}: dependency '{d}' not found in tasks.")
    if not dep_issues:
        score += SCHEDULE_SUB["dependencies"]
        msgs.append(f"All dependencies reference existing tasks -> +{SCHEDULE_SUB['dependencies']} pts.")
    else:
        # partial credit
        good = max(0, len(tasks) - len(dep_issues))
        pts = round(SCHEDULE_SUB["dependencies"] * (good / len(tasks)), 2) if len(tasks) > 0 else 0.0
        score += pts
        msgs.append(f"Dependency issues: {dep_issues} -> +{pts} pts (partial).")

    # 4) Resource allocation respecting headcount (8 points)
    # Build day-by-day allocations per owner by distributing estimate_hours evenly across duration_days
    # Map tasks to calendar days
    per_day_owner_hours = defaultdict(lambda: defaultdict(float))  # day -> owner -> hours
    overall_start = None
    overall_end = None
    try:
        for t in tasks:
            try:
                start_offset = int(t.get("earliest_start_offset", 0))
            except Exception:
                start_offset = 0
            dur = int(t.get("duration_days", 1)) if isinstance(t.get("duration_days"), int) else 1
            start_date = PROJECT_START_DATE + timedelta(days=start_offset)
            end_date = start_date + timedelta(days=dur - 1)
            if overall_start is None or start_date < overall_start:
                overall_start = start_date
            if overall_end is None or end_date > overall_end:
                overall_end = end_date
            # distribute hours evenly
            eh = float(t.get("estimate_hours", 0)) if t.get("estimate_hours", 0) is not None else 0.0
            daily_hours = (eh / dur) if dur > 0 else eh
            owner = t.get("owner")
            for single_date in date_range(start_date, dur):
                per_day_owner_hours[single_date.isoformat()][owner] += daily_hours
    except Exception as e:
        msgs.append(f"Error while computing per-day allocations: {e}")

    # Check constraints: each tester must not exceed 8 hours/day (approx)
    overloads = []
    for day, owner_map in per_day_owner_hours.items():
        for owner, hours in owner_map.items():
            if owner in ("Tester A", "Tester B"):
                if hours > 8.01:  # slight tolerance
                    overloads.append(f"{day}: {owner} allocated {round(hours,2)}h (>8h).")
    if not overloads:
        score += SCHEDULE_SUB["resource_allocation"]
        msgs.append(f"Resource allocation fits per-day tester capacity -> +{SCHEDULE_SUB['resource_allocation']} pts.")
    else:
        # partial: subtract proportionally
        # Determine fraction of days overloaded
        total_days = len(per_day_owner_hours) if per_day_owner_hours else 1
        pts = round(SCHEDULE_SUB["resource_allocation"] * max(0.0, (1 - (len(overloads) / total_days))), 2)
        pts = max(0.0, pts)
        score += pts
        msgs.append(f"Resource overloads detected ({len(overloads)}): {overloads} -> +{pts} pts (partial).")

    # 5) Critical path & dates (8 points)
    # Check schedule_summary dates and critical_path presence; ensure schedule_end <= release date
    schedule_summary = candidate.get("schedule_summary", {})
    schedule_start_str = schedule_summary.get("schedule_start")
    schedule_end_str = schedule_summary.get("schedule_end")
    critical_path = schedule_summary.get("critical_path", [])
    buffers_days = schedule_summary.get("buffers_days", 0)

    cp_issues = []
    # schedule_start should be >= project start date typically equal
    s_start_date = iso_to_date(schedule_start_str) if schedule_start_str else None
    s_end_date = iso_to_date(schedule_end_str) if schedule_end_str else None
    schedule_info["schedule_start"] = schedule_start_str
    schedule_info["schedule_end"] = schedule_end_str

    if s_end_date:
        if s_end_date <= RELEASE_DATE:
            score += SCHEDULE_SUB["critical_path_and_dates"]
            msgs.append(f"Schedule end {s_end_date.isoformat()} is on or before release date {RELEASE_DATE.isoformat()} -> +{SCHEDULE_SUB['critical_path_and_dates']} pts.")
        else:
            # Partial credit if there's explanation in final_comments or contingency_plan
            explanation = (candidate.get("final_comments","") + " " + candidate.get("contingency_plan","")).lower()
            if any(k in explanation for k in ["defer","trade-off","post-release","post release","post-release"]):
                pts = SCHEDULE_SUB["critical_path_and_dates"] * 0.5
                score += pts
                msgs.append(f"Schedule end {s_end_date.isoformat()} exceeds release date but trade-off described -> +{pts} pts (partial).")
            else:
                msgs.append(f"Schedule end {s_end_date.isoformat()} exceeds release date {RELEASE_DATE.isoformat()} -> 0 pts for critical_path_and_dates.")
    else:
        msgs.append("schedule_summary.schedule_end missing or not ISO date -> 0 pts for critical_path_and_dates.")

    # Validate critical_path IDs exist
    cp_missing = [tid for tid in critical_path if tid not in task_map]
    if cp_missing:
        msgs.append(f"Critical path contains unknown task IDs: {cp_missing} (this reduces confidence).")

    # Also check functional/integration/regression tasks completion before code-freeze or explanation provided
    late_core_tasks = []
    for t in tasks:
        typ = t.get("type")
        if typ in ("test-execution", "regression"):
            try:
                start_offset = int(t.get("earliest_start_offset", 0))
                dur = int(t.get("duration_days", 1))
            except Exception:
                start_offset = 0
                dur = 1
            end_date = PROJECT_START_DATE + timedelta(days=start_offset + dur - 1)
            # If this is a core task and ends after code freeze, flag
            # We'll consider tasks tied to pre-prod (offset >= PREPROD_USABLE_OFFSET) as allowed after freeze for performance
            if end_date > CODE_FREEZE_DATE and start_offset < PREPROD_USABLE_OFFSET:
                # If the task explicitly references pre-prod or performance skip
                desc = (t.get("description") or "").lower()
                if "performance" in desc or "pre-prod" in desc or "preprod" in desc:
                    continue
                late_core_tasks.append((t.get("id"), end_date.isoformat()))
    if late_core_tasks:
        explanation = (candidate.get("final_comments","") + " " + candidate.get("contingency_plan","")).lower()
        if any(k in explanation for k in ["defer", "trade-off", "post-release", "post release", "scope freeze", "code-freeze", "code freeze"]):
            # partial credit
            pts = SCHEDULE_SUB["critical_path_and_dates"] * 0.5
            # but if schedule_end already exceeded release we may have added earlier
            score += 0  # already handled above - don't double count; just note
            msgs.append(f"Core functional/regression tasks end after code-freeze: {late_core_tasks} but candidate provided trade-offs -> reviewer should inspect. (no extra points awarded here).")
        else:
            # Deduct (i.e., no extra points) and flag mandatory failure later
            msgs.append(f"Core functional/regression tasks finish after code-freeze: {late_core_tasks} -> may violate mandatory rule (no trade-offs provided).")

    # Cap schedule score
    score = min(score, RUBRIC["schedule_and_resourcing"])
    return round(score, 2), msgs, schedule_info

def evaluate_risk_and_contingency(candidate):
    msgs = []
    score = 0.0
    rp = candidate.get("risk_prioritization")
    if not isinstance(rp, list) or len(rp) == 0:
        msgs.append("risk_prioritization missing or empty -> 0 for risk mapping and contingency.")
        return 0.0, msgs

    # Mapping (6 points): check entries for F1..F6 present and testing_priority values
    feature_ids = {entry.get("feature_id") for entry in rp if isinstance(entry, dict)}
    missing_features = [f"F{i}" for i in range(1,7) if f"F{i}" not in feature_ids]
    if not missing_features:
        # also check that critical features (F1,F5) are high priority (testing_priority 1-2)
        mapping_pts = RISK_SUB["mapping"]
        score += mapping_pts
        msgs.append(f"All F1..F6 present in risk_prioritization -> +{mapping_pts} pts.")
        # check priorities sensible
        for entry in rp:
            fid = entry.get("feature_id")
            bp = (entry.get("business_priority") or "").lower()
            tp = entry.get("testing_priority")
            if fid in ("F1", "F5"):
                if isinstance(tp, int) and tp in (1,2):
                    msgs.append(f"{fid} has testing_priority {tp} (appropriate).")
                else:
                    msgs.append(f"{fid} expected high testing_priority (1-2) but found {tp}.")
                    # small deduction from mapping (reduce overall mapping score by up to 2)
                    score -= 1.0
    else:
        # partial credit proportional to presence
        present = 6 - len(missing_features)
        pts = round(RISK_SUB["mapping"] * (present / 6.0), 2)
        score += pts
        msgs.append(f"Missing risk entries for: {missing_features} -> +{pts} pts (partial).")

    # Contingency plan (5 points) - check for specific actions for 1-3 days and >3 days
    cont = candidate.get("contingency_plan", "").lower()
    if not cont:
        msgs.append("contingency_plan missing -> 0 pts.")
    else:
        # look for mentions of 1-3 days and >3 days or "if slip" and explicit actions like "defer F4" or "reduce regression"
        has_short = any(x in cont for x in ["1-3", "1 to 3", "if slip 1", "slip 1", "compress", "compress regression", "reduce regression", "defer", "post-release"])
        has_long = any(x in cont for x in [">3", "more than 3", "if slip >3", "if slip > 3", "freeze scope", "scope freeze", "post release"])
        if has_short and has_long:
            score += RISK_SUB["contingency_plan"]
            msgs.append(f"Contingency plan covers both short (1-3 days) and long (>3 days) slips -> +{RISK_SUB['contingency_plan']} pts.")
        elif has_short or has_long:
            pts = round(RISK_SUB["contingency_plan"] * 0.6, 2)
            score += pts
            msgs.append(f"Contingency plan partially covers slips -> +{pts} pts (partial).")
        else:
            msgs.append("Contingency plan present but not specific about 1-3 or >3 day slips -> 0 pts for this sub-item.")

    # Defect/environment contingency (4 points) - look for dev SLA mention and env fallback
    defect_env_ok = False
    if "2 business" in cont or "2 business days" in cont or "2-business" in cont or "dev fix" in cont or "dev fixes" in cont:
        defect_env_ok = True
    if "env" in cont or "environment" in cont or "integration" in cont or "pre-prod" in cont or "provision" in cont or "provisioning" in cont:
        defect_env_ok = defect_env_ok or True
    if defect_env_ok:
        score += RISK_SUB["defect_env_contingency"]
        msgs.append(f"Contingency includes defect/env handling -> +{RISK_SUB['defect_env_contingency']} pts.")
    else:
        msgs.append("Contingency plan lacks explicit defect turnaround or environment provisioning handling -> 0 pts for defect/env contingency.")

    # cap
    score = max(0.0, min(score, RUBRIC["risk_and_contingency"]))
    return round(score, 2), msgs

def evaluate_metrics(candidate):
    msgs = []
    score = 0.0
    m = candidate.get("metrics_and_reporting", "")
    if not m or not isinstance(m, str):
        msgs.append("metrics_and_reporting missing -> 0 pts.")
        return 0.0, msgs
    text = m.lower()
    has_daily = "daily" in text or "day" in text
    has_defects = any(k in text for k in ["defect", "open defects", "p0", "p1", "severity", "aging"])
    has_progress = any(k in text for k in ["executed", "pass", "fail", "pass rate", "progress", "%", "percentage"])
    # award proportionally
    factors = sum([1 for x in (has_daily, has_defects, has_progress) if x])
    if factors == 3:
        score = METRICS_POINTS
        msgs.append(f"Metrics and reporting include daily cadence, defect tracking, and progress metrics -> +{METRICS_POINTS} pts.")
    elif factors == 2:
        score = round(METRICS_POINTS * 0.7, 2)
        msgs.append(f"Metrics include some key items (2/3) -> +{score} pts (partial).")
    elif factors == 1:
        score = round(METRICS_POINTS * 0.4, 2)
        msgs.append(f"Metrics include limited items (1/3) -> +{score} pts (partial).")
    else:
        msgs.append("Metrics/reporting lacks daily cadence and defect/progress metrics -> 0 pts.")
    return score, msgs

def check_mandatory_elements(candidate, schedule_info):
    """
    Mandatory elements (automatic fail if any missing):
    - include env-setup task with correct earliest_start_offset respecting integration provisioning
        -> There must be an env-setup task that starts at offset <=3 and its provisioning spans to usable offset 5
           (i.e., earliest_start_offset + duration_days >= 5)
    - schedule critical functional/integration work to finish before code-freeze or explain trade-offs explicitly
    - respect two-testers headcount: no single tester assigned >8 hours/day (we already computed in schedule)
    - provide measurable entry/exit criteria in test_strategy
    """
    reasons = []
    tasks = candidate.get("tasks", [])
    # env-setup check
    env_ok = False
    for t in tasks:
        if t.get("type") == "env-setup":
            try:
                start = int(t.get("earliest_start_offset", 0))
                dur = int(t.get("duration_days", 0))
                if start <= 3 and (start + dur) >= INTEGRATION_ENV_USABLE_OFFSET:
                    env_ok = True
                    break
            except Exception:
                continue
    if not env_ok:
        reasons.append("No env-setup task provisioning integration environment correctly (must start offset <=3 and reach usable offset 5).")

    # core functional/integration/regression completion before code-freeze or explanation
    late_core_tasks = []
    for t in tasks:
        typ = t.get("type")
        if typ in ("test-execution", "regression"):
            try:
                start_offset = int(t.get("earliest_start_offset", 0))
                dur = int(t.get("duration_days", 1))
            except Exception:
                start_offset = 0
                dur = 1
            end_date = PROJECT_START_DATE + timedelta(days=start_offset + dur - 1)
            # skip pre-prod/performance tasks
            desc = (t.get("description") or "").lower()
            if "performance" in desc or "pre-prod" in desc or "preprod" in desc:
                continue
            # It's expected that main functional/integration & regression finish before code-freeze
            if end_date > CODE_FREEZE_DATE:
                late_core_tasks.append((t.get("id"), end_date.isoformat()))
    if late_core_tasks:
        # check if candidate provided trade-offs in contingency/final_comments
        expl = (candidate.get("contingency_plan", "") + " " + candidate.get("final_comments", "")).lower()
        if not any(k in expl for k in ["defer", "trade-off", "post-release", "post release", "freeze scope", "scope freeze", "code-freeze", "code freeze"]):
            reasons.append(f"Core functional/regression tasks finish after code-freeze: {late_core_tasks} and no explicit trade-offs described.")

    # resource headcount check: per-day per-tester hours <= 8
    # Build per-day allocations same as earlier logic
    per_day_owner_hours = defaultdict(lambda: defaultdict(float))
    for t in tasks:
        try:
            start_offset = int(t.get("earliest_start_offset", 0))
            dur = int(t.get("duration_days", 1))
            eh = float(t.get("estimate_hours", 0))
            owner = t.get("owner")
        except Exception:
            continue
        if dur <= 0:
            dur = 1
        daily_hours = eh / dur if dur > 0 else eh
        start_date = PROJECT_START_DATE + timedelta(days=start_offset)
        for single_date in date_range(start_date, dur):
            per_day_owner_hours[single_date.isoformat()][owner] += daily_hours
    overloads = []
    for day, owner_map in per_day_owner_hours.items():
        for owner, hours in owner_map.items():
            if owner in ("Tester A", "Tester B") and hours > 8.01:
                overloads.append(f"{day}: {owner} has {round(hours,2)}h (>8h).")
    if overloads:
        reasons.append("Tester over-allocation detected (per-day >8h): " + "; ".join(overloads))

    # Entry/Exit criteria check in test_strategy
    strat = candidate.get("test_strategy", "")
    strat_lower = strat.lower() if strat else ""
    if not any(k in strat_lower for k in ["entry", "exit", "criteria", "p0", "p1", "sign-off", "go/no-go", "go no go"]):
        reasons.append("test_strategy missing measurable entry/exit criteria (e.g., 'P0/P1 defects resolved', 'critical tests pass', 'UAT sign-off').")

    return reasons

def write_results(results):
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        # also ensure overall_score variable present for potential downstream automation
        # it's already in results["overall_score"]
    except Exception as e:
        print(f"Failed to write results to {out_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    # make answer_key variable globally available for some checks
    global answer_key
    # Load answer_key argument early for some comparisons in evaluate_schedule
    if len(sys.argv) == 3:
        try:
            answer_key = safe_load_json(sys.argv[2])
        except Exception:
            answer_key = {}
    else:
        answer_key = {}
    main()