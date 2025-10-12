# task_evaluation.py
"""
Automated grader for the "Updating Automated Test Scripts to Ensure Currency — Basic Practical Exam".

Usage:
    python task_evaluation.py path/to/test_submission.json path/to/answer_key.json

Outputs:
    Creates test_results.json in the current directory with a detailed breakdown of scores,
    deductions, and an overall percentage score (overall_score).
"""

import json
import os
import re
import sys
from datetime import datetime

# --- Utility parsing functions ---


def safe_load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading JSON file '{path}': {e}")


def parse_pytest_counts(output):
    """
    Parse pytest-like output for collected/passed/failed counts.
    Returns a dict: {'collected': int or None, 'passed': int, 'failed': int, 'other': dict}
    If values not found, default to 0 where reasonable.
    """
    if not output or not isinstance(output, str):
        return {'collected': None, 'passed': 0, 'failed': 0, 'other': {}}
    collected = None
    m = re.search(r"collected\s+(\d+)\s+items", output, re.IGNORECASE)
    if m:
        collected = int(m.group(1))
    # Look for summary patterns
    passed = 0
    failed = 0
    # "3 passed", "1 failed", etc. There can be multiple tokens like "1 passed, 2 failed"
    # We'll capture all occurrences and sum matching tokens.
    for token in re.findall(r"(\d+)\s+(passed|failed|skipped|xfailed|xpassed|errors?)", output, re.IGNORECASE):
        num = int(token[0])
        label = token[1].lower()
        if label == 'passed':
            passed += num
        elif label == 'failed':
            failed += num
        # skip others for now
    # As fallback, try "X failed in" message for failures
    m2 = re.search(r"(\d+)\s+failed\s+in", output, re.IGNORECASE)
    if m2:
        failed = int(m2.group(1))
    m3 = re.search(r"(\d+)\s+passed\s+in", output, re.IGNORECASE)
    if m3:
        passed = int(m3.group(1))
    # If collected known and passed/failed both zero, infer passed = collected
    if collected is not None and passed == 0 and failed == 0:
        # Sometimes summary omitted; try to infer from "X passed" earlier
        # If output contains "=== X passed in", we handled above. Otherwise assume all passed.
        # But if earlier there were FAIL lines, we should not assume.
        if "FAILED" not in output and "failed" not in output:
            passed = collected
    return {'collected': collected, 'passed': passed, 'failed': failed, 'other': {}}


def text_contains_ignore_space(a, b):
    if a is None or b is None:
        return False
    return b.strip().lower() in a.strip().lower()


# --- Scoring functions ---


def score_tests_passing(initial_output, updated_output, max_points=50):
    """
    Scoring logic for 'Tests passing after changes' category (50 points).
    Partial credit proportional to number of previously-failing tests fixed.
    """
    parsed_initial = parse_pytest_counts(initial_output)
    parsed_updated = parse_pytest_counts(updated_output)

    initial_failed = parsed_initial.get('failed', 0)
    updated_failed = parsed_updated.get('failed', 0)

    reasons = []
    earned = 0.0

    # If initial had no failures, consider candidate may have run a different baseline.
    if initial_failed == 0:
        # If updated shows all passing -> full credit; else partial 0
        if updated_failed == 0:
            earned = max_points
            reasons.append("Initial run reported 0 failed tests; updated run reports 0 failed -> full credit.")
        else:
            earned = 0.0
            reasons.append("Initial run had 0 failures but updated run still reports failures -> no points for this category.")
    else:
        fixed = initial_failed - updated_failed
        if fixed <= 0:
            earned = 0.0
            reasons.append(f"No failing tests were fixed (initial_failed={initial_failed}, updated_failed={updated_failed}).")
        else:
            fraction = fixed / initial_failed
            earned = round(max_points * fraction, 2)
            reasons.append(f"Fixed {fixed} of {initial_failed} previously failing tests -> {fraction*100:.1f}% of {max_points} points = {earned} points.")
        if updated_failed == 0:
            # full credit override for complete fix
            earned = max_points
            reasons.append("All previously failing tests are now passing -> full credit for this category.")
    details = {
        'initial_failed': initial_failed,
        'updated_failed': updated_failed,
        'reasons': reasons
    }
    return earned, max_points, details


def score_correctness_minimality(candidate_modified_files, answer_expected_modified_files, max_points=20):
    """
    Score correctness & minimality.
    Rules:
    - If any modified file is outside 'tests/' (or is app.py), this is disallowed -> 0 points (and fatal)
    - Otherwise, if number of modified files <= expected -> full points
    - If more modified than expected, proportional downscaling: score = max_points * (expected / num_modified)
    - Also require original_content present for each modified file; missing original_content halves this category.
    """
    reasons = []
    earned = 0.0
    fatal = False

    num_modified = len(candidate_modified_files or [])
    expected_num = len(answer_expected_modified_files or []) if answer_expected_modified_files is not None else num_modified or 0

    # Check for disallowed modifications
    non_test_files = []
    missing_original = []
    for f in candidate_modified_files or []:
        path = f.get('path', '')
        # Normalize path separator
        p = path.replace('\\', '/')
        if p == 'app.py' or not p.startswith('tests/'):
            non_test_files.append(path)
        if not f.get('original_content'):
            missing_original.append(path)

    if non_test_files:
        reasons.append(f"Disallowed modifications detected: non-test files modified: {non_test_files}. Modifying application code is not allowed.")
        earned = 0.0
        fatal = True
        return earned, max_points, {'reasons': reasons, 'fatal': fatal}

    # All modified files are test files
    if num_modified == 0:
        reasons.append("No modified files reported; expected minimal test-only edits.")
        earned = 0.0
        return earned, max_points, {'reasons': reasons, 'fatal': fatal}

    # Score proportional to expected/actual (cap at 1)
    if expected_num <= 0:
        # No expected reference; award full points if only test files modified
        base_score = max_points
        reasons.append("No expected modified-files count provided in answer key; awarding base points for test-only modifications.")
    else:
        ratio = min(1.0, expected_num / float(num_modified))
        base_score = round(max_points * ratio, 2)
        if num_modified <= expected_num:
            reasons.append(f"Modified {num_modified} files (<= expected {expected_num}) -> full/minimal changes -> base score {base_score}.")
        else:
            reasons.append(f"Modified {num_modified} files (expected {expected_num}) -> proportional score based on minimality -> base score {base_score}.")

    # Penalty for missing original_content
    if missing_original:
        reasons.append(f"Original content missing for files: {missing_original}. Halving this category's score.")
        base_score = round(base_score * 0.5, 2)

    earned = base_score
    return earned, max_points, {'reasons': reasons, 'fatal': fatal}


def score_clarity_repro(candidate_submission, candidate_modified_files, max_points=15):
    """
    Score clarity of explanations and reproducibility.
    - Require explanation_of_changes entries for each modified file with non-empty why_changed and what_changed.
    - Require reproduce_instructions to mention 'pytest' (per README).
    - Require initial_test_results and updated_test_results present.
    """
    reasons = []
    earned = 0.0

    explanations = candidate_submission.get('explanation_of_changes', [])
    reproduce_instructions = candidate_submission.get('reproduce_instructions', '') or ''
    initial_output = candidate_submission.get('initial_test_results', '') or ''
    updated_output = candidate_submission.get('updated_test_results', '') or ''

    num_modified = len(candidate_modified_files or [])
    if num_modified == 0:
        reasons.append("No modified files reported; cannot evaluate explanations.")
        return earned, max_points, {'reasons': reasons}

    # Map explanations by path
    exp_map = {}
    for e in explanations:
        p = e.get('path')
        if p:
            exp_map[p] = e

    missing_explanations = []
    incomplete_explanations = []
    for f in candidate_modified_files or []:
        path = f.get('path')
        ex = exp_map.get(path)
        if not ex:
            missing_explanations.append(path)
        else:
            if not ex.get('why_changed') or not ex.get('what_changed'):
                incomplete_explanations.append(path)

    # Scoring: start full, deduct per missing/incomplete
    earned = max_points
    if missing_explanations:
        deduction = round((len(missing_explanations) / num_modified) * max_points, 2)
        earned -= deduction
        reasons.append(f"Missing explanations for files: {missing_explanations} -> -{deduction} points.")
    if incomplete_explanations:
        deduction = round((len(incomplete_explanations) / num_modified) * max_points * 0.5, 2)
        earned -= deduction
        reasons.append(f"Incomplete explanations (missing why/what) for files: {incomplete_explanations} -> -{deduction} points.")

    # Reproduce instructions check
    if 'pytest' not in reproduce_instructions.lower():
        deduction = round(max_points * 0.3, 2)
        earned -= deduction
        reasons.append(f"reproduce_instructions does not mention pytest -> -{deduction} points.")

    # Check initial/updated outputs presence
    if not initial_output.strip():
        earned = max(0.0, earned - round(max_points * 0.25, 2))
        reasons.append("initial_test_results is missing or empty -> deducted 25% of this category.")
    if not updated_output.strip():
        earned = max(0.0, earned - round(max_points * 0.25, 2))
        reasons.append("updated_test_results is missing or empty -> deducted 25% of this category.")

    if earned < 0:
        earned = 0.0

    return round(earned, 2), max_points, {'reasons': reasons}


def score_test_quality(candidate_modified_files, max_points=10):
    """
    Score test quality / maintainability improvements.
    There are three expected improvements (each ~max_points/3):
      - user_api test uses 'username' instead of 'userName'
      - ui test uses stable data-test-id or LOGIN_BUTTON_TEST_ID constant
      - fixtures expiry_date updated to be >= app.TODAY (2025)
    Award points proportionally.
    """
    reasons = []
    earned = 0.0
    per_item = max_points / 3.0
    checks_passed = 0
    checks = []

    # Create a map of path -> new_content
    path_map = {f.get('path'): f.get('new_content', '') for f in candidate_modified_files or []}

    # 1) user_api fix
    user_api_ok = False
    user_api_paths = ['tests/test_user_api.py', 'test_user_api.py']
    for p in user_api_paths:
        content = path_map.get(p)
        if content and re.search(r"\busername\b", content):
            # ensure 'userName' not still asserted
            if "userName" not in content:
                user_api_ok = True
    if user_api_ok:
        checks_passed += 1
        reasons.append("User API test updated to use 'username' (lowercase).")
    else:
        reasons.append("User API test does not appear to be updated to 'username' or still contains 'userName'.")

    # 2) ui test fix (stable selector)
    ui_ok = False
    ui_paths = ['tests/test_ui.py', 'test_ui.py']
    for p in ui_paths:
        content = path_map.get(p)
        if content:
            if 'data-test-id' in content or 'LOGIN_BUTTON_TEST_ID' in content or 'login-btn' in content:
                ui_ok = True
    if ui_ok:
        checks_passed += 1
        reasons.append("UI test updated to use stable data-test-id / helper constant.")
    else:
        reasons.append("UI test not updated to use stable selector or still relies solely on visible text.")

    # 3) fixtures expiry_date updated
    fixtures_ok = False
    fixtures_paths = ['tests/fixtures.py', 'fixtures.py']
    for p in fixtures_paths:
        content = path_map.get(p)
        if content:
            # find expiry_date string like 'YYYY-MM-DD'
            m = re.search(r"'expiry_date'\s*:\s*'(\d{4})-(\d{2})-(\d{2})'", content)
            if m:
                year = int(m.group(1))
                if year >= 2025:
                    fixtures_ok = True
    if fixtures_ok:
        checks_passed += 1
        reasons.append("Fixture expiry_date updated to be after app.TODAY (>=2025).")
    else:
        reasons.append("Fixture expiry_date not updated or remains before app.TODAY (2025).")

    earned = round(per_item * checks_passed, 2)
    details = {'checks_passed': checks_passed, 'total_checks': 3, 'reasons': reasons}
    return earned, max_points, details


def score_time_management(candidate_submission, candidate_modified_files, max_points=5):
    """
    Score time management and completeness:
    - time_spent_minutes present and <=90 -> full points
    - modified_files present and include original_content for each -> required
    - initial and updated outputs present
    """
    reasons = []
    earned = 0.0

    time_spent = candidate_submission.get('time_spent_minutes')
    if time_spent is None:
        reasons.append("time_spent_minutes is missing.")
        return earned, max_points, {'reasons': reasons}

    try:
        tm = int(time_spent)
    except Exception:
        reasons.append("time_spent_minutes not an integer.")
        return earned, max_points, {'reasons': reasons}

    if tm <= 0:
        reasons.append("time_spent_minutes indicates 0 or negative time.")
        return earned, max_points, {'reasons': reasons}

    if tm > 90:
        # still give partial credit
        earned += round(max_points * 0.5, 2)
        reasons.append(f"time_spent_minutes = {tm} > 90 -> partial credit for time management.")
    else:
        earned += max_points
        reasons.append(f"time_spent_minutes = {tm} within allowed 90 minutes -> full credit for time management.")

    # completeness: check modified_files present and original_content included
    if not candidate_modified_files or len(candidate_modified_files) == 0:
        earned = max(0.0, earned - round(max_points * 0.5, 2))
        reasons.append("modified_files missing or empty -> deducted 50% of this category.")
    else:
        missing_original = [f.get('path') for f in candidate_modified_files if not f.get('original_content')]
        if missing_original:
            earned = max(0.0, earned - round(max_points * 0.4, 2))
            reasons.append(f"original_content missing for files: {missing_original} -> deducted 40% of this category.")

    # check initial and updated test results presence
    if not candidate_submission.get('initial_test_results', '').strip():
        earned = max(0.0, earned - round(max_points * 0.25, 2))
        reasons.append("initial_test_results missing or empty -> deducted 25% of this category.")
    if not candidate_submission.get('updated_test_results', '').strip():
        earned = max(0.0, earned - round(max_points * 0.25, 2))
        reasons.append("updated_test_results missing or empty -> deducted 25% of this category.")

    if earned < 0:
        earned = 0.0
    return round(earned, 2), max_points, {'reasons': reasons}


# --- Main grading orchestration ---


def grade_submission(candidate_path, answer_key_path, output_path='test_results.json'):
    results = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'candidate_path': candidate_path,
        'answer_key_path': answer_key_path,
        'categories': [],
        'deductions': [],
        'fatal_fail': False
    }
    try:
        candidate = safe_load_json(candidate_path)
    except Exception as e:
        results['error'] = str(e)
        with open(output_path, 'w', encoding='utf-8') as out:
            json.dump(results, out, indent=2)
        print(f"Error: {e}")
        return

    try:
        answer_key = safe_load_json(answer_key_path)
    except Exception as e:
        results['error'] = str(e)
        with open(output_path, 'w', encoding='utf-8') as out:
            json.dump(results, out, indent=2)
        print(f"Error: {e}")
        return

    # Extract needed pieces
    candidate_modified_files = candidate.get('modified_files', [])
    answer_modified_files = answer_key.get('modified_files', [])

    initial_output = candidate.get('initial_test_results', '')
    updated_output = candidate.get('updated_test_results', '')

    # Category 1
    c1_earned, c1_max, c1_details = score_tests_passing(initial_output, updated_output, max_points=50)
    results['categories'].append({
        'name': 'Tests passing after changes',
        'earned': c1_earned,
        'max': c1_max,
        'details': c1_details
    })

    # Category 2
    c2_earned, c2_max, c2_details = score_correctness_minimality(candidate_modified_files, answer_modified_files, max_points=20)
    results['categories'].append({
        'name': 'Correctness and minimality of changes',
        'earned': c2_earned,
        'max': c2_max,
        'details': c2_details
    })
    if c2_details.get('fatal'):
        results['fatal_fail'] = True
        results['deductions'].append("Fatal: Modified disallowed files (application source). Automatic fail per exam rules.")

    # Category 3
    c3_earned, c3_max, c3_details = score_clarity_repro(candidate, candidate_modified_files, max_points=15)
    results['categories'].append({
        'name': 'Clarity of explanations and reproducibility',
        'earned': c3_earned,
        'max': c3_max,
        'details': c3_details
    })
    results['deductions'].extend(c3_details.get('reasons', []))

    # Category 4
    c4_earned, c4_max, c4_details = score_test_quality(candidate_modified_files, max_points=10)
    results['categories'].append({
        'name': 'Test quality / maintainability improvements',
        'earned': c4_earned,
        'max': c4_max,
        'details': c4_details
    })

    # Category 5
    c5_earned, c5_max, c5_details = score_time_management(candidate, candidate_modified_files, max_points=5)
    results['categories'].append({
        'name': 'Time management and completeness',
        'earned': c5_earned,
        'max': c5_max,
        'details': c5_details
    })
    results['deductions'].extend(c5_details.get('reasons', []))

    # Aggregate totals
    total_earned = 0.0
    total_max = 0.0
    for cat in results['categories']:
        total_earned += float(cat['earned'])
        total_max += float(cat['max'])

    # Additional checks/deductions
    # Disallowed change: app.py modified -> fatal
    modified_paths = [f.get('path', '') for f in candidate_modified_files or []]
    modified_norm = [p.replace('\\', '/') for p in modified_paths]
    if any(p == 'app.py' or p.endswith('/app.py') for p in modified_norm):
        results['fatal_fail'] = True
        results['deductions'].append("Candidate modified application source file 'app.py' which is disallowed. Automatic fail.")
    # If fatal, set overall_score to 0
    if results['fatal_fail']:
        overall_score = 0.0
        total_earned = 0.0
        # Set categories earned to 0 to reflect fail
        for cat in results['categories']:
            cat['earned'] = 0.0
        results['note'] = "Fatal rule violation: application source modified. Overall score set to 0."
    else:
        overall_score = round((total_earned / total_max) * 100.0, 2) if total_max > 0 else 0.0

    results['total_earned'] = round(total_earned, 2)
    results['total_max'] = round(total_max, 2)
    results['percentage'] = overall_score
    results['overall_score'] = overall_score  # required variable
    # Also include parsed pytest counts for reviewer convenience
    results['parsed_initial'] = parse_pytest_counts(initial_output)
    results['parsed_updated'] = parse_pytest_counts(updated_output)

    # Prepare human-friendly deductions / notes
    # If not all previously failing tests fixed, add explicit deduction message
    parsed_initial = results['parsed_initial']
    parsed_updated = results['parsed_updated']
    if parsed_initial.get('failed', 0) > 0:
        fixed = parsed_initial.get('failed', 0) - parsed_updated.get('failed', 0)
        if fixed < parsed_initial.get('failed', 0):
            results['deductions'].append(
                f"Not all previously failing tests were fixed: {fixed}/{parsed_initial.get('failed', 0)} fixed."
            )
        else:
            results['deductions'].append("All previously failing tests appear to have been fixed.")

    # Remove duplicates in deductions
    results['deductions'] = list(dict.fromkeys(results['deductions']))

    # Save to output file
    try:
        with open(output_path, 'w', encoding='utf-8') as out:
            json.dump(results, out, indent=2)
    except Exception as e:
        print(f"Error writing results to '{output_path}': {e}")
        return

    print(f"Grading complete. Results saved to '{output_path}'. Overall score: {overall_score}%.")


# --- Entry point ---


def main():
    if len(sys.argv) < 3:
        print("Usage: python task_evaluation.py path/to/test_submission.json path/to/answer_key.json")
        sys.exit(2)
    candidate_path = sys.argv[1]
    answer_key_path = sys.argv[2]
    # output in same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'test_results.json')
    try:
        grade_submission(candidate_path, answer_key_path, output_path=output_path)
    except Exception as e:
        print(f"Unexpected error during grading: {e}")
        # Attempt to write minimal error file
        err_out = {
            'error': str(e),
            'generated_at': datetime.utcnow().isoformat() + 'Z'
        }
        try:
            with open(output_path, 'w', encoding='utf-8') as out:
                json.dump(err_out, out, indent=2)
        except Exception:
            pass
        sys.exit(1)


if __name__ == '__main__':
    main()