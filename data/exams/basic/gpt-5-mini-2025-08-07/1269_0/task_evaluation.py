# task_evaluation.py
"""
Automated grader for the "sales_agg" basic practical exam.

Usage:
    python task_evaluation.py <candidate_submission.json> <answer_key.json>

Generates test_results.json in the same directory as this script.

The grader:
- Loads candidate submission JSON and answer key JSON.
- Compares candidate outputs to expected outputs (tolerant normalization).
- Applies weighted scoring per exam specification.
- Produces a detailed breakdown with explanations for deductions.

Implementation notes:
- Uses only Python standard library.
- Robust to missing fields; attempts best-effort grading and records reasons.
"""

import json
import os
import sys
import traceback

# ---------- Helper utilities ----------

def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def safe_get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

def normalize_output(s):
    """
    Normalize program output for tolerant comparison:
    - Convert None to empty string.
    - Split into lines, strip trailing and leading whitespace of each line.
    - Remove leading/trailing blank lines.
    - Join with '\n'.
    """
    if s is None:
        return ''
    if not isinstance(s, str):
        s = str(s)
    # Normalize line endings
    lines = s.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    # Strip each line's trailing and leading whitespace
    stripped = [ln.strip() for ln in lines]
    # Remove leading/trailing empty lines
    while stripped and stripped[0] == '':
        stripped.pop(0)
    while stripped and stripped[-1] == '':
        stripped.pop()
    return '\n'.join(stripped)

def safe_str_contains(haystack, needles):
    """Case-insensitive containment check; needles may be string or iterable of strings."""
    if not isinstance(haystack, str):
        return False
    hay = haystack.lower()
    if isinstance(needles, str):
        needles = [needles]
    return all(n.lower() in hay for n in needles)

def count_comment_lines(content, language_hint=None):
    """
    Heuristic count of comment lines in a source file content.
    Supports simple detection for languages using:
    - '#' (python, shell)
    - '//' (c/c++/java/js)
    - '/*' ... '*/' (block comments counted naively)
    - triple-quoted strings in Python as docstrings counted as comments (naive: detect triple quotes)
    Returns (comment_lines, total_code_lines_nonblank).
    """
    if not isinstance(content, str) or content.strip() == '':
        return (0, 0)
    lines = content.splitlines()
    comment_lines = 0
    code_lines = 0
    in_block = False
    for ln in lines:
        stripped = ln.strip()
        if stripped == '':
            continue
        # Block comment start/end detection (naive)
        if '/*' in stripped:
            in_block = True
        if in_block:
            comment_lines += 1
            if '*/' in stripped:
                in_block = False
            continue
        # Python triple-quote docstring heuristic
        if stripped.startswith(('"""', "'''")) and stripped.count('"""') < 2 and stripped.count("'''") < 2:
            # start of triple-quote block - count as comment lines until we see closing triple
            comment_lines += 1
            quote = '"""' if stripped.startswith('"""') else "'''"
            if stripped.count(quote) >= 2:
                # opening and closing on same line
                pass
            else:
                # consume following lines until closing triple found
                # naive: we'll keep flag and continue marking as comments
                in_block = True
            continue
        if stripped.startswith('#') or stripped.startswith('//'):
            comment_lines += 1
        elif stripped.startswith(('"""', "'''")) and (stripped.endswith('"""') or stripped.endswith("'''")):
            comment_lines += 1
        else:
            # also treat lines that are only comment after code? simplistic: if '#' appears and code before it, still count as comment+code; count as code
            code_lines += 1
    # total non-blank lines considered as code_lines + comment_lines
    total_nonblank = code_lines + comment_lines
    return (comment_lines, total_nonblank)

# ---------- Grading logic and weights ----------

MAX_POINTS = 100.0

# Weight breakdown per exam overview (basic)
WEIGHTS = {
    'correctness': 50.0,      # functionality + tests correctness
    'documentation': 20.0,    # README + changelog
    'comments': 15.0,         # inline comments & clarity
    'tests_evidence': 10.0,   # test runs, build_instructions, captured outputs
    'revision': 5.0           # follow-up change implemented & documented
}

# ---------- Core grading function ----------

def grade_submission(candidate, answer_key):
    results = {
        'breakdown': {},
        'reasons': [],
        'total_points': 0.0,
        'max_points': MAX_POINTS,
    }

    # Basic presence checks
    required_top_fields = [
        'candidate_name', 'language', 'time_spent_minutes',
        'source_files', 'build_instructions', 'readme',
        'changelog', 'tests', 'assumptions', 'comments_coverage',
        'self_score_estimate', 'signature'
    ]
    missing_fields = [f for f in required_top_fields if f not in candidate]
    if missing_fields:
        results['reasons'].append(f"Missing required top-level fields: {missing_fields}")

    # --- Correctness (50 points) ---
    correctness_score = 0.0
    correctness_max = WEIGHTS['correctness']
    correctness_details = []

    # 1) Check presence of source file named sales_agg*
    source_files = candidate.get('source_files', [])
    has_sales_agg = False
    any_source = False
    sales_agg_filenames = []
    total_comment_ratio = 0.0  # for comments scoring later
    comment_lines_total = 0
    total_nonblank_lines_total = 0

    for sf in source_files:
        fname = safe_get(sf, 'filename', '')
        content = safe_get(sf, 'content', '')
        if fname:
            any_source = True
            if 'sales_agg' in fname.lower():
                has_sales_agg = True
                sales_agg_filenames.append(fname)
        # accumulate comment stats
        cl, tn = count_comment_lines(content)
        comment_lines_total += cl
        total_nonblank_lines_total += tn

    if any_source:
        correctness_details.append("Source files provided.")
    else:
        correctness_details.append("No source files found.")

    if has_sales_agg:
        correctness_details.append(f"Found source file(s) named like sales_agg: {sales_agg_filenames}.")
        correctness_score += 5.0 * (correctness_max / correctness_max) * (1.0) * (1.0)  # allocate 5 points out of correctness below
        # We'll credit 5 points out of 50 for presence of sales_agg and build instructions later in tests_evidence too.
        # For clarity, directly add 5 points to correctness here.
        # But to keep total weights consistent, we will cap later by WEIGHTS.
        # Instead of complicating, we'll allocate fixed 5 points here.
        # But ensure not to exceed correctness_max; we'll manage sums later.
        correctness_score = min(correctness_score, correctness_max)
    else:
        correctness_details.append("No source file named sales_agg found (deduction).")

    # 2) Compare test outputs: use answer_key['tests'] expected outputs as ground truth.
    expected_tests = answer_key.get('tests', [])
    candidate_tests = candidate.get('tests', [])

    # Build mapping of candidate tests by test_name for quick lookup
    cand_tests_by_name = {}
    for t in candidate_tests:
        tn = safe_get(t, 'test_name', '')
        if tn:
            cand_tests_by_name[tn] = t

    # For matching when test_name missing, allow fuzzy match by command substrings
    # We'll iterate over expected_tests and try to find best match
    test_matches = []
    tests_points_total = 45.0  # part of correctness weight allocated to tests (out of 50)
    single_test_point = tests_points_total / max(1, len(expected_tests))
    tests_passed = 0

    for et in expected_tests:
        et_name = safe_get(et, 'test_name', '')
        et_expected_output = safe_get(et, 'expected_output', '')
        et_command = safe_get(et, 'command', '')
        matched = None
        # Try exact name match
        if et_name and et_name in cand_tests_by_name:
            matched = cand_tests_by_name[et_name]
        else:
            # Try to find candidate test whose command contains recognizable substrings
            for ct in candidate_tests:
                cmd = safe_get(ct, 'command', '')
                if not cmd:
                    continue
                # Match by presence of key substrings from expected command (e.g., sample_sales.csv, malformed_sales.csv, --min-revenue)
                # Build list of keywords
                keywords = []
                if 'sample_sales.csv' in et_command:
                    keywords.append('sample_sales.csv')
                if 'malformed_sales.csv' in et_command:
                    keywords.append('malformed_sales.csv')
                if '--min-revenue' in et_command or '-m' in et_command:
                    keywords.append('--min-revenue')
                # Also include product file names if present directly in command
                if keywords and all(k in cmd for k in keywords):
                    matched = ct
                    break
        if matched is None and candidate_tests:
            # fallback: pick first candidate test that hasn't been matched yet
            for ct in candidate_tests:
                if ct not in [m[1] for m in test_matches]:
                    matched = ct
                    break

        # Evaluate matched test
        if matched is None:
            reason = f"Expected test '{et_name}' not found in candidate tests."
            correctness_details.append(reason)
            test_matches.append((et_name, None, False, reason))
            continue

        cand_actual = normalize_output(safe_get(matched, 'actual_output', ''))
        # Prefer using expected_output from answer key for strictness
        expected_norm = normalize_output(et_expected_output)
        passed = (cand_actual == expected_norm)
        if passed:
            tests_passed += 1
            correctness_score += single_test_point
            correctness_details.append(f"Test '{et_name}' passed (command: {safe_get(matched,'command','')}).")
        else:
            # Provide diff-like reason (first differing line)
            reason = f"Test '{et_name}' failed. Expected (normalized):\n{expected_norm}\nActual (normalized):\n{cand_actual}"
            correctness_details.append(reason)
        test_matches.append((et_name, matched, passed, None if passed else reason))

    # Cap correctness_score to correctness_max
    correctness_score = min(correctness_score, correctness_max)
    results['breakdown']['correctness'] = {
        'score': round(correctness_score, 2),
        'max_score': correctness_max,
        'details': correctness_details
    }
    results['total_points'] += correctness_score

    # --- Documentation (20 points) ---
    documentation_score = 0.0
    doc_max = WEIGHTS['documentation']
    doc_details = []
    readme = candidate.get('readme', '')
    changelog = candidate.get('changelog', [])

    if isinstance(readme, str) and readme.strip() != '':
        documentation_score += doc_max * 0.6  # 60% of doc weight for README presence/content
        doc_details.append("README present and non-empty.")
        # Check README mentions build/run commands or sample filenames
        if safe_str_contains(readme, ['sample_sales.csv', 'malformed_sales.csv']):
            documentation_score += doc_max * 0.15
            doc_details.append("README mentions sample input filenames.")
        if safe_str_contains(readme, ['--min-revenue', '-m']):
            documentation_score += doc_max * 0.1
            doc_details.append("README documents --min-revenue flag.")
        # Check README mentions assumptions
        if safe_str_contains(readme, ['assum', 'header', 'missing', 'round']):
            documentation_score += doc_max * 0.05
            doc_details.append("README documents assumptions (header/missing/rounding).")
    else:
        doc_details.append("README missing or empty (deduction).")

    # Changelog scoring
    if isinstance(changelog, list) and len(changelog) >= 1:
        # Base points for changelog presence
        documentation_score += doc_max * 0.1
        doc_details.append("Changelog present.")
        # Prefer multiple entries and mention of revision (--min-revenue)
        if len(changelog) >= 2:
            documentation_score += doc_max * 0.05
            doc_details.append("Changelog has multiple entries.")
        # Check changelog entries for mention of min-revenue/filter
        changelog_text = json.dumps(changelog).lower()
        if '--min-revenue' in changelog_text or 'min-revenue' in changelog_text or 'filter' in changelog_text:
            documentation_score += doc_max * 0.05
            doc_details.append("Changelog documents follow-up revision (--min-revenue/filter).")
    else:
        doc_details.append("Changelog missing or empty (deduction).")

    # Cap documentation_score
    documentation_score = min(documentation_score, doc_max)
    results['breakdown']['documentation'] = {
        'score': round(documentation_score, 2),
        'max_score': doc_max,
        'details': doc_details
    }
    results['total_points'] += documentation_score

    # --- Comments & clarity (15 points) ---
    comments_score = 0.0
    comments_max = WEIGHTS['comments']
    comments_details = []

    comments_coverage_field = candidate.get('comments_coverage', '')
    if isinstance(comments_coverage_field, str) and comments_coverage_field.strip() != '':
        comments_score += comments_max * 0.2  # base credit for having self-assessed comments coverage
        comments_details.append("comments_coverage field present.")
    else:
        comments_details.append("comments_coverage field missing or empty.")

    # Heuristic: evaluate actual comment ratio in source files computed earlier
    if total_nonblank_lines_total > 0:
        comment_ratio = comment_lines_total / total_nonblank_lines_total
        # Scale: >=8% comment ratio -> full points (remaining 80%); linear scaling otherwise
        if comment_ratio >= 0.08:
            comments_score += comments_max * 0.8
            comments_details.append(f"Comment density is high ({comment_ratio:.2%}), awarding full comment points.")
        else:
            partial = comments_max * 0.8 * (comment_ratio / 0.08)
            comments_score += partial
            comments_details.append(f"Comment density low ({comment_ratio:.2%}); awarding partial points ({partial:.2f}).")
    else:
        comments_details.append("No code lines to analyze for comment coverage (deduction).")

    # Cap comments_score
    comments_score = min(comments_score, comments_max)
    results['breakdown']['comments'] = {
        'score': round(comments_score, 2),
        'max_score': comments_max,
        'details': comments_details
    }
    results['total_points'] += comments_score

    # --- Tests & evidence (10 points) ---
    tests_evidence_score = 0.0
    tests_evidence_max = WEIGHTS['tests_evidence']
    tests_evidence_details = []

    # Check number of tests provided
    num_tests_provided = len(candidate.get('tests', []))
    if num_tests_provided >= 2:
        tests_evidence_score += tests_evidence_max * 0.2  # base for >=2 tests
        tests_evidence_details.append(f"{num_tests_provided} test runs provided (>=2).")
    if num_tests_provided >= 3:
        tests_evidence_score += tests_evidence_max * 0.2
        tests_evidence_details.append("At least 3 test runs provided.")

    # Check build_instructions mention required commands
    build_instr = candidate.get('build_instructions', '')
    if isinstance(build_instr, str) and build_instr.strip() != '':
        # Look for sample file names and min-revenue flag
        found_sample = 'sample_sales.csv' in build_instr
        found_malformed = 'malformed_sales.csv' in build_instr
        found_filter = ('--min-revenue' in build_instr) or ('-m' in build_instr)
        if found_sample and found_malformed and found_filter:
            tests_evidence_score += tests_evidence_max * 0.3
            tests_evidence_details.append("build_instructions include commands for sample, malformed, and --min-revenue runs.")
        else:
            # Partial credit for including some commands
            score_add = tests_evidence_max * 0.3 * ((1 if found_sample else 0) + (1 if found_malformed else 0) + (1 if found_filter else 0)) / 3.0
            tests_evidence_score += score_add
            tests_evidence_details.append(f"build_instructions partially include required commands (sample:{found_sample}, malformed:{found_malformed}, filter:{found_filter}).")
    else:
        tests_evidence_details.append("build_instructions missing or empty (deduction).")

    # Check actual_output captured for tests
    candidate_tests_list = candidate.get('tests', [])
    actual_outputs_nonempty = all((normalize_output(safe_get(t, 'actual_output','')) != '') for t in candidate_tests_list) and len(candidate_tests_list) > 0
    if actual_outputs_nonempty:
        tests_evidence_score += tests_evidence_max * 0.2
        tests_evidence_details.append("All candidate tests include non-empty actual_output.")
    else:
        tests_evidence_details.append("Some tests missing actual_output or actual_output empty (deduction).")

    # Check tests included 'passed' boolean flags
    passed_flag_present = all('passed' in t for t in candidate_tests_list) and len(candidate_tests_list) > 0
    if passed_flag_present:
        tests_evidence_score += tests_evidence_max * 0.1
        tests_evidence_details.append("Tests include 'passed' boolean flags.")
    else:
        tests_evidence_details.append("Tests missing 'passed' flags (deduction).")

    # Cap tests_evidence_score
    tests_evidence_score = min(tests_evidence_score, tests_evidence_max)
    results['breakdown']['tests_evidence'] = {
        'score': round(tests_evidence_score, 2),
        'max_score': tests_evidence_max,
        'details': tests_evidence_details
    }
    results['total_points'] += tests_evidence_score

    # --- Revision (5 points) ---
    revision_score = 0.0
    revision_max = WEIGHTS['revision']
    revision_details = []

    # Check changelog or readme or source files mention --min-revenue or min-revenue or filter
    combined_text = ''
    combined_text += '\n' + (readme or '')
    combined_text += '\n' + json.dumps(changelog or [])
    for sf in source_files:
        combined_text += '\n' + safe_get(sf, 'content', '')
    combined_text_l = combined_text.lower()

    if '--min-revenue' in combined_text_l or 'min-revenue' in combined_text_l or '-m' in combined_text_l or 'min revenue' in combined_text_l or 'filter' in combined_text_l:
        revision_score = revision_max
        revision_details.append("Follow-up revision (--min-revenue filter) implemented and documented.")
    else:
        # partial credit if changelog has entries (indicating revision attempts)
        if isinstance(changelog, list) and len(changelog) >= 2:
            revision_score = revision_max * 0.4
            revision_details.append("Changelog suggests multiple edits but no explicit mention of --min-revenue detected.")
        else:
            revision_details.append("No evidence of follow-up revision (--min-revenue) found.")

    results['breakdown']['revision'] = {
        'score': round(revision_score, 2),
        'max_score': revision_max,
        'details': revision_details
    }
    results['total_points'] += revision_score

    # --- Additional sanity checks and deductions summary ---
    # Time spent check
    time_spent = candidate.get('time_spent_minutes', None)
    if isinstance(time_spent, int) or isinstance(time_spent, float):
        if time_spent > 90:
            results['reasons'].append(f"time_spent_minutes is {time_spent} (>90) — exam time limit exceeded (informational).")
    else:
        results['reasons'].append("time_spent_minutes missing or not numeric (informational).")

    # signature presence
    signature = candidate.get('signature', '')
    if not isinstance(signature, str) or signature.strip() == '':
        results['reasons'].append("Missing signature line (deduction in documentation).")

    # Ensure totals are rounded and consistent
    results['total_points'] = float(round(results['total_points'], 2))
    results['overall_percentage'] = round((results['total_points'] / results['max_points']) * 100.0, 2)
    results['pass'] = results['overall_percentage'] >= 80.0

    # Include test match summary
    test_summary = []
    for et_name, matched, passed, reason in test_matches:
        entry = {
            'expected_test_name': et_name,
            'matched_candidate_test': safe_get(matched, 'test_name', None) if isinstance(matched, dict) else None,
            'command': safe_get(matched, 'command', None) if isinstance(matched, dict) else None,
            'passed': bool(passed)
        }
        if reason:
            entry['reason'] = reason
        test_summary.append(entry)
    results['test_summary'] = test_summary

    return results

# ---------- Main script ----------

def main():
    # Validate args
    if len(sys.argv) != 3:
        print("Usage: python task_evaluation.py <candidate_submission.json> <answer_key.json>", file=sys.stderr)
        sys.exit(2)

    cand_path = sys.argv[1]
    key_path = sys.argv[2]

    # Determine output path (same directory as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'test_results.json')

    # Load JSON inputs
    try:
        candidate = load_json_file(cand_path)
    except Exception as e:
        err = {
            'error': f"Failed to load candidate submission JSON from {cand_path}: {str(e)}",
            'traceback': traceback.format_exc()
        }
        with open(output_path, 'w', encoding='utf-8') as fout:
            json.dump(err, fout, indent=2)
        print(f"Error: could not load candidate submission. Details written to {output_path}", file=sys.stderr)
        sys.exit(1)

    try:
        answer_key = load_json_file(key_path)
    except Exception as e:
        err = {
            'error': f"Failed to load answer key JSON from {key_path}: {str(e)}",
            'traceback': traceback.format_exc()
        }
        with open(output_path, 'w', encoding='utf-8') as fout:
            json.dump(err, fout, indent=2)
        print(f"Error: could not load answer key. Details written to {output_path}", file=sys.stderr)
        sys.exit(1)

    # Perform grading
    results = grade_submission(candidate, answer_key)

    # Add overall_score alias as required
    results['overall_score'] = results.get('overall_percentage', 0.0)

    # Save results
    try:
        with open(output_path, 'w', encoding='utf-8') as fout:
            json.dump(results, fout, indent=2)
        print(f"Grading complete. Results written to {output_path}")
    except Exception as e:
        print(f"Failed to write results to {output_path}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()