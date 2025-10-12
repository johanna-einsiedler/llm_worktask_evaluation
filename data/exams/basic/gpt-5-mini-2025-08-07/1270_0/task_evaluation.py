#!/usr/bin/env python3
"""
task_evaluation.py

Automated grader for the Basic Practical Exam — Inventory Manager.

Usage:
    python3 task_evaluation.py test_submission.json answer_key.json

Output:
    Writes test_results.json in the same directory where the script is executed.
    The output contains detailed breakdown of scores, explanations for deductions,
    and overall percentage score (stored as overall_score in JSON).

Notes:
- Uses only the Python standard library.
- Robust to missing fields; produces helpful messages in the output.
"""

import json
import os
import sys
import traceback

# Scoring rubric constants
MAX_POINTS = 100.0

# Breakdown mapping per rubric (max points)
RUBRIC = {
    "functional": {
        "description": "Functional correctness (add/get/update/remove, persistence, import/export)",
        "max": 60.0,
        # subcomponents mapping to internal checks
        "sub": {
            "add": 10.0,
            "get": 10.0,
            "update": 10.0,
            "remove": 10.0,
            "persistence": 10.0,
            "import_export": 10.0
        }
    },
    "validation": {
        "description": "Validation & error handling (unique id, numeric validation, malformed CSV)",
        "max": 15.0,
        "sub": {
            "unique_id": 5.0,
            "quantity_price_validation": 5.0,
            "malformed_csv_handling": 5.0
        }
    },
    "demo_tests": {
        "description": "Demonstration & tests (demo runner + at least 4 automated tests)",
        "max": 15.0,
        "sub": {
            "demo_runner": 7.0,
            "automated_tests": 8.0
        }
    },
    "quality_docs": {
        "description": "Code quality & documentation",
        "max": 10.0,
        "sub": {
            "readme": 4.0,
            "code_quality": 3.0,
            "no_external_libs": 3.0
        }
    }
}

# Helper tolerant string search
def contains_case_insensitive(haystack, needle):
    if haystack is None:
        return False
    return needle.lower() in haystack.lower()

def safe_get_file_content(files_list, filename):
    if not isinstance(files_list, list):
        return None
    for f in files_list:
        if not isinstance(f, dict):
            continue
        if f.get("filename") == filename:
            return f.get("content", "")
    return None

def find_file_by_prefix(files_list, prefix):
    if not isinstance(files_list, list):
        return None
    for f in files_list:
        if not isinstance(f, dict):
            continue
        name = f.get("filename", "")
        if name.startswith(prefix):
            return f.get("content", "")
    return None

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def normalize_inventory(inv):
    """
    Normalize inventory JSON structure into list of dicts with string id keys,
    numeric quantity (int) and price (float). Returns list or None if invalid.
    """
    if inv is None:
        return None
    try:
        if isinstance(inv, dict):
            # Unexpected shape: convert to list if single item keyed by id
            return [inv]
        if isinstance(inv, list):
            normalized = []
            for item in inv:
                if not isinstance(item, dict):
                    continue
                nm = {}
                nm['id'] = str(item.get('id')) if item.get('id') is not None else None
                nm['name'] = item.get('name')
                # normalize quantity to int if possible
                qty = item.get('quantity')
                try:
                    nm['quantity'] = int(qty) if qty is not None else None
                except Exception:
                    try:
                        nm['quantity'] = int(float(qty))
                    except Exception:
                        nm['quantity'] = None
                # normalize price to float
                pr = item.get('price')
                try:
                    nm['price'] = float(pr) if pr is not None else None
                except Exception:
                    nm['price'] = None
                normalized.append(nm)
            return normalized
        # else unknown shape
        return None
    except Exception:
        return None

def score_candidate(candidate, answer_key):
    """
    Compute scores and explanations based on candidate submission and answer key.
    Returns a dict describing scores, breakdown, comments and overall_score.
    """
    results = {
        "breakdown": {},
        "total_awarded": 0.0,
        "total_possible": MAX_POINTS,
        "deductions": [],
        "notes": []
    }

    # Helper to award points for a sub-item, with comment
    def award(category_key, sub_key, points_awarded, comment):
        cat = results["breakdown"].setdefault(category_key, {})
        cat[sub_key] = {
            "awarded": round(points_awarded, 2),
            "possible": RUBRIC[category_key]["sub"][sub_key],
            "comment": comment
        }
        results["total_awarded"] += points_awarded

    # gather candidate top-level fields
    # Validate candidate structure
    # required top-level fields per exam: candidate_name, email, time_taken_minutes, language, run_commands, files, demo_stdout, test_results, final_inventory_json, notes
    files = candidate.get("files", [])
    run_commands = candidate.get("run_commands", [])
    demo_stdout = candidate.get("demo_stdout", "") or ""
    test_results = candidate.get("test_results", [])
    final_inventory = candidate.get("final_inventory_json", None)
    readme_txt = safe_get_file_content(files, "README.md") or safe_get_file_content(files, "README") or ""
    inventory_py = safe_get_file_content(files, "inventory.py") or ""
    demo_runner_py = safe_get_file_content(files, "demo_runner.py") or ""
    tests_py = safe_get_file_content(files, "tests.py") or ""
    import_csv = safe_get_file_content(files, "import.csv") or ""
    final_inventory_file_content = safe_get_file_content(files, "final_inventory.json")
    # If final_inventory.json file included in files, prefer that as exported content text
    if final_inventory_file_content:
        try:
            parsed = json.loads(final_inventory_file_content)
            # if candidate.final_inventory_json missing, use this file as the final inventory data
            if final_inventory is None:
                final_inventory = parsed
        except Exception:
            # if parsing fails, still leave final_inventory as is
            pass

    # Normalized final inventories
    candidate_inv_norm = normalize_inventory(final_inventory)
    answer_inv_norm = normalize_inventory(answer_key.get("final_inventory_json"))

    # ----- Functional correctness (60) -----
    # Subchecks: add/get/update/remove/persistence/import_export
    func_info = RUBRIC["functional"]
    # ADD operation (10)
    # We check demo_stdout for evidence of adding I100 and I101, or test_results indicating add passing.
    add_award = 0.0
    add_comment_parts = []
    # Look for 'Added' lines with I100 and I101 in demo stdout
    added_i100 = any(("added" in line.lower() and "i100" in line.lower()) or ("i100" in line and "Added" in line) for line in demo_stdout.splitlines())
    added_i101 = any(("added" in line.lower() and "i101" in line.lower()) or ("i101" in line and "Added" in line) for line in demo_stdout.splitlines())
    # Also check tests results for add success (some submissions put tests verifying add)
    tr_names = [str(tr.get("name", "")).lower() for tr in test_results if isinstance(tr, dict)]
    tr_pass_map = {str(tr.get("name", "")).lower(): bool(tr.get("passed", False)) for tr in test_results if isinstance(tr, dict)}
    add_test_present = any("add" in name and tr_pass_map.get(name, False) for name in tr_pass_map)
    # Award logic:
    # - If demo shows both adds succeeded -> full 10
    # - Else if demo shows one add succeeded or tests show add passed -> half credit
    if added_i100 and added_i101:
        add_award = func_info["sub"]["add"]
        add_comment = "Both adds (I100 and I101) observed in demo stdout."
    elif added_i100 or added_i101 or add_test_present:
        add_award = func_info["sub"]["add"] / 2.0
        add_comment = "Partial evidence of add operation (one add observed or tests assert add)."
    else:
        add_award = 0.0
        add_comment = "No clear evidence of add operation for required items in demo/test outputs."
    award("functional", "add", add_award, add_comment)

    # GET operation (10)
    # Look for get usage in tests/demo: a JSON printed for an individual id or a test named 'get'
    get_award = 0.0
    get_evidence = False
    # If any test output includes 'get:' or test name contains 'get' and passed
    for tr in test_results:
        if not isinstance(tr, dict):
            continue
        name = str(tr.get("name", "")).lower()
        out = str(tr.get("output", "")).lower()
        passed = bool(tr.get("passed", False))
        if ("get" in name and passed) or ("get:" in out and passed) or ('"id"' in out and passed and 'get' in name):
            get_evidence = True
            break
    # Demo may print a get result, but demo sequence doesn't include get. Rely on tests.
    if get_evidence:
        get_award = func_info["sub"]["get"]
        get_comment = "Found passing test evidence for get (or get result in test outputs)."
    else:
        # fallback: check demo_stdout for JSON lines containing an 'id' field (may indicate get)
        if any('"id"' in line or "'id'" in line for line in demo_stdout.splitlines()):
            get_award = func_info["sub"]["get"] / 2.0
            get_comment = "Possible get JSON output observed in demo stdout (partial evidence)."
        else:
            get_award = 0.0
            get_comment = "No evidence of get operation in tests or demo outputs."
    award("functional", "get", get_award, get_comment)

    # UPDATE operation (10)
    # Check final_inventory contains I100 with quantity == 12 (standard demo update)
    update_award = 0.0
    update_comment = ""
    if candidate_inv_norm is not None:
        found_i100 = next((it for it in candidate_inv_norm if it.get("id") == "I100"), None)
        if found_i100 and isinstance(found_i100.get("quantity"), int) and found_i100.get("quantity") == 12:
            update_award = func_info["sub"]["update"]
            update_comment = "I100 found in final inventory with quantity 12 (update applied)."
        else:
            # maybe tests show update; check test_results for update test passed
            for tr in test_results:
                if not isinstance(tr, dict):
                    continue
                name = str(tr.get("name", "")).lower()
                passed = bool(tr.get("passed", False))
                if "update" in name and passed:
                    update_award = func_info["sub"]["update"]
                    update_comment = "Passing test indicates update works (test output shows updated quantity)."
                    break
            if update_award == 0.0:
                update_award = 0.0
                update_comment = "No evidence that update operation applied (I100 quantity not 12 and no passing update test)."
    else:
        update_award = 0.0
        update_comment = "Final inventory could not be parsed; cannot verify update."
    award("functional", "update", update_award, update_comment)

    # REMOVE operation (10)
    remove_award = 0.0
    remove_comment = ""
    # Check demo_stdout for removal of I101 OR final inventory omits I101
    removed_i101_demo = any(("removed" in line.lower() and "i101" in line.lower()) for line in demo_stdout.splitlines())
    if removed_i101_demo:
        remove_award = func_info["sub"]["remove"]
        remove_comment = "Demo stdout shows removal of I101."
    else:
        # if final inventory does not contain I101, that's evidence removal
        if candidate_inv_norm is not None:
            if not any(it.get("id") == "I101" for it in candidate_inv_norm):
                remove_award = func_info["sub"]["remove"]
                remove_comment = "Final inventory does not contain I101 (evidence removal)."
            else:
                remove_award = 0.0
                remove_comment = "I101 still present in final inventory and no removal message observed."
        else:
            # fallback: check test results for remove test
            removed_by_test = any("remove" in str(tr.get("name", "")).lower() and bool(tr.get("passed", False)) for tr in test_results if isinstance(tr, dict))
            if removed_by_test:
                remove_award = func_info["sub"]["remove"]
                remove_comment = "Passing test indicates remove operation works."
            else:
                remove_award = 0.0
                remove_comment = "No evidence of remove operation."
    award("functional", "remove", remove_award, remove_comment)

    # Persistence (10)
    persistence_award = 0.0
    persistence_comment = ""
    # If final_inventory_json present and parsed into a non-empty list, award points
    if candidate.get("final_inventory_json") is not None:
        # if final inventory array exists and has items
        try:
            ci = candidate.get("final_inventory_json")
            if (isinstance(ci, list) and len(ci) >= 1) or (isinstance(ci, dict) and len(ci.keys()) >= 1):
                persistence_award = func_info["sub"]["persistence"]
                persistence_comment = "final_inventory_json present (persistence/export produced output)."
            else:
                persistence_award = func_info["sub"]["persistence"] / 2.0
                persistence_comment = "final_inventory_json present but empty."
        except Exception:
            persistence_award = 0.0
            persistence_comment = "final_inventory_json could not be interpreted."
    else:
        # maybe final_inventory.json file included in files and parsed earlier
        if final_inventory_file_content:
            persistence_award = func_info["sub"]["persistence"]
            persistence_comment = "final_inventory.json file content included in submission files (persistence)."
        else:
            persistence_award = 0.0
            persistence_comment = "No final inventory data found; persistence could not be verified."
    award("functional", "persistence", persistence_award, persistence_comment)

    # Import/Export (10)
    import_export_award = 0.0
    import_export_comment = ""
    # If demo stdout contains import messages about added/skipped and export message, award points.
    demo_lines_lower = [ln.lower() for ln in demo_stdout.splitlines()]
    import_ann = any("import" in ln and ("added" in ln or "skipped" in ln or "import complete" in ln) for ln in demo_lines_lower)
    export_ann = any("exported" in ln for ln in demo_lines_lower)
    # Compare candidate final inventory to answer key final inventory (if answer key provided)
    if answer_inv_norm is not None and candidate_inv_norm is not None:
        # Compare sets of ids and quantities roughly
        ans_ids = {it.get("id"): it for it in answer_inv_norm if it.get("id")}
        cand_ids = {it.get("id"): it for it in candidate_inv_norm if it.get("id")}
        # exact match awarding
        if set(ans_ids.keys()) == set(cand_ids.keys()):
            # check quantities and prices tolerance
            match_count = 0
            tot = len(ans_ids.keys())
            for idk, ans_item in ans_ids.items():
                cand_item = cand_ids.get(idk)
                if not cand_item:
                    continue
                # compare quantity and price with tolerance for numeric types
                if ans_item.get("quantity") == cand_item.get("quantity") and (ans_item.get("price") == cand_item.get("price")):
                    match_count += 1
            if tot > 0 and match_count == tot:
                import_export_award = func_info["sub"]["import_export"]
                import_export_comment = "Final inventory matches answer key expected final inventory exactly."
            else:
                import_export_award = func_info["sub"]["import_export"] / 2.0
                import_export_comment = "Final inventory IDs match expected, but some quantities/prices differ."
        else:
            # maybe at least contains imported item (I102) -> partial credit
            if any(it.get("id") == "I102" for it in candidate_inv_norm) and import_ann:
                import_export_award = func_info["sub"]["import_export"] / 2.0
                import_export_comment = "Import appears to have added I102 (partial match), but overall final inventory differs from expected."
            else:
                import_export_award = 0.0
                import_export_comment = "Final inventory does not match expected answer key and import/export evidence missing."
    else:
        # fallback to checking demo stdout for import/export lines
        if import_ann and export_ann:
            import_export_award = func_info["sub"]["import_export"] / 2.0
            import_export_comment = "Demo shows import/export activity but final inventory could not be compared to answer key."
        else:
            import_export_award = 0.0
            import_export_comment = "No clear evidence of import/export operations in demo outputs."
    award("functional", "import_export", import_export_award, import_export_comment)

    # ----- Validation & error handling (15) -----
    # unique_id (5)
    unique_award = 0.0
    unique_comment = ""
    # Evidence: test_results 'add_duplicate_id' passed OR demo_stdout shows 'already exists' for duplicate add attempt OR import shows 'Skipped duplicate id'
    duplicate_evidence = False
    # Check test_results names
    for tr in test_results:
        if not isinstance(tr, dict):
            continue
        name = str(tr.get("name", "")).lower()
        out = str(tr.get("output", "")).lower()
        passed = bool(tr.get("passed", False))
        if "duplicate" in name and passed:
            duplicate_evidence = True
            break
        if "already exists" in out and passed:
            duplicate_evidence = True
            break
    # Check demo stdout for duplicate messaging
    if not duplicate_evidence:
        if any("already exists" in ln or "skipped duplicate" in ln or "skipped duplicate id" in ln for ln in demo_lines_lower):
            duplicate_evidence = True
    if duplicate_evidence:
        unique_award = RUBRIC["validation"]["sub"]["unique_id"]
        unique_comment = "Evidence of duplicate ID handling (add or import) in demo/tests."
    else:
        unique_award = 0.0
        unique_comment = "No evidence of unique ID enforcement for duplicates in demo or tests."
    award("validation", "unique_id", unique_award, unique_comment)

    # quantity_price_validation (5)
    qp_award = 0.0
    qp_comment = ""
    # Check README mentions validation rules for quantity/price
    readme_lower = readme_txt.lower() if isinstance(readme_txt, str) else ""
    readme_mentions_q = ("quantity" in readme_lower and "integer" in readme_lower) or ("quantity" in readme_lower and ">=" in readme_lower) or ("quantity must" in readme_lower)
    readme_mentions_p = ("price" in readme_lower and ">=" in readme_lower) or ("price must" in readme_lower) or ("price" in readme_lower and "numeric" in readme_lower)
    # Check tests or demo show malformed numeric rejection
    numeric_validation_evidence = False
    # demo_lines_lower already defined
    if any("invalid integer" in ln or "quantity must" in ln or "price must" in ln or "invalid numeric" in ln for ln in demo_lines_lower):
        numeric_validation_evidence = True
    # tests outputs may include 'Skipped malformed' or messages
    for tr in test_results:
        if not isinstance(tr, dict):
            continue
        out = str(tr.get("output", "")).lower()
        if "invalid integer" in out or "quantity must" in out or "price must" in out:
            numeric_validation_evidence = True
            break
    if (readme_mentions_q and readme_mentions_p) or numeric_validation_evidence:
        qp_award = RUBRIC["validation"]["sub"]["quantity_price_validation"]
        qp_comment = "Validation for numeric quantity/price documented and/or demonstrated."
    elif readme_mentions_q or readme_mentions_p:
        qp_award = RUBRIC["validation"]["sub"]["quantity_price_validation"] / 2.0
        qp_comment = "Partial documentation of numeric validation in README or partial evidence."
    else:
        qp_award = 0.0
        qp_comment = "No evidence or documentation of numeric validation for quantity/price."
    award("validation", "quantity_price_validation", qp_award, qp_comment)

    # malformed_csv_handling (5)
    mal_award = 0.0
    mal_comment = ""
    # Check demo stdout or tests indicate malformed CSV row handling (skip/abort)
    mal_evidence = any("malformed" in ln or "skipped malformed" in ln or "skipped malformed row" in ln or "invalid integer for quantity" in ln for ln in demo_lines_lower)
    for tr in test_results:
        if not isinstance(tr, dict):
            continue
        out = str(tr.get("output", "")).lower()
        if "malformed" in out or "skipped malformed" in out:
            mal_evidence = True
            break
    if mal_evidence:
        mal_award = RUBRIC["validation"]["sub"]["malformed_csv_handling"]
        mal_comment = "Malformed CSV rows are handled gracefully (skipped or reported) as evidenced in demo/tests."
    else:
        mal_award = 0.0
        mal_comment = "No evidence that malformed CSV rows are handled gracefully."
    award("validation", "malformed_csv_handling", mal_award, mal_comment)

    # ----- Demonstration & tests (15) -----
    # demo_runner (7)
    demo_award = 0.0
    demo_comment = ""
    # Check run_commands includes demo_runner command and demo_stdout is non-empty
    rcmds_lower = [str(c).lower() for c in run_commands] if isinstance(run_commands, list) else []
    demo_cmd_present = any("demo_runner" in c or "demo_runner.py" in c for c in rcmds_lower)
    demo_stdout_present = isinstance(demo_stdout, str) and len(demo_stdout.strip()) > 0
    # Check demo printed final inventory JSON (presence of 'final inventory json' or final_inventory_json field)
    final_printed_in_demo = any("final inventory json" in ln.lower() for ln in demo_lines_lower)
    if demo_cmd_present and demo_stdout_present and final_printed_in_demo and candidate.get("final_inventory_json") is not None:
        demo_award = RUBRIC["demo_tests"]["sub"]["demo_runner"]
        demo_comment = "Demo runner command present and demo_stdout contains final inventory JSON printed (and final_inventory_json present)."
    elif demo_stdout_present and candidate.get("final_inventory_json") is not None:
        demo_award = RUBRIC["demo_tests"]["sub"]["demo_runner"] / 2.0
        demo_comment = "Demo stdout / final inventory present but run_commands may not list demo_runner or final JSON not printed explicitly."
    else:
        demo_award = 0.0
        demo_comment = "Demo runner evidence missing or incomplete."
    award("demo_tests", "demo_runner", demo_award, demo_comment)

    # automated_tests (8)
    tests_award = 0.0
    tests_comment = ""
    # Must have at least 4 test result objects and tests should pass
    if isinstance(test_results, list) and len(test_results) >= 4:
        passed_count = sum(1 for t in test_results if isinstance(t, dict) and bool(t.get("passed", False)))
        # award proportionally: full 8 if >=4 tests and all 4 pass; partial otherwise
        if passed_count >= 4:
            # cap at 4 baseline: if more tests exist, still only need 4
            tests_award = RUBRIC["demo_tests"]["sub"]["automated_tests"]
            tests_comment = f"{passed_count} tests passed (>=4)."
        else:
            tests_award = (passed_count / 4.0) * RUBRIC["demo_tests"]["sub"]["automated_tests"]
            tests_comment = f"{passed_count} tests passed (need at least 4 for full credit)."
    else:
        # Maybe tests.py not run or results not provided; partial credit if some passing tests exist
        if isinstance(test_results, list) and len(test_results) > 0:
            passed_count = sum(1 for t in test_results if isinstance(t, dict) and bool(t.get("passed", False)))
            tests_award = (passed_count / 4.0) * RUBRIC["demo_tests"]["sub"]["automated_tests"]
            tests_comment = f"Only {len(test_results)} test result(s) provided; {passed_count} passed."
        else:
            tests_award = 0.0
            tests_comment = "No test_results provided; cannot award automated tests credit."
    # Cap award
    if tests_award > RUBRIC["demo_tests"]["sub"]["automated_tests"]:
        tests_award = RUBRIC["demo_tests"]["sub"]["automated_tests"]
    award("demo_tests", "automated_tests", tests_award, tests_comment)

    # ----- Code quality & documentation (10) -----
    # README (4)
    readme_award = 0.0
    readme_comment = ""
    if readme_txt and isinstance(readme_txt, str) and len(readme_txt.strip()) > 0:
        # must include demo and tests run commands, persistence choice, and duplicate-import policy
        has_demo_cmd = 'python3 demo_runner.py' in readme_txt or 'demo_runner.py' in readme_txt
        has_tests_cmd = 'python3 tests.py' in readme_txt or 'tests.py' in readme_txt
        has_persistence = 'inventory.json' in readme_txt.lower() or 'sqlite' in readme_txt.lower()
        has_dup_policy = 'duplicate' in readme_txt.lower() or 'overwrite' in readme_txt.lower() or 'skip' in readme_txt.lower() or 'error' in readme_txt.lower()
        score_parts = 0
        if has_demo_cmd and has_tests_cmd:
            score_parts += 2  # half of README credit
        if has_persistence:
            score_parts += 1  # quarter
        if has_dup_policy:
            score_parts += 1  # quarter
        # total out of 4
        readme_award = min(RUBRIC["quality_docs"]["sub"]["readme"], float(score_parts))
        readme_comment = "README presence and content checked: demo/tests cmds, persistence, duplicate policy."
    else:
        readme_award = 0.0
        readme_comment = "README.md missing or empty."
    award("quality_docs", "readme", readme_award, readme_comment)

    # Code quality (3)
    code_quality_award = 0.0
    code_quality_comment = ""
    # Heuristic: inventory.py contains functions (def ), has some comments (#), and a main block
    if inventory_py and isinstance(inventory_py, str) and len(inventory_py.strip()) > 0:
        has_def = "def " in inventory_py
        has_comments = "#" in inventory_py
        has_main = ("if __name__" in inventory_py) or ("argparse" in inventory_py)
        score_q = 0
        if has_def:
            score_q += 1
        if has_comments:
            score_q += 1
        if has_main:
            score_q += 1
        code_quality_award = (score_q / 3.0) * RUBRIC["quality_docs"]["sub"]["code_quality"]
        code_quality_comment = "Heuristic checks for defs/comments/main block in inventory.py."
    else:
        code_quality_award = 0.0
        code_quality_comment = "inventory.py missing or empty; cannot evaluate code quality."
    award("quality_docs", "code_quality", code_quality_award, code_quality_comment)

    # No external libs (3)
    no_ext_award = 0.0
    no_ext_comment = ""
    # Check for suspicious imports like 'requests' or 'pip' or other non-stdlib names in inventory.py
    if inventory_py and isinstance(inventory_py, str):
        lower_inv = inventory_py.lower()
        # crude list of known non-stdlib module names common mistakes
        suspicious = ["requests", "numpy", "pandas", "pip", "boto3", "sqlalchemy"]
        found_suspicious = any(s in lower_inv for s in suspicious)
        if found_suspicious:
            no_ext_award = 0.0
            no_ext_comment = "Found imports of likely external libraries in inventory.py; external libs not allowed."
        else:
            no_ext_award = RUBRIC["quality_docs"]["sub"]["no_external_libs"]
            no_ext_comment = "No obvious external third-party libraries imported in inventory.py (heuristic)."
    else:
        no_ext_award = 0.0
        no_ext_comment = "inventory.py missing; cannot verify external library usage."
    award("quality_docs", "no_external_libs", no_ext_award, no_ext_comment)

    # finalize totals
    # Round totals to 2 decimal places
    results["total_awarded"] = round(results["total_awarded"], 2)
    results["percentage"] = round((results["total_awarded"] / results["total_possible"]) * 100.0, 2) if results["total_possible"] else 0.0
    results["overall_score"] = results["percentage"]
    # Add pass/fail based on >=80 threshold
    results["pass"] = results["overall_score"] >= 80.0

    # Add some contextual info and evidence summary for the evaluator
    evidence_summary = {
        "demo_cmd_present": demo_cmd_present,
        "demo_stdout_present": bool(demo_stdout.strip()),
        "demo_contains_final_json_print": any("final inventory json" in ln.lower() for ln in demo_lines_lower),
        "test_results_count": len(test_results) if isinstance(test_results, list) else 0,
        "test_results_passed_count": sum(1 for t in test_results if isinstance(t, dict) and bool(t.get("passed", False))),
        "candidate_final_inventory_parsed_count": len(candidate_inv_norm) if candidate_inv_norm is not None else 0,
        "answer_key_final_inventory_parsed_count": len(answer_inv_norm) if answer_inv_norm is not None else 0
    }
    results["evidence_summary"] = evidence_summary

    return results

def main():
    # Entry point
    if len(sys.argv) != 3:
        print("Usage: python3 task_evaluation.py test_submission.json answer_key.json")
        sys.exit(2)
    cand_path = sys.argv[1]
    answer_path = sys.argv[2]

    # Prepare output path
    out_filename = "test_results.json"
    try:
        # Load candidate submission JSON
        try:
            candidate = load_json_file(cand_path)
        except Exception as e:
            err_info = f"Failed to load candidate submission JSON from '{cand_path}': {e}"
            out = {
                "error": err_info,
                "overall_score": 0.0
            }
            with open(out_filename, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
            print(err_info)
            sys.exit(1)

        # Load answer key JSON
        try:
            answer_key = load_json_file(answer_path)
        except Exception as e:
            err_info = f"Failed to load answer key JSON from '{answer_path}': {e}"
            out = {
                "error": err_info,
                "overall_score": 0.0
            }
            with open(out_filename, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
            print(err_info)
            sys.exit(1)

        # Perform scoring
        grading = score_candidate(candidate, answer_key)

        # Save results to test_results.json
        with open(out_filename, "w", encoding="utf-8") as fh:
            json.dump(grading, fh, indent=2)

        # Print concise summary to stdout
        print("Grading complete. Results written to", out_filename)
        print("Total awarded: {}/{}  ({}%)".format(grading["total_awarded"], grading["total_possible"], grading["overall_score"]))
        if grading.get("pass"):
            print("PASS: Candidate score meets threshold (>= 80%)")
        else:
            print("FAIL: Candidate score below threshold (80%)")
        sys.exit(0)
    except Exception as e:
        # Unexpected error: write diagnostics
        tb = traceback.format_exc()
        out = {
            "error": "Internal grader error",
            "exception": str(e),
            "traceback": tb,
            "overall_score": 0.0
        }
        with open(out_filename, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print("Internal grader error; details written to", out_filename)
        sys.exit(1)

if __name__ == "__main__":
    main()