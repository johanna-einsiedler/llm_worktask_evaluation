#!/usr/bin/env python3
"""
Task Evaluation Script for Basic Practical Exam: Program Revision, Repair, and Expansion
Evaluates candidate submissions against the answer key and provides detailed scoring.
"""

import json
import sys
import os
from typing import Dict, Any, List, Tuple

class TaskEvaluator:
    def __init__(self):
        self.max_points = 100
        self.scoring_weights = {
            'critical_bugs': 50,  # 25 points each for BUG001 and BUG002
            'feature_enhancement': 30,
            'performance_optimization': 20
        }
        
    def load_json_file(self, filename: str) -> Dict[str, Any]:
        """Load and parse JSON file with error handling."""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in '{filename}': {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading '{filename}': {e}")
            sys.exit(1)

    def evaluate_critical_bugs(self, submission: Dict[str, Any], answer_key: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Evaluate critical bug fixes (50 points total)."""
        points = 0
        details = {}
        
        # Check if bugs_fixed field exists and is valid
        bugs_fixed = submission.get('bugs_fixed', [])
        expected_bugs = answer_key.get('bugs_fixed', ['BUG001', 'BUG002'])
        
        details['expected_bugs'] = expected_bugs
        details['submitted_bugs'] = bugs_fixed
        details['bug_scores'] = {}
        
        # BUG001 - Stock persistence (25 points)
        if 'BUG001' in bugs_fixed:
            points += 25
            details['bug_scores']['BUG001'] = {
                'points': 25,
                'max_points': 25,
                'status': 'Fixed',
                'description': 'Stock updates not saved to CSV file'
            }
        else:
            details['bug_scores']['BUG001'] = {
                'points': 0,
                'max_points': 25,
                'status': 'Not Fixed',
                'description': 'Stock updates not saved to CSV file'
            }
        
        # BUG002 - Menu pause (25 points)
        if 'BUG002' in bugs_fixed:
            points += 25
            details['bug_scores']['BUG002'] = {
                'points': 25,
                'max_points': 25,
                'status': 'Fixed',
                'description': 'Menu doesn\'t pause for user input'
            }
        else:
            details['bug_scores']['BUG002'] = {
                'points': 0,
                'max_points': 25,
                'status': 'Not Fixed',
                'description': 'Menu doesn\'t pause for user input'
            }
        
        details['total_points'] = points
        details['max_points'] = 50
        
        return points, details

    def evaluate_feature_enhancement(self, submission: Dict[str, Any], answer_key: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Evaluate low stock alert feature implementation (30 points total)."""
        points = 0
        details = {}
        
        feature_complete = submission.get('feature_complete', False)
        details['feature_complete_claimed'] = feature_complete
        
        # Basic implementation check (30 points if claimed complete)
        if feature_complete:
            points = 30
            details['implementation_status'] = 'Complete'
            details['points_breakdown'] = {
                'low_stock_detection_logic': 15,
                'file_output_format': 10,
                'display_formatting': 5
            }
        else:
            details['implementation_status'] = 'Incomplete or Not Implemented'
            details['points_breakdown'] = {
                'low_stock_detection_logic': 0,
                'file_output_format': 0,
                'display_formatting': 0
            }
        
        # Expected feature specifications from answer key
        if 'detailed_solutions' in answer_key and 'feature_enhancement' in answer_key['detailed_solutions']:
            feature_specs = answer_key['detailed_solutions']['feature_enhancement']
            details['expected_specifications'] = {
                'threshold': feature_specs.get('threshold', 10),
                'expected_low_stock_items': feature_specs.get('expected_low_stock_items', ['P002', 'P006', 'P009', 'P012']),
                'reorder_formula': feature_specs.get('reorder_formula', 'current_quantity + 50'),
                'file_format': feature_specs.get('file_format', 'PRODUCT_ID,PRODUCT_NAME,CURRENT_QTY,REORDER_QTY'),
                'menu_option': feature_specs.get('menu_option', '5')
            }
        
        details['total_points'] = points
        details['max_points'] = 30
        
        return points, details

    def evaluate_performance_optimization(self, submission: Dict[str, Any], answer_key: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Evaluate performance optimization implementation (20 points total)."""
        points = 0
        details = {}
        
        performance_optimized = submission.get('performance_optimized', False)
        details['performance_optimized_claimed'] = performance_optimized
        
        # Basic optimization check (20 points if claimed complete)
        if performance_optimized:
            points = 20
            details['optimization_status'] = 'Complete'
            details['points_breakdown'] = {
                'loop_removal': 15,
                'functionality_preservation': 5
            }
        else:
            details['optimization_status'] = 'Not Implemented'
            details['points_breakdown'] = {
                'loop_removal': 0,
                'functionality_preservation': 0
            }
        
        # Expected optimization details from answer key
        if 'detailed_solutions' in answer_key and 'performance_fix' in answer_key['detailed_solutions']:
            perf_specs = answer_key['detailed_solutions']['performance_fix']
            details['expected_optimizations'] = {
                'remove_loop': perf_specs.get('remove_loop', 'for _ in range(100):'),
                'remove_divisions': perf_specs.get('remove_divisions', 'All /100 compensation calculations'),
                'expected_improvement': perf_specs.get('expected_improvement', '99% faster execution time')
            }
        
        details['total_points'] = points
        details['max_points'] = 20
        
        return points, details

    def evaluate_testing_completion(self, submission: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate testing completion status (informational, not scored)."""
        testing_completed = submission.get('testing_completed', False)
        
        return {
            'testing_completed_claimed': testing_completed,
            'status': 'Complete' if testing_completed else 'Not Completed',
            'note': 'Testing is required for full credit but not separately scored'
        }

    def determine_performance_level(self, overall_score: float) -> Dict[str, Any]:
        """Determine performance level based on overall score."""
        if overall_score >= 80:
            return {
                'level': 'Pass - Entry Level Competent',
                'description': 'Demonstrates minimum required knowledge and skills for entry-level role',
                'recommendation': 'Candidate meets basic competency requirements'
            }
        elif overall_score >= 60:
            return {
                'level': 'Marginal - Needs Development',
                'description': 'Some functionality working but significant gaps remain',
                'recommendation': 'Additional training and development recommended'
            }
        else:
            return {
                'level': 'Fail - Insufficient Competency',
                'description': 'Major functionality remains broken',
                'recommendation': 'Candidate does not meet minimum competency requirements'
            }

    def check_automatic_failure_conditions(self, submission: Dict[str, Any], bug_points: int) -> List[str]:
        """Check for automatic failure conditions."""
        failures = []
        
        # Check if both critical bugs are fixed
        bugs_fixed = submission.get('bugs_fixed', [])
        if 'BUG001' not in bugs_fixed:
            failures.append('Critical BUG001 (stock persistence) not fixed')
        if 'BUG002' not in bugs_fixed:
            failures.append('Critical BUG002 (menu pause) not fixed')
        
        # Check JSON format completeness
        required_fields = ['bugs_fixed', 'feature_complete', 'performance_optimized', 'testing_completed']
        missing_fields = [field for field in required_fields if field not in submission]
        if missing_fields:
            failures.append(f'Missing required JSON fields: {missing_fields}')
        
        return failures

    def evaluate_submission(self, submission_file: str, answer_key_file: str) -> Dict[str, Any]:
        """Main evaluation function."""
        # Load files
        submission = self.load_json_file(submission_file)
        answer_key = self.load_json_file(answer_key_file)
        
        # Initialize results
        results = {
            'submission_file': submission_file,
            'answer_key_file': answer_key_file,
            'evaluation_timestamp': None,
            'overall_score': 0,
            'performance_level': {},
            'detailed_scores': {},
            'automatic_failures': [],
            'summary': {}
        }
        
        # Add timestamp
        from datetime import datetime
        results['evaluation_timestamp'] = datetime.now().isoformat()
        
        # Evaluate each component
        bug_points, bug_details = self.evaluate_critical_bugs(submission, answer_key)
        feature_points, feature_details = self.evaluate_feature_enhancement(submission, answer_key)
        perf_points, perf_details = self.evaluate_performance_optimization(submission, answer_key)
        testing_details = self.evaluate_testing_completion(submission)
        
        # Calculate overall score
        total_points = bug_points + feature_points + perf_points
        overall_score = (total_points / self.max_points) * 100
        
        # Check for automatic failure conditions
        automatic_failures = self.check_automatic_failure_conditions(submission, bug_points)
        
        # If automatic failures exist, cap score at 59%
        if automatic_failures:
            overall_score = min(overall_score, 59.0)
        
        # Populate results
        results['overall_score'] = round(overall_score, 1)
        results['performance_level'] = self.determine_performance_level(overall_score)
        results['automatic_failures'] = automatic_failures
        
        results['detailed_scores'] = {
            'critical_bugs': bug_details,
            'feature_enhancement': feature_details,
            'performance_optimization': perf_details,
            'testing_completion': testing_details
        }
        
        results['summary'] = {
            'total_points_earned': total_points,
            'total_points_possible': self.max_points,
            'percentage_score': overall_score,
            'points_breakdown': {
                'critical_bugs': f"{bug_points}/{self.scoring_weights['critical_bugs']}",
                'feature_enhancement': f"{feature_points}/{self.scoring_weights['feature_enhancement']}",
                'performance_optimization': f"{perf_points}/{self.scoring_weights['performance_optimization']}"
            }
        }
        
        return results

    def save_results(self, results: Dict[str, Any], output_file: str = 'test_results.json'):
        """Save evaluation results to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(results, file, indent=2, ensure_ascii=False)
            print(f"Evaluation results saved to '{output_file}'")
        except Exception as e:
            print(f"Error saving results to '{output_file}': {e}")
            sys.exit(1)

def main():
    """Main function to handle command line arguments and run evaluation."""
    if len(sys.argv) != 3:
        print("Usage: python task_evaluation.py <submission_file> <answer_key_file>")
        print("Example: python task_evaluation.py test_submission.json answer_key.json")
        sys.exit(1)
    
    submission_file = sys.argv[1]
    answer_key_file = sys.argv[2]
    
    # Verify files exist
    for filename in [submission_file, answer_key_file]:
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' does not exist.")
            sys.exit(1)
    
    # Run evaluation
    evaluator = TaskEvaluator()
    results = evaluator.evaluate_submission(submission_file, answer_key_file)
    
    # Save results
    evaluator.save_results(results)
    
    # Print summary
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Overall Score: {results['overall_score']}%")
    print(f"Performance Level: {results['performance_level']['level']}")
    print(f"Points Earned: {results['summary']['total_points_earned']}/{results['summary']['total_points_possible']}")
    
    if results['automatic_failures']:
        print(f"\nAutomatic Failure Conditions:")
        for failure in results['automatic_failures']:
            print(f"  - {failure}")
    
    print(f"\nDetailed results saved to 'test_results.json'")

if __name__ == "__main__":
    main()