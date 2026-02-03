"""Analyze results from .eval log files."""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter
import csv
import zipfile
import re


def extract_verdict_from_sample(sample_data: Dict[str, Any]) -> tuple[str, str, str]:
    """Extract the verdict, reason, and evidence from a sample's messages.
    
    Looks through the messages for content containing the verdict.
    """
    messages = sample_data.get('messages', [])
    
    # Look through messages from the end (verdict is usually at the end)
    for message in reversed(messages):
        content = message.get('content', '')
        
        # Content can be a string or a list of content items
        text_to_search = ''
        if isinstance(content, str):
            text_to_search = content
        elif isinstance(content, list):
            # Concatenate all text items
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_to_search += item.get('text', '') + '\n'
        
        # Look for VERDICT: PASS or VERDICT: FAIL
        if 'VERDICT' in text_to_search and 'EXPLANATION' not in text_to_search:
            match = re.search(r'VERDICT\s*:\s*(PASS|FAIL|INCONCLUSIVE|NOT APPLICABLE)', text_to_search, re.IGNORECASE)
            verdict = "UNKNOWN"
            if match:
                verdict = match.group(1).upper()

            match = re.search(r'REASON\s*:\s*(.*)EVIDENCE', text_to_search, re.DOTALL)
            reason = "None"
            if match:
                reason = match.group(1).strip()

            match = re.search(r'EVIDENCE\s*:\s*(.*)$', text_to_search, re.DOTALL)
            evidence = "None"
            if match:
                evidence = match.group(1).strip()
            
            return verdict, reason, evidence
        elif 'VERDICT' in text_to_search and 'EXPLANATION' in text_to_search:
            match = re.search(r'VERDICT\s*:\s*(PASS|FAIL|INCONCLUSIVE)', text_to_search, re.IGNORECASE)
            verdict = "UNKNOWN"
            if match:
                verdict = match.group(1).upper()

            match = re.search(r'EXPLANATION\s*:\s*(.*)$', text_to_search, re.DOTALL)
            reason = "None"
            if match:
                reason = match.group(1).strip()
            
            return verdict, reason, "None"
    
    return 'UNKNOWN', "None", "None"


def load_results_from_eval(filepath: str) -> List[Dict[str, Any]]:
    """Load results from .eval file (which is a zip archive).
    
    Extracts individual sample JSON files from the eval archive and parses them
    to extract test verdicts. Returns a list of test results grouped by repository.
    """
    results_by_repo: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        'tests': [],
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
    })
    
    with zipfile.ZipFile(filepath, 'r') as zip_file:
        # Get list of all sample files
        sample_files = [f for f in zip_file.namelist() if f.startswith('samples/') and f.endswith('.json')]
        # sample_files.sort()

        # Process each sample file
        for sample_file in sample_files:
            with zip_file.open(sample_file) as f:
                sample_data = json.load(f)
            
            # Extract repo name and test info from the sample
            sample_id = sample_data.get('id', '')
            # Extract repo name (everything before the final underscore and number)
            repo_parts = sample_id.rsplit('_', 1)
            repo_name = repo_parts[0] if repo_parts else 'Unknown'
            
            # Extract test description from input
            input_text = sample_data.get('input', '')
            test_description = ''
            if 'Test:' in input_text:
                test_parts = input_text.split('Test:', 1)[1].split('Repository:', 1)
                test_description = test_parts[0].strip() if test_parts else ''
            
            # Extract verdict from the sample's messages
            verdict, reason, evidence = extract_verdict_from_sample(sample_data)
            passed = (verdict == 'PASS')
            
            # Build test result object
            test = {
                'description': test_description,
                'passed': passed,
                'verdict': verdict,
                'reason': reason,
                'evidence': evidence,
                'metadata': {
                    'test_description': test_description,
                    'sample_id': sample_id,
                },
                'model_usage': sample_data.get('model_usage', {}),
                'total_time': sample_data.get('total_time', 0),
            }
            
            results_by_repo[repo_name]['tests'].append(test)
            results_by_repo[repo_name]['total_tests'] += 1
            if passed:
                results_by_repo[repo_name]['passed_tests'] += 1
            else:
                results_by_repo[repo_name]['failed_tests'] += 1
    
    # Convert to list format
    results = []
    for repo_name, repo_data in results_by_repo.items():
        results.append({
            'repo_name': repo_name,
            'repo': repo_name,
            'tests': repo_data['tests'],
            'total_tests': repo_data['total_tests'],
            'passed_tests': repo_data['passed_tests'],
            'failed_tests': repo_data['failed_tests'],
        })
    
    return results


def _normalize_test_description(test: Dict[str, Any]) -> str:
    metadata = test.get("metadata", {}) or {}
    return (
        metadata.get("test_description")
        or metadata.get("property_text")
        or metadata.get("property_id")
        or test.get("description")
        or "No specific test"
    )


def _coerce_reason(test: Dict[str, Any]) -> str:
    return test.get("description") or test.get("reason") or "None"


def _coerce_evidence(test: Dict[str, Any]) -> str:
    metadata = test.get("metadata", {}) or {}
    evidence_text = metadata.get("evidence_text")
    if evidence_text:
        return evidence_text
    evidence = test.get("evidence")
    if isinstance(evidence, list) and evidence:
        return "; ".join(str(e) for e in evidence)
    if isinstance(evidence, str) and evidence:
        return evidence
    return "None"


def load_results_from_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load results from a JSONL file (one repo result per line)."""
    results: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tests = obj.get("tests", [])
            for test in tests:
                metadata = test.get("metadata", {}) or {}
                metadata.setdefault("test_description", _normalize_test_description(test))
                verdict = metadata.get("verdict")
                if not verdict:
                    verdict = "PASS" if test.get("passed") else "FAIL"
                test["verdict"] = verdict
                test["reason"] = _coerce_reason(test)
                test["evidence"] = _coerce_evidence(test)
                test["metadata"] = metadata
            if "total_tests" not in obj:
                obj["total_tests"] = len(tests)
            if "passed_tests" not in obj:
                obj["passed_tests"] = sum(1 for t in tests if t.get("passed"))
            if "failed_tests" not in obj:
                obj["failed_tests"] = obj["total_tests"] - obj["passed_tests"]
            results.append(obj)
    return results


def calculate_basic_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate basic statistics across all repositories."""
    total_repos = len(results)
    total_tests_run = sum(r['total_tests'] for r in results)
    total_passed = sum(r['passed_tests'] for r in results)
    total_failed = sum(r['failed_tests'] for r in results)
    
    avg_pass_rate = (total_passed / total_tests_run * 100) if total_tests_run > 0 else 0
    
    repos_with_all_tests_passed = sum(1 for r in results if r['passed_tests'] == r['total_tests'])
    repos_with_some_failures = sum(1 for r in results if r['failed_tests'] > 0)
    
    return {
        'total_repos': total_repos,
        'total_tests_run': total_tests_run,
        'total_passed': total_passed,
        'total_failed': total_failed,
        'avg_pass_rate': avg_pass_rate,
        'repos_with_all_tests_passed': repos_with_all_tests_passed,
        'repos_with_some_failures': repos_with_some_failures,
    }


def analyze_test_performance(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Analyze performance of each test across all repositories."""
    test_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'passed': 0, 'failed': 0, 'total': 0})
    
    for repo_result in results:
        for test in repo_result.get('tests', []):
            # Extract the test description (first line or whole description)
            test_desc = test.get('description', '')
            # Try to extract the test type from metadata or description
            test_type = test.get('metadata', {}).get('test_description', '')
            
            # Use the extracted test_type or fall back to a short description
            if not test_type:
                test_type = test_desc[:80] if test_desc else 'Unknown test'
            
            passed = test.get('passed', False)
            test_stats[test_type]['total'] += 1
            if passed:
                test_stats[test_type]['passed'] += 1
            else:
                test_stats[test_type]['failed'] += 1
    
    # Calculate pass rates
    for test_type, stats in test_stats.items():
        stats['pass_rate'] = float((stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0)
    
    return dict(test_stats)


def analyze_repo_performance(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze performance of each repository."""
    repo_stats = []
    
    for repo_result in results:
        repo_name = repo_result.get('repo_name', repo_result.get('repo', 'Unknown'))
        total_tests = repo_result.get('total_tests', 0)
        passed_tests = repo_result.get('passed_tests', 0)
        failed_tests = repo_result.get('failed_tests', 0)
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        repo_stats.append({
            'repo_name': repo_name,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate': pass_rate,
        })
    
    return sorted(repo_stats, key=lambda x: x['pass_rate'], reverse=True)


def find_common_failures(results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Find which tests fail most commonly and which repos fail them."""
    test_failures = defaultdict(list)
    
    for repo_result in results:
        repo_name = repo_result.get('repo_name', repo_result.get('repo', 'Unknown'))
        for test in repo_result.get('tests', []):
            if not test.get('passed', False):
                test_type = test.get('metadata', {}).get('test_description', '')
                if not test_type:
                    # Try to extract from description
                    desc = test.get('description', '')
                    test_type = desc[:80] if desc else 'Unknown test'
                test_failures[test_type].append(repo_name)
    
    return dict(test_failures)


def print_analysis(results: List[Dict[str, Any]]):
    """Print comprehensive analysis of results."""
    print("=" * 80)
    print("EVAL RESULTS ANALYSIS")
    print("=" * 80)
    print()
    
    # Basic statistics
    print("BASIC STATISTICS")
    print("-" * 80)
    basic_stats = calculate_basic_stats(results)
    print(f"Total repositories tested: {basic_stats['total_repos']}")
    print(f"Total tests run: {basic_stats['total_tests_run']}")
    print(f"Total tests passed: {basic_stats['total_passed']}")
    print(f"Total tests failed: {basic_stats['total_failed']}")
    print(f"Overall pass rate: {basic_stats['avg_pass_rate']:.2f}%")
    print(f"Repos with all tests passed: {basic_stats['repos_with_all_tests_passed']}")
    print(f"Repos with some failures: {basic_stats['repos_with_some_failures']}")
    print()
    
    # Test performance
    print("TEST PERFORMANCE ACROSS REPOSITORIES")
    print("-" * 80)
    test_stats = analyze_test_performance(results)
    for test_type, stats in sorted(test_stats.items(), key=lambda x: x[1]['pass_rate'], reverse=True):
        print(f"\n{test_type}")
        print(f"  Pass rate: {stats['pass_rate']:.2f}% ({stats['passed']}/{stats['total']})")
        print(f"  Failed: {stats['failed']}")
    print()
    
    # Repository performance
    print("REPOSITORY PERFORMANCE")
    print("-" * 80)
    repo_stats = analyze_repo_performance(results)
    for repo in repo_stats:
        status = "✓" if repo['failed_tests'] == 0 else "✗"
        print(f"{status} {repo['repo_name']:<30} Pass rate: {repo['pass_rate']:>6.2f}% ({repo['passed_tests']}/{repo['total_tests']})")
    print()
    
    # Common failures
    print("COMMON TEST FAILURES")
    print("-" * 80)
    test_failures = find_common_failures(results)
    if test_failures:
        for test_type, repos in sorted(test_failures.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n{test_type}")
            print(f"  Failed in {len(repos)} repo(s): {', '.join(repos)}")
    else:
        print("No test failures found!")
    print()
    
    # Summary insights
    print("SUMMARY INSIGHTS")
    print("-" * 80)
    if basic_stats['repos_with_all_tests_passed'] == basic_stats['total_repos']:
        print("🎉 All repositories passed all tests!")
    elif basic_stats['avg_pass_rate'] >= 90:
        print("✨ Overall performance is excellent (>90% pass rate)")
    elif basic_stats['avg_pass_rate'] >= 75:
        print("👍 Overall performance is good (75-90% pass rate)")
    elif basic_stats['avg_pass_rate'] >= 50:
        print("⚠️  Overall performance needs improvement (50-75% pass rate)")
    else:
        print("❌ Overall performance is poor (<50% pass rate)")
    
    # Find the most reliable test
    if test_stats:
        most_reliable = max(test_stats.items(), key=lambda x: x[1]['pass_rate'])
        print(f"\nMost reliable test: {most_reliable[0]}")
        print(f"  Pass rate: {most_reliable[1]['pass_rate']:.2f}%")
        
        # Find the most challenging test
        least_reliable = min(test_stats.items(), key=lambda x: x[1]['pass_rate'])
        print(f"\nMost challenging test: {least_reliable[0]}")
        print(f"  Pass rate: {least_reliable[1]['pass_rate']:.2f}%")
    
    print()
    print("=" * 80)


def create_test_results_csv(results: List[Dict[str, Any]], output_file: Path, include_evidence=False):
    """Create a CSV file with repo names and pass/fail results for each test."""
    # Collect all test descriptions to use as column headers
    all_test_descriptions = set()
    
    # First pass: collect all unique test descriptions
    for repo_result in results:
        for test in repo_result.get('tests', []):
            test_desc = test.get('metadata', {}).get('test_description', '')
            if test_desc:
                all_test_descriptions.add(test_desc)
            else:
                all_test_descriptions.add("No specific test")
    
    # Sort test descriptions for consistent column ordering
    test_descriptions = sorted(all_test_descriptions)
    
    # Create CSV
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['repo_name'] + test_descriptions
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for repo_result in results:
            repo_name = repo_result.get('repo_name', repo_result.get('repo', 'Unknown'))
            
            # Build a mapping of test_description -> pass/fail
            test_results = {}
            for test in repo_result.get('tests', []):
                test_desc = test.get('metadata', {}).get('test_description', '')
                if not test_desc:
                    test_desc = "No specific test"
                test_results[test_desc] = test.get('verdict', 'UNKNOWN')
                if include_evidence:
                    test_results[test_desc] += f"\nReason: {test['reason']}\nEvidence: {test['evidence']}"
            
            # Create row with repo name and all test results
            row = {'repo_name': repo_name}
            for test_desc in test_descriptions:
                row[test_desc] = test_results.get(test_desc, 'N/A')
            
            writer.writerow(row)
    
    print(f"Test results CSV created at {output_file}")


def main():
    """Main analysis function."""
    # Default to the provided example file, but allow command line argument
    import sys
    
    if len(sys.argv) > 1:
        eval_file = Path(sys.argv[1])
    else:
        eval_file = Path(__file__).parent.parent / "logs" / "2025-10-10T00-01-51-04-00_task_UpjzSi2qoNeCpqLZQPwamf.eval"
    
    if not eval_file.exists():
        print(f"Error: Eval file not found at {eval_file}")
        print(f"Usage: python {sys.argv[0]} [path/to/file.eval]")
        return
    
    print(f"Loading results from {eval_file}...")
    if eval_file.suffix == ".jsonl":
        results = load_results_from_jsonl(str(eval_file))
    else:
        results = load_results_from_eval(str(eval_file))
    
    if not results:
        print("No results found in file!")
        return
    
    print(f"Loaded {len(results)} repository results")
    print()
    
    print_analysis(results)
    
    # Create CSV with test results
    csv_output_file = eval_file.with_suffix('.csv')
    create_test_results_csv(results, csv_output_file)

    csv_output_reason_file = eval_file.with_name(eval_file.stem + "_reason.csv")
    create_test_results_csv(results, csv_output_reason_file, include_evidence=True)


if __name__ == "__main__":
    main()
