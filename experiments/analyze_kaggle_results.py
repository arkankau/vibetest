"""Analyze results from kaggle_results.jsonl."""

import json
from pathlib import Path
from typing import Dict, List, Any
import jsonlines
from collections import defaultdict, Counter
import csv


def load_results(filepath: str) -> List[Dict[str, Any]]:
    """Load results from JSONL file."""
    results = []
    with jsonlines.open(filepath) as reader:
        for obj in reader:
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
    print("KAGGLE RESULTS ANALYSIS")
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


def create_test_results_csv(results: List[Dict[str, Any]], output_file: Path):
    """Create a CSV file with repo names and pass/fail results for each test."""
    # Collect all test descriptions to use as column headers
    all_test_descriptions = set()
    
    # First pass: collect all unique test descriptions
    for repo_result in results:
        for test in repo_result.get('tests', []):
            test_desc = test.get('metadata', {}).get('test_description', '')
            if test_desc:
                all_test_descriptions.add(test_desc)
    
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
                if test_desc:
                    test_results[test_desc] = 'PASS' if test.get('passed', False) else 'FAIL'
            
            # Create row with repo name and all test results
            row = {'repo_name': repo_name}
            for test_desc in test_descriptions:
                row[test_desc] = test_results.get(test_desc, 'N/A')
            
            writer.writerow(row)
    
    print(f"Test results CSV created at {output_file}")


def main():
    """Main analysis function."""
    results_file = Path(__file__).parent.parent / "results" / "kaggle_results_gpt-5-mini.jsonl"
    
    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        return
    
    print(f"Loading results from {results_file}...")
    results = load_results(str(results_file))
    
    if not results:
        print("No results found in file!")
        return
    
    print(f"Loaded {len(results)} repository results")
    print()
    
    print_analysis(results)
    
    # Create CSV with test results
    csv_output_file = results_file.with_suffix('.csv')
    create_test_results_csv(results, csv_output_file)


if __name__ == "__main__":
    main()
