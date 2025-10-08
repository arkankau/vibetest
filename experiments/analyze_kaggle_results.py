"""Analyze results from kaggle_results.jsonl."""

import json
from pathlib import Path
from typing import Dict, List, Any
import jsonlines
from collections import defaultdict, Counter


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
            
            if not test_type and test_desc:
                # If no metadata, try to extract from description
                lines = test_desc.split('\n')
                for line in lines:
                    if line.startswith('REASON:'):
                        # Try to find the test description in nearby lines
                        continue
                    if 'Training loss' in line or 'loss generally decreases' in test_desc:
                        test_type = 'Training loss decreases'
                        break
                    elif 'overfit' in line.lower():
                        test_type = 'Can overfit tiny batch'
                        break
                    elif 'randomiz' in line.lower() and 'label' in line.lower():
                        test_type = 'Randomized labels reduce accuracy'
                        break
                    elif 'leakage' in line.lower() or 'test to train' in line.lower():
                        test_type = 'No test/train leakage'
                        break
                    elif 'deterministic' in line.lower():
                        test_type = 'Deterministic test accuracy'
                        break
                    elif 'frozen layer' in line.lower() or 'parameters are updated' in line.lower():
                        test_type = 'All parameters updated'
                        break
                    elif 'NaN' in line or 'Inf' in line:
                        test_type = 'No NaN/Inf in parameters'
                        break
                    elif 'baseline' in line.lower() and 'outperform' in line.lower():
                        test_type = 'Outperforms baseline'
                        break
            
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


def main():
    """Main analysis function."""
    results_file = Path(__file__).parent.parent / "results" / "kaggle_results.jsonl"
    
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


if __name__ == "__main__":
    main()
