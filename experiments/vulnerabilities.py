"""Simple example of using vibetest programmatically."""

import argparse
from pathlib import Path
import jsonlines
import csv
import sys
import re
import zipfile
import json
from typing import List, Dict, Any
from collections import defaultdict
import random

from vibetest import TestCase, VibeTestAgent
from vibetest.agent import BaselineAgent

csv.field_size_limit(sys.maxsize)

def run_baseline(dataset: str):
    """Run baseline agent once per repository (no test cases).
    
    The baseline agent simply examines each repository for bugs without
    specific test criteria.
    """
    print("=" * 80)
    print("Starting Vulnerability Tests - Baseline Method")
    print("=" * 80)
    
    # Step 1: Collect all repositories and create one test case per repo
    all_test_cases = []
    repo_paths = []
    
    for repo_path in Path("./data/vuln/repos/").iterdir():
        if repo_path.is_dir():
            print(f"Queueing repository: {repo_path.name}")
            repo_paths.append(repo_path)
            
            all_test_cases.append(TestCase(description="", repo_path=repo_path))
    
    print(f"\nTotal repositories: {len(repo_paths)}")
    print(f"Total test cases: {len(all_test_cases)} (1 per repo)")
    print(f"\n{'=' * 80}")
    print("Executing baseline agent on all repositories...")
    print(f"{'=' * 80}\n")
    
    # Step 2: Execute baseline agent once per repository
    agent = BaselineAgent(static=True)
    all_results = agent.execute_tests(all_test_cases, sandbox="docker")
    
    print(f"\n{'=' * 80}")
    print("Processing Results")
    print(f"{'=' * 80}\n")
    
    # Step 3: Write results to file (one result per repo)
    with jsonlines.open(f"results/vuln_results_{agent.model_name.split('/')[1]}_baseline.jsonl", mode="w") as writer:
        for idx, (repo_path, result) in enumerate(zip(repo_paths, all_results)):
            print(f"\n{'=' * 80}")
            print(f"Repository: {repo_path.name}")
            print(f"{'=' * 80}")
            
            if result.passed:
                status = "✓ NO BUGS FOUND"
            else:
                status = "✗ BUGS FOUND"
            
            print(f"\n{status}")
            print(f"\nMessage: {result.message[:200]}...")
            
            # Write result for this repo
            writer.write({
                "repo": str(repo_path),
                "repo_name": repo_path.name,
                "bugs_found": not result.passed,
                "message": result.message,
                "execution_log": result.execution_log,
                "metadata": result.metadata,
            })
    
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: results/vuln_results_{agent.model_name.split('/')[1]}_baseline.jsonl")
    print(f"{'=' * 80}")


def run_vibetest(dataset: str):
    """Run vibetest with specific test cases across all repositories."""
    print("=" * 80)
    print("Starting Vulnerability Tests - VibeTest Method")
    print("=" * 80)

    with open(f"./data/vuln/{dataset}/properties.md", mode="r") as f:
        properties = f.read().split("- ")[1:]  # Split by headings
    properties = [p.strip() for p in properties if p.strip()]
    print(f"Loaded {len(properties)} properties from properties.md")

    # load vuln metadata csv
    vuln_metadata = {}
    with open(f"./data/vuln/{dataset}/vulnerability_info.csv", mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo_name = row["project_slug"] if dataset == "bibifi" else row["vuln_id"] + "_" + row["project_slug"]
            cwe_id = row["cwe_id"]
            cwe_name = row["cwe_name"]
            info = row["patch"] if "patch" in row else row["notes"]
            vuln_metadata[repo_name] = (cwe_id, cwe_name, info)

    # Step 1: Collect all repositories and create test cases
    all_test_cases = []
    repo_paths = []
    tests_per_repo = len(properties)

    # select 50 random repos from vuln_metadata for testing
    random.seed(42)
    selected_repos = random.sample(list(vuln_metadata.keys()), min(50, len(vuln_metadata)))

    i = 0
    for repo, meta in vuln_metadata.items():
        if repo not in selected_repos:
            continue

        i += 1
        repo_path = Path(f"./data/vuln/{dataset}/repos/{repo}")
        if repo_path.is_dir():
            # if "_".join(str(repo_path.name).split("_")[1:]) not in vuln_metadata:
            #     continue

            print(f"Queueing repository: {repo_path.name}")
            if dataset == "bibifi":
                repo_id = repo_path.name
                repo_path = repo_path / "build"
            else:
                repo_id = "_".join(str(repo_path.name).split("_")[1:])
            repo_paths.append(repo_path)
            
            # Create test cases for this repo
            for pid, prop in enumerate(properties):
                # get the CWE ID from the property line
                cwe_str = str(pid) if dataset == "bibifi" else re.search(r"CWE (\d+)", prop).group(1)
                desc = f"No violations of {cwe_str}."
                target = f"The prediction should describe the bug (as given in the reason and evidence) and should match the bug fixed by the following property description:\n{prop}\n\nNote that the prediction corresponds to the buggy version of the code which the property is describing. The evidence in the prediction should correspond to the same bug as described by the property. One way to determine if the evidence is correct is to see if any of the methods mentioned in the property are mentioned in the evidence. If so, then the prediction should be treated as correct."
                all_test_cases.append(TestCase(name=f"repo{repo_id}_vuln{cwe_str}" if dataset == "bibifi" else f"repo{repo_id}_cwe{cwe_str}", description=prop, repo_path=repo_path, sandbox_path="/workdir"))

    print(f"\nTotal repositories: {len(repo_paths)}")
    print(f"Total test cases: {len(all_test_cases)} ({tests_per_repo} tests × {len(repo_paths)} repos)")
    print(f"\n{'=' * 80}")
    print("Executing all tests in parallel...")
    print(f"{'=' * 80}\n")
    
    # Step 2: Execute ALL tests in parallel across all repositories
    agent = VibeTestAgent(static=True)
    all_results = agent.execute_tests(all_test_cases, sandbox="docker")
    
    print(f"\n{'=' * 80}")
    print("Processing Results")
    print(f"{'=' * 80}\n")
    
    # Step 3: Group results by repository and write to file
    with jsonlines.open(f"results/vuln_results_{agent.model_name.split('/')[1]}.jsonl", mode="w") as writer:
        # Group results by repository
        for repo_idx, repo_path in enumerate(repo_paths):
            print(f"\n{'=' * 80}")
            print(f"Repository: {repo_path.name}")
            print(f"{'=' * 80}")
            
            # Extract results for this repo (tests_per_repo consecutive results)
            start_idx = repo_idx * tests_per_repo
            end_idx = start_idx + tests_per_repo
            repo_results = all_results[start_idx:end_idx]
            
            # Collect test results for this repo
            test_results = []
            repo_passed = 0
            repo_total = len(repo_results)
            
            for r in repo_results:
                if r.passed:
                    repo_passed += 1
                    status = "✓ PASSED"
                else:
                    status = "✗ FAILED"
                
                print(f"\n{status}: {r.test_case.description}")
                if r.evidence:
                    print(f"  Evidence items: {len(r.evidence)}")
                
                test_results.append({
                    "description": r.message,
                    "passed": r.passed,
                    "evidence": [e.model_dump() for e in r.evidence],
                    "execution_log": r.execution_log,
                    "metadata": r.metadata,
                })
            
            # Write all results for this repo as a single entry
            writer.write({
                "repo": str(repo_path),
                "repo_name": repo_path.name,
                "total_tests": repo_total,
                "passed_tests": repo_passed,
                "failed_tests": repo_total - repo_passed,
                "tests": test_results,
            })
            
            print(f"\nRepo Summary: {repo_passed}/{repo_total} tests passed")
    
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: results/vuln_results_{agent.model_name.split('/')[1]}.jsonl")
    print(f"{'=' * 80}")


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
                'id': sample_id,
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
            
            results_by_repo[sample_id]['tests'].append(test)
            results_by_repo[sample_id]['total_tests'] += 1
            if passed:
                results_by_repo[sample_id]['passed_tests'] += 1
            else:
                results_by_repo[sample_id]['failed_tests'] += 1
    
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

def eval(dataset: str, eval_file: str):
    """Evaluate results from an existing .eval file."""
    print("=" * 80)
    print("Evaluating Vulnerability Test Results")

    results = load_results_from_eval(eval_file)
    print(f"\nTotal repositories evaluated: {len(results)}")

    gt = {}
    with open(f"./data/vuln/{dataset}/vulnerability_info.csv", mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo_name = row["project_slug"] if dataset == "bibifi" else row["project_slug"]
            cwe_id = row["cwe_id"] if dataset == "bibifi" else row["cwe_id"].split("CWE-")[1]
            if repo_name in gt:
                gt[repo_name].add(cwe_id)
            else:
                gt[repo_name] = set([cwe_id])

    print("GT:", gt)
    Y_pred = []
    Y_gt = []
    for repo_result in results:
        for test in repo_result['tests']:
            id = test['id']
            if dataset == "bibifi":
                repo_id = id.split("_vuln")[0].replace("repo", "")
                cwe_id = str(int(id.split("_vuln")[1]) + 1)
            else:
                # repo_id = "_".join(id.split("_cwe")[0].replace("repo", "").split("_")[1:])
                repo_id = id.split("_cwe")[0].replace("repo", "")
                cwe_id = id.split("_cwe")[1]
            verdict = test['verdict']
            pred = verdict == "FAIL"
            # if cwe_id == "7" or cwe_id == "5":
            #     continue
            # if cwe_id != str(i):
            #     continue
            actual = cwe_id in gt.get(repo_id, set())
            Y_pred.append(pred)
            Y_gt.append(actual)

    # Compute random accuracy
    import random
    # predict randomly based on the distribution of positives in Y_gt
    pos_rate = 0.5111#sum(1 for y in Y_gt if y) / len(Y_gt) if len(Y_gt) > 0 else 0.0
    random_preds = [random.random() < pos_rate for _ in Y_gt]
    correct_random = sum(1 for x, y in zip(random_preds, Y_gt) if x == y)
    total = len(Y_pred)
    accuracy_random = correct_random / total if total > 0 else 0.0
    print(f"\nRandom Baseline Accuracy: {accuracy_random*100:.2f}% ({correct_random}/{total} correct predictions)")
    tp = sum(1 for x, y in zip(random_preds, Y_gt) if x and y)
    fp = sum(1 for x, y in zip(random_preds, Y_gt) if x and not y)
    fn = sum(1 for x, y in zip(random_preds, Y_gt) if not x and y)
    precision_random = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_random = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score_random = 2 * (precision_random * recall_random) / (precision_random + recall_random) if (precision_random + recall_random) > 0 else 0.0
    pu_score_random = (recall_random * recall_random) / ((sum([1 for y in random_preds if y]) / len(random_preds)))
    print(f"Random Precision: {precision_random*100:.2f}%")
    print(f"Random Recall: {recall_random*100:.2f}%")
    print(f"Random F1-Score: {f1_score_random*100:.2f}%")
    print(f"Random PU Score: {pu_score_random*100:.2f}%")
    print(f"{'=' * 80}")

    # Compute accuracy
    correct = sum(1 for x, y in zip(Y_pred, Y_gt) if x == y)
    total = len(Y_pred)
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nOverall Accuracy: {accuracy*100:.2f}% ({correct}/{total} correct predictions)")

    # Compute precision, recall, F1-score
    tp = sum(1 for x, y in zip(Y_pred, Y_gt) if x and y)
    fp = sum(1 for x, y in zip(Y_pred, Y_gt) if x and not y)
    fn = sum(1 for x, y in zip(Y_pred, Y_gt) if not x and y)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    pu_score = (recall * recall) / ((sum([1 for y in Y_pred if y]) / len(Y_pred)))
    print(f"Percent of Positive Predictions: {(sum([1 for y in Y_pred if y]) / len(Y_pred))*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1_score*100:.2f}%")
    print(f"PU Score: {pu_score*100:.2f}%")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vibetest on Vulnerability repositories")
    parser.add_argument(
        "--method",
        type=str,
        choices=["vibetest", "baseline"],
        default="vibetest",
        help="Method to use: 'vibetest' for VibeTestAgent or 'baseline' for BaselineAgent"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["bibifi", "cwe-bench"],
        default="bibifi",
        help="Dataset to use: 'bibifi' or 'cwe-bench'"
    )
    parser.add_argument(
        "--eval",
        type=str,
        help="Path to .eval file to evaluate existing results"
    )
    args = parser.parse_args()

    if args.eval:
        eval(args.dataset, args.eval)
    elif args.method == "baseline":
        run_baseline(args.dataset)
    else:
        run_vibetest(args.dataset)