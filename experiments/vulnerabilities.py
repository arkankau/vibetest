"""Simple example of using vibetest programmatically."""

import argparse
from pathlib import Path
import jsonlines
import csv
import sys
import re
import zipfile
import json
import os
from typing import List, Dict, Any, Set
from collections import defaultdict
import random

from vibetest.baselines import analyze_repo_with_codeql

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
            
            all_test_cases.append(
                TestCase(
                    name=f"repo{repo_path.name}_baseline",
                    description="",
                    repo_path=repo_path,
                )
            )
    
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
        content = f.read()
        if "## " in content:
            # split by headings and remove heading
            properties = re.split(r"## .*\n", content)[1:]
        else:
            properties = content.split("- ")[1:]  # Split by headings
    properties = [p.strip() for p in properties if p.strip()]
    print(f"Loaded {len(properties)} properties from properties.md")

    # Check if instructions.md exists
    instructions = None
    instructions_path = f"./data/vuln/{dataset}/instructions.md"
    if Path(instructions_path).is_file():
        with open(instructions_path, mode="r") as f:
            instructions = f.read()

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
    selected_repos = selected_repos[5:10]

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
                if dataset == "bibifi":
                    cwe_str = str(pid)
                else:
                    match = re.search(r"CWE-(\d+)", prop)
                    if not match:
                        continue
                    cwe_str = match.group(1)
                desc = f"No violations of {cwe_str}."
                target = f"The prediction should describe the bug (as given in the reason and evidence) and should match the bug fixed by the following property description:\n{prop}\n\nNote that the prediction corresponds to the buggy version of the code which the property is describing. The evidence in the prediction should correspond to the same bug as described by the property. To determine if the evidence is correct, see if any of the methods mentioned in the property are mentioned in the evidence, or if the evidence describes the same issue as in the patch. If so, then the prediction should be treated as correct."
                all_test_cases.append(TestCase(
                    name=f"repo{repo_id}_vuln{cwe_str}" if dataset == "bibifi" else f"repo{repo_id}_cwe{cwe_str}",
                    description=prop,
                    extra_instructions=instructions,
                    repo_path=repo_path,
                    sandbox_path="/workdir"))

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


def run_codeql(dataset: str):
    """Run CodeQL baseline once per repository and score the 9 properties.

    Output format matches run_vibetest(): one JSONL entry per repo with a `tests`
    list and pass/fail counts.

    Requirement: if anything fails (e.g., CodeQL DB build), return all 9 PASS.
    """
    print("=" * 80)
    print("Starting Vulnerability Tests - CodeQL Baseline")
    print("=" * 80)

    bibifi_properties: list[str] = []
    cwe_properties: list[tuple[str, str]] = []

    if dataset == "bibifi":
        with open(f"./data/vuln/{dataset}/properties.md", mode="r") as f:
            properties_raw = f.read().split("- ")[1:]
        bibifi_properties = [p.strip() for p in properties_raw if p.strip()]
    else:
        # CWE-Bench: properties.md contains exactly the 4 CWE properties we want to evaluate.
        with open(f"./data/vuln/{dataset}/properties.md", mode="r") as f:
            properties_raw = f.read().split("- ")[1:]
        properties_texts = [p.strip() for p in properties_raw if p.strip()]

        for prop_text in properties_texts:
            m = re.search(r"CWE-(\d+)", prop_text)
            if not m:
                continue
            cwe_properties.append((m.group(1), prop_text))

        if not cwe_properties:
            # If parsing fails, fall back to the known 4 CWEs (empty descriptions if missing).
            cwe_properties = [
                ("22", properties_texts[0] if len(properties_texts) > 0 else ""),
                ("78", properties_texts[1] if len(properties_texts) > 1 else ""),
                ("79", properties_texts[2] if len(properties_texts) > 2 else ""),
                ("94", properties_texts[3] if len(properties_texts) > 3 else ""),
            ]

    # Load vuln metadata csv (same as run_vibetest)
    vuln_metadata = {}
    with open(f"./data/vuln/{dataset}/vulnerability_info.csv", mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo_name = row["project_slug"] if dataset == "bibifi" else row["vuln_id"] + "_" + row["project_slug"]
            cwe_id = row["cwe_id"]
            cwe_name = row["cwe_name"]
            info = row["patch"] if "patch" in row else row["notes"]
            vuln_metadata[repo_name] = (cwe_id, cwe_name, info)

    # Keep behavior consistent with run_vibetest: sample up to 50 repos.
    random.seed(42)
    selected_repos = random.sample(list(vuln_metadata.keys()), min(50, len(vuln_metadata)))

    repo_paths = []
    for repo in vuln_metadata:
        if repo not in selected_repos:
            continue
        repo_path = Path(f"./data/vuln/{dataset}/repos/{repo}")
        if not repo_path.is_dir():
            continue
        if dataset == "bibifi":
            analysis_path = repo_path / "build"
        else:
            analysis_path = repo_path

        repo_paths.append((repo, repo_path, analysis_path))
        print(f"Queueing repository: {repo}")

    output_path = Path("results") / f"vuln_results_{dataset}_codeql.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTotal repositories: {len(repo_paths)}")
    tests_per_repo = len(bibifi_properties) if dataset == "bibifi" else len(cwe_properties)
    print(f"Tests per repo: {tests_per_repo}")
    print(f"\n{'=' * 80}")
    print("Executing CodeQL analysis (sequential)...")
    print(f"{'=' * 80}\n")

    with jsonlines.open(str(output_path), mode="w") as writer:
        for repo, repo_root, analysis_path in repo_paths:
            print(f"\n{'=' * 80}")
            print(f"Repository: {repo}")
            print(f"{'=' * 80}")

            analysis = analyze_repo_with_codeql(analysis_path)
            ok = bool(analysis.get("ok"))

            tests = []
            passed_tests = 0
            total_tests = len(bibifi_properties) if dataset == "bibifi" else len(cwe_properties)
            property_failures = analysis.get("property_failures") or [[] for _ in range(total_tests)]
            all_findings = analysis.get("findings") or []

            # For cwe-bench, eval() expects repo_id to be the project_slug (no vuln_id prefix).
            repo_id_for_eval = repo if dataset == "bibifi" else "_".join(str(repo).split("_")[1:])

            # Precompute a quick CWE index for cwe-bench mapping.
            findings_by_cwe: dict[str, list[dict[str, Any]]] = {}
            if ok and dataset != "bibifi":
                for finding in all_findings:
                    cwes = set(str(c) for c in (finding.get("cwes") or []))
                    # Some SARIF emitters encode CWEs only in tags; recover those.
                    for tag in finding.get("tags") or []:
                        m = re.search(r"cwe-(\d+)", str(tag), flags=re.IGNORECASE)
                        if m:
                            cwes.add(m.group(1))
                    for cwe in cwes:
                        cwe = cwe.lstrip("0")  # Strip leading zeros
                        findings_by_cwe.setdefault(cwe, []).append(finding)

            if dataset == "bibifi":
                for pid, property_text in enumerate(bibifi_properties):
                    test_id = f"repo{repo_id_for_eval}_vuln{pid}"

                    # Requirement: if analysis failed, all properties pass.
                    if not ok:
                        passed = True
                        reason = "CodeQL analysis failed; defaulting to PASS for this property."
                        matched_findings_bibifi: list[dict[str, Any]] = []
                    else:
                        matched_findings_bibifi = (
                            property_failures[pid] if pid < len(property_failures) else []
                        )
                        passed = len(matched_findings_bibifi) == 0
                        reason = (
                            "No matching CodeQL findings."
                            if passed
                            else f"Matched {len(matched_findings_bibifi)} finding(s)."
                        )

                    if passed:
                        passed_tests += 1

                    tests.append(
                        {
                            "description": reason,
                            "passed": passed,
                            "evidence": [],
                            "execution_log": "",
                            "metadata": {
                                "test_id": test_id,
                                "property_index": pid,
                                "property_text": property_text,
                                "codeql_ok": ok,
                                "codeql_language": analysis.get("language"),
                                "codeql_suite": analysis.get("suite"),
                                "matched_findings": matched_findings_bibifi,
                            },
                        }
                    )

                    status = "✓ PASSED" if passed else "✗ FAILED"
                    print(f"\n{status}: property {pid}")
            else:
                print(findings_by_cwe.keys())
                for pid, cwe_prop in enumerate(cwe_properties):
                    cwe_num: str = cwe_prop[0]
                    property_text: str = cwe_prop[1]
                    test_id = f"repo{repo_id_for_eval}_cwe{cwe_num}"

                    # Requirement: if analysis failed, all properties pass.
                    if not ok:
                        passed = True
                        reason = "CodeQL analysis failed; defaulting to PASS for this property."
                        matched_findings_cwe: list[dict[str, Any]] = []
                    else:
                        matched_findings_cwe = findings_by_cwe.get(str(cwe_num), [])
                        passed = len(matched_findings_cwe) == 0
                        reason = (
                            "No matching CodeQL findings."
                            if passed
                            else f"Matched {len(matched_findings_cwe)} finding(s)."
                        )

                    if passed:
                        passed_tests += 1

                    tests.append(
                        {
                            "description": reason,
                            "passed": passed,
                            "evidence": [],
                            "execution_log": "",
                            "metadata": {
                                "test_id": test_id,
                                "property_index": pid,
                                "property_text": property_text,
                                "codeql_ok": ok,
                                "codeql_language": analysis.get("language"),
                                "codeql_suite": analysis.get("suite"),
                                "matched_findings": matched_findings_cwe,
                            },
                        }
                    )

                    status = "✓ PASSED" if passed else "✗ FAILED"
                    print(f"\n{status}: CWE-{cwe_num}")

            failed_tests = total_tests - passed_tests
            writer.write(
                {
                    "repo": str(analysis_path),
                    "repo_name": repo_id_for_eval,
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "tests": tests,
                    "metadata": {
                        "codeql_ok": ok,
                        "codeql_error": analysis.get("error"),
                        "codeql_language": analysis.get("language"),
                        "codeql_suite": analysis.get("suite"),
                        "num_findings": len(all_findings),
                        "findings": all_findings,
                        "repo_root": str(repo_root),
                    },
                }
            )

            if ok:
                print(f"\nCodeQL findings: {len(all_findings)}")
            else:
                print("\nCodeQL failed; emitted all-pass properties.")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: {output_path}")
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


def load_results_from_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load results from a JSONL file produced by run_codeql().

    Normalizes into the same structure expected by eval(): a list of repos,
    each containing a list of tests with 'id' and 'verdict'.
    """
    results: list[dict[str, Any]] = []
    with jsonlines.open(filepath, mode="r") as reader:
        for repo_entry in reader:
            repo_name = repo_entry.get("repo_name") or repo_entry.get("repo") or ""
            tests_in = repo_entry.get("tests") or []
            tests_out: list[dict[str, Any]] = []

            passed_tests = 0
            failed_tests = 0

            for idx, t in enumerate(tests_in):
                passed = bool(t.get("passed"))
                verdict = "PASS" if passed else "FAIL"
                if passed:
                    passed_tests += 1
                else:
                    failed_tests += 1

                meta = t.get("metadata") or {}
                test_id = meta.get("test_id") or meta.get("id") or f"{repo_name}_{idx}"

                matched_findings = meta.get("matched_findings") or []
                evidence = "None"
                if isinstance(matched_findings, list) and matched_findings:
                    # Keep this short; eval() does not inspect evidence.
                    evidence = f"Matched {len(matched_findings)} CodeQL finding(s)."

                tests_out.append(
                    {
                        "id": test_id,
                        "description": meta.get("property_text") or t.get("description") or "",
                        "passed": passed,
                        "verdict": verdict,
                        "reason": t.get("description") or "",
                        "evidence": evidence,
                        "metadata": meta,
                    }
                )

            total_tests = repo_entry.get("total_tests")
            if not isinstance(total_tests, int):
                total_tests = len(tests_out)

            results.append(
                {
                    "repo_name": repo_name,
                    "repo": repo_entry.get("repo") or repo_name,
                    "tests": tests_out,
                    "total_tests": total_tests,
                    "passed_tests": repo_entry.get("passed_tests")
                    if isinstance(repo_entry.get("passed_tests"), int)
                    else passed_tests,
                    "failed_tests": repo_entry.get("failed_tests")
                    if isinstance(repo_entry.get("failed_tests"), int)
                    else failed_tests,
                }
            )

    return results

def eval(dataset: str, eval_file: str):
    """Evaluate results from an existing .eval zip or CodeQL .jsonl output."""
    print("=" * 80)
    print("Evaluating Vulnerability Test Results")

    if eval_file.endswith(".eval"):
        results = load_results_from_eval(eval_file)
    elif eval_file.endswith(".jsonl"):
        results = load_results_from_jsonl(eval_file)
    else:
        # Best-effort fallback: try JSONL first, then .eval.
        try:
            results = load_results_from_jsonl(eval_file)
        except Exception:
            results = load_results_from_eval(eval_file)
    print(f"\nTotal repositories evaluated: {len(results)}")

    gt: Dict[str, Set[str]] = {}
    with open(f"./data/vuln/{dataset}/vulnerability_info.csv", mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo_name = row["project_slug"] if dataset == "bibifi" else row["project_slug"]
            cwe_id = row["cwe_id"] if dataset == "bibifi" else row["cwe_id"].split("CWE-")[1]
            if repo_name in gt:
                gt[repo_name].add(cwe_id)
            else:
                gt[repo_name] = set([cwe_id])

    def _parse_cwe_id(raw: Any) -> str:
        s = str(raw).strip()
        if not s:
            return ""
        m = re.search(r"(?:CWE-?)\s*(\d+)", s, flags=re.IGNORECASE)
        if m:
            s = m.group(1)
        # Normalize: strip leading zeros for stable matching (e.g. "079" -> "79").
        s = s.lstrip("0")
        return s or "0"

    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    Y_pred: list[bool] = []
    Y_gt: list[bool] = []
    Y_cwe: list[str] = []
    by_cwe: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for repo_result in results:
        for test in repo_result['tests']:
            id = test['id']
            if dataset == "bibifi":
                repo_id = id.split("_vuln")[0].replace("repo", "")
                cwe_id = _parse_cwe_id(str(int(id.split("_vuln")[1]) + 1))
            else:
                repo_id = id.split("_cwe")[0].replace("repo", "")
                cwe_id = _parse_cwe_id(id.split("_cwe")[1])
            verdict = test['verdict']
            pred = verdict == "FAIL"
            # if cwe_id == "7" or cwe_id == "5":
            #     continue
            # if cwe_id == "5" or cwe_id == "7" or cwe_id == "6" or cwe_id == "8":
            #     continue
            # if cwe_id != "79":
            #     continue
            actual = cwe_id in {_parse_cwe_id(x) for x in gt.get(repo_id, set())}
            Y_pred.append(pred)
            Y_gt.append(actual)
            Y_cwe.append(cwe_id)
            by_cwe[cwe_id].append((pred, actual))

    # Compute random baseline using per-CWE true positive rate.
    import random

    cwe_pos_rate: dict[str, float] = {}
    for cwe_id, pairs in by_cwe.items():
        if not pairs:
            cwe_pos_rate[cwe_id] = 0.0
            continue
        positives = sum(1 for _, a in pairs if a)
        cwe_pos_rate[cwe_id] = positives / len(pairs)

    random_preds: list[bool] = [
        (random.random() < cwe_pos_rate.get(cwe_id, 0.0)) for cwe_id in Y_cwe
    ]
    correct_random = sum(1 for x, y in zip(random_preds, Y_gt) if x == y)
    total = len(Y_pred)
    accuracy_random = correct_random / total if total > 0 else 0.0
    print(f"\nRandom Baseline Accuracy: {accuracy_random*100:.2f}% ({correct_random}/{total} correct predictions)")
    tp_r = sum(1 for x, y in zip(random_preds, Y_gt) if x and y)
    fp_r = sum(1 for x, y in zip(random_preds, Y_gt) if x and not y)
    fn_r = sum(1 for x, y in zip(random_preds, Y_gt) if (not x) and y)
    precision_random, recall_random, f1_score_random = _prf(tp_r, fp_r, fn_r)
    pu_score_random = (recall_random * recall_random) / ((sum(1 for y in random_preds if y) / len(random_preds))) if len(random_preds) else 0.0
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
    fn = sum(1 for x, y in zip(Y_pred, Y_gt) if (not x) and y)
    precision, recall, f1_score = _prf(tp, fp, fn)
    pu_score = (recall * recall) / ((sum(1 for y in Y_pred if y) / len(Y_pred))) if len(Y_pred) else 0.0
    print(f"Percent of Positive Predictions: {(sum([1 for y in Y_pred if y]) / len(Y_pred))*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1_score*100:.2f}%")
    print(f"PU Score: {pu_score*100:.2f}%")

    # Per-CWE metrics (macro over unique CWEs)
    if by_cwe:
        per_cwe_rows: list[tuple[str, float, float, float, int, int]] = []
        for cwe_id in sorted(by_cwe.keys(), key=lambda x: int(x) if str(x).isdigit() else 10**9):
            pairs = by_cwe[cwe_id]
            tp_c = sum(1 for p, a in pairs if p and a)
            fp_c = sum(1 for p, a in pairs if p and (not a))
            fn_c = sum(1 for p, a in pairs if (not p) and a)
            precision_c, recall_c, f1_c = _prf(tp_c, fp_c, fn_c)
            support = len(pairs)
            positives = sum(1 for _, a in pairs if a)
            per_cwe_rows.append((cwe_id, precision_c, recall_c, f1_c, support, positives))

        macro_precision = sum(r[1] for r in per_cwe_rows) / len(per_cwe_rows)
        macro_recall = sum(r[2] for r in per_cwe_rows) / len(per_cwe_rows)
        macro_f1 = sum(r[3] for r in per_cwe_rows) / len(per_cwe_rows)

        print(f"\nPer-CWE Precision/Recall/F1 (unique CWEs: {len(per_cwe_rows)})")
        for cwe_id, p, r, f1, support, positives in per_cwe_rows:
            print(
                f"  CWE-{cwe_id}: P={p*100:.2f}% R={r*100:.2f}% F1={f1*100:.2f}% "
                f"(support={support}, positives={positives})"
            )

        print("\nMacro Averages Over CWEs")
        print(f"Macro Precision: {macro_precision*100:.2f}%")
        print(f"Macro Recall: {macro_recall*100:.2f}%")
        print(f"Macro F1-Score: {macro_f1*100:.2f}%")

        # Random baseline per-CWE + macro, using per-CWE true positive rate.
        by_cwe_random: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        for cwe_id, rand_p, actual in zip(Y_cwe, random_preds, Y_gt):
            by_cwe_random[cwe_id].append((rand_p, actual))

        random_per_cwe_rows: list[tuple[str, float, float, float, float, int, int]] = []
        for cwe_id in sorted(by_cwe_random.keys(), key=lambda x: int(x) if str(x).isdigit() else 10**9):
            pairs = by_cwe_random[cwe_id]
            tp_c = sum(1 for p, a in pairs if p and a)
            fp_c = sum(1 for p, a in pairs if p and (not a))
            fn_c = sum(1 for p, a in pairs if (not p) and a)
            precision_c, recall_c, f1_c = _prf(tp_c, fp_c, fn_c)
            support = len(pairs)
            positives = sum(1 for _, a in pairs if a)
            pos_rate = cwe_pos_rate.get(cwe_id, 0.0)
            random_per_cwe_rows.append((cwe_id, precision_c, recall_c, f1_c, pos_rate, support, positives))

        macro_precision_r = sum(r[1] for r in random_per_cwe_rows) / len(random_per_cwe_rows)
        macro_recall_r = sum(r[2] for r in random_per_cwe_rows) / len(random_per_cwe_rows)
        macro_f1_r = sum(r[3] for r in random_per_cwe_rows) / len(random_per_cwe_rows)

        print(f"\nRandom Baseline Per-CWE Precision/Recall/F1 (unique CWEs: {len(random_per_cwe_rows)})")
        for cwe_id, p, r, f1, pos_rate, support, positives in random_per_cwe_rows:
            print(
                f"  CWE-{cwe_id}: P={p*100:.2f}% R={r*100:.2f}% F1={f1*100:.2f}% "
                f"(pos_rate={pos_rate*100:.2f}%, support={support}, positives={positives})"
            )

        print("\nRandom Baseline Macro Averages Over CWEs")
        print(f"Macro Precision: {macro_precision_r*100:.2f}%")
        print(f"Macro Recall: {macro_recall_r*100:.2f}%")
        print(f"Macro F1-Score: {macro_f1_r*100:.2f}%")

    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vibetest on Vulnerability repositories")
    parser.add_argument(
        "--method",
        type=str,
        choices=["vibetest", "baseline", "codeql"],
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
        help="Path to results file to evaluate (.eval from Inspect AI or .jsonl from codeql baseline)"
    )
    args = parser.parse_args()

    if args.eval:
        eval(args.dataset, args.eval)
    elif args.method == "baseline":
        run_baseline(args.dataset)
    elif args.method == "codeql":
        run_codeql(args.dataset)
    else:
        run_vibetest(args.dataset)