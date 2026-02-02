"""CodeQL baseline for vulnerability analysis.

This module provides a lightweight baseline that:
1) Detects the dominant programming language of a repo.
2) Builds a CodeQL database (best-effort autobuild).
3) Runs the default CodeQL security queries and parses SARIF results.
4) Maps findings to the 9 BiBiFi properties (pass/fail per property).

If anything fails (missing CodeQL binary, DB build failure, analyze failure), the
baseline returns *all 9 properties as PASS* as requested.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CODEQL_BINARY_ENV = "VIBETEST_CODEQL_BINARY"

# NOTE: This baseline currently groups findings for BiBiFi into 8 buckets
# (matching the mapping function below). Keep this consistent.
BIBIFI_PROPERTY_COUNT = 8


@dataclass(frozen=True)
class CodeQLFinding:
	rule_id: str
	rule_name: str | None
	message: str
	severity: str | None
	cwes: tuple[str, ...]
	tags: tuple[str, ...]
	file_path: str | None
	start_line: int | None
	start_column: int | None


def _run(
	args: list[str],
	*,
	cwd: Path | None = None,
	timeout_s: int | None = None,
) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		args,
		cwd=str(cwd) if cwd else None,
		text=True,
		capture_output=True,
		check=True,
		timeout=timeout_s,
	)


def _codeql_binary() -> str:
	# Requirement: env var should point to the codeql binary.
	return os.environ.get(CODEQL_BINARY_ENV, "")


def _iter_files(repo_path: Path) -> Iterable[Path]:
	for root, dirs, files in os.walk(repo_path):
		# Skip common heavy/irrelevant directories.
		base = os.path.basename(root)
		if base in {".git", ".hg", ".svn", "node_modules", "dist", ".venv", "venv"}:
			dirs[:] = []
			continue
		for name in files:
			yield Path(root) / name


def detect_codeql_language(repo_path: Path) -> str | None:
	"""Best-effort language detection for CodeQL database creation.

	Returns CodeQL language name such as: python, javascript, java, cpp, csharp, go.
	"""
	ext_counts: dict[str, int] = {}
	for path in _iter_files(repo_path):
		suffix = path.suffix.lower()
		if not suffix:
			continue
		ext_counts[suffix] = ext_counts.get(suffix, 0) + 1

	# Heuristic: pick the language with the highest weighted count.
	# (Weights help avoid header files dominating C/C++ and similar edge cases.)
	weights = {
		".py": ("python", 5),
		".js": ("javascript", 3),
		".jsx": ("javascript", 3),
		".ts": ("javascript", 3),
		".tsx": ("javascript", 3),
		".java": ("java", 4),
		".kt": ("java", 4),
		".kts": ("java", 4),
		".c": ("cpp", 3),
		".cc": ("cpp", 3),
		".cpp": ("cpp", 3),
		".cxx": ("cpp", 3),
		".h": ("cpp", 1),
		".hpp": ("cpp", 1),
		".cs": ("csharp", 4),
		".go": ("go", 4),
		".rb": ("ruby", 3),
	}

	scores: dict[str, int] = {}
	for ext, count in ext_counts.items():
		if ext not in weights:
			continue
		lang, w = weights[ext]
		scores[lang] = scores.get(lang, 0) + (count * w)

	if not scores:
		return None

	return max(scores.items(), key=lambda kv: kv[1])[0]


def _build_mode_for_language(language: str) -> str:
	"""Return a CodeQL build mode compatible with the language.

	Python (and other interpreted languages) do not support autobuild.
	"""
	# Compiled/compiled-ish languages where CodeQL can attempt to build.
	if language in {"cpp", "csharp", "go"}:
		return "autobuild"
	# Interpreted languages: do not build.
	return "none"


def _suite_for_language(language: str) -> str:
	"""Return a query suite spec that does not depend on local filesystem layout.

	We use the query pack + suite path format:
	  <pack>:<path-within-pack>

	Example:
	  codeql/python-queries:codeql-suites/python-security-and-quality.qls
	"""
	return f"codeql/{language}-queries:codeql-suites/{language}-security-and-quality.qls"


def _parse_sarif(sarif: dict[str, Any]) -> list[CodeQLFinding]:
	runs = sarif.get("runs") or []
	if not runs:
		return []
	run0 = runs[0]
	tool = run0.get("tool", {})
	driver = tool.get("driver", {})
	rules = driver.get("rules") or []

	rule_meta: dict[str, dict[str, Any]] = {}
	for rule in rules:
		rid = rule.get("id")
		if not rid:
			continue
		props = rule.get("properties") or {}
		rule_tags = list(props.get("tags") or [])
		rule_cwes: list[str] = []
		for t in rule_tags:
			m = re.search(r"cwe-(\d+)", str(t), flags=re.IGNORECASE)
			if m:
				rule_cwes.append(m.group(1))
		rule_meta[rid] = {
			"name": rule.get("name") or rule.get("shortDescription", {}).get("text"),
			"severity": props.get("security-severity")
			or (rule.get("defaultConfiguration", {}) or {}).get("level"),
			"tags": rule_tags,
			"cwes": sorted(set(rule_cwes), key=lambda x: int(x) if x.isdigit() else 10**9),
		}

	findings: list[CodeQLFinding] = []
	results = run0.get("results") or []
	for res in results:
		rid = res.get("ruleId") or ""
		msg = (res.get("message") or {}).get("text") or ""
		meta = rule_meta.get(rid, {})
		result_tags = tuple(
			str(t)
			for t in (
				meta.get("tags")
				or (res.get("properties") or {}).get("tags")
				or []
			)
		)
		result_cwes = tuple(
			str(c)
			for c in (
				meta.get("cwes")
				or (res.get("properties") or {}).get("cwe")
				or []
			)
		)
		severity = str(meta.get("severity")) if meta.get("severity") is not None else None
		rule_name = meta.get("name")

		file_path = None
		start_line = None
		start_column = None
		locs = res.get("locations") or []
		if locs:
			phys = (locs[0].get("physicalLocation") or {})
			artifact = (phys.get("artifactLocation") or {}).get("uri")
			region = phys.get("region") or {}
			if artifact:
				file_path = str(artifact)
			if "startLine" in region:
				start_line = int(region["startLine"])
			if "startColumn" in region:
				start_column = int(region["startColumn"])

		findings.append(
			CodeQLFinding(
				rule_id=rid,
				rule_name=str(rule_name) if rule_name else None,
				message=str(msg),
				severity=severity,
				cwes=result_cwes,
				tags=result_tags,
				file_path=file_path,
				start_line=start_line,
				start_column=start_column,
			)
		)

	return findings


def _finding_text(f: CodeQLFinding) -> str:
	parts = [f.rule_id]
	if f.rule_name:
		parts.append(f.rule_name)
	if f.message:
		parts.append(f.message)
	if f.cwes:
		parts.append("CWE:" + ",".join(f.cwes))
	return " | ".join(parts)


def map_findings_to_bibifi_properties(findings: list[CodeQLFinding]) -> list[list[CodeQLFinding]]:
	"""Return a list of length 9: findings grouped by BiBiFi property index."""
	grouped: list[list[CodeQLFinding]] = [[] for _ in range(9)]

	def has_kw(text: str, *keywords: str) -> bool:
		t = text.lower()
		return any(k.lower() in t for k in keywords)

	for f in findings:
		text = " ".join([
			f.rule_id,
			f.rule_name or "",
			f.message,
			" ".join(f.tags),
			" ".join(f.cwes),
		])
		cwe = set(f.cwes)

		# Property 1: encryption of sensitive fields / no cleartext.
		if (
			cwe.intersection({"311", "319"})
			or has_kw(text, "cleartext", "plaintext", "unencrypted", "insecure transport", "sensitive")
		):
			grouped[0].append(f)

		# Property 2: IV/seed not fixed.
		if (
			cwe.intersection({"329", "780", "326"})
			or has_kw(text, "initialization vector", "iv", "fixed iv", "nonce reuse", "static iv")
		):
			grouped[1].append(f)

		# Property 3: freshness / nonce / timestamp replay prevention.
		if (
			cwe.intersection({"294", "330", "340"})
			or has_kw(text, "replay", "nonce", "timestamp", "freshness", "window")
		):
			grouped[2].append(f)

		# Property 4: keyed MAC / authenticity verification.
		if (
			cwe.intersection({"345", "353", "347", "306", "287"})
			or has_kw(text, "mac", "hmac", "integrity", "authenticity", "signature", "verify")
		):
			grouped[3].append(f)

		# # Property 5: parser/validator rejects on any failed check.
		# if (
		# 	cwe.intersection({"20", "252", "703"})
		# 	or has_kw(text, "unchecked", "ignored", "not checked", "improper validation", "missing check")
		# ):
		# 	grouped[4].append(f)

		# Property 5: attacker-observable metadata independent of secrets.
		if (
			cwe.intersection({"203", "204", "208", "209"})
			or has_kw(text, "side channel", "timing", "error message", "information exposure", "length")
		):
			grouped[4].append(f)

		# Property 6: privileged operation only after authentication.
		if (
			cwe.intersection({"306", "287"})
			or has_kw(text, "missing authentication", "unauthenticated", "improper authentication", "authorization")
		):
			grouped[5].append(f)

		# Property 7: robustness (uncaught exceptions, size limits, desync).
		if (
			cwe.intersection({"248", "755", "703", "400"})
			or has_kw(text, "exception", "panic", "resource consumption", "allocation", "size", "dos")
		):
			grouped[6].append(f)

		# Property 8: guessable/derivable secrets (weak randomness, derived secrets).
		if (
			cwe.intersection({"321", "330", "335", "338"})
			or has_kw(text, "hardcoded", "predictable", "weak random", "prng", "seed", "derived")
		):
			grouped[7].append(f)

	# De-duplicate within each property.
	for i in range(8):
		seen = set()
		uniq: list[CodeQLFinding] = []
		for f in grouped[i]:
			key = (f.rule_id, f.file_path, f.start_line, f.message)
			if key in seen:
				continue
			seen.add(key)
			uniq.append(f)
		grouped[i] = uniq

	return grouped


def analyze_repo_with_codeql(
	repo_path: Path,
	*,
	timeout_s: int = 1800,
) -> dict[str, Any]:
	"""Analyze a repository with CodeQL.

	Returns a dict with:
	- ok: bool
	- language: str | None
	- findings: list[dict]
	- property_failures: list[list[dict]]  # length 9
	- error: str | None

	On any failure, returns ok=False and empty findings, and property_failures empty.
	(Caller may treat failure as all-pass.)
	"""
	codeql = _codeql_binary()
	if not codeql:
		return {
			"ok": False,
			"language": None,
			"findings": [],
			"property_failures": [[] for _ in range(BIBIFI_PROPERTY_COUNT)],
			"error": f"{CODEQL_BINARY_ENV} is not set",
		}

	language = detect_codeql_language(repo_path)
	if not language:
		return {
			"ok": False,
			"language": None,
			"findings": [],
			"property_failures": [[] for _ in range(BIBIFI_PROPERTY_COUNT)],
			"error": "Could not detect a supported CodeQL language",
		}

	suite = _suite_for_language(language)

	try:
		with tempfile.TemporaryDirectory(prefix="vibetest-codeql-") as tmp:
			tmp_path = Path(tmp)
			db_dir = tmp_path / "db"
			sarif_path = tmp_path / "results.sarif"

			build_mode = _build_mode_for_language(language)

			_run(
				[
					codeql,
					"database",
					"create",
					str(db_dir),
					"--source-root",
					str(repo_path),
					"--language",
					language,
					"--build-mode",
					build_mode,
					"--overwrite",
				],
				timeout_s=timeout_s,
			)

			_run(
				[
					codeql,
					"database",
					"analyze",
					str(db_dir),
					suite,
					"--download",
					"--format=sarifv2.1.0",
					f"--output={sarif_path}",
					"--rerun",
				],
				timeout_s=timeout_s,
			)

			sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
			findings = _parse_sarif(sarif)
			grouped = map_findings_to_bibifi_properties(findings)

			return {
				"ok": True,
				"language": language,
				"suite": suite,
				"findings": [
					{
						"rule_id": f.rule_id,
						"rule_name": f.rule_name,
						"message": f.message,
						"severity": f.severity,
						"cwes": list(f.cwes),
						"tags": list(f.tags),
						"file_path": f.file_path,
						"start_line": f.start_line,
						"start_column": f.start_column,
					}
					for f in findings
				],
				"property_failures": [
					[
						{
							"rule_id": f.rule_id,
							"rule_name": f.rule_name,
							"message": f.message,
							"severity": f.severity,
							"cwes": list(f.cwes),
							"file_path": f.file_path,
							"start_line": f.start_line,
						}
						for f in grouped[i]
					]
					for i in range(len(grouped))
				],
				"error": None,
			}
	except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as e:
		detail = str(e)
		if isinstance(e, subprocess.CalledProcessError):
			detail = (
				f"Command failed: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}\n"
				f"stdout:\n{e.stdout}\n\n"
				f"stderr:\n{e.stderr}"
			)
		return {
			"ok": False,
			"language": language,
			"suite": suite,
			"findings": [],
			"property_failures": [[] for _ in range(BIBIFI_PROPERTY_COUNT)],
			"error": detail,
		}
