Claude Code safety backend lives in `vibetest/agent/claude_code_agent.py` and is wired through `experiments/safety.py` via `--vibetest-backend claude-code`.
Use the host runner with read-only Claude tools (`Read,Grep,Glob,LS`); this path was validated end-to-end with the real `claude` CLI on single-trace and all-traces safety runs.
This checkout was missing `vibetest/usage.py`, `experiments/result_naming.py`, and `experiments/usage_utils.py`; lightweight replacements were added because existing experiment code already imported them.
PR for the Claude safety backend is open at `https://github.com/Penn-Agentic-Lab/vibetest/pull/1` from branch `claude-code-safety-agent`.
Public HF ImpossibleBench repos (`fjzzq2002/impossible_livecodebench` and `fjzzq2002/impossible_swebench`) expose parquet benchmark rows only; `experiments/safety.py --safety-mode impossiblebench` needs transcript-bearing `oneoff` and `original` `.eval` logs, so ask for the exact HF repo/path or direct URLs if those logs are private.
ImpossibleBench scoring also needs a real API-backed `--scorer-model`; this environment currently has no `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`, or `HUGGING_FACE_HUB_TOKEN` set.
`.venv312-lite` was recreated with the lightweight safety-run stack (`inspect-ai`, `jsonlines`, `datasets`, editable `vibetest`) so future safety commands should use `.venv312-lite/bin/python`.
