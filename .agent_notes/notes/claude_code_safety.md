Claude Code safety backend lives in `vibetest/agent/claude_code_agent.py` and is wired through `experiments/safety.py` via `--vibetest-backend claude-code`.
Use the host runner with read-only Claude tools (`Read,Grep,Glob,LS`); this path was validated end-to-end with the real `claude` CLI on single-trace and all-traces safety runs.
This checkout was missing `vibetest/usage.py`, `experiments/result_naming.py`, and `experiments/usage_utils.py`; lightweight replacements were added because existing experiment code already imported them.
