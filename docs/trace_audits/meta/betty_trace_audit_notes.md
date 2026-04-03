# Betty trace audit notes

- Use `login01.betty.parcc.upenn.edu`. `login` and `login02` were not accessible from this environment.
- Betty home is storage-constrained: `/vast/home/d/davisrbr` was `46.74 GB / 50 GB` and in inode `GRACE_EXPIRED`.
- Keep new work, caches, Hugging Face downloads, and virtualenvs out of Betty home. Prefer node-local `/tmp` or `/local` inside the Slurm job.
- Betty already sets `UV_CACHE_DIR=/tmp/uv-cache-$USER` in interactive shell config; keep that behavior for remote bootstrap.
- For this audit path, inject only `ANTHROPIC_API_KEY` transiently from local `env.sh` or the repo-local `.env`. Do not write long-lived secret files in Betty home.
- Use `claude-opus-4-6` as the first-pass monitor. Live API checks on 2026-04-01 confirmed `claude-opus-4-6` works.
