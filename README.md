# First Proof — Problem 10 (Clean Run)

An unguided attempt by Claude Code (Opus 4.6) to solve Problem 10 from the "First Proof" benchmark (arXiv:2602.05192v1).

Claude Code receives only the problem statement and instructions for how to work. No mathematical hints, solution strategies, or implementation plans are provided. See `EXPERIMENT.md` for the full experimental design.

## Files

- `PROBLEM.md` — The problem statement (LaTeX)
- `CLAUDE-CODE-PROMPT.md` — Initial prompt for Claude Code
- `CLAUDE-CODE-RESTART.md` — Restart prompt if needed
- `EXPERIMENT.md` — How and why this clean testbed was designed

## To run

```bash
cd ~/dev/first-proof-p10
git init && git add -A && git commit -m "Initial commit: clean testbed for Problem 10"
claude   # launch Claude Code
# paste contents of CLAUDE-CODE-PROMPT.md
```
