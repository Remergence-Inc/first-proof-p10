# Restart / Continue Prompt for Claude Code

Paste this if Claude Code stops, crashes, or needs to be restarted.

---

You are resuming work on a research mathematics problem (Problem 10 from the "First Proof" benchmark). The problem statement is in `PROBLEM.md`. Previous work has been done in this repo.

## Resume instructions

1. Run `git log --oneline -15` to see what has been done.
2. Check for any notes or dead-end logs in the repo.
3. Check for code in `src/` or similar directories, and run any existing tests.
4. Check if `writeup/solution.md` exists and assess its completeness.
5. Continue from wherever you left off. If the solution is written but incomplete, finish it. If tests are failing, fix them. If scaling experiments haven't been run, run them.

Do not ask for human input. Do not search the web. Continue working until the solution is complete.

Go.
