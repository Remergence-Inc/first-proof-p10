# Experiment Design: Clean Testbed for Problem 10

## Background

This is the second of two experiments testing whether Claude (Opus 4.6) can solve Problem 10 from the "First Proof" benchmark paper (arXiv:2602.05192v1).

### Experiment 1 (the guided run)

In the first experiment, a Claude Opus 4.6 instance running in claude.ai was given the full paper, asked to analyze which problems were tractable, and then asked to produce a detailed project scaffold. That instance:

- Selected Problem 10 as the most tractable (algorithmic design rather than abstract proof)
- Derived the key mathematical insight: the mat-vec can be computed in $O(n^2 r + qr)$ by exploiting the Kronecker identity $(C \otimes D)\operatorname{vec}(X) = \operatorname{vec}(DXC^\top)$ and only computing observed entries of the intermediate matrix
- Identified four candidate preconditioners and ranked them
- Produced detailed implementation plans with function signatures, test criteria, and phase gates

Claude Code (also Opus 4.6) was then given this scaffold and executed the plan. It completed all six phases — naive solver, efficient mat-vec, preconditioners, full PCG solver, scaling experiments, and writeup — in **under 13 minutes** with 106 passing tests.

The result was a correct and well-written solution. However, this primarily tested Claude Code's ability to **follow detailed instructions from a prior Claude instance** that had already done the hard mathematical thinking.

### Experiment 2 (this experiment — the clean run)

This experiment tests whether Claude Code can solve the problem **from scratch**, with no hints about the approach, no pre-derived identities, no suggested preconditioners, and no implementation plan.

## What Claude Code receives

Exactly three files:

1. **`PROBLEM.md`** — A faithful transcription of Problem 10 from the paper, rendered in LaTeX. This includes the full notation table, the linear system, and the question as stated by the authors. No analysis, commentary, or hints are included.

2. **`CLAUDE-CODE-PROMPT.md`** — Instructions for how to work (use Python via pyenv, write tests, commit to git, don't ask for help, don't search the web) and what the authors expect in a solution (step-by-step derivation, specific preconditioner, complexity analysis, ideally a working implementation). It does **not** tell Claude Code *what* the key identity is, *which* preconditioner to use, or *how* to structure the mat-vec.

3. **`CLAUDE-CODE-RESTART.md`** — A restart prompt in case Claude Code crashes or times out.

## What Claude Code does NOT receive

- Any mention of the Kronecker mixed-product identity
- Any analysis of how the mat-vec can be decomposed
- Any named preconditioner candidates
- Any implementation plan, function signatures, or phase structure
- Any notes, dead-ends logs, or previous analysis
- The seed notes from Experiment 1
- The solution from Experiment 1
- The full paper (only Problem 10 is given)

## What we are measuring

1. **Can Claude Code independently derive the efficient mat-vec?** The key insight — that the Kronecker identity converts the $N$-dimensional intermediate product into something that only needs to be evaluated at $q$ observed positions — is the mathematical crux of the problem.

2. **What preconditioner does it choose, and is the choice reasonable?** There are multiple valid options. We want to see if it identifies one and justifies it.

3. **Does it validate its own work?** Writing a naive solver and checking the efficient implementation against it is a critical self-verification step that we did not suggest.

4. **How long does it take?** The guided run took ~13 minutes. The clean run may take significantly longer since it must discover the approach rather than follow instructions.

5. **Is the solution quality comparable?** Both the mathematical rigor and the clarity of exposition.

## Environment

- Claude Code running Claude Opus 4.6
- Ubuntu under WSL2
- Python 3.14.2 via `~/.pyenv/versions/3.14.2/bin/python`
- numpy, scipy available via pip
- Git initialized in the repo

## Controls

- The human operator does not interact with Claude Code beyond pasting the initial prompt (or restart prompt if needed).
- No feedback, corrections, hints, or encouragement are provided.
- The human operator is a mathematician but did not read the paper in detail and makes no mathematical contributions.

## Relation to the First Proof benchmark

The First Proof paper (released February 5, 2026) presents ten research-level math questions and invites the community to test AI systems on them. The authors will release their own answers on February 13, 2026. This experiment is being conducted on February 9, 2026, before the answers are released.

The paper explicitly invites participants to "share a complete transcript of their interaction with an AI system." This repo, including git history, serves as that transcript.
