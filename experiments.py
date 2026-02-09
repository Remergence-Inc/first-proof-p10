"""
Experiments for the PCG solver.
Generates data for the writeup: convergence profiles, scaling behavior,
eigenvalue clustering analysis.
"""

import numpy as np
from solver import (
    generate_test_problem, solve_naive, pcg_solve,
    matvec_efficient, FullDataPreconditioner, build_system_naive
)
import time


def experiment_convergence_profiles():
    """Show convergence with/without preconditioner for various observation fractions."""
    print("=" * 70)
    print("EXPERIMENT 1: Convergence profiles")
    print("=" * 70)

    n, r, d = 20, 5, 3

    for frac in [0.05, 0.1, 0.3, 0.5]:
        prob = generate_test_problem(n=n, r=r, d=d, frac_observed=frac, seed=42)

        _, info_no = pcg_solve(prob, precond=None, tol=1e-12, maxiter=500)
        precond = FullDataPreconditioner(prob)
        _, info_yes = pcg_solve(prob, precond=precond, tol=1e-12, maxiter=500)

        print(f"\n  frac_observed = {frac}")
        print(f"    No precond:   {info_no['iterations']} iterations, "
              f"final residual = {info_no['final_residual']:.2e}")
        print(f"    With precond: {info_yes['iterations']} iterations, "
              f"final residual = {info_yes['final_residual']:.2e}")
        print(f"    Speedup: {info_no['iterations'] / max(info_yes['iterations'], 1):.1f}x")


def experiment_eigenvalue_analysis():
    """Analyze eigenvalue clustering of P^{-1} A."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Eigenvalue clustering of P^{-1} A")
    print("=" * 70)

    for frac in [0.1, 0.3, 0.5, 0.8]:
        prob = generate_test_problem(n=12, r=4, d=3, frac_observed=frac, seed=42)
        A_mat, _ = build_system_naive(prob)

        # Build preconditioner matrix P explicitly for eigenvalue analysis
        K = prob['K']
        Z = prob['Z']
        r = prob['r']
        lam = prob['lam']
        Gamma = Z.T @ Z
        P_mat = np.kron(Gamma, K @ K) + lam * np.kron(np.eye(r), K)

        # Compute eigenvalues of P^{-1} A
        P_inv_A = np.linalg.solve(P_mat, A_mat)
        eigvals = np.linalg.eigvals(P_inv_A).real
        eigvals.sort()

        print(f"\n  frac_observed = {frac}")
        print(f"    Eigenvalue range: [{eigvals[0]:.4f}, {eigvals[-1]:.4f}]")
        print(f"    Condition number of P^{{-1}}A: {eigvals[-1] / max(eigvals[0], 1e-15):.2f}")
        print(f"    Mean eigenvalue: {eigvals.mean():.4f}")
        print(f"    Std deviation:   {eigvals.std():.4f}")
        # How many eigenvalues are close to 1?
        near_one = np.sum(np.abs(eigvals - 1.0) < 0.1)
        print(f"    Eigenvalues within 0.1 of 1: {near_one} / {len(eigvals)}")


def experiment_scaling():
    """Measure wall-clock time scaling with problem size."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Scaling with problem size")
    print("=" * 70)

    r, d = 5, 3
    results = []

    for n in [10, 20, 40, 60, 80]:
        mode_sizes = [n, n + 5, n + 3]
        M = (n + 5) * (n + 3)
        N = n * M
        q = max(int(0.05 * N), n * r)

        prob = generate_test_problem(n=n, r=r, d=d, mode_sizes=mode_sizes,
                                     frac_observed=0.05, seed=42)

        w = np.random.default_rng(42).standard_normal(n * r)

        # Time mat-vec
        n_reps = max(100, 10000 // n)
        t0 = time.time()
        for _ in range(n_reps):
            matvec_efficient(w, prob)
        t_mv = (time.time() - t0) / n_reps

        # Time preconditioner
        precond = FullDataPreconditioner(prob)
        t0 = time.time()
        for _ in range(n_reps):
            precond.apply(w)
        t_pc = (time.time() - t0) / n_reps

        # Time full solve
        precond2 = FullDataPreconditioner(prob)
        t0 = time.time()
        _, info = pcg_solve(prob, precond=precond2, tol=1e-10, maxiter=500)
        t_solve = time.time() - t0

        results.append((n, prob['M'], prob['N'], prob['q'], t_mv, t_pc,
                        info['iterations'], t_solve))
        print(f"  n={n:3d}, M={prob['M']:6d}, N={prob['N']:8d}, q={prob['q']:6d}: "
              f"matvec={t_mv*1e6:.0f}μs, precond={t_pc*1e6:.0f}μs, "
              f"iters={info['iterations']}, solve={t_solve*1000:.1f}ms")

    return results


def experiment_rank_sensitivity():
    """How does rank r affect convergence?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Rank sensitivity")
    print("=" * 70)

    n, d = 20, 3
    for r in [2, 5, 10, 15]:
        prob = generate_test_problem(n=n, r=r, d=d, frac_observed=0.1, seed=42)

        _, info_no = pcg_solve(prob, precond=None, tol=1e-10, maxiter=500)
        precond = FullDataPreconditioner(prob)
        _, info_yes = pcg_solve(prob, precond=precond, tol=1e-10, maxiter=500)

        print(f"  r={r:2d}: no precond = {info_no['iterations']:3d} iters, "
              f"with precond = {info_yes['iterations']:3d} iters, "
              f"speedup = {info_no['iterations'] / max(info_yes['iterations'], 1):.1f}x")


def experiment_vs_naive():
    """
    Compare PCG solution accuracy with naive dense solve.
    Verify they produce the same answer.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: PCG vs naive dense solve (accuracy)")
    print("=" * 70)

    for frac in [0.1, 0.3, 0.5]:
        prob = generate_test_problem(n=15, r=4, d=3, frac_observed=frac, seed=42)
        W_naive = solve_naive(prob)

        precond = FullDataPreconditioner(prob)
        W_pcg, info = pcg_solve(prob, precond=precond, tol=1e-12, maxiter=500)

        err = np.linalg.norm(W_pcg - W_naive) / np.linalg.norm(W_naive)
        print(f"  frac={frac}: ||W_pcg - W_naive||/||W_naive|| = {err:.2e}, "
              f"iters = {info['iterations']}")


def experiment_condition_numbers():
    """Report condition numbers of the original system and preconditioned system."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Condition numbers")
    print("=" * 70)

    for frac in [0.05, 0.1, 0.3, 0.5, 0.8]:
        prob = generate_test_problem(n=12, r=4, d=3, frac_observed=frac, seed=42)
        A_mat, _ = build_system_naive(prob)

        cond_A = np.linalg.cond(A_mat)

        K = prob['K']
        Z = prob['Z']
        r = prob['r']
        lam = prob['lam']
        Gamma = Z.T @ Z
        P_mat = np.kron(Gamma, K @ K) + lam * np.kron(np.eye(r), K)
        P_inv_A = np.linalg.solve(P_mat, A_mat)
        cond_PA = np.linalg.cond(P_inv_A)

        print(f"  frac={frac:.2f}: κ(A) = {cond_A:.2e}, "
              f"κ(P⁻¹A) = {cond_PA:.2e}, "
              f"reduction = {cond_A / cond_PA:.1f}x")


if __name__ == '__main__':
    experiment_convergence_profiles()
    experiment_eigenvalue_analysis()
    experiment_scaling()
    experiment_rank_sensitivity()
    experiment_vs_naive()
    experiment_condition_numbers()
