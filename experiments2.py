"""
Additional experiments: preconditioner comparisons and spectral analysis.
"""

import numpy as np
from numpy import kron
from solver import (
    generate_test_problem, pcg_solve, FullDataPreconditioner,
    build_system_naive, matvec_efficient
)
from scipy.linalg import eigh
import time


class RegularizationPreconditioner:
    """
    Simple preconditioner: P = λ (I_r ⊗ K).
    Apply P^{-1}: solve K x_l = b_l / λ for each column l.
    Uses eigendecomposition of K.
    """
    def __init__(self, prob):
        K = prob['K']
        self.n = prob['n']
        self.r = prob['r']
        self.lam = prob['lam']

        eigvals, U = eigh(K)
        self.U = U
        self.eigvals = eigvals
        # P^{-1} diagonal: 1 / (lambda * lambda_i)
        self.inv_diag = 1.0 / (self.lam * eigvals)
        self.inv_diag[eigvals < 1e-14] = 0.0

    def apply(self, x_vec):
        X = x_vec.reshape((self.n, self.r), order='F')
        Y = self.U.T @ X             # n x r
        Y *= self.inv_diag[:, None]   # scale
        return (self.U @ Y).ravel(order='F')


class ScaledFullDataPreconditioner:
    """
    Like FullDataPreconditioner but scales the data term by q/N:
    P = (q/N) Γ ⊗ K² + λ I_r ⊗ K
    """
    def __init__(self, prob):
        K = prob['K']
        n = prob['n']
        r = prob['r']
        lam = prob['lam']
        factors = prob['factors']
        k = prob['k']
        q = prob['q']
        N = prob['N']

        Gamma = np.ones((r, r))
        for i in range(len(factors)):
            if i != k:
                AiTAi = factors[i].T @ factors[i]
                Gamma *= AiTAi

        scale = q / N

        eigvals_K, U = eigh(K)
        self.U = U
        self.eigvals_K = eigvals_K

        eigvals_G, V = eigh(Gamma)
        self.V = V
        self.eigvals_G = eigvals_G

        # d[i, p] = scale * sigma_p * lambda_i^2 + lambda * lambda_i
        self.diag = scale * np.outer(eigvals_K**2, eigvals_G) + lam * eigvals_K[:, None]
        self.diag[self.diag < 1e-14] = 1.0

        self.n = n
        self.r = r

    def apply(self, x_vec):
        X = x_vec.reshape((self.n, self.r), order='F')
        Y1 = self.U.T @ X
        Y2 = Y1 @ self.V
        Y2 /= self.diag
        Y3 = Y2 @ self.V.T
        return (self.U @ Y3).ravel(order='F')


def compare_preconditioners():
    """Compare different preconditioner choices."""
    print("=" * 70)
    print("Preconditioner comparison")
    print("=" * 70)

    n, r, d = 20, 5, 3

    print(f"\n{'frac':>6s}  {'None':>8s}  {'λ(I⊗K)':>8s}  {'Full':>8s}  {'Scaled':>8s}")
    print("-" * 50)

    for frac in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
        prob = generate_test_problem(n=n, r=r, d=d, frac_observed=frac, seed=42)

        _, i0 = pcg_solve(prob, precond=None, tol=1e-10, maxiter=500)
        _, i1 = pcg_solve(prob, precond=RegularizationPreconditioner(prob),
                          tol=1e-10, maxiter=500)
        _, i2 = pcg_solve(prob, precond=FullDataPreconditioner(prob),
                          tol=1e-10, maxiter=500)
        _, i3 = pcg_solve(prob, precond=ScaledFullDataPreconditioner(prob),
                          tol=1e-10, maxiter=500)

        print(f"{frac:6.2f}  {i0['iterations']:8d}  {i1['iterations']:8d}  "
              f"{i2['iterations']:8d}  {i3['iterations']:8d}")


def spectral_analysis():
    """
    Detailed spectral analysis of P^{-1}A for the full-data preconditioner.
    Shows why the preconditioner works: eigenvalues are bounded in (0, 1].
    """
    print("\n" + "=" * 70)
    print("Spectral analysis of P^{-1}A")
    print("=" * 70)

    prob = generate_test_problem(n=12, r=4, d=3, frac_observed=0.3, seed=42)
    A_mat, _ = build_system_naive(prob)

    K = prob['K']
    Z = prob['Z']
    r = prob['r']
    n = prob['n']
    lam = prob['lam']
    Gamma = Z.T @ Z
    P_mat = kron(Gamma, K @ K) + lam * kron(np.eye(r), K)

    eigvals_PA = np.sort(np.linalg.eigvals(np.linalg.solve(P_mat, A_mat)).real)

    print(f"\n  Eigenvalues of P^{{-1}}A:")
    print(f"    min = {eigvals_PA[0]:.6f}")
    print(f"    max = {eigvals_PA[-1]:.6f}")
    print(f"    All ≤ 1? {np.all(eigvals_PA <= 1.0 + 1e-10)}")
    print(f"    All > 0? {np.all(eigvals_PA > -1e-10)}")

    # Condition number
    cond = eigvals_PA[-1] / eigvals_PA[0]
    print(f"    Condition number: {cond:.2f}")

    # Compare with condition number of A
    eigvals_A = np.sort(np.linalg.eigvals(A_mat).real)
    cond_A = eigvals_A[-1] / eigvals_A[0]
    print(f"    Condition number of A: {cond_A:.2f}")
    print(f"    Reduction factor: {cond_A / cond:.1f}x")

    # Verify the theoretical bound: eigenvalues of P^{-1}A lie in (0, 1]
    # because A ≤ P (since SS^T ≤ I implies data term of A ≤ data term of P)
    print("\n  Theoretical bound verification:")
    print(f"    A ≤ P (data term)?  max eigenvalue of P^{{-1}}A = {eigvals_PA[-1]:.6f} ≤ 1")
    print(f"    Lower bound from regularization: min = {eigvals_PA[0]:.6f}")


def verify_psd_ordering():
    """
    Verify that A ≤ P in the PSD ordering.
    This is the key theoretical justification for the preconditioner.
    """
    print("\n" + "=" * 70)
    print("Verify A ≤ P (PSD ordering)")
    print("=" * 70)

    for frac in [0.1, 0.3, 0.5, 0.8]:
        prob = generate_test_problem(n=10, r=3, d=3, frac_observed=frac, seed=42)
        A_mat, _ = build_system_naive(prob)

        K = prob['K']
        Z = prob['Z']
        r = prob['r']
        lam = prob['lam']
        Gamma = Z.T @ Z
        P_mat = kron(Gamma, K @ K) + lam * kron(np.eye(r), K)

        # Check P - A is PSD
        diff = P_mat - A_mat
        eigvals = np.linalg.eigvalsh(diff)
        print(f"  frac={frac}: min eigenvalue of (P-A) = {eigvals[0]:.6e} "
              f"({'PSD ✓' if eigvals[0] > -1e-10 else 'NOT PSD ✗'})")


def timing_comparison():
    """
    Time the mat-vec vs forming and multiplying by the full matrix.
    Shows the practical speedup from implicit computation.
    """
    print("\n" + "=" * 70)
    print("Timing: efficient mat-vec vs explicit formation")
    print("=" * 70)

    r, d = 5, 3

    for n in [10, 20, 30]:
        mode_sizes = [n, n+5, n+3]
        prob = generate_test_problem(n=n, r=r, d=d, mode_sizes=mode_sizes,
                                     frac_observed=0.1, seed=42)
        w = np.random.default_rng(42).standard_normal(n * r)

        # Time efficient mat-vec
        n_reps = 100
        t0 = time.time()
        for _ in range(n_reps):
            matvec_efficient(w, prob)
        t_eff = (time.time() - t0) / n_reps

        # Time building and using full matrix
        t0 = time.time()
        A_mat, _ = build_system_naive(prob)
        t_build = time.time() - t0

        t0 = time.time()
        for _ in range(n_reps):
            A_mat @ w
        t_dense = (time.time() - t0) / n_reps

        print(f"  n={n:2d}, nr={n*r:3d}, N={prob['N']:7d}: "
              f"efficient={t_eff*1e6:.0f}μs, "
              f"dense_mv={t_dense*1e6:.0f}μs, "
              f"build={t_build*1000:.1f}ms")


if __name__ == '__main__':
    compare_preconditioners()
    spectral_analysis()
    verify_psd_ordering()
    timing_comparison()
