"""
Final verification: check that the theoretical eigenvalue bounds
from equation (13)-(14) of the solution match numerical computation.
"""

import numpy as np
from numpy import kron
from solver import generate_test_problem, build_system_naive
from scipy.linalg import eigh


def verify_bounds():
    print("Verification of theoretical eigenvalue bounds")
    print("=" * 60)

    for frac in [0.1, 0.3, 0.5, 0.8]:
        prob = generate_test_problem(n=12, r=4, d=3, frac_observed=frac, seed=42)
        A_mat, _ = build_system_naive(prob)

        K = prob['K']
        Z = prob['Z']
        r = prob['r']
        n = prob['n']
        lam = prob['lam']

        # Build P
        Gamma = Z.T @ Z
        P_mat = kron(Gamma, K @ K) + lam * kron(np.eye(r), K)

        # Eigenvalues of P^{-1} A
        P_inv_A = np.linalg.solve(P_mat, A_mat)
        eigvals_PA = np.sort(np.linalg.eigvals(P_inv_A).real)

        # Theoretical bounds
        eigvals_K = np.sort(eigh(K, eigvals_only=True))
        eigvals_G = np.sort(eigh(Gamma, eigvals_only=True))

        sigma_max = eigvals_G[-1]
        lambda_max_K = eigvals_K[-1]

        alpha_theory = lam / (sigma_max * lambda_max_K + lam)
        kappa_theory = 1 + sigma_max * lambda_max_K / lam

        print(f"\n  frac_observed = {frac}")
        print(f"    Eigenvalue range (numerical):   [{eigvals_PA[0]:.6f}, {eigvals_PA[-1]:.6f}]")
        print(f"    Theoretical lower bound α:       {alpha_theory:.6f}")
        print(f"    Theoretical upper bound:         1.000000")
        print(f"    α ≤ min(eig)?  {alpha_theory <= eigvals_PA[0] + 1e-10}")
        print(f"    max(eig) ≤ 1?  {eigvals_PA[-1] <= 1.0 + 1e-10}")
        print(f"    κ(P⁻¹A) numerical:              {eigvals_PA[-1]/eigvals_PA[0]:.2f}")
        print(f"    κ(P⁻¹A) theoretical bound:       {kappa_theory:.2f}")
        print(f"    Bound valid?  {eigvals_PA[-1]/eigvals_PA[0] <= kappa_theory + 1e-6}")


if __name__ == '__main__':
    verify_bounds()
