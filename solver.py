"""
Efficient PCG solver for the mode-k subproblem in CP tensor decomposition
with missing data and RKHS regularization.

System:  A vec(W) = b
where  A = (Z ⊗ K)^T S S^T (Z ⊗ K) + λ (I_r ⊗ K)
       b = (I_r ⊗ K) vec(B)

Key identity used throughout:
  (C ⊗ D) vec(X) = vec(D X C^T)

This allows computing mat-vecs in O(n^2 r + qr) instead of O(N).
"""

import numpy as np
from numpy import kron
from scipy.linalg import eigh
import time


# ============================================================
# Data generation
# ============================================================

def generate_test_problem(n, r, d, mode_sizes=None, frac_observed=0.1, seed=42):
    """
    Generate a random test instance.

    Parameters
    ----------
    n : int
        Size of mode k.
    r : int
        CP rank.
    d : int
        Number of tensor modes.
    mode_sizes : list or None
        Sizes of all modes. If None, all are set to n.
    frac_observed : float
        Fraction of tensor entries observed.
    seed : int
        Random seed.

    Returns
    -------
    dict with all problem data.
    """
    rng = np.random.default_rng(seed)

    if mode_sizes is None:
        mode_sizes = [n] * d

    k = 0  # mode k is the first mode for simplicity
    n_k = mode_sizes[k]
    assert n_k == n

    # Factor matrices for all modes
    factors = [rng.standard_normal((mode_sizes[i], r)) for i in range(d)]

    # Kernel matrix K (symmetric positive definite)
    H = rng.standard_normal((n, n))
    K = H @ H.T + 0.1 * np.eye(n)  # ensure positive definite

    # Regularization parameter
    lam = 0.01

    # Compute M = product of all mode sizes except k
    M = 1
    for i in range(d):
        if i != k:
            M *= mode_sizes[i]
    N = n * M

    # Number of observed entries
    q = max(int(frac_observed * N), n * r)  # at least nr to have a determined system
    q = min(q, N)

    # Select q random positions (as linear indices into the n x M unfolding)
    all_indices = rng.choice(N, size=q, replace=False)
    all_indices.sort()

    # Convert linear indices to (row, col) in the n x M unfolding
    # vec stacks columns: index = col * n + row
    obs_rows = all_indices % n       # row indices in 0..n-1
    obs_cols = all_indices // n      # col indices in 0..M-1

    # Compute the Khatri-Rao product Z (M x r) -- only for small problems!
    # For large problems, we compute rows on the fly.
    other_factors = [factors[i] for i in range(d) if i != k]

    # Build Z by Khatri-Rao product
    Z = _khatri_rao(other_factors)
    assert Z.shape == (M, r)

    # Precompute the rows of Z at observed column indices
    Z_obs = Z[obs_cols, :]  # q x r

    # Ground truth W
    W_true = rng.standard_normal((n, r))

    # Compute A_k = K W
    A_k_true = K @ W_true

    # Build the full tensor (mode-k unfolding): T_tilde = A_k Z^T
    # But we only observe q entries
    # T_full = A_k_true @ Z.T  # n x M -- this is O(N), only for testing
    # Observed values: T_full[obs_rows, obs_cols]

    # Compute observed values without forming full tensor
    obs_vals = np.sum(A_k_true[obs_rows, :] * Z_obs, axis=1)  # q-vector

    # Compute B = T_tilde Z (MTTKRP with sparse T_tilde)
    # B[i, l] = sum over observed entries with row i: t_s * Z(j_s, l)
    B = np.zeros((n, r))
    for s in range(q):
        B[obs_rows[s], :] += obs_vals[s] * Z_obs[s, :]

    return {
        'n': n, 'r': r, 'd': d, 'M': M, 'N': N, 'q': q, 'k': k,
        'mode_sizes': mode_sizes,
        'factors': factors,
        'K': K, 'lam': lam,
        'Z': Z,  # M x r (only for small problems)
        'Z_obs': Z_obs,  # q x r
        'obs_rows': obs_rows,  # q-vector, entries in 0..n-1
        'obs_cols': obs_cols,  # q-vector, entries in 0..M-1
        'obs_vals': obs_vals,  # q-vector
        'B': B,  # n x r
        'W_true': W_true,
    }


def _khatri_rao(matrices):
    """Compute Khatri-Rao product of a list of matrices (column-wise Kronecker)."""
    result = matrices[0]
    for mat in matrices[1:]:
        r = result.shape[1]
        assert mat.shape[1] == r
        # Khatri-Rao: column l of result is kron(result[:,l], mat[:,l])
        new_rows = result.shape[0] * mat.shape[0]
        new_result = np.zeros((new_rows, r))
        for l in range(r):
            new_result[:, l] = np.kron(result[:, l], mat[:, l])
        result = new_result
    return result


# ============================================================
# Naive solver (for verification)
# ============================================================

def build_system_naive(prob):
    """
    Build the full system matrix and RHS explicitly.
    Only feasible for small problems.

    A = (Z ⊗ K)^T S S^T (Z ⊗ K) + λ (I_r ⊗ K)
    b = (I_r ⊗ K) vec(B)
    """
    K = prob['K']
    Z = prob['Z']
    lam = prob['lam']
    B = prob['B']
    n = prob['n']
    r = prob['r']
    N = prob['N']
    q = prob['q']
    obs_rows = prob['obs_rows']
    obs_cols = prob['obs_cols']

    # Build S: N x q selection matrix
    S = np.zeros((N, q))
    for s in range(q):
        lin_idx = obs_cols[s] * n + obs_rows[s]
        S[lin_idx, s] = 1.0

    # Z ⊗ K
    ZK = kron(Z, K)  # N x nr

    # System matrix
    SZ = ZK.T @ S  # nr x q
    A_mat = SZ @ SZ.T + lam * kron(np.eye(r), K)

    # RHS
    b = kron(np.eye(r), K) @ B.ravel(order='F')

    return A_mat, b


def solve_naive(prob):
    """Solve using explicit matrix construction and dense linear solve."""
    A_mat, b = build_system_naive(prob)
    w = np.linalg.solve(A_mat, b)
    W = w.reshape((prob['n'], prob['r']), order='F')
    return W


# ============================================================
# Efficient matrix-vector product
# ============================================================

def matvec_efficient(w_vec, prob):
    """
    Compute A @ vec(W) efficiently without forming the full system matrix.

    A = (Z ⊗ K)^T S S^T (Z ⊗ K) + λ (I_r ⊗ K)

    Uses the identity: (C ⊗ D) vec(X) = vec(D X C^T)

    Steps:
    1. Reshape w to W (n x r)
    2. Compute V = KW  (n x r),  cost O(n^2 r)
    3. For each observed entry (i_s, j_s):
       y_s = V[i_s, :] . Z_obs[s, :]     cost O(qr)
    4. Scatter: G[i, :] += y_s * Z_obs[s, :]  for entries with row i
       cost O(qr)
    5. Compute KG  (n x r),  cost O(n^2 r)
    6. Add regularization: result = vec(KG + λ V)

    Total cost: O(n^2 r + qr)
    """
    n = prob['n']
    r = prob['r']
    K = prob['K']
    lam = prob['lam']
    Z_obs = prob['Z_obs']
    obs_rows = prob['obs_rows']
    q = prob['q']

    W = w_vec.reshape((n, r), order='F')

    # Step 1: V = KW
    V = K @ W  # n x r

    # Step 2: Evaluate at observed positions
    # y_s = sum_l V[i_s, l] * Z_obs[s, l]
    y = np.sum(V[obs_rows, :] * Z_obs, axis=1)  # q-vector

    # Step 3: Scatter-multiply to form G = Y Z (sparse)
    # G[i, l] = sum_{s: obs_rows[s]=i} y_s * Z_obs[s, l]
    G = np.zeros((n, r))
    np.add.at(G, obs_rows, y[:, None] * Z_obs)

    # Step 4: Apply K to G
    KG = K @ G  # n x r

    # Step 5: Add regularization
    result = KG + lam * V

    return result.ravel(order='F')


# ============================================================
# Preconditioner
# ============================================================

class FullDataPreconditioner:
    """
    Preconditioner based on the full-data approximation:

    P = Γ ⊗ K² + λ I_r ⊗ K

    where Γ = Z^T Z (computed via Hadamard product of A_i^T A_i).

    Using eigendecompositions K = U Λ U^T and Γ = V Σ V^T:
    P = (I_r ⊗ U)(V ⊗ I_n)[Σ ⊗ Λ² + λ I_r ⊗ Λ](V^T ⊗ I_n)(I_r ⊗ U^T)

    The middle factor is diagonal with entries σ_p λ_i² + λ λ_i.

    Cost of setup: O(n³ + r³ + r² Σ n_i)
    Cost per application: O(n² r + n r²)
    """

    def __init__(self, prob):
        K = prob['K']
        n = prob['n']
        r = prob['r']
        lam = prob['lam']
        factors = prob['factors']
        k = prob['k']

        # Compute Γ = Z^T Z via Hadamard product identity
        # (A ⊙ B)^T (A ⊙ B) = (A^T A) * (B^T B)
        Gamma = np.ones((r, r))
        for i in range(len(factors)):
            if i != k:
                AiTAi = factors[i].T @ factors[i]  # r x r
                Gamma *= AiTAi  # elementwise

        # Eigendecomposition of K
        eigvals_K, U = eigh(K)
        self.U = U  # n x n
        self.eigvals_K = eigvals_K  # n-vector

        # Eigendecomposition of Gamma
        eigvals_G, V = eigh(Gamma)
        self.V = V  # r x r
        self.eigvals_G = eigvals_G  # r-vector

        # Precompute diagonal: d[p, i] = sigma_p * lambda_i^2 + lambda * lambda_i
        # Shape: (n, r) -- entry (i, p)
        self.diag = np.outer(eigvals_K**2, eigvals_G) + lam * eigvals_K[:, None]
        # Handle zero eigenvalues of K (set to 1 to avoid division by zero;
        # the corresponding directions are in the null space)
        self.diag[self.diag < 1e-14] = 1.0

        self.n = n
        self.r = r

    def apply(self, x_vec):
        """Apply P^{-1} to a vector."""
        n, r = self.n, self.r
        X = x_vec.reshape((n, r), order='F')

        # Step 1: Y1 = U^T X
        Y1 = self.U.T @ X  # n x r

        # Step 2: Y2 = Y1 V
        Y2 = Y1 @ self.V  # n x r

        # Step 3: Scale
        Y2 /= self.diag  # elementwise

        # Step 4: Y3 = Y2 V^T
        Y3 = Y2 @ self.V.T  # n x r

        # Step 5: result = U Y3
        result = self.U @ Y3  # n x r

        return result.ravel(order='F')


# ============================================================
# PCG solver
# ============================================================

def pcg_solve(prob, precond=None, tol=1e-10, maxiter=500, verbose=False):
    """
    Solve A x = b using Preconditioned Conjugate Gradient.

    Parameters
    ----------
    prob : dict
        Problem data.
    precond : object with .apply(x) method, or None for no preconditioning.
    tol : float
        Relative residual tolerance.
    maxiter : int
        Maximum number of iterations.
    verbose : bool
        Print convergence info.

    Returns
    -------
    W : ndarray (n x r)
        Solution reshaped as matrix.
    info : dict
        Convergence information.
    """
    n = prob['n']
    r = prob['r']
    K = prob['K']
    B = prob['B']

    # RHS: b = (I_r ⊗ K) vec(B) = vec(KB)
    b = (K @ B).ravel(order='F')
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        return np.zeros((n, r)), {'iterations': 0, 'residuals': [0.0]}

    # Initial guess
    x = np.zeros(n * r)

    # Initial residual
    r_vec = b - matvec_efficient(x, prob)
    if precond is not None:
        z = precond.apply(r_vec)
    else:
        z = r_vec.copy()

    p = z.copy()
    rz = np.dot(r_vec, z)

    residuals = [np.linalg.norm(r_vec) / b_norm]

    for it in range(maxiter):
        # Matrix-vector product
        Ap = matvec_efficient(p, prob)

        # Step size
        pAp = np.dot(p, Ap)
        if pAp <= 0:
            if verbose:
                print(f"  PCG: non-positive pAp = {pAp} at iteration {it}")
            break
        alpha = rz / pAp

        # Update solution and residual
        x += alpha * p
        r_vec -= alpha * Ap

        rel_res = np.linalg.norm(r_vec) / b_norm
        residuals.append(rel_res)

        if verbose and (it % 10 == 0 or rel_res < tol):
            print(f"  PCG iter {it+1}: rel_residual = {rel_res:.2e}")

        if rel_res < tol:
            break

        # Preconditioner
        if precond is not None:
            z = precond.apply(r_vec)
        else:
            z = r_vec.copy()

        rz_new = np.dot(r_vec, z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    W = x.reshape((n, r), order='F')
    info = {
        'iterations': len(residuals) - 1,
        'residuals': residuals,
        'final_residual': residuals[-1],
    }
    return W, info


# ============================================================
# Tests
# ============================================================

def test_matvec():
    """Verify efficient mat-vec matches naive construction."""
    print("=" * 60)
    print("TEST: Efficient mat-vec vs naive")
    print("=" * 60)

    prob = generate_test_problem(n=8, r=3, d=3, frac_observed=0.3)

    A_naive, b_naive = build_system_naive(prob)

    # Test with several random vectors
    rng = np.random.default_rng(123)
    max_err = 0.0
    for trial in range(5):
        w = rng.standard_normal(prob['n'] * prob['r'])
        y_naive = A_naive @ w
        y_eff = matvec_efficient(w, prob)
        err = np.linalg.norm(y_naive - y_eff) / np.linalg.norm(y_naive)
        max_err = max(max_err, err)
        print(f"  Trial {trial+1}: relative error = {err:.2e}")

    assert max_err < 1e-12, f"Mat-vec mismatch: {max_err}"
    print(f"  PASS (max relative error = {max_err:.2e})\n")


def test_rhs():
    """Verify RHS computation."""
    print("=" * 60)
    print("TEST: RHS computation")
    print("=" * 60)

    prob = generate_test_problem(n=8, r=3, d=3, frac_observed=0.3)
    A_naive, b_naive = build_system_naive(prob)

    K = prob['K']
    B = prob['B']
    b_eff = (K @ B).ravel(order='F')

    err = np.linalg.norm(b_naive - b_eff) / np.linalg.norm(b_naive)
    print(f"  Relative error = {err:.2e}")
    assert err < 1e-12, f"RHS mismatch: {err}"
    print("  PASS\n")


def test_preconditioner():
    """Verify preconditioner: P should equal the full-data system matrix."""
    print("=" * 60)
    print("TEST: Preconditioner structure")
    print("=" * 60)

    prob = generate_test_problem(n=8, r=3, d=3, frac_observed=1.0)
    # With all data observed, SS^T = I, so the system matrix should equal P.

    A_naive, _ = build_system_naive(prob)

    K = prob['K']
    Z = prob['Z']
    r = prob['r']
    lam = prob['lam']

    # P = Z^T Z ⊗ K^2 + λ I_r ⊗ K
    Gamma = Z.T @ Z
    P_full = kron(Gamma, K @ K) + lam * kron(np.eye(r), K)

    err = np.linalg.norm(A_naive - P_full) / np.linalg.norm(A_naive)
    print(f"  ||A_full - P|| / ||A_full|| = {err:.2e}")
    assert err < 1e-10, f"Preconditioner structure mismatch: {err}"

    # Now verify preconditioner inverse application
    precond = FullDataPreconditioner(prob)
    rng = np.random.default_rng(456)
    max_err = 0.0
    for trial in range(5):
        x = rng.standard_normal(prob['n'] * prob['r'])
        Px = P_full @ precond.apply(x)
        err = np.linalg.norm(Px - x) / np.linalg.norm(x)
        max_err = max(max_err, err)

    print(f"  P * P^{-1} * x error: {max_err:.2e}")
    assert max_err < 1e-10, f"Preconditioner inverse mismatch: {max_err}"
    print("  PASS\n")


def test_gamma_hadamard():
    """Verify that Γ = Z^T Z can be computed via Hadamard product."""
    print("=" * 60)
    print("TEST: Γ via Hadamard product identity")
    print("=" * 60)

    prob = generate_test_problem(n=6, r=4, d=4, frac_observed=0.2)
    Z = prob['Z']
    k = prob['k']
    factors = prob['factors']
    r = prob['r']

    # Direct computation
    Gamma_direct = Z.T @ Z

    # Via Hadamard product
    Gamma_hadamard = np.ones((r, r))
    for i in range(len(factors)):
        if i != k:
            Gamma_hadamard *= (factors[i].T @ factors[i])

    err = np.linalg.norm(Gamma_direct - Gamma_hadamard) / np.linalg.norm(Gamma_direct)
    print(f"  Relative error = {err:.2e}")
    assert err < 1e-12, f"Gamma mismatch: {err}"
    print("  PASS\n")


def test_solver_naive():
    """Verify naive solver against numpy.linalg.solve."""
    print("=" * 60)
    print("TEST: Naive solver")
    print("=" * 60)

    prob = generate_test_problem(n=8, r=3, d=3, frac_observed=0.3)
    W = solve_naive(prob)
    A, b = build_system_naive(prob)
    residual = np.linalg.norm(A @ W.ravel(order='F') - b) / np.linalg.norm(b)
    print(f"  Residual = {residual:.2e}")
    assert residual < 1e-10
    print("  PASS\n")


def test_pcg_no_precond():
    """Test PCG without preconditioner."""
    print("=" * 60)
    print("TEST: PCG without preconditioner")
    print("=" * 60)

    prob = generate_test_problem(n=8, r=3, d=3, frac_observed=0.3)
    W_naive = solve_naive(prob)
    W_pcg, info = pcg_solve(prob, precond=None, tol=1e-12, maxiter=200, verbose=True)

    err = np.linalg.norm(W_pcg - W_naive) / np.linalg.norm(W_naive)
    print(f"  ||W_pcg - W_naive|| / ||W_naive|| = {err:.2e}")
    print(f"  Iterations: {info['iterations']}")
    assert err < 1e-8, f"PCG solution mismatch: {err}"
    print("  PASS\n")


def test_pcg_with_precond():
    """Test PCG with full-data preconditioner."""
    print("=" * 60)
    print("TEST: PCG with full-data preconditioner")
    print("=" * 60)

    prob = generate_test_problem(n=8, r=3, d=3, frac_observed=0.3)
    W_naive = solve_naive(prob)

    precond = FullDataPreconditioner(prob)
    W_pcg, info = pcg_solve(prob, precond=precond, tol=1e-12, maxiter=200, verbose=True)

    err = np.linalg.norm(W_pcg - W_naive) / np.linalg.norm(W_naive)
    print(f"  ||W_pcg - W_naive|| / ||W_naive|| = {err:.2e}")
    print(f"  Iterations: {info['iterations']}")
    assert err < 1e-8, f"PCG solution mismatch: {err}"
    print("  PASS\n")


def test_preconditioner_speedup():
    """Compare convergence with and without preconditioner."""
    print("=" * 60)
    print("TEST: Preconditioner convergence comparison")
    print("=" * 60)

    for frac in [0.1, 0.3, 0.5, 0.8]:
        prob = generate_test_problem(n=15, r=5, d=3, frac_observed=frac)

        _, info_no = pcg_solve(prob, precond=None, tol=1e-10, maxiter=500)
        precond = FullDataPreconditioner(prob)
        _, info_yes = pcg_solve(prob, precond=precond, tol=1e-10, maxiter=500)

        print(f"  frac={frac:.1f}: no precond = {info_no['iterations']} iters, "
              f"with precond = {info_yes['iterations']} iters, "
              f"speedup = {info_no['iterations'] / max(info_yes['iterations'], 1):.1f}x")

    print("  PASS\n")


def test_scaling():
    """Test that the method scales correctly."""
    print("=" * 60)
    print("TEST: Scaling test")
    print("=" * 60)

    # Test with larger n but same fraction observed
    for n in [10, 20, 40]:
        # Use mode_sizes to keep d=3 with varying sizes
        mode_sizes = [n, n+5, n+3]
        prob = generate_test_problem(n=n, r=4, d=3, mode_sizes=mode_sizes,
                                     frac_observed=0.05, seed=42)

        # Only test efficient mat-vec (naive too expensive for large n)
        w = np.random.default_rng(42).standard_normal(n * 4)

        t0 = time.time()
        for _ in range(10):
            matvec_efficient(w, prob)
        t_mv = (time.time() - t0) / 10

        precond = FullDataPreconditioner(prob)
        t0 = time.time()
        for _ in range(10):
            precond.apply(w)
        t_pc = (time.time() - t0) / 10

        print(f"  n={n}, M={prob['M']}, N={prob['N']}, q={prob['q']}: "
              f"matvec={t_mv*1000:.2f}ms, precond={t_pc*1000:.2f}ms")

    print("  PASS\n")


def run_all_tests():
    """Run all tests."""
    test_gamma_hadamard()
    test_matvec()
    test_rhs()
    test_preconditioner()
    test_solver_naive()
    test_pcg_no_precond()
    test_pcg_with_precond()
    test_preconditioner_speedup()
    test_scaling()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
