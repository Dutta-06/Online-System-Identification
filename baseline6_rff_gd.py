# baseline6_rff_gd.py
# Fair comparison baseline: RFF with gradient descent readout update (same O(n_f) complexity as proposed)
# This isolates the single variable of interest: adaptive geometry vs fixed geometry.
# Both this baseline and the proposed method use first-order gradient descent on e_id = f_actual - f_hat.

import os, sys, time
import numpy as np
from tqdm import tqdm
from utils_hd import (f_true_vdp, f_true_duffing6d,
                      f_true_with_shift, generate_reference,
                      vanderpol, coupled_duffing, rk4_step)

def rff_features(x, t, omega, b, n_features):
    inp = np.append(x, t)
    return np.sqrt(2.0 / n_features) * np.cos(omega @ inp + b)

def run_rff_gd(n=2, T=30, dt=0.001,
               use_shift=False, shift_time=15.0,
               n_features=54, gamma=1.0, lr=500.0):
    """
    RFF baseline with gradient descent readout update.

    Update rule: W_out += lr * e_id @ phi^T  (same structure as proposed W_dot = gamma_W * phi * e_id^T)

    Parameters
    ----------
    n_features : int
        Number of random Fourier features.
    lr : float
        Gradient descent learning rate (matched to gamma_W of proposed for fair comparison).
    """
    if n == 2:
        plant_fn = vanderpol
        x0 = [2.0, 0.0]
        f_true_fn = f_true_vdp
    else:
        plant_fn = coupled_duffing
        x0 = [1.0, 0.0, -1.0, 0.5, 0.5, -0.5]
        f_true_fn = f_true_duffing6d
        n_features = 125 * n

    t_eval, xm = generate_reference(plant_fn, x0, T, dt)
    N = t_eval.shape[0]

    np.random.seed(42)
    omega = np.random.normal(0, gamma, (n_features, n + 1))
    b = np.random.uniform(0, 2 * np.pi, n_features)

    # Readout weights: shape (n_features, n) — same convention as proposed W: (m, n)
    W_out = np.zeros((n_features, n))

    x = np.array(x0, dtype=np.float64)
    f_hat = np.zeros(n)

    id_history, t_history, ct_history = [], [], []

    for i in tqdm(range(N-1), desc=f"RFF-GD n={n}"):
        t = t_eval[i]

        if use_shift:
            f_actual = f_true_with_shift(x, t, n, shift_time)
        else:
            f_actual = f_true_fn(x, t)

        id_err = np.linalg.norm(f_actual - f_hat)
        id_history.append(id_err)
        t_history.append(t)

        x_ref = xm[:, i]
        u = -5.0 * (x - x_ref)

        phi = rff_features(x, t, omega, b, n_features)   # (n_features,)
        e_id = f_actual - f_hat                            # (n,)

        # Gradient descent: identical structure to proposed W_dot = gamma_W * outer(phi, e_id)
        # W_out shape: (n_features, n)
        t0 = time.perf_counter()
        W_out += dt * lr * np.outer(phi, e_id)
        ct_history.append(time.perf_counter() - t0)

        f_hat = W_out.T @ phi   # (n,)

        x = rk4_step(
            lambda s, tt: f_true_with_shift(s, tt, n, shift_time)
            if use_shift else f_true_fn(s, tt),
            x, u, t, dt
        )

    return (np.array(t_history), np.array(id_history), np.array(ct_history))


if __name__ == "__main__":
    for n, label in [(2, 'n2'), (6, 'n6')]:
        for use_shift, suf in [(False, 'normal'), (True, 'shift')]:
            print(f"\nRunning RFF-GD n={n} {'(shift)' if use_shift else ''}")
            t, e, ct = run_rff_gd(n=n, use_shift=use_shift)
            base = f"results/sysid/{label}/{suf}"
            os.makedirs(base, exist_ok=True)
            np.save(f"{base}/rff_gd_id_error.npy", e)
            np.save(f"{base}/rff_gd_time.npy", t)
            np.save(f"{base}/rff_gd_comptime.npy", ct)
            print(f"  Mean SS id error: {e[15000:].mean():.6f} | Mean step: {ct.mean()*1000:.3f} ms")
