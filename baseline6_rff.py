# baseline6_rff.py
import os, sys, time
import numpy as np
from tqdm import tqdm
from utils_hd import (f_true_vdp, f_true_duffing6d,
                      f_true_with_shift, generate_reference,
                      vanderpol, coupled_duffing, rk4_step)

def rff_features(x, t, omega, b, n_features):
    inp = np.append(x, t)
    return np.sqrt(2.0 / n_features) * np.cos(omega @ inp + b)

def run_rff(n=2, T=30, dt=0.001,
            use_shift=False, shift_time=15.0,
            n_features=54,
            gamma=1.0):
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

    W_out = np.zeros((n, n_features))
    lam = 0.99
    P = [np.eye(n_features) / 0.01 for _ in range(n)]

    x = np.array(x0, dtype=np.float64)
    f_hat = np.zeros(n)

    id_history, t_history, ct_history = [], [], []

    for i in tqdm(range(N-1), desc=f"RFF n={n}"):
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
        x_new = rk4_step(
            lambda s, tt: f_true_with_shift(s, tt, n, shift_time)
            if use_shift else f_true_fn(s, tt),
            x, u, t, dt
        )

        phi = rff_features(x, t, omega, b, n_features)

        t0 = time.perf_counter()
        for d in range(n):
            Pp = P[d] @ phi
            K = Pp / (lam + phi @ Pp)
            P[d] = (P[d] - np.outer(K, Pp)) / lam
            error_d = f_actual[d] - W_out[d] @ phi
            W_out[d] += K * error_d
        ct_history.append(time.perf_counter() - t0)

        f_hat = W_out @ phi
        x = x_new

    return (np.array(t_history), np.array(id_history), np.array(ct_history))

if __name__ == "__main__":
    for n, label in [(2, 'n2'), (6, 'n6')]:
        for use_shift, suf in [(False, 'normal'), (True, 'shift')]:
            print(f"\nRunning RFF n={n} {'(shift)' if use_shift else ''}")
            t, e, ct = run_rff(n=n, use_shift=use_shift)
            base = f"results/sysid/{label}/{suf}"
            os.makedirs(base, exist_ok=True)
            np.save(f"{base}/rff_id_error.npy", e)
            np.save(f"{base}/rff_time.npy", t)
            np.save(f"{base}/rff_comptime.npy", ct)
            print(f"  Mean SS id error: {e[15000:].mean():.6f} | Mean step: {ct.mean()*1000:.3f} ms")
