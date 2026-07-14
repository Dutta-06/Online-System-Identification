# baseline3_koopman_dmd.py
import os, sys, time
import numpy as np
from tqdm import tqdm
from utils_hd import (f_true_vdp, f_true_duffing6d,
                      f_true_with_shift, generate_reference,
                      vanderpol, coupled_duffing, rk4_step)

def koopman_lift(x, n):
    features = list(x)
    for i in range(n):
        features.append(x[i]**2)
    for i in range(n):
        for j in range(i+1, n):
            features.append(x[i]*x[j])
    if n == 2:
        features.extend([np.sin(x[0]), np.cos(x[0]), np.sin(x[1]), np.cos(x[1])])
    return np.array(features)

def run_koopman_dmd(n=2, T=30, dt=0.001, use_shift=False, shift_time=15.0):
    if n == 2:
        plant_fn = vanderpol
        x0 = [2.0, 0.0]
        f_true_fn = f_true_vdp
    else:
        plant_fn = coupled_duffing
        x0 = [1.0, 0.0, -1.0, 0.5, 0.5, -0.5]
        f_true_fn = f_true_duffing6d

    t_eval, xm = generate_reference(plant_fn, x0, T, dt)
    N = t_eval.shape[0]
    x_test = np.array(x0)
    psi_test = koopman_lift(x_test, n)
    p = len(psi_test)
    x = np.array(x0, dtype=np.float64)
    f_hat = np.zeros(n)
    id_history, t_history, ct_history = [], [], []
    Psi_buf = []
    A_koopman = None

    for i in tqdm(range(N-1), desc=f"Koopman DMD n={n}"):
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
            lambda s, tt: f_true_with_shift(s, tt, n, shift_time) if use_shift else f_true_fn(s, tt),
            x, u, t, dt
        )
        psi = koopman_lift(x, n)
        Psi_buf.append(psi)
        x = x_new

        if i % 100 == 0 and len(Psi_buf) > p + 10:
            t0 = time.perf_counter()
            window = min(500, len(Psi_buf))
            Psi_arr = np.array(Psi_buf[-window:]).T
            X_dmd = Psi_arr[:, :-1]
            Y_dmd = Psi_arr[:, 1:]
            try:
                A_koopman = Y_dmd @ np.linalg.pinv(X_dmd)
            except np.linalg.LinAlgError:
                A_koopman = None
            ct_history.append(time.perf_counter() - t0)

        if A_koopman is not None:
            psi_current = koopman_lift(x, n)
            psi_next = A_koopman @ psi_current
            f_hat = (psi_next[:n] - x) / dt
        else:
            f_hat = np.zeros(n)

    return (np.array(t_history), np.array(id_history), np.array(ct_history))

if __name__ == "__main__":
    for n, label in [(2, 'n2'), (6, 'n6')]:
        for use_shift, suf in [(False, 'normal'), (True, 'shift')]:
            print(f"\nRunning Koopman DMD n={n} {'(shift)' if use_shift else ''}")
            t, e, ct = run_koopman_dmd(n=n, use_shift=use_shift)
            base = f"results/sysid/{label}/{suf}"
            os.makedirs(base, exist_ok=True)
            np.save(f"{base}/koopman_id_error.npy", e)
            np.save(f"{base}/koopman_time.npy", t)
            np.save(f"{base}/koopman_comptime.npy", ct)
            print(f"  Mean SS id error: {e[15000:].mean():.6f} | Mean step: {ct.mean()*1000:.3f} ms")
