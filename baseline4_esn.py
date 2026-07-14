# baseline4_esn.py
import os, sys, time
import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from tqdm import tqdm
# rpy.verbosity(0) removed for reservoirpy 0.4.0+
from utils_hd import (f_true_vdp, f_true_duffing6d,
                      f_true_with_shift, generate_reference,
                      vanderpol, coupled_duffing, rk4_step)

def run_esn(n=2, T=30, dt=0.001,
            use_shift=False, shift_time=15.0,
            reservoir_size=200, spectral_radius=0.9,
            update_every=50):
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

    reservoir = Reservoir(
        reservoir_size,
        sr=spectral_radius,
        lr=0.3,
        input_scaling=1.0,
        seed=42
    )

    readout_dim = reservoir_size + n + 1
    W_out = np.zeros((n, readout_dim))
    A_ls = [np.eye(readout_dim) * 1e-4 for _ in range(n)]
    b_ls = [np.zeros(readout_dim) for _ in range(n)]

    x = np.array(x0, dtype=np.float64)
    f_hat = np.zeros(n)
    h = np.zeros(reservoir_size)

    id_history, t_history, ct_history = [], [], []

    for i in tqdm(range(N-1), desc=f"ESN n={n}"):
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

        inp = np.concatenate([x, u])[:n]
        h = reservoir(inp).reshape(-1)

        phi_r = np.concatenate([h, x, [t]])

        if i % update_every == 0:
            t0 = time.perf_counter()
            for d in range(n):
                A_ls[d] += np.outer(phi_r, phi_r)
                b_ls[d] += phi_r * f_actual[d]
                try:
                    W_out[d] = np.linalg.solve(A_ls[d], b_ls[d])
                except np.linalg.LinAlgError:
                    pass
            ct_history.append(time.perf_counter() - t0)

        f_hat = W_out @ phi_r
        x = x_new

    return (np.array(t_history), np.array(id_history), np.array(ct_history))

if __name__ == "__main__":
    for n, label in [(2, 'n2'), (6, 'n6')]:
        for use_shift, suf in [(False, 'normal'), (True, 'shift')]:
            print(f"\nRunning ESN n={n} {'(shift)' if use_shift else ''}")
            t, e, ct = run_esn(n=n, use_shift=use_shift)
            base = f"results/sysid/{label}/{suf}"
            os.makedirs(base, exist_ok=True)
            np.save(f"{base}/esn_id_error.npy", e)
            np.save(f"{base}/esn_time.npy", t)
            np.save(f"{base}/esn_comptime.npy", ct)
            print(f"  Mean SS id error: {e[15000:].mean():.6f} | Mean step: {ct.mean()*1000:.3f} ms")
