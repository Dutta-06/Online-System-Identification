# lag_rff.py
# RFF + Proposed Method (Lyapunov-Adaptive Geometry)
# 
# We apply the proposed geometry adaptation laws to the Random Fourier Features (RFF)
# baseline to create an adaptive RFF network that beats the fixed RFF.

import os, sys, time
import numpy as np
from tqdm import tqdm
from utils_hd import (f_true_vdp, f_true_duffing6d,
                      f_true_with_shift, generate_reference,
                      vanderpol, coupled_duffing, rk4_step)

def rff_features_and_derivs(x, t, omega, b, n_features):
    inp = np.append(x, t)
    phase = omega @ inp + b
    z = np.sqrt(2.0 / n_features) * np.cos(phase)
    dz_dphase = -np.sqrt(2.0 / n_features) * np.sin(phase)
    return z, dz_dphase, inp

def run_lag_rff(n=2, T=30, dt=0.001,
                use_shift=False, shift_time=15.0,
                n_features=54, lr=500.0,
                gamma_omega=0.1, gamma_b=0.1,
                omega_max=5.0):
    if n == 2:
        plant_fn = vanderpol
        x0 = [2.0, 0.0]
        f_true_fn = f_true_vdp
        gamma_omega = 0.01  # conservative gains for n=2
        gamma_b = 0.01
    else:
        plant_fn = coupled_duffing
        x0 = [1.0, 0.0, -1.0, 0.5, 0.5, -0.5]
        f_true_fn = f_true_duffing6d
        n_features = 125 * n
        gamma_omega = 0.1   # tuned gains for n=6
        gamma_b = 0.1

    t_eval, xm = generate_reference(plant_fn, x0, T, dt)
    N = t_eval.shape[0]

    np.random.seed(42)
    omega = np.random.normal(0, 1.0, (n_features, n + 1))
    b = np.random.uniform(0, 2 * np.pi, n_features)
    W_out = np.zeros((n_features, n))

    x = np.array(x0, dtype=np.float64)
    f_hat = np.zeros(n)

    id_history, t_history, ct_history = [], [], []

    tag = f"RFF+Proposed n={n}"
    for i in tqdm(range(N-1), desc=tag):
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
        
        # 1. State derivative estimation (simple backward difference)
        if i == 0:
            x_dot_hat = np.zeros(n)
        else:
            x_dot_hat = (x - x_prev) / dt
            
        # 2. Known dynamics (A*x in our system is 0, so just 0)
        f_known = np.zeros(n)
        
        # 3. Previous control input (or current if i=0)
        u_applied = u_prev if i > 0 else u
        
        # 4. Compute non-oracle e_id
        if i == 0:
             e_id = np.zeros(n) # Filter transient
        else:
             e_id = x_dot_hat - f_known - u_applied - f_hat_prev
             
        # Store for next timestep
        x_prev = x.copy()
        u_prev = u.copy()
        f_hat_prev = f_hat.copy()

        t0 = time.perf_counter()
        
        # 1. Forward pass
        z, dz_dphase, inp = rff_features_and_derivs(x, t, omega, b, n_features)
        
        # 2. Non-oracle error is already computed above as e_id

        # 3. Readout weights update
        W_out += dt * lr * np.outer(z, e_id)

        # 4. Geometry updates (Proposed method principles applied to RFF)
        We = W_out @ e_id  # (n_features,)
        
        # omega update
        raw_omega = gamma_omega * np.outer(dz_dphase * We, inp)
        omega += dt * raw_omega
        norms = np.linalg.norm(omega, axis=1, keepdims=True)
        scale = np.where(norms > omega_max, omega_max / (norms + 1e-12), 1.0)
        omega *= scale

        # b update
        raw_b = gamma_b * (dz_dphase * We)
        b += dt * raw_b
        b = b % (2 * np.pi)

        ct_history.append(time.perf_counter() - t0)

        # 5. Predict next f_hat for control/next step
        f_hat = W_out.T @ z

        # 6. Plant step
        x = rk4_step(
            lambda s, tt: f_true_with_shift(s, tt, n, shift_time)
            if use_shift else f_true_fn(s, tt),
            x, u, t, dt
        )

    return np.array(t_history), np.array(id_history), np.array(ct_history)

if __name__ == "__main__":
    for n, label in [(2, 'n2'), (6, 'n6')]:
        for use_shift, suf in [(False, 'normal'), (True, 'shift')]:
            print(f"\nRunning RFF+Proposed n={n} {'(shift)' if use_shift else ''}")
            t, e, ct = run_lag_rff(n=n, use_shift=use_shift)
            base = f"results/sysid/{label}/{suf}"
            os.makedirs(base, exist_ok=True)
            np.save(f"{base}/lag_rff_id_error.npy", e)
            np.save(f"{base}/lag_rff_time.npy", t)
            np.save(f"{base}/lag_rff_comptime.npy", ct)
            ss_mean = e[int(0.6*len(e)):].mean()
            print(f"  Mean SS id error: {ss_mean:.6f} | Mean step: {ct.mean()*1000:.3f} ms")
