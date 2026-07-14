# baseline2_sindy.py
import os, sys, time
import numpy as np
import pysindy as ps
from tqdm import tqdm
from utils_hd import (f_true_vdp, f_true_duffing6d,
                      f_true_with_shift, generate_reference,
                      vanderpol, coupled_duffing, rk4_step)

def run_sindy(n=2, T=30, dt=0.001,
              use_shift=False, shift_time=15.0,
              refit_every=500):
    if n == 2:
        plant_fn = vanderpol
        x0 = [2.0, 0.0]
        f_true_fn = f_true_vdp
        feature_library = ps.PolynomialLibrary(degree=3)
    else:
        plant_fn = coupled_duffing
        x0 = [1.0, 0.0, -1.0, 0.5, 0.5, -0.5]
        f_true_fn = f_true_duffing6d
        poly_lib = ps.PolynomialLibrary(degree=3)
        fourier_lib = ps.FourierLibrary(n_frequencies=2)
        feature_library = ps.GeneralizedLibrary([poly_lib, fourier_lib])

    t_eval, xm = generate_reference(plant_fn, x0, T, dt)
    N = t_eval.shape[0]

    x = np.array(x0, dtype=np.float64)
    f_hat = np.zeros(n)

    id_history, t_history, ct_history = [], [], []
    X_buf, Xdot_buf = [], []
    model = None

    for i in tqdm(range(N-1), desc=f"SINDy n={n}"):
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
        xdot_est = (x_new - x) / dt
        X_buf.append(x.tolist())
        Xdot_buf.append(xdot_est.tolist())
        x = x_new

        if i % refit_every == 0 and len(X_buf) > 50:
            t0 = time.perf_counter()
            window = min(2000, len(X_buf))
            X_arr = np.array(X_buf[-window:])
            Xdot_arr = np.array(Xdot_buf[-window:])
            try:
                model = ps.SINDy(
                    feature_library=feature_library,
                    optimizer=ps.STLSQ(threshold=0.05, alpha=0.1),
                    feature_names=[f'x{j}' for j in range(n)]
                )
                model.fit(X_arr, t=dt, x_dot=Xdot_arr, quiet=True)
                f_hat = model.predict(x[None])[0]
            except Exception:
                f_hat = np.zeros(n)
            ct_history.append(time.perf_counter() - t0)
        elif model is not None:
            try:
                f_hat = model.predict(x[None])[0]
            except Exception:
                f_hat = np.zeros(n)

    return (np.array(t_history), np.array(id_history), np.array(ct_history))

if __name__ == "__main__":
    for n, label in [(2, 'n2'), (6, 'n6')]:
        for use_shift, suf in [(False, 'normal'), (True, 'shift')]:
            print(f"\nRunning SINDy n={n} {'(shift)' if use_shift else ''}")
            t, e, ct = run_sindy(n=n, use_shift=use_shift)
            base = f"results/sysid/{label}/{suf}"
            os.makedirs(base, exist_ok=True)
            np.save(f"{base}/sindy_id_error.npy", e)
            np.save(f"{base}/sindy_time.npy", t)
            np.save(f"{base}/sindy_comptime.npy", ct)
            print(f"  Mean SS id error: {e[15000:].mean():.6f} | Mean step: {ct.mean()*1000:.3f} ms")
