# sim0_proposed_sysid.py
import os, sys, time
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.controller import AdaptiveController

# Import from local AAAI/utils_hd.py
from utils_hd import (f_true_vdp, f_true_duffing6d,
                      f_true_with_shift, generate_reference,
                      vanderpol, coupled_duffing, rk4_step)

def run_proposed_sysid(n=2, T=30, dt=0.001, use_shift=False, shift_time=15.0):
    if n == 2:
        plant_fn = vanderpol
        x0 = [2.0, 0.0]
        f_true_fn = f_true_vdp
        m = 27
        ke = 5.0
        # Uniform grid over 2D state space for n=2
        g = int(np.ceil(np.sqrt(m)))
        cx = np.linspace(-3, 3, g)
        cy = np.linspace(-3, 3, g)
        gx, gy = np.meshgrid(cx, cy)
        c = np.column_stack([gx.ravel()[:m], gy.ravel()[:m]])
        sigma = np.ones(m) * 1.5
        ctrl = AdaptiveController(
            m=m, n=n, ke=ke,
            gamma_W=500.0, gamma_c=100.0, gamma_sigma=50.0,
            W_max=50.0, c_max=5.0,
            sigma_min=0.1, sigma_max=5.0, delta_min=0.05,
            k_cl=0.0   # Pure gradient descent on e_id for fair geometry comparison
        )
    else:
        plant_fn = coupled_duffing
        x0 = [1.0, 0.0, -1.0, 0.5, 0.5, -0.5]
        f_true_fn = f_true_duffing6d
        m = 125
        ke = 5.0
        np.random.seed(42)
        c = np.random.randn(m, n) * 2.0  # Random init for 6D
        sigma = np.ones(m) * 3.0
        ctrl = AdaptiveController(
            m=m, n=n, ke=ke,
            gamma_W=300.0, gamma_c=50.0, gamma_sigma=10.0,
            W_max=200.0, c_max=10.0,
            sigma_min=0.3, sigma_max=8.0, delta_min=0.1,
            k_cl=0.0   # Pure gradient descent on e_id for fair geometry comparison
        )

    t_eval, xm = generate_reference(plant_fn, x0, T, dt)
    N = t_eval.shape[0]

    x = np.array(x0, dtype=np.float64)
    W = np.zeros((m, n))

    id_history, t_history, ct_history = [], [], []

    for i in tqdm(range(N-1), desc=f"Proposed n={n}"):
        t = t_eval[i]

        if use_shift:
            f_actual = f_true_with_shift(x, t, n, shift_time)
        else:
            f_actual = f_true_fn(x, t)

        x_ref = xm[:, i]
        xm_dot = (xm[:, i+1] - xm[:, i]) / dt if i < N-2 else np.zeros(n)

        # Control Law
        u_vec = ctrl.control_law(x, x_ref, xm_dot, W, c, sigma)
        
        # Compute exact f_hat produced by the network
        f_hat = ctrl.kernel.f_hat(x, W, c, sigma)
        
        id_err = np.linalg.norm(f_actual - f_hat)
        id_history.append(id_err)
        t_history.append(t)

        # The adaptation laws are driven strictly by the identification error (e_id)
        # to ensure an apples-to-apples comparison with sysid baselines like SINDy and RFF.
        # This isolates the geometry adaptation mechanism from the closed-loop tracking setup.
        # We use practical state-derivative estimation instead of oracle f_actual:
        # e_id = hat{x_dot} - f_known - u - f_hat
        
        # 1. State derivative estimation (simple backward difference)
        if i == 0:
            x_dot_hat = np.zeros(n)
        else:
            x_dot_hat = (x - x_prev) / dt
            
        # 2. Known dynamics (A*x in our system is 0, so just 0)
        f_known = np.zeros(n)
        
        # 3. Previous control input (or current if i=0)
        u_applied = u_prev if i > 0 else u_vec
        
        # 4. Compute non-oracle e_id
        if i == 0:
             e_id = np.zeros(n) # Filter transient
        else:
             e_id = x_dot_hat - f_known - u_applied - f_hat_prev
             
        # Store for next timestep
        x_prev = x.copy()
        u_prev = u_vec.copy()
        f_hat_prev = f_hat.copy()

        t0 = time.perf_counter()
        # Pass e_id as the primary error argument to force pure gradient descent
        # on the identification error (parameter estimation mode).
        W_dot, c_dot, sigma_dot = ctrl.adaptation_laws(x, e_id, W, c, sigma)
        W     = W     + dt * W_dot
        c     = c     + dt * c_dot.reshape(m, n)
        sigma = sigma + dt * sigma_dot
        W     = ctrl.proj_W.hard_clip_matrix(W)
        c_flat = ctrl.proj_c.hard_clip(c.flatten(), n)
        c     = c_flat.reshape(m, n)
        sigma = ctrl.proj_sigma.hard_clip(sigma)
        ct_history.append(time.perf_counter() - t0)

        # Plant dynamics step
        x = rk4_step(
            lambda s, tt: f_true_with_shift(s, tt, n, shift_time) if use_shift else f_true_fn(s, tt),
            x, u_vec, t, dt
        )

    return np.array(t_history), np.array(id_history), np.array(ct_history)

if __name__ == "__main__":
    for n, label in [(2, 'n2'), (6, 'n6')]:
        for use_shift, suf in [(False, 'normal'), (True, 'shift')]:
            print(f"\nRunning Proposed Method n={n} {'(shift)' if use_shift else ''}")
            t, e, ct = run_proposed_sysid(n=n, use_shift=use_shift)
            base = f"results/sysid/{label}/{suf}"
            os.makedirs(base, exist_ok=True)
            np.save(f"{base}/proposed_id_error.npy", e)
            np.save(f"{base}/proposed_time.npy", t)
            np.save(f"{base}/proposed_comptime.npy", ct)
            print(f"  Mean SS id error: {e[15000:].mean():.6f} | Mean step: {ct.mean()*1000:.3f} ms")
