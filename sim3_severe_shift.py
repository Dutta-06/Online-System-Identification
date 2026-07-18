import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.controller import AdaptiveController
from utils_hd import f_true_vdp, generate_reference, vanderpol, rk4_step
from baseline6_rff import rff_features

def f_true_severe_shift(x, t, shift_time=15.0, factor=10.0):
    if t < shift_time:
        return f_true_vdp(x, t)
    else:
        mu = 1.5 * factor
        x1, x2 = x[0], x[1]
        return np.array([x2, mu*(1 - x1**2)*x2 - x1])

def run_proposed_severe(T=30, dt=0.001, shift_time=15.0):
    n = 2
    plant_fn = vanderpol
    x0 = [2.0, 0.0]
    m = 27
    ke = 5.0
    
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
        sigma_min=0.1, sigma_max=5.0, delta_min=0.05, k_cl=5.0)
    
    t_eval, xm = generate_reference(plant_fn, x0, T, dt)
    N = t_eval.shape[0]
    
    x = np.array(x0, dtype=np.float64)
    W = np.zeros((m, n))
    
    id_history = []
    
    for i in tqdm(range(N-1), desc="Proposed (10x Shift)"):
        t = t_eval[i]
        f_actual = f_true_severe_shift(x, t, shift_time, factor=10.0)
        
        x_ref = xm[:, i]
        xm_dot = (xm[:, i+1] - xm[:, i]) / dt if i < N-2 else np.zeros(n)
        e = x - x_ref
        
        u_vec = ctrl.control_law(x, x_ref, xm_dot, W, c, sigma)
        f_hat = ctrl.kernel.f_hat(x, W, c, sigma)
        
        # Non-oracle state-derivative estimation for CL adaptation
        if i == 0:
            x_dot_hat = np.zeros(n)
        else:
            x_dot_hat = (x - x_prev) / dt

        f_known = np.zeros(n)
        u_applied = u_prev if i > 0 else u_vec

        if i == 0:
            e_id = np.zeros(n)
        else:
            e_id = x_dot_hat - f_known - u_applied - f_hat_prev

        x_prev = x.copy()
        u_prev = u_vec.copy()
        f_hat_prev = f_hat.copy()
        
        id_history.append(np.linalg.norm(f_actual - f_hat))
        
        W_dot, c_dot, sigma_dot = ctrl.adaptation_laws(x, e, W, c, sigma, e_id=e_id)
        W = W + dt * W_dot
        c = c + dt * c_dot.reshape(m, n)
        sigma = sigma + dt * sigma_dot
        
        W = ctrl.proj_W.hard_clip_matrix(W)
        c = ctrl.proj_c.hard_clip(c.flatten(), n).reshape(m, n)
        sigma = ctrl.proj_sigma.hard_clip(sigma)
        
        x = rk4_step(lambda s, tt: f_true_severe_shift(s, tt, shift_time, 10.0), x, u_vec, t, dt)
        
    return np.array(id_history)

def run_rff_severe(T=30, dt=0.001, shift_time=15.0):
    n = 2
    plant_fn = vanderpol
    x0 = [2.0, 0.0]
    n_features = 54
    gamma_rff = 1.0
    
    t_eval, xm = generate_reference(plant_fn, x0, T, dt)
    N = t_eval.shape[0]
    
    np.random.seed(42)
    omega = np.random.normal(0, gamma_rff, (n_features, n + 1))
    b = np.random.uniform(0, 2 * np.pi, n_features)
    
    W_out = np.zeros((n, n_features))
    lam = 0.99
    P = [np.eye(n_features) / 0.01 for _ in range(n)]
    
    x = np.array(x0, dtype=np.float64)
    f_hat = np.zeros(n)
    
    id_history = []
    
    for i in tqdm(range(N-1), desc="RFF (10x Shift)"):
        t = t_eval[i]
        f_actual = f_true_severe_shift(x, t, shift_time, factor=10.0)
        
        id_err = np.linalg.norm(f_actual - f_hat)
        id_history.append(id_err)
        
        x_ref = xm[:, i]
        u = -5.0 * (x - x_ref)
        
        phi = rff_features(x, t, omega, b, n_features)
        
        for d in range(n):
            Pp = P[d] @ phi
            K = Pp / (lam + phi @ Pp)
            P[d] = (P[d] - np.outer(K, Pp)) / lam
            error_d = f_actual[d] - W_out[d] @ phi
            W_out[d] += K * error_d
            
        f_hat = W_out @ phi
        x = rk4_step(lambda s, tt: f_true_severe_shift(s, tt, shift_time, 10.0), x, u, t, dt)
        
    return np.array(id_history)

if __name__ == "__main__":
    os.makedirs('results/sysid', exist_ok=True)
    
    print("Running Proposed Method on 10x Severe Shift...")
    e_proposed = run_proposed_severe()
    
    print("\nRunning RFF on 10x Severe Shift...")
    e_rff = run_rff_severe()
    
    # Calculate IAE
    iae_proposed = np.sum(e_proposed[15000:]) * 0.001
    iae_rff = np.sum(e_rff[15000:]) * 0.001
    
    print(f"\nFinal IAE Post-Shift (10x Shift):")
    print(f"Proposed: {iae_proposed:.4f}")
    print(f"RFF:      {iae_rff:.4f}")
    
    # Smooth and plot
    window = 500
    ep_s = np.convolve(e_proposed, np.ones(window)/window, mode='valid')
    er_s = np.convolve(e_rff, np.ones(window)/window, mode='valid')
    t_s = np.arange(len(ep_s)) * 0.001
    
    plt.figure(figsize=(10, 6))
    plt.title('Severe Distribution Shift (10x): Proposed Adaptive Geometry vs RFF Fixed Geometry', fontsize=12)
    plt.semilogy(t_s, er_s, label=f'RFF (Fixed Geometry) IAE={iae_rff:.2f}', color='#795548', linewidth=1.5)
    plt.semilogy(t_s, ep_s, label=f'Proposed (Moving Centers) IAE={iae_proposed:.2f}', color='#4CAF50', linewidth=2.0)
    plt.axvline(x=15.0, color='black', linestyle='--', alpha=0.5, label='Severe 10x Shift')
    
    plt.xlabel('Time (s)')
    plt.ylabel('ID Error ||f - f̂|| (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/sysid/fig6_severe_shift.png', dpi=200, bbox_inches='tight')
    print("Saved plot to results/sysid/fig6_severe_shift.png")
