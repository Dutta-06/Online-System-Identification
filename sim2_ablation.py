import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.controller import AdaptiveController
from utils_hd import f_true_vdp, f_true_with_shift, generate_reference, vanderpol, rk4_step

def run_ablation(gamma_c=100.0, gamma_sigma=50.0):
    n = 2
    T = 30.0
    dt = 0.001
    shift_time = 15.0
    
    plant_fn = vanderpol
    x0 = [2.0, 0.0]
    m = 27
    ke = 5.0
    
    # Initialize exactly the same for all ablations
    g = int(np.ceil(np.sqrt(m)))
    cx = np.linspace(-3, 3, g)
    cy = np.linspace(-3, 3, g)
    gx, gy = np.meshgrid(cx, cy)
    c = np.column_stack([gx.ravel()[:m], gy.ravel()[:m]])
    sigma = np.ones(m) * 1.5
    W = np.zeros((m, n))
    
    ctrl = AdaptiveController(
        m=m, n=n, ke=ke,
        gamma_W=500.0, gamma_c=gamma_c, gamma_sigma=gamma_sigma,
        W_max=50.0, c_max=5.0,
        sigma_min=0.1, sigma_max=5.0, delta_min=0.05, k_cl=5.0)
    
    t_eval, xm = generate_reference(plant_fn, x0, T, dt)
    N = t_eval.shape[0]
    x = np.array(x0, dtype=np.float64)
    
    id_history = []
    
    for i in tqdm(range(N-1), desc=f"Ablation gc={gamma_c}, gs={gamma_sigma}"):
        t = t_eval[i]
        f_actual = f_true_with_shift(x, t, n, shift_time)
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
        
        x = rk4_step(lambda s, tt: f_true_with_shift(s, tt, n, shift_time), x, u_vec, t, dt)
        
    return np.array(id_history)

if __name__ == "__main__":
    os.makedirs('results/sysid', exist_ok=True)
    
    print("Running Full Proposed Architecture...")
    e_full = run_ablation(gamma_c=100.0, gamma_sigma=50.0)
    
    print("Running Partial Ablation (No adaptive sigma)...")
    e_partial = run_ablation(gamma_c=100.0, gamma_sigma=0.0)
    
    print("Running Full Ablation (Static Grid, W only)...")
    e_static = run_ablation(gamma_c=0.0, gamma_sigma=0.0)
    
    # Smooth for plotting
    window = 500
    e_full_s = np.convolve(e_full, np.ones(window)/window, mode='valid')
    e_partial_s = np.convolve(e_partial, np.ones(window)/window, mode='valid')
    e_static_s = np.convolve(e_static, np.ones(window)/window, mode='valid')
    t_s = np.arange(len(e_full_s)) * 0.001
    
    plt.figure(figsize=(10, 6))
    plt.title('Ablation Study: Architecture Components vs. Identification Error (n=2)', fontsize=13)
    
    plt.semilogy(t_s, e_static_s, label='Full Ablation (Fixed Centers, Fixed Bandwidths)', color='#E91E63', linewidth=1.5)
    plt.semilogy(t_s, e_partial_s, label='Partial Ablation (Moving Centers, Fixed Bandwidths)', color='#FF9800', linewidth=1.5)
    plt.semilogy(t_s, e_full_s, label='Proposed Method (Fully Adaptive W, c, σ)', color='#4CAF50', linewidth=2.0)
    
    plt.axvline(x=15.0, color='black', linestyle='--', alpha=0.5, linewidth=1.2, label='Parameter Shift (t=15s)')
    plt.xlabel('Time (s)')
    plt.ylabel('ID Error ||f - f̂|| (Log Scale)')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/sysid/fig5_ablation_study.png', dpi=200, bbox_inches='tight')
    print("Plot saved as results/sysid/fig5_ablation_study.png")
    
    print("\nMean Steady-State Errors (t > 15s):")
    print(f"Full Ablation (Static): {e_static[15000:].mean():.4f}")
    print(f"Partial Ablation (No σ): {e_partial[15000:].mean():.4f}")
    print(f"Proposed Method:       {e_full[15000:].mean():.4f}")
