import numpy as np
import matplotlib.pyplot as plt
import os, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.controller import AdaptiveController
from utils_hd import f_true_vdp, rk4_step

# Parameters
n = 2
T = 30.0
dt = 0.001
N = int(T/dt)
t_eval = np.linspace(0, T, N)

# Massive spiral reference trajectory expanding out to radius ~30
xm = np.zeros((2, N))
xm[0, :] = t_eval * np.cos(3 * t_eval)
xm[1, :] = t_eval * np.sin(3 * t_eval)

# Initial conditions
x = np.array([0.0, 0.0])
m = 27
ke = 5.0

# Initialize centers tightly around the origin
g = int(np.ceil(np.sqrt(m)))
cx = np.linspace(-1, 1, g)
cy = np.linspace(-1, 1, g)
gx, gy = np.meshgrid(cx, cy)
c = np.column_stack([gx.ravel()[:m], gy.ravel()[:m]])

sigma = np.ones(m) * 2.0
W = np.zeros((m, n))

# INCREASED PROJECTION BOUNDS to allow kernels to travel massive distances
ctrl = AdaptiveController(
    m=m, n=n, ke=ke,
    gamma_W=500.0, gamma_c=50.0, gamma_sigma=10.0,
    W_max=5000.0, c_max=50.0,  # Massively increased c_max bounds!
    sigma_min=0.1, sigma_max=10.0, delta_min=0.05
)

x_history = np.zeros((2, N))
c_history = []
e_history = []

print('Simulating massive data spread (radius expanding to 30)...')
for i in range(N-1):
    t = t_eval[i]
    x_history[:, i] = x
    
    if i % 1000 == 0:
        c_history.append(c.copy())
        
    f_actual = f_true_vdp(x, t)
    x_ref = xm[:, i]
    xm_dot = (xm[:, i+1] - xm[:, i]) / dt if i < N-2 else np.zeros(n)
    e = x - x_ref
    
    u_vec = ctrl.control_law(x, x_ref, xm_dot, W, c, sigma)
    f_hat = ctrl.kernel.f_hat(x, W, c, sigma)
    
    e_id = f_actual - f_hat
    e_history.append(np.linalg.norm(e_id))
    
    W_dot, c_dot, sigma_dot = ctrl.adaptation_laws(x, e_id, W, c, sigma)
    W = W + dt * W_dot
    c = c + dt * c_dot.reshape(m, n)
    sigma = sigma + dt * sigma_dot
    
    W = ctrl.proj_W.hard_clip_matrix(W)
    c = ctrl.proj_c.hard_clip(c.flatten(), n).reshape(m, n)
    sigma = ctrl.proj_sigma.hard_clip(sigma)
    
    x = rk4_step(lambda s, tt: f_true_vdp(s, tt), x, u_vec, t, dt)

e_history.append(e_history[-1])
x_history[:, -1] = x

print(f'Final Mean ID Error: {np.mean(e_history):.4f}')

os.makedirs('results/sysid', exist_ok=True)
plt.figure(figsize=(10, 8))
plt.plot(xm[0, :], xm[1, :], 'k--', alpha=0.5, label='Reference Spiral (Radius 30)')
plt.plot(x_history[0, :], x_history[1, :], 'r-', alpha=0.8, label='Actual System State')

plt.scatter(c[:, 0], c[:, 1], c='blue', marker='X', s=100, label='Final Kernel Centers')

plt.title('Lyapunov Moving Centers Tracking a Massive Data Spread')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.savefig('results/sysid/massive_spread_tracking.png')
print('Plot saved as results/sysid/massive_spread_tracking.png')
