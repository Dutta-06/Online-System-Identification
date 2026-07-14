# baseline1_neural_ode.py
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
from utils_hd import (f_true_vdp, f_true_duffing6d,
                      f_true_with_shift, generate_reference,
                      vanderpol, coupled_duffing, rk4_step)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Neural ODE using: {DEVICE}")

class ODEFunc(nn.Module):
    """Learned dynamics function f_hat(x, t)."""
    def __init__(self, n, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n + 1, width),   # +1 for time input
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, n)
        )
    def forward(self, t, x):
        # x: (batch, n), t: scalar
        t_vec = t.expand(x.shape[0], 1)
        inp = torch.cat([x, t_vec], dim=-1)
        return self.net(inp)

def run_neural_ode(n=2, T=30, dt=0.001, width=128,
                   use_shift=False, shift_time=15.0):
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

    model = ODEFunc(n, width).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    x = np.array(x0, dtype=np.float64)
    f_hat = np.zeros(n)

    id_history, t_history, ct_history = [], [], []
    X_buf, F_buf = [], []

    for i in tqdm(range(N-1), desc=f"Neural ODE n={n}"):
        t = t_eval[i]

        # Use shift dynamics if requested
        if use_shift:
            f_actual = f_true_with_shift(x, t, n, shift_time)
        else:
            f_actual = f_true_fn(x, t)

        # Identification error
        id_err = np.linalg.norm(f_actual - f_hat)
        id_history.append(id_err)
        t_history.append(t)

        # Control and plant step (simple feedback for excitation)
        x_ref = xm[:, i]
        u = -5.0 * (x - x_ref)
        x = rk4_step(
            lambda s, tt: f_true_with_shift(s, tt, n, shift_time)
            if use_shift else f_true_fn(s, tt),
            x, u, t, dt
        )

        # Collect data
        X_buf.append(x.tolist())
        F_buf.append(f_actual.tolist())

        # Update every 50 steps
        if i % 50 == 0 and len(X_buf) > 20:
            t0 = time.perf_counter()
            window = min(200, len(X_buf))
            Xt = torch.tensor(X_buf[-window:],
                              dtype=torch.float32).to(DEVICE)
            Ft = torch.tensor(F_buf[-window:],
                              dtype=torch.float32).to(DEVICE)
            t_scalar = torch.tensor(t, dtype=torch.float32).to(DEVICE)

            model.train()
            for _ in range(10):
                optimizer.zero_grad()
                f_pred = model(t_scalar, Xt)
                loss = loss_fn(f_pred, Ft)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                xq = torch.tensor(x[None], dtype=torch.float32).to(DEVICE)
                f_hat = model(t_scalar, xq).cpu().numpy()[0]

            ct_history.append(time.perf_counter() - t0)

    return (np.array(t_history),
            np.array(id_history),
            np.array(ct_history))


if __name__ == "__main__":
    for n, label in [(2, 'n2'), (6, 'n6')]:
        for use_shift, suf in [(False, 'normal'), (True, 'shift')]:
            print(f"\nRunning Neural ODE n={n} {'(shift)' if use_shift else ''}")
            t, e, ct = run_neural_ode(n=n, use_shift=use_shift)
            base = f"results/sysid/{label}/{suf}"
            os.makedirs(base, exist_ok=True)
            np.save(f"{base}/neural_ode_id_error.npy", e)
            np.save(f"{base}/neural_ode_time.npy", t)
            np.save(f"{base}/neural_ode_comptime.npy", ct)
            print(f"  Mean SS id error: {e[15000:].mean():.6f} | "
                  f"Mean step: {ct.mean()*1000:.3f} ms")
