# baseline5_narx.py
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from tqdm import tqdm
from utils_hd import (f_true_vdp, f_true_duffing6d,
                      f_true_with_shift, generate_reference,
                      vanderpol, coupled_duffing, rk4_step)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"NARX using: {DEVICE}")

class NARXNet(nn.Module):
    def __init__(self, n, lag=5, width=64):
        super().__init__()
        input_dim = n * (lag + 1) + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, n)
        )
    def forward(self, x):
        return self.net(x)

def run_narx(n=2, T=30, dt=0.001,
             use_shift=False, shift_time=15.0,
             lag=5, width=64, update_every=50):
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

    model = NARXNet(n, lag, width).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    x = np.array(x0, dtype=np.float64)
    f_hat = np.zeros(n)

    x_lag = deque([np.zeros(n)] * (lag + 1), maxlen=lag + 1)
    id_history, t_history, ct_history = [], [], []
    X_buf, F_buf = [], []

    for i in tqdm(range(N-1), desc=f"NARX n={n}"):
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

        x_lag.appendleft(x.copy())
        feature = np.concatenate(list(x_lag) + [[t]])

        X_buf.append(feature.tolist())
        F_buf.append(f_actual.tolist())
        x = x_new

        if i % update_every == 0 and len(X_buf) > lag + 10:
            t0 = time.perf_counter()
            window = min(200, len(X_buf))
            Xt = torch.tensor(X_buf[-window:], dtype=torch.float32).to(DEVICE)
            Ft = torch.tensor(F_buf[-window:], dtype=torch.float32).to(DEVICE)

            model.train()
            for _ in range(5):
                optimizer.zero_grad()
                pred = model(Xt)
                loss = loss_fn(pred, Ft)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                xq = torch.tensor([feature], dtype=torch.float32).to(DEVICE)
                f_hat = model(xq).cpu().numpy()[0]

            ct_history.append(time.perf_counter() - t0)

    return (np.array(t_history), np.array(id_history), np.array(ct_history))

if __name__ == "__main__":
    for n, label in [(2, 'n2'), (6, 'n6')]:
        for use_shift, suf in [(False, 'normal'), (True, 'shift')]:
            print(f"\nRunning NARX n={n} {'(shift)' if use_shift else ''}")
            t, e, ct = run_narx(n=n, use_shift=use_shift)
            base = f"results/sysid/{label}/{suf}"
            os.makedirs(base, exist_ok=True)
            np.save(f"{base}/narx_id_error.npy", e)
            np.save(f"{base}/narx_time.npy", t)
            np.save(f"{base}/narx_comptime.npy", ct)
            print(f"  Mean SS id error: {e[15000:].mean():.6f} | Mean step: {ct.mean()*1000:.3f} ms")
