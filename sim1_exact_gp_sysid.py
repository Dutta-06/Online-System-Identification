# sim1_exact_gp_sysid.py
import os, sys, time
import numpy as np
import torch
import gpytorch
from tqdm import tqdm
from utils_hd import (f_true_vdp, f_true_duffing6d,
                      f_true_with_shift, generate_reference,
                      vanderpol, coupled_duffing, rk4_step)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Exact GP using: {DEVICE}")

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def run_exact_gp_sysid(n=2, T=30, dt=0.001, max_gp_points=500, use_shift=False, shift_time=15.0):
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

    x = np.array(x0, dtype=np.float64)
    f_hat = np.zeros(n)
    id_history, t_history, ct_history = [], [], []

    X_train = []
    # Train independent GPs for each output dimension
    Y_train = [[] for _ in range(n)]

    for i in tqdm(range(N-1), desc=f"Exact GP n={n}"):
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
        
        feature = np.concatenate([x, [t]])
        X_train.append(feature.tolist())
        for d in range(n):
            Y_train[d].append(f_actual[d])
            
        x = x_new

        if i % 50 == 0 and len(X_train) > 10:
            t0 = time.time()
            if max_gp_points and len(X_train) > max_gp_points:
                idx = np.random.choice(len(X_train), max_gp_points, replace=False)
                Xt = torch.tensor([X_train[j] for j in idx], dtype=torch.float32).to(DEVICE)
                Yt_list = [torch.tensor([Y_train[d][j] for j in idx], dtype=torch.float32).to(DEVICE) for d in range(n)]
            else:
                Xt = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
                Yt_list = [torch.tensor(Y_train[d], dtype=torch.float32).to(DEVICE) for d in range(n)]

            f_hat_new = np.zeros(n)
            for d in range(n):
                Yt = Yt_list[d]
                likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
                model = ExactGPModel(Xt, Yt, likelihood).to(DEVICE)
                model.train(); likelihood.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(DEVICE)
                
                for _ in range(10):
                    optimizer.zero_grad()
                    output = model(Xt)
                    loss = -mll(output, Yt)
                    loss.backward()
                    optimizer.step()
                    
                model.eval(); likelihood.eval()
                with torch.no_grad():
                    xq = torch.tensor([feature], dtype=torch.float32).to(DEVICE)
                    pred = likelihood(model(xq))
                    f_hat_new[d] = pred.mean.cpu().item()
                    
            f_hat = f_hat_new
            ct_history.append(time.time() - t0)

    return np.array(t_history), np.array(id_history), np.array(ct_history)

if __name__ == "__main__":
    for n, label in [(2, 'n2'), (6, 'n6')]:
        for use_shift, suf in [(False, 'normal'), (True, 'shift')]:
            print(f"\nRunning Exact GP n={n} {'(shift)' if use_shift else ''}")
            t, e, ct = run_exact_gp_sysid(n=n, use_shift=use_shift, max_gp_points=200)
            base = f"results/sysid/{label}/{suf}"
            os.makedirs(base, exist_ok=True)
            np.save(f"{base}/exact_gp_id_error.npy", e)
            np.save(f"{base}/exact_gp_time.npy", t)
            np.save(f"{base}/exact_gp_comptime.npy", ct)
            print(f"  Mean SS id error: {e[15000:].mean():.6f} | Mean step: {ct.mean()*1000:.3f} ms")
