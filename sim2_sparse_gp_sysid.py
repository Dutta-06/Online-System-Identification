# sim2_sparse_gp_sysid.py
import os, sys, time
import numpy as np
import torch
import gpytorch
from tqdm import tqdm
from utils_hd import (f_true_vdp, f_true_duffing6d,
                      f_true_with_shift, generate_reference,
                      vanderpol, coupled_duffing, rk4_step)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Sparse GP using: {DEVICE}")

class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=False
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def run_sparse_gp_sysid(n=2, T=30, dt=0.001, m=27, use_shift=False, shift_time=15.0):
    if n == 2:
        plant_fn = vanderpol
        x0 = [2.0, 0.0]
        f_true_fn = f_true_vdp
        inducing_x = torch.linspace(-3, 3, int(np.sqrt(m)))
        X1, X2 = torch.meshgrid(inducing_x, inducing_x, indexing='ij')
        inducing_state = torch.stack([X1.flatten(), X2.flatten()], dim=1)
        inducing_t = torch.linspace(0, T, m).unsqueeze(1)
        inducing_points = torch.cat([inducing_state, inducing_t], dim=1).float()
    else:
        plant_fn = coupled_duffing
        x0 = [1.0, 0.0, -1.0, 0.5, 0.5, -0.5]
        f_true_fn = f_true_duffing6d
        m_dim = max(2, int(m**(1/6)))
        pts = [torch.linspace(-3, 3, m_dim) for _ in range(6)]
        grids = torch.meshgrid(*pts, indexing='ij')
        inducing_state = torch.stack([g.flatten() for g in grids], dim=1)[:m]
        inducing_t = torch.linspace(0, T, len(inducing_state)).unsqueeze(1)
        inducing_points = torch.cat([inducing_state, inducing_t], dim=1).float()

    t_eval, xm = generate_reference(plant_fn, x0, T, dt)
    N = t_eval.shape[0]

    x = np.array(x0, dtype=np.float64)
    f_hat = np.zeros(n)
    id_history, t_history, ct_history = [], [], []

    models = []
    likelihoods = []
    optimizers = []
    mlls = []
    for _ in range(n):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
        model = SparseGPModel(inducing_points.to(DEVICE)).to(DEVICE)
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()}
        ], lr=0.01)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=1000).to(DEVICE)
        models.append(model)
        likelihoods.append(likelihood)
        optimizers.append(optimizer)
        mlls.append(mll)

    X_buffer = []
    Y_buffer = [[] for _ in range(n)]

    for i in tqdm(range(N-1), desc=f"Sparse GP n={n}"):
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
        X_buffer.append(feature.tolist())
        for d in range(n):
            Y_buffer[d].append(f_actual[d])
            
        x = x_new

        if i % 50 == 0 and len(X_buffer) > m:
            t0 = time.time()
            window = min(500, len(X_buffer))
            Xt = torch.tensor(X_buffer[-window:], dtype=torch.float32).to(DEVICE)
            
            f_hat_new = np.zeros(n)
            for d in range(n):
                Yt = torch.tensor(Y_buffer[d][-window:], dtype=torch.float32).to(DEVICE)
                model = models[d]
                likelihood = likelihoods[d]
                optimizer = optimizers[d]
                mll = mlls[d]
                
                model.train(); likelihood.train()
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
            print(f"\nRunning Sparse GP n={n} {'(shift)' if use_shift else ''}")
            t, e, ct = run_sparse_gp_sysid(n=n, use_shift=use_shift, m=64 if n==2 else 125)
            base = f"results/sysid/{label}/{suf}"
            os.makedirs(base, exist_ok=True)
            np.save(f"{base}/sparse_gp_id_error.npy", e)
            np.save(f"{base}/sparse_gp_time.npy", t)
            np.save(f"{base}/sparse_gp_comptime.npy", ct)
            print(f"  Mean SS id error: {e[15000:].mean():.6f} | Mean step: {ct.mean()*1000:.3f} ms")
