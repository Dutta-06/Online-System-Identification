import numpy as np
import time

def measure_rff_latency(n_f, n=2, trials=1000):
    lam = 0.99
    P = [np.eye(n_f) / 0.01 for _ in range(n)]
    phi = np.random.randn(n_f)
    W_out = np.zeros((n, n_f))
    f_actual = np.random.randn(n)
    
    # Warmup
    for _ in range(10):
        for d in range(n):
            Pp = P[d] @ phi
            K = Pp / (lam + phi @ Pp)
            P[d] = (P[d] - np.outer(K, Pp)) / lam
            error_d = f_actual[d] - W_out[d] @ phi
            W_out[d] += K * error_d
            
    t0 = time.perf_counter()
    for _ in range(trials):
        for d in range(n):
            Pp = P[d] @ phi
            K = Pp / (lam + phi @ Pp)
            P[d] = (P[d] - np.outer(K, Pp)) / lam
            error_d = f_actual[d] - W_out[d] @ phi
            W_out[d] += K * error_d
    t1 = time.perf_counter()
    
    return ((t1 - t0) / trials) * 1000  # ms per step

if __name__ == "__main__":
    print("Empirical Latency Scaling (ms per step) - RFF (RLS)")
    for n_f in [50, 100, 200, 400, 800]:
        latency = measure_rff_latency(n_f, n=6)
        print(f"n_f = {n_f:3d} : {latency:.4f} ms")
