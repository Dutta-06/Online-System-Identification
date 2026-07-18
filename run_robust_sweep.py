import numpy as np
from lag_rff import run_lag_rff

def main():
    print("=== Robust Validation Sweep on 5 Random Trajectories ===")
    kappas = [300, 400, 500, 600, 750, 1000]
    
    np.random.seed(456)
    x0_list = [np.random.uniform(-4, 4, 2) for _ in range(5)]
    
    best_kappa = -1
    best_avg_iae = float('inf')
    
    for k in kappas:
        tot_iaes = []
        for x0 in x0_list:
            _, e, _ = run_lag_rff(n=2, use_shift=True, noise_adaptive=True, 
                                  gamma_0=0.01, kappa=k, x0_override=x0)
            tot_iaes.append(np.sum(e[15000:]) * 0.001)
        
        avg_iae = np.mean(tot_iaes)
        print(f"kappa={k:<4} | Avg Total IAE: {avg_iae:.4f}")
        
        if avg_iae < best_avg_iae:
            best_avg_iae = avg_iae
            best_kappa = k
            
    print("-" * 40)
    print(f"Selected Validation kappa: {best_kappa}")

if __name__ == "__main__":
    main()
