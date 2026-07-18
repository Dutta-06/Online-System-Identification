import numpy as np
from lag_rff import run_lag_rff

def main():
    print("=== Validation Sweep on Throwaway Trajectory x0=[-3, 2] ===")
    kappas = [300, 400, 500, 600, 750, 1000]
    
    best_kappa = -1
    best_iae = float('inf')
    
    for k in kappas:
        _, e, _ = run_lag_rff(n=2, use_shift=True, noise_adaptive=True, 
                              gamma_0=0.01, kappa=k, x0_override=[-3.0, 2.0])
        tot = np.sum(e[15000:]) * 0.001
        print(f"kappa={k:<4} | Total IAE: {tot:.4f}")
        
        if tot < best_iae:
            best_iae = tot
            best_kappa = k
            
    print("-" * 40)
    print(f"Selected Validation kappa: {best_kappa}")

if __name__ == "__main__":
    main()
