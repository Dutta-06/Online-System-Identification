import numpy as np
from lag_rff import run_lag_rff

def main():
    # Pick kappa=600 for now, though we may adjust this based on the pre-shift sweep
    kappa_val = 600
    gamma0_val = 0.01
    
    # Generate 5 random trajectories within a plausible domain
    np.random.seed(123)
    x0_list = [np.random.uniform(-4, 4, 2) for _ in range(5)]
    
    results_adaptive = []
    results_fixed = []
    
    print("=== Multi-Trajectory Generalization Test ===")
    print(f"Using Fixed kappa={kappa_val}, gamma0={gamma0_val}")
    print(f"{'x0':<20} | {'Adaptive IAE':<15} | {'Fixed IAE':<15} | {'Better?':<10}")
    print("-" * 65)
    
    for x0 in x0_list:
        # Run Adaptive
        _, e_adapt, _ = run_lag_rff(n=2, use_shift=True, noise_adaptive=True, 
                                    gamma_0=gamma0_val, kappa=kappa_val, x0_override=x0)
        tot_adapt = np.sum(e_adapt[15000:]) * 0.001
        
        # Run Fixed (baseline)
        _, e_fix, _ = run_lag_rff(n=2, use_shift=True, noise_adaptive=False, 
                                  gamma_omega=0.0, gamma_b=0.0, x0_override=x0)
        tot_fix = np.sum(e_fix[15000:]) * 0.001
        
        results_adaptive.append(tot_adapt)
        results_fixed.append(tot_fix)
        
        x0_str = f"[{x0[0]:.2f}, {x0[1]:.2f}]"
        better = "YES" if tot_adapt < tot_fix else "NO"
        print(f"{x0_str:<20} | {tot_adapt:<15.4f} | {tot_fix:<15.4f} | {better:<10}")
        
    adapt_arr = np.array(results_adaptive)
    fix_arr = np.array(results_fixed)
    
    print("-" * 65)
    print(f"{'MEAN ± STD':<20} | {adapt_arr.mean():.4f} ± {adapt_arr.std():.4f} | {fix_arr.mean():.4f} ± {fix_arr.std():.4f}")

if __name__ == "__main__":
    main()
