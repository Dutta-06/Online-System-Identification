import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    gammas = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01]
    iae = [0.7367, 0.7175, 0.6966, 0.6444, 0.5749, 1.0134, 1.2974]
    
    fixed_rff = 0.7419
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gammas, iae, marker='o', linestyle='-', color='#4CAF50', label='RFF + Proposed (LAG)', linewidth=2, markersize=8)
    ax.axhline(fixed_rff, color='#2196F3', linestyle='--', label='RFF (Fixed Geometry)', linewidth=2)
    
    ax.set_xscale('log')
    ax.set_xlabel(r'Geometry Adaptation Gain ($\gamma$)', fontsize=12)
    ax.set_ylabel('Total IAE (Post-Shift)', fontsize=12)
    ax.set_title('Sensitivity Analysis: Noise Rejection vs Overfitting ($n=2$)', fontsize=14, fontweight='bold')
    
    ax.fill_between(gammas, 0, fixed_rff, alpha=0.1, color='#4CAF50', label='Improvement Region')
    
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax.legend(fontsize=10)
    
    out_dir = 'results/sysid'
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'fig5_sensitivity_sweep.png')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved {path}')

if __name__ == '__main__':
    main()
