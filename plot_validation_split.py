import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    gammas = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01]
    
    # Values generated from the validation split (pre-shift normal trajectory)
    pre_shift_ss_iae = [0.019769, 0.019730, 0.019682, 0.019540, 0.019310, 0.019447, 0.019528]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gammas, pre_shift_ss_iae, marker='o', linestyle='-', color='#9C27B0', label='Validation SS IAE ($t \\in [0, 15]$)', linewidth=2, markersize=8)
    
    # Highlight the minimum
    min_idx = pre_shift_ss_iae.index(min(pre_shift_ss_iae))
    ax.plot(gammas[min_idx], pre_shift_ss_iae[min_idx], marker='*', color='red', markersize=15, label=f'Selected Optimal $\\gamma$={gammas[min_idx]}')

    ax.set_xscale('log')
    ax.set_xlabel(r'Geometry Adaptation Gain ($\gamma$)', fontsize=12)
    ax.set_ylabel('Pre-Shift Steady-State IAE (Validation Error)', fontsize=12)
    ax.set_title('Cross-Validation of Adaptation Gain on Normal Trajectory ($n=2$)', fontsize=13, fontweight='bold')
    
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax.legend(fontsize=10)
    
    out_dir = 'results/sysid'
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'fig6_validation_split.png')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved {path}')

if __name__ == '__main__':
    main()
