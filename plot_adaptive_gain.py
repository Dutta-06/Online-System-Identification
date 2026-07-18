import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    # --- Left panel: Total IAE vs kappa for both trajectories ---
    kappas = [300, 400, 500, 600, 750, 1000]

    # x0=[2,0] results
    iae_x0_orig = [0.9796, 1.1110, 0.7095, 0.6768, 0.6838, 0.6921]
    fixed_rff_orig = 0.7419

    # x0=[3,1] results
    iae_x0_oos = [0.6390, 0.6498, 0.6577, 0.6637, 0.6704, 0.6780]
    fixed_rff_oos = 0.7225

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel
    ax1.plot(kappas, iae_x0_orig, marker='o', linestyle='-', color='#4CAF50',
             label=r'Adaptive-$\gamma$, $x_0=[2,0]$', linewidth=2, markersize=8)
    ax1.plot(kappas, iae_x0_oos, marker='s', linestyle='-', color='#2196F3',
             label=r'Adaptive-$\gamma$, $x_0=[3,1]$ (out-of-sample)', linewidth=2, markersize=8)
    ax1.axhline(fixed_rff_orig, color='#4CAF50', linestyle='--', alpha=0.6,
                label=f'Fixed RFF baseline, $x_0=[2,0]$ ({fixed_rff_orig})')
    ax1.axhline(fixed_rff_oos, color='#2196F3', linestyle='--', alpha=0.6,
                label=f'Fixed RFF baseline, $x_0=[3,1]$ ({fixed_rff_oos})')

    # Shade the region where BOTH trajectories beat their baseline
    ax1.axvspan(500, 1000, alpha=0.08, color='green', label='Both-win region')

    ax1.set_xlabel(r'Noise Sensitivity ($\kappa$)', fontsize=12)
    ax1.set_ylabel('Total IAE (Post-Shift)', fontsize=12)
    ax1.set_title(r'Noise-Adaptive Gain: $\gamma_{eff} = \gamma_0 / (1 + \kappa \hat{\nu})$',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Right panel: bar chart comparing Fixed vs Manual vs Adaptive on both trajectories
    methods = ['Fixed RFF', r'Manual $\gamma$=0.005', r'Adaptive-$\gamma$ ($\kappa$=600)']
    x0_20 = [0.7419, 0.5749, 0.6768]
    x0_31 = [0.7225, 0.5627, 0.6637]

    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax2.bar(x - width/2, x0_20, width, label=r'$x_0=[2,0]$', color='#4CAF50', alpha=0.8)
    bars2 = ax2.bar(x + width/2, x0_31, width, label=r'$x_0=[3,1]$', color='#2196F3', alpha=0.8)

    ax2.set_ylabel('Total IAE (Post-Shift)', fontsize=12)
    ax2.set_title('Cross-Trajectory Comparison ($n=2$)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)

    out_dir = 'results/sysid'
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'fig7_adaptive_gain.png')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved {path}')

if __name__ == '__main__':
    main()
