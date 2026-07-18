# plot_sysid_comparison.py
import numpy as np
import matplotlib.pyplot as plt
import os

METHODS = {
    'Proposed (Proj)': ('#4CAF50', 'proposed'),
    'Exact GP':        ('#2196F3', 'exact_gp'),
    'Sparse GP':       ('#FF9800', 'sparse_gp'),
    'Neural ODE':      ('#E91E63', 'neural_ode'),
    'SINDy':           ('#9C27B0', 'sindy'),
    'Koopman DMD':     ('#00BCD4', 'koopman'),
    'ESN':             ('#FF5722', 'esn'),
    'NARX':            ('#607D8B', 'narx'),
    'RFF (Fixed)':     ('#795548', 'rff_gd'),
    'RFF + Proposed':  ('#FFEB3B', 'lag_rff'),
}

DIMS = {'n=2 (Van der Pol)': 'n2',
        'n=6 (Duffing)':     'n6'}

CONDITIONS = {'Normal': 'normal', 'After Shift (t>15s)': 'shift'}

def load(path):
    return np.load(path) if os.path.exists(path) else None

def ss_mean(e, frac=0.6):
    if e is None: return None
    return e[int(frac*len(e)):].mean()

def recovery_time(e, shift_step=15000, threshold_factor=1.1):
    """Steps after shift until error returns within threshold of pre-shift mean."""
    if e is None or len(e) <= shift_step: return None
    pre_shift_mean = e[5000:shift_step].mean()
    for i in range(shift_step, len(e)):
        if e[i] <= pre_shift_mean * threshold_factor:
            return (i - shift_step) * 0.001  # convert to seconds
    return None  # never recovered

if __name__ == "__main__":
    os.makedirs('results/sysid', exist_ok=True)
    
    # Figure 1: Identification error over time
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Online System Identification — All Methods', fontsize=14)

    for col, (dim_label, dim_key) in enumerate(DIMS.items()):
        for row, (cond_label, cond_key) in enumerate(CONDITIONS.items()):
            ax = axes[row, col]
            for mname, (color, fkey) in METHODS.items():
                path = f"results/sysid/{dim_key}/{cond_key}/{fkey}_id_error.npy"
                e = load(path)
                if e is None: continue
                window = 500
                e_s = np.convolve(e, np.ones(window)/window, mode='valid')
                t_s = np.arange(len(e_s)) * 0.001
                ax.semilogy(t_s, e_s, label=mname, color=color, linewidth=1.2, alpha=0.85)
            if row == 0 and col == 1:
                ax.legend(fontsize=7, loc='upper right')
            ax.axvline(x=15.0, color='black', linestyle='--', alpha=0.5, linewidth=0.8,
                       label='Shift' if cond_key == 'shift' else '')
            ax.set_title(f"{dim_label} — {cond_label}", fontsize=10)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('ID Error ||f - f̂||')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/sysid/fig1_id_error_over_time.png', dpi=200, bbox_inches='tight')
    print("Saved fig1")

    # Figure 2: Summary table
    print("-" * 140)
    print(f"{'Method':<20} {'n=2 Normal SS':<15} {'n=2 Shift SS':<15} {'n=2 Total IAE':<15} {'n=2 Recovery':<15} {'n=6 Normal SS':<15} {'n=6 Shift SS':<15} {'n=6 Total IAE':<15} {'n=6 Recovery':<15}")
    print("-" * 140)

    for mname, (color, fkey) in METHODS.items():
        e_n2_norm = load(f"results/sysid/n2/normal/{fkey}_id_error.npy")
        e_n2_shift = load(f"results/sysid/n2/shift/{fkey}_id_error.npy")
        e_n6_norm = load(f"results/sysid/n6/normal/{fkey}_id_error.npy")
        e_n6_shift = load(f"results/sysid/n6/shift/{fkey}_id_error.npy")

        ss_n2_norm = ss_mean(e_n2_norm) if e_n2_norm is not None else 0.0
        ss_n2_shift = ss_mean(e_n2_shift) if e_n2_shift is not None else 0.0
        ss_n6_norm = ss_mean(e_n6_norm) if e_n6_norm is not None else 0.0
        ss_n6_shift = ss_mean(e_n6_shift) if e_n6_shift is not None else 0.0

        rec_n2 = recovery_time(e_n2_shift)
        rec_n6 = recovery_time(e_n6_shift)
        
        tot_n2 = np.sum(e_n2_shift[15000:]) * 0.001 if e_n2_shift is not None else 0.0
        tot_n6 = np.sum(e_n6_shift[15000:]) * 0.001 if e_n6_shift is not None else 0.0

        rec_n2_str = f"{rec_n2:.2f}s" if rec_n2 else "Never"
        rec_n6_str = f"{rec_n6:.2f}s" if rec_n6 else "Never"

        print(f"{mname:<20} {ss_n2_norm:<15.4f} {ss_n2_shift:<15.4f} {tot_n2:<15.4f} {rec_n2_str:<15} {ss_n6_norm:<15.4f} {ss_n6_shift:<15.4f} {tot_n6:<15.4f} {rec_n6_str:<15}")

    # Figure 3: Recovery time bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Recovery Time After Abrupt Parameter Shift (t=15s)', fontsize=13)

    for col, (dim_label, dim_key) in enumerate(DIMS.items()):
        ax = axes[col]
        names, times, colors = [], [], []
        for mname, (color, fkey) in METHODS.items():
            e = load(f"results/sysid/{dim_key}/shift/{fkey}_id_error.npy")
            rt = recovery_time(e)
            if e is not None:
                names.append(mname)
                times.append(rt if rt is not None else 15.0)
                colors.append(color)
        if names:
            bars = ax.bar(names, times, color=colors, alpha=0.8)
            ax.set_title(dim_label, fontsize=11)
            ax.set_ylabel('Recovery Time (s)')
            ax.tick_params(axis='x', rotation=30)
            ax.set_ylim(0, 16)
            ax.axhline(y=15.0, color='red', linestyle='--', alpha=0.5, label='Never recovered')
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('results/sysid/fig3_recovery_time.png', dpi=200, bbox_inches='tight')
    print("Saved fig3")

    # Figure 4: Integrated Absolute Error (IAE) Post-Shift
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Integrated Absolute Error (IAE) After Shift (t=15s to 30s)', fontsize=13)

    for col, (dim_label, dim_key) in enumerate(DIMS.items()):
        ax = axes[col]
        names, iaes, colors = [], [], []
        for mname, (color, fkey) in METHODS.items():
            e = load(f"results/sysid/{dim_key}/shift/{fkey}_id_error.npy")
            if e is not None and len(e) > 15000:
                iae = np.sum(e[15000:]) * 0.001
                names.append(mname)
                iaes.append(iae)
                colors.append(color)
        if names:
            bars = ax.bar(names, iaes, color=colors, alpha=0.8)
            ax.set_title(dim_label, fontsize=11)
            ax.set_ylabel('IAE (Lower is Better)')
            ax.tick_params(axis='x', rotation=30)
            ax.set_yscale('log') # Log scale because Koopman/SINDy explode
            ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/sysid/fig4_iae_post_shift.png', dpi=200, bbox_inches='tight')
    print("Saved fig4")
