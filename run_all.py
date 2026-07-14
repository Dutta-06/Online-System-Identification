import subprocess
import sys

def main():
    scripts = [
        # "sim0_proposed_sysid.py", # Currently running in the background!
        # "baseline1_neural_ode.py", # Already finished!
        # "baseline2_sindy.py", # Already finished!
        # "baseline3_koopman_dmd.py", # Already finished!
        "baseline4_esn.py",
        "baseline5_narx.py",
        # "baseline6_rff.py", # Skipped as requested
        "sim1_exact_gp_sysid.py",
        "sim2_sparse_gp_sysid.py",
        "plot_sysid_comparison.py"
    ]

    print("==================================================")
    print("Starting execution of AAAI baselines...")
    print("==================================================\n")

    for script in scripts:
        print(f"========== Running {script} ==========")
        try:
            # sys.executable ensures it uses the current virtual environment's python
            result = subprocess.run([sys.executable, script], check=True)
            print(f"========== Finished {script} ==========\n")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] {script} failed with exit code {e.returncode}.")
            print("Stopping execution sequence.")
            sys.exit(1)

    print("==================================================")
    print("All requested experiments completed successfully!")
    print("Check the 'results/sysid/' directory for the plots.")
    print("==================================================")

if __name__ == "__main__":
    main()
