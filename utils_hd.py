import sys, os
import numpy as np
import importlib.util

# Dynamically load the original utils_hd to avoid circular import collision
parent_utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../experiments/utils_hd.py'))
spec = importlib.util.spec_from_file_location("parent_utils_hd", parent_utils_path)
parent_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_utils)

f_true_vdp = parent_utils.f_true_vdp
f_true_duffing6d = parent_utils.f_true_duffing6d
generate_reference = parent_utils.generate_reference
vanderpol = parent_utils.vanderpol
coupled_duffing = parent_utils.coupled_duffing
rk4_step = parent_utils.rk4_step

def f_true_vdp_shifted(x, t, mu_factor=2.0):
    """Van der Pol with doubled mu after shift."""
    x1, x2 = x[0], x[1]
    mu = 1.5 * mu_factor
    return np.array([x2, mu*(1 - x1**2)*x2 - x1])

def f_true_duffing6d_shifted(x, t, alpha_factor=2.0,
                              beta_d=0.25, delta=0.3,
                              gamma=0.3, omega=1.2, k_couple=0.1):
    """Coupled Duffing with doubled alpha after shift."""
    alpha_t = (1.0 + 0.3*np.sin(0.5*t)) * alpha_factor
    x1,v1,x2,v2,x3,v3 = x
    return np.array([
        v1,
        -delta*v1 + alpha_t*x1 - beta_d*x1**3 +
        gamma*np.cos(omega*t) + k_couple*(x2-x1),
        v2,
        -delta*v2 + alpha_t*x2 - beta_d*x2**3 +
        gamma*np.cos(omega*t) + k_couple*(x1-x2+x3-x2),
        v3,
        -delta*v3 + alpha_t*x3 - beta_d*x3**3 +
        gamma*np.cos(omega*t) + k_couple*(x2-x3)
    ])

def f_true_with_shift(x, t, n, shift_time=15.0):
    """Unified shift wrapper for both plants."""
    if n == 2:
        if t < shift_time:
            return f_true_vdp(x, t)
        else:
            return f_true_vdp_shifted(x, t)
    elif n == 6:
        if t < shift_time:
            return f_true_duffing6d(x, t)
        else:
            return f_true_duffing6d_shifted(x, t)
