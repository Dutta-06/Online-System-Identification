"""
Observable Lyapunov Monitor (AAAI Paper Specific)
==================================================

Tracks the tracking error and identification error norms to verify 
convergence to the Uniform Ultimate Boundedness (UUB) region without
assuming oracle access to the optimal parameters W*, c*, σ*.

Properties verified at runtime:
    - ‖e(t)‖ ≤ R_e
    - ‖e_id(t)‖ ≤ R_f
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LyapunovSnapshot:
    """Single-timestep snapshot of observable error norms."""
    t: float
    e_norm: float      
    e_id_norm: float   
    R_e: Optional[float] = None
    R_f: Optional[float] = None
    e_bounded: Optional[bool] = None
    f_bounded: Optional[bool] = None

class LyapunovMonitor:
    def __init__(self, R_e: Optional[float] = None,
                 R_f: Optional[float] = None):
        self.R_e = R_e
        self.R_f = R_f
        self.history: List[LyapunovSnapshot] = []

    def compute(self, t: float, e: np.ndarray,
                e_id: np.ndarray) -> LyapunovSnapshot:
        e_norm = float(np.linalg.norm(e))
        e_id_norm = float(np.linalg.norm(e_id))

        e_bounded = (e_norm <= self.R_e) if self.R_e is not None else None
        f_bounded = (e_id_norm <= self.R_f) if self.R_f is not None else None

        snapshot = LyapunovSnapshot(
            t=t, e_norm=e_norm, e_id_norm=e_id_norm,
            R_e=self.R_e, R_f=self.R_f,
            e_bounded=e_bounded, f_bounded=f_bounded
        )

        self.history.append(snapshot)
        return snapshot

    def get_time_series(self):
        t = np.array([snap.t for snap in self.history])
        e_norm = np.array([snap.e_norm for snap in self.history])
        e_id_norm = np.array([snap.e_id_norm for snap in self.history])
        return t, e_norm, e_id_norm
