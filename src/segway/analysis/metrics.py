"""Quantitative performance metrics for a closed-loop trajectory.

These turn a simulation run into comparable numbers — the foundation of the benchmark.
All metrics operate on plain arrays so they work with any simulation backend.
"""

from __future__ import annotations

import numpy as np

# NumPy 2.0 renamed trapz -> trapezoid (and later removed trapz). Prefer the new name and only
# fall back to the old one on NumPy < 2.0 — never reference np.trapz when trapezoid exists, or
# the eager default would crash at import on NumPy versions that dropped trapz.
try:
    _trapz = np.trapezoid          # NumPy >= 2.0
except AttributeError:             # pragma: no cover -- NumPy < 2.0
    _trapz = np.trapz  # noqa: NPY201


def compute_metrics(
    t: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    fell: bool = False,
    angle_tol: float = 0.05,
    pos_tol: float = 0.05,
) -> dict[str, float | bool]:
    """Compute a dictionary of performance metrics for one trajectory.

    Args:
        t: time stamps, shape ``(N,)``.
        states: state history, shape ``(N, 4)`` = ``[x, x_dot, theta, theta_dot]``.
        controls: applied torque history, shape ``(N,)``.
        fell: whether the run terminated by falling over.
        angle_tol: settling band for ``|theta|`` [rad].
        pos_tol: settling band for ``|x|`` [m].

    Returns:
        Metrics including settling time, peak angle/overshoot, RMS tilt, IAE/ISE,
        control effort, peak torque, and final/peak position drift. Settling time is
        ``inf`` if the quantity never settles within the run.
    """
    t = np.asarray(t, dtype=float)
    states = np.asarray(states, dtype=float)
    controls = np.asarray(controls, dtype=float)

    x = states[:, 0]
    theta = states[:, 2]

    def settling_time(signal: np.ndarray, tol: float) -> float:
        outside = np.where(np.abs(signal) > tol)[0]
        if outside.size == 0:
            return 0.0
        last = int(outside[-1])
        if last >= signal.size - 1:
            return float("inf")  # never settled
        return float(t[last + 1])

    peak_angle = float(np.max(np.abs(theta))) if theta.size else 0.0

    return {
        "fell": bool(fell),
        "settling_time_angle": settling_time(theta, angle_tol),
        "settling_time_pos": settling_time(x, pos_tol),
        "peak_angle_rad": peak_angle,
        "peak_angle_deg": float(np.degrees(peak_angle)),
        "rms_angle_rad": float(np.sqrt(np.mean(theta**2))) if theta.size else 0.0,
        "iae_angle": float(_trapz(np.abs(theta), t)) if t.size > 1 else 0.0,
        "ise_angle": float(_trapz(theta**2, t)) if t.size > 1 else 0.0,
        "control_effort": float(_trapz(controls**2, t)) if t.size > 1 else 0.0,
        "total_abs_torque": float(_trapz(np.abs(controls), t)) if t.size > 1 else 0.0,
        "peak_torque": float(np.max(np.abs(controls))) if controls.size else 0.0,
        "final_pos_m": float(x[-1]) if x.size else 0.0,
        "max_pos_drift_m": float(np.max(np.abs(x))) if x.size else 0.0,
    }
