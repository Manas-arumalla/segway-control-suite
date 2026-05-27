"""Metrics behave correctly on synthetic trajectories."""

from __future__ import annotations

import numpy as np

from segway.analysis import compute_metrics


def _decaying_trajectory(n: int = 500, tau: float = 1.0):
    t = np.linspace(0, 10, n)
    theta = 0.3 * np.exp(-t / tau) * np.cos(3 * t)
    states = np.zeros((n, 4))
    states[:, 2] = theta
    states[:, 0] = 0.5 * (1 - np.exp(-t / tau))  # base settles to 0.5 m
    controls = np.gradient(theta, t)
    return t, states, controls


def test_metrics_on_decaying_signal():
    t, states, controls = _decaying_trajectory()
    m = compute_metrics(t, states, controls, fell=False)

    assert m["fell"] is False
    assert m["peak_angle_rad"] > 0
    assert m["peak_angle_deg"] == np.degrees(m["peak_angle_rad"])
    assert np.isfinite(m["settling_time_angle"])  # decays, so it settles
    assert m["rms_angle_rad"] > 0
    assert m["control_effort"] >= 0
    assert m["max_pos_drift_m"] >= abs(m["final_pos_m"]) - 1e-9


def test_settling_time_infinite_when_never_settles():
    t = np.linspace(0, 5, 200)
    theta = 0.2 * np.ones_like(t)  # never enters the band
    states = np.zeros((len(t), 4))
    states[:, 2] = theta
    m = compute_metrics(t, states, np.zeros_like(t), fell=False, angle_tol=0.05)
    assert m["settling_time_angle"] == float("inf")
