"""Sensor model for the self-balancing robot.

Default configuration measures the two physically available quantities — base position
(wheel encoder) and body tilt (IMU) — and corrupts them with Gaussian noise, a constant
bias, and optional quantization. Velocities are *not* measured; the estimator reconstructs
them. The measurement matrix is therefore ``C = [[1,0,0,0],[0,0,1,0]]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import DEFAULT_C


@dataclass
class SensorSpec:
    """Noise/bias/quantization for the position (encoder) and tilt (IMU) channels."""

    pos_noise_std: float = 0.005    # encoder position noise [m]
    tilt_noise_std: float = 0.01    # IMU tilt noise [rad] (~0.57 deg)
    pos_bias: float = 0.0           # constant encoder offset [m]
    tilt_bias: float = 0.0          # constant IMU offset [rad]
    pos_quant: float = 0.0          # encoder resolution [m]; 0 = none
    tilt_quant: float = 0.0         # IMU resolution [rad]; 0 = none


class SensorModel:
    """Turns a true state into a noisy measurement ``y = C x + bias + noise`` (quantized)."""

    def __init__(self, spec: SensorSpec | None = None, C: np.ndarray | None = None, seed: int = 0):
        self.spec = spec or SensorSpec()
        self.C = DEFAULT_C if C is None else np.asarray(C, dtype=float)
        self.rng = np.random.default_rng(seed)
        self._bias = np.array([self.spec.pos_bias, self.spec.tilt_bias])
        self._noise_std = np.array([self.spec.pos_noise_std, self.spec.tilt_noise_std])
        self._quant = np.array([self.spec.pos_quant, self.spec.tilt_quant])

    @property
    def R(self) -> np.ndarray:
        """Measurement noise covariance implied by the noise std-devs."""
        return np.diag(self._noise_std**2)

    def measure(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Return a noisy measurement of ``state``."""
        y = self.C @ np.asarray(state, dtype=float)
        y = y + self._bias + self.rng.normal(0.0, self._noise_std)
        for i, q in enumerate(self._quant):
            if q > 0:
                y[i] = np.round(y[i] / q) * q
        return y
