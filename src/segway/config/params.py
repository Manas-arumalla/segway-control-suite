"""Physical and simulation parameters.

`RobotParams` is the single source of truth for the plant's physics. It is shared by the
dynamics model, every controller, the simulator, and the analysis tools — so a controller
is always designed for *exactly* the robot it is simulated on (a bug that plagued the
legacy code, where two different parameter sets silently coexisted).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class RobotParams:
    """Physical parameters of the wheeled inverted pendulum.

    State convention: ``[x, x_dot, theta, theta_dot]`` with ``theta = 0`` upright.
    Input convention: ``u = tau`` (motor torque), producing traction force ``tau/r`` on
    the base and reaction torque ``-tau`` on the body. See ``docs/implementation-notes.md``.
    """

    m_base: float = 1.0      # chassis/base mass [kg]
    m_wheel: float = 0.432   # mass of EACH wheel [kg]
    m_pend: float = 5.0      # body (pendulum) mass [kg]
    l: float = 0.4           # distance from axle to body COM [m]
    r: float = 0.1           # wheel radius [m]
    I_pend: float = 0.2      # body moment of inertia about its COM [kg*m^2]
    g: float = 9.81          # gravitational acceleration [m/s^2]
    b_x: float = 0.05        # viscous damping on the axle (x) [N*s/m]
    b_theta: float = 0.2     # viscous damping on the body hinge [N*m*s/rad]

    def __post_init__(self) -> None:
        for name in ("m_base", "m_wheel", "m_pend", "l", "r", "I_pend", "g"):
            if getattr(self, name) <= 0:
                raise ValueError(f"RobotParams.{name} must be > 0, got {getattr(self, name)}")
        if self.b_x < 0 or self.b_theta < 0:
            raise ValueError("damping coefficients must be >= 0")

    @property
    def M(self) -> float:
        """Total translating base mass = chassis + two wheels [kg]."""
        return self.m_base + 2.0 * self.m_wheel

    @property
    def D(self) -> float:
        """Determinant of the linearized mass matrix: (M+m)(I+m*l^2) - (m*l)^2."""
        m = self.m_pend
        return (self.M + m) * (self.I_pend + m * self.l**2) - (m * self.l) ** 2

    # --- (de)serialization -------------------------------------------------
    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RobotParams:
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in known})

    def to_yaml(self, path: str | Path) -> None:
        import yaml  # lazy: only needed if YAML config is used

        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobotParams:
        import yaml

        return cls.from_dict(yaml.safe_load(Path(path).read_text()))


@dataclass
class SimConfig:
    """Numerical settings for the headless simulator."""

    dt: float = 0.001            # integrator step [s]
    control_dt: float | None = None  # controller update period [s]; None => every step
    duration: float = 10.0       # total simulated time [s]
    fall_angle: float = 1.2      # |theta| beyond this is considered "fallen" [rad]
    record_every: int = 10       # store every Nth step in the trajectory
    seed: int = 0                # RNG seed for any stochastic elements (noise, kicks)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.control_dt is not None and self.control_dt < self.dt:
            raise ValueError("control_dt must be >= dt")
        if self.duration <= 0:
            raise ValueError("duration must be > 0")


@dataclass
class TWIPParams:
    """Parameters for the planar two-wheeled inverted pendulum (TWIP) used in navigation.

    The longitudinal/balance subsystem is described by ``base`` (the same `RobotParams` the
    1-D controllers use), so existing balancing controllers are reused unchanged. The extra
    fields describe the yaw (heading) subsystem driven by the wheel-torque difference.
    """

    base: RobotParams = field(default_factory=RobotParams)
    track: float = 0.5     # lateral distance between the two wheels [m]
    I_yaw: float = 0.4     # yaw moment of inertia about the vertical axis [kg*m^2]
    b_yaw: float = 0.2     # yaw viscous damping [N*m*s/rad]

    def __post_init__(self) -> None:
        if self.track <= 0 or self.I_yaw <= 0:
            raise ValueError("track and I_yaw must be > 0")
        if self.b_yaw < 0:
            raise ValueError("b_yaw must be >= 0")


DEFAULT_PARAMS = RobotParams()
DEFAULT_SIM = SimConfig()
