"""Shared configuration/parameter helpers for the desktop GUI and the Streamlit dashboard.

Keeping the per-controller parameter specs and the value→kwargs mapping in one place means
the two front-ends always expose the *same* controls and behave identically.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Make `segway` importable from a source checkout regardless of how the app is launched.
_src = _Path(__file__).resolve().parents[1] / "src"
if _src.is_dir() and str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

# Editable parameters per controller: (label, key, default).
PARAM_SPECS: dict[str, list[tuple[str, str, str]]] = {
    "lqr": [("Q x", "q0", "10"), ("Q ẋ", "q1", "1"), ("Q θ", "q2", "100"), ("Q θ̇", "q3", "1"), ("R", "R", "0.1")],
    "mpc": [("Q x", "q0", "10"), ("Q ẋ", "q1", "1"), ("Q θ", "q2", "100"), ("Q θ̇", "q3", "1"),
            ("R", "R", "0.1"), ("Horizon", "horizon", "20")],
    "pole_placement": [("Pole 1", "p0", "-1.2+0.9j"), ("Pole 2", "p1", "-1.2-0.9j"),
                       ("Pole 3", "p2", "-10"), ("Pole 4", "p3", "-25")],
    "smc": [("λ x", "l0", "2.0"), ("λ θ", "l2", "6.0"), ("λ θ̇", "l3", "1.0"),
            ("Gain K", "K", "25"), ("Phi", "phi", "0.1")],
    "pid": [("Kp", "kp", "120"), ("Ki", "ki", "10"), ("Kd", "kd", "20")],
    "cascaded_pid": [("Kp", "kp", "140"), ("Kd", "kd", "45"), ("Kx", "kx", "-0.2"), ("Kv", "kv", "-0.45")],
    "hinf": [("γ margin", "gamma_margin", "2.0")],
    "mrac": [("γ adapt", "gamma", "6.0"), ("σ leak", "sigma", "0.4")],
    "swingup": [("k_accel", "k_accel", "4.0"), ("Switch angle", "switch_angle", "0.5")],
    "rl": [],
}
TUNABLE = {"lqr", "mpc", "pid", "cascaded_pid", "smc", "pole_placement"}

# One-line descriptions shown in both UIs.
DESCRIPTIONS = {
    "pid": "Classic tilt PID. Balances upright but drifts in position (no position feedback).",
    "cascaded_pid": "Outer position loop → inner tilt loop. Balances AND holds position.",
    "pole_placement": "Places the closed-loop poles directly. Robustness depends on the poles.",
    "lqr": "Optimal state feedback minimizing a quadratic cost. Strong all-rounder.",
    "mpc": "Receding-horizon optimization with input limits (CVXPY/OSQP).",
    "smc": "Sliding-mode control — robust & nonlinear, with a boundary layer vs chattering.",
    "hinf": "H-infinity — minimizes the worst-case disturbance-to-output gain.",
    "mrac": "Adaptive (MRAC) — augments LQR and adapts online to model error.",
    "swingup": "Energy swing-up from hanging + LQR catch (shown on the cart-pole view).",
    "rl": "Learned PPO policy, trained with domain randomization.",
}


def controllers_available() -> list[str]:
    from segway.controllers import list_controllers
    names = list(list_controllers())
    if (_Path("models/ppo_segway.zip")).exists():
        names.append("rl")
    return names


def kwargs_from_values(name: str, getval) -> dict:
    """Map a controller's editable field values (via ``getval(key) -> str``) to kwargs."""
    def f(key):
        return float(getval(key))

    if name in ("lqr", "mpc"):
        kw = {"Q": [f(f"q{i}") for i in range(4)], "R": f("R")}
        if name == "mpc":
            kw["horizon"] = int(f("horizon"))
        return kw
    if name == "pole_placement":
        return {"poles": [complex(getval(f"p{i}")) for i in range(4)]}
    if name == "smc":
        return {"lam": [f("l0"), 1.0, f("l2"), f("l3")], "K": f("K"), "phi": f("phi")}
    if name == "pid":
        return {"kp": f("kp"), "ki": f("ki"), "kd": f("kd")}
    if name == "cascaded_pid":
        return {"kp": f("kp"), "kd": f("kd"), "kx": f("kx"), "kv": f("kv")}
    if name == "hinf":
        return {"gamma_margin": f("gamma_margin")}
    if name == "mrac":
        return {"gamma": f("gamma"), "sigma": f("sigma")}
    if name == "swingup":
        return {"k_accel": f("k_accel"), "switch_angle": f("switch_angle")}
    return {}


def populate_targets(name: str, kw: dict) -> dict[str, str]:
    """Inverse of :func:`kwargs_from_values`: kwargs -> {field key: display string}."""
    out: dict[str, str] = {}
    if name in ("lqr", "mpc"):
        for i in range(4):
            out[f"q{i}"] = f"{kw['Q'][i]:.4g}"
        out["R"] = f"{kw['R']:.4g}"
        if name == "mpc" and "horizon" in kw:
            out["horizon"] = str(kw["horizon"])
    elif name == "smc":
        out["l0"] = f"{kw['lam'][0]:.4g}"
        out["l2"] = f"{kw['lam'][2]:.4g}"
        out["l3"] = f"{kw['lam'][3]:.4g}"
        out["K"] = f"{kw['K']:.4g}"
        out["phi"] = f"{kw['phi']:.4g}"
    elif name == "pid":
        for k in ("kp", "ki", "kd"):
            out[k] = f"{kw[k]:.4g}"
    elif name == "cascaded_pid":
        for k in ("kp", "kd", "kx", "kv"):
            out[k] = f"{kw[k]:.4g}"
    elif name == "pole_placement":
        for i in range(4):
            out[f"p{i}"] = f"{kw['poles'][i]:.4g}"
    return out


def tune(name: str, params, tuner_label: str = "Optuna (TPE)", n_trials: int = 45) -> dict:
    """Run the chosen optimizer and return the best controller kwargs."""
    if "Genetic" in tuner_label:
        from segway.tuning import ga_tune
        return ga_tune(name, params=params, pop_size=25, ngen=12).best_kwargs
    from segway.tuning import optuna_tune
    sampler = "cmaes" if "CMA" in tuner_label else "tpe"
    return optuna_tune(name, params=params, n_trials=n_trials, sampler=sampler).best_kwargs
