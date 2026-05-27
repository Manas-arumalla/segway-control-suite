"""Segway Control Suite — interactive web dashboard (Streamlit).

Feature parity with the desktop GUI: pick a controller (with a description) and **edit its
parameters**, set the **full physical properties**, environment & initial state, a
**multi-kick disturbance list**, toggle **Manual / Auto-Tune** (with a tuner picker) and
**noisy-sensor + Kalman/EKF estimation** — then run a single simulation, **compare every
controller**, or map the **region of attraction**. Same engine and the same shared parameter
logic (`apps/_common.py`) as the desktop GUI. (Desktop-only: the live 3D viewer / GIF render,
which need a local display.)

Run:  streamlit run apps/streamlit_app.py        (needs the `dashboard` extra)
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Allow running from a source checkout, and make `_common` importable.
_src = _Path(__file__).resolve().parents[1] / "src"
if _src.is_dir() and str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))
_here = _Path(__file__).resolve().parent
if str(_here) not in _sys.path:
    _sys.path.insert(0, str(_here))

import numpy as np
import pandas as pd
import streamlit as st
from _common import (
    DESCRIPTIONS,
    PARAM_SPECS,
    TUNABLE,
    controllers_available,
    kwargs_from_values,
    tune,
)

from segway.analysis import compute_roa
from segway.config import RobotParams, SimConfig
from segway.controllers import REGULATORS, build_controller
from segway.sim import Scenario, simulate
from segway.sim.scenarios import Disturbance
from segway.viz import plot_roa

st.set_page_config(page_title="Segway Control Suite", page_icon="🛴", layout="wide")
st.title("🛴 Segway Control Suite")
st.caption("A self-balancing wheeled inverted pendulum — classical to learned control, on one validated plant.")


@st.cache_data(show_spinner="Auto-tuning (cached per configuration)…")
def cached_tune(name: str, phys_items: tuple, tuner_label: str) -> dict:
    return tune(name, RobotParams(**dict(phys_items)), tuner_label)


# ===== sidebar: configuration ===========================================
sb = st.sidebar
sb.header("Configuration")
mode = sb.radio("Mode", ["Manual", "Auto-Tune"], horizontal=True)
tuner_label = sb.selectbox("Tuner", ["Optuna (TPE)", "Optuna (CMA-ES)", "Genetic Algorithm"]) if mode == "Auto-Tune" else "Optuna (TPE)"
controller = sb.selectbox("Controller", controllers_available())
sb.caption(DESCRIPTIONS.get(controller, ""))

sb.subheader("Control parameters")
param_values: dict[str, str] = {}
spec = PARAM_SPECS.get(controller, [])
if not spec:
    sb.caption("(no tunable parameters)")
elif mode == "Auto-Tune" and controller in TUNABLE:
    sb.caption("Parameters will be auto-tuned and shown below.")
else:
    cols = sb.columns(2)
    for i, (label, key, default) in enumerate(spec):
        param_values[key] = cols[i % 2].text_input(label, default, key=f"p_{controller}_{key}")

with sb.expander("Robot physical properties"):
    m_base = st.number_input("Base mass (kg)", 0.1, 20.0, 1.0, 0.1)
    m_wheel = st.number_input("Wheel mass (kg)", 0.01, 5.0, 0.432, 0.01)
    m_pend = st.number_input("Body mass (kg)", 0.5, 30.0, 5.0, 0.5)
    I_pend = st.number_input("Inertia (kg·m²)", 0.01, 5.0, 0.2, 0.05)
    length = st.number_input("COM length (m)", 0.1, 1.5, 0.4, 0.05)
    radius = st.number_input("Wheel radius (m)", 0.02, 0.5, 0.1, 0.01)

with sb.expander("Environment & initial state", expanded=True):
    b_x = st.number_input("Damping slide", 0.0, 5.0, 0.05, 0.05)
    b_theta = st.number_input("Damping hinge", 0.0, 5.0, 0.2, 0.05)
    init_pos = st.number_input("Initial position (m)", -3.0, 3.0, 0.0, 0.1)
    init_tilt = st.slider("Initial tilt (rad)", 0.0, 1.2, 0.2, 0.05)
    init_rate = st.number_input("Initial tilt rate (rad/s)", -3.0, 3.0, 0.0, 0.1)
    x_ref = st.number_input("Position setpoint (m)", -3.0, 3.0, 0.0, 0.1)
    duration = st.slider("Duration (s)", 3.0, 20.0, 10.0, 1.0)

use_est = sb.checkbox("Noisy sensors + filter")
est_kind = sb.selectbox("Filter", ["Kalman", "EKF"]) if use_est else "Kalman"

with sb.expander("Disturbances (kicks)"):
    kicks_df = st.data_editor(
        pd.DataFrame({"time (s)": [4.0, 8.0], "impulse (rad/s)": [0.4, -0.4]}),
        num_rows="dynamic", use_container_width=True, key="kicks",
    )

params = RobotParams(m_base=m_base, m_wheel=m_wheel, m_pend=m_pend, I_pend=I_pend,
                     l=length, r=radius, b_x=b_x, b_theta=b_theta)


def make_scenario(swingup: bool) -> Scenario:
    if swingup:
        return Scenario(name="swingup", initial_tilt=float(np.pi), initial_tilt_rate=0.5)
    kicks = []
    for _, row in kicks_df.iterrows():
        t, imp = row.get("time (s)"), row.get("impulse (rad/s)")
        if pd.notna(t) and pd.notna(imp):
            kicks.append(Disturbance(time=float(t), impulse=float(imp)))
    return Scenario(name="dashboard", initial_pos=init_pos, initial_tilt=init_tilt,
                    initial_tilt_rate=init_rate, disturbances=kicks,
                    reference=(x_ref, 0.0, 0.0, 0.0))


def resolved_kwargs(name: str) -> dict:
    if mode == "Auto-Tune" and name in TUNABLE:
        kw = cached_tune(name, tuple(sorted(params.to_dict().items())), tuner_label)
        st.sidebar.success(f"Tuned: { {k: (round(v, 3) if isinstance(v, float) else v) for k, v in kw.items() if k != 'Q'} }")
        return kw
    if name in PARAM_SPECS and param_values:
        return kwargs_from_values(name, lambda k: param_values[k])
    return {}


def build(name: str):
    if name == "rl":
        from segway.controllers import RLController
        return RLController(params, model_path="models/ppo_segway.zip")
    return build_controller(name, params, **resolved_kwargs(name))


def estimator():
    if not use_est:
        return None, None
    from segway.estimation import ExtendedKalmanFilter, LinearKalmanFilter, SensorModel
    sensors = SensorModel()
    cls = ExtendedKalmanFilter if est_kind == "EKF" else LinearKalmanFilter
    return sensors, cls(params, dt=0.01, R=sensors.R)


# ===== main: results =====================================================
tab_run, tab_compare, tab_roa = st.tabs(["Single run", "Compare all", "Region of attraction"])

with tab_run:
    swing = controller == "swingup"
    est_on = use_est and not swing
    sensors, est = estimator() if est_on else (None, None)
    sim = SimConfig(duration=duration, fall_angle=50.0 if swing else 1.2,
                    control_dt=0.01 if est_on else None)
    traj = simulate(params, build(controller), make_scenario(swing), sim, sensors=sensors, estimator=est)
    m = traj.metrics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fell over", "Yes" if m["fell"] else "No")
    settle = m["settling_time_angle"]
    c2.metric("Settling θ (s)", "—" if settle == float("inf") else f"{settle:.2f}")
    c3.metric("Peak tilt (deg)", f"{m['peak_angle_deg']:.1f}")
    c4.metric("Max drift (m)", f"{m['max_pos_drift_m']:.2f}")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Control effort", f"{m['control_effort']:.2f}")
    d2.metric("Peak torque (N·m)", f"{m['peak_torque']:.1f}")
    d3.metric("RMS tilt (deg)", f"{np.degrees(m['rms_angle_rad']):.2f}")
    d4.metric("Final pos (m)", f"{m['final_pos_m']:.2f}")

    tilt = {"time (s)": traj.t, "tilt (deg)": np.degrees(traj.theta)}
    if traj.estimates is not None:
        tilt["estimate (deg)"] = np.degrees(traj.estimates[:, 2])
    st.line_chart(pd.DataFrame(tilt), x="time (s)", height=240)
    cc1, cc2 = st.columns(2)
    cc1.line_chart(pd.DataFrame({"time (s)": traj.t, "position (m)": traj.x}), x="time (s)", height=220)
    cc2.line_chart(pd.DataFrame({"time (s)": traj.t, "torque (N·m)": traj.controls}), x="time (s)", height=220)

with tab_compare:
    st.write("Every regulator on the **same** scenario and physics:")
    scen = make_scenario(False)
    rows, curves = [], {}
    for name in REGULATORS:
        tr = simulate(params, build_controller(name, params), scen, SimConfig(duration=duration))
        mm = tr.metrics()
        s = mm["settling_time_angle"]
        rows.append({"controller": name, "fell": mm["fell"],
                     "settle θ (s)": np.nan if s == float("inf") else round(s, 2),
                     "peak (deg)": round(mm["peak_angle_deg"], 1),
                     "effort": round(mm["control_effort"], 2),
                     "drift (m)": round(mm["max_pos_drift_m"], 3)})
        curves[name] = np.degrees(tr.theta)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    min_len = min(len(v) for v in curves.values())
    st.line_chart(pd.DataFrame({k: v[:min_len] for k, v in curves.items()}), height=320)
    st.caption("Tilt (deg) per recorded step for each controller.")

with tab_roa:
    st.write("Which initial conditions can this controller recover from?")
    if st.button("Compute region of attraction (~15 s)"):
        with st.spinner("Sweeping initial conditions…"):
            res = compute_roa(params, build(controller), n_theta=21, n_thetadot=21)
            st.session_state["roa"] = (controller, res)
    if "roa" in st.session_state:
        cname, res = st.session_state["roa"]
        st.pyplot(plot_roa(res))
        st.caption(f"{cname}: recoverable area {res.area_fraction:.0%} "
                   "(green = recovered, red = fell).")
    else:
        st.info("Click the button to sweep a grid of initial tilt / tilt-rate and map "
                "the recoverable region for the selected controller.")
