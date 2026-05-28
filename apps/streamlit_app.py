"""Segway Control Suite — interactive web dashboard (Streamlit).

A clean, mode-based dashboard: pick **🛴 Balancing** or **🧭 Navigation** in the sidebar and the
whole app reconfigures to that workflow. Each workflow separates **Run** (headless results /
plots) from **👁 Watch** (a rendered 3-D MuJoCo animation), so the action is never ambiguous.
Balancing covers controller selection + parameter editing, Manual / Auto-Tune, noisy-sensor +
Kalman/EKF estimation, compare-all, and region of attraction; Navigation covers the composable
{balance × planner × follower} stack, preset and **custom** maps, terrain, and the full run
analysis. Shares one engine and parameter module (`apps/_common.py`) with the desktop GUI.

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
from segway.config import RobotParams, SimConfig, TWIPParams
from segway.controllers import REGULATORS, build_controller
from segway.navigation import (
    Obstacle,
    World,
    build_scenario,
    build_terrain,
    list_followers,
    list_planners,
    list_scenarios,
    list_terrains,
    navigate,
)
from segway.sim import Scenario, simulate
from segway.sim.scenarios import Disturbance
from segway.viz import plot_nav_analysis, plot_navigation, plot_roa

# Full-state controllers that can track a speed command (the tilt-only PID cannot drive).
NAV_BALANCE = ["lqr", "mpc", "smc", "hinf", "pole_placement", "cascaded_pid"]

st.set_page_config(page_title="Segway Control Suite", page_icon="🛴", layout="wide")
st.title("🛴 Segway Control Suite")
st.caption("A self-balancing wheeled inverted pendulum — classical to learned control, on one validated plant.")


@st.cache_data(show_spinner="Auto-tuning (cached per configuration)…")
def cached_tune(name: str, phys_items: tuple, tuner_label: str) -> dict:
    return tune(name, RobotParams(**dict(phys_items)), tuner_label)


# ===== sidebar: mode + configuration ====================================
sb = st.sidebar
sb.header("Workflow")
app_mode = sb.radio("workflow", ["🛴 Balancing", "🧭 Navigation"],
                    label_visibility="collapsed", horizontal=True)
is_nav = app_mode.startswith("🧭")

with sb.expander("🤖 Robot physical properties", expanded=False):
    m_base = st.number_input("Base mass (kg)", 0.1, 20.0, 1.0, 0.1)
    m_wheel = st.number_input("Wheel mass (kg)", 0.01, 5.0, 0.432, 0.01)
    m_pend = st.number_input("Body mass (kg)", 0.5, 30.0, 5.0, 0.5)
    I_pend = st.number_input("Inertia (kg·m²)", 0.01, 5.0, 0.2, 0.05)
    length = st.number_input("COM length (m)", 0.1, 1.5, 0.4, 0.05)
    radius = st.number_input("Wheel radius (m)", 0.02, 0.5, 0.1, 0.01)
    b_x = st.number_input("Damping slide", 0.0, 5.0, 0.05, 0.05)
    b_theta = st.number_input("Damping hinge", 0.0, 5.0, 0.2, 0.05)
params = RobotParams(m_base=m_base, m_wheel=m_wheel, m_pend=m_pend, I_pend=I_pend,
                     l=length, r=radius, b_x=b_x, b_theta=b_theta)

# Defaults so the balancing helpers never hit a NameError when in Navigation mode.
mode, tuner_label, controller = "Manual", "Optuna (TPE)", "lqr"
param_values: dict[str, str] = {}
use_est, est_kind, duration = False, "Kalman", 10.0
init_pos = init_tilt = init_rate = x_ref = 0.0
kicks_df = pd.DataFrame({"time (s)": [], "impulse (rad/s)": []})

if not is_nav:
    sb.divider()
    sb.subheader("🎛 Controller")
    mode = sb.radio("Tuning", ["Manual", "Auto-Tune"], horizontal=True)
    tuner_label = (sb.selectbox("Tuner", ["Optuna (TPE)", "Optuna (CMA-ES)", "Genetic Algorithm"])
                   if mode == "Auto-Tune" else "Optuna (TPE)")
    controller = sb.selectbox("Controller", controllers_available())
    sb.caption(DESCRIPTIONS.get(controller, ""))
    spec = PARAM_SPECS.get(controller, [])
    if not spec:
        sb.caption("(no tunable parameters)")
    elif mode == "Auto-Tune" and controller in TUNABLE:
        sb.caption("Parameters will be auto-tuned and shown in the result.")
    else:
        cols = sb.columns(2)
        for i, (label, key, default) in enumerate(spec):
            param_values[key] = cols[i % 2].text_input(label, default, key=f"p_{controller}_{key}")
    with sb.expander("🌍 Scenario & initial state", expanded=True):
        init_pos = st.number_input("Initial position (m)", -3.0, 3.0, 0.0, 0.1)
        init_tilt = st.slider("Initial tilt (rad)", 0.0, 1.2, 0.2, 0.05)
        init_rate = st.number_input("Initial tilt rate (rad/s)", -3.0, 3.0, 0.0, 0.1)
        x_ref = st.number_input("Position setpoint (m)", -3.0, 3.0, 0.0, 0.1)
        duration = st.slider("Duration (s)", 3.0, 20.0, 10.0, 1.0)
    use_est = sb.checkbox("Noisy sensors + filter")
    est_kind = sb.selectbox("Filter", ["Kalman", "EKF"]) if use_est else "Kalman"
    with sb.expander("💥 Disturbances (kicks)"):
        kicks_df = st.data_editor(
            pd.DataFrame({"time (s)": [4.0, 8.0], "impulse (rad/s)": [0.4, -0.4]}),
            num_rows="dynamic", use_container_width=True, key="kicks")
else:
    sb.divider()
    sb.subheader("🧭 Navigation stack")
    nav_balance = sb.selectbox("Balance controller", NAV_BALANCE,
                               help="The inner balance+speed loop (tilt-only PID can't drive).")
    nav_planner = sb.selectbox("Global planner", list_planners())
    nav_follower = sb.selectbox("Path follower", list_followers())
    nav_map = sb.selectbox("Map", [*list_scenarios(), "✏️ custom"])
    nav_backend = sb.selectbox("Backend (for Run)", ["analytic", "mujoco"],
                               help="Watch always runs full 3-D MuJoCo physics.")
    terrain_opts = ["none", *list_terrains()] if nav_backend == "mujoco" else ["none"]
    nav_terrain = sb.selectbox("Terrain", terrain_opts,
                               help="Uneven terrain requires the MuJoCo backend.")
    full_analysis = sb.checkbox("Full run analysis (speed / pitch / yaw / torque)")


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


# ===== main =============================================================
def _balancing_ui():
    import tempfile

    tab_run, tab_watch, tab_compare, tab_roa = st.tabs(
        ["▶ Run (results)", "👁 Watch in 3-D", "⚖ Compare all", "◌ Region of attraction"])
    swing = controller == "swingup"

    with tab_run:
        st.caption("Headless evaluation — response, metrics, and plots (no 3-D window).")
        est_on = use_est and not swing
        sensors, est = estimator() if est_on else (None, None)
        sim = SimConfig(duration=duration, fall_angle=50.0 if swing else 1.2,
                        control_dt=0.01 if est_on else None)
        traj = simulate(params, build(controller), make_scenario(swing), sim,
                        sensors=sensors, estimator=est)
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

    with tab_watch:
        st.caption("Render the controller balancing in full 3-D MuJoCo physics.")
        if st.button("🎬 Render & watch (≈15 s)", type="primary"):
            with st.spinner("Rendering MuJoCo rollout…"):
                try:
                    from segway.sim.mujoco_backend import CARTPOLE_PATH
                    from segway.viz import render_rollout
                    out = str(_Path(tempfile.gettempdir()) / "segway_watch_balance.gif")
                    st.session_state["watch_bal"] = render_rollout(
                        params, build(controller), make_scenario(swing),
                        SimConfig(duration=duration, fall_angle=50.0 if swing else 1.2),
                        path=out, xml_path=CARTPOLE_PATH if swing else None)
                except Exception as exc:
                    st.error(f"3-D rendering needs the sim + viz extras: {exc}")
        if "watch_bal" in st.session_state:
            st.image(st.session_state["watch_bal"], caption="MuJoCo 3-D rollout")

    with tab_compare:
        st.caption("Every regulator on the **same** scenario and physics.")
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
        st.caption("Which initial conditions can this controller recover from?")
        if st.button("Compute region of attraction (~15 s)"):
            with st.spinner("Sweeping initial conditions…"):
                res = compute_roa(params, build(controller), n_theta=21, n_thetadot=21)
                st.session_state["roa"] = (controller, res)
        if "roa" in st.session_state:
            cname, res = st.session_state["roa"]
            st.pyplot(plot_roa(res))
            st.caption(f"{cname}: recoverable area {res.area_fraction:.0%} (green = recovered, red = fell).")
        else:
            st.info("Click the button to sweep initial tilt / tilt-rate and map the recoverable region.")


def _nav_map_inputs():
    """Return (world, start, goal) for the chosen map, drawing the custom editor if needed."""
    if nav_map != "✏️ custom":
        sc = build_scenario(nav_map)
        return sc.world, sc.start, sc.goal
    st.markdown("**✏️ Custom map** — set the world size, edit the obstacle table, place start/goal.")
    w1, w2 = st.columns(2)
    ww = w1.number_input("World width (m)", 4.0, 30.0, 10.0, 1.0, key="cw")
    wh = w2.number_input("World height (m)", 4.0, 30.0, 8.0, 1.0, key="ch")
    obs_df = st.data_editor(
        pd.DataFrame({"x": [3.0, 6.0, 8.0], "y": [2.0, 5.0, 2.5], "r": [0.7, 0.8, 0.6]}),
        num_rows="dynamic", use_container_width=True, key="custom_obs")
    s1, s2, s3, s4 = st.columns(4)
    sx = s1.number_input("Start x", 0.0, float(ww), 1.0, 0.5, key="sx")
    sy = s2.number_input("Start y", 0.0, float(wh), 4.0, 0.5, key="sy")
    gx = s3.number_input("Goal x", 0.0, float(ww), float(ww) - 1.0, 0.5, key="gx")
    gy = s4.number_input("Goal y", 0.0, float(wh), 4.0, 0.5, key="gy")
    obstacles = [Obstacle(float(r.x), float(r.y), float(r.r))
                 for r in obs_df.itertuples() if pd.notna(r.x) and pd.notna(r.r)]
    world = World(width=float(ww), height=float(wh), resolution=0.1,
                  robot_radius=0.25, obstacles=obstacles)
    return world, (sx, sy), (gx, gy)


def _navigation_ui():
    import tempfile

    twip = TWIPParams(base=params)
    terrain = build_terrain(nav_terrain) if nav_terrain != "none" else None
    st.subheader(f"{nav_balance} · {nav_planner} · {nav_follower}")
    world, start, goal = _nav_map_inputs()

    tab_run, tab_watch = st.tabs(["🧭 Run (results)", "👁 Watch in 3-D"])
    with tab_run:
        st.caption("Plan a route and drive the balancing robot — top-down route and metrics.")
        if st.button("🧭 Plan & drive", type="primary"):
            spin = "Driving over terrain…" if terrain is not None else "Planning and driving…"
            with st.spinner(spin):
                st.session_state["nav"] = (
                    navigate(twip, world, start, goal, balance=nav_balance, planner=nav_planner,
                             follower=nav_follower, backend=nav_backend, terrain=terrain),
                    full_analysis)
        if "nav" in st.session_state:
            res, was_full = st.session_state["nav"]
            if not res.planned:
                st.error("The planner found no path — the goal may be unreachable (inside an obstacle).")
            else:
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("Reached", "Yes" if res.success else "No")
                g2.metric("Fell over", "Yes" if res.fell else "No")
                g3.metric("Time (s)", f"{res.time_to_goal:.1f}")
                g4.metric("Path length (m)", f"{res.path_length:.2f}")
                h1, h2 = st.columns(2)
                h1.metric("Min clearance (m)", f"{res.min_clearance:.3f}")
                h2.metric("Final goal dist (m)", f"{res.final_goal_distance:.3f}")
                st.pyplot(plot_nav_analysis(res) if was_full else plot_navigation(res).figure)
        else:
            st.info("Pick a stack and map in the sidebar, then click **Plan & drive**.")

    with tab_watch:
        st.caption("Render the robot driving the route in full 3-D MuJoCo physics (over terrain if set).")
        if st.button("🎬 Render & watch navigation (≈20 s)", type="primary"):
            with st.spinner("Rendering 3-D navigation…"):
                try:
                    from segway.viz import render_navigation
                    out = str(_Path(tempfile.gettempdir()) / "segway_watch_nav.gif")
                    st.session_state["watch_nav"] = render_navigation(
                        twip, world, start, goal, balance=nav_balance, planner=nav_planner,
                        follower=nav_follower, terrain=terrain, path=out)
                except Exception as exc:
                    st.error(f"3-D rendering needs the sim + viz extras: {exc}")
        if "watch_nav" in st.session_state:
            st.image(st.session_state["watch_nav"], caption="MuJoCo 3-D navigation")


if is_nav:
    _navigation_ui()
else:
    _balancing_ui()
