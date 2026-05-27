"""Segway Control Center — a polished desktop GUI (CustomTkinter).

A clean, organized control center for the whole suite. Configuration is grouped into tabs
(Controller · Robot · Scenario), a persistent action bar drives Run / View-3D / ROA /
Auto-Tune / Render / Compare, and results appear in tabbed panes (Response · Metrics · ROA)
with a progress bar and status line. Everything runs on background threads so the UI stays
responsive, and the two front-ends share one parameter module (`apps/_common.py`).

Carries over every capability of the original GUI — editable per-controller parameters,
full physical properties, environment & initial state, a multi-kick disturbance list,
Manual / Auto-Tune modes, and the live 3D viewer — and adds the new controllers, sensing +
Kalman/EKF estimation, multiple tuners, and a compare-all view.

Run:  python apps/desktop_gui.py        (needs the `gui` + `sim` + `viz` extras)
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Allow running directly from a source checkout, and make `_common` importable.
_src = _Path(__file__).resolve().parents[1] / "src"
if _src.is_dir() and str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))
_here = _Path(__file__).resolve().parent
if str(_here) not in _sys.path:
    _sys.path.insert(0, str(_here))

import threading
import tkinter as tk

import customtkinter as ctk
import numpy as np
from _common import (
    DESCRIPTIONS,
    PARAM_SPECS,
    TUNABLE,
    controllers_available,
    kwargs_from_values,
    populate_targets,
    tune,
)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from segway.analysis import compute_roa
from segway.config import RobotParams, SimConfig
from segway.controllers import REGULATORS, build_controller
from segway.sim import Scenario, simulate
from segway.sim.mujoco_backend import CARTPOLE_PATH
from segway.sim.scenarios import Disturbance

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")
ACCENT = "#00e5ff"
PANEL = "#2b2b2b"


class Tooltip:
    """Lightweight hover tooltip for any widget."""

    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, bg="#111", fg="#eee", font=("Segoe UI", 9),
                 padx=7, pady=4, justify="left", wraplength=280).pack()

    def _hide(self, _):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class SegwayGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Segway Control Center")
        self.geometry("1340x900")
        self.minsize(1180, 760)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.param_entries: dict[str, ctk.CTkEntry] = {}
        self.dist_rows: list[tuple] = []
        self.metric_labels: dict[str, ctk.CTkLabel] = {}
        self._busy = False

        self._build_config_panel()
        self._build_results_panel()
        self._rebuild_param_panel()
        self.add_kick("4.0", "0.4")
        self.add_kick("8.0", "-0.4")
        self._set_status("Ready — configure on the left, then Run Simulation.")

    # ===== left: configuration ==========================================
    def _build_config_panel(self):
        left = ctk.CTkFrame(self, width=440, corner_radius=0)
        left.grid(row=0, column=0, sticky="nsw")
        left.grid_propagate(False)
        left.grid_rowconfigure(1, weight=1)

        title = ctk.CTkFrame(left, fg_color="transparent")
        title.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        ctk.CTkLabel(title, text="🛴 Segway Control Center", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ctk.CTkLabel(title, text="Design · simulate · analyze · tune", text_color="#999",
                     font=("Segoe UI", 11)).pack(anchor="w")

        self.tabs = ctk.CTkTabview(left, width=410)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=6)
        for name in ("🎛 Controller", "🤖 Robot", "🌍 Scenario"):
            self.tabs.add(name)
        self._build_controller_tab(self.tabs.tab("🎛 Controller"))
        self._build_robot_tab(self.tabs.tab("🤖 Robot"))
        self._build_scenario_tab(self.tabs.tab("🌍 Scenario"))

        self._build_actions(left)

    def _scroll(self, tab):
        f = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        f.pack(fill="both", expand=True)
        return f

    def _build_controller_tab(self, tab):
        f = self._scroll(tab)
        seg = ctk.CTkSegmentedButton(f, values=["Manual", "Auto-Tune"], command=lambda _: self._on_mode())
        seg.set("Manual")
        seg.pack(fill="x", pady=(6, 2))
        self.mode = seg
        self.tuner = ctk.CTkOptionMenu(f, values=["Optuna (TPE)", "Optuna (CMA-ES)", "Genetic Algorithm"])
        self.tuner.set("Optuna (TPE)")

        ctk.CTkLabel(f, text="Strategy", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(10, 0))
        self.controller = ctk.CTkOptionMenu(f, values=controllers_available(),
                                            command=lambda _: self._rebuild_param_panel())
        self.controller.set("lqr")
        self.controller.pack(fill="x", pady=4)
        self.desc = ctk.CTkLabel(f, text="", text_color="#9bd", font=("Segoe UI", 11),
                                 wraplength=350, justify="left")
        self.desc.pack(anchor="w", pady=(0, 6))

        ctk.CTkLabel(f, text="Parameters", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(6, 0))
        self.param_frame = ctk.CTkFrame(f, fg_color=PANEL)
        self.param_frame.pack(fill="x", pady=4)

    def _build_robot_tab(self, tab):
        f = self._scroll(tab)
        ctk.CTkLabel(f, text="Physical properties", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(6, 2))
        grid = ctk.CTkFrame(f, fg_color=PANEL)
        grid.pack(fill="x")
        self.phys = {}
        specs = [("Base mass (kg)", "m_base", "1.0"), ("Wheel mass (kg)", "m_wheel", "0.432"),
                 ("Body mass (kg)", "m_pend", "5.0"), ("Inertia (kg·m²)", "I_pend", "0.2"),
                 ("COM length (m)", "l", "0.4"), ("Wheel radius (m)", "r", "0.1"),
                 ("Damping slide", "b_x", "0.05"), ("Damping hinge", "b_theta", "0.2")]
        for i, (label, key, default) in enumerate(specs):
            self.phys[key] = self._grid_entry(grid, label, default, i)
        ctk.CTkButton(f, text="↺ Reset to defaults", fg_color="#444", command=self._reset_robot).pack(anchor="w", pady=8)

    def _build_scenario_tab(self, tab):
        f = self._scroll(tab)
        ctk.CTkLabel(f, text="Initial state", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(6, 2))
        grid = ctk.CTkFrame(f, fg_color=PANEL)
        grid.pack(fill="x")
        self.env = {}
        specs = [("Initial position (m)", "init_pos", "0.0"), ("Initial tilt (rad)", "init_tilt", "0.2"),
                 ("Initial tilt rate", "init_rate", "0.0"), ("Position setpoint (m)", "x_ref", "0.0"),
                 ("Sim duration (s)", "duration", "10.0")]
        for i, (label, key, default) in enumerate(specs):
            self.env[key] = self._grid_entry(grid, label, default, i)

        ctk.CTkLabel(f, text="Disturbances (kicks)", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(12, 2))
        hdr = ctk.CTkFrame(f, fg_color="transparent")
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="time (s)", width=90, text_color="#999").pack(side="left", padx=4)
        ctk.CTkLabel(hdr, text="impulse (rad/s)", width=110, text_color="#999").pack(side="left", padx=4)
        self.dist_frame = ctk.CTkFrame(f, fg_color=PANEL)
        self.dist_frame.pack(fill="x", pady=2)
        ctk.CTkButton(f, text="+ Add kick", width=120, fg_color="#444",
                      command=lambda: self.add_kick()).pack(anchor="w", pady=6)

        ctk.CTkLabel(f, text="State estimation", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(12, 2))
        est = ctk.CTkFrame(f, fg_color=PANEL)
        est.pack(fill="x")
        self.use_est = ctk.CTkCheckBox(est, text="Noisy sensors + filter")
        self.use_est.pack(side="left", padx=10, pady=8)
        Tooltip(self.use_est, "Run the controller on noisy encoder+IMU measurements with a Kalman/EKF estimating the velocities.")
        self.est_kind = ctk.CTkOptionMenu(est, values=["Kalman", "EKF"], width=100)
        self.est_kind.set("Kalman")
        self.est_kind.pack(side="right", padx=10)

    def _build_actions(self, parent):
        bar = ctk.CTkFrame(parent, fg_color=PANEL)
        bar.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 10))
        self.btn_run = ctk.CTkButton(bar, text="▶  Run Simulation", height=42, font=("Segoe UI", 14, "bold"),
                                     fg_color=ACCENT, text_color="black", command=self.on_run)
        self.btn_run.pack(fill="x", padx=10, pady=(10, 6))
        row = ctk.CTkFrame(bar, fg_color="transparent")
        row.pack(fill="x", padx=10)
        self.btn_view = ctk.CTkButton(row, text="🧊 View 3D", width=120, command=self.on_view)
        self.btn_roa = ctk.CTkButton(row, text="◌ ROA", width=120, command=self.on_roa)
        self.btn_tune = ctk.CTkButton(row, text="✦ Tune", width=120, command=self.on_tune)
        self.btn_view.pack(side="left", expand=True, fill="x", padx=2)
        self.btn_roa.pack(side="left", expand=True, fill="x", padx=2)
        self.btn_tune.pack(side="left", expand=True, fill="x", padx=2)
        row2 = ctk.CTkFrame(bar, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(6, 4))
        self.btn_cmp = ctk.CTkButton(row2, text="⚖ Compare all", command=self.on_compare)
        self.btn_render = ctk.CTkButton(row2, text="🎬 Render GIF", command=self.on_render)
        self.btn_cmp.pack(side="left", expand=True, fill="x", padx=2)
        self.btn_render.pack(side="left", expand=True, fill="x", padx=2)
        self._buttons = [self.btn_run, self.btn_view, self.btn_roa, self.btn_tune, self.btn_cmp, self.btn_render]
        Tooltip(self.btn_view, "Open an interactive live MuJoCo 3D window.")
        Tooltip(self.btn_roa, "Sweep initial conditions and map the recoverable region.")
        Tooltip(self.btn_tune, "Auto-tune the controller's parameters (Manual mode shows the result).")
        Tooltip(self.btn_cmp, "Run every regulator on this scenario and overlay the responses.")

        self.progress = ctk.CTkProgressBar(bar, mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=(4, 8))
        self.progress.set(0)

    # ===== right: results ===============================================
    def _build_results_panel(self):
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.rtabs = ctk.CTkTabview(right)
        self.rtabs.grid(row=0, column=0, sticky="nsew")
        for name in ("📈 Response", "🧮 Metrics", "◌ Region of Attraction"):
            self.rtabs.add(name)

        resp = self.rtabs.tab("📈 Response")
        resp.grid_rowconfigure(0, weight=1)
        resp.grid_columnconfigure(0, weight=1)
        self.fig = Figure(figsize=(7.4, 6.6), facecolor="#242424")
        self.canvas = FigureCanvasTkAgg(self.fig, master=resp)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self._build_metrics_tab(self.rtabs.tab("🧮 Metrics"))

        roa = self.rtabs.tab("◌ Region of Attraction")
        roa.grid_rowconfigure(0, weight=1)
        roa.grid_columnconfigure(0, weight=1)
        self.roa_fig = Figure(figsize=(7.0, 6.0), facecolor="#242424")
        self.roa_canvas = FigureCanvasTkAgg(self.roa_fig, master=roa)
        self.roa_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.status = ctk.CTkLabel(right, text="", anchor="w", font=("Consolas", 12),
                                   wraplength=820, justify="left")
        self.status.grid(row=1, column=0, sticky="ew", padx=8, pady=(6, 2))
        self._blank_plot()

    def _build_metrics_tab(self, tab):
        cards = ["fell", "settling", "peak", "effort", "drift", "peak torque"]
        for i in range(3):
            tab.grid_columnconfigure(i, weight=1)
        for idx, key in enumerate(cards):
            r, c = idx // 3, idx % 3
            card = ctk.CTkFrame(tab, fg_color=PANEL, corner_radius=10)
            card.grid(row=r, column=c, padx=10, pady=10, sticky="nsew")
            val = ctk.CTkLabel(card, text="—", font=("Segoe UI", 26, "bold"), text_color=ACCENT)
            val.pack(pady=(18, 2))
            ctk.CTkLabel(card, text=key.upper(), text_color="#999", font=("Segoe UI", 11)).pack(pady=(0, 16))
            self.metric_labels[key] = val

    # ===== widget helpers ===============================================
    def _grid_entry(self, parent, label, default, idx):
        r, c = idx // 2, (idx % 2) * 2
        ctk.CTkLabel(parent, text=label, font=("Segoe UI", 11)).grid(row=r, column=c, padx=(8, 4), pady=5, sticky="w")
        e = ctk.CTkEntry(parent, width=86)
        e.insert(0, default)
        e.grid(row=r, column=c + 1, padx=(0, 8), pady=5)
        return e

    def _rebuild_param_panel(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        self.param_entries = {}
        name = self.controller.get()
        self.desc.configure(text=DESCRIPTIONS.get(name, ""))
        spec = PARAM_SPECS.get(name, [])
        if not spec:
            ctk.CTkLabel(self.param_frame, text="(no tunable parameters)", text_color="#888").pack(padx=8, pady=10)
        for i, (label, key, default) in enumerate(spec):
            self.param_entries[key] = self._grid_entry(self.param_frame, label, default, i)
        self._on_mode()

    def _on_mode(self):
        auto = self.mode.get() == "Auto-Tune"
        self.tuner.pack(fill="x", pady=4) if auto else self.tuner.pack_forget()
        for e in self.param_entries.values():
            e.configure(state="disabled" if auto else "normal")

    def _reset_robot(self):
        defaults = {"m_base": "1.0", "m_wheel": "0.432", "m_pend": "5.0", "I_pend": "0.2",
                    "l": "0.4", "r": "0.1", "b_x": "0.05", "b_theta": "0.2"}
        for k, v in defaults.items():
            self.phys[k].delete(0, "end")
            self.phys[k].insert(0, v)
        self._set_status("Robot parameters reset to defaults.")

    def add_kick(self, time="", impulse=""):
        row = ctk.CTkFrame(self.dist_frame, fg_color="transparent")
        row.pack(fill="x", pady=2)
        e_t = ctk.CTkEntry(row, width=90)
        e_i = ctk.CTkEntry(row, width=110)
        if time:
            e_t.insert(0, time)
        if impulse:
            e_i.insert(0, impulse)
        e_t.pack(side="left", padx=4)
        e_i.pack(side="left", padx=4)
        entry = (e_t, e_i, row)

        def remove():
            row.destroy()
            self.dist_rows.remove(entry)

        ctk.CTkButton(row, text="✕", width=28, fg_color="#7a2e2e", hover_color="#9e3a3a",
                      command=remove).pack(side="left", padx=4)
        self.dist_rows.append(entry)

    # ===== parse config =================================================
    def _params(self):
        p = {k: float(e.get()) for k, e in self.phys.items()}
        return RobotParams(m_base=p["m_base"], m_wheel=p["m_wheel"], m_pend=p["m_pend"],
                           I_pend=p["I_pend"], l=p["l"], r=p["r"], b_x=p["b_x"], b_theta=p["b_theta"])

    def _scenario(self, swingup=False):
        if swingup:
            return Scenario(name="swingup", initial_tilt=float(np.pi), initial_tilt_rate=0.5)
        kicks = [Disturbance(time=float(t.get()), impulse=float(i.get()))
                 for t, i, _ in self.dist_rows if t.get().strip() and i.get().strip()]
        return Scenario(name="gui", initial_pos=float(self.env["init_pos"].get()),
                        initial_tilt=float(self.env["init_tilt"].get()),
                        initial_tilt_rate=float(self.env["init_rate"].get()),
                        disturbances=kicks, reference=(float(self.env["x_ref"].get()), 0.0, 0.0, 0.0))

    def _build(self, name, params):
        if name == "rl":
            from segway.controllers import RLController
            return RLController(params, model_path="models/ppo_segway.zip")
        kwargs = {} if self.mode.get() == "Auto-Tune" else kwargs_from_values(name, lambda k: self.param_entries[k].get())
        return build_controller(name, params, **kwargs)

    def _populate(self, name, kw):
        for key, value in populate_targets(name, kw).items():
            if key in self.param_entries:
                self.param_entries[key].delete(0, "end")
                self.param_entries[key].insert(0, value)

    def _estimator(self, params):
        if not self.use_est.get():
            return None, None
        from segway.estimation import ExtendedKalmanFilter, LinearKalmanFilter, SensorModel
        sensors = SensorModel()
        cls = ExtendedKalmanFilter if self.est_kind.get() == "EKF" else LinearKalmanFilter
        return sensors, cls(params, dt=0.01, R=sensors.R)

    def _duration(self):
        return float(self.env["duration"].get())

    # ===== threaded actions =============================================
    def _run_bg(self, fn):
        if self._busy:
            return
        self._busy = True
        for b in self._buttons:
            b.configure(state="disabled")
        self.progress.start()

        def worker():
            try:
                fn()
            except Exception as exc:
                msg = str(exc)
                self.after(0, lambda m=msg: self._set_status(f"⚠ Error: {m}"))
            finally:
                self.after(0, self._done)

        threading.Thread(target=worker, daemon=True).start()

    def _done(self):
        self._busy = False
        self.progress.stop()
        self.progress.set(0)
        for b in self._buttons:
            b.configure(state="normal")

    def on_run(self):
        name = self.controller.get()
        self._set_status("Simulating…")

        def task():
            p = self._params()
            swing = name == "swingup"
            if self.mode.get() == "Auto-Tune" and name in TUNABLE:
                self.after(0, lambda: self._set_status(f"Auto-tuning {name}…"))
                kw = tune(name, p, self.tuner.get())
                self.after(0, lambda: self._populate(name, kw))
                ctrl = build_controller(name, p, **kw)
            else:
                ctrl = self._build(name, p)
            est_on = self.use_est.get() and not swing
            sensors, est = self._estimator(p) if est_on else (None, None)
            sim = SimConfig(duration=self._duration(), fall_angle=50.0 if swing else 1.2,
                            control_dt=0.01 if est_on else None)
            traj = simulate(p, ctrl, self._scenario(swingup=swing), sim, sensors=sensors, estimator=est)
            self.after(0, lambda: self._plot_traj(traj))
        self._run_bg(task)

    def on_view(self):
        name = self.controller.get()
        self._set_status("Opening live 3D viewer (close its window to return)…")

        def task():
            from segway.viz import live_view
            p, swing = self._params(), self.controller.get() == "swingup"
            sim = SimConfig(duration=120.0, fall_angle=50.0 if swing else 1.2)
            live_view(p, self._build(name, p), self._scenario(swingup=swing), sim,
                      xml_path=CARTPOLE_PATH if swing else None)
            self.after(0, lambda: self._set_status("Live viewer closed."))
        self._run_bg(task)

    def on_roa(self):
        name = self.controller.get()
        self._set_status("Computing region of attraction…")

        def task():
            p = self._params()
            res = compute_roa(p, self._build(name, p), n_theta=21, n_thetadot=21)
            self.after(0, lambda: self._plot_roa(res))
        self._run_bg(task)

    def on_tune(self):
        name = self.controller.get()
        if name not in TUNABLE:
            self._set_status(f"Auto-tune supports: {', '.join(sorted(TUNABLE))}.")
            return
        self._set_status(f"Auto-tuning {name}…")

        def task():
            p = self._params()
            kw = tune(name, p, self.tuner.get())
            traj = simulate(p, build_controller(name, p, **kw), self._scenario(), SimConfig(duration=self._duration()))
            self.after(0, lambda: (self._populate(name, kw), self._plot_traj(traj),
                                   self._set_status(f"Tuned {name}: {kw}")))
        self._run_bg(task)

    def on_compare(self):
        self._set_status("Comparing all regulators…")

        def task():
            p, scen = self._params(), self._scenario()
            trajs = {n: simulate(p, build_controller(n, p), scen, SimConfig(duration=self._duration()))
                     for n in REGULATORS}
            self.after(0, lambda: self._plot_compare(trajs))
        self._run_bg(task)

    def on_render(self):
        name = self.controller.get()
        self._set_status("Rendering MuJoCo GIF → assets/gui_render.gif …")

        def task():
            from segway.viz import render_rollout
            p, swing = self._params(), self.controller.get() == "swingup"
            out = render_rollout(p, self._build(name, p), self._scenario(swingup=swing),
                                 SimConfig(duration=self._duration(), fall_angle=50.0 if swing else 1.2),
                                 path="assets/gui_render.gif", xml_path=CARTPOLE_PATH if swing else None)
            self.after(0, lambda: self._set_status(f"Saved {out}"))
        self._run_bg(task)

    # ===== plotting =====================================================
    def _blank_plot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#1e1e1e")
        ax.text(0.5, 0.5, "Configure on the left → Run Simulation", ha="center", va="center",
                color="#888", fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()

    def _style(self, ax, fig):
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="#cccccc")
        for s in ax.spines.values():
            s.set_color("#555")
        ax.grid(alpha=0.2)

    def _plot_traj(self, traj):
        self.fig.clear()
        axes = self.fig.subplots(3, 1, sharex=True)
        axes[0].plot(traj.t, np.degrees(traj.theta), color="#ff6b6b", lw=2, label="true")
        if traj.estimates is not None:
            axes[0].plot(traj.t, np.degrees(traj.estimates[:, 2]), color=ACCENT, lw=1.2, ls="--", label="estimate")
            axes[0].legend(fontsize=8)
        axes[0].axhline(0, color="#888", ls="--", lw=0.8)
        axes[0].set_ylabel("Tilt (deg)", color="#ccc")
        axes[1].plot(traj.t, traj.x, color=ACCENT, lw=2)
        axes[1].set_ylabel("Position (m)", color="#ccc")
        axes[2].plot(traj.t, traj.controls, color="#51cf66", lw=2)
        axes[2].set_ylabel("Torque (N·m)", color="#ccc")
        axes[2].set_xlabel("Time (s)", color="#ccc")
        for ax in axes:
            self._style(ax, self.fig)
        self.fig.tight_layout()
        self.canvas.draw()
        self._update_metrics(traj.metrics())
        self.rtabs.set("📈 Response")
        m = traj.metrics()
        st = m["settling_time_angle"]
        st = "never" if st == float("inf") else f"{st:.2f}s"
        self._set_status(f"{traj.controller_name} | fell={m['fell']} | settle θ={st} | "
                         f"peak={m['peak_angle_deg']:.1f}° | effort={m['control_effort']:.2f}")

    def _plot_compare(self, trajs):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        for name, tr in trajs.items():
            ax.plot(tr.t, np.degrees(tr.theta), lw=1.8, label=name)
        ax.axhline(0, color="#888", ls="--", lw=0.8)
        ax.set_xlabel("Time (s)", color="#ccc")
        ax.set_ylabel("Tilt (deg)", color="#ccc")
        ax.legend(ncol=2, fontsize=8)
        self._style(ax, self.fig)
        self.fig.tight_layout()
        self.canvas.draw()
        self.rtabs.set("📈 Response")
        self._set_status("Compared all regulators on the current scenario (tilt overlay).")

    def _update_metrics(self, m):
        st = m["settling_time_angle"]
        vals = {
            "fell": "Yes" if m["fell"] else "No",
            "settling": "—" if st == float("inf") else f"{st:.2f}s",
            "peak": f"{m['peak_angle_deg']:.1f}°",
            "effort": f"{m['control_effort']:.2f}",
            "drift": f"{m['max_pos_drift_m']:.2f}m",
            "peak torque": f"{m['peak_torque']:.1f}",
        }
        for k, v in vals.items():
            self.metric_labels[k].configure(text=v)

    def _plot_roa(self, res):
        self.roa_fig.clear()
        ax = self.roa_fig.add_subplot(111)
        ax.contourf(res.thetadot_vals, res.theta_vals, res.grid.astype(float),
                    levels=[-0.5, 0.5, 1.5], colors=["#e63946", "#2a9d8f"], alpha=0.85)
        ax.set_xlabel("Initial θ̇ (rad/s)", color="#ccc")
        ax.set_ylabel("Initial θ (rad)", color="#ccc")
        self._style(ax, self.roa_fig)
        self.roa_fig.tight_layout()
        self.roa_canvas.draw()
        self.rtabs.set("◌ Region of Attraction")
        self._set_status(f"{res.controller_name} region of attraction — recoverable area {res.area_fraction:.0%}")

    def _set_status(self, text):
        self.status.configure(text=text)


def main():
    SegwayGUI().mainloop()


if __name__ == "__main__":
    main()
