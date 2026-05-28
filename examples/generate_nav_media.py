"""Generate the navigation showcase media used in the README and docs.

Run from the repo root:  python examples/generate_nav_media.py
Outputs land in assets/ (a top-down map montage + a 3-D navigation GIF, and — if a trained
policy and MuJoCo are available — a terrain-traversal GIF).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Allow running directly from a source checkout (before `pip install -e .`).
_src = _Path(__file__).resolve().parents[1] / "src"
if _src.is_dir() and str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from segway.config import SimConfig, TWIPParams  # noqa: E402
from segway.navigation import Navigator, build_scenario  # noqa: E402
from segway.viz import plot_nav_analysis, plot_navigation  # noqa: E402

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)


def make_map_montage() -> None:
    """A 2x2 montage of top-down navigation on each preset map."""
    maps = ["corridor", "slalom", "rooms", "forest"]
    nav = Navigator(TWIPParams(), balance="lqr", planner="a_star", follower="pure_pursuit")
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, name in zip(axes.ravel(), maps, strict=True):
        sc = build_scenario(name)
        res = nav.run(sc.world, sc.start, sc.goal)
        plot_navigation(res, ax=ax, title=f"{name}  —  {'reached' if res.success else 'failed'}")
    fig.suptitle("Goal navigation while balancing (LQR · A* · pure pursuit)", fontsize=15)
    fig.tight_layout()
    fig.savefig(ASSETS / "nav_maps.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote assets/nav_maps.png")


def make_analysis_figures() -> None:
    """A full single-run analysis figure, plus a terrain-traversal analysis."""
    sc = build_scenario("slalom")
    nav = Navigator(TWIPParams(), balance="lqr", planner="a_star", follower="pure_pursuit")
    res = nav.run(sc.world, sc.start, sc.goal,
                  sim=SimConfig(dt=0.005, duration=30.0, fall_angle=1.2))
    plot_nav_analysis(res, path=str(ASSETS / "nav_analysis.png"))
    print("wrote assets/nav_analysis.png")

    try:
        from segway.navigation import Obstacle, World, build_terrain
        from segway.viz import plot_terrain_run
    except Exception as exc:  # pragma: no cover
        print(f"skipping terrain analysis ({exc})")
        return
    world = World(width=8.0, height=8.0, resolution=0.2, robot_radius=0.25,
                  obstacles=[Obstacle(4.0, 4.0, 0.6)])
    terr = build_terrain("moderate")
    tres = Navigator(TWIPParams(), backend="mujoco").run(
        world, (1.0, 4.0), (7.0, 4.0), terrain=terr,
        sim=SimConfig(dt=0.002, duration=40.0, fall_angle=1.2))
    plot_terrain_run(tres, terr, path=str(ASSETS / "nav_terrain_analysis.png"))
    print("wrote assets/nav_terrain_analysis.png")


def make_benchmark_figures() -> None:
    """Run the navigation sweeps + learned-vs-classical and save the summary figures."""
    import importlib.util

    from segway.viz import plot_benchmark_summary, plot_rl_analysis

    spec = importlib.util.spec_from_file_location(
        "run_nav", Path(__file__).resolve().parents[1] / "benchmarks" / "run_nav.py")
    run_nav = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_nav)

    maps = ["corridor", "slalom", "rooms", "forest"]
    rows = []
    for sweep, stacks in (("planner", run_nav.planner_stacks()),
                          ("follower", run_nav.follower_stacks()),
                          ("balance", run_nav.balance_stacks())):
        rows += run_nav.run_sweep(sweep, stacks, maps)
    plot_benchmark_summary(rows, path=str(ASSETS / "nav_benchmark.png"))
    print("wrote assets/nav_benchmark.png")

    model_path = Path("models/ppo_twip_nav.zip")
    if model_path.exists():
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path), device="cpu")
        rl_rows = run_nav.run_rl_vs_classical(model, TWIPParams())
        log = model_path.with_name("nav_train_log") / "progress.csv"
        plot_rl_analysis(rl_rows, learning_csv=log, path=str(ASSETS / "nav_rl.png"))
        print("wrote assets/nav_rl.png")
    else:
        print("skipping assets/nav_rl.png (no trained policy at models/ppo_twip_nav.zip)")


def make_3d_gifs() -> None:
    """3-D navigation GIFs (flat slalom + rough terrain) — needs MuJoCo + imageio."""
    try:
        from segway.navigation import build_terrain
        from segway.viz import render_navigation
    except Exception as exc:  # pragma: no cover
        print(f"skipping 3-D GIFs ({exc})")
        return

    sc = build_scenario("slalom")
    sim = SimConfig(dt=0.002, duration=30.0, fall_angle=1.2)
    render_navigation(TWIPParams(), sc.world, sc.start, sc.goal,
                      balance="lqr", planner="a_star", follower="pure_pursuit",
                      sim=sim, path=str(ASSETS / "nav_twip_3d.gif"),
                      width=560, height=340, fps=20)
    print("wrote assets/nav_twip_3d.gif")

    cor = build_scenario("corridor")
    render_navigation(TWIPParams(), cor.world, cor.start, cor.goal,
                      balance="lqr", planner="a_star", follower="pure_pursuit",
                      sim=SimConfig(dt=0.002, duration=20.0, fall_angle=1.2),
                      terrain=build_terrain("rough"),
                      path=str(ASSETS / "nav_terrain_3d.gif"),
                      width=560, height=340, fps=20)
    print("wrote assets/nav_terrain_3d.gif")


def main() -> None:
    make_map_montage()
    make_analysis_figures()
    make_benchmark_figures()
    make_3d_gifs()


if __name__ == "__main__":
    main()
