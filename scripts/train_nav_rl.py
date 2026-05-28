"""Train a goal-conditioned PPO navigation policy for the balancing TWIP (NAV-7).

Usage:
    python scripts/train_nav_rl.py --timesteps 400000 --randomize --out models/ppo_twip_nav.zip

Requires the ``rl`` extra (Gymnasium + Stable-Baselines3).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Allow running directly from a source checkout (before `pip install -e .`).
_src = _Path(__file__).resolve().parents[1] / "src"
if _src.is_dir() and str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

import argparse
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train a PPO TWIP navigation policy")
    ap.add_argument("--timesteps", type=int, default=400_000)
    ap.add_argument("--randomize", action="store_true", help="domain randomization")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/ppo_twip_nav.zip")
    args = ap.parse_args(argv)

    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from segway.envs import make_nav_env

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Monitor wraps each env so episode reward/length are logged (the learning curve).
    venv = DummyVecEnv([lambda: Monitor(make_nav_env(randomize=args.randomize))
                        for _ in range(args.n_envs)])
    model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed, device="cpu",
                n_steps=1024, batch_size=256, gae_lambda=0.95, gamma=0.995, ent_coef=0.0)
    # Log progress to CSV next to the model so the learning curve can be plotted later.
    log_dir = out.with_name("nav_train_log")
    model.set_logger(configure(str(log_dir), ["stdout", "csv"]))
    model.learn(total_timesteps=args.timesteps)

    model.save(str(out))
    print(f"saved policy -> {out}  (training log -> {log_dir / 'progress.csv'})")

    # Quick evaluation: drive to a handful of goals on the nominal robot.
    from segway.config import TWIPParams
    from segway.navigation import rl_navigate

    p = TWIPParams()
    reached = 0
    goals = [(2.5, 0.0), (0.0, 2.5), (-2.0, 1.5), (1.8, -1.8), (3.0, 1.0)]
    for g in goals:
        res = rl_navigate(model, p, (0.0, 0.0), g)
        reached += int(res.success)
        print(f"goal {g}: reached={res.success}  final_dist={res.final_goal_distance:.2f}  "
              f"fell={res.fell}")
    print(f"reached {reached}/{len(goals)} goals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
