"""Train a PPO policy to balance the robot (optionally with domain randomization).

Usage:
    python scripts/train_rl.py --timesteps 200000 --randomize --out models/ppo_segway.zip

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
    ap = argparse.ArgumentParser(description="Train a PPO balancing policy")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--randomize", action="store_true", help="domain randomization")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/ppo_segway.zip")
    args = ap.parse_args(argv)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from segway.envs import make_env

    venv = DummyVecEnv([lambda: make_env(randomize=args.randomize) for _ in range(args.n_envs)])
    model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed, device="cpu",
                n_steps=1024, batch_size=256, gae_lambda=0.95, gamma=0.99, ent_coef=0.0)
    model.learn(total_timesteps=args.timesteps)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"saved policy -> {out}")

    # Quick evaluation on the analytic plant.
    from segway.config import RobotParams, SimConfig
    from segway.controllers import RLController
    from segway.sim import Scenario, simulate

    p = RobotParams()
    ctrl = RLController(p, model=model)
    for tilt in (0.1, 0.3):
        tr = simulate(p, ctrl, Scenario.balance(tilt), SimConfig(duration=8.0))
        print(f"eval balance({tilt}): fell={tr.fell}  final_theta={tr.theta[-1]:+.4f}  "
              f"max_drift={abs(tr.x).max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
