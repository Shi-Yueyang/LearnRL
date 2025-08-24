import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


@dataclass
class Config:
    env_id: str = "CartPole-v1"
    device: str = "auto"
    seed: int = 42
    max_episode_steps: int = 500
    model_path: str = os.path.join("runs", "best.pt")
    render_mode: str = "human"  # "rgb_array" for video capture


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if name == "cuda" else "cpu")


class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


def make_env(env_id: str, seed: int, max_steps: Optional[int], render_mode: str):
    env = gym.make(env_id, render_mode=render_mode)
    if max_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    env.reset(seed=seed)
    return env


def load_policy(cfg: Config, device: torch.device) -> PolicyNet:
    ckpt = torch.load(cfg.model_path, map_location=device)
    if isinstance(ckpt, dict) and "cfg" in ckpt and "model" in ckpt:
        obs_dim = 4  # CartPole
        n_actions = 2
        policy = PolicyNet(obs_dim, n_actions).to(device)
        policy.load_state_dict(ckpt["model"])
    else:
        # raw state dict
        obs_dim = 4
        n_actions = 2
        policy = PolicyNet(obs_dim, n_actions).to(device)
        policy.load_state_dict(ckpt)
    policy.eval()
    return policy


def run_episode(policy: PolicyNet, env, device: torch.device):
    obs, _ = env.reset()
    done = False
    ep_return = 0.0
    steps = 0
    while not done:
        obs_t = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = policy(obs_t)
            probs = F.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_return += reward
        obs = next_obs
        steps += 1
    return ep_return, steps


def main():
    cfg = Config()
    device = select_device(cfg.device)
    policy = load_policy(cfg, device)
    env = make_env(cfg.env_id, cfg.seed, cfg.max_episode_steps, cfg.render_mode)

    start = time.time()
    ret, steps = run_episode(policy, env, device)
    wall = time.time() - start
    print(f"Return {ret:.2f} | Steps {steps} | Time {wall:.2f}s")


if __name__ == "__main__":  # pragma: no cover
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
