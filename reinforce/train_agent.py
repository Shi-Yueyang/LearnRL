from collections import deque
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from datetime import datetime

SAVE_FILE = f"best.pt"


@dataclass
class Config:
    env_id: str = "CartPole-v1"
    device: str = "auto"
    seed: int = 42
    episodes: int = 3000
    max_episode_steps: int = 500
    gamma: float = 0.99
    lr: float = 1e-3
    save_dir: str = "runs"
    save_every: int = 1000  # save model every N episodes
    log_interval: int = 200
    entropy_coef: float = 0.0  # small positive for more exploration (e.g. 0.01)
    normalize_returns: bool = True
    reward_to_go: bool = True  # if False uses full return for each timestep

    def as_dict(self):
        return self.__dict__.copy()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if name == "cuda" else "cpu")


class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


@dataclass
class Trajectory:
    obs: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    log_probs: List[torch.Tensor]


def make_env(env_id: str, seed: int, max_steps: Optional[int]):
    env = gym.make(env_id)
    if max_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    env.reset(seed=seed)
    return env


def compute_returns(
    rewards: List[float], gamma: float, reward_to_go: bool
) -> List[float]:
    if reward_to_go:
        # reward-to-go (future discounted return for each timestep)
        returns = []
        running = 0.0
        for r in reversed(rewards):
            running = r + gamma * running
            returns.append(running)
        returns.reverse()
        return returns
    # full episodic return for every timestep
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    return [G for _ in rewards]


def main():
    cfg = Config()
    device = select_device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    env = make_env(cfg.env_id, cfg.seed, cfg.max_episode_steps)
    obs_space = env.observation_space
    act_space = env.action_space
    obs_dim = obs_space.shape[0]
    n_actions = act_space.n

    best_mean_return = -1e9
    policy = PolicyNet(obs_dim, n_actions).to(device)
    try:
        ckpt = torch.load(
            os.path.join("runs", SAVE_FILE), map_location="cpu", weights_only=True
        )
        policy.load_state_dict(ckpt.get("model"))
        policy.eval()
        policy.to(device)
        best_mean_return = ckpt.get("mean_return", best_mean_return)
        print(f"Loaded model with mean return {best_mean_return:.2f}")
    except:
        pass
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    recent_returns = deque(maxlen=100)
    start_time = time.time()

    for episode in range(1, cfg.episodes + 1):
        obs, _ = env.reset()
        traj_obs: List[np.ndarray] = []
        traj_actions: List[int] = []
        traj_rewards: List[float] = []
        traj_logps: List[torch.Tensor] = []

        done = False
        steps = 0
        while not done:
            obs_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
            logits = policy(obs_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            logp = dist.log_prob(action)
            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            traj_obs.append(np.array(obs, dtype=np.float32))
            traj_actions.append(int(action.item()))
            traj_rewards.append(float(reward))
            traj_logps.append(logp)

            obs = next_obs
            steps += 1

        returns = compute_returns(traj_rewards, cfg.gamma, cfg.reward_to_go)
        returns_np = np.array(returns, dtype=np.float32)
        if cfg.normalize_returns:
            # standardize returns
            returns_np = (returns_np - returns_np.mean()) / (returns_np.std() + 1e-8)

        returns_tensor = torch.from_numpy(returns_np).to(device)
        logps_tensor = torch.stack(traj_logps).to(device)

        # Policy gradient loss (optionally add entropy bonus)
        pg_loss = -(logps_tensor * returns_tensor).sum()
        if cfg.entropy_coef > 0.0:
            with torch.no_grad():
                # recompute distribution for entropy; small overhead
                obs_batch = torch.from_numpy(np.vstack(traj_obs)).float().to(device)
                logits_batch = policy(obs_batch)
                probs_batch = F.softmax(logits_batch, dim=-1)
                entropy = (
                    torch.distributions.Categorical(probs=probs_batch).entropy().sum()
                )
            pg_loss = pg_loss - cfg.entropy_coef * entropy

        optimizer.zero_grad()
        pg_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()

        ep_return = float(sum(traj_rewards))
        recent_returns.append(ep_return)
        mean_return = float(np.mean(recent_returns))

        if episode % cfg.log_interval == 0:
            wall = time.time() - start_time
            print(
                f"Episode {episode}/{cfg.episodes} | Return {ep_return:.2f} | MeanReturn({len(recent_returns)}) {mean_return:.2f} | Steps {steps} | Time {wall:.1f}s"
            )

        # Save best
        if mean_return > best_mean_return and recent_returns:
            best_mean_return = mean_return
            print(f"New best mean return {best_mean_return:.2f}")
            torch.save(
                {
                    "model": policy.state_dict(),
                    "mean_return": mean_return,
                    "episode": episode,
                    "cfg": cfg.as_dict(),
                },
                os.path.join(cfg.save_dir, SAVE_FILE),
            )

        if episode % cfg.save_every == 0:
            torch.save(
                policy.state_dict(),
                os.path.join(cfg.save_dir, f"policy_ep_{episode}.pt"),
            )

    print(f"Training complete. Best mean return {best_mean_return:.2f}")


if __name__ == "__main__":  # pragma: no cover
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
