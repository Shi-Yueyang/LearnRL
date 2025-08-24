import os
import time
from dataclasses import dataclass
from typing import Deque, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym  # Gymnasium-only (simplified, no legacy gym fallback)


@dataclass
class Config:
    # Hard-coded training configuration (was previously provided by argparse)
    env_id: str = "CartPole-v1"
    device: str = "auto"
    seed: int = 42
    updates: int = 1000
    episodes_per_update: int = 10
    batch_envs: int = 1
    max_episode_steps: int = 500
    gamma: float = 0.99
    lr: float = 1e-3
    buffer_size: int = 50_000
    batch_size: int = 128
    start_eps: float = 1.0
    end_eps: float = 0.05
    eps_decay_updates: int = 500
    target_update: int = 10
    train_iters_per_update: int = 50
    min_buffer: int = 1000
    save_dir: str = "runs"
    save_every: int = 1000

    def as_dict(self):
        return self.__dict__.copy()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if name == "cuda" else "cpu")


class MLP(nn.Module):
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


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.idx = 0
        self.full = False

    def add(self, tr: Transition):
        i = self.idx
        self.obs[i] = tr.obs
        self.next_obs[i] = tr.next_obs
        self.actions[i] = tr.action
        self.rewards[i] = tr.reward
        self.dones[i] = tr.done
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int):
        size = len(self)
        idxs = np.random.randint(0, size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idxs]),
            torch.from_numpy(self.actions[idxs]),
            torch.from_numpy(self.rewards[idxs]),
            torch.from_numpy(self.next_obs[idxs]),
            torch.from_numpy(self.dones[idxs].astype(np.float32)),
        )


def make_envs(env_id: str, batch_envs: int, seed: int, max_steps: Optional[int]):
    """Create (vector) environment(s) using Gymnasium only.

    Always returns an environment whose reset() returns (obs, info) and step()
    returns (obs, reward, terminated, truncated, info).
    """
    if batch_envs > 1:

        def env_fn():
            e = gym.make(env_id)
            if max_steps:
                e = gym.wrappers.TimeLimit(e, max_episode_steps=max_steps)
            e.reset(seed=seed)
            return e

        return gym.vector.AsyncVectorEnv([env_fn for _ in range(batch_envs)])
    env = gym.make(env_id)
    if max_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    env.reset(seed=seed)
    return env


def linear_epsilon(step: int, start: float, end: float, decay_updates: int) -> float:
    if decay_updates <= 0:
        return end
    t = min(step / decay_updates, 1.0)
    return start + (end - start) * t


def main():
    cfg = Config()
    device = select_device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_envs(cfg.env_id, cfg.batch_envs, cfg.seed, cfg.max_episode_steps)
    if cfg.batch_envs > 1:
        single_obs_space = env.single_observation_space
        single_act_space = env.single_action_space
    else:
        single_obs_space = env.observation_space
        single_act_space = env.action_space

    assert (
        len(single_obs_space.shape) == 1
    ), "This minimal DQN only supports flat observation spaces"
    assert hasattr(
        single_act_space, "n"
    ), "This minimal DQN only supports discrete action spaces"

    obs_dim = single_obs_space.shape[0]
    n_actions = single_act_space.n
    policy = MLP(obs_dim, n_actions).to(device)
    target = MLP(obs_dim, n_actions).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    buffer = ReplayBuffer(cfg.buffer_size, (obs_dim,))

    update = 0
    best_mean_return = -1e9

    # Initialize observations (Gymnasium API)
    obs, _ = env.reset(seed=cfg.seed)

    episode_returns = np.zeros(
        cfg.batch_envs if cfg.batch_envs > 1 else 1, dtype=np.float32
    )
    episode_lengths = np.zeros_like(episode_returns)
    recent_returns: Deque[float] = Deque(maxlen=100)

    start_time = time.time()

    while update < cfg.updates:
        episodes_to_collect = cfg.episodes_per_update
        collected_episodes = 0
        while collected_episodes < episodes_to_collect:
            eps = linear_epsilon(
                update, cfg.start_eps, cfg.end_eps, cfg.eps_decay_updates
            )
            if cfg.batch_envs > 1:
                obs_tensor = torch.from_numpy(obs).float().to(device)
                with torch.no_grad():
                    q_values = policy(obs_tensor)
                greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy()
                random_mask = rng.random(size=greedy_actions.shape[0]) < eps
                random_actions = rng.integers(
                    0, n_actions, size=greedy_actions.shape[0]
                )
                actions = np.where(random_mask, random_actions, greedy_actions)
                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                dones = np.logical_or(terminated, truncated)
                for i in range(len(actions)):
                    buffer.add(
                        Transition(
                            obs[i].copy(),
                            int(actions[i]),
                            float(rewards[i]),
                            next_obs[i].copy(),
                            bool(dones[i]),
                        )
                    )
                episode_returns += rewards
                episode_lengths += 1
                finished = np.where(dones)[0]
                for idx in finished:
                    recent_returns.append(float(episode_returns[idx]))
                    episode_returns[idx] = 0.0
                    episode_lengths[idx] = 0
                    collected_episodes += 1
                obs = next_obs
            else:
                obs_tensor = (
                    torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
                )
                with torch.no_grad():
                    q_values = policy(obs_tensor)
                action = (
                    int(rng.integers(0, n_actions))
                    if rng.random() < eps
                    else int(torch.argmax(q_values, dim=1).item())
                )
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buffer.add(
                    Transition(
                        np.array(obs, dtype=np.float32),
                        action,
                        float(reward),
                        np.array(next_obs, dtype=np.float32),
                        bool(done),
                    )
                )
                episode_returns[0] += reward
                episode_lengths[0] += 1

                if done:
                    recent_returns.append(float(episode_returns[0]))
                    episode_returns[0] = 0.0
                    episode_lengths[0] = 0
                    collected_episodes += 1
                    obs, _ = env.reset()
                else:
                    obs = next_obs

        if len(buffer) >= cfg.min_buffer:
            for _ in range(cfg.train_iters_per_update):
                obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(cfg.batch_size)
                obs_b = obs_b.to(device)
                act_b = act_b.to(device)
                rew_b = rew_b.to(device)
                next_obs_b = next_obs_b.to(device)
                done_b = done_b.to(device)

                q = policy(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target(next_obs_b).max(1)[0]
                    target_q = rew_b + cfg.gamma * (1 - done_b) * next_q
                loss = F.smooth_l1_loss(q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                optimizer.step()

        if (update + 1) % cfg.target_update == 0:
            target.load_state_dict(policy.state_dict())

        mean_return = float(np.mean(recent_returns)) if recent_returns else 0.0
        wall = time.time() - start_time
        print(
            f"Update {update+1}/{cfg.updates} | Episodes {len(recent_returns)} (window) | MeanReturn {mean_return:.2f} | Buffer {len(buffer)} | Eps {linear_epsilon(update, cfg.start_eps, cfg.end_eps, cfg.eps_decay_updates):.3f} | Time {wall:.1f}s"
        )

        if mean_return > best_mean_return and recent_returns:
            best_mean_return = mean_return
            torch.save(
                {
                    "model": policy.state_dict(),
                    "target": target.state_dict(),
                    "mean_return": mean_return,
                    "update": update,
                    "cfg": cfg.as_dict(),
                },
                os.path.join(cfg.save_dir, "best.pt"),
            )

        if (update + 1) % cfg.save_every == 0:
            torch.save(
                policy.state_dict(),
                os.path.join(cfg.save_dir, f"model_update_{update+1}.pt"),
            )

        update += 1

    print("Training complete. Best mean return:", best_mean_return)


if __name__ == "__main__":  # pragma: no cover
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
