import os
import time
from dataclasses import dataclass
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

SAVE_FILE = "best.pt"


@dataclass
class Config:
    env_id: str = "Pendulum-v1"  # continuous action env
    seed: int = 42
    episodes: int = 400
    max_episode_steps: int = 500
    gamma: float = 0.99
    tau: float = 0.005  # soft update coef
    buffer_size: int = 200_000
    batch_size: int = 256
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    start_random_episodes: int = 10
    noise_std: float = 0.2
    noise_std_final: float = 0.05
    noise_decay_episodes: int = 300
    min_buffer: int = 5_000
    updates_per_step: int = 1
    save_dir: str = "runs"
    log_interval: int = 10

    def as_dict(self):
        return self.__dict__.copy()


def linear_decay(ep: int, start: float, end: float, decay_episodes: int):
    if decay_episodes <= 0:
        return end
    t = min(ep / decay_episodes, 1.0)
    return start + (end - start) * t


class Actor(nn.Module):
    def __init__(
        self, obs_dim: int, act_dim: int, act_high: np.ndarray, act_low: np.ndarray
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh(),
        )
        self.register_buffer("act_high", torch.from_numpy(act_high.astype(np.float32)))
        self.register_buffer("act_low", torch.from_numpy(act_low.astype(np.float32)))

    def forward(self, x: torch.Tensor):
        tanh_output = self.net(x)
        # Scale and shift the tanh output from [-1, 1] to [act_low, act_high]
        scaled_output = (tanh_output + 1) / 2 * (
            self.act_high - self.act_low
        ) + self.act_low
        return scaled_output


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.q(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.idx = 0
        self.full = False

    def add(self, o, a, r, no, d):
        i = self.idx
        self.obs[i] = o
        self.next_obs[i] = no
        self.actions[i] = a
        self.rewards[i] = r
        self.dones[i] = d
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, len(self), size=batch_size)
        return (
            torch.from_numpy(self.obs[idxs]),
            torch.from_numpy(self.actions[idxs]),
            torch.from_numpy(self.rewards[idxs]),
            torch.from_numpy(self.next_obs[idxs]),
            torch.from_numpy(self.dones[idxs].astype(np.float32)),
        )


def soft_update(src: nn.Module, dst: nn.Module, tau: float):
    for p_src, p_dst in zip(src.parameters(), dst.parameters()):
        p_dst.data.lerp_(p_src.data, tau)


def train_ddpg(
    cfg: Config,
    env,
    actor,
    critic,
    target_actor,
    target_critic,
    buffer,
    device,
    env_option=None,
    save_file=SAVE_FILE,
):
    """Train a DDPG agent."""
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    recent_returns = deque(maxlen=50)
    best_mean_return = -1e9
    start_time = time.time()
    rng = np.random.default_rng(cfg.seed)

    for ep in range(1, cfg.episodes + 1):
        obs, _ = env.reset(seed=cfg.seed + ep, options=env_option)
        ep_ret = 0.0
        done = False
        steps = 0
        noise_std = linear_decay(
            ep, cfg.noise_std, cfg.noise_std_final, cfg.noise_decay_episodes
        )

        while not done:
            obs_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor(obs_tensor).cpu().numpy()[0]
            if ep <= cfg.start_random_episodes:
                action = env.action_space.sample()
            else:
                action = action + rng.normal(
                    0, noise_std, size=env.action_space.shape[0]
                )
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, float(reward), next_obs, bool(done))
            obs = next_obs
            ep_ret += reward
            steps += 1
            critic_loss = 0
            if len(buffer) >= cfg.min_buffer:
                for _ in range(cfg.updates_per_step):
                    o_b, a_b, r_b, no_b, d_b = buffer.sample(cfg.batch_size)
                    o_b = o_b.to(device)
                    a_b = a_b.to(device)
                    r_b = r_b.to(device)
                    no_b = no_b.to(device)
                    d_b = d_b.to(device)

                    with torch.no_grad():
                        target_a = target_actor(no_b)
                        target_q = target_critic(no_b, target_a)
                        y = r_b + cfg.gamma * (1 - d_b) * target_q
                    q = critic(o_b, a_b)
                    critic_loss = F.mse_loss(q, y)
                    critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_opt.step()

                    # actor
                    a_pred = actor(o_b)
                    actor_loss = -critic(o_b, a_pred).mean()
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    soft_update(actor, target_actor, cfg.tau)
                    soft_update(critic, target_critic, cfg.tau)

        recent_returns.append(ep_ret)
        mean_return = float(np.mean(recent_returns))
        if ep % cfg.log_interval == 0:
            wall = time.time() - start_time
            print(
                f"Episode {ep}/{cfg.episodes} | Return {ep_ret:.1f} | MeanReturn({len(recent_returns)}) {mean_return:.1f} | Buffer {len(buffer)} | Critc loss {critic_loss:.2f}| Time {wall:.1f}s"
            )

        if mean_return > best_mean_return and recent_returns:
            best_mean_return = mean_return
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "mean_return": mean_return,
                    "episode": ep,
                    "cfg": cfg.as_dict(),
                    "act_high": env.action_space.high,
                },
                os.path.join(cfg.save_dir, save_file),
            )

    print(f"Training complete. Best mean return {best_mean_return:.2f}")


def main():
    cfg = Config()
    cfg.episodes = 4000

    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = gym.make(cfg.env_id)
    if cfg.max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.max_episode_steps)

    obs_space = env.observation_space
    act_space = env.action_space
    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]
    act_high = act_space.high
    act_low = act_space.low
    actor = Actor(obs_dim, act_dim, act_high, act_low).to(device)
    critic = Critic(obs_dim, act_dim).to(device)
    target_actor = Actor(obs_dim, act_dim, act_high, act_low).to(device)
    target_critic = Critic(obs_dim, act_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    buffer = ReplayBuffer(cfg.buffer_size, (obs_dim,), act_dim)

    train_ddpg(cfg, env, actor, critic, target_actor, target_critic, buffer, device)


if __name__ == "__main__":  # pragma: no cover
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
