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
    episodes: int = 500
    max_episode_steps: int = 500
    gamma: float = 0.99
    tau: float = 0.005  # soft update coef
    policy_delay: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    buffer_size: int = 300_000
    batch_size: int = 256
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    start_random_episodes: int = 10
    action_noise_std: float = 0.2
    action_noise_final: float = 0.05
    action_noise_decay_episodes: int = 400
    min_buffer: int = 5_000
    updates_per_step: int = 1
    save_dir: str = "runs"
    log_interval: int = 1

    def as_dict(self):
        return self.__dict__.copy()


def linear_decay(ep: int, start: float, end: float, decay_episodes: int):
    if decay_episodes <= 0:
        return end
    t = min(ep / decay_episodes, 1.0)
    return start + (end - start) * t


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_high: np.ndarray):
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

    def forward(self, x: torch.Tensor):
        return self.net(x) * self.act_high


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1)


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


def main():
    cfg = Config()
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

    actor = Actor(obs_dim, act_dim, act_high).to(device)
    critic = Critic(obs_dim, act_dim).to(device)
    target_actor = Actor(obs_dim, act_dim, act_high).to(device)
    target_critic = Critic(obs_dim, act_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    best_mean_return = -1e9

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    buffer = ReplayBuffer(cfg.buffer_size, (obs_dim,), act_dim)
    recent_returns = deque(maxlen=50)
    start_time = time.time()
    total_updates = 0

    for ep in range(1, cfg.episodes + 1):
        obs, _ = env.reset(seed=cfg.seed + ep)
        ep_ret = 0.0
        done = False
        noise_std = linear_decay(ep, cfg.action_noise_std, cfg.action_noise_final, cfg.action_noise_decay_episodes)

        while not done:
            obs_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor(obs_tensor).cpu().numpy()[0]
            if ep <= cfg.start_random_episodes:
                action = act_space.sample()
            else:
                action = action + rng.normal(0, noise_std, size=act_dim) * act_high
            action = np.clip(action, -act_high, act_high)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, float(reward), next_obs, bool(done))
            obs = next_obs
            ep_ret += reward

            if len(buffer) >= cfg.min_buffer:
                for _ in range(cfg.updates_per_step):
                    o_b, a_b, r_b, no_b, d_b = buffer.sample(cfg.batch_size)
                    o_b = o_b.to(device)
                    a_b = a_b.to(device)
                    r_b = r_b.to(device)
                    no_b = no_b.to(device)
                    d_b = d_b.to(device)

                    with torch.no_grad():
                        # target policy smoothing
                        target_a = target_actor(no_b)
                        noise = (torch.randn_like(target_a) * cfg.policy_noise * actor.act_high).clamp(-cfg.noise_clip, cfg.noise_clip)
                        target_a = (target_a + noise).clamp(-actor.act_high, actor.act_high)
                        q1_t, q2_t = target_critic(no_b, target_a)
                        target_q = torch.min(q1_t, q2_t)
                        y = r_b + cfg.gamma * (1 - d_b) * target_q

                    q1, q2 = critic(o_b, a_b)
                    critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                    critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

                    if total_updates % cfg.policy_delay == 0:
                        # actor update
                        a_pred = actor(o_b)
                        actor_loss = -critic.q1_only(o_b, a_pred).mean()
                        actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
                        soft_update(actor, target_actor, cfg.tau)
                        soft_update(critic, target_critic, cfg.tau)

                    total_updates += 1

        recent_returns.append(ep_ret)
        mean_return = float(np.mean(recent_returns))
        if ep % cfg.log_interval == 0:
            wall = time.time() - start_time
            print(f"Ep {ep}/{cfg.episodes} | Ret {ep_ret:.1f} | Mean({len(recent_returns)}) {mean_return:.1f} | Buffer {len(buffer)} | Noise {noise_std:.3f} | Updates {total_updates}")

        if mean_return > best_mean_return and recent_returns:
            best_mean_return = mean_return
            print(f"New best mean return {best_mean_return:.2f}")
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "mean_return": mean_return,
                "episode": ep,
                "cfg": cfg.as_dict(),
                "act_high": act_high,
            }, os.path.join(cfg.save_dir, SAVE_FILE))

    print(f"Training complete. Best mean return {best_mean_return:.2f}")


if __name__ == "__main__":  # pragma: no cover
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
