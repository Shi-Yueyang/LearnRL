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
    env_id: str = "Pendulum-v1"
    seed: int = 42
    episodes: int = 600
    max_episode_steps: int = 500
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 300_000
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2
    target_entropy_scale: float = 0.5  # multiply by -act_dim
    min_buffer: int = 5_000
    updates_per_step: int = 1
    start_random_episodes: int = 10
    save_dir: str = "runs"
    log_interval: int = 1

    def as_dict(self):
        return self.__dict__.copy()


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_high: np.ndarray):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mean = nn.Linear(64, act_dim)
        self.log_std = nn.Linear(64, act_dim)
        self.register_buffer("act_high", torch.from_numpy(act_high.astype(np.float32)))

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, x: torch.Tensor):
        mean, std = self(x)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        a = torch.tanh(z)
        action = a * self.act_high
        # log_prob with tanh correction
        log_prob = normal.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        mean_action = torch.tanh(mean) * self.act_high
        return action, log_prob, mean_action


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
    target_critic = Critic(obs_dim, act_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())

    log_alpha = torch.tensor(np.log(cfg.init_alpha), requires_grad=True, device=device)
    target_entropy = -cfg.target_entropy_scale * act_dim

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)

    buffer = ReplayBuffer(cfg.buffer_size, (obs_dim,), act_dim)
    recent_returns = deque(maxlen=50)
    best_mean_return = -1e9
    start_time = time.time()

    total_steps = 0
    for ep in range(1, cfg.episodes + 1):
        obs, _ = env.reset(seed=cfg.seed + ep)
        ep_ret = 0.0
        done = False
        while not done:
            obs_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
            if ep <= cfg.start_random_episodes:
                action = act_space.sample()
            else:
                with torch.no_grad():
                    a, _, _ = actor.sample(obs_tensor)
                    action = a.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, float(reward), next_obs, bool(done))
            obs = next_obs
            ep_ret += reward
            total_steps += 1

            if len(buffer) >= cfg.min_buffer:
                for _ in range(cfg.updates_per_step):
                    o_b, a_b, r_b, no_b, d_b = buffer.sample(cfg.batch_size)
                    o_b = o_b.to(device)
                    a_b = a_b.to(device)
                    r_b = r_b.to(device)
                    no_b = no_b.to(device)
                    d_b = d_b.to(device)

                    with torch.no_grad():
                        next_a, next_logp, _ = actor.sample(no_b)
                        q1_t, q2_t = target_critic(no_b, next_a)
                        q_t = torch.min(q1_t, q2_t) - log_alpha.exp() * next_logp
                        y = r_b + cfg.gamma * (1 - d_b) * q_t

                    q1, q2 = critic(o_b, a_b)
                    critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                    critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

                    # actor and alpha updates
                    a_new, logp_new, _ = actor.sample(o_b)
                    q1_new, q2_new = critic(o_b, a_new)
                    q_new = torch.min(q1_new, q2_new)
                    actor_loss = (log_alpha.exp() * logp_new - q_new).mean()
                    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

                    alpha_loss = (-log_alpha.exp() * (logp_new + target_entropy).detach()).mean()
                    alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

                    soft_update(critic, target_critic, cfg.tau)

        recent_returns.append(ep_ret)
        mean_return = float(np.mean(recent_returns))
        if ep % cfg.log_interval == 0:
            wall = time.time() - start_time
            print(f"Ep {ep}/{cfg.episodes} | Ret {ep_ret:.1f} | Mean({len(recent_returns)}) {mean_return:.1f} | Buffer {len(buffer)} | Alpha {log_alpha.exp().item():.3f} | Steps {total_steps}")

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
                "log_alpha": log_alpha.detach().cpu(),
            }, os.path.join(cfg.save_dir, SAVE_FILE))

    print(f"Training complete. Best mean return {best_mean_return:.2f}")


if __name__ == "__main__":  # pragma: no cover
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
