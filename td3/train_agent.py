import os
import time
import csv
from dataclasses import dataclass
from collections import deque
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import math


# TensorBoard (assumed installed as requested)
from torch.utils.tensorboard import SummaryWriter


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
    log_interval: int = 10
    # logging
    save_dir: str = "runs"
    mirror_dir: str = "runs"
    csv_filename: str = "metrics.csv"
    save_file: str = "best.pt"

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


class MetricsLogger:
    """Simple CSV logger with optional TensorBoard support.

    Writes a header on first use and appends per-episode metrics afterwards.
    """

    def __init__(
        self,
        out_dir: str,
        filename: str = "metrics.csv",
        mirror_dir: Optional[str] = None,
        mirror_filename: Optional[str] = None,
    ):
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, filename)
        self._csv_exists = (
            os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
        )
        # newline="" to avoid blank lines on Windows
        self._fp = open(self.csv_path, mode="a", newline="", encoding="utf-8")
        self._writer: Optional[csv.DictWriter] = None
        # optional mirror to a root (latest) file
        self._mirror_fp = None
        self._mirror_writer: Optional[csv.DictWriter] = None
        self._mirror_header_written = False
        if mirror_dir is not None:
            os.makedirs(mirror_dir, exist_ok=True)
            mirror_name = mirror_filename if mirror_filename else filename
            mirror_path = os.path.join(mirror_dir, mirror_name)
            # overwrite for latest
            self._mirror_fp = open(mirror_path, mode="w", newline="", encoding="utf-8")
        # Put TB logs in a subdir to avoid clobbering checkpoints
        tb_dir = os.path.join(out_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        self._tb = SummaryWriter(tb_dir)

    def log(self, metrics: Dict[str, Any], tb_step: Optional[int] = None):
        # Initialize writer with fieldnames on first call
        if self._writer is None:
            fieldnames = list(metrics.keys())
            self._writer = csv.DictWriter(self._fp, fieldnames=fieldnames)
            if not self._csv_exists:
                self._writer.writeheader()
            if self._mirror_fp is not None and not self._mirror_header_written:
                self._mirror_writer = csv.DictWriter(
                    self._mirror_fp, fieldnames=fieldnames
                )
                self._mirror_writer.writeheader()
                self._mirror_header_written = True
        self._writer.writerow(metrics)
        self._fp.flush()
        if self._mirror_writer is not None:
            self._mirror_writer.writerow(metrics)
            self._mirror_fp.flush()

        # TensorBoard scalars (skip a few counters)
        step = tb_step if tb_step is not None else metrics.get("episode", 0)
        for k, v in metrics.items():
            if k in ("episode", "elapsed_sec"):
                continue
            self._tb.add_scalar(k, float(v), global_step=int(step))

    def close(self):  # pragma: no cover - trivial
        self._tb.flush()
        self._tb.close()
        self._fp.flush()
        self._fp.close()
        if self._mirror_fp is not None:
            self._mirror_fp.flush()
            self._mirror_fp.close()


class EpisodeMetrics:
    """Collects per-episode metrics and derives summary at the end."""

    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = low
        self.high = high
        self.reset(0.0)

    def reset(self, start_time: float):
        self.ep_start_time = start_time
        # action saturation tracking
        self.act_at_bounds = 0
        self.act_total_dims = 0
        # update counters
        self.critic_updates = 0
        self.actor_updates = 0
        # last-batch stats defaults
        self.q1_mean = float("nan")
        self.q2_mean = float("nan")
        self.q_min_mean = float("nan")
        self.q_gap_mean = float("nan")
        self.td_error_mean = float("nan")
        self.td_error_std = float("nan")
        self.target_y_mean = float("nan")
        self.target_y_std = float("nan")
        self.noise_clipped_frac = float("nan")
        self.critic_grad_norm = float("nan")
        self.actor_grad_norm = float("nan")

    def record_action(self, action: np.ndarray):
        bounds_mask = (action <= (self.low + 1e-6)) | (action >= (self.high - 1e-6))
        self.act_at_bounds += int(np.count_nonzero(bounds_mask))
        self.act_total_dims += int(bounds_mask.size)

    def record_batch_stats(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        y: torch.Tensor,
        raw_noise: Optional[torch.Tensor],
        noise_clip: float,
    ):
        with torch.no_grad():
            self.q1_mean = float(q1.mean().item())
            self.q2_mean = float(q2.mean().item())
            self.q_min_mean = float(torch.min(q1, q2).mean().item())
            self.q_gap_mean = float((q1 - q2).mean().item())
            self.target_y_mean = float(y.mean().item())
            self.target_y_std = float(y.std(unbiased=False).item())
            td1 = y - q1
            self.td_error_mean = float(td1.abs().mean().item())
            self.td_error_std = float(td1.std(unbiased=False).item())
            if raw_noise is not None:
                try:
                    self.noise_clipped_frac = float(
                        (raw_noise.abs() >= noise_clip).float().mean().item()
                    )
                except Exception:
                    pass

    @staticmethod
    def _compute_grad_norm(model: nn.Module) -> float:
        total_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                total_sq += float(g.norm(2).item() ** 2)
        return math.sqrt(total_sq) if total_sq > 0 else 0.0

    def update_critic_grad_norm(self, model: nn.Module):
        self.critic_grad_norm = self._compute_grad_norm(model)
        self.critic_updates += 1

    def update_actor_grad_norm(self, model: nn.Module):
        self.actor_grad_norm = self._compute_grad_norm(model)
        self.actor_updates += 1

    def finalize(self, steps: int) -> Dict[str, float]:
        ep_wall = max(1e-9, time.time() - self.ep_start_time)
        action_saturation_frac = (
            (self.act_at_bounds / self.act_total_dims)
            if self.act_total_dims > 0
            else 0.0
        )
        updates_per_sec = self.critic_updates / ep_wall
        env_steps_per_sec = steps / ep_wall
        return {
            "action_saturation_frac": float(action_saturation_frac),
            "noise_clipped_frac": float(self.noise_clipped_frac),
            "q1_mean": self.q1_mean,
            "q2_mean": self.q2_mean,
            "q_min_mean": self.q_min_mean,
            "q_gap_mean": self.q_gap_mean,
            "target_y_mean": self.target_y_mean,
            "target_y_std": self.target_y_std,
            "td_error_mean": self.td_error_mean,
            "td_error_std": self.td_error_std,
            "critic_grad_norm": self.critic_grad_norm,
            "actor_grad_norm": self.actor_grad_norm,
            "updates_per_sec": float(updates_per_sec),
            "env_steps_per_sec": float(env_steps_per_sec),
        }


def train_td3(
    cfg: Config,
    env,
    actor,
    critic,
    target_actor,
    target_critic,
    buffer: ReplayBuffer,
    device,
    env_option=None,
) -> float:
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    recent_returns = deque(maxlen=50)
    start_time = time.time()
    rng = np.random.default_rng(cfg.seed)
    best_mean_return = -1e9
    total_updates = 0
    total_steps = 0
    last_print_time = 0

    # Create a timestamped run directory under save_dir and mirror latest files in root save_dir
    os.makedirs(cfg.save_dir, exist_ok=True)

    logger = MetricsLogger(cfg.save_dir, cfg.csv_filename, mirror_dir=cfg.mirror_dir)

    try:
        for ep in range(1, cfg.episodes + 1):
            obs, _ = env.reset(seed=cfg.seed + ep, options=env_option)
            ep_ret = 0.0
            steps = 0
            critic_loss_val: float = 0.0
            actor_loss_val: Optional[float] = None
            done = False
            ep_metrics = EpisodeMetrics(env.action_space.low, env.action_space.high)
            ep_metrics.reset(time.time())
            noise_std = linear_decay(
                ep,
                cfg.action_noise_std,
                cfg.action_noise_final,
                cfg.action_noise_decay_episodes,
            )

            while not done:
                steps += 1
                total_steps += 1
                obs_tensor = (
                    torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
                )
                with torch.no_grad():
                    action = actor(obs_tensor).cpu().numpy()[0]
                if ep <= cfg.start_random_episodes:
                    action = env.action_space.sample()
                else:
                    action = action + rng.normal(
                        0, noise_std, size=env.action_space.shape[0]
                    )
                low = env.action_space.low
                high = env.action_space.high
                action = np.clip(action, low, high)
                ep_metrics.record_action(action)
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
                            raw_noise = (
                                torch.randn_like(target_a)
                                * cfg.policy_noise
                                * actor.act_high
                            )
                            noise = raw_noise.clamp(-cfg.noise_clip, cfg.noise_clip)
                            target_a = (target_a + noise).clamp(
                                -actor.act_high, actor.act_high
                            )
                            q1_t, q2_t = target_critic(no_b, target_a)
                            target_q = torch.min(q1_t, q2_t)
                            y = r_b + cfg.gamma * (1 - d_b) * target_q

                        q1, q2 = critic(o_b, a_b)
                        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                        critic_opt.zero_grad()
                        critic_loss.backward()
                        # grad norm and stats
                        ep_metrics.update_critic_grad_norm(critic)
                        critic_opt.step()
                        critic_loss_val = float(critic_loss.detach().item())
                        ep_metrics.record_batch_stats(
                            q1, q2, y, raw_noise, cfg.noise_clip
                        )

                        if total_updates % cfg.policy_delay == 0:
                            # actor update
                            a_pred = actor(o_b)
                            actor_loss = -critic.q1_only(o_b, a_pred).mean()
                            actor_opt.zero_grad()
                            actor_loss.backward()
                            ep_metrics.update_actor_grad_norm(actor)
                            actor_opt.step()
                            soft_update(actor, target_actor, cfg.tau)
                            soft_update(critic, target_critic, cfg.tau)
                            actor_loss_val = float(actor_loss.detach().item())
                            # actor_updates counted in update_actor_grad_norm

                        total_updates += 1
                        # critic update already counted in ep_metrics

            recent_returns.append(ep_ret)
            mean_return = float(np.mean(recent_returns))
            wall = time.time() - start_time
            if ep == 1 or (ep % cfg.log_interval == 0 and wall - last_print_time > 1):
                last_print_time = wall
                print(
                    f"Ep {ep}/{cfg.episodes} | Steps {steps} | Ret {ep_ret:.1f} | Mean50 {mean_return:.1f} | Buffer {len(buffer)} | Critic loss {critic_loss_val:.2f}"
                )

            # Log to CSV/TensorBoard
            ep_summary = ep_metrics.finalize(steps)
            logger.log(
                {
                    "episode": ep,
                    "return": float(ep_ret),
                    "mean50": mean_return,
                    "steps": steps,
                    "total_steps": total_steps,
                    "buffer_size": len(buffer),
                    "noise_std": float(noise_std),
                    "critic_loss": float(critic_loss_val),
                    "actor_loss": (
                        float(actor_loss_val)
                        if actor_loss_val is not None
                        else float("nan")
                    ),
                    # merged episode summary stats
                    **ep_summary,
                    "elapsed_sec": float(wall),
                },
                tb_step=total_steps,
            )

            if mean_return > best_mean_return and recent_returns:
                best_mean_return = mean_return
                payload = {
                    "act_dim": env.action_space.shape[0],
                    "obs_dim": env.observation_space.shape[0],
                    "act_high": env.action_space.high,
                    "act_low": env.action_space.low,
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "mean_return": mean_return,
                    "ep_ret": ep_ret,
                    "episode": ep,
                    "cfg": cfg.as_dict(),
                    "act_high": env.action_space.high,
                }
                # Save to run-specific dir
                torch.save(payload, os.path.join(cfg.save_dir, cfg.save_file))
                torch.save(payload, os.path.join(cfg.mirror_dir, cfg.save_file))
    finally:
        logger.close()

    print(f"Training complete. Best mean return {best_mean_return:.2f}")
    return best_mean_return

def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    train_td3(cfg, env, actor, critic, target_actor, target_critic, buffer, device)


if __name__ == "__main__":  # pragma: no cover
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
