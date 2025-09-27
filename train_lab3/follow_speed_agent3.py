import os
import sys
import time
import random
from typing import Literal, Callable, Dict, Any
from collections import deque

from helper import generate_random_target_speeds, interp_1d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from train2 import Train2


from td3.train_agent import Config, Actor, Critic, ReplayBuffer, train_td3
from track import Track
from train2 import high_speed_train_params_test
from track import default_track_layout
from plotting import (
    plot_learning_curve_with_shade,
    plot_learning_curve,
    plot_multiple_learning_curves_with_shade,
)


# TARGET_SPEEDS = {"speeds": 10}
# TARGET_SPEEDS = "random"
# TARGET_SPEEDS = {
#     "times": [0, 5, 10, 15, 20, 25, 30],
#     "speeds": [0, 30, 0, 20, 0, 30, 0],
# }

# TARGET_SPEEDS = {
#     "times": [0, 5, 10, 15, 20, 25, 30],
#     "speeds": [0, 30, 30, 50, 30, 0, 0],
# }

TARGET_SPEEDS = [generate_random_target_speeds(30.0, num_points=4) for _ in range(2)]
# TARGET_SPEEDS = {"times": [0, 10, 15, 20], "speeds": [0, 10, 10, 25]}

EPISODE_LENGTH = 30
EPISODES = 5000
INTERP_METHOD = "cubic"


class TrainSpeedEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, err_cnt, ext_dim, render_mode=None):
        super().__init__()

        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(1,),
            dtype=np.float32,  # Reasonable action range
        )

        self.extend_dim = ext_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(err_cnt + ext_dim,),
            dtype=np.float32,
        )

        self.recent_errors = deque([0] * err_cnt, maxlen=err_cnt)
        self.render_mode = render_mode
        self.reached_steps = 0
        # Environment state variables
        self.state = None
        self.previous_action = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.train = Train2(options["train_coeffs"])
        self.track = Track(options["track_layout"])
        self.target_speeds = options["target_speeds"]
        self.terminate_time = options["terminate_time"]

        self.train.position = options.get("start_pos", 0.0)
        self.train.velocity = options.get("start_vel", 0.0)
        self.train.time = options.get("start_time", 0.0)

        
        # Interpolation method: 'linear', 'nearest', 'previous'/'zoh', 'next',
        # and smooth methods: 'pchip', 'cubic', 'univariate', 'akima'
        self.interp_method = options.get("interp_method", "linear")
        # Optional smoothing parameter for UnivariateSpline
        self.interp_s = options.get("interp_s", 1.0)

        if self.target_speeds == "random":
            self.target_speeds = generate_random_target_speeds(
                self.terminate_time, num_points=4
            )

        if isinstance(self.target_speeds, list):
            self.target_speeds = random.choice(self.target_speeds)

        self.at_target_counter = 0
        self.steps = 0
        self.previous_action = 0.0

        self.state = np.array(
            [
                self.train.position,
                self.train.velocity,
                self.train.acceleration,
                self.train.time,
            ],
            dtype=np.float32,
        )
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray | float):
        if isinstance(action, np.ndarray) or isinstance(action, list):
            action = float(action[0])
        # Don't scale action here - let the actor learn the proper scaling
        action /= 10.0
        self._update_state(action)

        # Get observation
        observation = self._get_obs()

        # Calculate reward
        reward = self._calculate_reward(action)

        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        # Get info
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def get_target_speed(self):
        pos, vel, acc, time = self.state

        if "positions" in self.target_speeds:
            val = self._interp_1d(
                pos,
                self.target_speeds["positions"],
                self.target_speeds["speeds"],
            )
            target_speed = np.float32(val)
        elif "times" in self.target_speeds:
            val = interp_1d(
                time,
                self.target_speeds["times"],
                self.target_speeds["speeds"],
                self.interp_method,
                self.interp_s,
            )
            target_speed = np.float32(val)
        else:
            target_speed = np.float32(self.target_speeds["speeds"])

        return target_speed

    def _update_state(self, action: float):
        """Update the environment state based on action"""
        # Example state update logic - replace with your dynamics

        dt = 0.1  # time step
        time, pos, vel, acc = self.train.update_dynamics(dt, action, self.track)
        self.state = np.array([pos, vel, acc, time], dtype=np.float32)

    def _get_obs(self):
        """Get current observation"""
        # Return velocity error instead of acceleration
        pos, vel, acc, time = self.state

        track_props = self.track.get_current_properties(train_position=pos)
        extend_obs = [vel, acc, track_props["gradient"], track_props["curve_radius"]]
        extend_obs = extend_obs[: self.extend_dim]

        target_speed = self.get_target_speed()
        velocity_error = vel - target_speed
        self.recent_errors.append(velocity_error)

        return np.array(extend_obs + list(self.recent_errors), dtype=np.float32)

    def _calculate_reward(self, action):
        pos, vel, acc, time = self.state
        target_speed = self.get_target_speed()

        # --- Tracking ---
        velocity_error = abs(vel - target_speed)

        velocity_reward = -((velocity_error / (abs(target_speed) + 1)) ** 2)

        # velocity_reward = (
        #     -((velocity_error) ** 2) / 500.0
        # )

        # --- Action penalties ---
        action_change = action - self.previous_action
        action_smoothness_penalty = -0.05 * (action_change**2)

        survive_reward = 0.2
        self.previous_action = action

        return velocity_reward + survive_reward

    def _is_terminated(self):
        pos, vel, acc, time = self.state
        if time >= self.terminate_time:
            return True
        return False

    def _is_truncated(self):
        pos, vel, acc, time = self.state
        target_speed = self.get_target_speed()

        velocity_error = abs(vel - target_speed)
        max_allowed_error = max(2.0, abs(target_speed) * 0.5)

        if velocity_error > max_allowed_error and time > 2.0:
            return True

        return False

    def _get_info(self):
        """Get additional info dictionary"""
        return {
            "steps": self.steps,
            "position": self.state[0],
            "velocity": self.state[1],
            "acceleration": self.state[2],
            "time": self.state[3],
        }

    def render(self):
        """Render the environment(optional)"""
        if self.render_mode == "human":
            print(f"Step: {self.steps}, State: {self.state}")

    def close(self):
        """Clean up resources"""
        pass


def test_env():
    from train2 import high_speed_train_params_test
    from track import default_track_layout

    """Test the custom environment"""
    env = TrainSpeedEnv(err_cnt=1, ext_dim=3)

    target_speeds = {"times": [0, 5, 10, 15, 20], "speeds": [0, 10, 10, 0, 15]}
    option = {
        "train_coeffs": high_speed_train_params_test,
        "track_layout": default_track_layout,
        "target_speeds": target_speeds,
        "terminate_time": 20.0,
        "interp_method": "linear",
    }
    # Reset environment
    obs, info = env.reset(seed=42, options=option)
    print(f"Initial observation: {obs}")

    # Run a few steps
    for action in np.linspace(10, -10, num=30):
        # Random action
        if action < 0:
            pass
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"action={action:.1f}, obs= {  '|'.join([f'{ob:.1f}' for ob in obs ]) }, reward={reward:.3f}, obs=[ {' '.join([f'{v:.1f}' for v in obs])} ]"
        )

    env.close()


def option_change_cb(env_option: Dict, mean_return):
    # Example: If mean return is below a threshold, change target speeds
    if mean_return > 30:
        target_speeds = env_option["target_speeds"]
        speeds = target_speeds["speeds"]
        new_speeds = [max(0, s + random.uniform(-5, 5)) for s in speeds]
        env_option["target_speeds"]["speeds"] = new_speeds
        print(
            f"Changed target speeds to: [ {' '.join([f'{s:.1f}' for s in new_speeds])} ]"
        )
        return True
    return False


def train_agent(
    seed: int = 42,
    save_dir="runs",
    err_cnt=2,
    ext_dim=1,
    is_do_test=True,
):
    cfg = Config()
    cfg.episodes = EPISODES
    cfg.actor_lr = 1e-4  # Lower learning rate
    cfg.critic_lr = 1e-4
    cfg.noise_std = 10
    cfg.noise_std_final = 0.1
    cfg.start_random_episodes = 30
    cfg.batch_size = 300
    cfg.log_interval = 10
    cfg.seed = seed
    cfg.save_dir = save_dir

    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TrainSpeedEnv(err_cnt=err_cnt, ext_dim=ext_dim)

    if cfg.max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.max_episode_steps)

    # Interpolation method: 'linear', 'nearest', 'previous'/'zoh', 'next',
    # and smooth methods: 'pchip', 'cubic', 'univariate', 'akima'
    env_option = {
        "train_coeffs": high_speed_train_params_test,
        "track_layout": default_track_layout,
        "target_speeds": TARGET_SPEEDS,
        "terminate_time": EPISODE_LENGTH,
        "interp_method": INTERP_METHOD,
    }
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

    extra_save = {"err_cnt": err_cnt, "ext_dim": ext_dim}
    try:
        return train_td3(
            cfg,
            env,
            actor,
            critic,
            target_actor,
            target_critic,
            buffer,
            device,
            env_option,
            extra_save,
            # option_change_cb,
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        if is_do_test:
            test_agent()


def train_agents(iters=10):
    for err_cnt in [7, 4, 1]:
        best_ep_ret = -float("inf")
        for i in range(iters):
            print(f"Training agent {i+1}/10")
            seed = np.random.randint(0, 200)
            run_id = time.strftime("%Y%m%d-%H%M%S")
            save_dir = os.path.join(
                "runs", "history", str(err_cnt), f"{run_id}_seed_{seed}"
            )
            ep_return = train_agent(
                seed=seed,
                save_dir=save_dir,
                is_do_test=False,
                err_cnt=err_cnt,
            )
            plot_learning_curve(
                csv_path=os.path.join(save_dir, "metrics.csv"),
            )
            if ep_return > best_ep_ret:
                best_ep_ret = ep_return
                best_save_dir = save_dir

        # Rename the best run folder by prefixing the base name with 'best_'
        base = os.path.basename(best_save_dir)
        new_base = f"best_{base}"
        new_path = os.path.join(os.path.dirname(best_save_dir), new_base)
        os.rename(best_save_dir, new_path)
        plot_learning_curve_with_shade(
            history_dir=os.path.join(
                os.path.dirname(__file__),
                "runs",
                "history",
                str(err_cnt),
            )
        )
    plot_multiple_learning_curves_with_shade(
        set_dir=os.path.join(
            os.path.dirname(__file__),
            "runs",
            "history",
        )
    )
    print("finished")


def visualize_control_law(result: Dict[str, list], x_axes="time"):
    plt.style.use("dark_background")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 7))
    ep_return = sum(result["reward_history"])

    # Determine x-axis data based on x_axes parameter
    x_data = result["time_history"] if x_axes == "time" else result["pos_history"]

    # Position, Velocity, Acceleration plots
    plots = [
        (
            ax1,
            [
                (x_data, result["vel_history"], "lime", "vel"),
                (
                    x_data,
                    result["target_vel_history"],
                    "orange",
                    "target vel",
                ),
            ],
            "vel (m/s)",
        ),
        (
            ax2,
            [
                (x_data, result["acc_history"], "lime", "acc"),
            ],
            "acc (m/s2)",
        ),
        (
            ax3,
            [
                (x_data, result["action_history"], "orange", "action"),
            ],
            "Control",
        ),
        (
            ax4,
            [
                (x_data, result["reward_history"], "orange", "reward"),
            ],
            f"Reward (ep_ret={ep_return:.2f})",
        ),
    ]

    for ax, datas, title in plots:
        for data in datas:
            ax.plot(data[0], data[1], color=data[2], label=data[3])
        ax.set_title(title, color="white")
        ax.set_xlabel("Time (s)" if x_axes == "time" else "Position (m)", color="white")
        ax.set_ylabel(title, color="white")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(colors="white")
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def test_control_law(
    env: TrainSpeedEnv,
    env_option: Dict[str, Any],
    control_law: Callable[[np.ndarray], np.ndarray],
    is_render: bool = True,
    x_axes: Literal["time", "pos"] = "time",
) -> Dict[str, list]:
    obs, info = env.reset(seed=42, options=env_option)

    pos_history = []
    vel_history = []
    target_vel_history = []
    time_history = []
    action_history = []
    reward_history = []
    acc_history = []
    done = False
    steps = 0
    while not done:
        # Get action from the control law
        action = control_law(obs)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        if steps == 0:
            action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        target_speed = env.get_target_speed()

        state = env.state  # pos vel acc time
        vel_history.append(state[1])
        target_vel_history.append(target_speed)
        acc_history.append(state[2])
        time_history.append(state[3])
        action_history.append(action[0])
        reward_history.append(reward)
        pos_history.append(state[0])
        steps += 1

    result = {
        "pos_history": pos_history,
        "vel_history": vel_history,
        "target_vel_history": target_vel_history,
        "acc_history": acc_history,
        "time_history": time_history,
        "action_history": action_history,
        "reward_history": reward_history,
    }
    print(f"steps {steps}")
    if is_render:
        visualize_control_law(result, x_axes)
    return result


def test_agent(ckpt_path: str = None):

    if ckpt_path is None:
        ckpt_path = os.path.join(os.path.dirname(__file__), "runs", "best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_dim = ckpt["obs_dim"]
    extra_save = ckpt["extra_save"]

    err_cnt = extra_save["err_cnt"]
    ext_dim = extra_save["ext_dim"]

    act_dim = ckpt["act_dim"]
    act_high = ckpt["act_high"]
    act_low = ckpt["act_low"]
    actor = Actor(obs_dim, act_dim, act_high, act_low).to(device)
    actor.load_state_dict(ckpt["actor"])
    print(f"load model with ep_ret {ckpt['ep_ret']:.2f}")

    env = TrainSpeedEnv(err_cnt, ext_dim)
    # and smooth methods: 'pchip', 'cubic', 'univariate', 'akima'

    option = {
        "train_coeffs": high_speed_train_params_test,
        "track_layout": default_track_layout,
        "target_speeds": TARGET_SPEEDS,
        "terminate_time": EPISODE_LENGTH,
        "interp_method": INTERP_METHOD,
    }
    # Reset environment
    obs, _ = env.reset(seed=42, options=option)
    print(f"Initial err: {  ' |'.join([f'{ob:.1f}' for ob in obs ]) }")

    def control_law(obs):
        obs_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor(obs_tensor).cpu().numpy()[0]
        return action

    test_control_law(env, option, control_law, is_render=True, x_axes="time")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    train_agent(err_cnt=3, ext_dim=0)

    # test_agent()

    test_env()
