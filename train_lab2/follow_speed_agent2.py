import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.interpolate import (
    PchipInterpolator,
    CubicSpline,
    UnivariateSpline,
    Akima1DInterpolator,
)
from train2 import Train2
from track import Track
from typing import Callable, Dict, Any
import matplotlib.pyplot as plt
from typing import Literal
from collections import deque


from td3.train_agent import Config, Actor, Critic, ReplayBuffer, train_td3
from train_lab2.train2 import high_speed_train_params_test
from track import default_track_layout
from train_lab2.plotting import plot_all_histories, plot_learning_curve
import torch


class ConstSpeedEnv2(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, obs_dim, render_mode=None):
        super().__init__()

        # Define action and observation spaces
        # Continuous action space for throttle/brake control
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(1,),
            dtype=np.float32,  # Reasonable action range
        )

        # Example: observation space for position, velocity, etc.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),  # [position, velocity, acceleration, time]
            dtype=np.float32,
        )
        self.recent_errors = deque([0] * obs_dim, maxlen=obs_dim)
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
        # Interpolation method: 'linear', 'nearest', 'previous'/'zoh', 'next',
        # and smooth methods: 'pchip', 'cubic', 'univariate', 'akima'
        self.interp_method = options.get("interp_method", "linear")
        # Optional smoothing parameter for UnivariateSpline
        self.interp_s = options.get("interp_s", 1.0)

        if self.target_speeds == "random":
            time_points = np.sort(np.random.uniform(0, self.terminate_time, size=4))
            speed_points = np.random.uniform(0, 50, size=4)
            speed_points[2] = speed_points[1]
            self.target_speeds = {
                "times": time_points.tolist(),
                "speeds": speed_points.tolist(),
            }

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

    def _interp_1d(self, x: float, xp, fp) -> float:
        """Interpolation helper with multiple methods.
        Methods:
          - 'linear': piecewise linear (np.interp)
          - 'nearest': nearest neighbor
          - 'previous' / 'zoh' / 'step': zero-order hold (previous sample)
          - 'next': next-step hold (right continuous)
          - 'pchip': monotone cubic (PCHIP, no overshoot)
          - 'cubic': natural cubic spline
          - 'univariate': smoothed spline (UnivariateSpline), uses self.interp_s
          - 'akima': Akima1DInterpolator (robust to local oscillations)
        Boundary behavior: clamp to first/last fp.
        """
        xp_arr = np.asarray(xp, dtype=float)
        fp_arr = np.asarray(fp, dtype=float)

        method = str(self.interp_method).lower()
        # Clamp outside domain
        if x <= xp_arr[0]:
            return float(fp_arr[0])
        if x >= xp_arr[-1]:
            return float(fp_arr[-1])
        if method == "linear":
            return float(np.interp(x, xp_arr, fp_arr, left=fp_arr[0], right=fp_arr[-1]))
        elif method == "nearest":
            idx = int(np.argmin(np.abs(xp_arr - x)))
            return float(fp_arr[idx])
        elif method in ("previous", "zoh", "step"):
            idx = int(np.searchsorted(xp_arr, x, side="right") - 1)
            idx = int(np.clip(idx, 0, len(xp_arr) - 1))
            return float(fp_arr[idx])
        elif method == "next":
            idx = int(np.searchsorted(xp_arr, x, side="left"))
            idx = int(np.clip(idx, 0, len(xp_arr) - 1))
            return float(fp_arr[idx])
        elif method in ("pchip", "monotone"):
            f = PchipInterpolator(xp_arr, fp_arr, extrapolate=True)
            return float(f(x))
        elif method in ("cubic", "cubic_spline", "cspline"):
            f = CubicSpline(xp_arr, fp_arr, bc_type="natural", extrapolate=True)
            return float(f(x))
        elif method in ("univariate", "spline", "smooth"):
            f = UnivariateSpline(xp_arr, fp_arr, s=self.interp_s)
            return float(f(x))
        elif method == "akima":
            f = Akima1DInterpolator(xp_arr, fp_arr)
            return float(f(x))
        else:
            # Fallback to linear
            return float(np.interp(x, xp_arr, fp_arr, left=fp_arr[0], right=fp_arr[-1]))

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
            val = self._interp_1d(
                time,
                self.target_speeds["times"],
                self.target_speeds["speeds"],
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
        target_speed = self.get_target_speed()
        velocity_error = vel - target_speed
        self.recent_errors.append(velocity_error)

        return np.array(self.recent_errors, dtype=np.float32)

    def _calculate_reward(self, action):
        pos, vel, acc, time = self.state
        target_speed = self.get_target_speed()

        # --- Tracking ---
        velocity_error = abs(vel - target_speed)
        # velocity_reward = -((velocity_error / (abs(target_speed) + 1e-3)) ** 2)
        velocity_reward = (
            -((velocity_error) ** 2) / 50.0
        )  # Scale to keep reward manageable

        # --- Action penalties ---
        action_change = action - self.previous_action
        action_smoothness_penalty = -0.02 * (action_change**2)

        survive_reward = 0.1
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
        max_allowed_error = max(10.0, abs(target_speed) * 0.5)

        if velocity_error > max_allowed_error and time > 2.0:
            return True

        # if target_speed > 1.0 and vel < -1.0:
        #     return True

        return False

    def _get_info(self):
        """Get additional info dictionary"""
        return {
            "steps": self.steps,
            "position": self.state[0],
            "velocity": self.state[1],
        }

    def render(self):
        """Render the environment(optional)"""
        if self.render_mode == "human":
            print(f"Step: {self.steps}, State: {self.state}")

    def close(self):
        """Clean up resources"""
        pass


def test_env():
    from train_lab2.train2 import high_speed_train_params_test
    from track import default_track_layout

    """Test the custom environment"""
    env = ConstSpeedEnv2()

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
    for i in range(10):
        # Random action
        action = env.action_space.sample()
        action = action
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"Step {i+1}: action={action}, obs= {  '|'.join([f'{ob:.1f}' for ob in obs[-4:-1] ]) }, reward={reward:.3f}"
        )

        if terminated:
            print("Episode terminated!")
            break
        if truncated:
            print("Episode truncated!")
            break
    env.close()


def train_agent(
    seed: int = 42,
    episodes: int = 750,
    target_speeds="random",
    save_dir="runs",
    obs_dim=2,
    is_do_test=True,
):
    cfg = Config()
    cfg.episodes = episodes
    cfg.actor_lr = 1e-4  # Lower learning rate
    cfg.critic_lr = 1e-4
    cfg.noise_std = 10
    cfg.noise_std_final = 0.1
    cfg.start_random_episodes = 30
    cfg.batch_size = 128
    cfg.log_interval = 10
    cfg.seed = seed
    cfg.save_dir = save_dir

    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ConstSpeedEnv2(obs_dim=obs_dim)

    if cfg.max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.max_episode_steps)

    # Interpolation method: 'linear', 'nearest', 'previous'/'zoh', 'next',
    # and smooth methods: 'pchip', 'cubic', 'univariate', 'akima'
    option = {
        "train_coeffs": high_speed_train_params_test,
        "track_layout": default_track_layout,
        "target_speeds": target_speeds,
        "terminate_time": 15.0,
        "interp_method": "cubic",
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

    try:
        return train_td3(
            cfg, env, actor, critic, target_actor, target_critic, buffer, device, option
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        if is_do_test:
            test_agent()


def train_agents(iters=10):
    # target_speeds = {"times": [0, 5, 10, 15, 20], "speeds": [0, 10, 10, 0, 15]}
    target_speeds = {"speeds": 10}
    obs_dim = 3
    best_ep_ret = -float("inf")
    for i in range(iters):
        print(f"Training agent {i+1}/10")
        seed = np.random.randint(0, 200)
        run_id = time.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join("runs", "history", f"{run_id}_seed_{seed}")
        ep_return = train_agent(
            seed=seed,
            save_dir=save_dir,
            is_do_test=False,
            obs_dim=obs_dim,
            episodes=800,
            target_speeds=target_speeds,
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
    plot_all_histories(history_dir=os.path.join(os.path.dirname(__file__), "runs", "history"))

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
    env: ConstSpeedEnv2,
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
    act_dim = ckpt["act_dim"]
    act_high = ckpt["act_high"]
    act_low = ckpt["act_low"]
    actor = Actor(obs_dim, act_dim, act_high, act_low).to(device)
    actor.load_state_dict(ckpt["actor"])
    print(f"load model with ep_ret {ckpt['ep_ret']:.2f}")

    env = ConstSpeedEnv2(obs_dim)

    # target_speeds = {
    #     "positions": [0, 50, 120, 200, 300, 450, 600, 750, 900, 1100, 1300, 1500],
    #     "speeds": [1, 15, 15, 20, 20, 10, 25, 50, 30, 40, 15, 0],
    # }

    # target_speeds = {"speeds": 10}

    target_speeds = "random"

    option = {
        "train_coeffs": high_speed_train_params_test,
        "track_layout": default_track_layout,
        "target_speeds": target_speeds,
        "terminate_time": 50.0,
        "interp_method": "pchip",
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
    train_agents(10)
