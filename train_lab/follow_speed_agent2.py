import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from train import Train
from track import Track
from typing import Callable, Dict, Any
import matplotlib.pyplot as plt
from typing import Literal
from collections import deque


from td3.train_agent import Config, Actor, Critic, ReplayBuffer, train_td3
from train import high_speed_train_params_test
from track import default_track_layout
import torch

ERROR_COUNT = 3


class ConstSpeedEnv2(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
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
            shape=(ERROR_COUNT,),  # [position, velocity, acceleration, time]
            dtype=np.float32,
        )
        self.recent_errors = deque([0] * ERROR_COUNT, maxlen=ERROR_COUNT)
        self.render_mode = render_mode
        self.reached_steps = 0
        # Environment state variables
        self.state = None
        self.previous_action = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.train = Train(options["train_coeffs"])
        self.track = Track(options["track_layout"])
        self.target_speeds = options.get("target_speeds")  # m/s

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
        action /= 10
        # Don't scale action here - let the actor learn the proper scaling
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

        if isinstance(self.target_speeds, dict):
            if "positions" in self.target_speeds:
                target_speed = np.interp(
                    pos,
                    self.target_speeds["positions"],
                    self.target_speeds["speeds"],
                    left=self.target_speeds["speeds"][0],
                    right=self.target_speeds["speeds"][-1],
                ).astype(np.float32)

            elif "times" in self.target_speeds:
                target_speed = np.interp(
                    time,
                    self.target_speeds["times"],
                    self.target_speeds["speeds"],
                    left=self.target_speeds["speeds"][0],
                    right=self.target_speeds["speeds"][-1],
                ).astype(np.float32)
        else:
            target_speed = self.target_speeds
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
        velocity_reward = -((velocity_error) ** 2) / 10.0  # Scale to keep reward manageable
        
        # --- Action penalties ---
        action_change = action - self.previous_action
        action_smoothness_penalty = -0.02 * (action_change**2)
        
        survive_reward = 0.1
        self.previous_action = action

        return velocity_reward  + survive_reward

    def _is_terminated(self):
        pos, vel, acc, time = self.state
        if time >= 15.0:
            return True
        return False

    def _is_truncated(self):
        pos, vel, acc, time = self.state
        target_speed = self.get_target_speed()
        
        # Truncate if velocity difference is too large
        velocity_error = abs(vel - target_speed)
        max_allowed_error = max(5.0, abs(target_speed) * 0.5)  # 5 m/s or 50% of target speed, whichever is larger
        
        if velocity_error > max_allowed_error and time > 2.0:  # Allow some time for initial adjustment
            return True
        
        # Truncate if velocity becomes negative when target is positive (or vice versa)
        if target_speed > 1.0 and vel < -1.0:  # Going backwards when should go forward
            return True
        
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
    from train import high_speed_train_params_test
    from track import default_track_layout

    """Test the custom environment"""
    env = ConstSpeedEnv2()

    print("Testing environment...")
    option = {
        "train_coeffs": high_speed_train_params_test,
        "track_layout": default_track_layout,
        "target_speed": 25.0,  # Target speed in m/s
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


def train_agent():
    cfg = Config()
    cfg.episodes = 5000
    cfg.actor_lr = 1e-4  # Lower learning rate
    cfg.critic_lr = 1e-4
    cfg.noise_std = 10 
    cfg.noise_std_final = 0.1
    cfg.start_random_episodes = 30
    cfg.batch_size = 128
    cfg.log_interval = 10
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ConstSpeedEnv2()

    if cfg.max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.max_episode_steps)

    target_speeds = {"times": [0, 7, 10, 15], "speeds": [0, 10, 10, 0]}

    option = {
        "train_coeffs": high_speed_train_params_test,
        "track_layout": default_track_layout,
        "target_speeds": target_speeds,
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

        train_td3(
            cfg, env, actor, critic, target_actor, target_critic, buffer, device, option
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
        test_agent()


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


def test_agent():
    env = ConstSpeedEnv2()
    # target_speeds = {
    #     'positions': [0, 50, 120, 200, 300, 450, 600, 750, 900, 1100, 1300, 1500],
    #     'speeds': [1, 15, 15, 20, 20, 10, 25, 50, 30, 40, 15, 0]
    # }

    target_speeds = {"times": [0, 7, 10, 15], "speeds": [0, 10, 10, 0]}

    # target_speeds = 25
    option = {
        "train_coeffs": high_speed_train_params_test,
        "track_layout": default_track_layout,
        "target_speeds": target_speeds,
    }
    # Reset environment
    obs, _ = env.reset(seed=42, options=option)
    print(f"Initial err: {  ' |'.join([f'{ob:.1f}' for ob in obs ]) }")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_space = env.observation_space
    act_space = env.action_space
    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]
    act_high = act_space.high
    act_low = act_space.low

    actor = Actor(obs_dim, act_dim, act_high, act_low).to(device)

    ckpt = torch.load(
        os.path.join("runs", "best.pt"), map_location="cpu", weights_only=False
    )
    actor.load_state_dict(ckpt["actor"])
    print(f"load model with ep_ret {ckpt['ep_ret']:.2f}")

    def control_law(obs):
        obs_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor(obs_tensor).cpu().numpy()[0]
        return action

    test_control_law(env, option, control_law, is_render=True, x_axes="time")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    train_agent()
