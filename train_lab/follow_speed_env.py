import gymnasium as gym
from gymnasium import spaces
import numpy as np
from train import Train
from track import Track
from typing import Callable, Dict, Any
import matplotlib.pyplot as plt
from typing import Literal


class ConstSpeedEnv(gym.Env):

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
            shape=(1,),  # [position, velocity, acceleration, time]
            dtype=np.float32,
        )

        self.render_mode = render_mode
        self.reached_steps = 0
        # Environment state variables
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.train = Train(options["train_coeffs"])
        self.track = Track(options["track_layout"])
        self.target_speeds = options.get("target_speeds", 25.0)  # m/s

        self.at_target_counter = 0
        self.steps = 0

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
        return np.array([velocity_error], dtype=np.float32)

    def _calculate_reward(self, action):
        pos, vel, acc, time = self.state

        # Normalized velocity tracking error
        target_speed = self.get_target_speed()

        velocity_error = abs(vel - target_speed)
        velocity_reward = -velocity_error
        if target_speed > 1e-3:
            velocity_reward = velocity_reward / target_speed * 2

        return velocity_reward

    def _is_terminated(self):
        pos, vel, acc, time = self.state
        if time >= 15.0:
            return True
        return False

    def _is_truncated(self):
        pos, vel, acc, time = self.state
        target_speed = self.get_target_speed()
        if vel > target_speed * 3:
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
        """Render the environment (optional)"""
        if self.render_mode == "human":
            print(f"Step: {self.steps}, State: {self.state}")

    def close(self):
        """Clean up resources"""
        pass


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
    env: ConstSpeedEnv,
    env_option: Dict[str, Any],
    control_law: Callable[[np.ndarray], np.ndarray],
    steps: int = 100,
    is_render: bool = True,
    x_axes: Literal["time", "pos"] = "time",
) -> Dict[str, list]:
    obs, info = env.reset(seed=42, options=env_option)

    pos_history = []
    vel_err_history = []
    vel_history = []
    target_vel_history = []
    time_history = []
    action_history = []
    reward_history = []
    acc_history = []

    for i in range(steps):
        # Get action from the control law
        action = control_law(obs) if i > 0 else np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        target_speed = env.get_target_speed()

        state = env.state  # pos vel acc time
        vel_err_history.append(float(obs))
        vel_history.append(state[1])
        target_vel_history.append(target_speed)
        acc_history.append(state[2])
        time_history.append(state[3])
        action_history.append(action[0])
        reward_history.append(reward)
        pos_history.append(state[0])
        if truncated:
            break
    result = {
        "pos_history": pos_history,
        "vel_err_history": vel_err_history,
        "vel_history": vel_history,
        "target_vel_history": target_vel_history,
        "acc_history": acc_history,
        "time_history": time_history,
        "action_history": action_history,
        "reward_history": reward_history,
    }
    if is_render:
        visualize_control_law(result, x_axes)
    return result


# Example usage and testing
def main():
    from train import high_speed_train_params_test
    from track import default_track_layout

    """Test the custom environment"""
    env = ConstSpeedEnv()

    print("Testing environment...")
    option = {
        "train_coeffs": high_speed_train_params_test,
        "track_layout": default_track_layout,
        "target_speed": 25.0,  # Target speed in m/s
    }
    # Reset environment
    obs, info = env.reset(seed=42, options=option)
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")

    # Run a few steps
    for i in range(10):
        # Random action
        action = env.action_space.sample()
        action = action / 100
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {i+1}: action={action}, obs={obs}, reward={reward:.3f}")

        if terminated or truncated:
            print("Episode finished!")
            break

    env.close()


if __name__ == "__main__":
    main()
