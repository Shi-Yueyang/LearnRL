import gymnasium as gym
from gymnasium import spaces
import numpy as np
from train import Train
from track import Track
from typing import Callable, Dict, Any
import matplotlib.pyplot as plt


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
        self.target_speed = options.get("target_speed", 20.0)  # m/s
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
        velocity_error = vel - self.target_speed
        return np.array([velocity_error], dtype=np.float32)

    def _calculate_reward(self, action):
        pos, vel, acc, time = self.state

        # Normalized velocity tracking error
        velocity_error = abs(vel - self.target_speed)
        velocity_reward = -velocity_error

        # Small bonus for reaching near target
        if velocity_error < 1.0:
            velocity_reward += 10.0
        if vel > 0:
            velocity_reward += 2


        return velocity_reward

    def _is_terminated(self):
        pos, vel, acc, time = self.state
        if time >= 15.0:
            return True
        return False

    def _is_truncated(self):
        pos, vel, acc, time = self.state
        if vel > self.target_speed * 3:
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


def visualize_control_law(result: Dict[str, list]):
    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    # Position, Velocity, Acceleration plots
    plots = [
        (
            ax1,
            [
                (result["time_history"], result["vel_history"], "lime", "vel"),
            ],
            "vel (m/s)",
        ),
        (
            ax2,
            [
                (result["time_history"], result["action_history"], "orange", "action"),
            ],
            "Control",
        ),
        (
            ax3,
            [
                (result["time_history"], result["reward_history"], "orange", "reward"),
            ],
            "Reward",
        ),
    ]

    for ax, datas, ylabel in plots:
        for data in datas:
            ax.plot(data[0], data[1], color=data[2], label=data[3])
        ax.set_title(f"{ylabel.split()[0]} over Time", color="white")
        ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel(ylabel, color="white")
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
) -> Dict[str, list]:
    vel_err_history = []
    vel_history = []
    time_history = []
    action_history = []
    reward_history = []
    obs, info = env.reset(seed=42, options=env_option)
    for i in range(steps):
        # Get action from the control law
        action = control_law(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        state = env.state # pos vel acc time
        vel_err_history.append(float(obs))
        vel_history.append(state[1])
        time_history.append(i)
        action_history.append(action)
        reward_history.append(reward)
        if terminated or truncated:
            break
    result = {
        "vel_err_history": vel_err_history,
        "vel_history": vel_history,
        "time_history": time_history,
        "action_history": action_history,
        "reward_history": reward_history,
    }
    if is_render:
        visualize_control_law(result)
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
