import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class Car1DEnv(gym.Env):
    """
    Simple 1D car environment where the car is a first-order system.
    The goal is to follow a constant target speed.
    
    State: [position, velocity, target_speed]
    Action: acceleration command (continuous)
    
    The car dynamics follow: v_next = v_current + dt * (action - damping * v_current)
    where action is the acceleration command and damping represents air resistance.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, target_speed=10.0, max_speed=20.0, max_acceleration=5.0, 
                 damping=0.1, dt=0.1, max_episode_steps=1000):
        super().__init__()
        
        # Environment parameters
        self.target_speed = target_speed
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.damping = damping  # Air resistance coefficient
        self.dt = dt  # Time step
        self.max_episode_steps = max_episode_steps
        
        # State: [position, velocity, target_speed]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -self.max_speed, 0.0]),
            high=np.array([np.inf, self.max_speed, self.max_speed]),
            dtype=np.float32
        )
        
        # Action: acceleration command
        self.action_space = spaces.Box(
            low=-self.max_acceleration,
            high=self.max_acceleration,
            shape=(1,),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.step_count = 0
        
        # For rendering
        self.render_mode = None
        self.fig = None
        self.ax = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random initial conditions
        initial_position = 0.0
        initial_velocity = self.np_random.uniform(-2.0, 2.0)
        
        # Allow some variation in target speed if desired
        if options and 'target_speed' in options:
            target_speed = options['target_speed']
        else:
            target_speed = self.target_speed
            
        self.state = np.array([initial_position, initial_velocity, target_speed], dtype=np.float32)
        self.step_count = 0
        
        info = {"target_speed": target_speed}
        
        return self.state, info
    
    def step(self, action):
        if self.state is None:
            raise ValueError("Environment must be reset before calling step()")
            
        position, velocity, target_speed = self.state
        acceleration = np.clip(action[0], -self.max_acceleration, self.max_acceleration)
        
        # First-order system dynamics with damping
        # v_next = v_current + dt * (acceleration - damping * v_current)
        new_velocity = velocity + self.dt * (acceleration - self.damping * velocity)
        new_velocity = np.clip(new_velocity, -self.max_speed, self.max_speed)
        
        # Update position
        new_position = position + self.dt * new_velocity
        
        # Update state
        self.state = np.array([new_position, new_velocity, target_speed], dtype=np.float32)
        
        # Calculate reward
        speed_error = abs(new_velocity - target_speed)
        reward = -speed_error  # Negative error as reward
        
        # Add small penalty for large accelerations to encourage smooth control
        acceleration_penalty = 0.01 * acceleration**2
        reward -= acceleration_penalty
        
        # Check termination conditions
        self.step_count += 1
        terminated = False  # This environment doesn't have natural termination
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            "speed_error": speed_error,
            "acceleration": acceleration,
            "target_speed": target_speed
        }
        
        return self.state, reward, terminated, truncated, info
    
    def render(self, mode="human"):
        if self.state is None:
            return None
            
        position, velocity, target_speed = self.state
        
        if mode == "human":
            if self.fig is None:
                plt.ion()
                self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
                
            self.ax1.clear()
            self.ax2.clear()
            
            # Plot car position
            self.ax1.plot(position, 0, 'ro', markersize=10, label='Car')
            self.ax1.set_xlim(position - 50, position + 50)
            self.ax1.set_ylim(-1, 1)
            self.ax1.set_xlabel('Position')
            self.ax1.set_title('Car Position')
            self.ax1.grid(True)
            self.ax1.legend()
            
            # Plot velocity vs target
            self.ax2.axhline(y=target_speed, color='g', linestyle='--', label=f'Target Speed: {target_speed:.1f}')
            self.ax2.plot(self.step_count, velocity, 'bo', markersize=8, label=f'Current Speed: {velocity:.1f}')
            self.ax2.set_xlim(max(0, self.step_count - 100), self.step_count + 10)
            self.ax2.set_ylim(-self.max_speed, self.max_speed)
            self.ax2.set_xlabel('Time Step')
            self.ax2.set_ylabel('Velocity')
            self.ax2.set_title('Velocity Tracking')
            self.ax2.grid(True)
            self.ax2.legend()
            
            plt.tight_layout()
            plt.pause(0.01)
            
        elif mode == "rgb_array":
            # For recording videos - create a simple plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axhline(y=target_speed, color='g', linestyle='--', label=f'Target: {target_speed:.1f}')
            ax.plot(0, velocity, 'bo', markersize=10, label=f'Current: {velocity:.1f}')
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, self.max_speed)
            ax.set_title(f'Speed Tracking - Step {self.step_count}')
            ax.legend()
            ax.grid(True)
            
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image
            
        return None
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax1 = None
            self.ax2 = None


# Test function
def test_car_env():
    """Test the car environment with random actions"""
    env = Car1DEnv(target_speed=10.0)
    
    obs, info = env.reset()
    print(f"Initial state: position={obs[0]:.2f}, velocity={obs[1]:.2f}, target={obs[2]:.2f}")
    
    for step in range(100):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 20 == 0:
            print(f"Step {step}: velocity={obs[1]:.2f}, target={obs[2]:.2f}, "
                  f"reward={reward:.2f}, speed_error={info['speed_error']:.2f}")
        
        if terminated or truncated:
            break
    
    env.close()


if __name__ == "__main__":
    test_car_env()
