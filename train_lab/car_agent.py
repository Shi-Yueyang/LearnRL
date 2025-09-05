import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path to import ddpg
sys.path.append('..')
from ddpg.train_agent import Config, Actor, Critic, train_ddpg, ReplayBuffer

# Import our car environment
from car_env import Car1DEnv


def create_car_config():
    """Create configuration for training DDPG on the car environment."""
    cfg = Config()
    
    # Environment settings
    cfg.env_id = "Car1D-v0"  # Custom name for our environment
    cfg.episodes = 500
    cfg.max_episode_steps = 2000
    
    # DDPG hyperparameters optimized for the car task
    cfg.gamma = 0.95  # Slightly lower gamma for faster convergence
    cfg.tau = 0.01  # Slightly higher for faster target network updates
    cfg.buffer_size = 100_000
    cfg.batch_size = 128
    cfg.actor_lr = 1e-3
    cfg.critic_lr = 1e-3
    
    # Exploration parameters
    cfg.start_random_episodes = 20
    cfg.noise_std = 1.0  # Higher initial noise for acceleration commands
    cfg.noise_std_final = 0.1
    cfg.noise_decay_episodes = 500
    cfg.min_buffer = 2_000
    cfg.updates_per_step = 1
    
    # Logging and saving
    cfg.save_dir = "runs"
    cfg.log_interval = 50
    cfg.seed = 42
    
    return cfg


def evaluate_agent(env, actor, device, num_episodes=5, target_speed=10.0):
    """Evaluate the trained agent."""
    actor.eval()
    total_rewards = []
    speed_errors = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset(options={'target_speed': target_speed})
        ep_reward = 0
        ep_speed_errors = []
        done = False
        
        while not done:
            obs_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor(obs_tensor).cpu().numpy()[0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_speed_errors.append(info['speed_error'])
        
        total_rewards.append(ep_reward)
        speed_errors.extend(ep_speed_errors)
    
    actor.train()
    return np.mean(total_rewards), np.mean(speed_errors), np.std(speed_errors)


def plot_training_progress(rewards_history, speed_errors_history):
    """Plot training progress."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot episode rewards
    episodes = range(1, len(rewards_history) + 1)
    ax1.plot(episodes, rewards_history, 'b-', alpha=0.7, label='Episode Reward')
    
    # Add moving average
    if len(rewards_history) > 50:
        window = 50
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window})')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress - Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot speed tracking errors
    if speed_errors_history:
        episodes_errors = range(1, len(speed_errors_history) + 1)
        ax2.plot(episodes_errors, speed_errors_history, 'g-', alpha=0.7, label='Speed Error')
        
        if len(speed_errors_history) > 50:
            window = 50
            moving_avg_err = np.convolve(speed_errors_history, np.ones(window)/window, mode='valid')
            ax2.plot(episodes_errors[window-1:], moving_avg_err, 'orange', linewidth=2, 
                    label=f'Moving Average ({window})')
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Speed Error')
        ax2.set_title('Training Progress - Speed Tracking Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('runs/training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()


def train_agent():
    """Main training function."""
    print("Training DDPG agent on Car1D environment...")
    
    # Create configuration
    cfg = create_car_config()
    
    # Setup directories and device
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Create environment
    env = Car1DEnv(
        target_speed=10.0,
        max_speed=20.0,
        max_acceleration=8.0,  # Allow higher accelerations for training
        damping=0.1,
        dt=0.1,
        max_episode_steps=cfg.max_episode_steps
    )
    
    print(f"Environment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Get environment dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_high = env.action_space.high
    act_low = env.action_space.low
    
    print(f"  Obs dim: {obs_dim}, Act dim: {act_dim}")
    print(f"  Action bounds: [{act_low[0]:.2f}, {act_high[0]:.2f}]")
    
    # Create networks
    actor = Actor(obs_dim, act_dim, act_high, act_low).to(device)
    critic = Critic(obs_dim, act_dim).to(device)
    target_actor = Actor(obs_dim, act_dim, act_high, act_low).to(device)
    target_critic = Critic(obs_dim, act_dim).to(device)
    
    # Initialize target networks
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    
    # Create replay buffer
    buffer = ReplayBuffer(cfg.buffer_size, (obs_dim,), act_dim)
    
    

    # Train the agent
    try:
        train_ddpg(cfg, env, actor, critic, target_actor, target_critic, buffer, device,save_file="best_car.pt")
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        return
    
def test_agent():
    cfg = create_car_config()
    
    # Setup directories and device
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seeds for reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Create environment
    env = Car1DEnv(
        target_speed=10.0,
        max_speed=20.0,
        max_acceleration=8.0,  # Allow higher accelerations for training
        damping=0.1,
        dt=0.1,
        max_episode_steps=cfg.max_episode_steps
    )

    
    # Get environment dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_high = env.action_space.high
    act_low = env.action_space.low

    # Create networks
    actor = Actor(obs_dim, act_dim, act_high, act_low).to(device)
    # Load the best model for evaluation
    try:
        checkpoint = torch.load(os.path.join(cfg.save_dir, "best_car.pt"))
        actor.load_state_dict(checkpoint["actor"])
        print(f"Loaded best model with mean return: {checkpoint['mean_return']:.2f}")
        
        # Evaluate the trained agent
        print("\nEvaluating trained agent...")
        for target_speed in [5.0, 10.0, 15.0]:
            mean_reward, mean_error, std_error = evaluate_agent(
                env, actor, device, num_episodes=10, target_speed=target_speed
            )
            print(f"Target speed {target_speed:4.1f}: "
                  f"Reward={mean_reward:7.2f}, "
                  f"Error={mean_error:5.3f}±{std_error:5.3f}")
        
        # Run demonstration episode and collect data for plotting
        print("\nRunning demonstration episode...")
        obs, _ = env.reset(options={'target_speed': 10.0})
        
        # Data collection lists
        speeds = []
        target_speeds = []
        actions = []
        rewards = []
        time_steps = []
        positions = []
        speed_errors = []
        
        done = False
        step = 0
        
        while not done and step < 500:
            obs_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor(obs_tensor).cpu().numpy()[0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Collect data
            speeds.append(obs[1])  # Current speed
            target_speeds.append(obs[2])  # Target speed
            actions.append(action[0])  # Acceleration command
            rewards.append(reward)
            time_steps.append(step * env.dt)  # Convert to real time
            positions.append(obs[0])  # Position
            speed_errors.append(info['speed_error'])
            
            step += 1
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Speed tracking
        axes[0, 0].plot(time_steps, speeds, 'b-', linewidth=2, label='Actual Speed')
        axes[0, 0].plot(time_steps, target_speeds, 'r--', linewidth=2, label='Target Speed')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Speed (m/s)')
        axes[0, 0].set_title('Speed Tracking Performance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Action (acceleration commands)
        axes[0, 1].plot(time_steps, actions, 'g-', linewidth=2, label='Acceleration Command')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Acceleration (m/s²)')
        axes[0, 1].set_title('Control Actions')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Rewards
        axes[1, 0].plot(time_steps, rewards, 'm-', linewidth=2, label='Reward')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title('Reward Signal')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Speed error
        axes[1, 1].plot(time_steps, speed_errors, 'orange', linewidth=2, label='Speed Error')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Speed Error (m/s)')
        axes[1, 1].set_title('Speed Tracking Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('runs/test_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print(f"\nTest Episode Summary:")
        print(f"  Duration: {time_steps[-1]:.1f} seconds ({len(time_steps)} steps)")
        print(f"  Final speed: {speeds[-1]:.2f} m/s (target: {target_speeds[-1]:.2f} m/s)")
        print(f"  Mean speed error: {np.mean(speed_errors):.3f} ± {np.std(speed_errors):.3f} m/s")
        print(f"  Total reward: {sum(rewards):.2f}")
        print(f"  Mean reward per step: {np.mean(rewards):.3f}")
        print(f"  Final position: {positions[-1]:.1f} m")
        print(f"  Mean action magnitude: {np.mean(np.abs(actions)):.3f} m/s²")
        
    except FileNotFoundError:
        print("No saved model found. Training may not have completed successfully.")
        print("Run train_agent() first to train a model.")
    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    train_agent()
