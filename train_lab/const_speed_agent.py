import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt

from ddpg.train_agent import Config, Actor, Critic, train_ddpg, ReplayBuffer
from const_speed_env import ConstSpeedEnv, test_control_law
from train import high_speed_train_params_test
from track import default_track_layout


def train_agent():
    cfg = Config()
    cfg.episodes = 5000
    cfg.actor_lr = 3e-4  # Lower learning rate
    cfg.critic_lr = 3e-4
    cfg.noise_std = 15  # Lower initial noise
    cfg.noise_std_final = 5
    cfg.start_random_episodes = 30  
    cfg.batch_size = 128 
    cfg.log_interval = 10
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed)
    env = ConstSpeedEnv()
    if cfg.max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
    option = {
        "train_coeffs": high_speed_train_params_test,
        "track_layout": default_track_layout,
        "target_speed": 25.0,  # Target speed in m/s
    }
    obs_space = env.observation_space
    act_space = env.action_space
    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]
    act_high = act_space.high
    act_low = act_space.low

    actor = Actor(obs_dim, act_dim, act_high, act_low).to(device)
    critic = Critic(obs_dim, act_dim).to(device)
    # try:
    #     ckpt = torch.load(
    #         os.path.join("runs", "best.pt"), map_location="cpu", weights_only=False
    #     )
    #     actor.load_state_dict(ckpt["actor"])
    #     critic.load_state_dict(ckpt["critic"])
    #     print(f"load model with return {ckpt['mean_return']:.2f}")
    # except Exception as e:
    #     pass

    target_actor = Actor(obs_dim, act_dim, act_high, act_low).to(device)
    target_critic = Critic(obs_dim, act_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    buffer = ReplayBuffer(cfg.buffer_size, (obs_dim,), act_dim)
    try:
    
        train_ddpg(
            cfg, env, actor, critic, target_actor, target_critic, buffer, device, option
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")


def test_agent():
    env = ConstSpeedEnv()

    option = {
        "train_coeffs": high_speed_train_params_test,
        "track_layout": default_track_layout,
        "target_speed": 25.0,  # Target speed in m/s
    }
    # Reset environment
    obs, info = env.reset(seed=42, options=option)
    print(f"Initial vel: {obs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_space = env.observation_space
    act_space = env.action_space
    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]
    act_high = act_space.high
    act_low = act_space.low

    actor = Actor(obs_dim, act_dim, act_high, act_low).to(device)
    try:
        ckpt = torch.load(
            os.path.join("runs", "best.pt"), map_location="cpu", weights_only=False
        )
        actor.load_state_dict(ckpt["actor"])
        print(f"load model with return {ckpt['mean_return']:.2f}")
    except Exception as e:
        pass

    def control_law(obs):
        obs_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor(obs_tensor).cpu().numpy()[0]
        return action

    test_control_law(env, option, control_law, steps=1000, is_render=True)


def test_actor():
    env = ConstSpeedEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_space = env.observation_space
    act_space = env.action_space
    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]
    act_high = act_space.high
    act_low = act_space.low

    actor = Actor(obs_dim, act_dim, act_high, act_low).to(device)
    state_dict = actor.state_dict()
    try:
        ckpt = torch.load(
            os.path.join("runs", "best.pt"), map_location="cpu", weights_only=False
        )
        actor.load_state_dict(ckpt["actor"])
        print(f"load model with return {ckpt['mean_return']:.2f}")
    except Exception as e:
        pass
    input = np.linspace(-30, 30, 200).reshape(-1,1)
    output = []
    for i in input:
        obs_tensor = torch.from_numpy(np.array(i)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor(obs_tensor).cpu().numpy()[0]
        output.append(float(action))
    output = np.array(output)
    plt.plot(input, output)
    plt.show()

def test_critic():
    env = ConstSpeedEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    obs_space = env.observation_space
    act_space = env.action_space
    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]

    critic = Critic(obs_dim, act_dim).to(device)
    try:
        ckpt = torch.load(
            os.path.join("runs", "best.pt"), map_location="cpu", weights_only=False
        )
        critic.load_state_dict(ckpt["critic"])
        print(f"load critic with return {ckpt['mean_return']:.2f}")
    except Exception as e:
        print(f"No saved model found: {e}")
    
    input_act = np.linspace(-100, 100, 100)  # Fixed range function
    obs = 0.0  # velocity error = 0
    output = []
    for i in input_act:
        obs_tensor = torch.from_numpy(np.array([obs])).float().unsqueeze(0).to(device)
        act_tensor = torch.from_numpy(np.array([i])).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_value = critic(obs_tensor, act_tensor).cpu().numpy()[0]
        output.append(float(q_value))
    output = np.array(output)
    plt.plot(input_act, output)
    plt.xlabel('Action')
    plt.ylabel('Q-value')
    plt.title('Critic Q-values vs Actions (obs=0)')
    plt.show()
    
if __name__ == "__main__":  # pragma: no cover
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    test_agent()  # Change to train the agent
