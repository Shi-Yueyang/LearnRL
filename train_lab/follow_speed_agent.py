import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt

from ddpg.train_agent import Config, Actor, Critic, train_ddpg, ReplayBuffer
from train_lab.follow_speed_env import ConstSpeedEnv, test_control_law
from train import high_speed_train_params_test
from track import default_track_layout


def train_agent():
    cfg = Config()
    cfg.episodes = 5000
    cfg.actor_lr = 3e-4  # Lower learning rate
    cfg.critic_lr = 3e-4
    cfg.noise_std = 15  # Lower initial noise
    cfg.noise_std_final = 0.1
    cfg.start_random_episodes = 30
    cfg.batch_size = 128
    cfg.log_interval = 10
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed)
    env = ConstSpeedEnv()
    if cfg.max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
    target_speeds = {"times": [0, 10, 15, 20], "speeds": [0, 10, 15, 30]}
    option = {
        "train_coeffs": high_speed_train_params_test,
        "track_layout": default_track_layout,
        "target_speed": target_speeds,
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
    # target_speeds = {
    #     'times': [0, 2, 5, 8, 12, 15, 18, 22, 25, 28, 32, 35],
    #     'speeds': [0, 12, 25, 8, 30, 5, 20, 40, 15, 35, 10, 0]
    # }
    target_speeds = {
        "positions": [0, 50, 120, 200, 300, 450, 600, 750, 900, 1100, 1300, 1500],
        "speeds": [1, 15, 15, 20, 20, 10, 25, 50, 30, 40, 15, 0],
    }

    target_speeds = {"times": [0, 7, 10, 15], "speeds": [0, 10, 10, 0]}

    option = {
        "train_coeffs": high_speed_train_params_test,
        "track_layout": default_track_layout,
        "target_speeds": target_speeds,
    }
    # Reset environment
    obs, info = env.reset(seed=42, options=option)
    print(f"Initial err: {obs}")

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
            os.path.join("runs", "followspeed1.pt"),
            map_location="cpu",
            weights_only=False,
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

    test_control_law(
        env, option, control_law, steps=1000, is_render=True, x_axes="time"
    )


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
    input = np.linspace(-30, 30, 200).reshape(-1, 1)
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
    for obs in range(-20, 20, 4):  # Test for different obs values
        output = []
        for i in input_act:
            obs_tensor = (
                torch.from_numpy(np.array([obs])).float().unsqueeze(0).to(device)
            )
            act_tensor = torch.from_numpy(np.array([i])).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_value = critic(obs_tensor, act_tensor).cpu().numpy()[0]
            output.append(float(q_value))
        output = np.array(output)
        plt.plot(input_act, output)
    plt.xlabel("Action")
    plt.ylabel("Q-value")
    plt.title("Critic Q-values vs Actions (obs=0)")
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    test_agent()  # Change to train the agent
