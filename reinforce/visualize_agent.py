import os
import time
from dataclasses import dataclass
from typing import Optional
from train_agent import PolicyNet

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation


CHECKPOINT_PATH = "best.pt"
ANIMATION_FPS = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def build_model(ckpt: dict):
    cfg = ckpt.get("cfg", {})
    env_id = cfg.get("env_id", "CartPole-v1")
    temp_env = gym.make(env_id)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    obs_dim = obs_space.shape[0]
    n_actions = act_space.n
    model = PolicyNet(obs_dim, n_actions).to(device)
    model.load_state_dict(ckpt.get("model"))
    model.eval()
    temp_env.close()
    return model, env_id


def main():
    ckpt = torch.load(
        os.path.join("runs", CHECKPOINT_PATH), map_location="cpu", weights_only=True
    )
    policy, env_id = build_model(ckpt)
    env = gym.make(env_id, render_mode="rgb_array")
    print(f"Loaded {CHECKPOINT_PATH} | Env: {env_id}")
    frames = []
    obs, _ = env.reset()
    done = False

    while not done:
        frames.append(env.render())
        obs_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
        logits = policy(obs_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        obs, _, terminated, truncated, _ = env.step(int(action.item()))
        done = terminated or truncated

    env.close()
    os.makedirs("runs", exist_ok=True)
    anim_path = os.path.join("runs", f"{CHECKPOINT_PATH}.gif")

    fig, ax = plt.subplots()
    ax.axis("off")
    im = ax.imshow(frames[0])

    def animate(i):
        im.set_data(frames[i])  # update pixel data only
        return (im,)

    # blit=True makes updates cheaper; cache_frame_data=False avoids storing every frame twice
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(frames),
        interval=1000 / ANIMATION_FPS,
        blit=True,
        repeat=False,
        cache_frame_data=False,
    )
    print("Animation created successfully.")
    ani.save(anim_path, fps=ANIMATION_FPS, writer="pillow")


if __name__ == "__main__":  # pragma: no cover
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
