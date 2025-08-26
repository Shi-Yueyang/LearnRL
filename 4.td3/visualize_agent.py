import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from train_agent import Actor, SAVE_FILE

ANIMATION_FPS = 30


def build_actor(ckpt: dict):
    cfg = ckpt.get("cfg", {})
    env_id = cfg.get("env_id", "Pendulum-v1")
    temp_env = gym.make(env_id)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]
    act_high = act_space.high
    model = Actor(obs_dim, act_dim, act_high)
    model.load_state_dict(ckpt.get("actor"))
    model.eval()
    temp_env.close()
    return model, env_id


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join("runs", SAVE_FILE), map_location="cpu", weights_only=False)
    actor, env_id = build_actor(ckpt)
    actor.to(device)

    env = gym.make(env_id, render_mode="rgb_array")
    print(f"Loaded {SAVE_FILE} | Env: {env_id}")
    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        frames.append(env.render())
        obs_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor(obs_tensor).cpu().numpy()[0]
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    os.makedirs("runs", exist_ok=True)
    anim_path = os.path.join("runs", "episode_animation.gif")
    fig, ax = plt.subplots(); ax.axis("off")
    im = ax.imshow(frames[0])

    def animate(i):
        im.set_data(frames[i]); return (im,)

    ani = animation.FuncAnimation(
        fig, animate, frames=len(frames), interval=1000/ANIMATION_FPS, blit=True, repeat=False, cache_frame_data=False
    )
    print("Animation created successfully.")
    ani.save(anim_path, fps=ANIMATION_FPS, writer="pillow")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
