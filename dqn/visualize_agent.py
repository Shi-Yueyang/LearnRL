import os
import torch
import gymnasium as gym
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
CHECKPOINT_PATH = os.path.join("runs", "best.pt")
ANIMATION_FPS = 30
DEVICE = "auto" # "auto" | "cpu" | "cuda"

def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if name == "cuda" else "cpu")

def build_model_from_ckpt(ckpt: dict):
    cfg = ckpt.get("cfg", {})
    env_id = cfg.get("env_id", "CartPole-v1")
    temp_env = gym.make(env_id)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    assert obs_space is not None and len(obs_space.shape) == 1
    assert hasattr(act_space, "n")
    obs_dim = obs_space.shape[0]
    n_actions = act_space.n
    temp_env.close()
    
    from train_agent import MLP
    model = MLP(obs_dim, n_actions)
    state_dict = ckpt.get("model") or ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model, env_id

def main():
    device = select_device(DEVICE)
    ckpt =  torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if ckpt is None:
        return
    model, env_id = build_model_from_ckpt(ckpt)
    model.to(device)

    env = gym.make(env_id, render_mode="rgb_array")
    print(f"Loaded checkpoint from {CHECKPOINT_PATH} | Env: {env_id}")

    frames = []
    obs, info = env.reset()
    done = False
    
    while not done:
        frames.append(env.render())
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = model(obs_tensor).argmax(1).item()
        
        step_out = env.step(action)
        obs, _, terminated, truncated, _ = step_out
        done = terminated or truncated

    env.close()

    os.makedirs('runs', exist_ok=True)
    anim_path = os.path.join('runs', 'episode_animation.mp4')

    # Faster animation: reuse a single AxesImage instead of creating a new one per frame.
    fig, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(frames[0])

    def animate(i):
        im.set_data(frames[i])  # update pixel data only
        return (im,)

    # blit=True makes updates cheaper; cache_frame_data=False avoids storing every frame twice
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(frames),
        interval=1000/ANIMATION_FPS,
        blit=True,
        repeat=False,
        cache_frame_data=False,
    )
    print('Animation created successfully.')
    ani.save(anim_path, fps=ANIMATION_FPS, writer='ffmpeg')


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)    
    main()