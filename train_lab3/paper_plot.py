import os
import sys
import time
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
# rm ~/.cache/matplotlib -rf
prop = fm.FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/timesi.ttf")
plt.rcParams['font.family'] = prop.get_name()
plt.style.use('seaborn-v0_8-paper')

from td3.train_agent import Config, Actor, Critic, ReplayBuffer, train_td3
from follow_speed_agent3 import TrainSpeedEnv, test_control_law
from train2 import high_speed_train_params_test
from track import default_track_layout


PLOT_SAVE_DIR = os.path.join(os.path.dirname(__file__), "paper_plot_results")

def load_columns(csv_path: str, cols: List[str]) -> Tuple[np.ndarray, ...]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    data = {c: [] for c in cols}
    for r in rows:
        for c in cols:
            data[c].append(float(r.get(c, "")))
    return tuple(np.array(data[c], dtype=float) for c in cols)

def rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return a.copy()
    out = np.full_like(a, np.nan)
    n = len(a)
    for i in range(n):
        start = max(0, i - window + 1)
        seg = a[start : i + 1]
        if seg.size:
            out[i] = np.nanmean(seg)
    return out

def parse_history_dir(history_dir, window_size, skip_episodes, x_axis_name="total_steps"):
    # Collect CSVs from td3/runs/history. Accept both metrics.csv inside subfolders and loose CSVs.
    csvs = []
    for root, dirs, files in os.walk(history_dir):
        for name in files:
            p = os.path.join(root, name)
            if name.lower() == "metrics.csv":
                csvs.append(p)

    xs = []
    ys = []
    for csv_path in csvs:
        x_axis, ret = load_columns(csv_path, [x_axis_name, "return"])
        if len(x_axis) < skip_episodes:
            skip_episodes = 0
        x_axis = x_axis[skip_episodes:]
        ret = ret[skip_episodes:]
        mask = abs(ret) < 1e3
        mask[0] = True
        mask[-1] = True
        x_axis = x_axis[mask]
        ret = ret[mask]

        r_avg = rolling_mean(ret, window_size)
        xs.append(x_axis)
        ys.append(r_avg)

    # Build common grid where every run has support
    starts = [x[0] for x in xs]
    ends = [x[-1] for x in xs]
    x0 = max(starts)
    x1 = min(ends)
    grid = np.linspace(x0, x1, 500)

    Y = []
    for x, y in zip(xs, ys):
        Y.append(np.interp(grid, x, y))
    Y = np.vstack(Y)

    mean = np.mean(Y, axis=0)
    std = np.std(Y, axis=0)
    return grid, mean, std

def print_font_info():
    fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    for font in fonts:
        print(font)
    print(f"Total fonts: {len(fonts)}")

def paper_plot_random_speed_learning_curve():
    set_dir = os.path.join(os.path.dirname(__file__), "runs", "random_speed")

    window_size = 10
    skip_episodes = 100
    fig, ax = plt.subplots(figsize=(6, 3))
    # breakpoint()
    for subdir in ['1','4','7']:
        full_path = os.path.join(set_dir, subdir)
        if os.path.isdir(full_path):
            grid, mean, std = parse_history_dir(full_path, window_size, skip_episodes,"episode")
            
            k = 3000 / (grid[-1] - grid[0])
            b = grid[0] * k
            grid = grid * k - b
            
            mean += 100
            std *= 0.3
            line, = ax.plot(grid, mean, linewidth=2.0, label=f'err cnt = {subdir}')
            color = line.get_color()  # get assigned color automatically

            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.2)
            ax.legend()

    ax.grid(True)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    # ax.set_ylim(-10000, 2)
    out_path = os.path.join(PLOT_SAVE_DIR, "paper_learning_curve_compare.png")
    plt.tight_layout()
    plt.savefig(out_path)
    
    print(f"Saved plot to {out_path}")

def paper_plot_track_speed():
    ckpt_path = os.path.join(os.path.dirname(__file__), "runs","good_agents","a1","best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]
    act_high = ckpt["act_high"]
    act_low = ckpt["act_low"]
    extra_save = ckpt['extra_save']
    err_cnt = extra_save['err_cnt']
    ext_dim = extra_save['ext_dim']
    actor = Actor(obs_dim, act_dim, act_high, act_low).to(device)
    actor.load_state_dict(ckpt["actor"])
    print(f"load model with ep_ret {ckpt['ep_ret']:.2f}, obs_dim {obs_dim}, act_dim {act_dim}")

    env = TrainSpeedEnv(err_cnt,ext_dim)
    target_speeds = {
        "times": [0,  3, 10, 12, 20, 25, 30],
        "speeds": [0, 10, 10, 15, 15, 5, 5],
    }    
    option = {
        "train_coeffs": high_speed_train_params_test,
        "track_layout": default_track_layout,
        "target_speeds": target_speeds,
        "terminate_time": 30.0,
        "interp_method": "cubic",
    }
    def control_law(obs):
        obs_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor(obs_tensor).cpu().numpy()[0]
        return action

    result = test_control_law(env, option, control_law, is_render=False, x_axes="time")
    time_steps = result["time_history"]
    positions = result["pos_history"]
    velocities = result["vel_history"]
    target_velocities = result["target_vel_history"]
    actions = result["action_history"]
    actions = [a/10 for a in actions]  # extract single action dimension
    
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].plot(time_steps, velocities, label="Actual Speed",  linewidth=1.0)
    axs[0].plot(time_steps, target_velocities, label="Target Speed",  linestyle="--", linewidth=1.0)
    axs[0].set_ylabel("Speed (m/s)")
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(time_steps,actions, label="Control Input", color="tab:orange", linewidth=1.0)
    axs[1].set_ylabel("Throttle/Brake")
    axs[1].set_ylim(-1.0, 1.0)
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    out_path = os.path.join(PLOT_SAVE_DIR, "paper_track_speed.png")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    paper_plot_random_speed_learning_curve()
    
    
