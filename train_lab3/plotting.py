# ...existing code...
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


HISTORY_PATH = os.path.join(os.path.dirname(__file__), "runs", "history","7")
SET_PATH = os.path.join(os.path.dirname(__file__), "runs", "history")
CSV_PATH = os.path.join(os.path.dirname(__file__), "runs", "history","20250921-094128","metrics.csv")
WINDOW = 10  # episodes
SKIP = 200
X_AXIS_NAME = "episode"  # total_steps or episode

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


def plot_learning_curve(csv_path = CSV_PATH, window_size = WINDOW, skip_episodes = SKIP):
    # Resolve CSV path: prefer local train_lab2/runs/metrics.csv, else fallback to td3/runs/metrics.csv
    
    episode, total_steps, ret = load_columns(csv_path, ["episode", "total_steps", "return"])
    if len(episode) < skip_episodes:
        skip_episodes = 0
    episode = episode[skip_episodes:]
    total_steps = total_steps[skip_episodes:]
    ret = ret[skip_episodes:]

    x = total_steps
    r_avg = rolling_mean(ret, window_size)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, ret, color="tab:gray", alpha=0.25, label="return (per episode)")
    ax.plot(x, r_avg, color="tab:blue", linewidth=1.8, label=f"rolling mean {window_size}")

    ax.set_xlabel("Total steps")
    ax.set_ylabel("Return")
    ax.set_title(f"TD3 learning curve\n{os.path.basename(csv_path)}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    out_dir = os.path.dirname(csv_path)
    out_path = os.path.join(out_dir, "learning_curve.png")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

def plot_learning_curve_with_shade(history_dir = HISTORY_PATH, window_size = WINDOW, skip_episodes = SKIP):
    # print(plt.style.available) 
    # Collect CSVs from td3/runs/history. Accept both metrics.csv inside subfolders and loose CSVs.
    grid, mean, std = parse_history_dir(history_dir, window_size, skip_episodes)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot aggregate mean and shaded std
    ax.plot(grid, mean, color="tab:blue", linewidth=2.0, label=f"mean")
    ax.fill_between(grid, mean - std, mean + std, color="tab:blue", alpha=0.2, label="±1 std")

    ax.set_xlabel("Total steps")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    out_path = os.path.join(history_dir, "learning_curve_all.png")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

def plot_multiple_learning_curves_with_shade(set_dir=SET_PATH,window_size=WINDOW, skip_episodes=SKIP, x_axis_name=X_AXIS_NAME):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 5))
    for subdir in os.listdir(set_dir):
        full_path = os.path.join(set_dir, subdir)
        if os.path.isdir(full_path):
            grid, mean, std = parse_history_dir(full_path, window_size, skip_episodes,X_AXIS_NAME)
            
            line, = ax.plot(grid, mean, linewidth=2.0, label=subdir)
            color = line.get_color()  # get assigned color automatically

            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.2)
            ax.legend()
            plt.tight_layout()

    ax.grid(True)
    ax.set_xlabel("Total steps")
    ax.set_ylabel("Return")
    # ax.set_ylim(-10000, 2)
    out_path = os.path.join(set_dir, "learning_curve_compare.png")
    plt.savefig(out_path)
    
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_multiple_learning_curves_with_shade()