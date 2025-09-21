# ...existing code...
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


HISTORY = os.path.join(os.path.dirname(__file__), "runs", "set2")
CSV_PATH = os.path.join(os.path.dirname(__file__), "runs", "history","20250921-094128","metrics.csv")
WINDOW = 10  # episodes
SKIP = 75


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


def plot_all_histories(history_dir = HISTORY, window_size = WINDOW, skip_episodes = SKIP):
    # print(plt.style.available) 
    # Collect CSVs from td3/runs/history. Accept both metrics.csv inside subfolders and loose CSVs.
    csvs = []
    print(os.path.exists(history_dir))
    for root, dirs, files in os.walk(history_dir):
        for name in files:
            p = os.path.join(root, name)
            if name.lower() == "metrics.csv":
                csvs.append(p)

    xs = []
    ys = []
    for csv_path in csvs:
        total_steps, ret = load_columns(csv_path, ["total_steps", "return"])
        if len(total_steps) < skip_episodes:
            skip_episodes = 0
        total_steps = total_steps[skip_episodes:]
        ret = ret[skip_episodes:]
        r_avg = rolling_mean(ret, window_size)
        xs.append(total_steps)
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

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot aggregate mean and shaded std
    ax.plot(grid, mean, color="tab:blue", linewidth=2.0, label=f"mean (n={len(xs)})")
    ax.fill_between(grid, mean - std, mean + std, color="tab:blue", alpha=0.2, label="±1 std")

    ax.set_xlabel("Total steps")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    out_path = os.path.join(history_dir, "learning_curve_all.png")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    plot_all_histories()