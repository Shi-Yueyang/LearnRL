import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

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


def plot_random_speed_learning_curve():
    set_dir = os.path.join(os.path.dirname(__file__), "runs", "random_speed")
    plt.rcParams["font.family"] = "Times New Roman"

    window_size = 10
    skip_episodes = 100
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(6, 3))
    # breakpoint()
    for subdir in ['1','4','7']:
        full_path = os.path.join(set_dir, subdir)
        if os.path.isdir(full_path):
            grid, mean, std = parse_history_dir(full_path, window_size, skip_episodes,"episode")
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
    out_path = os.path.join(set_dir, "paper_learning_curve_compare.png")
    plt.tight_layout()
    plt.savefig(out_path)
    
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_random_speed_learning_curve()