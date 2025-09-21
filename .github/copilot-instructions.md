# Copilot instructions for LearnRL

Purpose: help AI coding agents be productive quickly in this small RL examples repo.

Summary
- This repo contains minimal reference implementations for DQN (`dqn/`) and REINFORCE (`reinforce/`) targeting Gym(Gymnasium) environments (default: `CartPole-v1`).
- Models and run artifacts are written to `runs/` (e.g. `runs/best.pt`). Visualization scripts read those checkpoints.

Key files and patterns
- `dqn/train_agent.py` — compact CLI-style trainer supporting vectorized envs (`--batch-envs`), target networks, replay buffer. Uses argparse-style flags; check `dqn/README.md` for flag summaries and examples.
- `dqn/visualize_agent.py` — loads a checkpoint (expects `checkpoint['model']` or raw state_dict) and creates an MP4 by rendering the env with `render_mode="rgb_array"`.
- `reinforce/train_agent.py` — single-file trainer using a `Config` dataclass (edit values in-file to change defaults). Saves best model to `runs/best.pt`.
- `reinforce/visualize_agent.py` — simple visualization helper; observation/action sizes are sometimes hard-coded there for simplicity.
- Network code (e.g. `MLP`) is defined near the top of each `train_agent.py` in each folder and reused by the visualizers via import.

Developer workflows (concrete commands)
- Install deps: `pip install -r requirements.txt` (PowerShell)
- Quick sanity train (example from `dqn/README.md`):
  `python .\dqn\train_agent.py --updates 1 --episodes-per-update 4 --batch-envs 2 --max-episode-steps 10`
- Regular train example:
  `python .\dqn\train_agent.py --updates 300 --episodes-per-update 10 --batch-envs 4 --max-episode-steps 500 --device auto`
- Render/visualize after saving a checkpoint:
  `python .\dqn\visualize_agent.py` or `python .\reinforce\visualize_agent.py`

Project-specific conventions
- Single-file reference trainers: small, readable scripts rather than large package structure.
- Checkpoint format: the trainer typically saves a dict with keys like `model` (state_dict) and `cfg` (training config). Visualizers handle either a dict containing `model` or a plain state_dict.
- Device selection: helper uses `device = "auto"` to choose CUDA if available (`torch.cuda.is_available()`), else CPU.
- Vectorized envs: many dqn defaults support `--batch-envs` > 1; visualization expects a single env (`batch-envs=1`) when rendering.

Integration & external dependencies
- Primary libs: `torch`, `gymnasium` (with automatic fallback to legacy `gym` in some code paths), `matplotlib`, `numpy`.
- Checkpoint loading: newer PyTorch may support `torch.load(..., weights_only=True)` — visualizers have a fallback to plain `torch.load` for older versions.

AI agent guidance (what to do first)
- Read `dqn/README.md` and `reinforce/README.md` to understand intended CLI flags and examples.
- Inspect `train_agent.py` in the folder you intend to modify to find model, optimizer setup, and the save/load format.
- When changing saved checkpoint format, update both trainer and visualizer loader (they assume `checkpoint['model']` or raw state_dict).

Do not assume
- There are no tests or CI definitions in the repo; do not add change-sensitive automation without asking.
- The repo is small and intentionally minimal — avoid adding large dependencies or complex refactors without discussion.

If anything above is unclear or you want this expanded into PR-ready developer docs or task checklists, say which area to expand.
