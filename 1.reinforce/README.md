# REINFORCE (Monte Carlo Policy Gradient) minimal implementation

A compact implementation of the vanilla REINFORCE algorithm for Gymnasium's `CartPole-v1`, styled similarly to the DQN example in this repo.

## Features
- Single-file training script `train_agent.py` with a `Config` dataclass (no argparse)
- Supports reward-to-go and (optional) return normalization
- Optional entropy bonus for exploration
- Gradient clipping
- Saves best checkpoint to `runs/best.pt`
- Simple visualization script `visualize_agent.py`

## Train
```
python reinforce/train_agent.py
```
Adjust hyperparameters by editing the `Config` dataclass in the script.

## Visualize
```
python reinforce/visualize_agent.py
```
Set `render_mode` to `"rgb_array"` in `Config` if you want to capture frames programmatically.

## Notes
- For CartPole the observation dimension (4) and action count (2) are hard-coded only inside visualization loading helper for simplicity.
- To adapt to other discrete environments, change `env_id` and (optionally) network size.
