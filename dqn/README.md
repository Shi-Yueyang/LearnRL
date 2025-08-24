# Minimal DQN Training Script

This repository contains a self-contained minimal Deep Q-Network (DQN) trainer for a built-in Gym / Gymnasium environment (default: `CartPole-v1`).

## Features

- Vectorized environment support (`--batch-envs > 1`)
- Epsilon-greedy exploration with linear decay
- Target network updates
- Replay buffer with uniform sampling
- Automatic fallback between `gymnasium` and legacy `gym`
- Automatic device selection with `--device auto`

## Quick Start

Install dependencies (adjust if you already have torch installed):

```powershell
pip install -r requirements.txt
```

Run a very short sanity training (1 update, few episodes):

```powershell
python .\train_agent.py --updates 1 --episodes-per-update 4 --batch-envs 2 --max-episode-steps 10
```

Regular training (example):

```powershell
python .\train_agent.py --updates 300 --episodes-per-update 10 --batch-envs 4 --max-episode-steps 500 --device auto
```

Render (single env recommended):

```powershell
python .\train_agent.py --render --batch-envs 1 --updates 50
```

Saved artifacts appear in the `runs/` directory (`best.pt` and periodic snapshots).

## Arguments (summary)

| Argument                 | Description                                   |
| ------------------------ | --------------------------------------------- |
| --env-id                 | Environment id (default CartPole-v1)          |
| --updates                | Number of outer update cycles                 |
| --episodes-per-update    | Episodes collected before each training phase |
| --batch-envs             | Number of parallel envs (vectorized)          |
| --max-episode-steps      | Time limit wrapper step cap                   |
| --gamma                  | Discount factor                               |
| --lr                     | Learning rate                                 |
| --buffer-size            | Replay buffer capacity                        |
| --batch-size             | SGD mini-batch size                           |
| --start-eps / --end-eps  | Epsilon schedule endpoints                    |
| --eps-decay-updates      | Linear decay horizon (updates)                |
| --target-update          | Target network sync frequency (updates)       |
| --train-iters-per-update | Gradient steps each update cycle              |
| --min-buffer             | Warmup before training begins                 |
| --render                 | Enable rendering (slows training)             |
| --save-dir               | Directory to save checkpoints                 |
| --save-every             | Save model every N updates                    |

## Notes

- This is a deliberately compact reference implementation; it omits advanced features (prioritized replay, double DQN, etc.).
- For faster learning you can increase `train-iters-per-update`, `batch-envs`, and `updates` gradually.
- On GPU, CartPole learns quickly; convergence ~200 average return.

## Loading a Saved Model

```python
import torch, gymnasium as gym
from train_agent import MLP

checkpoint = torch.load('runs/best.pt', map_location='cpu')
obs_dim = 4
n_actions = 2
model = MLP(obs_dim, n_actions)
model.load_state_dict(checkpoint['model'])
model.eval()

env = gym.make('CartPole-v1')
obs, _ = env.reset()
done = False
while not done:
    with torch.no_grad():
        action = model(torch.tensor(obs).float().unsqueeze(0)).argmax(1).item()
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()
```

Enjoy experimenting!

## Mathematical Foundations (DQN & CartPole)

### Notation & Symbols

| Symbol              | Meaning                           |
| ------------------- | --------------------------------- |
| $s_t, a_t, r_{t+1}$ | State, action, reward at time $t$ |
| $\gamma$            | Discount factor $(0<\gamma\le 1)$ |
| $Q_\theta$          | Parametric action-value function  |
| $Q_{\theta^-}$      | Target network (delayed copy)     |
| $d_i$               | Done flag (1 if terminal else 0)  |
| $B$                 | Batch size                        |

### 1. Return

$$\tag{1} G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+1+k}$$

### 2. Optimal Action-Value

$$\tag{2} Q^*(s,a) = \max_{\pi} \; \mathbb{E}[ G_t \mid s_t=s, a_t=a, \pi ]$$

### 3. Bellman Optimality

$$\tag{3} Q^*(s,a) = \mathbb{E}_{s'\sim P(\cdot|s,a)}\Big[ r(s,a) + \gamma \max_{a'} Q^*(s',a') \Big]$$

### 4. DQN Target (per sample)

Given replay tuple $(s_i,a_i,r_i,s'_i,d_i)$:
$$\tag{4} y_i = r_i + \gamma (1-d_i) \max_{a'} Q_{\theta^-}(s'_i,a')$$

### 5. TD Error

$$\tag{5} e_i = y_i - Q_\theta(s_i,a_i)$$

### 6. Huber Loss (\(\delta=1\))

$$\tag{6}
L(\theta)=\frac{1}{B}\sum_{i=1}^B \ell(e_i),\qquad
\ell(e)=\begin{cases}
	frac{1}{2} e^2 & |e| \le 1 \\
|e| - \tfrac{1}{2} & |e| > 1
\end{cases}
$$

### 7. Gradient (Conceptual)

$$\tag{7} \nabla_\theta L = -\frac{1}{B}\sum_{i}(\partial \ell/\partial e_i)\;\nabla_\theta Q_\theta(s_i,a_i)$$

### 8. Epsilon-Greedy Policy

Action selection distribution:

$$
\tag{8}
\pi(a\mid s)=
\begin{cases}
1-\epsilon + \dfrac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q_\theta(s,a') \\
\dfrac{\epsilon}{|\mathcal{A}|} & \text{otherwise}
\end{cases}
$$

Linear decay over update index $u$:

$$
\tag{9}
\epsilon(u)=\epsilon_{\text{start}} + (\epsilon_{\text{end}}-\epsilon_{\text{start}}) \; \min\!\left(\frac{u}{U_{\text{decay}}},\,1\right)
$$

### 9. Target Network Update

Periodic hard copy:
$$\tag{10} \theta^- \leftarrow \theta$$

### 10. CartPole State & Termination

State: $ (x, \dot{x}, \theta, \dot{\theta}) $. Episode ends if

$$
\tag{11}
|x|>2.4 \;\lor\; |\theta|>12^{\circ} \;\lor\; \text{step limit reached}
$$

### 11. Simplified Dynamics

With cart mass $m_c$, pole mass $m_p$, half-length $l$ (effective length), gravity $g$, and applied force $F$:

$$
\tag{12}
\ddot{\theta} = \frac{ g \sin\theta + \cos\theta\; \dfrac{-F - m_p l \dot{\theta}^2 \sin\theta}{m_c + m_p} }{ l \left( \tfrac{4}{3} - \tfrac{ m_p \cos^2\theta }{ m_c + m_p } \right) }
$$

$$
\tag{13}
\ddot{x} = \frac{ F + m_p l \big( \dot{\theta}^2 \sin\theta - \ddot{\theta} \cos\theta \big) }{ m_c + m_p }
$$

### 12. Reward Structure

CartPole gives reward 1 each surviving step ⇒ maximizing expected return ≡ maximizing expected episode length.

### 13. Stability Heuristics

- Replay buffer approximates i.i.d. sampling → reduced correlation.
- Target network stabilizes bootstrap target.
- Huber loss + gradient clipping reduce sensitivity to outliers.

### 14. Practical Convergence Note

DQN lacks universal convergence guarantees with nonlinear approximators; empirical stabilizers (Sections 8–13) suffice for small control tasks like CartPole.

## Symbol & Expression Explanations

1. $G_t$: Total discounted future reward from time $t$ onward; balances immediate vs long-term gains.
2. $Q^*(s,a)$: Best achievable expected return starting in $s$ taking action $a$, then acting optimally.
3. Bellman Optimality (Eq. 3): Recursive decomposition turning long-horizon optimization into local consistency constraints.
4. Target $y_i$ (Eq. 4): One-step bootstrap using target network to reduce moving-target instability.
5. TD Error $e_i$ (Eq. 5): Instant learning signal; positive if target under-estimates value, negative if over-estimates.
6. Huber Loss (Eq. 6): Smooth quadratic near zero (precision) and linear for large errors (robustness).
7. Gradient (Eq. 7): Semi-gradient ignoring target network dependence; stabilizes updates.
8. Epsilon-Greedy (Eq. 8/9): Ensures sufficient exploration early; decays to exploit learned value function.
9. Target Copy (Eq. 10): Freezes target distribution temporarily, reducing oscillations.
10. Termination (Eq. 11): Defines failure; maximizing steps before triggering thresholds yields higher return.
11. Dynamics (Eq. 12/13): Nonlinear coupled equations; DQN implicitly learns control without explicit model inversion.
12. Reward Structure: Dense, sparse-free shaping simplifies credit assignment.
13. Stability Heuristics: Practical measures combating correlation, non-stationarity, and divergence.
14. Convergence Caveat: Highlights theoretical limitations while noting empirical sufficiency for this domain.
