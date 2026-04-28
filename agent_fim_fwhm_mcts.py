import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import numpy as np
import scipy.io as sio
from utils import compute_reward, evaluate_fim_fwhm, save_history
from typing import Callable, Tuple, Dict, Optional, List
from network import ActorCriticNet


class MCTSNode:
    """A node in the MCTS search tree."""

    __slots__ = [
        "state", "step", "parent", "action", "children",
        "visit_count", "value_sum", "prior",
        "is_expanded", "child_priors",
    ]

    def __init__(self, state: np.ndarray, step: int, parent=None, action=None):
        self.state = state          # (54,) float32, 1 = removed
        self.step = step
        self.parent = parent
        self.action = action        # action from parent that led here

        self.children: Dict[int, "MCTSNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 0.0

        self.is_expanded = False
        self.child_priors: Optional[np.ndarray] = None  # (54,)

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def legal_actions(self):
        return np.where(self.state == 0)[0]

    def is_terminal(self):
        return self.step >= 22


class MCTS:
    """Monte-Carlo Tree Search with neural-network guidance (AlphaZero style)."""

    def __init__(self, net: ActorCriticNet, device: torch.device,
                 num_simulations: int = 100, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_frac: float = 0.25):
        self.net = net
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac

    # ---- neural-network evaluation ----------------------------------------

    @torch.no_grad()
    def _evaluate(self, state: np.ndarray, step: int) -> Tuple[np.ndarray, float]:
        """Return (prior_probs [54], value) for a single state."""
        self.net.eval()
        action_map = torch.tensor(
            state.reshape(3, 3, 6), dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        step_t = torch.tensor([step], dtype=torch.long, device=self.device)

        logits, value = self.net(action_map, step_t)

        logits_np = logits.cpu().numpy()[0]
        mask = (state == 0)
        logits_np[~mask] = -1e9
        logits_np -= logits_np.max()
        probs = np.exp(logits_np)
        probs /= probs.sum()
        return probs, float(value.item())

    # ---- tree operations ---------------------------------------------------

    def _expand(self, node: MCTSNode) -> float:
        """Expand a leaf node: set its child priors and return the NN value."""
        if node.is_terminal():
            return 0.0
        probs, value = self._evaluate(node.state, node.step)
        node.child_priors = probs
        node.is_expanded = True
        return value

    def _select_child(self, node: MCTSNode) -> Tuple[int, "MCTSNode"]:
        """Pick child with highest PUCT score."""
        best_score = -float("inf")
        best_action = -1
        best_child = None

        sqrt_N = math.sqrt(node.visit_count)

        for a in node.legal_actions():
            child = node.children.get(a)
            if child is not None:
                q = child.q_value
                u = self.c_puct * node.child_priors[a] * sqrt_N / (1 + child.visit_count)
            else:
                q = 0.0
                u = self.c_puct * node.child_priors[a] * sqrt_N

            score = q + u
            if score > best_score:
                best_score = score
                best_action = a
                best_child = child

        # create child lazily
        if best_child is None:
            new_state = node.state.copy()
            new_state[best_action] = 1
            best_child = MCTSNode(new_state, node.step + 1,
                                  parent=node, action=best_action)
            best_child.prior = node.child_priors[best_action]
            node.children[best_action] = best_child

        return best_action, best_child

    @staticmethod
    def _backpropagate(node: MCTSNode, value: float):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    # ---- public API --------------------------------------------------------

    def search(self, root_state: np.ndarray, root_step: int,
               add_noise: bool = True) -> np.ndarray:
        """
        Run MCTS from *root_state* and return the visit-count vector (54,).
        """
        root = MCTSNode(root_state.copy(), root_step)
        self._expand(root)

        # Dirichlet noise at root
        if add_noise and root.child_priors is not None:
            legal = root.legal_actions()
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(legal)
            )
            for i, a in enumerate(legal):
                root.child_priors[a] = (
                    (1 - self.dirichlet_frac) * root.child_priors[a]
                    + self.dirichlet_frac * noise[i]
                )

        for _ in range(self.num_simulations):
            node = root
            # selection
            while node.is_expanded and not node.is_terminal():
                _, node = self._select_child(node)
            # expansion + evaluation
            value = self._expand(node)
            # back-propagation
            self._backpropagate(node, value)

        # collect visit counts
        visits = np.zeros(54, dtype=np.float32)
        for a, child in root.children.items():
            visits[a] = child.visit_count
        return visits


class AlphaZeroAgent:
    """
    AlphaZero-style actor-critic agent with MCTS for collimator design.

    Same inputs / outputs per training step as the vanilla RLAgent:
      - takes  geom, evaluate_metric_fn, baseline_metric, sigma
      - train() returns a history dict with per-episode metrics and losses
      - saves checkpoints and geometry snapshots identically
    """

    def __init__(
        self,
        geom,
        evaluate_metric_fn: Callable[[np.ndarray, object], np.ndarray],
        baseline_metric: float,
        sigma: float = 20,
        lr: float = 1e-3,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        temp_threshold_ep: int = 30,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.geom = geom
        self.evaluate_metric_fn = evaluate_metric_fn
        self.baseline_metric = baseline_metric
        self.best_metric = baseline_metric
        self.sigma = sigma

        self.device = torch.device(device)
        self.net = ActorCriticNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.temperature = temperature
        self.temp_threshold_ep = temp_threshold_ep

        self.mcts = MCTS(
            net=self.net,
            device=self.device,
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_frac=dirichlet_frac,
        )

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _visits_to_policy(visits: np.ndarray, temperature: float) -> np.ndarray:
        """Convert raw visit counts to a probability distribution."""
        if temperature == 0:
            policy = np.zeros_like(visits)
            policy[np.argmax(visits)] = 1.0
            return policy
        counts = visits ** (1.0 / temperature)
        total = counts.sum()
        if total == 0:
            legal = np.where(visits > 0)[0]
            policy = np.zeros_like(visits)
            policy[legal] = 1.0 / len(legal)
            return policy
        return counts / total

    # ---- self-play ---------------------------------------------------------

    def _self_play_one(self, temperature: float):
        """
        Play one complete game (22 steps) guided by MCTS.

        Returns
        -------
        trajectory : list of (state [54], step, mcts_policy [54])
        sub_block  : (3, 3, 6) int8 array ready for the simulator
        """
        state = np.zeros(54, dtype=np.float32)
        trajectory: List[Tuple[np.ndarray, int, np.ndarray]] = []

        for step in range(22):
            visits = self.mcts.search(state, step, add_noise=True)
            policy = self._visits_to_policy(visits, temperature)

            trajectory.append((state.copy(), step, policy.copy()))

            action = int(np.random.choice(54, p=policy))
            state[action] = 1.0

        sub_block = (1.0 - state).astype(np.int8).reshape(3, 3, 6)
        return trajectory, sub_block

    # ---- training ----------------------------------------------------------

    def train(
        self,
        num_episodes: int,
        batch_size: int = 64,
        print_every: int = 1,
        save_every: int = 20,
    ) -> dict:
        """
        AlphaZero training loop.

        Each episode:
          1. Self-play *batch_size* games using MCTS to collect trajectories.
          2. Batch-evaluate all terminal designs with the external simulator.
          3. Update the network with cross-entropy policy loss + MSE value loss.
        """
        history = {
            "episode": [],
            "avg_metric": [],
            "min_metric": [],
            "avg_reward": [],
            "policy_loss": [],
            "value_loss": [],
        }

        for ep in range(1, num_episodes + 1):
            temp = self.temperature if ep <= self.temp_threshold_ep else 0.1

            # ---------- 1. self-play ----------
            all_trajectories: List[Tuple[np.ndarray, int, np.ndarray, int]] = []
            all_sub_blocks = []

            for game_idx in range(batch_size):
                traj, sub_block = self._self_play_one(temp)
                for (s, st, mp) in traj:
                    all_trajectories.append((s, st, mp, game_idx))
                all_sub_blocks.append(sub_block)

            sub_blocks = np.stack(all_sub_blocks, axis=0)  # (B, 3, 3, 6)

            # ---------- 2. simulator evaluation ----------
            metrics = self.evaluate_metric_fn(sub_blocks, self.geom)
            rewards = compute_reward(metrics, self.baseline_metric, self.sigma)

            avg_metric = float(metrics.mean())
            min_metric = float(metrics.min())
            avg_reward = float(rewards.mean())

            if avg_metric < self.best_metric:
                self.best_metric = avg_metric

            # ---------- 3. network update ----------
            self.net.train()

            n = len(all_trajectories)
            states_t = torch.zeros((n, 3, 3, 6), dtype=torch.float32, device=self.device)
            steps_t = torch.zeros((n,), dtype=torch.long, device=self.device)
            target_pi = torch.zeros((n, 54), dtype=torch.float32, device=self.device)
            target_v = torch.zeros((n,), dtype=torch.float32, device=self.device)

            for i, (s, st, mp, gi) in enumerate(all_trajectories):
                states_t[i] = torch.tensor(s.reshape(3, 3, 6), dtype=torch.float32)
                steps_t[i] = st
                target_pi[i] = torch.tensor(mp, dtype=torch.float32)
                target_v[i] = torch.tensor(rewards[gi], dtype=torch.float32)

            logits, values = self.net(states_t, steps_t)

            # mask illegal actions
            mask = (states_t.view(n, -1) == 0)
            neg_inf = torch.finfo(logits.dtype).min
            masked_logits = torch.where(mask, logits, torch.full_like(logits, neg_inf))
            log_probs = F.log_softmax(masked_logits, dim=-1)

            # policy loss: KL(mcts_policy || network) via cross-entropy
            policy_loss = -(target_pi * log_probs).sum(dim=-1).mean()

            # value loss: MSE
            value_loss = F.mse_loss(values, target_v)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ---------- 4. bookkeeping ----------
            history["episode"].append(ep)
            history["avg_metric"].append(avg_metric)
            history["min_metric"].append(min_metric)
            history["avg_reward"].append(avg_reward)
            history["policy_loss"].append(policy_loss.item())
            history["value_loss"].append(value_loss.item())

            if ep % print_every == 0:
                print(
                    f"[Ep {ep:4d}] "
                    f"avg_metric={avg_metric:.4f}  min_metric={min_metric:.4f}  "
                    f"avg_reward={avg_reward:.3f}  "
                    f"policy_loss={policy_loss.item():.4f}  value_loss={value_loss.item():.4f}"
                )
                cmd = f"cp fim_fwhm/fim_fwhm_agent.bin logs/fim_fwhm/ep{ep:03d}.bin"
                subprocess.run(cmd, shell=True, check=True)

            if ep % save_every == 0:
                cmd = f"cp -r crystal_geometry_agent_design crystal_geometry_history/crystal_geometry_ep{ep:03d}"
                subprocess.run(cmd, shell=True, check=True)

                checkpoint_path = f"checkpoints/agent_ep{ep:03d}.pt"
                torch.save(
                    {
                        "episode": ep,
                        "model_state_dict": self.net.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_metric": self.best_metric,
                        "baseline_metric": self.baseline_metric,
                    },
                    checkpoint_path,
                )
                print(f"[Checkpoint] Saved model to {checkpoint_path}")

        return history
