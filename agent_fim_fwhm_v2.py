import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import numpy as np
import scipy.io as sio
from crystal_geometry_v2 import CrystalGeometryTemplate
from utils import compute_reward, masked_softmax, evaluate_fim_fwhm, save_history
from typing import Callable
from network import ActorCriticNet
from typing import Callable, Tuple

class DummyRLAgent:

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def sample_sub_block(self) -> np.ndarray:

        sub_block = np.ones((9, 3, 6), dtype=np.int8)
        flat_indices = np.arange(162)

        # choose 11 non-crystal positions
        non_crystal_idx = self.rng.choice(flat_indices, size=22, replace=False)
        sub_block_flat = sub_block.reshape(-1)
        sub_block_flat[non_crystal_idx] = 0

        return sub_block.reshape((9, 3, 6))

class RLAgent:
    def __init__(
        self,
        geom,  
        evaluate_metric_fn: Callable[[np.ndarray], float],
        baseline_metric: float,
        sigma: float = 20,
        lr: float = 1e-3,
        value_coef: float = 0.5,
        entropy_coef: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        ):

        self.geom = geom
        self.evaluate_metric_fn = evaluate_metric_fn
        self.baseline_metric = baseline_metric
        self.best_metric = baseline_metric
        self.sigma = sigma

        self.device = torch.device(device)
        self.net = ActorCriticNet(n_actions=162,max_steps=66).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def run_episode(self, batch_size: int) -> dict:
        """
        Runs a single episode:
          - start with empty action_map
          - for t in 0..10: select 11 distinct voxels as non-crystal
          - build sub_block, call simulator, compute reward

        Returns dict with:
            'log_probs', 'values', 'entropies', 'metric', 'reward'
        """
        self.net.eval()

        B = batch_size
        T = 66

        # state init
        action_map = torch.zeros((B, 9, 3, 6), device=self.device) 
        step_t = torch.zeros((B,), device=self.device, dtype=torch.long)

        log_probs_list = []
        values_list = []
        entropies_list = []

        for t in range(T):
            # Forward pass
            logits, value = self.net(action_map, step_t)
            probs, entropy = masked_softmax(logits, action_map) 

            # Sample action
            m = torch.distributions.Categorical(probs=probs)
            action = m.sample()     
            log_prob = m.log_prob(action)  

            log_probs_list.append(log_prob)   
            values_list.append(value)         
            entropies_list.append(entropy)

            # Update state: mark chosen voxel as 1 (non-crystal)
            z = action // 18
            rem = action % 18
            y = rem // 6
            x = rem % 6

            batch_idx = torch.arange(B, device=self.device)
            action_map[batch_idx, z, y, x] = 1.0

            step_t += 1

        # Stack over time
        log_probs = torch.stack(log_probs_list, dim=0)    
        values = torch.stack(values_list, dim=0)         
        entropies = torch.stack(entropies_list, dim=0)

        # Build sub_block: 1 = crystal, 0 = non-crystal (complement)
        sub_block = (1.0 - action_map).detach().cpu().numpy().astype(np.int8).reshape(B, 9, 3, 6)

        # Evaluate metric via simulator 
        metric = self.evaluate_metric_fn(sub_block, self.geom)

        # Compute reward
        reward = compute_reward(metric, self.baseline_metric, self.sigma)

        avg_metric = np.mean(metric)
        if avg_metric < self.best_metric:
            self.best_metric = avg_metric
            # self.baseline_metric = avg_metric  

        return {
            "log_probs":log_probs,   
            "values": values,         
            "entropies": entropies,   
            "metric": metric,
            "reward": reward,
        }

    def train(self, num_episodes: int, batch_size: int = 64, print_every: int = 1, save_every: int = 20):
        """
        Simple on-policy advantage actor-critic training.
        """
        history = {
            "episode": [],
            "avg_metric": [],
            "min_metric": [],
            "avg_reward": [],
            "actor_loss": [],
            "critic_loss": []
        }

        for ep in range(1, num_episodes + 1):
            roll = self.run_episode(batch_size=batch_size)
            log_probs = roll["log_probs"]   
            values = roll["values"]         
            entropies = roll["entropies"]   
            metric = roll["metric"]
            reward = roll["reward"]

            T, B = values.shape

            # Same terminal reward for all steps (fixed-length episode)
            R = torch.tensor(reward).unsqueeze(0).expand(T, B).to(self.device)
 
            advantages = R - values

            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = (advantages ** 2).mean()
            entropy_loss = -entropies.mean()

            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

            self.net.train()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_metric = float(metric.mean())
            min_metric = float(metric.min())
            avg_reward = float(reward.mean().item())

            history["episode"].append(ep)
            history["avg_metric"].append(avg_metric)
            history["min_metric"].append(min_metric)
            history["avg_reward"].append(avg_reward)
            history["actor_loss"].append(actor_loss.item())
            history["critic_loss"].append(critic_loss.item())

            if ep % print_every == 0:
                print(
                    f"[Ep {ep:4d}] "
                    f"avg_metric={avg_metric:.4f}  min_metric={min_metric:.4f}  "
                    f"avg_reward={avg_reward:.3f}  "
                    f"actor_loss={actor_loss.item():.4f}  critic_loss={critic_loss.item():.4f}"
                )
                cmd = f"cp fim_fwhm/fim_fwhm_agent.bin logs/fim_fwhm/ep{ep:03d}.bin"
                subprocess.run(cmd, shell=True, check=True)

            if ep % save_every == 0:
                cmd = f"cp -r crystal_geometry_agent_design crystal_geometry_history/crystal_geometry_ep{ep:03d}"
                subprocess.run(cmd, shell=True, check=True)

                checkpoint_path = f"checkpoints/agent_ep{ep:03d}.pt"
                torch.save({
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


