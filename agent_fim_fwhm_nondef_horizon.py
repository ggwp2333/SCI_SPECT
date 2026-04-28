import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import numpy as np
import scipy.io as sio
# from crystal_geometry import CrystalGeometryTemplate
from utils import compute_reward, evaluate_fim_fwhm, save_history
from typing import Callable
from network import ActorCriticNet
from typing import Callable, Tuple

class DummyRLAgent:

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def sample_sub_block(self) -> np.ndarray:

        sub_block = np.ones((3, 3, 6), dtype=np.int8)
        flat_indices = np.arange(54)

        # choose 11 non-crystal positions
        non_crystal_idx = self.rng.choice(flat_indices, size=22, replace=False)
        sub_block_flat = sub_block.reshape(-1)
        sub_block_flat[non_crystal_idx] = 0

        return sub_block.reshape((3, 3, 6))

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
        uniform_horizon_warmup_episodes: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        ):

        self.geom = geom
        self.evaluate_metric_fn = evaluate_metric_fn
        self.baseline_metric = baseline_metric
        self.best_metric = baseline_metric
        self.sigma = sigma

        self.device = torch.device(device)
        self.net = ActorCriticNet().to(self.device)
        self.stop_action = 54
        self.stop_enabled_after = 15
        self.max_steps = 27
        self.net.max_steps = float(self.max_steps)
        self.uniform_horizon_warmup_episodes = uniform_horizon_warmup_episodes

        # Keep the original network untouched and add one small stop-action head.
        self.stop_head = nn.Linear(self.net.policy_head.in_features, 1).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.net.parameters()) + list(self.stop_head.parameters()),
            lr=lr,
        )

        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def run_episode(self, batch_size: int, episode_idx: int | None = None) -> dict:
        """
        Runs a single episode:
          - start with empty action_map
          - for t in 0..10: select 11 distinct voxels as non-crystal
          - build sub_block, call simulator, compute reward

        Returns dict with:
            'log_probs', 'values', 'entropies', 'metric', 'reward'
        """
        self.net.eval()
        self.stop_head.eval()

        B = batch_size
        T = self.max_steps

        # state init
        action_map = torch.zeros((B, 3, 3, 6), device=self.device) 
        step_t = torch.zeros((B,), device=self.device, dtype=torch.long)
        done = torch.zeros((B,), device=self.device, dtype=torch.bool)
        use_uniform_horizon_warmup = (
            episode_idx is not None and episode_idx <= self.uniform_horizon_warmup_episodes
        )
        if use_uniform_horizon_warmup:
            target_horizon = torch.randint(
                low=self.stop_enabled_after,
                high=self.max_steps + 1,
                size=(B,),
                device=self.device,
                dtype=torch.long,
            )
        else:
            target_horizon = None

        log_probs_list = []
        values_list = []
        entropies_list = []
        active_masks = []

        for t in range(T):
            if done.all():
                break

            # Forward pass (same trunk/value head, with an extra stop logit)
            x = action_map.view(B, -1).float()
            step_feat = (step_t.float() / self.net.max_steps).unsqueeze(-1)
            x = torch.cat([x, step_feat], dim=-1)
            h = self.net.trunk(x)

            voxel_logits = self.net.policy_head(h)
            stop_logits = self.stop_head(h)
            logits = torch.cat([voxel_logits, stop_logits], dim=-1)
            value = self.net.value_head(h).squeeze(-1)

            voxel_legal = (action_map.view(B, -1) == 0)
            stop_legal = (step_t >= self.stop_enabled_after).unsqueeze(-1)
            legal = torch.cat([voxel_legal, stop_legal], dim=-1)

            if use_uniform_horizon_warmup:
                force_stop = (~done) & (step_t >= target_horizon)
                if force_stop.any():
                    legal[force_stop, :] = False
                    legal[force_stop, self.stop_action] = True

            # Finished samples are forced to STOP for numerically stable sampling.
            legal[done, :] = False
            legal[done, self.stop_action] = True

            neg_inf = torch.finfo(logits.dtype).min
            masked_logits = torch.where(legal, logits, torch.full_like(logits, neg_inf))
            probs = F.softmax(masked_logits, dim=-1)
            entropy = -(probs * (masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True))).sum(dim=-1)

            # Sample action
            m = torch.distributions.Categorical(probs=probs)
            action = m.sample()     
            action = torch.where(done, torch.full_like(action, self.stop_action), action)
            log_prob = m.log_prob(action)  

            log_probs_list.append(log_prob)   
            values_list.append(value)         
            entropies_list.append(entropy)
            active_masks.append((~done).float())

            # Update state: mark chosen voxel as 1 (non-crystal)
            batch_idx = torch.arange(B, device=self.device)
            choose_voxel = (~done) & (action != self.stop_action)
            if choose_voxel.any():
                selected = action[choose_voxel]
                z = selected // 18
                rem = selected % 18
                y = rem // 6
                x = rem % 6
                action_map[batch_idx[choose_voxel], z, y, x] = 1.0

            step_t += (~done).long()
            done = done | (action == self.stop_action) | (step_t >= self.max_steps)

        # Stack over time
        log_probs = torch.stack(log_probs_list, dim=0)    
        values = torch.stack(values_list, dim=0)         
        entropies = torch.stack(entropies_list, dim=0)
        active_mask = torch.stack(active_masks, dim=0)

        # Build sub_block: 1 = crystal, 0 = non-crystal (complement)
        sub_block = (1.0 - action_map).detach().cpu().numpy().astype(np.int8).reshape(B, 3, 3, 6)

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
            "active_mask": active_mask,
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
            "critic_loss": [],
            "uniform_horizon_warmup": []
        }

        for ep in range(1, num_episodes + 1):
            roll = self.run_episode(batch_size=batch_size, episode_idx=ep)
            log_probs = roll["log_probs"]   
            values = roll["values"]         
            entropies = roll["entropies"]   
            active_mask = roll["active_mask"]
            metric = roll["metric"]
            reward = roll["reward"]

            T, B = values.shape

            # Same terminal reward for all steps (fixed-length episode)
            R = torch.tensor(reward).unsqueeze(0).expand(T, B).to(self.device)
 
            advantages = R - values
            valid_steps = active_mask.sum().clamp_min(1.0)

            actor_loss = -((log_probs * advantages.detach()) * active_mask).sum() / valid_steps
            critic_loss = (((advantages ** 2) * active_mask).sum() / valid_steps)
            entropy_loss = -((entropies * active_mask).sum() / valid_steps)

            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

            self.net.train()
            self.stop_head.train()
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
            history["uniform_horizon_warmup"].append(
                1.0 if ep <= self.uniform_horizon_warmup_episodes else 0.0
            )

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
                                "stop_head_state_dict": self.stop_head.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "best_metric": self.best_metric,
                                "baseline_metric": self.baseline_metric,
                            },
                    checkpoint_path,
                )
                print(f"[Checkpoint] Saved model to {checkpoint_path}")

        return history
