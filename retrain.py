import os
import argparse
import torch
import numpy as np
import subprocess

from crystal_geometry_compact import CrystalGeometryTemplate
from utils import evaluate_fim_fwhm, save_history
from agent_fim_fwhm import RLAgent

def load_checkpoint(agent: RLAgent, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.net.load_state_dict(ckpt["model_state_dict"])
    agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    agent.best_metric = float(ckpt.get("best_metric", agent.best_metric))
    agent.baseline_metric = float(ckpt.get("baseline_metric", agent.baseline_metric))
    start_ep = int(ckpt.get("episode", 0))
    return start_ep

def continue_train(
    agent: RLAgent,
    start_episode: int,
    num_more_episodes: int,
    batch_size: int,
    print_every: int,
    save_every: int,
    checkpoint_dir: str = "checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    history = {
        "episode": [],
        "avg_metric": [],
        "min_metric": [],
        "avg_reward": [],
        "actor_loss": [],
        "critic_loss": []
    }

    for ep in range(start_episode + 1, start_episode + num_more_episodes + 1):
        roll = agent.run_episode(batch_size=batch_size)
        log_probs = roll["log_probs"]
        values = roll["values"]
        entropies = roll["entropies"]
        metric = roll["metric"]
        reward = roll["reward"]

        T, B = values.shape

        R = torch.tensor(reward).unsqueeze(0).expand(T, B).to(agent.device)
        advantages = R - values

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = (advantages ** 2).mean()
        entropy_loss = -entropies.mean()

        loss = actor_loss + agent.value_coef * critic_loss + agent.entropy_coef * entropy_loss

        agent.net.train()
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

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
            subprocess.run(cmd, shell=True, check=False)

        if ep % save_every == 0:
            cmd = f"cp -r crystal_geometry_agent_design crystal_geometry_history/crystal_geometry_ep{ep:03d}"
            subprocess.run(cmd, shell=True, check=False)

            checkpoint_path = os.path.join(checkpoint_dir, f"agent_ep{ep:03d}.pt")
            torch.save(
                {
                    "episode": ep,
                    "model_state_dict": agent.net.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "best_metric": agent.best_metric,
                    "baseline_metric": agent.baseline_metric,
                },
                checkpoint_path,
            )
            print(f"[Checkpoint] Saved model to {checkpoint_path}")

    return history


def main():

    geom = CrystalGeometryTemplate()

    agent = RLAgent(
        geom=geom,
        evaluate_metric_fn=evaluate_fim_fwhm,
        baseline_metric=50,
        sigma=10,
        lr=1e-3,
        value_coef=0.5,
        entropy_coef=1e-3
    )

    start_ep = load_checkpoint(agent, "checkpoints/agent_ep300.pt", "cuda:0")

    print(f"[Resume] start_episode={start_ep}, best_metric={agent.best_metric}, baseline_metric={agent.baseline_metric}")

    history = continue_train(
        agent=agent,
        start_episode=300,
        num_more_episodes=500,
        batch_size=120,
        print_every=1,
        save_every=20
    )

    if len(history["episode"]) > 0:
        print(f"[Done] Trained through episode {history['episode'][-1]}.")


if __name__ == "__main__":
    main()
