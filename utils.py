import numpy as np
import scipy.io as sio
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple
import subprocess

def compute_reward(metrics: torch.Tensor, baseline_metric: float = 50, sigma: float = 20) -> float:
    advantage = (baseline_metric - metrics)/sigma
    rewards = np.tanh(advantage)
    return rewards

def masked_softmax(logits: torch.Tensor, action_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B = action_map.shape[0]
    mask = (action_map.view(B, -1) == 0)  # True for legal actions

    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask, logits, torch.full_like(logits, neg_inf))
    probs = F.softmax(masked_logits, dim=-1)

    entropy = -(probs * (masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True))).sum(dim=-1)
    return probs, entropy

def evaluate_fim_fwhm(sub_blocks: np.ndarray, geom) -> np.ndarray:

    n_designs = sub_blocks.shape[0]
    out_dir = "crystal_geometry_agent_design"
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(1, n_designs + 1):
        sub_block = sub_blocks[idx - 1]  
        full_map, cube_pos = geom.apply_module(sub_block)
        geom.write_geometry(full_map, cube_pos, out_dir=out_dir, idx=idx)

    os.makedirs("logs", exist_ok=True)
    subprocess.run("stdbuf -oL -eL ./executables/fast_fim_tc99m_120mmFOV_batch crystal_geometry_agent_design fim_fwhm/fim_fwhm_agent.bin> logs/simulator_output.log 2>&1", shell=True, check=True)

    metrics = np.fromfile("fim_fwhm/fim_fwhm_agent.bin", dtype=np.float64)
    metrics = metrics[:n_designs].astype(np.float32)

    return metrics

def evaluate_fourier_crosstalk(sub_blocks: np.ndarray, geom) -> np.ndarray:

    n_designs = sub_blocks.shape[0]
    out_dir = "crystal_geometry_agent_design"
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(1, n_designs + 1):
        sub_block = sub_blocks[idx - 1]  
        full_map, cube_pos = geom.apply_module(sub_block)
        geom.write_geometry(full_map, cube_pos, out_dir=out_dir, idx=idx)

    os.makedirs("logs", exist_ok=True)
    subprocess.run("stdbuf -oL -eL ./executables/fourier_crosstalk_tc99m_100mmFOV_batch crystal_geometry_agent_design fourier_crosstalk/fourier_crosstalk_agent.bin> logs/simulator_output.log 2>&1", shell=True, check=True)

    metrics = np.fromfile("fourier_crosstalk/fourier_crosstalk_agent.bin", dtype=np.float64)
    metrics = metrics[:n_designs].astype(np.float32)

    return metrics

def evaluate_mc_fwhm(sub_blocks: np.ndarray, geom) -> np.ndarray:

    n_designs = sub_blocks.shape[0]
    out_dir = "crystal_geometry_agent_design"
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(1, n_designs + 1):
        sub_block = sub_blocks[idx - 1]  
        full_map, cube_pos = geom.apply_module(sub_block)
        geom.write_geometry(full_map, cube_pos, out_dir=out_dir, idx=idx)

    os.makedirs("logs", exist_ok=True)
    subprocess.run("stdbuf -oL -eL ./executables/fast_mc_tc99m_100mmFOV_batch crystal_geometry_agent_design mc_fwhm/mc_fwhm_agent.bin> logs/simulator_output.log 2>&1", shell=True, check=True)

    metrics = np.fromfile("mc_fwhm/mc_fwhm_agent.bin", dtype=np.float64)
    metrics = metrics[:n_designs].astype(np.float32)

    return metrics

def save_history(history: dict, filename: str = "logs/training_history.mat"):

    history_mat = {}

    for key, values in history.items():
        arr = np.array(values)
        if key == "episode":
            arr = arr.astype(np.int32)
        else:
            arr = arr.astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        history_mat[key] = arr

    sio.savemat(filename, {"history": history_mat})
    print(f"Saved training history to {filename}")