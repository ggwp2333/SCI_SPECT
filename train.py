import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import numpy as np
import scipy.io as sio
from crystal_geometry_8x8 import CrystalGeometryTemplate
from utils import compute_reward, masked_softmax, evaluate_fim_fwhm, save_history
from agent_fim_fwhm_8x8 import RLAgent
from typing import Callable
from network import ActorCriticNet
from typing import Callable, Tuple

if __name__ == "__main__":

    geom = CrystalGeometryTemplate()

    baseline_metric = 45
    sigma = 15

    agent = RLAgent(
        geom=geom,
        evaluate_metric_fn=evaluate_fim_fwhm, 
        baseline_metric=baseline_metric,
        sigma=sigma,
        lr=1e-3,
        value_coef=0.5,
        entropy_coef=1e-3
    )

    history = agent.train(num_episodes=500, batch_size=300, print_every=1, save_every=50)
    save_history(history)
