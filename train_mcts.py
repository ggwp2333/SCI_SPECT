import os
from crystal_geometry import CrystalGeometryTemplate
from utils import evaluate_fim_fwhm, save_history
from agent_fim_fwhm_mcts import AlphaZeroAgent

if __name__ == "__main__":

    geom = CrystalGeometryTemplate()

    baseline_metric = 45
    sigma = 10

    agent = AlphaZeroAgent(
        geom=geom,
        evaluate_metric_fn=evaluate_fim_fwhm,
        baseline_metric=baseline_metric,
        sigma=sigma,
        lr=1e-3,
        num_simulations=100,    # MCTS simulations per move
        c_puct=0.8,             # exploration constant
        temperature=1.0,        # action selection temperature
        temp_threshold_ep=30,   # switch to greedy (temp=0.1) after this ep
        dirichlet_alpha=0.1,    # root noise for exploration
        dirichlet_frac=0.25,
    )

    # NOTE: MCTS is ~num_simulations*22 NN forward passes per game,
    # so batch_size should be much smaller than vanilla A2C.
    # Compensate with higher quality per sample.
    history = agent.train(num_episodes=600, batch_size=128, print_every=1, save_every=40)
    save_history(history)
