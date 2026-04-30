"""
GA training entry point — mirrors train.py for RLAgent.

Run with:
    python train_ga.py
"""
from crystal_geometry import CrystalGeometryTemplate
from utils import evaluate_fim_fwhm, save_history
from agent_fim_fwhm_GA import GAAgent


if __name__ == "__main__":
    geom = CrystalGeometryTemplate()

    baseline_metric = 45.0   # same starting point used for RLAgent

    agent = GAAgent(
        geom=geom,
        evaluate_metric_fn=evaluate_fim_fwhm,
        baseline_metric=baseline_metric,
        # ----- GA hyper-parameters -----
        pop_size=50,             # one batched simulator call evaluates all 50
        n_crystal=32,            # 32 GAGG + 22 acrylic in the 3x3x6 sub-block
        crossover_rate=0.85,
        crossover_type="uniform",
        mutation_rate=0.20,
        n_swaps=3,
        n_elite=2,
        tournament_k=3,
        adaptive_mut=True,
        stall_gens=8,
        mut_boost=0.40,
        mut_boost_gens=4,
        seed=42,
    )

    history = agent.train(
        num_generations=20,
        print_every=1,
        save_every=5,
    )
    save_history(history, filename="logs/ga_training_history.mat")
