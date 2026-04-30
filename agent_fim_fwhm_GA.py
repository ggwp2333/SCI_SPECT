import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import numpy as np
import scipy.io as sio
# from crystal_geometry import CrystalGeometryTemplate
from utils import compute_reward, masked_softmax, evaluate_fim_fwhm, save_history
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        ):

        self.geom = geom
        self.evaluate_metric_fn = evaluate_metric_fn
        self.baseline_metric = baseline_metric
        self.best_metric = baseline_metric
        self.sigma = sigma

        self.device = torch.device(device)
        self.net = ActorCriticNet().to(self.device)
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
        T = 22

        # state init
        action_map = torch.zeros((B, 3, 3, 6), device=self.device) 
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


# ============================================================================
#  Genetic Algorithm Agent
# ============================================================================

class GAAgent:
    """
    Genetic-algorithm agent for SCI-SPECT crystal-layout optimization.

    Operates on the same 3x3x6 sub_block parameterization as RLAgent:
        - 54 binary genes per individual (1 = crystal/GAGG, 0 = non-crystal)
        - exactly `n_crystal` ones per genome (default 32, i.e. 22 non-crystals,
          matching DummyRLAgent / RLAgent's T=22)

    The whole population is evaluated in one batched call to
    `evaluate_metric_fn`, so per-generation GPU time is dominated by the
    simulator, not Python overhead -- same pattern as RLAgent's run_episode.

    Genetic operators
    -----------------
    * Selection : tournament of size `tournament_k`
    * Crossover : uniform or two-point, followed by a `_repair` step that
                  re-imposes the exact-`n_crystal` count constraint
    * Mutation  : swap-mutation (count-preserving). Each mutation event
                  swaps `n_swaps` (1, 0) pairs, preserving the GAGG count
                  exactly so we never need to re-repair after mutation.
    * Elitism   : top `n_elite` survive each generation unchanged
    * Adaptive  : if the best fitness stalls for `stall_gens` generations,
                  mutation_rate is temporarily boosted to `mut_boost` for
                  `mut_boost_gens` generations, then restored.

    Lower fitness (FWHM) is better.

    Drop-in usage (mirrors train.py for RLAgent):

        geom = CrystalGeometryTemplate()
        agent = GAAgent(
            geom=geom,
            evaluate_metric_fn=evaluate_fim_fwhm,
            baseline_metric=45.0,
            pop_size=50,
        )
        history = agent.train(num_generations=80)
    """

    N_GENES = 54  # 3 * 3 * 6

    def __init__(
        self,
        geom,
        evaluate_metric_fn: Callable[[np.ndarray, object], np.ndarray],
        baseline_metric: float,
        pop_size: int = 20,
        n_crystal: int = 32,
        crossover_rate: float = 0.85,
        crossover_type: str = "uniform",   # "uniform" | "twopoint"
        mutation_rate: float = 0.20,
        n_swaps: int = 3,
        n_elite: int = 2,
        tournament_k: int = 3,
        adaptive_mut: bool = True,
        stall_gens: int = 8,
        mut_boost: float = 0.40,
        mut_boost_gens: int = 4,
        chunk_size: int =15,
        chunk_verbose: bool = True,
        seed: int = 42,
    ):
        assert crossover_type in ("uniform", "twopoint"), \
            f"crossover_type must be 'uniform' or 'twopoint', got {crossover_type!r}"
        assert 0 < n_crystal < self.N_GENES, \
            f"n_crystal must be in (0, {self.N_GENES})"
        assert pop_size >= 2 * n_elite + 2, \
            "pop_size too small for the requested elitism"
        assert chunk_size is None or chunk_size > 0, \
            "chunk_size must be a positive integer or None"


        # Pipeline objects
        self.geom = geom
        self.evaluate_metric_fn = evaluate_metric_fn
        self.baseline_metric = baseline_metric
        self.best_metric = baseline_metric
        self.best_genome = None  # flat int8 of shape (54,)

        # GA hyper-parameters
        self.pop_size = pop_size
        self.n_crystal = n_crystal
        self.crossover_rate = crossover_rate
        self.crossover_type = crossover_type
        self.mutation_rate = mutation_rate
        self.n_swaps = n_swaps
        self.n_elite = n_elite
        self.tournament_k = tournament_k
        self.chunk_size = chunk_size
        self.chunk_verbose = chunk_verbose


        # Adaptive-mutation state
        self.adaptive_mut = adaptive_mut
        self.stall_gens = stall_gens
        self.mut_boost = mut_boost
        self.mut_boost_gens = mut_boost_gens
        self._stall_counter = 0
        self._boost_counter = 0
        self._effective_mut_rate = mutation_rate

        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    #  Genome utilities
    # ------------------------------------------------------------------

    def _random_genome(self) -> np.ndarray:
        """Random flat genome with exactly n_crystal ones."""
        g = np.zeros(self.N_GENES, dtype=np.int8)
        idx = self.rng.choice(self.N_GENES, size=self.n_crystal, replace=False)
        g[idx] = 1
        return g

    def _initial_population(self) -> np.ndarray:
        return np.stack(
            [self._random_genome() for _ in range(self.pop_size)], axis=0
        )

    def _to_sub_blocks(self, pop_flat: np.ndarray) -> np.ndarray:
        """
        (B, 54) flat genomes -> (B, 3, 3, 6) sub_blocks.

        Uses Fortran ordering to mirror CrystalGeometryTemplate._build_full_block,
        which calls .reshape((3, 3, 6), order="F") on the same flat data.
        """
        B = pop_flat.shape[0]
        out = np.empty((B, 3, 3, 6), dtype=np.int8)
        for i in range(B):
            out[i] = pop_flat[i].reshape((3, 3, 6), order="F")
        return out

    def _write_population(self, pop_flat: np.ndarray, out_dir: str) -> None:
        """
        Write every individual in `pop_flat` (shape (B, 54)) as a geometry
        directory under `out_dir`. Designs are numbered 1..B in the same
        order as the input population, so pop_flat[0] -> design 001, etc.
        Since the population is sorted best-first, design 001 is the best
        individual.
        """
        os.makedirs(out_dir, exist_ok=True)
        sub_blocks = self._to_sub_blocks(pop_flat)
        B = sub_blocks.shape[0]
        for i in range(B):
            full_map, cube_pos = self.geom.apply_module(sub_blocks[i])
            self.geom.write_geometry(
                full_map, cube_pos, out_dir=out_dir, idx=i + 1
            )

    def _repair(self, g: np.ndarray) -> np.ndarray:
        """Force exactly n_crystal ones in g (in place). Used after crossover."""
        ones = int(g.sum())
        if ones == self.n_crystal:
            return g
        if ones > self.n_crystal:
            ones_idx = np.flatnonzero(g)
            drop = self.rng.choice(
                ones_idx, size=ones - self.n_crystal, replace=False
            )
            g[drop] = 0
        else:
            zeros_idx = np.flatnonzero(g == 0)
            add = self.rng.choice(
                zeros_idx, size=self.n_crystal - ones, replace=False
            )
            g[add] = 1
        return g

    # ------------------------------------------------------------------
    #  Genetic operators
    # ------------------------------------------------------------------

    def _tournament(self, fitness: np.ndarray) -> int:
        """Pick `tournament_k` random individuals, return index of the best (lowest)."""
        idx = self.rng.choice(self.pop_size, size=self.tournament_k, replace=False)
        return int(idx[np.argmin(fitness[idx])])

    def _crossover_uniform(
        self, p1: np.ndarray, p2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask = self.rng.random(self.N_GENES) < 0.5
        c1 = np.where(mask, p1, p2).astype(np.int8)
        c2 = np.where(mask, p2, p1).astype(np.int8)
        return self._repair(c1), self._repair(c2)

    def _crossover_twopoint(
        self, p1: np.ndarray, p2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        a, b = sorted(self.rng.integers(0, self.N_GENES, size=2).tolist())
        if a == b:
            b = min(b + 1, self.N_GENES)
        c1 = p1.copy()
        c2 = p2.copy()
        c1[a:b], c2[a:b] = p2[a:b].copy(), p1[a:b].copy()
        return self._repair(c1), self._repair(c2)

    def _mutate(self, g: np.ndarray) -> np.ndarray:
        """
        Swap-mutation: with prob `_effective_mut_rate`, swap n_swaps random
        (1, 0) pairs. Count-preserving, so no repair needed afterwards.
        """
        if self.rng.random() >= self._effective_mut_rate:
            return g
        ones_idx = np.flatnonzero(g)
        zeros_idx = np.flatnonzero(g == 0)
        n = min(self.n_swaps, len(ones_idx), len(zeros_idx))
        if n == 0:
            return g
        out = g.copy()
        a = self.rng.choice(ones_idx, size=n, replace=False)
        b = self.rng.choice(zeros_idx, size=n, replace=False)
        out[a] = 0
        out[b] = 1
        return out

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, pop_flat: np.ndarray) -> np.ndarray:
        """
        Evaluate `pop_flat` (shape (B, 54)) by calling evaluate_metric_fn one
        chunk at a time, so that no single simulator call has to handle more
        than `chunk_size` designs. Concatenates per-chunk fitness vectors
        back into a single array of length B.
 
        If `chunk_size` is None or B <= chunk_size, falls back to a single
        simulator call (original behaviour).
        """
        sub_blocks = self._to_sub_blocks(pop_flat)
        B = sub_blocks.shape[0]
 
        # Single-call fast path
        if self.chunk_size is None or B <= self.chunk_size:
            metrics = self.evaluate_metric_fn(sub_blocks, self.geom)
            return np.asarray(metrics, dtype=np.float32)
 
        # Chunked path
        out = np.empty(B, dtype=np.float32)
        n_chunks = (B + self.chunk_size - 1) // self.chunk_size  # ceil-div
        for c, start in enumerate(range(0, B, self.chunk_size)):
            stop = min(start + self.chunk_size, B)
            chunk = sub_blocks[start:stop]
            if self.chunk_verbose and n_chunks > 1:
                print(
                    f"    chunk {c + 1}/{n_chunks}  "
                    f"designs [{start}:{stop})"
                )
            chunk_metrics = self.evaluate_metric_fn(chunk, self.geom)
            out[start:stop] = np.asarray(chunk_metrics, dtype=np.float32)
        return out


    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------

    def train(
        self,
        num_generations: int,
        print_every: int = 1,
        save_every: int = 5,
    ) -> dict:
        """
        Run the GA for `num_generations` generations and return a history dict
        compatible with utils.save_history.
        """
        os.makedirs("logs/fim_fwhm", exist_ok=True)
        os.makedirs("crystal_geometry_history", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

        # ---- Generation 0 --------------------------------------------------
        population = self._initial_population()
        fitness = self._evaluate(population)

        order = np.argsort(fitness)
        population = population[order]
        fitness = fitness[order]

        if fitness[0] < self.best_metric:
            self.best_metric = float(fitness[0])
            self.best_genome = population[0].copy()

        history = {
            "generation":  [0],
            "avg_metric":  [float(fitness.mean())],
            "min_metric":  [float(fitness.min())],
            "best_metric": [self.best_metric],
            "mut_rate":    [self._effective_mut_rate],
        }

        print(
            f"[Gen   0] avg_metric={fitness.mean():.4f}  "
            f"min_metric={fitness.min():.4f}  best={self.best_metric:.4f}  "
            f"mut_rate={self._effective_mut_rate:.3f}"
        )

        # ---- Subsequent generations ---------------------------------------
        for gen in range(1, num_generations + 1):
            # Adaptive-mutation bookkeeping
            if self.adaptive_mut:
                if self._boost_counter > 0:
                    self._boost_counter -= 1
                    if self._boost_counter == 0:
                        self._effective_mut_rate = self.mutation_rate
                elif self._stall_counter >= self.stall_gens:
                    self._effective_mut_rate = self.mut_boost
                    self._boost_counter = self.mut_boost_gens
                    self._stall_counter = 0

            # Elitism: copy top-N unchanged (no re-evaluation needed)
            elites    = population[: self.n_elite].copy()
            elite_fit = fitness[: self.n_elite].copy()

            # Generate offspring via selection + crossover + mutation
            n_offspring = self.pop_size - self.n_elite
            offspring = []
            while len(offspring) < n_offspring:
                i1 = self._tournament(fitness)
                i2 = self._tournament(fitness)
                p1, p2 = population[i1], population[i2]

                if self.rng.random() < self.crossover_rate:
                    if self.crossover_type == "uniform":
                        c1, c2 = self._crossover_uniform(p1, p2)
                    else:
                        c1, c2 = self._crossover_twopoint(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                c1 = self._mutate(c1)
                c2 = self._mutate(c2)

                offspring.append(c1)
                if len(offspring) < n_offspring:
                    offspring.append(c2)

            offspring = np.stack(offspring, axis=0)
            off_fit = self._evaluate(offspring)

            # Combine elites + offspring -> next generation
            population = np.concatenate([elites,    offspring], axis=0)
            fitness    = np.concatenate([elite_fit, off_fit  ], axis=0)

            order = np.argsort(fitness)
            population = population[order]
            fitness    = fitness[order]

            # Track best
            if fitness[0] < self.best_metric:
                self.best_metric = float(fitness[0])
                self.best_genome = population[0].copy()
                self._stall_counter = 0
            else:
                self._stall_counter += 1

            # Logging
            history["generation" ].append(gen)
            history["avg_metric" ].append(float(fitness.mean()))
            history["min_metric" ].append(float(fitness.min()))
            history["best_metric"].append(self.best_metric)
            history["mut_rate"   ].append(self._effective_mut_rate)

            if gen % print_every == 0:
                print(
                    f"[Gen {gen:3d}] avg_metric={fitness.mean():.4f}  "
                    f"min_metric={fitness.min():.4f}  best={self.best_metric:.4f}  "
                    f"mut_rate={self._effective_mut_rate:.3f}  "
                    f"stall={self._stall_counter}"
                )
                # Same artifact-copy hook RLAgent uses; ignore if dirs missing
                try:
                    subprocess.run(
                        f"cp fim_fwhm/fim_fwhm_agent.bin "
                        f"logs/fim_fwhm/ga_gen{gen:03d}.bin",
                        shell=True, check=True,
                    )
                except subprocess.CalledProcessError:
                    pass

            if gen % save_every == 0:
                snapshot_dir = (
                    f"crystal_geometry_history/crystal_geometry_ga_gen{gen:03d}"
                )
                self._write_population(population, snapshot_dir)
                print(f"[Checkpoint] Wrote {len(population)} designs to {snapshot_dir}")

                ckpt_path = f"checkpoints/ga_gen{gen:03d}.npz"
                np.savez(
                    ckpt_path,
                    generation=gen,
                    population=population,
                    fitness=fitness,
                    best_metric=self.best_metric,
                    best_genome=(
                        self.best_genome
                        if self.best_genome is not None
                        else np.zeros(self.N_GENES, dtype=np.int8)
                    ),
                    baseline_metric=self.baseline_metric,
                )
                print(f"[Checkpoint] Saved GA state to {ckpt_path}")

        # Save the best sub_block found, for downstream inspection / plotting
        if self.best_genome is not None:
            best_sub = self.best_genome.reshape((3, 3, 6), order="F")
            np.save("checkpoints/ga_best_sub_block.npy", best_sub)
            print(
                f"[Done] Best FWHM = {self.best_metric:.4f}, "
                f"saved best sub_block to checkpoints/ga_best_sub_block.npy"
            )

        return history
