# SCI-SPECT Crystal Configuration Optimization

This repository contains reinforcement-learning and CUDA simulation code for optimizing crystal configurations for an SCI-SPECT detector geometry. The Python training scripts generate candidate crystal layouts, evaluate them with compiled simulator binaries, and use the resulting imaging metric to update the search policy.

## Repository Layout

- `train.py` - actor-critic training entry point for the 8x8-style geometry search.
- `train_mcts.py` - AlphaZero/MCTS-style training entry point.
- `agent_*.py` - reinforcement-learning agents for FIM/FWHM, mutual coherence, Fourier crosstalk, replay, and related search variants.
- `crystal_geometry*.py` - geometry template builders that map compact design variables into full detector maps and cube position files.
- `network.py` - PyTorch actor-critic network used by the RL agents.
- `utils.py` - reward calculation, action masking, simulator invocation, and history saving helpers.
- `cuda_script/` - CUDA/C++ simulator source files.
- `executables/` - prebuilt simulator binaries used by the Python evaluation pipeline.
- `shell_script/` - helper scripts for compiling/running CUDA simulations and cleaning outputs.
- `crystal_geometry_rand_design*/` - generated or sampled crystal geometry design files.
- `fim_fwhm/`, `fourier_crosstalk/`, `mc_fwhm/`, `mutual_coherence/`, `voxel_crosstalk/` - metric-specific output folders.

## Requirements

The Python workflow expects:

- Python 3.10 or newer
- PyTorch
- NumPy
- SciPy
- A CUDA-capable GPU for simulator execution
- CUDA toolkit and compatible NVIDIA driver if rebuilding the CUDA executables

The CUDA scripts were developed around CUDA 12.x and NVIDIA `sm_90` targets. If you use a different GPU architecture, update the `-arch=` option in the compile commands before rebuilding.

## Quick Start

From the repository root:

```bash
python train.py
```

For the MCTS/AlphaZero-style workflow:

```bash
python train_mcts.py
```

The training loop writes generated geometries to `crystal_geometry_agent_design/`, runs the appropriate binary from `executables/`, and stores logs/checkpoints under directories such as `logs/`, `fim_fwhm/`, `crystal_geometry_history/`, and `checkpoints/`.

## Simulator Evaluation Flow

1. The agent samples a compact crystal/non-crystal action map.
2. A `CrystalGeometryTemplate` expands that action map into a full `35 x 35 x 17` detector map.
3. The geometry is written as `map_###.txt` and `cube_pos_###.txt`.
4. A CUDA simulator binary evaluates each design.
5. The binary metric output is read back into Python and converted into a training reward.

For example, `utils.evaluate_fim_fwhm` writes the current batch to `crystal_geometry_agent_design/` and calls:

```bash
./executables/fast_fim_tc99m_120mmFOV_batch crystal_geometry_agent_design fim_fwhm/fim_fwhm_agent.bin
```

## Rebuilding CUDA Binaries

Example build commands are documented in the scripts under `shell_script/`. A typical compile command looks like:

```bash
nvcc -O3 -use_fast_math --extended-lambda -arch=sm_90 \
  -o executables/fim_tc99m_2D_focus_120mmFOV \
  cuda_script/fim_profile_fwhm_Tc99m_2D.cu \
  -lcurand -lcublas
```

Adjust include paths, library paths, output names, and GPU architecture as needed for your local CUDA installation.

## Notes

- Many run artifacts are intentionally ignored by `.gitignore`, including checkpoints, binary metric files, logs, and NumPy/MATLAB outputs.
- Several workflows assume the output directories already exist. Create directories such as `logs/fim_fwhm`, `fim_fwhm`, or `checkpoints` before running a new training job if needed.
- The simulator commands are configured for local paths and GPU resources. Review `utils.py` and `shell_script/` before launching large jobs on a shared system.
