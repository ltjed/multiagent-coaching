A framework for training multi-agent LLM systems with reinforcement learning, featuring external LLM judges for process reward evaluation.

## Quick Start

### Installation

```bash
# Create environment with UV (recommended)
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,judge]"
```

### Key Training Scripts

**1. MathChat Training** - Multi-agent math reasoning with code execution:
```bash
bash scripts/run_train_mathchat.sh <MODEL_PATH>
```
- 3-agent system: Problem Solver → Code Executor → Verifier
- External LLM coach (Gemini) provides per-action process rewards
- Trains on AIME competition problems

**2. DSBench Training** - Multi-agent data science pipeline:
```bash
bash scripts/run_train_dsbench.sh <MODEL_PATH>
```
- 3-agent system: Data Engineer → Modeler → Analyst
- Kaggle-style modeling tasks with code execution
- Process reward from external LLM coach

## Prerequisites

### SandboxFusion (Code Execution)

Both training scripts require [SandboxFusion](https://github.com/bytedance/SandboxFusion) for secure code execution. Set up once:

```bash
# Clone and install
git clone https://github.com/bytedance/SandboxFusion.git ~/SandboxFusion
cd ~/SandboxFusion

# Create conda environment
conda create -n sandbox python=3.12 -y
conda activate sandbox
pip install poetry
poetry install

# Create runtime environment
conda create -n sandbox-runtime python=3.11 -y
conda activate sandbox-runtime
pip install -r runtime/python/requirements.txt
```

The training scripts automatically start SandboxFusion on an available port (8080-8090).

To override the default location, set `SANDBOX_DIR`:
```bash
export SANDBOX_DIR=/path/to/SandboxFusion
```

### External LLM Coach (Vertex AI)

For Gemini-based process evaluation, configure Google Cloud:

```bash
export VERTEX_PROJECT=your-gcp-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

Alternatively, set `workflow_args.use_vertex_ai=false` to use the standard Gemini API with `GOOGLE_API_KEY`.

## Hardware Requirements

- Minimum: 2x 80GB GPUs (A100/H100) for single-agent training
- Recommended: 4-8x 80GB GPUs for multi-agent training

## Configuration

Training configs are in `marti/cli/configs/`:
- `mathchat_with_coach.yaml` - MathChat configuration
- `dsbench_ds_pipeline.yaml` - DSBench configuration

Key parameters can be overridden via command line:
```bash
python -m marti.cli.commands.train \
    --config-name mathchat_with_coach \
    default_agent.pretrain=/path/to/model \
    default_agent.vllm_num_engines=2
```

## Project Structure

```
marti/
├── cli/configs/          # Hydra configuration files
├── controllers/          # Multi-agent orchestration
├── trainers/ppo/         # RL training (REINFORCE++, GRPO)
├── verifiers/            # Task-specific reward computation
├── worlds/               # Agent environments and workflows
│   ├── workflows/        # MathChat, DSBench workflow definitions
│   └── tools/            # Code execution, search tools
└── models/               # vLLM inference, actor-critic models

scripts/
├── run_train_mathchat.sh # MathChat training script
├── run_train_dsbench.sh  # DSBench training script
└── setup_sandbox.sh      # SandboxFusion setup helper

data/
└── Bench/                # Evaluation datasets (AIME, AMC, DSBench)
```

## Logging

- **Weights & Biases**: Set `use_wandb=<API_KEY>`
- **TensorBoard**: Logs saved to `logs/` directory
- **Weave** (LLM tracing): Set `use_weave=true`
