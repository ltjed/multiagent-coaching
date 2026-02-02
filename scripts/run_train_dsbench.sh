#!/bin/bash
# ============================================================================
# DSBench MODELING Pipeline Training Script
# ============================================================================
#
# This script trains a 3-agent data science pipeline on MODELING TASKS ONLY:
# - Agent 1: Data Engineer (EDA + preprocessing + feature engineering)
# - Agent 2: Modeler (algorithm selection + training + tuning)
# - Agent 3: Analyst (prediction generation + format verification)
#
# Each agent is trained with REINFORCE++ using external coach (Gemini 2.5 Pro)
#
# KEY FEATURES:
# 1. Training data: dsbench_modeling_train.json (64 Kaggle-style tasks)
# 2. Eval data: dsbench_modeling_eval.json (8 tasks, held-out)
# 3. Total: 73 usable tasks → 64 train + 8 eval (8:1 ratio for batch efficiency)
# 4. Tool-enabled agents with code execution (SandboxFusion)
# 5. Per-action coaching for all agents
# 6. REINFORCE++ algorithm for stable training
#
# DATASET:
# - Modeling tasks ONLY (Kaggle-style prediction tasks)
# - Analysis tasks (416 questions) in separate pipeline
#
# REQUIREMENTS:
# - UV environment activated
# - Ray cluster running (started automatically if not running)
# - DSBench data prepared (run scripts/prepare_dsbench_data.py first)
# - SandboxFusion auto-started on available port (8080-8090)
#
# Usage: bash scripts/run_train_dsbench.sh <MODEL_DIR> [WANDB_KEY]
# Example: bash scripts/run_train_dsbench.sh Qwen/Qwen3-4B
# ============================================================================

MODEL_DIR=${1}
WANDB_KEY=${2:-""}
ROOT_DIR=$(pwd)

if [ -z "$MODEL_DIR" ]; then
    echo "Error: MODEL_DIR is required"
    echo "Usage: bash $0 <MODEL_DIR> [WANDB_KEY]"
    echo "Example: bash $0 Qwen/Qwen3-4B"
    exit 1
fi

echo "============================================================================"
echo "DSBench MODELING Pipeline Training"
echo "============================================================================"
echo "Training data: dsbench_modeling_train.json (64 Kaggle-style tasks)"
echo "Eval data: dsbench_modeling_eval.json (8 tasks)"
echo "Total: 73 usable tasks → 64 train + 8 eval (8:1 ratio for batch efficiency)"
echo "Algorithm: REINFORCE++"
echo "Model: $MODEL_DIR"
echo "Agents: Data Engineer → Modeler → Analyst"
echo "============================================================================"
echo ""

# Activate UV environment
source .venv/bin/activate || { echo "UV environment not found"; exit 1; }
echo "✓ UV environment activated"

# Ensure Ray is running
if ! ray status &> /dev/null; then
    echo "Starting Ray cluster..."
    ray start --head --port=6379 --disable-usage-stats
    sleep 3
else
    echo "✓ Ray cluster already running"
fi

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ Environment variables loaded"
fi

# ============================================================================
# SandboxFusion Setup (Code Execution Service)
# ============================================================================
# Each training session gets its own SandboxFusion instance to avoid conflicts
SANDBOX_DIR="${SANDBOX_DIR:-$HOME/SandboxFusion}"

# Cleanup function to stop SandboxFusion on exit
cleanup_sandbox() {
    if [ -n "$SANDBOX_PORT" ]; then
        local pid_file="/tmp/sandbox_fusion_${SANDBOX_PORT}.pid"
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file" 2>/dev/null)
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                echo "Stopping SandboxFusion (PID: $pid) on port ${SANDBOX_PORT}..."
                # Kill the entire process group to ensure uvicorn is stopped
                kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null
                rm -f "$pid_file"
            fi
        fi
        # Also kill any uvicorn on our port as fallback
        pkill -f "uvicorn.*--port ${SANDBOX_PORT}" 2>/dev/null || true
    fi
}
trap cleanup_sandbox EXIT INT TERM

SANDBOX_BASE_PORT=${SANDBOX_BASE_PORT:-8080}  # Starting port to try
SANDBOX_MAX_PORT=8090  # Max port to try before giving up
SANDBOX_MAX_WAIT=30  # Max seconds to wait for sandbox to become healthy

# Find first available port
find_available_port() {
    local port=$SANDBOX_BASE_PORT
    while [ $port -le $SANDBOX_MAX_PORT ]; do
        if ! lsof -i :${port} &>/dev/null; then
            echo $port
            return 0
        fi
        echo "Port ${port} is occupied, trying next..." >&2
        port=$((port + 1))
    done
    echo "Error: No available ports in range ${SANDBOX_BASE_PORT}-${SANDBOX_MAX_PORT}" >&2
    return 1
}

check_sandbox_health() {
    local url=$1
    curl -s --max-time 2 -X POST "$url" \
        -H "Content-Type: application/json" \
        -d '{"language":"python","code":"print(1+1)"}' 2>/dev/null | grep -q '"status":"Success"'
}

start_sandbox() {
    local port=$1
    local log_file="/tmp/sandbox_fusion_${port}.log"
    local pid_file="/tmp/sandbox_fusion_${port}.pid"

    echo "Starting SandboxFusion on port ${port}..."

    # Start in a new process group using setsid for clean shutdown
    (
        cd "$SANDBOX_DIR" || exit 1
        # Use full path to sandbox conda environment's uvicorn
        # This avoids PATH conflicts with the UV .venv environment
        SANDBOX_UVICORN="$HOME/miniconda3/envs/sandbox/bin/uvicorn"
        if [ ! -f "$SANDBOX_UVICORN" ]; then
            echo "Error: Sandbox uvicorn not found at $SANDBOX_UVICORN" >&2
            exit 1
        fi

        # Run uvicorn directly (not through make) in a new session for proper cleanup
        # setsid creates a new process group so we can kill the whole group later
        setsid nohup "$SANDBOX_UVICORN" sandbox.server.server:app --host 0.0.0.0 --port ${port} \
            > "$log_file" 2>&1 &
        echo $! > "$pid_file"
    )

    # Small delay to let the PID file be written
    sleep 0.5
    echo "SandboxFusion starting (log: ${log_file}, pid: ${pid_file})"
}

# Find an available port and start SandboxFusion
echo "Finding available port for SandboxFusion..."
SANDBOX_PORT=$(find_available_port)
if [ -z "$SANDBOX_PORT" ]; then
    echo "Error: Could not find available port for SandboxFusion"
    exit 1
fi

SANDBOX_URL="http://127.0.0.1:${SANDBOX_PORT}/run_code"
echo "Using port ${SANDBOX_PORT} for this session"

# Start SandboxFusion on the available port
start_sandbox $SANDBOX_PORT

# Wait for health check
echo "Waiting for SandboxFusion to become healthy at ${SANDBOX_URL}..."
for i in $(seq 1 $SANDBOX_MAX_WAIT); do
    if check_sandbox_health "$SANDBOX_URL"; then
        echo "✓ SandboxFusion is ready at ${SANDBOX_URL} (waited ${i}s)"
        break
    fi
    if [ $i -eq $SANDBOX_MAX_WAIT ]; then
        echo "Error: SandboxFusion failed to start within ${SANDBOX_MAX_WAIT}s"
        echo "Check log: /tmp/sandbox_fusion_${SANDBOX_PORT}.log"
        tail -20 /tmp/sandbox_fusion_${SANDBOX_PORT}.log
        exit 1
    fi
    sleep 1
done
# ============================================================================

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Enable debug logging for DSBench
export DSBENCH_DEBUG=1

# Experiment configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SHORT_NAME=$(basename "$MODEL_DIR")
EXP="${TIMESTAMP}-${SHORT_NAME}-reinforce++-dsbench-modeling"

echo "
Experiment: $EXP
Model: $MODEL_DIR
Algorithm: REINFORCE++
Training set: 64 modeling tasks (Kaggle-style)
Eval set: 8 modeling tasks (held-out)
Ratio: 8:1 (optimal for batch efficiency with small dataset)
"

# Check if DSBench modeling data exists
if [ ! -f "${ROOT_DIR}/data/Bench/dsbench_modeling_train.json" ]; then
    echo "ERROR: DSBench modeling data not found!"
    echo ""
    echo "Please prepare DSBench modeling data first:"
    echo "  Option 1 (default path): python scripts/prepare_dsbench_data.py"
    echo "  Option 2 (custom path):  python scripts/prepare_dsbench_data.py --dsbench-root /path/to/DSBench"
    echo "  Option 3 (env variable): export DSBENCH_ROOT=/path/to/DSBench && python scripts/prepare_dsbench_data.py"
    echo ""
    echo "This will create:"
    echo "  - data/Bench/dsbench_modeling_train.json (64 tasks)"
    echo "  - data/Bench/dsbench_modeling_eval.json (8 tasks)"
    echo ""
    exit 1
fi

# Run training directly (inherits UV environment)
# REINFORCE++ configuration:
#   - Per-action coaching (Gemini 2.5 Pro)
#   - 2 samples per prompt (matches MathChat)
#   - Global batch normalization for advantages
#   - KL penalty in advantages (not loss)
python -m marti.cli.commands.train \
    --config-name dsbench_ds_pipeline \
    default_agent.pretrain="${MODEL_DIR}" \
    default_agent.save_path="${ROOT_DIR}/outputs/dsbench-modeling/${TIMESTAMP}/${SHORT_NAME}/model" \
    default_agent.ckpt_path="${ROOT_DIR}/outputs/dsbench-modeling/${TIMESTAMP}/${SHORT_NAME}/ckpt" \
    default_agent.ref_num_nodes=1 \
    default_agent.ref_num_gpus_per_node=2 \
    default_agent.actor_num_nodes=1 \
    default_agent.actor_num_gpus_per_node=2 \
    default_agent.vllm_num_engines=2 \
    default_agent.vllm_tensor_parallel_size=1 \
    default_agent.vllm_sync_backend="nccl" \
    default_agent.vllm_gpu_memory_utilization=0.6 \
    `# Match MathChat: 0.6 GPU util for stability` \
    default_agent.vllm_enable_sleep=True \
    default_agent.enable_prefix_caching=True \
    default_agent.colocate_all_models=True \
    default_agent.deepspeed_enable_sleep=True \
    `# CRITICAL: Enable DeepSpeed sleep mode to free GPU memory during vLLM rollouts` \
    default_agent.advantage_estimator="reinforce_plus_plus" \
    default_agent.num_episodes=30 \
    `# DSBench-specific: 30 episodes for longer training` \
    default_agent.n_samples_per_prompt=2 \
    `# Match MathChat: 2 samples per prompt` \
    default_agent.rollout_batch_size=16 \
    `# DSBench-specific: 16 batch (64 tasks / 4 steps per epoch)` \
    default_agent.micro_rollout_batch_size=4 \
    `# Match MathChat: 4 micro batch for ~50K token packing limit` \
    default_agent.train_batch_size=16 \
    default_agent.micro_train_batch_size=1 \
    `# Set to 1: safest for 24K context to avoid OOM` \
    default_agent.max_epochs=1 \
    default_agent.prompt_max_len=24576 \
    `# Match MathChat: 24K context for sequential agent outputs (Data Engineer -> Modeler -> Analyst)` \
    default_agent.generate_max_len=16384 \
    `# Increased to 16K: Reasoning models need more tokens for <think> + code` \
    default_agent.temperature=1.0 \
    default_agent.eval_temperature=0.6 \
    default_agent.top_p=1.0 \
    default_agent.actor_learning_rate=1e-6 \
    default_agent.critic_learning_rate=9e-6 \
    default_agent.zero_stage=3 \
    default_agent.bf16=True \
    default_agent.flash_attn=True \
    `# Match MathChat: Enable flash attention for memory optimization` \
    default_agent.adam_offload=True \
    default_agent.gradient_checkpointing=True \
    default_agent.normalize_reward=True \
    default_agent.init_kl_coef=0.01 \
    default_agent.use_kl_loss=False \
    default_agent.use_kl_estimator_k3=False \
    default_agent.save_steps=4 \
    `# DSBench-specific: Save every 4 steps = 1 epoch (64 tasks / 16 batch = 4)` \
    `# With num_episodes=30, will save 30 checkpoints total (steps 4, 8, ..., 120)` \
    default_agent.eval_steps=2 \
    `# DSBench-specific: Evaluate every 2 steps = 0.5 epochs` \
    `# With num_episodes=30, will evaluate 61 times (steps 0, 2, 4, ..., 120)` \
    default_agent.n_eval_samples_per_prompt=2 \
    `# Match MathChat: Each eval prompt evaluated 2 times per vLLM engine` \
    default_agent.eval_aggregation="mean" \
    `# Match MathChat: Average accuracy across eval samples` \
    default_agent.logging_steps=1 \
    default_agent.max_ckpt_num=10 \
    default_agent.max_samples=400000 \
    prompt_data="${ROOT_DIR}/data/Bench/dsbench_modeling_train.json" \
    `# DSBench-specific: 64 modeling tasks (Kaggle-style prediction tasks)` \
    input_key="prompt" \
    label_key="answer" \
    apply_chat_template=false \
    mask_truncated_completions=True \
    packing_samples=True \
    `# Required for async workflow with tool support` \
    filter_agents_data=True \
    `# CRITICAL FIX from MathChat: Ensures equal sample counts across workers` \
    `# Without this, variable agent actions cause workers to have different BufferItem counts,` \
    `# leading to NCCL deadlock when AllReduce calls don't match across workers.` \
    `# This truncates all workers to min sample count, guaranteeing equal gradient syncs.` \
    verify_task="dsbench" \
    verify_task_eval="dsbench" \
    extra_eval_tasks="[dsbench_modeling_eval]" \
    `# DSBench-specific: 8 modeling tasks (held-out eval set)` \
    extra_eval_dir="${ROOT_DIR}/data/Bench" \
    tools_config.num_workers=16 \
    tools_config.tools.code_interpreter.base_url="${SANDBOX_URL}" \
    `# Override sandbox URL with dynamically allocated port` \
    workflow_args.use_vertex_ai=true \
    `# Vertex AI required for Gemini 3 Pro` \
    workflow_args.vertex_project="${VERTEX_PROJECT:-your-gcp-project}" \
    workflow_args.vertex_location="global" \
    workflow_args.coach_model="gemini-3-pro-preview" \
    `# Gemini 3 Pro: 1M context, adaptive thinking` \
    use_wandb="${WANDB_API_KEY:-}" \
    wandb_project="MARTI-DSBench-Modeling" \
    wandb_run_name="${EXP}" \
    use_tensorboard="${ROOT_DIR}/logs/dsbench-modeling-${TIMESTAMP}-${SHORT_NAME}" \
    use_weave=true \
    weave_project="marti-dsbench-modeling-coach" \
    2>&1 | tee "${ROOT_DIR}/logs/${EXP}.log"

echo "
============================================================================
Training Complete!
============================================================================
Experiment: $EXP
Dataset: 64 train + 8 eval modeling tasks (73 usable total)
Logs: logs/${EXP}.log
Model: outputs/dsbench-modeling/${TIMESTAMP}/${SHORT_NAME}/model
Checkpoints: outputs/dsbench-modeling/${TIMESTAMP}/${SHORT_NAME}/ckpt
============================================================================
"
