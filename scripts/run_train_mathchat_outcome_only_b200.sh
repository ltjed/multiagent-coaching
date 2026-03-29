#!/bin/bash
# ============================================================================
# MATHCHAT OUTCOME-ONLY BASELINE — 4×B200 (192GB each)
# ============================================================================
#
# Same as run_train_mathchat_outcome_only.sh but adapted for 4×B200 node.
#
# GPU LAYOUT:
#   GPU 0: problem_solver (1 GPU, colocated ref+actor+critic+vllm)
#   GPU 1: code_executor  (1 GPU, colocated)
#   GPU 2: verifier       (1 GPU, colocated)
#   GPU 3: idle
#
# CHANGES FROM 8×H100 VERSION:
#   - All agents: ref/actor/critic_num_gpus_per_node 2→1, vllm_num_engines 2→1
#   - B200 (192GB) holds what 2×H100 (160GB) held — no ZeRO sharding needed
#   - PYTORCH_CUDA_ALLOC_CONF: removed expandable_segments (incompatible with vLLM 0.17)
#   - Added NCCL_CUMEM_ENABLE=0 (required for sleep/wake on vLLM 0.17)
#
# ALL TRAINING HYPERPARAMETERS IDENTICAL for fair comparison with 8×H100 results.
#
# Usage: bash scripts/run_train_mathchat_outcome_only_b200.sh Qwen/Qwen3-4B
# ============================================================================

MODEL_DIR=${1}
WANDB_KEY=${2:-""}
ROOT_DIR=$(pwd)

if [ -z "$MODEL_DIR" ]; then
    echo "Error: MODEL_DIR is required"
    echo "Usage: bash scripts/run_train_mathchat_outcome_only_b200.sh <MODEL_PATH> [WANDB_KEY]"
    exit 1
fi

echo "============================================================================"
echo "MATHCHAT OUTCOME-ONLY BASELINE — 4×B200"
echo "============================================================================"
echo "Training data: AIME_1983_2024.json (512 randomly sampled from 933 total)"
echo "Reward: OUTCOME-ONLY (binary correct/incorrect, shared across all agents)"
echo "NO coach model (no API calls)"
echo "Algorithm: REINFORCE++"
echo "Model: $MODEL_DIR"
echo "GPU Layout: 3 agents × 1 GPU each (GPU 3 idle) on 4×B200"
echo "============================================================================"
echo ""

# Activate UV
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

# Load env vars
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ Environment variables loaded"
fi

# ============================================================================
# B200 / vLLM 0.17 Environment Variables
# ============================================================================
# DO NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — incompatible
# with vLLM 0.17's CuMemAllocator (used for sleep/wake memory management).
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export NCCL_CUMEM_ENABLE=0  # Required: prevents NCCL hangs during weight sync
export VLLM_ALLOW_INSECURE_SERIALIZATION=1  # Required: vLLM 0.17 msgspec can't serialize torch.dtype
export HYDRA_FULL_ERROR=1

# ============================================================================
# SandboxFusion Setup (Code Execution Service)
# ============================================================================
SANDBOX_DIR="${SANDBOX_DIR:-$HOME/SandboxFusion}"

cleanup_sandbox() {
    if [ -n "$SANDBOX_PORT" ]; then
        local pid_file="/tmp/sandbox_fusion_${SANDBOX_PORT}.pid"
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file" 2>/dev/null)
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                echo "Stopping SandboxFusion (PID: $pid) on port ${SANDBOX_PORT}..."
                kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null
                rm -f "$pid_file"
            fi
        fi
        pkill -f "uvicorn.*--port ${SANDBOX_PORT}" 2>/dev/null || true
    fi
}
trap cleanup_sandbox EXIT INT TERM

SANDBOX_BASE_PORT=${SANDBOX_BASE_PORT:-8080}
SANDBOX_MAX_PORT=8090
SANDBOX_MAX_WAIT=30

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
    (
        cd "$SANDBOX_DIR" || exit 1
        SANDBOX_UVICORN="$HOME/miniconda3/envs/sandbox/bin/uvicorn"
        if [ ! -f "$SANDBOX_UVICORN" ]; then
            echo "Error: Sandbox uvicorn not found at $SANDBOX_UVICORN" >&2
            exit 1
        fi
        setsid nohup "$SANDBOX_UVICORN" sandbox.server.server:app --host 0.0.0.0 --port ${port} \
            > "$log_file" 2>&1 &
        echo $! > "$pid_file"
    )
    sleep 0.5
    echo "SandboxFusion starting (log: ${log_file}, pid: ${pid_file})"
}

echo "Finding available port for SandboxFusion..."
SANDBOX_PORT=$(find_available_port)
if [ -z "$SANDBOX_PORT" ]; then
    echo "Error: Could not find available port for SandboxFusion"
    exit 1
fi

SANDBOX_URL="http://127.0.0.1:${SANDBOX_PORT}/run_code"
echo "Using port ${SANDBOX_PORT} for this session"

start_sandbox $SANDBOX_PORT

echo "Waiting for SandboxFusion to become healthy at ${SANDBOX_URL}..."
for i in $(seq 1 $SANDBOX_MAX_WAIT); do
    if check_sandbox_health "$SANDBOX_URL"; then
        echo "✓ SandboxFusion is ready at ${SANDBOX_URL} (waited ${i}s)"
        break
    fi
    if [ $i -eq $SANDBOX_MAX_WAIT ]; then
        echo "Error: SandboxFusion failed to start within ${SANDBOX_MAX_WAIT}s"
        tail -20 /tmp/sandbox_fusion_${SANDBOX_PORT}.log
        exit 1
    fi
    sleep 1
done
# ============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SHORT_NAME=$(basename "$MODEL_DIR")
EXP="${TIMESTAMP}-${SHORT_NAME}-reinforce++-mathchat-OUTCOME-ONLY-b200-aime_512"

echo "
Experiment: $EXP
Model: $MODEL_DIR
Algorithm: REINFORCE++ (OUTCOME-ONLY BASELINE)
Hardware: 4×B200 — 3 agents × 1 GPU each (GPU 3 idle)
Reward: Binary outcome (correct=10, incorrect=0) shared across all agents
Training set: 512 AIME problems (randomly sampled, seed=42)
Eval sets: 32 AIME 2025 + 32 AMC (disjoint from train)
"

# ============================================================================
# Training — uses B200-adapted config (1 GPU per agent instead of 2)
# All training hyperparameters identical to 8×H100 version.
# ============================================================================
python -m marti.cli.commands.train \
    --config-name mathchat_outcome_only_b200 \
    default_agent.pretrain="${MODEL_DIR}" \
    default_agent.save_path="${ROOT_DIR}/outputs/reinforce++-mathchat-outcome-only-b200/${TIMESTAMP}/${SHORT_NAME}/model" \
    default_agent.ckpt_path="${ROOT_DIR}/outputs/reinforce++-mathchat-outcome-only-b200/${TIMESTAMP}/${SHORT_NAME}/ckpt" \
    default_agent.max_samples=512 \
    prompt_data="${ROOT_DIR}/data/Bench/AIME_1983_2024.json" \
    default_agent.random_sample_train=true \
    default_agent.save_sampled_dataset=true \
    input_key="prompt" \
    label_key="answer" \
    apply_chat_template=false \
    mask_truncated_completions=True \
    packing_samples=True \
    filter_agents_data=True \
    verify_task="math" \
    verify_task_eval="math" \
    extra_eval_tasks='["amc_eval_32","aime_2025"]' \
    extra_eval_dir="${ROOT_DIR}/data/Bench" \
    tools_config.num_workers=16 \
    tools_config.tools.code_interpreter.base_url="${SANDBOX_URL}" \
    workflow_args.coder_max_turns=2 \
    use_wandb="${WANDB_KEY:-}" \
    wandb_project="MARTI-MathChat-Outcome-Only-B200" \
    wandb_run_name="${EXP}" \
    use_tensorboard="${ROOT_DIR}/logs/reinforce++-mathchat-outcome-only-b200-${TIMESTAMP}-${SHORT_NAME}" \
    use_weave=false \
    2>&1 | tee "${ROOT_DIR}/logs/${EXP}.log"

echo "
============================================================================
Training Complete!
============================================================================
Experiment: $EXP
Type: OUTCOME-ONLY BASELINE (no coach) on 4×B200
Logs: logs/${EXP}.log
Model: outputs/reinforce++-mathchat-outcome-only-b200/${TIMESTAMP}/${SHORT_NAME}/model
Checkpoints: outputs/reinforce++-mathchat-outcome-only-b200/${TIMESTAMP}/${SHORT_NAME}/ckpt
============================================================================
"
