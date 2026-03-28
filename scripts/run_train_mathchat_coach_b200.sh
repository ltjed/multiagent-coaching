#!/bin/bash
# ============================================================================
# MATHCHAT WITH COACH (MAPPA) — 4×B200 (192GB each)
# ============================================================================
#
# Adapted from run_train_mathchat.sh (8×H100) for 4×B200 node.
# Uses per-action coach process rewards (MAPPA).
#
# GPU LAYOUT: 3 agents × 1 GPU each (GPU 3 idle)
#
# COACH MODEL ABLATION:
#   Pass coach model as 2nd argument to compare different coach qualities:
#     bash scripts/run_train_mathchat_coach_b200.sh Qwen/Qwen3-4B gemini-2.5-flash   # paper default
#     bash scripts/run_train_mathchat_coach_b200.sh Qwen/Qwen3-4B gemini-2.0-flash-lite  # degraded
#     bash scripts/run_train_mathchat_coach_b200.sh Qwen/Qwen3-4B gemini-2.5-pro     # upgraded
#
# CHANGES FROM 8×H100: only GPU layout (num_gpus 2→1, num_engines 2→1)
# All training hyperparameters IDENTICAL for fair comparison.
# ============================================================================

MODEL_DIR=${1}
COACH_MODEL=${2:-"gemini-2.5-flash"}  # Default: paper's coach model
WANDB_KEY=${3:-""}
ROOT_DIR=$(pwd)

if [ -z "$MODEL_DIR" ]; then
    echo "Error: MODEL_DIR is required"
    echo "Usage: bash scripts/run_train_mathchat_coach_b200.sh <MODEL_PATH> [COACH_MODEL] [WANDB_KEY]"
    echo "  COACH_MODEL defaults to gemini-2.5-flash (paper default)"
    echo "  Examples: gemini-2.5-flash, gemini-2.0-flash-lite, gemini-2.5-pro"
    exit 1
fi

# Sanitize coach model name for file paths
COACH_SHORT=$(echo "$COACH_MODEL" | tr '/' '_' | tr '.' '-')

echo "============================================================================"
echo "MATHCHAT WITH COACH (MAPPA) — 4×B200"
echo "============================================================================"
echo "Training data: AIME_1983_2024.json (512 randomly sampled from 933 total)"
echo "Reward: PER-ACTION PROCESS REWARDS from coach (MAPPA)"
echo "Coach model: $COACH_MODEL"
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
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export NCCL_CUMEM_ENABLE=0
export HYDRA_FULL_ERROR=1

# ============================================================================
# SandboxFusion Setup
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
EXP="${TIMESTAMP}-${SHORT_NAME}-reinforce++-mathchat-coach-${COACH_SHORT}-b200-aime_512"

echo "
Experiment: $EXP
Model: $MODEL_DIR
Coach: $COACH_MODEL
Algorithm: REINFORCE++ with MAPPA (per-action process rewards)
Hardware: 4×B200 — 3 agents × 1 GPU each
Training set: 512 AIME problems (randomly sampled, seed=42)
Eval sets: 32 AIME 2025 + 32 AMC
"

# ============================================================================
# Training — B200 GPU layout, coach model from CLI argument
# All other hyperparameters identical to 8×H100 paper configuration.
# ============================================================================
python -m marti.cli.commands.train \
    --config-name mathchat_with_coach \
    default_agent.pretrain="${MODEL_DIR}" \
    default_agent.save_path="${ROOT_DIR}/outputs/reinforce++-mathchat-coach-${COACH_SHORT}-b200/${TIMESTAMP}/${SHORT_NAME}/model" \
    default_agent.ckpt_path="${ROOT_DIR}/outputs/reinforce++-mathchat-coach-${COACH_SHORT}-b200/${TIMESTAMP}/${SHORT_NAME}/ckpt" \
    default_agent.ref_num_nodes=1 \
    default_agent.ref_num_gpus_per_node=1 \
    default_agent.actor_num_nodes=1 \
    default_agent.actor_num_gpus_per_node=1 \
    default_agent.vllm_num_engines=1 \
    default_agent.vllm_tensor_parallel_size=1 \
    default_agent.vllm_sync_backend="nccl" \
    default_agent.vllm_gpu_memory_utilization=0.7 \
    default_agent.vllm_enable_sleep=True \
    default_agent.enable_prefix_caching=True \
    default_agent.colocate_all_models=True \
    default_agent.deepspeed_enable_sleep=True \
    default_agent.advantage_estimator="reinforce_plus_plus" \
    default_agent.num_episodes=8 \
    default_agent.n_samples_per_prompt=2 \
    default_agent.rollout_batch_size=32 \
    default_agent.micro_rollout_batch_size=8 \
    default_agent.train_batch_size=16 \
    default_agent.micro_train_batch_size=1 \
    default_agent.max_epochs=1 \
    default_agent.prompt_max_len=24576 \
    default_agent.generate_max_len=4096 \
    default_agent.temperature=1.0 \
    default_agent.eval_temperature=0.6 \
    default_agent.top_p=1.0 \
    default_agent.actor_learning_rate=1e-6 \
    default_agent.critic_learning_rate=9e-6 \
    default_agent.zero_stage=3 \
    default_agent.bf16=True \
    default_agent.flash_attn=True \
    default_agent.adam_offload=True \
    default_agent.gradient_checkpointing=True \
    default_agent.normalize_reward=True \
    default_agent.init_kl_coef=0.01 \
    default_agent.use_kl_loss=False \
    default_agent.use_kl_estimator_k3=False \
    default_agent.save_steps=32 \
    default_agent.eval_steps=4 \
    default_agent.n_eval_samples_per_prompt=2 \
    default_agent.eval_aggregation="mean" \
    default_agent.logging_steps=1 \
    default_agent.max_ckpt_num=10 \
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
    workflow_args.coach_model="${COACH_MODEL}" \
    workflow_args.use_vertex_ai=true \
    workflow_args.vertex_project="${VERTEX_PROJECT:-your-gcp-project}" \
    workflow_args.vertex_location="global" \
    use_wandb="${WANDB_KEY:-}" \
    wandb_project="MARTI-MathChat-Coach-B200" \
    wandb_run_name="${EXP}" \
    use_tensorboard="${ROOT_DIR}/logs/reinforce++-mathchat-coach-${COACH_SHORT}-b200-${TIMESTAMP}-${SHORT_NAME}" \
    use_weave=false \
    2>&1 | tee "${ROOT_DIR}/logs/${EXP}.log"

echo "
============================================================================
Training Complete!
============================================================================
Experiment: $EXP
Coach: $COACH_MODEL
Logs: logs/${EXP}.log
Model: outputs/reinforce++-mathchat-coach-${COACH_SHORT}-b200/${TIMESTAMP}/${SHORT_NAME}/model
============================================================================
"
