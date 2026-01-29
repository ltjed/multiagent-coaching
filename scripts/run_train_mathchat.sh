#!/bin/bash
# ============================================================================
# AIME RANDOM SAMPLING TRAINING: 512 Problems from 933 Total
# ============================================================================
#
# This script trains the MathChat multi-agent system on randomly sampled
# AIME problems from the full AIME_1983_2024 dataset (1983-2024)
#
# KEY FEATURES:
# 1. Training data: AIME_1983_2024.json (933 total problems)
# 2. Random sampling: Randomly selects 512 problems using seed=42 for reproducibility
# 3. Sampled dataset saved to outputs directory for documentation/reproducibility
# 4. All train/eval sets are disjoint (no overlap)
#
# TRAINING FLOW:
# - 512 problems randomly sampled from 933 total
# - 512 problems × 2 samples/prompt = 1024 samples generated
# - 32 rollout batches (16 prompts/batch)
# - 32 training steps (16 samples/step)
# - Multiple evaluations during training
#
# EVALUATION FLOW:
# - Each eval prompt evaluated: n_eval_samples_per_prompt × vllm_num_engines times
# - Example: n_eval=1, vllm_num_engines=2 → each prompt evaluated 2 times (1×2)
# - Example: n_eval=4, vllm_num_engines=2 → each prompt evaluated 8 times (4×2)
# - Results aggregated using eval_aggregation method (mean/majority_vote/pass@k)
# - Higher n_eval_samples_per_prompt reduces variance but increases eval time linearly
#
# DATASET DETAILS:
# - AIME_1983_2024.json: 933 problems (AIME 1983-2024 from HuggingFace)
#   Format: {"prompt": str, "answer": str, "id": str, "year": int, "problem_number": int, "part": str/null}
#   Source: AIME competition problems 1983-2024
# - Random sampling: 512 problems selected with seed=42 (reproducible)
# - aime_eval_32.json: 32 AIME problems (eval, disjoint from train)
# - amc_eval_32.json: 32 AMC problems (eval, disjoint from train)
#
# RANDOM SAMPLING PARAMETERS:
# - max_samples=512: Number of problems to randomly select
# - random_sample_train=true: Enable random sampling (vs sequential first N)
# - save_sampled_dataset=true: Save sampled problems to outputs dir for reproducibility
# - seed=42: Random seed (ensures same 512 problems selected every run)
#
# Usage: bash scripts/run_train_mathchat_coach_aime_512.sh Qwen/Qwen3-4B-Thinking-2507
# ============================================================================

MODEL_DIR=${1}
WANDB_KEY=${2:-""}
ROOT_DIR=$(pwd)

if [ -z "$MODEL_DIR" ]; then
    echo "Error: MODEL_DIR is required"
    exit 1
fi

echo "============================================================================"
echo "AIME RANDOM SAMPLING TRAINING: 512 Problems from 933 Total"
echo "============================================================================"
echo "Training data: AIME_1983_2024.json (512 randomly sampled from 933 total)"
echo "Random sampling: ENABLED (seed=42, reproducible)"
echo "Eval data: aime_eval_32.json (32) + amc_eval_32.json (32) = 64 problems"
echo "Algorithm: REINFORCE++"
echo "Model: $MODEL_DIR"
echo "============================================================================"
echo ""

# Activate UV
source .venv/bin/activate || { echo "UV environment not found"; exit 1; }
echo "✓ UV environment activated"

# Ensure Ray is running (for distributed workers, not for job submission)
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


# Reduce memory fragmentation for training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Experiment config
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SHORT_NAME=$(basename "$MODEL_DIR")
EXP="${TIMESTAMP}-${SHORT_NAME}-reinforce++-mathchat-coach-aime_512"

echo "
Experiment: $EXP
Model: $MODEL_DIR
Algorithm: REINFORCE++
Training set: 512 AIME problems (randomly sampled from 933 total, seed=42)
Random sampling: ENABLED (reproducible with seed=42)
Sampled dataset will be saved to outputs directory
Eval sets: 32 AIME + 32 AMC (disjoint from train)
"

# Run training DIRECTLY (inherits UV environment!)
# REINFORCE++ configuration:
#   - Global batch normalization (prompt-agnostic)
#   - KL penalty in advantages (not loss)
#   - Works with n_samples_per_prompt=1 (no NaN)
python -m marti.cli.commands.train \
    --config-name mathchat_with_coach \
    default_agent.pretrain="${MODEL_DIR}" \
    default_agent.save_path="${ROOT_DIR}/outputs/reinforce++-mathchat-coach-aime_512/${TIMESTAMP}/${SHORT_NAME}/model" \
    default_agent.ckpt_path="${ROOT_DIR}/outputs/reinforce++-mathchat-coach-aime_512/${TIMESTAMP}/${SHORT_NAME}/ckpt" \
    default_agent.ref_num_nodes=1 \
    default_agent.ref_num_gpus_per_node=2 \
    default_agent.actor_num_nodes=1 \
    default_agent.actor_num_gpus_per_node=2 \
    default_agent.vllm_num_engines=2 \
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
    `# Reduced from 8 to 2: limits packed tensor to ~50K tokens (15GB logits vs 50GB with 8), experimenting with 4` \
    default_agent.train_batch_size=16 \
    default_agent.micro_train_batch_size=1 \
    default_agent.max_epochs=1 \
    default_agent.prompt_max_len=24576 \
    default_agent.generate_max_len=4096 \
    `# Memory optimization: 24K+8K=32K context (2x original 16K)` \
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
    `# CHECKPOINT FREQUENCY: Save every 32 steps = 2 full epochs` \
    `# Each checkpoint saved to unique dir: global_step{N} (no overwriting)` \
    default_agent.eval_steps=4 \
    `# EVALUATION FREQUENCY: Evaluate every 4 steps = 0.25 epochs` \
    default_agent.n_eval_samples_per_prompt=2 \
    `# EVALUATION SAMPLING: Each eval prompt evaluated N times (default: 1)` \
    `# Actual evaluations per prompt = n_eval_samples_per_prompt × vllm_num_engines` \
    `# Example: n_eval=4, vllm_num_engines=2 → each prompt evaluated 8 times (4×2)` \
    `# Then aggregated using eval_aggregation method (mean/majority_vote/pass@k)` \
    `# Higher N reduces variance but increases eval time linearly` \
    default_agent.eval_aggregation="mean" \
    `# EVALUATION AGGREGATION: How to combine multiple eval results per prompt` \
    `# "mean" = average accuracy (default, for continuous rewards)` \
    `# "majority_vote" = binary majority vote (for binary correctness)` \
    `# "pass@k" = at least k correct (e.g., "pass@3" means ≥3 correct out of N)` \
    `# Only used when n_eval_samples_per_prompt > 1` \
    default_agent.logging_steps=1 \
    default_agent.max_ckpt_num=10 \
    default_agent.max_samples=512 \
    `# DATASET: AIME_1983_2024 (933 total problems)` \
    prompt_data="${ROOT_DIR}/data/Bench/AIME_1983_2024.json" \
    `# RANDOM SAMPLING: Randomly select 512 problems from 933 total` \
    default_agent.random_sample_train=true \
    `# Enable random subset sampling (vs sequential first N problems)` \
    default_agent.save_sampled_dataset=true \
    `# Save sampled dataset to outputs directory for reproducibility` \
    `# Dataset format: {"prompt": str, "answer": str, "id": str, "year": int, "problem_number": int, "part": str/null}` \
    input_key="prompt" \
    label_key="answer" \
    apply_chat_template=false \
    mask_truncated_completions=True \
    packing_samples=True \
    `# Must keep packing=True (MultiAgentWorldAsync requires it for tool support)` \
    filter_agents_data=True \
    `# CRITICAL FIX: Ensures equal sample counts across workers when coder_max_turns>1` \
    `# Without this, variable coder actions cause workers to have different BufferItem counts,` \
    `# leading to NCCL deadlock when AllReduce calls don't match across workers.` \
    `# This truncates all workers to min sample count, guaranteeing equal gradient syncs.` \
    verify_task="math" \
    verify_task_eval="math" \
    extra_eval_tasks='["amc_eval_32","aime_2025"]' \
    extra_eval_dir="${ROOT_DIR}/data/Bench" \
    tools_config.num_workers=16 \
    workflow_args.coder_max_turns=2 \
    workflow_args.use_vertex_ai=true \
    workflow_args.vertex_project="${VERTEX_PROJECT:-your-gcp-project}" \
    workflow_args.vertex_location="global" \
    use_wandb="${WANDB_API_KEY:-}" \
    wandb_project="MARTI-MathChat-Coach-AIME-512" \
    wandb_run_name="${EXP}" \
    use_tensorboard="${ROOT_DIR}/logs/reinforce++-mathchat-coach-aime_512-${TIMESTAMP}-${SHORT_NAME}" \
    use_weave=false \
    weave_project="marti-mathchat-coach-aime-512" \
    2>&1 | tee "${ROOT_DIR}/logs/${EXP}.log"

echo "
============================================================================
Training Complete!
============================================================================
Experiment: $EXP
Logs: logs/${EXP}.log
Model: outputs/reinforce++-mathchat-coach-aime_512/${TIMESTAMP}/${SHORT_NAME}/model
Checkpoints: outputs/reinforce++-mathchat-coach-aime_512/${TIMESTAMP}/${SHORT_NAME}/ckpt
Sampled Dataset: outputs/reinforce++-mathchat-coach-aime_512/${TIMESTAMP}/${SHORT_NAME}/sampled_training_data.json
Metadata: outputs/reinforce++-mathchat-coach-aime_512/${TIMESTAMP}/${SHORT_NAME}/sampled_training_data_metadata.json
============================================================================
"
