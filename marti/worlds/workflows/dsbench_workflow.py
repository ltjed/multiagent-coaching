"""
DSBench Modeling Pipeline Workflow with External Coach Evaluation

MODELING TASKS ONLY (Kaggle-style prediction tasks)
Total dataset: 92 tasks → 64 train + 28 eval

Workflow: Data Engineer → Modeler → Analyst
- Data Engineer: Performs EDA, preprocessing, feature engineering
- Modeler: Selects algorithms, trains models, tunes hyperparameters
- Analyst: Generates predictions, verifies format, creates submission

Each agent uses per-action coaching with Gemini 2.5 Pro.

Data Passing Strategy:
- Agent 1 outputs preprocessing summary (not raw data)
- Agent 2 regenerates preprocessing code (deterministic)
- Agent 3 generates final predictions

File Handling:
- CSV files (train.csv, test.csv) are uploaded to SandboxFusion for code execution
- Files persist only within single code execution (stateless sandbox)
- Files are re-uploaded with every agent action that executes code

Note: data_analysis tasks (416 questions) handled in separate pipeline
"""

import os
import re
import base64
import json
from typing import Dict, List, Any, Optional
from marti.helpers.logging import init_logger
from marti.worlds.workflows.utils import apply_template_with_tokenizer
from marti.worlds.workflows.coach_utils import run_agent_with_per_action_coaching
from marti.verifiers.coach import create_coach

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "INFO"))


def strip_thinking_blocks(text: str) -> str:
    """
    Remove <think>...</think> blocks from agent output.
    This saves context tokens when passing outputs to subsequent agents.

    The thinking blocks can be:
    - <think>content</think>
    - <think>\n<think>\n</think> (nested/empty)
    - Multiple blocks in sequence
    """
    if not text:
        return text

    # Remove all <think>...</think> blocks (including nested, multiline)
    # Use DOTALL to match across newlines
    cleaned = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

    # Also remove standalone <think> or </think> tags that might be orphaned
    cleaned = re.sub(r'</?think>\s*', '', cleaned)

    # Clean up excessive whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()

# ========================================================================
# Cleaned Prompts Loader
# ========================================================================
# Load cleaned task descriptions that remove distracting Kaggle boilerplate
# (Discord links, \boxed{} instructions, LaTeX formulas, promotional content)
# and keep only useful information (evaluation metric, submission format, etc.)

_cleaned_prompts_cache = None

def get_cleaned_prompt(task_id: str) -> Optional[str]:
    """
    Get cleaned task description for a given task_id.

    Returns the cleaned prompt if available, or None if not found.
    The cleaned prompts remove:
    - Story/background fluff
    - Discord/community links
    - LaTeX math formulas
    - "Please put your answer in \\boxed{}" instructions
    - Acknowledgements sections

    And keep:
    - Evaluation metric
    - Submission format with column names
    - Column/feature descriptions
    - Target variable description
    """
    global _cleaned_prompts_cache

    if _cleaned_prompts_cache is None:
        # Find the cleaned prompts file
        current_file = os.path.abspath(__file__)
        repo_root = current_file
        for _ in range(10):
            parent = os.path.dirname(repo_root)
            if os.path.exists(os.path.join(parent, '.git')):
                repo_root = parent
                break
            if parent == repo_root:
                break
            repo_root = parent

        cleaned_prompts_path = os.path.join(repo_root, "data", "Bench", "dsbench_modeling_cleaned_prompts.json")

        if os.path.exists(cleaned_prompts_path):
            try:
                with open(cleaned_prompts_path, 'r') as f:
                    _cleaned_prompts_cache = json.load(f)
                logger.info(f"Loaded {len(_cleaned_prompts_cache)} cleaned task prompts")
            except Exception as e:
                logger.warning(f"Failed to load cleaned prompts: {e}")
                _cleaned_prompts_cache = {}
        else:
            logger.warning(f"Cleaned prompts file not found: {cleaned_prompts_path}")
            _cleaned_prompts_cache = {}

    if task_id in _cleaned_prompts_cache:
        return _cleaned_prompts_cache[task_id].get("cleaned_prompt")

    return None

# Weave import (optional dependency)
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False

# Global coach instance
_global_coach = None
_weave_initialized = False


def get_coach(workflow_args: Dict) -> Any:
    """Get or create coach instance (singleton)"""
    global _global_coach, _weave_initialized

    if _global_coach is None:
        # Initialize Weave in Ray worker if enabled
        if workflow_args.get("use_weave", False):
            if WEAVE_AVAILABLE and not _weave_initialized:
                weave_project = workflow_args.get("weave_project", "marti-dsbench-coach")

                # Load .env file for WANDB_API_KEY
                from dotenv import load_dotenv
                current_file = os.path.abspath(__file__)
                repo_root = current_file
                for _ in range(10):
                    parent = os.path.dirname(repo_root)
                    if os.path.exists(os.path.join(parent, '.git')):
                        repo_root = parent
                        break
                    if parent == repo_root:
                        break
                    repo_root = parent

                env_file = os.path.join(repo_root, '.env')
                if os.path.exists(env_file):
                    load_dotenv(env_file)

                wandb_key = os.getenv("WANDB_API_KEY")

                try:
                    if wandb_key:
                        import wandb
                        wandb.login(key=wandb_key, relogin=True, force=True)

                    weave.init(project_name=weave_project)
                    logger.info(f"[Weave] Successfully initialized: {weave_project}")
                    _weave_initialized = True
                except Exception as e:
                    logger.warning(f"[Weave] Failed to initialize: {e}")

        coach_model = workflow_args.get("coach_model", "gemini-2.5-pro")
        coach_type = workflow_args.get("coach_type", "simple")
        use_vertex_ai = workflow_args.get("use_vertex_ai", False)
        vertex_project = workflow_args.get("vertex_project", None)
        vertex_location = workflow_args.get("vertex_location", "global")
        max_output_tokens = workflow_args.get("coach_max_output_tokens", None)
        thinking_budget = workflow_args.get("coach_thinking_budget", None)

        logger.info(f"Initializing coach: {coach_type}/{coach_model}")

        prompt_template = workflow_args.get("coach_prompt_template", None)

        _global_coach = create_coach(
            model=coach_model,
            coach_type=coach_type,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
            use_vertex_ai=use_vertex_ai,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            prompt_template=prompt_template
        )

        logger.info("Coach initialized successfully")

    return _global_coach


def _apply_weave_decorator(func):
    """Apply weave.op decorator if weave is available"""
    if WEAVE_AVAILABLE:
        return weave.op()(func)
    return func


def prepare_task_files(metadata: Dict) -> Dict[str, str]:
    """
    Prepare CSV files for upload to SandboxFusion.

    Reads task data files from the host filesystem and encodes them as base64
    for upload to the SandboxFusion API. Files will be available in the sandbox
    working directory with simplified names (train.csv, test.csv).

    Args:
        metadata: Task metadata containing file paths
            - train_file: Absolute path to training CSV
            - test_file: Absolute path to test CSV

    Returns:
        Dict mapping simplified filename → base64-encoded content
        Example: {"train.csv": "base64_content...", "test.csv": "base64_content..."}

    Notes:
        - Files are uploaded with EVERY code execution (sandbox is stateless)
        - File size limit: ~10MB per file (base64 increases size by ~33%)
        - Large files (>10MB) will trigger a warning but still upload
    """
    files = {}

    # Read train.csv
    train_file_path = metadata.get("train_file")
    if train_file_path and os.path.exists(train_file_path):
        file_size_mb = os.path.getsize(train_file_path) / (1024 * 1024)
        if file_size_mb > 10:
            logger.warning(f"Large file detected: train.csv ({file_size_mb:.1f}MB). Upload may be slow.")

        with open(train_file_path, 'rb') as f:
            files["train.csv"] = base64.b64encode(f.read()).decode('utf-8')
        logger.info(f"Prepared train.csv ({file_size_mb:.2f}MB)")
    else:
        logger.warning(f"Train file not found or not specified: {train_file_path}")

    # Read test.csv
    test_file_path = metadata.get("test_file")
    if test_file_path and os.path.exists(test_file_path):
        file_size_mb = os.path.getsize(test_file_path) / (1024 * 1024)
        if file_size_mb > 10:
            logger.warning(f"Large file detected: test.csv ({file_size_mb:.1f}MB). Upload may be slow.")

        with open(test_file_path, 'rb') as f:
            files["test.csv"] = base64.b64encode(f.read()).decode('utf-8')
        logger.info(f"Prepared test.csv ({file_size_mb:.2f}MB)")
    else:
        logger.warning(f"Test file not found or not specified: {test_file_path}")

    return files


@_apply_weave_decorator
async def workflow(
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],
    tool_manager,
    task: str,
    metadata: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    DSBench workflow with per-action coach evaluation

    Args:
        prompt: Task description
        label: Ground truth answer/metric (for evaluation)
        agents: [data_engineer, modeler, analyst]
        tool_manager: Tool manager (needs code_interpreter)
        task: Task type ("dsbench")
        metadata: Task metadata (train_file, test_file, etc.)
        **kwargs: Includes workflow_args

    Returns:
        {
            "prompt": str,
            "label": str,
            "trajectory": List[Dict],
            "final_reward": float
        }
    """
    workflow_args = kwargs.get("workflow_args", {})

    # ========================================================================
    # Debug Logging: Extract iteration and trajectory_id
    # ========================================================================
    # iteration: comes from training loop (defaults to 0)
    # trajectory_id: unique ID for this trajectory (use task hash or generate)
    import hashlib
    iteration = kwargs.get("iteration", workflow_args.get("iteration", 0))
    trajectory_id = kwargs.get("trajectory_id", "")
    if not trajectory_id:
        # Generate unique ID from prompt hash + timestamp
        import time
        hash_input = f"{prompt[:100]}_{time.time()}"
        trajectory_id = hashlib.md5(hash_input.encode()).hexdigest()[:16]

    # ========================================================================
    # FIX BUG #1: Deserialize metadata from JSON string
    # ========================================================================
    # The metadata arrives as a JSON string from prompts_loader.py but we need
    # a dict to access fields like train_file, test_file, etc.
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata JSON: {e}")
            metadata = {}

    # Ensure metadata is never None
    if metadata is None:
        metadata = {}

    # ========================================================================
    # FIX BUG #3: Prepare files for SandboxFusion upload
    # ========================================================================
    # DSBench tasks require CSV files (train.csv, test.csv) to be available
    # in the sandbox. We read them from the host filesystem and prepare them
    # for upload with each code execution.
    task_files = prepare_task_files(metadata)
    logger.info(f"Prepared {len(task_files)} files for sandbox: {list(task_files.keys())}")

    # Get coach
    coach = get_coach(workflow_args)

    # Validate tool_manager
    if tool_manager is None:
        raise ValueError("DSBench workflow requires tool_manager for code execution")

    # Validate agents
    if len(agents) < 3:
        raise ValueError(f"DSBench requires 3 agents, but only {len(agents)} provided")

    data_engineer_agent = agents[0]
    modeler_agent = agents[1]
    analyst_agent = agents[2]

    # Get max_turns for each agent
    data_engineer_max_turns = workflow_args.get("data_engineer_max_turns", 3)
    modeler_max_turns = workflow_args.get("modeler_max_turns", 5)
    analyst_max_turns = workflow_args.get("analyst_max_turns", 2)

    logger.info(f"Starting DSBench workflow: {metadata.get('task_id', 'unknown')}")
    logger.info(f"Max turns: DataEngineer={data_engineer_max_turns}, Modeler={modeler_max_turns}, Analyst={analyst_max_turns}")

    # ========================================================================
    # Use cleaned task description instead of raw Kaggle prompt
    # ========================================================================
    # The raw Kaggle prompts contain distracting content (Discord links, \boxed{}
    # instructions, LaTeX formulas, promotional content) that confuses the agents.
    # We use cleaned versions that keep only useful information.
    task_id = metadata.get("task_id", "")
    cleaned_prompt = get_cleaned_prompt(task_id)
    if cleaned_prompt:
        logger.info(f"Using cleaned prompt for task: {task_id}")
        task_description = cleaned_prompt
    else:
        logger.warning(f"No cleaned prompt found for task: {task_id}, using raw prompt")
        task_description = prompt

    trajectory = []
    turn_id_counter = 0

    # Extract task metadata (modeling tasks only)
    train_file = metadata.get("train_file", "train.csv")
    test_file = metadata.get("test_file", "test.csv")

    # ========================================================================
    # Agent 0: Data Engineer (EDA + Preprocessing)
    # ========================================================================

    data_engineer_prompt = """You are a Data Engineer in a data science pipeline.

**⚠️ CRITICAL: Keep your <think> reasoning BRIEF (under 1000 words). Then IMMEDIATELY write executable code in a ```python block. Do NOT describe code in thinking - just write it!**

**Task Description:**
{task_description}

## CRITICAL FILE SYSTEM OBSERVATION
**Files currently available in your workspace:**
{available_files_list}

NOTE: The file names shown above are the ACTUAL files you can access. Always check this list!

**Your Responsibilities:**
1. **Load Data**: Read training data from 'train.csv' and test data from 'test.csv'
2. **Exploratory Data Analysis (EDA)**:
   - Inspect schema (dtypes, shape, missing values)
   - Analyze distributions, correlations
   - Identify data quality issues
3. **Data Preprocessing**:
   - Handle missing values (imputation/removal)
   - Encode categorical variables (one-hot, label encoding)
   - Scale numerical features (StandardScaler, MinMaxScaler)
   - Engineer new features based on domain insights
4. **CRITICAL: Save ALL Required Artifacts**:
   - `X_train.pkl` - preprocessed training features (REQUIRED)
   - `y_train.pkl` - training labels (REQUIRED)
   - `X_test.pkl` - preprocessed TEST features (CRITICAL! The Analyst NEEDS this to make predictions!)
   - `scaler.pkl` or similar - fitted transformers
5. **Output Summary**: Print what files you saved and preprocessing details

## CRITICAL: You MUST save X_test.pkl!
The Analyst agent downstream will load X_test.pkl to generate predictions.
If you don't save X_test.pkl, the entire pipeline will FAIL.

**Example code (file names are ILLUSTRATIVE - check your actual task):**
```python
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Identify target and features (ADAPT THESE TO YOUR ACTUAL DATA!)
# Example: target_col = 'target'  # <-- CHANGE THIS based on your data
# Example: feature_cols = [col for col in train_df.columns if col != target_col and col != 'id']

# Preprocessing (fit on train, transform both)
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_test = scaler.transform(test_df[feature_cols])  # MUST ALSO TRANSFORM TEST!
y_train = train_df[target_col].values

# SAVE ALL ARTIFACTS - especially X_test.pkl!
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(X_train, open('X_train.pkl', 'wb'))
pickle.dump(y_train, open('y_train.pkl', 'wb'))
pickle.dump(X_test, open('X_test.pkl', 'wb'))  # CRITICAL!
pickle.dump(test_df, open('test_df.pkl', 'wb'))  # Save original test for IDs

print("Saved files: scaler.pkl, X_train.pkl, y_train.pkl, X_test.pkl, test_df.pkl")
print(f"X_train shape: {{X_train.shape}}")
print(f"X_test shape: {{X_test.shape}}")
```

**IMPORTANT OUTPUT FORMAT:**
At the end of your code, print:
---PREPROCESSING_SUMMARY---
Dataset shape: (rows, cols)
Target variable: <name>
Features: [list of feature names]
Missing values: <summary>
Saved files: [list ALL .pkl files you saved, including X_test.pkl]
---END_SUMMARY---

**Constraints:**
- Apply SAME transformations to both train and test data
- Do NOT use test data for fitting (no data leakage!)
- Make preprocessing deterministic (set random_state where applicable)
- ALWAYS save X_test.pkl - the downstream Analyst needs it!
"""

    logger.info(f"Agent 0 (Data Engineer) starting (max_turns={data_engineer_max_turns})...")

    # Prepare file list for Data Engineer (only task files at this point)
    data_engineer_files_list = list(task_files.keys()) if task_files else ["(no files uploaded yet)"]
    data_engineer_files_str = "\n".join(f"  - {f}" for f in data_engineer_files_list)

    data_engineer_input = apply_template_with_tokenizer(
        data_engineer_agent["tokenizer"],
        data_engineer_prompt.format(
            task_description=task_description,
            available_files_list=data_engineer_files_str
        )
    )

    # Use glob patterns to fetch any files the agent creates
    # This allows flexible file naming - agent can save any .pkl, .joblib, .csv files
    data_engineer_fetch_files = [
        "*.pkl",       # Any pickle files (preprocessor, scaler, encoder, etc.)
        "*.joblib",    # Joblib serialized files
        "*.npy",       # NumPy arrays
        "*.npz",       # Compressed NumPy arrays
    ]

    data_engineer_result = await run_agent_with_per_action_coaching(
        agent=data_engineer_agent,
        initial_prompt=data_engineer_input,
        coach=coach,
        problem=prompt,
        agent_role="data_engineer",
        tool_manager=tool_manager,
        max_turns=data_engineer_max_turns,
        context={
            "task": task,
            "label": label,
            "agent_name": data_engineer_agent["agent_id"],
            "files_available": data_engineer_files_str,
            "files_from_previous_agents": "(This is the first agent - no previous files)",
        },
        metadata=metadata,
        label=label,
        workflow_args=workflow_args,
        task_files=task_files,  # Pass files for code execution
        fetch_files=data_engineer_fetch_files,  # Retrieve saved artifacts
        iteration=iteration,
        trajectory_id=trajectory_id
    )

    # Get accumulated files from Data Engineer (includes original task_files + any saved files)
    accumulated_files = data_engineer_result.get("accumulated_files", task_files or {})

    # Convert actions to trajectory
    for action_idx, (action, reward) in enumerate(zip(data_engineer_result["actions"], data_engineer_result["rewards"])):
        tool_info = {}
        for tool_result in data_engineer_result["tool_results"]:
            if tool_result["turn_idx"] == action_idx:
                tool_info = {
                    "tools_used": tool_result["tools_used"],
                    "observation": tool_result["observation"]
                }
                break

        trajectory.append({
            "turn_id": turn_id_counter,
            "agent_index": 0,
            "agent_name": data_engineer_agent["agent_id"],
            "agent_role": data_engineer_agent["agent_role"],
            "agent_input": data_engineer_input if action_idx == 0 else data_engineer_result["observation"][action_idx],
            "agent_output": action,
            "agent_reward": reward,
            "metadata": {
                "action_index": action_idx,
                "total_actions": len(data_engineer_result["actions"]),
                **tool_info
            }
        })
        turn_id_counter += 1

    data_engineer_output_raw = data_engineer_result["final_output"]
    # Strip <think>...</think> blocks to save context for subsequent agents
    data_engineer_output = strip_thinking_blocks(data_engineer_output_raw)

    logger.info(f"Agent 0 complete: {len(data_engineer_result['actions'])} actions, avg reward={sum(data_engineer_result['rewards'])/len(data_engineer_result['rewards']):.3f}")
    logger.info(f"  Output stripped: {len(data_engineer_output_raw)} -> {len(data_engineer_output)} chars")

    # ========================================================================
    # Agent 1: Modeler (Algorithm Selection + Training)
    # ========================================================================

    modeler_prompt = """You are a Modeler in a data science pipeline.

**⚠️ CRITICAL: Keep your <think> reasoning BRIEF (under 1000 words). Then IMMEDIATELY write executable code in a ```python block. Do NOT describe code in thinking - just write it!**

**Task Description:**
{task_description}

## CRITICAL FILE SYSTEM OBSERVATION
**Files currently available in your workspace (saved by previous agents):**
{available_files_list}

NOTE: The file names shown above are the ACTUAL files you can load. You MUST use these exact file names!
Do NOT assume files exist if they are not listed above.

**Preprocessing Summary from Data Engineer:**
{preprocessing_summary}

**Your Responsibilities:**
1. **Check Available Files FIRST**: Look at the file list above and verify which files exist
2. **Load Data**: Load the preprocessed data saved by Data Engineer
   - ONLY load files that are listed in "Files currently available" above!
3. **Algorithm Selection**:
   - Analyze the problem type (classification/regression)
   - Consider dataset characteristics (size, features, target distribution)
   - Select appropriate algorithms (default recommendations: RandomForest, XGBoost, LightGBM)
4. **Model Training**:
   - Train selected models with cross-validation
   - Tune hyperparameters (grid search or random search)
   - Evaluate performance (accuracy, F1, RMSE, etc.)
5. **Model Selection**: Choose best model based on validation metrics
6. **CRITICAL: Save Model**: Save the trained model as `model.pkl` - the Analyst needs this!
7. **Output**: Print what files you saved and performance metrics

**Example code (file names are ILLUSTRATIVE - use ACTUAL files from the list above!):**
```python
import pickle
import os

# FIRST: Check what files are available
print("Available files:", os.listdir('.'))

# Load preprocessed data - USE ACTUAL FILE NAMES FROM THE LIST ABOVE!
# Example: If the list shows 'X_train.pkl', load it:
X_train = pickle.load(open('X_train.pkl', 'rb'))  # <-- Use actual filename!
y_train = pickle.load(open('y_train.pkl', 'rb'))  # <-- Use actual filename!

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# CRITICAL: Save trained model - Analyst needs this!
pickle.dump(model, open('model.pkl', 'wb'))
print("Saved files: model.pkl")
```

**IMPORTANT OUTPUT FORMAT:**
At the end, print:
---MODEL_SUMMARY---
Problem type: classification/regression
Best model: <model name>
Validation metric: <metric name> = <value>
Saved files: [list the .pkl files you saved, especially model.pkl]
---END_SUMMARY---

**Available Libraries:**
- sklearn (RandomForest, XGBoost, LogisticRegression, etc.)
- pandas, numpy
- lightgbm (if installed)

**Constraints:**
- Check available files BEFORE trying to load them
- Set random_state for reproducibility
- Use cross-validation for model selection
- Report validation metrics honestly
- ALWAYS save your trained model to model.pkl - the Analyst needs it!
"""

    logger.info(f"Agent 1 (Modeler) starting (max_turns={modeler_max_turns})...")

    # Get list of files saved by Data Engineer (excluding original task files)
    original_files = set(task_files.keys()) if task_files else set()
    data_engineer_saved_files = [f for f in accumulated_files.keys() if f not in original_files]

    # Prepare comprehensive file list for Modeler (task files + Data Engineer saved files)
    all_modeler_files = list(accumulated_files.keys())
    modeler_files_str = "\n".join(f"  - {f}" for f in all_modeler_files) if all_modeler_files else "  (no files available)"

    modeler_input = apply_template_with_tokenizer(
        modeler_agent["tokenizer"],
        modeler_prompt.format(
            task_description=task_description,
            available_files_list=modeler_files_str,
            preprocessing_summary=data_engineer_output
        )
    )

    # Use glob patterns to fetch any files the Modeler creates
    modeler_fetch_files = [
        "*.pkl",       # Any pickle files (model, predictions, etc.)
        "*.joblib",    # Joblib serialized files
        "*.npy",       # NumPy arrays
        "*.npz",       # Compressed NumPy arrays
    ]

    modeler_result = await run_agent_with_per_action_coaching(
        agent=modeler_agent,
        initial_prompt=modeler_input,
        coach=coach,
        problem=prompt,
        agent_role="modeler",
        tool_manager=tool_manager,
        max_turns=modeler_max_turns,
        context={
            "task": task,
            "label": label,
            "agent_name": modeler_agent["agent_id"],
            "files_available": modeler_files_str,
            "files_from_previous_agents": f"Data Engineer saved: {', '.join(data_engineer_saved_files) if data_engineer_saved_files else 'NO FILES SAVED'}",
        },
        metadata=metadata,
        label=label,
        workflow_args=workflow_args,
        task_files=accumulated_files,  # Pass accumulated files from Data Engineer
        fetch_files=modeler_fetch_files,  # Retrieve model artifacts
        iteration=iteration,
        trajectory_id=trajectory_id
    )

    # Get accumulated files from Modeler (includes Data Engineer files + model)
    accumulated_files = modeler_result.get("accumulated_files", accumulated_files)

    # Track files saved by Modeler (for coach context)
    files_before_modeler = set(task_files.keys()) if task_files else set()
    files_before_modeler.update(data_engineer_saved_files)
    modeler_saved_files = [f for f in accumulated_files.keys() if f not in files_before_modeler]

    # Convert actions to trajectory
    for action_idx, (action, reward) in enumerate(zip(modeler_result["actions"], modeler_result["rewards"])):
        tool_info = {}
        for tool_result in modeler_result["tool_results"]:
            if tool_result["turn_idx"] == action_idx:
                tool_info = {
                    "tools_used": tool_result["tools_used"],
                    "observation": tool_result["observation"]
                }
                break

        trajectory.append({
            "turn_id": turn_id_counter,
            "agent_index": 1,
            "agent_name": modeler_agent["agent_id"],
            "agent_role": modeler_agent["agent_role"],
            "agent_input": modeler_input if action_idx == 0 else modeler_result["observation"][action_idx],
            "agent_output": action,
            "agent_reward": reward,
            "metadata": {
                "action_index": action_idx,
                "total_actions": len(modeler_result["actions"]),
                **tool_info
            }
        })
        turn_id_counter += 1

    modeler_output_raw = modeler_result["final_output"]
    # Strip <think>...</think> blocks to save context for subsequent agents
    modeler_output = strip_thinking_blocks(modeler_output_raw)

    logger.info(f"Agent 1 complete: {len(modeler_result['actions'])} actions, avg reward={sum(modeler_result['rewards'])/len(modeler_result['rewards']):.3f}")
    logger.info(f"  Output stripped: {len(modeler_output_raw)} -> {len(modeler_output)} chars")

    # ========================================================================
    # Agent 2: Analyst (Prediction Generation + Reporting)
    # ========================================================================

    analyst_prompt = """You are an Analyst in a data science pipeline.

**⚠️ CRITICAL: Keep your <think> reasoning BRIEF (under 1000 words). Then IMMEDIATELY write executable code in a ```python block. Do NOT describe code in thinking - just write it!**

**Task Description:**
{task_description}

## CRITICAL FILE SYSTEM OBSERVATION
**Files currently available in your workspace (saved by previous agents):**
{available_files_list}

NOTE: The file names shown above are the ACTUAL files you can load. You MUST use these exact file names!
Do NOT assume files exist if they are not listed above. If a required file is missing, you cannot proceed.

**Preprocessing Summary from Data Engineer:**
{preprocessing_summary}

**Model Summary from Modeler:**
{model_summary}

**Your Responsibilities:**
1. **CHECK AVAILABLE FILES FIRST**: Look at the file list above CAREFULLY!
   - You NEED `model.pkl` (or similar) to make predictions
   - You NEED `X_test.pkl` (or similar) with preprocessed test features
   - If these files are MISSING, report the issue - you cannot create submission.csv without them!
2. **Load Saved Artifacts**: Load the model and preprocessed test data from files listed above
3. **Generate Predictions**:
   - Make predictions on test set using the loaded model
   - Verify prediction format matches requirements
4. **Get IDs from test.csv**: Load original test.csv to get the ID column for submission
5. **Create Submission**:
   - Save predictions to 'submission.csv'
   - MUST include correct ID column (from test.csv) and prediction column
   - Follow competition format (check task description for column names)
6. **Quality Checks**:
   - Validate no missing predictions
   - Check value ranges (e.g., probabilities in [0,1])
   - Verify column names and order

## CRITICAL: Check file availability before loading!
If `X_test.pkl` or `model.pkl` is NOT in the available files list above, you CANNOT create submission.csv.
In that case, report which files are missing and what you would need.

**Example code (file names are ILLUSTRATIVE - use ACTUAL files from the list above!):**
```python
import pickle
import pandas as pd
import os

# FIRST: Check what files are available
print("Available files:", os.listdir('.'))

# Load model - USE ACTUAL FILENAME FROM THE LIST ABOVE!
# If 'model.pkl' is not in the list, you cannot proceed!
model = pickle.load(open('model.pkl', 'rb'))  # <-- Use actual filename from list!

# Load preprocessed test data - USE ACTUAL FILENAME FROM THE LIST ABOVE!
# If 'X_test.pkl' is not in the list, you cannot proceed!
X_test = pickle.load(open('X_test.pkl', 'rb'))  # <-- Use actual filename from list!

# Load original test.csv to get IDs
test_df = pd.read_csv('test.csv')
# Get ID column (ADAPT TO YOUR ACTUAL DATA - check column names!)
# Example: id_col = test_df['id']  # or test_df.iloc[:, 0] for first column

# Generate predictions
predictions = model.predict(X_test)

# Create submission with CORRECT format (check task description for column names!)
submission = pd.DataFrame({{
    'id': test_df.iloc[:, 0],  # First column as ID (ADAPT TO YOUR DATA!)
    'target': predictions       # Prediction column (ADAPT NAME TO TASK REQUIREMENT!)
}})
submission.to_csv('submission.csv', index=False)
print(f"Saved {{len(predictions)}} predictions to submission.csv")
print(submission.head())
```

**IMPORTANT: Your main output is the submission.csv file!**
The file will be automatically evaluated against ground truth.
After saving, print:
- Number of predictions saved
- First few rows of the submission
- Column names used

**Constraints:**
- CHECK THE FILE LIST ABOVE before trying to load any file
- If required files are missing, report which ones and why you cannot proceed
- Load the model from model.pkl (or actual filename from list)
- Load preprocessed test data from X_test.pkl (or actual filename from list)
- Get IDs from original test.csv (not from pkl files - those may be numpy arrays!)
- Ensure predictions match test set size
- Save predictions to submission.csv (this is what gets evaluated!)
- Follow submission format strictly (check task description for column names)
"""

    logger.info(f"Agent 2 (Analyst) starting (max_turns={analyst_max_turns})...")

    # Get list of all files saved by previous agents (excluding original task files)
    all_saved_files = [f for f in accumulated_files.keys() if f not in original_files]

    # Prepare comprehensive file list for Analyst (task files + all saved files)
    all_analyst_files = list(accumulated_files.keys())
    analyst_files_str = "\n".join(f"  - {f}" for f in all_analyst_files) if all_analyst_files else "  (no files available)"

    # Check for critical missing files and add warnings
    missing_critical_files = []
    if not any('x_test' in f.lower() or 'xtest' in f.lower() for f in all_analyst_files):
        missing_critical_files.append("X_test.pkl (preprocessed test features)")
    if not any('model' in f.lower() for f in all_analyst_files):
        missing_critical_files.append("model.pkl (trained model)")

    if missing_critical_files:
        analyst_files_str += "\n\n  ⚠️ WARNING: CRITICAL FILES MISSING:\n"
        analyst_files_str += "\n".join(f"    - {f}" for f in missing_critical_files)
        analyst_files_str += "\n  You cannot create submission.csv without these files!"

    analyst_input = apply_template_with_tokenizer(
        analyst_agent["tokenizer"],
        analyst_prompt.format(
            task_description=task_description,
            available_files_list=analyst_files_str,
            preprocessing_summary=data_engineer_output,
            model_summary=modeler_output
        )
    )

    # Analyst saves submission.csv - we need to fetch it for evaluation
    analyst_fetch_files = [
        "submission.csv",    # Main output for evaluation
        "predictions.csv",   # Alternative name
    ]

    analyst_result = await run_agent_with_per_action_coaching(
        agent=analyst_agent,
        initial_prompt=analyst_input,
        coach=coach,
        problem=prompt,
        agent_role="analyst",
        tool_manager=tool_manager,
        max_turns=analyst_max_turns,
        context={
            "task": task,
            "label": label,
            "agent_name": analyst_agent["agent_id"],
            "previous_outputs": [data_engineer_output, modeler_output],
            "files_available": analyst_files_str,
            "files_from_previous_agents": (
                f"Data Engineer saved: {', '.join(data_engineer_saved_files) if data_engineer_saved_files else 'NO FILES'}\n"
                f"Modeler saved: {', '.join(modeler_saved_files) if modeler_saved_files else 'NO FILES'}"
            ),
        },
        metadata=metadata,
        label=label,
        workflow_args=workflow_args,
        task_files=accumulated_files,  # Pass all accumulated files (data + model)
        fetch_files=analyst_fetch_files,  # Fetch submission.csv for evaluation
        iteration=iteration,
        trajectory_id=trajectory_id
    )

    # ========================================================================
    # Debug Logging: Save trajectory to JSON file
    # ========================================================================
    from marti.worlds.workflows.coach_utils import _debug_logger
    _debug_logger.save_trajectory(iteration, trajectory_id)

    # Get submission.csv from accumulated files for ground truth evaluation
    analyst_accumulated = analyst_result.get("accumulated_files", {})

    # Convert actions to trajectory
    for action_idx, (action, reward) in enumerate(zip(analyst_result["actions"], analyst_result["rewards"])):
        tool_info = {}
        for tool_result in analyst_result["tool_results"]:
            if tool_result["turn_idx"] == action_idx:
                tool_info = {
                    "tools_used": tool_result["tools_used"],
                    "observation": tool_result["observation"]
                }
                break

        trajectory.append({
            "turn_id": turn_id_counter,
            "agent_index": 2,
            "agent_name": analyst_agent["agent_id"],
            "agent_role": analyst_agent["agent_role"],
            "agent_input": analyst_input if action_idx == 0 else analyst_result["observation"][action_idx],
            "agent_output": action,
            "agent_reward": reward,
            "metadata": {
                "action_index": action_idx,
                "total_actions": len(analyst_result["actions"]),
                **tool_info
            }
        })
        turn_id_counter += 1

    logger.info(f"Agent 2 complete: {len(analyst_result['actions'])} actions, avg reward={sum(analyst_result['rewards'])/len(analyst_result['rewards']):.3f}")

    # ========================================================================
    # Finalize trajectory
    # ========================================================================

    final_reward = analyst_result["rewards"][-1]
    outcome_metrics = analyst_result.get("outcome_metrics", None)

    total_actions = len(data_engineer_result["actions"]) + len(modeler_result["actions"]) + len(analyst_result["actions"])
    logger.info(f"Workflow complete: {total_actions} total actions, {len(trajectory)} turns")
    if outcome_metrics is not None:
        logger.info(f"  Ground truth metrics (logged, not used for training reward): {outcome_metrics}")

    # Log files that were accumulated during this trajectory
    if all_saved_files:
        logger.info(f"  Files saved by agents: {all_saved_files}")

    # ========================================================================
    # Cleanup: Clear accumulated files to free memory
    # ========================================================================
    # Files are only in memory (base64 encoded), clearing the dict frees the memory
    # The sandbox is stateless, so files don't persist between trajectories anyway
    accumulated_files.clear()
    logger.debug("  Cleaned up accumulated files from memory")

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "final_reward": final_reward,
        "outcome_metrics": outcome_metrics  # Dict of all computed metrics for logging
    }
