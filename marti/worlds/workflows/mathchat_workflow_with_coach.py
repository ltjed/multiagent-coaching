"""
MathChat Workflow with External Coach Evaluation (Per-Action Coaching)

Uses run_agent_with_per_action_coaching for all agents, enabling:
1. Per-action coach evaluation (each action gets its own reward)
2. Multi-turn support for each agent (configurable via max_turns in workflow_args)
3. Automatic tool execution for coder agent

Workflow: Generator (Problem Solver) -> Coder (Code Executor) -> Refiner (Verifier)
- Generator: Generate math solution (can think multiple turns)
- Coder: Write Python code to verify, auto-execute tools (can iterate multiple turns)
- Refiner: Synthesize previous two agents' outputs, provide final answer (can synthesize multiple turns)

Coach evaluation happens inside run_agent_with_per_action_coaching for each action.
"""

import os
from typing import Dict, List, Any, Optional
from marti.helpers.logging import init_logger
from marti.worlds.workflows.utils import apply_template_with_tokenizer
from marti.worlds.workflows.coach_utils import run_agent_with_per_action_coaching
from marti.verifiers.coach import create_coach

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "INFO"))

# Weave import (optional dependency)
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False

# Global coach instance
_global_coach = None
_weave_initialized = False  # Track if Weave has been initialized in this worker


def get_coach(workflow_args: Dict) -> Any:
    """Get or create coach instance (singleton)"""
    global _global_coach, _weave_initialized

    if _global_coach is None:
        # Initialize Weave in Ray worker if enabled (one-time per worker)
        print(f"[WEAVE DEBUG] get_coach() called, workflow_args keys: {list(workflow_args.keys())}")
        print(f"[WEAVE DEBUG] use_weave = {workflow_args.get('use_weave')}, weave_project = {workflow_args.get('weave_project')}")

        if workflow_args.get("use_weave", False):
            print(f"[WEAVE DEBUG] WEAVE_AVAILABLE={WEAVE_AVAILABLE}, _weave_initialized={_weave_initialized}")
            if WEAVE_AVAILABLE and not _weave_initialized:
                weave_project = workflow_args.get("weave_project", "marti-mathchat-coach")
                print(f"[WEAVE DEBUG] Attempting to initialize in Ray worker: {weave_project}")

                # Load .env file to get WANDB_API_KEY (Ray workers don't inherit shell env)
                import os
                from dotenv import load_dotenv

                # Find and load .env file
                current_file = os.path.abspath(__file__)
                repo_root = current_file
                for _ in range(10):  # Search up to 10 levels
                    parent = os.path.dirname(repo_root)
                    if os.path.exists(os.path.join(parent, '.git')):
                        repo_root = parent
                        break
                    if parent == repo_root:
                        break
                    repo_root = parent

                env_file = os.path.join(repo_root, '.env')
                if os.path.exists(env_file):
                    print(f"[WEAVE DEBUG] Loading .env from: {env_file}")
                    load_dotenv(env_file)
                else:
                    print(f"[WEAVE DEBUG] No .env file found at: {env_file}")

                wandb_key = os.getenv("WANDB_API_KEY")
                print(f"[WEAVE DEBUG] WANDB_API_KEY after load_dotenv: {wandb_key is not None} (length: {len(wandb_key) if wandb_key else 0})")

                try:
                    # First: Login to wandb with API key (required for weave)
                    if wandb_key:
                        print(f"[WEAVE DEBUG] Logging in to wandb with API key...")
                        import wandb
                        wandb.login(key=wandb_key, relogin=True, force=True)
                        print(f"[WEAVE DEBUG] wandb.login() completed")
                    else:
                        print(f"[WEAVE DEBUG] No WANDB_API_KEY found even after load_dotenv, weave.init() will fail")

                    # Then: Initialize Weave
                    print(f"[WEAVE DEBUG] Calling weave.init(project_name='{weave_project}')...")
                    weave.init(project_name=weave_project)
                    print(f"[Weave] ✓ Successfully initialized in Ray worker: {weave_project}")
                    _weave_initialized = True
                except Exception as e:
                    print(f"[Weave] ✗ Failed to initialize in Ray worker: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
            elif not WEAVE_AVAILABLE:
                print(f"[Weave] Package not available in Ray worker!")
        else:
            print(f"[WEAVE DEBUG] use_weave is False or not set, skipping weave init")

        coach_model = workflow_args.get("coach_model", "gemini-2.5-pro")
        coach_type = workflow_args.get("coach_type", "simple")
        use_vertex_ai = workflow_args.get("use_vertex_ai", False)
        vertex_project = workflow_args.get("vertex_project", None)
        vertex_location = workflow_args.get("vertex_location", "global")
        max_output_tokens = workflow_args.get("coach_max_output_tokens", None)
        thinking_budget = workflow_args.get("coach_thinking_budget", None)

        logger.info(f"Initializing coach: {coach_type}/{coach_model}")

        # Read custom template (if any)
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
            prompt_template=prompt_template  # Pass in custom template
        )

        logger.info(f"Coach initialized successfully")

    return _global_coach


# Apply weave decorator if available
def _apply_weave_decorator(func):
    """Apply weave.op decorator if weave is available"""
    if WEAVE_AVAILABLE:
        return weave.op()(func)
    return func

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
    MathChat workflow with per-action coach evaluation

    Args:
        prompt: Math problem
        label: Correct answer (for coach context)
        agents: [generator, coder, refiner]
        tool_manager: Tool manager (requires code_interpreter)
        task: Task type ("math")
        metadata: Metadata
        **kwargs: Contains workflow_args
            - generator_max_turns: Generator max actions count (default 1)
            - coder_max_turns: Coder max actions count (default 1)
            - refiner_max_turns: Refiner max actions count (default 1)

    Returns:
        {
            "prompt": str,
            "label": str,
            "trajectory": List[Dict],  # Variable number of turns (sum of all actions)
            "final_reward": float
        }
    """
    workflow_args = kwargs.get("workflow_args", {})

    # Get coach
    coach = get_coach(workflow_args)

    # Validate tool_manager
    if tool_manager is None:
        raise ValueError("MathChat workflow requires tool_manager (for code execution)")

    # Validate number of agents
    if len(agents) < 3:
        raise ValueError(f"MathChat requires 3 agents, but only {len(agents)} provided")

    generator_agent = agents[0]
    coder_agent = agents[1]
    refiner_agent = agents[2]

    # Get max_turns for each agent from workflow_args (default to 1 for backward compatibility)
    generator_max_turns = workflow_args.get("generator_max_turns", 1)
    coder_max_turns = workflow_args.get("coder_max_turns", 1)
    refiner_max_turns = workflow_args.get("refiner_max_turns", 1)

    logger.info(f"Starting MathChat workflow: {prompt[:100]}...")
    logger.info(f"Max turns: Generator={generator_max_turns}, Coder={coder_max_turns}, Refiner={refiner_max_turns}")

    trajectory = []
    turn_id_counter = 0

    # ========================================================================
    # Agent 0: Generator (Problem Solver)
    # ========================================================================

    generator_prompt = """You are Problem Solver in a 3-agent system: Problem Solver (you) → Code Executor → Verifier.

The system succeeds only if the Verifier (final agent) outputs the correct answer. Your job is to draft a solution to the problem.

You have a strict 4k token limit (your thinking inside <think> and </think> tags also counts). Anything beyond that will be truncated.

## Problem
{problem}"""

    logger.info(f"Agent 0 (Generator/Problem Solver) starting execution (max_turns={generator_max_turns})...")

    generator_input = apply_template_with_tokenizer(
        generator_agent["tokenizer"],
        generator_prompt.format(problem=prompt)
    )

    # Use run_agent_with_per_action_coaching
    generator_result = await run_agent_with_per_action_coaching(
        agent=generator_agent,
        initial_prompt=generator_input,
        coach=coach,
        problem=prompt,
        agent_role="Problem Solver",
        tool_manager=None,  # Generator doesn't use tools
        max_turns=generator_max_turns,
        context={
            "task": task,
            "label": label,
        },
        metadata=metadata,
        label=label,  # NEW: For dual-score evaluation
        workflow_args=workflow_args  # NEW: Config parameters
    )

    # Convert actions to trajectory turns
    for action_idx, (action, reward) in enumerate(zip(generator_result["actions"], generator_result["rewards"])):
        trajectory.append({
            "turn_id": turn_id_counter,
            "agent_index": 0,
            "agent_name": generator_agent["agent_id"],
            "agent_role": generator_agent["agent_role"],
            "agent_input": generator_input if action_idx == 0 else generator_result["observation"][action_idx],
            "agent_output": action,
            "agent_reward": reward,
            "metadata": {
                "action_index": action_idx,
                "total_actions": len(generator_result["actions"])
            }
        })
        turn_id_counter += 1

    # Use final output for next agent
    generator_output = generator_result["final_output"]

    logger.info(f"Agent 0 complete: {len(generator_result['actions'])} actions, avg reward={sum(generator_result['rewards'])/len(generator_result['rewards']):.3f}")

    # ========================================================================
    # Agent 1: Coder (Code Executor)
    # ========================================================================

    coder_prompt = """You are Code Executor in a 3-agent system: Problem Solver → Code Executor (you) → Verifier.

The system succeeds only if the Verifier (final agent) outputs the correct answer. Your job is to compute/verify the solution using Python code.

You can execute Python code. Write code in ```python``` blocks and it will be automatically executed by the user on your behalf, based on which you can iterate further or output final answers.

You have a strict 4k token limit (your thinking inside <think> and </think> tags also counts). Anything beyond that will be truncated.

## Problem
{problem}

## Input from Problem Solver
{solution}"""

    logger.info(f"Agent 1 (Coder/Code Executor) starting execution (max_turns={coder_max_turns})...")

    coder_input = apply_template_with_tokenizer(
        coder_agent["tokenizer"],
        coder_prompt.format(problem=prompt, solution=generator_output)
    )

    # Use run_agent_with_per_action_coaching with tool_manager
    coder_result = await run_agent_with_per_action_coaching(
        agent=coder_agent,
        initial_prompt=coder_input,
        coach=coach,
        problem=prompt,
        agent_role="Code Executor",
        tool_manager=tool_manager,  # Coder USES tools (code execution)
        max_turns=coder_max_turns,
        context={
            "task": task,
            "label": label,
        },
        metadata=metadata,
        label=label,  # NEW: For dual-score evaluation
        workflow_args=workflow_args  # NEW: Config parameters
    )

    # Convert actions to trajectory turns
    for action_idx, (action, reward) in enumerate(zip(coder_result["actions"], coder_result["rewards"])):
        # Get tool results for this action if available
        tool_info = {}
        for tool_result in coder_result["tool_results"]:
            if tool_result["turn_idx"] == action_idx:
                tool_info = {
                    "tools_used": tool_result["tools_used"],
                    "observation": tool_result["observation"]
                }
                break

        trajectory.append({
            "turn_id": turn_id_counter,
            "agent_index": 1,
            "agent_name": coder_agent["agent_id"],
            "agent_role": coder_agent["agent_role"],
            "agent_input": coder_input if action_idx == 0 else coder_result["observation"][action_idx],
            "agent_output": action,
            "agent_reward": reward,
            "metadata": {
                "action_index": action_idx,
                "total_actions": len(coder_result["actions"]),
                **tool_info
            }
        })
        turn_id_counter += 1

    # Use final output for next agent
    coder_output = coder_result["final_output"]

    # Extract execution results for refiner (from last tool observation if available)
    if coder_result["tool_results"]:
        last_tool_result = coder_result["tool_results"][-1]["observation"]
        execution = f"{coder_output}\n\n{last_tool_result}"
    else:
        execution = coder_output

    logger.info(f"Agent 1 complete: {len(coder_result['actions'])} actions, avg reward={sum(coder_result['rewards'])/len(coder_result['rewards']):.3f}")

    # ========================================================================
    # Agent 2: Refiner (Verifier)
    # ========================================================================

    refiner_prompt = """You are Verifier, the final agent in a 3-agent system: Problem Solver → Code Executor → Verifier (you).

You are the last agent. The system succeeds only if YOU output the correct answer. Evaluate the information below and provide the final answer.

You have a strict 4k token limit (your thinking inside <think> and </think> tags also counts). Anything beyond that will be truncated. Output your final answer as: **\\boxed{{answer}}**

## Problem
{problem}

## Input from Code Executor
{execution}"""

    logger.info(f"Agent 2 (Refiner/Verifier) starting execution (max_turns={refiner_max_turns})...")

    refiner_input = apply_template_with_tokenizer(
        refiner_agent["tokenizer"],
        refiner_prompt.format(problem=prompt, execution=execution)
    )

    # Use run_agent_with_per_action_coaching
    refiner_result = await run_agent_with_per_action_coaching(
        agent=refiner_agent,
        initial_prompt=refiner_input,
        coach=coach,
        problem=prompt,
        agent_role="Verifier",  # IMPORTANT: Must match for ground truth check in coach_utils.py
        tool_manager=None,  # Refiner doesn't use tools
        max_turns=refiner_max_turns,
        context={
            "task": task,
            "label": label,
        },
        metadata=metadata,
        label=label,  # NEW: For dual-score evaluation
        workflow_args=workflow_args  # NEW: Config parameters
    )

    # Convert actions to trajectory turns
    for action_idx, (action, reward) in enumerate(zip(refiner_result["actions"], refiner_result["rewards"])):
        trajectory.append({
            "turn_id": turn_id_counter,
            "agent_index": 2,
            "agent_name": refiner_agent["agent_id"],
            "agent_role": refiner_agent["agent_role"],
            "agent_input": refiner_input if action_idx == 0 else refiner_result["observation"][action_idx],
            "agent_output": action,
            "agent_reward": reward,
            "metadata": {
                "action_index": action_idx,
                "total_actions": len(refiner_result["actions"])
            }
        })
        turn_id_counter += 1

    logger.info(f"Agent 2 complete: {len(refiner_result['actions'])} actions, avg reward={sum(refiner_result['rewards'])/len(refiner_result['rewards']):.3f}")

    # ========================================================================
    # Finalize trajectory
    # ========================================================================

    # Final reward is the last refiner's reward
    final_reward = refiner_result["rewards"][-1]

    # NEW: Extract outcome_score from Verifier's result (if present)
    outcome_score = refiner_result.get("outcome_score", None)

    total_actions = len(generator_result["actions"]) + len(coder_result["actions"]) + len(refiner_result["actions"])
    logger.info(f"Workflow complete: {total_actions} total actions, {len(trajectory)} turns")
    if outcome_score is not None:
        logger.info(f"  Ground truth evaluation: outcome_score={outcome_score}")

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "final_reward": final_reward,  # Process quality score (0.0-1.0)
        "outcome_score": outcome_score  # NEW: Outcome quality (0.0-1.0, or None)
    }
