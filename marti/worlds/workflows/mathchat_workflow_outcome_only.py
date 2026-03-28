"""
MathChat Workflow with Outcome-Only Reward (Baseline)

Same 3-agent pipeline as mathchat_workflow_with_coach.py, but WITHOUT per-action
coach evaluation. All agents receive the same sparse binary reward based on whether
the Verifier's final answer matches ground truth.

This serves as an ablation baseline to isolate the contribution of dense per-action
process rewards (MAPPA) vs sparse outcome-only rewards.

Reward assignment:
- Correct answer: ALL actions get reward 10.0
- Incorrect answer: ALL actions get reward 0.0
- (Processor normalizes 0-10 -> 0.0-1.0)

Workflow: Generator (Problem Solver) -> Coder (Code Executor) -> Refiner (Verifier)
"""

import os
from typing import Dict, List, Any, Optional
from marti.helpers.logging import init_logger
from marti.worlds.workflows.utils import apply_template_with_tokenizer
from marti.worlds.workflows.mathchat_prompts import GENERATOR_PROMPT, CODER_PROMPT, REFINER_PROMPT
from marti.worlds.steps.mcp_step import step_with_tools
from marti.verifiers.qwen.qwen_eval import qwen_reward_fn

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "INFO"))


async def _run_agent(
    agent: Dict[str, Any],
    initial_prompt: str,
    tool_manager: Optional[Any] = None,
    max_turns: int = 1,
    metadata: Optional[Dict] = None,
    task_files: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run an agent (generation + optional tool use) WITHOUT coach evaluation.

    Returns:
        {
            "observation": List[str],
            "actions": List[str],
            "final_output": str,
            "tool_results": List[Dict],
        }
    """
    observation = [initial_prompt]
    actions = []
    tool_results = []
    accumulated_files = dict(task_files) if task_files else {}

    if tool_manager is None:
        # Single generation (no tools)
        response = await agent["llm"].generate_async.remote(
            initial_prompt,
            agent["sampling_params"]
        )
        action = response.outputs[0].text.strip()
        actions.append(action)
        observation.append(action)

        return {
            "observation": observation,
            "actions": actions,
            "final_output": action,
            "tool_results": [],
            "accumulated_files": accumulated_files,
        }

    # Multi-turn with tool execution
    for turn_idx in range(max_turns):
        observation_text = "".join(observation)

        response = await agent["llm"].generate_async.remote(
            observation_text,
            agent["sampling_params"]
        )
        action = response.outputs[0].text.strip()
        actions.append(action)

        # Execute tools
        step_result = await step_with_tools(
            observation,
            action,
            tool_manager,
            metadata=metadata,
            task_files=accumulated_files,
        )

        if not step_result["done"]:
            tool_observation = step_result["next_observation"][-1]
            tool_results.append({
                "turn_idx": turn_idx,
                "tools_used": step_result.get("extra_logs", {}).get("tools_used", {}),
                "observation": tool_observation,
            })
            if "fetched_files" in step_result:
                accumulated_files.update(step_result["fetched_files"])

        observation = step_result["next_observation"]

        if step_result["done"]:
            break

    final_output = actions[-1] if actions else ""

    return {
        "observation": observation,
        "actions": actions,
        "final_output": final_output,
        "tool_results": tool_results,
        "accumulated_files": accumulated_files,
    }


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
    MathChat workflow with outcome-only (sparse) reward.

    Same agent pipeline as the coach version, but rewards are assigned AFTER
    the full trajectory completes, based solely on final answer correctness.
    All actions receive the same binary reward (10.0 or 0.0).
    """
    workflow_args = kwargs.get("workflow_args", {})

    if tool_manager is None:
        raise ValueError("MathChat workflow requires tool_manager (for code execution)")
    if len(agents) < 3:
        raise ValueError(f"MathChat requires 3 agents, but only {len(agents)} provided")

    generator_agent = agents[0]
    coder_agent = agents[1]
    refiner_agent = agents[2]

    generator_max_turns = workflow_args.get("generator_max_turns", 1)
    coder_max_turns = workflow_args.get("coder_max_turns", 1)
    refiner_max_turns = workflow_args.get("refiner_max_turns", 1)

    logger.info(f"Starting MathChat OUTCOME-ONLY workflow: {prompt[:100]}...")

    trajectory = []

    # ========================================================================
    # Agent 0: Generator (Problem Solver)
    # ========================================================================

    generator_input = apply_template_with_tokenizer(
        generator_agent["tokenizer"],
        GENERATOR_PROMPT.format(problem=prompt)
    )

    generator_result = await _run_agent(
        agent=generator_agent,
        initial_prompt=generator_input,
        tool_manager=None,
        max_turns=generator_max_turns,
        metadata=metadata,
    )

    # Record actions with placeholder rewards (backfilled after correctness check)
    for action_idx, action in enumerate(generator_result["actions"]):
        trajectory.append({
            "turn_id": len(trajectory),
            "agent_index": 0,
            "agent_name": generator_agent["agent_id"],
            "agent_role": generator_agent["agent_role"],
            "agent_input": generator_input if action_idx == 0 else generator_result["observation"][action_idx],
            "agent_output": action,
            "agent_reward": 0.0,
            "metadata": {
                "action_index": action_idx,
                "total_actions": len(generator_result["actions"]),
            }
        })

    generator_output = generator_result["final_output"]
    logger.info(f"Agent 0 complete: {len(generator_result['actions'])} actions")

    # ========================================================================
    # Agent 1: Coder (Code Executor)
    # ========================================================================

    coder_input = apply_template_with_tokenizer(
        coder_agent["tokenizer"],
        CODER_PROMPT.format(problem=prompt, solution=generator_output)
    )

    coder_result = await _run_agent(
        agent=coder_agent,
        initial_prompt=coder_input,
        tool_manager=tool_manager,
        max_turns=coder_max_turns,
        metadata=metadata,
    )

    for action_idx, action in enumerate(coder_result["actions"]):
        tool_info = {}
        for tool_result in coder_result["tool_results"]:
            if tool_result["turn_idx"] == action_idx:
                tool_info = {
                    "tools_used": tool_result["tools_used"],
                    "observation": tool_result["observation"],
                }
                break

        trajectory.append({
            "turn_id": len(trajectory),
            "agent_index": 1,
            "agent_name": coder_agent["agent_id"],
            "agent_role": coder_agent["agent_role"],
            "agent_input": coder_input if action_idx == 0 else coder_result["observation"][action_idx],
            "agent_output": action,
            "agent_reward": 0.0,
            "metadata": {
                "action_index": action_idx,
                "total_actions": len(coder_result["actions"]),
                **tool_info,
            }
        })

    coder_output = coder_result["final_output"]

    if coder_result["tool_results"]:
        last_tool_result = coder_result["tool_results"][-1]["observation"]
        execution = f"{coder_output}\n\n{last_tool_result}"
    else:
        execution = coder_output

    logger.info(f"Agent 1 complete: {len(coder_result['actions'])} actions")

    # ========================================================================
    # Agent 2: Refiner (Verifier)
    # ========================================================================

    refiner_input = apply_template_with_tokenizer(
        refiner_agent["tokenizer"],
        REFINER_PROMPT.format(problem=prompt, execution=execution)
    )

    refiner_result = await _run_agent(
        agent=refiner_agent,
        initial_prompt=refiner_input,
        tool_manager=None,
        max_turns=refiner_max_turns,
        metadata=metadata,
    )

    for action_idx, action in enumerate(refiner_result["actions"]):
        trajectory.append({
            "turn_id": len(trajectory),
            "agent_index": 2,
            "agent_name": refiner_agent["agent_id"],
            "agent_role": refiner_agent["agent_role"],
            "agent_input": refiner_input if action_idx == 0 else refiner_result["observation"][action_idx],
            "agent_output": action,
            "agent_reward": 0.0,
            "metadata": {
                "action_index": action_idx,
                "total_actions": len(refiner_result["actions"]),
            }
        })

    logger.info(f"Agent 2 complete: {len(refiner_result['actions'])} actions")

    # ========================================================================
    # Check correctness and assign OUTCOME-ONLY reward to ALL actions
    # ========================================================================

    verifier_output = refiner_result["final_output"]
    is_correct = qwen_reward_fn(verifier_output, label, task=task)

    # Outcome reward on 0-10 scale (processor divides by 10)
    outcome_reward = 10.0 if is_correct == 1.0 else 0.0

    for turn in trajectory:
        turn["agent_reward"] = outcome_reward

    logger.info(
        f"Workflow complete: {len(trajectory)} actions, "
        f"correct={is_correct == 1.0}, outcome_reward={outcome_reward}"
    )

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "final_reward": outcome_reward,
        "outcome_score": is_correct,
    }
