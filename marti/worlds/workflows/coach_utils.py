"""
Coach utility functions: Support per-action coach evaluation

This module provides reusable functions for integrating external coach evaluation in workflows.

Training uses PROCESS reward only (coach's 0-10 score).
Outcome metrics (from ground truth comparison) are logged for performance tracking,
but NOT used in reward computation.
"""

import asyncio
import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from marti.helpers.logging import init_logger
from marti.worlds.steps.mcp_step import step_with_tools

logger = init_logger(__name__)


# ============================================================================
# Debug JSON Logging for Single Trajectory
# ============================================================================

class TrajectoryDebugLogger:
    """
    Logs detailed agent/coach I/O for ONE trajectory per iteration.
    Uses file-based locking to coordinate across Ray workers (separate processes).
    """

    def __init__(self):
        self._log_dir = os.environ.get("MARTI_DEBUG_LOG_DIR", "logs/debug_trajectories")
        self._enabled = os.environ.get("MARTI_DEBUG_LOGGING", "1") == "1"
        self._lock_file = os.path.join(self._log_dir, ".lock")
        self._current_file = None
        self._trajectory_id = None
        os.makedirs(self._log_dir, exist_ok=True)
        logger.info(f"TrajectoryDebugLogger initialized, log_dir={self._log_dir}, enabled={self._enabled}")

    def _get_iteration_marker(self, iteration: int) -> str:
        """Get the marker file path for this iteration."""
        return os.path.join(self._log_dir, f".iter_{iteration:04d}_claimed")

    def should_log(self, iteration: int, trajectory_id: str) -> bool:
        """
        Returns True if this trajectory should be logged.
        Uses file-based locking to ensure only ONE trajectory per iteration is logged.
        """
        if not self._enabled:
            return False

        marker_file = self._get_iteration_marker(iteration)

        # Try to claim this iteration by creating the marker file atomically
        try:
            # O_CREAT | O_EXCL ensures atomic creation - fails if file exists
            fd = os.open(marker_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, trajectory_id.encode())
            os.close(fd)
            self._trajectory_id = trajectory_id
            self._current_file = os.path.join(self._log_dir, f"iter_{iteration:04d}_traj_{trajectory_id[:8]}.json")
            # Initialize the JSON file
            with open(self._current_file, 'w') as f:
                json.dump({
                    "iteration": iteration,
                    "trajectory_id": trajectory_id,
                    "timestamp": datetime.now().isoformat(),
                    "agents": []
                }, f)
            logger.info(f"[DebugLog] Iteration {iteration}: logging trajectory {trajectory_id}")
            return True
        except FileExistsError:
            # Another process already claimed this iteration
            return False
        except Exception as e:
            logger.warning(f"[DebugLog] Error claiming iteration {iteration}: {e}")
            return False

    def log_agent_turn(
        self,
        iteration: int,
        trajectory_id: str,
        agent_name: str,
        agent_role: str,
        turn_idx: int,
        agent_input: str,
        agent_output: str,
        tool_observation: Optional[str],
        coach_prompt: str,
        coach_response: str,
        coach_score: float,
        files_available: Optional[str] = None,
        files_from_previous: Optional[str] = None
    ):
        """Log a single agent turn with all I/O details - writes directly to file."""
        if not self._enabled:
            return

        # First time for this trajectory? Try to claim it
        if self._trajectory_id is None:
            if not self.should_log(iteration, trajectory_id):
                return  # Another process claimed this iteration
        elif trajectory_id != self._trajectory_id:
            return  # We're logging a different trajectory

        if not self._current_file:
            return

        turn_data = {
            "agent_name": agent_name,
            "agent_role": agent_role,
            "turn_idx": turn_idx,
            "agent_input": agent_input[:20000] if agent_input else None,
            "agent_output": agent_output[:50000] if agent_output else None,  # Increased for reasoning models
            "tool_observation": tool_observation[:10000] if tool_observation else None,
            "files_available": files_available,
            "files_from_previous": files_from_previous,
            "coach": {
                "prompt": coach_prompt[:15000] if coach_prompt else None,
                "response": coach_response[:5000] if coach_response else None,
                "score": coach_score
            }
        }

        try:
            # Read current data, append, write back
            with open(self._current_file, 'r') as f:
                data = json.load(f)
            data["agents"].append(turn_data)
            with open(self._current_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"[DebugLog] Logged turn {turn_idx} for {agent_name}")
        except Exception as e:
            logger.warning(f"[DebugLog] Failed to log turn: {e}")

    def save_trajectory(self, iteration: int, trajectory_id: str):
        """Mark trajectory as complete (file already saved incrementally)."""
        if not self._enabled or trajectory_id != self._trajectory_id:
            return
        if self._current_file and os.path.exists(self._current_file):
            logger.info(f"[DebugLog] Trajectory complete: {self._current_file}")


# Global instance (each Ray worker gets its own, but file-based locking coordinates them)
_debug_logger = TrajectoryDebugLogger()

# Weave import (optional dependency)
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False

def _weave_op_if_available(func):
    """Apply weave.op decorator if weave is available"""
    if WEAVE_AVAILABLE:
        return weave.op()(func)
    return func


@_weave_op_if_available
async def run_agent_with_per_action_coaching(
    agent: Dict[str, Any],
    initial_prompt: str,
    coach,
    problem: str,
    agent_role: str,
    tool_manager: Optional[Any] = None,
    max_turns: int = 5,
    context: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    label: Optional[str] = None,
    workflow_args: Optional[Dict] = None,
    task_files: Optional[Dict[str, str]] = None,
    process_weight: float = 1.0,  # Weight for process score (default: process-only)
    outcome_weight: float = 0.0,  # Weight for outcome score (default: disabled)
    fetch_files: Optional[List[str]] = None,
    # Debug logging parameters
    iteration: int = 0,
    trajectory_id: str = ""
) -> Dict[str, Any]:
    """
    Run an agent round with per-action coach evaluation support

    Agent can:
    1. Generate action (LLM call)
    2. Observe tool execution results
    3. Repeat until complete or max_turns reached

    After each action is generated, wait for tool results, then immediately send coach evaluation.

    IMPORTANT: Training uses PROCESS reward only (coach's 0-10 score).
    Outcome metrics are computed and logged for performance tracking, but NOT
    used in reward computation.

    Args:
        agent: Agent configuration {"llm", "tokenizer", "sampling_params", ...}
        initial_prompt: Initial input (already processed with chat template)
        coach: Coach instance
        problem: Original problem (for coach to see)
        agent_role: Agent role name
        tool_manager: Tool manager (None for single generation)
        max_turns: Maximum number of actions
        context: Additional context to pass to coach
        metadata: Metadata to pass to tools
        label: Ground truth answer (for dual-score evaluation)
        workflow_args: Workflow configuration parameters (contains enable_ground_truth_eval)
        task_files: Optional dict of filename -> base64 content for SandboxFusion
                    (backward compatible - defaults to None for workflows without files)
        process_weight: Weight for process score (default 1.0)
        outcome_weight: Weight for outcome score (default 0.0)
        fetch_files: Optional list of filenames to retrieve after code execution.
                     Enables file persistence between agents (e.g., model.pkl, preprocessor.pkl)

    Returns:
        {
            "observation": List[str],  # Complete observation history
            "actions": List[str],      # All actions
            "rewards": List[float],    # Reward for each action (process score only, 0-10)
            "final_output": str,       # Last action
            "tool_results": List[Dict], # Tool call results
            "feedbacks": List[CoachFeedback],  # Coach feedback objects
            "outcome_metrics": Optional[Dict[str, float]],  # All computed metrics for logging
            "accumulated_files": Dict[str, str]  # Files saved during execution (base64)
        }
    """
    observation = [initial_prompt]
    actions = []
    rewards = []
    tool_results = []
    feedbacks = []  # Store CoachFeedback objects
    accumulated_files = dict(task_files) if task_files else {}  # Start with input files, accumulate saved files
    outcome_metrics = None  # Will store all computed metrics for Analyst (for logging)

    # Read config parameters to decide whether to enable ground truth evaluation
    enable_ground_truth = (workflow_args or {}).get('enable_ground_truth_eval', False)

    logger.info(f"Starting agent round ({agent_role}), max_turns={max_turns}")
    if enable_ground_truth:
        logger.info(f"  Ground truth evaluation enabled (metrics logged, process reward only for training)")

    # If no tool_manager, only do single generation
    if tool_manager is None:
        logger.info("  No tool_manager, performing single generation")

        # Single generation
        response = await agent["llm"].generate_async.remote(
            initial_prompt,
            agent["sampling_params"]
        )
        action = response.outputs[0].text.strip()
        actions.append(action)
        observation.append(action)

        # Evaluation
        eval_context = context.copy() if context else {}
        # Provide default values for template compatibility
        if "tool_observation" not in eval_context:
            eval_context["tool_observation"] = "N/A (no tools used)"

        # Conditionally pass ground truth metrics for Verifier or Analyst
        ground_truth_answer = None
        if enable_ground_truth:
            # For Verifier role (legacy): use label directly
            if agent_role == "Verifier" and label:
                ground_truth_answer = label
            # For Analyst role (DSBench): compute all metrics from ground truth
            elif agent_role.lower() == "analyst" and metadata:
                try:
                    from marti.verifiers.dsbench.ground_truth_utils import evaluate_analyst_output
                    result = evaluate_analyst_output(
                        analyst_output=action,
                        task_description=problem,
                        metadata=metadata
                    )
                    if result:
                        ground_truth_answer, outcome_metrics = result  # Unpack: (formatted_str, metrics_dict)
                        logger.info(f"  Ground truth evaluation:\n{ground_truth_answer}")
                        logger.info(f"  Outcome metrics for logging: {outcome_metrics}")
                except Exception as e:
                    logger.warning(f"  Failed to evaluate ground truth: {e}")
                    ground_truth_answer = None
                    outcome_metrics = None

        feedback = await coach.evaluate(
            problem=problem,
            agent_input=initial_prompt,
            agent_output=action,
            agent_role=agent_role,
            context=eval_context,
            ground_truth=ground_truth_answer  # Coach sees all metrics as formatted string
        )

        # Store outcome_metrics in feedback for logging (if available)
        if outcome_metrics is not None:
            feedback.outcome_metrics = outcome_metrics

        # TRAINING USES PROCESS REWARD ONLY (coach's 0-10 score)
        final_reward = feedback.reward
        rewards.append(final_reward)
        feedbacks.append(feedback)

        logger.info(f"  Single generation complete, process_reward={feedback.reward:.3f}")
        if outcome_metrics:
            logger.info(f"  (Logged metrics: {list(outcome_metrics.keys())})")

        # Extract outcome_score for backward compatibility (MathChat Verifier binary correctness)
        # For DSBench: outcome_metrics is used directly, no need for outcome_score
        outcome_score = None
        if feedbacks and feedbacks[-1].outcome_score is not None:
            # From coach's ANSWER_CORRECT parsing (MathChat Verifier role only)
            outcome_score = feedbacks[-1].outcome_score

        return {
            "observation": observation,
            "actions": actions,
            "rewards": rewards,
            "final_output": action,
            "tool_results": [],
            "feedbacks": feedbacks,
            "outcome_score": outcome_score,  # Backward compat: binary correctness or derived metric
            "outcome_metrics": outcome_metrics,  # All computed metrics for logging
            "accumulated_files": accumulated_files
        }

    # Have tool_manager: multi-turn tool call loop
    logger.info("  Have tool_manager, starting tool call loop")

    all_tool_observations = []  # Accumulate all tool observations

    for turn_idx in range(max_turns):
        logger.info(f"  Turn {turn_idx+1}/{max_turns}")

        # ====================================================================
        # 1. Agent generates action
        # ====================================================================

        observation_text = "".join(observation)

        logger.info(f"    Generating action...")
        response = await agent["llm"].generate_async.remote(
            observation_text,
            agent["sampling_params"]
        )
        action = response.outputs[0].text.strip()
        actions.append(action)

        logger.info(f"    Action generation complete, length: {len(action)} characters")

        # ====================================================================
        # 2. Execute tools (if action contains tool calls)
        # ====================================================================

        logger.info(f"    Checking and executing tools...")
        step_result = await step_with_tools(
            observation,
            action,
            tool_manager,
            metadata=metadata,
            task_files=accumulated_files,  # Pass accumulated files (includes previous agents' saved files)
            fetch_files=fetch_files  # Files to retrieve after execution
        )

        # Extract tool results
        tool_observation = None
        if not step_result["done"]:
            # Has tool call
            tool_observation = step_result["next_observation"][-1]
            all_tool_observations.append(tool_observation)  # Accumulate
            tool_results.append({
                "turn_idx": turn_idx,
                "tools_used": step_result.get("extra_logs", {}).get("tools_used", {}),
                "observation": tool_observation
            })
            logger.info(f"    Tool execution complete: {list(step_result.get('extra_logs', {}).get('tools_used', {}).keys())}")

            # Accumulate fetched files from this step
            if "fetched_files" in step_result:
                accumulated_files.update(step_result["fetched_files"])
                logger.info(f"    Accumulated files: {list(step_result['fetched_files'].keys())}")
        else:
            logger.info(f"    No tool call, agent complete")

        # ====================================================================
        # 3. Call coach evaluation (after seeing tool results)
        # ====================================================================

        logger.info(f"    Calling coach evaluation...")

        # Build complete context for coach (containing all information)
        eval_context = context.copy() if context else {}

        # Add tool observations
        # INCREASED LIMITS: Coach needs to see full error messages (FileNotFoundError, etc.)
        # to properly assign blame for pipeline failures
        if all_tool_observations:
            eval_context["tool_observation"] = tool_observation[:2000] if tool_observation else "N/A"
            eval_context["all_tool_observations"] = [obs[:500] for obs in all_tool_observations]
        else:
            # No tools used - provide default value for template compatibility
            eval_context["tool_observation"] = eval_context.get("tool_observation", "N/A (no tools used)")

        # Add previous actions (actions from earlier in this round)
        if len(actions) > 0:
            eval_context["previous_actions"] = actions  # Current action not included

        # Conditionally pass ground truth metrics for Verifier or Analyst
        ground_truth_answer = None
        turn_outcome_metrics = None  # Metrics for this turn (only for Analyst)
        if enable_ground_truth:
            # For Verifier role (legacy): use label directly
            if agent_role == "Verifier" and label:
                ground_truth_answer = label
            # For Analyst role (DSBench): compute all metrics from ground truth
            elif agent_role.lower() == "analyst" and metadata:
                try:
                    from marti.verifiers.dsbench.ground_truth_utils import evaluate_analyst_output
                    result = evaluate_analyst_output(
                        analyst_output=action,
                        task_description=problem,
                        metadata=metadata,
                        submission_files=accumulated_files  # Pass fetched files for file-based evaluation
                    )
                    if result:
                        ground_truth_answer, turn_outcome_metrics = result  # Unpack: (formatted_str, metrics_dict)
                        outcome_metrics = turn_outcome_metrics  # Save for final return
                        logger.info(f"    Ground truth evaluation:\n{ground_truth_answer}")
                        logger.info(f"    Outcome metrics for logging: {turn_outcome_metrics}")
                except Exception as e:
                    logger.warning(f"    Failed to evaluate ground truth: {e}")
                    ground_truth_answer = None
                    turn_outcome_metrics = None

        feedback = await coach.evaluate(
            problem=problem,
            agent_input=observation_text,
            agent_output=action,
            agent_role=agent_role,
            context=eval_context,
            ground_truth=ground_truth_answer  # Coach sees all metrics as formatted string
        )

        # Store outcome_metrics in feedback for logging (if available)
        if turn_outcome_metrics is not None:
            feedback.outcome_metrics = turn_outcome_metrics

        # TRAINING USES PROCESS REWARD ONLY (coach's 0-10 score)
        final_reward = feedback.reward
        rewards.append(final_reward)
        feedbacks.append(feedback)

        logger.info(f"    Coach evaluation complete, process_reward={feedback.reward:.3f}")
        if turn_outcome_metrics:
            logger.info(f"    (Logged metrics: {list(turn_outcome_metrics.keys())})")

        # ====================================================================
        # Debug Logging: Log agent and coach I/O for ONE trajectory per iteration
        # ====================================================================
        if trajectory_id:
            _debug_logger.log_agent_turn(
                iteration=iteration,
                trajectory_id=trajectory_id,
                agent_name=context.get("agent_name", agent_role) if context else agent_role,
                agent_role=agent_role,
                turn_idx=turn_idx,
                agent_input=observation_text,
                agent_output=action,
                tool_observation=tool_observation,
                coach_prompt=feedback.metadata.get("judge_prompt", "") if feedback.metadata else "",
                coach_response=feedback.feedback_text or "",
                coach_score=feedback.reward,
                files_available=context.get("files_available") if context else None,
                files_from_previous=context.get("files_from_previous_agents") if context else None
            )

        # ====================================================================
        # 4. Update observation, decide whether to continue
        # ====================================================================

        observation = step_result["next_observation"]

        if step_result["done"]:
            # Diagnostic logging: why did the loop exit early?
            has_python_block = '```python' in action
            logger.warning(
                f"[CODER_EXIT_DIAG] role={agent_role} turn={turn_idx+1}/{max_turns} "
                f"has_python_block={has_python_block} action_len={len(action)} "
                f"action_preview={action[:300]!r}"
            )
            logger.info(f"    Agent complete, total {turn_idx+1} actions")
            break

    final_output = actions[-1] if actions else ""

    # Count new files saved during this agent's execution (excluding input files)
    new_files = {k: v for k, v in accumulated_files.items() if task_files is None or k not in task_files}
    if new_files:
        logger.info(f"  Agent saved {len(new_files)} new files: {list(new_files.keys())}")

    logger.info(f"  Round complete: {len(actions)} actions, {len(rewards)} rewards (process only)")
    if outcome_metrics:
        logger.info(f"  Final outcome metrics: {outcome_metrics}")

    # Extract outcome_score for backward compatibility (MathChat Verifier binary correctness)
    # For DSBench: outcome_metrics is used directly, no need for outcome_score
    outcome_score = None
    if feedbacks and feedbacks[-1].outcome_score is not None:
        # From coach's ANSWER_CORRECT parsing (MathChat Verifier role only)
        outcome_score = feedbacks[-1].outcome_score

    return {
        "observation": observation,
        "actions": actions,
        "rewards": rewards,
        "final_output": final_output,
        "tool_results": tool_results,
        "feedbacks": feedbacks,
        "outcome_score": outcome_score,  # Backward compat: MathChat binary correctness
        "outcome_metrics": outcome_metrics,  # All computed metrics for logging (DSBench Analyst)
        "accumulated_files": accumulated_files
    }
