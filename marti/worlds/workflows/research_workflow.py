"""
Research Workflow: Ideation → Experimentation → Writeup

This workflow is used for research tasks, with 3 agents executing sequentially:
1. Ideation: Generate research ideas
2. Experimentation: Design experimental plans
3. Writeup: Synthesize into a research plan

Uses an external LLM coach (via API) to evaluate each agent's output.
After each agent generates, immediately sends an API evaluation request (non-blocking),
and waits for all API responses at the end.
"""

import os
import asyncio
from typing import Dict, List, Any, Optional
from marti.helpers.logging import init_logger
from marti.worlds.workflows.utils import apply_template_with_tokenizer
from marti.verifiers.coach import create_coach

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "INFO"))

# Global coach instance (initialized on first call)
_global_coach = None


def get_coach(workflow_args: Dict) -> Any:
    """
    Get or create coach instance (singleton pattern)

    Args:
        workflow_args: Workflow configuration, including coach settings

    Returns:
        Coach instance
    """
    global _global_coach

    if _global_coach is None:
        # Read coach configuration from workflow_args
        coach_model = workflow_args.get("coach_model", "gemini-2.5-pro")
        coach_type = workflow_args.get("coach_type", "simple")
        use_vertex_ai = workflow_args.get("use_vertex_ai", False)
        vertex_project = workflow_args.get("vertex_project", None)
        vertex_location = workflow_args.get("vertex_location", "us-central1")
        prompt_template = workflow_args.get("coach_prompt_template", None)
        max_output_tokens = workflow_args.get("coach_max_output_tokens", None)

        # Create coach
        logger.info(f"Initializing coach: {coach_type}/{coach_model}")

        _global_coach = create_coach(
            model=coach_model,
            coach_type=coach_type,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            use_vertex_ai=use_vertex_ai,
            vertex_project=vertex_project,
            vertex_location=vertex_location
            prompt_template=prompt_template
        )

        logger.info(f"Coach initialized successfully")

    return _global_coach


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
    Main function for Research workflow

    Execution order: Ideation → Experimentation → Writeup
    After each agent generates, immediately sends coach evaluation request,
    then waits for all at the end

    Args:
        prompt: Research question/task
        label: Label (may be used for outcome reward)
        agents: Agent configuration list [ideation_agent, experimentation_agent, writeup_agent]
        tool_manager: Tool manager (current workflow does not need tools)
        task: Task type identifier
        metadata: Additional metadata
        **kwargs: Contains workflow_args etc.

    Returns:
        {
            "prompt": str,
            "label": str,
            "trajectory": List[Dict],  # agent turns
            "final_reward": float
        }
    """
    workflow_args = kwargs.get("workflow_args", {})

    # Get coach instance
    coach = get_coach(workflow_args)

    # Validate number of agents
    if len(agents) < 3:
        raise ValueError(f"Research workflow requires 3 agents, but only {len(agents)} provided")

    ideation_agent = agents[0]
    experimentation_agent = agents[1]
    writeup_agent = agents[2]

    logger.info(f"Starting research workflow: {prompt[:100]}...")

    trajectory = []

    # ========================================================================
    # Agent 1: Ideation (Generate research ideas)
    # ========================================================================

    ideation_prompt = f"""You are a research ideation agent. Generate creative and feasible research ideas.

**Task/Problem:**
{prompt}

**Your Task:**
1. Analyze the problem domain
2. Generate 2-3 specific research ideas or hypotheses
3. For each idea, explain the core insight and potential impact

Provide clear, structured ideas.

**Your Response:**"""

    ideation_input = apply_template_with_tokenizer(
        ideation_agent["tokenizer"],
        ideation_prompt
    )

    logger.info("Agent 0 (Ideation) starting generation...")
    ideation_response = await ideation_agent["llm"].generate_async.remote(
        ideation_input,
        ideation_agent["sampling_params"]
    )
    ideation_output = ideation_response.outputs[0].text.strip()
    logger.info(f"Agent 0 generation complete, length: {len(ideation_output)} characters")

    # Immediately send coach evaluation request (non-blocking)
    logger.info("Sending Agent 0's coach evaluation request (async)...")
    ideation_reward_task = asyncio.create_task(
        coach.evaluate(
            problem=prompt,
            agent_input=ideation_prompt,
            agent_output=ideation_output,
            agent_role="ideation",
            context={
                "task": task,
                "agent_name": ideation_agent["agent_id"]
            }
        )
    )

    # Save to trajectory (reward will be filled in later)
    ideation_turn = {
        "turn_id": 0,
        "agent_index": 0,
        "agent_name": ideation_agent["agent_id"],
        "agent_role": ideation_agent["agent_role"],
        "agent_input": ideation_input,
        "agent_output": ideation_output,
        "agent_reward": None,  # To be filled in later
        "metadata": {}
    }
    trajectory.append(ideation_turn)

    # ========================================================================
    # Agent 2: Experimentation (Design experiments based on ideation)
    # ========================================================================

    # Create prompt containing ideation output
    # Note: Can choose to include only summary instead of full text to avoid overly long context
    ideation_summary = ideation_output[:500]  # Only take first 500 characters as summary

    experimentation_prompt = f"""You are an experimentation design agent. Design concrete experiments to test research ideas.

**Task/Problem:**
{prompt}

**Proposed Ideas (summary):**
{ideation_summary}

**Your Task:**
1. Select the most promising ideas
2. Design specific experiments to test them
3. Outline methodology and success criteria

Provide a clear experimental design.

**Your Response:**"""

    experimentation_input = apply_template_with_tokenizer(
        experimentation_agent["tokenizer"],
        experimentation_prompt
    )

    logger.info("Agent 1 (Experimentation) starting generation...")
    experimentation_response = await experimentation_agent["llm"].generate_async.remote(
        experimentation_input,
        experimentation_agent["sampling_params"]
    )
    experimentation_output = experimentation_response.outputs[0].text.strip()
    logger.info(f"Agent 1 generation complete, length: {len(experimentation_output)} characters")

    # Immediately send coach evaluation request (non-blocking)
    logger.info("Sending Agent 1's coach evaluation request (async)...")
    experimentation_reward_task = asyncio.create_task(
        coach.evaluate(
            problem=prompt,
            agent_input=experimentation_prompt,
            agent_output=experimentation_output,
            agent_role="experimentation",
            context={
                "task": task,
                "agent_name": experimentation_agent["agent_id"],
                "previous_outputs": [ideation_output]
            }
        )
    )

    # Save to trajectory
    experimentation_turn = {
        "turn_id": 1,
        "agent_index": 1,
        "agent_name": experimentation_agent["agent_id"],
        "agent_role": experimentation_agent["agent_role"],
        "agent_input": experimentation_input,
        "agent_output": experimentation_output,
        "agent_reward": None,  # To be filled in later
        "metadata": {}
    }
    trajectory.append(experimentation_turn)

    # ========================================================================
    # Agent 3: Writeup (Synthesize previous outputs)
    # ========================================================================

    # Create prompt containing summaries of the previous two agents' outputs
    exp_summary = experimentation_output[:500]

    writeup_prompt = f"""You are a research writeup agent. Synthesize ideas and experiments into a coherent research plan.

**Task/Problem:**
{prompt}

**Ideas Summary:**
{ideation_summary}

**Experiment Design Summary:**
{exp_summary}

**Your Task:**
1. Synthesize the ideas and experimental design
2. Create a structured research plan including objectives, methodology, expected outcomes
3. Write in clear, professional style

Provide a complete research plan.

**Your Response:**"""

    writeup_input = apply_template_with_tokenizer(
        writeup_agent["tokenizer"],
        writeup_prompt
    )

    logger.info("Agent 2 (Writeup) starting generation...")
    writeup_response = await writeup_agent["llm"].generate_async.remote(
        writeup_input,
        writeup_agent["sampling_params"]
    )
    writeup_output = writeup_response.outputs[0].text.strip()
    logger.info(f"Agent 2 generation complete, length: {len(writeup_output)} characters")

    # Immediately send coach evaluation request (non-blocking)
    logger.info("Sending Agent 2's coach evaluation request (async)...")
    writeup_reward_task = asyncio.create_task(
        coach.evaluate(
            problem=prompt,
            agent_input=writeup_prompt,
            agent_output=writeup_output,
            agent_role="writeup",
            context={
                "task": task,
                "agent_name": writeup_agent["agent_id"],
                "previous_outputs": [ideation_output, experimentation_output]
            }
        )
    )

    # Save to trajectory
    writeup_turn = {
        "turn_id": 2,
        "agent_index": 2,
        "agent_name": writeup_agent["agent_id"],
        "agent_role": writeup_agent["agent_role"],
        "agent_input": writeup_input,
        "agent_output": writeup_output,
        "agent_reward": None,  # To be filled in later
        "metadata": {}
    }
    trajectory.append(writeup_turn)

    # ========================================================================
    # Wait for all coach evaluations to return
    # ========================================================================

    logger.info("Waiting for all coach evaluations to return...")
    ideation_feedback, experimentation_feedback, writeup_feedback = await asyncio.gather(
        ideation_reward_task,
        experimentation_reward_task,
        writeup_reward_task
    )

    # Fill rewards into trajectory
    trajectory[0]["agent_reward"] = ideation_feedback.reward
    trajectory[1]["agent_reward"] = experimentation_feedback.reward
    trajectory[2]["agent_reward"] = writeup_feedback.reward

    # Save complete coach feedback to metadata (may be used for SFT in the future)
    trajectory[0]["metadata"]["coach_feedback"] = ideation_feedback.feedback_text
    trajectory[1]["metadata"]["coach_feedback"] = experimentation_feedback.feedback_text
    trajectory[2]["metadata"]["coach_feedback"] = writeup_feedback.feedback_text

    logger.info(f"All evaluations complete. Rewards: {[t['agent_reward'] for t in trajectory]}")

    # Final reward is the last agent's reward
    final_reward = writeup_feedback.reward

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "final_reward": final_reward
    }
