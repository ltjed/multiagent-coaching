"""
Research Workflow with Tool Support: Ideation → Experimentation → Writeup

Supported features:
1. Read system prompts from configuration (customizable roles for each agent)
2. Support for tool calls (each agent can call tools)
3. Variable number of actions (agent decides when to finish)
4. External LLM coach evaluation

Each agent can:
- Generate → Call tool → See result → Generate again → ... → Complete
- Up to max_turns rounds
"""

import os
import asyncio
from typing import Dict, List, Any, Optional
from marti.helpers.logging import init_logger
from marti.worlds.workflows.utils import apply_template_with_tokenizer
from marti.verifiers.coach import create_coach

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "INFO"))

# Global coach instance
_global_coach = None


def get_coach(workflow_args: Dict) -> Any:
    """Get or create coach instance (singleton)"""
    global _global_coach

    if _global_coach is None:
        coach_model = workflow_args.get("coach_model", "gemini-2.5-pro")
        coach_type = workflow_args.get("coach_type", "simple")
        use_vertex_ai = workflow_args.get("use_vertex_ai", False)
        vertex_project = workflow_args.get("vertex_project", None)
        vertex_location = workflow_args.get("vertex_location", "us-central1")
        max_output_tokens = workflow_args.get("coach_max_output_tokens", None)

        logger.info(f"Initializing coach: {coach_type}/{coach_model}")

        _global_coach = create_coach(
            model=coach_model,
            coach_type=coach_type,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            use_vertex_ai=use_vertex_ai,
            vertex_project=vertex_project,
            vertex_location=vertex_location
        )

        logger.info(f"Coach initialized successfully")

    return _global_coach


def get_agent_prompt_template(agent_role: str, workflow_args: Dict) -> str:
    """
    Get agent's prompt template (from configuration or use default)

    Args:
        agent_role: "ideation", "experimentation", "writeup"
        workflow_args: Workflow configuration

    Returns:
        Prompt template string
    """
    # Read custom templates from workflow_args
    custom_templates = workflow_args.get("agent_prompts", {})

    if agent_role in custom_templates:
        return custom_templates[agent_role]

    # Default templates
    default_templates = {
        "ideation": """You are a research ideation agent. Generate creative and feasible research ideas.

**Task/Problem:**
{problem}

{context}

**Your Task:**
1. Analyze the problem domain
2. Generate 2-3 specific research ideas or hypotheses
3. For each idea, explain the core insight and potential impact

You may use tools if needed (e.g., search for related work).

**Your Response:**""",

        "experimentation": """You are an experimentation design agent. Design concrete experiments to test research ideas.

**Task/Problem:**
{problem}

{context}

**Your Task:**
1. Select the most promising ideas
2. Design specific experiments to test them
3. Outline methodology and success criteria

You may use tools if needed (e.g., search for datasets, code execution for simulations).

**Your Response:**""",

        "writeup": """You are a research writeup agent. Synthesize ideas and experiments into a coherent research plan.

**Task/Problem:**
{problem}

{context}

**Your Task:**
1. Synthesize the ideas and experimental design
2. Create a structured research plan including objectives, methodology, expected outcomes
3. Write in clear, professional style

You may use tools if needed.

**Your Response:**"""
    }

    return default_templates.get(agent_role, "")


async def run_agent_with_tools(
    agent: Dict,
    prompt_template: str,
    problem: str,
    context: str,
    tool_manager,
    max_turns: int,
    metadata: Dict
) -> tuple[List[str], str]:
    """
    Run an agent with support for multi-turn tool calls

    Args:
        agent: Agent configuration
        prompt_template: Prompt template
        problem: Original problem
        context: Context (previous agents' outputs)
        tool_manager: Tool manager
        max_turns: Maximum number of turns
        metadata: Metadata

    Returns:
        (observation_history, final_output)
    """
    # Build initial prompt
    raw_prompt = prompt_template.format(problem=problem, context=context)

    # If tools are available, use multi-turn mode
    if tool_manager is not None and hasattr(agent["llm"], "get_trajectory"):
        # Use MARTI's multi-turn tool calls
        templated_prompt = apply_template_with_tokenizer(
            agent["tokenizer"],
            raw_prompt,
            tool_manager.tools if hasattr(tool_manager, "tools") else []
        )

        result = await agent["llm"].get_trajectory.remote(
            tool_manager,
            agent["tokenizer"],
            max_length=8192,  # Can be read from config
            prompt=templated_prompt,
            label=None,
            sampling_params=agent["sampling_params"],
            metadata=metadata
        )

        # result["observation"] is a list: [initial_prompt, agent_output, tool_response, agent_output, ...]
        observation = result["observation"]
        # The last element is the agent's final output
        final_output = observation[-1] if len(observation) > 1 else observation[0]

        return observation, final_output

    else:
        # Simple mode: single generation (no tools)
        templated_prompt = apply_template_with_tokenizer(
            agent["tokenizer"],
            raw_prompt
        )

        response = await agent["llm"].generate_async.remote(
            templated_prompt,
            agent["sampling_params"]
        )

        output = response.outputs[0].text.strip()
        return [templated_prompt, output], output


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
    Research workflow main function (with tool support)

    Execution order: Ideation → Experimentation → Writeup
    Each agent can:
    - Single generation (if no tools)
    - Multi-turn generation + tool calls (if tools available)

    Args:
        prompt: Research question/task
        label: Label
        agents: Agent configuration list
        tool_manager: Tool manager
        task: Task type
        metadata: Metadata
        **kwargs: Contains workflow_args

    Returns:
        {
            "prompt": str,
            "label": str,
            "trajectory": List[Dict],
            "final_reward": float
        }
    """
    workflow_args = kwargs.get("workflow_args", {})
    max_turns = workflow_args.get("max_turns", 3)  # Each agent up to 3 turns

    # Get coach
    coach = get_coach(workflow_args)

    # Validate number of agents
    if len(agents) < 3:
        raise ValueError(f"Research workflow requires 3 agents, but only {len(agents)} provided")

    ideation_agent = agents[0]
    experimentation_agent = agents[1]
    writeup_agent = agents[2]

    logger.info(f"Starting research workflow (with tools): {prompt[:100]}...")
    logger.info(f"  Tool manager: {tool_manager is not None}")
    logger.info(f"  Max turns per agent: {max_turns}")

    trajectory = []
    coach_tasks = []  # Collect all coach evaluation tasks

    # ========================================================================
    # Agent 0: Ideation
    # ========================================================================

    logger.info("Agent 0 (Ideation) starting execution...")

    ideation_template = get_agent_prompt_template("ideation", workflow_args)
    ideation_obs, ideation_output = await run_agent_with_tools(
        agent=ideation_agent,
        prompt_template=ideation_template,
        problem=prompt,
        context="",  # First agent has no prior context
        tool_manager=tool_manager,
        max_turns=max_turns,
        metadata=metadata or {}
    )

    logger.info(f"Agent 0 complete, output length: {len(ideation_output)} characters")
    logger.info(f"  Observation history length: {len(ideation_obs)}")

    # Immediately send coach evaluation
    logger.info("Sending Agent 0's coach evaluation request (async)...")
    ideation_task = asyncio.create_task(
        coach.evaluate(
            problem=prompt,
            agent_input=ideation_obs[0],  # Initial prompt
            agent_output=ideation_output,  # Final output
            agent_role="ideation",
            context={"task": task}
        )
    )
    coach_tasks.append(ideation_task)

    # Save turn
    ideation_turn = {
        "turn_id": 0,
        "agent_index": 0,
        "agent_name": ideation_agent["agent_id"],
        "agent_role": ideation_agent["agent_role"],
        "agent_input": ideation_obs[0],
        "agent_output": ideation_output,  # Use final output
        "agent_reward": None,
        "metadata": {
            "observation_history": ideation_obs,  # Save complete history
            "num_turns": len(ideation_obs) - 1  # Subtract initial prompt
        }
    }
    trajectory.append(ideation_turn)

    # ========================================================================
    # Agent 1: Experimentation
    # ========================================================================

    logger.info("Agent 1 (Experimentation) starting execution...")

    # Build context (summary of previous agent's output)
    ideation_summary = ideation_output[:500]
    context_1 = f"**Proposed Ideas (summary):**\n{ideation_summary}"

    experimentation_template = get_agent_prompt_template("experimentation", workflow_args)
    exp_obs, exp_output = await run_agent_with_tools(
        agent=experimentation_agent,
        prompt_template=experimentation_template,
        problem=prompt,
        context=context_1,
        tool_manager=tool_manager,
        max_turns=max_turns,
        metadata=metadata or {}
    )

    logger.info(f"Agent 1 complete, output length: {len(exp_output)} characters")
    logger.info(f"  Observation history length: {len(exp_obs)}")

    # Immediately send coach evaluation
    logger.info("Sending Agent 1's coach evaluation request (async)...")
    exp_task = asyncio.create_task(
        coach.evaluate(
            problem=prompt,
            agent_input=exp_obs[0],
            agent_output=exp_output,
            agent_role="experimentation",
            context={"task": task, "previous_outputs": [ideation_output]}
        )
    )
    coach_tasks.append(exp_task)

    # Save turn
    exp_turn = {
        "turn_id": 1,
        "agent_index": 1,
        "agent_name": experimentation_agent["agent_id"],
        "agent_role": experimentation_agent["agent_role"],
        "agent_input": exp_obs[0],
        "agent_output": exp_output,
        "agent_reward": None,
        "metadata": {
            "observation_history": exp_obs,
            "num_turns": len(exp_obs) - 1
        }
    }
    trajectory.append(exp_turn)

    # ========================================================================
    # Agent 2: Writeup
    # ========================================================================

    logger.info("Agent 2 (Writeup) starting execution...")

    # Build context
    exp_summary = exp_output[:500]
    context_2 = f"""**Ideas Summary:**
{ideation_summary}

**Experiment Design Summary:**
{exp_summary}"""

    writeup_template = get_agent_prompt_template("writeup", workflow_args)
    writeup_obs, writeup_output = await run_agent_with_tools(
        agent=writeup_agent,
        prompt_template=writeup_template,
        problem=prompt,
        context=context_2,
        tool_manager=tool_manager,
        max_turns=max_turns,
        metadata=metadata or {}
    )

    logger.info(f"Agent 2 complete, output length: {len(writeup_output)} characters")
    logger.info(f"  Observation history length: {len(writeup_obs)}")

    # Immediately send coach evaluation
    logger.info("Sending Agent 2's coach evaluation request (async)...")
    writeup_task = asyncio.create_task(
        coach.evaluate(
            problem=prompt,
            agent_input=writeup_obs[0],
            agent_output=writeup_output,
            agent_role="writeup",
            context={"task": task, "previous_outputs": [ideation_output, exp_output]}
        )
    )
    coach_tasks.append(writeup_task)

    # Save turn
    writeup_turn = {
        "turn_id": 2,
        "agent_index": 2,
        "agent_name": writeup_agent["agent_id"],
        "agent_role": writeup_agent["agent_role"],
        "agent_input": writeup_obs[0],
        "agent_output": writeup_output,
        "agent_reward": None,
        "metadata": {
            "observation_history": writeup_obs,
            "num_turns": len(writeup_obs) - 1
        }
    }
    trajectory.append(writeup_turn)

    # ========================================================================
    # Wait for all coach evaluations
    # ========================================================================

    logger.info("Waiting for all coach evaluations to return...")
    feedbacks = await asyncio.gather(*coach_tasks)

    # Fill rewards
    for i, feedback in enumerate(feedbacks):
        trajectory[i]["agent_reward"] = feedback.reward
        trajectory[i]["metadata"]["coach_feedback"] = feedback.feedback_text

    logger.info(f"All evaluations complete. Rewards: {[t['agent_reward'] for t in trajectory]}")

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "final_reward": feedbacks[-1].reward
    }
