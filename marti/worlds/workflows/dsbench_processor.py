"""
DSBench Data Science Pipeline Processor

Purpose:
1. Receives trajectories with rewards (from coach evaluation)
2. Distributes data by agent_index
3. Returns per-agent training samples

This processor follows MARTI's async workflow API:
- Input: trajectories (already containing rewards from coach)
- Output: per-agent samples for RL training
"""

from typing import List, Dict


def processor(
    trajectories: List[Dict],
    num_agents: int,
    global_args
) -> tuple[List[Dict[str, List]], Dict]:
    """
    Process trajectories into per-agent training data.

    Args:
        trajectories: List of trajectory dictionaries from workflow
            Each trajectory contains:
            {
                "prompt": str,  # Task description
                "label": str,  # Ground truth (for evaluation)
                "trajectory": [
                    {
                        "agent_index": int,  # 0=DataEngineer, 1=Modeler, 2=Analyst
                        "agent_input": str,  # Prompt for this action
                        "agent_output": str,  # Agent's response
                        "agent_reward": float,  # Coach reward (0.0-1.0)
                        ...
                    },
                    ...
                ],
                "final_reward": float  # Last reward (Analyst's final score)
            }

        num_agents: Number of agents (should be 3)
        global_args: Global configuration parameters

    Returns:
        Tuple of:
        - List of dicts, one per agent:
          [
              {"prompts": [...], "outputs": [...], "labels": [...]},  # Data Engineer
              {"prompts": [...], "outputs": [...], "labels": [...]},  # Modeler
              {"prompts": [...], "outputs": [...], "labels": [...]},  # Analyst
          ]
        - Dict of coach reward metadata for W&B logging
    """
    # Sort by prompt (required by MARTI for packing)
    trajectories = sorted(trajectories, key=lambda traj: traj.get("prompt", ""))

    # Initialize data buckets for each agent
    samples = [
        {"prompts": [], "outputs": [], "labels": []}
        for _ in range(num_agents)
    ]

    # Collect coach rewards for W&B logging
    coach_rewards = []
    trajectory_counter = 0

    # Iterate through all trajectories and distribute data
    for traj in trajectories:
        for turn in traj["trajectory"]:
            agent_idx = turn["agent_index"]

            # Validate agent_index
            if agent_idx >= num_agents:
                raise ValueError(
                    f"Invalid agent_index {agent_idx}, num_agents={num_agents}"
                )

            # Collect data for this agent
            samples[agent_idx]["prompts"].append(turn["agent_input"])
            samples[agent_idx]["outputs"].append(turn["agent_output"])
            samples[agent_idx]["labels"].append(turn["agent_reward"])

            # Collect coach reward for logging
            coach_rewards.append({
                "trajectory_id": trajectory_counter,
                "turn_id": turn.get("turn_id", 0),
                "agent_index": agent_idx,
                "agent_role": turn.get("agent_role", f"agent_{agent_idx}"),
                "reward": turn["agent_reward"],
                "action_index": turn.get("metadata", {}).get("action_index", 0),
                "total_actions": turn.get("metadata", {}).get("total_actions", 1)
            })

        trajectory_counter += 1

    # Print statistics
    agent_names = ["Data Engineer", "Modeler", "Analyst"]
    print(f"\n[DSBench Processor] Processed {len(trajectories)} trajectories")
    for agent_idx in range(num_agents):
        num_samples = len(samples[agent_idx]["prompts"])
        avg_reward = sum(samples[agent_idx]["labels"]) / num_samples if num_samples > 0 else 0
        agent_name = agent_names[agent_idx] if agent_idx < len(agent_names) else f"Agent {agent_idx}"
        print(f"  {agent_name}: {num_samples} samples, avg reward: {avg_reward:.3f}")

    # Prepare coach reward metadata for W&B
    coach_metadata = {
        "coach_rewards": coach_rewards,
        "num_trajectories": len(trajectories),
        "total_turns": len(coach_rewards),
        "agent_names": agent_names
    }

    return samples, coach_metadata
