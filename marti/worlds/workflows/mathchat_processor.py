"""
Processor for MathChat Workflow

Functions:
1. Receive trajectories with rewards
2. Distribute data by agent_index
3. Return training samples for each agent

This processor follows MARTI's async workflow API:
- Input: trajectories (already containing rewards)
- Output: per-agent samples
"""

from typing import List, Dict


def processor(
    trajectories: List[Dict],
    num_agents: int,
    global_args
) -> tuple[List[Dict[str, List]], Dict]:
    """
    Process trajectories into per-agent training data

    Args:
        trajectories: List of trajectory dicts returned by workflow
            Each trajectory contains:
            {
                "prompt": str,
                "label": str,
                "trajectory": [
                    {
                        "agent_index": int,
                        "agent_input": str,
                        "agent_output": str,
                        "agent_reward": float,
                        ...
                    },
                    ...
                ],
                "final_reward": float
            }

        num_agents: Number of agents (should be 3)
        global_args: Global configuration parameters

    Returns:
        Tuple of:
        - List of dicts, each dict corresponds to one agent's training data:
          [
              {"prompts": [...], "outputs": [...], "labels": [...]},  # Agent 0
              {"prompts": [...], "outputs": [...], "labels": [...]},  # Agent 1
              {"prompts": [...], "outputs": [...], "labels": [...]},  # Agent 2
          ]
        - Dict of coach reward metadata for W&B logging
    """
    # Sort by prompt (required by MARTI for packing)
    trajectories = sorted(trajectories, key=lambda traj: traj.get("prompt", ""))

    # Initialize data bucket for each agent
    samples = [
        {"prompts": [], "outputs": [], "labels": []}
        for _ in range(num_agents)
    ]

    # Collect coach rewards for W&B logging
    coach_rewards = []
    trajectory_counter = 0

    # Iterate through all trajectories, distribute data
    for traj in trajectories:
        for turn in traj["trajectory"]:
            agent_idx = turn["agent_index"]

            # Validate agent_index is valid
            if agent_idx >= num_agents:
                raise ValueError(
                    f"Invalid agent_index {agent_idx}, num_agents={num_agents}"
                )

            # Collect this agent's data
            # Process 0-10 reward to 0-1 range for training stability
            raw_reward = turn["agent_reward"]
            processed_reward = raw_reward / 10.0  # Normalize 0-10 -> 0.0-1.0

            samples[agent_idx]["prompts"].append(turn["agent_input"])
            samples[agent_idx]["outputs"].append(turn["agent_output"])
            samples[agent_idx]["labels"].append(processed_reward)  # Use processed reward for training

            # Collect coach reward for logging (keep raw score for observability)
            coach_rewards.append({
                "trajectory_id": trajectory_counter,
                "turn_id": turn.get("turn_id", 0),
                "agent_index": agent_idx,
                "agent_role": turn.get("agent_role", f"agent_{agent_idx}"),
                "reward": raw_reward,  # Log raw 0-10 score
                "processed_reward": processed_reward
            })

        trajectory_counter += 1

    # Print statistics
    print(f"\n[Processor] Processed {len(trajectories)} trajectories")
    for agent_idx in range(num_agents):
        num_samples = len(samples[agent_idx]["prompts"])
        avg_reward = sum(samples[agent_idx]["labels"]) / num_samples if num_samples > 0 else 0
        print(f"  Agent {agent_idx}: {num_samples} samples, avg reward: {avg_reward:.3f}")

    # Note: Rollout accuracy aggregation is now handled in MultiAgentWorldAsync.generate_samples()
    # This processor only handles data conversion, not metadata computation

    # Prepare coach reward metadata for W&B
    coach_metadata = {
        "coach_rewards": coach_rewards,
        "num_trajectories": len(trajectories),
        "total_turns": len(coach_rewards)
    }

    return samples, coach_metadata
