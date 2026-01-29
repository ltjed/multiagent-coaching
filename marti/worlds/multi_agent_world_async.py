# marti/marti/models/vllm/multi_agent_engine_async.py
import torch
import asyncio
import os
import time
import ray
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum
import copy
import importlib
import torch
from vllm import SamplingParams

from marti.models.model_utils import process_sequences
from marti.worlds.base_world import Samples

from marti.helpers.logging import init_logger
from marti.models.vllm.engine_async import LLMRayActorAsync, AgentInstance, get_tokenize_text_len
from marti.verifiers.auto_verify import auto_verify
from marti.worlds.base_world import BaseWorld

from marti.worlds.workflows.workflow_wrapper import MultiAgentWrapper
from marti.worlds.workflows.default_processor import processor
from marti.worlds.tools.manager import ToolManager
from marti.worlds.tools.mcp_manager import MCPManager
from marti.worlds.tool_world import register_mcp_tools, register_openai_tools, print_tools, assign_action_mask

logger = init_logger(__name__)

class MultiAgentWorldAsync(BaseWorld):
    def __init__(self, strategy, agents, *args, **kwargs):
        super().__init__(strategy, agents, *args, **kwargs)
        """
        agents: List[Dict[str, Any]]
             {
                "agent_id": unique agent id
                "agent_role": agent role (generator/refiner/verifier/coder/...)
                "pretrain": path to pretrain models
                "llms": a list of vllm engines
                "tokenizer": hf tokenizer
                "generate_kwargs": generate kwargs, which is different from vllm.SamplingParams
                "is_reasoning_model": reasoning model with <think> tags or not
            }
        """

        self.workflow_args = self.args.get("workflow_args", {})
        print("workflow args", self.workflow_args)
        self.num_agents = len(self.agents)
    
        self._init_tool_manager()
        self._init_processor()

    def _init_tool_manager(self):
        self.tools_config = self.args.get("tools_config", {})

        # assert self.packing_samples, "Only support packing samples"  # COMMENTED OUT: Testing non-packing mode for memory optimization

        if self.tools_config.get("mcp_url", None) is not None:
            self.tool_manager = MCPManager(self.tools_config)
            self.tools = register_mcp_tools(self.tool_manager)
        else:
            self.tool_manager = ToolManager(self.tools_config)
            self.tools = register_openai_tools(self.tools_config, self.tool_manager)

        self.tool_manager.set_tools(self.tools)
        print_tools(self.tools)

    def _init_processor(self):
        """
        We convert collected workflow trajectories into training samples for each agent with the given processor
        """
        if self.args.processor_func_path is None:
            self.processor = processor
        if self.args.processor_func_path.endswith(".py"):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "processor", self.args.processor_func_path)
            processor_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(processor_module)
            self.processor = processor_module.processor
        else:
            raise ValueError("Processor path must be a Python file")

    def get_rank_agent(self, rank, world_size, is_eval=False):
        """
        Get the first llm for async request
        """
        rank_agents = [{} for _ in range(self.num_agents)]
        for aid, agent in enumerate(self.agents):
            agent_llms = agent["llms"]
            if len(agent_llms) <= world_size:
                llms = [agent_llms[rank % len(agent_llms)]]
            else:
                llms = agent_llms[rank::world_size]

            generate_kwargs = agent["generate_kwargs"]
            sampling_params = SamplingParams(
                temperature=generate_kwargs.get(
                    "eval_temperature" if is_eval else "temperature", 1.0),
                top_p=generate_kwargs.get("top_p", 1.0),
                top_k=generate_kwargs.get("top_k", -1),
                max_tokens=generate_kwargs.get("max_new_tokens", 1024),
                min_tokens=generate_kwargs.get("min_new_tokens", 16),
                skip_special_tokens=generate_kwargs.get(
                    "skip_special_tokens", False),
                include_stop_str_in_output=True,
                truncate_prompt_tokens=self.args.prompt_max_len if self.args.truncate_prompt else None)

            agent_dict = {
                "llm": llms[0],
                "sampling_params": sampling_params
            }
            for use_key in ["agent_id", "agent_role", "tokenizer", "is_reasoning_model"]:
                agent_dict[use_key] = deepcopy(agent[use_key])

            rank_agents[aid] = agent_dict
        return rank_agents

    def tokenize_fn_with_tok(self, messages, tokenizer=None, max_len=None):
        tokenizer = self.shared_tokenizer if tokenizer is None else tokenizer
        # For inputs
        if isinstance(messages, list):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompt_max_len = self.args.prompt_max_len if max_len is None else max_len
        # For outputs
        elif isinstance(messages, str):
            prompt = messages
            prompt_max_len = self.args.generate_max_len if max_len is None else max_len
        else:
            raise NotImplementedError

        return self.tokenize_fn(tokenizer, prompt, prompt_max_len, padding=False)["input_ids"]

    def distribute_prompts(self, task, prompts, labels, metadata, rank_agents_list, is_eval=False):
        if is_eval:
            all_prompts = [prompts for _ in rank_agents_list]
            all_labels = [labels for _ in rank_agents_list]
            all_metadata = [metadata for _ in rank_agents_list]
        else:
            if len(prompts) < len(rank_agents_list):
                raise ValueError("Number of prompts must be more than rank_agents_list")
            chunk_size = (len(prompts) + len(rank_agents_list) - 1) // len(rank_agents_list)
            all_prompts = [
                prompts[i*chunk_size: (i+1)*chunk_size] for i in range(len(rank_agents_list))]
            # Handle None labels (DSBench case where ground truth is external)
            if labels is None:
                all_labels = [[None] * len(all_prompts[i]) for i in range(len(rank_agents_list))]
            else:
                all_labels = [
                    labels[i*chunk_size: (i+1)*chunk_size] for i in range(len(rank_agents_list))]
            all_metadata = [
                metadata[i*chunk_size: (i+1)*chunk_size] for i in range(len(rank_agents_list))]

        refs = []
        all_wrappers = []
        for per_llm_prompts, per_llm_labels, per_llm_metadata, rank_agents in zip(all_prompts, all_labels, all_metadata, rank_agents_list):
            multi_agent_wrapper = MultiAgentWrapper.remote(
                agents=rank_agents,
                workflow_args=self.workflow_args,
                workflow_func_path=self.args.workflow_func_path
            )
            ref = multi_agent_wrapper.add_requests.remote(
                tool_manager=self.tool_manager,
                prompts=per_llm_prompts,
                labels=per_llm_labels,
                task=task,
                metadata=per_llm_metadata,
                max_length=self.total_max_len,
                is_eval=is_eval
            )
            refs.append(ref)
            all_wrappers.append(multi_agent_wrapper)
        ray.get(refs)

        all_output_refs = []
        for i, (per_llm_prompts, wrapper) in enumerate(zip(all_prompts, all_wrappers)):
            logger.info(f"[distribute_prompts] Requesting {len(per_llm_prompts)} responses from wrapper {i}")
            all_output_refs.append(wrapper.get_responses.remote(expected_len=len(per_llm_prompts)))

        logger.info(f"[distribute_prompts] Waiting for {len(all_output_refs)} workers to return results...")
        all_trajectories = ray.get(all_output_refs)
        logger.info(f"[distribute_prompts] Received results from all workers")

        if is_eval:
            for i, trajectories in enumerate(all_trajectories):
                logger.info(f"[distribute_prompts] Worker {i} returned {len(trajectories)} trajectories (expected {len(prompts)})")
                # Check each trajectory
                for j, traj in enumerate(trajectories):
                    if traj is None:
                        logger.error(f"[distribute_prompts] Worker {i}, trajectory {j} is None!")
                    elif not isinstance(traj, dict):
                        logger.error(f"[distribute_prompts] Worker {i}, trajectory {j} is not dict! type={type(traj)}")
                    else:
                        has_reward = "final_reward" in traj
                        reward_value = traj.get("final_reward")
                        logger.info(f"[distribute_prompts] Worker {i}, trajectory {j}: has_reward={has_reward}, reward={reward_value}")

                assert len(trajectories) == len(
                    prompts), f"{len(trajectories)} vs {len(prompts)}"
            return all_trajectories
        else:
            all_trajectories = sum(all_trajectories, [])
            assert len(all_trajectories) == len(
                prompts), f"{len(all_trajectories)} vs {len(prompts)}"
            return all_trajectories

    def _compute_rollout_accuracy(self, trajectories):
        """
        Compute rollout accuracy by grouping trajectories by prompt.
        Handles multiple samples per prompt (e.g., from multiple engines or n_samples_per_prompt).

        Args:
            trajectories: List of trajectory dictionaries, each containing:
                - "prompt": str
                - "outcome_score": float or None (0.0-1.0, continuous)

        Returns:
            Dict with keys:
                - "rollout_accuracy": float or None
                - "rollout_correct": int (total correct trajectories)
                - "rollout_total": int (total number of trajectories)
                - "rollout_samples_per_prompt": float
        """
        # Group trajectories by prompt
        prompt_groups = {}
        for traj in trajectories:
            prompt = traj.get("prompt")
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            outcome_score = traj.get("outcome_score")
            if outcome_score is not None:
                prompt_groups[prompt].append(outcome_score)
        
        # Calculate accuracy per prompt, then average across prompts
        prompt_accuracies = []
        total_correct = 0
        total_trajectories = 0
        
        for prompt, scores in prompt_groups.items():
            if scores:
                prompt_acc = sum(scores) / len(scores)
                prompt_accuracies.append(prompt_acc)
                total_correct += sum(scores)
                total_trajectories += len(scores)
        
        if prompt_accuracies:
            rollout_accuracy = sum(prompt_accuracies) / len(prompt_accuracies)
            rollout_correct = total_correct  # Total correct trajectories across all prompts
            rollout_total = total_trajectories  # Total number of trajectories (including duplicates)
            rollout_samples_per_prompt = total_trajectories / len(prompt_groups) if len(prompt_groups) > 0 else 0
            logger.info(f"[Rollout] Accuracy: {rollout_accuracy:.3f} ({rollout_correct}/{rollout_total} trajectories, {len(prompt_groups)} unique prompts, {rollout_samples_per_prompt:.1f} samples/prompt)")
        else:
            rollout_accuracy = None
            rollout_correct = 0
            rollout_total = 0
            rollout_samples_per_prompt = 0
        
        return {
            "rollout_accuracy": rollout_accuracy,
            "rollout_correct": rollout_correct,
            "rollout_total": rollout_total,
            "rollout_samples_per_prompt": rollout_samples_per_prompt
        }

    def _aggregate_eval_results(self, all_accuracies, original_indices, original_length, aggregation="mean"):
        """
        Aggregate multiple evaluation results by original prompt index
        
        Args:
            all_accuracies: List[float] - All accuracy values (flattened)
            original_indices: List[int] - Original prompt index for each accuracy
            original_length: int - Number of original prompts
            aggregation: str - Aggregation method ("mean", "majority_vote", "pass@k")
        
        Returns:
            List[float] - Aggregated accuracy for each original prompt
        """
        # Group by original index
        grouped_accs = [[] for _ in range(original_length)]
        for acc, orig_idx in zip(all_accuracies, original_indices):
            if 0 <= orig_idx < original_length:
                grouped_accs[orig_idx].append(acc)
        
        aggregated = []
        for orig_idx, accs in enumerate(grouped_accs):
            if not accs:
                logger.warning(f"[_aggregate_eval_results] No results for prompt {orig_idx}, defaulting to 0.0")
                aggregated.append(0.0)
                continue
            
            if aggregation == "mean":
                aggregated.append(np.mean(accs))
            elif aggregation == "majority_vote":
                # For binary correctness, majority vote
                votes = [1 if acc > 0.5 else 0 for acc in accs]
                aggregated.append(1.0 if sum(votes) > len(votes) / 2 else 0.0)
            elif aggregation.startswith("pass@"):
                # pass@k: at least k correct
                k = int(aggregation.split("@")[1])
                correct_count = sum(1 for acc in accs if acc > 0.5)
                aggregated.append(1.0 if correct_count >= k else 0.0)
            else:
                logger.warning(f"[_aggregate_eval_results] Unknown aggregation '{aggregation}', using mean")
                aggregated.append(np.mean(accs))
        
        return aggregated

    @torch.no_grad()
    def evaluate_samples(self, eval_data):
        args = self.strategy.args

        # Extract original prompts
        all_prompts = eval_data["prompt"]
        all_labels = eval_data["label"]
        all_metadata = eval_data["metadata"]

        # Handle None labels (DSBench case where ground truth is external)
        if all_labels is None:
            all_labels = [None] * len(all_prompts)

        # Record original length and index mapping
        original_length = len(all_prompts)
        n_eval = getattr(args, "n_eval_samples_per_prompt", 1)

        # Expand prompts if n_eval > 1
        if n_eval > 1:
            logger.info(f"[evaluate_samples] Expanding {original_length} prompts to {original_length * n_eval} (n_eval={n_eval})")
            # Create index mapping: each expanded prompt maps to original index
            original_indices = []
            expanded_prompts = []
            expanded_labels = []
            expanded_metadata = []

            for orig_idx, (prompt, label, metadata) in enumerate(zip(all_prompts, all_labels, all_metadata)):
                for _ in range(n_eval):
                    original_indices.append(orig_idx)
                    expanded_prompts.append(prompt)
                    expanded_labels.append(label)
                    expanded_metadata.append(metadata)

            all_prompts = expanded_prompts
            all_labels = expanded_labels
            all_metadata = expanded_metadata
        else:
            original_indices = list(range(original_length))

        world_size = len(self.agents[0]["llms"])
        rank_agents_list = [self.get_rank_agent(
            rank=idx,
            world_size=world_size,
            is_eval=True) for idx in range(world_size)]

        if self.args.eval_workers > 0:
            rank_agents_list = rank_agents_list[:self.args.eval_workers]

        all_results = self.distribute_prompts(args.verify_task_eval,
                                              all_prompts,
                                              all_labels,
                                              all_metadata,
                                              rank_agents_list,
                                              is_eval=True)

        # Collect all accuracy values from trajectories
        all_accuracies = []
        all_outcome_metrics = []  # For DSBench: collect outcome_metrics for per-metric aggregation
        for worker_idx, trajectories in enumerate(all_results):
            logger.info(f"[evaluate_samples] Worker {worker_idx}: processing {len(trajectories)} trajectories")
            worker_accs = []
            for traj_idx, trajectory in enumerate(trajectories):
                if trajectory is None:
                    logger.warning(f"[evaluate_samples] Worker {worker_idx}, traj {traj_idx}: None trajectory (skipping)")
                    continue
                elif not isinstance(trajectory, dict):
                    logger.error(f"[evaluate_samples] Worker {worker_idx}, traj {traj_idx}: Not a dict! type={type(trajectory)}")
                    continue

                # Get accuracy value (outcome_score is continuous 0.0-1.0)
                outcome_score = trajectory.get("outcome_score")
                final_reward = trajectory.get("final_reward")
                outcome_metrics = trajectory.get("outcome_metrics")

                # Collect outcome_metrics for DSBench aggregation
                if outcome_metrics is not None and isinstance(outcome_metrics, dict) and 'error' not in outcome_metrics:
                    all_outcome_metrics.append(outcome_metrics)

                if outcome_score is not None:
                    acc_value = outcome_score
                elif final_reward is not None:
                    acc_value = final_reward
                else:
                    logger.error(f"[evaluate_samples] Worker {worker_idx}, traj {traj_idx}: BOTH outcome_score and final_reward are None! keys={trajectory.keys()}")
                    acc_value = 0.0

                worker_accs.append(acc_value)

            logger.info(f"[evaluate_samples] Worker {worker_idx}: collected {len(worker_accs)} valid accuracy values")
            all_accuracies.append(worker_accs)

        # Flatten all worker results
        flat_accuracies = []
        for acc_list in all_accuracies:
            flat_accuracies.extend(acc_list)
        
        # If n_eval > 1, aggregate by original prompt
        if n_eval > 1:
            # Note: original_indices needs to account for multiple workers (duplicate in is_eval=True)
            # Each worker processes all expanded prompts, so we need to repeat indices
            num_workers = len(all_results)
            repeated_indices = []
            for _ in range(num_workers):
                repeated_indices.extend(original_indices)
            
            aggregated_accs = self._aggregate_eval_results(
                flat_accuracies,
                repeated_indices,
                original_length,
                getattr(args, "eval_aggregation", "mean")
            )
            accuracy = np.mean(aggregated_accs)
            logger.info(f"[evaluate_samples] Final accuracy: {accuracy:.4f} (n_eval={n_eval}, original_prompts={original_length}, aggregated={len(aggregated_accs)} prompts)")
        else:
            # Original logic: directly average all results
            non_empty_accuracies = [acc_list for acc_list in all_accuracies if acc_list]
            logger.info(f"[evaluate_samples] Non-empty workers: {len(non_empty_accuracies)}/{len(all_accuracies)}")
            
            if non_empty_accuracies:
                accuracy = np.mean([np.mean(acc) for acc in non_empty_accuracies])
                logger.info(f"[evaluate_samples] Final accuracy: {accuracy:.4f}")
            else:
                accuracy = 0.0  # No valid trajectories
                logger.warning(f"[evaluate_samples] No valid trajectories! Accuracy defaulting to 0.0")

        # Aggregate outcome_metrics for DSBench (per-metric averages)
        aggregated_metrics = {}
        if all_outcome_metrics:
            # Group metrics by type
            metric_values = {}
            for om in all_outcome_metrics:
                for key, value in om.items():
                    # Skip non-numeric and metadata fields
                    if key in ['data_type', 'task_specified_metric', 'error'] or not isinstance(value, (int, float)):
                        continue
                    if key not in metric_values:
                        metric_values[key] = []
                    metric_values[key].append(value)

            # Compute averages
            for key, values in metric_values.items():
                if values:
                    aggregated_metrics[f"avg_{key}"] = float(np.mean(values))
                    aggregated_metrics[f"count_{key}"] = len(values)

            logger.info(f"[evaluate_samples] Aggregated outcome metrics: {aggregated_metrics}")

        return {"accuracy": accuracy, "metadata": all_results, "aggregated_metrics": aggregated_metrics}

    @torch.no_grad()
    def generate_samples(self, all_prompts, rank=0, world_size=8):
        args = self.strategy.args
        # Set return_list to False, and then we only get one llm for async request
        rank_agents_list = [self.get_rank_agent(
            rank=rank,
            world_size=world_size,
            is_eval=False)]

        all_prompts, all_labels, all_metadata = all_prompts[
            "prompt"], all_prompts["label"], all_prompts["metadata"]
        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum(
            [[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        # Handle None labels (DSBench case where ground truth is external)
        if all_labels is None:
            all_labels = [None] * len(all_prompts)
        else:
            all_labels = sum(
                [[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_metadata = sum(
            [[metadata] * args.n_samples_per_prompt for metadata in all_metadata], [])

        all_trajectories = self.distribute_prompts(args.verify_task,
                                              all_prompts,
                                              all_labels,
                                              all_metadata,
                                              rank_agents_list)

        processor_result = self.processor(all_trajectories, self.num_agents, self.args)

        # Handle both old (list) and new (tuple) processor return formats
        if isinstance(processor_result, tuple):
            training_samples, coach_metadata = processor_result
        else:
            training_samples = processor_result
            coach_metadata = None
        
        # Compute rollout accuracy from trajectories (if not already computed by processor)
        # This ensures all async tasks benefit from rollout accuracy aggregation
        if coach_metadata is None:
            coach_metadata = {}
        
        # Only compute if not already present (backward compatibility with processors that compute it)
        if "rollout_accuracy" not in coach_metadata or coach_metadata.get("rollout_accuracy") is None:
            rollout_accuracy_metadata = self._compute_rollout_accuracy(all_trajectories)
            coach_metadata.update(rollout_accuracy_metadata)

        if rank == 0:
            for index in range(3):
                for agent_index, agent_samples in enumerate(training_samples):
                    for key, values in agent_samples.items():
                        print(agent_index, key, str(values[index]))

        def flatten_list(full_list):
            return [ v for sublist in full_list for v in sublist]

        samples_list = [[] for _ in range(self.num_agents)]
        for agent_idx, samples in enumerate(training_samples):
            agent_tokenizer = self.agents[agent_idx]["tokenizer"]
            
            # prompt_ids = [self.tokenize_fn_with_tok(prompt, agent_tokenizer, max_len=self.args.prompt_max_len) for prompt in samples["prompts"]]
            # output_ids = [self.tokenize_fn_with_tok(output, agent_tokenizer, max_len=self.args.generate_max_len) for output in samples["outputs"]]
            
            prompt_ids, output_ids, action_mask = [], [], []
            for prompt, output in zip(samples["prompts"], samples["outputs"]):
                cur_prompt_ids = self.tokenize_fn_with_tok(prompt, agent_tokenizer, max_len=self.args.prompt_max_len)
                prompt_ids.append(cur_prompt_ids)
                cur_output_ids = self.tokenize_fn(agent_tokenizer, output, self.total_max_len, padding=False)["input_ids"]
                if isinstance(output, list):
                    actions = [assign_action_mask(turn) for turn in output]
                    cur_action_mask = [[action]*len(turn) for action, turn in zip(actions, cur_output_ids)]
                    output_ids.append(flatten_list(cur_output_ids))
                    action_mask.append(flatten_list(cur_action_mask))
                else:
                    output_ids.append(cur_output_ids)
                    action_mask.append([1]*len(cur_output_ids))
            
            all_labels = samples["labels"]

            for i in range(0, len(prompt_ids), args.micro_rollout_batch_size):
                prompts = prompt_ids[i: i + args.micro_rollout_batch_size]
                outputs = output_ids[i: i + args.micro_rollout_batch_size]
                labels = all_labels[i: i + args.micro_rollout_batch_size]
                actions = action_mask[i: i + args.micro_rollout_batch_size]
                
                samples_list[agent_idx].append(self.prepare_samples(
                    prompts=prompts,
                    outputs=outputs,
                    pred_labels=labels,
                    action_mask_list=actions,
                    tokenizer=agent_tokenizer,
                ))

        result = {"sample": samples_list}
        if coach_metadata is not None:
            result["coach_metadata"] = coach_metadata

        return result

    def prepare_samples(self,
                        prompts,
                        outputs,
                        pred_labels,
                        action_mask_list=None,
                        num_agent_actions=None,
                        tokenizer=None):
        # DEBUG: Entry point
        print(f"[DEBUG prepare_samples ENTRY] packing_samples={self.packing_samples}, len(prompts)={len(prompts)}, len(outputs)={len(outputs)}")
        print(f"[DEBUG prepare_samples ENTRY] prompts type={type(prompts)}, outputs type={type(outputs)}")

        pred_labels = torch.tensor(
            pred_labels, device="cpu", dtype=torch.float)

        pad_token_id, eos_token_id = tokenizer.pad_token_id, tokenizer.eos_token_id

        # Handle edge case: empty outputs
        if len(outputs) == 0:
            # Return minimal valid Samples object
            sequences = torch.zeros((1, 0), dtype=torch.long, device="cpu")
            attention_mask = torch.zeros((1, 0), dtype=torch.long, device="cpu")
            action_mask = torch.zeros((1, 0), dtype=torch.long, device="cpu")
            response_length = torch.zeros(0, dtype=torch.float, device="cpu")
            total_length = torch.zeros(0, dtype=torch.float, device="cpu")
            packed_seq_lens = [] if self.packing_samples else None
            print(f"[DEBUG prepare_samples] WARNING: Empty outputs, returning minimal Samples")
            return Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=[],
                packed_seq_lens=packed_seq_lens,
                response_length=response_length,
                total_length=total_length,
                num_agent_actions=num_agent_actions,
                labels=pred_labels,
            )

        if self.packing_samples:
            # PACKING MODE: concat all outputs to following format:
            #
            # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
            # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
            sequences = []
            packed_seq_lens = []
            attention_mask = []
            num_actions = []
            action_mask = []
            for i, output in enumerate(outputs):
                prompt = prompts[i]
                input_len = len(prompt)
                output_len = len(output)
                packed_seq_lens.append(input_len + output_len)
                sequences.extend(prompt + list(output))
                attention_mask.extend([i + 1] * (input_len + output_len))

                assert output_len > 1, f"output_len = {output_len}"
                num_actions.append(max(1, output_len))
                action_mask.extend(action_mask_list[i])

            sequences = torch.tensor(sequences, device="cpu").unsqueeze(0)
            attention_mask = torch.tensor(attention_mask, device="cpu").unsqueeze(0)
            action_mask = torch.tensor(action_mask, device="cpu").unsqueeze(0)
            response_length = torch.tensor(num_actions, device="cpu", dtype=torch.float)
            total_length = torch.tensor(packed_seq_lens, device="cpu", dtype=torch.float)

            # Validate packed length if limit is set
            total_packed_length = sum(packed_seq_lens)
            max_packed_length = getattr(self.args, 'max_packed_length', None)
            if max_packed_length is not None and total_packed_length > max_packed_length:
                # Calculate suggested batch size
                avg_seq_len = total_packed_length / len(packed_seq_lens)
                suggested_batch_size = max(1, int(max_packed_length / avg_seq_len))
                print(f"[WARNING] Rollout batch packed length {total_packed_length} exceeds limit {max_packed_length}")
                print(f"[WARNING] Current micro_rollout_batch_size={len(packed_seq_lens)}, suggested={suggested_batch_size}")
                print(f"[WARNING] Estimated logits memory: {(total_packed_length * 152000 * 2) / 1e9:.1f} GB")

            print(f"[DEBUG prepare_samples] PACKING: Packed {len(outputs)} sequences into shape={sequences.shape}, packed_seq_lens={packed_seq_lens}, total_length={total_packed_length}")
        else:
            # NON-PACKING MODE: Keep sequences separate with padding
            # DEBUG: Check prompts type and lengths
            print(f"[DEBUG prepare_samples NON-PACKING] len(prompts)={len(prompts)}, len(outputs)={len(outputs)}")
            print(f"[DEBUG prepare_samples NON-PACKING] prompts type={type(prompts)}, first prompt type={type(prompts[0]) if len(prompts) > 0 else 'N/A'}")
            if len(prompts) > 0:
                prompt_lens = [len(p) if isinstance(p, (list, torch.Tensor)) else 'N/A' for p in prompts[:3]]
                print(f"[DEBUG prepare_samples NON-PACKING] first 3 prompt lengths: {prompt_lens}")

            sequences_list = []
            attention_mask_list = []
            action_mask_list_out = []
            num_actions = []
            seq_lens = []

            for i, output in enumerate(outputs):
                prompt = prompts[i]
                seq = prompt + list(output)
                sequences_list.append(torch.tensor(seq, device="cpu"))
                attention_mask_list.append(torch.ones(len(seq), device="cpu", dtype=torch.long))

                # Create full action_mask aligned with sequence
                # Prepend zeros for prompt portion, then add action_mask for output portion
                full_action_mask = torch.zeros(len(seq), device="cpu")
                full_action_mask[len(prompt):] = torch.tensor(action_mask_list[i], device="cpu")
                action_mask_list_out.append(full_action_mask)

                num_actions.append(max(1, len(output)))
                seq_lens.append(len(seq))

            # Pad to max length
            max_len = max(seq_lens)
            padded_sequences = []
            padded_attention_mask = []
            padded_action_mask = []

            for seq, att_mask, act_mask in zip(sequences_list, attention_mask_list, action_mask_list_out):
                pad_len = max_len - seq.size(0)  # All tensors now have same length, use same padding
                padded_sequences.append(torch.nn.functional.pad(seq, (pad_len, 0), value=pad_token_id))
                padded_attention_mask.append(torch.nn.functional.pad(att_mask, (pad_len, 0), value=0))
                padded_action_mask.append(torch.nn.functional.pad(act_mask, (pad_len, 0), value=0))

            sequences = torch.stack(padded_sequences, dim=0)  # (batch, max_len)
            attention_mask = torch.stack(padded_attention_mask, dim=0)  # (batch, max_len)
            action_mask = torch.stack(padded_action_mask, dim=0)  # (batch, max_len)
            response_length = torch.tensor(num_actions, device="cpu", dtype=torch.float)
            total_length = torch.tensor(seq_lens, device="cpu", dtype=torch.float)
            packed_seq_lens = None  # Not used in non-packing mode
            print(f"[DEBUG prepare_samples] NON-PACKING: Created batch shape={sequences.shape}, seq_lens={seq_lens}, padding_waste={(max_len*len(outputs) - sum(seq_lens))/(max_len*len(outputs))*100:.1f}%")

        # if action_mask_list is not None:
        #     action_mask = sum(action_mask_list, [])
        #     assert len(action_mask) == sum(
        #         num_actions), f"action_mask ({len(action_mask)}) and num_actions ({sum(num_actions)}) should have the same length"
        #     # TODO: action_mask should be a int tensor not bool tensor
        #     action_mask = torch.tensor(
        #         action_mask, device="cpu", dtype=torch.int).unsqueeze(0)
        # else:
        #     action_mask = None

        # TODO: number of agents in each sample should keep consistent, so we save num_agent_actions in list rather than torch.tensor

        samples = Samples(
            sequences=sequences,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=num_actions,
            packed_seq_lens=packed_seq_lens,
            response_length=response_length,
            total_length=total_length,
            num_agent_actions=num_agent_actions,
            labels=pred_labels,
        )

        return samples
