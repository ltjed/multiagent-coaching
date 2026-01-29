import ray
import hydra
from omegaconf import DictConfig, OmegaConf
from vllm import SamplingParams
import torch
import random
import numpy as np
from typing import Union, Callable, Dict, List
from tqdm import tqdm
import json
from copy import deepcopy
import srsly

from marti.helpers.common import get_strategy, blending_datasets, get_tokenizer
from marti.models.vllm.engine import create_vllm_engines
from marti.models.openai import OpenAIModel, FakeTokenizer
from marti.worlds.base_world import BaseWorld, Samples
from marti.agents.multi_agent import MAGraph, get_kwargs
from marti.verifiers.qwen.qwen_eval import qwen_reward_fn, majority_vote
from marti.dataset.prompts_loader import PromptDatasetWithLabel
from marti.helpers.distributed.distributed_sampler import DistributedSampler, ResumableRandomSampler

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _validate_config(cfg: DictConfig):
    actor_world_size = cfg.actor_num_nodes * cfg.actor_num_gpus_per_node

    assert (
        actor_world_size & (actor_world_size - 1)
    ) == 0, f"actor_world_size must be power of 2, got {actor_world_size}"

    if cfg.critic_pretrain:
        critic_world_size = cfg.critic_num_nodes * cfg.critic_num_gpus_per_node
        assert (
            critic_world_size & (critic_world_size - 1)
        ) == 0, f"critic_world_size must be power of 2, got {critic_world_size}"
        assert (
            actor_world_size % critic_world_size == 0
        ), f"actor_world_size must be divisible by critic_world_size, got {actor_world_size} and {critic_world_size}"

    assert cfg.zero_stage != 3 or cfg.vllm_num_engines > 0, f"ZeRO-3 is only supported when vLLM enabled"

def _rationalize_config(cfg: DictConfig):
    if cfg.advantage_estimator not in ["gae"]:
        cfg.critic_pretrain = None
    elif cfg.critic_pretrain is None:
        if cfg.reward_pretrain is not None:
            cfg.critic_pretrain = cfg.reward_pretrain.split(",")[0]
        else:
            cfg.critic_pretrain = cfg.pretrain

    if cfg.advantage_estimator == "rloo":
        assert cfg.n_samples_per_prompt > 1, "RLOO requires n_samples_per_prompt > 1"

    if cfg.remote_rm_url:
        cfg.remote_rm_url = cfg.remote_rm_url.split(",")

    if cfg.vllm_num_engines >= 1 and cfg.enable_prefix_caching:
        import vllm
        if vllm.__version__ < "0.7.0":
            cfg.enable_prefix_caching = False
            print("[Warning] Disable prefix cache because vLLM updates weights without updating the old KV Cache for vLLM version below 0.7.0.")

    if cfg.input_template and "{}" not in cfg.input_template:
        print("[Warning] {} not in cfg.input_template, set to None")
        cfg.input_template = None

    if cfg.input_template and "\\n" in cfg.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if cfg.packing_samples:
        if not cfg.flash_attn:
            print(
                "[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
            cfg.flash_attn = True
        assert cfg.vllm_num_engines > 0, "Only support `--packing_samples` with vLLM."
        assert not cfg.pretrain_data, "`--pretrain_data` is not supported with `--packing_samples` yet."

    return cfg


class MultiAgentWorld(BaseWorld):
    def __init__(self, strategy, agents):
        super().__init__(strategy, agents)

    def run_collect(self, all_prompts: List[str], all_labels: List[str]):
        args = self.strategy.args
        print(args.chat_template)
        print(args.agents)

        kwargs = get_kwargs(args)

        graph = MAGraph(
            agents=self.agents,
            agent_ids=kwargs['agent_ids'],
            agent_roles=kwargs['agent_roles'],
            agent_workflow=args.agent_workflow,
            prompt=kwargs['prompt'],
            spatial_adj_mats=kwargs['spatial_adj_mats'],
            temporal_adj_mats=kwargs['temporal_adj_mats'],
            sampling_params=kwargs['sampling_params'],
            node_kwargs=kwargs['node_kwargs'] if 'node_kwargs' in kwargs else None,
        )
        history = graph.run(all_prompts, num_rounds=args.workflow_args.num_rounds)

        # eval the results
        history_per_problem = {problem: [] for problem in all_prompts}
        problem2id = {problem: all_prompts.index(problem) for problem in all_prompts}
        for problem in all_prompts:
            problem_id = problem2id[problem]
            for node_id, node_history in enumerate(history):
                temp_history = []
                for round_id, round_history in enumerate(node_history):
                    inputs = round_history['inputs']
                    outputs = round_history['outputs']
                    rewards = []
                    for output, label in zip(outputs, all_labels):
                        if isinstance(output, str):
                            reward = qwen_reward_fn(output, label)
                        elif isinstance(output, dict):
                            if isinstance(output['output'], str):
                                reward = qwen_reward_fn(output['output'], label)
                            elif isinstance(output['output'], list):
                                reward = majority_vote(output['output'], label)
                        rewards.append(reward)
                    temp_history.append({
                        'agent_id': round_history['agent_id'],
                        'agent_role': round_history['agent_role'],
                        'pretrain': round_history['pretrain'],
                        'turn_id': round_history['turn_id'],
                        'inputs': inputs[problem_id],
                        'outputs': outputs[problem_id],
                        'rewards': rewards[problem_id],
                        'spatial_predecessors': round_history['spatial_predecessors'],
                        'temporal_predecessors': round_history['temporal_predecessors'],
                    })
                if len(temp_history) == 1:
                    temp_history = temp_history[0]
                history_per_problem[problem].append(temp_history)

        return history_per_problem

    @torch.no_grad()
    def generate_samples(self, all_prompts: Union[List[str], dict], rank=0, world_size=8, **kwargs) -> List[Samples]:
        args = self.strategy.args

        all_prompts, all_labels = all_prompts["prompt"], all_prompts["label"]
        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])

        if len(all_prompts):
            history = self.run_collect(all_prompts=all_prompts, all_labels=all_labels)
        else:
            history = {}

        return history


@ray.remote
def generate_samples_remote(samples_maker, chunk_prompts, rank, world_size):
    history = samples_maker.generate_samples(chunk_prompts, rank, world_size)
    return history


def generate_shared_samples(samples_maker, rand_prompts, world_size):
    any_key = next(iter(rand_prompts.keys()))
    length = len(rand_prompts[any_key])
    chunk_size = (length + world_size - 1) // world_size
    chunked = [dict() for _ in range(world_size)]
    for key, data_list in rand_prompts.items():
        for i in range(world_size):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, length)
            sub_slice = data_list[start_idx:end_idx]
            chunked[i][key] = sub_slice

    all_refs = []
    for rank in range(world_size):
        samples_ref = generate_samples_remote.remote(samples_maker, chunked[rank], rank, world_size)
        all_refs.append(samples_ref)

    all_results = {}
    for r in ray.get(all_refs):
        all_results.update(r)

    return all_results


@hydra.main(config_path="../configs", config_name="default.yaml", version_base=None)
def train(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    for key, value in cfg.default_agent.items():
        cfg[key] = value

    _rationalize_config(cfg)
    _validate_config(cfg)
    print(OmegaConf.to_yaml(cfg))

    # configure strategy
    strategy = get_strategy(cfg)

    agent2config = {}
    for agent_dict in cfg.agents:
        for agent_name, agent_info in agent_dict.items():
            for key, value in cfg.default_agent.items():
                if key not in agent_info:
                    agent_info[key] = value
            agent2config[agent_name] = agent_info

    agent_list = []
    llm_dict = {}
    seed = 0
    for agent_name, agent_config in agent2config.items():
        if 'gpt' in agent_config.pretrain.lower():
            agent_llms = [
                OpenAIModel.remote(api_key=cfg.api_key, base_url=cfg.api_base_url, config={"model_name": agent_config.pretrain})
            ]
            tokenizer = FakeTokenizer()
        else:
            max_len = None
            if agent_config.max_len is not None:
                max_len = agent_config.max_len
            elif agent_config.prompt_max_len is not None and agent_config.generate_max_len is not None:
                max_len = agent_config.prompt_max_len + agent_config.generate_max_len

            if agent_config.pretrain in llm_dict:
                agent_llms = llm_dict[agent_config.pretrain]
            else:
                agent_llms = create_vllm_engines(
                    agent_config.vllm_num_engines,
                    agent_config.vllm_tensor_parallel_size,
                    agent_config.pretrain,
                    seed,
                    agent_config.enable_prefix_caching,
                    agent_config.enforce_eager,
                    max_len,
                    None,  # shared_pg
                    agent_config.vllm_gpu_memory_utilization,
                    getattr(agent_config, "vllm_enable_sleep", False),
                )
                llm_dict[agent_config.pretrain] = agent_llms
                seed += agent_config.vllm_num_engines

            # Create tokenizer
            tokenizer = get_tokenizer(
                agent_config.pretrain, None, "left", strategy, use_fast=not agent_config.disable_fast_tokenizer)

        generate_kwargs = {
            "do_sample": True,
            "max_new_tokens": agent_config.generate_max_len,
            "max_length": agent_config.max_len,
            "temperature": agent_config.temperature,
            "top_p": agent_config.top_p,
            "pad_token_id": tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else None,
            "eos_token_id": tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None,
        }

        sampling_params = SamplingParams(
            n=agent_config.n_samples_per_prompt,
            temperature=agent_config.temperature,
            top_p=agent_config.top_p,
            top_k=agent_config.get("top_k", -1),
            max_tokens=agent_config.generate_max_len,
            min_tokens=agent_config.get("min_new_tokens", 1),
            skip_special_tokens=agent_config.get("skip_special_tokens", False),
        )

        agent = {
            "agent_name": agent_name,
            "pretrain": agent_config.pretrain,
            "llms": agent_llms,
            "tokenizer": tokenizer,
            "generate_kwargs": generate_kwargs,
            "sampling_params": sampling_params,
            "is_reasoning_model": agent_config.get("is_reasoning_model", False),
            "enable_thinking": agent_config.get("enable_thinking", agent_config.get("is_reasoning_model", False)),
            "code_execution": agent_config.get("code_execution", False),
        }
        agent_list.append(agent)

    args = strategy.args

    # prepare datasets
    prompts_data, prompts_data_eval = blending_datasets(
        args.prompt_data,
        str(args.prompt_data_probs),
        strategy,
        args.seed,
        max_count=args.max_samples,
        return_eval=True,
        train_split=args.prompt_split,
    )
    prompts_data = prompts_data.select(
        range(min(args.max_samples, len(prompts_data))))
    prompts_dataset = PromptDatasetWithLabel(
        prompts_data, None, strategy, input_template=args.input_template, add_prompt_suffix=args.add_prompt_suffix
    )

    sampler = ResumableRandomSampler(
        data_source=prompts_dataset,
        batch_size=args.rollout_batch_size,
        drop_last=False,
        shuffle=False,
        seed=args.seed
    )

    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset, args.rollout_batch_size, True, shuffle=False,
        sampler=sampler, drop_last=False
    )

    # create samples maker
    sample_maker_class = MultiAgentWorld
    samples_maker = sample_maker_class(
        strategy=strategy, agents=agent_list)

    # Restore step and start_epoch
    steps = 1
    start_episode = 0
    consumed_samples = 0
    all_histories = {}
    final_histories = []
    question2idx = {}
    idx2question = []
    for episode in range(start_episode, args.num_episodes):
        if isinstance(prompts_dataloader.sampler, ResumableRandomSampler):
            prompts_dataloader.sampler.set_epoch(
                episode, consumed_samples=0
            )
        pbar = tqdm(
            range(prompts_dataloader.__len__()),
            desc=f"Episode [{episode + 1}/{args.num_episodes}]"
        )

        # pass generated samples to each actor worker
        for rand_prompts in prompts_dataloader:
            # make shared samples refs
            history = generate_shared_samples(samples_maker, rand_prompts=rand_prompts, world_size=args.vllm_num_engines)
            for idx, question in enumerate(rand_prompts["prompt"]):
                if question not in idx2question:
                    idx2question.append(question)
                    question2idx[question] = len(question2idx)
                    all_histories[question] = []
                all_histories[question].append(history[question])

            pbar.update()
            steps = steps + 1

    for idx, question in enumerate(idx2question):
        final_histories.append(all_histories[question][0])

    output_path = os.path.join(args.save_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    srsly.write_json(os.path.join(output_path, f"results.json"), final_histories)


if __name__ == "__main__":
    train()
