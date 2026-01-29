import srsly
import random
import os
import itertools
from abc import ABC
import math
import itertools
from collections import defaultdict

from typing import Callable, Dict, List, Optional

import torch
from tqdm import tqdm
from datasets import Dataset, load_dataset
import ray
import torch
from dataclasses import dataclass
from ray.util.placement_group import placement_group

from marti.trainers.ppo.actor import ActorModelRayActor
from marti.trainers.ppo.critic import CriticModelRayActor
from marti.models.vllm.engine import create_vllm_engines
from marti.models.ray_launcher import PPORayActorGroup, ReferenceModelRayActor, RewardModelRayActor
from marti.dataset.prompts_loader import PromptDatasetWithLabel
from marti.dataset.sft_loader import SFTDataset
from marti.helpers.common import blending_datasets, get_tokenizer
from marti.helpers.distributed.distributed_sampler import DistributedSampler, ResumableRandomSampler, StratifiedBatchSampler

from marti.worlds.base_world import BaseWorld
from marti.worlds.tool_world import ToolWorld

@dataclass
class Agent:
    # One agent includes actor model (many workers), critic model, reference model, and vllm engines for PPO training
    # The agent can start training after making trajectory samples by itself or with other agents
    def __init__(self,
                agent_id,
                agent_config,
                vllm_engines: List,
                actor_model_group: Optional[PPORayActorGroup] = None,
                critic_model_group: Optional[PPORayActorGroup] = None,
                tokenizer = None,
                generate_kwargs = None,
                is_reasoning_model = False):
        self.agent_id = agent_id
        self.agent_config = agent_config
        self.vllm_engines = vllm_engines
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.is_reasoning_model = is_reasoning_model

    def get_metadata(self):
        return {
            "agent_id": self.agent_id,
            "agent_role": self.agent_config.role,
            "pretrain": self.agent_config.pretrain,
            "llms": self.vllm_engines,
            "tokenizer": self.tokenizer,
            "generate_kwargs": self.generate_kwargs,
            "is_reasoning_model": self.is_reasoning_model
        }

    def save_actor_and_critic_model(self):
        if self.agent_config.is_tuning:
            ray.get(self.actor_model_group.async_save_model())
            if self.agent_config.critic_pretrain and self.agent_config.save_value_network:
                ray.get(self.critic_model_group.async_save_model())

            # save tokenizer
            self.tokenizer.save_pretrained(os.path.join(self.agent_config.save_path, self.agent_id))

@ray.remote
def generate_samples_remote(base_world, chunk_prompts, rank, world_size):
    return base_world.generate_samples(chunk_prompts, rank, world_size)


class BaseController(ABC):
    """
    Load prompt datasets
    Manage experience_maker
    Run global fit function (TODO: use MLFlow)
    Log status for each fit
        including various actors/critics
    """
    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
    
    def load_dataset(self, tokenizer=None):
        # prepare_datasets
        self.prepare_datasets(tokenizer)
        self.num_update_steps_per_episodes = (
            len(self.prompts_dataset) *
            self.args.n_samples_per_prompt // self.args.train_batch_size * self.args.max_epochs
        )
        max_steps = math.ceil(self.args.num_episodes *
                              self.num_update_steps_per_episodes)
        self._max_steps = max_steps

        # TODO: init logging, add MLFlow
        self._init_logging(self.strategy)

    def build(self):
        self._init_tok_kwargs(self.args, self.args.pretrain)
        self.load_dataset(self.tokenizer)
        # prepare agent includes many workers
        self.agent = self._init_agent()

        if self.args.agent_workflow == "base":
            world_class = BaseWorld
        elif self.args.agent_workflow == "tool":
            world_class = ToolWorld

        # create samples maker
        self.world = world_class(
            strategy=self.strategy, agents=[self.agent.get_metadata()])

    def _init_tok_kwargs(self, args, pretrain):
        self.tokenizer = get_tokenizer(
            pretrain, None, "left", self.strategy, use_fast=not self.strategy.args.disable_fast_tokenizer)

        self.generate_kwargs = {
            "do_sample": True,
            "max_new_tokens": args.generate_max_len,
            "max_length": args.max_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def _init_agent(self):
        args = self.args
        strategy = self.strategy

        # init vllm / actor /critic /ref /reward model
        # if colocated, create placement group for actor and ref model explicitly.
        pg = None
        if args.colocate_actor_ref or args.colocate_all_models:
            assert (
                args.actor_num_nodes == args.ref_num_nodes and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
            ), f"num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

            bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.actor_num_nodes * args.actor_num_gpus_per_node)]
            pg = placement_group(bundles, strategy="PACK")

            ray.get(pg.ready())


        # init vLLM engine for text generation
        vllm_engines = None
        if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
            max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            if args.colocate_all_models:
                assert (
                    args.actor_num_nodes * args.actor_num_gpus_per_node
                    == args.vllm_num_engines * args.vllm_tensor_parallel_size
                ), (
                    f"actor_num_nodes * actor_num_gpus_per_node must be equal to "
                    f"vllm_num_engines * vllm_tensor_parallel_size, got {args.actor_num_nodes * args.actor_num_gpus_per_node} "
                    f"and {args.vllm_num_engines * args.vllm_tensor_parallel_size}"
                )

            if args.agent_workflow == "tool":
                from marti.models.vllm.engine_async import LLMRayActorAsync as LLMRayActor
            else:
                from marti.models.vllm.engine import LLMRayActor

            vllm_engines = create_vllm_engines(
                args.vllm_num_engines,
                args.vllm_tensor_parallel_size,
                args.pretrain,
                args.seed,
                args.enable_prefix_caching,
                args.enforce_eager,
                max_len,
                pg,
                args.vllm_gpu_memory_utilization,
                args.vllm_enable_sleep,
                LLMRayActor,
                args.agent_func_path
            )

        actor_model = PPORayActorGroup(
            args.actor_num_nodes,
            args.actor_num_gpus_per_node,
            ActorModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
        )

        # if args.init_kl_coef <= 0:
        #     ref_model = None
        # else:
        ref_model = PPORayActorGroup(
            args.ref_num_nodes,
            args.ref_num_gpus_per_node,
            ReferenceModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
        )

        if not args.colocate_all_models:
            pg = None

        # if colocated, create placement group for critic and reward model explicitly.
        if args.critic_pretrain and args.colocate_critic_reward:
            assert (
                args.critic_num_nodes == args.reward_num_nodes
                and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
            ), f"num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

            bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.critic_num_nodes * args.critic_num_gpus_per_node)]
            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())

        if args.critic_pretrain:
            critic_model = PPORayActorGroup(
                args.critic_num_nodes,
                args.critic_num_gpus_per_node,
                CriticModelRayActor,
                pg=pg,
                num_gpus_per_actor=0.2 if pg else 1,
            )
        else:
            critic_model = None

        # multiple reward models
        if args.reward_pretrain is not None:
            reward_pretrains = args.reward_pretrain.split(",")
            reward_models = []
            for _ in reward_pretrains:
                reward_models.append(
                    PPORayActorGroup(
                        args.reward_num_nodes,
                        args.reward_num_gpus_per_node,
                        RewardModelRayActor,
                        pg=pg,
                        num_gpus_per_actor=0.2 if pg else 1,
                    )
                )
        else:
            reward_models = None

        if ref_model is not None:
            # init reference/reward/actor model
            refs = ref_model.async_init_model_from_pretrained(
                strategy, args.pretrain)
            ray.get(refs)

        if self.args.eos_token is not None:
            eos_token_ids = self.tokenizer(self.args.eos_token, add_special_tokens=False).input_ids
        else:
            eos_token_ids = self.tokenizer.eos_token_id

        refs = actor_model.async_init_model_from_pretrained(
            strategy, args.pretrain, self._max_steps, "actor", eos_token_ids)
        ray.get(refs)

        if args.reward_pretrain is not None:
            for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
                refs = reward_model.async_init_model_from_pretrained(
                    strategy, reward_pretrain)
                ray.get(refs)

        if args.critic_pretrain:
            # critic scheduler initialization depends on max_step, so we have to init critic after actor
            # TODO: use first reward model as critic model
            refs.extend(critic_model.async_init_model_from_pretrained(
                strategy, args.critic_pretrain, self._max_steps))
            ray.get(refs)

        # init actor and critic mdoel
        refs = actor_model.async_init_actor_trainer(
            critic_model, ref_model, reward_models, args.remote_rm_url, vllm_engines=vllm_engines
        )
        ray.get(refs)

        agent = Agent(
            agent_id="agent",
            agent_config=args,
            actor_model_group=actor_model,
            critic_model_group=critic_model,
            vllm_engines=vllm_engines,
            tokenizer=self.tokenizer,
            generate_kwargs=self.generate_kwargs,
            is_reasoning_model=args.is_reasoning_model
        )

        return agent

    def _init_logging(self, strategy):
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.eval_only:
            return

        if self.strategy.args.use_wandb:
            import wandb

            from omegaconf import OmegaConf

            config_dict = OmegaConf.to_container(strategy.args, resolve=True, enum_to_str=True)

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)

            # DEBUG: Check wandb version and GPU monitoring settings
            print(f"[DEBUG GPU MONITORING] wandb version: {wandb.__version__}")
            print(f"[DEBUG GPU MONITORING] Initializing wandb with reinit=True")

            # FIX: Force wandb to detect GPUs
            # Wandb's auto-detection fails in Ray distributed environments,
            # so we manually detect GPUs and pass the count to wandb
            import os
            detected_gpu_count = None
            try:
                import pynvml
                pynvml.nvmlInit()
                detected_gpu_count = pynvml.nvmlDeviceGetCount()
                print(f"[DEBUG GPU MONITORING] Pre-init GPU detection: {detected_gpu_count} GPUs")
                pynvml.nvmlShutdown()
            except Exception as e:
                print(f"[DEBUG GPU MONITORING] Could not pre-detect GPUs: {e}")

            # Initialize wandb with explicit GPU count to enable hardware monitoring
            wandb_settings = wandb.Settings()
            if detected_gpu_count is not None:
                wandb_settings.x_stats_gpu_count = detected_gpu_count
                print(f"[DEBUG GPU MONITORING] Setting x_stats_gpu_count={detected_gpu_count}")

            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=config_dict,
                reinit=True,
                settings=wandb_settings,
            )

            # DEBUG: Check if wandb run was initialized and GPU monitoring settings
            print(f"[DEBUG GPU MONITORING] wandb.run: {wandb.run is not None}")
            if wandb.run:
                print(f"[DEBUG GPU MONITORING] wandb.run.id: {wandb.run.id}")

                # Check critical GPU detection settings
                print(f"[DEBUG GPU MONITORING] GPU detection settings:")
                print(f"  x_disable_stats: {wandb.run.settings.x_disable_stats}")
                print(f"  x_stats_gpu_count: {wandb.run.settings.x_stats_gpu_count}")
                print(f"  x_stats_sampling_interval: {wandb.run.settings.x_stats_sampling_interval}")

                # Check system monitor thread
                import threading
                print(f"[DEBUG GPU MONITORING] Active threads: {threading.active_count()}")
                for thread in threading.enumerate():
                    print(f"  - {thread.name}: {thread.is_alive()}")

                # Check if wandb has system monitor
                if hasattr(wandb.run, '_system_monitor'):
                    print(f"[DEBUG GPU MONITORING] wandb.run._system_monitor exists: {wandb.run._system_monitor}")
                    if wandb.run._system_monitor:
                        print(f"  - Monitor type: {type(wandb.run._system_monitor)}")
                        if hasattr(wandb.run._system_monitor, 'is_alive'):
                            print(f"  - Monitor alive: {wandb.run._system_monitor.is_alive()}")
                        if hasattr(wandb.run._system_monitor, '_stats'):
                            print(f"  - Stats object: {wandb.run._system_monitor._stats}")
                else:
                    print(f"[DEBUG GPU MONITORING] WARNING: wandb.run._system_monitor does not exist!")

                # Check if GPU stats are being collected
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    print(f"[DEBUG GPU MONITORING] pynvml initialized successfully, {device_count} GPUs detected")
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        name = pynvml.nvmlDeviceGetName(handle)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        print(f"[DEBUG GPU MONITORING] GPU {i}: {name}, util={util.gpu}%, mem={util.memory}%")
                    pynvml.nvmlShutdown()
                except Exception as e:
                    print(f"[DEBUG GPU MONITORING] Error checking pynvml: {type(e).__name__}: {e}")

            wandb.define_metric("train/global_step")
            wandb.define_metric(
                "train/*", step_metric="train/global_step", step_sync=True)
            # eval/* metrics also use train/global_step as x-axis
            # (eval/epoch was never logged, causing eval metrics to only show at step 0)
            wandb.define_metric(
                "eval/*", step_metric="train/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            self._tensorboard = SummaryWriter(log_dir=self.strategy.args.use_tensorboard)

    def prepare_datasets(self, tokenizer):
        strategy = self.strategy
        args = self.strategy.args

        # prepare datasets
        # When random sampling is enabled, load full dataset (large max_count) to sample from
        # Otherwise, use max_samples to limit dataset size (existing behavior)
        random_sample_train = getattr(args, 'random_sample_train', False)
        effective_max_count = 999999999 if random_sample_train else args.max_samples

        prompts_data, prompts_dataset_eval = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=effective_max_count,
            return_eval=True,
            train_split=args.prompt_split,
        )

        # Sample dataset: random (new) or sequential (existing behavior)
        if random_sample_train:
            # Random sampling with reproducible seed (uses global pipeline seed)
            import numpy as np
            np.random.seed(args.seed)
            total_size = len(prompts_data)
            sample_size = min(args.max_samples, total_size)
            random_indices = np.random.choice(total_size, size=sample_size, replace=False)
            random_indices = sorted(random_indices.tolist())  # Sort for deterministic iteration order
            prompts_data = prompts_data.select(random_indices)
            print(f"[Random Sampling] Selected {sample_size}/{total_size} samples with seed={args.seed}")
        else:
            # Sequential sampling (existing behavior)
            prompts_data = prompts_data.select(
                range(min(args.max_samples, len(prompts_data))))
        self.prompts_dataset = PromptDatasetWithLabel(
            prompts_data, tokenizer, strategy, input_template=args.input_template, add_prompt_suffix=args.add_prompt_suffix
        )
        self.prompts_dataset_eval = PromptDatasetWithLabel(
            prompts_dataset_eval, tokenizer, strategy, input_template=args.input_template, add_prompt_suffix=args.add_prompt_suffix
        )

        # Save sampled dataset for reproducibility (optional, enabled by save_sampled_dataset=true)
        if random_sample_train and getattr(args, 'save_sampled_dataset', False):
            import json
            save_dir = os.path.dirname(args.save_path)
            os.makedirs(save_dir, exist_ok=True)
            sampled_data_path = os.path.join(save_dir, "sampled_training_data.json")

            # Convert HuggingFace Dataset to list of dicts
            sampled_data = [dict(item) for item in prompts_data]
            with open(sampled_data_path, 'w') as f:
                json.dump(sampled_data, f, indent=2)

            # Save metadata for reproducibility
            metadata = {
                "original_dataset": args.prompt_data,
                "total_available": total_size,
                "sampled_size": sample_size,
                "max_samples": args.max_samples,
                "seed": args.seed,
                "random_indices": random_indices,  # Exact indices sampled
            }
            metadata_path = os.path.join(save_dir, "sampled_training_data_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"✓ Sampled dataset saved: {sampled_data_path}")
            print(f"✓ Metadata saved: {metadata_path}")

        if args.extra_eval_tasks:
            self.prompts_dataset_extra_eval_tasks = {}
            extra_eval_tasks = args.extra_eval_tasks
            for task in extra_eval_tasks:
                task_path = os.path.join(args.extra_eval_dir, f"{task}.json")
                if not os.path.exists(task_path):
                    task_path = os.path.join(args.extra_eval_dir, f"{task}.jsonl")
                if not os.path.exists(task_path):
                    raise ValueError("Only support json or jsonl data")
                
                prompts_data_extra_eval_task = load_dataset("json", data_files=task_path)["train"]
                
                self.prompts_dataset_extra_eval_tasks[task] = PromptDatasetWithLabel(
                    prompts_data_extra_eval_task, tokenizer, strategy, input_template=args.input_template, add_prompt_suffix=args.add_prompt_suffix
                )

        # Choose sampler based on config
        # StratifiedBatchSampler: For DSBench, maintains classification/regression proportions per batch
        # ResumableRandomSampler: Default for MathChat and other tasks without stratification
        use_stratified_sampling = getattr(args, 'stratified_sampling', False)

        if use_stratified_sampling:
            # Use stratified sampling (DSBench: classification vs regression balance)
            stratify_key = getattr(args, 'stratify_key', 'data_type')
            sampler = StratifiedBatchSampler(
                data_source=self.prompts_dataset,
                batch_size=args.rollout_batch_size,
                drop_last=True,
                shuffle=True,
                seed=args.seed,
                stratify_key=stratify_key
            )
            # Log stratification info
            if hasattr(sampler, 'stratification_enabled') and sampler.stratification_enabled:
                print(f"[Stratified Sampling] Enabled with key='{stratify_key}'")
                print(f"  Strata: {dict(sampler.proportions)}")
                print(f"  Per-batch counts: {sampler.batch_counts}")
            else:
                print(f"[Stratified Sampling] Requested but disabled (insufficient strata)")
        else:
            # Default: random sampling (MathChat, etc.)
            sampler = ResumableRandomSampler(
                data_source=self.prompts_dataset,
                batch_size=args.rollout_batch_size,
                drop_last=True,
                shuffle=True,
                seed=args.seed
            )

        # Use custom collate function to handle None labels (e.g., DSBench tasks)
        from marti.dataset.prompts_loader import collate_with_none_labels

        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset, args.rollout_batch_size, True, True,
            sampler=sampler,
            collate_fn=collate_with_none_labels
        )

    def load_checkpoint_steps(self):
        args = self.args
        num_update_steps_per_episodes = (
            len(self.prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
        )
        consumed_samples = 0
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            states = torch.load(os.path.join(ckpt_path, "step_states.bin"))
            consumed_samples = states["consumed_samples"]
            print(f"Loaded the checkpoint: {ckpt_path}, consumed_samples: {consumed_samples}")

        return num_update_steps_per_episodes, consumed_samples

    def run(self):
        # update steps if load checkpoints
        num_update_steps_per_episodes, consumed_samples = self.load_checkpoint_steps()
        
        # start fitting
        self.fit(
            consumed_samples=consumed_samples,
            num_update_steps_per_episodes=num_update_steps_per_episodes
        )
        
        if not self.args.eval_only:
            # save actor and critic workers in agent
            self.agent.save_actor_and_critic_model()

    def clean_template(self, prompt):
        """
        clean the template
        '<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nhow are you?<|im_end|>\n<|im_start|>assistant\n'
        
        '<｜begin▁of▁sentence｜><｜User｜>how are you?<｜Assistant｜>'
        """
        if "<|im_start|>" in prompt:
            prompt = prompt.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0]
        elif "<｜begin▁of▁sentence｜>" in prompt:
            prompt = prompt.split("<｜begin▁of▁sentence｜><｜User｜>")[-1].split("<｜Assistant｜>")[0]
        return prompt

    def build_sft_sample_list(self, data):
        """
        Input: {
        "sample": samples_list,
        "prompt": all_prompts,
        "output": all_pred_outputs,
        "label": all_pred_labels
        }
        
        Output: List[dict]
            {input_key: "", label_key:""}
        """
        all_prompts = data["prompt"]
        all_outputs = data["output"]
        all_labels = data["label"]
        assert len(all_prompts) == len(all_outputs) == len(all_labels)
        n_samples_per_prompt = self.args.n_samples_per_prompt
        sample_list = []
        for idx in range(0, len(all_prompts), n_samples_per_prompt):
            tmp_sample = []
            for i in range(n_samples_per_prompt):
                prompt = all_prompts[idx + i]
                output = all_outputs[idx + i]
                label = all_labels[idx + i]
                if label == 1:
                    prompt = self.clean_template(prompt)
                    tmp_sample.append({
                        self.args.input_key: prompt,
                        self.args.output_key: output,
                    })
            if len(tmp_sample) <= n_samples_per_prompt // 2:
                # sample_list.append(random.choice(tmp_sample))
                sample_list.extend(tmp_sample)
        return sample_list

    def build_sft_dataset(self, data_list):
        pretrain_data = Dataset.from_list(data_list)
        pretrain_max_len = self.strategy.args.max_len if self.strategy.args.max_len else self.strategy.args.prompt_max_len + \
                self.strategy.args.generate_max_len
        pretrain_dataset = SFTDataset(
            pretrain_data,
            self.tokenizer,
            pretrain_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1
        )
        assert len(pretrain_dataset) > 0, f"{len(pretrain_dataset)} samples are generated."
        return pretrain_dataset
    def generate_shared_samples(self, rand_prompts):
        world_size = self.agent.actor_model_group.world_size
        num_engines  = len(self.agent.vllm_engines)
        micro_bs     = self.args.micro_rollout_batch_size

        any_key = next(iter(rand_prompts.keys()))
        length = len(rand_prompts[any_key])

        # When the number of prompts in a batch is smaller than the actor world
        # size or when prompts cannot be evenly distributed, some ranks may
        # receive an empty list after chunking. This leads to no experiences
        # generated on those ranks and later failures when processing an empty
        # experience list.  Duplicate prompts when necessary and use an even
        # splitting strategy so that each rank obtains at least one element.
        if length < world_size:
            repeat = (world_size + length - 1) // length
            for key, data_list in rand_prompts.items():
                rand_prompts[key] = (data_list * repeat)[:world_size]
            length = len(rand_prompts[any_key])

        def even_chunks(lst, n):
            k, m = divmod(len(lst), n)
            chunks = []
            start = 0
            for i in range(n):
                end = start + k + (1 if i < m else 0)
                chunks.append(lst[start:end])
                start = end
            return chunks

        chunked = [dict() for _ in range(world_size)]
        for key, data_list in rand_prompts.items():
            splits = even_chunks(data_list, world_size)
            for i in range(world_size):
                chunked[i][key] = splits[i]

        # 2. If the number of engines is less than world_size, divide evenly.
        if num_engines < world_size:
            # Divide world_size chunks into num_engines sublists
            engine_sublists = even_chunks(chunked, num_engines)

            # Merge each dict in the sublist
            engine_inputs = []
            for sublist in engine_sublists:
                merged = defaultdict(list)
                for d in sublist:
                    for k, vals in d.items():
                        merged[k].extend(vals)
                engine_inputs.append(merged)

            # parallel invocation
            refs = [
                generate_samples_remote.remote(self.world, engine_inputs[i], i, num_engines)
                for i in range(num_engines)
            ]
            engine_results = ray.get(refs)

            # 3. Unpack to world_size pieces
            all_results = []
            for sublist, res in zip(engine_sublists, engine_results):
                num_ranks = len(sublist)
                # Firstly, split each "list" field into num_ranks parts.
                split_fields = {
                    k: even_chunks(v, num_ranks)
                    for k, v in res.items() if isinstance(v, list)
                }
                # Assemble the results of each rank
                for i in range(num_ranks):
                    out = {}
                    for k, v in res.items():
                        out[k] = split_fields[k][i] if isinstance(v, list) else v
                    all_results.append(out)

        else:
            # The number of engines is greater than or equal to world_size, directly one-to-one
            refs = [
                generate_samples_remote.remote(self.world, chunked[r], r, world_size)
                for r in range(world_size)
            ]
            all_results = ray.get(refs)

        shared_data_refs = [None for _ in range(world_size)]
        sft_samples = []

        # Ensure all workers have the same number of samples, avoid stucking
        # Especially when we filter samples by rewards
        # TODO: Maybe we should decrease train_batch_size (acculate_grads_every_n_steps)
        # Reference: https://github.com/OpenRLHF/OpenRLHF/issues/717
        if self.args.filter_samples_by_reward:
            min_num_samples = min([len(x["sample"]) for x in all_results])
            num_full_batches = (min_num_samples * self.args.micro_rollout_batch_size) // self.args.train_batch_size
            num_elements_to_keep = (num_full_batches * self.args.train_batch_size) // self.args.micro_rollout_batch_size
        else:
            num_elements_to_keep = -1

        # create samples for each actor worker
        for rank in range(world_size):

            shared_data = all_results[rank]["sample"]
            if self.args.filter_samples_by_reward:
                shared_data = shared_data[:num_elements_to_keep]

            if self.args.training_mode in ["sft", "both", "mix"]:
                sft_samples.extend(self.build_sft_sample_list(shared_data))

            shared_ref = ray.put(shared_data)
            shared_data_refs[rank] = shared_ref
        
        if len(sft_samples) > 0:
            for sample in random.sample(sft_samples, 3):
                print(sample)
                print("="*20)
            return shared_data_refs, ray.put(self.build_sft_dataset(sft_samples)), num_elements_to_keep
        else:
            return shared_data_refs, None, num_elements_to_keep

    def step(self, rand_prompts, episode, steps, pbar):
        import os
        import logging
        debug_enabled = os.environ.get('DSBENCH_DEBUG', '0') == '1'
        logger = logging.getLogger(__name__)

        if debug_enabled:
            logger.info(f"[DEBUG] step() START - episode={episode}, steps={steps}")
            logger.info(f"[DEBUG] vllm_enable_sleep={self.args.vllm_enable_sleep}")

        if self.args.vllm_enable_sleep:
            from marti.models.vllm.engine import batch_vllm_engine_call
            if debug_enabled:
                logger.info(f"[DEBUG] BEFORE vLLM wake_up - num_engines={len(self.agent.vllm_engines)}")
            batch_vllm_engine_call(self.agent.vllm_engines, "wake_up")
            if debug_enabled:
                logger.info(f"[DEBUG] AFTER vLLM wake_up - SUCCESS")

        # make shared samples refs
        if debug_enabled:
            logger.info(f"[DEBUG] BEFORE generate_shared_samples - num_prompts={len(next(iter(rand_prompts.values())))}")
        shared_data_refs, sft_dataset, num_elements_to_keep = self.generate_shared_samples(
            rand_prompts=rand_prompts)
        if debug_enabled:
            logger.info(f"[DEBUG] AFTER generate_shared_samples - num_elements_to_keep={num_elements_to_keep}")

        if self.args.vllm_enable_sleep:
            from marti.models.vllm.engine import batch_vllm_engine_call
            if debug_enabled:
                logger.info(f"[DEBUG] BEFORE vLLM sleep - num_engines={len(self.agent.vllm_engines)}")
            batch_vllm_engine_call(self.agent.vllm_engines, "sleep")
            if debug_enabled:
                logger.info(f"[DEBUG] AFTER vLLM sleep - SUCCESS")

        # if there is not enough samples, then we generate samples again
        if self.args.filter_samples_by_reward and num_elements_to_keep == 0:
            if debug_enabled:
                logger.info(f"[DEBUG] Skipping training - num_elements_to_keep=0")
            return None

        # start training actor models (workers)
        if debug_enabled:
            logger.info(f"[DEBUG] BEFORE async_fit_actor_model - steps={steps}")
        refs = self.agent.actor_model_group.async_fit_actor_model(
            steps, shared_data_refs, sft_dataset)
        if debug_enabled:
            logger.info(f"[DEBUG] AFTER async_fit_actor_model - got {len(refs)} refs")
            logger.info(f"[DEBUG] BEFORE ray.get - waiting for training to complete")
        results = ray.get(refs)
        if debug_enabled:
            logger.info(f"[DEBUG] AFTER ray.get - training completed, got {len(results)} results")

        # find result from rank 0
        for result in results:
            logs_dict, is_rank_0, perf_stats = result["status"], result["is_rank_0"], result["perf_stats"]
            if is_rank_0:
                break

        return [logs_dict, perf_stats]

    def fit(self,
            consumed_samples=0,
            num_update_steps_per_episodes=1,
        ) -> None:
        args = self.args
        
        num_rollouts_per_episodes = (
            num_update_steps_per_episodes
            * args.train_batch_size
            // args.max_epochs
            // args.rollout_batch_size
            // args.n_samples_per_prompt
        )

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (
            num_rollouts_per_episodes * args.rollout_batch_size)

        # Eval before training (use step=0 to avoid overwriting step=1 eval after first training iteration)
        if args.eval_before_training or args.eval_only:
            self.save_logs(args, 0, self.evaluate_tasks(0), None)

        if args.eval_only:
            return

        for episode in range(start_episode, args.num_episodes):
            # Both ResumableRandomSampler and StratifiedBatchSampler support set_epoch
            if isinstance(self.prompts_dataloader.sampler, (ResumableRandomSampler, StratifiedBatchSampler)):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 #if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]"
            )

            # pass generated samples to each actor worker
            # FIX: Convert iterator to list to prevent premature batch fetching
            # Issue: Python's iterator protocol allows __next__() to be called while
            # step() is executing, causing parallel rollout before training completes
            num_batches = len(self.prompts_dataloader)
            total_rollout_steps = args.num_episodes * num_batches
            prompt_batches = list(self.prompts_dataloader)

            for batch_idx, rand_prompts in enumerate(prompt_batches):
                # Clear logging to distinguish ROLLOUT STEPS from gradient updates
                num_prompts = len(next(iter(rand_prompts.values())))
                print(f"\n{'='*80}")
                print(f"[ROLLOUT STEP {steps + 1}/{total_rollout_steps}] Episode {episode + 1}/{args.num_episodes}, Batch {batch_idx + 1}/{num_batches}")
                print(f"  → Generating {num_prompts} prompts × {args.n_samples_per_prompt} samples = {num_prompts * args.n_samples_per_prompt} trajectories")
                print(f"{'='*80}")

                step_result = self.step(rand_prompts, episode, steps, pbar)
                if step_result is None:
                    continue
                
                logs_dict, perf_stats = step_result

                if steps % args.eval_steps == 0:
                    logs_dict.update(self.evaluate_tasks(steps))

                self.save_logs(args, steps, logs_dict, perf_stats)

                # Save rollout accuracy to JSON (same format as eval results)
                if "_rollout_metadata" in logs_dict and logs_dict["_rollout_metadata"]["accuracy"] is not None:
                    rollout_results = {
                        "accuracy": logs_dict["_rollout_metadata"]["accuracy"],
                        "correct": logs_dict["_rollout_metadata"]["correct"],
                        "total": logs_dict["_rollout_metadata"]["total"],
                        "samples_per_prompt": logs_dict["_rollout_metadata"].get("samples_per_prompt", 0)
                    }
                    output_path = os.path.join(args.save_path, "eval_results")
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    srsly.write_json(
                        os.path.join(output_path, f"global_step{steps}_train.json"),
                        rollout_results
                    )

                pbar.set_postfix(logs_dict)
                pbar.update()
                steps = steps + 1
                print(f"[ROLLOUT STEP {steps}/{total_rollout_steps}] ✓ Complete\n")

        # Final evaluation if the last training step is an eval step
        final_training_step = steps - 1  # steps has been incremented, so get last actual step
        if final_training_step > 0 and final_training_step % args.eval_steps == 0:
            print(f"\n[Final Eval] Running evaluation at step {final_training_step}")
            final_eval_results = self.evaluate_tasks(final_training_step)
            self.save_logs(args, final_training_step, final_eval_results, None)

        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()

    def list_of_dicts_to_dict_of_lists(self, list_of_dicts):
        result = defaultdict(list)
        for d in list_of_dicts:
            for key, value in d.items():
                result[key].append(value)
        return dict(result)

    def evaluate_samples(self, steps, eval_dataset, task=""):
        data_list = eval_dataset.get_all_prompts()
        all_prompts = self.list_of_dicts_to_dict_of_lists(data_list)
        results = self.world.evaluate_samples(all_prompts)
        output_path = os.path.join(self.args.save_path, "eval_results")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        srsly.write_json(os.path.join(output_path, f"global_steps{steps}{task}.json"), results)
        # Return accuracy and aggregated_metrics (for DSBench per-metric logging)
        return {
            "accuracy": results["accuracy"],
            "aggregated_metrics": results.get("aggregated_metrics", {})
        }

    def set_agent_vllm_engine(self, command):
        if self.args.vllm_enable_sleep:
            from marti.models.vllm.engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.agent.vllm_engines, command)

    def clear_prefix_cache(self):
        """Clear prefix cache from all vLLM engines after eval to prevent cache pollution."""
        if self.agent.vllm_engines is None:
            return
        import ray
        cache_reset_refs = []
        for engine in self.agent.vllm_engines:
            cache_reset_refs.append(engine.reset_prefix_cache.remote())
        ray.get(cache_reset_refs)

    def evaluate_tasks(self, steps):
        self.set_agent_vllm_engine("wake_up")
        eval_result = self.evaluate_samples(steps, self.prompts_dataset_eval)
        acc_dict = {"eval/reward": eval_result["accuracy"]}
        # Add aggregated outcome_metrics to wandb (DSBench per-metric averages)
        for metric_name, value in eval_result.get("aggregated_metrics", {}).items():
            acc_dict[f"eval/{metric_name}"] = value

        if self.args.extra_eval_tasks:
            for task in self.prompts_dataset_extra_eval_tasks:
                task_result = self.evaluate_samples(steps, self.prompts_dataset_extra_eval_tasks[task], task)
                acc_dict[f"eval/acc_{task}"] = task_result["accuracy"]
                # Add per-task aggregated metrics
                for metric_name, value in task_result.get("aggregated_metrics", {}).items():
                    acc_dict[f"eval/{task}/{metric_name}"] = value

        self.set_agent_vllm_engine("sleep")
        # Clear prefix cache after eval to prevent stale cache entries from affecting
        # the next training rollout (fixes count=32 issue on post-eval steps)
        self.clear_prefix_cache()
        return acc_dict

    def save_logs(self, args, global_step, logs_dict, perf_stats):
        # DEBUG: Track when save_logs is called
        eval_keys = [k for k in logs_dict.keys() if 'eval/' in k]
        if eval_keys:
            print(f"[DEBUG save_logs] global_step={global_step}, eval_keys={eval_keys}, values={[(k, logs_dict[k]) for k in eval_keys]}")
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None:
                logs = {
                    ( k if "eval/" in k else f"train/{k}"): v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                if perf_stats is not None:
                    logs.update(
                        {f"perf/experience_maker/{k}": v for k, v in perf_stats.items()})

                # DEBUG: Print what's being logged to wandb
                print(f"[DEBUG GPU MONITORING] Step {global_step}: Logging to wandb")
                print(f"[DEBUG GPU MONITORING] logs_dict keys: {list(logs_dict.keys())}")
                print(f"[DEBUG GPU MONITORING] perf_stats: {perf_stats}")
                print(f"[DEBUG GPU MONITORING] Total keys being logged: {list(logs.keys())}")

                # DEBUG: Check if wandb is tracking GPU metrics automatically
                if hasattr(self._wandb.run, '_system_monitor') and self._wandb.run._system_monitor:
                    monitor = self._wandb.run._system_monitor
                    print(f"[DEBUG GPU MONITORING] wandb system monitor is running: {monitor.is_alive() if hasattr(monitor, 'is_alive') else 'unknown'}")

                    # Try to access the stats that wandb is collecting
                    if hasattr(monitor, '_stats'):
                        print(f"[DEBUG GPU MONITORING] Monitor stats object: {type(monitor._stats)}")
                    if hasattr(monitor, '_shutdown'):
                        print(f"[DEBUG GPU MONITORING] Monitor shutdown flag: {monitor._shutdown}")

                    # Check what the last collected stats were
                    try:
                        # Try to get stats from wandb's internal stats object
                        if hasattr(self._wandb.run, '_stats'):
                            run_stats = self._wandb.run._stats
                            print(f"[DEBUG GPU MONITORING] wandb.run._stats type: {type(run_stats)}")
                            if hasattr(run_stats, 'stats'):
                                recent_stats = run_stats.stats()
                                print(f"[DEBUG GPU MONITORING] Recent stats keys: {list(recent_stats.keys()) if isinstance(recent_stats, dict) else 'not a dict'}")
                                # Check for GPU-related keys
                                gpu_keys = [k for k in (recent_stats.keys() if isinstance(recent_stats, dict) else []) if 'gpu' in k.lower()]
                                print(f"[DEBUG GPU MONITORING] GPU-related stat keys: {gpu_keys}")
                    except Exception as e:
                        print(f"[DEBUG GPU MONITORING] Error accessing wandb stats: {type(e).__name__}: {e}")
                else:
                    print(f"[DEBUG GPU MONITORING] WARNING: wandb system monitor not found or not running!")

                # DEBUG: Try to manually log GPU stats to see if they show up
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    for i in range(min(device_count, 8)):  # Limit to 8 GPUs
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        # Add manual GPU metrics to logs
                        logs[f"system/gpu.{i}.gpu"] = util.gpu
                        logs[f"system/gpu.{i}.memory"] = util.memory
                        logs[f"system/gpu.{i}.memoryAllocated"] = (mem_info.used / mem_info.total) * 100
                        print(f"[DEBUG GPU MONITORING] Manually logging GPU {i}: util={util.gpu}%, mem={util.memory}%")
                    pynvml.nvmlShutdown()
                except Exception as e:
                    print(f"[DEBUG GPU MONITORING] Error manually logging GPU stats: {type(e).__name__}: {e}")

                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None:
                for k, v in logs_dict.items():
                    # save eval logs to eval folder
                    if len(k.split("/")) == 2:
                        split, k = k.split("/")
                    else:
                        split = "train"
                    self._tensorboard.add_scalar(
                        f"{split}/{k}", v, global_step)
                if perf_stats is not None:
                    for k, v in perf_stats.items():
                        self._tensorboard.add_scalar(
                            f"perf/experience_maker/{k}", v, global_step)
