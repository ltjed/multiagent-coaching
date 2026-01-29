import math
from itertools import groupby, accumulate

import os
import torch

from tqdm import tqdm
from typing import List
from torch.nn import functional as F

import ray
from transformers.trainer import get_scheduler

from marti.models.actor import Actor
from marti.trainers.ppo.trainer import PPOTrainer
from marti.helpers.distributed.deepspeed import DeepspeedStrategy
from marti.models.ray_launcher import BasePPORole
from marti.models.model_utils import unpacking_samples
from marti.models.loss import GPTLMLoss, PolicyLoss, ValueLoss
from marti.worlds.base_world import Samples

from torch.utils.data import Dataset

class PrimeSamplesDataset(Dataset):
    def __init__(self, prime_samples_list: List[Samples]):
        self.samples = prime_samples_list
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
    
class PrimePPOTrainer(PPOTrainer):
    def __init__(self, *args,
                 credit_model: Actor,
                 credit_optim,
                 credit_scheduler,
                 credit_beta: float = 0.05,
                 credit_granularity: str = "token",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.credit_model = credit_model
        self.credit_optim = credit_optim
        self.credit_scheduler = credit_scheduler
        self.credit_beta = credit_beta
        self.credit_granularity = credit_granularity
        self.ptx_loss_fn = GPTLMLoss()

        self.packing_samples = getattr(self.args, "packing_samples", False)

    def ppo_train(self, credit_samples: List[Samples]):
        credit_dataset = PrimeSamplesDataset(credit_samples)

        # get packed PrimeSamples
        dataloader = self.strategy.setup_dataloader(
            credit_dataset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            collate_fn=lambda x: x[0]
        )

        device = torch.cuda.current_device()

        status_list = []
        status_mean = []
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0()
            )

            for samples in pbar:
                samples.to_device(device)
                
                if hasattr(self.args, "num_agents"):
                    assert len(samples.info) == self.args.num_agents, f"num_agents should be provided in args, but got {len(samples.info)} agents"
                
                samples.base_action_log_probs = samples.base_action_log_probs.to(device)
                # self.strategy.print(samples)
                status = self.train_step_prime(samples)

                status = self.strategy.all_reduce(status)

                status_list.append(status)
                pbar.set_postfix(status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def train_step_prime(self, data: Samples):
        self.credit_model.train()
        if not self.packing_samples:
            sequences = torch.cat(data.sequences, dim=0).unsqueeze(0)
            base_action_log_probs = torch.cat(
                data.base_action_log_probs, dim=0).unsqueeze(0)
            attention_mask = torch.cat(data.attention_mask, dim=0).unsqueeze(0)
            num_actions = data.num_actions
            packed_seq_lens = data.packed_seq_lens
            # response_length = data.response_length
            num_agent_actions = data.num_agent_actions
            labels = data.labels
        else:
            sequences = data.sequences
            base_action_log_probs = data.base_action_log_probs
            attention_mask = data.attention_mask
            num_actions = data.num_actions
            packed_seq_lens = data.packed_seq_lens
            # response_length = data.response_length
            num_agent_actions = data.num_agent_actions
            labels = data.labels

        assert self.packing_samples is True, "packing_samples is only supported yet"

        # get log_probs for action parts
        policy_log_probs, output = self.credit_model(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens
        )

        agent_level_scores, q = self.compute_implicit_reward(
            log_prob=policy_log_probs,
            ref_log_prob=base_action_log_probs,
            num_actions=num_actions,
            num_agent_actions=num_agent_actions,
            is_avg_agents=True
        )
        
        # \sigma(\beta * sum)
        probs = (torch.stack(agent_level_scores) * self.credit_beta).sigmoid()  # [batch_size]
        labels = (labels == 1).to(probs.dtype)
        # print("Probs", probs)
        # print("Labels", labels)
        # print()
        credit_loss = F.binary_cross_entropy(probs, labels)

        if not self.strategy.args.credit_nll_loss:
            nll_loss = 0
        else:
            # Sequence: input1 output1 input2 output2 input3 output3
            # We need to mask inputs of all samples and outputs of negative samples
            device = labels.device
            final_mask = torch.zeros_like(attention_mask, dtype=torch.bool)

            offset = 0
            for i, total_len in enumerate(packed_seq_lens):
                out_len = num_actions[i]
                in_len = total_len - out_len
                if labels[i] == 1:
                    start_output = offset + in_len
                    end_output = offset + total_len
                    final_mask[0, start_output:end_output] = True
                offset += total_len

            final_mask = final_mask & attention_mask.bool()

            lm_labels = torch.where(
                final_mask,
                sequences,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            ptx_log_probs = output["logits"]
            # loss function
            nll_loss = self.ptx_loss_fn(ptx_log_probs, lm_labels)

        loss = credit_loss + nll_loss * self.args.credit_nll_loss_coef
        self.strategy.backward(
            loss, self.credit_model, self.credit_optim)

        # grad_norm = torch.nn.utils.clip_grad_norm_(
        #     self.credit_model.model.parameters(), max_norm=self.strategy.args.credit_grad_clip)

        self.strategy.optimizer_step(
            self.credit_optim, self.credit_model, self.credit_scheduler)
        return {
            "prime/loss": credit_loss.item(),
            "prime/lr": self.credit_scheduler.get_last_lr()[0],
            # "prime/grad_norm": grad_norm.item(),
        }

    def compute_implicit_reward(self, log_prob, ref_log_prob, num_actions, num_agent_actions, is_avg_agents=False):
        """
        Average agents' scores for training, while keep agents' scores for inference
        """
        q = log_prob - ref_log_prob

        agent_level_scores = []

        # reward computation does not need gradient. only q needs
        idx = 0
        for agent_counts in num_agent_actions:
            agent_means = []
            for count in agent_counts:
                agent_slice = q[0, idx : idx + count]
                idx += count

                agent_means.append(agent_slice.mean())

            if is_avg_agents:
                sample_mean = torch.stack(agent_means).mean()
                agent_level_scores.append(sample_mean)
            else:
                agent_level_scores.append(torch.stack(agent_means)) 

        # From PRIME - this method will still consider the relative value of rewards.
        # The key is to control the absolute value of RETURN from being too high.
        # so the normalization is done by controlling the maximum of reverse cumulative sum
        # Only using barch norm for inference
        if getattr(self.strategy.args, "credit_batch_norm", False) and not is_avg_agents:

            for i in range(len(agent_level_scores)):
                normalized_agent_level_scores = agent_level_scores[i]

                reverse_cumsum = torch.cumsum(
                    normalized_agent_level_scores.flip(dims=[0]), dim=0).flip(dims=[0])
                normalized_agent_level_scores = normalized_agent_level_scores / \
                    (reverse_cumsum.abs().max() + 1e-6)

                agent_level_scores[i] = normalized_agent_level_scores

        return agent_level_scores, q


    def compute_ref_log_probs_for_samples(self, credit_samples: List[Samples]):
        """
        Use `self.initial_model` to perform a forward pass only once for each `PrimeSamples`,
        and store the resulting `base_action_log_probs` in `credit_samples[i].base_action_log_probs`.
        """
        credit_dataset = PrimeSamplesDataset(credit_samples)
        dataloader = self.strategy.setup_dataloader(
            credit_dataset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            collate_fn=lambda x: x[0],
        )

        for samples in dataloader:
            
            if hasattr(self.args, "num_agents"):
                assert len(samples.info) == self.args.num_agents, f"num_agents should be provided in args, but got {len(samples.info)} agents"
            
            sequences_cpu = samples.sequences.to("cpu")
            attention_mask_cpu = samples.attention_mask.to("cpu")
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu,
                samples.num_actions,
                attention_mask_cpu,
                packed_seq_lens=samples.packed_seq_lens
            )
            base_action_log_probs = ray.get([base_action_log_probs_ref])[0]
            if self.strategy.args.colocate_actor_ref:
                ray.get([self.initial_model.empty_cache.remote()])
            samples.base_action_log_probs = base_action_log_probs

        return credit_dataset

    def compute_final_rewards_for_samples(self, credit_samples: List[Samples]) -> List[Samples]:
        """
        After the credit_model training is completed, perform a forward pass on all credit_samples,
        calculate the reward (for example, a certain aggregation of q = log_prob - ref_log_prob), and store the results in samples.rewards.
        """
        device = torch.cuda.current_device()
        self.credit_model.eval()

        # Distributed Sampler for DeepSpeed
        credit_dataset = PrimeSamplesDataset(credit_samples)
        dataloader = self.strategy.setup_dataloader(
            credit_dataset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            collate_fn=lambda x: x[0],
        )

        return_samples_list = []
        with torch.no_grad():
            for samples in dataloader:
                samples.to_device(device)
                samples.base_action_log_probs = samples.base_action_log_probs.to(
                    device)

                if hasattr(self.args, "num_agents"):
                    assert len(samples.info) == self.args.num_agents, f"num_agents should be provided in args, but got {len(samples.info)} agents"
                
                policy_log_probs, _ = self.credit_model(
                    samples.sequences,
                    samples.num_actions,
                    attention_mask=samples.attention_mask,
                    return_output=True,
                    packed_seq_lens=samples.packed_seq_lens
                )

                agent_level_scores, q = self.compute_implicit_reward(
                    log_prob=policy_log_probs,
                    ref_log_prob=samples.base_action_log_probs,
                    num_actions=samples.num_actions,
                    num_agent_actions=samples.num_agent_actions,
                    is_avg_agents=False
                )

                # return q or token-level scores?
                samples.agent_level_scores = agent_level_scores

                return_samples_list.append(samples)
        return return_samples_list


@ray.remote(num_gpus=1)
class PrimeModelRayActor(BasePPORole):
    def init_model_from_pretrained(self,
                                   strategy: DeepspeedStrategy,
                                   pretrain,
                                   max_steps,
                                   rolename="credit"):
        self._setup_distributed(strategy)
        args = self.strategy.args
        self.args = args
        self.rolename = rolename

        credit_model = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
        )

        # configure optimizer
        credit_optim = strategy.create_optimizer(
            credit_model, lr=args.credit_learning_rate, betas=strategy.args.credit_adam_betas, weight_decay=args.credit_l2
        )

        credit_scheduler = get_scheduler(
            getattr(args, "credit_scheduler", "cosine_with_min_lr"),
            credit_optim,
            num_warmup_steps=math.ceil(max_steps * args.credit_lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={
                "min_lr": args.credit_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            credit_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.credit_model, self.credit_optim, self.credit_scheduler = strategy.prepare(
            (credit_model, credit_optim, credit_scheduler),
            is_rlhf=True,
        )

    def init_trainer(self, initial_model: ray.actor.ActorHandle):
        strategy = self.strategy
        args = strategy.args
        self.trainer = PrimePPOTrainer(
            strategy,
            actor=None,
            critic=None,
            reward_model=None,
            initial_model=initial_model,
            ema_model=None,
            actor_optim=None,
            critic_optim=None,
            actor_scheduler=None,
            critic_scheduler=None,
            credit_model=self.credit_model,
            credit_optim=self.credit_optim,
            credit_scheduler=self.credit_scheduler,
            credit_beta=args.credit_beta,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            credit_granularity=args.credit_granularity,
            rolename=self.rolename
        )

    def fit(self, steps, credit_samples):
        torch.cuda.empty_cache()
        self.credit_model.train()
        status = self.trainer.ppo_train(credit_samples)
        torch.cuda.empty_cache()
        return status

    def fit_and_reward(self, steps, credit_samples: List[Samples]):
        """
        1) First, use the initial_model to calculate ref_log_probs and store them in credit_samples.
        2) Use credit_samples to train the credit_model.
        3) After training is complete, calculate the final reward and write it back to credit_samples.
        4) Return credit_samples with the reward.
        """
        self.empty_cache()
        self.trainer.compute_ref_log_probs_for_samples(credit_samples)
        self.empty_cache()
        status = self.trainer.ppo_train(credit_samples)
        for sample in credit_samples:
            sample.to_device("cpu")
        status["is_rank_0"] = self.strategy.is_rank_0()
        self.empty_cache()
        credit_samples = self.trainer.compute_final_rewards_for_samples(
            credit_samples)
        for samples in credit_samples:
            samples.to_device("cpu")

        self.empty_cache()
        self.save_checkpoint(steps)
        return credit_samples, status

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.credit_model,
            None,
            args.save_path + "_" + self.rolename,
        )

    def save_checkpoint(self, global_step):
        args = self.strategy.args
        if global_step % args.save_steps != 0:
            return

        tag = f"global_step{global_step}"

        self.strategy.save_ckpt(
            self.credit_model.model,
            os.path.join(args.ckpt_path, "_" + self.rolename),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem
        )
