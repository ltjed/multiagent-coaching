import math
from itertools import groupby, accumulate

import os
import torch

from tqdm import tqdm
from typing import List
from torch.nn import functional as F

import ray
from transformers.trainer import get_scheduler

from marti.models.base_model import get_llm_for_sequence_regression
from marti.trainers.ppo.trainer import PPOTrainer
from marti.helpers.distributed.deepspeed import DeepspeedStrategy
from marti.models.ray_launcher import BasePPORole
from marti.models.model_utils import unpacking_samples
from marti.models.loss import GPTLMLoss, PolicyLoss, ValueLoss
from marti.worlds.base_world import Samples
from marti.trainers.ppo.prime import PrimeSamplesDataset

class SaliencyPPOTrainer(PPOTrainer):
    def __init__(self, *args,
                 credit_model,
                 credit_optim,
                 credit_scheduler,
                 credit_beta: float = 0.05,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.credit_model = credit_model
        self.credit_optim = credit_optim
        self.credit_scheduler = credit_scheduler
        self.credit_beta = credit_beta
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
                
                # self.strategy.print(samples)
                status = self.train_step_saliency(samples)

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

    def train_step_saliency(self, data: Samples):
        self.credit_model.train()
        if not self.packing_samples:
            sequences = torch.cat(data.sequences, dim=0).unsqueeze(0)
            attention_mask = torch.cat(data.attention_mask, dim=0).unsqueeze(0)
            action_mask = torch.cat(data.action_mask, dim=0).unsqueeze(0)
            num_actions = data.num_actions
            packed_seq_lens = data.packed_seq_lens
            # response_length = data.response_length
            num_agent_actions = data.num_agent_actions
            labels = data.labels
        else:
            sequences = data.sequences
            attention_mask = data.attention_mask
            action_mask = data.action_mask
            num_actions = data.num_actions
            packed_seq_lens = data.packed_seq_lens
            # response_length = data.response_length
            num_agent_actions = data.num_agent_actions
            labels = data.labels

        assert self.packing_samples is True, "packing_samples is only supported yet"


        values, output = self.credit_model(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens
        )

        agent_level_scores = self.compute_agent_rewards(
            values=values,
            num_actions=num_actions,
            num_agent_actions=num_agent_actions,
            is_training=True
        )

        # \sigma(\beta * sum)
        # probs = torch.stack([agent_score.sum() for agent_score in agent_level_scores]).sigmoid()  # [batch_size]
        # print("Probs", probs)
        # print("Labels", labels)
        # print()
        # credit_loss = F.binary_cross_entropy(probs, labels)
        probs = torch.stack([agent_score.sum() for agent_score in agent_level_scores]).sigmoid()
        labels = (labels == 1).to(probs.dtype)

        mse_loss = F.mse_loss(probs, labels)

        if getattr(self.args, "use_diversity_reg", False):
            agent_probs = [score.softmax(dim=0) for score in agent_level_scores]
            entropy = -sum([(prob * torch.log(prob + 1e-8)).sum() for prob in agent_probs])
            reg_coef = getattr(self.args, "diversity_coef", 0.01)
            credit_loss = mse_loss - reg_coef * entropy
        else:
            credit_loss = mse_loss

        # if not self.strategy.args.credit_nll_loss:
        #     nll_loss = 0
        # else:
        #     # Sequence: input1 output1 input2 output2 input3 output3
        #     # We need to mask inputs of all samples and outputs of negative samples
        #     device = labels.device
        #     final_mask = torch.zeros_like(attention_mask, dtype=torch.bool)

        #     offset = 0
        #     for i, total_len in enumerate(packed_seq_lens):
        #         out_len = num_actions[i]
        #         in_len = total_len - out_len
        #         if labels[i] == 1:
        #             start_output = offset + in_len
        #             end_output = offset + total_len
        #             final_mask[0, start_output:end_output] = True
        #         offset += total_len

        #     final_mask = final_mask & attention_mask.bool()

        #     lm_labels = torch.where(
        #         final_mask,
        #         sequences,
        #         self.ptx_loss_fn.IGNORE_INDEX,
        #     )

        #     ptx_log_probs = output["logits"]
        #     # loss function
        #     nll_loss = self.ptx_loss_fn(ptx_log_probs, lm_labels)
        nll_loss = 0

        loss = credit_loss + nll_loss * self.args.credit_nll_loss_coef
        self.strategy.backward(
            loss, self.credit_model, self.credit_optim)

        # grad_norm = torch.nn.utils.clip_grad_norm_(
        #     self.credit_model.model.parameters(), max_norm=self.strategy.args.credit_grad_clip)

        self.strategy.optimizer_step(
            self.credit_optim, self.credit_model, self.credit_scheduler)
        return {
            "saliency/loss": credit_loss.item(),
            "saliency/lr": self.credit_scheduler.get_last_lr()[0],
            # "prime/grad_norm": grad_norm.item(),
        }

    def compute_agent_rewards(self, values, num_actions, num_agent_actions, is_training=True):
        """
        Average agents' scores for training, while keep agents' scores for inference
        """
        agent_level_scores = []

        # reward computation does not need gradient. only q needs
        idx = 0
        for agent_counts in num_agent_actions:
            agent_means = []
            for count in agent_counts:
                agent_slice = values[0, idx : idx + count]
                idx += count

                agent_means.append(agent_slice.mean())

            # if is_avg_agents:
            # sample_mean = torch.stack(agent_means).mean()
            # agent_level_scores.append(sample_mean)
            # else:
            agent_level_scores.append(torch.stack(agent_means)) 

        # Only using barch norm for inference
        if getattr(self.strategy.args, "credit_batch_norm", False) and not is_training: #and not is_avg_agents:
            for i in range(len(agent_level_scores)):
                agent_level_scores[i] = agent_level_scores[i].sigmoid()

        return agent_level_scores

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
            for samples in tqdm(dataloader, desc="Compute Rewards", disable=not self.strategy.is_rank_0()):
                samples.to_device(device)

                if hasattr(self.args, "num_agents"):
                    assert len(samples.info) == self.args.num_agents, f"num_agents should be provided in args, but got {len(samples.info)} agents"
                
                values, outputs = self.credit_model(
                    samples.sequences,
                    samples.num_actions,
                    attention_mask=samples.attention_mask,
                    return_output=True,
                    packed_seq_lens=samples.packed_seq_lens
                )

                agent_level_scores = self.compute_agent_rewards(
                    values=values,
                    num_actions=samples.num_actions,
                    num_agent_actions=samples.num_agent_actions,
                    is_training=False
                )

                # return q or token-level scores?
                samples.agent_level_scores = agent_level_scores

                return_samples_list.append(samples)
        return return_samples_list


@ray.remote(num_gpus=1)
class SaliencyModelRayActor(BasePPORole):
    def init_model_from_pretrained(self,
                                   strategy: DeepspeedStrategy,
                                   pretrain,
                                   max_steps,
                                   rolename="credit"):
        self._setup_distributed(strategy)
        args = self.strategy.args
        self.args = args
        self.rolename = rolename

        credit_model = get_llm_for_sequence_regression(
            pretrain,
            "critic",
            normalize_reward=strategy.args.normalize_reward,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=strategy.args.value_head_prefix,
            init_value_head=strategy.args.pretrain == strategy.args.critic_pretrain,
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

        self.trainer = SaliencyPPOTrainer(
            strategy,
            actor=None,
            critic=None,
            reward_model=None,
            initial_model=None,
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
