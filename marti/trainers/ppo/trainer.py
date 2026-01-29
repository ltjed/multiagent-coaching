import os
import os.path
from abc import ABC
from typing import Dict, List, Sequence, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from marti.models.actor import Actor
from marti.models.loss import GPTLMLoss, PolicyLoss, ValueLoss

from marti.models.model_utils import masked_mean
from marti.helpers.distributed.distributed_sampler import DistributedSampler
from marti.helpers.distributed.distributed_utils import init_process_group

from marti.trainers.experience_maker import Experience, ExperienceMaker
from marti.trainers.kl_controller import AdaptiveKLController, FixedKLController
from marti.trainers.replay_buffer import NaiveReplayBuffer


class PPOTrainer(ABC):
    """
    Trainer for Proximal Policy Optimization (PPO) algorithm.

    Args:
        strategy (Strategy): The training strategy to use.
        actor (Actor): The actor model in the PPO algorithm.
        critic (nn.Module): The critic model in the PPO algorithm.
        reward_model (nn.Module): The reward model for calculating rewards in the RLHF setup.
        initial_model (Actor): The initial model for reference logits to limit actor updates in RLHF.
        ema_model (Actor): The exponential moving average model for stable training.
        actor_optim (Optimizer): The optimizer for the actor model.
        critic_optim (Optimizer): The optimizer for the critic model.
        actor_scheduler (Scheduler): The learning rate scheduler for the actor.
        critic_scheduler (Scheduler): The learning rate scheduler for the critic.
        ema_beta (float, defaults to 0.992): EMA decay rate for model stability.
        init_kl_coef (float, defaults to 0.001): Initial coefficient for KL divergence.
        kl_target (float, optional): Target value for KL divergence.
        kl_horizon (int, defaults to 10000): Horizon for KL annealing.
        ptx_coef (float, defaults to 0): Coefficient for supervised loss from pre-trained data.
        micro_train_batch_size (int, defaults to 8): Micro-batch size for actor training.
        buffer_limit (int, defaults to 0): Maximum size of the replay buffer.
        buffer_cpu_offload (bool, defaults to True): If True, offloads replay buffer to CPU.
        eps_clip (float, defaults to 0.2): Clipping coefficient for policy loss.
        value_clip (float, defaults to 0.2): Clipping coefficient for value function loss.
        micro_rollout_batch_size (int, defaults to 8): Micro-batch size for generating rollouts.
        gradient_checkpointing (bool, defaults to False): If True, enables gradient checkpointing.
        max_epochs (int, defaults to 1): Number of epochs to train.
        max_norm (float, defaults to 1.0): Maximum gradient norm for gradient clipping.
        prompt_max_len (int, defaults to 128): Maximum length for prompts.
        dataloader_pin_memory (bool, defaults to True): If True, pins memory in the data loader.
        remote_rm_url (str, optional): URL for remote reward model API.
        **generate_kwargs: Additional arguments for model generation.
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        clip_eps_high: float = 0.2,
        policy_loss_type="ppo", 
        token_level_loss=False,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        pretrain_dataloader=None,
        rolename=None,
        eos_token_id=-1,
        **generate_kwargs,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler
        
        assert rolename is not None, "rolename should not be None"
        self.rolename = rolename
        self.eos_token_id = eos_token_id
        
        self.actor_loss_fn = PolicyLoss(eps_clip, clip_eps_high, policy_loss_type, token_level_loss)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        self.pretrain_dataloader = pretrain_dataloader

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.experience_maker = ExperienceMaker(
            actor,
            critic,
            reward_model,
            initial_model,
            prompt_max_len,
            self.kl_ctl,
            strategy,
            remote_rm_url,
        )

        packing_samples = getattr(self.args, "packing_samples", False)
        max_packed_length = getattr(self.args, "max_packed_length", None)
        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size, buffer_limit, buffer_cpu_offload, packing_samples, max_packed_length
        )

    def ppo_train(self, global_steps=0):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        num_gradient_updates = len(dataloader)
        print(f"  └─ [{self.rolename}] Training: {num_gradient_updates} gradient updates (micro_batch_size={self.replay_buffer.sample_batch_size}, buffer_size={len(self.replay_buffer)})")
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"  └─ Gradient updates [{self.rolename}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience, global_steps)

                # for DP
                # weighted mean for kl
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "ret": status["return"],
                        "glen": status["response_length"],
                        "tlen": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                    }

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)

                pbar.set_postfix(short_status)

                # Clear cache after each training step to prevent memory accumulation
                torch.cuda.empty_cache()

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        pass
    
    def fit(self, steps, samples_ref):
        all_samples = samples_ref
        args = self.args
        
        if args.training_mode != "sft":
            for i, experience in enumerate(
                self.experience_maker.make_experience_list(all_samples, **self.generate_kwargs)
            ):
                self.replay_buffer.append(experience)
        
        torch.cuda.empty_cache()
        if args.training_mode != "sft":
            self.replay_buffer.normalize("advantages", self.strategy)
        status = self.ppo_train(steps)
        if args.training_mode != "sft":
            self.replay_buffer.clear()
        torch.cuda.empty_cache()

        if "kl" in status:
            self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)

        # logs/checkpoints
        client_states = {"consumed_samples": steps * args.rollout_batch_size}
        self.save_checkpoints(args, steps, client_states)

        return {
            "status": status,
            "is_rank_0": self.strategy.is_rank_0(),
            "perf_stats": self.experience_maker.perf_stats
        }

    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = torch.cat(experience.base_action_log_probs, dim=0).unsqueeze(0)

            if experience.action_mask is not None:
                experience_mask = torch.cat(experience.action_mask, dim=0).unsqueeze(0)
            else:
                experience_mask = None
        else:
            sequences = experience.sequences
            old_action_log_probs = experience.action_log_probs
            advantages = experience.advantages
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = experience.base_action_log_probs

        # actor loss
        action_log_probs, output = self.actor(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )

        if self.args.mask_truncated_completions:
            if self.eos_token_id is None:
                raise ValueError("eos_token_id must be set when mask_truncated_completions=True")

            truncation_mask = _build_truncation_mask(
                sequences,
                packed_seq_lens,
                num_actions,
                self.eos_token_id,
                dtype=torch.bool,
            )
        else:
            truncation_mask = None

        # assert action_mask.shape == action_log_probs.shape, \
        #     "action_mask shape mismatch"
        if truncation_mask is not None and experience_mask is not None:
            action_mask = truncation_mask & experience_mask
        elif truncation_mask is not None:
            action_mask = truncation_mask
        else:
            action_mask = experience_mask

        # loss function
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=action_mask,
        )
        # DEBUG: Memory before backward
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[DEBUG before backward] allocated={mem_alloc:.2f}GB, reserved={mem_reserved:.2f}GB, loss={actor_loss.item():.4f}")

        if self.args.use_kl_loss:
            kl_loss = action_log_probs - base_action_log_probs
            if self.args.use_kl_estimator_k3:
                kl_loss = -kl_loss
                r = kl_loss.exp()
                kl_loss = r - 1.0 - kl_loss
            kl_loss = masked_mean(kl_loss, action_mask, dim=-1).mean()
        else:
            kl_loss = 0

        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = actor_loss + aux_loss * self.args.aux_loss_coef + kl_loss * self.kl_ctl.value
        self.strategy.backward(loss, self.actor, self.actor_optim)

        # ptx loss
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
            ptx_log_probs = output["logits"]

            # loss function
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # mixtral
            if self.aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # status
        status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0]}
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()

        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        return status

    def training_step_critic(self, experience: Experience) -> Dict[str, float]:
        self.critic.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_values = torch.cat(experience.values, dim=0).unsqueeze(0)
            returns = torch.cat(experience.returns, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_values = experience.values
            returns = experience.returns
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask

        # critic loss
        values, output = self.critic(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )
        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        return status

    def save_checkpoints(self, args, global_step, client_states={}):
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def _save_checkpoint(self, args, tag, client_states):
        self.strategy.save_ckpt(
            self.actor.model,
            os.path.join(args.ckpt_path, "_" + self.rolename),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )
        if self.critic is not None:
            self.strategy.save_ckpt(
                self.critic, os.path.join(args.ckpt_path, "_" + self.critic.rolename), tag, args.max_ckpt_num, args.max_ckpt_mem
            )

# def _build_truncation_mask(
#     sequences: torch.Tensor,                       # (1, Σ packed_seq_lens)
#     packed_seq_lens: Union[List[int], torch.Tensor],
#     num_actions: Union[List[int], torch.Tensor],
#     eos_token_id: int,
#     dtype: torch.dtype = torch.float32,
# ) -> torch.Tensor:
#     """
#     According to whether the last token of each sample is EOS, generate an action-level mask for the entire output.

#     Returns
#     -------
#     action_mask : torch.Tensor
#         The shape is (1, Σ num_actions) and aligns with action_log_probs / advantages. The range is {0, 1}, with data type `dtype`.
#     """
#     if eos_token_id is None:
#         raise ValueError("eos_token_id must be set when building truncation mask")

#     device = sequences.device
#     # ---------- Make sure packed_seq_lens / num_actions are LongTensor ----------
#     if not isinstance(packed_seq_lens, torch.Tensor):
#         packed_seq_lens = torch.as_tensor(packed_seq_lens, dtype=torch.long, device=device)
#     else:
#         packed_seq_lens = packed_seq_lens.to(device=device, dtype=torch.long)

#     if not isinstance(num_actions, torch.Tensor):
#         num_actions = torch.as_tensor(num_actions, dtype=torch.long, device=device)
#     else:
#         num_actions = num_actions.to(device=device, dtype=torch.long)

#     # ---------- check ----------
#     assert packed_seq_lens.shape == num_actions.shape, \
#         "`packed_seq_lens` and `num_actions` must have the same length (batch size)"
#     assert sequences.numel() == packed_seq_lens.sum().item(), \
#         "Sum(packed_seq_lens) must equal len(flattened sequences)"

#     # ---------- Calculate whether the last token of each sample is EOS ----------
#     seq_flat = sequences.view(-1)                           # (Σ packed_seq_lens,)
#     last_token_indices = packed_seq_lens.cumsum(0) - 1      # (B,)
#     ends_with_eos = seq_flat[last_token_indices] == eos_token_id  # bool (B,)

#     # ---------- Expand sample-level tags to action level ----------
#     sample_mask = ends_with_eos.to(dtype=dtype)             # (B,)
#     action_mask = sample_mask.repeat_interleave(num_actions)  # (Σ num_actions,)

#     return action_mask.unsqueeze(0)                         # (1, Σ num_actions)

def _build_truncation_mask(
    sequences: torch.Tensor,                       # shape: (1, Σ packed_seq_lens)
    packed_seq_lens: Union[List[int], torch.Tensor],
    num_actions: Union[List[int], torch.Tensor],
    eos_token_ids: Union[int, Sequence[int]],      # int or [1,2,3]
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Action-level truncation mask:

    If the last m tokens of the sample's (input+output) are completely identical to eos_token_ids → 1

    Otherwise → 0

    Return value shape = (1, Σ num_actions), aligned with action_log_probs.
    """
    # ---------- Parameter normalization ----------
    device = sequences.device
    seq_flat = sequences.view(-1)                               # (Σ packed_seq_lens,)

    if not isinstance(packed_seq_lens, torch.Tensor):
        packed_seq_lens = torch.as_tensor(packed_seq_lens, dtype=torch.long, device=device)
    else:
        packed_seq_lens = packed_seq_lens.to(device=device, dtype=torch.long)

    if not isinstance(num_actions, torch.Tensor):
        num_actions = torch.as_tensor(num_actions, dtype=torch.long, device=device)
    else:
        num_actions = num_actions.to(device=device, dtype=torch.long)

    eos_tokens = torch.as_tensor(
        eos_token_ids if isinstance(eos_token_ids, Sequence) else [int(eos_token_ids)],
        dtype=seq_flat.dtype,
        device=device,
    )
    m = eos_tokens.numel()                                      # End sequence length

    # ---------- Safety inspection ----------
    assert packed_seq_lens.shape == num_actions.shape, \
        "`packed_seq_lens` and `num_actions` must have same length (batch size)"
    assert seq_flat.numel() == packed_seq_lens.sum().item(), \
        "Sum(packed_seq_lens) must equal len(flattened sequences)"

    # ---------- Extract the last m tokens from each sample ----------
    ends = packed_seq_lens.cumsum(0)                            # (B,) End position(1-based)
    starts = ends - m                                           # (B,)
    valid_len = packed_seq_lens >= m                            # (B,) Is it long enough?

    # In order to vectorize, pull the end segments of all samples to (B, m)
    # 1) Calculate [0,1,2,...,m-1], broadcast to (B, m)
    offset = torch.arange(m, device=device).expand(len(starts), m)
    # 2) Index = starts.unsqueeze(1) + offset (may be negative; mask later)
    gather_idx = starts.unsqueeze(1) + offset                   # (B, m)
    # 3) All insufficiently long samples are set to -1, and immediate failure occurs during subsequent comparisons.
    gather_idx[~valid_len, :] = -1
    last_tokens = seq_flat[gather_idx.clamp(min=0)]             # (B, m)
    # For invalid samples (gather_idx=-1), last_tokens = seq_flat[0], but valid_len has been identified

    # ---------- Determine whether it is a complete match ----------
    match = (last_tokens == eos_tokens)                         # (B, m) Bitwise equality
    ends_with_eos = valid_len & match.all(dim=1)                # (B,) True/False

    # ---------- Expand to action level ----------
    sample_mask = ends_with_eos.to(dtype=dtype)                 # (B,)
    action_mask = sample_mask.repeat_interleave(num_actions)    # (Σ num_actions,)

    return action_mask.unsqueeze(0)                             # (1, Σ num_actions)