import hydra
from omegaconf import DictConfig, OmegaConf

from marti.helpers.common import get_strategy
from marti.controllers.base_controller import BaseController
from marti.controllers.multi_agent_controller import MultiAgentController

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Weave import (optional dependency)
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False

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

    if cfg.vllm_enable_sleep and not cfg.colocate_all_models:
        print("Set cfg.vllm_enable_sleep to False when cfg.colocate_all_models is disabled.")
        cfg.vllm_enable_sleep = False


    if cfg.packing_samples:
        if not cfg.flash_attn:
            print(
                "[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
            cfg.flash_attn = True
        assert cfg.vllm_num_engines > 0, "Only support `--packing_samples` with vLLM."
        assert not cfg.pretrain_data, "`--pretrain_data` is not supported with `--packing_samples` yet."

    return cfg

@hydra.main(config_path="../configs", config_name="default.yaml", version_base=None)
def train(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    for key, value in cfg.default_agent.items():
        cfg[key] = value

    _rationalize_config(cfg)
    _validate_config(cfg)
    print(OmegaConf.to_yaml(cfg))

    # Initialize Weave tracking if enabled
    if WEAVE_AVAILABLE and cfg.get("use_weave", False):
        weave_project = cfg.get("weave_project", "marti-training")
        print(f"[Weave] Initializing Weave tracking: {weave_project}")
        weave.init(project_name=weave_project)
        print(f"[Weave] ✓ Weave tracking enabled. View traces at: https://wandb.ai/{weave_project}")
    elif cfg.get("use_weave", False) and not WEAVE_AVAILABLE:
        print("[Weave] Warning: use_weave=True but weave package not installed. Install with: pip install weave>=0.51.0")

    # configure strategy
    strategy = get_strategy(cfg)

    agent_workflow = cfg.get("agent_workflow", "base")
    if agent_workflow in ["base", "tool"]:
        controller_class = BaseController
    # elif agent_workflow in ["multi-agents-debate", "chain-of-agents", "mixture-of-agents", "judge_workflow"]:
    else:
        controller_class = MultiAgentController
    # else:
    #     raise NotImplementedError

    # print(OmegaConf.to_yaml(cfg))
    controller = controller_class(strategy=strategy)
    controller.build()
    controller.run()

if __name__ == "__main__":
    train()
