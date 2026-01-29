# marti/marti/models/vllm/engine_async.py
import asyncio
import os
import time
import ray
import json

from marti.helpers.logging import init_logger
from marti.models.vllm.engine import BaseLLMRayActor
from marti.verifiers.auto_verify import auto_verify
from marti.worlds.steps.mcp_step import step_with_tools
logger = init_logger(__name__)

@ray.remote
class AgentInstance:
    def __init__(self, agent_func_path=None):
        
        if agent_func_path is None:
            self.agent_step = step_with_tools
        elif agent_func_path.endswith(".py"):
            import importlib.util

            spec = importlib.util.spec_from_file_location("step", agent_func_path)
            agent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_module)
            self.agent_step = agent_module.step
        else:
            raise ValueError("Agent path must be a Python file")

    async def step(self, observation, action, tool_manager,**kwargs):
        return await self.agent_step(observation, action, tool_manager, **kwargs)

@ray.remote
def get_tokenize_text_len(text, tokenizer):
    return len(tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0])


@ray.remote
class LLMRayActorAsync(BaseLLMRayActor):
    async def __init__(self, *args, bundle_indices: list = None,  **kwargs):
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)
        self.agent_func_path = kwargs.pop("agent_func_path")

        # # Initialize super class
        # super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        # Initialize result queue for streaming completed results
        self.result_queue = asyncio.Queue()

        os.environ["VLLM_USE_V1"] = "1"
        import vllm

        assert vllm.__version__ > "0.8.5", "Asyn VLLM version must be greater than 0.8.5"

        engine_args = vllm.AsyncEngineArgs(*args, **self.kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        await self.llm.is_sleeping()

        # Initialize concurrency logging
        self._concurrency_logging_enabled = False
        self._concurrency_logging_task = None

    async def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return await self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    async def update_weight(self, name, dtype, shape, empty_cache=False):
        return await self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    async def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return await self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    async def reset_prefix_cache(self):
        await self.llm.reset_prefix_cache()

    async def sleep(self, level=1):
        await self.llm.sleep(level=level)

    async def wake_up(self):
        await self.llm.wake_up()

    async def get_trajectory(self, tool_manager, hf_tokenizer, max_length, prompt, label, sampling_params, meta=None):
        # Create agent instance for this trajectory (back to original approach)
        agent_instance = AgentInstance.remote(self.agent_func_path)

        # Initialize observations and actions for the current prompt
        observation = [prompt]
        # action_ranges = []
        tools_used = {}
        trajectory_start = time.time()
        
        try:
            # Execute multiple steps of interaction
            for step_idx in range(tool_manager.get_max_turns()):
                # Next sampling budget
                observation_text = "".join(observation)
                observation_tokens_len = len(
                    hf_tokenizer(observation_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                )
                if observation_tokens_len >= max_length:
                    logger.warning(f"Trajectory exceeded max length at turn {step_idx}")
                    break
                
                # Generate response asynchronously
                request_output = await self.generate_async(observation_text, sampling_params)
                action = request_output.outputs[0].text

                # Call step function to get reward and next observation
                # Use asyncio.to_thread to make Ray remote call non-blocking
                # kwargs = {"sampling_params": sampling_params}
                result = await agent_instance.step.remote(observation, action, tool_manager, metadata=meta, label=label)

                # Collect tool usage
                step_tools = result.get("extra_logs", {}).get("tools_used", {})
                for tool, count in step_tools.items():
                    tools_used[tool] = tools_used.get(tool, 0) + count

                observation = result["next_observation"]
                done = result["done"]

                if done:
                    break
        finally:
            # Kill agent instance when done
            ray.kill(agent_instance)

        trajectory_time = time.time() - trajectory_start
        return {"result": result, "observation": observation, "trajectory_time": trajectory_time, "tools_used": tools_used}


    async def add_requests(self, tool_manager, sampling_params, prompts, labels, max_length, task, hf_tokenizer=None, metadata=None):
        """
        Process requests from rank0 and generate responses with multiple agent interactions.
        Each prompt will go through multiple steps of interaction using the step function.
        Results are streamed back as each agent completes its execution.

        Args:
            sampling_params: Parameters for sampling
            prompts: List of prompts to process
            labels: List of labels corresponding to prompts
            max_steps: Maximum number of interaction steps
        """

        # Create semaphore to control concurrent task execution
        semaphore = asyncio.Semaphore(tool_manager.get_num_workers())

        async def execute_agent(prompt, label, sampling_params, meta=None):
            async with semaphore:
                traj_dict = await self.get_trajectory(tool_manager, hf_tokenizer, max_length, prompt, label, sampling_params, meta=meta)
                result, observation, trajectory_time, tools_used = traj_dict["result"], traj_dict["observation"], traj_dict["trajectory_time"], traj_dict["tools_used"]

                if "final_reward" in result:
                    final_reward = result["final_reward"]
                else:
                    final_reward = auto_verify(meta.get("source", task), 1, ["".join(observation[1:])], [label])[0]

                # Store the final response when agent execution is complete
                final_response = {
                    "prompt": prompt,
                    "label": label,
                    "observation": observation[1:], # remove the prompt
                    "reward": final_reward,
                    "trajectory_time": trajectory_time,
                    "tools_used": tools_used,
                    "turns": len(observation) - 1
                }
                await self.result_queue.put(final_response)

        # Create and start tasks for all agent executions with controlled concurrency
        import copy

        if metadata is None:
            metadata = [{} for _ in range(len(prompts))]
        else:
            metadata = [json.loads(meta) for meta in metadata]

        tasks = []
        for prompt, label, meta in zip(prompts, labels, metadata):
            tasks.append(execute_agent(prompt, label, copy.deepcopy(sampling_params), meta))

        # Run the async code using the class's event loop
        await asyncio.gather(*tasks)

    async def generate_async(self, prompts, sampling_params):
        from vllm.utils import random_uuid

        request_id = random_uuid()
        results_generator = self.llm.generate(prompts, sampling_params, request_id)
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        return final_output

    async def get_responses(self):
        """
        Synchronously get all completed agent results from the queue.
        Waits for all tasks to complete before returning results.
        Returns: List of all completed agent results.
        """
        # Get all results from the queue
        results = []
        while not self.result_queue.empty():
            try:
                results.append(await self.result_queue.get())
            except asyncio.QueueEmpty:
                break
        return results

    async def start_concurrency_logging(self, interval_seconds=60):
        """Start periodic logging of concurrent requests being served."""
        import asyncio
        import torch

        if self._concurrency_logging_enabled:
            return  # Already running

        self._concurrency_logging_enabled = True

        async def log_concurrency():
            while self._concurrency_logging_enabled:
                try:
                    # For async engine, try different API paths
                    # vLLM 0.8.5+ might have different internal structure
                    num_running = None
                    if hasattr(self.llm, 'get_num_unfinished_requests'):
                        # Direct async method
                        num_running = await self.llm.get_num_unfinished_requests()
                    elif hasattr(self.llm, 'engine') and hasattr(self.llm.engine, 'get_num_unfinished_requests'):
                        # Via engine attribute (no await - sync method)
                        num_running = self.llm.engine.get_num_unfinished_requests()
                    else:
                        logger.warning("[vLLM Concurrency Async] API method not found, disabling logging")
                        self._concurrency_logging_enabled = False
                        break

                    gpu_id = torch.cuda.current_device() if torch.cuda.is_available() else "N/A"
                    logger.info(
                        f"[vLLM Concurrency Async] GPU {gpu_id}: {num_running} concurrent requests running"
                    )
                except Exception as e:
                    logger.warning(f"[vLLM Concurrency Async] Failed to get stats: {e}")

                await asyncio.sleep(interval_seconds)

        self._concurrency_logging_task = asyncio.create_task(log_concurrency())
        logger.info(f"[vLLM Concurrency Async] Started logging every {interval_seconds}s")

    async def stop_concurrency_logging(self):
        """Stop periodic concurrency logging."""
        if self._concurrency_logging_enabled:
            self._concurrency_logging_enabled = False
            if self._concurrency_logging_task:
                self._concurrency_logging_task.cancel()
                try:
                    await self._concurrency_logging_task
                except asyncio.CancelledError:
                    pass
            logger.info("[vLLM Concurrency Async] Stopped logging")