import time
import ray
import asyncio
import copy
from typing import Dict, List, Any, Optional
from marti.helpers.logging import init_logger
from marti.verifiers.auto_verify import auto_verify

logger = init_logger(__name__)


@ray.remote
class MultiAgentWrapper:
    """Wrapper for managing multiple agents and their interactions."""
    
    def __init__(self, agents: List[Dict[str, Any]], workflow_args, *args, **kwargs):
        """
        Initialize multi-agent wrapper.
        
        Args:
            agents: List of agent configurations, each containing:
                - name: Agent identifier
                - llm: LLMRayActorAsync instance
                - tokenizer: Tokenizer for the agent
                - sampling_params: Default sampling parameters
                - workflow_func_path: Path to agent step function (optional)
        """
        self.agents = agents
        self.workflow_args = workflow_args
        self.workflow_func_path = kwargs.pop("workflow_func_path")
        self.result_queue = asyncio.Queue()
        
        # Load workflow function once during initialization
        if self.workflow_func_path.endswith(".py"):
            import importlib.util
            spec = importlib.util.spec_from_file_location("workflow", self.workflow_func_path)
            workflow_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(workflow_module)
            self.workflow_func = workflow_module.workflow
        else:
            raise ValueError("Workflow path must be a Python file")

    async def add_requests(
        self,
        tool_manager,
        prompts: List[str],
        labels: List[str],
        task: str,
        metadata: Optional[List[Dict]] = None,
        max_length: int = None,
        is_eval: bool = False
    ) -> List[Dict[str, Any]]:
        """Process requests using multi-agent workflow."""
        
        # Create semaphore to control concurrent workflow execution
        semaphore = asyncio.Semaphore(tool_manager.get_num_workers())
        
        async def execute_workflow(prompt: str, label: str, meta: Optional[Dict] = None, prompt_id: int = 0):
            """Execute a single workflow instance."""
            async with semaphore:
                workflow_start = time.time()
                
                try:
                    # Execute workflow directly without creating remote instance
                    logger.info(f"[execute_workflow] Starting workflow for prompt_id={prompt_id}")
                    result = await self.workflow_func(
                        prompt=prompt,
                        label=label,
                        agents=self.agents,
                        tool_manager=tool_manager,
                        task=task,
                        metadata=meta,
                        workflow_args=self.workflow_args,
                        max_length=max_length,
                        prompt_id=prompt_id,
                        is_eval=is_eval,
                    )

                    # Add timing information
                    result["workflow_time"] = time.time() - workflow_start

                    # Validate result structure
                    if not isinstance(result, dict):
                        logger.error(f"[execute_workflow] Workflow returned non-dict! type={type(result)}, prompt_id={prompt_id}")
                        result = {
                            "prompt": prompt,
                            "label": label,
                            "trajectory": [],
                            "final_reward": 0,
                            "error": f"Workflow returned {type(result)} instead of dict",
                            "workflow_time": time.time() - workflow_start
                        }
                    else:
                        logger.info(f"[execute_workflow] Workflow succeeded for prompt_id={prompt_id}, final_reward={result.get('final_reward')}, outcome_score={result.get('outcome_score')}")

                    # Store result
                    await self.result_queue.put(result)

                except Exception as e:
                    logger.error(f"[execute_workflow] Workflow execution error for prompt_id={prompt_id}: {e}")
                    import traceback
                    logger.error(f"[execute_workflow] Traceback:\n{traceback.format_exc()}")
                    error_result = {
                        "prompt": prompt,
                        "label": label,
                        "trajectory": [],
                        "final_reward": 0,
                        "error": str(e),
                        "workflow_time": time.time() - workflow_start
                    }
                    await self.result_queue.put(error_result)

        # Create tasks for all workflows
        if metadata is None:
            metadata = [{} for _ in range(len(prompts))]

        tasks = []
        for idx, (prompt, label, meta) in enumerate(zip(prompts, labels, metadata)):
            tasks.append(execute_workflow(prompt, label, copy.deepcopy(meta), prompt_id=idx))

        # Execute all workflows concurrently
        await asyncio.gather(*tasks)

    async def get_responses(self, expected_len: int) -> List[Dict[str, Any]]:
        logger.info(f"[get_responses] Waiting for {expected_len} results from queue...")
        results = []
        for i in range(expected_len):
            result = await self.result_queue.get()
            logger.info(f"[get_responses] Got result {i+1}/{expected_len}, type={type(result)}, keys={result.keys() if isinstance(result, dict) else 'NOT_DICT'}")
            if isinstance(result, dict):
                has_reward = "final_reward" in result
                has_answer = "outcome_score" in result
                has_error = "error" in result
                logger.info(f"[get_responses] Result {i+1} structure: has_reward={has_reward}, has_answer={has_answer}, has_error={has_error}")
                if has_reward:
                    logger.info(f"[get_responses] Result {i+1} final_reward={result.get('final_reward')}")
                if has_error:
                    logger.warning(f"[get_responses] Result {i+1} has error: {result.get('error')}")
            else:
                logger.error(f"[get_responses] Result {i+1} is NOT a dict! type={type(result)}, value={result}")
            results.append(result)
        logger.info(f"[get_responses] Collected all {expected_len} results")
        return results