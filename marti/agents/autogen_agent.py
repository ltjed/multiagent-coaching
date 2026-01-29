import asyncio
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import EVENT_LOGGER_NAME
from .base_agent import Agent, AgentRegistry
# from autogen_agentchat.conditions import AnyOfTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
# -------------------------------------------------------------------- #
#             DialogueProcessor / parse_logs
# -------------------------------------------------------------------- #
class DialogueProcessor:
    def __init__(self, current_task: str, rounds: int):
        self.current_task = current_task
        self.rounds = rounds
        self.history: List[Dict[str, Any]] = []
        self.primary_outputs: List[str] = []

    def _extract_agent_name(self, agent_id: str) -> str:
        m = re.match(r"([a-zA-Z]+)_", agent_id)
        return m.group(1).capitalize() if m else "Unknown"

    def _extract_output(self, log_data: Dict) -> str:
        output = (
            log_data.get("response", {})
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return output.split("APPROVE")[0].strip()

    def process_line(self, line: str):
        if '"type": "LLMCall"' not in line:
            return
        # try:
        print(line.split(" - ", 2)[-1])
        log_data = json.loads(line.split(" - ", 2)[-1])
        agent_id = log_data.get("agent_id", "")
        agent_name = self._extract_agent_name(agent_id)
        node = {
            "node_idx": agent_name,
            "inputs": log_data.get("messages", []),
            "outputs": self._extract_output(log_data),
        }
        self.history.append(node)
        if agent_name == "Primary":
            self.primary_outputs.append(node["outputs"])
        # except Exception as e:
        #     logging.error(f"Failed to parse log: {e}")

    def finalize(self):
        if self.history and self.history[-1]["node_idx"] == "Critic":
            self.history.pop()

    def to_struct(self) -> Dict[str, Any]:
        self.finalize()
        return {
            "accuracy": 1,
            "metadata": [
                {
                    "indice": "0",
                    "prompt": self.current_task,
                    "label": self.primary_outputs[-1] if self.primary_outputs else "",
                    "history": self.history,
                }
            ],
        }


def parse_logs(log_path: str, task: str, rounds: int) -> Dict[str, Any]:
    dp = DialogueProcessor(task, rounds)
    with open(log_path, "r", encoding="utf-8") as f:
        for ln in f:
            dp.process_line(ln)
    return dp.to_struct()


# -------------------------------------------------------------------- #
#                            AutogenAgent
# -------------------------------------------------------------------- #
class LimitedRoundGroupChat(RoundRobinGroupChat):
    """Primary → Critic counts as one round; stops when limit reached or APPROVE received"""
    def __init__(self, agents, max_rounds: int):
        super().__init__(
            agents,
            termination_condition=TextMentionTermination("APPROVE"),
        )
        self.max_rounds = max_rounds
        self.primary_turns = 0

    # Increment count each time Primary speaks
    async def _post_agent_reply(self, msg):
        if msg.get("agent_id", "").startswith("primary"):
            self.primary_turns += 1
            if self.primary_turns >= self.max_rounds:
                # Mark as terminated: select_speaker will receive None
                self._terminated = True
        await super()._post_agent_reply(msg)
    
@AgentRegistry.register("autogen_agent")
class AutogenAgent(Agent):
    """
    Execution flow:
      1. Dynamically create primary_x / critic_x AssistantAgents for current prompt
      2. RoundRobinGroupChat + TextMentionTermination("APPROVE") controls dialogue
      3. Write autogen_core EVENT_LOGGER to separate file
      4. parse_logs → history / label / approved
    """

    def __init__(self, config: Dict, **kwargs):
        super().__init__(**kwargs)
        self.cfg = config
        os.makedirs(self.cfg.get("log_dir", "autogen_logs"), exist_ok=True)

    # ------------------ Single prompt execution + log writing ------------------ #
    async def _run_one(self, prompt: str, idx: int, log_path: str):
        # ========== Configure event logging ==========
        logger = logging.getLogger(EVENT_LOGGER_NAME)
        logger.handlers = []  # Clear previous handlers to avoid mixing logs from multiple prompts
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

        # ========== Create two agent roles ==========
        model_client = OpenAIChatCompletionClient(
            model=self.cfg["model"],
            api_key=self.cfg["api_key"],
            base_url=self.cfg["base_url"],
        )

        primary = AssistantAgent(
            f"primary_{idx}",
            model_client=model_client,
            system_message=self.cfg["generator_prompt"],
        )
        critic = AssistantAgent(
            f"critic_{idx}",
            model_client=model_client,
            system_message=self.cfg["critic_prompt"],
        )
        team = LimitedRoundGroupChat(
            [primary, critic],
            max_rounds=self.cfg.get("rounds", 3),       # reflection rounds
        )
        # ========== Run ==========
        await team.run(task=prompt)

        # Close handler to ensure file is fully written
        fh.close()
        logger.removeHandler(fh)

    # ------------------ execute interface ------------------ #
    def execute(self, inputs: List[Any]):
        return self._execute(inputs, {}, {}, 0)

    def _execute(
        self,
        inputs: List[Any],
        spatial_info: Dict = None,
        temporal_info: Dict = None,
        turn_id: int = 0,
    ):
        res = []
        for i, prompt in enumerate(inputs):
            log_path = os.path.join(
                self.cfg.get("log_dir", "autogen_logs"),
                f"log_{uuid.uuid4().hex}.log",
            )
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._run_one(str(prompt), i, log_path))
            finally:
                loop.close()

            parsed = parse_logs(
                log_path, task=str(prompt), rounds=self.cfg.get("rounds", 3)
            )
            meta = parsed["metadata"][0]
            res.append(
                {
                    "prompt": meta["prompt"],
                    "label": meta["label"],
                    "history": meta["history"],
                    "approved": any(
                        "APPROVE" in h["outputs"].upper()
                        for h in meta["history"]
                        if h["node_idx"] == "Critic"
                    ),
                    "turn_id": turn_id,
                }
            )
        return res

    # ---------- Reserved MARTI Hook ----------
    def _process_inputs(
        self,
        raw_inputs: List[Any],
        spatial_info: Dict = None,
        temporal_info: Dict = None,
        **kwargs,
    ) -> List[Any]:
        return raw_inputs