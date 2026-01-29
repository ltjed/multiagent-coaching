import json
from typing import Dict, Any, List, Optional

from marti.worlds.tools.manager import ToolManager
from marti.helpers.logging import init_logger

import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))


async def step(
    observation: List[str],
    action: str,
    tool_manager: Optional[ToolManager] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generic step function that works with any tools.
    Tool parser should return: [{"name": "tool_name", "args": "{...}"}, ...]
    """
    metadata = json.loads(kwargs["metadata"])
    # TODO: call openai api
    final_reward = None
    
    return {
        "next_observation": observation + [action],
        "done": True,
        "extra_logs": {"tools_used": {}},
        "final_reward": final_reward
    }
