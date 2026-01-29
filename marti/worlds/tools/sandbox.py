# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from: https://github.com/volcengine/verl/blob/main/verl/tools/sandbox_fusion_tools.py

import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, Tuple, TypeVar
from uuid import uuid4

import ray
import ray.actor
import ray.util.multiprocessing

from marti.worlds.tools.base import BaseToolExecutor
from marti.worlds.third_party.sandbox_fusion import _process_single_case, call_sandbox_api
from marti.helpers.logging import init_logger

logger = init_logger(__name__)


class SandboxFusionExecutor(BaseToolExecutor):
    """A tool for executing the code using sanbox fusion image.

    - `execute`: execute the tool.
    """

    def __init__(self, base_url, timeout=30, language="python", **kwargs):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "A tool for execute code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "code needs to be execute and grad",
                        },
                    },
                    "required": ["code"],
                },
            }
        })
        """
        # TODO: better documentation for the config
        self.default_timeout = timeout
        self.default_language = language
        self.sandbox_fusion_url = base_url
        self.config = kwargs
        if self.sandbox_fusion_url == "":
            raise ValueError("sandbox_fusion_url is not set")

    def get_name(self):
        return "code_interpreter"

    async def execute(self, parameters: dict[str, Any], **kwargs) -> Tuple[str, dict]:
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)

        # Extract files from kwargs (backward compatible - defaults to None)
        # Support both "files" and "task_files" parameter names for compatibility
        files = kwargs.get("task_files", kwargs.get("files", None))

        # Extract fetch_files from kwargs - list of filenames to retrieve after execution
        # This enables file persistence between agent executions
        fetch_files = kwargs.get("fetch_files", None)

        if not isinstance(code, str):
            code = str(code)

        import asyncio

        try:
            # Pass files and fetch_files to sandbox API
            response, last_error = call_sandbox_api(
                self.sandbox_fusion_url, code, None, timeout, timeout, language,
                files=files, fetch_files=fetch_files
            )

            if last_error is not None:
                return last_error, {"run_status": "Error", "exception": last_error}

            metadata = {
                "run_status": response["run_result"]["status"],
                "stdout": response["run_result"]["stdout"].strip() if response["run_result"]["stdout"] else "",
                "stderr": response["run_result"]["stderr"].strip() if response["run_result"].get("stderr") else "",
                "return_code": response["run_result"].get("return_code", None)
            }

            # Extract fetched files from response (if any)
            # These are base64-encoded and can be passed to the next agent
            fetched_files = response.get("files", {})
            if fetched_files:
                metadata["fetched_files"] = fetched_files
                logger.info(f"Fetched {len(fetched_files)} files from sandbox: {list(fetched_files.keys())}")

            # we should always expect this since we don't have correct answer
            if metadata["run_status"] == "Finished":
                actual_output = metadata["stdout"] if metadata["stdout"] else ""
                # Include stderr if present (warnings, debug output)
                if metadata["stderr"]:
                    actual_output += f"\n[stderr]: {metadata['stderr']}"
                logger.debug(f"actual_output from sandbox fusion: {actual_output}")
                return actual_output, metadata
            else:
                # CRITICAL: Include stderr in error output so coach can see the actual error!
                # This is where FileNotFoundError, KeyError, etc. will appear
                error_output = f"[Execution {metadata['run_status']}]"
                if metadata["stdout"]:
                    error_output += f"\n[stdout]: {metadata['stdout']}"
                if metadata["stderr"]:
                    error_output += f"\n[stderr/error]: {metadata['stderr']}"
                else:
                    error_output += "\n[No error details available]"
                logger.info(f"Sandbox execution failed: {error_output[:500]}")
                return error_output, metadata
        except Exception as e:
            logger.exception(f"Execution failed - {str(e)}")
            return str(e), {"run_status": "Error", "exception": str(e)}

